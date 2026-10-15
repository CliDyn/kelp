"""Backward tracking with temporal interpolation: RK-stage time direction.

Bug (review finding, 2026-06-05): in temporal-interp mode the
tracking loop always hands the integrator the FUTURE snapshot
(``vel_next = vel_b``, ``dt_to_next = tb - sim_time``) regardless of the
integration direction.  RK stages then blend ``vel → vel_next`` at
``alpha = c·dt/dt_to_next``, i.e. they sample the field at ``t + c·dt`` —
but a backward step moves to ``t - dt``, so the stages must sample
``t - c·dt`` (toward the PREVIOUS snapshot).  The discrete path already
passes the direction-correct neighbour (``next_idx = file_idx + direction``);
this suite pins the same contract for the temporal-interp path:

- ``_resolve_time_bracket(..., direction=-1)``: a sim_time sitting exactly
  ON a snapshot brackets DOWNWARD (``[t_{k-1}, t_k]``), including at the
  last snapshot in stop mode (where the forward bracket collapses) and at
  the wrap point in cycle mode.
- ``_build_vel_payload(..., direction=-1)``: ``vel_next`` is the snapshot
  at ``ta`` (the next one along backward motion) and
  ``dt_to_next = sim_time - ta > 0``.
- End-to-end: on a spatially-uniform field linear in time, lerp between
  snapshots reconstructs u(t) exactly, and RK2/RK4 quadrature of a linear
  integrand is exact — so backward trajectories must match the analytic
  integral to float precision.  With the bug they are off by
  ~a·T·dt (metres-to-km scale).
"""
import numpy as np
import pytest
import xarray as xr

from kelp.core.config import ModeConfig, NumericsConfig
from kelp.core.geometry import DEG2RAD, METERS_PER_DEG_LAT
from kelp.core.particles import make_state
from kelp.readers import get_reader, extract_time_info
from kelp.tracking import run_single_realization


# ---------------------------------------------------------------------------
# Fixture: FESOM-C NetCDF, spatially-uniform u linear in time (v = 0)
# (self-contained copy of the test_temporal_interp.py builder — test files
# in this repo do not import from each other)
# ---------------------------------------------------------------------------

def _make_fesomc_linear_in_time(tmp_path, n_time=5, dt_seconds=3600.0,
                                u_val=1.0):
    """u(t) = u_val * (1 + t/dt_seconds) everywhere; v = 0.

    Linear in time and uniform in space: temporal lerp between snapshots
    reconstructs u(t) EXACTLY, and spatial interpolation returns the same
    value at any position — so only the times the RK stages sample matter.
    """
    import pandas as pd

    xs = np.array([4.0, 5.0, 6.0])
    ys = np.array([4.0, 5.0, 6.0])
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel().astype(np.float64)
    lat = gy.ravel().astype(np.float64)
    depth = np.full(9, 10.0, dtype=np.float64)

    elems = []
    for j in range(2):
        for i in range(2):
            n0 = j * 3 + i
            elems.append([n0 + 1, n0 + 2, n0 + 4, n0 + 1])
            elems.append([n0 + 2, n0 + 5, n0 + 4, n0 + 2])
    nv = np.array(elems, dtype=np.int32)
    n_elem = nv.shape[0]

    elem_nodes_0 = nv[:, :3] - 1
    lon_c = np.array([lon[e].mean() for e in elem_nodes_0], dtype=np.float64)
    lat_c = np.array([lat[e].mean() for e in elem_nodes_0], dtype=np.float64)
    sigma_lev = np.array([0.0, 0.5, 1.0], dtype=np.float64)

    u2d = np.zeros((n_time, n_elem), dtype=np.float64)
    v2d = np.zeros((n_time, n_elem), dtype=np.float64)
    for t_idx in range(n_time):
        u2d[t_idx, :] = u_val * (1.0 + t_idx)  # u(t_k) = u_val*(1+k)
    eta = np.zeros((n_time, 9), dtype=np.float64)
    # Node scalar linear in time too: temp(t_k) = 1 + k, uniform in space
    # (read only when sample_vars asks for it — inert for the other tests).
    temp = np.zeros((n_time, 9), dtype=np.float64)
    for t_idx in range(n_time):
        temp[t_idx, :] = 1.0 + t_idx

    times = pd.date_range("2020-01-01", periods=n_time,
                          freq=f"{int(dt_seconds)}s")

    ds = xr.Dataset({
        "lon": (["node"], lon),
        "lat": (["node"], lat),
        "depth": (["node"], depth),
        "nv": (["nele", "four"], nv),
        "lon_elem": (["nele"], lon_c),
        "lat_elem": (["nele"], lat_c),
        "sigma_lev": (["sigma"], sigma_lev),
        "u2d": (["time", "nele"], u2d),
        "v2d": (["time", "nele"], v2d),
        "eta": (["time", "node"], eta),
        "temp": (["time", "node"], temp),
    }, coords={"time": times})

    nc_path = tmp_path / "fesomc_lin_t.nc"
    ds.to_netcdf(nc_path)
    return nc_path


def _make_fesomc_irregular(tmp_path, offsets_s=(0.0, 1000.0, 5000.0),
                           u_vals=(1.0, 2.0, 3.0)):
    """Irregularly spaced snapshots (for cycle-mode seam tests).

    Same 3x3-node quad/tri mesh; u uniform in space, one value per
    snapshot; v = 0.
    """
    import pandas as pd

    xs = np.array([4.0, 5.0, 6.0])
    ys = np.array([4.0, 5.0, 6.0])
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel().astype(np.float64)
    lat = gy.ravel().astype(np.float64)
    depth = np.full(9, 10.0, dtype=np.float64)

    elems = []
    for j in range(2):
        for i in range(2):
            n0 = j * 3 + i
            elems.append([n0 + 1, n0 + 2, n0 + 4, n0 + 1])
            elems.append([n0 + 2, n0 + 5, n0 + 4, n0 + 2])
    nv = np.array(elems, dtype=np.int32)
    n_elem = nv.shape[0]

    elem_nodes_0 = nv[:, :3] - 1
    lon_c = np.array([lon[e].mean() for e in elem_nodes_0], dtype=np.float64)
    lat_c = np.array([lat[e].mean() for e in elem_nodes_0], dtype=np.float64)
    sigma_lev = np.array([0.0, 0.5, 1.0], dtype=np.float64)

    n_time = len(offsets_s)
    u2d = np.zeros((n_time, n_elem), dtype=np.float64)
    v2d = np.zeros((n_time, n_elem), dtype=np.float64)
    for t_idx, uv in enumerate(u_vals):
        u2d[t_idx, :] = uv
    eta = np.zeros((n_time, 9), dtype=np.float64)

    times = (pd.Timestamp("2020-01-01")
             + pd.to_timedelta(list(offsets_s), unit="s"))

    ds = xr.Dataset({
        "lon": (["node"], lon),
        "lat": (["node"], lat),
        "depth": (["node"], depth),
        "nv": (["nele", "four"], nv),
        "lon_elem": (["nele"], lon_c),
        "lat_elem": (["nele"], lat_c),
        "sigma_lev": (["sigma"], sigma_lev),
        "u2d": (["time", "nele"], u2d),
        "v2d": (["time", "nele"], v2d),
        "eta": (["time", "node"], eta),
    }, coords={"time": times})

    nc_path = tmp_path / "fesomc_irregular.nc"
    ds.to_netcdf(nc_path)
    return nc_path


def _u_antiderivative(t, u_val=1.0, dt_seconds=3600.0):
    """∫ u dτ for u(τ) = u_val*(1 + τ/dt_seconds)."""
    return u_val * (t + t * t / (2.0 * dt_seconds))


# ---------------------------------------------------------------------------
# Unit: direction-aware bracket resolution
# ---------------------------------------------------------------------------

class TestBackwardBracket:
    def test_exact_snapshot_brackets_downward(self):
        from kelp.tracking import _resolve_time_bracket
        file_times = np.array([0.0, 3600.0, 7200.0, 10800.0])
        br = _resolve_time_bracket(7200.0, file_times, 4, "stop",
                                   direction=-1)
        idx_a, idx_b, ta, tb, wrapped = br
        # Backward from t=7200 moves INTO [3600, 7200] — that is the bracket.
        assert (idx_a, idx_b) == (1, 2)
        assert ta == 3600.0 and tb == 7200.0
        assert wrapped == 7200.0  # alpha = 1: vel_step is the snapshot itself

    def test_last_snapshot_stop_does_not_collapse(self):
        # Forward at file_times[-1] collapses (nothing ahead) → steady.
        # Backward has a perfectly good bracket behind it.
        from kelp.tracking import _resolve_time_bracket
        file_times = np.array([0.0, 3600.0])
        br = _resolve_time_bracket(3600.0, file_times, 2, "stop",
                                   direction=-1)
        idx_a, idx_b, ta, tb, wrapped = br
        assert (idx_a, idx_b) == (0, 1)
        assert ta == 0.0 and tb == 3600.0
        assert wrapped == 3600.0

    def test_first_snapshot_stop_collapses(self):
        # Backward at file_times[0]: nothing before it → degenerate bracket
        # (steady fallback), mirroring forward behaviour at file_times[-1].
        from kelp.tracking import _resolve_time_bracket
        file_times = np.array([0.0, 3600.0])
        br = _resolve_time_bracket(0.0, file_times, 2, "stop", direction=-1)
        idx_a, idx_b, ta, tb, wrapped = br
        assert idx_a == idx_b

    def test_mid_bracket_same_as_forward(self):
        # Strictly inside a bracket the indices/times do not depend on
        # direction (only the payload's vel_next/dt_to_next do).
        from kelp.tracking import _resolve_time_bracket
        file_times = np.array([0.0, 3600.0, 7200.0])
        br_f = _resolve_time_bracket(1800.0, file_times, 3, "stop")
        br_b = _resolve_time_bracket(1800.0, file_times, 3, "stop",
                                     direction=-1)
        assert br_f == br_b

    def test_cycle_wrap_at_period_start(self):
        # Backward at the wrap point (sim_time on file_times[0] in cycle
        # mode): the bracket is the trailing interval [t_last, t_last+dt],
        # with the position at its TOP (alpha = 1 → vel_step = snapshot 0).
        from kelp.tracking import _resolve_time_bracket
        file_times = np.array([0.0, 3600.0, 7200.0])
        br = _resolve_time_bracket(0.0, file_times, 3, "cycle", direction=-1)
        idx_a, idx_b, ta, tb, wrapped = br
        assert (idx_a, idx_b) == (2, 0)
        assert ta == 7200.0
        assert tb == pytest.approx(10800.0)   # ta + trailing interval
        assert wrapped == pytest.approx(10800.0)

    def test_forward_default_unchanged(self):
        # direction=+1 (and the no-kwarg default) keep today's behaviour:
        # an exact snapshot brackets UPWARD (alpha = 0).
        from kelp.tracking import _resolve_time_bracket
        file_times = np.array([0.0, 3600.0, 7200.0, 10800.0])
        br_default = _resolve_time_bracket(7200.0, file_times, 4, "stop")
        br_plus = _resolve_time_bracket(7200.0, file_times, 4, "stop",
                                        direction=1)
        assert br_default == br_plus
        idx_a, idx_b, ta, tb, wrapped = br_plus
        assert ta == 7200.0 and wrapped == 7200.0


# ---------------------------------------------------------------------------
# Unit: direction-aware payload (vel_next = previous snapshot)
# ---------------------------------------------------------------------------

class _StubReader:
    @staticmethod
    def read_velocities(ds, idx, mesh, mode, diffusion=None):
        from kelp.numerics.interpolate import VelocityField
        return VelocityField(
            u=np.full(1, float(idx)),
            v=np.zeros(1),
            eta=np.zeros(1),
            w=None,
        )


class TestBackwardPayload:
    def test_mid_bracket_vel_next_is_previous_snapshot(self):
        from kelp.tracking import (
            _resolve_time_bracket, _build_vel_payload, _VelCache,
        )
        file_times = np.array([0.0, 3600.0])
        br = _resolve_time_bracket(900.0, file_times, 2, "stop",
                                   direction=-1)
        cache = _VelCache()
        vs, vn, dtn, idx = _build_vel_payload(
            br, cache, _StubReader, None, None, None, direction=-1,
        )
        # vel_step: field at sim_time (direction-free) = lerp at alpha 0.25
        assert vs.u[0] == pytest.approx(0.25)
        # vel_next: the snapshot BEHIND (idx 0), 900 s away
        assert vn is not None
        assert vn.u[0] == pytest.approx(0.0)
        assert dtn == pytest.approx(900.0)

    def test_exact_snapshot_payload(self):
        from kelp.tracking import (
            _resolve_time_bracket, _build_vel_payload, _VelCache,
        )
        file_times = np.array([0.0, 3600.0, 7200.0])
        br = _resolve_time_bracket(3600.0, file_times, 3, "stop",
                                   direction=-1)
        cache = _VelCache()
        vs, vn, dtn, idx = _build_vel_payload(
            br, cache, _StubReader, None, None, None, direction=-1,
        )
        # Field at t=3600 is snapshot 1; next along backward motion is
        # snapshot 0, a full file interval away.
        assert vs.u[0] == pytest.approx(1.0)
        assert vn is not None
        assert vn.u[0] == pytest.approx(0.0)
        assert dtn == pytest.approx(3600.0)

    def test_first_snapshot_stop_steady(self):
        from kelp.tracking import (
            _resolve_time_bracket, _build_vel_payload, _VelCache,
        )
        file_times = np.array([0.0, 3600.0])
        br = _resolve_time_bracket(0.0, file_times, 2, "stop", direction=-1)
        cache = _VelCache()
        vs, vn, dtn, idx = _build_vel_payload(
            br, cache, _StubReader, None, None, None, direction=-1,
        )
        assert vs.u[0] == pytest.approx(0.0)
        assert vn is None and dtn is None

    def test_forward_default_unchanged(self):
        from kelp.tracking import (
            _resolve_time_bracket, _build_vel_payload, _VelCache,
        )
        file_times = np.array([0.0, 3600.0])
        br = _resolve_time_bracket(900.0, file_times, 2, "stop")
        cache = _VelCache()
        vs, vn, dtn, idx = _build_vel_payload(
            br, cache, _StubReader, None, None, None,
        )
        assert vs.u[0] == pytest.approx(0.25)
        assert vn.u[0] == pytest.approx(1.0)   # forward: future snapshot
        assert dtn == pytest.approx(2700.0)


# ---------------------------------------------------------------------------
# End-to-end: backward RK on a field linear in time must be exact
# ---------------------------------------------------------------------------

class TestBackwardTinterpTracking:
    def _open(self, tmp_path):
        nc_path = _make_fesomc_linear_in_time(tmp_path, n_time=5,
                                              dt_seconds=3600.0, u_val=1.0)
        reader = get_reader("fesomc")
        mesh = reader.load_mesh(type("", (), {"meshpath": str(nc_path)})())
        ds = reader.open_dataset(str(nc_path))
        file_times, file_dt = extract_time_info(ds)
        return reader, mesh, ds, file_times, file_dt

    def test_backward_rk2_matches_analytic(self, tmp_path):
        """Backward RK2 from the LAST snapshot, dt = file_dt/6.

        u(t) is linear in time and uniform in space, so the temporally
        lerped field is exact and RK2 (midpoint quadrature) integrates it
        exactly: lon(t) must match the analytic integral to fp precision.
        Exercises both bug surfaces: the collapsed bracket on the exact
        last snapshot (first step) and future-pointing stages mid-bracket.
        """
        reader, mesh, ds, file_times, file_dt = self._open(tmp_path)
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=-1.0)
        dt = file_dt / 6.0  # 600 s
        n_steps = 12        # two full file intervals: 14400 s → 7200 s
        numerics = NumericsConfig(dt=dt, integrator="rk2", output_freq=1)
        init = make_state(np.array([5.0]), np.array([5.0]))

        rec = run_single_realization(
            mesh, ds, reader, init, mode, numerics,
            start_time_idx=4, n_steps=n_steps,
            temporal_interp=True, file_times=file_times, file_dt=file_dt,
        )
        ds.close()

        t0 = file_times[4]              # 14400 s
        t1 = t0 - n_steps * dt          # 7200 s
        dist_m = _u_antiderivative(t0) - _u_antiderivative(t1)  # 28800 m
        z0 = np.cos(5.0 * DEG2RAD) * METERS_PER_DEG_LAT
        lon_expected = 5.0 - dist_m / z0

        assert rec.lats[-1][0] == pytest.approx(5.0, abs=1e-9)
        assert rec.lons[-1][0] == pytest.approx(lon_expected, abs=1e-6)

    def test_round_trip_rk4_returns_to_start(self, tmp_path):
        """Forward 12 steps then backward 12 steps lands on the start.

        Both legs are exact on a time-linear field (RK4 quadrature), so the
        round trip closes to fp precision.  The backward leg starts exactly
        ON a snapshot (idx 2), exercising the on-snapshot bracket shift.
        """
        reader, mesh, ds, file_times, file_dt = self._open(tmp_path)
        dt = file_dt / 6.0
        n_steps = 12
        numerics = NumericsConfig(dt=dt, integrator="rk4", output_freq=1)
        init = make_state(np.array([5.0]), np.array([5.0]))

        mode_f = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)
        rec_f = run_single_realization(
            mesh, ds, reader, init, mode_f, numerics,
            start_time_idx=0, n_steps=n_steps,
            temporal_interp=True, file_times=file_times, file_dt=file_dt,
        )
        lon_end = float(rec_f.lons[-1][0])
        lat_end = float(rec_f.lats[-1][0])

        mode_b = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=-1.0)
        init_b = make_state(np.array([lon_end]), np.array([lat_end]))
        rec_b = run_single_realization(
            mesh, ds, reader, init_b, mode_b, numerics,
            start_time_idx=2, n_steps=n_steps,   # file_times[2] = 7200 s
            temporal_interp=True, file_times=file_times, file_dt=file_dt,
        )
        ds.close()

        assert rec_b.lons[-1][0] == pytest.approx(5.0, abs=1e-6)
        assert rec_b.lats[-1][0] == pytest.approx(5.0, abs=1e-9)

    def test_cycle_round_trip_rk4_irregular_spacing(self, tmp_path):
        """Cycle mode, IRREGULAR snapshot spacing, crossing the wrap seam.

        For a spatially-uniform field the RK4 displacement depends only on
        the stage TIMES (c = 0,½,½,1 with symmetric weights), so a forward
        leg and a backward leg that sample the same reconstructed u(t) at
        the same times cancel exactly — for ANY time profile, including the
        sawtooth that cycling a 3-snapshot file produces.  Closure therefore
        pins the stage-time direction through the cycle seam and the
        trailing-interval bracket; with future-pointing backward stages the
        legs sample different times and the trip does not close (review
        follow-up: the reviewers asked for a cycle e2e).
        """
        nc_path = _make_fesomc_irregular(
            tmp_path, offsets_s=(0.0, 1000.0, 5000.0), u_vals=(1.0, 2.0, 3.0),
        )
        reader = get_reader("fesomc")
        mesh = reader.load_mesh(type("", (), {"meshpath": str(nc_path)})())
        ds = reader.open_dataset(str(nc_path))
        file_times, file_dt = extract_time_info(ds)

        dt = 500.0
        n_steps = 24            # 12000 s — past the 9000 s period
        numerics = NumericsConfig(dt=dt, integrator="rk4", output_freq=1)
        init = make_state(np.array([5.0]), np.array([5.0]))

        mode_f = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)
        rec_f = run_single_realization(
            mesh, ds, reader, init, mode_f, numerics,
            start_time_idx=0, n_steps=n_steps, on_exhausted="cycle",
            temporal_interp=True, file_times=file_times, file_dt=file_dt,
        )
        lon_end = float(rec_f.lons[-1][0])
        # sanity: the forward leg moved and crossed the seam
        assert lon_end != pytest.approx(5.0, abs=1e-4)

        mode_b = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=-1.0)
        init_b = make_state(np.array([lon_end]),
                            np.array([float(rec_f.lats[-1][0])]))
        rec_b = run_single_realization(
            mesh, ds, reader, init_b, mode_b, numerics,
            start_time_idx=0, n_steps=n_steps, on_exhausted="cycle",
            temporal_interp=True, file_times=file_times, file_dt=file_dt,
            sim_time_start=float(n_steps * dt),   # 12000 s, mid 2nd period
        )
        ds.close()

        assert rec_b.lons[-1][0] == pytest.approx(5.0, abs=1e-6)
        assert rec_b.lats[-1][0] == pytest.approx(5.0, abs=1e-9)

    def test_backward_sampling_records_field_at_record_times(self, tmp_path):
        """Field sampling along a BACKWARD temporal-interp run.

        Coverage pin, not a bug reproducer: the record-time re-lerp samples
        the field at one instant (direction-free value), so the old code
        already produced correct values here — this pins the backward
        record path against future regressions (agent review, finding 4).
        temp(t) = 1 + t/3600 uniform in space; records at post-step
        sim_times 12600, 10800, 9000, 7200 s must read 4.5, 4.0, 3.5, 3.0.
        """
        reader, mesh, ds, file_times, file_dt = self._open(tmp_path)
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=-1.0)
        dt = file_dt / 2.0      # 1800 s
        numerics = NumericsConfig(dt=dt, integrator="rk2", output_freq=1)
        init = make_state(np.array([5.0]), np.array([5.0]))

        rec = run_single_realization(
            mesh, ds, reader, init, mode, numerics,
            start_time_idx=4, n_steps=4,
            temporal_interp=True, file_times=file_times, file_dt=file_dt,
            sample_vars=["temp"],
        )
        ds.close()

        got = np.array([snap[0] for snap in rec.sampled["temp"]])
        expected = np.array([4.5, 4.0, 3.5, 3.0])
        np.testing.assert_allclose(got, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# Review follow-ups: payload guard band + legacy wrapper direction
# ---------------------------------------------------------------------------

class TestPayloadGuardBand:
    def test_unshifted_on_snapshot_bracket_degrades_to_steady(self):
        """A backward payload on an UNSHIFTED bracket with the position
        inside the 1e-6 s snap band must degrade to steady, not hand the
        integrator a sub-microsecond dt_to_next (agent review, finding 1).
        The run loop never produces this (it resolves brackets with the
        same direction), so the bracket tuple is constructed by hand.
        """
        from kelp.tracking import _build_vel_payload, _VelCache
        br = (0, 1, 0.0, 3600.0, 5e-7)   # pos 0.5 µs above the snapshot
        cache = _VelCache()
        vs, vn, dtn, idx = _build_vel_payload(
            br, cache, _StubReader, None, None, None, direction=-1,
        )
        # vel_step is the honest lerp at alpha = 5e-7/3600 ≈ 1.4e-10 —
        # indistinguishable from the snapshot field at fp-noise level.
        assert vs.u[0] == pytest.approx(0.0, abs=1e-9)
        assert vn is None and dtn is None


class TestLegacyWrapperDirection:
    def test_wrapper_forwards_direction(self):
        """_interp_velocity_at_time accepts ``direction`` and must resolve
        the bracket with it (review finding: the ignored parameter
        was an API trap).  Observable: at an exact snapshot the backward
        bracket shifts down, so file_idx is the LOWER index; the field
        value is the snapshot's either way.
        """
        from kelp.tracking import _interp_velocity_at_time, _VelCache
        file_times = np.array([0.0, 3600.0, 7200.0])
        cache = _VelCache()
        vel, file_idx = _interp_velocity_at_time(
            3600.0, file_times, 3, "stop", -1, cache,
            _StubReader, None, None, None,
        )
        assert vel.u[0] == pytest.approx(1.0)  # snapshot-1 field
        assert file_idx == 0                   # bracket [t0, t1]
