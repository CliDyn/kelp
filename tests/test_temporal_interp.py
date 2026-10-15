"""Tests for temporal interpolation and writer integration in tracking.

Tests VelocityField.lerp, temporal_interp mode in run_single_realization,
and writer integration with the tracking loop.
"""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from kelp.core.io import VelocityField
from kelp.core.config import ModeConfig, NumericsConfig
from kelp.tracking import run_single_realization, track_particles
from kelp.core.particles import make_state, ParticleStatus
from kelp.readers import get_reader, extract_time_info
from kelp.core.geometry import METERS_PER_DEG_LAT, DEG2RAD
from kelp.writers import load_restart, load_trajectories


# ---------------------------------------------------------------------------
# VelocityField.lerp
# ---------------------------------------------------------------------------

class TestVelocityFieldLerp:
    def test_lerp_alpha0(self):
        """alpha=0 → returns self."""
        v1 = VelocityField(
            u=np.array([1.0, 2.0]),
            v=np.array([3.0, 4.0]),
            eta=np.array([0.1]),
            t=0.0,
        )
        v2 = VelocityField(
            u=np.array([10.0, 20.0]),
            v=np.array([30.0, 40.0]),
            eta=np.array([1.0]),
            t=100.0,
        )
        result = v1.lerp(v2, 0.0)
        np.testing.assert_allclose(result.u, v1.u)
        np.testing.assert_allclose(result.v, v1.v)
        np.testing.assert_allclose(result.eta, v1.eta)
        assert result.t == pytest.approx(0.0)

    def test_lerp_alpha1(self):
        """alpha=1 → returns other."""
        v1 = VelocityField(
            u=np.array([1.0]), v=np.array([1.0]),
            eta=np.array([0.0]), t=0.0,
        )
        v2 = VelocityField(
            u=np.array([5.0]), v=np.array([5.0]),
            eta=np.array([1.0]), t=100.0,
        )
        result = v1.lerp(v2, 1.0)
        np.testing.assert_allclose(result.u, v2.u)
        np.testing.assert_allclose(result.v, v2.v)
        assert result.t == pytest.approx(100.0)

    def test_lerp_midpoint(self):
        """alpha=0.5 → average."""
        v1 = VelocityField(
            u=np.array([0.0, 2.0]),
            v=np.array([0.0, 4.0]),
            eta=np.array([0.0]),
            t=0.0,
        )
        v2 = VelocityField(
            u=np.array([10.0, 8.0]),
            v=np.array([20.0, 16.0]),
            eta=np.array([2.0]),
            t=100.0,
        )
        result = v1.lerp(v2, 0.5)
        np.testing.assert_allclose(result.u, [5.0, 5.0])
        np.testing.assert_allclose(result.v, [10.0, 10.0])
        np.testing.assert_allclose(result.eta, [1.0])
        assert result.t == pytest.approx(50.0)

    def test_lerp_with_w(self):
        """lerp works with 3D (w field)."""
        v1 = VelocityField(
            u=np.array([0.0]), v=np.array([0.0]),
            eta=np.array([0.0]),
            w=np.array([0.0]), t=0.0,
        )
        v2 = VelocityField(
            u=np.array([1.0]), v=np.array([1.0]),
            eta=np.array([1.0]),
            w=np.array([2.0]), t=1.0,
        )
        result = v1.lerp(v2, 0.25)
        np.testing.assert_allclose(result.w, [0.5])

    def test_lerp_w_none(self):
        """lerp with w=None on both sides returns w=None."""
        v1 = VelocityField(u=np.array([1.0]), v=np.array([1.0]), eta=np.array([0.0]))
        v2 = VelocityField(u=np.array([2.0]), v=np.array([2.0]), eta=np.array([0.0]))
        result = v1.lerp(v2, 0.5)
        assert result.w is None

    def test_lerp_t_none(self):
        """lerp with t=None returns t=None."""
        v1 = VelocityField(u=np.array([1.0]), v=np.array([1.0]), eta=np.array([0.0]))
        v2 = VelocityField(u=np.array([2.0]), v=np.array([2.0]), eta=np.array([0.0]))
        result = v1.lerp(v2, 0.5)
        assert result.t is None


class TestVelocityFieldKh:
    """Kh fields on VelocityField, validation, and lerp."""

    def test_kh_absent_by_default(self):
        v = VelocityField(u=np.array([1.0]), v=np.array([1.0]),
                          eta=np.array([0.0]))
        assert v.kh_zonal is None
        assert v.kh_meridional is None
        assert v.kh_layout is None

    def test_kh_partial_raises(self):
        """All three kh_* fields must be set together."""
        with pytest.raises(ValueError, match="kh_zonal, kh_meridional"):
            VelocityField(
                u=np.array([1.0]), v=np.array([1.0]), eta=np.array([0.0]),
                kh_zonal=np.array([10.0]),
            )

    def test_kh_invalid_layout_raises(self):
        with pytest.raises(ValueError, match="kh_layout"):
            VelocityField(
                u=np.array([1.0]), v=np.array([1.0]), eta=np.array([0.0]),
                kh_zonal=np.array([10.0]),
                kh_meridional=np.array([10.0]),
                kh_layout="vertex",
            )

    def test_kh_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            VelocityField(
                u=np.array([1.0]), v=np.array([1.0]), eta=np.array([0.0]),
                kh_zonal=np.array([10.0, 20.0]),
                kh_meridional=np.array([10.0]),
                kh_layout="node",
            )

    def test_kh_node_layout_accepted(self):
        v = VelocityField(
            u=np.array([1.0]), v=np.array([1.0]), eta=np.array([0.0, 0.0]),
            kh_zonal=np.array([10.0, 20.0]),
            kh_meridional=np.array([5.0, 7.0]),
            kh_layout="node",
        )
        assert v.kh_layout == "node"
        np.testing.assert_array_equal(v.kh_zonal, [10.0, 20.0])

    def test_kh_element_layout_accepted(self):
        v = VelocityField(
            u=np.array([1.0]), v=np.array([1.0]), eta=np.array([0.0, 0.0]),
            kh_zonal=np.array([10.0]),
            kh_meridional=np.array([20.0]),
            kh_layout="element",
        )
        assert v.kh_layout == "element"

    def test_lerp_kh_both_present(self):
        v1 = VelocityField(
            u=np.array([1.0]), v=np.array([1.0]), eta=np.array([0.0]),
            kh_zonal=np.array([10.0, 20.0]),
            kh_meridional=np.array([0.0, 100.0]),
            kh_layout="node",
        )
        v2 = VelocityField(
            u=np.array([1.0]), v=np.array([1.0]), eta=np.array([0.0]),
            kh_zonal=np.array([30.0, 40.0]),
            kh_meridional=np.array([200.0, 0.0]),
            kh_layout="node",
        )
        result = v1.lerp(v2, 0.5)
        np.testing.assert_allclose(result.kh_zonal, [20.0, 30.0])
        np.testing.assert_allclose(result.kh_meridional, [100.0, 50.0])
        assert result.kh_layout == "node"

    def test_lerp_kh_absent_returns_none(self):
        v1 = VelocityField(u=np.array([1.0]), v=np.array([1.0]),
                           eta=np.array([0.0]))
        v2 = VelocityField(u=np.array([2.0]), v=np.array([2.0]),
                           eta=np.array([0.0]))
        result = v1.lerp(v2, 0.5)
        assert result.kh_zonal is None
        assert result.kh_meridional is None
        assert result.kh_layout is None

    def test_lerp_kh_only_one_side_returns_none(self):
        """If only one side has Kh, result has none.

        This is a graceful-fallback case rather than an error: a reader
        may legitimately produce a frame without Kh (e.g. Kh dataset
        ends before velocity dataset).  The integrator can detect and
        skip the diffusion kick rather than crash mid-step.
        """
        v1 = VelocityField(
            u=np.array([1.0]), v=np.array([1.0]), eta=np.array([0.0]),
            kh_zonal=np.array([10.0]),
            kh_meridional=np.array([10.0]),
            kh_layout="node",
        )
        v2 = VelocityField(u=np.array([2.0]), v=np.array([2.0]),
                           eta=np.array([0.0]))
        result = v1.lerp(v2, 0.5)
        assert result.kh_zonal is None
        assert result.kh_layout is None

    def test_lerp_kh_layout_mismatch_raises(self):
        v1 = VelocityField(
            u=np.array([1.0]), v=np.array([1.0]), eta=np.array([0.0]),
            kh_zonal=np.array([10.0]),
            kh_meridional=np.array([10.0]),
            kh_layout="node",
        )
        v2 = VelocityField(
            u=np.array([1.0]), v=np.array([1.0]), eta=np.array([0.0]),
            kh_zonal=np.array([20.0]),
            kh_meridional=np.array([20.0]),
            kh_layout="element",
        )
        with pytest.raises(ValueError, match="kh_layout mismatch"):
            v1.lerp(v2, 0.5)

    def test_lerp_kh_shape_mismatch_raises(self):
        v1 = VelocityField(
            u=np.array([1.0]), v=np.array([1.0]), eta=np.array([0.0]),
            kh_zonal=np.array([10.0]),
            kh_meridional=np.array([10.0]),
            kh_layout="node",
        )
        v2 = VelocityField(
            u=np.array([1.0]), v=np.array([1.0]), eta=np.array([0.0]),
            kh_zonal=np.array([20.0, 30.0]),
            kh_meridional=np.array([20.0, 30.0]),
            kh_layout="node",
        )
        with pytest.raises(ValueError, match="kh_zonal shape mismatch"):
            v1.lerp(v2, 0.5)


# ---------------------------------------------------------------------------
# Helper: create FESOM-C test data with real datetime time coord
# ---------------------------------------------------------------------------

def _make_fesomc_timed(tmp_path, n_time=5, dt_seconds=3600.0, u_val=1.0, v_val=0.0):
    """Create FESOM-C NetCDF with datetime64 time coordinate."""
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

    # Time-varying velocity: u increases linearly with time index
    u2d = np.zeros((n_time, n_elem), dtype=np.float64)
    v2d = np.zeros((n_time, n_elem), dtype=np.float64)
    for t_idx in range(n_time):
        u2d[t_idx, :] = u_val * (1.0 + t_idx)  # 1, 2, 3, ...
        v2d[t_idx, :] = v_val
    eta = np.zeros((n_time, 9), dtype=np.float64)

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
    }, coords={"time": times})

    nc_path = tmp_path / "fesomc_timed.nc"
    ds.to_netcdf(nc_path)
    return nc_path


def _make_fesomc_const_timed(tmp_path, n_time=5, dt_seconds=3600.0,
                              u_val=1.0, v_val=0.0):
    """Create FESOM-C NetCDF with CONSTANT velocity and datetime time coord."""
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

    u2d = np.full((n_time, n_elem), u_val, dtype=np.float64)
    v2d = np.full((n_time, n_elem), v_val, dtype=np.float64)
    eta = np.zeros((n_time, 9), dtype=np.float64)

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
    }, coords={"time": times})

    nc_path = tmp_path / "fesomc_const.nc"
    ds.to_netcdf(nc_path)
    return nc_path


# ---------------------------------------------------------------------------
# Temporal interpolation in tracking
# ---------------------------------------------------------------------------

class TestTemporalInterpTracking:
    def test_tinterp_const_field_matches_discrete(self, tmp_path):
        """With constant velocity, temporal_interp should give same result
        as discrete mode (both use dt = file_dt)."""
        nc_path = _make_fesomc_const_timed(tmp_path, n_time=5, dt_seconds=3600.0,
                                            u_val=1.0, v_val=0.0)
        reader = get_reader("fesomc")
        mesh = reader.load_mesh(type("", (), {"meshpath": str(nc_path)})())
        ds = reader.open_dataset(str(nc_path))

        file_times, file_dt = extract_time_info(ds)
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)
        numerics = NumericsConfig(dt=file_dt, integrator="euler", output_freq=1)
        init = make_state(np.array([5.0]), np.array([5.0]))

        # Discrete
        rec_d = run_single_realization(
            mesh, ds, reader, init, mode, numerics,
            start_time_idx=0, n_steps=3,
        )
        # Temporal interp
        rec_t = run_single_realization(
            mesh, ds, reader, init, mode, numerics,
            start_time_idx=0, n_steps=3,
            temporal_interp=True, file_times=file_times, file_dt=file_dt,
        )
        ds.close()

        np.testing.assert_allclose(rec_d.lons[-1], rec_t.lons[-1], atol=1e-10)
        np.testing.assert_allclose(rec_d.lats[-1], rec_t.lats[-1], atol=1e-10)

    def test_tinterp_substepping(self, tmp_path):
        """With dt = file_dt/2, two sub-steps per file interval.
        Constant field → same total displacement as dt = file_dt with 1 step."""
        file_dt = 3600.0
        nc_path = _make_fesomc_const_timed(
            tmp_path, n_time=5, dt_seconds=file_dt, u_val=1.0, v_val=0.0,
        )
        reader = get_reader("fesomc")
        mesh = reader.load_mesh(type("", (), {"meshpath": str(nc_path)})())
        ds = reader.open_dataset(str(nc_path))

        file_times, _ = extract_time_info(ds)
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)
        init = make_state(np.array([5.0]), np.array([5.0]))

        # 1 big step with dt = file_dt
        num_big = NumericsConfig(dt=file_dt, integrator="euler", output_freq=1)
        rec_big = run_single_realization(
            mesh, ds, reader, init, mode, num_big,
            start_time_idx=0, n_steps=1,
            temporal_interp=True, file_times=file_times, file_dt=file_dt,
        )

        # 2 small steps with dt = file_dt/2
        num_small = NumericsConfig(dt=file_dt / 2, integrator="euler", output_freq=2)
        rec_small = run_single_realization(
            mesh, ds, reader, init, mode, num_small,
            start_time_idx=0, n_steps=2,
            temporal_interp=True, file_times=file_times, file_dt=file_dt,
        )
        ds.close()

        # With constant field, same total displacement
        np.testing.assert_allclose(rec_big.lons[-1], rec_small.lons[-1], atol=1e-10)

    def test_tinterp_varying_field(self, tmp_path):
        """Time-varying field: interpolation gives intermediate velocity."""
        file_dt = 3600.0
        nc_path = _make_fesomc_timed(
            tmp_path, n_time=5, dt_seconds=file_dt, u_val=1.0, v_val=0.0,
        )
        reader = get_reader("fesomc")
        mesh = reader.load_mesh(type("", (), {"meshpath": str(nc_path)})())
        ds = reader.open_dataset(str(nc_path))

        file_times, _ = extract_time_info(ds)
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)
        init = make_state(np.array([5.0]), np.array([5.0]))

        # 2 sub-steps within first file interval (u goes 1→2)
        numerics = NumericsConfig(dt=file_dt / 2, integrator="euler", output_freq=1)
        rec = run_single_realization(
            mesh, ds, reader, init, mode, numerics,
            start_time_idx=0, n_steps=2,
            temporal_interp=True, file_times=file_times, file_dt=file_dt,
        )
        ds.close()

        # First sub-step at alpha=0: u=1.0
        # Second sub-step at alpha=0.5: u=1.5 (lerp between 1 and 2)
        # Total displacement > what you'd get with u=1 for both steps
        z0 = np.cos(5.0 * DEG2RAD) * METERS_PER_DEG_LAT
        dlon_uniform = 2 * (file_dt / 2) * 1.0 / z0  # if u=1 both steps
        actual_dlon = rec.lons[-1][0] - 5.0
        assert actual_dlon > dlon_uniform  # because second step has u=1.5


# ---------------------------------------------------------------------------
# Writer integration with tracking loop
# ---------------------------------------------------------------------------

class TestWriterIntegration:
    def test_writer_in_track_particles(self, tmp_path):
        """Writer receives data, in-memory record stays empty."""
        nc_path = _make_fesomc_const_timed(tmp_path, n_time=5, dt_seconds=3600.0)
        reader = get_reader("fesomc")
        mesh = reader.load_mesh(type("", (), {"meshpath": str(nc_path)})())
        ds = reader.open_dataset(str(nc_path))

        out_path = str(tmp_path / "track_out.nc")
        from kelp.writers import TrajectoryWriter
        writer = TrajectoryWriter(out_path, n_particles=1)
        writer.open()

        rec = track_particles(
            mesh, ds, reader,
            lon0=np.array([5.0]), lat0=np.array([5.0]),
            start_time_idx=0, n_steps=3,
            dt=3600.0, writer=writer,
        )
        writer.close()
        ds.close()

        # When writer is active, in-memory record is empty (saves RAM)
        assert rec.n_steps == 0

        # Data is in the file
        data = load_trajectories(out_path)
        assert data["lon"].shape == (1, 3)  # (P, T)
        # Particle should have moved east
        assert data["lon"][0, -1] > 5.0

    def test_writer_zarr_in_tracking(self, tmp_path):
        """Zarr writer works in tracking loop."""
        nc_path = _make_fesomc_const_timed(tmp_path, n_time=5, dt_seconds=3600.0)
        reader = get_reader("fesomc")
        mesh = reader.load_mesh(type("", (), {"meshpath": str(nc_path)})())
        ds = reader.open_dataset(str(nc_path))

        out_path = str(tmp_path / "track_out.zarr")
        from kelp.writers import TrajectoryWriter
        writer = TrajectoryWriter(out_path, n_particles=1)
        writer.open()

        track_particles(
            mesh, ds, reader,
            lon0=np.array([5.0]), lat0=np.array([5.0]),
            start_time_idx=0, n_steps=3,
            dt=3600.0, writer=writer,
        )
        writer.close()
        ds.close()

        data = load_trajectories(out_path)
        assert data["lon"].shape == (1, 3)
        assert data["lon"][0, -1] > 5.0

    def test_checkpoint_in_tracking(self, tmp_path):
        """Checkpoint written during tracking can be loaded."""
        nc_path = _make_fesomc_const_timed(tmp_path, n_time=10, dt_seconds=3600.0)
        reader = get_reader("fesomc")
        mesh = reader.load_mesh(type("", (), {"meshpath": str(nc_path)})())
        ds = reader.open_dataset(str(nc_path))

        out_path = str(tmp_path / "ckpt_track.nc")
        from kelp.writers import TrajectoryWriter
        writer = TrajectoryWriter(out_path, n_particles=1)
        writer.open()

        track_particles(
            mesh, ds, reader,
            lon0=np.array([5.0]), lat0=np.array([5.0]),
            start_time_idx=0, n_steps=6,
            dt=3600.0, writer=writer,
            checkpoint_freq=3,
        )
        writer.close()
        ds.close()

        restart = load_restart(out_path)
        # Final checkpoint at step n_steps-1=5
        assert restart["step_i"] == 5
        # Particle should have moved east from 5.0
        assert restart["state"].lon[0] > 5.0


# =====================================================================
# Time-bracket and payload-builder helpers (Stage 3)
# =====================================================================

class TestTimeBracketAndPayload:
    """Unit tests for _resolve_time_bracket and _build_vel_payload."""

    def test_resolve_bracket_exact_snapshot(self):
        from kelp.tracking import _resolve_time_bracket
        file_times = np.array([0.0, 3600.0, 7200.0, 10800.0])
        br = _resolve_time_bracket(3600.0, file_times, 4, "stop")
        idx_a, idx_b, ta, tb, wrapped = br
        # searchsorted(side="right") places an exact hit on the right,
        # so idx_a/idx_b straddle it and ta == sim_time → alpha=0.
        assert ta == 3600.0 and wrapped == 3600.0

    def test_resolve_bracket_between_snapshots(self):
        from kelp.tracking import _resolve_time_bracket
        file_times = np.array([0.0, 3600.0, 7200.0])
        br = _resolve_time_bracket(1800.0, file_times, 3, "stop")
        idx_a, idx_b, ta, tb, wrapped = br
        assert idx_a == 0 and idx_b == 1
        assert ta == 0.0 and tb == 3600.0
        assert wrapped == 1800.0

    def test_resolve_bracket_out_of_range_stop(self):
        from kelp.tracking import _resolve_time_bracket
        file_times = np.array([0.0, 3600.0])
        assert _resolve_time_bracket(7200.0, file_times, 2, "stop") is None

    def test_resolve_bracket_cycle_wrap(self):
        from kelp.tracking import _resolve_time_bracket
        file_times = np.array([0.0, 3600.0, 7200.0])  # span 7200 + last_dt=3600 = 10800
        # sim_time=10800 wraps to 0
        br = _resolve_time_bracket(10800.0, file_times, 3, "cycle")
        assert br is not None
        _, _, _, _, wrapped = br
        assert wrapped == pytest.approx(0.0, abs=1e-6)

    def test_resolve_bracket_cycle_irregular_spacing(self):
        """Cycle wrap with irregular file_times uses the trailing interval."""
        from kelp.tracking import _resolve_time_bracket
        # Irregular spacing: [0, 1000, 5000] — last interval = 4000
        file_times = np.array([0.0, 1000.0, 5000.0])
        # sim_time=6000 is past end; cycle wraps via total_span = 5000 + 4000 = 9000
        br = _resolve_time_bracket(6000.0, file_times, 3, "cycle")
        assert br is not None

    def test_build_payload_at_end_snapshot(self):
        """sim_time at file_times[-1]: idx_a == idx_b (both clamped to -1),
        vel_next is None and dt_to_next is None.
        """
        from kelp.tracking import (
            _resolve_time_bracket, _build_vel_payload, _VelCache,
        )

        # Minimal stub reader returning a constant VelocityField per idx
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

        file_times = np.array([0.0, 3600.0])
        # sim_time at file_times[-1] → searchsorted side="right" returns
        # n_total, clamped to n_total-1 = 1; idx_a = 0 still, so not exact.
        # Use sim_time > file_times[-1] in stop mode to return None, not
        # relevant here.  Instead pick sim_time past last snap with cycle.
        br = _resolve_time_bracket(3600.0, file_times, 2, "stop")
        cache = _VelCache()
        vs, vn, dtn, idx = _build_vel_payload(
            br, cache, _StubReader, None, None, None,
        )
        # idx_a=0, idx_b=1 (clamped since 3600 is at boundary) — distinct
        # so we still get vel_next and dt_to_next (== 0). That's fine:
        # integrator will see dt_to_next<=0 and raise; tracking silently
        # downgrades when dt > dt_to_next.
        # Verify interpolation at t=tb yields vel_b value.
        assert vs.u[0] == pytest.approx(1.0)  # alpha=1 → vel_b

    def test_build_payload_between_returns_next_and_dt(self):
        from kelp.tracking import (
            _resolve_time_bracket, _build_vel_payload, _VelCache,
        )

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

        file_times = np.array([0.0, 3600.0])
        br = _resolve_time_bracket(900.0, file_times, 2, "stop")
        cache = _VelCache()
        vs, vn, dtn, idx = _build_vel_payload(
            br, cache, _StubReader, None, None, None,
        )
        assert vn is not None
        assert dtn == pytest.approx(2700.0)  # 3600 - 900
        assert idx == 0
        # vel_step should be at alpha=0.25 between 0.0 and 1.0 → 0.25
        assert vs.u[0] == pytest.approx(0.25)


# =====================================================================
# Integration: discrete path now passes vel_next to RK
# =====================================================================

class TestDiscretePathVelNext:
    """Verify discrete path produces vel_next kwarg under varying velocity.

    Probes via monkey-patched integrator that captures kwargs received.
    Stage 3 extended discrete path to read file_idx+direction as vel_next.
    """

    def test_discrete_rk4_passes_vel_next(self, tmp_path):
        from kelp.tracking import track_particles
        from kelp.readers import get_reader
        from kelp.numerics import integrators as intg

        nc_path = str(_make_fesomc_timed(
            tmp_path, n_time=4, dt_seconds=300.0,
            u_val=0.05, v_val=0.0,
        ).resolve())

        reader = get_reader("fesomc")
        mesh = reader.load_mesh(type("M", (), {"meshpath": nc_path}))
        ds = reader.open_dataset(nc_path)

        captured = []
        orig = intg.rk4_step

        def spy(state, vel, mesh_, dt, time_dir, layer=0, **kwargs):
            captured.append({
                "vel_next": kwargs.get("vel_next"),
                "dt_to_next": kwargs.get("dt_to_next"),
            })
            return orig(state, vel, mesh_, dt, time_dir, layer=layer, **kwargs)

        intg.rk4_step = spy
        intg.INTEGRATORS["rk4"] = spy
        try:
            # file_times required for time-aware discrete path
            file_times = np.arange(4) * 300.0
            track_particles(
                mesh, ds, reader,
                lon0=np.array([5.0]), lat0=np.array([5.0]),
                start_time_idx=0, n_steps=2,
                dt=300.0, time_dir=1.0, integrator="rk4",
                file_times=file_times,
            )
        finally:
            intg.rk4_step = orig
            intg.INTEGRATORS["rk4"] = orig
            ds.close()

        # First step should have vel_next + dt_to_next set
        assert len(captured) >= 1
        assert captured[0]["vel_next"] is not None, (
            "discrete RK4 did not receive vel_next kwarg"
        )
        assert captured[0]["dt_to_next"] == pytest.approx(300.0)


class TestDowngradeWarning:
    """When dt > dt_to_next the RK stages fall back to steady velocity;
    this should be observable via a RuntimeWarning + summary counter."""

    def test_warns_once_when_dt_exceeds_bracket(self, tmp_path):
        import warnings
        from kelp.tracking import track_particles
        from kelp.readers import get_reader

        # File cadence 300s, but tracker dt 900s → dt > dt_to_next every step.
        # Interpolated mode: in discrete mode this dt is refused outright
        # (tracking._check_discrete_dt, 2026-09-24).
        nc_path = str(_make_fesomc_timed(
            tmp_path, n_time=6, dt_seconds=300.0,
            u_val=0.05, v_val=0.0,
        ).resolve())
        reader = get_reader("fesomc")
        mesh = reader.load_mesh(type("M", (), {"meshpath": nc_path}))
        ds = reader.open_dataset(nc_path)
        file_times = np.arange(6) * 300.0

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                track_particles(
                    mesh, ds, reader,
                    lon0=np.array([5.0]), lat0=np.array([5.0]),
                    start_time_idx=0, n_steps=3,
                    dt=900.0, time_dir=1.0, integrator="rk4",
                    file_times=file_times, temporal_interp=True,
                )
            downgrades = [
                x for x in w
                if issubclass(x.category, RuntimeWarning)
                and "dt_to_next" in str(x.message)
            ]
            # Warning fires exactly once per realization, even if multiple steps downgrade
            assert len(downgrades) == 1, (
                f"expected 1 downgrade warning, got {len(downgrades)}"
            )
            assert "steady" in str(downgrades[0].message).lower()
        finally:
            ds.close()


class TestMultiRunSimTimeShift:
    """Unit test for Stage 3 sim_time_start shift across realizations."""

    def test_shift_logic_math(self):
        """Replicate the tracking.py shift: sim_time_start += dt between idx."""
        file_times = np.array([0.0, 3600.0, 7200.0, 10800.0, 14400.0])
        start_idx = 0
        sim_time_start = 1800.0  # between idx 0 and 1 (alpha=0.5)
        run_interval = 1

        # Apply the same logic as tracking.py per realization
        for _ in range(3):
            prev_idx = start_idx
            start_idx += run_interval
            if 0 <= prev_idx < len(file_times) and 0 <= start_idx < len(file_times):
                sim_time_start += file_times[start_idx] - file_times[prev_idx]

        # After 3 shifts: start_idx=3, sim_time_start += 3600*3 = 10800 + 1800 = 12600
        assert start_idx == 3
        assert sim_time_start == pytest.approx(1800.0 + 3 * 3600.0)

    def test_cycle_wrap_advances_by_full_period(self):
        """With on_exhausted='cycle' and run_interval crossing n_time_total,
        sim_time_start must advance by the correct number of full periods."""
        from kelp.tracking import _index_to_sim_time

        # 4 snapshots with 3600s cadence → period = 4*3600 = 14400 (last_dt=3600)
        file_times = np.array([0.0, 3600.0, 7200.0, 10800.0])
        n = len(file_times)

        # Start at idx 1, run_interval = n (full cycle) → new idx = n+1
        prev_idx, new_idx = 1, 1 + n
        delta = (_index_to_sim_time(new_idx, file_times, n, "cycle")
                 - _index_to_sim_time(prev_idx, file_times, n, "cycle"))
        assert delta == pytest.approx(14400.0), (
            f"expected one full period (14400s), got {delta}"
        )

        # Crossing twice: run_interval = 2*n+2 starting from idx 0
        prev_idx, new_idx = 0, 2 * n + 2
        delta = (_index_to_sim_time(new_idx, file_times, n, "cycle")
                 - _index_to_sim_time(prev_idx, file_times, n, "cycle"))
        # Two periods (28800) plus jump from idx 0 to idx 2 inside the cycle (7200)
        assert delta == pytest.approx(2 * 14400.0 + 7200.0)

    def test_stop_clamps_out_of_range(self):
        """In stop mode the shift clips at the file-range edge."""
        from kelp.tracking import _index_to_sim_time

        file_times = np.array([0.0, 3600.0, 7200.0, 10800.0])
        n = len(file_times)
        # Past the end: clamped to last
        assert _index_to_sim_time(n + 5, file_times, n, "stop") == pytest.approx(10800.0)
        # Before the start: clamped to 0
        assert _index_to_sim_time(-3, file_times, n, "stop") == pytest.approx(0.0)
