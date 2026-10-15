"""Periodic boundary conditions for idealized regional domains
(channels, ridges) — `numerics.periodic_lon` / `numerics.periodic_lat`.

Contract (ghost-band recipe): the MESH must carry a ghost band beyond
the periodic window wide enough for one step's displacement; particles
track INTO the ghost band and are folded back into the window after
each step (post-step, before beaching/record).  In-window particles are
bitwise untouched; wrapped particles get a fresh nearest-node anchor
and (when edge-walk tracking is on) a reset carried element.  Feature
off = byte-identical.
"""
import numpy as np
import pytest
import xarray as xr
from types import SimpleNamespace

from kelp.core.config import (AppConfig, DataConfig, MeshConfig,
                                  ModeConfig, NumericsConfig, TimeConfig)
from kelp.core.geometry import DEG2RAD, METERS_PER_DEG_LAT
from kelp.core.particles import ParticleStatus, make_state
from kelp.readers import get_reader
from kelp.tracking import (run_single_realization, _apply_periodic_wrap,
                               _validate_periodic_windows)


# ---------------------------------------------------------------------
# Channel fixture: FESOM-C NetCDF, core window [0, 10] deg lon + a
# 1-deg ghost band (10, 11], lat in [0, 2].
# ---------------------------------------------------------------------

def _make_fesomc_channel(tmp_path, n_time=3, u_val=5.0, v_val=0.0,
                         lon_min=-1.0, lon_max=11.0, dx=0.5):
    xs = np.arange(lon_min, lon_max + dx / 2, dx)
    ys = np.arange(0.0, 2.0 + dx / 2, dx)
    nx, ny = xs.size, ys.size
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel().astype(np.float64)
    lat = gy.ravel().astype(np.float64)
    n_nodes = lon.size
    depth = np.full(n_nodes, 50.0)

    elems = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            elems.append([n0 + 1, n0 + 2, n0 + nx + 1, n0 + 1])  # 1-based
            elems.append([n0 + 2, n0 + nx + 2, n0 + nx + 1, n0 + 2])
    nv = np.array(elems, dtype=np.int32)
    ne = nv.shape[0]
    e0 = nv[:, :3] - 1
    lon_c = np.array([lon[e].mean() for e in e0])
    lat_c = np.array([lat[e].mean() for e in e0])

    ds = xr.Dataset({
        "lon": (["node"], lon),
        "lat": (["node"], lat),
        "depth": (["node"], depth),
        "nv": (["nele", "four"], nv),
        "lon_elem": (["nele"], lon_c),
        "lat_elem": (["nele"], lat_c),
        "sigma_lev": (["sigma"], np.array([0.0, 0.5, 1.0])),
        "u2d": (["time", "nele"], np.full((n_time, ne), u_val)),
        "v2d": (["time", "nele"], np.full((n_time, ne), v_val)),
        "eta": (["time", "node"], np.zeros((n_time, n_nodes))),
    })
    nc_path = tmp_path / "channel.nc"
    ds.to_netcdf(nc_path)
    return nc_path


def _open_channel(nc_path):
    reader = get_reader("fesomc")
    mesh = reader.load_mesh(SimpleNamespace(meshpath=str(nc_path)))
    ds = reader.open_dataset(str(nc_path))
    return reader, mesh, ds


def _mode_num(dt=3600.0, **num_kw):
    mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)
    num = NumericsConfig(dt=dt, integrator="euler", output_freq=1,
                         **num_kw)
    return mode, num


# ---------------------------------------------------------------------
# Unit: _apply_periodic_wrap
# ---------------------------------------------------------------------

class TestApplyPeriodicWrapUnit:
    def _mesh(self, tmp_path):
        nc = _make_fesomc_channel(tmp_path)
        _, mesh, ds = _open_channel(nc)
        ds.close()
        return mesh

    def test_disabled_returns_same_object(self, tmp_path):
        mesh = self._mesh(tmp_path)
        st = make_state(np.array([5.0]), np.array([1.0]))
        assert _apply_periodic_wrap(st, mesh, None, None) is st

    def test_in_window_bitwise_untouched(self, tmp_path):
        mesh = self._mesh(tmp_path)
        lons = np.array([0.0, 3.141592653589793, 9.999999999])
        lats = np.array([0.3, 1.0, 1.7])
        st = make_state(lons.copy(), lats.copy())
        out = _apply_periodic_wrap(st, mesh, (0.0, 10.0), None)
        np.testing.assert_array_equal(out.lon, lons)   # bitwise
        np.testing.assert_array_equal(out.lat, lats)

    def test_east_crossing_wrapped_anchor_refreshed(self, tmp_path):
        mesh = self._mesh(tmp_path)
        st = make_state(np.array([10.4, 5.0]), np.array([1.0, 1.0]))
        from kelp.core.mesh import find_nearest_nodes
        nn0 = find_nearest_nodes(mesh, st.lon, st.lat)
        st = st.replace(nearest_node=nn0)
        out = _apply_periodic_wrap(st, mesh, (0.0, 10.0), None)
        assert out.lon[0] == pytest.approx(0.4)
        assert out.lon[1] == 5.0                        # untouched
        # anchor refreshed at the WRAPPED position
        nn_expect = find_nearest_nodes(mesh, np.array([0.4]),
                                       np.array([1.0]))[0]
        assert out.nearest_node[0] == nn_expect
        assert out.nearest_node[1] == nn0[1]

    def test_west_crossing_wrapped(self, tmp_path):
        mesh = self._mesh(tmp_path)
        st = make_state(np.array([-0.3]), np.array([1.0]))
        out = _apply_periodic_wrap(st, mesh, (0.0, 10.0), None)
        assert out.lon[0] == pytest.approx(9.7)

    def test_lat_window_wraps(self, tmp_path):
        mesh = self._mesh(tmp_path)
        st = make_state(np.array([5.0]), np.array([2.3]))
        out = _apply_periodic_wrap(st, mesh, None, (0.0, 2.0))
        assert out.lat[0] == pytest.approx(0.3)

    def test_dead_particles_untouched(self, tmp_path):
        mesh = self._mesh(tmp_path)
        st = make_state(np.array([10.4, 10.6]), np.array([1.0, 1.0]))
        status = st.status.copy()
        status[1] = ParticleStatus.OUTSIDE
        st = st.replace(status=status)
        out = _apply_periodic_wrap(st, mesh, (0.0, 10.0), None)
        assert out.lon[0] == pytest.approx(0.4)
        assert out.lon[1] == 10.6                       # death record kept

    def test_walk_elem_reset_for_wrapped_only(self, tmp_path):
        mesh = self._mesh(tmp_path)
        st = make_state(np.array([10.4, 5.0]), np.array([1.0, 1.0]))
        st = st.replace(elem=np.array([7, 9], dtype=np.int32))
        out = _apply_periodic_wrap(st, mesh, (0.0, 10.0), None)
        assert out.elem[0] == -1                        # re-seeded next walk
        assert out.elem[1] == 9

    def test_boundary_fold_never_lands_on_upper_edge(self, tmp_path):
        """float mod can round a tiny negative onto the period: the fold
        must land in [x0, x1), never exactly on x1."""
        mesh = self._mesh(tmp_path)
        st = make_state(np.array([-1e-18]), np.array([1.0]))
        out = _apply_periodic_wrap(st, mesh, (0.0, 10.0), None)
        assert 0.0 <= out.lon[0] < 10.0


# ---------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------

class TestConfigValidation:
    def test_reversed_window_raises(self):
        with pytest.raises(ValueError, match="periodic_lon"):
            NumericsConfig(periodic_lon=[10.0, 0.0])

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="periodic_lon"):
            NumericsConfig(periodic_lon=[0.0, 5.0, 10.0])

    def test_non_finite_raises(self):
        with pytest.raises(ValueError, match="periodic_lat"):
            NumericsConfig(periodic_lat=[0.0, np.inf])

    def test_lon_window_outside_branch_raises(self):
        """Integrators normalize positions to [-180, 180) after every
        displacement, so a window outside that branch would misfold even
        stationary particles (review round 1: [300, 307] folded
        305 -> 302); round 2 made the bound STRICT — a window merely
        touching ±180 lets the branch wrap teleport ghost-band
        crossings (measured 340-deg jump)."""
        for bad in ([300.0, 307.0], [170.0, 190.0],
                    [-180.0, -170.0], [170.0, 180.0]):
            with pytest.raises(ValueError, match="periodic_lon"):
                NumericsConfig(periodic_lon=bad)

    def test_lat_window_outside_poles_raises(self):
        with pytest.raises(ValueError, match="periodic_lat"):
            NumericsConfig(periodic_lat=[-95.0, 0.0])
        # strict at the poles too: clamp_lat swallows the overshoot
        # before the fold could see it
        with pytest.raises(ValueError, match="periodic_lat"):
            NumericsConfig(periodic_lat=[80.0, 90.0])

    def test_valid_windows_coerced_to_floats(self):
        num = NumericsConfig(periodic_lon=[0, 10], periodic_lat=[0, 2])
        assert num.periodic_lon == (0.0, 10.0)
        assert num.periodic_lat == (0.0, 2.0)


# ---------------------------------------------------------------------
# Channel tracking end-to-end
# ---------------------------------------------------------------------

class TestChannelTracking:
    def test_multi_lap_alive_and_exact(self, tmp_path):
        """~2 laps around a 10-deg channel: alive throughout, recorded
        positions always inside the window, final lon matches the
        unwrapped analytic displacement mod the window."""
        nc = _make_fesomc_channel(tmp_path, u_val=5.0)
        reader, mesh, ds = _open_channel(nc)
        mode, num = _mode_num(dt=3600.0, periodic_lon=[0.0, 10.0])
        st = make_state(np.array([9.9]), np.array([1.0]))
        n_steps = 130
        rec = run_single_realization(mesh, ds, reader, st, mode, num, 0,
                                     n_steps=n_steps, n_time_total=3,
                                     on_exhausted="cycle")
        ds.close()
        arrs = rec.to_arrays()
        lon_t = arrs["lon"][:, 0]
        status_t = arrs["status"][:, 0]
        assert np.all(status_t == ParticleStatus.ALIVE)
        assert np.all((lon_t >= 0.0) & (lon_t < 10.0))
        # per-step displacement of the euler step on a constant field
        dlon = 3600.0 * 5.0 / (np.cos(1.0 * DEG2RAD) * METERS_PER_DEG_LAT)
        expect = np.mod(9.9 + n_steps * dlon, 10.0)
        assert lon_t[-1] == pytest.approx(expect, abs=1e-6)
        # the particle genuinely lapped (unwrapped displacement > window)
        assert n_steps * dlon > 20.0

    def test_dies_without_periodic_survives_with(self, tmp_path):
        """The feature's raison d'etre: the same channel run exits the
        mesh and dies without periodic_lon, survives with it."""
        nc = _make_fesomc_channel(tmp_path, u_val=5.0)
        reader, mesh, ds = _open_channel(nc)
        st = make_state(np.array([9.9]), np.array([1.0]))
        mode, num_off = _mode_num(dt=3600.0)
        rec_off = run_single_realization(mesh, ds, reader, st, mode,
                                         num_off, 0, n_steps=60,
                                         n_time_total=3,
                                         on_exhausted="cycle")
        arrs_off = rec_off.to_arrays()
        assert arrs_off["status"][-1, 0] == ParticleStatus.OUTSIDE
        mode, num_on = _mode_num(dt=3600.0, periodic_lon=[0.0, 10.0])
        rec_on = run_single_realization(mesh, ds, reader, st, mode,
                                        num_on, 0, n_steps=60,
                                        n_time_total=3,
                                        on_exhausted="cycle")
        ds.close()
        arrs_on = rec_on.to_arrays()
        assert arrs_on["status"][-1, 0] == ParticleStatus.ALIVE

    def test_huge_window_bitwise_identical_to_off(self, tmp_path):
        """Feature ON with a window nothing ever crosses == feature OFF,
        bitwise (in-window particles must never be perturbed)."""
        nc = _make_fesomc_channel(tmp_path, u_val=1.0)
        reader, mesh, ds = _open_channel(nc)
        st = make_state(np.array([2.0, 5.0]), np.array([0.7, 1.3]))
        mode, num_off = _mode_num(dt=1800.0)
        rec_off = run_single_realization(mesh, ds, reader, st, mode,
                                         num_off, 0, n_steps=10,
                                         n_time_total=3,
                                         on_exhausted="cycle")
        mode, num_on = _mode_num(dt=1800.0, periodic_lon=[-50.0, 50.0],
                                 periodic_lat=[-50.0, 50.0])
        rec_on = run_single_realization(mesh, ds, reader, st, mode,
                                        num_on, 0, n_steps=10,
                                        n_time_total=3,
                                        on_exhausted="cycle")
        ds.close()
        a, b = rec_off.to_arrays(), rec_on.to_arrays()
        np.testing.assert_array_equal(a["lon"], b["lon"])
        np.testing.assert_array_equal(a["lat"], b["lat"])
        np.testing.assert_array_equal(a["status"], b["status"])

    def test_seed_outside_window_wrapped_at_start(self, tmp_path):
        nc = _make_fesomc_channel(tmp_path, u_val=0.0)
        reader, mesh, ds = _open_channel(nc)
        mode, num = _mode_num(dt=1800.0, periodic_lon=[0.0, 10.0])
        st = make_state(np.array([10.5]), np.array([1.0]))  # ghost band
        rec = run_single_realization(mesh, ds, reader, st, mode, num, 0,
                                     n_steps=2, n_time_total=3,
                                     on_exhausted="cycle")
        ds.close()
        arrs = rec.to_arrays()
        assert arrs["lon"][0, 0] == pytest.approx(0.5)
        assert np.all(arrs["status"][:, 0] == ParticleStatus.ALIVE)


    def test_backward_multi_lap(self, tmp_path):
        """Backward tracking crosses the WEST side repeatedly — the
        westward fold end-to-end (the two-sided ghost band makes it
        reachable), mirroring the forward multi-lap pin."""
        nc = _make_fesomc_channel(tmp_path, u_val=5.0)
        reader, mesh, ds = _open_channel(nc)
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=-1.0)
        num = NumericsConfig(dt=3600.0, integrator="euler", output_freq=1,
                             periodic_lon=[0.0, 10.0])
        st = make_state(np.array([0.1]), np.array([1.0]))
        n_steps = 130
        rec = run_single_realization(mesh, ds, reader, st, mode, num, 0,
                                     n_steps=n_steps, n_time_total=3,
                                     on_exhausted="cycle")
        ds.close()
        arrs = rec.to_arrays()
        lon_t = arrs["lon"][:, 0]
        assert np.all(arrs["status"][:, 0] == ParticleStatus.ALIVE)
        assert np.all((lon_t >= 0.0) & (lon_t < 10.0))
        dlon = 3600.0 * 5.0 / (np.cos(1.0 * DEG2RAD) * METERS_PER_DEG_LAT)
        expect = np.mod(0.1 - n_steps * dlon, 10.0)
        assert lon_t[-1] == pytest.approx(expect, abs=1e-6)

    def test_fold_fires_neighbor_stays_bitwise(self, tmp_path):
        """When the fold FIRES for one particle, an in-window neighbour
        must still be bitwise identical to the feature-off run (not just
        when nothing crosses)."""
        nc = _make_fesomc_channel(tmp_path, u_val=5.0)
        reader, mesh, ds = _open_channel(nc)
        seeds = (np.array([9.9, 4.0]), np.array([1.0, 1.0]))
        mode, num_on = _mode_num(dt=3600.0, periodic_lon=[0.0, 10.0])
        rec_on = run_single_realization(
            mesh, ds, reader, make_state(*seeds), mode, num_on, 0,
            n_steps=10, n_time_total=3, on_exhausted="cycle")
        mode, num_off = _mode_num(dt=3600.0)
        rec_off = run_single_realization(
            mesh, ds, reader, make_state(*seeds), mode, num_off, 0,
            n_steps=10, n_time_total=3, on_exhausted="cycle")
        ds.close()
        a_on, a_off = rec_on.to_arrays(), rec_off.to_arrays()
        # particle 0 GENUINELY folded on the ON run (in-window always,
        # diverging from OFF) — particle 1 must be bitwise untouched
        assert (a_on["lon"][:, 0] != a_off["lon"][:, 0]).any()
        assert np.all((a_on["lon"][:, 0] >= 0.0)
                      & (a_on["lon"][:, 0] < 10.0))
        np.testing.assert_array_equal(a_on["lon"][:, 1],
                                      a_off["lon"][:, 1])
        np.testing.assert_array_equal(a_on["lat"][:, 1],
                                      a_off["lat"][:, 1])


class TestGuards:
    def test_rotation_combo_rejected_step_level(self, tmp_path):
        nc = _make_fesomc_channel(tmp_path)
        reader, mesh, ds = _open_channel(nc)
        mesh.rotation_angles = (50.0, 15.0, -90.0)
        mode, num = _mode_num(periodic_lon=[0.0, 10.0])
        with pytest.raises(ValueError, match="MESH-FRAME"):
            run_single_realization(mesh, ds, reader,
                                   make_state(np.array([5.0]),
                                              np.array([1.0])),
                                   mode, num, 0, n_steps=1,
                                   n_time_total=3, on_exhausted="cycle")
        ds.close()

    def test_window_mesh_no_overlap_rejected(self, tmp_path):
        nc = _make_fesomc_channel(tmp_path)
        reader, mesh, ds = _open_channel(nc)
        mode, num = _mode_num(periodic_lon=[100.0, 110.0])
        with pytest.raises(ValueError, match="does not overlap"):
            run_single_realization(mesh, ds, reader,
                                   make_state(np.array([5.0]),
                                              np.array([1.0])),
                                   mode, num, 0, n_steps=1,
                                   n_time_total=3, on_exhausted="cycle")
        ds.close()

    def test_one_sided_mesh_warns(self, tmp_path):
        """Mesh flush with the window's west edge: crossing west would
        die before the fold — a one-time RuntimeWarning names the side."""
        nc = _make_fesomc_channel(tmp_path, lon_min=0.0)   # no west ghost
        reader, mesh, ds = _open_channel(nc)
        mode, num = _mode_num(periodic_lon=[0.0, 10.0])
        with pytest.warns(RuntimeWarning, match="WEST/SOUTH"):
            run_single_realization(mesh, ds, reader,
                                   make_state(np.array([5.0]),
                                              np.array([1.0])),
                                   mode, num, 0, n_steps=1,
                                   n_time_total=3, on_exhausted="cycle")
        ds.close()

    def test_touching_exclusive_edge_rejected(self):
        """A mesh that only touches the window's EXCLUSIVE upper edge
        has zero coverage inside [lo, hi) — must raise, not warn
        (round-2 off-by-one: '<' let it through)."""
        fake = SimpleNamespace(nodes_lon=np.array([10.0, 10.5, 11.0]),
                               nodes_lat=np.array([0.0, 1.0, 2.0]))
        with pytest.raises(ValueError, match="does not overlap"):
            _validate_periodic_windows(fake, (0.0, 10.0), None,
                                       warn=False)

    def test_dateline_touching_mesh_rejected_both_sides(self, tmp_path):
        """A channel whose mesh reaches ±180 is rejected from EITHER
        side (round-3 review: the first heuristic caught +180 but not
        -180): the branch wrap either teleports folds or makes the
        ghost band unreachable."""
        # east side: mesh up to +180
        nc = _make_fesomc_channel(tmp_path, lon_min=170.0, lon_max=180.0)
        reader, mesh, ds = _open_channel(nc)
        mode, num = _mode_num(periodic_lon=[172.0, 179.0])
        with pytest.raises(ValueError, match="±180|180 cut"):
            run_single_realization(mesh, ds, reader,
                                   make_state(np.array([175.0]),
                                              np.array([1.0])),
                                   mode, num, 0, n_steps=1,
                                   n_time_total=3, on_exhausted="cycle")
        ds.close()
        # west side: mesh down to -180 (unit-level; same rule)
        fake = SimpleNamespace(nodes_lon=np.linspace(-180.0, -170.0, 21),
                               nodes_lat=np.linspace(0.0, 2.0, 21))
        with pytest.raises(ValueError, match="±180|180 cut"):
            _validate_periodic_windows(fake, (-179.0, -172.0), None,
                                       warn=False)
        # 0..360-branch storage of the same east-touching geometry
        fake2 = SimpleNamespace(nodes_lon=np.linspace(180.0, 190.0, 21),
                                nodes_lat=np.linspace(0.0, 2.0, 21))
        with pytest.raises(ValueError, match="±180|180 cut"):
            _validate_periodic_windows(fake2, (-179.0, -172.0), None,
                                       warn=False)

    def test_global_band_mesh_rejected_with_periodic_lon(self):
        """A global band wraps natively at ±180 — a periodic lon window
        on it is nonsense and is rejected (policy pinned)."""
        fake = SimpleNamespace(
            nodes_lon=-180.0 + 360.0 * np.arange(72) / 72,
            nodes_lat=np.full(72, 45.0))
        with pytest.raises(ValueError, match="±180|180 cut"):
            _validate_periodic_windows(fake, (0.0, 10.0), None,
                                       warn=False)

    def test_zero_360_branch_mesh_validates_and_tracks(self, tmp_path):
        """A channel stored in the 0..360 branch (nodes 349..356) with
        its window expressed in [-180, 180) terms must validate (round-2
        review: raw min/max false-rejected it as non-overlapping) AND
        track multi-lap through the fold."""
        nc = _make_fesomc_channel(tmp_path, u_val=5.0,
                                  lon_min=349.0, lon_max=356.0)
        reader, mesh, ds = _open_channel(nc)
        mode, num = _mode_num(dt=3600.0, periodic_lon=[-10.0, -5.0])
        st = make_state(np.array([-5.1]), np.array([1.0]))
        rec = run_single_realization(mesh, ds, reader, st, mode, num, 0,
                                     n_steps=60, n_time_total=3,
                                     on_exhausted="cycle")
        ds.close()
        arrs = rec.to_arrays()
        assert np.all(arrs["status"][:, 0] == ParticleStatus.ALIVE)
        assert np.all((arrs["lon"][:, 0] >= -10.0)
                      & (arrs["lon"][:, 0] < -5.0))

    def test_run_tracking_rejects_before_creating_output(self, tmp_path):
        """The raising sanity checks run PRE-WRITER in run_tracking: a
        typo'd window must not leave an empty output file behind."""
        nc = _make_fesomc_channel(tmp_path)
        part = tmp_path / "seeds.dat"
        part.write_text("5.0 1.0\n")
        out_nc = tmp_path / "reject.nc"
        cfg = AppConfig(
            mode=ModeConfig(Fdrift2D=True, time_dir=1.0),
            numerics=NumericsConfig(dt=1800.0, integrator="euler",
                                    periodic_lon=[100.0, 110.0]),
            mesh=MeshConfig(type="fesomc", meshpath=str(nc)),
            data=DataConfig(ncinput=str(nc), inicloud=str(part),
                            output_path=str(out_nc)),
            time=TimeConfig(start_time_idx=0, n_steps=2),
        )
        from kelp.tracking import run_tracking
        with pytest.raises(ValueError, match="does not overlap"):
            run_tracking(cfg, verbose=False)
        assert not out_nc.exists()


class TestDelayedRelease:
    def test_dormant_ghost_seed_folds_at_init(self, tmp_path):
        """A NOT_RELEASED seed in the ghost band must fold at init like
        an immediate seed (include_dormant) — otherwise it activates at
        the unfolded position and dies where its twin survives.
        Discriminating geometry (round-2 review): seed DEEP in the band
        (10.9) with fast flow, so the unfolded dormant twin's first step
        exits the mesh before any post-step fold could catch it —
        include_dormant=False measurably yields [ALIVE, OUTSIDE]."""
        from kelp.core.release import build_release_state
        nc = _make_fesomc_channel(tmp_path, u_val=5.0)
        reader, mesh, ds = _open_channel(nc)
        dt = 3600.0
        # twin seeds at 10.9 (deep ghost): immediate + released at step 2
        st = build_release_state(
            np.array([10.9, 10.9]), np.array([1.0, 1.0]), None,
            np.array([0.0, 2.0 * dt]),
            n_pulses=1, pulse_interval=0.0, dt=dt, start_step_i=0)
        mode, num = _mode_num(dt=dt, periodic_lon=[0.0, 10.0])
        rec = run_single_realization(mesh, ds, reader, st, mode, num, 0,
                                     n_steps=5, n_time_total=3,
                                     on_exhausted="cycle")
        ds.close()
        arrs = rec.to_arrays()
        # BOTH alive at the end (the dormant twin would be OUTSIDE
        # without the init fold), both inside the window
        assert np.all(arrs["status"][-1, :] == ParticleStatus.ALIVE)
        final_lon = arrs["lon"][-1, :]
        assert np.all((final_lon >= 0.0) & (final_lon < 10.0))

    def test_unit_include_dormant_flag(self, tmp_path):
        nc = _make_fesomc_channel(tmp_path)
        _, mesh, ds = _open_channel(nc)
        ds.close()
        st = make_state(np.array([10.5]), np.array([1.0]))
        status = st.status.copy()
        status[0] = ParticleStatus.NOT_RELEASED
        st = st.replace(status=status)
        out = _apply_periodic_wrap(st, mesh, (0.0, 10.0), None)
        assert out.lon[0] == 10.5                       # excluded by default
        out2 = _apply_periodic_wrap(st, mesh, (0.0, 10.0), None,
                                    include_dormant=True)
        assert out2.lon[0] == pytest.approx(0.5)


class TestFunctionalAPI:
    def test_track_particles_threads_periodic(self, tmp_path):
        from kelp.tracking import track_particles
        nc = _make_fesomc_channel(tmp_path, u_val=5.0)
        reader, mesh, ds = _open_channel(nc)
        rec = track_particles(
            mesh, ds, reader,
            np.array([9.9]), np.array([1.0]),
            start_time_idx=0, n_steps=60, on_exhausted="cycle",
            dt=3600.0, integrator="euler",
            periodic_lon=(0.0, 10.0))
        ds.close()
        arrs = rec.to_arrays()
        assert np.all(arrs["status"][:, 0] == ParticleStatus.ALIVE)
        assert np.all((arrs["lon"][:, 0] >= 0.0)
                      & (arrs["lon"][:, 0] < 10.0))


class TestRunTrackingMetadata:
    @pytest.mark.parametrize("ext", [".nc", ".zarr"])
    def test_periodic_attrs_written_when_set(self, tmp_path, ext):
        nc = _make_fesomc_channel(tmp_path, u_val=1.0)
        part = tmp_path / "seeds.dat"
        part.write_text("5.0 1.0\n")
        out_path = tmp_path / f"out{ext}"
        cfg = AppConfig(
            mode=ModeConfig(Fdrift2D=True, time_dir=1.0),
            numerics=NumericsConfig(dt=1800.0, integrator="euler",
                                    periodic_lon=[0.0, 10.0]),
            mesh=MeshConfig(type="fesomc", meshpath=str(nc)),
            data=DataConfig(ncinput=str(nc), inicloud=str(part),
                            output_path=str(out_path)),
            time=TimeConfig(start_time_idx=0, n_steps=2),
        )
        from kelp.tracking import run_tracking
        run_tracking(cfg, verbose=False)
        if ext == ".zarr":
            with xr.open_zarr(out_path, consolidated=False) as d:
                assert list(d.attrs["periodic_lon"]) == [0.0, 10.0]
                assert "periodic_lat" not in d.attrs
        else:
            with xr.open_dataset(out_path) as d:
                assert list(d.attrs["periodic_lon"]) == [0.0, 10.0]
                assert "periodic_lat" not in d.attrs  # unset stays absent
