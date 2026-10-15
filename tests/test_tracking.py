"""End-to-end tests for kelp.tracking.

Uses synthetic FESOM-C and FESOM2 data to run actual tracking simulations
and verify particle trajectories are physically correct.
"""
from __future__ import annotations


import numpy as np
import pytest
import xarray as xr

import kelp.tracking as tracking_mod


from kelp.tracking import (
    TrajectoryRecord,
    run_single_realization,
    run_tracking,
    track_particles,
    _resolve_start_date,
    _shift_to_midpoints,
)
from kelp.core.config import (
    ModeConfig, NumericsConfig, MeshConfig, DataConfig,
    TimeConfig, AppConfig,
)
from kelp.readers import get_reader
from kelp.core.particles import make_state, ParticleState, ParticleStatus
from kelp.core.geometry import METERS_PER_DEG_LAT, DEG2RAD


# ---------------------------------------------------------------------------
# Helper: create FESOM-C test data with 3×3 grid (8 triangles)
# ---------------------------------------------------------------------------

def _make_fesomc_3x3(tmp_path, n_time=5, u_val=1.0, v_val=0.0):
    """Create a FESOM-C NetCDF with a 3×3 node grid and constant velocity.

    Returns (nc_path, mesh_cfg, mode_cfg).
    """
    from types import SimpleNamespace

    xs = np.array([4.0, 5.0, 6.0])
    ys = np.array([4.0, 5.0, 6.0])
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel().astype(np.float64)   # 9 nodes
    lat = gy.ravel().astype(np.float64)
    depth = np.full(9, 10.0, dtype=np.float64)

    # 8 triangles (lower-left + upper-right per cell)
    elems = []
    for j in range(2):
        for i in range(2):
            n0 = j * 3 + i
            n1 = n0 + 1
            n2 = n0 + 3
            n3 = n0 + 4
            elems.append([n0 + 1, n1 + 1, n2 + 1, n0 + 1])   # 1-based, tri padded
            elems.append([n1 + 1, n3 + 1, n2 + 1, n1 + 1])
    nv = np.array(elems, dtype=np.int32)
    n_elem = nv.shape[0]

    # Element centres
    elem_nodes_0 = nv[:, :3] - 1  # 0-based for computing centres
    lon_c = np.array([lon[e].mean() for e in elem_nodes_0], dtype=np.float64)
    lat_c = np.array([lat[e].mean() for e in elem_nodes_0], dtype=np.float64)

    sigma_lev = np.array([0.0, 0.5, 1.0], dtype=np.float64)

    u2d = np.full((n_time, n_elem), u_val, dtype=np.float64)
    v2d = np.full((n_time, n_elem), v_val, dtype=np.float64)
    eta = np.zeros((n_time, 9), dtype=np.float64)

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
    })

    nc_path = tmp_path / "fesomc_tracking.nc"
    ds.to_netcdf(nc_path)

    mesh_cfg = SimpleNamespace(meshpath=str(nc_path))
    mode_cfg = SimpleNamespace(Fdrift3D=False, Fdrift2D=True, Vpuls=False, time_dir=1.0)
    return nc_path, mesh_cfg, mode_cfg


# ---------------------------------------------------------------------------
# Helper: create FESOM2 test data with 3×3 grid
# ---------------------------------------------------------------------------

def _make_fesom2_3x3(tmp_path, n_time=5, u_val=1.0, v_val=0.5):
    """Create FESOM2 ASCII mesh + NetCDF velocity.

    Returns (mesh_dir, vel_path, mesh_cfg).
    """
    from types import SimpleNamespace

    mesh_dir = tmp_path / "f2mesh"
    mesh_dir.mkdir()

    xs = np.array([4.0, 5.0, 6.0])
    ys = np.array([4.0, 5.0, 6.0])
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel()  # 9 nodes
    lat = gy.ravel()

    with open(mesh_dir / "nod2d.out", "w") as f:
        f.write("9\n")
        for i in range(9):
            f.write(f"{i+1} {lon[i]:.6f} {lat[i]:.6f} 0\n")

    elems = []
    for j in range(2):
        for i in range(2):
            n0 = j * 3 + i
            elems.append([n0 + 1, n0 + 2, n0 + 4])       # lower-left tri
            elems.append([n0 + 2, n0 + 5, n0 + 4])        # upper-right tri
    with open(mesh_dir / "elem2d.out", "w") as f:
        f.write(f"{len(elems)}\n")
        for e in elems:
            f.write(f"{e[0]} {e[1]} {e[2]}\n")

    with open(mesh_dir / "depth.out", "w") as f:
        for _ in range(9):
            f.write("10.0\n")

    n_elem = len(elems)
    u = np.full((n_time, n_elem), u_val, dtype=np.float64)
    v = np.full((n_time, n_elem), v_val, dtype=np.float64)
    ssh = np.zeros((n_time, 9), dtype=np.float64)

    # FESOM2 2D velocities live on ELEMENTS (size = n_elem); the dim name
    # must be an element hint so layout detection matches the element-count
    # data (the 3D fixture uses "elem" likewise).  Naming it "nod2" here
    # would misdeclare element data as node-layout and trip the
    # uv_interp x uv_layout guard.
    vel_path = tmp_path / "f2vel.nc"
    xr.Dataset({
        "u": (["time", "elem"], u),
        "v": (["time", "elem"], v),
        "ssh": (["time", "node"], ssh),
    }).to_netcdf(vel_path)

    mesh_cfg = SimpleNamespace(meshpath=str(mesh_dir))
    return mesh_dir, vel_path, mesh_cfg


# ===================================================================
# TrajectoryRecord tests
# ===================================================================

class TestTrajectoryRecord:
    def test_append_and_count(self):
        rec = TrajectoryRecord()
        state = make_state(np.array([5.0]), np.array([5.0]))
        rec.append(state, 0)
        rec.append(state, 1)
        assert rec.n_steps == 2

    def test_to_arrays(self):
        rec = TrajectoryRecord()
        s1 = make_state(np.array([5.0, 6.0]), np.array([5.0, 6.0]))
        s2 = make_state(np.array([5.1, 6.1]), np.array([5.1, 6.1]))
        rec.append(s1, 0)
        rec.append(s2, 1)
        arrs = rec.to_arrays()
        assert arrs["lon"].shape == (2, 2)  # (T, P)
        assert arrs["time"].shape == (2,)

    def test_append_copies(self):
        """Appending should copy arrays so later mutations don't affect record."""
        rec = TrajectoryRecord()
        state = make_state(np.array([5.0]), np.array([5.0]))
        rec.append(state, 0)
        state.lon[0] = 99.0
        assert rec.lons[0][0] == 5.0   # should not be mutated


# ===================================================================
# End-to-end FESOM-C tracking
# ===================================================================

class TestFesomCTracking:
    def test_eastward_flow(self, tmp_path):
        """Constant eastward flow → particle moves east over 3 steps."""
        nc_path, mesh_cfg, mode_cfg = _make_fesomc_3x3(tmp_path, u_val=1.0, v_val=0.0)
        reader = get_reader("fesomc")
        mesh = reader.load_mesh(mesh_cfg)
        ds = reader.open_dataset(str(nc_path))

        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)
        numerics = NumericsConfig(dt=900.0, integrator="euler", output_freq=1)
        init = make_state(np.array([5.0]), np.array([5.0]))

        rec = run_single_realization(mesh, ds, reader, init, mode, numerics, 0, end_time_idx=2)
        ds.close()

        arrs = rec.to_arrays()
        assert arrs["lon"].shape[0] == 2   # end - start = 2 steps (records after steps 1, 2)
        # Particle should move east monotonically
        assert arrs["lon"][-1, 0] > arrs["lon"][0, 0]
        # Latitude should stay ~constant
        assert abs(arrs["lat"][-1, 0] - 5.0) < 0.01

    def test_northward_flow(self, tmp_path):
        """Constant northward flow → particle moves north."""
        nc_path, mesh_cfg, mode_cfg = _make_fesomc_3x3(tmp_path, u_val=0.0, v_val=1.0)
        reader = get_reader("fesomc")
        mesh = reader.load_mesh(mesh_cfg)
        ds = reader.open_dataset(str(nc_path))

        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)
        numerics = NumericsConfig(dt=900.0, integrator="euler", output_freq=1)
        init = make_state(np.array([5.0]), np.array([5.0]))

        rec = run_single_realization(mesh, ds, reader, init, mode, numerics, 0, end_time_idx=2)
        ds.close()

        arrs = rec.to_arrays()
        assert arrs["lat"][-1, 0] > 5.0
        assert abs(arrs["lon"][-1, 0] - 5.0) < 0.01

    def test_backward_tracking(self, tmp_path):
        """time_dir=-1 with eastward field → particle moves west."""
        nc_path, mesh_cfg, _ = _make_fesomc_3x3(tmp_path, u_val=1.0, v_val=0.0)
        reader = get_reader("fesomc")
        mesh = reader.load_mesh(mesh_cfg)
        ds = reader.open_dataset(str(nc_path))

        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=-1.0)
        numerics = NumericsConfig(dt=900.0, integrator="euler", output_freq=1)
        init = make_state(np.array([5.0]), np.array([5.0]))

        rec = run_single_realization(mesh, ds, reader, init, mode, numerics, 0, end_time_idx=2)
        ds.close()

        arrs = rec.to_arrays()
        assert arrs["lon"][-1, 0] < 5.0

    def test_displacement_magnitude(self, tmp_path):
        """Check total displacement after N steps matches analytical.

        3 steps × dt=900s × U=1 m/s → total_dx = 2700m
        Δlon = 1800 / (cos(5°)·METERS_PER_DEG_LAT) ≈ 0.01623° (end_time_idx=2 → 2 steps)
        """
        nc_path, mesh_cfg, _ = _make_fesomc_3x3(tmp_path, u_val=1.0, v_val=0.0, n_time=5)
        reader = get_reader("fesomc")
        mesh = reader.load_mesh(mesh_cfg)
        ds = reader.open_dataset(str(nc_path))

        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)
        numerics = NumericsConfig(dt=900.0, integrator="euler", output_freq=1)
        init = make_state(np.array([5.0]), np.array([5.0]))

        rec = run_single_realization(mesh, ds, reader, init, mode, numerics, 0, end_time_idx=2)
        ds.close()

        z0 = np.cos(5.0 * DEG2RAD) * METERS_PER_DEG_LAT
        expected_dlon = 2 * 900.0 / z0
        actual_dlon = rec.lons[-1][0] - 5.0
        assert actual_dlon == pytest.approx(expected_dlon, rel=0.1)

    def test_multiple_particles(self, tmp_path):
        """Two particles should both track correctly."""
        nc_path, mesh_cfg, _ = _make_fesomc_3x3(tmp_path, u_val=1.0, v_val=0.5)
        reader = get_reader("fesomc")
        mesh = reader.load_mesh(mesh_cfg)
        ds = reader.open_dataset(str(nc_path))

        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)
        numerics = NumericsConfig(dt=600.0, integrator="euler", output_freq=1)
        init = make_state(
            np.array([4.8, 5.2]),
            np.array([5.0, 5.0]),
        )

        rec = run_single_realization(mesh, ds, reader, init, mode, numerics, 0, end_time_idx=2)
        ds.close()

        arrs = rec.to_arrays()
        # Both particles should move NE
        assert arrs["lon"][-1, 0] > 4.8
        assert arrs["lon"][-1, 1] > 5.2
        assert arrs["lat"][-1, 0] > 5.0
        assert arrs["lat"][-1, 1] > 5.0

    def test_itf_output_interval(self, tmp_path):
        """output_freq=2 should output every other step."""
        nc_path, mesh_cfg, _ = _make_fesomc_3x3(tmp_path, u_val=1.0, v_val=0.0, n_time=6)
        reader = get_reader("fesomc")
        mesh = reader.load_mesh(mesh_cfg)
        ds = reader.open_dataset(str(nc_path))

        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)
        numerics = NumericsConfig(dt=900.0, integrator="euler", output_freq=2)
        init = make_state(np.array([5.0]), np.array([5.0]))

        # 5 timesteps (0..4), output_freq=2 → output at steps 2, 4 → 2 snapshots
        # (after step 2 (itout hits 2), after step 4 (itout hits 2 again))
        rec = run_single_realization(mesh, ds, reader, init, mode, numerics, 0, end_time_idx=4)
        ds.close()

        # Should have fewer outputs than timesteps
        assert rec.n_steps < 5
        assert rec.n_steps >= 2


# ===================================================================
# End-to-end FESOM2 tracking
# ===================================================================

class TestFesom2Tracking:
    def test_ne_flow(self, tmp_path):
        """NE flow (u=1, v=0.5) → particle moves northeast."""
        mesh_dir, vel_path, mesh_cfg = _make_fesom2_3x3(tmp_path, u_val=1.0, v_val=0.5)
        reader = get_reader("fesom2")
        mesh = reader.load_mesh(mesh_cfg)
        ds = reader.open_dataset(str(vel_path))

        rec = track_particles(
            mesh, ds, reader,
            lon0=np.array([5.0]),
            lat0=np.array([5.0]),
            start_time_idx=0, end_time_idx=2, dt=600.0,
        )
        ds.close()

        arrs = rec.to_arrays()
        assert arrs["lon"][-1, 0] > 5.0
        assert arrs["lat"][-1, 0] > 5.0

    def test_rk2_integrator(self, tmp_path):
        """RK2 should produce valid trajectory."""
        mesh_dir, vel_path, mesh_cfg = _make_fesom2_3x3(tmp_path, u_val=1.0, v_val=0.0)
        reader = get_reader("fesom2")
        mesh = reader.load_mesh(mesh_cfg)
        ds = reader.open_dataset(str(vel_path))

        rec = track_particles(
            mesh, ds, reader,
            lon0=np.array([5.0]),
            lat0=np.array([5.0]),
            start_time_idx=0, end_time_idx=2, dt=600.0,
            integrator="rk2",
        )
        ds.close()

        assert rec.n_steps == 2
        assert rec.lons[-1][0] > 5.0

    def test_rk4_integrator(self, tmp_path):
        """RK4 should produce valid trajectory."""
        mesh_dir, vel_path, mesh_cfg = _make_fesom2_3x3(tmp_path, u_val=0.5, v_val=0.5)
        reader = get_reader("fesom2")
        mesh = reader.load_mesh(mesh_cfg)
        ds = reader.open_dataset(str(vel_path))

        rec = track_particles(
            mesh, ds, reader,
            lon0=np.array([5.0]),
            lat0=np.array([5.0]),
            start_time_idx=0, end_time_idx=2, dt=600.0,
            integrator="rk4",
        )
        ds.close()

        assert rec.n_steps == 2
        assert rec.lons[-1][0] > 5.0
        assert rec.lats[-1][0] > 5.0


# ===================================================================
# track_particles functional interface
# ===================================================================

class TestTrackParticles:
    def test_defaults(self, tmp_path):
        """track_particles with defaults should work."""
        nc_path, mesh_cfg, _ = _make_fesomc_3x3(tmp_path, n_time=12, u_val=0.5, v_val=0.0)
        reader = get_reader("fesomc")
        mesh = reader.load_mesh(mesh_cfg)
        ds = reader.open_dataset(str(nc_path))

        rec = track_particles(
            mesh, ds, reader,
            lon0=np.array([5.0]),
            lat0=np.array([5.0]),
            start_time_idx=0, end_time_idx=3,
        )
        ds.close()

        assert rec.n_steps == 3
        assert rec.lons[-1][0] > 5.0

    def test_backward(self, tmp_path):
        """track_particles with time_dir=-1."""
        nc_path, mesh_cfg, _ = _make_fesomc_3x3(tmp_path, n_time=12, u_val=1.0, v_val=0.0)
        reader = get_reader("fesomc")
        mesh = reader.load_mesh(mesh_cfg)
        ds = reader.open_dataset(str(nc_path))

        rec = track_particles(
            mesh, ds, reader,
            lon0=np.array([5.0]),
            lat0=np.array([5.0]),
            start_time_idx=0, end_time_idx=3,
            time_dir=-1.0,
        )
        ds.close()

        assert rec.lons[-1][0] < 5.0

    def test_passes_vertical_kwargs_into_numerics(self, monkeypatch):
        """Functional interface should preserve sigma-ready vertical kwargs."""
        captured = {}

        def fake_run_single_realization(mesh, ds, reader, init_state, mode, numerics, **kwargs):
            captured["Dmin"] = numerics.Dmin
            captured["z0b_min"] = numerics.z0b_min
            return TrajectoryRecord()

        monkeypatch.setattr(tracking_mod, "run_single_realization", fake_run_single_realization)

        rec = track_particles(
            mesh=object(),
            ds=object(),
            reader=object(),
            lon0=np.array([5.0]),
            lat0=np.array([5.0]),
            n_steps=0,
            Dmin=1.5,
            z0b_min=0.03,
        )

        assert isinstance(rec, TrajectoryRecord)
        assert captured["Dmin"] == pytest.approx(1.5)
        assert captured["z0b_min"] == pytest.approx(0.03)


# ===================================================================
# Time handling: n_steps, stop/cycle, backward compat
# ===================================================================

class TestTimeHandling:
    def test_n_steps_basic(self, tmp_path):
        """n_steps=3 on a 5-step dataset → 3 output snapshots."""
        nc_path, mesh_cfg, _ = _make_fesomc_3x3(tmp_path, n_time=5, u_val=1.0, v_val=0.0)
        reader = get_reader("fesomc")
        mesh = reader.load_mesh(mesh_cfg)
        ds = reader.open_dataset(str(nc_path))

        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)
        numerics = NumericsConfig(dt=900.0, integrator="euler", output_freq=1)
        init = make_state(np.array([5.0]), np.array([5.0]))

        rec = run_single_realization(
            mesh, ds, reader, init, mode, numerics,
            start_time_idx=0, n_steps=3,
        )
        ds.close()

        assert rec.n_steps == 3

    def test_stop_on_exhausted(self, tmp_path):
        """n_steps=10 on 3-step dataset with on_exhausted='stop' → stops at 3."""
        nc_path, mesh_cfg, _ = _make_fesomc_3x3(tmp_path, n_time=3, u_val=1.0, v_val=0.0)
        reader = get_reader("fesomc")
        mesh = reader.load_mesh(mesh_cfg)
        ds = reader.open_dataset(str(nc_path))

        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)
        numerics = NumericsConfig(dt=900.0, integrator="euler", output_freq=1)
        init = make_state(np.array([5.0]), np.array([5.0]))

        rec = run_single_realization(
            mesh, ds, reader, init, mode, numerics,
            start_time_idx=0, n_steps=10, on_exhausted="stop",
        )
        ds.close()

        # Should stop after 3 steps (indices 0, 1, 2), not crash
        assert rec.n_steps == 3

    def test_cycle_on_exhausted(self, tmp_path):
        """n_steps=7 on 3-step dataset with on_exhausted='cycle' → 7 snapshots."""
        nc_path, mesh_cfg, _ = _make_fesomc_3x3(tmp_path, n_time=3, u_val=1.0, v_val=0.0)
        reader = get_reader("fesomc")
        mesh = reader.load_mesh(mesh_cfg)
        ds = reader.open_dataset(str(nc_path))

        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)
        numerics = NumericsConfig(dt=900.0, integrator="euler", output_freq=1)
        init = make_state(np.array([5.0]), np.array([5.0]))

        rec = run_single_realization(
            mesh, ds, reader, init, mode, numerics,
            start_time_idx=0, n_steps=7, on_exhausted="cycle",
        )
        ds.close()

        assert rec.n_steps == 7
        # File indices should wrap: 0,1,2,0,1,2,0
        # format 2: labels are the run clock in seconds (this axis has no
        # time coordinate: records dt apart from record 0), not the cycled
        # record indices 0,1,2,0,1,2,0 of format 1
        assert list(rec.times) == [900.0 * k for k in range(1, 8)]

    def test_backward_cycle(self, tmp_path):
        """Backward cycling: start=2, time_dir=-1, n_steps=5, cycle."""
        nc_path, mesh_cfg, _ = _make_fesomc_3x3(tmp_path, n_time=3, u_val=1.0, v_val=0.0)
        reader = get_reader("fesomc")
        mesh = reader.load_mesh(mesh_cfg)
        ds = reader.open_dataset(str(nc_path))

        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=-1.0)
        numerics = NumericsConfig(dt=900.0, integrator="euler", output_freq=1)
        init = make_state(np.array([5.0]), np.array([5.0]))

        rec = run_single_realization(
            mesh, ds, reader, init, mode, numerics,
            start_time_idx=2, n_steps=5, on_exhausted="cycle",
        )
        ds.close()

        assert rec.n_steps == 5
        # file_idx = 2+step*(-1): 2, 1, 0, -1%3=2, -2%3=1
        # format 2: start record 2 = 1800 s, labels end-of-step backward
        assert list(rec.times) == [900.0, 0.0, -900.0, -1800.0, -2700.0]

    def test_end_time_idx_backward_compat(self, tmp_path):
        """Old-style end_time_idx=2 produces same result as n_steps=2
        (end - start, the rule of config.resolve_time_params)."""
        nc_path, mesh_cfg, _ = _make_fesomc_3x3(tmp_path, n_time=5, u_val=1.0, v_val=0.0)
        reader = get_reader("fesomc")
        mesh = reader.load_mesh(mesh_cfg)

        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)
        numerics = NumericsConfig(dt=900.0, integrator="euler", output_freq=1)
        init = make_state(np.array([5.0]), np.array([5.0]))

        ds1 = reader.open_dataset(str(nc_path))
        rec_old = run_single_realization(
            mesh, ds1, reader, init, mode, numerics,
            start_time_idx=0, end_time_idx=2,
        )
        ds1.close()

        ds2 = reader.open_dataset(str(nc_path))
        rec_new = run_single_realization(
            mesh, ds2, reader, init, mode, numerics,
            start_time_idx=0, n_steps=2,
        )
        ds2.close()

        assert rec_old.n_steps == rec_new.n_steps
        np.testing.assert_allclose(rec_old.lons[-1], rec_new.lons[-1])


# ===================================================================
# dt auto-detection and validation
# ===================================================================

def _make_fesomc_3x3_with_time(tmp_path, n_time=5, dt_seconds=1800.0,
                                u_val=1.0, v_val=0.0):
    """Like _make_fesomc_3x3 but with a datetime64 time coordinate."""
    import pandas as pd
    from types import SimpleNamespace

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

    nc_path = tmp_path / "fesomc_timed.nc"
    ds.to_netcdf(nc_path)
    return nc_path


class TestDtAutoDetect:
    def test_auto_detect_dt(self, tmp_path):
        """dt=None in config → auto-detected from file time coordinate."""
        nc_path = _make_fesomc_3x3_with_time(tmp_path, n_time=5, dt_seconds=1800.0)

        # Create particles file
        state = make_state(np.array([5.0]), np.array([5.0]))
        particles_path = tmp_path / "particles.dat"
        state.save(str(particles_path))

        config = AppConfig(
            mode=ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0),
            numerics=NumericsConfig(dt=None, integrator="euler", output_freq=1),
            mesh=MeshConfig(type="fesomc", meshpath=str(nc_path)),
            data=DataConfig(ncinput=str(nc_path), inicloud=str(particles_path)),
            time=TimeConfig(n_steps=3),
        )

        records = run_tracking(config, verbose=False)
        # dt should have been set to 1800.0
        assert config.numerics.dt == pytest.approx(1800.0)
        assert records[0].n_steps == 3

    def test_dt_mismatch_in_discrete_mode_refuses(self, tmp_path):
        """Configured dt unlike the file step in discrete mode → ValueError
        before any output (was a printed warning until 2026-09-24)."""
        nc_path = _make_fesomc_3x3_with_time(tmp_path, n_time=5, dt_seconds=1800.0)

        state = make_state(np.array([5.0]), np.array([5.0]))
        particles_path = tmp_path / "particles.dat"
        state.save(str(particles_path))

        config = AppConfig(
            mode=ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0),
            numerics=NumericsConfig(dt=1234.0, integrator="euler", output_freq=1),
            mesh=MeshConfig(type="fesomc", meshpath=str(nc_path)),
            data=DataConfig(ncinput=str(nc_path), inicloud=str(particles_path)),
            time=TimeConfig(n_steps=2),
        )

        with pytest.raises(ValueError, match="discrete mode"):
            run_tracking(config, verbose=False)

    def test_dt_matching_file_dt_no_warning(self, tmp_path, capsys):
        """Configured dt == file_dt in discrete mode → no warning."""
        nc_path = _make_fesomc_3x3_with_time(tmp_path, n_time=5, dt_seconds=1800.0)

        state = make_state(np.array([5.0]), np.array([5.0]))
        particles_path = tmp_path / "particles.dat"
        state.save(str(particles_path))

        config = AppConfig(
            mode=ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0),
            numerics=NumericsConfig(dt=1800.0, integrator="euler", output_freq=1),
            mesh=MeshConfig(type="fesomc", meshpath=str(nc_path)),
            data=DataConfig(ncinput=str(nc_path), inicloud=str(particles_path)),
            time=TimeConfig(n_steps=2),
        )

        run_tracking(config, verbose=False)
        captured = capsys.readouterr()
        assert "WARNING" not in captured.out

    def test_no_time_coord_requires_explicit_dt(self, tmp_path):
        """No time coordinate + dt=None → ValueError."""
        nc_path, mesh_cfg, _ = _make_fesomc_3x3(tmp_path, n_time=5)

        state = make_state(np.array([5.0]), np.array([5.0]))
        particles_path = tmp_path / "particles.dat"
        state.save(str(particles_path))

        config = AppConfig(
            mode=ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0),
            numerics=NumericsConfig(dt=None, integrator="euler", output_freq=1),
            mesh=MeshConfig(type="fesomc", meshpath=str(nc_path)),
            data=DataConfig(ncinput=str(nc_path), inicloud=str(particles_path)),
            time=TimeConfig(n_steps=2),
        )

        with pytest.raises(ValueError, match="Cannot auto-detect dt"):
            run_tracking(config, verbose=False)


# ===================================================================
# 3D Tracking tests
# ===================================================================

def _make_fesom2_3x3_3d(tmp_path, n_time=3, u_val=1.0, v_val=0.0, w_val=0.005):
    """Create FESOM2 ASCII mesh + 3D velocity NetCDF.

    Returns (mesh_dir, vel_path, mesh_cfg).
    """
    from types import SimpleNamespace

    mesh_dir = tmp_path / "f2mesh3d"
    mesh_dir.mkdir()

    xs = np.array([4.0, 5.0, 6.0])
    ys = np.array([4.0, 5.0, 6.0])
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel()  # 9 nodes
    lat = gy.ravel()

    with open(mesh_dir / "nod2d.out", "w") as f:
        f.write("9\n")
        for i in range(9):
            f.write(f"{i+1} {lon[i]:.6f} {lat[i]:.6f} 0\n")

    elems = []
    for j in range(2):
        for i in range(2):
            n0 = j * 3 + i
            elems.append([n0 + 1, n0 + 2, n0 + 4])
            elems.append([n0 + 2, n0 + 5, n0 + 4])
    with open(mesh_dir / "elem2d.out", "w") as f:
        f.write(f"{len(elems)}\n")
        for e in elems:
            f.write(f"{e[0]} {e[1]} {e[2]}\n")

    with open(mesh_dir / "depth.out", "w") as f:
        for _ in range(9):
            f.write("100.0\n")

    # Write z-levels file
    with open(mesh_dir / "aux3d.out", "w") as f:
        for z in [0.0, 10.0, 25.0, 50.0, 100.0]:
            f.write(f"{z}\n")

    n_elem = len(elems)
    n_node = 9
    n_z_levels = 5
    n_mid = n_z_levels - 1

    # 3D velocity: u(time, nz1, elem), v(time, nz1, elem), w(time, nz, nod2)
    u = np.full((n_time, n_mid, n_elem), u_val, dtype=np.float64)
    v = np.full((n_time, n_mid, n_elem), v_val, dtype=np.float64)
    w = np.full((n_time, n_z_levels, n_node), w_val, dtype=np.float64)
    ssh = np.zeros((n_time, n_node), dtype=np.float64)

    vel_path = tmp_path / "f2vel3d.nc"
    xr.Dataset({
        "u": (["time", "nz1", "elem"], u),
        "v": (["time", "nz1", "elem"], v),
        "w": (["time", "nz", "nod2"], w),
        "ssh": (["time", "node"], ssh),
    }).to_netcdf(vel_path)

    mesh_cfg = SimpleNamespace(meshpath=str(mesh_dir))
    return mesh_dir, vel_path, mesh_cfg


class TestTracking3D:
    def test_3d_tracking_z_changes(self, tmp_path):
        """3D tracking: particle depth should change with constant W.

        The FESOM2 file stores w positive UP: -0.01 is downwelling.
        """
        mesh_dir, vel_path, mesh_cfg = _make_fesom2_3x3_3d(
            tmp_path, u_val=0.0, v_val=0.0, w_val=-0.01
        )
        reader = get_reader("fesom2")
        mesh = reader.load_mesh(mesh_cfg)
        ds = reader.open_dataset(str(vel_path))

        mode = ModeConfig(Fdrift2D=False, Fdrift3D=True, time_dir=1.0)
        rec = track_particles(
            mesh, ds, reader,
            lon0=np.array([5.0]),
            lat0=np.array([5.0]),
            z0=np.array([20.0]),
            start_time_idx=0, n_steps=3, dt=100.0,
            mode=mode,
        )
        ds.close()

        arrs = rec.to_arrays()
        # Particle should have moved deeper: z_new ~ 20 + 0.01*100*3 = 20.3
        assert arrs["z"][-1, 0] > 20.0
        # Horizontal should be ~unchanged
        assert arrs["lon"][-1, 0] == pytest.approx(5.0, abs=0.01)

    def test_3d_tracking_combined(self, tmp_path):
        """3D tracking: particle moves both horizontally and vertically."""
        mesh_dir, vel_path, mesh_cfg = _make_fesom2_3x3_3d(
            tmp_path, u_val=1.0, v_val=0.5, w_val=-0.005   # positive-up file
        )
        reader = get_reader("fesom2")
        mesh = reader.load_mesh(mesh_cfg)
        ds = reader.open_dataset(str(vel_path))

        mode = ModeConfig(Fdrift2D=False, Fdrift3D=True, time_dir=1.0)
        rec = track_particles(
            mesh, ds, reader,
            lon0=np.array([5.0]),
            lat0=np.array([5.0]),
            z0=np.array([15.0]),
            start_time_idx=0, n_steps=3, dt=600.0,
            mode=mode,
        )
        ds.close()

        arrs = rec.to_arrays()
        assert arrs["lon"][-1, 0] > 5.0   # moved east
        assert arrs["lat"][-1, 0] > 5.0   # moved north
        assert arrs["z"][-1, 0] > 15.0    # moved deeper


# ===================================================================
# _resolve_start_date tests
# ===================================================================

def _make_ds_with_datetimes(n_time=5, dt_hours=6):
    """Create a minimal dataset with datetime64 time coordinate."""
    import pandas as pd
    times = pd.date_range("2020-06-01", periods=n_time, freq=f"{dt_hours}h")
    ds = xr.Dataset({"dummy": (["time"], np.zeros(n_time))}, coords={"time": times})
    file_times = np.arange(n_time, dtype=np.float64) * dt_hours * 3600.0
    return ds, file_times


class TestResolveStartDate:
    """Tests for _resolve_start_date."""

    def test_exact_match(self):
        """start_date matching a file snapshot returns that index."""
        ds, file_times = _make_ds_with_datetimes(n_time=5, dt_hours=6)
        # Third snapshot: 2020-06-01T12:00
        idx, sim_t = _resolve_start_date(ds, "2020-06-01T12:00", file_times, temporal_interp=False)
        assert idx == 2
        assert sim_t == file_times[2]

    def test_exact_match_first(self):
        ds, file_times = _make_ds_with_datetimes()
        idx, sim_t = _resolve_start_date(ds, "2020-06-01T00:00", file_times, temporal_interp=False)
        assert idx == 0
        assert sim_t == 0.0

    def test_no_exact_match_without_temporal_interp_raises(self):
        """Non-exact start_date without temporal_interp must raise."""
        ds, file_times = _make_ds_with_datetimes(n_time=5, dt_hours=6)
        with pytest.raises(ValueError, match="does not match any file timestep"):
            _resolve_start_date(ds, "2020-06-01T03:00", file_times, temporal_interp=False)

    def test_between_snapshots_with_temporal_interp(self):
        """Non-exact start_date with temporal_interp returns correct sim_time."""
        ds, file_times = _make_ds_with_datetimes(n_time=5, dt_hours=6)
        # 3 hours into a 6-hour gap → between index 0 and 1
        idx, sim_t = _resolve_start_date(ds, "2020-06-01T03:00", file_times, temporal_interp=True)
        assert idx == 0  # snapshot just before
        assert sim_t == pytest.approx(3 * 3600.0)

    def test_between_later_snapshots(self):
        """Between snapshots 2 and 3."""
        ds, file_times = _make_ds_with_datetimes(n_time=5, dt_hours=6)
        # 2020-06-01T15:00 = 15h from start, between idx=2 (12h) and idx=3 (18h)
        idx, sim_t = _resolve_start_date(ds, "2020-06-01T15:00", file_times, temporal_interp=True)
        assert idx == 2
        assert sim_t == pytest.approx(15 * 3600.0)

    def test_outside_range_raises(self):
        """start_date outside file time range raises."""
        ds, file_times = _make_ds_with_datetimes(n_time=5, dt_hours=6)
        with pytest.raises(ValueError, match="outside the file time range"):
            _resolve_start_date(ds, "2020-07-01T00:00", file_times, temporal_interp=True)

    def test_before_range_raises(self):
        ds, file_times = _make_ds_with_datetimes(n_time=5, dt_hours=6)
        with pytest.raises(ValueError, match="outside the file time range"):
            _resolve_start_date(ds, "2020-05-01T00:00", file_times, temporal_interp=True)

    def test_no_time_coord_raises(self):
        """Dataset without time dimension raises."""
        ds = xr.Dataset({"dummy": (["x"], np.zeros(3))})
        file_times = np.array([0.0, 3600.0])
        with pytest.raises(ValueError, match="interpretable time"):
            _resolve_start_date(ds, "2020-06-01", file_times, temporal_interp=True)


# ===================================================================
# _shift_to_midpoints tests
# ===================================================================

class TestShiftToMidpoints:
    """Tests for _shift_to_midpoints (time-averaging correction)."""

    def test_regular_spacing(self):
        """Evenly spaced records shift back by half the interval."""
        # 4 records, 10s apart: 0, 10, 20, 30
        ft = np.array([0.0, 10.0, 20.0, 30.0])
        shifted = _shift_to_midpoints(ft)
        # midpoints: [-5, 5, 15, 25]
        np.testing.assert_allclose(shifted, [-5.0, 5.0, 15.0, 25.0])

    def test_irregular_spacing(self):
        """Monthly-like irregular intervals: each midpoint correct."""
        # Simulate end-of-month timestamps in seconds from epoch
        # Jan=31d, Feb=28d, Mar=31d
        day = 86400.0
        ft = np.array([0.0, 31 * day, (31 + 28) * day, (31 + 28 + 31) * day])
        shifted = _shift_to_midpoints(ft)
        # [0]: extrapolated back by half of first interval (31d/2)
        assert shifted[0] == pytest.approx(-15.5 * day)
        # [1]: midpoint of [0, 31d] = 15.5d
        assert shifted[1] == pytest.approx(15.5 * day)
        # [2]: midpoint of [31d, 59d] = 45d
        assert shifted[2] == pytest.approx(45.0 * day)
        # [3]: midpoint of [59d, 90d] = 74.5d
        assert shifted[3] == pytest.approx(74.5 * day)

    def test_single_record(self):
        """Single record returns unchanged."""
        ft = np.array([100.0])
        shifted = _shift_to_midpoints(ft)
        assert shifted[0] == 100.0

    def test_two_records(self):
        ft = np.array([0.0, 86400.0])
        shifted = _shift_to_midpoints(ft)
        assert shifted[0] == pytest.approx(-43200.0)
        assert shifted[1] == pytest.approx(43200.0)


class TestTimeAveragingWarning:
    """Test that run_tracking warns when time_averaging is not set."""

    def test_warns_fesom2(self, tmp_path):
        """FESOM2 without time_averaging warns and suggests 'mean'."""
        import warnings
        # Minimal FESOM2 setup
        mesh_dir, vel_path, mesh_cfg = _make_fesom2_3x3(tmp_path, n_time=3)
        part_path = tmp_path / "particles.dat"
        part_path.write_text("5.0 5.0\n")

        cfg = AppConfig(
            mode=ModeConfig(Fdrift2D=True, time_dir=1.0),
            numerics=NumericsConfig(dt=3600.0, integrator="euler"),
            mesh=MeshConfig(type="fesom2", meshpath=str(mesh_dir)),
            data=DataConfig(ncinput=str(vel_path), inicloud=str(part_path)),
            time=TimeConfig(start_time_idx=0, n_steps=2),
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            run_tracking(cfg, verbose=False)
            avg_warnings = [x for x in w if "time_averaging" in str(x.message)]
            assert len(avg_warnings) == 1
            assert "mean" in str(avg_warnings[0].message)

    def test_warns_fesomc(self, tmp_path):
        """FESOM-C without time_averaging warns and suggests 'instant'."""
        import warnings
        nc_path, mesh_cfg, mode_cfg = _make_fesomc_3x3(tmp_path, n_time=3)
        part_path = tmp_path / "particles.dat"
        part_path.write_text("5.0 5.0\n")

        cfg = AppConfig(
            mode=ModeConfig(Fdrift2D=True, time_dir=1.0),
            numerics=NumericsConfig(dt=3600.0, integrator="euler"),
            mesh=MeshConfig(type="fesomc", meshpath=str(nc_path)),
            data=DataConfig(ncinput=str(nc_path), inicloud=str(part_path)),
            time=TimeConfig(start_time_idx=0, n_steps=2),
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            run_tracking(cfg, verbose=False)
            avg_warnings = [x for x in w if "time_averaging" in str(x.message)]
            assert len(avg_warnings) == 1
            assert "instant" in str(avg_warnings[0].message)


class TestVerbosePhaseBreakdown:
    """TODO 6.6: verbose per-step output carries an orchestration-level
    phase breakdown (io / step / rec, + beach when enabled) and a mean
    summary line.  Kernel-internal phases are not separable for the
    fused kernels — this is deliberately the tracking-layer view."""

    def test_phase_labels_in_verbose_output(self, tmp_path, capsys):
        nc_path = _make_fesomc_3x3_with_time(tmp_path, n_time=4,
                                             dt_seconds=3600.0,
                                             u_val=0.5, v_val=0.0)
        from kelp.readers import get_reader
        from kelp.tracking import run_single_realization
        from kelp.core.config import ModeConfig, NumericsConfig
        from kelp.core.particles import make_state
        from types import SimpleNamespace
        reader = get_reader("fesomc")
        mesh = reader.load_mesh(SimpleNamespace(meshpath=str(nc_path)))
        ds = reader.open_dataset(str(nc_path))
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)
        num = NumericsConfig(dt=1800.0, integrator="euler", output_freq=1)
        st = make_state(np.array([5.0]), np.array([5.0]))
        run_single_realization(mesh, ds, reader, st, mode, num, 0,
                               n_steps=3, n_time_total=4, verbose=True, temporal_interp=True)
        out = capsys.readouterr().out
        assert "(io=" in out and "step=" in out and "rec=" in out
        assert "Phases (mean ms):" in out
        assert "beach=" not in out.split("Phases")[0]   # dormant: no label


def test_end_time_idx_means_end_minus_start_steps_everywhere(tmp_path):
    """The legacy key gave end - start steps in run_tracking (via
    resolve_time_params) but end - start + 1 in run_single_realization and
    track_particles; one rule now: end - start (audit 2026-10-04)."""
    from kelp.core.config import TimeConfig, resolve_time_params
    from kelp.tracking import track_particles
    assert resolve_time_params(TimeConfig(start_time_idx=0, end_time_idx=2)) == (0, 2)
    nc_path, mesh_cfg, _ = _make_fesomc_3x3(tmp_path, u_val=1.0, v_val=0.0)
    reader = get_reader("fesomc")
    mesh = reader.load_mesh(mesh_cfg)
    ds = reader.open_dataset(str(nc_path))
    try:
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)
        numerics = NumericsConfig(dt=900.0, integrator="euler", output_freq=1)
        rec = run_single_realization(mesh, ds, reader,
                                     make_state(np.array([5.0]), np.array([5.0])),
                                     mode, numerics, 0, end_time_idx=2)
        assert rec.to_arrays()["lon"].shape[0] == 2
        rec2 = track_particles(mesh, ds, reader, np.array([5.0]), np.array([5.0]),
                               dt=900.0, end_time_idx=2)
        assert rec2.to_arrays()["lon"].shape[0] == 2
    finally:
        ds.close()


def test_particle_offset_none_means_an_unsharded_run(tmp_path):
    """load_restart returns None for a restart written without the offset;
    passing it on must work and mean 0 (review round 3, 2026-10-04)."""
    from kelp.tracking import track_particles
    nc_path, mesh_cfg, _ = _make_fesomc_3x3(tmp_path, u_val=1.0, v_val=0.0)
    reader = get_reader("fesomc")
    mesh = reader.load_mesh(mesh_cfg)
    ds = reader.open_dataset(str(nc_path))
    try:
        a = track_particles(mesh, ds, reader, np.array([5.0]), np.array([5.0]),
                            dt=900.0, n_steps=2, particle_offset=None)
        b = track_particles(mesh, ds, reader, np.array([5.0]), np.array([5.0]),
                            dt=900.0, n_steps=2, particle_offset=0)
        np.testing.assert_array_equal(a.to_arrays()["lon"], b.to_arrays()["lon"])
    finally:
        ds.close()
