"""Tests for kelp.numerics.integrators — Euler, RK2, RK4 time stepping.

Key physical tests:
  1. Constant eastward flow → particle moves linearly in longitude.
  2. Displacement magnitude matches analytical prediction.
  3. Backward tracking (time_dir=-1) reverses direction.
  4. RK2/RK4 agree with Euler for spatially constant velocity.
  5. Inactive particles do not move.
  6. Multiple steps accumulate correctly.
"""
from __future__ import annotations


import numpy as np
import pytest

import kelp.numerics.integrators as integrators_mod


from kelp.numerics.integrators import euler_step, rk2_step, rk4_step, get_integrator
from kelp.numerics.interpolate import VelocityField
from kelp.core.mesh import Mesh, build_node_to_elems, build_search, find_nearest_nodes
from kelp.core.particles import make_state, ParticleStatus
from kelp.core.geometry import METERS_PER_DEG_LAT, DEG2RAD


# ---------------------------------------------------------------------------
# Fixture: 3×3 grid with 8 triangles (same as test_interpolate)
# ---------------------------------------------------------------------------

@pytest.fixture
def grid3x3():
    """3×3 node grid with 8 triangles centred around (5°, 5°)."""
    xs = np.array([4.0, 5.0, 6.0])
    ys = np.array([4.0, 5.0, 6.0])
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel().astype(np.float64)
    lat = gy.ravel().astype(np.float64)
    depth = np.zeros(9, dtype=np.float64)

    elems = []
    for j in range(2):
        for i in range(2):
            n0 = j * 3 + i
            n1 = n0 + 1
            n2 = n0 + 3
            n3 = n0 + 4
            elems.append([n0, n1, n2])
            elems.append([n1, n3, n2])
    elem_nodes = np.array(elems, dtype=np.int32)

    lon_c = np.array([lon[e].mean() for e in elem_nodes], dtype=np.float64)
    lat_c = np.array([lat[e].mean() for e in elem_nodes], dtype=np.float64)

    n2e, n2e_count = build_node_to_elems(9, elem_nodes)
    mesh = Mesh(
        name="test3x3",
        nodes_lon=lon, nodes_lat=lat, depth=depth,
        elem_nodes=elem_nodes,
        elem_center_lon=lon_c, elem_center_lat=lat_c,
        node_to_elems=n2e, node_to_elems_count=n2e_count,
    )
    build_search(mesh)
    return mesh


def _const_vel(mesh, u=1.0, v=0.0):
    """Uniform velocity field on all elements."""
    ne = mesh.n_elements
    return VelocityField(
        u=np.full(ne, u, dtype=np.float64),
        v=np.full(ne, v, dtype=np.float64),
        eta=np.zeros(mesh.n_nodes, dtype=np.float64),
    )


# ===================================================================
# Euler tests
# ===================================================================

class TestEuler:
    def test_eastward_flow_moves_east(self, grid3x3):
        """Particle at (5°,5°) with U=1 m/s should move eastward."""
        vel = _const_vel(grid3x3, u=1.0, v=0.0)
        state = make_state(lon=np.array([5.0]), lat=np.array([5.0]))
        new = euler_step(state, vel, grid3x3, dt=900.0, time_dir=1.0)
        assert new.lon[0] > 5.0
        assert new.lat[0] == pytest.approx(5.0, abs=0.001)
        assert new.status[0] == ParticleStatus.ALIVE

    def test_northward_flow_moves_north(self, grid3x3):
        """V=1 m/s → particle moves northward."""
        vel = _const_vel(grid3x3, u=0.0, v=1.0)
        state = make_state(lon=np.array([5.0]), lat=np.array([5.0]))
        new = euler_step(state, vel, grid3x3, dt=900.0, time_dir=1.0)
        assert new.lon[0] == pytest.approx(5.0, abs=0.001)
        assert new.lat[0] > 5.0

    def test_displacement_magnitude(self, grid3x3):
        """Check Euler displacement matches analytical prediction.

        For U=1 m/s, dt=3600s at lat=5°:
            Δlon = U·dt / (cos(lat)·METERS_PER_DEG_LAT) ≈ 3600 / (cos(5°)·111120)
        """
        vel = _const_vel(grid3x3, u=1.0, v=0.0)
        state = make_state(lon=np.array([5.0]), lat=np.array([5.0]))
        dt = 3600.0
        new = euler_step(state, vel, grid3x3, dt=dt, time_dir=1.0)

        z0 = np.cos(5.0 * DEG2RAD) * METERS_PER_DEG_LAT
        expected_dlon = dt / z0
        actual_dlon = new.lon[0] - 5.0
        assert actual_dlon == pytest.approx(expected_dlon, rel=0.05)

    def test_lat_displacement_magnitude(self, grid3x3):
        """Check latitude displacement: Δlat = V·dt / METERS_PER_DEG_LAT."""
        vel = _const_vel(grid3x3, u=0.0, v=1.0)
        state = make_state(lon=np.array([5.0]), lat=np.array([5.0]))
        dt = 3600.0
        new = euler_step(state, vel, grid3x3, dt=dt, time_dir=1.0)

        expected_dlat = dt / METERS_PER_DEG_LAT
        actual_dlat = new.lat[0] - 5.0
        assert actual_dlat == pytest.approx(expected_dlat, rel=0.05)

    def test_backward_tracking(self, grid3x3):
        """time_dir=-1 should move particle westward when U>0."""
        vel = _const_vel(grid3x3, u=1.0, v=0.0)
        state_fwd = make_state(lon=np.array([5.0]), lat=np.array([5.0]))
        state_bwd = make_state(lon=np.array([5.0]), lat=np.array([5.0]))

        new_fwd = euler_step(state_fwd, vel, grid3x3, dt=900.0, time_dir=1.0)
        new_bwd = euler_step(state_bwd, vel, grid3x3, dt=900.0, time_dir=-1.0)

        assert new_fwd.lon[0] > 5.0   # forward → east
        assert new_bwd.lon[0] < 5.0   # backward → west

    def test_inactive_particle_unchanged(self, grid3x3):
        """Particle with status OUTSIDE should not move."""
        vel = _const_vel(grid3x3, u=10.0, v=10.0)
        state = make_state(lon=np.array([5.0]), lat=np.array([5.0]))
        state.status[0] = ParticleStatus.OUTSIDE

        new = euler_step(state, vel, grid3x3, dt=3600.0, time_dir=1.0)
        assert new.lon[0] == pytest.approx(5.0, abs=1e-10)
        assert new.lat[0] == pytest.approx(5.0, abs=1e-10)
        assert new.status[0] == ParticleStatus.OUTSIDE

    def test_returns_new_state(self, grid3x3):
        """Euler should return a new ParticleState, not mutate the original."""
        vel = _const_vel(grid3x3, u=1.0, v=0.0)
        state = make_state(lon=np.array([5.0]), lat=np.array([5.0]))
        old_lon = state.lon[0]
        new = euler_step(state, vel, grid3x3, dt=900.0, time_dir=1.0)
        # Original should be unchanged
        assert state.lon[0] == pytest.approx(old_lon, abs=1e-14)
        # New should be different
        assert new.lon[0] != old_lon


# ===================================================================
# RK2 tests
# ===================================================================

class TestRK2:
    def test_constant_flow_matches_euler(self, grid3x3):
        """For constant flow, RK2 and Euler should give nearly identical results."""
        vel = _const_vel(grid3x3, u=1.0, v=0.0)
        state_e = make_state(lon=np.array([5.0]), lat=np.array([5.0]))
        state_r = make_state(lon=np.array([5.0]), lat=np.array([5.0]))

        new_e = euler_step(state_e, vel, grid3x3, dt=900.0, time_dir=1.0)
        new_r = rk2_step(state_r, vel, grid3x3, dt=900.0, time_dir=1.0)

        assert new_r.lon[0] == pytest.approx(new_e.lon[0], abs=0.001)
        assert new_r.lat[0] == pytest.approx(new_e.lat[0], abs=0.001)

    def test_moves_southward(self, grid3x3):
        """V=-1 → particle moves south."""
        vel = _const_vel(grid3x3, u=0.0, v=-1.0)
        state = make_state(lon=np.array([5.0]), lat=np.array([5.0]))
        new = rk2_step(state, vel, grid3x3, dt=900.0, time_dir=1.0)
        assert new.lat[0] < 5.0

    def test_particle_stays_alive(self, grid3x3):
        vel = _const_vel(grid3x3, u=0.5, v=0.5)
        state = make_state(lon=np.array([5.0]), lat=np.array([5.0]))
        new = rk2_step(state, vel, grid3x3, dt=600.0, time_dir=1.0)
        assert new.status[0] == ParticleStatus.ALIVE


# ===================================================================
# RK4 tests
# ===================================================================

class TestRK4:
    def test_constant_flow(self, grid3x3):
        """RK4 with constant flow should move particle in correct direction."""
        vel = _const_vel(grid3x3, u=1.0, v=0.5)
        state = make_state(lon=np.array([5.0]), lat=np.array([5.0]))
        new = rk4_step(state, vel, grid3x3, dt=900.0, time_dir=1.0)
        assert new.lon[0] > 5.0
        assert new.lat[0] > 5.0
        assert new.status[0] == ParticleStatus.ALIVE

    def test_matches_euler_for_constant(self, grid3x3):
        """For constant velocity, RK4 ≈ Euler (within small tolerance)."""
        vel = _const_vel(grid3x3, u=2.0, v=-0.5)
        state_e = make_state(lon=np.array([5.0]), lat=np.array([5.0]))
        state_r = make_state(lon=np.array([5.0]), lat=np.array([5.0]))

        new_e = euler_step(state_e, vel, grid3x3, dt=600.0, time_dir=1.0)
        new_r = rk4_step(state_r, vel, grid3x3, dt=600.0, time_dir=1.0)

        assert new_r.lon[0] == pytest.approx(new_e.lon[0], abs=0.005)
        assert new_r.lat[0] == pytest.approx(new_e.lat[0], abs=0.005)

    def test_backward(self, grid3x3):
        """RK4 backward should move in opposite direction."""
        vel = _const_vel(grid3x3, u=1.0, v=0.0)
        state = make_state(lon=np.array([5.0]), lat=np.array([5.0]))
        new = rk4_step(state, vel, grid3x3, dt=900.0, time_dir=-1.0)
        assert new.lon[0] < 5.0


# ===================================================================
# Multi-step and multi-particle tests
# ===================================================================

class TestMultiStep:
    def test_10_euler_steps_accumulate(self, grid3x3):
        """10 Euler steps with constant flow → cumulative displacement."""
        vel = _const_vel(grid3x3, u=1.0, v=0.0)
        state = make_state(lon=np.array([5.0]), lat=np.array([5.0]))
        dt = 360.0

        for _ in range(10):
            state = euler_step(state, vel, grid3x3, dt=dt, time_dir=1.0)

        total_time = 10 * dt
        z0 = np.cos(5.0 * DEG2RAD) * METERS_PER_DEG_LAT
        expected_dlon = total_time / z0
        actual_dlon = state.lon[0] - 5.0
        assert actual_dlon == pytest.approx(expected_dlon, rel=0.1)

    def test_two_particles_independent(self, grid3x3):
        """Two particles at different positions should track independently."""
        vel = _const_vel(grid3x3, u=1.0, v=0.0)
        state = make_state(
            lon=np.array([4.8, 5.2]),
            lat=np.array([5.0, 5.0]),
        )
        new = euler_step(state, vel, grid3x3, dt=900.0, time_dir=1.0)

        # Both should move east
        assert new.lon[0] > 4.8
        assert new.lon[1] > 5.2
        # Both still alive
        assert np.all(new.status == ParticleStatus.ALIVE)

    def test_one_alive_one_dead(self, grid3x3):
        """Only the alive particle should move."""
        vel = _const_vel(grid3x3, u=1.0, v=0.0)
        state = make_state(
            lon=np.array([5.0, 5.0]),
            lat=np.array([5.0, 5.0]),
        )
        state.status[1] = ParticleStatus.BEACHED

        new = euler_step(state, vel, grid3x3, dt=900.0, time_dir=1.0)
        assert new.lon[0] > 5.0      # alive → moved
        assert new.lon[1] == pytest.approx(5.0, abs=1e-10)  # beached → unchanged
        assert new.status[0] == ParticleStatus.ALIVE
        assert new.status[1] == ParticleStatus.BEACHED

    def test_diagonal_flow(self, grid3x3):
        """NE flow: both lon and lat should increase."""
        vel = _const_vel(grid3x3, u=1.0, v=1.0)
        state = make_state(lon=np.array([5.0]), lat=np.array([5.0]))
        new = euler_step(state, vel, grid3x3, dt=900.0, time_dir=1.0)
        assert new.lon[0] > 5.0
        assert new.lat[0] > 5.0


# ===================================================================
# Registry tests
# ===================================================================

class TestRegistry:
    def test_known_integrators(self):
        assert get_integrator("euler") is euler_step
        assert get_integrator("rk2") is rk2_step
        assert get_integrator("rk4") is rk4_step

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown integrator"):
            get_integrator("leapfrog")


# ===================================================================
# 3D tracking tests
# ===================================================================

@pytest.fixture
def grid3x3_3d():
    """3×3 node grid with 8 triangles and z-levels for 3D testing."""
    xs = np.array([4.0, 5.0, 6.0])
    ys = np.array([4.0, 5.0, 6.0])
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel().astype(np.float64)
    lat = gy.ravel().astype(np.float64)
    depth = np.full(9, 100.0, dtype=np.float64)  # all nodes 100m deep

    elems = []
    for j in range(2):
        for i in range(2):
            n0 = j * 3 + i
            n1 = n0 + 1
            n2 = n0 + 3
            n3 = n0 + 4
            elems.append([n0, n1, n2])
            elems.append([n1, n3, n2])
    elem_nodes = np.array(elems, dtype=np.int32)

    lon_c = np.array([lon[e].mean() for e in elem_nodes], dtype=np.float64)
    lat_c = np.array([lat[e].mean() for e in elem_nodes], dtype=np.float64)

    n2e, n2e_count = build_node_to_elems(9, elem_nodes)
    # z-levels: 0, 10, 25, 50, 100m
    sigma = np.array([0.0, 10.0, 25.0, 50.0, 100.0], dtype=np.float64)
    mesh = Mesh(
        name="test3x3_3d",
        nodes_lon=lon, nodes_lat=lat, depth=depth,
        elem_nodes=elem_nodes,
        elem_center_lon=lon_c, elem_center_lat=lat_c,
        node_to_elems=n2e, node_to_elems_count=n2e_count,
        sigma=sigma,
    )
    build_search(mesh)
    return mesh


def _const_vel_3d(mesh, u=1.0, v=0.0, w_val=0.0):
    """Uniform 3D velocity field.

    u/v: shape (n_layers, n_elements) — element-based, same value all layers.
    w: shape (n_z_levels, n_nodes) — node-based, same value all levels.
    """
    ne = mesh.n_elements
    nn = mesh.n_nodes
    n_z = mesh.sigma.shape[0]
    n_layers = n_z - 1

    u_3d = np.full((n_layers, ne), u, dtype=np.float64)
    v_3d = np.full((n_layers, ne), v, dtype=np.float64)
    w_3d = np.full((n_z, nn), w_val, dtype=np.float64)

    return VelocityField(
        u=u_3d, v=v_3d,
        eta=np.zeros(nn, dtype=np.float64),
        w=w_3d,
    )


class TestEuler3D:
    def test_vertical_displacement_downward(self, grid3x3_3d):
        """Constant downward W moves particle deeper."""
        vel = _const_vel_3d(grid3x3_3d, u=0.0, v=0.0, w_val=0.01)
        state = make_state(
            lon=np.array([5.0]), lat=np.array([5.0]), z=np.array([20.0])
        )
        new = euler_step(state, vel, grid3x3_3d, dt=100.0, time_dir=1.0)
        # z_new = 20 + 0.01 * 100 = 21.0
        assert new.z[0] == pytest.approx(21.0, abs=0.1)

    def test_vertical_displacement_upward(self, grid3x3_3d):
        """Negative W moves particle shallower."""
        vel = _const_vel_3d(grid3x3_3d, u=0.0, v=0.0, w_val=-0.02)
        state = make_state(
            lon=np.array([5.0]), lat=np.array([5.0]), z=np.array([30.0])
        )
        new = euler_step(state, vel, grid3x3_3d, dt=100.0, time_dir=1.0)
        # z_new = 30 + (-0.02) * 100 = 28.0
        assert new.z[0] == pytest.approx(28.0, abs=0.1)

    def test_2d_mode_z_unchanged(self, grid3x3):
        """2D velocity (w=None) leaves z unchanged."""
        vel = _const_vel(grid3x3, u=1.0, v=0.0)
        state = make_state(
            lon=np.array([5.0]), lat=np.array([5.0]), z=np.array([10.0])
        )
        new = euler_step(state, vel, grid3x3, dt=900.0, time_dir=1.0)
        assert new.z[0] == pytest.approx(10.0)

    def test_surface_clamp(self, grid3x3_3d):
        """Particle near surface with strong upward W doesn't go negative."""
        vel = _const_vel_3d(grid3x3_3d, u=0.0, v=0.0, w_val=-1.0)
        state = make_state(
            lon=np.array([5.0]), lat=np.array([5.0]), z=np.array([2.0])
        )
        new = euler_step(state, vel, grid3x3_3d, dt=100.0, time_dir=1.0)
        assert new.z[0] >= 0.0  # clamped to surface

    def test_bottom_clamp(self, grid3x3_3d):
        """Particle near bottom with downward W doesn't exceed bottom depth."""
        vel = _const_vel_3d(grid3x3_3d, u=0.0, v=0.0, w_val=1.0)
        state = make_state(
            lon=np.array([5.0]), lat=np.array([5.0]), z=np.array([95.0])
        )
        new = euler_step(state, vel, grid3x3_3d, dt=100.0, time_dir=1.0)
        assert new.z[0] <= 100.0  # clamped to bottom (node depth=100)

    def test_combined_horizontal_and_vertical(self, grid3x3_3d):
        """3D: particle moves both horizontally and vertically."""
        vel = _const_vel_3d(grid3x3_3d, u=1.0, v=0.5, w_val=0.005)
        state = make_state(
            lon=np.array([5.0]), lat=np.array([5.0]), z=np.array([30.0])
        )
        new = euler_step(state, vel, grid3x3_3d, dt=100.0, time_dir=1.0)
        assert new.lon[0] > 5.0   # moved east
        assert new.lat[0] > 5.0   # moved north
        assert new.z[0] == pytest.approx(30.5, abs=0.1)  # moved deeper

    def test_sigma_mesh_moves_particle_with_nearest_w(self, grid3x3_3d):
        """Sigma meshes run through the unfused 3D Euler path and
        reproduce a constant downward W just like the z-level case."""
        # Re-cast the fixture as a sigma mesh (FESOM-C-style sigma_lev,
        # bottom=100 m given the fixture's uniform depth).
        grid3x3_3d.vertical_coord = "sigma"
        grid3x3_3d.sigma = np.array(
            [1.0, 0.8, 0.6, 0.4, 0.2, 0.0], dtype=np.float64
        )
        vel = _const_vel_3d(grid3x3_3d, u=0.0, v=0.0, w_val=0.01)

        state = make_state(
            lon=np.array([5.0]), lat=np.array([5.0]), z=np.array([20.0])
        )
        new = euler_step(
            state, vel, grid3x3_3d, dt=100.0, time_dir=1.0,
            w_interp="nearest",
        )

        # z_new = 20 + 0.01 * 100 = 21.0, same as z-level equivalent
        assert new.z[0] == pytest.approx(21.0, abs=0.1)
        assert new.status[0] == ParticleStatus.ALIVE

    @pytest.mark.parametrize("w_interp", ["nearest", "leastsq",
                                             "barycentric", "idw"])
    def test_sigma_mesh_supports_all_w_methods(self, grid3x3_3d, w_interp):
        """All four W interpolation methods must work on sigma meshes."""
        grid3x3_3d.vertical_coord = "sigma"
        grid3x3_3d.sigma = np.array(
            [1.0, 0.8, 0.6, 0.4, 0.2, 0.0], dtype=np.float64
        )
        vel = _const_vel_3d(grid3x3_3d, u=0.0, v=0.0, w_val=0.01)
        state = make_state(
            lon=np.array([5.0]), lat=np.array([5.0]), z=np.array([20.0])
        )
        new = euler_step(
            state, vel, grid3x3_3d, dt=100.0, time_dir=1.0,
            w_interp=w_interp,
        )
        # Uniform W → every method should return 0.01 exactly, so
        # z_new = 20 + 0.01 * 100 = 21.0.
        assert new.status[0] == ParticleStatus.ALIVE, (
            f"sigma {w_interp}: particle died"
        )
        assert new.z[0] == pytest.approx(21.0, abs=1e-9), (
            f"sigma {w_interp}: unexpected vertical displacement "
            f"{new.z[0]} (expected 21.0)"
        )

    def test_sigma_mesh_unknown_w_method_raises(self, grid3x3_3d):
        """Unknown method names should raise a clear ValueError."""
        grid3x3_3d.vertical_coord = "sigma"
        grid3x3_3d.sigma = np.array(
            [1.0, 0.8, 0.6, 0.4, 0.2, 0.0], dtype=np.float64
        )
        vel = _const_vel_3d(grid3x3_3d, u=0.0, v=0.0, w_val=0.01)
        state = make_state(
            lon=np.array([5.0]), lat=np.array([5.0]), z=np.array([20.0])
        )
        with pytest.raises(ValueError, match="Unknown sigma w_interp"):
            euler_step(
                state, vel, grid3x3_3d, dt=100.0, time_dir=1.0,
                w_interp="bogus",
            )

    @pytest.mark.parametrize(
        "integrator,expected_z",
        [(euler_step, 21.0),
         (rk2_step, 21.0),
         (rk4_step, 21.0)],
    )
    def test_sigma_mesh_works_with_all_integrators(
        self, grid3x3_3d, integrator, expected_z,
    ):
        """All three integrators must drive a sigma 3D particle correctly.

        Sigma RK2/RK4 transitively use _compute_layers/_compute_w/_clamp_z
        via the unfused path — this test makes that coverage explicit.
        """
        grid3x3_3d.vertical_coord = "sigma"
        grid3x3_3d.sigma = np.array(
            [1.0, 0.8, 0.6, 0.4, 0.2, 0.0], dtype=np.float64
        )
        vel = _const_vel_3d(grid3x3_3d, u=0.0, v=0.0, w_val=0.01)
        state = make_state(
            lon=np.array([5.0]), lat=np.array([5.0]), z=np.array([20.0])
        )
        new = integrator(
            state, vel, grid3x3_3d, dt=100.0, time_dir=1.0,
            w_interp="leastsq",
        )
        assert new.status[0] == ParticleStatus.ALIVE, (
            f"sigma {integrator.__name__}: particle died"
        )
        assert new.z[0] == pytest.approx(expected_z, abs=1e-9), (
            f"sigma {integrator.__name__}: z={new.z[0]} "
            f"(expected {expected_z})"
        )

    def test_unfused_3d_helpers_receive_vertical_kwargs(self, grid3x3_3d, monkeypatch):
        """Phase-0 plumbing: Dmin/z0b_min reach the 3D helper layer."""
        vel = _const_vel_3d(grid3x3_3d, u=0.0, v=0.0, w_val=0.0)
        state = make_state(
            lon=np.array([5.0]), lat=np.array([5.0]), z=np.array([20.0])
        )
        seen = {}

        def fake_compute_layers(mesh, vel_stage, z, nn_full, alive, *, Dmin, z0b_min,
                                H_col=None):
            seen["layers"] = (vel_stage is vel, Dmin, z0b_min)
            return np.zeros(z.shape[0], dtype=np.int32)

        def fake_compute_w(vel_stage, mesh, z, nn_full, alive, time_dir, **kwargs):
            seen["w"] = (vel_stage is vel, kwargs["Dmin"], kwargs["z0b_min"])
            return np.zeros(z.shape[0], dtype=np.float64)

        def fake_clamp_z(z, vel_stage, mesh, nn_full, alive, *, Dmin, z0b_min,
                         H_col=None):
            seen["clamp"] = (vel_stage is vel, Dmin, z0b_min)
            return z

        monkeypatch.setattr(integrators_mod, "_compute_layers", fake_compute_layers)
        monkeypatch.setattr(integrators_mod, "_compute_w", fake_compute_w)
        monkeypatch.setattr(integrators_mod, "_clamp_z", fake_clamp_z)

        _ = integrators_mod._euler_step_unfused(
            state, vel, grid3x3_3d, dt=100.0, time_dir=1.0,
            layer=0, is3d=True, w_interp="nearest",
            Dmin=1.25, z0b_min=0.05,
        )

        assert seen["layers"] == (True, 1.25, 0.05)
        assert seen["w"] == (True, 1.25, 0.05)
        assert seen["clamp"] == (True, 1.25, 0.05)


class TestRK2_3D:
    def test_constant_w_matches_euler(self, grid3x3_3d):
        """For constant W, RK2 should give same result as Euler."""
        vel = _const_vel_3d(grid3x3_3d, u=1.0, v=0.0, w_val=0.01)
        state = make_state(
            lon=np.array([5.0]), lat=np.array([5.0]), z=np.array([20.0])
        )
        new_euler = euler_step(state, vel, grid3x3_3d, dt=100.0, time_dir=1.0)
        new_rk2 = rk2_step(state, vel, grid3x3_3d, dt=100.0, time_dir=1.0)
        assert new_rk2.z[0] == pytest.approx(new_euler.z[0], abs=0.5)
        assert new_rk2.lon[0] == pytest.approx(new_euler.lon[0], abs=1e-4)


class TestRK4_3D:
    def test_constant_w_matches_euler(self, grid3x3_3d):
        """For constant W, RK4 should give same result as Euler."""
        vel = _const_vel_3d(grid3x3_3d, u=1.0, v=0.0, w_val=0.01)
        state = make_state(
            lon=np.array([5.0]), lat=np.array([5.0]), z=np.array([20.0])
        )
        new_euler = euler_step(state, vel, grid3x3_3d, dt=100.0, time_dir=1.0)
        new_rk4 = rk4_step(state, vel, grid3x3_3d, dt=100.0, time_dir=1.0)
        assert new_rk4.z[0] == pytest.approx(new_euler.z[0], abs=0.5)
        assert new_rk4.lon[0] == pytest.approx(new_euler.lon[0], abs=1e-4)


class TestStageInterpParity:
    """The parallel stage interpolation used by RK2/RK4 stage 2+
    (``_interp_uv_at``) must be BIT-FOR-BIT identical to the original
    per-particle serial loop over the single-point interpolator — that is the
    invariant the 3D RK2/RK4 parallelisation relies on.  Checked for every
    uv_interp method and velocity layout, forward and backward, in 2D and 3D,
    with a dead-masked particle and a point outside the mesh (the ok=False
    path)."""

    @staticmethod
    def _vary_vel(mesh, layout, ndim):
        """Spatially-varying velocity (constant fields would pass trivially)."""
        N, E = mesh.n_nodes, mesh.n_elements
        n_layers = mesh.sigma.shape[0] - 1
        if layout == "node":
            base = 0.3 + 0.5 * mesh.nodes_lon - 0.2 * mesh.nodes_lat      # (N,)
            kw = dict(uv_layout="node")
        else:
            base = (0.3 + 0.5 * mesh.elem_center_lon
                    - 0.2 * mesh.elem_center_lat)                         # (E,)
            kw = dict()
        if ndim == 3:
            u = np.vstack([base * (1.0 + 0.1 * L) for L in range(n_layers)])
            w = np.zeros((mesh.sigma.shape[0], N), dtype=np.float64)
        else:
            u = base.copy()
            w = None
        return VelocityField(u=u, v=2.0 * u,
                             eta=np.zeros(N, dtype=np.float64), w=w, **kw)

    @pytest.mark.parametrize("uv_interp,layout", [
        ("leastsq", "element"),
        ("barycentric", "element"),
        ("barycentric", "node"),     # the production production path (node unod/vnod)
    ])
    @pytest.mark.parametrize("ndim", [2, 3])
    @pytest.mark.parametrize("time_dir", [1.0, -1.0])
    def test_matches_serial_single_loop(self, grid3x3_3d, uv_interp, layout,
                                        ndim, time_dir):
        from kelp.numerics.integrators import _interp_uv_at
        from kelp.numerics.interpolate import (
            get_interpolator, get_interpolator_single)

        mesh = grid3x3_3d
        vel = self._vary_vel(mesh, layout, ndim)

        # Stage positions: interior blends (not nodes); index 2 masked off;
        # index 4 outside the mesh (containment fails → ok=False both paths).
        lons = np.array([4.6, 5.3, 4.9, 5.1, 20.0])
        lats = np.array([4.7, 5.2, 5.4, 4.8, 20.0])
        mask = np.array([True, True, False, True, True])
        P = lons.shape[0]
        layer_arg = (np.array([0, 2, 1, 3, 0], dtype=np.int32)
                     if ndim == 3 else 0)

        nn = find_nearest_nodes(mesh, lons, lats)

        # Reference: the original serial per-particle loop.
        single = get_interpolator_single(uv_interp)
        U_ref = np.zeros(P); V_ref = np.zeros(P)
        ok_ref = np.zeros(P, dtype=bool)
        for i in np.where(mask)[0]:
            li = (int(layer_arg[i]) if isinstance(layer_arg, np.ndarray)
                  else layer_arg)
            u, v, s = single(lons[i], lats[i], li, mesh, vel,
                             int(nn[i]), time_dir)
            U_ref[i], V_ref[i], ok_ref[i] = u, v, s

        # Parallel helper — what RK2/RK4 stage 2+ now call.
        base = make_state(lons.copy(), lats.copy())
        U, V, ok = _interp_uv_at(get_interpolator(uv_interp), base, mesh, vel,
                                 lons, lats, mask, nn, time_dir, layer_arg)

        np.testing.assert_array_equal(U, U_ref)
        np.testing.assert_array_equal(V, V_ref)
        np.testing.assert_array_equal(ok, ok_ref)
        # Meaningful: interior masked-True particles actually interpolated.
        assert ok_ref[[0, 1, 3]].all()
        # The masked-off particle stays zero/False on both paths.
        assert not ok[2] and U[2] == 0.0 and V[2] == 0.0
        # The outside particle fails identically.
        assert not ok[4]


class TestFullStep3DNodeBarycentricGolden:
    """Full-step regression for the parallelised unfused 3D path (the production
    production case: node-layout unod/vnod + barycentric + RK in 3D).

    Pins the WHOLE step — stage interpolation -> displacement -> W -> depth
    clamp -> stage-failure revert — against future drift, for both RK2 and
    RK4, with particles spanning vertical layers (incl. a bottom-clamp) and
    one that exits the mesh mid-step (status -> OUTSIDE, position reverted).
    Golden values were captured after the adversarial differential review
    confirmed this path is bit-identical to the pre-parallelisation code.
    (Complements TestStageInterpParity, which pins the stage interp itself.)"""

    @staticmethod
    def _setup(mesh):
        N = mesh.n_nodes
        nL = mesh.sigma.shape[0] - 1
        base = 0.3 + 0.5 * mesh.nodes_lon - 0.2 * mesh.nodes_lat
        u = np.vstack([base * (1.0 + 0.1 * L) for L in range(nL)])
        w = np.full((mesh.sigma.shape[0], N), 2e-3)
        vel = VelocityField(u=u, v=0.5 * u, eta=np.zeros(N), w=w,
                            uv_layout="node")
        st = make_state(np.array([4.7, 5.0, 5.3, 5.95]),
                        np.array([4.8, 5.2, 5.1, 5.5]),
                        np.array([5.0, 20.0, 60.0, 30.0]))
        return vel, st

    @pytest.mark.parametrize("name,exp_lon,exp_lat,exp_z,exp_status", [
        ("rk2",
         [5.079581676690793, 5.396948107874465, 5.774785024204817, 5.95],
         [4.989103970684939, 5.397629635951441, 5.336410650517136, 5.5],
         [45.0, 59.99999999999999, 100.0, 30.0],
         [1, 1, 1, 3]),
        ("rk4",
         [5.071612481408664, 5.398829914310824, 5.77547732372318, 5.95],
         [4.985129154155817, 5.398562519258694, 5.3367531279072, 5.5],
         [45.0, 59.99999999999999, 100.0, 30.0],
         [1, 1, 1, 3]),
    ])
    def test_golden(self, grid3x3_3d, name, exp_lon, exp_lat, exp_z, exp_status):
        step = rk2_step if name == "rk2" else rk4_step
        vel, st = self._setup(grid3x3_3d)
        out = step(st, vel, grid3x3_3d, dt=20000.0, time_dir=1.0,
                   uv_interp="barycentric", w_interp="leastsq")
        np.testing.assert_allclose(out.lon, exp_lon, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(out.lat, exp_lat, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(out.z, exp_z, rtol=1e-12, atol=1e-9)
        assert out.status.tolist() == exp_status
        # the edge particle exited and was reverted to its origin
        assert out.status[3] == ParticleStatus.OUTSIDE
        assert out.lon[3] == 5.95 and out.lat[3] == 5.5 and out.z[3] == 30.0
