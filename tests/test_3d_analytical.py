"""Analytical verification tests for 3D tracking.

Creates small FESOM-like meshes at various geographic locations
(equator, prime meridian, dateline, north pole) with known constant
velocities, and verifies particle positions match analytical predictions.

All meshes are local grids — the algorithm does not require a global mesh.

Plotting
--------
Set environment variable ``PLOT_3D=1`` to generate diagnostic plots
during manual test runs::

    PLOT_3D=1 python -m pytest tests/test_3d_analytical.py -v -s

Plots are saved to ``tests/plots_3d/`` (created automatically).
"""
from __future__ import annotations

import os
import numpy as np
import pytest

from kelp.numerics.integrators import euler_step, rk2_step, rk4_step
from kelp.numerics.interpolate import VelocityField
from kelp.core.mesh import Mesh, build_node_to_elems, build_search
from kelp.core.particles import make_state, ParticleStatus
from kelp.core.geometry import METERS_PER_DEG_LAT, DEG2RAD

# Plotting is off by default.  Set PLOT_3D=1 env var to enable.
PLOT_3D = os.environ.get("PLOT_3D", "0") == "1"
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots_3d")


# ---------------------------------------------------------------------------
# Reusable mesh builder
# ---------------------------------------------------------------------------

def _make_mesh(center_lon: float, center_lat: float,
               span: float = 1.0, nx: int = 3, ny: int = 3,
               z_levels=None, bottom_depth: float = 200.0):
    """Build a local triangular mesh centred at (center_lon, center_lat).

    Parameters
    ----------
    center_lon, center_lat : float
        Centre of the mesh in degrees.
    span : float
        Half-extent in degrees (mesh covers ±span in lon and lat).
    nx, ny : int
        Number of nodes in lon and lat directions.
    z_levels : array-like or None
        Z-level boundaries (positive-down, meters).  If None, uses
        [0, 10, 25, 50, 100, 200].
    bottom_depth : float
        Uniform bottom depth for all nodes (positive-down, meters).

    Returns
    -------
    Mesh
        Ready-to-use mesh with KD-tree built.
    """
    if z_levels is None:
        z_levels = [0.0, 10.0, 25.0, 50.0, 100.0, 200.0]

    xs = np.linspace(center_lon - span, center_lon + span, nx)
    ys = np.linspace(center_lat - span, center_lat + span, ny)
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel().astype(np.float64)
    lat = gy.ravel().astype(np.float64)
    n_nodes = lon.shape[0]
    depth = np.full(n_nodes, bottom_depth, dtype=np.float64)

    # Build triangulation: 2 triangles per grid cell
    elems = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            n1 = n0 + 1
            n2 = n0 + nx
            n3 = n0 + nx + 1
            elems.append([n0, n1, n2])
            elems.append([n1, n3, n2])
    elem_nodes = np.array(elems, dtype=np.int32)

    lon_c = np.array([lon[e].mean() for e in elem_nodes], dtype=np.float64)
    lat_c = np.array([lat[e].mean() for e in elem_nodes], dtype=np.float64)

    n2e, n2e_count = build_node_to_elems(n_nodes, elem_nodes)
    sigma = np.array(z_levels, dtype=np.float64)

    mesh = Mesh(
        name="test_local",
        nodes_lon=lon, nodes_lat=lat, depth=depth,
        elem_nodes=elem_nodes,
        elem_center_lon=lon_c, elem_center_lat=lat_c,
        node_to_elems=n2e, node_to_elems_count=n2e_count,
        sigma=sigma,
    )
    build_search(mesh)
    return mesh


def _make_vel_3d(mesh, u=0.0, v=0.0, w=0.0):
    """Uniform 3D velocity on all elements/nodes/layers.

    u, v: element-based (n_layers, n_elements)
    w:    node-based   (n_z_levels, n_nodes)
    """
    ne = mesh.n_elements
    nn = mesh.n_nodes
    n_z = mesh.sigma.shape[0]
    n_layers = n_z - 1
    return VelocityField(
        u=np.full((n_layers, ne), u, dtype=np.float64),
        v=np.full((n_layers, ne), v, dtype=np.float64),
        eta=np.zeros(nn, dtype=np.float64),
        w=np.full((n_z, nn), w, dtype=np.float64),
    )


def _make_vel_3d_layered(mesh, u_per_layer, v_per_layer, w=0.0):
    """3D velocity with different U/V per layer, uniform W.

    Parameters
    ----------
    u_per_layer, v_per_layer : list of float
        One value per mid-layer (len = n_z_levels - 1).
    w : float
        Uniform W on all level interfaces.
    """
    ne = mesh.n_elements
    nn = mesh.n_nodes
    n_z = mesh.sigma.shape[0]
    n_layers = n_z - 1
    assert len(u_per_layer) == n_layers
    assert len(v_per_layer) == n_layers

    u_3d = np.zeros((n_layers, ne), dtype=np.float64)
    v_3d = np.zeros((n_layers, ne), dtype=np.float64)
    for k in range(n_layers):
        u_3d[k, :] = u_per_layer[k]
        v_3d[k, :] = v_per_layer[k]

    return VelocityField(
        u=u_3d, v=v_3d,
        eta=np.zeros(nn, dtype=np.float64),
        w=np.full((n_z, nn), w, dtype=np.float64),
    )


def _predict_euler(lon0, lat0, z0, U, V, W, dt, n_steps):
    """Predict particle position after n Euler steps analytically.

    For constant velocity, each step uses the UPDATED lat for z0 metric.
    Returns (lon_final, lat_final, z_final).
    """
    lon, lat, z = float(lon0), float(lat0), float(z0)
    SLM = METERS_PER_DEG_LAT
    for _ in range(n_steps):
        z0_metric = max(np.cos(lat * DEG2RAD) * SLM, 1.0)
        lon = lon + dt * U / z0_metric
        lat = lat + dt * V / SLM
        z = z + W * dt
        # wrap/clamp like the code does
        lon = ((lon + 180.0) % 360.0) - 180.0
        lat = max(-90.0, min(90.0, lat))
    return lon, lat, z


# ===================================================================
# Test 1: Pure vertical motion — exact z prediction
# ===================================================================

class TestPureVertical:
    """No horizontal flow, only vertical.  Z should change linearly."""

    def test_downward_constant_w(self):
        """Particle sinks at constant rate, exact z after 10 steps."""
        mesh = _make_mesh(5.0, 5.0)
        W = 0.05   # 5 cm/s downward
        dt = 100.0
        n_steps = 10
        vel = _make_vel_3d(mesh, u=0.0, v=0.0, w=W)

        state = make_state(np.array([5.0]), np.array([5.0]), np.array([10.0]))
        for _ in range(n_steps):
            state = euler_step(state, vel, mesh, dt, time_dir=1.0)

        expected_z = 10.0 + W * dt * n_steps  # 10 + 0.05*100*10 = 60
        assert state.z[0] == pytest.approx(expected_z, abs=0.01)
        assert state.lon[0] == pytest.approx(5.0, abs=1e-10)
        assert state.lat[0] == pytest.approx(5.0, abs=1e-10)
        assert state.status[0] == ParticleStatus.ALIVE

    def test_upward_constant_w(self):
        """Particle rises at constant rate."""
        mesh = _make_mesh(5.0, 5.0)
        W = -0.1   # 10 cm/s upward (negative = shallower)
        dt = 50.0
        n_steps = 5
        vel = _make_vel_3d(mesh, u=0.0, v=0.0, w=W)

        state = make_state(np.array([5.0]), np.array([5.0]), np.array([80.0]))
        for _ in range(n_steps):
            state = euler_step(state, vel, mesh, dt, time_dir=1.0)

        expected_z = 80.0 + W * dt * n_steps  # 80 + (-0.1)*50*5 = 55
        assert state.z[0] == pytest.approx(expected_z, abs=0.01)

    def test_surface_clamp_prevents_negative_z(self):
        """Strong upward W clamps particle at surface, not negative."""
        mesh = _make_mesh(5.0, 5.0)
        vel = _make_vel_3d(mesh, w=-1.0)  # very strong upward
        state = make_state(np.array([5.0]), np.array([5.0]), np.array([5.0]))

        for _ in range(20):
            state = euler_step(state, vel, mesh, dt=10.0, time_dir=1.0)

        assert state.z[0] == pytest.approx(0.0)  # clamped at surface
        assert state.status[0] == ParticleStatus.ALIVE

    def test_bottom_clamp_at_local_depth(self):
        """Particle sinks but stops at mesh bottom depth."""
        bottom = 50.0
        mesh = _make_mesh(5.0, 5.0, bottom_depth=bottom,
                          z_levels=[0, 10, 25, 50])
        vel = _make_vel_3d(mesh, w=1.0)  # strong downward
        state = make_state(np.array([5.0]), np.array([5.0]), np.array([30.0]))

        for _ in range(20):
            state = euler_step(state, vel, mesh, dt=10.0, time_dir=1.0)

        assert state.z[0] == pytest.approx(bottom)
        assert state.status[0] == ParticleStatus.ALIVE

    def test_multiple_particles_different_depths(self):
        """3 particles at different depths all sink at same rate."""
        mesh = _make_mesh(5.0, 5.0)
        W = 0.02
        dt = 100.0
        vel = _make_vel_3d(mesh, w=W)

        z_init = np.array([5.0, 30.0, 80.0])
        state = make_state(
            np.array([5.0, 5.0, 5.0]),
            np.array([5.0, 5.0, 5.0]),
            z_init,
        )
        state = euler_step(state, vel, mesh, dt, time_dir=1.0)

        for i in range(3):
            expected = z_init[i] + W * dt
            assert state.z[i] == pytest.approx(expected, abs=0.01), \
                f"particle {i}: z={state.z[i]}, expected={expected}"


# ===================================================================
# Test 2: Depth-varying horizontal velocity
# ===================================================================

class TestDepthVaryingFlow:
    """U/V differ by layer; particle at specific depth gets correct velocity."""

    def test_particle_uses_correct_layer_velocity(self):
        """Two particles at different depths get different U values."""
        z_levels = [0.0, 10.0, 25.0, 50.0, 100.0, 200.0]
        mesh = _make_mesh(5.0, 5.0, z_levels=z_levels)

        # 5 mid-layers: different U per layer
        u_layers = [1.0, 2.0, 3.0, 4.0, 5.0]
        v_layers = [0.0, 0.0, 0.0, 0.0, 0.0]
        vel = _make_vel_3d_layered(mesh, u_layers, v_layers)

        # Particle A at z=5 → layer 0 (0-10m) → U=1.0
        # Particle B at z=30 → layer 2 (25-50m) → U=3.0
        state = make_state(
            np.array([5.0, 5.0]),
            np.array([5.0, 5.0]),
            np.array([5.0, 30.0]),
        )
        dt = 100.0
        state = euler_step(state, vel, mesh, dt, time_dir=1.0)

        # Both particles at same lat → same z0
        lat = 5.0
        z0 = np.cos(lat * DEG2RAD) * METERS_PER_DEG_LAT
        dlon_A = dt * 1.0 / z0   # layer 0, U=1.0
        dlon_B = dt * 3.0 / z0   # layer 2, U=3.0

        assert state.lon[0] == pytest.approx(5.0 + dlon_A, abs=1e-6)
        assert state.lon[1] == pytest.approx(5.0 + dlon_B, abs=1e-6)
        # B moved 3× farther than A
        assert (state.lon[1] - 5.0) == pytest.approx(
            3.0 * (state.lon[0] - 5.0), rel=0.01
        )

    def test_layer_switches_during_sinking(self):
        """Particle sinks from layer 0 (U=1) into layer 1 (U=2).

        After crossing the boundary, subsequent steps should use U=2.
        """
        z_levels = [0.0, 10.0, 50.0, 100.0]
        mesh = _make_mesh(5.0, 5.0, z_levels=z_levels, bottom_depth=100.0)

        u_layers = [1.0, 2.0, 0.5]
        v_layers = [0.0, 0.0, 0.0]
        W = 0.5  # sinking at 0.5 m/s
        vel = _make_vel_3d_layered(mesh, u_layers, v_layers, w=W)

        # Start at z=8 in layer 0 (0-10m), will cross into layer 1 at z=10
        state = make_state(np.array([5.0]), np.array([5.0]), np.array([8.0]))
        dt = 10.0

        # Step 1: z = 8+0.5*10 = 13 → now in layer 1 (10-50m)
        state = euler_step(state, vel, mesh, dt, time_dir=1.0)
        assert state.z[0] == pytest.approx(13.0, abs=0.1)

        # Record lon after first step (used layer 0: U=1.0)
        lon_after_1 = state.lon[0]

        # Step 2: now in layer 1 → U=2.0, z = 13+5 = 18
        state = euler_step(state, vel, mesh, dt, time_dir=1.0)
        dlon_step2 = state.lon[0] - lon_after_1

        # Verify step 2 used U=2.0 (twice the velocity of step 1)
        lat = state.lat[0]
        z0 = np.cos(lat * DEG2RAD) * METERS_PER_DEG_LAT
        expected_dlon = dt * 2.0 / z0
        assert dlon_step2 == pytest.approx(expected_dlon, rel=0.05)


# ===================================================================
# Test 3: Combined 3D — analytical multi-step prediction
# ===================================================================

class TestCombined3D:
    """Full 3D tracking with analytical predictions."""

    def test_10_steps_exact_prediction(self):
        """10 Euler steps with constant U, V, W — match analytical prediction."""
        mesh = _make_mesh(10.0, 40.0)
        U, V, W = 0.8, -0.3, 0.02
        dt = 300.0
        n_steps = 10
        vel = _make_vel_3d(mesh, u=U, v=V, w=W)

        lon0, lat0, z0 = 10.0, 40.0, 25.0
        state = make_state(np.array([lon0]), np.array([lat0]), np.array([z0]))

        for _ in range(n_steps):
            state = euler_step(state, vel, mesh, dt, time_dir=1.0)

        lon_exp, lat_exp, z_exp = _predict_euler(lon0, lat0, z0, U, V, W, dt, n_steps)

        assert state.lon[0] == pytest.approx(lon_exp, abs=1e-8)
        assert state.lat[0] == pytest.approx(lat_exp, abs=1e-8)
        assert state.z[0] == pytest.approx(z_exp, abs=1e-4)

    def test_backward_tracking_3d(self):
        """Backward tracking (time_dir=-1) reverses all three components."""
        mesh = _make_mesh(20.0, 30.0)
        U, V, W = 1.0, 0.5, 0.01
        dt = 200.0
        vel = _make_vel_3d(mesh, u=U, v=V, w=W)

        lon0, lat0, z0 = 20.0, 30.0, 40.0
        state = make_state(np.array([lon0]), np.array([lat0]), np.array([z0]))

        # Forward 5 steps
        for _ in range(5):
            state = euler_step(state, vel, mesh, dt, time_dir=1.0)

        # Backward 5 steps
        for _ in range(5):
            state = euler_step(state, vel, mesh, dt, time_dir=-1.0)

        assert state.lon[0] == pytest.approx(lon0, abs=1e-6)
        assert state.lat[0] == pytest.approx(lat0, abs=1e-6)
        assert state.z[0] == pytest.approx(z0, abs=1e-4)


# ===================================================================
# Test 4: Equator crossing
# ===================================================================

class TestEquatorCrossing:
    """Mesh centred at the equator.  cos(0°)=1 → z0 = METERS_PER_DEG_LAT."""

    @pytest.fixture
    def equator_mesh(self):
        return _make_mesh(30.0, 0.0, span=2.0)

    def test_southward_crosses_equator(self, equator_mesh):
        """Particle near equator moves south across it to Southern Hemisphere."""
        V = -0.5   # southward
        dt = 600.0
        n_steps = 50
        vel = _make_vel_3d(equator_mesh, v=V, w=0.001)

        state = make_state(np.array([30.0]), np.array([0.1]), np.array([20.0]))
        for _ in range(n_steps):
            state = euler_step(state, vel, equator_mesh, dt, time_dir=1.0)

        lon_exp, lat_exp, z_exp = _predict_euler(30.0, 0.1, 20.0, 0.0, V, 0.001, dt, n_steps)
        assert state.lat[0] < 0.0, "particle should have crossed the equator"
        assert state.lat[0] == pytest.approx(lat_exp, abs=1e-8)
        assert state.z[0] == pytest.approx(z_exp, abs=0.01)

    def test_metric_at_equator(self, equator_mesh):
        """At equator, z0 = cos(0)*SLM = SLM → Δlon = U*dt/SLM."""
        U = 1.0
        dt = 1000.0
        vel = _make_vel_3d(equator_mesh, u=U)
        state = make_state(np.array([30.0]), np.array([0.0]), np.array([10.0]))
        state = euler_step(state, vel, equator_mesh, dt, time_dir=1.0)

        expected_dlon = U * dt / METERS_PER_DEG_LAT
        assert (state.lon[0] - 30.0) == pytest.approx(expected_dlon, abs=1e-8)

    def test_northward_across_equator_3d(self, equator_mesh):
        """Particle near equator moves north across it while sinking."""
        U, V, W = 0.3, 0.8, 0.003
        dt = 500.0
        n_steps = 50
        vel = _make_vel_3d(equator_mesh, u=U, v=V, w=W)

        lon0, lat0, z0 = 30.0, -0.1, 15.0
        state = make_state(np.array([lon0]), np.array([lat0]), np.array([z0]))
        for _ in range(n_steps):
            state = euler_step(state, vel, equator_mesh, dt, time_dir=1.0)

        lon_exp, lat_exp, z_exp = _predict_euler(lon0, lat0, z0, U, V, W, dt, n_steps)
        assert state.lat[0] > 0.0, "particle should have crossed to Northern Hemisphere"
        assert state.lon[0] == pytest.approx(lon_exp, abs=1e-7)
        assert state.lat[0] == pytest.approx(lat_exp, abs=1e-7)
        assert state.z[0] == pytest.approx(z_exp, abs=0.1)


# ===================================================================
# Test 5: Prime meridian (0° longitude) crossing
# ===================================================================

class TestPrimeMeridianCrossing:
    """Mesh centred at 0° longitude.  Tests wrap_lon near zero."""

    @pytest.fixture
    def pm_mesh(self):
        return _make_mesh(0.0, 45.0, span=2.0)

    def test_westward_crosses_zero(self, pm_mesh):
        """Particle near 0°E moves west past 0° to negative longitudes."""
        U = -1.0  # westward
        dt = 600.0
        n_steps = 50
        vel = _make_vel_3d(pm_mesh, u=U, w=0.001)

        state = make_state(np.array([0.05]), np.array([45.0]), np.array([20.0]))
        for _ in range(n_steps):
            state = euler_step(state, vel, pm_mesh, dt, time_dir=1.0)

        lon_exp, lat_exp, z_exp = _predict_euler(0.05, 45.0, 20.0, U, 0.0, 0.001, dt, n_steps)
        assert state.lon[0] < 0.0, "particle should be at negative longitude"
        assert state.lon[0] == pytest.approx(lon_exp, abs=1e-7)

    def test_eastward_crosses_zero(self, pm_mesh):
        """Particle near 0°W moves east past 0° to positive longitudes."""
        U = 1.0
        dt = 600.0
        n_steps = 50
        vel = _make_vel_3d(pm_mesh, u=U, w=-0.001)

        state = make_state(np.array([-0.05]), np.array([45.0]), np.array([50.0]))
        for _ in range(n_steps):
            state = euler_step(state, vel, pm_mesh, dt, time_dir=1.0)

        lon_exp, _, z_exp = _predict_euler(-0.05, 45.0, 50.0, U, 0.0, -0.001, dt, n_steps)
        assert state.lon[0] > 0.0, "particle should be at positive longitude"
        assert state.lon[0] == pytest.approx(lon_exp, abs=1e-7)
        assert state.z[0] == pytest.approx(z_exp, abs=0.1)


# ===================================================================
# Test 6: Dateline (±180°) crossing
# ===================================================================

class TestDatelineCrossing:
    """Mesh centred near 180° longitude — particles cross the dateline."""

    @pytest.fixture
    def dateline_mesh(self):
        return _make_mesh(179.0, 45.0, span=2.0)

    def test_eastward_crosses_dateline(self, dateline_mesh):
        """Particle crosses from +179° to -179° (through 180°)."""
        U = 10.0   # strong eastward to cross within mesh extent
        dt = 600.0
        n_steps = 20
        vel = _make_vel_3d(dateline_mesh, u=U, w=0.005)

        lon0 = 179.0
        state = make_state(np.array([lon0]), np.array([45.0]), np.array([20.0]))
        for _ in range(n_steps):
            state = euler_step(state, vel, dateline_mesh, dt, time_dir=1.0)

        lon_exp, _, z_exp = _predict_euler(lon0, 45.0, 20.0, U, 0.0, 0.005, dt, n_steps)

        # Particle should have wrapped to negative longitude
        assert state.lon[0] < 0.0, \
            f"particle should have crossed dateline, got lon={state.lon[0]:.4f}"
        assert state.lon[0] == pytest.approx(lon_exp, abs=1e-5)
        assert state.z[0] == pytest.approx(z_exp, abs=0.1)
        assert state.status[0] == ParticleStatus.ALIVE

    def test_westward_crosses_dateline(self, dateline_mesh):
        """Particle crosses from -179° to +179° (through -180°)."""
        mesh = _make_mesh(-179.0, 45.0, span=2.0)
        U = -10.0
        dt = 600.0
        n_steps = 20
        vel = _make_vel_3d(mesh, u=U, w=-0.002)

        lon0 = -179.0
        state = make_state(np.array([lon0]), np.array([45.0]), np.array([30.0]))
        for _ in range(n_steps):
            state = euler_step(state, vel, mesh, dt, time_dir=1.0)

        lon_exp, _, z_exp = _predict_euler(lon0, 45.0, 30.0, U, 0.0, -0.002, dt, n_steps)

        # Particle should have wrapped to positive longitude
        assert state.lon[0] > 0.0, \
            f"particle should have crossed dateline, got lon={state.lon[0]:.4f}"
        assert state.lon[0] == pytest.approx(lon_exp, abs=1e-5)
        assert state.z[0] == pytest.approx(z_exp, abs=0.1)
        assert state.status[0] == ParticleStatus.ALIVE

    def test_analytical_across_dateline(self, dateline_mesh):
        """Multi-step 3D tracking across dateline matches analytical prediction."""
        U, V, W = 8.0, 0.2, 0.01
        dt = 600.0
        n_steps = 15
        vel = _make_vel_3d(dateline_mesh, u=U, v=V, w=W)

        lon0, lat0, z0 = 179.5, 45.0, 25.0
        state = make_state(np.array([lon0]), np.array([lat0]), np.array([z0]))
        for _ in range(n_steps):
            state = euler_step(state, vel, dateline_mesh, dt, time_dir=1.0)

        lon_exp, lat_exp, z_exp = _predict_euler(lon0, lat0, z0, U, V, W, dt, n_steps)
        assert state.lon[0] == pytest.approx(lon_exp, abs=1e-5)
        assert state.lat[0] == pytest.approx(lat_exp, abs=1e-7)
        assert state.z[0] == pytest.approx(z_exp, abs=0.1)
        assert state.status[0] == ParticleStatus.ALIVE


# ===================================================================
# Test 7: Near North Pole
# ===================================================================

class TestNorthPole:
    """Mesh at high latitude.  cos(88°) ≈ 0.035 → z0 very small,
    so 1 m/s eastward produces large Δlon (≈ 0.26°/step for dt=1000s).

    This tests that the metric scaling is correct near the pole.
    """

    @pytest.fixture
    def pole_mesh(self):
        # At 88°N, 1° lon ≈ 3.88 km.  Use smaller span to keep mesh valid.
        return _make_mesh(0.0, 88.0, span=0.5, nx=3, ny=3)

    def test_metric_scaling_at_high_lat(self, pole_mesh):
        """At 88°N, Δlon is ~28× larger than at equator for same U."""
        U = 1.0
        dt = 100.0
        vel = _make_vel_3d(pole_mesh, u=U)

        state = make_state(np.array([0.0]), np.array([88.0]), np.array([10.0]))
        state = euler_step(state, vel, pole_mesh, dt, time_dir=1.0)

        z0_pole = np.cos(88.0 * DEG2RAD) * METERS_PER_DEG_LAT
        z0_equator = METERS_PER_DEG_LAT
        expected_dlon = U * dt / z0_pole
        actual_dlon = state.lon[0] - 0.0

        # Verify the actual displacement matches prediction
        assert actual_dlon == pytest.approx(expected_dlon, abs=1e-6)
        # And it's much larger than at equator
        equator_dlon = U * dt / z0_equator
        assert actual_dlon > 20 * equator_dlon

    def test_northward_toward_pole(self, pole_mesh):
        """Particle moves northward toward the pole."""
        V = 0.3
        dt = 100.0
        vel = _make_vel_3d(pole_mesh, v=V, w=0.01)

        state = make_state(np.array([0.0]), np.array([88.0]), np.array([15.0]))
        state = euler_step(state, vel, pole_mesh, dt, time_dir=1.0)

        expected_dlat = V * dt / METERS_PER_DEG_LAT
        assert (state.lat[0] - 88.0) == pytest.approx(expected_dlat, abs=1e-8)
        # z should have increased
        assert state.z[0] == pytest.approx(15.0 + 0.01 * 100.0, abs=0.01)

    def test_3d_tracking_near_pole(self, pole_mesh):
        """Multi-step 3D tracking near the pole matches analytical prediction."""
        U, V, W = 0.5, 0.1, 0.005
        dt = 50.0
        n_steps = 5
        vel = _make_vel_3d(pole_mesh, u=U, v=V, w=W)

        lon0, lat0, z0 = 0.0, 88.0, 20.0
        state = make_state(np.array([lon0]), np.array([lat0]), np.array([z0]))
        for _ in range(n_steps):
            state = euler_step(state, vel, pole_mesh, dt, time_dir=1.0)

        lon_exp, lat_exp, z_exp = _predict_euler(lon0, lat0, z0, U, V, W, dt, n_steps)
        assert state.lon[0] == pytest.approx(lon_exp, abs=1e-6)
        assert state.lat[0] == pytest.approx(lat_exp, abs=1e-8)
        assert state.z[0] == pytest.approx(z_exp, abs=0.01)


# ===================================================================
# Test 8: RK2/RK4 consistency for constant fields
# ===================================================================

class TestRKConsistency3D:
    """For spatially and vertically constant velocity, all integrators
    should give identical results (within floating-point tolerance)."""

    @pytest.mark.parametrize("center_lon,center_lat", [
        (5.0, 5.0),       # generic mid-latitude
        (30.0, 0.0),      # equator
        (0.0, 45.0),      # prime meridian
        (0.0, 88.0),      # near pole
    ])
    def test_all_integrators_agree(self, center_lon, center_lat):
        span = 0.5 if center_lat > 85 else 1.0
        mesh = _make_mesh(center_lon, center_lat, span=span)
        U, V, W = 0.5, -0.2, 0.01
        dt = 100.0
        vel = _make_vel_3d(mesh, u=U, v=V, w=W)

        state0 = make_state(
            np.array([center_lon]),
            np.array([center_lat]),
            np.array([25.0]),
        )

        s_euler = euler_step(state0, vel, mesh, dt, time_dir=1.0)
        s_rk2 = rk2_step(state0, vel, mesh, dt, time_dir=1.0)
        s_rk4 = rk4_step(state0, vel, mesh, dt, time_dir=1.0)

        # All should agree for constant velocity
        assert s_rk2.lon[0] == pytest.approx(s_euler.lon[0], abs=1e-6)
        assert s_rk2.lat[0] == pytest.approx(s_euler.lat[0], abs=1e-8)
        assert s_rk2.z[0] == pytest.approx(s_euler.z[0], abs=0.01)

        assert s_rk4.lon[0] == pytest.approx(s_euler.lon[0], abs=1e-6)
        assert s_rk4.lat[0] == pytest.approx(s_euler.lat[0], abs=1e-8)
        assert s_rk4.z[0] == pytest.approx(s_euler.z[0], abs=0.01)


# ===================================================================
# Diagnostic plotting (PLOT_3D=1 to enable)
# ===================================================================

def _plot_3d_test(mesh, vel, trajectories, title, filename,
                  z_levels=None):
    """Plot mesh, velocity arrows, and trajectories coloured by depth.

    Uses cartopy for map panels — handles dateline crossing seamlessly.

    Parameters
    ----------
    mesh : Mesh
    vel : VelocityField
        Used for velocity arrows (layer 0 for horizontal).
    trajectories : list of dict
        Each dict has keys "lon", "lat", "z" — arrays of shape (n_steps,)
        for one particle.
    title : str
    filename : str
        Saved to PLOT_DIR/filename.
    z_levels : array-like or None
        For depth colorbar limits.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Determine map extent from mesh + trajectories
    all_lons = np.concatenate([mesh.nodes_lon] + [t["lon"] for t in trajectories])
    all_lats = np.concatenate([mesh.nodes_lat] + [t["lat"] for t in trajectories])

    # Check if data spans the dateline
    has_positive = np.any(all_lons > 90)
    has_negative = np.any(all_lons < -90)
    spans_dateline = has_positive and has_negative

    # Choose projection centre: 180° for dateline cases, mesh centre otherwise
    if spans_dateline:
        central_lon = 180.0
    else:
        central_lon = 0.5 * (all_lons.min() + all_lons.max())

    proj = ccrs.PlateCarree(central_longitude=central_lon)
    data_crs = ccrs.PlateCarree()  # data is always in plain lon/lat

    # Extent with margin
    lon_min, lon_max = all_lons.min(), all_lons.max()
    lat_min, lat_max = all_lats.min(), all_lats.max()
    lon_margin = max((lon_max - lon_min) * 0.15, 0.5)
    lat_margin = max((lat_max - lat_min) * 0.15, 0.5)
    # For dateline spanning, compute extent in shifted coordinates
    if spans_dateline:
        shifted = (all_lons + 360.0) % 360.0  # [0, 360)
        lon_min_s, lon_max_s = shifted.min(), shifted.max()
        lon_span = lon_max_s - lon_min_s + 2 * lon_margin
        # Ensure lat range is at least 1/3 of lon range for readable plots
        lat_span = lat_max - lat_min + 2 * lat_margin
        if lat_span < lon_span / 3:
            lat_extra = (lon_span / 3 - lat_span) / 2
            lat_margin += lat_extra
        extent = [lon_min_s - lon_margin - central_lon,
                  lon_max_s + lon_margin - central_lon,
                  max(lat_min - lat_margin, -90),
                  min(lat_max + lat_margin, 90)]
    else:
        extent = [lon_min - lon_margin - central_lon,
                  lon_max + lon_margin - central_lon,
                  lat_min - lat_margin, lat_max + lat_margin]

    fig = plt.figure(figsize=(20, 7))
    fig.suptitle(title, fontsize=14)

    # --- Panel 1: Mesh + velocity arrows + trajectories ---
    ax = fig.add_subplot(1, 3, 1, projection=proj)
    ax.set_title("Mesh, velocity & trajectories")
    ax.set_extent(extent, crs=proj)
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    ax.coastlines(resolution="110m", linewidth=0.3, color="gray")

    # Draw mesh triangles
    elem_nodes = mesh.elem_nodes
    for e in range(mesh.n_elements):
        nodes = elem_nodes[e]
        valid = nodes[nodes >= 0]
        if len(valid) < 3:
            continue
        xs = list(mesh.nodes_lon[valid]) + [mesh.nodes_lon[valid[0]]]
        ys = list(mesh.nodes_lat[valid]) + [mesh.nodes_lat[valid[0]]]
        ax.plot(xs, ys, "-", color="lightgray", linewidth=0.5, zorder=0,
                transform=data_crs)

    # Mesh nodes
    ax.plot(mesh.nodes_lon, mesh.nodes_lat, ".", color="gray",
            markersize=3, zorder=1, transform=data_crs)

    # Velocity arrows at element centres (layer 0)
    u_arr = vel.u[0, :] if vel.u.ndim == 2 else vel.u
    v_arr = vel.v[0, :] if vel.v.ndim == 2 else vel.v
    ax.quiver(mesh.elem_center_lon, mesh.elem_center_lat,
              u_arr, v_arr, color="steelblue", alpha=0.6,
              scale=None, zorder=2, transform=data_crs)

    # Trajectories coloured by depth
    all_z = np.concatenate([t["z"] for t in trajectories])
    z_min, z_max = all_z.min(), all_z.max()
    if z_max - z_min < 0.01:
        z_min, z_max = z_min - 1, z_max + 1
    cmap = plt.cm.viridis
    norm = plt.Normalize(z_min, z_max)

    for t in trajectories:
        lons, lats, zs = t["lon"], t["lat"], t["z"]
        # Scatter points coloured by depth (no connecting lines —
        # cartopy draws great-circle arcs that wrap around the globe
        # when consecutive points cross the dateline)
        sc = ax.scatter(lons, lats, c=zs, cmap=cmap, norm=norm,
                        s=12, zorder=3, transform=data_crs)
        ax.plot(lons[0], lats[0], "go", markersize=7, zorder=4,
                transform=data_crs)
        ax.plot(lons[-1], lats[-1], "rs", markersize=7, zorder=4,
                transform=data_crs)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, label="Depth (m)", shrink=0.8)

    # --- Panel 2: Depth vs step ---
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title("Depth profile")
    for i, t in enumerate(trajectories):
        steps = np.arange(len(t["z"]))
        ax2.plot(steps, t["z"], "-o", markersize=3, label=f"P{i}")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Depth (m)")
    ax2.invert_yaxis()
    if z_levels is not None:
        for zl in z_levels:
            ax2.axhline(zl, color="lightgray", linewidth=0.5, linestyle="--")
    ax2.legend(fontsize=8)

    # --- Panel 3: Top view with cartopy ---
    ax3 = fig.add_subplot(1, 3, 3, projection=proj)
    ax3.set_title("Top view (lon/lat)")
    ax3.set_extent(extent, crs=proj)
    ax3.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    ax3.coastlines(resolution="110m", linewidth=0.3, color="gray")

    colors_cycle = plt.cm.tab10.colors
    for i, t in enumerate(trajectories):
        lons, lats = t["lon"], t["lat"]
        color = colors_cycle[i % len(colors_cycle)]
        ax3.scatter(lons, lats, s=12, color=color, label=f"P{i}",
                    zorder=3, transform=data_crs)
        ax3.plot(lons[0], lats[0], "go", markersize=7, zorder=4,
                 transform=data_crs)
        ax3.plot(lons[-1], lats[-1], "rs", markersize=7, zorder=4,
                 transform=data_crs)
    ax3.legend(fontsize=8, loc="lower left")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {path}")

    # Write ASCII dat file with tracer positions
    dat_path = os.path.join(PLOT_DIR, filename.replace(".png", ".dat"))
    with open(dat_path, "w") as f:
        f.write(f"# {title}\n")
        f.write(f"# step  particle  lon           lat           z(depth)\n")
        n_steps_max = max(len(t["lon"]) for t in trajectories)
        for step in range(n_steps_max):
            for p, t in enumerate(trajectories):
                if step < len(t["lon"]):
                    f.write(f"  {step:4d}  {p:4d}     "
                            f"{t['lon'][step]:+13.6f}  "
                            f"{t['lat'][step]:+13.6f}  "
                            f"{t['z'][step]:10.4f}\n")
    print(f"  Data saved: {dat_path}")


def _run_and_collect(mesh, vel, state, dt, n_steps, time_dir=1.0):
    """Run Euler steps and collect trajectory history per particle.

    Returns list of dicts with "lon", "lat", "z" arrays.
    """
    P = state.n_particles
    history = [{"lon": [state.lon[p]], "lat": [state.lat[p]], "z": [state.z[p]]}
               for p in range(P)]

    for _ in range(n_steps):
        state = euler_step(state, vel, mesh, dt, time_dir=time_dir)
        for p in range(P):
            history[p]["lon"].append(state.lon[p])
            history[p]["lat"].append(state.lat[p])
            history[p]["z"].append(state.z[p])

    for p in range(P):
        for k in ("lon", "lat", "z"):
            history[p][k] = np.array(history[p][k])

    return history, state


@pytest.mark.skipif(not PLOT_3D, reason="Set PLOT_3D=1 to enable plotting")
class TestVisual3D:
    """Visual diagnostic tests — generate plots for manual inspection.

    Run with: PLOT_3D=1 python -m pytest tests/test_3d_analytical.py::TestVisual3D -v -s
    """

    def test_plot_sinking_particles(self):
        """Multiple particles sinking at different starting depths."""
        mesh = _make_mesh(5.0, 5.0, nx=5, ny=5, span=1.5)
        vel = _make_vel_3d(mesh, u=0.5, v=0.2, w=0.03)

        state = make_state(
            np.array([5.0, 5.0, 5.0, 4.8, 5.2]),
            np.array([5.0, 5.0, 5.0, 5.1, 4.9]),
            np.array([5.0, 30.0, 80.0, 10.0, 50.0]),
        )
        history, _ = _run_and_collect(mesh, vel, state, dt=200.0, n_steps=20)

        _plot_3d_test(mesh, vel, history,
                      "Sinking particles (5°N, 5°E) — U=0.5, V=0.2, W=0.03",
                      "sinking_particles.png",
                      z_levels=mesh.z_levels)

    def test_plot_depth_varying_flow(self):
        """Depth-varying U: shallow particles move slow, deep ones fast."""
        z_levels = [0.0, 10.0, 25.0, 50.0, 100.0, 200.0]
        mesh = _make_mesh(10.0, 40.0, nx=5, ny=5, span=1.5, z_levels=z_levels)

        u_layers = [0.2, 0.5, 1.0, 2.0, 3.0]
        v_layers = [0.1, 0.1, 0.1, 0.1, 0.1]
        vel = _make_vel_3d_layered(mesh, u_layers, v_layers, w=0.0)

        state = make_state(
            np.array([10.0, 10.0, 10.0, 10.0, 10.0]),
            np.array([40.0, 40.0, 40.0, 40.0, 40.0]),
            np.array([5.0, 15.0, 35.0, 70.0, 150.0]),
        )
        history, _ = _run_and_collect(mesh, vel, state, dt=300.0, n_steps=15)

        _plot_3d_test(mesh, vel, history,
                      "Depth-varying U (10°E, 40°N) — deeper = faster",
                      "depth_varying_flow.png",
                      z_levels=z_levels)

    def test_plot_equator_crossing(self):
        """Particles crossing the equator with vertical motion."""
        mesh = _make_mesh(30.0, 0.0, nx=5, ny=5, span=2.0)
        vel = _make_vel_3d(mesh, u=0.3, v=-0.5, w=0.005)

        state = make_state(
            np.array([30.0, 30.5, 29.5]),
            np.array([0.5, 0.3, 0.8]),
            np.array([10.0, 40.0, 80.0]),
        )
        history, _ = _run_and_collect(mesh, vel, state, dt=600.0, n_steps=50)

        _plot_3d_test(mesh, vel, history,
                      "Equator crossing (30°E) — V=-0.5, W=0.005",
                      "equator_crossing.png",
                      z_levels=mesh.z_levels)

    def test_plot_prime_meridian(self):
        """Particles crossing the prime meridian."""
        mesh = _make_mesh(0.0, 45.0, nx=5, ny=5, span=2.0)
        vel = _make_vel_3d(mesh, u=-0.8, v=0.2, w=-0.01)

        state = make_state(
            np.array([0.5, 0.3, 0.1]),
            np.array([45.0, 45.2, 44.8]),
            np.array([50.0, 30.0, 80.0]),
        )
        history, _ = _run_and_collect(mesh, vel, state, dt=600.0, n_steps=50)

        _plot_3d_test(mesh, vel, history,
                      "Prime meridian crossing (45°N) — U=-0.8, rising",
                      "prime_meridian.png",
                      z_levels=mesh.z_levels)

    def test_plot_dateline_crossing(self):
        """Particles crossing the dateline from +179° to -179°.

        W=0 (pure horizontal) for clarity.  Verifies positions against
        analytical prediction and writes both to the dat file.
        """
        mesh = _make_mesh(179.0, 45.0, nx=5, ny=5, span=2.0)
        U, V, W = 10.0, 0.0, 0.0
        dt = 600.0
        n_steps = 15
        vel = _make_vel_3d(mesh, u=U, v=V, w=W)

        lon0s = [179.0, 179.5, 178.5]
        lat0s = [45.0, 45.0, 45.0]
        z0s = [20.0, 20.0, 20.0]
        state = make_state(
            np.array(lon0s), np.array(lat0s), np.array(z0s),
        )
        history, final = _run_and_collect(mesh, vel, state, dt, n_steps)

        # Verify against analytical prediction
        for p in range(len(lon0s)):
            lon_exp, lat_exp, z_exp = _predict_euler(
                lon0s[p], lat0s[p], z0s[p], U, V, W, dt, n_steps)
            if final.status[p] == ParticleStatus.ALIVE:
                assert final.lon[p] == pytest.approx(lon_exp, abs=1e-5), \
                    f"P{p}: lon={final.lon[p]:.6f} expected={lon_exp:.6f}"
                assert final.lat[p] == pytest.approx(lat_exp, abs=1e-7), \
                    f"P{p}: lat mismatch"
                assert final.z[p] == pytest.approx(z_exp, abs=0.1), \
                    f"P{p}: z mismatch"

        _plot_3d_test(mesh, vel, history,
                      f"Dateline crossing — U={U} m/s, W=0 (pure horizontal)",
                      "dateline_crossing.png",
                      z_levels=mesh.z_levels)

    def test_plot_north_pole(self):
        """Particles near the North Pole — large metric distortion."""
        mesh = _make_mesh(0.0, 88.0, nx=5, ny=5, span=0.5)
        vel = _make_vel_3d(mesh, u=0.3, v=0.05, w=0.008)

        state = make_state(
            np.array([0.0, 0.1, -0.1]),
            np.array([88.0, 88.0, 88.0]),
            np.array([10.0, 40.0, 80.0]),
        )
        history, _ = _run_and_collect(mesh, vel, state, dt=100.0, n_steps=20)

        _plot_3d_test(mesh, vel, history,
                      "Near North Pole (88°N) — large Δlon, small Δlat",
                      "north_pole.png",
                      z_levels=mesh.z_levels)

    def test_plot_layer_switching(self):
        """Particle sinks through layers with different velocities."""
        z_levels = [0.0, 10.0, 25.0, 50.0, 100.0, 200.0]
        mesh = _make_mesh(5.0, 5.0, nx=5, ny=5, span=1.5, z_levels=z_levels)

        u_layers = [0.3, 0.8, 1.5, 2.5, 0.5]
        v_layers = [0.1, 0.2, -0.1, -0.3, 0.0]
        vel = _make_vel_3d_layered(mesh, u_layers, v_layers, w=0.05)

        state = make_state(
            np.array([5.0, 5.0]),
            np.array([5.0, 5.0]),
            np.array([2.0, 2.0]),
        )
        # One particle sinks, the other stays at surface (2D vel)
        vel_nosink = _make_vel_3d_layered(mesh, u_layers, v_layers, w=0.0)

        hist_sink, _ = _run_and_collect(mesh, vel, state, dt=100.0, n_steps=40)
        state2 = make_state(
            np.array([5.0]),
            np.array([5.0]),
            np.array([2.0]),
        )
        hist_nosink, _ = _run_and_collect(mesh, vel_nosink, state2,
                                           dt=100.0, n_steps=40)

        all_hist = hist_sink + hist_nosink
        _plot_3d_test(mesh, vel, all_hist,
                      "Layer switching — sinking particles accelerate then slow down",
                      "layer_switching.png",
                      z_levels=z_levels)

    def test_plot_bottom_clamp(self):
        """Particle sinks and gets clamped at the bottom."""
        z_levels = [0.0, 10.0, 25.0, 50.0]
        mesh = _make_mesh(5.0, 5.0, nx=5, ny=5, span=1.5,
                          z_levels=z_levels, bottom_depth=50.0)
        vel = _make_vel_3d(mesh, u=0.5, v=0.1, w=0.2)

        state = make_state(
            np.array([5.0, 5.0]),
            np.array([5.0, 5.0]),
            np.array([10.0, 40.0]),
        )
        history, _ = _run_and_collect(mesh, vel, state, dt=50.0, n_steps=30)

        _plot_3d_test(mesh, vel, history,
                      "Bottom clamp at 50m — particles stop sinking",
                      "bottom_clamp.png",
                      z_levels=z_levels)
