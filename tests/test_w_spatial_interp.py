"""Tests for horizontal spatial interpolation of W velocity.

Tests the four methods: nearest, leastsq, barycentric, idw.
All use synthetic data — no external files needed.
"""
from __future__ import annotations

import numpy as np
import pytest

from kelp.numerics.integrators import euler_step, rk2_step, rk4_step
from kelp.numerics.interpolate import VelocityField
from kelp.core.mesh import Mesh, build_node_to_elems, build_search
from kelp.core.particles import make_state, ParticleStatus
from kelp.core.geometry import METERS_PER_DEG_LAT, DEG2RAD
from kelp.numerics.vertical import (
    _collect_neighbour_nodes,
    _interp_w_leastsq_single,
    _interp_w_barycentric_single,
    _interp_w_idw_single,
    interp_w_spatial,
    interp_w_linear,
    W_NEAREST, W_LEASTSQ, W_BARYCENTRIC, W_IDW,
)


# ---------------------------------------------------------------------------
# Mesh builders
# ---------------------------------------------------------------------------

def _make_mesh(center_lon=5.0, center_lat=5.0, span=1.0, nx=3, ny=3,
               z_levels=None, bottom_depth=200.0):
    """Build a local triangular mesh."""
    if z_levels is None:
        z_levels = [0.0, 10.0, 25.0, 50.0, 100.0, 200.0]

    xs = np.linspace(center_lon - span, center_lon + span, nx)
    ys = np.linspace(center_lat - span, center_lat + span, ny)
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel().astype(np.float64)
    lat = gy.ravel().astype(np.float64)
    n_nodes = lon.shape[0]
    depth = np.full(n_nodes, bottom_depth, dtype=np.float64)

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


def _make_vel_3d(mesh, w_values=None, u=0.0, v=0.0):
    """Build 3D velocity.  w_values: (K, N) array or scalar."""
    ne = mesh.n_elements
    nn = mesh.n_nodes
    n_z = mesh.sigma.shape[0]
    n_layers = n_z - 1

    if w_values is None:
        w_arr = np.zeros((n_z, nn), dtype=np.float64)
    elif np.isscalar(w_values):
        w_arr = np.full((n_z, nn), w_values, dtype=np.float64)
    else:
        w_arr = np.asarray(w_values, dtype=np.float64)

    return VelocityField(
        u=np.full((n_layers, ne), u, dtype=np.float64),
        v=np.full((n_layers, ne), v, dtype=np.float64),
        eta=np.zeros(nn, dtype=np.float64),
        w=w_arr,
    )


# ---------------------------------------------------------------------------
# Test _collect_neighbour_nodes
# ---------------------------------------------------------------------------

class TestCollectNeighbourNodes:
    def test_interior_node(self):
        """Interior node of 3x3 mesh should have ~7 neighbours (self + ring)."""
        mesh = _make_mesh()
        center_node = 4  # middle of 3x3 grid
        buf, count = _collect_neighbour_nodes(
            center_node, mesh.node_to_elems, mesh.node_to_elems_count,
            mesh.elem_nodes,
        )
        nodes = set(buf[:count])
        assert center_node in nodes  # includes self
        assert count >= 5  # at least self + 4 immediate neighbours
        assert count <= 9  # at most all 9 nodes in 3x3

    def test_corner_node(self):
        """Corner node should have fewer neighbours."""
        mesh = _make_mesh()
        buf, count = _collect_neighbour_nodes(
            0, mesh.node_to_elems, mesh.node_to_elems_count,
            mesh.elem_nodes,
        )
        assert count >= 2  # at least self + 1 neighbour
        assert count < 9

    def test_no_duplicates(self):
        """Collected nodes should be unique."""
        mesh = _make_mesh()
        for node_idx in range(mesh.n_nodes):
            buf, count = _collect_neighbour_nodes(
                node_idx, mesh.node_to_elems, mesh.node_to_elems_count,
                mesh.elem_nodes,
            )
            nodes = buf[:count]
            assert len(set(nodes)) == count


# ---------------------------------------------------------------------------
# Test constant W field — all methods should agree
# ---------------------------------------------------------------------------

class TestConstantW:
    """All methods should return the same value for a uniform W field."""

    @pytest.fixture
    def setup(self):
        mesh = _make_mesh()
        W_CONST = 0.005
        vel = _make_vel_3d(mesh, w_values=W_CONST)
        z_levels = np.abs(mesh.sigma)
        nn = np.array([4], dtype=np.int32)  # center node
        alive = np.array([True])
        z_p = np.array([30.0])
        lons = np.array([5.0])
        lats = np.array([5.0])
        return mesh, vel, z_levels, nn, alive, z_p, lons, lats, W_CONST

    @pytest.mark.parametrize("method", ["nearest", "leastsq", "barycentric", "idw"])
    def test_constant_all_methods(self, setup, method):
        mesh, vel, z_levels, nn, alive, z_p, lons, lats, W_CONST = setup
        W = interp_w_spatial(
            vel.w, z_levels, z_p, lons, lats, nn, alive, 1.0,
            method=method, mesh=mesh,
        )
        assert W[0] == pytest.approx(W_CONST, abs=1e-10)


# ---------------------------------------------------------------------------
# Test linear gradient W field — leastsq and bary should be exact
# ---------------------------------------------------------------------------

class TestLinearGradientW:
    """Linear W(lon, lat) = a*lon + b*lat + c.
    LSQ and barycentric should reproduce exactly at interior points.
    """

    @pytest.fixture
    def setup(self):
        mesh = _make_mesh()
        z_levels = np.abs(mesh.sigma)
        n_z = z_levels.shape[0]

        # W varies linearly with lon and lat, constant in z
        a, b, c = 0.01, -0.005, 0.002
        w_arr = np.empty((n_z, mesh.n_nodes), dtype=np.float64)
        for k in range(n_z):
            for n in range(mesh.n_nodes):
                w_arr[k, n] = a * mesh.nodes_lon[n] + b * mesh.nodes_lat[n] + c

        vel = _make_vel_3d(mesh, w_values=w_arr)
        # Particle at mesh center
        xp, yp = 5.0, 5.0
        expected = a * xp + b * yp + c
        nn = np.array([4], dtype=np.int32)
        alive = np.array([True])
        z_p = np.array([30.0])
        lons = np.array([xp])
        lats = np.array([yp])
        return mesh, vel, z_levels, nn, alive, z_p, lons, lats, expected

    def test_leastsq_exact(self, setup):
        mesh, vel, z_levels, nn, alive, z_p, lons, lats, expected = setup
        W = interp_w_spatial(
            vel.w, z_levels, z_p, lons, lats, nn, alive, 1.0,
            method="leastsq", mesh=mesh,
        )
        assert W[0] == pytest.approx(expected, abs=1e-8)

    def test_barycentric_exact(self, setup):
        mesh, vel, z_levels, nn, alive, z_p, lons, lats, expected = setup
        W = interp_w_spatial(
            vel.w, z_levels, z_p, lons, lats, nn, alive, 1.0,
            method="barycentric", mesh=mesh,
        )
        assert W[0] == pytest.approx(expected, abs=1e-6)

    def test_nearest_approximate(self, setup):
        """Nearest just returns center node value — may not match linear field."""
        mesh, vel, z_levels, nn, alive, z_p, lons, lats, expected = setup
        W = interp_w_spatial(
            vel.w, z_levels, z_p, lons, lats, nn, alive, 1.0,
            method="nearest", mesh=mesh,
        )
        # Nearest returns W at nearest node (which IS the center here)
        # For this specific case it should be close since particle is at center
        assert W[0] == pytest.approx(expected, abs=1e-6)

    def test_idw_close(self, setup):
        """IDW should be reasonable but not exact for linear field."""
        mesh, vel, z_levels, nn, alive, z_p, lons, lats, expected = setup
        W = interp_w_spatial(
            vel.w, z_levels, z_p, lons, lats, nn, alive, 1.0,
            method="idw", mesh=mesh,
        )
        # IDW at center node with symmetric neighbours should be close
        assert W[0] == pytest.approx(expected, abs=1e-3)


# ---------------------------------------------------------------------------
# Test cross-term field: W(k,n) = a*lon + b*lat + c*z + d
# ---------------------------------------------------------------------------

class TestCrossTermW:
    """LSQ should be exact for W that is linear in (x, y) at any fixed z."""

    def test_leastsq_cross_term(self):
        mesh = _make_mesh()
        z_levels = np.abs(mesh.sigma)
        n_z = z_levels.shape[0]

        a, b, c, d = 0.01, -0.005, 0.0001, 0.002
        w_arr = np.empty((n_z, mesh.n_nodes), dtype=np.float64)
        for k in range(n_z):
            for n in range(mesh.n_nodes):
                w_arr[k, n] = (a * mesh.nodes_lon[n]
                               + b * mesh.nodes_lat[n]
                               + c * z_levels[k] + d)

        vel = _make_vel_3d(mesh, w_values=w_arr)
        z_test = 37.5  # between z_levels[2]=25 and z_levels[3]=50
        xp, yp = 5.0, 5.0
        # At z=37.5, after vertical linear interp:
        # alpha = (37.5 - 25) / (50 - 25) = 0.5
        # W(node, z=37.5) = (1-0.5)*W(node, z=25) + 0.5*W(node, z=50)
        # = 0.5 * (a*lon + b*lat + c*25 + d) + 0.5 * (a*lon + b*lat + c*50 + d)
        # = a*lon + b*lat + c*37.5 + d  (linear in z → exact)
        expected = a * xp + b * yp + c * z_test + d

        nn = np.array([4], dtype=np.int32)
        alive = np.array([True])
        z_p = np.array([z_test])
        lons = np.array([xp])
        lats = np.array([yp])

        W = interp_w_spatial(
            vel.w, z_levels, z_p, lons, lats, nn, alive, 1.0,
            method="leastsq", mesh=mesh,
        )
        assert W[0] == pytest.approx(expected, abs=1e-8)


# ---------------------------------------------------------------------------
# Test nearest compatibility
# ---------------------------------------------------------------------------

class TestNearestCompat:
    """Nearest method must match existing interp_w_linear exactly."""

    def test_matches_interp_w_linear(self):
        mesh = _make_mesh()
        z_levels = np.abs(mesh.sigma)
        vel = _make_vel_3d(mesh, w_values=0.003)
        nn = np.array([4, 0, 8], dtype=np.int32)
        alive = np.array([True, True, True])
        z_p = np.array([15.0, 40.0, 5.0])
        lons = np.array([5.0, 4.0, 6.0])
        lats = np.array([5.0, 4.0, 6.0])

        W_old = interp_w_linear(vel.w, z_levels, z_p, nn, alive, 1.0)
        W_new = interp_w_spatial(
            vel.w, z_levels, z_p, lons, lats, nn, alive, 1.0,
            method="nearest", mesh=mesh,
        )
        np.testing.assert_array_almost_equal(W_new, W_old, decimal=14)


# ---------------------------------------------------------------------------
# Test barycentric specifics
# ---------------------------------------------------------------------------

class TestBarycentric:
    def test_at_vertex(self):
        """Particle exactly at a node should return that node's W."""
        mesh = _make_mesh()
        z_levels = np.abs(mesh.sigma)
        n_z = z_levels.shape[0]
        # Different W per node
        w_arr = np.zeros((n_z, mesh.n_nodes), dtype=np.float64)
        for n in range(mesh.n_nodes):
            w_arr[:, n] = 0.001 * (n + 1)

        vel = _make_vel_3d(mesh, w_values=w_arr)
        node_idx = 4  # center
        nn = np.array([node_idx], dtype=np.int32)
        alive = np.array([True])
        z_p = np.array([5.0])  # surface-ish
        lons = np.array([mesh.nodes_lon[node_idx]])
        lats = np.array([mesh.nodes_lat[node_idx]])

        W = interp_w_spatial(
            vel.w, z_levels, z_p, lons, lats, nn, alive, 1.0,
            method="barycentric", mesh=mesh,
        )
        expected = 0.001 * (node_idx + 1)
        assert W[0] == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# Test IDW specifics
# ---------------------------------------------------------------------------

class TestIDW:
    def test_on_node(self):
        """Particle exactly on a node should return that node's W (no div-by-zero)."""
        mesh = _make_mesh()
        z_levels = np.abs(mesh.sigma)
        n_z = z_levels.shape[0]
        w_arr = np.zeros((n_z, mesh.n_nodes), dtype=np.float64)
        for n in range(mesh.n_nodes):
            w_arr[:, n] = 0.001 * (n + 1)

        vel = _make_vel_3d(mesh, w_values=w_arr)
        node_idx = 4
        nn = np.array([node_idx], dtype=np.int32)
        alive = np.array([True])
        z_p = np.array([5.0])
        lons = np.array([mesh.nodes_lon[node_idx]])
        lats = np.array([mesh.nodes_lat[node_idx]])

        W = interp_w_spatial(
            vel.w, z_levels, z_p, lons, lats, nn, alive, 1.0,
            method="idw", mesh=mesh,
        )
        expected = 0.001 * (node_idx + 1)
        assert W[0] == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# Test bottom-aware clamping
# ---------------------------------------------------------------------------

class TestBottomAware:
    """Verify that shallow neighbours don't contribute below-bottom W."""

    def test_varying_bathymetry(self):
        """Mesh with varying bottom depth — below-bottom W should not leak."""
        mesh = _make_mesh(bottom_depth=200.0)
        z_levels = np.abs(mesh.sigma)
        n_z = z_levels.shape[0]

        # Make one neighbour shallow (bottom at 30m)
        mesh.depth[0] = 30.0  # corner node

        # Set below-bottom W to a distinctive value
        w_arr = np.full((n_z, mesh.n_nodes), 0.001, dtype=np.float64)
        # Below 30m at node 0: set to large distinctive value
        for k in range(n_z):
            if z_levels[k] > 30.0:
                w_arr[k, 0] = 999.0

        vel = _make_vel_3d(mesh, w_values=w_arr)
        nn = np.array([4], dtype=np.int32)
        alive = np.array([True])
        z_p = np.array([80.0])  # particle at 80m
        lons = np.array([5.0])
        lats = np.array([5.0])

        # With bottom-aware clamping, the 999 values should not appear
        for method in ["leastsq", "idw"]:
            W = interp_w_spatial(
                vel.w, z_levels, z_p, lons, lats, nn, alive, 1.0,
                method=method, mesh=mesh,
            )
            assert abs(W[0]) < 1.0, (
                f"method={method}: W={W[0]}, "
                f"bottom-aware clamping not working"
            )


# ---------------------------------------------------------------------------
# Test boundary fallback
# ---------------------------------------------------------------------------

class TestBoundaryFallback:
    """LSQ should fall back when too few neighbours."""

    def test_leastsq_still_returns_value(self):
        """Even at boundary nodes with fewer neighbours, should get a value."""
        mesh = _make_mesh()
        z_levels = np.abs(mesh.sigma)
        vel = _make_vel_3d(mesh, w_values=0.005)
        # Corner node
        nn = np.array([0], dtype=np.int32)
        alive = np.array([True])
        z_p = np.array([30.0])
        lons = np.array([mesh.nodes_lon[0]])
        lats = np.array([mesh.nodes_lat[0]])

        W = interp_w_spatial(
            vel.w, z_levels, z_p, lons, lats, nn, alive, 1.0,
            method="leastsq", mesh=mesh,
        )
        # Should still return something reasonable (either fit or fallback)
        assert W[0] == pytest.approx(0.005, abs=1e-6)


# ---------------------------------------------------------------------------
# Test time_dir (backward tracking)
# ---------------------------------------------------------------------------

class TestTimeDir:
    def test_backward_flips_sign(self):
        mesh = _make_mesh()
        z_levels = np.abs(mesh.sigma)
        vel = _make_vel_3d(mesh, w_values=0.005)
        nn = np.array([4], dtype=np.int32)
        alive = np.array([True])
        z_p = np.array([30.0])
        lons = np.array([5.0])
        lats = np.array([5.0])

        W_fwd = interp_w_spatial(
            vel.w, z_levels, z_p, lons, lats, nn, alive, 1.0,
            method="leastsq", mesh=mesh,
        )
        W_bwd = interp_w_spatial(
            vel.w, z_levels, z_p, lons, lats, nn, alive, -1.0,
            method="leastsq", mesh=mesh,
        )
        assert W_bwd[0] == pytest.approx(-W_fwd[0], abs=1e-14)


# ---------------------------------------------------------------------------
# Test invalid method
# ---------------------------------------------------------------------------

def test_invalid_method():
    mesh = _make_mesh()
    z_levels = np.abs(mesh.sigma)
    vel = _make_vel_3d(mesh, w_values=0.005)
    with pytest.raises(ValueError, match="Unknown w_interp method"):
        interp_w_spatial(
            vel.w, z_levels, np.array([30.0]),
            np.array([5.0]), np.array([5.0]),
            np.array([4], dtype=np.int32),
            np.array([True]), 1.0,
            method="nonexistent", mesh=mesh,
        )


# ---------------------------------------------------------------------------
# Integration test: 3D tracking with each W interp method
# ---------------------------------------------------------------------------

class TestIntegration3D:
    """End-to-end 3D tracking with different W interpolation methods."""

    @pytest.fixture
    def setup(self):
        mesh = _make_mesh()
        # Constant downward W, small horizontal u
        vel = _make_vel_3d(mesh, w_values=0.01, u=0.5)
        state = make_state(
            np.array([5.0]), np.array([5.0]), np.array([30.0])
        )
        return mesh, vel, state

    @pytest.mark.parametrize("method", ["nearest", "leastsq", "barycentric", "idw"])
    def test_euler_3d(self, setup, method):
        mesh, vel, state = setup
        for _ in range(5):
            state = euler_step(state, vel, mesh, dt=60.0, time_dir=1.0,
                               w_interp=method)
        assert state.status[0] == ParticleStatus.ALIVE
        # Particle should have moved down (W > 0 = downwelling)
        assert state.z[0] > 30.0

    @pytest.mark.parametrize("method", ["nearest", "leastsq"])
    def test_rk2_3d(self, setup, method):
        mesh, vel, state = setup
        for _ in range(5):
            state = rk2_step(state, vel, mesh, dt=60.0, time_dir=1.0,
                             w_interp=method)
        assert state.status[0] == ParticleStatus.ALIVE
        assert state.z[0] > 30.0

    @pytest.mark.parametrize("method", ["nearest", "leastsq"])
    def test_rk4_3d(self, setup, method):
        mesh, vel, state = setup
        for _ in range(5):
            state = rk4_step(state, vel, mesh, dt=60.0, time_dir=1.0,
                             w_interp=method)
        assert state.status[0] == ParticleStatus.ALIVE
        assert state.z[0] > 30.0


# ---------------------------------------------------------------------------
# Config propagation test
# ---------------------------------------------------------------------------

class TestConfigPropagation:
    def test_w_interp_in_numerics(self):
        from kelp.core.config import NumericsConfig
        cfg = NumericsConfig(w_interp="idw")
        assert cfg.w_interp == "idw"

    def test_default_is_leastsq(self):
        from kelp.core.config import NumericsConfig
        cfg = NumericsConfig()
        assert cfg.w_interp == "leastsq"


# ---------------------------------------------------------------------------
# Dead particles should be skipped
# ---------------------------------------------------------------------------

class TestDeadParticles:
    def test_dead_particle_zero_w(self):
        mesh = _make_mesh()
        z_levels = np.abs(mesh.sigma)
        vel = _make_vel_3d(mesh, w_values=0.005)
        nn = np.array([4, 4], dtype=np.int32)
        alive = np.array([True, False])
        z_p = np.array([30.0, 30.0])
        lons = np.array([5.0, 5.0])
        lats = np.array([5.0, 5.0])

        for method in ["nearest", "leastsq", "barycentric", "idw"]:
            W = interp_w_spatial(
                vel.w, z_levels, z_p, lons, lats, nn, alive, 1.0,
                method=method, mesh=mesh,
            )
            assert W[0] != 0.0  # alive particle gets W
            assert W[1] == 0.0  # dead particle stays at 0


# ---------------------------------------------------------------------------
# Dateline safety tests
# ---------------------------------------------------------------------------

class TestDateline:
    """Ensure spatial W interpolation works near ±180° longitude."""

    def test_dateline_constant_w(self):
        """All methods agree on constant W near dateline."""
        mesh = _make_mesh(center_lon=179.8, center_lat=5.0)
        z_levels = np.abs(mesh.sigma)
        W_CONST = 0.003
        vel = _make_vel_3d(mesh, w_values=W_CONST)
        nn = np.array([4], dtype=np.int32)
        alive = np.array([True])
        z_p = np.array([30.0])
        lons = np.array([179.8])
        lats = np.array([5.0])

        for method in ["nearest", "leastsq", "barycentric", "idw"]:
            W = interp_w_spatial(
                vel.w, z_levels, z_p, lons, lats, nn, alive, 1.0,
                method=method, mesh=mesh,
            )
            assert W[0] == pytest.approx(W_CONST, abs=1e-8), (
                f"method={method}: W={W[0]}, expected={W_CONST}"
            )

    def test_dateline_linear_gradient(self):
        """LSQ should handle linear gradient near dateline correctly."""
        mesh = _make_mesh(center_lon=179.8, center_lat=5.0)
        z_levels = np.abs(mesh.sigma)
        n_z = z_levels.shape[0]

        a, b, c = 0.01, -0.005, 0.002
        w_arr = np.empty((n_z, mesh.n_nodes), dtype=np.float64)
        for k in range(n_z):
            for n in range(mesh.n_nodes):
                w_arr[k, n] = a * mesh.nodes_lon[n] + b * mesh.nodes_lat[n] + c

        vel = _make_vel_3d(mesh, w_values=w_arr)
        xp, yp = 179.8, 5.0
        expected = a * xp + b * yp + c
        nn = np.array([4], dtype=np.int32)
        alive = np.array([True])
        z_p = np.array([30.0])
        lons = np.array([xp])
        lats = np.array([yp])

        W = interp_w_spatial(
            vel.w, z_levels, z_p, lons, lats, nn, alive, 1.0,
            method="leastsq", mesh=mesh,
        )
        assert W[0] == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# High latitude tests
# ---------------------------------------------------------------------------

class TestHighLatitude:
    """Ensure spatial W interpolation is stable near poles."""

    def test_high_latitude_constant_w(self):
        """All methods agree on constant W at high latitude (z0 very small)."""
        mesh = _make_mesh(center_lon=10.0, center_lat=89.5, span=0.3)
        z_levels = np.abs(mesh.sigma)
        W_CONST = 0.004
        vel = _make_vel_3d(mesh, w_values=W_CONST)
        nn = np.array([4], dtype=np.int32)
        alive = np.array([True])
        z_p = np.array([30.0])
        lons = np.array([10.0])
        lats = np.array([89.5])

        for method in ["nearest", "leastsq", "barycentric", "idw"]:
            W = interp_w_spatial(
                vel.w, z_levels, z_p, lons, lats, nn, alive, 1.0,
                method=method, mesh=mesh,
            )
            assert W[0] == pytest.approx(W_CONST, abs=1e-8), (
                f"method={method}: W={W[0]}, expected={W_CONST}"
            )


# ---------------------------------------------------------------------------
# Pure-quad mesh — Wachspress barycentric weights (Task 3)
# ---------------------------------------------------------------------------

from kelp.core.mesh import find_nearest_nodes


def _fan_weights_ref(px, py, vx, vy, nv):
    """In-test fan-triangulation reference — the pre-2026-07 fallback
    behaviour (the production `_fan_weights` was removed together with
    the guard chain; quads now always take product-form Wachspress)."""
    for t in range(nv - 2):
        ix = (0, t + 1, t + 2)
        ax, ay = vx[ix[0]], vy[ix[0]]
        bx, by = vx[ix[1]], vy[ix[1]]
        cx, cy = vx[ix[2]], vy[ix[2]]
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-10:
            continue
        w1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / den
        w2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / den
        w3 = 1.0 - w1 - w2
        if w1 >= -1e-6 and w2 >= -1e-6 and w3 >= -1e-6:
            out = np.zeros(nv)
            out[ix[0]] += w1
            out[ix[1]] += w2
            out[ix[2]] += w3
            return out
    return np.full(nv, -1.0)


def _make_quad_mesh(center_lon=5.0, center_lat=0.0, span=1.0,
                    z_levels=None, bottom_depth=200.0):
    """Build a pure-QUAD 2x2-cell mesh: 9 nodes, 4 quad elements.

    ``elem_nodes`` is (4, 4) with all four unique vertices per quad,
    wound counter-clockwise (SW, SE, NE, NW).  Centred at the equator so
    cos(lat) ~ 1 keeps the local metric projection nearly isotropic.
    """
    if z_levels is None:
        z_levels = [0.0, 10.0, 25.0, 50.0, 100.0, 200.0]

    xs = np.linspace(center_lon - span, center_lon + span, 3)
    ys = np.linspace(center_lat - span, center_lat + span, 3)
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel().astype(np.float64)
    lat = gy.ravel().astype(np.float64)
    n_nodes = lon.shape[0]
    depth = np.full(n_nodes, bottom_depth, dtype=np.float64)

    # 4 quads, CCW: SW, SE, NE, NW
    elems = []
    for j in range(2):
        for i in range(2):
            n_sw = j * 3 + i
            n_se = n_sw + 1
            n_ne = n_sw + 3 + 1
            n_nw = n_sw + 3
            elems.append([n_sw, n_se, n_ne, n_nw])
    elem_nodes = np.array(elems, dtype=np.int32)

    lon_c = np.array([lon[e].mean() for e in elem_nodes], dtype=np.float64)
    lat_c = np.array([lat[e].mean() for e in elem_nodes], dtype=np.float64)
    n2e, n2e_count = build_node_to_elems(n_nodes, elem_nodes)
    sigma = np.array(z_levels, dtype=np.float64)

    mesh = Mesh(
        name="test_quad",
        nodes_lon=lon, nodes_lat=lat, depth=depth,
        elem_nodes=elem_nodes,
        elem_center_lon=lon_c, elem_center_lat=lat_c,
        node_to_elems=n2e, node_to_elems_count=n2e_count,
        sigma=sigma,
    )
    build_search(mesh)
    return mesh


def _kernel_projection(mesh, elem, lon_p, lat_p):
    """Replicate the barycentric W kernel's centroid-relative projection.

    Returns ``(verts, vx, vy, px, py)`` exactly as the njit kernel builds
    them so an in-test fan reference is geometrically bit-identical.
    """
    R = 6371000.0
    en = mesh.elem_nodes[elem]
    verts = []
    for nd in en:
        nd = int(nd)
        if nd < 0:
            continue
        if verts and nd == verts[0]:
            continue
        verts.append(nd)
    verts = np.array(verts, dtype=np.int32)
    lon0 = mesh.nodes_lon[verts].mean()
    lat0 = mesh.nodes_lat[verts].mean()
    cos_lat0 = np.cos(lat0 * DEG2RAD)
    dlon = ((mesh.nodes_lon[verts] - lon0 + 180.0) % 360.0) - 180.0
    dlat = mesh.nodes_lat[verts] - lat0
    vx = dlon * DEG2RAD * 6371000.0 * cos_lat0
    vy = dlat * DEG2RAD * 6371000.0
    dlon_p = ((lon_p - lon0 + 180.0) % 360.0) - 180.0
    px = dlon_p * DEG2RAD * 6371000.0 * cos_lat0
    py = (lat_p - lat0) * DEG2RAD * 6371000.0
    return verts, vx, vy, px, py


class TestBarycentricQuadWachspress:
    """Quad barycentric W now uses Wachspress weights (all four vertices
    contribute, vertex-order independent, smooth across the old fan
    diagonal).  Tri path is unchanged (covered elsewhere)."""

    def _w_field_from_node_vals(self, mesh, node_vals):
        """Tile a per-node value array across all K interfaces (no z
        variation -> vertical interp returns the node value unchanged)."""
        K = mesh.sigma.shape[0]
        w = np.empty((K, mesh.n_nodes), dtype=np.float64)
        for k in range(K):
            w[k, :] = node_vals
        return w

    def test_linear_field_exact_inside_quad(self):
        # GREEN before AND after the change: a globally linear field is
        # reproduced exactly by BOTH the fan and Wachspress.  Kept as a
        # post-change regression property (Wachspress linear reproduction).
        mesh = _make_quad_mesh()
        z_levels = np.abs(mesh.sigma)
        node_vals = 2.0 + 3.0 * mesh.nodes_lon
        w_field = self._w_field_from_node_vals(mesh, node_vals)

        lon_probe, lat_probe = 5.6, 0.3   # interior of an upper-right quad
        nn = find_nearest_nodes(mesh, np.array([lon_probe]),
                                np.array([lat_probe]))
        out = interp_w_spatial(
            w_field, z_levels, np.array([5.0]),
            np.array([lon_probe]), np.array([lat_probe]),
            nn, np.array([True]), 1.0,
            method="barycentric", mesh=mesh, node_depths=mesh.depth,
        )
        np.testing.assert_allclose(out[0], 2.0 + 3.0 * lon_probe, rtol=1e-9)

    def test_fourth_vertex_participates(self):
        # RED under the fan: a field nonzero ONLY at a vertex that the fan
        # excludes inside one sub-triangle.  Fan from vertex 0 (SW) splits
        # the quad along the SW->NE diagonal into (SW,SE,NE) and (SW,NE,NW);
        # the NW vertex (slot 3) has weight exactly 0 in the lower
        # sub-triangle (SW,SE,NE).  Wachspress gives NW weight > 0.
        mesh = _make_quad_mesh()
        z_levels = np.abs(mesh.sigma)

        # Lower-left quad (element 0): vertices SW,SE,NE,NW (slots 0,1,2,3).
        elem = 0
        verts = mesh.elem_nodes[elem]
        excluded = int(verts[3])  # NW — zeroed by the fan in sub-tri (SW,SE,NE)

        node_vals = np.zeros(mesh.n_nodes, dtype=np.float64)
        node_vals[excluded] = 1.0
        w_field = self._w_field_from_node_vals(mesh, node_vals)

        # Probe inside the lower fan sub-triangle (SW,SE,NE), below the
        # SW->NE diagonal (verified against the in-test fan reference).
        lon_probe, lat_probe = 4.4, -0.75
        nn = find_nearest_nodes(mesh, np.array([lon_probe]),
                                np.array([lat_probe]))
        out = interp_w_spatial(
            w_field, z_levels, np.array([5.0]),
            np.array([lon_probe]), np.array([lat_probe]),
            nn, np.array([True]), 1.0,
            method="barycentric", mesh=mesh, node_depths=mesh.depth,
        )
        verts_k, vx, vy, px, py = _kernel_projection(
            mesh, elem, lon_probe, lat_probe)
        w_fan = _fan_weights_ref(px, py, vx, vy, len(verts_k))
        assert w_fan[0] != -1.0, "probe must be inside element 0"
        nw_slot = int(np.where(verts_k == excluded)[0][0])
        assert w_fan[nw_slot] == 0.0, (
            "probe must sit in the fan sub-triangle that excludes NW "
            "for this to be RED under the fan"
        )
        assert out[0] > 1e-3, (
            f"Wachspress must give the excluded vertex weight; W={out[0]}"
        )

    def test_renumbering_invariance(self):
        # RED under the fan with a BILINEAR (non-affine) field: cyclically
        # rotating one quad's vertex order changes the fan diagonal and
        # thus the interpolated value; Wachspress is invariant.
        mesh_a = _make_quad_mesh()
        node_vals = mesh_a.nodes_lon * mesh_a.nodes_lat   # bilinear
        w_a = self._w_field_from_node_vals(mesh_a, node_vals)

        # mesh_b: identical geometry, but roll element 3's vertex order by 1.
        mesh_b = _make_quad_mesh()
        en_b = mesh_b.elem_nodes.copy()
        en_b[3] = np.roll(en_b[3], 1)
        mesh_b.elem_nodes = en_b
        mesh_b.elem_center_lon = np.array(
            [mesh_b.nodes_lon[e].mean() for e in mesh_b.elem_nodes],
            dtype=np.float64)
        mesh_b.elem_center_lat = np.array(
            [mesh_b.nodes_lat[e].mean() for e in mesh_b.elem_nodes],
            dtype=np.float64)
        n2e, n2e_count = build_node_to_elems(mesh_b.n_nodes, mesh_b.elem_nodes)
        mesh_b.node_to_elems = n2e
        mesh_b.node_to_elems_count = n2e_count
        build_search(mesh_b)
        w_b = self._w_field_from_node_vals(mesh_b, node_vals)

        # Probe inside element 3 (upper-right quad), off-centre so the
        # diagonal choice matters.
        lon_probe, lat_probe = 5.6, 0.55
        nn_a = find_nearest_nodes(mesh_a, np.array([lon_probe]),
                                  np.array([lat_probe]))
        nn_b = find_nearest_nodes(mesh_b, np.array([lon_probe]),
                                  np.array([lat_probe]))
        out_a = interp_w_spatial(
            w_a, np.abs(mesh_a.sigma), np.array([5.0]),
            np.array([lon_probe]), np.array([lat_probe]),
            nn_a, np.array([True]), 1.0,
            method="barycentric", mesh=mesh_a, node_depths=mesh_a.depth,
        )
        out_b = interp_w_spatial(
            w_b, np.abs(mesh_b.sigma), np.array([5.0]),
            np.array([lon_probe]), np.array([lat_probe]),
            nn_b, np.array([True]), 1.0,
            method="barycentric", mesh=mesh_b, node_depths=mesh_b.depth,
        )
        np.testing.assert_allclose(out_a[0], out_b[0], rtol=1e-10)

    def test_smooth_across_old_diagonal(self):
        # RED under the fan: a bilinear field has a first-derivative kink
        # along the fan's arbitrary diagonal.  Compare SECOND DIFFERENCES
        # of the production (Wachspress) path against an in-test fan
        # reference computed in the SAME metre projection.  RATIO, not an
        # absolute threshold: |d2| scales with the field magnitude and the
        # probe spacing h (an absolute bound silently passed BOTH methods
        # when first written in the wrong coordinate system — review
        # finding), while the fan/Wachspress ratio cancels both factors.
        mesh = _make_quad_mesh()
        z_levels = np.abs(mesh.sigma)
        node_vals = mesh.nodes_lon * mesh.nodes_lat   # bilinear
        w_field = self._w_field_from_node_vals(mesh, node_vals)

        # Segment crossing the old fan diagonal (SW->NE) of element 0
        # TRANSVERSALLY (SE->NW corner-to-corner), so the fan's
        # first-derivative kink along the diagonal is sampled.  A segment
        # PARALLEL to the diagonal would see no fan kink and the test would
        # be meaningless.  (lower-left quad: lon in [4,5], lat in [-1,0]).
        N = 9
        lon_seg = np.linspace(4.85, 4.15, N)
        lat_seg = np.linspace(-0.85, -0.15, N)

        nn = find_nearest_nodes(mesh, lon_seg, lat_seg)
        alive = np.ones(N, dtype=bool)
        w_prod = interp_w_spatial(
            w_field, z_levels, np.full(N, 5.0),
            lon_seg, lat_seg, nn, alive, 1.0,
            method="barycentric", mesh=mesh, node_depths=mesh.depth,
        )

        # In-test fan reference: same projection, fan weights, same field.
        w_fan_ref = np.empty(N, dtype=np.float64)
        for i in range(N):
            verts, vx, vy, px, py = _kernel_projection(
                mesh, 0, lon_seg[i], lat_seg[i])
            wts = _fan_weights_ref(px, py, vx, vy, len(verts))
            assert wts[0] != -1.0, "probe fell outside element 0"
            w_fan_ref[i] = float(np.dot(wts, node_vals[verts]))

        def max_abs_d2(f):
            d2 = f[2:] - 2.0 * f[1:-1] + f[:-2]
            return float(np.max(np.abs(d2)))

        d2_w = max_abs_d2(w_prod)
        d2_f = max_abs_d2(w_fan_ref)
        assert d2_w < d2_f / 5.0, (
            f"Wachspress should be far smoother across the old diagonal: "
            f"max|d2_wachspress|={d2_w:.3e}, max|d2_fan|={d2_f:.3e}"
        )

    def test_mixed_mesh_tri_bit_identical_quad_linear(self):
        # Mixed (E,4) table: 1 quad + 2 -1-padded tris in ONE mesh.
        # Probe in a TRI element matches the inline tri math bit-for-bit;
        # probe in the QUAD matches the analytic linear field.
        z_levels = [0.0, 10.0, 25.0, 50.0, 100.0, 200.0]
        sigma = np.array(z_levels, dtype=np.float64)

        # 6 nodes, two stacked rows (lon, lat):
        #   0:(4,-0.5) 1:(5,-0.5) 2:(6,-0.5)
        #   3:(4, 0.5) 4:(5, 0.5) 5:(6, 0.5)
        lon = np.array([4., 5., 6., 4., 5., 6.], dtype=np.float64)
        lat = np.array([-0.5, -0.5, -0.5, 0.5, 0.5, 0.5], dtype=np.float64)
        depth = np.full(6, 200.0, dtype=np.float64)

        # elem 0: quad (0,1,4,3) CCW.  elem 1,2: tris padded with -1.
        elem_nodes = np.array([
            [0, 1, 4, 3],      # quad
            [1, 2, 4, -1],     # tri
            [2, 5, 4, -1],     # tri
        ], dtype=np.int32)
        lon_c = np.array(
            [lon[e[e >= 0]].mean() for e in elem_nodes], dtype=np.float64)
        lat_c = np.array(
            [lat[e[e >= 0]].mean() for e in elem_nodes], dtype=np.float64)
        n2e, n2e_count = build_node_to_elems(6, elem_nodes)
        mesh = Mesh(
            name="mixed", nodes_lon=lon, nodes_lat=lat, depth=depth,
            elem_nodes=elem_nodes, elem_center_lon=lon_c, elem_center_lat=lat_c,
            node_to_elems=n2e, node_to_elems_count=n2e_count, sigma=sigma,
        )
        build_search(mesh)

        # Linear field f = 1 + 2*lon - 0.5*lat (exact for tri AND quad).
        a, b, c = 1.0, 2.0, -0.5
        node_vals = a + b * lon + c * lat
        w_field = self._w_field_from_node_vals(mesh, node_vals)
        zl = np.abs(sigma)

        # --- TRI probe (inside element 1, the lower-right tri) ---
        lon_t, lat_t = 5.5, -0.15
        nn_t = find_nearest_nodes(mesh, np.array([lon_t]), np.array([lat_t]))
        out_t = interp_w_spatial(
            w_field, zl, np.array([5.0]), np.array([lon_t]), np.array([lat_t]),
            nn_t, np.array([True]), 1.0,
            method="barycentric", mesh=mesh, node_depths=mesh.depth,
        )
        # Tri reference via classical barycentric (== fan single sub-tri).
        verts_t, vx_t, vy_t, px_t, py_t = _kernel_projection(
            mesh, 1, lon_t, lat_t)
        w_tri_ref = _fan_weights_ref(px_t, py_t, vx_t, vy_t, 3)
        ref_t = float(np.dot(w_tri_ref, node_vals[verts_t]))
        assert out_t[0] == ref_t, (
            f"tri path must be bit-identical: {out_t[0]!r} != {ref_t!r}"
        )
        np.testing.assert_allclose(out_t[0], a + b * lon_t + c * lat_t,
                                   rtol=1e-9)

        # --- QUAD probe (inside element 0) ---
        lon_q, lat_q = 4.6, 0.2
        nn_q = find_nearest_nodes(mesh, np.array([lon_q]), np.array([lat_q]))
        out_q = interp_w_spatial(
            w_field, zl, np.array([5.0]), np.array([lon_q]), np.array([lat_q]),
            nn_q, np.array([True]), 1.0,
            method="barycentric", mesh=mesh, node_depths=mesh.depth,
        )
        np.testing.assert_allclose(out_q[0], a + b * lon_q + c * lat_q,
                                   rtol=1e-9)


# ---------------------------------------------------------------------------
# High latitude tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TODO 6.4: bracket precompute — bit-identity pins
# ---------------------------------------------------------------------------

class TestBracketPrecompute:
    """The bracket (k, alpha) is computed once per particle and reused for
    every unclamped neighbour; bottom-clamped neighbours get their own.
    Everything below pins BIT-identity with the pre-6.4 per-neighbour
    ``_interp_w_linear_single`` evaluation."""

    def test_bracket_composition_matches_linear_single(self):
        from kelp.numerics.vertical import (
            _interp_w_linear_single, _w_bracket, _w_at_bracket)
        rng = np.random.default_rng(1)
        K, N = 9, 4
        zl = np.sort(rng.uniform(1.0, 500.0, K))
        zl[0] = 0.0
        # Degenerate layer: 9e-13 survives rounding at this magnitude
        # (ulp(zl[4]) ~ 2.8e-14; the earlier +1e-14 rounded back to the
        # SAME double and the dz<1e-12 sentinel was never exercised —
        # review 2026-08-15) and still sits under the 1e-12 threshold.
        zl[5] = zl[4] + 9e-13
        assert zl[5] > zl[4]                       # probe is self-verifying
        degen_probe = zl[4] + 4.5e-13              # interior of the sliver
        assert _w_bracket(zl, float(degen_probe)) == (4, -1.0)
        w = rng.standard_normal((K, N))
        probes = np.concatenate([
            [-5.0, 0.0, zl[-1], zl[-1] + 10.0, zl[4], zl[5], degen_probe,
             zl[3]],
            rng.uniform(0.0, zl[-1], 60),
        ])
        for z in probes:
            k, a = _w_bracket(zl, float(z))
            for node in range(N):
                for td in (1.0, -1.0):
                    ref = _interp_w_linear_single(w, zl, float(z), node, td)
                    got = _w_at_bracket(w, node, k, a, td)
                    # bit-compare (== would let a ±0.0 drift pass)
                    assert (np.float64(got).view(np.uint64)
                            == np.float64(ref).view(np.uint64)), (z, node, td)

    def test_bracket_composition_single_interface_column(self):
        """K < 2: every depth resolves to the single interface."""
        from kelp.numerics.vertical import (
            _interp_w_linear_single, _w_bracket, _w_at_bracket)
        zl = np.array([0.0])
        w = np.array([[3.5, -1.25]])
        for z in (-1.0, 0.0, 5.0):
            k, a = _w_bracket(zl, z)
            for node in range(2):
                for td in (1.0, -1.0):
                    ref = _interp_w_linear_single(w, zl, z, node, td)
                    got = _w_at_bracket(w, node, k, a, td)
                    assert (np.float64(got).view(np.uint64)
                            == np.float64(ref).view(np.uint64))

    def _setup(self, seed=0, P=300):
        rng = np.random.default_rng(seed)
        K = 12
        zl = np.linspace(0.0, 1000.0, K)
        mesh = _make_mesh(5.0, 5.0, span=1.0, nx=7, ny=7, z_levels=zl,
                          bottom_depth=1000.0)
        # MIXED bottoms so both the shared-bracket and the clamped-
        # neighbour branches run (incl. exact-boundary cases below)
        mesh.depth[:] = rng.uniform(150.0, 1000.0, mesh.n_nodes)
        w = rng.standard_normal((K, mesh.n_nodes))
        lons = rng.uniform(4.2, 5.8, P)
        lats = rng.uniform(4.2, 5.8, P)
        zs = rng.uniform(0.0, 900.0, P)
        nn = find_nearest_nodes(mesh, lons, lats)
        # Exact-boundary probes that are GUARANTEED to hit a stencil
        # neighbour: pin some particles' depth to their own nearest
        # node's bottom (the earlier node-i/particle-i pairing only hit
        # by coincidence of the random placement — review 2026-08-15).
        for i in range(0, P, 10):
            zs[i] = mesh.depth[nn[i]]
        return mesh, w, zl, lons, lats, zs, nn

    @staticmethod
    def _frozen_vlerp(w, zl, z, node, td):
        """TRULY frozen pre-6.4 vertical lerp (no live imports — a
        future co-edit of ``_interp_w_linear_single`` must not silently
        re-base this reference; review 2026-08-15)."""
        K = zl.shape[0]
        if K < 2:
            return td * w[0, node]
        if z <= zl[0]:
            return td * w[0, node]
        if z >= zl[K - 1]:
            return td * w[K - 1, node]
        k = K - 2
        for j in range(K - 1):
            if z < zl[j + 1]:
                k = j
                break
        dz = zl[k + 1] - zl[k]
        if dz < 1e-12:
            return td * w[k, node]
        alpha = (z - zl[k]) / dz
        return td * ((1.0 - alpha) * w[k, node] + alpha * w[k + 1, node])

    @staticmethod
    def _frozen_leastsq(xp, yp, z, node_idx, w, zl, td, mesh):
        """FROZEN python copy of the pre-6.4 ``_interp_w_leastsq_single``
        (characterization reference; the ring collection stays a live
        import — it is topology shared by both sides, not the math
        under test)."""
        from kelp.numerics.vertical import _collect_neighbour_nodes
        _interp_w_linear_single = TestBracketPrecompute._frozen_vlerp
        SLM = 111120.0
        DEG2R = 0.017453292519943295
        buf, count = _collect_neighbour_nodes(
            node_idx, mesh.node_to_elems, mesh.node_to_elems_count,
            mesh.elem_nodes)
        nd_depth = mesh.depth
        if count < 3:
            zc = min(z, nd_depth[node_idx])
            return _interp_w_linear_single(w, zl, zc, node_idx, td)
        z0 = max(np.cos(yp * DEG2R) * SLM, 1.0)
        a11 = a12 = a13 = a22 = a23 = a33 = 0.0
        fw = f1w = f2w = 0.0
        for i in range(count):
            nd = buf[i]
            z_eval = min(z, nd_depth[nd])
            w_val = _interp_w_linear_single(w, zl, z_eval, nd, td)
            dlon = ((mesh.nodes_lon[nd] - xp + 180.0) % 360.0) - 180.0
            ex = dlon * z0
            ey = (mesh.nodes_lat[nd] - yp) * SLM
            a11 += 1.0; a12 += ex; a13 += ey
            a22 += ex * ex; a23 += ex * ey; a33 += ey * ey
            fw += w_val; f1w += w_val * ex; f2w += w_val * ey
        a21, a31, a32 = a12, a13, a23
        det = (a11 * a22 * a33 + a12 * a23 * a31 + a13 * a21 * a32
               - a13 * a22 * a31 - a12 * a21 * a33 - a11 * a23 * a32)
        if abs(det) < 1e-30:
            zc = min(z, nd_depth[node_idx])
            return _interp_w_linear_single(w, zl, zc, node_idx, td)
        inv = 1.0 / abs(det)
        b11 = a22 * a33 - a23 * a32
        b12 = a13 * a32 - a12 * a33
        b13 = a12 * a23 - a13 * a22
        return (b11 * fw + b12 * f1w + b13 * f2w) * inv

    @staticmethod
    def _frozen_idw(xp, yp, z, node_idx, w, zl, td, mesh):
        """FROZEN python copy of the pre-6.4 ``_interp_w_idw_single``."""
        from kelp.numerics.vertical import _collect_neighbour_nodes
        _interp_w_linear_single = TestBracketPrecompute._frozen_vlerp
        SLM = 111120.0
        DEG2R = 0.017453292519943295
        EPS = 1.0
        buf, count = _collect_neighbour_nodes(
            node_idx, mesh.node_to_elems, mesh.node_to_elems_count,
            mesh.elem_nodes)
        if count < 1:
            zc = min(z, mesh.depth[node_idx])
            return _interp_w_linear_single(w, zl, zc, node_idx, td)
        z0 = max(np.cos(yp * DEG2R) * SLM, 1.0)
        sum_w = sum_wv = 0.0
        for i in range(count):
            nd = buf[i]
            z_eval = min(z, mesh.depth[nd])
            w_val = _interp_w_linear_single(w, zl, z_eval, nd, td)
            dlon = ((mesh.nodes_lon[nd] - xp + 180.0) % 360.0) - 180.0
            dx = dlon * z0
            dy = (mesh.nodes_lat[nd] - yp) * SLM
            d2 = dx * dx + dy * dy
            if d2 < EPS * EPS:
                return w_val
            weight = 1.0 / d2
            sum_w += weight
            sum_wv += weight * w_val
        if sum_w < 1e-30:
            zc = min(z, mesh.depth[node_idx])
            return _interp_w_linear_single(w, zl, zc, node_idx, td)
        return sum_wv / sum_w

    @pytest.mark.parametrize("method,frozen", [
        ("leastsq", "_frozen_leastsq"), ("idw", "_frozen_idw")])
    @pytest.mark.parametrize("td", [1.0, -1.0])
    def test_kernels_bitwise_vs_frozen_reference(self, method, frozen, td):
        from kelp.numerics.vertical import interp_w_spatial
        mesh, w, zl, lons, lats, zs, nn = self._setup()
        alive = np.ones(lons.size, dtype=bool)
        got = interp_w_spatial(w, zl, zs, lons, lats, nn, alive, td,
                               method, mesh, mesh.depth)
        ref_fn = getattr(self, frozen)
        for i in range(lons.size):
            ref = ref_fn(lons[i], lats[i], zs[i], nn[i], w, zl, td, mesh)
            # bit-compare (== would let a ±0.0 drift pass)
            assert (np.float64(got[i]).view(np.uint64)
                    == np.float64(ref).view(np.uint64)), i
