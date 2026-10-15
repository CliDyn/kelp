"""Unit tests for the sigma-coordinate vertical backend.

Covers column geometry, layer detection, depth clamping and
interface-based W interpolation on small synthetic fields.  No file I/O.
"""
from __future__ import annotations

import numpy as np
import pytest

from kelp.numerics.vertical_sigma import (
    clamp_depth_sigma,
    column_depth,
    find_layer_sigma,
    find_layers_sigma,
    interp_w_sigma_linear,
    sigma_interfaces_to_z,
)


# FESOM-C style: sigma=1 at surface, sigma=0 at bottom, 11 interfaces.
SIGMA_LEV = np.array(
    [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# column_depth
# ---------------------------------------------------------------------------

class TestColumnDepth:
    def test_without_eta(self):
        assert column_depth(100.0, 0.0, 0.5) == pytest.approx(100.0)

    def test_with_positive_eta(self):
        assert column_depth(100.0, 0.5, 0.5) == pytest.approx(100.5)

    def test_with_negative_eta(self):
        assert column_depth(100.0, -0.3, 0.5) == pytest.approx(99.7)

    def test_dry_cell_clipped_to_dmin(self):
        # depth 0.1 m + eta 0 below Dmin=0.5 → clipped
        assert column_depth(0.1, 0.0, 0.5) == pytest.approx(0.5)

    def test_negative_after_eta_clipped(self):
        # unlikely, but protect against eta pushing column negative
        assert column_depth(0.2, -0.8, 0.5) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# sigma_interfaces_to_z
# ---------------------------------------------------------------------------

class TestSigmaInterfacesToZ:
    def test_surface_and_bottom(self):
        z = sigma_interfaces_to_z(SIGMA_LEV, H_col=1000.0)
        assert z[0] == pytest.approx(0.0)
        assert z[-1] == pytest.approx(1000.0)

    def test_layer_center_matches_fesomc_Z(self):
        """Reproduces layer centres from the real medi.nc file.

        For H=1497.77m, Z[0] is 74.88m (sigma=[1,0.9] midpoint),
        Z[-1] is 1422.88m.  These values were measured in-file on
        node 50000 and are a regression anchor for the sigma transform.
        """
        H = 1497.77
        z_ifc = sigma_interfaces_to_z(SIGMA_LEV, H)
        z_center = 0.5 * (z_ifc[:-1] + z_ifc[1:])
        assert z_center[0] == pytest.approx(74.8885, abs=1e-3)
        assert z_center[-1] == pytest.approx(1422.8819, abs=1e-3)

    def test_monotonic_surface_to_bottom(self):
        z = sigma_interfaces_to_z(SIGMA_LEV, 500.0)
        assert np.all(np.diff(z) > 0)  # strictly increasing


# ---------------------------------------------------------------------------
# find_layer_sigma (single)
# ---------------------------------------------------------------------------

class TestFindLayerSigma:
    def test_surface_particle(self):
        assert find_layer_sigma(SIGMA_LEV, 0.0, 1000.0) == 0

    def test_just_below_surface_first_layer(self):
        assert find_layer_sigma(SIGMA_LEV, 10.0, 1000.0) == 0

    def test_in_first_layer_upper_bound(self):
        # first layer spans [0, 100) for H=1000
        assert find_layer_sigma(SIGMA_LEV, 99.0, 1000.0) == 0

    def test_second_layer_lower_bound(self):
        assert find_layer_sigma(SIGMA_LEV, 100.1, 1000.0) == 1

    def test_middle_of_column(self):
        # 500m in a 1000m column → layer index ~5 (interface at 500m is k=5)
        # 499m falls in layer 4, 501m in layer 5
        assert find_layer_sigma(SIGMA_LEV, 499.0, 1000.0) == 4
        assert find_layer_sigma(SIGMA_LEV, 501.0, 1000.0) == 5

    def test_bottom_clamp(self):
        # at bottom or deeper → max layer K-2 = 9
        assert find_layer_sigma(SIGMA_LEV, 1000.0, 1000.0) == 9
        assert find_layer_sigma(SIGMA_LEV, 1050.0, 1000.0) == 9

    def test_negative_z_clamps_to_top(self):
        assert find_layer_sigma(SIGMA_LEV, -10.0, 1000.0) == 0

    def test_degenerate_two_interface_column(self):
        # K=2 → one layer, always returns 0 within the column
        sigma = np.array([1.0, 0.0], dtype=np.float64)
        assert find_layer_sigma(sigma, 0.0, 100.0) == 0
        assert find_layer_sigma(sigma, 50.0, 100.0) == 0
        assert find_layer_sigma(sigma, 100.0, 100.0) == 0


# ---------------------------------------------------------------------------
# find_layers_sigma (vectorised)
# ---------------------------------------------------------------------------

class TestFindLayersSigma:
    def test_vector_matches_single(self):
        node_depths = np.array([1000.0, 500.0, 100.0], dtype=np.float64)
        eta_nodes = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        nn = np.array([0, 1, 2], dtype=np.int32)
        z = np.array([50.0, 250.0, 50.0], dtype=np.float64)
        alive = np.array([True, True, True])

        layers = find_layers_sigma(SIGMA_LEV, z, nn, node_depths, eta_nodes,
                                    alive, Dmin=0.5)
        # H=1000, z=50 → layer 0
        # H=500,  z=250 → layer 5 (midpoint)
        # H=100,  z=50  → layer 5 (midpoint)
        assert layers[0] == 0
        assert layers[1] == 5
        assert layers[2] == 5

    def test_eta_shifts_column_depth(self):
        """Adding eta > 0 must increase the effective column, shifting layers."""
        node_depths = np.array([100.0], dtype=np.float64)
        nn = np.array([0], dtype=np.int32)
        z = np.array([95.0], dtype=np.float64)
        alive = np.array([True])

        # Without eta: H=100, z=95 falls in layer 9 (bottom-most)
        eta0 = np.array([0.0])
        assert find_layers_sigma(SIGMA_LEV, z, nn, node_depths, eta0, alive)[0] == 9

        # With eta=+50: H=150, z=95 → layer 6 (midway)
        eta_up = np.array([50.0])
        layer = find_layers_sigma(SIGMA_LEV, z, nn, node_depths, eta_up, alive)[0]
        assert layer == 6

    def test_dead_particles_return_zero(self):
        node_depths = np.array([1000.0], dtype=np.float64)
        eta_nodes = np.array([0.0], dtype=np.float64)
        nn = np.array([0], dtype=np.int32)
        z = np.array([500.0], dtype=np.float64)
        alive = np.array([False])

        layers = find_layers_sigma(SIGMA_LEV, z, nn, node_depths, eta_nodes,
                                    alive, Dmin=0.5)
        assert layers[0] == 0

    def test_dmin_protects_shallow_nodes(self):
        # Node depth 0.1 is below Dmin=0.5 → effective H=0.5
        # Particle at z=0.3 should be inside the column, layer 5 (middle of H=0.5)
        node_depths = np.array([0.1], dtype=np.float64)
        eta_nodes = np.array([0.0], dtype=np.float64)
        nn = np.array([0], dtype=np.int32)
        z = np.array([0.3], dtype=np.float64)
        alive = np.array([True])

        layers = find_layers_sigma(SIGMA_LEV, z, nn, node_depths, eta_nodes,
                                    alive, Dmin=0.5)
        # H=0.5, z=0.3 → fractional depth 0.6 → layer 6
        assert layers[0] == 6


# ---------------------------------------------------------------------------
# clamp_depth_sigma
# ---------------------------------------------------------------------------

class TestClampDepthSigma:
    def test_surface_clamp(self):
        z = np.array([-5.0, 0.0, 10.0])
        nn = np.array([0, 0, 0], dtype=np.int32)
        node_depths = np.array([100.0])
        eta = np.array([0.0])
        alive = np.array([True, True, True])

        out = clamp_depth_sigma(z, nn, node_depths, eta, alive,
                                 Dmin=0.5, z0b_min=0.1)
        assert out[0] == 0.0   # surface clamp
        assert out[1] == 0.0
        assert out[2] == 10.0  # unchanged

    def test_bottom_clamp_uses_z0b_min(self):
        z = np.array([100.0, 95.0, 99.95])
        nn = np.array([0, 0, 0], dtype=np.int32)
        node_depths = np.array([100.0])
        eta = np.array([0.0])
        alive = np.array([True, True, True])

        out = clamp_depth_sigma(z, nn, node_depths, eta, alive,
                                 Dmin=0.5, z0b_min=0.1)
        # bottom is 100 - 0.1 = 99.9
        assert out[0] == pytest.approx(99.9)
        assert out[1] == 95.0   # unchanged
        assert out[2] == pytest.approx(99.9)

    def test_eta_raises_bottom(self):
        # eta=+5 → H=105, bottom at 104.9
        z = np.array([103.0, 106.0])
        nn = np.array([0, 0], dtype=np.int32)
        node_depths = np.array([100.0])
        eta = np.array([5.0])
        alive = np.array([True, True])

        out = clamp_depth_sigma(z, nn, node_depths, eta, alive,
                                 Dmin=0.5, z0b_min=0.1)
        assert out[0] == 103.0   # within column
        assert out[1] == pytest.approx(104.9)

    def test_dead_particles_pass_through(self):
        z = np.array([-100.0, 1000.0])
        nn = np.array([0, 0], dtype=np.int32)
        node_depths = np.array([50.0])
        eta = np.array([0.0])
        alive = np.array([False, False])

        out = clamp_depth_sigma(z, nn, node_depths, eta, alive)
        assert out[0] == -100.0
        assert out[1] == 1000.0


# ---------------------------------------------------------------------------
# interp_w_sigma_linear
# ---------------------------------------------------------------------------

class TestInterpWSigmaLinear:
    @staticmethod
    def _uniform_w_field(value, K=11, N=4):
        return np.full((K, N), value, dtype=np.float64)

    def test_uniform_field(self):
        """Any bracket of a uniform field returns that same value."""
        w = self._uniform_w_field(0.01)
        node_depths = np.array([1000.0] * 4)
        eta = np.zeros(4)
        nn = np.array([0, 1, 2, 3], dtype=np.int32)
        z = np.array([10.0, 500.0, 999.0, 0.0])
        alive = np.ones(4, dtype=bool)

        W = interp_w_sigma_linear(w, SIGMA_LEV, z, nn, node_depths, eta, alive)
        np.testing.assert_allclose(W, 0.01)

    def test_linear_gradient_in_column(self):
        """W increasing linearly with depth should be reproduced exactly
        at interface positions."""
        K = 11
        N = 1
        w = np.zeros((K, N))
        for k in range(K):
            w[k, 0] = float(k) * 0.1  # w[0]=0, w[10]=1.0

        node_depths = np.array([1000.0])
        eta = np.array([0.0])
        nn = np.array([0], dtype=np.int32)
        alive = np.array([True])

        # At each interface depth, W must equal w[k, 0] exactly
        for k in range(K):
            z_at_ifc = (1.0 - SIGMA_LEV[k]) * 1000.0
            W = interp_w_sigma_linear(w, SIGMA_LEV, np.array([z_at_ifc]),
                                       nn, node_depths, eta, alive)
            assert W[0] == pytest.approx(w[k, 0], abs=1e-9)

    def test_midway_between_interfaces(self):
        """At the midpoint of [z_ifc[k], z_ifc[k+1]], W should be the
        arithmetic mean of w[k] and w[k+1]."""
        K = 11
        N = 1
        w = np.zeros((K, N))
        w[3, 0] = 0.0
        w[4, 0] = 1.0

        node_depths = np.array([1000.0])
        eta = np.array([0.0])
        nn = np.array([0], dtype=np.int32)
        alive = np.array([True])

        # layer 3 spans [300, 400); midpoint = 350
        W = interp_w_sigma_linear(w, SIGMA_LEV, np.array([350.0]),
                                   nn, node_depths, eta, alive)
        assert W[0] == pytest.approx(0.5, abs=1e-9)

    def test_bottom_zero_preserved(self):
        """When bottom interface stores 0 (FESOM-C no-flow BC), particles
        clamped to the bottom see W=0."""
        K = 11
        N = 1
        w = np.ones((K, N)) * 0.05
        w[-1, 0] = 0.0  # bottom BC

        node_depths = np.array([1000.0])
        eta = np.array([0.0])
        nn = np.array([0], dtype=np.int32)
        alive = np.array([True])

        W = interp_w_sigma_linear(w, SIGMA_LEV, np.array([1000.0]),
                                   nn, node_depths, eta, alive)
        assert W[0] == pytest.approx(0.0)

    def test_time_dir_negates(self):
        w = self._uniform_w_field(0.02)
        node_depths = np.array([1000.0] * 4)
        eta = np.zeros(4)
        nn = np.array([0, 1, 2, 3], dtype=np.int32)
        z = np.array([100.0] * 4)
        alive = np.ones(4, dtype=bool)

        W_fwd = interp_w_sigma_linear(w, SIGMA_LEV, z, nn, node_depths, eta,
                                       alive, time_dir=1.0)
        W_bwd = interp_w_sigma_linear(w, SIGMA_LEV, z, nn, node_depths, eta,
                                       alive, time_dir=-1.0)
        np.testing.assert_allclose(W_fwd, 0.02)
        np.testing.assert_allclose(W_bwd, -0.02)

    def test_eta_changes_column_height(self):
        """Raising eta expands the column; a particle at fixed z that was
        near the bottom now sits higher up and samples a different W."""
        K = 11
        N = 1
        w = np.zeros((K, N))
        for k in range(K):
            w[k, 0] = float(k) * 0.1

        node_depths = np.array([100.0])
        nn = np.array([0], dtype=np.int32)
        alive = np.array([True])
        z = np.array([95.0])

        # Without eta: H=100, z=95 → near bottom, W ≈ w[9..10] ≈ 0.9..1.0
        eta0 = np.array([0.0])
        W0 = interp_w_sigma_linear(w, SIGMA_LEV, z, nn, node_depths, eta0,
                                    alive)[0]
        # With eta=+50: H=150, z=95 → 63% down → ≈ w[6..7] ≈ 0.6..0.7
        eta_up = np.array([50.0])
        W_up = interp_w_sigma_linear(w, SIGMA_LEV, z, nn, node_depths, eta_up,
                                      alive)[0]

        assert W0 > W_up
        assert 0.9 <= W0 <= 1.0
        assert 0.6 <= W_up <= 0.7


# ---------------------------------------------------------------------------
# Horizontal W interpolation dispatcher (leastsq / barycentric / idw)
# ---------------------------------------------------------------------------

from kelp.numerics.vertical_sigma import interp_w_sigma_spatial
from kelp.core.mesh import Mesh, build_node_to_elems, build_search


def _fan_weights_ref(px, py, vx, vy, nv):
    """In-test fan-triangulation reference — the pre-2026-07 fallback
    behaviour (the production `_fan_weights` was removed with the guard
    chain; quads now always take product-form Wachspress)."""
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


def _tiny_triangular_mesh():
    """3x3 node grid at the equator (cos(lat)=1), 8 triangles, depth=100m."""
    xs = np.array([4.0, 5.0, 6.0])
    ys = np.array([-1.0, 0.0, 1.0])
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel().astype(np.float64)
    lat = gy.ravel().astype(np.float64)
    depth = np.full(9, 100.0, dtype=np.float64)

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
        name="sigma_test",
        nodes_lon=lon, nodes_lat=lat, depth=depth,
        elem_nodes=elem_nodes, elem_center_lon=lon_c, elem_center_lat=lat_c,
        node_to_elems=n2e, node_to_elems_count=n2e_count,
        sigma=SIGMA_LEV.copy(), vertical_coord="sigma",
    )
    build_search(mesh)
    return mesh


class TestHorizontalMethodsUniformField:
    """With uniform W every method must give the same answer as ``nearest``."""

    @pytest.mark.parametrize("method",
                              ["nearest", "leastsq", "barycentric", "idw"])
    def test_uniform_field_returns_constant(self, method):
        mesh = _tiny_triangular_mesh()
        K = mesh.sigma.shape[0]
        N = mesh.n_nodes
        w = np.full((K, N), 0.02, dtype=np.float64)
        eta = np.zeros(N, dtype=np.float64)

        lons = np.array([5.2, 4.7, 5.5])
        lats = np.array([0.1, -0.4, 0.3])
        z = np.array([20.0, 30.0, 10.0])
        from kelp.core.mesh import find_nearest_nodes
        nn = find_nearest_nodes(mesh, lons, lats)
        alive = np.ones(3, dtype=bool)

        W = interp_w_sigma_spatial(
            w, mesh.sigma, z, lons, lats, nn, alive,
            mesh.depth, eta, Dmin=0.5, time_dir=1.0,
            method=method, mesh=mesh,
        )
        np.testing.assert_allclose(W, 0.02)


class TestHorizontalMethodsLinearField:
    """With a purely horizontal gradient W(x) = a * lon, leastsq must
    reproduce it exactly; others should give reasonable approximations."""

    def _horizontal_gradient_field(self, mesh, K=11, a=0.001):
        # W = a * lon at every interface (no vertical gradient)
        w = np.empty((K, mesh.n_nodes), dtype=np.float64)
        for k in range(K):
            w[k, :] = a * mesh.nodes_lon
        return w

    def test_leastsq_reproduces_linear_gradient_exactly(self):
        mesh = _tiny_triangular_mesh()
        a = 0.001
        w = self._horizontal_gradient_field(mesh, K=mesh.sigma.shape[0], a=a)
        eta = np.zeros(mesh.n_nodes, dtype=np.float64)

        lons = np.array([4.85, 5.15, 5.47])
        lats = np.array([0.1, -0.2, 0.3])
        z = np.array([40.0, 40.0, 40.0])
        from kelp.core.mesh import find_nearest_nodes
        nn = find_nearest_nodes(mesh, lons, lats)
        alive = np.ones(3, dtype=bool)

        W = interp_w_sigma_spatial(
            w, mesh.sigma, z, lons, lats, nn, alive,
            mesh.depth, eta, Dmin=0.5, time_dir=1.0,
            method="leastsq", mesh=mesh,
        )
        # Exact value is a * lon
        np.testing.assert_allclose(W, a * lons, atol=1e-10)

    def test_idw_weighted_near_node(self):
        """IDW at a point very close to a node returns ~that node's value."""
        mesh = _tiny_triangular_mesh()
        a = 0.01
        w = self._horizontal_gradient_field(mesh, K=mesh.sigma.shape[0], a=a)
        eta = np.zeros(mesh.n_nodes, dtype=np.float64)

        # Particle ~ at node 4 (5.0, 0.0)
        lons = np.array([5.0 + 1e-4])
        lats = np.array([0.0 - 1e-4])
        z = np.array([40.0])
        from kelp.core.mesh import find_nearest_nodes
        nn = find_nearest_nodes(mesh, lons, lats)
        alive = np.array([True])

        W = interp_w_sigma_spatial(
            w, mesh.sigma, z, lons, lats, nn, alive,
            mesh.depth, eta, Dmin=0.5, time_dir=1.0,
            method="idw", mesh=mesh,
        )
        # Node 4 value = a * 5.0 = 0.05
        assert W[0] == pytest.approx(0.05, rel=0.01)


class TestSigmaEquivalentsZLevelOnConstantH:
    """Flat bottom + eta=0 ⇒ sigma interfaces coincide with constant z-levels;
    sigma's interpolated W must equal the z-level equivalent."""

    def test_leastsq_matches_zlevel_flat_bottom(self):
        from kelp.numerics.vertical import interp_w_spatial as zlvl_spatial

        mesh = _tiny_triangular_mesh()
        # Flat 100 m bottom + 11 equally spaced z-level interfaces = same
        # geometry as sigma = [1, 0.9, ..., 0] with H=100 everywhere.
        z_levels = np.linspace(0.0, 100.0, 11)
        K = len(z_levels)
        N = mesh.n_nodes

        # Non-trivial field: vertical gradient + horizontal gradient
        w_cart = np.zeros((K, N))
        for k in range(K):
            for n in range(N):
                w_cart[k, n] = 0.002 * mesh.nodes_lon[n] + 0.0001 * z_levels[k]

        eta = np.zeros(N, dtype=np.float64)
        lons = np.array([4.85, 5.25, 5.5, 4.6])
        lats = np.array([0.15, -0.3, 0.4, -0.2])
        z = np.array([20.0, 55.0, 35.0, 80.0])
        from kelp.core.mesh import find_nearest_nodes
        nn = find_nearest_nodes(mesh, lons, lats)
        alive = np.ones(4, dtype=bool)

        # z-level result using the original backend
        zlvl_mesh = Mesh(
            name="zlvl",
            nodes_lon=mesh.nodes_lon, nodes_lat=mesh.nodes_lat,
            depth=mesh.depth, elem_nodes=mesh.elem_nodes,
            elem_center_lon=mesh.elem_center_lon,
            elem_center_lat=mesh.elem_center_lat,
            node_to_elems=mesh.node_to_elems,
            node_to_elems_count=mesh.node_to_elems_count,
            sigma=z_levels, vertical_coord="zlevel",
        )
        build_search(zlvl_mesh)
        W_zlvl = zlvl_spatial(
            w_cart, z_levels, z, lons, lats, nn, alive,
            time_dir=1.0, method="leastsq", mesh=zlvl_mesh,
        )

        W_sigma = interp_w_sigma_spatial(
            w_cart, mesh.sigma, z, lons, lats, nn, alive,
            mesh.depth, eta, Dmin=0.5, time_dir=1.0,
            method="leastsq", mesh=mesh,
        )

        np.testing.assert_allclose(W_sigma, W_zlvl, atol=1e-9)


def _tiny_quad_mesh():
    """2x2 quad mesh (4 nodes, 1 quad, depth=100m) — pure FESOM-C layout.

    Quad connectivity is stored in the (E, 4) ``elem_nodes`` array with
    all 4 unique vertices (no repeat).
    """
    lon = np.array([4.0, 5.0, 5.0, 4.0], dtype=np.float64)
    lat = np.array([-0.5, -0.5, 0.5, 0.5], dtype=np.float64)
    depth = np.full(4, 100.0, dtype=np.float64)

    # One quad: CCW ordering n0=SW, n1=SE, n2=NE, n3=NW
    elem_nodes = np.array([[0, 1, 2, 3]], dtype=np.int32)
    lon_c = np.array([lon[elem_nodes[0]].mean()], dtype=np.float64)
    lat_c = np.array([lat[elem_nodes[0]].mean()], dtype=np.float64)
    n2e, n2e_count = build_node_to_elems(4, elem_nodes)

    mesh = Mesh(
        name="sigma_quad_test",
        nodes_lon=lon, nodes_lat=lat, depth=depth,
        elem_nodes=elem_nodes, elem_center_lon=lon_c, elem_center_lat=lat_c,
        node_to_elems=n2e, node_to_elems_count=n2e_count,
        sigma=SIGMA_LEV.copy(), vertical_coord="sigma",
    )
    build_search(mesh)
    return mesh


class TestHorizontalMethodsQuadMesh:
    """Quad-mesh smoke coverage — FESOM-C stores mixed tri/quad elements
    and barycentric needs fan triangulation to handle quads."""

    @pytest.mark.parametrize("method",
                              ["nearest", "leastsq", "barycentric", "idw"])
    def test_uniform_field_on_quad(self, method):
        mesh = _tiny_quad_mesh()
        K = mesh.sigma.shape[0]
        N = mesh.n_nodes
        w = np.full((K, N), 0.015, dtype=np.float64)
        eta = np.zeros(N, dtype=np.float64)

        lons = np.array([4.5, 4.25, 4.75])   # inside the quad
        lats = np.array([0.0, -0.3, 0.3])
        z = np.array([20.0, 60.0, 80.0])
        from kelp.core.mesh import find_nearest_nodes
        nn = find_nearest_nodes(mesh, lons, lats)
        alive = np.ones(3, dtype=bool)

        W = interp_w_sigma_spatial(
            w, mesh.sigma, z, lons, lats, nn, alive,
            mesh.depth, eta, Dmin=0.5, time_dir=1.0,
            method=method, mesh=mesh,
        )
        np.testing.assert_allclose(W, 0.015, err_msg=f"method={method}")

    def test_barycentric_linear_field_on_quad(self):
        """Barycentric on a quad uses fan triangulation; a purely linear
        horizontal field must be recovered exactly everywhere inside."""
        mesh = _tiny_quad_mesh()
        a = 0.01
        K = mesh.sigma.shape[0]
        N = mesh.n_nodes
        w = np.empty((K, N), dtype=np.float64)
        for k in range(K):
            w[k, :] = a * mesh.nodes_lon
        eta = np.zeros(N, dtype=np.float64)

        lons = np.array([4.25, 4.75, 4.5])
        lats = np.array([0.1, -0.2, 0.0])
        z = np.array([40.0, 40.0, 40.0])
        from kelp.core.mesh import find_nearest_nodes
        nn = find_nearest_nodes(mesh, lons, lats)
        alive = np.ones(3, dtype=bool)

        W = interp_w_sigma_spatial(
            w, mesh.sigma, z, lons, lats, nn, alive,
            mesh.depth, eta, Dmin=0.5, time_dir=1.0,
            method="barycentric", mesh=mesh,
        )
        np.testing.assert_allclose(W, a * lons, atol=1e-8)

    def test_barycentric_bilinear_field_matches_wachspress(self):
        """NON-affine (bilinear) field: this is the case where the old fan
        and Wachspress genuinely differ.  The expectation is computed
        in-test from the Wachspress weights (never copied from output),
        replicating the kernel's centroid-relative metre projection."""
        from kelp.core.mesh import (
            _polygon_weights, find_nearest_nodes,
        )
        DEG2R = 0.017453292519943295

        mesh = _tiny_quad_mesh()
        K = mesh.sigma.shape[0]
        N = mesh.n_nodes
        # Bilinear vertex field f = lon*lat, tiled across all interfaces
        # (no vertical gradient -> the per-vertex column value is f itself
        # for a particle near the surface, depth uniform = 100 m).
        node_vals = mesh.nodes_lon * mesh.nodes_lat
        w = np.empty((K, N), dtype=np.float64)
        for k in range(K):
            w[k, :] = node_vals
        eta = np.zeros(N, dtype=np.float64)

        # Off-centre probe (the only quad is element 0, vertices 0,1,2,3).
        lon_p, lat_p = 4.7, 0.2
        nn = find_nearest_nodes(mesh, np.array([lon_p]), np.array([lat_p]))
        alive = np.array([True])
        W = interp_w_sigma_spatial(
            w, mesh.sigma, np.array([5.0]), np.array([lon_p]),
            np.array([lat_p]), nn, alive,
            mesh.depth, eta, Dmin=0.5, time_dir=1.0,
            method="barycentric", mesh=mesh,
        )

        # In-test Wachspress expectation: replicate the kernel projection.
        verts = mesh.elem_nodes[0]
        R = 6371000.0
        lon0 = mesh.nodes_lon[verts].mean()
        lat0 = mesh.nodes_lat[verts].mean()
        cos_lat0 = np.cos(lat0 * DEG2R)
        dlon = ((mesh.nodes_lon[verts] - lon0 + 180.0) % 360.0) - 180.0
        vx = dlon * DEG2R * R * cos_lat0
        vy = (mesh.nodes_lat[verts] - lat0) * DEG2R * R
        dlon_p = ((lon_p - lon0 + 180.0) % 360.0) - 180.0
        px = dlon_p * DEG2R * R * cos_lat0
        py = (lat_p - lat0) * DEG2R * R
        wts = _polygon_weights(px, py, vx, vy, 4)
        expected = float(np.dot(wts, node_vals[verts]))

        np.testing.assert_allclose(W[0], expected, rtol=1e-9)
        # And it must DIFFER from the old fan result (proving the change).
        wts_fan = _fan_weights_ref(px, py, vx, vy, 4)
        fan_val = float(np.dot(wts_fan, node_vals[verts]))
        assert not np.isclose(W[0], fan_val, rtol=1e-6), (
            "bilinear field: Wachspress must differ from the fan result"
        )
