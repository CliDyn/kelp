"""Tests for polygon vertex weights: tri barycentric + quad Wachspress.

The triangle primitive must stay bit-identical to the original code;
quads use PRODUCT-FORM Wachspress coordinates (division-free, so exact
on edges and vertices — no sentinel, no fan fallback; convexity of quads
is a mesh contract, reviewer guidance 2026-07).  All functions operate
in a local metric plane (metres).
"""
import numpy as np
import pytest

from kelp.core.mesh import (
    _barycentric_weights_tri, _wachspress_weights_quad, _polygon_weights,
)


def _unit_square():
    # CCW unit square: v0=(0,0) v1=(1,0) v2=(1,1) v3=(0,1)
    return (np.array([0.0, 1.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0, 1.0]))


def _random_convex_quad(rng):
    # Perturbed CCW square — stays convex for perturbation < 0.3
    base_x = np.array([0.0, 1.0, 1.0, 0.0])
    base_y = np.array([0.0, 0.0, 1.0, 1.0])
    return (base_x + rng.uniform(-0.25, 0.25, 4),
            base_y + rng.uniform(-0.25, 0.25, 4))


class TestWachspressQuad:
    def test_partition_of_unity_and_positivity(self):
        rng = np.random.default_rng(42)
        for _ in range(50):
            x, y = _random_convex_quad(rng)
            # strictly interior point: blend of vertices
            px, py = x.mean() + 0.05, y.mean() - 0.03
            w = _wachspress_weights_quad(px, py, x, y)
            assert np.isfinite(w).all()
            np.testing.assert_allclose(w.sum(), 1.0, rtol=1e-12)
            assert (w > 0).all()

    def test_linear_reproduction(self):
        # sum(w_i * v_i) == p  =>  any linear field is exact
        rng = np.random.default_rng(7)
        for _ in range(50):
            x, y = _random_convex_quad(rng)
            px, py = x.mean() + 0.07, y.mean() + 0.02
            w = _wachspress_weights_quad(px, py, x, y)
            np.testing.assert_allclose(w @ x, px, atol=1e-10)
            np.testing.assert_allclose(w @ y, py, atol=1e-10)

    def test_all_four_vertices_contribute(self):
        x, y = _unit_square()
        # point inside the OLD fan triangle (0,1,2): under the fan,
        # vertex 3 had weight exactly 0; Wachspress gives it weight > 0
        w = _wachspress_weights_quad(0.6, 0.3, x, y)
        assert w[3] > 1e-3

    def test_cyclic_renumbering_invariance(self):
        x, y = _unit_square()
        px, py = 0.37, 0.71
        w0 = _wachspress_weights_quad(px, py, x, y)
        for shift in (1, 2, 3):
            xs, ys = np.roll(x, shift), np.roll(y, shift)
            ws = _wachspress_weights_quad(px, py, xs, ys)
            np.testing.assert_allclose(np.roll(w0, shift), ws, atol=1e-12)

    def test_clockwise_input_matches_ccw(self):
        x, y = _unit_square()
        px, py = 0.37, 0.71
        w_ccw = _wachspress_weights_quad(px, py, x, y)
        # reverse order -> CW winding of the same square
        w_cw = _wachspress_weights_quad(px, py, x[::-1].copy(),
                                        y[::-1].copy())
        np.testing.assert_allclose(w_cw, w_ccw[::-1], atol=1e-12)

    def test_centre_of_square_is_quarter_each(self):
        x, y = _unit_square()
        w = _wachspress_weights_quad(0.5, 0.5, x, y)
        np.testing.assert_allclose(w, 0.25, atol=1e-12)

    def test_on_edge_reduces_to_linear_edge_interpolation(self):
        # The product form has no on-edge singularity: a point ON edge
        # (v0, v1) gets exactly the 1-D linear interpolation of that edge
        # (off-edge vertices weight 0) — no sentinel, no fallback.
        x, y = _unit_square()
        w = _wachspress_weights_quad(0.5, 0.0, x, y)
        np.testing.assert_allclose(w, [0.5, 0.5, 0.0, 0.0], atol=1e-14)
        w = _wachspress_weights_quad(0.25, 0.0, x, y)
        np.testing.assert_allclose(w, [0.75, 0.25, 0.0, 0.0], atol=1e-14)

    def test_on_vertex_gives_indicator(self):
        x, y = _unit_square()
        for k in range(4):
            w = _wachspress_weights_quad(float(x[k]), float(y[k]), x, y)
            expected = np.zeros(4)
            expected[k] = 1.0
            np.testing.assert_allclose(w, expected, atol=1e-14)


class TestPolygonWeightsDispatcher:
    def test_tri_path_is_bit_identical(self):
        # nv=3 must route to the UNTOUCHED triangle primitive
        rng = np.random.default_rng(3)
        for _ in range(25):
            x = rng.uniform(0, 1, 3); y = rng.uniform(0, 1, 3)
            px, py = float(x.mean()), float(y.mean())
            w_direct = _barycentric_weights_tri(px, py, x, y)
            w_disp = _polygon_weights(px, py, x, y, 3)
            assert (w_direct == w_disp[:3]).all()   # BIT-identical

    def test_tri_path_bit_identical_outside_too(self):
        # review pin: the tri primitive returns RAW (possibly negative)
        # weights for exterior points, while fan routing would return the
        # all -1 sentinel — the dispatcher must pass the primitive's
        # output through untouched even there (distinguishes a dispatcher
        # that quietly reroutes nv==3 through _fan_weights).
        x = np.array([0.0, 1.0, 0.0])
        y = np.array([0.0, 0.0, 1.0])
        w_direct = _barycentric_weights_tri(0.8, 0.8, x, y)
        assert (w_direct < 0).any()              # genuinely exterior
        w_disp = _polygon_weights(0.8, 0.8, x, y, 3)
        assert (w_direct == w_disp[:3]).all()

    def test_near_edge_band_is_smooth_no_dead_band(self):
        # The old division form had a scale-aware dead band (~2e-10 for a
        # unit square) where it bailed to fan weights; the product form
        # gives finite, smooth weights arbitrarily close to the edge,
        # converging to the on-edge linear interpolation.
        x, y = _unit_square()
        for eps in (1e-9, 1e-12, 1e-15):
            w = _wachspress_weights_quad(0.5, eps, x, y)
            assert np.isfinite(w).all()
            np.testing.assert_allclose(w.sum(), 1.0, rtol=1e-12)
            np.testing.assert_allclose(w, [0.5, 0.5, 0.0, 0.0],
                                       atol=10 * eps + 1e-14)

    def test_quad_path_uses_wachspress(self):
        x, y = _unit_square()
        w = _polygon_weights(0.6, 0.3, x, y, 4)
        np.testing.assert_allclose(
            w, _wachspress_weights_quad(0.6, 0.3, x, y), atol=0)

    def test_quad_slightly_outside_band_stays_finite(self):
        # The containment tolerance (_BARY_EPS / _WALK_EPS ~ 1e-6
        # normalized) can accept points marginally OUTSIDE the quad; the
        # old division form always tripped its guard there (fan fallback,
        # which could even sentinel and falsely kill the particle).  The
        # product form must give finite sum-1 weights, only mildly
        # negative, still reproducing linear fields (review 2026-07-03).
        rng = np.random.default_rng(11)
        for _ in range(20):
            x, y = _random_convex_quad(rng)
            # point just below the (v0, v1) edge midpoint, ~1e-7 of cell
            mx, my = (x[0] + x[1]) / 2, (y[0] + y[1]) / 2
            ex, ey = x[1] - x[0], y[1] - y[0]
            L = np.hypot(ex, ey)
            nx_, ny_ = ey / L, -ex / L      # outward normal (CCW quad)
            px, py = mx + 1e-7 * nx_, my + 1e-7 * ny_
            w = _wachspress_weights_quad(px, py, x, y)
            assert np.isfinite(w).all()
            np.testing.assert_allclose(w.sum(), 1.0, rtol=1e-9)
            assert w.min() > -1e-3
            np.testing.assert_allclose(w @ x, px, atol=1e-9)
            np.testing.assert_allclose(w @ y, py, atol=1e-9)

    def test_edge_point_through_dispatcher(self):
        x, y = _unit_square()
        w = _polygon_weights(0.5, 0.0, x, y, 4)
        np.testing.assert_allclose(w, [0.5, 0.5, 0.0, 0.0], atol=1e-14)
        np.testing.assert_allclose(w.sum(), 1.0, rtol=1e-12)

    def test_cross_element_edge_continuity(self):
        # two quads sharing edge x=1: a point ON the shared edge gets the
        # same value from both cells (product-form Wachspress reduces to
        # the 1-D linear interpolation along the edge from either side)
        xL, yL = _unit_square()
        xR, yR = xL + 1.0, yL.copy()
        f_L = np.array([10.0, 20.0, 30.0, 40.0])   # values at L vertices
        # shared edge: L vertices 1,2 == R vertices 0,3
        f_R = np.array([20.0, 99.0, 77.0, 30.0])
        py = 0.25
        wL = _polygon_weights(1.0, py, xL, yL, 4)
        wR = _polygon_weights(1.0, py, xR, yR, 4)
        np.testing.assert_allclose(wL @ f_L, wR @ f_R, rtol=1e-12)


class TestPolygonContainsQuadWeights:
    def test_quad_weights_are_wachspress(self):
        from kelp.core.mesh import _polygon_contains
        # unit square in lon/lat degrees near the equator (the local
        # metric projection is affine here, Wachspress is projection-
        # invariant for affine maps)
        lon = np.array([0.0, 0.1, 0.1, 0.0])
        lat = np.array([0.0, 0.0, 0.1, 0.1])
        w = _polygon_contains(lon, lat, 0.06, 0.03)
        assert w is not None
        # all four vertices contribute (fan gave w[3] == 0 here)
        assert w[3] > 1e-3
        np.testing.assert_allclose(w.sum(), 1.0, rtol=1e-12)
        # linear reproduction in lon/lat (affine-invariance)
        np.testing.assert_allclose(w @ lon, 0.06, atol=1e-9)
        np.testing.assert_allclose(w @ lat, 0.03, atol=1e-9)

    def test_tri_weights_unchanged(self):
        from kelp.core.mesh import _polygon_contains
        lon = np.array([0.0, 0.1, 0.05])
        lat = np.array([0.0, 0.0, 0.1])
        w = _polygon_contains(lon, lat, 0.05, 0.03)
        assert w is not None and w.shape == (3,)
        np.testing.assert_allclose(w.sum(), 1.0, rtol=1e-12)

    def test_outside_still_none(self):
        from kelp.core.mesh import _polygon_contains
        lon = np.array([0.0, 0.1, 0.1, 0.0])
        lat = np.array([0.0, 0.0, 0.1, 0.1])
        assert _polygon_contains(lon, lat, 0.5, 0.5) is None


class TestContainmentBoolEquivalence:
    def test_bool_matches_weights_version(self):
        # The boolean twin (_polygon_contains_bool, used on the hot
        # containment path) must agree with (weights is not None) EXACTLY,
        # including the _BARY_EPS edge band, so swapping it in is behaviour-
        # invariant.  500 random points in [-0.05, 0.15]^2 — spanning the
        # interior, all edges and the exterior — against a tri and a quad.
        from kelp.core.mesh import _polygon_contains, _polygon_contains_bool

        # Same fixtures as the _polygon_contains tests above.
        tri_lon = np.array([0.0, 0.1, 0.05])
        tri_lat = np.array([0.0, 0.0, 0.1])
        quad_lon = np.array([0.0, 0.1, 0.1, 0.0])
        quad_lat = np.array([0.0, 0.0, 0.1, 0.1])

        rng = np.random.default_rng(20260605)
        pts = rng.uniform(-0.05, 0.15, size=(500, 2))

        for lon_n, lat_n in ((tri_lon, tri_lat), (quad_lon, quad_lat)):
            for px, py in pts:
                px = float(px)
                py = float(py)
                expected = _polygon_contains(lon_n, lat_n, px, py) is not None
                got = _polygon_contains_bool(lon_n, lat_n, px, py)
                assert got == expected, (
                    f"mismatch at ({px}, {py}) for "
                    f"{'tri' if lon_n.shape[0] == 3 else 'quad'}: "
                    f"bool={got} weights_is_not_None={expected}"
                )
