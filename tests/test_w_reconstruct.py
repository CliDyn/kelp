"""Tests for kelp.w_reconstruct — W velocity reconstruction.

Tests cover:
- Sign convention (positive-down)
- Analytical divergence on a simple mesh
- ALE heuristic
- Bottom correction
- Irregular time intervals
- Variant C cross-validation diagnostics
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from kelp.core.mesh import Mesh, build_node_to_elems, build_search
from kelp.w_reconstruct import (
    compute_ssh_tendency,
    compute_divergence_approx,
    integrate_continuity,
    reconstruct_w_ale,
    compare_variants,
    reconstruct_w,
)
from kelp.core.geometry import METERS_PER_DEG_LAT, DEG2RAD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mesh(center_lon=5.0, center_lat=5.0,
               span=1.0, nx=5, ny=5,
               z_levels=None, bottom_depth=100.0):
    """Build a local triangular mesh for testing."""
    if z_levels is None:
        z_levels = [0.0, 10.0, 25.0, 50.0, 100.0]

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
            elems.append([n0, n0 + 1, n0 + nx])
            elems.append([n0 + 1, n0 + nx + 1, n0 + nx])
    elem_nodes = np.array(elems, dtype=np.int32)

    lon_c = np.array([lon[e].mean() for e in elem_nodes], dtype=np.float64)
    lat_c = np.array([lat[e].mean() for e in elem_nodes], dtype=np.float64)
    n2e, n2e_count = build_node_to_elems(n_nodes, elem_nodes)
    sigma = np.array(z_levels, dtype=np.float64)

    mesh = Mesh(
        name="test", nodes_lon=lon, nodes_lat=lat, depth=depth,
        elem_nodes=elem_nodes,
        elem_center_lon=lon_c, elem_center_lat=lat_c,
        node_to_elems=n2e, node_to_elems_count=n2e_count,
        sigma=sigma,
    )
    build_search(mesh)
    return mesh


# ---------------------------------------------------------------------------
# SSH tendency
# ---------------------------------------------------------------------------

class TestSSHTendency:
    def test_positive_rise(self):
        """SSH rising → ssh_t > 0."""
        eta_prev = np.zeros(5)
        eta_next = np.ones(5) * 0.1
        ssh_t = compute_ssh_tendency(eta_prev, eta_next, 100.0)
        assert ssh_t[0] == pytest.approx(0.001)

    def test_zero_dt_raises(self):
        with pytest.raises(ValueError, match="positive"):
            compute_ssh_tendency(np.zeros(3), np.ones(3), 0.0)

    def test_irregular_intervals(self):
        """Monthly means: 28 vs 31 days should give different ssh_t."""
        eta = np.array([0.0, 0.0, 0.0])
        eta_next = np.array([0.1, 0.1, 0.1])

        dt_28 = 28 * 86400.0
        dt_31 = 31 * 86400.0
        ssh_t_28 = compute_ssh_tendency(eta, eta_next, dt_28)
        ssh_t_31 = compute_ssh_tendency(eta, eta_next, dt_31)

        assert ssh_t_28[0] > ssh_t_31[0]
        assert ssh_t_28[0] == pytest.approx(0.1 / dt_28)
        assert ssh_t_31[0] == pytest.approx(0.1 / dt_31)


# ---------------------------------------------------------------------------
# Sign convention
# ---------------------------------------------------------------------------

class TestSignConvention:
    def test_w_surf_sign(self):
        """SSH rising (ssh_t > 0) → w_surf = -ssh_t < 0 (upward in positive-down)."""
        mesh = _make_mesh()
        z_levels = np.abs(mesh.sigma)
        N = mesh.n_nodes
        K = len(z_levels)

        ssh_t = np.ones(N) * 0.001  # SSH rising

        # Zero divergence → w constant throughout column
        div_h = np.zeros((K - 1, N))
        w = integrate_continuity(div_h, z_levels, ssh_t, mesh.depth,
                                 correct_bottom=False)

        # Surface: w = -ssh_t = -0.001
        assert w[0, 0] == pytest.approx(-0.001)
        # Without divergence and no bottom correction, w is constant
        for k in range(K):
            assert w[k, 0] == pytest.approx(-0.001)

    def test_ale_sign(self):
        """ALE heuristic: ssh_t > 0 → correction is negative (upward)."""
        z_levels = np.array([0.0, 50.0, 100.0])
        N = 3
        ssh_t = np.ones(N) * 0.001
        w_ale = np.zeros((3, N))
        depths = np.full(N, 100.0)

        w_cart = reconstruct_w_ale(w_ale, z_levels, ssh_t, depths)

        # At surface (D=0): w_cart = 0 - (1-0/100)*0.001 = -0.001
        assert w_cart[0, 0] == pytest.approx(-0.001)
        # At bottom (D=100): w_cart = 0 - (1-100/100)*0.001 = 0.0
        assert w_cart[2, 0] == pytest.approx(0.0)
        # Mid-depth (D=50): w_cart = 0 - (1-50/100)*0.001 = -0.0005
        assert w_cart[1, 0] == pytest.approx(-0.0005)


# ---------------------------------------------------------------------------
# Divergence (Tier 1)
# ---------------------------------------------------------------------------

class TestDivergenceApprox:
    def test_uniform_velocity_zero_divergence(self):
        """Uniform u, v → div = 0 everywhere."""
        mesh = _make_mesh()
        n_layers = len(mesh.sigma) - 1
        E = mesh.n_elements

        u = np.ones((n_layers, E)) * 1.0
        v = np.ones((n_layers, E)) * 0.5

        div_h = compute_divergence_approx(u, v, mesh)

        # Interior nodes should have near-zero divergence
        # (boundary nodes may not due to incomplete stencils)
        # Use the center node as reference
        nx, ny = 5, 5
        center_node = (ny // 2) * nx + (nx // 2)
        for k in range(n_layers):
            assert abs(div_h[k, center_node]) < 1e-10

    def test_linear_u_constant_divergence(self):
        """u = u0 * x / L → div_h = u0 / L (constant).

        On interior nodes of a regular mesh, the shape-function
        divergence should recover this exactly.
        """
        mesh = _make_mesh(center_lon=5.0, center_lat=5.0, span=1.0, nx=5, ny=5)
        cos_lat = math.cos(5.0 * DEG2RAD)
        SLM = METERS_PER_DEG_LAT

        # Element-based u proportional to element center longitude
        E = mesh.n_elements
        n_layers = len(mesh.sigma) - 1

        # u = slope * x_m, where x_m = lon * cos(lat) * SLM
        # du/dx = slope (in 1/s)
        slope = 1e-6  # 1/s
        u = np.zeros((n_layers, E))
        for e in range(E):
            x_m = mesh.elem_center_lon[e] * cos_lat * SLM
            u[:, e] = slope * x_m

        v = np.zeros((n_layers, E))

        div_h = compute_divergence_approx(u, v, mesh)

        # Interior nodes: expect div ≈ slope
        nx, ny = 5, 5
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                n = j * nx + i
                assert div_h[0, n] == pytest.approx(slope, rel=0.15), \
                    f"Node ({i},{j}): div={div_h[0, n]:.2e}, expected={slope:.2e}"


# ---------------------------------------------------------------------------
# Vertical integration
# ---------------------------------------------------------------------------

class TestIntegrateContinuity:
    def test_zero_divergence_constant_w(self):
        """No divergence → w = -ssh_t everywhere."""
        z_levels = np.array([0.0, 10.0, 50.0, 100.0])
        N = 5
        ssh_t = np.ones(N) * 0.002
        div_h = np.zeros((3, N))
        depths = np.full(N, 100.0)

        w = integrate_continuity(div_h, z_levels, ssh_t, depths,
                                 correct_bottom=False)

        for k in range(4):
            for n in range(N):
                assert w[k, n] == pytest.approx(-0.002)

    def test_constant_divergence_linear_w(self):
        """Constant div_h → w changes linearly with depth."""
        z_levels = np.array([0.0, 50.0, 100.0])
        N = 3
        ssh_t = np.zeros(N)
        div_h = np.ones((2, N)) * 1e-5  # positive divergence
        depths = np.full(N, 100.0)

        w = integrate_continuity(div_h, z_levels, ssh_t, depths,
                                 correct_bottom=False)

        # w(0) = -ssh_t = 0
        assert w[0, 0] == pytest.approx(0.0)
        # w(1) = 0 - div*50 = -5e-4
        assert w[1, 0] == pytest.approx(-5e-4)
        # w(2) = -5e-4 - div*50 = -1e-3
        assert w[2, 0] == pytest.approx(-1e-3)


# ---------------------------------------------------------------------------
# Bottom correction
# ---------------------------------------------------------------------------

class TestBottomCorrection:
    def test_correction_zeroes_bottom(self):
        """After correction, w(H) = 0."""
        z_levels = np.array([0.0, 25.0, 50.0, 100.0])
        N = 3
        ssh_t = np.zeros(N)
        div_h = np.ones((3, N)) * 1e-5
        depths = np.full(N, 100.0)

        w = integrate_continuity(div_h, z_levels, ssh_t, depths,
                                 correct_bottom=True)

        # Bottom interface (z=100) should be ≈ 0
        for n in range(N):
            assert w[3, n] == pytest.approx(0.0, abs=1e-15)

    def test_residual_taken_at_the_local_bottom(self):
        """Review 2026-09-22 (F05): with a node shallower than the table the
        residual was read at the GLOBAL last interface, so w(H) != 0, and
        below-bottom interfaces came back non-zero."""
        z_levels = np.array([0.0, 10.0, 20.0])
        div_h = np.full((2, 1), 0.1)
        w = integrate_continuity(div_h, z_levels, np.zeros(1),
                                 np.array([10.0]), correct_bottom=True)
        np.testing.assert_allclose(w[:, 0], [0.0, 0.0, 0.0], atol=1e-15)

    def test_mask_applied_after_the_correction(self):
        z_levels = np.array([0.0, 10.0, 20.0])
        div_h = np.full((2, 1), 0.1)
        w = integrate_continuity(div_h, z_levels, np.zeros(1),
                                 np.array([10.0]), nlevels_nod=np.array([2]),
                                 correct_bottom=True)
        np.testing.assert_allclose(w[:, 0], [0.0, 0.0, 0.0], atol=1e-15)

    def test_partial_bottom_layer(self):
        # H = 15 m inside the layer [10, 20]: the residual is the raw w
        # interpolated to 15 m; the interface at 10 m is corrected by
        # (10/15) of it; the interface at 20 m lies in the ground -> 0
        z_levels = np.array([0.0, 10.0, 20.0])
        div_h = np.array([[0.1], [0.3]])
        raw = integrate_continuity(div_h, z_levels, np.zeros(1),
                                   np.array([15.0]), correct_bottom=False)
        R = raw[1, 0] + 0.5 * (raw[2, 0] - raw[1, 0])
        w = integrate_continuity(div_h, z_levels, np.zeros(1),
                                 np.array([15.0]), correct_bottom=True)
        np.testing.assert_allclose(w[:, 0],
                                   [0.0, raw[1, 0] - (10.0 / 15.0) * R, 0.0],
                                   atol=1e-15)

    def test_correction_preserves_surface(self):
        """Surface value unchanged: w(0) = -ssh_t."""
        z_levels = np.array([0.0, 50.0, 100.0])
        N = 2
        ssh_t = np.ones(N) * 0.001
        div_h = np.ones((2, N)) * 1e-5
        depths = np.full(N, 100.0)

        w_uncorr = integrate_continuity(div_h, z_levels, ssh_t, depths,
                                        correct_bottom=False)
        w_corr = integrate_continuity(div_h, z_levels, ssh_t, depths,
                                      correct_bottom=True)

        # Surface: D=0, so correction = (0/H)*R = 0 → unchanged
        for n in range(N):
            assert w_corr[0, n] == pytest.approx(w_uncorr[0, n])


# ---------------------------------------------------------------------------
# ALE heuristic (Variant B)
# ---------------------------------------------------------------------------

class TestALEHeuristic:
    def test_zero_w_ale_gives_grid_velocity(self):
        """w_ALE = 0 → w_cart = -(1-D/H)*ssh_t."""
        z_levels = np.array([0.0, 20.0, 50.0, 100.0])
        N = 2
        K = 4
        w_ale = np.zeros((K, N))
        ssh_t = np.ones(N) * 0.002
        depths = np.full(N, 100.0)

        w_cart = reconstruct_w_ale(w_ale, z_levels, ssh_t, depths)

        for k, D in enumerate(z_levels):
            expected = -(1.0 - D / 100.0) * 0.002
            for n in range(N):
                assert w_cart[k, n] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Variant C: cross-validation
# ---------------------------------------------------------------------------

class TestCompareVariants:
    def test_identical_fields_zero_error(self):
        """If both variants produce the same w, error is 0."""
        z_levels = np.array([0.0, 50.0, 100.0])
        N = 2
        K = 3
        w = np.ones((K, N)) * 0.001
        ssh_t = np.zeros(N)
        w_ale = np.ones((K, N)) * 0.001
        depths = np.full(N, 100.0)

        diag = compare_variants(w, w, z_levels, ssh_t, w_ale, depths)
        assert diag["err_abs_max"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Full pipeline: reconstruct_w
# ---------------------------------------------------------------------------

class TestReconstructW:
    def test_divergence_method_shape(self):
        """Output shape is (K, N) for divergence method."""
        mesh = _make_mesh()
        K = len(mesh.sigma)
        N = mesh.n_nodes
        n_layers = K - 1
        E = mesh.n_elements

        u = np.zeros((n_layers, E))
        v = np.zeros((n_layers, E))
        eta_prev = np.zeros(N)
        eta_next = np.ones(N) * 0.01

        w = reconstruct_w(u, v, eta_prev, eta_next, 3600.0, mesh,
                          method="divergence")
        assert w.shape == (K, N)

    def test_ale_method_requires_w_ale(self):
        """method='ale' without w_ale raises ValueError."""
        mesh = _make_mesh()
        N = mesh.n_nodes
        E = mesh.n_elements
        n_layers = len(mesh.sigma) - 1

        with pytest.raises(ValueError, match="w_ale"):
            reconstruct_w(
                np.zeros((n_layers, E)), np.zeros((n_layers, E)),
                np.zeros(N), np.zeros(N), 3600.0, mesh,
                method="ale",
            )

    def test_invalid_method_raises(self):
        mesh = _make_mesh()
        N = mesh.n_nodes
        E = mesh.n_elements
        n_layers = len(mesh.sigma) - 1

        with pytest.raises(ValueError, match="method"):
            reconstruct_w(
                np.zeros((n_layers, E)), np.zeros((n_layers, E)),
                np.zeros(N), np.zeros(N), 3600.0, mesh,
                method="invalid",
            )

    def test_auto_with_w_ale_prints_diagnostics(self, capsys):
        """Auto method with w_ale prints comparison diagnostics."""
        mesh = _make_mesh()
        K = len(mesh.sigma)
        N = mesh.n_nodes
        E = mesh.n_elements
        n_layers = K - 1

        u = np.zeros((n_layers, E))
        v = np.zeros((n_layers, E))
        eta_prev = np.zeros(N)
        eta_next = np.ones(N) * 0.01
        w_ale = np.zeros((K, N))

        w = reconstruct_w(u, v, eta_prev, eta_next, 3600.0, mesh,
                          w_ale=w_ale, method="auto")
        assert w.shape == (K, N)

        captured = capsys.readouterr()
        assert "diagnostics" in captured.out.lower()

    def test_analytical_pipeline(self):
        """Full pipeline with known divergence matches analytical solution.

        u = slope * x_m (linear in x), v = 0, ssh_t = 0.
        → div_h = slope = const
        → w(D) = 0 - slope * D (before bottom correction)
        After bottom correction: w_corr(D) = -slope*D - (D/H)*(-slope*H) = 0.
        So with bottom correction and zero ssh_t, w should be ≈ 0 everywhere
        (the divergence is perfectly balanced).
        """
        mesh = _make_mesh(nx=5, ny=5, z_levels=[0.0, 50.0, 100.0],
                          bottom_depth=100.0)
        cos_lat = math.cos(5.0 * DEG2RAD)
        SLM = METERS_PER_DEG_LAT
        E = mesh.n_elements
        n_layers = 2
        N = mesh.n_nodes

        slope = 1e-6
        u = np.zeros((n_layers, E))
        for e in range(E):
            u[:, e] = slope * mesh.elem_center_lon[e] * cos_lat * SLM
        v = np.zeros((n_layers, E))

        eta = np.zeros(N)

        w = reconstruct_w(u, v, eta, eta, 3600.0, mesh,
                          method="divergence", correct_bottom=True)

        # With perfect divergence and bottom correction, interior nodes
        # should be near zero (the linear correction exactly cancels
        # the linear w profile when div_h is uniform)
        nx_grid = 5
        ny_grid = 5
        for j in range(1, ny_grid - 1):
            for i in range(1, nx_grid - 1):
                n = j * nx_grid + i
                for k in range(len(mesh.sigma)):
                    assert abs(w[k, n]) < 1e-3, \
                        f"w[{k},{n}] = {w[k,n]:.2e}, expected ≈ 0"


# ---------------------------------------------------------------------------
# Antimeridian safety of the weak-form divergence (fix 2026-07)
# ---------------------------------------------------------------------------

def _wrap_lon(x):
    return ((np.asarray(x) + 180.0) % 360.0) - 180.0


def _make_mesh_wrapped(center_lon, **kw):
    """_make_mesh whose node longitudes are wrapped to [-180, 180).

    _make_mesh itself builds an unwrapped linspace (e.g. 179..181); real
    readers store wrapped coordinates, which is exactly the regime where
    raw lon differences break at the seam.
    """
    mesh = _make_mesh(center_lon=center_lon, **kw)
    mesh.nodes_lon = _wrap_lon(mesh.nodes_lon)
    mesh.elem_center_lon = _wrap_lon(mesh.elem_center_lon)
    return mesh


class TestDivergenceSeam:
    """The divergence kernels used RAW node-lon metric differences: a
    seam-straddling element saw its neighbours ~360 deg away and the
    weak-form flux was corrupted in the one-element band along +-180.
    Pinned by translation invariance: the same mesh + wrapped-offset
    linear field built AT the seam and 30 deg away must give identical
    divergence (the off-seam twin also pins off-seam arithmetic against
    the linear-field expectation already tested above)."""

    def _div_for(self, lon0):
        mesh = _make_mesh_wrapped(center_lon=lon0, center_lat=5.0,
                                  span=1.0, nx=5, ny=5)
        cos_lat = math.cos(5.0 * DEG2RAD)
        SLM = METERS_PER_DEG_LAT
        E = mesh.n_elements
        n_layers = len(mesh.sigma) - 1
        slope = 1e-6
        # u linear in the WRAPPED offset from lon0 (dateline-safe field)
        off = _wrap_lon(np.asarray(mesh.elem_center_lon) - lon0)
        u = np.tile(slope * off * cos_lat * SLM, (n_layers, 1))
        # constant v: contributes zero TRUE divergence but multiplies the
        # lon-delta term of the weak-form flux — catches a raw-delta seam
        # bug even where the area normalization is externally supplied
        v = np.full((n_layers, E), 0.5)
        return mesh, compute_divergence_approx(u, v, mesh)

    def test_translation_invariance_at_seam(self):
        mesh_s, div_s = self._div_for(180.0)   # nodes span 179..-179
        mesh_o, div_o = self._div_for(-150.0)  # same geometry, off-seam
        np.testing.assert_allclose(div_s, div_o, rtol=1e-9, atol=1e-16)
        # and the interior value is the analytic constant
        slope = 1e-6
        nx = ny = 5
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                n = j * nx + i
                assert div_s[0, n] == pytest.approx(slope, rel=0.15)

    def test_with_area_translation_invariance_at_seam(self):
        from kelp.w_reconstruct import compute_divergence_with_area
        res = {}
        for tag, lon0 in (("s", 180.0), ("o", -150.0)):
            mesh = _make_mesh_wrapped(center_lon=lon0, center_lat=5.0,
                                      span=1.0, nx=5, ny=5)
            cos_lat = math.cos(5.0 * DEG2RAD)
            E = mesh.n_elements
            n_layers = len(mesh.sigma) - 1
            off = _wrap_lon(np.asarray(mesh.elem_center_lon) - lon0)
            u = np.tile(1e-6 * off * cos_lat * METERS_PER_DEG_LAT,
                        (n_layers, 1))
            v = np.full((n_layers, E), 0.5)
            nod_area = np.ones((len(mesh.sigma), mesh.n_nodes))
            res[tag] = compute_divergence_with_area(u, v, mesh, nod_area)
        np.testing.assert_allclose(res["s"], res["o"], rtol=1e-9, atol=1e-16)

    def test_single_far_neighbour_gated_independently(self):
        # review suggestion: an element may straddle the seam with only ONE
        # far vertex — each neighbour must be gated independently.  Node n
        # at 179.6, i_next across the seam at -179.9, i_prev on n's side
        # at 179.0: only i_next needs the shift.  Pinned by translation
        # invariance of the single-node kernel against the same triangle
        # 30 deg away.
        from kelp.w_reconstruct import _divergence_layer_single_node
        res = {}
        for tag, lon0 in (("s", 179.6), ("o", -150.4)):
            lon = _wrap_lon(np.array([lon0, lon0 + 0.5, lon0 - 0.6]))
            lat = np.array([5.0, 5.3, 5.4])
            en = np.array([[0, 1, 2]], dtype=np.int32)
            n2e, n2e_cnt = build_node_to_elems(3, en)
            u = np.array([0.3])
            v = np.array([0.5])
            res[tag] = _divergence_layer_single_node(
                0, u, v, en, lon, lat, n2e, n2e_cnt)
        assert np.isfinite(res["s"])
        assert res["s"] == pytest.approx(res["o"], rel=1e-12)


# ---------------------------------------------------------------------------
# Winding of the elements (FESOM2 writes its triangles clockwise)
# ---------------------------------------------------------------------------

_NX = _NY = 5
_INTERIOR = [j * _NX + i for j in range(1, _NY - 1) for i in range(1, _NX - 1)]


def _winding_mesh(winding, z_levels=(0.0, 10.0, 25.0, 50.0, 100.0)):
    """The 5x5 test mesh with its interior nodes moved off the grid
    (irregular triangles) and its elements written counter-clockwise
    ("ccw"), clockwise ("cw", the order FESOM2 writes) or alternating
    ("mixed").  The element indices and positions do not depend on the
    winding, only the vertex order inside each element does."""
    base = _make_mesh(nx=_NX, ny=_NY, z_levels=list(z_levels))
    lon = base.nodes_lon.copy()
    lat = base.nodes_lat.copy()
    rng = np.random.default_rng(7)
    for n in _INTERIOR:                      # spacing 0.5 deg: no flips
        lon[n] += rng.uniform(-0.1, 0.1)
        lat[n] += rng.uniform(-0.1, 0.1)
    elem = base.elem_nodes.copy()
    if winding == "cw":
        elem = elem[:, ::-1].copy()
    elif winding == "mixed":
        elem[::2] = elem[::2, ::-1]
    n2e, n2e_count = build_node_to_elems(lon.size, elem)
    mesh = Mesh(
        name=f"test_{winding}", nodes_lon=lon, nodes_lat=lat,
        depth=base.depth.copy(), elem_nodes=elem,
        elem_center_lon=lon[elem].mean(axis=1),
        elem_center_lat=lat[elem].mean(axis=1),
        node_to_elems=n2e, node_to_elems_count=n2e_count,
        sigma=base.sigma.copy(),
    )
    x, y = lon[elem], lat[elem]
    two_a = ((x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0])
             - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0]))
    want = {"ccw": two_a > 0, "cw": two_a < 0}.get(winding)
    if want is not None:
        assert want.all()
    else:
        assert (two_a > 0).any() and (two_a < 0).any()
    return mesh


def _cluster_area(mesh):
    """Sum of |A_e| / 3 over the elements around each node, in the metric of
    that node (x = lon * cos(lat_node) * METERS_PER_DEG_LAT): the area the
    Tier-1 kernel normalises by, here from the shoelace formula."""
    area = np.zeros(mesh.n_nodes)
    for n in range(mesh.n_nodes):
        c = math.cos(mesh.nodes_lat[n] * DEG2RAD)
        for e in mesh.node_to_elems[n, :mesh.node_to_elems_count[n]]:
            nd = mesh.elem_nodes[e]
            x = mesh.nodes_lon[nd] * c * METERS_PER_DEG_LAT
            y = mesh.nodes_lat[nd] * METERS_PER_DEG_LAT
            area[n] += abs(np.dot(x, np.roll(y, -1))
                           - np.dot(y, np.roll(x, -1))) / 6.0
    return area


class TestDivergenceWinding:
    """The weak-form flux of node n through element e is u . grad(N_n)
    times the element area; grad(N_n) carries 1 / (signed double area), so
    the term flips sign with the vertex order.  Both kernels dropped that
    sign: on a clockwise element (every element of a FESOM2 mesh but a few)
    the divergence came out negated.  Pinned by linear fields, whose
    weak-form divergence at an interior node is exact: with
    u_e = a cos(lat_ref) SLM lon_c(e) and v_e = b SLM lat_c(e) (element
    centroids), the kernel's metric at node n gives
    a cos(lat_ref) / cos(lat_n) + b."""

    A, B = 1e-6, -3e-7
    COS_REF = math.cos(5.0 * DEG2RAD)

    def _fields(self, mesh, n_layers):
        u = self.A * self.COS_REF * METERS_PER_DEG_LAT * mesh.elem_center_lon
        v = self.B * METERS_PER_DEG_LAT * mesh.elem_center_lat
        return np.tile(u, (n_layers, 1)), np.tile(v, (n_layers, 1))

    def _expected(self, mesh):
        lat = mesh.nodes_lat[_INTERIOR] * DEG2RAD
        return self.A * self.COS_REF / np.cos(lat) + self.B

    @pytest.mark.parametrize("winding", ["ccw", "cw", "mixed"])
    def test_tier1_linear_field(self, winding):
        mesh = _winding_mesh(winding)
        u, v = self._fields(mesh, len(mesh.sigma) - 1)
        div = compute_divergence_approx(u, v, mesh)
        np.testing.assert_allclose(div[:, _INTERIOR],
                                   np.tile(self._expected(mesh), (u.shape[0], 1)),
                                   rtol=1e-9)

    @pytest.mark.parametrize("winding", ["ccw", "cw", "mixed"])
    def test_tier2_linear_field(self, winding):
        from kelp.w_reconstruct import compute_divergence_with_area
        mesh = _winding_mesh(winding)
        K = len(mesh.sigma)
        u, v = self._fields(mesh, K - 1)
        nod_area = np.tile(_cluster_area(mesh), (K, 1))
        div = compute_divergence_with_area(u, v, mesh, nod_area)
        np.testing.assert_allclose(div[:, _INTERIOR],
                                   np.tile(self._expected(mesh), (K - 1, 1)),
                                   rtol=1e-9)

    @pytest.mark.parametrize("tier", [1, 2])
    @pytest.mark.parametrize("winding", ["cw", "mixed"])
    def test_reconstructed_w_does_not_depend_on_winding(self, winding, tier):
        rng = np.random.default_rng(3)
        ref = _winding_mesh("ccw")
        K, E, N = len(ref.sigma), ref.n_elements, ref.n_nodes
        u = 0.1 * rng.standard_normal((K - 1, E))
        v = 0.1 * rng.standard_normal((K - 1, E))
        eta_prev = 0.01 * rng.standard_normal(N)
        eta_next = 0.01 * rng.standard_normal(N)
        out = {}
        for tag, mesh in (("ccw", ref), (winding, _winding_mesh(winding))):
            extras = ({"nod_area": np.tile(_cluster_area(mesh), (K, 1))}
                      if tier == 2 else None)
            out[tag] = reconstruct_w(u, v, eta_prev, eta_next, 3600.0, mesh,
                                     mesh_extras=extras, method="divergence",
                                     correct_bottom=True)
        assert np.abs(out["ccw"]).max() > 1e-6
        np.testing.assert_allclose(out[winding], out["ccw"],
                                   rtol=1e-12, atol=1e-20)

    def test_degenerate_element_is_skipped(self):
        """A zero-area element has no shape-function gradient; it adds no
        flux to either kernel (before, its edge term went in without area)."""
        from kelp.w_reconstruct import (_divergence_layer_single_node,
                                        _divergence_layer_with_area)
        lon = np.array([5.0, 5.5, 5.0, 4.5, 4.0])
        lat = np.array([5.0, 5.0, 5.5, 5.0, 5.0])   # 0, 3, 4 collinear
        res = {}
        for tag, en in (("valid", [[0, 1, 2]]),
                        ("with_degenerate", [[0, 1, 2], [0, 3, 4]])):
            en = np.array(en, dtype=np.int32)
            n2e, cnt = build_node_to_elems(lon.size, en)
            u = np.full(len(en), 0.3)
            v = np.full(len(en), 0.5)
            t1 = _divergence_layer_single_node(0, u, v, en, lon, lat, n2e, cnt)
            out = np.zeros(lon.size)
            _divergence_layer_with_area(
                u, v, lon.size, en, lon, lat, n2e, cnt, np.ones(lon.size),
                np.full(len(en), 2, dtype=np.int32),
                np.ones(len(en), dtype=np.int32), 0, out)
            res[tag] = (t1, out[0])
        assert res["valid"][0] != 0.0 and res["valid"][1] != 0.0
        assert res["with_degenerate"] == res["valid"]


class TestBottomCorrectionLevelCut:
    """A node deeper than the last integrated interface (the tool's
    ``max_levels`` cut) has no bottom in the column: np.interp clamped the
    residual to the last interface and the correction pulled the cut
    interface to zero as if it were the bed."""

    Z = np.array([0.0, 10.0, 20.0])
    DIV = np.array([[0.1, 0.1, 0.1], [0.3, 0.3, 0.3]])
    SSH_T = np.full(3, 1e-3)
    DEPTHS = np.array([50.0, 20.0, 15.0])     # below the cut, at it, above it

    def _w(self, correct_bottom):
        return integrate_continuity(self.DIV, self.Z, self.SSH_T, self.DEPTHS,
                                    correct_bottom=correct_bottom)

    def test_node_below_the_cut_is_left_uncorrected(self):
        np.testing.assert_array_equal(self._w(True)[:, 0], self._w(False)[:, 0])

    def test_nodes_reaching_their_bottom_corrected_as_before(self):
        raw, w = self._w(False), self._w(True)
        # H = 20 m on the last interface: R = raw w there
        np.testing.assert_allclose(
            w[:, 1], raw[:, 1] - (self.Z / 20.0) * raw[2, 1], atol=1e-15)
        assert w[2, 1] == pytest.approx(0.0, abs=1e-15)
        # H = 15 m inside the last layer: R between the interfaces, the
        # interface at 20 m in the ground
        R = raw[1, 2] + 0.5 * (raw[2, 2] - raw[1, 2])
        np.testing.assert_allclose(
            w[:, 2], [raw[0, 2], raw[1, 2] - (10.0 / 15.0) * R, 0.0],
            atol=1e-15)


class TestCliOutputLayout:
    """The CLI's file must be readable back by the FESOM2 reader.

    The reconstructed W lives on the K vertical INTERFACES.  FESOM2 names that
    axis ``nz`` and reserves ``nz1`` for the K-1 mid-layers, and
    ``readers.fesom2`` rejects a W that carries ``nz1`` — so writing the
    interfaces under the mid-layer name silently broke the whole point of the
    utility (reconstruct, then track on the result).
    """

    @staticmethod
    def _load_cli():
        import importlib.util
        import pathlib
        import sys
        script = (pathlib.Path(__file__).resolve().parents[1]
                  / "utils" / "reconstruct_w.py")
        spec = importlib.util.spec_from_file_location("reconstruct_w_cli", script)
        mod = importlib.util.module_from_spec(spec)
        # dataclasses resolve their module through sys.modules while the class
        # body runs, so register before executing.
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    def test_ale_input_converted_to_positive_down(self):
        """FESOM2's ALE w is positive UP; the ALE heuristic needs positive
        down (review 2026-09-23: +1e-5 upwelling came out as downwelling)."""
        xr = pytest.importorskip("xarray")
        cli = self._load_cli()
        up = xr.Dataset({"w": (("time", "nz", "nod2"), np.full((2, 3, 4), 1e-5))})
        np.testing.assert_allclose(cli._w_ale_positive_down(up, "w", 0), -1e-5)
        dn = up.copy()
        dn["w"].attrs["positive"] = "down"
        np.testing.assert_allclose(cli._w_ale_positive_down(dn, "w", 0), 1e-5)
        assert cli._w_ale_positive_down(up, "w", 1, n_keep=2).shape == (2, 4)

    def test_written_w_is_interface_based(self, tmp_path):
        xr = pytest.importorskip("xarray")
        cli = self._load_cli()
        n_times, K, N = 2, 4, 5
        w_all = np.zeros((n_times, K, N), dtype=np.float64)
        z_levels = np.array([0.0, 10.0, 25.0, 60.0])
        cfg = cli.OutputConfig(dir=str(tmp_path), format="netcdf",
                               filename="w.nc")
        cli._write_output(w_all, np.array([0.0, 86400.0]), z_levels, cfg,
                          meshpath="mesh.nc")
        with xr.open_dataset(tmp_path / "w.nc") as ds:
            assert "nz" in ds["w"].dims, ds["w"].dims
            assert "nz1" not in ds["w"].dims        # the reader raises on this
            assert ds.sizes["nz"] == K
            # KELP-internal sign, declared so the reader's "auto" does not
            # flip it as it flips FESOM2's positive-up w (review 2026-09)
            assert ds["w"].attrs["positive"] == "down"
