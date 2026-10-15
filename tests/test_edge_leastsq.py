"""Edge-neighbor least-squares u/v interpolation (uv_interp: edge_leastsq).

FESOM-style elemental interpolation: LSQ plane fit over the containing
element and the elements sharing an edge with it (3 for a triangle, 4 for
a quad); missing neighbors at boundaries are replaced by ghost points
mirrored across the boundary edge carrying the lateral boundary condition
(freeslip: the mirror image 2(u·t̂)t̂ − u — tangential kept, normal reversed;
noslip: −u_own).
"""
import numpy as np
import pytest

from kelp.core.mesh import (
    Mesh, build_node_to_elems, build_search, find_nearest_nodes,
    build_elem_neighbors,
)


def _anchored_lon_mean(vals):
    """Seam-aware element-center longitude (raw mean off-seam, wrapped-delta
    mean anchored on the first vertex when the cell straddles the seam)."""
    vals = np.asarray(vals, float)
    if vals.max() - vals.min() <= 180.0:
        return vals.mean()
    ref = vals[0]
    d = ((vals - ref + 180.0) % 360.0) - 180.0
    return ((ref + d.mean() + 180.0) % 360.0) - 180.0


def _mesh_from(lon, lat, elem_nodes, depth_val=10.0):
    """Minimal 2D Mesh from arrays (idiom of test_uv_registry.py, with
    seam-aware element centers — review round 1)."""
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    en = np.asarray(elem_nodes, np.int32)
    n2e, n2e_cnt = build_node_to_elems(lon.size, en)
    cl = np.array([_anchored_lon_mean(lon[e[e >= 0]]) for e in en])
    ca = np.array([lat[e[e >= 0]].mean() for e in en])
    return Mesh(name="test", nodes_lon=lon, nodes_lat=lat,
                depth=np.full(lon.size, depth_val),
                elem_nodes=en, elem_center_lon=cl, elem_center_lat=ca,
                node_to_elems=n2e, node_to_elems_count=n2e_cnt,
                sigma=np.empty(0), vertical_coord="zlevel")


def _grid_mesh():
    """3x3-node, 8-tri canonical mesh around (5,5), with search built."""
    lonn = [4, 5, 6, 4, 5, 6, 4, 5, 6]
    latn = [4, 4, 4, 5, 5, 5, 6, 6, 6]
    en = [[0, 1, 4, -1], [0, 4, 3, -1], [1, 2, 5, -1], [1, 5, 4, -1],
          [3, 4, 7, -1], [3, 7, 6, -1], [4, 5, 8, -1], [4, 8, 7, -1]]
    m = _mesh_from(lonn, latn, en)
    build_search(m)
    return m


def _quad_mesh_2x2():
    """2x2 quad cells (9 nodes, 4 quads) around (5,5)."""
    lonn = [4, 5, 6, 4, 5, 6, 4, 5, 6]
    latn = [4, 4, 4, 5, 5, 5, 6, 6, 6]
    en = [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]]
    m = _mesh_from(lonn, latn, en)
    build_search(m)
    return m


class TestElemNeighbors:
    def test_tri_mesh_adjacency(self):
        # Element 0 = [0,1,4]: edge (0,1) is the south boundary, edge (1,4)
        # is shared with element 3 = [1,5,4], edge (4,0) with element 1 =
        # [0,4,3].  Slot k <-> edge (verts[k], verts[k+1]) on the compacted
        # ring; the padded 4th slot is unused and stays -1.
        m = _grid_mesh()
        nb = build_elem_neighbors(m)
        assert nb.shape == (8, 4)
        assert nb.dtype == np.int32
        np.testing.assert_array_equal(nb[0], [-1, 3, 1, -1])
        # Element 3 = [1,5,4]: (1,5)->elem 2, (5,4)->elem 6, (4,1)->elem 0.
        np.testing.assert_array_equal(nb[3], [2, 6, 0, -1])

    def test_adjacency_is_mutual(self):
        m = _grid_mesh()
        nb = build_elem_neighbors(m)
        for e in range(m.n_elements):
            for k in range(4):
                n = nb[e, k]
                if n < 0:
                    continue
                assert e in nb[n], (e, k, n)

    def test_quad_mesh_adjacency(self):
        # Quad 0 = [0,1,4,3]: south (0,1) boundary; east (1,4) -> quad 1;
        # north (4,3) -> quad 2; west (3,0) boundary.
        m = _quad_mesh_2x2()
        nb = build_elem_neighbors(m)
        np.testing.assert_array_equal(nb[0], [-1, 1, 2, -1])
        # Quad 3 = [4,5,8,7]: south (4,5) -> quad 1; east (5,8) boundary;
        # north (8,7) boundary; west (7,4) -> quad 2.
        np.testing.assert_array_equal(nb[3], [1, -1, -1, 2])

    def test_fesomc_padded_and_repeated_encodings_agree(self):
        # The SAME two triangles sharing edge (1,2), stored -1-padded vs
        # first-vertex-repeated, must give identical neighbor tables.
        lon = [0.0, 0.1, 0.0, 0.12]
        lat = [0.0, 0.0, 0.1, 0.11]
        en_pad = [[0, 1, 2, -1], [1, 3, 2, -1]]
        en_rep = [[0, 1, 2, 0], [1, 3, 2, 1]]
        nb_pad = build_elem_neighbors(_mesh_from(lon, lat, en_pad))
        nb_rep = build_elem_neighbors(_mesh_from(lon, lat, en_rep))
        np.testing.assert_array_equal(nb_pad, nb_rep)
        np.testing.assert_array_equal(nb_pad[0], [-1, 1, -1, -1])

    def test_non_manifold_edge_raises(self):
        lon = [0.0, 0.1, 0.0, 0.1, -0.1]
        lat = [0.0, 0.0, 0.1, -0.1, 0.05]
        en = [[0, 1, 2, -1], [0, 1, 3, -1], [0, 1, 4, -1]]
        m = _mesh_from(lon, lat, en)
        with pytest.raises(ValueError, match="non-manifold"):
            build_elem_neighbors(m)

    def test_cache_idempotent(self):
        m = _grid_mesh()
        nb1 = build_elem_neighbors(m)
        nb2 = build_elem_neighbors(m)
        assert nb1 is nb2


# ---------------------------------------------------------------------------
# The interpolator itself
# ---------------------------------------------------------------------------

from kelp.core.geometry import DEG2RAD, METERS_PER_DEG_LAT
from kelp.core.particles import make_state, ParticleStatus
from kelp.numerics.interpolate import (
    VelocityField, get_interpolator, get_interpolator_single,
    interp_edge_leastsq, interp_edge_leastsq_single,
)


def _quad_mesh_3x3():
    """3x3 quad cells (16 nodes, 9 quads); center quad has 4 real neighbors."""
    lonn, latn = np.meshgrid(np.arange(3.0, 7.0), np.arange(3.0, 7.0))
    lonn = lonn.ravel(); latn = latn.ravel()
    en = []
    for r in range(3):
        for c in range(3):
            i = r * 4 + c
            en.append([i, i + 1, i + 5, i + 4])
    m = _mesh_from(lonn, latn, en)
    build_search(m)
    return m


def _vf_elem(mesh, fn):
    """Element-layout field: u_e = fn(center_lon, center_lat), v = 2*u."""
    u = fn(mesh.elem_center_lon, mesh.elem_center_lat)
    return VelocityField(u=u, v=2.0 * u, eta=np.zeros(mesh.n_nodes), w=None)


def _lin(x, y):
    return 0.3 + 0.5 * x - 0.2 * y


class TestRegistryEdgeLeastsq:
    def test_registered_both_forms(self):
        assert get_interpolator("edge_leastsq") is interp_edge_leastsq
        assert (get_interpolator_single("edge_leastsq")
                is interp_edge_leastsq_single)

    def test_bc_partial_and_validation(self):
        fn = get_interpolator("edge_leastsq", edge_leastsq_bc="noslip")
        assert callable(fn) and fn is not interp_edge_leastsq
        with pytest.raises(ValueError, match="freeslip"):
            get_interpolator("edge_leastsq", edge_leastsq_bc="free_slip")
        # bc irrelevant for other methods — silently ignored
        from kelp.numerics.interpolate import interp_leastsq
        assert get_interpolator("leastsq", edge_leastsq_bc="noslip") \
            is interp_leastsq


class TestEdgeLeastsqValues:
    def test_linear_field_exact_interior_tri(self):
        # Element 3 = [1,5,4] has three real neighbors (2, 6, 0): the fit
        # sees 4 centroids of a lon/lat-linear field -> exact reproduction.
        mesh = _grid_mesh()
        vf = _vf_elem(mesh, _lin)
        pts = np.array([[5.3, 4.7], [5.15, 4.9], [5.4, 4.55]])
        st = make_state(pts[:, 0], pts[:, 1])
        nn = find_nearest_nodes(mesh, st.lon, st.lat)
        U, V, ok = interp_edge_leastsq(st, mesh, vf, nn)
        assert ok.all()
        expected = _lin(pts[:, 0], pts[:, 1])
        np.testing.assert_allclose(U, expected, rtol=1e-9)
        np.testing.assert_allclose(V, 2 * expected, rtol=1e-9)

    def test_linear_field_exact_interior_quad(self):
        # Center quad of the 3x3 grid: 5 stencil points, all real.
        mesh = _quad_mesh_3x3()
        vf = _vf_elem(mesh, _lin)
        pts = np.array([[4.5, 4.5], [4.2, 4.8], [4.9, 4.1]])
        st = make_state(pts[:, 0], pts[:, 1])
        nn = find_nearest_nodes(mesh, st.lon, st.lat)
        U, V, ok = interp_edge_leastsq(st, mesh, vf, nn)
        assert ok.all()
        expected = _lin(pts[:, 0], pts[:, 1])
        np.testing.assert_allclose(U, expected, rtol=1e-9)
        np.testing.assert_allclose(V, 2 * expected, rtol=1e-9)

    def test_freeslip_wall_parallel_constant_exact(self):
        # Free-slip ghost = mirror image (tangential kept, normal reversed):
        # a constant flow ALONG the wall is reproduced exactly in a
        # boundary element.  Element 0 = [0,1,4] of the grid mesh has one
        # boundary edge, (0,1) on the southern wall (lat 4).
        mesh = _grid_mesh()
        u_e = np.full(mesh.n_elements, 0.7)
        vf = VelocityField(u=u_e, v=np.zeros_like(u_e),
                           eta=np.zeros(mesh.n_nodes), w=None)
        for x, y in ((5.0, 4.2), (4.7, 4.05), (4.9, 4.6)):
            nn = int(find_nearest_nodes(mesh, np.array([x]),
                                        np.array([y]))[0])
            u, v, ok = interp_edge_leastsq_single(x, y, 0, mesh, vf, nn,
                                                  1.0, bc="freeslip")
            assert ok
            assert u == pytest.approx(0.7, rel=1e-12)
            assert v == pytest.approx(0.0, abs=1e-12)

    def test_freeslip_blocks_flow_through_the_wall(self):
        # A constant flow NORMAL to the southern wall: the old ghost copied
        # the whole vector, so the reconstruction kept v = 0.7 AT the wall
        # (water through the coast; review 2026-09-22, M07).  The mirror
        # ghost reverses the normal component: v at the wall must drop to
        # a small fraction (soft constraint of an over-determined plane
        # fit, not an exact zero).
        mesh = _grid_mesh()
        v_e = np.full(mesh.n_elements, 0.7)
        vf = VelocityField(u=np.zeros_like(v_e), v=v_e,
                           eta=np.zeros(mesh.n_nodes), w=None)
        u, v, ok = interp_edge_leastsq_single(4.7, 4.0 + 1e-9, 0, mesh, vf,
                                              0, 1.0, bc="freeslip")
        assert ok
        assert abs(v) < 0.25 * 0.7, v
        assert abs(u) < 1e-12

    def test_constant_field_exact_in_interior_elements(self):
        # Away from walls every constant is exact whatever the bc.
        mesh = _grid_mesh()
        vf = _vf_elem(mesh, lambda x, y: np.full(x.shape, 0.7))
        for bc in ("freeslip", "noslip"):
            u, v, ok = interp_edge_leastsq_single(5.3, 4.7, 0, mesh, vf, 4,
                                                  1.0, bc=bc)
            assert ok
            assert u == pytest.approx(0.7, rel=1e-12)
            assert v == pytest.approx(1.4, rel=1e-12)

    def test_noslip_single_triangle_matches_independent_reference(self):
        # One triangle, all three edges boundary.  Independent numpy
        # reference: ghost = centroid reflected across each edge (computed
        # via the edge NORMAL, a different route than the implementation's
        # tangent projection), values -u_own, plane via np.linalg.lstsq.
        lon = [0.0, 0.1, 0.0]
        lat = [0.0, 0.0, 0.1]
        m = _mesh_from(lon, lat, [[0, 1, 2, -1]])
        build_search(m)
        u0 = 0.8
        vf = _vf_elem(m, lambda x, y: np.full(x.shape, u0))
        xp, yp = 0.03, 0.03
        nn = int(find_nearest_nodes(m, np.array([xp]), np.array([yp]))[0])
        U, V, ok = interp_edge_leastsq_single(xp, yp, 0, m, vf, nn,
                                              1.0, bc="noslip")
        assert ok

        z0 = np.cos(yp * DEG2RAD) * METERS_PER_DEG_LAT

        def met(lo, la):
            dlon = ((lo - xp + 180.0) % 360.0) - 180.0
            return np.array([dlon * z0, (la - yp) * METERS_PER_DEG_LAT])

        C = met(m.elem_center_lon[0], m.elem_center_lat[0])
        P = [C]
        vals = [u0]
        ring = [(0, 1), (1, 2), (2, 0)]
        for a, b in ring:
            A = met(lon[a], lat[a]); B = met(lon[b], lat[b])
            t = B - A
            n = np.array([-t[1], t[0]]); n /= np.linalg.norm(n)
            ghost = C - 2.0 * np.dot(C - A, n) * n
            P.append(ghost)
            vals.append(-u0)
        M = np.column_stack([np.ones(4),
                             [p[0] for p in P], [p[1] for p in P]])
        coef, *_ = np.linalg.lstsq(M, np.array(vals), rcond=None)
        expected = coef[0]
        assert U == pytest.approx(expected, rel=1e-9)
        assert V == pytest.approx(2 * expected, rel=1e-9)
        # noslip pulls the magnitude down toward the coast
        assert abs(U) < u0

    def test_interior_element_bc_irrelevant(self):
        mesh = _grid_mesh()
        rng = np.random.default_rng(3)
        u_e = rng.uniform(-1, 1, mesh.n_elements)
        vf = VelocityField(u=u_e, v=u_e.copy(),
                           eta=np.zeros(mesh.n_nodes), w=None)
        # point inside interior element 3 = [1,5,4]
        u_f, _, ok_f = interp_edge_leastsq_single(5.3, 4.7, 0, mesh, vf, 4,
                                                  1.0, bc="freeslip")
        u_n, _, ok_n = interp_edge_leastsq_single(5.3, 4.7, 0, mesh, vf, 4,
                                                  1.0, bc="noslip")
        assert ok_f and ok_n
        assert u_f == u_n

    def test_boundary_element_bc_matters(self):
        mesh = _grid_mesh()
        u_e = np.full(mesh.n_elements, 0.7)     # parallel to the south wall
        vf = VelocityField(u=u_e, v=np.zeros_like(u_e),
                           eta=np.zeros(mesh.n_nodes), w=None)
        # point inside element 0 = [0,1,4] (boundary edge (0,1))
        u_f, _, _ = interp_edge_leastsq_single(5.0, 4.2, 0, mesh, vf, 1,
                                               1.0, bc="freeslip")
        u_n, _, _ = interp_edge_leastsq_single(5.0, 4.2, 0, mesh, vf, 1,
                                               1.0, bc="noslip")
        assert u_f == pytest.approx(0.7, rel=1e-12)
        assert u_n != pytest.approx(0.7, rel=1e-6)
        assert abs(u_n) < 0.7

    def test_time_dir_flips_sign(self):
        mesh = _grid_mesh()
        vf = _vf_elem(mesh, lambda x, y: np.full(x.shape, 0.5))
        u_f, _, _ = interp_edge_leastsq_single(5.1, 4.8, 0, mesh, vf, 4, 1.0)
        u_b, _, _ = interp_edge_leastsq_single(5.1, 4.8, 0, mesh, vf, 4, -1.0)
        assert u_b == pytest.approx(-u_f, rel=1e-14)

    def test_outside_mesh_fails(self):
        mesh = _grid_mesh()
        vf = _vf_elem(mesh, lambda x, y: np.ones(x.shape))
        u, v, ok = interp_edge_leastsq_single(9.0, 9.0, 0, mesh, vf, 8)
        assert not ok and u == 0.0 and v == 0.0

    def test_dead_particles_untouched(self):
        mesh = _grid_mesh()
        vf = _vf_elem(mesh, lambda x, y: np.ones(x.shape))
        st = make_state(np.array([5.2, 4.6]), np.array([4.7, 5.3]))
        status = st.status.copy(); status[1] = ParticleStatus.OUTSIDE
        st = st.replace(status=status)
        nn = find_nearest_nodes(mesh, st.lon, st.lat)
        U, V, ok = interp_edge_leastsq(st, mesh, vf, nn)
        assert ok[0] and not ok[1]
        assert U[1] == 0.0 and V[1] == 0.0

    def test_3d_per_particle_layers(self):
        mesh = _grid_mesh()
        u3 = np.vstack([np.full(mesh.n_elements, 1.0),
                        np.full(mesh.n_elements, 9.0)])
        vf = VelocityField(u=u3, v=2 * u3, eta=np.zeros(mesh.n_nodes),
                           w=np.zeros((3, mesh.n_nodes)))
        st = make_state(np.array([5.2, 4.6]), np.array([4.7, 5.3]),
                        np.array([1.0, 50.0]))
        nn = find_nearest_nodes(mesh, st.lon, st.lat)
        layers = np.array([0, 1], dtype=np.int32)
        U, V, ok = interp_edge_leastsq(st, mesh, vf, nn, 1.0, layers)
        assert ok.all()
        np.testing.assert_allclose(U, [1.0, 9.0], rtol=1e-12)

    def test_node_layout_raises(self):
        mesh = _grid_mesh()
        u_n = np.ones(mesh.n_nodes)
        vf = VelocityField(u=u_n, v=u_n.copy(), eta=np.zeros(mesh.n_nodes),
                           w=None, uv_layout="node")
        st = make_state(np.array([5.2]), np.array([4.7]))
        nn = find_nearest_nodes(mesh, st.lon, st.lat)
        with pytest.raises(ValueError, match="barycentric"):
            interp_edge_leastsq(st, mesh, vf, nn)
        with pytest.raises(ValueError, match="barycentric"):
            interp_edge_leastsq_single(5.2, 4.7, 0, mesh, vf, int(nn[0]))

    def test_dateline_seam_constant_matches_off_seam(self):
        # Two triangles sharing an edge across +/-180 (every outer edge a
        # wall): wrapped deltas and ghost mirrors must give the same
        # values as the identical geometry far from the seam.
        def build(lon0):
            w = lambda x: ((x + 180.0) % 360.0) - 180.0
            lon = [lon0, w(lon0 + 1.0), w(lon0 + 1.0), lon0]
            lat = [0.0, 0.0, 1.0, 1.0]
            m = _mesh_from(lon, lat, [[0, 1, 2, -1], [0, 2, 3, -1]])
            build_search(m)
            return m, _vf_elem(m, lambda x, y: np.full(x.shape, 0.7))

        w = lambda x: ((x + 180.0) % 360.0) - 180.0
        m_s, vf_s = build(179.5)
        m_o, vf_o = build(-30.0)
        for dx, dy in ((0.4, 0.3), (0.6, 0.4), (0.1, 0.8)):
            res = []
            for m, vf, lon0 in ((m_s, vf_s, 179.5), (m_o, vf_o, -30.0)):
                q = w(lon0 + dx)
                nn = int(find_nearest_nodes(m, np.array([q]),
                                            np.array([dy]))[0])
                u, v, ok = interp_edge_leastsq_single(q, dy, 0, m, vf, nn)
                assert ok, (dx, dy)
                res.append((u, v))
            assert res[0][0] == pytest.approx(res[1][0], rel=1e-9)
            assert res[0][1] == pytest.approx(res[1][1], rel=1e-9)

    def test_noslip_overshoot_and_corner_reversal_characterized(self):
        # Documented artifacts of the mirror-ghost + single-plane fit
        # (review round 1, important 1): on a single-wall element the tilted
        # plane OVERSHOOTS the ambient speed on the interior side (measured
        # ~+1.06 for ambient 0.7) while decaying toward ~0 at the wall
        # (~-0.05); in a two-wall corner cell the plane crosses zero and the
        # velocity REVERSES sign near the corner (~-0.42).  These are
        # properties of the boundary-condition model, pinned here so a
        # future ghost-geometry change is caught.
        gm = _grid_mesh()
        vf = _vf_elem(gm, lambda x, y: np.full(x.shape, 0.7))
        u_int, _, ok1 = interp_edge_leastsq_single(4.95, 4.85, 0, gm, vf, 4,
                                                   1.0, bc="noslip")
        u_wall, _, ok2 = interp_edge_leastsq_single(4.7, 4.05, 0, gm, vf, 0,
                                                    1.0, bc="noslip")
        assert ok1 and ok2
        assert u_int > 0.7           # interior-side overshoot above ambient
        assert abs(u_wall) < 0.15    # near-zero at the wall
        qm = _quad_mesh_2x2()
        vfq = _vf_elem(qm, lambda x, y: np.full(x.shape, 0.7))
        u_c, _, ok3 = interp_edge_leastsq_single(4.1, 4.1, 0, qm, vfq, 0,
                                                 1.0, bc="noslip")
        assert ok3
        assert u_c < 0.0             # corner-cell sign reversal

    def test_sliver_element_stays_conditioned(self):
        # review round 1 proposed a scale-aware determinant guard for skinny
        # boundary stencils; measurement shows the mirror ghosts keep the
        # normalized conditioning det/(a11*a22*a33) ~ 0.7 even at aspect
        # 1e-4 (the reflections always span both directions), so no guard is
        # added — pin that the fit stays ok and constant-exact instead.
        lon = [0.0, 0.1, 0.05]
        lat = [0.0, 0.0, 1e-5]
        m = _mesh_from(lon, lat, [[0, 1, 2, -1]])
        build_search(m)
        # flow along the (nearly east-west) sliver: the mirror ghosts of the
        # two slightly tilted edges rotate it by ~2θ², θ ≈ 2e-4
        u_e = np.full(1, 0.7)
        vf = VelocityField(u=u_e, v=np.zeros(1), eta=np.zeros(3), w=None)
        u, v, ok = interp_edge_leastsq_single(0.05, 3e-6, 0, m, vf, 2)
        assert ok
        assert u == pytest.approx(0.7, rel=1e-6)

    def test_dateline_translation_invariance_linear_field(self):
        # Same 2-tri geometry built AT the seam and 150 deg away, element
        # values from the same wrapped-offset linear law: edge_leastsq must
        # return identical values at corresponding query points.  Pins the
        # seam-aware element centers AND the wrapped stencil/ghost geometry
        # (review round 1: the constant-field seam test could not see a
        # misplaced centroid).
        def build(lon0):
            w = lambda x: ((x + 180.0) % 360.0) - 180.0
            lon = [lon0, w(lon0 + 1.0), w(lon0 + 1.0), lon0]
            lat = [0.0, 0.0, 1.0, 1.0]
            en = [[0, 1, 2, -1], [0, 2, 3, -1]]
            m = _mesh_from(lon, lat, en)
            build_search(m)
            off = ((np.asarray(m.elem_center_lon) - lon0 + 180.0)
                   % 360.0) - 180.0
            u = 0.2 + 3.0 * off
            vf = VelocityField(u=u, v=2 * u, eta=np.zeros(4), w=None)
            return m, vf

        w = lambda x: ((x + 180.0) % 360.0) - 180.0
        m_seam, vf_seam = build(179.5)
        m_off, vf_off = build(-30.0)
        for dx, dy in ((0.7, 0.25), (0.2, 0.6), (0.95, 0.5)):
            q_s, q_o = w(179.5 + dx), w(-30.0 + dx)
            nn_s = int(find_nearest_nodes(m_seam, np.array([q_s]),
                                          np.array([dy]))[0])
            nn_o = int(find_nearest_nodes(m_off, np.array([q_o]),
                                          np.array([dy]))[0])
            u_s, v_s, ok_s = interp_edge_leastsq_single(
                q_s, dy, 0, m_seam, vf_seam, nn_s)
            u_o, v_o, ok_o = interp_edge_leastsq_single(
                q_o, dy, 0, m_off, vf_off, nn_o)
            assert ok_s and ok_o, (dx, dy)
            assert u_s == pytest.approx(u_o, rel=1e-9), (dx, dy)
            assert v_s == pytest.approx(v_o, rel=1e-9), (dx, dy)

    def test_config_accepts_edge_leastsq(self):
        from kelp.core.config import NumericsConfig
        cfg = NumericsConfig(uv_interp="edge_leastsq")
        assert cfg.edge_leastsq_bc == "freeslip"
        NumericsConfig(uv_interp="edge_leastsq", edge_leastsq_bc="noslip")
        with pytest.raises(ValueError, match="edge_leastsq_bc"):
            NumericsConfig(edge_leastsq_bc="free_slip")

    def test_single_matches_vectorized(self):
        mesh = _grid_mesh()
        rng = np.random.default_rng(11)
        u_e = rng.uniform(-1, 1, mesh.n_elements)
        vf = VelocityField(u=u_e, v=2 * u_e,
                           eta=np.zeros(mesh.n_nodes), w=None)
        pts = rng.uniform(4.1, 5.9, size=(25, 2))
        st = make_state(pts[:, 0], pts[:, 1])
        nn = find_nearest_nodes(mesh, st.lon, st.lat)
        U, V, ok = interp_edge_leastsq(st, mesh, vf, nn)
        for i in range(pts.shape[0]):
            u1, v1, ok1 = interp_edge_leastsq_single(
                pts[i, 0], pts[i, 1], 0, mesh, vf, int(nn[i]))
            assert ok1 == ok[i]
            assert u1 == U[i] and v1 == V[i]


# ---------------------------------------------------------------------------
# Integrator + tracking-layer wiring
# ---------------------------------------------------------------------------

def _make_fesomc_const(tmp_path, uv_dim="nele", u_val=1.0, n_time=4):
    """Synthetic FESOM-C file (idiom of test_uv_tracking.py): constant u,
    v = 0; u2d/v2d on "nele" (element layout) or "node".

    5x5 nodes over [4, 8] x [4, 8]: the particles of these tests start at
    lon 5-5.5 on lat 5 and move along it, through elements that are
    interior or whose wall is the southern one (parallel to the flow), so
    the free-slip reconstruction stays exact.  On the former 3x3 box every
    element touched a wall and the east/west walls (normal to the flow)
    correctly decelerate it (mirror ghost, review 2026-09-22)."""
    import pandas as pd
    import xarray as xr
    nx = 5
    n_nodes, n_elem = nx * nx, 2 * (nx - 1) ** 2
    xs = np.arange(4.0, 4.0 + nx); ys = np.arange(4.0, 4.0 + nx)
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel(); lat = gy.ravel()
    elems = []
    for j in range(nx - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            elems.append([n0 + 1, n0 + 2, n0 + nx + 1, n0 + 1])
            elems.append([n0 + 2, n0 + nx + 2, n0 + nx + 1, n0 + 2])
    nv = np.array(elems, dtype=np.int32)
    en0 = nv[:, :3] - 1
    size = n_nodes if uv_dim == "node" else n_elem
    ds = xr.Dataset({
        "lon": (["node"], lon), "lat": (["node"], lat),
        "depth": (["node"], np.full(n_nodes, 10.0)),
        "nv": (["nele", "four"], nv),
        "lon_elem": (["nele"], np.array([lon[e].mean() for e in en0])),
        "lat_elem": (["nele"], np.array([lat[e].mean() for e in en0])),
        "sigma_lev": (["sigma"], np.array([0.0, 0.5, 1.0])),
        "u2d": (["time", uv_dim], np.full((n_time, size), u_val)),
        "v2d": (["time", uv_dim], np.zeros((n_time, size))),
        "eta": (["time", "node"], np.zeros((n_time, n_nodes))),
    }, coords={"time": pd.date_range("2020-01-01", periods=n_time,
                                     freq="3600s")})
    p = tmp_path / f"fesomc_{uv_dim}_{u_val}.nc"
    ds.to_netcdf(p)
    return p


def _open_fesomc(tmp_path, uv_dim, u_val=1.0):
    from types import SimpleNamespace
    from kelp.readers import get_reader
    p = _make_fesomc_const(tmp_path, uv_dim=uv_dim, u_val=u_val)
    r = get_reader("fesomc")
    mesh = r.load_mesh(SimpleNamespace(meshpath=str(p)))
    ds = r.open_dataset(str(p))
    return p, r, mesh, ds


class TestEdgeLeastsqWiring:
    def test_validate_uv_compat(self):
        from kelp.tracking import _validate_uv_compat
        _validate_uv_compat("element", "edge_leastsq")     # accepted
        with pytest.raises(ValueError, match="edge_leastsq"):
            _validate_uv_compat("node", "edge_leastsq")

    def test_euler_step_constant_field_analytic(self):
        from kelp.numerics.integrators import euler_step
        mesh = _grid_mesh()
        U0 = 0.5
        vf = _vf_elem(mesh, lambda x, y: np.full(x.shape, U0))  # v = 2*u
        st = make_state(np.array([5.3]), np.array([4.7]))
        nn = find_nearest_nodes(mesh, st.lon, st.lat)
        st = st.replace(nearest_node=nn)
        dt = 360.0
        out = euler_step(st, vf, mesh, dt, 1.0, uv_interp="edge_leastsq")
        assert out.status[0] == ParticleStatus.ALIVE
        z0 = np.cos(4.7 * DEG2RAD) * METERS_PER_DEG_LAT
        assert out.lon[0] == pytest.approx(5.3 + dt * U0 / z0, rel=1e-12)
        assert out.lat[0] == pytest.approx(
            4.7 + dt * 2 * U0 / METERS_PER_DEG_LAT, rel=1e-12)

    def test_step_bc_kwarg_threads(self):
        # In a boundary element the freeslip and noslip fits differ, so a
        # step honouring the edge_leastsq_bc kwarg must move differently.
        from kelp.numerics.integrators import rk2_step
        mesh = _grid_mesh()
        vf = _vf_elem(mesh, lambda x, y: np.full(x.shape, 0.7))
        st = make_state(np.array([4.8]), np.array([4.15]))  # inside elem 0
        nn = find_nearest_nodes(mesh, st.lon, st.lat)
        st = st.replace(nearest_node=nn)
        out_f = rk2_step(st, vf, mesh, 300.0, 1.0,
                         uv_interp="edge_leastsq")
        out_n = rk2_step(st, vf, mesh, 300.0, 1.0,
                         uv_interp="edge_leastsq", edge_leastsq_bc="noslip")
        assert out_f.status[0] == ParticleStatus.ALIVE
        assert out_n.status[0] == ParticleStatus.ALIVE
        assert out_f.lon[0] != out_n.lon[0]

    def test_e2e_run_single_realization_analytic(self, tmp_path, monkeypatch):
        # Element-layout FESOM-C file through rk2 + edge_leastsq: the spy
        # proves the registry entry is exercised; the constant field keeps
        # the displacement closed-form.
        from kelp.core.config import ModeConfig, NumericsConfig
        from kelp.tracking import run_single_realization
        import kelp.numerics.interpolate as interp_mod
        calls = {"edge": 0}
        real_edge = interp_mod.interp_edge_leastsq

        def spy_edge(*a, **k):
            calls["edge"] += 1
            return real_edge(*a, **k)

        monkeypatch.setitem(interp_mod._UV_INTERPOLATORS,
                            "edge_leastsq", spy_edge)
        _, reader, mesh, ds = _open_fesomc(tmp_path, "nele", u_val=1.0)
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False)
        numerics = NumericsConfig(dt=3600.0, integrator="rk2",
                                  output_freq=1, uv_interp="edge_leastsq")
        init = make_state(np.array([5.0]), np.array([5.0]))
        rec = run_single_realization(mesh, ds, reader, init, mode, numerics,
                                     start_time_idx=0, n_steps=3)
        ds.close()
        assert calls["edge"] >= 1
        z0 = np.cos(5.0 * DEG2RAD) * METERS_PER_DEG_LAT
        np.testing.assert_allclose(rec.lons[-1][0] - 5.0,
                                   3 * 3600.0 * 1.0 / z0, rtol=1e-12)
        np.testing.assert_allclose(rec.lats[-1][0], 5.0, atol=1e-12)

    def _run_tracking_config(self, nc_path, particles_path, out_path,
                             bc="freeslip"):
        from kelp.core.config import (
            AppConfig, DataConfig, MeshConfig, ModeConfig, NumericsConfig,
            TimeConfig,
        )
        return AppConfig(
            mode=ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0),
            numerics=NumericsConfig(dt=3600.0, integrator="euler",
                                    output_freq=1, uv_interp="edge_leastsq",
                                    edge_leastsq_bc=bc),
            mesh=MeshConfig(type="fesomc", meshpath=str(nc_path)),
            data=DataConfig(ncinput=str(nc_path),
                            inicloud=str(particles_path),
                            output_path=str(out_path),
                            time_averaging="instant"),
            time=TimeConfig(start_time_idx=0, n_steps=2),
        )

    def test_run_tracking_node_file_fails_before_writer(self, tmp_path):
        from kelp.tracking import run_tracking
        nc = _make_fesomc_const(tmp_path, uv_dim="node")
        parts = tmp_path / "p.dat"
        parts.write_text("5.0 5.0\n")
        out = tmp_path / "out.nc"
        cfg = self._run_tracking_config(nc, parts, out)
        with pytest.raises(ValueError, match="edge_leastsq"):
            run_tracking(cfg, output_path=str(out))
        assert not out.exists()

    def test_run_tracking_metadata_records_bc(self, tmp_path):
        import netCDF4
        from kelp.tracking import run_tracking
        nc = _make_fesomc_const(tmp_path, uv_dim="nele")
        parts = tmp_path / "p.dat"
        parts.write_text("5.0 5.0\n")
        out = tmp_path / "out.nc"
        cfg = self._run_tracking_config(nc, parts, out, bc="noslip")
        run_tracking(cfg, output_path=str(out))
        with netCDF4.Dataset(out) as f:
            assert f.getncattr("uv_interp") == "edge_leastsq"
            assert f.getncattr("edge_leastsq_bc") == "noslip"

    def test_track_particles_accepts_edge_leastsq(self, tmp_path):
        from kelp.tracking import track_particles
        _, reader, mesh, ds = _open_fesomc(tmp_path, "nele", u_val=1.0)
        rec = track_particles(mesh, ds, reader,
                              np.array([5.0]), np.array([5.0]),
                              dt=3600.0, n_steps=2, integrator="euler",
                              uv_interp="edge_leastsq",
                              edge_leastsq_bc="noslip")
        ds.close()
        assert rec.lons[-1][0] > 5.0

    def test_rk4_step_bc_kwarg_threads(self):
        # rk4 twin of the rk2 threading test (review round 1, minor 4).
        from kelp.numerics.integrators import rk4_step
        mesh = _grid_mesh()
        vf = _vf_elem(mesh, lambda x, y: np.full(x.shape, 0.7))
        st = make_state(np.array([4.8]), np.array([4.15]))
        nn = find_nearest_nodes(mesh, st.lon, st.lat)
        st = st.replace(nearest_node=nn)
        out_f = rk4_step(st, vf, mesh, 300.0, 1.0,
                         uv_interp="edge_leastsq")
        out_n = rk4_step(st, vf, mesh, 300.0, 1.0,
                         uv_interp="edge_leastsq", edge_leastsq_bc="noslip")
        assert out_f.status[0] == ParticleStatus.ALIVE
        assert out_n.status[0] == ParticleStatus.ALIVE
        assert out_f.lon[0] != out_n.lon[0]

    def test_3d_euler_step_edge_leastsq_per_layer(self):
        # 3D through an ACTUAL integrator step (review round 1, important 3a):
        # per-particle layers slice the field, displacement is analytic.
        from kelp.numerics.integrators import euler_step
        mesh = _grid_mesh()
        mesh.sigma = np.array([0.0, 50.0, 100.0])
        mesh.depth = np.full(mesh.n_nodes, 100.0)
        u3 = np.vstack([np.full(mesh.n_elements, 1.0),
                        np.full(mesh.n_elements, 9.0)])
        vf = VelocityField(u=u3, v=np.zeros_like(u3),
                           eta=np.zeros(mesh.n_nodes),
                           w=np.zeros((3, mesh.n_nodes)))
        st = make_state(np.array([5.3, 5.3]), np.array([4.7, 4.7]),
                        np.array([10.0, 75.0]))
        nn = find_nearest_nodes(mesh, st.lon, st.lat)
        st = st.replace(nearest_node=nn)
        dt = 360.0
        out = euler_step(st, vf, mesh, dt, 1.0, uv_interp="edge_leastsq")
        assert (out.status == ParticleStatus.ALIVE).all()
        z0 = np.cos(4.7 * DEG2RAD) * METERS_PER_DEG_LAT
        assert out.lon[0] == pytest.approx(5.3 + dt * 1.0 / z0, rel=1e-12)
        assert out.lon[1] == pytest.approx(5.3 + dt * 9.0 / z0, rel=1e-12)

    def test_tinterp_e2e_edge_leastsq(self, tmp_path):
        # temporal-interp mode: the LazyLerpField payload flows through the
        # registry wrapper (review round 1, important 3b).  Constant field
        # keeps the displacement closed-form.
        from kelp.readers import extract_time_info
        from kelp.tracking import track_particles
        _, reader, mesh, ds = _open_fesomc(tmp_path, "nele", u_val=1.0)
        file_times, file_dt = extract_time_info(ds)
        rec = track_particles(mesh, ds, reader,
                              np.array([5.0]), np.array([5.0]),
                              dt=1800.0, n_steps=4, integrator="rk2",
                              uv_interp="edge_leastsq",
                              temporal_interp=True, file_times=file_times,
                              file_dt=file_dt)
        ds.close()
        z0 = np.cos(5.0 * DEG2RAD) * METERS_PER_DEG_LAT
        np.testing.assert_allclose(rec.lons[-1][0] - 5.0,
                                   4 * 1800.0 * 1.0 / z0, rtol=1e-12)

    def test_backward_e2e_edge_leastsq(self, tmp_path):
        # backward tracking above the core level (review round 1,
        # important 3c): displacement is minus the forward one.
        from kelp.core.config import ModeConfig, NumericsConfig
        from kelp.tracking import run_single_realization
        _, reader, mesh, ds = _open_fesomc(tmp_path, "nele", u_val=1.0)
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=-1.0)
        numerics = NumericsConfig(dt=3600.0, integrator="euler",
                                  output_freq=1, uv_interp="edge_leastsq")
        init = make_state(np.array([5.5]), np.array([5.0]))
        rec = run_single_realization(mesh, ds, reader, init, mode, numerics,
                                     start_time_idx=3, n_steps=3)
        ds.close()
        z0 = np.cos(5.0 * DEG2RAD) * METERS_PER_DEG_LAT
        np.testing.assert_allclose(rec.lons[-1][0] - 5.5,
                                   -3 * 3600.0 * 1.0 / z0, rtol=1e-12)

    def test_run_tracking_combined_em_nonleastsq_fails_before_writer(
            self, tmp_path):
        # em/m1 + euler auto-resolves to composition='combined', whose drift
        # kernel is leastsq-only.  The conflict must fail BEFORE the writer
        # creates the output file (review round 1, important 2 — closes the
        # pre-existing barycentric hole too).
        from kelp.core.config import DiffusionConfig
        from kelp.tracking import run_tracking
        nc = _make_fesomc_const(tmp_path, uv_dim="nele")
        parts = tmp_path / "p.dat"
        parts.write_text("5.0 5.0\n")
        for uv in ("edge_leastsq", "barycentric"):
            out = tmp_path / f"out_{uv}.nc"
            cfg = self._run_tracking_config(nc, parts, out)
            cfg.numerics.uv_interp = uv
            cfg.numerics.diffusion = DiffusionConfig(enabled=True,
                                                     scheme="em")
            with pytest.raises(ValueError, match="combined"):
                run_tracking(cfg, output_path=str(out))
            assert not out.exists(), uv


def _make_fesomc_seam_no_centers(tmp_path):
    """FESOM-C file with a seam-straddling 2-tri mesh and NO lon_elem/
    lat_elem — forces the reader's element-center fallback."""
    import pandas as pd
    import xarray as xr
    lon = np.array([179.5, -179.5, -179.5, 179.5])
    lat = np.array([0.0, 0.0, 1.0, 1.0])
    nv = np.array([[1, 2, 3, 1], [1, 3, 4, 1]], dtype=np.int32)  # 1-based
    ds = xr.Dataset({
        "lon": (["node"], lon), "lat": (["node"], lat),
        "depth": (["node"], np.full(4, 10.0)),
        "nv": (["nele", "four"], nv),
        "sigma_lev": (["sigma"], np.array([0.0, 1.0])),
        "u2d": (["time", "nele"], np.full((2, 2), 0.5)),
        "v2d": (["time", "nele"], np.zeros((2, 2))),
        "eta": (["time", "node"], np.zeros((2, 4))),
    }, coords={"time": pd.date_range("2020-01-01", periods=2,
                                     freq="3600s")})
    p = tmp_path / "fesomc_seam_nocenters.nc"
    ds.to_netcdf(p)
    return p


class TestFesomcCenterFallbackSeam:
    def test_fallback_centers_anchored_at_seam(self, tmp_path):
        # review round 1: the raw-mean fallback put seam-cell centers ~180
        # deg away (e.g. -59.8 for [179.5, -179.5, -179.5]); every consumer
        # of elem_center_lon (leastsq stencils, edge_leastsq ghosts)
        # inherited the corruption.  Fixed: wrapped-delta mean anchored on
        # the first vertex, gated on raw span > 180 (off-seam bitwise).
        from types import SimpleNamespace
        from kelp.readers import get_reader
        p = _make_fesomc_seam_no_centers(tmp_path)
        r = get_reader("fesomc")
        mesh = r.load_mesh(SimpleNamespace(meshpath=str(p)))
        d180 = np.abs(((mesh.elem_center_lon - 180.0 + 180.0) % 360.0)
                      - 180.0)
        assert (d180 < 1.0).all(), mesh.elem_center_lon

    def test_fallback_centers_off_seam_bitwise_raw_mean(self, tmp_path):
        import pandas as pd
        import xarray as xr
        from types import SimpleNamespace
        from kelp.readers import get_reader
        lon = np.array([10.0, 11.0, 11.0, 10.0])
        lat = np.array([0.0, 0.0, 1.0, 1.0])
        nv = np.array([[1, 2, 3, 1], [1, 3, 4, 1]], dtype=np.int32)
        ds = xr.Dataset({
            "lon": (["node"], lon), "lat": (["node"], lat),
            "depth": (["node"], np.full(4, 10.0)),
            "nv": (["nele", "four"], nv),
            "sigma_lev": (["sigma"], np.array([0.0, 1.0])),
            "u2d": (["time", "nele"], np.full((2, 2), 0.5)),
            "v2d": (["time", "nele"], np.zeros((2, 2))),
            "eta": (["time", "node"], np.zeros((2, 4))),
        }, coords={"time": pd.date_range("2020-01-01", periods=2,
                                         freq="3600s")})
        p = tmp_path / "fesomc_offseam_nocenters.nc"
        ds.to_netcdf(p)
        mesh = get_reader("fesomc").load_mesh(SimpleNamespace(meshpath=str(p)))
        conn = mesh.elem_nodes
        expected = np.array([lon[row[row >= 0]].mean() for row in conn])
        np.testing.assert_array_equal(mesh.elem_center_lon, expected)
