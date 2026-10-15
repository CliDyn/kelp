"""u/v interpolator registry, element areas, vertex-barycentric method."""
import numpy as np
import pytest

from kelp.core.geometry import DEG2RAD
from kelp.core.mesh import Mesh, build_node_to_elems, build_elem_areas


def _mesh_from(lon, lat, elem_nodes, depth_val=10.0):
    """Minimal 2D Mesh from arrays (no search structures needed here)."""
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    en = np.asarray(elem_nodes, np.int32)
    n2e, n2e_cnt = build_node_to_elems(lon.size, en)
    cl = np.array([lon[e[e >= 0]].mean() for e in en])
    ca = np.array([lat[e[e >= 0]].mean() for e in en])
    return Mesh(name="test", nodes_lon=lon, nodes_lat=lat,
                depth=np.full(lon.size, depth_val),
                elem_nodes=en, elem_center_lon=cl, elem_center_lat=ca,
                node_to_elems=n2e, node_to_elems_count=n2e_cnt,
                sigma=np.empty(0), vertical_coord="zlevel")


class TestElemAreas:
    def test_right_triangle_at_equator(self):
        # build_elem_areas now uses SPHERICAL EXCESS x mesh.r_earth^2.
        # Pin against an independent l'Huilier spherical-triangle area on
        # the unit sphere scaled by r_earth^2 (default R_EARTH_STANDARD).
        lon = [0.0, 0.1, 0.0]; lat = [0.0, 0.0, 0.1]
        m = _mesh_from(lon, lat, [[0, 1, 2, -1]])
        A = build_elem_areas(m)
        ref = _poly_area_spherical(list(zip(lon, lat)), m.r_earth)
        # The two stable spherical forms (code: 2*atan2(triple, 1+dots);
        # ref: l'Huilier) agree to ~11 digits — for a tiny 0.1-deg triangle
        # the near-degenerate excess loses a little to roundoff in the
        # (1 + a.b + b.c + a.c) denominator, so 1e-10 not 1e-12 here.
        np.testing.assert_allclose(A[0], ref, rtol=1e-10)

    def test_quad_and_padding_conventions(self):
        # 1x2-deg rectangle as a true quad; a tri padded with -1; a tri
        # stored FESOM-C style (first vertex repeated as 4th).  Areas come
        # from the SPHERICAL-EXCESS formula now, so the independent check
        # is an independent spherical (l'Huilier fan) computation, not a
        # planar shoelace.
        lon = [0.0, 1.0, 1.0, 0.0, 3.0, 4.0, 3.0]
        lat = [0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 1.0]
        en = [[0, 1, 2, 3],      # rectangle
              [4, 5, 6, -1],     # -1 pad
              [4, 5, 6, 4]]      # repeated first vertex
        m = _mesh_from(lon, lat, en)
        A = build_elem_areas(m)
        assert A[1] == pytest.approx(A[2], rel=1e-14)   # pad == repeat
        for ei, verts in ((0, [0, 1, 2, 3]), (1, [4, 5, 6])):
            ref = _poly_area_spherical(
                [(lon[v], lat[v]) for v in verts], m.r_earth)
            np.testing.assert_allclose(A[ei], ref, rtol=1e-12)

    def test_cache_idempotent(self):
        m = _mesh_from([0, 1, 0], [0, 0, 1], [[0, 1, 2, -1]])
        A1 = build_elem_areas(m)
        A2 = build_elem_areas(m)
        assert A1 is A2            # second call returns the cached array

    def test_clockwise_winding_same_area(self):
        # the shoelace must be winding-robust (|signed area|): the same
        # triangle wound CW gives the same positive area as CCW
        lon = [0.0, 0.1, 0.0]; lat = [0.0, 0.0, 0.1]
        m_ccw = _mesh_from(lon, lat, [[0, 1, 2, -1]])
        m_cw = _mesh_from(lon, lat, [[0, 2, 1, -1]])   # reversed order
        a_ccw = build_elem_areas(m_ccw)[0]
        a_cw = build_elem_areas(m_cw)[0]
        assert a_cw > 0.0
        assert a_cw == pytest.approx(a_ccw, rel=1e-14)

    def test_dateline_straddling_element_area_translation_invariant(self):
        # build_elem_areas anchors its projection origin on the first
        # vertex.  The shoelace is shift-invariant so AREA values were
        # already correct for connected cells (the real dateline bug was
        # in the containment bbox, see TestBarycentricElementLayout::
        # test_dateline_uniform_field_exact for the RED-proven path); this
        # is a robustness regression pinning translation invariance at the
        # seam.
        lon_seam = [179.5, -179.6, 179.7]; lat = [0.0, 0.0, 1.0]
        lon_off = [-179.5, -178.6, -179.3]                # shifted +1 deg
        m_seam = _mesh_from(lon_seam, lat, [[0, 1, 2, -1]])
        m_off = _mesh_from(lon_off, lat, [[0, 1, 2, -1]])
        a_seam = build_elem_areas(m_seam)[0]
        a_off = build_elem_areas(m_off)[0]
        assert a_seam > 0.0
        np.testing.assert_allclose(a_seam, a_off, rtol=1e-9)


# --- independent spherical-area references (in-test, formula-by-name) -------

def _xyz(lon_deg, lat_deg):
    """Unit-sphere Cartesian for a lon/lat pair (degrees)."""
    lo = np.deg2rad(lon_deg); la = np.deg2rad(lat_deg)
    return np.array([np.cos(la) * np.cos(lo),
                     np.cos(la) * np.sin(lo),
                     np.sin(la)])


def _tri_excess_lhuilier(a_xyz, b_xyz, c_xyz):
    """Spherical-triangle area on the UNIT sphere via l'Huilier's theorem.

    Independent of the atan2 form used in build_elem_areas: side lengths
    are great-circle angles, area = 4*atan(sqrt(tan(s/2) tan((s-a)/2)
    tan((s-b)/2) tan((s-c)/2))) with s the semiperimeter.
    """
    def _ang(p, q):
        return np.arccos(np.clip(np.dot(p, q), -1.0, 1.0))
    a = _ang(b_xyz, c_xyz)
    b = _ang(a_xyz, c_xyz)
    c = _ang(a_xyz, b_xyz)
    s = 0.5 * (a + b + c)
    t = (np.tan(s / 2) * np.tan((s - a) / 2)
         * np.tan((s - b) / 2) * np.tan((s - c) / 2))
    return 4.0 * np.arctan(np.sqrt(max(t, 0.0)))


def _poly_area_spherical(lonlats, r_earth):
    """Fan-triangulated spherical polygon area (m^2) via l'Huilier."""
    verts = [_xyz(lo, la) for lo, la in lonlats]
    exc = 0.0
    for k in range(1, len(verts) - 1):
        exc += _tri_excess_lhuilier(verts[0], verts[k], verts[k + 1])
    return exc * r_earth * r_earth


def _planar_cos_area(lonlats, r_earth):
    """The OLD planar shoelace x cos(centroid lat) estimate (m^2)."""
    lons = np.array([p[0] for p in lonlats])
    lats = np.array([p[1] for p in lonlats])
    lon0 = lons.mean(); lat0 = lats.mean()
    c = np.cos(np.deg2rad(lat0))
    x = ((lons - lon0 + 180) % 360 - 180) * DEG2RAD * r_earth * c
    y = (lats - lat0) * DEG2RAD * r_earth
    s = 0.0
    n = len(lons)
    for i in range(n):
        j = (i + 1) % n
        s += x[i] * y[j] - x[j] * y[i]
    return 0.5 * abs(s)


def _rotate_xyz(v, axis, angle):
    """Rodrigues rotation of a 3-vector about a unit axis."""
    axis = axis / np.linalg.norm(axis)
    return (v * np.cos(angle)
            + np.cross(axis, v) * np.sin(angle)
            + axis * np.dot(axis, v) * (1 - np.cos(angle)))


class TestSphericalAreas:
    """Pole-safe spherical-excess fallback (deliverable 3 / 5c)."""

    def test_polar_cap_matches_lhuilier(self):
        # A triangle near the north pole: the OLD planar cos(lat) formula
        # collapses x by cos(89.8 deg) ~ 0.0035 and is badly wrong; the
        # spherical-excess fallback matches an independent l'Huilier area.
        lon = [0.0, 120.0, 240.0]; lat = [89.8, 89.8, 89.8]
        m = _mesh_from(lon, lat, [[0, 1, 2, -1]])
        m.r_earth = 6_371_000.0
        A = build_elem_areas(m)[0]
        ref = _poly_area_spherical(list(zip(lon, lat)), 6_371_000.0)
        np.testing.assert_allclose(A, ref, rtol=1e-12)
        # Motivation pin: the planar cos formula the code used to run would
        # have been off by far more than 1% on this polar cap.
        planar = _planar_cos_area(list(zip(lon, lat)), 6_371_000.0)
        assert abs(planar - ref) / ref > 0.01

    def test_rotation_invariant(self):
        # The SAME spherical triangle rotated from the pole down to the
        # equator must give the same area — the property the planar
        # cos-formula lacks.  Rotate the three unit vectors by a 3D
        # rotation and feed the rotated lon/lat back in.
        lon = [0.0, 120.0, 240.0]; lat = [89.8, 89.8, 89.8]
        m_pole = _mesh_from(lon, lat, [[0, 1, 2, -1]])
        m_pole.r_earth = 6_371_000.0
        a_pole = build_elem_areas(m_pole)[0]

        # Rotate each vertex 89.8 deg about the x-axis so the cap centre
        # lands near the equator, then convert back to lon/lat.
        axis = np.array([1.0, 0.0, 0.0])
        ang = np.deg2rad(89.8)
        rot_ll = []
        for lo, la in zip(lon, lat):
            v = _rotate_xyz(_xyz(lo, la), axis, ang)
            rlat = np.rad2deg(np.arcsin(np.clip(v[2], -1.0, 1.0)))
            rlon = np.rad2deg(np.arctan2(v[1], v[0]))
            rot_ll.append((rlon, rlat))
        rlon = [p[0] for p in rot_ll]; rlat = [p[1] for p in rot_ll]
        m_eq = _mesh_from(rlon, rlat, [[0, 1, 2, -1]])
        m_eq.r_earth = 6_371_000.0
        a_eq = build_elem_areas(m_eq)[0]
        np.testing.assert_allclose(a_eq, a_pole, rtol=1e-9)
        # And the planar estimate at the pole is NOT rotation-invariant.
        planar_pole = _planar_cos_area(list(zip(lon, lat)), 6_371_000.0)
        planar_eq = _planar_cos_area(rot_ll, 6_371_000.0)
        assert abs(planar_pole - planar_eq) / planar_eq > 0.01

    def test_r_earth_squared_scaling(self):
        # Area scales with r_earth^2 (the radius enters only as r^2).
        lon = [0.0, 0.1, 0.0]; lat = [0.0, 0.0, 0.1]
        m1 = _mesh_from(lon, lat, [[0, 1, 2, -1]]); m1.r_earth = 6_371_000.0
        m2 = _mesh_from(lon, lat, [[0, 1, 2, -1]]); m2.r_earth = 6_400_000.0
        a1 = build_elem_areas(m1)[0]
        a2 = build_elem_areas(m2)[0]
        np.testing.assert_allclose(
            a2 / a1, (6_400_000.0 / 6_371_000.0) ** 2, rtol=1e-12)


from kelp.numerics.interpolate import (
    VelocityField, get_interpolator, get_interpolator_single,
    interp_leastsq, interp_leastsq_single,
    interp_barycentric, interp_barycentric_single,
)
from kelp.core.mesh import build_search, find_nearest_nodes
from kelp.core.particles import make_state


def _grid_mesh():
    """3x3-node, 8-tri mesh around (5,5) — canonical test mesh, with search."""
    lonn = [4, 5, 6, 4, 5, 6, 4, 5, 6]
    latn = [4, 4, 4, 5, 5, 5, 6, 6, 6]
    en = [[0, 1, 4, -1], [0, 4, 3, -1], [1, 2, 5, -1], [1, 5, 4, -1],
          [3, 4, 7, -1], [3, 7, 6, -1], [4, 5, 8, -1], [4, 8, 7, -1]]
    m = _mesh_from(lonn, latn, en)
    build_search(m)
    return m


def _quad_mesh():
    """2x2 quad cells (9 nodes, 4 quads) around (5,5)."""
    lonn = [4, 5, 6, 4, 5, 6, 4, 5, 6]
    latn = [4, 4, 4, 5, 5, 5, 6, 6, 6]
    en = [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]]
    m = _mesh_from(lonn, latn, en)
    build_search(m)
    return m


class TestRegistry:
    def test_leastsq_is_the_existing_function(self):
        assert get_interpolator("leastsq") is interp_leastsq
        assert get_interpolator_single("leastsq") is interp_leastsq_single

    def test_barycentric_registered(self):
        assert get_interpolator("barycentric") is interp_barycentric
        assert (get_interpolator_single("barycentric")
                is interp_barycentric_single)

    def test_unknown_name_lists_valid(self):
        with pytest.raises(ValueError, match="barycentric"):
            get_interpolator("cubic")
        with pytest.raises(ValueError, match="leastsq"):
            get_interpolator_single("cubic")

    def test_config_validation(self):
        from kelp.core.config import NumericsConfig
        NumericsConfig(uv_interp="barycentric")          # accepted
        with pytest.raises(ValueError, match="uv_interp"):
            NumericsConfig(uv_interp="nearest")

    def test_config_list_matches_registry(self):
        # config.py must not import the registry; this pin keeps the two
        # name lists in sync instead
        from kelp.core.config import _VALID_UV_INTERP
        from kelp.numerics.interpolate import (
            _UV_INTERPOLATORS, _UV_INTERPOLATORS_SINGLE,
        )
        assert set(_VALID_UV_INTERP) == set(_UV_INTERPOLATORS)
        assert set(_VALID_UV_INTERP) == set(_UV_INTERPOLATORS_SINGLE)


def _vf_elem(mesh, fn):
    """Element-layout field: u_e = fn(center_lon, center_lat), v = 2*u."""
    u = fn(mesh.elem_center_lon, mesh.elem_center_lat)
    return VelocityField(u=u, v=2.0 * u, eta=np.zeros(mesh.n_nodes), w=None)


def _vf_node(mesh, fn):
    u = fn(mesh.nodes_lon, mesh.nodes_lat)
    return VelocityField(u=u, v=2.0 * u, eta=np.zeros(mesh.n_nodes), w=None,
                         uv_layout="node")


class TestBarycentricElementLayout:
    def test_uniform_field_exact(self):
        for mesh in (_grid_mesh(), _quad_mesh()):
            vf = _vf_elem(mesh, lambda x, y: np.full(x.shape, 0.7))
            st = make_state(np.array([5.2, 4.6]), np.array([4.7, 5.3]))
            nn = find_nearest_nodes(mesh, st.lon, st.lat)
            U, V, ok = interp_barycentric(st, mesh, vf, nn)
            assert ok.all()
            np.testing.assert_allclose(U, 0.7, rtol=1e-14)
            np.testing.assert_allclose(V, 1.4, rtol=1e-14)

    def test_within_stencil_bounds(self):
        mesh = _grid_mesh()
        rng = np.random.default_rng(6)
        u_e = rng.uniform(-1, 1, mesh.n_elements)
        vf = VelocityField(u=u_e, v=u_e.copy(),
                           eta=np.zeros(mesh.n_nodes), w=None)
        pts = rng.uniform(4.1, 5.9, size=(40, 2))
        st = make_state(pts[:, 0], pts[:, 1])
        nn = find_nearest_nodes(mesh, st.lon, st.lat)
        U, V, ok = interp_barycentric(st, mesh, vf, nn)
        assert ok.all()
        assert (U >= u_e.min() - 1e-12).all() and (U <= u_e.max() + 1e-12).all()

    def test_continuity_across_shared_edge(self):
        # Two points an EPSILON to either side of the shared edge between
        # nodes 1 (5,4) and 4 (5,5) — provably in DIFFERENT triangles
        # ([0,1,4] west of lon=5, [1,5,4] east), so a discontinuous
        # implementation cannot pass via both queries landing in the same
        # element.
        mesh = _grid_mesh()
        rng = np.random.default_rng(9)
        u_e = rng.uniform(-1, 1, mesh.n_elements)
        vf = VelocityField(u=u_e, v=u_e.copy(),
                           eta=np.zeros(mesh.n_nodes), w=None)
        eps = 1e-9
        u_w, _, ok_w = interp_barycentric_single(5.0 - eps, 4.5, 0,
                                                 mesh, vf, 4)
        u_e_, _, ok_e = interp_barycentric_single(5.0 + eps, 4.5, 0,
                                                  mesh, vf, 1)
        assert ok_w and ok_e
        # continuity: the jump across 2*eps must be ~1e-9 of the field
        # range; a per-element-discontinuous method jumps by O(range)
        assert abs(u_w - u_e_) < 1e-6 * (u_e.max() - u_e.min())

    def test_time_dir_flips_sign(self):
        mesh = _grid_mesh()
        vf = _vf_elem(mesh, lambda x, y: np.full(x.shape, 0.5))
        u_f, _, _ = interp_barycentric_single(5.1, 4.8, 0, mesh, vf, 4, 1.0)
        u_b, _, _ = interp_barycentric_single(5.1, 4.8, 0, mesh, vf, 4, -1.0)
        assert u_b == pytest.approx(-u_f, rel=1e-14)

    def test_outside_mesh_fails(self):
        mesh = _grid_mesh()
        vf = _vf_elem(mesh, lambda x, y: np.ones(x.shape))
        u, v, ok = interp_barycentric_single(9.0, 9.0, 0, mesh, vf, 8)
        assert not ok and u == 0.0 and v == 0.0

    def test_area_weighted_vertex_value(self):
        # Two triangles sharing the edge (0,1): areas A and 3A, element
        # values 1.0 / 5.0.  Node-1 vertex value = (A*1 + 3A*5)/(4A) = 4.0.
        # Query just inside tri1 hugging node 1: blended value ~4.0
        # (node-0 vertex value is also 4.0; node-2 only belongs to tri1).
        lon = [0.0, 0.1, 0.0, 0.2]
        lat = [0.0, 0.0, 0.1, -0.3]
        en = [[0, 1, 2, -1], [0, 1, 3, -1]]
        m = _mesh_from(lon, lat, en); build_search(m)
        from kelp.core.mesh import build_elem_areas
        A = build_elem_areas(m)
        assert A[1] == pytest.approx(3 * A[0], rel=2e-5)
        u_e = np.array([1.0, 5.0])
        vf = VelocityField(u=u_e, v=u_e.copy(), eta=np.zeros(4), w=None)
        u, v, ok = interp_barycentric_single(0.0999, 0.0001, 0, m, vf, 1)
        assert ok
        assert u == pytest.approx(4.0, abs=0.02)

    def test_dateline_uniform_field_exact(self):
        # Element-layout barycentric on a tri straddling +/-180.  A uniform
        # field must reproduce exactly, which requires (1) containment to
        # SUCCEED at the seam (the _element_contains bbox prefilter used a
        # raw-mean centroid + raw min/max that rejected every interior
        # point of a seam cell) and (2) the area-weighted vertex projection.
        # RED-proven: without the dateline-safe bbox, ok is False here.
        lon = [179.5, -179.8, 179.8]; lat = [0.0, 0.0, 1.0]
        en = [[0, 1, 2, -1]]
        m = _mesh_from(lon, lat, en); build_search(m)
        u_e = np.array([0.7])
        vf = VelocityField(u=u_e, v=2.0 * u_e, eta=np.zeros(3), w=None)
        ref = 179.5
        dnode = (np.asarray(lon) - ref + 180.0) % 360.0 - 180.0
        qlon = ref + dnode.mean(); qlat = float(np.mean(lat))
        u, v, ok = interp_barycentric_single(qlon, qlat, 0, m, vf, 0)
        assert ok
        assert u == pytest.approx(0.7, rel=1e-12)
        assert v == pytest.approx(1.4, rel=1e-12)


class TestBarycentricNodeLayout:
    def test_linear_field_exact(self):
        # node-layout headline pin: globally linear field reproduced
        # exactly (within each cell the metric projection is affine, so
        # the actual error is machine-level; 5e-6 is pure slack)
        for mesh in (_grid_mesh(), _quad_mesh()):
            vf = _vf_node(mesh, lambda x, y: 0.3 + 0.5 * x - 0.2 * y)
            pts = np.array([[5.2, 4.7], [4.4, 5.6], [5.0, 5.0], [4.9, 4.3]])
            st = make_state(pts[:, 0], pts[:, 1])
            nn = find_nearest_nodes(mesh, st.lon, st.lat)
            U, V, ok = interp_barycentric(st, mesh, vf, nn)
            assert ok.all()
            expected = 0.3 + 0.5 * pts[:, 0] - 0.2 * pts[:, 1]
            np.testing.assert_allclose(U, expected, rtol=5e-6)
            np.testing.assert_allclose(V, 2 * expected, rtol=5e-6)

    def test_quad_fourth_vertex_participates(self):
        # Quad 0 is [0, 1, 4, 3]; the point (4.6, 4.3) lies inside the fan
        # triangle (0, 1, 4) (local coords (0.6, 0.3): y < x), where fan
        # weights give vertex 3 EXACTLY zero.  Put the only nonzero value
        # on node 3: a fan implementation returns u == 0.0 here, Wachspress
        # returns a strictly positive blend.
        mesh = _quad_mesh()
        u_n = np.zeros(mesh.n_nodes); u_n[3] = 1.0   # node (4, 5)
        vf = VelocityField(u=u_n, v=u_n.copy(),
                           eta=np.zeros(mesh.n_nodes), w=None,
                           uv_layout="node")
        u, v, ok = interp_barycentric_single(4.6, 4.3, 0, mesh, vf, 4)
        assert ok and u > 1e-3

    def test_3d_per_particle_layers(self):
        mesh = _grid_mesh()
        u3 = np.vstack([np.full(mesh.n_nodes, 1.0),
                        np.full(mesh.n_nodes, 9.0)])
        vf = VelocityField(u=u3, v=2 * u3, eta=np.zeros(mesh.n_nodes),
                           w=np.zeros((3, mesh.n_nodes)), uv_layout="node")
        st = make_state(np.array([5.2, 4.6]), np.array([4.7, 5.3]),
                        np.array([1.0, 50.0]))
        nn = find_nearest_nodes(mesh, st.lon, st.lat)
        layers = np.array([0, 1], dtype=np.int32)
        U, V, ok = interp_barycentric(st, mesh, vf, nn, 1.0, layers)
        assert ok.all()
        np.testing.assert_allclose(U, [1.0, 9.0], rtol=1e-14)

    def test_3d_element_layout_layers(self):
        # element layout with per-particle layers: area-averaging on the
        # particle's own layer slice
        mesh = _grid_mesh()
        u3 = np.vstack([np.full(mesh.n_elements, 1.0),
                        np.full(mesh.n_elements, 9.0)])
        vf = VelocityField(u=u3, v=2 * u3, eta=np.zeros(mesh.n_nodes),
                           w=np.zeros((3, mesh.n_nodes)))
        st = make_state(np.array([5.2, 4.6]), np.array([4.7, 5.3]),
                        np.array([1.0, 50.0]))
        nn = find_nearest_nodes(mesh, st.lon, st.lat)
        layers = np.array([0, 1], dtype=np.int32)
        U, V, ok = interp_barycentric(st, mesh, vf, nn, 1.0, layers)
        assert ok.all()
        np.testing.assert_allclose(U, [1.0, 9.0], rtol=1e-14)

    def test_dead_particles_untouched(self):
        # inactive particles: U=V=0, ok=False (the interp_leastsq
        # convention, zero-init + active-only fill)
        from kelp.core.particles import ParticleStatus
        mesh = _grid_mesh()
        vf = _vf_node(mesh, lambda x, y: np.ones(x.shape))
        st = make_state(np.array([5.2, 4.6]), np.array([4.7, 5.3]))
        status = st.status.copy(); status[1] = ParticleStatus.OUTSIDE
        st = st.replace(status=status)
        nn = find_nearest_nodes(mesh, st.lon, st.lat)
        U, V, ok = interp_barycentric(st, mesh, vf, nn)
        assert ok[0] and not ok[1]
        assert U[1] == 0.0 and V[1] == 0.0

    def test_dateline_linear_field_exact(self):
        # A node-layout linear field on a tri straddling +/-180 must be
        # reproduced exactly at an interior point.  The barycentric core
        # anchors its projection origin on the first vertex; a raw-mean
        # origin would land ~180 deg from the cell and corrupt the wrapped
        # deltas, breaking the affine reproduction.
        lon = [179.5, -179.8, 179.8]; lat = [0.0, 0.0, 1.0]
        en = [[0, 1, 2, -1]]
        m = _mesh_from(lon, lat, en); build_search(m)
        # field linear in the WRAPPED lon offset from the first vertex
        ref = 179.5
        dnode = (np.asarray(lon) - ref + 180.0) % 360.0 - 180.0
        u_n = 10.0 + 3.0 * dnode               # node values
        vf = VelocityField(u=u_n, v=2 * u_n, eta=np.zeros(3), w=None,
                           uv_layout="node")
        # query the (anchored) centroid: lon0 = ref + mean(dnode)
        qlon = ref + dnode.mean()
        qlat = float(np.mean(lat))
        u, v, ok = interp_barycentric_single(qlon, qlat, 0, m, vf, 0)
        assert ok
        dq = (qlon - ref + 180.0) % 360.0 - 180.0
        expected = 10.0 + 3.0 * dq
        assert u == pytest.approx(expected, rel=1e-6)
        assert v == pytest.approx(2 * expected, rel=1e-6)

    def test_repeated_first_vertex_tri_matches_padded(self):
        # FESOM-C stores triangles as quads either with -1 padding or by
        # repeating the FIRST vertex as the 4th entry.  Both encodings of
        # the SAME triangle must interpolate identically AND succeed —
        # this is the user-visible encoding-equivalence contract, and the
        # repeated-vertex path through the kernel had no coverage at all.
        # NOTE (review finding, mutation-probed): the kernel's dedup line
        # itself is VALUE-SHADOWED for this encoding — without it the row
        # routes Wachspress -> degenerate-corner guard -> fan fallback,
        # where the duplicate vertex gets weight exactly 0, i.e. the same
        # value.  The dedup is a clean-path contract (nv==3 primitive,
        # no allocation churn), not a value contract; no value assertion
        # can pin it, by construction of the fallback chain.
        lon = [0.0, 0.1, 0.0]
        lat = [0.0, 0.0, 0.1]
        u_n = np.array([1.0, 2.0, 4.0])
        for en in ([[0, 1, 2, -1]], [[0, 1, 2, 0]]):
            m = _mesh_from(lon, lat, en)
            build_search(m)
            vf = VelocityField(u=u_n, v=2 * u_n, eta=np.zeros(3), w=None,
                               uv_layout="node")
            u, v, ok = interp_barycentric_single(0.03, 0.03, 0, m, vf, 0)
            assert ok, en
            # classical tri barycentric expectation, computed in-test:
            # weights for P=(0.03,0.03) in tri (0,0),(0.1,0),(0,0.1) are
            # w = (1 - x/0.1 - y/0.1, x/0.1, y/0.1) = (0.4, 0.3, 0.3)
            expected = 0.4 * 1.0 + 0.3 * 2.0 + 0.3 * 4.0
            assert u == pytest.approx(expected, rel=5e-6), en


class TestDatelineSeamQuad:
    def test_seam_quad_containment_and_uniform_field(self):
        # Symmetric seam quad: raw-mean lon0 ~ 0 deg (the broken case the
        # triangle tests cannot reach — their wrapped deltas stay
        # contiguous).  Interior points on BOTH sides of the seam must be
        # contained and interpolate the uniform field exactly.
        lonn = [179.5, -179.5, -179.5, 179.5]
        latn = [0.0, 0.0, 1.0, 1.0]
        en = [[0, 1, 2, 3]]
        m = _mesh_from(lonn, latn, en)
        build_search(m)
        u_n = np.full(4, 0.7)
        vf = VelocityField(u=u_n, v=2 * u_n, eta=np.zeros(4), w=None,
                           uv_layout="node")
        for qlon in (179.9, -179.9, 179.999, -179.999):
            nn = int(find_nearest_nodes(m, np.array([qlon]),
                                        np.array([0.5]))[0])
            u, v, ok = interp_barycentric_single(qlon, 0.5, 0, m, vf, nn)
            assert ok, qlon
            assert u == pytest.approx(0.7, rel=1e-12), qlon

    def test_seam_quad_linear_field(self):
        # linear in the wrapped offset from the west edge: pins that the
        # projected geometry (not just containment) is correct on the seam
        lonn = [179.5, -179.5, -179.5, 179.5]
        latn = [0.0, 0.0, 1.0, 1.0]
        en = [[0, 1, 2, 3]]
        m = _mesh_from(lonn, latn, en)
        build_search(m)
        off = lambda lon: ((lon - 179.5 + 180.0) % 360.0) - 180.0
        u_n = np.array([0.0 + 2.0 * off(l) for l in lonn])
        vf = VelocityField(u=u_n, v=np.zeros(4), eta=np.zeros(4), w=None,
                           uv_layout="node")
        qlon, qlat = -179.8, 0.25      # 0.7 deg east of the west edge
        nn = int(find_nearest_nodes(m, np.array([qlon]),
                                    np.array([qlat]))[0])
        u, v, ok = interp_barycentric_single(qlon, qlat, 0, m, vf, nn)
        assert ok
        assert u == pytest.approx(2.0 * off(qlon), rel=1e-9)
