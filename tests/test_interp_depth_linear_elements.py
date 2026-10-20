"""Depth-linear u/v on element layout: ``leastsq``, ``edge_leastsq`` and
element ``barycentric``.

``interp_leastsq`` dispatches an element-layout ``"layer_linear"`` field to
the depth-linear twins (``_interp_leastsq_dl`` ->
``_loop_interp_leastsq_3d_dl`` -> ``_interp_leastsq_core_dl``): the plane
fit of ``_interp_leastsq_core`` over the elements around the nearest node,
each element read linearly in depth between its own layer centres (clipped
to ``p[0] .. p[last_e]``).  An element whose bed lies above the particle is
a wall and is left out of the fit, as elements beyond the coast are absent
from a coastal node's ring.  When the wet elements are too few for the
fit (fewer than three, or a singular fit) the value is their area-weighted
mean; when no element is wet, the containing element read at the
particle depth (below its own bed too: its last usable point).  With every
element wet the arithmetic is the layer core's, so a singular all-wet ring
stays frozen.  A node-layout depth-linear field is refused.

``interp_edge_leastsq`` (``_interp_edge_leastsq_dl`` ->
``_loop_interp_edge_lsq_3d_dl`` -> ``_interp_edge_leastsq_core_dl``): the
fit of ``_interp_edge_leastsq_core`` over the containing element and its
edge neighbours.  A neighbour dry at the particle depth is a boundary edge
whose ghost ``edge_leastsq_bc`` builds from the own element's value at that
depth, as for a coastal edge.  A dry containing element: the area-weighted
mean of its wet edge neighbours, no ghosts; none wet, the containing
element at its last usable point.

``interp_barycentric`` on element layout (``_interp_barycentric_dl`` ->
``_loop_interp_bary_elem_dl`` -> ``_interp_bary_elem_core_dl``): each vertex
value is the area-weighted mean of its wet ring elements at the particle
depth; a vertex without one is left out and the weights of the others
renormalised, a negative one counted as zero (their equal-weight mean when
those weights sum below 1e-12); no vertex left, the containing element at
its last usable point.

All synthetic: the 5x5-node test mesh of ``test_3d_analytical`` (32
triangles, 0.5 deg spacing, centred at 5E 5N) with levels
``[0, 10, 30, 60, 100, 150]`` m (layer centres ``[5, 20, 45, 80, 125]``)
and the element bottoms set by hand as the FESOM2 reader declares them
(``metadata["uv_data_last_elem"]``, ``metadata["elem_bed"]``).
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from kelp.core.io import LazyLerpField, VelocityField
from kelp.core.mesh import (
    Mesh,
    _find_containing_element_local,
    build_elem_areas,
    build_elem_neighbors,
    build_node_to_elems,
    build_search,
    find_containing_elements_walk,
    find_nearest_nodes,
)
from kelp.core.particles import ParticleStatus, make_state
from kelp.numerics import interpolate as ip
from kelp.numerics import vertical_uv as vu

from tests._bitwise import assert_bitwise
from tests.test_3d_analytical import _make_mesh
from tests.test_interp_depth_linear import _lam

LEVELS = [0.0, 10.0, 30.0, 60.0, 100.0, 150.0]
#: layer centres (the data points of "layer_linear")
P = np.array([5.0, 20.0, 45.0, 80.0, 125.0])
M = P.size

#: the interior node at (5E, 5N) and its element ring, in node_to_elems order
CENTRE = 12
RING = [11, 12, 13, 18, 19, 20]

#: inside the ring of CENTRE (nearest node 12)
CENTRE_PTS = [(5.1, 4.93), (4.88, 5.07), (5.2, 5.15), (4.95, 4.8),
              (5.17, 4.86), (4.81, 5.2)]

#: the interior nodes of the mesh
INTERIOR = [6, 7, 8, 11, 12, 13, 16, 17, 18]

E_IDX = np.arange(32, dtype=np.float64)
#: per-element constants (depth-constant field) and slopes (linear field)
#: (u in 0.1 .. 1.1 m/s, v in -1.1 .. -0.1 m/s down to 150 m: away from
#: zero, so relative tolerances are meaningful)
CU = 0.6 + 0.2 * np.sin(0.9 * E_IDX)
CV = -0.6 + 0.2 * np.cos(1.3 * E_IDX)
BU = 0.002 * np.cos(0.7 * E_IDX)
BV = -0.002 * np.sin(1.1 * E_IDX)


def _geo_last(bed):
    """The reader's declaration: the deepest layer whose lower level is at
    or above the bed (-1 when none is)."""
    lower = np.asarray(LEVELS[1:])
    return (np.searchsorted(lower, np.asarray(bed, dtype=np.float64),
                            side="right") - 1).astype(np.int32)


def _mesh(bed=150.0, *, decl=None, areas=None):
    """The test mesh; element beds ``bed`` (scalar or ``(E,)``), the
    declaration geometric unless ``decl`` is given, model-provided element
    areas when ``areas`` is given."""
    mesh = _make_mesh(5.0, 5.0, span=1.0, nx=5, ny=5, z_levels=LEVELS,
                      bottom_depth=150.0)
    n_el = mesh.n_elements
    bed = np.array(np.broadcast_to(np.asarray(bed, dtype=np.float64),
                                   (n_el,)))
    if decl is None:
        decl = _geo_last(bed)
    mesh.metadata["uv_data_last_elem"] = np.asarray(decl, dtype=np.int32)
    mesh.metadata["elem_bed"] = bed
    if areas is not None:
        mesh._elem_areas = np.asarray(areas, dtype=np.float64)
    return mesh


def _table(mesh):
    return vu.depth_points(mesh, "layer_linear", layout="element")


def _efield(mesh, fu, fv, *, below=np.nan):
    """``u[k, e] = fu(e, p_k)`` (v likewise) at the points ``k <= last_e``
    of the mesh's element table; ``below`` at the points below."""
    dp = _table(mesh)
    n_el = mesh.n_elements
    u = np.full((M, n_el), below, dtype=np.float64)
    v = np.full((M, n_el), below, dtype=np.float64)
    for e in range(n_el):
        k = int(dp.last[e]) + 1
        u[:k, e] = fu(e, P[:k])
        v[:k, e] = fv(e, P[:k])
    n = mesh.n_nodes
    return VelocityField(u=u, v=v, eta=np.zeros(n),
                         w=np.zeros((len(LEVELS), n)),
                         uv_layout="element", uv_vertical="layer_linear")


def _const_field(mesh, **kw):
    return _efield(mesh, lambda e, p: np.full(p.shape, CU[e]),
                   lambda e, p: np.full(p.shape, CV[e]), **kw)


def _linear_field(mesh, **kw):
    return _efield(mesh, lambda e, p: CU[e] + BU[e] * p,
                   lambda e, p: CV[e] + BV[e] * p, **kw)


def _state(pts, z, *, dead=()):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    z = np.broadcast_to(np.asarray(z, dtype=np.float64), (pts.shape[0],))
    st = make_state(pts[:, 0].copy(), pts[:, 1].copy(), z.copy())
    for i in dead:
        st.status[i] = ParticleStatus.OUTSIDE
    return st


def _run(mesh, vel, st, td=1.0, layer=0):
    nn = find_nearest_nodes(mesh, st.lon, st.lat)
    U, V, ok = ip.interp_leastsq(st, mesh, vel, nn, td, layer)
    return U, V, ok, nn


def _values_at(mesh, vel, z):
    """Every element's depth-linear value at ``z`` (the kernel's read:
    ``uv_bracket`` + ``uv_at``, a NaN ``z`` read as ``+inf``), wet or dry;
    never NaN, since the read never passes the element's last usable
    point."""
    dp = _table(mesh)
    zq = np.inf if np.isnan(z) else z
    uz = np.empty(mesh.n_elements)
    vz = np.empty(mesh.n_elements)
    for e in range(mesh.n_elements):
        k, a = vu.uv_bracket(dp.p, dp.last[e], zq)
        uz[e] = vu.uv_at(vel.u, e, k, a)
        vz[e] = vu.uv_at(vel.v, e, k, a)
    return uz, vz


def _layer_fit(mesh, x, y, nd, td, uvals, vvals, ring=None):
    """The LAYER core ``_interp_leastsq_core`` at ``(x, y)`` around node
    ``nd`` on per-element values ``uvals``/``vvals``; ``ring`` replaces the
    node's element ring (a sub-ring in the same order: a mesh where the
    other elements are absent, like elements beyond a coast)."""
    n2e = mesh.node_to_elems
    cnt = mesh.node_to_elems_count
    if ring is not None:
        n2e = n2e.copy()
        cnt = cnt.copy()
        n2e[nd, :] = -1
        n2e[nd, :len(ring)] = ring
        cnt[nd] = len(ring)
    return ip._interp_leastsq_core(
        float(x), float(y), int(nd), float(td), cnt, n2e,
        mesh.elem_center_lon, mesh.elem_center_lat, uvals, vvals)


def _containing(mesh, x, y, nd):
    e, ok = _find_containing_element_local(
        x, y, int(nd), mesh.node_to_elems, mesh.node_to_elems_count,
        mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat)
    return int(e), bool(ok)


def _ring(mesh, nd):
    return [int(e) for e in
            mesh.node_to_elems[nd, :mesh.node_to_elems_count[nd]]]


def test_fixture():
    """The ring of the centre node and the hand-written declaration rule."""
    mesh = _mesh()
    assert _ring(mesh, CENTRE) == RING
    st = _state(CENTRE_PTS, 50.0)
    assert np.all(find_nearest_nodes(mesh, st.lon, st.lat) == CENTRE)
    assert_bitwise(_geo_last([150.0, 30.0, 25.0, 40.0, 9.0]),
                   np.array([4, 1, 0, 1, -1], dtype=np.int32), "decl")
    dp = _table(mesh)
    assert_bitwise(dp.p, P, "p")
    assert np.all(dp.last == M - 1)


# ---------------------------------------------------------------------------
# Every member wet: the layer arithmetic
# ---------------------------------------------------------------------------

def _interior_points(rng, per_node):
    pts, nds = [], []
    for nd in INTERIOR:
        x0 = 4.0 + 0.5 * (nd % 5)
        y0 = 4.0 + 0.5 * (nd // 5)
        for _ in range(per_node):
            pts.append((x0 + rng.uniform(-0.2, 0.2),
                        y0 + rng.uniform(-0.2, 0.2)))
            nds.append(nd)
    return pts, np.array(nds)


#: depths where every member reads a stored value (``alpha < 0``): the
#: data points, at or above the first, below every column's last point
#: (down to the bed, 150 m: still wet)
DIRECT_Z = [-1.0, 0.0, 2.0, 5.0, 20.0, 45.0, 80.0, 125.0, 130.0, 149.9,
            150.0]


def test_depth_constant_matches_layer():
    """A depth-constant field (``c_e`` at every usable point of element
    ``e``, NaN below ``last_e``; declared prefixes of 2..5 points, every bed
    150 m: every member wet) gives the layer core's answer on ``c``:
    BITWISE wherever every member reads a stored value, and between data
    points within ``8 ulp`` of ``max|c|`` over the ring (each blend
    ``(1 - a) c + a c`` may move ``c`` by 1 ulp, and the fit's sums pass
    that on).
    Measured worst over the 1,512 evaluations between points (of 2,700;
    u and v): 4 ulp of ``max|c|``."""
    rng = np.random.default_rng(41)
    decl = rng.integers(1, M, size=32).astype(np.int32)
    mesh = _mesh(decl=decl)
    vel = _const_field(mesh)
    dp = _table(mesh)
    assert_bitwise(dp.last, decl, "declared prefixes")
    assert np.isnan(vel.u).any()
    pts, nds = _interior_points(rng, 6)
    between = sorted(set(rng.uniform(5.0, 125.0, size=14).tolist())
                     - set(P.tolist()))
    for td in (1.0, -1.0):
        for z_set, exact in ((DIRECT_Z, True), (between, False)):
            for z in z_set:
                st = _state(pts, z)
                U, V, ok, nn = _run(mesh, vel, st, td)
                assert np.all(nn == nds)
                assert ok.all(), z
                for i, (x, y) in enumerate(pts):
                    Ur, Vr, okr = _layer_fit(mesh, x, y, nds[i], td, CU, CV)
                    assert okr
                    if exact:
                        assert_bitwise(np.array([U[i], V[i]]),
                                       np.array([Ur, Vr]), f"z={z} i={i}")
                        continue
                    ring = _ring(mesh, nds[i])
                    ulp_u = np.spacing(np.max(np.abs(CU[ring])))
                    ulp_v = np.spacing(np.max(np.abs(CV[ring])))
                    du = abs(U[i] - Ur) / ulp_u
                    dv = abs(V[i] - Vr) / ulp_v
                    assert du <= 8.0 and dv <= 8.0, (z, i, du, dv)


def test_linear_in_depth_exact():
    """``u = a_e + b_e z`` per element (declared prefixes of 2..5 points,
    every member wet): the result is the plane fit of the exact values
    ``a_e + b_e clip(z, p[0], p[last_e])``, to 1e-12 relative; the layer
    answer (every element at the centre of the depth's layer) is not."""
    rng = np.random.default_rng(43)
    decl = rng.integers(1, M, size=32).astype(np.int32)
    mesh = _mesh(decl=decl)
    vel = _linear_field(mesh)
    dp = _table(mesh)
    pts, nds = _interior_points(rng, 4)
    for td in (1.0, -1.0):
        for z in rng.uniform(5.0, 140.0, size=12):
            st = _state(pts, z)
            U, V, ok, _ = _run(mesh, vel, st, td)
            assert ok.all()
            zc = np.clip(z, P[0], P[dp.last])
            ue = CU + BU * zc
            ve = CV + BV * zc
            for i, (x, y) in enumerate(pts):
                Ur, Vr, okr = _layer_fit(mesh, x, y, nds[i], td, ue, ve)
                assert okr
                assert abs(U[i] - Ur) <= 1e-12 * abs(Ur), (z, i, U[i], Ur)
                assert abs(V[i] - Vr) <= 1e-12 * abs(Vr), (z, i, V[i], Vr)
    # 52.5 m lies in layer 2 (30..60 m, centre 45 m): the layer answer
    # reads every element at 45 m (or its last point above)
    mesh = _mesh()
    vel = _linear_field(mesh)
    st = _state(CENTRE_PTS, 52.5)
    U, V, ok, _ = _run(mesh, vel, st)
    ul, vl = _values_at(mesh, vel, 45.0)
    for i, (x, y) in enumerate(CENTRE_PTS):
        Ul, Vl, _ = _layer_fit(mesh, x, y, CENTRE, 1.0, ul, vl)
        assert abs(U[i] - Ul) > 1e-4 and abs(V[i] - Vl) > 1e-4, (i, U[i], Ul)


# ---------------------------------------------------------------------------
# Walls at depth
# ---------------------------------------------------------------------------

def _wall_mesh(dry_e=13, dry_bed=30.0):
    bed = np.full(32, 150.0)
    bed[dry_e] = dry_bed
    return _mesh(bed)


def _wall_field(mesh, dry_e=13, big=1000.0):
    """The linear field with ``big`` in element ``dry_e``'s usable points
    (finite: the record check passes; any weight it gets shows)."""
    return _efield(
        mesh, lambda e, p: big + 0.0 * p if e == dry_e else CU[e] + BU[e] * p,
        lambda e, p: -big + 0.0 * p if e == dry_e else CV[e] + BV[e] * p)


def test_dry_member_left_out():
    """Element 13 of the centre ring has its bed at 30 m and 1000 m/s in
    its column.  Below 30 m it is a wall: the result is BITWISE the layer
    core on the ring without element 13 (a mesh where it is absent), with
    every other element's value at ``z``; at and above its bed it is in
    the fit."""
    mesh = _wall_mesh()
    vel = _wall_field(mesh)
    reduced = [e for e in RING if e != 13]
    for td in (1.0, -1.0):
        for z in (30.5, 45.0, 52.5, 80.0, 125.0, 140.0, 150.0):
            st = _state(CENTRE_PTS, z)
            U, V, ok, nn = _run(mesh, vel, st, td)
            assert ok.all() and np.all(nn == CENTRE)
            uz, vz = _values_at(mesh, vel, z)
            for i, (x, y) in enumerate(CENTRE_PTS):
                Ur, Vr, okr = _layer_fit(mesh, x, y, CENTRE, td, uz, vz,
                                         ring=reduced)
                assert okr
                assert_bitwise(np.array([U[i], V[i]]), np.array([Ur, Vr]),
                               f"z={z} i={i}")
                assert abs(U[i]) < 2.0
        for z in (5.0, 20.0, 30.0):
            st = _state(CENTRE_PTS, z)
            U, V, ok, _ = _run(mesh, vel, st, td)
            uz, vz = _values_at(mesh, vel, z)
            for i, (x, y) in enumerate(CENTRE_PTS):
                Ur, Vr, _ = _layer_fit(mesh, x, y, CENTRE, td, uz, vz)
                assert_bitwise(np.array([U[i], V[i]]), np.array([Ur, Vr]),
                               f"wet z={z} i={i}")
                assert abs(U[i]) > 10.0          # the 1000 m/s is in the fit


def test_member_at_bed_is_wet():
    """A particle exactly at a neighbour's bed (40 m; the neighbour's last
    usable point is the layer centre at 20 m): the neighbour is wet and
    reads its last point — the full-ring fit, bitwise; one ulp deeper it is
    a wall (the reduced ring); the velocity is continuous from above and
    jumps across the bed."""
    mesh = _wall_mesh(dry_bed=40.0)
    assert _table(mesh).last[13] == 1
    vel = _wall_field(mesh, big=3.0)
    reduced = [e for e in RING if e != 13]
    for td in (1.0, -1.0):
        z_bed = 40.0
        z_below = float(np.nextafter(40.0, np.inf))
        st = _state(CENTRE_PTS, z_bed)
        U, V, ok, _ = _run(mesh, vel, st, td)
        Ub, Vb, okb, _ = _run(mesh, vel, _state(CENTRE_PTS, z_below), td)
        Ua, _, _, _ = _run(mesh, vel, _state(CENTRE_PTS, 40.0 - 1e-7), td)
        assert ok.all() and okb.all()
        uz, vz = _values_at(mesh, vel, z_bed)
        uzb, vzb = _values_at(mesh, vel, z_below)
        assert uz[13] == 3.0
        for i, (x, y) in enumerate(CENTRE_PTS):
            Ur, Vr, _ = _layer_fit(mesh, x, y, CENTRE, td, uz, vz)
            assert_bitwise(np.array([U[i], V[i]]), np.array([Ur, Vr]),
                           f"at bed i={i}")
            Ur, Vr, _ = _layer_fit(mesh, x, y, CENTRE, td, uzb, vzb,
                                   ring=reduced)
            assert_bitwise(np.array([Ub[i], Vb[i]]), np.array([Ur, Vr]),
                           f"below bed i={i}")
        assert np.all(np.abs(Ua - U) <= 1e-8)
        assert np.all(np.abs(Ub - U) > 1e-3)


def test_single_element_trench_no_freeze():
    """Every ring element but 19 has its bed at 30 m; element 19 is a
    trench (150 m).  At 80 m the fit over one element is singular (the
    layer core on that ring freezes the particle); the depth-linear core
    returns element 19's value at ``z`` (its area-weighted mean)."""
    bed = np.full(32, 150.0)
    bed[[11, 12, 13, 18, 20]] = 30.0
    mesh = _mesh(bed)
    vel = _linear_field(mesh)
    pts = [(4.9, 5.2), (4.95, 5.15), (4.8, 5.22)]
    for x, y in pts:
        nd = find_nearest_nodes(mesh, np.array([x]), np.array([y]))[0]
        assert nd == CENTRE and _containing(mesh, x, y, nd) == (19, True)
    a19 = build_elem_areas(mesh)[19]
    for td in (1.0, -1.0):
        for z in (30.5, 80.0, 133.0, 150.0):
            U, V, ok, _ = _run(mesh, vel, _state(pts, z), td)
            assert ok.all()
            uz, vz = _values_at(mesh, vel, z)
            assert np.all(U == td * ((a19 * uz[19]) / a19))
            assert np.all(V == td * ((a19 * vz[19]) / a19))
            assert np.all(np.abs(U - td * uz[19])
                          <= np.spacing(abs(uz[19])))
            _, _, okr = _layer_fit(mesh, pts[0][0], pts[0][1], CENTRE, td,
                                   uz, vz, ring=[19])
            assert not okr                     # the fit alone would freeze


def test_two_wet_members_area_mean():
    """Elements 18 and 19 wet (150 m), the rest of the ring dry (30 m),
    model-provided areas 3:1: the value is ``(A18 u18 + A19 u19) /
    (A18 + A19)`` at ``z`` in ring order, bitwise, and differs from the
    equal-weight mean."""
    bed = np.full(32, 150.0)
    bed[[11, 12, 13, 20]] = 30.0
    areas = np.full(32, 2.0e9)
    areas[18] = 6.0e9
    mesh = _mesh(bed, areas=areas)
    vel = _linear_field(mesh)
    pts = [(4.85, 5.1), (4.9, 5.2)]
    for td in (1.0, -1.0):
        for z in (45.0, 52.5, 140.0):
            U, V, ok, nn = _run(mesh, vel, _state(pts, z), td)
            assert ok.all() and np.all(nn == CENTRE)
            uz, vz = _values_at(mesh, vel, z)
            a18, a19 = areas[18], areas[19]
            Ue = td * ((a18 * uz[18] + a19 * uz[19]) / (a18 + a19))
            Ve = td * ((a18 * vz[18] + a19 * vz[19]) / (a18 + a19))
            assert np.all(U == Ue) and np.all(V == Ve), (z, U, Ue)
            assert np.all(np.abs(U - td * 0.5 * (uz[18] + uz[19])) > 1e-2)


def test_singular_wet_fit_area_mean():
    """Elements 11, 12, 13 wet with their centres moved onto one parallel,
    the other three dry: three wet members, but the fit is singular — the
    area-weighted mean of the three.  With every member wet the same
    centres give a regular six-member fit."""
    bed = np.full(32, 150.0)
    bed[[18, 19, 20]] = 30.0
    mesh = _mesh(bed)
    lat = mesh.elem_center_lat.copy()
    lat[[11, 12, 13]] = 4.75
    mesh.elem_center_lat = lat
    vel = _linear_field(mesh)
    areas = build_elem_areas(mesh)
    for td in (1.0, -1.0):
        U, V, ok, _ = _run(mesh, vel, _state(CENTRE_PTS, 80.0), td)
        assert ok.all()
        uz, vz = _values_at(mesh, vel, 80.0)
        _, _, okr = _layer_fit(mesh, *CENTRE_PTS[0], CENTRE, td, uz, vz,
                               ring=[11, 12, 13])
        assert not okr
        num_u = num_v = den = 0.0
        for e in (11, 12, 13):
            num_u += areas[e] * uz[e]
            num_v += areas[e] * vz[e]
            den += areas[e]
        assert np.all(U == td * (num_u / den))
        assert np.all(V == td * (num_v / den))
        # every member wet (20 m, above every bed): the six-member fit
        U, V, ok, _ = _run(mesh, vel, _state(CENTRE_PTS, 20.0), td)
        uz, vz = _values_at(mesh, vel, 20.0)
        assert ok.all()
        for i, (x, y) in enumerate(CENTRE_PTS):
            Ur, Vr, _ = _layer_fit(mesh, x, y, CENTRE, td, uz, vz)
            assert_bitwise(np.array([U[i], V[i]]), np.array([Ur, Vr]),
                           f"all wet i={i}")


def test_all_dry_reads_containing_element():
    """Every element of the centre ring is shallower than the particle;
    the deepest of them is 18 (60 m), the particle lies in 13 (25 m): the
    value is element 13 at its last usable point, also for a NaN depth
    (dry wherever the bed is finite).  A particle outside the mesh whose
    nearest node's ring is dry gets ``ok = False``."""
    bed = np.full(32, 150.0)
    bed[RING] = [30.0, 40.0, 25.0, 60.0, 35.0, 45.0]
    bed[0] = 30.0
    mesh = _mesh(bed)
    vel = _linear_field(mesh)
    dp = _table(mesh)
    pts = [(5.2, 4.9), (5.15, 4.95), (5.22, 4.83)]
    for x, y in pts:
        assert _containing(mesh, x, y, CENTRE) == (13, True)
    assert int(np.argmax(bed[RING])) == RING.index(18)
    k13 = int(dp.last[13])
    assert k13 == 0
    for td in (1.0, -1.0):
        for z in (61.0, 80.0, 1.0e6, np.nan):
            U, V, ok, nn = _run(mesh, vel, _state(pts, z), td)
            assert ok.all() and np.all(nn == CENTRE)
            assert np.all(U == td * vel.u[k13, 13]), (z, U)
            assert np.all(V == td * vel.v[k13, 13]), (z, V)
            assert np.all(U != td * vel.u[dp.last[18], 18])
        out = _state([(3.9, 3.95)], 80.0)
        U, V, ok, nn = _run(mesh, vel, out, td)
        assert nn[0] == 0 and not ok[0] and U[0] == 0.0 and V[0] == 0.0


def test_all_wet_singular_stays_frozen():
    """Corner nodes: node 4 has two elements (6, 7), node 0 one (0).  With
    every member wet the fit is singular in linear mode as in layer mode:
    ``ok = False``.  With element 7 dry the area mean of the wet member
    (element 6 at ``z``) is returned."""
    mesh = _mesh()
    vel = _linear_field(mesh)
    assert _ring(mesh, 4) == [6, 7] and _ring(mesh, 0) == [0]
    pts = [(5.9, 4.1), (4.1, 4.05)]
    for td in (1.0, -1.0):
        U, V, ok, nn = _run(mesh, vel, _state(pts, 52.5), td)
        assert list(nn) == [4, 0]
        uz, vz = _values_at(mesh, vel, 52.5)
        for i, (x, y) in enumerate(pts):
            _, _, okr = _layer_fit(mesh, x, y, nn[i], td, uz, vz)
            assert not okr
        assert not ok.any() and np.all(U == 0.0) and np.all(V == 0.0)
    bed = np.full(32, 150.0)
    bed[7] = 30.0
    mesh = _mesh(bed)
    vel = _linear_field(mesh)
    for td in (1.0, -1.0):
        U, V, ok, _ = _run(mesh, vel, _state(pts[:1], 52.5), td)
        uz, vz = _values_at(mesh, vel, 52.5)
        a6 = build_elem_areas(mesh)[6]
        assert ok[0]
        assert U[0] == td * ((a6 * uz[6]) / a6)
        assert V[0] == td * ((a6 * vz[6]) / a6)


def _mixed_mesh():
    """Beds from 9 m to 150 m and declared prefixes shorter than the beds
    allow, so walls, both fallbacks and partial prefixes all occur."""
    rng = np.random.default_rng(47)
    bed = rng.choice([9.0, 25.0, 30.0, 40.0, 60.0, 90.0, 150.0], size=32)
    decl = np.minimum(_geo_last(bed), rng.integers(0, M, size=32))
    return _mesh(bed, decl=decl.astype(np.int32))


def _regimes(mesh, st, nn):
    """Counts of alive particles by regime: every ring element wet; a
    wall with at least three wet elements; one or two wet; none wet."""
    bed = _table(mesh).bed
    out = {"all_wet": 0, "wall": 0, "few_wet": 0, "all_dry": 0}
    for i in np.flatnonzero(st.alive_mask):
        zq = np.inf if np.isnan(st.z[i]) else st.z[i]
        ring = _ring(mesh, nn[i])
        n_wet = sum(zq <= bed[e] for e in ring)
        key = ("all_wet" if n_wet == len(ring) else "all_dry" if n_wet == 0
               else "few_wet" if n_wet < 3 else "wall")
        out[key] += 1
    return out


def _mixed_state():
    rng = np.random.default_rng(53)
    n = 300
    lon = rng.uniform(3.9, 6.1, size=n)
    lat = rng.uniform(3.9, 6.1, size=n)
    z = rng.choice([-1.0, 5.0, 12.0, 25.0, 30.0, 33.3, 45.0, 60.0, 71.0,
                    90.0, 110.0, 150.0, 400.0, np.nan], size=n)
    return _state(np.column_stack([lon, lat]), z, dead=(3, 17))


def test_nan_below_last_never_read():
    """NaN below each element's last usable point never reaches a result:
    the same field with +inf / 1e300 below ``last`` gives the same bits
    over particles in every regime (walls, both fallbacks, NaN depth,
    outside the mesh)."""
    mesh = _mixed_mesh()
    dp = _table(mesh)
    assert (dp.last < M - 1).sum() > 10
    vel_nan = _linear_field(mesh)
    vel_inf = _linear_field(mesh, below=np.inf)
    vel_big = _linear_field(mesh, below=1.0e300)
    st = _mixed_state()
    nn = find_nearest_nodes(mesh, st.lon, st.lat)
    assert all(c >= 30 for c in _regimes(mesh, st, nn).values())
    for td in (1.0, -1.0):
        ref = _run(mesh, vel_nan, st, td)[:3]
        assert ref[2].sum() > 200
        assert np.all(np.isfinite(ref[0])) and np.all(np.isfinite(ref[1]))
        for other in (vel_inf, vel_big):
            got = _run(mesh, other, st, td)[:3]
            for a, b, name in zip(got, ref, ("U", "V", "ok")):
                assert_bitwise(a, b, f"{name} td={td}")


# ---------------------------------------------------------------------------
# Dispatch, refusals, fallback
# ---------------------------------------------------------------------------

def test_dispatch_reads_state_z_ignores_layer():
    """The layer argument is ignored (scalar or per particle), each
    particle is read at ``state.z``, dead particles get zeros and
    ``ok = False``, and a lazy blend of two fields is read like the eager
    one."""
    mesh = _mixed_mesh()
    vel = _linear_field(mesh)
    st = _mixed_state()
    ref = _run(mesh, vel, st)[:3]
    assert not ref[2][3] and ref[0][3] == 0.0 and ref[1][17] == 0.0
    deep = _mesh()
    U2 = _run(deep, _linear_field(deep),
              _state([CENTRE_PTS[0]] * 2, [45.0, 52.5]))[0]
    assert abs(U2[1] - U2[0]) > 1e-4
    for layer in (3, np.full(st.n_particles, 2, dtype=np.int32)):
        got = _run(mesh, vel, st, layer=layer)[:3]
        for a, b, name in zip(got, ref, ("U", "V", "ok")):
            assert_bitwise(a, b, f"{name} layer={layer!r}")
    lazy = LazyLerpField(vel, _linear_field(mesh), 0.5)
    got = _run(mesh, lazy, st)[:3]
    for a, b, name in zip(got, ref, ("U", "V", "ok")):
        assert_bitwise(a, b, f"{name} lazy")


def test_record_checked_before_any_read():
    """A non-finite value at a usable point is refused, naming the
    element; a passing field is stamped as checked for the table."""
    mesh = _mesh()
    vel = _linear_field(mesh)
    vel.u[2, 13] = np.nan
    st = _state(CENTRE_PTS, 50.0)
    nn = find_nearest_nodes(mesh, st.lon, st.lat)
    with pytest.raises(ValueError, match=r"u = nan is non-finite at element "
                                         r"13, point 2"):
        ip.interp_leastsq(st, mesh, vel, nn, 1.0, 0)
    good = _linear_field(mesh)
    assert not vu.is_checked(good, mesh)
    ip.interp_leastsq(st, mesh, good, nn, 1.0, 0)
    assert vu.is_checked(good, mesh)


@pytest.mark.parametrize("placement", ["level", "layer_linear"])
def test_node_field_refused(placement):
    """A depth-linear NODE field: refused before any slicing (``layer=7``
    is out of range) with the layout reason, which names the layout."""
    mesh = _mesh()
    m = len(LEVELS) if placement == "level" else M
    n = mesh.n_nodes
    vel = VelocityField(u=np.zeros((m, n)), v=np.zeros((m, n)),
                        eta=np.zeros(n), w=np.zeros((len(LEVELS), n)),
                        uv_layout="node", uv_vertical=placement)
    st = _state(CENTRE_PTS, 50.0)
    nn = find_nearest_nodes(mesh, st.lon, st.lat)
    msg = ("interp_leastsq: depth-linear horizontal velocity "
           f"(uv_vertical={placement!r}) " + ip._DL_LAYOUT)
    assert "node" in ip._DL_LAYOUT and "barycentric" in ip._DL_LAYOUT
    for call in (lambda: ip.interp_leastsq(st, mesh, vel, nn, 1.0, 7),
                 lambda: ip.get_interpolator("leastsq")(
                     st, mesh, vel, nn, 1.0,
                     np.full(st.n_particles, 7, dtype=np.int32))):
        with pytest.raises(ValueError, match="^" + re.escape(msg) + "$"):
            call()


def test_python_fallback_equals_numba(monkeypatch):
    """The pure-Python core (``.py_func``) and the wrapper's no-Numba loop
    give the compiled bits in every regime."""
    mesh = _mixed_mesh()
    vel = _linear_field(mesh)
    st = _mixed_state()
    nn = find_nearest_nodes(mesh, st.lon, st.lat)
    dp = _table(mesh)
    areas = build_elem_areas(mesh)
    for td in (1.0, -1.0):
        ref = ip.interp_leastsq(st, mesh, vel, nn, td, 0)
        with monkeypatch.context() as mp:
            mp.setattr(ip, "HAS_NUMBA", False)
            got = ip.interp_leastsq(st, mesh, vel, nn, td, 0)
        for a, b, name in zip(got, ref, ("U", "V", "ok")):
            assert_bitwise(a, b, f"{name} wrapper td={td}")
        for i in range(st.n_particles):
            args = (float(st.lon[i]), float(st.lat[i]), float(st.z[i]),
                    int(nn[i]), mesh.node_to_elems_count, mesh.node_to_elems,
                    mesh.elem_center_lon, mesh.elem_center_lat, td,
                    dp.p, dp.last, dp.bed, areas, mesh.elem_nodes,
                    mesh.nodes_lon, mesh.nodes_lat, vel.u, vel.v)
            c = ip._interp_leastsq_core_dl(*args)
            p = ip._interp_leastsq_core_dl.py_func(*args)
            assert_bitwise(np.array(p[:2]), np.array(c[:2]), f"core i={i}")
            assert p[2] == c[2]
            if st.alive_mask[i]:
                assert_bitwise(np.array(c[:2]),
                               np.array([ref[0][i], ref[1][i]]),
                               f"core vs wrapper i={i}")


# ---------------------------------------------------------------------------
# leastsq: a containing element outside the ring
# ---------------------------------------------------------------------------

def _obtuse_mesh():
    """Two obtuse triangles on the common edge A-B: element 0 =
    ``[A, B, C]`` above it, element 1 = ``[A, D, B]`` below it, ``D``
    close to the edge.  A particle in element 0 near the edge has ``D``
    (node 3) as its nearest node, whose ring is ``[1]``: the containing
    element lies outside that ring (the 2-ring scan of
    ``_find_containing_element_local`` finds it).  Beds: element 0 150 m,
    element 1 30 m."""
    lon = np.array([5.0, 6.0, 5.5, 5.5])
    lat = np.array([5.0, 5.0, 5.15, 4.95])
    en = np.array([[0, 1, 2], [0, 3, 1]], dtype=np.int32)
    n2e, cnt = build_node_to_elems(lon.size, en)
    mesh = Mesh(name="obtuse", nodes_lon=lon, nodes_lat=lat,
                depth=np.full(lon.size, 150.0), elem_nodes=en,
                elem_center_lon=np.array([lon[e].mean() for e in en]),
                elem_center_lat=np.array([lat[e].mean() for e in en]),
                node_to_elems=n2e, node_to_elems_count=cnt,
                sigma=np.array(LEVELS))
    build_search(mesh)
    bed = np.array([150.0, 30.0])
    mesh.metadata["uv_data_last_elem"] = _geo_last(bed)
    mesh.metadata["elem_bed"] = bed
    return mesh


def test_all_dry_ring_reads_wet_containing_element_at_z():
    """The nearest node's ring (element 1, bed 30 m) is dry below 30 m;
    the containing element 0 lies outside that ring and is wet down to
    150 m: it is read at the particle depth (``uv_bracket``), not at its
    last usable point."""
    mesh = _obtuse_mesh()
    vel = _linear_field(mesh)
    dp = _table(mesh)
    pts = [(5.5, 5.01), (5.45, 5.02), (5.56, 5.015)]
    for x, y in pts:
        nd = find_nearest_nodes(mesh, np.array([x]), np.array([y]))[0]
        assert nd == 3 and _ring(mesh, nd) == [1]
        assert _containing(mesh, x, y, nd) == (0, True)
    for td in (1.0, -1.0):
        for z in (45.0, 52.5, 80.0, 110.0):
            U, V, ok, _ = _run(mesh, vel, _state(pts, z), td)
            uz, vz = _values_at(mesh, vel, z)
            assert ok.all()
            assert np.all(U == td * uz[0]) and np.all(V == td * vz[0]), z
            assert np.all(U != td * vel.u[dp.last[0], 0])


# ---------------------------------------------------------------------------
# edge_leastsq
# ---------------------------------------------------------------------------

BCS = ("freeslip", "noslip")

#: barycentric coordinates of the test points inside one element
LAMS = np.array([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6],
                 [0.34, 0.33, 0.33], [0.1, 0.45, 0.45],
                 [0.45, 0.1, 0.45]])


def _pts_in(mesh, e, lams=LAMS):
    """Points inside element ``e`` at the barycentric coordinates ``lams``
    of its vertices."""
    nds = mesh.elem_nodes[e][mesh.elem_nodes[e] >= 0]
    return [(float(lam @ mesh.nodes_lon[nds]),
             float(lam @ mesh.nodes_lat[nds])) for lam in lams]


def _big_field(mesh, dry, big=1000.0):
    """The linear field with ``big`` (u) and ``-big`` (v) in the usable
    points of the elements ``dry`` (finite: the record check passes; any
    weight they get shows)."""
    return _efield(
        mesh,
        lambda e, p: big + 0.0 * p if e in dry else CU[e] + BU[e] * p,
        lambda e, p: -big + 0.0 * p if e in dry else CV[e] + BV[e] * p)


def _run_edge(mesh, vel, st, td=1.0, bc="freeslip", *, elems=None,
              layer=0):
    nn = find_nearest_nodes(mesh, st.lon, st.lat)
    U, V, ok = ip.interp_edge_leastsq(st, mesh, vel, nn, td, layer, bc=bc,
                                      elems=elems)
    return U, V, ok, nn


def _layer_edge(mesh, x, y, nd, td, bc, uvals, vvals, absent=()):
    """The LAYER core ``_interp_edge_leastsq_core`` at ``(x, y)`` (nearest
    node ``nd``, the core's own containment search) on per-element values;
    the elements ``absent`` are removed from the edge adjacency, so their
    edges are boundary edges (a mesh where they are absent, like elements
    beyond a coast)."""
    nb = build_elem_neighbors(mesh)
    if absent:
        nb = nb.copy()
        nb[np.isin(nb, list(absent))] = -1
    return ip._interp_edge_leastsq_core(
        float(x), float(y), int(nd), -1, float(td), ip._bc_to_int(bc),
        mesh.node_to_elems, mesh.node_to_elems_count, mesh.elem_nodes,
        mesh.nodes_lon, mesh.nodes_lat, nb,
        mesh.elem_center_lon, mesh.elem_center_lat, uvals, vvals)


def _edge_members(mesh, e):
    """The stencil of ``edge_leastsq`` in element ``e``: ``e`` and its
    edge neighbours."""
    return [int(e)] + [int(n) for n in build_elem_neighbors(mesh)[e]
                       if n >= 0]


#: (containing element, dry edge neighbours): an interior element with one
#: wall and with two; a corner element whose only neighbour is dry (every
#: edge a ghost); a coastal element with a wall
EDGE_WALL_CASES = [(19, (26,)), (19, (20, 18)), (0, (1,)), (15, (22,))]


def _check_dry_neighbour_is_boundary(bc):
    """Below the 30 m bed of the dry neighbours (1000 m/s in their
    columns): BITWISE the layer core on a mesh where they are absent —
    their edges boundary edges, whose ghosts are built from the own
    element's value at ``z`` — every element at its value at ``z``; at and
    above that bed: the full stencil, bitwise."""
    for own, dry in EDGE_WALL_CASES:
        bed = np.full(32, 150.0)
        bed[list(dry)] = 30.0
        mesh = _mesh(bed)
        assert set(dry) <= set(build_elem_neighbors(mesh)[own].tolist())
        vel = _big_field(mesh, dry)
        pts = _pts_in(mesh, own)
        st = _state(pts, 50.0)
        nn = find_nearest_nodes(mesh, st.lon, st.lat)
        for (x, y), nd in zip(pts, nn):
            assert _containing(mesh, x, y, nd) == (own, True)
        for td in (1.0, -1.0):
            for z in (30.5, 45.0, 52.5, 80.0, 125.0, 140.0, 150.0):
                U, V, ok, nn = _run_edge(mesh, vel, _state(pts, z), td, bc)
                assert ok.all()
                uz, vz = _values_at(mesh, vel, z)
                for i, (x, y) in enumerate(pts):
                    Ur, Vr, okr = _layer_edge(mesh, x, y, nn[i], td, bc,
                                              uz, vz, absent=dry)
                    assert okr
                    assert_bitwise(np.array([U[i], V[i]]),
                                   np.array([Ur, Vr]),
                                   f"{own} {dry} z={z} i={i}")
                assert np.all(np.abs(U) < 5.0) and np.all(np.abs(V) < 5.0)
            for z in (5.0, 20.0, 30.0):
                U, V, ok, nn = _run_edge(mesh, vel, _state(pts, z), td, bc)
                assert ok.all()
                uz, vz = _values_at(mesh, vel, z)
                for i, (x, y) in enumerate(pts):
                    Ur, Vr, _ = _layer_edge(mesh, x, y, nn[i], td, bc,
                                            uz, vz)
                    assert_bitwise(np.array([U[i], V[i]]),
                                   np.array([Ur, Vr]),
                                   f"wet {own} {dry} z={z} i={i}")
                assert np.abs(U).max() > 10.0    # the 1000 m/s is in the fit


def test_edge_dry_neighbour_is_boundary_freeslip():
    """``edge_leastsq_bc: freeslip``: a neighbour dry at ``z`` is a
    boundary edge with the free-slip ghost (the own value at ``z``
    mirrored across the edge)."""
    _check_dry_neighbour_is_boundary("freeslip")


def test_edge_dry_neighbour_is_boundary_noslip():
    """``edge_leastsq_bc: noslip``: a neighbour dry at ``z`` is a boundary
    edge with the no-slip ghost (the own value at ``z`` negated)."""
    _check_dry_neighbour_is_boundary("noslip")


def test_edge_own_element_dry():
    """The containing element 19 is dry below 30 m (1000 m/s in its
    column, never read while dry); its edge neighbours ``[20, 26, 18]``
    have beds 150, 150 and 40 m and model-provided areas 6:2:1.  The value
    is the area-weighted mean of the wet neighbours in slot order, no
    ghosts, for both boundary conditions: 20, 26 and 18 down to 40 m, 20
    and 26 below.  With every neighbour dry too (beds 50, 50, 40 m),
    element 19 at its last usable point, also for a NaN depth."""
    bed = np.full(32, 150.0)
    bed[19] = 30.0
    bed[18] = 40.0
    areas = np.full(32, 2.0e9)
    areas[20] = 6.0e9
    areas[18] = 1.0e9
    mesh = _mesh(bed, areas=areas)
    assert build_elem_neighbors(mesh)[19, :3].tolist() == [20, 26, 18]
    vel = _big_field(mesh, (19,))
    pts = _pts_in(mesh, 19)
    a18, a20, a26 = areas[18], areas[20], areas[26]
    for bc in BCS:
        for td in (1.0, -1.0):
            for z in (30.5, 35.0, 40.0):
                U, V, ok, _ = _run_edge(mesh, vel, _state(pts, z), td, bc)
                uz, vz = _values_at(mesh, vel, z)
                den = a20 + a26 + a18
                Ue = td * ((a20 * uz[20] + a26 * uz[26] + a18 * uz[18])
                           / den)
                Ve = td * ((a20 * vz[20] + a26 * vz[26] + a18 * vz[18])
                           / den)
                assert ok.all()
                assert np.all(U == Ue) and np.all(V == Ve), (bc, z, U, Ue)
            for z in (40.5, 52.5, 80.0, 140.0):
                U, V, ok, _ = _run_edge(mesh, vel, _state(pts, z), td, bc)
                uz, vz = _values_at(mesh, vel, z)
                Ue = td * ((a20 * uz[20] + a26 * uz[26]) / (a20 + a26))
                Ve = td * ((a20 * vz[20] + a26 * vz[26]) / (a20 + a26))
                assert ok.all()
                assert np.all(U == Ue) and np.all(V == Ve), (bc, z, U, Ue)
                assert np.all(np.abs(U - td * 0.5 * (uz[20] + uz[26]))
                              > 1e-4)
    bed[[20, 26]] = 50.0
    mesh = _mesh(bed, areas=areas)
    vel = _linear_field(mesh)
    dp = _table(mesh)
    k19 = int(dp.last[19])
    assert k19 == 1
    for bc in BCS:
        for td in (1.0, -1.0):
            for z in (50.5, 80.0, 1.0e6, np.nan):
                U, V, ok, _ = _run_edge(mesh, vel, _state(pts, z), td, bc)
                assert ok.all()
                assert np.all(U == td * vel.u[k19, 19]), (bc, z, U)
                assert np.all(V == td * vel.v[k19, 19]), (bc, z, V)
                for e in (18, 20, 26):
                    assert np.all(U != td * vel.u[dp.last[e], e])


def _all_elem_points(mesh, lams):
    pts, own = [], []
    for e in range(mesh.n_elements):
        pts += _pts_in(mesh, e, lams)
        own += [e] * len(lams)
    return pts, own


def test_edge_depth_constant_matches_layer():
    """A depth-constant field (``c_e`` at every usable point of element
    ``e``, NaN below ``last_e``; declared prefixes of 2..5 points, every bed
    150 m: every member wet) gives the layer core's answer on ``c`` in
    every element, coastal ones with their ghosts, for both boundary
    conditions: BITWISE wherever every member reads a stored value, and
    between data points within ``8 ulp`` of ``max|c|`` over the stencil
    (u and v together: the free-slip ghost mixes the components).
    Measured worst over the 7,680 evaluations between points (of 16,128;
    u and v): 3 ulp."""
    rng = np.random.default_rng(59)
    decl = rng.integers(1, M, size=32).astype(np.int32)
    mesh = _mesh(decl=decl)
    vel = _const_field(mesh)
    assert np.isnan(vel.u).any()
    pts, own = _all_elem_points(mesh, LAMS[:3])
    between = sorted(set(rng.uniform(5.0, 125.0, size=10).tolist())
                     - set(P.tolist()))
    for bc in BCS:
        for td in (1.0, -1.0):
            for z_set, exact in ((DIRECT_Z, True), (between, False)):
                for z in z_set:
                    U, V, ok, nn = _run_edge(mesh, vel, _state(pts, z), td,
                                             bc)
                    assert ok.all(), z
                    for i, (x, y) in enumerate(pts):
                        Ur, Vr, okr = _layer_edge(mesh, x, y, nn[i], td, bc,
                                                  CU, CV)
                        assert okr
                        if exact:
                            assert_bitwise(np.array([U[i], V[i]]),
                                           np.array([Ur, Vr]),
                                           f"{bc} z={z} i={i}")
                            continue
                        mem = _edge_members(mesh, own[i])
                        ulp = np.spacing(max(np.abs(CU[mem]).max(),
                                             np.abs(CV[mem]).max()))
                        du = abs(U[i] - Ur) / ulp
                        dv = abs(V[i] - Vr) / ulp
                        assert du <= 8.0 and dv <= 8.0, (bc, z, i, du, dv)


def test_edge_linear_in_depth_exact():
    """``u = a_e + b_e z`` per element (declared prefixes of 2..5 points,
    every member wet): the layer core on the exact values
    ``a_e + b_e clip(z, p[0], p[last_e])``, to 1e-12 of the largest
    stencil value, in every element and for both boundary conditions; the
    layer answer (every element at the centre of the depth's layer) is
    not."""
    rng = np.random.default_rng(61)
    decl = rng.integers(1, M, size=32).astype(np.int32)
    mesh = _mesh(decl=decl)
    vel = _linear_field(mesh)
    dp = _table(mesh)
    pts, own = _all_elem_points(mesh, LAMS[:2])
    for bc in BCS:
        for td in (1.0, -1.0):
            for z in rng.uniform(5.0, 140.0, size=8):
                U, V, ok, nn = _run_edge(mesh, vel, _state(pts, z), td, bc)
                assert ok.all()
                zc = np.clip(z, P[0], P[dp.last])
                ue = CU + BU * zc
                ve = CV + BV * zc
                for i, (x, y) in enumerate(pts):
                    Ur, Vr, okr = _layer_edge(mesh, x, y, nn[i], td, bc,
                                              ue, ve)
                    assert okr
                    mem = _edge_members(mesh, own[i])
                    tol = 1e-12 * max(np.abs(ue[mem]).max(),
                                      np.abs(ve[mem]).max())
                    assert abs(U[i] - Ur) <= tol, (bc, z, i, U[i], Ur)
                    assert abs(V[i] - Vr) <= tol, (bc, z, i, V[i], Vr)
    mesh = _mesh()
    vel = _linear_field(mesh)
    pts = _pts_in(mesh, 19)
    ul, vl = _values_at(mesh, vel, 45.0)
    for bc in BCS:
        U, V, ok, nn = _run_edge(mesh, vel, _state(pts, 52.5), 1.0, bc)
        for i, (x, y) in enumerate(pts):
            Ul, _, _ = _layer_edge(mesh, x, y, nn[i], 1.0, bc, ul, vl)
            assert abs(U[i] - Ul) > 1e-4, (bc, i, U[i], Ul)


def test_edge_singular_with_wall_area_mean():
    """Element 19's centroid moved onto its edge (17, 16) (lat 5.5), the
    centroids of its neighbours 20 and 18 onto the same parallel;
    neighbour 26, across that edge, dry below 30 m.  Its ghost — the own
    centroid mirrored across the edge — is the centroid itself: every
    point of the fit lies on one parallel, the fit is singular, and the
    value is the area-weighted mean of the wet members in stencil order
    (19, 20, 18).  With 26 wet the same centroids give a regular fit: the
    layer core, bitwise."""
    bed = np.full(32, 150.0)
    bed[26] = 30.0
    mesh = _mesh(bed)
    lat = mesh.elem_center_lat.copy()
    lat[[18, 19, 20]] = 5.5
    mesh.elem_center_lat = lat
    vel = _linear_field(mesh)
    areas = build_elem_areas(mesh)
    pts = _pts_in(mesh, 19)
    for bc in BCS:
        for td in (1.0, -1.0):
            U, V, ok, nn = _run_edge(mesh, vel, _state(pts, 80.0), td, bc)
            assert ok.all()
            uz, vz = _values_at(mesh, vel, 80.0)
            for i, (x, y) in enumerate(pts):
                _, _, okr = _layer_edge(mesh, x, y, nn[i], td, bc, uz, vz,
                                        absent=(26,))
                assert not okr                 # the fit alone would freeze
            num_u = num_v = den = 0.0
            for e in (19, 20, 18):
                num_u += areas[e] * uz[e]
                num_v += areas[e] * vz[e]
                den += areas[e]
            assert np.all(U == td * (num_u / den)), (bc, U)
            assert np.all(V == td * (num_v / den)), (bc, V)
            U, V, ok, nn = _run_edge(mesh, vel, _state(pts, 20.0), td, bc)
            assert ok.all()
            uz, vz = _values_at(mesh, vel, 20.0)
            for i, (x, y) in enumerate(pts):
                Ur, Vr, okr = _layer_edge(mesh, x, y, nn[i], td, bc, uz, vz)
                assert okr
                assert_bitwise(np.array([U[i], V[i]]), np.array([Ur, Vr]),
                               f"all wet {bc} i={i}")


def test_walk_equals_local_in_trench():
    """A trench one cell row wide (elements 16..23, bed 150 m; every other
    element 30 m): below 30 m the off-trench neighbours of each trench
    element are walls.  The containing elements the edge walk tracks
    (walked from element 0 across the walls, or from the right element)
    give the bits of the local search, both boundary conditions, both
    directions."""
    bed = np.full(32, 30.0)
    bed[16:24] = 150.0
    mesh = _mesh(bed)
    vel = _linear_field(mesh)
    pts, local = [], []
    for e in range(16, 24):
        pts += _pts_in(mesh, e, LAMS[:3])
        local += [e] * 3
    local = np.array(local, dtype=np.int32)
    lons = np.array([x for x, _ in pts])
    lats = np.array([y for _, y in pts])
    walks = []
    for start in (np.zeros(local.size, dtype=np.int32), local.copy()):
        el, inside = find_containing_elements_walk(mesh, lons, lats, start)
        assert inside.all()
        assert_bitwise(np.asarray(el, dtype=np.int32), local, "walk")
        walks.append(el)
    for bc in BCS:
        for td in (1.0, -1.0):
            for z in (20.0, 52.5, 80.0, 140.0):
                st = _state(pts, z)
                ref = _run_edge(mesh, vel, st, td, bc)
                assert ref[2].all()
                for el in walks:
                    got = _run_edge(mesh, vel, st, td, bc, elems=el)
                    for a, b, name in zip(got[:3], ref[:3], ("U", "V", "ok")):
                        assert_bitwise(a, b, f"{name} {bc} td={td} z={z}")
                if z > 30.0:                   # the walls are in effect
                    uz, vz = _values_at(mesh, vel, z)
                    full = np.array([
                        _layer_edge(mesh, x, y, ref[3][i], td, bc, uz, vz)[0]
                        for i, (x, y) in enumerate(pts)])
                    assert np.any(np.abs(ref[0] - full) > 1e-6)


# ---------------------------------------------------------------------------
# barycentric, element layout
# ---------------------------------------------------------------------------

def _run_bary(mesh, vel, st, td=1.0, *, elems=None, layer=0):
    nn = find_nearest_nodes(mesh, st.lon, st.lat)
    U, V, ok = ip.interp_barycentric(st, mesh, vel, nn, td, layer,
                                     elems=elems)
    return U, V, ok, nn


def _layer_bary(mesh, x, y, nd, td, uvals, vvals, absent=(), hint=-1):
    """The LAYER core ``_interp_barycentric_core`` (element layout) at
    ``(x, y)`` on per-element values; the elements ``absent`` are removed
    from every vertex ring, the order of the others kept (a mesh where
    they are absent); ``hint`` is the containing element (``-1``: the
    core's own search)."""
    n2e = mesh.node_to_elems
    cnt = mesh.node_to_elems_count
    if absent:
        n2e = np.full_like(n2e, -1)
        cnt = cnt.copy()
        for k in range(mesh.n_nodes):
            ring = [e for e in _ring(mesh, k) if e not in absent]
            n2e[k, :len(ring)] = ring
            cnt[k] = len(ring)
    return ip._interp_barycentric_core(
        float(x), float(y), int(nd), int(hint), float(td),
        ip.UV_LAYOUT_ELEMENT, n2e, cnt, mesh.elem_nodes, mesh.nodes_lon,
        mesh.nodes_lat, build_elem_areas(mesh), uvals, vvals, mesh.r_earth)


def _vertex_mean(mesh, vel, nd, z):
    """Vertex ``nd``'s value at ``z`` by hand: the area-weighted mean of its
    wet ring elements in ring order (the kernel's summation); ``None``
    when none is wet."""
    bed = _table(mesh).bed
    areas = build_elem_areas(mesh)
    uz, vz = _values_at(mesh, vel, z)
    zq = np.inf if np.isnan(z) else z
    wet = [e for e in _ring(mesh, nd) if zq <= bed[e]]
    if not wet:
        return None
    num_u = num_v = den = 0.0
    for e in wet:
        num_u += areas[e] * uz[e]
        num_v += areas[e] * vz[e]
        den += areas[e]
    return num_u / den, num_v / den


def _bary_members(mesh, e):
    """Every element in the rings of element ``e``'s vertices."""
    return sorted({m for nd in mesh.elem_nodes[e] if nd >= 0
                   for m in _ring(mesh, int(nd))})


def test_bary_elem_dry_ring_member():
    """Element 13 (in the rings of vertices 12 and 13) has its bed at
    30 m and 1000 m/s in its column.  Below 30 m it is left out of each
    vertex mean: BITWISE the layer core on a mesh where it is absent from
    the rings (the same containing element), every element at its value at
    ``z``; at and above its bed it is in the means."""
    mesh = _wall_mesh()
    vel = _wall_field(mesh)
    pts = _pts_in(mesh, 19) + _pts_in(mesh, 20)
    st = _state(pts, 50.0)
    nn = find_nearest_nodes(mesh, st.lon, st.lat)
    hints = [_containing(mesh, x, y, nd) for (x, y), nd in zip(pts, nn)]
    assert hints == [(19, True)] * 6 + [(20, True)] * 6
    for td in (1.0, -1.0):
        for z in (30.5, 45.0, 52.5, 80.0, 125.0, 140.0, 150.0):
            U, V, ok, nn = _run_bary(mesh, vel, _state(pts, z), td)
            assert ok.all()
            uz, vz = _values_at(mesh, vel, z)
            for i, (x, y) in enumerate(pts):
                Ur, Vr, okr = _layer_bary(mesh, x, y, nn[i], td, uz, vz,
                                          absent=(13,), hint=hints[i][0])
                assert okr
                assert_bitwise(np.array([U[i], V[i]]), np.array([Ur, Vr]),
                               f"z={z} i={i}")
            assert np.all(np.abs(U) < 2.0)
        for z in (5.0, 20.0, 30.0):
            U, V, ok, nn = _run_bary(mesh, vel, _state(pts, z), td)
            uz, vz = _values_at(mesh, vel, z)
            for i, (x, y) in enumerate(pts):
                Ur, Vr, _ = _layer_bary(mesh, x, y, nn[i], td, uz, vz)
                assert_bitwise(np.array([U[i], V[i]]), np.array([Ur, Vr]),
                               f"wet z={z} i={i}")
            assert np.abs(U).max() > 10.0      # the 1000 m/s is in the means


def test_bary_elem_vertex_without_wet_element():
    """Every element around vertex 12 has its bed at 30 m (the others
    150 m; unequal model-provided areas).  Below 30 m vertex 12 has no wet
    element: it is left out and the barycentric weights of the other two
    vertices renormalised, each of them the area-weighted mean of its wet
    ring elements (1e-13 of the largest vertex value: the hand weights
    are computed in degrees); at 30 m (every element wet) the layer core,
    bitwise."""
    bed = np.full(32, 150.0)
    bed[RING] = 30.0
    areas = np.random.default_rng(67).uniform(1.0e9, 5.0e9, size=32)
    mesh = _mesh(bed, areas=areas)
    vel = _linear_field(mesh)
    for own in (19, 18, 11):
        tri = [int(nd) for nd in mesh.elem_nodes[own]]
        assert CENTRE in tri
        pts = _pts_in(mesh, own)
        for td in (1.0, -1.0):
            for z in (30.5, 45.0, 80.0, 140.0):
                U, V, ok, _ = _run_bary(mesh, vel, _state(pts, z), td)
                assert ok.all()
                vals = {nd: _vertex_mean(mesh, vel, nd, z) for nd in tri}
                assert vals[CENTRE] is None
                wet = [nd for nd in tri if nd != CENTRE]
                scale = max(abs(c) for nd in wet for c in vals[nd])
                for i, (x, y) in enumerate(pts):
                    lam = dict(zip(tri, _lam(mesh, tri, x, y)))
                    ws = sum(lam[nd] for nd in wet)
                    ue = td * sum(lam[nd] * vals[nd][0] for nd in wet) / ws
                    ve = td * sum(lam[nd] * vals[nd][1] for nd in wet) / ws
                    assert abs(U[i] - ue) <= 1e-13 * scale, (own, z, i)
                    assert abs(V[i] - ve) <= 1e-13 * scale, (own, z, i)
            U, V, ok, nn = _run_bary(mesh, vel, _state(pts, 30.0), td)
            uz, vz = _values_at(mesh, vel, 30.0)
            for i, (x, y) in enumerate(pts):
                Ur, Vr, _ = _layer_bary(mesh, x, y, nn[i], td, uz, vz)
                assert_bitwise(np.array([U[i], V[i]]), np.array([Ur, Vr]),
                               f"all wet {own} i={i}")


def test_bary_elem_at_dry_vertex_equal_mean():
    """A particle exactly ON vertex 12, whose ring is dry (beds 30 m;
    particle at 45..140 m): the weights of the other two vertices of the
    containing element sum to zero, so the value is the equal-weight mean
    of their values — each the area-weighted mean of its wet ring elements
    (unequal model-provided areas and velocities)."""
    bed = np.full(32, 150.0)
    bed[RING] = 30.0
    areas = np.random.default_rng(71).uniform(1.0e9, 5.0e9, size=32)
    mesh = _mesh(bed, areas=areas)
    vel = _linear_field(mesh)
    x, y = float(mesh.nodes_lon[CENTRE]), float(mesh.nodes_lat[CENTRE])
    e, inside = _containing(mesh, x, y, CENTRE)
    assert inside
    others = [int(nd) for nd in mesh.elem_nodes[e] if nd != CENTRE]
    assert len(others) == 2
    for td in (1.0, -1.0):
        for z in (45.0, 52.5, 80.0, 140.0):
            U, V, ok, nn = _run_bary(mesh, vel, _state([(x, y)], z), td)
            assert ok[0] and nn[0] == CENTRE
            (ua, va), (ub, vb) = (_vertex_mean(mesh, vel, nd, z)
                                  for nd in others)
            assert U[0] == td * ((ua + ub) / 2.0), (z, U[0])
            assert V[0] == td * ((va + vb) / 2.0), (z, V[0])
            # the area weights matter: the plain mean differs
            uz, _ = _values_at(mesh, vel, z)
            wet_a = [m for m in _ring(mesh, others[0]) if m not in RING]
            assert abs(ua - np.mean(uz[wet_a])) > 1e-6


def test_bary_elem_negative_weight_convex():
    """A particle just outside an edge of its element (the edge walk's
    pre-validated element 19 = vertices 12, 17, 16; containment accepts
    weights down to -1e-6), vertex 12 without a wet element (its ring's
    beds 30 m, particle at 45 m), vertex 16's weight -1e-7: the negative
    wet weight counts as zero, so the value is vertex 17's mean, a convex
    blend of the remaining vertex values — not the renormalised mixed-sign
    sum, which leaves their hull (by 2e-7 of the difference at the middle
    of the edge, by 3e4 of it next to vertex 12, where the two weights sum
    to 3e-12).  With every element wet (at 30 m) the layer core with the
    negative weight, bit for bit."""
    bed = np.full(32, 150.0)
    bed[RING] = 30.0
    areas = np.random.default_rng(79).uniform(1.0e9, 5.0e9, size=32)
    mesh = _mesh(bed, areas=areas)
    vel = _linear_field(mesh)
    tri = [int(nd) for nd in mesh.elem_nodes[19]]
    assert tri == [CENTRE, 17, 16]
    # (w12, w17, w16) = (., w17, -1e-7): lon = 5 + 5e-8,
    # lat = 5 + (w17 + w16) / 2
    pts = [(5.0 + 5.0e-8, 5.25), (5.0 + 5.0e-8, 5.0 + 1.5e-12)]
    for x, y in pts:
        lam = _lam(mesh, tri, x, y)
        assert abs(lam[2] + 1.0e-7) < 1e-12 and lam[1] > 0.0, lam
        assert lam[1] + lam[2] > 1e-12           # renormalised, not equal
    elems = np.full(len(pts), 19, dtype=np.int32)
    for td in (1.0, -1.0):
        U, V, ok, _ = _run_bary(mesh, vel, _state(pts, 45.0), td,
                                elems=elems)
        assert ok.all()
        assert _vertex_mean(mesh, vel, CENTRE, 45.0) is None
        (u17, v17), (u16, v16) = (_vertex_mean(mesh, vel, nd, 45.0)
                                  for nd in (17, 16))
        for got, a, b in ((U, u16, u17), (V, v16, v17)):
            assert abs(a - b) > 1e-3
            ulp = 2.0 * np.spacing(max(abs(a), abs(b)))
            assert np.all(np.abs(got - td * b) <= ulp), (td, got, b)
            lo, hi = sorted((td * a, td * b))
            assert np.all((got >= lo - ulp) & (got <= hi + ulp)), got
    # every element wet: the layer core with the negative weight
    U, V, ok, nn = _run_bary(mesh, vel, _state(pts, 30.0), elems=elems)
    assert ok.all()
    uz, vz = _values_at(mesh, vel, 30.0)
    for i, (x, y) in enumerate(pts):
        Ur, Vr, okr = _layer_bary(mesh, x, y, nn[i], 1.0, uz, vz, hint=19)
        assert okr
        assert_bitwise(np.array([U[i], V[i]]), np.array([Ur, Vr]),
                       f"all wet i={i}")


def test_bary_elem_all_rings_dry_reads_containing_element():
    """Every element around the vertices of element 19 is shallower than
    the particle; the deepest of them is 28 (60 m), element 19 itself
    35 m: no vertex is left, so the value is element 19 at its last usable
    point (unequal last-point values), also for a NaN depth."""
    members = _bary_members(_mesh(), 19)
    bed = np.full(32, 150.0)
    bed[members] = np.linspace(26.0, 58.0, len(members))
    bed[19] = 35.0
    bed[28] = 60.0
    mesh = _mesh(bed)
    vel = _linear_field(mesh)
    dp = _table(mesh)
    pts = _pts_in(mesh, 19)
    assert members[int(np.argmax(bed[members]))] == 28
    k19 = int(dp.last[19])
    assert k19 == 1 and dp.last[28] == 2
    for td in (1.0, -1.0):
        for z in (60.5, 80.0, 1.0e6, np.nan):
            U, V, ok, _ = _run_bary(mesh, vel, _state(pts, z), td)
            assert ok.all()
            assert np.all(U == td * vel.u[k19, 19]), (z, U)
            assert np.all(V == td * vel.v[k19, 19]), (z, V)
            assert np.all(U != td * vel.u[dp.last[28], 28])


def test_bary_elem_depth_constant_matches_layer():
    """A depth-constant field (declared prefixes of 2..5 points, every bed
    150 m: every member wet) gives the layer core's answer on ``c`` in
    every element: BITWISE wherever every member reads a stored value, and
    between data points within ``4 ulp`` of ``max|c|`` over the vertex
    rings — the vertex value is an area mean inside the barycentric sum, a
    two-level sum (node layout: one level, ``2 ulp``); in the worst case
    seen, one element's blend ``(1 - a) c + a c`` 1 ulp off moved its
    vertex mean by 2 ulp and the sum by 3.
    Measured worst over the 3,840 evaluations between points (of 8,064;
    u and v): 2 ulp; over 40 further seeds: 3 ulp, in 4 of 122,880
    values."""
    rng = np.random.default_rng(73)
    decl = rng.integers(1, M, size=32).astype(np.int32)
    mesh = _mesh(decl=decl)
    vel = _const_field(mesh)
    pts, own = _all_elem_points(mesh, LAMS[:3])
    between = sorted(set(rng.uniform(5.0, 125.0, size=10).tolist())
                     - set(P.tolist()))
    for td in (1.0, -1.0):
        for z_set, exact in ((DIRECT_Z, True), (between, False)):
            for z in z_set:
                U, V, ok, nn = _run_bary(mesh, vel, _state(pts, z), td)
                assert ok.all(), z
                for i, (x, y) in enumerate(pts):
                    Ur, Vr, okr = _layer_bary(mesh, x, y, nn[i], td, CU, CV)
                    assert okr
                    if exact:
                        assert_bitwise(np.array([U[i], V[i]]),
                                       np.array([Ur, Vr]), f"z={z} i={i}")
                        continue
                    mem = _bary_members(mesh, own[i])
                    du = abs(U[i] - Ur) / np.spacing(np.abs(CU[mem]).max())
                    dv = abs(V[i] - Vr) / np.spacing(np.abs(CV[mem]).max())
                    assert du <= 4.0 and dv <= 4.0, (z, i, du, dv)


# ---------------------------------------------------------------------------
# edge_leastsq and element barycentric: regimes, dispatch, fallback
# ---------------------------------------------------------------------------

#: the depth-linear entries of the two sections above
KINDS = ["edge_freeslip", "edge_noslip", "barycentric"]


def _run_kind(kind, mesh, vel, st, td=1.0, *, layer=0, elems=None):
    if kind == "barycentric":
        return _run_bary(mesh, vel, st, td, elems=elems, layer=layer)
    return _run_edge(mesh, vel, st, td, kind.split("_")[1], elems=elems,
                     layer=layer)


def _kind_regimes(kind, mesh, st, nn):
    """Counts of alive particles inside the mesh by regime.  Edge: every
    stencil element wet; the own element wet and a wall; the own element
    dry and a neighbour wet; none wet.  Barycentric: every ring element
    wet; a dry ring element but every vertex wet; a vertex left out; no
    vertex left."""
    bed = _table(mesh).bed
    nb = build_elem_neighbors(mesh)
    out = {}
    for i in np.flatnonzero(st.alive_mask):
        e, inside = _containing(mesh, st.lon[i], st.lat[i], nn[i])
        if not inside:
            continue
        zq = np.inf if np.isnan(st.z[i]) else st.z[i]
        if kind == "barycentric":
            rings = [[zq <= bed[m] for m in _ring(mesh, int(nd))]
                     for nd in mesh.elem_nodes[e] if nd >= 0]
            key = ("all_wet" if all(all(r) for r in rings)
                   else "dry_member" if all(any(r) for r in rings)
                   else "vertex_out" if any(any(r) for r in rings)
                   else "all_dry")
        else:
            wet = [zq <= bed[m] for m in nb[e] if m >= 0]
            if zq <= bed[e]:
                key = "all_wet" if all(wet) else "wall"
            else:
                key = "own_dry" if any(wet) else "all_dry"
        out[key] = out.get(key, 0) + 1
    return out


@pytest.mark.parametrize("kind", KINDS)
def test_new_cores_nan_below_last_never_read(kind):
    """NaN below each element's last usable point never reaches a result:
    the same field with +inf / 1e300 below ``last`` gives the same bits
    over particles in every regime (walls, both fallbacks, NaN depth,
    outside the mesh)."""
    mesh = _mixed_mesh()
    vel_nan = _linear_field(mesh)
    vel_inf = _linear_field(mesh, below=np.inf)
    vel_big = _linear_field(mesh, below=1.0e300)
    st = _mixed_state()
    nn = find_nearest_nodes(mesh, st.lon, st.lat)
    regimes = _kind_regimes(kind, mesh, st, nn)
    assert len(regimes) == 4 and min(regimes.values()) >= 10, regimes
    for td in (1.0, -1.0):
        ref = _run_kind(kind, mesh, vel_nan, st, td)[:3]
        assert ref[2].sum() > 200
        assert np.all(np.isfinite(ref[0])) and np.all(np.isfinite(ref[1]))
        for other in (vel_inf, vel_big):
            got = _run_kind(kind, mesh, other, st, td)[:3]
            for a, b, name in zip(got, ref, ("U", "V", "ok")):
                assert_bitwise(a, b, f"{name} td={td}")


@pytest.mark.parametrize("kind", KINDS)
def test_new_cores_dispatch(kind):
    """The layer argument is ignored (scalar or per particle), each
    particle is read at ``state.z``, dead particles get zeros and
    ``ok = False``, a lazy blend of two fields is read like the eager one,
    and pre-validated containing elements (``elems=``; ``-1`` = search)
    give the bits of the search."""
    mesh = _mixed_mesh()
    vel = _linear_field(mesh)
    st = _mixed_state()
    ref = _run_kind(kind, mesh, vel, st)
    assert not ref[2][3] and ref[0][3] == 0.0 and ref[1][17] == 0.0
    for layer in (3, np.full(st.n_particles, 2, dtype=np.int32)):
        got = _run_kind(kind, mesh, vel, st, layer=layer)
        for a, b, name in zip(got[:3], ref[:3], ("U", "V", "ok")):
            assert_bitwise(a, b, f"{name} layer={layer!r}")
    lazy = LazyLerpField(vel, _linear_field(mesh), 0.5)
    got = _run_kind(kind, mesh, lazy, st)
    for a, b, name in zip(got[:3], ref[:3], ("U", "V", "ok")):
        assert_bitwise(a, b, f"{name} lazy")
    nn = ref[3]
    elems = np.array([_containing(mesh, st.lon[i], st.lat[i], nn[i])[0]
                      for i in range(st.n_particles)], dtype=np.int32)
    assert (elems >= 0).sum() > 200
    elems[::4] = -1
    got = _run_kind(kind, mesh, vel, st, elems=elems)
    for a, b, name in zip(got[:3], ref[:3], ("U", "V", "ok")):
        assert_bitwise(a, b, f"{name} elems")
    deep = _mesh()
    U2 = _run_kind(kind, deep, _linear_field(deep),
                   _state([CENTRE_PTS[0]] * 2, [45.0, 52.5]))[0]
    assert abs(U2[1] - U2[0]) > 1e-4


@pytest.mark.parametrize("kind", KINDS)
def test_new_cores_record_checked(kind):
    """A non-finite value at a usable point is refused, naming the
    element; a passing field is stamped as checked for the table."""
    mesh = _mesh()
    vel = _linear_field(mesh)
    vel.u[2, 13] = np.nan
    st = _state(CENTRE_PTS, 50.0)
    with pytest.raises(ValueError, match=r"u = nan is non-finite at element "
                                         r"13, point 2"):
        _run_kind(kind, mesh, vel, st)
    good = _linear_field(mesh)
    assert not vu.is_checked(good, mesh)
    _run_kind(kind, mesh, good, st)
    assert vu.is_checked(good, mesh)


@pytest.mark.parametrize("kind", KINDS)
def test_new_cores_python_fallback_equals_numba(kind, monkeypatch):
    """The pure-Python core (``.py_func``) and the wrapper's no-Numba loop
    give the compiled bits in every regime."""
    mesh = _mixed_mesh()
    vel = _linear_field(mesh)
    st = _mixed_state()
    nn = find_nearest_nodes(mesh, st.lon, st.lat)
    dp = _table(mesh)
    areas = build_elem_areas(mesh)
    nb = build_elem_neighbors(mesh)
    for td in (1.0, -1.0):
        ref = _run_kind(kind, mesh, vel, st, td)[:3]
        with monkeypatch.context() as mp:
            mp.setattr(ip, "HAS_NUMBA", False)
            got = _run_kind(kind, mesh, vel, st, td)[:3]
        for a, b, name in zip(got, ref, ("U", "V", "ok")):
            assert_bitwise(a, b, f"{name} wrapper td={td}")
        for i in range(st.n_particles):
            head = (float(st.lon[i]), float(st.lat[i]), float(st.z[i]),
                    int(nn[i]), -1, td)
            if kind == "barycentric":
                core = ip._interp_bary_elem_core_dl
                args = head + (
                    mesh.node_to_elems, mesh.node_to_elems_count,
                    mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat,
                    dp.p, dp.last, dp.bed, areas, vel.u, vel.v,
                    mesh.r_earth)
            else:
                core = ip._interp_edge_leastsq_core_dl
                args = head + (
                    ip._bc_to_int(kind.split("_")[1]),
                    mesh.node_to_elems, mesh.node_to_elems_count,
                    mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat, nb,
                    mesh.elem_center_lon, mesh.elem_center_lat,
                    dp.p, dp.last, dp.bed, areas, vel.u, vel.v)
            c = core(*args)
            p = core.py_func(*args)
            assert_bitwise(np.array(p[:2]), np.array(c[:2]), f"core i={i}")
            assert p[2] == c[2]
            if st.alive_mask[i]:
                assert_bitwise(np.array(c[:2]),
                               np.array([ref[0][i], ref[1][i]]),
                               f"core vs wrapper i={i}")
