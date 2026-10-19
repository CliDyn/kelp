"""Depth-linear least-squares u/v on element layout.

``interp_leastsq`` dispatches an element-layout ``"layer_linear"`` field to
the depth-linear twins (``_interp_leastsq_dl`` ->
``_loop_interp_leastsq_3d_dl`` -> ``_interp_leastsq_core_dl``): the plane
fit of ``_interp_leastsq_core`` over the elements around the nearest node,
each element read linearly in depth between its own layer centres (clipped
to ``p[0] .. p[last_e]``).  An element whose bed lies above the particle is
a wall and is left out of the fit, as elements beyond the coast are absent
from a coastal node's ring.  When the wet elements are too few for the
fit (fewer than three, or a singular fit) the value is their area-weighted
mean; when no element is wet, the containing element at its last usable
point.  With every element wet the arithmetic is the layer core's, so a
singular all-wet ring stays frozen.  A node-layout depth-linear field is
refused.

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
    _find_containing_element_local,
    build_elem_areas,
    find_nearest_nodes,
)
from kelp.core.particles import ParticleStatus, make_state
from kelp.numerics import interpolate as ip
from kelp.numerics import vertical_uv as vu

from tests._bitwise import assert_bitwise
from tests.test_3d_analytical import _make_mesh

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
    ``uv_bracket`` + ``uv_at``); NaN for an element whose column is empty
    there is impossible (the read never passes ``last``)."""
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
    worst = 0.0
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
                    worst = max(worst, du, dv)
                    assert du <= 8.0 and dv <= 8.0, (z, i, du, dv)
    assert worst <= 8.0


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
