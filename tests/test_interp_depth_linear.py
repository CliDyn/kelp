"""Depth-linear barycentric u/v.

``interp_barycentric`` dispatches a ``"level"`` / ``"layer_linear"`` field to
the depth-linear twins (``_interp_barycentric_dl`` ->
``_loop_interp_bary_dl`` -> ``_interp_barycentric_core_dl``): the same
containment, weights and 2-ring fallback as the node-layout
``_interp_barycentric_core``, with each vertex value read linearly in depth
between the bracketing data points of that vertex (clipped to
``p[0] .. p[last_i]``), and a vertex whose bed lies above the particle left
out as a wall (the other weights renormalised; their equal-weight mean when
those weights vanish; below every bed of the triangle the deepest vertex at
its last point).  Every other u/v interpolator refuses a depth-linear field
at entry.

All synthetic: the 3x3 test mesh of ``test_3d_analytical`` with levels
``[0, 10, 20, 30, 40]`` m, three shallow nodes, and ``u = a + b*z`` stored
at the data points (NaN below each node's last usable point, ``+inf`` in
one column), so every expected value is written down by hand.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from kelp.core.io import VelocityField
from kelp.core.mesh import _find_containing_element_local, find_nearest_nodes
from kelp.core.particles import ParticleStatus, make_state
from kelp.numerics import interpolate as ip
from kelp.numerics import vertical_uv as vu
from kelp.numerics.vertical import find_layers

from tests._bitwise import assert_bitwise
from tests.test_3d_analytical import _make_mesh

LEVELS = [0.0, 10.0, 20.0, 30.0, 40.0]
PLACEMENTS = ["level", "layer_linear"]

#: data-point depths per placement (levels; layer centres)
P_PTS = {
    "level": np.array([0.0, 10.0, 20.0, 30.0, 40.0]),
    "layer_linear": np.array([5.0, 15.0, 25.0, 35.0]),
}

#: bed depths: node 2 (6E, 4N) 10 m, node 6 (4E, 6N) 35 m, node 8 (6E, 6N)
#: 25 m, the rest 200 m (below the deepest level)
DEPTHS = np.array([200.0, 200.0, 10.0, 200.0, 200.0, 200.0, 35.0, 200.0,
                   25.0])

#: last usable point per node, by hand: "level" = the deepest level at or
#: above the bed; "layer_linear" = the deepest layer whose LOWER level is at
#: or above the bed (find_layer's rule: 10 m bed -> layer 0, 25 m -> 1,
#: 35 m -> 2)
LAST = {
    "level": np.array([4, 4, 1, 4, 4, 4, 3, 4, 2], dtype=np.int32),
    "layer_linear": np.array([3, 3, 0, 3, 3, 3, 2, 3, 1], dtype=np.int32),
}

A_U, B_U = 0.25, 0.02
A_V, B_V = -0.1, -0.004

#: the column that carries +inf (just below its last usable point)
INF_NODE = 2

#: inside the cell (4..5E, 4..5N), whose four vertices are 200 m deep
DEEP_PTS = [(4.3, 4.2), (4.7, 4.6), (4.25, 4.6)]

#: inside triangle [1, 2, 4] (5E 4N, 6E 4N, 5E 5N): node 2 is 10 m deep
SHALLOW_PT = (5.6, 4.3)


def _mesh(depths=DEPTHS):
    mesh = _make_mesh(5.0, 5.0, z_levels=LEVELS, bottom_depth=200.0)
    mesh.depth = np.asarray(depths, dtype=np.float64)
    return mesh


def _field(placement):
    """``u = A_U + B_U * p_k`` (v likewise) at points ``k <= last_i``; NaN
    below, ``+inf`` at ``(last + 1, INF_NODE)``."""
    p = P_PTS[placement]
    last = LAST[placement]
    n = last.size
    m = p.size
    u = np.full((m, n), np.nan)
    v = np.full((m, n), np.nan)
    for i in range(n):
        u[: last[i] + 1, i] = A_U + B_U * p[: last[i] + 1]
        v[: last[i] + 1, i] = A_V + B_V * p[: last[i] + 1]
    u[last[INF_NODE] + 1, INF_NODE] = np.inf
    v[last[INF_NODE] + 1, INF_NODE] = np.inf
    return VelocityField(u=u, v=v, eta=np.zeros(n),
                         w=np.zeros((len(LEVELS), n)),
                         uv_layout="node", uv_vertical=placement)


def _state(pts, z, *, dead=()):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    z = np.broadcast_to(np.asarray(z, dtype=np.float64), (pts.shape[0],))
    st = make_state(pts[:, 0].copy(), pts[:, 1].copy(), z.copy())
    for i in dead:
        st.status[i] = ParticleStatus.OUTSIDE
    return st


def _run(mesh, vel, st, time_dir=1.0, **kw):
    nn = find_nearest_nodes(mesh, st.lon, st.lat)
    return ip.interp_barycentric(st, mesh, vel, nn, time_dir, **kw)


def _lam(mesh, tri, x, y):
    """Barycentric coordinates of (x, y) in triangle ``tri`` (lon/lat
    degrees: an affine image of the kernel's local metric frame, so the
    coordinates are the same)."""
    xs = mesh.nodes_lon[list(tri)]
    ys = mesh.nodes_lat[list(tri)]
    mat = np.array([[xs[1] - xs[0], xs[2] - xs[0]],
                    [ys[1] - ys[0], ys[2] - ys[0]]])
    l1, l2 = np.linalg.solve(mat, np.array([x - xs[0], y - ys[0]]))
    return np.array([1.0 - l1 - l2, l1, l2])


def _rel(a, b):
    return np.abs(np.asarray(a) - np.asarray(b)) / np.abs(np.asarray(b))


def _hand_wall(mesh, placement, tri, lam, z, value):
    """The wall rule by hand at depth ``z`` in triangle ``tri`` with
    barycentric coordinates ``lam``; ``value(nd, zc)`` is node ``nd``'s
    stored profile at the clipped depth ``zc`` (linear in depth between
    the data points, so the clip is the whole vertical rule).

    A vertex is wet when ``z <= mesh.depth[nd]`` (a NaN ``z`` reads as
    ``+inf``); a wet vertex reads ``clip(z, p[0], p[last])``.  Every
    vertex wet: the barycentric sum.  Some dry: the dry ones are left out,
    the weights of the wet ones renormalised (their equal-weight mean when
    those weights sum below 1e-12).  None wet: the vertex with the largest
    bed (the first of ``tri`` on ties) at its last point."""
    dp = vu.depth_points(mesh, placement)
    p, last = dp.p, dp.last
    tri = [int(nd) for nd in tri]
    zq = np.inf if np.isnan(z) else z

    def at(nd, zz):
        return value(nd, min(max(zz, p[0]), p[last[nd]]))
    wet = [zq <= mesh.depth[nd] for nd in tri]
    if all(wet):
        return sum(l_ * at(nd, zq) for l_, nd in zip(lam, tri))
    if not any(wet):
        return at(tri[int(np.argmax(mesh.depth[tri]))], np.inf)
    ws = sum(l_ for l_, w_ in zip(lam, wet) if w_)
    if ws < 1e-12:
        return float(np.mean([at(nd, zq) for nd, w_ in zip(tri, wet) if w_]))
    return sum(l_ * at(nd, zq)
               for l_, nd, w_ in zip(lam, tri, wet) if w_) / ws


#: per-node offsets of the horizontally varying fixtures (m/s)
OFF_U = 0.03 * np.arange(1, 10)
OFF_V = -0.02 * np.arange(1, 10)


def _field_varying(mesh, placement, *, column=None):
    """``u = A_U + B_U * p_k + OFF_U[nd]`` (v likewise) at the points
    ``k <= last`` of the mesh's table, NaN below: the vertices differ, so
    the weights matter.  ``column = (nd, c)`` stores the constant ``c`` in
    node ``nd``'s usable points instead (finite: the record check passes;
    a large ``c`` shows any weight the node gets)."""
    dp = vu.depth_points(mesh, placement)
    n = mesh.n_nodes
    u = np.full((dp.p.size, n), np.nan)
    v = np.full((dp.p.size, n), np.nan)
    for i in range(n):
        k = dp.last[i] + 1
        u[:k, i] = A_U + B_U * dp.p[:k] + OFF_U[i]
        v[:k, i] = A_V + B_V * dp.p[:k] + OFF_V[i]
    if column is not None:
        nd, c = column
        u[:dp.last[nd] + 1, nd] = c
        v[:dp.last[nd] + 1, nd] = -c
    return VelocityField(u=u, v=v, eta=np.zeros(n),
                         w=np.zeros((mesh.sigma.size, n)),
                         uv_layout="node", uv_vertical=placement)


def test_mesh_fixture_matches_table():
    """The hand-written ``LAST`` is the engine's table."""
    from kelp.numerics.vertical_uv import depth_points
    mesh = _mesh()
    for placement in ("level", "layer_linear"):
        dp = depth_points(mesh, placement)
        assert_bitwise(dp.p, P_PTS[placement], f"p {placement}")
        assert_bitwise(dp.last, LAST[placement], f"last {placement}")


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------

def test_level_offcentre_exact():
    mesh = _mesh()
    vel = _field("level")
    zs = [2.5, 7.5, 12.5]
    for z in zs:
        st = _state(DEEP_PTS, z)
        U, V, ok = _run(mesh, vel, st)
        assert ok.all()
        assert np.all(_rel(U, A_U + B_U * z) < 1e-14), (z, U)
        assert np.all(_rel(V, A_V + B_V * z) < 1e-14), (z, V)
        k = int(z // 10.0)
        layer_const = A_U + B_U * P_PTS["level"][k]
        half_sum = A_U + B_U * 0.5 * (P_PTS["level"][k]
                                      + P_PTS["level"][k + 1])
        assert np.all(np.abs(U - layer_const) > 1e-3), (z, U, layer_const)
        assert np.all(np.abs(U - half_sum) > 1e-3), (z, U, half_sum)
    # backward: the exact negation
    st = _state(DEEP_PTS, 7.5)
    Uf, Vf, _ = _run(mesh, vel, st, 1.0)
    Ub, Vb, okb = _run(mesh, vel, st, -1.0)
    assert okb.all()
    assert_bitwise(Ub, -Uf, "U backward")
    assert_bitwise(Vb, -Vf, "V backward")


def test_layer_linear_offcentre_exact():
    mesh = _mesh()
    vel = _field("layer_linear")
    p = P_PTS["layer_linear"]
    for z in [6.0, 12.5, 14.0, 21.0, 33.0]:
        st = _state(DEEP_PTS, z)
        U, V, ok = _run(mesh, vel, st)
        assert ok.all()
        assert np.all(_rel(U, A_U + B_U * z) < 1e-14), (z, U)
        assert np.all(_rel(V, A_V + B_V * z) < 1e-14), (z, V)
        # the layer-constant answer (layer centre of z's layer) differs
        layer_const = A_U + B_U * p[int(z // 10.0)]
        assert np.all(np.abs(U - layer_const) > 1e-3), (z, U, layer_const)


@pytest.mark.parametrize("placement", ["level", "layer_linear"])
def test_shallow_vertex_is_a_wall(placement):
    """Triangle ``[1, 2, 4]``, node 2's bed at 10 m.  Above it (7 m and on
    the bed) every vertex is wet and clips the depth to its own last
    point: ``U = a + b * sum_i lambda_i * clip(z, p[0], p[last_i])``.
    Below it node 2 is dry and left out: the wet vertices' renormalised
    blend, ``a + b * clip(z)`` (the field is horizontally uniform)."""
    mesh = _mesh()
    vel = _field(placement)
    p = P_PTS[placement]
    tri = (1, 2, 4)
    x, y = SHALLOW_PT
    lam = _lam(mesh, tri, x, y)
    assert np.all(lam > 0.05)
    for z in [7.0, 10.0, 12.5, 25.0, 39.0]:
        st = _state([SHALLOW_PT], z)
        U, V, ok = _run(mesh, vel, st)
        assert ok.all()
        if z <= DEPTHS[2]:
            zi = np.array([min(max(z, p[0]), p[LAST[placement][nd]])
                           for nd in tri])
            want_u = A_U + B_U * float(lam @ zi)
            want_v = A_V + B_V * float(lam @ zi)
        else:
            zc = min(max(z, p[0]), p[LAST[placement][1]])
            want_u, want_v = A_U + B_U * zc, A_V + B_V * zc
        assert _rel(U[0], want_u) < 1e-13, (z, U)
        assert _rel(V[0], want_v) < 1e-13, (z, V)
        # the shared hand rule (used by the integrator tests) agrees
        hand_u = _hand_wall(mesh, placement, tri, lam, z,
                            lambda nd, zc: A_U + B_U * zc)
        assert _rel(hand_u, want_u) < 1e-13, (z, hand_u)


@pytest.mark.parametrize("placement", ["level", "layer_linear"])
def test_surface_and_bottom_constant(placement):
    mesh = _mesh()
    vel = _field(placement)
    p = P_PTS[placement]
    pts = DEEP_PTS + [SHALLOW_PT]
    # above the first point: point 0 everywhere, the same bits as z = p[0]
    U, V, ok = _run(mesh, vel, _state(pts, -1.0))
    U0, V0, _ = _run(mesh, vel, _state(pts, p[0]))
    assert ok.all()
    assert_bitwise(U, U0, "U surface")
    assert_bitwise(V, V0, "V surface")
    assert np.all(_rel(U, A_U + B_U * p[0]) < 1e-14)
    # below every bed of the containing triangle: no vertex is wet, the
    # vertex with the largest bed (the first on ties) reads its last point
    U, V, ok = _run(mesh, vel, _state(pts, 500.0))
    U2, V2, _ = _run(mesh, vel, _state(pts, 1e6))
    assert ok.all()
    assert_bitwise(U, U2, "U bottom")
    assert_bitwise(V, V2, "V bottom")
    # the deep cell (every vertex 200 m): that fallback gives the old
    # blend of the vertices' last points only because the field is
    # horizontally uniform there
    assert np.all(_rel(U[:3], A_U + B_U * p[LAST[placement][0]]) < 1e-14)
    # triangle [1, 2, 4]: nodes 1 and 4 (200 m) tie, node 1 comes first
    k1 = LAST[placement][1]
    assert_bitwise(U[3:], np.array([vel.u[k1, 1]]), "U shallow bottom")
    assert_bitwise(V[3:], np.array([vel.v[k1, 1]]), "V shallow bottom")


@pytest.mark.parametrize("placement", ["level", "layer_linear"])
def test_nan_depth_reads_like_below_the_bed(placement):
    """A NaN depth reads like ``+inf``, as in the layer path (``find_layer``
    puts NaN in the deepest valid layer, as ``+inf``): finite with
    ok=True.  For the depth-linear read every finite bed is then dry, so
    the containing triangle's deepest vertex (the first on ties) gives its
    value at its last point."""
    mesh = _mesh()
    pts = DEEP_PTS + [SHALLOW_PT]
    # the layer path's rule, pinned here as the reference
    nn = find_nearest_nodes(mesh, np.array([p[0] for p in pts]),
                            np.array([p[1] for p in pts]))
    alive = np.ones(len(pts), dtype=bool)
    zl = np.array(LEVELS)
    assert_bitwise(find_layers(zl, np.full(len(pts), np.nan), nn,
                               mesh.depth, alive),
                   find_layers(zl, np.full(len(pts), np.inf), nn,
                               mesh.depth, alive), "find_layers NaN vs inf")
    vel = _field(placement)
    U, V, ok = _run(mesh, vel, _state(pts, np.nan))
    Ub, Vb, okb = _run(mesh, vel, _state(pts, np.inf))
    assert ok.all() and okb.all()
    assert np.all(np.isfinite(U)) and np.all(np.isfinite(V))
    assert_bitwise(U, Ub, "U NaN vs below bed")
    assert_bitwise(V, Vb, "V NaN vs below bed")
    p = P_PTS[placement]
    assert np.all(_rel(U[:3], A_U + B_U * p[LAST[placement][0]]) < 1e-14)
    # triangle [1, 2, 4]: nodes 1 and 4 (200 m) tie, node 1 comes first
    k1 = LAST[placement][1]
    assert_bitwise(U[3:], np.array([vel.u[k1, 1]]), "U shallow NaN")
    assert_bitwise(V[3:], np.array([vel.v[k1, 1]]), "V shallow NaN")


# ---------------------------------------------------------------------------
# Walls at depth: a vertex dry at the particle depth is left out
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("z", [45.0, 32.0])
def test_dry_vertex_left_out(z):
    """Node 2 of triangle ``[1, 2, 4]`` has its bed at 30 m, the other
    two at 200 m; at ``z`` = 45 m (below every data point) and 32 m
    (between the 25 and 35 m centres) node 2 is dry and is left out:
    ``U = (w1 u1(z) + w4 u4(z)) / (w1 + w4)``.  Node 2 stores 1000 m/s at
    its usable points, so any weight it got would show.  At 30 m (on its
    bed) it is wet and enters the sum with its last point (25 m)."""
    depths = np.full(9, 200.0)
    depths[2] = 30.0
    mesh = _mesh(depths)
    placement = "layer_linear"
    dp = vu.depth_points(mesh, placement)
    assert dp.last[2] == 2 and dp.last[1] == dp.last[4] == 3   # 25 / 35 m
    vel = _field_varying(mesh, placement, column=(2, 1000.0))
    tri = (1, 2, 4)
    lam = _lam(mesh, tri, *SHALLOW_PT)
    assert np.all(lam > 0.05)
    for td in (1.0, -1.0):
        U, V, ok = _run(mesh, vel, _state([SHALLOW_PT], z), td)
        assert ok.all()
        zc = min(z, 35.0)                    # the wet vertices' last point
        u1, u4 = (A_U + B_U * zc + OFF_U[nd] for nd in (1, 4))
        v1, v4 = (A_V + B_V * zc + OFF_V[nd] for nd in (1, 4))
        want_u = (lam[0] * u1 + lam[2] * u4) / (lam[0] + lam[2])
        want_v = (lam[0] * v1 + lam[2] * v4) / (lam[0] + lam[2])
        assert _rel(U[0], td * want_u) < 1e-12, (z, U)
        assert _rel(V[0], td * want_v) < 1e-12, (z, V)
    # on node 2's bed: wet, the plain sum with node 2's 1000 m/s
    U, V, ok = _run(mesh, vel, _state([SHALLOW_PT], 30.0))
    assert ok.all()
    want = (lam[0] * (A_U + B_U * 30.0 + OFF_U[1]) + lam[1] * 1000.0
            + lam[2] * (A_U + B_U * 30.0 + OFF_U[4]))
    assert _rel(U[0], want) < 1e-12, U


@pytest.mark.parametrize("placement", PLACEMENTS)
def test_partial_bottom_blend(placement):
    """Every bed at 25 m on levels ``[0, 10, 20, 30]``: the layer
    ``20-30 m`` reaches below the bed and is not read, so the last usable
    point is the 15 m centre (``"level"``: the 20 m level).  A particle at
    22 m is above every bed — every vertex is WET (wet is decided by the
    bed, not by the last point) and reads its last usable point: the
    barycentric blend of those values, not one vertex's value."""
    mesh = _make_mesh(5.0, 5.0, z_levels=[0.0, 10.0, 20.0, 30.0],
                      bottom_depth=25.0)
    dp = vu.depth_points(mesh, placement)
    pl = {"layer_linear": 15.0, "level": 20.0}[placement]
    assert np.all(dp.p[dp.last] == pl)
    vel = _field_varying(mesh, placement)
    tri = (1, 2, 4)
    lam = _lam(mesh, tri, *SHALLOW_PT)
    for z in (22.0, 25.0):                    # inside, on the bed
        U, V, ok = _run(mesh, vel, _state([SHALLOW_PT], z))
        assert ok.all()
        want_u = float(lam @ (A_U + B_U * pl + OFF_U[list(tri)]))
        want_v = float(lam @ (A_V + B_V * pl + OFF_V[list(tri)]))
        assert _rel(U[0], want_u) < 1e-13, (z, U)
        assert _rel(V[0], want_v) < 1e-13, (z, V)
        for nd in tri:
            assert abs(U[0] - (A_U + B_U * pl + OFF_U[nd])) > 1e-3


@pytest.mark.parametrize("placement", PLACEMENTS)
def test_all_vertices_dry_reads_deepest(placement):
    """Below every bed of the containing triangle ``[1, 2, 4]`` (and for a
    NaN depth, which reads as ``+inf``): the vertex with the largest bed
    at its last usable point, the stored value itself; on a tie the first
    of the element's vertices."""
    p = P_PTS[placement]
    cases = [((30.0, 10.0, 35.0), 4),         # node 4 deepest
             ((35.0, 10.0, 35.0), 1),         # tie 1 / 4: node 1 first
             ((30.0, 35.0, 35.0), 2)]         # tie 2 / 4: node 2 first
    for beds, deepest in cases:
        depths = np.full(9, 200.0)
        depths[[1, 2, 4]] = beds
        mesh = _mesh(depths)
        dp = vu.depth_points(mesh, placement)
        vel = _field_varying(mesh, placement)
        k = int(dp.last[deepest])
        assert p[k] < 40.0                   # a shallow last point
        st = _state([SHALLOW_PT] * 3, [50.0, 1e6, np.nan])
        for td in (1.0, -1.0):
            U, V, ok = _run(mesh, vel, st, td)
            assert ok.all()
            assert_bitwise(U, np.full(3, td * vel.u[k, deepest]),
                           f"U {beds}")
            assert_bitwise(V, np.full(3, td * vel.v[k, deepest]),
                           f"V {beds}")


@pytest.mark.parametrize("placement", PLACEMENTS)
def test_dry_vertex_zero_weight_equal_mean(placement):
    """A particle ON the dry vertex (node 4, the mesh centre, bed 10 m;
    particle at 20 m): the wet vertices' weights sum to zero, so the
    value is the equal-weight mean of the two wet vertices of the
    containing triangle."""
    depths = np.full(9, 200.0)
    depths[4] = 10.0
    mesh = _mesh(depths)
    vel = _field_varying(mesh, placement)
    x, y = float(mesh.nodes_lon[4]), float(mesh.nodes_lat[4])
    st = _state([(x, y)], 20.0)
    nn = find_nearest_nodes(mesh, st.lon, st.lat)
    assert nn[0] == 4
    e, inside = _find_containing_element_local(
        x, y, 4, mesh.node_to_elems, mesh.node_to_elems_count,
        mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat)
    assert inside
    others = [int(nd) for nd in mesh.elem_nodes[e] if nd != 4]
    assert len(others) == 2 and (OFF_U[others[0]] != OFF_U[others[1]])
    U, V, ok = _run(mesh, vel, st)
    assert ok.all()
    want_u = np.mean([A_U + B_U * 20.0 + OFF_U[nd] for nd in others])
    want_v = np.mean([A_V + B_V * 20.0 + OFF_V[nd] for nd in others])
    assert _rel(U[0], want_u) < 1e-13, U
    assert _rel(V[0], want_v) < 1e-13, V


@pytest.mark.parametrize("placement", PLACEMENTS)
def test_no_wall_unchanged(placement):
    """Every vertex of the containing triangle wet (``z <= bed``, on the
    bed included): the plain barycentric sum of the depth-linear vertex
    values, bit for bit — checked against the layer kernel
    ``_interp_barycentric_core`` (the same containment, weights and
    summation order) fed with those vertex values as a 2-D row."""
    mesh = _mesh()
    vel = _field(placement)
    dp = vu.depth_points(mesh, placement)
    rng = np.random.default_rng(41)
    pts = rng.uniform(4.02, 5.98, (90, 2))
    nn = find_nearest_nodes(mesh, pts[:, 0], pts[:, 1])
    zs = np.empty(len(pts))
    for i, (x, y) in enumerate(pts):
        e, inside = _find_containing_element_local(
            x, y, int(nn[i]), mesh.node_to_elems, mesh.node_to_elems_count,
            mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat)
        assert inside
        top = float(mesh.depth[mesh.elem_nodes[e]].min())
        zs[i] = top if i % 5 == 0 else rng.uniform(-2.0, top)
    st = _state(pts, zs)
    n = mesh.n_nodes
    for td in (1.0, -1.0):
        U, V, ok = _run(mesh, vel, st, td)
        assert ok.all()
        Ur, Vr = np.empty(len(pts)), np.empty(len(pts))
        for i, (x, y) in enumerate(pts):
            row_u, row_v = np.empty(n), np.empty(n)
            for nd in range(n):
                k, a = vu.uv_bracket(dp.p, dp.last[nd], zs[i])
                row_u[nd] = vu.uv_at(vel.u, nd, k, a)
                row_v[nd] = vu.uv_at(vel.v, nd, k, a)
            Ur[i], Vr[i], okr = ip._interp_barycentric_core(
                x, y, int(nn[i]), -1, td, ip.UV_LAYOUT_NODE,
                mesh.node_to_elems, mesh.node_to_elems_count,
                mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat,
                np.empty(0, dtype=np.float64), row_u, row_v, mesh.r_earth)
            assert okr
        assert_bitwise(U, Ur, f"U td={td}")
        assert_bitwise(V, Vr, f"V td={td}")


# ---------------------------------------------------------------------------
# Dispatch, hints, fallback
# ---------------------------------------------------------------------------

def _mixed_state():
    pts = DEEP_PTS + [SHALLOW_PT, (5.3, 5.8), (5.8, 5.7), (4.4, 5.5),
                      (5.5, 4.5), (20.0, 20.0)]
    z = [2.5, 13.0, 37.0, 25.0, 8.0, 22.0, 31.0, 1.0, 5.0]
    return _state(pts, z, dead=(7,))


def _check_mixed_values(U, V, placement):
    """Particles 0-2 of ``_mixed_state`` sit in the all-deep cell: their
    values are ``a + b * clip(z, p[0], p[last])`` by hand."""
    p = P_PTS[placement]
    zc = np.clip(np.array([2.5, 13.0, 37.0]), p[0], p[LAST[placement][0]])
    assert np.all(_rel(U[:3], A_U + B_U * zc) < 1e-14), (placement, U[:3])
    assert np.all(_rel(V[:3], A_V + B_V * zc) < 1e-14), (placement, V[:3])


@pytest.mark.parametrize("placement", ["level", "layer_linear"])
def test_dispatch_ignores_layer_argument(placement):
    mesh = _mesh()
    vel = _field(placement)
    st = _mixed_state()
    nn = find_nearest_nodes(mesh, st.lon, st.lat)
    ref = ip.interp_barycentric(st, mesh, vel, nn, 1.0)
    U, V, ok = ref
    _check_mixed_values(U, V, placement)
    alive = st.alive_mask
    assert not alive[7] and U[7] == 0.0 and not ok[7]
    assert not ok[8]                      # outside the mesh
    assert ok[:7].all()
    for layer in (0, 3, np.full(st.n_particles, 2, dtype=np.int32),
                  np.arange(st.n_particles, dtype=np.int32) % 3):
        got = ip.interp_barycentric(st, mesh, vel, nn, 1.0, layer)
        for a, b, name in zip(got, ref, ("U", "V", "ok")):
            assert_bitwise(a, b, f"{name} layer={layer!r}")
    # the dispatch reads state.z: a different depth gives a different U
    st2 = _mixed_state()
    st2.z[0] = 6.5
    U2, _, _ = ip.interp_barycentric(st2, mesh, vel, nn, 1.0, 0)
    assert U2[0] != U[0]
    assert_bitwise(U2[1:], U[1:], "U other particles")


@pytest.mark.parametrize("placement", ["level", "layer_linear"])
def test_elems_hint_path(placement):
    """``elems=`` (pre-validated containing elements) gives the bits of
    the internal search, and a hint is trusted as given (not re-searched),
    as in the layer kernel."""
    mesh = _mesh()
    st = _mixed_state()
    nn = find_nearest_nodes(mesh, st.lon, st.lat)
    _check_mixed_values(*ip.interp_barycentric(
        st, mesh, _field(placement), nn, 1.0)[:2], placement)
    # columns that differ from node to node, so the element matters
    vel = _field(placement)
    vel.u += 0.01 * np.arange(mesh.n_nodes)
    vel.v -= 0.02 * np.arange(mesh.n_nodes)
    ref = ip.interp_barycentric(st, mesh, vel, nn, 1.0)
    elems = np.full(st.n_particles, -1, dtype=np.int32)
    for i in range(st.n_particles):
        e, inside = _find_containing_element_local(
            st.lon[i], st.lat[i], int(nn[i]), mesh.node_to_elems,
            mesh.node_to_elems_count, mesh.elem_nodes, mesh.nodes_lon,
            mesh.nodes_lat)
        if inside:
            elems[i] = e
    assert (elems[:7] >= 0).all() and elems[8] == -1
    got = ip.interp_barycentric(st, mesh, vel, nn, 1.0, elems=elems)
    for a, b, name in zip(got, ref, ("U", "V", "ok")):
        assert_bitwise(a, b, f"{name} elems")
    # partly hinted (-1 = search here)
    half = elems.copy()
    half[::2] = -1
    got = ip.interp_barycentric(st, mesh, vel, nn, 1.0, elems=half)
    for a, b, name in zip(got, ref, ("U", "V", "ok")):
        assert_bitwise(a, b, f"{name} half elems")
    # a (wrong) hint is used as given: particle 0 sits in element 0
    assert elems[0] == 0
    wrong = elems.copy()
    wrong[0] = 1
    got = ip.interp_barycentric(st, mesh, vel, nn, 1.0, elems=wrong)
    assert got[2][0] and got[0][0] != ref[0][0]
    assert_bitwise(got[0][1:], ref[0][1:], "U other particles")


@pytest.mark.parametrize("placement", ["level", "layer_linear"])
def test_python_fallback_bitwise(placement, monkeypatch):
    mesh = _mesh()
    vel = _field(placement)
    st = _mixed_state()
    nn = find_nearest_nodes(mesh, st.lon, st.lat)
    elems = np.full(st.n_particles, -1, dtype=np.int32)
    elems[1] = int(_find_containing_element_local(
        st.lon[1], st.lat[1], int(nn[1]), mesh.node_to_elems,
        mesh.node_to_elems_count, mesh.elem_nodes, mesh.nodes_lon,
        mesh.nodes_lat)[0])
    for kw in ({}, {"elems": elems}):
        ref = ip.interp_barycentric(st, mesh, vel, nn, -1.0, 0, **kw)
        _check_mixed_values(-ref[0], -ref[1], placement)
        with monkeypatch.context() as mp:
            mp.setattr(ip, "HAS_NUMBA", False)
            got = ip.interp_barycentric(st, mesh, vel, nn, -1.0, 0, **kw)
        for a, b, name in zip(got, ref, ("U", "V", "ok")):
            assert_bitwise(a, b, f"{name} fallback {kw and 'elems'}")


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("placement", ["level", "layer_linear"])
def test_direct_call_refuses_nan_in_prefix(placement):
    mesh = _mesh()
    vel = _field(placement)
    vel.u[1, 0] = np.nan                # node 0 is 200 m deep: point 1 is wet
    st = _state(DEEP_PTS, 5.0)
    nn = find_nearest_nodes(mesh, st.lon, st.lat)
    with pytest.raises(ValueError, match=r"u = nan is non-finite at node 0, "
                                         r"point 1"):
        ip._interp_barycentric_dl(st, mesh, vel, nn, 1.0)
    with pytest.raises(ValueError, match=r"non-finite at node 0, point 1"):
        ip.interp_barycentric(st, mesh, vel, nn, 1.0)


@pytest.mark.parametrize("placement", ["level", "layer_linear"])
def test_lazy_field_checked_through_parents_before_any_blend(placement):
    """A lazy blend with a NaN in one parent's prefix is refused through
    that parent before the wrapper reads ``u`` (no blend is cached); a good
    pair passes and is then blended once for the kernel."""
    from kelp.core.io import LazyLerpField
    from kelp.numerics import vertical_uv as vu
    mesh = _mesh()
    st = _state(DEEP_PTS, 12.5)
    nn = find_nearest_nodes(mesh, st.lon, st.lat)
    bad = _field(placement)
    bad.v[1, 4] = np.nan                # node 4 is 200 m deep: point 1 wet
    lazy = LazyLerpField(_field(placement), bad, 0.25)
    with pytest.raises(ValueError, match=r"v = nan is non-finite at node 4, "
                                         r"point 1"):
        ip.interp_barycentric(st, mesh, lazy, nn, 1.0)
    assert "u" not in lazy.__dict__ and "v" not in lazy.__dict__
    good = LazyLerpField(_field(placement), _field(placement), 0.25)
    U, V, ok = ip.interp_barycentric(st, mesh, good, nn, 1.0)
    assert ok.all() and vu.is_checked(good, mesh)
    assert vu.is_checked(good.vel_a, mesh) and vu.is_checked(good.vel_b, mesh)
    np.testing.assert_allclose(U, A_U + B_U * 12.5, rtol=1e-14)


def test_direct_call_checks_once_and_rechecks_on_new_mesh():
    """A checked field is not scanned again; a new table (another mesh)
    re-runs the check."""
    from kelp.numerics import vertical_uv as vu
    mesh = _mesh()
    vel = _field("level")
    st = _state(DEEP_PTS, 5.0)
    nn = find_nearest_nodes(mesh, st.lon, st.lat)
    assert not vu.is_checked(vel, mesh)
    ip._interp_barycentric_dl(st, mesh, vel, nn, 1.0)
    assert vu.is_checked(vel, mesh)
    vel.u[1, 0] = np.nan                # in place: no new check on this mesh
    ip._interp_barycentric_dl(st, mesh, vel, nn, 1.0)
    mesh2 = _mesh()
    with pytest.raises(ValueError, match=r"non-finite at node 0, point 1"):
        ip._interp_barycentric_dl(st, mesh2, vel, nn, 1.0)


@pytest.mark.parametrize("placement", ["level", "layer_linear"])
def test_direct_sigma_mesh_refused(placement):
    mesh = _mesh()
    vel = _field(placement)
    st = _state(DEEP_PTS, 5.0)
    nn = find_nearest_nodes(mesh, st.lon, st.lat)
    ip._interp_barycentric_dl(st, mesh, vel, nn, 1.0)   # checked on z-level
    sig = _mesh()
    sig.vertical_coord = "sigma"
    for fld in (vel, _field(placement)):                # checked / unchecked
        with pytest.raises(ValueError, match=r"needs a z-level mesh"):
            ip._interp_barycentric_dl(st, sig, fld, nn, 1.0)
        with pytest.raises(ValueError, match=r"needs a z-level mesh"):
            ip.interp_barycentric(st, sig, fld, nn, 1.0)


@pytest.mark.parametrize("placement", ["level", "layer_linear"])
def test_single_point_refuses_depth_linear(placement):
    mesh = _mesh()
    vel = _field(placement)
    msg = (f"interp_barycentric_single: depth-linear horizontal velocity "
           f"(uv_vertical={placement!r}) has no depth argument (no "
           f"single-point depth-linear API in this version)")
    with pytest.raises(ValueError, match="^" + re.escape(msg) + "$"):
        ip.interp_barycentric_single(4.3, 4.2, 0, mesh, vel, 0)
    with pytest.raises(ValueError, match="^" + re.escape(msg) + "$"):
        ip.get_interpolator_single("barycentric")(4.3, 4.2, 0, mesh, vel, 0)


_NEEDS_BARY = "needs uv_interp='barycentric'"
_NO_DEPTH = ("has no depth argument (no single-point depth-linear API in "
             "this version)")
_LAYOUT = ("on node layout (uv_layout='node') needs uv_interp='barycentric'; "
           "this interpolator evaluates depth-linear u/v on element layout "
           "only")


@pytest.mark.parametrize("placement", ["level", "layer_linear"])
def test_other_interpolators_refuse(placement):
    """Each entry refuses before any slicing: ``layer=7`` is out of range
    for the field's rows, so a slice would raise IndexError (and the
    edge entries' node-layout refusal would come first).  ``leastsq``
    evaluates element-layout depth-linear u/v and refuses this node field
    by its layout."""
    mesh = _mesh()
    vel = _field(placement)
    st = _state(DEEP_PTS, 5.0)
    nn = find_nearest_nodes(mesh, st.lon, st.lat)
    with pytest.raises(IndexError):
        vel.u[7, :]
    elem_ids = np.zeros(st.n_particles, dtype=np.int32)
    calls = [
        ("interp_leastsq", _LAYOUT,
         lambda: ip.interp_leastsq(st, mesh, vel, nn, 1.0, 7)),
        ("interp_leastsq", _LAYOUT,
         lambda: ip.interp_leastsq(
             st, mesh, vel, nn, 1.0,
             np.full(st.n_particles, 7, dtype=np.int32))),
        ("interp_leastsq_single", _NO_DEPTH,
         lambda: ip.interp_leastsq_single(4.3, 4.2, 7, mesh, vel, 0)),
        ("interp_nearest_element", _NEEDS_BARY,
         lambda: ip.interp_nearest_element(elem_ids, vel, 7)),
        ("interp_edge_leastsq", _NEEDS_BARY,
         lambda: ip.interp_edge_leastsq(st, mesh, vel, nn, 1.0, 7)),
        ("interp_edge_leastsq_single", _NO_DEPTH,
         lambda: ip.interp_edge_leastsq_single(4.3, 4.2, 7, mesh, vel, 0)),
        ("interp_edge_leastsq", _NEEDS_BARY,
         lambda: ip.get_interpolator("edge_leastsq", "noslip")(
             st, mesh, vel, nn, 1.0, 7)),
        ("interp_leastsq", _LAYOUT,
         lambda: ip.get_interpolator("leastsq")(st, mesh, vel, nn, 1.0, 7)),
    ]
    for where, reason, call in calls:
        msg = (f"{where}: depth-linear horizontal velocity "
               f"(uv_vertical={placement!r}) {reason}")
        with pytest.raises(ValueError, match="^" + re.escape(msg) + "$"):
            call()


def test_reject_depth_linear_passes_layer_fields():
    for vel in (_field_layer_node(), _field_layer_element()):
        ip._reject_depth_linear(vel, "x", "y")      # no raise


# ---------------------------------------------------------------------------
# "layer" fields: untouched
# ---------------------------------------------------------------------------

def _field_layer_node(mesh=None):
    mesh = _mesh() if mesh is None else mesh
    n = mesh.n_nodes
    rng = np.random.default_rng(7)
    u = rng.normal(size=(len(LEVELS) - 1, n))
    v = rng.normal(size=(len(LEVELS) - 1, n))
    return VelocityField(u=u, v=v, eta=np.zeros(n),
                         w=np.zeros((len(LEVELS), n)), uv_layout="node")


def _field_layer_element(mesh=None):
    mesh = _mesh() if mesh is None else mesh
    n = mesh.n_nodes
    e = mesh.n_elements
    rng = np.random.default_rng(8)
    return VelocityField(u=rng.normal(size=(len(LEVELS) - 1, e)),
                         v=rng.normal(size=(len(LEVELS) - 1, e)),
                         eta=np.zeros(n), w=np.zeros((len(LEVELS), n)))


def test_layer_fields_unchanged():
    mesh = _mesh()
    vel = _field_layer_node(mesh)
    st = _mixed_state()
    nn = find_nearest_nodes(mesh, st.lon, st.lat)
    P = st.n_particles
    layers = find_layers(mesh.z_levels, st.z, nn, mesh.depth, st.alive_mask)
    assert len(set(layers[st.alive_mask].tolist())) > 1
    U = np.zeros(P)
    V = np.zeros(P)
    ok = np.zeros(P, dtype=bool)
    ip._loop_interp_bary_3d(
        P, st.alive_mask, st.lon, st.lat, nn,
        np.full(P, -1, dtype=np.int32), layers, 1.0, ip.UV_LAYOUT_NODE,
        mesh.node_to_elems, mesh.node_to_elems_count, mesh.elem_nodes,
        mesh.nodes_lon, mesh.nodes_lat, np.empty(0, dtype=np.float64),
        vel.u, vel.v, U, V, ok, mesh.r_earth)
    got = ip.interp_barycentric(st, mesh, vel, nn, 1.0, layers)
    for a, b, name in zip(got, (U, V, ok), ("U", "V", "ok")):
        assert_bitwise(a, b, f"{name} layer 3-D")
    # scalar layer: the 2-D loop on the pre-sliced row
    U = np.zeros(P)
    V = np.zeros(P)
    ok = np.zeros(P, dtype=bool)
    ip._loop_interp_bary(
        P, st.alive_mask, st.lon, st.lat, nn,
        np.full(P, -1, dtype=np.int32), 1.0, ip.UV_LAYOUT_NODE,
        mesh.node_to_elems, mesh.node_to_elems_count, mesh.elem_nodes,
        mesh.nodes_lon, mesh.nodes_lat, np.empty(0, dtype=np.float64),
        vel.u[2, :], vel.v[2, :], U, V, ok, mesh.r_earth)
    got = ip.interp_barycentric(st, mesh, vel, nn, 1.0, 2)
    for a, b, name in zip(got, (U, V, ok), ("U", "V", "ok")):
        assert_bitwise(a, b, f"{name} layer 2")
    # the other entries still accept layer fields
    ip.interp_barycentric_single(4.3, 4.2, 1, mesh, vel, 0)
    vel_e = _field_layer_element(mesh)
    ip.interp_leastsq(st, mesh, vel_e, nn, 1.0, layers)
    ip.interp_leastsq_single(4.3, 4.2, 1, mesh, vel_e, 0)
    ip.interp_nearest_element(np.zeros(P, dtype=np.int32), vel_e, 1)
    ip.interp_edge_leastsq(st, mesh, vel_e, nn, 1.0, layers)
    ip.interp_edge_leastsq_single(4.3, 4.2, 1, mesh, vel_e, 0)


@pytest.mark.parametrize("placement", ["level", "layer_linear"])
def test_depth_uniform_field_reproduces_layer_answer(placement):
    """u/v uniform in depth (a value per node, NaN below each node's last
    point).  Where every vertex of the containing triangle is wet
    (``depth >= z`` at all three), the depth-linear answer is the layer
    kernel's answer BIT FOR BIT wherever every vertex reads a data point
    directly (the depth on a point, or clipped at either end); between
    points each vertex value ``(1 - a) c + a c`` is within 1 ulp of ``c``,
    so the sum stays within 2 ulp of the largest vertex value
    (cancellation between vertices of opposite sign can make that many
    ulp of the sum itself).  Where a vertex is dry the layer kernel still
    reads it (layer mode has no walls): the depth-linear answer is the
    renormalised blend of the wet vertices' constants, or, with no vertex
    wet (a NaN depth included), the deepest vertex's constant itself."""
    mesh = _mesh()
    dp = vu.depth_points(mesh, placement)
    m, n = dp.p.size, mesh.n_nodes
    rng = np.random.default_rng(23)
    c_u = rng.normal(size=n) * 10.0 ** rng.uniform(-2, 2, n)
    c_v = rng.normal(size=n)
    below = np.arange(m)[:, None] > dp.last[None, :]
    dl = VelocityField(
        u=np.where(below, np.nan, np.repeat(c_u[None, :], m, axis=0)),
        v=np.where(below, np.nan, np.repeat(c_v[None, :], m, axis=0)),
        eta=np.zeros(n), w=np.zeros((len(LEVELS), n)), uv_layout="node",
        uv_vertical=placement)
    lay = VelocityField(u=np.repeat(c_u[None, :], len(LEVELS) - 1, axis=0),
                        v=np.repeat(c_v[None, :], len(LEVELS) - 1, axis=0),
                        eta=np.zeros(n), w=np.zeros((len(LEVELS), n)),
                        uv_layout="node")
    pts = rng.uniform(4.02, 5.98, (60, 2))
    nn = find_nearest_nodes(mesh, pts[:, 0], pts[:, 1])
    tris = []
    for i, (x, y) in enumerate(pts):
        e, inside = _find_containing_element_local(
            x, y, int(nn[i]), mesh.node_to_elems, mesh.node_to_elems_count,
            mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat)
        assert inside
        tris.append([int(nd) for nd in mesh.elem_nodes[e]])
    beds = mesh.depth[np.array(tris)]                  # (60, 3)

    def both(z):
        st = _state(pts, z)
        layers = find_layers(mesh.z_levels, st.z, nn, mesh.depth,
                             st.alive_mask)
        wet = beds >= z                                # NaN z: all False
        return (ip.interp_barycentric(st, mesh, dl, nn, 1.0),
                ip.interp_barycentric(st, mesh, lay, nn, 1.0, layers),
                wet.all(axis=1))

    counts = {"wet": 0, "wall": 0}

    def check_walls(z, Ud, Vd, all_wet):
        """The pairs with a dry vertex, by hand."""
        for i in np.flatnonzero(~all_wet):
            tri = tris[i]
            lam = _lam(mesh, tri, *pts[i])
            for got, c in ((Ud[i], c_u), (Vd[i], c_v)):
                want = _hand_wall(mesh, placement, tri, lam, z,
                                  lambda nd, zc, c=c: c[nd])
                if not (beds[i] >= z).any():
                    assert got == want, (z, i, got, want)  # a stored value
                else:
                    scale = np.abs(c[tri]).max()
                    assert abs(got - want) <= 1e-13 * scale, (z, i, got,
                                                              want)
        counts["wet"] += int(all_wet.sum())
        counts["wall"] += int((~all_wet).sum())
    direct = list(dp.p) + [-3.0, 0.0, 41.0, 1e4, np.nan]
    for z in direct:
        (Ud, Vd, okd), (Ul, Vl, okl), all_wet = both(z)
        assert okd.all() and okl.all()
        assert_bitwise(Ud[all_wet], Ul[all_wet], f"U at {z} m")
        assert_bitwise(Vd[all_wet], Vl[all_wet], f"V at {z} m")
        check_walls(z, Ud, Vd, all_wet)
    worst = 0.0
    for z in rng.uniform(dp.p[0], dp.p[-1], 40):
        # per vertex: within 1 ulp of the stored value
        for nd in range(n):
            k, a = vu.uv_bracket(dp.p, dp.last[nd], z)
            for f, c in ((dl.u, c_u[nd]), (dl.v, c_v[nd])):
                got = vu.uv_at(f, nd, k, a)
                assert abs(got - c) <= np.spacing(abs(c)), (z, nd, got, c)
        (Ud, Vd, _), (Ul, Vl, _), all_wet = both(z)
        for a_, b_, c in ((Ud, Ul, c_u), (Vd, Vl, c_v)):
            ulp = (np.abs(a_ - b_)[all_wet]
                   / np.spacing(np.abs(c).max()))
            worst = max(worst, float(ulp.max(initial=0.0)))
        check_walls(z, Ud, Vd, all_wet)
    assert worst <= 2.0, worst
    assert counts["wet"] > 1000 and counts["wall"] > 500, counts


# ---------------------------------------------------------------------------
# NUMBA_DISABLE_JIT: the pure-Python kernels give the same values
# ---------------------------------------------------------------------------

_SUBPROCESS_SCRIPT = r"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).resolve().parents[0]))
sys.path.insert(0, sys.argv[1])
import numpy as np
from numba.core import config
assert config.DISABLE_JIT, "NUMBA_DISABLE_JIT not honoured"
import kelp.numerics.interpolate as ip
assert str(Path(ip.__file__).resolve()).startswith(
    str(Path(sys.argv[1]).resolve())), ip.__file__
import tests.test_interp_depth_linear as t
from kelp.core.mesh import find_nearest_nodes
rows = {}
for placement in ("level", "layer_linear"):
    mesh = t._mesh()
    vel = t._field(placement)
    st = t._mixed_state()
    nn = find_nearest_nodes(mesh, st.lon, st.lat)
    U, V, ok = ip.interp_barycentric(st, mesh, vel, nn, 1.0)
    rows[placement] = [U.tolist(), V.tolist(), ok.tolist()]
print(json.dumps(rows))
"""


def test_numba_disabled_subprocess_matches(tmp_path):
    script = tmp_path / "interp_dl_nojit.py"
    script.write_text(_SUBPROCESS_SCRIPT)
    src = str(Path(__file__).resolve().parents[1] / "src")
    env = dict(os.environ, NUMBA_DISABLE_JIT="1")
    out = subprocess.run([sys.executable, str(script), src],
                         capture_output=True, text=True, timeout=300,
                         env=env)
    assert out.returncode == 0, out.stderr
    rows = json.loads(out.stdout)
    for placement in ("level", "layer_linear"):
        mesh = _mesh()
        vel = _field(placement)
        st = _mixed_state()
        U, V, ok = _run(mesh, vel, st)
        Up, Vp, okp = (np.array(r) for r in rows[placement])
        _check_mixed_values(Up, Vp, placement)
        assert_bitwise(okp, ok, f"ok {placement}")
        np.testing.assert_allclose(Up, U, rtol=1e-13, atol=0.0)
        np.testing.assert_allclose(Vp, V, rtol=1e-13, atol=0.0)
