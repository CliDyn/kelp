"""Exact triangle recovery from FESOM 1.4 cell polygons.

Published FESOM 1.4 output (e.g. CMIP6) carries no element table.  It carries
each node's median-dual cell as a polygon (``lon_bnds``/``lat_bnds``), padded
to a fixed vertex count ``nv`` by repeating the last vertex.  An interior
node's polygon alternates the midpoints of its edges and the centroids of its
triangles; a coastal node's polygon also holds the midpoints of its boundary
edges and the node itself.  A polygon vertex shared by exactly three nodes is
therefore the centroid of their triangle, and :func:`recover_triangles`
rebuilds the triangles from that rule.  At a pinch node (a node whose
triangles form several fans that touch only at the node) the published
polygon lists one fan only; a centroid listed by just two nodes then names its
third node, and that triangle is recovered as well.

Vertices are matched by tolerance clustering on the unit sphere: no seam at
the dateline, and no rounding grid that could split one point lying on a
rounding boundary.  The result is validated before it is returned; the checks
are listed in :func:`recover_triangles`.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from ...core.geometry import R_EARTH_STANDARD, lonlat_to_xyz, xyz_to_lonlat

_PREFIX = "FESOM 1.4 cell polygons (lon_bnds/lat_bnds): "
_MAX_LISTED = 10


def _chord(angle_deg: float) -> float:
    """Chord length on the unit sphere for an arc of ``angle_deg`` degrees."""
    return 2.0 * np.sin(np.radians(angle_deg) / 2.0)


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _components(n: int, i: np.ndarray, j: np.ndarray) -> Tuple[int, np.ndarray]:
    """Connected components of the undirected graph on ``n`` vertices with
    edges ``(i[k], j[k])`` (the union-find of the pairs)."""
    graph = coo_matrix((np.ones(len(i), dtype=bool), (i, j)), shape=(n, n))
    return connected_components(graph, directed=False)


def _listed(values) -> str:
    """``[a, b, c]``, at most ``_MAX_LISTED`` entries, then the total."""
    values = [int(v) for v in values]
    text = "[" + ", ".join(str(v) for v in values[:_MAX_LISTED])
    if len(values) > _MAX_LISTED:
        return text + f", ...] ({len(values)} in all)"
    return text + "]"


def _where(xyz: np.ndarray) -> str:
    lon, lat = xyz_to_lonlat(xyz)
    return f"({float(lon):.7f}, {float(lat):.7f})"


def recover_triangles(
    lon: np.ndarray,
    lat: np.ndarray,
    lon_bnds: np.ndarray,
    lat_bnds: np.ndarray,
    *,
    radius_deg: float = 1e-6,
    separation_ratio: float = 1000.0,
    position_tol_m: float = 0.01,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Recover the triangles of a FESOM 1.4 mesh from its node cell polygons.

    The polygon vertices of all nodes are clustered on the unit sphere (chord
    radius ``radius_deg``, union-find over the pairs closer than that); each
    polygon counts its distinct vertices once, after its trailing padding
    (repeats of the last vertex) is dropped.  A cluster owned by exactly three
    nodes is the centroid of their triangle; a centroid owned by two nodes
    whose third vertex is a node with a polygon that has room is a pinch
    triangle (see the accounting check) and is recovered too.  Every
    recovered triangle goes through the checks below.

    Refused with a ``ValueError`` (the message names the nodes involved):

    - two distinct clusters closer than ``separation_ratio * radius_deg``
      (such points cannot be told from one point split by rounding);
    - a vertex shared by more than three nodes;
    - a shared point farther than ``position_tol_m`` from the spherical
      centroid of its three nodes;
    - an edge in more than two triangles, a node in no triangle, a duplicate
      or degenerate triangle;
    - a triangle edge with no polygon vertex at its midpoint (the triangle's
      nodes are not pairwise adjacent);
    - the **accounting check**: every distinct polygon vertex must be the
      node's own position, the midpoint of a recovered edge, or the centroid
      of a recovered triangle.  A centroid listed by only two nodes ``a, b``
      names its third vertex ``c`` (``G`` is proportional to ``a + b + c``).
      It is refused when ``c`` matches no node within ``position_tol_m``, when
      it matches more than one, and when the matched node's polygon fills all
      ``nv`` slots: that polygon was cut off at ``nv``, and the hole it leaves
      passes every other check.  When the matched node's polygon does not fill
      its slots, the polygon builder walked only one fan of triangles around a
      pinch node (a node whose triangles touch only at the node): the triangle
      is recovered like the others and counted as a pinch triangle;
    - an Euler characteristic that does not match the boundary loops:
      ``V - E + F == 2 C - L`` with ``C`` the edge-connected components and
      ``L`` the boundary loops, after each pinch node (a node whose triangles
      form several fans) is split into one vertex per fan.

    Parameters
    ----------
    lon, lat : np.ndarray
        Node coordinates in degrees, shape ``(N,)``.
    lon_bnds, lat_bnds : np.ndarray
        Cell polygon vertices in degrees, shape ``(N, nv)``; unused trailing
        slots repeat the last vertex.
    radius_deg : float, optional
        Clustering radius in degrees (default 1e-6, about 0.1 m).
    separation_ratio : float, optional
        Two distinct clusters must lie at least ``separation_ratio *
        radius_deg`` apart (default 1000, about 111 m).
    position_tol_m : float, optional
        Largest distance in metres between a shared point and the spherical
        centroid (or midpoint) it stands for (default 0.01).

    Returns
    -------
    tri : np.ndarray
        ``(E, 3)`` int32 node indices, each row counter-clockwise seen from
        outside the sphere and starting at its smallest node; rows sorted by
        their sorted node triple.
    report : dict
        ``n_full_slot_polygons`` (polygons that fill all ``nv`` slots),
        ``n_boundary_loops``, ``n_duplicate_vertices`` (polygon slots that
        repeat a vertex of the same polygon, padding excluded),
        ``n_pinch_triangles`` (triangles recovered from two of their three
        polygons), ``n_pinch_nodes``, ``n_components`` and
        ``euler_characteristic`` (``N - E + F`` of the recovered mesh).

    Raises
    ------
    ValueError
        On malformed input or any failed check above.
    """
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    lon_bnds = np.asarray(lon_bnds, dtype=np.float64)
    lat_bnds = np.asarray(lat_bnds, dtype=np.float64)
    if lon.ndim != 1 or lat.shape != lon.shape:
        raise ValueError(_PREFIX + f"lon and lat must be 1-D of equal shape, got "
                         f"{lon.shape} and {lat.shape}")
    n_nodes = lon.size
    if n_nodes == 0:
        raise ValueError(_PREFIX + "no nodes")
    if (lon_bnds.ndim != 2 or lon_bnds.shape[0] != n_nodes
            or lat_bnds.shape != lon_bnds.shape):
        raise ValueError(_PREFIX + f"lon_bnds and lat_bnds must both have shape "
                         f"(N, nv) with N = {n_nodes} nodes, got {lon_bnds.shape} "
                         f"and {lat_bnds.shape}")
    nv = lon_bnds.shape[1]
    if nv < 3:
        raise ValueError(_PREFIX + f"nv = {nv} vertex slots; a cell polygon needs "
                         "at least 3")
    bad = ~(np.isfinite(lon) & np.isfinite(lat)
            & np.isfinite(lon_bnds).all(axis=1) & np.isfinite(lat_bnds).all(axis=1))
    if bad.any():
        raise ValueError(_PREFIX + "non-finite coordinates at node(s) "
                         f"{_listed(np.flatnonzero(bad))}")
    if not (np.isfinite(radius_deg) and radius_deg > 0.0):
        raise ValueError(_PREFIX + f"radius_deg must be a positive number, got "
                         f"{radius_deg!r}")
    if not (np.isfinite(separation_ratio) and separation_ratio >= 1.0):
        raise ValueError(_PREFIX + f"separation_ratio must be >= 1, got "
                         f"{separation_ratio!r}")
    if not (np.isfinite(position_tol_m) and position_tol_m > 0.0):
        raise ValueError(_PREFIX + f"position_tol_m must be a positive number, got "
                         f"{position_tol_m!r}")

    # -- used slots: trailing repeats of the last vertex are padding ----------
    same = (lon_bnds[:, 1:] == lon_bnds[:, :-1]) & (lat_bnds[:, 1:] == lat_bnds[:, :-1])
    pad = np.zeros((n_nodes, nv), dtype=bool)
    pad[:, 1:] = np.logical_and.accumulate(same[:, ::-1], axis=1)[:, ::-1]
    used = ~pad
    full = used[:, -1]                       # the polygon fills all nv slots
    owner = np.nonzero(used)[0]              # node of each used slot (row-major)
    pts = lonlat_to_xyz(lon_bnds[used], lat_bnds[used])
    node_xyz = lonlat_to_xyz(lon, lat)

    # -- cluster the vertices: union-find over pairs within the chord radius --
    r_clu = _chord(radius_deg)
    pairs = cKDTree(pts).query_pairs(r_clu, output_type="ndarray")
    n_clu, label = _components(len(pts), pairs[:, 0], pairs[:, 1])
    rep = _normalize(np.stack(
        [np.bincount(label, weights=pts[:, k], minlength=n_clu) for k in range(3)],
        axis=1))

    def owners_of(c: int) -> np.ndarray:
        return np.unique(owner[label == c])

    # -- distinct clusters must be well separated -----------------------------
    sep_deg = separation_ratio * radius_deg
    close = cKDTree(rep).query_pairs(_chord(sep_deg), output_type="ndarray")
    if len(close):
        d = np.linalg.norm(rep[close[:, 0]] - rep[close[:, 1]], axis=1)
        k = int(np.argmin(d))
        ca, cb = int(close[k, 0]), int(close[k, 1])
        raise ValueError(
            _PREFIX + f"two distinct polygon vertices lie {d[k] * R_EARTH_STANDARD:.3g} m "
            f"apart, closer than separation_ratio * radius_deg = {separation_ratio:g} * "
            f"{radius_deg:g} deg (~{_chord(sep_deg) * R_EARTH_STANDARD:.3g} m): the vertex "
            f"at {_where(rep[ca])} of nodes {_listed(owners_of(ca))} and the vertex at "
            f"{_where(rep[cb])} of nodes {_listed(owners_of(cb))}.  Points this close "
            "cannot be told from one point split by rounding; if the mesh really has "
            "features this small, pass a smaller separation_ratio")

    # -- distinct (node, vertex) pairs, owners per vertex ---------------------
    key = owner.astype(np.int64) * n_clu + label
    ukey = np.unique(key)
    n_duplicate = len(key) - len(ukey)
    p_node = ukey // n_clu
    p_clu = ukey % n_clu
    n_own = np.bincount(p_clu, minlength=n_clu)
    by_clu = p_node[np.argsort(p_clu, kind="stable")]   # owners grouped, ascending
    start = np.concatenate(([0], np.cumsum(n_own)[:-1]))

    many = np.flatnonzero(n_own > 3)
    if many.size:
        c = int(many[0])
        raise ValueError(
            _PREFIX + f"accounting check failed: the vertex at {_where(rep[c])} is shared "
            f"by {int(n_own[c])} nodes {_listed(owners_of(c))}; a vertex of a median-dual "
            f"cell of a triangle mesh belongs to at most three cells "
            f"({many.size} such vertices)")

    tol = position_tol_m / R_EARTH_STANDARD

    # -- triangles listed by all three nodes: vertices with three owners ------
    tri_c = np.flatnonzero(n_own == 3)
    s = start[tri_c]
    tri = np.stack([by_clu[s], by_clu[s + 1], by_clu[s + 2]], axis=1)   # a < b < c
    off = np.linalg.norm(_normalize(node_xyz[tri].sum(axis=1)) - rep[tri_c], axis=1)
    if (off > tol).any():
        k = int(np.argmax(off))
        raise ValueError(
            _PREFIX + f"the vertex at {_where(rep[tri_c[k]])} shared by nodes "
            f"{_listed(tri[k])} lies {off[k] * R_EARTH_STANDARD:.3g} m from their "
            f"spherical centroid (position_tol_m = {position_tol_m:g}); "
            f"{int((off > tol).sum())} shared point(s) off")

    # -- classify the other vertices ------------------------------------------
    # two owners: the midpoint of their edge, else a centroid one node left out
    two_c = np.flatnonzero(n_own == 2)
    ea, eb = by_clu[start[two_c]], by_clu[start[two_c] + 1]
    mid_off = np.linalg.norm(_normalize(node_xyz[ea] + node_xyz[eb]) - rep[two_c], axis=1)
    is_mid = mid_off <= tol
    mid_keys = [ea[is_mid].astype(np.int64) * n_nodes + eb[is_mid]]
    o2_c, o2_a, o2_b = two_c[~is_mid], ea[~is_mid], eb[~is_mid]

    # one owner: the node itself, else a midpoint the other node left out
    one_c = np.flatnonzero(n_own == 1)
    o1_n = by_clu[start[one_c]]
    is_self = np.linalg.norm(rep[one_c] - node_xyz[o1_n], axis=1) <= r_clu
    o1_c, o1_n = one_c[~is_self], o1_n[~is_self]

    pinch = np.zeros((0, 3), dtype=np.int64)
    if o2_c.size or o1_c.size:
        node_tree = cKDTree(node_xyz)
    if o2_c.size:
        # A centroid G = |a + b + c|^-1 (a + b + c) listed by a and b only: the
        # third vertex c = lam G - a - b, with lam fixed by |c| = 1.
        g, sab = rep[o2_c], node_xyz[o2_a] + node_xyz[o2_b]
        gs = np.einsum("ij,ij->i", g, sab)
        disc = gs ** 2 - np.einsum("ij,ij->i", sab, sab) + 1.0
        third = (gs + np.sqrt(np.maximum(disc, 0.0)))[:, None] * g - sab
        n_near = node_tree.query_ball_point(third, tol, return_length=True)
        _, o2_t = node_tree.query(third)
        amb = (disc >= 0.0) & (n_near > 1)
        if amb.any():
            k = int(np.argmax(amb))
            near = node_tree.query_ball_point(third[k], tol)
            raise ValueError(
                _PREFIX + f"accounting check failed: the vertex at {_where(rep[o2_c[k]])} "
                f"listed by nodes {int(o2_a[k])} and {int(o2_b[k])} is the centroid of a "
                f"triangle whose third vertex at {_where(_normalize(third[k]))} matches "
                f"{len(near)} nodes {_listed(sorted(near))} within position_tol_m = "
                f"{position_tol_m:g} (ambiguous)")
        found = ((disc >= 0.0) & (n_near == 1)
                 & (o2_t != o2_a) & (o2_t != o2_b))
        cut = found & full[o2_t]
        if cut.any():
            k = int(np.flatnonzero(cut)[0])
            a, b, c = int(o2_a[k]), int(o2_b[k]), int(o2_t[k])
            raise ValueError(
                _PREFIX + f"accounting check failed: the polygon of node {c} fills "
                f"all nv = {nv} slots and lacks the centroid of its triangle "
                f"{_listed(sorted((a, b, c)))}, which the polygons of nodes {a} and "
                f"{b} list: the polygon was cut off at nv slots.  Polygons cut this "
                f"way: nodes {_listed(np.unique(o2_t[cut]))}")
        if not found.all():
            k = int(np.argmin(found))
            raise ValueError(
                _PREFIX + f"accounting check failed: the vertex at {_where(rep[o2_c[k]])} "
                f"of node {int(o2_a[k])} and node {int(o2_b[k])} is not the midpoint of "
                "their edge, and as the centroid of a triangle with them its third "
                f"vertex at {_where(_normalize(third[k]))} matches no node within "
                f"position_tol_m = {position_tol_m:g} "
                f"({int((~found).sum())} such vertices)")
        # Listed by two nodes, left out by the third, whose polygon has room:
        # the polygon builder walked one fan of a pinch node.  Recovered.
        pinch = np.sort(np.stack([o2_a, o2_b, o2_t], axis=1), axis=1)
    if o1_c.size:
        # A midpoint M = |a + m|^-1 (a + m) listed by a only: m = 2 (a.M) M - a.
        mp, pa = rep[o1_c], node_xyz[o1_n]
        guess = 2.0 * np.einsum("ij,ij->i", pa, mp)[:, None] * mp - pa
        _, o1_m = node_tree.query(guess)
        mid = _normalize(pa + node_xyz[o1_m])
        ok = (np.linalg.norm(mid - mp, axis=1) <= tol) & (o1_m != o1_n)
        if not ok.all():
            k = int(np.argmin(ok))
            raise ValueError(
                _PREFIX + f"accounting check failed: the vertex at {_where(rep[o1_c[k]])} "
                f"of node {int(o1_n[k])} is not the node's own position, not the "
                "midpoint of an edge and not the centroid of a triangle of the mesh "
                f"({int((~ok).sum())} such vertices)")
        lo, hi = np.minimum(o1_n, o1_m), np.maximum(o1_n, o1_m)
        mid_keys.append(lo.astype(np.int64) * n_nodes + hi)

    # -- the triangle set: edges, adjacency -----------------------------------
    tri = np.concatenate([tri, pinch])
    n_tri = len(tri)
    ekey_t = np.concatenate([tri[:, 0] * n_nodes + tri[:, 1],
                             tri[:, 0] * n_nodes + tri[:, 2],
                             tri[:, 1] * n_nodes + tri[:, 2]]).astype(np.int64)
    ekeys, ecount = np.unique(ekey_t, return_counts=True)
    if (ecount > 2).any():
        k = int(np.argmax(ecount))
        raise ValueError(
            _PREFIX + f"edge between nodes {_listed(divmod(int(ekeys[k]), n_nodes))} "
            f"lies in {int(ecount[k])} triangles (at most 2 allowed); "
            f"{int((ecount > 2).sum())} such edge(s)")

    mid_keys = np.concatenate(mid_keys)
    stray = ~np.isin(mid_keys, ekeys)
    if stray.any():
        a, b = divmod(int(mid_keys[np.argmax(stray)]), n_nodes)
        raise ValueError(
            _PREFIX + f"accounting check failed: a polygon vertex is the midpoint of "
            f"nodes {a} and {b}, which share no triangle of the mesh "
            f"({int(stray.sum())} such vertices)")
    lonely = ~np.isin(ekeys, mid_keys)
    if lonely.any():
        a, b = divmod(int(ekeys[np.argmax(lonely)]), n_nodes)
        raise ValueError(
            _PREFIX + f"nodes {a} and {b} share a triangle but are not adjacent: neither "
            f"polygon has a vertex at the midpoint of their edge "
            f"({int(lonely.sum())} such edges)")

    in_tri = np.zeros(n_nodes, dtype=bool)
    in_tri[tri.ravel()] = True
    if not in_tri.all():
        raise ValueError(_PREFIX + "node(s) "
                         f"{_listed(np.flatnonzero(~in_tri))} belong to no triangle")

    # -- canonical order: sorted triples, then counter-clockwise rows ---------
    tri = tri[np.lexsort((tri[:, 2], tri[:, 1], tri[:, 0]))]
    if n_tri > 1:
        dup = (tri[1:] == tri[:-1]).all(axis=1)
        if dup.any():
            raise ValueError(_PREFIX + f"triangle {_listed(tri[np.argmax(dup)])} is "
                             "listed twice")
    pa, pb, pc = node_xyz[tri[:, 0]], node_xyz[tri[:, 1]], node_xyz[tri[:, 2]]
    det = np.einsum("ij,ij->i", pa, np.cross(pb - pa, pc - pa))
    if (det == 0.0).any():
        raise ValueError(_PREFIX + f"triangle {_listed(tri[np.argmax(det == 0.0)])} is "
                         "degenerate (its nodes lie on one great circle)")
    cw = det < 0.0
    tri[cw, 1], tri[cw, 2] = tri[cw, 2], tri[cw, 1].copy()

    # -- topology: fans, components, boundary loops, Euler characteristic -----
    n_edges = len(ekeys)
    t_idx = np.repeat(np.arange(n_tri), 3)
    j0 = np.tile(np.arange(3), n_tri)
    j1 = (j0 + 1) % 3
    u, v = tri[t_idx, j0], tri[t_idx, j1]
    inc_lo = np.where(u < v, 3 * t_idx + j0, 3 * t_idx + j1)    # corner of the lower node
    inc_hi = np.where(u < v, 3 * t_idx + j1, 3 * t_idx + j0)
    okey = np.minimum(u, v).astype(np.int64) * n_nodes + np.maximum(u, v)
    order = np.argsort(okey, kind="stable")
    okey, t_idx, inc_lo, inc_hi = okey[order], t_idx[order], inc_lo[order], inc_hi[order]
    paired = np.flatnonzero(okey[1:] == okey[:-1])              # interior edges
    lone = np.ones(len(okey), dtype=bool)
    lone[paired] = False
    lone[paired + 1] = False                                    # boundary edges
    n_fans, fan = _components(3 * n_tri,
                              np.concatenate([inc_lo[paired], inc_hi[paired]]),
                              np.concatenate([inc_lo[paired + 1], inc_hi[paired + 1]]))
    n_comp = _components(n_tri, t_idx[paired], t_idx[paired + 1])[0]
    b_lo, b_hi = fan[inc_lo[lone]], fan[inc_hi[lone]]
    loop = _components(n_fans, b_lo, b_hi)[1]
    n_loops = len(np.unique(loop[b_lo]))
    node_fan = np.unique(tri.ravel().astype(np.int64) * n_fans + fan)   # corner 3t + j
    n_pinch = int((np.bincount(node_fan // n_fans, minlength=n_nodes) > 1).sum())
    chi_split = n_fans - n_edges + n_tri
    if chi_split != 2 * n_comp - n_loops:
        raise ValueError(
            _PREFIX + f"the recovered triangles do not form a surface on the sphere: "
            f"V - E + F = {chi_split} (V counted per fan of triangles) but 2 C - L = "
            f"{2 * n_comp - n_loops} with C = {n_comp} components and L = {n_loops} "
            "boundary loops (overlapping or folded triangles)")

    report = {
        "n_full_slot_polygons": int(full.sum()),
        "n_boundary_loops": int(n_loops),
        "n_duplicate_vertices": int(n_duplicate),
        "n_pinch_triangles": int(len(pinch)),
        "n_pinch_nodes": n_pinch,
        "n_components": int(n_comp),
        "euler_characteristic": int(n_nodes - n_edges + n_tri),
    }
    return np.ascontiguousarray(tri, dtype=np.int32), report
