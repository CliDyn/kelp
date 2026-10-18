"""Triangle recovery from FESOM 1.4 cell polygons (``readers/fesom14/_polygons.py``).

Published FESOM 1.4 output carries no element table, only each node's
median-dual cell as a polygon (``lon_bnds``/``lat_bnds``, padded to a fixed
vertex count ``nv`` by repeating the last vertex).  The fixtures build such
polygons from a known triangulation the way the published files do: around
each node, edge midpoint, triangle centroid, edge midpoint, ...; a coastal
node's polygon ends with the midpoint of its second boundary edge and the node
itself.  The recovered triangles must equal the generating ones in canonical
order (rows sorted by their sorted node triple, each counter-clockwise and
starting at its smallest node).
"""
import re

import numpy as np
import pytest

from kelp.readers.fesom14._polygons import recover_triangles

R_EARTH = 6_371_000.0


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _unit(lon, lat):
    lon = np.radians(np.asarray(lon, dtype=float))
    lat = np.radians(np.asarray(lat, dtype=float))
    c = np.cos(lat)
    return np.stack([c * np.cos(lon), c * np.sin(lon), np.sin(lat)], axis=-1)


def _lonlat(v):
    v = np.asarray(v, dtype=float)
    v = v / np.linalg.norm(v)
    return (float(np.degrees(np.arctan2(v[1], v[0]))),
            float(np.degrees(np.arcsin(v[2]))))


def _polygon_lists(lon, lat, tri):
    """Each node's cell polygon as a list of (lon, lat), in the published order.

    ``tri`` must be counter-clockwise.  Every fan of triangles around a node is
    walked counter-clockwise; a fan that ends at the coast adds its last edge
    midpoint and then the node itself.  Midpoints and centroids are spherical.
    """
    xyz = _unit(lon, lat)
    fan = [dict() for _ in range(len(lon))]   # fan[n][p] = (t, q): t = (n, p, q) CCW
    for t, (a, b, c) in enumerate(tri):
        for n, p, q in ((a, b, c), (b, c, a), (c, a, b)):
            fan[n][int(p)] = (t, int(q))
    polys = []
    for n, out in enumerate(fan):
        qs = {q for _, q in out.values()}
        starts = sorted(p for p in out if p not in qs)   # coast edges opening a fan
        if not starts:
            starts = [min(out)]                          # interior: one closed fan
        verts, walked = [], set()
        for p0 in starts:
            p = p0
            verts.append(_lonlat(xyz[n] + xyz[p]))
            while True:
                t, q = out[p]
                walked.add(t)
                verts.append(_lonlat(xyz[tri[t]].sum(axis=0)))
                if q == p0:
                    break
                verts.append(_lonlat(xyz[n] + xyz[q]))
                if q not in out:
                    verts.append((float(lon[n]), float(lat[n])))
                    break
                p = q
        assert len(walked) == len(out)
        polys.append(verts)
    return polys


def _polygons_from_triangulation(lon, lat, tri, nv, polys=None):
    """``(lon_bnds, lat_bnds)`` of shape ``(N, nv)`` as the published files hold them.

    Each polygon is padded to ``nv`` slots by repeating its last vertex; a
    polygon longer than ``nv`` is cut off after ``nv`` vertices.
    """
    if polys is None:
        polys = _polygon_lists(lon, lat, tri)
    lon_b = np.empty((len(polys), nv))
    lat_b = np.empty((len(polys), nv))
    for n, verts in enumerate(polys):
        verts = list(verts[:nv])
        verts += [verts[-1]] * (nv - len(verts))
        lon_b[n], lat_b[n] = zip(*verts)
    return lon_b, lat_b


def _unordered_polygons(lon, lat, tri, nv):
    """Each node's own position, the midpoints of its edges and the centroids
    of its triangles, in no particular order (the recovery does not use the
    order).  Works for vertex sets that are not a planar triangulation."""
    xyz = _unit(lon, lat)
    polys = [[(float(lon[n]), float(lat[n]))] for n in range(len(lon))]
    edges = set()
    for t in tri:
        cen = _lonlat(xyz[t].sum(axis=0))
        for n in t:
            polys[n].append(cen)
        for i, j in ((0, 1), (0, 2), (1, 2)):
            edges.add((min(t[i], t[j]), max(t[i], t[j])))
    for a, b in sorted(edges):
        mid = _lonlat(xyz[a] + xyz[b])
        polys[a].append(mid)
        polys[b].append(mid)
    return _polygons_from_triangulation(lon, lat, tri, nv, polys=polys)


def _patch(lon0, lat0, *, holes=((2, 2),), backslash=((3, 0), (2, 1)), seed=7):
    """A 6x6 jittered 1-degree grid, each cell split into two triangles.

    Cells listed in ``backslash`` take the SE-NW diagonal, the others SW-NE;
    the two given make grid node (3, 1) the only node with eight triangles
    (16 polygon vertices, all other polygons <= 12).  The cells in ``holes``
    are left out (the default: one island).  Nodes are numbered in a shuffled order.  Returns
    ``lon, lat, tri (CCW), idx`` with ``idx[i, j]`` the node at lon index i,
    lat index j.
    """
    rng = np.random.default_rng(seed)
    n = 6
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    glon = lon0 + ii + rng.uniform(-0.15, 0.15, ii.shape)
    glat = lat0 + jj + rng.uniform(-0.15, 0.15, jj.shape)
    idx = rng.permutation(n * n).reshape(n, n)
    lon = np.empty(n * n)
    lat = np.empty(n * n)
    lon[idx.ravel()] = glon.ravel()
    lat[idx.ravel()] = glat.ravel()
    tri = []
    for i in range(n - 1):
        for j in range(n - 1):
            if (i, j) in holes:
                continue
            sw, se, ne, nw = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
            if (i, j) in backslash:
                tri += [(sw, se, nw), (se, ne, nw)]
            else:
                tri += [(sw, se, ne), (sw, ne, nw)]
    lon = ((lon + 180.0) % 360.0) - 180.0
    return lon, lat, np.array(tri), idx


def _canonical(tri):
    """Rows sorted by sorted node triple; each row rotated to start at its
    smallest node, keeping the (counter-clockwise) cyclic order."""
    tri = np.asarray(tri)
    rows = np.array([np.roll(t, -int(np.argmin(t))) for t in tri])
    s = np.sort(rows, axis=1)
    return rows[np.lexsort((s[:, 2], s[:, 1], s[:, 0]))]


def _chord_m(p, q):
    return float(np.linalg.norm(_unit(*p) - _unit(*q))) * R_EARTH


@pytest.fixture(scope="module")
def island():
    return _patch(10.0, 40.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_recovers_generating_triangles(island):
    lon, lat, tri, idx = island
    lengths = np.array([len(p) for p in _polygon_lists(lon, lat, tri)])
    assert lengths.max() == 16 and np.flatnonzero(lengths == 16).tolist() == [idx[3, 1]]
    lon_b, lat_b = _polygons_from_triangulation(lon, lat, tri, nv=16)
    got, report = recover_triangles(lon, lat, lon_b, lat_b)
    assert got.dtype == np.int32 and got.shape == (len(tri), 3)
    np.testing.assert_array_equal(got, _canonical(tri))
    assert report["n_full_slot_polygons"] == 1      # node (3, 1): 16 of 16 slots
    assert report["n_duplicate_vertices"] == 0
    assert report["n_pinch_triangles"] == 0


def test_vertex_on_rounding_boundary(island):
    """Copies of one shared centroid straddle a boundary of rounding to 1e-7
    degrees (+-3e-8 deg around it): keys built by rounding would split the
    point, the tolerance clustering does not."""
    lon, lat, tri, idx = island
    polys = _polygon_lists(lon, lat, tri)
    t = 5
    cen = _lonlat(_unit(lon[tri[t]], lat[tri[t]]).sum(axis=0))
    edge = (np.floor(cen[0] / 1e-7) + 0.5) * 1e-7
    copies = dict(zip(tri[t].tolist(), (edge - 3e-8, edge + 3e-8, edge - 3e-8)))
    vals = list(copies.values())
    assert round(vals[0], 7) != round(vals[1], 7)
    for n, x in copies.items():
        k = [i for i, v in enumerate(polys[n]) if v == cen]
        assert len(k) == 1
        polys[n][k[0]] = (x, cen[1])
    lon_b, lat_b = _polygons_from_triangulation(lon, lat, tri, nv=16, polys=polys)
    got, _ = recover_triangles(lon, lat, lon_b, lat_b)
    np.testing.assert_array_equal(got, _canonical(tri))


def test_truncated_polygon_refused(island):
    """nv = 15 cuts node (3, 1)'s 16-vertex polygon: it fills every slot and
    loses its last vertex, a triangle centroid.  That leaves a hole the other
    checks accept; the accounting check refuses it and names the node."""
    lon, lat, tri, idx = island
    n0 = int(idx[3, 1])
    polys = _polygon_lists(lon, lat, tri)
    last = polys[n0][-1]
    assert any(last == _lonlat(_unit(lon[t], lat[t]).sum(axis=0)) for t in tri if n0 in t)
    lon_b, lat_b = _polygons_from_triangulation(lon, lat, tri, nv=15)
    with pytest.raises(ValueError, match="accounting check") as exc:
        recover_triangles(lon, lat, lon_b, lat_b)
    assert re.search(rf"\bnode {n0}\b", str(exc.value))


def _sliver():
    """Two triangles; node 2 lies 150 m north of the midpoint of edge 0-1, so
    the centroid of (0, 1, 2) lies ~50 m from that midpoint."""
    a, b = (10.0, 40.0), (11.0, 40.0)
    mid = _lonlat(_unit(*a) + _unit(*b))
    c = (mid[0], mid[1] + 150.0 / (R_EARTH * np.pi / 180.0))
    d = (10.5, 39.0)
    lon = np.array([a[0], b[0], c[0], d[0]])
    lat = np.array([a[1], b[1], c[1], d[1]])
    return lon, lat, np.array([[0, 1, 2], [0, 3, 1]])


def test_separation_ratio_refusal_and_override():
    lon, lat, tri = _sliver()
    cen = _lonlat(_unit(lon[:3], lat[:3]).sum(axis=0))
    mid = _lonlat(_unit(lon[:2], lat[:2]).sum(axis=0))
    assert 45.0 < _chord_m(cen, mid) < 55.0
    lon_b, lat_b = _polygons_from_triangulation(lon, lat, tri, nv=8)
    with pytest.raises(ValueError, match="separation_ratio") as exc:
        recover_triangles(lon, lat, lon_b, lat_b)
    msg = str(exc.value)
    assert "[0, 1, 2]" in msg and "[0, 1]" in msg
    got, _ = recover_triangles(lon, lat, lon_b, lat_b, separation_ratio=10.0)
    np.testing.assert_array_equal(got, _canonical(tri))


def test_duplicate_vertex_harmless(island):
    lon, lat, tri, idx = island
    polys = _polygon_lists(lon, lat, tri)
    inner, coast = int(idx[1, 1]), int(idx[0, 2])
    assert (float(lon[coast]), float(lat[coast])) == polys[coast][-1]
    polys[inner].insert(3, polys[inner][0])     # a midpoint listed twice
    polys[coast].insert(0, polys[coast][-1])    # the node itself first and last
    lon_b, lat_b = _polygons_from_triangulation(lon, lat, tri, nv=18, polys=polys)
    got, report = recover_triangles(lon, lat, lon_b, lat_b)
    np.testing.assert_array_equal(got, _canonical(tri))
    assert report["n_duplicate_vertices"] == 2


def test_dateline_patch():
    lon, lat, tri, idx = _patch(177.5, -20.0)
    assert lon.min() < -170.0 and lon.max() > 170.0
    lon_b, lat_b = _polygons_from_triangulation(lon, lat, tri, nv=16)
    assert (lon_b < -170.0).any() and (lon_b > 170.0).any()
    got, report = recover_triangles(lon, lat, lon_b, lat_b)
    np.testing.assert_array_equal(got, _canonical(tri))
    assert report["n_boundary_loops"] == 2


def test_euler_characteristic_and_loops(island):
    lon, lat, tri, idx = island
    lon_b, lat_b = _polygons_from_triangulation(lon, lat, tri, nv=16)
    _, report = recover_triangles(lon, lat, lon_b, lat_b)
    assert report["n_boundary_loops"] == 2      # the coast and the island
    assert report["euler_characteristic"] == 0  # 36 - 84 + 48
    assert report["n_components"] == 1

    lon2, lat2, tri2, _ = _patch(30.0, 40.0, holes=())
    lon_b2, lat_b2 = _polygons_from_triangulation(lon2, lat2, tri2, nv=16)
    _, report = recover_triangles(lon2, lat2, lon_b2, lat_b2)
    assert (report["n_boundary_loops"], report["euler_characteristic"]) == (1, 1)

    # both patches in one file: two components
    both_lon, both_lat = np.r_[lon, lon2], np.r_[lat, lat2]
    both_tri = np.r_[tri, tri2 + len(lon)]
    lon_b3, lat_b3 = _polygons_from_triangulation(both_lon, both_lat, both_tri, nv=16)
    got, report = recover_triangles(both_lon, both_lat, lon_b3, lat_b3)
    np.testing.assert_array_equal(got, _canonical(both_tri))
    assert report["n_components"] == 2
    assert report["n_boundary_loops"] == 3
    assert report["euler_characteristic"] == 1


def test_pinch_node():
    """Two triangles meet only at node 0: two fans, two boundary loops."""
    lon = np.array([10.0, 11.0, 11.0, 9.0, 9.0])
    lat = np.array([40.0, 39.5, 40.5, 40.5, 39.5])
    tri = np.array([[0, 1, 2], [0, 3, 4]])
    lon_b, lat_b = _polygons_from_triangulation(lon, lat, tri, nv=8)
    got, report = recover_triangles(lon, lat, lon_b, lat_b)
    np.testing.assert_array_equal(got, _canonical(tri))
    assert report["n_pinch_nodes"] == 1
    assert report["n_boundary_loops"] == 2
    assert report["n_components"] == 2
    assert report["euler_characteristic"] == 1
    assert report["n_duplicate_vertices"] == 1     # node 0 closes both fans


def _pinch_patch():
    """The 6x6 patch without cells (3, 2) and (2, 3): node (3, 3) is a pinch
    node, its triangles form two fans of two triangles that touch only at the
    node.  Its polygon lists only the first fan, as the published polygons do;
    the triangles of the second fan are listed by their two other nodes."""
    lon, lat, tri, idx = _patch(10.0, 40.0, holes=((3, 2), (2, 3)))
    y = int(idx[3, 3])
    polys = _polygon_lists(lon, lat, tri)
    own = (float(lon[y]), float(lat[y]))
    assert polys[y].count(own) == 2                  # two fans, each ends at the node
    polys[y] = polys[y][:polys[y].index(own) + 1]
    assert len(polys[y]) == 6                        # one fan: 2 triangles
    return lon, lat, tri, y, polys


def test_pinch_node_one_fan_listed():
    lon, lat, tri, y, polys = _pinch_patch()
    lon_b, lat_b = _polygons_from_triangulation(lon, lat, tri, nv=16, polys=polys)
    got, report = recover_triangles(lon, lat, lon_b, lat_b)
    np.testing.assert_array_equal(got, _canonical(tri))
    assert report["n_pinch_triangles"] == 2
    assert report["n_pinch_nodes"] == 1
    assert report["n_boundary_loops"] == 2   # the coast; both holes form one loop
    assert report["euler_characteristic"] == -1      # 36 - 83 + 46


def test_pinch_triangle_refused_when_polygon_full():
    """The same omission in a polygon that fills all nv slots: it may have
    been cut off at nv, so it is refused, not recovered."""
    lon, lat, tri, y, polys = _pinch_patch()
    polys[y] = [polys[y][0]] * (16 - len(polys[y])) + polys[y]
    lon_b, lat_b = _polygons_from_triangulation(lon, lat, tri, nv=16, polys=polys)
    with pytest.raises(ValueError, match="cut off") as exc:
        recover_triangles(lon, lat, lon_b, lat_b)
    assert re.search(rf"\bnode {y}\b", str(exc.value))


def test_pinch_triangle_ambiguous_third_node_refused():
    """A second node at the pinch node's position: the third vertex of the
    unlisted centroids matches two nodes."""
    lon, lat, tri, y, polys = _pinch_patch()
    lon2, lat2 = np.r_[lon, lon[y]], np.r_[lat, lat[y]]
    polys = polys + [[(float(lon[y]), float(lat[y]))]]
    lon_b, lat_b = _polygons_from_triangulation(lon2, lat2, tri, nv=16, polys=polys)
    with pytest.raises(ValueError, match="ambiguous") as exc:
        recover_triangles(lon2, lat2, lon_b, lat_b)
    assert f"[{y}, {len(lon)}]" in str(exc.value)


def test_centroid_of_two_matching_no_node_refused(island):
    """A centroid listed by two coastal nodes whose third vertex lies off the
    mesh, where there is no node."""
    lon, lat, tri, idx = island
    a, b = int(idx[0, 0]), int(idx[1, 0])
    ghost = _unit(lon[a] + 0.5, lat[a] - 1.0)
    cen = _lonlat(_unit(lon[[a, b]], lat[[a, b]]).sum(axis=0) + ghost)
    polys = _polygon_lists(lon, lat, tri)
    polys[a].insert(1, cen)
    polys[b].insert(1, cen)
    lon_b, lat_b = _polygons_from_triangulation(lon, lat, tri, nv=16, polys=polys)
    with pytest.raises(ValueError, match="matches no node") as exc:
        recover_triangles(lon, lat, lon_b, lat_b)
    assert f"node {min(a, b)} and node {max(a, b)}" in str(exc.value)


def test_edge_in_three_triangles_refused():
    lon = np.array([10.0, 11.0, 10.5, 10.5, 10.5])
    lat = np.array([40.0, 40.0, 41.0, 39.0, 42.0])
    tri = np.array([[0, 1, 2], [0, 3, 1], [0, 1, 4]])
    lon_b, lat_b = _unordered_polygons(lon, lat, tri, nv=12)
    with pytest.raises(ValueError, match=r"nodes \[0, 1\] lies in 3 triangles"):
        recover_triangles(lon, lat, lon_b, lat_b)


def test_duplicate_triangle_refused():
    """Two centroid copies 10 m apart, both within a loosened position_tol_m:
    two clusters owned by the same three nodes."""
    lon = np.array([10.0, 11.0, 10.5])
    lat = np.array([40.0, 40.0, 41.0])
    tri = np.array([[0, 1, 2]])
    cen = _lonlat(_unit(lon, lat).sum(axis=0))
    dlat = 5.0 / (R_EARTH * np.pi / 180.0)
    polys = [[(float(lon[n]), float(lat[n]))] for n in range(3)]
    for n in range(3):
        polys[n] += [(cen[0], cen[1] - dlat), (cen[0], cen[1] + dlat)]
    for a, b in ((0, 1), (0, 2), (1, 2)):
        mid = _lonlat(_unit(lon[[a, b]], lat[[a, b]]).sum(axis=0))
        polys[a].append(mid)
        polys[b].append(mid)
    lon_b, lat_b = _polygons_from_triangulation(lon, lat, tri, nv=8, polys=polys)
    with pytest.raises(ValueError, match=r"triangle \[0, 1, 2\] is listed twice"):
        recover_triangles(lon, lat, lon_b, lat_b, position_tol_m=100.0,
                          separation_ratio=1.0)


def test_degenerate_triangle_refused():
    """Three nodes on the equator, one great circle: zero orientation."""
    lon = np.array([10.0, 11.0, 13.0])
    lat = np.zeros(3)
    lon_b, lat_b = _unordered_polygons(lon, lat, np.array([[0, 1, 2]]), nv=8)
    with pytest.raises(ValueError, match="degenerate"):
        recover_triangles(lon, lat, lon_b, lat_b)


def test_euler_mismatch_refused():
    """A Moebius strip (five nodes, triangles (k, k+1, k+2) mod 5): every edge
    in one or two triangles, every vertex classified, but V - E + F = 0 while
    one component with one boundary loop needs 1."""
    rng = np.random.default_rng(3)
    ang = np.radians(90.0 + 72.0 * np.arange(5)) + rng.uniform(-0.1, 0.1, 5)
    lon = 10.0 + np.cos(ang) / np.cos(np.radians(40.0))
    lat = 40.0 + np.sin(ang)
    tri = np.array([[k, (k + 1) % 5, (k + 2) % 5] for k in range(5)])
    lon_b, lat_b = _unordered_polygons(lon, lat, tri, nv=12)
    with pytest.raises(ValueError, match="do not form a surface"):
        recover_triangles(lon, lat, lon_b, lat_b)


def test_unexplained_vertex_refused(island):
    lon, lat, tri, idx = island
    n = int(idx[1, 1])
    polys = _polygon_lists(lon, lat, tri)
    polys[n].insert(2, (float(lon[n]) + 0.3, float(lat[n]) + 0.1))
    lon_b, lat_b = _polygons_from_triangulation(lon, lat, tri, nv=18, polys=polys)
    with pytest.raises(ValueError, match="accounting check") as exc:
        recover_triangles(lon, lat, lon_b, lat_b)
    assert re.search(rf"\bnode {n}\b", str(exc.value))


def test_missing_edge_midpoint_refused(island):
    """The midpoint of an interior edge dropped from both polygons: the
    triangles are still found, but their nodes are not adjacent."""
    lon, lat, tri, idx = island
    a, b = int(idx[1, 1]), int(idx[2, 1])
    mid = _lonlat(_unit(lon[[a, b]], lat[[a, b]]).sum(axis=0))
    polys = _polygon_lists(lon, lat, tri)
    for n in (a, b):
        assert sum(_chord_m(v, mid) < 1e-6 for v in polys[n]) == 1
        polys[n] = [v for v in polys[n] if _chord_m(v, mid) >= 1e-6]
    lon_b, lat_b = _polygons_from_triangulation(lon, lat, tri, nv=16, polys=polys)
    with pytest.raises(ValueError, match="not adjacent") as exc:
        recover_triangles(lon, lat, lon_b, lat_b)
    assert f"nodes {min(a, b)} and {max(a, b)}" in str(exc.value)


def test_midpoint_of_non_adjacent_nodes_refused(island):
    """Both polygons list the midpoint of two nodes that share no triangle
    (it lies inside the island)."""
    lon, lat, tri, idx = island
    a, b = int(idx[1, 1]), int(idx[4, 4])
    mid = _lonlat(_unit(lon[[a, b]], lat[[a, b]]).sum(axis=0))
    polys = _polygon_lists(lon, lat, tri)
    polys[a].insert(1, mid)
    polys[b].insert(1, mid)
    lon_b, lat_b = _polygons_from_triangulation(lon, lat, tri, nv=18, polys=polys)
    with pytest.raises(ValueError, match="share no triangle") as exc:
        recover_triangles(lon, lat, lon_b, lat_b)
    assert f"nodes {min(a, b)} and {max(a, b)}" in str(exc.value)


def test_vertex_of_four_nodes_refused(island):
    lon, lat, tri, idx = island
    polys = _polygon_lists(lon, lat, tri)
    cen = _lonlat(_unit(lon[tri[3]], lat[tri[3]]).sum(axis=0))
    other = next(n for n in range(len(lon)) if n not in tri[3])
    polys[other].insert(1, cen)
    lon_b, lat_b = _polygons_from_triangulation(lon, lat, tri, nv=18, polys=polys)
    with pytest.raises(ValueError, match="shared by 4 nodes"):
        recover_triangles(lon, lat, lon_b, lat_b)


def test_node_in_no_triangle_refused(island):
    """An extra node whose polygon is only its own position."""
    lon, lat, tri, idx = island
    lon_b, lat_b = _polygons_from_triangulation(lon, lat, tri, nv=16)
    lon2, lat2 = np.r_[lon, 20.0], np.r_[lat, 30.0]
    lon_b2 = np.r_[lon_b, np.full((1, 16), 20.0)]
    lat_b2 = np.r_[lat_b, np.full((1, 16), 30.0)]
    with pytest.raises(ValueError, match=rf"\[{len(lon)}\] belong to no triangle"):
        recover_triangles(lon2, lat2, lon_b2, lat_b2)


def test_shared_point_off_centroid_refused(island):
    """All three copies of a centroid moved 5 cm north: one cluster, owned by
    three nodes, but not their spherical centroid within 1 cm."""
    lon, lat, tri, idx = island
    polys = _polygon_lists(lon, lat, tri)
    t = 7
    cen = _lonlat(_unit(lon[tri[t]], lat[tri[t]]).sum(axis=0))
    moved = (cen[0], cen[1] + 0.05 / (R_EARTH * np.pi / 180.0))
    for n in tri[t]:
        polys[n] = [moved if v == cen else v for v in polys[n]]
    lon_b, lat_b = _polygons_from_triangulation(lon, lat, tri, nv=16, polys=polys)
    with pytest.raises(ValueError, match="position_tol_m"):
        recover_triangles(lon, lat, lon_b, lat_b)
    got, _ = recover_triangles(lon, lat, lon_b, lat_b, position_tol_m=0.1)
    np.testing.assert_array_equal(got, _canonical(tri))


@pytest.mark.parametrize("case, match", [
    ("nan_bound", "finite"),
    ("shape", "shape"),
    ("radius", "radius_deg"),
    ("ratio", "separation_ratio"),
])
def test_refuses_bad_input(island, case, match):
    lon, lat, tri, idx = island
    lon_b, lat_b = _polygons_from_triangulation(lon, lat, tri, nv=16)
    kw = {}
    if case == "nan_bound":
        lon_b[4, 3] = np.nan
    elif case == "shape":
        lat_b = lat_b[:, :-1]
    elif case == "radius":
        kw = {"radius_deg": 0.0}
    elif case == "ratio":
        kw = {"separation_ratio": 0.5}
    with pytest.raises(ValueError, match=match):
        recover_triangles(lon, lat, lon_b, lat_b, **kw)
