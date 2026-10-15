"""Tests for kelp.core.mesh."""
import numpy as np
import pytest

from kelp.core.mesh import (
    Mesh, build_search, find_nearest_nodes,
    find_containing_elements, build_node_to_elems,
)


# ---------------------------------------------------------------------------
# Fixtures: small synthetic meshes
# ---------------------------------------------------------------------------

def _make_two_triangles():
    """Two triangles forming a unit square [0,1]x[0,1].

    Nodes: 0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1)
    E0: [0,1,2]   E1: [1,3,2]
    """
    nodes_lon = np.array([0.0, 1.0, 0.0, 1.0])
    nodes_lat = np.array([0.0, 0.0, 1.0, 1.0])
    elem_nodes = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    n2e, n2e_count = build_node_to_elems(4, elem_nodes)
    cx = np.array([np.mean(nodes_lon[e]) for e in elem_nodes])
    cy = np.array([np.mean(nodes_lat[e]) for e in elem_nodes])
    return Mesh(name="test_tri", nodes_lon=nodes_lon, nodes_lat=nodes_lat,
                depth=np.zeros(4), elem_nodes=elem_nodes,
                elem_center_lon=cx, elem_center_lat=cy,
                node_to_elems=n2e, node_to_elems_count=n2e_count)


def _make_quad_mesh():
    """One quad [0,1]x[0,1]. Nodes: 0:(0,0), 1:(1,0), 2:(1,1), 3:(0,1)."""
    nodes_lon = np.array([0.0, 1.0, 1.0, 0.0])
    nodes_lat = np.array([0.0, 0.0, 1.0, 1.0])
    elem_nodes = np.array([[0, 1, 2, 3]], dtype=np.int32)
    n2e, n2e_count = build_node_to_elems(4, elem_nodes)
    return Mesh(name="test_quad", nodes_lon=nodes_lon, nodes_lat=nodes_lat,
                depth=np.zeros(4), elem_nodes=elem_nodes,
                elem_center_lon=np.array([0.5]), elem_center_lat=np.array([0.5]),
                node_to_elems=n2e, node_to_elems_count=n2e_count)


def _make_mixed_mesh():
    """One tri + one quad (FESOM-C style, padded with -1).

    Nodes: 0:(0,0), 1:(1,0), 2:(0.5,1), 3:(2,0), 4:(2,1)
    E0: tri  [0,1,2,-1]
    E1: quad [1,3,4,2]
    """
    nodes_lon = np.array([0.0, 1.0, 0.5, 2.0, 2.0])
    nodes_lat = np.array([0.0, 0.0, 1.0, 0.0, 1.0])
    elem_nodes = np.array([[0, 1, 2, -1], [1, 3, 4, 2]], dtype=np.int32)
    n2e, n2e_count = build_node_to_elems(5, elem_nodes)
    cx = np.array([np.mean(nodes_lon[[0,1,2]]), np.mean(nodes_lon[[1,3,4,2]])])
    cy = np.array([np.mean(nodes_lat[[0,1,2]]), np.mean(nodes_lat[[1,3,4,2]])])
    return Mesh(name="test_mixed", nodes_lon=nodes_lon, nodes_lat=nodes_lat,
                depth=np.zeros(5), elem_nodes=elem_nodes,
                elem_center_lon=cx, elem_center_lat=cy,
                node_to_elems=n2e, node_to_elems_count=n2e_count)


# ---------------------------------------------------------------------------
# Mesh properties
# ---------------------------------------------------------------------------

class TestMeshBasics:
    def test_n_nodes(self):
        assert _make_two_triangles().n_nodes == 4

    def test_n_elements(self):
        assert _make_two_triangles().n_elements == 2

    def test_elem_is_triangle_tri_mesh(self):
        mesh = _make_two_triangles()
        assert mesh.elem_is_triangle(0) is True

    def test_elem_is_triangle_mixed(self):
        mesh = _make_mixed_mesh()
        assert mesh.elem_is_triangle(0) == True   # padded -1
        assert mesh.elem_is_triangle(1) == False   # quad

    def test_elem_valid_nodes_tri(self):
        mesh = _make_mixed_mesh()
        np.testing.assert_array_equal(mesh.elem_valid_nodes(0), [0, 1, 2])

    def test_elem_valid_nodes_quad(self):
        mesh = _make_mixed_mesh()
        np.testing.assert_array_equal(mesh.elem_valid_nodes(1), [1, 3, 4, 2])

    def test_n_sigma_empty(self):
        assert _make_two_triangles().n_sigma == 0

    def test_vertical_coord_default(self):
        assert _make_two_triangles().vertical_coord == "zlevel"

    def test_z_levels_raises_for_sigma_mesh(self):
        mesh = _make_two_triangles()
        mesh.sigma = np.array([1.0, 0.5, 0.0], dtype=np.float64)
        mesh.vertical_coord = "sigma"
        with pytest.raises(NotImplementedError, match="only valid for z-level meshes"):
            _ = mesh.z_levels


# ---------------------------------------------------------------------------
# build_node_to_elems
# ---------------------------------------------------------------------------

class TestBuildNodeToElems:
    def test_two_triangles(self):
        elem_nodes = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
        n2e, counts = build_node_to_elems(4, elem_nodes)
        assert counts[0] == 1  # node 0: E0 only
        assert counts[1] == 2  # node 1: E0, E1
        assert counts[2] == 2  # node 2: E0, E1
        assert counts[3] == 1  # node 3: E1 only
        assert set(n2e[1, :2]) == {0, 1}

    def test_mixed_mesh(self):
        elem_nodes = np.array([[0, 1, 2, -1], [1, 3, 4, 2]], dtype=np.int32)
        n2e, counts = build_node_to_elems(5, elem_nodes)
        assert counts[0] == 1  # only in E0
        assert counts[1] == 2  # in E0 and E1
        assert counts[3] == 1  # only in E1

    def test_empty(self):
        _, counts = build_node_to_elems(3, np.zeros((0, 3), dtype=np.int32))
        np.testing.assert_array_equal(counts, [0, 0, 0])


# ---------------------------------------------------------------------------
# KD-tree
# ---------------------------------------------------------------------------

class TestKDTree:
    def test_exact_node_positions(self):
        mesh = _make_two_triangles()
        build_search(mesh)
        lons = np.array([0.0, 1.0, 0.0, 1.0])
        lats = np.array([0.0, 0.0, 1.0, 1.0])
        nn = find_nearest_nodes(mesh, lons, lats)
        np.testing.assert_array_equal(nn, [0, 1, 2, 3])

    def test_nearest_interior_point(self):
        mesh = _make_two_triangles()
        build_search(mesh)
        nn = find_nearest_nodes(mesh, np.array([0.1]), np.array([0.1]))
        assert nn[0] == 0  # closest to origin node

    def test_lazy_kdtree_fallback(self):
        """Without build_search(), KD-tree is built lazily on first query."""
        mesh = _make_two_triangles()
        # No build_search() called — should still work via lazy KD-tree
        nn = find_nearest_nodes(mesh, np.array([0.0]), np.array([0.0]))
        assert nn[0] >= 0  # valid node index


# ---------------------------------------------------------------------------
# Element containment
# ---------------------------------------------------------------------------

class TestFindContainingElements:
    def test_inside_lower_tri(self):
        mesh = _make_two_triangles()
        build_search(mesh)
        lons, lats = np.array([0.2]), np.array([0.2])
        nn = find_nearest_nodes(mesh, lons, lats)
        elem_ids, in_domain = find_containing_elements(mesh, lons, lats, nn)
        assert in_domain[0] == True

    def test_inside_upper_tri(self):
        mesh = _make_two_triangles()
        build_search(mesh)
        lons, lats = np.array([0.8]), np.array([0.8])
        nn = find_nearest_nodes(mesh, lons, lats)
        _, in_domain = find_containing_elements(mesh, lons, lats, nn)
        assert in_domain[0] == True

    def test_outside_domain(self):
        mesh = _make_two_triangles()
        build_search(mesh)
        lons, lats = np.array([5.0]), np.array([5.0])
        nn = find_nearest_nodes(mesh, lons, lats)
        elem_ids, in_domain = find_containing_elements(mesh, lons, lats, nn)
        assert in_domain[0] == False
        assert elem_ids[0] == -1

    def test_inside_quad(self):
        mesh = _make_quad_mesh()
        build_search(mesh)
        lons, lats = np.array([0.5]), np.array([0.5])
        nn = find_nearest_nodes(mesh, lons, lats)
        elem_ids, in_domain = find_containing_elements(mesh, lons, lats, nn)
        assert in_domain[0] == True
        assert elem_ids[0] == 0

    def test_mixed_tri_region(self):
        mesh = _make_mixed_mesh()
        build_search(mesh)
        lons, lats = np.array([0.5]), np.array([0.3])
        nn = find_nearest_nodes(mesh, lons, lats)
        elem_ids, in_domain = find_containing_elements(mesh, lons, lats, nn)
        assert in_domain[0] == True
        assert elem_ids[0] == 0  # triangle

    def test_mixed_quad_region(self):
        mesh = _make_mixed_mesh()
        build_search(mesh)
        lons, lats = np.array([1.5]), np.array([0.5])
        nn = find_nearest_nodes(mesh, lons, lats)
        elem_ids, in_domain = find_containing_elements(mesh, lons, lats, nn)
        assert in_domain[0] == True
        assert elem_ids[0] == 1  # quad

    def test_multiple_particles(self):
        mesh = _make_two_triangles()
        build_search(mesh)
        lons = np.array([0.2, 0.8, 5.0])
        lats = np.array([0.2, 0.8, 5.0])
        nn = find_nearest_nodes(mesh, lons, lats)
        _, in_domain = find_containing_elements(mesh, lons, lats, nn)
        assert in_domain[0] == True
        assert in_domain[1] == True
        assert in_domain[2] == False


# ---------------------------------------------------------------------------
# Grid hash: longitude branch selection (prime meridian / dateline / global)
# ---------------------------------------------------------------------------

def _grid_mesh(xs: np.ndarray, ys: np.ndarray) -> Mesh:
    """Regular triangulated grid mesh from 1-D lon/lat axes (degrees)."""
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel().astype(np.float64)
    lat = gy.ravel().astype(np.float64)
    nx, ny = len(xs), len(ys)
    elems = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            n1, n2 = n0 + 1, n0 + nx
            elems.append([n0, n1, n2])
            elems.append([n1, n2 + 1, n2])
    elem_nodes = np.array(elems, dtype=np.int32)
    n2e, n2e_count = build_node_to_elems(nx * ny, elem_nodes)
    mesh = Mesh(name="grid", nodes_lon=lon, nodes_lat=lat,
                depth=np.zeros(nx * ny), elem_nodes=elem_nodes,
                elem_center_lon=np.array([lon[e].mean() for e in elem_nodes]),
                elem_center_lat=np.array([lat[e].mean() for e in elem_nodes]),
                node_to_elems=n2e, node_to_elems_count=n2e_count)
    build_search(mesh)
    return mesh


def _nearest_bruteforce(mesh: Mesh, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Oracle: argmin of 3-D cartesian distance on the unit sphere."""
    def xyz(lo, la):
        lo_r, la_r = np.radians(lo), np.radians(la)
        return np.column_stack((np.cos(la_r) * np.cos(lo_r),
                                np.cos(la_r) * np.sin(lo_r),
                                np.sin(la_r)))
    nodes = xyz(mesh.nodes_lon, mesh.nodes_lat)
    pts = xyz(np.asarray(lons, float), np.asarray(lats, float))
    d2 = ((nodes[None, :, :] - pts[:, None, :]) ** 2).sum(axis=2)
    return d2.argmin(axis=1).astype(np.int32)


class TestGridHashLonBranch:
    """Known bug (TODO 2026-05-09): meshes crossing lon=0 fragment in the
    [0, 360) normalization and the cell wrap misses the cell holding lon=0."""

    def test_query_west_of_prime_meridian(self):
        # Documented reproducer: 21x21 grid over [-10, 10]^2, query just
        # west of lon=0 must return the (lon=0, lat=0) node, not one 1° away.
        mesh = _grid_mesh(np.linspace(-10, 10, 21), np.linspace(-10, 10, 21))
        nn = find_nearest_nodes(mesh, np.array([-0.001]), np.array([0.0026]))
        expected = _nearest_bruteforce(mesh, [-0.001], [0.0026])
        assert nn[0] == expected[0]
        assert mesh.nodes_lon[nn[0]] == 0.0  # node 220, not 219 (lon=-1)

    def test_zero_crossing_parity_with_bruteforce(self):
        mesh = _grid_mesh(np.linspace(-10, 10, 21), np.linspace(-10, 10, 21))
        rng = np.random.default_rng(7)
        lons = rng.uniform(-9.9, 9.9, 200)
        lats = rng.uniform(-9.9, 9.9, 200)
        nn = find_nearest_nodes(mesh, lons, lats)
        np.testing.assert_array_equal(nn, _nearest_bruteforce(mesh, lons, lats))

    def test_dateline_mesh_parity_with_bruteforce(self):
        # Regional mesh crossing the dateline (signed lons fragment instead);
        # guards the [0, 360) branch that the current code handles correctly.
        xs = np.linspace(170.0, 190.0, 21)
        xs = (xs + 180.0) % 360.0 - 180.0  # -> [170..180) U [-180..-170]
        mesh = _grid_mesh(xs, np.linspace(-10, 10, 21))
        rng = np.random.default_rng(8)
        lons = (rng.uniform(170.1, 189.9, 200) + 180.0) % 360.0 - 180.0
        lats = rng.uniform(-9.9, 9.9, 200)
        nn = find_nearest_nodes(mesh, lons, lats)
        np.testing.assert_array_equal(nn, _nearest_bruteforce(mesh, lons, lats))

    def test_global_mesh_seam_parity_with_bruteforce(self):
        # Near-global mesh (wrap=True): queries straddling the 0/360 seam
        # must see nodes on both sides.  The index wrap ci2 +/- nx is only
        # exact when the grid spans exactly 360 deg (no lon padding).
        mesh = _grid_mesh(np.arange(0.0, 360.0, 2.0), np.linspace(-10, 10, 11))
        lons = np.array([-0.9, -0.3, 0.4, 359.1, 359.8, 358.6, 0.0, 179.3])
        lats = np.full(lons.shape, 0.7)
        nn = find_nearest_nodes(mesh, lons, lats)
        np.testing.assert_array_equal(nn, _nearest_bruteforce(mesh, lons, lats))

    def test_out_of_domain_query_folds_to_nearer_edge(self):
        # Pins the query-side fold (review finding: mutation-survivable).
        # Mesh [20, 40]: a query 5° WEST of the west edge must clamp to the
        # west edge (lon=20), not wrap around to the east edge (lon=40).
        mesh = _grid_mesh(np.linspace(20, 40, 21), np.linspace(-3, 3, 7))
        nn = find_nearest_nodes(mesh, np.array([15.0]), np.array([0.0]))[0]
        assert mesh.nodes_lon[nn] == 20.0
        nn = find_nearest_nodes(mesh, np.array([45.0]), np.array([0.0]))[0]
        assert mesh.nodes_lon[nn] == 40.0

    def test_prime_meridian_mesh_uses_signed_noncyclic_grid(self):
        # Pins the branch selection (review finding: with [0, 360) forced,
        # the fragmented span 359° flips wrap=True and the cyclic path
        # *also* passes the reproducer — so assert the build state itself).
        mesh = _grid_mesh(np.linspace(-10, 10, 21), np.linspace(-10, 10, 21))
        assert not mesh._gh_wrap_lon          # regional, not treated as global
        assert mesh._gh_lon_min < 0.0         # signed branch chosen
        assert mesh._gh_cell_size < 5.0       # cell size from 20° span, not 359°


# ---------------------------------------------------------------------------
# Boundary-edge topology / coastal element flags
# ---------------------------------------------------------------------------

def _extract_boundary_edges(mesh):
    """Mirror build_boundary_info: return count-1 (boundary) edges.

    Returns an (M, 2) int array of (min_node, max_node) pairs that belong
    to exactly one element — built over each element's COMPACTED valid
    vertices (closing edge included), same machinery as the implementation.
    """
    rows = []
    for e in range(mesh.n_elements):
        verts = mesh.elem_valid_nodes(e)
        nv = len(verts)
        if nv < 2:
            continue
        for j in range(nv):
            a = int(verts[j]); b = int(verts[(j + 1) % nv])
            if a == b:
                continue
            rows.append((a, b) if a < b else (b, a))
    arr = np.asarray(rows, dtype=np.int64)
    key = arr[:, 0] * np.int64(mesh.n_nodes + 1) + arr[:, 1]
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    return arr[cnt[inv] == 1]


class TestBoundaryTopology:
    def test_outer_ring_coastal_square_mesh(self):
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5))
        from kelp.core.mesh import build_boundary_info
        build_boundary_info(mesh)
        flags = mesh._elem_is_coastal
        assert flags.dtype == np.bool_ and flags.shape == (mesh.n_elements,)
        assert flags[0]                       # corner element
        cx, cy = mesh.elem_center_lon, mesh.elem_center_lat
        interior = np.argmin((cx - 2.0) ** 2 + (cy - 2.0) ** 2)
        assert not flags[interior]

    def test_island_hole_elements_coastal(self):
        # Remove the full 2x2 block of CELLS centred on (3, 3) — all 8
        # triangles of cells [2,4)x[2,4) — creating a genuine island hole
        # with a CLOSED 8-edge interior boundary loop.
        xs = np.linspace(0, 6, 7); ys = np.linspace(0, 6, 7)
        mesh_full = _grid_mesh(xs, ys)
        cx, cy = mesh_full.elem_center_lon, mesh_full.elem_center_lat
        keep = ~((cx > 2.0) & (cx < 4.0) & (cy > 2.0) & (cy < 4.0))
        assert (~keep).sum() == 8             # exactly the 8 hole triangles
        elem_nodes = mesh_full.elem_nodes[keep]
        n2e, n2e_count = build_node_to_elems(mesh_full.n_nodes, elem_nodes)
        mesh = Mesh(name="island", nodes_lon=mesh_full.nodes_lon,
                    nodes_lat=mesh_full.nodes_lat,
                    depth=np.full(mesh_full.n_nodes, 50.0),
                    elem_nodes=elem_nodes,
                    elem_center_lon=cx[keep], elem_center_lat=cy[keep],
                    node_to_elems=n2e, node_to_elems_count=n2e_count)
        from kelp.core.mesh import build_boundary_info
        build_boundary_info(mesh)
        # The hole rim must be ONE closed loop of exactly 8 boundary
        # edges that don't touch the outer rectangle:
        bedges = _extract_boundary_edges(mesh)
        on_rect = ((mesh.nodes_lon == 0.0) | (mesh.nodes_lon == 6.0)
                   | (mesh.nodes_lat == 0.0) | (mesh.nodes_lat == 6.0))
        both_on_rect = on_rect[bedges[:, 0]] & on_rect[bedges[:, 1]]
        hole_edges = bedges[~both_on_rect]
        assert hole_edges.shape[0] == 8       # exactly the hole rim
        # Necessary cycle condition: every rim node appears exactly
        # twice (the fixed 2x2-hole input makes this a single 8-cycle;
        # the assertion alone would also accept two disjoint 4-cycles).
        node_occ = np.bincount(hole_edges.ravel())
        assert np.all(node_occ[node_occ > 0] == 2)
        ring = np.argmin((mesh.elem_center_lon - 3.0) ** 2
                         + (mesh.elem_center_lat - 1.7) ** 2)
        assert mesh._elem_is_coastal[ring]          # hole-adjacent
        deep_int = np.argmin((mesh.elem_center_lon - 1.2) ** 2
                             + (mesh.elem_center_lat - 1.2) ** 2)
        assert not mesh._elem_is_coastal[deep_int]  # >1 cell from both rims

    def test_quad_and_padded_connectivity(self):
        mesh = _make_mixed_mesh()
        from kelp.core.mesh import build_boundary_info
        build_boundary_info(mesh)
        assert mesh._elem_is_coastal.all()    # tiny 2-element mesh: all coastal

    def test_padded_tri_shared_closing_edge_is_interior(self):
        # REVIEW-MANDATED: a -1-padded triangle whose CLOSING edge
        # (last-valid -> first-valid vertex) is shared with a neighbour
        # must count that edge as interior (count 2).  Naive vectorized
        # edge extraction over the padded width drops closing edges and
        # would wrongly flag both elements coastal THROUGH that edge.
        # Build 4 nodes / 2 triangles in FESOM-C 4-column padding:
        #   E0: [0, 1, 2, -1]  (closing edge 2-0)
        #   E1: [0, 2, 3, -1]  (shares edge 0-2 == E0's closing edge)
        nodes_lon = np.array([0.0, 1.0, 1.0, 0.0])
        nodes_lat = np.array([0.0, 0.0, 1.0, 1.0])
        elem_nodes = np.array([[0, 1, 2, -1], [0, 2, 3, -1]], dtype=np.int32)
        n2e, n2e_count = build_node_to_elems(4, elem_nodes)
        mesh = Mesh(name="pad", nodes_lon=nodes_lon, nodes_lat=nodes_lat,
                    depth=np.zeros(4), elem_nodes=elem_nodes,
                    elem_center_lon=np.array([2/3, 1/3]),
                    elem_center_lat=np.array([1/3, 2/3]),
                    node_to_elems=n2e, node_to_elems_count=n2e_count)
        from kelp.core.mesh import build_boundary_info
        build_boundary_info(mesh)
        # Both elements are coastal via their OUTER edges — but the shared
        # diagonal must be interior: verify by re-extracting edge counts
        # in-test and asserting edge (0,2) has count 2.
        rows = []
        for e in range(mesh.n_elements):
            verts = mesh.elem_valid_nodes(e)
            nv = len(verts)
            for j in range(nv):
                a = int(verts[j]); b = int(verts[(j + 1) % nv])
                if a == b:
                    continue
                rows.append((a, b) if a < b else (b, a))
        arr = np.asarray(rows, dtype=np.int64)
        key = arr[:, 0] * np.int64(mesh.n_nodes + 1) + arr[:, 1]
        uniq, cnt = np.unique(key, return_counts=True)
        shared_key = 0 * (mesh.n_nodes + 1) + 2      # edge (0, 2)
        assert cnt[np.searchsorted(uniq, shared_key)] == 2
        assert mesh._elem_is_coastal.all()           # both coastal via outer edges

    def test_idempotent_and_cached(self):
        mesh = _grid_mesh(np.linspace(0, 2, 3), np.linspace(0, 2, 3))
        from kelp.core.mesh import build_boundary_info
        build_boundary_info(mesh)
        first = mesh._elem_is_coastal
        build_boundary_info(mesh)
        assert mesh._elem_is_coastal is first

    def test_padded_grid_interior_not_coastal(self):
        # PRODUCTION-OUTPUT canary for the closing-edge-drop bug (spec
        # review round 3): convert a 5x5 grid mesh to FESOM-C 4-column
        # -1-padded connectivity.  Interior elements stay interior ONLY
        # if every padded triangle's closing edge is emitted — a naive
        # padded-slot roll drops them, leaks phantom boundary edges and
        # flips interior flags in mesh._elem_is_coastal itself.
        from kelp.core.mesh import build_boundary_info
        base = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5))
        E = base.n_elements
        padded = np.full((E, 4), -1, dtype=np.int32)
        padded[:, :3] = base.elem_nodes
        n2e, n2e_count = build_node_to_elems(base.n_nodes, padded)
        mesh = Mesh(name="padded_grid", nodes_lon=base.nodes_lon,
                    nodes_lat=base.nodes_lat, depth=base.depth,
                    elem_nodes=padded,
                    elem_center_lon=base.elem_center_lon,
                    elem_center_lat=base.elem_center_lat,
                    node_to_elems=n2e, node_to_elems_count=n2e_count)
        build_boundary_info(mesh)
        cx, cy = mesh.elem_center_lon, mesh.elem_center_lat
        interior = np.argmin((cx - 2.0) ** 2 + (cy - 2.0) ** 2)
        assert not mesh._elem_is_coastal[interior]
        assert mesh._elem_is_coastal[0]

    def test_degenerate_two_vertex_element_no_edge_corruption(self):
        # M2: a degenerate 2-vertex element [a, b, -1, -1] must NOT poison
        # the undirected-edge count.  Build a 6-triangle fan around centre 0
        # with ring nodes 1..6: every triangle [0, k, k+1] has its two RADIAL
        # edges (0,k) shared (interior) and ONE OUTER edge (k, k+1) on the
        # boundary — so each triangle's coastal flag hinges SOLELY on that
        # single outer edge.  Then add a degenerate element [1, 2, -1, -1]
        # whose two vertices are exactly triangle 0's outer edge (1,2).
        #   - With the fix (skip nv<3 + per-element dedup): the degenerate
        #     element is ignored, edge (1,2) stays count-1, triangle 0 coastal.
        #   - Without it: the degenerate element emits (1,2) AND (2,1) -> the
        #     (1,2) count jumps to 3, no longer 1, and triangle 0 — whose only
        #     boundary edge that was — is wrongly demoted to interior.
        from kelp.core.mesh import build_boundary_info
        ang = np.deg2rad(np.arange(6) * 60.0)
        ring_lon = np.cos(ang); ring_lat = np.sin(ang)
        nodes_lon = np.concatenate([[0.0], ring_lon])
        nodes_lat = np.concatenate([[0.0], ring_lat])
        tris = [[0, k, 1 + (k % 6)] for k in range(1, 7)]   # 6 fan triangles
        degen = [1, 2, -1, -1]                              # outer edge (1,2)
        rows = [t + [-1] for t in tris] + [degen]          # FESOM-C 4-column
        elem_nodes = np.array(rows, dtype=np.int32)
        n2e, n2e_count = build_node_to_elems(7, elem_nodes)
        cx = np.array([nodes_lon[[c for c in r if c >= 0]].mean()
                       for r in elem_nodes])
        cy = np.array([nodes_lat[[c for c in r if c >= 0]].mean()
                       for r in elem_nodes])
        mesh = Mesh(name="fan_degen", nodes_lon=nodes_lon, nodes_lat=nodes_lat,
                    depth=np.full(7, 50.0), elem_nodes=elem_nodes,
                    elem_center_lon=cx, elem_center_lat=cy,
                    node_to_elems=n2e, node_to_elems_count=n2e_count)
        build_boundary_info(mesh)
        # Triangle 0 (= [0,1,2]) keeps its boundary edge -> coastal.
        assert mesh._elem_is_coastal[0]
        # All 6 real fan triangles stay coastal (control).
        assert mesh._elem_is_coastal[:6].all()
        # The degenerate element owns no real ring -> not flagged coastal.
        assert not mesh._elem_is_coastal[6]


class TestGridHashSeamGap:
    """Known bug (review MINOR 2026-06-05, TODO 'grid-hash seam-gap'): on a
    near-global wrap=True mesh with an empty longitude gap wider than a
    cell, a SINGLE-kernel query inside the gap finds an empty 3x3 window
    and returns -1.  The vectorized ``find_nearest_nodes`` heals via the
    KD-tree fallback, but the fused/RK-stage kernels consume
    ``_grid_hash_nearest_widened`` directly — a stage probing across the
    gap failed for no physical reason.  Fixed by ring-widening on miss."""

    def _gap_mesh(self):
        # nodes span [0, 352] at 0.5 deg: hash cell ~ 1 deg, so the 8-deg
        # empty gap is far wider than the 3x3 window; span > 350 ->
        # cyclic (wrap=True) build
        return _grid_mesh(np.arange(0.0, 352.01, 0.5),
                          np.linspace(-5.0, 5.0, 21))

    def test_single_kernel_query_in_gap_finds_seam_node(self):
        from kelp.core.mesh import _grid_hash_nearest_widened
        mesh = self._gap_mesh()
        assert mesh._gh_wrap_lon and mesh._gh_cell_size < 2.0
        for lon_q in (353.5, 354.0, 355.0, 356.0, 357.0, 358.0):
            got = _grid_hash_nearest_widened(
                lon_q, 0.3,
                mesh._gh_offsets, mesh._gh_node_ids, mesh._gh_cell_size,
                mesh._gh_lon_min, mesh._gh_lat_min, mesh._gh_nx,
                mesh._gh_ny, mesh._gh_wrap_lon,
                mesh.nodes_lon, mesh.nodes_lat, mesh._node_xyz)
            expected = _nearest_bruteforce(mesh, [lon_q], [0.3])[0]
            assert got == expected, (lon_q, got, expected)

    def test_in_domain_queries_bitwise_unchanged(self):
        """The widening only fires on a miss: populated-window queries
        keep their exact pre-fix answers (parity with brute force)."""
        from kelp.core.mesh import _grid_hash_nearest_widened
        mesh = self._gap_mesh()
        rng = np.random.default_rng(11)
        lons = rng.uniform(0.1, 351.9, 300)
        lats = rng.uniform(-4.9, 4.9, 300)
        exp = _nearest_bruteforce(mesh, lons, lats)
        for i in range(lons.size):
            got = _grid_hash_nearest_widened(
                lons[i], lats[i],
                mesh._gh_offsets, mesh._gh_node_ids, mesh._gh_cell_size,
                mesh._gh_lon_min, mesh._gh_lat_min, mesh._gh_nx,
                mesh._gh_ny, mesh._gh_wrap_lon,
                mesh.nodes_lon, mesh.nodes_lat, mesh._node_xyz)
            assert got == exp[i], i

    def test_vectorized_path_still_kd_exact(self):
        mesh = self._gap_mesh()
        lons = np.array([354.0, 356.0, 357.9])
        lats = np.full(3, 0.3)
        nn = find_nearest_nodes(mesh, lons, lats)
        np.testing.assert_array_equal(nn, _nearest_bruteforce(mesh, lons, lats))

    def test_batch_miss_still_kd_heals_polar_case(self):
        """The vectorized path must keep its EXACT contract: batch misses
        stay -1 internally and find_nearest_nodes heals them with the
        KD-tree.  Review probe (review 2026-08-16): near the pole the
        ring-widened single kernel can pick a marginally farther node —
        the batch path must NOT inherit that approximation."""
        # two nodes near the pole at opposite longitudes; a tiny mesh
        # whose hash window misses for a query between them
        lons = np.array([0.0, 179.0, 1.0, 178.0])
        lats = np.array([89.6, 89.7, 20.0, 20.0])
        elem_nodes = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
        n2e, n2e_count = build_node_to_elems(4, elem_nodes)
        mesh = Mesh(name="polar", nodes_lon=lons, nodes_lat=lats,
                    depth=np.full(4, 10.0), elem_nodes=elem_nodes,
                    elem_center_lon=np.array([60.0, 119.3]),
                    elem_center_lat=np.array([66.4, 43.2]),
                    node_to_elems=n2e, node_to_elems_count=n2e_count)
        build_search(mesh)
        # A tiny cell size makes the 3x3 window at the query EMPTY —
        # without this the auto cell (~111 deg here) is populated and the
        # test never exercises the miss it documents (round-2 review).
        from kelp.core.mesh import (build_grid_hash,
                                        _grid_hash_nearest_3x3)
        build_grid_hash(mesh, cell_size=0.25)
        qlon, qlat = np.array([90.0]), np.array([89.9])
        assert _grid_hash_nearest_3x3(
            float(qlon[0]), float(qlat[0]),
            mesh._gh_offsets, mesh._gh_node_ids, mesh._gh_cell_size,
            mesh._gh_lon_min, mesh._gh_lat_min, mesh._gh_nx,
            mesh._gh_ny, mesh._gh_wrap_lon,
            mesh.nodes_lon, mesh.nodes_lat, mesh._node_xyz) == -1   # the miss REALLY fires
        nn = find_nearest_nodes(mesh, qlon, qlat)
        np.testing.assert_array_equal(
            nn, _nearest_bruteforce(mesh, qlon, qlat))

    def test_wrap_false_empty_window_heals_in_single_kernel(self):
        """Non-cyclic mesh with an interior two-cluster gap: the clamped
        window can be empty too; the widened single kernel must anchor
        to the nearer cluster and the batch path must stay KD-exact."""
        from kelp.core.mesh import (build_grid_hash,
                                        _grid_hash_nearest_3x3,
                                        _grid_hash_nearest_widened)
        xs = np.concatenate([np.arange(0.0, 5.1, 0.5),
                             np.arange(30.0, 35.1, 0.5)])
        mesh = _grid_mesh(xs, np.linspace(-2.0, 2.0, 9))
        build_grid_hash(mesh, cell_size=0.5)
        args = (mesh._gh_offsets, mesh._gh_node_ids, mesh._gh_cell_size,
                mesh._gh_lon_min, mesh._gh_lat_min, mesh._gh_nx,
                mesh._gh_ny, mesh._gh_wrap_lon,
                mesh.nodes_lon, mesh.nodes_lat, mesh._node_xyz)
        assert _grid_hash_nearest_3x3(17.0, 0.3, *args) == -1
        got = _grid_hash_nearest_widened(17.0, 0.3, *args)
        expected = _nearest_bruteforce(mesh, [17.0], [0.3])[0]
        assert got == expected
        nn = find_nearest_nodes(mesh, np.array([17.0]), np.array([0.3]))
        np.testing.assert_array_equal(
            nn, _nearest_bruteforce(mesh, [17.0], [0.3]))

    def test_fused_stage_across_gap_survives(self):
        """e2e: an rk2 midpoint stage probing INTO the empty gap used to
        get nn=-1 and kill the particle; with the widened single kernel
        it anchors to the seam edge and the particle lives.  The flow is
        +5 m/s now, -5 m/s at the next snapshot, so the midpoint stage
        sits ~2.2 deg inside the gap while the NET displacement is ~0
        (the particle itself never leaves the mesh)."""
        from kelp.core.io import VelocityField
        from kelp.core.particles import ParticleStatus, make_state
        from kelp.numerics.integrators import rk2_step
        mesh = self._gap_mesh()
        ne, nn_ = mesh.n_elements, mesh.n_nodes
        vel_a = VelocityField(u=np.full(ne, 5.0), v=np.zeros(ne),
                              eta=np.zeros(nn_))
        vel_b = VelocityField(u=np.full(ne, -5.0), v=np.zeros(ne),
                              eta=np.zeros(nn_))
        dt = 97000.0            # midpoint ~ 351.8 + 2.2 = 354 (gap)
        st = make_state(np.array([351.8]), np.array([0.3]))
        from kelp.core.mesh import find_nearest_nodes as _fnn
        st = st.replace(nearest_node=_fnn(mesh, st.lon, st.lat))
        out = rk2_step(st, vel_a, mesh, dt, 1.0,
                       vel_next=vel_b, dt_to_next=dt)
        assert out.status[0] == ParticleStatus.ALIVE
        dlon = ((out.lon[0] - 351.8 + 180.0) % 360.0) - 180.0
        assert abs(dlon) < 0.2                    # net drift ~ 0 (wrapped)

    def test_walk_fallback_verdicts_match_local_exactly(self):
        """The walk contract: fallback verdicts identical to
        find_containing_elements (KD-exact anchors).  The in-loop
        fallback anchors on the APPROXIMATE widened nearest — the
        wrapper must re-heal the not-contained remainder through the
        exact machinery (round-2 review blocker)."""
        from kelp.core.mesh import (find_containing_elements,
                                        find_containing_elements_walk)
        mesh = self._gap_mesh()
        rng = np.random.default_rng(3)
        lons = np.concatenate([rng.uniform(0.5, 351.5, 100),
                               np.array([353.5, 355.0, 357.0])])  # gap
        lats = np.concatenate([rng.uniform(-4.5, 4.5, 100),
                               np.full(3, 0.3)])
        starts = np.full(lons.size, -1, dtype=np.int32)  # force fallback
        e_walk, ok_walk = find_containing_elements_walk(
            mesh, lons, lats, starts)
        nn = find_nearest_nodes(mesh, lons, lats)
        e_loc, ok_loc = find_containing_elements(mesh, lons, lats, nn)
        np.testing.assert_array_equal(ok_walk, ok_loc)
        np.testing.assert_array_equal(e_walk[ok_loc], e_loc[ok_loc])

    def test_walk_heal_flips_verdict_on_anisotropic_miss(self):
        """Non-vacuous heal pin (round-3 review): geometry where the
        in-loop WIDENED anchor picks a ring-order decoy whose element
        ring does NOT contain the query, while the KD-exact anchor's
        ring DOES — only the wrapper heal makes the walk verdict match
        the local machinery.  Anisotropy at lat 71: the containing
        triangle's vertices sit 2 deg away in LONGITUDE (chord-cheap,
        ring-expensive: cos71 ~ 0.33), the decoy 1 deg away in LATITUDE
        (ring-cheap, chord-expensive), and the query's own 3x3 window is
        empty at cell_size=0.25."""
        from kelp.core.mesh import (build_grid_hash,
                                        find_containing_elements,
                                        find_containing_elements_walk,
                                        _grid_hash_nearest_3x3,
                                        _grid_hash_nearest_widened)
        lons = np.array([1.0, 5.0, 3.0, 3.0, 3.3, 3.15])
        lats = np.array([71.0, 71.0, 69.5, 72.0, 72.0, 72.25])
        elem_nodes = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        n2e, cnt = build_node_to_elems(6, elem_nodes)
        mesh = Mesh(name="heal", nodes_lon=lons, nodes_lat=lats,
                    depth=np.full(6, 10.0), elem_nodes=elem_nodes,
                    elem_center_lon=np.array([3.0, 3.15]),
                    elem_center_lat=np.array([70.5, 72.08]),
                    node_to_elems=n2e, node_to_elems_count=cnt)
        build_search(mesh)
        build_grid_hash(mesh, cell_size=0.25)
        q = np.array([3.0]), np.array([70.95])
        args = (mesh._gh_offsets, mesh._gh_node_ids, mesh._gh_cell_size,
                mesh._gh_lon_min, mesh._gh_lat_min, mesh._gh_nx,
                mesh._gh_ny, mesh._gh_wrap_lon,
                mesh.nodes_lon, mesh.nodes_lat, mesh._node_xyz)
        # the trap really arms: empty window, widened anchor = the decoy
        assert _grid_hash_nearest_3x3(3.0, 70.95, *args) == -1
        wid = _grid_hash_nearest_widened(3.0, 70.95, *args)
        assert wid == 3
        _, ok_wid = find_containing_elements(
            mesh, q[0], q[1], np.array([wid], dtype=np.int32))
        assert not ok_wid[0]                      # decoy ring: no element
        # the walk (forced to the fallback) must still match the exact
        # local machinery thanks to the wrapper heal
        e_walk, ok_walk = find_containing_elements_walk(
            mesh, q[0], q[1], np.array([-1], dtype=np.int32))
        nn = find_nearest_nodes(mesh, q[0], q[1])
        e_loc, ok_loc = find_containing_elements(mesh, q[0], q[1], nn)
        assert ok_loc[0] and ok_walk[0]
        assert e_walk[0] == e_loc[0] == 0
