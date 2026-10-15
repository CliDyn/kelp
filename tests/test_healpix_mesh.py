"""HEALPix dual-mesh builder: invariants, cross-checks, machinery integration.

Phase 1 of the HEALPix support (docs/healpix.md).  The all-wet globe obeys exact combinatorial
laws that pin the whole construction:

- every pixel-corner point is shared by 4 pixels except the 8 special
  vertices of the base tessellation (valence 3) -> exactly **8 triangles**
  at any nside, and Euler's formula forces ``E = npix + 2`` elements;
- the dual cells partition the sphere -> spherical-excess areas sum to
  exactly ``4 * pi * R**2`` and there are **zero boundary edges**;
- every pixel touches exactly its own 4 corner cells -> node valence 4.

Cross-checks of the dual-mesh construction: the neighbour-label corner grouping is
verified geometrically against ``healpy.boundaries`` corner coordinates,
and the edge set against the spherical Delaunay triangulation (convex hull
of the unit-sphere centres).
"""
import sys
from functools import lru_cache

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")

from kelp.core.geometry import METERS_PER_DEG_LAT, delta_lon
from kelp.core.io import VelocityField
from kelp.core.mesh import (build_boundary_info, build_elem_areas,
                                build_elem_neighbors,
                                find_containing_elements, find_nearest_nodes,
                                find_nonconvex_quads)
from kelp.core.particles import ParticleStatus, make_state
from kelp.readers.healpix import build_healpix_dual_mesh


@lru_cache(maxsize=None)
def _globe(nside):
    """All-wet build, cached across tests (lazy caches are idempotent)."""
    return build_healpix_dual_mesh(nside)


def _edge_set(mesh):
    """Unique undirected edges over the compacted element rings."""
    edges = set()
    for e in range(mesh.n_elements):
        verts = mesh.elem_valid_nodes(e)
        nv = len(verts)
        for j in range(nv):
            a, b = int(verts[j]), int(verts[(j + 1) % nv])
            edges.add((min(a, b), max(a, b)))
    return edges


# ---------------------------------------------------------------------------
# All-wet globe invariants
# ---------------------------------------------------------------------------

ALL_WET_NSIDES = [1, 2, 4, 8, 16]


class TestAllWetInvariants:
    @pytest.mark.parametrize("nside", ALL_WET_NSIDES)
    def test_counts(self, nside):
        mesh = _globe(nside)
        npix = 12 * nside * nside
        n_tris = int((mesh.elem_nodes[:, 3] == -1).sum())
        assert mesh.n_nodes == npix                 # no orphans on a globe
        assert mesh.n_elements == npix + 2          # Euler: V - E + F = 2
        assert n_tris == 8                          # the 8 valence-3 corners
        assert mesh.metadata["n_triangles"] == 8
        assert mesh.metadata["n_quads"] == npix - 6
        assert mesh.metadata["healpix_nside"] == nside
        assert mesh.metadata["healpix_order"] == "nest"

    @pytest.mark.parametrize("nside", ALL_WET_NSIDES)
    def test_node_valence_is_four(self, nside):
        # each pixel belongs to exactly its own 4 corner cells
        mesh = _globe(nside)
        np.testing.assert_array_equal(mesh.node_to_elems_count, 4)

    @pytest.mark.parametrize("nside", ALL_WET_NSIDES)
    def test_closed_surface(self, nside):
        mesh = _globe(nside)
        build_boundary_info(mesh)
        assert not mesh._elem_is_coastal.any()      # zero boundary edges
        nb = build_elem_neighbors(mesh)             # raises if non-manifold
        quad = mesh.elem_nodes[:, 3] >= 0
        assert (nb[quad] >= 0).all()                # every quad edge paired
        assert (nb[~quad][:, :3] >= 0).all()        # every tri edge paired

    @pytest.mark.parametrize("nside", ALL_WET_NSIDES)
    def test_euler_characteristic(self, nside):
        mesh = _globe(nside)
        assert mesh.n_nodes - len(_edge_set(mesh)) + mesh.n_elements == 2

    @pytest.mark.parametrize("nside", ALL_WET_NSIDES)
    def test_area_partition(self, nside):
        mesh = _globe(nside)
        areas = build_elem_areas(mesh)
        sphere = 4.0 * np.pi * mesh.r_earth ** 2
        assert np.isclose(areas.sum(), sphere, rtol=1e-9)
        # Dual cells are NOT equal-area (the pixels are): measured spread
        # is min/mean ~0.49-0.61 and max/mean ~1.27-1.76 (widest at
        # nside=1), stable in nside.  Pin generous envelopes around that.
        mean = sphere / mesh.n_elements
        assert areas.min() > 0.45 * mean
        assert areas.max() < 1.85 * mean

    @pytest.mark.parametrize("nside", ALL_WET_NSIDES)
    def test_convexity_contract(self, nside):
        # the builder already validates at load; belt and braces
        mesh = _globe(nside)
        assert find_nonconvex_quads(mesh.elem_nodes, mesh.nodes_lon,
                                    mesh.nodes_lat).size == 0

    @pytest.mark.parametrize("nside", [4, 8])
    def test_consistent_ccw_winding(self, nside):
        # (owner, edge-nb, diag, edge-nb) is CCW viewed from outside the
        # sphere -- verified globally (incl. the southern cap, where the
        # compass reasoning is least obvious).  Projected onto the 3D
        # tangent plane at each element's centroid (the lon/lat chart is
        # degenerate for the two POLE quads: their 4 vertices sit on one
        # latitude ring -> collinear in the chart, zero shoelace).
        mesh = _globe(nside)
        z_hat = np.array([0.0, 0.0, 1.0])
        x_hat = np.array([1.0, 0.0, 0.0])
        for e in range(mesh.n_elements):
            verts = mesh.elem_valid_nodes(e)
            lon = np.radians(mesh.nodes_lon[verts])
            lat = np.radians(mesh.nodes_lat[verts])
            v3 = np.column_stack((np.cos(lat) * np.cos(lon),
                                  np.cos(lat) * np.sin(lon), np.sin(lat)))
            c = v3.mean(axis=0)
            c /= np.linalg.norm(c)
            e1 = np.cross(z_hat, c)
            if np.linalg.norm(e1) < 1e-9:           # polar centroid
                e1 = np.cross(x_hat, c)
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(c, e1)                    # (e1, e2, c) right-handed
            x_n, y_n = v3 @ e1, v3 @ e2
            s = 0.0
            nv = len(verts)
            for k in range(nv):
                j = (k + 1) % nv
                s += x_n[k] * y_n[j] - x_n[j] * y_n[k]
            assert s > 0.0                          # CCW seen from outside

    @pytest.mark.parametrize("nside", [2, 8])
    def test_node_pix_mapping(self, nside):
        mesh = _globe(nside)
        pix = mesh._healpix_node_pix
        assert pix.dtype == np.int64
        assert (np.diff(pix) > 0).all()             # sorted, unique
        np.testing.assert_array_equal(pix, np.arange(12 * nside * nside))
        lon, lat = healpy.pix2ang(nside, pix, nest=True, lonlat=True)
        lon = ((lon + 180.0) % 360.0) - 180.0
        np.testing.assert_allclose(mesh.nodes_lon, lon, atol=1e-12)
        np.testing.assert_allclose(mesh.nodes_lat, lat, atol=1e-12)
        assert mesh._healpix_nside == nside


# ---------------------------------------------------------------------------
# Geometric cross-checks (corner grouping, Delaunay)
# ---------------------------------------------------------------------------

class TestCornerGeometry:
    def test_members_share_a_geometric_corner(self):
        # The neighbour-label construction never touches corner
        # coordinates; verify against healpy.boundaries that the member
        # pixels of every element really do meet at one corner point.
        nside = 8
        mesh = _globe(nside)
        node_pix = mesh._healpix_node_pix
        tol = 0.02 * healpy.nside2resol(nside)      # << half pixel spacing
        for e in range(mesh.n_elements):
            verts = mesh.elem_valid_nodes(e)
            corners = [healpy.boundaries(nside, int(node_pix[v]), step=1,
                                         nest=True).T          # (4, 3) xyz
                       for v in verts]
            shared = False
            for cand in corners[0]:
                if all(np.linalg.norm(c - cand, axis=1).min() < tol
                       for c in corners[1:]):
                    shared = True
                    break
            assert shared, f"element {e}: members do not meet at a corner"

    @pytest.mark.parametrize("nside", [2, 4])
    def test_delaunay_cross_check(self, nside):
        # Spherical Delaunay = convex hull of the unit-sphere centres.
        # The dual-mesh edge set must be a subset of the hull edge set,
        # and the hull's extra edges must be exactly one diagonal per
        # dual quad: E_hull = 3*npix - 6 = E_dual (2*npix) + n_quads.
        from scipy.spatial import ConvexHull
        mesh = _globe(nside)
        xyz = np.column_stack(
            healpy.pix2vec(nside, mesh._healpix_node_pix, nest=True))
        hull = ConvexHull(xyz)
        hull_edges = set()
        for tri in hull.simplices:
            for i in range(3):
                a, b = int(tri[i]), int(tri[(i + 1) % 3])
                hull_edges.add((min(a, b), max(a, b)))
        dual_edges = _edge_set(mesh)
        assert dual_edges <= hull_edges
        extra = hull_edges - dual_edges
        quads = mesh.elem_nodes[mesh.elem_nodes[:, 3] >= 0]
        diagonals = set()
        for q in quads:
            diagonals.add((min(int(q[0]), int(q[2])),
                           max(int(q[0]), int(q[2]))))
            diagonals.add((min(int(q[1]), int(q[3])),
                           max(int(q[1]), int(q[3]))))
        assert extra <= diagonals
        assert len(extra) == len(quads)


# ---------------------------------------------------------------------------
# Wet mask: coastline, orphans, failure modes
# ---------------------------------------------------------------------------

class TestWetMask:
    def test_polar_cap_coastline(self):
        nside = 8
        npix = 12 * nside * nside
        _, lat = healpy.pix2ang(nside, np.arange(npix), nest=True,
                                lonlat=True)
        wet = lat < 60.0                            # dry north polar cap
        mesh = build_healpix_dual_mesh(nside, wet_mask=wet)
        assert (mesh.nodes_lat < 60.0).all()
        build_boundary_info(mesh)
        assert mesh._elem_is_coastal.any()
        build_elem_neighbors(mesh)                  # manifold
        # sphere minus a disk = disk: Euler characteristic 1
        assert mesh.n_nodes - len(_edge_set(mesh)) + mesh.n_elements == 1
        # every kept node's own centre is locatable (vertex containment)
        nn = find_nearest_nodes(mesh, mesh.nodes_lon, mesh.nodes_lat)
        _, in_dom = find_containing_elements(
            mesh, mesh.nodes_lon, mesh.nodes_lat, nn)
        assert in_dom.all()
        # dry-side points are not
        rng = np.random.default_rng(3)
        lons_dry = rng.uniform(-180, 180, 64)
        lats_dry = rng.uniform(66.0, 84.0, 64)
        nn = find_nearest_nodes(mesh, lons_dry, lats_dry)
        _, in_dom = find_containing_elements(mesh, lons_dry, lats_dry, nn)
        assert not in_dom.any()

    def test_orphan_wet_pixel_dropped(self):
        # a wet pixel whose 8 neighbours are all dry has no corner cell:
        # it must be dropped from the nodes (dead-end otherwise)
        nside = 8
        npix = 12 * nside * nside
        p0 = int(healpy.ang2pix(nside, 0.0, 0.0, nest=True, lonlat=True))
        ring = healpy.get_all_neighbours(nside, p0, nest=True)
        assert (ring >= 0).all()                    # equatorial: all 8 exist
        wet = np.ones(npix, dtype=bool)
        wet[ring] = False
        mesh = build_healpix_dual_mesh(nside, wet_mask=wet)
        assert mesh.n_nodes == npix - 9             # 8 dry + 1 orphan
        assert p0 not in mesh._healpix_node_pix
        lon0, lat0 = healpy.pix2ang(nside, p0, nest=True, lonlat=True)
        lon0 = ((lon0 + 180.0) % 360.0) - 180.0
        nn = find_nearest_nodes(mesh, np.array([lon0]), np.array([lat0]))
        _, in_dom = find_containing_elements(
            mesh, np.array([lon0]), np.array([lat0]), nn)
        assert not in_dom.any()

    def test_all_dry_raises(self):
        with pytest.raises(ValueError, match="no ocean"):
            build_healpix_dual_mesh(4, wet_mask=np.zeros(192, dtype=bool))

    def test_isolated_wet_pixel_raises(self):
        wet = np.zeros(768, dtype=bool)
        wet[100] = True
        with pytest.raises(ValueError, match="no elements"):
            build_healpix_dual_mesh(8, wet_mask=wet)

    def test_invalid_nside_raises(self):
        for bad in (0, 3, -8):
            with pytest.raises(ValueError, match="nside"):
                build_healpix_dual_mesh(bad)

    def test_int32_capacity_guard(self):
        # nside 16384 -> npix = 3.2e9 > int32: must refuse loudly (and
        # BEFORE allocating npix-sized arrays — this test would OOM
        # otherwise)
        with pytest.raises(ValueError, match="int32"):
            build_healpix_dual_mesh(16384)

    def test_bad_order_raises(self):
        with pytest.raises(ValueError, match="order"):
            build_healpix_dual_mesh(2, order="rings")

    def test_bad_shapes_raise(self):
        with pytest.raises(ValueError, match="wet_mask"):
            build_healpix_dual_mesh(2, wet_mask=np.ones(47, dtype=bool))
        with pytest.raises(ValueError, match="depth"):
            build_healpix_dual_mesh(2, depth=np.ones(47))


# ---------------------------------------------------------------------------
# Determinism: ring == nest, block-size invariance
# ---------------------------------------------------------------------------

def _assert_same_mesh(a, b):
    np.testing.assert_array_equal(a.elem_nodes, b.elem_nodes)
    np.testing.assert_array_equal(a._healpix_node_pix, b._healpix_node_pix)
    np.testing.assert_array_equal(a.nodes_lon, b.nodes_lon)
    np.testing.assert_array_equal(a.nodes_lat, b.nodes_lat)
    np.testing.assert_array_equal(a.depth, b.depth)


class TestDeterminism:
    def test_ring_input_equals_nest_input(self):
        nside = 4
        npix = 12 * nside * nside
        rng = np.random.default_rng(42)
        wet_nest = rng.random(npix) < 0.7
        depth_nest = rng.uniform(5.0, 500.0, npix)
        to_nest = healpy.ring2nest(nside, np.arange(npix))
        wet_ring = wet_nest[to_nest]        # value_ring[r] = value_nest[n(r)]
        depth_ring = depth_nest[to_nest]
        a = build_healpix_dual_mesh(nside, wet_mask=wet_nest,
                                    depth=depth_nest, order="nest")
        b = build_healpix_dual_mesh(nside, wet_mask=wet_ring,
                                    depth=depth_ring, order="ring")
        _assert_same_mesh(a, b)

    def test_block_size_invariance(self):
        nside = 8
        npix = 12 * nside * nside
        rng = np.random.default_rng(1)
        wet = rng.random(npix) < 0.8
        a = build_healpix_dual_mesh(nside, wet_mask=wet)
        b = build_healpix_dual_mesh(nside, wet_mask=wet, block_size=64)
        _assert_same_mesh(a, b)


# ---------------------------------------------------------------------------
# Search machinery integration: coverage, dateline, the polar gap
# ---------------------------------------------------------------------------

class TestContainmentCoverage:
    def test_global_coverage_below_polar_ring(self):
        # all-wet globe: every point below the first pixel ring must land
        # in exactly one element (incl. dateline and prime meridian)
        mesh = _globe(8)
        rng = np.random.default_rng(7)
        z = rng.uniform(-np.sin(np.radians(80.0)), np.sin(np.radians(80.0)),
                        2000)
        lats = np.degrees(np.arcsin(z))
        lons = rng.uniform(-180.0, 180.0, 2000)
        # high-latitude band up to just below the first pixel ring
        # (84.15 deg at nside 8) — the reachable part of the polar caps
        band_lat = rng.uniform(80.0, 83.5, 100) * rng.choice([-1, 1], 100)
        band_lon = rng.uniform(-180.0, 180.0, 100)
        lons = np.concatenate([lons, band_lon,
                               [179.99, -179.99, 180.0, 0.0, -0.01]])
        lats = np.concatenate([lats, band_lat,
                               [10.0, -35.0, 0.5, 66.0, -66.0]])
        nn = find_nearest_nodes(mesh, lons, lats)
        _, in_dom = find_containing_elements(mesh, lons, lats, nn)
        assert in_dom.all()

    def test_polar_grid_hash_miss_falls_back_to_kdtree(self):
        # At nside >= 32 the polar-ring longitude gaps dwarf the grid-hash
        # cell size, so a cold nearest-node query near the pole can find
        # an EMPTY 3x3 window.  Before the KD-tree fallback in
        # find_nearest_nodes this returned -1 (~40% of queries at lat
        # 89.5, nside 32) — an invalid anchor that wraps Python-style
        # into node_to_elems[-1] downstream.  Pin the fixed contract:
        # anchors are always valid node ids.
        mesh = _globe(32)
        rng = np.random.default_rng(11)
        lons = rng.uniform(-180.0, 180.0, 1500)
        lats = rng.uniform(85.0, 89.9, 1500) * rng.choice([-1, 1], 1500)
        nn = find_nearest_nodes(mesh, lons, lats)
        assert (nn >= 0).all()

    def test_polar_gap_is_pinned(self):
        # KNOWN LIMITATION (docs/healpix.md §3): the polar cap quad's
        # vertices top out at the first-ring latitude (~84.15 deg at
        # nside 8), and the lon/lat containment chart rejects points
        # poleward of its bbox — the pole region is geometrically covered
        # by the mesh but not reachable by the current search machinery.
        # The real fix is the deferred rotated-frame generalization; if
        # this assert ever flips, that gap has been closed — update the
        # plan and the reader docs.
        mesh = _globe(8)
        lons = np.array([15.0])
        lats = np.array([89.5])
        nn = find_nearest_nodes(mesh, lons, lats)
        _, in_dom = find_containing_elements(mesh, lons, lats, nn)
        assert not in_dom.any()


# ---------------------------------------------------------------------------
# Kinematics on the dual mesh (house style: constant velocity
# -> analytic displacement; node layout + barycentric = the HEALPix recipe)
# ---------------------------------------------------------------------------

class TestAdvectionOnDualMesh:
    def _step_n(self, mesh, vf, lons, lats, n_steps, dt):
        from kelp.numerics.integrators import euler_step
        st = make_state(lons.copy(), lats.copy())
        nn = find_nearest_nodes(mesh, st.lon, st.lat)
        st = st.replace(nearest_node=nn)
        for _ in range(n_steps):
            st = euler_step(st, vf, mesh, dt, 1.0, uv_interp="barycentric")
        return st

    def test_constant_zonal_flow_analytic(self):
        mesh = _globe(16)
        n = mesh.n_nodes
        u0, dt, n_steps = 1.0, 3600.0, 24
        vf = VelocityField(u=np.full(n, u0), v=np.zeros(n),
                           eta=np.zeros(n), w=None, uv_layout="node")
        lons = np.array([0.0, 100.0, 179.95, -45.0])   # incl. dateline crosser
        lats = np.array([0.0, 30.0, -20.0, 55.0])
        st = self._step_n(mesh, vf, lons, lats, n_steps, dt)
        assert (st.status == ParticleStatus.ALIVE).all()
        # v == 0 -> lat frozen -> each Euler step adds the same exact dlon
        dlon = n_steps * u0 * dt / (np.cos(np.radians(lats))
                                    * METERS_PER_DEG_LAT)
        np.testing.assert_allclose(st.lat, lats, atol=1e-12)
        np.testing.assert_allclose(delta_lon(st.lon, lons), dlon, atol=1e-9)
        assert st.lon[2] < -179.0                      # wrapped the dateline

    def test_constant_meridional_flow_analytic(self):
        mesh = _globe(16)
        n = mesh.n_nodes
        v0, dt, n_steps = 0.5, 3600.0, 24
        vf = VelocityField(u=np.zeros(n), v=np.full(n, v0),
                           eta=np.zeros(n), w=None, uv_layout="node")
        lons = np.array([10.0, -170.0, 120.0])
        lats = np.array([-40.0, 5.0, 33.0])
        st = self._step_n(mesh, vf, lons, lats, n_steps, dt)
        assert (st.status == ParticleStatus.ALIVE).all()
        dlat = n_steps * v0 * dt / METERS_PER_DEG_LAT
        np.testing.assert_allclose(st.lon, lons, atol=1e-12)  # u == 0
        np.testing.assert_allclose(st.lat, lats + dlat, atol=1e-9)


# ---------------------------------------------------------------------------
# Config-surface plumbing + lazy healpy
# ---------------------------------------------------------------------------

class TestPlumbing:
    def test_depth_z_levels_r_earth_name(self):
        nside = 2
        npix = 48
        depth = -np.arange(npix, dtype=np.float64)     # negative -> abs
        mesh = build_healpix_dual_mesh(nside, depth=depth,
                                       z_levels=[0.0, -10.0, 25.0],
                                       name="hpx-test", r_earth=6_400_000.0)
        assert mesh.name == "hpx-test"
        assert mesh.r_earth == 6_400_000.0
        assert mesh.vertical_coord == "zlevel"
        np.testing.assert_array_equal(mesh.sigma, [0.0, 10.0, 25.0])
        np.testing.assert_array_equal(
            mesh.depth, np.arange(npix, dtype=np.float64)[
                mesh._healpix_node_pix])

    def test_missing_healpy_error(self, monkeypatch):
        import kelp.readers.healpix as hx
        monkeypatch.setitem(sys.modules, "healpy", None)
        with pytest.raises(ImportError, match="healpy"):
            hx.build_healpix_dual_mesh(2)
