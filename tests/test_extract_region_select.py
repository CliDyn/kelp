"""utils/extract_fesom_region.py: which elements a region selects.

``select_region`` once took every element whose bounding box touched the
region's bounding box (a circle came out as its square), and it built the
element bounding boxes from raw longitudes, so a triangle across the +-180
seam spanned 360 degrees and joined any region in its latitude band.  An
element is now selected when it intersects the closed region (bbox and
polygon in the lon/lat plane, the circle on the sphere), with each
triangle's longitudes unwrapped across the seam.  The circle's candidate box
is its spherical cap: all longitudes once the cap reaches a pole.
"""
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

from tests._fesom2_synth import grid_geometry, write_meshdiag

_EXT_PATH = pathlib.Path(__file__).resolve().parents[1] / "utils" / "extract_fesom_region.py"
_spec = importlib.util.spec_from_file_location("extract_fesom_region", _EXT_PATH)
extract_fesom_region = importlib.util.module_from_spec(_spec)
sys.modules["extract_fesom_region"] = extract_fesom_region
_spec.loader.exec_module(extract_fesom_region)
RegionSpec = extract_fesom_region.RegionSpec


def _wrap(lon):
    return (np.asarray(lon) + 180.0) % 360.0 - 180.0


def _mesh(tmp_path, *patches):
    """One mesh of disjoint regular patches ``(nx, ny, lon0, lat0, step)``,
    longitudes wrapped to [-180, 180) as a global FESOM2 mesh stores them."""
    lons, lats, tris, offset = [], [], [], 0
    for nx, ny, lon0, lat0, step in patches:
        lon, lat, tri = grid_geometry(nx, ny, lon0, lat0, step)
        lons.append(_wrap(lon))
        lats.append(lat)
        tris.append(tri + offset)
        offset += lon.size
    geometry = (np.concatenate(lons), np.concatenate(lats), np.vstack(tris))
    path = write_meshdiag(tmp_path / "mesh.diag.nc", "comp", geometry=geometry)
    return extract_fesom_region.load_mesh_source(str(path))


def _haversine_km(lon, lat, lon0, lat0):
    return extract_fesom_region._haversine_km(np.asarray(lon, float),
                                              np.asarray(lat, float), lon0, lat0)


def _in_circle(mesh, center, radius_km):
    """Elements with a vertex or the centre within the circle (brute force)."""
    tri_lon, tri_lat = mesh.lon[mesh.elem_nodes], mesh.lat[mesh.elem_nodes]
    vertex = (_haversine_km(tri_lon.ravel(), tri_lat.ravel(), *center)
              <= radius_km).reshape(-1, 3).any(axis=1)
    centre = _haversine_km(mesh.elem_center_lon, mesh.elem_center_lat,
                           *center) <= radius_km
    return vertex | centre


def test_center_radius_keeps_the_circle_and_drops_the_square_corners(tmp_path):
    mesh = _mesh(tmp_path, (21, 21, -10.0, -10.0, 1.0))
    spec = RegionSpec(mode="center_radius", center=(0.0, 0.0), radius_km=300.0)
    mask = extract_fesom_region.select_region(mesh, spec)
    np.testing.assert_array_equal(mask, _in_circle(mesh, (0.0, 0.0), 300.0))
    # the corner cell of the circle's bounding square, (2..3, 2..3): its
    # nearest vertex (2, 2) is 314 km from the centre
    corner = np.flatnonzero(
        (mesh.lon[mesh.elem_nodes].min(axis=1) == 2.0)
        & (mesh.lat[mesh.elem_nodes].min(axis=1) == 2.0))
    assert corner.size == 2
    assert not mask[corner].any()
    assert mask.any()


def _seam_elements(mesh):
    tri_lon = mesh.lon[mesh.elem_nodes]
    return (tri_lon.max(axis=1) - tri_lon.min(axis=1)) > 180.0


def _write_polygon(tmp_path, lon, lat):
    ring = [[float(x), float(y)] for x, y in zip(lon, lat)]
    path = tmp_path / "region.geojson"
    path.write_text(json.dumps({"type": "Polygon", "coordinates": [ring + ring[:1]]}))
    return str(path)


@pytest.mark.parametrize("mode", ["bbox", "center_radius", "polygon"])
def test_seam_triangle_not_selected_for_a_region_far_from_the_seam(tmp_path, mode):
    # a patch around 5E and a patch across the dateline, same latitudes
    mesh = _mesh(tmp_path, (11, 7, 0.0, -3.0, 1.0), (11, 7, 174.5, -3.0, 1.0))
    seam = _seam_elements(mesh)
    assert seam.sum() == 12                          # the 179.5 / -179.5 column
    if mode == "bbox":
        spec = RegionSpec(mode="bbox", bbox=(2.0, 8.0, -2.0, 2.0))
    elif mode == "center_radius":
        spec = RegionSpec(mode="center_radius", center=(5.0, 0.0), radius_km=250.0)
    else:
        spec = RegionSpec(mode="polygon", polygon_path=_write_polygon(
            tmp_path, [2.2, 7.7, 7.9, 2.4], [-1.8, -2.1, 1.9, 2.2]))
    mask = extract_fesom_region.select_region(mesh, spec)
    assert not (mask & seam).any()
    patch_far = np.abs(mesh.elem_center_lon) > 100.0
    assert not (mask & patch_far).any()
    assert mask.any()


def test_circle_on_the_seam_selects_both_sides(tmp_path):
    mesh = _mesh(tmp_path, (11, 7, 174.5, -3.0, 1.0))
    seam = _seam_elements(mesh)
    spec = RegionSpec(mode="center_radius", center=(180.0, 0.0), radius_km=100.0)
    mask = extract_fesom_region.select_region(mesh, spec)
    np.testing.assert_array_equal(mask, _in_circle(mesh, (180.0, 0.0), 100.0))
    tri_lon = mesh.lon[mesh.elem_nodes]
    east = (tri_lon < 0).all(axis=1)                 # -179.5 .. -178.5
    west = (tri_lon > 0).all(axis=1)                 #  178.5 ..  179.5
    assert (mask & seam).any() and (mask & east).any() and (mask & west).any()


def test_bbox_on_the_seam_selects_the_seam_triangles(tmp_path):
    """bboxes on either side of the seam take the seam triangles they reach,
    and only those: a box reaching into one seam triangle leaves the other
    triangle of its cell out (the old superset took every seam triangle
    whose 360-degree raw box met the region, both triangles here)."""
    mesh = _mesh(tmp_path, (11, 7, 174.5, -3.0, 1.0))
    seam = _seam_elements(mesh)
    for bbox in ((178.0, 180.0, -1.0, 1.0), (-180.0, -178.0, -1.0, 1.0)):
        mask = extract_fesom_region.select_region(
            mesh, RegionSpec(mode="bbox", bbox=bbox))
        tri_lat = mesh.lat[mesh.elem_nodes]
        expected = seam & (tri_lat.min(axis=1) < 1.0) & (tri_lat.max(axis=1) > -1.0)
        assert expected.any()
        assert (mask[expected]).all()
    # the seam cell 179.5..180.5 x 0..1, split by its SW-NE diagonal
    upper_left = _elem_of(mesh, (179.5, 0.0), (-179.5, 1.0), (179.5, 1.0))
    lower_right = _elem_of(mesh, (179.5, 0.0), (-179.5, 0.0), (-179.5, 1.0))
    tri_lon, tri_lat = mesh.lon[mesh.elem_nodes], mesh.lat[mesh.elem_nodes]
    lon_east = np.where(tri_lon < 0, tri_lon + 360.0, tri_lon)   # 174.5 .. 184.5
    for bbox, inside, outside in (
            ((179.0, 179.55, 0.6, 0.9), upper_left, lower_right),
            ((-179.55, -179.0, 0.1, 0.4), lower_right, upper_left)):
        mask = extract_fesom_region.select_region(
            mesh, RegionSpec(mode="bbox", bbox=bbox))
        assert mask[inside] and not mask[outside], bbox
        box_east = (bbox[0] % 360.0, bbox[1] % 360.0, bbox[2], bbox[3])
        expected = np.array([_clipped_area(x, y, box_east) > 1e-12
                             for x, y in zip(lon_east, tri_lat)])
        np.testing.assert_array_equal(mask, expected)


def _elem_of(mesh, *corners):
    """Index of the element whose vertices are ``corners`` (lon, lat)."""
    want = sorted((round(float(x), 9), round(float(y), 9)) for x, y in corners)
    for e, nodes in enumerate(mesh.elem_nodes):
        got = sorted((round(float(mesh.lon[n]), 9), round(float(mesh.lat[n]), 9))
                     for n in nodes)
        if got == want:
            return e
    raise AssertionError(f"no element with vertices {corners}")


def _clipped_area(tri_x, tri_y, bbox):
    """Area of a triangle clipped to an axis-aligned box (Sutherland-Hodgman),
    in the lon/lat plane."""
    min_lon, max_lon, min_lat, max_lat = bbox
    poly = list(zip(map(float, tri_x), map(float, tri_y)))
    for axis, bound, upper in ((0, min_lon, False), (0, max_lon, True),
                               (1, min_lat, False), (1, max_lat, True)):
        def keep(p):
            return p[axis] <= bound if upper else p[axis] >= bound
        out = []
        for k, p in enumerate(poly):
            q = poly[k - 1]
            if keep(p) != keep(q):
                t = (bound - q[axis]) / (p[axis] - q[axis])
                out.append((q[0] + t * (p[0] - q[0]), q[1] + t * (p[1] - q[1])))
            if keep(p):
                out.append(p)
        poly = out
        if not poly:
            return 0.0
    x, y = np.array(poly).T
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# cell 4..5 x 0..1 of the patch below, split by its SW-NE diagonal
_LOWER_RIGHT = ((4.0, 0.0), (5.0, 0.0), (5.0, 1.0))


@pytest.mark.parametrize("spec", [
    RegionSpec(mode="bbox", bbox=(4.6, 4.8, 0.1, 0.2)),
    RegionSpec(mode="center_radius", center=(4.75, 0.15), radius_km=5.0),
], ids=["bbox", "center_radius"])
def test_region_inside_one_triangle_selects_that_triangle(tmp_path, spec):
    """A region inside one triangle holds no vertex and not the triangle's
    centre: it selected nothing ("Region selection produced zero
    elements").  The circle is 17 km from the nearest edge, radius 5 km."""
    mesh = _mesh(tmp_path, (11, 7, 0.0, -3.0, 1.0))
    mask = extract_fesom_region.select_region(mesh, spec)
    assert np.flatnonzero(mask).tolist() == [_elem_of(mesh, *_LOWER_RIGHT)]


def test_bbox_keeps_every_triangle_its_edge_crosses(tmp_path):
    """Triangles the box's edge cuts with no vertex and no centre inside
    were dropped (holes along the crop edge); triangles outside stay out.
    No box edge runs through a vertex or along a triangle edge, so the
    clipped area decides."""
    mesh = _mesh(tmp_path, (11, 7, 0.0, -3.0, 1.0))
    bbox = (2.4, 7.7, -1.4, 0.3)
    mask = extract_fesom_region.select_region(
        mesh, RegionSpec(mode="bbox", bbox=bbox))
    tri_lon, tri_lat = mesh.lon[mesh.elem_nodes], mesh.lat[mesh.elem_nodes]
    expected = np.array([_clipped_area(x, y, bbox) > 1e-12
                         for x, y in zip(tri_lon, tri_lat)])
    np.testing.assert_array_equal(mask, expected)
    # the box's corner (7.7, -1.4) lies in this triangle, none of its
    # vertices or its centre in the box
    corner = _elem_of(mesh, (7.0, -2.0), (8.0, -2.0), (8.0, -1.0))
    assert mask[corner]
    assert not (mask & ~expected).any() and (~mask).any()


def test_circle_keeps_a_triangle_its_rim_crosses(tmp_path):
    """The circle centre sits 0.05 deg below the edge (4, 0)-(5, 0); its
    20 km rim crosses into the triangle above, whose nearest vertex is
    55 km away (dropped before).  The triangle holding the centre is taken
    too; every other edge is 35 km or more away."""
    mesh = _mesh(tmp_path, (11, 7, 0.0, -3.0, 1.0))
    mask = extract_fesom_region.select_region(
        mesh, RegionSpec(mode="center_radius", center=(4.5, -0.05),
                         radius_km=20.0))
    holder = _elem_of(mesh, (4.0, -1.0), (5.0, 0.0), (4.0, 0.0))
    crossed = _elem_of(mesh, *_LOWER_RIGHT)
    assert np.flatnonzero(mask).tolist() == sorted([holder, crossed])


@pytest.mark.parametrize("center, radius_km, patch", [
    # the cap's widest longitude is 64.1 deg at 87.8N; the old box stopped
    # at 51.7 deg (r / (111 cos lat0))
    ((0.0, 85.0), 500.0, (3, 3, 58.8, 87.3, 0.2)),
    # half-width asin(sin(r/R) / cos 60) = 18.22 deg against the old 18.02
    ((0.0, 60.0), 1000.0, (3, 3, 18.04, 61.2, 0.04)),
    # the cap reaches the pole: every longitude (the old box stopped at
    # 129 deg)
    ((0.0, 88.0), 500.0, (3, 3, 149.8, 88.8, 0.2)),
], ids=["85N", "60N", "over_the_pole"])
def test_circle_candidate_box_is_the_spherical_cap(tmp_path, center,
                                                   radius_km, patch):
    """A patch wholly inside the circle (every vertex within the radius)
    but outside the old flat candidate box is selected; a patch on the far
    side of the pole stays out."""
    mesh = _mesh(tmp_path, patch, (3, 3, 170.0, 80.0, 1.0))
    tri_lon, tri_lat = mesh.lon[mesh.elem_nodes], mesh.lat[mesh.elem_nodes]
    _, _, lon0, _, step = patch
    near = ((tri_lon.min(axis=1) > lon0 - 1e-6)
            & (tri_lon.max(axis=1) < lon0 + 2 * step + 1e-6))
    far = ~near
    assert near.sum() == 8 and far.sum() == 8
    assert (_haversine_km(tri_lon[near], tri_lat[near], *center)
            < radius_km).all()
    assert (_haversine_km(tri_lon[far], tri_lat[far], *center)
            > radius_km).all()
    mask = extract_fesom_region.select_region(
        mesh, RegionSpec(mode="center_radius", center=center,
                         radius_km=radius_km))
    np.testing.assert_array_equal(mask, near)


def test_polygon_on_the_seam_selects_the_seam_triangles(tmp_path):
    """A small polygon inside one seam cell, holding no vertex and no centre:
    only the triangle test on the unwrapped triangle can see it."""
    mesh = _mesh(tmp_path, (11, 7, 174.5, -3.0, 1.0))
    seam = _seam_elements(mesh)
    path = _write_polygon(tmp_path, [179.6, 179.95, 179.95, 179.6],
                          [0.2, 0.2, 0.4, 0.4])
    mask = extract_fesom_region.select_region(
        mesh, RegionSpec(mode="polygon", polygon_path=path))
    tri_lat = mesh.lat[mesh.elem_nodes]
    expected = seam & (tri_lat.min(axis=1) == 0.0) & (tri_lat.max(axis=1) == 1.0)
    assert expected.sum() == 2
    np.testing.assert_array_equal(mask, expected)


def test_subset_carries_elem_area(tmp_path):
    from kelp.readers.fesom2 import _load_mesh_nc
    geometry = grid_geometry(3, 2, 0.0, 0.0, 1.0)                # 4 triangles
    for convention in ("core2", "comp", "ugrid"):
        path = write_meshdiag(tmp_path / f"{convention}.mesh.diag.nc", convention,
                              geometry=geometry, elem_area=[0.1, 0.2, 0.3, 0.4])
        mesh = extract_fesom_region.load_mesh_source(str(path))
        sel = extract_fesom_region.build_subset_selection(
            mesh, np.array([False, True, False, True]))
        ds = extract_fesom_region.build_subset_mesh_dataset(mesh, sel, {}, False)
        assert ds["elem_area"].dims == ("elem",)
        np.testing.assert_array_equal(ds["elem_area"].values, [0.2, 0.4])
        out = tmp_path / f"{convention}.region.nc"
        extract_fesom_region._write_dataset(ds, out, "netcdf")
        np.testing.assert_array_equal(_load_mesh_nc(out)._elem_areas, [0.2, 0.4])


def test_region_across_the_dateline_in_continuous_longitudes(tmp_path):
    """A region across the dateline is given as one interval past 180
    (here 177 to 183): it reaches the triangles stored at -179.5 .. -177."""
    mesh = _mesh(tmp_path, (11, 7, 174.5, -3.0, 1.0))
    tri_lon, tri_lat = mesh.lon[mesh.elem_nodes], mesh.lat[mesh.elem_nodes]
    lon_east = np.where(tri_lon < 0, tri_lon + 360.0, tri_lon)   # 174.5 .. 184.5
    vertex_in = ((lon_east >= 177.0) & (lon_east <= 183.0)
                 & (tri_lat >= -1.0) & (tri_lat <= 1.0)).any(axis=1)
    bbox_mask = extract_fesom_region.select_region(
        mesh, RegionSpec(mode="bbox", bbox=(177.0, 183.0, -1.0, 1.0)))
    np.testing.assert_array_equal(bbox_mask, vertex_in)
    assert (bbox_mask & (tri_lon < 0).all(axis=1)).any()
    path = _write_polygon(tmp_path, [177.0, 183.0, 183.0, 177.0], [-1.0, -1.0, 1.0, 1.0])
    poly_mask = extract_fesom_region.select_region(
        mesh, RegionSpec(mode="polygon", polygon_path=path))
    assert (poly_mask[vertex_in]).all()
    assert (poly_mask & (tri_lon < 0).all(axis=1)).any()
