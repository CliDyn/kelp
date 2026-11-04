"""utils/check_grid_hash_cell.py — window statistics and the recommendation on a
mesh that is refined where the particles are (the CORE2-Arctic situation)."""
import importlib.util
import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from kelp.core.mesh import Mesh, build_node_to_elems, build_search  # noqa: E402

spec = importlib.util.spec_from_file_location("check_grid_hash_cell", REPO / "utils" / "check_grid_hash_cell.py")
cgh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cgh)


def _refined_mesh():
    """Sparse background (~1 deg) plus a dense patch (~0.05 deg) around (5, 5)."""
    from scipy.spatial import Delaunay
    rng = np.random.default_rng(0)
    bg = np.column_stack([rng.uniform(-20, 30, 1500), rng.uniform(-20, 30, 1500)])
    patch = np.column_stack([rng.uniform(3, 7, 6000), rng.uniform(3, 7, 6000)])
    pts = np.vstack([bg, patch])
    tri = Delaunay(pts).simplices.astype(np.int32)
    lon, lat = pts[:, 0].copy(), pts[:, 1].copy()
    mesh = Mesh(name="refined", nodes_lon=lon, nodes_lat=lat, depth=np.full(lon.size, 100.0),
                elem_nodes=tri, elem_center_lon=lon[tri].mean(1), elem_center_lat=lat[tri].mean(1),
                node_to_elems=None, node_to_elems_count=None, sigma=np.array([]))
    mesh.node_to_elems, mesh.node_to_elems_count = build_node_to_elems(lon.size, tri)
    build_search(mesh)
    return mesh, patch


def test_auto_cell_overfills_window_in_the_refined_patch_and_a_smaller_cell_fixes_it():
    mesh, patch = _refined_mesh()
    lon, lat = patch[::10, 0], patch[::10, 1]
    rows = cgh.probe_cells(mesh, lon, lat, [0.5, 0.2, 0.05])
    auto = rows[0]
    assert auto["auto"] and auto["median"] > cgh.TARGET_MEDIAN          # the symptom
    by_cell = {round(r["cell"], 3): r for r in rows if not r["auto"]}
    assert by_cell[0.2]["median"] < auto["median"] / 3 and by_cell[0.2]["empty"] == 0
    assert all(r["approx"] == 0 for r in rows if r["cell"] >= 0.2)   # dense patch: exact anchors
    cell, msg = cgh.recommend(rows)
    assert cell is not None and cell <= 0.2 and "grid_hash_cell_size" in msg


def test_too_small_cell_is_not_recommended_where_it_leaves_empty_windows():
    mesh, patch = _refined_mesh()
    # queries in the SPARSE background: a 0.05 deg cell leaves empty windows there
    rng = np.random.default_rng(1)
    lon, lat = rng.uniform(-15, 0, 300), rng.uniform(-15, 0, 300)
    rows = cgh.probe_cells(mesh, lon, lat, [0.05])
    assert rows[1]["empty"] > 0
    cell, _ = cgh.recommend(rows)
    assert cell is None


def test_probe_restores_the_original_hash():
    mesh, patch = _refined_mesh()
    before = mesh._gh_cell_size
    cgh.probe_cells(mesh, patch[:50, 0], patch[:50, 1], [0.3])
    assert mesh._gh_cell_size == before


def test_approximate_anchors_are_counted_and_block_the_recommendation():
    mesh, patch = _refined_mesh()
    rng = np.random.default_rng(2)
    # queries in the SPARSE background (spacing ~1 deg): a 0.3 deg cell keeps the windows
    # non-empty for many queries but the true nearest node often sits > 1 cell away
    lon, lat = rng.uniform(-15, 0, 400), rng.uniform(-15, 0, 400)
    rows = cgh.probe_cells(mesh, lon, lat, [0.3])
    r = rows[1]
    assert r["approx"] > 0
    cell, msg = cgh.recommend(rows)
    assert cell is None and "approximate anchor" in msg or cell is None
    # with a tolerance the same cell can be accepted knowingly
    frac = r["approx"] / r["n"]
    cell2, _ = cgh.recommend(rows, allow_approx=frac + 1e-9) if r["empty"] == 0 and r["median"] >= cgh.MIN_MEDIAN else (None, "")
    assert cell2 in (0.3, None)


# --- the tool must use the kernel's own cell rule: the longitude fold of
# core.mesh._gh_cell_of (a query west of the grid's lon_min lands in the
# grid's own branch) and the scan's neighbours (wrapped on a cyclic grid,
# skipped past the edge of a regional one).

from scipy.spatial import Delaunay, cKDTree  # noqa: E402
from kelp.core.mesh import _gh_cell_of, _grid_hash_nearest_3x3, build_grid_hash  # noqa: E402


def _cloud_mesh(lon, lat):
    tri = Delaunay(np.column_stack([lon, lat])).simplices.astype(np.int32)
    mesh = Mesh(name="cloud", nodes_lon=lon, nodes_lat=lat, depth=np.full(lon.size, 100.0),
                elem_nodes=tri, elem_center_lon=lon[tri].mean(1), elem_center_lat=lat[tri].mean(1),
                node_to_elems=None, node_to_elems_count=None, sigma=np.array([]))
    mesh.node_to_elems, mesh.node_to_elems_count = build_node_to_elems(lon.size, tri)
    build_search(mesh)
    return mesh


def _kernel_window(mesh, lon, lat):
    """The flat cells the kernel's 3x3 scan visits for one query."""
    nx, ny, wrap = mesh._gh_nx, mesh._gh_ny, mesh._gh_wrap_lon
    ci, cj = _gh_cell_of(float(lon), float(lat), mesh._gh_cell_size, mesh._gh_lon_min,
                         mesh._gh_lat_min, nx, ny, wrap)
    cells = []
    for dj in (-1, 0, 1):
        if not 0 <= cj + dj < ny:
            continue
        for di in (-1, 0, 1):
            ci2 = ci + di
            if wrap:
                ci2 %= nx
            elif not 0 <= ci2 < nx:
                continue
            cells.append((cj + dj) * nx + ci2)
    return cells


def _xyz(lon, lat):
    lam, phi = np.radians(lon), np.radians(lat)
    return np.column_stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)])


def _clouds():
    rng = np.random.default_rng(3)
    glob = (rng.uniform(-180, 180, 4000), rng.uniform(-75, 75, 4000),
            rng.uniform(-180, 180, 2000), rng.uniform(-70, 70, 2000))
    # a regional mesh at negative longitudes: both branches have the same span,
    # so the hash is built in [0, 360) and every query lies west of lon_min;
    # the queries reach past the outer nodes into the edge cells of the grid
    atl = (rng.uniform(-40, -10, 3000), rng.uniform(30, 50, 3000),
           rng.uniform(-41, -9, 2000), rng.uniform(29, 51, 2000))
    # a regional mesh across the dateline
    lon = rng.uniform(160, 200, 3000)
    qlon = rng.uniform(162, 198, 2000)
    pac = (np.where(lon > 180, lon - 360, lon), rng.uniform(-10, 10, 3000),
           np.where(qlon > 180, qlon - 360, qlon), rng.uniform(-9, 9, 2000))
    # a regular 0.5-degree lattice: its span is a whole number of cells, so the
    # outermost nodes fill the grid's last row and column, where the scan
    # skips the neighbours past the edge (a clamp would count them twice)
    ii, jj = np.meshgrid(np.arange(61), np.arange(41))
    lattice = (-40.0 + 0.5 * ii.ravel(), 30.0 + 0.5 * jj.ravel(),
               rng.uniform(-41, -9, 2000), rng.uniform(29, 51, 2000))
    return {"global": glob, "negative_lon": atl, "dateline": pac, "lattice": lattice}


@pytest.mark.parametrize("name,cells", [("global", (3.0, 1.0, 2.9)),
                                        ("negative_lon", (1.0, 0.5)),
                                        ("dateline", (1.0, 0.3)),
                                        ("lattice", (1.0, 0.5))])
def test_window_counts_equal_the_kernel_window(name, cells):
    lon, lat, qlon, qlat = _clouds()[name]
    mesh = _cloud_mesh(lon, lat)
    for cell in cells:
        build_grid_hash(mesh, cell)
        counts = np.diff(mesh._gh_offsets)
        expected = np.array([counts[_kernel_window(mesh, x, y)].sum()
                             for x, y in zip(qlon, qlat)])
        got = cgh.window_counts(mesh, qlon, qlat)
        assert (got != expected).sum() == 0, (name, cell, int((got != expected).sum()))


@pytest.mark.parametrize("name,cells", [("global", (3.0, 2.9)),
                                        ("dateline", (1.0, 0.5))])
def test_approx_anchor_count_equals_the_kernel_verdict(name, cells):
    """Approximate anchor = the kernel's 3x3 scan returns another node than
    the true nearest (ties do not occur in a random cloud)."""
    lon, lat, qlon, qlat = _clouds()[name]
    mesh = _cloud_mesh(lon, lat)
    nearest = cKDTree(_xyz(lon, lat)).query(_xyz(qlon, qlat))[1]
    for cell in cells:
        build_grid_hash(mesh, cell)
        scan = np.array([_grid_hash_nearest_3x3(
            float(x), float(y), mesh._gh_offsets, mesh._gh_node_ids, mesh._gh_cell_size,
            mesh._gh_lon_min, mesh._gh_lat_min, mesh._gh_nx, mesh._gh_ny, mesh._gh_wrap_lon,
            mesh.nodes_lon, mesh.nodes_lat, mesh._node_xyz) for x, y in zip(qlon, qlat)])
        expected = int((scan != nearest).sum())
        assert cgh.approx_anchor_count(mesh, qlon, qlat, nearest, cell) == expected, (name, cell)
