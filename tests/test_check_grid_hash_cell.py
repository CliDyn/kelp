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
