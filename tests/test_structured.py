"""Structured-grid -> Mesh foundation (readers/structured.py, TODO 4.6)."""
import numpy as np
import pytest

from kelp.readers.structured import (
    build_structured_mesh, c_grid_to_t, field_to_nodes)


def _grid(ny=4, nx=5):
    # mildly curvilinear: lon skewed by latitude, lat regular
    y = np.linspace(0.0, 3.0, ny)
    x = np.linspace(0.0, 4.0, nx)
    X, Y = np.meshgrid(x, y)            # (ny, nx)
    lon2d = X + 0.1 * Y
    lat2d = Y + 0.0 * X
    return lon2d, lat2d


def test_all_wet_node_and_quad_counts():
    lon2d, lat2d = _grid(4, 5)
    wet = np.ones((4, 5), dtype=bool)
    mesh = build_structured_mesh(lon2d, lat2d, wet)
    assert mesh.n_nodes == 20                  # all T-points
    assert mesh.elem_nodes.shape == (3 * 4, 4)  # (ny-1)*(nx-1) quads, 4 corners
    assert mesh.elem_nodes.min() == 0 and mesh.elem_nodes.max() == 19
    assert mesh._structured_shape == (4, 5)


def test_land_cell_drops_node_and_touching_quads():
    # deep-interior land cell (5x5, centre) — far enough from every edge that
    # no wet neighbour is orphaned, so the count is clean: 25 - 1 land = 24.
    lon2d, lat2d = _grid(5, 5)
    wet = np.ones((5, 5), dtype=bool)
    wet[2, 2] = False                          # one interior land cell
    mesh = build_structured_mesh(lon2d, lat2d, wet)
    assert mesh.n_nodes == 24                  # land T-point dropped, no orphans
    # the 4 quads with corner (2,2) are dropped -> 16 - 4 = 12
    assert mesh.elem_nodes.shape[0] == 12
    # no orphan nodes: every kept node belongs to >=1 element
    assert mesh.node_to_elems_count.min() >= 1
    # every element's nodes are valid + each node belongs to its listed elems
    for e in range(mesh.elem_nodes.shape[0]):
        for nd in mesh.elem_nodes[e]:
            assert 0 <= nd < mesh.n_nodes


def test_orphan_wet_cell_excluded():
    # A wet cell connected to the rest only diagonally has no fully-wet quad:
    # it would be a dead-end node (false OUTSIDE) and must be excluded.
    #   row0: W W .      row2: . . W   <- (2,2) touches (1,1) only at a corner
    lon2d, lat2d = _grid(3, 3)
    wet = np.ones((3, 3), dtype=bool)
    wet[0, 2] = wet[1, 2] = wet[2, 0] = wet[2, 1] = False
    mesh = build_structured_mesh(lon2d, lat2d, wet)
    # only the fully-wet 2x2 block (0,0)(0,1)(1,0)(1,1) survives -> 1 quad, 4 nodes
    assert mesh.elem_nodes.shape[0] == 1
    assert mesh.n_nodes == 4
    assert mesh.node_to_elems_count.min() >= 1          # no orphans kept
    assert mesh._structured_node_id[2, 2] == -1         # orphan T-point excluded


def test_node_to_elems_consistent():
    lon2d, lat2d = _grid(4, 5)
    mesh = build_structured_mesh(lon2d, lat2d, np.ones((4, 5), bool))
    for i in range(mesh.n_nodes):
        for e in mesh.node_to_elems[i, : mesh.node_to_elems_count[i]]:
            if e >= 0:
                assert i in set(mesh.elem_nodes[e].tolist())


def test_validation():
    with pytest.raises(ValueError, match="2-D"):
        build_structured_mesh(np.zeros(5), np.zeros(5), np.ones(5, bool))
    with pytest.raises(ValueError, match="wet_mask shape"):
        build_structured_mesh(np.zeros((3, 4)), np.zeros((3, 4)), np.ones((2, 2), bool))
    with pytest.raises(ValueError, match="no ocean cells|no fully-wet"):
        build_structured_mesh(np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3), bool))


def test_c_grid_to_t_averages_faces():
    # node_id: 3x3 all wet, raster order
    node_id = np.arange(9, dtype=np.int64).reshape(3, 3)
    u_faces = np.array([[1.0, 3.0, 5.0]] * 3)      # varies in x (east faces)
    v_faces = np.zeros((3, 3))
    u_t, v_t = c_grid_to_t(u_faces, v_faces, node_id)
    # west+east average; i=0 keeps the east face (1.0); i=1 -> (1+3)/2=2; i=2 -> 4
    np.testing.assert_allclose(u_t.reshape(3, 3)[0], [1.0, 2.0, 4.0])
    np.testing.assert_array_equal(v_t, np.zeros(9))


def test_c_grid_to_t_masked_face_zeroed():
    # A land face carries NaN (model mask), not 0; it must be zeroed before
    # averaging so the wet neighbour is halved, not NaN-poisoned.
    node_id = np.arange(9, dtype=np.int64).reshape(3, 3)
    u_faces = np.array([[2.0, 2.0, 2.0],
                        [np.nan, 4.0, 4.0],          # (1,0) = land face -> NaN
                        [2.0, 2.0, 2.0]])
    v_faces = np.zeros((3, 3))
    u_t, _ = c_grid_to_t(u_faces, v_faces, node_id)
    grid = u_t.reshape(3, 3)
    assert np.isfinite(grid).all()                   # no NaN leaked through
    assert grid[1, 0] == 0.0                         # west edge keeps zeroed face
    assert grid[1, 1] == pytest.approx(2.0)          # 0.5*(0 + 4) -> halved


def test_c_grid_to_t_varying_field_with_land():
    # land at the interior (2,2): node_id has a -1 gap. A field varying in BOTH
    # x and y must land on the right node (catches a scrambled C->T scatter and
    # exercises the v-face averaging, not just u).
    lon2d, lat2d = _grid(5, 5)
    wet = np.ones((5, 5), bool); wet[2, 2] = False
    mesh = build_structured_mesh(lon2d, lat2d, wet)
    node_id = mesh._structured_node_id
    assert mesh.node_to_elems_count.min() >= 1            # no orphans (interior land)
    u_faces = np.tile(np.arange(5.0), (5, 1))             # u(j,i) = i
    v_faces = np.tile(np.arange(5.0)[:, None], (1, 5))    # v(j,i) = j
    u_t, v_t = c_grid_to_t(u_faces, v_faces, node_id)
    for (j, i) in [(0, 0), (1, 3), (4, 4), (3, 1)]:
        nd = int(node_id[j, i])
        assert nd >= 0
        exp_u = u_faces[j, i] if i == 0 else 0.5 * (u_faces[j, i - 1] + u_faces[j, i])
        exp_v = v_faces[j, i] if j == 0 else 0.5 * (v_faces[j - 1, i] + v_faces[j, i])
        assert u_t[nd] == pytest.approx(exp_u)
        assert v_t[nd] == pytest.approx(exp_v)


def test_field_to_nodes_compaction():
    node_id = np.array([[0, -1, 1], [2, 3, 4]], dtype=np.int64)   # one land
    f2d = np.array([[10.0, 99.0, 20.0], [30.0, 40.0, 50.0]])
    out = field_to_nodes(f2d, node_id)
    np.testing.assert_array_equal(out, [10.0, 20.0, 30.0, 40.0, 50.0])
