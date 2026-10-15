import os
import pathlib

import numpy as np
import pytest
from kelp.readers.fesom2 import (
    _load_mesh_nc,
    _load_mesh_zarr,
    _nod_in_elem2d_is_degenerate,
)
from tests._fesom2_synth import write_meshdiag, write_meshdiag_zarr


@pytest.mark.parametrize("convention", ["core2", "comp", "ugrid"])
def test_meshdiag_convention_loads_equivalently(tmp_path, convention):
    p = write_meshdiag(tmp_path / f"{convention}.mesh.diag.nc", convention)
    mesh = _load_mesh_nc(p)
    # coords in degrees
    np.testing.assert_allclose(np.sort(mesh.nodes_lon), [0, 0, 1, 1], atol=1e-9)
    np.testing.assert_allclose(np.sort(mesh.nodes_lat), [0, 0, 1, 1], atol=1e-9)
    # connectivity 0-based (E, 3), max id == n_nodes-1
    assert mesh.elem_nodes.shape == (2, 3)
    assert mesh.elem_nodes.min() == 0 and mesh.elem_nodes.max() == 3
    # depth positive-down
    assert mesh.depth.min() >= 0 and mesh.depth.max() == 30.0
    # z-levels positive
    assert mesh.sigma.min() >= 0 and np.isclose(mesh.sigma.max(), 30.0)
    assert mesh.vertical_coord == "zlevel"


def test_ugrid_face_nodes_start_index_zero(tmp_path):
    # FIX D: UGRID face_nodes with start_index=0 (0-based ids) must load to the
    # same Mesh as the 1-based UGRID fixture (loader subtracts start_index,
    # not a hardcoded 1).
    p1 = write_meshdiag(tmp_path / "ug1.nc", "ugrid")            # 1-based, no attr
    p0 = write_meshdiag(tmp_path / "ug0.nc", "ugrid", start_index=0)
    m1 = _load_mesh_nc(p1)
    m0 = _load_mesh_nc(p0)
    np.testing.assert_array_equal(m0.elem_nodes, m1.elem_nodes)
    assert m0.elem_nodes.min() == 0 and m0.elem_nodes.max() == 3


def test_packed_nodes_degrees_units_not_rescaled(tmp_path):
    # FIX E: packed 'nodes' labelled units=degrees must NOT be treated as
    # radians (no *180/pi rescale).
    p = write_meshdiag(tmp_path / "deg.nc", "core2", nodes_units="degrees")
    mesh = _load_mesh_nc(p)
    np.testing.assert_allclose(np.sort(mesh.nodes_lon), [0, 0, 1, 1], atol=1e-9)
    np.testing.assert_allclose(np.sort(mesh.nodes_lat), [0, 0, 1, 1], atol=1e-9)


@pytest.mark.parametrize("convention", ["core2", "comp", "ugrid"])
def test_empty_nod_in_elem2d_rebuilds_from_elements(tmp_path, convention, monkeypatch):
    # Some mesh diags (the core2 fesom.mesh.diag.nc among them) ship an all-zero
    # nod_in_elem2D while nod_in_elem2D_num is valid. After the 1-based->0-based
    # shift that is all -1, which breaks the containing-element search for EVERY
    # particle (all OUTSIDE, nothing advects). The loader must detect the
    # degenerate table and rebuild node_to_elems from the element connectivity.
    calls = _spy_build_node_to_elems(monkeypatch)
    p = write_meshdiag(tmp_path / f"{convention}_empty.mesh.diag.nc", convention,
                       zero_nod_in_elem2D=True)
    mesh = _load_mesh_nc(p)
    assert calls["n"] >= 1  # rebuild branch was taken
    # every node has at least one valid incident element after the rebuild
    assert (mesh.node_to_elems[:, 0] >= 0).all()
    # and each listed element actually contains that node (adjacency is correct)
    for i in range(mesh.n_nodes):
        for e in mesh.node_to_elems[i, : mesh.node_to_elems_count[i]]:
            if e >= 0:
                assert i in set(mesh.elem_nodes[e].tolist())


def _spy_build_node_to_elems(monkeypatch):
    """Wrap fesom2.build_node_to_elems so we can assert whether it was called.

    The 4-node rebuilt table is byte-identical to a healthy stored one, so the
    array contents alone cannot tell "used as-is" from "rebuilt". Spying on the
    rebuild entry point is the only way to pin the branch taken.
    """
    import kelp.readers.fesom2 as f2
    calls = {"n": 0}
    real = f2.build_node_to_elems

    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(f2, "build_node_to_elems", spy)
    return calls


def test_populated_nod_in_elem2d_is_used_as_is(tmp_path, monkeypatch):
    # Control: a healthy table must NOT trip the degenerate-rebuild guard.
    calls = _spy_build_node_to_elems(monkeypatch)
    p = write_meshdiag(tmp_path / "core2_ok.mesh.diag.nc", "core2")
    mesh = _load_mesh_nc(p)
    assert (mesh.node_to_elems[:, 0] < 0).mean() == 0.0  # nothing missing
    assert calls["n"] == 0  # precomputed table used as-is, NOT rebuilt


@pytest.mark.parametrize("convention", ["core2", "comp", "ugrid"])
def test_partial_corrupt_nod_in_elem2d_rebuilds_from_elements(tmp_path, convention, monkeypatch):
    # PARTIAL corruption: only one of four node rows is zeroed (25 % < the 50 %
    # all-empty bar), but its nod_in_elem2D_num still says it has an incident
    # element. The old mean-only guard missed this; the count/data consistency
    # check must catch the single bad row and rebuild from the connectivity.
    calls = _spy_build_node_to_elems(monkeypatch)
    p = write_meshdiag(tmp_path / f"{convention}_partial.mesh.diag.nc", convention,
                       corrupt_nodes=[1])
    mesh = _load_mesh_nc(p)
    assert calls["n"] >= 1  # the single bad row tripped the rebuild
    # every node has a valid incident element again after the rebuild
    assert (mesh.node_to_elems[:, 0] >= 0).all()
    # node 1 (corrupted) really gets its element(s) back
    n1 = set(mesh.node_to_elems[1, : mesh.node_to_elems_count[1]].tolist())
    assert all(1 in set(mesh.elem_nodes[e].tolist()) for e in n1 if e >= 0)
    assert len(n1) >= 1


def test_zarr_empty_nod_in_elem2d_rebuilds_from_elements(tmp_path, monkeypatch):
    # Regression on the Zarr path (extract_fesom_region.py output): an all-zero
    # nod_in_elem2D in a Zarr store must be detected and rebuilt, same as the
    # NetCDF path.
    calls = _spy_build_node_to_elems(monkeypatch)
    p = write_meshdiag_zarr(tmp_path / "core2_empty.zarr", zero_nod_in_elem2D=True)
    mesh = _load_mesh_zarr(p)
    assert calls["n"] >= 1  # rebuild branch was taken on the zarr path
    assert (mesh.node_to_elems[:, 0] >= 0).all()
    for i in range(mesh.n_nodes):
        for e in mesh.node_to_elems[i, : mesh.node_to_elems_count[i]]:
            if e >= 0:
                assert i in set(mesh.elem_nodes[e].tolist())


def test_zarr_populated_nod_in_elem2d_is_used_as_is(tmp_path, monkeypatch):
    # Control on the Zarr path: a healthy table is NOT rebuilt (the spy confirms
    # the branch, since the 4-node rebuilt table is byte-identical to the stored
    # one and the array contents alone cannot distinguish them).
    calls = _spy_build_node_to_elems(monkeypatch)
    p = write_meshdiag_zarr(tmp_path / "core2_ok.zarr")
    mesh = _load_mesh_zarr(p)
    assert (mesh.node_to_elems[:, 0] < 0).mean() == 0.0
    assert calls["n"] == 0  # precomputed table used as-is, NOT rebuilt


def test_degenerate_table_detector_unit():
    # Direct unit coverage of the guard predicate.
    # Healthy: every node has a valid first slot; count agrees -> not degenerate.
    n2e = np.array([[0, 1], [0, -1], [0, 1], [1, -1]], dtype=np.int32)
    cnt = np.array([2, 1, 2, 1], dtype=np.int32)
    assert _nod_in_elem2d_is_degenerate(n2e, cnt) is False
    # Fully empty (all -1) -> mostly-empty trips even if the count is also zero.
    empty = np.full((4, 2), -1, dtype=np.int32)
    assert _nod_in_elem2d_is_degenerate(empty, np.zeros(4, np.int32)) is True
    # Partial: one row zeroed but its count stays > 0 -> inconsistency trips
    # (25 % empty, below the mean bar).
    partial = n2e.copy()
    partial[1, :] = -1                       # count[1] still says 1
    assert _nod_in_elem2d_is_degenerate(partial, cnt) is True
    # A genuine dangling node (count 0, slot -1) is NOT an inconsistency.
    dangling = n2e.copy()
    dangling[1, :] = -1
    cnt0 = cnt.copy(); cnt0[1] = 0
    assert _nod_in_elem2d_is_degenerate(dangling, cnt0) is False
    # Count exceeds the table width (truncated/corrupt row) -> degenerate even
    # though every stored slot is a valid id (would drive OOB adjacency reads).
    cnt_over = cnt.copy(); cnt_over[0] = 3            # > max_conn (=2)
    assert _nod_in_elem2d_is_degenerate(n2e, cnt_over) is True
    # A negative count is corrupt -> degenerate (slots all look valid).
    cnt_neg = cnt.copy(); cnt_neg[2] = -1
    assert _nod_in_elem2d_is_degenerate(n2e, cnt_neg) is True
    # Empty input -> not degenerate (guard against 0-node edge).
    assert _nod_in_elem2d_is_degenerate(np.empty((0, 2), np.int32),
                                        np.empty((0,), np.int32)) is False
    # Zero-width table (no adjacency stored at all) -> degenerate, no IndexError.
    assert _nod_in_elem2d_is_degenerate(np.empty((4, 0), np.int32),
                                        np.zeros(4, np.int32)) is True


# Set KELP_TEST_CORE2_MESH (or the older LAGDRIFT_TEST_CORE2_MESH) to a real core2 fesom.mesh.diag.nc to run this
# integration check.  No data path is committed (portable, non-identifying);
# the test skips unless the env var points at an existing file.
_CORE2 = os.environ.get("KELP_TEST_CORE2_MESH", "") or os.environ.get("LAGDRIFT_TEST_CORE2_MESH", "")


@pytest.mark.skipif(not (_CORE2 and pathlib.Path(_CORE2).exists()),
                    reason="set KELP_TEST_CORE2_MESH to a real core2 mesh.diag")
def test_real_core2_mesh_loads():
    mesh = _load_mesh_nc(pathlib.Path(_CORE2))
    assert mesh.n_nodes == 126858
    assert mesh.elem_nodes.shape == (244659, 3)
    assert mesh.elem_nodes.min() == 0
    assert -180.0 <= mesh.nodes_lon.min() and mesh.nodes_lon.max() <= 180.0
    assert mesh.depth.min() >= 0.0
    assert mesh._elem_areas is not None  # elem_area read from file
    # This file's nod_in_elem2D is unpopulated (all zeros); the loader must have
    # rebuilt node_to_elems from elements so every node has an incident element
    # (regression: otherwise the containing-element search puts all particles
    # OUTSIDE and nothing advects).
    assert (mesh.node_to_elems[:, 0] >= 0).all()
