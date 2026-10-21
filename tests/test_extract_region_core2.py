"""Regression tests: extract_fesom_region.load_mesh_source must read every
FESOM2 mesh-diag convention — including the older core2 packed-radian
``nodes``/``elem`` form — and rebuild an unpopulated ``nod_in_elem2D``.

Before this fix the extractor hard-required ``lon``/``lat``/``elements`` and
died with ``KeyError`` on the core2 ``fesom.mesh.diag.nc`` (which ships packed
``nodes`` in radians + ``elem`` + an all-zero ``nod_in_elem2D``).
"""
import importlib.util
import pathlib
import sys

import numpy as np
import pytest

from tests._fesom2_synth import write_meshdiag

# utils/ is not an importable package — load the script module by file path.
# Register it in sys.modules before exec so its @dataclass definitions resolve.
_EXT_PATH = pathlib.Path(__file__).resolve().parents[1] / "utils" / "extract_fesom_region.py"
_spec = importlib.util.spec_from_file_location("extract_fesom_region", _EXT_PATH)
extract_fesom_region = importlib.util.module_from_spec(_spec)
sys.modules["extract_fesom_region"] = extract_fesom_region
_spec.loader.exec_module(extract_fesom_region)
load_mesh_source = extract_fesom_region.load_mesh_source


@pytest.mark.parametrize("convention", ["core2", "comp", "ugrid"])
def test_extractor_load_mesh_all_conventions(tmp_path, convention):
    p = write_meshdiag(tmp_path / f"{convention}.mesh.diag.nc", convention)
    mesh = load_mesh_source(str(p))
    # coordinates resolved to degrees (core2 packed nodes are radians)
    np.testing.assert_allclose(np.sort(mesh.lon), [0, 0, 1, 1], atol=1e-9)
    np.testing.assert_allclose(np.sort(mesh.lat), [0, 0, 1, 1], atol=1e-9)
    # connectivity 0-based (E, 3)
    assert mesh.elem_nodes.shape == (2, 3)
    assert mesh.elem_nodes.min() == 0 and mesh.elem_nodes.max() == 3
    # z-levels picked up (core2 stores 'zbar', comp/ugrid 'nz')
    assert mesh.sigma.size == 3 and mesh.sigma.min() >= 0
    # adjacency present and consistent
    assert (mesh.node_to_elems[:, 0] >= 0).all()


@pytest.mark.parametrize("convention", ["core2", "comp", "ugrid"])
def test_extractor_rebuilds_unpopulated_nod_in_elem2d(tmp_path, convention):
    # An all-zero nod_in_elem2D (the core2 file's real state) must be detected
    # and rebuilt from the connectivity, not trusted.
    p = write_meshdiag(tmp_path / f"{convention}_empty.mesh.diag.nc", convention,
                       zero_nod_in_elem2D=True)
    mesh = load_mesh_source(str(p))
    assert (mesh.node_to_elems[:, 0] >= 0).all()
    # every listed element actually contains its node (adjacency is correct)
    for i in range(mesh.lon.shape[0]):
        for e in mesh.node_to_elems[i, : mesh.node_to_elems_count[i]]:
            if e >= 0:
                assert i in set(mesh.elem_nodes[e].tolist())


def test_extractor_core2_packed_nodes_radians_to_degrees(tmp_path):
    # The packed core2 'nodes' are radians; the loader must convert to degrees
    # (a 1-degree square must come back as ~1 degree, not ~0.017 rad).
    p = write_meshdiag(tmp_path / "core2.mesh.diag.nc", "core2")
    mesh = load_mesh_source(str(p))
    assert mesh.lon.max() == pytest.approx(1.0, abs=1e-9)
    assert mesh.lat.max() == pytest.approx(1.0, abs=1e-9)


def _spy_build_node_to_elems(monkeypatch):
    """Spy on the extractor's build_node_to_elems so we can assert the rebuild
    branch (the 4-node rebuilt table is byte-identical to a healthy stored one,
    so array contents alone cannot tell 'used as-is' from 'rebuilt')."""
    calls = {"n": 0}
    real = extract_fesom_region.build_node_to_elems

    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(extract_fesom_region, "build_node_to_elems", spy)
    return calls


def test_extractor_healthy_table_used_as_is(tmp_path, monkeypatch):
    # Control: a healthy nod_in_elem2D must NOT trip the degenerate rebuild.
    calls = _spy_build_node_to_elems(monkeypatch)
    p = write_meshdiag(tmp_path / "core2_ok.mesh.diag.nc", "core2")
    mesh = load_mesh_source(str(p))
    assert (mesh.node_to_elems[:, 0] >= 0).all()
    assert calls["n"] == 0


@pytest.mark.parametrize("convention", ["core2", "comp", "ugrid"])
def test_extractor_partial_corrupt_nod_in_elem2d_rebuilds(tmp_path, convention, monkeypatch):
    # One zeroed row (25 % < the 50 % all-empty bar) whose count still says >0
    # must trip the consistency check and rebuild.
    calls = _spy_build_node_to_elems(monkeypatch)
    p = write_meshdiag(tmp_path / f"{convention}_partial.mesh.diag.nc", convention,
                       corrupt_nodes=[1])
    mesh = load_mesh_source(str(p))
    assert calls["n"] >= 1
    assert (mesh.node_to_elems[:, 0] >= 0).all()


def test_extractor_packed_nodes_degrees_not_rescaled(tmp_path):
    # FIX E analogue: packed 'nodes' labelled units=degrees must NOT be *180/pi.
    p = write_meshdiag(tmp_path / "deg.mesh.diag.nc", "core2", nodes_units="degrees")
    mesh = load_mesh_source(str(p))
    np.testing.assert_allclose(np.sort(mesh.lon), [0, 0, 1, 1], atol=1e-9)


@pytest.mark.parametrize("start_index", [0, 2])
def test_extractor_honors_connectivity_start_index(tmp_path, start_index):
    # The connectivity base must come from the start_index attr, not a min()
    # guess: a non-{0,1} base (here 2) is mis-indexed by the old heuristic.
    p = write_meshdiag(tmp_path / f"ug{start_index}.nc", "ugrid", start_index=start_index)
    p1 = write_meshdiag(tmp_path / "ug1.nc", "ugrid")
    m = load_mesh_source(str(p))
    m1 = load_mesh_source(str(p1))
    np.testing.assert_array_equal(m.elem_nodes, m1.elem_nodes)
    assert m.elem_nodes.min() == 0 and m.elem_nodes.max() == 3


def test_extractor_malformed_count_vector_rebuilds(monkeypatch):
    # A nod_in_elem2D_num whose length != n_nodes (e.g. a singleton) must NOT be
    # trusted: it would silently broadcast inside the degeneracy check and be
    # returned as an invalid node_to_elems_count. The loader must rebuild from
    # the connectivity instead (the old _normalize_node_to_elems raised here).
    import xarray as xr
    calls = _spy_build_node_to_elems(monkeypatch)
    # 4 nodes, 2 triangles; a healthy nod_in_elem2D but a length-1 count vector.
    n2e_1based = np.array([[1, 2], [1, 0], [1, 2], [2, 0]], dtype=np.int32)  # (nod=4, N=2)
    ds = xr.Dataset({
        "nod_in_elem2D": (("N", "nod2"), n2e_1based.T),       # (2, 4)
        "nod_in_elem2D_num": (("bad",), np.array([2], dtype=np.int32)),  # singleton
    })
    elem_nodes = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    n2e, cnt = extract_fesom_region._resolve_node_to_elems(ds, 4, elem_nodes)
    assert calls["n"] >= 1                       # rebuilt, not trusted
    assert cnt.shape == (4,)                     # valid count vector, not (1,)
    assert (n2e[:, 0] >= 0).all()


# --- the element levels of the mesh diagnostic (nlevels, ulevels,
# zbar_e_bottom; nlevels_nod2D on the nodes) survive the subset, values and
# signs as in the source, and reach the FESOM2 reader's declaration.

_LEVELS_NEG = [0.0, -10.0, -30.0, -60.0, -100.0]
_LEVEL_SETS = {
    "all": dict(nlevels=[4, 5], ulevels=[1, 2], zbar_e_bottom=[-45.0, -100.0],
                nlevels_nod2D=[5, 4, 5, 3]),
    "nlevels_only": dict(nlevels=[4, 5]),
}
_LEVEL_VARS = ("nlevels", "ulevels", "zbar_e_bottom", "nlevels_nod2D")


def _subset_of_second_element(path):
    """The subset holding element 1 only (nodes 0, 2, 3)."""
    mesh = load_mesh_source(str(path))
    sel = extract_fesom_region.build_subset_selection(
        mesh, np.array([False, True]))
    np.testing.assert_array_equal(sel.elem_ids, [1])
    np.testing.assert_array_equal(sel.node_ids, [0, 2, 3])
    return mesh, extract_fesom_region.build_subset_mesh_dataset(
        mesh, sel, {}, False)


@pytest.mark.parametrize("convention", ["core2", "comp", "ugrid"])
@pytest.mark.parametrize("which", sorted(_LEVEL_SETS))
def test_extract_keeps_level_variables(tmp_path, convention, which):
    levels = _LEVEL_SETS[which]
    p = write_meshdiag(tmp_path / f"{convention}.mesh.diag.nc", convention,
                       zbar=_LEVELS_NEG, **levels)
    mesh, ds = _subset_of_second_element(p)
    expected = {
        "nlevels": ("elem", [5]),
        "ulevels": ("elem", [2]),
        "zbar_e_bottom": ("elem", [-100.0]),          # negative, as the source
        "nlevels_nod2D": ("nod2", [5, 5, 3]),
    }
    for name in _LEVEL_VARS:
        if name not in levels:
            assert name not in ds.variables
            continue
        dim, values = expected[name]
        assert ds[name].dims == (dim,)
        assert ds[name].dtype == mesh.ds[name].dtype
        np.testing.assert_array_equal(ds[name].values, values)


@pytest.mark.parametrize("fmt", ["netcdf", "zarr"])
def test_extract_level_variables_reach_the_reader(tmp_path, fmt):
    from kelp.readers.fesom2 import _load_mesh_nc, _load_mesh_zarr
    p = write_meshdiag(tmp_path / "core2.mesh.diag.nc", "core2",
                       zbar=_LEVELS_NEG, **_LEVEL_SETS["all"])
    _, ds = _subset_of_second_element(p)
    out = tmp_path / ("region.nc" if fmt == "netcdf" else "region.zarr")
    extract_fesom_region._write_dataset(ds, out, fmt)
    mesh = _load_mesh_nc(out) if fmt == "netcdf" else _load_mesh_zarr(out)
    md = mesh.metadata
    assert md["elem_levels_source"] == "zbar_e_bottom"
    np.testing.assert_array_equal(md["elem_bed"], [100.0])
    np.testing.assert_array_equal(md["uv_data_last_elem"], [3])
    np.testing.assert_array_equal(md["elem_ulevels"], [2])
    assert md["uv_data_last_elem"].dtype == np.int32
    assert md["elem_ulevels"].dtype == np.int32


@pytest.mark.parametrize("convention", ["core2", "comp", "ugrid"])
def test_extract_without_level_variables(tmp_path, convention):
    from kelp.readers.fesom2 import _load_mesh_nc
    p = write_meshdiag(tmp_path / f"{convention}.mesh.diag.nc", convention)
    _, ds = _subset_of_second_element(p)
    assert not set(_LEVEL_VARS) & set(ds.variables)
    out = tmp_path / "region.nc"
    extract_fesom_region._write_dataset(ds, out, "netcdf")
    md = _load_mesh_nc(out).metadata
    assert "elem_bed" not in md and "uv_data_last_elem" not in md


def test_extract_refuses_misshapen_level_variable(tmp_path):
    p = write_meshdiag(tmp_path / "comp.mesh.diag.nc", "comp",
                       nlevels=[3, 3])
    mesh = load_mesh_source(str(p))
    mesh.ds = mesh.ds.drop_vars("nlevels").assign(
        nlevels=(("other",), np.array([3, 3, 3], dtype=np.int32)))
    sel = extract_fesom_region.build_subset_selection(
        mesh, np.array([False, True]))
    with pytest.raises(ValueError, match="'nlevels'.*\\(2,\\)"):
        extract_fesom_region.build_subset_mesh_dataset(mesh, sel, {}, False)
