"""The FESOM2 reader declares each element's bottom from the mesh diagnostic.

A FESOM2 mesh diagnostic holds, per element, ``nlevels`` (1-based: the
element has ``nlevels - 1`` wet layers), ``ulevels`` (its first level;
> 1 under an ice-shelf cavity) and ``zbar_e_bottom`` (its bed, written
negative).  The NetCDF and the Zarr loaders turn them into the mesh
metadata the depth-linear element table reads
(``vertical_uv.depth_points(mesh, "layer_linear", layout="element")``):

- ``elem_bed`` (float64 ``(E,)``, positive down): ``|zbar_e_bottom|`` when
  present, else ``|sigma|[nlevels - 1]``;
- ``uv_data_last_elem`` (int32 ``(E,)``): the last layer whose lower level
  is at or above the bed,
  ``searchsorted(|sigma|[1:], elem_bed, side="right") - 1``;
- ``elem_ulevels`` (int32 ``(E,)``) when the diagnostic has ``ulevels``;
- ``elem_levels_source``: ``"zbar_e_bottom"`` or ``"nlevels"``.

A diagnostic with neither ``nlevels`` nor ``zbar_e_bottom`` (and an ASCII
mesh directory, which has no diagnostic) declares nothing.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from kelp.core.config import MeshConfig
from kelp.numerics.vertical_uv import depth_points
from kelp.readers.fesom2 import Fesom2Reader, _load_mesh_nc, _load_mesh_zarr
from tests._fesom2_synth import write_meshdiag, write_meshdiag_zarr

LEVELS_NEG = np.array([0.0, -10.0, -30.0, -60.0, -100.0])
SIGMA_ABS = np.abs(LEVELS_NEG)
KEYS = ("uv_data_last_elem", "elem_bed", "elem_ulevels",
        "elem_levels_source")


def _declare(variables, n_elem, sigma_abs=SIGMA_ABS, **kw):
    """Run the reader's helper on a dict of diagnostic variables."""
    fesom2 = importlib.import_module("kelp.readers.fesom2")

    def get_var(name):
        value = variables.get(name)
        return None if value is None else np.asanyarray(value)  # keeps masks

    return fesom2._elem_level_metadata(get_var, sigma_abs, n_elem, **kw)


def _check(md, last, bed, source):
    assert md["uv_data_last_elem"].dtype == np.int32
    assert md["elem_bed"].dtype == np.float64
    np.testing.assert_array_equal(md["uv_data_last_elem"], last)
    np.testing.assert_array_equal(md["elem_bed"], bed)
    assert md["elem_levels_source"] == source


# ---------------------------------------------------------------------------
# The declaration rule (the helper both loaders use)
# ---------------------------------------------------------------------------

def test_declaration_from_zbar_e_bottom():
    md = _declare({"zbar_e_bottom": [-60.0, -30.0, -100.0]}, 3)
    _check(md, [2, 1, 3], [60.0, 30.0, 100.0], "zbar_e_bottom")
    assert "elem_ulevels" not in md


def test_declaration_from_nlevels():
    md = _declare({"nlevels": np.array([4, 3, 5], dtype=np.int32)}, 3)
    _check(md, [2, 1, 3], [60.0, 30.0, 100.0], "nlevels")


def test_partial_cell_bed():
    # a bed between two levels: the layer it cuts (30-60 m) is not counted
    md = _declare({"zbar_e_bottom": [-45.0]}, 1)
    _check(md, [1], [45.0], "zbar_e_bottom")


def test_both_present_prefers_zbar_e_bottom():
    md = _declare({"nlevels": np.array([4, 3, 5], dtype=np.int32),
                   "zbar_e_bottom": [-45.0, -30.0, -100.0]}, 3)
    _check(md, [1, 1, 3], [45.0, 30.0, 100.0], "zbar_e_bottom")


def test_no_levels_no_declaration():
    assert _declare({}, 3) == {}
    # ulevels alone declares nothing either
    assert _declare({"ulevels": np.array([1, 1, 2], dtype=np.int32)}, 3) == {}


def test_no_level_table_no_declaration():
    # a mesh without levels (e.g. a Zarr store without ``nz``): nothing to
    # place the bottoms on, nothing read or refused
    assert _declare({"nlevels": np.array([4, 3, 5], dtype=np.int32),
                     "zbar_e_bottom": [-60.0, -30.0, -100.0]}, 3,
                    sigma_abs=np.array([])) == {}


def test_ulevels_kept():
    md = _declare({"nlevels": np.array([4, 3, 5], dtype=np.int32),
                   "ulevels": np.array([1, 1, 2], dtype=np.int32)}, 3)
    _check(md, [2, 1, 3], [60.0, 30.0, 100.0], "nlevels")
    assert md["elem_ulevels"].dtype == np.int32
    np.testing.assert_array_equal(md["elem_ulevels"], [1, 1, 2])


def test_bed_above_first_level_is_surface_only():
    # a bed shallower than the first layer's lower level: -1 (surface only)
    md = _declare({"zbar_e_bottom": [-5.0, -10.0]}, 2)
    _check(md, [-1, 0], [5.0, 10.0], "zbar_e_bottom")


def test_nlevels_as_float_accepted():
    # a decoder may hand integer levels back as floats
    md = _declare({"nlevels": np.array([4.0, 3.0, 5.0])}, 3)
    _check(md, [2, 1, 3], [60.0, 30.0, 100.0], "nlevels")


@pytest.mark.parametrize("variables, words", [
    ({"nlevels": np.array([4, 3], dtype=np.int32)}, ["'nlevels'", "(3,)"]),
    ({"nlevels": np.array([4, 0, 5], dtype=np.int32)}, ["'nlevels'", "1..5"]),
    ({"nlevels": np.array([4, 3, 6], dtype=np.int32)}, ["'nlevels'", "1..5"]),
    ({"nlevels": np.array([4.5, 3.0, 5.0])}, ["'nlevels'", "integer"]),
    ({"zbar_e_bottom": np.array([-60.0, np.nan, -100.0])},
     ["'zbar_e_bottom'", "element 1"]),
    ({"zbar_e_bottom": np.ma.masked_array([-60.0, -30.0, -100.0],
                                          mask=[False, False, True])},
     ["'zbar_e_bottom'", "element 2"]),
    ({"zbar_e_bottom": [-60.0, -30.0, -100.0, -100.0]},
     ["'zbar_e_bottom'", "(3,)"]),
    ({"nlevels": np.array([4, 3, 5], dtype=np.int32),
      "ulevels": np.array([1, 1], dtype=np.int32)}, ["'ulevels'", "(3,)"]),
])
def test_malformed_levels_refused(variables, words):
    """Each refusal names the mesh file, the variable, what is wrong, and
    the fix."""
    with pytest.raises(ValueError) as err:
        _declare(variables, 3, meshpath="some/fesom.mesh.diag.nc")
    msg = str(err.value)
    assert "some/fesom.mesh.diag.nc" in msg
    for word in words:
        assert word in msg, msg
    assert msg.endswith(
        "; use the run's full fesom.mesh.diag.nc, or re-extract the region "
        "with utils/extract_fesom_region.py"), msg


# ---------------------------------------------------------------------------
# The loaders (every diagnostic convention, NetCDF and Zarr)
# ---------------------------------------------------------------------------

LOADERS = ["core2", "comp", "ugrid", "zarr"]


def _load(tmp_path, convention, **levels):
    if convention == "zarr":
        path = write_meshdiag_zarr(tmp_path / "mesh.diag.zarr",
                                   zbar=LEVELS_NEG, **levels)
        return _load_mesh_zarr(path), path
    path = write_meshdiag(tmp_path / f"{convention}.mesh.diag.nc", convention,
                          zbar=LEVELS_NEG, **levels)
    return _load_mesh_nc(path), path


@pytest.mark.parametrize("convention", LOADERS)
@pytest.mark.parametrize("levels, last, bed, source", [
    (dict(zbar_e_bottom=[-60.0, -100.0]), [2, 3], [60.0, 100.0],
     "zbar_e_bottom"),
    (dict(nlevels=[4, 5]), [2, 3], [60.0, 100.0], "nlevels"),
    (dict(nlevels=[4, 5], zbar_e_bottom=[-45.0, -100.0]), [1, 3],
     [45.0, 100.0], "zbar_e_bottom"),
])
def test_loaders_declare_element_bottoms(tmp_path, convention, levels, last,
                                         bed, source):
    mesh, path = _load(tmp_path, convention, ulevels=[1, 2], **levels)
    _check(mesh.metadata, last, bed, source)
    np.testing.assert_array_equal(mesh.metadata["elem_ulevels"], [1, 2])
    assert mesh.metadata["elem_ulevels"].dtype == np.int32
    assert mesh.metadata["meshpath"] == str(path)
    # the cavity (ulevels 2) refuses the depth-linear element table; without
    # it the table builds from the declaration
    with pytest.raises(ValueError, match="ice-shelf cavities"):
        depth_points(mesh, "layer_linear", layout="element")
    del mesh.metadata["elem_ulevels"]
    dp = depth_points(mesh, "layer_linear", layout="element")
    np.testing.assert_array_equal(dp.last, np.maximum(last, 0))
    np.testing.assert_array_equal(dp.bed, bed)


@pytest.mark.parametrize("convention", LOADERS)
def test_loaders_without_levels_declare_nothing(tmp_path, convention):
    mesh, _ = _load(tmp_path, convention)
    assert not set(KEYS) & set(mesh.metadata)


@pytest.mark.parametrize("convention", LOADERS)
def test_loaders_refuse_malformed_levels(tmp_path, convention):
    # nlevels 6 on a five-level mesh: no such level
    with pytest.raises(ValueError) as err:
        _load(tmp_path, convention, nlevels=[6, 4])
    msg = str(err.value)
    assert "'nlevels'" in msg and "mesh.diag" in msg, msg


@pytest.mark.parametrize("convention", ["core2", "comp", "ugrid"])
def test_loaders_refuse_missing_bed(tmp_path, convention):
    # a NaN bed (written or decoded as missing) is refused, never declared
    with pytest.raises(ValueError, match="'zbar_e_bottom'.*element 0"):
        _load(tmp_path, convention, zbar_e_bottom=[np.nan, -100.0])


def test_ascii_mesh_declares_nothing(tmp_path):
    (tmp_path / "nod2d.out").write_text(
        "4\n1 0.0 0.0 0\n2 1.0 0.0 0\n3 1.0 1.0 0\n4 0.0 1.0 0\n")
    (tmp_path / "elem2d.out").write_text("2\n1 2 3\n1 3 4\n")
    mesh = Fesom2Reader.load_mesh(MeshConfig(type="fesom2",
                                             meshpath=str(tmp_path)))
    assert not set(KEYS) & set(mesh.metadata)
