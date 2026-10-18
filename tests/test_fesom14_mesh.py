"""FESOM 1.4 reader: ``load_mesh`` (``kelp.readers.fesom14.reader``).

The mesh comes from one of two sources:

- an ASCII mesh directory (``nod2d.out``, ``elem2d.out``, ``nod3d.out``,
  ``aux3d.out``);
- level-wise data files that carry the node cell polygons
  (``lon_bnds``/``lat_bnds``): the triangles are recovered from the polygons
  and the static wet mask is read from record 0 of ``uo``.

Expected values are written by hand (the five-node ASCII mesh of
``test_fesom14_ascii_mesh``) or derived in the test, independently of the
reader: column depths from the generating column lengths, the Euler-rotation
convention from ``core.rotation.euler_matrix`` alone, circular means from the
file's own longitudes.  All data is synthetic.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from kelp.core.config import MeshConfig, ModeConfig
from kelp.core.geometry import DEG2RAD, METERS_PER_DEG_LAT
from kelp.core.io import VelocityField
from kelp.core.particles import ParticleStatus, make_state
from kelp.core.rotation import euler_matrix
from kelp.numerics.integrators import euler_step
from kelp.numerics.vertical_uv import depth_points
from kelp.readers import get_reader
from kelp.readers.fesom14 import Fesom14Reader
from kelp.readers.fesom2 import Fesom2Reader

from tests.test_fesom14_ascii_mesh import (EXPECTED_IDX, EXPECTED_LEVELS,
                                           EXPECTED_TRI)
from tests.test_fesom14_ascii_mesh import _write as _write_hand_mesh
from tests.test_fesom14_polygons import (_canonical, _patch,
                                         _polygons_from_triangulation)

LEVELS = np.array([0.0, 10.0, 25.0, 50.0])


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _num(x) -> str:
    """A float written with all its digits (exact round trip)."""
    return repr(float(x))


def _write_ascii(meshdir: Path, lon, lat, tri, ncol, levels=LEVELS) -> Path:
    """Write a FESOM 1.4 ASCII mesh: 2-D nodes ``(lon, lat)`` as given, 0-based
    triangles ``tri``, ``ncol[n]`` wet levels below node ``n``.  The 3-D nodes
    are numbered level-major; ``aux3d.out`` has one trailing slot that is dry
    everywhere."""
    meshdir.mkdir(parents=True, exist_ok=True)
    n2d, n_lev = len(lon), len(levels)
    table = np.full((n2d, n_lev + 1), -999, dtype=np.int64)
    nod3d, corresp = [], []
    for k in range(n_lev):
        z = -levels[k] if levels[k] else 0.0
        for n in range(n2d):
            if ncol[n] > k:
                nod3d.append(f"{len(nod3d) + 1} {_num(lon[n])} {_num(lat[n])} "
                             f"{_num(z)} 0")
                table[n, k] = len(nod3d)
                corresp.append(n + 1)
    files = {
        "nod2d.out": [str(n2d)] + [f"{n + 1} {_num(lon[n])} {_num(lat[n])} 0"
                                   for n in range(n2d)],
        "elem2d.out": [str(len(tri))] + [" ".join(str(int(v) + 1) for v in t)
                                         for t in tri],
        "nod3d.out": [str(len(nod3d))] + nod3d,
        "aux3d.out": ([str(n_lev + 1)] + [str(v) for v in table.ravel()]
                      + [str(c) for c in corresp] + ["1", "1", "1"]),
    }
    for name, lines in files.items():
        (meshdir / name).write_text("\n".join(lines) + "\n")
    return meshdir


def _write_levelwise(path: Path, lon, lat, lon_b, lat_b, ncol, *, levels=LEVELS,
                     var="uo", below="nan", n_time=2, deptho=None,
                     polygons=True, depth_values=None) -> Path:
    """Write a level-wise file ``var(time, depth, ncells)`` with node
    coordinates and cell polygons, data in the top ``ncol[n]`` levels of each
    column and ``below`` under the bed: ``"nan"`` (a decoded ``_FillValue``),
    ``"1e30"`` (a raw fill, no ``_FillValue``) or ``"zero"`` (0 written below
    the bed)."""
    n_nodes, n_lev = len(lon), len(levels)
    k = np.arange(n_lev)[:, None]
    wet = k < np.asarray(ncol)[None, :]
    base = 0.1 + 0.01 * k + 0.001 * np.arange(n_nodes)[None, :]
    fill = {"nan": np.nan, "1e30": 1e30, "zero": 0.0}[below]
    vals = np.stack([np.where(wet, base + t, fill) for t in range(n_time)])
    zvals = levels if depth_values is None else depth_values
    ds = xr.Dataset(
        {var: (("time", "depth", "ncells"), vals.astype(np.float32))},
        coords={
            "time": ("time", 15.5 + 30.0 * np.arange(n_time),
                     {"units": "days since 1951-1-1",
                      "calendar": "proleptic_gregorian"}),
            "depth": ("depth", np.asarray(zvals, dtype=np.float64),
                      {"units": "m", "positive": "down"}),
            "lon": ("ncells", np.asarray(lon, dtype=np.float64),
                    {"standard_name": "longitude", "bounds": "lon_bnds"}),
            "lat": ("ncells", np.asarray(lat, dtype=np.float64),
                    {"standard_name": "latitude", "bounds": "lat_bnds"}),
        },
    )
    if polygons:
        ds["lon_bnds"] = (("ncells", "vertices"), lon_b)
        ds["lat_bnds"] = (("ncells", "vertices"), lat_b)
    if deptho is not None:
        ds["deptho"] = ("ncells", np.asarray(deptho, dtype=np.float32))
    enc = {var: {"_FillValue": 1e20 if below == "nan" else None}}
    ds.to_netcdf(path, encoding=enc)
    return path


def _write_deptho(path: Path, lon, lat, deptho) -> Path:
    """A static file with ``deptho(ncells)`` and the node coordinates."""
    ds = xr.Dataset(
        {"deptho": ("ncells", np.asarray(deptho, dtype=np.float32))},
        coords={"lon": ("ncells", np.asarray(lon, dtype=np.float64)),
                "lat": ("ncells", np.asarray(lat, dtype=np.float64))},
    )
    ds.to_netcdf(path)
    return path


def _cfg(path, **kw) -> MeshConfig:
    return MeshConfig(type="fesom14", meshpath=str(path), **kw)


def _load(path, mode=None, **kw):
    return Fesom14Reader.load_mesh(_cfg(path, **kw), mode_cfg=mode)


def _wrap(x):
    return ((np.asarray(x, dtype=np.float64) + 180.0) % 360.0) - 180.0


def _circular_mean(lon_deg):
    r = np.deg2rad(lon_deg)
    return np.rad2deg(np.arctan2(np.sin(r).mean(axis=-1), np.cos(r).mean(axis=-1)))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def island():
    """The 6x6 patch with an island of ``test_fesom14_polygons``: 36 nodes in
    shuffled order, the generating triangles, and a column length 1..4 per
    node."""
    lon, lat, tri, _ = _patch(10.0, 40.0)
    ncol = 1 + (np.arange(lon.size) * 7) % 4
    lon_b, lat_b = _polygons_from_triangulation(lon, lat, tri, nv=16)
    return lon, lat, tri, ncol, lon_b, lat_b


def _band(lats, dlon=5.0, lon0=0.0):
    """A global band of nodes on a ``dlon`` x lat grid, longitudes
    ``lon0 .. lon0 + 360`` (exclusive), triangulated counter-clockwise and
    closed across the seam."""
    lons = lon0 + np.arange(0.0, 360.0, dlon)
    nx, ny = lons.size, len(lats)
    lon = np.tile(lons, ny)
    lat = np.repeat(np.asarray(lats, dtype=np.float64), nx)
    tri = []
    for j in range(ny - 1):
        for i in range(nx):
            i1 = (i + 1) % nx
            sw, se = j * nx + i, j * nx + i1
            nw, ne = sw + nx, se + nx
            tri += [(sw, se, ne), (sw, ne, nw)]
    return lon, lat, np.array(tri)


def _grid(lon0, lat0, nx, ny, d=1.0):
    """A regular ``nx`` x ``ny`` grid of nodes ``d`` degrees apart, starting
    at (lon0, lat0), triangulated counter-clockwise; longitudes as given
    (no wrapping)."""
    lon = np.tile(lon0 + d * np.arange(nx), ny)
    lat = np.repeat(lat0 + d * np.arange(ny), nx)
    tri = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            sw = j * nx + i
            se, nw = sw + 1, sw + nx
            tri += [(sw, se, nw + 1), (sw, nw + 1, nw)]
    return lon, lat, np.array(tri)


# ---------------------------------------------------------------------------
# Registration and the engine contract
# ---------------------------------------------------------------------------

def test_registered_with_level_contract():
    assert get_reader("fesom14") is Fesom14Reader
    assert Fesom14Reader.native_uv_vertical == "level"
    # a fresh process finds the reader by name alone (as the CLI does)
    code = ("from kelp.readers import get_reader; "
            "r = get_reader('fesom14'); print(r.__name__, r.native_uv_vertical)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True,
                         env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
    assert out.stdout.split() == ["Fesom14Reader", "level"]


# ---------------------------------------------------------------------------
# ASCII mesh
# ---------------------------------------------------------------------------

def test_load_ascii_mesh(tmp_path):
    """The hand-written five-node mesh: columns of 4, 3, 2, 3, 1 levels at
    0, 10, 25, 50 m."""
    mesh = _load(_write_hand_mesh(tmp_path / "mesh"))

    assert mesh.vertical_coord == "zlevel"
    assert mesh.sigma.dtype == np.float64
    np.testing.assert_array_equal(mesh.sigma, [0.0, 10.0, 25.0, 50.0])
    np.testing.assert_array_equal(mesh.sigma, EXPECTED_LEVELS)
    # the bed of each column: its last wet level, exactly
    assert mesh.depth.dtype == np.float64
    assert mesh.depth.tolist() == [50.0, 25.0, 10.0, 25.0, 0.0]
    last = mesh.metadata["uv_data_last"]
    assert last.dtype == np.int32 and last.shape == (5,)
    assert last.tolist() == [3, 2, 1, 2, 0]

    md = mesh.metadata
    assert md["fesom14_source"] == "ascii"
    np.testing.assert_array_equal(md["fesom14_idx"], EXPECTED_IDX)
    wet = md["fesom14_wet"]
    assert wet.dtype == bool and wet.shape == (4, 5) and wet.flags.c_contiguous
    np.testing.assert_array_equal(wet, EXPECTED_IDX.T >= 0)
    levels = md["fesom14_levels"]
    assert levels.dtype == np.float64 and levels is not mesh.sigma
    np.testing.assert_array_equal(levels, EXPECTED_LEVELS)
    np.testing.assert_array_equal(md["fesom14_lon"], [10.0, 11.0, 10.5, 11.5, 12.0])
    np.testing.assert_array_equal(md["fesom14_lat"], [50.0, 50.0, 51.0, 51.0, 50.0])

    np.testing.assert_array_equal(mesh.nodes_lon, [10.0, 11.0, 10.5, 11.5, 12.0])
    np.testing.assert_array_equal(mesh.nodes_lat, [50.0, 50.0, 51.0, 51.0, 50.0])
    np.testing.assert_array_equal(mesh.elem_nodes, EXPECTED_TRI)
    assert mesh.name == "fesom14" and mesh.rotation_angles is None
    assert mesh._vec_rot_factors is None
    assert mesh.r_earth == 6_367_500.0
    # node 1 (0-based) lies in all three triangles
    assert mesh.node_to_elems_count.tolist() == [1, 3, 2, 2, 1]


def test_mesh_facts_feed_the_depth_linear_table(tmp_path):
    """The declaration is a new int32 array the engine accepts as given; the
    full level table survives a cut of ``mesh.sigma``; the static facts are
    read-only."""
    mesh = _load(_write_hand_mesh(tmp_path / "mesh"))
    md = mesh.metadata
    pts = depth_points(mesh, "level")
    np.testing.assert_array_equal(pts.last, [3, 2, 1, 2, 0])
    np.testing.assert_array_equal(pts.p, [0.0, 10.0, 25.0, 50.0])

    mesh.sigma = mesh.sigma[:2]           # what a max_levels cut does
    np.testing.assert_array_equal(md["fesom14_levels"], EXPECTED_LEVELS)
    for key in ("fesom14_levels", "fesom14_wet", "fesom14_idx",
                "fesom14_lon", "fesom14_lat"):
        assert not md[key].flags.writeable, key

    # two loads never share the declaration
    other = _load(_write_hand_mesh(tmp_path / "mesh2"))
    assert other.metadata["uv_data_last"] is not md["uv_data_last"]


# ---------------------------------------------------------------------------
# Mesh from cell polygons
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("below", ["nan", "1e30"])
def test_load_mesh_from_polygons(tmp_path, island, below):
    lon, lat, tri, ncol, lon_b, lat_b = island
    true_depth = LEVELS[ncol - 1]
    expected_wet = np.arange(4)[:, None] < ncol[None, :]

    def check(mesh, *, deptho_file):
        md = mesh.metadata
        assert md["fesom14_source"] == "polygons"
        assert "fesom14_idx" not in md
        np.testing.assert_array_equal(mesh.elem_nodes, _canonical(tri))
        assert mesh.elem_nodes.dtype == np.int32
        np.testing.assert_array_equal(md["fesom14_wet"], expected_wet)
        assert md["fesom14_wet"].dtype == bool
        assert md["fesom14_wet"].flags.c_contiguous
        assert mesh.vertical_coord == "zlevel"
        np.testing.assert_array_equal(mesh.sigma, LEVELS)
        np.testing.assert_array_equal(md["fesom14_levels"], LEVELS)
        assert mesh.depth.tolist() == true_depth.tolist()
        assert md["uv_data_last"].dtype == np.int32
        np.testing.assert_array_equal(md["uv_data_last"], ncol - 1)
        np.testing.assert_array_equal(md["fesom14_lon"], lon)
        np.testing.assert_array_equal(md["fesom14_lat"], lat)
        np.testing.assert_array_equal(mesh.nodes_lon, lon)
        rep = md["fesom14_polygon_report"]
        assert rep["n_pinch_triangles"] == 0
        assert rep["n_boundary_loops"] == 2
        assert set(rep) >= {"n_full_slot_polygons", "n_duplicate_vertices",
                            "n_pinch_nodes", "n_components",
                            "euler_characteristic"}
        assert md["fesom14_deptho_file"] == deptho_file

    # deptho present (a static file next to the data) and agreeing: accepted
    d1 = tmp_path / "with_deptho"
    d1.mkdir()
    _write_levelwise(d1 / "uo_1951-1960.nc", lon, lat, lon_b, lat_b, ncol, below=below)
    dfile = _write_deptho(d1 / "deptho_fx.nc", lon, lat, true_depth)
    check(_load(d1), deptho_file=str(dfile))

    # deptho absent: the mask of record 0 is used as it is
    f2 = _write_levelwise(tmp_path / "uo_alone.nc", lon, lat, lon_b, lat_b, ncol,
                          below=below)
    check(_load(f2), deptho_file=None)

    # deptho disagreeing with the record-0 mask: data written as 0 below the
    # bed looks wet everywhere, deptho gives the real beds -> refused
    d3 = tmp_path / "zero_below_bed"
    d3.mkdir()
    _write_levelwise(d3 / "uo_1951-1960.nc", lon, lat, lon_b, lat_b, ncol, below="zero")
    _write_deptho(d3 / "deptho_fx.nc", lon, lat, true_depth)
    with pytest.raises(ValueError) as info:
        _load(d3)
    msg = str(info.value)
    n = int(np.flatnonzero(ncol < 4)[0])
    assert re.search(rf"\bnode {n}\b", msg), msg
    assert f"deptho = {true_depth[n]:g} m" in msg, msg
    assert "50 m" in msg and "deptho_fx.nc" in msg and "'uo'" in msg, msg
    assert "below the bed" in msg, msg


def test_polygons_deptho_in_the_data_file(tmp_path, island):
    """In a single file, a ``deptho`` variable of the same file is checked."""
    lon, lat, tri, ncol, lon_b, lat_b = island
    good = _write_levelwise(tmp_path / "good.nc", lon, lat, lon_b, lat_b, ncol,
                            deptho=LEVELS[ncol - 1])
    mesh = _load(good)
    assert mesh.metadata["fesom14_deptho_file"] == str(good)
    bad_depth = LEVELS[ncol - 1].copy()
    bad_depth[7] += 15.0
    bad = _write_levelwise(tmp_path / "bad.nc", lon, lat, lon_b, lat_b, ncol,
                           deptho=bad_depth)
    with pytest.raises(ValueError, match=r"node 7\b.*deptho") as info:
        _load(bad)
    # deptho DEEPER than the last level with data: data are missing where
    # deptho says ocean -- not the "0 written below the bed" explanation
    msg = str(info.value)
    assert "deeper" in msg and "missing" in msg, msg
    assert "below the bed" not in msg, msg


def test_polygons_without_deptho_cannot_see_zero_below_bed(tmp_path, island):
    """The documented limit: without ``deptho``, data written as 0 below the
    bed gives an all-wet mask and is accepted (only ``deptho`` or an ASCII
    mesh can expose it)."""
    lon, lat, tri, ncol, lon_b, lat_b = island
    f = _write_levelwise(tmp_path / "uo.nc", lon, lat, lon_b, lat_b, ncol,
                         below="zero")
    mesh = _load(f)
    assert mesh.metadata["fesom14_wet"].all()
    assert (mesh.depth == 50.0).all()


@pytest.mark.parametrize("case, needles", [
    ("nan_at_node", ["node 3", "deptho", "no value"]),
    ("moved_node", ["node 5", "deptho_fx.nc", "coordinates"]),
    ("wrong_size", ["deptho_fx.nc", "35", "36"]),
])
def test_polygons_deptho_refusals(tmp_path, island, case, needles):
    lon, lat, tri, ncol, lon_b, lat_b = island
    d = tmp_path / case
    d.mkdir()
    _write_levelwise(d / "uo.nc", lon, lat, lon_b, lat_b, ncol)
    depth = LEVELS[ncol - 1].astype(float)
    dlon, dlat = lon.copy(), lat.copy()
    if case == "nan_at_node":
        depth[3] = np.nan
    elif case == "moved_node":
        dlon[5] += 1e-4
    else:
        depth, dlon, dlat = depth[:-1], dlon[:-1], dlat[:-1]
    _write_deptho(d / "deptho_fx.nc", dlon, dlat, depth)
    with pytest.raises(ValueError) as info:
        _load(d)
    for needle in needles:
        assert needle in str(info.value), (needle, str(info.value))


@pytest.mark.parametrize("data_east", [False, True])
def test_polygons_deptho_rounding_and_longitude_convention(tmp_path, data_east):
    """``deptho`` node coordinates that differ by 1e-9 deg or by the
    longitude convention (0..360 against -180..180, either way round) are
    the same nodes."""
    lon, lat, tri, _ = _patch(-20.0, 40.0)          # west of Greenwich
    assert (lon < 0.0).all()
    ncol = 1 + np.arange(lon.size) % 4
    lon_b, lat_b = _polygons_from_triangulation(lon, lat, tri, nv=16)
    d = tmp_path / "d"
    d.mkdir()
    if data_east:       # the data in 0..360, deptho in -180..180
        lon_data, lon_deptho = lon + 360.0, lon + 1e-9
    else:               # the data in -180..180, deptho in 0..360
        lon_data, lon_deptho = lon, lon + 360.0 + 1e-9
    _write_levelwise(d / "uo.nc", lon_data, lat, lon_b, lat_b, ncol)
    _write_deptho(d / "deptho_fx.nc", lon_deptho, lat - 1e-9, LEVELS[ncol - 1])
    mesh = _load(d)
    assert mesh.metadata["fesom14_deptho_file"] == str(d / "deptho_fx.nc")
    np.testing.assert_array_equal(mesh.depth, LEVELS[ncol - 1])


def test_polygons_mask_errors_name_variable_and_file(tmp_path, island):
    lon, lat, tri, ncol, lon_b, lat_b = island
    f = tmp_path / "uo_hole.nc"
    _write_levelwise(f, lon, lat, lon_b, lat_b, np.full(lon.size, 4))
    with xr.open_dataset(f) as ds:
        ds = ds.load()
    ds["uo"][0, 1, 9] = np.nan                  # a hole in node 9's column
    f2 = tmp_path / "uo_hole2.nc"
    ds.to_netcdf(f2, encoding={"uo": {"_FillValue": 1e20}})
    with pytest.raises(ValueError) as info:
        _load(f2)
    msg = str(info.value)
    assert "'uo'" in msg and "uo_hole2.nc" in msg and "node 9" in msg, msg
    assert "record 0" in msg, msg


def test_polygons_var_u_from_mode(tmp_path, island):
    """The mask comes from ``mode.var_u`` when it is set."""
    lon, lat, tri, ncol, lon_b, lat_b = island
    f = _write_levelwise(tmp_path / "u.nc", lon, lat, lon_b, lat_b, ncol,
                         var="u_east")
    mesh = _load(f, mode=ModeConfig(var_u="u_east"))
    np.testing.assert_array_equal(mesh.metadata["uv_data_last"], ncol - 1)
    with pytest.raises(ValueError, match=r"'uo'.*mode\.var_u"):
        _load(f)


@pytest.mark.parametrize("depth_values, needles", [
    ([5.0, 10.0, 25.0, 50.0], ["level 0", "5", "surface"]),
    ([0.0, -10.0, -25.0, -50.0], ["level 1", "positive down"]),
    ([0.0, 10.0, np.nan, 50.0], ["level 2", "nan"]),
])
def test_polygons_level_axis_refused(tmp_path, island, depth_values, needles):
    lon, lat, tri, ncol, lon_b, lat_b = island
    f = _write_levelwise(tmp_path / "uo.nc", lon, lat, lon_b, lat_b, ncol,
                         depth_values=depth_values)
    with pytest.raises(ValueError) as info:
        _load(f)
    msg = str(info.value)
    assert "uo.nc" in msg and "'depth'" in msg, msg
    for needle in needles:
        assert needle in msg, (needle, msg)


def test_polygons_recovery_errors_name_the_file(tmp_path, island):
    lon, lat, tri, ncol, lon_b, lat_b = island
    lon_b = lon_b.copy()
    lon_b[4, :] = lon_b[4, 0]                    # node 4's polygon collapsed
    f = _write_levelwise(tmp_path / "uo_bad_poly.nc", lon, lat, lon_b, lat_b, ncol)
    with pytest.raises(ValueError, match=r"uo_bad_poly\.nc.*cell polygons"):
        _load(f)


# ---------------------------------------------------------------------------
# What mesh.meshpath may be
# ---------------------------------------------------------------------------

def test_meshpath_kinds(tmp_path, island):
    lon, lat, tri, ncol, lon_b, lat_b = island

    # a directory with nod3d.out: the ASCII mesh (data files next to it are
    # not read)
    asc = _write_hand_mesh(tmp_path / "ascii")
    _write_levelwise(asc / "uo.nc", lon, lat, lon_b, lat_b, ncol)
    assert _load(asc).metadata["fesom14_source"] == "ascii"

    # one level-wise data file
    f = _write_levelwise(tmp_path / "uo_195101-196012.nc", lon, lat, lon_b, lat_b,
                         ncol)
    m = _load(f)
    assert m.metadata["fesom14_source"] == "polygons"
    assert m.metadata["fesom14_mask_file"] == str(f)

    # a directory of level-wise files: the mask comes from the first file in
    # name order that holds uo; other variables and files are not used
    d = tmp_path / "data"
    d.mkdir()
    for name in ("vo_195101-196012.nc", "uo_196101-197012.nc",
                 "uo_195101-196012.nc", "thetao_195101-196012.nc"):
        _write_levelwise(d / name, lon, lat, lon_b, lat_b, ncol,
                         var=name.split("_")[0])
    (d / "README.txt").write_text("not a data file\n")
    m = _load(d)
    assert m.metadata["fesom14_source"] == "polygons"
    assert m.metadata["fesom14_mask_file"] == str(d / "uo_195101-196012.nc")
    np.testing.assert_array_equal(m.elem_nodes, _canonical(tri))


def _raw_nodes3d_file(path: Path) -> Path:
    raw = np.zeros((1, 13), np.float32)
    xr.Dataset({"uo": (("time", "nodes_3d"), raw)}).to_netcdf(path)
    return path


@pytest.mark.parametrize("kind, error, needles", [
    ("unset", ValueError, ["mesh.meshpath", "not set"]),
    ("missing", FileNotFoundError, ["nowhere"]),
    ("empty_dir", ValueError, ["nod3d.out", "*.nc"]),
    ("fesom2_dir", ValueError, ["nod3d.out", "fesom2"]),
    ("text_file", ValueError, ["notes.txt", "NetCDF"]),
    ("raw_file", ValueError, ["raw.nc", "nodes_3d", "ASCII mesh"]),
    ("no_polygons", ValueError, ["uo_nopoly.nc", "lon_bnds"]),
    ("no_uo_in_dir", ValueError, ["'uo'", "mode.var_u"]),
    ("no_lon", ValueError, ["uo_nolon.nc", "'lon'"]),
    ("flat_uo", ValueError, ["uo_flat.nc", "('time', 'ncells')", "mesh.meshpath"]),
])
def test_meshpath_refused(tmp_path, island, kind, error, needles):
    lon, lat, tri, ncol, lon_b, lat_b = island
    if kind == "unset":
        path = ""
    elif kind == "missing":
        path = tmp_path / "nowhere"
    elif kind == "empty_dir":
        path = tmp_path / "empty"
        path.mkdir()
    elif kind == "fesom2_dir":
        path = tmp_path / "f2"
        path.mkdir()
        (path / "nod2d.out").write_text("1\n1 0.0 0.0 0\n")
        (path / "elem2d.out").write_text("0\n")
    elif kind == "text_file":
        path = tmp_path / "notes.txt"
        path.write_text("hello\n")
    elif kind == "raw_file":
        path = _raw_nodes3d_file(tmp_path / "raw.nc")
    elif kind == "no_polygons":
        path = _write_levelwise(tmp_path / "uo_nopoly.nc", lon, lat, lon_b, lat_b,
                                ncol, polygons=False)
    elif kind == "flat_uo":
        path = tmp_path / "uo_flat.nc"
        xr.Dataset({"uo": (("time", "ncells"), np.zeros((1, lon.size)))},
                   coords={"lon": ("ncells", lon), "lat": ("ncells", lat)}
                   ).to_netcdf(path)
    elif kind == "no_uo_in_dir":
        path = tmp_path / "d"
        path.mkdir()
        _write_levelwise(path / "vo.nc", lon, lat, lon_b, lat_b, ncol, var="vo")
    else:
        path = _write_levelwise(tmp_path / "uo_nolon.nc", lon, lat, lon_b, lat_b,
                                ncol)
        with xr.open_dataset(path) as ds:
            ds = ds.load()
        path = tmp_path / "uo_nolon2.nc"
        needles = ["uo_nolon2.nc", "'lon'"]
        ds.drop_vars("lon").to_netcdf(path, encoding={"uo": {"_FillValue": 1e20}})
    with pytest.raises(error) as info:
        _load(path)
    for needle in needles:
        assert needle in str(info.value), (needle, str(info.value))


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------

def _rotate_into_frame(lon, lat, angles):
    """Rotated-frame coordinates, derived here from ``euler_matrix`` alone:
    ``v_rot = R @ v_geo`` on unit vectors."""
    R = euler_matrix(*angles)
    lo, la = np.deg2rad(lon), np.deg2rad(lat)
    v = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    x, y, z = R @ v
    # atan2, not arcsin: a node lies on the rotated pole, where arcsin(z)
    # loses half the digits
    return np.rad2deg(np.arctan2(y, x)), np.rad2deg(np.arctan2(z, np.hypot(x, y)))


def test_input_rotation(tmp_path):
    """An ASCII mesh stored in a rotated frame and the frame's Euler angles in
    ``mesh.input_rotation``: the geographic coordinates come back to 1e-10
    deg."""
    angles = (30.0, 40.0, -60.0)
    geo_lon, geo_lat, tri = _band(np.arange(-80.0, 81.0, 10.0).tolist() + [86.0],
                                  dlon=10.0, lon0=-180.0)
    rot_lon, rot_lat = _rotate_into_frame(geo_lon, geo_lat, angles)
    ncol = np.full(geo_lon.size, 2)
    meshdir = _write_ascii(tmp_path / "rot", rot_lon, rot_lat, tri, ncol)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        mesh = _load(meshdir, input_rotation=list(angles))
    md = mesh.metadata
    for got_lon, got_lat in ((mesh.nodes_lon, mesh.nodes_lat),
                             (md["fesom14_lon"], md["fesom14_lat"])):
        assert np.abs(_wrap(got_lon - geo_lon)).max() < 1e-10
        assert np.abs(got_lat - geo_lat).max() < 1e-10
    # the element centres are the geographic ones
    np.testing.assert_allclose(mesh.elem_center_lat, geo_lat[tri].mean(axis=1),
                               atol=1e-10)

    # without the angles, the stored (rotated) coordinates are taken as they are
    with pytest.warns(RuntimeWarning, match="pole"):   # a node on the rotated pole
        plain = _load(meshdir)
    assert np.abs(_wrap(plain.nodes_lon - rot_lon)).max() < 1e-12
    assert np.abs(plain.nodes_lat - geo_lat).max() > 1.0
    # the inverse convention (R instead of R^T) would not give them back
    bad_lon, bad_lat = _rotate_into_frame(rot_lon, rot_lat, angles)
    assert np.abs(bad_lat - geo_lat).max() > 1.0

    # combined with the tracking frame: nodes in the tracking frame,
    # metadata geographic
    both = _load(meshdir, input_rotation=list(angles), rotation="fesom2")
    assert np.abs(_wrap(both.metadata["fesom14_lon"] - geo_lon)).max() < 1e-10
    trk_lon, trk_lat = _rotate_into_frame(geo_lon, geo_lat, (50.0, 15.0, -90.0))
    assert np.abs(_wrap(both.nodes_lon - trk_lon)).max() < 1e-9
    assert np.abs(both.nodes_lat - trk_lat).max() < 1e-9


def test_input_rotation_refused_for_polygons(tmp_path, island):
    lon, lat, tri, ncol, lon_b, lat_b = island
    f = _write_levelwise(tmp_path / "uo.nc", lon, lat, lon_b, lat_b, ncol)
    with pytest.raises(ValueError, match=r"input_rotation.*ASCII"):
        _load(f, input_rotation=[50.0, 15.0, -90.0])


@pytest.mark.parametrize("value", [
    [50.0, 15.0], [50.0, 15.0, -90.0, 0.0], [50.0, float("nan"), -90.0],
    [50.0, float("inf"), 0.0], "50 15 -90", "123", [True, 0.0, 0.0], 50.0,
    ["a", "b", "c"],
])
def test_input_rotation_config_refused(value):
    with pytest.raises(ValueError, match="mesh.input_rotation"):
        MeshConfig(type="fesom14", meshpath="m", input_rotation=value)


def test_input_rotation_config_accepted():
    cfg = MeshConfig(type="fesom14", meshpath="m", input_rotation=[50, "15", -90.0])
    assert cfg.input_rotation == [50.0, 15.0, -90.0]
    assert all(type(v) is float for v in cfg.input_rotation)
    assert MeshConfig().input_rotation is None


def test_rotated_mesh_warning(tmp_path):
    """A global ASCII mesh with no node north of 85 N and no
    ``mesh.input_rotation`` looks like a mesh stored in a rotated frame."""
    lats = np.arange(-80.0, 81.0, 10.0)
    lon, lat, tri = _band(lats)
    ncol = np.full(lon.size, 2)
    meshdir = _write_ascii(tmp_path / "global", lon, lat, tri, ncol)
    with pytest.warns(UserWarning, match=r"input_rotation") as rec:
        _load(meshdir)
    msg = str(rec[0].message)
    assert "85" in msg and str(meshdir) in msg

    quiet = []
    # the same mesh with its angles given
    quiet.append((meshdir, {"input_rotation": [0.0, 0.0, 0.0]}))
    # a global mesh that reaches the Arctic
    lon2, lat2, tri2 = _band(np.r_[lats, 86.0])
    quiet.append((_write_ascii(tmp_path / "arctic", lon2, lat2, tri2,
                               np.full(lon2.size, 2)), {}))
    # a regional mesh across the dateline (wide in raw longitude, not global)
    lon3, lat3, tri3 = _grid(178.0, 40.0, 5, 4)
    quiet.append((_write_ascii(tmp_path / "regional", _wrap(lon3), lat3, tri3,
                               np.full(lon3.size, 2)), {}))
    for path, kw in quiet:
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            _load(path, **kw)


def test_tracking_frame_factors(tmp_path):
    """``mesh.rotation: fesom2``: the same rotated coordinates and vector
    factors as the fesom2 reader builds for the same nodes and triangles."""
    lon, lat, tri = _grid(-30.0, 55.0, 6, 5)
    ncol = 1 + np.arange(lon.size) % 3
    f14 = _load(_write_ascii(tmp_path / "f14", lon, lat, tri, ncol),
                rotation="fesom2")

    f2dir = tmp_path / "f2"
    f2dir.mkdir()
    (f2dir / "nod2d.out").write_text(
        f"{lon.size}\n" + "".join(f"{n + 1} {_num(lon[n])} {_num(lat[n])} 0\n"
                                  for n in range(lon.size)))
    (f2dir / "elem2d.out").write_text(
        f"{len(tri)}\n" + "".join(" ".join(str(v + 1) for v in t) + "\n" for t in tri))
    f2 = Fesom2Reader.load_mesh(MeshConfig(type="fesom2", meshpath=str(f2dir),
                                           rotation="fesom2"))

    assert f14.rotation_angles == f2.rotation_angles == (50.0, 15.0, -90.0)
    for layout in ("node", "element"):
        c14, s14 = f14._vec_rot_factors[layout]
        c2, s2 = f2._vec_rot_factors[layout]
        np.testing.assert_array_equal(c14, c2)
        np.testing.assert_array_equal(s14, s2)
    np.testing.assert_array_equal(f14.nodes_lon, f2.nodes_lon)
    np.testing.assert_array_equal(f14.nodes_lat, f2.nodes_lat)
    np.testing.assert_array_equal(f14.elem_center_lon, f2.elem_center_lon)
    np.testing.assert_array_equal(f14.elem_center_lat, f2.elem_center_lat)
    # the metadata keeps the geographic coordinates
    np.testing.assert_array_equal(f14.metadata["fesom14_lon"], lon)
    np.testing.assert_array_equal(f14.metadata["fesom14_lat"], lat)
    assert np.abs(_wrap(f14.nodes_lon - lon)).max() > 1.0


def test_ascii_mesh_dateline(tmp_path):
    """An ASCII mesh across the dateline written with longitudes 0..360:
    element centres by circular mean, and a particle crosses the seam."""
    lon360, lat, tri = _grid(178.0, 40.0, 5, 4)          # 178 .. 182 deg E
    assert lon360.max() > 180.0
    ncol = np.full(lon360.size, 2)
    mesh = _load(_write_ascii(tmp_path / "seam", lon360, lat, tri, ncol))

    assert mesh.nodes_lon.min() >= -180.0 and mesh.nodes_lon.max() < 180.0
    np.testing.assert_array_equal(mesh.nodes_lon, _wrap(lon360))
    np.testing.assert_array_equal(mesh.metadata["fesom14_lon"], _wrap(lon360))
    # the circular mean of the file's own longitudes (no seam among them)
    expected = _wrap(_circular_mean(lon360[tri]))
    assert np.abs(_wrap(mesh.elem_center_lon - expected)).max() < 1e-9
    # every centre lies inside its triangle's longitude range, not 180 deg off
    assert np.abs(_wrap(mesh.elem_center_lon - lon360[tri].mean(axis=1))).max() < 1e-3
    np.testing.assert_allclose(mesh.elem_center_lat, lat[tri].mean(axis=1))

    # a particle advected east by a constant 10 m/s crosses 180 deg
    n_nodes = mesh.n_nodes
    vel = VelocityField(u=np.full(n_nodes, 10.0), v=np.zeros(n_nodes),
                        eta=np.zeros(n_nodes), w=None, uv_layout="node")
    lon0, lat0, dt, n_steps = 179.3, 41.5, 600.0, 20
    state = make_state(np.array([lon0]), np.array([lat0]))
    for _ in range(n_steps):
        state = euler_step(state, vel, mesh, dt, time_dir=1.0,
                           uv_interp="barycentric")
    expected_lon = _wrap(lon0 + n_steps * dt * 10.0
                         / (np.cos(lat0 * DEG2RAD) * METERS_PER_DEG_LAT))
    assert state.status[0] == ParticleStatus.ALIVE
    assert state.lon[0] < 0.0
    assert state.lon[0] == pytest.approx(float(expected_lon), abs=1e-9)
    assert state.lat[0] == pytest.approx(lat0, abs=1e-12)
