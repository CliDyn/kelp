"""FESOM 1.4 reader: ``prepare(mesh, ds)`` (``kelp.readers.fesom14.reader``).

``open_dataset`` checks only what the files say about themselves; ``prepare``
checks a dataset against the mesh: the horizontal size of every variable,
the full level axis against the mesh's full level table (``mesh.sigma`` may
already be cut by ``max_levels``), the node coordinates when the data carry
them, and the wet mask of record 0 of every variable against the mesh's
static mask (the ASCII mesh's 3-D node index table, or the mask a mesh from
cell polygons was built with).  It changes nothing, so it can run any number
of times; the engine runs it again for every other dataset.

Meshes: the five-node ASCII mesh of ``test_fesom14_ascii_mesh`` (columns of
4, 3, 2, 3, 1 levels at 0, 10, 25, 50 m; 13 3-D nodes), the same nodes with
four levels everywhere, and the 36-node island patch of
``test_fesom14_polygons`` as a mesh from cell polygons.  Data files are
written here, one per variable: level-wise ``var(time, depth, ncells)`` with
NaN (a decoded ``_FillValue``) below the bed, or ``var(time, nodes_3d)``
vectors; three daily snapshots read with ``data.time_stamp: instant``.
Expected nodes and levels are written by hand.  All data is synthetic.
"""
from __future__ import annotations

import contextlib
import gc
import io
from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")
pytest.importorskip("cftime")

from kelp import tracking  # noqa: E402
from kelp.core.config import MeshConfig, ModeConfig, NumericsConfig  # noqa: E402
from kelp.core.io import VelocityField  # noqa: E402
from kelp.core.mesh import find_nearest_nodes  # noqa: E402
from kelp.core.particles import make_state  # noqa: E402
from kelp.readers.fesom14 import Fesom14Reader  # noqa: E402

from tests.test_fesom14_ascii_mesh import EXPECTED_IDX, EXPECTED_TRI  # noqa: E402
from tests.test_fesom14_ascii_mesh import _write as _write_hand_mesh  # noqa: E402
from tests.test_fesom14_mesh import _write_ascii, _write_levelwise  # noqa: E402
from tests.test_fesom14_polygons import (  # noqa: E402
    _patch, _polygons_from_triangulation)

LEVELS = np.array([0.0, 10.0, 25.0, 50.0])

#: the hand-written ASCII mesh (test_fesom14_ascii_mesh)
HAND_LON = np.array([10.0, 11.0, 10.5, 11.5, 12.0])
HAND_LAT = np.array([50.0, 50.0, 51.0, 51.0, 50.0])
HAND_NCOL = np.array([4, 3, 2, 3, 1])
N3D = 13

MODE_3D = ModeConfig(Fdrift3D=True)
MODE_3D_T = ModeConfig(Fdrift3D=True, sample_vars=["thetao"])

#: the base of each variable's values
BASE = {"uo": 0.1, "vo": 0.2, "wo": 0.3, "thetao": 4.0, "zos": 0.5}


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _time(nc, n_time):
    nc.createDimension("time", None)
    t = nc.createVariable("time", "f8", ("time",))
    t.units = "days since 2000-01-01"
    t.calendar = "standard"
    t[:] = np.arange(n_time, dtype=np.float64)


def _write_var(path: Path, var: str, lon, lat, ncol, *, levels=LEVELS,
               n_time=3, below="nan", coords=True, perm=None, edit=None,
               two_d=False, hdim="ncells") -> Path:
    """A level-wise file ``var(time, depth, <hdim>)``: data in the top
    ``ncol[n]`` levels of each column, ``below`` under the bed (``"nan"``: a
    decoded ``_FillValue``; ``"zero"``: 0 written below the bed); with
    ``two_d``, ``var(time, <hdim>)``.  ``perm`` reorders the nodes (values
    and coordinates alike); ``edit(values)`` plants a defect in the
    ``(time, depth, ncells)`` values before they are written."""
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    ncol = np.asarray(ncol)
    n_nodes, n_lev = lon.size, len(levels)
    t = np.arange(n_time)[:, None, None]
    k = np.arange(n_lev)[None, :, None]
    n = np.arange(n_nodes)[None, None, :]
    wet = k < ncol[None, None, :]
    fill = {"nan": np.nan, "zero": 0.0}[below]
    vals = np.where(wet, BASE[var] + t + 0.01 * k + 0.001 * n, fill)
    if edit is not None:
        edit(vals)
    if perm is not None:
        vals, lon, lat = vals[..., perm], lon[perm], lat[perm]
    with netCDF4.Dataset(str(path), "w") as nc:
        _time(nc, n_time)
        nc.createDimension(hdim, n_nodes)
        if two_d:
            v = nc.createVariable(var, "f4", ("time", hdim),
                                  fill_value=np.float32(1e20))
            v[:] = np.ma.masked_invalid(vals[:, 0, :])
        else:
            nc.createDimension("depth", n_lev)
            z = nc.createVariable("depth", "f8", ("depth",))
            z.units, z.positive = "m", "down"
            z[:] = np.asarray(levels, dtype=np.float64)
            v = nc.createVariable(var, "f4", ("time", "depth", hdim),
                                  fill_value=np.float32(1e20))
            v[:] = np.ma.masked_invalid(vals)
        if coords:
            for name, values in (("lon", lon), ("lat", lat)):
                c = nc.createVariable(name, "f8", (hdim,))
                c[:] = values
    return path


def _write_raw(path: Path, var: str, size: int, *, n_time=3, two_d=False,
               fill_at=None, coords=None) -> Path:
    """A file ``var(time, nodes_3d)`` with ``size`` entries (with
    ``two_d``: ``var(time, nodes_2d)``); ``fill_at`` = (record, entry) of a
    raw 1e30; ``coords`` = (lon, lat) written over the same dimension."""
    vals = BASE[var] + np.arange(n_time)[:, None] + 0.001 * np.arange(size)
    if fill_at is not None:
        vals[fill_at] = 1e30
    dim = "nodes_2d" if two_d else "nodes_3d"
    with netCDF4.Dataset(str(path), "w") as nc:
        _time(nc, n_time)
        nc.createDimension(dim, size)
        v = nc.createVariable(var, "f4", ("time", dim))
        v[:] = vals.astype(np.float32)
        for name, values in zip(("lon", "lat"), coords or ()):
            c = nc.createVariable(name, "f8", (dim,))
            c[:] = values
    return path


def _levelwise(d: Path, lon=HAND_LON, lat=HAND_LAT, ncol=HAND_NCOL, *,
               variables=("uo", "vo", "wo"), per_var=None, **kw) -> Path:
    """A directory with one level-wise file per variable; ``per_var`` holds
    keyword overrides for single variables."""
    d.mkdir(parents=True, exist_ok=True)
    for var in variables:
        opts = {"ncol": ncol, **kw, **(per_var or {}).get(var, {})}
        if var == "zos":
            opts["two_d"] = True
        _write_var(d / f"{var}.nc", var, lon, lat, **opts)
    return d


def _raw(d: Path, size=N3D, *, variables=("uo", "vo", "wo"), zos_size=None,
         per_var=None, **kw) -> Path:
    """A directory of ``(time, nodes_3d)`` files (and ``zos(time, nodes_2d)``
    with ``zos_size`` entries)."""
    d.mkdir(parents=True, exist_ok=True)
    for var in variables:
        _write_raw(d / f"{var}.nc", var, size,
                   **{**kw, **(per_var or {}).get(var, {})})
    if zos_size is not None:
        _write_raw(d / "zos.nc", "zos", zos_size, two_d=True,
                   **(per_var or {}).get("zos", {}))
    return d


def _open(d: Path, mode=MODE_3D):
    return Fesom14Reader.open_dataset(str(d), mode_cfg=mode,
                                      time_stamp="instant", time_window="1 d")


def _mesh(path: Path):
    with contextlib.redirect_stdout(io.StringIO()):
        return Fesom14Reader.load_mesh(
            MeshConfig(type="fesom14", meshpath=str(path)))


def _hand_mesh(tmp_path: Path):
    return _mesh(_write_hand_mesh(tmp_path / "mesh"))


def _uniform_ascii_mesh(tmp_path: Path):
    """The hand mesh's nodes and triangles with four levels everywhere."""
    return _mesh(_write_ascii(tmp_path / "umesh", HAND_LON, HAND_LAT,
                              EXPECTED_TRI, [4] * 5))


def _refusal(mesh, ds) -> str:
    with pytest.raises(ValueError) as info:
        Fesom14Reader.prepare(mesh, ds)
    return str(info.value)


@pytest.fixture(scope="module")
def island():
    """The 36-node island patch: coordinates, cell polygons, column lengths
    1..4."""
    lon, lat, tri, _ = _patch(10.0, 40.0)
    lon_b, lat_b = _polygons_from_triangulation(lon, lat, tri, nv=16)
    ncol = 1 + (np.arange(lon.size) * 7) % 4
    return lon, lat, ncol, lon_b, lat_b


def _polygon_mesh(tmp_path: Path, island, ncol=None):
    lon, lat, ncol0, lon_b, lat_b = island
    ncol = ncol0 if ncol is None else ncol
    path = _write_levelwise(tmp_path / "polymesh.nc", lon, lat, lon_b, lat_b,
                            ncol)
    return _mesh(path)


# ---------------------------------------------------------------------------
# The hook and the good cases
# ---------------------------------------------------------------------------

def test_good_datasets_accepted(tmp_path, island):
    """Level-wise data (with node coordinates, without them, with ``zos`` and
    a sampled 3-D scalar) on the ASCII mesh, ``(time, nodes_3d)`` data on the
    ASCII mesh, and level-wise data on a mesh from cell polygons pass."""
    mesh = _hand_mesh(tmp_path)
    for d, mode in ((_levelwise(tmp_path / "lw"), MODE_3D),
                    (_levelwise(tmp_path / "lw_nocoords", coords=False), MODE_3D),
                    (_levelwise(tmp_path / "lw_more",
                                variables=("uo", "vo", "wo", "zos", "thetao")),
                     MODE_3D_T),
                    (_raw(tmp_path / "raw", zos_size=5), MODE_3D)):
        with _open(d, mode) as ds:
            assert Fesom14Reader.prepare(mesh, ds) is None
    lon, lat, ncol, _, _ = island
    pmesh = _polygon_mesh(tmp_path, island)
    with _open(_levelwise(tmp_path / "plw", lon, lat, ncol)) as ds:
        assert Fesom14Reader.prepare(pmesh, ds) is None


def test_not_a_fesom14_mesh_refused(tmp_path):
    mesh = _hand_mesh(tmp_path)
    del mesh.metadata["fesom14_levels"]
    with _open(_levelwise(tmp_path / "lw")) as ds:
        msg = _refusal(mesh, ds)
    assert "fesom14" in msg and "load_mesh" in msg


def test_foreign_dataset_refused(tmp_path):
    """Datasets ``open_dataset`` does not make: a variable of another
    layout, no time-dependent variable, no level coordinate, an empty
    time window."""
    mesh = _hand_mesh(tmp_path)
    with _open(_levelwise(tmp_path / "lw")) as ds:
        junk = ds.assign(junk=(("time", "x"), np.zeros((3, 2))))
        msg = _refusal(mesh, junk)
        assert "'junk'" in msg and "('time', 'x')" in msg
        assert "open_dataset" in _refusal(mesh, ds[[]])
        msg = _refusal(mesh, ds.drop_vars("depth"))
        assert "'depth'" in msg and "no coordinate" in msg
        msg = _refusal(mesh, ds.isel(time=slice(0, 0)))
        assert "no record" in msg


# ---------------------------------------------------------------------------
# Sizes
# ---------------------------------------------------------------------------

def test_wrong_horizontal_size_refused(tmp_path, island):
    """Level-wise data of another mesh: six nodes on the five-node ASCII
    mesh, 35 on the 36-node polygon mesh; a 2-D field of the wrong size
    next to ``(time, nodes_3d)`` velocities (``open_dataset`` cannot compare
    the two)."""
    mesh = _hand_mesh(tmp_path)
    lon6 = np.append(HAND_LON, 12.5)
    lat6 = np.append(HAND_LAT, 51.0)
    with _open(_levelwise(tmp_path / "lw6", lon6, lat6, [4, 3, 2, 3, 1, 2])) as ds:
        msg = _refusal(mesh, ds)
    assert "'uo'" in msg and "6 nodes" in msg and "the mesh has 5" in msg
    assert "uo.nc" in msg                       # the file
    assert "mesh.meshpath" in msg and "data.ncinput" in msg   # the fix

    lon, lat, ncol, _, _ = island
    pmesh = _polygon_mesh(tmp_path, island)
    with _open(_levelwise(tmp_path / "plw35", lon[:35], lat[:35], ncol[:35])) as ds:
        msg = _refusal(pmesh, ds)
    assert "'uo'" in msg and "35 nodes" in msg and "the mesh has 36" in msg

    with _open(_raw(tmp_path / "raw_zos6", zos_size=6)) as ds:
        msg = _refusal(mesh, ds)
    assert "'zos'" in msg and "6 nodes" in msg and "the mesh has 5" in msg


@pytest.mark.parametrize("size", [12, 14, 5])
def test_wrong_nodes3d_length_refused(tmp_path, size):
    """A ``(time, nodes_3d)`` vector must have one entry per 3-D node of the
    ASCII mesh (13): the gather checks only index bounds, so 14 entries
    would be read silently without this check."""
    mesh = _hand_mesh(tmp_path)
    with _open(_raw(tmp_path / "raw", size)) as ds:
        msg = _refusal(mesh, ds)
    assert "'uo'" in msg and f"{size} entries" in msg
    assert "13 3-D nodes" in msg
    if size == 5:
        assert "2-D nodes" in msg


def test_nodes3d_on_polygon_mesh_refused(tmp_path, island):
    pmesh = _polygon_mesh(tmp_path, island)
    with _open(_raw(tmp_path / "raw")) as ds:
        msg = _refusal(pmesh, ds)
    assert "'uo'" in msg and "3-D node table" in msg
    assert "ASCII mesh" in msg and "mesh.meshpath" in msg


# ---------------------------------------------------------------------------
# Levels
# ---------------------------------------------------------------------------

def test_level_axis_mismatch_refused(tmp_path):
    mesh = _hand_mesh(tmp_path)
    with _open(_levelwise(tmp_path / "l30",
                          levels=[0.0, 10.0, 30.0, 50.0])) as ds:
        msg = _refusal(mesh, ds)
    assert "'uo'" in msg and "level 2: 30 m against 25 m" in msg
    assert "1 level(s)" in msg

    # beyond the tolerance (1e-3 m) by a little: refused; within it: accepted
    with _open(_levelwise(tmp_path / "l_off",
                          levels=[0.0, 10.0, 25.002, 50.0])) as ds:
        assert "level 2" in _refusal(mesh, ds)
    with _open(_levelwise(tmp_path / "l_ok",
                          levels=[0.0, 10.0, 25.0005, 50.0])) as ds:
        Fesom14Reader.prepare(mesh, ds)

    # another number of levels
    with _open(_levelwise(tmp_path / "l5", ncol=HAND_NCOL,
                          levels=[0.0, 10.0, 25.0, 50.0, 80.0])) as ds:
        msg = _refusal(mesh, ds)
    assert "'uo'" in msg and "5 levels" in msg and "the mesh has 4" in msg


def test_full_axis_after_cut(tmp_path):
    """``max_levels: 3`` cuts ``mesh.sigma`` before ``prepare`` runs: the
    data's full axis is compared with the mesh's FULL level table, so the
    full data pass and data holding only the kept levels are refused."""
    mesh = _hand_mesh(tmp_path)
    tracking._apply_max_levels(mesh, ModeConfig(Fdrift3D=True, max_levels=3))
    assert mesh.sigma.tolist() == [0.0, 10.0, 25.0]
    sigma = mesh.sigma
    with _open(_levelwise(tmp_path / "full")) as ds:
        Fesom14Reader.prepare(mesh, ds)
    assert mesh.sigma is sigma
    with _open(_levelwise(tmp_path / "cut", ncol=np.minimum(HAND_NCOL, 3),
                          levels=LEVELS[:3])) as ds:
        msg = _refusal(mesh, ds)
    assert "3 levels" in msg and "the mesh has 4" in msg
    assert "max_levels" in msg


# ---------------------------------------------------------------------------
# Node coordinates
# ---------------------------------------------------------------------------

def test_common_permutation_vs_mesh_refused(tmp_path, island):
    """Every variable with its nodes in the same other order, on meshes with
    uniform columns: the masks agree, ``open_dataset`` sees nothing (all
    variables agree with each other), only the coordinates show it."""
    umesh = _uniform_ascii_mesh(tmp_path)
    # swap nodes 0 and 1: the same latitude, another longitude
    perm = [1, 0, 2, 3, 4]
    with _open(_levelwise(tmp_path / "perm", ncol=[4] * 5, perm=perm)) as ds:
        msg = _refusal(umesh, ds)
    assert "node coordinates" in msg and "at node 0" in msg
    assert "(11.0000000, 50.0000000) against (10.0000000, 50.0000000)" in msg
    assert "2 node(s)" in msg and "order" in msg
    # without coordinates the same data pass: the masks cannot see it
    with _open(_levelwise(tmp_path / "perm_nc", ncol=[4] * 5, perm=perm,
                          coords=False)) as ds:
        Fesom14Reader.prepare(umesh, ds)

    lon, lat, _, _, _ = island
    full = np.full(lon.size, 4)
    pmesh = _polygon_mesh(tmp_path, island, ncol=full)
    perm = np.random.default_rng(3).permutation(lon.size)
    with _open(_levelwise(tmp_path / "pperm", lon, lat, full, perm=perm)) as ds:
        msg = _refusal(pmesh, ds)
    n = int(np.argmax(perm != np.arange(lon.size)))
    assert "node coordinates" in msg and f"at node {n}:" in msg


@pytest.mark.parametrize("shift", ["lon", "lat"])
def test_coordinate_tolerance(tmp_path, shift):
    """A node moved by 2e-6 deg in longitude or latitude alone is refused;
    1e-9 deg (rounding) is accepted."""
    mesh = _hand_mesh(tmp_path)
    for delta, ok in ((2e-6, False), (1e-9, True)):
        lon, lat = HAND_LON.copy(), HAND_LAT.copy()
        (lon if shift == "lon" else lat)[3] += delta
        with _open(_levelwise(tmp_path / f"{shift}{delta}", lon, lat)) as ds:
            if ok:
                Fesom14Reader.prepare(mesh, ds)
            else:
                assert "at node 3:" in _refusal(mesh, ds)


def test_longitude_convention_accepted(tmp_path):
    """A mesh at the dateline (longitudes -180 .. -178 after normalisation)
    and data written with longitudes 180 .. 182: the same nodes."""
    lon = HAND_LON - 190.0
    mesh = _mesh(_write_ascii(tmp_path / "dmesh", lon, HAND_LAT, EXPECTED_TRI,
                              HAND_NCOL))
    np.testing.assert_array_equal(mesh.metadata["fesom14_lon"], lon)
    with _open(_levelwise(tmp_path / "d360", lon + 360.0, HAND_LAT)) as ds:
        Fesom14Reader.prepare(mesh, ds)
    # the same convention, one node off: still refused
    off = lon + 360.0
    off[4] += 1e-3
    with _open(_levelwise(tmp_path / "d360_off", off, HAND_LAT)) as ds:
        assert "at node 4:" in _refusal(mesh, ds)


def test_nodes3d_coordinates(tmp_path):
    """Vectors over the 3-D nodes that carry ``lon``/``lat`` over
    ``nodes_3d``: each entry has the coordinates of its 2-D node."""
    mesh = _hand_mesh(tmp_path)
    nodes, levels = np.nonzero(EXPECTED_IDX >= 0)
    owner = np.empty(N3D, dtype=int)
    owner[EXPECTED_IDX[nodes, levels]] = nodes
    # the 3-D nodes are numbered level-major: node 0..4 at level 0, then ...
    assert owner.tolist() == [0, 1, 2, 3, 4, 0, 1, 2, 3, 0, 1, 3, 0]
    lon3, lat3 = HAND_LON[owner], HAND_LAT[owner]
    with _open(_raw(tmp_path / "ok", coords=(lon3 + 360.0, lat3))) as ds:
        assert ds["lon"].dims == ("nodes_3d",)
        Fesom14Reader.prepare(mesh, ds)
    lat_off = lat3.copy()
    lat_off[10] += 1e-3                     # 3-D node 10: node 1, level 2
    with _open(_raw(tmp_path / "off", coords=(lon3, lat_off))) as ds:
        msg = _refusal(mesh, ds)
    assert "at 3-D node 10 (node 1, level 2)" in msg
    assert "(11.0000000, 50.0010000) against (11.0000000, 50.0000000)" in msg


def test_levelwise_dimension_named_nodes_3d(tmp_path, island):
    """The level-wise writer names its horizontal dimension ``nodes_3d``
    (length nod2D): a 2-D field over it is a 2-D field, not a vector over
    the 3-D nodes, on either kind of mesh."""
    variables = ("uo", "vo", "wo", "zos")
    mesh = _hand_mesh(tmp_path)
    with _open(_levelwise(tmp_path / "lw", variables=variables,
                          hdim="nodes_3d")) as ds:
        assert ds["zos"].dims == ("time", "nodes_3d")
        Fesom14Reader.prepare(mesh, ds)
    lon, lat, ncol, _, _ = island
    pmesh = _polygon_mesh(tmp_path, island)
    with _open(_levelwise(tmp_path / "plw", lon, lat, ncol, variables=variables,
                          hdim="nodes_3d")) as ds:
        Fesom14Reader.prepare(pmesh, ds)


# ---------------------------------------------------------------------------
# Masks (record 0)
# ---------------------------------------------------------------------------

def _set(t, k, n, value):
    def edit(vals):
        vals[t, k, n] = value
    return edit


@pytest.mark.parametrize("case", ["extra", "missing", "zero_below_bed"])
def test_ascii_mask_disagreement_refused(tmp_path, case):
    """The ASCII mesh's index table fixes the wet mask; the data of record 0
    must hold values in exactly its wet slots.  ``vo`` alone is wrong in the
    first two cases (every variable is checked); in the third every
    variable writes 0 below the bed (the level-wise variant this check
    exists for: its own mask is wet everywhere)."""
    mesh = _hand_mesh(tmp_path)
    if case == "extra":          # node 2 has 2 levels: level 2 is dry
        d = _levelwise(tmp_path / case,
                       per_var={"vo": {"ncol": [4, 3, 3, 3, 1]}})
        where, mesh_state, data_state = "node 2, level 2", "dry", "the value 0.222"
    elif case == "missing":      # node 1 has 3 levels: level 2 is wet
        d = _levelwise(tmp_path / case,
                       per_var={"vo": {"ncol": [4, 2, 2, 3, 1]}})
        where, mesh_state, data_state = "node 1, level 2", "wet", "no value"
    else:                        # node 1: level 3 dry, 0 written there
        d = _levelwise(tmp_path / case, below="zero")
        where, mesh_state, data_state = "node 1, level 3", "dry", "the value 0"
    with _open(d) as ds:
        msg = _refusal(mesh, ds)
    var = "'uo'" if case == "zero_below_bed" else "'vo'"
    assert var in msg and where in msg
    assert "index table" in msg
    assert f"the mesh: {mesh_state}" in msg and f"the data: {data_state}" in msg
    assert "2000-01-01T00:00:00" in msg          # the record, by its instant
    if case == "zero_below_bed":
        assert "written as 0 below the bed" in msg
    else:
        assert "written as 0" not in msg


@pytest.mark.parametrize("case", ["extra", "missing", "zero_below_bed",
                                  "sampled"])
def test_polygon_mask_disagreement_refused(tmp_path, island, case):
    """A mesh from cell polygons carries the mask of record 0 of the u
    variable it was built from; every variable of every dataset must have
    it, a sampled 3-D scalar too."""
    lon, lat, ncol, _, _ = island
    pmesh = _polygon_mesh(tmp_path, island)
    n = int(np.argmax(ncol == 2))             # a node with 2 levels
    mode = MODE_3D
    if case == "extra":
        d = _levelwise(tmp_path / case, lon, lat, ncol,
                       per_var={"wo": {"edit": _set(0, 2, n, 0.5)}})
        var, where, data_state = "'wo'", f"node {n}, level 2", "the value 0.5"
    elif case == "missing":
        d = _levelwise(tmp_path / case, lon, lat, ncol,
                       per_var={"vo": {"edit": _set(0, 1, n, np.nan)}})
        var, where, data_state = "'vo'", f"node {n}, level 1", "no value"
    elif case == "zero_below_bed":
        d = _levelwise(tmp_path / case, lon, lat, ncol, below="zero")
        first = int(np.argmax(ncol < 4))
        var, where = "'uo'", f"node {first}, level {ncol[first]}"
        data_state = "the value 0"
    else:
        d = _levelwise(tmp_path / case, lon, lat, ncol,
                       variables=("uo", "vo", "wo", "thetao"),
                       per_var={"thetao": {"edit": _set(0, 1, n, np.nan)}})
        var, where, data_state = "'thetao'", f"node {n}, level 1", "no value"
        mode = MODE_3D_T
    with _open(d, mode) as ds:
        msg = _refusal(pmesh, ds)
    assert var in msg and where in msg and f"the data: {data_state}" in msg
    assert "polymesh.nc" in msg                  # where the mesh's mask is from
    assert ("written as 0" in msg) == (case == "zero_below_bed")


def test_first_record_of_nodes3d_and_2d_fields(tmp_path):
    """Record 0 of a ``(time, nodes_3d)`` vector holds a value at every 3-D
    node; record 0 of a 2-D field a value at every node."""
    mesh = _hand_mesh(tmp_path)
    # entry 9 is 3-D node 9: node 0, level 2 (EXPECTED_IDX)
    assert EXPECTED_IDX[0, 2] == 9
    with _open(_raw(tmp_path / "raw", per_var={"vo": {"fill_at": (0, 9)}})) as ds:
        msg = _refusal(mesh, ds)
    assert "'vo'" in msg and "3-D node 9 (node 0, level 2)" in msg
    with _open(_levelwise(tmp_path / "lw", variables=("uo", "vo", "wo", "zos"),
                          per_var={"zos": {"edit": _set(0, 0, 3, np.nan)}})) as ds:
        msg = _refusal(mesh, ds)
    assert "'zos'" in msg and "node 3" in msg


def test_reads_record_zero_only(tmp_path):
    """The masks come from record 0 alone, one record per variable, read
    lazily: a defect in record 1 is not seen here (the reads check every
    record), and no other record is loaded."""
    mesh = _hand_mesh(tmp_path)
    d = _levelwise(tmp_path / "lw", variables=("uo", "vo", "wo", "zos"),
                   per_var={"uo": {"edit": _set(1, 0, 0, np.nan)}})
    with _open(d) as ds:
        Fesom14Reader.prepare(mesh, ds)
        log = []
        for name in list(ds.data_vars):
            da = ds[name]

            def tap(block, block_info=None, name=name):
                log.append((name, block_info[0]["chunk-location"][0]))
                return block
            ds[name] = da.copy(data=da.data.map_blocks(
                tap, dtype=da.dtype, meta=da.data._meta))
        Fesom14Reader.prepare(mesh, ds)
    assert sorted(log) == [("uo", 0), ("vo", 0), ("wo", 0), ("zos", 0)]


# ---------------------------------------------------------------------------
# Datasets in turn, and the engine contract
# ---------------------------------------------------------------------------

class _StubReads(Fesom14Reader):
    """The reader with zero velocities on the kept levels, so that
    ``run_single_realization`` runs through ``prepare``."""

    @staticmethod
    def read_velocities(ds, timestep, mesh, mode_cfg=None, **kw):
        k, n = mesh.sigma.size, mesh.n_nodes
        return VelocityField(u=np.zeros((k, n)), v=np.zeros((k, n)),
                             w=np.zeros((k, n)), eta=np.zeros(n),
                             uv_layout="node", uv_vertical="level")


def _rsr(mesh, ds):
    """One euler step of one particle inside the first triangle at 5 m."""
    tri = EXPECTED_TRI[0]
    lon = np.array([HAND_LON[tri].mean()])
    lat = np.array([HAND_LAT[tri].mean()])
    st = make_state(lon, lat, np.array([5.0]))
    st = st.replace(nearest_node=find_nearest_nodes(mesh, st.lon, st.lat))
    numerics = NumericsConfig(dt=86400.0, integrator="euler",
                              uv_interp="barycentric", w_interp="barycentric")
    with contextlib.redirect_stdout(io.StringIO()):
        return tracking.run_single_realization(
            mesh, ds, _StubReads, st, MODE_3D, numerics, start_time_idx=0,
            n_steps=1)


def test_second_dataset_rechecked(tmp_path):
    """One mesh, a good dataset, then a dataset whose levels differ and one
    whose nodes are all in another (common) order: each is checked again
    and refused, directly and through ``run_single_realization`` with the
    datasets opened as temporaries (freed between the calls)."""
    umesh = _uniform_ascii_mesh(tmp_path)
    good = _levelwise(tmp_path / "good", ncol=[4] * 5)
    levels = _levelwise(tmp_path / "levels", ncol=[4] * 5,
                        levels=[0.0, 10.0, 30.0, 50.0])
    perm = _levelwise(tmp_path / "perm", ncol=[4] * 5, perm=[1, 0, 2, 3, 4])

    with _open(good) as ds:
        Fesom14Reader.prepare(umesh, ds)
    for d, what in ((levels, "level 2"), (perm, "node coordinates")):
        with _open(d) as ds:
            assert what in _refusal(umesh, ds)

    _rsr(umesh, _open(good))
    for d, what in ((levels, "level 2"), (perm, "node coordinates"),
                    (levels, "level 2")):
        gc.collect()
        with pytest.raises(ValueError, match=what):
            _rsr(umesh, _open(d))
        assert "kelp_prepared_dataset" not in umesh.metadata
        gc.collect()
        _rsr(umesh, _open(good))
        assert "kelp_prepared_dataset" in umesh.metadata


def test_idempotent(tmp_path):
    """``prepare`` twice on one pair: no error, nothing changed — the mesh's
    level table, depths, metadata entries (the same objects) and the
    declaration ``uv_data_last``, frozen as the engine freezes it."""
    mesh = _hand_mesh(tmp_path)
    tracking._apply_max_levels(mesh, ModeConfig(Fdrift3D=True, max_levels=3))
    last = mesh.metadata["uv_data_last"]
    last.setflags(write=False)
    before = dict(mesh.metadata)
    sigma, depth = mesh.sigma, mesh.depth
    depth_copy = depth.copy()
    with _open(_levelwise(tmp_path / "lw", variables=("uo", "vo", "wo", "zos"))) as ds:
        attrs = dict(ds.attrs)
        assert Fesom14Reader.prepare(mesh, ds) is None
        assert Fesom14Reader.prepare(mesh, ds) is None
        tracking._prepare_dataset(Fesom14Reader, mesh, ds, always=True)
        tracking._prepare_dataset(Fesom14Reader, mesh, ds, always=True)
        assert ds.attrs.pop("kelp_dataset_token")
        assert ds.attrs == attrs
    for key in ("kelp_prepared_dataset", "kelp_prepared_sigma"):
        mesh.metadata.pop(key)
    assert mesh.metadata.keys() == before.keys()
    assert all(mesh.metadata[k] is v for k, v in before.items())
    assert mesh.metadata["uv_data_last"].tolist() == [3, 2, 1, 2, 0]
    assert mesh.sigma is sigma and mesh.depth is depth
    np.testing.assert_array_equal(mesh.depth, depth_copy)
