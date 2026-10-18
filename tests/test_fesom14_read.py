"""FESOM 1.4 reader: the reads (``kelp.readers.fesom14.reader``).

``read_velocities`` hands the engine one record: in 3-D, u/v/w on the
levels, ``(K, N)`` with ``uv_vertical="level"``; in 2-D, u/v ``(N,)`` from one
level or the depth mean.  ``read_scalar_fields`` reads the sampled variables
(3-D: forward-filled below the bed; 2-D: the same reduction as u/v).  Every
record of every variable is checked against the mesh's static wet mask
BEFORE anything is changed (zeroing below the bed, cutting to the kept
levels), and a refusal names the variable and the record by its instant.

Mesh: the five-node ASCII mesh of ``test_fesom14_ascii_mesh`` (columns of 4,
3, 2, 3, 1 levels at 0, 10, 25, 50 m, beds at 50, 25, 10, 25, 0 m; 13 3-D
nodes).  Data files are written here, level-wise ``var(time, depth,
ncells)`` with NaN (a decoded ``_FillValue``) below the bed, or
``var(time, nodes_3d)`` vectors placed with the hand-written index table;
daily snapshots from 2000-01-01, read with ``data.time_stamp: instant``.
Expected values, wet slots and depth means are written by hand.  All data
is synthetic.
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")
pytest.importorskip("cftime")

from kelp import tracking  # noqa: E402
from kelp.core.config import MeshConfig, ModeConfig  # noqa: E402
from kelp.core.rotation import (FESOM2_ANGLES, euler_matrix,  # noqa: E402
                                rotate_vectors)
from kelp.numerics.sampling import sample_field  # noqa: E402
from kelp.readers.fesom14 import Fesom14Reader  # noqa: E402

from tests.test_fesom14_ascii_mesh import EXPECTED_IDX  # noqa: E402
from tests.test_fesom14_ascii_mesh import _write as _write_hand_mesh  # noqa: E402
from tests.test_fesom14_mesh import _write_ascii  # noqa: E402
from tests.test_rotated_frame import _GRID_ELEMS_0, _grid_coords  # noqa: E402

LEVELS = np.array([0.0, 10.0, 25.0, 50.0])
HAND_LON = np.array([10.0, 11.0, 10.5, 11.5, 12.0])
HAND_LAT = np.array([50.0, 50.0, 51.0, 51.0, 50.0])

#: the static wet mask of the hand mesh, (level, node), by hand
WET = np.array([[1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0],
                [1, 1, 0, 1, 0],
                [1, 0, 0, 0, 0]], dtype=bool)
BED = np.array([50.0, 25.0, 10.0, 25.0, 0.0])

#: the base of each variable's values
BASE = {"uo": 0.25, "vo": -0.5, "wo": 0.0625, "thetao": 4.0, "so": 34.0,
        "zos": 0.75}

MODE_3D = ModeConfig(Fdrift3D=True)
MODE_3D_TS = ModeConfig(Fdrift3D=True, sample_vars=["thetao", "so"])
MODE_2D = ModeConfig()

T0 = "2000-01-01"


# ---------------------------------------------------------------------------
# Values and writers
# ---------------------------------------------------------------------------

def _values(var: str, n_time: int, wet=WET) -> np.ndarray:
    """``(time, level, node)``: base + t + k/8 + n/64 on wet slots (exact in
    float32), NaN below the bed."""
    t = np.arange(n_time)[:, None, None]
    k = np.arange(wet.shape[0])[None, :, None]
    n = np.arange(wet.shape[1])[None, None, :]
    return np.where(wet[None], BASE[var] + t + k / 8.0 + n / 64.0, np.nan)


def _expected(var: str, t: int) -> np.ndarray:
    """Record ``t`` of ``var`` as the reader must return it: 0 below the bed."""
    return np.where(WET, _values(var, t + 1)[t], 0.0)


def _time(nc, n_time: int, unit: str = "days") -> None:
    nc.createDimension("time", None)
    tv = nc.createVariable("time", "f8", ("time",))
    tv.units = f"{unit} since {T0}"
    tv.calendar = "standard"
    tv[:] = np.arange(n_time, dtype=np.float64)


def _write_levelwise(path: Path, var: str, vals: np.ndarray, *, attrs=None,
                     lon=HAND_LON, lat=HAND_LAT, levels=LEVELS,
                     unit: str = "days") -> Path:
    """``var(time, depth, ncells)`` from ``vals`` (time, level, node), one
    record per ``unit``; NaN is written as the ``_FillValue``."""
    n_time, n_lev, n_nodes = vals.shape
    with netCDF4.Dataset(str(path), "w") as nc:
        _time(nc, n_time, unit)
        nc.createDimension("depth", n_lev)
        nc.createDimension("ncells", n_nodes)
        z = nc.createVariable("depth", "f8", ("depth",))
        z.units, z.positive = "m", "down"
        z[:] = levels
        for name, values in (("lon", lon), ("lat", lat)):
            c = nc.createVariable(name, "f8", ("ncells",))
            c[:] = values
        v = nc.createVariable(var, "f4", ("time", "depth", "ncells"),
                              fill_value=np.float32(1e20))
        for key, value in (attrs or {}).items():
            v.setncattr(key, value)
        v[:] = np.ma.masked_invalid(vals)
    return path


def _write_nodes3d(path: Path, var: str, vals: np.ndarray, *,
                   attrs=None) -> Path:
    """``var(time, nodes_3d)``: entry ``EXPECTED_IDX[n, k]`` holds
    ``vals[t, k, n]`` (the hand-written index table); a NaN in a wet slot
    is written as the raw fill 1e30."""
    n_time = vals.shape[0]
    vec = np.zeros((n_time, 13))
    for n in range(EXPECTED_IDX.shape[0]):
        for k in range(EXPECTED_IDX.shape[1]):
            if EXPECTED_IDX[n, k] >= 0:
                vec[:, EXPECTED_IDX[n, k]] = vals[:, k, n]
    vec[~np.isfinite(vec)] = 1e30
    with netCDF4.Dataset(str(path), "w") as nc:
        _time(nc, n_time)
        nc.createDimension("nodes_3d", vec.shape[1])
        v = nc.createVariable(var, "f4", ("time", "nodes_3d"))
        for key, value in (attrs or {}).items():
            v.setncattr(key, value)
        v[:] = vec.astype(np.float32)
    return path


def _write_2d(path: Path, var: str, vals: np.ndarray, *,
              dim: str = "ncells") -> Path:
    """``var(time, <dim>)`` from ``vals`` (time, entries)."""
    with netCDF4.Dataset(str(path), "w") as nc:
        _time(nc, vals.shape[0])
        nc.createDimension(dim, vals.shape[1])
        v = nc.createVariable(var, "f4", ("time", dim),
                              fill_value=np.float32(1e20))
        v[:] = np.ma.masked_invalid(vals)
    return path


def _dataset(d: Path, form: str = "levelwise",
             variables=("uo", "vo", "wo"), *, n_time: int = 3, edit=None,
             attrs=None, tables=None, zos=None) -> Path:
    """One file per variable.  ``edit[var](vals)`` plants a defect,
    ``attrs[var]`` sets attributes, ``tables[var]`` = a (level, node) table
    used for every record instead of the generated values; ``zos`` =
    ``(dim, values (time, entries))`` adds a 2-D sea-surface height."""
    d.mkdir(parents=True, exist_ok=True)
    for var in variables:
        if tables and var in tables:
            vals = np.repeat(np.asarray(tables[var], float)[None], n_time, 0)
        else:
            vals = _values(var, n_time)
        if edit and var in edit:
            edit[var](vals)
        write = _write_levelwise if form == "levelwise" else _write_nodes3d
        write(d / f"{var}.nc", var, vals, attrs=(attrs or {}).get(var))
    if zos is not None:
        _write_2d(d / "zos.nc", "zos", zos[1], dim=zos[0])
    return d


def _open(d: Path, mode=MODE_3D, window: str = "1 d"):
    return Fesom14Reader.open_dataset(str(d), mode_cfg=mode,
                                      time_stamp="instant", time_window=window)


def _quiet(fn, *args, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kw)


def _hand_mesh(tmp_path: Path, **cfg):
    return _quiet(Fesom14Reader.load_mesh,
                  MeshConfig(type="fesom14",
                             meshpath=str(_write_hand_mesh(tmp_path / "mesh")),
                             **cfg))


def _cut(mesh, max_levels: int, three_d: bool = True) -> None:
    """Cut ``mesh.sigma`` as ``run_tracking`` does."""
    tracking._apply_max_levels(
        mesh, ModeConfig(Fdrift3D=three_d, max_levels=max_levels))


def _read(ds, t, mesh, mode=MODE_3D, **kw):
    return Fesom14Reader.read_velocities(ds, t, mesh, mode, **kw)


# ---------------------------------------------------------------------------
# 3-D: u/v/w on the levels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("form", ["levelwise", "nodes3d"])
def test_3d_level_field(tmp_path, form):
    """``(K, N)`` u/v/w with ``uv_vertical="level"``, C-contiguous, 0 below
    the bed; ``w`` positive down; ``velocity_float32`` honoured."""
    mesh = _hand_mesh(tmp_path)
    with _open(_dataset(tmp_path / "data", form)) as ds:
        vel = _read(ds, 1, mesh)
        vel32 = _read(ds, 1, mesh, velocity_float32=True)
        window = ds.isel(time=slice(1, 3)).load()
        vel_w = _read(window, 0, mesh)

    assert vel.uv_vertical == "level" and vel.uv_layout == "node"
    for comp, var in (("u", "uo"), ("v", "vo")):
        arr = getattr(vel, comp)
        assert arr.shape == (4, 5) and arr.dtype == np.float64
        assert arr.flags.c_contiguous
        np.testing.assert_array_equal(arr, _expected(var, 1))
        np.testing.assert_array_equal(getattr(vel_w, comp), arr)
        assert (arr[~WET] == 0.0).all()
    # raw wo without attributes is positive up: negated into positive down
    assert vel.w.shape == (4, 5) and vel.w.flags.c_contiguous
    np.testing.assert_array_equal(vel.w, -_expected("wo", 1))
    assert (vel.w[~WET] == 0.0).all() and not np.signbit(vel.w[~WET]).any()
    np.testing.assert_array_equal(vel.eta, np.zeros(5))
    assert vel.scalars is None

    for comp, var in (("u", "uo"), ("v", "vo"), ("w", "wo")):
        arr = getattr(vel32, comp)
        assert arr.dtype == np.float32 and arr.flags.c_contiguous
        sign = -1.0 if comp == "w" else 1.0
        np.testing.assert_array_equal(
            arr, (sign * _expected(var, 1)).astype(np.float32))
    assert vel32.eta.dtype == np.float64


@pytest.mark.parametrize("attrs, mode, sign", [
    ({"standard_name": "upward_sea_water_velocity"}, MODE_3D, -1.0),
    ({"positive": "up"}, MODE_3D, -1.0),
    ({"positive": "down"}, MODE_3D, 1.0),
    ({}, ModeConfig(Fdrift3D=True, w_positive="down"), 1.0),
])
def test_w_sign(tmp_path, attrs, mode, sign):
    """The file's convention decides: ``upward_sea_water_velocity`` and
    ``positive: up`` are negated, ``positive: down`` and
    ``mode.w_positive: down`` are not."""
    mesh = _hand_mesh(tmp_path)
    with _open(_dataset(tmp_path / "data", attrs={"wo": attrs}), mode) as ds:
        vel = _read(ds, 2, mesh, mode)
    np.testing.assert_array_equal(vel.w, sign * _expected("wo", 2))


def test_probe_uv_layout(tmp_path):
    mesh = _hand_mesh(tmp_path)
    for form in ("levelwise", "nodes3d"):
        with _open(_dataset(tmp_path / form, form)) as ds:
            assert Fesom14Reader.probe_uv_layout(ds) == "node"
            assert Fesom14Reader.probe_uv_layout(ds, MODE_3D, mesh) == "node"


# ---------------------------------------------------------------------------
# Rotation into the tracking frame
# ---------------------------------------------------------------------------

def test_tracking_frame_rotation_factors(tmp_path):
    """With ``mesh.rotation``, u/v are the geographic values rotated by
    ``core.rotation.rotate_vectors`` at the nodes; w and eta are not
    rotated."""
    mesh = _hand_mesh(tmp_path, rotation="fesom2")
    d = _dataset(tmp_path / "data")
    R = euler_matrix(*FESOM2_ANGLES)
    with _open(d) as ds:
        vel = _read(ds, 1, mesh)
        vel32 = _read(ds, 1, mesh, velocity_float32=True)
    eu, ev = _expected("uo", 1), _expected("vo", 1)
    for k in range(4):
        ur, vr = rotate_vectors(eu[k], ev[k], HAND_LON, HAND_LAT, R)
        assert np.abs(vel.u[k] - ur).max() <= 1e-12
        assert np.abs(vel.v[k] - vr).max() <= 1e-12
        assert np.abs(vel32.u[k] - ur).max() <= 1e-6
    assert np.abs(vel.u - eu).max() > 0.1        # the rotation is not small
    assert vel.u.flags.c_contiguous and vel32.u.dtype == np.float32
    np.testing.assert_array_equal(vel.w, -_expected("wo", 1))

    with _open(d, MODE_2D) as ds:
        vel2 = _read(ds, 1, mesh, ModeConfig(sigma_layer=1))
    ur, vr = rotate_vectors(eu[1], ev[1], HAND_LON, HAND_LAT, R)
    assert np.abs(vel2.u - ur).max() <= 1e-12
    assert np.abs(vel2.v - vr).max() <= 1e-12


def _grid_run(tmp_path, three_d: bool, rotation, factors=None):
    """The node-layout recipe of ``test_rotated_frame``: a 3x3 grid at
    (0 E, 5 N), constant geographic u = 1, v = 0.5 m/s on every level (w =
    0) in hourly records, one particle from (0, 5), RK2, barycentric, dt =
    3600 s, 3 steps."""
    lon, lat = _grid_coords(lat_centre=5.0, lon_centre=0.0)
    meshdir = tmp_path / "grid_mesh"
    if not meshdir.exists():
        _write_ascii(meshdir, lon, lat, _GRID_ELEMS_0, [4] * lon.size)
    d = tmp_path / "grid_data"
    if not d.exists():
        d.mkdir()
        wet = np.ones((4, lon.size), dtype=bool)
        for var, value in (("uo", 1.0), ("vo", 0.5), ("wo", 0.0)):
            vals = np.where(wet[None], value, np.nan) * np.ones((5, 1, 1))
            _write_levelwise(d / f"{var}.nc", var, vals, lon=lon, lat=lat,
                             unit="hours")
    mesh = _quiet(Fesom14Reader.load_mesh,
                  MeshConfig(type="fesom14", meshpath=str(meshdir),
                             rotation=rotation))
    if factors is not None:
        mesh._vec_rot_factors = factors(mesh._vec_rot_factors)
    mode = ModeConfig(Fdrift3D=three_d)
    with _open(d, mode, "1 h") as ds:
        rec = _quiet(tracking.track_particles,
                     mesh, ds, Fesom14Reader, np.array([0.0]), np.array([5.0]),
                     z0=np.array([5.0]) if three_d else None,
                     start_time_idx=0, n_steps=3, dt=3600.0, integrator="rk2",
                     output_freq=1, mode=mode, uv_interp="barycentric",
                     w_interp="barycentric")
    return rec.to_arrays()


def _chart_diff(a, b) -> float:
    return float(max(np.abs(b["lon"] - a["lon"]).max(),
                     np.abs(b["lat"] - a["lat"]).max()))


def _identity(f):
    """No vector rotation: the factors of the identity."""
    c, s = f["node"]
    return {**f, "node": (np.ones_like(c), np.zeros_like(s))}


def _twice(f):
    """The vector rotation applied twice (twice the angle)."""
    c, s = f["node"]
    return {**f, "node": (c * c - s * s, 2.0 * s * c)}


@pytest.mark.parametrize("three_d", [False, True], ids=["2d", "3d"])
def test_tracking_frame_rotation_trajectory(tmp_path, three_d):
    """``mesh.rotation: fesom2`` and the geographic frame give the same
    trajectory within the chart bound of the node-layout recipe; a missing
    and a doubled vector rotation exceed it."""
    geo = _grid_run(tmp_path, three_d, None)
    rot = _grid_run(tmp_path, three_d, "fesom2")
    d = _chart_diff(geo, rot)
    assert d < 5e-7, d
    # the particle moved (about 0.1 deg east, 0.05 deg north)
    assert geo["lon"][-1, 0] > 0.09 and geo["lat"][-1, 0] > 5.04
    for broken in (_identity, _twice):
        bad = _grid_run(tmp_path, three_d, "fesom2", broken)
        d_bad = _chart_diff(geo, bad)
        assert d_bad > 5e-7, (broken.__name__, d_bad)


# ---------------------------------------------------------------------------
# max_levels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("max_levels", [3, 2])
def test_max_levels_trim(tmp_path, max_levels):
    """u/v/w and 3-D scalars are cut to the kept levels ``mesh.sigma.size``
    after the read."""
    mesh = _hand_mesh(tmp_path)
    full = Fesom14Reader.read_velocities
    _cut(mesh, max_levels)
    assert mesh.sigma.size == max_levels
    mode = ModeConfig(Fdrift3D=True, max_levels=max_levels,
                      sample_vars=["thetao"])
    with _open(_dataset(tmp_path / "data", variables=("uo", "vo", "wo",
                                                       "thetao")), mode) as ds:
        vel = full(ds, 2, mesh, mode, sample_vars=["thetao"])
    for comp, var, sign in (("u", "uo", 1), ("v", "vo", 1), ("w", "wo", -1)):
        arr = getattr(vel, comp)
        assert arr.shape == (max_levels, 5) and arr.flags.c_contiguous
        np.testing.assert_array_equal(arr, sign * _expected(var, 2)[:max_levels])
    th = vel.scalars["thetao"]
    assert th.shape == (max_levels, 5) and th.flags.c_contiguous
    assert np.isfinite(th).all()


# ---------------------------------------------------------------------------
# Every record is checked before anything is changed
# ---------------------------------------------------------------------------

#: (level, node) of the planted defect: by kind, for no cut and below a cut
#: at three levels (level 3 is not kept)
_DEFECT = {
    ("missing", None): (1, 2),     # a wet slot of node 2 (bed 10 m)
    ("extra", None): (2, 2),       # a dry slot of node 2
    ("missing", 3): (3, 0),        # node 0's wet bottom level, below the cut
    ("extra", 3): (3, 1),          # a dry slot of node 1, below the cut
}


@pytest.mark.parametrize("max_levels", [None, 3], ids=["full", "cut3"])
@pytest.mark.parametrize("window", ["first", "prefetch"])
@pytest.mark.parametrize("kind", ["missing", "extra"])
@pytest.mark.parametrize("var", ["uo", "vo", "wo", "thetao", "so"])
def test_corruption_each_variable(tmp_path, var, kind, window, max_levels):
    """A record with no value in a wet slot, or a value in a dry slot, is
    refused, naming the variable, the record's instant, the node and the
    level — in the first window and in a later prefetch window, and also
    when the defect lies below the kept levels (the check runs before the
    cut)."""
    k, n = _DEFECT[(kind, max_levels)]

    def plant(vals):
        vals[6, k, n] = np.nan if kind == "missing" else 7.5

    mesh = _hand_mesh(tmp_path)
    variables = ("uo", "vo", "wo", "thetao", "so")
    d = _dataset(tmp_path / "data", variables=variables, n_time=8,
                 edit={var: plant})
    mode = ModeConfig(Fdrift3D=True, max_levels=max_levels,
                      sample_vars=["thetao", "so"])
    if max_levels is not None:
        _cut(mesh, max_levels)
    with _open(d, mode) as ds:
        Fesom14Reader.prepare(mesh, ds)            # record 0 is clean
        src, t = ((ds, 6) if window == "first"
                  else (ds.isel(time=slice(5, 8)).load(), 1))
        ok = _read(src, t - 1, mesh, mode, sample_vars=["thetao", "so"])
        assert ok.u.shape == (mesh.sigma.size, 5)
        with pytest.raises(ValueError) as info:
            _read(src, t, mesh, mode, sample_vars=["thetao", "so"])
    msg = str(info.value)
    assert f"variable {var!r}" in msg
    assert "2000-01-07" in msg                     # record 6, by its instant
    assert f"node {n}, level {k}" in msg
    assert ("a wet slot" if kind == "missing" else "a dry slot") in msg


@pytest.mark.parametrize("var", ["uo", "vo"])
@pytest.mark.parametrize("mode", [ModeConfig(), ModeConfig(sigma_mean=True,
                                                           max_levels=2)],
                         ids=["level0", "mean_cut2"])
def test_corruption_2d(tmp_path, var, mode):
    """2-D tracking reads one level or the mean of the kept levels, but the
    whole record is checked: a value in a dry slot at 50 m is refused."""
    def plant(vals):
        vals[2, 3, 1] = 7.5                        # node 1 is dry at 50 m

    mesh = _hand_mesh(tmp_path)
    if mode.max_levels is not None:
        _cut(mesh, mode.max_levels, three_d=False)
    d = _dataset(tmp_path / "data", variables=("uo", "vo"), edit={var: plant})
    with _open(d, MODE_2D) as ds:
        _read(ds, 1, mesh, mode)
        with pytest.raises(ValueError,
                           match=rf"{var!r}, record 2000-01-03.*node 1, level 3"):
            _read(ds, 2, mesh, mode)


def test_corruption_nodes3d(tmp_path):
    """A raw fill (1e30) in a vector over the 3-D nodes, in a later prefetch
    window, is refused naming the node and level of that 3-D node."""
    def plant(vals):
        vals[4, 2, 3] = np.nan                     # 3-D node 11: node 3, level 2

    mesh = _hand_mesh(tmp_path)
    d = _dataset(tmp_path / "raw", "nodes3d", n_time=6, edit={"vo": plant})
    with _open(d) as ds:
        Fesom14Reader.prepare(mesh, ds)
        window = ds.isel(time=slice(3, 6)).load()
        _read(window, 0, mesh)
        with pytest.raises(ValueError,
                           match=r"'vo', record 2000-01-05.*node 3, level 2"):
            _read(window, 1, mesh)


def test_nodes3d_length_refused(tmp_path):
    """A read without ``prepare``: a vector of another length than the mesh's
    3-D nodes is refused, never gathered."""
    mesh = _hand_mesh(tmp_path)
    d = tmp_path / "raw14"
    d.mkdir()
    for var in ("uo", "vo", "wo"):
        with netCDF4.Dataset(str(d / f"{var}.nc"), "w") as nc:
            _time(nc, 2)
            nc.createDimension("nodes_3d", 14)
            nc.createVariable(var, "f4", ("time", "nodes_3d"))[:] = np.ones((2, 14))
    with _open(d) as ds:
        with pytest.raises(ValueError, match=r"'uo'.*14 entries.*13"):
            _read(ds, 0, mesh)


# ---------------------------------------------------------------------------
# 2-D tracking on 3-D data
# ---------------------------------------------------------------------------

def test_2d_level_k(tmp_path):
    """``mode.sigma_layer: k`` gives u/v of level k (0 below the bed),
    ``(N,)``, ``"layer"``, no w; neither option gives level 0; a level that
    is not kept is refused, also by the pre-writer probe."""
    mesh = _hand_mesh(tmp_path)
    with _open(_dataset(tmp_path / "data"), MODE_2D) as ds:
        assert "wo" not in ds
        vel = _read(ds, 1, mesh, ModeConfig(sigma_layer=2))
        assert vel.u.shape == (5,) and vel.uv_vertical == "layer"
        assert vel.w is None and vel.u.flags.c_contiguous
        u2 = _values("uo", 2)[1, 2]
        np.testing.assert_array_equal(vel.u, [u2[0], u2[1], 0.0, u2[3], 0.0])
        np.testing.assert_array_equal(vel.v, _expected("vo", 1)[2])
        vel32 = _read(ds, 1, mesh, ModeConfig(sigma_layer=2),
                      velocity_float32=True)
        assert vel32.u.dtype == np.float32
        np.testing.assert_array_equal(vel32.u, vel.u.astype(np.float32))
        for mode in (MODE_2D, ModeConfig(sigma_layer=0), None):
            np.testing.assert_array_equal(_read(ds, 1, mesh, mode).u,
                                          _expected("uo", 1)[0])

        for layer in (4, -1):
            mode = ModeConfig(sigma_layer=layer)
            with pytest.raises(ValueError, match=r"mode\.sigma_layer.*0 \.\. 3"):
                _read(ds, 1, mesh, mode)
            with pytest.raises(ValueError, match=r"mode\.sigma_layer"):
                Fesom14Reader.probe_uv_layout(ds, mode, mesh)
        _cut(mesh, 2, three_d=False)
        with pytest.raises(ValueError, match=r"sigma_layer.*0 \.\. 1.*max_levels"):
            _read(ds, 1, mesh, ModeConfig(sigma_layer=2, max_levels=2))
        np.testing.assert_array_equal(
            _read(ds, 1, mesh, ModeConfig(sigma_layer=1, max_levels=2)).u,
            _expected("uo", 1)[1])


#: u on (level, node), exact in binary; NaN below the bed
U_TABLE = np.array([[0.5, 1.0, -0.25, 2.0, 0.75],
                    [0.25, 0.5, 0.75, 1.0, np.nan],
                    [0.125, -0.5, np.nan, 0.5, np.nan],
                    [1.0, np.nan, np.nan, np.nan, np.nan]])

#: depth means of U_TABLE from the surface to the bed, by hand:
#: trapezoids between wet levels, divided by the bed depth
U_MEAN = np.array([
    ((0.5 + 0.25) / 2 * 10 + (0.25 + 0.125) / 2 * 15 + (0.125 + 1.0) / 2 * 25) / 50,
    ((1.0 + 0.5) / 2 * 10 + (0.5 - 0.5) / 2 * 15) / 25,
    ((-0.25 + 0.75) / 2 * 10) / 10,
    ((2.0 + 1.0) / 2 * 10 + (1.0 + 0.5) / 2 * 15) / 25,
    0.75,                                  # one wet level: its value
])


def _mean_data(tmp_path):
    return _dataset(tmp_path / "data", variables=("uo", "vo"),
                    tables={"uo": U_TABLE, "vo": 2.0 * U_TABLE})


def test_2d_depth_mean(tmp_path):
    """``mode.sigma_mean: true``: the depth mean of each column, checked
    against hand-computed trapezoids, incl. a single-wet-level column."""
    np.testing.assert_allclose(U_MEAN, [0.4125, 0.3, 0.25, 1.05, 0.75],
                               rtol=1e-15)
    mesh = _hand_mesh(tmp_path)
    mode = ModeConfig(sigma_mean=True)
    with _open(_mean_data(tmp_path), MODE_2D) as ds:
        vel = _read(ds, 0, mesh, mode)
        vel32 = _read(ds, 2, mesh, mode, velocity_float32=True)
    assert vel.u.shape == (5,) and vel.uv_vertical == "layer" and vel.w is None
    np.testing.assert_allclose(vel.u, U_MEAN, rtol=1e-14, atol=0)
    np.testing.assert_allclose(vel.v, 2.0 * U_MEAN, rtol=1e-14, atol=0)
    assert vel32.u.dtype == np.float32
    np.testing.assert_allclose(vel32.u, U_MEAN, rtol=1e-7)


@pytest.mark.parametrize("max_levels, expected", [
    # kept 0/10/25 m: node 0 (bed 50 m) gets the constant tail 0.125 x 25 m
    (3, [((0.5 + 0.25) / 2 * 10 + (0.25 + 0.125) / 2 * 15 + 0.125 * 25) / 50,
         0.3, 0.25, 1.05, 0.75]),
    # kept 0/10 m: tails below 10 m wherever the bed is deeper
    (2, [((0.5 + 0.25) / 2 * 10 + 0.25 * 40) / 50,
         ((1.0 + 0.5) / 2 * 10 + 0.5 * 15) / 25,
         0.25,
         ((2.0 + 1.0) / 2 * 10 + 1.0 * 15) / 25,
         0.75]),
    # the surface only: every column's mean is its surface value
    (1, [0.5, 1.0, -0.25, 2.0, 0.75]),
])
def test_depth_mean_cut_column(tmp_path, max_levels, expected):
    """``max_levels`` with ``sigma_mean`` on columns cut above their bed: the
    mean still runs to the bed, the last kept level held constant below."""
    np.testing.assert_allclose(expected[0], {3: 0.19375, 2: 0.275, 1: 0.5}[max_levels],
                               rtol=1e-15)
    mesh = _hand_mesh(tmp_path)
    _cut(mesh, max_levels, three_d=False)
    mode = ModeConfig(sigma_mean=True, max_levels=max_levels)
    with _open(_mean_data(tmp_path), MODE_2D) as ds:
        vel = _read(ds, 1, mesh, mode)
    np.testing.assert_allclose(vel.u, expected, rtol=1e-14, atol=0)


# ---------------------------------------------------------------------------
# Sampled scalars
# ---------------------------------------------------------------------------

#: thetao on (level, node); NaN below the bed
T_TABLE = np.array([[4.0, 5.0, 6.0, 7.0, 8.0],
                    [3.5, 4.5, 5.5, 6.5, np.nan],
                    [3.0, 4.0, np.nan, 6.0, np.nan],
                    [2.5, np.nan, np.nan, np.nan, np.nan]])

#: T_TABLE forward-filled below the bed, by hand
T_FILLED = np.array([[4.0, 5.0, 6.0, 7.0, 8.0],
                     [3.5, 4.5, 5.5, 6.5, 8.0],
                     [3.0, 4.0, 5.5, 6.0, 8.0],
                     [2.5, 4.0, 5.5, 6.0, 8.0]])


def test_scalars_3d_ffill_and_2d_reduction(tmp_path):
    """3-D: ``(K, N)`` forward-filled below the bed (so the sampler never
    meets a NaN at a shallow neighbour), cut to the kept levels; 2-D: the
    reduction of u/v (level k, the depth mean, level 0), ``(N,)``; a 2-D
    field is ``(N,)`` in both."""
    mesh = _hand_mesh(tmp_path)
    zos = ("ncells", 0.25 + 0.125 * np.arange(15.0).reshape(3, 5))
    d = _dataset(tmp_path / "data", variables=("uo", "vo", "wo", "thetao"),
                 tables={"thetao": T_TABLE}, zos=zos)
    names = ["thetao", "zos"]
    mode3 = ModeConfig(Fdrift3D=True, sample_vars=names)
    with _open(d, mode3) as ds:
        out = Fesom14Reader.read_scalar_fields(ds, 1, mesh, names, mode3)
        vel = _read(ds, 1, mesh, mode3, sample_vars=names)
    th = out["thetao"]
    assert th.shape == (4, 5) and th.dtype == np.float64 and th.flags.c_contiguous
    np.testing.assert_array_equal(th, T_FILLED)
    np.testing.assert_array_equal(vel.scalars["thetao"], T_FILLED)
    np.testing.assert_array_equal(out["zos"], zos[1][1])

    # a particle at 40 m inside the triangle of nodes 0, 1, 2 (beds 50, 25,
    # 10 m): nodes 1 and 2 are shallower; the filled field gives a value
    # between the three columns' bottom values, the raw one would give NaN
    lon = np.array([HAND_LON[[0, 1, 2]].mean()])
    lat = np.array([HAND_LAT[[0, 1, 2]].mean()])
    s = sample_field(th, mesh, lon, lat, np.array([40.0]), method="barycentric")
    assert np.isfinite(s).all() and 2.5 < s[0] < 5.5
    raw = sample_field(np.where(WET, T_TABLE, np.nan), mesh, lon, lat,
                       np.array([40.0]), method="barycentric")
    assert np.isnan(raw).all()

    _cut(mesh, 3)
    mode3c = ModeConfig(Fdrift3D=True, max_levels=3, sample_vars=names)
    with _open(d, mode3c) as ds:
        cut = Fesom14Reader.read_scalar_fields(ds, 0, mesh, names, mode3c)
    np.testing.assert_array_equal(cut["thetao"], T_FILLED[:3])
    assert cut["thetao"].flags.c_contiguous

    mesh2 = _hand_mesh(tmp_path / "m2")
    mean = [((4.0 + 3.5) / 2 * 10 + (3.5 + 3.0) / 2 * 15 + (3.0 + 2.5) / 2 * 25) / 50,
            ((5.0 + 4.5) / 2 * 10 + (4.5 + 4.0) / 2 * 15) / 25,
            ((6.0 + 5.5) / 2 * 10) / 10,
            ((7.0 + 6.5) / 2 * 10 + (6.5 + 6.0) / 2 * 15) / 25,
            8.0]
    np.testing.assert_allclose(mean, [3.1, 4.45, 5.75, 6.45, 8.0], rtol=1e-15)
    for mode, want in ((ModeConfig(sigma_layer=2, sample_vars=names), T_FILLED[2]),
                       (ModeConfig(sigma_mean=True, sample_vars=names), mean),
                       (ModeConfig(sample_vars=names), T_FILLED[0])):
        with _open(d, mode) as ds:
            got = Fesom14Reader.read_scalar_fields(ds, 2, mesh2, names, mode)
        assert got["thetao"].shape == (5,)
        np.testing.assert_allclose(got["thetao"], want, rtol=1e-14, atol=0)
        np.testing.assert_array_equal(got["zos"], zos[1][2])


def test_scalar_refusals(tmp_path):
    """A sampled variable that is not in the dataset, and a 2-D field with a
    missing value at some record, are refused naming the variable."""
    mesh = _hand_mesh(tmp_path)
    zos = 0.5 * np.ones((3, 5))
    zos[2, 4] = np.nan
    d = _dataset(tmp_path / "data", zos=("ncells", zos))
    with _open(d) as ds:
        with pytest.raises(ValueError, match=r"'thetao'.*mode\.sample_vars"):
            Fesom14Reader.read_scalar_fields(ds, 0, mesh, ["thetao"], MODE_3D)
        Fesom14Reader.read_scalar_fields(ds, 1, mesh, ["zos"], MODE_3D)
        with pytest.raises(ValueError, match=r"'zos'.*2000-01-03.*node 4"):
            Fesom14Reader.read_scalar_fields(ds, 2, mesh, ["zos"], MODE_3D)


# ---------------------------------------------------------------------------
# The sea-surface height
# ---------------------------------------------------------------------------

def test_eta_optional(tmp_path, capsys):
    """No ``zos``: eta is 0 and the run summary says so in one line;
    present: its values, every record checked; a ``zos`` over the 3-D nodes
    is refused (one value per 2-D node), also by the pre-writer probe."""
    mesh = _hand_mesh(tmp_path)
    capsys.readouterr()
    with _open(_dataset(tmp_path / "none")) as ds:
        for t in range(3):
            eta = _read(ds, t, mesh).eta
            assert eta.dtype == np.float64
            np.testing.assert_array_equal(eta, np.zeros(5))
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert len(lines) == 1, lines
    assert "sea-surface height" in lines[0] and "'zos'" in lines[0]

    zos = 0.25 + 0.125 * np.arange(15.0).reshape(3, 5)
    for form, dim in (("levelwise", "ncells"), ("nodes3d", "nodes_2d")):
        with _open(_dataset(tmp_path / form, form, zos=(dim, zos))) as ds:
            capsys.readouterr()
            np.testing.assert_array_equal(
                _read(ds, 2, mesh, velocity_float32=True).eta, zos[2])
            assert capsys.readouterr().out == ""

    bad = zos.copy()
    bad[1, 3] = np.nan
    with _open(_dataset(tmp_path / "hole", zos=("ncells", bad))) as ds:
        _read(ds, 0, mesh)
        with pytest.raises(ValueError, match=r"'zos'.*2000-01-02.*node 3"):
            _read(ds, 1, mesh)

    wide = 0.5 * np.ones((3, 13))
    with _open(_dataset(tmp_path / "wide", "nodes3d",
                        zos=("nodes_3d", wide))) as ds:
        Fesom14Reader.prepare(mesh, ds)            # a vector over the 3-D nodes
        for call in (lambda: _read(ds, 0, mesh),
                     lambda: Fesom14Reader.probe_uv_layout(ds, MODE_3D, mesh)):
            with pytest.raises(ValueError) as info:
                call()
            msg = str(info.value)
            assert "'zos'" in msg and "13" in msg and "5 2-D nodes" in msg
