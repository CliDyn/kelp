"""FESOM 1.4 runs end to end: config keys, the tracking time rule, ``kelp info``.

The FESOM 1.4 reader resolves what time each record represents (the centre of
its averaging window for means, the stamp for snapshots), so a run takes the
reader's time axis as it is: no midpoint shift, ``data.time_averaging``
refused (it points to ``data.time_stamp`` / ``data.time_window``), and the
output metadata says ``time_averaging = "resolved_by_reader"``.  The stamp
options ``data.time_stamp`` / ``data.time_window`` / ``data.time_offset``
reach ``open_dataset`` only for ``mesh.type: fesom14``; with any other mesh
type they are refused by name.  Also pinned here: the refusals that must come
before the output file exists (``beaching.use_eta`` without a sea-surface
height, em/m1 diffusion), ``mesh.input_rotation`` with other mesh types,
``kelp info`` with the stamp options, the sharding concat, and the W
reconstruction's refusal of data on the levels.

Fixture: a 5 x 5 grid of nodes one degree apart (10..14 E, 40..44 N),
triangulated, with levels at 0/10/25/50 m; every column is four levels deep
except the northern row (two levels).  Level-wise files ``var(time, depth,
ncells)`` carry the node coordinates and the cell polygons (the mesh is
recovered from them unless a test writes an ASCII mesh).  Daily means with
time bounds from 2000-01-01 (``proleptic_gregorian``): record ``i`` covers day
``i``, its instant is 2000-01-01 12:00 + ``i`` days.  On the levels ``u = 0.1
+ 0.02 i + 0.002 z`` (m/s) and ``v = w = 0``, so a particle at a fixed depth
moves east by a closed-form distance: between records u is linear in time and
RK2 integrates it exactly.  Expected positions and dates are computed here
from that formula, never from the reader.  All data is synthetic.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest
import yaml

netCDF4 = pytest.importorskip("netCDF4")
cftime = pytest.importorskip("cftime")

from kelp import tracking  # noqa: E402
from kelp.cli.main import main as cli_main  # noqa: E402
from kelp.core.config import (AppConfig, BeachingConfig, DataConfig,  # noqa: E402
                              DiffusionConfig, MeshConfig, ModeConfig,
                              NumericsConfig, TimeConfig, load_config)
from kelp.core.geometry import METERS_PER_DEG_LAT  # noqa: E402
from kelp.core.mesh import find_nearest_nodes  # noqa: E402
from kelp.core.particles import ParticleStatus, make_state  # noqa: E402
from kelp.readers import extract_time_info  # noqa: E402
from kelp.readers.fesom14 import Fesom14Reader  # noqa: E402
from kelp.w_reconstruct import (compute_divergence_approx,  # noqa: E402
                                compute_divergence_with_area, reconstruct_w)
from kelp.writers import TrajectoryWriter, load_restart, load_trajectories  # noqa: E402

from tests._bitwise import assert_bitwise  # noqa: E402
from tests.test_fesom14_ascii_mesh import _write as _write_hand_mesh  # noqa: E402
from tests.test_fesom14_mesh import _grid, _write_ascii  # noqa: E402
from tests.test_fesom14_open import _write_nc  # noqa: E402
from tests.test_fesom14_polygons import _polygons_from_triangulation  # noqa: E402
from tests.test_w_reconstruct import _make_mesh  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

LEVELS = np.array([0.0, 10.0, 25.0, 50.0])
DAY = 86400.0
CAL = "proleptic_gregorian"

#: the grid: 25 nodes, 10..14 E x 40..44 N, counter-clockwise triangles
LON, LAT, TRI = _grid(10.0, 40.0, 5, 5)
#: wet levels per column: four, the northern row two
NCOL = np.where(LAT > 43.5, 2, 4)
WET = np.arange(LEVELS.size)[:, None] < NCOL[None, :]
LON_B, LAT_B = _polygons_from_triangulation(LON, LAT, TRI, nv=14)

#: the instant of record 0 of the daily means (the centre of 2000-01-01)
T0 = cftime.datetime(2000, 1, 1, 12, calendar=CAL)

#: seeds (lon, lat, z): every vertex of their elements is four levels deep
SEEDS = [(10.6, 41.5, 5.0), (11.3, 42.2, 40.0)]


# ---------------------------------------------------------------------------
# Fields, files, configs
# ---------------------------------------------------------------------------

def _u(t, z, lon, lat):
    """u on the levels (m/s) at record ``t``."""
    return 0.1 + 0.02 * t + 0.002 * z + 0.0 * lon


def _zero(t, z, lon, lat):
    return 0.0 * (t + z + lon + lat)


def _sheared_u(t, z, lon, lat):
    return 0.1 + 0.02 * t + 0.002 * z + 0.03 * (lat - 42.0)


def _sheared_v(t, z, lon, lat):
    return -0.05 + 0.01 * (lon - 12.0) + 0.0005 * z + 0.002 * t


def _sinking_w(t, z, lon, lat):
    """``wo`` as written (positive up): sinking by about 1 m per day."""
    return -1.2e-5 + 0.0 * (t + z + lon + lat)


FIELDS = {"uo": _u, "vo": _zero, "wo": _zero}
SHEARED = {"uo": _sheared_u, "vo": _sheared_v, "wo": _sinking_w}


def _time_vars(n_time: int, bounds: bool):
    """Daily means with bounds (stamps at 12:00), or bounds-free stamps at
    00:00 (only the stamp options can say what they are)."""
    i = np.arange(n_time, dtype=np.float64)
    attrs = {"units": "days since 2000-01-01", "calendar": CAL,
             "standard_name": "time"}
    if not bounds:
        return [("time", ("time",), i, attrs)], {}
    attrs["bounds"] = "time_bnds"
    return ([("time", ("time",), i + 0.5, attrs),
             ("time_bnds", ("time", "bnds"), np.stack([i, i + 1.0], axis=1),
              {})], {"bnds": 2})


def _write_var(path: Path, var: str, fn, *, n_time: int, bounds: bool,
               polygons: bool) -> Path:
    t = np.arange(n_time)[:, None, None]
    z = LEVELS[None, :, None]
    vals = np.where(WET[None], fn(t, z, LON[None, None, :], LAT[None, None, :]),
                    np.nan)
    tvars, tdims = _time_vars(n_time, bounds)
    attrs = {"_FillValue": 1e20}
    if var == "wo":
        attrs["standard_name"] = "upward_sea_water_velocity"
    variables = tvars + [
        ("depth", ("depth",), LEVELS, {"units": "m", "positive": "down"}),
        ("lon", ("ncells",), LON, {"standard_name": "longitude",
                                   "bounds": "lon_bnds"}),
        ("lat", ("ncells",), LAT, {"standard_name": "latitude",
                                   "bounds": "lat_bnds"}),
        (var, ("time", "depth", "ncells"), vals, attrs),
    ]
    dims = {"time": None, **tdims, "depth": LEVELS.size, "ncells": LON.size}
    if polygons:
        dims["vertices"] = LON_B.shape[1]
        variables += [("lon_bnds", ("ncells", "vertices"), LON_B, {}),
                      ("lat_bnds", ("ncells", "vertices"), LAT_B, {})]
    return _write_nc(path, dims, variables)


def _write_zos(path: Path, n_time: int) -> Path:
    tvars, tdims = _time_vars(n_time, True)
    return _write_nc(path, {"time": None, **tdims, "ncells": LON.size},
                     tvars + [("zos", ("time", "ncells"),
                               np.zeros((n_time, LON.size)),
                               {"_FillValue": 1e20})])


def _data(d: Path, fields=FIELDS, *, n_time: int = 6, bounds: bool = True,
          zos: bool = False) -> Path:
    """One level-wise file per variable; the u file holds the polygons."""
    if not d.exists():
        d.mkdir(parents=True)
        for var, fn in fields.items():
            _write_var(d / f"{var}.nc", var, fn, n_time=n_time, bounds=bounds,
                       polygons=(var == "uo"))
        if zos:
            _write_zos(d / "zos.nc", n_time)
    return d


def _ascii_mesh(d: Path) -> Path:
    if not d.exists():
        _write_ascii(d, LON, LAT, TRI, NCOL, LEVELS)
    return d


def _seeds(path: Path, seeds=SEEDS) -> Path:
    path.write_text("".join(f"{lo} {la} {z}\n" for lo, la, z in seeds))
    return path


def _cfg(tmp_path: Path, *, data: Path = None, meshpath: Path = None,
         out: str = "traj.nc", seeds=SEEDS, n_steps: int = 3,
         mode: dict = None, numerics: dict = None, **data_kw) -> AppConfig:
    """A 3-D RK2 run from the first instant, dt = 1 day, lerp in time."""
    data = data if data is not None else _data(tmp_path / "data")
    mode_kw = dict(Fdrift2D=False, Fdrift3D=True)
    mode_kw.update(mode or {})
    num_kw = dict(dt=DAY, integrator="rk2", uv_interp="barycentric",
                  w_interp="barycentric", temporal_interp=True, output_freq=1)
    num_kw.update(numerics or {})
    return AppConfig(
        mode=ModeConfig(**mode_kw),
        numerics=NumericsConfig(**num_kw),
        mesh=MeshConfig(type="fesom14",
                        meshpath=str(meshpath if meshpath is not None else data)),
        data=DataConfig(ncinput=str(data),
                        inicloud=str(_seeds(tmp_path / f"seeds_{out}.dat",
                                            seeds)),
                        output_path=str(tmp_path / out), **data_kw),
        time=TimeConfig(start_date="2000-01-01T12:00", n_steps=n_steps),
    )


def _run(cfg: AppConfig) -> str:
    """run_tracking with the summary captured; returns the summary."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tracking.run_tracking(cfg)
    return buf.getvalue()


def _quiet(fn, *args, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kw)


def _output_dates(path: str):
    """The record dates and the release dates of an output, decoded here
    with ``cftime`` from the file's own units and calendar."""
    with netCDF4.Dataset(path) as nc:
        tv, rv = nc["time"], nc["release_time"]
        dates = cftime.num2date(tv[:], tv.units, tv.calendar,
                                only_use_cftime_datetimes=True)
        release = cftime.num2date(rv[:], rv.units, rv.calendar,
                                  only_use_cftime_datetimes=True)
        attrs = {k: nc.getncattr(k) for k in nc.ncattrs()}
    return list(dates), list(release), attrs


def _lon_after(lon0: float, lat0: float, days: float, z_term: float) -> float:
    """Where u = 0.1 + 0.02 t + z_term (t in days from the first instant)
    takes a particle in ``days``: v = 0 keeps the latitude."""
    east = ((0.1 + z_term) * days + 0.01 * days ** 2) * DAY
    return lon0 + east / (np.cos(np.deg2rad(lat0)) * METERS_PER_DEG_LAT)


def _expected_lon(seeds, z_terms, n_steps, step_days=1.0):
    return np.array([[_lon_after(lo, la, (k + 1) * step_days, zt)
                      for k in range(n_steps)]
                     for (lo, la, _), zt in zip(seeds, z_terms)])


# ---------------------------------------------------------------------------
# The time rule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hours, n_steps", [(24, 3), (6, 8)])
def test_run_tracking_levelwise_3d(tmp_path, hours, n_steps):
    """3-D RK2 barycentric with lerp in time, forward, output every step,
    started at the first resolved instant (2000-01-01 12:00): the release is
    at that instant, record ``k`` (0-based) at that instant + (k + 1) dt (a
    record is written after each step), the positions follow the closed
    form, and the metadata says ``resolved_by_reader``.  dt = 1 day is the
    record spacing; dt = 6 h is the case that sees a half-day shift of the
    axis in the positions (with dt = 1 day a shifted axis falls back to the
    velocity at the step start, the mean of the two records: the exact
    integral again)."""
    cfg = _cfg(tmp_path, n_steps=n_steps, numerics={"dt": hours * 3600.0})
    _run(cfg)
    with _quiet(Fesom14Reader.open_dataset, cfg.data.ncinput,
                mode_cfg=cfg.mode) as ds:
        assert ds["time"].values[0] == T0
        assert ds.attrs["fesom14_time_source"] == "bounds"
    dates, release, attrs = _output_dates(cfg.data.output_path)
    if hours == 24:
        assert dates == [cftime.datetime(2000, 1, 2, 12, calendar=CAL),
                         cftime.datetime(2000, 1, 3, 12, calendar=CAL),
                         cftime.datetime(2000, 1, 4, 12, calendar=CAL)]
    else:
        assert dates[0] == cftime.datetime(2000, 1, 1, 18, calendar=CAL)
        assert dates[-1] == cftime.datetime(2000, 1, 3, 12, calendar=CAL)
    assert dates == [T0 + timedelta(hours=(k + 1) * hours)
                     for k in range(n_steps)]
    assert release == [T0, T0]
    assert attrs["time_averaging"] == "resolved_by_reader"
    tr = load_trajectories(cfg.data.output_path)
    assert (tr["status"] == ParticleStatus.ALIVE).all()
    # u at 5 m is a + 0.01, at 40 m a + 0.08 (between the 25 and 50 m levels)
    np.testing.assert_allclose(
        tr["lon"], _expected_lon(SEEDS, [0.01, 0.08], n_steps, hours / 24.0),
        rtol=0, atol=1e-10)
    np.testing.assert_array_equal(tr["lat"], [[41.5] * n_steps,
                                              [42.2] * n_steps])
    np.testing.assert_array_equal(tr["z"], [[5.0] * n_steps,
                                            [40.0] * n_steps])


def test_no_warning_for_fesom14(tmp_path, monkeypatch):
    """No time_averaging warning, the midpoint shift never runs, and the run
    summary names what resolved the record times and the first and last
    instants; with bounds-free files the stamp options resolve them."""
    def no_shift(_times):
        raise AssertionError("the midpoint shift ran on a FESOM 1.4 axis")
    monkeypatch.setattr(tracking, "_shift_to_midpoints", no_shift)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _run(_cfg(tmp_path))
    assert not [w for w in caught if "time_averaging" in str(w.message)], \
        [str(w.message) for w in caught]
    line = [ln for ln in out.splitlines() if "Record times" in ln]
    assert len(line) == 1, out
    assert "time bounds" in line[0]
    assert "2000-01-01 12:00:00" in line[0] and "2000-01-06 12:00:00" in line[0]

    raw = _data(tmp_path / "raw", bounds=False)
    cfg = _cfg(tmp_path, data=raw, out="opts.nc", time_stamp="instant",
               time_window="1 d")
    cfg.time.start_date = "2000-01-01T00:00"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _run(cfg)
    assert not [w for w in caught if "time_averaging" in str(w.message)]
    line = [ln for ln in out.splitlines() if "Record times" in ln]
    assert len(line) == 1 and "data.time_stamp" in line[0], out
    assert "2000-01-01 00:00:00" in line[0] and "2000-01-06 00:00:00" in line[0]
    dates, _, attrs = _output_dates(cfg.data.output_path)
    assert dates[0] == cftime.datetime(2000, 1, 2, calendar=CAL)
    assert attrs["time_averaging"] == "resolved_by_reader"


def test_time_offset_reaches_the_reader(tmp_path):
    """``data.time_offset`` moves the raw stamps before anything else: the
    bounds-free stamps at 00:00 shifted by +12 h are instants at 12:00, so a
    run from 2000-01-01 12:00 starts on record 0."""
    raw = _data(tmp_path / "raw", bounds=False)
    cfg = _cfg(tmp_path, data=raw, time_stamp="instant", time_window="1 d",
               time_offset="+12 h")
    out = _run(cfg)
    assert "data.time_offset +12 h" in out
    dates, release, _ = _output_dates(cfg.data.output_path)
    assert release == [T0, T0]
    assert dates == [T0 + timedelta(days=k + 1) for k in range(3)]
    tr = load_trajectories(cfg.data.output_path)
    np.testing.assert_allclose(tr["lon"], _expected_lon(SEEDS, [0.01, 0.08], 3),
                               rtol=0, atol=1e-10)


@pytest.mark.parametrize("value", ["mean", "instant"])
def test_time_averaging_set_refused(tmp_path, value):
    """``data.time_averaging`` with ``mesh.type: fesom14``: refused before
    anything is written, pointing to the stamp options."""
    cfg = _cfg(tmp_path, time_averaging=value)
    with pytest.raises(ValueError) as info:
        _run(cfg)
    msg = str(info.value)
    for needle in ("data.time_averaging", "fesom14", "data.time_stamp",
                   "data.time_window"):
        assert needle in msg, msg
    assert not Path(cfg.data.output_path).exists()


@pytest.mark.parametrize("mesh_type", ["fesom2", "fesomc", "healpix", "nemo"])
@pytest.mark.parametrize("keys", [
    {"time_stamp": "end", "time_window": "calendar_month"},
    {"time_offset": "+6 h"},
], ids=["stamp_and_window", "offset"])
def test_time_keys_rejected_for_other_readers(tmp_path, mesh_type, keys):
    """The stamp options with any other mesh type: a TypeError naming each
    key set and the mesh type, before the mesh path is even looked at."""
    cfg = AppConfig(
        mode=ModeConfig(), numerics=NumericsConfig(dt=3600.0),
        mesh=MeshConfig(type=mesh_type, meshpath=str(tmp_path / "absent")),
        data=DataConfig(ncinput=str(tmp_path / "absent"),
                        inicloud=str(tmp_path / "absent.dat"),
                        time_averaging="instant", **keys),
        time=TimeConfig(n_steps=1))
    with pytest.raises(TypeError) as info:
        tracking.run_tracking(cfg)
    msg = str(info.value)
    for key in keys:
        assert f"data.{key}" in msg, msg
    assert repr(mesh_type) in msg and "fesom14" in msg, msg


@pytest.mark.parametrize("keys, needles", [
    ({"time_stamp": "end"}, ["data.time_stamp", "data.time_window"]),
    ({"time_window": "calendar_month"}, ["data.time_window", "data.time_stamp"]),
    ({"time_stamp": "middle", "time_window": "calendar_month"},
     ["data.time_stamp", "middle"]),
    ({"time_stamp": "end", "time_window": "monthly"},
     ["data.time_window", "monthly"]),
    ({"time_offset": "6 hours"}, ["data.time_offset", "6 hours"]),
])
def test_time_keys_validated_in_config(keys, needles):
    """The keys are checked when the config is built: the incomplete pair and
    malformed values are refused by name."""
    with pytest.raises(ValueError) as info:
        DataConfig(**keys)
    for needle in needles:
        assert needle in str(info.value), str(info.value)


def test_time_keys_from_yaml(tmp_path):
    """``load_config`` keeps the three keys (they are not dropped as
    unknown keys) and their defaults are None."""
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({
        "mode": {}, "numerics": {}, "time": {"n_steps": 1},
        "mesh": {"type": "fesom14", "meshpath": "m"},
        "data": {"ncinput": "d", "time_stamp": "centre",
                 "time_window": "calendar_month", "time_offset": "-6 h"}}))
    data = load_config(str(p)).data
    assert (data.time_stamp, data.time_window, data.time_offset) == (
        "centre", "calendar_month", "-6 h")
    assert DataConfig().time_stamp is None
    assert DataConfig().time_window is None and DataConfig().time_offset is None


# ---------------------------------------------------------------------------
# Restart
# ---------------------------------------------------------------------------

def test_restart_continuous_equals_resumed_fesom14(tmp_path):
    """A run cut at a checkpoint and resumed equals the continuous run bit
    for bit (record times, positions, statuses, recorded velocities, the
    final restart state), on a sheared, sinking flow with lerp in time."""
    data = _data(tmp_path / "data", SHEARED)
    mode = ModeConfig(Fdrift2D=False, Fdrift3D=True)
    mesh = _quiet(Fesom14Reader.load_mesh,
                  MeshConfig(type="fesom14", meshpath=str(data)), mode_cfg=mode)
    ds = _quiet(Fesom14Reader.open_dataset, str(data), mode_cfg=mode)
    ft, fdt = extract_time_info(ds)
    num = NumericsConfig(dt=DAY, integrator="rk2", uv_interp="barycentric",
                         w_interp="barycentric", temporal_interp=True,
                         output_freq=1)

    def run(name, state, n, step0):
        path = str(tmp_path / f"{name}.nc")
        w = TrajectoryWriter(path, state.n_particles).open()
        try:
            _quiet(tracking.run_single_realization,
                   mesh, ds, Fesom14Reader, state, mode, num,
                   start_time_idx=step0, n_steps=n,
                   n_time_total=ds.sizes["time"], writer=w,
                   checkpoint_freq=2, temporal_interp=True, file_times=ft,
                   file_dt=fdt, sim_time_start=step0 * DAY,
                   start_step_i=step0)
        finally:
            w.close()
        return path

    lon, lat, z = (np.array(c, dtype=np.float64) for c in zip(*SEEDS))
    init = make_state(lon, lat, z)
    init = init.replace(nearest_node=find_nearest_nodes(mesh, lon, lat))
    whole = run("whole", init, 4, 0)
    a = run("a", init, 2, 0)
    r = load_restart(a)
    n = r["step_i"] + 1
    b = run("b", r["state"], 2, n)
    tw, ta, tb = (load_trajectories(p) for p in (whole, a, b))
    assert_bitwise(np.concatenate([ta["time"], tb["time"]]), tw["time"], "time")
    for k in ("lon", "lat", "z", "status", "vel_u", "vel_v", "vel_w"):
        assert_bitwise(np.concatenate([ta[k], tb[k]], axis=1), tw[k], k)
    assert np.ptp(tw["z"]) > 1.0 and np.ptp(tw["lat"]) > 0.01   # it moves
    assert (tw["status"] == ParticleStatus.ALIVE).all()
    rw, rb = load_restart(whole), load_restart(b)
    assert (rb["step_i"], rb["file_idx"]) == (rw["step_i"], rw["file_idx"])
    for k in ("lon", "lat", "z"):
        assert_bitwise(getattr(rb["state"], k), getattr(rw["state"], k), k)
    assert _output_dates(b)[0] == [T0 + timedelta(days=3), T0 + timedelta(days=4)]
    ds.close()


# ---------------------------------------------------------------------------
# max_levels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["ascii", "polygons"])
def test_run_tracking_max_levels_fesom14(tmp_path, kind, monkeypatch):
    """3-D, ``max_levels: 3`` (levels 0/10/25 m kept): a particle at 40 m,
    below the last kept level, sees the 25 m value held constant (a + 0.05,
    not the a + 0.08 between 25 and 50 m); the one at 5 m is unchanged.
    ``prepare`` runs on the cut mesh and passes (it checks the full table)."""
    data = _data(tmp_path / "data")
    meshpath = _ascii_mesh(tmp_path / "mesh") if kind == "ascii" else data
    calls = []
    real = Fesom14Reader.prepare

    def spy(mesh, ds):
        calls.append((mesh.sigma.size,
                      mesh.metadata["fesom14_levels"].size))
        return real(mesh, ds)
    monkeypatch.setattr(Fesom14Reader, "prepare", staticmethod(spy))
    cfg = _cfg(tmp_path, data=data, meshpath=meshpath,
               mode={"max_levels": 3})
    _run(cfg)
    assert calls and set(calls) == {(3, 4)}, calls
    tr = load_trajectories(cfg.data.output_path)
    assert (tr["status"] == ParticleStatus.ALIVE).all()
    np.testing.assert_array_equal(tr["z"], [[5.0] * 3, [40.0] * 3])
    np.testing.assert_allclose(tr["lon"], _expected_lon(SEEDS, [0.01, 0.05], 3),
                               rtol=0, atol=1e-10)


@pytest.mark.parametrize("max_levels, z_term", [(None, 0.05), (3, 0.0375)])
def test_run_tracking_sigma_mean_cut_column(tmp_path, max_levels, z_term):
    """2-D with ``sigma_mean``: the depth mean of u = a + 0.002 z to the 50 m
    bed is a + 0.05; with ``max_levels: 3`` the column is cut at 25 m and the
    25 m value is held to the bed: (10 (2a + 0.02) / 2 + 15 (2a + 0.07) / 2
    + 25 (a + 0.05)) / 50 = a + 0.0375."""
    cfg = _cfg(tmp_path, mode={"Fdrift2D": True, "Fdrift3D": False,
                               "sigma_mean": True, "max_levels": max_levels})
    _run(cfg)
    tr = load_trajectories(cfg.data.output_path)
    assert (tr["status"] == ParticleStatus.ALIVE).all()
    np.testing.assert_allclose(tr["lon"],
                               _expected_lon(SEEDS, [z_term, z_term], 3),
                               rtol=0, atol=1e-10)


# ---------------------------------------------------------------------------
# Refusals before the writer
# ---------------------------------------------------------------------------

def test_use_eta_without_zos_refused(tmp_path):
    """``beaching.use_eta: true`` without a sea-surface height in the files:
    refused before the output exists, naming the variable and the fix; with
    the ``zos`` files the same run goes through."""
    beach = BeachingConfig(enabled=True, criteria=["depth"], use_eta=True)
    cfg = _cfg(tmp_path, numerics={"beaching": beach})
    with pytest.raises(ValueError) as info:
        _run(cfg)
    msg = str(info.value)
    for needle in ("beaching.use_eta", "'zos'", "data.ncinput"):
        assert needle in msg, msg
    assert not Path(cfg.data.output_path).exists()

    cfg = _cfg(tmp_path, data=_data(tmp_path / "with_zos", zos=True),
               out="zos.nc", numerics={"beaching": beach})
    _run(cfg)
    assert (load_trajectories(cfg.data.output_path)["status"]
            == ParticleStatus.ALIVE).all()


@pytest.mark.parametrize("scheme, integrator, composition", [
    ("em", "rk2", "split_rk"), ("m1", "euler", "split")])
def test_em_m1_diffusion_refused_before_writer(tmp_path, scheme, integrator,
                                               composition):
    """em/m1 diffusion reads spatial Kh fields, which FESOM 1.4 output does
    not carry: refused before the output exists; uniform Kh runs."""
    diff = DiffusionConfig(enabled=True, scheme=scheme, composition=composition,
                           var_kh_zonal="kh", var_kh_meridional="kh")
    cfg = _cfg(tmp_path, mode={"Fdrift2D": True, "Fdrift3D": False},
               numerics={"integrator": integrator, "diffusion": diff})
    with pytest.raises(ValueError) as info:
        _run(cfg)
    msg = str(info.value)
    for needle in ("fesom14", scheme, "uniform"):
        assert needle in msg, msg
    assert not Path(cfg.data.output_path).exists()

    uniform = DiffusionConfig(enabled=True, scheme="uniform", kh_zonal=10.0,
                              kh_meridional=10.0, seed=1)
    cfg = _cfg(tmp_path, out="uniform.nc", numerics={"diffusion": uniform})
    _run(cfg)
    tr = load_trajectories(cfg.data.output_path)
    assert np.isfinite(tr["lon"]).all()
    assert not np.array_equal(tr["lon"], _expected_lon(SEEDS, [0.01, 0.08], 3))


@pytest.mark.parametrize("mesh_type", ["fesomc", "fesom2", "healpix", "nemo"])
def test_input_rotation_refused_for_other_mesh_types(mesh_type):
    with pytest.raises(ValueError) as info:
        MeshConfig(type=mesh_type, meshpath="m", input_rotation=[50.0, 15.0, -90.0])
    msg = str(info.value)
    assert "mesh.input_rotation" in msg and repr(mesh_type) in msg, msg
    assert "fesom14" in msg, msg
    # the default type is not fesom14 either
    with pytest.raises(ValueError, match="mesh.input_rotation"):
        MeshConfig(input_rotation=[0.0, 0.0, 0.0])
    assert MeshConfig(type="fesom14", input_rotation=[1, 2, 3]).input_rotation \
        == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# kelp info
# ---------------------------------------------------------------------------

def _info(tmp_path: Path, name: str, cfg: dict, capsys):
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(cfg))
    capsys.readouterr()
    rc = cli_main(["info", str(p)])
    out = capsys.readouterr()
    return rc, out.out, out.err


def _info_cfg(meshpath, data, **data_kw) -> dict:
    return {"mode": {"Fdrift2D": False, "Fdrift3D": True},
            "numerics": {"uv_interp": "barycentric"},
            "time": {"n_steps": 1},
            "mesh": {"type": "fesom14", "meshpath": str(meshpath)},
            "data": {"ncinput": str(data), **data_kw}}


def test_kelp_info_options(tmp_path, capsys):
    """``kelp info`` on bounds-free files: exit 1 with the options message;
    with ``data.time_stamp``/``data.time_window`` it prints the resolved
    first and last instants and the time source; a mesh the data does not
    fit is refused by ``prepare``; the options with another mesh type are
    refused by name."""
    raw = _data(tmp_path / "raw", bounds=False)
    rc, out, err = _info(tmp_path, "bare", _info_cfg(raw, raw), capsys)
    assert rc == 1, (out, err)
    assert "data.time_stamp" in err and "data.time_window" in err, err

    rc, out, err = _info(tmp_path, "opts", _info_cfg(
        raw, raw, time_stamp="instant", time_window="1 d"), capsys)
    assert rc == 0, (out, err)
    assert "2000-01-01 00:00:00" in out and "2000-01-06 00:00:00" in out, out
    assert "6 steps" in out
    line = [ln for ln in out.splitlines() if "record times" in ln.lower()]
    assert len(line) == 1 and "data.time_stamp" in line[0], out

    bnds = _data(tmp_path / "bnds")
    rc, out, err = _info(tmp_path, "bnds", _info_cfg(bnds, bnds), capsys)
    assert rc == 0, (out, err)
    assert "2000-01-01 12:00:00" in out and "2000-01-06 12:00:00" in out, out
    assert "time bounds" in out

    # every column four levels deep: the data's northern row is dry below
    # 10 m, prepare refuses (record 0 of uo has no value in a wet slot)
    deep = tmp_path / "deep_mesh"
    _write_ascii(deep, LON, LAT, TRI, np.full(LON.size, 4), LEVELS)
    rc, out, err = _info(tmp_path, "deep", _info_cfg(deep, bnds), capsys)
    assert rc == 1, (out, err)
    assert "does not fit the mesh" in err and "'uo'" in err, err
    assert "2000-01-01T12:00:00" in err and "node 20" in err, err

    rc, out, err = _info(tmp_path, "avg", _info_cfg(
        bnds, bnds, time_averaging="mean"), capsys)
    assert rc == 1, (out, err)
    assert "data.time_averaging" in err and "data.time_stamp" in err, err

    cfg = _info_cfg(tmp_path / "absent", raw, time_stamp="instant",
                    time_window="1 d")
    cfg["mesh"]["type"] = "fesom2"
    rc, out, err = _info(tmp_path, "other", cfg, capsys)
    assert rc == 1, (out, err)
    assert "data.time_stamp" in err and "'fesom2'" in err, err


def test_kelp_info_reads_split_paths(tmp_path, capsys):
    """``data.ncinput_v``/``data.ncinput_w`` reach ``open_dataset`` as in a
    run: u, v and w in three directories."""
    src = _data(tmp_path / "all")
    dirs = {}
    for var in ("uo", "vo", "wo"):
        dirs[var] = tmp_path / var
        dirs[var].mkdir()
        (src / f"{var}.nc").rename(dirs[var] / f"{var}.nc")
    rc, out, err = _info(tmp_path, "split", _info_cfg(
        dirs["uo"], dirs["uo"], ncinput_v=str(dirs["vo"]),
        ncinput_w=str(dirs["wo"])), capsys)
    assert rc == 0, (out, err)
    assert "2000-01-06 12:00:00" in out


# ---------------------------------------------------------------------------
# Sharding
# ---------------------------------------------------------------------------

def test_sharding_concat_accepts_resolved_by_reader(tmp_path):
    """Two shards of a FESOM 1.4 run merge (their ``time_averaging`` is the
    same ``resolved_by_reader``) into the unsharded run, value for value."""
    zarr = pytest.importorskip("zarr")
    sys.path.insert(0, str(REPO / "utils"))
    from concat_trajectories import concat_trajectories
    from split_seeds import split_seeds
    seeds = SEEDS + [(12.4, 41.2, 12.0), (12.9, 42.6, 30.0)]
    seed_file = _seeds(tmp_path / "all.dat", seeds)
    whole = _cfg(tmp_path, out="whole.zarr", seeds=seeds)
    _run(whole)
    with pytest.warns(UserWarning, match="sharded runs"):
        files = split_seeds(seed_file, 2, outdir=tmp_path / "sh")
    outs = []
    for k, f in enumerate(files):
        cfg = _cfg(tmp_path, out=f"s{k}.zarr", seeds=seeds)
        cfg.data.inicloud = str(f)
        _run(cfg)
        outs.append(Path(cfg.data.output_path))
    merged = tmp_path / "merged.zarr"
    concat_trajectories(outs, merged)
    gw = zarr.open_group(whole.data.output_path, mode="r")
    gm = zarr.open_group(str(merged), mode="r")
    assert gm.attrs["time_averaging"] == "resolved_by_reader"
    assert gw.attrs["time_averaging"] == "resolved_by_reader"
    for name in gw.array_keys():
        if name.startswith("restart_") or name == "trajectory":
            continue
        assert np.array_equal(np.asarray(gw[name]), np.asarray(gm[name]),
                              equal_nan=True), name


# ---------------------------------------------------------------------------
# W reconstruction: layer data only
# ---------------------------------------------------------------------------

def test_w_reconstruct_refuses_level_data(tmp_path):
    """The W reconstruction integrates layer-centred u/v: a FESOM 1.4 mesh
    (whose output carries w) and u/v declared on the levels are refused;
    layer data on another mesh still runs."""
    mesh = _quiet(Fesom14Reader.load_mesh, MeshConfig(
        type="fesom14", meshpath=str(_write_hand_mesh(tmp_path / "mesh"))))
    k, e, n = mesh.sigma.size, mesh.n_elements, mesh.n_nodes
    u = np.zeros((k - 1, e))
    calls = [
        lambda: reconstruct_w(u, u, np.zeros(n), np.zeros(n), DAY, mesh),
        lambda: compute_divergence_approx(u, u, mesh),
        lambda: compute_divergence_with_area(u, u, mesh, np.ones((k, n))),
    ]
    for call in calls:
        with pytest.raises(ValueError) as info:
            call()
        msg = str(info.value)
        assert "FESOM 1.4" in msg and "'wo'" in msg and "layer" in msg, msg

    other = _make_mesh()
    k, e, n = other.sigma.size, other.n_elements, other.n_nodes
    u = np.zeros((k - 1, e))
    with pytest.raises(ValueError, match="on the levels"):
        reconstruct_w(u, u, np.zeros(n), np.zeros(n), DAY, other,
                      uv_vertical="level")
    with pytest.raises(ValueError, match="on the levels"):
        compute_divergence_approx(u, u, other, uv_vertical="level")
    with pytest.raises(ValueError, match="on the levels"):
        compute_divergence_with_area(u, u, other, np.ones((k, n)),
                                     uv_vertical="level")
    w = reconstruct_w(u, u, np.zeros(n), np.zeros(n), DAY, other,
                      uv_vertical="layer")
    assert w.shape == (k, n)


def _reconstruct_cli():
    spec = importlib.util.spec_from_file_location(
        "reconstruct_w_cli_fesom14", REPO / "utils" / "reconstruct_w.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod     # dataclasses look their module up there
    spec.loader.exec_module(mod)
    return mod


def _raw_u(path: Path) -> Path:
    return _write_nc(path, {"time": None, "nodes_3d": 13}, [
        ("time", ("time",), np.array([0.0, 1.0]),
         {"units": "seconds since 2000-01-01 0:0:0"}),
        ("u", ("time", "nodes_3d"), np.zeros((2, 13)), {})])


def test_reconstruct_w_utility_refuses_fesom14(tmp_path, capsys):
    """``utils/reconstruct_w.py`` refuses a FESOM 1.4 ASCII mesh directory
    and u files in a FESOM 1.4 layout (on the levels, or one vector over
    the 3-D nodes) before it loads a mesh or reads a record."""
    cli = _reconstruct_cli()
    hand = _write_hand_mesh(tmp_path / "hand")
    level_u = _data(tmp_path / "data") / "uo.nc"
    raw_u = _raw_u(tmp_path / "u_run_20000101.nc")
    cases = [
        (cli.SourceConfig(meshpath=str(hand), u_path=str(tmp_path / "absent.nc")),
         ["FESOM 1.4", "nod3d.out", "'wo'"]),
        (cli.SourceConfig(meshpath=str(tmp_path / "absent"), u_path=str(level_u)),
         ["FESOM 1.4", "'uo'", "levels", "'wo'"]),
        (cli.SourceConfig(meshpath=str(tmp_path / "absent"), u_path=str(raw_u)),
         ["FESOM 1.4", "'u'", "nodes_3d", "'wo'"]),
    ]
    for source, needles in cases:
        with pytest.raises(ValueError) as info:
            cli.run(cli.ReconstructConfig(source=source))
        msg = str(info.value)
        for needle in needles:
            assert needle in msg, msg
