"""FESOM 1.4 reader: ``open_dataset`` (``kelp.readers.fesom14.reader``).

``open_dataset`` finds the files of every variable the run consumes (the
velocities, ``zos`` when present, the sampled scalars), resolves the time axis
of each variable on its own, aligns the variables on one axis and refuses
inconsistent inputs.  Only checks intrinsic to the files happen here; every
check against the mesh comes later (``prepare``).

The files are written by the tests with ``netCDF4`` in two layouts:

- level-wise files ``var(time, depth, ncells)`` with node coordinates and time
  bounds, one file per variable and decade, each with its own epoch, the
  calendar/bounds mix of the published data (``zos``: ``standard`` + ``bnds``;
  the 3-D variables: ``proleptic_gregorian`` + ``number_of_time_bounds``);
- raw-writer files ``<var>_<runid>_<YYYYMMDD>.nc`` with ``var(time,
  nodes_3d)`` (or ``nodes_2d``), ``seconds since <job start> 0:0:0`` stamps at
  the end of each averaging window and an ``output_schedule`` attribute.

Every expected instant is an absolute date written by hand.  Each value
stored in a file encodes its variable, global record, level and node, so the
order of the concatenated records is checked too.  All data is synthetic.
"""
from __future__ import annotations

import re
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")
pytest.importorskip("cftime")

from kelp.core.config import ModeConfig  # noqa: E402
from kelp.readers.fesom14 import Fesom14Reader  # noqa: E402

LEVELS = np.array([0.0, 10.0, 25.0, 50.0])
NCOL = np.array([4, 3, 2, 3, 1])           # wet levels per column
N_NODES = NCOL.size
N3D = int(NCOL.sum())                       # 13 3-D nodes
LON = np.array([-10.0, 0.0, 10.0, 20.0, 170.0])
LAT = np.array([0.0, 5.0, 10.0, 15.0, 20.0])

MONTHLY = "unit: m first: 1 rate: 1"
DAILY = "unit: d first: 1 rate: 1"

#: day offsets of the month ends of a noleap year from its 1 January
NOLEAP_MONTH_ENDS = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]

#: base value of each variable (values encode the variable)
BASE = {"uo": 1.0, "vo": 2.0, "wo": 3.0, "thetao": 4.0, "so": 5.0,
        "zos": 6.0, "u": 1.0, "v": 2.0, "w": 3.0}

MODE_3D = ModeConfig(Fdrift3D=True)
MODE_3D_TS = ModeConfig(Fdrift3D=True, sample_vars=["thetao", "so"])


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _value(var: str, t, k, n):
    """The value stored for ``var`` at global record ``t``, level ``k``,
    node ``n``."""
    return BASE[var] * 1000.0 + t + 0.01 * k + 0.001 * n


def _write_nc(path: Path, dims: dict, variables: list, gattrs=None) -> Path:
    """Write a NetCDF file: ``variables`` = [(name, dims, values, attrs)];
    ``attrs`` may hold ``_FillValue``."""
    with netCDF4.Dataset(str(path), "w") as nc:
        for name, size in dims.items():
            nc.createDimension(name, size)
        for name, vdims, values, attrs in variables:
            attrs = dict(attrs or {})
            values = np.asarray(values)
            fill = attrs.pop("_FillValue", None)
            var = nc.createVariable(name, values.dtype, vdims, fill_value=fill)
            for key, val in attrs.items():
                var.setncattr(key, val)
            var[...] = values
        for key, val in (gattrs or {}).items():
            nc.setncattr(key, val)
    return path


def _month_bounds(epoch_year: int, first_year: int, n_months: int):
    """Month bounds in days since ``epoch_year``-01-01, counted with
    Python's proleptic-Gregorian ``date``, and their mid points."""
    e = date(epoch_year, 1, 1)
    b, y, m = [], first_year, 1
    for _ in range(n_months):
        y2, m2 = (y + 1, 1) if m == 12 else (y, m + 1)
        b.append(((date(y, m, 1) - e).days, (date(y2, m2, 1) - e).days))
        y, m = y2, m2
    b = np.asarray(b, dtype=np.float64)
    return b.mean(axis=1), b


def _levelwise(path: Path, var: str, *, stamps, units, t0=0, bounds=None,
               calendar="proleptic_gregorian", bnds_dim="number_of_time_bounds",
               two_d=False, lon=LON, lat=LAT, levels=LEVELS, hdim="ncells",
               zdim="depth", schedule=None) -> Path:
    """A level-wise file of ``var`` with records ``t0, t0 + 1, ...``: NaN
    (decoded ``_FillValue``) below each column's bed."""
    stamps = np.asarray(stamps, dtype=np.float64)
    n_t = stamps.size
    t = (t0 + np.arange(n_t))[:, None, None]
    k = np.arange(len(levels))[None, :, None]
    n = np.arange(N_NODES)[None, None, :]
    wet = k < NCOL[None, None, :]
    time_attrs = {"units": units, "calendar": calendar, "standard_name": "time"}
    dims = {"time": None, hdim: N_NODES}
    variables = []
    if bounds is not None:
        time_attrs["bounds"] = "time_bnds"
        dims[bnds_dim] = 2
        variables.append(("time_bnds", ("time", bnds_dim),
                          np.asarray(bounds, dtype=np.float64), {}))
    variables.insert(0, ("time", ("time",), stamps, time_attrs))
    if two_d:
        vals = _value(var, t[:, 0, :], 0, n[:, 0, :]).astype(np.float32)
        variables.append((var, ("time", hdim), vals, {"_FillValue": np.float32(1e20)}))
    else:
        dims[zdim] = len(levels)
        variables.append((zdim, (zdim,), np.asarray(levels, dtype=np.float64),
                          {"units": "m", "positive": "down"}))
        vals = np.where(wet, _value(var, t, k, n), np.nan).astype(np.float32)
        attrs = {"_FillValue": np.float32(1e20)}
        if var == "wo":
            attrs["standard_name"] = "upward_sea_water_velocity"
        variables.append((var, ("time", zdim, hdim), vals, attrs))
    if lon is not None:
        variables.append(("lon", (hdim,), np.asarray(lon, dtype=np.float64),
                          {"standard_name": "longitude"}))
        variables.append(("lat", (hdim,), np.asarray(lat, dtype=np.float64),
                          {"standard_name": "latitude"}))
    gattrs = {"output_schedule": schedule} if schedule else None
    return _write_nc(path, dims, variables, gattrs)


def _decade(d: Path, var: str, decade: int, **kw) -> Path:
    """One decade (0: 1951-1960, 1: 1961-1970) of monthly means of ``var``
    as published: its own epoch, mid-month stamps, time bounds."""
    y0 = 1951 + 10 * decade
    stamps, bounds = _month_bounds(y0, y0, 120)
    if var == "zos":
        kw.setdefault("calendar", "standard")
        kw.setdefault("bnds_dim", "bnds")
        kw.setdefault("two_d", True)
    name = f"{var}_Omon_model_{y0}01-{y0 + 9}12.nc"
    return _levelwise(d / name, var, stamps=stamps, bounds=bounds,
                      units=f"days since {y0}-1-1", t0=120 * decade, **kw)


def _raw(d: Path, var: str, epoch: str, days, *, runid="run",
         schedule=MONTHLY, calendar="noleap", t0=0, two_d=False) -> Path:
    """A raw-writer file ``<var>_<runid>_<YYYYMMDD>.nc``: stamps at ``days``
    after the job start ``epoch`` (``YYYY-MM-DD``), in seconds."""
    stamps = np.asarray(days, dtype=np.float64) * 86400.0
    t = (t0 + np.arange(stamps.size))[:, None]
    dim, size = ("nodes_2d", N_NODES) if two_d else ("nodes_3d", N3D)
    vals = _value(var, t, 0, np.arange(size)[None, :]).astype(np.float32)
    name = f"{var}_{runid}_{epoch.replace('-', '')}.nc"
    return _write_nc(
        d / name, {"time": None, dim: size},
        [("time", ("time",), stamps,
          {"long_name": "time", "units": f"seconds since {epoch} 0:0:0",
           "calendar": calendar}),
         (var, ("time", dim), vals,
          {"units": "m/s", "_FillValue": np.float32(1e30)})],
        {"output_schedule": schedule})


def _raw_months(d: Path, var: str, n_months=3, **kw) -> Path:
    """``n_months`` monthly means of a noleap job started 1948-01-01."""
    return _raw(d, var, "1948-01-01", NOLEAP_MONTH_ENDS[:n_months], **kw)


def _raw_days(d: Path, var: str, n_days=90, **kw) -> Path:
    """``n_days`` daily means of a noleap job started 1948-01-01."""
    return _raw(d, var, "1948-01-01", range(1, n_days + 1),
                schedule=DAILY, **kw)


def _open(path, mode=None, **kw):
    return Fesom14Reader.open_dataset(str(path), mode_cfg=mode, **kw)


def _iso(ds):
    return [t.isoformat() for t in ds["time"].values]


def _refusal(path, mode=None, **kw) -> str:
    with pytest.raises(ValueError) as info:
        _open(path, mode, **kw)
    return str(info.value)


def _assert_names(msg: str, *needles: str) -> None:
    for needle in needles:
        assert needle in msg, (needle, msg)


# ---------------------------------------------------------------------------
# Level-wise files, raw segments
# ---------------------------------------------------------------------------

def test_levelwise_decadal_files(tmp_path):
    """Two decades per variable, each file with its own epoch; the 3-D
    variables in ``proleptic_gregorian`` with bounds on
    ``number_of_time_bounds``, ``zos`` in ``standard`` with ``bnds``."""
    for var in ("uo", "vo", "wo", "thetao", "so", "zos"):
        for decade in (1, 0):
            _decade(tmp_path, var, decade)
    ds = _open(tmp_path, MODE_3D_TS)

    assert set(ds.data_vars) == {"uo", "vo", "wo", "thetao", "so", "zos"}
    iso = _iso(ds)
    assert len(iso) == 240
    assert iso[0] == "1951-01-16T12:00:00"
    assert iso[1] == "1951-02-15T00:00:00"          # 28 days
    assert iso[13] == "1952-02-15T12:00:00"         # leap February
    assert iso[119] == "1960-12-16T12:00:00"
    assert iso[120] == "1961-01-16T12:00:00"        # the second file's epoch
    assert iso[-1] == "1970-12-16T12:00:00"
    assert ds["time"].values[0].calendar == "proleptic_gregorian"
    assert ds.attrs["fesom14_form"] == "levelwise"
    assert int(ds.attrs["fesom14_has_eta"]) == 1
    assert ds.attrs["fesom14_time_source"] == "bounds"

    # record order: the global record index t is stored in every value
    t = np.arange(240)
    np.testing.assert_allclose(ds["uo"].values[:, 0, 0], _value("uo", t, 0, 0),
                               rtol=0, atol=1e-3)
    np.testing.assert_allclose(ds["zos"].values[:, 2], _value("zos", t, 0, 2),
                               rtol=0, atol=1e-3)
    assert ds["uo"].dims == ("time", "depth", "ncells")
    assert ds["zos"].dims == ("time", "ncells")
    np.testing.assert_array_equal(ds["depth"].values, LEVELS)
    np.testing.assert_array_equal(ds["lon"].values, LON)
    np.testing.assert_array_equal(ds["lat"].values, LAT)
    # below the bed: missing (NaN after decoding), not filled here
    assert np.isnan(ds["uo"].values[5, 3, 1])
    assert ds["wo"].attrs.get("standard_name") == "upward_sea_water_velocity"
    ds.close()

    # the axis is the u axis, in the calendar of the u files, also when
    # the variable opened last (zos) is in another calendar name
    ds = _open(tmp_path, MODE_3D)
    assert set(ds.data_vars) == {"uo", "vo", "wo", "zos"}
    assert ds["time"].values[0].calendar == "proleptic_gregorian"
    assert _iso(ds)[120] == "1961-01-16T12:00:00"
    ds.close()


def test_raw_segments_nodes3d(tmp_path):
    """Raw files over two job segments: stamps at month ends in seconds
    since each job's start, noleap, ``output_schedule`` monthly.  The run
    ids are chosen so that the file names sort against time order."""
    for var in ("uo", "vo", "wo"):
        _raw(tmp_path, var, "1948-01-01", NOLEAP_MONTH_ENDS, runid="runB")
        _raw(tmp_path, var, "1949-01-01", NOLEAP_MONTH_ENDS, runid="runA",
             t0=12)
    _raw(tmp_path, "zos", "1948-01-01", NOLEAP_MONTH_ENDS, runid="runB",
         two_d=True)
    _raw(tmp_path, "zos", "1949-01-01", NOLEAP_MONTH_ENDS, runid="runA",
         two_d=True, t0=12)
    ds = _open(tmp_path, MODE_3D)

    iso = _iso(ds)
    assert len(iso) == 24
    assert iso[:4] == ["1948-01-16T12:00:00", "1948-02-15T00:00:00",
                       "1948-03-16T12:00:00", "1948-04-16T00:00:00"]
    assert iso[11] == "1948-12-16T12:00:00"
    assert iso[12] == "1949-01-16T12:00:00"
    assert iso[-1] == "1949-12-16T12:00:00"
    assert ds["time"].values[0].calendar == "noleap"
    assert ds.attrs["fesom14_form"] == "nodes3d"
    assert ds.attrs["fesom14_time_source"] == "schedule"
    assert int(ds.attrs["fesom14_has_eta"]) == 1
    assert ds["uo"].dims == ("time", "nodes_3d")
    assert ds["uo"].shape == (24, N3D)
    assert ds["zos"].dims == ("time", "nodes_2d")
    t = np.arange(24)
    np.testing.assert_allclose(ds["wo"].values[:, 7], _value("wo", t, 0, 7),
                               rtol=0, atol=1e-3)
    np.testing.assert_allclose(ds["zos"].values[:, 4], _value("zos", t, 0, 4),
                               rtol=0, atol=1e-3)
    ds.close()


def test_raw_segment_without_records(tmp_path):
    """A job segment that ended before its first output event leaves files
    without records; they add none (the next segment restarts at the same
    date under another run id)."""
    for var in ("uo", "vo", "wo"):
        _raw(tmp_path, var, "1948-01-01", NOLEAP_MONTH_ENDS[:2], runid="a")
        _raw(tmp_path, var, "1948-03-01", [], runid="b")
        _raw(tmp_path, var, "1948-03-01", [31], runid="c", t0=2)
    ds = _open(tmp_path, MODE_3D)
    assert _iso(ds) == ["1948-01-16T12:00:00", "1948-02-15T00:00:00",
                        "1948-03-16T12:00:00"]
    np.testing.assert_allclose(ds["uo"].values[:, 0], _value("uo", np.arange(3), 0, 0),
                               rtol=0, atol=1e-3)
    ds.close()


def test_time_axis_for_tracking(tmp_path):
    """The resolved axis is what tracking reads: seconds between the
    instants in the files' calendar, and the epoch of the first instant."""
    from kelp.readers import extract_time_info, time_axis_epoch
    for var in ("uo", "vo", "wo"):
        _raw_months(tmp_path, var, n_months=4)
    ds = _open(tmp_path, MODE_3D)
    secs, file_dt = extract_time_info(ds)
    # window centres of Jan..Apr of a noleap year: 29.5, 29.5, then 30.5 d
    np.testing.assert_array_equal(secs, np.array([0.0, 29.5, 59.0, 89.5]) * 86400.0)
    assert file_dt is None                       # irregular: monthly
    assert time_axis_epoch(ds) == ("seconds since 1948-01-16 12:00:00", "noleap")
    ds.close()


def test_lazy_records(tmp_path):
    """The data stay on disk until read, one record per chunk; ``close``
    closes every file."""
    for var in ("uo", "vo", "wo"):
        for decade in (0, 1):
            _decade(tmp_path, var, decade)
    ds = _open(tmp_path, MODE_3D)
    dask_array = pytest.importorskip("dask.array")
    assert isinstance(ds["uo"].data, dask_array.Array)
    assert set(ds["uo"].chunks[0]) == {1}
    window = ds.isel(time=slice(118, 122)).load()
    np.testing.assert_allclose(window["vo"].values[:, 0, 0],
                               _value("vo", np.arange(118, 122), 0, 0),
                               rtol=0, atol=1e-3)
    ds.close()


# ---------------------------------------------------------------------------
# Alignment of the consumed variables
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kept", [0, 1])
def test_variable_missing_a_decade_refused(tmp_path, kept):
    """``wo`` has only one of the two decades: refused, naming ``wo`` and
    the missing interval (the later or the earlier decade)."""
    for var in ("uo", "vo"):
        for decade in (0, 1):
            _decade(tmp_path, var, decade)
    _decade(tmp_path, "wo", kept)
    msg = _refusal(tmp_path, MODE_3D)
    missing = (("1961-01-01T00:00:00", "1971-01-01T00:00:00") if kept == 0
               else ("1951-01-01T00:00:00", "1961-01-01T00:00:00"))
    _assert_names(msg, "'wo'", "'uo'", *missing, "120 records")
    assert re.search(r"'wo' has no record", msg), msg


def test_different_schedules_refused(tmp_path):
    """Raw velocities on different output schedules: refused, naming both
    variables and both schedules."""
    _raw_months(tmp_path, "uo")
    _raw_months(tmp_path, "vo")
    _raw_days(tmp_path, "wo")
    msg = _refusal(tmp_path, MODE_3D)
    _assert_names(msg, "'uo'", "'wo'", MONTHLY, DAILY, "one schedule")


def test_schedules_compared_by_events(tmp_path):
    """Output schedules are compared by the records they give, not by their
    text: ``first: 0`` and ``first: 1`` give the same monthly events, as do
    ``first: 5`` and ``first: 12`` with ``rate: 12`` (one event a year at the
    end of each 12th month of the job); ``first: 5 rate: 1`` gives a first
    window of five months and is refused, naming both schedules."""
    same = tmp_path / "first_0_1"
    same.mkdir()
    _raw_months(same, "uo")
    _raw_months(same, "vo")
    _raw_months(same, "wo", schedule="unit: m first: 0 rate: 1")
    ds = _open(same, MODE_3D)
    assert _iso(ds) == ["1948-01-16T12:00:00", "1948-02-15T00:00:00",
                        "1948-03-16T12:00:00"]
    ds.close()

    yearly = tmp_path / "first_5_12"
    yearly.mkdir()
    for var, first in (("uo", 12), ("vo", 12), ("wo", 5)):
        _raw(yearly, var, "1948-01-01", [365, 730],
             schedule=f"unit: m first: {first} rate: 12")
    ds = _open(yearly, MODE_3D)
    # noleap: the centre of each 365-day year is 182.5 days after 1 January
    assert _iso(ds) == ["1948-07-02T12:00:00", "1949-07-02T12:00:00"]
    ds.close()

    other = tmp_path / "first_5_rate_1"
    other.mkdir()
    _raw(other, "uo", "1948-01-01", NOLEAP_MONTH_ENDS[:7])
    _raw(other, "vo", "1948-01-01", NOLEAP_MONTH_ENDS[:7])
    _raw(other, "wo", "1948-01-01", [151, 181, 212],
         schedule="unit: m first: 5 rate: 1")
    msg = _refusal(other, MODE_3D)
    _assert_names(msg, "'uo'", "'wo'", "different output schedules", MONTHLY,
                  "unit: m first: 5 rate: 1", "1948-01-16T12:00:00",
                  "1948-06-01T00:00:00")


def test_daily_zos_with_monthly_velocities_refused(tmp_path):
    for var in ("uo", "vo", "wo"):
        _raw_months(tmp_path, var)
    _raw_days(tmp_path, "zos", two_d=True)
    msg = _refusal(tmp_path, MODE_3D)
    _assert_names(msg, "'zos'", MONTHLY, DAILY, "data.ncinput")
    assert re.search(r"leave the 'zos' files out of data\.ncinput", msg), msg


def test_requested_scalar_with_other_schedule_refused(tmp_path):
    for var in ("uo", "vo", "wo"):
        _raw_months(tmp_path, var)
    _raw_days(tmp_path, "thetao")
    msg = _refusal(tmp_path, ModeConfig(Fdrift3D=True, sample_vars=["thetao"]))
    _assert_names(msg, "'thetao'", "'uo'", MONTHLY, DAILY, "mode.sample_vars")


def test_unrequested_scalar_ignored(tmp_path):
    """The same daily ``thetao`` without ``sample_vars`` is not consumed: it
    is neither resolved nor aligned, and not in the dataset."""
    for var in ("uo", "vo", "wo"):
        _raw_months(tmp_path, var)
    _raw_days(tmp_path, "thetao")
    ds = _open(tmp_path, MODE_3D)
    assert "thetao" not in ds.variables
    assert set(ds.data_vars) == {"uo", "vo", "wo"}
    assert len(ds["time"]) == 3
    ds.close()


def test_equal_centres_unequal_windows_refused(tmp_path):
    """``uo`` and ``vo`` share every centre (days 1, 3, 5) but not the
    windows: refused naming both windows of the first record."""
    units = "days since 1951-1-1"
    _levelwise(tmp_path / "uo.nc", "uo", stamps=[1, 3, 5], units=units,
               bounds=[[0, 2], [2, 4], [4, 6]])
    _levelwise(tmp_path / "vo.nc", "vo", stamps=[1, 3, 5], units=units,
               bounds=[[0.5, 1.5], [1.5, 4.5], [4.5, 5.5]])
    _levelwise(tmp_path / "wo.nc", "wo", stamps=[1, 3, 5], units=units,
               bounds=[[0, 2], [2, 4], [4, 6]])
    msg = _refusal(tmp_path, MODE_3D)
    _assert_names(msg, "'uo'", "'vo'", "1951-01-02T00:00:00",
                  "1951-01-01T00:00:00", "1951-01-03T00:00:00",
                  "1951-01-01T12:00:00", "1951-01-02T12:00:00")


def test_calendars_that_differ_refused(tmp_path):
    """``noleap`` velocities with a ``standard`` ``vo``: the same dates are
    different instants; the message names both calendars."""
    _raw_months(tmp_path, "uo", n_months=1)    # January: same in both
    _raw_months(tmp_path, "vo", n_months=1, calendar="standard")
    _raw_months(tmp_path, "wo", n_months=1)
    msg = _refusal(tmp_path, MODE_3D)
    _assert_names(msg, "'uo'", "'vo'", "noleap", "standard",
                  "1948-01-16T12:00:00")


def test_node_order_across_variables(tmp_path):
    """Node coordinates of two variables that differ only by the longitude
    convention (0..360 against -180..180) or by 1e-9 deg rounding are the
    same nodes in the same order; two swapped nodes, or a node moved by
    1e-5 deg, are refused naming the variable and the node."""
    units = "days since 1951-1-1"
    stamps, bounds = _month_bounds(1951, 1951, 3)

    def write(d, var, lon, lat):
        d.mkdir(exist_ok=True)
        _levelwise(d / f"{var}.nc", var, stamps=stamps, bounds=bounds,
                   units=units, lon=lon, lat=lat)

    ok = tmp_path / "ok"
    write(ok, "uo", LON, LAT)
    write(ok, "vo", np.mod(LON, 360.0), LAT + 1e-9)
    write(ok, "wo", LON - 1e-9, LAT)
    ds = _open(ok, MODE_3D)
    np.testing.assert_array_equal(ds["lon"].values, LON)
    ds.close()

    swapped = LON.copy(), LAT.copy()
    for a in swapped:
        a[[1, 2]] = a[[2, 1]]
    moved_lat = LAT.copy()
    moved_lat[3] += 1e-5
    for case, (lon, lat), node in (("swapped", swapped, 1),
                                   ("moved", (LON, moved_lat), 3)):
        d = tmp_path / case
        write(d, "uo", LON, LAT)
        write(d, "vo", LON, LAT)
        write(d, "wo", lon, lat)
        msg = _refusal(d, MODE_3D)
        _assert_names(msg, "'wo'", "'uo'", "wo.nc", "uo.nc")
        assert re.search(rf"\bnode {node}\b", msg), msg


def test_level_axis_differs_between_variables_refused(tmp_path):
    units = "days since 1951-1-1"
    stamps, bounds = _month_bounds(1951, 1951, 3)
    for var in ("uo", "vo", "wo"):
        levels = LEVELS.copy()
        if var == "vo":
            levels[2] = 30.0
        _levelwise(tmp_path / f"{var}.nc", var, stamps=stamps, bounds=bounds,
                   units=units, levels=levels)
    msg = _refusal(tmp_path, MODE_3D)
    _assert_names(msg, "'vo'", "'uo'", "level 2", "30", "25")


def test_mixed_forms_refused(tmp_path):
    """Level-wise ``uo`` with a raw ``vo`` over the 3-D nodes: refused."""
    stamps, bounds = _month_bounds(1951, 1951, 3)
    _levelwise(tmp_path / "uo.nc", "uo", stamps=stamps, bounds=bounds,
               units="days since 1951-1-1")
    _raw_months(tmp_path, "vo")
    _raw_months(tmp_path, "wo")
    msg = _refusal(tmp_path, MODE_3D)
    _assert_names(msg, "'uo'", "'vo'", "nodes_3d", "depth")


# ---------------------------------------------------------------------------
# Which variables are consumed
# ---------------------------------------------------------------------------

def test_missing_velocity_refused(tmp_path):
    _raw_months(tmp_path, "uo")
    _raw_months(tmp_path, "vo")
    msg = _refusal(tmp_path, MODE_3D)
    _assert_names(msg, "'wo'", "mode.var_w", "uo, vo")
    (tmp_path / "vo_run_19480101.nc").unlink()
    msg = _refusal(tmp_path, ModeConfig())
    _assert_names(msg, "'vo'", "mode.var_v")


def test_2d_tracking_does_not_consume_w(tmp_path):
    """2-D tracking reads no ``w``: a daily ``wo`` among monthly ``uo/vo``
    is ignored there, and no ``wo`` at all is fine."""
    _raw_months(tmp_path, "uo")
    _raw_months(tmp_path, "vo")
    _raw_days(tmp_path, "wo")
    ds = _open(tmp_path, ModeConfig(Fdrift3D=False))
    assert set(ds.data_vars) == {"uo", "vo"}
    ds.close()
    # without a mode config: w is consumed when present (and aligned)
    msg = _refusal(tmp_path)
    _assert_names(msg, "'wo'", DAILY)


def test_variable_names_from_mode(tmp_path):
    """Overridden names select the variables; a raw-writer file per variable
    named ``u``/``v``/``w`` is ordinary output."""
    for var in ("u", "v", "w"):
        _raw_months(tmp_path, var)
    ds = _open(tmp_path, ModeConfig(Fdrift3D=True, var_u="u", var_v="v",
                                    var_w="w"))
    assert set(ds.data_vars) == {"u", "v", "w"}
    ds.close()


def test_glob_and_separate_paths(tmp_path):
    """``path`` may be a glob; ``v_path``/``w_path`` add their files."""
    for var in ("uo", "vo", "wo"):
        (tmp_path / var).mkdir()
        _raw_months(tmp_path / var, var)
    ds = Fesom14Reader.open_dataset(str(tmp_path / "uo" / "uo_*.nc"),
                                    v_path=str(tmp_path / "vo"),
                                    w_path=str(tmp_path / "wo" / "wo_run_19480101.nc"),
                                    mode_cfg=MODE_3D)
    assert set(ds.data_vars) == {"uo", "vo", "wo"}
    ds.close()
    with pytest.raises(FileNotFoundError, match="nothing_.*nc"):
        _open(tmp_path / "nothing_*.nc", MODE_3D)
    with pytest.raises(FileNotFoundError, match="absent"):
        _open(tmp_path / "absent", MODE_3D)


def test_has_eta_attribute(tmp_path):
    for var in ("uo", "vo", "wo"):
        _raw_months(tmp_path, var)
    ds = _open(tmp_path, MODE_3D)
    assert int(ds.attrs["fesom14_has_eta"]) == 0
    assert "zos" not in ds.variables
    ds.close()
    _raw_months(tmp_path, "zos", two_d=True)
    ds = _open(tmp_path, MODE_3D)
    assert int(ds.attrs["fesom14_has_eta"]) == 1
    assert ds["zos"].dims == ("time", "nodes_2d")
    ds.close()


def test_sampled_zos(tmp_path):
    """``zos`` in ``mode.sample_vars``: required, named by that key when
    missing; when present it is also the sea-surface height."""
    for var in ("uo", "vo", "wo"):
        _raw_months(tmp_path, var)
    mode = ModeConfig(Fdrift3D=True, sample_vars=["zos"])
    msg = _refusal(tmp_path, mode)
    _assert_names(msg, "'zos'", "mode.sample_vars")
    assert "mode.var_eta" not in msg, msg
    _raw_months(tmp_path, "zos", two_d=True)
    ds = _open(tmp_path, mode)
    assert "zos" in ds.data_vars
    assert int(ds.attrs["fesom14_has_eta"]) == 1
    ds.close()


def test_missing_variable_not_blamed_on_a_restart(tmp_path):
    """A missing ``wo`` is reported as missing; a restart file that does not
    hold it is only noted as not read."""
    _raw_months(tmp_path, "uo")
    _raw_months(tmp_path, "vo")
    _restart(tmp_path / "run.1948.oce.nc")
    msg = _refusal(tmp_path, MODE_3D)
    assert msg.startswith("fesom14: no file under "), msg
    _assert_names(msg, "holds the variable 'wo'", "mode.var_w", "Not read:",
                  "run.1948.oce.nc", "restart")
    assert msg.index("holds the variable 'wo'") < msg.index("restart"), msg
    assert "no other file" not in msg, msg


def test_explicit_var_eta_missing_refused(tmp_path):
    """``zos`` is optional, but a ``mode.var_eta`` set by the user must be
    found."""
    for var in ("uo", "vo", "wo"):
        _raw_months(tmp_path, var)
    msg = _refusal(tmp_path, ModeConfig(Fdrift3D=True, var_eta="ssh"))
    _assert_names(msg, "'ssh'", "mode.var_eta")


# ---------------------------------------------------------------------------
# Files KELP does not read
# ---------------------------------------------------------------------------

def _restart(path: Path, *, wpot=True) -> Path:
    """A restart-style file: one snapshot over the 3-D nodes with ``iter``
    (and ``wpot``)."""
    names = ["u", "v", "w"] + (["wpot"] if wpot else []) + ["temp", "salt"]
    variables = [("time", ("time",), np.array([3.1536e7]),
                  {"units": "seconds since 1948-01-01 0:0:0"}),
                 ("iter", ("time",), np.array([35040], dtype=np.int32), {}),
                 ("ssh", ("time", "nodes_2d"), np.zeros((1, N_NODES)), {})]
    variables += [(n, ("time", "nodes_3d"), np.zeros((1, N3D)), {}) for n in names]
    return _write_nc(path, {"time": None, "nodes_2d": N_NODES, "nodes_3d": N3D},
                     variables)


def _older_branch(path: Path, *, record_dim: str = "time") -> Path:
    """An older-branch mean file: u, v, w, temp, salt and ssh together over
    the nodes, no ``output_schedule``."""
    variables = [(record_dim, (record_dim,), np.array([2.6784e6, 5.0976e6]),
                  {"units": "seconds since 1948-01-01 0:0:0"}),
                 ("ssh", (record_dim, "nodes_2d"), np.zeros((2, N_NODES)), {})]
    variables += [(n, (record_dim, "nodes_3d"), np.zeros((2, N3D), dtype=np.float32),
                   {}) for n in ("u", "v", "w", "temp", "salt")]
    return _write_nc(path, {record_dim: None, "nodes_2d": N_NODES, "nodes_3d": N3D},
                     variables)


UVW = ModeConfig(Fdrift3D=True, var_u="u", var_v="v", var_w="w")


@pytest.mark.parametrize("wpot", [True, False])
def test_refuses_restart_file(tmp_path, wpot):
    f = _restart(tmp_path / "run.1948.oce.nc", wpot=wpot)
    for mode in (MODE_3D, UVW):
        msg = _refusal(f, mode)
        _assert_names(msg, "run.1948.oce.nc", "restart", "rotated", "snapshot")
    # a directory: the restart next to the mean output is not read with the
    # default names ...
    for var in ("uo", "vo", "wo"):
        _raw_months(tmp_path, var)
    ds = _open(tmp_path, MODE_3D)
    assert set(ds.data_vars) == {"uo", "vo", "wo"}
    ds.close()
    # ... and refused when the names select its variables
    msg = _refusal(tmp_path, UVW)
    _assert_names(msg, "run.1948.oce.nc", "restart")


@pytest.mark.parametrize("record_dim", ["time", "T"])
def test_refuses_older_branch_oce_mean(tmp_path, record_dim):
    """Recognised by its content (several fields over the 3-D nodes in one
    file, no ``output_schedule``), whatever the file is called or how its
    record dimension is named; also when the names select its variables."""
    f = _older_branch(tmp_path / "run.1948.oce.mean.nc", record_dim=record_dim)
    renamed = _older_branch(tmp_path / "old_means.nc", record_dim=record_dim)
    for path in (f, renamed):
        for mode in (MODE_3D, UVW, ModeConfig(var_u="u")):
            msg = _refusal(path, mode)
            _assert_names(msg, path.name, "older branch", "oce.mean.nc",
                          "frame", "output_schedule")
            assert "restart" not in msg, msg
        path.unlink()


# ---------------------------------------------------------------------------
# Time options
# ---------------------------------------------------------------------------

def test_options_reach_resolution(tmp_path):
    """Files without bounds or schedule (stamps at month ends, as FESOM2
    writes them): refused without options; resolved exactly with
    ``end`` + ``calendar_month``; the offset moves every stamp; options
    override bounds (with a warning)."""
    plain = tmp_path / "plain"
    plain.mkdir()
    for var in ("uo", "vo", "wo"):
        _levelwise(plain / f"{var}.nc", var, stamps=[31, 59, 90],
                   units="days since 1951-1-1")
    msg = _refusal(plain, MODE_3D)
    _assert_names(msg, "data.time_stamp", "data.time_window", "'uo'")

    ds = _open(plain, MODE_3D, time_stamp="end", time_window="calendar_month")
    assert _iso(ds) == ["1951-01-16T12:00:00", "1951-02-15T00:00:00",
                        "1951-03-16T12:00:00"]
    assert ds.attrs["fesom14_time_source"] == "options"
    ds.close()

    # snapshots one day apart, every stamp 6 h early
    daily = tmp_path / "daily"
    daily.mkdir()
    for var in ("uo", "vo", "wo"):
        _levelwise(daily / f"{var}.nc", var, stamps=[0.75, 1.75, 2.75],
                   units="days since 1951-1-1")
    ds = _open(daily, MODE_3D, time_stamp="instant", time_window="1 d",
               time_offset="+6 h")
    assert _iso(ds) == ["1951-01-02T00:00:00", "1951-01-03T00:00:00",
                        "1951-01-04T00:00:00"]
    ds.close()

    # options override time bounds, with a warning naming them
    bounded = tmp_path / "bounded"
    bounded.mkdir()
    for var in ("uo", "vo", "wo"):
        _decade(bounded, var, 0)
    with pytest.warns(UserWarning, match="override the time bounds"):
        ds = _open(bounded, MODE_3D, time_stamp="centre",
                   time_window="calendar_month")
    assert _iso(ds)[:2] == ["1951-01-16T12:00:00", "1951-02-15T00:00:00"]
    assert ds.attrs["fesom14_time_source"] == "options"
    ds.close()

    # an incomplete or invalid set is refused before any file is read
    for kw, needle in (({"time_stamp": "end"}, "data.time_window"),
                       ({"time_window": "calendar_month"}, "data.time_stamp"),
                       ({"time_stamp": "end", "time_window": "monthly"},
                        "data.time_window"),
                       ({"time_offset": "6 hours"}, "data.time_offset")):
        with pytest.raises(ValueError, match=needle):
            _open(tmp_path / "does_not_exist", MODE_3D, **kw)


def test_no_unexpected_warning(tmp_path):
    for var in ("uo", "vo", "wo"):
        _raw_months(tmp_path, var)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ds = _open(tmp_path, MODE_3D)
    ds.close()
