"""The exact cycle period: the averaging windows on the dataset, ``exact_cycle``.

``time.on_exhausted: cycle`` repeats the forcing.  For means over calendar
months or years the records are not equally spaced, so the period is the sum
of the averaging windows, not the span of the records plus the last interval.
The two places that resolve record times (``apply_time_options`` for every
reader but FESOM 1.4, the FESOM 1.4 reader's own open) attach those windows to
the dataset as the coordinate ``TIME_WINDOWS``; ``exact_cycle`` reads it and
says whether the exact rule applies (windows of unequal length) and with what
period and seam.

Reference numbers: twelve monthly means of one ``noleap`` year, record times
relative to the first centre 0, 29.5, 59, ..., 334 d: period 365 d, seam 31 d
(the legacy rule gives 364.5 d).  All data is synthetic, built in memory or in
``tmp_path``.
"""
import cftime
import numpy as np
import pytest
import xarray as xr

import kelp.readers
from kelp.core.config import DataConfig
from kelp.readers import (TIME_WINDOW_EDGE, TIME_WINDOWS, apply_time_options,
                          exact_cycle, extract_time_info)

DAY = 86400.0
NL, PG = cftime.DatetimeNoLeap, cftime.DatetimeProlepticGregorian
YEAR = (365 * DAY, 31 * DAY)
#: centres of twelve noleap months, seconds since the first centre
FT = np.array([0, 29.5, 59, 89.5, 120, 150.5, 181, 212, 242.5, 273, 303.5,
               334]) * DAY

#: the long fractional axis (S6): records 86400.1 s apart, the last stamp as
#: close to the end of the ``ns`` range as a whole hour allows
N_LONG = 40000
LONG_STEP_NS = 86_400_100_000_000
LONG_LAST = "2262-04-11T06:00:00"


def _ds(times, **extra):
    data = {"u": (("time", "n"), np.zeros((len(times), 2))), **extra}
    return xr.Dataset(data, coords={"time": np.asarray(times)})


def _opened(times, stamp="start", window="calendar_month", **extra):
    cfg = DataConfig(ncinput="data/dir", time_stamp=stamp, time_window=window)
    return apply_time_options(_ds(times, **extra), cfg, "fesom2")


def _months(cls, year, n=12):                      # cftime month starts
    return np.array([cls(year + m // 12, m % 12 + 1, 1) for m in range(n)],
                    dtype=object)


def _months64(year, n=12, unit="ns"):              # datetime64 month starts
    return np.array([f"{year + m // 12}-{m % 12 + 1:02d}-01" for m in range(n)],
                    dtype=f"datetime64[{unit}]")


def _ft(ds):
    return extract_time_info(ds)[0]


def _keeps_us_coords() -> bool:
    """Whether this xarray keeps a ``datetime64[us]`` time coordinate (S5)."""
    return (_ds(_months64(2001, unit="us"))["time"].dtype
            == np.dtype("datetime64[us]"))


def _long_axis():
    """The S6 axis: ``ns`` stamps 86400.1 s apart, the last 2262-04-11 06:00,
    and the float seconds of their window centres computed in ``ns``."""
    last = np.datetime64(LONG_LAST, "ns")
    stamps = last - np.timedelta64(LONG_STEP_NS, "ns") * np.arange(
        N_LONG - 1, -1, -1)
    assert stamps[-1] == last
    c = stamps + np.timedelta64(43_200_050_000_000, "ns")
    f_ns = ((c - c[0]) / np.timedelta64(1, "s")).astype(np.float64)
    return stamps, f_ns


# ---------------------------------------------------------------------------
# the carrier: apply_time_options attaches the windows
# ---------------------------------------------------------------------------

def test_carrier_on_a_cftime_axis():
    ds = _ds(_months(NL, 2001))
    out = _opened(_months(NL, 2001))
    win = out[TIME_WINDOWS]
    assert win.dims == ("time", TIME_WINDOW_EDGE)
    assert win.shape == (12, 2)
    values = win.values
    assert values[0, 0] == NL(2001, 1, 1)
    assert values[11, 1] == NL(2002, 1, 1)
    assert type(values[0, 0]) is type(out["time"].values[0])
    assert TIME_WINDOWS not in ds.coords
    assert np.array_equal(_ft(out), FT)


def test_carrier_on_a_datetime64_axis():
    if not _keeps_us_coords():
        pytest.skip("this xarray stores every datetime64 coordinate as ns")
    ds = _ds(_months64(2001))
    out = _opened(_months64(2001))
    assert out[TIME_WINDOWS].dtype == np.dtype("datetime64[us]")
    assert out[TIME_WINDOWS].dims == ("time", TIME_WINDOW_EDGE)
    from datetime import datetime
    assert out[TIME_WINDOWS].values[1].tolist() == [datetime(2001, 2, 1),
                                                    datetime(2001, 3, 1)]
    assert out["time"].dtype == np.dtype("datetime64[ns]")
    assert TIME_WINDOWS not in ds.coords


def test_close_hook_survives_the_carrier():
    calls = []
    ds = _ds(_months64(2001))
    ds.set_close(lambda: calls.append(1))
    cfg = DataConfig(ncinput="data/dir", time_stamp="start",
                     time_window="calendar_month")
    out = apply_time_options(ds, cfg, "fesom2")
    assert TIME_WINDOWS in out.coords
    out.close()
    assert calls == [1]


def test_no_carrier_without_windows():
    daily = np.array(["2001-01-01", "2001-01-02", "2001-01-03"],
                     dtype="datetime64[ns]")
    out = _opened(daily, stamp="instant", window="1 d")
    assert TIME_WINDOWS not in out.coords
    assert TIME_WINDOW_EDGE not in out.dims
    out = _opened(np.array([NL(2001, 1, d) for d in (1, 2, 3)], dtype=object),
                  stamp="instant", window="1 d")
    assert TIME_WINDOWS not in out.coords
    # no key set: the dataset itself, untouched
    ds = _ds(daily)
    assert apply_time_options(ds, DataConfig(ncinput="data/dir"),
                              "fesom2") is ds


def test_cf_polygon_bounds_named_nv_open():
    out = _opened(_months64(2001), lon_bnds=(("n", "nv"), np.zeros((2, 6))))
    assert out.sizes["nv"] == 6
    assert out.sizes[TIME_WINDOW_EDGE] == 2


# ---------------------------------------------------------------------------
# exact_cycle: the period and the seam
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("axis", [_months(NL, 2001), _months64(2001)],
                         ids=["cftime_noleap", "datetime64"])
def test_exact_cycle_values(axis):
    out = _opened(axis)
    assert exact_cycle(out, _ft(out), 12) == YEAR
    # January to March, as its own axis and as the cut of the year
    three = _opened(axis[:3])
    assert exact_cycle(three, _ft(three), 3) == (90 * DAY, 31 * DAY)
    assert exact_cycle(out, _ft(out), 3) == (90 * DAY, 31 * DAY)


def test_exact_cycle_follows_the_cut():
    out = _opened(_months(NL, 2001))
    ft = _ft(out)
    assert exact_cycle(out, ft, 11) == (334 * DAY, 30.5 * DAY)
    assert exact_cycle(out.isel(time=slice(0, 11)), ft[:11], 11) == (
        334 * DAY, 30.5 * DAY)
    assert exact_cycle(out, ft, 13) is None
    assert exact_cycle(out, ft[:5], 12) is None


def test_exact_cycle_is_none():
    year = _opened(_months(NL, 2001))
    ft = _ft(year)
    assert exact_cycle(None, ft, 12) is None
    assert exact_cycle(object(), ft, 12) is None
    assert exact_cycle(_ds(_months64(2001)), ft, 12) is None   # no coordinate
    # equal windows: the legacy rule, although the coordinate is there
    days = np.array([f"2001-01-{d:02d}" for d in range(1, 11)],
                    dtype="datetime64[ns]")
    daily = _opened(days, window="calendar_day")
    assert TIME_WINDOWS in daily.coords
    assert exact_cycle(daily, _ft(daily), 10) is None
    hours = (np.datetime64("2001-01-01T00:00:00", "ns")
             + np.timedelta64(3_600_100_000_000, "ns") * np.arange(50))
    hourly = _opened(hours, window="3600.1 s")
    assert TIME_WINDOWS in hourly.coords
    assert exact_cycle(hourly, _ft(hourly), 50) is None
    assert exact_cycle(year, ft, 1) is None
    assert exact_cycle(year, None, 12) is None


def _hand_windows(extra_us):
    e0 = np.datetime64("2001-01-01T00:00:00", "us")
    day = np.timedelta64(86_400_000_000, "us")
    edges = [e0, e0 + day, e0 + 2 * day,
             e0 + 3 * day + np.timedelta64(extra_us, "us")]
    win = np.array([[edges[i], edges[i + 1]] for i in range(3)],
                   dtype="datetime64[us]")
    return xr.Dataset(coords={TIME_WINDOWS: (("time", TIME_WINDOW_EDGE),
                                             win)})


def test_window_equality_tolerance():
    ft = np.array([0.0, DAY, 2 * DAY])
    assert exact_cycle(_hand_windows(500), ft, 3) is None
    got = exact_cycle(_hand_windows(2000), ft, 3)
    assert got is not None
    assert got[0] == pytest.approx(3 * DAY + 0.002, abs=1e-9)
    assert got[1] == pytest.approx(DAY + 0.002, abs=1e-9)


def test_exact_cycle_leap_and_multi_year():
    leap = _opened(_months(PG, 2000))
    assert exact_cycle(leap, _ft(leap), 12) == (366 * DAY, 31 * DAY)
    four = _opened(_months64(2001, n=48))
    assert exact_cycle(four, _ft(four), 48) == (1461 * DAY, 31 * DAY)
    years = _opened(np.array([PG(y, 1, 1) for y in range(2001, 2005)],
                             dtype=object), window="calendar_year")
    assert exact_cycle(years, _ft(years), 4) == (1461 * DAY, 365.5 * DAY)


# ---------------------------------------------------------------------------
# the window unit does not follow the unit of time (S5, S6)
# ---------------------------------------------------------------------------

def test_window_unit_is_independent_of_time():
    stamps, f_ns = _long_axis()
    out = _opened(stamps, window="86400.1 s")
    assert out["time"].dtype == np.dtype("datetime64[ns]")
    ft = _ft(out)
    assert np.array_equal(ft, f_ns)
    # sums of rounded float seconds: exact equality is not guaranteed
    assert ft[-1] - ft[0] + (ft[-1] - ft[-2]) == pytest.approx(
        3456004000.0, abs=1e-5)
    if not _keeps_us_coords():
        pytest.skip("this xarray stores every datetime64 coordinate as ns")
    win = out[TIME_WINDOWS]
    assert win.dtype == np.dtype("datetime64[us]")
    assert win.values[-1, 1] == np.datetime64("2262-04-12T06:00:00.100000",
                                              "us")
    assert exact_cycle(out, ft, N_LONG) is None


def test_no_carrier_beyond_ns_without_us_coordinates(monkeypatch):
    monkeypatch.setattr(kelp.readers, "_xarray_keeps_us", lambda: False)
    stamps, f_ns = _long_axis()
    out = _opened(stamps, window="86400.1 s")
    assert TIME_WINDOWS not in out.coords
    assert out["time"].dtype == np.dtype("datetime64[ns]")
    assert np.array_equal(_ft(out), f_ns)
    # edges inside the ns range: the carrier is attached all the same
    out = _opened(_months64(2001))
    assert TIME_WINDOWS in out.coords
    assert exact_cycle(out, _ft(out), 12) == YEAR


# ---------------------------------------------------------------------------
# the FESOM 1.4 reader attaches its own windows
# ---------------------------------------------------------------------------

#: month edges of one noleap year, days since 2001-01-01
_NOLEAP_EDGES = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304,
                          334, 365], dtype=np.float64)


def _monthly_time_vars(n_time, bounds):
    """Twelve monthly means of 2001 (``noleap``): stamps at the centres,
    bounds at the month edges."""
    assert n_time == 12 and bounds
    attrs = {"units": "days since 2001-01-01", "calendar": "noleap",
             "standard_name": "time", "bounds": "time_bnds"}
    lo, hi = _NOLEAP_EDGES[:-1], _NOLEAP_EDGES[1:]
    return ([("time", ("time",), (lo + hi) / 2.0, attrs),
             ("time_bnds", ("time", "bnds"), np.stack([lo, hi], axis=1),
              {})], {"bnds": 2})


def test_fesom14_dataset_carries_its_windows(tmp_path, monkeypatch):
    pytest.importorskip("netCDF4")
    import tests.test_fesom14_tracking as f14
    from kelp.core.config import ModeConfig
    from kelp.readers.fesom14 import Fesom14Reader

    mode = ModeConfig(Fdrift2D=False, Fdrift3D=True)

    # (a) daily means with time bounds: windows of equal length
    ds = Fesom14Reader.open_dataset(str(f14._data(tmp_path / "d")),
                                    mode_cfg=mode)
    try:
        assert ds[TIME_WINDOWS].dims == ("time", TIME_WINDOW_EDGE)
        assert ds[TIME_WINDOWS].shape == (6, 2)
        assert exact_cycle(ds, _ft(ds), 6) is None
    finally:
        ds.close()

    # (b) twelve monthly means of a noleap year, bounds at the month edges
    with monkeypatch.context() as m:
        m.setattr(f14, "_time_vars", _monthly_time_vars)
        data = f14._data(tmp_path / "m", n_time=12)
    ds = Fesom14Reader.open_dataset(str(data), mode_cfg=mode)
    try:
        assert ds[TIME_WINDOWS].shape == (12, 2)
        assert np.allclose(_ft(ds), FT, rtol=0, atol=1e-6)
        assert exact_cycle(ds, FT, 12) == YEAR
    finally:
        ds.close()

    # (c) bounds-free snapshots: no windows, no coordinate
    data = f14._data(tmp_path / "c", bounds=False)
    ds = Fesom14Reader.open_dataset(str(data), mode_cfg=mode,
                                    time_stamp="instant", time_window="1 d")
    try:
        assert TIME_WINDOWS not in ds.coords
        assert TIME_WINDOW_EDGE not in ds.dims
    finally:
        ds.close()
