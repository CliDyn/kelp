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


# ---------------------------------------------------------------------------
# the run's time functions: the keyword ``cycle`` (None or (period, seam))
# ---------------------------------------------------------------------------

D = DAY


class _Stub:
    """A reader whose record ``idx`` holds ``u = eta = idx`` at one node."""

    @staticmethod
    def read_velocities(ds, idx, mesh, mode, diffusion=None):
        from kelp.core.io import VelocityField
        return VelocityField(u=np.full(1, float(idx)), v=np.zeros(1),
                             eta=np.full(1, float(idx)), w=None)


def test_cycle_keyword_is_required():
    import ast
    import inspect

    import kelp.tracking as tr
    with pytest.raises(TypeError):
        tr._resolve_time_bracket(0.0, FT, 12, "stop")
    with pytest.raises(TypeError):
        tr._index_to_sim_time(0, FT, 12, "stop")

    names = {"_resolve_time_bracket", "_index_to_sim_time", "_eta_at_time"}
    calls, constant_none = [], set()

    def visit(node, func):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, child.name)
                continue
            if (isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Name)
                    and child.func.id in names):
                kw = {k.arg: k.value for k in child.keywords}
                calls.append((func, child.func.id, "cycle" in kw))
                value = kw.get("cycle")
                if isinstance(value, ast.Constant) and value.value is None:
                    constant_none.add(func)
            visit(child, func)

    visit(ast.parse(inspect.getsource(tr)), None)
    missing = [c for c in calls if not c[2]]
    assert calls and not missing, missing
    assert {"run_single_realization", "run_tracking", "_eta_at_time",
            "_interp_velocity_at_time",
            "_record_clock_origin"} <= {c[0] for c in calls}
    assert constant_none == {"_record_clock_origin"}


def test_bracket_exact_seam_forward():
    from kelp.tracking import _resolve_recon_stencil, _resolve_time_bracket
    exact = _resolve_time_bracket(340 * D, FT, 12, "cycle", cycle=YEAR)
    assert exact == (11, 0, 334 * D, 365 * D, 340 * D)
    assert _resolve_time_bracket(340 * D, FT, 12, "cycle", cycle=None) == (
        11, 0, 334 * D, 364.5 * D, 340 * D)
    # the seam bracket has no 4-record stencil: PCHIP falls back to linear
    assert _resolve_recon_stencil(exact, 12) is None


def test_bracket_exact_wrap():
    from kelp.tracking import _resolve_time_bracket
    assert _resolve_time_bracket(364.75 * D, FT, 12, "cycle", cycle=YEAR) == (
        11, 0, 334 * D, 365 * D, 364.75 * D)
    assert _resolve_time_bracket(364.75 * D, FT, 12, "cycle", cycle=None) == (
        0, 1, 0.0, 29.5 * D, 0.25 * D)
    assert _resolve_time_bracket(375 * D, FT, 12, "cycle", cycle=YEAR) == (
        0, 1, 0.0, 29.5 * D, 10 * D)
    assert _resolve_time_bracket(375 * D, FT, 12, "cycle", cycle=None) == (
        0, 1, 0.0, 29.5 * D, 10.5 * D)


def test_bracket_exact_backward():
    from kelp.tracking import _resolve_time_bracket
    seam_top = (11, 0, 334 * D, 365 * D, 365 * D)
    for t in (0.0, 365 * D):
        assert _resolve_time_bracket(t, FT, 12, "cycle", direction=-1,
                                     cycle=YEAR) == seam_top
    assert _resolve_time_bracket(0.0, FT, 12, "cycle", direction=-1,
                                 cycle=None) == (
        11, 0, 334 * D, 364.5 * D, 364.5 * D)
    # below the first record (a negative run clock): the seam from its top
    assert _resolve_time_bracket(-1 * D, FT, 12, "cycle", direction=-1,
                                 cycle=YEAR) == (
        11, 0, 334 * D, 365 * D, 364 * D)


def test_stop_ignores_the_cycle_value():
    from kelp.tracking import _index_to_sim_time, _resolve_time_bracket
    assert _resolve_time_bracket(340 * D, FT, 12, "stop", cycle=YEAR) is None
    assert (_resolve_time_bracket(100 * D, FT, 12, "stop", cycle=YEAR)
            == _resolve_time_bracket(100 * D, FT, 12, "stop", cycle=None))
    assert _index_to_sim_time(20, FT, 12, "stop", cycle=YEAR) == 334 * D


def test_bracket_legacy_keeps_trailing_interval_verbatim():
    """``cycle=None`` keeps the trailing interval itself: ``period - span``
    differs from it in the last bit on this axis (264607.35000000003)."""
    from kelp.tracking import _resolve_time_bracket, _shift_to_midpoints
    ft = _shift_to_midpoints(np.arange(74) * 3600.1)
    n = 74
    last = ft[-1] + (ft[-1] - ft[-2])
    assert _resolve_time_bracket(ft[-1] + 0.1, ft, n, "cycle",
                                 cycle=None)[3] == last
    assert _resolve_time_bracket(ft[0], ft, n, "cycle", direction=-1,
                                 cycle=None)[3] == last


def test_index_to_sim_time_exact():
    from kelp.tracking import _index_to_sim_time
    assert _index_to_sim_time(12, FT, 12, "cycle", cycle=YEAR) == 365 * D
    assert _index_to_sim_time(13, FT, 12, "cycle", cycle=YEAR) == 394.5 * D
    assert _index_to_sim_time(-1, FT, 12, "cycle", cycle=YEAR) == -31 * D
    assert _index_to_sim_time(12, FT, 12, "cycle", cycle=None) == 364.5 * D
    # one record: the period is dt, whatever the tuple
    assert _index_to_sim_time(3, FT[:1], 1, "cycle", dt=3600.0,
                              cycle=YEAR) == 10800.0


def test_eta_at_time_follows_the_exact_seam():
    from kelp.tracking import _eta_at_time, _VelCache
    eta = _eta_at_time(349.5 * D, FT, 12, "cycle", 1, _VelCache(), _Stub,
                       None, None, None, None, None, cycle=YEAR)
    assert eta == pytest.approx([5.5], abs=1e-12)   # half way 11 -> 0
    eta = _eta_at_time(349.5 * D, FT, 12, "cycle", 1, _VelCache(), _Stub,
                       None, None, None, None, None, cycle=None)
    assert eta == pytest.approx([11 * 15 / 30.5])


def test_interp_velocity_at_time_reads_the_dataset():
    from kelp.tracking import _interp_velocity_at_time, _VelCache
    ds = _opened(_months(NL, 2001))
    vel = _interp_velocity_at_time(340 * D, FT, 12, "cycle", 1, _VelCache(),
                                   _Stub, ds, None, None)[0]
    assert vel.u[0] == pytest.approx(11 * 25 / 31)
    vel = _interp_velocity_at_time(340 * D, FT, 12, "cycle", 1, _VelCache(),
                                   _Stub, None, None, None)[0]
    assert vel.u[0] == pytest.approx(11 * 24.5 / 30.5)
    assert _interp_velocity_at_time(340 * D, FT, 12, "stop", 1, _VelCache(),
                                    _Stub, ds, None, None) == (None, -1)
