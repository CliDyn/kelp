"""Stamp options for every reader: the rules and ``apply_time_options``.

``data.time_stamp`` / ``data.time_window`` / ``data.time_offset`` say what
each record's time stamp means.  The FESOM 1.4 reader resolves them while it
opens the files; for every other reader ``apply_time_options`` turns the
opened dataset's ``time`` coordinate into the resolved instants (window
centres for means, the shifted stamps for snapshots).  These tests pin the
rules (one validation, the refusals, the cycle refusal) and the axis
conversion on small in-memory datasets.
"""
from types import SimpleNamespace

import cftime
import numpy as np
import pytest
import xarray as xr

from kelp.core.config import DataConfig
from kelp.readers import (RESOLVED_BY_READER, RESOLVED_FROM_OPTIONS,
                          _check_cycle_window, _check_time_options,
                          _record_time_mode, apply_time_options,
                          time_axis_epoch)

OTHERS = ["fesom2", "fesomc", "healpix", "nemo"]


def _ds(times, attrs=None, encoding=None):
    t = xr.Variable(("time",), np.asarray(times), attrs or {}, encoding or {})
    return xr.Dataset({"u": (("time", "n"), np.zeros((len(times), 2)))},
                      coords={"time": t})


def _d64(strings, unit="ns"):
    return np.array(strings, dtype=f"datetime64[{unit}]")


def _data(**kw):
    kw.setdefault("ncinput", "data/dir")
    return DataConfig(**kw)


def _iso(ds):
    v = ds["time"].values
    if v.dtype == object:
        return [t.isoformat() for t in v]
    return [str(t)[:19] for t in v.astype("datetime64[us]")]


def _cfg(mesh_type, on_exhausted="stop", **data_kw):
    return SimpleNamespace(mesh=SimpleNamespace(type=mesh_type),
                           data=_data(**data_kw),
                           time=SimpleNamespace(on_exhausted=on_exhausted))


def _cf(cls, *fields):
    """Object array of cftime dates of one class from (y, m, d[, h, mi])."""
    out = np.empty(len(fields), dtype=object)
    for i, f in enumerate(fields):
        out[i] = cls(*f)
    return out


# ---------------------------------------------------------------------------
# no key: nothing changes
# ---------------------------------------------------------------------------

def test_no_key_returns_the_same_object():
    ds = _ds(_d64(["2001-01-01", "2001-01-02", "2001-01-03"]))
    assert apply_time_options(ds, _data(), "fesom2") is ds
    # the FESOM 1.4 reader resolves the keys itself: never twice
    assert apply_time_options(
        ds, _data(time_stamp="start", time_window="1 d"), "fesom14") is ds


# ---------------------------------------------------------------------------
# window arithmetic on the dataset axis
# ---------------------------------------------------------------------------

_MONTH_CASES = {
    "d64_2000_leap": (
        _d64(["2000-01-01", "2000-02-01", "2000-03-01"]),
        ["2000-01-16T12:00:00", "2000-02-15T12:00:00",
         "2000-03-16T12:00:00"]),
    "d64_2001": (
        _d64(["2001-01-01", "2001-02-01", "2001-03-01"]),
        ["2001-01-16T12:00:00", "2001-02-15T00:00:00",
         "2001-03-16T12:00:00"]),
    "noleap_2000": (
        _cf(cftime.DatetimeNoLeap, (2000, 1, 1), (2000, 2, 1), (2000, 3, 1)),
        ["2000-01-16T12:00:00", "2000-02-15T00:00:00",
         "2000-03-16T12:00:00"]),
    "noleap_2001": (
        _cf(cftime.DatetimeNoLeap, (2001, 1, 1), (2001, 2, 1), (2001, 3, 1)),
        ["2001-01-16T12:00:00", "2001-02-15T00:00:00",
         "2001-03-16T12:00:00"]),
    "360day_2001": (
        _cf(cftime.Datetime360Day, (2001, 1, 1), (2001, 2, 1), (2001, 3, 1)),
        ["2001-01-16T00:00:00", "2001-02-16T00:00:00",
         "2001-03-16T00:00:00"]),
}


@pytest.mark.parametrize("case", sorted(_MONTH_CASES))
def test_month_centres(case):
    times, expected = _MONTH_CASES[case]
    out = apply_time_options(
        _ds(times), _data(time_stamp="start", time_window="calendar_month"),
        "fesom2")
    assert _iso(out) == expected


def test_offset_is_applied_before_centring():
    stamps = _d64(["2001-01-01T23:00", "2001-01-02T23:00",
                   "2001-01-03T23:00"])
    out = apply_time_options(
        _ds(stamps), _data(time_stamp="start", time_window="calendar_day",
                           time_offset="+1 h"), "fesom2")
    assert _iso(out) == ["2001-01-02T12:00:00", "2001-01-03T12:00:00",
                         "2001-01-04T12:00:00"]
    with pytest.raises(ValueError,
                       match="does not sit on a calendar_day boundary") as e:
        apply_time_options(
            _ds(stamps), _data(time_stamp="start",
                               time_window="calendar_day"), "fesom2")
    assert "data/dir" in str(e.value)


def test_offset_moves_fesom2_stamps_onto_the_day_boundary():
    # FESOM2 daily means stamped one model step before the window end
    stamps = _d64(["1958-01-01T23:15", "1958-01-02T23:15",
                   "1958-01-03T23:15"])
    out = apply_time_options(
        _ds(stamps), _data(time_stamp="end", time_window="calendar_day",
                           time_offset="+45 min"), "fesom2")
    assert _iso(out) == ["1958-01-01T12:00:00", "1958-01-02T12:00:00",
                         "1958-01-03T12:00:00"]
    with pytest.raises(ValueError) as e:
        apply_time_options(
            _ds(stamps), _data(time_stamp="end", time_window="calendar_day"),
            "fesom2")
    assert "data.time_offset" in str(e.value)


_TILE_CASES = {
    "year_end": (
        _d64(["1999-12-31", "2000-01-01", "2000-01-02"]),
        ["1999-12-30T12:00:00", "1999-12-31T12:00:00",
         "2000-01-01T12:00:00"]),
    "leap_day": (
        _d64(["2000-02-28", "2000-02-29", "2000-03-01", "2000-03-02"]),
        ["2000-02-27T12:00:00", "2000-02-28T12:00:00",
         "2000-02-29T12:00:00", "2000-03-01T12:00:00"]),
    "noleap_feb": (
        _cf(cftime.DatetimeNoLeap, (2001, 2, 28), (2001, 3, 1), (2001, 3, 2)),
        ["2001-02-27T12:00:00", "2001-02-28T12:00:00",
         "2001-03-01T12:00:00"]),
}


@pytest.mark.parametrize("case", sorted(_TILE_CASES))
def test_daily_windows_tile_across_year_end_and_leap_day(case):
    times, expected = _TILE_CASES[case]
    out = apply_time_options(
        _ds(times), _data(time_stamp="end", time_window="calendar_day"),
        "fesom2")
    assert _iso(out) == expected


@pytest.mark.parametrize("window", ["calendar_day", "1 d"])
def test_three_conventions_give_one_axis(window):
    days = ["2001-01-01", "2001-01-02", "2001-01-03"]
    start = _d64([f"{d}T00:00" for d in days])
    centre = _d64([f"{d}T12:00" for d in days])
    end = _d64(["2001-01-02T00:00", "2001-01-03T00:00", "2001-01-04T00:00"])
    outs = [apply_time_options(_ds(t), _data(time_stamp=s, time_window=window),
                               "fesom2")["time"].values
            for t, s in ((start, "start"), (centre, "centre"), (end, "end"))]
    assert np.array_equal(outs[0], outs[1])
    assert np.array_equal(outs[0], outs[2])
    assert _iso(xr.Dataset(coords={"time": outs[0]})) == [
        "2001-01-01T12:00:00", "2001-01-02T12:00:00", "2001-01-03T12:00:00"]


# ---------------------------------------------------------------------------
# the axis keeps its form and calendar
# ---------------------------------------------------------------------------

_DAYS = ["2001-01-01", "2001-01-02", "2001-01-03"]
_DAY_OPTS = dict(time_stamp="start", time_window="calendar_day")


@pytest.mark.parametrize("part", ["a", "b", "c", "d", "e"])
def test_axis_forms_keep_their_form_and_calendar(part):
    if part == "a":
        # a datetime64[ns] axis stays at ns
        out = apply_time_options(_ds(_d64(_DAYS)), _data(**_DAY_OPTS),
                                 "fesom2")
        assert out["time"].dtype == np.dtype("datetime64[ns]")
        assert _iso(out) == ["2001-01-01T12:00:00", "2001-01-02T12:00:00",
                             "2001-01-03T12:00:00"]
    elif part == "b":
        # a coarser input unit is refined to us, never kept: a half-window
        # centre of a 3 s window would be truncated to whole seconds
        ds = _ds(_d64(["2001-01-01T00:00:00", "2001-01-01T00:00:03",
                       "2001-01-01T00:00:06"], "s"))
        if ds["time"].dtype != np.dtype("datetime64[s]"):
            pytest.skip("this xarray converts a datetime64[s] coordinate")
        out = apply_time_options(
            ds, _data(time_stamp="start", time_window="3 s"), "fesom2")
        assert out["time"].dtype == np.dtype("datetime64[us]")
        assert out["time"].values[0] == np.datetime64(
            "2001-01-01T00:00:01.500000")
    elif part == "c":
        # a cftime axis stays cftime of the same class; the same instants
        # as the datetime64 axis of the same dates
        cf = _cf(cftime.DatetimeProlepticGregorian,
                 (2001, 1, 1), (2001, 1, 2), (2001, 1, 3))
        out_cf = apply_time_options(_ds(cf), _data(**_DAY_OPTS), "fesom2")
        out_64 = apply_time_options(_ds(_d64(_DAYS)), _data(**_DAY_OPTS),
                                    "fesom2")
        assert out_cf["time"].dtype == object
        assert all(type(t) is cftime.DatetimeProlepticGregorian
                   for t in out_cf["time"].values)
        assert _iso(out_cf) == _iso(out_64)
        nl = _cf(cftime.DatetimeNoLeap, (2001, 1, 1), (2001, 1, 2),
                 (2001, 1, 3))
        out_nl = apply_time_options(_ds(nl), _data(**_DAY_OPTS), "fesom2")
        assert all(type(t) is cftime.DatetimeNoLeap
                   for t in out_nl["time"].values)
    elif part == "d":
        # the source calendar kept by xarray survives: time_axis_epoch
        # reports the same calendar before and after
        ds = _ds(_d64(_DAYS), encoding={"calendar": "gregorian"})
        out = apply_time_options(ds, _data(**_DAY_OPTS), "fesom2")
        assert time_axis_epoch(out)[1] == "gregorian" == time_axis_epoch(ds)[1]
    else:
        ds = _ds(_d64(_DAYS), attrs={"long_name": "time"})
        out = apply_time_options(ds, _data(**_DAY_OPTS), "fesom2")
        assert out["time"].attrs == {"long_name": "time"}


def test_ns_axis_beyond_the_ns_range_becomes_us():
    # every stamp fits ns (to 2262-04-11), the April centre does not
    ds = _ds(_d64(["2262-02-01", "2262-03-01", "2262-04-01"]))
    assert ds["time"].dtype == np.dtype("datetime64[ns]")
    out = apply_time_options(
        ds, _data(time_stamp="start", time_window="calendar_month"), "fesom2")
    assert out["time"].dtype == np.dtype("datetime64[us]")
    assert _iso(out) == ["2262-02-15T00:00:00", "2262-03-16T12:00:00",
                         "2262-04-16T00:00:00"]
    assert not any(s.startswith("1677") for s in _iso(out))
    assert time_axis_epoch(out)[0] == "seconds since 2262-02-15 00:00:00"


def test_input_is_not_mutated():
    ds = _ds(_d64(_DAYS))
    before = ds["time"].values.copy()
    out = apply_time_options(ds, _data(**_DAY_OPTS), "fesom2")
    assert np.array_equal(ds["time"].values, before)
    assert out is not ds
    assert (out["u"].values is ds["u"].values
            or np.shares_memory(out["u"].values, ds["u"].values))


def test_close_hook_is_kept():
    calls = []
    ds = _ds(_d64(_DAYS))
    ds.set_close(lambda: calls.append(1))
    apply_time_options(ds, _data(**_DAY_OPTS), "fesom2").close()
    assert calls == [1]


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------

def _no_coordinate():
    return xr.Dataset({"u": (("time", "n"), np.zeros((3, 2)))})


@pytest.mark.parametrize("make", [
    lambda: _ds(np.array([0.0, 86400.0, 172800.0]),
                attrs={"units": "seconds"}),
    lambda: _ds(np.arange(3)),
    _no_coordinate,
], ids=["float_seconds", "index", "no_coordinate"])
def test_numeric_axis_refused(make):
    with pytest.raises(ValueError) as e:
        apply_time_options(make(), _data(**_DAY_OPTS), "fesom2")
    msg = str(e.value)
    assert "carries no calendar dates" in msg
    assert "data.time_stamp" in msg
    assert "data/dir" in msg


def test_no_time_dimension_refused():
    ds = xr.Dataset({"u": ("n", [0.0])})
    with pytest.raises(ValueError, match="no 'time' dimension"):
        apply_time_options(ds, _data(**_DAY_OPTS), "fesom2")


def test_gap_and_cadence_messages():
    stamps = _d64(["2001-01-01", "2001-01-02", "2001-01-04"])
    with pytest.raises(ValueError) as e:
        apply_time_options(_ds(stamps), _data(**_DAY_OPTS), "fesom2")
    msg = str(e.value)
    assert msg.startswith("data/dir: gap — no record covers "
                          "2001-01-03T00:00:00 to 2001-01-04T00:00:00"), msg
    with pytest.raises(ValueError) as e2:
        apply_time_options(
            _ds(stamps), _data(time_stamp="instant", time_window="1 d"),
            "fesom2")
    msg2 = str(e2.value)
    assert "the snapshot at 2001-01-03T00:00:00 is missing" in msg2
    for m in (msg, msg2):
        assert "variable None" not in m
        assert "files after it" not in m


def test_label_names_the_configured_paths():
    gap = _d64(["2001-01-01", "2001-01-02", "2001-01-04"])
    numeric = np.arange(3)
    # the set paths, each once, in the order u, v, w, eta
    cfg = _data(ncinput="a", ncinput_v="a", ncinput_w="w", ncinput_eta="e",
                **_DAY_OPTS)
    for times in (gap, numeric):
        with pytest.raises(ValueError) as e:
            apply_time_options(_ds(times), cfg, "healpix")
        assert str(e.value).startswith("a, w, e: "), str(e.value)
    # no configured path: the dataset's source, else "the dataset"
    for source, head in (("src.nc", "src.nc: "), (None, "the dataset: ")):
        for times in (gap, numeric):
            ds = _ds(times)
            if source is not None:
                ds.encoding["source"] = source
            with pytest.raises(ValueError) as e:
                apply_time_options(ds, _data(ncinput="", **_DAY_OPTS),
                                   "fesom2")
            assert str(e.value).startswith(head), str(e.value)


@pytest.mark.parametrize("averaging", ["mean", "instant"])
@pytest.mark.parametrize("mesh_type", OTHERS)
def test_time_averaging_with_the_options_refused(mesh_type, averaging):
    cfg = _data(time_averaging=averaging, **_DAY_OPTS)
    for call in (lambda: _check_time_options(cfg, mesh_type),
                 lambda: apply_time_options(_ds(_d64(_DAYS)), cfg,
                                            mesh_type)):
        with pytest.raises(ValueError) as e:
            call()
        msg = str(e.value)
        assert "data.time_averaging" in msg
        assert "Remove data.time_averaging" in msg
        assert "data.time_stamp" in msg


@pytest.mark.parametrize("mesh_type", OTHERS)
def test_offset_alone_refused(mesh_type):
    cfg = _data(time_offset="+6 h")
    for call in (lambda: _check_time_options(cfg, mesh_type),
                 lambda: apply_time_options(_ds(_d64(_DAYS)), cfg,
                                            mesh_type)):
        with pytest.raises(ValueError) as e:
            call()
        msg = str(e.value)
        for part in ("data.time_offset", "data.time_stamp",
                     "data.time_window", repr(mesh_type)):
            assert part in msg, (part, msg)
    # the FESOM 1.4 reader keeps the offset alone
    assert _check_time_options(cfg, "fesom14") is None
    # time_averaging is named first, so removing it is the first step
    with pytest.raises(ValueError) as e:
        _check_time_options(_data(time_offset="+6 h", time_averaging="mean"),
                            mesh_type)
    assert "Remove data.time_averaging" in str(e.value)
    assert "an offset alone" not in str(e.value)


@pytest.mark.parametrize("mesh_type", OTHERS)
def test_cycle_with_a_calendar_month_or_year_window_refused(mesh_type):
    for window in ("calendar_month", "calendar_year"):
        for stamp in ("start", "instant"):
            with pytest.raises(ValueError) as e:
                _check_cycle_window(_cfg(mesh_type, "cycle", time_stamp=stamp,
                                         time_window=window))
            msg = str(e.value)
            assert "time.on_exhausted" in msg
            assert "stop" in msg
            assert window in msg
    # equal windows cycle exactly; no cycle, no key or fesom14: nothing
    for window in ("calendar_day", "1 d"):
        assert _check_cycle_window(_cfg(mesh_type, "cycle", time_stamp="start",
                                        time_window=window)) is None
    assert _check_cycle_window(_cfg(mesh_type, "stop", time_stamp="start",
                                    time_window="calendar_month")) is None
    assert _check_cycle_window(_cfg(mesh_type, "cycle")) is None
    assert _check_cycle_window(_cfg("fesom14", "cycle", time_stamp="start",
                                    time_window="calendar_month")) is None


def test_record_time_mode():
    assert _record_time_mode(_cfg("fesom14")) == RESOLVED_BY_READER
    assert _record_time_mode(
        _cfg("fesom14", time_stamp="start", time_window="1 d")
    ) == RESOLVED_BY_READER
    assert _record_time_mode(
        _cfg("fesom2", time_stamp="start", time_window="1 d")
    ) == RESOLVED_FROM_OPTIONS
    assert _record_time_mode(_cfg("fesom2")) is None
