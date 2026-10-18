"""FESOM 1.4 time axis: decoding, what each record represents, gaps.

Every expected instant and window below is an absolute date written by hand;
none is computed with the code under test.  Fixture stamps are built from
hand-counted day numbers (or, for the decadal bounds fixture, with Python's
proleptic-Gregorian ``datetime.date``).
"""
from __future__ import annotations

import warnings
from datetime import date, timedelta

import numpy as np
import pytest

cftime = pytest.importorskip("cftime")

from kelp.readers.fesom14._timeaxis import (  # noqa: E402
    FileAxis,
    ResolvedAxis,
    TimeOptions,
    first_difference,
    parse_output_schedule,
    resolve_axis,
)

DAY = 86400.0
MONTHLY = "unit: m first: 1 rate: 1"
NO_OPTS = TimeOptions()


# ----------------------------------------------------------------------------
# fixture helpers
# ----------------------------------------------------------------------------

def raw(path, epoch, days, *, schedule=MONTHLY, cal="noleap", early_s=0.0):
    """A raw-writer file: ``seconds since <epoch> 0:0:0``, stamps at ``days``
    (minus ``early_s`` seconds, the mid-day start case)."""
    vals = np.asarray(days, dtype=np.float64) * DAY - early_s
    return FileAxis(values=vals, units=f"seconds since {epoch} 0:0:0",
                    calendar=cal, bounds=None, schedule=schedule, path=path)


def plain(path, units, values, *, cal="standard"):
    """A file with neither time bounds nor an output schedule."""
    return FileAxis(values=np.asarray(values, dtype=np.float64), units=units,
                    calendar=cal, bounds=None, schedule=None, path=path)


def bounded(path, units, stamps, bounds, *, cal="proleptic_gregorian"):
    """A file with time bounds."""
    return FileAxis(values=np.asarray(stamps, dtype=np.float64), units=units,
                    calendar=cal, bounds=np.asarray(bounds, dtype=np.float64),
                    schedule=None, path=path)


def monthly_bounds(epoch_year, first_year, n_months):
    """Mid-month stamps and month bounds in days since ``epoch_year``-01-01,
    counted with Python's proleptic-Gregorian ``date`` (not the code under
    test)."""
    e = date(epoch_year, 1, 1)
    b = []
    y, m = first_year, 1
    for _ in range(n_months):
        y2, m2 = (y + 1, 1) if m == 12 else (y, m + 1)
        b.append(((date(y, m, 1) - e).days, (date(y2, m2, 1) - e).days))
        y, m = y2, m2
    b = np.asarray(b, dtype=np.float64)
    return b.mean(axis=1), b


def iso(arr):
    return [t.isoformat() for t in arr]


def iso_w(windows):
    return [(a.isoformat(), b.isoformat()) for a, b in windows]


def refusal(files, opts=NO_OPTS, variable="uo"):
    with pytest.raises(ValueError) as exc:
        resolve_axis(files, opts, variable=variable)
    return str(exc.value)


def assert_names(msg, *parts):
    for p in parts:
        assert p in msg, f"{p!r} not in message:\n{msg}"


# ----------------------------------------------------------------------------
# options and schedule parsing
# ----------------------------------------------------------------------------

class TestOptionsValidate:
    @pytest.mark.parametrize("opts", [
        TimeOptions(),
        TimeOptions(offset="+6 h"),
        TimeOptions(offset="-1 d"),
        TimeOptions(offset="30 min"),
        TimeOptions(stamp="end", window="calendar_month"),
        TimeOptions(stamp="start", window="calendar_year"),
        TimeOptions(stamp="centre", window="calendar_day"),
        TimeOptions(stamp="instant", window="1 d"),
        TimeOptions(stamp="end", window="5 d", offset="+0.5 h"),
        TimeOptions(stamp="centre", window="3600 s"),
        TimeOptions(stamp="instant", window="90 min"),
    ])
    def test_valid(self, opts):
        opts.validate()

    @pytest.mark.parametrize("opts,key", [
        (TimeOptions(stamp="middle", window="calendar_month"), "data.time_stamp"),
        (TimeOptions(stamp="end", window="monthly"), "data.time_window"),
        (TimeOptions(stamp="end", window="1 week"), "data.time_window"),
        (TimeOptions(stamp="end", window="0 d"), "data.time_window"),
        (TimeOptions(stamp="end", window="-5 d"), "data.time_window"),
        (TimeOptions(offset="6 hours"), "data.time_offset"),
        (TimeOptions(offset="six h"), "data.time_offset"),
    ])
    def test_invalid(self, opts, key):
        with pytest.raises(ValueError, match=key):
            opts.validate()


def test_options_incomplete_refused():
    with pytest.raises(ValueError) as exc:
        TimeOptions(stamp="end").validate()
    assert_names(str(exc.value), "data.time_stamp", "data.time_window")
    with pytest.raises(ValueError) as exc:
        TimeOptions(window="calendar_month", offset="+6 h").validate()
    assert_names(str(exc.value), "data.time_stamp", "data.time_window")
    # resolve_axis validates too, before anything is decoded
    f = plain("a.nc", "days since 1950-01-01", [0.0, 1.0])
    msg = refusal([f], TimeOptions(stamp="instant"))
    assert_names(msg, "data.time_stamp", "data.time_window")


class TestParseOutputSchedule:
    def test_writer_forms(self):
        assert parse_output_schedule("unit: m first: 1 rate: 1") == ("m", 1, 1)
        assert parse_output_schedule("unit: d first: 1 rate: 5") == ("d", 1, 5)
        assert parse_output_schedule("unit: y first: 1 rate: 2") == ("y", 1, 2)
        assert parse_output_schedule("unit: s first: 1 rate: 96") == ("s", 1, 96)
        assert parse_output_schedule("  unit:m  first:72 rate:1 ") == ("m", 72, 1)

    @pytest.mark.parametrize("text", [
        "", "monthly", "unit: m rate: 1", "unit: w first: 1 rate: 1",
        "unit: m first: 1 rate: 0", "unit: m first: -1 rate: 1",
        "unit: m first: 1 rate: 1 extra",
    ])
    def test_refused(self, text):
        with pytest.raises(ValueError):
            parse_output_schedule(text)


# ----------------------------------------------------------------------------
# bounds
# ----------------------------------------------------------------------------

def test_bounds_per_file_epochs():
    sa, ba = monthly_bounds(1951, 1951, 120)          # 1951-01 .. 1960-12
    sb, bb = monthly_bounds(1961, 1961, 12)           # 1961-01 .. 1961-12
    fa = bounded("uo_1951.nc", "days since 1951-1-1", sa, ba)
    fb = bounded("uo_1961.nc", "days since 1961-1-1", sb, bb)
    for files in ([fa, fb], [fb, fa]):                # caller order is irrelevant
        ax = resolve_axis(files, NO_OPTS, variable="uo")
        assert isinstance(ax, ResolvedAxis)
        assert ax.source == "bounds"
        assert len(ax.instants) == 132
        s = iso(ax.instants)
        assert s[0] == "1951-01-16T12:00:00"
        assert s[1] == "1951-02-15T00:00:00"
        assert s[13] == "1952-02-15T12:00:00"          # leap February, 29 d
        assert s[119] == "1960-12-16T12:00:00"
        assert s[120] == "1961-01-16T12:00:00"         # second file, own epoch
        assert s[131] == "1961-12-16T12:00:00"
        w = iso_w(ax.windows)
        assert w[0] == ("1951-01-01T00:00:00", "1951-02-01T00:00:00")
        assert w[119] == ("1960-12-01T00:00:00", "1961-01-01T00:00:00")
        assert w[120] == ("1961-01-01T00:00:00", "1961-02-01T00:00:00")
        assert {t.calendar for t in ax.instants} == {"proleptic_gregorian"}


def test_bounds_window_decides_not_the_stamp():
    # stamps at the month END, bounds give the month: the centre follows the
    # bounds, the stamp is not the instant
    f = bounded("t.nc", "days since 1951-1-1", [31.0, 59.0, 90.0],
                [[0, 31], [31, 59], [59, 90]], cal="standard")
    ax = resolve_axis([f], NO_OPTS, variable="uo")
    assert iso(ax.instants) == ["1951-01-16T12:00:00", "1951-02-15T00:00:00",
                                "1951-03-16T12:00:00"]


def test_calendar_strings_compare_by_value():
    stamps = [15.5, 45.0, 74.5]
    bnds = [[0, 31], [31, 59], [59, 90]]
    a = resolve_axis([bounded("uo.nc", "days since 1951-1-1", stamps, bnds,
                              cal="proleptic_gregorian")], NO_OPTS, variable="uo")
    b = resolve_axis([bounded("zos.nc", "days since 1951-1-1", stamps, bnds,
                              cal="standard")], NO_OPTS, variable="zos")
    expected = ["1951-01-16T12:00:00", "1951-02-15T00:00:00",
                "1951-03-16T12:00:00"]
    assert iso(a.instants) == expected
    assert iso(b.instants) == expected
    assert first_difference(a, b) is None
    assert first_difference(b, a) is None
    # one variable whose files carry the two names: one calendar after 1582
    sa, ba = monthly_bounds(1951, 1951, 120)
    sb, bb = monthly_bounds(1961, 1961, 12)
    ax = resolve_axis([
        bounded("v_1951.nc", "days since 1951-1-1", sa, ba, cal="standard"),
        bounded("v_1961.nc", "days since 1961-1-1", sb, bb,
                cal="proleptic_gregorian")], NO_OPTS, variable="vo")
    assert len({t.calendar for t in ax.instants}) == 1
    assert {w.calendar for w in ax.windows.ravel()} == {ax.instants[0].calendar}
    assert iso(ax.instants)[119:121] == ["1960-12-16T12:00:00",
                                         "1961-01-16T12:00:00"]


def test_bounds_missing_record_refused():
    # January, February, April: March is missing inside the file
    f = bounded("uo_1951.nc", "days since 1951-1-1", [15.5, 45.0, 105.0],
                [[0, 31], [31, 59], [90, 120]])
    msg = refusal([f])
    assert_names(msg, "uo_1951.nc", "'uo'", "1951-03-01T00:00:00",
                 "1951-04-01T00:00:00")


def test_bounds_with_offset():
    f = bounded("uo.nc", "days since 1951-1-1", [15.5, 45.0],
                [[0, 31], [31, 59]], cal="standard")
    ax = resolve_axis([f], TimeOptions(offset="+6 h"), variable="uo")
    assert ax.source == "bounds"
    assert iso(ax.instants) == ["1951-01-16T18:00:00", "1951-02-15T06:00:00"]
    assert iso_w(ax.windows) == [
        ("1951-01-01T06:00:00", "1951-02-01T06:00:00"),
        ("1951-02-01T06:00:00", "1951-03-01T06:00:00")]


def test_bounds_empty_window_refused():
    f = bounded("uo.nc", "days since 1951-1-1", [15.5, 31.0],
                [[0, 31], [31, 31]])
    msg = refusal([f])
    assert_names(msg, "uo.nc", "'uo'", "1951-02-01T00:00:00")


# ----------------------------------------------------------------------------
# raw output_schedule
# ----------------------------------------------------------------------------

def test_raw_monthly_noleap():
    # noleap: 1948 has a 28-day February; stamps at day 31, 59, 90
    f = raw("uo_run_19480101.nc", "1948-01-01", [31, 59, 90])
    ax = resolve_axis([f], NO_OPTS, variable="uo")
    assert ax.source == "schedule"
    assert iso(ax.instants) == ["1948-01-16T12:00:00", "1948-02-15T00:00:00",
                                "1948-03-16T12:00:00"]
    assert iso_w(ax.windows) == [
        ("1948-01-01T00:00:00", "1948-02-01T00:00:00"),
        ("1948-02-01T00:00:00", "1948-03-01T00:00:00"),
        ("1948-03-01T00:00:00", "1948-04-01T00:00:00")]
    assert ax.windows[1][1] - ax.windows[1][0] == timedelta(days=28)
    assert {t.calendar for t in ax.instants} == {"noleap"}


def test_raw_rate_3():
    sched = "unit: m first: 1 rate: 3"
    f = raw("uo_run_19480101.nc", "1948-01-01", [90, 181, 273, 365],
            schedule=sched)
    ax = resolve_axis([f], NO_OPTS, variable="uo")
    assert iso_w(ax.windows) == [
        ("1948-01-01T00:00:00", "1948-04-01T00:00:00"),
        ("1948-04-01T00:00:00", "1948-07-01T00:00:00"),
        ("1948-07-01T00:00:00", "1948-10-01T00:00:00"),
        ("1948-10-01T00:00:00", "1949-01-01T00:00:00")]
    assert iso(ax.instants) == ["1948-02-15T00:00:00", "1948-05-16T12:00:00",
                                "1948-08-16T00:00:00", "1948-11-16T00:00:00"]
    # monthly stamps under a rate-3 schedule are not its output events
    g = raw("uo_run_19480101.nc", "1948-01-01", [31, 59, 90], schedule=sched)
    assert_names(refusal([g]), "uo_run_19480101.nc", "1948-02-01T00:00:00",
                 "1948-04-01T00:00:00")


def test_raw_daily():
    # noleap: day 2 of the job is 02-28, whose end is 03-01
    f = raw("thetao_run_19480227.nc", "1948-02-27", [1, 2, 3],
            schedule="unit: d first: 1 rate: 1")
    ax = resolve_axis([f], NO_OPTS, variable="thetao")
    assert iso(ax.instants) == ["1948-02-27T12:00:00", "1948-02-28T12:00:00",
                                "1948-03-01T12:00:00"]
    assert iso_w(ax.windows) == [
        ("1948-02-27T00:00:00", "1948-02-28T00:00:00"),
        ("1948-02-28T00:00:00", "1948-03-01T00:00:00"),
        ("1948-03-01T00:00:00", "1948-03-02T00:00:00")]


def test_raw_daily_rate_2():
    f = raw("zos_run_19480101.nc", "1948-01-01", [2, 4],
            schedule="unit: d first: 1 rate: 2")
    ax = resolve_axis([f], NO_OPTS, variable="zos")
    assert iso(ax.instants) == ["1948-01-02T00:00:00", "1948-01-04T00:00:00"]
    assert iso_w(ax.windows) == [
        ("1948-01-01T00:00:00", "1948-01-03T00:00:00"),
        ("1948-01-03T00:00:00", "1948-01-05T00:00:00")]


def test_raw_two_segments():
    s1 = raw("uo_run_19480101.nc", "1948-01-01", [31, 59])
    s2 = raw("uo_run_19480301.nc", "1948-03-01", [31, 61])
    for files in ([s1, s2], [s2, s1]):
        ax = resolve_axis(files, NO_OPTS, variable="uo")
        assert iso(ax.instants) == [
            "1948-01-16T12:00:00", "1948-02-15T00:00:00",
            "1948-03-16T12:00:00", "1948-04-16T00:00:00"]
        assert iso_w(ax.windows)[2] == ("1948-03-01T00:00:00",
                                        "1948-04-01T00:00:00")


def test_raw_single_record_last_segment():
    # standard calendar: the last segment is the leap February of 1948 alone
    s1 = raw("uo_run_19471201.nc", "1947-12-01", [31, 62], cal="standard")
    s2 = raw("uo_run_19480201.nc", "1948-02-01", [29], cal="standard")
    ax = resolve_axis([s1, s2], NO_OPTS, variable="uo")
    assert iso(ax.instants) == ["1947-12-16T12:00:00", "1948-01-16T12:00:00",
                                "1948-02-15T12:00:00"]
    assert iso_w(ax.windows)[2] == ("1948-02-01T00:00:00",
                                    "1948-03-01T00:00:00")
    # the single-record segment alone resolves the same way
    alone = resolve_axis([s2], NO_OPTS, variable="uo")
    assert iso(alone.instants) == ["1948-02-15T12:00:00"]
    assert iso_w(alone.windows) == [("1948-02-01T00:00:00",
                                     "1948-03-01T00:00:00")]


def test_raw_missing_interior_record():
    f = raw("uo_run_19480101.nc", "1948-01-01", [31, 90])   # no 03-01 record
    msg = refusal([f])
    assert_names(msg, "uo_run_19480101.nc", "'uo'", "1948-02-01T00:00:00",
                 "1948-03-01T00:00:00")


def test_raw_missing_first_record():
    f = raw("uo_run_19480101.nc", "1948-01-01", [59, 90])   # no 02-01 record
    msg = refusal([f])
    assert_names(msg, "uo_run_19480101.nc", "'uo'", "1948-01-01T00:00:00",
                 "1948-02-01T00:00:00")


def test_raw_midday_start_offset():
    # a job started at 1948-01-01 06:00 stamps dt*istep against a midnight
    # epoch: its month ends sit 6 h early
    f = raw("uo_run_19480101.nc", "1948-01-01", [31, 59, 90],
            early_s=6 * 3600.0)
    msg = refusal([f])
    assert_names(msg, "uo_run_19480101.nc", "'uo'", "1948-01-31T18:00:00",
                 "data.time_offset")
    ax = resolve_axis([f], TimeOptions(offset="+6 h"), variable="uo")
    assert ax.source == "schedule"
    assert iso_w(ax.windows) == [
        ("1948-01-01T06:00:00", "1948-02-01T00:00:00"),
        ("1948-02-01T00:00:00", "1948-03-01T00:00:00"),
        ("1948-03-01T00:00:00", "1948-04-01T00:00:00")]
    assert iso(ax.instants) == ["1948-01-16T15:00:00", "1948-02-15T00:00:00",
                                "1948-03-16T12:00:00"]


def test_uniform_offset_shift():
    f = plain("snap.nc", "days since 1950-01-01", [0, 1, 2, 3, 4])
    base = resolve_axis([f], TimeOptions(stamp="instant", window="1 d"),
                        variable="uo")
    assert base.source == "options"
    assert base.windows is None
    assert iso(base.instants) == [
        "1950-01-01T00:00:00", "1950-01-02T00:00:00", "1950-01-03T00:00:00",
        "1950-01-04T00:00:00", "1950-01-05T00:00:00"]
    shifted = resolve_axis(
        [f], TimeOptions(stamp="instant", window="1 d", offset="-1 d"),
        variable="uo")
    assert iso(shifted.instants) == [
        "1949-12-31T00:00:00", "1950-01-01T00:00:00", "1950-01-02T00:00:00",
        "1950-01-03T00:00:00", "1950-01-04T00:00:00"]


def test_raw_first_2():
    f = raw("uo_run_19480101.nc", "1948-01-01", [59, 90],
            schedule="unit: m first: 2 rate: 1")
    ax = resolve_axis([f], NO_OPTS, variable="uo")
    assert iso_w(ax.windows) == [
        ("1948-01-01T00:00:00", "1948-03-01T00:00:00"),
        ("1948-03-01T00:00:00", "1948-04-01T00:00:00")]
    assert iso(ax.instants) == ["1948-01-30T12:00:00", "1948-03-16T12:00:00"]


def test_raw_yearly_rate_2():
    # from an odd start year: the writer fires at the end of the even years
    sched = "unit: y first: 1 rate: 2"
    f = raw("uo_run_19490101.nc", "1949-01-01", [730, 1461], schedule=sched,
            cal="standard")
    ax = resolve_axis([f], NO_OPTS, variable="uo")
    assert iso_w(ax.windows) == [
        ("1949-01-01T00:00:00", "1951-01-01T00:00:00"),
        ("1951-01-01T00:00:00", "1953-01-01T00:00:00")]
    assert iso(ax.instants) == ["1950-01-01T00:00:00", "1952-01-01T12:00:00"]
    # a stamp at the end of 1949 (odd) is not an output event
    g = raw("uo_run_19490101.nc", "1949-01-01", [365], schedule=sched,
            cal="standard")
    assert_names(refusal([g]), "uo_run_19490101.nc", "1950-01-01T00:00:00",
                 "1951-01-01T00:00:00")


def test_raw_yearly_not_january_start_refused():
    # the writer counts the months of a year from the job's start month: a
    # yearly schedule from a job that started in July is not resolved here
    f = raw("uo_run_19490701.nc", "1949-07-01", [365],
            schedule="unit: y first: 1 rate: 1", cal="standard")
    msg = refusal([f])
    assert_names(msg, "uo_run_19490701.nc", "'uo'", "data.time_stamp",
                 "data.time_window")


def test_raw_gap_between_segments_refused():
    s1 = raw("uo_run_19480101.nc", "1948-01-01", [31, 59])
    s2 = raw("uo_run_19480401.nc", "1948-04-01", [30])
    assert_names(refusal([s1, s2]), "uo_run_19480101.nc", "uo_run_19480401.nc",
                 "'uo'", "1948-03-01T00:00:00", "1948-04-01T00:00:00")
    # after the offset: two jobs that each started at 06:00 -- the six hours
    # between the first job's last month end and the second job's start were
    # never written
    e1 = raw("uo_run_19480101.nc", "1948-01-01", [31, 59], early_s=6 * 3600.0)
    e2 = raw("uo_run_19480301.nc", "1948-03-01", [31], early_s=6 * 3600.0)
    off = TimeOptions(offset="+6 h")
    assert iso(resolve_axis([e1], off, variable="uo").instants) == [
        "1948-01-16T15:00:00", "1948-02-15T00:00:00"]
    assert iso(resolve_axis([e2], off, variable="uo").instants) == [
        "1948-03-16T15:00:00"]
    assert_names(refusal([e1, e2], off), "uo_run_19480101.nc",
                 "uo_run_19480301.nc", "'uo'", "1948-03-01T00:00:00",
                 "1948-03-01T06:00:00")


def test_raw_unit_s_refused():
    f = raw("uo_run_19480101.nc", "1948-01-01", [1, 2],
            schedule="unit: s first: 1 rate: 96")
    assert_names(refusal([f]), "uo_run_19480101.nc", "'uo'", "unit: s",
                 "data.time_stamp", "data.time_window")


def test_raw_unparsable_schedule_refused():
    f = raw("uo_run_19480101.nc", "1948-01-01", [31],
            schedule="every month")
    assert_names(refusal([f]), "uo_run_19480101.nc", "'uo'", "every month",
                 "data.time_stamp", "data.time_window")


def test_raw_units_not_writer_style_refused():
    # the raw writer stamps seconds since a date at 0:0:0
    in_days = FileAxis(values=np.array([31.0, 59.0]),
                       units="days since 1948-01-01", calendar="noleap",
                       bounds=None, schedule=MONTHLY, path="re_timed.nc")
    assert_names(refusal([in_days]), "re_timed.nc", "'uo'", "seconds since",
                 "data.time_stamp")
    not_midnight = FileAxis(values=np.array([31.0, 59.0]) * DAY,
                            units="seconds since 1948-01-01 06:00:00",
                            calendar="noleap", bounds=None, schedule=MONTHLY,
                            path="shifted_epoch.nc")
    assert_names(refusal([not_midnight]), "shifted_epoch.nc", "0:0:0")


# ----------------------------------------------------------------------------
# within a variable
# ----------------------------------------------------------------------------

def test_overlapping_segments_refused():
    s1 = raw("uo_run_19480101.nc", "1948-01-01", [31, 59, 90])
    s2 = raw("uo_run_19480301.nc", "1948-03-01", [31, 61])
    msg = refusal([s1, s2])
    assert_names(msg, "uo_run_19480101.nc", "uo_run_19480301.nc", "'uo'",
                 "overlap")
    # the same decade given twice
    sa, ba = monthly_bounds(1951, 1951, 12)
    fa = bounded("uo_a.nc", "days since 1951-1-1", sa, ba)
    fb = bounded("uo_b.nc", "days since 1951-1-1", sa, ba)
    assert_names(refusal([fa, fb]), "uo_a.nc", "uo_b.nc")


def test_duplicate_record_refused():
    f = bounded("uo.nc", "days since 1951-1-1", [15.5, 45.0, 45.0],
                [[0, 31], [31, 59], [31, 59]])
    assert_names(refusal([f]), "uo.nc", "'uo'", "duplicate",
                 "1951-02-15T00:00:00")
    g = raw("uo_run_19480101.nc", "1948-01-01", [31, 31, 59])
    assert_names(refusal([g]), "uo_run_19480101.nc", "duplicate",
                 "1948-02-01T00:00:00")
    h = plain("snap.nc", "days since 1950-01-01", [0, 1, 1, 2])
    assert_names(refusal([h], TimeOptions(stamp="instant", window="1 d")),
                 "snap.nc", "duplicate", "1950-01-02T00:00:00")


def test_decreasing_stamps_refused():
    f = plain("snap.nc", "days since 1950-01-01", [0, 2, 1])
    assert_names(refusal([f], TimeOptions(stamp="instant", window="1 d")),
                 "snap.nc", "'uo'", "1950-01-03T00:00:00",
                 "1950-01-02T00:00:00")


def test_mixed_calendar_within_variable_refused():
    s1 = raw("uo_run_19480101.nc", "1948-01-01", [31, 59], cal="noleap")
    s2 = raw("uo_run_19480301.nc", "1948-03-01", [31], cal="standard")
    msg = refusal([s1, s2])
    assert_names(msg, "uo_run_19480101.nc", "uo_run_19480301.nc", "'uo'",
                 "noleap", "standard")


def test_mixed_sources_within_variable_refused():
    sa, ba = monthly_bounds(1948, 1948, 2)
    fa = bounded("uo_bounds.nc", "days since 1948-1-1", sa, ba, cal="standard")
    fb = raw("uo_run_19480301.nc", "1948-03-01", [31], cal="standard")
    msg = refusal([fa, fb])
    assert_names(msg, "uo_bounds.nc", "uo_run_19480301.nc", "'uo'",
                 "data.time_stamp", "data.time_window")


def test_empty_file_contributes_no_record():
    s1 = raw("uo_run_19480101.nc", "1948-01-01", [31, 59])
    empty = raw("uo_run_19480301.nc", "1948-03-01", [])
    ax = resolve_axis([s1, empty], NO_OPTS, variable="uo")
    assert iso(ax.instants) == ["1948-01-16T12:00:00", "1948-02-15T00:00:00"]
    assert_names(refusal([empty]), "'uo'", "no record")


def test_no_files_refused():
    with pytest.raises(ValueError, match="'uo'"):
        resolve_axis([], NO_OPTS, variable="uo")


def test_missing_stamp_refused():
    f = plain("snap.nc", "days since 1950-01-01", [0.0, np.nan, 2.0])
    assert_names(refusal([f], TimeOptions(stamp="instant", window="1 d")),
                 "snap.nc", "'uo'")


# ----------------------------------------------------------------------------
# explicit options
# ----------------------------------------------------------------------------

END_MONTH = TimeOptions(stamp="end", window="calendar_month")


def test_options_end_calendar_month():
    # FESOM2-style month-end stamps (standard calendar, leap 1948) across
    # unequal months, then a file holding a single record
    a = plain("a.nc", "days since 1948-01-01", [31, 60, 91])
    b = plain("b.nc", "days since 1948-05-01", [0])
    ax = resolve_axis([a, b], END_MONTH, variable="uo")
    assert ax.source == "options"
    assert iso_w(ax.windows) == [
        ("1948-01-01T00:00:00", "1948-02-01T00:00:00"),
        ("1948-02-01T00:00:00", "1948-03-01T00:00:00"),
        ("1948-03-01T00:00:00", "1948-04-01T00:00:00"),
        ("1948-04-01T00:00:00", "1948-05-01T00:00:00")]
    assert iso(ax.instants) == ["1948-01-16T12:00:00", "1948-02-15T12:00:00",
                                "1948-03-16T12:00:00", "1948-04-16T00:00:00"]
    single = resolve_axis([b], END_MONTH, variable="uo")
    assert iso(single.instants) == ["1948-04-16T00:00:00"]


def test_options_start_calendar_month():
    f = plain("a.nc", "days since 1948-01-01", [0, 31, 60])
    ax = resolve_axis([f], TimeOptions(stamp="start", window="calendar_month"),
                      variable="uo")
    assert iso_w(ax.windows) == [
        ("1948-01-01T00:00:00", "1948-02-01T00:00:00"),
        ("1948-02-01T00:00:00", "1948-03-01T00:00:00"),
        ("1948-03-01T00:00:00", "1948-04-01T00:00:00")]
    assert iso(ax.instants) == ["1948-01-16T12:00:00", "1948-02-15T12:00:00",
                                "1948-03-16T12:00:00"]


def test_options_centre_calendar_month():
    opts = TimeOptions(stamp="centre", window="calendar_month")
    f = plain("a.nc", "days since 1948-01-01", [15.5, 45.0, 74.5], cal="noleap")
    ax = resolve_axis([f], opts, variable="uo")
    assert iso(ax.instants) == ["1948-01-16T12:00:00", "1948-02-15T00:00:00",
                                "1948-03-16T12:00:00"]
    assert iso_w(ax.windows) == [
        ("1948-01-01T00:00:00", "1948-02-01T00:00:00"),
        ("1948-02-01T00:00:00", "1948-03-01T00:00:00"),
        ("1948-03-01T00:00:00", "1948-04-01T00:00:00")]
    # a stamp that is not the centre of its month
    g = plain("b.nc", "days since 1948-01-01", [15.0], cal="noleap")
    assert_names(refusal([g], opts), "b.nc", "'uo'", "1948-01-16T00:00:00",
                 "data.time_offset")


def test_options_calendar_year_and_day():
    # standard calendar: 1948 has 366 days, its centre is 07-02 00:00
    f = plain("y.nc", "days since 1948-01-01", [366, 731])
    ax = resolve_axis([f], TimeOptions(stamp="end", window="calendar_year"),
                      variable="uo")
    assert iso(ax.instants) == ["1948-07-02T00:00:00", "1949-07-02T12:00:00"]
    d = plain("d.nc", "hours since 1950-01-01", [12, 36])
    ax = resolve_axis([d], TimeOptions(stamp="centre", window="calendar_day"),
                      variable="uo")
    assert iso_w(ax.windows) == [
        ("1950-01-01T00:00:00", "1950-01-02T00:00:00"),
        ("1950-01-02T00:00:00", "1950-01-03T00:00:00")]


def test_options_instant_1d():
    # a fixed cadence has no calendar anchor: 12:00 snapshots are fine
    f = plain("snap.nc", "hours since 1950-01-01", [12, 36, 60])
    ax = resolve_axis([f], TimeOptions(stamp="instant", window="1 d"),
                      variable="uo")
    assert ax.windows is None
    assert iso(ax.instants) == ["1950-01-01T12:00:00", "1950-01-02T12:00:00",
                                "1950-01-03T12:00:00"]
    # snapshots at month starts with a calendar_month cadence
    g = plain("m.nc", "days since 1948-01-01", [0, 31, 60], cal="standard")
    ax = resolve_axis([g], TimeOptions(stamp="instant", window="calendar_month"),
                      variable="uo")
    assert iso(ax.instants) == ["1948-01-01T00:00:00", "1948-02-01T00:00:00",
                                "1948-03-01T00:00:00"]


def test_options_5d():
    f = plain("a.nc", "days since 1950-01-01", [5, 10, 15])
    ax = resolve_axis([f], TimeOptions(stamp="end", window="5 d"),
                      variable="uo")
    assert iso_w(ax.windows) == [
        ("1950-01-01T00:00:00", "1950-01-06T00:00:00"),
        ("1950-01-06T00:00:00", "1950-01-11T00:00:00"),
        ("1950-01-11T00:00:00", "1950-01-16T00:00:00")]
    assert iso(ax.instants) == ["1950-01-03T12:00:00", "1950-01-08T12:00:00",
                                "1950-01-13T12:00:00"]
    g = plain("b.nc", "days since 1950-01-01", [2.5, 7.5])
    ax = resolve_axis([g], TimeOptions(stamp="centre", window="5 d"),
                      variable="uo")
    assert iso_w(ax.windows) == [
        ("1950-01-01T00:00:00", "1950-01-06T00:00:00"),
        ("1950-01-06T00:00:00", "1950-01-11T00:00:00")]
    assert iso(ax.instants) == ["1950-01-03T12:00:00", "1950-01-08T12:00:00"]
    h = plain("c.nc", "days since 1950-01-01", [0, 5])
    ax = resolve_axis([h], TimeOptions(stamp="start", window="5 d"),
                      variable="uo")
    assert iso(ax.instants) == ["1950-01-03T12:00:00", "1950-01-08T12:00:00"]


def test_options_off_boundary_refused():
    # month-end stamps 6 h early (a mid-day job start)
    f = FileAxis(values=np.array([31.0, 59.0]) * DAY - 6 * 3600.0,
                 units="seconds since 1948-01-01 0:0:0", calendar="noleap",
                 bounds=None, schedule=None, path="early.nc")
    msg = refusal([f], END_MONTH)
    assert_names(msg, "early.nc", "'uo'", "1948-01-31T18:00:00",
                 "data.time_offset")
    ax = resolve_axis([f], TimeOptions(stamp="end", window="calendar_month",
                                       offset="+6 h"), variable="uo")
    assert iso(ax.instants) == ["1948-01-16T12:00:00", "1948-02-15T00:00:00"]
    g = plain("d.nc", "hours since 1950-01-01", [30])
    assert_names(refusal([g], TimeOptions(stamp="end", window="calendar_day")),
                 "d.nc", "1950-01-02T06:00:00", "data.time_offset")


def test_options_override_warns():
    # bounds: the second record's bounds are not its calendar month
    f = bounded("uo_b.nc", "days since 1951-1-1", [31.0, 59.0, 90.0],
                [[0, 31], [31, 45], [45, 90]], cal="standard")
    with pytest.warns(UserWarning) as rec:
        ax = resolve_axis([f], END_MONTH, variable="uo")
    assert ax.source == "options"
    assert iso(ax.instants) == ["1951-01-16T12:00:00", "1951-02-15T00:00:00",
                                "1951-03-16T12:00:00"]
    msg = " ".join(str(r.message) for r in rec)
    assert_names(msg, "uo_b.nc", "'uo'", "bounds", "1951-02-15T00:00:00",
                 "1951-02-08T00:00:00")
    # schedule: a job that started mid-month (01-15)
    g = raw("uo_run_19480115.nc", "1948-01-15", [17, 45])
    with pytest.warns(UserWarning) as rec:
        ax = resolve_axis([g], END_MONTH, variable="uo")
    assert ax.source == "options"
    assert iso(ax.instants) == ["1948-01-16T12:00:00", "1948-02-15T00:00:00"]
    msg = " ".join(str(r.message) for r in rec)
    assert_names(msg, "uo_run_19480115.nc", "'uo'", "output_schedule",
                 "1948-01-16T12:00:00", "1948-01-23T12:00:00")
    # without options the schedule decides
    auto = resolve_axis([g], NO_OPTS, variable="uo")
    assert iso(auto.instants) == ["1948-01-23T12:00:00", "1948-02-15T00:00:00"]
    # a schedule the reader cannot use (unit s): the options still decide, and
    # the warning says why the schedule was not usable
    h = raw("uo_run_19480101.nc", "1948-01-01", [31, 59],
            schedule="unit: s first: 1 rate: 96")
    with pytest.warns(UserWarning, match="output_schedule"):
        ax = resolve_axis([h], END_MONTH, variable="uo")
    assert iso(ax.instants) == ["1948-01-16T12:00:00", "1948-02-15T00:00:00"]
    # a file with neither source: no warning
    p = plain("p.nc", "days since 1948-01-01", [31, 59], cal="noleap")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        resolve_axis([p], END_MONTH, variable="uo")


def test_options_gap_refused():
    # standard 1948: February, March and May ends — April's window is missing
    f = plain("a.nc", "days since 1948-01-01", [31, 60, 121])
    msg = refusal([f], END_MONTH)
    assert_names(msg, "a.nc", "'uo'", "1948-03-01T00:00:00",
                 "1948-04-01T00:00:00")
    # snapshots: the 1950-01-03 record is missing
    g = plain("s.nc", "days since 1950-01-01", [0, 1, 3])
    msg = refusal([g], TimeOptions(stamp="instant", window="1 d"))
    assert_names(msg, "s.nc", "'uo'", "1950-01-03T00:00:00")
    # a gap between two files
    a = plain("a1.nc", "days since 1948-01-01", [31])
    b = plain("a2.nc", "days since 1948-04-01", [0])
    assert_names(refusal([a, b], END_MONTH), "a1.nc", "a2.nc",
                 "1948-02-01T00:00:00", "1948-03-01T00:00:00")


def test_options_irregular_fixed_refused():
    f = plain("a.nc", "days since 1950-01-01", [5, 8])
    msg = refusal([f], TimeOptions(stamp="end", window="5 d"))
    assert_names(msg, "a.nc", "'uo'", "overlap")


def test_no_source_refused():
    f = plain("plain.nc", "days since 1950-01-01", [0, 1, 2])
    msg = refusal([f])
    assert_names(msg, "plain.nc", "'uo'", "data.time_stamp",
                 "data.time_window")
    # an offset alone is not a source
    msg = refusal([f], TimeOptions(offset="+6 h"))
    assert_names(msg, "data.time_stamp", "data.time_window")


# ----------------------------------------------------------------------------
# first_difference
# ----------------------------------------------------------------------------

def _monthly_axis(cal, n, path="x.nc"):
    stamps = [15.5, 45.0, 74.5, 105.0][:n]
    bnds = [[0, 31], [31, 59], [59, 90], [90, 120]][:n]
    return resolve_axis([bounded(path, "days since 1951-1-1", stamps, bnds,
                                 cal=cal)], NO_OPTS, variable="x")


def test_first_difference():
    a = _monthly_axis("standard", 4)
    assert first_difference(a, a) is None
    assert first_difference(a, _monthly_axis("standard", 3)) == 3
    assert first_difference(_monthly_axis("standard", 3), a) == 3
    # equal instants, unequal windows
    w = a.windows.copy()
    w[3, 0] = cftime.datetime(1951, 3, 31, calendar="standard")
    w[3, 1] = cftime.datetime(1951, 5, 2, calendar="standard")
    b = ResolvedAxis(instants=a.instants.copy(), windows=w, source="bounds")
    assert first_difference(a, b) == 3
    # a mean and a snapshot at the same instants never agree
    snap = ResolvedAxis(instants=a.instants.copy(), windows=None,
                        source="options")
    assert first_difference(a, snap) == 0
    assert first_difference(snap, snap) is None
    # another calendar differs from the first record on
    assert first_difference(a, _monthly_axis("noleap", 4)) == 0


# ----------------------------------------------------------------------------
# the instants form one date type (xarray builds a CFTimeIndex from them)
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("make", [
    lambda: resolve_axis([raw("a.nc", "1948-01-15", [17, 45])], NO_OPTS,
                         variable="uo"),
    lambda: resolve_axis([plain("b.nc", "days since 1948-01-01", [31, 60, 91])],
                         END_MONTH, variable="uo"),
    lambda: resolve_axis([plain("c.nc", "days since 1948-01-01", [0, 31])],
                         TimeOptions(stamp="instant", window="calendar_month"),
                         variable="uo"),
    lambda: resolve_axis([
        bounded("d1.nc", "days since 1951-1-1", [15.5], [[0, 31]],
                cal="standard"),
        bounded("d2.nc", "days since 1951-2-1", [14.0], [[0, 28]],
                cal="proleptic_gregorian")], NO_OPTS, variable="uo"),
])
def test_one_date_type(make):
    xr = pytest.importorskip("xarray")
    ax = make()
    dates = list(ax.instants)
    if ax.windows is not None:
        dates += list(ax.windows.ravel())
    assert len({type(t) for t in dates}) == 1, {type(t) for t in dates}
    da = xr.DataArray(np.zeros(len(ax.instants)), dims="time",
                      coords={"time": ax.instants})
    assert type(da.indexes["time"]).__name__ == "CFTimeIndex"
