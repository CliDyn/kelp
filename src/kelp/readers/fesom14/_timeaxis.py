"""Time axis of FESOM 1.4 output: decoding, what each record represents, gaps.

Every file is decoded on its own (its own epoch and calendar).  For every
record of one variable the instant it represents — the centre of its averaging
window for a time mean, the stamp itself for a snapshot — and, for means, the
window are resolved ONCE, from exactly one source:

1. **explicit options** — ``data.time_stamp`` (where the stamp sits in the
   window: ``start | centre | end``, or ``instant`` for snapshots) together
   with ``data.time_window`` (the window or the snapshot cadence:
   ``calendar_month | calendar_year | calendar_day`` or a fixed duration
   ``"<number> <s|min|h|d>"``).  They decide for every file, whatever the
   files carry; when the files carry time bounds or an ``output_schedule``,
   a :class:`UserWarning` names what was overridden and the first record
   where the two disagree;
2. otherwise the files' **time bounds** (``time.attrs["bounds"]``): the window
   is given, its centre is the instant;
3. otherwise the raw writer's **output schedule** (global attribute
   ``output_schedule = "unit: <y|m|d|s> first: <n> rate: <n>"``).  The raw
   writer stamps each record with the model time in seconds since the job's
   start DATE at 00:00, at the END of the averaging window; the window starts
   right after the previous output event, the first one at the job start.
   Its output events are enumerated from the file's epoch and the stamps must
   be exactly those events.  Units ``d``/``m`` fire at the end of the
   ``count``-th day/month of the job (the day or month the job starts in is
   ``count == 1``) when ``first <= count`` and ``count % rate == 0``.  Unit
   ``y`` counts a year's months from the job's start month: it fires at the
   end of the 12th, 24th, ... month of the job when the calendar year ``Y``
   of that ending month satisfies ``first <= Y`` and ``Y % rate == 0`` (for
   a job that starts in January: the end of calendar year ``Y``; for one that
   starts in July: the end of June of ``Y``).  Unit ``s`` (a window of
   ``rate`` model steps; the step is not in the file) is refused.  Segment
   ``i + 1`` of a run must start (its epoch, after the offset) exactly at the
   last output event of segment ``i``.  A job that started at a time of day
   other than 00:00 also ends at that time, so the hours after its last
   output event are never written: a run of several such segments is refused
   as a gap.

An optional fixed offset (``data.time_offset``, ``"<±number> <s|min|h|d>"``)
is added to the decoded stamps, the bounds and the epoch of every file before
anything else, for every source — e.g. for a raw job that started at a time of
day other than 00:00, whose stamps all sit early by that time of day.  For a
raw-writer file the offset must keep the job start within its day (from 0 up
to, not including, 24 h); any other offset is refused.

A window is never estimated from the spacing of neighbouring stamps.  Every
record the source says must exist, from the first to the last one observed,
must be present: a gap (a missing record, month, year or file) refuses, as do
overlapping or duplicate records and files of one variable in calendars that
differ.  ``standard`` and ``proleptic_gregorian`` are one calendar for dates on
or after 1582-10-15.  Every refusal is a :class:`ValueError` naming the file,
the variable and the record (by its instant, ISO 8601) and saying the fix.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from datetime import timedelta
from typing import Iterator, List, Optional, Sequence, Tuple

import cftime
import numpy as np

__all__ = [
    "STAMPS",
    "CALENDAR_WINDOWS",
    "TOLERANCE",
    "TimeOptions",
    "FileAxis",
    "ResolvedAxis",
    "parse_output_schedule",
    "resolve_axis",
    "first_difference",
]

#: accepted values of ``data.time_stamp``
STAMPS = ("start", "centre", "end", "instant")
#: calendar conventions accepted by ``data.time_window``
CALENDAR_WINDOWS = ("calendar_month", "calendar_year", "calendar_day")
#: two times closer than this are the same instant (decoding noise of float
#: stamps is at the microsecond level)
TOLERANCE = timedelta(milliseconds=1)

_DURATION_SECONDS = {"s": 1.0, "min": 60.0, "h": 3600.0, "d": 86400.0}
_DURATION_RE = re.compile(
    r"^\s*([+-]?)\s*(\d+(?:\.\d*)?|\.\d+)\s*(s|min|h|d)\s*$")
_SCHEDULE_RE = re.compile(
    r"^\s*unit:\s*(\S+)\s+first:\s*(\S+)\s+rate:\s*(\S+)\s*$")
_SCHEDULE_UNITS = ("y", "m", "d", "s")
_SECONDS_TOKENS = ("seconds", "second", "secs", "sec", "s")
_REAL_CALENDARS = ("standard", "proleptic_gregorian")
_CALENDAR_ALIASES = {"gregorian": "standard", "365_day": "noleap",
                     "366_day": "all_leap"}
_GREGORIAN_START = (1582, 10, 15)

_OPTIONS_FIX = ("set data.time_stamp (start | centre | end | instant) and "
                "data.time_window (calendar_month | calendar_year | "
                "calendar_day | '<number> <s|min|h|d>')")


# ----------------------------------------------------------------------------
# public data classes
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class TimeOptions:
    """User options that decide what each record's time stamp means.

    Attributes
    ----------
    stamp : str or None
        ``data.time_stamp``: where the stamp sits in its averaging window
        (``start``, ``centre``, ``end``), or ``instant`` for snapshots.
    window : str or None
        ``data.time_window``: the averaging window (or, with ``instant``, the
        snapshot cadence): ``calendar_month``, ``calendar_year``,
        ``calendar_day`` or a fixed duration ``"<number> <s|min|h|d>"``.
    offset : str or None
        ``data.time_offset``: a fixed shift ``"<±number> <s|min|h|d>"`` added
        to the decoded stamps, bounds and epoch of every file before anything
        else.

    ``stamp`` and ``window`` are given together or not at all; ``offset``
    may be given alone.
    """

    stamp: Optional[str] = None
    window: Optional[str] = None
    offset: Optional[str] = None

    @property
    def explicit(self) -> bool:
        """True when the stamp placement and the window are given."""
        return self.stamp is not None and self.window is not None

    def validate(self) -> None:
        """Check every option; refuse an incomplete or invalid set.

        Raises
        ------
        ValueError
            Naming the offending ``data.*`` key and the accepted forms.
        """
        if self.stamp is not None and self.stamp not in STAMPS:
            raise ValueError(
                f"data.time_stamp must be one of {', '.join(STAMPS)}; "
                f"got {self.stamp!r}")
        if self.window is not None:
            _parse_window(self.window)
        if self.offset is not None:
            _parse_offset(self.offset)
        if (self.stamp is None) != (self.window is None):
            given, missing = (("data.time_stamp", "data.time_window")
                              if self.stamp is not None else
                              ("data.time_window", "data.time_stamp"))
            raise ValueError(
                f"{given} is set but {missing} is not: the stamp placement "
                "(data.time_stamp) and the window or cadence "
                "(data.time_window) are given together or not at all; "
                f"{_OPTIONS_FIX}, or remove {given}")


@dataclass(frozen=True)
class FileAxis:
    """The undecoded time axis of one file of one variable.

    Attributes
    ----------
    values : np.ndarray
        ``(T,)`` numeric stamps as stored (``T`` may be 0).
    units : str
        CF ``"<unit> since <date>"``.
    calendar : str
        CF calendar name (an empty string means ``standard``).
    bounds : np.ndarray or None
        ``(T, 2)`` numeric window bounds in ``units``, or None.
    schedule : str or None
        The raw writer's ``output_schedule`` global attribute, or None.
    path : str
        The file, for messages.
    """

    values: np.ndarray
    units: str
    calendar: str
    bounds: Optional[np.ndarray]
    schedule: Optional[str]
    path: str


@dataclass(frozen=True)
class ResolvedAxis:
    """What each record of one variable represents.

    Attributes
    ----------
    instants : np.ndarray
        ``(T,)`` object array of ``cftime`` datetimes in ONE calendar: the
        window centre for means, the stamp for snapshots.
    windows : np.ndarray or None
        ``(T, 2)`` object array ``[start, end]`` of each averaging window, or
        None for snapshots (``data.time_stamp: instant``).
    source : str
        ``"options"``, ``"bounds"`` or ``"schedule"``.
    """

    instants: np.ndarray
    windows: Optional[np.ndarray]
    source: str

    @property
    def calendar(self) -> str:
        """The calendar of the instants (``""`` for an empty axis)."""
        return str(self.instants[0].calendar) if len(self.instants) else ""


# ----------------------------------------------------------------------------
# parsing
# ----------------------------------------------------------------------------

def _parse_duration(text: str, *, key: str, signed: bool) -> timedelta:
    m = _DURATION_RE.match(text) if isinstance(text, str) else None
    if m is None or (m.group(1) and not signed):
        form = "'<±number> <s|min|h|d>'" if signed else "'<number> <s|min|h|d>'"
        raise ValueError(f"{key} must be a duration {form} (e.g. "
                         f"{'+6 h' if signed else '5 d'!r}); got {text!r}")
    sign = -1.0 if m.group(1) == "-" else 1.0
    seconds = sign * float(m.group(2)) * _DURATION_SECONDS[m.group(3)]
    return timedelta(seconds=seconds)


def _parse_window(text: str) -> Tuple[str, object]:
    """``("calendar", "month"|"year"|"day")`` or ``("fixed", timedelta)``."""
    if isinstance(text, str) and text.strip() in CALENDAR_WINDOWS:
        return "calendar", text.strip().split("_", 1)[1]
    try:
        d = _parse_duration(text, key="data.time_window", signed=False)
    except ValueError:
        raise ValueError(
            f"data.time_window must be one of {', '.join(CALENDAR_WINDOWS)} "
            f"or a duration '<number> <s|min|h|d>' (e.g. '5 d'); got "
            f"{text!r}") from None
    if d <= timedelta(0):
        raise ValueError(f"data.time_window must be a positive duration; "
                         f"got {text!r}")
    return "fixed", d


def _parse_offset(text: str) -> timedelta:
    return _parse_duration(text, key="data.time_offset", signed=True)


def parse_output_schedule(text: str) -> Tuple[str, int, int]:
    """Parse the raw writer's ``output_schedule`` attribute.

    Parameters
    ----------
    text : str
        ``"unit: <y|m|d|s> first: <n> rate: <n>"``.

    Returns
    -------
    (unit, first, rate) : tuple of (str, int, int)
        ``first >= 0`` and ``rate >= 1``.  Unit ``s`` parses; resolving a
        time axis from it is refused (:func:`resolve_axis`).

    Raises
    ------
    ValueError
        When the text does not have that form.
    """
    m = _SCHEDULE_RE.match(text) if isinstance(text, str) else None
    if m is None:
        raise ValueError(f"output_schedule {text!r} is not of the form "
                         "'unit: <y|m|d|s> first: <n> rate: <n>'")
    unit, first_s, rate_s = m.groups()
    if unit not in _SCHEDULE_UNITS:
        raise ValueError(f"output_schedule {text!r}: unknown unit {unit!r} "
                         "(y, m, d or s)")
    if not (first_s.isdigit() and rate_s.isdigit()):
        raise ValueError(f"output_schedule {text!r}: first and rate must be "
                         "non-negative integers")
    first, rate = int(first_s), int(rate_s)
    if rate < 1:
        raise ValueError(f"output_schedule {text!r}: rate must be >= 1")
    return unit, first, rate


# ----------------------------------------------------------------------------
# small time helpers (cftime, any calendar)
# ----------------------------------------------------------------------------

def _iso(t) -> str:
    return t.isoformat()


def _same(a, b) -> bool:
    return abs(a - b) <= TOLERANCE


def _mid(a, b):
    return a + (b - a) / 2


#: the date type ``cftime.num2date`` returns per calendar: every date built
#: here has that type, so an axis never mixes types (xarray builds a
#: CFTimeIndex only from dates of one type)
_DATE_TYPES = {
    "standard": cftime.DatetimeGregorian,
    "proleptic_gregorian": cftime.DatetimeProlepticGregorian,
    "noleap": cftime.DatetimeNoLeap,
    "all_leap": cftime.DatetimeAllLeap,
    "360_day": cftime.Datetime360Day,
    "julian": cftime.DatetimeJulian,
}


def _date(calendar: str, year: int, month: int, day: int, hour: int = 0,
          minute: int = 0, second: int = 0, microsecond: int = 0,
          has_year_zero: Optional[bool] = None):
    cls = _DATE_TYPES[calendar]
    kw = {} if has_year_zero is None else {"has_year_zero": has_year_zero}
    return cls(year, month, day, hour, minute, second, microsecond, **kw)


def _make(proto, year: int, month: int, day: int):
    return _date(proto.calendar, year, month, day,
                 has_year_zero=proto.has_year_zero)


def _add_months(proto, year: int, month: int, n: int):
    y, m = divmod((month - 1) + n, 12)
    return _make(proto, year + y, m + 1, 1)


def _floor(t, period: str):
    if period == "day":
        return _make(t, t.year, t.month, t.day)
    if period == "month":
        return _make(t, t.year, t.month, 1)
    return _make(t, t.year, 1, 1)


def _next(boundary, period: str):
    if period == "day":
        return boundary + timedelta(days=1)
    if period == "month":
        return _add_months(boundary, boundary.year, boundary.month, 1)
    return _make(boundary, boundary.year + 1, 1, 1)


def _prev(boundary, period: str):
    if period == "day":
        return boundary - timedelta(days=1)
    if period == "month":
        return _add_months(boundary, boundary.year, boundary.month, -1)
    return _make(boundary, boundary.year - 1, 1, 1)


def _snap(t, period: str):
    """The calendar boundary ``t`` sits on (within TOLERANCE), else None."""
    f = _floor(t, period)
    if _same(t, f):
        return f
    n = _next(f, period)
    return n if _same(t, n) else None


def _canonical_calendar(name: str) -> str:
    n = (name or "standard").strip().lower()
    return _CALENDAR_ALIASES.get(n, n)


def _after_gregorian_start(t) -> bool:
    return (t.year, t.month, t.day) >= _GREGORIAN_START


def _recalendar(t, calendar: str):
    """``t`` with the same fields in ``calendar`` (standard ↔ proleptic
    after 1582-10-15 only); None when that would change the instant."""
    if t.calendar == calendar:
        return t
    if (t.calendar in _REAL_CALENDARS and calendar in _REAL_CALENDARS
            and _after_gregorian_start(t)):
        return _date(calendar, t.year, t.month, t.day, t.hour, t.minute,
                     t.second, t.microsecond)
    return None


def _describe_duration(d: timedelta) -> str:
    s = d.total_seconds()
    for unit, size in (("d", 86400.0), ("h", 3600.0), ("min", 60.0)):
        if abs(s) >= size and float(s / size).is_integer():
            return f"{s / size:+g} {unit}"
    return f"{s:+g} s"


def _objects(seq) -> np.ndarray:
    out = np.empty(len(seq), dtype=object)
    for i, v in enumerate(seq):
        out[i] = v
    return out


# ----------------------------------------------------------------------------
# decoding
# ----------------------------------------------------------------------------

@dataclass
class _Decoded:
    path: str
    calendar: str
    units: str
    raw_epoch: object          # epoch as written (before the offset)
    epoch: object              # epoch + offset
    stamps: list               # decoded stamps + offset
    bounds: Optional[list]     # [(start, end)] decoded + offset, or None
    schedule: Optional[str]


def _who(path: str, variable: Optional[str]) -> str:
    return f"{path} (variable {variable!r})" if variable else str(path)


def _numeric(arr, what: str, path: str, variable: Optional[str]) -> np.ndarray:
    if np.ma.isMaskedArray(arr) and np.ma.getmaskarray(arr).any():
        raise ValueError(f"{_who(path, variable)}: a {what} value is missing "
                         "(masked); repair the file or leave it out")
    a = np.asarray(np.ma.getdata(arr) if np.ma.isMaskedArray(arr) else arr)
    if a.dtype.kind not in "iuf":
        raise ValueError(f"{_who(path, variable)}: the {what} values must be "
                         f"the stored numbers (dtype {a.dtype}); open the file "
                         "without decoding times")
    if a.dtype.kind == "f" and not np.isfinite(a).all():
        i = int(np.flatnonzero(~np.isfinite(a.ravel()))[0])
        raise ValueError(f"{_who(path, variable)}: {what} value number {i} is "
                         "not finite; repair the file or leave it out")
    return a


def _decode(f: FileAxis, offset: timedelta, variable: Optional[str]) -> _Decoded:
    who = _who(f.path, variable)
    units = str(f.units).strip()
    if " since " not in units.lower():
        raise ValueError(f"{who}: time units {units!r} are not "
                         "'<unit> since <date>'; the file's stamps cannot be "
                         "decoded — repair the units or leave the file out")
    cal = _canonical_calendar(f.calendar)
    vals = _numeric(f.values, "time", f.path, variable)
    if vals.ndim != 1:
        raise ValueError(f"{who}: the time axis must be one-dimensional "
                         f"(shape {vals.shape})")
    bnds = None
    if f.bounds is not None:
        bnds = _numeric(f.bounds, "time bounds", f.path, variable)
        if bnds.shape != (vals.size, 2):
            raise ValueError(f"{who}: time bounds of shape {bnds.shape} for "
                             f"{vals.size} records (expected ({vals.size}, 2))")
    try:
        kw = dict(calendar=cal, only_use_cftime_datetimes=True)
        raw_epoch = cftime.num2date(0, units, **kw)
        stamps = list(np.atleast_1d(cftime.num2date(vals, units, **kw))) \
            if vals.size else []
        dec_b = (np.asarray(cftime.num2date(bnds, units, **kw)).reshape(-1, 2)
                 if bnds is not None and bnds.size else None)
    except Exception as exc:  # cftime: unknown calendar, invalid date, ...
        raise ValueError(f"{who}: cannot decode the time axis (units "
                         f"{units!r}, calendar {f.calendar!r}): {exc}") from exc
    bounds = None
    if bnds is not None:
        bounds = ([] if dec_b is None else
                  [(a + offset, b + offset) for a, b in dec_b])
    return _Decoded(path=str(f.path), calendar=cal, units=units,
                    raw_epoch=raw_epoch, epoch=raw_epoch + offset,
                    stamps=[s + offset for s in stamps], bounds=bounds,
                    schedule=f.schedule)


def _unify_calendars(decoded: List[_Decoded], variable: Optional[str]) -> None:
    """One calendar for all files of a variable (in place)."""
    cals = {d.calendar for d in decoded}
    if len(cals) == 1:
        return
    if cals <= set(_REAL_CALENDARS):
        def dates(d):
            yield d.epoch
            yield from d.stamps
            for w in d.bounds or ():
                yield from w
        if all(_after_gregorian_start(t) for d in decoded for t in dates(d)):
            target = "proleptic_gregorian"

            def conv(t):
                return _recalendar(t, target)

            for d in decoded:
                if d.calendar == target:
                    continue
                d.epoch = conv(d.epoch)
                # only its fields are read (the raw writer's 0:0:0 check)
                d.raw_epoch = conv(d.raw_epoch) or d.raw_epoch
                d.stamps = [conv(t) for t in d.stamps]
                if d.bounds is not None:
                    d.bounds = [(conv(a), conv(b)) for a, b in d.bounds]
                d.calendar = target
            return
    by_cal = {}
    for d in decoded:
        by_cal.setdefault(d.calendar, d.path)
    (c1, p1), (c2, p2) = list(by_cal.items())[:2]
    raise ValueError(
        f"variable {variable!r}: its files use different calendars — {p1} "
        f"({c1}) and {p2} ({c2}); one variable needs one calendar "
        "('standard' and 'proleptic_gregorian' count as one only on or after "
        "1582-10-15): leave out the files of the other run")


def _check_stamp_order(d: _Decoded, variable: Optional[str]) -> None:
    for a, b in zip(d.stamps[:-1], d.stamps[1:]):
        if _same(a, b):
            raise ValueError(
                f"{_who(d.path, variable)}: duplicate record — two records "
                f"are stamped {_iso(b)}; repair the file or leave it out")
        if b < a:
            raise ValueError(
                f"{_who(d.path, variable)}: records are not in time order — "
                f"the record stamped {_iso(b)} follows the one stamped "
                f"{_iso(a)}; repair the file or leave it out")


# ----------------------------------------------------------------------------
# records and the no-gap rule
# ----------------------------------------------------------------------------

@dataclass
class _Rec:
    path: str
    start: object      # window start, or None for a snapshot
    end: object        # window end, or None for a snapshot
    instant: object


def _check_tiling(recs: List[_Rec], variable: Optional[str],
                  hint: str = "", short_gap_note: str = "") -> None:
    """Consecutive windows must touch: no gap, no overlap.

    ``hint`` follows every gap's interval; ``short_gap_note`` is added to a
    gap shorter than one day.
    """
    for r, n in zip(recs[:-1], recs[1:]):
        if _same(r.end, n.start):
            continue
        files = (f"in {r.path}" if r.path == n.path
                 else f"between {r.path} and {n.path}")
        if n.start > r.end:
            note = (short_gap_note
                    if n.start - r.end < timedelta(days=1) else "")
            raise ValueError(
                f"variable {variable!r}: gap {files} — no record covers "
                f"{_iso(r.end)} to {_iso(n.start)} (between the record at "
                f"{_iso(r.instant)}, window {_iso(r.start)} – {_iso(r.end)}, "
                f"and the record at {_iso(n.instant)}, window "
                f"{_iso(n.start)} – {_iso(n.end)}){hint}.{note} Gaps are not "
                f"allowed: add the missing record or file, or leave out "
                f"{n.path} and the files after it")
        dup = " (a duplicate record)" if _same(r.instant, n.instant) else ""
        fix = ("leave one of the overlapping files out" if r.path != n.path
               else "repair the file's time axis (or check data.time_window "
                    "when it is set) or leave the file out")
        raise ValueError(
            f"variable {variable!r}: records overlap {files}{dup} — the "
            f"record at {_iso(r.instant)} (window {_iso(r.start)} – "
            f"{_iso(r.end)}, {r.path}) and the record at {_iso(n.instant)} "
            f"(window {_iso(n.start)} – {_iso(n.end)}, {n.path}); {fix}")


def _check_cadence(recs: List[_Rec], step, variable: Optional[str]) -> None:
    """Consecutive snapshots must be exactly one cadence step apart."""
    for r, n in zip(recs[:-1], recs[1:]):
        expected = step(r.instant)
        if _same(n.instant, expected):
            continue
        files = (f"in {r.path}" if r.path == n.path
                 else f"between {r.path} and {n.path}")
        if n.instant > expected:
            raise ValueError(
                f"variable {variable!r}: gap {files} — the snapshot at "
                f"{_iso(expected)} is missing (between the records at "
                f"{_iso(r.instant)} and {_iso(n.instant)}). Gaps are not "
                f"allowed: add the missing record or file, or leave out "
                f"{n.path} and the files after it")
        dup = (" (a duplicate record)" if _same(r.instant, n.instant) else "")
        raise ValueError(
            f"variable {variable!r}: records overlap {files}{dup} — the "
            f"record at {_iso(n.instant)} comes before the next snapshot "
            f"expected at {_iso(expected)} (data.time_window cadence); "
            "check data.time_window, or leave one of the overlapping files "
            "out")


def _to_axis(recs: List[_Rec], means: bool, source: str) -> ResolvedAxis:
    inst = _objects([r.instant for r in recs])
    win = None
    if means:
        win = np.empty((len(recs), 2), dtype=object)
        for i, r in enumerate(recs):
            win[i, 0], win[i, 1] = r.start, r.end
    return ResolvedAxis(instants=inst, windows=win, source=source)


# ----------------------------------------------------------------------------
# source: time bounds
# ----------------------------------------------------------------------------

def _bounds_records(decoded: List[_Decoded], variable) -> List[_Rec]:
    recs: List[_Rec] = []
    for d in decoded:
        for s, (a, b) in zip(d.stamps, d.bounds):
            if not b - a > TOLERANCE:
                raise ValueError(
                    f"{_who(d.path, variable)}: the record stamped {_iso(s)} "
                    f"has an empty or reversed window {_iso(a)} – {_iso(b)} "
                    "in its time bounds; repair the bounds, or set "
                    "data.time_stamp and data.time_window to ignore them")
            recs.append(_Rec(d.path, a, b, _mid(a, b)))
    _check_tiling(recs, variable)
    return recs


# ----------------------------------------------------------------------------
# source: the raw writer's output schedule
# ----------------------------------------------------------------------------

def _first_multiple(lowest: int, rate: int) -> int:
    lowest = max(lowest, 1)
    return -(-lowest // rate) * rate


def _events(epoch, unit: str, first: int, rate: int) -> Iterator:
    """The raw writer's output events after ``epoch``, in order (endless)."""
    if unit == "d":
        day0 = _floor(epoch, "day")
        k = _first_multiple(first, rate)
        while True:
            yield day0 + timedelta(days=k)
            k += rate
    elif unit == "m":
        k = _first_multiple(first, rate)
        while True:
            yield _add_months(epoch, epoch.year, epoch.month, k)
            k += rate
    else:
        # "y": the end of the 12th, 24th, ... month counted from the job's
        # start month, kept when the calendar year Y of that ending month
        # satisfies first <= Y and Y % rate == 0; the ending month of the
        # j-th such boundary lies in year y1 + j - 1
        start = epoch.year * 12 + (epoch.month - 1)
        y1 = (start + 11) // 12
        year = -(-max(first, y1) // rate) * rate
        while True:
            j = year - y1 + 1
            yield _add_months(epoch, epoch.year, epoch.month, 12 * j)
            year += rate


def _schedule_records(d: _Decoded, variable) -> List[_Rec]:
    who = _who(d.path, variable)
    try:
        unit, first, rate = parse_output_schedule(d.schedule)
    except ValueError as exc:
        raise ValueError(f"{who}: cannot use output_schedule {d.schedule!r} "
                         f"({exc}); {_OPTIONS_FIX}") from None
    if unit == "s":
        raise ValueError(
            f"{who}: output_schedule {d.schedule!r} counts model steps "
            "(unit: s); the step length is not in the file, so the averaging "
            f"windows cannot be established: {_OPTIONS_FIX}")
    token = d.units.lower().split(" since ", 1)[0].strip()
    e = d.raw_epoch
    if token not in _SECONDS_TOKENS or (e.hour, e.minute, e.second,
                                        e.microsecond) != (0, 0, 0, 0):
        raise ValueError(
            f"{who}: it carries output_schedule {d.schedule!r}, but its time "
            f"units {d.units!r} are not the raw writer's 'seconds since "
            "<date> 0:0:0' — the file was re-timed after the run, so the "
            f"schedule does not describe its stamps: {_OPTIONS_FIX}")
    if (d.epoch.year, d.epoch.month, d.epoch.day) != (e.year, e.month, e.day):
        raise ValueError(
            f"{who}: data.time_offset ({_describe_duration(d.epoch - e)}) "
            f"moves the job start of this raw-writer file from {_iso(e)} to "
            f"{_iso(d.epoch)}, out of its day. The raw writer stamps against "
            "the job's start DATE at 00:00, so its stamps are only ever early "
            "by the time of day at which the job started: a raw file takes an "
            "offset from 0 up to (not including) 24 h. Correct "
            "data.time_offset, or set data.time_stamp and data.time_window")
    events = _events(d.epoch, unit, first, rate)
    prev, ev = d.epoch, next(events)
    recs: List[_Rec] = []
    for s in d.stamps:
        if not _same(s, ev) and s > ev:
            probe = ev
            while probe < s - TOLERANCE:
                probe = next(events)
            if _same(probe, s):
                raise ValueError(
                    f"{who}: missing record — output_schedule "
                    f"{d.schedule!r} writes a record for the window "
                    f"{_iso(prev)} – {_iso(ev)} (stamped {_iso(ev)}), but "
                    f"the file's next record is stamped {_iso(s)}. Gaps are "
                    "not allowed: use complete files, or leave this file and "
                    "the files after it out")
        if not _same(s, ev):
            ahead = ev - s
            fix = ""
            if timedelta(0) < ahead < timedelta(days=1):
                fix = (" If the job started at a time of day other than "
                       f"00:00, its stamps sit early by that time: set "
                       f"data.time_offset (here {_describe_duration(ahead)!r} "
                       "moves this stamp onto the event).")
            raise ValueError(
                f"{who}: the record stamped {_iso(s)} is not an output event "
                f"of output_schedule {d.schedule!r} counted from the job "
                f"start {_iso(d.epoch)}: the next event is {_iso(ev)}.{fix} "
                f"Otherwise the file was re-timed after the run: "
                f"{_OPTIONS_FIX}")
        recs.append(_Rec(d.path, prev, ev, _mid(prev, ev)))
        prev, ev = ev, next(events)
    return recs


def _schedule_all(decoded: List[_Decoded], variable) -> List[_Rec]:
    recs: List[_Rec] = []
    for d in decoded:
        recs.extend(_schedule_records(d, variable))
    _check_tiling(recs, variable,
                  hint=(": a job segment must start at the previous one's "
                        "last output event"),
                  short_gap_note=(
                      " A gap shorter than a day between job segments: a job "
                      "that started at a time of day other than 00:00 also "
                      "ends at that time, and the hours after its last output "
                      "event are never written, so a run of several segments "
                      "started that way cannot be read."))
    return recs


# ----------------------------------------------------------------------------
# source: explicit options
# ----------------------------------------------------------------------------

def _options_records(decoded: List[_Decoded], opts: TimeOptions,
                     variable) -> Tuple[List[_Rec], bool]:
    kind, val = _parse_window(opts.window)
    stamp = opts.stamp
    recs: List[_Rec] = []
    for d in decoded:
        who = _who(d.path, variable)
        for s in d.stamps:
            if kind == "calendar":
                if stamp == "centre":
                    p0 = _floor(s, val)
                    p1 = _next(p0, val)
                    c = _mid(p0, p1)
                    if not _same(c, s):
                        raise ValueError(
                            f"{who}: the record stamped {_iso(s)} is not the "
                            f"centre of its calendar_{val} ({_iso(c)}), as "
                            "data.time_stamp: centre requires; set "
                            "data.time_offset to move the stamps, or choose "
                            "another data.time_stamp/data.time_window")
                    recs.append(_Rec(d.path, p0, p1, c))
                    continue
                b = _snap(s, val)
                if b is None:
                    raise ValueError(
                        f"{who}: the record stamped {_iso(s)} does not sit on "
                        f"a calendar_{val} boundary, as data.time_stamp: "
                        f"{stamp} with data.time_window: calendar_{val} "
                        "requires; set data.time_offset to move the stamps "
                        "onto the boundaries (e.g. '+6 h'), or choose another "
                        "data.time_stamp/data.time_window")
                if stamp == "end":
                    p0 = _prev(b, val)
                    recs.append(_Rec(d.path, p0, b, _mid(p0, b)))
                elif stamp == "start":
                    p1 = _next(b, val)
                    recs.append(_Rec(d.path, b, p1, _mid(b, p1)))
                else:
                    recs.append(_Rec(d.path, None, None, b))
            else:
                if stamp == "end":
                    recs.append(_Rec(d.path, s - val, s, s - val / 2))
                elif stamp == "start":
                    recs.append(_Rec(d.path, s, s + val, s + val / 2))
                elif stamp == "centre":
                    recs.append(_Rec(d.path, s - val / 2, s + val / 2, s))
                else:
                    recs.append(_Rec(d.path, None, None, s))
    if stamp == "instant":
        if kind == "calendar":
            step = lambda t: _next(t, val)  # noqa: E731
        else:
            step = lambda t: t + val  # noqa: E731
        _check_cadence(recs, step, variable)
        return recs, False
    _check_tiling(recs, variable)
    return recs, True


# ----------------------------------------------------------------------------
# resolution
# ----------------------------------------------------------------------------

def _file_source(d: _Decoded) -> Optional[str]:
    if d.bounds is not None:
        return "bounds"
    if d.schedule is not None:
        return "schedule"
    return None


def _resolve_auto(decoded: List[_Decoded], variable) -> ResolvedAxis:
    srcs = [(d, _file_source(d)) for d in decoded]
    for d, src in srcs:
        if src is None:
            raise ValueError(
                f"{_who(d.path, variable)}: the file has neither time bounds "
                "nor an output_schedule attribute, so what time its records "
                f"(the first stamped {_iso(d.stamps[0])}) represent cannot be "
                f"established: {_OPTIONS_FIX}")
    kinds = {}
    for d, src in srcs:
        kinds.setdefault(src, d.path)
    if len(kinds) > 1:
        raise ValueError(
            f"variable {variable!r}: {kinds['bounds']} carries time bounds "
            f"but {kinds['schedule']} only an output_schedule; one variable "
            f"needs one source of time: {_OPTIONS_FIX}, or leave out the "
            "files of the other kind")
    if "bounds" in kinds:
        return _to_axis(_bounds_records(decoded, variable), True, "bounds")
    return _to_axis(_schedule_all(decoded, variable), True, "schedule")


def _describe_record(ax: ResolvedAxis, i: int) -> str:
    if i >= len(ax.instants):
        return "no record"
    s = _iso(ax.instants[i])
    if ax.windows is not None:
        s += f" (window {_iso(ax.windows[i, 0])} – {_iso(ax.windows[i, 1])})"
    else:
        s += " (a snapshot)"
    return s


def _warn_override(decoded: List[_Decoded], result: ResolvedAxis,
                   opts: TimeOptions, variable) -> None:
    carried = [d for d in decoded if _file_source(d) is not None]
    if not carried:
        return
    names = []
    if any(d.bounds is not None for d in carried):
        names.append("time bounds")
    if any(d.bounds is None and d.schedule is not None for d in carried):
        names.append("output_schedule")
    what = " and the ".join(names)
    try:
        auto = _resolve_auto(decoded, variable)
    except ValueError as exc:
        detail = f"the files' own {what} could not be used anyway ({exc})"
    else:
        i = first_difference(result, auto)
        label = "time bounds" if auto.source == "bounds" else "output_schedule"
        if i is None:
            detail = "the two agree on every record"
        else:
            detail = (f"first record where they disagree: "
                      f"{_describe_record(result, i)} from the options "
                      f"against {_describe_record(auto, i)} from the {label}")
    files = ", ".join(sorted({d.path for d in carried}))
    warnings.warn(
        f"variable {variable!r}: data.time_stamp={opts.stamp!r} and "
        f"data.time_window={opts.window!r} override the {what} of {files}; "
        f"{detail}", UserWarning, stacklevel=3)


def resolve_axis(files: Sequence[FileAxis], opts: TimeOptions, *,
                 variable: Optional[str] = None) -> ResolvedAxis:
    """Resolve what every record of one variable represents.

    Parameters
    ----------
    files : sequence of FileAxis
        All files of ONE variable, in any order (sorted here by their first
        stamp).  Files without records contribute none.
    opts : TimeOptions
        The user's options; validated here.
    variable : str, optional
        The variable's name, for messages (callers should pass it).

    Returns
    -------
    ResolvedAxis
        Instants (window centres for means, stamps for snapshots) in one
        calendar, windows (None for snapshots) and the source used.

    Raises
    ------
    ValueError
        Invalid options; undecodable or unordered stamps; files of one
        variable in different calendars; no source of time; a stamp that
        does not fit the schedule or the chosen convention; a gap, an
        overlap or a duplicate record.

    Warns
    -----
    UserWarning
        When explicit options override time bounds or an output_schedule
        that the files carry.
    """
    opts.validate()
    if not files:
        raise ValueError(f"variable {variable!r}: no files to resolve a time "
                         "axis from")
    offset = (_parse_offset(opts.offset) if opts.offset is not None
              else timedelta(0))
    decoded = [_decode(f, offset, variable) for f in files]
    _unify_calendars(decoded, variable)
    for d in decoded:
        _check_stamp_order(d, variable)
    decoded = sorted((d for d in decoded if d.stamps),
                     key=lambda d: d.stamps[0])
    if not decoded:
        raise ValueError(
            f"variable {variable!r}: no record in "
            f"{', '.join(str(f.path) for f in files)}; leave the empty files "
            "out or add files with records")
    if opts.explicit:
        recs, means = _options_records(decoded, opts, variable)
        result = _to_axis(recs, means, "options")
        _warn_override(decoded, result, opts, variable)
        return result
    return _resolve_auto(decoded, variable)


def first_difference(a: ResolvedAxis, b: ResolvedAxis) -> Optional[int]:
    """Index of the first record where two resolved axes differ.

    Records are compared by value: instants and window bounds within
    :data:`TOLERANCE`, ``standard`` and ``proleptic_gregorian`` as one
    calendar on or after 1582-10-15.  A mean (with a window) and a snapshot
    (without) always differ; so do instants in calendars that are not
    interchangeable.

    Parameters
    ----------
    a, b : ResolvedAxis
        The two axes.

    Returns
    -------
    int or None
        The first differing record; ``min(len(a), len(b))`` when one axis
        is a prefix of the other; None when they are identical.
    """
    n = min(len(a.instants), len(b.instants))
    for i in range(n):
        ta = a.instants[i]
        tb = _recalendar(b.instants[i], ta.calendar)
        if tb is None or not _same(ta, tb):
            return i
        if (a.windows is None) != (b.windows is None):
            return i
        if a.windows is not None:
            for j in (0, 1):
                wb = _recalendar(b.windows[i, j], ta.calendar)
                if wb is None or not _same(a.windows[i, j], wb):
                    return i
    return None if len(a.instants) == len(b.instants) else n
