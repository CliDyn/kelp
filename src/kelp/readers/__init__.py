"""Reader registry for model-specific mesh and velocity I/O.

Each reader module must provide:
    load_mesh(config) -> Mesh
    open_dataset(path) -> xr.Dataset
    read_velocities(ds, timestep, mesh, mode) -> VelocityField

A reader may also provide the optional staticmethod
    read_scalar_fields(ds, timestep, mesh, varnames, mode_cfg=None)
        -> {name: (N,) | (K-1, N) | (K, N) float64}
returning node-based scalar fields (vertical dim trimmed to ``mesh.sigma``)
for trajectory sampling; ``read_velocities`` attaches them to
``VelocityField.scalars`` when given a ``sample_vars`` list.

A reader may also provide the optional hook
    prepare(mesh, ds[, *, mode_cfg=None]) -> None
that checks a dataset against the mesh; every entry of a run (and ``kelp
info``) calls it through :func:`_prepare_dataset`, which passes the run's
mode to a hook whose signature names ``mode_cfg``.  A reader that resolves
what time each record represents (``_READER_RESOLVES_TIME``) takes the
``data.time_*`` stamp options as ``open_dataset`` keywords
(:func:`_time_option_kwargs`); for every other reader
:func:`apply_time_options` resolves the opened dataset's ``time`` coordinate
from them (:func:`_open_run_dataset` applies it once, at the open).  A reader
that opens only the variables a run consumes (``_READER_OPENS_CONSUMED``)
takes the beaching dry mask (:func:`_dry_mask_kwargs`).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import functools
import inspect
import warnings

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

    from .fesom14._timeaxis import TimeOptions

_READERS = {}


def register(name: str):
    """Decorator to register a reader module by model name."""
    def decorator(cls):
        _READERS[name] = cls
        return cls
    return decorator


#: reader modules imported on first use by ``get_reader``
_KNOWN_READERS = {"fesomc": ".fesomc", "fesom2": ".fesom2", "nemo": ".nemo",
                  "healpix": ".healpix", "fesom14": ".fesom14"}


def get_reader(name: str):
    """Get a registered reader by model name.

    Parameters
    ----------
    name : str
        Model name, e.g. "fesomc" or "fesom2".

    Returns
    -------
    Reader class with load_mesh, open_dataset, read_velocities methods.

    Raises
    ------
    KeyError
        If no reader is registered for the given name.
    """
    if name not in _READERS:
        # Auto-import known reader modules
        if name in _KNOWN_READERS:
            import importlib
            importlib.import_module(_KNOWN_READERS[name], package=__name__)
    if name not in _READERS:
        available = ", ".join(sorted(set(_READERS) | set(_KNOWN_READERS)))
        raise KeyError(f"No reader registered for '{name}'. Available: {available}")
    return _READERS[name]


def _prepare_dataset(reader, mesh, ds, mode=None) -> None:
    """Run the reader's optional ``prepare(mesh, ds[, *, mode_cfg])`` hook.

    A hook whose signature has a parameter named ``mode_cfg`` (the
    convention of ``load_mesh``) gets ``mode`` — the run's ``ModeConfig``
    — by keyword; every other hook is called with ``(mesh, ds)``, so
    two-argument hooks work as before.  A call without a mode passes
    ``mode_cfg=None``, with which such a hook checks nothing that needs
    the mode (the HEALPix hook checks u, v and w of a 3-D run).

    The hook runs at every call: the engine keeps no record of what a mesh
    was prepared for (no dataset token, no level table, bed or reader
    stamp) and writes nothing into ``ds.attrs`` or ``mesh.metadata`` for
    it, so whatever the mesh, the dataset, the kept levels or the reader
    of this call, the hook sees exactly them.  ``run_single_realization``
    calls it at its entry, after the ``max_levels`` cut (the hook sees the
    levels in use): once per realization.  ``run_tracking`` calls it once
    more, right after ``open_dataset``, so a refusal of the hook comes
    before the sampling probes and before any output file exists; ``kelp
    info`` calls it once, after ``open_dataset``, as a run does.  A hook
    that raises leaves nothing behind in the engine.  Readers without the
    hook: nothing happens.  It lives next to the registry, not in
    ``tracking``, so that ``kelp info`` does not import the integrators.
    """
    prep = getattr(reader, "prepare", None)
    if prep is None:
        return
    if "mode_cfg" in inspect.signature(prep).parameters:
        prep(mesh, ds, mode_cfg=mode)
    else:
        prep(mesh, ds)


def _apply_max_levels(mesh, mode, verbose: bool = False) -> None:
    """Cut ``mesh.sigma`` to ``mode.max_levels`` interfaces from the surface.

    The uncut table is kept once in ``mesh.metadata["sigma_full"]`` and
    every cut is taken from it, so calls with different limits on one mesh
    (direct ``run_single_realization`` entries) each get their own table;
    ``max_levels`` None restores the full table of a mesh cut earlier.  A
    call that leaves the table as it is keeps the very array object, so a
    depth-point table built for it stays valid.  On a mesh never cut, None
    changes nothing and stores nothing.  It lives next to the registry, not
    in ``tracking``, so that ``kelp info`` cuts as a run does without
    importing the integrators (``_open_run_dataset``).

    Raises
    ------
    ValueError
        ``max_levels < 2`` in 3-D mode.
    """
    max_levels = getattr(mode, "max_levels", None)
    md = getattr(mesh, "metadata", None)
    if md is None:
        md = {}
    full = md.get("sigma_full")
    sig = mesh.sigma
    if full is not None and not (
            sig is full or (sig.size <= full.size
                            and np.array_equal(sig, full[:sig.size]))):
        # the table was replaced by someone else since: it is the new full
        full = None
        md.pop("sigma_full", None)
    if max_levels is None:
        if full is not None and sig.size != full.size:
            mesh.sigma = full
        return
    if full is None and sig.size == 0:
        return
    if getattr(mode, "Fdrift3D", False) and max_levels < 2:
        raise ValueError(
            f"max_levels must be >= 2 in 3D mode, got {max_levels}"
        )
    if full is None:
        full = sig
        if getattr(mesh, "metadata", None) is None:
            mesh.metadata = md
        md["sigma_full"] = full
    keep = min(int(max_levels), full.size)
    if sig.size != keep:
        mesh.sigma = full[:keep]
    if verbose:
        if getattr(mesh, "vertical_coord", "zlevel") == "sigma":
            print(f"Vertical interfaces limited to {keep} (sigma mesh)")
        else:
            print(f"Depth levels limited to {keep} "
                  f"(max depth {mesh.sigma[-1]:.1f}m)")


def _open_run_dataset(reader, mesh, config, verbose: bool = False):
    """Cut the levels and open the dataset as a run does.

    One place for ``run_tracking`` and ``kelp info``: ``mode.max_levels``
    is applied to ``mesh.sigma`` (:func:`_apply_max_levels`), then
    ``open_dataset`` gets ``data.ncinput`` with the separate v / w paths and
    only the keywords that are set — ``eta_path`` (``data.ncinput_eta``;
    only readers with a separate surface-elevation store accept it, another
    reader fails loudly naming it), the diffusion block for the FESOM2
    ``var_az`` field, the stamp options (:func:`_time_option_kwargs`) and
    the beaching dry mask (:func:`_dry_mask_kwargs`) — so the other
    readers' signatures stay untouched.  For a reader that does not resolve
    its record times itself the stamp options are then applied to the
    opened dataset (:func:`apply_time_options`, once: a second call shifts
    again); a refusal there closes the dataset before it propagates.

    Parameters
    ----------
    reader : reader class
    mesh : Mesh
        Cut in place.
    config : AppConfig
    verbose : bool
        Print the cut.

    Returns
    -------
    xr.Dataset
        The open dataset, its ``time`` resolved from the stamp options when
        they are set.

    Raises
    ------
    ValueError
        The rules of :func:`_time_option_kwargs`; a refusal of
        :func:`apply_time_options` (the dataset is closed first).
    """
    _apply_max_levels(mesh, config.mode, verbose)
    extra = {}
    if getattr(config.data, "ncinput_eta", ""):
        extra["eta_path"] = config.data.ncinput_eta
    diff = config.numerics.diffusion
    if (config.mesh.type == "fesom2" and diff.enabled
            and getattr(diff, "az_enabled", False)
            and getattr(diff, "var_az", None)):
        # the multi-year exact layout (Av.fesom.<year>.nc) is discovered
        # from it
        extra["diffusion"] = diff
    if getattr(config.data, "ncinput_u", ""):
        if config.mesh.type != "nemo":
            raise ValueError("data.ncinput_u applies to mesh.type 'nemo' only")
        extra["u_path"] = config.data.ncinput_u
    extra.update(_time_option_kwargs(config))
    extra.update(_dry_mask_kwargs(config))
    ds = reader.open_dataset(config.data.ncinput,
                             v_path=config.data.ncinput_v or None,
                             w_path=config.data.ncinput_w or None,
                             mode_cfg=config.mode, **extra)
    try:
        return apply_time_options(ds, config.data, config.mesh.type)
    except Exception:
        ds.close()
        raise


#: exact unit tokens of a numeric time axis without an epoch
_TIME_UNIT_SECONDS = {
    **{k: 86400.0 for k in ("days", "day", "d")},
    **{k: 3600.0 for k in ("hours", "hour", "h", "hr", "hrs")},
    **{k: 60.0 for k in ("minutes", "minute", "min", "mins")},
    **{k: 1.0 for k in ("seconds", "second", "s", "sec", "secs")},
    **{k: 1e-3 for k in ("milliseconds", "millisecond", "ms", "msec", "msecs")},
    **{k: 1e-6 for k in ("microseconds", "microsecond", "us", "usec", "usecs")},
}


def extract_time_info(ds) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """Extract time coordinate and compute file dt from a dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Open velocity dataset with a ``time`` dimension.

    Returns
    -------
    (times, file_dt)
        ``times`` — 1-D array of seconds since first timestep (float64),
        or None if the time coordinate cannot be interpreted.
        ``file_dt`` — time step between consecutive records in seconds,
        or None if there are fewer than 2 timesteps or times are irregular.
    """
    if "time" not in ds.dims:
        return None, None

    time_var = ds["time"]
    n = ds.sizes["time"]

    # Skip auto-generated integer indices (0, 1, 2, ...) — not real time —
    # unless the coordinate says it is time ("<unit> since <epoch>"): an
    # undecoded integer axis with physical units is a real axis (review
    # 2026-09-22).
    vals = time_var.values
    units_attr = str(time_var.attrs.get("units", "")).strip().lower()
    # physical: a CF epoch or an exact unit token ("h", "d", "ms", ... —
    # the old substring test missed the short ones, review round 2)
    has_time_units = (" since " in units_attr
                      or units_attr in _TIME_UNIT_SECONDS)
    if np.issubdtype(vals.dtype, np.integer) and not has_time_units:
        expected_range = np.arange(n, dtype=vals.dtype)
        if np.array_equal(vals, expected_range):
            return None, None

    # Try to get numeric seconds from the coordinate
    try:
        if np.issubdtype(vals.dtype, np.datetime64):
            # numpy datetime64: float seconds from the native resolution
            # (casting to timedelta64[s] truncated sub-second steps)
            secs = ((vals - vals[0]) / np.timedelta64(1, "s")).astype(
                np.float64)
        elif vals.dtype == object:
            # cftime (any calendar) or Python datetimes: the calendar-aware
            # difference is a datetime.timedelta
            secs = np.array([(v - vals[0]).total_seconds() for v in vals],
                            dtype=np.float64)
        elif np.issubdtype(vals.dtype, np.number):
            # Numeric (undecoded) axis: CF "<unit> since <epoch>" goes through
            # xarray's CF decoder (calendar-aware); a bare unit is scaled by an
            # exact token match — the old substring test read "milliseconds"
            # as seconds (review 2026-09-23); an unknown unit is not guessed.
            units = str(time_var.attrs.get("units", "")).strip()
            if " since " in units.lower():
                import xarray as _xr
                dec = _xr.coding.times.decode_cf_datetime(
                    np.asarray(vals), units,
                    calendar=time_var.attrs.get("calendar", "standard"))
                dec = np.asarray(dec)
                if np.issubdtype(dec.dtype, np.datetime64):
                    secs = ((dec - dec[0]) / np.timedelta64(1, "s")).astype(np.float64)
                else:
                    secs = np.array([(v - dec[0]).total_seconds() for v in dec],
                                    dtype=np.float64)
            else:
                scale = _TIME_UNIT_SECONDS.get(units.lower(), None) if units else 1.0
                if scale is None:
                    return None, None
                secs = np.asarray(vals, dtype=np.float64) * scale
                secs = secs - secs[0]
        else:
            return None, None
    except Exception:
        return None, None

    if n < 2:
        return secs, None

    diffs = np.diff(secs)
    dt0 = diffs[0]
    if dt0 <= 0:
        return secs, None

    # Check regularity (allow 0.1% tolerance)
    if np.allclose(diffs, dt0, rtol=1e-3):
        return secs, float(dt0)

    # Irregular spacing — return median as best guess but flag via None
    return secs, None


#: ``time_origin`` attribute of an output whose forcing has no calendar epoch
NO_EPOCH_NOTE = ("first record of the forcing time axis "
                 "(the forcing has no calendar epoch)")


def _render_fields(v) -> str:
    """``YYYY-MM-DD hh:mm:ss[.ffffff]`` from an object with datetime fields
    (cftime, datetime); trailing zeros of the fraction dropped."""
    s = (f"{v.year:04d}-{v.month:02d}-{v.day:02d} "
         f"{v.hour:02d}:{v.minute:02d}:{v.second:02d}")
    us = int(getattr(v, "microsecond", 0) or 0)
    if us:
        s += "." + f"{us:06d}".rstrip("0")
    return s


def _render_datetime64(t) -> str:
    """``YYYY-MM-DD hh:mm:ss[.f]`` of a datetime64 at its own resolution
    (calendar units coarser than a day are rendered at seconds)."""
    t = np.datetime64(t)
    if np.datetime_data(t.dtype)[0] in ("Y", "M", "W"):
        t = t.astype("datetime64[s]")
    s = np.datetime_as_string(t)
    date, _, clock = s.partition("T")
    if not clock:
        return f"{date} 00:00:00"
    hms, _, frac = clock.partition(".")
    hms = (hms + ":00:00")[:8]          # 'hh' / 'hh:mm' at coarse units
    frac = frac.rstrip("0")
    return f"{date} {hms}" + (f".{frac}" if frac else "")


def time_axis_epoch(ds) -> Optional[Tuple[str, str]]:
    """CF ``units`` and ``calendar`` for a time measured in seconds from the
    dataset's first RAW record — the origin :func:`extract_time_info`
    measures from — or ``None`` when the axis carries no calendar epoch
    (bare unit, index axis, no axis).  Format-2 trajectory outputs stamp
    ``time`` with it.

    Returns
    -------
    (units, calendar) or None
        ``units = "seconds since YYYY-MM-DD hh:mm:ss[.f]"``; ``calendar`` is
        the source's (``encoding["calendar"]`` when xarray kept it, the
        cftime object's calendar, ``proleptic_gregorian`` for datetime64,
        ``standard`` for Python datetimes, or the numeric axis's attribute).
    """
    if "time" not in ds.dims or ds.sizes["time"] < 1:
        return None
    tv = ds["time"]
    vals = tv.values
    enc_cal = (getattr(tv, "encoding", None) or {}).get("calendar")
    v0 = vals[0]
    if np.issubdtype(vals.dtype, np.datetime64):
        if np.isnat(v0):
            return None
        return (f"seconds since {_render_datetime64(v0)}",
                str(enc_cal or "proleptic_gregorian"))
    if vals.dtype == object:
        if not hasattr(v0, "year"):
            return None
        cal = getattr(v0, "calendar", None) or enc_cal or "standard"
        return f"seconds since {_render_fields(v0)}", str(cal)
    if np.issubdtype(vals.dtype, np.number):
        units = str(tv.attrs.get("units", "")).strip()
        if " since " not in units.lower():
            return None
        cal = str(tv.attrs.get("calendar", "standard"))
        # an axis xarray cannot decode (year 0 in a standard calendar, a bad
        # date, an unknown calendar) has no usable epoch — the output then
        # says so instead of crashing a run whose writer is already open
        try:
            import xarray as _xr
            dec = np.asarray(_xr.coding.times.decode_cf_datetime(
                np.asarray(vals[:1]), units, calendar=cal))
            d0 = dec[0]
            if np.issubdtype(dec.dtype, np.datetime64):
                if np.isnat(d0):
                    return None
                return f"seconds since {_render_datetime64(d0)}", cal
            return f"seconds since {_render_fields(d0)}", cal
        except Exception:
            return None
    return None


#: ``data.*`` keys that say what each record's time stamp means; the mesh
#: types listed in ``_READER_RESOLVES_TIME`` take them as ``open_dataset``
#: keywords, for any other :func:`apply_time_options` resolves the opened
#: dataset's axis from them
_TIME_OPTION_KEYS = ("time_stamp", "time_window", "time_offset")

#: name of the dataset coordinate ``(time, TIME_WINDOW_EDGE)`` that holds each
#: record's averaging window ``[start, end]``, attached where the record times
#: are resolved (:func:`apply_time_options`, the FESOM 1.4 reader) and read by
#: :func:`exact_cycle`; private to KELP, so that a store's own ``time_bnds``
#: is never mistaken for it
TIME_WINDOWS = "kelp_time_windows"

#: the edge dimension of ``TIME_WINDOWS`` (length 2: start, end); private to
#: KELP, so that a CF ``nv`` dimension of another size never collides with it
TIME_WINDOW_EDGE = "kelp_time_window_edge"

#: mesh types whose reader resolves what time each record represents (the
#: centre of its averaging window for means, the stamp for snapshots): no
#: midpoint shift, no ``data.time_averaging``, the stamp options above
_READER_RESOLVES_TIME = ("fesom14",)

#: the output's ``time_averaging`` attribute for those readers
RESOLVED_BY_READER = "resolved_by_reader"

#: the output's ``time_averaging`` attribute for another reader whose record
#: times were resolved from the stamp options (:func:`apply_time_options`)
RESOLVED_FROM_OPTIONS = "resolved_from_options"

#: windows of unequal length: means over them cycle with the sum of their
#: windows (:func:`exact_cycle`); snapshots at that cadence carry no window,
#: so the period after which they repeat is not known and ``cycle`` is
#: refused for them (:func:`_check_cycle_window`)
_UNEQUAL_WINDOWS = ("calendar_month", "calendar_year")

#: what resolved the record times, by ``ds.attrs["fesom14_time_source"]``
_TIME_SOURCE_TEXT = {
    "bounds": "the files' time bounds",
    "schedule": "the files' output_schedule",
    "options": "data.time_stamp and data.time_window",
}


def _time_option_kwargs(config) -> Dict[str, str]:
    """The stamp options set in ``config.data``, as ``open_dataset`` kwargs,
    checked before the mesh or any data is read.

    Only a reader that resolves its record times takes them (``mesh.type:
    fesom14``); it also takes the place of ``data.time_averaging``.  For
    any other mesh type the options are validated here
    (:func:`_check_time_options`) and no keyword is returned:
    :func:`_open_run_dataset` applies them to the opened dataset
    (:func:`apply_time_options`).  The cycle rule of snapshots
    (:func:`_check_cycle_window`) is checked for every mesh type.

    Parameters
    ----------
    config : AppConfig

    Returns
    -------
    dict
        For a mesh type in ``_READER_RESOLVES_TIME``: ``{key: value}`` for
        each of ``time_stamp``, ``time_window`` and ``time_offset`` that is
        set (empty when none is).  For any other: empty.

    Raises
    ------
    ValueError
        For a reader that resolves its record times: ``data.time_averaging``
        is set; the message points to the stamp options.  For any other: the
        rules of :func:`_check_time_options` (an invalid key,
        ``data.time_averaging`` next to a key, ``data.time_offset`` without
        ``data.time_stamp`` and ``data.time_window``).  For every mesh type:
        the refusal of ``time.on_exhausted: cycle`` for snapshots
        (``data.time_stamp: instant``) with a calendar month or year window
        (:func:`_check_cycle_window`).
    """
    data = config.data
    mesh_type = config.mesh.type
    if mesh_type not in _READER_RESOLVES_TIME:
        _check_time_options(data, mesh_type)
        _check_cycle_window(config)
        return {}
    kw = _set_time_options(data)
    if getattr(data, "time_averaging", None) is not None:
        raise ValueError(
            f"data.time_averaging = {data.time_averaging!r} is set, but the "
            f"{mesh_type} reader resolves what time each record represents "
            "itself (the centre of its averaging window for means, from the "
            "files' time bounds or output schedule), and nothing shifts its "
            "time axis again. Remove data.time_averaging; when the files do "
            "not say what their stamps are, set data.time_stamp (start | "
            "centre | end | instant) and data.time_window (the averaging "
            "window or the snapshot cadence), and data.time_offset for "
            "stamps that sit early or late")
    _check_cycle_window(config)
    return kw


#: mesh types whose reader opens only the variables a run consumes: the
#: beaching dry mask is named to its ``open_dataset``
_READER_OPENS_CONSUMED = ("fesom14",)


def _dry_mask_kwargs(config) -> Dict[str, str]:
    """``beaching.dry_mask_var`` as an ``open_dataset`` kwarg, for a reader
    that opens only the consumed variables (``mesh.type: fesom14``).

    Parameters
    ----------
    config : AppConfig

    Returns
    -------
    dict
        ``{"dry_mask_var": name}`` when the mesh type is in
        ``_READER_OPENS_CONSUMED``, beaching is enabled and
        ``beaching.dry_mask_var`` is set; else empty, so the other readers'
        ``open_dataset`` calls stay exactly as they are.
    """
    if config.mesh.type not in _READER_OPENS_CONSUMED:
        return {}
    beach = getattr(config.numerics, "beaching", None)
    name = (getattr(beach, "dry_mask_var", None)
            if beach is not None and beach.enabled else None)
    return {"dry_mask_var": name} if name else {}


def _describe_record_times(ds, data_cfg=None) -> Optional[str]:
    """One summary line on resolved record times, or None.

    Parameters
    ----------
    ds : xarray.Dataset
        A dataset whose reader resolved its record times (it carries
        ``attrs["fesom14_time_source"]``), or one whose time axis
        :func:`apply_time_options` resolved from the stamp options.
    data_cfg : DataConfig, optional
        ``config.data``: ``time_offset`` is named when set; for a dataset
        without ``attrs["fesom14_time_source"]``, ``time_stamp`` and
        ``time_window`` (both set) give the line, else None.

    Returns
    -------
    str or None
        What resolved the times and the first and last instants.
    """
    offset = getattr(data_cfg, "time_offset", None)
    source = getattr(ds, "attrs", {}).get("fesom14_time_source")
    if source is not None:
        text = (f"resolved by the reader from "
                f"{_TIME_SOURCE_TEXT.get(source, str(source))}")
        if offset is not None:
            text += f", data.time_offset {offset} applied"
    else:
        stamp = getattr(data_cfg, "time_stamp", None)
        window = getattr(data_cfg, "time_window", None)
        if stamp is None or window is None:
            return None
        text = (f"resolved from data.time_stamp={stamp}, "
                f"data.time_window={window}")
        if offset is not None:
            text += f", data.time_offset={offset}"
    t = ds["time"].values
    if len(t):
        text += f"; first {t[0]}, last {t[-1]}"
    return text


def _set_time_options(data_cfg) -> Dict[str, str]:
    """``{key: value}`` for each stamp option set in ``data_cfg``."""
    return {k: getattr(data_cfg, k) for k in _TIME_OPTION_KEYS
            if getattr(data_cfg, k, None) is not None}


def _keys_text(kw) -> str:
    """The set keys as ``data.<key>`` joined by ``", "``."""
    return ", ".join(f"data.{k}" for k in kw)


def _check_time_options(data_cfg,
                        mesh_type: str) -> Optional[TimeOptions]:
    """The one validation of the stamp options for a reader that does not
    resolve its record times itself.

    Parameters
    ----------
    data_cfg : DataConfig
    mesh_type : str
        ``mesh.type``.

    Returns
    -------
    TimeOptions or None
        None when no key is set or the mesh type's reader resolves the
        keys itself (``_READER_RESOLVES_TIME``); else the validated options
        with ``data.time_stamp`` and ``data.time_window`` both set.

    Raises
    ------
    ValueError
        An invalid key; ``data.time_averaging`` set next to any key (named
        first, so ``time_offset`` + ``time_averaging`` says to remove
        ``time_averaging``); ``data.time_offset`` without
        ``data.time_stamp`` and ``data.time_window``.
    """
    kw = _set_time_options(data_cfg)
    if not kw or mesh_type in _READER_RESOLVES_TIME:
        return None
    from .fesom14._timeaxis import TimeOptions
    opts = TimeOptions(stamp=kw.get("time_stamp"),
                       window=kw.get("time_window"),
                       offset=kw.get("time_offset"))
    opts.validate()
    value = getattr(data_cfg, "time_averaging", None)
    if value is not None:
        raise ValueError(
            f"data.time_averaging = {value!r} is set together with "
            f"{_keys_text(kw)}: the stamp options place every record at the "
            "time it represents, and nothing shifts the time axis again. "
            "Remove data.time_averaging")
    if not opts.explicit:
        raise ValueError(
            "data.time_offset is set without data.time_stamp and "
            f"data.time_window, but mesh.type is {mesh_type!r}: this reader "
            "has no time bounds or output schedule to resolve record times "
            "from, so an offset alone has nothing to act on. Set "
            "data.time_stamp (start | centre | end | instant) and "
            "data.time_window with it, or remove data.time_offset")
    return opts


def _check_cycle_window(config) -> None:
    """Refuse ``time.on_exhausted: cycle`` for snapshots
    (``data.time_stamp: instant``) with a calendar month or year window, for
    every mesh type (the FESOM 1.4 reader included).

    Means over calendar months or years cycle with the sum of their
    averaging windows, which the places that resolve record times attach to
    the dataset (:func:`exact_cycle`).  Snapshots carry no averaging window,
    and calendar months and years differ in length, so the period after
    which such records repeat is not known.  Fixed-duration and
    ``calendar_day`` windows cycle exactly with the legacy rule.

    Parameters
    ----------
    config : AppConfig

    Raises
    ------
    ValueError
        Naming ``time.on_exhausted: stop`` as the alternative.
    """
    if getattr(config.time, "on_exhausted", None) != "cycle":
        return None
    window = getattr(config.data, "time_window", None)
    if window is None or window.strip() not in _UNEQUAL_WINDOWS:
        return None
    stamp = getattr(config.data, "time_stamp", None)
    if stamp is None or stamp.strip() != "instant":
        return None
    raise ValueError(
        "time.on_exhausted: cycle with data.time_stamp: instant and "
        f"data.time_window: {window.strip()} is refused: snapshots carry no "
        "averaging window, and calendar months and years differ in length, "
        "so the period after which such records repeat is not known. Set "
        "time.on_exhausted: stop, or use a fixed-duration or calendar_day "
        "cadence")


def _record_time_mode(config) -> Optional[str]:
    """What resolved a run's record times, for the output's
    ``time_averaging`` attribute.

    Parameters
    ----------
    config : AppConfig

    Returns
    -------
    str or None
        ``RESOLVED_BY_READER`` for a mesh type in ``_READER_RESOLVES_TIME``;
        ``RESOLVED_FROM_OPTIONS`` for another one with the stamp options
        set (validated by :func:`_check_time_options`); else None — then
        ``data.time_averaging`` decides and the caller applies the midpoint
        shift.
    """
    if config.mesh.type in _READER_RESOLVES_TIME:
        return RESOLVED_BY_READER
    if _check_time_options(config.data, config.mesh.type) is not None:
        return RESOLVED_FROM_OPTIONS
    return None


def _time_metadata(config, time_averaging) -> Dict[str, object]:
    """The output attributes that say how a run's record times were
    resolved: ``time_averaging`` and the stamp options that are set.

    One helper for ``run_tracking`` and the chain-driver template, so both
    write the same attributes; the shard concat compares them like every
    other run attribute.

    Parameters
    ----------
    config : AppConfig
    time_averaging : str or None
        What the caller writes as ``time_averaging`` for a run without the
        stamp options (``run_tracking``: ``"mean"`` / ``"instant"``, or
        ``RESOLVED_BY_READER``; the chain-driver template:
        ``config.data.time_averaging``).

    Returns
    -------
    dict
        Without any of ``data.time_stamp`` / ``time_window`` /
        ``time_offset`` set: ``{"time_averaging": time_averaging}``, the
        caller's value unchanged, so such a run writes exactly the
        attributes it wrote before.  With one set: ``time_averaging`` from
        :func:`_record_time_mode` and each set key as a string; a key not
        set is absent (never ``"None"``, and the NetCDF writer cannot store
        a ``None`` value).
    """
    keys = _set_time_options(config.data)
    if not keys:
        return {"time_averaging": time_averaging}
    meta: Dict[str, object] = {"time_averaging": _record_time_mode(config)}
    meta.update({k: str(v) for k, v in keys.items()})
    return meta


def _time_axis_label(ds, data_cfg) -> str:
    """What messages about a dataset's time axis name: the configured paths
    (``data.ncinput``, ``ncinput_v``, ``ncinput_w``, ``ncinput_eta``; the set
    ones, each once, joined by ``", "``), else the dataset's ``source``
    encoding, else ``"the dataset"``."""
    paths = []
    for key in ("ncinput", "ncinput_u", "ncinput_v", "ncinput_w", "ncinput_eta"):
        v = getattr(data_cfg, key, "") or ""
        for p in ([str(q) for q in v] if isinstance(v, (list, tuple)) else [str(v)]):
            if p and p not in paths:
                paths.append(p)
    if paths:
        return ", ".join(paths)
    source = (getattr(ds, "encoding", None) or {}).get("source")
    return str(source) if source else "the dataset"


@functools.lru_cache(maxsize=None)
def _xarray_keeps_us() -> bool:
    """Whether the xarray in use keeps a ``datetime64[us]`` coordinate at
    ``us`` (older versions store every datetime64 at ``ns``, which cannot
    hold a date beyond 2262-04-11)."""
    import xarray as xr
    t = np.array(["2000-01-01"], dtype="datetime64[us]")
    return xr.Dataset(coords={"t": t})["t"].dtype == np.dtype("datetime64[us]")


def apply_time_options(ds, data_cfg, mesh_type: str) -> xr.Dataset:
    """The dataset with its ``time`` coordinate resolved from the stamp
    options, for a reader that does not resolve its record times itself.

    The offset is added to the stamps first; each record then sits at the
    centre of its window (``start`` / ``centre`` / ``end``) or at its
    shifted stamp (``instant``) — the arithmetic and the refusals of the
    FESOM 1.4 reader's options (:func:`.fesom14._timeaxis.resolve_stamps`)
    on the dataset's one, already concatenated axis.  Nothing shifts the axis
    again: no midpoint shift follows.  Applying it to an already resolved
    dataset shifts again.

    Parameters
    ----------
    ds : xr.Dataset
        As the reader's ``open_dataset`` returned it; not mutated.
    data_cfg : DataConfig
    mesh_type : str
        ``mesh.type``.

    Returns
    -------
    xr.Dataset
        ``ds`` itself when no key is set or the mesh type's reader resolves
        the keys (``_READER_RESOLVES_TIME``).  Otherwise a new dataset whose
        ``time`` holds the resolved instants in the axis's form: cftime
        stays cftime of its calendar; datetime64 (read in
        ``proleptic_gregorian``) stays datetime64 at ``ns`` while every
        instant fits that range, else at ``us`` (never coarser, so a
        half-window centre is exact), with the input's
        ``encoding["calendar"]`` and attributes; the data variables are
        shared, the close hook is kept.  For means (every
        ``data.time_stamp`` but ``instant``) the new dataset also carries
        the coordinate ``TIME_WINDOWS`` with dimensions ``("time",
        TIME_WINDOW_EDGE)``: each record's averaging window ``[start,
        end]``, cftime of the axis's calendar on a cftime axis,
        ``datetime64[us]`` on a datetime64 axis whatever the unit of
        ``time``.  It is what :func:`exact_cycle` reads.  It is left out for
        snapshots, and when a window edge lies outside the ``ns`` range and
        the xarray in use stores datetime64 coordinates only at ``ns``.

    Raises
    ------
    ValueError
        The rules of :func:`_check_time_options`; no ``time`` dimension; an
        axis without calendar dates (numeric, an index); a stamp off the
        convention, a gap, an overlap or a broken cadence (the messages
        start with the label of :func:`_time_axis_label`).
    """
    opts = _check_time_options(data_cfg, mesh_type)
    if opts is None:
        return ds
    from datetime import datetime

    import cftime
    import xarray as xr

    from .fesom14._timeaxis import resolve_stamps

    label = _time_axis_label(ds, data_cfg)
    keys = _keys_text(_set_time_options(data_cfg))
    if "time" not in ds.dims:
        raise ValueError(
            f"{label}: {keys} are set, but the dataset has no 'time' "
            "dimension, so there are no record times to resolve. Remove the "
            "stamp options")
    tv = ds["time"]
    vals = tv.values
    if np.issubdtype(vals.dtype, np.datetime64):
        # numpy's calendar is proleptic_gregorian
        py = vals.astype("datetime64[us]").astype(object)
        stamps = [cftime.DatetimeProlepticGregorian(
            p.year, p.month, p.day, p.hour, p.minute, p.second,
            p.microsecond) for p in py]
    elif vals.dtype == object and all(hasattr(v, "calendar")
                                      for v in vals[:1]):
        stamps = list(vals)
    else:
        raise ValueError(
            f"{label}: {keys} are set, but the time axis carries no calendar "
            f"dates (dtype {vals.dtype}): the stamp options need a decoded "
            "axis (datetime64 or cftime), i.e. time units of the form "
            "'<unit> since <date>' in the files. Remove the stamp options")
    ax = resolve_stamps(stamps, opts, label=label)
    windows = (ax.windows if ax.windows is not None and len(ax.windows) > 0
               else None)
    win = None
    if vals.dtype == object:
        new = ax.instants
        win = windows
    else:
        us = np.array([np.datetime64(datetime(t.year, t.month, t.day, t.hour,
                                              t.minute, t.second,
                                              t.microsecond), "us")
                       for t in ax.instants], dtype="datetime64[us]")
        # ns only while every instant fits its range (1677-09-21 ..
        # 2262-04-11): a window centre can cross it.  numpy < 2.5 wraps on
        # overflow without an error (the round trip shows it), numpy >= 2.5
        # raises OverflowError
        try:
            fits_ns = bool((us.astype("datetime64[ns]")
                            .astype("datetime64[us]") == us).all())
        except OverflowError:
            fits_ns = False
        unit = ("ns" if np.datetime_data(vals.dtype)[0] == "ns" and fits_ns
                else "us")
        new = us.astype(f"datetime64[{unit}]")
        if windows is not None:
            # the windows always at us, whatever the unit of time: an edge
            # can lie beyond the ns range where no instant does
            win = np.array([[np.datetime64(datetime(t.year, t.month, t.day,
                                                    t.hour, t.minute, t.second,
                                                    t.microsecond), "us")
                             for t in row] for row in windows],
                           dtype="datetime64[us]")
            if not _xarray_keeps_us():
                # this xarray would store the edges at ns: leave them out
                # when one does not fit (the run keeps the legacy period)
                try:
                    edges_fit = bool((win.astype("datetime64[ns]")
                                      .astype("datetime64[us]") == win).all())
                except OverflowError:
                    edges_fit = False
                if not edges_fit:
                    win = None
    enc = {k: tv.encoding[k] for k in ("calendar",) if k in tv.encoding}
    coords = {"time": xr.Variable(("time",), new, dict(tv.attrs), enc)}
    if win is not None:
        coords[TIME_WINDOWS] = xr.Variable(("time", TIME_WINDOW_EDGE), win)
    # assign_coords drops the close hook: ONE call for both coordinates, then
    # set_close (a second assign_coords after it would lose the hook again)
    out = ds.assign_coords(coords)
    out.set_close(ds._close)
    return out


def exact_cycle(ds, file_times,
                n_time_total: int) -> Optional[Tuple[float, float]]:
    """The exact period and seam of ``time.on_exhausted: cycle``, or None
    for the legacy rule.

    Records that are means over windows of unequal length (calendar months
    or years) are not equally spaced, so the forcing repeats after the sum
    of the windows, not after the span of the records plus the last
    interval.  The windows come from the coordinate ``TIME_WINDOWS``, which
    the places that resolve record times attach (:func:`apply_time_options`,
    the FESOM 1.4 reader).

    Parameters
    ----------
    ds : xr.Dataset or None
        The run's dataset; anything without the ``TIME_WINDOWS`` coordinate
        (or without ``coords``) gives None.
    file_times : array_like of float or None
        The record times of the run's axis, seconds since its first record
        (:func:`extract_time_info`).
    n_time_total : int
        The number of records the run uses (the first ``n_time_total`` of
        the axis).

    Returns
    -------
    (period, seam) or None
        ``period``: the sum of the window lengths of the first
        ``n_time_total`` records, in seconds; ``seam``: ``period -
        (file_times[n - 1] - file_times[0])``, the time from the last
        record to the first record of the next pass, in seconds.  A tuple
        only when (1) ``ds`` carries ``TIME_WINDOWS``; (2) ``n_time_total
        >= 2``; (3) ``file_times`` is given with at least ``n_time_total``
        entries and the coordinate has at least ``n_time_total`` rows; (4)
        the window lengths of those records differ by more than 1 ms.
        Otherwise None.  The selection is by window equality, never by
        comparing the two periods: a caller keeps its legacy arithmetic
        verbatim for ``None``.
    """
    coords = getattr(ds, "coords", None)
    if coords is None or TIME_WINDOWS not in coords:
        return None
    n = int(n_time_total)
    if n < 2 or file_times is None or len(file_times) < n:
        return None
    win = np.asarray(coords[TIME_WINDOWS].values)
    if win.shape[0] < n:
        return None
    win = win[:n]
    if win.dtype == object:                      # cftime: calendar-aware
        lengths = np.array([(e - s).total_seconds() for s, e in win],
                           dtype=np.float64)
    else:                                        # datetime64, overflow-safe
        lengths = ((win[:, 1].astype("datetime64[us]")
                    - win[:, 0].astype("datetime64[us]"))
                   / np.timedelta64(1, "s")).astype(np.float64)
    if lengths.max() - lengths.min() <= 1e-3:    # equal windows: legacy rule
        return None
    period = float(lengths.sum())
    return period, period - float(file_times[n - 1] - file_times[0])


# ---------------------------------------------------------------------------
# Diffusivity (Kh) loading helper — shared between readers
# ---------------------------------------------------------------------------

# Default Kh variable names tried in order when the user does not set
# ``DiffusionConfig.var_kh_zonal`` / ``var_kh_meridional`` explicitly.
_DEFAULT_KH_X_NAMES = ("Kh_zonal", "kh_zonal", "kh_x", "Kh")
_DEFAULT_KH_Y_NAMES = ("Kh_meridional", "kh_meridional", "kh_y")

#: one-time dedup for the anisotropic-Kh-under-rotation warning
_ANISO_KH_ROT_WARNED = set()

# Dim-name hints used to detect whether a Kh array is at nodes or elements.
# These match both FESOM-C (``node``, ``nele``) and FESOM2 (``nod2``,
# ``elem``) conventions.  Any unrecognised dim falls through to a
# shape-based fallback against ``mesh.n_nodes`` / ``mesh.n_elements``.
_NODE_DIM_HINTS = frozenset(("node", "nodes", "nod2", "nod2d", "nodes_2d"))
_ELEM_DIM_HINTS = frozenset(
    ("nele", "elem", "elem2", "elements", "elements_2d")
)


# Vertical / time dims that carry no horizontal location info; stripped
# before a metadata-only spatial-dim classification.
_VERTICAL_DIM_HINTS = frozenset(
    ("time", "nz1", "nz", "sigma", "sigmam1", "sigma_lay", "depth", "level")
)


def _detect_field_layout(da, mesh) -> str:
    """Classify a mesh-located DataArray as ``"node"`` or ``"element"``.

    Used for both diffusivity (Kh) and velocity (u/v) fields.  Tries
    dim-name hints first; falls back to a shape comparison against
    ``mesh.n_nodes`` / ``mesh.n_elements`` if the dim names are not
    recognised (handles ad-hoc test datasets).  Raises if neither check
    succeeds — silently picking one would smuggle an averaging /
    nearest-projection error all the way downstream.  The shape fallback
    reads ``da.values`` and therefore assumes any ``time`` axis has already
    been ``isel``'d away; use :func:`_classify_layout_meta` when a
    metadata-only (no data load) classification is required.
    """
    dims = set(da.dims)
    if dims & _NODE_DIM_HINTS:
        return "node"
    if dims & _ELEM_DIM_HINTS:
        return "element"
    # Fallback: shape match.  After ``isel(time=...)`` the array is 1D.
    arr = da.values
    n_nodes = getattr(mesh, "n_nodes", mesh.nodes_lon.shape[0])
    n_elems = getattr(mesh, "n_elements", mesh.elem_nodes.shape[0])
    if arr.size == n_nodes and arr.size != n_elems:
        return "node"
    if arr.size == n_elems and arr.size != n_nodes:
        return "element"
    raise ValueError(
        f"Field layout cannot be inferred from variable {da.name!r}: "
        f"dims={tuple(da.dims)}, shape={arr.shape}, "
        f"n_nodes={n_nodes}, n_elems={n_elems}.  "
        "Add a recognised dim name "
        f"({sorted(_NODE_DIM_HINTS)} or {sorted(_ELEM_DIM_HINTS)})."
    )


# kept for backwards compatibility (external callers); in-repo callers use either name
_detect_kh_layout = _detect_field_layout


def _classify_layout_meta(da, mesh) -> str:
    """Classify a DataArray's horizontal layout from METADATA only.

    Unlike :func:`_detect_field_layout`, this never touches ``da.values``
    — it inspects ``da.dims`` / ``da.sizes`` only, so it is safe for a
    pre-I/O probe (``probe_uv_layout``).  After stripping known
    vertical/time dims, a remaining dim is classified by name hint, or —
    for the single leftover spatial dim — by its size against
    ``mesh.n_nodes`` / ``mesh.n_elements``.  Raises if the layout is
    ambiguous (both/neither size match) or no spatial dim remains.
    """
    spatial = [d for d in da.dims if d not in _VERTICAL_DIM_HINTS]
    for d in spatial:
        if d in _NODE_DIM_HINTS:
            return "node"
        if d in _ELEM_DIM_HINTS:
            return "element"
    n_nodes = getattr(mesh, "n_nodes", mesh.nodes_lon.shape[0])
    n_elems = getattr(mesh, "n_elements", mesh.elem_nodes.shape[0])
    if len(spatial) != 1:
        raise ValueError(
            f"u/v layout cannot be inferred from variable {da.name!r}: "
            f"dims={tuple(da.dims)} leave spatial dims {spatial!r}; "
            "expected exactly one recognised or size-matchable spatial dim."
        )
    dim = spatial[0]
    size = da.sizes[dim]      # metadata — no data load
    if size == n_nodes and size != n_elems:
        return "node"
    if size == n_elems and size != n_nodes:
        return "element"
    raise ValueError(
        f"u/v layout cannot be inferred from variable {da.name!r}: "
        f"dim {dim!r} has size {size}, ambiguous against "
        f"n_nodes={n_nodes}, n_elems={n_elems}."
    )


def _resolve_kh_var_name(ds, override: Optional[str], defaults):
    """Return the dataset variable name for one Kh component, or None.

    ``override`` is the user-set ``DiffusionConfig.var_kh_*`` field; when
    provided, that exact name *must* exist in the dataset — otherwise we
    raise loudly so the user gets a clear error rather than a silent
    no-op.  When unset, the helper tries each entry in ``defaults`` in
    order and returns the first match.  Returns ``None`` if no override
    and no default name is found.
    """
    if override is not None:
        if override in ds:
            return override
        raise KeyError(
            f"Kh variable {override!r} not found in dataset; "
            f"available variables: {sorted(ds.data_vars)}"
        )
    for name in defaults:
        if name in ds:
            return name
    return None


def _read_kh_fields(ds, timestep, mesh, diffusion):
    """Load (kh_zonal, kh_meridional, kh_layout) from a dataset.

    Returns ``None`` when ``diffusion`` is None, disabled, or no Kh
    variables are present.  Reader call sites can pass the result
    straight into ``VelocityField(kh_zonal=..., kh_meridional=...,
    kh_layout=...)``.

    Raises
    ------
    KeyError
        If the user explicitly set ``var_kh_zonal`` / ``var_kh_meridional``
        in the config but the corresponding variable is missing.
    ValueError
        If the two Kh variables disagree on layout or shape, or if the
        layout cannot be inferred.
    """
    if diffusion is None:
        return None
    if not getattr(diffusion, "enabled", False):
        return None

    name_x = _resolve_kh_var_name(
        ds, getattr(diffusion, "var_kh_zonal", None), _DEFAULT_KH_X_NAMES
    )
    name_y = _resolve_kh_var_name(
        ds, getattr(diffusion, "var_kh_meridional", None), _DEFAULT_KH_Y_NAMES
    )
    if name_x is None or name_y is None:
        return None

    da_x = ds[name_x]
    da_y = ds[name_y]
    if "time" in da_x.dims:
        da_x = da_x.isel(time=timestep)
    if "time" in da_y.dims:
        da_y = da_y.isel(time=timestep)

    layout_x = _detect_kh_layout(da_x, mesh)
    layout_y = _detect_kh_layout(da_y, mesh)
    if layout_x != layout_y:
        raise ValueError(
            f"Kh layout mismatch: {name_x!r} is {layout_x}-based but "
            f"{name_y!r} is {layout_y}-based.  Re-export both fields on "
            "the same mesh location."
        )

    kh_x = np.asarray(da_x.values, dtype=np.float64)
    kh_y = np.asarray(da_y.values, dtype=np.float64)
    # Squeeze singleton extra dims (e.g. a vertical layer slice arriving
    # with an unsqueezed depth=1 axis).  Anything beyond a single
    # horizontal dim remains a hard error — the integrator kernels
    # assume 1-D Kh on the mesh.
    kh_x = np.squeeze(kh_x)
    kh_y = np.squeeze(kh_y)
    # C-order explicitly: with a non-leading time dim the isel above leaves a
    # STRIDED float64 view that asarray keeps, and the parallel Kh lerp in
    # VelocityField gates on contiguity (2026-08 review).  No-op when C.
    kh_x = np.ascontiguousarray(kh_x)
    kh_y = np.ascontiguousarray(kh_y)

    n_nodes = getattr(mesh, "n_nodes", mesh.nodes_lon.shape[0])
    n_elems = getattr(mesh, "n_elements", mesh.elem_nodes.shape[0])
    expected_size = n_nodes if layout_x == "node" else n_elems
    for nm, arr in ((name_x, kh_x), (name_y, kh_y)):
        if arr.ndim != 1:
            raise ValueError(
                f"Kh variable {nm!r} must be 1-D after squeezing "
                f"(got shape {arr.shape}).  The reader expects a single "
                "horizontal dim matching the mesh."
            )
        if arr.size != expected_size:
            raise ValueError(
                f"Kh variable {nm!r} has size {arr.size} but layout "
                f"{layout_x!r} expects {expected_size} "
                f"(n_nodes={n_nodes}, n_elements={n_elems})."
            )
    if kh_x.shape != kh_y.shape:
        raise ValueError(
            f"Kh shape mismatch: {name_x!r} {kh_x.shape} vs "
            f"{name_y!r} {kh_y.shape}"
        )
    # Reject NaN/Inf — they would silently propagate into the SDE step
    # and quietly NaN out particle positions one step later.
    if not (np.all(np.isfinite(kh_x)) and np.all(np.isfinite(kh_y))):
        raise ValueError(
            f"Kh fields must be finite; {name_x!r} / {name_y!r} contain "
            "NaN or Inf values."
        )
    if np.any(kh_x < 0) or np.any(kh_y < 0):
        raise ValueError(
            "Kh fields must be non-negative; "
            f"min({name_x!r})={kh_x.min()}, min({name_y!r})={kh_y.min()}"
        )
    # Rotated frame + ANISOTROPIC pair: the two fields are a diagonal
    # diffusion tensor whose axes are the FRAME's east/north — under
    # mesh.rotation the ellipse turns with the frame instead of staying
    # geographic (declared behaviour; isotropic Kh is exactly
    # frame-invariant, spatial variation included).  Warn once per
    # (fields, frame) so the run is not silently mis-oriented.
    if (getattr(mesh, "rotation_angles", None) is not None
            and not np.array_equal(kh_x, kh_y)):
        key = (name_x, name_y, tuple(mesh.rotation_angles))
        if key not in _ANISO_KH_ROT_WARNED:
            _ANISO_KH_ROT_WARNED.add(key)
            warnings.warn(
                f"ANISOTROPIC Kh ({name_x!r} != {name_y!r}) under "
                "mesh.rotation: the pair is applied along the ROTATED "
                "axes, not geographic east/north. Use isotropic Kh "
                "(equal fields) under rotation; see "
                "docs/coordinate_frame.md §3.",
                RuntimeWarning,
            )
    return kh_x, kh_y, layout_x


def _detect_engine(path) -> str:
    """Return xarray engine name based on file path.

    Parameters
    ----------
    path : str or Path
        Path to a dataset file or directory.

    Returns
    -------
    str
        ``"zarr"`` for ``.zarr`` paths or directories containing ``.zattrs``
        (Zarr v2) or ``zarr.json`` (Zarr v3), ``"netcdf4"`` otherwise.
    """
    from pathlib import Path as _Path
    p = _Path(path)
    if p.suffix == ".zarr":
        return "zarr"
    if p.is_dir() and ((p / ".zattrs").exists() or (p / "zarr.json").exists()):
        return "zarr"
    return "netcdf4"


def _time_instants(obj):
    """Comparable form of ``obj``'s time coordinate: ``(kind, values)``.

    ``None`` when ``obj`` has no ``time`` index.  Instants that xarray
    would align are made equal here: a datetime64 index and a
    standard-calendar cftime index become ``datetime64[ns]``; an undecoded
    CF axis (``<unit> since <epoch>``) is decoded; a numeric axis with a bare
    unit token is scaled to seconds.  Anything else keeps its raw values,
    tagged with its units, so equal numbers in other units never match.
    """
    if "time" not in obj.dims or "time" not in obj.indexes:
        return None
    import pandas as pd
    idx = obj.indexes["time"]
    if isinstance(idx, pd.DatetimeIndex):
        return "datetime", np.asarray(idx.values).astype("datetime64[ns]")
    cal = getattr(idx, "calendar", None)
    if cal is not None:                                  # CFTimeIndex
        if cal in ("standard", "gregorian", "proleptic_gregorian"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        dti = idx.to_datetimeindex(time_unit="ns")
                    except TypeError:            # xarray < 2025.01: no time_unit
                        dti = idx.to_datetimeindex()
                return "datetime", np.asarray(dti.values).astype("datetime64[ns]")
            except Exception:                            # out of datetime64 range
                pass
        return f"cftime:{cal}", np.asarray(list(idx), dtype=object)
    vals = np.asarray(idx.values)
    attrs = obj["time"].attrs
    units = str(attrs.get("units", "")).strip()
    if np.issubdtype(vals.dtype, np.number) and " since " in units.lower():
        import xarray as _xr
        dec = _xr.coding.times.decode_cf_datetime(
            vals, units, calendar=attrs.get("calendar", "standard"))
        tmp = _xr.DataArray(np.zeros(len(vals)), dims="time",
                            coords={"time": np.asarray(dec)})
        return _time_instants(tmp)
    scale = _TIME_UNIT_SECONDS.get(units.lower())
    if scale is not None and np.issubdtype(vals.dtype, np.number):
        # integer microseconds: 0.1 s and 100 ms are one instant, and float
        # scaling must not split them
        us = vals.astype(np.float64) * (scale * 1e6)
        if np.isfinite(us).all():
            return "microseconds", np.round(us).astype(np.int64)
        return "microseconds", us
    return f"raw:{units}", vals


def _instant_keys(inst):
    """Hashable, ordered keys of a :func:`_time_instants` result."""
    kind, vals = inst
    return (vals.astype("int64") if kind == "datetime" else vals).tolist()


def _match_records(base, other):
    """Position in ``other`` of each of ``base``'s records.

    Returns ``(pos, why)``: ``pos`` is an int array over ``base``'s records
    (-1 where ``other`` has no such record), ``why`` a reason string when
    the two axes cannot be matched (``pos`` then None).  Rules:

    * either side without a ``time`` dimension: ``(None, None)``;
    * either side without a time coordinate: records pair by position,
      which needs equal counts;
    * identical axes (also with repeated stamps, e.g. concatenated segments
      with an overlap): records pair by position;
    * otherwise by instant (:func:`_time_instants`); refused when the kinds
      differ, a stamp is invalid (NaT/NaN), a side repeats a stamp, no stamp
      is shared, or — for numeric axes without an epoch, where a shared
      number can be a coincidence (0 days == 0 hours) — ``other`` holds a
      record inside ``base``'s span that is not one of ``base``'s.  Dates
      (datetime64 / cftime) are unambiguous instants, so a finer ``other``
      (a daily W next to a 2-day u extract) matches.
    """
    if "time" not in base.dims or "time" not in other.dims:
        return None, None
    nb, no = int(base.sizes["time"]), int(other.sizes["time"])
    if no == 0 and nb > 0:
        return None, "it has no records"
    tb, to = _time_instants(base), _time_instants(other)
    if tb is None or to is None:
        if nb != no:
            return None, f"{no} records against {nb} (no time coordinate to match them by)"
        return np.arange(nb), None
    (kb, vb), (ko, vo) = tb, to
    if kb != ko:
        return None, (f"time axes of another kind ({ko} against {kb}; "
                      f"{type(other.indexes['time']).__name__} against "
                      f"{type(base.indexes['time']).__name__})")
    kb_list, ko_list = _instant_keys(tb), _instant_keys(to)
    if nb == no and kb_list == ko_list:
        return np.arange(nb), None
    for v in (vb, vo):
        if (kb == "datetime" and np.isnat(v).any()) or (
                v.dtype.kind == "f" and np.isnan(v).any()):
            return None, "invalid time stamps (NaT/NaN)"
    if len(set(kb_list)) < nb or len(set(ko_list)) < no:
        return None, ("repeated time stamps that the other file does not "
                      "repeat identically: records cannot be matched by instant")
    lookup = {v: i for i, v in enumerate(ko_list)}
    pos = np.array([lookup.get(v, -1) for v in kb_list], dtype=np.int64)
    if nb and (pos < 0).all():
        return None, (f"no common time stamp ({vo[0]} .. {vo[-1]} against "
                      f"{vb[0]} .. {vb[-1]})")
    is_date = kb == "datetime" or kb.startswith("cftime:")
    if nb:
        lo, hi = min(kb_list), max(kb_list)
        known = set(kb_list)
        foreign = sum(1 for v in ko_list if lo <= v <= hi and v not in known)
        if foreign and not is_date:
            return None, (f"{foreign} of its records inside the span "
                          f"{vb[0]} .. {vb[-1]} are not records of it "
                          "(another cadence, origin or unit)")
        if foreign:
            # dates are unambiguous instants: a finer cadence is sub-sampled
            # onto base's records -- but say so, since a time-MEAN base next
            # to a finer field takes that field at single instants
            warnings.warn(
                f"{foreign} of its records lie between the records it is "
                f"attached to ({vb[0]} .. {vb[-1]}): a finer cadence is "
                "sub-sampled onto those records (if they are time means over "
                "a longer window, the attached values are not their mean)",
                RuntimeWarning, stacklevel=4)
    return pos, None


def _time_axis_mismatch(base, other) -> Optional[str]:
    """Why ``other`` is not on exactly ``base``'s records, or ``None``.

    xarray attaches one field onto another dataset by aligning the ``time``
    INDEX: a file whose stamps differ (another epoch, numeric seconds without
    units, another cadence, a sub-range) is silently padded with NaN at every
    unmatched record.  The answer is ``None`` when either side has no
    ``time`` dimension, or when both hold the same records in the same order
    (compared as instants, see :func:`_time_instants`; without a time
    coordinate on one side, by position).
    """
    pos, why = _match_records(base, other)
    if why is not None:
        return why
    if pos is None:
        return None
    nb, no = int(base.sizes["time"]), int(other.sizes["time"])
    if nb != no:
        return f"{no} records against {nb}"
    if not np.array_equal(pos, np.arange(nb)):
        return "the same number of records but other time stamps"
    return None


def _require_same_time_axis(base, other, what: str,
                            base_what: str = "the velocity file") -> None:
    """Raise ``ValueError`` unless ``other`` holds exactly ``base``'s records.

    For inputs that are paired by record POSITION (e.g. the u/v/ssh files of
    ``utils/reconstruct_w.py``); attachment by label uses
    :func:`_align_onto_time_axis`.
    """
    why = _time_axis_mismatch(base, other)
    if why is not None:
        raise ValueError(
            f"{what} does not share the time axis of {base_what}: {why}.")


def _align_onto_time_axis(base, other, what: str,
                          base_what: str = "the velocity file", *,
                          allow_missing: bool = False,
                          require_time: bool = False,
                          hint: str = "",
                          warn_missing: bool = False):
    """``other`` re-positioned onto ``base``'s records, or ``ValueError``.

    xarray attaches a field by its time labels and pads every unmatched
    record with NaN; this returns ``other`` with exactly ``base``'s records
    (and ``base``'s time coordinate), so the attachment is exact:

    * every record of ``base`` found in ``other`` (``other`` may hold more,
      e.g. a full-year W next to a cropped u file) -> those records;
    * ``allow_missing`` -> records of ``base`` absent from ``other`` become
      NaN (a W reconstructed for a ``time_range`` only; the reader refuses
      such a record when a run asks for it); without it they are refused;
    * no common stamp, another kind of axis, other units -> refused;
    * missing records may lie only outside ``other``'s own first..last
      stamp: a record of ``base`` inside that range that ``other`` lacks is
      another cadence (a daily W next to 6-hourly u), not the edge of a
      ``time_range``, and is refused;
    * ``require_time`` -> ``other`` without a ``time`` dimension is refused
      (a time-varying field must vary in time).

    The NaN fill of missing records is lazy (dask), so a partly covering
    file is not read into memory at open.
    """
    if "time" not in base.dims:
        return other
    if "time" not in other.dims:
        if require_time:
            raise ValueError(
                f"{what} has no time dimension, but {base_what} has "
                f"{int(base.sizes['time'])} records")
        return other
    pos, why = _match_records(base, other)
    if why is not None:
        raise ValueError(
            f"{what} does not share the time axis of {base_what}: {why}. "
            "Attached by label, its unmatched records would be NaN."
            + (f" {hint}" if hint else ""))
    missing = pos < 0
    if missing.any() and not allow_missing:
        raise ValueError(
            f"{what} does not cover the time axis of {base_what}: it lacks "
            f"{int(missing.sum())} of its {int(base.sizes['time'])} records; "
            "attached by label they would be NaN.")
    if missing.any():
        # Missing records may lie only OUTSIDE other's own first..last stamp
        # (the edges of a time_range block).  A velocity record between two
        # of other's stamps that other lacks is another cadence (a daily W
        # next to 6-hourly u, a monthly field next to daily u).
        kb = _instant_keys(_time_instants(base))
        ko = _instant_keys(_time_instants(other))
        lo, hi = min(ko), max(ko)
        holes = sum(1 for k, m in zip(kb, missing) if m and lo <= k <= hi)
        if holes:
            raise ValueError(
                f"{what} lacks {holes} records of {base_what} inside its own "
                "time range (another cadence): only a block of the velocity "
                "file's records (a reconstruct_w time_range) can be attached "
                "partly.")
    out = other.isel(time=np.where(missing, 0, pos))
    if "time" in base.coords:
        out = out.drop_vars("time", errors="ignore")
        out = out.assign_coords(time=base["time"].variable)
    # else: base has no clock — keep other's dates (the records paired by
    # position), as the label attach did; they may be the only time axis
    if missing.any() and warn_missing:
        idx = np.flatnonzero(missing)
        shown = ", ".join(str(int(i)) for i in idx[:6]) + (", ..." if idx.size > 6 else "")
        warnings.warn(
            f"{what} covers only part of the time axis of {base_what}: its "
            f"records {shown} ({idx.size} of {missing.size}) are NaN — a run "
            "that reads them (e.g. the sea surface for beaching.use_eta) "
            "sees no value there", RuntimeWarning, stacklevel=3)
    if missing.any():
        import xarray as _xr
        try:
            out = out.chunk({"time": 1})     # keep the NaN fill lazy
        except Exception:                    # no dask: fill eagerly
            pass
        out = out.where(_xr.DataArray(~missing, dims="time"))
    return out


_W_POSITIVE_VALUES = ("up", "down")


def resolve_w_sign(da, mode_cfg, family_default: str) -> float:
    """Factor that turns the file's vertical velocity into KELP's internal one.

    KELP integrates ``z += W * dt`` with ``z`` the positive-DOWN depth, so
    the internal ``VelocityField.w`` is positive downward.  Ocean models
    write the opposite: FESOM2 ``w`` (``oce_ale.F90``: ``dh/dt + div(hu)
    + (w^t - w^b) = 0``), FESOM-C ``w`` (``Wvel``, the flux through the
    sigma interfaces) and the HEALPix products (``wo``/``avg_wo``,
    "upward sea water velocity") are positive UP.

    The file's convention is resolved in this order: ``mode.w_positive``
    when it is ``"up"`` or ``"down"``; else the variable's ``positive``
    attribute; else a CF ``standard_name`` starting with ``upward`` /
    ``downward`` (``upward_sea_water_velocity``); else ``family_default``
    (the reader's verified producer convention).

    Parameters
    ----------
    da : xarray.DataArray
        The vertical-velocity variable as opened (only ``attrs`` is read).
    mode_cfg : ModeConfig or namespace
        May lack ``w_positive`` (direct callers) — treated as ``"auto"``.
    family_default : {"up", "down"}
        Convention of the reader's producer when nothing else says.

    Returns
    -------
    float
        ``-1.0`` for a positive-up file, ``+1.0`` for positive-down.
    """
    choice = str(getattr(mode_cfg, "w_positive", "auto") or "auto").lower()
    if choice not in _W_POSITIVE_VALUES:
        attrs = getattr(da, "attrs", {})
        attr = str(attrs.get("positive", "")).strip().lower()
        std = str(attrs.get("standard_name", "")).strip().lower()
        if attr in _W_POSITIVE_VALUES:
            choice = attr
        elif std.startswith("upward"):          # CF: upward_sea_water_velocity
            choice = "up"
        elif std.startswith("downward"):
            choice = "down"
        else:
            choice = family_default
    if choice not in _W_POSITIVE_VALUES:
        raise ValueError(f"family_default must be 'up' or 'down', got "
                         f"{family_default!r}")
    return -1.0 if choice == "up" else 1.0


__all__ = [
    "register",
    "get_reader",
    "extract_time_info",
    "exact_cycle",
    "resolve_w_sign",
    "_detect_engine",
    "_read_kh_fields",
    "_detect_kh_layout",
    "_detect_field_layout",
    "_classify_layout_meta",
]
