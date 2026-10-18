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
"""
from __future__ import annotations

from typing import Optional, Tuple

import warnings

import numpy as np

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
    "resolve_w_sign",
    "_detect_engine",
    "_read_kh_fields",
    "_detect_kh_layout",
    "_detect_field_layout",
    "_classify_layout_meta",
]
