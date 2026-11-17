"""NEMO structured-grid reader (TODO 4.6).

Reads NEMO's Arakawa C-grid curvilinear output onto the unstructured ``Mesh`` via
:mod:`readers.structured`: nodes = wet T-points, elements = quads of 4 adjacent
T-points; u/v are averaged from the C-grid faces to the T-points (**node
layout** -> the run needs ``numerics.uv_interp: barycentric``).  Scalars
(temp/salt/ssh) are already at T-points.

v1: **2-D, E/N-aligned** grid only.  NEMO grid_U/grid_V are along the LOCAL grid
axes; v1 uses them as geographic east/north, valid only when the axes align with
E/N (regular lat/lon or pre-rotated output) — a grid whose axes deviate > 2deg
from E/N is rejected (``NotImplementedError``).  Vector rotation (the gate for
true curvilinear + global ORCA grids), global ORCA (north-fold seam + east-west
periodicity), and 3-D (deptht / e3t / partial cells / W on interfaces) are staged
follow-ups for this T-point path.  The exact C-grid path (``mesh.nemo_grid:
cgrid``) already handles curvilinear grids, global ORCA (E-W seam, T- and
F-point-pivot north fold, with or without the halo: ``detect_cgrid_layout``)
and ``mesh.rotation``.

File layout: NEMO splits output by C-grid point into ``*_grid_T.nc`` (temp/salt/
ssh), ``*_grid_U.nc`` (u), ``*_grid_V.nc`` (v), ``*_grid_W.nc`` (w).  Point
``data.ncinput`` at the **grid_T** file; the grid_U/grid_V siblings are derived by
swapping the ``grid_T`` token in the basename (override grid_V with
``data.ncinput_v``).  ``grid_W`` / ``ncinput_w`` and ``open_dataset``'s
``w_path`` are **reserved for the 3-D follow-up and inert in v1** — ``w`` is
always None.  The land-sea mask + coordinates come from
``mesh.meshpath`` (``mesh_mask.nc`` / ``domain_cfg.nc``), falling back to the
grid_T file's ``nav_lon``/``nav_lat`` with an all-wet mask.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import xarray as xr

from ..core.io import VelocityField
from ..core.mesh import Mesh
from . import register, _detect_engine
from .structured import (CgridLayout, _copy_sets, cgrid_to_canonical, cgrid_w_from_continuity, cgrid_thickness, SAME_POINT_CHORD, _unit_xyz, build_cgrid_mesh,
                         build_structured_mesh, c_grid_to_t, field_to_nodes)


# Variable-name aliases (NEMO output naming varies by config / XIOS field defs).
_U_ALIASES = ("uo", "vozocrtx", "uos", "u", "ubar")
_V_ALIASES = ("vo", "vomecrty", "vos", "v", "vbar")
_W_ALIASES = ("wo", "vovecrtz", "w")
_SSH_ALIASES = ("ssh", "zos", "sossheig", "sshn")
_LON_ALIASES = ("nav_lon", "glamt", "longitude", "lon")
_LAT_ALIASES = ("nav_lat", "gphit", "latitude", "lat")
_TMASK_ALIASES = ("tmask", "tmaskutil", "mask")


def _engine(path):
    return _detect_engine(str(path))


_CALENDAR_ALIASES = {"360d": "360_day", "365d": "noleap", "366d": "all_leap"}


def _expand(spec) -> list:
    """One grid's files: a list/tuple as given, a string with glob characters
    expanded (sorted; no match refuses), else the one path."""
    if isinstance(spec, (list, tuple)):
        files = [str(f) for f in spec]
    elif any(c in str(spec) for c in "*?["):
        import glob
        files = sorted(glob.glob(str(spec)))
        if not files:
            raise FileNotFoundError(f"NEMO reader: no file matches {str(spec)!r}")
    else:
        files = [str(spec)]
    if not files:
        raise FileNotFoundError("NEMO reader: an empty file list")
    return files


def _open_nemo(spec) -> xr.Dataset:
    """Open one grid's file(s); ``xr.open_dataset`` that accepts NEMO/XIOS
    calendar spellings (``360d`` -> ``360_day``, ``365d`` -> ``noleap``,
    ``366d`` -> ``all_leap``), which cftime rejects.  Several files (a list or
    a glob) are joined along time with ``open_mfdataset`` (lazy, one chunk per
    file) and must be in strictly increasing time order.  A single file
    without such an alias opens exactly as before."""
    files = _expand(spec)
    eng = _engine(files[0])

    def decode(raw):
        fix = {n: _CALENDAR_ALIASES[str(v.attrs["calendar"]).lower()]
               for n, v in raw.variables.items()
               if str(v.attrs.get("calendar", "")).lower() in _CALENDAR_ALIASES}
        for n, cal in fix.items():
            raw[n].attrs["calendar"] = cal
        return xr.decode_cf(raw), bool(fix)

    if len(files) == 1:
        raw = xr.open_dataset(files[0], engine=eng, decode_times=False)
        ds, fixed = decode(raw)
        if not fixed:
            raw.close()
            return xr.open_dataset(files[0], engine=eng)
        return ds
    with xr.open_dataset(files[0], engine=eng, decode_times=False) as first:
        tname = next((d for d in first.dims if d.lower().startswith("time")), None)
    if tname is None:
        raise ValueError(f"NEMO reader: several files need a time dimension "
                         f"to be joined along; {files[0]!r} has none")
    # every file is decoded on its own (each carries its own time epoch)
    ds = xr.open_mfdataset(files, engine=eng, decode_times=False, combine="nested",
                           concat_dim=tname, data_vars="minimal", coords="minimal",
                           compat="override", chunks={},
                           preprocess=lambda d: decode(d)[0])
    if tname in ds.indexes:
        ix = ds.indexes[tname]
        if not (ix.is_monotonic_increasing and ix.is_unique):
            ds.close()
            raise ValueError(
                f"NEMO reader: the time axis of {len(files)} files (first "
                f"{files[0]!r}) is not strictly increasing: overlapping, repeated "
                "or out-of-order files")
    return ds


def _read_layout_flags(ds) -> dict:
    """NEMO 4.2 topology attributes (global attributes or scalar variables):
    ``Iperio`` / ``Jperio`` / ``NFold`` (ints) and ``NFtype`` (``"T"`` / ``"F"``).
    Only a cross-check of what the coordinates show (NEMO <= 4.0 mesh files
    carry none of them)."""
    out = {}
    for k in ("Iperio", "Jperio", "NFold", "NFtype"):
        v = ds.attrs.get(k)
        if v is None and k in ds.variables and ds[k].size == 1:
            v = ds[k].values.reshape(-1)[0]
        if v is None:
            continue
        if isinstance(v, bytes):
            v = v.decode()
        out[k] = str(v).strip().upper() if k == "NFtype" else int(v)
    return out


#: neighbouring columns this many median spacings apart across the wrap still
#: count as one periodic raster
_WRAP_SPACINGS = 1.5


def _wrap_ratio(xyz) -> float:
    """Median distance from the last to the first column over the median column
    spacing, on the rows below the fold region."""
    ny = xyz.shape[0]
    rows = slice(0, max(1, ny - 3))
    wrap = np.linalg.norm(xyz[rows, -1] - xyz[rows, 0], axis=-1)
    step = np.median(np.linalg.norm(xyz[rows, 1:] - xyz[rows, :-1], axis=-1), axis=1)
    ok = step > 0
    return float(np.median(wrap[ok] / step[ok])) if ok.any() else np.inf


def detect_cgrid_layout(tlon, tlat, flon=None, flat=None, flags=None):
    """Classify the raw raster for ``mesh.nemo_grid: cgrid`` -> ``(CgridLayout, notes)``.

    The coordinates decide what the raster contains: the E-W halo columns, the
    north halo row and the fold are identities between points (the copies NEMO's
    ``lbc_lnk`` writes), checked exactly for the T points and the F points of
    the real rows (the halo-row F is not reliable in ORCA05; F optional), no thresholds.
    The nine candidates ``{halo columns, none} x {no fold, T / F fold, with /
    without the halo row}`` are tried; the passing one with the most copies wins
    (no-halo and halo F-pivot share a fold centre, so the count is the
    tie-break); equal counts refuse.  ``flags`` (:func:`_read_layout_flags`) only
    cross-check: a contradiction refuses; flags that claim MORE than the raster
    shows (all of our crops inherit the parent file's) make it a crop -- the
    missing topology is an open edge, noted in ``notes``.  Periodicity without
    halo columns (halo-free fold or ``Iperio``) must also pass the wrap-gap test
    (last -> first column within 1.5 median spacings): fold identities on a
    partial-width crop symmetric about a pivot would otherwise join its edges.
    Also refuses zero-filled tiles (land-processor elimination) and a pole row.
    """
    flags = flags or {}
    tlon, tlat = (np.asarray(a, dtype=np.float64) for a in (tlon, tlat))
    ny, nx = raw = tlon.shape
    have_f = flon is not None
    if have_f:
        flon, flat = (np.asarray(a, dtype=np.float64) for a in (flon, flat))
    if tlat.shape != raw or (have_f and (flon.shape != raw or flat.shape != raw)):
        raise ValueError("glamt/gphit and glamf/gphif must share one (ny, nx) shape")
    if not all(np.isfinite(a).all() for a in (tlon, tlat) + ((flon, flat) if have_f else ())):
        raise ValueError("the mesh file's glamt/gphit/glamf/gphif hold non-finite values: "
                         "the layout cannot be detected")
    if ((tlon == 0) & (tlat == 0)).sum() + (((flon == 0) & (flat == 0)).sum() if have_f else 0) > 1:
        raise ValueError(
            "the mesh file has zero-filled tiles (land-processor elimination when "
            "it was assembled): regenerate it without land elimination or use domain_cfg.nc")
    if flags.get("Jperio", 0) == 1:
        raise NotImplementedError("mesh.nemo_grid: cgrid: N-S periodic grids (Jperio = 1) are not supported")
    xt, xf = _unit_xyz(tlon, tlat), (_unit_xyz(flon, flat) if have_f else None)
    if np.linalg.norm(xt[-1] - xt[-1, :1], axis=-1).max() <= SAME_POINT_CHORD:
        raise ValueError("the top row of T points collapses to one point (a pole row): "
                         "crop below it")
    notes = []
    if ny < 4 or nx < 4:
        return CgridLayout(raw), notes

    def copies(lay):
        """Number of copy pairs compared, or None when one does not coincide."""
        n = 0
        for kind, xyz in (("T", xt), ("F", xf)):
            if xyz is None:                                # no F coordinates (CMIP output)
                continue
            dst, src, _s, present = _copy_sets(lay, kind)
            nc, pc = lay.shape[1], int(lay.pad_cols)
            k = present[dst]
            if kind == "F" and lay.fold:
                k &= dst // nc <= lay.J
            d, s_ = dst[k], src[k]
            a = xyz[d // nc, d % nc - pc]
            b = xyz[s_ // nc, s_ % nc - pc]
            if d.size and np.linalg.norm(a - b, axis=-1).max() > SAME_POINT_CHORD:
                return None
            n += int(d.size)
        return n

    found = []
    for halo_cols in (True, False):
        for fold, halo_row in ((None, False), ("T", True), ("T", False), ("F", True), ("F", False)):
            if fold is None and not halo_cols:
                continue                                   # implies no copy at all
            lay = CgridLayout(raw, halo_cols, halo_row, True, fold)
            n = copies(lay)
            if n:
                found.append((n, lay))
    found.sort(key=lambda t: -t[0])
    if len(found) > 1 and found[0][0] == found[1][0]:
        raise ValueError("the raster matches two layouts equally well: "
                         f"{found[0][1].label()} / {found[1][1].label()}")
    iperio, nfold, nftype = flags.get("Iperio"), flags.get("NFold"), flags.get("NFtype")
    ratio = _wrap_ratio(xt)
    if found:
        lay = found[0][1]
        if lay.fold and (nfold == 0 or nftype not in (None, lay.fold)):
            raise ValueError(f"the coordinates show a {lay.fold}-pivot fold, the file's flags say "
                             f"NFold={nfold}, NFtype={nftype!r}")
        if iperio == 0:
            raise ValueError("the coordinates show E-W periodicity, the file's flag says Iperio=0")
        if not lay.halo_cols and ratio > _WRAP_SPACINGS:
            raise ValueError(
                f"the fold identities hold, but the raster is not full width (the last column is "
                f"{ratio:.1f} grid spacings from the first): a partial-width crop through the fold "
                "cannot be joined; crop the full width")
        if not lay.fold and nfold == 1:
            notes.append("the flags claim a north fold the raster does not show: treated as a crop "
                         "(the north edge is open)")
        return lay, notes
    looks_periodic = ratio <= _WRAP_SPACINGS
    if iperio == 1:
        if looks_periodic:
            if nfold == 1:
                notes.append("the flags claim a north fold the raster does not show: treated as a crop "
                             "(the north edge is open)")
            return CgridLayout(raw, False, False, True, None), notes
        notes.append("the flags claim E-W periodicity the raster does not show (not full width): "
                     "treated as a crop (the east and west edges are open)")
    elif looks_periodic and iperio is None:
        raise ValueError(
            "the raster looks E-W periodic without halo columns (the last column is "
            f"{ratio:.1f} grid spacings from the first) but the file says nothing: add Iperio=1 "
            "(and NFold / NFtype for a fold) to the mesh file's attributes, or crop it")
    return CgridLayout(raw), notes



def _first(ds, aliases: Sequence[str], cfg_name: Optional[str] = None):
    """Return the variable name to use: an explicit ``cfg_name`` override if set,
    else the first present alias (``None`` if no alias matches).

    An explicit override is a REQUIREMENT, not a preference: if ``cfg_name`` is
    set but absent from the dataset it raises ``KeyError`` rather than silently
    falling back to an alias (a typoed ``var_u`` must not quietly read a
    different velocity product).
    """
    if cfg_name:
        if cfg_name in ds.variables:
            return cfg_name
        raise KeyError(
            f"NEMO reader: configured variable {cfg_name!r} not found in the "
            f"dataset (present: {list(ds.variables)[:12]}…).")
    for nm in aliases:
        if nm and nm in ds.variables:
            return nm
    return None


def _time_values(ds):
    """Return a NEMO dataset's time-coordinate values (``None`` if it has none)."""
    tn = next((d for d in ds.dims if d.lower().startswith("time")), None)
    if tn is not None and tn in ds.variables:
        return ds[tn].to_numpy()
    return None


def _time_len(ds):
    """Return the length of a NEMO dataset's time dimension (``None`` if none)."""
    tn = next((d for d in ds.dims if d.lower().startswith("time")), None)
    return int(ds.sizes[tn]) if tn is not None else None


def _squeeze_to_2d(arr: np.ndarray) -> np.ndarray:
    """Drop singleton leading axes (time/depth) to get a (y, x) field."""
    a = np.asarray(arr)
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2:
        raise ValueError(
            f"expected a 2-D (y, x) field after squeezing, got shape {a.shape}")
    return a


def _surface_2d(da: xr.DataArray) -> np.ndarray:
    """Select the surface level + squeeze a NEMO field to (y, x).

    Picks index 0 of any vertical dim (deptht/depthu/depthv/z/nav_lev/...).
    """
    for d in da.dims:
        dl = d.lower()
        if dl.startswith("depth") or dl in ("z", "nav_lev", "lev", "deptht"):
            da = da.isel({d: 0})
    return _squeeze_to_2d(da.values)


def _read_raster(da, layout, kind, name, *, vector=False, check=False, surface=False):
    """The ONLY way the cgrid path reads a raster: ``da`` (a DataArray or array,
    trailing dims the raw ``(ny, nx)`` of ``layout``; ``surface`` takes the
    surface level of a depth dimension) -> its canonical float64 form
    (:func:`cgrid_to_canonical`), so no raw halo / fold array ever meets a
    canonical-shape one.  ``kind`` is the C-grid point (``"T"``, ``"U"``,
    ``"V"``), ``vector`` only for u and v.  ``check=True`` REFUSES when a halo /
    fold copy of the raw raster differs from its canonical point: used for
    every raster that defines the flux (u, v of every record, tmask, e1t / e2t,
    e2u / e1v, e3t_0 / e3u_0 / e3v_0).  The T / S samples, bathymetry,
    ``w_file`` and ``e3t_file`` are canonicalised silently (their stale copies
    -- ``bathy_metry`` on the fold row, eddy-induced velocities -- are real)."""
    a = _surface_2d(da) if surface else np.asarray(da)
    return cgrid_to_canonical(a, layout, kind, vector=vector, check=check, name=name)


def _max_axis_misalignment_deg(lon2d: np.ndarray, lat2d: np.ndarray) -> float:
    """Max deviation (deg) of the grid i-axis from EAST and j-axis from NORTH.

    NEMO ``grid_U``/``grid_V`` velocities are along the local grid axes.  v1
    treats them as geographic east/north, which is only valid when the grid is
    locally aligned with E/N (regular lat/lon, or pre-rotated output).  A skewed
    / curvilinear (e.g. ORCA) grid needs vector rotation (a staged follow-up);
    this metric drives the fail-fast.
    """
    def _dev(dlon, dlat, latmid, want_east):
        e = ((dlon + 180.0) % 360.0 - 180.0) * np.cos(np.deg2rad(latmid))
        n = dlat
        # SIGNED angle of the axis vector from the wanted geographic direction,
        # then |.|: an axis that points the WRONG WAY (i decreasing east -> west,
        # or j decreasing north -> south) reads ~180deg and is rejected. Using
        # abs(e)/abs(n) here would fold west onto east / south onto north and
        # silently flip the sign of the grid-relative u/v we read as east/north.
        ang = (np.degrees(np.arctan2(n, e)) if want_east           # from EAST
               else np.degrees(np.arctan2(e, n)))                  # from NORTH
        ang = np.abs(ang)
        return float(np.nanmax(ang)) if ang.size else 0.0

    # i-axis (along x): should point east
    di_lat = 0.5 * (lat2d[:, :-1] + lat2d[:, 1:])
    dev_i = _dev(np.diff(lon2d, axis=1), np.diff(lat2d, axis=1), di_lat, True)
    # j-axis (along y): should point north
    dj_lat = 0.5 * (lat2d[:-1, :] + lat2d[1:, :])
    dev_j = _dev(np.diff(lon2d, axis=0), np.diff(lat2d, axis=0), dj_lat, False)
    return max(dev_i, dev_j)


_ALIGN_TOL_DEG = 2.0


def _sibling(path: str, letter: str) -> Optional[str]:
    """Derive a NEMO grid sibling by swapping the grid_X token in the FILENAME.

    Only the basename is rewritten — never directory components, so a parent
    directory like ``grid_T_run/`` is left untouched (a whole-path replace would
    point ``data/grid_T_run/EXP_grid_T.nc`` at a different directory).
    """
    p = Path(str(path))
    name = p.name
    for src in ("grid_T", "grid_U", "grid_V", "grid_W",
                "gridT", "gridU", "gridV", "gridW"):
        if src in name:
            tgt = src[:-1] + letter
            cand = str(p.with_name(name.replace(src, tgt, 1)))
            if cand != str(p) and Path(cand).exists():
                return cand
    return None

def _siblings(files, letter):
    """The grid_<letter> file of every grid_T file (``None`` when one is missing)."""
    out = [_sibling(f, letter) for f in files]
    return None if None in out else out


def _grid_point(da) -> str:
    """C-grid point of a field from its dimension / coordinate names
    (``depthu``, ``nav_lon_grid_U``, ...); ``"T"`` when nothing says otherwise."""
    import re
    names = [str(n).lower() for n in (*da.dims, *da.coords)]
    names += [str(da.attrs.get("coordinates", "")).lower()]
    for n in names:
        m = re.search(r"depth([uvwf])\b|grid_([uvwf])\b|_([uvwf])_(?:lon|lat)", n)
        if m:
            return next(g for g in m.groups() if g).upper()
    return "T"


def _refuse_unsupported_mode(mode_cfg, where):
    """Fail fast on mode settings the v1 reader cannot honour."""
    if mode_cfg is None:
        return
    if getattr(mode_cfg, "sigma_mean", False):
        raise NotImplementedError(
            "the NEMO reader tracks the surface level only in v1; "
            "mode.sigma_mean (a vertical mean) is not supported.")
    layer = getattr(mode_cfg, "sigma_layer", None)
    if layer not in (None, 0):
        # _surface_2d always takes index 0 of every depth dimension; a deeper
        # layer used to be ignored silently (audit 2026-10-04).
        raise NotImplementedError(
            "the NEMO reader tracks the surface level only in v1; "
            f"mode.sigma_layer = {layer!r} is not supported (use 0 or null).")


def _refuse_unsupported_vertical(ds, path) -> None:
    """z-levels with partial bottom cells only: s-coordinates and ice-shelf
    cavities are refused (the file's attributes, ``top_level``, ``misf``)."""
    vc = ds.attrs.get("VertCoord")
    if vc is not None and str(vc).strip().lower() not in ("zco", "zps"):
        raise NotImplementedError(
            f"mesh file {path!r}: VertCoord = {str(vc)!r}; only z-levels "
            "(zco, zps) are supported, not s-coordinates")
    # only real cavities: the variables count at WET columns (ORCA05 mesh_mask
    # carries misf = 1 and isfdraft = 0 there, and 0 on land)
    if "tmask" in ds.variables:
        tm = np.asarray(ds["tmask"]).squeeze()
        wet = (tm > 0.5).any(0) if tm.ndim == 3 else tm > 0.5
    elif "bottom_level" in ds.variables:
        wet = np.asarray(ds["bottom_level"]).squeeze() > 0
    else:
        wet = None
    cav = int(np.asarray(ds.attrs.get("IsfCav", 0)).reshape(-1)[0]) != 0
    for name, bad in (("misf", lambda a: a > 1), ("top_level", lambda a: a > 1),
                      ("isfdraft", lambda a: a > 0)):
        if cav or name not in ds.variables:
            continue
        a = np.asarray(ds[name]).squeeze()
        cav = bool((bad(a) if wet is None or a.shape != wet.shape else bad(a) & wet).any())
    if cav:
        raise NotImplementedError(
            f"mesh file {path!r} has ice-shelf cavities (IsfCav / misf > 1 / "
            "top_level > 1 / isfdraft > 0 at wet columns); they are not "
            "supported (levels must start at the surface)")


def _check_zps_rule(zw, nlev, e3b, e3u0, e3v0):
    """Load-time consistency check: the model's face thickness at a face's
    bottom level must be the NEMO partial-step rule, the smaller of its two
    cells' thickness there (``e3u(j,i) = min(e3t(j,i), e3t(j,i+1))``; ``v``
    likewise towards ``j+1``).  Interior faces of wet cells only (the crop's
    last column / row has no neighbour cell here; the mesh keeps the file's
    value there).  The kernel uses the file's e3u_0 / e3v_0, not this rule."""
    ref = np.diff(zw)

    bad = 0.0
    for e3f, sl_a, sl_b in ((e3u0, np.s_[:, :-1], np.s_[:, 1:]),
                            (e3v0, np.s_[:-1, :], np.s_[1:, :])):
        na, nb = nlev[sl_a], nlev[sl_b]
        kb = np.minimum(na, nb) - 1
        wet = kb >= 0
        k = np.maximum(kb, 0)
        rule = np.minimum(np.where(na - 1 == k, e3b[sl_a], ref[k]),
                          np.where(nb - 1 == k, e3b[sl_b], ref[k]))
        got = np.take_along_axis(e3f[(slice(None),) + sl_a], k[None], 0)[0]
        bad = max(bad, float(np.max(np.abs(got - rule)[wet] / rule[wet], initial=0.0)))
        above = np.arange(len(ref))[:, None, None] < kb[None]      # wet levels above the bottom
        bad = max(bad, float(np.max(np.abs(e3f[(slice(None),) + sl_a] / ref[:, None, None] - 1)
                                    * above, initial=0.0)))
    if bad > 1e-4:
        raise ValueError(
            "mesh.nemo_grid: cgrid 3-D: e3u_0/e3v_0 differ from the partial-step "
            f"rule min(e3t_0 of the two cells) by {bad:.2e} (relative); only "
            "z-levels with partial steps (zps) are supported")


@register("nemo")
class NemoReader:
    """Reader for NEMO curvilinear C-grid output (2-D v1)."""

    @staticmethod
    def load_mesh(mesh_cfg) -> Mesh:
        if (getattr(mesh_cfg, "rotation", None) is not None
                and getattr(mesh_cfg, "nemo_grid", "tpoint") != "cgrid"):
            # The T-point path has no vector-rotation of its averaged u/v
            # (mirrors FESOM-C); the cgrid path needs none (its u/v come from
            # the cell geometry in the mesh frame).
            raise ValueError(
                "mesh.rotation is not supported by the NEMO reader with "
                "mesh.nemo_grid: tpoint (no vector-rotation path); use "
                "rotation: null, or mesh.nemo_grid: cgrid.")
        path = str(mesh_cfg.meshpath)
        ds = _open_nemo(path)
        try:
            if getattr(mesh_cfg, "nemo_grid", "tpoint") == "cgrid":
                return NemoReader._load_cgrid_mesh(ds, path, mesh_cfg)
            lon_n = _first(ds, _LON_ALIASES)
            lat_n = _first(ds, _LAT_ALIASES)
            if lon_n is None or lat_n is None:
                raise KeyError(
                    f"NEMO mesh {path!r} must contain nav_lon/nav_lat "
                    f"(or glamt/gphit). Found: {list(ds.variables)[:12]}…")
            lon2d = _squeeze_to_2d(ds[lon_n].values)
            lat2d = _squeeze_to_2d(ds[lat_n].values)
            # v1 fail-fast: u/v are taken as geographic east/north, valid only on
            # a grid whose axes align with E/N. A curvilinear / rotated grid
            # (e.g. global ORCA) needs vector rotation -> a staged follow-up.
            mis = _max_axis_misalignment_deg(lon2d, lat2d)
            if mis > _ALIGN_TOL_DEG:
                raise NotImplementedError(
                    f"NEMO reader v1: the grid axes deviate {mis:.1f}° from "
                    f"geographic east/north (> {_ALIGN_TOL_DEG}°), so its "
                    "grid-relative u/v cannot be used as east/north without "
                    "vector rotation (a staged follow-up). v1 supports "
                    "geographic-aligned grids (regular lat/lon or pre-rotated "
                    "output) only.")
            tmask_n = _first(ds, _TMASK_ALIASES)
            if tmask_n is not None:
                wet = _surface_2d(ds[tmask_n]) > 0.5
            else:
                wet = np.ones(lon2d.shape, dtype=bool)
            depth2d = None
            for dn in ("hdept", "hdepw", "bathy_meter", "bathy_metry",
                       "bathymetry", "Bathymetry"):
                if dn in ds.variables:
                    depth2d = _squeeze_to_2d(ds[dn].values)
                    break
            r_earth = getattr(mesh_cfg, "r_earth", None)
            mesh = build_structured_mesh(
                lon2d, lat2d, wet, depth2d=depth2d, name="nemo", r_earth=r_earth)
        finally:
            ds.close()
        return mesh

    @staticmethod
    def _load_cgrid_mesh(ds, path, mesh_cfg) -> Mesh:
        """Exact C-grid mesh (``mesh.nemo_grid: cgrid``): F-point cells from
        glamf/gphif + tmask.  No axis-alignment gate: the C-grid interpolator
        works in index space and returns geographic u/v itself."""
        try:
            glamf, gphif = _squeeze_to_2d(ds["glamf"].values), _squeeze_to_2d(ds["gphif"].values)
        except KeyError:
            raise KeyError(
                f"mesh.nemo_grid: cgrid needs glamf/gphif (F-point coordinates) "
                f"in the mesh file {path!r}; found: {list(ds.variables)[:12]}…"
            ) from None
        tl = _first(ds, ("glamt", "nav_lon"))
        tp = _first(ds, ("gphit", "nav_lat"))
        if tl is None or tp is None:
            raise KeyError(
                f"mesh.nemo_grid: cgrid needs the T-point coordinates "
                f"(glamt/gphit or nav_lon/nav_lat) in {path!r} to classify the "
                "grid (regional / global ORCA)")
        layout, notes = detect_cgrid_layout(
            _squeeze_to_2d(ds[tl].values), _squeeze_to_2d(ds[tp].values), glamf, gphif,
            _read_layout_flags(ds))
        print(f"  NEMO cgrid layout: {layout.label()}")
        for n in notes:
            print(f"    note: {n}")

        def rd(nm, kind, check=True):
            return _read_raster(ds[nm].squeeze(), layout, kind, nm, check=check)
        glamf = cgrid_to_canonical(glamf, layout, "F", name="glamf")
        gphif = cgrid_to_canonical(gphif, layout, "F", name="gphif")
        _refuse_unsupported_vertical(ds, path)
        tmask_n = _first(ds, ("tmask",))
        if tmask_n is not None:
            tm3 = rd(tmask_n, "T") > 0.5          # boolean: no float64 3-D copy kept
        elif all(v in ds.variables for v in ("bottom_level", "top_level", "e3t_1d")):
            # domain_cfg.nc (NEMO 4.x): no tmask; wet levels 1..bottom_level
            bot = rd("bottom_level", "T")
            tm3 = ((np.arange(1, ds["e3t_1d"].size + 1)[:, None, None] <= bot)
                   & (bot > 0))
        else:
            raise KeyError(
                f"mesh.nemo_grid: cgrid needs tmask (mesh_mask.nc) or top_level, "
                f"bottom_level and e3t_1d (domain_cfg.nc) in the mesh file {path!r}")
        wet = tm3[0] if tm3.ndim == 3 else tm3
        depth2d = None
        vert = {}
        if tm3.ndim == 3 and all(
                v in ds.variables for v in ("e3t_1d", "e1t", "e2t")):
            # 3-D z-levels: interfaces from the 1-D e3t; partial (zps) bottom
            # cells from e3t_0 (without it: full reference cells)
            e3 = np.squeeze(ds["e3t_1d"].values).astype(np.float64)
            zw = np.r_[0.0, np.cumsum(e3)]   # not gdepw_1d: its differences are
            # up to 1e-3 off e3t_1d, which the model's fluxes use
            nlev = tm3.sum(0)
            if not (tm3 == (np.arange(tm3.shape[0])[:, None, None] < nlev)).all():
                raise ValueError("mesh.nemo_grid: cgrid 3-D: tmask is not a "
                                 "contiguous wet column from the surface "
                                 "(ice-shelf cavities are not supported)")
            e3b = e3u0 = e3v0 = None
            if all(v in ds.variables for v in ("e3t_0", "e3u_0", "e3v_0")):
                e3t0, e3u0, e3v0 = (rd(v, k) for v, k in
                                    (("e3t_0", "T"), ("e3u_0", "U"), ("e3v_0", "V")))
                jj, ii = np.nonzero(nlev > 0)
                e3b = np.zeros(nlev.shape)
                e3b[jj, ii] = e3t0[nlev[jj, ii] - 1, jj, ii]
                above = np.arange(e3.size)[:, None, None] < nlev - 1     # not the bottom cell
                ref = np.broadcast_to(e3[:, None, None], e3t0.shape)
                if (np.abs(e3t0 - ref)[above] > 1e-3 * ref[above]).any():
                    raise NotImplementedError(
                        f"mesh file {path!r}: e3t_0 differs from e3t_1d above the "
                        "bottom cell (s-coordinates / terrain-following levels); "
                        "only z-levels with partial bottom cells (zco / zps) are "
                        "supported")
                _check_zps_rule(zw, nlev, e3b, e3u0, e3v0)
            for nm in ("e1t", "e2t"):
                if (np.asarray(ds[nm]) <= 0).any():
                    raise ValueError(
                        f"the mesh file's {nm} has non-positive entries: zero-filled "
                        "tiles (land-processor elimination when it was assembled)? "
                        "regenerate it without land elimination or use domain_cfg.nc")
            vert = dict(nlev2d=nlev, zw=zw, e3b=e3b, e3u0=e3u0, e3v0=e3v0,
                        area=rd("e1t", "T") * rd("e2t", "T"))
            print(f"  NEMO cgrid 3-D: {tm3.shape[0]} levels, "
                  + ("partial bottom cells (e3t_0, e3u_0, e3v_0)" if e3b is not None else
                     "full reference cells (no e3t_0 / e3u_0 / e3v_0 in the mesh file)"))
        for dn in (() if vert else ("hdept", "hdepw", "bathy_meter", "bathy_metry",
                                    "bathymetry", "Bathymetry")):
            if dn in ds.variables:
                depth2d = rd(dn, "T", check=False)
                break
        widths = {}
        if "e2u" in ds.variables and "e1v" in ds.variables:
            widths = dict(e2u=rd("e2u", "U"), e1v=rd("e1v", "V"))
            print("  NEMO cgrid: fluxes use the model face widths e2u/e1v")
        else:
            print("  NEMO cgrid: e2u/e1v not in the mesh file; fluxes use plane "
                  "edge lengths (cos at each edge's mean latitude)")
        from .fesom2 import POLE_WARN_LAT, _resolve_rotation_angles
        angles = _resolve_rotation_angles(mesh_cfg)
        max_lat = float(np.nanmax(np.abs(gphif)))
        if angles is None and max_lat > POLE_WARN_LAT:
            import warnings
            warnings.warn(
                f"NEMO cgrid mesh reaches |lat| = {max_lat:.2f} deg - the "
                "lon/lat cell geometry degenerates near the pole (containment "
                "and the local plane fail there); set mesh.rotation: fesom2.",
                RuntimeWarning, stacklevel=2)
        if angles is not None:
            # Rotate the F-points AS POINTS; the interpolator builds u/v from
            # the cell geometry, so it returns them in the rotated frame with
            # no vector rotation (the face values are grid-relative normals;
            # the widths e2u/e1v are lengths, frame-free).
            from ..core.rotation import euler_matrix, rotate_lonlat
            glamf, gphif = rotate_lonlat(glamf, gphif, euler_matrix(*angles))
            print(f"    rotated frame: alpha/beta/gamma = {tuple(angles)} "
                  "(pole-safe)")
        mesh = build_cgrid_mesh(glamf, gphif, wet, depth2d=depth2d, name="nemo",
                                layout=layout,
                                partial_cells=getattr(mesh_cfg, "nemo_partial_cells", "shrunk"),
                                r_earth=getattr(mesh_cfg, "r_earth", None),
                                **widths, **vert)
        mesh.rotation_angles = angles
        return mesh

    @staticmethod
    def open_dataset(path, v_path=None, w_path=None, mode_cfg=None, u_path=None) -> xr.Dataset:
        """Assemble grid_T (+ derived/overridden grid_U, grid_V[, grid_W]) into
        ONE dataset with vars u, v, [w], ssh, temp, salt sharing (time, [z],
        y, x) — so the per-step reader and the prefetch ``isel(time=slice)`` work.

        Every path argument may be a glob or a list: the files of one grid are
        joined along time.  grid_U/grid_V/grid_W default to the name-siblings of
        each grid_T file (the ``grid_T`` token swapped); ``u_path`` / ``v_path`` /
        ``w_path`` replace that for files without the token.
        """
        _refuse_unsupported_mode(mode_cfg, "open_dataset")
        t_files = _expand(path)
        u_files = _expand(u_path) if u_path else _siblings(t_files, "U")
        v_files = _expand(v_path) if v_path else _siblings(t_files, "V")
        if u_files is None or v_files is None:
            raise FileNotFoundError(
                "NEMO reader: could not locate the grid_U / grid_V files of "
                f"{path!r}. They must be name-siblings of the grid_T files (the "
                "'grid_T' token swapped to 'grid_U'), or be given by "
                "data.ncinput_u / data.ncinput_v.")
        dsT = _open_nemo(t_files)
        dsU = _open_nemo(u_files)
        dsV = _open_nemo(v_files)

        # The merge below takes the time axis from grid_T only and indexes every
        # variable by the same integer timestep, so grid_U / grid_V MUST share
        # grid_T's timestamps. A mismatched (e.g. shifted / differently-sampled)
        # axis would silently pair u(t_i) with ssh(t_j) — fail fast instead.
        # Three layers: record COUNT (works even with no decodable time coord, so
        # a cadence mismatch can't slip through); ALL-OR-NONE coordinate presence
        # (a coord on one side but not the other can't be validated -> refuse,
        # never graft T's axis onto an unlabelled sibling); exact TIMESTAMP
        # comparison when both carry a coordinate.
        nT, tT = _time_len(dsT), _time_values(dsT)
        for nm, dsX in (("grid_U", dsU), ("grid_V", dsV)):
            nX = _time_len(dsX)
            if nT is not None and nX is not None and nX != nT:
                raise ValueError(
                    f"NEMO reader: {nm} has {nX} time records but grid_T has "
                    f"{nT}. U/V/T must be on the same time axis so a timestep "
                    "index refers to one physical time.")
            tX = _time_values(dsX)
            if nX is not None and (tT is None) != (tX is None):
                raise ValueError(
                    f"NEMO reader: {nm} and grid_T disagree on whether a time "
                    "coordinate is present, so their pairing cannot be "
                    "verified. Write all of grid_T/U/V with a time coordinate "
                    "(or none of them).")
            if tT is not None and tX is not None and (
                    tX.shape != tT.shape or not np.array_equal(tX, tT)):
                raise ValueError(
                    f"NEMO reader: {nm} time axis does not match grid_T "
                    f"({tX.shape} vs {tT.shape}; or differing timestamps). "
                    "U/V/T must be written on the same time axis so a "
                    "timestep index refers to one physical time.")

        cfg = mode_cfg
        u_n = _first(dsU, _U_ALIASES, getattr(cfg, "var_u", None) if cfg else None)
        v_n = _first(dsV, _V_ALIASES, getattr(cfg, "var_v", None) if cfg else None)
        if u_n is None or v_n is None:
            raise KeyError("NEMO reader: u/v variable not found in grid_U/grid_V "
                           f"(looked for {_U_ALIASES} / {_V_ALIASES}).")

        def _rename_tz(da):
            ren = {}
            for d in da.dims:
                dl = d.lower()
                if dl.startswith("time"):
                    ren[d] = "time"
                elif dl.startswith("depth") or dl in ("nav_lev", "lev", "z"):
                    ren[d] = "z"
            return da.rename(ren) if ren else da

        data = {"u": _rename_tz(dsU[u_n]), "v": _rename_tz(dsV[v_n])}
        ssh_n = _first(dsT, _SSH_ALIASES, getattr(cfg, "var_eta", None) if cfg else None)
        if ssh_n is not None:
            data["ssh"] = _rename_tz(dsT[ssh_n])
        # carry sampled T-point scalars (temp/salt/any) through under their names.
        # A missing/typoed sample_var is dropped silently HERE but caught at the
        # run_tracking startup probe (read_scalar_fields raises ValueError on an
        # absent var before any I/O), so it is never a silent wrong answer.
        for nm in (getattr(cfg, "sample_vars", None) or []):
            for ds_x, g in ((dsT, "T"), (dsU, "U"), (dsV, "V")):
                if nm in ds_x.variables:
                    pt = _grid_point(ds_x[nm])
                    if g != "T" or pt != "T":
                        raise ValueError(
                            f"NEMO reader: sample_vars {nm!r} lives on the "
                            f"{pt if g == 'T' else g}-points; only T-point fields "
                            "(grid_T: temperature, salinity, ...) can be sampled")
                    data[nm] = _rename_tz(dsT[nm])
                    break

        w_files = _expand(w_path) if w_path else _siblings(t_files, "W")
        if getattr(cfg, "Fdrift3D", False) and w_files is not None:
            dsW = _open_nemo(w_files)
            wn = _first(dsW, ("vovecrtz", "w", "wo"))
            if wn is not None:            # the file's W: only compared with the
                data["w_file"] = _rename_tz(dsW[wn])   # continuity W
        if getattr(cfg, "Fdrift3D", False) and "e3t" in dsT.variables:
            data["e3t_file"] = _rename_tz(dsT["e3t"])    # z*/vvl check only
        merged = xr.Dataset({k: v.variable for k, v in data.items()})
        # Preserve a usable time axis (``.variable`` dropped coords) so dt
        # auto-detect / temporal_interp work, mirroring the other readers.
        tname = next((d for d in dsT.dims if d.lower().startswith("time")), None)
        if tname is not None and tname in dsT.variables:
            # keep attrs and encoding: the output's time:calendar comes from
            # them (bare values made a 'gregorian' axis 'proleptic_gregorian')
            tv = dsT[tname].variable
            merged = merged.assign_coords(time=xr.Variable(
                ("time",), tv.values, tv.attrs, tv.encoding))
        elif "time" in merged.dims and "time" not in merged.coords:
            # No decodable time coordinate anywhere: still give the merged "time"
            # dim an integer coordinate so ds["time"] exists downstream
            # (extract_time_info treats a plain arange as "no real time" ->
            # the run must set numerics.dt explicitly), instead of leaving a
            # bare dim that KeyErrors in extract_time_info / the writer.
            merged = merged.assign_coords(
                time=np.arange(merged.sizes["time"], dtype=np.int64))
        return merged

    @staticmethod
    def probe_uv_layout(ds, mode_cfg=None, mesh=None) -> str:
        # NEMO u/v are averaged onto T-points -> node layout (needs barycentric),
        # or kept on the faces for the exact C-grid scheme (mesh.nemo_grid: cgrid).
        # Surfacing it here lets run_tracking fail fast BEFORE creating a writer.
        if mesh is not None and getattr(mesh, "_cgrid", False):
            return "cgrid"
        return "node"

    @staticmethod
    def read_velocities(ds, timestep, mesh, mode_cfg=None, diffusion=None,
                        sample_vars=None, velocity_float32=False) -> VelocityField:
        _refuse_unsupported_mode(mode_cfg, "read_velocities")
        # em/m1 spatial diffusion would need Kh fields read onto T-points (node
        # layout); not wired yet -> fail fast (uniform diffusion works: its Kh is
        # a config scalar, not read from the dataset).
        if (diffusion is not None
                and getattr(diffusion, "enabled", False)
                and getattr(diffusion, "scheme", "uniform") in ("em", "m1")):
            raise NotImplementedError(
                "NEMO reader v1: only uniform diffusion is supported "
                "(em/m1 read spatial Kh fields, not yet mapped to T-points).")
        vd = np.float32 if velocity_float32 else np.float64
        if getattr(mesh, "_cgrid", False):
            lay = mesh._cgrid_layout
            scal = (NemoReader.read_scalar_fields(ds, timestep, mesh, sample_vars,
                                                  mode_cfg) if sample_vars else None)
            if getattr(mode_cfg, "Fdrift3D", False):
                return NemoReader._read_cgrid_3d(ds, timestep, mesh, vd, scal)
            faces = [_read_raster(ds[nm].isel(time=timestep), lay, k, nm, vector=True,
                                  check=True, surface=True).reshape(-1).astype(vd, copy=False)
                     for nm, k in (("u", "U"), ("v", "V"))]
            return VelocityField(
                u=faces[0], v=faces[1], eta=np.zeros(mesh.n_nodes), w=None,
                uv_layout="cgrid", scalars=scal)
        if getattr(mode_cfg, "Fdrift3D", False):
            raise NotImplementedError(
                "NEMO reader: mode.Fdrift3D needs mesh.nemo_grid: cgrid; the T-point "
                "path is 2-D only. Use mesh.nemo_grid: cgrid or mode.Fdrift3D: false.")
        node_id = mesh._structured_node_id

        u2d = _surface_2d(ds["u"].isel(time=timestep))
        v2d = _surface_2d(ds["v"].isel(time=timestep))
        u_t, v_t = c_grid_to_t(u2d, v2d, node_id)   # node-indexed, float64

        eta = np.zeros(mesh.n_nodes, dtype=np.float64)
        if "ssh" in ds.variables:
            eta = field_to_nodes(
                _squeeze_to_2d(ds["ssh"].isel(time=timestep).values), node_id)
            # A wet T-node should carry valid ssh; guard against a grid_T mask
            # that disagrees with tmask (masked/NaN ssh at a wet node) poisoning
            # eta and, with beaching use_eta, the drying logic.
            eta = np.nan_to_num(eta, nan=0.0, posinf=0.0, neginf=0.0)

        scalars = None
        if sample_vars:
            scalars = NemoReader.read_scalar_fields(
                ds, timestep, mesh, sample_vars, mode_cfg)

        return VelocityField(
            u=u_t.astype(vd, copy=False), v=v_t.astype(vd, copy=False),
            eta=eta, w=None, uv_layout="node", scalars=scalars)

    @staticmethod
    def prepare(mesh, ds) -> None:
        """z*/vvl check (3-D cgrid only; changes nothing): when ``grid_T`` carries
        ``e3t`` (merged as ``e3t_file``), warn if records 0 / last deviate from
        the reference thickness.  Fixed reference thicknesses (``e3*_0``) are
        used either way, so z* transports are off by about eta/H."""
        if not (getattr(mesh, "_cgrid", False) and mesh._cgrid_zw.size
                and "e3t_file" in ds.variables):
            return
        ny, nx = mesh._structured_shape
        nlev = mesh._cgrid_nlev_raster
        ref = cgrid_thickness(mesh._cgrid_zw, np.where(nlev > 0, nlev - 1, -1),
                              mesh._cgrid_e3b.reshape(ny, nx))
        wet = np.arange(ref.shape[0])[:, None, None] < nlev
        dev = max(float(np.nanmax(np.abs(_read_raster(ds["e3t_file"].isel(time=r), mesh._cgrid_layout,
                                                      "T", "e3t_file") - ref)[wet] / ref[wet]))
                  for r in (0, ds.sizes["time"] - 1))
        if dev > 1e-4:
            import warnings
            warnings.warn(
                f"NEMO cgrid 3-D: the file's e3t differs from the reference "
                f"e3t_0 by up to {dev:.1%} (records 0 and last): z*/vvl "
                "detected. Fixed reference thicknesses (e3*_0) are used, so "
                "the transports are off by about eta/H (z* is not supported "
                "yet).", RuntimeWarning, stacklevel=2)

    @staticmethod
    def _read_cgrid_3d(ds, timestep, mesh, vd, scalars):
        """3-D faces of one record on the partial-step cells, plus the W
        that makes them non-divergent up to float32 rounding (``cgrid_w_from_continuity``)."""
        ny, nx = mesh._structured_shape
        zw = mesh._cgrid_zw
        if zw.size == 0:
            raise ValueError("mode.Fdrift3D with mesh.nemo_grid: cgrid needs a "
                             "3-D mesh_mask (3-D tmask, e3t_1d, e1t, e2t)")
        lay = mesh._cgrid_layout
        uv = []
        for nm, k in (("u", "U"), ("v", "V")):
            a = _read_raster(ds[nm].isel(time=timestep), lay, k, nm, vector=True, check=True)
            if a.shape != (zw.size - 1, ny, nx):
                raise ValueError(f"NEMO {nm} shape {a.shape} != (K, ny, nx) "
                                 f"{(zw.size - 1, ny, nx)}")
            uv.append(a.astype(vd).astype(np.float64))   # w consistent with stored u/v
        e2u, e1v = (m.reshape(ny, nx) for m in (mesh._cgrid_e2u, mesh._cgrid_e1v))
        e3u, e3v = (cgrid_thickness(zw, getattr(mesh, f"_cgrid_kb{n}").reshape(ny, nx),
                                    getattr(mesh, f"_cgrid_e3{n}").reshape(ny, nx))
                    for n in ("u", "v"))
        w = cgrid_w_from_continuity(uv[0], uv[1], e2u, e1v, e3u, e3v,
                                    mesh._cgrid_area.reshape(ny, nx),
                                    mesh._cgrid_nlev_raster)
        if not getattr(mesh, "_cgrid_w_logged", False):
            mesh._cgrid_w_logged = True
            wet = w[:-1] != 0
            msg = (f"  NEMO cgrid 3-D: W from continuity, max |w(0)| = "
                   f"{np.abs(w[0]).max():.2e} m/s (surface residual)")
            if "w_file" in ds.variables:
                wf = -_read_raster(ds["w_file"].isel(time=timestep), lay, "T", "w_file")
                if wf.shape == w[:-1].shape and wet.any():
                    msg += (f"; corr with the file's W = "
                            f"{np.corrcoef(w[:-1][wet], wf[wet])[0, 1]:.3f}")
            print(msg)
        return VelocityField(
            u=uv[0].reshape(uv[0].shape[0], -1).astype(vd),
            v=uv[1].reshape(uv[1].shape[0], -1).astype(vd),
            w=w.reshape(w.shape[0], -1).astype(vd), eta=np.zeros(mesh.n_nodes),
            uv_layout="cgrid", scalars=scalars)

    @staticmethod
    def read_scalar_fields(ds, timestep, mesh, varnames, mode_cfg=None) -> dict:
        _refuse_unsupported_mode(mode_cfg, "read_scalar_fields")
        if getattr(mesh, "_cgrid", False):
            # T-cell values on the raster (land -> NaN: the files carry 0.0
            # there), flattened (ny*nx,) / (K, ny*nx) for sample_field_cgrid
            ny, nx = mesh._structured_shape
            out = {}
            for nm in varnames:
                if nm not in ds.variables:
                    raise ValueError(f"NEMO reader: sampled variable {nm!r} not "
                                     "in the dataset (grid_T, mode.sample_vars).")
                three = getattr(mode_cfg, "Fdrift3D", False)
                a = _read_raster(ds[nm].isel(time=timestep), mesh._cgrid_layout, "T", nm,
                                 surface=not three)
                if three:
                    nl = mesh._cgrid_nlev_raster
                    a = np.where(np.arange(a.shape[0])[:, None, None] < nl, a, np.nan)
                    out[nm] = a.reshape(a.shape[0], -1)
                else:
                    out[nm] = np.where(mesh._cgrid_nlev_raster > 0, a, np.nan).reshape(-1)
            return out
        node_id = mesh._structured_node_id
        out = {}
        for nm in varnames:
            if nm not in ds.variables:
                raise ValueError(
                    f"NEMO reader: sampled variable {nm!r} not in the dataset "
                    "(it must be a T-point field in grid_T and listed in "
                    "mode.sample_vars at open_dataset time).")
            f2d = _surface_2d(ds[nm].isel(time=timestep))
            out[nm] = field_to_nodes(f2d, node_id)
        return out
