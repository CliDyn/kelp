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
cgrid``) already handles curvilinear grids, global ORCA with halo columns and an
F-point-pivot fold (``_detect_orca_layout``) and ``mesh.rotation``.

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
from .structured import (SAME_POINT_CHORD, _unit_xyz, build_cgrid_mesh,
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


def _open_nemo(path) -> xr.Dataset:
    """``xr.open_dataset`` that accepts NEMO/XIOS calendar spellings
    (``360d`` -> ``360_day``, ``365d`` -> ``noleap``, ``366d`` -> ``all_leap``),
    which cftime rejects.  Files without such an alias open exactly as before."""
    path = str(path)
    raw = xr.open_dataset(path, engine=_engine(path), decode_times=False)
    fix = {n: _CALENDAR_ALIASES[str(v.attrs["calendar"]).lower()]
           for n, v in raw.variables.items()
           if str(v.attrs.get("calendar", "")).lower() in _CALENDAR_ALIASES}
    if not fix:
        raw.close()
        return xr.open_dataset(path, engine=_engine(path))
    for n, cal in fix.items():
        raw[n].attrs["calendar"] = cal
    return xr.decode_cf(raw)


def _detect_orca_layout(lon, lat) -> Optional[str]:
    """Classify the T-point raster for ``mesh.nemo_grid: cgrid``.

    Returns ``None`` (closed regional raster: its edges are walls), ``"ew"``
    (east-west halo columns: ``col 0 == col nx-2``, ``col nx-1 == col 1``) or
    ``"fpivot"`` (halo columns + F-point-pivot north fold, ``T(ny-1, i) ==
    T(ny-2, nx-1-i)``; ORCA05 / ORCA1).  Points are "the same" within
    :data:`SAME_POINT_CHORD` on the unit sphere (the copies are exact in
    model output).  Raises ``NotImplementedError`` for the global layouts it
    does not handle rather than letting them become walls: a T-point-pivot
    fold (ORCA2 / ORCA025 / ORCA12), a fold without halo columns, and
    halo-free periodic output (NEMO >= 4.2 without the halo columns: the
    longitudes span 360 deg and the gap from the last column to the first is
    about one grid spacing).
    """
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    ny, nx = lon.shape
    if ny < 4 or nx < 4:
        return None
    xyz = _unit_xyz(lon, lat)

    def same(a, b):
        return bool((np.linalg.norm(a - b, axis=-1) <= SAME_POINT_CHORD).all())

    def gap(a, b):
        return np.linalg.norm(a - b, axis=-1)

    ew = same(xyz[:, 0], xyz[:, nx - 2]) and same(xyz[:, nx - 1], xyz[:, 1])
    i = np.arange(1, nx - 1)
    fpivot = same(xyz[ny - 1, i], xyz[ny - 2, nx - 1 - i])
    i2 = np.arange(2, nx - 1)
    tpivot = same(xyz[ny - 1, i2], xyz[ny - 3, nx - i2])
    # a self-mirrored top row (halo-free fold: the pivot row is kept whole)
    selfmirror = (same(xyz[ny - 1, i], xyz[ny - 1, nx - 1 - i])
                  or same(xyz[ny - 1, i2], xyz[ny - 1, nx - i2]))
    if not ew:
        # periodic without halo: the wrap gap (last -> first column) is about
        # one grid spacing on the rows below the fold region
        rows = slice(0, max(1, ny - 3))
        wrap = gap(xyz[rows, nx - 1], xyz[rows, 0])
        step = np.median(gap(xyz[rows, 1:], xyz[rows, :-1]), axis=1)
        ok = step > 0
        periodic = ok.any() and np.median(wrap[ok] / step[ok]) < 1.5
        if periodic or fpivot or tpivot or selfmirror:
            what = ("east-west periodic without halo columns (the last column "
                    "is one grid spacing from the first: NEMO >= 4.2 halo-free "
                    "output?)" if periodic else
                    "a north fold without east-west halo columns")
            raise NotImplementedError(
                f"mesh.nemo_grid: cgrid: the grid is {what}; supported global "
                "layout: ORCA with east-west halo columns (col 0 == col nx-2, "
                "col nx-1 == col 1) and an F-point-pivot north fold. Write the "
                "output with the halo columns, or crop a closed regional "
                "domain.")
        return None
    if fpivot:
        return "fpivot"
    if tpivot or selfmirror:
        raise NotImplementedError(
            "mesh.nemo_grid: cgrid: T-point-pivot north fold (ORCA2 / ORCA025 / "
            "ORCA12 type) is not supported; the F-point-pivot fold (ORCA05 / "
            "ORCA1) is.")
    return "ew"


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

def _refuse_unsupported_mode(mode_cfg, where):
    """Fail fast on mode settings the v1 reader cannot honour."""
    if mode_cfg is None:
        return
    if getattr(mode_cfg, "Fdrift3D", False):
        raise NotImplementedError(
            f"the NEMO reader is 2-D only in v1 ({where}; deptht/e3t/partial "
            "cells/W are a staged follow-up); set mode.Fdrift3D: false.")
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
        orca = _detect_orca_layout(_squeeze_to_2d(ds[tl].values),
                                   _squeeze_to_2d(ds[tp].values))
        if orca is not None:
            print(f"  NEMO cgrid: global ORCA layout ({orca}): east-west halo "
                  "columns" + (" + F-pivot north fold" if orca == "fpivot"
                               else "") + " merged")
        tmask_n = _first(ds, ("tmask",))
        if tmask_n is None:
            raise KeyError(
                f"mesh.nemo_grid: cgrid needs tmask in the mesh file {path!r}")
        wet = _surface_2d(ds[tmask_n]) > 0.5
        depth2d = None
        for dn in ("hdept", "hdepw", "bathy_meter", "bathy_metry",
                   "bathymetry", "Bathymetry"):
            if dn in ds.variables:
                depth2d = _squeeze_to_2d(ds[dn].values)
                break
        widths = {}
        if "e2u" in ds.variables and "e1v" in ds.variables:
            widths = dict(e2u=_squeeze_to_2d(ds["e2u"].values),
                          e1v=_squeeze_to_2d(ds["e1v"].values))
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
                                orca=orca,
                                r_earth=getattr(mesh_cfg, "r_earth", None),
                                **widths)
        mesh.rotation_angles = angles
        return mesh

    @staticmethod
    def open_dataset(path, v_path=None, w_path=None, mode_cfg=None) -> xr.Dataset:
        """Assemble grid_T (+ derived/overridden grid_U, grid_V[, grid_W]) into
        ONE dataset with vars u, v, [w], ssh, temp, salt sharing (time, [z],
        y, x) — so the per-step reader and the prefetch ``isel(time=slice)`` work.
        """
        _refuse_unsupported_mode(mode_cfg, "open_dataset")
        dsT = _open_nemo(path)
        u_file = _sibling(path, "U")
        v_file = v_path or _sibling(path, "V")
        if u_file is None or v_file is None:
            raise FileNotFoundError(
                "NEMO reader: could not locate grid_U / grid_V siblings of "
                f"{path!r}. grid_U must be a name-sibling of grid_T (the "
                "'grid_T' token swapped to 'grid_U'); grid_V likewise or via "
                "data.ncinput_v. Point data.ncinput at the *_grid_T.nc file.")
        dsU = _open_nemo(u_file)
        dsV = _open_nemo(v_file)

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
            if nm in dsT.variables:
                data[nm] = _rename_tz(dsT[nm])

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
            if sample_vars:
                raise NotImplementedError(
                    "mesh.nemo_grid: cgrid does not support sample_vars (v1)")
            ny, nx = mesh._structured_shape
            faces = []
            for nm in ("u", "v"):
                a = _surface_2d(ds[nm].isel(time=timestep))
                if a.shape != (ny, nx):
                    raise ValueError(
                        f"NEMO {nm} shape {a.shape} != mesh grid {(ny, nx)}")
                faces.append(np.nan_to_num(
                    np.asarray(a, dtype=np.float64), nan=0.0, posinf=0.0,
                    neginf=0.0).reshape(-1).astype(vd, copy=False))
            return VelocityField(
                u=faces[0], v=faces[1], eta=np.zeros(mesh.n_nodes), w=None,
                uv_layout="cgrid")
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
    def read_scalar_fields(ds, timestep, mesh, varnames, mode_cfg=None) -> dict:
        _refuse_unsupported_mode(mode_cfg, "read_scalar_fields")
        if getattr(mesh, "_cgrid", False):
            raise NotImplementedError(
                "mesh.nemo_grid: cgrid does not support scalar sampling (v1)")
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
