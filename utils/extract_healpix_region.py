#!/usr/bin/env python3
"""Regional pixel-subset extractor for HEALPix stores -> plain Zarr.

The HEALPix analog of ``extract_fesom_region.py``, but simpler and exact
(docs/healpix.md §4): a region is a PIXEL-INDEX SUBSET — a bbox test on
pixel centres plus an optional topological halo — no mesh surgery.  The
output is a plain Zarr store in native healpix terms, **self-describing**:

- ``healpix_index (cell,)`` — the source NEST pixel id of every kept cell
  (sorted ascending; the exact geometry is reconstructible without the
  source store), plus a ``crs`` variable and provenance attrs (grid type,
  nside/zoom, ordering, source URI, bbox, halo, time range, variables,
  creation stamp, tool version).
- The kelp healpix reader treats an extract like any other store: it
  finds ``healpix_index``, builds the dual mesh on the subset, and skips
  nothing else — ``mesh.crop_bbox`` composes (a further crop).
- Extracts are **NEST-canonical**: a ring-ordered source is reordered at
  extraction, so the reader never needs healpy index maps on extracts.

Boundary semantics match the FESOM extractor: particles die at the sector
edge.  ``region.halo_layers`` widens the sector by N rings of pixels
(``healpy.get_all_neighbours`` dilation) so the dual mesh strictly inside
the working area is identical to the whole-store build (default 1).

Usage::

    python utils/extract_healpix_region.py config.yaml [--force] [--dry-run]

See ``utils/extract_healpix_example.yaml`` for the configuration format.
The 2-D eta recipe is resolved AT EXTRACTION: ssh from ``source.eta_path``
(or the auto-derived ``3D``->``2D`` filename sibling) is merged into the
extract, so an extract is a single self-contained store.

Vertical axes (the convention of the kelp healpix reader: K interface
depths, w on the K interfaces, every other variable with a vertical dim
a layer variable on K-1 entries, or on K with an empty last level):

- **The interface table** is the source's own ``level_interfaces_m`` when
  it carries one, else ``source.levels``; both given and different ->
  refused.  The extract carries the table (cut with the levels).
- **w** is ``source.var_w`` when set (added to the default variable list;
  with an explicit ``variables:`` list it must be listed), else the first
  w alias among the extracted variables.
- **Before anything is cut**, with a table, every extracted vertical axis
  is checked against the FULL table: w exactly K entries, a layer variable
  K-1 or K.  A wrong table, or a layer variable on the interfaces, is
  refused here rather than hidden by a cut.
- **``region.max_levels: M`` with a table** keeps ``keep = min(M, K)``
  interfaces: w its first ``keep`` entries, every layer variable its first
  ``keep - 1``, the table its first ``keep``.  A layer variable on K
  entries loses its K-th plane, which must be empty at the exported cells
  in every exported record (else refused, naming the source record).
  When w shares its vertical dim with a layer variable (a padded store,
  everything on one ``level`` axis), w's dim is renamed to the first
  unused ``level_w``, ``level_w_1``, ... (with its coordinates), since one
  dim cannot hold two lengths.  When ``keep < K`` and the source has no
  depth variable, the extract carries ``deptho``, the bed the reader
  computes on the whole store (from the finite levels of the uncut u at
  the first exported record and the full table), so a run on the extract
  has the source's bed.
- **``region.max_levels: 1``** is a surface extract for 2-D runs: every
  vertical axis keeps one entry, no table is resolved, checked or
  written, no bed is synthesised (a source depth variable is carried).
  Without a table, ``max_levels: M`` keeps the first M entries of every
  vertical axis.  ``max_levels`` below 1 is refused.
- ``--dry-run`` makes every metadata check but reads no data plane (the
  last-level check is announced as not run).

Extracts carry ``tool = "extract_healpix_region v1.1"``; the reader reads
those written by v1.0 (whose cut kept M entries on every vertical axis)
as before, with a warning.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from kelp.readers.healpix import (_DEPTH_ALIASES,  # noqa: E402
                                      _ETA_ALIASES, _LEVELS_VAR,
                                      _U_ALIASES, _V_ALIASES, _W_ALIASES,
                                      _bbox_wet_mask, _derive_eta_sibling,
                                      _find_crs, _first, _first_soft,
                                      _open_healpix_store, _require_healpy,
                                      _resolve_levels,
                                      _validate_time_axes_match,
                                      _vertical_dim)

TOOL = "extract_healpix_region"
TOOL_VERSION = "1.1"

#: printed for region.max_levels: 1 (the table-less surface cut)
_SURFACE_NOTE = ("region.max_levels: 1 -> a surface extract for 2-D runs "
                 "(no interface table written; a 3-D extract needs "
                 "max_levels >= 2)")


def _cell_dim(da):
    """The healpix cell dim = the last dim (ng5: ``value``; CMIP: ``cell``)."""
    return da.dims[-1]


def _max_levels(cfg):
    """``region.max_levels`` (else the legacy top-level ``levels:``) as an
    int, or None; below 1 -> refused."""
    m = (cfg.get("region") or {}).get("max_levels", cfg.get("levels"))
    if m is None:
        return None
    m = int(m)
    if m < 1:
        raise ValueError(
            f"region.max_levels: {m} is refused: it counts the levels to "
            "keep and must be at least 1 (1 = a surface extract for 2-D "
            "runs; a 3-D extract needs at least 2 interfaces).")
    return m


def _source_table(ds, src):
    """The interface table of the source and its label: the store's own
    ``level_interfaces_m`` when present, else ``source.levels``; (None,
    None) without either.  Both present and different -> refused."""
    cfg_tab = _resolve_levels(src.get("levels"))
    if _LEVELS_VAR in ds.variables:
        emb = np.abs(np.asarray(ds[_LEVELS_VAR].values,
                                dtype=np.float64)).reshape(-1)
        if cfg_tab is not None and not np.array_equal(cfg_tab, emb):
            raise ValueError(
                f"source.levels ({cfg_tab.size} interfaces) conflicts with "
                "the interface table the source store carries "
                f"(level_interfaces_m, {emb.size} interfaces) — drop "
                "source.levels.")
        return emb, "the store's level_interfaces_m"
    if cfg_tab is not None:
        return cfg_tab, "source.levels"
    return None, None


def _unused(base, used):
    """``base``, else the first of ``base_1``, ``base_2``, ... not in
    ``used``."""
    name, i = base, 0
    while name in used:
        i += 1
        name = f"{base}_{i}"
    return name


def _check_tail(da, name, lev_dim, k_src, src_pos, tsl, table_label):
    """The K-th plane of the layer variable ``da`` (K entries on
    ``lev_dim``) must hold no non-NaN value at the exported cells
    (``src_pos``, positions on the source's cell axis) in every exported
    record; read record by record.  Refuses naming the first offending
    record by its index on the source's time axis."""
    cd = _cell_dim(da)
    plane = da.isel({lev_dim: k_src - 1})
    if "time" in plane.dims:
        records = np.arange(int(plane.sizes["time"]))[tsl]
    else:
        records = [None]
    for i in records:
        p = plane if i is None else plane.isel(time=int(i))
        vals = np.asarray(p.isel({cd: src_pos}).values)
        m = int((~np.isnan(vals)).sum())
        if m == 0:
            continue
        if i is None:
            where = "(no time axis)"
        else:
            try:
                stamp = str(da["time"].values[int(i)])[:19]
            except Exception:
                stamp = "unknown"
            where = f"in source record {int(i)} (time {stamp})"
        raise ValueError(
            f"{name!r} has {k_src} entries on {lev_dim!r} and its last "
            f"level carries data at {m} of the {int(src_pos.size)} "
            f"exported cells {where}; region.max_levels would drop that "
            f"level. The table ({table_label}) may be one interface "
            f"short, or {name!r} sits on the interfaces (not supported).")


def _check_lengths(out, expect, where):
    """Every ``{name: (dim, n)}`` of ``expect`` has n entries on dim."""
    for nm, (dim, n) in expect.items():
        got = (int(out[nm].sizes[dim])
               if nm in out.variables and dim in out[nm].dims else None)
        if got != n:
            raise RuntimeError(
                f"{TOOL}: {where}, {nm!r} has {got} entries on {dim!r}, "
                f"expected {n}.")


def _pixel_selection(hp, nside: int, order: str, bbox, halo_layers: int):
    """NEST pixel ids of the sector: bbox test on centres + halo dilation."""
    sel_store = _bbox_wet_mask(hp, nside, order, bbox)   # store order
    if order == "ring":
        # reorder mask to NEST: value_nest[p] = value_ring[ring_of(p)]
        ring_of_nest = hp.nest2ring(nside, np.arange(len(sel_store)))
        sel = sel_store[ring_of_nest]
    else:
        sel = sel_store
    if not sel.any():
        raise ValueError(f"region.bbox {list(bbox)} selects no pixels at "
                         f"nside {nside}")
    for _ in range(int(halo_layers)):
        pix = np.flatnonzero(sel).astype(np.int64)
        nb = hp.get_all_neighbours(nside, pix, nest=True).reshape(-1)
        nb = nb[nb >= 0]
        sel = sel.copy()
        sel[nb] = True
    return np.flatnonzero(sel).astype(np.int64)          # sorted NEST ids


def extract_region(cfg: dict, *, dry_run: bool = False,
                   force: bool = False) -> Path:
    """Run one extraction described by a config dict; returns the output path."""
    import xarray as xr

    hp = _require_healpy()
    src = cfg.get("source") or {}
    region = cfg.get("region") or {}
    time_cfg = cfg.get("time") or {}
    out_cfg = cfg.get("output") or {}
    meta_cfg = cfg.get("metadata") or {}

    src_path = str(Path(str(src["path"])).expanduser())
    out_path = Path(str(out_cfg["path"])).expanduser()
    if out_path.suffix != ".zarr":
        raise ValueError(f"output.path must end in .zarr, got {out_path}")
    if out_path.exists() and not force:
        raise FileExistsError(
            f"{out_path} exists — pass --force to overwrite")
    bbox = region.get("bbox")
    if bbox is None:
        raise ValueError("region.bbox [W, E, S, N] is required")
    halo = int(region.get("halo_layers", 1))
    if halo < 0:
        raise ValueError(f"region.halo_layers must be >= 0, got {halo}")

    ds = _open_healpix_store(src_path)
    try:
        return _extract_region_inner(
            cfg, ds, src, region, time_cfg, out_cfg, meta_cfg, src_path,
            out_path, bbox, halo, dry_run=dry_run)
    finally:
        ds.close()


def _extract_region_inner(cfg, ds, src, region, time_cfg, out_cfg, meta_cfg,
                          src_path, out_path, bbox, halo, *, dry_run):
    hp = _require_healpy()
    nside, order = _find_crs(ds)
    if nside is None:
        zoom = src.get("healpix_zoom")
        if zoom is None:
            raise ValueError(
                f"source store {src_path!r} carries no healpix crs attrs; "
                "set source.healpix_zoom.")
        nside, order = 1 << int(zoom), "nest"
    npix = int(hp.nside2npix(nside))
    if "healpix_index" in ds.variables:
        raise ValueError(
            f"source store {src_path!r} is already an extract — extract "
            "from the original whole store (or widen the first extract).")

    # --- region.max_levels, before anything else about the levels --------
    # (renamed from a top-level 'levels' key — kept as an alias — to
    # disambiguate from source.levels, the interface-depth TABLE)
    max_lev = _max_levels(cfg)
    if max_lev == 1:
        print(_SURFACE_NOTE)

    # --- variables -------------------------------------------------------
    # source.var_u / source.var_v / source.var_w: names outside the alias
    # lists (a run on such an extract needs the same mode.var_*)
    var_u = src.get("var_u")
    var_v = src.get("var_v")
    var_w = src.get("var_w")
    if var_w:
        _first(ds, _W_ALIASES, var_w)      # set but absent -> KeyError
    want = cfg.get("variables")
    if want:
        names = [str(v) for v in want]
        missing = [n for n in names if n not in ds.variables]
        if missing:
            raise KeyError(
                f"variables {missing} not in the source store (present: "
                f"{sorted(ds.data_vars)[:12]}…)")
        # the extract must stay reader-consumable: u AND v must be among
        # the extracted variables (the reader builds the mesh from them)
        ok_u = (var_u in names) if var_u else bool(
            set(names) & set(_U_ALIASES))
        ok_v = (var_v in names) if var_v else bool(
            set(names) & set(_V_ALIASES))
        if not (ok_u and ok_v):
            raise ValueError(
                f"variables {names} lack a u/v pair (aliases {_U_ALIASES} "
                f"/ {_V_ALIASES}, or source.var_u/var_v) — the kelp "
                "reader cannot consume an extract without velocities.")
        if var_w and var_w not in names:
            raise ValueError(
                f"source.var_w {var_w!r} is not in variables: {names} — "
                "list it there.")
    else:
        u_name = _first(ds, _U_ALIASES, var_u)
        v_name = _first(ds, _V_ALIASES, var_v)
        if u_name is None or v_name is None:
            raise KeyError(
                f"source store {src_path!r}: no u/v variable found "
                f"(looked for {_U_ALIASES} / {_V_ALIASES}); list "
                "variables: explicitly or set source.var_u/var_v.")
        names = [u_name, v_name]
        if var_w and var_w not in names:
            names.append(var_w)

    # --- eta (merged INTO the extract -> single self-contained store) ----
    ds_eta = None
    eta_source = ""
    eta_name = _first_soft(ds, _ETA_ALIASES)
    if eta_name is not None:
        eta_source = f"source:{eta_name}"
        if eta_name not in names:
            names.append(eta_name)
    else:
        eta_p = src.get("eta_path") or _derive_eta_sibling(src_path)
        if eta_p:
            ds_eta = _open_healpix_store(str(eta_p))
            try:
                eta_name = _first(ds_eta, _ETA_ALIASES)
                if eta_name is None:
                    raise KeyError(
                        f"eta store {eta_p!r} has no ssh variable "
                        f"(looked for {_ETA_ALIASES})")
                _validate_time_axes_match(ds, ds_eta,
                                          f"eta store {eta_p!r}")
            except Exception:
                ds_eta.close()
                raise
            eta_source = str(eta_p)
            if eta_name not in names:
                names.append(eta_name)

    # ds_eta must survive until the lazy write below has computed — and
    # must be closed on EVERY exit path in between (round-2 review).
    try:
        return _assemble_and_write(
            cfg, ds, ds_eta, hp, nside, order, npix, names, eta_name,
            eta_source, time_cfg, out_cfg, meta_cfg, src_path, out_path,
            bbox, halo, max_lev, dry_run=dry_run)
    finally:
        if ds_eta is not None:
            ds_eta.close()


def _assemble_and_write(cfg, ds, ds_eta, hp, nside, order, npix, names,
                        eta_name, eta_source, time_cfg, out_cfg, meta_cfg,
                        src_path, out_path, bbox, halo, max_lev, *,
                        dry_run):
    import xarray as xr

    src = cfg.get("source") or {}

    def _src_da(nm):
        """The source DataArray of an extracted name: the eta of the
        second store (absent from ``ds``), else the main store's."""
        if ds_eta is not None and nm == eta_name:
            return ds_eta[nm]
        return ds[nm]

    # bathymetry travels with the extract when the source carries one
    # (feeds the reader's wet-mask refinement + beaching depth criteria)
    d_name = _first_soft(ds, _DEPTH_ALIASES)
    if d_name is not None and d_name not in names:
        names.append(d_name)

    # --- pixel subset ----------------------------------------------------
    keep_nest = _pixel_selection(hp, nside, order, bbox, halo)
    if order == "ring":
        src_pos = hp.nest2ring(nside, keep_nest)   # row position per pixel
    else:
        src_pos = keep_nest

    def _idx(key):
        v = time_cfg.get(key)
        return None if v is None else int(v)   # NOT `or None`: 0 is valid

    tsl = slice(_idx("start"), _idx("stop"), _idx("step"))

    # --- vertical axes against the interface table -----------------------
    # region.max_levels: 1 is the table-less surface cut: no table is
    # resolved, checked or written there
    table, table_label = None, None
    if max_lev is None or max_lev >= 2:
        table, table_label = _source_table(ds, src)
    cut_active = table is not None and max_lev is not None and max_lev >= 2
    k_src = None if table is None else int(table.size)
    keep = min(max_lev, k_src) if cut_active else None
    w_name = src.get("var_w") or next(
        (nm for nm in _W_ALIASES if nm in names), None)
    # (vertical dim, length) of every extracted variable that has one
    vaxes = {}
    for nm in names:
        if nm == _LEVELS_VAR:
            continue
        da = _src_da(nm)
        vd = _vertical_dim(da, nm)
        if vd is not None:
            vaxes[nm] = (vd, int(da.sizes[vd]))

    # with a table, BEFORE any cut (metadata only): w exactly K entries,
    # every layer variable K-1 or K — a wrong table or a layer variable on
    # the interfaces is refused before a cut could hide it
    tail = []
    if table is not None:
        if w_name is not None:
            if w_name not in vaxes:
                raise ValueError(
                    f"the vertical velocity {w_name!r} has no vertical dim "
                    f"(dims {_src_da(w_name).dims}); with an interface "
                    f"table ({table_label}) w must sit on exactly the "
                    f"{k_src} interfaces.")
            wd, n = vaxes[w_name]
            if n != k_src:
                raise ValueError(
                    f"the vertical velocity {w_name!r} has {n} entries on "
                    f"{wd!r}, but the interface table ({table_label}) has "
                    f"{k_src} interfaces; w must sit on exactly the K "
                    "interfaces — check that the table lists the "
                    "INTERFACE depths of the source store.")
        for nm, (vd, n) in vaxes.items():
            if nm != w_name and n not in (k_src - 1, k_src):
                raise ValueError(
                    f"the layer variable {nm!r} has {n} entries on {vd!r}; "
                    f"with {k_src} interfaces ({table_label}) a layer "
                    f"variable needs {k_src - 1} entries (layer mids), or "
                    f"{k_src} with an empty last level.")
        if cut_active:
            # the cut drops the K-th plane of a layer variable on K
            # entries: it must be empty at the exported cells
            tail = [nm for nm, (vd, n) in vaxes.items()
                    if nm != w_name and n == k_src]
    if tail and dry_run:
        print(f"dry run — the last-level check of {', '.join(tail)} was "
              "not run")
    elif tail:
        for nm in tail:
            _check_tail(_src_da(nm), nm, vaxes[nm][0], k_src, src_pos, tsl,
                        table_label)

    # w's dim is renamed when it shares one with a layer variable (cut
    # only): one dim cannot hold w's `keep` and the layers' `keep - 1`
    # entries (xarray pads the shorter back or refuses); every coordinate
    # of w on that dim goes with it
    w_rename = {}
    if cut_active and w_name is not None:
        wd = vaxes[w_name][0]
        if any(vd == wd for nm, (vd, _) in vaxes.items() if nm != w_name):
            used = (set(map(str, ds.dims)) | set(map(str, ds.variables))
                    | set(names)
                    | {"healpix_index", "crs", "lon", "lat", "deptho",
                       _LEVELS_VAR, "level_interface"})
            if ds_eta is not None:
                used |= (set(map(str, ds_eta.dims))
                         | set(map(str, ds_eta.variables)))
            w_rename[wd] = _unused("level_w", used)
            used.add(w_rename[wd])
            w_da = _src_da(w_name)
            for c in w_da.coords:
                if c != wd and wd in w_da[c].dims:
                    w_rename[c] = _unused(f"{c}_w", used)
                    used.add(w_rename[c])

    # --- assemble the output dataset (lazy) ------------------------------
    cd_main = _cell_dim(_src_da(names[0]))
    data = {}
    expect = {}          # name -> (vertical dim in the extract, length)
    for nm in names:
        da = _src_da(nm)
        cd = _cell_dim(da)
        if int(da.sizes[cd]) != npix:
            raise ValueError(
                f"variable {nm!r}: cell dim {cd!r} has {da.sizes[cd]} "
                f"entries, expected npix={npix} for nside {nside}")
        if "time" in da.dims:
            da = da.isel(time=tsl)
        if nm in vaxes:
            vd, n = vaxes[nm]
            if cut_active:           # w keeps `keep`, the layers keep - 1
                n_keep = keep if nm == w_name else keep - 1
            elif max_lev is not None:   # table-less: the first M entries
                n_keep = min(max_lev, n)
            else:
                n_keep = n
            if n_keep != n:
                da = da.isel({vd: slice(0, n_keep)})
            expect[nm] = (w_rename.get(vd, vd) if nm == w_name else vd,
                          n_keep)
        da = da.isel({cd: src_pos})
        if cd != cd_main:
            # e.g. an eta sibling naming its cell axis differently —
            # unify, or the Dataset would grow two cell dims and
            # healpix_index would attach to only one
            da = da.rename({cd: cd_main})
        if nm == w_name and w_rename:
            da = da.rename(w_rename)
        data[nm] = da
    out = xr.Dataset(data)

    # the interface table travels with the extract (cut with the levels),
    # so 3-D runs on it need no mesh.levels
    if table is not None:
        tab = table[:keep] if cut_active else table
        out[_LEVELS_VAR] = ("level_interface", tab)
        out[_LEVELS_VAR].attrs.update({
            "long_name": "vertical interface depths",
            "units": "m", "positive": "down"})
        expect[_LEVELS_VAR] = ("level_interface", int(tab.size))
    _check_lengths(out, expect, "after the assembly")

    # the bed of a table cut that shortens the table: the reader's rule on
    # the whole store (finite levels of the UNCUT u at the first exported
    # record, the FULL table), so a run on the extract has the source's
    # bed and not one at the deepest kept interface
    written = list(names)
    if cut_active and keep < k_src and d_name is None and not dry_run:
        u_name = src.get("var_u") or next(
            (nm for nm in _U_ALIASES if nm in names), None)
        if u_name in vaxes:
            u0 = _src_da(u_name)
            if "time" in u0.dims:
                recs = np.arange(int(u0.sizes["time"]))[tsl]
                u0 = u0.isel(time=int(recs[0]) if recs.size else 0)
            col = u0.isel({_cell_dim(u0): src_pos})
            nfin = np.asarray(np.isfinite(col).sum(
                dim=vaxes[u_name][0]).values).reshape(-1)
            bed = np.where(nfin > 0, table[np.minimum(nfin, k_src - 1)],
                           0.0)
            out["deptho"] = (cd_main, bed)
            out["deptho"].attrs.update({
                "long_name": f"sea floor depth from the finite levels of "
                             f"{u_name} and the interface table",
                "units": "m", "positive": "down"})
            written.append("deptho")

    cd = _cell_dim(out[names[0]])
    lon, lat = hp.pix2ang(nside, keep_nest, nest=True, lonlat=True)
    out = out.assign_coords({
        "lon": (cd, lon), "lat": (cd, lat)})
    out["healpix_index"] = (cd, keep_nest)
    out["healpix_index"].attrs.update({
        "long_name": "source HEALPix pixel id of each kept cell",
        "ordering": "nest", "healpix_nside": int(nside)})
    out["crs"] = ((), np.int32(0), {
        "grid_mapping_name": "healpix",
        "healpix_nside": int(nside),
        "healpix_order": "nest"})
    tvals = out["time"].values if "time" in out.variables else None
    out.attrs.update({
        "grid_type": "healpix",
        "healpix_nside": int(nside),
        "healpix_zoom": int(nside).bit_length() - 1,
        "healpix_order": "nest",
        "source_store": src_path,
        "source_order": order,
        "eta_merged": bool(eta_name is not None),
        "eta_source": eta_source,
        "levels_source": str(src.get("levels") or ""),
        "region_bbox": [float(b) for b in bbox],
        "halo_layers": halo,
        "n_cells": int(keep_nest.size),
        "variables": written,
        "time_range": ([str(tvals[0]), str(tvals[-1])]
                       if tvals is not None and tvals.size else []),
        "time_slice": [(-1 if _idx(k) is None else _idx(k))
                       for k in ("start", "stop", "step")],
        "created": datetime.now(timezone.utc).isoformat(),
        "tool": f"{TOOL} v{TOOL_VERSION}",
        "subset_name": str(meta_cfg.get("subset_name", "")),
        "description": str(meta_cfg.get("description", "")),
    })

    n_t = int(out.sizes.get("time", 0))
    print(f"{TOOL}: nside={nside} ({order} source) bbox={list(bbox)} "
          f"halo={halo} -> {keep_nest.size:,}/{npix:,} cells, "
          f"{n_t} time records, vars {names}")
    if dry_run:
        print("dry run — nothing written")
        return out_path

    # Drop encodings inherited from the source: kerchunk-GRIB variables
    # carry the gribscan read codec, which zarr rejects as a WRITE codec
    # (caught on the real ng5 store); compressors / chunking from the
    # source are equally stale for the subset.
    for v in out.variables:
        out[v].encoding.clear()
    chunks = {"time": int(out_cfg.get("time_chunk", 8))}
    out = out.chunk({k: v for k, v in chunks.items() if k in out.dims})
    if out_path.exists():
        import shutil
        shutil.rmtree(out_path)
    out.to_zarr(str(out_path), mode="w")
    with xr.open_zarr(str(out_path)) as written_ds:
        _check_lengths(written_ds, expect, "in the written store")
    print(f"wrote {out_path}")
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Extract a regional pixel subset from a HEALPix store "
                    "into a self-describing Zarr (docs/healpix.md §4).")
    ap.add_argument("config", help="YAML config path "
                    "(see utils/extract_healpix_example.yaml)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing output store")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the selection, write nothing")
    args = ap.parse_args(argv)
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh) or {}
    try:
        extract_region(cfg, dry_run=args.dry_run, force=args.force)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
