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

from kelp.readers.healpix import (_ETA_ALIASES, _LEVEL_DIMS,  # noqa: E402
                                      _LEVELS_VAR, _U_ALIASES, _V_ALIASES,
                                      _bbox_wet_mask, _derive_eta_sibling,
                                      _find_crs, _first, _first_soft,
                                      _open_healpix_store, _require_healpy,
                                      _resolve_levels,
                                      _validate_time_axes_match)

TOOL = "extract_healpix_region"
TOOL_VERSION = "1.0"


def _cell_dim(da):
    """The healpix cell dim = the last dim (ng5: ``value``; CMIP: ``cell``)."""
    return da.dims[-1]


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

    # --- variables -------------------------------------------------------
    # source.var_u / source.var_v: velocity names outside the alias lists
    # (a run on such an extract needs the same mode.var_u/var_v)
    var_u = src.get("var_u")
    var_v = src.get("var_v")
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
    else:
        u_name = _first(ds, _U_ALIASES, var_u)
        v_name = _first(ds, _V_ALIASES, var_v)
        if u_name is None or v_name is None:
            raise KeyError(
                f"source store {src_path!r}: no u/v variable found "
                f"(looked for {_U_ALIASES} / {_V_ALIASES}); list "
                "variables: explicitly or set source.var_u/var_v.")
        names = [u_name, v_name]

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
            bbox, halo, dry_run=dry_run)
    finally:
        if ds_eta is not None:
            ds_eta.close()


def _assemble_and_write(cfg, ds, ds_eta, hp, nside, order, npix, names,
                        eta_name, eta_source, time_cfg, out_cfg, meta_cfg,
                        src_path, out_path, bbox, halo, *, dry_run):
    import xarray as xr

    # bathymetry travels with the extract when the source carries one
    # (feeds the reader's wet-mask refinement + beaching depth criteria)
    from kelp.readers.healpix import _DEPTH_ALIASES
    d_name = _first_soft(ds, _DEPTH_ALIASES)
    if d_name is not None and d_name not in names:
        names.append(d_name)

    # --- pixel subset ----------------------------------------------------
    keep_nest = _pixel_selection(hp, nside, order, bbox, halo)
    if order == "ring":
        src_pos = hp.nest2ring(nside, keep_nest)   # row position per pixel
    else:
        src_pos = keep_nest

    # --- assemble the output dataset (lazy) ------------------------------
    def _idx(key):
        v = time_cfg.get(key)
        return None if v is None else int(v)   # NOT `or None`: 0 is valid

    tsl = slice(_idx("start"), _idx("stop"), _idx("step"))
    # region.max_levels: keep only the first K entries of the level axis
    # (renamed from a top-level 'levels' key — kept as an alias — to
    # disambiguate from source.levels, the interface-depth TABLE)
    max_lev = (cfg.get("region") or {}).get("max_levels",
                                            cfg.get("levels"))
    cd_main = _cell_dim(ds[names[0]])
    data = {}
    for nm in names:
        da = ds[nm] if (ds_eta is None or nm != eta_name) else ds_eta[nm]
        cd = _cell_dim(da)
        if int(da.sizes[cd]) != npix:
            raise ValueError(
                f"variable {nm!r}: cell dim {cd!r} has {da.sizes[cd]} "
                f"entries, expected npix={npix} for nside {nside}")
        if "time" in da.dims:
            da = da.isel(time=tsl)
        if max_lev is not None:
            for d in da.dims:
                if str(d).lower() in _LEVEL_DIMS:
                    da = da.isel({d: slice(0, int(max_lev))})
        da = da.isel({cd: src_pos})
        if cd != cd_main:
            # e.g. an eta sibling naming its cell axis differently —
            # unify, or the Dataset would grow two cell dims and
            # healpix_index would attach to only one
            da = da.rename({cd: cd_main})
        data[nm] = da
    out = xr.Dataset(data)

    # vertical interface table (source.levels: aux3d path or inline
    # metres) -> the extract is self-contained for 3-D runs;
    # sliced together with the level axis when `levels:` limits it
    lev_tab = None
    src = cfg.get("source") or {}
    if src.get("levels") is not None:
        lev_tab = _resolve_levels(src["levels"])
        if max_lev is not None:
            lev_tab = lev_tab[: int(max_lev)]
        lev_sizes = {int(out[nm].sizes[d])
                     for nm in names for d in out[nm].dims
                     if str(d).lower() in _LEVEL_DIMS}
        # K interfaces bound K-1 layers: a store may carry u/v on exactly
        # K-1 mid-layer entries (no padding level) alongside w on K
        # interfaces — both are valid against a K-interface table
        ok = {lev_tab.size, lev_tab.size - 1}
        if lev_sizes and not lev_sizes <= ok:
            raise ValueError(
                f"source.levels gives {lev_tab.size} interfaces but the "
                f"extracted level axes have sizes {sorted(lev_sizes)} — "
                f"expected {lev_tab.size} (interface/padded) or "
                f"{lev_tab.size - 1} (mid-layer) entries.")
        out[_LEVELS_VAR] = ("level_interface", lev_tab)
        out[_LEVELS_VAR].attrs.update({
            "long_name": "vertical interface depths",
            "units": "m", "positive": "down"})

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
        "variables": list(names),
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
