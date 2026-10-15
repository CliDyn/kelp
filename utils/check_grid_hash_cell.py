#!/usr/bin/env python
"""Check the nearest-node grid-hash cell for a run BEFORE launching it.

The engine finds each particle's nearest node by hashing the nodes into a
lon/lat grid and scanning the 3x3 window of cells around the query, twice per
RK2 step (four times per RK4 step).  The automatic cell size is derived from
the MEAN node spacing of the whole mesh, so on a mesh that is refined where the
particles are (the Arctic of CORE2, a coastal focus region, ...) the window
holds far more nodes than needed and the scan dominates the tracking phase.
Measured on the CORE2 Beaufort configuration (2026-09-12): auto cell 1.2-1.4 deg
= 100-135 nodes per window at the particles, 0.5 deg = 17 nodes -> the tracking
phase of the daily step fell by 25-43 % with identical results.

This tool loads the mesh exactly as the run will (same config, rotation
included), places the seeds, and for the automatic cell and each candidate
prints the nodes per 3x3 window at the seed positions (median / p90 / max),
the number of EMPTY windows, and — the decisive column — how many seeds have
their TRUE nearest node OUTSIDE the 3x3 window ("approx. anchors"): there the
scan returns a near-nearest node, the interpolation stencil changes, and the
trajectory diverges from the exact-anchor run (measured on the Beaufort run,
2026-09-12: 0.10 % approximate anchors at 0.5 deg -> 3 % of the positions and
0.16 % of the statuses differed after 91 days).  A window that is not empty
can still miss the true nearest node: in a rotated frame a degree of
longitude shrinks with cos(lat), so the true nearest can sit more than one
cell away in degrees.  Recommendation: the smallest candidate with NO
approximate anchor (and no empty window, median >= MIN_MEDIAN nodes); pass
``--allow-approx FRACTION`` to trade exactness for speed knowingly.  Put the
value in the config as ``mesh.grid_hash_cell_size`` (degrees).

Two query sets are probed: the SEEDS (where the run starts) and the ELEMENT
CENTRES of the whole mesh (a domain-wide proxy for wherever the particles may
drift — the Beaufort seeds showed no approximate anchor at 0.5 deg, the
particles that later reached the rotated pole over Greenland did).  The
recommendation must hold on both sets.

Usage (from any directory; the script finds the repository's ``src`` itself):
    python <clone>/utils/check_grid_hash_cell.py path/to/config.yaml [--seeds seeds.dat]
        [--cells 1.0 0.5 0.25] [--max-seeds 200000] [--allow-approx 0]
``--seeds`` defaults to ``data.inicloud`` of the config; when no default candidate
passes, widen ``--cells`` (e.g. ``--cells 0.8 0.6 0.4 0.3``).  Seeds are geographic
(lon lat [depth ...]); the tool rotates them into the mesh frame like the run.

Since 2026-09-22 every run applies this measurement itself
(``mesh.grid_hash_cell_rule: occupancy``, ``core.mesh.choose_grid_hash_cell`` on
sampled element centres, mode by the exact-nearest flag) and stamps the resolved
cell in the output metadata; this tool remains for explicit candidates,
``--allow-approx`` and the SEED positions (the run samples the mesh, not the
seeds).  Its last line prints what the run would choose.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from typing import Sequence

import numpy as np

# runnable from any directory: make the repository's src importable without PYTHONPATH
_SRC = __import__("pathlib").Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

MIN_MEDIAN = 8          # fewer nodes per window than this = too small a cell
TARGET_MEDIAN = 30      # more than this in the AUTO cell = worth pinning a cell


def window_counts(mesh, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Nodes in the 3x3 grid-hash window around each query (mesh frame)."""
    cs, lmn, ltm = mesh._gh_cell_size, mesh._gh_lon_min, mesh._gh_lat_min
    nx, ny, wrap = mesh._gh_nx, mesh._gh_ny, mesh._gh_wrap_lon
    counts = np.diff(mesh._gh_offsets)
    if wrap:
        ci = ((lon - lmn) / cs).astype(np.int64) % nx
    else:
        ci = np.clip(((lon - lmn) / cs).astype(np.int64), 0, nx - 1)
    cj = np.clip(((lat - ltm) / cs).astype(np.int64), 0, ny - 1)
    tot = np.zeros(lon.size, dtype=np.int64)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            cjj = np.clip(cj + dj, 0, ny - 1)
            cii = (ci + di) % nx if wrap else np.clip(ci + di, 0, nx - 1)
            tot += counts[cjj * nx + cii]
    return tot


def approx_anchor_count(mesh, lon: np.ndarray, lat: np.ndarray, nearest: np.ndarray,
                        cell: float) -> int:
    """How many queries have their TRUE nearest node (``nearest``, exact) outside the
    3x3 window of a hash with this cell (mesh-frame degrees, lon wrap-aware)."""
    lmn, ltm, wrap = mesh._gh_lon_min, mesh._gh_lat_min, mesh._gh_wrap_lon
    nlon, nlat = mesh.nodes_lon[nearest], mesh.nodes_lat[nearest]
    ci_p = np.floor((lon - lmn) / cell); ci_n = np.floor((nlon - lmn) / cell)
    cj_p = np.floor((lat - ltm) / cell); cj_n = np.floor((nlat - ltm) / cell)
    di = np.abs(ci_n - ci_p)
    if wrap:
        nx = np.ceil(360.0 / cell); di = np.minimum(di, nx - di)
    dj = np.abs(cj_n - cj_p)
    return int(((di > 1) | (dj > 1)).sum())


def probe_cells(mesh, lon: np.ndarray, lat: np.ndarray,
                cells: Sequence[float]) -> list[dict]:
    """Window statistics at the query points for the auto cell and each
    candidate; leaves the mesh with its ORIGINAL hash rebuilt."""
    from kelp.core.mesh import build_grid_hash, find_nearest_nodes
    pinned = (mesh.metadata or {}).get("grid_hash_cell")
    nearest = find_nearest_nodes(mesh, lon, lat)     # exact: grid hash + KD-tree heal (batch path)
    rows = []
    for cell in (0.0, *cells):
        build_grid_hash(mesh, float(cell))
        tot = window_counts(mesh, lon, lat)
        rows.append(dict(cell=float(mesh._gh_cell_size), auto=(cell == 0.0),
                         median=float(np.median(tot)), p90=float(np.percentile(tot, 90)),
                         max=int(tot.max()), empty=int((tot == 0).sum()), n=int(tot.size),
                         approx=approx_anchor_count(mesh, lon, lat, nearest, float(mesh._gh_cell_size))))
    build_grid_hash(mesh, float(pinned) if pinned else 0.0)   # restore
    return rows


#: under mesh.exact_nearest, the widened-search share tolerated by default
GUARD_APPROX_TOL = 0.10


def recommend(rows: list[dict], allow_approx: float = 0.0) -> tuple[float | None, str]:
    auto = next(r for r in rows if r["auto"])
    n = auto["n"]
    exact = lambda r: r["approx"] <= allow_approx * n
    ok = [r for r in rows if not r["auto"] and r["empty"] == 0 and r["median"] >= MIN_MEDIAN and exact(r)]
    note = "" if exact(auto) else (f" NOTE: even the auto cell leaves {auto['approx']} of {n} seeds "
                                   f"({100*auto['approx']/n:.3f} %) with an approximate anchor")
    if auto["median"] <= TARGET_MEDIAN:
        return None, (f"auto cell {auto['cell']:.3g} deg is fine: median {auto['median']:.0f} "
                      f"nodes per window (<= {TARGET_MEDIAN}); no grid_hash_cell_size needed" + note)
    if not ok:
        cand = [r for r in rows if not r["auto"]]
        return None, (f"auto cell leaves {auto['median']:.0f} nodes per window, but no candidate keeps the "
                      f"anchor exact (approx. anchors: " + ", ".join(f"{r['cell']:g} deg -> {r['approx']}" for r in cand) +
                      f"); try larger cells between {max(r['cell'] for r in cand):.3g} and {auto['cell']:.3g} deg, "
                      f"or accept a tolerance with --allow-approx" + note)
    best = min(ok, key=lambda r: r["cell"])
    trade = ", ".join(f"{r['cell']:g} deg -> {100*r['approx']/n:.2f} % approx." for r in rows
                      if not r["auto"] and r["cell"] < auto["cell"] and not exact(r))
    if best["cell"] >= auto["cell"]:
        return None, (f"keep the auto cell {auto['cell']:.3g} deg: no SMALLER candidate keeps the anchor exact "
                      f"(smaller cells trade exactness for a shorter scan: {trade or 'none probed'}); "
                      f"accept a fraction knowingly with --allow-approx (under mesh.exact_nearest every cell is "
                      f"exact and the approximate share only measures the widening cost)" + note)
    return best["cell"], (f"set mesh.grid_hash_cell_size: {best['cell']:g}  (median {best['median']:.0f} nodes per "
                          f"window instead of {auto['median']:.0f}; {best['approx']} approximate anchors of {n} probed"
                          f"{'' if allow_approx == 0 else f' (tolerance {allow_approx:g})'})" + note)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config")
    ap.add_argument("--seeds", default=None, help="seed file (default: data.inicloud)")
    ap.add_argument("--cells", type=float, nargs="+", default=[1.0, 0.5, 0.25],
                    help="candidate cell sizes in degrees")
    ap.add_argument("--max-seeds", type=int, default=200_000)
    ap.add_argument("--no-domain", action="store_true", help="probe the seeds only (skip the element-centre set)")
    ap.add_argument("--allow-approx", type=float, default=0.0,
                    help="accepted FRACTION of seeds with an approximate anchor (default 0 = exact only)")
    a = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    from kelp.core.config import load_config
    from kelp.core.particles import make_state
    from kelp.readers import get_reader
    from kelp.tracking import _maybe_rotate_seeds
    cfg = load_config(a.config)
    reader = get_reader(cfg.mesh.type)
    try:
        mesh = reader.load_mesh(cfg.mesh, mode_cfg=cfg.mode)
    except TypeError:
        mesh = reader.load_mesh(cfg.mesh)
    seeds = np.loadtxt(a.seeds or cfg.data.inicloud, ndmin=2)
    if seeds.shape[0] > a.max_seeds:
        seeds = seeds[:: int(np.ceil(seeds.shape[0] / a.max_seeds))]
    st = _maybe_rotate_seeds(make_state(seeds[:, 0].copy(), seeds[:, 1].copy()), mesh)
    sets = [("seeds", st.lon, st.lat)]
    if not a.no_domain:
        clon, clat = mesh.elem_center_lon, mesh.elem_center_lat            # already in the mesh frame
        if clon.size > a.max_seeds:
            step = int(np.ceil(clon.size / a.max_seeds)); clon, clat = clon[::step], clat[::step]
        sets.append(("element centres (domain-wide)", clon, clat))
    print(f"mesh: {mesh.nodes_lon.size} nodes; configured grid_hash_cell_size: {getattr(cfg.mesh, 'grid_hash_cell_size', None)}")
    merged = None
    for label, qlon, qlat in sets:
        rows = probe_cells(mesh, qlon, qlat, a.cells)
        print(f"-- query set: {label} ({qlon.size} points)")
        print(f"{'cell (deg)':>12s} {'nodes per 3x3 window: median':>28s} {'p90':>6s} {'max':>6s} {'empty windows':>14s} {'approx. anchors':>16s}")
        for r in rows:
            tag = " (auto)" if r["auto"] else ""
            print(f"{r['cell']:9.3f}{tag:7s} {r['median']:28.0f} {r['p90']:6.0f} {r['max']:6d} {r['empty']:8d} / {r['n']} {r['approx']:8d} ({100*r['approx']/r['n']:.3f} %)")
        if merged is None:
            merged = [dict(r) for r in rows]
        else:                       # the recommendation must hold on every query set: keep the worst of each
            for m, r in zip(merged, rows):
                m["median"] = min(m["median"], r["median"]); m["empty"] += r["empty"]; m["approx"] += r["approx"]; m["n"] += r["n"]
    rows = merged
    guard = bool(getattr(cfg.mesh, "exact_nearest", False))
    tol = a.allow_approx
    if guard and tol == 0.0:
        # under mesh.exact_nearest an approximate 3x3 hit costs a widened
        # in-kernel search, it is no longer a correctness problem: allow a
        # cost share by default and label the column accordingly
        tol = GUARD_APPROX_TOL
        print(f"mesh.exact_nearest is on: 'approx. anchors' = the share of anchors that widen the "
              f"in-kernel search (a cost, not an error); tolerating {100*tol:.0f} % (override with --allow-approx)")
    cell, msg = recommend(rows, tol)
    print("recommendation:", msg)
    # what the run itself would choose (the automatic rule, on element centres)
    if getattr(cfg.mesh, "grid_hash_cell_size", None) is not None:
        print(f"the run would use: the explicit cell {float(cfg.mesh.grid_hash_cell_size):g} deg "
              f"(mesh.grid_hash_cell_size; the rule is not consulted)")
        return 0
    try:
        from kelp.core.mesh import (apply_grid_hash_cell_rule, describe_grid_hash,
                                    set_grid_hash_cell, set_exact_nearest)
        auto_row = next((r for r in rows if r.get("auto")), None)
        if auto_row is not None:
            set_grid_hash_cell(mesh, auto_row["cell"])   # back to the reader's cell
        mesh.metadata.pop("grid_hash_cell", None)      # un-pin: let the rule decide
        mesh.metadata.pop("grid_hash_cell_rule", None)
        mesh.metadata.pop("grid_hash_cell_mode", None)
        set_exact_nearest(mesh, guard)
        apply_grid_hash_cell_rule(mesh, getattr(cfg.mesh, "grid_hash_cell_rule", "occupancy"))
        print("the run would use:", describe_grid_hash(mesh))
    except Exception as exc:                            # diagnostics only
        print(f"(the run's automatic rule could not be evaluated here: {exc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
