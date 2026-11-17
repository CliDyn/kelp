"""Per-step hooks of the exact C-grid path (``uv_interp: cgrid``, 2-D).

The mesh carries ONE ring of dry cells around the wet region as zero-velocity
elements (``readers.structured.build_cgrid_mesh(ring=True)``): a Runge-Kutta
stage that lands just on land then reads zero velocity instead of killing the
particle, while a stage beyond the ring still fails loudly.  Two hooks keep
every RECORDED position of an ALIVE particle in a wet cell:

* :func:`cgrid_pre_step` -- a particle that starts a step inside a ring cell
  (a seed on land) dies ``OUT_OF_BOUNDS``;
* :func:`cgrid_post_step` -- a particle whose step ENDS in a ring cell is
  pulled back along the step to the last wet point (bisection on the straight
  segment, classified by the very search the next step's pre-check uses), so
  it stops at the wall and the flow along the wall carries it on.

Known limitation: RK2 with its midpoint in the ring takes a zero midpoint
velocity, so the step moves nothing (a stall in a steady field); RK4 always
moves by at least ``dt*k1/6``.
"""
from __future__ import annotations

import numpy as np

from ..core.mesh import find_containing_elements, find_nearest_nodes
from ..core.particles import FailureReason, ParticleStatus

#: run counters (reset by the tracking loop, printed at the run end)
CGRID_STATS = {"pullbacks": 0, "start_in_ring": 0, "ring_stage_evals": 0}

_BISECT_ITERS = 32          # 2^-32 of a <= 1 km step ~ 0.2 micron


def reset_stats() -> None:
    for k in CGRID_STATS:
        CGRID_STATS[k] = 0


def stats_line() -> str:
    s = CGRID_STATS
    return (f"  cgrid dry ring: {s['pullbacks']} end-of-step pull-back(s), "
            f"{s['ring_stage_evals']} zero-velocity ring evaluation(s), "
            f"{s['start_in_ring']} start(s) in a ring cell")


def _wrap_lon(lon):
    return ((np.asarray(lon) + 180.0) % 360.0) - 180.0


def cgrid_classify(mesh, lon, lat):
    """``(wet, elem)`` of points: ``wet`` = inside a wet cell, ``elem`` the
    containing element (``-1`` outside every element, ring cells included)."""
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    if lon.size == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.int32)
    nn = find_nearest_nodes(mesh, lon, lat)
    e, ok = find_containing_elements(mesh, lon, lat, nn)
    e = np.where(ok, e, -1).astype(np.int32)
    wet = ok & (mesh._cgrid_nlev[np.maximum(e, 0)] > 0)
    return wet, e


def count_ring_seeds(mesh, lon, lat) -> int:
    """Number of points that lie in a ring (dry) cell."""
    wet, e = cgrid_classify(mesh, lon, lat)
    return int(((e >= 0) & ~wet).sum())


def count_ring_evals(mesh, lon, lat, U, V, ok, alive) -> int:
    """Zero-velocity evaluations that were in ring cells (stage diagnostics)."""
    cand = alive & ok & (U == 0.0) & (V == 0.0)
    if not cand.any():
        return 0
    return count_ring_seeds(mesh, lon[cand], lat[cand])


def cgrid_pre_step(state, mesh):
    """Kill alive particles that start the step inside a ring cell."""
    idx = np.flatnonzero(state.alive_mask)
    if idx.size == 0:
        return state
    wet, e = cgrid_classify(mesh, state.lon[idx], state.lat[idx])
    bad = idx[(e >= 0) & ~wet]
    if bad.size == 0:
        return state
    CGRID_STATS["start_in_ring"] += int(bad.size)
    status = state.status.copy()
    status[bad] = ParticleStatus.OUTSIDE
    fr = state.failure_reason
    if fr is not None:
        fr = fr.copy()
        fr[bad] = FailureReason.OUT_OF_BOUNDS
    return state.replace(status=status, failure_reason=fr)


def cgrid_settle(mesh, lon0, lat0, dlon, dlat):
    """Last wet point on the straight step ``P(f) = (lon0 + f dlon, lat0 +
    f dlat)``, ``f in [0, 1]``, for steps that start wet and end in a ring
    cell.  Bisection with ``cls(P(lo)) == WET`` kept as the invariant, so the
    returned point is classified wet by the same function as the next step's
    start check.  Returns ``(lon, lat, elem)``."""
    lo = np.zeros(lon0.shape[0])
    hi = np.ones(lon0.shape[0])
    for _ in range(_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        wet, _ = cgrid_classify(mesh, _wrap_lon(lon0 + mid * dlon),
                                lat0 + mid * dlat)
        lo = np.where(wet, mid, lo)
        hi = np.where(wet, hi, mid)
    lon = _wrap_lon(lon0 + lo * dlon)
    lat = lat0 + lo * dlat
    _, e = cgrid_classify(mesh, lon, lat)
    return lon, lat, e


def cgrid_post_step(state0, result, mesh):
    """Pull back particles whose step ended in a ring cell (see module doc)."""
    idx = np.flatnonzero(state0.alive_mask & result.alive_mask)
    if idx.size == 0:
        return result
    wet, e = cgrid_classify(mesh, result.lon[idx], result.lat[idx])
    sel = idx[(e >= 0) & ~wet]
    if sel.size == 0:
        return result
    lon0, lat0 = state0.lon[sel], state0.lat[sel]
    dlon = _wrap_lon(result.lon[sel] - lon0)
    dlat = result.lat[sel] - lat0
    lon, lat, e_new = cgrid_settle(mesh, lon0, lat0, dlon, dlat)
    CGRID_STATS["pullbacks"] += int(sel.size)
    new_lon = result.lon.copy(); new_lat = result.lat.copy()
    new_lon[sel] = lon; new_lat[sel] = lat
    nn = result.nearest_node.copy()
    nn[sel] = find_nearest_nodes(mesh, lon, lat)
    out = result.replace(lon=new_lon, lat=new_lat, nearest_node=nn)
    if result.elem is not None:
        el = result.elem.copy()
        el[sel] = np.where(e_new >= 0, e_new, state0.elem[sel] if state0.elem is not None else e_new)
        out = out.replace(elem=el.astype(np.int32))
    return out
