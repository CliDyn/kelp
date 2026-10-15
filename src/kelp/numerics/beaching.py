"""Beaching model (TODO 2.2): criteria + resuspension evaluation.

Post-step hook logic — pure numpy, runs once per tracking step over all
particles (output-rate cost; revisit with Numba only above ~1e6 particles —
the per-particle RNG loops are the first candidates).

Status transitions are applied by the TRACKING layer; this module only
computes boolean decisions:
  to_beach   — ALIVE particles that strand this step
  to_release — BEACHED particles that resuspend this step

Counter-RNG: same counter hash as diffusion (a keyed SplitMix64,
``_philox_uint64``), distinct component ids, keyed
by (seed, real_id, ORIGINAL particle index, global step_i, component) —
parallel- and restart-safe.  Stochastic probabilities use abs(dt)
(sign-symmetric for backward runs).

``H_raw = mesh.depth[nn] + vel.eta[nn]`` (when use_eta) is the UNCLAMPED
column height: the sigma machinery clamps columns to Dmin for geometry,
but drying detection must see raw near-zero/negative columns (tidal
flats).

Same-step flicker (accepted v1 behaviour): a particle remapped
OUTSIDE->BEACHED by the tracking hook enters this evaluation as BEACHED in
the same step and — when its anchor is already wet with min_wet_steps=1 —
may release immediately (a grazing contact).
"""
from __future__ import annotations

from typing import Any, NamedTuple, Optional

import numpy as np

from ..core.particles import ParticleStatus
from .diffusion import _philox_uint64

COMP_BEACH = 10
COMP_RESUSP = 11

RESUSP_NEVER, RESUSP_AUTO, RESUSP_STOCH, RESUSP_BOTH = 0, 1, 2, 3
_RESUSP_CODES = {"never": RESUSP_NEVER, "auto": RESUSP_AUTO,
                 "stochastic": RESUSP_STOCH,
                 "auto_or_stochastic": RESUSP_BOTH}


def _u01(seed, real_id, particle_i, step_i, component) -> float:
    """Uniform [0,1) from the counter hash (``_philox_uint64``).

    53-bit construction (h >> 11) / 2**53: float(h)/2**64 can round UP to
    exactly 1.0 for the top ~2048 uint64 values, breaking the half-open
    contract.
    """
    h = _philox_uint64(np.uint64(seed), np.uint64(real_id),
                       np.uint64(particle_i), np.uint64(step_i),
                       np.uint64(component))
    return float(np.uint64(h) >> np.uint64(11)) * (1.0 / 9007199254740992.0)


class BeachingStepArgs(NamedTuple):
    use_depth: bool
    use_coastal: bool
    use_stochastic: bool
    depth_min: float
    use_eta: bool
    zone_depth: float       # resolved (config None -> 2*depth_min)
    t_beach: float          # > 0 whenever use_stochastic (config-validated)
    resuspension: int       # RESUSP_* code
    rewet_hysteresis: float
    min_wet_steps: int
    t_resuspend: float      # > 0 whenever stochastic release (config-validated)
    seed: int
    real_id: int
    step_i: int
    dt: float               # signed ok; probabilities use abs()
    pid_offset: int = 0     # global index of particle 0 (sharded runs)


class BeachRunConfig(NamedTuple):
    """Per-run resolved beaching constants (config + derived).

    Built ONCE in ``run_single_realization``'s startup gate and threaded
    into ``_apply_beaching_hook`` — the house pattern (cf.
    ``DiffusionStepArgs``) of bundling per-run constants so the hook
    signature carries only per-step variability.
    """
    cfg: Any                            # the validated BeachingConfig object
    crit: frozenset                     # frozenset(cfg.criteria)
    need_topo: bool
    elem_coastal: Optional[np.ndarray]  # mesh._elem_is_coastal or None
    seed: int                           # resolved (fallback to diffusion.seed)
    zone: float                         # resolved (None -> 2*depth_min)
    pid_offset: int = 0                 # global index of particle 0 (shards)


def evaluate_beaching(state, mesh, vel, args: BeachingStepArgs, *,
                      dry_mask: Optional[np.ndarray] = None,
                      elem_is_coastal: Optional[np.ndarray] = None,
                      containing_elems: Optional[np.ndarray] = None,
                      wet_streak: Optional[np.ndarray] = None):
    """Evaluate beaching + resuspension decisions for one step.

    Returns ``(to_beach, to_release, wet_streak_updated)``; the caller
    applies status transitions.  ``wet_streak`` counts consecutive wet
    steps for BEACHED particles (int32, carried by the tracking loop;
    zeros when None).
    """
    P = state.lon.shape[0]
    alive = state.status == ParticleStatus.ALIVE
    beached = state.status == ParticleStatus.BEACHED
    nn = state.nearest_node
    valid_nn = nn >= 0

    h_raw = np.full(P, np.inf)
    sel = valid_nn
    h_raw[sel] = mesh.depth[nn[sel]]
    if args.use_eta:
        h_raw[sel] = h_raw[sel] + vel.eta[nn[sel]]

    coastal_contact = np.zeros(P, dtype=np.bool_)
    if elem_is_coastal is not None and containing_elems is not None:
        inside = containing_elems >= 0
        coastal_contact[inside] = elem_is_coastal[containing_elems[inside]]

    to_beach = np.zeros(P, dtype=np.bool_)
    if args.use_depth:
        dry = h_raw < args.depth_min
        if dry_mask is not None:
            # clip(-1 -> 0) is harmless: valid_nn masks the bogus row out
            dry |= valid_nn & (dry_mask[np.clip(nn, 0, None)] > 0.5)
        to_beach |= dry
    if args.use_coastal:
        to_beach |= coastal_contact
    if args.use_stochastic and args.t_beach > 0:
        in_zone = h_raw < args.zone_depth
        # zone includes coastal contact whenever topology data is supplied
        # (works with use_coastal=False: probabilistic-only coastal beaching)
        if elem_is_coastal is not None and containing_elems is not None:
            in_zone |= coastal_contact
        p = 1.0 - np.exp(-abs(args.dt) / args.t_beach)
        for i in np.nonzero(alive & in_zone)[0]:
            if _u01(args.seed, args.real_id, int(i) + int(args.pid_offset), args.step_i,
                    COMP_BEACH) < p:
                to_beach[i] = True
    to_beach &= alive

    # --- resuspension of BEACHED ---
    if wet_streak is None:
        wet_streak = np.zeros(P, dtype=np.int32)
    wet_streak = wet_streak.copy()
    to_release = np.zeros(P, dtype=np.bool_)
    if args.resuspension != RESUSP_NEVER and beached.any():
        rewet = h_raw > (args.depth_min + args.rewet_hysteresis)
        # The same wetness the beaching test uses: a node the dry mask
        # marks dry is never wet, and an unknown nearest node (h_raw = inf)
        # proves nothing (review 2026-09-22, F06: a permanently masked node
        # with a large nominal depth released its particle every step).
        rewet &= valid_nn
        if dry_mask is not None:
            rewet &= ~(dry_mask[np.clip(nn, 0, None)] > 0.5)
        wet_streak[beached & rewet] += 1
        wet_streak[beached & ~rewet] = 0
        if args.resuspension in (RESUSP_AUTO, RESUSP_BOTH):
            to_release |= beached & (wet_streak >= args.min_wet_steps)
        if args.resuspension in (RESUSP_STOCH, RESUSP_BOTH) \
                and args.t_resuspend > 0:
            p_r = 1.0 - np.exp(-abs(args.dt) / args.t_resuspend)
            # stochastic release ALWAYS water-gated (spec): a particle
            # cannot refloat without water.
            for i in np.nonzero(beached & rewet)[0]:
                if _u01(args.seed, args.real_id, int(i) + int(args.pid_offset), args.step_i,
                        COMP_RESUSP) < p_r:
                    to_release[i] = True
    wet_streak[to_release] = 0
    wet_streak[~beached] = 0
    return to_beach, to_release, wet_streak


__all__ = ["BeachingStepArgs", "BeachRunConfig", "evaluate_beaching",
           "COMP_BEACH", "COMP_RESUSP", "_u01",
           "RESUSP_NEVER", "RESUSP_AUTO", "RESUSP_STOCH", "RESUSP_BOTH",
           "_RESUSP_CODES"]
