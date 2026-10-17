"""Time integration schemes for Lagrangian particle advection.

All integrators share the signature::

    step(state, vel, mesh, dt, time_dir, layer=0, **kwargs) -> ParticleState

Position update uses the Fortran convention (drift-fesom-3d.f90 lines 383-384)::

    lon_new = lon + dt · UU / z0     where z0 = cos(lat·π/180) · METERS_PER_DEG_LAT
    lat_new = lat + dt · VV / METERS_PER_DEG_LAT

For 3D tracking (auto-detected when ``vel.w is not None``), the vertical
position is also updated::

    z_new = z + W · dt

with surface/bottom clamping.

Particles whose interpolation fails are marked OUTSIDE and reverted
to their pre-step position (matching Fortran fix_lon/fix_lat logic).

numba-cache-bust 2026-07-03: the leastsq cores this module inlines gained
the antimeridian gate; touching THIS file invalidates the fused kernels'
on-disk cache so a warm clone cannot keep serving the stale (seam-broken)
inlined callee after a pull.
"""
from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(*args, **kwargs):
        return range(*args)

from ..core.geometry import METERS_PER_DEG_LAT, DEG2RAD, wrap_lon, clamp_lat
from .interpolate import (
    interp_leastsq, interp_leastsq_single, VelocityField,
    _interp_leastsq_core, _interp_leastsq_core_tinterp,
    _interp_leastsq_core_pchip4,
    _interp_barycentric_core_tinterp,
    _interp_barycentric_core_tinterp_3d, _interp_barycentric_core_pchip4_3d,
    get_interpolator, UV_INTERP_USES_ELEMS,
)
from ..core.mesh import (
    Mesh, find_nearest_nodes, find_containing_elements,
    find_containing_elements_walk, grid_hash_dense_for_elements,
    _local_nearest_node, _find_containing_element_local,
    _grid_hash_nearest_widened, _grid_hash_nearest_3x3,
    _grid_hash_nearest_exact, _anchor_provable, _require_node_xyz,
)
from ..core.particles import ParticleState, ParticleStatus, FailureReason
# The fused 3-D RK2 kernels below inline the W barycentric twins, which take
# their projection origin from core.mesh._seam_safe_lon0 (dateline fix,
# 2026-09-24).  Numba re-validates a cached kernel against THIS file only:
# after changing a callee in vertical.py or mesh.py, touch this file or wipe
# src/**/__pycache__, or cached fused kernels keep the old callee.
from .vertical import (
    find_layer, _interp_w_linear_single, _interp_w_leastsq_single,
    W_NEAREST, W_LEASTSQ, W_BARYCENTRIC, W_IDW, _W_METHOD_MAP,
    _interp_w_linear_single_tinterp, _interp_w_linear_single_pchip4,
    _interp_w_bary_single_tinterp, _interp_w_bary_single_pchip4,
)
from .diffusion import (
    _fill_uniform_kicks, DiffusionStepArgs,
    _loop_em_combined_2d, _loop_em_split_rk_2d,
    _loop_az_kick_zlevel, _loop_az_kick_sigma,
    _scheme_to_int, _layout_to_int,
    SCHEME_EM, SCHEME_M1,
    sphere_displace_arrays, R_METRIC, _ANISO_LAT_CAP,
)


# ---------------------------------------------------------------------------
# Position update helpers
# ---------------------------------------------------------------------------

def _scatter(values: np.ndarray, idx: np.ndarray, P: int, fill: int = 0) -> np.ndarray:
    """Scatter a compact array back into a full-size array."""
    out = np.full(P, fill, dtype=values.dtype)
    out[idx] = values
    return out


def _z0(lat: np.ndarray) -> np.ndarray:
    """Metric scale factor: metres per degree longitude at given latitude.

    z0 = cos(lat · π/180) · METERS_PER_DEG_LAT,  clamped to avoid division by zero at poles.
    """
    return np.maximum(np.cos(np.asarray(lat, dtype=np.float64) * DEG2RAD) * METERS_PER_DEG_LAT, 1.0)


def _check_in_domain(mesh, lon_new, lat_new, nn):
    """Check which particles are still inside the mesh after displacement."""
    _, in_domain = find_containing_elements(mesh, lon_new, lat_new, nn)
    return in_domain


def _walk_update_elems(mesh, lons, lats, prev_elems, mask):
    """Edge-walk containment at ``(lons, lats)`` for the masked subset.

    The carried elements are warm HINTS, never trusted blindly: the walk
    re-validates containment at exactly these positions (diffusion kicks,
    restarts or beaching may have moved particles since the last
    validation) and self-corrects by crossing edges; any non-contained
    outcome fell back to the global machinery inside
    ``find_containing_elements_walk``, so verdicts equal
    ``elem_search: local``.

    Returns ``(elems_new, ok_full)``: masked entries hold the walked
    element (``-1`` when not contained — ready to feed the interpolator
    ``elems=`` hints, where ``-1`` re-enables the internal search);
    entries outside the mask keep their previous value.
    """
    P = lons.shape[0]
    elems_new = prev_elems.copy()
    ok_full = np.zeros(P, dtype=bool)
    idx = np.where(mask)[0]
    if idx.size > 0:
        e, ok = find_containing_elements_walk(
            mesh, lons[idx], lats[idx], prev_elems[idx])
        elems_new[idx] = e
        ok_full[idx] = ok
    return elems_new, ok_full


def _update_failure_reason(prev, masks_to_codes):
    """Stamp per-particle ``FailureReason`` codes onto a copy of ``prev``.

    Semantics: ``failure_reason`` is **sticky** for terminal events
    (i.e. ``status = OUTSIDE`` transitions).  Once a particle dies, we
    record *why* it died and don't touch the field again — even if the
    integrator on a later step still has the same particle in its
    failure mask, re-assigning the same code is a no-op.

    ``prev`` is either ``None`` (failure-reason tracking disabled by
    the caller; return None) or an int8 array of shape (P,) carrying
    the running per-particle reason.  ``masks_to_codes`` is a
    ``{FailureReason.X: bool_mask}`` mapping for *new* failures this
    step.  Masks must be disjoint per step.

    TODO 2.6.  Transient events (surface/bottom clamps that don't kill
    the particle) are not represented in this field by design — they
    would conflict with the sticky-on-death semantics.
    """
    if prev is None:
        return None
    out = prev.copy()
    for code, mask in masks_to_codes.items():
        out[mask] = code
    return out


def _merge_fused_failure(prev, fr_kernel):
    """Sticky-merge kernel failure codes into the opt-in array.

    ``fr_kernel`` is the int8 scratch every fused kernel fills with
    ``_FR_INTERP``/``_FR_OOB`` for the particles IT failed this step
    (zeros elsewhere).  ``prev is None`` = tracking disabled → None.
    """
    if prev is None:
        return None
    return _update_failure_reason(prev, {
        FailureReason.INTERPOLATION_FAILED: fr_kernel == _FR_INTERP,
        FailureReason.OUT_OF_BOUNDS: fr_kernel == _FR_OOB,
    })


def _apply_displacement(lon, lat, U, V, dt, mask, z0_lat=None):
    """Compute new lon/lat from velocity, only where mask is True.

    ``z0_lat`` selects the latitude for the longitude metric ``z0``: pass the
    stage/midpoint latitude for an RK update (the displacement base stays
    ``lon``/``lat``) so the midpoint velocity is converted with the midpoint
    metric; ``None`` uses ``lat`` (correct for Euler and stage starts).
    """
    z0 = _z0(lat if z0_lat is None else z0_lat)
    lon_new = np.where(mask, lon + dt * U / z0, lon)
    lat_new = np.where(mask, lat + dt * V / METERS_PER_DEG_LAT, lat)
    # Wrap/clamp ONLY the moved entries: a frozen particle (dead,
    # NOT_RELEASED) must come out verbatim — wrap_lon moves +180.0 to
    # -180.0 and can round a value next to the branch cut (2026-08
    # review; the fused kernels copy frozen positions verbatim and were
    # right).
    lon_new = np.where(mask, wrap_lon(lon_new), lon)
    lat_new = np.where(mask, clamp_lat(lat_new), lat)
    return lon_new, lat_new


def _reject_combined_em_on_rk(diffusion, integrator_name: str) -> None:
    """RK2/RK4 + EM/M1 needs an explicit ``"split_rk"`` opt-in.

    The composition rules: spatially-varying schemes ("em" / "m1")
    compose with Euler via ``"combined"`` and with RK2/RK4
    only via ``"split_rk"``.  Any other shape
    of em/m1 + RK is rejected here.
    """
    if diffusion is None:
        return
    if diffusion.scheme not in ("em", "m1"):
        return
    if diffusion.composition == "split_rk":
        return  # legitimate split-RK call — caller handles it
    raise NotImplementedError(
        f"diffusion.scheme={diffusion.scheme!r} on integrator "
        f"{integrator_name!r} requires composition='split_rk' (an explicit "
        "opt-in).  Variable-Kh schemes compose with Euler via 'combined' "
        "or with RK2/RK4 via 'split_rk' — see docs/diffusion.md §5."
    )


# Module-level set of (id(vel.kh_zonal), id(vel.kh_meridional))
# combinations we've already warned about.  Review finding #2:
# the multidimensional Milstein QV correction implemented here only
# matches the analytical M1 scheme when the diffusion tensor is
# *separable* (Kx=Kx(x), Ky=Ky(y)).  For non-separable variable Kh
# the cross-derivative iterated-integral terms are missing.  We can't
# cheaply prove separability on an unstructured mesh, so the policy
# is: warn once per realization whenever m1 + spatially-varying Kh is
# in use, with an exhortation to fall back to em for general 2-D Kh.
_M1_VARIABLE_KH_WARNED = set()


def _maybe_warn_m1_variable_kh(diffusion: DiffusionStepArgs,
                                vel: VelocityField) -> None:
    """Emit a one-time warning for m1 + non-trivially-varying Kh."""
    if diffusion.scheme != "m1":
        return
    kx = vel.kh_zonal
    ky = vel.kh_meridional
    if kx is None or ky is None:
        return
    # ``ptp`` (peak-to-peak) is a cheap proxy for "varies at all";
    # constant fields skip the warning since M1 reduces to EM there.
    if np.ptp(kx) <= 0.0 and np.ptp(ky) <= 0.0:
        return
    # Keyed by the run, not by the arrays: in temporal-interp mode every step
    # blends a fresh Kh array, and an id key re-issued the warning each step.
    key = (int(diffusion.seed), int(diffusion.real_id), str(diffusion.scheme))
    if key in _M1_VARIABLE_KH_WARNED:
        return
    _M1_VARIABLE_KH_WARNED.add(key)
    import warnings
    warnings.warn(
        "diffusion.scheme='m1' with spatially-varying Kh assumes a "
        "separable diagonal tensor (Kx=Kx(x), Ky=Ky(y)); the "
        "cross-derivative Milstein iterated-integral terms are not "
        "implemented in this phase.  For general non-separable 2-D Kh "
        "use scheme='em' instead.  This warning is emitted once per "
        "realization.",
        RuntimeWarning,
        stacklevel=2,
    )


def _zero_dead_velocity(state: ParticleState) -> ParticleState:
    """The all-dead early exit: no particle moves, and — like every kernel
    path, which writes 0 for the non-alive particles — the velocity
    diagnostics are 0, not the last values a particle had before it died.
    (A shard whose particles all died used to record a stale velocity where
    the unsharded run, with live particles elsewhere, recorded 0.)"""
    if state.u is None:
        return state
    return state.replace(u=np.zeros_like(state.u), v=np.zeros_like(state.v),
                         w=None if state.w is None else np.zeros_like(state.w))


def _apply_combined_em_step(state: ParticleState,
                            vel: VelocityField,
                            mesh: Mesh, dt: float, time_dir: float,
                            diffusion: DiffusionStepArgs,
                            *, Dmin: float = 0.5,
                            z0b_min: float = 0.1,
                            uv_interp: str = "leastsq") -> ParticleState:
    """Combined Euler + EM/M1 step.

    This *replaces* the deterministic Euler step when
    ``diffusion.composition == "combined"``.  It:

    1. invokes ``_loop_em_combined_2d`` to compute tentative new
       positions in metre space using the spatially-varying Kh fields
       carried on ``vel``;
    2. runs the standard ``find_nearest_nodes`` + ``_check_in_domain``
       re-validation;
    3. reverts particles that went off-mesh or whose interp failed,
       marking them ``OUTSIDE`` (matches the existing revert-on-failure
       policy the deterministic path uses; docs/diffusion.md §5).

    2D only: ``vel.w`` is ignored.  RK2/RK4 + em/m1 routes through
    ``rk2_step`` / ``rk4_step`` which raise an explicit error pointing
    to the split-RK path — this function never sees a 3D velocity.

    Bit-identical short-circuit: ``Kh = 0 / ∂K = 0`` is *not*
    short-circuited here because the gradient is field-derived and we
    don't know it's zero at construction time.  Instead the
    one-step-formula tests in B4 already proved the kernel reduces to
    deterministic Euler for ``Kh = 0`` field, and B5's regression test
    pins it bit-for-bit on the integrator side.
    """
    if vel.kh_zonal is None or vel.kh_meridional is None or vel.kh_layout is None:
        raise ValueError(
            "diffusion scheme 'em' / 'm1' requires VelocityField to carry "
            "kh_zonal, kh_meridional, kh_layout — check that the reader "
            "saw a non-disabled DiffusionConfig and that the dataset "
            "exposes Kh variables (defaults: Kh_zonal / Kh_meridional)."
        )
    if diffusion.dres_m <= 0.0:
        raise ValueError(
            f"diffusion.dres_m must be positive for em/m1 schemes, "
            f"got {diffusion.dres_m}; set DiffusionConfig.dres_m or rely "
            "on the tracking layer's mesh-derived default."
        )

    # The combined EM/M1 kernel evaluates the deterministic drift with the
    # element-centre leastsq core (diffusion._loop_em_combined_2d) — it
    # cannot honour uv_interp="barycentric" and cannot consume node-based
    # velocities (node arrays would be indexed by element id).  Spec
    # 2026-06-06: fail fast, no silent fallbacks.
    if uv_interp != "leastsq":
        raise ValueError(
            "diffusion.composition='combined' (em/m1 + euler) evaluates "
            "u/v with the element-centre leastsq core and cannot honour "
            f"uv_interp={uv_interp!r}; use integrator rk2/rk4 (split_rk) "
            "or uv_interp: leastsq.")
    if getattr(vel, "uv_layout", "element") == "node":
        raise ValueError(
            "diffusion.composition='combined' (em/m1 + euler) cannot "
            "consume node-based velocities (the leastsq drift core indexes "
            "element-centre arrays); use rk2/rk4 (split_rk) or "
            "element-layout velocities.")

    _maybe_warn_m1_variable_kh(diffusion, vel)

    alive = state.alive_mask
    if not np.any(alive):
        return _zero_dead_velocity(state)

    P = state.n_particles
    layout_id = _layout_to_int(vel.kh_layout)
    scheme_id = _scheme_to_int(diffusion.scheme)

    # Refresh nearest-node before the kernel.  ``make_state`` seeds
    # nn=-1 and the kernel needs a valid centre node for the LSQ
    # stencil.  Run the KD-tree once on the alive set; dead slots keep
    # whatever value they had (they're skipped anyway).
    nn_arr = state.nearest_node.copy().astype(np.int32)
    alive_idx = np.where(alive)[0]
    if alive_idx.size > 0:
        fresh_nn = find_nearest_nodes(
            mesh, state.lon[alive_idx], state.lat[alive_idx]
        )
        nn_arr[alive_idx] = fresh_nn.astype(np.int32)

    lon_out = state.lon.copy()
    lat_out = state.lat.copy()
    ok_out = np.zeros(P, dtype=np.bool_)

    _loop_em_combined_2d(
        P, state.lon, state.lat,
        alive, state.status,
        nn_arr,
        np.ascontiguousarray(vel.u, dtype=np.float64),
        np.ascontiguousarray(vel.v, dtype=np.float64),
        np.ascontiguousarray(vel.kh_zonal, dtype=np.float64),
        np.ascontiguousarray(vel.kh_meridional, dtype=np.float64),
        layout_id,
        float(diffusion.dres_m),
        scheme_id,
        float(time_dir * dt),  # signed dt for advection; |.| for ∂K drift + noise
        np.uint64(diffusion.seed),
        np.uint64(diffusion.real_id),
        np.uint64(diffusion.step_i),
        mesh.node_to_elems, mesh.node_to_elems_count, mesh.elem_nodes,
        mesh.elem_center_lon, mesh.elem_center_lat,
        mesh.nodes_lon, mesh.nodes_lat,
        lon_out, lat_out, ok_out,
        pid_offset=np.uint64(diffusion.pid_offset),
    )

    # Re-validate the survivors only; the kernel set lon_out=state.lon
    # for failed particles so we keep them in place automatically.
    nn_new = state.nearest_node.copy()
    in_domain_full = np.ones(P, dtype=bool)
    survivors_idx = np.where(ok_out)[0]
    if survivors_idx.size > 0:
        m_nn = find_nearest_nodes(
            mesh, lon_out[survivors_idx], lat_out[survivors_idx]
        )
        m_in = _check_in_domain(
            mesh, lon_out[survivors_idx], lat_out[survivors_idx], m_nn
        )
        nn_new[survivors_idx] = m_nn
        in_domain_full[survivors_idx] = m_in

    # Failure modes:
    #   * alive but interp failed (~ok_out)         → INTERPOLATION_FAILED
    #   * alive, interp ok, but landed outside     → OUT_OF_BOUNDS
    # Both revert + OUTSIDE.
    interp_failed = alive & ~ok_out
    domain_failed = alive & ok_out & ~in_domain_full
    failed = interp_failed | domain_failed
    new_status = state.status.copy()
    new_status[failed] = ParticleStatus.OUTSIDE

    lon_final = np.where(failed, state.lon, lon_out)
    lat_final = np.where(failed, state.lat, lat_out)
    nn_final = np.where(failed, state.nearest_node, nn_new)

    new_failure_reason = _update_failure_reason(
        state.failure_reason,
        {FailureReason.INTERPOLATION_FAILED: interp_failed,
         FailureReason.OUT_OF_BOUNDS: domain_failed},
    )

    return state.replace(
        lon=lon_final, lat=lat_final,
        z=state.z,  # 2D only — z untouched
        status=new_status, nearest_node=nn_final,
        failure_reason=new_failure_reason,
    )


# Module-level set of (real_id,) values we've already warned about for
# split_rk.  The policy: emit a one-time RuntimeWarning per
# realization explaining that split_rk on RK is a pragmatic kelp
# extension, not a rigorous SDE scheme.
_SPLIT_RK_WARNED = set()


def _maybe_warn_split_rk(diffusion: DiffusionStepArgs) -> None:
    """One-time warning per (seed, real_id, scheme) for split_rk.

    review C-#1: keying by ``real_id`` alone meant two
    consecutive ``run_tracking`` calls in the same Python process
    that both used ``real_id=0`` would silence the second run's
    warning.  Adding ``seed`` and ``scheme`` to the key gives a
    realistic per-run scope without requiring orchestration to
    explicitly clear state.
    """
    if diffusion.composition != "split_rk":
        return
    key = (int(diffusion.seed), int(diffusion.real_id), diffusion.scheme)
    if key in _SPLIT_RK_WARNED:
        return
    _SPLIT_RK_WARNED.add(key)
    import warnings
    warnings.warn(
        "diffusion.composition='split_rk' is a pragmatic kelp "
        "extension: the deterministic RK step runs first, then the "
        f"{diffusion.scheme!r} SDE kick is added on top.  This is not "
        "a rigorous reference SDE scheme — for the canonical "
        "Euler-Maruyama / Milstein form use composition='combined' "
        "with the Euler integrator.  This warning is emitted once per "
        "(seed, realization_id, scheme) tuple.",
        RuntimeWarning,
        stacklevel=3,
    )


def _pick_kh_eval_vel(vel: "VelocityField",
                      vel_next: Optional["VelocityField"],
                      dt_to_next: Optional[float],
                      dt: float,
                      kh_eval_position: str) -> "VelocityField":
    """Pick the VelocityField to sample Kh from for a split-RK kick.

    ``vel`` is the field at the start of the step (already lerped to
    ``sim_time`` by the tracking layer in temporal-interp mode).
    ``vel_next`` is the snapshot the step is moving TOWARD (the next
    file snapshot forward, the previous one backward), ``dt_to_next``
    the positive seconds from ``sim_time`` to that snapshot.

    For ``kh_eval_position="start"`` always returns ``vel``.  For
    ``"end"`` linearly interpolates Kh (and the rest of the field)
    to the end of the step (``sim_time + dt`` forward, ``sim_time -
    dt`` backward) when ``vel_next``/``dt_to_next`` are available;
    falls back to ``vel`` when they aren't (steady-state or no
    temporal info).  The fallback is a documented limitation —
    in steady-state runs Kh is constant in time anyway, and in
    discrete mode without ``vel_next`` the user has no other option.

    review C-#3: without this, split-RK + temporal_interp would
    sample Kh at the start snapshot even when the deterministic RK
    used the time-interpolated u/v.
    """
    if kh_eval_position == "start":
        return vel
    if vel_next is None or dt_to_next is None or dt_to_next <= 0.0:
        return vel
    alpha = min(dt / dt_to_next, 1.0)
    if alpha < 1e-12:
        return vel
    if alpha > 1.0 - 1e-12:
        return vel_next
    return vel.lerp(vel_next, alpha)


def _apply_split_rk_em_kick(pre_state: ParticleState,
                            post_det: ParticleState,
                            mesh: Mesh, dt: float, time_dir: float,
                            diffusion: DiffusionStepArgs,
                            vel: "VelocityField",
                            *, Dmin: float = 0.5,
                            z0b_min: float = 0.1) -> ParticleState:
    """Add an EM/M1 SDE kick on top of a deterministic RK step.

    The deterministic RK step (rk2/rk4) has already been performed by
    the caller and produced ``post_det``.  This function:

    1. Optionally re-evaluates Kh and ∂K at either the start
       (``pre_state``) or end (``post_det``) position, controlled by
       ``DiffusionStepArgs.kh_eval_position`` (default ``"end"``).
    2. Computes a Brownian + Itô-drift displacement in metres.
    3. Applies the displacement at the post-RK position.
    4. Re-validates and reverts particles that left the domain.

    Emits the split-RK ``RuntimeWarning`` once per realization (see
    ``_maybe_warn_split_rk``).
    """
    # 3D belt-and-braces: rk2_step/rk4_step already raise upstream when
    # they detect a 3D vel field, but a direct internal call would
    # bypass that check.  Reject here too — the kernel only knows how
    # to apply the 2D horizontal kick (review C-#2).
    if _is_3d(vel):
        raise NotImplementedError(
            "diffusion.composition='split_rk' is 2D-only: spatially "
            "varying horizontal Kh in 3D is not implemented.  "
            "See docs/diffusion.md §5."
        )
    if vel.kh_zonal is None or vel.kh_meridional is None or vel.kh_layout is None:
        raise ValueError(
            "diffusion.composition='split_rk' requires VelocityField to "
            "carry kh_zonal, kh_meridional, kh_layout — check that the "
            "reader saw a non-disabled DiffusionConfig and the dataset "
            "exposes Kh variables."
        )
    if diffusion.dres_m <= 0.0:
        raise ValueError(
            f"diffusion.dres_m must be positive for em/m1 schemes, "
            f"got {diffusion.dres_m}."
        )
    if diffusion.kh_eval_position not in ("start", "end"):
        raise ValueError(
            "diffusion.kh_eval_position must be 'start' or 'end', got "
            f"{diffusion.kh_eval_position!r}."
        )

    _maybe_warn_split_rk(diffusion)
    _maybe_warn_m1_variable_kh(diffusion, vel)

    alive = post_det.alive_mask
    if not np.any(alive):
        return post_det

    P = post_det.n_particles
    layout_id = _layout_to_int(vel.kh_layout)
    scheme_id = _scheme_to_int(diffusion.scheme)

    # Pick eval / apply arrays per kh_eval_position.
    if diffusion.kh_eval_position == "start":
        lons_eval = pre_state.lon
        lats_eval = pre_state.lat
    else:
        lons_eval = post_det.lon
        lats_eval = post_det.lat
    lons_apply = post_det.lon
    lats_apply = post_det.lat

    # Refresh nearest-node for the eval positions (the deterministic
    # step may have moved particles a long way for stiff RK4 + strong
    # currents; we cannot trust the carried nn).
    nn_arr = pre_state.nearest_node.copy().astype(np.int32)
    alive_idx = np.where(alive)[0]
    if alive_idx.size > 0:
        fresh_nn = find_nearest_nodes(
            mesh, lons_eval[alive_idx], lats_eval[alive_idx]
        )
        nn_arr[alive_idx] = fresh_nn.astype(np.int32)

    lon_out = post_det.lon.copy()
    lat_out = post_det.lat.copy()
    ok_out = np.zeros(P, dtype=np.bool_)

    _loop_em_split_rk_2d(
        P,
        np.ascontiguousarray(lons_eval, dtype=np.float64),
        np.ascontiguousarray(lats_eval, dtype=np.float64),
        np.ascontiguousarray(lons_apply, dtype=np.float64),
        np.ascontiguousarray(lats_apply, dtype=np.float64),
        alive,
        nn_arr,
        np.ascontiguousarray(vel.kh_zonal, dtype=np.float64),
        np.ascontiguousarray(vel.kh_meridional, dtype=np.float64),
        layout_id,
        float(diffusion.dres_m),
        scheme_id,
        float(time_dir * dt),
        np.uint64(diffusion.seed),
        np.uint64(diffusion.real_id),
        np.uint64(diffusion.step_i),
        mesh.node_to_elems, mesh.node_to_elems_count, mesh.elem_nodes,
        mesh.elem_center_lon, mesh.elem_center_lat,
        mesh.nodes_lon, mesh.nodes_lat,
        lon_out, lat_out, ok_out,
        pid_offset=np.uint64(diffusion.pid_offset),
    )

    # Re-validate the kicked positions.
    nn_new = post_det.nearest_node.copy()
    in_domain_full = np.ones(P, dtype=bool)
    survivors_idx = np.where(ok_out)[0]
    if survivors_idx.size > 0:
        m_nn = find_nearest_nodes(
            mesh, lon_out[survivors_idx], lat_out[survivors_idx]
        )
        m_in = _check_in_domain(
            mesh, lon_out[survivors_idx], lat_out[survivors_idx], m_nn
        )
        nn_new[survivors_idx] = m_nn
        in_domain_full[survivors_idx] = m_in

    # Failure modes:
    #   * Kh interp failed at eval position (~ok_out) → INTERPOLATION_FAILED
    #   * Kicked position is out-of-domain (~in_domain) → OUT_OF_BOUNDS
    # In both cases revert to ``pre_state`` (matching the
    # ``_apply_split_diffusion_kick`` policy of the uniform kick).
    interp_failed = alive & ~ok_out
    domain_failed = alive & ok_out & ~in_domain_full
    failed = interp_failed | domain_failed
    new_status = post_det.status.copy()
    new_status[failed] = ParticleStatus.OUTSIDE

    lon_final = np.where(failed, pre_state.lon, lon_out)
    lat_final = np.where(failed, pre_state.lat, lat_out)
    nn_final = np.where(failed, pre_state.nearest_node, nn_new)

    new_failure_reason = _update_failure_reason(
        post_det.failure_reason,
        {FailureReason.INTERPOLATION_FAILED: interp_failed,
         FailureReason.OUT_OF_BOUNDS: domain_failed},
    )

    return post_det.replace(
        lon=lon_final, lat=lat_final,
        z=post_det.z,
        status=new_status, nearest_node=nn_final,
        failure_reason=new_failure_reason,
    )


def _apply_split_diffusion_kick(pre_state: ParticleState,
                                post_det: ParticleState,
                                mesh: Mesh, dt: float,
                                diffusion: DiffusionStepArgs,
                                vel: "VelocityField",
                                *,
                                Dmin: float = 0.5,
                                z0b_min: float = 0.1,
                                sigma_col=None) -> ParticleState:
    """Apply a Brownian kick after a deterministic integrator step.

    This kick handles only ``scheme="uniform"`` (constant ``Kh``); ``em`` and
    ``m1`` reach it only through a hand-built ``DiffusionStepArgs`` and raise
    ``NotImplementedError`` below.

    The kick is added to ``post_det`` for every particle that survived
    the deterministic step.  Re-validation runs ``find_nearest_nodes``
    + ``_check_in_domain`` once on the kicked positions; particles
    whose kicked position lands outside the mesh are reverted to
    ``pre_state`` and marked ``OUTSIDE`` — the same revert-on-failure
    policy the deterministic path already uses (docs/diffusion.md §5).

    For 3D particles, the kick changes the horizontal location and
    therefore the local bottom (and sigma column).  After re-validation
    the surviving particles' depths are reclamped via ``_clamp_z`` so a
    deep-water particle kicked into shallow water cannot end up below
    the local bottom.  ``vel`` provides the eta context required by the
    sigma-coordinate column geometry; for z-level meshes ``vel`` is not
    consumed by the clamp (only ``mesh.depth`` matters).  On sigma meshes
    the steps pass ``sigma_col = (eta_end, w_interp)``: the kick keeps the
    particle's sigma fraction ``s = z / H_p`` (horizontal diffusion acts
    along the sigma surfaces) and clamps in the END-of-step column the
    deterministic step placed it in (review of PR #102).

    Bit-identical short-circuit: if both ``kh_x`` and ``kh_y`` are
    zero, return ``post_det`` unchanged.  This guarantees that a
    diffusion-enabled run with ``Kh=0`` matches the diffusion-disabled
    run byte-for-byte.
    """
    if diffusion.scheme != "uniform":
        # Variable-Kh schemes never run through the split kick.  On
        # Euler they take the combined dispatch; on RK2/RK4
        # they need "split_rk".  Reaching this branch means
        # a manually-constructed DiffusionStepArgs picked
        # ``scheme="em"|"m1"`` together with ``composition="split"``,
        # which is unsupported.
        raise NotImplementedError(
            f"diffusion.scheme={diffusion.scheme!r} with "
            f"composition='split' is unsupported; em/m1 must use "
            "composition='combined' on Euler or "
            "composition='split_rk' on RK2/RK4.  "
            "See docs/diffusion.md §5."
        )
    if diffusion.kh_x == 0.0 and diffusion.kh_y == 0.0:
        return post_det

    alive_post = post_det.alive_mask
    if not np.any(alive_post):
        return post_det

    P = post_det.n_particles

    # Counter-RNG kick: index i is the *original* ParticleState slot
    # (the counter-RNG keying rule), so particles that drop out of the
    # alive mask between steps still receive their own RNG stream.
    dx_m, dy_m = _fill_uniform_kicks(
        float(diffusion.kh_x), float(diffusion.kh_y), float(dt),
        np.uint64(diffusion.seed), np.uint64(diffusion.real_id),
        np.uint64(diffusion.step_i), alive_post, P,
        np.uint64(diffusion.pid_offset),
    )

    # The kick moves along the great circle (exponential map of the
    # tangent displacement), not along the lon/lat chart: the chart walk
    # lacks the metric drift -K tan(lat)/R and is not a diffusion on the
    # sphere (review 2026-09-22).  A diagonal Kx != Ky tensor adds the
    # extra northward drift of anisotropic_metric_drift_m.  Frozen
    # (non-alive) entries come back verbatim.
    kx = float(diffusion.kh_x)
    ky = float(diffusion.kh_y)
    if kx != ky:
        # anisotropic_metric_drift_m, vectorised (a per-particle Python call
        # cost 0.4 s per 1M particles and step — review 2026-09-23)
        la = np.clip(post_det.lat, -_ANISO_LAT_CAP, _ANISO_LAT_CAP)
        corr = (kx - ky) * np.tan(la * DEG2RAD) / R_METRIC * abs(float(dt))
        dy_m = np.where(alive_post, dy_m + corr, dy_m)
    lon_kicked, lat_kicked = sphere_displace_arrays(
        np.ascontiguousarray(post_det.lon, dtype=np.float64),
        np.ascontiguousarray(post_det.lat, dtype=np.float64),
        dx_m, dy_m, np.ascontiguousarray(alive_post))

    # Re-validate only the kicked particles.
    nn_new = post_det.nearest_node.copy()
    in_domain_full = np.ones(P, dtype=bool)  # untouched slots stay valid
    moved_idx = np.where(alive_post)[0]
    if moved_idx.size > 0:
        m_nn = find_nearest_nodes(mesh, lon_kicked[moved_idx], lat_kicked[moved_idx])
        m_in = _check_in_domain(mesh, lon_kicked[moved_idx], lat_kicked[moved_idx], m_nn)
        nn_new[moved_idx] = m_nn
        in_domain_full[moved_idx] = m_in

    kick_failed = alive_post & ~in_domain_full
    new_status = post_det.status.copy()
    new_status[kick_failed] = ParticleStatus.OUTSIDE

    lon_final = np.where(kick_failed, pre_state.lon, lon_kicked)
    lat_final = np.where(kick_failed, pre_state.lat, lat_kicked)
    z_final = np.where(kick_failed, pre_state.z, post_det.z)
    nn_final = np.where(kick_failed, pre_state.nearest_node, nn_new)

    # 3D: reclamp z in the column at the kicked position (z-level: the
    # nearest node's bottom; sigma with the step's column: keep s = z/H_p
    # and clamp in H_p at the new position), so a particle that was valid
    # in deep water cannot end up below the bottom in shallow water.  Skip
    # for 2D and for runs with no surviving kicks.
    if _is_3d(vel):
        survived = alive_post & ~kick_failed
        if np.any(survived):
            H_k = None
            if sigma_col is not None and mesh.vertical_coord == "sigma":
                eta_end, w_interp = sigma_col
                H_post = _sigma_col_H(mesh, post_det.lon, post_det.lat,
                                      post_det.nearest_node, eta_end,
                                      survived, w_interp, Dmin)
                H_k = _sigma_col_H(mesh, lon_final, lat_final, nn_final,
                                   eta_end, survived, w_interp, Dmin)
                z_final = np.where(survived, z_final / H_post * H_k, z_final)
            z_final = _clamp_z(
                z_final, vel, mesh, nn_final, survived,
                Dmin=Dmin, z0b_min=z0b_min, H_col=H_k,
            )

    # Brownian kick can only fail by leaving the domain (no interp
    # involvement), so all kick_failed particles get OUT_OF_BOUNDS.
    new_failure_reason = _update_failure_reason(
        post_det.failure_reason,
        {FailureReason.OUT_OF_BOUNDS: kick_failed},
    )

    return post_det.replace(
        lon=lon_final, lat=lat_final, z=z_final,
        status=new_status, nearest_node=nn_final,
        failure_reason=new_failure_reason,
    )


#: dummy (1, 1) field handed to the z-level Az loop when the run uses a
#: constant Az — keeps the Numba signature stable (always a 2-D array).
_AZ_DUMMY_FIELD = np.zeros((1, 1), dtype=np.float64)


def _reject_em_m1_az(diffusion: Optional[DiffusionStepArgs]) -> None:
    """Fail fast on the config-forbidden em/m1 + az combination.

    Called FIRST in every step function — before any all-dead early
    return, combined dispatch, or composition branch — so hand-built
    args raise deterministically instead of silently dropping the
    vertical kick (the em/m1 branches never reach the az hook).
    """
    if (diffusion is not None
            and getattr(diffusion, "az_enabled", False)
            and diffusion.scheme in ("em", "m1")):
        raise ValueError(
            "diffusion.az_enabled composes with scheme='uniform' only "
            f"(v1); got scheme={diffusion.scheme!r} — the em/m1 paths "
            "never apply the az kick"
        )


def _apply_split_az_kick(pre_state: ParticleState,
                         post_det: ParticleState,
                         mesh: Mesh, dt: float,
                         diffusion: Optional[DiffusionStepArgs],
                         vel: "VelocityField",
                         *,
                         time_dir: float = 1.0,
                         Dmin: float = 0.5,
                         z0b_min: float = 0.1,
                         sigma_col=None) -> ParticleState:
    """Apply the vertical Az kick after the deterministic step (+ the
    horizontal split kick), TODO 2.2b.  On sigma meshes the reflective
    bounds are the particle's END-of-step column (``sigma_col = (eta_end,
    w_interp)`` from the step; the nearest node's column from ``vel.eta``
    when absent).

    ``Δz = ∂Az/∂z·|dt| + sqrt(2·Az·|dt|)·ξ`` with the mandatory Itô drift
    (Visser 1997), reflective surface/bottom boundaries (specular fold —
    never a clamp, which would violate the well-mixed condition), drawn
    from the counter-RNG's ``COMP_VERTICAL`` component.  The kick moves
    only ``z`` and always stays inside the local column, so lon/lat/
    status/nearest_node pass through untouched and no re-validation is
    needed.

    Az source: ``vel.az`` (node-based ``(K, N)`` at interfaces, set by
    the reader for ``var_az`` runs) when present, else the scalar
    ``diffusion.az_const``.  Sigma meshes support the constant source
    only (v1); a sigma mesh with a field raises.

    Byte-identity short-circuits: ``diffusion is None``, ``az_enabled``
    false, or a zero source (no field and ``az_const == 0``) return
    ``post_det`` unchanged (same object).
    """
    if diffusion is None or not getattr(diffusion, "az_enabled", False):
        return post_det
    az_field = getattr(vel, "az", None)
    az_const = float(getattr(diffusion, "az_const", 0.0))
    if az_const < 0.0:
        # Config validates az_const >= 0; a hand-built negative value is
        # treated as "no source" (the sigma loop has no per-value guard
        # and would hit sqrt(negative)).
        az_const = 0.0
    if az_field is None and getattr(diffusion, "az_requires_field", False):
        raise ValueError(
            "diffusion.var_az is configured but the velocity payload "
            "carries no az field — refusing to silently fall back to "
            "az_const (check the reader / temporal-interp payload)"
        )
    if az_field is None and az_const == 0.0:
        return post_det
    if not _is_3d(vel):
        raise ValueError(
            "diffusion.az_enabled requires a 3-D run (Fdrift3D): the "
            "vertical Az kick is meaningless in 2-D — disable az_enabled "
            "or switch to 3-D tracking"
        )
    alive = post_det.alive_mask
    if not np.any(alive):
        return post_det

    # House convention (mirrors the em/m1 drift kernels, lines passing
    # ``float(time_dir * dt)``): tracking always hands a POSITIVE dt and
    # carries the direction in time_dir; the kernel receives the SIGNED dt
    # and takes |dt| for the Itô drift and the noise (the drift belongs to
    # the diffusion — review 2026-09-22).
    sdt = float(time_dir) * float(dt)
    z_out = np.empty(post_det.n_particles, dtype=np.float64)
    if mesh.vertical_coord == "sigma":
        if az_field is not None:
            raise ValueError(
                "diffusion.var_az is not supported on sigma meshes (v1) "
                "— use the constant az_const source instead"
            )
        if sigma_col is not None:
            eta_end, w_interp = sigma_col
            H_col = _sigma_col_H(mesh, post_det.lon, post_det.lat,
                                 post_det.nearest_node, eta_end, alive,
                                 w_interp, Dmin)
        else:
            nn_c = np.clip(post_det.nearest_node, 0, mesh.n_nodes - 1)
            H_col = np.asarray(mesh.depth, dtype=np.float64)[nn_c] + \
                np.asarray(vel.eta, dtype=np.float64)[nn_c]
        _loop_az_kick_sigma(
            post_det.z, alive, np.ascontiguousarray(H_col, dtype=np.float64),
            az_const, float(Dmin), float(z0b_min), sdt,
            np.uint64(diffusion.seed), np.uint64(diffusion.real_id),
            np.uint64(diffusion.step_i), z_out,
            np.uint64(diffusion.pid_offset),
        )
    else:
        has_field = az_field is not None
        field = az_field if has_field else _AZ_DUMMY_FIELD
        _loop_az_kick_zlevel(
            post_det.z, alive, post_det.nearest_node, has_field,
            np.ascontiguousarray(field), az_const,
            _get_z_levels(mesh), mesh.depth, sdt,
            np.uint64(diffusion.seed), np.uint64(diffusion.real_id),
            np.uint64(diffusion.step_i), z_out,
            np.uint64(diffusion.pid_offset),
        )
    return post_det.replace(z=z_out)


# ---------------------------------------------------------------------------
# 3D vertical helpers
# ---------------------------------------------------------------------------

def _is_3d(vel: VelocityField) -> bool:
    """Check if velocity field is 3D (has W and multi-layer u/v).

    Probes go through ``_probe_field`` so a LazyLerpField answers from
    its PARENTS: touching ``vel.w`` / ``vel.u`` directly would
    materialize two whole-field blends per step that the caller may
    never read (measured 30-38 ms/step of pure waste at 196k nodes x
    K=70 — 2026-08 review).
    """
    w = _probe_field(vel, "w")
    if w is None:
        return False
    u = _probe_field(vel, "u")
    return u is not None and u.ndim == 2


def _validate_time_aware_kwargs(vel_next, dt_to_next):
    """Reject partial specification of time-aware parameters.

    Either both vel_next and dt_to_next must be given (time-aware mode) or
    neither (steady mode).  Silent fallback would hide user mistakes.
    """
    if (vel_next is None) != (dt_to_next is None):
        raise ValueError(
            "vel_next and dt_to_next must be provided together "
            "(got vel_next=%r, dt_to_next=%r)" % (vel_next, dt_to_next)
        )


def _probe_field(vel, name):
    """Presence/shape probe that never materializes a LazyLerpField.

    For a lazy field the blend preserves the parents' shape, and a field
    is None iff EITHER parent lacks it (mirroring VelocityField.lerp) —
    so probing the parents answers shape questions for free.
    """
    parent_a = getattr(vel, "vel_a", None)
    if parent_a is None:
        return getattr(vel, name)
    fa = getattr(parent_a, name)
    fb = getattr(vel.vel_b, name)
    if fa is None or fb is None:
        return None
    return fa


def _validate_time_aware_inputs(vel, vel_next, dt_to_next, dt):
    """No-allocation validation of time-aware RK inputs.

    Extracted from _make_stage_vels so that fused paths can validate without
    paying the cost of materializing per-stage lerped VelocityFields.
    Field probes go through _probe_field so a LazyLerpField is not
    materialized just to look at shapes.
    """
    if vel_next is None and dt_to_next is None:
        return
    _validate_time_aware_kwargs(vel_next, dt_to_next)

    # Placement of u/v (a plain attribute on a LazyLerpField too, so no
    # blend); a field-like object without it counts as "layer".
    pa = getattr(vel, "uv_vertical", "layer")
    pb = getattr(vel_next, "uv_vertical", "layer")
    if pa != pb:
        raise ValueError(
            f"vel and vel_next uv_vertical mismatch: {pa!r} vs {pb!r}"
        )

    vel_u = _probe_field(vel, "u")
    vel_v = _probe_field(vel, "v")
    vel_eta = _probe_field(vel, "eta")
    vel_w = _probe_field(vel, "w")

    # Shape compatibility
    if vel_u.shape != vel_next.u.shape:
        raise ValueError(
            f"vel.u and vel_next.u shape mismatch: "
            f"{vel_u.shape} vs {vel_next.u.shape}"
        )
    if vel_v.shape != vel_next.v.shape:
        raise ValueError(
            f"vel.v and vel_next.v shape mismatch: "
            f"{vel_v.shape} vs {vel_next.v.shape}"
        )
    if vel_eta.shape != vel_next.eta.shape:
        raise ValueError(
            f"vel.eta and vel_next.eta shape mismatch: "
            f"{vel_eta.shape} vs {vel_next.eta.shape}"
        )
    if (vel_w is None) != (vel_next.w is None):
        raise ValueError(
            f"vel.w and vel_next.w must both be None or both arrays "
            f"(got {type(vel_w).__name__} vs {type(vel_next.w).__name__})"
        )
    if vel_w is not None and vel_w.shape != vel_next.w.shape:
        raise ValueError(
            f"vel.w and vel_next.w shape mismatch: "
            f"{vel_w.shape} vs {vel_next.w.shape}"
        )

    # Bracket constraints
    if dt_to_next <= 0.0:
        raise ValueError(f"dt_to_next must be > 0, got {dt_to_next}")
    if dt > dt_to_next + 1e-9:
        raise ValueError(
            f"step endpoint past next snapshot: "
            f"dt={dt:.6f}s > dt_to_next={dt_to_next:.6f}s"
        )


def _make_stage_vels(vel: VelocityField, vel_next: Optional[VelocityField],
                     dt_to_next: Optional[float], dt: float,
                     c_coeffs: tuple) -> list:
    """Build per-RK-stage VelocityFields with intra-substep time interpolation.

    Parameters
    ----------
    vel : VelocityField
        Velocity field at the start of the step (t_step).
    vel_next : VelocityField or None
        Velocity field at the snapshot the step is moving TOWARD: the
        next file snapshot for forward tracking, the PREVIOUS one for
        backward (the tracking layer passes the direction-correct
        neighbour on both the discrete and temporal-interp paths).
        If None → steady mode: every stage reuses ``vel``.
    dt_to_next : float or None
        Positive seconds from t_step to that snapshot.  Required if
        vel_next given.  Stages then sample t + c·dt forward and
        t - c·dt backward — the times the trajectory actually visits.
    dt : float
        Tracking step size (positive seconds).
    c_coeffs : tuple of float
        RK stage time fractions, e.g. (0, 0.5, 0.5, 1) for RK4 or
        (0, 0.5) for RK2 (midpoint), (0,) for Euler.

    Returns
    -------
    list[VelocityField]
        One VelocityField per stage. In steady mode all entries are the
        same ``vel`` object (no copy).

    Notes
    -----
    For ``alpha == 0`` returns ``vel`` directly (no copy).
    For ``alpha == 1`` returns ``vel_next`` directly.
    Asserts when ``dt > dt_to_next`` (would step past the bracket).
    """
    _validate_time_aware_inputs(vel, vel_next, dt_to_next, dt)
    if vel_next is None:
        return [vel] * len(c_coeffs)

    out = []
    for c in c_coeffs:
        alpha = c * dt / dt_to_next
        if alpha < 1e-12:
            out.append(vel)
        elif alpha > 1.0 - 1e-12:
            out.append(vel_next)
        else:
            out.append(vel.lerp(vel_next, alpha))
    return out


def _get_z_levels(mesh: Mesh) -> np.ndarray:
    """Get physical z-level boundaries from a z-level mesh."""
    return mesh.z_levels


def _compute_layers(mesh, vel, z, nn_full, alive, *, Dmin=0.5, z0b_min=0.1,
                    H_col=None):
    """Compute per-particle vertical layers for the active vertical backend
    (sigma: in the column ``H_col`` when given, else the nearest node's)."""
    if mesh.vertical_coord == "sigma":
        from .vertical_sigma import find_layers_sigma
        return find_layers_sigma(
            mesh.sigma, z, nn_full, mesh.depth, vel.eta, alive, Dmin=Dmin,
            H_particle=H_col,
        )
    from .vertical import find_layers
    z_levels = _get_z_levels(mesh)
    return find_layers(z_levels, z, nn_full, mesh.depth, alive)


def _compute_w(vel, mesh, z, nn_full, alive, time_dir,
               w_interp="nearest", lons=None, lats=None,
               *, Dmin=0.5, z0b_min=0.1):
    """Compute vertical velocity W at particle positions.

    Parameters
    ----------
    w_interp : str
        Interpolation method: "nearest", "leastsq", "barycentric", "idw".
        All four methods are supported on both z-level and sigma meshes.
    lons, lats : np.ndarray or None
        Particle positions.  Required for all methods except "nearest".
    """
    if mesh.vertical_coord == "sigma":
        from .vertical_sigma import interp_w_sigma_spatial
        return interp_w_sigma_spatial(
            vel.w, mesh.sigma, z, lons, lats, nn_full, alive,
            mesh.depth, vel.eta, Dmin=Dmin, time_dir=time_dir,
            method=w_interp, mesh=mesh,
        )
    z_levels = _get_z_levels(mesh)
    if w_interp == "nearest":
        from .vertical import interp_w_linear
        return interp_w_linear(vel.w, z_levels, z, nn_full, alive, time_dir)
    from .vertical import interp_w_spatial
    return interp_w_spatial(
        vel.w, z_levels, z, lons, lats, nn_full, alive, time_dir,
        method=w_interp, mesh=mesh,
    )


def _clamp_z(z, vel, mesh, nn_full, alive, *, Dmin=0.5, z0b_min=0.1,
             H_col=None):
    """Clamp particle depths to surface/bottom boundaries (sigma: the
    column ``H_col`` when given, else the nearest node's)."""
    if mesh.vertical_coord == "sigma":
        from .vertical_sigma import clamp_depth_sigma
        return clamp_depth_sigma(
            z, nn_full, mesh.depth, vel.eta, alive,
            Dmin=Dmin, z0b_min=z0b_min, H_particle=H_col,
        )
    from .vertical import clamp_depth
    z_levels = _get_z_levels(mesh)
    return clamp_depth(z, z_levels, nn_full, mesh.depth, alive)


# ---------------------------------------------------------------------------
# Sigma meshes: the step integrates s = z / H_p (review of PR #102)
# ---------------------------------------------------------------------------
#
# On sigma meshes the reader supplies W = -Wvel, the dia-sigma flux (positive
# down), and the particle's sigma fraction s = z / H_p (H_p = max(depth + eta,
# Dmin) interpolated at the particle like W) obeys ds/dt = W / H_p.  Every RK
# stage advances s and places the particle at z = s * H_p(x_stage, t_stage);
# the end of the step at z = s_new * H_p(x_new, t_end).  The surface motion
# and the flow along a sloping bottom are then carried exactly, for any
# temporal interpolation of eta; earlier versions added them as depth rates
# (first order when combined with a dia-sigma flux, lost on steady-fallback
# steps).

def _sigma_col_H(mesh, lons, lats, nn_full, eta, mask, w_interp, Dmin):
    """Column height ``max(depth + eta, Dmin)`` at the particles (sigma
    meshes), interpolated like W; 1.0 where ``mask`` is False."""
    from .vertical_sigma import sigma_column_height
    return sigma_column_height(mesh, lons, lats, nn_full, eta, mask,
                               method=w_interp, Dmin=Dmin)


def _eta_blend(eta0, eta1, c):
    """Surface elevation a fraction ``c`` of the way from ``eta0`` to
    ``eta1`` — the same blend as ``VelocityField.lerp``."""
    if c <= 0.0:
        return eta0
    if c >= 1.0:
        return eta1
    from ..core.io import _blend
    return _blend(np.asarray(eta0), np.asarray(eta1), float(c))


def _recon_eta(recon, dt, c):
    """Surface elevation of the PCHIP stage at ``c`` — bit-identical to
    ``make_stage_vels_recon(..., (c,))[0].eta`` without reconstructing the
    other fields."""
    from .temporal_reconstruct import _pchip_blend4
    rs, rt, rq, rd = recon
    t0, t1, t2, t3 = (float(t) for t in rt)
    tq = rq + rd * c * dt
    return _pchip_blend4(rs[0].eta, rs[1].eta, rs[2].eta, rs[3].eta,
                         t0, t1, t2, t3, tq)


def _sigma_end_eta(vel, vel_next, dt_to_next, dt, recon, eta_end,
                   eta_start=None):
    """Surface elevation at the END of a step on a sigma mesh: the tracking
    layer's ``eta_end`` (the surface the next step starts from), else the
    PCHIP reconstruction at c = 1, the blend towards ``vel_next``, else the
    start surface the step used (a steady surface; ``eta_start`` when given,
    e.g. the last step of a record that ends inside the step)."""
    if eta_end is not None:
        return eta_end
    if recon is not None:
        return _recon_eta(recon, dt, 1.0)
    if vel_next is not None and dt_to_next is not None and dt_to_next > 0:
        a = dt / dt_to_next
        if a < 1e-12:
            return vel.eta
        if a > 1.0 - 1e-12:
            return vel_next.eta
        return _eta_blend(vel.eta, vel_next.eta, a)
    return vel.eta if eta_start is None else eta_start


def _sigma_kick_col(state, vel, mesh, dt, is3d, w_interp, vel_next,
                    dt_to_next, recon, sigma_eta_end, sigma_eta_start=None):
    """``(eta_end, w_interp)`` for the diffusion kicks on a sigma mesh (the
    column the step just placed the particles in), else None."""
    if not (is3d and mesh.vertical_coord == "sigma"):
        return None
    return (_sigma_end_eta(vel, vel_next, dt_to_next, dt, recon,
                           sigma_eta_end, eta_start=sigma_eta_start),
            w_interp)


def _sigma_given(eta_given, own_eta):
    """The surface the tracking layer supplied for this instant (a pure
    function of the time, identical for the end of one step and the start of
    the next, and in a resumed run), else the step's own."""
    if eta_given is None:
        return own_eta
    if np.shape(eta_given) != np.shape(own_eta):
        raise ValueError(
            f"sigma step: supplied surface has shape {np.shape(eta_given)}, "
            f"the velocity field's eta {np.shape(own_eta)}")
    return eta_given


class _EtaView:
    """A velocity field seen with another surface elevation: the supplied
    start surface for stage 1, or a steady-fallback step's stage between the
    start and end surfaces."""

    def __init__(self, base, eta):
        self._base = base
        self.eta = eta

    def __getattr__(self, name):
        return getattr(self._base, name)


def _sigma_stage_field(stage_vel, eta):
    return stage_vel if eta is stage_vel.eta else _EtaView(stage_vel, eta)


# ---------------------------------------------------------------------------
# Fused Numba kernels — interpolation + displacement + domain check in one
# ---------------------------------------------------------------------------

_SLM = METERS_PER_DEG_LAT  # alias for Numba kernels
# failure_reason codes for the fused kernels (candidate 7), pinned to
# the enum at import time (plain ints so numba can bake them)
_FR_INTERP = int(FailureReason.INTERPOLATION_FAILED)   # 1
_FR_OOB = int(FailureReason.OUT_OF_BOUNDS)             # 2


# ---------------------------------------------------------------------------
# Anchor exactness inside the fused kernels (mesh.exact_nearest, 2026-09-12)
#
# Every fused kernel anchors each of its evaluation sites (start position,
# RK stage positions, final position) on a nearest node found in-kernel.
# Guard off: the historical widened 3x3 search (approximate on an empty
# window) — byte-identical to the pre-guard kernels.  Guard on: the exact
# in-kernel search `_grid_hash_nearest_exact` (3x3 accepted only when the
# certificate proves it, else a rectangle widened by the certificate's
# deficits) — the same answer the batch path's KD heal gives, with no
# runner round trip, so a step costs the widened scans of its unprovable
# anchors and nothing else.  (The fused 3-D RK2 barycentric kernels search
# exactly in-kernel under the guard too; their flag / runner-KD-heal
# handshake from 2026-08 serves only the guard-off empty-window case.)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _fused_site_anchor(
    lon_p, lat_p,
    gh_offsets, gh_node_ids, cell_size, lon_min, lat_min,
    nx, ny, wrap_lon, nodes_lon, nodes_lat, node_xyz, exact, far_rows,
):
    """Nearest-node anchor for one evaluation site of a fused kernel:
    ``exact=False`` the widened in-kernel window search (unchanged),
    ``exact=True`` the provably exact one (``far_rows`` = the mesh's row
    budget, ``mesh._gh_far_rows``)."""
    if not exact:
        return _grid_hash_nearest_widened(
            lon_p, lat_p, gh_offsets, gh_node_ids, cell_size,
            lon_min, lat_min, nx, ny, wrap_lon, nodes_lon, nodes_lat, node_xyz)
    return _grid_hash_nearest_exact(
        lon_p, lat_p, gh_offsets, gh_node_ids, cell_size,
        lon_min, lat_min, nx, ny, wrap_lon, nodes_lon, nodes_lat, node_xyz,
        far_rows)


@njit(parallel=True, cache=True)
def _fused_euler_step_2d(
    P, lons, lats, alive_mask, status_in,
    gh_offsets, gh_node_ids, gh_cell_size, gh_lon_min, gh_lat_min,
    gh_nx, gh_ny, gh_wrap_lon, gh_far_rows,
    node_to_elems_count, node_to_elems,
    elem_center_lon, elem_center_lat,
    elem_nodes, nodes_lon, nodes_lat, node_xyz,
    u_fi, v_fi,
    dt, time_dir,
    lon_out, lat_out, status_out, nn_out, U_out, V_out, fr_out, exact
):
    """Fused Euler step for 2D: nn → interp → displace → nn → domain check."""
    ALIVE = 1
    OUTSIDE = 3
    for i in prange(P):
        if not alive_mask[i]:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            status_out[i] = status_in[i]
            continue

        # 1. Find nearest node via grid hash
        nn_i = _fused_site_anchor(
            lons[i], lats[i],
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )

        # 2. Interpolate velocity
        UU, VV, ok = _interp_leastsq_core(
            lons[i], lats[i], nn_i, time_dir,
            node_to_elems_count, node_to_elems,
            elem_center_lon, elem_center_lat,
            u_fi, v_fi,
        )
        U_out[i] = UU
        V_out[i] = VV

        if not ok:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_INTERP
            nn_out[i] = nn_i
            continue

        # 3. Compute displacement
        lat_rad = lats[i] * DEG2RAD
        z0 = math.cos(lat_rad) * _SLM
        if z0 < 1.0:
            z0 = 1.0
        lon_new = lons[i] + dt * UU / z0
        lat_new = lats[i] + dt * VV / _SLM
        lon_new = ((lon_new + 180.0) % 360.0) - 180.0
        if lat_new > 90.0:
            lat_new = 90.0
        elif lat_new < -90.0:
            lat_new = -90.0

        # 4. Find nearest node at new position via grid hash
        nn_new = _fused_site_anchor(
            lon_new, lat_new,
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )

        # 5. Domain check
        _, in_dom = _find_containing_element_local(
            lon_new, lat_new, nn_new,
            node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat,
        )

        if not in_dom:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_OOB
            nn_out[i] = nn_i
        else:
            lon_out[i] = lon_new
            lat_out[i] = lat_new
            status_out[i] = ALIVE
            nn_out[i] = nn_new


@njit(parallel=True, cache=True)
def _fused_euler_step_2d_tinterp(
    P, lons, lats, alive_mask, status_in,
    gh_offsets, gh_node_ids, gh_cell_size, gh_lon_min, gh_lat_min,
    gh_nx, gh_ny, gh_wrap_lon, gh_far_rows,
    node_to_elems_count, node_to_elems,
    elem_center_lon, elem_center_lat,
    elem_nodes, nodes_lon, nodes_lat, node_xyz,
    u_a, v_a, u_b, v_b, alpha0,
    dt, time_dir,
    lon_out, lat_out, status_out, nn_out, U_out, V_out, fr_out, exact
):
    """Fused Euler step for 2D over a RAW snapshot pair.

    Euler is single-stage: its velocity is the start-of-step field, i.e.
    the bracket blend at alpha0.  The nested tinterp core with stage
    alpha = 0 reproduces the materialized field bitwise (the stage line
    collapses to ``1.0*ua0 + 0.0*u_t``), so this kernel equals
    _fused_euler_step_2d on a pre-blended field without ever building it.
    """
    ALIVE = 1
    OUTSIDE = 3
    for i in prange(P):
        if not alive_mask[i]:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            status_out[i] = status_in[i]
            continue

        # 1. Find nearest node via grid hash
        nn_i = _fused_site_anchor(
            lons[i], lats[i],
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )

        # 2. Interpolate velocity (bracket blend at alpha0, no stage move)
        UU, VV, ok = _interp_leastsq_core_tinterp(
            lons[i], lats[i], nn_i, time_dir, alpha0, 0.0,
            node_to_elems_count, node_to_elems,
            elem_center_lon, elem_center_lat,
            u_a, v_a, u_b, v_b, u_b, v_b,
        )
        U_out[i] = UU
        V_out[i] = VV

        if not ok:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_INTERP
            nn_out[i] = nn_i
            continue

        # 3. Compute displacement
        lat_rad = lats[i] * DEG2RAD
        z0 = math.cos(lat_rad) * _SLM
        if z0 < 1.0:
            z0 = 1.0
        lon_new = lons[i] + dt * UU / z0
        lat_new = lats[i] + dt * VV / _SLM
        lon_new = ((lon_new + 180.0) % 360.0) - 180.0
        if lat_new > 90.0:
            lat_new = 90.0
        elif lat_new < -90.0:
            lat_new = -90.0

        # 4. Find nearest node at new position via grid hash
        nn_new = _fused_site_anchor(
            lon_new, lat_new,
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )

        # 5. Domain check
        _, in_dom = _find_containing_element_local(
            lon_new, lat_new, nn_new,
            node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat,
        )

        if not in_dom:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_OOB
            nn_out[i] = nn_i
        else:
            lon_out[i] = lon_new
            lat_out[i] = lat_new
            status_out[i] = ALIVE
            nn_out[i] = nn_new


@njit(parallel=True, cache=True)
def _fused_euler_step_3d(
    P, lons, lats, z_particles, alive_mask, status_in,
    gh_offsets, gh_node_ids, gh_cell_size, gh_lon_min, gh_lat_min,
    gh_nx, gh_ny, gh_wrap_lon, gh_far_rows,
    node_to_elems_count, node_to_elems,
    elem_center_lon, elem_center_lat,
    elem_nodes, nodes_lon, nodes_lat, node_xyz,
    node_depths, z_levels,
    u_2d, v_2d, w_field,
    dt, time_dir, w_method,
    lon_out, lat_out, z_out, status_out, nn_out, U_out, V_out, W_out, fr_out, exact
):
    """Fused Euler step for 3D: nn → layer → interp → W → displace → nn → domain."""
    ALIVE = 1
    OUTSIDE = 3
    _W_NEAREST = 0
    _W_LEASTSQ = 1
    for i in prange(P):
        if not alive_mask[i]:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            z_out[i] = z_particles[i]
            status_out[i] = status_in[i]
            continue

        node_i = _fused_site_anchor(
            lons[i], lats[i],
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )
        layer_i = find_layer(z_levels, z_particles[i], node_depths[node_i])

        u_fi = u_2d[layer_i]
        v_fi = v_2d[layer_i]
        UU, VV, ok = _interp_leastsq_core(
            lons[i], lats[i], node_i, time_dir,
            node_to_elems_count, node_to_elems,
            elem_center_lon, elem_center_lat,
            u_fi, v_fi,
        )
        U_out[i] = UU
        V_out[i] = VV

        # W for EVERY alive particle, before the u/v verdict — the unfused
        # path computes it with mask=alive and it is also the vel_w
        # diagnostic of a particle that dies here (parity, 2026-09-12).
        if w_method == _W_LEASTSQ:
            WW, _ = _interp_w_leastsq_single(
                lons[i], lats[i], z_particles[i], node_i,
                w_field, z_levels, time_dir, node_depths,
                nodes_lon, nodes_lat,
                node_to_elems, node_to_elems_count, elem_nodes,
            )
        else:
            WW = _interp_w_linear_single(w_field, z_levels, z_particles[i], node_i, time_dir)
        W_out[i] = WW

        if not ok:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            z_out[i] = z_particles[i]
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_INTERP
            nn_out[i] = node_i
            continue

        lat_rad = lats[i] * DEG2RAD
        z0 = math.cos(lat_rad) * _SLM
        if z0 < 1.0:
            z0 = 1.0
        lon_new = lons[i] + dt * UU / z0
        lat_new = lats[i] + dt * VV / _SLM
        z_new = z_particles[i] + WW * dt
        lon_new = ((lon_new + 180.0) % 360.0) - 180.0
        if lat_new > 90.0:
            lat_new = 90.0
        elif lat_new < -90.0:
            lat_new = -90.0

        nn_new = _fused_site_anchor(
            lon_new, lat_new,
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )

        _, in_dom = _find_containing_element_local(
            lon_new, lat_new, nn_new,
            node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat,
        )

        if not in_dom:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            z_out[i] = z_particles[i]
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_OOB
            nn_out[i] = node_i
        else:
            lon_out[i] = lon_new
            lat_out[i] = lat_new
            z_clamped = z_new
            if z_clamped < z_levels[0]:
                z_clamped = z_levels[0]
            bd = node_depths[nn_new]
            if z_clamped > bd:
                z_clamped = bd
            z_out[i] = z_clamped
            status_out[i] = ALIVE
            nn_out[i] = nn_new


# ---------------------------------------------------------------------------
# Fused RK2 kernel
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _fused_rk2_step_2d(
    P, lons, lats, alive_mask, status_in,
    gh_offsets, gh_node_ids, gh_cell_size, gh_lon_min, gh_lat_min,
    gh_nx, gh_ny, gh_wrap_lon, gh_far_rows,
    node_to_elems_count, node_to_elems,
    elem_center_lon, elem_center_lat,
    elem_nodes, nodes_lon, nodes_lat, node_xyz,
    u_fi, v_fi,
    dt, time_dir,
    lon_out, lat_out, status_out, nn_out, U_out, V_out, fr_out, exact
):
    """Fused RK2 (midpoint) step for 2D."""
    ALIVE = 1
    OUTSIDE = 3
    for i in prange(P):
        if not alive_mask[i]:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            status_out[i] = status_in[i]
            continue

        node_i = _fused_site_anchor(
            lons[i], lats[i],
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )

        U1, V1, ok1 = _interp_leastsq_core(
            lons[i], lats[i], node_i, time_dir,
            node_to_elems_count, node_to_elems,
            elem_center_lon, elem_center_lat,
            u_fi, v_fi,
        )
        U_out[i] = U1
        V_out[i] = V1

        if not ok1:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_INTERP
            nn_out[i] = node_i
            continue

        lat_rad = lats[i] * DEG2RAD
        z0 = math.cos(lat_rad) * _SLM
        if z0 < 1.0:
            z0 = 1.0
        lon_mid = lons[i] + 0.5 * dt * U1 / z0
        lat_mid = lats[i] + 0.5 * dt * V1 / _SLM
        lon_mid = ((lon_mid + 180.0) % 360.0) - 180.0
        if lat_mid > 90.0:
            lat_mid = 90.0
        elif lat_mid < -90.0:
            lat_mid = -90.0

        nn_mid = _fused_site_anchor(
            lon_mid, lat_mid,
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )

        U2, V2, ok2 = _interp_leastsq_core(
            lon_mid, lat_mid, nn_mid, time_dir,
            node_to_elems_count, node_to_elems,
            elem_center_lon, elem_center_lat,
            u_fi, v_fi,
        )

        if not ok2:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_INTERP
            nn_out[i] = node_i
            continue

        z0m = math.cos(lat_mid * DEG2RAD) * _SLM
        if z0m < 1.0:
            z0m = 1.0
        lon_new = lons[i] + dt * U2 / z0m
        lat_new = lats[i] + dt * V2 / _SLM
        lon_new = ((lon_new + 180.0) % 360.0) - 180.0
        if lat_new > 90.0:
            lat_new = 90.0
        elif lat_new < -90.0:
            lat_new = -90.0

        nn_new = _fused_site_anchor(
            lon_new, lat_new,
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )
        _, in_dom = _find_containing_element_local(
            lon_new, lat_new, nn_new,
            node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat,
        )

        if not in_dom:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_OOB
            nn_out[i] = node_i
        else:
            lon_out[i] = lon_new
            lat_out[i] = lat_new
            status_out[i] = ALIVE
            nn_out[i] = nn_new


# ---------------------------------------------------------------------------
# Fused RK2 (time-aware) kernel
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _fused_rk2_step_2d_tinterp(
    P, lons, lats, alive_mask, status_in,
    gh_offsets, gh_node_ids, gh_cell_size, gh_lon_min, gh_lat_min,
    gh_nx, gh_ny, gh_wrap_lon, gh_far_rows,
    node_to_elems_count, node_to_elems,
    elem_center_lon, elem_center_lat,
    elem_nodes, nodes_lon, nodes_lat, node_xyz,
    u_a, v_a, u_b, v_b, u_t, v_t, alpha0,
    alpha_k1, alpha_k2,
    dt, time_dir,
    lon_out, lat_out, status_out, nn_out, U_out, V_out, fr_out, exact
):
    """Fused RK2 (midpoint) step for 2D with intra-substep time interpolation.

    Stage k uses velocity = (1 - alpha_k) * (u_a, v_a) + alpha_k * (u_b, v_b)
    where alpha_k = c_k * dt / dt_to_next, c_k = (0, 0.5).

    When alpha_k1 == alpha_k2 == 0 (steady input), output is numerically
    equivalent to _fused_rk2_step_2d(..., u_a, v_a, ...) within roundoff
    introduced by the extra blend multiplication; the regression test
    TestFusedVsUnfused covers the steady kernel only.
    """
    ALIVE = 1
    OUTSIDE = 3
    for i in prange(P):
        if not alive_mask[i]:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            status_out[i] = status_in[i]
            continue

        node_i = _fused_site_anchor(
            lons[i], lats[i],
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )

        # K1 at (lon, lat), time alpha_k1
        U1, V1, ok1 = _interp_leastsq_core_tinterp(
            lons[i], lats[i], node_i, time_dir, alpha0, alpha_k1,
            node_to_elems_count, node_to_elems,
            elem_center_lon, elem_center_lat,
            u_a, v_a, u_b, v_b, u_t, v_t,
        )
        U_out[i] = U1
        V_out[i] = V1

        if not ok1:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_INTERP
            nn_out[i] = node_i
            continue

        lat_rad = lats[i] * DEG2RAD
        z0 = math.cos(lat_rad) * _SLM
        if z0 < 1.0:
            z0 = 1.0
        lon_mid = lons[i] + 0.5 * dt * U1 / z0
        lat_mid = lats[i] + 0.5 * dt * V1 / _SLM
        lon_mid = ((lon_mid + 180.0) % 360.0) - 180.0
        if lat_mid > 90.0:
            lat_mid = 90.0
        elif lat_mid < -90.0:
            lat_mid = -90.0

        nn_mid = _fused_site_anchor(
            lon_mid, lat_mid,
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )

        # K2 at (lon_mid, lat_mid), time alpha_k2
        U2, V2, ok2 = _interp_leastsq_core_tinterp(
            lon_mid, lat_mid, nn_mid, time_dir, alpha0, alpha_k2,
            node_to_elems_count, node_to_elems,
            elem_center_lon, elem_center_lat,
            u_a, v_a, u_b, v_b, u_t, v_t,
        )

        if not ok2:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_INTERP
            nn_out[i] = node_i
            continue

        z0m = math.cos(lat_mid * DEG2RAD) * _SLM
        if z0m < 1.0:
            z0m = 1.0
        lon_new = lons[i] + dt * U2 / z0m
        lat_new = lats[i] + dt * V2 / _SLM
        lon_new = ((lon_new + 180.0) % 360.0) - 180.0
        if lat_new > 90.0:
            lat_new = 90.0
        elif lat_new < -90.0:
            lat_new = -90.0

        nn_new = _fused_site_anchor(
            lon_new, lat_new,
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )
        _, in_dom = _find_containing_element_local(
            lon_new, lat_new, nn_new,
            node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat,
        )

        if not in_dom:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_OOB
            nn_out[i] = node_i
        else:
            lon_out[i] = lon_new
            lat_out[i] = lat_new
            status_out[i] = ALIVE
            nn_out[i] = nn_new


# ---------------------------------------------------------------------------
# Fused RK4 kernel
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _fused_rk4_step_2d(
    P, lons, lats, alive_mask, status_in,
    gh_offsets, gh_node_ids, gh_cell_size, gh_lon_min, gh_lat_min,
    gh_nx, gh_ny, gh_wrap_lon, gh_far_rows,
    node_to_elems_count, node_to_elems,
    elem_center_lon, elem_center_lat,
    elem_nodes, nodes_lon, nodes_lat, node_xyz,
    u_fi, v_fi,
    dt, time_dir,
    lon_out, lat_out, status_out, nn_out, U_out, V_out, fr_out, exact
):
    """Fused RK4 step for 2D: all 4 stages per particle in one prange loop."""
    ALIVE = 1
    OUTSIDE = 3
    for i in prange(P):
        if not alive_mask[i]:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            status_out[i] = status_in[i]
            continue

        node_i = _fused_site_anchor(
            lons[i], lats[i],
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )
        lon0 = lons[i]
        lat0 = lats[i]

        # K1
        U1, V1, ok1 = _interp_leastsq_core(
            lon0, lat0, node_i, time_dir,
            node_to_elems_count, node_to_elems,
            elem_center_lon, elem_center_lat,
            u_fi, v_fi,
        )
        U_out[i] = U1
        V_out[i] = V1
        if not ok1:
            lon_out[i] = lon0
            lat_out[i] = lat0
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_INTERP
            nn_out[i] = node_i
            continue

        z0_1 = math.cos(lat0 * DEG2RAD) * _SLM
        if z0_1 < 1.0:
            z0_1 = 1.0
        dlon1 = dt * U1 / z0_1
        dlat1 = dt * V1 / _SLM

        # K2
        lon2 = ((lon0 + 0.5 * dlon1) + 180.0) % 360.0 - 180.0
        lat2 = lat0 + 0.5 * dlat1
        if lat2 > 90.0:
            lat2 = 90.0
        elif lat2 < -90.0:
            lat2 = -90.0
        nn2 = _fused_site_anchor(
            lon2, lat2,
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )
        U2, V2, ok2 = _interp_leastsq_core(
            lon2, lat2, nn2, time_dir,
            node_to_elems_count, node_to_elems,
            elem_center_lon, elem_center_lat, u_fi, v_fi,
        )
        if not ok2:
            lon_out[i] = lon0; lat_out[i] = lat0
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_INTERP; nn_out[i] = node_i; continue
        z0_2 = math.cos(lat2 * DEG2RAD) * _SLM
        if z0_2 < 1.0: z0_2 = 1.0
        dlon2 = dt * U2 / z0_2
        dlat2 = dt * V2 / _SLM

        # K3
        lon3 = ((lon0 + 0.5 * dlon2) + 180.0) % 360.0 - 180.0
        lat3 = lat0 + 0.5 * dlat2
        if lat3 > 90.0: lat3 = 90.0
        elif lat3 < -90.0: lat3 = -90.0
        nn3 = _fused_site_anchor(
            lon3, lat3,
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )
        U3, V3, ok3 = _interp_leastsq_core(
            lon3, lat3, nn3, time_dir,
            node_to_elems_count, node_to_elems,
            elem_center_lon, elem_center_lat, u_fi, v_fi,
        )
        if not ok3:
            lon_out[i] = lon0; lat_out[i] = lat0
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_INTERP; nn_out[i] = node_i; continue
        z0_3 = math.cos(lat3 * DEG2RAD) * _SLM
        if z0_3 < 1.0: z0_3 = 1.0
        dlon3 = dt * U3 / z0_3
        dlat3 = dt * V3 / _SLM

        # K4
        lon4 = ((lon0 + dlon3) + 180.0) % 360.0 - 180.0
        lat4 = lat0 + dlat3
        if lat4 > 90.0: lat4 = 90.0
        elif lat4 < -90.0: lat4 = -90.0
        nn4 = _fused_site_anchor(
            lon4, lat4,
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )
        U4, V4, ok4 = _interp_leastsq_core(
            lon4, lat4, nn4, time_dir,
            node_to_elems_count, node_to_elems,
            elem_center_lon, elem_center_lat, u_fi, v_fi,
        )
        if not ok4:
            lon_out[i] = lon0; lat_out[i] = lat0
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_INTERP; nn_out[i] = node_i; continue
        z0_4 = math.cos(lat4 * DEG2RAD) * _SLM
        if z0_4 < 1.0: z0_4 = 1.0
        dlon4 = dt * U4 / z0_4
        dlat4 = dt * V4 / _SLM

        # Weighted average
        dlon = (dlon1 + 2.0 * dlon2 + 2.0 * dlon3 + dlon4) / 6.0
        dlat = (dlat1 + 2.0 * dlat2 + 2.0 * dlat3 + dlat4) / 6.0
        lon_new = ((lon0 + dlon) + 180.0) % 360.0 - 180.0
        lat_new = lat0 + dlat
        if lat_new > 90.0: lat_new = 90.0
        elif lat_new < -90.0: lat_new = -90.0

        nn_new = _fused_site_anchor(
            lon_new, lat_new,
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )
        _, in_dom = _find_containing_element_local(
            lon_new, lat_new, nn_new,
            node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat,
        )

        if not in_dom:
            lon_out[i] = lon0
            lat_out[i] = lat0
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_OOB
            nn_out[i] = node_i
        else:
            lon_out[i] = lon_new
            lat_out[i] = lat_new
            status_out[i] = ALIVE
            nn_out[i] = nn_new


# ---------------------------------------------------------------------------
# Fused RK4 (time-aware) kernel
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _fused_rk4_step_2d_tinterp(
    P, lons, lats, alive_mask, status_in,
    gh_offsets, gh_node_ids, gh_cell_size, gh_lon_min, gh_lat_min,
    gh_nx, gh_ny, gh_wrap_lon, gh_far_rows,
    node_to_elems_count, node_to_elems,
    elem_center_lon, elem_center_lat,
    elem_nodes, nodes_lon, nodes_lat, node_xyz,
    u_a, v_a, u_b, v_b, u_t, v_t, alpha0,
    alpha_k1, alpha_k2, alpha_k3, alpha_k4,
    dt, time_dir,
    lon_out, lat_out, status_out, nn_out, U_out, V_out, fr_out, exact
):
    """Fused RK4 step for 2D with intra-substep time interpolation.

    Stage k uses velocity = (1 - alpha_k) * (u_a, v_a) + alpha_k * (u_b, v_b)
    where alpha_k = c_k * dt / dt_to_next, c_k = (0, 0.5, 0.5, 1).

    When all alphas are 0, this is mathematically equivalent (within roundoff)
    to _fused_rk4_step_2d(..., u_a, v_a, ...).  The steady kernel is preserved
    byte-for-byte elsewhere; this one is used only when vel_next is provided.
    """
    ALIVE = 1
    OUTSIDE = 3
    for i in prange(P):
        if not alive_mask[i]:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            status_out[i] = status_in[i]
            continue

        node_i = _fused_site_anchor(
            lons[i], lats[i],
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )
        lon0 = lons[i]
        lat0 = lats[i]

        # K1 at (lon0, lat0), time alpha_k1
        U1, V1, ok1 = _interp_leastsq_core_tinterp(
            lon0, lat0, node_i, time_dir, alpha0, alpha_k1,
            node_to_elems_count, node_to_elems,
            elem_center_lon, elem_center_lat,
            u_a, v_a, u_b, v_b, u_t, v_t,
        )
        U_out[i] = U1
        V_out[i] = V1
        if not ok1:
            lon_out[i] = lon0
            lat_out[i] = lat0
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_INTERP
            nn_out[i] = node_i
            continue

        z0_1 = math.cos(lat0 * DEG2RAD) * _SLM
        if z0_1 < 1.0:
            z0_1 = 1.0
        dlon1 = dt * U1 / z0_1
        dlat1 = dt * V1 / _SLM

        # K2 at (lon0 + dlon1/2, lat0 + dlat1/2), time alpha_k2
        lon2 = ((lon0 + 0.5 * dlon1) + 180.0) % 360.0 - 180.0
        lat2 = lat0 + 0.5 * dlat1
        if lat2 > 90.0: lat2 = 90.0
        elif lat2 < -90.0: lat2 = -90.0
        nn2 = _fused_site_anchor(
            lon2, lat2,
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )
        U2, V2, ok2 = _interp_leastsq_core_tinterp(
            lon2, lat2, nn2, time_dir, alpha0, alpha_k2,
            node_to_elems_count, node_to_elems,
            elem_center_lon, elem_center_lat,
            u_a, v_a, u_b, v_b, u_t, v_t,
        )
        if not ok2:
            lon_out[i] = lon0; lat_out[i] = lat0
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_INTERP; nn_out[i] = node_i; continue
        z0_2 = math.cos(lat2 * DEG2RAD) * _SLM
        if z0_2 < 1.0: z0_2 = 1.0
        dlon2 = dt * U2 / z0_2
        dlat2 = dt * V2 / _SLM

        # K3 at (lon0 + dlon2/2, lat0 + dlat2/2), time alpha_k3
        lon3 = ((lon0 + 0.5 * dlon2) + 180.0) % 360.0 - 180.0
        lat3 = lat0 + 0.5 * dlat2
        if lat3 > 90.0: lat3 = 90.0
        elif lat3 < -90.0: lat3 = -90.0
        nn3 = _fused_site_anchor(
            lon3, lat3,
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )
        U3, V3, ok3 = _interp_leastsq_core_tinterp(
            lon3, lat3, nn3, time_dir, alpha0, alpha_k3,
            node_to_elems_count, node_to_elems,
            elem_center_lon, elem_center_lat,
            u_a, v_a, u_b, v_b, u_t, v_t,
        )
        if not ok3:
            lon_out[i] = lon0; lat_out[i] = lat0
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_INTERP; nn_out[i] = node_i; continue
        z0_3 = math.cos(lat3 * DEG2RAD) * _SLM
        if z0_3 < 1.0: z0_3 = 1.0
        dlon3 = dt * U3 / z0_3
        dlat3 = dt * V3 / _SLM

        # K4 at (lon0 + dlon3, lat0 + dlat3), time alpha_k4
        lon4 = ((lon0 + dlon3) + 180.0) % 360.0 - 180.0
        lat4 = lat0 + dlat3
        if lat4 > 90.0: lat4 = 90.0
        elif lat4 < -90.0: lat4 = -90.0
        nn4 = _fused_site_anchor(
            lon4, lat4,
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )
        U4, V4, ok4 = _interp_leastsq_core_tinterp(
            lon4, lat4, nn4, time_dir, alpha0, alpha_k4,
            node_to_elems_count, node_to_elems,
            elem_center_lon, elem_center_lat,
            u_a, v_a, u_b, v_b, u_t, v_t,
        )
        if not ok4:
            lon_out[i] = lon0; lat_out[i] = lat0
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_INTERP; nn_out[i] = node_i; continue
        z0_4 = math.cos(lat4 * DEG2RAD) * _SLM
        if z0_4 < 1.0: z0_4 = 1.0
        dlon4 = dt * U4 / z0_4
        dlat4 = dt * V4 / _SLM

        # Weighted average
        dlon = (dlon1 + 2.0 * dlon2 + 2.0 * dlon3 + dlon4) / 6.0
        dlat = (dlat1 + 2.0 * dlat2 + 2.0 * dlat3 + dlat4) / 6.0
        lon_new = ((lon0 + dlon) + 180.0) % 360.0 - 180.0
        lat_new = lat0 + dlat
        if lat_new > 90.0: lat_new = 90.0
        elif lat_new < -90.0: lat_new = -90.0

        nn_new = _fused_site_anchor(
            lon_new, lat_new,
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )
        _, in_dom = _find_containing_element_local(
            lon_new, lat_new, nn_new,
            node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat,
        )

        if not in_dom:
            lon_out[i] = lon0
            lat_out[i] = lat0
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_OOB
            nn_out[i] = node_i
        else:
            lon_out[i] = lon_new
            lat_out[i] = lat_new
            status_out[i] = ALIVE
            nn_out[i] = nn_new


# ---------------------------------------------------------------------------
# Fused barycentric (NODE-layout) time-aware kernels
#
# Twins of the *_tinterp kernels above for uv_interp="barycentric" on
# node-layout velocities (FESOM2 unod/vnod, NEMO C->T): each stage blends the
# two snapshots inline per VERTEX via _interp_barycentric_core_tinterp instead
# of materializing a whole node-field lerp per stage on the unfused path.
# Node layout only (the dispatch gate enforces it); element-layout
# barycentric, walk-carried elems, steady and PCHIP runs stay unfused.
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _fused_rk2_step_2d_bary_tinterp(
    P, lons, lats, alive_mask, status_in,
    gh_offsets, gh_node_ids, gh_cell_size, gh_lon_min, gh_lat_min,
    gh_nx, gh_ny, gh_wrap_lon, gh_far_rows,
    node_to_elems_count, node_to_elems,
    elem_nodes, nodes_lon, nodes_lat, node_xyz,
    u_a, v_a, u_b, v_b, u_t, v_t, alpha0,
    alpha_k1, alpha_k2,
    dt, time_dir, r_earth,
    lon_out, lat_out, status_out, nn_out, U_out, V_out, fr_out, exact
):
    """Fused RK2 (midpoint) step, 2D barycentric u/v on NODE-layout fields,
    with intra-substep time interpolation (same stage-alpha convention as
    :func:`_fused_rk2_step_2d_tinterp`)."""
    ALIVE = 1
    OUTSIDE = 3
    for i in prange(P):
        if not alive_mask[i]:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            status_out[i] = status_in[i]
            continue

        node_i = _fused_site_anchor(
            lons[i], lats[i],
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )

        # K1 at (lon, lat), time alpha_k1
        U1, V1, ok1 = _interp_barycentric_core_tinterp(
            lons[i], lats[i], node_i, time_dir, alpha0, alpha_k1,
            node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat,
            u_a, v_a, u_b, v_b, u_t, v_t, r_earth,
        )
        U_out[i] = U1
        V_out[i] = V1

        if not ok1:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_INTERP
            nn_out[i] = node_i
            continue

        lat_rad = lats[i] * DEG2RAD
        z0 = math.cos(lat_rad) * _SLM
        if z0 < 1.0:
            z0 = 1.0
        lon_mid = lons[i] + 0.5 * dt * U1 / z0
        lat_mid = lats[i] + 0.5 * dt * V1 / _SLM
        lon_mid = ((lon_mid + 180.0) % 360.0) - 180.0
        if lat_mid > 90.0:
            lat_mid = 90.0
        elif lat_mid < -90.0:
            lat_mid = -90.0

        nn_mid = _fused_site_anchor(
            lon_mid, lat_mid,
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )

        # K2 at (lon_mid, lat_mid), time alpha_k2
        U2, V2, ok2 = _interp_barycentric_core_tinterp(
            lon_mid, lat_mid, nn_mid, time_dir, alpha0, alpha_k2,
            node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat,
            u_a, v_a, u_b, v_b, u_t, v_t, r_earth,
        )

        if not ok2:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_INTERP
            nn_out[i] = node_i
            continue

        z0m = math.cos(lat_mid * DEG2RAD) * _SLM
        if z0m < 1.0:
            z0m = 1.0
        lon_new = lons[i] + dt * U2 / z0m
        lat_new = lats[i] + dt * V2 / _SLM
        lon_new = ((lon_new + 180.0) % 360.0) - 180.0
        if lat_new > 90.0:
            lat_new = 90.0
        elif lat_new < -90.0:
            lat_new = -90.0

        nn_new = _fused_site_anchor(
            lon_new, lat_new,
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )
        _, in_dom = _find_containing_element_local(
            lon_new, lat_new, nn_new,
            node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat,
        )

        if not in_dom:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_OOB
            nn_out[i] = node_i
        else:
            lon_out[i] = lon_new
            lat_out[i] = lat_new
            status_out[i] = ALIVE
            nn_out[i] = nn_new


@njit(parallel=True, cache=True)
def _fused_rk4_step_2d_bary_tinterp(
    P, lons, lats, alive_mask, status_in,
    gh_offsets, gh_node_ids, gh_cell_size, gh_lon_min, gh_lat_min,
    gh_nx, gh_ny, gh_wrap_lon, gh_far_rows,
    node_to_elems_count, node_to_elems,
    elem_nodes, nodes_lon, nodes_lat, node_xyz,
    u_a, v_a, u_b, v_b, u_t, v_t, alpha0,
    alpha_k1, alpha_k2, alpha_k3, alpha_k4,
    dt, time_dir, r_earth,
    lon_out, lat_out, status_out, nn_out, U_out, V_out, fr_out, exact
):
    """Fused RK4 step, 2D barycentric u/v on NODE-layout fields, with
    intra-substep time interpolation (same stage-alpha convention as
    :func:`_fused_rk4_step_2d_tinterp`)."""
    ALIVE = 1
    OUTSIDE = 3
    for i in prange(P):
        if not alive_mask[i]:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            status_out[i] = status_in[i]
            continue

        node_i = _fused_site_anchor(
            lons[i], lats[i],
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )
        lon0 = lons[i]
        lat0 = lats[i]

        # K1 at (lon0, lat0), time alpha_k1
        U1, V1, ok1 = _interp_barycentric_core_tinterp(
            lon0, lat0, node_i, time_dir, alpha0, alpha_k1,
            node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat,
            u_a, v_a, u_b, v_b, u_t, v_t, r_earth,
        )
        U_out[i] = U1
        V_out[i] = V1
        if not ok1:
            lon_out[i] = lon0
            lat_out[i] = lat0
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_INTERP
            nn_out[i] = node_i
            continue

        z0_1 = math.cos(lat0 * DEG2RAD) * _SLM
        if z0_1 < 1.0:
            z0_1 = 1.0
        dlon1 = dt * U1 / z0_1
        dlat1 = dt * V1 / _SLM

        # K2 at (lon0 + dlon1/2, lat0 + dlat1/2), time alpha_k2
        lon2 = ((lon0 + 0.5 * dlon1) + 180.0) % 360.0 - 180.0
        lat2 = lat0 + 0.5 * dlat1
        if lat2 > 90.0: lat2 = 90.0
        elif lat2 < -90.0: lat2 = -90.0
        nn2 = _fused_site_anchor(
            lon2, lat2,
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )
        U2, V2, ok2 = _interp_barycentric_core_tinterp(
            lon2, lat2, nn2, time_dir, alpha0, alpha_k2,
            node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat,
            u_a, v_a, u_b, v_b, u_t, v_t, r_earth,
        )
        if not ok2:
            lon_out[i] = lon0; lat_out[i] = lat0
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_INTERP; nn_out[i] = node_i; continue
        z0_2 = math.cos(lat2 * DEG2RAD) * _SLM
        if z0_2 < 1.0: z0_2 = 1.0
        dlon2 = dt * U2 / z0_2
        dlat2 = dt * V2 / _SLM

        # K3 at (lon0 + dlon2/2, lat0 + dlat2/2), time alpha_k3
        lon3 = ((lon0 + 0.5 * dlon2) + 180.0) % 360.0 - 180.0
        lat3 = lat0 + 0.5 * dlat2
        if lat3 > 90.0: lat3 = 90.0
        elif lat3 < -90.0: lat3 = -90.0
        nn3 = _fused_site_anchor(
            lon3, lat3,
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )
        U3, V3, ok3 = _interp_barycentric_core_tinterp(
            lon3, lat3, nn3, time_dir, alpha0, alpha_k3,
            node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat,
            u_a, v_a, u_b, v_b, u_t, v_t, r_earth,
        )
        if not ok3:
            lon_out[i] = lon0; lat_out[i] = lat0
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_INTERP; nn_out[i] = node_i; continue
        z0_3 = math.cos(lat3 * DEG2RAD) * _SLM
        if z0_3 < 1.0: z0_3 = 1.0
        dlon3 = dt * U3 / z0_3
        dlat3 = dt * V3 / _SLM

        # K4 at (lon0 + dlon3, lat0 + dlat3), time alpha_k4
        lon4 = ((lon0 + dlon3) + 180.0) % 360.0 - 180.0
        lat4 = lat0 + dlat3
        if lat4 > 90.0: lat4 = 90.0
        elif lat4 < -90.0: lat4 = -90.0
        nn4 = _fused_site_anchor(
            lon4, lat4,
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )
        U4, V4, ok4 = _interp_barycentric_core_tinterp(
            lon4, lat4, nn4, time_dir, alpha0, alpha_k4,
            node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat,
            u_a, v_a, u_b, v_b, u_t, v_t, r_earth,
        )
        if not ok4:
            lon_out[i] = lon0; lat_out[i] = lat0
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_INTERP; nn_out[i] = node_i; continue
        z0_4 = math.cos(lat4 * DEG2RAD) * _SLM
        if z0_4 < 1.0: z0_4 = 1.0
        dlon4 = dt * U4 / z0_4
        dlat4 = dt * V4 / _SLM

        # Weighted average
        dlon = (dlon1 + 2.0 * dlon2 + 2.0 * dlon3 + dlon4) / 6.0
        dlat = (dlat1 + 2.0 * dlat2 + 2.0 * dlat3 + dlat4) / 6.0
        lon_new = ((lon0 + dlon) + 180.0) % 360.0 - 180.0
        lat_new = lat0 + dlat
        if lat_new > 90.0: lat_new = 90.0
        elif lat_new < -90.0: lat_new = -90.0

        nn_new = _fused_site_anchor(
            lon_new, lat_new,
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )
        _, in_dom = _find_containing_element_local(
            lon_new, lat_new, nn_new,
            node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat,
        )

        if not in_dom:
            lon_out[i] = lon0
            lat_out[i] = lat0
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_OOB
            nn_out[i] = node_i
        else:
            lon_out[i] = lon_new
            lat_out[i] = lat_new
            status_out[i] = ALIVE
            nn_out[i] = nn_new


# ---------------------------------------------------------------------------
# Fused PCHIP (4-snapshot monotone-cubic) kernels
#
# Twins of the *_tinterp kernels above, but each stage samples a 4-snapshot
# Fritsch-Carlson cubic in time (inline, per element) instead of the
# 2-snapshot linear blend.  This gives the PCHIP path the same fused speed the
# linear path already has — no per-stage full-field materialization.  Result
# is equivalent to the unfused make_stage_vels_recon -> reconstruct_pchip path
# element-for-element (pinned by test_temporal_reconstruction.py).
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _fused_rk2_step_2d_pchip(
    P, lons, lats, alive_mask, status_in,
    gh_offsets, gh_node_ids, gh_cell_size, gh_lon_min, gh_lat_min,
    gh_nx, gh_ny, gh_wrap_lon, gh_far_rows,
    node_to_elems_count, node_to_elems,
    elem_center_lon, elem_center_lat,
    elem_nodes, nodes_lon, nodes_lat, node_xyz,
    u0, v0, u1, v1, u2, v2, u3, v3,
    t0, t1, t2, t3,
    tq_k1, tq_k2,
    dt, time_dir,
    lon_out, lat_out, status_out, nn_out, U_out, V_out, fr_out, exact
):
    """Fused RK2 (midpoint) step for 2D with 4-snapshot PCHIP time blend.

    Stage k samples the monotone-cubic velocity at query time ``tq_k`` (the
    central interval ``[t1, t2]``), reconstructed inline from the four
    snapshot value arrays ``u0..u3``/``v0..v3``.  ``tq_k = q_time +
    time_dir * c_k * dt`` with ``c_k = (0, 0.5)`` — computed by the wrapper.
    """
    ALIVE = 1
    OUTSIDE = 3
    for i in prange(P):
        if not alive_mask[i]:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            status_out[i] = status_in[i]
            continue

        node_i = _fused_site_anchor(
            lons[i], lats[i],
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )

        # K1 at (lon, lat), query time tq_k1
        U1, V1, ok1 = _interp_leastsq_core_pchip4(
            lons[i], lats[i], node_i, time_dir, t0, t1, t2, t3, tq_k1,
            node_to_elems_count, node_to_elems,
            elem_center_lon, elem_center_lat,
            u0, v0, u1, v1, u2, v2, u3, v3,
        )
        U_out[i] = U1
        V_out[i] = V1

        if not ok1:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_INTERP
            nn_out[i] = node_i
            continue

        lat_rad = lats[i] * DEG2RAD
        z0 = math.cos(lat_rad) * _SLM
        if z0 < 1.0:
            z0 = 1.0
        lon_mid = lons[i] + 0.5 * dt * U1 / z0
        lat_mid = lats[i] + 0.5 * dt * V1 / _SLM
        lon_mid = ((lon_mid + 180.0) % 360.0) - 180.0
        if lat_mid > 90.0:
            lat_mid = 90.0
        elif lat_mid < -90.0:
            lat_mid = -90.0

        nn_mid = _fused_site_anchor(
            lon_mid, lat_mid,
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )

        # K2 at (lon_mid, lat_mid), query time tq_k2
        U2, V2, ok2 = _interp_leastsq_core_pchip4(
            lon_mid, lat_mid, nn_mid, time_dir, t0, t1, t2, t3, tq_k2,
            node_to_elems_count, node_to_elems,
            elem_center_lon, elem_center_lat,
            u0, v0, u1, v1, u2, v2, u3, v3,
        )

        if not ok2:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_INTERP
            nn_out[i] = node_i
            continue

        z0m = math.cos(lat_mid * DEG2RAD) * _SLM
        if z0m < 1.0:
            z0m = 1.0
        lon_new = lons[i] + dt * U2 / z0m
        lat_new = lats[i] + dt * V2 / _SLM
        lon_new = ((lon_new + 180.0) % 360.0) - 180.0
        if lat_new > 90.0:
            lat_new = 90.0
        elif lat_new < -90.0:
            lat_new = -90.0

        nn_new = _fused_site_anchor(
            lon_new, lat_new,
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )
        _, in_dom = _find_containing_element_local(
            lon_new, lat_new, nn_new,
            node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat,
        )

        if not in_dom:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_OOB
            nn_out[i] = node_i
        else:
            lon_out[i] = lon_new
            lat_out[i] = lat_new
            status_out[i] = ALIVE
            nn_out[i] = nn_new


@njit(parallel=True, cache=True)
def _fused_rk4_step_2d_pchip(
    P, lons, lats, alive_mask, status_in,
    gh_offsets, gh_node_ids, gh_cell_size, gh_lon_min, gh_lat_min,
    gh_nx, gh_ny, gh_wrap_lon, gh_far_rows,
    node_to_elems_count, node_to_elems,
    elem_center_lon, elem_center_lat,
    elem_nodes, nodes_lon, nodes_lat, node_xyz,
    u0, v0, u1, v1, u2, v2, u3, v3,
    t0, t1, t2, t3,
    tq_k1, tq_k2, tq_k3, tq_k4,
    dt, time_dir,
    lon_out, lat_out, status_out, nn_out, U_out, V_out, fr_out, exact
):
    """Fused RK4 step for 2D with 4-snapshot PCHIP time blend.

    Stage k samples the monotone-cubic velocity at query time ``tq_k`` with
    ``c_k = (0, 0.5, 0.5, 1)`` (computed by the wrapper).  Cubic-in-time twin
    of :func:`_fused_rk4_step_2d_tinterp`.
    """
    ALIVE = 1
    OUTSIDE = 3
    for i in prange(P):
        if not alive_mask[i]:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            status_out[i] = status_in[i]
            continue

        node_i = _fused_site_anchor(
            lons[i], lats[i],
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )
        lon0 = lons[i]
        lat0 = lats[i]

        # K1 at (lon0, lat0), query time tq_k1
        U1, V1, ok1 = _interp_leastsq_core_pchip4(
            lon0, lat0, node_i, time_dir, t0, t1, t2, t3, tq_k1,
            node_to_elems_count, node_to_elems,
            elem_center_lon, elem_center_lat,
            u0, v0, u1, v1, u2, v2, u3, v3,
        )
        U_out[i] = U1
        V_out[i] = V1
        if not ok1:
            lon_out[i] = lon0
            lat_out[i] = lat0
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_INTERP
            nn_out[i] = node_i
            continue

        z0_1 = math.cos(lat0 * DEG2RAD) * _SLM
        if z0_1 < 1.0:
            z0_1 = 1.0
        dlon1 = dt * U1 / z0_1
        dlat1 = dt * V1 / _SLM

        # K2 at (lon0 + dlon1/2, lat0 + dlat1/2), query time tq_k2
        lon2 = ((lon0 + 0.5 * dlon1) + 180.0) % 360.0 - 180.0
        lat2 = lat0 + 0.5 * dlat1
        if lat2 > 90.0: lat2 = 90.0
        elif lat2 < -90.0: lat2 = -90.0
        nn2 = _fused_site_anchor(
            lon2, lat2,
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )
        U2, V2, ok2 = _interp_leastsq_core_pchip4(
            lon2, lat2, nn2, time_dir, t0, t1, t2, t3, tq_k2,
            node_to_elems_count, node_to_elems,
            elem_center_lon, elem_center_lat,
            u0, v0, u1, v1, u2, v2, u3, v3,
        )
        if not ok2:
            lon_out[i] = lon0; lat_out[i] = lat0
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_INTERP; nn_out[i] = node_i; continue
        z0_2 = math.cos(lat2 * DEG2RAD) * _SLM
        if z0_2 < 1.0: z0_2 = 1.0
        dlon2 = dt * U2 / z0_2
        dlat2 = dt * V2 / _SLM

        # K3 at (lon0 + dlon2/2, lat0 + dlat2/2), query time tq_k3
        lon3 = ((lon0 + 0.5 * dlon2) + 180.0) % 360.0 - 180.0
        lat3 = lat0 + 0.5 * dlat2
        if lat3 > 90.0: lat3 = 90.0
        elif lat3 < -90.0: lat3 = -90.0
        nn3 = _fused_site_anchor(
            lon3, lat3,
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )
        U3, V3, ok3 = _interp_leastsq_core_pchip4(
            lon3, lat3, nn3, time_dir, t0, t1, t2, t3, tq_k3,
            node_to_elems_count, node_to_elems,
            elem_center_lon, elem_center_lat,
            u0, v0, u1, v1, u2, v2, u3, v3,
        )
        if not ok3:
            lon_out[i] = lon0; lat_out[i] = lat0
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_INTERP; nn_out[i] = node_i; continue
        z0_3 = math.cos(lat3 * DEG2RAD) * _SLM
        if z0_3 < 1.0: z0_3 = 1.0
        dlon3 = dt * U3 / z0_3
        dlat3 = dt * V3 / _SLM

        # K4 at (lon0 + dlon3, lat0 + dlat3), query time tq_k4
        lon4 = ((lon0 + dlon3) + 180.0) % 360.0 - 180.0
        lat4 = lat0 + dlat3
        if lat4 > 90.0: lat4 = 90.0
        elif lat4 < -90.0: lat4 = -90.0
        nn4 = _fused_site_anchor(
            lon4, lat4,
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )
        U4, V4, ok4 = _interp_leastsq_core_pchip4(
            lon4, lat4, nn4, time_dir, t0, t1, t2, t3, tq_k4,
            node_to_elems_count, node_to_elems,
            elem_center_lon, elem_center_lat,
            u0, v0, u1, v1, u2, v2, u3, v3,
        )
        if not ok4:
            lon_out[i] = lon0; lat_out[i] = lat0
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_INTERP; nn_out[i] = node_i; continue
        z0_4 = math.cos(lat4 * DEG2RAD) * _SLM
        if z0_4 < 1.0: z0_4 = 1.0
        dlon4 = dt * U4 / z0_4
        dlat4 = dt * V4 / _SLM

        # Weighted average
        dlon = (dlon1 + 2.0 * dlon2 + 2.0 * dlon3 + dlon4) / 6.0
        dlat = (dlat1 + 2.0 * dlat2 + 2.0 * dlat3 + dlat4) / 6.0
        lon_new = ((lon0 + dlon) + 180.0) % 360.0 - 180.0
        lat_new = lat0 + dlat
        if lat_new > 90.0: lat_new = 90.0
        elif lat_new < -90.0: lat_new = -90.0

        nn_new = _fused_site_anchor(
            lon_new, lat_new,
            gh_offsets, gh_node_ids, gh_cell_size,
            gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            nodes_lon, nodes_lat, node_xyz, exact, gh_far_rows,
        )
        _, in_dom = _find_containing_element_local(
            lon_new, lat_new, nn_new,
            node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat,
        )

        if not in_dom:
            lon_out[i] = lon0
            lat_out[i] = lat0
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_OOB
            nn_out[i] = node_i
        else:
            lon_out[i] = lon_new
            lat_out[i] = lat_new
            status_out[i] = ALIVE
            nn_out[i] = nn_new


# ---------------------------------------------------------------------------
# Common wrapper for fused 2D kernels
# ---------------------------------------------------------------------------

def _mesh_gh_args(mesh):
    return (mesh._gh_offsets, mesh._gh_node_ids, mesh._gh_cell_size,
            mesh._gh_lon_min, mesh._gh_lat_min, mesh._gh_nx, mesh._gh_ny,
            mesh._gh_wrap_lon, int(mesh._gh_far_rows))


def _run_fused_2d(state, vel, mesh, dt, time_dir, layer, fused_kernel):
    """Run a fused 2D kernel and return the new ParticleState."""
    P = state.n_particles
    alive = state.alive_mask

    # Pre-slice velocity to layer
    u_fi = vel.u if vel.u.ndim == 1 else vel.u[layer, :]
    v_fi = vel.v if vel.v.ndim == 1 else vel.v[layer, :]

    # Pre-allocate output arrays
    lon_out = state.lon.copy()
    lat_out = state.lat.copy()
    status_out = state.status.copy()
    nn_out = state.nearest_node.copy()
    U_out = np.zeros(P, dtype=np.float64)
    V_out = np.zeros(P, dtype=np.float64)

    fr_out = np.zeros(P, dtype=np.int8)
    fused_kernel(
        P, state.lon, state.lat, alive, state.status,
        *_mesh_gh_args(mesh),
        mesh.node_to_elems_count, mesh.node_to_elems,
        mesh.elem_center_lon, mesh.elem_center_lat,
        mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat, _require_node_xyz(mesh),
        u_fi, v_fi,
        dt, time_dir,
        lon_out, lat_out, status_out, nn_out, U_out, V_out, fr_out,
        _exact_nearest_on(mesh),
    )

    return state.replace(
        lon=lon_out, lat=lat_out, z=state.z.copy(),
        status=status_out, nearest_node=nn_out,
        u=U_out, v=V_out, w=None,
        failure_reason=_merge_fused_failure(state.failure_reason, fr_out),
    )


def _run_fused_euler_2d_pair(state, vel, mesh, dt, time_dir, layer):
    """Run the pair-aware fused Euler 2D kernel on a LazyLerpField.

    Same plumbing as _run_fused_2d but the kernel receives the RAW
    bracket pair + alpha0 instead of a materialized field.
    """
    P = state.n_particles
    alive = state.alive_mask

    vel_a_full, vel_b_full = vel.vel_a, vel.vel_b
    alpha0 = float(vel.alpha)

    u_a = vel_a_full.u if vel_a_full.u.ndim == 1 else vel_a_full.u[layer, :]
    v_a = vel_a_full.v if vel_a_full.v.ndim == 1 else vel_a_full.v[layer, :]
    u_b = vel_b_full.u if vel_b_full.u.ndim == 1 else vel_b_full.u[layer, :]
    v_b = vel_b_full.v if vel_b_full.v.ndim == 1 else vel_b_full.v[layer, :]

    lon_out = state.lon.copy()
    lat_out = state.lat.copy()
    status_out = state.status.copy()
    nn_out = state.nearest_node.copy()
    U_out = np.zeros(P, dtype=np.float64)
    V_out = np.zeros(P, dtype=np.float64)

    fr_out = np.zeros(P, dtype=np.int8)
    _fused_euler_step_2d_tinterp(
        P, state.lon, state.lat, alive, state.status,
        *_mesh_gh_args(mesh),
        mesh.node_to_elems_count, mesh.node_to_elems,
        mesh.elem_center_lon, mesh.elem_center_lat,
        mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat, _require_node_xyz(mesh),
        u_a, v_a, u_b, v_b, alpha0,
        dt, time_dir,
        lon_out, lat_out, status_out, nn_out, U_out, V_out, fr_out,
        _exact_nearest_on(mesh),
    )

    return state.replace(
        lon=lon_out, lat=lat_out, z=state.z.copy(),
        status=status_out, nearest_node=nn_out,
        u=U_out, v=V_out, w=None,
        failure_reason=_merge_fused_failure(state.failure_reason, fr_out),
    )


def _run_fused_2d_tinterp(state, vel, vel_next, dt_to_next, mesh, dt,
                          time_dir, layer, fused_kernel, c_coeffs):
    """Run a time-aware fused 2D kernel (RK2 or RK4).

    Parameters
    ----------
    c_coeffs : tuple
        Per-stage time fractions, e.g. (0.0, 0.5) for RK2 or
        (0.0, 0.5, 0.5, 1.0) for RK4.  Length determines the kernel variant.
    fused_kernel : callable
        Either _fused_rk2_step_2d_tinterp or _fused_rk4_step_2d_tinterp.
    """
    # No-allocation validation — avoids the per-stage lerp materialization
    # that _make_stage_vels would do.  Kernel blends inline per element.
    _validate_time_aware_inputs(vel, vel_next, dt_to_next, dt)

    P = state.n_particles

    # LazyLerpField fast path: hand the kernel the RAW bracket pair plus
    # the bracket fraction alpha0 — the kernel's nested blend reproduces
    # the materialized start-of-step field bitwise without ever building
    # it.  The STAGE TARGET (u_t/v_t) is vel_next: the bracket's upper
    # snapshot forward, but its LOWER one for backward integration — so
    # the bracket pair and the target must be threaded separately.
    # A plain (materialized or raw-snapshot) field passes alpha0=0.0,
    # which collapses the nested blend to the old arithmetic exactly
    # (the then-unused bracket-B slot is fed vel_next).
    vel_a_full = getattr(vel, "vel_a", vel)
    vel_b_full = getattr(vel, "vel_b", vel_next)
    alpha0 = float(getattr(vel, "alpha", 0.0))

    # Pre-slice bracket pair and stage target to the chosen layer
    u_a = vel_a_full.u if vel_a_full.u.ndim == 1 else vel_a_full.u[layer, :]
    v_a = vel_a_full.v if vel_a_full.v.ndim == 1 else vel_a_full.v[layer, :]
    u_b = vel_b_full.u if vel_b_full.u.ndim == 1 else vel_b_full.u[layer, :]
    v_b = vel_b_full.v if vel_b_full.v.ndim == 1 else vel_b_full.v[layer, :]
    u_t = vel_next.u if vel_next.u.ndim == 1 else vel_next.u[layer, :]
    v_t = vel_next.v if vel_next.v.ndim == 1 else vel_next.v[layer, :]

    # Compute stage alphas
    alphas = [float(c) * dt / dt_to_next for c in c_coeffs]

    alive = state.alive_mask
    lon_out = state.lon.copy()
    lat_out = state.lat.copy()
    status_out = state.status.copy()
    nn_out = state.nearest_node.copy()
    U_out = np.zeros(P, dtype=np.float64)
    V_out = np.zeros(P, dtype=np.float64)
    fr_out = np.zeros(P, dtype=np.int8)

    fused_kernel(
        P, state.lon, state.lat, alive, state.status,
        *_mesh_gh_args(mesh),
        mesh.node_to_elems_count, mesh.node_to_elems,
        mesh.elem_center_lon, mesh.elem_center_lat,
        mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat, _require_node_xyz(mesh),
        u_a, v_a, u_b, v_b, u_t, v_t, alpha0,
        *alphas,
        dt, time_dir,
        lon_out, lat_out, status_out, nn_out, U_out, V_out, fr_out,
        _exact_nearest_on(mesh),
    )

    return state.replace(
        lon=lon_out, lat=lat_out, z=state.z.copy(),
        status=status_out, nearest_node=nn_out,
        u=U_out, v=V_out, w=None,
        failure_reason=_merge_fused_failure(state.failure_reason, fr_out),
    )



def _exact_nearest_on(mesh) -> bool:
    """mesh.exact_nearest (set by run_tracking via core.mesh.set_exact_nearest)."""
    md = getattr(mesh, "metadata", None)
    return bool(md and md.get("exact_nearest", False))


def _bary_fused_anchor_exact(mesh):
    """Gate for the fused barycentric kernels.  Under ``mesh.exact_nearest``
    always True: the kernels anchor with the in-kernel exact search, so
    fusing cannot change a value on any mesh.  Otherwise True iff the
    grid-hash density certificate holds, i.e. the in-kernel widened
    nearest (`_grid_hash_nearest_widened`) is provably bit-identical to
    the batch KD-healed nearest for every containable query — so fusing
    cannot change any life/death verdict or value.  Denied (HEALPix
    polar rows, exotic multi-resolution meshes) → the caller stays on
    the unfused exact path; warn once per mesh so the perf downgrade is
    visible."""
    if _exact_nearest_on(mesh):
        return True
    dense = grid_hash_dense_for_elements(mesh)
    if not dense and not mesh._gh_dense_warned:
        warnings.warn(
            "fused kernels disabled for this mesh: the grid "
            "hash has empty 3x3 windows over element cells (e.g. HEALPix "
            "polar rows), where the in-kernel nearest-node anchor would "
            "be approximate. Falling back to the unfused (exact) path.",
            RuntimeWarning,
            stacklevel=3,       # point at rk2_step/rk4_step's caller
        )
        # Set AFTER the warn: under -W error the raise must not consume
        # the one-shot before a warning was ever delivered.
        mesh._gh_dense_warned = True
    return dense


def _leastsq_fused_anchor_exact(mesh):
    """Certificate gate for the fused LEASTSQ kernels — consulted only
    for meshes that declare HEALPix PIXEL geometry (2026-08 review: the
    pixel mesh fails the density certificate on its polar rows exactly
    like the dual mesh, and the fused in-kernel widened anchor measurably
    diverges from the batch KD-healed one there).  FESOM-family meshes
    keep the historical ungated fused path: quasi-uniform grids satisfy
    the certificate trivially and are not re-checked, so their behaviour
    and cost are untouched."""
    if _exact_nearest_on(mesh):
        return True     # in-kernel exact search (see above)
    md = getattr(mesh, "metadata", None)
    if not md or md.get("healpix_geometry") != "pixel":
        return True
    return _bary_fused_anchor_exact(mesh)


def _run_fused_2d_bary_tinterp(state, vel, vel_next, dt_to_next, mesh, dt,
                               time_dir, layer, fused_kernel, c_coeffs):
    """Run a time-aware fused 2D BARYCENTRIC kernel (RK2 or RK4).

    The barycentric twin of :func:`_run_fused_2d_tinterp` for NODE-layout
    velocities (the only layout the fused barycentric kernels implement —
    the dispatch gate guarantees it).  Same bracket-pair / stage-target
    threading: a ``LazyLerpField`` hands the kernel its raw parents plus
    ``alpha0``; a plain field passes ``alpha0 = 0.0``.

    Parameters
    ----------
    c_coeffs : tuple
        Per-stage time fractions, e.g. (0.0, 0.5) for RK2 or
        (0.0, 0.5, 0.5, 1.0) for RK4.  Length determines the kernel variant.
    fused_kernel : callable
        Either _fused_rk2_step_2d_bary_tinterp or
        _fused_rk4_step_2d_bary_tinterp.
    """
    _validate_time_aware_inputs(vel, vel_next, dt_to_next, dt)

    # The kernel indexes u/v by NODE id with no bounds checking — enforce
    # the node-layout contract on the stage target too (the dispatch gate
    # checks only `vel`; a direct caller could pair a node-layout `vel`
    # with an element-layout `vel_next` whose shapes happen to match).
    if getattr(vel_next, "uv_layout", "element") != "node":
        raise ValueError(
            "_run_fused_2d_bary_tinterp requires node-layout vel_next, "
            f"got {getattr(vel_next, 'uv_layout', 'element')!r}"
        )

    P = state.n_particles

    vel_a_full = getattr(vel, "vel_a", vel)
    vel_b_full = getattr(vel, "vel_b", vel_next)
    alpha0 = float(getattr(vel, "alpha", 0.0))

    # 2D node-layout fields are (N,); keep the layer guard for symmetry
    # with the leastsq wrapper (harmless on 1-D input).
    u_a = vel_a_full.u if vel_a_full.u.ndim == 1 else vel_a_full.u[layer, :]
    v_a = vel_a_full.v if vel_a_full.v.ndim == 1 else vel_a_full.v[layer, :]
    u_b = vel_b_full.u if vel_b_full.u.ndim == 1 else vel_b_full.u[layer, :]
    v_b = vel_b_full.v if vel_b_full.v.ndim == 1 else vel_b_full.v[layer, :]
    u_t = vel_next.u if vel_next.u.ndim == 1 else vel_next.u[layer, :]
    v_t = vel_next.v if vel_next.v.ndim == 1 else vel_next.v[layer, :]

    n_nodes = mesh.nodes_lon.shape[0]
    for name, arr in (("u_a", u_a), ("v_a", v_a), ("u_b", u_b),
                      ("v_b", v_b), ("u_t", u_t), ("v_t", v_t)):
        if arr.shape[0] != n_nodes:
            raise ValueError(
                f"_run_fused_2d_bary_tinterp: {name} has {arr.shape[0]} "
                f"values for {n_nodes} mesh nodes (not node layout?)"
            )

    alphas = [float(c) * dt / dt_to_next for c in c_coeffs]

    alive = state.alive_mask
    lon_out = state.lon.copy()
    lat_out = state.lat.copy()
    status_out = state.status.copy()
    nn_out = state.nearest_node.copy()
    U_out = np.zeros(P, dtype=np.float64)
    V_out = np.zeros(P, dtype=np.float64)
    fr_out = np.zeros(P, dtype=np.int8)

    fused_kernel(
        P, state.lon, state.lat, alive, state.status,
        *_mesh_gh_args(mesh),
        mesh.node_to_elems_count, mesh.node_to_elems,
        mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat, _require_node_xyz(mesh),
        u_a, v_a, u_b, v_b, u_t, v_t, alpha0,
        *alphas,
        dt, time_dir, float(mesh.r_earth),
        lon_out, lat_out, status_out, nn_out, U_out, V_out, fr_out,
        _exact_nearest_on(mesh),
    )

    return state.replace(
        lon=lon_out, lat=lat_out, z=state.z.copy(),
        status=status_out, nearest_node=nn_out,
        u=U_out, v=V_out, w=None,
        failure_reason=_merge_fused_failure(state.failure_reason, fr_out),
    )



def _run_fused_2d_pchip(state, recon, mesh, dt, time_dir, layer,
                        fused_kernel, c_coeffs):
    """Run a time-aware fused 2D kernel (RK2/RK4) with inline PCHIP blending.

    The cubic-in-time twin of :func:`_run_fused_2d_tinterp`.  Instead of a raw
    bracket pair + alpha, it unpacks the 4-snapshot reconstruction bundle and
    hands the kernel the four element-value arrays per component plus the four
    snapshot times; the kernel blends a Fritsch-Carlson cubic at each stage
    query time inline (only at the elements each particle touches).  Result
    equals the unfused ``make_stage_vels_recon`` path element-for-element, but
    without materializing a full reconstructed field per stage.

    Parameters
    ----------
    recon : tuple
        ``(snaps4, times4, q_time, direction)`` exactly as built by
        ``tracking._resolve_recon_stencil`` — 4 consecutive VelocityFields, the
        4 file times, the query (sim) time, and the integration direction.
    c_coeffs : tuple
        Per-stage time fractions, e.g. ``(0.0, 0.5)`` (RK2) or
        ``(0.0, 0.5, 0.5, 1.0)`` (RK4).
    """
    snaps4, times4, q_time, direction = recon

    # --- Validation parity with the unfused reconstruct_pchip path ---
    # The fused kernel calls _pchip_value_scalar directly with NO checks; a
    # malformed recon bundle would divide by zero (duplicate / non-finite
    # times), index out of bounds inside Numba (mismatched snapshot shapes),
    # or silently EXTRAPOLATE a non-monotone velocity (stage time outside the
    # central bracket — the very failure PCHIP exists to prevent).
    # reconstruct_pchip RAISES on all of these, so the fused wrapper must too.
    # The tracking layer already builds a valid bundle and nulls ``recon`` when
    # dt exceeds the bracket, so these guard only direct callers of
    # rk2_step / rk4_step / track_particles.
    if len(snaps4) != 4 or len(times4) != 4:
        raise ValueError("_run_fused_2d_pchip needs exactly 4 snapshots/times")
    t0, t1, t2, t3 = (float(t) for t in times4)
    if not all(np.isfinite(t) for t in (t0, t1, t2, t3)):
        raise ValueError(f"_run_fused_2d_pchip: non-finite times {times4}")
    if not (t0 < t1 < t2 < t3):
        raise ValueError(
            f"_run_fused_2d_pchip: times must be strictly increasing, "
            f"got {times4}")
    if direction * time_dir <= 0.0:
        raise ValueError(
            f"_run_fused_2d_pchip: recon direction {direction} disagrees with "
            f"step time_dir {time_dir}")
    for s in snaps4:
        if s.u is None or s.v is None:
            raise ValueError("_run_fused_2d_pchip: a snapshot is missing u/v")
        # The fused least-squares kernel indexes velocity by ELEMENT id; node
        # layout would read the wrong (or out-of-bounds) array — that case must
        # take the unfused barycentric path (tracking blocks it at startup via
        # _validate_uv_compat; this guards direct rk2_step/rk4_step callers).
        if s.uv_layout != "element":
            raise ValueError(
                f"_run_fused_2d_pchip requires element-layout velocity, got "
                f"uv_layout={s.uv_layout!r}; node layout must use the unfused "
                f"barycentric path (uv_interp='barycentric').")

    def _slice(field):
        # 2D element-layout fields are 1-D (E,); be defensive about a layer
        # axis even though the fused path is 2D-only (the dispatch guards is3d).
        return field if field.ndim == 1 else field[layer, :]

    u0 = _slice(snaps4[0].u); v0 = _slice(snaps4[0].v)
    u1 = _slice(snaps4[1].u); v1 = _slice(snaps4[1].v)
    u2 = _slice(snaps4[2].u); v2 = _slice(snaps4[2].v)
    u3 = _slice(snaps4[3].u); v3 = _slice(snaps4[3].v)

    # Every component array must be exactly (n_elements,): the kernel indexes
    # by element id (0 .. n_elements-1) from node_to_elems, so a shorter array
    # (e.g. all four snapshots one element short) or a node-sized / leftover-3D
    # array would read out of bounds (unchecked) inside Numba.
    ne = mesh.n_elements
    shapes = {a.shape for a in (u0, v0, u1, v1, u2, v2, u3, v3)}
    if shapes != {(ne,)}:
        raise ValueError(
            f"_run_fused_2d_pchip: snapshot u/v shapes {shapes} must all be "
            f"(n_elements={ne},); the fused 2D kernel indexes by element id.")

    # Per-stage query times in the central interval [t1, t2].  ``direction``
    # mirrors the unfused make_stage_vels_recon contract (same sign as time_dir,
    # checked above).
    tqs = [float(q_time) + direction * float(c) * dt for c in c_coeffs]
    tol = 1e-9 * (t2 - t1)
    for tq in tqs:
        if not (t1 - tol <= tq <= t2 + tol):
            raise ValueError(
                f"_run_fused_2d_pchip: stage query time {tq} outside the "
                f"central PCHIP interval [{t1}, {t2}] (dt={dt} exceeds the "
                f"bracket); pass dt <= the snapshot spacing, or let the "
                f"tracking layer fall back to linear.")

    P = state.n_particles
    alive = state.alive_mask
    lon_out = state.lon.copy()
    lat_out = state.lat.copy()
    status_out = state.status.copy()
    nn_out = state.nearest_node.copy()
    U_out = np.zeros(P, dtype=np.float64)
    V_out = np.zeros(P, dtype=np.float64)
    fr_out = np.zeros(P, dtype=np.int8)

    fused_kernel(
        P, state.lon, state.lat, alive, state.status,
        *_mesh_gh_args(mesh),
        mesh.node_to_elems_count, mesh.node_to_elems,
        mesh.elem_center_lon, mesh.elem_center_lat,
        mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat, _require_node_xyz(mesh),
        u0, v0, u1, v1, u2, v2, u3, v3,
        t0, t1, t2, t3,
        *tqs,
        dt, time_dir,
        lon_out, lat_out, status_out, nn_out, U_out, V_out, fr_out,
        _exact_nearest_on(mesh),
    )

    return state.replace(
        lon=lon_out, lat=lat_out, z=state.z.copy(),
        status=status_out, nearest_node=nn_out,
        u=U_out, v=V_out, w=None,
        failure_reason=_merge_fused_failure(state.failure_reason, fr_out),
    )



def _run_fused_euler_3d(state, vel, mesh, dt, time_dir, w_interp="leastsq"):
    """Run the fused 3D Euler kernel and return the new ParticleState."""
    P = state.n_particles
    alive = state.alive_mask
    z_levels = _get_z_levels(mesh)

    w_method = _W_METHOD_MAP.get(w_interp, W_LEASTSQ)

    lon_out = state.lon.copy()
    lat_out = state.lat.copy()
    z_out = state.z.copy()
    status_out = state.status.copy()
    nn_out = state.nearest_node.copy()
    U_out = np.zeros(P, dtype=np.float64)
    V_out = np.zeros(P, dtype=np.float64)
    W_out = np.zeros(P, dtype=np.float64)
    fr_out = np.zeros(P, dtype=np.int8)

    _fused_euler_step_3d(
        P, state.lon, state.lat, state.z, alive, state.status,
        *_mesh_gh_args(mesh),
        mesh.node_to_elems_count, mesh.node_to_elems,
        mesh.elem_center_lon, mesh.elem_center_lat,
        mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat, _require_node_xyz(mesh),
        mesh.depth, z_levels,
        vel.u, vel.v, vel.w,
        dt, time_dir, w_method,
        lon_out, lat_out, z_out, status_out, nn_out, U_out, V_out, W_out,
        fr_out, _exact_nearest_on(mesh),
    )

    return state.replace(
        lon=lon_out, lat=lat_out, z=z_out,
        status=status_out, nearest_node=nn_out,
        u=U_out, v=V_out, w=W_out,
        failure_reason=_merge_fused_failure(state.failure_reason, fr_out),
    )


# ---------------------------------------------------------------------------
# Fused 3-D RK2 for NODE-layout barycentric velocities (2026-08)
#
# The z-level 3-D twin of the 2-D barycentric kernels: per particle, both
# stages run entirely in-loop — per-stage layer find, per-vertex
# barycentric u/v at the particle's mid-layer with the inline time blend
# (linear or 4-snapshot PCHIP), single-particle W (nearest or
# barycentric) with the same inline blend, midpoint move, final
# containment check, surface/bottom clamp.  Bit-identical to
# _rk2_step_unfused on the same inputs (pinned by
# tests/test_fused_3d_bary.py): the stage-1 anchor comes in EXACT from
# the runner (batch KD-healed), and the stage-2/final anchors are
# 3x3-first with HYBRID exactness — an empty hash window flags the
# particle and the runner reruns those few through the exact unfused
# path — KD-heal of the flagged stage position + subset rerun through the
# same kernel (_run_fused_3d_with_heal); the whole-field unfused splice
# (_splice_exact_rerun) is only the terminal fallback — so no
# density-certificate gate is needed
# on any mesh.  Sigma meshes, walk, other w_interp methods and element
# layout stay on the unfused path (dispatch gate).
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _fused_rk2_step_3d_bary_tinterp(
    P, lons, lats, zs, alive_mask, status_in, nn1,
    gh_offsets, gh_node_ids, gh_cell_size, gh_lon_min, gh_lat_min,
    gh_nx, gh_ny, gh_wrap_lon, gh_far_rows,
    node_to_elems_count, node_to_elems,
    elem_nodes, nodes_lon, nodes_lat, node_xyz, node_depths, z_levels,
    u_a, v_a, u_b, v_b, u_t, v_t, w_a, w_b, w_t, alpha0,
    alpha_k1, alpha_k2,
    dt, time_dir, r_earth, w_code,
    lon_out, lat_out, z_out, status_out, nn_out,
    U_out, V_out, W_out, fr_out, need_exact,
    ov_mid, ov_new, flag_lon, flag_lat, flag_site, exact
):
    ALIVE = 1
    OUTSIDE = 3
    z_surface = z_levels[0]
    for i in prange(P):
        if not alive_mask[i]:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            z_out[i] = zs[i]
            status_out[i] = status_in[i]
            continue

        # stage-1 anchor comes in EXACT (batch KD-healed, computed by
        # the runner like the unfused path) — the widened in-kernel search
        # is certificate-exact only for containable queries, and an alive
        # particle already off the element footprint (regional extracts,
        # direct callers) would otherwise diverge in its nn/W diagnostics
        # (2026-08 review).
        node_i = nn1[i]
        layer1 = find_layer(z_levels, zs[i], node_depths[node_i])

        U1, V1, ok1 = _interp_barycentric_core_tinterp_3d(
            lons[i], lats[i], node_i, time_dir, layer1, alpha0, alpha_k1,
            node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat,
            u_a, v_a, u_b, v_b, u_t, v_t, r_earth,
        )
        # W1 for EVERY alive particle (the unfused path computes it with
        # mask=alive, before the ok1 verdict) — it is also the diagnostic.
        if w_code == W_NEAREST:
            W1 = _interp_w_linear_single_tinterp(
                w_a, w_b, w_t, alpha0, alpha_k1,
                z_levels, zs[i], node_i, time_dir)
        else:
            W1, _wok = _interp_w_bary_single_tinterp(
                lons[i], lats[i], zs[i], node_i,
                w_a, w_b, w_t, alpha0, alpha_k1,
                z_levels, time_dir, node_depths,
                nodes_lon, nodes_lat,
                node_to_elems, node_to_elems_count, elem_nodes)
        U_out[i] = U1
        V_out[i] = V1
        W_out[i] = W1

        if not ok1:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            z_out[i] = zs[i]
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_INTERP
            nn_out[i] = node_i
            continue

        z0 = math.cos(lats[i] * DEG2RAD) * _SLM
        if z0 < 1.0:
            z0 = 1.0
        lon_mid = lons[i] + 0.5 * dt * U1 / z0
        lat_mid = lats[i] + 0.5 * dt * V1 / _SLM
        lon_mid = ((lon_mid + 180.0) % 360.0) - 180.0
        if lat_mid > 90.0:
            lat_mid = 90.0
        elif lat_mid < -90.0:
            lat_mid = -90.0
        z_mid = zs[i] + W1 * 0.5 * dt

        # 3x3-first: a non-empty window is bit-identical to the batch
        # KD-healed nearest; an EMPTY window is the only case where the
        # in-kernel search could diverge — flag the particle for the
        # exact unfused rerun instead (hybrid exactness, 2026-08:
        # replaces the density-certificate gate for the 3-D kernels).
        if exact:
            # the in-kernel exact search (round 4: the flag/runner-heal
            # rounds were slower than the unfused path here too)
            nn_mid = _grid_hash_nearest_exact(
                lon_mid, lat_mid,
                gh_offsets, gh_node_ids, gh_cell_size,
                gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
                nodes_lon, nodes_lat, node_xyz, gh_far_rows,
            )
        else:
            nn_mid = _grid_hash_nearest_3x3(
                lon_mid, lat_mid,
                gh_offsets, gh_node_ids, gh_cell_size,
                gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
                nodes_lon, nodes_lat, node_xyz,
            )
        if nn_mid < 0:
            # anchor-heal handshake (subset-cost hybrid, 2026-08): the
            # runner KD-heals the flagged position and re-runs JUST the
            # flagged particles with the exact anchor injected — the
            # whole-field unfused splice stays a terminal fallback only
            nn_mid = ov_mid[i]
        if nn_mid < 0:
            need_exact[i] = 1
            flag_site[i] = 1
            flag_lon[i] = lon_mid
            flag_lat[i] = lat_mid
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            z_out[i] = zs[i]
            status_out[i] = status_in[i]
            continue
        layer2 = find_layer(z_levels, z_mid, node_depths[nn_mid])

        U2, V2, ok2 = _interp_barycentric_core_tinterp_3d(
            lon_mid, lat_mid, nn_mid, time_dir, layer2, alpha0, alpha_k2,
            node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat,
            u_a, v_a, u_b, v_b, u_t, v_t, r_earth,
        )
        if not ok2:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            z_out[i] = zs[i]
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_INTERP
            nn_out[i] = node_i
            continue

        if w_code == W_NEAREST:
            W2 = _interp_w_linear_single_tinterp(
                w_a, w_b, w_t, alpha0, alpha_k2,
                z_levels, z_mid, nn_mid, time_dir)
        else:
            W2, _wok2 = _interp_w_bary_single_tinterp(
                lon_mid, lat_mid, z_mid, nn_mid,
                w_a, w_b, w_t, alpha0, alpha_k2,
                z_levels, time_dir, node_depths,
                nodes_lon, nodes_lat,
                node_to_elems, node_to_elems_count, elem_nodes)

        z0m = math.cos(lat_mid * DEG2RAD) * _SLM
        if z0m < 1.0:
            z0m = 1.0
        lon_new = lons[i] + dt * U2 / z0m
        lat_new = lats[i] + dt * V2 / _SLM
        lon_new = ((lon_new + 180.0) % 360.0) - 180.0
        if lat_new > 90.0:
            lat_new = 90.0
        elif lat_new < -90.0:
            lat_new = -90.0
        z_new = zs[i] + W2 * dt

        if exact:
            # the in-kernel exact search (round 4: the flag/runner-heal
            # rounds were slower than the unfused path here too)
            nn_new = _grid_hash_nearest_exact(
                lon_new, lat_new,
                gh_offsets, gh_node_ids, gh_cell_size,
                gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
                nodes_lon, nodes_lat, node_xyz, gh_far_rows,
            )
        else:
            nn_new = _grid_hash_nearest_3x3(
                lon_new, lat_new,
                gh_offsets, gh_node_ids, gh_cell_size,
                gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
                nodes_lon, nodes_lat, node_xyz,
            )
        if nn_new < 0:
            nn_new = ov_new[i]
        if nn_new < 0:
            need_exact[i] = 1
            flag_site[i] = 2
            flag_lon[i] = lon_new
            flag_lat[i] = lat_new
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            z_out[i] = zs[i]
            status_out[i] = status_in[i]
            continue
        _, in_dom = _find_containing_element_local(
            lon_new, lat_new, nn_new,
            node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat,
        )
        if not in_dom:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            z_out[i] = zs[i]
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_OOB
            nn_out[i] = node_i
        else:
            # surface/bottom clamp (the _loop_clamp_depth arithmetic,
            # applied — like the unfused path — with the FINAL anchor)
            if z_new < z_surface:
                z_new = z_surface
            bottom = node_depths[nn_new]
            if z_new > bottom:
                z_new = bottom
            lon_out[i] = lon_new
            lat_out[i] = lat_new
            z_out[i] = z_new
            status_out[i] = ALIVE
            nn_out[i] = nn_new


@njit(parallel=True, cache=True)
def _fused_rk2_step_3d_bary_pchip(
    P, lons, lats, zs, alive_mask, status_in, nn1,
    gh_offsets, gh_node_ids, gh_cell_size, gh_lon_min, gh_lat_min,
    gh_nx, gh_ny, gh_wrap_lon, gh_far_rows,
    node_to_elems_count, node_to_elems,
    elem_nodes, nodes_lon, nodes_lat, node_xyz, node_depths, z_levels,
    u_0, v_0, u_1, v_1, u_2, v_2, u_3, v_3,
    w_0, w_1, w_2, w_3,
    t0, t1, t2, t3, tq1, tq2,
    dt, time_dir, r_earth, w_code,
    lon_out, lat_out, z_out, status_out, nn_out,
    U_out, V_out, W_out, fr_out, need_exact,
    ov_mid, ov_new, flag_lon, flag_lat, flag_site, exact
):
    """PCHIP twin of :func:`_fused_rk2_step_3d_bary_tinterp` (4-snapshot
    monotone-cubic inline blend at the two stage times ``tq1``/``tq2``)."""
    ALIVE = 1
    OUTSIDE = 3
    z_surface = z_levels[0]
    for i in prange(P):
        if not alive_mask[i]:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            z_out[i] = zs[i]
            status_out[i] = status_in[i]
            continue

        # stage-1 anchor comes in EXACT (batch KD-healed, computed by
        # the runner like the unfused path) — the widened in-kernel search
        # is certificate-exact only for containable queries, and an alive
        # particle already off the element footprint (regional extracts,
        # direct callers) would otherwise diverge in its nn/W diagnostics
        # (2026-08 review).
        node_i = nn1[i]
        layer1 = find_layer(z_levels, zs[i], node_depths[node_i])

        U1, V1, ok1 = _interp_barycentric_core_pchip4_3d(
            lons[i], lats[i], node_i, time_dir, layer1,
            node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat,
            u_0, v_0, u_1, v_1, u_2, v_2, u_3, v_3,
            t0, t1, t2, t3, tq1, r_earth,
        )
        if w_code == W_NEAREST:
            W1 = _interp_w_linear_single_pchip4(
                w_0, w_1, w_2, w_3, t0, t1, t2, t3, tq1,
                z_levels, zs[i], node_i, time_dir)
        else:
            W1, _wok = _interp_w_bary_single_pchip4(
                lons[i], lats[i], zs[i], node_i,
                w_0, w_1, w_2, w_3, t0, t1, t2, t3, tq1,
                z_levels, time_dir, node_depths,
                nodes_lon, nodes_lat,
                node_to_elems, node_to_elems_count, elem_nodes)
        U_out[i] = U1
        V_out[i] = V1
        W_out[i] = W1

        if not ok1:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            z_out[i] = zs[i]
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_INTERP
            nn_out[i] = node_i
            continue

        z0 = math.cos(lats[i] * DEG2RAD) * _SLM
        if z0 < 1.0:
            z0 = 1.0
        lon_mid = lons[i] + 0.5 * dt * U1 / z0
        lat_mid = lats[i] + 0.5 * dt * V1 / _SLM
        lon_mid = ((lon_mid + 180.0) % 360.0) - 180.0
        if lat_mid > 90.0:
            lat_mid = 90.0
        elif lat_mid < -90.0:
            lat_mid = -90.0
        z_mid = zs[i] + W1 * 0.5 * dt

        # 3x3-first: a non-empty window is bit-identical to the batch
        # KD-healed nearest; an EMPTY window is the only case where the
        # in-kernel search could diverge — flag the particle for the
        # exact unfused rerun instead (hybrid exactness, 2026-08:
        # replaces the density-certificate gate for the 3-D kernels).
        if exact:
            # the in-kernel exact search (round 4: the flag/runner-heal
            # rounds were slower than the unfused path here too)
            nn_mid = _grid_hash_nearest_exact(
                lon_mid, lat_mid,
                gh_offsets, gh_node_ids, gh_cell_size,
                gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
                nodes_lon, nodes_lat, node_xyz, gh_far_rows,
            )
        else:
            nn_mid = _grid_hash_nearest_3x3(
                lon_mid, lat_mid,
                gh_offsets, gh_node_ids, gh_cell_size,
                gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
                nodes_lon, nodes_lat, node_xyz,
            )
        if nn_mid < 0:
            # anchor-heal handshake (subset-cost hybrid, 2026-08): the
            # runner KD-heals the flagged position and re-runs JUST the
            # flagged particles with the exact anchor injected — the
            # whole-field unfused splice stays a terminal fallback only
            nn_mid = ov_mid[i]
        if nn_mid < 0:
            need_exact[i] = 1
            flag_site[i] = 1
            flag_lon[i] = lon_mid
            flag_lat[i] = lat_mid
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            z_out[i] = zs[i]
            status_out[i] = status_in[i]
            continue
        layer2 = find_layer(z_levels, z_mid, node_depths[nn_mid])

        U2, V2, ok2 = _interp_barycentric_core_pchip4_3d(
            lon_mid, lat_mid, nn_mid, time_dir, layer2,
            node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat,
            u_0, v_0, u_1, v_1, u_2, v_2, u_3, v_3,
            t0, t1, t2, t3, tq2, r_earth,
        )
        if not ok2:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            z_out[i] = zs[i]
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_INTERP
            nn_out[i] = node_i
            continue

        if w_code == W_NEAREST:
            W2 = _interp_w_linear_single_pchip4(
                w_0, w_1, w_2, w_3, t0, t1, t2, t3, tq2,
                z_levels, z_mid, nn_mid, time_dir)
        else:
            W2, _wok2 = _interp_w_bary_single_pchip4(
                lon_mid, lat_mid, z_mid, nn_mid,
                w_0, w_1, w_2, w_3, t0, t1, t2, t3, tq2,
                z_levels, time_dir, node_depths,
                nodes_lon, nodes_lat,
                node_to_elems, node_to_elems_count, elem_nodes)

        z0m = math.cos(lat_mid * DEG2RAD) * _SLM
        if z0m < 1.0:
            z0m = 1.0
        lon_new = lons[i] + dt * U2 / z0m
        lat_new = lats[i] + dt * V2 / _SLM
        lon_new = ((lon_new + 180.0) % 360.0) - 180.0
        if lat_new > 90.0:
            lat_new = 90.0
        elif lat_new < -90.0:
            lat_new = -90.0
        z_new = zs[i] + W2 * dt

        if exact:
            # the in-kernel exact search (round 4: the flag/runner-heal
            # rounds were slower than the unfused path here too)
            nn_new = _grid_hash_nearest_exact(
                lon_new, lat_new,
                gh_offsets, gh_node_ids, gh_cell_size,
                gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
                nodes_lon, nodes_lat, node_xyz, gh_far_rows,
            )
        else:
            nn_new = _grid_hash_nearest_3x3(
                lon_new, lat_new,
                gh_offsets, gh_node_ids, gh_cell_size,
                gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
                nodes_lon, nodes_lat, node_xyz,
            )
        if nn_new < 0:
            nn_new = ov_new[i]
        if nn_new < 0:
            need_exact[i] = 1
            flag_site[i] = 2
            flag_lon[i] = lon_new
            flag_lat[i] = lat_new
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            z_out[i] = zs[i]
            status_out[i] = status_in[i]
            continue
        _, in_dom = _find_containing_element_local(
            lon_new, lat_new, nn_new,
            node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat,
        )
        if not in_dom:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            z_out[i] = zs[i]
            status_out[i] = OUTSIDE
            fr_out[i] = _FR_OOB
            nn_out[i] = node_i
        else:
            if z_new < z_surface:
                z_new = z_surface
            bottom = node_depths[nn_new]
            if z_new > bottom:
                z_new = bottom
            lon_out[i] = lon_new
            lat_out[i] = lat_new
            z_out[i] = z_new
            status_out[i] = ALIVE
            nn_out[i] = nn_new


def _fused_3d_bary_field_checks(mesh, fields_kn1, fields_kn):
    """Shape guards for the 3-D barycentric runners: u/v are (K-1, N),
    w is (K, N) on the mesh's node count/vertical table (no bounds checks
    inside the kernels)."""
    n_nodes = mesh.nodes_lon.shape[0]
    K = int(mesh.sigma.size)
    for name, arr in fields_kn1:
        if arr is None or arr.ndim != 2 or arr.shape != (K - 1, n_nodes):
            raise ValueError(
                f"fused 3-D barycentric kernel: {name} must be "
                f"(K-1, N) = ({K - 1}, {n_nodes}), got "
                f"{None if arr is None else arr.shape}")
    for name, arr in fields_kn:
        if arr is None or arr.ndim != 2 or arr.shape != (K, n_nodes):
            raise ValueError(
                f"fused 3-D barycentric kernel: {name} must be "
                f"(K, N) = ({K}, {n_nodes}), got "
                f"{None if arr is None else arr.shape}")


_W_CODE_NAME = {0: "nearest", 2: "barycentric"}  # W_NEAREST/W_BARYCENTRIC


#: max KD-heal rounds for flagged stage anchors before the terminal
#: whole-field splice fallback (2 sites -> 2 rounds suffice; the spare
#: capacity is paranoia, not a tuning knob)
_FLAG_HEAL_ROUNDS = 2


def _run_fused_3d_with_heal(run_kernel, state, mesh, P):
    """Drive a fused 3-D kernel with the subset-cost anchor-heal loop.

    ``run_kernel(alive_mask, ov_mid, ov_new) -> dict`` invokes the numba
    kernel with the given alive mask and anchor overrides and returns its
    freshly allocated output arrays (keys: lon, lat, z, status, nn, U, V,
    W, fr, need_exact, flag_lon, flag_lat, flag_site).

    A flagged particle (empty 3x3 hash window at a stage anchor) gets its
    recorded stage position KD-healed via ``find_nearest_nodes`` — the
    IDENTICAL call the unfused reference path makes at stage positions,
    so the injected anchor is bit-identical by construction — and only
    the flagged subset re-runs through the same fused kernel.  Cost is
    proportional to the flagged count; the whole-field unfused splice
    (which blends stage fields over the entire mesh even for one
    particle — the 2026-08 zoom-10 scan anomaly) remains only as the
    terminal fallback for anything still flagged after the rounds.

    Returns ``(outs, flagged)`` — merged outputs and the still-flagged
    mask for the terminal fallback.
    """
    alive = state.alive_mask
    ov_mid = np.full(P, -1, dtype=np.int32)
    ov_new = np.full(P, -1, dtype=np.int32)
    outs = run_kernel(alive, ov_mid, ov_new)
    flagged = outs["need_exact"].astype(bool)
    rounds = 0
    while flagged.any() and rounds < _FLAG_HEAL_ROUNDS:
        idx = np.where(flagged)[0]
        healed = find_nearest_nodes(
            mesh, outs["flag_lon"][idx],
            outs["flag_lat"][idx]).astype(np.int32)
        mid_sel = outs["flag_site"][idx] == 1
        ov_mid[idx[mid_sel]] = healed[mid_sel]
        ov_new[idx[~mid_sel]] = healed[~mid_sel]
        sub = run_kernel(flagged & alive, ov_mid, ov_new)
        for k in outs:
            outs[k] = np.where(flagged, sub[k], outs[k])
        flagged = outs["need_exact"].astype(bool)
        rounds += 1
    return outs, flagged


def _splice_exact_rerun(kernel_out, state, flagged, unfused_fn):
    """Hybrid exactness: rerun the FLAGGED particles (empty hash window at
    a stage-2/final anchor — the only case where the in-kernel nearest
    could diverge from the batch KD-healed one) through the exact unfused
    path, and splice their results into the kernel output.  The rerun
    restricts the alive mask by marking every non-flagged particle
    OUTSIDE in a THROWAWAY status copy; splicing covers every per-
    particle field including diagnostics and failure codes."""
    sub_status = np.where(flagged, state.status,
                          np.int32(int(ParticleStatus.OUTSIDE)))
    sub_state = state.replace(status=sub_status)
    exact = unfused_fn(sub_state)
    out = kernel_out
    sel = flagged
    repl = {}
    for name in ("lon", "lat", "z", "nearest_node", "u", "v", "w",
                 "status"):
        ko = getattr(out, name)
        ex = getattr(exact, name)
        if ko is None or ex is None:
            continue
        repl[name] = np.where(sel, ex, ko)
    if out.failure_reason is not None:
        repl["failure_reason"] = np.where(
            sel, exact.failure_reason, out.failure_reason).astype(np.int8)
    return out.replace(**repl)


def _fused_3d_optional_payload_checks(a, b):
    """Error-parity guards for optional payload fields the fused kernel
    never reads: the unfused stage build materializes ``a.lerp(b, ...)``,
    which RAISES on mismatched scalars / Kh / Az — a malformed direct
    payload must not silently enter the fused kernel instead (2026-08
    review).  No allocation, shape/key checks only."""
    sa = getattr(a, "scalars", None)
    sb = getattr(b, "scalars", None)
    if sa and sb:
        if set(sa) != set(sb):
            raise ValueError(
                "scalar keys differ between the temporal-interp pair "
                f"({sorted(sa)} vs {sorted(sb)})")
        for k in sa:
            if sa[k].shape != sb[k].shape:
                raise ValueError(
                    f"scalar {k!r} shape mismatch between the pair: "
                    f"{sa[k].shape} vs {sb[k].shape}")
    aza = getattr(a, "az", None)
    azb = getattr(b, "az", None)
    if aza is not None and azb is not None and aza.shape != azb.shape:
        raise ValueError(
            f"az shape mismatch between the pair: {aza.shape} vs "
            f"{azb.shape}")
    ka = getattr(a, "kh_zonal", None)
    kb = getattr(b, "kh_zonal", None)
    if (ka is not None and kb is not None
            and (ka.shape != kb.shape
                 or getattr(a, "kh_layout", None)
                 != getattr(b, "kh_layout", None))):
        raise ValueError("Kh layout/shape mismatch between the pair")


def _run_fused_rk2_3d_bary_tinterp(state, vel, vel_next, dt_to_next, mesh,
                                   dt, time_dir, w_code):
    """Run the fused 3-D barycentric RK2 kernel (steady or linear tinterp).

    Steady mode (``vel_next is None``): both stage alphas are 0.0, so
    every blend short-circuits to the in-hand field — bit-identical to
    the unfused ``_make_stage_vels(vel, None, ...)`` (vel, vel) pair.
    Time-aware mode threads the LazyLerpField bracket pair + ``alpha0``
    and the stage target exactly like ``_run_fused_2d_bary_tinterp``.
    """
    P = state.n_particles
    alive = state.alive_mask
    if w_code not in (W_NEAREST, W_BARYCENTRIC):
        raise ValueError(f"unsupported w_code {w_code} for the fused 3-D "
                         "barycentric kernel (nearest/barycentric only)")
    if getattr(vel, "uv_layout", "element") != "node":
        raise ValueError(
            "_run_fused_rk2_3d_bary_tinterp requires node-layout vel, "
            f"got {getattr(vel, 'uv_layout', 'element')!r}")
    if (hasattr(vel, "vel_a")
            and float(getattr(vel, "alpha", 1.0)) == 0.0):
        # the alpha0 == 0 shortcut in the kernels means "PLAIN field";
        # an endpoint LazyLerpField materializes through _blend with
        # 0*B semantics (non-finite B -> NaN) — direct callers must use
        # the unfused path for that exotic payload (the rk2_step
        # dispatch already routes it there; 2026-08 review round 2).
        raise ValueError(
            "_run_fused_rk2_3d_bary_tinterp: endpoint LazyLerpField "
            "payload (alpha0 == 0.0) is not supported by the fused "
            "kernel — use the unfused path")

    if vel_next is None:
        vel_a_full = getattr(vel, "vel_a", vel)
        u_a = vel_a_full.u; v_a = vel_a_full.v; w_a = vel_a_full.w
        u_b, v_b, w_b = u_a, v_a, w_a
        u_t, v_t, w_t = u_a, v_a, w_a
        alpha0 = float(getattr(vel, "alpha", 0.0))
        if hasattr(vel, "vel_a"):
            vel_b_full = vel.vel_b
            u_b, v_b, w_b = vel_b_full.u, vel_b_full.v, vel_b_full.w
        alphas = (0.0, 0.0)
    else:
        _validate_time_aware_inputs(vel, vel_next, dt_to_next, dt)
        if getattr(vel_next, "uv_layout", "element") != "node":
            raise ValueError(
                "_run_fused_rk2_3d_bary_tinterp requires node-layout "
                f"vel_next, got "
                f"{getattr(vel_next, 'uv_layout', 'element')!r}")
        vel_a_full = getattr(vel, "vel_a", vel)
        vel_b_full = getattr(vel, "vel_b", vel_next)
        alpha0 = float(getattr(vel, "alpha", 0.0))
        u_a, v_a, w_a = vel_a_full.u, vel_a_full.v, vel_a_full.w
        u_b, v_b, w_b = vel_b_full.u, vel_b_full.v, vel_b_full.w
        u_t, v_t, w_t = vel_next.u, vel_next.v, vel_next.w
        alphas = tuple(float(c) * dt / dt_to_next for c in (0.0, 0.5))
        # on the RAW parents — touching a LazyLerpField's .scalars/.az
        # would materialize whole-field blends the kernel never needs
        _fused_3d_optional_payload_checks(vel_a_full, vel_b_full)
        _fused_3d_optional_payload_checks(vel_a_full, vel_next)

    _fused_3d_bary_field_checks(
        mesh,
        [("u_a", u_a), ("v_a", v_a), ("u_b", u_b), ("v_b", v_b),
         ("u_t", u_t), ("v_t", v_t)],
        [("w_a", w_a), ("w_b", w_b), ("w_t", w_t)],
    )

    z_levels = _get_z_levels(mesh)
    # exact stage-1 anchor, same call the unfused path makes
    alive_idx = np.where(alive)[0]
    nn1 = state.nearest_node.copy()
    if alive_idx.size > 0:
        nn1[alive_idx] = find_nearest_nodes(
            mesh, state.lon[alive_idx], state.lat[alive_idx])

    def run_kernel(alive_arr, ov_mid, ov_new):
        lon_out = state.lon.copy()
        lat_out = state.lat.copy()
        z_out = state.z.copy()
        status_out = state.status.copy()
        nn_out = state.nearest_node.copy()
        U_out = np.zeros(P, dtype=np.float64)
        V_out = np.zeros(P, dtype=np.float64)
        W_out = np.zeros(P, dtype=np.float64)
        fr_out = np.zeros(P, dtype=np.int8)
        need_exact = np.zeros(P, dtype=np.int8)
        flag_lon = np.zeros(P, dtype=np.float64)
        flag_lat = np.zeros(P, dtype=np.float64)
        flag_site = np.zeros(P, dtype=np.int8)
        _fused_rk2_step_3d_bary_tinterp(
            P, state.lon, state.lat, state.z, alive_arr, state.status,
            nn1,
            mesh._gh_offsets, mesh._gh_node_ids, mesh._gh_cell_size,
            mesh._gh_lon_min, mesh._gh_lat_min, mesh._gh_nx, mesh._gh_ny,
            mesh._gh_wrap_lon, int(mesh._gh_far_rows),
            mesh.node_to_elems_count, mesh.node_to_elems,
            mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat, _require_node_xyz(mesh),
            mesh.depth, z_levels,
            u_a, v_a, u_b, v_b, u_t, v_t, w_a, w_b, w_t, alpha0,
            alphas[0], alphas[1],
            dt, time_dir, float(mesh.r_earth), w_code,
            lon_out, lat_out, z_out, status_out, nn_out,
            U_out, V_out, W_out, fr_out, need_exact,
            ov_mid, ov_new, flag_lon, flag_lat, flag_site,
            bool((mesh.metadata or {}).get("exact_nearest", False)),
        )
        return {"lon": lon_out, "lat": lat_out, "z": z_out,
                "status": status_out, "nn": nn_out, "U": U_out,
                "V": V_out, "W": W_out, "fr": fr_out,
                "need_exact": need_exact, "flag_lon": flag_lon,
                "flag_lat": flag_lat, "flag_site": flag_site}

    outs, flagged = _run_fused_3d_with_heal(run_kernel, state, mesh, P)
    result = state.replace(
        lon=outs["lon"], lat=outs["lat"], z=outs["z"],
        status=outs["status"], nearest_node=outs["nn"],
        u=outs["U"], v=outs["V"], w=outs["W"],
        failure_reason=_merge_fused_failure(state.failure_reason,
                                            outs["fr"]),
    )
    if flagged.any():
        result = _splice_exact_rerun(
            result, state, flagged,
            lambda sub: _rk2_step_unfused(
                sub, vel, mesh, dt, time_dir, 0, True,
                w_interp=_W_CODE_NAME[int(w_code)],
                vel_next=vel_next, dt_to_next=dt_to_next,
                uv_interp="barycentric"))
    return result


def _run_fused_rk2_3d_bary_pchip(state, recon, mesh, dt, time_dir, w_code):
    """Run the fused 3-D barycentric RK2 kernel with inline PCHIP blending.

    Re-validates the recon bundle exactly like ``_run_fused_2d_pchip``
    (reconstruct_pchip raises on every malformed case, so the fused
    wrapper must too), with the node-layout twist: snapshots must carry
    NODE-layout u/v and a w field.
    """
    snaps4, times4, q_time, direction = recon
    if w_code not in (W_NEAREST, W_BARYCENTRIC):
        raise ValueError(f"unsupported w_code {w_code} for the fused 3-D "
                         "barycentric kernel (nearest/barycentric only)")
    if len(snaps4) != 4 or len(times4) != 4:
        raise ValueError(
            "_run_fused_rk2_3d_bary_pchip needs exactly 4 snapshots/times")
    t0, t1, t2, t3 = (float(t) for t in times4)
    if not all(np.isfinite(t) for t in (t0, t1, t2, t3)):
        raise ValueError(
            f"_run_fused_rk2_3d_bary_pchip: non-finite times {times4}")
    if not (t0 < t1 < t2 < t3):
        raise ValueError(
            f"_run_fused_rk2_3d_bary_pchip: times must be strictly "
            f"increasing, got {times4}")
    if direction * time_dir <= 0.0:
        raise ValueError(
            f"_run_fused_rk2_3d_bary_pchip: recon direction {direction} "
            f"disagrees with step time_dir {time_dir}")
    for s in snaps4:
        if s.u is None or s.v is None or s.w is None:
            raise ValueError(
                "_run_fused_rk2_3d_bary_pchip: a snapshot is missing u/v/w")
        if s.uv_layout != "node":
            raise ValueError(
                "_run_fused_rk2_3d_bary_pchip requires node-layout "
                f"snapshots, got uv_layout={s.uv_layout!r}")
    tqs = []
    tol = 1e-9 * (t2 - t1)
    for c in (0.0, 0.5):
        tq = float(q_time) + float(direction) * c * dt
        if not (t1 - tol <= tq <= t2 + tol):
            raise ValueError(
                f"_run_fused_rk2_3d_bary_pchip: stage time {tq} outside "
                f"the central bracket [{t1}, {t2}] — dt exceeds the "
                "bracket (tracking nulls recon in that case).")
        tqs.append(tq)

    a, b, c_, d = snaps4
    # error-parity with reconstruct_pchip EXACTLY (round 3): it validates
    # scalars (keys identical across all four, or all absent; shapes
    # equal) and eta — and deliberately IGNORES Kh/Az (not
    # reconstructed), so this guard must not be stricter there.
    skeys = [set(s_.scalars) if s_.scalars else None for s_ in snaps4]
    if any(k is not None for k in skeys):
        ref = next(k for k in skeys if k is not None)
        if any(k != ref for k in skeys):
            raise ValueError(
                "scalar keys differ (or present in only some) across "
                "reconstruction snapshots")
        for k in ref:
            shp = {s_.scalars[k].shape for s_ in snaps4}
            if len(shp) > 1:
                raise ValueError(
                    f"scalar {k!r} shape mismatch across reconstruction "
                    f"snapshots: {shp}")
    eta_present = [s_.eta is not None for s_ in snaps4]
    if any(eta_present) and not all(eta_present):
        raise ValueError(
            "_run_fused_rk2_3d_bary_pchip: eta present in only some "
            "reconstruction snapshots")
    if all(eta_present):
        eshape = {s_.eta.shape for s_ in snaps4}
        if len(eshape) > 1:
            raise ValueError(
                f"_run_fused_rk2_3d_bary_pchip: eta shape mismatch "
                f"across reconstruction snapshots: {eshape}")
    _fused_3d_bary_field_checks(
        mesh,
        [("u0", a.u), ("v0", a.v), ("u1", b.u), ("v1", b.v),
         ("u2", c_.u), ("v2", c_.v), ("u3", d.u), ("v3", d.v)],
        [("w0", a.w), ("w1", b.w), ("w2", c_.w), ("w3", d.w)],
    )

    P = state.n_particles
    z_levels = _get_z_levels(mesh)
    alive = state.alive_mask
    alive_idx = np.where(alive)[0]
    nn1 = state.nearest_node.copy()
    if alive_idx.size > 0:
        nn1[alive_idx] = find_nearest_nodes(
            mesh, state.lon[alive_idx], state.lat[alive_idx])
    def run_kernel(alive_arr, ov_mid, ov_new):
        lon_out = state.lon.copy()
        lat_out = state.lat.copy()
        z_out = state.z.copy()
        status_out = state.status.copy()
        nn_out = state.nearest_node.copy()
        U_out = np.zeros(P, dtype=np.float64)
        V_out = np.zeros(P, dtype=np.float64)
        W_out = np.zeros(P, dtype=np.float64)
        fr_out = np.zeros(P, dtype=np.int8)
        need_exact = np.zeros(P, dtype=np.int8)
        flag_lon = np.zeros(P, dtype=np.float64)
        flag_lat = np.zeros(P, dtype=np.float64)
        flag_site = np.zeros(P, dtype=np.int8)
        _fused_rk2_step_3d_bary_pchip(
            P, state.lon, state.lat, state.z, alive_arr, state.status,
            nn1,
            mesh._gh_offsets, mesh._gh_node_ids, mesh._gh_cell_size,
            mesh._gh_lon_min, mesh._gh_lat_min, mesh._gh_nx, mesh._gh_ny,
            mesh._gh_wrap_lon, int(mesh._gh_far_rows),
            mesh.node_to_elems_count, mesh.node_to_elems,
            mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat, _require_node_xyz(mesh),
            mesh.depth, z_levels,
            a.u, a.v, b.u, b.v, c_.u, c_.v, d.u, d.v,
            a.w, b.w, c_.w, d.w,
            t0, t1, t2, t3, tqs[0], tqs[1],
            dt, time_dir, float(mesh.r_earth), w_code,
            lon_out, lat_out, z_out, status_out, nn_out,
            U_out, V_out, W_out, fr_out, need_exact,
            ov_mid, ov_new, flag_lon, flag_lat, flag_site,
            bool((mesh.metadata or {}).get("exact_nearest", False)),
        )
        return {"lon": lon_out, "lat": lat_out, "z": z_out,
                "status": status_out, "nn": nn_out, "U": U_out,
                "V": V_out, "W": W_out, "fr": fr_out,
                "need_exact": need_exact, "flag_lon": flag_lon,
                "flag_lat": flag_lat, "flag_site": flag_site}

    outs, flagged = _run_fused_3d_with_heal(run_kernel, state, mesh, P)
    result = state.replace(
        lon=outs["lon"], lat=outs["lat"], z=outs["z"],
        status=outs["status"], nearest_node=outs["nn"],
        u=outs["U"], v=outs["V"], w=outs["W"],
        failure_reason=_merge_fused_failure(state.failure_reason,
                                            outs["fr"]),
    )
    if flagged.any():
        result = _splice_exact_rerun(
            result, state, flagged,
            lambda sub: _rk2_step_unfused(
                sub, snaps4[1], mesh, dt, time_dir, 0, True,
                w_interp=_W_CODE_NAME[int(w_code)],
                uv_interp="barycentric", recon=recon))
    return result


# ---------------------------------------------------------------------------
# Unfused fallback implementations (used when Numba unavailable or 3D RK2/RK4)
# ---------------------------------------------------------------------------

def _euler_step_unfused(state, vel, mesh, dt, time_dir, layer, is3d,
                        w_interp="nearest", vel_next=None, dt_to_next=None,
                        Dmin=0.5, z0b_min=0.1, uv_interp="leastsq",
                        edge_leastsq_bc="freeslip", sigma_eta_end=None,
                        sigma_eta_start=None):
    """Original multi-pass Euler implementation (fallback).

    Forward Euler is a single-stage method evaluating velocity at t_step.
    The vel_next/dt_to_next parameters are accepted for API symmetry with
    RK2/RK4 but ignored — Euler order in time is 1 regardless.

    ``uv_interp`` selects the horizontal u/v interpolator from the registry
    (``"leastsq"`` element-centre default / ``"barycentric"`` vertex blend /
    ``"edge_leastsq"`` edge-neighbor fit); node-based velocity fields
    require ``"barycentric"``.  ``edge_leastsq_bc`` is the boundary-ghost
    value for edge_leastsq (ignored by the other methods).
    """
    interp_fn = get_interpolator(uv_interp, edge_leastsq_bc)
    alive = state.alive_mask
    P = state.n_particles
    alive_idx = np.where(alive)[0]

    # Edge-walk element tracking is ON when the state carries elems
    # (seeded once at release by the tracking layer, elem_search: walk).
    use_walk = state.elem is not None
    walk_interp = use_walk and uv_interp in UV_INTERP_USES_ELEMS

    a_lon = state.lon[alive_idx]
    a_lat = state.lat[alive_idx]
    nn = find_nearest_nodes(mesh, a_lon, a_lat)
    nn_full = _scatter(nn, alive_idx, P)

    sig3d = is3d and mesh.vertical_coord == "sigma"
    H0 = None
    vel_w = vel
    if sig3d:
        eta0 = _sigma_given(sigma_eta_start, vel.eta)
        vel_w = _sigma_stage_field(vel, eta0)
        H0 = _sigma_col_H(mesh, state.lon, state.lat, nn_full, eta0, alive,
                          w_interp, Dmin)
    if is3d:
        interp_layer = _compute_layers(
            mesh, vel_w, state.z, nn_full, alive,
            Dmin=Dmin, z0b_min=z0b_min, H_col=H0,
        )
    else:
        interp_layer = layer

    cur_elems = state.elem
    if walk_interp:
        cur_elems, _ = _walk_update_elems(
            mesh, state.lon, state.lat, state.elem, alive)
        U, V, ok = interp_fn(state, mesh, vel, nn_full, time_dir,
                             interp_layer, elems=cur_elems)
    else:
        U, V, ok = interp_fn(state, mesh, vel, nn_full, time_dir,
                             interp_layer)
    moves = alive & ok

    old_lon = state.lon.copy()
    old_lat = state.lat.copy()
    old_z = state.z.copy()
    lon_new, lat_new = _apply_displacement(state.lon, state.lat, U, V, dt, moves)

    z_new = state.z.copy()
    W = None
    if is3d:
        W = _compute_w(vel_w, mesh, state.z, nn_full, alive, time_dir,
                       w_interp=w_interp, lons=state.lon, lats=state.lat,
                       Dmin=Dmin, z0b_min=z0b_min)
        if sig3d:
            # ds/dt = W / H_p; the depth follows the end-of-step column
            s_new = (state.z + W * dt) / H0
            eta_end = _sigma_end_eta(vel, vel_next, dt_to_next, dt, None,
                                     sigma_eta_end, eta_start=eta0)
        else:
            z_new = np.where(moves, state.z + W * dt, state.z)

    moved_idx = np.where(moves)[0]
    nn_new_full = np.zeros(P, dtype=np.int32)
    in_domain_full = np.zeros(P, dtype=bool)
    elem_new = None
    if use_walk:
        start_elems = cur_elems if walk_interp else state.elem
        elem_new, walk_ok = _walk_update_elems(
            mesh, lon_new, lat_new, start_elems, moves)
    if moved_idx.size > 0:
        m_nn = find_nearest_nodes(mesh, lon_new[moved_idx], lat_new[moved_idx])
        nn_new_full[moved_idx] = m_nn
        if use_walk:
            in_domain_full[moved_idx] = walk_ok[moved_idx]
        else:
            m_in = _check_in_domain(mesh, lon_new[moved_idx],
                                    lat_new[moved_idx], m_nn)
            in_domain_full[moved_idx] = m_in

    interp_failed = alive & ~ok
    domain_failed = alive & ok & moves & ~in_domain_full
    failed = interp_failed | domain_failed
    new_status = state.status.copy()
    new_status[failed] = ParticleStatus.OUTSIDE
    lon_new = np.where(failed, old_lon, lon_new)
    lat_new = np.where(failed, old_lat, lat_new)
    z_new = np.where(failed, old_z, z_new)
    nn_new_full = np.where(moves & ~failed, nn_new_full, nn_full)
    nn_new_full = np.where(~alive, state.nearest_node, nn_new_full)
    if use_walk:
        # reverted particles keep their start-of-step element; dead ones
        # keep whatever they carried
        elem_new = np.where(failed, start_elems, elem_new)
        elem_new = np.where(~alive, state.elem, elem_new).astype(np.int32)

    if sig3d:
        ok_mv = alive & ~failed
        H_end = _sigma_col_H(mesh, lon_new, lat_new, nn_new_full, eta_end,
                             ok_mv, w_interp, Dmin)
        z_new = np.where(ok_mv, s_new * H_end, z_new)
        z_new = _clamp_z(z_new, vel, mesh, nn_new_full, ok_mv,
                         Dmin=Dmin, z0b_min=z0b_min, H_col=H_end)
    elif is3d:
        z_new = _clamp_z(
            z_new, vel, mesh, nn_new_full, alive & ~failed,
            Dmin=Dmin, z0b_min=z0b_min,
        )

    new_failure_reason = _update_failure_reason(
        state.failure_reason,
        {FailureReason.INTERPOLATION_FAILED: interp_failed,
         FailureReason.OUT_OF_BOUNDS: domain_failed},
    )

    out = state.replace(
        lon=lon_new, lat=lat_new, z=z_new,
        status=new_status, nearest_node=nn_new_full,
        u=U, v=V, w=W,
        failure_reason=new_failure_reason,
    )
    if use_walk:
        out = out.replace(elem=elem_new)
    return out


# ---------------------------------------------------------------------------
# Euler (forward)
# ---------------------------------------------------------------------------

def euler_step(
    state: ParticleState,
    vel: VelocityField,
    mesh: Mesh,
    dt: float,
    time_dir: float,
    layer: int = 0,
    **kwargs,
) -> ParticleState:
    """Single Euler forward step.

    Parameters
    ----------
    state : ParticleState
        Current particle state.
    vel : VelocityField
        Velocity field for this timestep.
    mesh : Mesh
        Mesh with search structures built.
    dt : float
        Time step in seconds (always positive; direction via time_dir).
    time_dir : float
        +1 forward, -1 backward.
    layer : int
        Vertical layer index (0 for 2-D).  Ignored in 3D mode
        (per-particle layers computed automatically).

    Returns
    -------
    ParticleState
        New state after one Euler step.
    """
    diffusion: Optional[DiffusionStepArgs] = kwargs.pop("diffusion", None)
    _reject_em_m1_az(diffusion)
    alive = state.alive_mask
    if not np.any(alive):
        return _zero_dead_velocity(state)

    w_interp = kwargs.get("w_interp", "leastsq")
    uv_interp = kwargs.get("uv_interp", "leastsq")
    edge_leastsq_bc = kwargs.get("edge_leastsq_bc", "freeslip")
    Dmin = kwargs.get("Dmin", 0.5)
    z0b_min = kwargs.get("z0b_min", 0.1)
    vel_next = kwargs.get("vel_next", None)
    dt_to_next = kwargs.get("dt_to_next", None)
    # sigma meshes: the surface at the step's start and end (supplied by the
    # tracking layer for every step, one function of time)
    sigma_eta_end = kwargs.get("sigma_eta_end", None)
    sigma_eta_start = kwargs.get("sigma_eta_start", None)
    _validate_time_aware_kwargs(vel_next, dt_to_next)
    is3d = _is_3d(vel)
    # Sigma 3D is routed through the unfused path: _run_fused_euler_3d is
    # hard-wired to z-level geometry and cannot yet handle per-node sigma
    # column maths.  See docs/vertical.md.
    is_sigma_3d = is3d and mesh.vertical_coord == "sigma"

    # Combined-mode dispatch: EM/M1 schemes that the
    # tracking layer auto-resolved to ``composition="combined"`` *replace*
    # the deterministic Euler step rather than running afterwards.  3D
    # combined-mode is not supported in this phase — fail fast with a
    # phase pointer.
    if (diffusion is not None
            and diffusion.composition == "combined"
            and diffusion.scheme in ("em", "m1")):
        if is3d:
            raise NotImplementedError(
                "diffusion.scheme='em'/'m1' is 2D-only: spatially "
                "varying horizontal Kh is not wired into the 3D Euler "
                "path.  See docs/diffusion.md §5."
            )
        return _apply_combined_em_step(
            state, vel, mesh, dt, time_dir, diffusion,
            Dmin=Dmin, z0b_min=z0b_min, uv_interp=uv_interp,
        )

    # Euler is single-stage: vel_next/dt_to_next are no-ops for the
    # velocity (documented in _euler_step_unfused); on sigma meshes they
    # give the column's surface at the end of the step.
    if HAS_NUMBA:
        if is3d:
            # The 3D fused kernel bakes leastsq u/v math, so it is only
            # eligible when uv_interp is leastsq (barycentric/node falls
            # through to the unfused dispatcher).  Under mesh.exact_nearest
            # it anchors every site through the in-kernel exact search
            # (_fused_site_anchor -> _grid_hash_nearest_exact), like every
            # other fused kernel; no flag/heal round trip.
            fused_ok = (not is_sigma_3d and w_interp in ("nearest", "leastsq")
                        and uv_interp == "leastsq")
            if fused_ok:
                result = _run_fused_euler_3d(state, vel, mesh, dt, time_dir,
                                             w_interp=w_interp)
            else:
                result = _euler_step_unfused(state, vel, mesh, dt, time_dir,
                                             layer, is3d, w_interp=w_interp,
                                             vel_next=vel_next,
                                             dt_to_next=dt_to_next,
                                             Dmin=Dmin, z0b_min=z0b_min,
                                             uv_interp=uv_interp,
                                             edge_leastsq_bc=edge_leastsq_bc,
                                             sigma_eta_end=sigma_eta_end,
                                   sigma_eta_start=sigma_eta_start)
        elif uv_interp == "leastsq" and _leastsq_fused_anchor_exact(mesh):
            # LazyLerpField: feed the kernel the raw bracket pair —
            # Euler's velocity is the start-of-step field, blended
            # inline per accessed element (no O(mesh) materialization).
            if hasattr(vel, "vel_a"):
                result = _run_fused_euler_2d_pair(state, vel, mesh, dt,
                                                  time_dir, layer)
            else:
                result = _run_fused_2d(state, vel, mesh, dt, time_dir, layer,
                                       _fused_euler_step_2d)
        else:
            result = _euler_step_unfused(state, vel, mesh, dt, time_dir, layer,
                                         is3d, w_interp=w_interp,
                                         Dmin=Dmin, z0b_min=z0b_min,
                                         uv_interp=uv_interp,
                                         edge_leastsq_bc=edge_leastsq_bc)
    else:
        result = _euler_step_unfused(state, vel, mesh, dt, time_dir, layer, is3d,
                                     w_interp=w_interp,
                                     vel_next=vel_next, dt_to_next=dt_to_next,
                                     Dmin=Dmin, z0b_min=z0b_min,
                                     uv_interp=uv_interp,
                                     edge_leastsq_bc=edge_leastsq_bc,
                                     sigma_eta_end=sigma_eta_end,
                                   sigma_eta_start=sigma_eta_start)

    if diffusion is not None:
        sigma_col = _sigma_kick_col(state, vel, mesh, dt, is3d, w_interp,
                                    vel_next, dt_to_next, None,
                                    sigma_eta_end, sigma_eta_start)
        result = _apply_split_diffusion_kick(
            state, result, mesh, dt, diffusion, vel,
            Dmin=Dmin, z0b_min=z0b_min, sigma_col=sigma_col,
        )
        result = _apply_split_az_kick(
            state, result, mesh, dt, diffusion, vel,
            time_dir=time_dir, Dmin=Dmin, z0b_min=z0b_min,
            sigma_col=sigma_col,
        )
    return result


# ---------------------------------------------------------------------------
# RK2 (midpoint method)
# ---------------------------------------------------------------------------

def _interp_uv_at(interp_fn, state, mesh, vel, lons, lats, mask,
                  nn_full, time_dir, layer_arg, elems=None):
    """Vectorised (parallel) u/v interpolation at explicit stage positions.

    RK stages after the first evaluate u/v at the midpoint / stage positions,
    which are *not* ``state.lon``/``state.lat``.  The original unfused 3D
    RK2/RK4 path therefore looped the single-point interpolator over particles
    (serial Python) — the one part of the step that did not run in parallel and
    that dominated wall time at millions of particles.

    This reuses the registry's *vectorised* interpolator (its
    ``@njit(parallel=True)`` kernels) by handing it a shallow stage state whose
    ``lon``/``lat`` are the stage positions ``lons``/``lats`` and whose
    ``alive_mask`` equals ``mask`` (encoded as a status array — ``.replace`` is
    a cheap reference-sharing copy).  The vectorised kernel and the single-point
    function call the identical ``_interp_*_core`` with identical arguments
    (same node, same layer slice, same field), so the result is bit-for-bit
    identical to the per-particle loop — only parallel.  Particles outside
    ``mask`` are skipped by the kernel and return ``0.0 / 0.0 / False``,
    matching the loop's pre-zeroed outputs.

    Parameters
    ----------
    interp_fn : callable
        Vectorised interpolator from ``get_interpolator(uv_interp)``.
    state : ParticleState
        Step-entry state, used only as a ``.replace`` template.
    vel : VelocityField
        Stage velocity field.
    lons, lats : np.ndarray, shape ``(P,)``
        Stage positions.
    mask : np.ndarray of bool, shape ``(P,)``
        Particles to interpolate (others return ``0.0 / 0.0 / False``).
    nn_full : np.ndarray, shape ``(P,)``
        Nearest-node index per particle at the stage positions.
    layer_arg : int or np.ndarray
        Scalar layer (2-D) or per-particle layers ``(P,)`` (3-D).

    elems : np.ndarray or None
        Optional pre-validated containing elements at the STAGE positions
        (edge-walk tracker); forwarded to interpolators that accept them.

    Returns
    -------
    U, V, ok : np.ndarray, shape ``(P,)``
    """
    stage_status = np.where(
        mask, int(ParticleStatus.ALIVE), int(ParticleStatus.OUTSIDE)
    ).astype(state.status.dtype)
    stage_state = state.replace(lon=lons, lat=lats, status=stage_status)
    if elems is not None:
        return interp_fn(stage_state, mesh, vel, nn_full, time_dir,
                         layer_arg, elems=elems)
    return interp_fn(stage_state, mesh, vel, nn_full, time_dir, layer_arg)


def _rk2_step_unfused(state, vel, mesh, dt, time_dir, layer, is3d,
                      w_interp="nearest", vel_next=None, dt_to_next=None,
                      Dmin=0.5, z0b_min=0.1, uv_interp="leastsq", recon=None,
                      edge_leastsq_bc="freeslip", sigma_eta_end=None,
                      sigma_eta_start=None):
    """Original multi-pass RK2 (midpoint) implementation (fallback).

    Time-aware mode: stages use vel_k = lerp(vel, vel_next, c_k * dt / dt_to_next)
    with c_k = (0, 0.5).  Steady mode (vel_next=None): both stages reuse vel.

    ``uv_interp`` selects the horizontal u/v interpolator from the registry;
    both stages now use the vectorised (parallel) form.
    """
    interp_fn = get_interpolator(uv_interp, edge_leastsq_bc)
    alive = state.alive_mask
    P = state.n_particles
    alive_idx = np.where(alive)[0]

    # Stage velocities: stage 0 (t_step), stage 1 (t_step + dt/2)
    if recon is not None:
        from .temporal_reconstruct import make_stage_vels_recon
        _snaps, _times, _sim_time, _tdir = recon
        stage_vels = make_stage_vels_recon(_snaps, _times, _sim_time, dt,
                                           _tdir, (0.0, 0.5))
    else:
        stage_vels = _make_stage_vels(vel, vel_next, dt_to_next, dt,
                                      c_coeffs=(0.0, 0.5))
    vel_k1, vel_k2 = stage_vels

    # The final clamp's field: z-level clamps ignore eta, and on sigma meshes
    # the clamp takes the particle's end-of-step column (H_col, from the
    # step's own end-of-step surface, below) — no whole-field end-of-step
    # lerp/reconstruction is needed (it was built and discarded before).
    vel_end = vel

    # Edge-walk element tracking (elem_search: walk)
    use_walk = state.elem is not None
    walk_interp = use_walk and uv_interp in UV_INTERP_USES_ELEMS

    a_lon = state.lon[alive_idx]
    a_lat = state.lat[alive_idx]
    nn1_compact = find_nearest_nodes(mesh, a_lon, a_lat)
    nn1 = _scatter(nn1_compact, alive_idx, P)

    # sigma meshes: integrate s = z / H_p (see _sigma_col_H); a steady
    # fallback step blends the surface towards the tracking layer's eta_end
    sig3d = is3d and mesh.vertical_coord == "sigma"
    steady = recon is None and (vel_next is None or dt_to_next is None
                                or dt_to_next <= 0)
    H1 = None
    if sig3d:
        eta0 = _sigma_given(sigma_eta_start, vel_k1.eta)
        eta_end = _sigma_end_eta(vel, vel_next, dt_to_next, dt, recon,
                                 sigma_eta_end, eta_start=eta0)
        eta_mid = (_eta_blend(eta0, eta_end, 0.5) if steady
                   else vel_k2.eta)
        vel_k1 = _sigma_stage_field(vel_k1, eta0)
        H1 = _sigma_col_H(mesh, state.lon, state.lat, nn1, eta0, alive,
                          w_interp, Dmin)
    if is3d:
        interp_layer1 = _compute_layers(
            mesh, vel_k1, state.z, nn1, alive,
            Dmin=Dmin, z0b_min=z0b_min, H_col=H1,
        )
    else:
        interp_layer1 = layer

    cur_elems = state.elem
    if walk_interp:
        cur_elems, _ = _walk_update_elems(
            mesh, state.lon, state.lat, state.elem, alive)
        U1, V1, ok1 = interp_fn(state, mesh, vel_k1, nn1, time_dir,
                                interp_layer1, elems=cur_elems)
    else:
        U1, V1, ok1 = interp_fn(state, mesh, vel_k1, nn1, time_dir,
                                interp_layer1)

    moves1 = alive & ok1
    lon_mid, lat_mid = _apply_displacement(
        state.lon, state.lat, U1, V1, 0.5 * dt, moves1
    )

    z_mid = state.z.copy()
    W1 = None
    if is3d:
        W1 = _compute_w(vel_k1, mesh, state.z, nn1, alive, time_dir,
                        w_interp=w_interp, lons=state.lon, lats=state.lat,
                        Dmin=Dmin, z0b_min=z0b_min)
        if sig3d:
            s_mid = (state.z + W1 * 0.5 * dt) / H1
        else:
            z_mid = np.where(moves1, state.z + W1 * 0.5 * dt, state.z)

    moves1_idx = np.where(moves1)[0]
    U2 = np.zeros(P, dtype=np.float64)
    V2 = np.zeros(P, dtype=np.float64)
    ok2 = np.zeros(P, dtype=bool)
    W2 = np.zeros(P, dtype=np.float64)
    H2 = np.ones(P, dtype=np.float64)

    last_elems = cur_elems
    if moves1_idx.size > 0:
        m_nn = find_nearest_nodes(mesh, lon_mid[moves1_idx], lat_mid[moves1_idx])
        nn_mid_full = nn1.copy()
        nn_mid_full[moves1_idx] = m_nn

        vel_k2_w = vel_k2
        if sig3d:
            H2 = _sigma_col_H(mesh, lon_mid, lat_mid, nn_mid_full, eta_mid,
                              moves1, w_interp, Dmin)
            z_mid = np.where(moves1, s_mid * H2, state.z)
            vel_k2_w = _sigma_stage_field(vel_k2, eta_mid)
        if is3d:
            layers_mid = _compute_layers(
                mesh, vel_k2_w, z_mid, nn_mid_full, moves1,
                Dmin=Dmin, z0b_min=z0b_min, H_col=H2 if sig3d else None,
            )
            layer_arg = layers_mid
        else:
            layer_arg = layer

        mid_elems = None
        if walk_interp:
            mid_elems, _ = _walk_update_elems(
                mesh, lon_mid, lat_mid, cur_elems, moves1)
            last_elems = mid_elems

        # Stage-2 u/v in parallel (was a serial per-particle loop).
        U2, V2, ok2 = _interp_uv_at(
            interp_fn, state, mesh, vel_k2,
            lon_mid, lat_mid, moves1, nn_mid_full, time_dir, layer_arg,
            elems=mid_elems,
        )

        if is3d:
            W2 = _compute_w(vel_k2_w, mesh, z_mid, nn_mid_full, moves1, time_dir,
                            w_interp=w_interp, lons=lon_mid, lats=lat_mid,
                            Dmin=Dmin, z0b_min=z0b_min)

    ok = ok1 & ok2
    moves = alive & ok
    old_lon = state.lon.copy()
    old_lat = state.lat.copy()
    old_z = state.z.copy()
    lon_new, lat_new = _apply_displacement(state.lon, state.lat, U2, V2, dt, moves,
                                           z0_lat=lat_mid)

    z_new = state.z.copy()
    if sig3d:
        s_new = state.z / H1 + dt * W2 / H2
    elif is3d:
        z_new = np.where(moves, state.z + W2 * dt, state.z)

    moved_idx = np.where(moves)[0]
    nn_new = np.copy(nn1)
    in_domain_full = np.zeros(P, dtype=bool)
    elem_new = None
    if use_walk:
        start_elems = cur_elems if walk_interp else state.elem
        elem_new, walk_ok = _walk_update_elems(
            mesh, lon_new, lat_new,
            last_elems if walk_interp else state.elem, moves)
    if moved_idx.size > 0:
        m_nn = find_nearest_nodes(mesh, lon_new[moved_idx], lat_new[moved_idx])
        nn_new[moved_idx] = m_nn
        if use_walk:
            in_domain_full[moved_idx] = walk_ok[moved_idx]
        else:
            m_in = _check_in_domain(mesh, lon_new[moved_idx],
                                    lat_new[moved_idx], m_nn)
            in_domain_full[moved_idx] = m_in

    interp_failed = alive & ~ok
    domain_failed = alive & ok & moves & ~in_domain_full
    failed = interp_failed | domain_failed
    new_status = state.status.copy()
    new_status[failed] = ParticleStatus.OUTSIDE
    lon_new = np.where(failed, old_lon, lon_new)
    lat_new = np.where(failed, old_lat, lat_new)
    z_new = np.where(failed, old_z, z_new)
    nn_new = np.where(failed, nn1, nn_new)
    nn_new = np.where(~alive, state.nearest_node, nn_new)
    if use_walk:
        elem_new = np.where(failed, start_elems, elem_new)
        elem_new = np.where(~alive, state.elem, elem_new).astype(np.int32)

    if sig3d:
        ok_mv = alive & ~failed
        H_end = _sigma_col_H(mesh, lon_new, lat_new, nn_new, eta_end, ok_mv,
                             w_interp, Dmin)
        z_new = np.where(ok_mv, s_new * H_end, z_new)
        z_new = _clamp_z(z_new, vel_end, mesh, nn_new, ok_mv,
                         Dmin=Dmin, z0b_min=z0b_min, H_col=H_end)
    elif is3d:
        z_new = _clamp_z(
            z_new, vel_end, mesh, nn_new, alive & ~failed,
            Dmin=Dmin, z0b_min=z0b_min,
        )

    new_failure_reason = _update_failure_reason(
        state.failure_reason,
        {FailureReason.INTERPOLATION_FAILED: interp_failed,
         FailureReason.OUT_OF_BOUNDS: domain_failed},
    )

    out = state.replace(
        lon=lon_new, lat=lat_new, z=z_new,
        status=new_status, nearest_node=nn_new,
        u=U1, v=V1, w=W1,
        failure_reason=new_failure_reason,
    )
    if use_walk:
        out = out.replace(elem=elem_new)
    return out


def rk2_step(
    state: ParticleState,
    vel: VelocityField,
    mesh: Mesh,
    dt: float,
    time_dir: float,
    layer: int = 0,
    **kwargs,
) -> ParticleState:
    """Single RK2 (midpoint) step.

    Stage 1: half Euler step → midpoint position.
    Stage 2: evaluate velocity at midpoint, full step from original position.
    """
    diffusion: Optional[DiffusionStepArgs] = kwargs.pop("diffusion", None)
    _reject_em_m1_az(diffusion)
    _reject_combined_em_on_rk(diffusion, "rk2")
    alive = state.alive_mask
    if not np.any(alive):
        return _zero_dead_velocity(state)

    w_interp = kwargs.get("w_interp", "leastsq")
    uv_interp = kwargs.get("uv_interp", "leastsq")
    edge_leastsq_bc = kwargs.get("edge_leastsq_bc", "freeslip")
    Dmin = kwargs.get("Dmin", 0.5)
    z0b_min = kwargs.get("z0b_min", 0.1)
    vel_next = kwargs.get("vel_next", None)
    dt_to_next = kwargs.get("dt_to_next", None)
    # sigma meshes: the surface at the step's start and end (supplied by the
    # tracking layer for every step, one function of time)
    sigma_eta_end = kwargs.get("sigma_eta_end", None)
    sigma_eta_start = kwargs.get("sigma_eta_start", None)
    recon = kwargs.get("recon", None)
    _validate_time_aware_kwargs(vel_next, dt_to_next)
    is3d = _is_3d(vel)
    time_aware = vel_next is not None  # both or neither (validated above)

    # The fused 2D RK2 kernels bake leastsq u/v math; eligible only when
    # uv_interp is leastsq and the field is 2D.  A PCHIP recon bundle now has
    # its own fused 4-snapshot cubic kernel; the steady and 2-snapshot linear
    # blends keep their kernels.  Node-layout barycentric with temporal
    # interpolation has a fused twin too (per-vertex inline blend — no
    # per-stage whole-node-field lerp); element-layout barycentric, walk
    # (state.elem carried), steady, PCHIP and 3D fall to the unfused path
    # (3D PCHIP included — is3d gates the whole block).
    if (HAS_NUMBA and not is3d and uv_interp == "leastsq"
            and _leastsq_fused_anchor_exact(mesh)):
        if recon is not None:
            result = _run_fused_2d_pchip(
                state, recon, mesh, dt, time_dir, layer,
                _fused_rk2_step_2d_pchip, c_coeffs=(0.0, 0.5),
            )
        elif not time_aware:
            result = _run_fused_2d(state, vel, mesh, dt, time_dir, layer,
                                   _fused_rk2_step_2d)
        else:
            result = _run_fused_2d_tinterp(
                state, vel, vel_next, dt_to_next, mesh, dt, time_dir, layer,
                _fused_rk2_step_2d_tinterp, c_coeffs=(0.0, 0.5),
            )
    elif (HAS_NUMBA and not is3d and uv_interp == "barycentric"
          and time_aware and recon is None and state.elem is None
          and vel.uv_layout == "node" and _bary_fused_anchor_exact(mesh)):
        result = _run_fused_2d_bary_tinterp(
            state, vel, vel_next, dt_to_next, mesh, dt, time_dir, layer,
            _fused_rk2_step_2d_bary_tinterp, c_coeffs=(0.0, 0.5),
        )
    elif (HAS_NUMBA and is3d and uv_interp == "barycentric"
          and getattr(vel, "uv_layout", "element") == "node"
          and mesh.vertical_coord == "zlevel"
          and state.elem is None
          and w_interp in ("nearest", "barycentric")
          and not (hasattr(vel, "vel_a")
                   and float(getattr(vel, "alpha", 1.0)) == 0.0)):
        # Fused 3-D barycentric RK2 (2026-08): z-level node-layout only;
        # NO density-certificate gate — the kernels detect empty-window
        # anchors per particle and the runners rerun those few through
        # the exact unfused path (hybrid exactness), so the fused route
        # is bit-identical on ANY mesh, including high-zoom HEALPix
        # where the certificate is denied.
        # sigma meshes, walk-carried elements and the other w_interp
        # methods keep the unfused multi-pass path.  Steady, linear
        # tinterp and PCHIP all run in-kernel (per-vertex / per-interface
        # inline time blends — no per-stage whole-field materialization).
        w_code = W_NEAREST if w_interp == "nearest" else W_BARYCENTRIC
        if recon is not None:
            result = _run_fused_rk2_3d_bary_pchip(
                state, recon, mesh, dt, time_dir, w_code)
        else:
            result = _run_fused_rk2_3d_bary_tinterp(
                state, vel, vel_next, dt_to_next, mesh, dt, time_dir,
                w_code)
    else:
        result = _rk2_step_unfused(state, vel, mesh, dt, time_dir, layer, is3d,
                                   w_interp=w_interp,
                                   vel_next=vel_next, dt_to_next=dt_to_next,
                                   Dmin=Dmin, z0b_min=z0b_min,
                                   uv_interp=uv_interp, recon=recon,
                                   edge_leastsq_bc=edge_leastsq_bc,
                                   sigma_eta_end=sigma_eta_end,
                                   sigma_eta_start=sigma_eta_start)

    if diffusion is not None:
        if (diffusion.scheme in ("em", "m1")
                and diffusion.composition == "split_rk"):
            if is3d:
                raise NotImplementedError(
                    "diffusion.composition='split_rk' is 2D-only: "
                    "spatially varying horizontal Kh in 3D is not "
                    "implemented.  See docs/diffusion.md §5."
                )
            # Pick the Kh-evaluation field — for kh_eval_position='end'
            # with temporal-interp available, lerp to sim_time + dt so
            # the kick samples Kh at the post-RK time, not the start
            # snapshot (review C-#3).
            vel_kick = _pick_kh_eval_vel(
                vel, vel_next, dt_to_next, dt,
                diffusion.kh_eval_position,
            )
            result = _apply_split_rk_em_kick(
                state, result, mesh, dt, time_dir, diffusion, vel_kick,
                Dmin=Dmin, z0b_min=z0b_min,
            )
        else:
            sigma_col = _sigma_kick_col(state, vel, mesh, dt, is3d, w_interp,
                                        vel_next, dt_to_next, recon,
                                        sigma_eta_end, sigma_eta_start)
            result = _apply_split_diffusion_kick(
                state, result, mesh, dt, diffusion, vel,
                Dmin=Dmin, z0b_min=z0b_min, sigma_col=sigma_col,
            )
            result = _apply_split_az_kick(
                state, result, mesh, dt, diffusion, vel,
                time_dir=time_dir, Dmin=Dmin, z0b_min=z0b_min,
                sigma_col=sigma_col,
            )
    return result


# ---------------------------------------------------------------------------
# RK4 (classical 4th-order Runge-Kutta)
# ---------------------------------------------------------------------------

def _rk4_step_unfused(state, vel, mesh, dt, time_dir, layer, is3d,
                      w_interp="nearest", vel_next=None, dt_to_next=None,
                      Dmin=0.5, z0b_min=0.1, uv_interp="leastsq", recon=None,
                      edge_leastsq_bc="freeslip", sigma_eta_end=None,
                      sigma_eta_start=None):
    """Original multi-pass RK4 implementation (fallback).

    Time-aware mode (vel_next + dt_to_next given): each RK4 stage materializes
    velocity at its substep time via lerp(vel, vel_next, alpha_k) where
    alpha_k = c_k * dt / dt_to_next, c_k = (0, 0.5, 0.5, 1).

    Steady mode (vel_next=None or dt_to_next=None): all stages reuse `vel`,
    equivalent to original behavior — first-order in time for non-autonomous
    fields.

    ``uv_interp`` selects the u/v interpolator from the registry; every RK4
    stage now uses the vectorised (parallel) form.
    """
    interp_fn = get_interpolator(uv_interp, edge_leastsq_bc)
    alive = state.alive_mask
    P = state.n_particles

    # Compute per-stage velocity fields (one or four)
    if recon is not None:
        from .temporal_reconstruct import make_stage_vels_recon
        _snaps, _times, _sim_time, _tdir = recon
        stage_vels = make_stage_vels_recon(_snaps, _times, _sim_time, dt,
                                           _tdir, (0.0, 0.5, 0.5, 1.0))
    else:
        stage_vels = _make_stage_vels(vel, vel_next, dt_to_next, dt,
                                      c_coeffs=(0.0, 0.5, 0.5, 1.0))

    # Edge-walk element tracking (elem_search: walk)
    use_walk = state.elem is not None
    walk_interp = use_walk and uv_interp in UV_INTERP_USES_ELEMS

    # sigma meshes: integrate s = z / H_p (see _sigma_col_H); stage k sits
    # at z = s_k * H_p(x_k, t_k); a steady fallback step blends the surface
    # towards the tracking layer's eta_end
    sig3d = is3d and mesh.vertical_coord == "sigma"
    steady = recon is None and (vel_next is None or dt_to_next is None
                                or dt_to_next <= 0)
    c_rk4 = (0.0, 0.5, 0.5, 1.0)
    if sig3d:
        eta0 = _sigma_given(sigma_eta_start, stage_vels[0].eta)
        # the surface the next step starts from (tracking layer), else the
        # c = 1 stage's own; a steady step with neither holds its start
        if sigma_eta_end is not None:
            eta_end = sigma_eta_end
        else:
            eta_end = eta0 if steady else stage_vels[3].eta
        if steady:
            stage_etas = [_eta_blend(eta0, eta_end, c) for c in c_rk4]
        else:
            stage_etas = [eta0] + [sv.eta for sv in stage_vels[1:]]

    def _vel_at(lon, lat, z, mask, vel_at_stage, prev_elems, s=None,
                eta_stage=None):
        idx = np.where(mask)[0]
        W_out = np.zeros(P, dtype=np.float64)
        H_out = np.ones(P, dtype=np.float64)
        if idx.size == 0:
            U_out = np.zeros(P, dtype=np.float64)
            V_out = np.zeros(P, dtype=np.float64)
            ok_out = np.zeros(P, dtype=bool)
            return U_out, V_out, W_out, ok_out, prev_elems, H_out
        nn = find_nearest_nodes(mesh, lon[idx], lat[idx])
        nn_full_local = np.zeros(P, dtype=np.int32)
        nn_full_local[idx] = nn
        H_col = None
        if sig3d:
            H_out = _sigma_col_H(mesh, lon, lat, nn_full_local, eta_stage,
                                 mask, w_interp, Dmin)
            H_col = H_out
            if s is not None:
                z = np.where(mask, s * H_out, z)
            vel_at_stage = _sigma_stage_field(vel_at_stage, eta_stage)
        if is3d:
            layers_local = _compute_layers(
                mesh, vel_at_stage, z, nn_full_local, mask,
                Dmin=Dmin, z0b_min=z0b_min, H_col=H_col,
            )
            layer_arg = layers_local
        else:
            layer_arg = layer
        stage_elems = prev_elems
        if walk_interp:
            stage_elems, _ = _walk_update_elems(
                mesh, lon, lat, prev_elems, mask)
        # Per-stage u/v in parallel (was a serial per-particle loop).
        U_out, V_out, ok_out = _interp_uv_at(
            interp_fn, state, mesh, vel_at_stage,
            lon, lat, mask, nn_full_local, time_dir, layer_arg,
            elems=stage_elems if walk_interp else None,
        )
        if is3d:
            W_out = _compute_w(vel_at_stage, mesh, z, nn_full_local, mask, time_dir,
                              w_interp=w_interp, lons=lon, lats=lat,
                              Dmin=Dmin, z0b_min=z0b_min)
        return U_out, V_out, W_out, ok_out, stage_elems, H_out

    et = stage_etas if sig3d else [None] * 4
    U1, V1, W1, ok1, el1, H1 = _vel_at(state.lon, state.lat, state.z, alive,
                                       stage_vels[0], state.elem,
                                       eta_stage=et[0])
    z0_1 = _z0(state.lat)
    dlon1 = dt * U1 / z0_1
    dlat1 = dt * V1 / METERS_PER_DEG_LAT
    dz1 = dt * W1
    if sig3d:
        s0 = state.z / H1
        f1 = W1 / H1

    mask2 = alive & ok1
    lon2 = wrap_lon(state.lon + 0.5 * dlon1)
    lat2 = clamp_lat(state.lat + 0.5 * dlat1)
    z2 = state.z + 0.5 * dz1
    U2, V2, W2, ok2, el2, H2 = _vel_at(
        lon2, lat2, z2, mask2, stage_vels[1], el1,
        s=(s0 + 0.5 * dt * f1) if sig3d else None, eta_stage=et[1])
    z0_2 = _z0(lat2)
    dlon2 = dt * U2 / z0_2
    dlat2 = dt * V2 / METERS_PER_DEG_LAT
    dz2 = dt * W2
    if sig3d:
        f2 = W2 / H2

    mask3 = mask2 & ok2
    lon3 = wrap_lon(state.lon + 0.5 * dlon2)
    lat3 = clamp_lat(state.lat + 0.5 * dlat2)
    z3 = state.z + 0.5 * dz2
    U3, V3, W3, ok3, el3, H3 = _vel_at(
        lon3, lat3, z3, mask3, stage_vels[2], el2,
        s=(s0 + 0.5 * dt * f2) if sig3d else None, eta_stage=et[2])
    z0_3 = _z0(lat3)
    dlon3 = dt * U3 / z0_3
    dlat3 = dt * V3 / METERS_PER_DEG_LAT
    dz3 = dt * W3
    if sig3d:
        f3 = W3 / H3

    mask4 = mask3 & ok3
    lon4 = wrap_lon(state.lon + dlon3)
    lat4 = clamp_lat(state.lat + dlat3)
    z4 = state.z + dz3
    U4, V4, W4, ok4, el4, H4 = _vel_at(
        lon4, lat4, z4, mask4, stage_vels[3], el3,
        s=(s0 + dt * f3) if sig3d else None, eta_stage=et[3])
    z0_4 = _z0(lat4)
    dlon4 = dt * U4 / z0_4
    dlat4 = dt * V4 / METERS_PER_DEG_LAT
    dz4 = dt * W4
    if sig3d:
        f4 = W4 / H4

    ok = ok1 & ok2 & ok3 & ok4
    moves = alive & ok
    dlon = (dlon1 + 2.0 * dlon2 + 2.0 * dlon3 + dlon4) / 6.0
    dlat = (dlat1 + 2.0 * dlat2 + 2.0 * dlat3 + dlat4) / 6.0
    dz = (dz1 + 2.0 * dz2 + 2.0 * dz3 + dz4) / 6.0

    old_lon = state.lon.copy()
    old_lat = state.lat.copy()
    old_z = state.z.copy()
    # wrap/clamp ONLY the moved entries — a frozen particle comes out verbatim
    # (same rule as _apply_displacement; review 2026-09-12)
    lon_new = np.where(moves, wrap_lon(state.lon + dlon), state.lon)
    lat_new = np.where(moves, clamp_lat(state.lat + dlat), state.lat)
    if sig3d:
        s_new = s0 + dt * (f1 + 2.0 * f2 + 2.0 * f3 + f4) / 6.0
        z_new = state.z.copy()          # set after the domain check
    else:
        z_new = (np.where(moves, state.z + dz, state.z) if is3d
                 else state.z.copy())

    moved_idx = np.where(moves)[0]
    nn_full = np.copy(state.nearest_node)
    in_domain_full = np.zeros(P, dtype=bool)
    elem_new = None
    if use_walk:
        start_elems = el1 if walk_interp else state.elem
        elem_new, walk_ok = _walk_update_elems(
            mesh, lon_new, lat_new,
            el4 if walk_interp else state.elem, moves)
    if moved_idx.size > 0:
        m_nn = find_nearest_nodes(mesh, lon_new[moved_idx], lat_new[moved_idx])
        nn_full[moved_idx] = m_nn
        if use_walk:
            in_domain_full[moved_idx] = walk_ok[moved_idx]
        else:
            m_in = _check_in_domain(mesh, lon_new[moved_idx],
                                    lat_new[moved_idx], m_nn)
            in_domain_full[moved_idx] = m_in

    interp_failed = alive & ~ok
    domain_failed = alive & ok & moves & ~in_domain_full
    failed = interp_failed | domain_failed
    new_status = state.status.copy()
    new_status[failed] = ParticleStatus.OUTSIDE
    lon_new = np.where(failed, old_lon, lon_new)
    lat_new = np.where(failed, old_lat, lat_new)
    z_new = np.where(failed, old_z, z_new)
    # Failed particles revert position; refresh their nearest_node at the
    # (unchanged) start position — matching the RK2 unfused restore (nn1)
    # and the fused kernels' start anchor, instead of echoing a possibly
    # stale/-1 incoming value (round-2 review: the only bitwise
    # fused/unfused divergence left was dead particles' nearest_node).
    fail_idx = np.where(failed)[0]
    if fail_idx.size:
        nn_full[fail_idx] = find_nearest_nodes(
            mesh, state.lon[fail_idx], state.lat[fail_idx])
    if use_walk:
        elem_new = np.where(failed, start_elems, elem_new)
        elem_new = np.where(~alive, state.elem, elem_new).astype(np.int32)

    if sig3d:
        ok_mv = alive & ~failed
        H_end = _sigma_col_H(mesh, lon_new, lat_new, nn_full, eta_end,
                             ok_mv, w_interp, Dmin)
        z_new = np.where(ok_mv, s_new * H_end, z_new)
        z_new = _clamp_z(z_new, stage_vels[3], mesh, nn_full, ok_mv,
                         Dmin=Dmin, z0b_min=z0b_min, H_col=H_end)
    elif is3d:
        z_new = _clamp_z(
            z_new, stage_vels[3], mesh, nn_full, alive & ~failed,
            Dmin=Dmin, z0b_min=z0b_min,
        )

    new_failure_reason = _update_failure_reason(
        state.failure_reason,
        {FailureReason.INTERPOLATION_FAILED: interp_failed,
         FailureReason.OUT_OF_BOUNDS: domain_failed},
    )

    out = state.replace(
        lon=lon_new, lat=lat_new, z=z_new,
        status=new_status, nearest_node=nn_full,
        u=U1, v=V1, w=W1 if is3d else None,
        failure_reason=new_failure_reason,
    )
    if use_walk:
        out = out.replace(elem=elem_new)
    return out


def rk4_step(
    state: ParticleState,
    vel: VelocityField,
    mesh: Mesh,
    dt: float,
    time_dir: float,
    layer: int = 0,
    **kwargs,
) -> ParticleState:
    """Single RK4 step (classical 4th-order Runge-Kutta).

    Uses a single velocity field (no temporal interpolation).
    Four spatial evaluations at staggered positions.
    """
    diffusion: Optional[DiffusionStepArgs] = kwargs.pop("diffusion", None)
    _reject_em_m1_az(diffusion)
    _reject_combined_em_on_rk(diffusion, "rk4")
    alive = state.alive_mask
    if not np.any(alive):
        return _zero_dead_velocity(state)

    w_interp = kwargs.get("w_interp", "leastsq")
    uv_interp = kwargs.get("uv_interp", "leastsq")
    edge_leastsq_bc = kwargs.get("edge_leastsq_bc", "freeslip")
    Dmin = kwargs.get("Dmin", 0.5)
    z0b_min = kwargs.get("z0b_min", 0.1)
    vel_next = kwargs.get("vel_next", None)
    dt_to_next = kwargs.get("dt_to_next", None)
    # sigma meshes: the surface at the step's start and end (supplied by the
    # tracking layer for every step, one function of time)
    sigma_eta_end = kwargs.get("sigma_eta_end", None)
    sigma_eta_start = kwargs.get("sigma_eta_start", None)
    recon = kwargs.get("recon", None)
    _validate_time_aware_kwargs(vel_next, dt_to_next)
    is3d = _is_3d(vel)
    time_aware = vel_next is not None  # both or neither (validated above)

    # The fused 2D RK4 kernels bake leastsq u/v math; eligible only when
    # uv_interp is leastsq and the field is 2D.  A PCHIP recon bundle now has
    # its own fused 4-snapshot cubic kernel; the steady and 2-snapshot linear
    # blends keep their kernels.  Node-layout barycentric with temporal
    # interpolation has a fused twin too (per-vertex inline blend — no
    # per-stage whole-node-field lerp); element-layout barycentric, walk
    # (state.elem carried), steady, PCHIP and 3D fall to the unfused path
    # (3D PCHIP included — is3d gates the whole block).
    if (HAS_NUMBA and not is3d and uv_interp == "leastsq"
            and _leastsq_fused_anchor_exact(mesh)):
        if recon is not None:
            result = _run_fused_2d_pchip(
                state, recon, mesh, dt, time_dir, layer,
                _fused_rk4_step_2d_pchip, c_coeffs=(0.0, 0.5, 0.5, 1.0),
            )
        elif not time_aware:
            result = _run_fused_2d(state, vel, mesh, dt, time_dir, layer,
                                   _fused_rk4_step_2d)
        else:
            result = _run_fused_2d_tinterp(
                state, vel, vel_next, dt_to_next, mesh, dt, time_dir, layer,
                _fused_rk4_step_2d_tinterp, c_coeffs=(0.0, 0.5, 0.5, 1.0),
            )
    elif (HAS_NUMBA and not is3d and uv_interp == "barycentric"
          and time_aware and recon is None and state.elem is None
          and vel.uv_layout == "node" and _bary_fused_anchor_exact(mesh)):
        result = _run_fused_2d_bary_tinterp(
            state, vel, vel_next, dt_to_next, mesh, dt, time_dir, layer,
            _fused_rk4_step_2d_bary_tinterp, c_coeffs=(0.0, 0.5, 0.5, 1.0),
        )
    else:
        result = _rk4_step_unfused(state, vel, mesh, dt, time_dir, layer, is3d,
                                   w_interp=w_interp,
                                   vel_next=vel_next, dt_to_next=dt_to_next,
                                   Dmin=Dmin, z0b_min=z0b_min,
                                   uv_interp=uv_interp, recon=recon,
                                   edge_leastsq_bc=edge_leastsq_bc,
                                   sigma_eta_end=sigma_eta_end,
                                   sigma_eta_start=sigma_eta_start)

    if diffusion is not None:
        if (diffusion.scheme in ("em", "m1")
                and diffusion.composition == "split_rk"):
            if is3d:
                raise NotImplementedError(
                    "diffusion.composition='split_rk' is 2D-only: "
                    "spatially varying horizontal Kh in 3D is not "
                    "implemented.  See docs/diffusion.md §5."
                )
            # Pick the Kh-evaluation field — for kh_eval_position='end'
            # with temporal-interp available, lerp to sim_time + dt so
            # the kick samples Kh at the post-RK time, not the start
            # snapshot (review C-#3).
            vel_kick = _pick_kh_eval_vel(
                vel, vel_next, dt_to_next, dt,
                diffusion.kh_eval_position,
            )
            result = _apply_split_rk_em_kick(
                state, result, mesh, dt, time_dir, diffusion, vel_kick,
                Dmin=Dmin, z0b_min=z0b_min,
            )
        else:
            sigma_col = _sigma_kick_col(state, vel, mesh, dt, is3d, w_interp,
                                        vel_next, dt_to_next, recon,
                                        sigma_eta_end, sigma_eta_start)
            result = _apply_split_diffusion_kick(
                state, result, mesh, dt, diffusion, vel,
                Dmin=Dmin, z0b_min=z0b_min, sigma_col=sigma_col,
            )
            result = _apply_split_az_kick(
                state, result, mesh, dt, diffusion, vel,
                time_dir=time_dir, Dmin=Dmin, z0b_min=z0b_min,
                sigma_col=sigma_col,
            )
    return result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# RK45 — Dormand-Prince 5(4) with per-particle adaptive substepping
# (TODO 2.3; design notes in docs/time_integration.md section 7)
# ---------------------------------------------------------------------------

# DP54 Butcher tableau (Dormand & Prince 1980; the ode45 coefficients).
_DP_C2 = 1.0 / 5.0
_DP_C3 = 3.0 / 10.0
_DP_C4 = 4.0 / 5.0
_DP_C5 = 8.0 / 9.0
_DP_A21 = 1.0 / 5.0
_DP_A31 = 3.0 / 40.0
_DP_A32 = 9.0 / 40.0
_DP_A41 = 44.0 / 45.0
_DP_A42 = -56.0 / 15.0
_DP_A43 = 32.0 / 9.0
_DP_A51 = 19372.0 / 6561.0
_DP_A52 = -25360.0 / 2187.0
_DP_A53 = 64448.0 / 6561.0
_DP_A54 = -212.0 / 729.0
_DP_A61 = 9017.0 / 3168.0
_DP_A62 = -355.0 / 33.0
_DP_A63 = 46732.0 / 5247.0
_DP_A64 = 49.0 / 176.0
_DP_A65 = -5103.0 / 18656.0
# 5th-order solution weights (row 7 of A; B2 = B7 = 0)
_DP_B1 = 35.0 / 384.0
_DP_B3 = 500.0 / 1113.0
_DP_B4 = 125.0 / 192.0
_DP_B5 = -2187.0 / 6784.0
_DP_B6 = 11.0 / 84.0
# error weights: b5 - b4 (embedded 4th order), as EXACT fractions (the
# float subtraction of the b-rows is 1-2 ulp off these; the canonical
# values are Hairer's printed DOPRI5 error coefficients); E2 = 0
_DP_E1 = 71.0 / 57600.0
_DP_E3 = -71.0 / 16695.0
_DP_E4 = 71.0 / 1920.0
_DP_E5 = -17253.0 / 339200.0
_DP_E6 = 22.0 / 525.0
_DP_E7 = -1.0 / 40.0

_RK45_MAX_SUBSTEPS = 10_000


@njit(cache=True)
def _norm_lonlat(lon: float, lat: float):
    """Normalize a trial position: wrap lon into [-180, 180) only when it
    left the range (the unconditional modulo costs ~1e-13 deg on tiny
    longitudes), clamp lat to the poles — the same treatment rk2/rk4
    stage positions get via ``_apply_displacement``."""
    if lon >= 180.0 or lon < -180.0:
        lon = ((lon + 180.0) % 360.0) - 180.0
    if lat > 90.0:
        lat = 90.0
    elif lat < -90.0:
        lat = -90.0
    return lon, lat


@njit(cache=True)
def _rk45_eval(lon, lat, alpha, time_dir,
               gh_offsets, gh_node_ids, gh_cell_size, gh_lon_min,
               gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
               n2e_count, n2e, ec_lon, ec_lat, nodes_lon, nodes_lat, node_xyz,
               u_a, v_a, u_b, v_b, u_t, v_t, alpha0, exact, far_rows):
    """One RK45 stage: velocity (m/s) and degree-rates at (lon, lat).

    Returns ``(U, V, dlon_dt, dlat_dt, ok)``: raw signed velocity for
    the diagnostics plus the ODE right-hand side in degrees/s with the
    pole-guarded metric.  Time enters through the nested tinterp blend
    (``alpha0`` = bracket fraction at step start, ``alpha`` = stage
    fraction toward ``u_t``; both 0 → plain leastsq bitwise).
    """
    # the same per-site anchor as the fused kernels: the widened window
    # search, or under mesh.exact_nearest the in-kernel exact search
    nn = _fused_site_anchor(
        lon, lat, gh_offsets, gh_node_ids, gh_cell_size,
        gh_lon_min, gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
        nodes_lon, nodes_lat, node_xyz, exact, far_rows)
    if nn < 0:
        # defensive: both searches return -1 only when the whole hash is
        # empty (the widened one ring-heals ordinary misses since 2026-08)
        return 0.0, 0.0, 0.0, 0.0, False
    U, V, ok = _interp_leastsq_core_tinterp(
        lon, lat, nn, time_dir, alpha0, alpha,
        n2e_count, n2e, ec_lon, ec_lat,
        u_a, v_a, u_b, v_b, u_t, v_t)
    if not ok:
        return 0.0, 0.0, 0.0, 0.0, False
    z0 = np.cos(lat * 0.017453292519943295) * _SLM
    if z0 < 1.0:
        z0 = 1.0
    return U, V, U / z0, V / _SLM, True


@njit(parallel=True, cache=True)
def _rk45_kernel_2d(P, lons, lats, alive,
                    gh_offsets, gh_node_ids, gh_cell_size, gh_lon_min,
                    gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
                    n2e_count, n2e, ec_lon, ec_lat, nodes_lon, nodes_lat, node_xyz,
                    u_a, v_a, u_b, v_b, u_t, v_t, alpha0, inv_dtn,
                    dt, time_dir, tol_m, h_min, max_sub,
                    lon_out, lat_out, ok_out, U_out, V_out,
                    nsub_out, clamp_out, exact, far_rows):
    """Per-particle adaptive DP54 integration over exactly ``dt`` seconds.

    Classic controller: reject when the embedded-error estimate (metres)
    exceeds ``tol_m``; ``h ← h·clip(0.9·(tol/err)^{1/5}, 0.2, 5)``; the
    final substep is clipped to land exactly on ``dt``; FSAL carries the
    7th stage into the next substep's first.  ``h`` never drops below
    ``h_min`` — a clamped substep is force-accepted (counted when it was
    actually above tolerance), so the loop always terminates in at most
    ``~dt/h_min`` accepts; ``max_sub`` (wrapper-computed to exceed that
    by construction) only backstops pathological inputs.  A FAILED stage
    is a rejected attempt retried at ``h·0.2`` — terminal only at the
    ``h_min`` floor or the cap → ``ok_out=False`` (wrapper reverts).
    """
    for i in prange(P):
        lon_out[i] = lons[i]
        lat_out[i] = lats[i]
        ok_out[i] = True
        nsub_out[i] = 0
        clamp_out[i] = 0
        if not alive[i]:
            continue
        y0 = lons[i]
        y1 = lats[i]
        t = 0.0
        h = dt
        a_now = t * inv_dtn
        if a_now > 1.0:
            a_now = 1.0
        Uv, Vv, k1x, k1y, ok = _rk45_eval(
            y0, y1, a_now, time_dir,
            gh_offsets, gh_node_ids, gh_cell_size, gh_lon_min,
            gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
            n2e_count, n2e, ec_lon, ec_lat, nodes_lon, nodes_lat, node_xyz,
            u_a, v_a, u_b, v_b, u_t, v_t, alpha0, exact, far_rows)
        if not ok:
            ok_out[i] = False
            continue
        nsub = 0
        k2x = 0.0; k2y = 0.0; k3x = 0.0; k3y = 0.0
        k4x = 0.0; k4y = 0.0; k5x = 0.0; k5y = 0.0
        k6x = 0.0; k6y = 0.0; k7x = 0.0; k7y = 0.0
        U7 = 0.0; V7 = 0.0
        y5x = y0; y5y = y1
        while t < dt * (1.0 - 1e-12):
            if h > dt - t:
                h = dt - t
            clamped = False
            if h < h_min:
                h = h_min
                if h > dt - t:
                    h = dt - t
                clamped = True

            # -- stages 2..6 (trial positions normalized like the rk2/rk4
            #    stages; okst chains so one failed stage skips the rest) --
            okst = True
            a2 = (t + _DP_C2 * h) * inv_dtn
            if a2 > 1.0:
                a2 = 1.0
            p_lon, p_lat = _norm_lonlat(y0 + h * _DP_A21 * k1x,
                                        y1 + h * _DP_A21 * k1y)
            _, _, k2x, k2y, ok = _rk45_eval(
                p_lon, p_lat, a2, time_dir,
                gh_offsets, gh_node_ids, gh_cell_size, gh_lon_min,
                gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
                n2e_count, n2e, ec_lon, ec_lat, nodes_lon, nodes_lat, node_xyz,
                u_a, v_a, u_b, v_b, u_t, v_t, alpha0, exact, far_rows)
            okst = okst and ok
            if okst:
                a3 = (t + _DP_C3 * h) * inv_dtn
                if a3 > 1.0:
                    a3 = 1.0
                p_lon, p_lat = _norm_lonlat(
                    y0 + h * (_DP_A31 * k1x + _DP_A32 * k2x),
                    y1 + h * (_DP_A31 * k1y + _DP_A32 * k2y))
                _, _, k3x, k3y, ok = _rk45_eval(
                    p_lon, p_lat, a3, time_dir,
                    gh_offsets, gh_node_ids, gh_cell_size, gh_lon_min,
                    gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
                    n2e_count, n2e, ec_lon, ec_lat, nodes_lon, nodes_lat, node_xyz,
                    u_a, v_a, u_b, v_b, u_t, v_t, alpha0, exact, far_rows)
                okst = okst and ok
            if okst:
                a4 = (t + _DP_C4 * h) * inv_dtn
                if a4 > 1.0:
                    a4 = 1.0
                p_lon, p_lat = _norm_lonlat(
                    y0 + h * (_DP_A41 * k1x + _DP_A42 * k2x + _DP_A43 * k3x),
                    y1 + h * (_DP_A41 * k1y + _DP_A42 * k2y + _DP_A43 * k3y))
                _, _, k4x, k4y, ok = _rk45_eval(
                    p_lon, p_lat, a4, time_dir,
                    gh_offsets, gh_node_ids, gh_cell_size, gh_lon_min,
                    gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
                    n2e_count, n2e, ec_lon, ec_lat, nodes_lon, nodes_lat, node_xyz,
                    u_a, v_a, u_b, v_b, u_t, v_t, alpha0, exact, far_rows)
                okst = okst and ok
            if okst:
                a5 = (t + _DP_C5 * h) * inv_dtn
                if a5 > 1.0:
                    a5 = 1.0
                p_lon, p_lat = _norm_lonlat(
                    y0 + h * (_DP_A51 * k1x + _DP_A52 * k2x
                              + _DP_A53 * k3x + _DP_A54 * k4x),
                    y1 + h * (_DP_A51 * k1y + _DP_A52 * k2y
                              + _DP_A53 * k3y + _DP_A54 * k4y))
                _, _, k5x, k5y, ok = _rk45_eval(
                    p_lon, p_lat, a5, time_dir,
                    gh_offsets, gh_node_ids, gh_cell_size, gh_lon_min,
                    gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
                    n2e_count, n2e, ec_lon, ec_lat, nodes_lon, nodes_lat, node_xyz,
                    u_a, v_a, u_b, v_b, u_t, v_t, alpha0, exact, far_rows)
                okst = okst and ok
            a6 = (t + h) * inv_dtn
            if a6 > 1.0:
                a6 = 1.0
            if okst:
                p_lon, p_lat = _norm_lonlat(
                    y0 + h * (_DP_A61 * k1x + _DP_A62 * k2x + _DP_A63 * k3x
                              + _DP_A64 * k4x + _DP_A65 * k5x),
                    y1 + h * (_DP_A61 * k1y + _DP_A62 * k2y + _DP_A63 * k3y
                              + _DP_A64 * k4y + _DP_A65 * k5y))
                _, _, k6x, k6y, ok = _rk45_eval(
                    p_lon, p_lat, a6, time_dir,
                    gh_offsets, gh_node_ids, gh_cell_size, gh_lon_min,
                    gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
                    n2e_count, n2e, ec_lon, ec_lat, nodes_lon, nodes_lat, node_xyz,
                    u_a, v_a, u_b, v_b, u_t, v_t, alpha0, exact, far_rows)
                okst = okst and ok

            # -- 5th-order solution, NORMALIZED, then FSAL stage 7 there --
            if okst:
                y5x, y5y = _norm_lonlat(
                    y0 + h * (_DP_B1 * k1x + _DP_B3 * k3x + _DP_B4 * k4x
                              + _DP_B5 * k5x + _DP_B6 * k6x),
                    y1 + h * (_DP_B1 * k1y + _DP_B3 * k3y + _DP_B4 * k4y
                              + _DP_B5 * k5y + _DP_B6 * k6y))
                U7, V7, k7x, k7y, ok = _rk45_eval(
                    y5x, y5y, a6, time_dir,
                    gh_offsets, gh_node_ids, gh_cell_size, gh_lon_min,
                    gh_lat_min, gh_nx, gh_ny, gh_wrap_lon,
                    n2e_count, n2e, ec_lon, ec_lat, nodes_lon, nodes_lat, node_xyz,
                    u_a, v_a, u_b, v_b, u_t, v_t, alpha0, exact, far_rows)
                okst = okst and ok

            nsub += 1
            if not okst:
                # A failed stage is a REJECTED attempt, not a death: a
                # too-large trial can probe past a concave coast where a
                # smaller substep stays valid (review 2026-08-16).  Only
                # a failure at the h_min floor (or the cap) is terminal.
                if nsub > max_sub or h <= h_min * (1.0 + 1e-12):
                    ok_out[i] = False
                    break
                h = h * 0.2
                if h < h_min:
                    h = h_min
                continue

            errx = h * (_DP_E1 * k1x + _DP_E3 * k3x + _DP_E4 * k4x
                        + _DP_E5 * k5x + _DP_E6 * k6x + _DP_E7 * k7x)
            erry = h * (_DP_E1 * k1y + _DP_E3 * k3y + _DP_E4 * k4y
                        + _DP_E5 * k5y + _DP_E6 * k6y + _DP_E7 * k7y)
            z0m = np.cos(y1 * 0.017453292519943295) * _SLM
            if z0m < 1.0:
                z0m = 1.0
            err = np.sqrt((errx * z0m) ** 2 + (erry * _SLM) ** 2)

            if nsub > max_sub:
                ok_out[i] = False
                break

            if err <= tol_m or clamped:
                # accept: advance to the already-normalized endpoint; the
                # FSAL stage was evaluated exactly there, so k7 -> k1.
                t += h
                y0 = y5x
                y1 = y5y
                k1x = k7x
                k1y = k7y
                Uv = U7
                Vv = V7
                if clamped and err > tol_m:
                    # count (and later warn about) only substeps that were
                    # genuinely forced through ABOVE tolerance — an exact
                    # step that merely ran at the floor is not a problem
                    # (review 2026-08-16)
                    clamp_out[i] += 1

            # controller (both accept and reject)
            if err > 0.0:
                fac = 0.9 * (tol_m / err) ** 0.2
                if fac < 0.2:
                    fac = 0.2
                elif fac > 5.0:
                    fac = 5.0
            else:
                fac = 5.0
            h = h * fac

        if ok_out[i]:
            lon_out[i] = y0
            lat_out[i] = y1
            U_out[i] = Uv
            V_out[i] = Vv
        nsub_out[i] = nsub


def rk45_step(
    state: ParticleState,
    vel: VelocityField,
    mesh: Mesh,
    dt: float,
    time_dir: float,
    layer: int = 0,
    **kwargs,
) -> ParticleState:
    """Adaptive Dormand-Prince 5(4) step over exactly ``dt`` seconds.

    Per-particle adaptive substepping inside the fixed tracking step —
    externally a drop-in for euler/rk2/rk4 (TODO 2.3).  v1 supports 2-D
    element-layout velocities with ``uv_interp='leastsq'``, steady or
    linearly time-interpolated (``vel_next``/``dt_to_next``); everything
    else fails fast.  Stage times sample the same nested tinterp blend
    as the fused RK kernels; the error estimate is the embedded
    4th-order difference converted to metres, compared against
    ``rk45_tol_m`` per substep.

    Extra kwargs: ``rk45_tol_m`` (default 1.0 m), ``rk45_h_min``
    (default ``dt * 1e-4`` seconds), ``rk45_diag`` (optional dict —
    filled with ``nsub`` per particle, the total ``hmin_clamped``
    count and ``lon_end``/``lat_end``, the position where the returned
    ``u``/``v`` were evaluated — the end of the step BEFORE any diffusion
    kick; the tracking layer un-rotates the recorded velocity there).
    """
    diffusion: Optional[DiffusionStepArgs] = kwargs.pop("diffusion", None)
    _reject_em_m1_az(diffusion)
    if diffusion is not None and diffusion.scheme in ("em", "m1"):
        raise ValueError(
            "diffusion scheme 'em'/'m1' is not supported with the rk45 "
            "integrator (v1); use scheme='uniform' or rk2/rk4 + split_rk")
    uv_interp = kwargs.get("uv_interp", "leastsq")
    if uv_interp != "leastsq":
        raise NotImplementedError(
            f"rk45 supports uv_interp='leastsq' only (v1), got {uv_interp!r}")
    if kwargs.get("recon", None) is not None:
        raise NotImplementedError(
            "rk45 does not support temporal_reconstruction='pchip' (v1) — "
            "per-particle substep times cannot use the 4-snapshot recon "
            "bundle yet; use temporal_reconstruction='linear'")
    if state.elem is not None:
        raise ValueError(
            "rk45 does not support elem_search='walk' (v1): carried "
            "elements would be neither used nor updated")
    if _is_3d(vel):
        raise NotImplementedError(
            "rk45 is 2-D only (v1); 3-D adaptive stepping is a staged "
            "follow-up")
    if getattr(vel, "uv_layout", "element") != "element":
        raise ValueError(
            "rk45 requires element-layout velocities (v1); node-layout "
            "(unod/vnod, NEMO) needs the barycentric path")
    if not _leastsq_fused_anchor_exact(mesh):
        # rk45 has NO unfused fallback (per-particle adaptive kernel with
        # the widened in-kernel anchor) — a pixel-geometry mesh that fails
        # the grid-hash density certificate would get the same polar-row
        # anchor divergence the euler/rk2/rk4 gate prevents (2026-08
        # review round 3).
        raise ValueError(
            "rk45 on a HEALPix pixel-geometry mesh requires the grid-hash "
            "density certificate, which this mesh fails (polar rows in "
            "the geographic frame) — use mesh.rotation (rotated meshes "
            "pass) or the rk2/rk4 integrators (they fall back to the "
            "unfused exact path).")

    vel_next = kwargs.get("vel_next", None)
    dt_to_next = kwargs.get("dt_to_next", None)
    _validate_time_aware_kwargs(vel_next, dt_to_next)
    if vel_next is not None:
        _validate_time_aware_inputs(vel, vel_next, dt_to_next, dt)

    tol_m = float(kwargs.get("rk45_tol_m", 1.0))
    h_min = kwargs.get("rk45_h_min", None)
    h_min = float(h_min) if h_min is not None else float(dt) * 1e-4
    # The cap must EXCEED the worst legitimate substep count — covering
    # dt in h_min-clamped accepts plus their rejections — or the default
    # h_min = dt*1e-4 collides with a fixed 1e4 cap and kills particles
    # the docs promise get force-accepted + warned (review 2026-08-16).
    max_sub = max(_RK45_MAX_SUBSTEPS, 2 * int(float(dt) / h_min) + 16)
    diag = kwargs.get("rk45_diag", None)
    Dmin = kwargs.get("Dmin", 0.5)
    z0b_min = kwargs.get("z0b_min", 0.1)

    # Bracket pair / stage target (mirrors _run_fused_2d_tinterp).
    vel_a_full = getattr(vel, "vel_a", vel)
    vel_b_full = getattr(vel, "vel_b", vel_next if vel_next is not None
                         else vel)
    alpha0 = float(getattr(vel, "alpha", 0.0))
    u_a = vel_a_full.u if vel_a_full.u.ndim == 1 else vel_a_full.u[layer, :]
    v_a = vel_a_full.v if vel_a_full.v.ndim == 1 else vel_a_full.v[layer, :]
    u_b = vel_b_full.u if vel_b_full.u.ndim == 1 else vel_b_full.u[layer, :]
    v_b = vel_b_full.v if vel_b_full.v.ndim == 1 else vel_b_full.v[layer, :]
    if vel_next is not None:
        u_t = vel_next.u if vel_next.u.ndim == 1 else vel_next.u[layer, :]
        v_t = vel_next.v if vel_next.v.ndim == 1 else vel_next.v[layer, :]
        inv_dtn = 1.0 / float(dt_to_next)
    else:
        u_t, v_t = u_a, v_a
        inv_dtn = 0.0

    P = state.n_particles
    alive = state.alive_mask
    lon_out = state.lon.copy()
    lat_out = state.lat.copy()
    ok_out = np.ones(P, dtype=np.bool_)
    U_out = np.zeros(P, dtype=np.float64)
    V_out = np.zeros(P, dtype=np.float64)
    nsub_out = np.zeros(P, dtype=np.int64)
    clamp_out = np.zeros(P, dtype=np.int64)

    _rk45_kernel_2d(
        P, state.lon, state.lat, alive,
        mesh._gh_offsets, mesh._gh_node_ids, mesh._gh_cell_size,
        mesh._gh_lon_min, mesh._gh_lat_min, mesh._gh_nx, mesh._gh_ny,
        mesh._gh_wrap_lon,
        mesh.node_to_elems_count, mesh.node_to_elems,
        mesh.elem_center_lon, mesh.elem_center_lat,
        mesh.nodes_lon, mesh.nodes_lat, _require_node_xyz(mesh),
        u_a, v_a, u_b, v_b, u_t, v_t, alpha0, inv_dtn,
        float(dt), float(time_dir), tol_m, h_min, max_sub,
        lon_out, lat_out, ok_out, U_out, V_out, nsub_out, clamp_out,
        _exact_nearest_on(mesh), int(mesh._gh_far_rows),
    )

    clamp_total = int(clamp_out.sum())
    if clamp_total > 0:
        import warnings
        warnings.warn(
            f"rk45: {clamp_total} substep(s) hit rk45_h_min={h_min:.3g}s "
            "and were force-accepted above rk45_tol_m — tighten h_min or "
            "loosen the tolerance",
            RuntimeWarning, stacklevel=2)

    # Re-validate final positions (unfused-style): nearest node + domain.
    nn_new = state.nearest_node.copy()
    in_domain = np.ones(P, dtype=bool)
    moved = alive & ok_out
    idx = np.where(moved)[0]
    if idx.size > 0:
        m_nn = find_nearest_nodes(mesh, lon_out[idx], lat_out[idx])
        m_in = _check_in_domain(mesh, lon_out[idx], lat_out[idx], m_nn)
        nn_new[idx] = m_nn
        in_domain[idx] = m_in

    interp_failed = alive & ~ok_out
    domain_failed = alive & ok_out & ~in_domain
    failed = interp_failed | domain_failed
    status_new = state.status.copy()
    status_new[failed] = ParticleStatus.OUTSIDE
    lon_final = np.where(failed, state.lon, lon_out)
    lat_final = np.where(failed, state.lat, lat_out)
    nn_final = np.where(failed, state.nearest_node, nn_new)
    fr_new = _update_failure_reason(
        state.failure_reason,
        {FailureReason.INTERPOLATION_FAILED: interp_failed,
         FailureReason.OUT_OF_BOUNDS: domain_failed})

    result = state.replace(
        lon=lon_final, lat=lat_final, z=state.z.copy(),
        status=status_new, nearest_node=nn_final,
        u=U_out, v=V_out, w=None,
        failure_reason=fr_new,
    )

    if diag is not None:
        diag["nsub"] = nsub_out
        diag["hmin_clamped"] = clamp_total
        diag["lon_end"] = lon_final.copy()
        diag["lat_end"] = lat_final.copy()

    if diffusion is not None:
        result = _apply_split_diffusion_kick(
            state, result, mesh, dt, diffusion, vel,
            Dmin=Dmin, z0b_min=z0b_min,
        )
        result = _apply_split_az_kick(
            state, result, mesh, dt, diffusion, vel,
            time_dir=time_dir, Dmin=Dmin, z0b_min=z0b_min,
        )
    return result


INTEGRATORS = {
    "euler": euler_step,
    "rk2": rk2_step,
    "rk4": rk4_step,
    "rk45": rk45_step,
}


def get_integrator(name: str):
    """Return the integrator function by name.

    Parameters
    ----------
    name : str
        One of "euler", "rk2", "rk4", "rk45".

    Returns
    -------
    Callable
        The integrator step function.
    """
    if name not in INTEGRATORS:
        raise ValueError(f"Unknown integrator '{name}'. Available: {sorted(INTEGRATORS)}")
    return INTEGRATORS[name]


__all__ = [
    "euler_step",
    "rk2_step",
    "rk4_step",
    "get_integrator",
    "INTEGRATORS",
]
