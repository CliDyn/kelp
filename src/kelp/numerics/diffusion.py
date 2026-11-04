"""Stochastic diffusion kernels for kelp.

numba-cache-bust 2026-07-03: the leastsq drift core this module inlines
gained the antimeridian gate; touching THIS file invalidates the cached
combined/split-RK kernels so a warm clone recompiles them after a pull.

This module hosts the counter-based RNG and (in later phases) the SDE
step kernels used by the integrators.  Each random draw is fully
determined by the input key tuple ``(seed, real_id, particle_i,
step_i, comp_id)`` — there is no global RNG state — which makes the
stream:

* parallel-safe (any number of threads may evaluate any subset of
  particles, in any order, with bit-identical results);
* restart-safe (resuming from a checkpoint at ``step_i = N`` produces
  the same draws as a single continuous run);
* reproducible across machines and across alive-mask compactions, as
  long as ``particle_i`` stays the *original* index in the
  ``ParticleState`` arrays.

The hash is a SplitMix64-style mixer (Vigna).  It is not a CSPRNG, but
it is well-tested for Monte Carlo / Quasi-Monte Carlo and is what
``numpy.random.SeedSequence`` uses internally.  64-bit precision is
ample for the per-step normals consumed here.

The variable-Kh stage adds spatial Kh sampling helpers:

* ``kh_value_at_point`` — interpolate a Kh field at an arbitrary
  particle position, dispatching by ``kh_layout``;
* ``kh_gradient_at_point`` — central-difference gradient with
  one-sided / least-squares fallbacks (see docs/interpolation.md §5).

These feed the EM and M1 SDE schemes (B4 / B5) and are written as
Numba kernels with one thin Python wrapper for tests.
"""
from __future__ import annotations

import math
from typing import NamedTuple, Optional, Tuple

import numpy as np

from ..core.geometry import DEG2RAD, METERS_PER_DEG_LAT
from ..core.mesh import _local_nearest_node, _find_containing_element_local

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


# Layout-id constants — Numba kernels take an int instead of the string
# ``"node" | "element"`` so the dispatch is branch-cheap and ``@njit``
# friendly.  Keep these in lock-step with VelocityField.kh_layout.
KH_LAYOUT_NODE = 0
KH_LAYOUT_ELEMENT = 1


def _layout_to_int(layout: str) -> int:
    """Map ``"node" | "element"`` → ``KH_LAYOUT_*`` int constant."""
    if layout == "node":
        return KH_LAYOUT_NODE
    if layout == "element":
        return KH_LAYOUT_ELEMENT
    raise ValueError(
        f"kh_layout must be 'node' or 'element', got {layout!r}"
    )


# Floor for the metric factor near the poles.  Mirrors the same guard
# used by ``vertical._interp_w_leastsq_single``.
_Z0_MIN = 1.0


class DiffusionStepArgs(NamedTuple):
    """Per-step diffusion configuration consumed by the integrator wrappers.

    ``scheme`` is one of ``"uniform" | "em" | "m1"``.  ``composition``
    is one of ``"split" | "combined" | "split_rk"`` (auto-resolved by
    the tracking layer; see the composition rules).  ``kh_x`` and
    ``kh_y`` are scalar diffusivities for the uniform scheme; for
    ``"em" / "m1"`` they are ignored — the integrator pulls the spatial
    Kh fields from ``VelocityField.kh_zonal / kh_meridional`` instead.

    ``dres_m`` is the central-difference offset in metres used by the
    EM/M1 path's gradient sampling; the tracking layer fills it from
    ``DiffusionConfig.dres_m`` (or a mesh-derived default).  The
    remaining fields seed the counter-RNG so the draw is fully
    reproducible across restarts and parallel runs.

    ``composition`` and ``dres_m`` have defaults so older constructions
    that omit them keep working (the integrator interprets them as
    "split"-mode uniform-Kh — the original behaviour).

    ``kh_eval_position`` selects whether the split-RK kick
    samples Kh at the start-of-step or end-of-step position; default
    ``"end"`` is the default.  Ignored for split and
    combined modes.
    """
    scheme: str
    kh_x: float
    kh_y: float
    seed: int
    real_id: int
    step_i: int
    composition: str = "split"
    dres_m: float = 0.0
    kh_eval_position: str = "end"
    # Vertical Az kick (TODO 2.2b).  ``az_enabled`` gates the split-mode
    # vertical kick; the Az source is ``VelocityField.az`` when the run
    # configures ``var_az``, else the scalar ``az_const``.  Appended
    # with defaults so older positional constructions keep working.
    az_enabled: bool = False
    az_const: float = 0.0
    # True when the run configured ``var_az``: the kick then REQUIRES
    # ``vel.az`` and raises if the payload lacks it, instead of silently
    # degrading to ``az_const`` (hardening — unreachable via run_tracking).
    az_requires_field: bool = False
    # Global index of this run's particle 0 in the full (unsharded) release:
    # random draws are keyed by ``pid_offset + i`` so a sharded run draws
    # exactly the numbers of the unsharded run (0 = byte-identical default).
    pid_offset: int = 0


# Scheme-id constants — Numba kernels switch on int rather than string
# so the dispatch is branch-cheap.  Keep in lock-step with
# ``DiffusionConfig.scheme``.
SCHEME_UNIFORM = 0
SCHEME_EM = 1
SCHEME_M1 = 2


def _scheme_to_int(scheme: str) -> int:
    """Map ``"uniform" | "em" | "m1"`` → ``SCHEME_*`` int constant."""
    if scheme == "uniform":
        return SCHEME_UNIFORM
    if scheme == "em":
        return SCHEME_EM
    if scheme == "m1":
        return SCHEME_M1
    raise ValueError(f"unknown diffusion scheme {scheme!r}")


# SplitMix64 round constants (Vigna).
_SM_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_SM_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_SM_MIX2 = np.uint64(0x94D049BB133111EB)
_SHIFT_30 = np.uint64(30)
_SHIFT_27 = np.uint64(27)
_SHIFT_31 = np.uint64(31)
_SHIFT_32 = np.uint64(32)
_MASK_32 = np.uint64(0xFFFFFFFF)

# Conversion constant for uint32 → (0, 1) float.  Adding 0.5 keeps the
# result strictly inside the open interval, which lets Box–Muller take
# log() without ever seeing zero.
_INV_2_32 = 1.0 / 4294967296.0  # 2**-32
_HALF_INV_2_32 = 0.5 * _INV_2_32

_TWO_PI = 2.0 * math.pi


# Component-id slots in the RNG key tuple.  One slot per *kind* of noise:
# Box-Muller already produces two independent samples per call, so one
# slot covers a horizontal (zonal, meridional) pair.  Vertical Kz, if it
# is ever added, would claim its own slot to stay independent.
COMP_HORIZONTAL = 0  # zonal+meridional pair (uniform / EM / M1)
COMP_VERTICAL = 1    # vertical Az kick (TODO 2.2b)


@njit(cache=True)
def _splitmix64(x: np.uint64) -> np.uint64:
    """One SplitMix64 round.

    Takes a uint64 ``x`` and returns a thoroughly mixed uint64.
    """
    x = x + _SM_GAMMA
    x = (x ^ (x >> _SHIFT_30)) * _SM_MIX1
    x = (x ^ (x >> _SHIFT_27)) * _SM_MIX2
    x = x ^ (x >> _SHIFT_31)
    return x


@njit(cache=True)
def _philox_uint64(k0: np.uint64, k1: np.uint64, k2: np.uint64,
                   k3: np.uint64, k4: np.uint64) -> np.uint64:
    """Mix five 64-bit keys into a single uint64 hash.

    Counter-based in the spirit of Salmon et al. (2011).  Despite the
    name, the mixer is a keyed SplitMix64 chain (Steele, Lea & Flood
    2014), not the Philox multiply-high Feistel network; the name is
    kept for stability of the public seeding contract.

    The expected key tuple is
    ``(seed, realization_id, particle_id, step_i, component_id)``.
    Order matters: swapping any two keys changes the output.

    All inputs must be non-negative integers (cast to uint64
    internally).
    """
    h = np.uint64(k0)
    h = _splitmix64(h ^ np.uint64(k1))
    h = _splitmix64(h ^ np.uint64(k2))
    h = _splitmix64(h ^ np.uint64(k3))
    h = _splitmix64(h ^ np.uint64(k4))
    return h


@njit(cache=True)
def _uint64_to_two_uniforms(h: np.uint64):
    """Split a uint64 into two open-(0, 1) doubles.

    Uses the high and low 32-bit halves of ``h``; the half-bit shift
    (``+ 0.5/2**32``) keeps both outputs strictly inside the open
    interval, so callers can take ``log`` without guarding for zero.
    """
    high = float((h >> _SHIFT_32) & _MASK_32)
    low = float(h & _MASK_32)
    u1 = high * _INV_2_32 + _HALF_INV_2_32
    u2 = low * _INV_2_32 + _HALF_INV_2_32
    return u1, u2


@njit(cache=True)
def _normal_pair(seed: np.uint64, real_id: np.uint64, particle_i: np.uint64,
                 step_i: np.uint64, comp_id: np.uint64):
    """Return a pair of independent N(0, 1) variates via Box–Muller.

    The pair is fully determined by the input key tuple.  When the
    integrator needs one normal per spatial axis it calls this once
    per particle per step and uses ``g1`` for the zonal kick and
    ``g2`` for the meridional kick — both axes share the ``comp_id``
    and remain pairwise independent.
    """
    h = _philox_uint64(seed, real_id, particle_i, step_i, comp_id)
    u1, u2 = _uint64_to_two_uniforms(h)
    r = math.sqrt(-2.0 * math.log(u1))
    g1 = r * math.cos(_TWO_PI * u2)
    g2 = r * math.sin(_TWO_PI * u2)
    return g1, g2


@njit(cache=True)
def _standard_normal(seed: np.uint64, real_id: np.uint64, particle_i: np.uint64,
                     step_i: np.uint64, comp_id: np.uint64) -> float:
    """Single N(0, 1) variate.  Discards the second Box–Muller half."""
    g1, _ = _normal_pair(seed, real_id, particle_i, step_i, comp_id)
    return g1


@njit(cache=True)
def diffusion_uniform_step_meters(kh_x: float, kh_y: float, dt: float,
                                  seed: np.uint64, real_id: np.uint64,
                                  particle_i: np.uint64, step_i: np.uint64):
    """Stochastic kick for the uniform-Kh scheme in pure metres.

    Implements the Brownian increment

    .. math::

        \\Delta x = \\sqrt{2 K_x}\\,\\sqrt{|\\Delta t|}\\,N_1,
        \\quad
        \\Delta y = \\sqrt{2 K_y}\\,\\sqrt{|\\Delta t|}\\,N_2,

    where ``(N_1, N_2)`` is an independent N(0, 1) pair drawn from the
    counter-RNG with key tuple
    ``(seed, real_id, particle_i, step_i, COMP_HORIZONTAL)``.

    The amplitude uses ``|dt|`` so the kick magnitude is the same in
    forward (``dt > 0``) and backward (``dt < 0``) integration; the
    deterministic drift handles the sign separately.

    Returns
    -------
    (dx, dy) : tuple of float
        Displacement in metres on the local tangent plane.  The caller
        is responsible for converting to lon/lat via the metric factor
        ``z0 = cos(lat)·METERS_PER_DEG_LAT`` (with the existing pole
        guard).
    """
    g1, g2 = _normal_pair(seed, real_id, particle_i, step_i,
                          np.uint64(COMP_HORIZONTAL))
    abs_dt = dt if dt >= 0.0 else -dt
    sqrt_abs_dt = math.sqrt(abs_dt)
    sigma_x = math.sqrt(2.0 * kh_x)
    sigma_y = math.sqrt(2.0 * kh_y)
    dx = sigma_x * sqrt_abs_dt * g1
    dy = sigma_y * sqrt_abs_dt * g2
    return dx, dy


@njit(cache=True)
def diffusion_az_step_meters(az: float, daz_dz: float, dt: float,
                             seed: np.uint64, real_id: np.uint64,
                             particle_i: np.uint64,
                             step_i: np.uint64) -> float:
    """Vertical stochastic kick in metres (positive-down depth).

    Implements the Euler–Maruyama increment with the mandatory Itô
    drift correction (Visser 1997)

    .. math::

        \\Delta z = \\frac{\\partial A_z}{\\partial z}\\,\\Delta t
                    + \\sqrt{2 A_z |\\Delta t|}\\,\\xi,

    in positive-down depth coordinates (the z→−depth flip cancels in
    the drift term, so the sign convention matches the z-up formula).
    Both the drift and the noise use ``|dt|``: the drift belongs to the
    diffusion, not to the advection, so a backward run keeps it (the
    adjoint of the advection–diffusion operator; a signed drift gave a
    backward stationary density ∝ Az⁻², review 2026-09-22).

    Draws the single normal from the counter-RNG with key tuple
    ``(seed, real_id, particle_i, step_i, COMP_VERTICAL)`` — an
    independent stream from the horizontal pair by component id.
    """
    xi = _standard_normal(seed, real_id, particle_i, step_i,
                          np.uint64(COMP_VERTICAL))
    abs_dt = dt if dt >= 0.0 else -dt
    return daz_dz * abs_dt + math.sqrt(2.0 * az * abs_dt) * xi


@njit(cache=True)
def reflect_depth(z: float, top: float, bot: float) -> float:
    """Fold a depth back into ``[top, bot]`` by specular reflection.

    Reflective — NOT absorbing and NOT a clamp: a kick that overshoots
    a boundary bounces back into the column by the overshoot distance,
    so the well-mixed equilibrium is preserved (a clamp would pile
    particles up on the boundary).  The modulo fold handles kicks of
    any size (multiple bounces).  A degenerate column (``bot <= top``)
    collapses to ``top``.
    """
    L = bot - top
    if L <= 0.0:
        return top
    y = (z - top) % (2.0 * L)
    if y < 0.0:          # belt-and-braces; Python % is divisor-signed
        y += 2.0 * L
    if y > L:
        y = 2.0 * L - y
    return top + y


@njit(cache=True)
def _az_interp_column(az_field: np.ndarray, z_levels: np.ndarray,
                      node: int, z_eval: float) -> float:
    """Linear interpolation of interface-based Az along one node column."""
    K = z_levels.shape[0]
    if K < 2:
        return az_field[0, node]
    if z_eval <= z_levels[0]:
        return az_field[0, node]
    if z_eval >= z_levels[K - 1]:
        return az_field[K - 1, node]
    for k in range(K - 1):
        if z_eval <= z_levels[k + 1]:
            dz_l = z_levels[k + 1] - z_levels[k]
            if dz_l <= 0.0:
                return az_field[k, node]
            frac = (z_eval - z_levels[k]) / dz_l
            return ((1.0 - frac) * az_field[k, node]
                    + frac * az_field[k + 1, node])
    return az_field[K - 1, node]


@njit(cache=True)
def az_column_value_gradient(az_field: np.ndarray, z_levels: np.ndarray,
                             node: int, z: float, node_bottom: float):
    """Az value and ∂Az/∂z at depth ``z`` on the nearest-node column.

    ``az_field`` is node-based ``(K, N)`` at interfaces (same layout as
    ``w``).  The evaluation depth is clamped to the wet part of the
    column ``[z_levels[0], min(node_bottom, z_levels[-1])]``
    (bottom-aware, like the W interpolation).  The value is linear
    between interfaces and the gradient is the slope of THAT segment —
    the derivative of the same interpolant whose value sets the noise.
    Exactly on an interface the segment above it is used (the layer
    search).  The well-mixed condition needs this identity: the old
    half-layer central difference straddled the neighbouring interface
    and left the uniform density non-stationary even as dt → 0 (review
    2026-09-22).  Degenerate columns (single interface, zero-thickness
    segment) return zero gradient.

    Returns
    -------
    (az, daz_dz) : tuple of float
        Diffusivity (m²/s) and its vertical gradient (m/s).
    """
    K = z_levels.shape[0]
    top = z_levels[0]
    bot = node_bottom if node_bottom < z_levels[K - 1] else z_levels[K - 1]
    if bot < top:
        bot = top
    z_eval = z
    if z_eval < top:
        z_eval = top
    if z_eval > bot:
        z_eval = bot
    val = _az_interp_column(az_field, z_levels, node, z_eval)
    if K < 2:
        return val, 0.0

    # Segment containing the evaluation depth → its slope.
    k = K - 2
    for kk in range(K - 1):
        if z_eval <= z_levels[kk + 1]:
            k = kk
            break
    dz = z_levels[k + 1] - z_levels[k]
    if dz <= 0.0:
        return val, 0.0
    return val, (az_field[k + 1, node] - az_field[k, node]) / dz


@njit(parallel=True, cache=True)
def _loop_az_kick_zlevel(z, alive, nn, has_field, az_field, az_const,
                         z_levels, node_depth, dt,
                         seed, real_id, step_i, z_out, pid_offset=0):
    """Vertical Az kick for all alive particles on a z-level mesh.

    Kick → reflective fold into ``[z_levels[0], depth[nn]]`` — the same
    bounds the deterministic ``clamp_depth`` enforces.  ``particle_id``
    is the stable original index ``i`` (counter-RNG contract).
    """
    P = z.shape[0]
    top = z_levels[0]
    for i in prange(P):
        if not alive[i]:
            z_out[i] = z[i]
            continue
        node = nn[i]
        if has_field:
            az, g = az_column_value_gradient(az_field, z_levels, node,
                                             z[i], node_depth[node])
        else:
            az = az_const
            g = 0.0
        if az < 0.0:      # corrupt field value: kill the WHOLE kick —
            az = 0.0      # a live gradient from garbage would still
            g = 0.0       # drift the particle (review 2026-08-15)
        dz = diffusion_az_step_meters(az, g, dt, seed, real_id,
                                      np.uint64(i) + np.uint64(pid_offset),
                                      step_i)
        z_out[i] = reflect_depth(z[i] + dz, top, node_depth[node])


@njit(parallel=True, cache=True)
def _loop_az_kick_sigma(z, alive, H_col, az_const,
                        Dmin, z0b_min, dt, seed, real_id, step_i, z_out,
                        pid_offset=0):
    """Vertical Az kick on a sigma mesh (constant Az only in v1).

    Reflective fold into ``[0, max(H_col, Dmin) − z0b_min]`` with ``H_col``
    the particle's column at the END of the step (the column the step just
    placed it in) — mirroring ``clamp_depth_sigma``'s bounds exactly.
    """
    P = z.shape[0]
    for i in prange(P):
        if not alive[i]:
            z_out[i] = z[i]
            continue
        H = H_col[i]
        if H < Dmin:
            H = Dmin
        bot = H - z0b_min
        if bot < 0.0:
            bot = 0.0
        dz = diffusion_az_step_meters(az_const, 0.0, dt, seed, real_id,
                                      np.uint64(i) + np.uint64(pid_offset),
                                      step_i)
        z_out[i] = reflect_depth(z[i] + dz, 0.0, bot)


@njit(parallel=True, cache=True)
def _fill_uniform_kicks(kh_x: float, kh_y: float, dt: float,
                        seed: np.uint64, real_id: np.uint64,
                        step_i: np.uint64, alive: np.ndarray, P: int,
                        pid_offset=0):
    """Vectorised counter-RNG kick for an array of particles.

    Computes ``(dx_m, dy_m)`` in metres for every alive particle using
    its **stable original index** ``i`` as ``particle_id``.  The
    returned arrays are sized ``P`` (full state length) with zeros for
    dead particles, so downstream code can mask without re-indexing.

    The integrator is responsible for ensuring ``alive`` and ``i``
    correspond to the same indexing convention as the rest of the
    ``ParticleState`` arrays — see "The trap" in ``docs/diffusion.md`` §4.
    """
    dx_m = np.zeros(P, dtype=np.float64)
    dy_m = np.zeros(P, dtype=np.float64)
    for i in prange(P):
        if alive[i]:
            dx, dy = diffusion_uniform_step_meters(
                kh_x, kh_y, dt, seed, real_id,
                np.uint64(i) + np.uint64(pid_offset), step_i
            )
            dx_m[i] = dx
            dy_m[i] = dy
    return dx_m, dy_m


# ===========================================================================
# Kicks on the sphere (review 2026-09-22, M04)
# ===========================================================================
#
# A lon/lat random walk (dlon = dx / (R cos lat), dlat = dy / R) lacks the
# metric drift -K tan(lat) / R of the Laplace-Beltrami operator: its
# invariant density is uniform in latitude, not in area.  The kicks are
# therefore applied along the great circle — the exponential map of the
# tangent displacement (dx east, dy north); for isotropic constant K it is
# rotation-invariant (the uniform area density is kept exactly;
# weak first order in time, not the exact heat kernel) and regular through
# the pole.  The exponential map itself
# carries the northward drift -Kx tan(lat) / R; a diagonal tensor with
# Kx != Ky needs the extra (Kx - Ky) tan(lat) / R to reach the
# Laplace-Beltrami value -Ky tan(lat) / R (anisotropic_metric_drift_m).
# R is the metric radius of the tracker (METERS_PER_DEG_LAT / DEG2RAD), so a
# pure meridional kick moves exactly dy / METERS_PER_DEG_LAT degrees.

R_METRIC = METERS_PER_DEG_LAT / DEG2RAD
#: latitude cap (degrees) of the tan() in the anisotropic correction —
#: east/north anisotropy is undefined at the pole itself
_ANISO_LAT_CAP = 89.9


@njit(cache=True)
def sphere_displace(lon: float, lat: float, dx_m: float, dy_m: float):
    """Move ``(lon, lat)`` (degrees) along the great circle whose initial
    tangent is ``(dx_m east, dy_m north)`` metres; returns ``(lon, lat)``
    with the longitude wrapped to ``[-180, 180)``.

    Cartesian form of the exponential map: position ``r``, local east/north
    unit vectors built from the STORED longitude, ``r' = r cos(d) + t sin(d)/d``.
    At a pole the east/north basis is still defined by that longitude, so a
    northward kick from 90°N continues down the opposite meridian (the
    bearing formula degenerated there — review 2026-09-23).
    """
    dist = math.sqrt(dx_m * dx_m + dy_m * dy_m)
    if dist == 0.0:
        return lon, lat
    lam = lon * DEG2RAD
    phi = lat * DEG2RAD
    cl = math.cos(lam)
    sl = math.sin(lam)
    cp = math.cos(phi)
    sp = math.sin(phi)
    delta = dist / R_METRIC
    c = math.cos(delta)
    s = math.sin(delta) / dist            # sin(delta)/delta / R_METRIC
    tx = -sl * dx_m - sp * cl * dy_m      # tangent in metres (east, north)
    ty = cl * dx_m - sp * sl * dy_m
    tz = cp * dy_m
    x2 = cp * cl * c + tx * s
    y2 = cp * sl * c + ty * s
    z2 = sp * c + tz * s
    if z2 > 1.0:
        z2 = 1.0
    elif z2 < -1.0:
        z2 = -1.0
    lat2 = math.asin(z2) / DEG2RAD
    lon2 = math.atan2(y2, x2) / DEG2RAD
    lon2 = ((lon2 + 180.0) % 360.0) - 180.0
    return lon2, lat2


@njit(parallel=True, cache=True)
def sphere_displace_arrays(lon, lat, dx_m, dy_m, alive):
    """Vectorised :func:`sphere_displace`; entries with ``alive`` False
    come back verbatim."""
    n = lon.shape[0]
    lon_out = lon.copy()
    lat_out = lat.copy()
    for i in prange(n):
        if alive[i]:
            a, b = sphere_displace(lon[i], lat[i], dx_m[i], dy_m[i])
            lon_out[i] = a
            lat_out[i] = b
    return lon_out, lat_out


@njit(cache=True)
def anisotropic_metric_drift_m(kx: float, ky: float, lat: float,
                               dt: float) -> float:
    """Extra northward displacement (m) a diagonal ``Kx != Ky`` tensor needs
    on top of the exponential map: ``(Kx - Ky) tan(lat) / R |dt|``."""
    if kx == ky:
        return 0.0
    la = lat
    if la > _ANISO_LAT_CAP:
        la = _ANISO_LAT_CAP
    elif la < -_ANISO_LAT_CAP:
        la = -_ANISO_LAT_CAP
    adt = dt if dt >= 0.0 else -dt
    return (kx - ky) * math.tan(la * DEG2RAD) / R_METRIC * adt


# ===========================================================================
# Kh interpolation + central-difference gradient
# ===========================================================================
#
# Two Numba kernels handle the two Kh layouts:
#
#   * ``_kh_lsq_node_planar``   — least-squares plane through 1-ring node
#                                 Kh values.  Returns (a0, a1, a2, ok).
#   * ``_kh_lsq_elem_planar``   — least-squares plane through 1-ring
#                                 element-centre Kh values around a node.
#                                 Returns (a0, a1, a2, ok).
#
# All return values are evaluated in the local centred-coordinate frame
# (``ex = (lon - xp) * z0``, ``ey = (lat - yp) * SLM``), so ``a0`` is
# already the interpolated Kh at the particle and ``a1`` / ``a2`` are
# already ∂K/∂x and ∂K/∂y in m^-1.
#
# Interpolation dispatch (both layouts use LSQ-planar, not barycentric):
#   * "element" layout fits the plane over the 1-ring element centres
#     (the same stencil as u/v); "node" layout fits it over the 1-ring
#     nodes.  In BOTH layouts the value is a0 of the fit, and the
#     gradient's PRIMARY path is central differences at +-dres_m
#     (``_kh_gradient_at_xy``), falling back to one-sided differences and
#     finally to the analytic LSQ plane gradient (a1, a2) when offsets
#     leave the domain — fact-checked against the kernel 2026-06-05.
#
# Shared invariants (mirrors ``_interp_leastsq_core`` in interpolate.py):
#   * dateline-safe longitude delta via ``((lon - xp + 180) % 360) - 180``;
#   * pole guard ``z0 = max(cos(lat) · METERS_PER_DEG_LAT, _Z0_MIN)``;
#   * Cramer's rule on the symmetric 3x3 normal matrix; reject a singular
#     matrix with the two-stage guard ``|det| < 1e-30`` or
#     ``|det| < 1e-12·|a11·a22·a33|`` (the scale-relative test is the one
#     that fires on rank-deficient stencils).


@njit(cache=True)
def _kh_lsq_node_planar(
    xp: float, yp: float, node_idx: int,
    kh_node: np.ndarray,
    node_to_elems: np.ndarray,
    node_to_elems_count: np.ndarray,
    elem_nodes: np.ndarray,
    nodes_lon: np.ndarray,
    nodes_lat: np.ndarray,
):
    """Plane fit ``Kh = a0 + a1·x + a2·y`` over 1-ring nodes around ``node_idx``.

    Returns ``(a0, a1, a2, ok)``.  ``ok=False`` if the stencil has fewer
    than 3 nodes or the normal matrix is singular.  ``a0`` is the
    interpolated value at the particle (centred coordinates put dx=dy=0
    there); ``a1`` / ``a2`` are ∂Kh/∂x and ∂Kh/∂y in 1/metre.
    """
    # Collect 1-ring neighbour nodes (centre + ring) without dedup
    # malloc — keep it inline to stay Numba-friendly.
    n_elems = int(node_to_elems_count[node_idx])
    if n_elems < 1:
        return 0.0, 0.0, 0.0, False

    max_v = elem_nodes.shape[1]
    # Modest fixed buffer — typical 1-ring is 6-8 nodes; cap at 64 for
    # pathological meshes.
    BUF = 64
    buf = np.empty(BUF, dtype=np.int32)
    buf[0] = node_idx
    count = 1
    for k in range(n_elems):
        e = int(node_to_elems[node_idx, k])
        if e < 0:
            continue
        for v in range(max_v):
            nd = int(elem_nodes[e, v])
            if nd < 0:
                continue
            seen = False
            for j in range(count):
                if buf[j] == nd:
                    seen = True
                    break
            if not seen and count < BUF:
                buf[count] = nd
                count += 1

    if count < 3:
        return 0.0, 0.0, 0.0, False

    z0 = math.cos(yp * DEG2RAD) * METERS_PER_DEG_LAT
    if z0 < _Z0_MIN:
        z0 = _Z0_MIN

    a11 = 0.0
    a12 = 0.0
    a13 = 0.0
    a22 = 0.0
    a23 = 0.0
    a33 = 0.0
    fk = 0.0
    f1k = 0.0
    f2k = 0.0
    for i in range(count):
        nd = int(buf[i])
        dlon = ((nodes_lon[nd] - xp + 180.0) % 360.0) - 180.0
        ex = dlon * z0
        ey = (nodes_lat[nd] - yp) * METERS_PER_DEG_LAT
        kv = float(kh_node[nd])

        a11 += 1.0
        a12 += ex
        a13 += ey
        a22 += ex * ex
        a23 += ex * ey
        a33 += ey * ey
        fk += kv
        f1k += kv * ex
        f2k += kv * ey

    a21 = a12
    a31 = a13
    a32 = a23

    det = (a11 * a22 * a33
           + a12 * a23 * a31
           + a13 * a21 * a32
           - a13 * a22 * a31
           - a12 * a21 * a33
           - a11 * a23 * a32)
    # scale-relative singularity guard — see _interp_leastsq_core
    if abs(det) < 1e-30 or abs(det) < 1e-12 * abs(a11 * a22 * a33):
        return 0.0, 0.0, 0.0, False

    inv = 1.0 / abs(det)

    b11 = a22 * a33 - a23 * a32
    b12 = a13 * a32 - a12 * a33
    b13 = a12 * a23 - a13 * a22
    b21 = a23 * a31 - a21 * a33
    b22 = a11 * a33 - a13 * a31
    b23 = a13 * a21 - a11 * a23
    b31 = a21 * a32 - a22 * a31
    b32 = a12 * a31 - a11 * a32
    b33 = a11 * a22 - a12 * a21

    a0 = (b11 * fk + b12 * f1k + b13 * f2k) * inv
    a1 = (b21 * fk + b22 * f1k + b23 * f2k) * inv
    a2 = (b31 * fk + b32 * f1k + b33 * f2k) * inv
    return a0, a1, a2, True


@njit(cache=True)
def _kh_lsq_elem_planar(
    xp: float, yp: float, node_idx: int,
    kh_elem: np.ndarray,
    node_to_elems: np.ndarray,
    node_to_elems_count: np.ndarray,
    elem_center_lon: np.ndarray,
    elem_center_lat: np.ndarray,
):
    """Plane fit over element-centre Kh values around ``node_idx``.

    Same maths as ``_kh_lsq_node_planar`` but the stencil is the set of
    elements whose centres ring the nearest node — exactly the stencil
    used by ``_interp_leastsq_core`` for u/v.
    """
    count = int(node_to_elems_count[node_idx])
    if count < 3:
        return 0.0, 0.0, 0.0, False

    z0 = math.cos(yp * DEG2RAD) * METERS_PER_DEG_LAT
    if z0 < _Z0_MIN:
        z0 = _Z0_MIN

    a11 = 0.0
    a12 = 0.0
    a13 = 0.0
    a22 = 0.0
    a23 = 0.0
    a33 = 0.0
    fk = 0.0
    f1k = 0.0
    f2k = 0.0
    n_used = 0
    for k in range(count):
        e = int(node_to_elems[node_idx, k])
        if e < 0:
            continue
        dlon = ((elem_center_lon[e] - xp + 180.0) % 360.0) - 180.0
        ex = dlon * z0
        ey = (elem_center_lat[e] - yp) * METERS_PER_DEG_LAT
        kv = float(kh_elem[e])

        a11 += 1.0
        a12 += ex
        a13 += ey
        a22 += ex * ex
        a23 += ex * ey
        a33 += ey * ey
        fk += kv
        f1k += kv * ex
        f2k += kv * ey
        n_used += 1

    if n_used < 3:
        return 0.0, 0.0, 0.0, False

    a21 = a12
    a31 = a13
    a32 = a23

    det = (a11 * a22 * a33
           + a12 * a23 * a31
           + a13 * a21 * a32
           - a13 * a22 * a31
           - a12 * a21 * a33
           - a11 * a23 * a32)
    # scale-relative singularity guard — see _interp_leastsq_core
    if abs(det) < 1e-30 or abs(det) < 1e-12 * abs(a11 * a22 * a33):
        return 0.0, 0.0, 0.0, False

    inv = 1.0 / abs(det)

    b11 = a22 * a33 - a23 * a32
    b12 = a13 * a32 - a12 * a33
    b13 = a12 * a23 - a13 * a22
    b21 = a23 * a31 - a21 * a33
    b22 = a11 * a33 - a13 * a31
    b23 = a13 * a21 - a11 * a23
    b31 = a21 * a32 - a22 * a31
    b32 = a12 * a31 - a11 * a32
    b33 = a11 * a22 - a12 * a21

    a0 = (b11 * fk + b12 * f1k + b13 * f2k) * inv
    a1 = (b21 * fk + b22 * f1k + b23 * f2k) * inv
    a2 = (b31 * fk + b32 * f1k + b33 * f2k) * inv
    return a0, a1, a2, True


@njit(cache=True)
def _kh_value_at_xy(
    xp: float, yp: float, node_idx: int,
    layout_id: int,
    kh_field: np.ndarray,
    node_to_elems: np.ndarray,
    node_to_elems_count: np.ndarray,
    elem_nodes: np.ndarray,
    elem_center_lon: np.ndarray,
    elem_center_lat: np.ndarray,
    nodes_lon: np.ndarray,
    nodes_lat: np.ndarray,
):
    """Interpolate Kh at ``(xp, yp)`` near ``node_idx``.  Returns (val, ok).

    Strict variant: ``ok=True`` only when the query point sits inside
    a mesh element near ``node_idx`` (domain check via
    ``_find_containing_element_local`` — 1-ring scan with a 2-ring
    fallback for irregular meshes) AND the LSQ planar fit is solvable.
    This prevents silent extrapolation across coastlines — an off-mesh
    offset point used for central-difference sampling would otherwise
    return a meaningless value with ``ok=True``, bypassing the gradient
    fallback chain.  Note the 2-ring fallback only ever accepts points
    inside *real* (wet) mesh elements, so land points are still
    rejected; for an accepted 2-ring point the LSQ stencil centred on
    ``node_idx`` extrapolates by at most ~one element — consistent with
    how u/v are sampled for such particles.

    The element-layout primary path is **least-squares on the 1-ring of
    surrounding element centres**, mirroring how u/v are sampled.  This
    keeps the stencil populated even for particles that sit slightly
    outside the strict containing element after a substep, which would
    otherwise give a hard ``ok=False`` from a barycentric-only path.
    The domain check above is what reins this in for offset points.
    """
    _, in_dom = _find_containing_element_local(
        xp, yp, node_idx,
        node_to_elems, node_to_elems_count, elem_nodes,
        nodes_lon, nodes_lat,
    )
    if not in_dom:
        return 0.0, False
    if layout_id == KH_LAYOUT_NODE:
        a0, _, _, ok = _kh_lsq_node_planar(
            xp, yp, node_idx, kh_field,
            node_to_elems, node_to_elems_count, elem_nodes,
            nodes_lon, nodes_lat,
        )
    else:
        a0, _, _, ok = _kh_lsq_elem_planar(
            xp, yp, node_idx, kh_field,
            node_to_elems, node_to_elems_count,
            elem_center_lon, elem_center_lat,
        )
    return a0, ok


@njit(cache=True)
def _kh_value_at_offset(
    xq: float, yq: float, node_idx_centre: int,
    layout_id: int,
    kh_field: np.ndarray,
    node_to_elems: np.ndarray,
    node_to_elems_count: np.ndarray,
    elem_nodes: np.ndarray,
    elem_center_lon: np.ndarray,
    elem_center_lat: np.ndarray,
    nodes_lon: np.ndarray,
    nodes_lat: np.ndarray,
):
    """Sample Kh at an offset point with nn-refinement + domain check.

    The centre's nearest node ``node_idx_centre`` is a starting hint;
    we refine via ``_local_nearest_node`` so the LSQ stencil is built
    around the actual closest mesh node to ``(xq, yq)``.  The domain
    check then runs against the refined node's 1-ring, so an off-mesh
    offset is reliably reported as ``ok=False``.
    """
    nn_q = _local_nearest_node(
        xq, yq, node_idx_centre,
        node_to_elems, node_to_elems_count, elem_nodes,
        nodes_lon, nodes_lat,
    )
    return _kh_value_at_xy(
        xq, yq, nn_q, layout_id, kh_field,
        node_to_elems, node_to_elems_count, elem_nodes,
        elem_center_lon, elem_center_lat,
        nodes_lon, nodes_lat,
    )


@njit(cache=True)
def _kh_gradient_at_xy(
    xp: float, yp: float, dres_m: float, node_idx: int,
    layout_id: int,
    kh_field: np.ndarray,
    node_to_elems: np.ndarray,
    node_to_elems_count: np.ndarray,
    elem_nodes: np.ndarray,
    elem_center_lon: np.ndarray,
    elem_center_lat: np.ndarray,
    nodes_lon: np.ndarray,
    nodes_lat: np.ndarray,
):
    """Central-difference gradient of Kh at ``(xp, yp)`` with fallbacks.

    The fallback chain (see ``docs/interpolation.md`` §5) per axis:

    1. both ``+`` and ``−`` offsets in-domain → ``(k_+ − k_−) / (2·dres)``;
    2. exactly one offset in-domain → one-sided diff against the centre;
    3. neither offset in-domain → use the LSQ planar gradient at the
       centre point on the 1-ring stencil (always available for an
       in-domain particle);
    4. centre interp itself fails → return ``(0, 0, False)`` so the
       integrator can revert + mark OUTSIDE.

    Offsets are computed in metric space — ``±dres_m`` in metres — and
    converted to ``Δlon = dres_m / z0(yp)``, ``Δlat = dres_m / SLM``.

    The local nearest node for each offset is approximated by the
    centre-particle ``node_idx``: as long as ``dres_m`` is on the order
    of one element edge or smaller (the default median-edge scaling
    enforces this), the offset usually still falls inside the centre's
    1-ring stencil, which is what the LSQ kernel needs.  Larger
    ``dres_m`` will degrade the accuracy of the offset value but never
    crash the kernel.
    """
    # --- centre value via LSQ; LSQ also gives the analytic gradient,
    #     so we keep it as a unified-fallback candidate -----------------
    if layout_id == KH_LAYOUT_NODE:
        kc, gxc, gyc, centre_ok = _kh_lsq_node_planar(
            xp, yp, node_idx, kh_field,
            node_to_elems, node_to_elems_count, elem_nodes,
            nodes_lon, nodes_lat,
        )
    else:
        kc, gxc, gyc, centre_ok = _kh_lsq_elem_planar(
            xp, yp, node_idx, kh_field,
            node_to_elems, node_to_elems_count,
            elem_center_lon, elem_center_lat,
        )
    if not centre_ok:
        return 0.0, 0.0, False

    z0 = math.cos(yp * DEG2RAD) * METERS_PER_DEG_LAT
    if z0 < _Z0_MIN:
        z0 = _Z0_MIN

    dlon = dres_m / z0
    dlat = dres_m / METERS_PER_DEG_LAT

    # --- ±x offsets ---------------------------------------------------
    kxp, ok_xp = _kh_value_at_offset(
        xp + dlon, yp, node_idx, layout_id, kh_field,
        node_to_elems, node_to_elems_count, elem_nodes,
        elem_center_lon, elem_center_lat, nodes_lon, nodes_lat,
    )
    kxm, ok_xm = _kh_value_at_offset(
        xp - dlon, yp, node_idx, layout_id, kh_field,
        node_to_elems, node_to_elems_count, elem_nodes,
        elem_center_lon, elem_center_lat, nodes_lon, nodes_lat,
    )
    if ok_xp and ok_xm:
        dKdx = (kxp - kxm) / (2.0 * dres_m)
    elif ok_xp:
        dKdx = (kxp - kc) / dres_m
    elif ok_xm:
        dKdx = (kc - kxm) / dres_m
    else:
        dKdx = gxc

    # --- ±y offsets ---------------------------------------------------
    kyp, ok_yp = _kh_value_at_offset(
        xp, yp + dlat, node_idx, layout_id, kh_field,
        node_to_elems, node_to_elems_count, elem_nodes,
        elem_center_lon, elem_center_lat, nodes_lon, nodes_lat,
    )
    kym, ok_ym = _kh_value_at_offset(
        xp, yp - dlat, node_idx, layout_id, kh_field,
        node_to_elems, node_to_elems_count, elem_nodes,
        elem_center_lon, elem_center_lat, nodes_lon, nodes_lat,
    )
    if ok_yp and ok_ym:
        dKdy = (kyp - kym) / (2.0 * dres_m)
    elif ok_yp:
        dKdy = (kyp - kc) / dres_m
    elif ok_ym:
        dKdy = (kc - kym) / dres_m
    else:
        dKdy = gyc

    return dKdx, dKdy, True


# ===========================================================================
# EM and M1 step kernels
# ===========================================================================
#
# Combined advection + variable-Kh stochastic step in pure metres.
# Both schemes use a single counter-RNG draw (one zonal + one meridional
# normal per particle per step) under ``COMP_HORIZONTAL`` so the EM/M1
# stream is identical to the uniform-scheme stream for the same key
# tuple.  This is what makes ``Kh=0`` regress bitwise to the
# deterministic Euler step and a constant-Kh field regress bitwise to
# the uniform-Kh kick.
#
# Sign convention (forward + backward integration):
#
#   dW    = N · √|dt|             (sign-symmetric Wiener increment)
#   advection uses signed dt      → reversibility of advection
#   ∂K drift uses |dt|            → part of the diffusion (adjoint: −u + ∇K)
#   noise amplitude uses √|dt|    → variance always grows with |t|
#
# This matches the uniform scheme's convention.  Backward stochastic
# tracking is allowed without a warning — see docs/diffusion.md §2,
# which sets out the two sign rules that make the kick correct in
# both directions.


@njit(cache=True)
def diffusion_em_step_meters(
    u_ms: float, v_ms: float,
    kh_x: float, kh_y: float,
    dKdx_x: float, dKdy_y: float,
    dt: float,
    seed: np.uint64, real_id: np.uint64,
    particle_i: np.uint64, step_i: np.uint64,
):
    """Combined Euler-Maruyama step in pure metres.

    .. math::

        \\Delta x = u \\Delta t + \\partial_x K_x |\\Delta t|
                  + \\sqrt{2 K_x} \\sqrt{|\\Delta t|}\\, N_1, \\\\
        \\Delta y = v \\Delta t + \\partial_y K_y |\\Delta t|
                  + \\sqrt{2 K_y} \\sqrt{|\\Delta t|}\\, N_2.

    The Itô correction term ``∂K`` ensures the ensemble mean drift in a
    spatially-varying Kh field reproduces Fick's-law diffusion (without
    it the cloud spuriously concentrates at low-Kh regions).  Only the
    advection reverses with the sign of ``dt``: ``∂K`` belongs to the
    diffusion (backward = adjoint operator, ``-u + ∇K``).

    Returns ``(dx, dy)`` in metres on the local tangent plane — caller
    converts to lon/lat via the existing ``z0`` factor.
    """
    g1, g2 = _normal_pair(seed, real_id, particle_i, step_i,
                          np.uint64(COMP_HORIZONTAL))
    abs_dt = dt if dt >= 0.0 else -dt
    sqrt_abs_dt = math.sqrt(abs_dt)
    sigma_x = math.sqrt(2.0 * kh_x) if kh_x > 0.0 else 0.0
    sigma_y = math.sqrt(2.0 * kh_y) if kh_y > 0.0 else 0.0
    dx = u_ms * dt + dKdx_x * abs_dt + sigma_x * sqrt_abs_dt * g1
    dy = v_ms * dt + dKdy_y * abs_dt + sigma_y * sqrt_abs_dt * g2
    return dx, dy


@njit(cache=True)
def diffusion_m1_step_meters(
    u_ms: float, v_ms: float,
    kh_x: float, kh_y: float,
    dKdx_x: float, dKdy_y: float,
    dt: float,
    seed: np.uint64, real_id: np.uint64,
    particle_i: np.uint64, step_i: np.uint64,
):
    """Combined Milstein step in pure metres.

    .. math::

        \\Delta x = u \\Delta t + \\partial_x K_x \\cdot |\\Delta t|
                  + \\tfrac{1}{2}\\,\\partial_x K_x \\cdot |\\Delta t|
                    \\cdot (N_1^2 - 1)
                  + \\sqrt{2 K_x} \\sqrt{|\\Delta t|}\\, N_1.

    For ``dt > 0`` this is the canonical Milstein form.  Only the
    advection ``u Δt`` changes sign with ``dt``; the ∂K drift, the
    quadratic-variation correction and the noise use ``|dt|`` (the
    backward process is the adjoint, review 2026-09-22).  Reduces to EM
    when ``∂K = 0``.

    Returns ``(dx, dy)`` in metres.
    """
    g1, g2 = _normal_pair(seed, real_id, particle_i, step_i,
                          np.uint64(COMP_HORIZONTAL))
    abs_dt = dt if dt >= 0.0 else -dt
    sqrt_abs_dt = math.sqrt(abs_dt)
    sigma_x = math.sqrt(2.0 * kh_x) if kh_x > 0.0 else 0.0
    sigma_y = math.sqrt(2.0 * kh_y) if kh_y > 0.0 else 0.0

    qv_x = 0.5 * dKdx_x * abs_dt * (g1 * g1 - 1.0)
    qv_y = 0.5 * dKdy_y * abs_dt * (g2 * g2 - 1.0)

    dx = u_ms * dt + dKdx_x * abs_dt + qv_x + sigma_x * sqrt_abs_dt * g1
    dy = v_ms * dt + dKdy_y * abs_dt + qv_y + sigma_y * sqrt_abs_dt * g2
    return dx, dy


# ===========================================================================
# Combined Euler + EM/M1 loop kernel
# ===========================================================================
#
# This is the integrator-side workhorse.  For every alive particle it:
#   1. interpolates u, v (raw, no time_dir folding) at the particle;
#   2. interpolates Kh_x and Kh_y at the particle;
#   3. computes the gradients ∂Kh_x/∂x, ∂Kh_y/∂y (the only components
#      the SDE drift correction needs in the diagonal-Kh case);
#   4. assembles an Euler-Maruyama or Milstein metre-space displacement:
#      advection with ``dt_signed = time_dir · dt_step``, the ∂K drift and
#      the noise with ``|dt_signed|``;
#   5. converts to lon/lat via the local metric.
#
# Domain re-validation is left to the Python wrapper — the kernel
# returns tentative new positions plus an ``ok`` flag per particle, and
# the wrapper runs ``find_nearest_nodes`` + ``_check_in_domain`` once on
# the survivors and reverts the failures (mirroring the
# ``_apply_split_diffusion_kick`` pattern of the uniform kick).
#
# Anisotropic note: the off-diagonal gradient terms (∂Kh_x/∂y,
# ∂Kh_y/∂x) are dropped because kelp's Kh is a *diagonal* tensor.
# A full anisotropic implementation would carry them; that lives in
# the anisotropic-Kh work, which is not implemented.

# Lazy import below: ``_interp_leastsq_core`` lives in interpolate.py
# but interpolate imports from diffusion, so we need a local copy of
# its imports inside the kernel by referencing through a function that
# defers the import until kernel build.  Numba allows imports inside
# the kernel body as long as the imported function is itself ``njit``'d.

from .interpolate import _interp_leastsq_core  # noqa: E402


@njit(parallel=True, cache=True)
def _loop_em_combined_2d(
    P: int,
    lons: np.ndarray, lats: np.ndarray,
    alive: np.ndarray, status_in: np.ndarray,
    nn_arr: np.ndarray,
    u_field: np.ndarray, v_field: np.ndarray,
    kh_x_field: np.ndarray, kh_y_field: np.ndarray,
    kh_layout_id: int,
    dres_m: float,
    scheme_id: int,
    dt_signed: float,
    seed: np.uint64, real_id: np.uint64, step_i: np.uint64,
    node_to_elems: np.ndarray, node_to_elems_count: np.ndarray,
    elem_nodes: np.ndarray,
    elem_center_lon: np.ndarray, elem_center_lat: np.ndarray,
    nodes_lon: np.ndarray, nodes_lat: np.ndarray,
    lon_out: np.ndarray, lat_out: np.ndarray,
    ok_out: np.ndarray,
    U_out: np.ndarray, V_out: np.ndarray,
    pid_offset=0,
):
    """Combined Euler + EM/M1 loop kernel (2D horizontal).

    ``dt_signed`` carries time direction: forward runs pass the positive
    integrator step, backward runs pass the negation.  The kernel uses
    ``dt_signed`` for the advection only and ``|dt_signed|`` for the ∂K
    drift, the noise and the quadratic-variation terms (the backward
    process is the adjoint: ``-u + ∇K``, review 2026-09-22).

    ``ok_out[i] = True`` if every interp succeeded for particle ``i``;
    the wrapper treats ``False`` slots as failed-this-step (revert +
    OUTSIDE).  ``U_out``/``V_out`` receive ``time_dir`` times the
    interpolated velocity of every alive particle (0 where the
    interpolation failed) — what the fused Euler kernel stores; dead slots
    are left untouched.
    """
    abs_dt = dt_signed if dt_signed >= 0.0 else -dt_signed
    sqrt_abs_dt = math.sqrt(abs_dt)
    # the step's u/v are the tracking velocity time_dir * u, like every
    # Euler/RK kernel (the tracking layer flips them back for output)
    tdir = 1.0 if dt_signed >= 0.0 else -1.0

    for i in prange(P):
        if not alive[i]:
            lon_out[i] = lons[i]
            lat_out[i] = lats[i]
            ok_out[i] = False  # nothing to do; integrator preserves status
            continue

        nn_i = int(nn_arr[i])
        xp = lons[i]
        yp = lats[i]

        # 1. Velocity (raw — the advection uses signed dt).
        UU, VV, ok_uv = _interp_leastsq_core(
            xp, yp, nn_i, 1.0,
            node_to_elems_count, node_to_elems,
            elem_center_lon, elem_center_lat,
            u_field, v_field,
        )
        U_out[i] = tdir * UU
        V_out[i] = tdir * VV
        if not ok_uv:
            lon_out[i] = xp
            lat_out[i] = yp
            ok_out[i] = False
            continue

        # 2. Kh values at the particle.
        kh_x, ok_kx = _kh_value_at_xy(
            xp, yp, nn_i, kh_layout_id, kh_x_field,
            node_to_elems, node_to_elems_count, elem_nodes,
            elem_center_lon, elem_center_lat, nodes_lon, nodes_lat,
        )
        kh_y, ok_ky = _kh_value_at_xy(
            xp, yp, nn_i, kh_layout_id, kh_y_field,
            node_to_elems, node_to_elems_count, elem_nodes,
            elem_center_lon, elem_center_lat, nodes_lon, nodes_lat,
        )
        if not (ok_kx and ok_ky):
            lon_out[i] = xp
            lat_out[i] = yp
            ok_out[i] = False
            continue

        # 3. Diagonal Kh gradients: central-diff with
        # one-sided / LSQ fallbacks — _kh_gradient_at_xy already does
        # the chain.  For the diagonal-tensor case we keep ∂Kh_x/∂x and
        # ∂Kh_y/∂y; cross terms belong to the unimplemented anisotropic case.
        gx_x, _, ok_gx = _kh_gradient_at_xy(
            xp, yp, dres_m, nn_i, kh_layout_id, kh_x_field,
            node_to_elems, node_to_elems_count, elem_nodes,
            elem_center_lon, elem_center_lat, nodes_lon, nodes_lat,
        )
        _, gy_y, ok_gy = _kh_gradient_at_xy(
            xp, yp, dres_m, nn_i, kh_layout_id, kh_y_field,
            node_to_elems, node_to_elems_count, elem_nodes,
            elem_center_lon, elem_center_lat, nodes_lon, nodes_lat,
        )
        if not (ok_gx and ok_gy):
            lon_out[i] = xp
            lat_out[i] = yp
            ok_out[i] = False
            continue

        # 4. Assemble SDE displacement in metres.
        g1, g2 = _normal_pair(
            seed, real_id, np.uint64(i) + np.uint64(pid_offset), step_i,
            np.uint64(COMP_HORIZONTAL),
        )
        sigma_x = math.sqrt(2.0 * kh_x) if kh_x > 0.0 else 0.0
        sigma_y = math.sqrt(2.0 * kh_y) if kh_y > 0.0 else 0.0

        if scheme_id == SCHEME_M1:
            qv_x = 0.5 * gx_x * abs_dt * (g1 * g1 - 1.0)
            qv_y = 0.5 * gy_y * abs_dt * (g2 * g2 - 1.0)
        else:
            qv_x = 0.0
            qv_y = 0.0

        # Diffusive part (drift + Milstein + noise), all evaluated at the
        # start position, moved along the great circle from there (the
        # chart walk lacks the metric drift, review 2026-09-22); the
        # advection keeps the chart step of the deterministic Euler update.
        dxd = (gx_x * abs_dt
               + qv_x
               + sigma_x * sqrt_abs_dt * g1)
        dyd = (gy_y * abs_dt
               + qv_y
               + sigma_y * sqrt_abs_dt * g2)
        dyd += anisotropic_metric_drift_m(kh_x, kh_y, yp, abs_dt)
        lon_d, lat_d = sphere_displace(xp, yp, dxd, dyd)

        # 5. Advection: metres → degrees with the pole-safe start metric.
        z0 = math.cos(yp * DEG2RAD) * METERS_PER_DEG_LAT
        if z0 < _Z0_MIN:
            z0 = _Z0_MIN
        lon_new = lon_d + UU * dt_signed / z0
        lat_new = lat_d + VV * dt_signed / METERS_PER_DEG_LAT

        # Inline wrap_lon / clamp_lat to keep the kernel self-contained.
        lon_new = ((lon_new + 180.0) % 360.0) - 180.0
        if lat_new > 90.0:
            lat_new = 90.0
        elif lat_new < -90.0:
            lat_new = -90.0

        lon_out[i] = lon_new
        lat_out[i] = lat_new
        ok_out[i] = True


# ===========================================================================
# Split-RK EM/M1 kick (deterministic RK already applied)
# ===========================================================================
#
# This is the kelp extension that lets ``scheme="em" / "m1"`` compose
# with RK2/RK4 via a *split* — the integrator runs the deterministic RK
# step first, then this kernel applies just the SDE kick (Itô drift
# correction + Brownian noise) on top.  The result is not a rigorous
# multidim SDE scheme; the warning policy and ``"split_rk"`` opt-in are
# documented in docs/diffusion.md and surfaced via
# ``_apply_split_rk_em_kick`` in integrators.py.
#
# Sampling position for Kh and ∂K is *configurable* via
# ``DiffusionStepArgs.kh_eval_position`` ∈ {"end", "start"}.  Default
# is "end": the post-RK position.  The caller is responsible for
# passing the correct ``(lons, lats, nn_arr)`` arrays — start-of-step
# or end-of-step — to the kernel.


@njit(parallel=True, cache=True)
def _loop_em_split_rk_2d(
    P: int,
    lons_eval: np.ndarray, lats_eval: np.ndarray,
    lons_apply: np.ndarray, lats_apply: np.ndarray,
    alive: np.ndarray,
    nn_arr: np.ndarray,
    kh_x_field: np.ndarray, kh_y_field: np.ndarray,
    kh_layout_id: int,
    dres_m: float,
    scheme_id: int,
    dt_signed: float,
    seed: np.uint64, real_id: np.uint64, step_i: np.uint64,
    node_to_elems: np.ndarray, node_to_elems_count: np.ndarray,
    elem_nodes: np.ndarray,
    elem_center_lon: np.ndarray, elem_center_lat: np.ndarray,
    nodes_lon: np.ndarray, nodes_lat: np.ndarray,
    lon_out: np.ndarray, lat_out: np.ndarray,
    ok_out: np.ndarray,
    pid_offset=0,
):
    """Split-RK kick: SDE displacement only, no advection.

    The deterministic step (RK2/RK4) is performed by the caller; this
    kernel adds the Itô drift correction and Brownian noise on top.

    Two position arrays:
      * ``(lons_eval, lats_eval)``: where Kh and ∂K are sampled
        (controlled by ``kh_eval_position`` — caller chooses "start"
        vs "end" of the deterministic step).
      * ``(lons_apply, lats_apply)``: where the kick is applied
        (always the post-RK position — that's the point of split mode).

    For ``kh_eval_position="end"`` (default) both arrays are the same
    (post-RK).  For ``"start"`` the eval arrays are pre-step.

    ``nn_arr`` must correspond to ``(lons_eval, lats_eval)`` since it
    seeds the LSQ stencil for Kh sampling.

    Same RNG keying as ``_loop_em_combined_2d`` — same particle index
    + step + COMP_HORIZONTAL slot — so a user switching between
    ``"combined"`` (Euler) and ``"split_rk"`` (RK) on the same key
    tuple draws the same Box-Muller pair.
    """
    abs_dt = dt_signed if dt_signed >= 0.0 else -dt_signed
    sqrt_abs_dt = math.sqrt(abs_dt)

    for i in prange(P):
        if not alive[i]:
            lon_out[i] = lons_apply[i]
            lat_out[i] = lats_apply[i]
            ok_out[i] = False
            continue

        nn_i = int(nn_arr[i])
        xp_eval = lons_eval[i]
        yp_eval = lats_eval[i]
        xp_apply = lons_apply[i]
        yp_apply = lats_apply[i]

        # 1. Kh values at the evaluation position.
        kh_x, ok_kx = _kh_value_at_xy(
            xp_eval, yp_eval, nn_i, kh_layout_id, kh_x_field,
            node_to_elems, node_to_elems_count, elem_nodes,
            elem_center_lon, elem_center_lat, nodes_lon, nodes_lat,
        )
        kh_y, ok_ky = _kh_value_at_xy(
            xp_eval, yp_eval, nn_i, kh_layout_id, kh_y_field,
            node_to_elems, node_to_elems_count, elem_nodes,
            elem_center_lon, elem_center_lat, nodes_lon, nodes_lat,
        )
        if not (ok_kx and ok_ky):
            lon_out[i] = xp_apply
            lat_out[i] = yp_apply
            ok_out[i] = False
            continue

        # 2. Diagonal Kh gradients at the evaluation position.
        gx_x, _, ok_gx = _kh_gradient_at_xy(
            xp_eval, yp_eval, dres_m, nn_i, kh_layout_id, kh_x_field,
            node_to_elems, node_to_elems_count, elem_nodes,
            elem_center_lon, elem_center_lat, nodes_lon, nodes_lat,
        )
        _, gy_y, ok_gy = _kh_gradient_at_xy(
            xp_eval, yp_eval, dres_m, nn_i, kh_layout_id, kh_y_field,
            node_to_elems, node_to_elems_count, elem_nodes,
            elem_center_lon, elem_center_lat, nodes_lon, nodes_lat,
        )
        if not (ok_gx and ok_gy):
            lon_out[i] = xp_apply
            lat_out[i] = yp_apply
            ok_out[i] = False
            continue

        # 3. SDE kick (advection already done by caller).
        g1, g2 = _normal_pair(
            seed, real_id, np.uint64(i) + np.uint64(pid_offset), step_i,
            np.uint64(COMP_HORIZONTAL),
        )
        sigma_x = math.sqrt(2.0 * kh_x) if kh_x > 0.0 else 0.0
        sigma_y = math.sqrt(2.0 * kh_y) if kh_y > 0.0 else 0.0

        if scheme_id == SCHEME_M1:
            qv_x = 0.5 * gx_x * abs_dt * (g1 * g1 - 1.0)
            qv_y = 0.5 * gy_y * abs_dt * (g2 * g2 - 1.0)
        else:
            qv_x = 0.0
            qv_y = 0.0

        dx_m = (gx_x * abs_dt
                + qv_x
                + sigma_x * sqrt_abs_dt * g1)
        dy_m = (gy_y * abs_dt
                + qv_y
                + sigma_y * sqrt_abs_dt * g2)

        # 4. Apply at the post-RK position along the great circle (the
        # chart walk lacks the metric drift, review 2026-09-22).
        dy_m += anisotropic_metric_drift_m(kh_x, kh_y, yp_apply, abs_dt)
        lon_new, lat_new = sphere_displace(xp_apply, yp_apply, dx_m, dy_m)

        lon_out[i] = lon_new
        lat_out[i] = lat_new
        ok_out[i] = True


# --- Python wrappers (used by tests; integrator B5 calls Numba kernels
# directly to avoid object dispatch in the hot loop). -------------------

def kh_value_at_point(
    xp: float, yp: float,
    kh_field: np.ndarray,
    kh_layout: str,
    mesh,
    node_idx: Optional[int] = None,
) -> Tuple[float, bool]:
    """Interpolate a Kh field at a particle position.

    Convenience wrapper.  When ``node_idx`` is ``None``, falls back to
    the mesh KD-tree (build via ``build_search`` first).
    """
    if node_idx is None:
        from ..core.mesh import find_nearest_nodes
        node_idx = int(find_nearest_nodes(
            mesh, np.array([xp]), np.array([yp])
        )[0])
    layout_id = _layout_to_int(kh_layout)
    val, ok = _kh_value_at_xy(
        float(xp), float(yp), int(node_idx), layout_id,
        np.asarray(kh_field, dtype=np.float64),
        mesh.node_to_elems, mesh.node_to_elems_count,
        mesh.elem_nodes,
        mesh.elem_center_lon, mesh.elem_center_lat,
        mesh.nodes_lon, mesh.nodes_lat,
    )
    return float(val), bool(ok)


def kh_gradient_at_point(
    xp: float, yp: float,
    dres_m: float,
    kh_field: np.ndarray,
    kh_layout: str,
    mesh,
    node_idx: Optional[int] = None,
) -> Tuple[float, float, bool]:
    """Central-difference Kh gradient at a particle position.

    See ``_kh_gradient_at_xy`` for the fallback chain.
    """
    if dres_m <= 0:
        raise ValueError(f"dres_m must be positive, got {dres_m}")
    if node_idx is None:
        from ..core.mesh import find_nearest_nodes
        node_idx = int(find_nearest_nodes(
            mesh, np.array([xp]), np.array([yp])
        )[0])
    layout_id = _layout_to_int(kh_layout)
    dKdx, dKdy, ok = _kh_gradient_at_xy(
        float(xp), float(yp), float(dres_m),
        int(node_idx), layout_id,
        np.asarray(kh_field, dtype=np.float64),
        mesh.node_to_elems, mesh.node_to_elems_count,
        mesh.elem_nodes,
        mesh.elem_center_lon, mesh.elem_center_lat,
        mesh.nodes_lon, mesh.nodes_lat,
    )
    return float(dKdx), float(dKdy), bool(ok)


__all__ = [
    "HAS_NUMBA",
    "COMP_HORIZONTAL",
    "COMP_VERTICAL",
    "DiffusionStepArgs",
    "KH_LAYOUT_NODE",
    "KH_LAYOUT_ELEMENT",
    "SCHEME_UNIFORM",
    "SCHEME_EM",
    "SCHEME_M1",
    "_scheme_to_int",
    "_layout_to_int",
    "_philox_uint64",
    "_normal_pair",
    "_standard_normal",
    "diffusion_uniform_step_meters",
    "diffusion_em_step_meters",
    "diffusion_m1_step_meters",
    "diffusion_az_step_meters",
    "reflect_depth",
    "az_column_value_gradient",
    "_fill_uniform_kicks",
    "_loop_em_combined_2d",
    "_loop_em_split_rk_2d",
    "_loop_az_kick_zlevel",
    "_loop_az_kick_sigma",
    "_kh_value_at_xy",
    "_kh_gradient_at_xy",
    "kh_value_at_point",
    "kh_gradient_at_point",
]
