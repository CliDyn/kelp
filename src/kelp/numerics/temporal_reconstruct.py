"""Monotone cubic (PCHIP) temporal reconstruction of velocity fields.

Replaces the 2-snapshot linear-in-time blend with a monotone piecewise cubic
(PCHIP) over a 4-snapshot stencil — Fritsch & Carlson (1980) monotonicity with
the weighted-harmonic interior slopes of Fritsch & Butland (1984), the form
SciPy's ``PchipInterpolator`` uses — so RK4 stage sampling is higher-order in
time (RK4 separates from RK2 on coarse file cadence) without new extrema per
COMPONENT.  Shape preservation is componentwise: it does not preserve linear
constraints between components or snapshots (a zero divergence), rotation
equivariance, or the time averages of averaged archives (review 2026-09-22).

Once-per-step O(N) whole-field blends (like ``core.io._blend``), not a
per-particle hot path.  Large C-contiguous fields go through a Numba
``prange`` kernel (TODO 6.7 — the serial numpy blend was the ~5 s/step,
N-independent floor of 3-D PCHIP runs and scaled only ×1.5 with threads);
everything else keeps the original numpy expressions, bit-identically.
Kh fields are NOT temporally reconstructed in v1 (the reconstructed field
carries velocity/eta/scalars only).
"""
from __future__ import annotations
import warnings
from typing import Sequence
import numpy as np
try:
    from numba import njit, prange
    HAS_NUMBA = True
except Exception:  # pragma: no cover - broad on purpose: a numpy/numba
    # binary (ABI) mismatch raises more than ImportError; the module must
    # stay importable on the numpy fallback (same shim as core/io.py).
    HAS_NUMBA = False
    prange = range
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from ..core.io import VelocityField


def _fc_slope(d_prev, d_next, h_prev, h_next):
    """Fritsch-Carlson interior slope (vectorized, monotone).

    ``d_prev``/``d_next`` are the secant slopes of the two intervals meeting at
    the node; ``h_prev``/``h_next`` their lengths.  Returns 0 where the secants
    change sign or either is zero (-> no overshoot), else the weighted harmonic
    mean.  All inputs are arrays of the same shape (or broadcastable scalars).
    """
    w1 = 2.0 * h_next + h_prev
    w2 = h_next + 2.0 * h_prev
    same = (d_prev * d_next) > 0.0
    # guard the divisions: where not same, use 1.0 so no zero-division; the
    # outer where discards that branch anyway.
    safe_prev = np.where(same, d_prev, 1.0)
    safe_next = np.where(same, d_next, 1.0)
    denom = w1 / safe_prev + w2 / safe_next
    return np.where(same, (w1 + w2) / denom, 0.0)


#: Arrays smaller than this take the plain numpy path in
#: :func:`_pchip_blend4` — below ~64K elements the numba thread
#: fork/join overhead exceeds the blend itself (same threshold as
#: ``core.io._BLEND_MIN_SIZE``, and the test-suite's tiny meshes stay
#: off the JIT path).
_PCHIP_MIN_SIZE = 1 << 16

#: One-time per-process marker for the "fast path missed only on
#: contiguity" warning (twin of ``core.io._CONTIG_WARNED``).
_CONTIG_WARNED: set = set()


@njit(parallel=True, cache=True)
def _pchip_blend4_flat(f0, f1, f2, f3, t0, t1, t2, t3, tq,
                       out):  # pragma: no cover - numba-compiled
    """Parallel per-element monotone-cubic blend over flat float64 arrays.

    Each element goes through :func:`_pchip_value_scalar` — bit-for-bit
    the arithmetic of the vectorized numpy expression in
    :func:`_pchip_blend4` (no fastmath, no FMA contraction), so the two
    paths differ only in memory-bandwidth parallelism (TODO 6.7: this
    whole-field blend was the ~5 s/step serial floor of 3-D PCHIP).
    """
    for i in prange(out.shape[0]):
        out[i] = _pchip_value_scalar(f0[i], f1[i], f2[i], f3[i],
                                     t0, t1, t2, t3, tq)


def _pchip_blend4(f0, f1, f2, f3, t0, t1, t2, t3, tq):
    """Monotone cubic value at ``tq`` in the central interval ``[t1, t2]``.

    ``f0..f3`` are arrays (any shape) sampled at scalar times ``t0<t1<t2<t3``;
    ``t1 <= tq <= t2``.  Returns an array of f's shape (float64).

    Large same-shape C-contiguous fields run through the parallel
    :func:`_pchip_blend4_flat` kernel; anything else takes the original
    numpy expression.  Both paths are bit-identical for finite values.
    """
    # Compute in float64 even for float32 inputs (float32 velocity fields,
    # TODO 5.3): under NEP 50 float32 arithmetic stays float32 and would round,
    # diverging from the float64 path.  no-op for float64 inputs.
    f0 = np.asarray(f0, dtype=np.float64)
    f1 = np.asarray(f1, dtype=np.float64)
    f2 = np.asarray(f2, dtype=np.float64)
    f3 = np.asarray(f3, dtype=np.float64)
    # Normalize the time scalars BEFORE the path split so both branches
    # see identical float64 values (a np.float32/np.longdouble time scalar
    # must not make the result depend on array size/contiguity).
    t0, t1, t2, t3, tq = (float(t0), float(t1), float(t2), float(t3),
                          float(tq))
    if (
        HAS_NUMBA
        and f0.shape == f1.shape == f2.shape == f3.shape
        and f1.size >= _PCHIP_MIN_SIZE
    ):
        if (f0.flags.c_contiguous and f1.flags.c_contiguous
                and f2.flags.c_contiguous and f3.flags.c_contiguous):
            out = np.empty(f1.shape, dtype=np.float64)
            _pchip_blend4_flat(
                f0.reshape(-1), f1.reshape(-1), f2.reshape(-1),
                f3.reshape(-1),
                t0, t1, t2, t3, tq,
                out.reshape(-1),
            )
            return out
        # Qualifies for the parallel kernel in every way EXCEPT memory
        # order — a reader is handing back Fortran-ordered / strided
        # fields; the silent serial fallback cost ~10x per step on a real
        # healpix 3-D run before anyone noticed (2026-08).  Say so, once.
        if "pchip" not in _CONTIG_WARNED:
            _CONTIG_WARNED.add("pchip")
            warnings.warn(
                "4-snapshot PCHIP blend fell back to the serial numpy "
                f"path: a large float64 field (shape {f1.shape}) is not "
                "C-contiguous.  A reader is producing Fortran-ordered or "
                "strided fields — wrap them in np.ascontiguousarray at "
                "the read site (see docs/performance.md).",
                RuntimeWarning, stacklevel=2,
            )
    h0 = t1 - t0
    h1 = t2 - t1
    h2 = t3 - t2
    d0 = (f1 - f0) / h0
    d1 = (f2 - f1) / h1
    d2 = (f3 - f2) / h2
    m1 = _fc_slope(d0, d1, h0, h1)   # slope at t1
    m2 = _fc_slope(d1, d2, h1, h2)   # slope at t2
    s = (tq - t1) / h1
    s2 = s * s
    s3 = s2 * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    return h00 * f1 + h10 * h1 * m1 + h01 * f2 + h11 * h1 * m2


# ---------------------------------------------------------------------------
# Scalar (Numba-compatible) twins of the vectorized helpers above.
#
# The fused PCHIP kernels (numerics/integrators.py) blend the 4-snapshot
# stencil INLINE, one element at a time, inside an @njit hot loop — they
# cannot call the NumPy/np.where versions.  These scalar forms are bit-for-bit
# the same arithmetic as ``_fc_slope`` / ``_pchip_blend4`` for finite scalar
# inputs (pinned by test_temporal_reconstruction.py), so the fused kernel
# reproduces the unfused ``reconstruct_pchip`` path exactly.
# ---------------------------------------------------------------------------

@njit(cache=True)
def _fc_slope_scalar(d_prev: float, d_next: float,
                     h_prev: float, h_next: float) -> float:
    """Scalar Fritsch-Carlson interior slope (twin of :func:`_fc_slope`).

    Returns 0 where the two secant slopes change sign or either is zero (no
    overshoot), else the weighted harmonic mean — identical to the vectorized
    ``_fc_slope`` evaluated at a single point.
    """
    if d_prev * d_next > 0.0:
        w1 = 2.0 * h_next + h_prev
        w2 = h_next + 2.0 * h_prev
        return (w1 + w2) / (w1 / d_prev + w2 / d_next)
    return 0.0


@njit(cache=True)
def _pchip_value_scalar(f0: float, f1: float, f2: float, f3: float,
                        t0: float, t1: float, t2: float, t3: float,
                        tq: float) -> float:
    """Scalar monotone-cubic value at ``tq`` (twin of :func:`_pchip_blend4`).

    ``f0..f3`` are the four snapshot values at scalar times ``t0<t1<t2<t3``;
    the query ``tq`` lies in the central interval ``[t1, t2]``.  Bit-for-bit
    the same Hermite arithmetic as ``_pchip_blend4`` on scalar inputs.
    """
    # Force float64: float32 snapshot values (TODO 5.3) would otherwise round
    # the (f1-f0) subtraction in float32 under numba, diverging from the
    # float64 path; np.float64(...) is a no-op for float64 inputs.
    f0 = np.float64(f0)
    f1 = np.float64(f1)
    f2 = np.float64(f2)
    f3 = np.float64(f3)
    h0 = t1 - t0
    h1 = t2 - t1
    h2 = t3 - t2
    d0 = (f1 - f0) / h0
    d1 = (f2 - f1) / h1
    d2 = (f3 - f2) / h2
    m1 = _fc_slope_scalar(d0, d1, h0, h1)   # slope at t1
    m2 = _fc_slope_scalar(d1, d2, h1, h2)   # slope at t2
    s = (tq - t1) / h1
    s2 = s * s
    s3 = s2 * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    return h00 * f1 + h10 * h1 * m1 + h01 * f2 + h11 * h1 * m2


def reconstruct_pchip(snaps: Sequence[VelocityField],
                      times: Sequence[float],
                      t_query: float) -> VelocityField:
    """Reconstruct a VelocityField at ``t_query`` from 4 snapshots.

    ``snaps`` are 4 VelocityFields at strictly increasing ``times`` with
    ``times[1] <= t_query <= times[2]`` (the central interval).  Each field
    (``u``, ``v``, ``w`` if present, ``eta``, every ``scalars[k]``) is blended
    per-element via :func:`_pchip_blend4`.  Kh is not reconstructed.

    The four snapshots must share one ``uv_vertical`` placement (checked
    before the field shapes); the result carries it, and carries the
    snapshots' ``_dl_token`` only when all four hold the SAME object.
    """
    if len(snaps) != 4 or len(times) != 4:
        raise ValueError("reconstruct_pchip needs exactly 4 snapshots/times")
    t0, t1, t2, t3 = (float(t) for t in times)
    # Validate the stencil up front — callers (tracking) guarantee these, but
    # this is a public function: bad/duplicate times would otherwise yield
    # divide-by-zero / NaN / extrapolated non-monotone values.
    if not all(np.isfinite(t) for t in (t0, t1, t2, t3)):
        raise ValueError(f"reconstruct_pchip: non-finite times {times}")
    if not (t0 < t1 < t2 < t3):
        raise ValueError(
            f"reconstruct_pchip: times must be strictly increasing, got {times}")
    tol = 1e-9 * (t2 - t1)
    if not (t1 - tol <= t_query <= t2 + tol):
        raise ValueError(
            f"reconstruct_pchip: t_query={t_query} outside the central "
            f"interval [{t1}, {t2}]")
    a, b, c, d = snaps
    placements = [s.uv_vertical for s in snaps]
    if len(set(placements)) > 1:
        raise ValueError(
            "reconstruct_pchip: uv_vertical mismatch across reconstruction "
            f"snapshots: {placements}")
    tok = a._dl_token
    if any(s._dl_token is not tok for s in snaps):
        tok = None

    def blend(attr):
        xs = [getattr(s, attr) for s in (a, b, c, d)]
        present = [x is not None for x in xs]
        if not any(present):
            return None
        if not all(present):
            raise ValueError(
                f"field {attr!r} present in only some reconstruction snapshots")
        shapes = {x.shape for x in xs}
        if len(shapes) > 1:
            raise ValueError(
                f"field {attr!r} shape mismatch across reconstruction "
                f"snapshots: {shapes}")
        return _pchip_blend4(xs[0], xs[1], xs[2], xs[3], t0, t1, t2, t3, t_query)

    u = blend("u")
    v = blend("v")
    w = blend("w")
    eta = blend("eta")

    # Scalars: all four snapshots must agree (all absent, or all carrying the
    # same key set) — checked symmetrically, not just off the second snapshot.
    def blend_scalar(k):
        xs = [s.scalars[k] for s in (a, b, c, d)]
        shapes = {x.shape for x in xs}
        if len(shapes) > 1:
            raise ValueError(
                f"scalar {k!r} shape mismatch across reconstruction "
                f"snapshots: {shapes}")
        return _pchip_blend4(xs[0], xs[1], xs[2], xs[3], t0, t1, t2, t3, t_query)

    skeys = [set(s.scalars) if s.scalars else None for s in snaps]
    scalars = None
    if any(k is not None for k in skeys):
        ref = next(k for k in skeys if k is not None)
        if any(k != ref for k in skeys):
            raise ValueError(
                "scalar keys differ (or present in only some) across "
                "reconstruction snapshots")
        scalars = {k: blend_scalar(k) for k in ref}

    out = VelocityField(u=u, v=v, eta=eta, w=w,
                        uv_layout=b.uv_layout, scalars=scalars,
                        uv_vertical=b.uv_vertical)
    out._dl_token = tok
    return out


def make_stage_vels_recon(snaps, times, sim_time, dt, time_dir, c_coeffs):
    """Per-RK-stage VelocityFields via PCHIP at ``t = sim_time + time_dir*c*dt``.

    ``time_dir`` is +1 forward / -1 backward.  The caller (tracking) guarantees
    every stage time stays within ``[times[1], times[2]]`` (i.e. ``dt`` does not
    exceed the central bracket), so reconstruction stays in the central
    interval.  Mirrors the contract of ``integrators._make_stage_vels`` (one
    VelocityField per stage), but cubic-in-time instead of linear.

    Duplicate stage times are reconstructed ONCE and the resulting field
    object shared (RK4's ``c = (0, 0.5, 0.5, 1)`` midpoint pair — the
    reconstruction is a whole-field, N-independent cost, TODO 6.7).
    Identical ``c`` values produce bit-identical ``tq`` (same float
    arithmetic), so the cache key is exact; consumers treat stage fields
    as read-only.
    """
    cache = {}
    out = []
    for c in c_coeffs:
        tq = sim_time + time_dir * c * dt
        if tq not in cache:
            cache[tq] = reconstruct_pchip(snaps, times, tq)
        out.append(cache[tq])
    return out
