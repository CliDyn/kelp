"""Velocity interpolation at particle positions.

Two methods:
    1. **Least-squares planar fit** — translated from Fortran ``drift-fesom-3d.f90``
       lines 317-376.  Fits ``U = a0 + a1·x + a2·y`` to element-centre values
       around the nearest node, solves the 3×3 normal system via Cramer's rule.
    2. **Nearest-element** — simple fallback using the containing element value.

Re-exports ``VelocityField`` from ``io.py`` for convenience (readers import
VelocityField from here).
"""
from __future__ import annotations

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

from ..core.geometry import METERS_PER_DEG_LAT, DEG2RAD
from ..core.io import VelocityField      # noqa: F401  (re-export; readers import it from core.io)
from ..core.mesh import (
    Mesh,
    _find_containing_element_local,
    _polygon_weights,
    build_elem_areas,
    build_elem_neighbors,
)
from ..core.particles import ParticleState
from .temporal_reconstruct import _pchip_value_scalar  # njit scalar PCHIP
from .vertical_uv import (
    check_depth_linear_record,
    depth_points,
    is_checked,
    uv_at,
    uv_bracket,
    uv_wet,
)


# ---------------------------------------------------------------------------
# Per-particle least-squares interpolation (Numba optimized core)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _interp_leastsq_core(
    xp: float,
    yp: float,
    node_idx: int,
    time_dir: float,
    node_to_elems_count: np.ndarray,
    node_to_elems: np.ndarray,
    elem_center_lon: np.ndarray,
    elem_center_lat: np.ndarray,
    u_field: np.ndarray,
    v_field: np.ndarray,
) -> tuple[float, float, bool]:
    """Least-squares linear velocity fit at a single particle position.

    Fits a plane ``f = a0 + a1*x + a2*y`` through element-centre velocities
    around the nearest mesh node, using Cramer's rule on the 3x3 normal matrix.

    Uses coordinates centered around the particle position to avoid catastrophic
    cancellation in the determinant (see docs/interpolation.md §1).

    Parameters
    ----------
    xp, yp : float
        Particle longitude and latitude in degrees.
    node_idx : int
        Index of the nearest mesh node.
    time_dir : float
        Forward (+1) or backward (-1) tracking flag.

    Returns
    -------
    UU, VV : float
        Interpolated velocity components (m/s) at the particle position.
    ok : bool
        False if the normal matrix is singular (particle frozen).
    """
    count = int(node_to_elems_count[node_idx])
    if count < 1:
        return 0.0, 0.0, False

    elem_ids = node_to_elems[node_idx, :count]

    # -- metric scaling: degrees -> approximate metres --
    z0 = np.cos(yp * DEG2RAD) * METERS_PER_DEG_LAT

    # -- element-centre coordinates in metres, CENTERED around particle --
    # Centering eliminates catastrophic cancellation in the determinant.
    xp_m = xp * z0
    yp_m = yp * METERS_PER_DEG_LAT

    # -- build matrix and RHS in one pass (sequential, like Fortran) --
    a11 = 0.0
    a12 = 0.0
    a13 = 0.0
    a22 = 0.0
    a23 = 0.0
    a33 = 0.0
    fUn = 0.0
    f1Un = 0.0
    f2Un = 0.0
    fVn = 0.0
    f1Vn = 0.0
    f2Vn = 0.0

    for n in range(count):
        e_id = elem_ids[n]
        ex = elem_center_lon[e_id] * z0 - xp_m
        # Antimeridian gate (2026-07): a raw centre-minus-particle
        # longitude difference beyond 180 deg means this stencil element
        # sits across the seam — the raw delta is ~+-360 deg off and the
        # fit sees the element half a world away.  Shift by one full turn.
        # Off-seam stencils never take the branch, so their arithmetic is
        # BIT-IDENTICAL to the historical (Fortran-parity) expression;
        # only seam-straddling stencils change (they were wrong).  The
        # exact +-180.0 tie keeps the raw delta (strict inequalities) —
        # unlike the canonical ((d+180)%360)-180 wrap this keeps +180 as
        # +180; reachable only with ~180-deg stencil spans (degenerate
        # meshes), where both interpretations are equally (in)valid.
        dlon_raw = elem_center_lon[e_id] - xp
        if dlon_raw > 180.0:
            ex -= 360.0 * z0
        elif dlon_raw < -180.0:
            ex += 360.0 * z0
        ey = elem_center_lat[e_id] * METERS_PER_DEG_LAT - yp_m
        uu = time_dir * u_field[e_id]
        vv = time_dir * v_field[e_id]

        a11 = a11 + 1.0
        a12 = a12 + ex
        a13 = a13 + ey
        a22 = a22 + ex * ex
        a23 = a23 + ex * ey
        a33 = a33 + ey * ey
        fUn = fUn + uu
        f1Un = f1Un + uu * ex
        f2Un = f2Un + uu * ey
        fVn = fVn + vv
        f1Vn = f1Vn + vv * ex
        f2Vn = f2Vn + vv * ey

    a21 = a12
    a31 = a13
    a32 = a23

    # -- determinant via Sarrus' rule --
    det = (a11 * a22 * a33
           + a12 * a23 * a31
           + a13 * a21 * a32
           - a13 * a22 * a31
           - a12 * a21 * a33
           - a11 * a23 * a32)

    # Singularity guard is SCALE-RELATIVE (2026-08 review): the Gram
    # entries are metric (metres^2), so a rank-deficient stencil's pure
    # ROUNDOFF det can be ~1e6..1e24 and an absolute 1e-30 test never
    # fires — Cramer then divides noise by noise (measured: 8 m/s out of
    # a 0.5 m/s constant field at corners with 2 incident elements).
    # Well-posed stencils keep det/(a11*a22*a33) >~ 1e-3 (sliver aspect
    # 1e-4 measured ~0.7); rank-deficient ones sit at machine epsilon —
    # 1e-12 splits the classes with >3 orders of margin on each side.
    # The absolute test stays for the exact-zero case (degenerate axis
    # makes the diagonal product itself zero).
    if abs(det) < 1e-30 or abs(det) < 1e-12 * abs(a11 * a22 * a33):
        return 0.0, 0.0, False

    inv = 1.0 / abs(det)

    # -- cofactor matrix (adjugate transpose) --
    b11 = a22 * a33 - a23 * a32
    b12 = a13 * a32 - a12 * a33
    b13 = a12 * a23 - a13 * a22
    b21 = a23 * a31 - a21 * a33
    b22 = a11 * a33 - a13 * a31
    b23 = a13 * a21 - a11 * a23
    b31 = a21 * a32 - a22 * a31
    b32 = a12 * a31 - a11 * a32
    b33 = a11 * a22 - a12 * a21

    # -- solve for plane coefficients --
    a0U = (b11 * fUn + b12 * f1Un + b13 * f2Un) * inv
    a1U = (b21 * fUn + b22 * f1Un + b23 * f2Un) * inv
    a2U = (b31 * fUn + b32 * f1Un + b33 * f2Un) * inv

    a0V = (b11 * fVn + b12 * f1Vn + b13 * f2Vn) * inv
    a1V = (b21 * fVn + b22 * f1Vn + b23 * f2Vn) * inv
    a2V = (b31 * fVn + b32 * f1Vn + b33 * f2Vn) * inv

    # -- evaluate at particle position (dx=0, dy=0 in centered coords) --
    UU = a0U
    VV = a0V

    return UU, VV, True


@njit(cache=True)
def _interp_leastsq_core_tinterp(
    xp: float,
    yp: float,
    node_idx: int,
    time_dir: float,
    alpha0: float,
    alpha: float,
    node_to_elems_count: np.ndarray,
    node_to_elems: np.ndarray,
    elem_center_lon: np.ndarray,
    elem_center_lat: np.ndarray,
    u_a: np.ndarray,
    v_a: np.ndarray,
    u_b: np.ndarray,
    v_b: np.ndarray,
    u_t: np.ndarray,
    v_t: np.ndarray,
) -> tuple[float, float, bool]:
    """Same as _interp_leastsq_core but with time-interpolation of velocity.

    Velocity at each element is blended inline with a NESTED two-level
    blend so the raw-pair fast path is bit-identical to the historical
    materialized-field path:

        ua0  = (1 - alpha0) * u_a[e] + alpha0 * u_b[e]   # "field at sim_time"
        u[e] = (1 - alpha)  * ua0    + alpha  * u_t[e]   # stage blend

    ``(u_a, u_b)`` is the BRACKET pair (``alpha0`` = bracket fraction:
    the first line reproduces the pre-refactor ``_blend`` materialization
    bitwise) while ``(u_t, v_t)`` is the snapshot the step moves TOWARD —
    ``u_b`` forward, but ``u_a`` for BACKWARD integration, where the
    motion target is the bracket's lower snapshot.  Callers holding an
    already-materialized field pass ``alpha0 = 0.0`` (collapsing the
    first line to ``u_a[e]``) with ``u_t`` = the old ``vel_next`` array.

    Used by time-aware fused Numba kernels.  When alpha0=alpha=0 the
    result is the same as _interp_leastsq_core(u_a, v_a); when alpha=1 it
    is _interp_leastsq_core(u_t, v_t).
    """
    count = int(node_to_elems_count[node_idx])
    if count < 1:
        return 0.0, 0.0, False

    elem_ids = node_to_elems[node_idx, :count]
    beta0 = 1.0 - alpha0
    beta = 1.0 - alpha

    z0 = np.cos(yp * DEG2RAD) * METERS_PER_DEG_LAT
    xp_m = xp * z0
    yp_m = yp * METERS_PER_DEG_LAT

    a11 = 0.0
    a12 = 0.0
    a13 = 0.0
    a22 = 0.0
    a23 = 0.0
    a33 = 0.0
    fUn = 0.0
    f1Un = 0.0
    f2Un = 0.0
    fVn = 0.0
    f1Vn = 0.0
    f2Vn = 0.0

    for n in range(count):
        e_id = elem_ids[n]
        ex = elem_center_lon[e_id] * z0 - xp_m
        # Same antimeridian gate as _interp_leastsq_core (off-seam
        # bit-identical, seam-straddling stencils corrected).
        dlon_raw = elem_center_lon[e_id] - xp
        if dlon_raw > 180.0:
            ex -= 360.0 * z0
        elif dlon_raw < -180.0:
            ex += 360.0 * z0
        ey = elem_center_lat[e_id] * METERS_PER_DEG_LAT - yp_m
        # Nested inline blend — ua0/va0 reproduce the materialized
        # start-of-step value bitwise; the stage blend then moves toward
        # the TARGET snapshot (u_t/v_t), unchanged from the old kernel.
        # alpha0 == 0.0 (plain pre-blended/raw field) takes the exact old
        # single-blend expression: the nested form would compute
        # 0.0 * u_b[e], which differs for non-finite u_b (0*Inf = NaN
        # where the old code propagated Inf).  Stage-endpoint alphas
        # short-circuit with _make_stage_vels' 1e-12 thresholds (K1
        # always has alpha == 0 — multiplying the unused target by 0
        # would likewise NaN a non-finite entry; finite fields are
        # bit-unchanged, x + 0.0 == x) — mirrors the barycentric twin.
        if alpha < 1e-12:
            if alpha0 == 0.0:
                u_val = u_a[e_id]
                v_val = v_a[e_id]
            else:
                u_val = beta0 * u_a[e_id] + alpha0 * u_b[e_id]
                v_val = beta0 * v_a[e_id] + alpha0 * v_b[e_id]
        elif alpha > 1.0 - 1e-12:
            u_val = u_t[e_id]
            v_val = v_t[e_id]
        elif alpha0 == 0.0:
            u_val = beta * u_a[e_id] + alpha * u_t[e_id]
            v_val = beta * v_a[e_id] + alpha * v_t[e_id]
        else:
            ua0 = beta0 * u_a[e_id] + alpha0 * u_b[e_id]
            va0 = beta0 * v_a[e_id] + alpha0 * v_b[e_id]
            u_val = beta * ua0 + alpha * u_t[e_id]
            v_val = beta * va0 + alpha * v_t[e_id]
        uu = time_dir * u_val
        vv = time_dir * v_val

        a11 = a11 + 1.0
        a12 = a12 + ex
        a13 = a13 + ey
        a22 = a22 + ex * ex
        a23 = a23 + ex * ey
        a33 = a33 + ey * ey
        fUn = fUn + uu
        f1Un = f1Un + uu * ex
        f2Un = f2Un + uu * ey
        fVn = fVn + vv
        f1Vn = f1Vn + vv * ex
        f2Vn = f2Vn + vv * ey

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
        return 0.0, 0.0, False

    inv = 1.0 / abs(det)

    b11 = a22 * a33 - a23 * a32
    b12 = a13 * a32 - a12 * a33
    b13 = a12 * a23 - a13 * a22

    a0U = (b11 * fUn + b12 * f1Un + b13 * f2Un) * inv
    a0V = (b11 * fVn + b12 * f1Vn + b13 * f2Vn) * inv

    return a0U, a0V, True


@njit(cache=True)
def _interp_leastsq_core_pchip4(
    xp: float,
    yp: float,
    node_idx: int,
    time_dir: float,
    t0: float,
    t1: float,
    t2: float,
    t3: float,
    tq: float,
    node_to_elems_count: np.ndarray,
    node_to_elems: np.ndarray,
    elem_center_lon: np.ndarray,
    elem_center_lat: np.ndarray,
    u0: np.ndarray,
    v0: np.ndarray,
    u1: np.ndarray,
    v1: np.ndarray,
    u2: np.ndarray,
    v2: np.ndarray,
    u3: np.ndarray,
    v3: np.ndarray,
) -> tuple[float, float, bool]:
    """Same as :func:`_interp_leastsq_core` but with 4-snapshot PCHIP-in-time.

    The per-element velocity is the monotone-cubic (Fritsch-Carlson) value at
    query time ``tq`` reconstructed inline from the four snapshots at scalar
    times ``t0<t1<t2<t3`` (``t1 <= tq <= t2``), via
    :func:`numerics.temporal_reconstruct._pchip_value_scalar`.  Everything else
    — metric scaling, centered coordinates, the 3×3 normal system and Cramer
    solve for the constant term ``a0`` — is identical to the steady/linear
    cores, so the result equals the unfused
    ``reconstruct_pchip`` → ``_interp_leastsq_core`` path element-for-element.

    Kh is not reconstructed (velocity/eta/scalars only, per the v1 contract);
    this core handles ``u``/``v`` only.  Used by the fused PCHIP kernels.
    """
    count = int(node_to_elems_count[node_idx])
    if count < 1:
        return 0.0, 0.0, False

    elem_ids = node_to_elems[node_idx, :count]

    z0 = np.cos(yp * DEG2RAD) * METERS_PER_DEG_LAT
    xp_m = xp * z0
    yp_m = yp * METERS_PER_DEG_LAT

    a11 = 0.0
    a12 = 0.0
    a13 = 0.0
    a22 = 0.0
    a23 = 0.0
    a33 = 0.0
    fUn = 0.0
    f1Un = 0.0
    f2Un = 0.0
    fVn = 0.0
    f1Vn = 0.0
    f2Vn = 0.0

    for n in range(count):
        e_id = elem_ids[n]
        ex = elem_center_lon[e_id] * z0 - xp_m
        # Same antimeridian gate as _interp_leastsq_core (off-seam
        # bit-identical, seam-straddling stencils corrected).
        dlon_raw = elem_center_lon[e_id] - xp
        if dlon_raw > 180.0:
            ex -= 360.0 * z0
        elif dlon_raw < -180.0:
            ex += 360.0 * z0
        ey = elem_center_lat[e_id] * METERS_PER_DEG_LAT - yp_m
        # Monotone-cubic blend of the 4 snapshots at this element, inline.
        u_val = _pchip_value_scalar(u0[e_id], u1[e_id], u2[e_id], u3[e_id],
                                    t0, t1, t2, t3, tq)
        v_val = _pchip_value_scalar(v0[e_id], v1[e_id], v2[e_id], v3[e_id],
                                    t0, t1, t2, t3, tq)
        uu = time_dir * u_val
        vv = time_dir * v_val

        a11 = a11 + 1.0
        a12 = a12 + ex
        a13 = a13 + ey
        a22 = a22 + ex * ex
        a23 = a23 + ex * ey
        a33 = a33 + ey * ey
        fUn = fUn + uu
        f1Un = f1Un + uu * ex
        f2Un = f2Un + uu * ey
        fVn = fVn + vv
        f1Vn = f1Vn + vv * ex
        f2Vn = f2Vn + vv * ey

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
        return 0.0, 0.0, False

    inv = 1.0 / abs(det)

    b11 = a22 * a33 - a23 * a32
    b12 = a13 * a32 - a12 * a33
    b13 = a12 * a23 - a13 * a22

    a0U = (b11 * fUn + b12 * f1Un + b13 * f2Un) * inv
    a0V = (b11 * fVn + b12 * f1Vn + b13 * f2Vn) * inv

    return a0U, a0V, True


@njit(cache=True)
def _interp_leastsq_core_fortran(
    xp: float,
    yp: float,
    node_idx: int,
    time_dir: float,
    node_to_elems_count: np.ndarray,
    node_to_elems: np.ndarray,
    elem_center_lon: np.ndarray,
    elem_center_lat: np.ndarray,
    u_field: np.ndarray,
    v_field: np.ndarray,
) -> tuple[float, float, bool]:
    """Fortran-compatible interpolation: absolute coordinates, sequential loop.

    Reproduces the exact floating-point arithmetic of drift-fesom-3d.f90
    lines 317-376 (with the a11 bug fixed). Uses absolute coordinates
    (not centered), which suffers from catastrophic cancellation but
    matches the Fortran output bit-for-bit.
    """
    count = int(node_to_elems_count[node_idx])
    if count < 1:
        return 0.0, 0.0, False

    elem_ids = node_to_elems[node_idx, :count]

    z0 = np.cos(yp * DEG2RAD) * METERS_PER_DEG_LAT
    SLM = METERS_PER_DEG_LAT

    a11 = 0.0
    a12 = 0.0
    a13 = 0.0
    a22 = 0.0
    a23 = 0.0
    a33 = 0.0
    fUn = 0.0
    f1Un = 0.0
    f2Un = 0.0
    fVn = 0.0
    f1Vn = 0.0
    f2Vn = 0.0

    for n in range(count):
        e_id = elem_ids[n]
        a11 = a11 + 1.0
        a12 = a12 + elem_center_lon[e_id] * z0
        a13 = a13 + elem_center_lat[e_id] * SLM
        a22 = a22 + (elem_center_lon[e_id] * z0) ** 2
        a23 = a23 + elem_center_lon[e_id] * z0 * elem_center_lat[e_id] * SLM
        a33 = a33 + (elem_center_lat[e_id] * SLM) ** 2
        fUn = fUn + time_dir * u_field[e_id]
        f1Un = f1Un + time_dir * u_field[e_id] * elem_center_lon[e_id] * z0
        f2Un = f2Un + time_dir * u_field[e_id] * elem_center_lat[e_id] * SLM
        fVn = fVn + time_dir * v_field[e_id]
        f1Vn = f1Vn + time_dir * v_field[e_id] * elem_center_lon[e_id] * z0
        f2Vn = f2Vn + time_dir * v_field[e_id] * elem_center_lat[e_id] * SLM

    a21 = a12
    a31 = a13
    a32 = a23

    AA = (a11 * a22 * a33
          + a12 * a23 * a31
          + a13 * a21 * a32
          - a13 * a22 * a31
          - a12 * a21 * a33
          - a11 * a23 * a32)

    if AA == 0.0:
        return 0.0, 0.0, False

    AA_inv = 1.0 / abs(AA)

    b11 = a22 * a33 - a23 * a32
    b12 = a13 * a32 - a12 * a33
    b13 = a12 * a23 - a13 * a22
    b21 = a23 * a31 - a21 * a33
    b22 = a11 * a33 - a13 * a31
    b23 = a13 * a21 - a11 * a23
    b31 = a21 * a32 - a22 * a31
    b32 = a12 * a31 - a11 * a32
    b33 = a11 * a22 - a12 * a21

    a0U = (b11 * fUn + b12 * f1Un + b13 * f2Un) * AA_inv
    a1U = (b21 * fUn + b22 * f1Un + b23 * f2Un) * AA_inv
    a2U = (b31 * fUn + b32 * f1Un + b33 * f2Un) * AA_inv
    a0V = (b11 * fVn + b12 * f1Vn + b13 * f2Vn) * AA_inv
    a1V = (b21 * fVn + b22 * f1Vn + b23 * f2Vn) * AA_inv
    a2V = (b31 * fVn + b32 * f1Vn + b33 * f2Vn) * AA_inv

    UU = a0U + a1U * xp * z0 + a2U * yp * SLM
    VV = a0V + a1V * xp * z0 + a2V * yp * SLM

    return UU, VV, True


# ---------------------------------------------------------------------------
# Depth-linear u/v (uv_vertical "level" / "layer_linear"): the vectorised
# barycentric interpolator evaluates both layouts (interp_barycentric ->
# _interp_barycentric_dl), the vectorised leastsq and edge_leastsq
# interpolators element layout (interp_leastsq -> _interp_leastsq_dl,
# interp_edge_leastsq -> _interp_edge_leastsq_dl); every other entry, and
# these two for node layout, refuses it before any slicing.
# ---------------------------------------------------------------------------

#: refusal reason of the nearest-element entry
_DL_NEEDS_BARYCENTRIC = "needs uv_interp='barycentric'"
#: refusal reason of the element-layout interpolators for a node field
_DL_LAYOUT = ("on node layout (uv_layout='node') needs "
              "uv_interp='barycentric'; this interpolator evaluates "
              "depth-linear u/v on element layout only")
#: refusal reason of the single-point entries (they take a layer, no depth)
_DL_NO_SINGLE_POINT = ("has no depth argument (no single-point depth-linear "
                       "API in this version)")


def _reject_depth_linear(vel, where: str, reason: str) -> None:
    """Refuse a depth-linear u/v field at an entry that cannot evaluate it.

    Parameters
    ----------
    vel : VelocityField or LazyLerpField
        The field; one without ``uv_vertical`` counts as ``"layer"``.
    where : str
        The entry's name (message prefix).
    reason : str
        Why the entry cannot serve the field (message tail).

    Raises
    ------
    ValueError
        ``vel.uv_vertical`` is not ``"layer"``.
    """
    placement = getattr(vel, "uv_vertical", "layer")
    if placement != "layer":
        raise ValueError(
            f"{where}: depth-linear horizontal velocity "
            f"(uv_vertical={placement!r}) {reason}")


def interp_leastsq_single(
    xp: float,
    yp: float,
    layer: int,
    mesh: Mesh,
    vel: VelocityField,
    node_idx: int,
    time_dir: float = 1.0,
) -> tuple[float, float, bool]:
    """Single-particle polynomial fit (Python wrapper over _interp_leastsq_core)."""
    _reject_depth_linear(vel, "interp_leastsq_single", _DL_NO_SINGLE_POINT)
    if vel.u.ndim == 1:
        u_fi = vel.u
        v_fi = vel.v
    else:
        u_fi = vel.u[layer, :]
        v_fi = vel.v[layer, :]
        
    return _interp_leastsq_core(
        xp, yp, node_idx, time_dir,
        mesh.node_to_elems_count, mesh.node_to_elems,
        mesh.elem_center_lon, mesh.elem_center_lat,
        u_fi, v_fi
    )


# ---------------------------------------------------------------------------
# Vectorised wrapper — loop over all active particles
# ---------------------------------------------------------------------------

def interp_leastsq(
    state: ParticleState,
    mesh: Mesh,
    vel: VelocityField,
    nn_nodes: np.ndarray,
    time_dir: float = 1.0,
    layer=0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate velocity for all active particles using least-squares.

    Parameters
    ----------
    state : ParticleState
    mesh : Mesh
    vel : VelocityField
    nn_nodes : np.ndarray, shape ``(P,)``
        Nearest-node index per particle.
    time_dir : float
        Forward (+1) or backward (-1).
    layer : int or np.ndarray
        Vertical layer index.  Scalar ``int`` for 2-D (all particles use
        the same layer).  ``np.ndarray`` of shape ``(P,)`` for 3-D
        (per-particle layer).  Ignored for a depth-linear field
        (``vel.uv_vertical`` ``"layer_linear"``, element layout): that
        field is read at each particle's depth ``state.z``, on a sigma mesh
        its sigma fraction (:func:`_interp_leastsq_dl`).

    Returns
    -------
    U, V : np.ndarray, shape ``(P,)``
        Interpolated velocity (m/s).  Zero for inactive particles.
    ok : np.ndarray of bool, shape ``(P,)``
        True where interpolation succeeded.

    Raises
    ------
    ValueError
        A depth-linear field on node layout (``uv_interp='barycentric'``
        evaluates it), or a depth-linear record that fails the check.
    """
    if getattr(vel, "uv_vertical", "layer") != "layer":
        return _interp_leastsq_dl(state, mesh, vel, nn_nodes, time_dir)
    P = state.n_particles
    U = np.zeros(P, dtype=np.float64)
    V = np.zeros(P, dtype=np.float64)
    ok = np.zeros(P, dtype=bool)

    # Per-particle layer: 3D path
    per_particle = isinstance(layer, np.ndarray)

    if vel.u.ndim == 1:
        # 2D velocity data — single layer, ignore layer param
        u_fi = vel.u
        v_fi = vel.v
        per_particle = False
    elif not per_particle:
        # 2D tracking with 3D data — pre-slice to single layer
        u_fi = vel.u[layer, :]
        v_fi = vel.v[layer, :]

    if per_particle:
        # 3D: each particle uses its own layer
        layers = np.asarray(layer, dtype=np.int32)
        if HAS_NUMBA:
            _loop_interp_leastsq_3d(
                P, state.alive_mask, state.lon, state.lat, nn_nodes,
                layers, time_dir,
                mesh.node_to_elems_count, mesh.node_to_elems,
                mesh.elem_center_lon, mesh.elem_center_lat,
                vel.u, vel.v, U, V, ok
            )
        else:
            for i in range(P):
                if not state.alive_mask[i]:
                    continue
                u_fi_i = vel.u[layers[i], :]
                v_fi_i = vel.v[layers[i], :]
                u, v, success = _interp_leastsq_core(
                    state.lon[i], state.lat[i],
                    int(nn_nodes[i]), time_dir,
                    mesh.node_to_elems_count, mesh.node_to_elems,
                    mesh.elem_center_lon, mesh.elem_center_lat,
                    u_fi_i, v_fi_i
                )
                U[i] = u
                V[i] = v
                ok[i] = success
    else:
        # 2D: all particles use same pre-sliced layer
        if HAS_NUMBA:
            _loop_interp_leastsq(
                P, state.alive_mask, state.lon, state.lat, nn_nodes, time_dir,
                mesh.node_to_elems_count, mesh.node_to_elems,
                mesh.elem_center_lon, mesh.elem_center_lat,
                u_fi, v_fi, U, V, ok
            )
        else:
            for i in range(P):
                if not state.alive_mask[i]:
                    continue
                u, v, success = _interp_leastsq_core(
                    state.lon[i], state.lat[i],
                    int(nn_nodes[i]), time_dir,
                    mesh.node_to_elems_count, mesh.node_to_elems,
                    mesh.elem_center_lon, mesh.elem_center_lat,
                    u_fi, v_fi
                )
                U[i] = u
                V[i] = v
                ok[i] = success

    return U, V, ok

@njit(parallel=True, cache=True)
def _loop_interp_leastsq(
    P: int, alive_mask: np.ndarray, lons: np.ndarray, lats: np.ndarray,
    nn_nodes: np.ndarray, time_dir: float,
    node_to_elems_count: np.ndarray, node_to_elems: np.ndarray,
    elem_center_lon: np.ndarray, elem_center_lat: np.ndarray,
    u_fi: np.ndarray, v_fi: np.ndarray,
    U: np.ndarray, V: np.ndarray, ok: np.ndarray
):
    for i in prange(P):
        if not alive_mask[i]:
            continue
        u, v, success = _interp_leastsq_core(
            lons[i], lats[i], int(nn_nodes[i]), time_dir,
            node_to_elems_count, node_to_elems,
            elem_center_lon, elem_center_lat,
            u_fi, v_fi
        )
        U[i] = u
        V[i] = v
        ok[i] = success


@njit(parallel=True, cache=True)
def _loop_interp_leastsq_3d(
    P: int, alive_mask: np.ndarray, lons: np.ndarray, lats: np.ndarray,
    nn_nodes: np.ndarray, layers: np.ndarray, time_dir: float,
    node_to_elems_count: np.ndarray, node_to_elems: np.ndarray,
    elem_center_lon: np.ndarray, elem_center_lat: np.ndarray,
    u_2d: np.ndarray, v_2d: np.ndarray,
    U: np.ndarray, V: np.ndarray, ok: np.ndarray
):
    """Per-particle layer interpolation for 3D tracking.

    Each particle uses its own layer index to slice u_2d/v_2d.
    """
    for i in prange(P):
        if not alive_mask[i]:
            continue
        layer_i = layers[i]
        u_fi = u_2d[layer_i]
        v_fi = v_2d[layer_i]
        u, v, success = _interp_leastsq_core(
            lons[i], lats[i], int(nn_nodes[i]), time_dir,
            node_to_elems_count, node_to_elems,
            elem_center_lon, elem_center_lat,
            u_fi, v_fi
        )
        U[i] = u
        V[i] = v
        ok[i] = success


# ---------------------------------------------------------------------------
# Depth-linear twins (uv_vertical "layer_linear", element layout): each
# element evaluated at the particle depth between its own layer centres
# (numerics.vertical_uv.uv_bracket / uv_at); an element dry at that depth
# is a wall, left out like the elements beyond a coast.
# ---------------------------------------------------------------------------

@njit(cache=True)
def _dl_elem_wet_mean(members, n, zq, own, time_dir, p, last, bed,
                      elem_areas, u_f, v_f):
    """Area-weighted mean of the wet elements among ``members[:n]``.

    The fallback of the depth-linear element-layout interpolators when
    their wet members cannot support the interpolator (too few, or a
    singular fit), and when no member is wet; the vertex value of the
    element-layout barycentric twin.  An element ``e`` (entries ``< 0``
    are skipped) is wet at depth ``zq`` when ``uv_wet(bed, e, zq)`` and
    reads its depth-linear value there (``uv_bracket(p, last[e], zq)``,
    ``uv_at``).  The result is ``time_dir * sum(A_e u_e) / sum(A_e)`` over
    the wet members in the order given, ``A_e = elem_areas[e]``
    (:func:`kelp.core.mesh.build_elem_areas`).

    No member wet: the element ``own`` (the particle's containing element)
    read at ``zq`` with ``uv_bracket(p, last[own], zq)`` — its last usable
    point when it is dry too (``zq`` below its bed, so below
    ``p[last[own]]``), its value at ``zq`` when it lies outside the members
    and is wet (``leastsq``'s containing element from the 2-ring search);
    ``own < 0`` (no containing element, or none asked for) gives
    ``ok = False``.  Wet members whose areas sum to zero or less:
    ``ok = False``.  A NaN ``zq`` is read as ``+inf``: dry wherever the bed
    is finite.  The number of wet members is returned too, so a caller can
    tell "none wet" from "wet areas summing to zero or less" without a
    second scan.

    Parameters
    ----------
    members : np.ndarray of int
        Element ids; the first ``n`` are the members.
    n : int
        Number of members.
    zq : float
        Particle depth (m, positive down).
    own : int
        The element read when no member is wet, or ``-1``.
    time_dir : float
        Forward (+1) or backward (-1).
    p, last, bed : np.ndarray
        The element table :func:`kelp.numerics.vertical_uv.depth_points`
        (``layout="element"``): points ``(M,)``, last usable point and bed
        per element ``(E,)``.
    elem_areas : np.ndarray
        Element areas ``(E,)``.
    u_f, v_f : np.ndarray
        Element-layout u/v, ``(M, E)``.

    Returns
    -------
    U, V : float
    ok : bool
    n_wet : int
        The wet members among ``members[:n]``.
    """
    if np.isnan(zq):
        zq = np.inf
    num_u = 0.0
    num_v = 0.0
    den = 0.0
    n_wet = 0
    for j in range(n):
        e = int(members[j])
        if e < 0:
            continue
        if not uv_wet(bed, e, zq):
            continue
        n_wet += 1
        kb, ab = uv_bracket(p, last[e], zq)
        A = elem_areas[e]
        num_u += A * uv_at(u_f, e, kb, ab)
        num_v += A * uv_at(v_f, e, kb, ab)
        den += A
    if n_wet == 0:
        if own < 0:
            return 0.0, 0.0, False, 0
        kb, ab = uv_bracket(p, last[own], zq)
        return (time_dir * uv_at(u_f, own, kb, ab),
                time_dir * uv_at(v_f, own, kb, ab), True, 0)
    if not den > 0.0:
        return 0.0, 0.0, False, n_wet
    return (time_dir * (num_u / den), time_dir * (num_v / den), True,
            n_wet)


@njit(cache=True)
def _interp_leastsq_core_dl(
    xp: float,
    yp: float,
    z: float,
    node_idx: int,
    node_to_elems_count: np.ndarray,
    node_to_elems: np.ndarray,
    elem_center_lon: np.ndarray,
    elem_center_lat: np.ndarray,
    time_dir: float,
    p: np.ndarray,
    last: np.ndarray,
    bed: np.ndarray,
    elem_areas: np.ndarray,
    elem_nodes: np.ndarray,
    nodes_lon: np.ndarray,
    nodes_lat: np.ndarray,
    u_f: np.ndarray,
    v_f: np.ndarray,
) -> tuple[float, float, bool]:
    """Depth-linear least-squares u/v at one point (element layout).

    The plane fit of :func:`_interp_leastsq_core` — the elements around the
    nearest node, the metric ``METERS_PER_DEG_LAT``, particle-centred
    coordinates with the antimeridian gate, the summation order, Sarrus
    and the scale-relative singular guard — on each element's depth-linear
    value at ``z``: ``(k, alpha) = uv_bracket(p, last[e], z)``,
    ``u = uv_at(u_f, e, k, alpha)``, linear in depth between the layer
    centres ``p[k]``, ``p[k + 1]``, clipped to ``p[0] .. p[last[e]]``; a
    point below ``last[e]`` is never read.

    Walls at depth: an element whose bed lies above ``z``
    (``z > bed[e]``) is dry and left out of the fit, as elements beyond a
    coast are absent from a coastal node's ring.  Every element wet: the
    layer core's arithmetic, so a singular all-wet fit stays
    ``ok = False``.  Some dry and the wet ones too few for the fit (fewer
    than three, or a singular fit): their area-weighted mean
    (:func:`_dl_elem_wet_mean`).  None wet (the particle below every bed
    of the ring): the containing element
    (:func:`~kelp.core.mesh._find_containing_element_local`, searched in
    this case only) read at ``z`` — its last usable point, since it is dry
    too unless the 2-ring search found it outside the ring; outside the
    mesh ``ok = False``.  A wet element below ``p[last[e]]`` (a partial bottom
    layer, a ``max_levels`` cut, a declared prefix) reads its last usable
    point.

    ``z`` is the particle depth (m, positive down, as ``state.z``; on a
    sigma mesh the sigma fraction ``s``, the table's coordinate).  A NaN
    ``z`` is read as ``+inf``: dry wherever the bed is finite.

    ``p`` (``(M,)``), ``last`` and ``bed`` (``(E,)``, positive down) are
    the element table of :func:`kelp.numerics.vertical_uv.depth_points`;
    ``elem_areas`` is ``(E,)`` (:func:`kelp.core.mesh.build_elem_areas`);
    ``u_f``/``v_f`` are ``(M, E)``.  Returns ``(U, V, ok)``.
    """
    count = int(node_to_elems_count[node_idx])
    if count < 1:
        return 0.0, 0.0, False

    elem_ids = node_to_elems[node_idx, :count]
    zq = z
    if np.isnan(zq):
        zq = np.inf           # dry wherever the bed is finite
    n_wet = 0
    for n in range(count):
        if uv_wet(bed, elem_ids[n], zq):
            n_wet += 1
    if n_wet == 0:
        # below every bed of the ring: the containing element, read at z
        # (its last usable point, unless the 2-ring search found it
        # outside the ring and wet)
        own, in_dom = _find_containing_element_local(
            xp, yp, node_idx, node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat)
        if not in_dom:
            return 0.0, 0.0, False
        kb, ab = uv_bracket(p, last[own], zq)
        return (time_dir * uv_at(u_f, own, kb, ab),
                time_dir * uv_at(v_f, own, kb, ab), True)
    if n_wet < count and n_wet < 3:
        # too few wet elements for a plane
        U, V, ok, _ = _dl_elem_wet_mean(elem_ids, count, zq, -1, time_dir,
                                        p, last, bed, elem_areas, u_f, v_f)
        return U, V, ok

    # -- the layer core's fit over the wet elements --
    z0 = np.cos(yp * DEG2RAD) * METERS_PER_DEG_LAT
    xp_m = xp * z0
    yp_m = yp * METERS_PER_DEG_LAT

    a11 = 0.0
    a12 = 0.0
    a13 = 0.0
    a22 = 0.0
    a23 = 0.0
    a33 = 0.0
    fUn = 0.0
    f1Un = 0.0
    f2Un = 0.0
    fVn = 0.0
    f1Vn = 0.0
    f2Vn = 0.0

    for n in range(count):
        e_id = elem_ids[n]
        if n_wet < count and not uv_wet(bed, e_id, zq):
            continue          # a wall at this depth: left out
        ex = elem_center_lon[e_id] * z0 - xp_m
        # the antimeridian gate of _interp_leastsq_core
        dlon_raw = elem_center_lon[e_id] - xp
        if dlon_raw > 180.0:
            ex -= 360.0 * z0
        elif dlon_raw < -180.0:
            ex += 360.0 * z0
        ey = elem_center_lat[e_id] * METERS_PER_DEG_LAT - yp_m
        kb, ab = uv_bracket(p, last[e_id], zq)
        uu = time_dir * uv_at(u_f, e_id, kb, ab)
        vv = time_dir * uv_at(v_f, e_id, kb, ab)

        a11 = a11 + 1.0
        a12 = a12 + ex
        a13 = a13 + ey
        a22 = a22 + ex * ex
        a23 = a23 + ex * ey
        a33 = a33 + ey * ey
        fUn = fUn + uu
        f1Un = f1Un + uu * ex
        f2Un = f2Un + uu * ey
        fVn = fVn + vv
        f1Vn = f1Vn + vv * ex
        f2Vn = f2Vn + vv * ey

    a21 = a12
    a31 = a13
    a32 = a23

    det = (a11 * a22 * a33
           + a12 * a23 * a31
           + a13 * a21 * a32
           - a13 * a22 * a31
           - a12 * a21 * a33
           - a11 * a23 * a32)

    # the scale-relative singular guard of _interp_leastsq_core
    if abs(det) < 1e-30 or abs(det) < 1e-12 * abs(a11 * a22 * a33):
        if n_wet == count:
            return 0.0, 0.0, False
        # a wall left a singular fit: the wet elements' area mean
        U, V, ok, _ = _dl_elem_wet_mean(elem_ids, count, zq, -1, time_dir,
                                        p, last, bed, elem_areas, u_f, v_f)
        return U, V, ok

    inv = 1.0 / abs(det)

    b11 = a22 * a33 - a23 * a32
    b12 = a13 * a32 - a12 * a33
    b13 = a12 * a23 - a13 * a22

    a0U = (b11 * fUn + b12 * f1Un + b13 * f2Un) * inv
    a0V = (b11 * fVn + b12 * f1Vn + b13 * f2Vn) * inv

    # value at the particle (dx = dy = 0 in centred coordinates)
    return a0U, a0V, True


@njit(parallel=True, cache=True)
def _loop_interp_leastsq_3d_dl(
    P: int, alive_mask: np.ndarray, lons: np.ndarray, lats: np.ndarray,
    zs: np.ndarray, nn_nodes: np.ndarray, time_dir: float,
    node_to_elems_count: np.ndarray, node_to_elems: np.ndarray,
    elem_center_lon: np.ndarray, elem_center_lat: np.ndarray,
    p: np.ndarray, last: np.ndarray, bed: np.ndarray,
    elem_areas: np.ndarray, elem_nodes: np.ndarray,
    nodes_lon: np.ndarray, nodes_lat: np.ndarray,
    u_f: np.ndarray, v_f: np.ndarray,
    U: np.ndarray, V: np.ndarray, ok: np.ndarray
):
    """Depth-linear least-squares u/v for every alive particle (each at
    its own depth ``zs[i]``); see :func:`_interp_leastsq_core_dl`."""
    for i in prange(P):
        if not alive_mask[i]:
            continue
        u, v, success = _interp_leastsq_core_dl(
            lons[i], lats[i], zs[i], int(nn_nodes[i]),
            node_to_elems_count, node_to_elems,
            elem_center_lon, elem_center_lat, time_dir,
            p, last, bed, elem_areas, elem_nodes, nodes_lon, nodes_lat,
            u_f, v_f
        )
        U[i] = u
        V[i] = v
        ok[i] = success


def _interp_leastsq_dl(
    state: ParticleState,
    mesh: Mesh,
    vel: VelocityField,
    nn_nodes: np.ndarray,
    time_dir: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Depth-linear least-squares u/v for all alive particles.

    The ``interp_leastsq`` route for ``vel.uv_vertical`` ``"layer_linear"``
    on element layout: each particle is read at its depth ``state.z`` (m,
    positive down; on a sigma mesh its sigma fraction ``s = z / H_p``,
    which the steps put in ``state.z``); there is no layer argument.  An
    element whose bed (``mesh.metadata["elem_bed"]``) lies above the
    particle is a wall and left out of the fit
    (:func:`_interp_leastsq_core_dl`).

    Parameters
    ----------
    state : ParticleState
    mesh : Mesh
        A z-level mesh with the element bottoms
        (``mesh.metadata["uv_data_last_elem"]``, ``["elem_bed"]``), or a
        sigma mesh (no bottoms: every element is wet at every ``s``); the
        table is ``depth_points(mesh, vel.uv_vertical, "element")``.
    vel : VelocityField or LazyLerpField
        Element-layout u/v of shape ``(M, E)``.  Checked once per mesh
        table with :func:`kelp.numerics.vertical_uv.check_depth_linear_record`
        (skipped when :func:`~kelp.numerics.vertical_uv.is_checked`).
    nn_nodes : np.ndarray, shape ``(P,)``
        Nearest-node index per particle.
    time_dir : float
        Forward (+1) or backward (-1).

    Returns
    -------
    U, V : np.ndarray, shape ``(P,)``
        Interpolated velocity (m/s).  Zero for inactive particles.
    ok : np.ndarray of bool, shape ``(P,)``
        True where interpolation succeeded.

    Raises
    ------
    ValueError
        Node layout (``uv_interp='barycentric'`` evaluates it), u/v not
        2-D, a mesh neither z-level nor sigma, a z-level mesh without the
        element bottoms, or a record that fails the check (shape,
        non-finite value at a usable point).
    """
    placement = getattr(vel, "uv_vertical", "layer")
    if getattr(vel, "uv_layout", "element") != "element":
        _reject_depth_linear(vel, "interp_leastsq", _DL_LAYOUT)
    # the record check first: a LazyLerpField is checked through its
    # parents (shape and finite prefix) before anything below blends it
    if not is_checked(vel, mesh):
        check_depth_linear_record(vel, mesh)
    # unreachable for a field checked as it stands (the check refuses
    # non-2-D u/v); kept because the token survives an attribute replaced
    # after the check (``vel.u = ...``), which would reach the kernel
    if getattr(vel.u, "ndim", None) != 2 or getattr(vel.v, "ndim", None) != 2:
        raise ValueError(
            f"interp_leastsq: depth-linear u/v (uv_vertical="
            f"{placement!r}) must be 2-D (points, elements), got u "
            f"{getattr(vel.u, 'shape', None)}, v "
            f"{getattr(vel.v, 'shape', None)}")
    dp = depth_points(mesh, placement, "element")
    areas = build_elem_areas(mesh)

    P = state.n_particles
    U = np.zeros(P, dtype=np.float64)
    V = np.zeros(P, dtype=np.float64)
    ok = np.zeros(P, dtype=bool)

    u_f = vel.u
    v_f = vel.v
    if HAS_NUMBA:
        _loop_interp_leastsq_3d_dl(
            P, state.alive_mask, state.lon, state.lat, state.z, nn_nodes,
            time_dir,
            mesh.node_to_elems_count, mesh.node_to_elems,
            mesh.elem_center_lon, mesh.elem_center_lat,
            dp.p, dp.last, dp.bed, areas,
            mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat,
            u_f, v_f, U, V, ok
        )
    else:
        for i in range(P):
            if not state.alive_mask[i]:
                continue
            u, v, success = _interp_leastsq_core_dl(
                state.lon[i], state.lat[i], state.z[i], int(nn_nodes[i]),
                mesh.node_to_elems_count, mesh.node_to_elems,
                mesh.elem_center_lon, mesh.elem_center_lat, time_dir,
                dp.p, dp.last, dp.bed, areas,
                mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat,
                u_f, v_f
            )
            U[i] = u
            V[i] = v
            ok[i] = success

    return U, V, ok


# ---------------------------------------------------------------------------
# Nearest-element fallback
# ---------------------------------------------------------------------------

def interp_nearest_element(
    elem_ids: np.ndarray,
    vel: VelocityField,
    layer: int = 0,
    time_dir: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple nearest-element interpolation (no spatial fitting).

    Parameters
    ----------
    elem_ids : np.ndarray of int, shape ``(P,)``
        Element index per particle (-1 if outside).
    vel : VelocityField
    layer : int
    time_dir : float

    Returns
    -------
    U, V : np.ndarray, shape ``(P,)``
    ok : np.ndarray of bool
    """
    _reject_depth_linear(vel, "interp_nearest_element", _DL_NEEDS_BARYCENTRIC)
    valid = elem_ids >= 0
    safe_ids = np.where(valid, elem_ids, 0)

    if vel.u.ndim == 1:
        U = time_dir * vel.u[safe_ids]
        V = time_dir * vel.v[safe_ids]
    else:
        U = time_dir * vel.u[layer, safe_ids]
        V = time_dir * vel.v[layer, safe_ids]

    U = np.where(valid, U, 0.0)
    V = np.where(valid, V, 0.0)
    return U, V, valid


# ---------------------------------------------------------------------------
# Vertex-barycentric u/v (spec 2026-06-06): containing element -> vertex
# values (node layout: direct; element layout: area-weighted average of the
# elements sharing the vertex) -> _polygon_weights blend.
# ---------------------------------------------------------------------------
UV_LAYOUT_ELEMENT = 0
UV_LAYOUT_NODE = 1


@njit(cache=True)
def _interp_barycentric_core(xp, yp, nn_node, elem_hint, time_dir,
                             layout_code,
                             node_to_elems, node_to_elems_count,
                             elem_nodes, nodes_lon, nodes_lat,
                             elem_areas, u_f, v_f, r_earth):
    """Vertex-barycentric u/v at one point.

    ``u_f``/``v_f`` are 1-D slices: ``(E,)`` for
    ``layout_code == UV_LAYOUT_ELEMENT``, ``(N,)`` for ``UV_LAYOUT_NODE``.
    ``elem_areas`` is ``(E,)`` for the element layout and may be empty for
    the node layout (never indexed there).  ``r_earth`` is the mesh's Earth
    radius (m): the weights are scale-invariant so the VALUE returned is
    independent of it — the parameter exists only to keep the local metric
    projection consistent with ``mesh.r_earth`` (the other mesh metrics).

    ``elem_hint >= 0`` is a PRE-VALIDATED containing element (from the
    edge-walk tracker, computed for exactly this position) — the internal
    nearest-node containment search is skipped; ``-1`` = search here.
    Returns ``(U, V, ok)``.
    """
    if elem_hint >= 0:
        elem_id = int(elem_hint)
    else:
        elem_id, in_dom = _find_containing_element_local(
            xp, yp, nn_node, node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat)
        if not in_dom:
            return 0.0, 0.0, False
    max_v = elem_nodes.shape[1]
    verts = np.empty(max_v, dtype=np.int64)
    nv = 0
    for k in range(max_v):
        nd = int(elem_nodes[elem_id, k])
        if nd < 0:
            continue
        if nv > 0 and nd == verts[0]:
            continue          # FESOM-C tri stored as quad repeats vertex 0
        verts[nv] = nd
        nv += 1
    if nv < 3:
        return 0.0, 0.0, False
    # local metric projection around the element centroid (idiom of
    # vertical._interp_w_barycentric_single).  Anchor the longitude
    # origin on the FIRST vertex and accumulate wrapped deltas: a raw
    # arithmetic mean of [179.5, -179.5, ...] collapses to ~0, putting
    # the projection origin 180 deg from the cell and corrupting every
    # wrapped delta below (dateline straddle).
    DEG2R = 0.017453292519943295
    # Weights are scale-invariant, so the returned U/V do not depend on R;
    # use mesh.r_earth purely for unit consistency with the mesh metrics.
    R = r_earth
    lon_ref = nodes_lon[verts[0]]
    dsum = 0.0; lat0 = 0.0
    for k in range(nv):
        dsum += ((nodes_lon[verts[k]] - lon_ref + 180.0) % 360.0) - 180.0
        lat0 += nodes_lat[verts[k]]
    lon0 = lon_ref + dsum / nv
    lat0 /= nv
    cosl = np.cos(lat0 * DEG2R)
    vx = np.empty(nv); vy = np.empty(nv)
    for k in range(nv):
        dlon = ((nodes_lon[verts[k]] - lon0 + 180.0) % 360.0) - 180.0
        vx[k] = dlon * DEG2R * R * cosl
        vy[k] = (nodes_lat[verts[k]] - lat0) * DEG2R * R
    dlon_p = ((xp - lon0 + 180.0) % 360.0) - 180.0
    px = dlon_p * DEG2R * R * cosl
    py = (yp - lat0) * DEG2R * R
    w = _polygon_weights(px, py, vx, vy, nv)
    if w[0] == -1.0:
        # Only the degenerate-TRIANGLE sentinel of the untouched tri
        # primitive can land here (quads are sentinel-free product-form
        # Wachspress since 2026-07).
        return 0.0, 0.0, False
    U = 0.0; V = 0.0
    for k in range(nv):
        nd = verts[k]
        if layout_code == UV_LAYOUT_NODE:
            un = u_f[nd]; vn = v_f[nd]
        else:
            num_u = 0.0; num_v = 0.0; den = 0.0
            cnt = int(node_to_elems_count[nd])
            for j in range(cnt):
                e = int(node_to_elems[nd, j])
                if e < 0:
                    continue
                A = elem_areas[e]
                num_u += A * u_f[e]; num_v += A * v_f[e]; den += A
            if den <= 0.0:
                return 0.0, 0.0, False
            un = num_u / den; vn = num_v / den
        U += w[k] * un; V += w[k] * vn
    return time_dir * U, time_dir * V, True


@njit(cache=True)
def _interp_barycentric_core_tinterp(xp, yp, nn_node, time_dir,
                                     alpha0, alpha,
                                     node_to_elems, node_to_elems_count,
                                     elem_nodes, nodes_lon, nodes_lat,
                                     u_a, v_a, u_b, v_b, u_t, v_t, r_earth):
    """NODE-layout twin of :func:`_interp_barycentric_core` with the inline
    two-level time blend of :func:`_interp_leastsq_core_tinterp`.

    Per VERTEX ``nd``::

        ua0    = (1 - alpha0) * u_a[nd] + alpha0 * u_b[nd]  # field at sim_time
        u[nd]  = (1 - alpha)  * ua0     + alpha  * u_t[nd]  # stage blend

    ``(u_a, u_b)`` is the raw bracket pair (``alpha0`` = bracket fraction,
    reproducing the ``_blend`` materialization bitwise), ``(u_t, v_t)`` the
    stage-target snapshot (``vel_next``).  ``alpha0 == 0.0`` collapses to the
    single-blend expression exactly (the nested form would compute
    ``0.0 * u_b[nd]``, which turns a non-finite ``u_b`` into NaN where the
    single blend propagates it); stage-endpoint alphas short-circuit the same
    way (``alpha == 0.0`` → the bracket value, ``alpha == 1.0`` → the target
    value), mirroring ``_make_stage_vels``' object shortcuts — K1 always has
    ``alpha == 0.0``.  Geometry (containment search, compacted
    vertex ring, centroid-anchored metric projection, ``_polygon_weights``)
    is byte-for-byte the untouched core's.  Node layout ONLY — the fused
    kernels never run for element-layout barycentric (dispatch gate).
    Returns ``(U, V, ok)``.
    """
    elem_id, in_dom = _find_containing_element_local(
        xp, yp, nn_node, node_to_elems, node_to_elems_count,
        elem_nodes, nodes_lon, nodes_lat)
    if not in_dom:
        return 0.0, 0.0, False
    max_v = elem_nodes.shape[1]
    verts = np.empty(max_v, dtype=np.int64)
    nv = 0
    for k in range(max_v):
        nd = int(elem_nodes[elem_id, k])
        if nd < 0:
            continue
        if nv > 0 and nd == verts[0]:
            continue          # FESOM-C tri stored as quad repeats vertex 0
        verts[nv] = nd
        nv += 1
    if nv < 3:
        return 0.0, 0.0, False
    DEG2R = 0.017453292519943295
    R = r_earth
    lon_ref = nodes_lon[verts[0]]
    dsum = 0.0; lat0 = 0.0
    for k in range(nv):
        dsum += ((nodes_lon[verts[k]] - lon_ref + 180.0) % 360.0) - 180.0
        lat0 += nodes_lat[verts[k]]
    lon0 = lon_ref + dsum / nv
    lat0 /= nv
    cosl = np.cos(lat0 * DEG2R)
    vx = np.empty(nv); vy = np.empty(nv)
    for k in range(nv):
        dlon = ((nodes_lon[verts[k]] - lon0 + 180.0) % 360.0) - 180.0
        vx[k] = dlon * DEG2R * R * cosl
        vy[k] = (nodes_lat[verts[k]] - lat0) * DEG2R * R
    dlon_p = ((xp - lon0 + 180.0) % 360.0) - 180.0
    px = dlon_p * DEG2R * R * cosl
    py = (yp - lat0) * DEG2R * R
    w = _polygon_weights(px, py, vx, vy, nv)
    if w[0] == -1.0:
        return 0.0, 0.0, False
    beta0 = 1.0 - alpha0
    beta = 1.0 - alpha
    U = 0.0; V = 0.0
    for k in range(nv):
        nd = verts[k]
        # Endpoint stages skip the blend entirely, with the SAME 1e-12
        # thresholds as _make_stage_vels' object shortcuts (bitwise
        # parity with the unfused path even for freak dt/dt_to_next
        # ratios): K1 always has alpha == 0.0 exactly (c1 = 0), and K4
        # hits alpha == 1.0 when dt == dt_to_next — multiplying the
        # unused endpoint array by ~0 would turn a non-finite entry
        # there into NaN (0*Inf) where the unfused path returns the
        # finite endpoint value.  For finite fields the exact-endpoint
        # shortcut is the same value (x + 0.0 == x).
        if alpha < 1e-12:
            if alpha0 == 0.0:
                un = u_a[nd]
                vn = v_a[nd]
            else:
                un = beta0 * u_a[nd] + alpha0 * u_b[nd]
                vn = beta0 * v_a[nd] + alpha0 * v_b[nd]
        elif alpha > 1.0 - 1e-12:
            un = u_t[nd]
            vn = v_t[nd]
        elif alpha0 == 0.0:
            un = beta * u_a[nd] + alpha * u_t[nd]
            vn = beta * v_a[nd] + alpha * v_t[nd]
        else:
            ua0 = beta0 * u_a[nd] + alpha0 * u_b[nd]
            va0 = beta0 * v_a[nd] + alpha0 * v_b[nd]
            un = beta * ua0 + alpha * u_t[nd]
            vn = beta * va0 + alpha * v_t[nd]
        U += w[k] * un; V += w[k] * vn
    return time_dir * U, time_dir * V, True


@njit(cache=True)
def _interp_barycentric_core_tinterp_3d(xp, yp, nn_node, time_dir, layer,
                                     alpha0, alpha,
                                     node_to_elems, node_to_elems_count,
                                     elem_nodes, nodes_lon, nodes_lat,
                                     u_a, v_a, u_b, v_b, u_t, v_t, r_earth):
    """3-D twin of :func:`_interp_barycentric_core_tinterp`: identical
    geometry and blend conventions, u/v fields are ``(K-1, N)`` and
    every vertex read is taken at the particle's mid-layer ``layer``
    (u/v are layer-piecewise in the vertical — the unfused 3-D path
    interpolates horizontally within the particle's layer only).
    Returns ``(U, V, ok)``."""
    elem_id, in_dom = _find_containing_element_local(
        xp, yp, nn_node, node_to_elems, node_to_elems_count,
        elem_nodes, nodes_lon, nodes_lat)
    if not in_dom:
        return 0.0, 0.0, False
    max_v = elem_nodes.shape[1]
    verts = np.empty(max_v, dtype=np.int64)
    nv = 0
    for k in range(max_v):
        nd = int(elem_nodes[elem_id, k])
        if nd < 0:
            continue
        if nv > 0 and nd == verts[0]:
            continue          # FESOM-C tri stored as quad repeats vertex 0
        verts[nv] = nd
        nv += 1
    if nv < 3:
        return 0.0, 0.0, False
    DEG2R = 0.017453292519943295
    R = r_earth
    lon_ref = nodes_lon[verts[0]]
    dsum = 0.0; lat0 = 0.0
    for k in range(nv):
        dsum += ((nodes_lon[verts[k]] - lon_ref + 180.0) % 360.0) - 180.0
        lat0 += nodes_lat[verts[k]]
    lon0 = lon_ref + dsum / nv
    lat0 /= nv
    cosl = np.cos(lat0 * DEG2R)
    vx = np.empty(nv); vy = np.empty(nv)
    for k in range(nv):
        dlon = ((nodes_lon[verts[k]] - lon0 + 180.0) % 360.0) - 180.0
        vx[k] = dlon * DEG2R * R * cosl
        vy[k] = (nodes_lat[verts[k]] - lat0) * DEG2R * R
    dlon_p = ((xp - lon0 + 180.0) % 360.0) - 180.0
    px = dlon_p * DEG2R * R * cosl
    py = (yp - lat0) * DEG2R * R
    w = _polygon_weights(px, py, vx, vy, nv)
    if w[0] == -1.0:
        return 0.0, 0.0, False
    beta0 = 1.0 - alpha0
    beta = 1.0 - alpha
    U = 0.0; V = 0.0
    for k in range(nv):
        nd = verts[k]
        # Endpoint stages skip the blend entirely, with the SAME 1e-12
        # thresholds as _make_stage_vels' object shortcuts (bitwise
        # parity with the unfused path even for freak dt/dt_to_next
        # ratios): K1 always has alpha == 0.0 exactly (c1 = 0), and K4
        # hits alpha == 1.0 when dt == dt_to_next — multiplying the
        # unused endpoint array by ~0 would turn a non-finite entry
        # there into NaN (0*Inf) where the unfused path returns the
        # finite endpoint value.  For finite fields the exact-endpoint
        # shortcut is the same value (x + 0.0 == x).
        if alpha < 1e-12:
            if alpha0 == 0.0:
                un = u_a[layer, nd]
                vn = v_a[layer, nd]
            else:
                un = beta0 * u_a[layer, nd] + alpha0 * u_b[layer, nd]
                vn = beta0 * v_a[layer, nd] + alpha0 * v_b[layer, nd]
        elif alpha > 1.0 - 1e-12:
            un = u_t[layer, nd]
            vn = v_t[layer, nd]
        elif alpha0 == 0.0:
            un = beta * u_a[layer, nd] + alpha * u_t[layer, nd]
            vn = beta * v_a[layer, nd] + alpha * v_t[layer, nd]
        else:
            ua0 = beta0 * u_a[layer, nd] + alpha0 * u_b[layer, nd]
            va0 = beta0 * v_a[layer, nd] + alpha0 * v_b[layer, nd]
            un = beta * ua0 + alpha * u_t[layer, nd]
            vn = beta * va0 + alpha * v_t[layer, nd]
        U += w[k] * un; V += w[k] * vn
    return time_dir * U, time_dir * V, True




@njit(cache=True)
def _interp_barycentric_core_pchip4_3d(xp, yp, nn_node, time_dir, layer,
                                       node_to_elems, node_to_elems_count,
                                       elem_nodes, nodes_lon, nodes_lat,
                                       u_0, v_0, u_1, v_1, u_2, v_2,
                                       u_3, v_3, t0, t1, t2, t3, tq,
                                       r_earth):
    """PCHIP twin of :func:`_interp_barycentric_core_tinterp_3d`: identical
    geometry, per-vertex 4-snapshot monotone-cubic blend via
    ``_pchip_value_scalar`` (bit-for-bit the arithmetic of the unfused
    ``reconstruct_pchip`` materialization).  Returns ``(U, V, ok)``."""
    elem_id, in_dom = _find_containing_element_local(
        xp, yp, nn_node, node_to_elems, node_to_elems_count,
        elem_nodes, nodes_lon, nodes_lat)
    if not in_dom:
        return 0.0, 0.0, False
    max_v = elem_nodes.shape[1]
    verts = np.empty(max_v, dtype=np.int64)
    nv = 0
    for k in range(max_v):
        nd = int(elem_nodes[elem_id, k])
        if nd < 0:
            continue
        if nv > 0 and nd == verts[0]:
            continue          # FESOM-C tri stored as quad repeats vertex 0
        verts[nv] = nd
        nv += 1
    if nv < 3:
        return 0.0, 0.0, False
    DEG2R = 0.017453292519943295
    R = r_earth
    lon_ref = nodes_lon[verts[0]]
    dsum = 0.0; lat0 = 0.0
    for k in range(nv):
        dsum += ((nodes_lon[verts[k]] - lon_ref + 180.0) % 360.0) - 180.0
        lat0 += nodes_lat[verts[k]]
    lon0 = lon_ref + dsum / nv
    lat0 /= nv
    cosl = np.cos(lat0 * DEG2R)
    vx = np.empty(nv); vy = np.empty(nv)
    for k in range(nv):
        dlon = ((nodes_lon[verts[k]] - lon0 + 180.0) % 360.0) - 180.0
        vx[k] = dlon * DEG2R * R * cosl
        vy[k] = (nodes_lat[verts[k]] - lat0) * DEG2R * R
    dlon_p = ((xp - lon0 + 180.0) % 360.0) - 180.0
    px = dlon_p * DEG2R * R * cosl
    py = (yp - lat0) * DEG2R * R
    w = _polygon_weights(px, py, vx, vy, nv)
    if w[0] == -1.0:
        return 0.0, 0.0, False
    U = 0.0; V = 0.0
    for k in range(nv):
        nd = verts[k]
        un = _pchip_value_scalar(u_0[layer, nd], u_1[layer, nd],
                                 u_2[layer, nd], u_3[layer, nd],
                                 t0, t1, t2, t3, tq)
        vn = _pchip_value_scalar(v_0[layer, nd], v_1[layer, nd],
                                 v_2[layer, nd], v_3[layer, nd],
                                 t0, t1, t2, t3, tq)
        U += w[k] * un; V += w[k] * vn
    return time_dir * U, time_dir * V, True


@njit(parallel=True, cache=True)
def _loop_interp_bary(
    P: int, alive_mask: np.ndarray, lons: np.ndarray, lats: np.ndarray,
    nn_nodes: np.ndarray, elems: np.ndarray, time_dir: float,
    layout_code: int,
    node_to_elems: np.ndarray, node_to_elems_count: np.ndarray,
    elem_nodes: np.ndarray, nodes_lon: np.ndarray, nodes_lat: np.ndarray,
    elem_areas: np.ndarray, u_fi: np.ndarray, v_fi: np.ndarray,
    U: np.ndarray, V: np.ndarray, ok: np.ndarray, r_earth: float
):
    for i in prange(P):
        if not alive_mask[i]:
            continue
        u, v, success = _interp_barycentric_core(
            lons[i], lats[i], int(nn_nodes[i]), int(elems[i]), time_dir,
            layout_code,
            node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat,
            elem_areas, u_fi, v_fi, r_earth
        )
        U[i] = u
        V[i] = v
        ok[i] = success


@njit(parallel=True, cache=True)
def _loop_interp_bary_3d(
    P: int, alive_mask: np.ndarray, lons: np.ndarray, lats: np.ndarray,
    nn_nodes: np.ndarray, elems: np.ndarray, layers: np.ndarray,
    time_dir: float,
    layout_code: int,
    node_to_elems: np.ndarray, node_to_elems_count: np.ndarray,
    elem_nodes: np.ndarray, nodes_lon: np.ndarray, nodes_lat: np.ndarray,
    elem_areas: np.ndarray, u_2d: np.ndarray, v_2d: np.ndarray,
    U: np.ndarray, V: np.ndarray, ok: np.ndarray, r_earth: float
):
    """Per-particle layer vertex-barycentric u/v for 3D tracking.

    Each particle slices ``u_2d``/``v_2d`` on its own layer; the layout
    code (node vs element) decides whether the slice is indexed by node or
    area-averaged over the elements sharing each vertex — so the SAME loop
    serves both layouts.
    """
    for i in prange(P):
        if not alive_mask[i]:
            continue
        layer_i = layers[i]
        u_fi = u_2d[layer_i]
        v_fi = v_2d[layer_i]
        u, v, success = _interp_barycentric_core(
            lons[i], lats[i], int(nn_nodes[i]), int(elems[i]), time_dir,
            layout_code,
            node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat,
            elem_areas, u_fi, v_fi, r_earth
        )
        U[i] = u
        V[i] = v
        ok[i] = success


# ---------------------------------------------------------------------------
# Depth-linear twins (uv_vertical "level" / "layer_linear", node layout):
# the geometry of _interp_barycentric_core, each vertex value read linearly
# in depth between that vertex's bracketing data points
# (numerics.vertical_uv.uv_bracket / uv_at); a vertex dry at the particle
# depth is a wall, left out like the coast (_dl_node_wall).
# ---------------------------------------------------------------------------

@njit(cache=True)
def _dl_node_wall(w, verts, nv, bed, zq):
    """The wall rule over the ``nv`` vertices of one element (node layout).

    A vertex ``nd`` is wet at depth ``zq`` when ``uv_wet(bed, nd, zq)``
    (``zq <= bed[nd]``, ``bed`` the physical bed; the caller has mapped a
    NaN particle depth to ``+inf``: dry wherever the bed is finite).
    Returns ``(n_wet, den, k_deep)``:

    - ``n_wet == nv`` — every vertex wet: the caller's plain barycentric
      sum, unchanged (``den``, ``k_deep`` unused; ``w`` untouched);
    - ``0 < n_wet < nv`` — the dry vertices are left out and the sum over
      the wet ones is divided by ``den``: the sum of their weights, each
      negative one first set to 0.0 IN ``w`` (containment accepts a
      particle just outside an edge, weights down to ``_BARY_EPS``; a
      mixed-sign renormalised sum would leave the hull of the wet values),
      or, when that sum is below 1e-12 (the particle on a dry vertex),
      ``n_wet`` — the weights of the wet vertices are then set to 1.0 IN
      ``w`` (the equal-weight mean of the wet vertex values);
    - ``n_wet == 0`` — no vertex wet: ``k_deep`` is the position in
      ``verts`` of the vertex with the largest bed (the first on ties),
      which the caller reads at its last usable point.
    """
    n_wet = 0
    for k in range(nv):
        if uv_wet(bed, verts[k], zq):
            n_wet += 1
    if n_wet == nv:
        return n_wet, 1.0, -1
    if n_wet == 0:
        k_deep = 0
        for k in range(1, nv):
            if bed[verts[k]] > bed[verts[k_deep]]:
                k_deep = k
        return 0, 1.0, k_deep
    # some vertex dry: the wet weights, a negative one counted as zero
    wsum = 0.0
    for k in range(nv):
        if uv_wet(bed, verts[k], zq):
            if w[k] < 0.0:
                w[k] = 0.0
            wsum += w[k]
    if wsum < 1e-12:
        for k in range(nv):
            if uv_wet(bed, verts[k], zq):
                w[k] = 1.0
        return n_wet, float(n_wet), -1
    return n_wet, wsum, -1


@njit(cache=True)
def _interp_barycentric_core_dl(xp, yp, z, nn_node, elem_hint, time_dir,
                                node_to_elems, node_to_elems_count,
                                elem_nodes, nodes_lon, nodes_lat,
                                p, last, bed, u_f, v_f, r_earth):
    """Depth-linear vertex-barycentric u/v at one point (node layout).

    Containment search (1-ring, then the 2-ring fallback of
    ``_find_containing_element_local``), the ``elem_hint`` shortcut, the
    compacted vertex ring, the centroid-anchored metric projection and
    ``_polygon_weights`` are those of :func:`_interp_barycentric_core`
    (node layout).  The per-vertex value differs: at vertex ``nd``,
    ``(k, alpha) = uv_bracket(p, last[nd], z)`` and
    ``u = uv_at(u_f, nd, k, alpha)`` — linear in depth between the data
    points ``p[k]``, ``p[k + 1]``, clipped to ``p[0] .. p[last[nd]]``; a
    point below ``last[nd]`` is never read.

    Walls at depth (:func:`_dl_node_wall`): a vertex whose bed lies above
    ``z`` (``z > bed[nd]``) is dry and left out, as the coast is; the
    weights of the wet vertices are renormalised, a negative one (a
    particle just outside an edge) counted as zero (their equal-weight
    mean when those weights sum below 1e-12).  Every vertex wet: the plain
    sum, the arithmetic of the layer kernel (a negative weight kept).  No
    vertex wet (the particle below every bed of the element): the vertex
    with the largest bed, first on ties, at its last usable point.  A wet
    vertex below ``p[last[nd]]`` (a partial bottom layer, a ``max_levels``
    cut, a declared prefix) reads its last usable point.

    ``z`` is the particle depth (m, positive down, as ``state.z``; on a
    sigma mesh the sigma fraction ``s``, the table's coordinate).  A NaN
    ``z`` is read as ``+inf``: dry wherever the bed is finite, so the
    deepest vertex at its last point (ok=True).

    ``p`` (``(M,)``), ``last`` (``(N,)``) and ``bed`` (``(N,)``, positive
    down) are the :func:`kelp.numerics.vertical_uv.depth_points` table;
    ``u_f``/``v_f`` are ``(M, N)``.  Returns ``(U, V, ok)``.
    """
    if elem_hint >= 0:
        elem_id = int(elem_hint)
    else:
        elem_id, in_dom = _find_containing_element_local(
            xp, yp, nn_node, node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat)
        if not in_dom:
            return 0.0, 0.0, False
    max_v = elem_nodes.shape[1]
    verts = np.empty(max_v, dtype=np.int64)
    nv = 0
    for k in range(max_v):
        nd = int(elem_nodes[elem_id, k])
        if nd < 0:
            continue
        if nv > 0 and nd == verts[0]:
            continue          # FESOM-C tri stored as quad repeats vertex 0
        verts[nv] = nd
        nv += 1
    if nv < 3:
        return 0.0, 0.0, False
    DEG2R = 0.017453292519943295
    R = r_earth
    lon_ref = nodes_lon[verts[0]]
    dsum = 0.0; lat0 = 0.0
    for k in range(nv):
        dsum += ((nodes_lon[verts[k]] - lon_ref + 180.0) % 360.0) - 180.0
        lat0 += nodes_lat[verts[k]]
    lon0 = lon_ref + dsum / nv
    lat0 /= nv
    cosl = np.cos(lat0 * DEG2R)
    vx = np.empty(nv); vy = np.empty(nv)
    for k in range(nv):
        dlon = ((nodes_lon[verts[k]] - lon0 + 180.0) % 360.0) - 180.0
        vx[k] = dlon * DEG2R * R * cosl
        vy[k] = (nodes_lat[verts[k]] - lat0) * DEG2R * R
    dlon_p = ((xp - lon0 + 180.0) % 360.0) - 180.0
    px = dlon_p * DEG2R * R * cosl
    py = (yp - lat0) * DEG2R * R
    w = _polygon_weights(px, py, vx, vy, nv)
    if w[0] == -1.0:
        return 0.0, 0.0, False
    zq = z
    if np.isnan(zq):
        zq = np.inf           # dry wherever the bed is finite
    # may set the wet weights to 1.0 in w (the equal-weight mean)
    n_wet, den, k_deep = _dl_node_wall(w, verts, nv, bed, zq)
    if n_wet == 0:
        nd = verts[k_deep]    # below every bed: the deepest vertex
        return (time_dir * uv_at(u_f, nd, last[nd], -1.0),
                time_dir * uv_at(v_f, nd, last[nd], -1.0), True)
    U = 0.0; V = 0.0
    for k in range(nv):
        nd = verts[k]
        if n_wet < nv and not uv_wet(bed, nd, zq):
            continue          # a wall at this depth: left out
        kb, ab = uv_bracket(p, last[nd], zq)
        un = uv_at(u_f, nd, kb, ab)
        vn = uv_at(v_f, nd, kb, ab)
        U += w[k] * un; V += w[k] * vn
    if n_wet < nv:
        U = U / den; V = V / den
    return time_dir * U, time_dir * V, True


@njit(parallel=True, cache=True)
def _loop_interp_bary_dl(
    P: int, alive_mask: np.ndarray, lons: np.ndarray, lats: np.ndarray,
    zs: np.ndarray, nn_nodes: np.ndarray, elems: np.ndarray,
    time_dir: float,
    node_to_elems: np.ndarray, node_to_elems_count: np.ndarray,
    elem_nodes: np.ndarray, nodes_lon: np.ndarray, nodes_lat: np.ndarray,
    p: np.ndarray, last: np.ndarray, bed: np.ndarray,
    u_f: np.ndarray, v_f: np.ndarray,
    U: np.ndarray, V: np.ndarray, ok: np.ndarray, r_earth: float
):
    """Depth-linear vertex-barycentric u/v for every alive particle (each
    at its own depth ``zs[i]``); see :func:`_interp_barycentric_core_dl`."""
    for i in prange(P):
        if not alive_mask[i]:
            continue
        u, v, success = _interp_barycentric_core_dl(
            lons[i], lats[i], zs[i], int(nn_nodes[i]), int(elems[i]),
            time_dir,
            node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat,
            p, last, bed, u_f, v_f, r_earth
        )
        U[i] = u
        V[i] = v
        ok[i] = success


# Fused-kernel cores (time-blended inline): each stored (point, node) value
# is blended in time first, then read in depth, then combined.

@njit(cache=True)
def _tinterp_point_uv(u_a, v_a, u_b, v_b, u_t, v_t, k, nd, alpha0, alpha):
    """Time blend of the stored u/v value ``(k, nd)`` for the fused
    depth-linear twins: the per-vertex expressions and endpoint shortcuts of
    :func:`_interp_barycentric_core_tinterp_3d` (``u_*[layer, nd]`` there),
    which reproduce the unfused ``_make_stage_vels`` / ``_blend`` values bit
    for bit.  The endpoint stages (``alpha`` within 1e-12 of 0 or 1, the
    thresholds of ``_make_stage_vels``) skip the blend, so a non-finite
    value in the unused endpoint never turns into NaN (0 * inf).  Every
    stored value is widened to float64 before any arithmetic, as ``_blend``
    widens the arrays: without JIT a float32 numpy scalar would otherwise
    keep the blend in float32 (NEP 50); compiled, the widening is exact and
    changes no bit.  Returns float64 ``(u, v)``."""
    if alpha < 1e-12:
        if alpha0 == 0.0:
            return float(u_a[k, nd]), float(v_a[k, nd])
        beta0 = 1.0 - alpha0
        return (beta0 * float(u_a[k, nd]) + alpha0 * float(u_b[k, nd]),
                beta0 * float(v_a[k, nd]) + alpha0 * float(v_b[k, nd]))
    if alpha > 1.0 - 1e-12:
        return float(u_t[k, nd]), float(v_t[k, nd])
    beta = 1.0 - alpha
    if alpha0 == 0.0:
        return (beta * float(u_a[k, nd]) + alpha * float(u_t[k, nd]),
                beta * float(v_a[k, nd]) + alpha * float(v_t[k, nd]))
    beta0 = 1.0 - alpha0
    ua0 = beta0 * float(u_a[k, nd]) + alpha0 * float(u_b[k, nd])
    va0 = beta0 * float(v_a[k, nd]) + alpha0 * float(v_b[k, nd])
    return (beta * ua0 + alpha * float(u_t[k, nd]),
            beta * va0 + alpha * float(v_t[k, nd]))


@njit(cache=True)
def _interp_barycentric_core_tinterp_dl(xp, yp, nn_node, z, time_dir,
                                        alpha0, alpha,
                                        node_to_elems, node_to_elems_count,
                                        elem_nodes, nodes_lon, nodes_lat,
                                        p, last, bed,
                                        u_a, v_a, u_b, v_b, u_t, v_t,
                                        r_earth):
    """Depth-linear twin of :func:`_interp_barycentric_core_tinterp_3d`
    (node layout, ``uv_vertical`` ``"level"`` / ``"layer_linear"``).

    Identical containment, vertex ring, metric projection and weights;
    the ``layer`` argument is replaced by the particle depth ``z`` (m,
    positive down) and the depth-point table ``p`` (``(M,)``), ``last``
    (``(N,)``), ``bed`` (``(N,)``); u/v fields are ``(M, N)``.  Per vertex
    ``nd``, TIME FIRST: each stored value ``(k, nd)`` the vertex reads is
    blended in time exactly as the layer twin blends ``(layer, nd)`` (same
    expressions, same endpoint shortcuts — :func:`_tinterp_point_uv`),
    THEN the vertical rule of ``uv_bracket`` / ``uv_at`` runs on the
    blended values: ``(k, a) = uv_bracket(p, last[nd], z)``; ``a < 0``
    reads point ``k`` only, otherwise ``(1 - a) * f_k + a * f_{k+1}``.  A
    point below ``last[nd]`` is never read.  The walls at depth of
    :func:`_interp_barycentric_core_dl` (:func:`_dl_node_wall`) decide
    which vertices enter and how they are weighted; a dry vertex is never
    read.  This is bit for bit the unfused path (whole-field ``_blend``
    materialization, then :func:`_interp_barycentric_core_dl`), including
    its NaN-depth rule (read as ``+inf``).  Returns ``(U, V, ok)``."""
    elem_id, in_dom = _find_containing_element_local(
        xp, yp, nn_node, node_to_elems, node_to_elems_count,
        elem_nodes, nodes_lon, nodes_lat)
    if not in_dom:
        return 0.0, 0.0, False
    max_v = elem_nodes.shape[1]
    verts = np.empty(max_v, dtype=np.int64)
    nv = 0
    for k in range(max_v):
        nd = int(elem_nodes[elem_id, k])
        if nd < 0:
            continue
        if nv > 0 and nd == verts[0]:
            continue          # FESOM-C tri stored as quad repeats vertex 0
        verts[nv] = nd
        nv += 1
    if nv < 3:
        return 0.0, 0.0, False
    DEG2R = 0.017453292519943295
    R = r_earth
    lon_ref = nodes_lon[verts[0]]
    dsum = 0.0; lat0 = 0.0
    for k in range(nv):
        dsum += ((nodes_lon[verts[k]] - lon_ref + 180.0) % 360.0) - 180.0
        lat0 += nodes_lat[verts[k]]
    lon0 = lon_ref + dsum / nv
    lat0 /= nv
    cosl = np.cos(lat0 * DEG2R)
    vx = np.empty(nv); vy = np.empty(nv)
    for k in range(nv):
        dlon = ((nodes_lon[verts[k]] - lon0 + 180.0) % 360.0) - 180.0
        vx[k] = dlon * DEG2R * R * cosl
        vy[k] = (nodes_lat[verts[k]] - lat0) * DEG2R * R
    dlon_p = ((xp - lon0 + 180.0) % 360.0) - 180.0
    px = dlon_p * DEG2R * R * cosl
    py = (yp - lat0) * DEG2R * R
    w = _polygon_weights(px, py, vx, vy, nv)
    if w[0] == -1.0:
        return 0.0, 0.0, False
    zq = z
    if np.isnan(zq):
        zq = np.inf           # dry wherever the bed is finite
    # may set the wet weights to 1.0 in w (the equal-weight mean)
    n_wet, den, k_deep = _dl_node_wall(w, verts, nv, bed, zq)
    if n_wet == 0:
        nd = verts[k_deep]    # below every bed: the deepest vertex
        un, vn = _tinterp_point_uv(u_a, v_a, u_b, v_b, u_t, v_t, last[nd],
                                   nd, alpha0, alpha)
        return time_dir * un, time_dir * vn, True
    U = 0.0; V = 0.0
    for k in range(nv):
        nd = verts[k]
        if n_wet < nv and not uv_wet(bed, nd, zq):
            continue          # a wall at this depth: left out
        kb, ab = uv_bracket(p, last[nd], zq)
        un, vn = _tinterp_point_uv(u_a, v_a, u_b, v_b, u_t, v_t, kb, nd,
                                   alpha0, alpha)
        if not ab < 0.0:      # uv_at: alpha < 0 reads point kb only
            un1, vn1 = _tinterp_point_uv(u_a, v_a, u_b, v_b, u_t, v_t,
                                         kb + 1, nd, alpha0, alpha)
            un = (1.0 - ab) * un + ab * un1
            vn = (1.0 - ab) * vn + ab * vn1
        U += w[k] * un; V += w[k] * vn
    if n_wet < nv:
        U = U / den; V = V / den
    return time_dir * U, time_dir * V, True


@njit(cache=True)
def _interp_barycentric_core_pchip4_dl(xp, yp, nn_node, z, time_dir,
                                       node_to_elems, node_to_elems_count,
                                       elem_nodes, nodes_lon, nodes_lat,
                                       p, last, bed,
                                       u_0, v_0, u_1, v_1, u_2, v_2,
                                       u_3, v_3, t0, t1, t2, t3, tq,
                                       r_earth):
    """Depth-linear twin of :func:`_interp_barycentric_core_pchip4_3d`
    (node layout, ``uv_vertical`` ``"level"`` / ``"layer_linear"``).

    Identical geometry; ``layer`` is replaced by the particle depth ``z``
    and the table ``p``, ``last``, ``bed``; u/v snapshots are ``(M, N)``.
    Per vertex, TIME FIRST: each stored value ``(k, nd)`` the vertex reads
    goes through ``_pchip_value_scalar`` (the unfused ``reconstruct_pchip``
    arithmetic), THEN the ``uv_bracket`` / ``uv_at`` rule runs on the
    reconstructed values.  PCHIP is not linear, so the order matters (the
    vertical-first result differs).  A point below ``last[nd]`` is never
    read; the walls at depth of :func:`_interp_barycentric_core_dl`
    (:func:`_dl_node_wall`) decide which vertices enter and how they are
    weighted.  Returns ``(U, V, ok)``."""
    elem_id, in_dom = _find_containing_element_local(
        xp, yp, nn_node, node_to_elems, node_to_elems_count,
        elem_nodes, nodes_lon, nodes_lat)
    if not in_dom:
        return 0.0, 0.0, False
    max_v = elem_nodes.shape[1]
    verts = np.empty(max_v, dtype=np.int64)
    nv = 0
    for k in range(max_v):
        nd = int(elem_nodes[elem_id, k])
        if nd < 0:
            continue
        if nv > 0 and nd == verts[0]:
            continue          # FESOM-C tri stored as quad repeats vertex 0
        verts[nv] = nd
        nv += 1
    if nv < 3:
        return 0.0, 0.0, False
    DEG2R = 0.017453292519943295
    R = r_earth
    lon_ref = nodes_lon[verts[0]]
    dsum = 0.0; lat0 = 0.0
    for k in range(nv):
        dsum += ((nodes_lon[verts[k]] - lon_ref + 180.0) % 360.0) - 180.0
        lat0 += nodes_lat[verts[k]]
    lon0 = lon_ref + dsum / nv
    lat0 /= nv
    cosl = np.cos(lat0 * DEG2R)
    vx = np.empty(nv); vy = np.empty(nv)
    for k in range(nv):
        dlon = ((nodes_lon[verts[k]] - lon0 + 180.0) % 360.0) - 180.0
        vx[k] = dlon * DEG2R * R * cosl
        vy[k] = (nodes_lat[verts[k]] - lat0) * DEG2R * R
    dlon_p = ((xp - lon0 + 180.0) % 360.0) - 180.0
    px = dlon_p * DEG2R * R * cosl
    py = (yp - lat0) * DEG2R * R
    w = _polygon_weights(px, py, vx, vy, nv)
    if w[0] == -1.0:
        return 0.0, 0.0, False
    zq = z
    if np.isnan(zq):
        zq = np.inf           # dry wherever the bed is finite
    # may set the wet weights to 1.0 in w (the equal-weight mean)
    n_wet, den, k_deep = _dl_node_wall(w, verts, nv, bed, zq)
    if n_wet == 0:
        nd = verts[k_deep]    # below every bed: the deepest vertex
        kl = last[nd]
        un = _pchip_value_scalar(u_0[kl, nd], u_1[kl, nd],
                                 u_2[kl, nd], u_3[kl, nd],
                                 t0, t1, t2, t3, tq)
        vn = _pchip_value_scalar(v_0[kl, nd], v_1[kl, nd],
                                 v_2[kl, nd], v_3[kl, nd],
                                 t0, t1, t2, t3, tq)
        return time_dir * un, time_dir * vn, True
    U = 0.0; V = 0.0
    for k in range(nv):
        nd = verts[k]
        if n_wet < nv and not uv_wet(bed, nd, zq):
            continue          # a wall at this depth: left out
        kb, ab = uv_bracket(p, last[nd], zq)
        un = _pchip_value_scalar(u_0[kb, nd], u_1[kb, nd],
                                 u_2[kb, nd], u_3[kb, nd],
                                 t0, t1, t2, t3, tq)
        vn = _pchip_value_scalar(v_0[kb, nd], v_1[kb, nd],
                                 v_2[kb, nd], v_3[kb, nd],
                                 t0, t1, t2, t3, tq)
        if not ab < 0.0:      # uv_at: alpha < 0 reads point kb only
            kn = kb + 1
            un1 = _pchip_value_scalar(u_0[kn, nd], u_1[kn, nd],
                                      u_2[kn, nd], u_3[kn, nd],
                                      t0, t1, t2, t3, tq)
            vn1 = _pchip_value_scalar(v_0[kn, nd], v_1[kn, nd],
                                      v_2[kn, nd], v_3[kn, nd],
                                      t0, t1, t2, t3, tq)
            un = (1.0 - ab) * un + ab * un1
            vn = (1.0 - ab) * vn + ab * vn1
        U += w[k] * un; V += w[k] * vn
    if n_wet < nv:
        U = U / den; V = V / den
    return time_dir * U, time_dir * V, True


# ---------------------------------------------------------------------------
# Depth-linear twin (uv_vertical "layer_linear", element layout): the
# geometry of _interp_barycentric_core, each vertex value the area-weighted
# mean of the ring elements wet at the particle depth (_dl_elem_wet_mean);
# a vertex without one is a wall, left out like a dry vertex of the node
# layout.
# ---------------------------------------------------------------------------

@njit(cache=True)
def _interp_bary_elem_core_dl(xp, yp, z, nn_node, elem_hint, time_dir,
                              node_to_elems, node_to_elems_count,
                              elem_nodes, nodes_lon, nodes_lat,
                              p, last, bed, elem_areas, u_f, v_f, r_earth):
    """Depth-linear vertex-barycentric u/v at one point (element layout).

    Containment search (1-ring, then the 2-ring fallback of
    ``_find_containing_element_local``), the ``elem_hint`` shortcut, the
    compacted vertex ring, the centroid-anchored metric projection and
    ``_polygon_weights`` are those of :func:`_interp_barycentric_core`.
    The vertex value differs: at vertex ``nd``, the area-weighted mean of
    the elements around it (``node_to_elems``) that are wet at ``z``, each
    read linearly in depth between its own layer centres
    (``uv_bracket(p, last[e], z)``, ``uv_at``; a point below ``last[e]``
    is never read) — :func:`_dl_elem_wet_mean`, the summation of the layer
    kernel's vertex mean.

    Walls at depth: an element whose bed lies above ``z``
    (``z > bed[e]``) is dry and left out of each vertex mean, as elements
    beyond a coast are absent from a coastal vertex's ring.  A vertex with
    no wet element is left out and the weights of the others renormalised,
    a negative one (a particle just outside an edge) counted as zero
    (their equal-weight mean when those weights sum below 1e-12: the
    particle on that vertex).  Every element wet: the layer kernel's
    arithmetic; every vertex with a wet element: the plain barycentric
    sum (a negative weight kept).  No vertex left (the particle below
    every bed around the element): the containing element, dry too, at its
    last usable point.  A vertex whose wet elements' areas sum to zero or
    less gives ``ok = False``, as in the layer kernel.

    ``z`` is the particle depth (m, positive down, as ``state.z``; on a
    sigma mesh the sigma fraction ``s``, the table's coordinate).  A NaN
    ``z`` is read as ``+inf``: dry wherever the bed is finite.

    ``p`` (``(M,)``), ``last`` and ``bed`` (``(E,)``, positive down) are
    the element table of :func:`kelp.numerics.vertical_uv.depth_points`;
    ``elem_areas`` is ``(E,)`` (:func:`kelp.core.mesh.build_elem_areas`);
    ``u_f``/``v_f`` are ``(M, E)``.  Returns ``(U, V, ok)``.
    """
    if elem_hint >= 0:
        elem_id = int(elem_hint)
    else:
        elem_id, in_dom = _find_containing_element_local(
            xp, yp, nn_node, node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat)
        if not in_dom:
            return 0.0, 0.0, False
    max_v = elem_nodes.shape[1]
    verts = np.empty(max_v, dtype=np.int64)
    nv = 0
    for k in range(max_v):
        nd = int(elem_nodes[elem_id, k])
        if nd < 0:
            continue
        if nv > 0 and nd == verts[0]:
            continue          # FESOM-C tri stored as quad repeats vertex 0
        verts[nv] = nd
        nv += 1
    if nv < 3:
        return 0.0, 0.0, False
    DEG2R = 0.017453292519943295
    R = r_earth
    lon_ref = nodes_lon[verts[0]]
    dsum = 0.0; lat0 = 0.0
    for k in range(nv):
        dsum += ((nodes_lon[verts[k]] - lon_ref + 180.0) % 360.0) - 180.0
        lat0 += nodes_lat[verts[k]]
    lon0 = lon_ref + dsum / nv
    lat0 /= nv
    cosl = np.cos(lat0 * DEG2R)
    vx = np.empty(nv); vy = np.empty(nv)
    for k in range(nv):
        dlon = ((nodes_lon[verts[k]] - lon0 + 180.0) % 360.0) - 180.0
        vx[k] = dlon * DEG2R * R * cosl
        vy[k] = (nodes_lat[verts[k]] - lat0) * DEG2R * R
    dlon_p = ((xp - lon0 + 180.0) % 360.0) - 180.0
    px = dlon_p * DEG2R * R * cosl
    py = (yp - lat0) * DEG2R * R
    w = _polygon_weights(px, py, vx, vy, nv)
    if w[0] == -1.0:
        return 0.0, 0.0, False
    zq = z
    if np.isnan(zq):
        zq = np.inf           # dry wherever the bed is finite
    # vertex values: the area mean of the wet ring elements (time_dir 1.0:
    # the layer kernel applies time_dir to the sum)
    un = np.empty(nv); vn = np.empty(nv)
    wet = np.zeros(nv, dtype=np.bool_)
    n_wet = 0
    for k in range(nv):
        nd = verts[k]
        cnt = int(node_to_elems_count[nd])
        uk, vk, okk, n_e = _dl_elem_wet_mean(node_to_elems[nd], cnt, zq,
                                             -1, 1.0, p, last, bed,
                                             elem_areas, u_f, v_f)
        if not okk:
            if n_e > 0:
                # wet elements whose areas sum to <= 0: the layer
                # kernel's refusal
                return 0.0, 0.0, False
            continue          # no wet element: a wall at this depth
        un[k] = uk; vn[k] = vk
        wet[k] = True
        n_wet += 1
    if n_wet == 0:
        # below every bed around the element: the containing element (in
        # every vertex ring, so dry too) at its last usable point
        kb, ab = uv_bracket(p, last[elem_id], zq)
        return (time_dir * uv_at(u_f, elem_id, kb, ab),
                time_dir * uv_at(v_f, elem_id, kb, ab), True)
    den = 1.0
    if n_wet < nv:
        # a vertex left out: the weights of the others, a negative one
        # (a particle just outside an edge) counted as zero
        wsum = 0.0
        for k in range(nv):
            if wet[k]:
                if w[k] < 0.0:
                    w[k] = 0.0
                wsum += w[k]
        if wsum < 1e-12:
            # the particle on a left-out vertex: the equal-weight mean
            for k in range(nv):
                if wet[k]:
                    w[k] = 1.0
            den = float(n_wet)
        else:
            den = wsum
    U = 0.0; V = 0.0
    for k in range(nv):
        if not wet[k]:
            continue          # left out
        U += w[k] * un[k]; V += w[k] * vn[k]
    if n_wet < nv:
        U = U / den; V = V / den
    return time_dir * U, time_dir * V, True


@njit(parallel=True, cache=True)
def _loop_interp_bary_elem_dl(
    P: int, alive_mask: np.ndarray, lons: np.ndarray, lats: np.ndarray,
    zs: np.ndarray, nn_nodes: np.ndarray, elems: np.ndarray,
    time_dir: float,
    node_to_elems: np.ndarray, node_to_elems_count: np.ndarray,
    elem_nodes: np.ndarray, nodes_lon: np.ndarray, nodes_lat: np.ndarray,
    p: np.ndarray, last: np.ndarray, bed: np.ndarray,
    elem_areas: np.ndarray, u_f: np.ndarray, v_f: np.ndarray,
    U: np.ndarray, V: np.ndarray, ok: np.ndarray, r_earth: float
):
    """Depth-linear element-layout vertex-barycentric u/v for every alive
    particle (each at its own depth ``zs[i]``); see
    :func:`_interp_bary_elem_core_dl`."""
    for i in prange(P):
        if not alive_mask[i]:
            continue
        u, v, success = _interp_bary_elem_core_dl(
            lons[i], lats[i], zs[i], int(nn_nodes[i]), int(elems[i]),
            time_dir,
            node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat,
            p, last, bed, elem_areas, u_f, v_f, r_earth
        )
        U[i] = u
        V[i] = v
        ok[i] = success


def _interp_barycentric_dl(
    state: ParticleState,
    mesh: Mesh,
    vel: VelocityField,
    nn_nodes: np.ndarray,
    time_dir: float,
    *,
    elems=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Depth-linear vertex-barycentric u/v for all alive particles.

    The ``interp_barycentric`` route for ``vel.uv_vertical`` ``"level"`` /
    ``"layer_linear"``: each particle is read at its depth ``state.z`` (m,
    positive down; on a sigma mesh, which serves ``"layer_linear"`` data
    only, its sigma fraction ``s = z / H_p``, which the steps put in
    ``state.z``); there is no layer argument.  Node layout: a vertex
    whose bed (``mesh.depth``) lies above the particle is a wall and left
    out (:func:`_interp_barycentric_core_dl`).  Element layout
    (``"layer_linear"``): an element whose bed
    (``mesh.metadata["elem_bed"]``) lies above the particle is left out of
    each vertex mean, a vertex without a wet element is left out
    (:func:`_interp_bary_elem_core_dl`).

    Parameters
    ----------
    state : ParticleState
    mesh : Mesh
        A z-level or a sigma mesh; the depth-point table is
        ``depth_points(mesh, vel.uv_vertical, vel.uv_layout)`` (element
        layout on a z-level mesh: with the element bottoms
        ``mesh.metadata["uv_data_last_elem"]``, ``["elem_bed"]``; a sigma
        mesh needs none, every column is wet at every ``s``).
    vel : VelocityField or LazyLerpField
        Node-layout u/v of shape ``(M, N)`` or element-layout u/v of shape
        ``(M, E)``.  Checked once per mesh table with
        :func:`kelp.numerics.vertical_uv.check_depth_linear_record`
        (skipped when :func:`~kelp.numerics.vertical_uv.is_checked`).
    nn_nodes : np.ndarray, shape ``(P,)``
        Nearest-node index per particle.
    time_dir : float
        Forward (+1) or backward (-1).
    elems : np.ndarray or None
        Optional per-particle PRE-VALIDATED containing elements (edge-walk
        tracker); ``-1`` entries fall back to the internal search.

    Returns
    -------
    U, V : np.ndarray, shape ``(P,)``
        Interpolated velocity (m/s).  Zero for inactive particles.
    ok : np.ndarray of bool, shape ``(P,)``
        True where interpolation succeeded.

    Raises
    ------
    ValueError
        A layout other than node or element, u/v not 2-D, a mesh neither
        z-level nor sigma, ``"level"`` data on a sigma mesh, an
        element-layout z-level mesh without the element bottoms, or a
        record that fails the check (shape, non-finite value at a usable
        point).
    """
    placement = getattr(vel, "uv_vertical", "layer")
    layout = vel.uv_layout
    if layout not in ("node", "element"):
        raise ValueError(
            f"interp_barycentric: depth-linear u/v (uv_vertical="
            f"{placement!r}) needs uv_layout 'node' or 'element', got "
            f"{layout!r}")
    # the record check first: a LazyLerpField is checked through its
    # parents (shape and finite prefix) before anything below blends it
    if not is_checked(vel, mesh):
        check_depth_linear_record(vel, mesh)
    # unreachable for a field checked as it stands (the check refuses
    # non-2-D u/v); kept because the token survives an attribute replaced
    # after the check (``vel.u = ...``), which would reach the kernel
    if getattr(vel.u, "ndim", None) != 2 or getattr(vel.v, "ndim", None) != 2:
        raise ValueError(
            f"interp_barycentric: depth-linear u/v (uv_vertical="
            f"{placement!r}) must be 2-D (points, {layout}s), got u "
            f"{getattr(vel.u, 'shape', None)}, v "
            f"{getattr(vel.v, 'shape', None)}")
    dp = depth_points(mesh, placement, layout)

    P = state.n_particles
    U = np.zeros(P, dtype=np.float64)
    V = np.zeros(P, dtype=np.float64)
    ok = np.zeros(P, dtype=bool)

    if elems is None:
        elems_arr = np.full(P, -1, dtype=np.int32)
    else:
        elems_arr = np.asarray(elems, dtype=np.int32)

    u_f = vel.u
    v_f = vel.v
    if layout == "element":
        areas = build_elem_areas(mesh)
        if HAS_NUMBA:
            _loop_interp_bary_elem_dl(
                P, state.alive_mask, state.lon, state.lat, state.z,
                nn_nodes, elems_arr, time_dir,
                mesh.node_to_elems, mesh.node_to_elems_count,
                mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat,
                dp.p, dp.last, dp.bed, areas, u_f, v_f, U, V, ok,
                mesh.r_earth
            )
        else:
            for i in range(P):
                if not state.alive_mask[i]:
                    continue
                u, v, success = _interp_bary_elem_core_dl(
                    state.lon[i], state.lat[i], state.z[i],
                    int(nn_nodes[i]), int(elems_arr[i]), time_dir,
                    mesh.node_to_elems, mesh.node_to_elems_count,
                    mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat,
                    dp.p, dp.last, dp.bed, areas, u_f, v_f, mesh.r_earth
                )
                U[i] = u
                V[i] = v
                ok[i] = success
    elif HAS_NUMBA:
        _loop_interp_bary_dl(
            P, state.alive_mask, state.lon, state.lat, state.z, nn_nodes,
            elems_arr, time_dir,
            mesh.node_to_elems, mesh.node_to_elems_count,
            mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat,
            dp.p, dp.last, dp.bed, u_f, v_f, U, V, ok, mesh.r_earth
        )
    else:
        for i in range(P):
            if not state.alive_mask[i]:
                continue
            u, v, success = _interp_barycentric_core_dl(
                state.lon[i], state.lat[i], state.z[i],
                int(nn_nodes[i]), int(elems_arr[i]), time_dir,
                mesh.node_to_elems, mesh.node_to_elems_count,
                mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat,
                dp.p, dp.last, dp.bed, u_f, v_f, mesh.r_earth
            )
            U[i] = u
            V[i] = v
            ok[i] = success

    return U, V, ok


def interp_barycentric_single(xp, yp, layer, mesh, vel, node_idx,
                              time_dir=1.0, *, elem=-1):
    """Single-point vertex-barycentric u/v (signature of
    interp_leastsq_single).

    Finds the containing element around ``node_idx`` (or uses the
    pre-validated ``elem`` hint when ``>= 0``), builds the vertex
    velocity values (node layout: direct lookup; element layout:
    area-weighted average over the elements sharing each vertex) and blends
    them with ``_polygon_weights``.  Returns ``(U, V, ok)``.
    """
    _reject_depth_linear(vel, "interp_barycentric_single", _DL_NO_SINGLE_POINT)
    if vel.uv_layout == "node":
        layout_code = UV_LAYOUT_NODE
        areas = np.empty(0, dtype=np.float64)
    else:
        layout_code = UV_LAYOUT_ELEMENT
        areas = build_elem_areas(mesh)

    if vel.u.ndim == 1:
        u_fi = vel.u
        v_fi = vel.v
    else:
        u_fi = vel.u[layer, :]
        v_fi = vel.v[layer, :]

    return _interp_barycentric_core(
        xp, yp, node_idx, int(elem), time_dir, layout_code,
        mesh.node_to_elems, mesh.node_to_elems_count,
        mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat,
        areas, u_fi, v_fi, mesh.r_earth
    )


def interp_barycentric(
    state: ParticleState,
    mesh: Mesh,
    vel: VelocityField,
    nn_nodes: np.ndarray,
    time_dir: float = 1.0,
    layer=0,
    *,
    elems=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate velocity for all active particles using vertex-barycentric.

    Locates each particle's containing element, computes vertex weights
    (classical barycentric for triangles, guard-free product-form
    Wachspress for quads) and blends per-vertex velocity values.
    Node-layout fields are read directly at the vertex; element-layout
    fields are area-weighted averages of the elements sharing each vertex.

    Parameters
    ----------
    state : ParticleState
    mesh : Mesh
    vel : VelocityField
    nn_nodes : np.ndarray, shape ``(P,)``
        Nearest-node index per particle.
    time_dir : float
        Forward (+1) or backward (-1).
    layer : int or np.ndarray
        Vertical layer index.  Scalar ``int`` for 2-D (all particles use
        the same layer).  ``np.ndarray`` of shape ``(P,)`` for 3-D
        (per-particle layer).  Ignored for a depth-linear field
        (``vel.uv_vertical`` ``"level"`` / ``"layer_linear"``): that field
        is read at each particle's depth ``state.z``, on a sigma mesh its
        sigma fraction (:func:`_interp_barycentric_dl`).
    elems : np.ndarray or None
        Optional per-particle PRE-VALIDATED containing elements (edge-walk
        tracker); ``-1`` entries fall back to the internal search.  None
        (default) = search for every particle (byte-identical behaviour).

    Returns
    -------
    U, V : np.ndarray, shape ``(P,)``
        Interpolated velocity (m/s).  Zero for inactive particles.
    ok : np.ndarray of bool, shape ``(P,)``
        True where interpolation succeeded.
    """
    if getattr(vel, "uv_vertical", "layer") != "layer":
        return _interp_barycentric_dl(state, mesh, vel, nn_nodes, time_dir,
                                      elems=elems)
    P = state.n_particles
    U = np.zeros(P, dtype=np.float64)
    V = np.zeros(P, dtype=np.float64)
    ok = np.zeros(P, dtype=bool)

    if elems is None:
        elems_arr = np.full(P, -1, dtype=np.int32)
    else:
        elems_arr = np.asarray(elems, dtype=np.int32)

    # Resolve layout + element areas once (areas only needed/indexed for
    # the element layout; the node layout passes an empty array).
    if vel.uv_layout == "node":
        layout_code = UV_LAYOUT_NODE
        areas = np.empty(0, dtype=np.float64)
    else:
        layout_code = UV_LAYOUT_ELEMENT
        areas = build_elem_areas(mesh)

    # Per-particle layer: 3D path
    per_particle = isinstance(layer, np.ndarray)

    if vel.u.ndim == 1:
        # 2D velocity data — single layer, ignore layer param
        u_fi = vel.u
        v_fi = vel.v
        per_particle = False
    elif not per_particle:
        # 2D tracking with 3D data — pre-slice to single layer
        u_fi = vel.u[layer, :]
        v_fi = vel.v[layer, :]

    if per_particle:
        # 3D: each particle uses its own layer
        layers = np.asarray(layer, dtype=np.int32)
        if HAS_NUMBA:
            _loop_interp_bary_3d(
                P, state.alive_mask, state.lon, state.lat, nn_nodes,
                elems_arr, layers, time_dir, layout_code,
                mesh.node_to_elems, mesh.node_to_elems_count,
                mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat,
                areas, vel.u, vel.v, U, V, ok, mesh.r_earth
            )
        else:
            for i in range(P):
                if not state.alive_mask[i]:
                    continue
                u_fi_i = vel.u[layers[i], :]
                v_fi_i = vel.v[layers[i], :]
                u, v, success = _interp_barycentric_core(
                    state.lon[i], state.lat[i],
                    int(nn_nodes[i]), int(elems_arr[i]), time_dir,
                    layout_code,
                    mesh.node_to_elems, mesh.node_to_elems_count,
                    mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat,
                    areas, u_fi_i, v_fi_i, mesh.r_earth
                )
                U[i] = u
                V[i] = v
                ok[i] = success
    else:
        # 2D: all particles use same pre-sliced layer
        if HAS_NUMBA:
            _loop_interp_bary(
                P, state.alive_mask, state.lon, state.lat, nn_nodes,
                elems_arr, time_dir, layout_code,
                mesh.node_to_elems, mesh.node_to_elems_count,
                mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat,
                areas, u_fi, v_fi, U, V, ok, mesh.r_earth
            )
        else:
            for i in range(P):
                if not state.alive_mask[i]:
                    continue
                u, v, success = _interp_barycentric_core(
                    state.lon[i], state.lat[i],
                    int(nn_nodes[i]), int(elems_arr[i]), time_dir,
                    layout_code,
                    mesh.node_to_elems, mesh.node_to_elems_count,
                    mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat,
                    areas, u_fi, v_fi, mesh.r_earth
                )
                U[i] = u
                V[i] = v
                ok[i] = success

    return U, V, ok


# ---------------------------------------------------------------------------
# Edge-neighbor least-squares u/v (FESOM-style elemental interpolation):
# LSQ plane fit over the containing element and the elements sharing an
# edge with it (3 for a triangle, 4 for a quad).  Boundary edges get a
# ghost point — the element centroid mirrored across the edge — carrying
# the lateral boundary condition: freeslip (the mirror image of u_own —
# tangential kept, normal reversed) or noslip (-u_own, fitted velocity ~0
# at the coast edge).
# ---------------------------------------------------------------------------
BC_FREESLIP = 0
BC_NOSLIP = 1
_EDGE_LSQ_BC_CODES = {"freeslip": BC_FREESLIP, "noslip": BC_NOSLIP}


def _bc_to_int(bc: str) -> int:
    """Map an edge_leastsq_bc name to its Numba dispatch code."""
    try:
        return _EDGE_LSQ_BC_CODES[bc]
    except KeyError:
        raise ValueError(
            f"unknown edge_leastsq_bc {bc!r}; valid: "
            f"{sorted(_EDGE_LSQ_BC_CODES)}") from None


@njit(cache=True)
def _interp_edge_leastsq_core(
    xp: float,
    yp: float,
    nn_node: int,
    elem_hint: int,
    time_dir: float,
    bc_code: int,
    node_to_elems: np.ndarray,
    node_to_elems_count: np.ndarray,
    elem_nodes: np.ndarray,
    nodes_lon: np.ndarray,
    nodes_lat: np.ndarray,
    elem_neighbors: np.ndarray,
    elem_center_lon: np.ndarray,
    elem_center_lat: np.ndarray,
    u_f: np.ndarray,
    v_f: np.ndarray,
) -> tuple[float, float, bool]:
    """Edge-neighbor LSQ u/v at one point (element-layout fields only).

    Finds the containing element around ``nn_node`` (or uses the
    PRE-VALIDATED ``elem_hint`` when ``>= 0`` — edge-walk tracker), then
    fits the plane ``f = a0 + a1*x + a2*y`` through the element-centre
    velocities of the containing element and its edge-sharing neighbors
    (``build_elem_neighbors``).  A boundary edge contributes a ghost
    point instead: the containing element's centroid mirrored across
    that edge, with value the mirror image of ``u_own`` (freeslip:
    tangential component kept, normal reversed) or ``-u_own`` (noslip).

    Coordinates are metric and CENTERED on the particle (the
    anti-cancellation idiom of ``_interp_leastsq_core``) with WRAPPED
    longitude deltas (dateline-safe).  Returns ``(U, V, ok)``.
    """
    if elem_hint >= 0:
        elem_id = int(elem_hint)
    else:
        elem_id, in_dom = _find_containing_element_local(
            xp, yp, nn_node, node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat)
        if not in_dom:
            return 0.0, 0.0, False

    # Compacted vertex ring — same rule as _interp_barycentric_core and
    # build_elem_neighbors (skip -1 pads and repeats of the first vertex).
    max_v = elem_nodes.shape[1]
    verts = np.empty(max_v, dtype=np.int64)
    nv = 0
    for k in range(max_v):
        nd = int(elem_nodes[elem_id, k])
        if nd < 0:
            continue
        if nv > 0 and nd == verts[0]:
            continue
        verts[nv] = nd
        nv += 1
    if nv < 3:
        return 0.0, 0.0, False

    z0 = np.cos(yp * DEG2RAD) * METERS_PER_DEG_LAT

    # Own centroid in particle-centered metric metres (wrapped dlon).
    dlon_c = ((elem_center_lon[elem_id] - xp + 180.0) % 360.0) - 180.0
    cx = dlon_c * z0
    cy = (elem_center_lat[elem_id] - yp) * METERS_PER_DEG_LAT
    u_own = time_dir * u_f[elem_id]
    v_own = time_dir * v_f[elem_id]

    # Normal system seeded with the own-centroid point.
    a11 = 1.0
    a12 = cx
    a13 = cy
    a22 = cx * cx
    a23 = cx * cy
    a33 = cy * cy
    fUn = u_own
    f1Un = u_own * cx
    f2Un = u_own * cy
    fVn = v_own
    f1Vn = v_own * cx
    f2Vn = v_own * cy

    for k in range(nv):
        nb = int(elem_neighbors[elem_id, k])
        if nb >= 0:
            dlon_n = ((elem_center_lon[nb] - xp + 180.0) % 360.0) - 180.0
            ex = dlon_n * z0
            ey = (elem_center_lat[nb] - yp) * METERS_PER_DEG_LAT
            uu = time_dir * u_f[nb]
            vv = time_dir * v_f[nb]
        else:
            # Ghost: mirror the own centroid across boundary edge k
            # (verts[k] -> verts[(k+1) % nv]), in the same metric plane.
            na = verts[k]
            nb2 = verts[(k + 1) % nv]
            dax = (((nodes_lon[na] - xp + 180.0) % 360.0) - 180.0) * z0
            day = (nodes_lat[na] - yp) * METERS_PER_DEG_LAT
            dbx = (((nodes_lon[nb2] - xp + 180.0) % 360.0) - 180.0) * z0
            dby = (nodes_lat[nb2] - yp) * METERS_PER_DEG_LAT
            tx = dbx - dax
            ty = dby - day
            tt = tx * tx + ty * ty
            if tt <= 0.0:
                continue          # degenerate edge: drop the point
            proj = ((cx - dax) * tx + (cy - day) * ty) / tt
            fx = dax + proj * tx
            fy = day + proj * ty
            ex = 2.0 * fx - cx
            ey = 2.0 * fy - cy
            if bc_code == BC_NOSLIP:
                uu = -u_own
                vv = -v_own
            else:
                # free slip: the mirror image across the wall — tangential
                # component kept, normal component reversed.  Copying the
                # whole vector (before 2026-09-23) let the flow cross the
                # wall (a zero-gradient condition, review 2026-09-22).
                # (tx, ty) is the edge tangent in the same metric plane as
                # the east/north velocity components.
                ut = (u_own * tx + v_own * ty) / tt
                uu = 2.0 * ut * tx - u_own
                vv = 2.0 * ut * ty - v_own
        a11 = a11 + 1.0
        a12 = a12 + ex
        a13 = a13 + ey
        a22 = a22 + ex * ex
        a23 = a23 + ex * ey
        a33 = a33 + ey * ey
        fUn = fUn + uu
        f1Un = f1Un + uu * ex
        f2Un = f2Un + uu * ey
        fVn = fVn + vv
        f1Vn = f1Vn + vv * ex
        f2Vn = f2Vn + vv * ey

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
        return 0.0, 0.0, False

    inv = 1.0 / abs(det)

    b11 = a22 * a33 - a23 * a32
    b12 = a13 * a32 - a12 * a33
    b13 = a12 * a23 - a13 * a22

    a0U = (b11 * fUn + b12 * f1Un + b13 * f2Un) * inv
    a0V = (b11 * fVn + b12 * f1Vn + b13 * f2Vn) * inv

    return a0U, a0V, True


@njit(parallel=True, cache=True)
def _loop_interp_edge_lsq(
    P: int, alive_mask: np.ndarray, lons: np.ndarray, lats: np.ndarray,
    nn_nodes: np.ndarray, elems: np.ndarray, time_dir: float, bc_code: int,
    node_to_elems: np.ndarray, node_to_elems_count: np.ndarray,
    elem_nodes: np.ndarray, nodes_lon: np.ndarray, nodes_lat: np.ndarray,
    elem_neighbors: np.ndarray,
    elem_center_lon: np.ndarray, elem_center_lat: np.ndarray,
    u_fi: np.ndarray, v_fi: np.ndarray,
    U: np.ndarray, V: np.ndarray, ok: np.ndarray,
):
    for i in prange(P):
        if not alive_mask[i]:
            continue
        u, v, success = _interp_edge_leastsq_core(
            lons[i], lats[i], int(nn_nodes[i]), int(elems[i]), time_dir,
            bc_code,
            node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat,
            elem_neighbors, elem_center_lon, elem_center_lat,
            u_fi, v_fi,
        )
        U[i] = u
        V[i] = v
        ok[i] = success


@njit(parallel=True, cache=True)
def _loop_interp_edge_lsq_3d(
    P: int, alive_mask: np.ndarray, lons: np.ndarray, lats: np.ndarray,
    nn_nodes: np.ndarray, elems: np.ndarray, layers: np.ndarray,
    time_dir: float, bc_code: int,
    node_to_elems: np.ndarray, node_to_elems_count: np.ndarray,
    elem_nodes: np.ndarray, nodes_lon: np.ndarray, nodes_lat: np.ndarray,
    elem_neighbors: np.ndarray,
    elem_center_lon: np.ndarray, elem_center_lat: np.ndarray,
    u_2d: np.ndarray, v_2d: np.ndarray,
    U: np.ndarray, V: np.ndarray, ok: np.ndarray,
):
    """Per-particle layer edge-neighbor LSQ for 3D tracking."""
    for i in prange(P):
        if not alive_mask[i]:
            continue
        layer_i = layers[i]
        u_fi = u_2d[layer_i]
        v_fi = v_2d[layer_i]
        u, v, success = _interp_edge_leastsq_core(
            lons[i], lats[i], int(nn_nodes[i]), int(elems[i]), time_dir,
            bc_code,
            node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat,
            elem_neighbors, elem_center_lon, elem_center_lat,
            u_fi, v_fi,
        )
        U[i] = u
        V[i] = v
        ok[i] = success


# ---------------------------------------------------------------------------
# Depth-linear twin (uv_vertical "layer_linear", element layout): the fit of
# _interp_edge_leastsq_core on each element's value at the particle depth;
# an edge neighbour dry at that depth is a wall — its edge a boundary edge
# with the bc's ghost, exactly as at the coast.
# ---------------------------------------------------------------------------

@njit(cache=True)
def _interp_edge_leastsq_core_dl(
    xp: float,
    yp: float,
    z: float,
    nn_node: int,
    elem_hint: int,
    time_dir: float,
    bc_code: int,
    node_to_elems: np.ndarray,
    node_to_elems_count: np.ndarray,
    elem_nodes: np.ndarray,
    nodes_lon: np.ndarray,
    nodes_lat: np.ndarray,
    elem_neighbors: np.ndarray,
    elem_center_lon: np.ndarray,
    elem_center_lat: np.ndarray,
    p: np.ndarray,
    last: np.ndarray,
    bed: np.ndarray,
    elem_areas: np.ndarray,
    u_f: np.ndarray,
    v_f: np.ndarray,
) -> tuple[float, float, bool]:
    """Depth-linear edge-neighbor LSQ u/v at one point (element layout).

    The fit of :func:`_interp_edge_leastsq_core` — the containing element
    (``_find_containing_element_local``, or the PRE-VALIDATED ``elem_hint``
    when ``>= 0``), the compacted vertex ring, particle-centred metric
    coordinates with wrapped longitude deltas, the own centroid first and
    then edge by edge a neighbour or a boundary ghost, the summation
    order, Sarrus and the scale-relative singular guard — on each
    element's depth-linear value at ``z``:
    ``(k, alpha) = uv_bracket(p, last[e], z)``,
    ``u = uv_at(u_f, e, k, alpha)``, linear in depth between the layer
    centres ``p[k]``, ``p[k + 1]``, clipped to ``p[0] .. p[last[e]]``; a
    point below ``last[e]`` is never read.

    Walls at depth: an edge neighbour whose bed lies above ``z``
    (``z > bed[e]``) is dry and its edge a boundary edge — the ghost (the
    own centroid mirrored across the edge) takes the own element's value
    at ``z``, mirrored (``BC_FREESLIP``) or negated (``BC_NOSLIP``), the
    coastal construction.  Every member wet: the layer core's arithmetic,
    so a singular all-wet fit stays ``ok = False``.  A wall and a singular
    fit: the area-weighted mean of the wet members (the own element, then
    its wet neighbours in slot order; :func:`_dl_elem_wet_mean`).  The own
    element dry: the area-weighted mean of its wet edge neighbours, no
    ghosts; none wet, the own element at its last usable point.  A wet
    element below ``p[last[e]]`` (a partial bottom layer, a ``max_levels``
    cut, a declared prefix) reads its last usable point.

    ``z`` is the particle depth (m, positive down, as ``state.z``; on a
    sigma mesh the sigma fraction ``s``, the table's coordinate).  A NaN
    ``z`` is read as ``+inf``: dry wherever the bed is finite.

    ``p`` (``(M,)``), ``last`` and ``bed`` (``(E,)``, positive down) are
    the element table of :func:`kelp.numerics.vertical_uv.depth_points`;
    ``elem_areas`` is ``(E,)`` (:func:`kelp.core.mesh.build_elem_areas`);
    ``u_f``/``v_f`` are ``(M, E)``.  Returns ``(U, V, ok)``.
    """
    if elem_hint >= 0:
        elem_id = int(elem_hint)
    else:
        elem_id, in_dom = _find_containing_element_local(
            xp, yp, nn_node, node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat)
        if not in_dom:
            return 0.0, 0.0, False

    # Compacted vertex ring — same rule as _interp_edge_leastsq_core
    max_v = elem_nodes.shape[1]
    verts = np.empty(max_v, dtype=np.int64)
    nv = 0
    for k in range(max_v):
        nd = int(elem_nodes[elem_id, k])
        if nd < 0:
            continue
        if nv > 0 and nd == verts[0]:
            continue
        verts[nv] = nd
        nv += 1
    if nv < 3:
        return 0.0, 0.0, False

    zq = z
    if np.isnan(zq):
        zq = np.inf           # dry wherever the bed is finite
    if not uv_wet(bed, elem_id, zq):
        # the own element dry: its wet edge neighbours' area mean, no
        # ghosts; none wet, the own element at its last usable point
        U, V, ok, _ = _dl_elem_wet_mean(elem_neighbors[elem_id], nv, zq,
                                        elem_id, time_dir, p, last, bed,
                                        elem_areas, u_f, v_f)
        return U, V, ok

    # -- the layer core's fit, a dry neighbour's edge a boundary edge --
    z0 = np.cos(yp * DEG2RAD) * METERS_PER_DEG_LAT

    dlon_c = ((elem_center_lon[elem_id] - xp + 180.0) % 360.0) - 180.0
    cx = dlon_c * z0
    cy = (elem_center_lat[elem_id] - yp) * METERS_PER_DEG_LAT
    kb, ab = uv_bracket(p, last[elem_id], zq)
    u_own = time_dir * uv_at(u_f, elem_id, kb, ab)
    v_own = time_dir * uv_at(v_f, elem_id, kb, ab)

    a11 = 1.0
    a12 = cx
    a13 = cy
    a22 = cx * cx
    a23 = cx * cy
    a33 = cy * cy
    fUn = u_own
    f1Un = u_own * cx
    f2Un = u_own * cy
    fVn = v_own
    f1Vn = v_own * cx
    f2Vn = v_own * cy

    n_dry = 0
    for k in range(nv):
        nb = int(elem_neighbors[elem_id, k])
        if nb >= 0 and not uv_wet(bed, nb, zq):
            nb = -1           # a wall at this depth: a boundary edge
            n_dry += 1
        if nb >= 0:
            dlon_n = ((elem_center_lon[nb] - xp + 180.0) % 360.0) - 180.0
            ex = dlon_n * z0
            ey = (elem_center_lat[nb] - yp) * METERS_PER_DEG_LAT
            kb, ab = uv_bracket(p, last[nb], zq)
            uu = time_dir * uv_at(u_f, nb, kb, ab)
            vv = time_dir * uv_at(v_f, nb, kb, ab)
        else:
            # the ghost of _interp_edge_leastsq_core: the own centroid
            # mirrored across edge k (verts[k] -> verts[(k+1) % nv])
            na = verts[k]
            nb2 = verts[(k + 1) % nv]
            dax = (((nodes_lon[na] - xp + 180.0) % 360.0) - 180.0) * z0
            day = (nodes_lat[na] - yp) * METERS_PER_DEG_LAT
            dbx = (((nodes_lon[nb2] - xp + 180.0) % 360.0) - 180.0) * z0
            dby = (nodes_lat[nb2] - yp) * METERS_PER_DEG_LAT
            tx = dbx - dax
            ty = dby - day
            tt = tx * tx + ty * ty
            if tt <= 0.0:
                continue          # degenerate edge: drop the point
            proj = ((cx - dax) * tx + (cy - day) * ty) / tt
            fx = dax + proj * tx
            fy = day + proj * ty
            ex = 2.0 * fx - cx
            ey = 2.0 * fy - cy
            if bc_code == BC_NOSLIP:
                uu = -u_own
                vv = -v_own
            else:
                # free slip: tangential component kept, normal reversed
                ut = (u_own * tx + v_own * ty) / tt
                uu = 2.0 * ut * tx - u_own
                vv = 2.0 * ut * ty - v_own
        a11 = a11 + 1.0
        a12 = a12 + ex
        a13 = a13 + ey
        a22 = a22 + ex * ex
        a23 = a23 + ex * ey
        a33 = a33 + ey * ey
        fUn = fUn + uu
        f1Un = f1Un + uu * ex
        f2Un = f2Un + uu * ey
        fVn = fVn + vv
        f1Vn = f1Vn + vv * ex
        f2Vn = f2Vn + vv * ey

    a21 = a12
    a31 = a13
    a32 = a23

    det = (a11 * a22 * a33
           + a12 * a23 * a31
           + a13 * a21 * a32
           - a13 * a22 * a31
           - a12 * a21 * a33
           - a11 * a23 * a32)

    # the scale-relative singular guard of _interp_edge_leastsq_core
    if abs(det) < 1e-30 or abs(det) < 1e-12 * abs(a11 * a22 * a33):
        if n_dry == 0:
            return 0.0, 0.0, False
        # a wall left a singular fit: the wet members' area mean
        members = np.empty(nv + 1, dtype=np.int64)
        members[0] = elem_id
        for k in range(nv):
            members[k + 1] = elem_neighbors[elem_id, k]
        U, V, ok, _ = _dl_elem_wet_mean(members, nv + 1, zq, -1, time_dir,
                                        p, last, bed, elem_areas, u_f, v_f)
        return U, V, ok

    inv = 1.0 / abs(det)

    b11 = a22 * a33 - a23 * a32
    b12 = a13 * a32 - a12 * a33
    b13 = a12 * a23 - a13 * a22

    a0U = (b11 * fUn + b12 * f1Un + b13 * f2Un) * inv
    a0V = (b11 * fVn + b12 * f1Vn + b13 * f2Vn) * inv

    return a0U, a0V, True


@njit(parallel=True, cache=True)
def _loop_interp_edge_lsq_3d_dl(
    P: int, alive_mask: np.ndarray, lons: np.ndarray, lats: np.ndarray,
    zs: np.ndarray, nn_nodes: np.ndarray, elems: np.ndarray,
    time_dir: float, bc_code: int,
    node_to_elems: np.ndarray, node_to_elems_count: np.ndarray,
    elem_nodes: np.ndarray, nodes_lon: np.ndarray, nodes_lat: np.ndarray,
    elem_neighbors: np.ndarray,
    elem_center_lon: np.ndarray, elem_center_lat: np.ndarray,
    p: np.ndarray, last: np.ndarray, bed: np.ndarray,
    elem_areas: np.ndarray, u_f: np.ndarray, v_f: np.ndarray,
    U: np.ndarray, V: np.ndarray, ok: np.ndarray,
):
    """Depth-linear edge-neighbor LSQ u/v for every alive particle (each at
    its own depth ``zs[i]``); see :func:`_interp_edge_leastsq_core_dl`."""
    for i in prange(P):
        if not alive_mask[i]:
            continue
        u, v, success = _interp_edge_leastsq_core_dl(
            lons[i], lats[i], zs[i], int(nn_nodes[i]), int(elems[i]),
            time_dir, bc_code,
            node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat,
            elem_neighbors, elem_center_lon, elem_center_lat,
            p, last, bed, elem_areas, u_f, v_f,
        )
        U[i] = u
        V[i] = v
        ok[i] = success


def _interp_edge_leastsq_dl(
    state: ParticleState,
    mesh: Mesh,
    vel: VelocityField,
    nn_nodes: np.ndarray,
    time_dir: float,
    *,
    bc: str = "freeslip",
    elems=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Depth-linear edge-neighbor LSQ u/v for all alive particles.

    The ``interp_edge_leastsq`` route for ``vel.uv_vertical``
    ``"layer_linear"`` on element layout: each particle is read at its
    depth ``state.z`` (m, positive down; on a sigma mesh its sigma fraction
    ``s = z / H_p``, which the steps put in ``state.z``); there is no layer
    argument.  An edge neighbour whose bed (``mesh.metadata["elem_bed"]``)
    lies above the particle is a wall, its edge a boundary edge with the
    ``bc`` ghost (:func:`_interp_edge_leastsq_core_dl`).

    Parameters
    ----------
    state : ParticleState
    mesh : Mesh
        A z-level mesh with the element bottoms
        (``mesh.metadata["uv_data_last_elem"]``, ``["elem_bed"]``), or a
        sigma mesh (no bottoms: every element is wet at every ``s``); the
        table is ``depth_points(mesh, vel.uv_vertical, "element")``.
    vel : VelocityField or LazyLerpField
        Element-layout u/v of shape ``(M, E)``.  Checked once per mesh
        table with :func:`kelp.numerics.vertical_uv.check_depth_linear_record`
        (skipped when :func:`~kelp.numerics.vertical_uv.is_checked`).
    nn_nodes : np.ndarray, shape ``(P,)``
        Nearest-node index per particle (containment search anchor).
    time_dir : float
        Forward (+1) or backward (-1).
    bc : str
        Boundary ghost value at coastal edges and at walls:
        ``"freeslip"`` or ``"noslip"`` (:func:`interp_edge_leastsq`).
    elems : np.ndarray or None
        Optional per-particle PRE-VALIDATED containing elements (edge-walk
        tracker); ``-1`` entries fall back to the internal search.

    Returns
    -------
    U, V : np.ndarray, shape ``(P,)``
        Interpolated velocity (m/s).  Zero for inactive particles.
    ok : np.ndarray of bool, shape ``(P,)``
        True where interpolation succeeded.

    Raises
    ------
    ValueError
        Node layout (``uv_interp='barycentric'`` evaluates it), an unknown
        ``bc``, u/v not 2-D, a mesh neither z-level nor sigma, a z-level
        mesh without the element bottoms, or a record that fails the check
        (shape, non-finite value at a usable point).
    """
    placement = getattr(vel, "uv_vertical", "layer")
    if getattr(vel, "uv_layout", "element") != "element":
        _reject_depth_linear(vel, "interp_edge_leastsq", _DL_LAYOUT)
    bc_code = _bc_to_int(bc)
    # the record check first: a LazyLerpField is checked through its
    # parents (shape and finite prefix) before anything below blends it
    if not is_checked(vel, mesh):
        check_depth_linear_record(vel, mesh)
    # unreachable for a field checked as it stands (the check refuses
    # non-2-D u/v); kept because the token survives an attribute replaced
    # after the check (``vel.u = ...``), which would reach the kernel
    if getattr(vel.u, "ndim", None) != 2 or getattr(vel.v, "ndim", None) != 2:
        raise ValueError(
            f"interp_edge_leastsq: depth-linear u/v (uv_vertical="
            f"{placement!r}) must be 2-D (points, elements), got u "
            f"{getattr(vel.u, 'shape', None)}, v "
            f"{getattr(vel.v, 'shape', None)}")
    dp = depth_points(mesh, placement, "element")
    areas = build_elem_areas(mesh)
    neighbors = build_elem_neighbors(mesh)

    P = state.n_particles
    U = np.zeros(P, dtype=np.float64)
    V = np.zeros(P, dtype=np.float64)
    ok = np.zeros(P, dtype=bool)

    if elems is None:
        elems_arr = np.full(P, -1, dtype=np.int32)
    else:
        elems_arr = np.asarray(elems, dtype=np.int32)

    u_f = vel.u
    v_f = vel.v
    if HAS_NUMBA:
        _loop_interp_edge_lsq_3d_dl(
            P, state.alive_mask, state.lon, state.lat, state.z, nn_nodes,
            elems_arr, time_dir, bc_code,
            mesh.node_to_elems, mesh.node_to_elems_count,
            mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat,
            neighbors, mesh.elem_center_lon, mesh.elem_center_lat,
            dp.p, dp.last, dp.bed, areas, u_f, v_f, U, V, ok,
        )
    else:
        for i in range(P):
            if not state.alive_mask[i]:
                continue
            u, v, success = _interp_edge_leastsq_core_dl(
                state.lon[i], state.lat[i], state.z[i], int(nn_nodes[i]),
                int(elems_arr[i]), time_dir, bc_code,
                mesh.node_to_elems, mesh.node_to_elems_count,
                mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat,
                neighbors, mesh.elem_center_lon, mesh.elem_center_lat,
                dp.p, dp.last, dp.bed, areas, u_f, v_f,
            )
            U[i] = u
            V[i] = v
            ok[i] = success

    return U, V, ok


def _reject_node_layout_edge_lsq(vel: VelocityField) -> None:
    if getattr(vel, "uv_layout", "element") == "node":
        raise ValueError(
            "edge_leastsq interpolates element-centre velocities and "
            "cannot consume node-based fields (uv_layout='node'); use "
            "uv_interp: 'barycentric' for node velocities.")


def interp_edge_leastsq_single(
    xp: float,
    yp: float,
    layer: int,
    mesh: Mesh,
    vel: VelocityField,
    node_idx: int,
    time_dir: float = 1.0,
    *,
    bc: str = "freeslip",
    elem: int = -1,
) -> tuple[float, float, bool]:
    """Single-point edge-neighbor LSQ u/v (signature of
    interp_leastsq_single, plus the boundary-condition keyword and the
    optional pre-validated ``elem`` hint)."""
    _reject_depth_linear(vel, "interp_edge_leastsq_single",
                         _DL_NO_SINGLE_POINT)
    _reject_node_layout_edge_lsq(vel)
    bc_code = _bc_to_int(bc)
    neighbors = build_elem_neighbors(mesh)

    if vel.u.ndim == 1:
        u_fi = vel.u
        v_fi = vel.v
    else:
        u_fi = vel.u[layer, :]
        v_fi = vel.v[layer, :]

    return _interp_edge_leastsq_core(
        xp, yp, node_idx, int(elem), time_dir, bc_code,
        mesh.node_to_elems, mesh.node_to_elems_count,
        mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat,
        neighbors, mesh.elem_center_lon, mesh.elem_center_lat,
        u_fi, v_fi,
    )


def interp_edge_leastsq(
    state: ParticleState,
    mesh: Mesh,
    vel: VelocityField,
    nn_nodes: np.ndarray,
    time_dir: float = 1.0,
    layer=0,
    *,
    bc: str = "freeslip",
    elems=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Edge-neighbor least-squares u/v for all active particles.

    FESOM-style elemental interpolation: locates each particle's
    containing element and fits a plane through the element-centre
    velocities of that element and its edge-sharing neighbors (3 for a
    triangle, 4 for a quad on well-formed meshes — boundary edges
    contribute mirrored ghost points carrying the ``bc`` lateral
    boundary condition instead; degenerate zero-length edges are
    silently dropped from the stencil).

    Parameters
    ----------
    state : ParticleState
    mesh : Mesh
    vel : VelocityField
        Element-layout u/v (node layout raises ValueError).
    nn_nodes : np.ndarray, shape ``(P,)``
        Nearest-node index per particle (containment search anchor).
    time_dir : float
        Forward (+1) or backward (-1).
    layer : int or np.ndarray
        Scalar ``int`` for 2-D, per-particle ``(P,)`` array for 3-D.
        Ignored for a depth-linear field (``vel.uv_vertical``
        ``"layer_linear"``, element layout): that field is read at each
        particle's depth ``state.z``, on a sigma mesh its sigma fraction
        (:func:`_interp_edge_leastsq_dl`).
    bc : str
        Boundary ghost value: ``"freeslip"`` (mirror image of u_own —
        wall-parallel constants exact, normal component damped at the
        wall) or ``"noslip"`` (-u_own, fitted velocity ~0 at the coast
        edge).
    elems : np.ndarray or None
        Optional per-particle PRE-VALIDATED containing elements (edge-walk
        tracker); ``-1`` entries fall back to the internal search.

    Returns
    -------
    U, V : np.ndarray, shape ``(P,)``
        Interpolated velocity (m/s).  Zero for inactive particles.
    ok : np.ndarray of bool, shape ``(P,)``
        True where interpolation succeeded.

    Raises
    ------
    ValueError
        Node layout (a depth-linear node field: ``uv_interp='barycentric'``
        evaluates it), an unknown ``bc``, or a depth-linear record that
        fails the check.
    """
    if getattr(vel, "uv_vertical", "layer") != "layer":
        return _interp_edge_leastsq_dl(state, mesh, vel, nn_nodes, time_dir,
                                       bc=bc, elems=elems)
    _reject_node_layout_edge_lsq(vel)
    bc_code = _bc_to_int(bc)
    neighbors = build_elem_neighbors(mesh)

    P = state.n_particles
    U = np.zeros(P, dtype=np.float64)
    V = np.zeros(P, dtype=np.float64)
    ok = np.zeros(P, dtype=bool)

    if elems is None:
        elems_arr = np.full(P, -1, dtype=np.int32)
    else:
        elems_arr = np.asarray(elems, dtype=np.int32)

    per_particle = isinstance(layer, np.ndarray)

    if vel.u.ndim == 1:
        u_fi = vel.u
        v_fi = vel.v
        per_particle = False
    elif not per_particle:
        u_fi = vel.u[layer, :]
        v_fi = vel.v[layer, :]

    if per_particle:
        layers = np.asarray(layer, dtype=np.int32)
        if HAS_NUMBA:
            _loop_interp_edge_lsq_3d(
                P, state.alive_mask, state.lon, state.lat, nn_nodes,
                elems_arr, layers, time_dir, bc_code,
                mesh.node_to_elems, mesh.node_to_elems_count,
                mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat,
                neighbors, mesh.elem_center_lon, mesh.elem_center_lat,
                vel.u, vel.v, U, V, ok,
            )
        else:
            for i in range(P):
                if not state.alive_mask[i]:
                    continue
                u, v, success = _interp_edge_leastsq_core(
                    state.lon[i], state.lat[i], int(nn_nodes[i]),
                    int(elems_arr[i]), time_dir, bc_code,
                    mesh.node_to_elems, mesh.node_to_elems_count,
                    mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat,
                    neighbors, mesh.elem_center_lon, mesh.elem_center_lat,
                    vel.u[layers[i], :], vel.v[layers[i], :],
                )
                U[i] = u
                V[i] = v
                ok[i] = success
    else:
        if HAS_NUMBA:
            _loop_interp_edge_lsq(
                P, state.alive_mask, state.lon, state.lat, nn_nodes,
                elems_arr, time_dir, bc_code,
                mesh.node_to_elems, mesh.node_to_elems_count,
                mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat,
                neighbors, mesh.elem_center_lon, mesh.elem_center_lat,
                u_fi, v_fi, U, V, ok,
            )
        else:
            for i in range(P):
                if not state.alive_mask[i]:
                    continue
                u, v, success = _interp_edge_leastsq_core(
                    state.lon[i], state.lat[i], int(nn_nodes[i]),
                    int(elems_arr[i]), time_dir, bc_code,
                    mesh.node_to_elems, mesh.node_to_elems_count,
                    mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat,
                    neighbors, mesh.elem_center_lon, mesh.elem_center_lat,
                    u_fi, v_fi,
                )
                U[i] = u
                V[i] = v
                ok[i] = success

    return U, V, ok


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------
#: methods whose vectorised form accepts the ``elems=`` pre-validated
#: containing-element hints (edge-walk tracker); ``leastsq`` needs no
#: containment and ignores the walk entirely.
UV_INTERP_USES_ELEMS = frozenset({"barycentric", "edge_leastsq"})

_UV_INTERPOLATORS = {
    "leastsq": interp_leastsq,
    "barycentric": interp_barycentric,
    "edge_leastsq": interp_edge_leastsq,
}
_UV_INTERPOLATORS_SINGLE = {
    "leastsq": interp_leastsq_single,
    "barycentric": interp_barycentric_single,
    "edge_leastsq": interp_edge_leastsq_single,
}


def _resolve_from_registry(registry, name, edge_leastsq_bc):
    try:
        fn = registry[name]
    except KeyError:
        raise ValueError(
            f"unknown uv_interp {name!r}; valid: "
            f"{sorted(registry)}") from None
    if name == "edge_leastsq":
        _bc_to_int(edge_leastsq_bc)          # validate even for the default
        if edge_leastsq_bc != "freeslip":
            import functools
            return functools.partial(fn, bc=edge_leastsq_bc)
    return fn


def get_interpolator(name: str, edge_leastsq_bc: str = "freeslip"):
    """Resolve a u/v interpolation method by config name (vectorised
    form).  Unknown names raise with the valid list.

    ``edge_leastsq_bc`` selects the boundary-ghost value for the
    ``"edge_leastsq"`` method (``"freeslip"`` default / ``"noslip"``);
    it is ignored by every other method.
    """
    return _resolve_from_registry(_UV_INTERPOLATORS, name, edge_leastsq_bc)


def get_interpolator_single(name: str, edge_leastsq_bc: str = "freeslip"):
    """Single-point twin of get_interpolator (3D RK stage loops)."""
    return _resolve_from_registry(
        _UV_INTERPOLATORS_SINGLE, name, edge_leastsq_bc)


__all__ = [
    "VelocityField",
    "interp_leastsq_single",
    "interp_leastsq",
    "interp_barycentric",
    "interp_barycentric_single",
    "interp_edge_leastsq",
    "interp_edge_leastsq_single",
    "interp_nearest_element",
    "get_interpolator",
    "get_interpolator_single",
    "UV_LAYOUT_ELEMENT",
    "UV_LAYOUT_NODE",
    "BC_FREESLIP",
    "BC_NOSLIP",
    "UV_INTERP_USES_ELEMS",
]
