"""Vertical coordinate utilities for 3D particle tracking.

Provides z-level layer detection, vertical velocity interpolation,
and depth clamping for FESOM2-style z-level meshes.

All hot-path functions are Numba-optimized with pure-Python fallback.
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

# Quad/tri vertex-weight dispatcher (njit->njit cross-module call, the house
# pattern; diffusion.py already imports mesh kernels this way).  Used ONLY on
# the quad (n_valid == 4) branch of the barycentric kernel below — triangles
# keep their inline scalar math (performance gate, see _interp_w_barycentric_single).
from ..core.mesh import _polygon_weights, _seam_safe_lon0
# scalar 4-snapshot monotone-cubic twin (njit; no import cycle:
# temporal_reconstruct depends only on numpy + core.io) — consumed by the
# *_pchip4 W twins below.  NOTE: vertical.py is no longer import-standalone.
from .temporal_reconstruct import _pchip_value_scalar


# ---------------------------------------------------------------------------
# Single-particle layer detection
# ---------------------------------------------------------------------------

@njit(cache=True)
def find_layer(z_levels: np.ndarray, z_particle: float,
               bottom_depth: float) -> int:
    """Find the mid-layer index for a particle at a given depth.

    For z-levels [z0, z1, ..., zK-1], mid-layer k spans [zk, zk+1).
    Returns k such that z_levels[k] <= z_particle < z_levels[k+1],
    clamped to valid layers above the local bottom.

    Parameters
    ----------
    z_levels : np.ndarray
        Z-level boundaries in meters, positive-down, monotonically
        increasing.  Shape ``(K,)``.
    z_particle : float
        Particle depth in meters (positive-down).
    bottom_depth : float
        Local bottom depth at the nearest node (positive-down meters).

    Returns
    -------
    int
        Mid-layer index (0 .. K-2).
    """
    K = z_levels.shape[0]
    if K < 2:
        return 0

    # Clamp particle to surface
    if z_particle <= z_levels[0]:
        return 0

    # Find deepest valid layer (z_levels[k+1] <= bottom_depth)
    max_layer = K - 2  # absolute maximum
    for k in range(K - 1):
        if z_levels[k + 1] > bottom_depth:
            max_layer = max(k - 1, 0)
            break

    # Clamp particle below bottom
    if z_particle >= bottom_depth or z_particle >= z_levels[max_layer + 1]:
        return max_layer

    # Bracket search
    for k in range(max_layer + 1):
        if z_particle < z_levels[k + 1]:
            return k

    return max_layer


# ---------------------------------------------------------------------------
# Vectorised layer detection
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _loop_find_layers(
    z_levels: np.ndarray,
    z_particles: np.ndarray,
    nn_nodes: np.ndarray,
    node_depths: np.ndarray,
    alive_mask: np.ndarray,
    layers_out: np.ndarray,
):
    P = z_particles.shape[0]
    for i in prange(P):
        if not alive_mask[i]:
            continue
        layers_out[i] = find_layer(
            z_levels, z_particles[i], node_depths[nn_nodes[i]]
        )


def find_layers(
    z_levels: np.ndarray,
    z_particles: np.ndarray,
    nn_nodes: np.ndarray,
    node_depths: np.ndarray,
    alive_mask: np.ndarray,
) -> np.ndarray:
    """Find mid-layer index for each particle.

    Parameters
    ----------
    z_levels : np.ndarray, shape ``(K,)``
        Z-level boundaries (positive-down meters).
    z_particles : np.ndarray, shape ``(P,)``
        Particle depths (positive-down meters).
    nn_nodes : np.ndarray, shape ``(P,)``
        Nearest node index per particle.
    node_depths : np.ndarray, shape ``(N,)``
        Bottom depth per node (positive-down meters).
    alive_mask : np.ndarray of bool, shape ``(P,)``
        True for active particles.

    Returns
    -------
    np.ndarray of int32, shape ``(P,)``
        Mid-layer index per particle.  0 for dead particles.
    """
    P = z_particles.shape[0]
    layers = np.zeros(P, dtype=np.int32)
    _loop_find_layers(z_levels, z_particles, nn_nodes, node_depths,
                      alive_mask, layers)
    return layers


# ---------------------------------------------------------------------------
# Vertical velocity interpolation: nearest level
# ---------------------------------------------------------------------------

@njit(cache=True)
def _find_nearest_level(z_levels: np.ndarray, z: float) -> int:
    """Find the z-level index closest to depth z."""
    K = z_levels.shape[0]
    best = 0
    best_dist = abs(z - z_levels[0])
    for k in range(1, K):
        d = abs(z - z_levels[k])
        if d < best_dist:
            best_dist = d
            best = k
    return best


@njit(parallel=True, cache=True)
def _loop_interp_w_nearest(
    w_field: np.ndarray,
    z_levels: np.ndarray,
    z_particles: np.ndarray,
    nn_nodes: np.ndarray,
    alive_mask: np.ndarray,
    time_dir: float,
    W_out: np.ndarray,
):
    P = z_particles.shape[0]
    for i in prange(P):
        if not alive_mask[i]:
            continue
        k = _find_nearest_level(z_levels, z_particles[i])
        W_out[i] = time_dir * w_field[k, nn_nodes[i]]


def interp_w_nearest(
    w_field: np.ndarray,
    z_levels: np.ndarray,
    z_particles: np.ndarray,
    nn_nodes: np.ndarray,
    alive_mask: np.ndarray,
    time_dir: float = 1.0,
) -> np.ndarray:
    """Interpolate vertical velocity using nearest z-level interface.

    Parameters
    ----------
    w_field : np.ndarray, shape ``(K, N)``
        Vertical velocity at level interfaces, node-based.
    z_levels : np.ndarray, shape ``(K,)``
        Z-level boundaries (positive-down meters).
    z_particles : np.ndarray, shape ``(P,)``
        Particle depths.
    nn_nodes : np.ndarray, shape ``(P,)``
        Nearest node index per particle.
    alive_mask : np.ndarray of bool, shape ``(P,)``
    time_dir : float
        Forward (+1) or backward (-1).

    Returns
    -------
    np.ndarray, shape ``(P,)``
        Interpolated W velocity (m/s).
    """
    P = z_particles.shape[0]
    W = np.zeros(P, dtype=np.float64)
    _loop_interp_w_nearest(w_field, z_levels, z_particles, nn_nodes,
                           alive_mask, time_dir, W)
    return W


# ---------------------------------------------------------------------------
# Vertical velocity interpolation: linear between bracketing levels
# ---------------------------------------------------------------------------

@njit(cache=True)
def _interp_w_linear_single(
    w_field: np.ndarray,
    z_levels: np.ndarray,
    z: float,
    node: int,
    time_dir: float,
) -> float:
    """Linear interpolation of W between two bracketing z-level interfaces."""
    K = z_levels.shape[0]
    if K < 2:
        return time_dir * w_field[0, node]

    # Clamp to bounds
    if z <= z_levels[0]:
        return time_dir * w_field[0, node]
    if z >= z_levels[K - 1]:
        return time_dir * w_field[K - 1, node]

    # Find bracket: z_levels[k] <= z < z_levels[k+1]
    k = 0
    for j in range(K - 1):
        if z < z_levels[j + 1]:
            k = j
            break
    else:
        k = K - 2

    dz = z_levels[k + 1] - z_levels[k]
    if dz < 1e-12:
        return time_dir * w_field[k, node]

    alpha = (z - z_levels[k]) / dz
    w_val = (1.0 - alpha) * w_field[k, node] + alpha * w_field[k + 1, node]
    return time_dir * w_val


@njit(cache=True)
def _w_bracket(z_levels: np.ndarray, z: float):
    """Vertical bracket ``(k, alpha)`` for linear interpolation at ``z``.

    Decomposition of :func:`_interp_w_linear_single`'s branch structure
    (TODO 6.4): ``alpha >= 0`` → lerp between interfaces ``k``/``k+1``;
    ``alpha = -1.0`` → take level ``k`` directly (surface/bottom clamp,
    degenerate layer, or a single-interface column).  Composing with
    :func:`_w_at_bracket` is bit-identical to the original function —
    the bracket depends only on ``(z_levels, z)``, so callers evaluating
    several NODES at the same depth compute it once.
    """
    K = z_levels.shape[0]
    if K < 2:
        return 0, -1.0
    if z <= z_levels[0]:
        return 0, -1.0
    if z >= z_levels[K - 1]:
        return K - 1, -1.0
    k = K - 2
    for j in range(K - 1):
        if z < z_levels[j + 1]:
            k = j
            break
    dz = z_levels[k + 1] - z_levels[k]
    if dz < 1e-12:
        return k, -1.0
    return k, (z - z_levels[k]) / dz


@njit(cache=True)
def _w_at_bracket(w_field: np.ndarray, node: int, k: int, alpha: float,
                  time_dir: float) -> float:
    """Evaluate W at a precomputed vertical bracket (see :func:`_w_bracket`)."""
    if alpha < 0.0:
        return time_dir * w_field[k, node]
    return time_dir * ((1.0 - alpha) * w_field[k, node]
                       + alpha * w_field[k + 1, node])


@njit(parallel=True, cache=True)
def _loop_interp_w_linear(
    w_field: np.ndarray,
    z_levels: np.ndarray,
    z_particles: np.ndarray,
    nn_nodes: np.ndarray,
    alive_mask: np.ndarray,
    time_dir: float,
    W_out: np.ndarray,
):
    P = z_particles.shape[0]
    for i in prange(P):
        if not alive_mask[i]:
            continue
        W_out[i] = _interp_w_linear_single(
            w_field, z_levels, z_particles[i], nn_nodes[i], time_dir,
        )


def interp_w_linear(
    w_field: np.ndarray,
    z_levels: np.ndarray,
    z_particles: np.ndarray,
    nn_nodes: np.ndarray,
    alive_mask: np.ndarray,
    time_dir: float = 1.0,
) -> np.ndarray:
    """Interpolate vertical velocity with linear interpolation between levels.

    Linearly interpolates W between the two bracketing z-level interfaces
    at the nearest node.

    Parameters
    ----------
    w_field : np.ndarray, shape ``(K, N)``
        Vertical velocity at level interfaces, node-based.
    z_levels : np.ndarray, shape ``(K,)``
        Z-level boundaries (positive-down meters).
    z_particles : np.ndarray, shape ``(P,)``
        Particle depths.
    nn_nodes : np.ndarray, shape ``(P,)``
        Nearest node index per particle.
    alive_mask : np.ndarray of bool, shape ``(P,)``
    time_dir : float
        Forward (+1) or backward (-1).

    Returns
    -------
    np.ndarray, shape ``(P,)``
        Interpolated W velocity (m/s).
    """
    P = z_particles.shape[0]
    W = np.zeros(P, dtype=np.float64)
    _loop_interp_w_linear(w_field, z_levels, z_particles, nn_nodes,
                          alive_mask, time_dir, W)
    return W


# ---------------------------------------------------------------------------
# Depth clamping
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _loop_clamp_depth(
    z: np.ndarray,
    z_levels: np.ndarray,
    nn_nodes: np.ndarray,
    node_depths: np.ndarray,
    alive_mask: np.ndarray,
    z_out: np.ndarray,
):
    P = z.shape[0]
    z_surface = z_levels[0]
    for i in prange(P):
        if not alive_mask[i]:
            z_out[i] = z[i]
            continue
        z_val = z[i]
        # Surface clamp
        if z_val < z_surface:
            z_val = z_surface
        # Bottom clamp (local bottom at nearest node)
        bottom = node_depths[nn_nodes[i]]
        if z_val > bottom:
            z_val = bottom
        z_out[i] = z_val


def clamp_depth(
    z: np.ndarray,
    z_levels: np.ndarray,
    nn_nodes: np.ndarray,
    node_depths: np.ndarray,
    alive_mask: np.ndarray,
) -> np.ndarray:
    """Clamp particle depths to valid range.

    Surface boundary: ``z >= z_levels[0]`` (typically 0 m).
    Bottom boundary: ``z <= node_depths[nearest_node]`` (local bottom).

    Parameters
    ----------
    z : np.ndarray, shape ``(P,)``
        Particle depths (positive-down meters).
    z_levels : np.ndarray, shape ``(K,)``
        Z-level boundaries.
    nn_nodes : np.ndarray, shape ``(P,)``
        Nearest node index per particle.
    node_depths : np.ndarray, shape ``(N,)``
        Bottom depth per node (positive-down meters).
    alive_mask : np.ndarray of bool, shape ``(P,)``

    Returns
    -------
    np.ndarray, shape ``(P,)``
        Clamped depth values.
    """
    P = z.shape[0]
    z_out = np.empty(P, dtype=np.float64)
    _loop_clamp_depth(z, z_levels, nn_nodes, node_depths, alive_mask, z_out)
    return z_out


# ---------------------------------------------------------------------------
# Method constants (int flags for Numba — no string dispatch)
# ---------------------------------------------------------------------------

W_NEAREST = 0
W_LEASTSQ = 1
W_BARYCENTRIC = 2
W_IDW = 3

_W_METHOD_MAP = {
    "nearest": W_NEAREST,
    "leastsq": W_LEASTSQ,
    "barycentric": W_BARYCENTRIC,
    "idw": W_IDW,
}


# ---------------------------------------------------------------------------
# Neighbour node collection
# ---------------------------------------------------------------------------

_MAX_NEIGHBOURS = 64


@njit(cache=True)
def _collect_neighbour_nodes(
    node_idx: int,
    node_to_elems: np.ndarray,
    node_to_elems_count: np.ndarray,
    elem_nodes: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Collect unique neighbour nodes around *node_idx* (1-ring stencil).

    Returns a fixed-size array and a count.  The centre node is included.
    """
    buf = np.empty(_MAX_NEIGHBOURS, dtype=np.int32)
    count = 0

    # Centre node first
    buf[0] = node_idx
    count = 1

    n_elems = int(node_to_elems_count[node_idx])
    max_v = elem_nodes.shape[1]

    for k in range(n_elems):
        e = int(node_to_elems[node_idx, k])
        if e < 0:
            continue
        for v in range(max_v):
            nd = int(elem_nodes[e, v])
            if nd < 0:
                continue
            # Linear dedup
            found = False
            for j in range(count):
                if buf[j] == nd:
                    found = True
                    break
            if not found and count < _MAX_NEIGHBOURS:
                buf[count] = nd
                count += 1

    return buf, count


# ---------------------------------------------------------------------------
# Horizontal interpolation: least-squares planar fit on node W values
# ---------------------------------------------------------------------------

@njit(cache=True)
def _interp_w_leastsq_single(
    xp: float,
    yp: float,
    z: float,
    node_idx: int,
    w_field: np.ndarray,
    z_levels: np.ndarray,
    time_dir: float,
    node_depths: np.ndarray,
    nodes_lon: np.ndarray,
    nodes_lat: np.ndarray,
    node_to_elems: np.ndarray,
    node_to_elems_count: np.ndarray,
    elem_nodes: np.ndarray,
) -> tuple[float, bool]:
    """Least-squares planar fit of W at particle position.

    Collects 1-ring neighbour nodes, vertically interpolates W at each
    (with bottom-aware clamping), fits ``W = a0 + a1*x + a2*y`` using
    Cramer's rule, evaluates at particle.

    Returns ``(W, ok)``.  ``ok=False`` if < 3 neighbours or singular matrix.
    """
    SLM = 111120.0  # METERS_PER_DEG_LAT
    DEG2R = 0.017453292519943295  # DEG2RAD

    buf, count = _collect_neighbour_nodes(
        node_idx, node_to_elems, node_to_elems_count, elem_nodes
    )

    if count < 3:
        # Fallback: nearest-node vertical interp (bottom-aware)
        z_clamped = min(z, node_depths[node_idx])
        w_val = _interp_w_linear_single(w_field, z_levels, z_clamped, node_idx, time_dir)
        return w_val, False

    z0 = np.cos(yp * DEG2R) * SLM
    if z0 < 1.0:
        z0 = 1.0

    # Particle's own vertical bracket, computed at most ONCE (TODO 6.4),
    # lazily on the first unclamped neighbour (an all-clamped stencil —
    # particle deeper than every neighbour's bottom — must not pay an
    # unused extra O(K) search); every later unclamped neighbour reuses
    # it, clamped neighbours re-bracket at their own bottom depth.
    kp = -1
    ap = 0.0

    # Build normal matrix and RHS
    a11 = 0.0; a12 = 0.0; a13 = 0.0
    a22 = 0.0; a23 = 0.0; a33 = 0.0
    fw = 0.0; f1w = 0.0; f2w = 0.0

    for i in range(count):
        nd = buf[i]
        # Bottom-aware depth clamping per neighbour
        bottom = node_depths[nd]
        if z > bottom:
            kb, ab = _w_bracket(z_levels, bottom)
            w_val = _w_at_bracket(w_field, nd, kb, ab, time_dir)
        else:
            if kp < 0:
                kp, ap = _w_bracket(z_levels, z)
            w_val = _w_at_bracket(w_field, nd, kp, ap, time_dir)

        # Dateline-safe longitude delta
        dlon = ((nodes_lon[nd] - xp + 180.0) % 360.0) - 180.0
        ex = dlon * z0
        ey = (nodes_lat[nd] - yp) * SLM

        a11 += 1.0
        a12 += ex
        a13 += ey
        a22 += ex * ex
        a23 += ex * ey
        a33 += ey * ey
        fw += w_val
        f1w += w_val * ex
        f2w += w_val * ey

    a21 = a12; a31 = a13; a32 = a23

    det = (a11 * a22 * a33
           + a12 * a23 * a31
           + a13 * a21 * a32
           - a13 * a22 * a31
           - a12 * a21 * a33
           - a11 * a23 * a32)

    # scale-relative singularity guard — see _interp_leastsq_core
    if abs(det) < 1e-30 or abs(det) < 1e-12 * abs(a11 * a22 * a33):
        z_clamped = min(z, node_depths[node_idx])
        w_val = _interp_w_linear_single(w_field, z_levels, z_clamped, node_idx, time_dir)
        return w_val, False

    inv = 1.0 / abs(det)

    b11 = a22 * a33 - a23 * a32
    b12 = a13 * a32 - a12 * a33
    b13 = a12 * a23 - a13 * a22

    a0W = (b11 * fw + b12 * f1w + b13 * f2w) * inv

    return a0W, True


# ---------------------------------------------------------------------------
# Horizontal interpolation: barycentric on containing element
# ---------------------------------------------------------------------------

@njit(cache=True)
def _interp_w_barycentric_single(
    xp: float,
    yp: float,
    z: float,
    node_idx: int,
    w_field: np.ndarray,
    z_levels: np.ndarray,
    time_dir: float,
    node_depths: np.ndarray,
    nodes_lon: np.ndarray,
    nodes_lat: np.ndarray,
    node_to_elems: np.ndarray,
    node_to_elems_count: np.ndarray,
    elem_nodes: np.ndarray,
) -> tuple[float, bool]:
    """Barycentric interpolation of W inside the containing element.

    Finds the containing element via local search, computes barycentric
    weights on each sub-triangle (fan triangulation for quads), and
    blends vertex W values.

    Returns ``(W, ok)``.  ``ok=False`` if no containing element found.
    """
    SLM = 111120.0
    DEG2R = 0.017453292519943295

    n_elems = int(node_to_elems_count[node_idx])
    if n_elems == 0:
        z_clamped = min(z, node_depths[node_idx])
        w_val = _interp_w_linear_single(w_field, z_levels, z_clamped, node_idx, time_dir)
        return w_val, False

    z0 = np.cos(yp * DEG2R) * SLM
    if z0 < 1.0:
        z0 = 1.0

    max_v = elem_nodes.shape[1]

    for k in range(n_elems):
        e = int(node_to_elems[node_idx, k])
        if e < 0:
            continue

        # Collect valid vertices
        verts = np.empty(max_v, dtype=np.int32)
        n_valid = 0
        for v in range(max_v):
            nd = int(elem_nodes[e, v])
            if nd < 0:
                continue
            # Skip duplicate (tri stored as quad with repeated vertex)
            if n_valid > 0 and nd == verts[0]:
                continue
            verts[n_valid] = nd
            n_valid += 1

        if n_valid < 3:
            continue

        # seam-safe origin: the raw vertex mean, or the first vertex plus the
        # mean wrapped offset for an element straddling +-180 (2026-09-24)
        lon0 = _seam_safe_lon0(nodes_lon, verts, n_valid)
        lat0 = 0.0
        for v in range(n_valid):
            lat0 += nodes_lat[verts[v]]
        lat0 /= n_valid

        cos_lat0 = np.cos(lat0 * DEG2R)
        R = 6371000.0  # R_EARTH_STANDARD used in mesh.py

        # Project vertices to local XY
        vx = np.empty(n_valid, dtype=np.float64)
        vy = np.empty(n_valid, dtype=np.float64)
        for v in range(n_valid):
            dlon = ((nodes_lon[verts[v]] - lon0 + 180.0) % 360.0) - 180.0
            dlat = nodes_lat[verts[v]] - lat0
            vx[v] = dlon * DEG2R * R * cos_lat0
            vy[v] = dlat * DEG2R * R

        dlon_p = ((xp - lon0 + 180.0) % 360.0) - 180.0
        dlat_p = yp - lat0
        px = dlon_p * DEG2R * R * cos_lat0
        py = dlat_p * DEG2R * R

        # Fan triangulation from vertex 0
        for t in range(n_valid - 2):
            i0 = 0
            i1 = t + 1
            i2 = t + 2

            ax, ay = vx[i0], vy[i0]
            bx, by = vx[i1], vy[i1]
            cx, cy = vx[i2], vy[i2]

            denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
            if abs(denom) < 1e-14:
                continue

            w1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denom
            w2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denom
            w3 = 1.0 - w1 - w2

            eps = -1e-6
            if w1 >= eps and w2 >= eps and w3 >= eps:
                # Inside this sub-triangle. PERFORMANCE GATE: quads
                # (n_valid == 4) go through the Wachspress dispatcher so
                # all four vertices contribute and the result is smooth and
                # vertex-order independent; triangles keep the inline scalar
                # 3-vertex blend below (zero allocations, bit-identical).
                if n_valid == 4:
                    # Product-form Wachspress: no sentinel, valid on edges
                    # and vertices too (guard chain removed 2026-07).
                    wts = _polygon_weights(px, py, vx, vy, n_valid)
                    acc = 0.0
                    for vi in range(n_valid):
                        nv = verts[vi]
                        z_nv = min(z, node_depths[nv])
                        acc += wts[vi] * _interp_w_linear_single(
                            w_field, z_levels, z_nv, nv, time_dir)
                    return acc, True

                # Triangle path: inline blend.
                n0 = verts[i0]; n1 = verts[i1]; n2 = verts[i2]

                z0_n0 = min(z, node_depths[n0])
                z0_n1 = min(z, node_depths[n1])
                z0_n2 = min(z, node_depths[n2])

                w_v0 = _interp_w_linear_single(w_field, z_levels, z0_n0, n0, time_dir)
                w_v1 = _interp_w_linear_single(w_field, z_levels, z0_n1, n1, time_dir)
                w_v2 = _interp_w_linear_single(w_field, z_levels, z0_n2, n2, time_dir)

                return w1 * w_v0 + w2 * w_v1 + w3 * w_v2, True

    # No containing element found — fallback (bottom-aware)
    z_clamped = min(z, node_depths[node_idx])
    w_val = _interp_w_linear_single(w_field, z_levels, z_clamped, node_idx, time_dir)
    return w_val, False


@njit(cache=True)
def _interp_w_bary_single_tinterp(
    xp: float,
    yp: float,
    z: float,
    node_idx: int,
    w_a: np.ndarray,
    w_b: np.ndarray,
    w_t: np.ndarray,
    alpha0: float,
    alpha: float,
    z_levels: np.ndarray,
    time_dir: float,
    node_depths: np.ndarray,
    nodes_lon: np.ndarray,
    nodes_lat: np.ndarray,
    node_to_elems: np.ndarray,
    node_to_elems_count: np.ndarray,
    elem_nodes: np.ndarray,
) -> tuple[float, bool]:
    """Twin of :func:`_interp_w_barycentric_single` — identical
    geometry, W reads time-blended inline (see the scalar-twins
    block comment).  Returns ``(W, ok)``."""
    SLM = 111120.0
    DEG2R = 0.017453292519943295

    n_elems = int(node_to_elems_count[node_idx])
    if n_elems == 0:
        z_clamped = min(z, node_depths[node_idx])
        w_val = _interp_w_linear_single_tinterp(w_a, w_b, w_t, alpha0, alpha, z_levels, z_clamped, node_idx, time_dir)
        return w_val, False

    z0 = np.cos(yp * DEG2R) * SLM
    if z0 < 1.0:
        z0 = 1.0

    max_v = elem_nodes.shape[1]

    for k in range(n_elems):
        e = int(node_to_elems[node_idx, k])
        if e < 0:
            continue

        # Collect valid vertices
        verts = np.empty(max_v, dtype=np.int32)
        n_valid = 0
        for v in range(max_v):
            nd = int(elem_nodes[e, v])
            if nd < 0:
                continue
            # Skip duplicate (tri stored as quad with repeated vertex)
            if n_valid > 0 and nd == verts[0]:
                continue
            verts[n_valid] = nd
            n_valid += 1

        if n_valid < 3:
            continue

        # seam-safe origin: the raw vertex mean, or the first vertex plus the
        # mean wrapped offset for an element straddling +-180 (2026-09-24)
        lon0 = _seam_safe_lon0(nodes_lon, verts, n_valid)
        lat0 = 0.0
        for v in range(n_valid):
            lat0 += nodes_lat[verts[v]]
        lat0 /= n_valid

        cos_lat0 = np.cos(lat0 * DEG2R)
        R = 6371000.0  # R_EARTH_STANDARD used in mesh.py

        # Project vertices to local XY
        vx = np.empty(n_valid, dtype=np.float64)
        vy = np.empty(n_valid, dtype=np.float64)
        for v in range(n_valid):
            dlon = ((nodes_lon[verts[v]] - lon0 + 180.0) % 360.0) - 180.0
            dlat = nodes_lat[verts[v]] - lat0
            vx[v] = dlon * DEG2R * R * cos_lat0
            vy[v] = dlat * DEG2R * R

        dlon_p = ((xp - lon0 + 180.0) % 360.0) - 180.0
        dlat_p = yp - lat0
        px = dlon_p * DEG2R * R * cos_lat0
        py = dlat_p * DEG2R * R

        # Fan triangulation from vertex 0
        for t in range(n_valid - 2):
            i0 = 0
            i1 = t + 1
            i2 = t + 2

            ax, ay = vx[i0], vy[i0]
            bx, by = vx[i1], vy[i1]
            cx, cy = vx[i2], vy[i2]

            denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
            if abs(denom) < 1e-14:
                continue

            w1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denom
            w2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denom
            w3 = 1.0 - w1 - w2

            eps = -1e-6
            if w1 >= eps and w2 >= eps and w3 >= eps:
                # Inside this sub-triangle. PERFORMANCE GATE: quads
                # (n_valid == 4) go through the Wachspress dispatcher so
                # all four vertices contribute and the result is smooth and
                # vertex-order independent; triangles keep the inline scalar
                # 3-vertex blend below (zero allocations, bit-identical).
                if n_valid == 4:
                    # Product-form Wachspress: no sentinel, valid on edges
                    # and vertices too (guard chain removed 2026-07).
                    wts = _polygon_weights(px, py, vx, vy, n_valid)
                    acc = 0.0
                    for vi in range(n_valid):
                        nv = verts[vi]
                        z_nv = min(z, node_depths[nv])
                        acc += wts[vi] * _interp_w_linear_single_tinterp(w_a, w_b, w_t, alpha0, alpha, z_levels, z_nv, nv, time_dir)
                    return acc, True

                # Triangle path: inline blend.
                n0 = verts[i0]; n1 = verts[i1]; n2 = verts[i2]

                z0_n0 = min(z, node_depths[n0])
                z0_n1 = min(z, node_depths[n1])
                z0_n2 = min(z, node_depths[n2])

                w_v0 = _interp_w_linear_single_tinterp(w_a, w_b, w_t, alpha0, alpha, z_levels, z0_n0, n0, time_dir)
                w_v1 = _interp_w_linear_single_tinterp(w_a, w_b, w_t, alpha0, alpha, z_levels, z0_n1, n1, time_dir)
                w_v2 = _interp_w_linear_single_tinterp(w_a, w_b, w_t, alpha0, alpha, z_levels, z0_n2, n2, time_dir)

                return w1 * w_v0 + w2 * w_v1 + w3 * w_v2, True

    # No containing element found — fallback (bottom-aware)
    z_clamped = min(z, node_depths[node_idx])
    w_val = _interp_w_linear_single_tinterp(w_a, w_b, w_t, alpha0, alpha, z_levels, z_clamped, node_idx, time_dir)
    return w_val, False



@njit(cache=True)
def _interp_w_bary_single_pchip4(
    xp: float,
    yp: float,
    z: float,
    node_idx: int,
    w_0: np.ndarray,
    w_1: np.ndarray,
    w_2: np.ndarray,
    w_3: np.ndarray,
    t0: float,
    t1: float,
    t2: float,
    t3: float,
    tq: float,
    z_levels: np.ndarray,
    time_dir: float,
    node_depths: np.ndarray,
    nodes_lon: np.ndarray,
    nodes_lat: np.ndarray,
    node_to_elems: np.ndarray,
    node_to_elems_count: np.ndarray,
    elem_nodes: np.ndarray,
) -> tuple[float, bool]:
    """Twin of :func:`_interp_w_barycentric_single` — identical
    geometry, W reads time-blended inline (see the scalar-twins
    block comment).  Returns ``(W, ok)``."""
    SLM = 111120.0
    DEG2R = 0.017453292519943295

    n_elems = int(node_to_elems_count[node_idx])
    if n_elems == 0:
        z_clamped = min(z, node_depths[node_idx])
        w_val = _interp_w_linear_single_pchip4(w_0, w_1, w_2, w_3, t0, t1, t2, t3, tq, z_levels, z_clamped, node_idx, time_dir)
        return w_val, False

    z0 = np.cos(yp * DEG2R) * SLM
    if z0 < 1.0:
        z0 = 1.0

    max_v = elem_nodes.shape[1]

    for k in range(n_elems):
        e = int(node_to_elems[node_idx, k])
        if e < 0:
            continue

        # Collect valid vertices
        verts = np.empty(max_v, dtype=np.int32)
        n_valid = 0
        for v in range(max_v):
            nd = int(elem_nodes[e, v])
            if nd < 0:
                continue
            # Skip duplicate (tri stored as quad with repeated vertex)
            if n_valid > 0 and nd == verts[0]:
                continue
            verts[n_valid] = nd
            n_valid += 1

        if n_valid < 3:
            continue

        # seam-safe origin: the raw vertex mean, or the first vertex plus the
        # mean wrapped offset for an element straddling +-180 (2026-09-24)
        lon0 = _seam_safe_lon0(nodes_lon, verts, n_valid)
        lat0 = 0.0
        for v in range(n_valid):
            lat0 += nodes_lat[verts[v]]
        lat0 /= n_valid

        cos_lat0 = np.cos(lat0 * DEG2R)
        R = 6371000.0  # R_EARTH_STANDARD used in mesh.py

        # Project vertices to local XY
        vx = np.empty(n_valid, dtype=np.float64)
        vy = np.empty(n_valid, dtype=np.float64)
        for v in range(n_valid):
            dlon = ((nodes_lon[verts[v]] - lon0 + 180.0) % 360.0) - 180.0
            dlat = nodes_lat[verts[v]] - lat0
            vx[v] = dlon * DEG2R * R * cos_lat0
            vy[v] = dlat * DEG2R * R

        dlon_p = ((xp - lon0 + 180.0) % 360.0) - 180.0
        dlat_p = yp - lat0
        px = dlon_p * DEG2R * R * cos_lat0
        py = dlat_p * DEG2R * R

        # Fan triangulation from vertex 0
        for t in range(n_valid - 2):
            i0 = 0
            i1 = t + 1
            i2 = t + 2

            ax, ay = vx[i0], vy[i0]
            bx, by = vx[i1], vy[i1]
            cx, cy = vx[i2], vy[i2]

            denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
            if abs(denom) < 1e-14:
                continue

            w1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denom
            w2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denom
            w3 = 1.0 - w1 - w2

            eps = -1e-6
            if w1 >= eps and w2 >= eps and w3 >= eps:
                # Inside this sub-triangle. PERFORMANCE GATE: quads
                # (n_valid == 4) go through the Wachspress dispatcher so
                # all four vertices contribute and the result is smooth and
                # vertex-order independent; triangles keep the inline scalar
                # 3-vertex blend below (zero allocations, bit-identical).
                if n_valid == 4:
                    # Product-form Wachspress: no sentinel, valid on edges
                    # and vertices too (guard chain removed 2026-07).
                    wts = _polygon_weights(px, py, vx, vy, n_valid)
                    acc = 0.0
                    for vi in range(n_valid):
                        nv = verts[vi]
                        z_nv = min(z, node_depths[nv])
                        acc += wts[vi] * _interp_w_linear_single_pchip4(w_0, w_1, w_2, w_3, t0, t1, t2, t3, tq, z_levels, z_nv, nv, time_dir)
                    return acc, True

                # Triangle path: inline blend.
                n0 = verts[i0]; n1 = verts[i1]; n2 = verts[i2]

                z0_n0 = min(z, node_depths[n0])
                z0_n1 = min(z, node_depths[n1])
                z0_n2 = min(z, node_depths[n2])

                w_v0 = _interp_w_linear_single_pchip4(w_0, w_1, w_2, w_3, t0, t1, t2, t3, tq, z_levels, z0_n0, n0, time_dir)
                w_v1 = _interp_w_linear_single_pchip4(w_0, w_1, w_2, w_3, t0, t1, t2, t3, tq, z_levels, z0_n1, n1, time_dir)
                w_v2 = _interp_w_linear_single_pchip4(w_0, w_1, w_2, w_3, t0, t1, t2, t3, tq, z_levels, z0_n2, n2, time_dir)

                return w1 * w_v0 + w2 * w_v1 + w3 * w_v2, True

    # No containing element found — fallback (bottom-aware)
    z_clamped = min(z, node_depths[node_idx])
    w_val = _interp_w_linear_single_pchip4(w_0, w_1, w_2, w_3, t0, t1, t2, t3, tq, z_levels, z_clamped, node_idx, time_dir)
    return w_val, False



# ---------------------------------------------------------------------------
# Time-aware scalar twins for the fused 3-D kernels (2026-08)
#
# The unfused RK path MATERIALIZES a stage velocity field
# ``lerp(lerp(A, B, alpha0), T, alpha_k)`` (or the 4-snapshot PCHIP
# reconstruction) and then runs the singles above on it.  The twins below
# reproduce that value PER ACCESSED ELEMENT with bit-for-bit the same
# arithmetic and the same endpoint shortcuts as
# ``interpolate._interp_barycentric_core_tinterp`` /
# ``temporal_reconstruct._pchip_value_scalar`` — so a fused kernel that
# calls them matches the unfused path exactly (pinned by
# tests/test_fused_3d_bary.py).
# ---------------------------------------------------------------------------

@njit(cache=True)
def _tblend_scalar(a: float, b: float, t: float,
                   alpha0: float, alpha: float) -> float:
    """Nested two-level time blend of one scalar (see block comment).

    Mirrors ``_make_stage_vels``' object shortcuts: ``alpha < 1e-12`` →
    the bracket value (lazy pair materialized iff ``alpha0 != 0``),
    ``alpha > 1 - 1e-12`` → the target value; otherwise the exact
    ``(1-alpha)*x + alpha*t`` lerp arithmetic of ``core.io._blend``.
    """
    if alpha < 1e-12:
        if alpha0 == 0.0:
            return a
        return (1.0 - alpha0) * a + alpha0 * b
    if alpha > 1.0 - 1e-12:
        return t
    if alpha0 == 0.0:
        return (1.0 - alpha) * a + alpha * t
    a0 = (1.0 - alpha0) * a + alpha0 * b
    return (1.0 - alpha) * a0 + alpha * t


@njit(cache=True)
def _interp_w_linear_single_tinterp(
    w_a: np.ndarray, w_b: np.ndarray, w_t: np.ndarray,
    alpha0: float, alpha: float,
    z_levels: np.ndarray, z: float, node: int, time_dir: float,
) -> float:
    """Twin of :func:`_interp_w_linear_single` with inline time blending."""
    K = z_levels.shape[0]
    if K < 2:
        return time_dir * _tblend_scalar(
            w_a[0, node], w_b[0, node], w_t[0, node], alpha0, alpha)
    if z <= z_levels[0]:
        return time_dir * _tblend_scalar(
            w_a[0, node], w_b[0, node], w_t[0, node], alpha0, alpha)
    if z >= z_levels[K - 1]:
        return time_dir * _tblend_scalar(
            w_a[K - 1, node], w_b[K - 1, node], w_t[K - 1, node],
            alpha0, alpha)
    k = 0
    for j in range(K - 1):
        if z < z_levels[j + 1]:
            k = j
            break
    else:
        k = K - 2
    dz = z_levels[k + 1] - z_levels[k]
    if dz < 1e-12:
        return time_dir * _tblend_scalar(
            w_a[k, node], w_b[k, node], w_t[k, node], alpha0, alpha)
    a = (z - z_levels[k]) / dz
    w_lo = _tblend_scalar(w_a[k, node], w_b[k, node], w_t[k, node],
                          alpha0, alpha)
    w_hi = _tblend_scalar(w_a[k + 1, node], w_b[k + 1, node],
                          w_t[k + 1, node], alpha0, alpha)
    return time_dir * ((1.0 - a) * w_lo + a * w_hi)


@njit(cache=True)
def _interp_w_linear_single_pchip4(
    w_0: np.ndarray, w_1: np.ndarray, w_2: np.ndarray, w_3: np.ndarray,
    t0: float, t1: float, t2: float, t3: float, tq: float,
    z_levels: np.ndarray, z: float, node: int, time_dir: float,
) -> float:
    """Twin of :func:`_interp_w_linear_single` with inline PCHIP blending."""
    K = z_levels.shape[0]
    if K < 2:
        return time_dir * _pchip_value_scalar(
            w_0[0, node], w_1[0, node], w_2[0, node], w_3[0, node],
            t0, t1, t2, t3, tq)
    if z <= z_levels[0]:
        return time_dir * _pchip_value_scalar(
            w_0[0, node], w_1[0, node], w_2[0, node], w_3[0, node],
            t0, t1, t2, t3, tq)
    if z >= z_levels[K - 1]:
        return time_dir * _pchip_value_scalar(
            w_0[K - 1, node], w_1[K - 1, node], w_2[K - 1, node],
            w_3[K - 1, node], t0, t1, t2, t3, tq)
    k = 0
    for j in range(K - 1):
        if z < z_levels[j + 1]:
            k = j
            break
    else:
        k = K - 2
    dz = z_levels[k + 1] - z_levels[k]
    if dz < 1e-12:
        return time_dir * _pchip_value_scalar(
            w_0[k, node], w_1[k, node], w_2[k, node], w_3[k, node],
            t0, t1, t2, t3, tq)
    a = (z - z_levels[k]) / dz
    w_lo = _pchip_value_scalar(
        w_0[k, node], w_1[k, node], w_2[k, node], w_3[k, node],
        t0, t1, t2, t3, tq)
    w_hi = _pchip_value_scalar(
        w_0[k + 1, node], w_1[k + 1, node], w_2[k + 1, node],
        w_3[k + 1, node], t0, t1, t2, t3, tq)
    return time_dir * ((1.0 - a) * w_lo + a * w_hi)


# ---------------------------------------------------------------------------
# Horizontal interpolation: inverse-distance weighting
# ---------------------------------------------------------------------------

@njit(cache=True)
def _interp_w_idw_single(
    xp: float,
    yp: float,
    z: float,
    node_idx: int,
    w_field: np.ndarray,
    z_levels: np.ndarray,
    time_dir: float,
    node_depths: np.ndarray,
    nodes_lon: np.ndarray,
    nodes_lat: np.ndarray,
    node_to_elems: np.ndarray,
    node_to_elems_count: np.ndarray,
    elem_nodes: np.ndarray,
) -> tuple[float, bool]:
    """Inverse-distance-weighted interpolation of W.

    Collects 1-ring neighbour nodes, vertically interpolates W at each
    (with bottom-aware clamping), weights by 1/d².

    Returns ``(W, ok)``.  If particle sits on a node (d < ε), returns
    that node's W directly.
    """
    SLM = 111120.0
    DEG2R = 0.017453292519943295
    EPS_DIST = 1.0  # 1 metre — treat as coincident

    buf, count = _collect_neighbour_nodes(
        node_idx, node_to_elems, node_to_elems_count, elem_nodes
    )

    if count < 1:
        z_clamped = min(z, node_depths[node_idx])
        w_val = _interp_w_linear_single(w_field, z_levels, z_clamped, node_idx, time_dir)
        return w_val, False

    z0 = np.cos(yp * DEG2R) * SLM
    if z0 < 1.0:
        z0 = 1.0

    # Particle's own vertical bracket, computed lazily at most ONCE
    # (TODO 6.4) — see _interp_w_leastsq_single for the reuse contract.
    kp = -1
    ap = 0.0

    sum_w = 0.0
    sum_wv = 0.0

    for i in range(count):
        nd = buf[i]
        bottom = node_depths[nd]
        if z > bottom:
            kb, ab = _w_bracket(z_levels, bottom)
            w_val = _w_at_bracket(w_field, nd, kb, ab, time_dir)
        else:
            if kp < 0:
                kp, ap = _w_bracket(z_levels, z)
            w_val = _w_at_bracket(w_field, nd, kp, ap, time_dir)

        # Dateline-safe longitude delta
        dlon = ((nodes_lon[nd] - xp + 180.0) % 360.0) - 180.0
        dx = dlon * z0
        dy = (nodes_lat[nd] - yp) * SLM
        d2 = dx * dx + dy * dy

        if d2 < EPS_DIST * EPS_DIST:
            # Coincident with node — return directly
            return w_val, True

        weight = 1.0 / d2
        sum_w += weight
        sum_wv += weight * w_val

    if sum_w < 1e-30:
        z_clamped = min(z, node_depths[node_idx])
        w_val = _interp_w_linear_single(w_field, z_levels, z_clamped, node_idx, time_dir)
        return w_val, False

    return sum_wv / sum_w, True


# ---------------------------------------------------------------------------
# Vectorised loops for each method
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _loop_interp_w_leastsq(
    w_field: np.ndarray,
    z_levels: np.ndarray,
    z_particles: np.ndarray,
    nn_nodes: np.ndarray,
    alive_mask: np.ndarray,
    time_dir: float,
    node_depths: np.ndarray,
    nodes_lon: np.ndarray,
    nodes_lat: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    node_to_elems: np.ndarray,
    node_to_elems_count: np.ndarray,
    elem_nodes: np.ndarray,
    W_out: np.ndarray,
):
    P = z_particles.shape[0]
    for i in prange(P):
        if not alive_mask[i]:
            continue
        w, _ = _interp_w_leastsq_single(
            lons[i], lats[i], z_particles[i], nn_nodes[i],
            w_field, z_levels, time_dir, node_depths,
            nodes_lon, nodes_lat,
            node_to_elems, node_to_elems_count, elem_nodes,
        )
        W_out[i] = w


@njit(parallel=True, cache=True)
def _loop_interp_w_barycentric(
    w_field: np.ndarray,
    z_levels: np.ndarray,
    z_particles: np.ndarray,
    nn_nodes: np.ndarray,
    alive_mask: np.ndarray,
    time_dir: float,
    node_depths: np.ndarray,
    nodes_lon: np.ndarray,
    nodes_lat: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    node_to_elems: np.ndarray,
    node_to_elems_count: np.ndarray,
    elem_nodes: np.ndarray,
    W_out: np.ndarray,
):
    P = z_particles.shape[0]
    for i in prange(P):
        if not alive_mask[i]:
            continue
        w, _ = _interp_w_barycentric_single(
            lons[i], lats[i], z_particles[i], nn_nodes[i],
            w_field, z_levels, time_dir, node_depths,
            nodes_lon, nodes_lat,
            node_to_elems, node_to_elems_count, elem_nodes,
        )
        W_out[i] = w


@njit(parallel=True, cache=True)
def _loop_interp_w_idw(
    w_field: np.ndarray,
    z_levels: np.ndarray,
    z_particles: np.ndarray,
    nn_nodes: np.ndarray,
    alive_mask: np.ndarray,
    time_dir: float,
    node_depths: np.ndarray,
    nodes_lon: np.ndarray,
    nodes_lat: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    node_to_elems: np.ndarray,
    node_to_elems_count: np.ndarray,
    elem_nodes: np.ndarray,
    W_out: np.ndarray,
):
    P = z_particles.shape[0]
    for i in prange(P):
        if not alive_mask[i]:
            continue
        w, _ = _interp_w_idw_single(
            lons[i], lats[i], z_particles[i], nn_nodes[i],
            w_field, z_levels, time_dir, node_depths,
            nodes_lon, nodes_lat,
            node_to_elems, node_to_elems_count, elem_nodes,
        )
        W_out[i] = w


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def interp_w_spatial(
    w_field: np.ndarray,
    z_levels: np.ndarray,
    z_particles: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    nn_nodes: np.ndarray,
    alive_mask: np.ndarray,
    time_dir: float = 1.0,
    method: str = "leastsq",
    mesh=None,
    node_depths: np.ndarray | None = None,
) -> np.ndarray:
    """Interpolate vertical velocity with horizontal spatial methods.

    Full version that accepts particle lon/lat explicitly.

    Parameters
    ----------
    w_field : np.ndarray, shape ``(K, N)``
        Vertical velocity at level interfaces, node-based.
    z_levels : np.ndarray, shape ``(K,)``
        Z-level boundaries (positive-down meters).
    z_particles : np.ndarray, shape ``(P,)``
        Particle depths.
    lons, lats : np.ndarray, shape ``(P,)``
        Particle longitude and latitude in degrees.
    nn_nodes : np.ndarray, shape ``(P,)``
        Nearest node index per particle.
    alive_mask : np.ndarray of bool, shape ``(P,)``
    time_dir : float
        Forward (+1) or backward (-1).
    method : str
        One of ``"nearest"``, ``"leastsq"``, ``"barycentric"``, ``"idw"``.
    mesh : Mesh or None
        Required for all methods except ``"nearest"``.
    node_depths : np.ndarray or None, shape ``(N,)``
        Bottom depth per node.  If None, uses ``mesh.depth``.

    Returns
    -------
    np.ndarray, shape ``(P,)``
        Interpolated W velocity (m/s).
    """
    if method not in _W_METHOD_MAP:
        raise ValueError(
            f"Unknown w_interp method '{method}'. "
            f"Available: {sorted(_W_METHOD_MAP)}"
        )

    P = z_particles.shape[0]
    W = np.zeros(P, dtype=np.float64)

    if method == "nearest":
        _loop_interp_w_linear(
            w_field, z_levels, z_particles, nn_nodes,
            alive_mask, time_dir, W,
        )
        return W

    if mesh is None:
        raise ValueError(f"mesh is required for w_interp method '{method}'")

    if node_depths is None:
        node_depths = mesh.depth

    args = (
        w_field, z_levels, z_particles, nn_nodes, alive_mask, time_dir,
        node_depths, mesh.nodes_lon, mesh.nodes_lat,
        lons, lats,
        mesh.node_to_elems, mesh.node_to_elems_count, mesh.elem_nodes,
        W,
    )

    if method == "leastsq":
        if HAS_NUMBA:
            _loop_interp_w_leastsq(*args)
        else:
            for i in range(P):
                if not alive_mask[i]:
                    continue
                w, _ = _interp_w_leastsq_single(
                    lons[i], lats[i], z_particles[i], int(nn_nodes[i]),
                    w_field, z_levels, time_dir, node_depths,
                    mesh.nodes_lon, mesh.nodes_lat,
                    mesh.node_to_elems, mesh.node_to_elems_count,
                    mesh.elem_nodes,
                )
                W[i] = w
    elif method == "barycentric":
        if HAS_NUMBA:
            _loop_interp_w_barycentric(*args)
        else:
            for i in range(P):
                if not alive_mask[i]:
                    continue
                w, _ = _interp_w_barycentric_single(
                    lons[i], lats[i], z_particles[i], int(nn_nodes[i]),
                    w_field, z_levels, time_dir, node_depths,
                    mesh.nodes_lon, mesh.nodes_lat,
                    mesh.node_to_elems, mesh.node_to_elems_count,
                    mesh.elem_nodes,
                )
                W[i] = w
    elif method == "idw":
        if HAS_NUMBA:
            _loop_interp_w_idw(*args)
        else:
            for i in range(P):
                if not alive_mask[i]:
                    continue
                w, _ = _interp_w_idw_single(
                    lons[i], lats[i], z_particles[i], int(nn_nodes[i]),
                    w_field, z_levels, time_dir, node_depths,
                    mesh.nodes_lon, mesh.nodes_lat,
                    mesh.node_to_elems, mesh.node_to_elems_count,
                    mesh.elem_nodes,
                )
                W[i] = w

    return W


__all__ = [
    "find_layer",
    "find_layers",
    "interp_w_nearest",
    "interp_w_linear",
    "interp_w_spatial",
    "clamp_depth",
    "W_NEAREST",
    "W_LEASTSQ",
    "W_BARYCENTRIC",
    "W_IDW",
    "_W_METHOD_MAP",
    "_collect_neighbour_nodes",
    "_interp_w_leastsq_single",
    "_interp_w_barycentric_single",
    "_interp_w_idw_single",
]
