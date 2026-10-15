"""Sigma-coordinate vertical utilities for 3D particle tracking.

Provides per-node sigma layer detection, sigma-aware depth clamping and
interface-based W interpolation (the dia-sigma flux ``-Wvel``) for
FESOM-C-style sigma meshes.

Particle state stays in physical depth ``z`` (positive-down meters).
The transformation between sigma and physical depth is done locally for
each water column using ``H_col = max(depth_node + eta_node, Dmin)`` and
``z_ifc = (1 - sigma_lev) * H_col`` (FESOM-C linear sigma transform —
σ=1 is surface, σ=0 is bottom).

All hot-path functions are Numba-optimised with pure-Python fallback.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:  # pragma: no cover — local dev fallback
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def prange(*args, **kwargs):
        return range(*args)

# Quad/tri vertex-weight dispatcher (njit->njit cross-module call).  Used ONLY
# on the quad (n_valid == 4) branch of the sigma barycentric kernel; triangles
# keep their inline scalar blend (performance gate).
from ..core.mesh import _polygon_weights, _seam_safe_lon0


# ---------------------------------------------------------------------------
# Column geometry (small helpers, used by tests and as building blocks)
# ---------------------------------------------------------------------------

@njit(cache=True)
def column_depth(depth_node: float, eta_node: float, Dmin: float) -> float:
    """Return local water-column thickness ``H_col`` in metres.

    ``H_col = max(depth_node + eta_node, Dmin)``.  Protects against
    degenerate near-zero columns (dry cells, mesh boundaries) by
    clipping to ``Dmin``.
    """
    H = depth_node + eta_node
    if H < Dmin:
        H = Dmin
    return H


def sigma_interfaces_to_z(sigma_lev: np.ndarray, H_col: float) -> np.ndarray:
    """Compute physical interface depths for one water column.

    Applies the FESOM-C linear sigma transform ``z = (1 - sigma) * H``.
    For ``sigma_lev = [1.0, 0.9, ..., 0.0]`` this yields depths
    ``[0, 0.1*H, ..., H]`` (surface → bottom, positive-down).

    Returned array has the same shape as ``sigma_lev``.
    """
    return (1.0 - np.asarray(sigma_lev, dtype=np.float64)) * float(H_col)


# ---------------------------------------------------------------------------
# Single-particle layer detection (sigma)
# ---------------------------------------------------------------------------

@njit(cache=True)
def find_layer_sigma(sigma_lev: np.ndarray, z_particle: float,
                     H_col: float) -> int:
    """Find the mid-layer index for a particle in a local sigma column.

    Interfaces are computed inline as ``z_ifc[k] = (1 - sigma_lev[k]) *
    H_col``.  For FESOM-C linear sigma with ``sigma_lev = [1, …, 0]``
    these interface depths increase from 0 (surface) to ``H_col``
    (bottom), and layer ``k`` spans ``[z_ifc[k], z_ifc[k+1])``.

    Returns the 0-based mid-layer index clamped to ``[0, K-2]``.
    """
    K = sigma_lev.shape[0]
    if K < 2:
        return 0

    # Top interface (normally z=0 for FESOM-C)
    z_top = (1.0 - sigma_lev[0]) * H_col
    if z_particle <= z_top:
        return 0

    # Bottom interface (normally z=H_col for FESOM-C)
    z_bot = (1.0 - sigma_lev[K - 1]) * H_col
    if z_particle >= z_bot:
        return K - 2

    # Bracket search: layer k is [(1 - sigma[k])*H, (1 - sigma[k+1])*H)
    for k in range(K - 1):
        z_next = (1.0 - sigma_lev[k + 1]) * H_col
        if z_particle < z_next:
            return k

    return K - 2


# ---------------------------------------------------------------------------
# Vectorised layer detection (sigma)
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _loop_find_layers_sigma(
    sigma_lev: np.ndarray,
    z_particles: np.ndarray,
    nn_nodes: np.ndarray,
    node_depths: np.ndarray,
    eta_nodes: np.ndarray,
    alive_mask: np.ndarray,
    Dmin: float,
    layers_out: np.ndarray,
):
    P = z_particles.shape[0]
    for i in prange(P):
        if not alive_mask[i]:
            continue
        node = nn_nodes[i]
        H = node_depths[node] + eta_nodes[node]
        if H < Dmin:
            H = Dmin
        layers_out[i] = find_layer_sigma(sigma_lev, z_particles[i], H)


@njit(parallel=True, cache=True)
def _loop_find_layers_sigma_H(sigma_lev, z_particles, H_particle, alive_mask,
                              Dmin, layers_out):
    P = z_particles.shape[0]
    for i in prange(P):
        if not alive_mask[i]:
            continue
        H = H_particle[i]
        if H < Dmin:
            H = Dmin
        layers_out[i] = find_layer_sigma(sigma_lev, z_particles[i], H)


def find_layers_sigma(
    sigma_lev: np.ndarray,
    z_particles: np.ndarray,
    nn_nodes: np.ndarray,
    node_depths: np.ndarray,
    eta_nodes: np.ndarray,
    alive_mask: np.ndarray,
    Dmin: float = 0.5,
    H_particle: "np.ndarray | None" = None,
) -> np.ndarray:
    """Per-particle sigma layer index using local water-column geometry.

    Parameters
    ----------
    sigma_lev : np.ndarray, shape ``(K,)``
        Native sigma interfaces (FESOM-C: 1 surface → 0 bottom).
    z_particles : np.ndarray, shape ``(P,)``
        Particle depths (positive-down metres).
    nn_nodes : np.ndarray, shape ``(P,)``
        Nearest node index per particle.
    node_depths : np.ndarray, shape ``(N,)``
        Bottom depth per node (positive-down metres).
    eta_nodes : np.ndarray, shape ``(N,)``
        Free-surface elevation per node.  Pass zeros when ``eta`` is
        unavailable.
    alive_mask : np.ndarray of bool, shape ``(P,)``
        True for active particles.
    Dmin : float
        Minimum column thickness in metres.
    H_particle : np.ndarray, shape ``(P,)``, optional
        The particle's own column height (the sigma integrators interpolate
        it like W, ``sigma_column_height``); default: the nearest node's.

    Returns
    -------
    np.ndarray of int32, shape ``(P,)``
        Mid-layer index ``[0, K-2]``.  Zero for dead particles.
    """
    P = z_particles.shape[0]
    layers = np.zeros(P, dtype=np.int32)
    if H_particle is not None:
        _loop_find_layers_sigma_H(
            sigma_lev, z_particles, np.asarray(H_particle, dtype=np.float64),
            alive_mask, float(Dmin), layers)
        return layers
    _loop_find_layers_sigma(
        sigma_lev, z_particles, nn_nodes, node_depths, eta_nodes,
        alive_mask, float(Dmin), layers,
    )
    return layers


# ---------------------------------------------------------------------------
# Depth clamping (sigma)
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _loop_clamp_depth_sigma(
    z: np.ndarray,
    nn_nodes: np.ndarray,
    node_depths: np.ndarray,
    eta_nodes: np.ndarray,
    alive_mask: np.ndarray,
    Dmin: float,
    z0b_min: float,
    z_out: np.ndarray,
):
    P = z.shape[0]
    for i in prange(P):
        if not alive_mask[i]:
            z_out[i] = z[i]
            continue
        node = nn_nodes[i]
        H = node_depths[node] + eta_nodes[node]
        if H < Dmin:
            H = Dmin
        # Surface boundary (positive-down convention → surface is z = 0)
        z_val = z[i]
        if z_val < 0.0:
            z_val = 0.0
        # Bottom boundary: keep at least z0b_min above the local bottom
        bottom = H - z0b_min
        if bottom < 0.0:
            bottom = 0.0
        if z_val > bottom:
            z_val = bottom
        z_out[i] = z_val


@njit(parallel=True, cache=True)
def _loop_clamp_depth_sigma_H(z, H_particle, alive_mask, Dmin, z0b_min,
                              z_out):
    P = z.shape[0]
    for i in prange(P):
        if not alive_mask[i]:
            z_out[i] = z[i]
            continue
        H = H_particle[i]
        if H < Dmin:
            H = Dmin
        z_val = z[i]
        if z_val < 0.0:
            z_val = 0.0
        bottom = H - z0b_min
        if bottom < 0.0:
            bottom = 0.0
        if z_val > bottom:
            z_val = bottom
        z_out[i] = z_val


def clamp_depth_sigma(
    z: np.ndarray,
    nn_nodes: np.ndarray,
    node_depths: np.ndarray,
    eta_nodes: np.ndarray,
    alive_mask: np.ndarray,
    Dmin: float = 0.5,
    z0b_min: float = 0.1,
    H_particle: "np.ndarray | None" = None,
) -> np.ndarray:
    """Clamp particle depths to a local sigma water column.

    Surface boundary is fixed at ``z = 0`` (positive-down convention).
    Bottom boundary is the column height minus ``z0b_min``, ensuring
    particles sit strictly above the bed; the column is ``H_particle``
    when given (the particle's interpolated column, used by the sigma
    integrators), else ``depth + eta`` at the nearest node.

    Dead particles pass through unchanged.
    """
    P = z.shape[0]
    z_out = np.empty(P, dtype=np.float64)
    if H_particle is not None:
        _loop_clamp_depth_sigma_H(
            z, np.asarray(H_particle, dtype=np.float64), alive_mask,
            float(Dmin), float(z0b_min), z_out)
        return z_out
    _loop_clamp_depth_sigma(
        z, nn_nodes, node_depths, eta_nodes, alive_mask,
        float(Dmin), float(z0b_min), z_out,
    )
    return z_out


# ---------------------------------------------------------------------------
# Vertical W interpolation (sigma, interface-based, linear)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _interp_w_sigma_linear_single(
    w_field: np.ndarray,
    sigma_lev: np.ndarray,
    z: float,
    node: int,
    H_col: float,
    time_dir: float,
) -> float:
    """Linear interpolation of interface-based W along the local column."""
    K = sigma_lev.shape[0]
    if K < 2:
        return time_dir * w_field[0, node]

    z_top = (1.0 - sigma_lev[0]) * H_col
    z_bot = (1.0 - sigma_lev[K - 1]) * H_col

    if z <= z_top:
        return time_dir * w_field[0, node]
    if z >= z_bot:
        return time_dir * w_field[K - 1, node]

    # Bracket
    k = K - 2
    for j in range(K - 1):
        z_next = (1.0 - sigma_lev[j + 1]) * H_col
        if z < z_next:
            k = j
            break

    z_k = (1.0 - sigma_lev[k]) * H_col
    z_k1 = (1.0 - sigma_lev[k + 1]) * H_col
    dz = z_k1 - z_k
    if dz < 1e-12:
        return time_dir * w_field[k, node]

    alpha = (z - z_k) / dz
    w_val = (1.0 - alpha) * w_field[k, node] + alpha * w_field[k + 1, node]
    return time_dir * w_val


@njit(parallel=True, cache=True)
def _loop_interp_w_sigma_linear(
    w_field: np.ndarray,
    sigma_lev: np.ndarray,
    z_particles: np.ndarray,
    nn_nodes: np.ndarray,
    node_depths: np.ndarray,
    eta_nodes: np.ndarray,
    alive_mask: np.ndarray,
    Dmin: float,
    time_dir: float,
    W_out: np.ndarray,
):
    P = z_particles.shape[0]
    for i in prange(P):
        if not alive_mask[i]:
            continue
        node = nn_nodes[i]
        H = node_depths[node] + eta_nodes[node]
        if H < Dmin:
            H = Dmin
        W_out[i] = _interp_w_sigma_linear_single(
            w_field, sigma_lev, z_particles[i], node, H, time_dir,
        )


def interp_w_sigma_linear(
    w_field: np.ndarray,
    sigma_lev: np.ndarray,
    z_particles: np.ndarray,
    nn_nodes: np.ndarray,
    node_depths: np.ndarray,
    eta_nodes: np.ndarray,
    alive_mask: np.ndarray,
    Dmin: float = 0.5,
    time_dir: float = 1.0,
) -> np.ndarray:
    """Linear interpolation of W between bracketing sigma interfaces.

    ``w_field`` is interface-centred with shape ``(K, N)``.  At each
    particle the local water-column thickness is computed from the
    nearest node's ``depth + eta`` (floor ``Dmin``); the particle depth
    is then bracketed by the two surrounding sigma interfaces and
    W is linearly interpolated between the node values at
    those interfaces.

    Parameters
    ----------
    w_field : np.ndarray, shape ``(K, N)``
    sigma_lev : np.ndarray, shape ``(K,)``
    z_particles : np.ndarray, shape ``(P,)``
    nn_nodes : np.ndarray, shape ``(P,)``
    node_depths : np.ndarray, shape ``(N,)``
    eta_nodes : np.ndarray, shape ``(N,)``
    alive_mask : np.ndarray of bool, shape ``(P,)``
    Dmin : float
        Column-thickness floor.
    time_dir : float
        Forward (+1) or backward (-1) integration sign.

    Returns
    -------
    np.ndarray, shape ``(P,)``
        Vertical velocity at particle depths (m/s).
    """
    P = z_particles.shape[0]
    W = np.zeros(P, dtype=np.float64)
    _loop_interp_w_sigma_linear(
        w_field, sigma_lev, z_particles, nn_nodes, node_depths, eta_nodes,
        alive_mask, float(Dmin), float(time_dir), W,
    )
    return W


# ---------------------------------------------------------------------------
# Horizontal W interpolation: shared neighbour stencil
# ---------------------------------------------------------------------------

_MAX_NEIGHBOURS = 64


@njit(cache=True)
def _collect_neighbour_nodes_sigma(
    node_idx: int,
    node_to_elems: np.ndarray,
    node_to_elems_count: np.ndarray,
    elem_nodes: np.ndarray,
):
    """Collect unique neighbour nodes around ``node_idx`` (1-ring stencil).

    Duplicate of ``vertical._collect_neighbour_nodes`` — kept here so
    Numba can jit the sigma horizontal kernels without importing the
    z-level module (which would otherwise try to compile together).
    """
    buf = np.empty(_MAX_NEIGHBOURS, dtype=np.int32)
    count = 0
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
# Column height at the particle (sigma-following sampling, review 2026-09-23)
# ---------------------------------------------------------------------------
#
# Fields of a sigma model are smooth along the sigma surfaces, not along
# levels of constant depth: combining the neighbour columns at the particle's
# DEPTH mixes different sigma levels wherever the bottom slopes, and a
# particle carried along a sigma surface drifts off it (review round 2).  The
# kernels therefore evaluate every neighbour at the particle's sigma,
# ``z_j = (z / H_p) * H_j``, with ``H_p`` the column height interpolated at
# the particle by the SAME horizontal method (exact for a linear bottom with
# leastsq and barycentric).  The own-node fallbacks are unchanged
# (``z / H_p * H_p = z``).

@njit(cache=True)
def _col_H(node: int, node_depths: np.ndarray, eta_nodes: np.ndarray,
           Dmin: float) -> float:
    H = node_depths[node] + eta_nodes[node]
    if H < Dmin:
        H = Dmin
    return H


@njit(cache=True)
def _sigma_frac(z: float, H_p: float) -> float:
    """``1 - sigma`` of depth ``z`` in a column of height ``H_p``, in [0, 1]."""
    f = z / H_p
    if f < 0.0:
        f = 0.0
    elif f > 1.0:
        f = 1.0
    return f


@njit(cache=True)
def _H_leastsq_single(xp, yp, node_idx, node_depths, eta_nodes, Dmin,
                      nodes_lon, nodes_lat, node_to_elems,
                      node_to_elems_count, elem_nodes):
    """Plane fit of the column height over the 1-ring (own node on failure)."""
    SLM = 111120.0
    DEG2R = 0.017453292519943295
    buf, count = _collect_neighbour_nodes_sigma(
        node_idx, node_to_elems, node_to_elems_count, elem_nodes)
    if count < 3:
        return _col_H(node_idx, node_depths, eta_nodes, Dmin)
    z0 = np.cos(yp * DEG2R) * SLM
    if z0 < 1.0:
        z0 = 1.0
    a11 = 0.0; a12 = 0.0; a13 = 0.0
    a22 = 0.0; a23 = 0.0; a33 = 0.0
    fw = 0.0; f1w = 0.0; f2w = 0.0
    for i in range(count):
        nd = buf[i]
        h = _col_H(nd, node_depths, eta_nodes, Dmin)
        dlon = ((nodes_lon[nd] - xp + 180.0) % 360.0) - 180.0
        ex = dlon * z0
        ey = (nodes_lat[nd] - yp) * SLM
        a11 += 1.0; a12 += ex; a13 += ey
        a22 += ex * ex; a23 += ex * ey; a33 += ey * ey
        fw += h; f1w += h * ex; f2w += h * ey
    a21 = a12; a31 = a13; a32 = a23
    det = (a11 * a22 * a33 + a12 * a23 * a31 + a13 * a21 * a32
           - a13 * a22 * a31 - a12 * a21 * a33 - a11 * a23 * a32)
    if abs(det) < 1e-30 or abs(det) < 1e-12 * abs(a11 * a22 * a33):
        return _col_H(node_idx, node_depths, eta_nodes, Dmin)
    inv = 1.0 / abs(det)
    b11 = a22 * a33 - a23 * a32
    b12 = a13 * a32 - a12 * a33
    b13 = a12 * a23 - a13 * a22
    H = (b11 * fw + b12 * f1w + b13 * f2w) * inv
    if H < Dmin:
        H = Dmin
    return H


@njit(cache=True)
def _H_idw_single(xp, yp, node_idx, node_depths, eta_nodes, Dmin,
                  nodes_lon, nodes_lat, node_to_elems,
                  node_to_elems_count, elem_nodes):
    """Inverse-distance-weighted column height (own node on failure)."""
    SLM = 111120.0
    DEG2R = 0.017453292519943295
    buf, count = _collect_neighbour_nodes_sigma(
        node_idx, node_to_elems, node_to_elems_count, elem_nodes)
    if count < 1:
        return _col_H(node_idx, node_depths, eta_nodes, Dmin)
    z0 = np.cos(yp * DEG2R) * SLM
    if z0 < 1.0:
        z0 = 1.0
    sw = 0.0
    swh = 0.0
    for i in range(count):
        nd = buf[i]
        h = _col_H(nd, node_depths, eta_nodes, Dmin)
        dlon = ((nodes_lon[nd] - xp + 180.0) % 360.0) - 180.0
        dx = dlon * z0
        dy = (nodes_lat[nd] - yp) * SLM
        d2 = dx * dx + dy * dy
        if d2 < 1.0:
            return h
        sw += 1.0 / d2
        swh += h / d2
    if sw < 1e-30:
        return _col_H(node_idx, node_depths, eta_nodes, Dmin)
    return swh / sw


@njit(cache=True)
def _bary_locate(xp, yp, node_idx, nodes_lon, nodes_lat, node_to_elems,
                 node_to_elems_count, elem_nodes):
    """Containing element among the nearest node's elements, with vertex
    weights: ``(verts, weights, n_valid)``; ``n_valid == 0`` = not found.
    Tris: the fan barycentric weights; quads: product-form Wachspress."""
    DEG2R = 0.017453292519943295
    max_v = elem_nodes.shape[1]
    verts = np.empty(max_v, dtype=np.int32)
    wts = np.zeros(max_v, dtype=np.float64)
    n_elems = int(node_to_elems_count[node_idx])
    for k in range(n_elems):
        e = int(node_to_elems[node_idx, k])
        if e < 0:
            continue
        n_valid = 0
        for v in range(max_v):
            nd = int(elem_nodes[e, v])
            if nd < 0:
                continue
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
        R = 6371000.0
        vx = np.empty(n_valid, dtype=np.float64)
        vy = np.empty(n_valid, dtype=np.float64)
        for v in range(n_valid):
            dlon = ((nodes_lon[verts[v]] - lon0 + 180.0) % 360.0) - 180.0
            vx[v] = dlon * DEG2R * R * cos_lat0
            vy[v] = (nodes_lat[verts[v]] - lat0) * DEG2R * R
        dlon_p = ((xp - lon0 + 180.0) % 360.0) - 180.0
        px = dlon_p * DEG2R * R * cos_lat0
        py = (yp - lat0) * DEG2R * R
        for t in range(n_valid - 2):
            ax, ay = vx[0], vy[0]
            bx, by = vx[t + 1], vy[t + 1]
            cx, cy = vx[t + 2], vy[t + 2]
            denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
            if abs(denom) < 1e-14:
                continue
            w1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denom
            w2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denom
            w3 = 1.0 - w1 - w2
            eps = -1e-6
            if w1 >= eps and w2 >= eps and w3 >= eps:
                for v in range(max_v):
                    wts[v] = 0.0
                if n_valid == 4:
                    q = _polygon_weights(px, py, vx, vy, n_valid)
                    for v in range(4):
                        wts[v] = q[v]
                else:
                    wts[0] = w1
                    wts[t + 1] = w2
                    wts[t + 2] = w3
                return verts, wts, n_valid
    return verts, wts, 0


@njit(cache=True)
def _H_bary_single(xp, yp, node_idx, node_depths, eta_nodes, Dmin,
                   nodes_lon, nodes_lat, node_to_elems,
                   node_to_elems_count, elem_nodes):
    """Column height blended in the containing element (own node if none)."""
    verts, wts, n_valid = _bary_locate(
        xp, yp, node_idx, nodes_lon, nodes_lat, node_to_elems,
        node_to_elems_count, elem_nodes)
    if n_valid == 0:
        return _col_H(node_idx, node_depths, eta_nodes, Dmin)
    H = 0.0
    for v in range(n_valid):
        H += wts[v] * _col_H(verts[v], node_depths, eta_nodes, Dmin)
    return H


# ---------------------------------------------------------------------------
# Horizontal interpolation: least-squares planar fit (sigma)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _interp_w_sigma_leastsq_single(
    xp: float,
    yp: float,
    z: float,
    node_idx: int,
    w_field: np.ndarray,
    sigma_lev: np.ndarray,
    time_dir: float,
    node_depths: np.ndarray,
    eta_nodes: np.ndarray,
    Dmin: float,
    nodes_lon: np.ndarray,
    nodes_lat: np.ndarray,
    node_to_elems: np.ndarray,
    node_to_elems_count: np.ndarray,
    elem_nodes: np.ndarray,
):
    """Least-squares planar fit of W at particle position (sigma backend).

    Collects 1-ring neighbour nodes, samples W in each column at the
    particle's SIGMA (``z_j = (z / H_p) H_j``, ``H_p`` the plane-fitted
    column height at the particle, ``H = max(depth + eta, Dmin)``) and fits
    ``W = a0 + a1*x + a2*y`` via Cramer's rule.  Sigma-following since
    2026-09-23: sampling at the particle's depth mixed sigma levels on a
    sloping bottom and moved particles off their sigma surface.
    """
    SLM = 111120.0
    DEG2R = 0.017453292519943295

    buf, count = _collect_neighbour_nodes_sigma(
        node_idx, node_to_elems, node_to_elems_count, elem_nodes
    )

    if count < 3:
        H = node_depths[node_idx] + eta_nodes[node_idx]
        if H < Dmin:
            H = Dmin
        z_clamped = z if z < H else H
        w_val = _interp_w_sigma_linear_single(
            w_field, sigma_lev, z_clamped, node_idx, H, time_dir
        )
        return w_val, False

    z0 = np.cos(yp * DEG2R) * SLM
    if z0 < 1.0:
        z0 = 1.0
    # sigma-following: every neighbour at the particle's sigma
    frac = _sigma_frac(z, _H_leastsq_single(
        xp, yp, node_idx, node_depths, eta_nodes, Dmin, nodes_lon,
        nodes_lat, node_to_elems, node_to_elems_count, elem_nodes))

    a11 = 0.0; a12 = 0.0; a13 = 0.0
    a22 = 0.0; a23 = 0.0; a33 = 0.0
    fw = 0.0; f1w = 0.0; f2w = 0.0

    for i in range(count):
        nd = buf[i]
        H = node_depths[nd] + eta_nodes[nd]
        if H < Dmin:
            H = Dmin
        z_eval = frac * H
        w_val = _interp_w_sigma_linear_single(
            w_field, sigma_lev, z_eval, nd, H, time_dir
        )

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
        H = node_depths[node_idx] + eta_nodes[node_idx]
        if H < Dmin:
            H = Dmin
        z_clamped = z if z < H else H
        w_val = _interp_w_sigma_linear_single(
            w_field, sigma_lev, z_clamped, node_idx, H, time_dir
        )
        return w_val, False

    inv = 1.0 / abs(det)
    b11 = a22 * a33 - a23 * a32
    b12 = a13 * a32 - a12 * a33
    b13 = a12 * a23 - a13 * a22
    a0W = (b11 * fw + b12 * f1w + b13 * f2w) * inv
    return a0W, True


# ---------------------------------------------------------------------------
# Horizontal interpolation: barycentric in containing element (sigma)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _interp_w_sigma_barycentric_single(
    xp: float,
    yp: float,
    z: float,
    node_idx: int,
    w_field: np.ndarray,
    sigma_lev: np.ndarray,
    time_dir: float,
    node_depths: np.ndarray,
    eta_nodes: np.ndarray,
    Dmin: float,
    nodes_lon: np.ndarray,
    nodes_lat: np.ndarray,
    node_to_elems: np.ndarray,
    node_to_elems_count: np.ndarray,
    elem_nodes: np.ndarray,
):
    """Barycentric interpolation of sigma-column W inside the containing
    element (tris: barycentric weights, quads: Wachspress).  Each vertex
    column is sampled at the particle's SIGMA, ``z_v = (z / H_p) H_v`` with
    ``H_p`` the column height blended with the same weights
    (sigma-following, 2026-09-23); outside every element of the nearest
    node the own column is used at ``min(z, H)``."""
    verts, wts, n_valid = _bary_locate(
        xp, yp, node_idx, nodes_lon, nodes_lat, node_to_elems,
        node_to_elems_count, elem_nodes)
    if n_valid == 0:
        H = node_depths[node_idx] + eta_nodes[node_idx]
        if H < Dmin:
            H = Dmin
        z_clamped = z if z < H else H
        w_val = _interp_w_sigma_linear_single(
            w_field, sigma_lev, z_clamped, node_idx, H, time_dir
        )
        return w_val, False
    H_p = 0.0
    for v in range(n_valid):
        H_p += wts[v] * _col_H(verts[v], node_depths, eta_nodes, Dmin)
    frac = _sigma_frac(z, H_p)
    acc = 0.0
    for v in range(n_valid):
        nv = verts[v]
        Hv = _col_H(nv, node_depths, eta_nodes, Dmin)
        acc += wts[v] * _interp_w_sigma_linear_single(
            w_field, sigma_lev, frac * Hv, nv, Hv, time_dir)
    return acc, True


# ---------------------------------------------------------------------------
# Horizontal interpolation: inverse-distance weighting (sigma)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _interp_w_sigma_idw_single(
    xp: float,
    yp: float,
    z: float,
    node_idx: int,
    w_field: np.ndarray,
    sigma_lev: np.ndarray,
    time_dir: float,
    node_depths: np.ndarray,
    eta_nodes: np.ndarray,
    Dmin: float,
    nodes_lon: np.ndarray,
    nodes_lat: np.ndarray,
    node_to_elems: np.ndarray,
    node_to_elems_count: np.ndarray,
    elem_nodes: np.ndarray,
):
    """Inverse-distance-weighted interpolation of W (sigma backend);
    neighbours sampled at the particle's sigma (see the leastsq kernel)."""
    SLM = 111120.0
    DEG2R = 0.017453292519943295
    EPS_DIST = 1.0

    buf, count = _collect_neighbour_nodes_sigma(
        node_idx, node_to_elems, node_to_elems_count, elem_nodes
    )

    if count < 1:
        H = node_depths[node_idx] + eta_nodes[node_idx]
        if H < Dmin:
            H = Dmin
        z_clamped = z if z < H else H
        w_val = _interp_w_sigma_linear_single(
            w_field, sigma_lev, z_clamped, node_idx, H, time_dir
        )
        return w_val, False

    z0 = np.cos(yp * DEG2R) * SLM
    if z0 < 1.0:
        z0 = 1.0
    # sigma-following: every neighbour at the particle's sigma
    frac = _sigma_frac(z, _H_idw_single(
        xp, yp, node_idx, node_depths, eta_nodes, Dmin, nodes_lon,
        nodes_lat, node_to_elems, node_to_elems_count, elem_nodes))

    sum_w = 0.0
    sum_wv = 0.0

    for i in range(count):
        nd = buf[i]
        H = node_depths[nd] + eta_nodes[nd]
        if H < Dmin:
            H = Dmin
        z_eval = frac * H
        w_val = _interp_w_sigma_linear_single(
            w_field, sigma_lev, z_eval, nd, H, time_dir
        )

        dlon = ((nodes_lon[nd] - xp + 180.0) % 360.0) - 180.0
        dx = dlon * z0
        dy = (nodes_lat[nd] - yp) * SLM
        d2 = dx * dx + dy * dy

        if d2 < EPS_DIST * EPS_DIST:
            return w_val, True

        weight = 1.0 / d2
        sum_w += weight
        sum_wv += weight * w_val

    if sum_w < 1e-30:
        H = node_depths[node_idx] + eta_nodes[node_idx]
        if H < Dmin:
            H = Dmin
        z_clamped = z if z < H else H
        w_val = _interp_w_sigma_linear_single(
            w_field, sigma_lev, z_clamped, node_idx, H, time_dir
        )
        return w_val, False

    return sum_wv / sum_w, True


# ---------------------------------------------------------------------------
# Vectorised parallel loops
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _loop_interp_w_sigma_leastsq(
    w_field, sigma_lev, z_particles, nn_nodes, alive_mask, time_dir,
    node_depths, eta_nodes, Dmin,
    nodes_lon, nodes_lat, lons, lats,
    node_to_elems, node_to_elems_count, elem_nodes, W_out,
):
    P = z_particles.shape[0]
    for i in prange(P):
        if not alive_mask[i]:
            continue
        w, _ = _interp_w_sigma_leastsq_single(
            lons[i], lats[i], z_particles[i], nn_nodes[i],
            w_field, sigma_lev, time_dir, node_depths, eta_nodes, Dmin,
            nodes_lon, nodes_lat,
            node_to_elems, node_to_elems_count, elem_nodes,
        )
        W_out[i] = w


@njit(parallel=True, cache=True)
def _loop_interp_w_sigma_barycentric(
    w_field, sigma_lev, z_particles, nn_nodes, alive_mask, time_dir,
    node_depths, eta_nodes, Dmin,
    nodes_lon, nodes_lat, lons, lats,
    node_to_elems, node_to_elems_count, elem_nodes, W_out,
):
    P = z_particles.shape[0]
    for i in prange(P):
        if not alive_mask[i]:
            continue
        w, _ = _interp_w_sigma_barycentric_single(
            lons[i], lats[i], z_particles[i], nn_nodes[i],
            w_field, sigma_lev, time_dir, node_depths, eta_nodes, Dmin,
            nodes_lon, nodes_lat,
            node_to_elems, node_to_elems_count, elem_nodes,
        )
        W_out[i] = w


@njit(parallel=True, cache=True)
def _loop_interp_w_sigma_idw(
    w_field, sigma_lev, z_particles, nn_nodes, alive_mask, time_dir,
    node_depths, eta_nodes, Dmin,
    nodes_lon, nodes_lat, lons, lats,
    node_to_elems, node_to_elems_count, elem_nodes, W_out,
):
    P = z_particles.shape[0]
    for i in prange(P):
        if not alive_mask[i]:
            continue
        w, _ = _interp_w_sigma_idw_single(
            lons[i], lats[i], z_particles[i], nn_nodes[i],
            w_field, sigma_lev, time_dir, node_depths, eta_nodes, Dmin,
            nodes_lon, nodes_lat,
            node_to_elems, node_to_elems_count, elem_nodes,
        )
        W_out[i] = w


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def interp_w_sigma_spatial(
    w_field: np.ndarray,
    sigma_lev: np.ndarray,
    z_particles: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    nn_nodes: np.ndarray,
    alive_mask: np.ndarray,
    node_depths: np.ndarray,
    eta_nodes: np.ndarray,
    Dmin: float = 0.5,
    time_dir: float = 1.0,
    method: str = "nearest",
    mesh=None,
) -> np.ndarray:
    """Horizontal W interpolation on sigma meshes.

    ``method`` is one of ``"nearest"``, ``"leastsq"``, ``"barycentric"``,
    ``"idw"``.  For anything other than ``"nearest"``, ``mesh`` must be
    provided (it carries node connectivity and coordinates).
    """
    P = z_particles.shape[0]
    W = np.zeros(P, dtype=np.float64)

    if method == "nearest":
        return interp_w_sigma_linear(
            w_field, sigma_lev, z_particles, nn_nodes, node_depths,
            eta_nodes, alive_mask, Dmin=Dmin, time_dir=time_dir,
        )

    if mesh is None:
        raise ValueError(f"mesh is required for sigma w_interp '{method}'")

    args = (
        w_field, sigma_lev, z_particles, nn_nodes, alive_mask,
        float(time_dir), node_depths, eta_nodes, float(Dmin),
        mesh.nodes_lon, mesh.nodes_lat, lons, lats,
        mesh.node_to_elems, mesh.node_to_elems_count, mesh.elem_nodes,
        W,
    )

    if method == "leastsq":
        _loop_interp_w_sigma_leastsq(*args)
    elif method == "barycentric":
        _loop_interp_w_sigma_barycentric(*args)
    elif method == "idw":
        _loop_interp_w_sigma_idw(*args)
    else:
        raise ValueError(
            f"Unknown sigma w_interp method '{method}'. "
            "Available: 'nearest', 'leastsq', 'barycentric', 'idw'"
        )

    return W


# ---------------------------------------------------------------------------
# Column height at particle positions (the particle's column in the sigma step)
# ---------------------------------------------------------------------------

_H_METHOD = {"nearest": 0, "leastsq": 1, "barycentric": 2, "idw": 3}


@njit(parallel=True, cache=True)
def _loop_sigma_column_height(method, lons, lats, nn_nodes, alive_mask,
                              node_depths, eta_nodes, Dmin, nodes_lon,
                              nodes_lat, node_to_elems, node_to_elems_count,
                              elem_nodes, H_out):
    P = lons.shape[0]
    for i in prange(P):
        if not alive_mask[i]:
            continue
        nd = nn_nodes[i]
        if method == 1:
            H_out[i] = _H_leastsq_single(
                lons[i], lats[i], nd, node_depths, eta_nodes, Dmin,
                nodes_lon, nodes_lat, node_to_elems, node_to_elems_count,
                elem_nodes)
        elif method == 2:
            H_out[i] = _H_bary_single(
                lons[i], lats[i], nd, node_depths, eta_nodes, Dmin,
                nodes_lon, nodes_lat, node_to_elems, node_to_elems_count,
                elem_nodes)
        elif method == 3:
            H_out[i] = _H_idw_single(
                lons[i], lats[i], nd, node_depths, eta_nodes, Dmin,
                nodes_lon, nodes_lat, node_to_elems, node_to_elems_count,
                elem_nodes)
        else:
            H_out[i] = _col_H(nd, node_depths, eta_nodes, Dmin)


def sigma_column_height(mesh, lons: np.ndarray, lats: np.ndarray,
                        nn_nodes: np.ndarray, eta_nodes: np.ndarray,
                        alive_mask: np.ndarray, method: str = "nearest",
                        Dmin: float = 0.5) -> np.ndarray:
    """Water-column height ``max(depth + eta, Dmin)`` at particle positions,
    interpolated with the same horizontal method as W (``nearest`` = the
    nearest node's column).  Entries of non-alive particles are 1.0.

    The sigma integrators advance ``s = z / H_p`` and place the particle at
    ``z = s * H_p`` with it (``numerics/integrators.py:_sigma_col_H``).
    """
    if method not in _H_METHOD:
        raise ValueError(
            f"Unknown sigma w_interp method '{method}'. "
            "Available: 'nearest', 'leastsq', 'barycentric', 'idw'")
    P = lons.shape[0]
    H = np.ones(P, dtype=np.float64)
    _loop_sigma_column_height(
        _H_METHOD[method], np.asarray(lons, dtype=np.float64),
        np.asarray(lats, dtype=np.float64),
        np.asarray(nn_nodes, dtype=np.int64), np.asarray(alive_mask),
        np.asarray(mesh.depth, dtype=np.float64),
        np.asarray(eta_nodes, dtype=np.float64), float(Dmin),
        mesh.nodes_lon, mesh.nodes_lat, mesh.node_to_elems,
        mesh.node_to_elems_count, mesh.elem_nodes, H)
    return H
