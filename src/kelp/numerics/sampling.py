"""Scalar field sampling at particle positions.

Samples node-based scalar fields of shape ``(N,)`` (2D), ``(K-1, N)``
(mid-layer 3D) or ``(K, N)`` (interface 3D) at arbitrary particle
positions.  Supports four horizontal interpolation methods:

    "nearest"    — value at the nearest node (fast, no spatial smoothing)
    "leastsq"    — planar least-squares fit over the 1-ring stencil
    "barycentric"— barycentric blend inside the containing element
    "idw"        — inverse-distance-weighted 1-ring average

Design principles
-----------------
- **Direction-free**: every result is a plain scalar, unaffected by the
  direction of integration.  Any delegation to W-interpolation machinery
  MUST hardcode ``time_dir=+1.0`` so no sign is baked in.
- **Fallback semantics**: every method falls back gracefully to the
  nearest-node value when the stencil is degenerate (< 3 neighbours,
  singular normal matrix, no containing element found).
- **NaN for dead/invalid particles**: dead particles (``alive=False``)
  or particles with ``nearest < 0`` produce ``NaN`` in the output.

See docs/field_sampling.md.

Numba hot-path kernels use the project's standard try/except fallback
so the module is importable without Numba (tests run either way).
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


from ..core.mesh import Mesh, find_nearest_nodes, _polygon_weights, _seam_safe_lon0
from .vertical import (
    _collect_neighbour_nodes,
    interp_w_spatial,
)
from .vertical_sigma import interp_w_sigma_spatial


# ---------------------------------------------------------------------------
# Method code constants (int flags for Numba — no string dispatch inside njit)
# ---------------------------------------------------------------------------

S_NEAREST = 0
S_LEASTSQ = 1
S_BARYCENTRIC = 2
S_IDW = 3

_METHOD_CODES = {
    "nearest": S_NEAREST,
    "leastsq": S_LEASTSQ,
    "barycentric": S_BARYCENTRIC,
    "idw": S_IDW,
}

_VALID_METHODS = sorted(_METHOD_CODES)


# ---------------------------------------------------------------------------
# Core 2D horizontal combine kernel
# ---------------------------------------------------------------------------
#
# Why a dedicated kernel instead of reusing vertical.py's per-method helpers
# (_interp_w_*_single): those bake vertical interpolation AND the integration
# ``time_dir`` sign into one fused W-kernel.  Scalar sampling is direction-free
# and needs the vertical interp done separately (with per-neighbour bottom-aware
# clamping in the 3D loops), so it operates on already-vertically-resolved node
# values — hence this standalone horizontal-only combine.

@njit(cache=True)
def _combine_horizontal(
    node_vals: np.ndarray,
    neigh: np.ndarray,
    n_neigh: int,
    lon_p: float,
    lat_p: float,
    nodes_lon: np.ndarray,
    nodes_lat: np.ndarray,
    nn: int,
    method_code: int,
    elem_nodes: np.ndarray,
    node_to_elems: np.ndarray,
    node_to_elems_count: np.ndarray,
) -> float:
    """Combine 1-ring node values at particle position using chosen method.

    Parameters
    ----------
    node_vals : np.ndarray, shape ``(n_neigh,)``
        Scalar values at each neighbour node (same order as ``neigh``).
    neigh : np.ndarray, shape ``(64,)``
        Fixed-size neighbour node index array (first ``n_neigh`` valid).
    n_neigh : int
        Number of valid entries in ``neigh``.
    lon_p, lat_p : float
        Particle position in degrees.
    nodes_lon, nodes_lat : np.ndarray, shape ``(N,)``
        Mesh node coordinates in degrees.
    nn : int
        Nearest node index (= neigh[0]).
    method_code : int
        One of ``S_NEAREST``, ``S_LEASTSQ``, ``S_BARYCENTRIC``, ``S_IDW``.
    elem_nodes : np.ndarray, shape ``(E, max_verts)``
        Element connectivity (0-based, -1 padded).
    node_to_elems, node_to_elems_count : np.ndarray
        Node-to-element adjacency.

    Returns
    -------
    float
        Interpolated scalar value.  Falls back to ``node_vals[0]`` (the
        nearest-node value) on any degenerate case.
    """
    SLM = 111120.0  # METERS_PER_DEG_LAT
    DEG2R = 0.017453292519943295  # DEG2RAD

    # Fallback value: nearest-node
    fallback = node_vals[0]

    if method_code == S_NEAREST:
        return fallback

    # -----------------------------------------------------------------
    # IDW: inverse-distance-squared weighting over the 1-ring
    # -----------------------------------------------------------------
    if method_code == S_IDW:
        z0 = np.cos(lat_p * DEG2R) * SLM
        if z0 < 1.0:
            z0 = 1.0

        sum_w = 0.0
        sum_wv = 0.0
        for i in range(n_neigh):
            nd = neigh[i]
            dlon = ((nodes_lon[nd] - lon_p + 180.0) % 360.0) - 180.0
            dx = dlon * z0
            dy = (nodes_lat[nd] - lat_p) * SLM
            d2 = dx * dx + dy * dy

            if d2 < 1e-12:
                # Coincident with a node — return its RAW value, even a
                # fill NaN: the sample point IS that node, and blending
                # the remaining ring would invent water on a dry spot
                # (same raw contract as method='nearest'; review
                # 2026-08-16).
                return node_vals[i]

            if not np.isfinite(node_vals[i]):
                # coastal/bottom fill value — skip, don't poison the sum
                continue

            w = 1.0 / d2
            sum_w += w
            sum_wv += w * node_vals[i]

        if sum_w < 1e-30:
            return fallback
        return sum_wv / sum_w

    # -----------------------------------------------------------------
    # BARYCENTRIC: find the containing element via the 1-ring of nn
    # -----------------------------------------------------------------
    if method_code == S_BARYCENTRIC:
        R = 6371000.0  # R_EARTH_STANDARD

        n_elems = int(node_to_elems_count[nn])
        max_v = elem_nodes.shape[1]

        for k in range(n_elems):
            e = int(node_to_elems[nn, k])
            if e < 0:
                continue

            # Collect valid distinct vertices
            verts = np.empty(max_v, dtype=np.int32)
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

            vx = np.empty(n_valid, dtype=np.float64)
            vy = np.empty(n_valid, dtype=np.float64)
            for v in range(n_valid):
                dlon = ((nodes_lon[verts[v]] - lon0 + 180.0) % 360.0) - 180.0
                dlat = nodes_lat[verts[v]] - lat0
                vx[v] = dlon * DEG2R * R * cos_lat0
                vy[v] = dlat * DEG2R * R

            dlon_p = ((lon_p - lon0 + 180.0) % 360.0) - 180.0
            dlat_p = lat_p - lat0
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
                    # PERFORMANCE GATE: quads (n_valid == 4) use Wachspress
                    # weights (all four vertices, smooth, order-independent);
                    # triangles keep the inline 3-vertex blend below.
                    if n_valid == 4:
                        # Product-form Wachspress: no sentinel, valid on
                        # edges and vertices too (guard chain removed
                        # 2026-07).
                        wts = _polygon_weights(px, py, vx, vy, n_valid)
                        acc = 0.0
                        ok = True
                        for vi in range(n_valid):
                            nvtx = verts[vi]
                            vval = node_vals[0]
                            fnd = False
                            for j in range(n_neigh):
                                if neigh[j] == nvtx:
                                    vval = node_vals[j]
                                    fnd = True
                                    break
                            if not fnd or not np.isfinite(vval):
                                # missing vertex (defensive) or a
                                # coastal/bottom fill vertex — fall back
                                ok = False
                                break
                            acc += wts[vi] * vval
                        if ok:
                            return acc
                        return fallback

                    # Triangle path: blend the (i0, i1, i2) sub-triangle
                    # vertices.
                    n0v = verts[i0]
                    n1v = verts[i1]
                    n2v = verts[i2]

                    # Look up values for these vertex nodes from neigh array
                    val0 = node_vals[0]  # will be overwritten if found
                    val1 = node_vals[0]
                    val2 = node_vals[0]
                    found0 = False; found1 = False; found2 = False
                    for j in range(n_neigh):
                        if neigh[j] == n0v:
                            val0 = node_vals[j]; found0 = True
                        if neigh[j] == n1v:
                            val1 = node_vals[j]; found1 = True
                        if neigh[j] == n2v:
                            val2 = node_vals[j]; found2 = True

                    if not (found0 and found1 and found2):
                        # Missing vertex in 1-ring — fallback
                        # defensive: unreachable — 1-ring construction guarantees all vertices of nn's elements are in neigh
                        return fallback
                    if not (np.isfinite(val0) and np.isfinite(val1)
                            and np.isfinite(val2)):
                        # coastal/bottom fill vertex — fall back to nearest
                        return fallback

                    return w1 * val0 + w2 * val1 + w3 * val2

        # No containing element found in 1-ring — fallback
        return fallback

    # -----------------------------------------------------------------
    # LEASTSQ: planar fit a0 + a1*dx + a2*dy over the 1-ring in
    # CENTERED metric coords (dx, dy relative to particle position —
    # eliminates catastrophic cancellation; see CATASTROPHIC_CANCELLATION.md)
    # -----------------------------------------------------------------
    if method_code == S_LEASTSQ:
        if n_neigh < 3:
            return fallback

        z0 = np.cos(lat_p * DEG2R) * SLM
        if z0 < 1.0:
            z0 = 1.0

        # Build symmetric 3×3 normal matrix and RHS
        a11 = 0.0; a12 = 0.0; a13 = 0.0
        a22 = 0.0; a23 = 0.0; a33 = 0.0
        f0 = 0.0; f1 = 0.0; f2 = 0.0

        n_fin = 0
        for i in range(n_neigh):
            nd = neigh[i]
            val = node_vals[i]
            if not np.isfinite(val):
                # coastal/bottom fill value — skip, don't poison the fit
                continue
            n_fin += 1

            # Centered metric coords relative to particle
            dlon = ((nodes_lon[nd] - lon_p + 180.0) % 360.0) - 180.0
            ex = dlon * z0
            ey = (nodes_lat[nd] - lat_p) * SLM

            a11 += 1.0
            a12 += ex
            a13 += ey
            a22 += ex * ex
            a23 += ex * ey
            a33 += ey * ey
            f0 += val
            f1 += val * ex
            f2 += val * ey

        if n_fin < 3:
            return fallback          # not enough finite points for a plane

        a21 = a12; a31 = a13; a32 = a23

        det = (a11 * a22 * a33
               + a12 * a23 * a31
               + a13 * a21 * a32
               - a13 * a22 * a31
               - a12 * a21 * a33
               - a11 * a23 * a32)

        # scale-relative singularity guard — see _interp_leastsq_core
        # (the node-1-ring stencil here is rarely rank-deficient, unlike
        # the element rings; hygiene alignment with the 2026-08 fix)
        if abs(det) < 1e-12 or abs(det) < 1e-12 * abs(a11 * a22 * a33):
            return fallback

        inv = 1.0 / abs(det)
        b11 = a22 * a33 - a23 * a32
        b12 = a13 * a32 - a12 * a33
        b13 = a12 * a23 - a13 * a22

        # At the particle (dx=dy=0), result is just a0
        a0 = (b11 * f0 + b12 * f1 + b13 * f2) * inv
        return a0

    # Unknown code — should not reach here
    return fallback


# ---------------------------------------------------------------------------
# 2D loop over all particles
# ---------------------------------------------------------------------------

@njit(cache=True)
def _loop_sample_2d(
    values: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    nearest: np.ndarray,
    alive: np.ndarray,
    method_code: int,
    nodes_lon: np.ndarray,
    nodes_lat: np.ndarray,
    elem_nodes: np.ndarray,
    node_to_elems: np.ndarray,
    node_to_elems_count: np.ndarray,
    out: np.ndarray,
) -> None:
    """Per-particle 2D sampling loop.

    Dead particles (``alive[i]=False``) or invalid nearest nodes
    (``nearest[i] < 0``) produce ``NaN`` in ``out``.
    """
    P = lons.shape[0]
    for i in range(P):
        if not alive[i] or nearest[i] < 0:
            out[i] = np.nan
            continue

        nn = int(nearest[i])
        buf, count = _collect_neighbour_nodes(
            nn, node_to_elems, node_to_elems_count, elem_nodes
        )

        # Gather scalar values for neighbours
        node_vals = np.empty(count, dtype=np.float64)
        for k in range(count):
            node_vals[k] = values[buf[k]]

        out[i] = _combine_horizontal(
            node_vals, buf, count,
            lons[i], lats[i],
            nodes_lon, nodes_lat, nn,
            method_code,
            elem_nodes, node_to_elems, node_to_elems_count,
        )


# ---------------------------------------------------------------------------
# 3D z-level mid-layer: vertical linear interp then horizontal combine
# ---------------------------------------------------------------------------

@njit(cache=True)
def _interp_mid_linear_single(
    field: np.ndarray,
    z_mids: np.ndarray,
    z: float,
    node: int,
) -> float:
    """Linearly interpolate between mid-layer values at a single node.

    Parameters
    ----------
    field : np.ndarray, shape ``(K-1, N)``
        Mid-layer scalar field.
    z_mids : np.ndarray, shape ``(K-1,)``
        Mid-layer depths (positive-down metres), monotonically increasing.
    z : float
        Depth to evaluate at (positive-down metres).
    node : int
        Node index.

    Returns
    -------
    float
        Interpolated scalar value.  Clamped to first/last layer if outside
        the mid-layer range.

    Notes
    -----
    Direction-free: no time_dir applied.
    """
    K1 = z_mids.shape[0]
    if K1 == 0:
        return 0.0
    if K1 == 1:
        return field[0, node]

    # Clamp above first midpoint
    if z <= z_mids[0]:
        return field[0, node]
    # Clamp below last midpoint
    if z >= z_mids[K1 - 1]:
        return field[K1 - 1, node]

    # Linear bracket
    for k in range(K1 - 1):
        if z < z_mids[k + 1]:
            dz = z_mids[k + 1] - z_mids[k]
            if dz < 1e-12:
                return field[k, node]
            alpha = (z - z_mids[k]) / dz
            return (1.0 - alpha) * field[k, node] + alpha * field[k + 1, node]

    return field[K1 - 1, node]


@njit(cache=True)
def _loop_sample_mid_3d(
    field: np.ndarray,
    z_mids: np.ndarray,
    zs: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    nearest: np.ndarray,
    alive: np.ndarray,
    method_code: int,
    node_depths: np.ndarray,
    nodes_lon: np.ndarray,
    nodes_lat: np.ndarray,
    elem_nodes: np.ndarray,
    node_to_elems: np.ndarray,
    node_to_elems_count: np.ndarray,
    out: np.ndarray,
) -> None:
    """Per-particle 3D mid-layer sampling loop (z-level backend).

    For each particle: collect 1-ring neighbour nodes, apply bottom-aware
    z-clamping per neighbour, vertically interpolate ``field`` at each
    neighbour's clamped depth, then combine horizontally.
    """
    P = lons.shape[0]
    for i in range(P):
        if not alive[i] or nearest[i] < 0:
            out[i] = np.nan
            continue

        nn = int(nearest[i])
        z_p = zs[i]

        buf, count = _collect_neighbour_nodes(
            nn, node_to_elems, node_to_elems_count, elem_nodes
        )

        node_vals = np.empty(count, dtype=np.float64)
        for k in range(count):
            nd = buf[k]
            # Bottom-aware clamping per neighbour
            z_eval = z_p if z_p < node_depths[nd] else node_depths[nd]
            node_vals[k] = _interp_mid_linear_single(field, z_mids, z_eval, nd)

        out[i] = _combine_horizontal(
            node_vals, buf, count,
            lons[i], lats[i],
            nodes_lon, nodes_lat, nn,
            method_code,
            elem_nodes, node_to_elems, node_to_elems_count,
        )


# ---------------------------------------------------------------------------
# 3D sigma mid-layer: per-column sigma-aware vertical interp then combine
# ---------------------------------------------------------------------------

@njit(cache=True)
def _loop_sample_mid_sigma(
    field: np.ndarray,
    sigma_lev: np.ndarray,
    zs: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    nearest: np.ndarray,
    alive: np.ndarray,
    method_code: int,
    node_depths: np.ndarray,
    eta_nodes: np.ndarray,
    Dmin: float,
    nodes_lon: np.ndarray,
    nodes_lat: np.ndarray,
    elem_nodes: np.ndarray,
    node_to_elems: np.ndarray,
    node_to_elems_count: np.ndarray,
    out: np.ndarray,
) -> None:
    """Per-particle 3D sigma mid-layer sampling loop.

    For each neighbour node: compute local column depth H, sigma mid-layer
    depths, clamp z to H, then vertically interpolate ``field``.
    """
    K1 = field.shape[0]  # K-1 mid-layers
    P = lons.shape[0]

    for i in range(P):
        if not alive[i] or nearest[i] < 0:
            out[i] = np.nan
            continue

        nn = int(nearest[i])
        z_p = zs[i]

        buf, count = _collect_neighbour_nodes(
            nn, node_to_elems, node_to_elems_count, elem_nodes
        )

        node_vals = np.empty(count, dtype=np.float64)
        for k in range(count):
            nd = buf[k]
            H = node_depths[nd] + eta_nodes[nd]
            if H < Dmin:
                H = Dmin
            # Defensive only: shadowed by the last-midpoint clamp of
            # _interp_mid_linear_single (z_mids = (1-sig_mid)*H < H always, so
            # any z >= H already clamps to the bottom layer with or without this
            # min(z, H) guard).  Kept for symmetry with the z-level path.
            z_eval = z_p if z_p < H else H

            # Build per-column mid-layer depths and interpolate
            # z_mid[m] = (1 - 0.5*(sigma[m] + sigma[m+1])) * H
            z_mids_col = np.empty(K1, dtype=np.float64)
            for m in range(K1):
                sig_mid = 0.5 * (sigma_lev[m] + sigma_lev[m + 1])
                z_mids_col[m] = (1.0 - sig_mid) * H

            node_vals[k] = _interp_mid_linear_single(
                field, z_mids_col, z_eval, nd
            )

        out[i] = _combine_horizontal(
            node_vals, buf, count,
            lons[i], lats[i],
            nodes_lon, nodes_lat, nn,
            method_code,
            elem_nodes, node_to_elems, node_to_elems_count,
        )


# ---------------------------------------------------------------------------
# Sigma 3D dispatcher (interface + mid-layer)
# ---------------------------------------------------------------------------

def _sample_sigma_3d(
    values: np.ndarray,
    mesh: Mesh,
    lons: np.ndarray,
    lats: np.ndarray,
    zs: np.ndarray,
    nearest: np.ndarray,
    alive: np.ndarray,
    eta: np.ndarray | None,
    method: str,
    Dmin: float = 0.5,
) -> np.ndarray:
    """Sample a sigma 3D field ``(K, N)`` or ``(K-1, N)`` at particle positions.

    Interface fields ``(K, N)`` delegate to ``interp_w_sigma_spatial`` with
    ``time_dir=+1.0`` (direction-free rule).  Mid-layer fields ``(K-1, N)``
    use ``_loop_sample_mid_sigma`` which computes per-neighbour-column sigma
    interface depths inline: ``z_mid[m] = (1 - 0.5*(σ[m]+σ[m+1])) * H_col``.

    Parameters
    ----------
    values : np.ndarray
        Shape ``(K, N)`` for interface fields, ``(K-1, N)`` for mid-layer.
    mesh : Mesh
        Sigma mesh (``mesh.vertical_coord == "sigma"``).
    lons, lats, zs : np.ndarray, shape ``(P,)``
        Particle positions.
    nearest : np.ndarray of int32, shape ``(P,)``
        Pre-computed nearest node indices.
    alive : np.ndarray of bool, shape ``(P,)``
    eta : np.ndarray or None, shape ``(N,)``
        Free-surface elevation per node.  Required for sigma meshes.
    method : str
        Interpolation method.

    Returns
    -------
    np.ndarray, shape ``(P,)``
    """
    if eta is None:
        raise ValueError(
            "eta (free-surface elevation) is required for sigma mesh sampling"
        )

    K = mesh.sigma.size
    P = lons.shape[0]
    out = np.full(P, np.nan)

    if values.shape[0] == K:
        # Interface field (K, N) — delegate to sigma W machinery with
        # time_dir=+1.0 (direction-free rule)
        out = interp_w_sigma_spatial(
            values,
            mesh.sigma,
            zs,
            lons,
            lats,
            nearest,
            alive,
            mesh.depth,
            eta,
            Dmin=Dmin,
            time_dir=+1.0,
            method=method,
            mesh=mesh,
        )
        # NaN-mask dead/invalid particles
        # Load-bearing for nearest<0: the delegated W kernels skip dead particles but still index nearest as-is — do not remove.
        out[~alive | (nearest < 0)] = np.nan
        return out

    elif values.shape[0] == K - 1:
        # Mid-layer field (K-1, N)
        method_code = _METHOD_CODES[method]
        out = np.empty(P, dtype=np.float64)
        _loop_sample_mid_sigma(
            values,
            mesh.sigma,
            zs,
            lons,
            lats,
            nearest,
            alive,
            method_code,
            mesh.depth,
            eta,
            Dmin,
            mesh.nodes_lon,
            mesh.nodes_lat,
            mesh.elem_nodes,
            mesh.node_to_elems,
            mesh.node_to_elems_count,
            out,
        )
        return out

    else:
        raise ValueError(
            f"sigma 3D field shape[0]={values.shape[0]} does not match "
            f"K={K} (interface) or K-1={K - 1} (mid-layer) from mesh.sigma"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sample_field(
    values: np.ndarray,
    mesh: Mesh,
    lons: np.ndarray,
    lats: np.ndarray,
    zs: np.ndarray | None = None,
    *,
    nearest: np.ndarray | None = None,
    alive: np.ndarray | None = None,
    eta: np.ndarray | None = None,
    method: str = "leastsq",
    Dmin: float = 0.5,
) -> np.ndarray:
    """Sample a scalar field at particle positions.

    Supports node-based fields of shape ``(N,)`` (2D), ``(K-1, N)``
    (3D mid-layer) or ``(K, N)`` (3D interface).  Four horizontal
    interpolation methods are provided (see module docstring).

    Direction-free: outputs are plain scalars, not multiplied by any
    integration-direction sign.  All internal delegations to W-machinery
    hardcode ``time_dir=+1.0``.

    Parameters
    ----------
    values : np.ndarray
        Scalar field, shape ``(N,)``, ``(K-1, N)``, or ``(K, N)``.
        The last axis must equal ``mesh.n_nodes``.  Element-based fields
        are not supported (raise ``ValueError``).
    mesh : Mesh
        Mesh with ``build_search`` already called.
    lons, lats : np.ndarray, shape ``(P,)``
        Particle longitude and latitude in degrees.
    zs : np.ndarray or None, shape ``(P,)``
        Particle depths (positive-down metres).  Required for 3D fields.
    nearest : np.ndarray of int32 or None, shape ``(P,)``
        Pre-computed nearest node indices.  If ``None``, computed via
        ``find_nearest_nodes``.  Pass ``nearest[i] = -1`` to force
        ``NaN`` for particle ``i``.
    alive : np.ndarray of bool or None, shape ``(P,)``
        Active-particle mask.  Defaults to all-True.  Dead particles
        (``alive[i] = False``) produce ``NaN`` in the output.
    eta : np.ndarray or None, shape ``(N,)``
        Free-surface elevation per node.  Required when ``mesh.vertical_coord
        == "sigma"`` and ``values.ndim == 2``.
    method : str
        Horizontal interpolation method: ``"nearest"``, ``"leastsq"``,
        ``"barycentric"``, or ``"idw"``.
    Dmin : float
        Minimum water-column thickness (m) of a sigma column,
        ``H = max(depth + eta, Dmin)`` -- pass the run's
        ``numerics.Dmin`` so the sample sits on the particle's column.
        Ignored on z-level meshes.

    Returns
    -------
    np.ndarray, shape ``(P,)``
        Sampled scalar values.  ``NaN`` for dead/invalid particles.

    Raises
    ------
    ValueError
        If ``method`` is not one of the four valid options (message contains
        ``"method"``).
    ValueError
        If ``values.shape[-1] != mesh.n_nodes`` (message contains ``"node"``).
    ValueError
        If a 3D field is requested but ``zs`` is ``None``.

    Notes
    -----
    Source-data NaNs (decoded ``_FillValue`` / masked cells, e.g. below-bottom
    padding) are FILTERED from the horizontal stencils (2026-08-16, review
    finding of 2026-06-05): leastsq skips non-finite neighbours and needs
    >= 3 finite points (else nearest fallback), idw skips them (>= 1), a
    barycentric containing-element with a non-finite vertex falls back to
    the nearest value.  The nearest value itself is returned RAW — if the
    particle's own node carries fill, the honest answer is that fill.
    All-finite stencils are bit-identical to the unfiltered path.
    LIMITATION: interface-shaped ``(K, N)`` sampled fields delegate to
    the W-interpolation kernels (``interp_w_spatial``), which do NOT
    filter — a poisoned neighbour there still yields NaN (interface
    scalars are rare; the W kernels' own contract is unchanged).
    """
    # -- Validate method ----------------------------------------------------
    if method not in _METHOD_CODES:
        raise ValueError(
            f"Unknown method '{method}'. Valid method values are: "
            f"{_VALID_METHODS}"
        )

    # -- Coerce inputs -------------------------------------------------------
    values = np.asarray(values, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    lats = np.asarray(lats, dtype=np.float64)
    P = lons.shape[0]

    # -- Validate field size -------------------------------------------------
    if values.shape[-1] != mesh.n_nodes:
        raise ValueError(
            f"values.shape[-1]={values.shape[-1]} does not match "
            f"mesh.n_nodes={mesh.n_nodes}. "
            "Element-based fields are not supported; provide a node-based field."
        )

    # -- Default nearest / alive --------------------------------------------
    if nearest is None:
        nearest = find_nearest_nodes(mesh, lons, lats)
    nearest = np.asarray(nearest, dtype=np.int32)

    if alive is None:
        alive = np.ones(P, dtype=np.bool_)
    alive = np.asarray(alive, dtype=np.bool_)

    # -- 2D field -----------------------------------------------------------
    if values.ndim == 1:
        out = np.empty(P, dtype=np.float64)
        method_code = _METHOD_CODES[method]
        _loop_sample_2d(
            values, lons, lats, nearest, alive, method_code,
            mesh.nodes_lon, mesh.nodes_lat,
            mesh.elem_nodes, mesh.node_to_elems, mesh.node_to_elems_count,
            out,
        )
        return out

    # -- 3D field -----------------------------------------------------------
    if values.ndim == 2:
        if zs is None:
            raise ValueError(
                "zs (particle depths) is required for 3D field sampling"
            )
        zs = np.asarray(zs, dtype=np.float64)

        # Sigma mesh: must come FIRST (mesh.z_levels raises for sigma)
        if mesh.vertical_coord == "sigma":
            return _sample_sigma_3d(
                values, mesh, lons, lats, zs, nearest, alive, eta, method,
                Dmin=Dmin,
            )

        # Z-level mesh
        K = mesh.sigma.size  # number of interface levels
        z_levels = mesh.z_levels  # shape (K,)

        if values.shape[0] == K:
            # Interface field (K, N) — delegate to W-interp machinery
            # time_dir=+1.0: direction-free rule
            out = interp_w_spatial(
                values,
                z_levels,
                zs,
                lons,
                lats,
                nearest,
                alive,
                time_dir=+1.0,
                method=method,
                mesh=mesh,
                node_depths=mesh.depth,
            )
            # NaN-mask dead/invalid
            # Load-bearing for nearest<0: the delegated W kernels skip dead particles but still index nearest as-is — do not remove.
            out[~alive | (nearest < 0)] = np.nan
            return out

        elif values.shape[0] == K - 1:
            # Mid-layer field (K-1, N)
            # Compute mid-layer depths from interface boundaries
            z_mids = 0.5 * (z_levels[:-1] + z_levels[1:])
            method_code = _METHOD_CODES[method]
            out = np.empty(P, dtype=np.float64)
            _loop_sample_mid_3d(
                values, z_mids, zs, lons, lats, nearest, alive,
                method_code,
                mesh.depth,
                mesh.nodes_lon, mesh.nodes_lat,
                mesh.elem_nodes, mesh.node_to_elems, mesh.node_to_elems_count,
                out,
            )
            return out

        else:
            raise ValueError(
                f"3D z-level field shape[0]={values.shape[0]} does not match "
                f"K={K} (interface) or K-1={K - 1} (mid-layer) from mesh.sigma"
            )

    raise ValueError(
        f"values must be 1D (N,) or 2D (K, N) or (K-1, N), got shape {values.shape}"
    )


__all__ = [
    "sample_field",
    "S_NEAREST",
    "S_LEASTSQ",
    "S_BARYCENTRIC",
    "S_IDW",
    "_METHOD_CODES",
    "_combine_horizontal",
    "_loop_sample_2d",
    "_interp_mid_linear_single",
    "_loop_sample_mid_3d",
    "_loop_sample_mid_sigma",
    "_sample_sigma_3d",
]
