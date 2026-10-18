"""Offline reconstruction of Cartesian vertical velocity for FESOM2.

FESOM2 often does not output vertical velocity W, or outputs w_ALE
(velocity relative to the moving ALE mesh) rather than the Cartesian
w_cart needed for Lagrangian tracking.

This module provides two reconstruction methods:

**Variant A (divergence)**: Integrates the continuity equation
``∂u/∂x + ∂v/∂y + ∂w/∂z = 0`` downward from the surface, using
``w_surf = -d(ssh)/dt`` as surface boundary condition.  This is an
*approximation* — the divergence is computed via weak-form shape
functions on the native triangular mesh, not via FESOM2's internal
edge-flux operator.

**Variant B (ALE heuristic)**: Corrects w_ALE by adding an estimate
of the grid velocity: ``w_cart ≈ w_ALE - (1 - D/H) × ssh_t``.
This is a *heuristic* — real z-star distributes the SSH change
non-linearly across layers, not via (1 - D/H).

Sign convention (positive-down, KELP's internal one; FESOM2's own w output is
positive UP and the readers negate it — docs/vertical.md §3):
- z increases downward (``mesh.sigma`` stores positive depths)
- w > 0 means downwelling (particle moves deeper)
- ssh rising → w_surf < 0 (particle moves up at surface)

See ``docs/w_reconstruction.md`` for the derivation and references.
"""
from __future__ import annotations

import math
import warnings
from typing import Dict, Optional

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

from .core.geometry import DEG2RAD, METERS_PER_DEG_LAT

_SLM = METERS_PER_DEG_LAT


# ---------------------------------------------------------------------------
# SSH tendency
# ---------------------------------------------------------------------------

def compute_ssh_tendency(
    eta_prev: np.ndarray,
    eta_next: np.ndarray,
    dt_real: float,
) -> np.ndarray:
    """Compute d(ssh)/dt using finite difference with actual time interval.

    Parameters
    ----------
    eta_prev : np.ndarray, shape (N,)
        SSH at the earlier timestep.
    eta_next : np.ndarray, shape (N,)
        SSH at the later timestep.
    dt_real : float
        Time interval between ``eta_prev`` and ``eta_next`` in seconds.
        Must be > 0.  For centered difference, this is ``t(i+1) - t(i-1)``.

    Returns
    -------
    np.ndarray, shape (N,)
        SSH tendency in m/s.
    """
    if dt_real <= 0:
        raise ValueError(f"dt_real must be positive, got {dt_real}")
    return (eta_next - eta_prev) / dt_real


# ---------------------------------------------------------------------------
# Tier 1 divergence: shape-function weak form (approximate)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _divergence_layer_single_node(
    n: int,
    u_layer: np.ndarray,
    v_layer: np.ndarray,
    elem_nodes: np.ndarray,
    nodes_lon: np.ndarray,
    nodes_lat: np.ndarray,
    node_to_elems: np.ndarray,
    node_to_elems_count: np.ndarray,
) -> float:
    """Compute weak-form divergence at a single node (Numba-compatible)."""
    lat_ref_rad = nodes_lat[n] * DEG2RAD
    cos_lat = math.cos(lat_ref_rad)
    if cos_lat < 1e-10:
        cos_lat = 1e-10

    flux_sum = 0.0
    area_sum = 0.0

    n_adj = node_to_elems_count[n]
    for ie in range(n_adj):
        e = node_to_elems[n, ie]
        if e < 0:
            break

        n0 = elem_nodes[e, 0]
        n1 = elem_nodes[e, 1]
        n2 = elem_nodes[e, 2]

        # Identify which vertex is node n, get the other two
        if n0 == n:
            i_next, i_prev = n1, n2
        elif n1 == n:
            i_next, i_prev = n2, n0
        else:
            i_next, i_prev = n0, n1

        # Metric coordinates (flat-Earth at node latitude).  Antimeridian
        # gate (2026-07, idiom of the leastsq cores): a raw neighbour
        # longitude more than 180 deg from node n means the element
        # straddles the seam — shift that neighbour by one full turn.
        # Off-seam elements never take the branch (bit-identical).
        lon_next = nodes_lon[i_next]
        d = lon_next - nodes_lon[n]
        if d > 180.0:
            lon_next -= 360.0
        elif d < -180.0:
            lon_next += 360.0
        lon_prev = nodes_lon[i_prev]
        d = lon_prev - nodes_lon[n]
        if d > 180.0:
            lon_prev -= 360.0
        elif d < -180.0:
            lon_prev += 360.0
        x_next = lon_next * cos_lat * _SLM
        y_next = nodes_lat[i_next] * _SLM
        x_prev = lon_prev * cos_lat * _SLM
        y_prev = nodes_lat[i_prev] * _SLM
        x_n = nodes_lon[n] * cos_lat * _SLM
        y_n = nodes_lat[n] * _SLM

        # Element area (signed)
        dx1 = x_next - x_n
        dy1 = y_next - y_n
        dx2 = x_prev - x_n
        dy2 = y_prev - y_n
        two_A = dx1 * dy2 - dx2 * dy1

        # Weak-form divergence: ∫ div(u) N_n = -∫ u·∇N_n + boundary
        # δ_n = -[u(y_next-y_prev) + v(x_prev-x_next)] / 2
        flux_sum -= (u_layer[e] * (y_next - y_prev)
                     + v_layer[e] * (x_prev - x_next)) / 2.0
        area_sum += abs(two_A) / 6.0  # |A_e| / 3

    if area_sum > 0:
        return flux_sum / area_sum
    return 0.0


@njit(parallel=True, cache=True)
def _divergence_layer_approx(
    u_layer: np.ndarray,
    v_layer: np.ndarray,
    N: int,
    elem_nodes: np.ndarray,
    nodes_lon: np.ndarray,
    nodes_lat: np.ndarray,
    node_to_elems: np.ndarray,
    node_to_elems_count: np.ndarray,
    div_out: np.ndarray,
):
    """Weak-form horizontal divergence at all nodes for one depth layer.

    This is an *approximation*: uses constant (surface) area per node
    and flat-Earth metric.  Accuracy degrades near steep topography
    and for regions larger than ~10°.
    """
    for n in prange(N):
        div_out[n] = _divergence_layer_single_node(
            n, u_layer, v_layer, elem_nodes,
            nodes_lon, nodes_lat, node_to_elems, node_to_elems_count,
        )


def require_layer_data(mesh, uv_vertical: str = "layer") -> None:
    """Refuse u/v that lie on the levels instead of the layers.

    The reconstruction integrates the divergence of layer-centred u/v
    (``K-1`` layers between ``K`` interfaces, FESOM2).  A FESOM 1.4 mesh
    (``mesh.type: fesom14``) carries u/v on the levels, and its output has
    the vertical velocity itself.

    Parameters
    ----------
    mesh : Mesh
        A mesh from a reader; one built by the FESOM 1.4 reader
        (``mesh.metadata["fesom14_source"]``) is refused.
    uv_vertical : str
        Where u/v lie (``VelocityField.uv_vertical``): ``"level"`` is
        refused; ``"layer"`` (the default) and ``"layer_linear"`` pass.

    Raises
    ------
    ValueError
        For a FESOM 1.4 mesh or u/v on the levels.
    """
    md = getattr(mesh, "metadata", None) or {}
    if "fesom14_source" in md:
        what = ("the mesh is a FESOM 1.4 mesh (mesh.type: fesom14), whose "
                "u/v lie on the levels")
    elif str(uv_vertical) == "level":
        what = "u/v lie on the levels (uv_vertical 'level')"
    else:
        return
    raise ValueError(
        f"W reconstruction: {what}. The reconstruction integrates layer "
        "u/v (K-1 layers between the K interfaces, FESOM2) only. FESOM 1.4 "
        "output carries the vertical velocity ('wo'): give its files to the "
        "run (mode.var_w names another variable) instead of reconstructing "
        "it")


def compute_divergence_approx(
    u: np.ndarray,
    v: np.ndarray,
    mesh,
    *,
    uv_vertical: str = "layer",
) -> np.ndarray:
    """Compute horizontal divergence for all depth layers (Tier 1).

    Parameters
    ----------
    u : np.ndarray, shape (K-1, E)
        Zonal velocity per element per layer (m/s).
    v : np.ndarray, shape (K-1, E)
        Meridional velocity per element per layer (m/s).
    mesh : Mesh
        Must have ``elem_nodes``, ``nodes_lon``, ``nodes_lat``,
        ``node_to_elems``, ``node_to_elems_count``.
    uv_vertical : str
        Where u/v lie; ``"level"`` (and a FESOM 1.4 mesh) is refused by
        :func:`require_layer_data`.

    Returns
    -------
    np.ndarray, shape (K-1, N)
        Horizontal divergence at each node for each layer (1/s).
    """
    require_layer_data(mesh, uv_vertical)
    n_layers = u.shape[0]
    N = mesh.n_nodes
    div_h = np.zeros((n_layers, N), dtype=np.float64)
    for k in range(n_layers):
        _divergence_layer_approx(
            u[k], v[k], N,
            mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat,
            mesh.node_to_elems, mesh.node_to_elems_count,
            div_h[k],
        )
    return div_h


# ---------------------------------------------------------------------------
# Tier 2 divergence: with depth-dependent area + element mask
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _divergence_layer_with_area(
    u_layer: np.ndarray,
    v_layer: np.ndarray,
    N: int,
    elem_nodes: np.ndarray,
    nodes_lon: np.ndarray,
    nodes_lat: np.ndarray,
    node_to_elems: np.ndarray,
    node_to_elems_count: np.ndarray,
    nod_area_layer: np.ndarray,
    nlevels_elem: np.ndarray,
    ulevels_elem: np.ndarray,
    layer_idx: int,
    div_out: np.ndarray,
):
    """Divergence with depth-dependent area and per-element vertical mask.

    Element e contributes only if ``ulevels_elem[e]-1 <= layer_idx < nlevels_elem[e]-1``.
    Normalization uses ``nod_area_layer[n]`` (from mesh diag) instead of
    the approximate surface cluster area.
    """
    for n in prange(N):
        area_n = nod_area_layer[n]
        if area_n <= 0:
            div_out[n] = 0.0
            continue

        lat_ref_rad = nodes_lat[n] * DEG2RAD
        cos_lat = math.cos(lat_ref_rad)
        if cos_lat < 1e-10:
            cos_lat = 1e-10

        flux_sum = 0.0
        n_adj = node_to_elems_count[n]
        for ie in range(n_adj):
            e = node_to_elems[n, ie]
            if e < 0:
                break

            # Per-element vertical mask: active on [ulevels-1, nlevels-1)
            if layer_idx < ulevels_elem[e] - 1:
                continue
            if layer_idx >= nlevels_elem[e] - 1:
                continue

            n0 = elem_nodes[e, 0]
            n1 = elem_nodes[e, 1]
            n2 = elem_nodes[e, 2]

            if n0 == n:
                i_next, i_prev = n1, n2
            elif n1 == n:
                i_next, i_prev = n2, n0
            else:
                i_next, i_prev = n0, n1

            # Same antimeridian gate as the Tier-1 kernel above.
            lon_next = nodes_lon[i_next]
            d = lon_next - nodes_lon[n]
            if d > 180.0:
                lon_next -= 360.0
            elif d < -180.0:
                lon_next += 360.0
            lon_prev = nodes_lon[i_prev]
            d = lon_prev - nodes_lon[n]
            if d > 180.0:
                lon_prev -= 360.0
            elif d < -180.0:
                lon_prev += 360.0
            x_next = lon_next * cos_lat * _SLM
            y_next = nodes_lat[i_next] * _SLM
            x_prev = lon_prev * cos_lat * _SLM
            y_prev = nodes_lat[i_prev] * _SLM

            flux_sum -= (u_layer[e] * (y_next - y_prev)
                         + v_layer[e] * (x_prev - x_next)) / 2.0

        div_out[n] = flux_sum / area_n


def compute_divergence_with_area(
    u: np.ndarray,
    v: np.ndarray,
    mesh,
    nod_area: np.ndarray,
    nlevels_elem: Optional[np.ndarray] = None,
    ulevels_elem: Optional[np.ndarray] = None,
    *,
    uv_vertical: str = "layer",
) -> np.ndarray:
    """Compute horizontal divergence with depth-dependent area (Tier 2).

    Parameters
    ----------
    u : np.ndarray, shape (K-1, E)
        Zonal velocity per element per layer (m/s).
    v : np.ndarray, shape (K-1, E)
        Meridional velocity per element per layer (m/s).
    mesh : Mesh
    nod_area : np.ndarray, shape (K-1, N) or (K, N)
        Depth-dependent control volume area from mesh diag.
    nlevels_elem : np.ndarray, shape (E,), optional
        Number of active levels per element.  If None, all layers active.
    ulevels_elem : np.ndarray, shape (E,), optional
        Upper active level per element (1-based).  If None, defaults to 1.
    uv_vertical : str
        Where u/v lie; ``"level"`` (and a FESOM 1.4 mesh) is refused by
        :func:`require_layer_data`.

    Returns
    -------
    np.ndarray, shape (K-1, N)
        Horizontal divergence at each node for each layer (1/s).
    """
    require_layer_data(mesh, uv_vertical)
    n_layers = u.shape[0]
    N = mesh.n_nodes
    E = mesh.n_elements
    div_h = np.zeros((n_layers, N), dtype=np.float64)

    if nlevels_elem is None:
        nlevels_elem = np.full(E, n_layers + 1, dtype=np.int32)
    if ulevels_elem is None:
        ulevels_elem = np.ones(E, dtype=np.int32)

    for k in range(n_layers):
        # nod_area may be (K, N) or (K-1, N); use min to avoid overflow
        area_idx = min(k, nod_area.shape[0] - 1)
        _divergence_layer_with_area(
            u[k], v[k], N,
            mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat,
            mesh.node_to_elems, mesh.node_to_elems_count,
            nod_area[area_idx],
            nlevels_elem, ulevels_elem, k,
            div_h[k],
        )
    return div_h


# ---------------------------------------------------------------------------
# Vertical integration
# ---------------------------------------------------------------------------

def integrate_continuity(
    div_h: np.ndarray,
    z_levels: np.ndarray,
    ssh_t: np.ndarray,
    node_depths: np.ndarray,
    nlevels_nod: Optional[np.ndarray] = None,
    correct_bottom: bool = True,
) -> np.ndarray:
    """Integrate continuity equation from surface downward.

    Positive-down convention:
    - ``w(0) = -ssh_t`` (ssh rising → w negative at surface)
    - ``w(k) = w(k-1) - div_h(k-1) × Δz(k-1)``

    Parameters
    ----------
    div_h : np.ndarray, shape (K-1, N)
        Horizontal divergence per layer per node.
    z_levels : np.ndarray, shape (K,)
        Depth of z-level interfaces (positive-down, metres).
    ssh_t : np.ndarray, shape (N,)
        SSH tendency d(ssh)/dt (m/s).
    node_depths : np.ndarray, shape (N,)
        Per-node bottom depth (positive-down, metres).
    nlevels_nod : np.ndarray, shape (N,), optional
        Active level count per node.  Layers below this are masked to 0.
    correct_bottom : bool
        If True, apply linear correction so w(H) = 0.

    Returns
    -------
    np.ndarray, shape (K, N)
        Vertical velocity at z-level interfaces (positive-down).
    """
    K = len(z_levels)
    N = len(ssh_t)
    w = np.zeros((K, N), dtype=np.float64)

    # Surface BC: w_surf = -ssh_t
    w[0, :] = -ssh_t

    # Integrate downward
    for k in range(1, K):
        dz = z_levels[k] - z_levels[k - 1]
        w[k, :] = w[k - 1, :] - div_h[k - 1, :] * dz

    # Linear bottom correction: w_corr(D) = w_raw(D) - (D/H) × w_raw(H), with
    # the residual read at the LOCAL bottom H (linear between the bracketing
    # interfaces — a partial bottom layer included) BEFORE any masking, and
    # applied to the interfaces at or above H only.  Until 2026-09-23 the
    # residual came from the global last interface (w(H) != 0 on shallow
    # nodes) and the correction re-filled masked below-bottom interfaces
    # (review 2026-09-22, F05).
    if correct_bottom:
        for n in range(N):
            H = node_depths[n]
            if H <= 0:
                continue
            R = float(np.interp(H, z_levels, w[:, n]))
            for k in range(K):
                if z_levels[k] <= H:
                    w[k, n] -= (z_levels[k] / H) * R
                else:
                    w[k, n] = 0.0          # interface in the ground

    # Mask inactive layers — after the correction, so nothing refills them
    if nlevels_nod is not None:
        for n in range(N):
            k_bot = int(nlevels_nod[n])
            if k_bot < K:
                w[k_bot:, n] = 0.0

    return w


# ---------------------------------------------------------------------------
# Variant B: ALE heuristic
# ---------------------------------------------------------------------------

def reconstruct_w_ale(
    w_ale: np.ndarray,
    z_levels: np.ndarray,
    ssh_t: np.ndarray,
    node_depths: np.ndarray,
) -> np.ndarray:
    """Reconstruct w_cart from w_ALE using ALE grid velocity heuristic.

    ``w_cart(D) ≈ w_ALE(D) - (1 - D/H) × ssh_t``

    This is a *heuristic* approximation.  Real FESOM2 z-star distributes
    the SSH change non-linearly across eligible layers, not via the
    linear profile ``(1 - D/H)``.

    Parameters
    ----------
    w_ale : np.ndarray, shape (K, N)
        ALE vertical velocity from FESOM2 output.
    z_levels : np.ndarray, shape (K,)
        Depth of z-level interfaces (positive-down, metres).
    ssh_t : np.ndarray, shape (N,)
        SSH tendency d(ssh)/dt (m/s).
    node_depths : np.ndarray, shape (N,)
        Per-node bottom depth (positive-down, metres).

    Returns
    -------
    np.ndarray, shape (K, N)
        Estimated Cartesian vertical velocity.
    """
    K, N = w_ale.shape
    w_cart = np.empty_like(w_ale)
    for k in range(K):
        D = z_levels[k]
        for n in range(N):
            H = max(node_depths[n], 1.0)
            w_cart[k, n] = w_ale[k, n] - (1.0 - D / H) * ssh_t[n]
    return w_cart


# ---------------------------------------------------------------------------
# Diagnostics (Variant C)
# ---------------------------------------------------------------------------

def compare_variants(
    w_div: np.ndarray,
    w_ale_cart: np.ndarray,
    z_levels: np.ndarray,
    ssh_t: np.ndarray,
    w_ale: np.ndarray,
    node_depths: np.ndarray,
) -> Dict[str, float]:
    """Compare divergence-based and ALE-based reconstructions.

    Returns diagnostic statistics as described in the reference document.

    Parameters
    ----------
    w_div : np.ndarray, shape (K, N)
        W from Variant A (divergence).
    w_ale_cart : np.ndarray, shape (K, N)
        W from Variant B (ALE heuristic).
    z_levels : np.ndarray, shape (K,)
    ssh_t : np.ndarray, shape (N,)
    w_ale : np.ndarray, shape (K, N)
        Raw w_ALE from file (before correction).
    node_depths : np.ndarray, shape (N,)

    Returns
    -------
    dict
        ``err_abs_max``: max |w_div - w_ale_cart|
        ``err_abs_mean``: mean |w_div - w_ale_cart|
        ``chi_ALE_max``: max |(1-D/H)*ssh_t| / max(|w_ALE|, 1e-8)
        ``chi_ALE_mean``: mean of chi_ALE
    """
    err_abs = np.abs(w_div - w_ale_cart)

    K, N = w_ale.shape
    chi = np.zeros_like(w_ale)
    for k in range(K):
        D = z_levels[k]
        for n in range(N):
            H = max(node_depths[n], 1.0)
            delta = abs((1.0 - D / H) * ssh_t[n])
            chi[k, n] = delta / max(abs(w_ale[k, n]), 1e-8)

    return {
        "err_abs_max": float(np.max(err_abs)),
        "err_abs_mean": float(np.mean(err_abs)),
        "chi_ALE_max": float(np.max(chi)),
        "chi_ALE_mean": float(np.mean(chi)),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def reconstruct_w(
    u: np.ndarray,
    v: np.ndarray,
    eta_prev: np.ndarray,
    eta_next: np.ndarray,
    dt_real: float,
    mesh,
    w_ale: Optional[np.ndarray] = None,
    mesh_extras: Optional[Dict[str, np.ndarray]] = None,
    method: str = "auto",
    correct_bottom: bool = True,
    *,
    uv_vertical: str = "layer",
) -> np.ndarray:
    """Reconstruct Cartesian vertical velocity from horizontal fields.

    This function is z-level-only and assumes ``mesh.vertical_coord == "zlevel"``.
    u/v must lie on the layers: a FESOM 1.4 mesh, or ``uv_vertical="level"``,
    is refused (FESOM 1.4 output carries w).

    Parameters
    ----------
    u : np.ndarray, shape (K-1, E)
        Zonal velocity per element per layer (m/s).
    v : np.ndarray, shape (K-1, E)
        Meridional velocity per element per layer (m/s).
    eta_prev : np.ndarray, shape (N,)
        SSH at the earlier timestep (for time derivative).
    eta_next : np.ndarray, shape (N,)
        SSH at the later timestep.
    dt_real : float
        Time interval between ``eta_prev`` and ``eta_next`` (seconds).
    mesh : Mesh
        Must have ``sigma`` (z-levels) and ``depth`` (per-node bottom).
    w_ale : np.ndarray, shape (K, N), optional
        ALE vertical velocity from FESOM2 output.
    mesh_extras : dict, optional
        Additional fields from mesh diag: ``"nod_area"``, ``"nlevels"``,
        ``"ulevels"``, ``"nlevels_nod2D"``.
    method : str
        ``"divergence"`` — Variant A only.
        ``"ale"`` — Variant B only (explicit user choice).
        ``"auto"`` — always compute divergence (Variant A); if w_ale
        is provided, also compute Variant B and print comparison
        diagnostics (Variant C).
    correct_bottom : bool
        Apply linear bottom correction (Variant A only).
    uv_vertical : str
        Where u/v lie (``VelocityField.uv_vertical``): ``"layer"`` (the
        default); ``"level"`` is refused by :func:`require_layer_data`.

    Returns
    -------
    np.ndarray, shape (K, N)
        Cartesian vertical velocity at z-level interfaces.
        Positive-down convention (w > 0 = downwelling).

    Raises
    ------
    ValueError
        An unknown ``method``; ``method="ale"`` without ``w_ale``; a FESOM
        1.4 mesh or u/v on the levels.
    """
    require_layer_data(mesh, uv_vertical)
    if method not in ("divergence", "ale", "auto"):
        raise ValueError(f"method must be 'divergence', 'ale', or 'auto', got '{method}'")

    z_levels = np.abs(mesh.sigma)
    node_depths = mesh.depth
    K = len(z_levels)
    N = mesh.n_nodes

    # SSH tendency
    ssh_t = compute_ssh_tendency(eta_prev, eta_next, dt_real)

    # Variant B only (explicit user choice)
    if method == "ale":
        if w_ale is None:
            raise ValueError("method='ale' requires w_ale to be provided")
        return reconstruct_w_ale(w_ale, z_levels, ssh_t, node_depths)

    # Variant A: divergence-based
    extras = mesh_extras or {}
    nod_area = extras.get("nod_area")
    nlevels_elem = extras.get("nlevels")
    ulevels_elem = extras.get("ulevels")
    nlevels_nod = extras.get("nlevels_nod2D")

    if nod_area is not None:
        div_h = compute_divergence_with_area(
            u, v, mesh, nod_area, nlevels_elem, ulevels_elem,
        )
    else:
        div_h = compute_divergence_approx(u, v, mesh)

    w_div = integrate_continuity(
        div_h, z_levels, ssh_t, node_depths,
        nlevels_nod=nlevels_nod,
        correct_bottom=correct_bottom,
    )

    # Variant C: cross-validation diagnostics (auto mode with w_ale)
    if method == "auto" and w_ale is not None:
        w_ale_cart = reconstruct_w_ale(w_ale, z_levels, ssh_t, node_depths)
        diag = compare_variants(
            w_div, w_ale_cart, z_levels, ssh_t, w_ale, node_depths,
        )
        print(f"  W reconstruction diagnostics (divergence vs ALE heuristic):")
        print(f"    |w_div - w_ale_cart| max={diag['err_abs_max']:.2e} m/s"
              f"  mean={diag['err_abs_mean']:.2e} m/s")
        print(f"    chi_ALE (ALE correction significance)"
              f"  max={diag['chi_ALE_max']:.2f}"
              f"  mean={diag['chi_ALE_mean']:.2f}")
        if diag["chi_ALE_mean"] >= 0.5:
            warnings.warn(
                f"chi_ALE_mean={diag['chi_ALE_mean']:.2f} >= 0.5 — "
                "using w_ALE directly as w_cart would be inaccurate."
            )

    return w_div


__all__ = [
    "require_layer_data",
    "compute_ssh_tendency",
    "compute_divergence_approx",
    "compute_divergence_with_area",
    "integrate_continuity",
    "reconstruct_w_ale",
    "compare_variants",
    "reconstruct_w",
]
