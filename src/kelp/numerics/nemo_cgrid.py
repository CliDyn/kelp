"""Exact Arakawa C-grid numerics of the NEMO path (``mesh.nemo_grid: cgrid``).

- **Interpolation** (Delandmeter & van Sebille 2019): the Numba kernel
  ``_interp_nemo_cgrid_core`` and its entries ``interp_nemo_cgrid`` (2-D, the
  ``numerics.uv_interp: nemo_cgrid`` entry of the interpolator registry in
  :mod:`.interpolate`), ``interp_nemo_cgrid_single`` and ``interp_nemo_cgrid3d``.
- **3-D step** ``nemo_cgrid3d_step`` (``mode.Fdrift3D``), called by the Euler /
  RK2 / RK4 steps of :mod:`.integrators`.
- **Sampling** ``sample_field_nemo_cgrid``: the cell value of a T-point scalar.

The 3-D step runs in the fractional level coordinate ``s = k + zeta`` (level ``k``,
``zeta`` the fraction within the containing cell's own thickness; the porous /
TRACMASS view of the C-grid): inside a cell the kernel is the index-space
scheme ``dxi/dt = Uflux/Vol``, ``dzeta/dt = Wflux/Vol``, and a side-face
crossing keeps ``s`` (the same flux per unit ``zeta`` on both sides), so a
crossing from a thick cell into a thinner neighbour can never put a particle
below the neighbour's floor.  ``state.z`` stays metres everywhere else: ``z``
-> ``s`` with the start cell, ``s`` -> ``z`` with the end cell.  The metres are
measured on ``mesh._nemo_cgrid_zw`` (cumulative ``e3t_1d``; a diagnostic of ``s``,
the dynamics use the thicknesses), plus the partial bottom cell's own
thickness; depth in metres therefore jumps at a face between cells of
different bottom thickness (bottom level only).  ``mesh.nemo_partial_cells:
porous`` (NEMO 5) measures the bottom cell on the full reference layer instead
of ``e3t_0``: only metres change, never ``s``, lon or lat.

Each stage reads ``(U, V, ds/dt)`` from ONE containment search.  No stage
clamping, no pull-back: a stage or end point in a dry ring cell reads zero
velocity; a stage beyond the ring fails (``INTERPOLATION_FAILED``) and an end
outside the mesh is ``OUT_OF_BOUNDS``, as in every other path.  ``s`` is
clamped at the surface only (``s >= 0``; the continuity W leaves a surface
residual), never at the bottom: a level at or below a column's floor is land,
zero velocity.
"""
from __future__ import annotations

from typing import Tuple

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

from ..core.geometry import DEG2RAD, METERS_PER_DEG_LAT, clamp_lat, wrap_lon
from ..core.mesh import (_find_containing_element_local, find_containing_elements,
                         find_nearest_nodes)
from ..core.particles import FailureReason, ParticleState, ParticleStatus


# ---------------------------------------------------------------------------
# Exact Arakawa C-grid u/v (Delandmeter & van Sebille 2019; NEMO, 2-D).
# The element is a T-cell with F-point corners SW, SE, NE, NW; the field is
# the full face raster (u on the east face, v on the north face of T(j,i)),
# flattened to ``(ny*nx,)``.  The normal flux through each cell edge is the
# face velocity times the edge length; inside the cell the logical velocity
# is linear along its own axis, d(xi)/dt = ((1-xi)U0 + xi U1)/detJ, which
# keeps the interpolated field divergence-free where the face fluxes are, and
# zero-normal-flow through land faces.
# ---------------------------------------------------------------------------
@njit(cache=True)
def _nemo_cgrid_e3face(zw, kb_f, e3_f, f, k):
    """Thickness of face ``f`` at level ``k``: the reference layer, or the
    face's one partial level ``kb_f[f]`` (zps) with thickness ``e3_f[f]``."""
    if k == kb_f[f]:
        return e3_f[f]
    return zw[k + 1] - zw[k]


@njit(cache=True)
def _interp_nemo_cgrid_core(xp, yp, zp, nn_node, elem_hint, time_dir,
                       node_to_elems, node_to_elems_count, elem_nodes,
                       nodes_lon, nodes_lat, nemo_cgrid_corners, nemo_cgrid_ji, nemo_cgrid_nlev, nx,
                       e2u_f, e1v_f, area_f, zw, e3b_f, kbu_f, e3u_f, kbv_f, e3v_f,
                       u_f, v_f, w_f):
    """C-grid u/v(/w) at one point; ``u_f``/``v_f`` are ``(ny*nx,)`` face
    rasters, or ``(K*ny*nx,)`` layer-wise with ``zw`` (K+1 interfaces) given.

    3-D (``zw`` non-empty): ``zp`` is the FRACTIONAL LEVEL ``s = k + zeta``
    (level ``k = floor(s)``, ``zeta`` the fraction within the containing
    cell's own thickness; see :func:`nemo_cgrid3d_step`).  u/v are the 2-D formula
    on the faces of level ``k`` (constant within the level); ``w_f``
    ``((K+1)*ny*nx,)`` is linear in ``zeta``, and the third return value is
    ``ds/dt = ((1-zeta) w_k + zeta w_k+1) * area / (|detJ| e3cell)`` in levels
    per second: inside the cell ``dxi/dt = Uflux/Vol`` and ``dzeta/dt =
    Wflux/Vol`` with ``Vol = |detJ| e3cell``, so the field is non-divergent
    (up to float32 rounding of the stored rasters).  Partial bottom cells
    (zps): the column's bottom wet cell has thickness ``e3b_f``; a face's
    thickness at level ``k`` is the reference layer except at its one partial
    level (``kb*_f`` / ``e3*_f``, the model's ``e3u_0`` / ``e3v_0``), and the
    face fluxes are weighted by ``e3face/e3cell``.  ``s >= nlev`` (the floor)
    and the levels below are land: zero velocity, as dry columns are in 2-D.

    ``e2u_f``/``e1v_f`` are the model face widths (same raster; empty = absent):
    the fluxes are then ``u*e2u`` (W/E faces) and ``v*e1v`` (S/N faces), so the
    model's discrete continuity carries over (the Jacobian stays that of the
    local plane).  Without them the fluxes use plane edge lengths, each edge at
    the cosine of ITS OWN mean latitude.

    Returns ``(U, V, W, ok)`` in geographic m/s (east, north), consistent with
    the integrators' ``METERS_PER_DEG_LAT`` metric (``W``: see above).
    """
    if elem_hint >= 0:
        e = int(elem_hint)
    else:
        e, in_dom = _find_containing_element_local(
            xp, yp, nn_node, node_to_elems, node_to_elems_count,
            elem_nodes, nodes_lon, nodes_lat)
        if not in_dom:
            return 0.0, 0.0, 0.0, False
    if nemo_cgrid_nlev[e] <= 0:
        # dry-ring cell: zero velocity (land, as in Parcels)
        return 0.0, 0.0, 0.0, True
    j = int(nemo_cgrid_ji[e, 0])
    i = int(nemo_cgrid_ji[e, 1])
    kz = 0
    nynx = 0
    fW = 1.0; fE = 1.0; fS = 1.0; fN = 1.0
    e3c = 1.0
    if zw.shape[0] > 0:
        nynx = area_f.shape[0]
        if zp >= nemo_cgrid_nlev[e]:
            return 0.0, 0.0, 0.0, True
        kz = min(max(int(np.floor(zp)), 0), zw.shape[0] - 2)
        c = j * nx + i
        e3c = zw[kz + 1] - zw[kz]
        if kz == nemo_cgrid_nlev[e] - 1:
            e3c = e3b_f[c]
        fW = _nemo_cgrid_e3face(zw, kbu_f, e3u_f, c - 1, kz) / e3c
        fE = _nemo_cgrid_e3face(zw, kbu_f, e3u_f, c, kz) / e3c
        fS = _nemo_cgrid_e3face(zw, kbv_f, e3v_f, c - nx, kz) / e3c
        fN = _nemo_cgrid_e3face(zw, kbv_f, e3v_f, c, kz) / e3c
    ko = kz * nynx
    uW = u_f[ko + j * nx + i - 1]
    uE = u_f[ko + j * nx + i]
    vS = v_f[ko + (j - 1) * nx + i]
    vN = v_f[ko + j * nx + i]
    M = METERS_PER_DEG_LAT
    cosl = np.cos(yp * DEG2RAD)
    px = np.empty(4); py = np.empty(4)
    for k in range(4):
        nd = int(nemo_cgrid_corners[e, k])
        px[k] = (((nodes_lon[nd] - xp + 180.0) % 360.0) - 180.0) * cosl * M
        py[k] = (nodes_lat[nd] - yp) * M
    # closed-form bilinear inverse of the point (0, 0) (Parcels v4)
    a0 = px[0]; a1 = px[1] - px[0]; a2 = px[3] - px[0]
    a3 = px[0] - px[1] + px[2] - px[3]
    b0 = py[0]; b1 = py[1] - py[0]; b2 = py[3] - py[0]
    b3 = py[0] - py[1] + py[2] - py[3]
    scale = max(abs(a1), abs(a2), abs(b1), abs(b2), 1e-30)
    aa = a3 * b2 - a2 * b3
    bb = a3 * b0 - a0 * b3 + a1 * b2 - a2 * b1
    cc = a1 * b0 - a0 * b1
    eta = 0.0
    if abs(aa) < 1e-12 * scale * scale:
        if abs(bb) < 1e-300:
            return 0.0, 0.0, 0.0, False
        eta = -cc / bb
    else:
        disc = bb * bb - 4.0 * aa * cc
        if disc < 0.0:
            disc = 0.0
        sq = np.sqrt(disc)
        r1 = (-bb + sq) / (2.0 * aa)
        r2 = (-bb - sq) / (2.0 * aa)
        d1 = max(0.0, -r1, r1 - 1.0)       # distance outside [0, 1]
        d2 = max(0.0, -r2, r2 - 1.0)
        eta = r1 if d1 <= d2 else r2
    eta = min(max(eta, 0.0), 1.0)
    den = a1 + a3 * eta
    den2 = b1 + b3 * eta
    if abs(den) >= abs(den2):
        xi = (0.0 - a0 - a2 * eta) / den
    else:
        xi = (0.0 - b0 - b2 * eta) / den2
    xi = min(max(xi, 0.0), 1.0)
    # edge lengths and normal fluxes (positive towards +xi / +eta)
    if e2u_f.shape[0] > 0:
        U0 = fW * uW * e2u_f[j * nx + i - 1]; U1 = fE * uE * e2u_f[j * nx + i]
        V0 = fS * vS * e1v_f[(j - 1) * nx + i]; V1 = fN * vN * e1v_f[j * nx + i]
    else:
        # true edge lengths: each edge at the cosine of its mean latitude
        L = np.empty(4)
        for k in range(4):
            na = int(nemo_cgrid_corners[e, k]); nb = int(nemo_cgrid_corners[e, (k + 1) % 4])
            dl = ((nodes_lon[nb] - nodes_lon[na] + 180.0) % 360.0) - 180.0
            cm = np.cos(0.5 * (nodes_lat[na] + nodes_lat[nb]) * DEG2RAD)
            L[k] = np.hypot(dl * cm * M, (nodes_lat[nb] - nodes_lat[na]) * M)
        # edges: 0 = S (c0-c1), 1 = E (c1-c2), 2 = N (c2-c3), 3 = W (c3-c0)
        U0 = fW * uW * L[3]; U1 = fE * uE * L[1]; V0 = fS * vS * L[0]; V1 = fN * vN * L[2]
    dxdxi = (1.0 - eta) * a1 + eta * (px[2] - px[3])
    dydxi = (1.0 - eta) * b1 + eta * (py[2] - py[3])
    dxdeta = (1.0 - xi) * a2 + xi * (px[2] - px[1])
    dydeta = (1.0 - xi) * b2 + xi * (py[2] - py[1])
    det = dxdxi * dydeta - dxdeta * dydxi
    adet = abs(det)                       # either index-space handedness
    if adet < 1e-12 * scale * scale:
        return 0.0, 0.0, 0.0, False
    dxi = ((1.0 - xi) * U0 + xi * U1) / adet
    deta = ((1.0 - eta) * V0 + eta * V1) / adet
    U = dxdxi * dxi + dxdeta * deta
    V = dydxi * dxi + dydeta * deta
    W = 0.0
    if w_f.shape[0] > 0:
        zeta = min(max(zp - kz, 0.0), 1.0)
        W = ((1.0 - zeta) * w_f[ko + j * nx + i]
             + zeta * w_f[ko + nynx + j * nx + i]) * area_f[j * nx + i] / (adet * e3c)
    return time_dir * U, time_dir * V, time_dir * W, True


@njit(parallel=True, cache=True)
def _loop_interp_nemo_cgrid(P, alive_mask, lons, lats, zs, nn_nodes, elems, time_dir,
                       node_to_elems, node_to_elems_count, elem_nodes,
                       nodes_lon, nodes_lat, nemo_cgrid_corners, nemo_cgrid_ji, nemo_cgrid_nlev, nx,
                       e2u_f, e1v_f, area_f, zw, e3b_f, kbu_f, e3u_f, kbv_f, e3v_f,
                       u_f, v_f, w_f, U, V, W, ok):
    for i in prange(P):
        if not alive_mask[i]:
            continue
        u, v, w, success = _interp_nemo_cgrid_core(
            lons[i], lats[i], zs[i], int(nn_nodes[i]), int(elems[i]), time_dir,
            node_to_elems, node_to_elems_count, elem_nodes, nodes_lon,
            nodes_lat, nemo_cgrid_corners, nemo_cgrid_ji, nemo_cgrid_nlev, nx, e2u_f, e1v_f,
            area_f, zw, e3b_f, kbu_f, e3u_f, kbv_f, e3v_f, u_f, v_f, w_f)
        U[i] = u
        V[i] = v
        W[i] = w
        ok[i] = success


def _nemo_cgrid_args(mesh, vel):
    if not getattr(mesh, "_nemo_cgrid", False):
        raise ValueError(
            "uv_interp 'nemo_cgrid' needs a C-grid mesh (NEMO reader with "
            "mesh.nemo_grid: cgrid)")
    if vel.uv_layout != "nemo_cgrid":
        raise ValueError(
            f"uv_interp 'nemo_cgrid' needs uv_layout='nemo_cgrid', got {vel.uv_layout!r}")
    return (mesh.node_to_elems, mesh.node_to_elems_count, mesh.elem_nodes,
            mesh.nodes_lon, mesh.nodes_lat, mesh._nemo_cgrid_corners,
            mesh._nemo_cgrid_elem_ji, mesh._nemo_cgrid_nlev,
            int(mesh._structured_shape[1]),
            mesh._nemo_cgrid_e2u, mesh._nemo_cgrid_e1v, mesh._nemo_cgrid_area, mesh._nemo_cgrid_zw,
            mesh._nemo_cgrid_e3b, mesh._nemo_cgrid_kbu, mesh._nemo_cgrid_e3u,
            mesh._nemo_cgrid_kbv, mesh._nemo_cgrid_e3v)


_ABSENT = np.empty(0)     # stands in for an absent w field


def interp_nemo_cgrid_single(xp: float, yp: float, layer: int, mesh, vel, node_idx: int,
                        time_dir: float = 1.0, *, elem: int = -1
                        ) -> Tuple[float, float, bool]:
    """Exact C-grid u/v at one point (signature of ``interp_barycentric_single``).

    Parameters
    ----------
    xp, yp : float
        Position (deg).
    layer : int
        Unused (the 2-D path reads one layer).
    mesh : Mesh
        A ``build_nemo_cgrid_mesh`` mesh.
    vel : VelocityField
        ``u`` / ``v`` on the flattened face rasters.
    node_idx : int
        Nearest-node hint for the containment search.
    time_dir : float
        ``+1`` forward, ``-1`` backward in time (flips the velocity sign).
    elem : int
        Pre-validated containing element, ``-1`` = search.

    Returns
    -------
    U, V : float
        East / north velocity (m/s).
    ok : bool
        False outside every element.
    """
    a = _nemo_cgrid_args(mesh, vel)
    U, V, _W, ok = _interp_nemo_cgrid_core(xp, yp, 0.0, int(node_idx), int(elem),
                                      time_dir, *a, vel.u, vel.v, _ABSENT)
    return U, V, ok


def interp_nemo_cgrid3d(lon, lat, z, mask, mesh, vel, nn_nodes, time_dir=1.0, *,
                   elems=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact C-grid u/v(/w) for the ``mask`` particles (Delandmeter & van Sebille 2019).

    Parameters
    ----------
    lon, lat : ndarray (P,)
        Positions (deg).
    z : ndarray (P,)
        Fractional level ``s`` on a 3-D mesh; ignored on a 2-D one.
    mask : ndarray of bool (P,)
        Particles to evaluate.
    mesh : Mesh
        A ``build_nemo_cgrid_mesh`` mesh.
    vel : VelocityField
        Face velocities ``u`` / ``v`` (and ``w`` on the level interfaces in 3-D)
        on the flattened rasters.
    nn_nodes : ndarray of int (P,)
        Nearest-node hints for the containment search.
    time_dir : float
        ``+1`` forward, ``-1`` backward in time.
    elems : ndarray of int (P,) or None
        Pre-validated containing elements (edge-walk tracker), ``-1`` = search.

    Returns
    -------
    U, V : ndarray (P,)
        East / north velocity (m/s).
    W : ndarray (P,)
        Zero in 2-D; ``ds/dt`` (levels per second) in 3-D.
    ok : ndarray of bool (P,)
        False for a point outside every element, or beyond the dry ring.
    """
    a = _nemo_cgrid_args(mesh, vel)
    P = lon.shape[0]
    U = np.zeros(P); V = np.zeros(P); W = np.zeros(P)
    ok = np.zeros(P, dtype=bool)
    el = (np.full(P, -1, dtype=np.int32) if elems is None
          else np.asarray(elems, dtype=np.int32))
    w_f = _ABSENT if vel.w is None else np.ascontiguousarray(vel.w).reshape(-1)
    uf = np.ascontiguousarray(vel.u).reshape(-1)
    vf = np.ascontiguousarray(vel.v).reshape(-1)
    args = (P, mask, lon, lat, np.asarray(z, dtype=np.float64), nn_nodes, el,
            time_dir, *a, uf, vf, w_f, U, V, W, ok)
    _loop_interp_nemo_cgrid(*args)
    return U, V, W, ok


def interp_nemo_cgrid(state, mesh, vel, nn_nodes, time_dir=1.0, layer=0, *,
                 elems=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact Arakawa C-grid u/v for all active particles (2-D, NEMO).

    Parameters
    ----------
    state : ParticleState
        Particles; the alive ones are evaluated.
    mesh, vel, nn_nodes, time_dir, elems
        As for :func:`interp_nemo_cgrid3d`.
    layer : int
        Unused (the 2-D path reads one layer).

    Returns
    -------
    U, V : ndarray (P,)
        East / north velocity (m/s).
    ok : ndarray of bool (P,)
        False for a particle outside every element.
    """
    U, V, _W, ok = interp_nemo_cgrid3d(state.lon, state.lat, np.zeros(state.n_particles),
                                  state.alive_mask, mesh, vel, nn_nodes,
                                  time_dir, elems=elems)
    return U, V, ok


# ---------------------------------------------------------------------------
# 3-D step (fractional level coordinate)
# ---------------------------------------------------------------------------
# scheme -> (stage fractions a_n: stage n starts at x + a_n * dt * k_{n-1},
#            weights b_n)
_BUTCHER = {"euler": ((0.0,), (1.0,)),
            "rk2": ((0.0, 0.5), (0.0, 1.0)),
            "rk4": ((0.0, 0.5, 0.5, 1.0), (1 / 6, 1 / 3, 1 / 3, 1 / 6))}


def _cell_thickness(mesh, e, ok, k):
    """Thickness (m) of level ``k`` (clipped to the valid range) in the cell
    ``e`` (``ok``: found); returns ``(k, e3)``.  The cell's own thickness is
    ``e3b`` at its bottom level (``shrunk``; ``porous``: the reference layer
    there too), else the reference layer."""
    zw = mesh._nemo_cgrid_zw
    c = np.where(ok, mesh._nemo_cgrid_elem_ji[e, 0] * mesh._structured_shape[1]
                 + mesh._nemo_cgrid_elem_ji[e, 1], 0)
    nl, e3b = mesh._nemo_cgrid_nlev_raster.reshape(-1)[c], mesh._nemo_cgrid_e3b[c]
    k = np.clip(k, 0, len(zw) - 2)
    e3 = np.where((k == nl - 1) & (mesh._nemo_cgrid_partial == "shrunk"), e3b, zw[k + 1] - zw[k])
    return k, e3


def _level_coord(mesh, e, ok, z=None, s=None):
    """Metres <-> fractional level in the cell ``e`` (``ok``: found):
    ``z -> s`` (given ``z``) or ``s -> z`` (given ``s``), measured with the
    thickness of :func:`_cell_thickness`."""
    zw = mesh._nemo_cgrid_zw
    k = (np.searchsorted(zw, z, side="right") - 1 if s is None
         else np.floor(s).astype(int))
    k, e3 = _cell_thickness(mesh, e, ok, k)
    return k + (z - zw[k]) / e3 if s is None else zw[k] + (s - k) * e3


def nemo_cgrid3d_step(state: ParticleState, vel, mesh, dt: float, time_dir: float,
                 scheme: str, kwargs: dict) -> ParticleState:
    """One Euler / RK2 / RK4 step of the exact 3-D C-grid path.

    Parameters
    ----------
    state : ParticleState
        Particles with ``z`` in metres (positive down).
    vel : VelocityField
        Face u / v and interface w of the current record.
    mesh : Mesh
        A 3-D ``build_nemo_cgrid_mesh`` mesh.
    dt : float
        Step (s), positive.
    time_dir : float
        ``+1`` forward, ``-1`` backward in time.
    scheme : {"euler", "rk2", "rk4"}
        Stage scheme.
    kwargs : dict
        Integrator options: ``vel_next`` / ``dt_to_next`` (temporal
        interpolation between two records).  ``diffusion`` and ``recon``
        (PCHIP) are refused.

    Returns
    -------
    ParticleState
        New state.  A stage or end point that leaves the mesh keeps the start
        position and gets ``status = OUTSIDE`` with the matching
        ``FailureReason``.  ``u``, ``v`` are the first stage's velocities (m/s)
        and ``w`` is dz/dt (m/s) in the start cell.

    Raises
    ------
    NotImplementedError
        ``kwargs`` carries diffusion or PCHIP reconstruction.
    """
    from . import integrators as I
    if kwargs.get("diffusion") is not None or kwargs.get("recon") is not None:
        raise NotImplementedError(
            "the 3-D C-grid step supports neither diffusion nor temporal "
            "(PCHIP) reconstruction")
    alive = state.alive_mask
    if not alive.any():
        return I._zero_dead_velocity(state)
    a, b = _BUTCHER[scheme]
    stages = I._make_stage_vels(vel, kwargs.get("vel_next"),
                                kwargs.get("dt_to_next"), dt, a)
    ok = alive.copy()
    nn0 = find_nearest_nodes(mesh, state.lon, state.lat)     # fresh: a carried nearest_node may be stale (-1)
    e0, ok0 = find_containing_elements(mesh, state.lon, state.lat, nn0)   # start cell: s0 and stage 1
    s0 = _level_coord(mesh, e0, ok0, z=state.z)
    ks, first = [], None
    for n, sv in enumerate(stages):
        if n:
            lon = wrap_lon(state.lon + a[n] * ks[-1][0])
            lat = clamp_lat(state.lat + a[n] * ks[-1][1])
            s = s0 + a[n] * ks[-1][2]
        else:
            lon, lat, s = state.lon, state.lat, s0
        U, V, W, ok_n = interp_nemo_cgrid3d(
            lon, lat, s, ok, mesh, sv, nn0 if n == 0 else find_nearest_nodes(mesh, lon, lat),
            time_dir, elems=np.where(ok0, e0, -1) if n == 0 else None)
        if n == 0:
            first = (U, V, W)
        ok = ok & ok_n
        ks.append((dt * U / I._z0(lat), dt * V / METERS_PER_DEG_LAT, dt * W))
    d = [sum(bn * k[c] for bn, k in zip(b, ks)) for c in range(3)]
    lon = np.where(ok, wrap_lon(state.lon + d[0]), state.lon)
    lat = np.where(ok, clamp_lat(state.lat + d[1]), state.lat)
    s = np.where(ok, np.maximum(s0 + d[2], 0.0), s0)
    nn = find_nearest_nodes(mesh, lon, lat)
    e1, in_dom = find_containing_elements(mesh, lon, lat, nn)       # end cell: the domain check and s -> z
    interp_failed = alive & ~ok
    domain_failed = ok & ~in_dom
    failed = interp_failed | domain_failed
    lon, lat, s = (np.where(failed, o, n_) for o, n_ in
                   ((state.lon, lon), (state.lat, lat), (s0, s)))
    nn = np.where(failed, nn0, nn)
    status = state.status.copy()
    status[failed] = ParticleStatus.OUTSIDE
    # unchanged s keeps z (a particle at rest, or on land, is not re-measured)
    z = np.where(s == s0, state.z,
                 _level_coord(mesh, np.where(failed, e0, e1), np.where(failed, ok0, in_dom), s=s))
    # vel_w in m/s: the first stage's ds/dt times the start cell's thickness
    _, e3 = _cell_thickness(mesh, e0, ok0, np.floor(s0).astype(int))
    w_ms = first[2] * e3
    reason = I._update_failure_reason(
        state.failure_reason, {FailureReason.INTERPOLATION_FAILED: interp_failed,
                               FailureReason.OUT_OF_BOUNDS: domain_failed})
    return state.replace(lon=lon, lat=lat, z=z, status=status, nearest_node=nn,
                         u=first[0], v=first[1], w=w_ms, failure_reason=reason)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def sample_field_nemo_cgrid(values, mesh, lons, lats, zs=None, *, alive=None,
                       nearest=None) -> np.ndarray:
    """Cell value of a T-point scalar on the exact C-grid layout.

    Parameters
    ----------
    values : ndarray (ny*nx,) or (K, ny*nx)
        Flattened T raster (land NaN), 2-D or layer-wise.
    mesh : Mesh
        A ``build_nemo_cgrid_mesh`` mesh.
    lons, lats : ndarray (P,)
        Particle positions (deg).
    zs : ndarray (P,) or None
        Depths (m); required for layer-wise ``values``, the layer is the one
        containing the depth.
    alive : ndarray of bool (P,) or None
        Particles to sample (default: all).
    nearest : ndarray of int (P,) or None
        Nearest-node hints for the containment search.

    Returns
    -------
    ndarray (P,)
        The value of the wet cell (and layer) containing the particle; NaN
        outside every cell, for a dry-ring cell, or for a dead particle.
    """
    P = len(lons)
    alive = np.ones(P, dtype=bool) if alive is None else np.asarray(alive)
    out = np.full(P, np.nan)
    idx = np.nonzero(alive)[0]
    if idx.size == 0:
        return out
    nn = (find_nearest_nodes(mesh, lons[idx], lats[idx]) if nearest is None
          else np.asarray(nearest)[idx])
    e, ok = find_containing_elements(mesh, lons[idx], lats[idx], nn)
    e = np.where(ok, e, 0)
    flat = (mesh._nemo_cgrid_elem_ji[e, 0] * mesh._structured_shape[1]
            + mesh._nemo_cgrid_elem_ji[e, 1])
    if values.ndim == 2:
        zw = mesh._nemo_cgrid_zw
        k = np.clip(np.searchsorted(zw, zs[idx], side="right") - 1, 0, values.shape[0] - 1)
        out[idx] = values[k, flat]
    else:
        out[idx] = values[flat]
    out[idx[~ok]] = np.nan
    return out


__all__ = [
    "interp_nemo_cgrid",
    "interp_nemo_cgrid_single",
    "interp_nemo_cgrid3d",
    "nemo_cgrid3d_step",
    "sample_field_nemo_cgrid",
]
