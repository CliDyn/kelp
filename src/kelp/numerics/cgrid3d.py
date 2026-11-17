"""Plain Euler / RK2 / RK4 step of the exact 3-D C-grid path (``uv_interp:
cgrid`` with ``mode.Fdrift3D``).

The step runs in the fractional level coordinate ``s = k + zeta`` (level ``k``,
``zeta`` the fraction within the containing cell's own thickness; the porous /
TRACMASS view of the C-grid): inside a cell the kernel is the index-space
scheme ``dxi/dt = Uflux/Vol``, ``dzeta/dt = Wflux/Vol``, and a side-face
crossing keeps ``s`` (the same flux per unit ``zeta`` on both sides), so a
crossing from a thick cell into a thinner neighbour can never put a particle
below the neighbour's floor.  ``state.z`` stays metres everywhere else: ``z``
-> ``s`` with the start cell, ``s`` -> ``z`` with the end cell.  The metres are
measured on ``mesh._cgrid_zw`` (cumulative ``e3t_1d``; a diagnostic of ``s``,
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

import numpy as np

from ..core.geometry import METERS_PER_DEG_LAT, clamp_lat, wrap_lon
from ..core.mesh import find_containing_elements, find_nearest_nodes
from ..core.particles import FailureReason, ParticleStatus
from .interpolate import interp_cgrid3d

# scheme -> (stage fractions a_n: stage n starts at x + a_n * dt * k_{n-1},
#            weights b_n)
_BUTCHER = {"euler": ((0.0,), (1.0,)),
            "rk2": ((0.0, 0.5), (0.0, 1.0)),
            "rk4": ((0.0, 0.5, 0.5, 1.0), (1 / 6, 1 / 3, 1 / 3, 1 / 6))}


def _level_coord(mesh, e, ok, z=None, s=None):
    """Metres <-> fractional level in the cell ``e`` (``ok``: found):
    ``z -> s`` (given ``z``) or ``s -> z`` (given ``s``).  The cell's own
    thickness is ``e3b`` at its bottom level (``shrunk``; ``porous``: the
    reference layer there too), else the reference layer."""
    zw = mesh._cgrid_zw
    c = np.where(ok, mesh._cgrid_elem_ji[e, 0] * mesh._structured_shape[1]
                 + mesh._cgrid_elem_ji[e, 1], 0)
    nl, e3b = mesh._cgrid_nlev_raster.reshape(-1)[c], mesh._cgrid_e3b[c]
    k = (np.searchsorted(zw, z, side="right") - 1 if s is None
         else np.floor(s).astype(int))
    k = np.clip(k, 0, len(zw) - 2)
    e3 = np.where((k == nl - 1) & (mesh._cgrid_partial == "shrunk"), e3b, zw[k + 1] - zw[k])
    return k + (z - zw[k]) / e3 if s is None else zw[k] + (s - k) * e3


def cgrid3d_step(state, vel, mesh, dt, time_dir, scheme, kwargs):
    from . import integrators as I
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
        U, V, W, ok_n = interp_cgrid3d(
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
    reason = I._update_failure_reason(
        state.failure_reason, {FailureReason.INTERPOLATION_FAILED: interp_failed,
                               FailureReason.OUT_OF_BOUNDS: domain_failed})
    return state.replace(lon=lon, lat=lat, z=z, status=status, nearest_node=nn,
                         u=first[0], v=first[1], w=first[2], failure_reason=reason)
