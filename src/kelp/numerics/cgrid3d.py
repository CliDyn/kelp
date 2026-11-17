"""Plain Euler / RK2 / RK4 step of the exact 3-D C-grid path (``uv_interp:
cgrid`` with ``mode.Fdrift3D``).

Each stage reads ``(U, V, W)`` from ONE containment search (the layer and the
column bottom belong to the containing T-cell, not to the nearest F-node, which
is what the shared z-level helpers use).  No stage clamping, no pull-back: a
stage or end point in a dry ring cell reads zero velocity; a stage beyond the
ring fails (``INTERPOLATION_FAILED``) and an end outside the mesh is
``OUT_OF_BOUNDS``, as in every other path.  The end depth is clamped at the
surface only (``z >= zw[0]``; the continuity W leaves a surface residual), never
at the bottom: a level at or below a column's floor is land, zero velocity.
"""
from __future__ import annotations

import numpy as np

from ..core.geometry import METERS_PER_DEG_LAT, clamp_lat, wrap_lon
from ..core.mesh import find_nearest_nodes
from ..core.particles import FailureReason, ParticleStatus
from .interpolate import interp_cgrid3d

# scheme -> (stage fractions a_n: stage n starts at x + a_n * dt * k_{n-1},
#            weights b_n)
_BUTCHER = {"euler": ((0.0,), (1.0,)),
            "rk2": ((0.0, 0.5), (0.0, 1.0)),
            "rk4": ((0.0, 0.5, 0.5, 1.0), (1 / 6, 1 / 3, 1 / 3, 1 / 6))}


def cgrid3d_step(state, vel, mesh, dt, time_dir, scheme, kwargs):
    from . import integrators as I
    alive = state.alive_mask
    if not alive.any():
        return I._zero_dead_velocity(state)
    a, b = _BUTCHER[scheme]
    stages = I._make_stage_vels(vel, kwargs.get("vel_next"),
                                kwargs.get("dt_to_next"), dt, a)
    ok = alive.copy()
    ks, first = [], None
    for n, sv in enumerate(stages):
        if n:
            lon = wrap_lon(state.lon + a[n] * ks[-1][0])
            lat = clamp_lat(state.lat + a[n] * ks[-1][1])
            z = state.z + a[n] * ks[-1][2]
        else:
            lon, lat, z = state.lon, state.lat, state.z
        U, V, W, ok_n = interp_cgrid3d(lon, lat, z, ok, mesh, sv,
                                       find_nearest_nodes(mesh, lon, lat), time_dir)
        if n == 0:
            first = (U, V, W)
        ok = ok & ok_n
        ks.append((dt * U / I._z0(lat), dt * V / METERS_PER_DEG_LAT, dt * W))
    d = [sum(bn * k[c] for bn, k in zip(b, ks)) for c in range(3)]
    lon = np.where(ok, wrap_lon(state.lon + d[0]), state.lon)
    lat = np.where(ok, clamp_lat(state.lat + d[1]), state.lat)
    z = np.where(ok, state.z + d[2], state.z)
    nn = find_nearest_nodes(mesh, lon, lat)
    in_dom = I._check_in_domain(mesh, lon, lat, nn)
    interp_failed = alive & ~ok
    domain_failed = ok & ~in_dom
    failed = interp_failed | domain_failed
    lon, lat, z = (np.where(failed, o, n_) for o, n_ in
                   ((state.lon, lon), (state.lat, lat), (state.z, z)))
    nn = np.where(failed, find_nearest_nodes(mesh, state.lon, state.lat), nn)
    status = state.status.copy()
    status[failed] = ParticleStatus.OUTSIDE
    z = np.where(alive & ~failed, np.maximum(z, mesh._cgrid_zw[0]), z)
    reason = I._update_failure_reason(
        state.failure_reason, {FailureReason.INTERPOLATION_FAILED: interp_failed,
                               FailureReason.OUT_OF_BOUNDS: domain_failed})
    return state.replace(lon=lon, lat=lat, z=z, status=status, nearest_node=nn,
                         u=first[0], v=first[1], w=first[2], failure_reason=reason)
