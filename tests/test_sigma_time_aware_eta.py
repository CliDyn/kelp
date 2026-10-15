"""Phase 3 test: stage-time eta actually flows into sigma column geometry.

When the surface elevation ``eta`` changes between file snapshots and
we use the time-aware RK API (``vel_next`` + ``dt_to_next``), each RK
stage must see an interpolated ``eta`` value that reflects the stage
time ``t_k + c_k * dt``, not the step-start ``eta``.

This is what makes sigma tracking actually *time-aware* — otherwise the
RK stages would integrate the wrong column geometry.

The test constructs two snapshots with very different ``eta`` (e.g.
0 m and +50 m on a 100 m water column) and compares three runs of
``rk4_step``:

- steady with ``vel`` only  (eta=0 throughout, single step)
- steady with ``vel_next``  (eta=+50 throughout, single step)
- time-aware lerp           (eta interpolated per stage)

The time-aware result must sit *between* the two steady extremes and
must not equal either — that's the signature of lerp actually firing.
"""
from __future__ import annotations

import numpy as np
import pytest

from kelp.numerics.integrators import rk4_step
from kelp.numerics.interpolate import VelocityField
from kelp.core.mesh import (
    Mesh, build_node_to_elems, build_search, find_nearest_nodes,
)
from kelp.core.particles import make_state, ParticleStatus


SIGMA_LEV = np.array(
    [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
    dtype=np.float64,
)


def _sigma_mesh():
    """3x3 grid at the equator, bottom depth 100 m, sigma coordinate."""
    xs = np.array([4.0, 5.0, 6.0])
    ys = np.array([-1.0, 0.0, 1.0])
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel().astype(np.float64)
    lat = gy.ravel().astype(np.float64)
    depth = np.full(9, 100.0, dtype=np.float64)

    elems = []
    for j in range(2):
        for i in range(2):
            n0 = j * 3 + i
            n1 = n0 + 1
            n2 = n0 + 3
            n3 = n0 + 4
            elems.append([n0, n1, n2])
            elems.append([n1, n3, n2])
    elem_nodes = np.array(elems, dtype=np.int32)

    lon_c = np.array([lon[e].mean() for e in elem_nodes], dtype=np.float64)
    lat_c = np.array([lat[e].mean() for e in elem_nodes], dtype=np.float64)
    n2e, n2e_count = build_node_to_elems(9, elem_nodes)

    mesh = Mesh(
        name="sigma_time_aware",
        nodes_lon=lon, nodes_lat=lat, depth=depth,
        elem_nodes=elem_nodes, elem_center_lon=lon_c, elem_center_lat=lat_c,
        node_to_elems=n2e, node_to_elems_count=n2e_count,
        sigma=SIGMA_LEV.copy(), vertical_coord="sigma",
    )
    build_search(mesh)
    return mesh


def _vel_with_eta(mesh, eta_value, w_value):
    """Uniform sigma VelocityField with a chosen constant eta and W."""
    ne = mesh.n_elements
    nn = mesh.n_nodes
    K = mesh.sigma.shape[0]
    u = np.zeros((K - 1, ne), dtype=np.float64)
    v = np.zeros((K - 1, ne), dtype=np.float64)
    # Vertical gradient: W depends on the sigma index so that particle
    # depth (which depends on eta via H_col) changes the sampled W.
    w = np.empty((K, nn), dtype=np.float64)
    for k in range(K):
        w[k, :] = w_value * (1.0 - SIGMA_LEV[k])   # 0 at surface, w_value at bottom
    eta = np.full(nn, eta_value, dtype=np.float64)
    return VelocityField(u=u, v=v, eta=eta, w=w)


class TestSigmaStageEta:
    """Stage-time ``eta`` must flow through sigma layer detection, W sampling
    and clamping in the time-aware RK path."""

    def test_time_aware_rk4_stage_eta_shifts_result(self):
        mesh = _sigma_mesh()
        w_magnitude = 0.01   # m/s at sigma=0 (bottom); 0 at surface

        vel_a = _vel_with_eta(mesh, eta_value=0.0,  w_value=w_magnitude)
        vel_b = _vel_with_eta(mesh, eta_value=50.0, w_value=w_magnitude)

        # Particle seeded in the middle of the column (z=50 m).
        lons = np.array([5.0]); lats = np.array([0.0])
        z = np.array([50.0])
        nn = find_nearest_nodes(mesh, lons, lats)
        state0 = make_state(lon=lons, lat=lats, z=z).replace(nearest_node=nn)

        dt = 100.0
        dt_to_next = 100.0

        # Three variants of the same step:
        # (a) steady vel_a (eta=0) — H=100, particle at z/H=0.5, samples
        #     w = w_magnitude * (1-sigma_at_0.5H) = w_magnitude * 0.5 = 0.005
        # (b) steady vel_b (eta=+50) — H=150, particle at z/H=0.333, samples
        #     w = w_magnitude * 0.333 = 0.00333
        # (c) time-aware from a to b — each stage sees a different lerped
        #     eta, effective H mid-way through; W sample lands between
        #     the two steady cases.
        new_a = rk4_step(state0, vel_a, mesh, dt=dt, time_dir=1.0,
                          w_interp="nearest")
        new_b = rk4_step(state0, vel_b, mesh, dt=dt, time_dir=1.0,
                          w_interp="nearest")
        new_ta = rk4_step(
            state0, vel_a, mesh, dt=dt, time_dir=1.0,
            w_interp="nearest",
            vel_next=vel_b, dt_to_next=dt_to_next,
        )

        assert new_a.status[0] == ParticleStatus.ALIVE
        assert new_b.status[0] == ParticleStatus.ALIVE
        assert new_ta.status[0] == ParticleStatus.ALIVE

        # Since the review of PR #102 sigma steps integrate s = z / H with
        # ds/dt = W / H: the particle moves with its column (s0 = 0.5 kept
        # while eta goes 0 -> 50) and W is a DIA-SIGMA displacement, so the
        # comparison is in s: the change of s over the step.
        dz_a = new_a.z[0] / 100.0 - 0.5
        dz_b = new_b.z[0] / 150.0 - 50.0 / 150.0
        dz_ta = new_ta.z[0] / 150.0 - 0.5

        # Both steady runs produce downward drift (W > 0 at sigma < 1 in
        # both fields), but with different magnitudes because H differs.
        assert dz_a > 0 and dz_b > 0 and dz_ta > 0

        # Time-aware W part sits strictly between the two steady cases
        # (inclusive tolerance for edge equality is unneeded — lerp uses
        # alpha in (0, 1) on all four stages).
        lo, hi = sorted([dz_a, dz_b])
        assert lo < dz_ta < hi, (
            f"time-aware dz={dz_ta} not strictly between steady "
            f"vel_a dz={dz_a} and vel_b dz={dz_b}"
        )

        # And it must differ measurably from both extremes, otherwise
        # lerp isn't firing.
        assert abs(dz_ta - dz_a) > 1e-6, "time-aware dz matches vel_a exactly"
        assert abs(dz_ta - dz_b) > 1e-6, "time-aware dz matches vel_b exactly"

    def test_rk2_clamps_to_end_of_step_eta(self):
        """BLOCKER regression: RK2 was clamping with midpoint eta (c=0.5)
        instead of end-of-step eta.  When eta drops sharply between
        snapshots (e.g. 0 → -80 on a 100 m bottom), the particle must be
        clamped to the end-time column H=20, not the midpoint H=60.

        A review found this on 2026-04-17.
        """
        mesh = _sigma_mesh()
        ne = mesh.n_elements; nn = mesh.n_nodes; K = mesh.sigma.shape[0]

        def _vel(eta_val, w_val=0.0):
            u = np.zeros((K - 1, ne)); v = np.zeros((K - 1, ne))
            w = np.full((K, nn), w_val)
            eta = np.full(nn, eta_val, dtype=np.float64)
            return VelocityField(u=u, v=v, eta=eta, w=w)

        lons = np.array([5.0]); lats = np.array([0.0])
        nn_arr = find_nearest_nodes(mesh, lons, lats)
        from kelp.numerics.integrators import rk2_step as rk2

        # (1) no W: since 2026-09-23 the particle moves WITH its column
        # (sigma 0.4 kept): z = 60 * 20 / 100 = 12 m, no clamp involved
        state0 = make_state(lon=lons, lat=lats, z=np.array([60.0])
                             ).replace(nearest_node=nn_arr)
        new = rk2(state0, _vel(0.0), mesh, dt=100.0, time_dir=1.0,
                  w_interp="nearest", vel_next=_vel(-80.0), dt_to_next=100.0)
        assert new.z[0] == pytest.approx(12.0, rel=1e-12)

        # (2) a strong downward W (50 m per step) pushes it through the
        # bottom: the clamp must use the END-of-step column H = 20 (19.9),
        # not the midpoint H = 60
        state0 = make_state(lon=lons, lat=lats, z=np.array([99.0])
                             ).replace(nearest_node=nn_arr)
        new = rk2(state0, _vel(0.0, 0.5), mesh, dt=100.0, time_dir=1.0,
                  w_interp="nearest", vel_next=_vel(-80.0, 0.5),
                  dt_to_next=100.0)
        assert new.z[0] <= 20.0, (
            f"RK2 z={new.z[0]:.1f} exceeds end-time bottom 20.0 m; "
            "final clamp is using midpoint eta, not end-of-step eta"
        )
        assert new.z[0] == pytest.approx(19.9, abs=0.1)

    def test_steady_eta_equals_time_aware_with_matching_snapshots(self):
        """Sanity: if vel == vel_next the time-aware path must reduce
        to the steady one."""
        mesh = _sigma_mesh()
        vel = _vel_with_eta(mesh, eta_value=10.0, w_value=0.005)

        lons = np.array([5.0]); lats = np.array([0.0])
        nn = find_nearest_nodes(mesh, lons, lats)
        state0 = make_state(lon=lons, lat=lats, z=np.array([40.0])
                             ).replace(nearest_node=nn)

        steady = rk4_step(state0, vel, mesh, dt=100.0, time_dir=1.0,
                           w_interp="nearest")
        time_aware = rk4_step(
            state0, vel, mesh, dt=100.0, time_dir=1.0,
            w_interp="nearest",
            vel_next=vel, dt_to_next=100.0,
        )
        assert time_aware.z[0] == pytest.approx(steady.z[0], abs=1e-12)
        assert time_aware.lon[0] == pytest.approx(steady.lon[0], abs=1e-12)
        assert time_aware.lat[0] == pytest.approx(steady.lat[0], abs=1e-12)
