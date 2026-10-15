"""Time-aware RK API tests.

Verifies that rk2_step / rk4_step support intra-substep time interpolation
via vel_next + dt_to_next keyword arguments.

Test problem: spatially uniform, time-dependent velocity (analytical):
    U(t) = U0 * cos(omega * t)
    V(t) = U0 * sin(omega * t)
    integral: x(t) = x0 + (U0/omega) * sin(omega * t)
              y(t) = y0 + (U0/omega) * (1 - cos(omega * t))

Two velocity snapshots vel_a (t=t_step) and vel_b (t=t_step + dt_to_next)
are passed to integrator. Expected: RK4 gets ~4th-order accuracy in time.
"""
from __future__ import annotations

import numpy as np
import pytest

from kelp.numerics.integrators import euler_step, rk2_step, rk4_step
from kelp.numerics.interpolate import VelocityField
from kelp.core.mesh import Mesh, build_node_to_elems, build_search
from kelp.core.particles import make_state, ParticleStatus
from kelp.core.geometry import METERS_PER_DEG_LAT, DEG2RAD


U0 = 0.3
OMEGA = 1e-4  # angular frequency
LON0 = 0.0
LAT0 = 0.0  # equator: cos(lat) = 1


def _make_mesh():
    """Build small flat-ish patch."""
    span = 0.15
    nx = 5
    xs = np.linspace(-span, span, nx)
    ys = np.linspace(-span, span, nx)
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel().astype(np.float64)
    lat = gy.ravel().astype(np.float64)
    n_nodes = lon.shape[0]
    depth = np.full(n_nodes, 200.0, dtype=np.float64)

    elems = []
    for j in range(nx - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            elems.append([n0, n0 + 1, n0 + nx])
            elems.append([n0 + 1, n0 + nx + 1, n0 + nx])
    elem_nodes = np.array(elems, dtype=np.int32)
    lon_c = np.array([lon[e].mean() for e in elem_nodes], dtype=np.float64)
    lat_c = np.array([lat[e].mean() for e in elem_nodes], dtype=np.float64)
    n2e, n2e_count = build_node_to_elems(n_nodes, elem_nodes)
    z_levels = np.linspace(0.0, 200.0, 5)

    mesh = Mesh(name="t", nodes_lon=lon, nodes_lat=lat, depth=depth,
                elem_nodes=elem_nodes,
                elem_center_lon=lon_c, elem_center_lat=lat_c,
                node_to_elems=n2e, node_to_elems_count=n2e_count,
                sigma=z_levels)
    build_search(mesh)
    return mesh


def _make_vel(mesh, t):
    """Spatially uniform U(t) = U0 cos(omega t), V(t) = U0 sin(omega t)."""
    ne = mesh.n_elements
    nn = mesh.n_nodes
    u_val = U0 * np.cos(OMEGA * t)
    v_val = U0 * np.sin(OMEGA * t)
    u = np.full(ne, u_val, dtype=np.float64)
    v = np.full(ne, v_val, dtype=np.float64)
    eta = np.zeros(nn, dtype=np.float64)
    return VelocityField(u=u, v=v, eta=eta, w=None)


def _analytical(t):
    """Closed-form trajectory in metres from (LON0, LAT0)."""
    x = (U0 / OMEGA) * np.sin(OMEGA * t)
    y = (U0 / OMEGA) * (1.0 - np.cos(OMEGA * t))
    return x, y


def _xy_err(state, t):
    """Compute |numerical - analytical| in metres."""
    x_a, y_a = _analytical(t)
    cos_lat0 = np.cos(LAT0 * DEG2RAD)
    x_n = (state.lon[0] - LON0) * cos_lat0 * METERS_PER_DEG_LAT
    y_n = (state.lat[0] - LAT0) * METERS_PER_DEG_LAT
    return np.sqrt((x_n - x_a) ** 2 + (y_n - y_a) ** 2)


# =====================================================================
# Steady mode (vel_next=None): old behavior preserved
# =====================================================================

class TestSteadyModePreserved:
    """Steady velocity passed without vel_next → first-order in time as before."""

    def test_rk4_steady_first_order_time(self):
        """RK4 with steady vel (single field) should be first-order in time
        for a non-autonomous problem — Euler-equivalent error.

        This documents pre-refactor behavior is preserved when vel_next=None.
        """
        mesh = _make_mesh()
        runtime = 3600.0
        dt = 1800.0
        n_steps = int(runtime / dt)

        state = make_state(np.array([LON0]), np.array([LAT0]))
        for k in range(n_steps):
            t_k = k * dt
            vel = _make_vel(mesh, t_k)
            state = rk4_step(state, vel, mesh, dt, time_dir=1.0)

        assert state.status[0] == ParticleStatus.ALIVE
        err = _xy_err(state, runtime)
        # Without time-aware mode, error is large (~tens of metres for
        # this oscillating field over 1h with dt=30min)
        assert err > 1.0, (
            f"steady-mode RK4 should NOT be perfect (got {err:.3e}m)"
        )


# =====================================================================
# Time-aware mode: improves accuracy
# =====================================================================

class TestTimeAwareRK:
    """vel_next + dt_to_next enables proper RK time order."""

    def test_3d_rk4_time_aware_runs(self):
        """3D time-aware RK4 (w + layered u/v) runs correctly.

        Covers the 3D unfused branch; stage_vels each contain lerped w
        and layered u/v.  Verifies particle moves by the expected amount
        under linear-in-time velocity interpolation.
        """
        from kelp.numerics.interpolate import VelocityField
        mesh = _make_mesh()
        K = 5  # from mesh.sigma = linspace(0, 200, 5)
        ne = mesh.n_elements
        nn = mesh.n_nodes

        u = np.full((K - 1, ne), 0.1)
        v = np.full((K - 1, ne), 0.0)
        w = np.full((K, nn), 0.001)
        u_next = np.full((K - 1, ne), 0.2)
        v_next = np.full((K - 1, ne), 0.0)
        w_next = np.full((K, nn), 0.002)

        vel = VelocityField(u=u, v=v, eta=np.zeros(nn), w=w)
        vel_next = VelocityField(u=u_next, v=v_next, eta=np.zeros(nn), w=w_next)

        state = make_state(np.array([LON0]), np.array([LAT0]), np.array([100.0]))
        dt = 300.0
        state = rk4_step(state, vel, mesh, dt, time_dir=1.0,
                         vel_next=vel_next, dt_to_next=dt)

        assert state.status[0] == ParticleStatus.ALIVE
        # Linear w: w(t) = 0.001 + (0.001/dt) * t, RK4 avg over dt ≈ 0.0015 m/s
        # dz ≈ 0.0015 * 300 = 0.45m
        dz = state.z[0] - 100.0
        assert 0.4 < dz < 0.5, f"3D RK4 z displacement {dz:.3f}m out of [0.4, 0.5]"

    def test_3d_rk2_time_aware_runs(self):
        """3D time-aware RK2 also runs correctly."""
        from kelp.numerics.interpolate import VelocityField
        mesh = _make_mesh()
        K = 5
        ne = mesh.n_elements
        nn = mesh.n_nodes

        u = np.full((K - 1, ne), 0.1)
        v = np.full((K - 1, ne), 0.0)
        w = np.full((K, nn), 0.001)
        u_next = np.full((K - 1, ne), 0.2)
        v_next = np.full((K - 1, ne), 0.0)
        w_next = np.full((K, nn), 0.002)

        vel = VelocityField(u=u, v=v, eta=np.zeros(nn), w=w)
        vel_next = VelocityField(u=u_next, v=v_next, eta=np.zeros(nn), w=w_next)

        state = make_state(np.array([LON0]), np.array([LAT0]), np.array([100.0]))
        state = rk2_step(state, vel, mesh, 300.0, time_dir=1.0,
                         vel_next=vel_next, dt_to_next=300.0)

        assert state.status[0] == ParticleStatus.ALIVE
        dz = state.z[0] - 100.0
        assert 0.4 < dz < 0.5, f"3D RK2 z displacement {dz:.3f}m out of [0.4, 0.5]"

    def test_rk4_time_aware_better_than_steady(self):
        """RK4 with time-aware should be much more accurate than without."""
        mesh = _make_mesh()
        runtime = 3600.0
        dt = 1800.0
        n_steps = int(runtime / dt)

        # Steady run
        state = make_state(np.array([LON0]), np.array([LAT0]))
        for k in range(n_steps):
            vel = _make_vel(mesh, k * dt)
            state = rk4_step(state, vel, mesh, dt, time_dir=1.0)
        err_steady = _xy_err(state, runtime)

        # Time-aware run (vel_next at t + dt_to_next = t + dt)
        state = make_state(np.array([LON0]), np.array([LAT0]))
        for k in range(n_steps):
            t_k = k * dt
            vel = _make_vel(mesh, t_k)
            vel_next = _make_vel(mesh, t_k + dt)
            state = rk4_step(state, vel, mesh, dt, time_dir=1.0,
                             vel_next=vel_next, dt_to_next=dt)
        err_aware = _xy_err(state, runtime)

        print(f"\n  RK4 steady err={err_steady:.4e}m  "
              f"time-aware err={err_aware:.4e}m  "
              f"improvement={err_steady/err_aware:.1f}x")

        # Time-aware should be at least 10x better for this oscillating field
        assert err_aware < err_steady / 10, (
            f"time-aware ({err_aware:.4e}) not >>10x better than steady ({err_steady:.4e})"
        )

    def test_rk2_time_aware_better_than_steady(self):
        """Same for RK2."""
        mesh = _make_mesh()
        runtime = 3600.0
        dt = 1800.0
        n_steps = int(runtime / dt)

        state = make_state(np.array([LON0]), np.array([LAT0]))
        for k in range(n_steps):
            vel = _make_vel(mesh, k * dt)
            state = rk2_step(state, vel, mesh, dt, time_dir=1.0)
        err_steady = _xy_err(state, runtime)

        state = make_state(np.array([LON0]), np.array([LAT0]))
        for k in range(n_steps):
            t_k = k * dt
            vel = _make_vel(mesh, t_k)
            vel_next = _make_vel(mesh, t_k + dt)
            state = rk2_step(state, vel, mesh, dt, time_dir=1.0,
                             vel_next=vel_next, dt_to_next=dt)
        err_aware = _xy_err(state, runtime)

        print(f"\n  RK2 steady err={err_steady:.4e}m  "
              f"time-aware err={err_aware:.4e}m  "
              f"improvement={err_steady/err_aware:.1f}x")

        assert err_aware < err_steady / 5, (
            f"time-aware RK2 ({err_aware:.4e}) not >5x better than steady ({err_steady:.4e})"
        )

    def test_rk4_tinterp_alpha_zero_matches_steady(self):
        """When vel == vel_next (alphas=0) tinterp fused kernel must match
        the steady fused kernel to high precision.  Documents that the new
        tinterp path does not drift when used for a steady field.
        """
        mesh = _make_mesh()
        dt = 900.0
        rng = np.random.default_rng(3)
        npart = 5
        lons = rng.uniform(-0.05, 0.05, npart)
        lats = rng.uniform(-0.05, 0.05, npart)
        state_init = make_state(lons, lats)

        vel = _make_vel(mesh, 0.0)

        # Steady fused path
        s_steady = rk4_step(state_init.replace(), vel, mesh, dt, time_dir=1.0)
        # Tinterp fused path with identical snapshots (alpha_k*=0 effectively)
        s_tinterp = rk4_step(
            state_init.replace(), vel, mesh, dt, time_dir=1.0,
            vel_next=vel, dt_to_next=dt,
        )
        # Allow very small float drift from the extra blend multiplication
        np.testing.assert_allclose(s_steady.lon, s_tinterp.lon, atol=1e-11)
        np.testing.assert_allclose(s_steady.lat, s_tinterp.lat, atol=1e-11)

    def test_rk4_fused_vs_unfused_tinterp(self):
        """Fused RK4 tinterp kernel must match unfused RK4 with same inputs.

        Both should produce numerically-close results (allow tiny float drift
        from different summation order).
        """
        from kelp.numerics.integrators import _rk4_step_unfused
        mesh = _make_mesh()
        dt = 900.0

        # Seed particles around the centre
        rng = np.random.default_rng(7)
        npart = 5
        lons = rng.uniform(-0.05, 0.05, npart)
        lats = rng.uniform(-0.05, 0.05, npart)
        state_init = make_state(lons, lats)

        vel = _make_vel(mesh, 0.0)
        vel_next = _make_vel(mesh, dt)

        # Fused path
        s_fused = rk4_step(state_init.replace(), vel, mesh, dt, time_dir=1.0,
                           vel_next=vel_next, dt_to_next=dt)

        # Unfused path
        s_unfused = _rk4_step_unfused(
            state_init.replace(), vel, mesh, dt, time_dir=1.0,
            layer=0, is3d=False, w_interp="nearest",
            vel_next=vel_next, dt_to_next=dt,
        )

        # Allow small numerical diff (different sum ordering in Numba)
        np.testing.assert_allclose(s_fused.lon, s_unfused.lon, atol=1e-12)
        np.testing.assert_allclose(s_fused.lat, s_unfused.lat, atol=1e-12)

    def test_rk2_fused_vs_unfused_tinterp(self):
        """Fused RK2 tinterp kernel must match unfused RK2."""
        from kelp.numerics.integrators import _rk2_step_unfused
        mesh = _make_mesh()
        dt = 900.0
        rng = np.random.default_rng(11)
        npart = 5
        lons = rng.uniform(-0.05, 0.05, npart)
        lats = rng.uniform(-0.05, 0.05, npart)
        state_init = make_state(lons, lats)

        vel = _make_vel(mesh, 0.0)
        vel_next = _make_vel(mesh, dt)

        s_fused = rk2_step(state_init.replace(), vel, mesh, dt, time_dir=1.0,
                           vel_next=vel_next, dt_to_next=dt)
        s_unfused = _rk2_step_unfused(
            state_init.replace(), vel, mesh, dt, time_dir=1.0,
            layer=0, is3d=False, w_interp="nearest",
            vel_next=vel_next, dt_to_next=dt,
        )
        np.testing.assert_allclose(s_fused.lon, s_unfused.lon, atol=1e-12)
        np.testing.assert_allclose(s_fused.lat, s_unfused.lat, atol=1e-12)

    def test_euler_unaffected(self):
        """Euler is single-stage; vel_next/dt_to_next are accepted but
        do not change the result (single evaluation at t_step)."""
        mesh = _make_mesh()
        dt = 600.0
        vel = _make_vel(mesh, 0.0)
        vel_next = _make_vel(mesh, dt)

        s1 = make_state(np.array([LON0]), np.array([LAT0]))
        s1 = euler_step(s1, vel, mesh, dt, time_dir=1.0)

        s2 = make_state(np.array([LON0]), np.array([LAT0]))
        s2 = euler_step(s2, vel, mesh, dt, time_dir=1.0,
                        vel_next=vel_next, dt_to_next=dt)

        np.testing.assert_array_equal(s1.lon, s2.lon)
        np.testing.assert_array_equal(s1.lat, s2.lat)


# =====================================================================
# Edge cases
# =====================================================================

class TestEdgeCases:
    """Validate input guards and short-circuit behavior."""

    def test_dt_to_next_zero_raises(self):
        mesh = _make_mesh()
        vel = _make_vel(mesh, 0.0)
        vel_next = _make_vel(mesh, 1.0)
        state = make_state(np.array([LON0]), np.array([LAT0]))
        with pytest.raises(ValueError, match="dt_to_next"):
            rk4_step(state, vel, mesh, dt=600.0, time_dir=1.0,
                     vel_next=vel_next, dt_to_next=0.0)

    def test_dt_exceeds_dt_to_next_raises(self):
        """RK4: dt > dt_to_next stepping past the bracket endpoint."""
        mesh = _make_mesh()
        vel = _make_vel(mesh, 0.0)
        vel_next = _make_vel(mesh, 100.0)
        state = make_state(np.array([LON0]), np.array([LAT0]))
        with pytest.raises(ValueError, match="endpoint past next snapshot"):
            rk4_step(state, vel, mesh, dt=200.0, time_dir=1.0,
                     vel_next=vel_next, dt_to_next=100.0)

    def test_rk2_dt_exceeds_dt_to_next_raises(self):
        """RK2: dt > dt_to_next must raise even though c_max=0.5.

        Regression test: previously _make_stage_vels only checked
        c_max * dt > dt_to_next, so RK2 with dt = 1.5 * dt_to_next slipped
        through (dt/2 < dt_to_next) but the step endpoint went past the
        bracket.
        """
        mesh = _make_mesh()
        vel = _make_vel(mesh, 0.0)
        vel_next = _make_vel(mesh, 100.0)
        state = make_state(np.array([LON0]), np.array([LAT0]))
        with pytest.raises(ValueError, match="endpoint past next snapshot"):
            rk2_step(state, vel, mesh, dt=150.0, time_dir=1.0,
                     vel_next=vel_next, dt_to_next=100.0)

    def test_shape_mismatch_raises(self):
        """vel and vel_next must have matching u/v/w shapes/dimensionality."""
        mesh = _make_mesh()
        vel = _make_vel(mesh, 0.0)  # 2D, w=None
        # Build incompatible vel_next: with w array
        vel_next_3d = VelocityField(
            u=vel.u.copy(),
            v=vel.v.copy(),
            eta=vel.eta.copy(),
            w=np.zeros((5, mesh.n_nodes), dtype=np.float64),
        )
        state = make_state(np.array([LON0]), np.array([LAT0]))
        with pytest.raises(ValueError, match="vel.w and vel_next.w must both be"):
            rk4_step(state, vel, mesh, dt=600.0, time_dir=1.0,
                     vel_next=vel_next_3d, dt_to_next=600.0)

        # Wrong u shape
        vel_bad_u = VelocityField(
            u=vel.u[:5].copy(),  # truncated
            v=vel.v.copy(),
            eta=vel.eta.copy(),
            w=None,
        )
        with pytest.raises(ValueError, match="vel.u and vel_next.u shape mismatch"):
            rk4_step(state, vel, mesh, dt=600.0, time_dir=1.0,
                     vel_next=vel_bad_u, dt_to_next=600.0)

        # Wrong eta shape
        vel_bad_eta = VelocityField(
            u=vel.u.copy(),
            v=vel.v.copy(),
            eta=vel.eta[:3].copy(),  # truncated
            w=None,
        )
        with pytest.raises(ValueError, match="vel.eta and vel_next.eta shape mismatch"):
            rk4_step(state, vel, mesh, dt=600.0, time_dir=1.0,
                     vel_next=vel_bad_eta, dt_to_next=600.0)

    def test_backward_time_dir(self):
        """Time-aware RK4 should work with time_dir=-1 (backward tracking).

        For backward integration over [-dt, 0]: vel = U(t_step), vel_next = U(t_step - dt),
        dt_to_next = dt. RK4 stages should evaluate correctly going back in time.
        """
        mesh = _make_mesh()
        runtime = 1800.0
        dt = 900.0
        n_steps = int(runtime / dt)

        # Forward run from t=0
        s_fwd = make_state(np.array([LON0]), np.array([LAT0]))
        for k in range(n_steps):
            t_k = k * dt
            vel = _make_vel(mesh, t_k)
            vel_next = _make_vel(mesh, t_k + dt)
            s_fwd = rk4_step(s_fwd, vel, mesh, dt, time_dir=1.0,
                             vel_next=vel_next, dt_to_next=dt)
        assert s_fwd.status[0] == ParticleStatus.ALIVE

        # Backward run from t=runtime back to 0 — should return near origin
        s_bwd = s_fwd.replace()
        for k in range(n_steps):
            t_k = runtime - k * dt
            vel = _make_vel(mesh, t_k)
            vel_next = _make_vel(mesh, t_k - dt)
            s_bwd = rk4_step(s_bwd, vel, mesh, dt, time_dir=-1.0,
                             vel_next=vel_next, dt_to_next=dt)
        # Should return to origin within RK4 precision
        cos_lat0 = np.cos(LAT0 * DEG2RAD)
        x_ret = (s_bwd.lon[0] - LON0) * cos_lat0 * METERS_PER_DEG_LAT
        y_ret = (s_bwd.lat[0] - LAT0) * METERS_PER_DEG_LAT
        ret_err = np.sqrt(x_ret ** 2 + y_ret ** 2)
        print(f"\n  backward closure: |Δr|={ret_err:.4e}m")
        assert ret_err < 0.5, f"backward roundtrip {ret_err:.4f}m too large"

    def test_partial_spec_raises(self):
        """Providing only one of (vel_next, dt_to_next) is an error."""
        mesh = _make_mesh()
        vel = _make_vel(mesh, 0.0)
        vel_next = _make_vel(mesh, 1000.0)
        state = make_state(np.array([LON0]), np.array([LAT0]))
        with pytest.raises(ValueError, match="provided together"):
            rk4_step(state, vel, mesh, dt=600.0, time_dir=1.0,
                     vel_next=vel_next)  # dt_to_next missing
        with pytest.raises(ValueError, match="provided together"):
            rk4_step(state, vel, mesh, dt=600.0, time_dir=1.0,
                     dt_to_next=600.0)  # vel_next missing
