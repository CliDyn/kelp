"""Tests for vertical Az diffusion (TODO 2.2b).

Covers, in TDD order: the scalar kick kernel (moments, Itô drift,
forward/backward symmetry, determinism), the reflective boundary fold,
nearest-node column Az evaluation, the state-level split hook (incl. the
well-mixed condition), config validation, VelocityField/LazyLerpField
plumbing, the FESOM2 ``var_az`` read, and tracking-level wiring.

Design notes: docs/diffusion.md (vertical Az section).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kelp.core.mesh import find_nearest_nodes  # noqa: E402
from kelp.core.particles import ParticleStatus, make_state  # noqa: E402
from kelp.numerics.diffusion import (  # noqa: E402
    COMP_HORIZONTAL,
    COMP_VERTICAL,
    DiffusionStepArgs,
    diffusion_az_step_meters,
)

from test_3d_analytical import _make_mesh, _make_vel_3d  # noqa: E402


def _az_args(az_const=0.0, az_enabled=True, seed=1, step_i=0):
    """DiffusionStepArgs for a vertical-only kick (no horizontal Kh)."""
    return DiffusionStepArgs(
        scheme="uniform", kh_x=0.0, kh_y=0.0,
        seed=seed, real_id=0, step_i=step_i,
        az_enabled=az_enabled, az_const=az_const,
    )


def _state_at(mesh, P, z0):
    """P particles at the mesh centre with valid nearest nodes."""
    st = make_state(np.full(P, 5.0), np.full(P, 5.0),
                    np.asarray(z0, dtype=np.float64) * np.ones(P))
    nn = find_nearest_nodes(mesh, st.lon, st.lat)
    return st.replace(nearest_node=nn)


# ---------------------------------------------------------------------------
# Scalar kick kernel
# ---------------------------------------------------------------------------

def _collect_kicks(az, daz_dz, dt, P, seed=42, real_id=0, step_i=3):
    dz = np.empty(P, dtype=np.float64)
    for i in range(P):
        dz[i] = diffusion_az_step_meters(
            az, daz_dz, dt,
            np.uint64(seed), np.uint64(real_id), np.uint64(i),
            np.uint64(step_i),
        )
    return dz


class TestAzKickKernel:
    def test_comp_vertical_slot(self):
        """The reserved counter-RNG component slot for the vertical kick."""
        assert COMP_VERTICAL == 1
        assert COMP_VERTICAL != COMP_HORIZONTAL

    def test_constant_az_moments(self):
        """Zero-gradient kick is unbiased with variance 2·Az·dt."""
        az, dt, P = 0.01, 600.0, 100_000
        dz = _collect_kicks(az, 0.0, dt, P)
        expected_var = 2.0 * az * dt
        # SE of the mean ≈ sqrt(var/P) ≈ 0.011 m — allow ~4σ.
        assert abs(dz.mean()) < 0.045
        assert abs(dz.var() / expected_var - 1.0) < 0.05

    def test_ito_drift_shifts_mean(self):
        """A nonzero ∂Az/∂z contributes the deterministic drift b·dt.

        Visser (1997): without this term particles pile up in Az minima.
        """
        az, b, dt, P = 1.0e-4, 1.0e-4, 600.0, 100_000
        dz = _collect_kicks(az, b, dt, P)
        drift = b * dt  # 0.06 m; noise SE of the mean ≈ 0.0011 m
        assert abs(dz.mean() - drift) < 0.1 * drift
        # The drift must not change the spread.
        assert abs(dz.var() / (2.0 * az * dt) - 1.0) < 0.05

    def test_backward_dt_keeps_the_drift(self):
        """The Itô drift is part of the diffusion, not of the advection.

        Backward tracking reverses the advection only: the adjoint of the
        advection-diffusion operator keeps ``+dAz/dz`` (review 2026-09-22,
        M06).  A signed drift would give a backward stationary density
        proportional to Az**-2 in a closed column.
        """
        az, b, dt, P = 1.0e-4, 1.0e-4, 600.0, 100_000
        dz_fwd = _collect_kicks(az, b, dt, P)
        dz_back = _collect_kicks(az, b, -dt, P)
        # same RNG keys, drift and noise both direction-free
        np.testing.assert_array_equal(dz_back, dz_fwd)
        assert abs(dz_back.mean() - b * dt) < 0.1 * b * dt
        assert abs(dz_back.var() / (2.0 * az * dt) - 1.0) < 0.05

    def test_deterministic_and_component_independent(self):
        """Same key → identical kick; the draw sits on COMP_VERTICAL."""
        args = (0.01, 0.0, 600.0, np.uint64(7), np.uint64(1),
                np.uint64(123), np.uint64(9))
        a = diffusion_az_step_meters(*args)
        b = diffusion_az_step_meters(*args)
        assert a == b
        # The vertical draw must NOT reuse the horizontal component:
        # the same key on COMP_HORIZONTAL yields a different normal.
        from kelp.numerics.diffusion import _standard_normal
        g_v = _standard_normal(np.uint64(7), np.uint64(1), np.uint64(123),
                               np.uint64(9), np.uint64(COMP_VERTICAL))
        g_h = _standard_normal(np.uint64(7), np.uint64(1), np.uint64(123),
                               np.uint64(9), np.uint64(COMP_HORIZONTAL))
        assert g_v != g_h
        assert a == np.sqrt(2.0 * 0.01 * 600.0) * g_v


# ---------------------------------------------------------------------------
# Reflective boundary fold
# ---------------------------------------------------------------------------

class TestReflectDepth:
    def test_inside_unchanged(self):
        from kelp.numerics.diffusion import reflect_depth
        assert reflect_depth(5.0, 0.0, 10.0) == 5.0

    def test_surface_fold(self):
        from kelp.numerics.diffusion import reflect_depth
        assert reflect_depth(-1.5, 0.0, 10.0) == pytest.approx(1.5)

    def test_bottom_fold(self):
        from kelp.numerics.diffusion import reflect_depth
        assert reflect_depth(11.0, 0.0, 10.0) == pytest.approx(9.0)

    def test_nonzero_top(self):
        from kelp.numerics.diffusion import reflect_depth
        assert reflect_depth(1.0, 2.0, 10.0) == pytest.approx(3.0)

    def test_multi_bounce(self):
        from kelp.numerics.diffusion import reflect_depth
        # 37 m into a 10 m column: fold period 20 → 37 mod 20 = 17 → 3.
        assert reflect_depth(37.0, 0.0, 10.0) == pytest.approx(3.0)
        # Deep negative overshoot: -23 mod 20 = 17 → 3.
        assert reflect_depth(-23.0, 0.0, 10.0) == pytest.approx(3.0)

    def test_boundaries_are_fixed_points(self):
        from kelp.numerics.diffusion import reflect_depth
        assert reflect_depth(0.0, 0.0, 10.0) == 0.0
        assert reflect_depth(10.0, 0.0, 10.0) == pytest.approx(10.0)

    def test_degenerate_column(self):
        from kelp.numerics.diffusion import reflect_depth
        assert reflect_depth(5.0, 3.0, 3.0) == 3.0     # zero height
        assert reflect_depth(5.0, 4.0, 3.0) == 4.0     # inverted → top

    def test_result_always_in_range(self):
        from kelp.numerics.diffusion import reflect_depth
        rng = np.random.default_rng(0)
        for z in rng.uniform(-100.0, 100.0, size=500):
            r = reflect_depth(float(z), 1.0, 7.5)
            assert 1.0 <= r <= 7.5


# ---------------------------------------------------------------------------
# Nearest-node column Az evaluation (field mode, z-level)
# ---------------------------------------------------------------------------

class TestAzColumnValueGradient:
    """Az field is node-based (K, N) at interfaces, like ``w``."""

    Z_LEVELS = np.array([0.0, 10.0, 20.0, 30.0, 40.0])

    def _field(self, col):
        """One-node mesh column → (K, 1) field."""
        return np.asarray(col, dtype=np.float64).reshape(-1, 1)

    def test_linear_profile_exact(self):
        from kelp.numerics.diffusion import az_column_value_gradient
        a, b = 2.0e-3, 1.0e-4
        field = self._field(a + b * self.Z_LEVELS)
        az, g = az_column_value_gradient(field, self.Z_LEVELS, 0, 15.0, 40.0)
        assert az == pytest.approx(a + b * 15.0)
        assert g == pytest.approx(b)

    def test_quadratic_profile_layer_slope(self):
        from kelp.numerics.diffusion import az_column_value_gradient
        field = self._field((self.Z_LEVELS / 10.0) ** 2)  # 0, 1, 4, 9, 16
        az, g = az_column_value_gradient(field, self.Z_LEVELS, 0, 15.0, 40.0)
        # Value: lerp inside layer [10, 20] → 2.5.
        assert az == pytest.approx(2.5)
        # Gradient: slope of the segment [10, 20] → (4 − 1) / 10 = 0.3.
        assert g == pytest.approx(0.3)

    def test_surface_one_sided(self):
        from kelp.numerics.diffusion import az_column_value_gradient
        field = self._field((self.Z_LEVELS / 10.0) ** 2)
        az, g = az_column_value_gradient(field, self.Z_LEVELS, 0, 0.0, 40.0)
        assert az == pytest.approx(0.0)
        # Slope of the top segment [0, 10]: (1 − 0) / 10 = 0.1 (the field
        # is linear BETWEEN interfaces, not the continuous quadratic).
        assert g == pytest.approx(0.1)

    def test_bottom_aware_clamp(self):
        from kelp.numerics.diffusion import az_column_value_gradient
        a, b = 2.0e-3, 1.0e-4
        field = self._field(a + b * self.Z_LEVELS)
        # Node bottom at 25 m — evaluation below bottom clamps to 25.
        az, g = az_column_value_gradient(field, self.Z_LEVELS, 0, 28.0, 25.0)
        assert az == pytest.approx(a + b * 25.0)
        assert g == pytest.approx(b)  # one-sided upward, linear → exact

    def test_gradient_is_slope_of_the_interpolant(self):
        """The drift is dAz/dz of the SAME piecewise-linear Az the noise
        uses (review 2026-09-22, M03).  The old half-layer central
        difference straddled the next interface: at 0.75 m on
        Az = 1, 2, 1 at 0, 1, 2 m it returned 0.5 instead of 1.0."""
        from kelp.numerics.diffusion import az_column_value_gradient
        field = self._field([1.0, 2.0, 1.0])
        z = np.array([0.0, 1.0, 2.0])
        az, g = az_column_value_gradient(field, z, 0, 0.75, 2.0)
        assert az == pytest.approx(1.75)
        assert g == pytest.approx(1.0)
        az, g = az_column_value_gradient(field, z, 0, 1.25, 2.0)
        assert az == pytest.approx(1.75)
        assert g == pytest.approx(-1.0)
        # exactly on an interface: the segment above it (the layer search)
        _, g = az_column_value_gradient(field, z, 0, 1.0, 2.0)
        assert g == pytest.approx(1.0)

    def test_uniform_density_is_stationary_for_the_generator(self):
        """Generator check, free of Monte-Carlo noise: for a closed column
        and any f with f' = 0 at both walls, the uniform density is
        stationary iff  ∫ (b f' + Az f'') dz = 0.  With b the derivative of
        the interpolant this is integration by parts; the old stencil gave
        0.363 (normalised) on this profile."""
        from kelp.numerics.diffusion import az_column_value_gradient
        field = self._field([1.0, 2.0, 1.0])
        zl = np.array([0.0, 1.0, 2.0])
        n = 20000
        zc = (np.arange(n) + 0.5) * (2.0 / n)
        fp = -np.pi * np.sin(np.pi * zc)
        fpp = -np.pi ** 2 * np.cos(np.pi * zc)
        acc = 0.0
        for zi, a1, a2 in zip(zc, fp, fpp):
            az, b = az_column_value_gradient(field, zl, 0, float(zi), 2.0)
            acc += b * a1 + az * a2
        assert abs(acc * (2.0 / n) / 2.0) < 1e-5

    def test_single_interface_column(self):
        from kelp.numerics.diffusion import az_column_value_gradient
        field = self._field([3.0e-3])
        az, g = az_column_value_gradient(
            field, np.array([0.0]), 0, 5.0, 40.0)
        assert az == pytest.approx(3.0e-3)
        assert g == 0.0


# ---------------------------------------------------------------------------
# State-level split hook
# ---------------------------------------------------------------------------

class TestApplySplitAzKick:
    def test_az_disabled_returns_same_object(self):
        from kelp.numerics.integrators import _apply_split_az_kick
        mesh = _make_mesh(5.0, 5.0)
        vel = _make_vel_3d(mesh)
        post = _state_at(mesh, 4, 100.0)
        out = _apply_split_az_kick(post, post, mesh, 600.0,
                                   _az_args(az_const=1.0, az_enabled=False),
                                   vel)
        assert out is post

    def test_no_diffusion_returns_same_object(self):
        from kelp.numerics.integrators import _apply_split_az_kick
        mesh = _make_mesh(5.0, 5.0)
        vel = _make_vel_3d(mesh)
        post = _state_at(mesh, 4, 100.0)
        assert _apply_split_az_kick(post, post, mesh, 600.0, None, vel) is post

    def test_zero_source_short_circuits(self):
        from kelp.numerics.integrators import _apply_split_az_kick
        mesh = _make_mesh(5.0, 5.0)
        vel = _make_vel_3d(mesh)
        post = _state_at(mesh, 4, 100.0)
        out = _apply_split_az_kick(post, post, mesh, 600.0,
                                   _az_args(az_const=0.0), vel)
        assert out is post

    def test_2d_field_raises(self):
        from kelp.numerics.integrators import _apply_split_az_kick
        from kelp.core.io import VelocityField
        mesh = _make_mesh(5.0, 5.0)
        ne, nn = mesh.n_elements, mesh.n_nodes
        vel2d = VelocityField(u=np.zeros(ne), v=np.zeros(ne),
                              eta=np.zeros(nn))
        post = _state_at(mesh, 4, 0.0)
        with pytest.raises(ValueError, match="az"):
            _apply_split_az_kick(post, post, mesh, 600.0,
                                 _az_args(az_const=1e-3), vel2d)

    def test_const_kick_variance_and_bounds(self):
        from kelp.numerics.integrators import _apply_split_az_kick
        mesh = _make_mesh(5.0, 5.0)          # bottom 200 m
        vel = _make_vel_3d(mesh)
        P, az, dt = 20_000, 1.0e-3, 600.0
        post = _state_at(mesh, P, 100.0)
        out = _apply_split_az_kick(post, post, mesh, dt,
                                   _az_args(az_const=az), vel)
        dz = out.z - post.z
        assert not np.allclose(dz, 0.0)
        assert np.all(out.z >= 0.0) and np.all(out.z <= 200.0)
        assert abs(dz.var() / (2.0 * az * dt) - 1.0) < 0.05
        assert abs(dz.mean()) < 4.0 * np.sqrt(2.0 * az * dt / P)

    @pytest.mark.parametrize("dead_status", [
        ParticleStatus.OUTSIDE, ParticleStatus.BEACHED,
        ParticleStatus.NOT_RELEASED,
    ])
    def test_dead_particles_untouched(self, dead_status):
        from kelp.numerics.integrators import _apply_split_az_kick
        mesh = _make_mesh(5.0, 5.0)
        vel = _make_vel_3d(mesh)
        post = _state_at(mesh, 6, 100.0)
        status = post.status.copy()
        status[::2] = dead_status
        post = post.replace(status=status)
        out = _apply_split_az_kick(post, post, mesh, 600.0,
                                   _az_args(az_const=1e-3), vel)
        assert np.array_equal(out.z[::2], post.z[::2])
        assert not np.allclose(out.z[1::2], post.z[1::2])

    def test_backward_drift_kept_at_hook_level(self):
        """time_dir=-1 with POSITIVE dt must NOT reverse the Itô drift.

        Tracking passes a positive dt and carries the direction in
        time_dir; the hook receives the signed dt, and the kernel must
        use |dt| for the drift as for the noise (review 2026-09-22, M06).
        """
        from kelp.numerics.integrators import _apply_split_az_kick
        mesh = _make_mesh(5.0, 5.0)
        vel = _make_vel_3d(mesh)
        b = 1.0e-4
        vel.az = np.repeat((1.0e-6 + b * mesh.sigma)[:, None],
                           mesh.n_nodes, axis=1)
        P, dt, z0 = 200_000, 600.0, 10.0
        post = _state_at(mesh, P, z0)
        out = _apply_split_az_kick(post, post, mesh, dt,
                                   _az_args(az_const=0.0), vel,
                                   time_dir=-1.0)
        drift = b * dt
        assert abs((out.z - post.z).mean() - drift) < 0.15 * abs(drift)

    def test_field_required_but_missing_raises(self):
        """az_requires_field pins a configured var_az to a present field.

        If a payload ever arrives without .az while the run configured
        var_az, the kick must fail loud instead of silently degrading
        to az_const (hardening; unreachable through run_tracking today).
        """
        from kelp.numerics.integrators import _apply_split_az_kick
        mesh = _make_mesh(5.0, 5.0)
        vel = _make_vel_3d(mesh)          # no .az on the payload
        post = _state_at(mesh, 4, 100.0)
        args = _az_args(az_const=1e-3)._replace(az_requires_field=True)
        with pytest.raises(ValueError, match="var_az"):
            _apply_split_az_kick(post, post, mesh, 600.0, args, vel)

    def test_negative_az_zeroes_kick_entirely(self):
        """Corrupt negative Az must not leave a live gradient drift."""
        from kelp.numerics.integrators import _apply_split_az_kick
        mesh = _make_mesh(5.0, 5.0)
        vel = _make_vel_3d(mesh)
        # Strongly negative, strongly sloped garbage field.
        vel.az = np.repeat((-1.0 - 0.01 * mesh.sigma)[:, None],
                           mesh.n_nodes, axis=1)
        post = _state_at(mesh, 100, 50.0)
        out = _apply_split_az_kick(post, post, mesh, 600.0,
                                   _az_args(az_const=0.0), vel)
        assert np.array_equal(out.z, post.z)

    def test_horizontal_and_status_preserved(self):
        from kelp.numerics.integrators import _apply_split_az_kick
        mesh = _make_mesh(5.0, 5.0)
        vel = _make_vel_3d(mesh)
        post = _state_at(mesh, 8, 50.0)
        out = _apply_split_az_kick(post, post, mesh, 600.0,
                                   _az_args(az_const=1e-3), vel)
        assert np.array_equal(out.lon, post.lon)
        assert np.array_equal(out.lat, post.lat)
        assert np.array_equal(out.status, post.status)
        assert np.array_equal(out.nearest_node, post.nearest_node)

    def test_field_az_drift(self):
        """With a field Az the Itô drift pushes the mean by ∂Az/∂z·dt."""
        from kelp.numerics.integrators import _apply_split_az_kick
        mesh = _make_mesh(5.0, 5.0)
        vel = _make_vel_3d(mesh)
        b = 1.0e-4
        profile = 1.0e-6 + b * mesh.sigma          # linear in z
        vel.az = np.repeat(profile[:, None], mesh.n_nodes, axis=1)
        P, dt, z0 = 200_000, 600.0, 10.0
        post = _state_at(mesh, P, z0)
        out = _apply_split_az_kick(post, post, mesh, dt,
                                   _az_args(az_const=0.0), vel)
        drift = b * dt
        assert abs((out.z - post.z).mean() - drift) < 0.15 * drift

    @pytest.mark.parametrize("time_dir", [1.0, -1.0])
    def test_well_mixed_condition(self, time_dir):
        """Uniformly seeded particles STAY uniform in nonuniform Az.

        Forward and backward alike: the backward run reverses advection
        only (review 2026-09-22, M06); with the old signed drift a
        backward ensemble drifted towards the Az minimum.

        The standard Visser criterion: with the Itô drift in place the
        equilibrium density in a closed column is uniform.  Without the
        drift the density would pile up toward the Az minimum
        (p ∝ 1/Az — a factor-10 surface excess for this profile),
        which the bin bounds below reject by a wide margin.
        """
        from kelp.numerics.integrators import _apply_split_az_kick
        mesh = _make_mesh(5.0, 5.0)          # column [0, 200] m
        vel = _make_vel_3d(mesh)
        profile = 1.0e-3 + (9.0e-3 / 200.0) * mesh.sigma   # 1e-3 → 1e-2
        vel.az = np.repeat(profile[:, None], mesh.n_nodes, axis=1)
        P, dt, n_steps = 2000, 300.0, 5000
        z0 = np.linspace(0.02, 199.98, P)
        state = _state_at(mesh, P, 0.0).replace(z=z0.copy())
        base = _az_args(az_const=0.0)
        for k in range(n_steps):
            state = _apply_split_az_kick(
                state, state, mesh, dt, base._replace(step_i=k), vel,
                time_dir=time_dir)
        counts, _ = np.histogram(state.z, bins=8, range=(0.0, 200.0))
        expected = P / 8.0
        assert np.all(np.abs(counts - expected) < 0.2 * expected), counts
        assert abs(state.z.mean() - 100.0) < 6.0

    def test_sigma_const_kick_in_column(self):
        from kelp.numerics.integrators import _apply_split_az_kick
        mesh = _make_mesh(5.0, 5.0, bottom_depth=50.0)
        mesh.sigma = np.linspace(1.0, 0.0, 6)   # sigma interfaces
        mesh.vertical_coord = "sigma"
        vel = _make_vel_3d(mesh)
        vel.eta[:] = 0.5
        P, az, dt = 5000, 1.0e-3, 600.0
        post = _state_at(mesh, P, 25.0)
        out = _apply_split_az_kick(post, post, mesh, dt,
                                   _az_args(az_const=az), vel,
                                   Dmin=0.5, z0b_min=0.1)
        dz = out.z - post.z
        assert not np.allclose(dz, 0.0)
        # Column: [0, depth + eta − z0b_min] = [0, 50.4].
        assert np.all(out.z >= 0.0) and np.all(out.z <= 50.5 - 0.1)
        assert abs(dz.var() / (2.0 * az * dt) - 1.0) < 0.10

    def test_sigma_negative_handbuilt_const_dormant(self):
        """A hand-built negative az_const (config rejects it, direct
        callers can still pass it) must be treated as no-source, never
        reach sqrt(negative) in the sigma loop."""
        from kelp.numerics.integrators import _apply_split_az_kick
        mesh = _make_mesh(5.0, 5.0, bottom_depth=50.0)
        mesh.sigma = np.linspace(1.0, 0.0, 6)
        mesh.vertical_coord = "sigma"
        vel = _make_vel_3d(mesh)
        post = _state_at(mesh, 8, 25.0)
        out = _apply_split_az_kick(post, post, mesh, 600.0,
                                   _az_args(az_const=-1.0), vel)
        assert out is post

    def test_sigma_field_az_raises(self):
        from kelp.numerics.integrators import _apply_split_az_kick
        mesh = _make_mesh(5.0, 5.0, bottom_depth=50.0)
        mesh.sigma = np.linspace(1.0, 0.0, 6)
        mesh.vertical_coord = "sigma"
        vel = _make_vel_3d(mesh)
        vel.az = np.ones((6, mesh.n_nodes)) * 1e-3
        post = _state_at(mesh, 4, 25.0)
        with pytest.raises(ValueError, match="sigma"):
            _apply_split_az_kick(post, post, mesh, 600.0,
                                 _az_args(az_const=0.0), vel)


# ---------------------------------------------------------------------------
# Integrator wiring: the kick fires inside euler/rk2/rk4 steps
# ---------------------------------------------------------------------------

class TestIntegratorWiring:
    @pytest.mark.parametrize("name", ["euler", "rk2", "rk4"])
    def test_step_applies_az_kick(self, name):
        from kelp.numerics.integrators import get_integrator
        mesh = _make_mesh(5.0, 5.0)
        vel = _make_vel_3d(mesh)                 # zero flow, w = 0
        P, az, dt = 5000, 1.0e-3, 600.0
        state = _state_at(mesh, P, 100.0)
        step = get_integrator(name)
        out = step(state, vel, mesh, dt, 1.0,
                   diffusion=_az_args(az_const=az))
        dz = out.z - state.z
        assert not np.allclose(dz, 0.0)
        assert np.all(out.z >= 0.0) and np.all(out.z <= 200.0)
        assert abs(dz.var() / (2.0 * az * dt) - 1.0) < 0.10

    def test_step_without_az_leaves_z(self):
        """kh-only diffusion must not move z beyond the clamp."""
        from kelp.numerics.integrators import get_integrator
        mesh = _make_mesh(5.0, 5.0)
        vel = _make_vel_3d(mesh)
        state = _state_at(mesh, 100, 100.0)
        args = DiffusionStepArgs(scheme="uniform", kh_x=5.0, kh_y=5.0,
                                 seed=1, real_id=0, step_i=0)
        out = get_integrator("euler")(state, vel, mesh, 600.0, 1.0,
                                      diffusion=args)
        assert np.array_equal(out.z, state.z)

    def test_horizontal_and_vertical_kicks_compose(self):
        from kelp.numerics.integrators import get_integrator
        mesh = _make_mesh(5.0, 5.0)
        vel = _make_vel_3d(mesh)
        P = 2000
        state = _state_at(mesh, P, 100.0)
        args = DiffusionStepArgs(scheme="uniform", kh_x=2.0, kh_y=2.0,
                                 seed=1, real_id=0, step_i=0,
                                 az_enabled=True, az_const=1e-3)
        out = get_integrator("euler")(state, vel, mesh, 600.0, 1.0,
                                      diffusion=args)
        assert not np.allclose(out.lon, state.lon)
        assert not np.allclose(out.lat, state.lat)
        assert not np.allclose(out.z, state.z)

    def test_2d_step_with_az_raises(self):
        from kelp.numerics.integrators import get_integrator
        from kelp.core.io import VelocityField
        mesh = _make_mesh(5.0, 5.0)
        vel2d = VelocityField(u=np.zeros(mesh.n_elements),
                              v=np.zeros(mesh.n_elements),
                              eta=np.zeros(mesh.n_nodes))
        state = _state_at(mesh, 4, 0.0)
        with pytest.raises(ValueError, match="az"):
            get_integrator("euler")(state, vel2d, mesh, 600.0, 1.0,
                                    diffusion=_az_args(az_const=1e-3))

    @pytest.mark.parametrize("name", ["euler", "rk2", "rk4"])
    def test_step_backward_drift_kept(self, name):
        """Positive dt + time_dir=-1 through the integrator: drift kept."""
        from kelp.numerics.integrators import get_integrator
        mesh = _make_mesh(5.0, 5.0)
        vel = _make_vel_3d(mesh)
        b = 1.0e-4
        vel.az = np.repeat((1.0e-6 + b * mesh.sigma)[:, None],
                           mesh.n_nodes, axis=1)
        P, dt = 100_000, 600.0
        state = _state_at(mesh, P, 10.0)
        out = get_integrator(name)(state, vel, mesh, dt, -1.0,
                                   diffusion=_az_args(az_const=0.0))
        drift = b * dt
        assert abs((out.z - state.z).mean() - drift) < 0.2 * abs(drift)

    @pytest.mark.parametrize("scheme,composition", [
        ("em", "combined"), ("em", "split_rk"), ("m1", "split_rk"),
    ])
    def test_em_m1_args_with_az_raise_in_step(self, scheme, composition):
        """Hand-built em/m1 + az args must raise, never silently skip az.

        Config forbids the combo; direct integrator callers bypass the
        config, and the combined/split_rk branches never reach the az
        hook — without this backstop az would be dropped silently.
        """
        from kelp.numerics.integrators import get_integrator
        from kelp.core.io import VelocityField
        mesh = _make_mesh(5.0, 5.0)
        vel2d = VelocityField(u=np.zeros(mesh.n_elements),
                              v=np.zeros(mesh.n_elements),
                              eta=np.zeros(mesh.n_nodes))
        state = _state_at(mesh, 4, 0.0)
        args = DiffusionStepArgs(
            scheme=scheme, kh_x=0.0, kh_y=0.0, seed=1, real_id=0,
            step_i=0, composition=composition,
            az_enabled=True, az_const=1e-3)
        integ = "euler" if composition == "combined" else "rk2"
        with pytest.raises(ValueError, match="az_enabled"):
            get_integrator(integ)(state, vel2d, mesh, 600.0, 1.0,
                                  diffusion=args)

    @pytest.mark.parametrize("name", ["euler", "rk2", "rk4"])
    def test_em_m1_az_backstop_fires_even_all_dead(self, name):
        """The backstop must sit before any all-dead early return —
        an invalid combo raises deterministically, not only while
        particles happen to be alive."""
        from kelp.numerics.integrators import get_integrator
        from kelp.core.io import VelocityField
        mesh = _make_mesh(5.0, 5.0)
        vel2d = VelocityField(u=np.zeros(mesh.n_elements),
                              v=np.zeros(mesh.n_elements),
                              eta=np.zeros(mesh.n_nodes))
        state = _state_at(mesh, 4, 0.0)
        state = state.replace(status=np.full(
            4, ParticleStatus.OUTSIDE, dtype=state.status.dtype))
        args = DiffusionStepArgs(
            scheme="em", kh_x=0.0, kh_y=0.0, seed=1, real_id=0,
            step_i=0, composition="split_rk",
            az_enabled=True, az_const=1e-3)
        with pytest.raises(ValueError, match="az_enabled"):
            get_integrator(name)(state, vel2d, mesh, 600.0, 1.0,
                                 diffusion=args)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestDiffusionConfigAz:
    def test_defaults_off(self):
        from kelp.core.config import DiffusionConfig
        cfg = DiffusionConfig()
        assert cfg.az_enabled is False
        assert cfg.az_const == 0.0
        assert cfg.var_az is None

    def test_negative_az_const_raises(self):
        from kelp.core.config import DiffusionConfig
        with pytest.raises(ValueError, match="az_const"):
            DiffusionConfig(az_const=-1.0)

    @pytest.mark.parametrize("scheme", ["em", "m1"])
    def test_az_with_variable_kh_scheme_raises(self, scheme):
        """v1: the Az kick composes with the uniform/split path only.

        ``enabled=True`` here on purpose: with the default False the
        enabled-coupling guard fires first and this test would pass
        even with the em/m1 check deleted (round-2 review finding).
        """
        from kelp.core.config import DiffusionConfig
        with pytest.raises(ValueError, match="em|m1|uniform"):
            DiffusionConfig(enabled=True, scheme=scheme, az_enabled=True)

    def test_az_enabled_without_enabled_raises(self):
        """az_enabled under a disabled diffusion block would silently
        no-op through the whole stack — reject it at config time."""
        from kelp.core.config import DiffusionConfig
        with pytest.raises(ValueError, match="enabled"):
            DiffusionConfig(enabled=False, az_enabled=True, az_const=1e-4)

    def test_yaml_section_flows_through(self, tmp_path):
        from kelp.core.config import load_config
        cfg_path = tmp_path / "c.yaml"
        cfg_path.write_text(
            "mode: {Fdrift2D: false, Fdrift3D: true}\n"
            "numerics:\n"
            "  diffusion:\n"
            "    enabled: true\n"
            "    az_enabled: true\n"
            "    az_const: 0.001\n"
            "    var_az: Av\n"
            "mesh: {type: fesom2, meshpath: m.nc}\n"
            "data: {ncinput: d.nc, inicloud: p.dat}\n"
            "time: {start_time_idx: 0, n_steps: 2}\n"
        )
        cfg = load_config(str(cfg_path))
        d = cfg.numerics.diffusion
        assert d.az_enabled is True
        assert d.az_const == pytest.approx(1.0e-3)
        assert d.var_az == "Av"


# ---------------------------------------------------------------------------
# VelocityField / LazyLerpField plumbing
# ---------------------------------------------------------------------------

def _vel3d_with_az(mesh, az_value):
    vel = _make_vel_3d(mesh)
    K, N = mesh.sigma.shape[0], mesh.n_nodes
    from kelp.core.io import VelocityField
    return VelocityField(u=vel.u, v=vel.v, eta=vel.eta, w=vel.w,
                         az=np.full((K, N), az_value, dtype=np.float64))


class TestVelocityFieldAz:
    def test_default_none(self):
        mesh = _make_mesh(5.0, 5.0)
        assert _make_vel_3d(mesh).az is None

    def test_lerp_blends_az(self):
        mesh = _make_mesh(5.0, 5.0)
        a = _vel3d_with_az(mesh, 1.0e-3)
        b = _vel3d_with_az(mesh, 3.0e-3)
        out = a.lerp(b, 0.5)
        assert out.az is not None
        assert np.allclose(out.az, 2.0e-3)

    def test_lerp_one_sided_az_drops(self):
        mesh = _make_mesh(5.0, 5.0)
        a = _vel3d_with_az(mesh, 1.0e-3)
        b = _make_vel_3d(mesh)
        assert a.lerp(b, 0.5).az is None

    def test_lerp_az_shape_mismatch_raises(self):
        from kelp.core.io import VelocityField
        mesh = _make_mesh(5.0, 5.0)
        a = _vel3d_with_az(mesh, 1.0e-3)
        b = _make_vel_3d(mesh)
        bad = VelocityField(u=b.u, v=b.v, eta=b.eta, w=b.w,
                            az=np.zeros((2, mesh.n_nodes)))
        with pytest.raises(ValueError, match="az"):
            a.lerp(bad, 0.5)

    def test_lazy_lerp_blends_az(self):
        from kelp.core.io import LazyLerpField
        mesh = _make_mesh(5.0, 5.0)
        a = _vel3d_with_az(mesh, 1.0e-3)
        b = _vel3d_with_az(mesh, 3.0e-3)
        lazy = LazyLerpField(a, b, 0.25)
        assert np.allclose(lazy.az, 1.5e-3)
        assert np.allclose(lazy.materialize().az, 1.5e-3)

    def test_lazy_lerp_one_sided_az_none(self):
        from kelp.core.io import LazyLerpField
        mesh = _make_mesh(5.0, 5.0)
        a = _vel3d_with_az(mesh, 1.0e-3)
        b = _make_vel_3d(mesh)
        assert LazyLerpField(a, b, 0.5).az is None

    def test_lazy_lerp_az_shape_mismatch_raises(self):
        from kelp.core.io import LazyLerpField, VelocityField
        mesh = _make_mesh(5.0, 5.0)
        a = _vel3d_with_az(mesh, 1.0e-3)
        b = _make_vel_3d(mesh)
        bad = VelocityField(u=b.u, v=b.v, eta=b.eta, w=b.w,
                            az=np.zeros((2, mesh.n_nodes)))
        with pytest.raises(ValueError, match="az"):
            LazyLerpField(a, bad, 0.5)


# ---------------------------------------------------------------------------
# FESOM2 reader: var_az
# ---------------------------------------------------------------------------

import xarray as xr  # noqa: E402

from kelp.core.config import DiffusionConfig, ModeConfig  # noqa: E402

_MODE_3D = ModeConfig(Fdrift2D=False, Fdrift3D=True)


def _az_diff_cfg(var_az="Av"):
    return DiffusionConfig(enabled=True, az_enabled=True, var_az=var_az)


class TestFesom2VarAzRead:
    K, N, E, T = 5, 9, 8, 2
    Z = [0.0, 10.0, 25.0, 50.0, 100.0]

    def _mesh(self):
        return _make_mesh(5.0, 5.0, z_levels=self.Z, bottom_depth=100.0)

    def _ds(self, az=None, az_dims=("time", "nz", "nod2")):
        data = {
            "u": (("time", "nz1", "elem"),
                  np.zeros((self.T, self.K - 1, self.E))),
            "v": (("time", "nz1", "elem"),
                  np.zeros((self.T, self.K - 1, self.E))),
            "w": (("time", "nz", "nod2"),
                  np.zeros((self.T, self.K, self.N))),
            "ssh": (("time", "nod2"), np.zeros((self.T, self.N))),
        }
        if az is not None:
            data["Av"] = (az_dims, az)
        return xr.Dataset(data)

    def test_reads_az_field_and_zero_fills(self):
        from kelp.readers.fesom2 import Fesom2Reader
        mesh = self._mesh()
        az = np.full((self.T, self.K, self.N), 2.0e-3)
        az[:, -1, 0] = np.nan          # below-bottom fill value
        vel = Fesom2Reader.read_velocities(
            self._ds(az), 0, mesh, _MODE_3D, diffusion=_az_diff_cfg())
        assert vel.az is not None
        assert vel.az.shape == (self.K, self.N)
        assert vel.az[0, 0] == pytest.approx(2.0e-3)
        assert vel.az[-1, 0] == 0.0    # NaN zero-filled
        assert vel.az.dtype == np.float64

    def test_max_levels_trims_az(self):
        from kelp.readers.fesom2 import Fesom2Reader
        mesh = self._mesh()
        mode = ModeConfig(Fdrift2D=False, Fdrift3D=True, max_levels=3)
        az = np.full((self.T, self.K, self.N), 1.0e-3)
        vel = Fesom2Reader.read_velocities(
            self._ds(az), 0, mesh, mode, diffusion=_az_diff_cfg())
        assert vel.az.shape == (3, self.N)

    def test_missing_var_raises(self):
        from kelp.readers.fesom2 import Fesom2Reader
        mesh = self._mesh()
        with pytest.raises(ValueError, match="Kv"):
            Fesom2Reader.read_velocities(
                self._ds(), 0, mesh, _MODE_3D,
                diffusion=_az_diff_cfg(var_az="Kv"))

    def test_mid_layer_shaped_var_raises(self):
        from kelp.readers.fesom2 import Fesom2Reader
        mesh = self._mesh()
        az = np.full((self.T, self.K - 1, self.N), 1.0e-3)
        with pytest.raises(ValueError, match="interface"):
            Fesom2Reader.read_velocities(
                self._ds(az, az_dims=("time", "nz1", "nod2")), 0, mesh,
                _MODE_3D, diffusion=_az_diff_cfg())

    def test_no_var_az_reads_none(self):
        from kelp.readers.fesom2 import Fesom2Reader
        mesh = self._mesh()
        vel = Fesom2Reader.read_velocities(
            self._ds(), 0, mesh, _MODE_3D,
            diffusion=DiffusionConfig(enabled=True, az_enabled=True,
                                      az_const=1e-3))
        assert vel.az is None

    def test_read_skipped_when_az_disabled(self):
        """var_az staged in the YAML with the kick OFF (A/B config):
        no read, no failure on the absent variable, no extra I/O."""
        from kelp.readers.fesom2 import Fesom2Reader
        mesh = self._mesh()
        vel = Fesom2Reader.read_velocities(
            self._ds(), 0, mesh, _MODE_3D,     # ds has NO Av at all
            diffusion=DiffusionConfig(enabled=True, az_enabled=False,
                                      var_az="Av"))
        assert vel.az is None

    def test_read_skipped_when_diffusion_disabled(self):
        from kelp.readers.fesom2 import Fesom2Reader
        mesh = self._mesh()
        vel = Fesom2Reader.read_velocities(
            self._ds(), 0, mesh, _MODE_3D,
            diffusion=DiffusionConfig(enabled=False, var_az="Av"))
        assert vel.az is None

    def test_transposed_storage_oriented(self):
        """(time, nod2, nz) storage must be oriented, not rejected —
        the same house rule u/v and scalars follow."""
        from kelp.readers.fesom2 import Fesom2Reader
        mesh = self._mesh()
        az = np.zeros((self.T, self.N, self.K))
        az[:, :, :] = np.arange(self.K)[None, None, :]   # value = level idx
        vel = Fesom2Reader.read_velocities(
            self._ds(az, az_dims=("time", "nod2", "nz")), 0, mesh,
            _MODE_3D, diffusion=_az_diff_cfg())
        assert vel.az.shape == (self.K, self.N)
        assert np.allclose(vel.az[:, 0], np.arange(self.K))

    def test_mid_layer_rejected_even_with_max_levels(self):
        """max_levels trimming must not let a (K-1) mid-layer variable
        slip past the interface-shape check (reviewer finding)."""
        from kelp.readers.fesom2 import Fesom2Reader
        mesh = self._mesh()
        mode = ModeConfig(Fdrift2D=False, Fdrift3D=True, max_levels=3)
        az = np.full((self.T, self.K - 1, self.N), 1.0e-3)
        with pytest.raises(ValueError, match="interface"):
            Fesom2Reader.read_velocities(
                self._ds(az, az_dims=("time", "nz1", "nod2")), 0, mesh,
                mode, diffusion=_az_diff_cfg())

    def test_mid_layer_depth_named_dim_rejected(self):
        """A (K-1) variable on a non-canonical 'depth' dim is classified
        mid-layer by PRE-TRIM size against the file's nz dim (round-2
        review) — max_levels must not mask it either."""
        from kelp.readers.fesom2 import Fesom2Reader
        mesh = self._mesh()
        mode = ModeConfig(Fdrift2D=False, Fdrift3D=True, max_levels=3)
        az = np.full((self.T, self.K - 1, self.N), 1.0e-3)
        with pytest.raises(ValueError, match="interface"):
            Fesom2Reader.read_velocities(
                self._ds(az, az_dims=("time", "depth", "nod2")), 0, mesh,
                mode, diffusion=_az_diff_cfg())


# ---------------------------------------------------------------------------
# Multi-year exact layout: the Av family must be assembled
# ---------------------------------------------------------------------------

class TestMultiyearVarAz:
    def _family(self, tmp_path, with_av=True):
        from tests.test_fesom2_multiyear import write_multiyear
        yl = {1958: 365, 1959: 365}
        for v in ("unod", "vnod"):
            write_multiyear(tmp_path, v, yl, node_layout=True,
                            vertical="nz1")
        write_multiyear(tmp_path, "w", yl, node_layout=True, vertical="nz")
        write_multiyear(tmp_path, "temp", yl, node_layout=True,
                        vertical="nz1")
        write_multiyear(tmp_path, "ssh", yl, node_layout=True,
                        vertical=None)
        if with_av:
            write_multiyear(tmp_path, "Av", yl, node_layout=True,
                            vertical="nz")

    def test_exact_layout_assembles_av(self, tmp_path):
        """cycle1-style per-var-per-year output: Av.fesom.<year>.nc must
        join the assembled family when the az kick is on (review B4 —
        without this the feature is unusable on its target dataset)."""
        from tests.test_fesom2_multiyear import _mode
        from kelp.readers.fesom2 import Fesom2Reader
        self._family(tmp_path)
        ds = Fesom2Reader.open_dataset(
            str(tmp_path), mode_cfg=_mode(),
            diffusion=_az_diff_cfg(var_az="Av"))
        assert "Av" in ds.data_vars
        assert ds.sizes["time"] == 730

    def test_exact_layout_av_not_pulled_when_dormant(self, tmp_path):
        from tests.test_fesom2_multiyear import _mode
        from kelp.readers.fesom2 import Fesom2Reader
        self._family(tmp_path)
        ds = Fesom2Reader.open_dataset(str(tmp_path), mode_cfg=_mode())
        assert "Av" not in ds.data_vars

    def test_exact_layout_av_missing_raises(self, tmp_path):
        from tests.test_fesom2_multiyear import _mode
        from kelp.readers.fesom2 import Fesom2Reader
        self._family(tmp_path, with_av=False)
        with pytest.raises((ValueError, FileNotFoundError), match="Av"):
            Fesom2Reader.open_dataset(
                str(tmp_path), mode_cfg=_mode(),
                diffusion=_az_diff_cfg(var_az="Av"))

    @pytest.mark.parametrize("years", [{1958: 3, 1959: 3}, {1958: 3}])
    def test_mixed_layout_raises(self, tmp_path, years):
        """Per-year Av in a directory whose velocities are NOT in the
        exact layout must fail loud — the broad glob would outer-align
        it and silently zero the kick outside the covered year(s).
        Even a SINGLE yearly-pattern file is anomalous there."""
        from tests.test_fesom2_multiyear import write_multiyear, _mode
        from kelp.readers.fesom2 import Fesom2Reader
        write_multiyear(tmp_path, "Av", years,
                        node_layout=True, vertical="nz")
        with pytest.raises(ValueError, match="exact|mixed"):
            Fesom2Reader.open_dataset(
                str(tmp_path), mode_cfg=_mode(),
                diffusion=_az_diff_cfg(var_az="Av"))


# ---------------------------------------------------------------------------
# run_tracking startup guards + end-to-end
# ---------------------------------------------------------------------------

class TestTrackingGuardsAndE2E:
    def _fesomc_2d_config(self, tmp_path, **diff_kw):
        from tests.test_tracking_diffusion import (
            _make_fesomc_wide, _build_app_config, _write_particles)
        nc, _ = _make_fesomc_wide(tmp_path)
        seeds = tmp_path / "p.dat"
        _write_particles(seeds)
        out = tmp_path / "out.nc"
        cfg = _build_app_config(nc, seeds, out)
        cfg.numerics.diffusion = DiffusionConfig(enabled=True, **diff_kw)
        return cfg

    def test_az_2d_run_raises(self, tmp_path):
        from kelp.tracking import run_tracking
        cfg = self._fesomc_2d_config(tmp_path, az_enabled=True,
                                     az_const=1.0e-3)
        with pytest.raises(ValueError, match="3-D|Fdrift3D"):
            run_tracking(cfg)

    def test_var_az_non_fesom2_raises(self, tmp_path):
        from kelp.tracking import run_tracking
        cfg = self._fesomc_2d_config(tmp_path, az_enabled=True,
                                     var_az="Av")
        with pytest.raises(ValueError, match="fesom2"):
            run_tracking(cfg)

    def test_zero_source_az_dormant_everywhere(self, tmp_path):
        """az_enabled with a zero source must be fully dormant: a 2-D
        run completes (no guard fires) and the output carries no
        diffusion_az_* attrs — the byte-identity the docs promise."""
        import netCDF4
        from kelp.tracking import run_tracking
        cfg = self._fesomc_2d_config(tmp_path, az_enabled=True,
                                     az_const=0.0)
        run_tracking(cfg)
        out = cfg.data.output_path
        with netCDF4.Dataset(out) as nc:
            assert "diffusion_az_enabled" not in nc.ncattrs()
            assert "diffusion_az_const" not in nc.ncattrs()

    def _fesom2_3d_cfg(self, tmp_path, az_profile=None, az_const=0.0,
                       var_az=None, n_steps=6, az_dims=("time", "nz", "nod2"),
                       temporal_interp=False, dt=600.0, seed=7, n_time=4):
        from tests.test_tracking import _make_fesom2_3x3_3d
        from kelp.core.config import (AppConfig, DataConfig, MeshConfig,
                                          NumericsConfig, TimeConfig)
        mesh_dir, vel_path, _ = _make_fesom2_3x3_3d(
            tmp_path, n_time=n_time, u_val=0.0, v_val=0.0, w_val=0.0)
        if az_profile is not None:
            ds = xr.load_dataset(vel_path)
            prof = np.asarray(az_profile, dtype=np.float64)
            sizes = {"time": ds.sizes["time"], "nod2": ds.sizes["nod2"]}
            vdim = next(d for d in az_dims if d not in sizes)
            sizes[vdim] = prof.size
            expand = [None, None, None]
            expand[az_dims.index(vdim)] = slice(None)
            az = np.broadcast_to(
                prof[tuple(expand)],
                tuple(sizes[d] for d in az_dims)).copy()
            ds["Av"] = (az_dims, az)
            ds.to_netcdf(vel_path, mode="w")
        seeds = tmp_path / "p3d.dat"
        seeds.write_text("".join(f"5.0 5.0 {z}\n"
                                 for z in np.linspace(10, 90, 16)))
        out = tmp_path / "traj_az.nc"
        return AppConfig(
            mode=ModeConfig(Fdrift2D=False, Fdrift3D=True),
            numerics=NumericsConfig(
                dt=dt, integrator="euler",
                temporal_interp=temporal_interp,
                # the fixture's time axis is an index: interpolation needs
                # the interval declared (it used to fall back to a discrete
                # run silently — review 2026-09-22, F03)
                file_dt=3600.0 if temporal_interp else None,
                diffusion=dict(enabled=True, az_enabled=True,
                               az_const=az_const, var_az=var_az, seed=seed),
            ),
            mesh=MeshConfig(type="fesom2", meshpath=str(tmp_path / "f2mesh3d")),
            data=DataConfig(ncinput=str(vel_path), inicloud=str(seeds),
                            output_path=str(out), time_averaging="instant"),
            time=TimeConfig(start_time_idx=0, n_steps=n_steps),
        )

    def _fesom2_3d_run(self, tmp_path, **kw):
        from kelp.tracking import run_tracking
        cfg = self._fesom2_3d_cfg(tmp_path, **kw)
        run_tracking(cfg)
        return cfg.data.output_path

    def test_e2e_const_az_and_metadata(self, tmp_path):
        import netCDF4
        out = self._fesom2_3d_run(tmp_path, az_const=1.0e-3)
        with netCDF4.Dataset(out) as nc:
            z = np.array(nc.variables["z"][:])
            assert nc.getncattr("diffusion_az_enabled") == 1
            assert float(nc.getncattr("diffusion_az_const")) == \
                pytest.approx(1.0e-3)
        dz_final = z[:, -1] - z[:, 0]
        assert np.std(dz_final) > 0.1          # kicks actually moved z
        assert np.all(z[np.isfinite(z)] >= 0.0)
        assert np.all(z[np.isfinite(z)] <= 100.0)

    def test_e2e_field_az_runs(self, tmp_path):
        import netCDF4
        profile = 1.0e-3 + np.array([0.0, 10.0, 25.0, 50.0, 100.0]) * 1e-5
        out = self._fesom2_3d_run(tmp_path, az_profile=profile,
                                  var_az="Av")
        with netCDF4.Dataset(out) as nc:
            assert nc.getncattr("diffusion_var_az") == "Av"
            z = np.array(nc.variables["z"][:])
        assert np.std(z[:, -1] - z[:, 0]) > 0.1

    def test_probe_rejects_mid_layer_before_writer(self, tmp_path):
        """A wrong-layout Av must fail BEFORE the output file exists
        (metadata-only probe, mirroring the sample_vars fail-fast)."""
        from kelp.tracking import run_tracking
        profile = np.full(4, 1.0e-3)               # K-1 mid-layers
        cfg = self._fesom2_3d_cfg(tmp_path, az_profile=profile,
                                  var_az="Av",
                                  az_dims=("time", "nz1", "nod2"))
        with pytest.raises(ValueError, match="interface"):
            run_tracking(cfg)
        assert not Path(cfg.data.output_path).exists()

    def test_e2e_temporal_interp_field_az(self, tmp_path):
        """LazyLerpField.az through the real tracking loop."""
        import netCDF4
        profile = 1.0e-3 + np.array([0.0, 10.0, 25.0, 50.0, 100.0]) * 1e-5
        out = self._fesom2_3d_run(tmp_path, az_profile=profile,
                                  var_az="Av", temporal_interp=True,
                                  dt=300.0, n_steps=8)
        with netCDF4.Dataset(out) as nc:
            z = np.array(nc.variables["z"][:])
        assert np.std(z[:, -1] - z[:, 0]) > 0.05
        assert np.all(z[np.isfinite(z)] >= 0.0)

    def test_restart_bitwise_with_az(self, tmp_path):
        """Resume via start_step_i reproduces a continuous 3-D az run
        bit-for-bit — pins the GLOBAL-step_i RNG keying of the vertical
        draw at the tracking layer (mirrors the horizontal restart test
        in test_tracking_diffusion.py)."""
        from kelp.readers import get_reader
        from kelp.tracking import run_single_realization
        from kelp.core.config import NumericsConfig, DiffusionConfig
        from kelp.core.particles import ParticleState, make_state

        cfg = self._fesom2_3d_cfg(tmp_path, az_const=1.0e-3, n_steps=6,
                                  n_time=8)
        reader = get_reader("fesom2")
        mesh = reader.load_mesh(cfg.mesh)
        ds = reader.open_dataset(cfg.data.ncinput)
        init = make_state(np.full(16, 5.0), np.full(16, 5.0),
                          np.linspace(10, 90, 16))
        num = cfg.numerics

        rec_cont = run_single_realization(
            mesh, ds, reader, init, cfg.mode, num,
            start_time_idx=0, n_steps=6, n_time_total=8)

        rec_a = run_single_realization(
            mesh, ds, reader, init, cfg.mode, num,
            start_time_idx=0, n_steps=3, n_time_total=8)
        last = rec_a.to_arrays()
        mid_state = ParticleState(
            lon=last["lon"][-1].copy(), lat=last["lat"][-1].copy(),
            z=last["z"][-1].copy(),
            status=last["status"][-1].astype(np.int8).copy(),
            nearest_node=np.full(init.n_particles, -1, dtype=np.int32))
        rec_b = run_single_realization(
            mesh, ds, reader, mid_state, cfg.mode, num,
            start_time_idx=3, n_steps=3, n_time_total=8,
            start_step_i=3)
        ds.close()

        cont = rec_cont.to_arrays()
        b = rec_b.to_arrays()
        np.testing.assert_array_equal(cont["z"][-1], b["z"][-1])


class TestShardOffsetThroughTheHook:
    """Audit 2026-10-04 (option a), review round: the offset reaches the Az
    kick through ``_apply_split_az_kick`` on both vertical coordinates —
    a block of particles kicked with ``pid_offset = lo`` moves exactly like
    the same particles inside the whole release."""

    @staticmethod
    def _check(mesh, vel, **kw):
        from kelp.numerics.integrators import _apply_split_az_kick
        rng = np.random.default_rng(4)
        post = _state_at(mesh, 30, 0.0).replace(z=rng.uniform(5.0, 40.0, 30))
        args = _az_args(az_const=2e-3, seed=7, step_i=5)
        whole = _apply_split_az_kick(post, post, mesh, 600.0, args, vel, **kw)
        lo = 12
        sub = post.replace(lon=post.lon[lo:], lat=post.lat[lo:], z=post.z[lo:],
                           status=post.status[lo:],
                           nearest_node=post.nearest_node[lo:])
        part = _apply_split_az_kick(sub, sub, mesh, 600.0,
                                    args._replace(pid_offset=lo), vel, **kw)
        np.testing.assert_array_equal(part.z, whole.z[lo:])
        plain = _apply_split_az_kick(sub, sub, mesh, 600.0, args, vel, **kw)
        assert not np.array_equal(plain.z, whole.z[lo:])

    def test_zlevel(self):
        mesh = _make_mesh(5.0, 5.0)
        self._check(mesh, _make_vel_3d(mesh))

    def test_sigma(self):
        mesh = _make_mesh(5.0, 5.0, bottom_depth=50.0)
        mesh.sigma = np.linspace(1.0, 0.0, 6)
        mesh.vertical_coord = "sigma"
        vel = _make_vel_3d(mesh)
        vel.eta[:] = 0.5
        self._check(mesh, vel, Dmin=0.5, z0b_min=0.1)
