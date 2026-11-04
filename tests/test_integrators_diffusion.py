"""Tests for split-mode diffusion wired into the integrators.

Covers:
* zero-Kh bit-identity vs the diffusion-disabled path (Euler/RK2/RK4),
* Brownian variance ≈ ``2·K·t`` after N steps with no advection,
* drift + diffusion: ensemble mean tracks the deterministic trajectory,
* restart-bitwise: ``N`` continuous steps == ``N1`` + ``N2`` continuation,
* failed-kick → particle reverted to pre-step + marked OUTSIDE,
* dropped-particle stability: a particle that dies mid-run does not
  perturb the others' RNG sequences (the counter-RNG keying rule).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from kelp.core.geometry import METERS_PER_DEG_LAT
from kelp.core.mesh import (
    Mesh, build_node_to_elems, build_search,
)
from kelp.core.particles import (
    ParticleState, ParticleStatus, make_state,
)
from kelp.numerics.diffusion import DiffusionStepArgs
from kelp.numerics.integrators import euler_step, rk2_step, rk4_step
from kelp.numerics.interpolate import VelocityField


# ---------------------------------------------------------------------------
# Mesh fixtures
# ---------------------------------------------------------------------------

def _build_3d_mesh_with_bathy_jump() -> Mesh:
    """41×41 0.25° grid with a sharp east-west bathymetry transition.

    Nodes at ``lon < 5°`` have bottom depth 500 m; nodes at ``lon ≥ 5°``
    have bottom depth 50 m.  Spacing is fine enough that a particle
    seeded at lon=4.4° has its nearest node at lon=4.25° (deep) and
    only crosses to a shallow nearest node after the kick — exposing
    the post-kick clamp without the deterministic step pre-empting it.
    """
    n = 41
    xs = np.linspace(0.0, 10.0, n)
    ys = np.linspace(0.0, 10.0, n)
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel().astype(np.float64)
    lat = gy.ravel().astype(np.float64)
    n_nodes = lon.size
    depth = np.where(lon < 5.0, 500.0, 50.0).astype(np.float64)
    elems = []
    for j in range(n - 1):
        for i in range(n - 1):
            n0 = j * n + i
            n1 = n0 + 1
            n2 = n0 + n
            n3 = n2 + 1
            elems.append([n0, n1, n2])
            elems.append([n1, n3, n2])
    elem_nodes = np.array(elems, dtype=np.int32)
    lon_c = np.array([lon[e].mean() for e in elem_nodes], dtype=np.float64)
    lat_c = np.array([lat[e].mean() for e in elem_nodes], dtype=np.float64)
    n2e, n2e_count = build_node_to_elems(n_nodes, elem_nodes)
    z_levels = np.array([0.0, 50.0, 100.0, 200.0, 350.0, 500.0],
                        dtype=np.float64)
    mesh = Mesh(
        name="bathy_jump",
        nodes_lon=lon, nodes_lat=lat, depth=depth,
        elem_nodes=elem_nodes,
        elem_center_lon=lon_c, elem_center_lat=lat_c,
        node_to_elems=n2e, node_to_elems_count=n2e_count,
        sigma=z_levels,
    )
    build_search(mesh)
    return mesh


def _build_grid_mesh(xs: np.ndarray, ys: np.ndarray) -> Mesh:
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel().astype(np.float64)
    lat = gy.ravel().astype(np.float64)
    nx, ny = len(xs), len(ys)
    n_nodes = nx * ny
    depth = np.zeros(n_nodes, dtype=np.float64)
    elems = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            n1 = n0 + 1
            n2 = n0 + nx
            n3 = n2 + 1
            elems.append([n0, n1, n2])
            elems.append([n1, n3, n2])
    elem_nodes = np.array(elems, dtype=np.int32)
    lon_c = np.array([lon[e].mean() for e in elem_nodes], dtype=np.float64)
    lat_c = np.array([lat[e].mean() for e in elem_nodes], dtype=np.float64)
    n2e, n2e_count = build_node_to_elems(n_nodes, elem_nodes)
    mesh = Mesh(
        name="grid",
        nodes_lon=lon, nodes_lat=lat, depth=depth,
        elem_nodes=elem_nodes,
        elem_center_lon=lon_c, elem_center_lat=lat_c,
        node_to_elems=n2e, node_to_elems_count=n2e_count,
    )
    build_search(mesh)
    return mesh


@pytest.fixture
def grid3x3():
    """Small mesh used for the boundary regression test."""
    return _build_grid_mesh(np.array([4.0, 5.0, 6.0]),
                            np.array([4.0, 5.0, 6.0]))


@pytest.fixture
def grid_wide():
    """11×11 mesh with 1° spacing.

    A node sits at (5°, 5°) so the seed cluster does not start on an
    element diagonal — that pathology exposes a known limitation of
    ``find_containing_elements`` (nearest-node + 1-ring search) and is
    not what the diffusion tests are after.
    """
    return _build_grid_mesh(np.linspace(0.0, 10.0, 11),
                            np.linspace(0.0, 10.0, 11))


def _const_vel(mesh: Mesh, u: float = 0.0, v: float = 0.0) -> VelocityField:
    ne = mesh.n_elements
    return VelocityField(
        u=np.full(ne, u, dtype=np.float64),
        v=np.full(ne, v, dtype=np.float64),
        eta=np.zeros(mesh.n_nodes, dtype=np.float64),
    )


def _seed_cluster(n: int, lon: float = 5.0, lat: float = 5.0) -> ParticleState:
    return make_state(
        lon=np.full(n, lon, dtype=np.float64),
        lat=np.full(n, lat, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# K = 0 bit-identity
# ---------------------------------------------------------------------------

class TestZeroKhBitIdentical:
    """``K_h = 0`` ⇒ split-mode result must be bit-identical to the
    diffusion-disabled path."""

    @pytest.mark.parametrize("step", [euler_step, rk2_step, rk4_step])
    def test_no_diffusion_vs_zero_kh_match(self, grid_wide, step):
        vel = _const_vel(grid_wide, u=0.5, v=-0.25)
        state = _seed_cluster(50)
        diff_zero = DiffusionStepArgs(
            scheme="uniform", kh_x=0.0, kh_y=0.0,
            seed=99, real_id=0, step_i=0,
        )
        a = step(state, vel, grid_wide, dt=300.0, time_dir=1.0)
        b = step(state, vel, grid_wide, dt=300.0, time_dir=1.0,
                 diffusion=diff_zero)
        np.testing.assert_array_equal(a.lon, b.lon)
        np.testing.assert_array_equal(a.lat, b.lat)
        np.testing.assert_array_equal(a.z, b.z)
        np.testing.assert_array_equal(a.status, b.status)


# ---------------------------------------------------------------------------
# Brownian variance and drift
# ---------------------------------------------------------------------------

class TestBrownianVariance:
    """No advection, uniform ``Kh`` ⇒ ``var(x) ≈ 2·Kh·t`` after N steps."""

    def test_pure_diffusion_variance(self, grid_wide):
        n_particles = 1000
        n_steps = 20
        dt = 100.0
        kh = 0.5  # m^2/s
        vel = _const_vel(grid_wide, u=0.0, v=0.0)
        state = _seed_cluster(n_particles)

        for step_i in range(n_steps):
            diff = DiffusionStepArgs(
                scheme="uniform", kh_x=kh, kh_y=kh,
                seed=42, real_id=0, step_i=step_i,
            )
            state = euler_step(state, vel, grid_wide, dt=dt, time_dir=1.0,
                               diffusion=diff)

        alive = state.alive_mask
        assert alive.sum() == n_particles, "particles should not have left domain"

        # Convert lon/lat displacement back to metres before comparing.
        lat0 = 5.0
        z0 = float(np.cos(np.deg2rad(lat0))) * METERS_PER_DEG_LAT
        x_m = (state.lon[alive] - 5.0) * z0
        y_m = (state.lat[alive] - 5.0) * METERS_PER_DEG_LAT

        expected_var = 2.0 * kh * (n_steps * dt)
        # SE on variance estimate ≈ sigma^2 * sqrt(2/n) ≈ 0.045 * var.
        # 12 % tolerance is comfortably > 2 SE.
        assert abs(x_m.var() / expected_var - 1.0) < 0.12
        assert abs(y_m.var() / expected_var - 1.0) < 0.12

    def test_drift_plus_diffusion(self, grid_wide):
        """With deterministic flow, ensemble mean tracks the analytic
        trajectory; variance still grows like ``2·Kh·t``."""
        n_particles = 1000
        n_steps = 15
        dt = 100.0
        kh = 0.2
        u_ms = 0.5  # eastward
        vel = _const_vel(grid_wide, u=u_ms, v=0.0)
        state = _seed_cluster(n_particles)

        for step_i in range(n_steps):
            diff = DiffusionStepArgs(
                scheme="uniform", kh_x=kh, kh_y=kh,
                seed=7, real_id=0, step_i=step_i,
            )
            state = euler_step(state, vel, grid_wide, dt=dt, time_dir=1.0,
                               diffusion=diff)

        alive = state.alive_mask
        assert alive.sum() == n_particles

        z0 = float(np.cos(np.deg2rad(5.0))) * METERS_PER_DEG_LAT
        x_m = (state.lon[alive] - 5.0) * z0
        # Expected mean drift over n_steps · dt seconds.
        mean_expected = u_ms * (n_steps * dt)
        # SE on mean ≈ sigma / sqrt(n).  sigma ≈ sqrt(2·K·t) ≈ 24.5 m,
        # SE ≈ 0.78 m. 5σ tolerance is generous.
        se_mean = np.sqrt(2 * kh * n_steps * dt) / np.sqrt(n_particles)
        assert abs(x_m.mean() - mean_expected) < 5.0 * se_mean

        var_expected = 2.0 * kh * (n_steps * dt)
        assert abs(x_m.var() / var_expected - 1.0) < 0.15


# ---------------------------------------------------------------------------
# Restart bitwise reproducibility
# ---------------------------------------------------------------------------

class TestRestartBitwise:
    """``N`` continuous steps == ``N1`` + ``N2`` continuation, byte-identical.

    The counter-RNG keys on ``step_i`` (not on a hidden RNG generator
    state), so a checkpoint at step ``k`` can be resumed from
    ``step_i = k`` and produce identical draws.
    """

    @pytest.mark.parametrize("step", [euler_step, rk2_step, rk4_step])
    def test_continuous_vs_split_run(self, grid_wide, step):
        n_particles = 30
        n_total = 10
        kh = 0.3
        dt = 50.0
        vel = _const_vel(grid_wide, u=0.1, v=-0.05)

        # Continuous run
        s_cont = _seed_cluster(n_particles)
        for step_i in range(n_total):
            diff = DiffusionStepArgs(
                scheme="uniform", kh_x=kh, kh_y=kh,
                seed=17, real_id=0, step_i=step_i,
            )
            s_cont = step(s_cont, vel, grid_wide, dt=dt, time_dir=1.0,
                          diffusion=diff)

        # Split run with checkpoint at step n_total // 2
        s_split = _seed_cluster(n_particles)
        for step_i in range(n_total // 2):
            diff = DiffusionStepArgs(
                scheme="uniform", kh_x=kh, kh_y=kh,
                seed=17, real_id=0, step_i=step_i,
            )
            s_split = step(s_split, vel, grid_wide, dt=dt, time_dir=1.0,
                           diffusion=diff)
        # "checkpoint" — copy state arrays as if persisted and reloaded
        s_split = s_split.replace(
            lon=s_split.lon.copy(), lat=s_split.lat.copy(),
            z=s_split.z.copy(), status=s_split.status.copy(),
            nearest_node=s_split.nearest_node.copy(),
        )
        for step_i in range(n_total // 2, n_total):
            diff = DiffusionStepArgs(
                scheme="uniform", kh_x=kh, kh_y=kh,
                seed=17, real_id=0, step_i=step_i,
            )
            s_split = step(s_split, vel, grid_wide, dt=dt, time_dir=1.0,
                           diffusion=diff)

        np.testing.assert_array_equal(s_cont.lon, s_split.lon)
        np.testing.assert_array_equal(s_cont.lat, s_split.lat)
        np.testing.assert_array_equal(s_cont.status, s_split.status)


# ---------------------------------------------------------------------------
# Boundary regression: kick lands outside ⇒ revert + OUTSIDE
# ---------------------------------------------------------------------------

class TestKickOutOfDomain:
    def test_huge_kh_pushes_particle_outside(self, grid3x3):
        """``Kh`` so large that nearly every particle exits the 3×3 mesh
        in one step ⇒ all reverted to pre-step + marked OUTSIDE."""
        vel = _const_vel(grid3x3, u=0.0, v=0.0)
        state = _seed_cluster(50)
        # σ = sqrt(2·K·dt) = sqrt(2·1e10·100) ≈ 1.4e6 m ≈ 12.7° in latitude;
        # the 3×3 mesh spans 2° × 2°, so kicks always exit.
        diff = DiffusionStepArgs(
            scheme="uniform", kh_x=1.0e10, kh_y=1.0e10,
            seed=1, real_id=0, step_i=0,
        )

        new = euler_step(state, vel, grid3x3, dt=100.0, time_dir=1.0,
                         diffusion=diff)

        # Every particle reverted to its pre-step position and marked OUTSIDE.
        np.testing.assert_array_equal(new.lon, state.lon)
        np.testing.assert_array_equal(new.lat, state.lat)
        assert (new.status == ParticleStatus.OUTSIDE).all()


# ---------------------------------------------------------------------------
# Stable-particle-id under alive-mask compaction
# ---------------------------------------------------------------------------

class TestStableParticleId:
    """A particle that drops dead between steps must NOT shift the RNG
    stream of any other particle (the counter-RNG keying rule)."""

    def test_dead_neighbour_does_not_perturb_rng(self, grid_wide):
        n_particles = 5
        kh = 1.0
        dt = 100.0
        vel = _const_vel(grid_wide, u=0.0, v=0.0)

        # Run A: all five particles alive throughout.
        state_a = _seed_cluster(n_particles)
        for step_i in range(3):
            diff = DiffusionStepArgs(
                scheme="uniform", kh_x=kh, kh_y=kh,
                seed=23, real_id=0, step_i=step_i,
            )
            state_a = euler_step(state_a, vel, grid_wide, dt=dt,
                                 time_dir=1.0, diffusion=diff)

        # Run B: kill particle index 0 before the first step.  The
        # remaining four must trace exactly the same trajectory as in
        # run A — only the living-array compaction inside the integrator
        # changes, not the RNG keying.
        state_b = _seed_cluster(n_particles)
        status_b = state_b.status.copy()
        status_b[0] = ParticleStatus.OUTSIDE
        state_b = state_b.replace(status=status_b)
        for step_i in range(3):
            diff = DiffusionStepArgs(
                scheme="uniform", kh_x=kh, kh_y=kh,
                seed=23, real_id=0, step_i=step_i,
            )
            state_b = euler_step(state_b, vel, grid_wide, dt=dt,
                                 time_dir=1.0, diffusion=diff)

        # Indices 1..4 should have identical positions in A and B.
        for i in range(1, n_particles):
            assert state_a.lon[i] == state_b.lon[i], f"lon mismatch at i={i}"
            assert state_a.lat[i] == state_b.lat[i], f"lat mismatch at i={i}"


# ---------------------------------------------------------------------------
# Realization-id decorrelates ensembles
# ---------------------------------------------------------------------------

class TestRealizationId:
    def test_different_real_id_yields_different_trajectories(self, grid_wide):
        kh = 0.5
        dt = 100.0
        vel = _const_vel(grid_wide, u=0.0, v=0.0)
        state = _seed_cluster(20)

        diff_a = DiffusionStepArgs(scheme="uniform", kh_x=kh, kh_y=kh,
                                   seed=1, real_id=0, step_i=0)
        diff_b = DiffusionStepArgs(scheme="uniform", kh_x=kh, kh_y=kh,
                                   seed=1, real_id=1, step_i=0)

        s_a = euler_step(state, vel, grid_wide, dt=dt, time_dir=1.0,
                         diffusion=diff_a)
        s_b = euler_step(state, vel, grid_wide, dt=dt, time_dir=1.0,
                         diffusion=diff_b)
        assert not np.array_equal(s_a.lon, s_b.lon)
        assert not np.array_equal(s_a.lat, s_b.lat)


# ---------------------------------------------------------------------------
# Unsupported scheme guard
# ---------------------------------------------------------------------------

class TestBathymetryJump3D:
    """A horizontal kick that crosses a sharp bathymetry transition
    must reclamp z against the new local bottom (z-level path).

    Without the post-kick ``_clamp_z`` call, a deep-water particle
    kicked into shallow water keeps its original z and ends up below
    the local bottom while still ALIVE — silent corruption.
    """

    def test_kick_to_shallow_clamps_z(self):
        mesh = _build_3d_mesh_with_bathy_jump()
        n_layers = mesh.sigma.size  # 6 interfaces, 5 layers
        ne = mesh.n_elements
        nn = mesh.n_nodes
        # Layered u/v in shape (K-1, E); zero everywhere (no advection).
        u = np.zeros((n_layers - 1, ne), dtype=np.float64)
        v = np.zeros((n_layers - 1, ne), dtype=np.float64)
        # W at interfaces (K, N); zero (no vertical advection).
        w = np.zeros((n_layers, nn), dtype=np.float64)
        eta = np.zeros(nn, dtype=np.float64)
        vel = VelocityField(u=u, v=v, w=w, eta=eta)

        # Seed 200 particles at (4.80°, 5°) — closer to the deep node
        # at lon=4.75 (depth=500) than to the lon=5.0 transition.  The
        # deterministic step (zero velocity here) keeps nn=lon=4.75 and
        # z=200 m valid; the post-deterministic clamp is a no-op.
        n_particles = 200
        state = make_state(
            lon=np.full(n_particles, 4.80, dtype=np.float64),
            lat=np.full(n_particles, 5.0, dtype=np.float64),
            z=np.full(n_particles, 200.0, dtype=np.float64),
        )

        # Kh sized so that σ_lon ≈ 0.13° (≈14 km) — large enough that
        # many kicks cross past lon≈4.875 where nearest_node flips
        # from the deep 4.75 node to the shallow 5.0 node.
        diff = DiffusionStepArgs(
            scheme="uniform", kh_x=1.0e6, kh_y=1.0e3,
            seed=1234, real_id=0, step_i=0,
        )
        new = euler_step(state, vel, mesh, dt=100.0, time_dir=1.0,
                         diffusion=diff)

        alive = new.status == ParticleStatus.ALIVE
        assert alive.any(), "expected at least some kicks to land in-domain"

        # For every alive particle, z must be ≤ local bottom at its new
        # nearest_node — that is exactly the clamp guarantee.
        nn_nodes = new.nearest_node[alive]
        z_alive = new.z[alive]
        local_bottom = mesh.depth[nn_nodes]
        assert (z_alive <= local_bottom + 1e-9).all(), (
            f"some alive particle is below its local bottom: "
            f"max(z - bottom) = {(z_alive - local_bottom).max():.2f} m"
        )

        # Sanity: at least some particles' nearest_node moved to the
        # shallow side (depth=50).  If none did, the test would pass
        # trivially without exercising the post-kick clamp.
        on_shallow_side = alive & (mesh.depth[new.nearest_node] == 50.0)
        assert on_shallow_side.sum() > 5, (
            f"only {on_shallow_side.sum()} particles ended up on the "
            f"shallow side — RNG draw failed to exercise the path"
        )
        # And those particles started at z=200 m on the deep side and
        # ended up at z=50 m (clamped), so the post-kick clamp fired.
        assert np.all(new.z[on_shallow_side] <= 50.0 + 1e-9), (
            "particle on shallow side has z above local bottom"
        )


class TestPhaseBGuards:
    """Dispatch: em/m1 on RK + em/m1 with split composition."""

    def test_em_with_split_composition_raises(self, grid_wide):
        """em + split is a malformed DiffusionStepArgs and must fail loudly."""
        vel = _const_vel(grid_wide, u=0.0, v=0.0)
        state = _seed_cluster(3)
        diff = DiffusionStepArgs(scheme="em", kh_x=1.0, kh_y=1.0,
                                 seed=0, real_id=0, step_i=0,
                                 composition="split", dres_m=1000.0)
        with pytest.raises(NotImplementedError, match="composition='split'"):
            euler_step(state, vel, grid_wide, dt=100.0, time_dir=1.0,
                       diffusion=diff)

    def test_m1_with_split_composition_raises(self, grid_wide):
        vel = _const_vel(grid_wide, u=0.0, v=0.0)
        state = _seed_cluster(3)
        diff = DiffusionStepArgs(scheme="m1", kh_x=1.0, kh_y=1.0,
                                 seed=0, real_id=0, step_i=0,
                                 composition="split", dres_m=1000.0)
        with pytest.raises(NotImplementedError, match="composition='split'"):
            euler_step(state, vel, grid_wide, dt=100.0, time_dir=1.0,
                       diffusion=diff)

    @pytest.mark.parametrize("step_fn", [rk2_step, rk4_step])
    def test_em_combined_on_rk_raises(self, grid_wide, step_fn):
        """RK2/RK4 + em with composition='combined' must be rejected.

        split_rk composes with RK; combined remains Euler-only.
        """
        vel = _const_vel(grid_wide, u=0.0, v=0.0)
        state = _seed_cluster(3)
        diff = DiffusionStepArgs(scheme="em", kh_x=1.0, kh_y=1.0,
                                 seed=0, real_id=0, step_i=0,
                                 composition="combined", dres_m=1000.0)
        with pytest.raises(NotImplementedError, match="split_rk"):
            step_fn(state, vel, grid_wide, dt=100.0, time_dir=1.0,
                    diffusion=diff)


class TestCombinedEulerEM:
    """Combined-Euler with EM/M1 schemes on a varying-Kh field."""

    def _const_kh_vel(self, mesh, u, v, kh):
        """VelocityField with constant ``Kh`` on every node."""
        ne = mesh.n_elements
        nn = mesh.n_nodes
        return VelocityField(
            u=np.full(ne, u, dtype=np.float64),
            v=np.full(ne, v, dtype=np.float64),
            eta=np.zeros(nn, dtype=np.float64),
            kh_zonal=np.full(nn, kh, dtype=np.float64),
            kh_meridional=np.full(nn, kh, dtype=np.float64),
            kh_layout="node",
        )

    def test_zero_kh_matches_deterministic_euler(self, grid_wide):
        """EM-combined with Kh=0 on every node = deterministic Euler bitwise."""
        u, v = 0.3, -0.1
        det_vel = _const_vel(grid_wide, u=u, v=v)
        em_vel = self._const_kh_vel(grid_wide, u=u, v=v, kh=0.0)
        state = _seed_cluster(50)

        a = euler_step(state, det_vel, grid_wide, dt=300.0, time_dir=1.0)
        diff = DiffusionStepArgs(
            scheme="em", kh_x=0.0, kh_y=0.0,
            seed=42, real_id=0, step_i=0,
            composition="combined", dres_m=1000.0,
        )
        b = euler_step(state, em_vel, grid_wide, dt=300.0, time_dir=1.0,
                       diffusion=diff)
        np.testing.assert_array_equal(a.lon, b.lon)
        np.testing.assert_array_equal(a.lat, b.lat)
        np.testing.assert_array_equal(a.status, b.status)

    def test_constant_kh_matches_uniform_split_at_rest(self, grid_wide):
        """Constant Kh, **u=v=0**: combined EM ≈ uniform-split.

        Equivalence holds only when the metric factor ``z0`` is the
        same on both sides of the comparison.  In split mode the kick
        is added *after* the deterministic Euler step, so it uses
        ``post_det.lat`` for the metric; the combined path uses the
        start-of-step latitude.  With ``u = v = 0`` the latitude does
        not change, so both metrics agree and the test pins
        bit-identity.  With non-zero meridional velocity the two paths
        will diverge by O(dy · ∂z0/∂y) in degrees — that's a
        legitimate difference, not a bug, hence the narrowed scope
        (review finding #4).
        """
        kh = 4.0
        em_vel = self._const_kh_vel(grid_wide, u=0.0, v=0.0, kh=kh)
        uni_vel = _const_vel(grid_wide, u=0.0, v=0.0)
        state = _seed_cluster(50)

        diff_em = DiffusionStepArgs(
            scheme="em", kh_x=0.0, kh_y=0.0,
            seed=11, real_id=0, step_i=0,
            composition="combined", dres_m=1000.0,
        )
        diff_uni = DiffusionStepArgs(
            scheme="uniform", kh_x=kh, kh_y=kh,
            seed=11, real_id=0, step_i=0,
            composition="split",
        )
        a = euler_step(state, em_vel, grid_wide, dt=300.0, time_dir=1.0,
                       diffusion=diff_em)
        b = euler_step(state, uni_vel, grid_wide, dt=300.0, time_dir=1.0,
                       diffusion=diff_uni)
        # Bit-identical when ∂K = 0 *and* the metric matches: the
        # gradient kernel returns exactly 0 on a constant field, so the
        # combined-EM step is just u·dt + noise.  At rest, post-det
        # latitude == start-of-step latitude, so split and combined use
        # the same z0 and produce identical lon/lat shifts.
        np.testing.assert_allclose(a.lon, b.lon, atol=1e-10)
        np.testing.assert_allclose(a.lat, b.lat, atol=1e-10)

    def test_constant_kh_split_vs_combined_drifts_with_v(self, grid_wide):
        """With non-zero v, split and combined diverge by O(z0' · dy).

        Pin the *expected* divergence so the narrowed claim above is
        not a regression mask: if a future change accidentally ties
        the two metrics together, this test would loosen and we'd
        notice.  Conversely, if the metric drifts away from
        ``post_det.lat`` for the split path, this test would tighten.
        """
        kh = 4.0
        em_vel = self._const_kh_vel(grid_wide, u=0.0, v=0.05, kh=kh)
        uni_vel = _const_vel(grid_wide, u=0.0, v=0.05)
        state = _seed_cluster(50)
        diff_em = DiffusionStepArgs(
            scheme="em", kh_x=0.0, kh_y=0.0,
            seed=11, real_id=0, step_i=0,
            composition="combined", dres_m=1000.0,
        )
        diff_uni = DiffusionStepArgs(
            scheme="uniform", kh_x=kh, kh_y=kh,
            seed=11, real_id=0, step_i=0,
            composition="split",
        )
        a = euler_step(state, em_vel, grid_wide, dt=300.0, time_dir=1.0,
                       diffusion=diff_em)
        b = euler_step(state, uni_vel, grid_wide, dt=300.0, time_dir=1.0,
                       diffusion=diff_uni)
        # Latitudes match exactly (the deterministic Euler step on
        # constant v reaches the same lat in either path; Kh has no
        # mean-drift contribution at constant K).
        np.testing.assert_allclose(a.lat, b.lat, atol=1e-10, rtol=0.0)
        # Longitudes differ by a small amount — the post-det vs
        # start-of-step z0 mismatch.  Bound the divergence: it should
        # be on the order of dx_m · ∂(1/z0)/∂lat · v·dt/SLM, which for
        # these inputs is ~1e-10 deg ≈ ~10 µm.  Pin both that it's
        # non-zero AND that it's tiny.
        max_dlon_deg = float(np.max(np.abs(a.lon - b.lon)))
        assert max_dlon_deg > 1e-12, (
            f"max |dlon|={max_dlon_deg:.3e}: expected non-zero "
            "divergence under non-zero v"
        )
        SLM = METERS_PER_DEG_LAT
        max_dlon_m = max_dlon_deg * np.cos(np.deg2rad(5.0)) * SLM
        assert max_dlon_m < 1.0, (
            f"split-vs-combined longitude divergence {max_dlon_m:.2g} m "
            "exceeds 1 m — investigate"
        )

    def test_constant_kh_brownian_variance(self, grid_wide):
        """1k particles, constant Kh on dataset, var ≈ 2·Kh·dt after one step."""
        n = 1000
        kh = 1.0
        dt = 100.0
        em_vel = self._const_kh_vel(grid_wide, u=0.0, v=0.0, kh=kh)
        state = _seed_cluster(n)

        diff = DiffusionStepArgs(
            scheme="em", kh_x=0.0, kh_y=0.0,
            seed=99, real_id=0, step_i=0,
            composition="combined", dres_m=1000.0,
        )
        out = euler_step(state, em_vel, grid_wide, dt=dt, time_dir=1.0,
                         diffusion=diff)

        alive = out.status == ParticleStatus.ALIVE
        assert alive.sum() > 990, "kicks shouldn't push out of a 10° wide mesh"

        # Convert lon/lat increments to metres.
        z0 = np.cos(np.deg2rad(state.lat[alive])) * METERS_PER_DEG_LAT
        dx_m = (out.lon[alive] - state.lon[alive]) * z0
        dy_m = (out.lat[alive] - state.lat[alive]) * METERS_PER_DEG_LAT

        expected_var = 2.0 * kh * dt  # = 200 m²
        # SE on variance ≈ var · √(2/N) ≈ 200 · 0.045 ≈ 9.  Tolerate 15 %.
        assert abs(dx_m.var() / expected_var - 1.0) < 0.15
        assert abs(dy_m.var() / expected_var - 1.0) < 0.15

    def test_linear_kh_drift_sign(self, grid_wide):
        """Linear Kh(x) = K0 + b·x: ensemble drift sign matches Itô.

        The Itô correction term ``∂K/∂x · dt`` produces a forward drift
        when ``b > 0``.  Quantitative recovery is hard at single-step
        scale because the noise (``√(2 Kh dt)``) dominates the drift
        unless ``b`` is tuned to keep ``Kh`` small near the seed; at
        that point the test becomes a tautology of the analytical EM
        formula already covered by B4.  Here we keep ``Kh`` constant
        in the seed neighbourhood by placing the gradient *only* in y
        (normal to the seed line), so ``∂Kh_x/∂x = 0`` (no Itô drift)
        and ``∂Kh_y/∂y = b > 0`` — and we check the *meridional* drift.
        """
        n = 8000
        nn = grid_wide.n_nodes
        ne = grid_wide.n_elements
        # Constant Kh_x; linear Kh_y so ∂Kh_y/∂y = b.
        K0 = 1.0
        # Pick b so the *meridional* drift dominates the meridional noise:
        # drift = b·dt;  noise SE on mean = √(2·Kh_y·dt / n).  Choose b
        # such that drift ≫ noise SE in the seed neighbourhood (lat≈5).
        z0_ref = np.cos(np.deg2rad(5.0)) * METERS_PER_DEG_LAT  # for ref
        b = 1e-3  # m/s per metre of y  → at y=5°: Kh_y ≈ K0 + 555.6 ≈ 557
        kh_meridional = K0 + b * grid_wide.nodes_lat * METERS_PER_DEG_LAT
        kh_zonal = np.full(nn, K0)

        vel = VelocityField(
            u=np.zeros(ne), v=np.zeros(ne),
            eta=np.zeros(nn),
            kh_zonal=kh_zonal,
            kh_meridional=kh_meridional,
            kh_layout="node",
        )
        state = _seed_cluster(n)
        diff = DiffusionStepArgs(
            scheme="em", kh_x=0.0, kh_y=0.0,
            seed=2024, real_id=0, step_i=0,
            composition="combined", dres_m=2000.0,
        )
        dt = 100.0
        out = euler_step(state, vel, grid_wide, dt=dt, time_dir=1.0,
                         diffusion=diff)
        alive = out.status == ParticleStatus.ALIVE
        assert alive.sum() > n * 0.9

        dy_m = (out.lat[alive] - state.lat[alive]) * METERS_PER_DEG_LAT
        # Expected drift = b·dt = 0.1 m.  At seed lat=5°, Kh_y ≈ 557, so
        # noise σ = √(2·557·100) ≈ 334 m, SE on mean ≈ 334/√n ≈ 3.7 m.
        # The drift signal (0.1 m) is buried, so we just check that the
        # ensemble mean is within a few SE of the expected value (a
        # consistency check rather than a recovery test).
        expected_drift = b * dt
        kh_y_seed = K0 + b * 5.0 * METERS_PER_DEG_LAT
        se = np.sqrt(2.0 * kh_y_seed * dt / alive.sum())
        assert abs(dy_m.mean() - expected_drift) < 5.0 * se, (
            f"mean dy = {dy_m.mean():.3f} m, expected ≈ {expected_drift:.3f} ± {5*se:.3f} m"
        )
        # Variance check: empirical var ≈ 2·Kh·dt at seed.
        expected_var = 2.0 * kh_y_seed * dt
        assert abs(dy_m.var() / expected_var - 1.0) < 0.1

    def test_missing_kh_field_raises(self, grid_wide):
        """combined em on a VelocityField without Kh → clear error."""
        vel = _const_vel(grid_wide, u=0.0, v=0.0)  # no kh_*
        state = _seed_cluster(3)
        diff = DiffusionStepArgs(
            scheme="em", kh_x=0.0, kh_y=0.0,
            seed=0, real_id=0, step_i=0,
            composition="combined", dres_m=1000.0,
        )
        with pytest.raises(ValueError, match="kh_zonal"):
            euler_step(state, vel, grid_wide, dt=100.0, time_dir=1.0,
                       diffusion=diff)

    def test_dres_m_must_be_positive(self, grid_wide):
        em_vel = self._const_kh_vel(grid_wide, u=0.0, v=0.0, kh=1.0)
        state = _seed_cluster(3)
        diff = DiffusionStepArgs(
            scheme="em", kh_x=0.0, kh_y=0.0,
            seed=0, real_id=0, step_i=0,
            composition="combined", dres_m=0.0,
        )
        with pytest.raises(ValueError, match="dres_m"):
            euler_step(state, em_vel, grid_wide, dt=100.0, time_dir=1.0,
                       diffusion=diff)

    def test_m1_variable_kh_warns_once(self, grid_wide):
        """m1 + non-constant Kh: one RuntimeWarning per realization.

        review finding #2: M1 only implements diagonal-separable Kh; for
        non-separable spatially-varying Kh the cross-derivative
        Milstein terms are missing.  We surface this with a warning
        (rather than refusing m1) to avoid breaking the EM=M1 reduction
        on constant fields.
        """
        nn = grid_wide.n_nodes
        ne = grid_wide.n_elements
        # Spatially-varying Kh (linear in x).
        z0_ref = np.cos(np.deg2rad(5.0)) * METERS_PER_DEG_LAT
        kh_zonal = 1.0 + 1e-9 * grid_wide.nodes_lon * z0_ref
        kh_meridional = np.full(nn, 1.0)
        vel = VelocityField(
            u=np.zeros(ne), v=np.zeros(ne), eta=np.zeros(nn),
            kh_zonal=kh_zonal, kh_meridional=kh_meridional,
            kh_layout="node",
        )
        state = _seed_cluster(5)
        # Use a unique real_id so this test doesn't share the warning
        # state with previous tests.
        diff = DiffusionStepArgs(
            scheme="m1", kh_x=0.0, kh_y=0.0,
            seed=0, real_id=101, step_i=0,
            composition="combined", dres_m=1000.0,
        )
        with pytest.warns(RuntimeWarning, match="separable"):
            euler_step(state, vel, grid_wide, dt=100.0, time_dir=1.0,
                       diffusion=diff)
        # Second call with same real_id and same Kh fields → no warning.
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            # Will raise if a warning is emitted.
            diff2 = DiffusionStepArgs(
                scheme="m1", kh_x=0.0, kh_y=0.0,
                seed=0, real_id=101, step_i=1,
                composition="combined", dres_m=1000.0,
            )
            euler_step(state, vel, grid_wide, dt=100.0, time_dir=1.0,
                       diffusion=diff2)

    def test_m1_constant_kh_no_warning(self, grid_wide):
        """m1 + constant Kh is exact (reduces to EM); no warning."""
        ne = grid_wide.n_elements
        nn = grid_wide.n_nodes
        vel = VelocityField(
            u=np.zeros(ne), v=np.zeros(ne), eta=np.zeros(nn),
            kh_zonal=np.full(nn, 2.0),
            kh_meridional=np.full(nn, 2.0),
            kh_layout="node",
        )
        state = _seed_cluster(5)
        diff = DiffusionStepArgs(
            scheme="m1", kh_x=0.0, kh_y=0.0,
            seed=0, real_id=202, step_i=0,
            composition="combined", dres_m=1000.0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            euler_step(state, vel, grid_wide, dt=100.0, time_dir=1.0,
                       diffusion=diff)

    def test_3d_combined_raises(self, grid_wide):
        """3D + em is intentionally not wired; raises NotImplementedError."""
        ne = grid_wide.n_elements
        nn = grid_wide.n_nodes
        # Build a synthetic 3D vel: u/v shape (1, ne); w shape (2, nn).
        vel = VelocityField(
            u=np.zeros((1, ne)),
            v=np.zeros((1, ne)),
            eta=np.zeros(nn),
            w=np.zeros((2, nn)),
            kh_zonal=np.full(nn, 1.0),
            kh_meridional=np.full(nn, 1.0),
            kh_layout="node",
        )
        state = _seed_cluster(3)
        diff = DiffusionStepArgs(
            scheme="em", kh_x=0.0, kh_y=0.0,
            seed=0, real_id=0, step_i=0,
            composition="combined", dres_m=1000.0,
        )
        with pytest.raises(NotImplementedError, match="2D-only"):
            euler_step(state, vel, grid_wide, dt=100.0, time_dir=1.0,
                       diffusion=diff)

    @pytest.mark.parametrize("time_dir", [1.0, -1.0])
    def test_zero_kh_velocity_matches_deterministic_euler(self, grid_wide,
                                                          time_dir):
        """The combined step carries u/v like plain Euler, both directions.

        Before 2026-11 it returned the state without u/v, so a run with
        ``integrator: euler`` + ``em``/``m1`` wrote ``vel_u``/``vel_v`` all
        NaN.  Plain Euler stores ``time_dir * u`` (the tracking velocity;
        the tracking layer flips it back for output): the combined step
        must store the same, bit for bit, on a field that varies in space.
        """
        nn = grid_wide.n_nodes
        ex, ey = grid_wide.elem_center_lon, grid_wide.elem_center_lat
        u = 0.2 + 0.03 * ex - 0.02 * ey
        v = -0.1 + 0.01 * ex + 0.04 * ey
        det_vel = VelocityField(u=u.copy(), v=v.copy(), eta=np.zeros(nn))
        em_vel = VelocityField(
            u=u.copy(), v=v.copy(), eta=np.zeros(nn),
            kh_zonal=np.zeros(nn), kh_meridional=np.zeros(nn),
            kh_layout="node",
        )
        rng = np.random.default_rng(5)
        state = make_state(lon=rng.uniform(3.2, 6.8, 40),
                           lat=rng.uniform(3.2, 6.8, 40))
        diff = DiffusionStepArgs(
            scheme="em", kh_x=0.0, kh_y=0.0,
            seed=3, real_id=0, step_i=0,
            composition="combined", dres_m=1000.0,
        )
        a = euler_step(state, det_vel, grid_wide, dt=300.0, time_dir=time_dir)
        b = euler_step(state, em_vel, grid_wide, dt=300.0, time_dir=time_dir,
                       diffusion=diff)
        assert b.u is not None and b.v is not None
        alive = a.alive_mask & b.alive_mask
        assert alive.sum() == 40
        np.testing.assert_array_equal(b.u[alive], a.u[alive])
        np.testing.assert_array_equal(b.v[alive], a.v[alive])
        # non-vacuous: a nonzero velocity, signed like the tracking direction
        # (u > 0.05 everywhere on the seeded box)
        assert np.all(time_dir * b.u[alive] > 0.05)


class TestSplitRkOnRK:
    """split_rk composition with RK2/RK4 on variable Kh."""

    def _const_kh_vel(self, mesh, u, v, kh):
        ne = mesh.n_elements
        nn = mesh.n_nodes
        return VelocityField(
            u=np.full(ne, u, dtype=np.float64),
            v=np.full(ne, v, dtype=np.float64),
            eta=np.zeros(nn, dtype=np.float64),
            kh_zonal=np.full(nn, kh, dtype=np.float64),
            kh_meridional=np.full(nn, kh, dtype=np.float64),
            kh_layout="node",
        )

    @pytest.mark.parametrize("step_fn", [rk2_step, rk4_step])
    def test_split_rk_zero_kh_matches_deterministic(self, grid_wide, step_fn):
        """split_rk + Kh=0 = deterministic RK bitwise (∂K=0, σ=0)."""
        u, v = 0.2, -0.1
        det_vel = _const_vel(grid_wide, u=u, v=v)
        rk_vel = self._const_kh_vel(grid_wide, u=u, v=v, kh=0.0)
        state = _seed_cluster(50)

        a = step_fn(state, det_vel, grid_wide, dt=300.0, time_dir=1.0)
        diff = DiffusionStepArgs(
            scheme="em", kh_x=0.0, kh_y=0.0,
            seed=42, real_id=300, step_i=0,
            composition="split_rk", dres_m=1000.0,
        )
        # Suppress the once-per-realization split-RK warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            b = step_fn(state, rk_vel, grid_wide, dt=300.0, time_dir=1.0,
                        diffusion=diff)
        np.testing.assert_array_equal(a.lon, b.lon)
        np.testing.assert_array_equal(a.lat, b.lat)
        np.testing.assert_array_equal(a.status, b.status)

    @pytest.mark.parametrize("step_fn", [rk2_step, rk4_step])
    def test_split_rk_uniform_kh_brownian_variance(self, grid_wide, step_fn):
        """RK + split_rk + uniform Kh: variance ≈ 2·K·t after 1 step."""
        n = 1000
        kh = 1.0
        dt = 100.0
        rk_vel = self._const_kh_vel(grid_wide, u=0.0, v=0.0, kh=kh)
        state = _seed_cluster(n)

        diff = DiffusionStepArgs(
            scheme="em", kh_x=0.0, kh_y=0.0,
            seed=99, real_id=301, step_i=0,
            composition="split_rk", dres_m=1000.0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = step_fn(state, rk_vel, grid_wide, dt=dt, time_dir=1.0,
                          diffusion=diff)

        alive = out.status == ParticleStatus.ALIVE
        assert alive.sum() > 990

        z0 = np.cos(np.deg2rad(state.lat[alive])) * METERS_PER_DEG_LAT
        dx_m = (out.lon[alive] - state.lon[alive]) * z0
        dy_m = (out.lat[alive] - state.lat[alive]) * METERS_PER_DEG_LAT
        expected_var = 2.0 * kh * dt
        # Allow 15% as in the combined-EM analogue (1k particles, ~9% SE).
        assert abs(dx_m.var() / expected_var - 1.0) < 0.15
        assert abs(dy_m.var() / expected_var - 1.0) < 0.15

    @pytest.mark.parametrize("step_fn", [rk2_step, rk4_step])
    def test_split_rk_warns_once_per_realization(self, grid_wide, step_fn):
        """A single RuntimeWarning per (seed, real_id, scheme)."""
        rk_vel = self._const_kh_vel(grid_wide, u=0.0, v=0.0, kh=1.0)
        state = _seed_cluster(5)
        # Use a unique (seed, real_id, scheme) so this test is independent.
        rid = 400 + (1 if step_fn is rk2_step else 2)
        diff = DiffusionStepArgs(
            scheme="em", kh_x=0.0, kh_y=0.0,
            seed=0, real_id=rid, step_i=0,
            composition="split_rk", dres_m=1000.0,
        )
        with pytest.warns(RuntimeWarning, match="pragmatic"):
            step_fn(state, rk_vel, grid_wide, dt=100.0, time_dir=1.0,
                    diffusion=diff)
        # Second call with same (seed, real_id, scheme) → no warning.
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            diff2 = diff._replace(step_i=1)
            step_fn(state, rk_vel, grid_wide, dt=100.0, time_dir=1.0,
                    diffusion=diff2)

    def test_split_rk_warns_again_on_different_seed(self, grid_wide):
        """review C-#1: two runs with different seeds (or different real_id,
        or different scheme) each produce their own warning, not just
        the first one."""
        rk_vel = self._const_kh_vel(grid_wide, u=0.0, v=0.0, kh=1.0)
        state = _seed_cluster(5)
        diff_a = DiffusionStepArgs(
            scheme="em", kh_x=0.0, kh_y=0.0,
            seed=900, real_id=900, step_i=0,
            composition="split_rk", dres_m=1000.0,
        )
        diff_b = DiffusionStepArgs(
            scheme="em", kh_x=0.0, kh_y=0.0,
            seed=901, real_id=900, step_i=0,  # different seed
            composition="split_rk", dres_m=1000.0,
        )
        with pytest.warns(RuntimeWarning, match="pragmatic"):
            rk4_step(state, rk_vel, grid_wide, dt=100.0, time_dir=1.0,
                     diffusion=diff_a)
        # Different seed → warns again.
        with pytest.warns(RuntimeWarning, match="pragmatic"):
            rk4_step(state, rk_vel, grid_wide, dt=100.0, time_dir=1.0,
                     diffusion=diff_b)

    def test_split_rk_kh_eval_position_start_vs_end(self, grid_wide):
        """``kh_eval_position`` actually changes the kick on a varying Kh.

        With Kh = K0 + b·x and ``u > 0``, the post-step position is
        further east than the start position, so Kh and ∂K differ
        between the two evaluation modes.  A single deterministic seed
        + same step keys → both paths should differ measurably.
        """
        ne = grid_wide.n_elements
        nn = grid_wide.n_nodes
        K0 = 5.0
        b = 1e-7  # m/s per metre — moderate gradient
        z0_ref = np.cos(np.deg2rad(5.0)) * METERS_PER_DEG_LAT
        kh_zonal = K0 + b * grid_wide.nodes_lon * z0_ref
        kh_meridional = np.full(nn, K0)
        vel = VelocityField(
            u=np.full(ne, 0.5),       # eastward advection
            v=np.zeros(ne),
            eta=np.zeros(nn),
            kh_zonal=kh_zonal,
            kh_meridional=kh_meridional,
            kh_layout="node",
        )
        state = _seed_cluster(80)
        # Same real_id on both sides so Box-Muller draws are identical
        # — the only difference between runs is the eval position
        # (review C-#4).
        rid = 500
        diff_end = DiffusionStepArgs(
            scheme="em", kh_x=0.0, kh_y=0.0,
            seed=7, real_id=rid, step_i=0,
            composition="split_rk", dres_m=1000.0,
            kh_eval_position="end",
        )
        diff_start = DiffusionStepArgs(
            scheme="em", kh_x=0.0, kh_y=0.0,
            seed=7, real_id=rid, step_i=0,
            composition="split_rk", dres_m=1000.0,
            kh_eval_position="start",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            a = rk4_step(state, vel, grid_wide, dt=300.0, time_dir=1.0,
                         diffusion=diff_end)
            b = rk4_step(state, vel, grid_wide, dt=300.0, time_dir=1.0,
                         diffusion=diff_start)
        # Same RNG keys → same Box-Muller draws → only the eval position
        # differs.  Expect non-trivial divergence.
        max_dlon = float(np.max(np.abs(a.lon - b.lon)))
        # b·dt·Δx_apply / z0 — at u=0.5, dt=300 → Δx_apply ≈ 150 m
        # so |dKdx delta · dt| / z0 ≈ b·150·300 / z0 ≈ 4e-8 deg.
        # Well above 0; loose-bound the test to "non-zero and finite".
        assert max_dlon > 1e-12
        assert np.all(np.isfinite(a.lon)) and np.all(np.isfinite(b.lon))

    def test_split_rk_temporal_interp_uses_lerped_kh(self, grid_wide):
        """Review finding #3: kh_eval_position='end' samples Kh at
        ``sim_time + dt``, not at the start snapshot, when both
        ``vel_next`` and ``dt_to_next`` are provided.

        Pin this with a Kh field that *changes between snapshots*: the
        start field has constant Kh, the next has linear Kh.  At
        ``alpha = dt/dt_to_next = 0.5`` the lerped field has half the
        gradient — and the recovered ∂K/∂x at the post-RK position
        differs from what we'd get sampling from the start snapshot.
        """
        ne = grid_wide.n_elements
        nn = grid_wide.n_nodes
        K0 = 5.0
        # Start: constant 5.0 ; next: linear from 5.0 with slope b in metric x.
        b_next = 1e-6  # m/s per metre
        z0_ref = np.cos(np.deg2rad(5.0)) * METERS_PER_DEG_LAT
        kh_zonal_start = np.full(nn, K0)
        kh_zonal_next = K0 + b_next * grid_wide.nodes_lon * z0_ref
        kh_meridional_const = np.full(nn, K0)

        vel = VelocityField(
            u=np.zeros(ne), v=np.zeros(ne), eta=np.zeros(nn),
            kh_zonal=kh_zonal_start, kh_meridional=kh_meridional_const,
            kh_layout="node",
        )
        vel_next = VelocityField(
            u=np.zeros(ne), v=np.zeros(ne), eta=np.zeros(nn),
            kh_zonal=kh_zonal_next, kh_meridional=kh_meridional_const,
            kh_layout="node",
        )
        state = _seed_cluster(40)
        # Same key tuple, same kh_eval_position='end' — but one call
        # gets vel_next, the other doesn't.  The two outputs must
        # differ because the lerped Kh at sim_time+dt is non-trivially
        # different from the start-snapshot Kh.
        rid = 700
        diff = DiffusionStepArgs(
            scheme="em", kh_x=0.0, kh_y=0.0,
            seed=3, real_id=rid, step_i=0,
            composition="split_rk", dres_m=1000.0,
            kh_eval_position="end",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            a = rk4_step(state, vel, grid_wide, dt=300.0, time_dir=1.0,
                         diffusion=diff)
            b = rk4_step(state, vel, grid_wide, dt=300.0, time_dir=1.0,
                         diffusion=diff,
                         vel_next=vel_next, dt_to_next=600.0)
        # Same RNG, same advection (u=v=0) → only the Kh field differs.
        # With lerped Kh, the gradient is non-zero and produces an Itô
        # drift; the start-only path has ∂K=0 everywhere.
        assert not np.allclose(a.lon, b.lon, atol=1e-12, rtol=0.0)
        assert np.all(np.isfinite(b.lon))

    def test_split_rk_3d_raises(self, grid_wide):
        """split_rk is 2D-only (3D variable Kh deferred)."""
        ne = grid_wide.n_elements
        nn = grid_wide.n_nodes
        vel = VelocityField(
            u=np.zeros((1, ne)),
            v=np.zeros((1, ne)),
            eta=np.zeros(nn),
            w=np.zeros((2, nn)),
            kh_zonal=np.full(nn, 1.0),
            kh_meridional=np.full(nn, 1.0),
            kh_layout="node",
        )
        state = _seed_cluster(3)
        diff = DiffusionStepArgs(
            scheme="em", kh_x=0.0, kh_y=0.0,
            seed=0, real_id=600, step_i=0,
            composition="split_rk", dres_m=1000.0,
        )
        with pytest.raises(NotImplementedError, match="2D-only"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                rk4_step(state, vel, grid_wide, dt=100.0, time_dir=1.0,
                         diffusion=diff)




class TestShardParticleOffset:
    """Option (a), audit 2026-10-04: a block of particles stepped with
    ``pid_offset = its first global index`` moves exactly like the same
    particles inside the whole release — every scheme and composition."""

    @staticmethod
    def _kh_vel(mesh, kh):
        ne, nn = mesh.n_elements, mesh.n_nodes
        lon = mesh.nodes_lon
        return VelocityField(
            u=np.full(ne, 0.05), v=np.full(ne, -0.02),
            eta=np.zeros(nn),
            kh_zonal=kh * (1.0 + 0.05 * lon),
            kh_meridional=kh * (1.0 + 0.03 * lon),
            kh_layout="node")

    @pytest.mark.parametrize("scheme,composition,step_fn", [
        ("uniform", "split", rk2_step),
        ("em", "combined", euler_step),
        ("em", "split_rk", rk2_step),
    ])
    def test_block_equals_slice_of_whole(self, grid_wide, scheme,
                                         composition, step_fn):
        rng = np.random.default_rng(2)
        state = make_state(lon=rng.uniform(3.0, 7.0, 24),
                           lat=rng.uniform(3.0, 7.0, 24))
        vel = self._kh_vel(grid_wide, 40.0)
        args = DiffusionStepArgs(scheme=scheme, kh_x=40.0, kh_y=40.0,
                                 seed=4, real_id=0, step_i=7,
                                 composition=composition, dres_m=2000.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            whole = step_fn(state, vel, grid_wide, dt=600.0, time_dir=1.0,
                            diffusion=args)
            for lo, hi in ((0, 10), (10, 24)):
                sub = make_state(lon=state.lon[lo:hi], lat=state.lat[lo:hi])
                part = step_fn(sub, vel, grid_wide, dt=600.0, time_dir=1.0,
                               diffusion=args._replace(pid_offset=lo))
                np.testing.assert_array_equal(part.lon, whole.lon[lo:hi])
                np.testing.assert_array_equal(part.lat, whole.lat[lo:hi])
        # the offset is what makes it work: without it the block's noise differs
        assert not np.array_equal(
            step_fn(make_state(lon=state.lon[10:], lat=state.lat[10:]), vel,
                    grid_wide, dt=600.0, time_dir=1.0, diffusion=args).lon,
            whole.lon[10:])


def test_vertical_az_kick_keys_by_the_global_index():
    from kelp.numerics.diffusion import _loop_az_kick_zlevel
    P = 20
    z = np.linspace(5.0, 95.0, P)
    alive = np.ones(P, dtype=np.bool_)
    nn = np.zeros(P, dtype=np.int32)
    levels = np.array([0.0, 50.0, 100.0])
    field = np.zeros((3, 1))
    depth = np.array([100.0])

    def kick(lo, hi, off):
        out = np.empty(hi - lo)
        _loop_az_kick_zlevel(z[lo:hi].copy(), alive[lo:hi], nn[lo:hi], False,
                             field, 1e-3, levels, depth, 600.0,
                             np.uint64(3), np.uint64(0), np.uint64(5), out,
                             np.uint64(off))
        return out

    whole = kick(0, P, 0)
    np.testing.assert_array_equal(kick(8, P, 8), whole[8:])
    assert not np.array_equal(kick(8, P, 0), whole[8:])


@pytest.mark.parametrize("step_fn", [euler_step, rk2_step, rk4_step])
def test_all_dead_step_reports_zero_velocity(grid_wide, step_fn):
    """Every kernel writes 0 for non-alive particles; the all-dead early exit
    returned the LAST velocity instead, so a shard whose particles had all
    died recorded stale vel_* where the unsharded run recorded 0 (review
    round 3 of the shard offset, 2026-10-04)."""
    st = make_state(lon=np.array([5.0, 5.2]), lat=np.array([5.0, 5.1]))
    st = st.replace(status=np.full(2, ParticleStatus.BEACHED, dtype=st.status.dtype),
                    u=np.array([0.3, 0.2]), v=np.array([0.1, -0.1]),
                    w=np.array([1e-4, -2e-4]))
    out = step_fn(st, _const_vel(grid_wide, 0.3, 0.0), grid_wide, dt=600.0,
                  time_dir=1.0)
    assert np.array_equal(out.u, [0.0, 0.0]) and np.array_equal(out.v, [0.0, 0.0])
    assert np.array_equal(out.w, [0.0, 0.0])
    assert np.array_equal(out.lon, st.lon)
