"""Tests for ``FailureReason`` (TODO 2.6).

Covers:

* Enum + helper functions in ``kelp.core.particles``.
* ``ParticleState.failure_reason`` field is correctly threaded through
  ``replace`` / ``snapshot`` / ``make_state`` and stays ``None`` by
  default.
* The integrator paths (unfused AND, since 2026-08-16, fused) populate
  ``failure_reason`` with the
  correct codes when the user opts in by setting an int8 array on
  limitation).
* The trajectory writer optionally emits a ``failure_reason``
  NetCDF variable when present on the appended state.
"""
from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np
import pytest

from kelp.core.particles import (
    FailureReason, ParticleState, ParticleStatus,
    count_failure_reasons, failure_reason_to_string, make_state,
)


class TestEnumAndHelpers:
    def test_enum_values(self):
        # Pin values so existing pickled state / NetCDF flag_values
        # remain stable.
        assert int(FailureReason.NONE) == 0
        assert int(FailureReason.INTERPOLATION_FAILED) == 1
        assert int(FailureReason.OUT_OF_BOUNDS) == 2
        assert int(FailureReason.SURFACE_HIT) == 3
        assert int(FailureReason.BOTTOM_HIT) == 4

    def test_failure_reason_to_string(self):
        assert failure_reason_to_string(FailureReason.NONE) == "none"
        assert (
            failure_reason_to_string(FailureReason.INTERPOLATION_FAILED)
            == "interpolation_failed"
        )
        assert (
            failure_reason_to_string(FailureReason.OUT_OF_BOUNDS)
            == "out_of_bounds"
        )
        assert "unknown" in failure_reason_to_string(99)

    def test_count_none_returns_empty_dict(self):
        assert count_failure_reasons(None) == {}

    def test_count_failure_reasons(self):
        arr = np.array([
            FailureReason.NONE,
            FailureReason.INTERPOLATION_FAILED,
            FailureReason.OUT_OF_BOUNDS,
            FailureReason.OUT_OF_BOUNDS,
            FailureReason.NONE,
        ], dtype=np.int8)
        c = count_failure_reasons(arr)
        assert c["none"] == 2
        assert c["interpolation_failed"] == 1
        assert c["out_of_bounds"] == 2
        assert c["surface_hit"] == 0
        assert c["bottom_hit"] == 0
        assert c["total"] == 5


class TestParticleStateField:
    def test_failure_reason_default_none(self):
        st = make_state(np.array([1.0]), np.array([2.0]))
        assert st.failure_reason is None

    def test_failure_reason_survives_replace(self):
        fr = np.array([0, 1], dtype=np.int8)
        st = ParticleState(
            lon=np.array([1.0, 2.0]), lat=np.array([3.0, 4.0]),
            z=np.zeros(2), status=np.full(2, ParticleStatus.ALIVE),
            nearest_node=np.zeros(2, dtype=np.int32),
            failure_reason=fr,
        )
        st2 = st.replace(lon=np.array([10.0, 20.0]))
        # Replace doesn't touch failure_reason → same array
        assert st2.failure_reason is fr

    def test_failure_reason_in_snapshot(self):
        fr = np.array([1, 2], dtype=np.int8)
        st = ParticleState(
            lon=np.zeros(2), lat=np.zeros(2),
            z=np.zeros(2), status=np.zeros(2, dtype=np.int32),
            nearest_node=np.zeros(2, dtype=np.int32),
            failure_reason=fr,
        )
        snap = st.snapshot()
        assert "failure_reason" in snap
        np.testing.assert_array_equal(snap["failure_reason"], fr)
        # Snapshot is a deep copy
        assert snap["failure_reason"] is not fr


# ---------------------------------------------------------------------------
# Integrator-side: failure_reason populated on revert paths
# ---------------------------------------------------------------------------

# Reuse the synthetic mesh helper from another test file to avoid
# re-typing 30 lines.  These tests don't share state with other test
# modules.
from tests.test_integrators_diffusion import _build_grid_mesh, _const_vel
from kelp.numerics.integrators import euler_step
from kelp.numerics.diffusion import DiffusionStepArgs


@pytest.fixture
def small_mesh():
    """3×3 mesh: tiny so a single step easily exits the domain."""
    return _build_grid_mesh(np.array([4.0, 5.0, 6.0]),
                            np.array([4.0, 5.0, 6.0]))


class TestUnfusedFailureReasonPropagation:
    """Verify the unfused diffusion-kick path stamps the right code.

    The split-mode diffusion kick runs the Python-side
    ``_apply_split_diffusion_kick`` — exercised here; the fused kernels
    stamp too since 2026-08-16 (TestFusedFailureReason below).
    """

    def test_oob_kick_marks_out_of_bounds(self, small_mesh):
        """Huge-Kh kick on a tiny mesh → OUT_OF_BOUNDS, not NONE."""
        ne = small_mesh.n_elements
        nn = small_mesh.n_nodes
        from kelp.numerics.interpolate import VelocityField
        vel = VelocityField(
            u=np.zeros(ne), v=np.zeros(ne), eta=np.zeros(nn),
        )
        n = 50
        # Allocate the optional failure_reason array → opt-in tracking.
        state = ParticleState(
            lon=np.full(n, 5.0, dtype=np.float64),
            lat=np.full(n, 5.0, dtype=np.float64),
            z=np.zeros(n, dtype=np.float64),
            status=np.full(n, ParticleStatus.ALIVE, dtype=np.int32),
            nearest_node=np.full(n, -1, dtype=np.int32),
            failure_reason=np.zeros(n, dtype=np.int8),
        )
        # Kh huge enough that almost every kick lands outside the
        # 3×3 1° mesh.
        diff = DiffusionStepArgs(
            scheme="uniform", kh_x=1e8, kh_y=1e8,
            seed=42, real_id=0, step_i=0,
        )
        out = euler_step(state, vel, small_mesh, dt=300.0, time_dir=1.0,
                         diffusion=diff)
        assert out.failure_reason is not None
        # All particles should have died with OUT_OF_BOUNDS (the
        # uniform-Kh split kick has no interp dependency, only a
        # domain check on the kicked position).
        oob = out.failure_reason == FailureReason.OUT_OF_BOUNDS
        outside = out.status == ParticleStatus.OUTSIDE
        # Every dead particle has OUT_OF_BOUNDS, no INTERPOLATION_FAILED.
        assert np.all(oob[outside])
        no_interp_fail = np.all(
            out.failure_reason != FailureReason.INTERPOLATION_FAILED
        )
        assert no_interp_fail

    def test_no_failure_when_kh_zero(self, small_mesh):
        """No diffusion → no failures → failure_reason all NONE."""
        ne = small_mesh.n_elements
        nn = small_mesh.n_nodes
        from kelp.numerics.interpolate import VelocityField
        vel = VelocityField(
            u=np.zeros(ne), v=np.zeros(ne), eta=np.zeros(nn),
        )
        n = 10
        state = ParticleState(
            lon=np.full(n, 5.0, dtype=np.float64),
            lat=np.full(n, 5.0, dtype=np.float64),
            z=np.zeros(n, dtype=np.float64),
            status=np.full(n, ParticleStatus.ALIVE, dtype=np.int32),
            nearest_node=np.full(n, -1, dtype=np.int32),
            failure_reason=np.zeros(n, dtype=np.int8),
        )
        out = euler_step(state, vel, small_mesh, dt=300.0, time_dir=1.0)
        assert out.failure_reason is not None
        assert np.all(out.failure_reason == FailureReason.NONE)

    def test_field_stays_none_when_not_opted_in(self, small_mesh):
        """When state.failure_reason is None, the integrator doesn't
        magically allocate it — stays None."""
        ne = small_mesh.n_elements
        nn = small_mesh.n_nodes
        from kelp.numerics.interpolate import VelocityField
        vel = VelocityField(
            u=np.zeros(ne), v=np.zeros(ne), eta=np.zeros(nn),
        )
        state = make_state(np.array([5.0, 5.0]), np.array([5.0, 5.0]))
        assert state.failure_reason is None
        diff = DiffusionStepArgs(
            scheme="uniform", kh_x=1e8, kh_y=1e8,
            seed=42, real_id=0, step_i=0,
        )
        out = euler_step(state, vel, small_mesh, dt=300.0, time_dir=1.0,
                         diffusion=diff)
        assert out.failure_reason is None


# ---------------------------------------------------------------------------
# Writer round-trip
# ---------------------------------------------------------------------------


class TestWriterFailureReason:
    def test_writer_emits_failure_reason_when_present(self, tmp_path):
        """If a state appended to TrajectoryWriter has failure_reason
        set, a corresponding NetCDF variable is created and populated.
        """
        from kelp.writers import TrajectoryWriter

        n_particles = 4
        out_path = tmp_path / "traj.nc"
        with TrajectoryWriter(str(out_path), n_particles) as w:
            fr = np.array([
                FailureReason.NONE,
                FailureReason.INTERPOLATION_FAILED,
                FailureReason.OUT_OF_BOUNDS,
                FailureReason.NONE,
            ], dtype=np.int8)
            state = ParticleState(
                lon=np.zeros(n_particles), lat=np.zeros(n_particles),
                z=np.zeros(n_particles),
                status=np.full(n_particles, ParticleStatus.ALIVE,
                               dtype=np.int32),
                nearest_node=np.zeros(n_particles, dtype=np.int32),
                failure_reason=fr,
            )
            w.append(state, 0.0)

        with netCDF4.Dataset(out_path) as ds:
            assert "failure_reason" in ds.variables
            data = ds.variables["failure_reason"][:, 0]
            np.testing.assert_array_equal(np.asarray(data), fr)

    def test_writer_skips_failure_reason_when_absent(self, tmp_path):
        """Backward-compat: a state without failure_reason produces a
        file without the failure_reason variable."""
        from kelp.writers import TrajectoryWriter

        n_particles = 4
        out_path = tmp_path / "traj.nc"
        with TrajectoryWriter(str(out_path), n_particles) as w:
            state = ParticleState(
                lon=np.zeros(n_particles), lat=np.zeros(n_particles),
                z=np.zeros(n_particles),
                status=np.full(n_particles, ParticleStatus.ALIVE,
                               dtype=np.int32),
                nearest_node=np.zeros(n_particles, dtype=np.int32),
            )
            w.append(state, 0.0)

        with netCDF4.Dataset(out_path) as ds:
            assert "failure_reason" not in ds.variables


class TestFailureReasonRoundTrip:
    """review finding #2: round-trip ``failure_reason`` through NetCDF
    and Zarr writers and the ``load_trajectories`` / ``load_restart``
    readers."""

    def _make_state(self, n=4, fr=None):
        return ParticleState(
            lon=np.zeros(n), lat=np.zeros(n), z=np.zeros(n),
            status=np.full(n, ParticleStatus.ALIVE, dtype=np.int32),
            nearest_node=np.zeros(n, dtype=np.int32),
            failure_reason=fr,
        )

    @pytest.mark.parametrize("ext", [".nc", ".zarr"])
    def test_load_trajectories_includes_failure_reason(self, tmp_path, ext):
        """Both backends: appended failure_reason flows back through
        ``load_trajectories``."""
        from kelp.writers import TrajectoryWriter, load_trajectories

        out_path = tmp_path / f"traj{ext}"
        fr_t0 = np.array([
            FailureReason.NONE,
            FailureReason.INTERPOLATION_FAILED,
            FailureReason.OUT_OF_BOUNDS,
            FailureReason.NONE,
        ], dtype=np.int8)
        fr_t1 = np.array([
            FailureReason.NONE,
            FailureReason.INTERPOLATION_FAILED,
            FailureReason.OUT_OF_BOUNDS,
            FailureReason.OUT_OF_BOUNDS,
        ], dtype=np.int8)
        with TrajectoryWriter(str(out_path), 4) as w:
            w.append(self._make_state(fr=fr_t0), 0.0)
            w.append(self._make_state(fr=fr_t1), 100.0)

        data = load_trajectories(str(out_path))
        assert "failure_reason" in data, (
            f"failure_reason missing from load_trajectories output "
            f"({ext} backend)"
        )
        # Shape (P, T) — same as the other obs-axis variables.
        assert data["failure_reason"].shape == (4, 2)
        np.testing.assert_array_equal(data["failure_reason"][:, 0], fr_t0)
        np.testing.assert_array_equal(data["failure_reason"][:, 1], fr_t1)

    @pytest.mark.parametrize("ext", [".nc", ".zarr"])
    def test_load_trajectories_omits_when_writer_didnt_track(self, tmp_path, ext):
        """If the writer never saw a failure_reason, it shouldn't be in
        the loaded dict."""
        from kelp.writers import TrajectoryWriter, load_trajectories

        out_path = tmp_path / f"traj{ext}"
        with TrajectoryWriter(str(out_path), 4) as w:
            w.append(self._make_state(), 0.0)

        data = load_trajectories(str(out_path))
        assert "failure_reason" not in data

    @pytest.mark.parametrize("ext", [".nc", ".zarr"])
    def test_restart_preserves_failure_reason(self, tmp_path, ext):
        """``write_checkpoint`` + ``load_restart`` round-trips the
        failure_reason field on both backends."""
        from kelp.writers import TrajectoryWriter, load_restart

        out_path = tmp_path / f"traj{ext}"
        fr = np.array([
            FailureReason.NONE,
            FailureReason.OUT_OF_BOUNDS,
            FailureReason.INTERPOLATION_FAILED,
            FailureReason.NONE,
        ], dtype=np.int8)
        with TrajectoryWriter(str(out_path), 4) as w:
            state = self._make_state(fr=fr)
            w.append(state, 0.0)
            w.write_checkpoint(state, step_i=42, file_idx=7)

        restart = load_restart(str(out_path))
        assert restart["state"].failure_reason is not None
        np.testing.assert_array_equal(restart["state"].failure_reason, fr)
        assert restart["step_i"] == 42
        assert restart["file_idx"] == 7

    @pytest.mark.parametrize("ext", [".nc", ".zarr"])
    def test_restart_works_without_failure_reason(self, tmp_path, ext):
        """Backward-compat: checkpoint without failure_reason loads
        with failure_reason=None on both backends."""
        from kelp.writers import TrajectoryWriter, load_restart

        out_path = tmp_path / f"traj{ext}"
        with TrajectoryWriter(str(out_path), 4) as w:
            state = self._make_state(fr=None)
            w.append(state, 0.0)
            w.write_checkpoint(state, step_i=1, file_idx=0)

        restart = load_restart(str(out_path))
        assert restart["state"].failure_reason is None


# ---------------------------------------------------------------------------
# Fused kernels populate failure_reason (candidate 7 — closes the
# documented fused-2D gap, incl. tinterp/pchip twins and fused 3D Euler)
# ---------------------------------------------------------------------------

def _opted_state(n=8, lon=5.0, lat=5.0, z=0.0):
    return ParticleState(
        lon=np.full(n, lon, dtype=np.float64),
        lat=np.full(n, lat, dtype=np.float64),
        z=np.full(n, z, dtype=np.float64),
        status=np.full(n, ParticleStatus.ALIVE, dtype=np.int32),
        nearest_node=np.full(n, -1, dtype=np.int32),
        failure_reason=np.zeros(n, dtype=np.int8),
    )


class TestFusedFailureReason:
    """Every fused kernel must stamp INTERPOLATION_FAILED on a failed
    stage interpolation and OUT_OF_BOUNDS on a failed final containment
    — the same semantics as the unfused paths.  Opt-in only: a None
    failure_reason stays None bit-identically."""

    def _strong_vel(self, mesh, u=500.0):
        from kelp.numerics.interpolate import VelocityField
        return VelocityField(u=np.full(mesh.n_elements, u),
                             v=np.zeros(mesh.n_elements),
                             eta=np.zeros(mesh.n_nodes))

    @pytest.mark.parametrize("name", ["euler", "rk2", "rk4"])
    def test_fused_2d_out_of_bounds(self, small_mesh, name):
        from kelp.numerics.integrators import get_integrator
        vel = self._strong_vel(small_mesh)      # exits the 3x3 mesh in 1 step
        out = get_integrator(name)(_opted_state(), vel, small_mesh,
                                   3600.0, 1.0)
        assert np.all(out.status == ParticleStatus.OUTSIDE)
        assert np.all(out.failure_reason == FailureReason.OUT_OF_BOUNDS)

    @pytest.mark.parametrize("name", ["rk2", "rk4"])
    def test_fused_2d_tinterp_out_of_bounds(self, small_mesh, name):
        # euler deliberately absent: a plain field + vel_next routes
        # euler to its PLAIN kernel (vel_next ignored); the pair kernel
        # is pinned by test_euler_pair_wrapper_stamps instead.
        from kelp.numerics.integrators import get_integrator
        vel = self._strong_vel(small_mesh)
        vel_b = self._strong_vel(small_mesh, u=600.0)
        out = get_integrator(name)(_opted_state(), vel, small_mesh,
                                   3600.0, 1.0,
                                   vel_next=vel_b, dt_to_next=3600.0)
        assert np.all(out.failure_reason == FailureReason.OUT_OF_BOUNDS)

    @pytest.mark.parametrize("name", ["rk2", "rk4"])
    def test_fused_2d_pchip_out_of_bounds(self, small_mesh, name):
        from kelp.numerics.integrators import get_integrator
        snaps = [self._strong_vel(small_mesh, u=500.0 + 10 * k)
                 for k in range(4)]
        times = [0.0, 3600.0, 7200.0, 10800.0]
        out = get_integrator(name)(_opted_state(), snaps[1], small_mesh,
                                   1800.0, 1.0,
                                   recon=(snaps, times, 3600.0, 1.0))
        assert np.all(out.failure_reason == FailureReason.OUT_OF_BOUNDS)

    def test_fused_2d_interp_failed(self, small_mesh):
        """Doctored mesh: the seed's nearest node has an EMPTY element
        ring, so the stage-1 interpolation fails -> code 1."""
        from kelp.numerics.integrators import get_integrator
        import copy
        mesh = small_mesh
        mesh.node_to_elems_count = mesh.node_to_elems_count.copy()
        from kelp.core.mesh import find_nearest_nodes
        nn = int(find_nearest_nodes(mesh, np.array([5.0]),
                                    np.array([5.0]))[0])
        mesh.node_to_elems_count[nn] = 0
        vel = self._strong_vel(mesh, u=0.1)
        out = get_integrator("euler")(_opted_state(), vel, mesh,
                                      300.0, 1.0)
        assert np.all(out.status == ParticleStatus.OUTSIDE)
        assert np.all(out.failure_reason
                      == FailureReason.INTERPOLATION_FAILED)

    def test_fused_3d_euler_out_of_bounds(self, small_mesh):
        from kelp.numerics.integrators import get_integrator
        from kelp.numerics.interpolate import VelocityField
        mesh = small_mesh
        mesh.sigma = np.array([0.0, 10.0, 50.0])
        K = 3
        vel = VelocityField(
            u=np.full((K - 1, mesh.n_elements), 500.0),
            v=np.zeros((K - 1, mesh.n_elements)),
            eta=np.zeros(mesh.n_nodes),
            w=np.zeros((K, mesh.n_nodes)))
        st = _opted_state(z=20.0)
        out = get_integrator("euler")(st, vel, mesh, 3600.0, 1.0,
                                      w_interp="leastsq")
        assert np.all(out.failure_reason == FailureReason.OUT_OF_BOUNDS)

    def test_opt_out_stays_none(self, small_mesh):
        from kelp.numerics.integrators import get_integrator
        vel = self._strong_vel(small_mesh)
        st = _opted_state()
        st2 = ParticleState(lon=st.lon.copy(), lat=st.lat.copy(),
                            z=st.z.copy(), status=st.status.copy(),
                            nearest_node=st.nearest_node.copy())
        out = get_integrator("rk4")(st2, vel, small_mesh, 3600.0, 1.0)
        assert out.failure_reason is None

    def test_sticky_previous_code_kept(self, small_mesh):
        from kelp.numerics.integrators import get_integrator
        vel = self._strong_vel(small_mesh)
        st = _opted_state()
        fr = st.failure_reason.copy()
        status = st.status.copy()
        fr[0] = FailureReason.INTERPOLATION_FAILED   # died earlier
        status[0] = ParticleStatus.OUTSIDE
        st = st.replace(failure_reason=fr, status=status)
        out = get_integrator("euler")(st, vel, small_mesh, 3600.0, 1.0)
        assert out.failure_reason[0] == FailureReason.INTERPOLATION_FAILED
        assert np.all(out.failure_reason[1:] == FailureReason.OUT_OF_BOUNDS)

    def test_fused_codes_pinned_to_enum(self):
        from kelp.numerics.integrators import _FR_INTERP, _FR_OOB
        assert _FR_INTERP == FailureReason.INTERPOLATION_FAILED
        assert _FR_OOB == FailureReason.OUT_OF_BOUNDS

    def test_euler_pair_wrapper_stamps(self, small_mesh):
        """LazyLerpField vel routes Euler through _run_fused_euler_2d_pair
        (plain Euler ignores vel_next) — that wrapper must stamp too."""
        from kelp.core.io import LazyLerpField
        from kelp.numerics.integrators import get_integrator
        a = self._strong_vel(small_mesh, u=500.0)
        b = self._strong_vel(small_mesh, u=600.0)
        lazy = LazyLerpField(a, b, 0.25)
        out = get_integrator("euler")(_opted_state(), lazy, small_mesh,
                                      3600.0, 1.0,
                                      vel_next=b, dt_to_next=3600.0)
        assert np.all(out.failure_reason == FailureReason.OUT_OF_BOUNDS)

    @pytest.mark.parametrize("name", ["rk2", "rk4"])
    def test_later_stage_interp_failure_stamped(self, small_mesh, name):
        """Stage-2+ interpolation failure (ok2..ok4 guards): a strong jet
        pushes the midpoint trial next to a doctored empty-ring node
        while the stage-1 stencil is healthy."""
        from kelp.core.mesh import find_nearest_nodes
        from kelp.numerics.integrators import get_integrator
        mesh = small_mesh
        mesh.node_to_elems_count = mesh.node_to_elems_count.copy()
        # doctor the node the midpoint will anchor to: seed at (4.3, 5),
        # u=30 m/s, dt so the half-step lands near (5, 5)
        target = int(find_nearest_nodes(mesh, np.array([5.0]),
                                        np.array([5.0]))[0])
        mesh.node_to_elems_count[target] = 0
        st = _opted_state(n=4, lon=4.3, lat=5.0)
        dt = 0.7 * 111120.0 * 2.0 / 30.0     # half-step ~ 0.7 deg east
        vel = self._strong_vel(mesh, u=30.0)
        out = get_integrator(name)(st, vel, mesh, dt, 1.0)
        assert np.all(out.status == ParticleStatus.OUTSIDE)
        assert np.all(out.failure_reason
                      == FailureReason.INTERPOLATION_FAILED)

    def test_beaching_remap_clears_code(self, small_mesh):
        """A remapped (OUTSIDE->BEACHED) particle is not dead: the code
        stamped by the integrator must be cleared so failure_reason
        keeps cause-of-death semantics (review round 1)."""
        from kelp.tracking import _apply_beaching_hook
        from kelp.numerics.beaching import BeachRunConfig
        # minimal hand-built scenario: previous ALIVE, now OUTSIDE with a
        # code; anchor context forces the remap via a shallow column
        pytest.importorskip("numba")
        st = _opted_state(n=2)
        prev_status = st.status.copy()
        status = st.status.copy()
        fr = st.failure_reason.copy()
        status[0] = ParticleStatus.OUTSIDE
        fr[0] = FailureReason.OUT_OF_BOUNDS
        st = st.replace(status=status, failure_reason=fr)
        from kelp.numerics.interpolate import VelocityField
        from kelp.core.config import BeachingConfig
        vel = VelocityField(u=np.zeros(small_mesh.n_elements),
                            v=np.zeros(small_mesh.n_elements),
                            eta=np.zeros(small_mesh.n_nodes))
        bcfg = BeachingConfig(enabled=True, criteria=["on_failure"])
        run_cfg = BeachRunConfig(
            cfg=bcfg, crit=frozenset({"on_failure"}), need_topo=False,
            elem_coastal=None, seed=0,
            zone=1.0e9)                    # anchor always in-zone -> remap
        out, _ = _apply_beaching_hook(st, prev_status, small_mesh, vel,
                                      run_cfg, None, 0, 0, 100.0)
        assert out.status[0] == ParticleStatus.BEACHED
        assert out.failure_reason[0] == FailureReason.NONE


class TestFusedInterpLegsAndConfigKnob:
    """Round-1 agent findings: (a) the _FR_INTERP legs of the tinterp /
    pchip / pair / 3D kernels had no suite pin (only their OOB legs);
    (b) config-driven runs could never opt in at all."""

    def _doctored(self, small_mesh):
        from kelp.core.mesh import find_nearest_nodes
        mesh = small_mesh
        mesh.node_to_elems_count = mesh.node_to_elems_count.copy()
        nn = int(find_nearest_nodes(mesh, np.array([5.0]),
                                    np.array([5.0]))[0])
        mesh.node_to_elems_count[nn] = 0     # empty ring at the seed node
        return mesh

    def _vel(self, mesh, u=0.1):
        from kelp.numerics.interpolate import VelocityField
        return VelocityField(u=np.full(mesh.n_elements, u),
                             v=np.zeros(mesh.n_elements),
                             eta=np.zeros(mesh.n_nodes))

    @pytest.mark.parametrize("name", ["euler", "rk2", "rk4"])
    def test_tinterp_kernels_interp_failed(self, small_mesh, name):
        from kelp.core.io import LazyLerpField
        from kelp.numerics.integrators import get_integrator
        mesh = self._doctored(small_mesh)
        a, b = self._vel(mesh), self._vel(mesh, u=0.2)
        # euler reaches its tinterp (pair) kernel only via LazyLerpField;
        # rk2/rk4 route to their tinterp kernels via vel_next
        vel = LazyLerpField(a, b, 0.25) if name == "euler" else a
        out = get_integrator(name)(_opted_state(), vel, mesh, 300.0, 1.0,
                                   vel_next=b, dt_to_next=300.0)
        assert np.all(out.failure_reason
                      == FailureReason.INTERPOLATION_FAILED)

    @pytest.mark.parametrize("name", ["rk2", "rk4"])
    def test_pchip_kernels_interp_failed(self, small_mesh, name):
        from kelp.numerics.integrators import get_integrator
        mesh = self._doctored(small_mesh)
        snaps = [self._vel(mesh, u=0.1 + 0.01 * k) for k in range(4)]
        times = [0.0, 300.0, 600.0, 900.0]
        out = get_integrator(name)(_opted_state(), snaps[1], mesh,
                                   150.0, 1.0,
                                   recon=(snaps, times, 300.0, 1.0))
        assert np.all(out.failure_reason
                      == FailureReason.INTERPOLATION_FAILED)

    def test_fused_3d_interp_failed(self, small_mesh):
        from kelp.numerics.interpolate import VelocityField
        from kelp.numerics.integrators import get_integrator
        mesh = self._doctored(small_mesh)
        mesh.sigma = np.array([0.0, 10.0, 50.0])
        K = 3
        vel = VelocityField(u=np.full((K - 1, mesh.n_elements), 0.1),
                            v=np.zeros((K - 1, mesh.n_elements)),
                            eta=np.zeros(mesh.n_nodes),
                            w=np.zeros((K, mesh.n_nodes)))
        out = get_integrator("euler")(_opted_state(z=20.0), vel, mesh,
                                      300.0, 1.0, w_interp="leastsq")
        assert np.all(out.failure_reason
                      == FailureReason.INTERPOLATION_FAILED)

    def test_config_run_emits_failure_reason(self, tmp_path):
        """mode.track_failure_reason: a config-driven fused run writes
        the failure_reason output variable with stamped codes (config
        runs previously could not opt in at all)."""
        import netCDF4
        from tests.test_tracking import _make_fesomc_3x3_with_time
        from kelp.core.config import (AppConfig, DataConfig,
                                          MeshConfig, ModeConfig,
                                          NumericsConfig, TimeConfig)
        from kelp.tracking import run_tracking
        nc = _make_fesomc_3x3_with_time(tmp_path, n_time=4,
                                        dt_seconds=3600.0,
                                        u_val=100.0, v_val=0.0)
        seeds = tmp_path / "p.dat"
        seeds.write_text("5.0 5.0\n5.5 5.0\n")
        out = tmp_path / "fr.nc"
        cfg = AppConfig(
            mode=ModeConfig(Fdrift2D=True, Fdrift3D=False,
                            track_failure_reason=True),
            numerics=NumericsConfig(dt=3600.0, integrator="euler",
                                    output_freq=1),
            mesh=MeshConfig(type="fesomc", meshpath=str(nc)),
            data=DataConfig(ncinput=str(nc), inicloud=str(seeds),
                            output_path=str(out),
                            time_averaging="instant"),
            time=TimeConfig(start_time_idx=0, n_steps=3),
        )
        run_tracking(cfg)
        with netCDF4.Dataset(out) as ds:
            assert "failure_reason" in ds.variables
            fr = np.array(ds.variables["failure_reason"][:])
        # u=100 m/s exits the small mesh quickly -> OOB codes appear
        assert (fr == FailureReason.OUT_OF_BOUNDS).any()
