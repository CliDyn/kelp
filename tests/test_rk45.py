"""Tests for the RK45 adaptive integrator (TODO 2.3).

Dormand-Prince 5(4) with per-particle adaptive substepping inside the
fixed tracking step ``dt``.  Design notes: docs/time_integration.md section 7.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kelp.core.geometry import METERS_PER_DEG_LAT  # noqa: E402
from kelp.core.io import VelocityField  # noqa: E402
from kelp.core.particles import (  # noqa: E402
    FailureReason, ParticleStatus, make_state)

from test_integrators_time_aware import (  # noqa: E402
    _make_mesh, _make_vel, _analytical, LON0, LAT0, U0, OMEGA)


def _state0(n=1):
    return make_state(np.full(n, LON0), np.full(n, LAT0))


# ---------------------------------------------------------------------------
# Registry + config
# ---------------------------------------------------------------------------

class TestRegistryAndConfig:
    def test_registry_has_rk45(self):
        from kelp.numerics.integrators import INTEGRATORS, get_integrator
        assert "rk45" in INTEGRATORS
        assert callable(get_integrator("rk45"))

    def test_config_defaults(self):
        from kelp.core.config import NumericsConfig
        cfg = NumericsConfig()
        assert cfg.rk45_tol_m == 1.0
        assert cfg.rk45_h_min is None

    def test_config_validation(self):
        from kelp.core.config import NumericsConfig
        with pytest.raises(ValueError, match="rk45_tol_m"):
            NumericsConfig(rk45_tol_m=0.0)
        with pytest.raises(ValueError, match="rk45_h_min"):
            NumericsConfig(rk45_h_min=-1.0)


# ---------------------------------------------------------------------------
# v1 fail-fasts
# ---------------------------------------------------------------------------

class TestFailFasts:
    def _mesh_vel(self):
        mesh = _make_mesh()
        return mesh, _make_vel(mesh, 0.0)

    def test_3d_raises(self):
        from kelp.numerics.integrators import rk45_step
        mesh, vel = self._mesh_vel()
        K = 3
        vel3d = VelocityField(
            u=np.zeros((K - 1, mesh.n_elements)),
            v=np.zeros((K - 1, mesh.n_elements)),
            eta=np.zeros(mesh.n_nodes),
            w=np.zeros((K, mesh.n_nodes)))
        with pytest.raises(NotImplementedError, match="3-D|3D"):
            rk45_step(_state0(), vel3d, mesh, 600.0, 1.0)

    def test_node_layout_raises(self):
        from kelp.numerics.integrators import rk45_step
        mesh, _ = self._mesh_vel()
        veln = VelocityField(u=np.zeros(mesh.n_nodes),
                             v=np.zeros(mesh.n_nodes),
                             eta=np.zeros(mesh.n_nodes),
                             uv_layout="node")
        with pytest.raises((ValueError, NotImplementedError), match="node"):
            rk45_step(_state0(), veln, mesh, 600.0, 1.0)

    @pytest.mark.parametrize("interp", ["barycentric", "edge_leastsq"])
    def test_non_leastsq_uv_interp_raises(self, interp):
        from kelp.numerics.integrators import rk45_step
        mesh, vel = self._mesh_vel()
        with pytest.raises(NotImplementedError, match="leastsq"):
            rk45_step(_state0(), vel, mesh, 600.0, 1.0, uv_interp=interp)

    def test_recon_bundle_raises(self):
        from kelp.numerics.integrators import rk45_step
        mesh, vel = self._mesh_vel()
        snaps = [_make_vel(mesh, t) for t in (0.0, 1.0, 2.0, 3.0)]
        with pytest.raises(NotImplementedError, match="pchip|recon"):
            rk45_step(_state0(), vel, mesh, 0.5, 1.0,
                      recon=(snaps, [0.0, 1.0, 2.0, 3.0], 1.0, 1.0))

    def test_em_diffusion_raises(self):
        from kelp.numerics.integrators import rk45_step
        from kelp.numerics.diffusion import DiffusionStepArgs
        mesh, vel = self._mesh_vel()
        args = DiffusionStepArgs(scheme="em", kh_x=0.0, kh_y=0.0,
                                 seed=1, real_id=0, step_i=0,
                                 composition="split_rk")
        with pytest.raises((ValueError, NotImplementedError), match="em|m1"):
            rk45_step(_state0(), vel, mesh, 600.0, 1.0, diffusion=args)

    def test_carried_elems_raise(self):
        """elem_search: walk carries elems the rk45 path neither uses
        nor updates — reject rather than silently break the contract."""
        from kelp.numerics.integrators import rk45_step
        mesh, vel = self._mesh_vel()
        st = _state0().replace(elem=np.zeros(1, dtype=np.int32))
        with pytest.raises((ValueError, NotImplementedError), match="walk"):
            rk45_step(st, vel, mesh, 600.0, 1.0)


# ---------------------------------------------------------------------------
# Accuracy: constant field (exactness), rotating field (analytic circle)
# ---------------------------------------------------------------------------

class TestAccuracy:
    def test_constant_field_exact_single_substep(self):
        """A constant field on the equator: zero error estimate, exactly
        one substep, displacement equal to dt*U0/SLM."""
        from kelp.numerics.integrators import rk45_step
        mesh = _make_mesh()
        vel = _make_vel(mesh, 0.0)      # U = U0, V = 0
        dt = 600.0
        diag = {}
        out = rk45_step(_state0(), vel, mesh, dt, 1.0, rk45_diag=diag)
        expected = LON0 + dt * U0 / METERS_PER_DEG_LAT
        np.testing.assert_allclose(out.lon[0], expected, rtol=1e-13)
        np.testing.assert_allclose(out.lat[0], LAT0, atol=1e-13)
        assert out.status[0] == ParticleStatus.ALIVE
        assert diag["nsub"][0] == 1
        assert diag["hmin_clamped"] == 0
        # last evaluated velocity recorded for the vel_* diagnostics
        np.testing.assert_allclose(out.u[0], U0, rtol=1e-12)

    def _integrate(self, tol, n_steps=10, dt=600.0, time_dir=1.0,
                   collect_nsub=False):
        """March the rotating uniform field with per-step snapshots
        (temporal interpolation), starting at t=0."""
        from kelp.numerics.integrators import rk45_step
        mesh = _make_mesh()
        st = _state0()
        nsub_total = 0
        for k in range(n_steps):
            t = k * dt
            vel = _make_vel(mesh, t)
            vel_next = _make_vel(mesh, t + dt)
            diag = {} if collect_nsub else None
            st = rk45_step(st, vel, mesh, dt, time_dir,
                           vel_next=vel_next, dt_to_next=dt,
                           rk45_tol_m=tol, rk45_diag=diag)
            if collect_nsub:
                nsub_total += int(diag["nsub"][0])
        return st, nsub_total

    def test_linear_in_time_field_exact_any_tol(self):
        """A spatially-uniform field is PIECEWISE-LINEAR in time between
        snapshots — DP54 integrates that exactly, so every tolerance
        yields a bit-identical single-substep result.  (This is a
        deliberate exactness pin, NOT a convergence test: the reviewer
        proved the earlier 'convergence' variant compared equal zeros.
        Real tolerance-accuracy coupling is pinned on the vortex below.)
        """
        ref, nsub_ref = self._integrate(1e-6, collect_nsub=True)
        for tol in (100.0, 1.0, 1e-3):
            st, nsub = self._integrate(tol, collect_nsub=True)
            assert st.lon[0] == ref.lon[0]
            assert st.lat[0] == ref.lat[0]
            assert nsub == nsub_ref == 10          # one substep per step

    def test_vortex_accuracy_improves_with_tol_and_beats_rk4(self):
        """Spatially-varying field: closure against the self-converged
        reference improves monotonically with tolerance, and tight-tol
        rk45 beats fixed-step rk4 at the same record dt (spec item 3)."""
        from kelp.numerics.integrators import rk45_step, rk4_step
        mesh, vel, c = self._vortex()
        period = 2.0 * np.pi / 3.6e-5
        n = 8
        dtstep = period / n                    # omega*dt ~ 0.785 rad/step

        def run(fn, **kw):
            st = make_state(np.array([c + 0.5]), np.array([c]))
            for _ in range(n):
                st = fn(st, vel, mesh, dtstep, 1.0, **kw)
            return st

        ref = run(rk45_step, rk45_tol_m=1e-6)
        cosf = np.cos(np.deg2rad(c))

        def err(st):
            return np.hypot(
                (st.lon[0] - ref.lon[0]) * METERS_PER_DEG_LAT * cosf,
                (st.lat[0] - ref.lat[0]) * METERS_PER_DEG_LAT)

        e_loose = err(run(rk45_step, rk45_tol_m=100.0))
        e_mid = err(run(rk45_step, rk45_tol_m=1.0))
        e_tight = err(run(rk45_step, rk45_tol_m=1e-3))
        assert e_loose > e_mid > e_tight
        e_rk4 = err(run(rk4_step))
        assert e_tight < e_rk4                 # adaptivity buys accuracy

    def test_rotating_field_vs_analytic(self):
        """With snapshots at step ends, the piecewise-linear-in-time
        reconstruction is close to the true field at this cadence; a
        tight-tolerance rk45 must land near the closed-form circle."""
        n, dt = 10, 600.0
        st, _ = self._integrate(1e-4, n_steps=n, dt=dt)
        x_m, y_m = _analytical(n * dt)
        lon_exp = LON0 + x_m / METERS_PER_DEG_LAT
        lat_exp = LAT0 + y_m / METERS_PER_DEG_LAT
        err_m = np.hypot((st.lon[0] - lon_exp) * METERS_PER_DEG_LAT,
                         (st.lat[0] - lat_exp) * METERS_PER_DEG_LAT)
        # reconstruction cap dominates; just require physically tiny
        assert err_m < 5.0

    @staticmethod
    def _vortex(span=1.0, nx=40, omega=3.6e-5, center=5.0):
        """Solid-body vortex on a bigger mesh: genuinely nonlinear along
        the path (the uniform rotating FIELD is linear in time between
        snapshots and is integrated exactly by any RK >= 2, so it cannot
        exercise the substep controller)."""
        from tests.test_3d_analytical import _make_mesh as _mk
        mesh = _mk(center, center, span=span, nx=nx, ny=nx)
        x_m = (np.asarray(mesh.elem_center_lon) - center) \
            * METERS_PER_DEG_LAT * np.cos(np.deg2rad(center))
        y_m = (np.asarray(mesh.elem_center_lat) - center) \
            * METERS_PER_DEG_LAT
        u = -omega * y_m
        v = omega * x_m
        vel = VelocityField(u=u, v=v, eta=np.zeros(mesh.n_nodes), w=None)
        return mesh, vel, center

    def test_adaptivity_more_substeps_at_tighter_tol(self):
        from kelp.numerics.integrators import rk45_step
        mesh, vel, c = self._vortex()
        st = make_state(np.array([c + 0.5]), np.array([c]))
        d_loose, d_tight = {}, {}
        rk45_step(st, vel, mesh, 21600.0, 1.0, rk45_tol_m=10.0,
                  rk45_diag=d_loose)
        rk45_step(st, vel, mesh, 21600.0, 1.0, rk45_tol_m=1e-3,
                  rk45_diag=d_tight)
        assert d_tight["nsub"][0] > d_loose["nsub"][0]

    def test_backward_round_trip(self):
        """Forward 6 steps then backward 6 steps returns to the seed."""
        from kelp.numerics.integrators import rk45_step
        mesh = _make_mesh()
        dt, n = 600.0, 6
        st = _state0()
        for k in range(n):
            t = k * dt
            st = rk45_step(st, _make_vel(mesh, t), mesh, dt, 1.0,
                           vel_next=_make_vel(mesh, t + dt), dt_to_next=dt,
                           rk45_tol_m=1e-4)
        for k in range(n - 1, -1, -1):
            t = (k + 1) * dt
            st = rk45_step(st, _make_vel(mesh, t), mesh, dt, -1.0,
                           vel_next=_make_vel(mesh, t - dt), dt_to_next=dt,
                           rk45_tol_m=1e-4)
        err_m = np.hypot((st.lon[0] - LON0) * METERS_PER_DEG_LAT,
                         (st.lat[0] - LAT0) * METERS_PER_DEG_LAT)
        assert err_m < 0.1

    def test_spatial_adaptivity_vortex_rim_vs_center(self):
        """Per-PARTICLE adaptivity: a rim particle sweeping a long arc
        across many stencils needs more substeps than one sitting near
        the vortex core at the same tolerance."""
        from kelp.numerics.integrators import rk45_step
        mesh, vel, c = self._vortex()
        st = make_state(np.array([c + 0.02, c + 0.5]),
                        np.array([c, c]))          # core vs rim
        diag = {}
        rk45_step(st, vel, mesh, 21600.0, 1.0, rk45_tol_m=1e-3,
                  rk45_diag=diag)
        assert diag["nsub"][1] > diag["nsub"][0]


# ---------------------------------------------------------------------------
# Failure semantics
# ---------------------------------------------------------------------------

class TestFailures:
    def test_exit_reverts_with_out_of_bounds(self):
        from kelp.numerics.integrators import rk45_step
        mesh = _make_mesh()
        # strong uniform eastward flow, particle near the east rim
        vel = VelocityField(u=np.full(mesh.n_elements, 5.0),
                            v=np.zeros(mesh.n_elements),
                            eta=np.zeros(mesh.n_nodes), w=None)
        st = make_state(np.array([0.14]), np.array([0.0]))
        st = st.replace(failure_reason=np.zeros(1, dtype=np.int8))  # opt-in
        out = rk45_step(st, vel, mesh, 3600.0, 1.0)
        assert out.status[0] == ParticleStatus.OUTSIDE
        assert out.lon[0] == st.lon[0]          # reverted
        assert out.failure_reason is not None
        assert out.failure_reason[0] in (FailureReason.OUT_OF_BOUNDS,
                                         FailureReason.INTERPOLATION_FAILED)

    def test_dead_particles_untouched_and_input_immutable(self):
        from kelp.numerics.integrators import rk45_step
        mesh = _make_mesh()
        vel = _make_vel(mesh, 0.0)
        st = make_state(np.full(3, LON0), np.full(3, LAT0))
        status = st.status.copy()
        status[1] = ParticleStatus.OUTSIDE
        st = st.replace(status=status)
        lon_before = st.lon.copy()
        out = rk45_step(st, vel, mesh, 600.0, 1.0)
        assert np.array_equal(st.lon, lon_before)      # immutability
        assert out.lon[1] == st.lon[1]                 # dead untouched
        assert out.lon[0] != st.lon[0]

    def test_hmin_clamp_warns_once(self):
        from kelp.numerics.integrators import rk45_step
        mesh = _make_mesh()
        vel = _make_vel(mesh, 0.0)
        # curvature via rotation + absurd tolerance forces h below h_min
        with pytest.warns(RuntimeWarning, match="h_min"):
            rk45_step(_state0(), vel, mesh, 3600.0, 1.0,
                      vel_next=_make_vel(mesh, 3600.0), dt_to_next=3600.0,
                      rk45_tol_m=1e-300, rk45_h_min=100.0)

    def test_uniform_split_diffusion_composes(self):
        from kelp.numerics.integrators import rk45_step
        from kelp.numerics.diffusion import DiffusionStepArgs
        mesh = _make_mesh()
        vel = _make_vel(mesh, 0.0)
        args = DiffusionStepArgs(scheme="uniform", kh_x=5.0, kh_y=5.0,
                                 seed=3, real_id=0, step_i=0)
        st = make_state(np.full(50, LON0), np.full(50, LAT0))
        out = rk45_step(st, vel, mesh, 600.0, 1.0, diffusion=args)
        det = rk45_step(st, vel, mesh, 600.0, 1.0)
        assert not np.allclose(out.lon, det.lon)       # kick applied
        assert not np.allclose(out.lat, det.lat)


# ---------------------------------------------------------------------------
# Tracking integration
# ---------------------------------------------------------------------------

class TestTrackingIntegration:
    def _cfg(self, tmp_path, **numerics_kw):
        from tests.test_tracking_diffusion import (
            _make_fesomc_wide, _write_particles)
        from kelp.core.config import (AppConfig, DataConfig, MeshConfig,
                                          ModeConfig, NumericsConfig,
                                          TimeConfig)
        nc, _ = _make_fesomc_wide(tmp_path, u_val=0.2, v_val=0.1)
        seeds = tmp_path / "p.dat"
        _write_particles(seeds)
        out = tmp_path / "out_rk45.nc"
        return AppConfig(
            mode=ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0),
            numerics=NumericsConfig(dt=100.0, integrator="rk45",
                                    output_freq=1, **numerics_kw),
            mesh=MeshConfig(type="fesomc", meshpath=str(nc)),
            data=DataConfig(ncinput=str(nc), inicloud=str(seeds),
                            output_path=str(out), time_averaging="instant"),
            time=TimeConfig(start_time_idx=0, n_steps=3),
        )

    def test_e2e_run_and_metadata(self, tmp_path):
        import netCDF4
        from kelp.tracking import run_tracking
        cfg = self._cfg(tmp_path, rk45_tol_m=0.5)
        run_tracking(cfg)
        with netCDF4.Dataset(cfg.data.output_path) as nc:
            assert float(nc.getncattr("rk45_tol_m")) == pytest.approx(0.5)
            lon = np.array(nc.variables["lon"][:])
        assert np.all(np.isfinite(lon))
        assert not np.allclose(lon[:, -1], lon[:, 0])   # particles moved

    def test_pchip_with_rk45_rejected_pre_writer(self, tmp_path):
        from kelp.tracking import run_tracking
        cfg = self._cfg(tmp_path, temporal_interp=True, file_dt=3600.0,
                        temporal_reconstruction="pchip")
        with pytest.raises((ValueError, NotImplementedError),
                           match="rk45|pchip"):
            run_tracking(cfg)
        assert not Path(cfg.data.output_path).exists()

    def test_walk_with_rk45_rejected_pre_writer(self, tmp_path):
        from kelp.tracking import run_tracking
        cfg = self._cfg(tmp_path, elem_search="walk")
        with pytest.raises((ValueError, NotImplementedError), match="rk45"):
            run_tracking(cfg)
        assert not Path(cfg.data.output_path).exists()

    def test_em_with_rk45_rejected(self, tmp_path):
        from kelp.tracking import run_tracking
        from kelp.core.config import DiffusionConfig
        cfg = self._cfg(tmp_path)
        cfg.numerics.diffusion = DiffusionConfig(
            enabled=True, scheme="em", composition="split_rk")
        with pytest.raises((ValueError, NotImplementedError),
                           match="rk45"):
            run_tracking(cfg)

    def test_3d_with_rk45_rejected_pre_writer(self, tmp_path):
        from tests.test_tracking import _make_fesom2_3x3_3d
        from kelp.core.config import (AppConfig, DataConfig, MeshConfig,
                                          ModeConfig, NumericsConfig,
                                          TimeConfig)
        from kelp.tracking import run_tracking
        mesh_dir, vel_path, _ = _make_fesom2_3x3_3d(tmp_path)
        seeds = tmp_path / "p3.dat"
        seeds.write_text("5.0 5.0 20.0\n")
        out = tmp_path / "o3.nc"
        cfg = AppConfig(
            mode=ModeConfig(Fdrift2D=False, Fdrift3D=True),
            numerics=NumericsConfig(dt=100.0, integrator="rk45"),
            mesh=MeshConfig(type="fesom2",
                            meshpath=str(tmp_path / "f2mesh3d")),
            data=DataConfig(ncinput=str(vel_path), inicloud=str(seeds),
                            output_path=str(out),
                            time_averaging="instant"),
            time=TimeConfig(start_time_idx=0, n_steps=2),
        )
        with pytest.raises(NotImplementedError, match="rk45|3-D|3D"):
            run_tracking(cfg)
        assert not out.exists()


# ---------------------------------------------------------------------------
# Butcher tableau pins (order conditions — a single wrong digit passes
# every behavioral test above, so pin the algebra directly)
# ---------------------------------------------------------------------------

class TestTableau:
    def test_dp54_order_conditions_and_error_weights(self):
        from fractions import Fraction as F
        from kelp.numerics import integrators as I
        c = [F(0), F(1, 5), F(3, 10), F(4, 5), F(8, 9), F(1), F(1)]
        rows = [
            [I._DP_A21],
            [I._DP_A31, I._DP_A32],
            [I._DP_A41, I._DP_A42, I._DP_A43],
            [I._DP_A51, I._DP_A52, I._DP_A53, I._DP_A54],
            [I._DP_A61, I._DP_A62, I._DP_A63, I._DP_A64, I._DP_A65],
        ]
        for i, row in enumerate(rows, start=1):
            assert abs(sum(row) - float(c[i])) < 1e-14, i   # Σa_ij = c_i (float summation of ±12-magnitude terms rounds ~5e-15)
        b = [I._DP_B1, 0.0, I._DP_B3, I._DP_B4, I._DP_B5, I._DP_B6, 0.0]
        assert abs(sum(b) - 1.0) < 5e-16                     # order 1
        assert abs(sum(bi * float(ci) for bi, ci in zip(b, c))
                   - 0.5) < 5e-16                            # order 2
        assert abs(sum(bi * float(ci) ** 2 for bi, ci in zip(b, c))
                   - 1.0 / 3.0) < 5e-16                      # order 3
        # canonical DOPRI5 error weights (Hairer), exact fractions
        exact_e = {
            "_DP_E1": F(71, 57600), "_DP_E3": F(-71, 16695),
            "_DP_E4": F(71, 1920), "_DP_E5": F(-17253, 339200),
            "_DP_E6": F(22, 525), "_DP_E7": F(-1, 40),
        }
        for name, frac in exact_e.items():
            assert getattr(I, name) == float(frac), name
        # E = b5 - b4 must hold algebraically (embedded pair consistency)
        b4 = [F(5179, 57600), F(0), F(7571, 16695), F(393, 640),
              F(-92097, 339200), F(187, 2100), F(1, 40)]
        b5 = [F(35, 384), F(0), F(500, 1113), F(125, 192),
              F(-2187, 6784), F(11, 84), F(0)]
        for name, i in (("_DP_E1", 0), ("_DP_E3", 2), ("_DP_E4", 3),
                        ("_DP_E5", 4), ("_DP_E6", 5), ("_DP_E7", 6)):
            assert getattr(I, name) == float(b5[i] - b4[i] + F(0)), name

    def test_every_tableau_constant_exact(self):
        """Pin every constant to its exact fraction — balanced wrong
        digits could pass the row-sum/order checks (review nitpick)."""
        from fractions import Fraction as F
        from kelp.numerics import integrators as I
        exact = {
            "_DP_C2": F(1, 5), "_DP_C3": F(3, 10), "_DP_C4": F(4, 5),
            "_DP_C5": F(8, 9),
            "_DP_A21": F(1, 5), "_DP_A31": F(3, 40), "_DP_A32": F(9, 40),
            "_DP_A41": F(44, 45), "_DP_A42": F(-56, 15),
            "_DP_A43": F(32, 9),
            "_DP_A51": F(19372, 6561), "_DP_A52": F(-25360, 2187),
            "_DP_A53": F(64448, 6561), "_DP_A54": F(-212, 729),
            "_DP_A61": F(9017, 3168), "_DP_A62": F(-355, 33),
            "_DP_A63": F(46732, 5247), "_DP_A64": F(49, 176),
            "_DP_A65": F(-5103, 18656),
            "_DP_B1": F(35, 384), "_DP_B3": F(500, 1113),
            "_DP_B4": F(125, 192), "_DP_B5": F(-2187, 6784),
            "_DP_B6": F(11, 84),
        }
        for name, frac in exact.items():
            assert getattr(I, name) == float(frac), name


class TestReviewPins:
    def test_pathological_tol_force_accepts_alive(self):
        """B1 (round-1 review): at the DEFAULT h_min = dt*1e-4 a
        pathological tolerance must crawl through in force-accepted
        substeps with ONE warning — never die on the substep cap with a
        contradictory 'force-accepted' message."""
        from kelp.numerics.integrators import rk45_step
        mesh, vel, c = TestAccuracy._vortex()
        st = make_state(np.array([c + 0.5]), np.array([c]))
        with pytest.warns(RuntimeWarning, match="h_min"):
            out = rk45_step(st, vel, mesh, 3600.0, 1.0,
                            rk45_tol_m=1e-300)
        assert out.status[0] == ParticleStatus.ALIVE
        assert out.lon[0] != st.lon[0]          # it actually advanced

    def test_within_tol_hmin_run_does_not_warn(self):
        """A substep at the h_min floor whose error is WITHIN tolerance
        is not a force-accept — no warning, no clamp count."""
        import warnings as _w
        from kelp.numerics.integrators import rk45_step
        mesh = _make_mesh()
        vel = _make_vel(mesh, 0.0)              # constant field, err = 0
        diag = {}
        with _w.catch_warnings():
            _w.simplefilter("error", RuntimeWarning)
            out = rk45_step(_state0(), vel, mesh, 600.0, 1.0,
                            rk45_h_min=7200.0, rk45_diag=diag)
        assert out.status[0] == ParticleStatus.ALIVE
        assert diag["hmin_clamped"] == 0

    def test_stage_failure_retried_at_smaller_h(self):
        """A failed stage TRIAL is a rejection, not a death: a strong
        near-seed jet makes late first-attempt stage trials (k4-k7 at
        h=dt; verified by probe) overshoot onto a doctored corner node
        (empty element ring -> interpolation fails); smaller substeps
        stay on healthy stencils and the particle survives."""
        from kelp.numerics.integrators import rk45_step
        mesh = _make_mesh()
        corner = int(np.argmax(np.asarray(mesh.nodes_lon)
                               + np.asarray(mesh.nodes_lat)))
        mesh.node_to_elems_count = mesh.node_to_elems_count.copy()
        mesh.node_to_elems_count[corner] = 0
        # NE jet only in elements touching the seed; calm elsewhere
        ec_lon = np.asarray(mesh.elem_center_lon)
        ec_lat = np.asarray(mesh.elem_center_lat)
        near = np.hypot(ec_lon, ec_lat) < 0.05
        u = np.where(near, 5.0, 0.01)
        vel = VelocityField(u=u, v=u.copy(),
                            eta=np.zeros(mesh.n_nodes), w=None)
        diag = {}
        out = rk45_step(_state0(), vel, mesh, 30000.0, 1.0,
                        rk45_tol_m=1.0, rk45_diag=diag)
        assert out.status[0] == ParticleStatus.ALIVE
        assert diag["nsub"][0] > 1              # at least one retry/substep
        assert abs(out.lon[0]) < 0.14           # stayed inside the mesh
