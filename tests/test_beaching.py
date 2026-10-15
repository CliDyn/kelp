"""Tests for the beaching model (TODO 2.2): criteria + resuspension."""
import numpy as np
import pytest

from kelp.core.mesh import Mesh, build_search, build_node_to_elems
from kelp.core.particles import ParticleStatus, make_state
from kelp.core.io import VelocityField
from kelp.numerics.beaching import (
    BeachingStepArgs, evaluate_beaching, COMP_BEACH, COMP_RESUSP, _u01,
    RESUSP_NEVER, RESUSP_AUTO, RESUSP_STOCH, RESUSP_BOTH,
)


def _grid_mesh(xs, ys, depth=50.0):
    # identical to tests/test_mesh.py::_grid_mesh + a depth parameter
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel().astype(np.float64)
    lat = gy.ravel().astype(np.float64)
    nx, ny = len(xs), len(ys)
    elems = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            n1, n2 = n0 + 1, n0 + nx
            elems.append([n0, n1, n2]); elems.append([n1, n2 + 1, n2])
    elem_nodes = np.array(elems, dtype=np.int32)
    n2e, n2e_count = build_node_to_elems(nx * ny, elem_nodes)
    mesh = Mesh(name="grid", nodes_lon=lon, nodes_lat=lat,
                depth=np.full(nx * ny, float(depth)), elem_nodes=elem_nodes,
                elem_center_lon=np.array([lon[e].mean() for e in elem_nodes]),
                elem_center_lat=np.array([lat[e].mean() for e in elem_nodes]),
                node_to_elems=n2e, node_to_elems_count=n2e_count)
    build_search(mesh)
    return mesh


def _vel(mesh, eta=0.0):
    return VelocityField(u=np.zeros(mesh.n_elements),
                         v=np.zeros(mesh.n_elements),
                         eta=np.full(mesh.n_nodes, float(eta)))


def _args(**kw):
    base = dict(use_depth=True, use_coastal=False,
                use_stochastic=False, depth_min=0.3, use_eta=False,
                zone_depth=0.6, t_beach=86400.0, resuspension=RESUSP_NEVER,
                rewet_hysteresis=0.1, min_wet_steps=1, t_resuspend=0.0,
                seed=42, real_id=0, step_i=0, dt=3600.0)
    base.update(kw)
    return BeachingStepArgs(**base)


def _state_at(mesh, lon, lat, status=None):
    st = make_state(np.atleast_1d(np.float64(lon)),
                    np.atleast_1d(np.float64(lat)))
    from kelp.core.mesh import find_nearest_nodes
    nn = find_nearest_nodes(mesh, st.lon, st.lat)
    st = st.replace(nearest_node=nn)
    if status is not None:
        st = st.replace(status=np.array([status], dtype=st.status.dtype))
    return st


class TestDepthCriterion:
    def test_shallow_node_beaches(self):
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5))
        mesh.depth[12] = 0.1                      # node (2,2): shallow
        st = _state_at(mesh, 2.0, 2.0)
        to_beach, to_release, streak = evaluate_beaching(
            st, mesh, _vel(mesh), _args())
        assert to_beach[0] and not to_release[0]

    def test_deep_node_stays(self):
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5))
        st = _state_at(mesh, 2.0, 2.0)
        to_beach, _, _ = evaluate_beaching(st, mesh, _vel(mesh), _args())
        assert not to_beach[0]

    def test_use_eta_false_ignores_eta(self):
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5))
        st = _state_at(mesh, 2.0, 2.0)
        to_beach, _, _ = evaluate_beaching(
            st, mesh, _vel(mesh, eta=-49.9), _args(use_eta=False))
        assert not to_beach[0]

    def test_raw_h_not_dmin_clamped(self):
        # depth + eta = 0.2 < depth_min=0.3, even though numerics would
        # clamp the column to Dmin=0.5: beaching must see RAW H.
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5))
        mesh.depth[:] = 1.0
        st = _state_at(mesh, 2.0, 2.0)
        to_beach, _, _ = evaluate_beaching(
            st, mesh, _vel(mesh, eta=-0.8), _args(use_eta=True))
        assert to_beach[0]

    def test_dry_mask_forces_beach_on_deep_water(self):
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5))
        st = _state_at(mesh, 2.0, 2.0)
        mask = np.zeros(mesh.n_nodes); mask[st.nearest_node[0]] = 1.0
        to_beach, _, _ = evaluate_beaching(
            st, mesh, _vel(mesh), _args(), dry_mask=mask)
        assert to_beach[0]


class TestCoastalElementCriterion:
    def test_boundary_element_beaches_deep_particle(self):
        # user's case: boundary element NOT shallow
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5))
        from kelp.core.mesh import build_boundary_info, \
            find_containing_elements
        build_boundary_info(mesh)
        st = _state_at(mesh, 0.2, 0.2)            # in a rim element
        elems, _ = find_containing_elements(mesh, st.lon, st.lat,
                                            st.nearest_node)
        to_beach, _, _ = evaluate_beaching(
            st, mesh, _vel(mesh),
            _args(use_depth=False, use_coastal=True),
            elem_is_coastal=mesh._elem_is_coastal,
            containing_elems=elems)
        assert to_beach[0]

    def test_interior_element_does_not(self):
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5))
        from kelp.core.mesh import build_boundary_info, \
            find_containing_elements
        build_boundary_info(mesh)
        st = _state_at(mesh, 2.0, 2.1)
        elems, _ = find_containing_elements(mesh, st.lon, st.lat,
                                            st.nearest_node)
        to_beach, _, _ = evaluate_beaching(
            st, mesh, _vel(mesh),
            _args(use_depth=False, use_coastal=True),
            elem_is_coastal=mesh._elem_is_coastal,
            containing_elems=elems)
        assert not to_beach[0]


class TestStochasticCriterion:
    def test_exact_counter_rng_decision(self):
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5))
        mesh.depth[:] = 0.5                       # inside zone (< 0.6)
        st = _state_at(mesh, 2.0, 2.0)
        args = _args(use_depth=False, use_stochastic=True,
                     t_beach=3600.0, dt=3600.0, seed=7, step_i=3)
        u = _u01(7, 0, 0, 3, COMP_BEACH)   # same 53-bit helper as the kernel
        p = 1.0 - np.exp(-1.0)
        to_beach, _, _ = evaluate_beaching(st, mesh, _vel(mesh), args)
        assert to_beach[0] == (u < p)

    def test_outside_zone_never(self):
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5))  # 50 m
        st = _state_at(mesh, 2.0, 2.0)
        args = _args(use_depth=False, use_stochastic=True,
                     t_beach=1e-9, dt=3600.0)     # p ~ 1
        to_beach, _, _ = evaluate_beaching(st, mesh, _vel(mesh), args)
        assert not to_beach[0]

    def test_coastal_contact_is_in_zone(self):
        # zone-OR works WITHOUT the deterministic coastal criterion:
        # deep particle (H=50, out of depth-zone) in a coastal element,
        # use_coastal=False, stochastic p~1 -> beaches ONLY via the OR.
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5))
        from kelp.core.mesh import build_boundary_info, \
            find_containing_elements
        build_boundary_info(mesh)
        st = _state_at(mesh, 0.2, 0.2)
        elems, _ = find_containing_elements(mesh, st.lon, st.lat,
                                            st.nearest_node)
        args = _args(use_depth=False, use_coastal=False,
                     use_stochastic=True, t_beach=1e-9, dt=3600.0)
        # without topology data: deep -> out of zone -> never
        to_beach, _, _ = evaluate_beaching(st, mesh, _vel(mesh), args)
        assert not to_beach[0]
        # with topology data (use_coastal still False): in-zone via OR
        to_beach2, _, _ = evaluate_beaching(
            st, mesh, _vel(mesh), args,
            elem_is_coastal=mesh._elem_is_coastal, containing_elems=elems)
        assert to_beach2[0]

    def test_fraction_approx_p(self):
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5))
        mesh.depth[:] = 0.5
        P = 2000
        st = make_state(np.full(P, 2.0), np.full(P, 2.0))
        from kelp.core.mesh import find_nearest_nodes
        st = st.replace(nearest_node=find_nearest_nodes(mesh, st.lon, st.lat))
        args = _args(use_depth=False, use_stochastic=True,
                     t_beach=3600.0 / np.log(2.0), dt=3600.0)
        to_beach, _, _ = evaluate_beaching(st, mesh, _vel(mesh), args)
        frac = to_beach.mean()
        assert abs(frac - 0.5) < 0.05

    def test_rng_keyed_by_original_index(self):
        # Project trap: counter-RNG must use the ORIGINAL particle index,
        # not the in-zone compaction index. Particle 0 deep (out of
        # zone), particle 1 shallow (in zone): the selection compacts to
        # [1]; a compacted impl would draw u01(..., 0, ...).
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5))
        node1 = None
        st = make_state(np.array([2.0, 1.0]), np.array([2.0, 1.0]))
        from kelp.core.mesh import find_nearest_nodes
        nn = find_nearest_nodes(mesh, st.lon, st.lat)
        st = st.replace(nearest_node=nn)
        mesh.depth[nn[1]] = 0.5                    # particle 1 in zone
        u1 = _u01(42, 0, 1, 0, COMP_BEACH)         # original index 1
        u0 = _u01(42, 0, 0, 0, COMP_BEACH)         # compacted would use 0
        assert u0 != u1                            # discriminating draws
        p = (min(u0, u1) + max(u0, u1)) / 2.0      # p strictly between
        t_beach = -3600.0 / np.log(1.0 - p)        # invert p=1-exp(-dt/t)
        args = _args(use_depth=False, use_stochastic=True,
                     t_beach=t_beach, dt=3600.0, seed=42, step_i=0)
        to_beach, _, _ = evaluate_beaching(st, mesh, _vel(mesh), args)
        assert not to_beach[0]                     # out of zone
        assert to_beach[1] == (u1 < p)             # original-index draw


class TestResuspension:
    def _beached_state(self, mesh):
        return _state_at(mesh, 2.0, 2.0, status=ParticleStatus.BEACHED)

    def test_auto_release_after_rewet(self):
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5))
        mesh.depth[:] = 1.0
        st = self._beached_state(mesh)
        args = _args(use_eta=True, resuspension=RESUSP_AUTO,
                     min_wet_steps=2)
        wet = np.zeros(1, dtype=np.int32)
        _, rel1, wet = evaluate_beaching(st, mesh, _vel(mesh, eta=0.0),
                                         args, wet_streak=wet)
        assert not rel1[0] and wet[0] == 1
        _, rel2, wet = evaluate_beaching(st, mesh, _vel(mesh, eta=0.0),
                                         args, wet_streak=wet)
        assert rel2[0]

    def test_streak_resets_on_dry(self):
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5))
        mesh.depth[:] = 1.0
        st = self._beached_state(mesh)
        args = _args(use_eta=True, resuspension=RESUSP_AUTO,
                     min_wet_steps=2)
        wet = np.ones(1, dtype=np.int32)
        _, rel, wet = evaluate_beaching(st, mesh, _vel(mesh, eta=-0.9),
                                        args, wet_streak=wet)
        assert not rel[0] and wet[0] == 0

    def test_hysteresis_band_stays_beached(self):
        # depth_min=0.3, hysteresis=0.1: H=0.35 is wet for beaching but
        # below the release threshold 0.4 -> no release, streak 0
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5))
        mesh.depth[:] = 0.35
        st = self._beached_state(mesh)
        args = _args(resuspension=RESUSP_AUTO)
        _, rel, wet = evaluate_beaching(st, mesh, _vel(mesh), args,
                                        wet_streak=np.zeros(1, np.int32))
        assert not rel[0] and wet[0] == 0

    @pytest.mark.parametrize("mode", [RESUSP_AUTO, RESUSP_STOCH,
                                      RESUSP_BOTH])
    def test_dry_mask_blocks_resuspension(self, mode):
        """Review 2026-09-22 (F06): the dry mask beached particles but
        rewetting looked at depth + eta only — a permanently masked node
        with a large nominal depth released its particle every step."""
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5))
        st = self._beached_state(mesh)            # depth 50 m: 'wet' by H
        mask = np.zeros(mesh.n_nodes)
        mask[st.nearest_node[0]] = 1.0
        args = _args(resuspension=mode, t_resuspend=1.0)
        _, rel, wet = evaluate_beaching(st, mesh, _vel(mesh), args,
                                        dry_mask=mask,
                                        wet_streak=np.ones(1, np.int32))
        assert not rel[0] and wet[0] == 0

    def test_unknown_nearest_node_is_not_wet(self):
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5))
        st = self._beached_state(mesh)
        st = st.replace(nearest_node=np.array([-1], dtype=np.int32))
        _, rel, wet = evaluate_beaching(
            st, mesh, _vel(mesh), _args(resuspension=RESUSP_AUTO),
            wet_streak=np.ones(1, np.int32))
        assert not rel[0] and wet[0] == 0

    def test_never_mode_never_releases(self):
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5))
        st = self._beached_state(mesh)
        _, rel, _ = evaluate_beaching(st, mesh, _vel(mesh),
                                      _args(resuspension=RESUSP_NEVER),
                                      wet_streak=np.zeros(1, np.int32))
        assert not rel[0]

    def test_stochastic_release_exact(self):
        # deep water (rewet true) -> release decided purely by the draw
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5))
        st = self._beached_state(mesh)
        args = _args(resuspension=RESUSP_STOCH, t_resuspend=3600.0,
                     dt=3600.0, seed=11, step_i=5)
        u = _u01(11, 0, 0, 5, COMP_RESUSP)
        _, rel, _ = evaluate_beaching(st, mesh, _vel(mesh), args,
                                      wet_streak=np.zeros(1, np.int32))
        assert rel[0] == (u < 1.0 - np.exp(-1.0))

    def test_stochastic_release_gated_by_water(self):
        # dry anchor: stochastic release must NEVER fire (spec)
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5))
        mesh.depth[:] = 0.1                        # dry everywhere
        st = self._beached_state(mesh)
        args = _args(resuspension=RESUSP_STOCH, t_resuspend=1e-9,
                     dt=3600.0)                    # p ~ 1
        _, rel, _ = evaluate_beaching(st, mesh, _vel(mesh), args,
                                      wet_streak=np.zeros(1, np.int32))
        assert not rel[0]


class TestStatusInteraction:
    def test_outside_particles_untouched(self):
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5))
        mesh.depth[:] = 0.1
        st = _state_at(mesh, 2.0, 2.0, status=ParticleStatus.OUTSIDE)
        to_beach, to_release, _ = evaluate_beaching(
            st, mesh, _vel(mesh), _args())
        assert not to_beach[0] and not to_release[0]


# ===================================================================
# Tracking-layer integration (Task 4)
# ===================================================================

import xarray as xr
from kelp.readers import get_reader
from kelp.core.config import ModeConfig, NumericsConfig, BeachingConfig


def _fesom2_flat_ds(n_nodes, n_time=6, eta_series=None, n_elem=32):
    eta = np.zeros((n_time, n_nodes))
    if eta_series is not None:
        eta[:] = np.asarray(eta_series, dtype=np.float64)[:, None]
    return xr.Dataset(
        {"u": (("time", "elem"), np.zeros((n_time, n_elem))),
         "v": (("time", "elem"), np.zeros((n_time, n_elem))),
         "ssh": (("time", "nod2"), eta)},
        coords={"time": np.arange(n_time)})


class TestTrackingBeaching:
    def _run(self, mesh, ds, beaching, n_steps=4, init_lon=2.0,
             init_lat=2.0, **kw):
        from kelp.tracking import run_single_realization
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False)
        numerics = NumericsConfig(dt=3600.0, integrator="euler",
                                  beaching=beaching)
        init = make_state(np.array([init_lon]), np.array([init_lat]))
        reader = get_reader("fesom2")
        return run_single_realization(
            mesh, ds, reader, init, mode, numerics,
            start_time_idx=0, n_steps=n_steps, **kw)

    def test_tidal_cycle_beach_then_resuspend(self):
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5),
                          depth=1.0)
        eta_series = [0.0, -0.9, -0.9, 0.0, 0.0, 0.0]
        ds = _fesom2_flat_ds(mesh.n_nodes, eta_series=eta_series)
        ds["u"][:] = 0.2          # nonzero drift: freeze is OBSERVABLE
        b = BeachingConfig(enabled=True, criteria=["depth"],
                           depth_min=0.3, use_eta=True,
                           resuspension="auto", min_wet_steps=1)
        rec = self._run(mesh, ds, b, n_steps=5)
        arrs = rec.to_arrays()
        s = arrs["status"][:, 0]
        A, B = ParticleStatus.ALIVE, ParticleStatus.BEACHED
        # hook after step N with step N's start-of-step eta payload:
        assert list(s) == [A, B, B, A, A]
        lon = arrs["lon"][:, 0]
        assert lon[1] == lon[2]                # frozen while beached
        # released at the obs-3 hook AFTER step 3's advection -> ALIVE
        # but stationary at obs 3; first MOVES on step 4:
        assert lon[3] == lon[2]
        assert lon[4] > lon[3]
        assert lon[0] > 2.0                    # moved before beaching

    def test_beached_obs_sampled_var_is_nan(self):
        # Spec item 13: BEACHED particles get NaN in sampled vars (documented
        # contract — the sampler NaNs non-ALIVE via the alive mask).  Reuses
        # the tidal config from test_tidal_cycle_beach_then_resuspend (status
        # sequence [A, B, B, A, A]) with a constant, finite temp scalar so the
        # NaN at obs 1-2 can only come from the BEACHED status, not the field.
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5),
                          depth=1.0)
        eta_series = [0.0, -0.9, -0.9, 0.0, 0.0, 0.0]
        ds = _fesom2_flat_ds(mesh.n_nodes, eta_series=eta_series)
        ds["u"][:] = 0.2
        # constant, finite node-based scalar (time, nod2): ALIVE obs MUST be
        # finite, so a NaN can only be the beaching contract firing.
        ds["temp"] = (("time", "nod2"),
                      np.full((ds.sizes["time"], mesh.n_nodes), 5.0))
        b = BeachingConfig(enabled=True, criteria=["depth"],
                           depth_min=0.3, use_eta=True,
                           resuspension="auto", min_wet_steps=1)
        rec = self._run(mesh, ds, b, n_steps=5, sample_vars=["temp"],
                        sample_interp="nearest")
        arrs = rec.to_arrays()
        s = arrs["status"][:, 0]
        temp = arrs["temp"][:, 0]
        A, B = ParticleStatus.ALIVE, ParticleStatus.BEACHED
        assert list(s) == [A, B, B, A, A]      # same sequence as the headline
        beached = s == B
        alive = s == A
        assert np.all(np.isnan(temp[beached]))     # BEACHED -> NaN
        assert np.all(np.isfinite(temp[alive]))    # ALIVE   -> finite (5.0)

    def test_on_failure_remap_shallow_coast(self):
        # context-gated remap on the NORMAL fused path
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5))
        mesh.depth[:] = 0.5                    # zone = 0.6 > 0.5
        ds = _fesom2_flat_ds(mesh.n_nodes)
        ds["u"][:] = 50.0
        b_on = BeachingConfig(enabled=True, criteria=["on_failure"],
                              depth_min=0.3)
        rec = self._run(mesh, ds, b_on, n_steps=4, init_lon=3.9)
        assert rec.to_arrays()["status"][-1, 0] == ParticleStatus.BEACHED

    def test_on_failure_deep_exit_stays_outside(self):
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5))
        ds = _fesom2_flat_ds(mesh.n_nodes)
        ds["u"][:] = 50.0
        b_on = BeachingConfig(enabled=True, criteria=["on_failure"],
                              depth_min=0.3)
        rec = self._run(mesh, ds, b_on, n_steps=4, init_lon=3.9)
        assert rec.to_arrays()["status"][-1, 0] == ParticleStatus.OUTSIDE

    def test_stochastic_only_builds_topology_for_zone_or(self):
        # criteria=[stochastic], deep coastal element (H=50 >> zone), p~1:
        # MUST beach via the coastal zone-OR -> the hook must build the
        # boundary topology even though coastal_element is not enabled.
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5),
                          depth=50.0)
        ds = _fesom2_flat_ds(mesh.n_nodes)            # zero velocity
        b = BeachingConfig(enabled=True, criteria=["stochastic"],
                           depth_min=0.3, t_beach=1e-9,
                           resuspension="never", seed=7)
        # seed near an EDGE node (3-element ring, well-posed leastsq
        # stencil) — the old (0.2, 0.2) corner anchor has a rank-deficient
        # ring and now fails interpolation cleanly (scale-relative det
        # guard, 2026-08) instead of surviving on the zero-RHS fluke.
        rec = self._run(mesh, ds, b, n_steps=2, init_lon=0.6, init_lat=0.2)
        assert rec.to_arrays()["status"][-1, 0] == ParticleStatus.BEACHED

    def test_disabled_no_remap(self):
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5))
        ds = _fesom2_flat_ds(mesh.n_nodes)
        ds["u"][:] = 50.0
        rec = self._run(mesh, ds, BeachingConfig(), n_steps=4,
                        init_lon=3.9)
        assert rec.to_arrays()["status"][-1, 0] == ParticleStatus.OUTSIDE

    def test_use_eta_failfast_without_ssh_direct(self):
        # run_single_realization is reachable directly (bypassing
        # run_tracking's startup guard): use_eta=true + a ds with no
        # ssh/eta/zos must STILL raise rather than silently zero eta.
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5))
        ds = _fesom2_flat_ds(mesh.n_nodes).drop_vars(
            ["ssh", "eta", "zos"], errors="ignore")
        b = BeachingConfig(enabled=True, criteria=["depth"],
                           depth_min=0.3, use_eta=True)
        with pytest.raises(ValueError, match="ssh"):
            self._run(mesh, ds, b, n_steps=2)

    def test_dry_mask_var_failfast_without_wiring_direct(self):
        # dry_mask_var set but the variable is NOT in sample_vars (the
        # run_tracking wiring that injects it) -> ValueError, not a silent
        # skip of the dry-mask read.
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5))
        ds = _fesom2_flat_ds(mesh.n_nodes)
        b = BeachingConfig(enabled=True, criteria=["depth"],
                           depth_min=0.3, dry_mask_var="wetdry")
        with pytest.raises(ValueError, match="dry_mask_var"):
            self._run(mesh, ds, b, n_steps=2)

    def test_dormant_byte_identical(self, tmp_path):
        # two writer files: beaching=default-disabled vs beaching absent —
        # identical variable sets, global attrs, and values
        import netCDF4 as nc4
        from kelp.tracking import run_single_realization
        from kelp.writers import TrajectoryWriter

        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5),
                          depth=1.0)
        ds = _fesom2_flat_ds(mesh.n_nodes)
        ds["u"][:] = 0.1
        reader = get_reader("fesom2")
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False)
        init = make_state(np.array([2.0]), np.array([2.0]))

        def _go(path, numerics):
            w = TrajectoryWriter(path, init.n_particles)
            w.open()
            try:
                run_single_realization(
                    mesh, ds, reader, init, mode, numerics,
                    start_time_idx=0, n_steps=4, writer=w)
            finally:
                w.close()

        path_a = str(tmp_path / "absent.nc")
        path_b = str(tmp_path / "disabled.nc")
        _go(path_a, NumericsConfig(dt=3600.0, integrator="euler"))
        _go(path_b, NumericsConfig(dt=3600.0, integrator="euler",
                                   beaching=BeachingConfig()))  # enabled=False

        with nc4.Dataset(path_a) as da, nc4.Dataset(path_b) as db:
            assert set(da.variables) == set(db.variables)
            assert set(da.ncattrs()) == set(db.ncattrs())
            assert not any("beach" in a.lower() for a in da.ncattrs())
            for name in da.variables:
                va = np.ma.filled(da.variables[name][:], -999.0)
                vb = np.ma.filled(db.variables[name][:], -999.0)
                np.testing.assert_array_equal(
                    va, vb, err_msg=f"variable {name!r} differs")


def _island_mesh(depth=50.0):
    """9x9 grid (0..8) with the 2x2 cell block centred on (4,4) removed →
    an island hole with a closed interior boundary loop.  Same keep-mask
    machinery as tests/test_mesh.py::test_island_hole_elements_coastal, on
    a larger grid so there is a non-coastal interior buffer west of the
    hole for an advection test.
    """
    xs = np.linspace(0, 8, 9)
    ys = np.linspace(0, 8, 9)
    full = _grid_mesh(xs, ys, depth=depth)
    cx, cy = full.elem_center_lon, full.elem_center_lat
    keep = ~((cx > 3.0) & (cx < 5.0) & (cy > 3.0) & (cy < 5.0))
    elem_nodes = full.elem_nodes[keep]
    n2e, n2e_count = build_node_to_elems(full.n_nodes, elem_nodes)
    mesh = Mesh(name="island", nodes_lon=full.nodes_lon,
                nodes_lat=full.nodes_lat,
                depth=np.full(full.n_nodes, float(depth)),
                elem_nodes=elem_nodes,
                elem_center_lon=cx[keep], elem_center_lat=cy[keep],
                node_to_elems=n2e, node_to_elems_count=n2e_count)
    build_search(mesh)
    return mesh


class TestBeachingEdges:
    def _run(self, mesh, ds, beaching, n_steps=4, init_lon=2.0, init_lat=2.0,
             integrator="euler", z0=None, **kw):
        from kelp.tracking import run_single_realization
        is_3d = z0 is not None
        mode = ModeConfig(Fdrift2D=not is_3d, Fdrift3D=is_3d)
        numerics = NumericsConfig(dt=3600.0, integrator=integrator,
                                  beaching=beaching)
        z = None if z0 is None else np.array([z0])
        init = make_state(np.array([init_lon]), np.array([init_lat]), z)
        reader = get_reader("fesom2")
        return run_single_realization(
            mesh, ds, reader, init, mode, numerics,
            start_time_idx=0, n_steps=n_steps, **kw)

    def test_island_contact_beaches_deep_particle(self):
        # Particle advected EAST toward the island shore beaches via
        # coastal_element even though H = 50 (deep) everywhere — pins that
        # island holes are covered with no special code.
        mesh = _island_mesh(depth=50.0)
        ds = _fesom2_flat_ds(mesh.n_nodes, n_elem=mesh.n_elements)
        ds["u"][:] = 12.0          # ~0.4 deg/step east at this latitude
        b = BeachingConfig(enabled=True, criteria=["coastal_element"],
                           depth_min=0.3)
        # Seed at (1.5, 4.0): a deep NON-coastal interior element (cell
        # lon[1,2]), well west of the hole rim (lon≈3) and >1 cell from the
        # outer rim.  Advecting east reaches a coastal (hole-adjacent)
        # element (cell lon[2,3]) within a couple of steps.
        rec = self._run(mesh, ds, b, n_steps=4, init_lon=1.5, init_lat=4.0)
        s = rec.to_arrays()["status"][:, 0]
        assert s[0] == ParticleStatus.ALIVE          # starts interior, deep
        assert s[-1] == ParticleStatus.BEACHED       # stranded on the shore

    def test_middomain_bank_no_boundary(self):
        # Interior shallow bank on a hole-free mesh: coastal_element alone
        # must NOT beach (no boundary edges), but the depth criterion must —
        # pins the documented asymmetry in BOTH directions.
        mesh = _grid_mesh(np.linspace(0, 6, 7), np.linspace(0, 6, 7),
                          depth=50.0)
        # Shallow only at the deep-interior node (3,3) = node 24; its
        # surrounding elements are NOT coastal (>1 cell from every rim).
        from kelp.core.mesh import find_nearest_nodes
        st = make_state(np.array([3.0]), np.array([3.0]))
        nn = find_nearest_nodes(mesh, st.lon, st.lat)[0]
        mesh.depth[nn] = 0.1
        ds = _fesom2_flat_ds(mesh.n_nodes, n_elem=mesh.n_elements)

        # coastal_element alone → never beaches (interior bank).
        b_coast = BeachingConfig(enabled=True, criteria=["coastal_element"],
                                 depth_min=0.3)
        rec_c = self._run(mesh, ds, b_coast, n_steps=2,
                          init_lon=3.0, init_lat=3.0)
        assert rec_c.to_arrays()["status"][-1, 0] == ParticleStatus.ALIVE

        # depth criterion → beaches on the shallow bank.
        b_depth = BeachingConfig(enabled=True, criteria=["depth"],
                                 depth_min=0.3)
        rec_d = self._run(mesh, ds, b_depth, n_steps=2,
                          init_lon=3.0, init_lat=3.0)
        assert rec_d.to_arrays()["status"][-1, 0] == ParticleStatus.BEACHED

    def test_use_eta_failfast_without_ssh(self, tmp_path):
        # beaching.use_eta=true + dataset WITHOUT ssh/eta/zos ->
        # ValueError mentioning ssh at startup, no output file created.
        from kelp.tracking import run_tracking
        beaching_yaml = (
            "  beaching:\n"
            "    enabled: true\n"
            "    criteria: [depth]\n"
            "    depth_min: 0.3\n"
            "    use_eta: true\n")
        # Reuse TestBeachingRunTracking's tmp_path config builder (same as
        # test_dry_mask_e2e_wiring), then strip ssh from the velocity file
        # so the use_eta startup guard has nothing to read.
        cfg, out = TestBeachingRunTracking()._config(
            tmp_path, beaching_yaml, n_steps=3, u=0.0)
        nc = tmp_path / "vel.nc"
        with xr.open_dataset(nc) as _d:
            ds_no_ssh = _d.drop_vars(["ssh", "eta", "zos"],
                                     errors="ignore").load()
        ds_no_ssh.to_netcdf(nc)
        with pytest.raises(ValueError, match="ssh"):
            run_tracking(cfg, output_path=str(out))
        assert not out.exists()

    def test_dry_mask_var_failfast_when_missing(self, tmp_path):
        # beaching.dry_mask_var names a var absent from the dataset ->
        # ValueError naming the var, no output file created.
        from kelp.tracking import run_tracking
        beaching_yaml = (
            "  beaching:\n"
            "    enabled: true\n"
            "    criteria: [depth]\n"
            "    depth_min: 0.3\n"
            "    dry_mask_var: nope\n")
        cfg, out = TestBeachingRunTracking()._config(
            tmp_path, beaching_yaml, n_steps=3, u=0.0)
        with pytest.raises(ValueError, match="nope"):
            run_tracking(cfg, output_path=str(out))
        assert not out.exists()


def _write_fesom2_mesh_dir(d, depth=50.0):
    """Write a 5x5-grid FESOM2 ASCII mesh (nod2d.out/elem2d.out/depth.out)."""
    d.mkdir(parents=True, exist_ok=True)
    xs = np.linspace(0.0, 4.0, 5)
    ys = np.linspace(0.0, 4.0, 5)
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel(); lat = gy.ravel()
    nx = ny = 5
    nod_lines = [str(lon.size)]
    for i in range(lon.size):
        nod_lines.append(f"{i + 1} {lon[i]:.6f} {lat[i]:.6f} 0")
    (d / "nod2d.out").write_text("\n".join(nod_lines) + "\n")
    elems = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            n1, n2 = n0 + 1, n0 + nx
            elems.append((n0, n1, n2)); elems.append((n1, n2 + 1, n2))
    elem_lines = [str(len(elems))]
    for a, b, c in elems:
        elem_lines.append(f"{a + 1} {b + 1} {c + 1}")
    (d / "elem2d.out").write_text("\n".join(elem_lines) + "\n")
    (d / "depth.out").write_text(
        "\n".join(f"{float(depth):.3f}" for _ in range(lon.size)) + "\n")
    return len(elems)


def _write_fesom2_vel_nc(path, n_nodes, n_elem, n_time=4, u=0.0,
                         wetdry=None, eta_series=None):
    """FESOM2 velocity NetCDF: u/v on elements, ssh on nodes, optional
    node-based ``wetdry`` mask."""
    eta = np.zeros((n_time, n_nodes), np.float64)
    if eta_series is not None:
        eta[:] = np.asarray(eta_series, np.float64)[:, None]
    data = {
        "u": (("time", "elem"), np.full((n_time, n_elem), u, np.float64)),
        "v": (("time", "elem"), np.zeros((n_time, n_elem), np.float64)),
        "ssh": (("time", "nod2"), eta),
    }
    if wetdry is not None:
        data["wetdry"] = (("time", "nod2"), np.asarray(wetdry, np.float64))
    xr.Dataset(data, coords={"time": np.arange(n_time)}).to_netcdf(path)


class TestBeachingRunTracking:
    def _config(self, tmp_path, beaching_yaml, n_steps=4, u=0.0,
                wetdry=None, eta_series=None, depth=50.0, n_time=6):
        from kelp.core.config import load_config
        mesh_dir = tmp_path / "mesh"
        n_elem = _write_fesom2_mesh_dir(mesh_dir, depth=depth)
        nc = tmp_path / "vel.nc"
        n_nodes = 25
        _write_fesom2_vel_nc(nc, n_nodes, n_elem, n_time=n_time, u=u,
                             wetdry=wetdry, eta_series=eta_series)
        particles = tmp_path / "particles.dat"
        particles.write_text("2.0 2.0\n")
        out = tmp_path / "out.nc"
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(f"""\
mode:
  Fdrift2D: true
  time_dir: 1.0
numerics:
  dt: 3600.0
  integrator: euler
  output_freq: 1
{beaching_yaml}
mesh:
  type: fesom2
  meshpath: {mesh_dir}
data:
  ncinput: {nc}
  inicloud: {particles}
  output_path: {out}
  time_averaging: instant
time:
  start_time_idx: 0
  n_steps: {n_steps}
""")
        return load_config(str(cfg)), out

    def test_dry_mask_e2e_wiring(self, tmp_path):
        # wetdry=1 at the particle's node forces a beach on DEEP water; the
        # internal-only mask var must NOT appear in the output file, and the
        # beaching_* attrs must be present (enabled run).
        import netCDF4 as nc4
        from kelp.tracking import run_tracking

        n_nodes = 25
        wd = np.zeros((6, n_nodes), np.float64)
        # particle at (2,2) → node 12 on the 5x5 grid (row 2 * 5 + col 2).
        wd[:, 12] = 1.0
        beaching_yaml = (
            "  beaching:\n"
            "    enabled: true\n"
            "    criteria: [depth]\n"
            "    depth_min: 0.3\n"
            "    dry_mask_var: wetdry\n")
        cfg, out = self._config(tmp_path, beaching_yaml, n_steps=3,
                                u=0.0, wetdry=wd, depth=50.0)
        run_tracking(cfg, output_path=str(out))
        from kelp.writers import load_trajectories
        traj = load_trajectories(str(out))
        # Beached despite H=50 (deep) — only the dry mask can do this.
        # status is (P, T); particle 0's final obs.
        assert traj["status"][0, -1] == ParticleStatus.BEACHED
        with nc4.Dataset(out) as ds:
            assert "wetdry" not in ds.variables       # internal-only
            assert ds.getncattr("beaching_enabled") == 1
            assert ds.getncattr("beaching_criteria") == "depth"

    def test_dormant_metadata_via_run_tracking(self, tmp_path):
        # run_tracking twice (beaching absent vs enabled:false): output
        # files have IDENTICAL global attrs (no beaching_* leak).
        import netCDF4 as nc4
        from kelp.tracking import run_tracking

        cfg_absent, out_a = self._config(tmp_path / "a", "", n_steps=3,
                                         u=0.1)
        beaching_off = (
            "  beaching:\n"
            "    enabled: false\n"
            "    criteria: [depth]\n")
        cfg_off, out_b = self._config(tmp_path / "b", beaching_off,
                                      n_steps=3, u=0.1)
        run_tracking(cfg_absent, output_path=str(out_a))
        run_tracking(cfg_off, output_path=str(out_b))
        with nc4.Dataset(out_a) as da, nc4.Dataset(out_b) as db:
            assert set(da.ncattrs()) == set(db.ncattrs())
            assert not any("beach" in a.lower() for a in da.ncattrs())
            assert not any("beach" in a.lower() for a in db.ncattrs())


class TestBeachingRestartAndRNG:
    def _setup(self, eta_series):
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5),
                          depth=1.0)
        ds = _fesom2_flat_ds(mesh.n_nodes, eta_series=eta_series)
        reader = get_reader("fesom2")
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False)
        b = BeachingConfig(
            enabled=True, criteria=["depth", "stochastic"], depth_min=0.3,
            use_eta=True, t_beach=1.0e9,           # rarely beaches, but RNG drawn
            resuspension="auto_or_stochastic", t_resuspend=4000.0,
            rewet_hysteresis=0.1, min_wet_steps=2, seed=2024)
        numerics = NumericsConfig(dt=3600.0, integrator="euler", beaching=b)
        init = make_state(np.array([2.0]), np.array([2.0]))
        return mesh, ds, reader, mode, numerics, init

    def test_restart_bitwise_with_beaching(self, tmp_path):
        # tidal cycle + stochastic resuspension + checkpoint_freq=1:
        # continuous 5-step run vs checkpoint-at-2 resume — statuses,
        # positions and the restart wet_streak bitwise equal on the tail.
        from kelp.tracking import run_single_realization
        from kelp.writers import (
            TrajectoryWriter, load_restart, load_trajectories,
        )
        # beach (dry), rewet, dry-again (streak reset), rewet, rewet — drives
        # both the beach and resuspension RNG components + the streak counter.
        eta = [-0.9, 0.0, -0.9, 0.0, 0.0, 0.0]

        # --- Continuous 5 steps ---
        mesh, ds, reader, mode, numerics, init = self._setup(eta)
        out_c = str(tmp_path / "cont.nc")
        wc = TrajectoryWriter(out_c, init.n_particles); wc.open()
        try:
            run_single_realization(
                mesh, ds, reader, init, mode, numerics,
                start_time_idx=0, n_steps=5, writer=wc, checkpoint_freq=1)
        finally:
            wc.close()
        traj_c = load_trajectories(out_c)

        # --- Split: 2 steps, checkpoint, resume 3 ---
        mesh, ds, reader, mode, numerics, init = self._setup(eta)
        out_a = str(tmp_path / "a.nc")
        wa = TrajectoryWriter(out_a, init.n_particles); wa.open()
        try:
            run_single_realization(
                mesh, ds, reader, init, mode, numerics,
                start_time_idx=0, n_steps=2, writer=wa, checkpoint_freq=1)
        finally:
            wa.close()
        rs = load_restart(out_a)
        assert rs["wet_streak"] is not None       # beaching → wet_streak stored

        out_b = str(tmp_path / "b.nc")
        wb = TrajectoryWriter(out_b, init.n_particles); wb.open()
        try:
            run_single_realization(
                mesh, ds, reader, rs["state"], mode, numerics,
                start_time_idx=rs["file_idx"] + 1, n_steps=3,
                start_step_i=rs["step_i"] + 1,
                start_wet_streak=rs["wet_streak"],
                writer=wb, checkpoint_freq=1)
        finally:
            wb.close()
        traj_b = load_trajectories(out_b)
        rs_b = load_restart(out_b)
        rs_c = load_restart(out_c)

        # Continuous tail (obs 2,3,4) == resumed run (obs 0,1,2), bitwise.
        np.testing.assert_array_equal(
            traj_c["status"][:, 2:], traj_b["status"][:, :])
        np.testing.assert_array_equal(
            traj_c["lon"][:, 2:], traj_b["lon"][:, :])
        np.testing.assert_array_equal(
            traj_c["lat"][:, 2:], traj_b["lat"][:, :])
        # Final wet_streak (from the last checkpoint) matches bitwise.
        np.testing.assert_array_equal(rs_c["wet_streak"], rs_b["wet_streak"])

    def test_multi_realization_rng_decorrelated(self):
        # two realization ids, stochastic criterion, same seed: each rerun
        # reproduces, and the realizations are not identical streams.
        from kelp.tracking import run_single_realization
        # Strong stochastic beaching inside the zone so the RNG actually
        # changes outcomes: shallow mesh (H=0.5 < zone 0.6), p~0.5/step.
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5),
                          depth=0.5)
        # 2000 particles → the per-particle counter stream is observable.
        P = 2000
        ds = _fesom2_flat_ds(mesh.n_nodes)
        reader = get_reader("fesom2")
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False)
        b = BeachingConfig(
            enabled=True, criteria=["stochastic"], depth_min=0.3,
            t_beach=3600.0 / np.log(2.0), resuspension="never", seed=555)
        numerics = NumericsConfig(dt=3600.0, integrator="euler", beaching=b)
        init = make_state(np.full(P, 2.0), np.full(P, 2.0))

        def _final_status(real_id):
            rec = run_single_realization(
                mesh, ds, reader, init, mode, numerics,
                start_time_idx=0, n_steps=1, realization_id=real_id)
            return rec.to_arrays()["status"][-1].copy()

        s0a = _final_status(0)
        s0b = _final_status(0)         # identical rerun
        s1 = _final_status(1)          # different realization
        np.testing.assert_array_equal(s0a, s0b)        # reproducible
        # ~half beach in each, but the SETS differ between realizations.
        assert (s0a == ParticleStatus.BEACHED).mean() > 0.3
        assert not np.array_equal(s0a, s1)             # decorrelated

    def test_3d_smoke(self):
        # 3-D z-level run with the depth criterion: no crash, legal statuses.
        from kelp.tracking import run_single_realization
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5),
                          depth=50.0)
        mesh.sigma = np.array([0.0, 10.0, 20.0, 40.0])   # z-level interfaces
        K = mesh.sigma.size
        n_time = 4
        ds = xr.Dataset(
            {"u": (("time", "nz1", "elem"),
                   np.zeros((n_time, K - 1, mesh.n_elements))),
             "v": (("time", "nz1", "elem"),
                   np.zeros((n_time, K - 1, mesh.n_elements))),
             "ssh": (("time", "nod2"), np.zeros((n_time, mesh.n_nodes))),
             "w": (("time", "nz", "nod2"),
                   np.zeros((n_time, K, mesh.n_nodes)))},
            coords={"time": np.arange(n_time)})
        reader = get_reader("fesom2")
        mode = ModeConfig(Fdrift2D=False, Fdrift3D=True)
        b = BeachingConfig(enabled=True, criteria=["depth"], depth_min=0.3)
        numerics = NumericsConfig(dt=3600.0, integrator="euler", beaching=b)
        init = make_state(np.array([2.0]), np.array([2.0]), np.array([12.0]))
        rec = run_single_realization(
            mesh, ds, reader, init, mode, numerics,
            start_time_idx=0, n_steps=3)
        s = rec.to_arrays()["status"]
        legal = {int(ParticleStatus.ALIVE), int(ParticleStatus.BEACHED),
                 int(ParticleStatus.OUTSIDE)}
        assert set(np.unique(s).tolist()) <= legal

    def test_3d_sigma_smoke(self):
        # Spec item 12 (sigma half): FESOM-C SIGMA mesh + 3-D run + depth
        # criterion -> no crash, legal statuses.  Pure smoke: depth=10-12
        # everywhere >> depth_min, so nothing actually beaches (the point is
        # that the hook is vertical-coordinate agnostic and the sigma 3-D
        # path runs cleanly with beaching on).  Reuses the loadable 11x11
        # sigma e2e fixture from the sampling suite (house cross-file pattern).
        import os
        import tempfile
        from types import SimpleNamespace
        from tests.test_sampling import _make_fesomc_sigma_e2e
        from kelp.tracking import run_single_realization

        tmpdir = tempfile.mkdtemp()
        nc = os.path.join(tmpdir, "fesomc_sigma_beach_smoke.nc")
        _make_fesomc_sigma_e2e(nc, eta_val=2.0, depth_val=10.0)

        reader = get_reader("fesomc")
        mesh = reader.load_mesh(SimpleNamespace(meshpath=nc))
        ds = reader.open_dataset(nc)
        mode = ModeConfig(Fdrift2D=False, Fdrift3D=True)
        b = BeachingConfig(enabled=True, criteria=["depth"], depth_min=0.3)
        numerics = NumericsConfig(dt=3600.0, integrator="euler", beaching=b)
        # (5, 5) = deep interior node of the 11x11 grid; z within the column.
        init = make_state(np.array([5.0]), np.array([5.0]), np.array([6.0]))
        rec = run_single_realization(
            mesh, ds, reader, init, mode, numerics,
            start_time_idx=0, n_steps=2)
        ds.close()
        s = rec.to_arrays()["status"]
        legal = {int(ParticleStatus.ALIVE), int(ParticleStatus.BEACHED),
                 int(ParticleStatus.OUTSIDE)}
        assert set(np.unique(s).tolist()) <= legal


class TestBeachingHookAnchorRefresh:
    """B1: the on_failure remap must work for FIRST-step failures.

    A particle killed by the integrator on its very first step (or on the
    unfused 3D-RK / sigma and diffusion combined/split-kick failure paths)
    still carries the fresh-state ``nearest_node == -1``.  Without the
    defensive anchor refresh in ``_apply_beaching_hook`` its H_raw is inf
    and the containment subset skips it, so the coastal-CONTEXT gate can
    never remap it OUTSIDE->BEACHED (silent on_failure degradation).  These
    tests call the hook DIRECTLY (it is module-level) with a frozen,
    already-reverted in-domain anchor whose nearest_node is -1.
    """

    def _hook(self, mesh, *, depth, status, prev_status, nn,
              crit=("on_failure",), need_topo=False, zone=0.6,
              use_eta=False, elem_coastal=None, lon=2.0, lat=2.0):
        from kelp.tracking import _apply_beaching_hook
        from kelp.numerics.beaching import BeachRunConfig
        mesh.depth[:] = float(depth)
        st = make_state(np.array([float(lon)]), np.array([float(lat)]))
        st = st.replace(
            status=np.array([status], dtype=st.status.dtype),
            nearest_node=np.array([nn], dtype=st.nearest_node.dtype))
        prev = np.array([prev_status], dtype=st.status.dtype)
        cfg = BeachingConfig(enabled=True, criteria=list(crit),
                             depth_min=0.3, use_eta=use_eta)
        run_cfg = BeachRunConfig(
            cfg=cfg, crit=frozenset(crit), need_topo=need_topo,
            elem_coastal=elem_coastal, seed=42, zone=zone)
        wet0 = np.zeros(1, dtype=np.int32)
        out, _ = _apply_beaching_hook(
            st, prev, mesh, _vel(mesh), run_cfg, wet0,
            real_id=0, step_i=0, dt=3600.0)
        return out

    def test_first_step_failure_in_shallow_context_beaches(self):
        # Shallow column (H=0.5 < zone 0.6): the refreshed anchor's H_raw
        # is < zone, so the context gate fires -> BEACHED.  With nn left at
        # -1 (no refresh) H_raw would be inf and this would stay OUTSIDE.
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5))
        out = self._hook(
            mesh, depth=0.5,
            status=int(ParticleStatus.OUTSIDE),
            prev_status=int(ParticleStatus.ALIVE), nn=-1)
        assert out.nearest_node[0] >= 0                 # anchor was refreshed
        assert out.status[0] == ParticleStatus.BEACHED

    def test_first_step_failure_in_deep_context_stays_outside(self):
        # Control: deep column (H=50 >= zone) -> open-boundary exit, the
        # context gate must NOT remap; particle stays OUTSIDE.
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5))
        out = self._hook(
            mesh, depth=50.0,
            status=int(ParticleStatus.OUTSIDE),
            prev_status=int(ParticleStatus.ALIVE), nn=-1)
        assert out.status[0] == ParticleStatus.OUTSIDE

    def test_topology_context_remaps_first_step_failure(self):
        # need_topo path: the refreshed anchor also feeds find_containing_
        # elements, so a deep-but-coastal first-step exit beaches via the
        # element-coastal branch of the context gate.  The corner node (0,0)
        # of this tiny grid sits in a coastal element -> BEACHED even at
        # H=50 (deep), which the zone-OR alone would not catch.  Pins that
        # the refreshed anchor reaches the containment subset, not just H_raw.
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5))
        from kelp.core.mesh import build_boundary_info
        build_boundary_info(mesh)
        out = self._hook(
            mesh, depth=50.0, zone=0.6, need_topo=True,
            elem_coastal=mesh._elem_is_coastal,
            status=int(ParticleStatus.OUTSIDE),
            prev_status=int(ParticleStatus.ALIVE), nn=-1,
            lon=0.0, lat=0.0)
        assert out.nearest_node[0] >= 0
        assert out.status[0] == ParticleStatus.BEACHED


class TestBeachingVarEtaFailFast:
    """M3: the use_eta fail-fast must honor a configured mode.var_eta.

    A user storing eta under a non-default name (e.g. ``elevation``) and
    declaring it via ``mode.var_eta`` must NOT trip the startup guard — and
    the value must actually flow through (the FESOM2 reader now honors
    var_eta too, mirroring the FESOM-C reader).  Both guard sites are
    covered: ``run_single_realization`` (direct) and ``run_tracking``
    (config).  The negatives remain: an unconfigured custom name still raises.
    """

    def test_run_single_realization_accepts_var_eta(self):
        # Same tidal beach/resuspend scenario as
        # TestTrackingBeaching::test_tidal_cycle_beach_then_resuspend, but
        # eta is stored as ``elevation`` and declared via mode.var_eta.  The
        # guard must accept it AND the reader must read it: the SAME status
        # sequence [A, B, B, A, A] proves the elevation field drove beaching
        # (had var_eta been ignored, eta would be zeros -> never beaches).
        from kelp.tracking import run_single_realization
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5),
                          depth=1.0)
        eta_series = [0.0, -0.9, -0.9, 0.0, 0.0, 0.0]
        ds = _fesom2_flat_ds(mesh.n_nodes, eta_series=eta_series).rename(
            {"ssh": "elevation"})
        ds["u"][:] = 0.2
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, var_eta="elevation")
        b = BeachingConfig(enabled=True, criteria=["depth"], depth_min=0.3,
                           use_eta=True, resuspension="auto", min_wet_steps=1)
        numerics = NumericsConfig(dt=3600.0, integrator="euler", beaching=b)
        init = make_state(np.array([2.0]), np.array([2.0]))
        reader = get_reader("fesom2")
        rec = run_single_realization(           # must NOT raise
            mesh, ds, reader, init, mode, numerics,
            start_time_idx=0, n_steps=5)
        s = list(rec.to_arrays()["status"][:, 0])
        A, B = ParticleStatus.ALIVE, ParticleStatus.BEACHED
        assert s == [A, B, B, A, A]             # elevation read as eta

    def test_run_single_realization_missing_var_eta_still_raises(self):
        # Negative: var_eta names a var ABSENT from the dataset and no
        # ssh/eta/zos present -> the guard must still raise (the configured
        # name doesn't magically appear).
        from kelp.tracking import run_single_realization
        mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5))
        ds = _fesom2_flat_ds(mesh.n_nodes).drop_vars(
            ["ssh", "eta", "zos"], errors="ignore")
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, var_eta="elevation")
        b = BeachingConfig(enabled=True, criteria=["depth"],
                           depth_min=0.3, use_eta=True)
        numerics = NumericsConfig(dt=3600.0, integrator="euler", beaching=b)
        init = make_state(np.array([2.0]), np.array([2.0]))
        reader = get_reader("fesom2")
        with pytest.raises(ValueError, match="ssh"):
            run_single_realization(
                mesh, ds, reader, init, mode, numerics,
                start_time_idx=0, n_steps=2)

    def test_run_tracking_accepts_var_eta(self, tmp_path):
        # run_tracking startup guard: eta stored as ``elevation`` + mode
        # var_eta declared -> no raise, run completes, output written.
        from kelp.tracking import run_tracking
        beaching_yaml = (
            "  beaching:\n"
            "    enabled: true\n"
            "    criteria: [depth]\n"
            "    depth_min: 0.3\n"
            "    use_eta: true\n")
        cfg, out = TestBeachingRunTracking()._config(
            tmp_path, beaching_yaml, n_steps=3, u=0.0)
        # rename ssh -> elevation in the written velocity file (load fully
        # inside the context, write back AFTER the handle is released — the
        # in-place rewrite trips a PermissionError otherwise).
        nc = tmp_path / "vel.nc"
        with xr.open_dataset(nc) as _d:
            ds_elev = _d.rename({"ssh": "elevation"}).load()
        ds_elev.to_netcdf(nc)
        # inject mode.var_eta: elevation into the config's mode section
        cfg.mode.var_eta = "elevation"
        run_tracking(cfg, output_path=str(out))     # must NOT raise
        assert out.exists()


def test_stochastic_beaching_keys_a_shard_by_its_global_index():
    """A shard (pid_offset = its first particle's global index) draws the
    stranding numbers its particles draw in the unsharded run (audit
    2026-10-04, option (a))."""
    mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5))
    mesh.depth[:] = 0.5                       # inside the 0.6 m zone everywhere
    rng = np.random.default_rng(5)
    lon = rng.uniform(0.2, 3.8, 40)
    lat = rng.uniform(0.2, 3.8, 40)
    from kelp.core.mesh import find_nearest_nodes
    st = make_state(lon, lat)
    st = st.replace(nearest_node=find_nearest_nodes(mesh, st.lon, st.lat))
    args = _args(use_depth=False, use_stochastic=True, t_beach=3600.0,
                 dt=3600.0, seed=9, step_i=3)
    whole, _, _ = evaluate_beaching(st, mesh, _vel(mesh), args)
    assert 0 < whole.sum() < 40               # the draws decide
    for lo, hi in ((0, 17), (17, 40)):
        sub = st.replace(lon=st.lon[lo:hi], lat=st.lat[lo:hi], z=st.z[lo:hi],
                         status=st.status[lo:hi],
                         nearest_node=st.nearest_node[lo:hi])
        part, _, _ = evaluate_beaching(sub, mesh, _vel(mesh),
                                       args._replace(pid_offset=lo))
        np.testing.assert_array_equal(part, whole[lo:hi])


def test_stochastic_resuspension_keys_a_shard_by_its_global_index():
    mesh = _grid_mesh(np.linspace(0, 4, 5), np.linspace(0, 4, 5))
    mesh.depth[:] = 5.0                        # deep: rewet, the draw decides
    rng = np.random.default_rng(6)
    st = make_state(rng.uniform(0.2, 3.8, 40), rng.uniform(0.2, 3.8, 40))
    from kelp.core.mesh import find_nearest_nodes
    st = st.replace(nearest_node=find_nearest_nodes(mesh, st.lon, st.lat),
                    status=np.full(40, ParticleStatus.BEACHED,
                                   dtype=st.status.dtype))
    args = _args(resuspension=RESUSP_STOCH, t_resuspend=3600.0, dt=3600.0,
                 seed=13, step_i=4)
    _, whole, _ = evaluate_beaching(st, mesh, _vel(mesh), args,
                                    wet_streak=np.zeros(40, np.int32))
    assert 0 < whole.sum() < 40
    lo = 15
    sub = st.replace(lon=st.lon[lo:], lat=st.lat[lo:], z=st.z[lo:],
                     status=st.status[lo:], nearest_node=st.nearest_node[lo:])
    _, part, _ = evaluate_beaching(sub, mesh, _vel(mesh),
                                   args._replace(pid_offset=lo),
                                   wet_streak=np.zeros(40 - lo, np.int32))
    np.testing.assert_array_equal(part, whole[lo:])
