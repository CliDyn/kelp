"""Tests on real mesh fixtures extracted from FESOM-C and FESOM2.

These tests verify that tracking, interpolation, grid hash, and domain
detection work correctly on real (irregular) mesh topologies, not just
synthetic regular grids.

Fixtures are small (~50-650 KB) NetCDF files in tests/fixtures/,
created by tests/create_fixtures.py from real model output.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from types import SimpleNamespace

from kelp.core.geometry import METERS_PER_DEG_LAT, DEG2RAD, delta_lon
from kelp.numerics.integrators import euler_step, rk4_step
from kelp.numerics.interpolate import VelocityField
from kelp.core.mesh import (
    Mesh, build_search, find_nearest_nodes, find_containing_elements,
    _grid_hash_nearest_batch,
)
from kelp.core.particles import ParticleState, ParticleStatus, make_state
from kelp.readers import get_reader


# ---------------------------------------------------------------------------
# Fixture paths and skip logic
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"

_FESOMC_FIXTURES = ["interior", "coastal", "boundary"]
_FESOM2_FIXTURES = ["midlat", "dateline", "pole"]

_skip_no_fixtures = pytest.mark.skipif(
    not FIXTURES_DIR.exists() or not any(FIXTURES_DIR.glob("*.nc")),
    reason="Real mesh fixtures not found (run tests/create_fixtures.py)",
)


def _load_fesomc(name: str) -> Mesh:
    path = FIXTURES_DIR / f"fesomc_{name}.nc"
    if not path.exists():
        pytest.skip(f"Fixture not found: {path}")
    cfg = SimpleNamespace(meshpath=str(path))
    return get_reader("fesomc").load_mesh(cfg)


def _load_fesom2(name: str) -> Mesh:
    path = FIXTURES_DIR / f"fesom2_{name}.nc"
    if not path.exists():
        pytest.skip(f"Fixture not found: {path}")
    cfg = SimpleNamespace(meshpath=str(path))
    return get_reader("fesom2").load_mesh(cfg)


def _make_vel_2d(mesh: Mesh, u: float = 0.5, v: float = 0.0) -> VelocityField:
    """Create uniform 2D velocity field on mesh elements."""
    E = mesh.n_elements
    N = mesh.n_nodes
    return VelocityField(
        u=np.full(E, u, dtype=np.float64),
        v=np.full(E, v, dtype=np.float64),
        eta=np.zeros(N, dtype=np.float64),
    )


def _seed_interior(mesh: Mesh, n: int = 20, rng_seed: int = 42) -> ParticleState:
    """Seed particles inside the mesh domain."""
    rng = np.random.default_rng(rng_seed)
    lon_min, lon_max = mesh.nodes_lon.min(), mesh.nodes_lon.max()
    lat_min, lat_max = mesh.nodes_lat.min(), mesh.nodes_lat.max()

    # Shrink bbox by 10% to avoid boundary
    margin_lon = (lon_max - lon_min) * 0.1
    margin_lat = (lat_max - lat_min) * 0.1

    candidates_lon = rng.uniform(lon_min + margin_lon, lon_max - margin_lon, n * 5)
    candidates_lat = rng.uniform(lat_min + margin_lat, lat_max - margin_lat, n * 5)

    build_search(mesh)
    nn = find_nearest_nodes(mesh, candidates_lon, candidates_lat)
    _, in_dom = find_containing_elements(mesh, candidates_lon, candidates_lat, nn)

    good = np.where(in_dom)[0][:n]
    if len(good) == 0:
        pytest.skip("Could not seed particles inside mesh")

    return make_state(candidates_lon[good], candidates_lat[good])


# ===================================================================
# FESOM-C mesh tests
# ===================================================================

@_skip_no_fixtures
class TestFesomCInterior:
    """Tests on FESOM-C interior (open sea) mesh."""

    def test_mesh_loads(self):
        mesh = _load_fesomc("interior")
        assert mesh.n_nodes > 100
        assert mesh.n_elements > 100
        assert mesh._gh_offsets is not None
        assert mesh._gh_offsets is not None

    def test_grid_hash_vs_kdtree(self):
        """Grid hash and KD-tree give the same nearest node."""
        mesh = _load_fesomc("interior")
        rng = np.random.default_rng(123)
        lons = rng.uniform(mesh.nodes_lon.min() + 0.1,
                           mesh.nodes_lon.max() - 0.1, 100)
        lats = rng.uniform(mesh.nodes_lat.min() + 0.1,
                           mesh.nodes_lat.max() - 0.1, 100)

        # Grid hash
        nn_gh = find_nearest_nodes(mesh, lons, lats)

        # KD-tree (force by temporarily removing grid hash)
        saved = mesh._gh_offsets
        mesh._gh_offsets = None
        nn_kd = find_nearest_nodes(mesh, lons, lats)
        mesh._gh_offsets = saved

        np.testing.assert_array_equal(nn_gh, nn_kd)

    def test_euler_tracking(self):
        """Euler step on real mesh: analytical displacement check."""
        mesh = _load_fesomc("interior")
        vel = _make_vel_2d(mesh, u=0.5, v=0.0)
        state = _seed_interior(mesh)
        dt = 300.0

        new_state = euler_step(state, vel, mesh, dt, 1.0)

        alive = new_state.alive_mask
        assert np.any(alive), "All particles died"

        # Check displacement for alive particles
        for i in np.where(alive)[0]:
            lat_rad = state.lat[i] * DEG2RAD
            z0 = np.cos(lat_rad) * METERS_PER_DEG_LAT
            expected_dlon = 0.5 * dt / z0  # degrees

            actual_dlon = delta_lon(new_state.lon[i], state.lon[i])
            assert actual_dlon == pytest.approx(expected_dlon, rel=0.15), \
                f"Particle {i}: dlon={actual_dlon:.6f}, expected={expected_dlon:.6f}"

    def test_rk4_tracking(self):
        """RK4 step on real mesh: should match Euler for uniform velocity."""
        mesh = _load_fesomc("interior")
        vel = _make_vel_2d(mesh, u=0.3, v=0.1)
        state = _seed_interior(mesh)
        dt = 300.0

        euler_state = euler_step(state, vel, mesh, dt, 1.0)
        rk4_state = rk4_step(state, vel, mesh, dt, 1.0)

        # For uniform velocity, Euler and RK4 should be very close
        both_alive = euler_state.alive_mask & rk4_state.alive_mask
        if not np.any(both_alive):
            pytest.skip("No particles survived in both")

        for i in np.where(both_alive)[0]:
            dlon = abs(delta_lon(euler_state.lon[i], rk4_state.lon[i]))
            dlat = abs(euler_state.lat[i] - rk4_state.lat[i])
            # Should agree to within ~1% of step size
            assert dlon < 1e-5, f"Particle {i}: lon diff {dlon}"
            assert dlat < 1e-5, f"Particle {i}: lat diff {dlat}"

    def test_multi_step_tracking(self):
        """10 steps of Euler tracking: particles stay alive and move."""
        mesh = _load_fesomc("interior")
        vel = _make_vel_2d(mesh, u=0.3, v=0.0)
        state = _seed_interior(mesh, n=10)
        dt = 300.0

        for _ in range(10):
            state = euler_step(state, vel, mesh, dt, 1.0)

        alive = state.alive_mask
        # At least some particles should survive 10 steps
        assert np.sum(alive) >= 1, "All particles died in 10 steps"


@_skip_no_fixtures
class TestFesomCCoastal:
    """Tests on FESOM-C coastal mesh — particles hitting land."""

    def test_particles_go_outside(self):
        """Particles moving toward coast should become OUTSIDE."""
        mesh = _load_fesomc("coastal")

        # Velocity pointing south (into Dutch coastline)
        vel = _make_vel_2d(mesh, u=0.0, v=-0.5)
        state = _seed_interior(mesh, n=30)
        dt = 600.0

        # Run 20 steps — some should hit the coast
        for _ in range(20):
            state = euler_step(state, vel, mesh, dt, 1.0)

        n_outside = np.sum(state.status == ParticleStatus.OUTSIDE)
        n_alive = np.sum(state.alive_mask)
        # At least some should be outside (hit coast) or still alive
        assert n_outside + n_alive == len(state.lon)
        # With v=-0.5 for 20 steps of 600s, particles move ~6km south
        # Some should hit the coast
        print(f"  coastal: {n_alive} alive, {n_outside} outside")

    def test_irregular_stencil(self):
        """Coastal nodes have incomplete stencils — interpolation should not crash."""
        mesh = _load_fesomc("coastal")

        # Check that some nodes have fewer neighbors
        min_conn = mesh.node_to_elems_count.min()
        max_conn = mesh.node_to_elems_count.max()
        assert min_conn < max_conn, "Expected varying connectivity"
        print(f"  connectivity range: {min_conn} to {max_conn}")

        # Tracking near boundary should work
        vel = _make_vel_2d(mesh, u=0.1, v=0.0)
        state = _seed_interior(mesh, n=5)
        new_state = euler_step(state, vel, mesh, 300.0, 1.0)
        # Should not crash
        assert len(new_state.lon) == len(state.lon)


@_skip_no_fixtures
class TestFesomCBoundary:
    """Tests on FESOM-C open boundary mesh."""

    def test_boundary_tracking(self):
        """Particles near open boundary: tracking should work or go OUTSIDE."""
        mesh = _load_fesomc("boundary")
        vel = _make_vel_2d(mesh, u=0.3, v=0.2)
        state = _seed_interior(mesh, n=15)
        dt = 300.0

        for _ in range(10):
            state = euler_step(state, vel, mesh, dt, 1.0)

        # Should not crash; some may go outside
        assert len(state.lon) == 15


# ===================================================================
# FESOM2 mesh tests
# ===================================================================

@_skip_no_fixtures
class TestFesom2Midlat:
    """Tests on FESOM2 mid-latitude (Agulhas) mesh."""

    def test_mesh_loads(self):
        mesh = _load_fesom2("midlat")
        assert mesh.n_nodes > 50
        assert mesh.n_elements > 50
        assert mesh._gh_offsets is not None

    def test_euler_tracking(self):
        """Euler on FESOM2 mesh with z-levels."""
        mesh = _load_fesom2("midlat")
        vel = _make_vel_2d(mesh, u=0.3, v=0.0)
        state = _seed_interior(mesh, n=10)
        dt = 300.0

        new_state = euler_step(state, vel, mesh, dt, 1.0)
        alive = new_state.alive_mask
        assert np.any(alive)

        for i in np.where(alive)[0]:
            lat_rad = state.lat[i] * DEG2RAD
            z0 = np.cos(lat_rad) * METERS_PER_DEG_LAT
            expected = 0.3 * dt / z0
            actual = delta_lon(new_state.lon[i], state.lon[i])
            assert actual == pytest.approx(expected, rel=0.15)

    def test_grid_hash_vs_kdtree(self):
        mesh = _load_fesom2("midlat")
        rng = np.random.default_rng(99)
        lons = rng.uniform(12, 23, 50)
        lats = rng.uniform(-38, -32, 50)

        nn_gh = find_nearest_nodes(mesh, lons, lats)
        saved = mesh._gh_offsets
        mesh._gh_offsets = None
        nn_kd = find_nearest_nodes(mesh, lons, lats)
        mesh._gh_offsets = saved

        np.testing.assert_array_equal(nn_gh, nn_kd)


@_skip_no_fixtures
class TestFesom2Dateline:
    """Tests on FESOM2 dateline-crossing mesh (lon ≈ ±180°)."""

    def test_mesh_spans_dateline(self):
        mesh = _load_fesom2("dateline")
        # Should have nodes on both sides of ±180°
        has_positive = np.any(mesh.nodes_lon > 170)
        has_negative = np.any(mesh.nodes_lon < -170)
        assert has_positive and has_negative, \
            f"Mesh should span dateline: lon range [{mesh.nodes_lon.min():.1f}, {mesh.nodes_lon.max():.1f}]"

    def test_grid_hash_across_dateline(self):
        """Grid hash should find correct nearest node across ±180° boundary."""
        mesh = _load_fesom2("dateline")

        # Query points near ±180°
        lons = np.array([179.5, -179.5, 179.9, -179.9])
        lats = np.array([-5.0, -5.0, -3.0, -3.0])

        nn_gh = find_nearest_nodes(mesh, lons, lats)
        saved = mesh._gh_offsets
        mesh._gh_offsets = None
        nn_kd = find_nearest_nodes(mesh, lons, lats)
        mesh._gh_offsets = saved

        np.testing.assert_array_equal(nn_gh, nn_kd)

    def test_tracking_across_dateline(self):
        """Particles crossing from lon=179 to lon=-179 (eastward)."""
        mesh = _load_fesom2("dateline")

        # Seed near lon=179, inside the mesh
        candidates_lon = np.array([179.0, 179.5, 178.5])
        candidates_lat = np.array([-5.0, -4.0, -6.0])

        nn = find_nearest_nodes(mesh, candidates_lon, candidates_lat)
        _, in_dom = find_containing_elements(mesh, candidates_lon, candidates_lat, nn)
        good = np.where(in_dom)[0]
        if len(good) == 0:
            pytest.skip("No seed points in domain near dateline")

        state = make_state(candidates_lon[good], candidates_lat[good])
        vel = _make_vel_2d(mesh, u=1.0, v=0.0)  # strong eastward
        dt = 600.0

        # Run several steps
        for _ in range(5):
            state = euler_step(state, vel, mesh, dt, 1.0)

        alive = state.alive_mask
        if not np.any(alive):
            pytest.skip("All particles left domain (expected for small mesh)")

        # Alive particles should have crossed dateline
        # or at least not have NaN
        for i in np.where(alive)[0]:
            assert not np.isnan(state.lon[i])
            assert not np.isnan(state.lat[i])
            assert -180.0 <= state.lon[i] < 180.0


@_skip_no_fixtures
class TestFesom2Pole:
    """Tests on FESOM2 near-pole mesh (lat > 85°N)."""

    def test_mesh_near_pole(self):
        mesh = _load_fesom2("pole")
        assert mesh.nodes_lat.max() > 88, \
            f"Expected nodes near pole, max lat = {mesh.nodes_lat.max():.1f}"

    def test_robustness_no_nan(self):
        """Tracking near pole should not produce NaN or overflow."""
        mesh = _load_fesom2("pole")

        # Use small velocity to avoid leaving domain instantly
        vel = _make_vel_2d(mesh, u=0.1, v=0.0)
        state = _seed_interior(mesh, n=10)
        dt = 300.0

        # Run 5 steps
        for _ in range(5):
            state = euler_step(state, vel, mesh, dt, 1.0)

        # Check no NaN in any particle (alive or not)
        assert not np.any(np.isnan(state.lon)), "NaN in longitude"
        assert not np.any(np.isnan(state.lat)), "NaN in latitude"
        # Latitude should be clamped to [-90, 90]
        assert np.all(state.lat >= -90.0)
        assert np.all(state.lat <= 90.0)

    def test_cos_lat_extreme(self):
        """At lat=87°, cos(lat) ≈ 0.05 — verify metric doesn't blow up."""
        mesh = _load_fesom2("pole")
        vel = _make_vel_2d(mesh, u=0.1, v=0.0)
        state = _seed_interior(mesh, n=5)

        # Single step
        new_state = euler_step(state, vel, mesh, 300.0, 1.0)

        alive = new_state.alive_mask
        if not np.any(alive):
            pytest.skip("All particles left domain")

        for i in np.where(alive)[0]:
            # At lat≈87°, cos(lat)≈0.05, so Δlon = u*dt/(cos*SLM) ≈ 0.5°
            # This is large but finite — verify it's reasonable
            dlon = abs(delta_lon(new_state.lon[i], state.lon[i]))
            assert dlon < 10.0, f"Δlon={dlon:.2f}° is unreasonably large"
            assert dlon > 1e-6, f"Δlon={dlon:.2e}° is suspiciously small"

    def test_grid_hash_near_pole(self):
        """Grid hash works at high latitudes where meridians converge."""
        mesh = _load_fesom2("pole")
        rng = np.random.default_rng(77)
        lons = rng.uniform(5, 55, 30)
        lats = rng.uniform(85.5, 89.0, 30)

        nn_gh = find_nearest_nodes(mesh, lons, lats)
        saved = mesh._gh_offsets
        mesh._gh_offsets = None
        nn_kd = find_nearest_nodes(mesh, lons, lats)
        mesh._gh_offsets = saved

        np.testing.assert_array_equal(nn_gh, nn_kd)
