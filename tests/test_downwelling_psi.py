"""Streamfunction conservation test for 3D downwelling cell.

The analytical velocity field satisfies continuity (∂u/∂x + ∂W/∂z = 0),
so a streamfunction ψ exists.  In steady flow, pathlines = streamlines,
hence ψ must be conserved along particle trajectories.

We test that ψ drift stays below a threshold for a short simulation
(before particles hit bottom clamping, which breaks the invariant).
"""
from __future__ import annotations

import numpy as np
import pytest

from kelp.numerics.integrators import euler_step
from kelp.numerics.interpolate import VelocityField
from kelp.core.mesh import Mesh, build_node_to_elems, build_search
from kelp.core.particles import make_state, ParticleStatus
from kelp.core.geometry import METERS_PER_DEG_LAT, DEG2RAD


# ── Physical parameters ──────────────────────────────────────────────
U0 = 0.3
LAMBDA = 8_000.0
H_MAX = 200.0
CENTER_LON = 10.0
CENTER_LAT = 50.0
COS_LAT = np.cos(CENTER_LAT * DEG2RAD)


def _streamfunction(lon, z, coast_lon):
    """ψ(x, z) = -(U0·H/π) · (1 - exp(-x/λ)) · sin(πz/H)."""
    x = (lon - coast_lon) * COS_LAT * METERS_PER_DEG_LAT
    x = np.maximum(x, 0.0)
    return -(U0 * H_MAX / np.pi) * (1 - np.exp(-x / LAMBDA)) * np.sin(
        np.pi * z / H_MAX
    )


def _build_mesh():
    """Build high-resolution mesh with flat bottom (no clamping interference).

    161×81 = 13041 nodes, Δx ≈ 800 m — fine enough for spatial error
    to be small relative to method differences.
    """
    span_x = 0.90
    span_y = 0.90
    nx, ny = 161, 81

    xs = np.linspace(CENTER_LON - span_x, CENTER_LON + span_x, nx)
    ys = np.linspace(CENTER_LAT - span_y, CENTER_LAT + span_y, ny)
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel().astype(np.float64)
    lat = gy.ravel().astype(np.float64)
    n_nodes = lon.shape[0]

    # Flat bottom — no clamping interference
    depth = np.full(n_nodes, H_MAX, dtype=np.float64)

    # Jitter interior nodes (10%)
    rng = np.random.default_rng(42)
    dx = (xs[1] - xs[0]) * 0.10
    dy = (ys[1] - ys[0]) * 0.10
    for i in range(n_nodes):
        ix, iy = i % nx, i // nx
        if 0 < ix < nx - 1 and 0 < iy < ny - 1:
            lon[i] += rng.uniform(-dx, dx)
            lat[i] += rng.uniform(-dy, dy)

    elems = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            elems.append([n0, n0 + 1, n0 + nx])
            elems.append([n0 + 1, n0 + nx + 1, n0 + nx])
    elem_nodes = np.array(elems, dtype=np.int32)

    lon_c = np.array([lon[e].mean() for e in elem_nodes], dtype=np.float64)
    lat_c = np.array([lat[e].mean() for e in elem_nodes], dtype=np.float64)
    n2e, n2e_count = build_node_to_elems(n_nodes, elem_nodes)

    z_levels = np.linspace(0.0, H_MAX, 21)
    mesh = Mesh(
        name="test_downwelling",
        nodes_lon=lon, nodes_lat=lat, depth=depth,
        elem_nodes=elem_nodes,
        elem_center_lon=lon_c, elem_center_lat=lat_c,
        node_to_elems=n2e, node_to_elems_count=n2e_count,
        sigma=z_levels,
    )
    build_search(mesh)

    coast_lon = CENTER_LON - span_x * 0.7
    return mesh, coast_lon


def _build_velocity(mesh, coast_lon):
    """Build continuity-consistent velocity field."""
    z_levels = np.abs(mesh.sigma)
    K = z_levels.shape[0]
    ne = mesh.n_elements
    nn = mesh.n_nodes

    dist_elem = np.maximum(
        (mesh.elem_center_lon - coast_lon) * COS_LAT * METERS_PER_DEG_LAT, 0.0
    )
    dist_node = np.maximum(
        (mesh.nodes_lon - coast_lon) * COS_LAT * METERS_PER_DEG_LAT, 0.0
    )

    u_3d = np.zeros((K - 1, ne), dtype=np.float64)
    for k in range(K - 1):
        z_mid = 0.5 * (z_levels[k] + z_levels[k + 1])
        u_3d[k, :] = -U0 * np.cos(np.pi * z_mid / H_MAX) * (
            1 - np.exp(-dist_elem / LAMBDA)
        )

    W0 = U0 * H_MAX / (np.pi * LAMBDA)
    w_3d = np.zeros((K, nn), dtype=np.float64)
    for k in range(K):
        w_3d[k, :] = W0 * np.sin(np.pi * z_levels[k] / H_MAX) * np.exp(
            -dist_node / LAMBDA
        )

    return VelocityField(
        u=u_3d,
        v=np.zeros((K - 1, ne), dtype=np.float64),
        eta=np.zeros(nn, dtype=np.float64),
        w=w_3d,
    )


@pytest.fixture
def downwelling_setup():
    mesh, coast_lon = _build_mesh()
    vel = _build_velocity(mesh, coast_lon)
    # Seed 10 particles offshore, mid-depth (avoid surface where sin(πz/H)≈0)
    lon0 = coast_lon + 15_000 / (COS_LAT * METERS_PER_DEG_LAT)
    lons = np.full(10, lon0, dtype=np.float64)
    lats = np.full(10, CENTER_LAT, dtype=np.float64)
    z0 = np.full(10, 50.0, dtype=np.float64)  # 50m — in the sinking zone
    state = make_state(lons, lats, z0)
    return mesh, vel, state, coast_lon


class TestStreamfunctionConservation:
    """ψ should be approximately conserved over short integrations."""

    def _run_and_check_psi(self, mesh, vel, state, coast_lon, w_interp,
                           dt, n_steps):
        """Run tracking and return relative ψ drift."""
        psi0 = _streamfunction(state.lon, state.z, coast_lon)
        for _ in range(n_steps):
            state = euler_step(state, vel, mesh, dt, time_dir=1.0,
                               w_interp=w_interp)
        psi_f = _streamfunction(state.lon, state.z, coast_lon)

        # Only check alive particles
        alive = state.status == ParticleStatus.ALIVE
        if not np.any(alive):
            pytest.skip("No alive particles")

        drift = np.abs(psi_f[alive] - psi0[alive])
        scale = np.abs(psi0[alive])
        safe_scale = np.where(scale > 1e-12, scale, 1.0)
        rel_drift = drift / safe_scale
        return np.max(rel_drift), state

    @pytest.mark.parametrize("method", ["nearest", "leastsq", "barycentric", "idw"])
    def test_psi_bounded(self, downwelling_setup, method):
        """ψ drift should be bounded for all methods (short run, small dt)."""
        mesh, vel, state, coast_lon = downwelling_setup
        # Short run: 50 steps × dt=60s = 50 min — particles shouldn't hit bottom
        rel_max, final = self._run_and_check_psi(
            mesh, vel, state, coast_lon, method, dt=60.0, n_steps=50
        )
        assert np.all(final.status == ParticleStatus.ALIVE), \
            f"Some particles died with method={method}"
        # On 161×81 mesh (Δx≈800m) with dt=60s, all methods < 0.15%
        assert rel_max < 0.0015, (
            f"method={method}: ψ relative drift {rel_max:.6f} > 0.15%"
        )

    def test_leastsq_better_than_nearest(self, downwelling_setup):
        """LSQ should conserve ψ at least as well as nearest."""
        mesh, vel, state, coast_lon = downwelling_setup
        drift_nn, _ = self._run_and_check_psi(
            mesh, vel, state, coast_lon, "nearest", dt=60.0, n_steps=50
        )
        drift_lsq, _ = self._run_and_check_psi(
            mesh, vel, state, coast_lon, "leastsq", dt=60.0, n_steps=50
        )
        # leastsq should be no worse (allow 10% margin for numerical noise)
        assert drift_lsq <= drift_nn * 1.10, (
            f"leastsq drift {drift_lsq:.6f} > nearest drift {drift_nn:.6f} × 1.10"
        )

    def test_all_particles_stay_alive(self, downwelling_setup):
        """All particles should remain in domain during short tracking."""
        mesh, vel, state, coast_lon = downwelling_setup
        for _ in range(50):
            state = euler_step(state, vel, mesh, 60.0, time_dir=1.0,
                               w_interp="leastsq")
        assert np.all(state.status == ParticleStatus.ALIVE)


# ---------------------------------------------------------------------------
# High-resolution test: 65K nodes, 500 particles, dt=5s
#
# On a smooth analytical field all methods converge to similar results.
# The value of this test is verifying correctness on a realistic-size
# mesh (not catching large method differences — those require sharp
# gradients or irregular real-world meshes).
# ---------------------------------------------------------------------------

def _build_mesh_hires():
    """Build 361×181 = 65K node mesh with 10% jitter, flat bottom."""
    span_x = 0.90
    span_y = 0.90
    nx, ny = 361, 181

    xs = np.linspace(CENTER_LON - span_x, CENTER_LON + span_x, nx)
    ys = np.linspace(CENTER_LAT - span_y, CENTER_LAT + span_y, ny)
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel().astype(np.float64)
    lat = gy.ravel().astype(np.float64)
    n_nodes = lon.shape[0]
    depth = np.full(n_nodes, H_MAX, dtype=np.float64)

    rng = np.random.default_rng(42)
    dx = (xs[1] - xs[0]) * 0.10
    dy = (ys[1] - ys[0]) * 0.10
    for i in range(n_nodes):
        ix, iy = i % nx, i // nx
        if 0 < ix < nx - 1 and 0 < iy < ny - 1:
            lon[i] += rng.uniform(-dx, dx)
            lat[i] += rng.uniform(-dy, dy)

    elems = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            elems.append([n0, n0 + 1, n0 + nx])
            elems.append([n0 + 1, n0 + nx + 1, n0 + nx])
    elem_nodes = np.array(elems, dtype=np.int32)

    lon_c = np.array([lon[e].mean() for e in elem_nodes], dtype=np.float64)
    lat_c = np.array([lat[e].mean() for e in elem_nodes], dtype=np.float64)
    n2e, n2e_count = build_node_to_elems(n_nodes, elem_nodes)

    z_levels = np.linspace(0.0, H_MAX, 21)
    mesh = Mesh(
        name="test_downwelling_hires",
        nodes_lon=lon, nodes_lat=lat, depth=depth,
        elem_nodes=elem_nodes,
        elem_center_lon=lon_c, elem_center_lat=lat_c,
        node_to_elems=n2e, node_to_elems_count=n2e_count,
        sigma=z_levels,
    )
    build_search(mesh)

    coast_lon = CENTER_LON - span_x * 0.7
    return mesh, coast_lon


@pytest.fixture(scope="module")
def hires_setup():
    """Build high-res mesh + velocity once for all tests in this class."""
    mesh, coast_lon = _build_mesh_hires()
    vel = _build_velocity(mesh, coast_lon)
    # 500 particles, slightly spread, at z=50m
    rng = np.random.default_rng(123)
    lon0 = coast_lon + 15_000 / (COS_LAT * METERS_PER_DEG_LAT)
    lons = lon0 + rng.uniform(-0.02, 0.02, 500)
    lats = np.full(500, CENTER_LAT) + rng.uniform(-0.02, 0.02, 500)
    z0 = np.full(500, 50.0)
    state = make_state(lons, lats, z0)
    return mesh, vel, state, coast_lon


class TestStreamfunctionHiRes:
    """High-resolution validation: 65K nodes, 500 particles, dt=5s.

    On a smooth analytical field all 4 methods produce nearly identical
    trajectories (ΔZ < 0.02m).  This test verifies correctness at scale,
    not method differentiation — sharp gradients on real meshes would
    show larger differences.
    """

    def _run(self, mesh, vel, state, coast_lon, method):
        state = state.replace()  # copy
        psi0 = _streamfunction(state.lon, state.z, coast_lon)
        for _ in range(600):
            state = euler_step(state, vel, mesh, 5.0, time_dir=1.0,
                               w_interp=method)
        psi_f = _streamfunction(state.lon, state.z, coast_lon)
        alive = state.status == ParticleStatus.ALIVE
        drift = np.abs(psi_f[alive] - psi0[alive])
        scale = np.maximum(np.abs(psi0[alive]), 1e-12)
        rel = drift / scale
        return state, np.max(rel)

    @pytest.mark.parametrize("method", ["nearest", "leastsq", "barycentric", "idw"])
    def test_psi_bounded_hires(self, hires_setup, method):
        """ψ drift < 0.15% on 65K mesh with dt=5s."""
        mesh, vel, state, coast_lon = hires_setup
        _, rel_max = self._run(mesh, vel, state, coast_lon, method)
        assert rel_max < 0.0015, (
            f"method={method}: ψ rel drift {rel_max:.6f} > 0.15%"
        )

    def test_method_ranking(self, hires_setup):
        """Barycentric should be closest to leastsq in Z position."""
        mesh, vel, state, coast_lon = hires_setup
        states = {}
        for method in ["leastsq", "nearest", "barycentric", "idw"]:
            states[method], _ = self._run(mesh, vel, state, coast_lon, method)

        ref_z = states["leastsq"].z
        dz_nearest = np.mean(np.abs(states["nearest"].z - ref_z))
        dz_bary = np.mean(np.abs(states["barycentric"].z - ref_z))
        dz_idw = np.mean(np.abs(states["idw"].z - ref_z))

        # Barycentric should be closest to leastsq (exact linear interp)
        assert dz_bary < dz_nearest, (
            f"barycentric ΔZ {dz_bary:.6f} >= nearest ΔZ {dz_nearest:.6f}"
        )
        assert dz_bary < dz_idw, (
            f"barycentric ΔZ {dz_bary:.6f} >= idw ΔZ {dz_idw:.6f}"
        )
