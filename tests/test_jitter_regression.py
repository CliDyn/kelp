"""Regression tests for the jittered-mesh false-OUTSIDE bug.

Bug (surfaced on 161x161+ jittered meshes): the containment
search (`_find_containing_element_local` / `_loop_find_containing_elements`)
scans only the elements adjacent to the particle's nearest node (1-ring).
On an irregular (jittered) triangulation the nearest node to a point is
not always a vertex of the containing triangle — near the diagonal of a
grid cell all four corners are nearly equidistant, so a few-percent node
perturbation can make a non-vertex node the nearest one.  The 1-ring scan
then misses the containing element and the particle is falsely reverted
to OUTSIDE.

Root-cause probe (2026-06-04): on a jitter=0.03 mesh, ~0.05% of interior
points fail the 1-ring scan; in 100% of failures the containing element
exists and lies in the 2-ring of the nearest node (elements adjacent to
the vertices of the 1-ring elements).
"""

from __future__ import annotations

import numpy as np
import pytest

from kelp.core.mesh import (
    _find_containing_element_local,
    find_containing_elements,
    find_nearest_nodes,
)
from kelp.core.particles import ParticleStatus, make_state
from kelp.core.io import VelocityField
from kelp.core.geometry import DEG2RAD, METERS_PER_DEG_LAT
from kelp.numerics.integrators import euler_step

from tests.test_analytical_3d_benchmarks import _build_patch_mesh

LON0, LAT0, SPAN = 10.0, 45.0, 0.3


@pytest.fixture(scope="module")
def jittered_mesh():
    """161x161 patch mesh with 3% node jitter (the documented reproducer)."""
    return _build_patch_mesh(LON0, LAT0, span=SPAN, nx=161, ny=161, jitter=0.03)


def _interior_points(n: int, seed: int = 11):
    """Random points well inside the mesh (10% margin from the boundary)."""
    rng = np.random.default_rng(seed)
    lons = rng.uniform(LON0 - 0.9 * SPAN, LON0 + 0.9 * SPAN, n)
    lats = rng.uniform(LAT0 - 0.9 * SPAN, LAT0 + 0.9 * SPAN, n)
    return lons, lats


class TestJitteredContainment:
    """Every interior point must be located in some element."""

    N_POINTS = 20_000

    def test_batch_kernel_finds_all_interior_points(self, jittered_mesh):
        mesh = jittered_mesh
        lons, lats = _interior_points(self.N_POINTS)
        nns = find_nearest_nodes(mesh, lons, lats)
        _, in_domain = find_containing_elements(mesh, lons, lats, nns)
        missed = int(np.count_nonzero(~in_domain))
        assert missed == 0, (
            f"{missed}/{self.N_POINTS} interior points not found by the "
            f"batch containment kernel on a jitter=0.03 mesh"
        )

    def test_single_kernel_finds_all_interior_points(self, jittered_mesh):
        mesh = jittered_mesh
        lons, lats = _interior_points(self.N_POINTS)
        nns = find_nearest_nodes(mesh, lons, lats)
        missed = 0
        for i in range(self.N_POINTS):
            _, ok = _find_containing_element_local(
                lons[i], lats[i], int(nns[i]),
                mesh.node_to_elems, mesh.node_to_elems_count,
                mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat,
            )
            if not ok:
                missed += 1
        assert missed == 0, (
            f"{missed}/{self.N_POINTS} interior points not found by the "
            f"single-particle containment kernel on a jitter=0.03 mesh"
        )


_BARY_EPS = -1e-6  # same containment tolerance as kelp.core.mesh


def _tris_contain_point(mesh, elem_ids, lon_p: float, lat_p: float):
    """Pure-numpy point-in-triangle test for a batch of candidate elements.

    Barycentric weights are invariant under affine maps, so testing
    directly in (lon, lat) coordinates is equivalent to the
    metric-projected test inside ``kelp.core.mesh`` (same formula,
    same ``-1e-6`` tolerance).  Deliberately numpy-only: keeps the
    danger-point classifier fully independent of the njit kernels under
    test, and avoids calling a ``cache=True`` njit function that returns
    ``Optional[ndarray]`` across the njit→Python boundary — a fragile
    numba path that raised ``RuntimeError: In
    'NRT_adapt_ndarray_to_python', 'descr' is NULL`` on some on-disk
    cache states (order-dependent, full-suite only).
    """
    tri = mesh.elem_nodes[np.asarray(elem_ids, dtype=np.int64)]  # (M, 3)
    x = mesh.nodes_lon[tri]
    y = mesh.nodes_lat[tri]
    ax, ay = x[:, 0], y[:, 0]
    bx, by = x[:, 1], y[:, 1]
    cx, cy = x[:, 2], y[:, 2]
    denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    nondegenerate = np.abs(denom) > 0.0
    safe = np.where(nondegenerate, denom, 1.0)
    w1 = ((by - cy) * (lon_p - cx) + (cx - bx) * (lat_p - cy)) / safe
    w2 = ((cy - ay) * (lon_p - cx) + (ax - cx) * (lat_p - cy)) / safe
    w3 = 1.0 - w1 - w2
    return (nondegenerate & (w1 >= _BARY_EPS)
            & (w2 >= _BARY_EPS) & (w3 >= _BARY_EPS))


def _danger_points(mesh, n_sample: int = 20_000, seed: int = 11):
    """Find interior points where a 1-ring-only containment scan must fail.

    A point is "dangerous" when at least one element contains it but
    *none* of the containing elements is adjacent to its nearest node —
    exactly the condition under which a 1-ring scan returns a false
    OUTSIDE.  Computed from mesh topology alone with pure numpy/python
    (see ``_tris_contain_point``), independent of the containment
    kernels under test.
    """
    assert mesh.elem_nodes.shape[1] == 3, (
        "danger-point helper assumes the triangular patch mesh fixture"
    )
    lons, lats = _interior_points(n_sample, seed=seed)
    nns = find_nearest_nodes(mesh, lons, lats)

    danger_lon, danger_lat = [], []
    for i in range(n_sample):
        nn = int(nns[i])
        ring1 = {int(e) for e in
                 mesh.node_to_elems[nn][: mesh.node_to_elems_count[nn]]
                 if e >= 0}
        # 2-ring candidate elements via the vertices of the 1-ring
        verts = {int(v) for e in ring1 for v in mesh.elem_nodes[e] if v >= 0}
        cand = sorted({int(e) for v in verts
                       for e in mesh.node_to_elems[v][: mesh.node_to_elems_count[v]]
                       if e >= 0})
        mask = _tris_contain_point(mesh, cand, lons[i], lats[i])
        hits = [cand[j] for j in np.nonzero(mask)[0]]
        if hits and not any(e in ring1 for e in hits):
            danger_lon.append(lons[i])
            danger_lat.append(lats[i])
    return np.asarray(danger_lon), np.asarray(danger_lat)


class TestJitteredTrackingDangerPoints:
    """End-to-end regression: particles seeded on danger points survive.

    Danger points (containing element not adjacent to the nearest node)
    are found geometrically, then particles are advected across them
    with a slow uniform flow.  Without the 2-ring fallback the very
    first domain check kills them.
    """

    def test_particles_on_danger_points_stay_alive(self, jittered_mesh):
        mesh = jittered_mesh
        d_lon, d_lat = _danger_points(mesh)
        # The jitter=0.03 mesh (seed 42) must expose a handful of these;
        # if the mesh generator ever changes, regenerate the expectation.
        assert d_lon.shape[0] >= 3, (
            f"mesh no longer exposes danger points ({d_lon.shape[0]} found) "
            f"— the fixture lost its regression value, adjust seeds"
        )

        # Slow drift: 0.6 m per step keeps the particle inside the thin
        # danger sliver (whose width scales with the jitter amplitude,
        # ~metres) when the post-step domain check runs.
        vel = VelocityField(
            u=np.full(mesh.elem_nodes.shape[0], 1e-3),
            v=np.zeros(mesh.elem_nodes.shape[0]),
            eta=np.zeros(mesh.nodes_lon.shape[0]),
        )
        state = make_state(d_lon, d_lat)
        for _ in range(3):
            state = euler_step(state, vel, mesh, 600.0, time_dir=1.0)

        n_dead = int(np.count_nonzero(state.status != ParticleStatus.ALIVE))
        assert n_dead == 0, (
            f"{n_dead}/{d_lon.shape[0]} particles seeded on danger points "
            f"falsely went OUTSIDE (1-ring containment miss)"
        )


class TestJitteredTracking:
    """End-to-end regression: spiralling advection must not kill particles.

    The velocity field is solid-body rotation plus a weak radial outflow
    (in metres), so each particle spirals outward from r0=0.04° to
    r1=0.16° (physical radii, in latitude degrees) over the run and
    lands in fresh grid cells on every step — sampling a broad annulus
    instead of re-visiting one ring.  A physical radius of r spans
    r/cos(45°) ≈ 1.41·r degrees in longitude, so the final radius keeps
    every trajectory >= 0.07° (≈ 20 grid cells) away from the 0.3° mesh
    boundary.  Any OUTSIDE status is therefore a false positive from the
    containment search.
    """

    def test_spiral_advection_keeps_all_particles_alive(self, jittered_mesh):
        mesh = jittered_mesh

        n_steps = 400
        dt = 600.0
        period = n_steps * dt  # one full revolution
        omega = 2.0 * np.pi / period          # angular velocity, rad/s
        r0, r1 = 0.04, 0.16                   # physical radii, lat-degrees
        alpha = np.log(r1 / r0) / period      # radial growth rate, 1/s

        # u = -omega*y + alpha*x ; v = omega*x + alpha*y  (metres/s)
        cos0 = np.cos(LAT0 * DEG2RAD)
        xe = (mesh.elem_center_lon - LON0) * cos0 * METERS_PER_DEG_LAT
        ye = (mesh.elem_center_lat - LAT0) * METERS_PER_DEG_LAT
        vel = VelocityField(
            u=-omega * ye + alpha * xe,
            v=omega * xe + alpha * ye,
            eta=np.zeros(mesh.nodes_lon.shape[0]),
        )

        # 64 particles on the inner ring — deterministic seeding
        angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
        lon0 = LON0 + r0 * np.cos(angles) / cos0
        lat0 = LAT0 + r0 * np.sin(angles)
        state = make_state(lon0, lat0)

        for _ in range(n_steps):
            state = euler_step(state, vel, mesh, dt, time_dir=1.0)

        n_dead = int(np.count_nonzero(state.status != ParticleStatus.ALIVE))
        assert n_dead == 0, (
            f"{n_dead}/64 spiralling particles falsely went OUTSIDE on a "
            f"jitter=0.03 mesh (all trajectories stay >= 0.07 deg from the "
            f"boundary)"
        )
