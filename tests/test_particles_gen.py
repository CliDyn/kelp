"""Tests for kelp.core.particles_gen — particle seeding utilities."""
from __future__ import annotations

import numpy as np
import pytest

from kelp.core.particles_gen import generate_grid, generate_around_point
from kelp.core.mesh import Mesh, build_search, build_node_to_elems


# ---------------------------------------------------------------------------
# Helper: 3×3 node mesh (8 triangles) centred at (5, 5)
# ---------------------------------------------------------------------------

def _make_simple_mesh():
    """Create a 3×3 node grid with 8 triangles."""
    xs = np.array([4.0, 5.0, 6.0])
    ys = np.array([4.0, 5.0, 6.0])
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel().astype(np.float64)
    lat = gy.ravel().astype(np.float64)

    elems = []
    for j in range(2):
        for i in range(2):
            n0 = j * 3 + i
            n1 = n0 + 1
            n2 = n0 + 3
            n3 = n0 + 4
            elems.append([n0, n1, n2, -1])
            elems.append([n1, n3, n2, -1])
    elem_nodes = np.array(elems, dtype=np.int32)

    elem_nodes_tri = elem_nodes[:, :3]
    lon_c = np.array([lon[e].mean() for e in elem_nodes_tri], dtype=np.float64)
    lat_c = np.array([lat[e].mean() for e in elem_nodes_tri], dtype=np.float64)

    n2e, cnt = build_node_to_elems(len(lon), elem_nodes)

    mesh = Mesh(
        name="test",
        nodes_lon=lon,
        nodes_lat=lat,
        depth=np.full(len(lon), 10.0),
        elem_nodes=elem_nodes,
        elem_center_lon=lon_c,
        elem_center_lat=lat_c,
        node_to_elems=n2e,
        node_to_elems_count=cnt,
        sigma=np.array([], dtype=np.float64),
    )
    build_search(mesh)
    return mesh


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGenerateGrid:
    def test_basic_grid(self):
        mesh = _make_simple_mesh()
        state = generate_grid(mesh, nx=3, ny=3)
        # all returned particles must be in [4,6] x [4,6]
        assert state.n_particles > 0
        assert np.all(state.lon >= 4.0)
        assert np.all(state.lon <= 6.0)
        assert np.all(state.lat >= 4.0)
        assert np.all(state.lat <= 6.0)

    def test_grid_with_margin(self):
        mesh = _make_simple_mesh()
        state_full = generate_grid(mesh, nx=5, ny=5, margin=0.0)
        state_margin = generate_grid(mesh, nx=5, ny=5, margin=0.1)
        # margin shrinks the candidate area, so interior particles stay the same
        # or fewer get filtered — but all should be inside mesh
        assert state_margin.n_particles > 0
        assert state_margin.n_particles <= state_full.n_particles

    def test_fine_grid_covers_interior(self):
        mesh = _make_simple_mesh()
        state = generate_grid(mesh, nx=10, ny=10)
        # interior of 2×2 degree mesh should have many particles
        assert state.n_particles >= 20

    def test_1x1_grid(self):
        mesh = _make_simple_mesh()
        # linspace(min, max, 1) returns [min], so single point at corner
        state = generate_grid(mesh, nx=1, ny=1)
        assert state.n_particles == 1


class TestGenerateAroundPoint:
    def test_basic(self):
        mesh = _make_simple_mesh()
        state = generate_around_point(mesh, 5.0, 5.0, n=50,
                                      radius_km=50.0, seed=42)
        assert state.n_particles > 0

    def test_all_within_radius(self):
        mesh = _make_simple_mesh()
        from kelp.core.geometry import great_circle_distance
        state = generate_around_point(mesh, 5.0, 5.0, n=100,
                                      radius_km=50.0, seed=123)
        for i in range(state.n_particles):
            d = great_circle_distance(5.0, 5.0, state.lon[i], state.lat[i])
            assert d < 55_000  # 50 km + small tolerance

    def test_reproducible_with_seed(self):
        mesh = _make_simple_mesh()
        s1 = generate_around_point(mesh, 5.0, 5.0, n=30,
                                   radius_km=20.0, seed=99)
        s2 = generate_around_point(mesh, 5.0, 5.0, n=30,
                                   radius_km=20.0, seed=99)
        np.testing.assert_array_equal(s1.lon, s2.lon)
        np.testing.assert_array_equal(s1.lat, s2.lat)

    def test_outside_mesh_raises(self):
        mesh = _make_simple_mesh()
        # point far from mesh — all candidates should be filtered
        with pytest.raises(ValueError, match="No candidate particles"):
            generate_around_point(mesh, 90.0, 90.0, n=10,
                                  radius_km=1.0, seed=0)
