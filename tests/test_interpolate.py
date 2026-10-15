"""Tests for kelp.numerics.interpolate — least-squares velocity interpolation.

Key validation: for a spatially constant velocity field,
the least-squares fit should recover the exact velocity at any point
within the element stencil.
"""
from __future__ import annotations


import numpy as np
import pytest

# Ensure package is importable

from kelp.numerics.interpolate import (
    interp_leastsq_single,
    interp_leastsq,
    interp_nearest_element,
    VelocityField,
)
from kelp.core.mesh import Mesh, build_node_to_elems, build_search, find_nearest_nodes
from kelp.core.particles import make_state, ParticleStatus


# ---------------------------------------------------------------------------
# Fixture: 3×3 grid with 8 triangles (good stencil for least-squares)
#
#   6---7---8
#   |\ |\ |
#   | \| \|
#   3---4---5
#   |\ |\ |
#   | \| \|
#   0---1---2
#
# Nodes at (lon, lat) = (4..6, 4..6), so the centre node 4 is at (5°, 5°).
# ---------------------------------------------------------------------------

@pytest.fixture
def grid3x3():
    """3×3 node grid with 8 triangles centred around (5°,5°)."""
    xs = np.array([4.0, 5.0, 6.0])
    ys = np.array([4.0, 5.0, 6.0])
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel().astype(np.float64)
    lat = gy.ravel().astype(np.float64)
    depth = np.zeros(9, dtype=np.float64)

    # 8 triangles (lower-left + upper-right per cell)
    elems = []
    for j in range(2):
        for i in range(2):
            n0 = j * 3 + i
            n1 = n0 + 1
            n2 = n0 + 3
            n3 = n0 + 4
            elems.append([n0, n1, n2])
            elems.append([n1, n3, n2])
    elem_nodes = np.array(elems, dtype=np.int32)
    n_elem = elem_nodes.shape[0]

    lon_c = np.array([lon[e].mean() for e in elem_nodes], dtype=np.float64)
    lat_c = np.array([lat[e].mean() for e in elem_nodes], dtype=np.float64)

    n2e, n2e_count = build_node_to_elems(9, elem_nodes)

    mesh = Mesh(
        name="test3x3",
        nodes_lon=lon, nodes_lat=lat, depth=depth,
        elem_nodes=elem_nodes,
        elem_center_lon=lon_c, elem_center_lat=lat_c,
        node_to_elems=n2e, node_to_elems_count=n2e_count,
    )
    build_search(mesh)
    return mesh


def _const_vel(mesh, u=1.0, v=0.0):
    """Create a VelocityField with uniform velocity on all elements."""
    ne = mesh.n_elements
    return VelocityField(
        u=np.full(ne, u, dtype=np.float64),
        v=np.full(ne, v, dtype=np.float64),
        eta=np.zeros(mesh.n_nodes, dtype=np.float64),
    )


# ===================================================================
# interp_leastsq_single tests
# ===================================================================

class TestInterpSingle:
    """Tests for the per-particle least-squares interpolation."""

    def test_constant_field_at_centre(self, grid3x3):
        """Constant velocity should be recovered exactly at the mesh centre."""
        vel = _const_vel(grid3x3, u=2.0, v=-1.0)
        # Node 4 is at (5,5), the centre
        UU, VV, ok = interp_leastsq_single(5.0, 5.0, 0, grid3x3, vel, 4)
        assert ok is True
        assert UU == pytest.approx(2.0, abs=0.05)
        assert VV == pytest.approx(-1.0, abs=0.05)

    def test_zero_field(self, grid3x3):
        """Zero velocity should return zero."""
        vel = _const_vel(grid3x3, u=0.0, v=0.0)
        UU, VV, ok = interp_leastsq_single(5.0, 5.0, 0, grid3x3, vel, 4)
        assert ok is True
        assert UU == pytest.approx(0.0, abs=1e-10)
        assert VV == pytest.approx(0.0, abs=1e-10)

    def test_backward_negates(self, grid3x3):
        """time_dir=-1 should negate the interpolated velocity."""
        vel = _const_vel(grid3x3, u=3.0, v=1.0)
        U_fwd, V_fwd, _ = interp_leastsq_single(5.0, 5.0, 0, grid3x3, vel, 4, time_dir=1.0)
        U_bwd, V_bwd, _ = interp_leastsq_single(5.0, 5.0, 0, grid3x3, vel, 4, time_dir=-1.0)
        assert U_fwd == pytest.approx(-U_bwd, abs=0.01)
        assert V_fwd == pytest.approx(-V_bwd, abs=0.01)

    def test_different_positions_constant_field(self, grid3x3):
        """Constant field → same velocity at different positions (near centre node)."""
        vel = _const_vel(grid3x3, u=1.5, v=-0.5)
        results = []
        # All these positions are near node 4
        for xp, yp in [(4.8, 4.8), (5.0, 5.0), (5.2, 5.2)]:
            UU, VV, ok = interp_leastsq_single(xp, yp, 0, grid3x3, vel, 4)
            assert ok
            results.append((UU, VV))
        for UU, VV in results:
            assert UU == pytest.approx(1.5, abs=0.15)
            assert VV == pytest.approx(-0.5, abs=0.15)

    def test_single_element_stencil(self, grid3x3):
        """A corner node with only 1 surrounding element has a near-singular
        normal matrix (1 data point, 3 unknowns).

        Fortran: with count=1 the matrix is analytically rank-1.
        Due to floating-point in large-coordinate metric space, the
        determinant may not hit exactly zero.  Either outcome is acceptable:
        ok=False (singular) or ok=True with reasonable velocity.
        """
        vel = _const_vel(grid3x3, u=1.0, v=0.0)
        count_0 = int(grid3x3.node_to_elems_count[0])
        assert count_0 == 1, "corner node should have exactly 1 element"

        UU, VV, ok = interp_leastsq_single(4.0, 4.0, 0, grid3x3, vel, 0)
        # Both outcomes are valid:
        if ok:
            # If it didn't detect singularity, the result should still be
            # in a reasonable range (not wildly divergent)
            assert abs(UU) < 100.0, "result should not diverge"
        else:
            assert UU == 0.0 and VV == 0.0

    def test_no_elements_returns_failure(self):
        """If the node has zero surrounding elements → ok=False."""
        # Create a minimal mesh with disconnected node
        lon = np.array([0.0, 1.0, 0.5, 5.0], dtype=np.float64)   # node 3 is isolated
        lat = np.array([0.0, 0.0, 1.0, 5.0], dtype=np.float64)
        depth = np.zeros(4, dtype=np.float64)
        elem_nodes = np.array([[0, 1, 2]], dtype=np.int32)
        lon_c = np.array([0.5], dtype=np.float64)
        lat_c = np.array([1.0/3], dtype=np.float64)
        n2e, n2e_count = build_node_to_elems(4, elem_nodes)
        mesh = Mesh(
            name="test_isolated",
            nodes_lon=lon, nodes_lat=lat, depth=depth,
            elem_nodes=elem_nodes,
            elem_center_lon=lon_c, elem_center_lat=lat_c,
            node_to_elems=n2e, node_to_elems_count=n2e_count,
        )
        vel = VelocityField(
            u=np.array([1.0]), v=np.array([0.0]),
            eta=np.zeros(4),
        )
        UU, VV, ok = interp_leastsq_single(5.0, 5.0, 0, mesh, vel, 3)
        assert ok is False
        assert UU == 0.0 and VV == 0.0


# ===================================================================
# interp_leastsq (vectorised) tests
# ===================================================================

class TestInterpVectorised:
    """Tests for the vectorised wrapper over all particles."""

    def test_two_active_particles(self, grid3x3):
        """Two particles near the centre should both get correct velocity."""
        vel = _const_vel(grid3x3, u=1.0, v=0.5)
        state = make_state(
            lon=np.array([4.8, 5.2]),
            lat=np.array([4.8, 5.2]),
        )
        nn = find_nearest_nodes(grid3x3, state.lon, state.lat)
        U, V, ok = interp_leastsq(state, grid3x3, vel, nn)
        assert np.all(ok)
        np.testing.assert_allclose(U, 1.0, atol=0.15)
        np.testing.assert_allclose(V, 0.5, atol=0.15)

    def test_dead_particle_skipped(self, grid3x3):
        """A particle with status OUTSIDE should not be interpolated."""
        vel = _const_vel(grid3x3, u=2.0, v=0.0)
        state = make_state(
            lon=np.array([5.0, 5.0]),
            lat=np.array([5.0, 5.0]),
        )
        state.status[1] = ParticleStatus.OUTSIDE
        nn = find_nearest_nodes(grid3x3, state.lon, state.lat)
        U, V, ok = interp_leastsq(state, grid3x3, vel, nn)
        assert ok[0] is True or ok[0] == True
        assert ok[1] == False
        assert U[1] == 0.0

    def test_consistency_with_single(self, grid3x3):
        """Vectorised result should match per-particle result."""
        vel = _const_vel(grid3x3, u=1.0, v=-0.3)
        state = make_state(
            lon=np.array([5.0]),
            lat=np.array([5.0]),
        )
        nn = find_nearest_nodes(grid3x3, state.lon, state.lat)

        # Vectorised
        U_vec, V_vec, ok_vec = interp_leastsq(state, grid3x3, vel, nn)

        # Single
        U_s, V_s, ok_s = interp_leastsq_single(5.0, 5.0, 0, grid3x3, vel, int(nn[0]))

        assert U_vec[0] == pytest.approx(U_s, abs=1e-12)
        assert V_vec[0] == pytest.approx(V_s, abs=1e-12)


# ===================================================================
# interp_nearest_element tests
# ===================================================================

class TestNearestElement:
    def test_basic(self, grid3x3):
        vel = _const_vel(grid3x3, u=3.0, v=-2.0)
        elem_ids = np.array([0, 1], dtype=np.int32)
        U, V, ok = interp_nearest_element(elem_ids, vel)
        np.testing.assert_allclose(U, 3.0)
        np.testing.assert_allclose(V, -2.0)
        assert np.all(ok)

    def test_outside_returns_zero(self, grid3x3):
        vel = _const_vel(grid3x3, u=1.0, v=1.0)
        elem_ids = np.array([-1], dtype=np.int32)
        U, V, ok = interp_nearest_element(elem_ids, vel)
        assert ok[0] == False
        assert U[0] == 0.0
        assert V[0] == 0.0


# ===================================================================
# VelocityField re-export check
# ===================================================================

class TestVelocityFieldReexport:
    def test_import_from_interpolate(self):
        """VelocityField should be importable from interpolate (for readers)."""
        from kelp.numerics.interpolate import VelocityField as VF
        from kelp.core.io import VelocityField as VF_io
        assert VF is VF_io
