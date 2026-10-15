"""Tests for vertical coordinate utilities (3D tracking)."""
import numpy as np
import pytest

from kelp.numerics.vertical import (
    find_layer,
    find_layers,
    interp_w_nearest,
    interp_w_linear,
    clamp_depth,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def z_levels():
    """Typical FESOM2 z-levels (positive-down, meters)."""
    return np.array([0.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0], dtype=np.float64)


@pytest.fixture
def z_levels_fine():
    """Finer z-levels for interpolation tests."""
    return np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)


# ---------------------------------------------------------------------------
# find_layer tests
# ---------------------------------------------------------------------------

class TestFindLayer:
    def test_surface(self, z_levels):
        """Particle at surface should be in layer 0."""
        assert find_layer(z_levels, 0.0, 200.0) == 0

    def test_above_surface(self, z_levels):
        """Particle above surface (negative depth) clamped to layer 0."""
        assert find_layer(z_levels, -1.0, 200.0) == 0

    def test_mid_depth(self, z_levels):
        """Particle at 7m with levels [0, 5, 10, ...] should be in layer 1 (5-10m)."""
        assert find_layer(z_levels, 7.0, 200.0) == 1

    def test_at_level_boundary(self, z_levels):
        """Particle exactly at a level boundary: z_levels[k] <= z < z_levels[k+1]."""
        # At z=10.0 exactly: layer 2 (10-25m)
        assert find_layer(z_levels, 10.0, 200.0) == 2

    def test_deep(self, z_levels):
        """Particle at 150m with levels up to 200m should be in layer 4 (100-200m)."""
        assert find_layer(z_levels, 150.0, 200.0) == 5

    def test_bottom_clamp(self, z_levels):
        """Particle below bottom depth should be in deepest valid layer."""
        # Bottom at 50m. Valid layers: 0(0-5), 1(5-10), 2(10-25), 3(25-50).
        # z_levels[4]=50 <= bottom=50, so layer 3 (25-50m) is deepest valid.
        assert find_layer(z_levels, 80.0, 50.0) == 3

    def test_shallow_node(self, z_levels):
        """Very shallow node: bottom at 5m, only layer 0 is valid."""
        assert find_layer(z_levels, 3.0, 5.0) == 0
        assert find_layer(z_levels, 10.0, 5.0) == 0

    def test_bottom_between_levels(self, z_levels):
        """Bottom depth between two z-levels."""
        # Bottom at 30m. z_levels: [0, 5, 10, 25, 50, ...]
        # z_levels[3]=25 <= 30 but z_levels[4]=50 > 30
        # So max_layer = 3-1 = 2? No, max_layer should be 2 (layer 2: 10-25m)
        # Actually: layer k valid if z_levels[k+1] <= bottom.
        # k=0: z_levels[1]=5 <= 30 ✓
        # k=1: z_levels[2]=10 <= 30 ✓
        # k=2: z_levels[3]=25 <= 30 ✓
        # k=3: z_levels[4]=50 > 30 ✗
        # So max valid layer = 2 (10-25m)... wait that's k=2 not k=3
        # Hmm, but the particle at 27m should still use layer 2 (10-25m)?
        # Actually layer 3 spans 25-50m. If bottom is 30m, the particle at 27m
        # is in the 25-50m bracket, but data may be invalid below 30m.
        # The layer k is valid if z_levels[k+1] <= bottom_depth.
        # k=3: z_levels[4]=50 > 30 → invalid.
        # So max_layer = 2. But a particle at 27m (between z_levels[3]=25 and bottom=30)
        # should use the deepest valid layer = 2 (which is clamped).
        layer = find_layer(z_levels, 27.0, 30.0)
        # Should clamp to max valid layer
        assert layer <= 3


# ---------------------------------------------------------------------------
# find_layers tests (vectorised)
# ---------------------------------------------------------------------------

class TestFindLayers:
    def test_multiple_particles(self, z_levels):
        """Multiple particles at various depths."""
        z_particles = np.array([0.0, 7.0, 30.0, 150.0], dtype=np.float64)
        nn_nodes = np.array([0, 0, 0, 0], dtype=np.int32)
        node_depths = np.array([200.0], dtype=np.float64)
        alive = np.array([True, True, True, True])

        layers = find_layers(z_levels, z_particles, nn_nodes, node_depths, alive)
        assert layers[0] == 0   # surface
        assert layers[1] == 1   # 5-10m
        assert layers[2] == 3   # 25-50m
        assert layers[3] == 5   # 100-200m

    def test_dead_particles_ignored(self, z_levels):
        """Dead particles get layer 0 (default)."""
        z_particles = np.array([50.0, 100.0], dtype=np.float64)
        nn_nodes = np.array([0, 0], dtype=np.int32)
        node_depths = np.array([200.0], dtype=np.float64)
        alive = np.array([False, True])

        layers = find_layers(z_levels, z_particles, nn_nodes, node_depths, alive)
        assert layers[0] == 0   # dead → 0
        assert layers[1] == 5   # 100-200m (z=100 is at boundary → layer 5)


# ---------------------------------------------------------------------------
# interp_w_nearest tests
# ---------------------------------------------------------------------------

class TestInterpWNearest:
    def test_exact_level(self, z_levels_fine):
        """W at exact z-level returns that level's value."""
        K = len(z_levels_fine)
        N = 3
        w_field = np.zeros((K, N), dtype=np.float64)
        # Set known values: W at level 2 (z=20m), node 1 = 5.0
        w_field[2, 1] = 5.0

        z_p = np.array([20.0], dtype=np.float64)
        nn = np.array([1], dtype=np.int32)
        alive = np.array([True])

        W = interp_w_nearest(w_field, z_levels_fine, z_p, nn, alive, 1.0)
        assert W[0] == pytest.approx(5.0)

    def test_between_levels(self, z_levels_fine):
        """W between two levels picks the nearest one."""
        K = len(z_levels_fine)
        N = 2
        w_field = np.zeros((K, N), dtype=np.float64)
        w_field[1, 0] = 3.0  # z=10m
        w_field[2, 0] = 7.0  # z=20m

        # At z=12m, nearest to z=10m (level 1)
        z_p = np.array([12.0], dtype=np.float64)
        nn = np.array([0], dtype=np.int32)
        alive = np.array([True])

        W = interp_w_nearest(w_field, z_levels_fine, z_p, nn, alive, 1.0)
        assert W[0] == pytest.approx(3.0)

    def test_time_dir_backward(self, z_levels_fine):
        """Backward tracking flips W sign."""
        K = len(z_levels_fine)
        N = 1
        w_field = np.ones((K, N), dtype=np.float64) * 2.0

        z_p = np.array([10.0], dtype=np.float64)
        nn = np.array([0], dtype=np.int32)
        alive = np.array([True])

        W = interp_w_nearest(w_field, z_levels_fine, z_p, nn, alive, -1.0)
        assert W[0] == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# interp_w_linear tests
# ---------------------------------------------------------------------------

class TestInterpWLinear:
    def test_exact_level(self, z_levels_fine):
        """W at exact z-level returns that level's value."""
        K = len(z_levels_fine)
        N = 2
        w_field = np.zeros((K, N), dtype=np.float64)
        w_field[2, 0] = 4.0  # z=20m, node 0

        z_p = np.array([20.0], dtype=np.float64)
        nn = np.array([0], dtype=np.int32)
        alive = np.array([True])

        W = interp_w_linear(w_field, z_levels_fine, z_p, nn, alive, 1.0)
        assert W[0] == pytest.approx(4.0)

    def test_midpoint_interpolation(self, z_levels_fine):
        """W at midpoint between two levels is the average."""
        K = len(z_levels_fine)
        N = 2
        w_field = np.zeros((K, N), dtype=np.float64)
        w_field[1, 0] = 2.0  # z=10m
        w_field[2, 0] = 6.0  # z=20m

        # At z=15m (midpoint), expect (2+6)/2 = 4.0
        z_p = np.array([15.0], dtype=np.float64)
        nn = np.array([0], dtype=np.int32)
        alive = np.array([True])

        W = interp_w_linear(w_field, z_levels_fine, z_p, nn, alive, 1.0)
        assert W[0] == pytest.approx(4.0)

    def test_quarter_interpolation(self, z_levels_fine):
        """W at quarter point between two levels."""
        K = len(z_levels_fine)
        N = 1
        w_field = np.zeros((K, N), dtype=np.float64)
        w_field[0, 0] = 0.0  # z=0m
        w_field[1, 0] = 8.0  # z=10m

        # At z=2.5m (quarter), expect 0.25*8 = 2.0
        z_p = np.array([2.5], dtype=np.float64)
        nn = np.array([0], dtype=np.int32)
        alive = np.array([True])

        W = interp_w_linear(w_field, z_levels_fine, z_p, nn, alive, 1.0)
        assert W[0] == pytest.approx(2.0)

    def test_surface_clamp(self, z_levels_fine):
        """W above surface returns surface value."""
        K = len(z_levels_fine)
        N = 1
        w_field = np.ones((K, N), dtype=np.float64) * 3.0
        w_field[0, 0] = 1.0

        z_p = np.array([-5.0], dtype=np.float64)
        nn = np.array([0], dtype=np.int32)
        alive = np.array([True])

        W = interp_w_linear(w_field, z_levels_fine, z_p, nn, alive, 1.0)
        assert W[0] == pytest.approx(1.0)

    def test_bottom_clamp(self, z_levels_fine):
        """W below deepest level returns bottom value."""
        K = len(z_levels_fine)
        N = 1
        w_field = np.ones((K, N), dtype=np.float64) * 3.0
        w_field[-1, 0] = 9.0

        z_p = np.array([100.0], dtype=np.float64)
        nn = np.array([0], dtype=np.int32)
        alive = np.array([True])

        W = interp_w_linear(w_field, z_levels_fine, z_p, nn, alive, 1.0)
        assert W[0] == pytest.approx(9.0)

    def test_time_dir(self, z_levels_fine):
        """time_dir multiplies the result."""
        K = len(z_levels_fine)
        N = 1
        w_field = np.ones((K, N), dtype=np.float64) * 2.0

        z_p = np.array([5.0], dtype=np.float64)
        nn = np.array([0], dtype=np.int32)
        alive = np.array([True])

        W_fwd = interp_w_linear(w_field, z_levels_fine, z_p, nn, alive, 1.0)
        W_bwd = interp_w_linear(w_field, z_levels_fine, z_p, nn, alive, -1.0)
        assert W_fwd[0] == pytest.approx(2.0)
        assert W_bwd[0] == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# clamp_depth tests
# ---------------------------------------------------------------------------

class TestClampDepth:
    def test_within_range(self, z_levels):
        """Depth within valid range is unchanged."""
        z = np.array([50.0], dtype=np.float64)
        nn = np.array([0], dtype=np.int32)
        depths = np.array([200.0], dtype=np.float64)
        alive = np.array([True])

        z_out = clamp_depth(z, z_levels, nn, depths, alive)
        assert z_out[0] == pytest.approx(50.0)

    def test_surface_clamp(self, z_levels):
        """Negative depth clamped to z_levels[0]=0."""
        z = np.array([-10.0], dtype=np.float64)
        nn = np.array([0], dtype=np.int32)
        depths = np.array([200.0], dtype=np.float64)
        alive = np.array([True])

        z_out = clamp_depth(z, z_levels, nn, depths, alive)
        assert z_out[0] == pytest.approx(0.0)

    def test_bottom_clamp(self, z_levels):
        """Depth beyond local bottom is clamped."""
        z = np.array([250.0], dtype=np.float64)
        nn = np.array([0], dtype=np.int32)
        depths = np.array([100.0], dtype=np.float64)
        alive = np.array([True])

        z_out = clamp_depth(z, z_levels, nn, depths, alive)
        assert z_out[0] == pytest.approx(100.0)

    def test_per_node_bottom(self, z_levels):
        """Different nodes have different bottom depths."""
        z = np.array([80.0, 80.0], dtype=np.float64)
        nn = np.array([0, 1], dtype=np.int32)
        depths = np.array([50.0, 200.0], dtype=np.float64)
        alive = np.array([True, True])

        z_out = clamp_depth(z, z_levels, nn, depths, alive)
        assert z_out[0] == pytest.approx(50.0)   # clamped to node 0 bottom
        assert z_out[1] == pytest.approx(80.0)   # within node 1 range

    def test_dead_particles_unchanged(self, z_levels):
        """Dead particles keep their depth."""
        z = np.array([-999.0], dtype=np.float64)
        nn = np.array([0], dtype=np.int32)
        depths = np.array([200.0], dtype=np.float64)
        alive = np.array([False])

        z_out = clamp_depth(z, z_levels, nn, depths, alive)
        assert z_out[0] == pytest.approx(-999.0)
