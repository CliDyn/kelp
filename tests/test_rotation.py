"""Rotation primitives: Euler matrix, coordinate and vector rotation."""
import numpy as np
import pytest

from kelp.core.rotation import (
    FESOM2_ANGLES, euler_matrix, rotate_lonlat, unrotate_lonlat,
    rotate_vectors, unrotate_vectors,
)


class TestEulerMatrix:
    def test_orthonormal(self):
        R = euler_matrix(*FESOM2_ANGLES)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-15)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-14)

    def test_identity_angles(self):
        np.testing.assert_allclose(euler_matrix(0.0, 0.0, 0.0), np.eye(3),
                                   atol=1e-15)

    def test_fesom2_pole_moves_off_ninety(self):
        # the geographic North Pole lands well away from the rotated pole
        lon_r, lat_r = rotate_lonlat(np.array([0.0]), np.array([90.0]),
                                     euler_matrix(*FESOM2_ANGLES))
        assert abs(lat_r[0]) < 80.0
        # exact pin from the matrix: R @ (0,0,1) = third column
        R = euler_matrix(*FESOM2_ANGLES)
        v = R @ np.array([0.0, 0.0, 1.0])
        lat_expect = np.rad2deg(np.arcsin(v[2]))
        assert lat_r[0] == pytest.approx(lat_expect, abs=1e-12)


class TestCoordinateRoundTrip:
    def test_round_trip_random(self):
        rng = np.random.default_rng(11)
        lon = rng.uniform(-180, 180, 500)
        lat = rng.uniform(-89.9, 89.9, 500)
        R = euler_matrix(*FESOM2_ANGLES)
        lon_r, lat_r = rotate_lonlat(lon, lat, R)
        lon_b, lat_b = unrotate_lonlat(lon_r, lat_r, R)
        def xyz(lo, la):
            lo = np.deg2rad(lo); la = np.deg2rad(la)
            return np.stack([np.cos(la)*np.cos(lo),
                             np.cos(la)*np.sin(lo), np.sin(la)], -1)
        d = np.linalg.norm(xyz(lon, lat) - xyz(lon_b, lat_b), axis=-1)
        assert d.max() < 1e-12

    def test_output_ranges(self):
        R = euler_matrix(*FESOM2_ANGLES)
        lon_r, lat_r = rotate_lonlat(np.array([170.0, -170.0]),
                                     np.array([85.0, -85.0]), R)
        assert ((lon_r >= -180.0) & (lon_r < 180.0)).all()
        assert ((lat_r >= -90.0) & (lat_r <= 90.0)).all()

    def test_lon_wrap_180_maps_to_minus_180(self):
        # image landing on the antimeridian wraps to -180, never +180
        # (identity rotation keeps lon=180 -> must wrap to -180.0)
        lon_r, lat_r = rotate_lonlat(np.array([180.0]), np.array([30.0]),
                                     euler_matrix(0.0, 0.0, 0.0))
        assert lon_r[0] == -180.0
        assert lon_r[0] < 180.0


class TestVectorRotation:
    def test_magnitude_preserved(self):
        rng = np.random.default_rng(7)
        lon = rng.uniform(-180, 180, 300)
        lat = rng.uniform(-89.0, 89.0, 300)
        u = rng.normal(0, 1, 300); v = rng.normal(0, 1, 300)
        R = euler_matrix(*FESOM2_ANGLES)
        u_r, v_r = rotate_vectors(u, v, lon, lat, R)
        np.testing.assert_allclose(np.hypot(u_r, v_r), np.hypot(u, v),
                                   rtol=1e-12)

    def test_round_trip(self):
        rng = np.random.default_rng(3)
        lon = rng.uniform(-180, 180, 300)
        lat = rng.uniform(-89.0, 89.0, 300)
        u = rng.normal(0, 1, 300); v = rng.normal(0, 1, 300)
        R = euler_matrix(*FESOM2_ANGLES)
        u_r, v_r = rotate_vectors(u, v, lon, lat, R)
        u_b, v_b = unrotate_vectors(u_r, v_r, lon, lat, R)
        np.testing.assert_allclose(u_b, u, atol=1e-12)
        np.testing.assert_allclose(v_b, v, atol=1e-12)

    def test_identity_rotation_is_noop(self):
        lon = np.array([10.0, 50.0]); lat = np.array([0.0, 45.0])
        u = np.array([1.0, 0.3]); v = np.array([0.0, -0.7])
        R = euler_matrix(0.0, 0.0, 0.0)
        u_r, v_r = rotate_vectors(u, v, lon, lat, R)
        np.testing.assert_allclose(u_r, u, atol=1e-14)
        np.testing.assert_allclose(v_r, v, atol=1e-14)

    def test_z_axis_rotation_keeps_east_east(self):
        # alpha=90 (z-axis) only shifts longitudes: eastward stays eastward
        R = euler_matrix(90.0, 0.0, 0.0)
        u_r, v_r = rotate_vectors(np.array([1.0]), np.array([0.0]),
                                  np.array([0.0]), np.array([0.0]), R)
        assert u_r[0] == pytest.approx(1.0, abs=1e-14)
        assert v_r[0] == pytest.approx(0.0, abs=1e-14)

    def test_finite_difference_consistency(self):
        # independent check of the analytic vector formula: displacing a
        # point a tiny step along (u, v) in the geographic chart and
        # rotating both points reproduces the rotated components
        R2 = euler_matrix(0.0, 90.0, 0.0)
        lon0, lat0 = np.array([10.0]), np.array([20.0])
        u0, v0 = np.array([0.7]), np.array([-0.4])
        eps = 1e-6
        dlon = u0 * eps / np.cos(np.deg2rad(lat0))
        dlat = v0 * eps
        p_r = np.stack(rotate_lonlat(lon0, lat0, R2), -1)
        q_r = np.stack(rotate_lonlat(lon0 + dlon, lat0 + dlat, R2), -1)
        fd_u = (q_r[:, 0] - p_r[:, 0]) * np.cos(np.deg2rad(p_r[:, 1])) / eps
        fd_v = (q_r[:, 1] - p_r[:, 1]) / eps
        u_r2, v_r2 = rotate_vectors(u0, v0, lon0, lat0, R2)
        assert u_r2[0] == pytest.approx(fd_u[0], abs=1e-4)
        assert v_r2[0] == pytest.approx(fd_v[0], abs=1e-4)

    def test_unit_north_is_minus_s_c(self):
        # the tangent-plane map is a pure rotation [[c,-s],[s,c]]:
        # column 1 = image of unit-east = (c, s); column 2 = image of
        # unit-north MUST be (-s, c) — a reflection would give (s, -c).
        rng = np.random.default_rng(21)
        lon = rng.uniform(-180, 180, 200)
        lat = rng.uniform(-89.0, 89.0, 200)
        R = euler_matrix(*FESOM2_ANGLES)
        c, s = rotate_vectors(np.ones(200), np.zeros(200), lon, lat, R)
        un, vn = rotate_vectors(np.zeros(200), np.ones(200), lon, lat, R)
        np.testing.assert_allclose(un, -s, atol=1e-12)
        np.testing.assert_allclose(vn, c, atol=1e-12)


class TestConfigAndMeshField:
    def test_mesh_config_rotation_values(self):
        from kelp.core.config import MeshConfig
        assert MeshConfig().rotation is None
        MeshConfig(rotation="fesom2")
        MeshConfig(rotation=[50.0, 15.0, -90.0])
        with pytest.raises(ValueError, match="rotation"):
            MeshConfig(rotation="nemo")
        with pytest.raises(ValueError, match="rotation"):
            MeshConfig(rotation=[50.0, 15.0])
        with pytest.raises(ValueError, match="rotation"):
            MeshConfig(rotation=[50.0, float("nan"), -90.0])

    def test_mesh_field_default(self):
        from tests.test_uv_registry import _mesh_from
        m = _mesh_from([0, 1, 0], [0, 0, 1], [[0, 1, 2, -1]])
        assert m.rotation_angles is None
        assert m._vec_rot_factors is None


class TestCachedFactorMapEqualsRotateVectors:
    """The reader caches per-carrier factors ``(c, s) = rotate_vectors(unit-east)``
    and applies the 2x2 map ``u' = c*u - s*v, v' = s*u + c*v`` in read_velocities.
    Pin that this map is EXACTLY ``rotate_vectors(u, v, ...)`` for arbitrary
    vectors — the rotated-frame velocity rotation relies on this equivalence
    (it holds because R is a proper rotation, so the induced tangent-plane map is
    a proper rotation: unit-east -> (c, s) implies unit-north -> (-s, c))."""

    def test_factor_map_equals_full_rotate_vectors(self):
        rng = np.random.default_rng(0)
        n = 600
        lon = rng.uniform(-180.0, 180.0, n)
        lat = rng.uniform(-89.0, 89.0, n)
        R = euler_matrix(*FESOM2_ANGLES)
        # factors exactly as readers.fesom2._apply_mesh_rotation computes them
        c, s = rotate_vectors(np.ones(n), np.zeros(n), lon, lat, R)
        u, v = rng.standard_normal(n), rng.standard_normal(n)
        cached_u, cached_v = c * u - s * v, s * u + c * v
        full_u, full_v = rotate_vectors(u, v, lon, lat, R)
        np.testing.assert_allclose(cached_u, full_u, atol=1e-12)
        np.testing.assert_allclose(cached_v, full_v, atol=1e-12)
        # and the map is magnitude-preserving (a proper rotation)
        np.testing.assert_allclose(np.hypot(cached_u, cached_v),
                                   np.hypot(u, v), atol=1e-12)
