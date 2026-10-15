"""Tests for kelp.core.geometry."""
import numpy as np
import pytest

from kelp.core.geometry import (
    METERS_PER_DEG_LAT, DEG2RAD,
    wrap_lon, clamp_lat,
    lonlat_to_xyz, xyz_to_lonlat,
    metric_z0, angular_rates,
    great_circle_distance, delta_lon,
)


class TestWrapLon:
    def test_basic_wrapping(self):
        assert wrap_lon(181.0) == pytest.approx(-179.0)
        assert wrap_lon(-181.0) == pytest.approx(179.0)
        assert wrap_lon(360.0) == pytest.approx(0.0)

    def test_already_in_range(self):
        assert wrap_lon(45.0) == pytest.approx(45.0)
        assert wrap_lon(-90.0) == pytest.approx(-90.0)

    def test_edge_cases(self):
        assert wrap_lon(-180.0) == pytest.approx(-180.0)
        assert wrap_lon(180.0) == pytest.approx(-180.0)

    def test_array(self):
        result = wrap_lon(np.array([0.0, 190.0, -200.0, 360.0]))
        np.testing.assert_allclose(result, [0.0, -170.0, 160.0, 0.0])


class TestClampLat:
    def test_clamping(self):
        assert clamp_lat(95.0) == pytest.approx(90.0)
        assert clamp_lat(-95.0) == pytest.approx(-90.0)

    def test_array(self):
        result = clamp_lat(np.array([-100, -45, 0, 45, 100]))
        np.testing.assert_allclose(result, [-90, -45, 0, 45, 90])


class TestLonLatXyz:
    def test_north_pole(self):
        np.testing.assert_allclose(lonlat_to_xyz(0.0, 90.0), [0, 0, 1], atol=1e-12)

    def test_equator_prime_meridian(self):
        np.testing.assert_allclose(lonlat_to_xyz(0.0, 0.0), [1, 0, 0], atol=1e-12)

    def test_round_trip(self):
        lons = np.array([10.5, -175.3, 0.0, 45.0])
        lats = np.array([45.2, -60.1, 0.0, 89.9])
        lon2, lat2 = xyz_to_lonlat(lonlat_to_xyz(lons, lats))
        np.testing.assert_allclose(lon2, lons, atol=1e-10)
        np.testing.assert_allclose(lat2, lats, atol=1e-10)


class TestMetricZ0:
    def test_equator(self):
        assert metric_z0(0.0) == pytest.approx(METERS_PER_DEG_LAT)

    def test_pole(self):
        assert abs(metric_z0(90.0)) < 1e-10

    def test_60deg(self):
        assert metric_z0(60.0) == pytest.approx(0.5 * METERS_PER_DEG_LAT, rel=1e-10)

    def test_matches_fortran(self):
        """z0 = dcos(yp*rad)*METERS_PER_DEG_LAT from Fortran."""
        lat = 54.3
        assert metric_z0(lat) == pytest.approx(np.cos(lat * DEG2RAD) * METERS_PER_DEG_LAT)


class TestAngularRates:
    def test_stationary(self):
        dlon, dlat = angular_rates(0.0, 0.0, 45.0)
        assert dlon == pytest.approx(0.0)
        assert dlat == pytest.approx(0.0)

    def test_pure_eastward_equator(self):
        dlon, dlat = angular_rates(1.0, 0.0, 0.0)
        assert dlon == pytest.approx(1.0 / METERS_PER_DEG_LAT)
        assert dlat == pytest.approx(0.0)

    def test_pure_northward(self):
        _, dlat = angular_rates(0.0, 1.0, 45.0)
        assert dlat == pytest.approx(1.0 / METERS_PER_DEG_LAT)

    def test_fortran_compatible(self):
        """dlon = U/z0, dlat = V/METERS_PER_DEG_LAT."""
        u, v, lat = 0.5, -0.3, 55.0
        dlon, dlat = angular_rates(u, v, lat)
        z0 = np.cos(lat * DEG2RAD) * METERS_PER_DEG_LAT
        assert dlon == pytest.approx(u / z0)
        assert dlat == pytest.approx(v / METERS_PER_DEG_LAT)


class TestDeltaLon:
    def test_simple(self):
        assert delta_lon(10.0, 5.0) == pytest.approx(5.0)

    def test_across_dateline(self):
        assert delta_lon(-170.0, 170.0) == pytest.approx(20.0)
        assert delta_lon(170.0, -170.0) == pytest.approx(-20.0)

    def test_zero(self):
        assert delta_lon(45.0, 45.0) == pytest.approx(0.0)


class TestGreatCircleDistance:
    def test_zero(self):
        assert great_circle_distance(10.0, 20.0, 10.0, 20.0) == pytest.approx(0.0, abs=1e-6)

    def test_quarter_circle(self):
        d = great_circle_distance(0.0, 0.0, 90.0, 0.0)
        assert d == pytest.approx(np.pi / 2.0 * 6_371_000.0, rel=1e-6)

    def test_antipodal(self):
        d = great_circle_distance(0.0, 90.0, 0.0, -90.0)
        assert d == pytest.approx(np.pi * 6_371_000.0, rel=1e-6)
