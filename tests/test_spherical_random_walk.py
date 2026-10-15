"""Random walk on the sphere (review 2026-09-22, M04).

A longitude-latitude random walk ``dlon = dx/(R cos lat)``, ``dlat = dy/R``
misses the metric drift ``-K tan(lat)/R`` of the Laplace-Beltrami operator:
its generator acting on ``f = sin(lat)`` is ``-K sin(lat)/R^2`` instead of
``-2 K sin(lat)/R^2`` — the chart's invariant density is uniform in latitude
instead of uniform in area.  The kick is now applied along the great circle
(the exponential map of the tangent displacement), which is rotation-invariant
for isotropic constant K (uniform area density kept exactly) and passes
through the pole; an anisotropic diagonal
tensor gets the extra northward drift ``(Kx - Ky) tan(lat)/R``.
"""
import numpy as np
import pytest

from kelp.core.geometry import DEG2RAD, METERS_PER_DEG_LAT
from kelp.numerics.diffusion import (anisotropic_metric_drift_m,
                                     sphere_displace, sphere_displace_arrays)

R = METERS_PER_DEG_LAT / DEG2RAD


def test_north_kick_follows_the_meridian():
    lon, lat = sphere_displace(10.0, 45.0, 0.0, 1000.0)
    assert lon == pytest.approx(10.0, abs=1e-12)
    assert lat == pytest.approx(45.0 + 1000.0 / METERS_PER_DEG_LAT, abs=1e-12)


def test_kick_crosses_the_pole():
    # 0.01 deg from the pole, 2 km northward: 1111.2 m to the pole, then
    # 888.8 m down the opposite meridian
    lon, lat = sphere_displace(10.0, 89.99, 0.0, 2000.0)
    assert lat == pytest.approx(90.0 - (2000.0 - 0.01 * METERS_PER_DEG_LAT)
                                / METERS_PER_DEG_LAT, abs=1e-9)
    assert lon == pytest.approx(-170.0, abs=1e-6)


def test_kick_from_exactly_the_pole():
    """Review 2026-09-23: the bearing formula degenerated at ±90°.
    At the pole the stored longitude defines the local east/north basis, so
    a northward kick continues down the opposite meridian."""
    d = 1000.0
    lon, lat = sphere_displace(10.0, 90.0, 0.0, d)
    assert lat == pytest.approx(90.0 - d / METERS_PER_DEG_LAT, abs=1e-9)
    assert lon == pytest.approx(-170.0, abs=1e-6)
    lon, lat = sphere_displace(10.0, 90.0, 0.0, -d)
    assert lat == pytest.approx(90.0 - d / METERS_PER_DEG_LAT, abs=1e-9)
    assert lon == pytest.approx(10.0, abs=1e-6)


def test_east_kick_bends_equatorward():
    # a great circle leaving 60N heading east ends south of 60N
    _lon, lat = sphere_displace(0.0, 60.0, 50_000.0, 0.0)
    assert lat < 60.0


def test_zero_kick_is_the_identity():
    assert sphere_displace(-179.5, -33.0, 0.0, 0.0) == (-179.5, -33.0)


def _mean_dsinlat(kx, ky, lat0, h, n=400_000, seed=3):
    rng = np.random.default_rng(seed)
    dx = np.sqrt(2 * kx * h) * rng.standard_normal(n)
    dy = np.sqrt(2 * ky * h) * rng.standard_normal(n)
    dy = dy + anisotropic_metric_drift_m(kx, ky, lat0, h)
    lon = np.zeros(n)
    lat = np.full(n, lat0)
    alive = np.ones(n, dtype=np.bool_)
    lon2, lat2 = sphere_displace_arrays(lon, lat, dx, dy, alive)
    d = np.sin(lat2 * DEG2RAD) - np.sin(lat0 * DEG2RAD)
    return d.mean(), d.std() / np.sqrt(n)


@pytest.mark.parametrize("kx_over_ky", [1.0, 4.0])
def test_generator_of_sin_lat(kx_over_ky):
    """E[d sin(lat)] = -2 Ky h sin(lat) / R^2 (Laplace-Beltrami; only the
    meridional diffusivity enters for a zonally uniform f).  The chart walk
    gave half of it for the isotropic case."""
    lat0, h = 60.0, 1.0
    ky = 1e-4 * R ** 2 / h            # Ky h / R^2 = 1e-4 (visible in MC)
    kx = kx_over_ky * ky
    mean, se = _mean_dsinlat(kx, ky, lat0, h)
    expected = -2.0 * 1e-4 * np.sin(lat0 * DEG2RAD)
    assert abs(mean - expected) < 4.0 * se + 5e-3 * abs(expected), (mean, se)


def test_isotropic_drift_correction_is_zero():
    assert anisotropic_metric_drift_m(3.0, 3.0, 70.0, 600.0) == 0.0


def test_split_kick_moves_along_the_great_circle():
    """The integrator's uniform kick = the exponential map of the raw
    counter-RNG kick (same key), from the post-deterministic position."""
    from kelp.core.io import VelocityField
    from kelp.core.particles import make_state
    from kelp.numerics.diffusion import DiffusionStepArgs, _fill_uniform_kicks
    from kelp.numerics.integrators import euler_step
    from tests.test_integrators_diffusion import _build_grid_mesh
    mesh = _build_grid_mesh(np.linspace(0.0, 10.0, 11),
                            np.linspace(50.0, 60.0, 11))
    n, kh, dt = 64, 50.0, 3600.0
    vel = VelocityField(u=np.zeros(mesh.n_elements),
                        v=np.zeros(mesh.n_elements),
                        eta=np.zeros(mesh.n_nodes))
    st = make_state(np.full(n, 5.0), np.full(n, 55.0))
    diff = DiffusionStepArgs(scheme="uniform", kh_x=kh, kh_y=kh, seed=9,
                             real_id=0, step_i=3)
    out = euler_step(st, vel, mesh, dt=dt, time_dir=1.0, diffusion=diff)
    dx, dy = _fill_uniform_kicks(kh, kh, dt, np.uint64(9), np.uint64(0),
                                 np.uint64(3), np.ones(n, dtype=np.bool_), n)
    lon_e, lat_e = sphere_displace_arrays(st.lon, st.lat, dx, dy,
                                          np.ones(n, dtype=np.bool_))
    np.testing.assert_allclose(out.lon, lon_e, rtol=0, atol=1e-12)
    np.testing.assert_allclose(out.lat, lat_e, rtol=0, atol=1e-12)
