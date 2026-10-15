"""Regression tests against stored reference trajectories.

Ensures that code changes (fused kernels, grid hash, etc.) produce
bit-identical results to the validated reference output.

Reference data generated from FESOM-C mesh + 10 particles + 11 Euler steps.
The input mesh/velocity NetCDF (``tests/data/fesomc_combined.nc``) is not
committed; without it these tests skip. ``tests/data/particles.dat`` is.

NOTE: Fortran parity was intentionally broken when METERS_PER_DEG_LAT was
changed from 111300 → 111120 (see geometry.py).  If the reference file is
regenerated from the current code, tests will pass; comparisons with the
original Fortran trajectories will show a systematic ~11m offset.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from kelp.core.mesh import Mesh, build_node_to_elems, build_search
from kelp.core.particles import load_particles
from kelp.numerics.interpolate import VelocityField
from kelp.numerics.integrators import (
    euler_step, rk2_step, rk4_step,
    _euler_step_unfused, _rk2_step_unfused, _rk4_step_unfused,
)
from kelp.core.geometry import METERS_PER_DEG_LAT


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
_DATA = _HERE / "data"
_COMBINED_NC = _DATA / "fesomc_combined.nc"
_PARTICLES = _DATA / "particles.dat"
_FORTRAN_FWD = _DATA / "fortran_tracer_forward.dat"
_REFERENCE = _HERE / "reference_fesomc_euler_forward.npz"

# Skip all tests if data files are missing
pytestmark = pytest.mark.skipif(
    not _COMBINED_NC.exists() or not _PARTICLES.exists(),
    reason="FESOM-C validation data not available",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fesomc_setup():
    """Load FESOM-C mesh, velocity, and initial particles (cached per module)."""
    ds = xr.open_dataset(str(_COMBINED_NC))
    lon = ds["lon"].values.astype(np.float64)
    lat = ds["lat"].values.astype(np.float64)
    depth = ds["depth"].values.astype(np.float64)
    nv = ds["nv"].values.astype(np.int32) - 1
    for i in range(nv.shape[0]):
        if nv[i, 3] == nv[i, 0]:
            nv[i, 3] = -1
    lon_c = ds["lon_elem"].values.astype(np.float64)
    lat_c = ds["lat_elem"].values.astype(np.float64)
    n2e, cnt = build_node_to_elems(len(lon), nv)
    mesh = Mesh("fesomc", lon, lat, depth, nv, lon_c, lat_c, n2e, cnt)
    build_search(mesh)
    state = load_particles(str(_PARTICLES))

    def read_vel(step):
        return VelocityField(
            u=ds["u2d"].isel(time=step).values.astype(np.float64),
            v=ds["v2d"].isel(time=step).values.astype(np.float64),
            eta=ds["eta"].isel(time=step).values.astype(np.float64),
        )

    return mesh, state, read_vel, ds


def _run_steps(state, mesh, read_vel, step_fn, n_steps=11, dt=3600.0, time_dir=1.0):
    """Run n_steps with the given step function, return final state."""
    current = state
    for step in range(n_steps):
        vel = read_vel(step)
        current = step_fn(current, vel, mesh, dt, time_dir)
    return current


def _spatial_dist(lon1, lat1, lon2, lat2):
    """Compute spatial distance in metres between two positions."""
    cos_lat = np.cos(np.radians(0.5 * (lat1 + lat2)))
    dlon_m = (lon1 - lon2) * cos_lat * METERS_PER_DEG_LAT
    dlat_m = (lat1 - lat2) * METERS_PER_DEG_LAT
    return np.sqrt(dlon_m ** 2 + dlat_m ** 2)


def _read_fortran_output(path, n_particles=10):
    """Read Fortran tracer output file."""
    data = []
    with open(path) as f:
        for line in f:
            vals = [float(x) for x in line.split()]
            data.append(vals)
    arr = np.array(data)
    return arr[:, 0::3], arr[:, 1::3]  # lon, lat


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFortranComparison:
    """Compare Python tracking against Fortran reference output.

    DISABLED: parity with the original Fortran tracker was intentionally
    broken when METERS_PER_DEG_LAT changed from 111300 (Fortran SLM) to
    111120 (nautical-mile convention used by Parcels and most modern
    oceanographic tooling).  This introduces a systematic ~11.6 m offset
    over 2 h of tracking at 1 m/s, which violates the historical < 10 m
    threshold.

    To re-enable for archival comparison, override METERS_PER_DEG_LAT
    locally to 111300 and adjust the threshold accordingly.
    """

    @pytest.mark.skip(reason="Fortran parity broken by METERS_PER_DEG_LAT change (111300→111120)")
    def test_forward_euler_vs_fortran(self, fesomc_setup):
        mesh, state, read_vel, _ = fesomc_setup
        current = _run_steps(state, mesh, read_vel, euler_step)

        f_lon, f_lat = _read_fortran_output(str(_FORTRAN_FWD))

        dists = _spatial_dist(current.lon, current.lat, f_lon[-1], f_lat[-1])
        max_dist = dists.max()

        assert max_dist < 10.0, (
            f"Max distance to Fortran = {max_dist:.1f} m (expected < 10 m)"
        )


class TestReferenceRegression:
    """Bit-exact regression against stored reference trajectories.

    If this test fails after a code change, it means the tracking output
    has changed.  Either the change is intentional (update the reference)
    or a bug was introduced.
    """

    @pytest.mark.skipif(not _REFERENCE.exists(), reason="Reference file not generated")
    def test_euler_forward_matches_reference(self, fesomc_setup):
        mesh, state, read_vel, _ = fesomc_setup

        ref = np.load(str(_REFERENCE))
        ref_lon = ref["lon"]  # (12, 10): initial + 11 steps
        ref_lat = ref["lat"]
        ref_status = ref["status"]

        current = state
        for step in range(11):
            vel = read_vel(step)
            current = euler_step(current, vel, mesh, 3600.0, 1.0)

            # Check each step matches exactly
            np.testing.assert_array_equal(
                current.lon, ref_lon[step + 1],
                err_msg=f"Longitude mismatch at step {step + 1}",
            )
            np.testing.assert_array_equal(
                current.lat, ref_lat[step + 1],
                err_msg=f"Latitude mismatch at step {step + 1}",
            )

        np.testing.assert_array_equal(current.status, ref_status)


class TestFusedVsUnfused:
    """Verify fused kernels produce identical results to unfused fallback."""

    def test_euler_fused_vs_unfused(self, fesomc_setup):
        mesh, state, read_vel, _ = fesomc_setup
        s_fused = _run_steps(state, mesh, read_vel, euler_step)
        s_unfused = _run_steps(
            state, mesh, read_vel,
            lambda s, v, m, dt, td: _euler_step_unfused(s, v, m, dt, td, 0, False),
        )
        np.testing.assert_array_equal(s_fused.lon, s_unfused.lon)
        np.testing.assert_array_equal(s_fused.lat, s_unfused.lat)
        np.testing.assert_array_equal(s_fused.status, s_unfused.status)

    def test_rk2_fused_vs_unfused(self, fesomc_setup):
        mesh, state, read_vel, _ = fesomc_setup
        s_fused = _run_steps(state, mesh, read_vel, rk2_step)
        s_unfused = _run_steps(
            state, mesh, read_vel,
            lambda s, v, m, dt, td: _rk2_step_unfused(s, v, m, dt, td, 0, False),
        )
        np.testing.assert_array_equal(s_fused.lon, s_unfused.lon)
        np.testing.assert_array_equal(s_fused.lat, s_unfused.lat)
        np.testing.assert_array_equal(s_fused.status, s_unfused.status)

    def test_rk4_fused_vs_unfused(self, fesomc_setup):
        mesh, state, read_vel, _ = fesomc_setup
        s_fused = _run_steps(state, mesh, read_vel, rk4_step)
        s_unfused = _run_steps(
            state, mesh, read_vel,
            lambda s, v, m, dt, td: _rk4_step_unfused(s, v, m, dt, td, 0, False),
        )
        np.testing.assert_array_equal(s_fused.lon, s_unfused.lon)
        np.testing.assert_array_equal(s_fused.lat, s_unfused.lat)
        np.testing.assert_array_equal(s_fused.status, s_unfused.status)
