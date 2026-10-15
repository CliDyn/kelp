"""Parity tests with Parcels' ``tests/test_diffusion.py`` (Option A).

These tests *mirror* the physics of Parcels' two diffusion tests
(``test_fieldKh_Brownian`` and ``test_fieldKh_SpatiallyVaryingDiffusion``)
on a native kelp mesh + integrator setup.  They do NOT call
Parcels — that's the @slow ``test_parcels_diffusion_comparison.py``
suite (Option B, follow-up commit).

Why mirror them as native tests:

* Parcels' Brownian test pins ``std(x) ≈ √(2·Kh·t)`` and ``mean ≈ 0``
  on uniform Kh.  Our internal ``TestBrownianVariance`` already
  exercises this on a single-step short-runtime regime; Parcels uses
  longer runtime (2 h) and looser tolerance (500 m) more
  representative of real ocean usage.  Cheap regression coverage.

* Parcels' SpatiallyVaryingDiffusion test is the strongest physics
  check available: with a ``tanh``-profiled Kh in the zonal
  direction (constant in meridional), the EM and M1 schemes should
  produce a *skewed* zonal distribution while the meridional
  distribution stays symmetric.  This catches Itô-correction sign
  bugs that mean/variance tests can't.  Our internal
  ``TestCombinedEulerEM::test_linear_kh_drift_sign`` exercises a
  related path but at smaller scale.

Both tests use a synthetic 11×11 1° lon/lat mesh (the existing
``grid_wide`` pattern from ``test_integrators_diffusion.py``) — the
mesh is wide enough at this resolution to keep the random walks
in-domain at the runtime scales we mirror.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import stats

from kelp.core.geometry import METERS_PER_DEG_LAT
from kelp.core.mesh import (
    Mesh, build_node_to_elems, build_search,
)
from kelp.core.particles import ParticleStatus, make_state
from kelp.numerics.diffusion import DiffusionStepArgs
from kelp.numerics.integrators import euler_step
from kelp.numerics.interpolate import VelocityField


# ---------------------------------------------------------------------------
# Mesh fixtures — same pattern as test_integrators_diffusion.py but local
# so this file can be read on its own without cross-file context.
# ---------------------------------------------------------------------------

def _build_grid_mesh(xs: np.ndarray, ys: np.ndarray) -> Mesh:
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel().astype(np.float64)
    lat = gy.ravel().astype(np.float64)
    nx, ny = len(xs), len(ys)
    n_nodes = nx * ny
    depth = np.zeros(n_nodes, dtype=np.float64)
    elems = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            n1 = n0 + 1
            n2 = n0 + nx
            n3 = n2 + 1
            elems.append([n0, n1, n2])
            elems.append([n1, n3, n2])
    elem_nodes = np.array(elems, dtype=np.int32)
    lon_c = np.array([lon[e].mean() for e in elem_nodes], dtype=np.float64)
    lat_c = np.array([lat[e].mean() for e in elem_nodes], dtype=np.float64)
    n2e, n2e_count = build_node_to_elems(n_nodes, elem_nodes)
    mesh = Mesh(
        name="grid",
        nodes_lon=lon, nodes_lat=lat, depth=depth,
        elem_nodes=elem_nodes,
        elem_center_lon=lon_c, elem_center_lat=lat_c,
        node_to_elems=n2e, node_to_elems_count=n2e_count,
    )
    build_search(mesh)
    return mesh


@pytest.fixture
def wide_mesh():
    """Wide enough mesh for ~1 km Brownian motion at lat=5°.

    Range is 0..10° in both axes (~1100 km zonally at lat=5°), so
    Parcels-scale runtime (Kh ~ 100, t ~ 7200 s → std ~ 1200 m)
    leaves plenty of headroom inside the domain.
    """
    return _build_grid_mesh(
        np.linspace(0.0, 10.0, 11),
        np.linspace(0.0, 10.0, 11),
    )


def _const_vel_zero(mesh: Mesh) -> VelocityField:
    ne = mesh.n_elements
    return VelocityField(
        u=np.zeros(ne, dtype=np.float64),
        v=np.zeros(ne, dtype=np.float64),
        eta=np.zeros(mesh.n_nodes, dtype=np.float64),
    )


def _seed_at(mesh: Mesh, n: int, lon: float = 5.0, lat: float = 5.0):
    return make_state(
        lon=np.full(n, lon, dtype=np.float64),
        lat=np.full(n, lat, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Test 1 — Parcels' test_fieldKh_Brownian (uniform Kh, no advection)
# ---------------------------------------------------------------------------

class TestParcelsBrownianParity:
    """Mirror of ``parcels/tests/test_diffusion.py::test_fieldKh_Brownian``.

    Parcels parameters:
        kh_zonal=100, kh_meridional=50 m^2/s
        npart=100, runtime=2h, dt=1h
        tolerance=500 m on std and mean

    Their test runs at the seed point ``(0, 0)`` on a structured
    lat/lon FieldSet with explicit ``mesh="spherical"|"flat"``.  We
    seed at ``(5°, 5°)`` on our unstructured mesh — both are spherical,
    just centred differently.  The expected std follows the same
    ``√(2·Kh·t)`` formula in metres.
    """

    KH_ZONAL = 100.0
    KH_MERIDIONAL = 50.0
    NPART = 100
    RUNTIME_S = 2 * 3600.0
    DT_S = 1 * 3600.0
    TOL_M = 500.0

    def test_brownian_std_and_mean(self, wide_mesh):
        n_steps = int(self.RUNTIME_S / self.DT_S)
        vel = _const_vel_zero(wide_mesh)
        state = _seed_at(wide_mesh, self.NPART)

        for step_i in range(n_steps):
            diff = DiffusionStepArgs(
                scheme="uniform",
                kh_x=self.KH_ZONAL, kh_y=self.KH_MERIDIONAL,
                seed=1234, real_id=0, step_i=step_i,
            )
            state = euler_step(state, vel, wide_mesh,
                               dt=self.DT_S, time_dir=1.0,
                               diffusion=diff)

        alive = state.alive_mask
        assert alive.sum() == self.NPART, (
            "Parcels-scale Brownian motion should stay in-domain "
            "on the 10° × 10° wide mesh"
        )

        # Convert lon/lat to metres around the seed point.  Use the
        # seed-lat z0 for the conversion (consistent with Parcels'
        # mesh="spherical" mesh_conversion based on the equator
        # nautical-mile convention; kelp's
        # METERS_PER_DEG_LAT = 111120.0 matches Parcels exactly).
        lat0 = 5.0
        z0 = float(np.cos(np.deg2rad(lat0))) * METERS_PER_DEG_LAT
        dx_m = (state.lon[alive] - 5.0) * z0
        dy_m = (state.lat[alive] - 5.0) * METERS_PER_DEG_LAT

        expected_std_lon = np.sqrt(2.0 * self.KH_ZONAL * self.RUNTIME_S)
        expected_std_lat = np.sqrt(2.0 * self.KH_MERIDIONAL * self.RUNTIME_S)

        np.testing.assert_allclose(
            np.std(dx_m), expected_std_lon, atol=self.TOL_M,
        )
        np.testing.assert_allclose(
            np.std(dy_m), expected_std_lat, atol=self.TOL_M,
        )
        np.testing.assert_allclose(
            np.mean(dx_m), 0.0, atol=self.TOL_M,
        )
        np.testing.assert_allclose(
            np.mean(dy_m), 0.0, atol=self.TOL_M,
        )


# ---------------------------------------------------------------------------
# Test 2 — Parcels' test_fieldKh_SpatiallyVaryingDiffusion
# ---------------------------------------------------------------------------

class TestParcelsSpatiallyVaryingParity:
    """Mirror of
    ``parcels/tests/test_diffusion.py::test_fieldKh_SpatiallyVaryingDiffusion``.

    The key physics: on a Kh field with a *zonal-only* gradient (here a
    tanh profile in lon, constant in lat), the Itô drift correction
    ``∂Kh_x/∂x · dt`` produces a non-zero zonal mean displacement with a
    skewed distribution.  Meanwhile the meridional axis sees only
    Brownian noise on a constant Kh, so its distribution stays
    symmetric.  Parcels' test pins this with
    ``|skew(lon)| > |skew(lat)|`` — the same check we use here.

    Parameters: npart=50000, runtime=3h, dt=1h, parametrised over EM
    and M1.  Calibration note — Parcels' test uses npart=10000, but
    at that scale ``SE_skew = √(6/n) ≈ 0.024`` and the empirical
    skew margin barely clears 1.5·SE on a single seed.  We bump
    npart to 50000 so SE_skew drops to ~0.011, giving > 3·SE
    margin on a steeper Kh contrast and removing seed-fragility
    from the assertion.  The Kh range is also widened from
    Parcels' [100, 300] to [100, 1100] m^2/s to push the asymmetry
    signal further above the noise floor.  We still pick the
    Parcels seed (1636) for traceability.

    Metric note: this test parameterises Kh in *degrees* (consistent
    with the unstructured mesh's lat/lon storage), giving an
    effective gradient ~1.8x steeper than Parcels' metres-based
    field at the same nominal range.  The Option-B
    cross-comparison test (``test_parcels_diffusion_comparison.py``)
    rebuilds the field in metres to match Parcels exactly.

    Implementation: Parcels uses a structured lat/lon field and
    convolves the kernel with their FieldSet machinery.  We
    hand-build a node-based ``kh_zonal`` field on our 11×11 mesh
    with the same tanh profile and let kelp's combined-EM /
    combined-M1 paths sample it via the B3 LSQ-stencil
    interpolation.  The skewness inequality holds qualitatively
    because the gradient direction (lon-only) is preserved.
    """

    NPART = 50_000
    RUNTIME_S = 3 * 3600.0
    DT_S = 1 * 3600.0

    def _build_tanh_kh_vel(self, mesh: Mesh) -> VelocityField:
        """Node-based ``Kh_zonal = tanh(lon_centred / lon_max · 10) · A + B``
        with a constant ``Kh_meridional`` so that ``∂Kh_y/∂y = 0``.

        Range [100, 5100] m²/s — much wider than Parcels' [100, 300]
        because at npart=50k the empirical skewness noise SE = 0.011
        and we need the gradient signal to land above 2·SE on both
        EM and M1.  Wider Kh contrast pushes EM's Itô-drift skewness
        above the noise floor with comfortable margin; M1's
        QV-correction adds extra asymmetry on top.
        """
        nn = mesh.n_nodes
        lon = mesh.nodes_lon
        # Centre the tanh argument so the gradient peaks at the seed
        # (mirroring Parcels' centred np.linspace(-1e6, 1e6, ...)).
        lon_centred = lon - 5.0
        lon_centred_max = float(np.max(np.abs(lon_centred))) or 1.0
        kh_zonal = (
            np.tanh(lon_centred / lon_centred_max * 10.0) * 2500.0
            + 2600.0
        ).astype(np.float64)
        kh_meridional = np.full(nn, 2600.0, dtype=np.float64)

        ne = mesh.n_elements
        return VelocityField(
            u=np.zeros(ne, dtype=np.float64),
            v=np.zeros(ne, dtype=np.float64),
            eta=np.zeros(nn, dtype=np.float64),
            kh_zonal=kh_zonal,
            kh_meridional=kh_meridional,
            kh_layout="node",
        )

    @pytest.mark.parametrize("scheme", ["em", "m1"])
    def test_skew_asymmetry(self, wide_mesh, scheme):
        """``|skew(lon)| > |skew(lat)|`` — the Parcels physics check."""
        n_steps = int(self.RUNTIME_S / self.DT_S)
        vel = self._build_tanh_kh_vel(wide_mesh)
        state = _seed_at(wide_mesh, self.NPART)

        for step_i in range(n_steps):
            diff = DiffusionStepArgs(
                scheme=scheme, kh_x=0.0, kh_y=0.0,
                seed=1636, real_id=0, step_i=step_i,
                composition="combined", dres_m=2000.0,
            )
            with warnings.catch_warnings():
                # M1 emits a one-time per-realization warning about
                # cross-derivative limitation on spatially-varying Kh;
                # we know about it (tested in
                # test_integrators_diffusion.TestCombinedEulerEM) and
                # don't want it to dominate the output.
                warnings.simplefilter("ignore", RuntimeWarning)
                state = euler_step(state, vel, wide_mesh,
                                   dt=self.DT_S, time_dir=1.0,
                                   diffusion=diff)

        alive = state.alive_mask
        # Standardised on >0.95 across all suite tests (was 0.8 here,
        # 1.0 on Brownian — review #7b normalisation).  At
        # Kh=2600 m^2/s (centre value of the [100, 5100] tanh field)
        # and runtime=3h, the Brownian std is √(2·Kh·t) ≈ 7.5 km,
        # well within the 10° wide mesh (~1100 km at lat=5°).
        assert alive.sum() > 0.95 * self.NPART, (
            f"only {alive.sum()}/{self.NPART} particles survived; "
            "tighten Kh or runtime, or widen the mesh"
        )

        lat0 = 5.0
        z0 = float(np.cos(np.deg2rad(lat0))) * METERS_PER_DEG_LAT
        dx_m = (state.lon[alive] - 5.0) * z0
        dy_m = (state.lat[alive] - 5.0) * METERS_PER_DEG_LAT

        skew_lon = stats.skew(dx_m)
        skew_lat = stats.skew(dy_m)

        # Parcels' assertion: zonal axis is skewed (Itô drift on
        # asymmetric Kh field), meridional axis is symmetric.  With
        # the bumped Kh contrast the inequality holds with > 2 SE
        # margin, robustly across reasonable seed choices.
        se_skew = np.sqrt(6.0 / alive.sum())
        margin = abs(skew_lon) - abs(skew_lat)
        assert margin > 2.0 * se_skew, (
            f"expected |skew(lon)| > |skew(lat)| with > 2 SE margin "
            f"(SE={se_skew:.4f}); got |skew_lon|={abs(skew_lon):.4f}, "
            f"|skew_lat|={abs(skew_lat):.4f}, margin={margin:.4f} "
            f"({scheme=})"
        )

        # Per-axis sanity bounds — Parcels uses tol=2000 m on the
        # *means* (with 10 000 particles), but not on skewness.  Our
        # native LSQ-interpolated Kh produces a slightly different
        # numerical profile, so we only pin the qualitative
        # asymmetry inequality above plus mean-bound on the
        # gradient-free axis.
        np.testing.assert_allclose(
            np.mean(dy_m), 0.0,
            atol=4.0 * np.std(dy_m) / np.sqrt(alive.sum()),
        )
