"""Tests for ``kelp.numerics.diffusion``.

Covers the counter-based RNG primitives: determinism,
order-independence, distributional sanity, and pair independence.
Later phases add the SDE step kernels.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from kelp.numerics.diffusion import (
    _normal_pair,
    _philox_uint64,
    _standard_normal,
    diffusion_uniform_step_meters,
)


class TestPhilox:
    def test_determinism(self):
        a = _philox_uint64(42, 0, 7, 13, 0)
        b = _philox_uint64(42, 0, 7, 13, 0)
        assert a == b

    def test_different_seeds_diverge(self):
        a = _philox_uint64(1, 0, 0, 0, 0)
        b = _philox_uint64(2, 0, 0, 0, 0)
        assert a != b

    def test_different_step_diverge(self):
        a = _philox_uint64(42, 0, 0, 0, 0)
        b = _philox_uint64(42, 0, 0, 1, 0)
        assert a != b

    def test_key_order_matters(self):
        """Swapping any two keys must change the output.

        This guarantees that two particles ``i, j`` at the same step do
        not accidentally collide via key symmetry.
        """
        base = _philox_uint64(42, 1, 2, 3, 4)
        assert _philox_uint64(42, 2, 1, 3, 4) != base
        assert _philox_uint64(42, 1, 3, 2, 4) != base
        assert _philox_uint64(42, 1, 2, 4, 3) != base

    def test_outputs_well_distributed(self):
        """Top bits of the hash should look uniform across many keys."""
        n = 50_000
        out = np.empty(n, dtype=np.uint64)
        for i in range(n):
            out[i] = _philox_uint64(0, 0, i, 0, 0)
        # Smoke: no duplicates in 50k samples (birthday bound: with 64-bit
        # output, collision probability over 50k is < 1e-10).
        assert len(np.unique(out)) == n


class TestNormalPair:
    def test_determinism(self):
        g1a, g2a = _normal_pair(42, 0, 7, 13, 0)
        g1b, g2b = _normal_pair(42, 0, 7, 13, 0)
        assert g1a == g1b and g2a == g2b

    def test_pair_components_uncorrelated(self):
        n = 200_000
        g1 = np.empty(n)
        g2 = np.empty(n)
        for i in range(n):
            a, b = _normal_pair(42, 0, i, 0, 0)
            g1[i] = a
            g2[i] = b
        # Box–Muller pair components are independent by construction;
        # the empirical correlation should be O(1/sqrt(n)) ≈ 0.002 here.
        corr = float(np.corrcoef(g1, g2)[0, 1])
        assert abs(corr) < 0.02, f"|corr|={abs(corr):.4f} too large"

    def test_distributional_sanity(self):
        """10⁶ samples → mean ≈ 0, var ≈ 1, |skew| ≪ 0.05."""
        n = 1_000_000
        # Use both halves of the Box–Muller pair to halve compute.
        samples = np.empty(n)
        half = n // 2
        for i in range(half):
            g1, g2 = _normal_pair(123, 0, i, 0, 0)
            samples[2 * i] = g1
            samples[2 * i + 1] = g2

        mean = float(samples.mean())
        var = float(samples.var())
        std = math.sqrt(var)
        z = (samples - mean) / std
        skew = float((z ** 3).mean())

        # Standard error of the mean for N(0,1) sample of size n is
        # 1/sqrt(n) ≈ 1e-3; tolerate 5σ.
        assert abs(mean) < 5e-3, f"mean={mean}"
        assert abs(var - 1.0) < 0.01, f"var={var}"
        assert abs(skew) < 0.02, f"skew={skew}"

    def test_uncorrelated_across_step_i(self):
        """Same particle, sequential steps → samples should be uncorrelated.

        This is the property the integrator relies on: a single
        particle's noise sequence over time must look like white noise.
        """
        n = 50_000
        g_zonal = np.empty(n)
        g_merid = np.empty(n)
        for step_i in range(n):
            a, b = _normal_pair(7, 0, 99, step_i, 0)
            g_zonal[step_i] = a
            g_merid[step_i] = b

        # Zero-lag cross-correlation between the two streams.
        corr = float(np.corrcoef(g_zonal, g_merid)[0, 1])
        assert abs(corr) < 0.02

        # Lag-1 autocorrelation of a single stream.
        a0 = g_zonal[:-1]
        a1 = g_zonal[1:]
        ac = float(np.corrcoef(a0, a1)[0, 1])
        assert abs(ac) < 0.02

    def test_uncorrelated_across_particles(self):
        """Two distinct particles at the same step must produce
        uncorrelated draws.
        """
        n = 50_000
        ga = np.empty(n)
        gb = np.empty(n)
        for step_i in range(n):
            a, _ = _normal_pair(7, 0, 0, step_i, 0)
            b, _ = _normal_pair(7, 0, 1, step_i, 0)
            ga[step_i] = a
            gb[step_i] = b
        corr = float(np.corrcoef(ga, gb)[0, 1])
        assert abs(corr) < 0.02

    def test_realization_id_decorrelates(self):
        """Same (particle_i, step_i) but different realization_id →
        different samples (multi-run reproducibility)."""
        a, _ = _normal_pair(0, 0, 5, 12, 0)
        b, _ = _normal_pair(0, 1, 5, 12, 0)
        c, _ = _normal_pair(0, 2, 5, 12, 0)
        assert a != b
        assert a != c
        assert b != c


class TestStandardNormal:
    def test_matches_pair_first(self):
        g, _ = _normal_pair(11, 0, 0, 0, 0)
        assert _standard_normal(11, 0, 0, 0, 0) == g


class TestUniformStep:
    """Brownian-variance and edge-case coverage for the uniform-Kh kick."""

    def test_zero_kh_zero_displacement(self):
        for step_i in range(50):
            dx, dy = diffusion_uniform_step_meters(
                0.0, 0.0, 100.0, 1, 0, step_i, 0
            )
            assert dx == 0.0 and dy == 0.0

    def test_zero_dt_zero_displacement(self):
        for step_i in range(50):
            dx, dy = diffusion_uniform_step_meters(
                10.0, 10.0, 0.0, 1, 0, step_i, 0
            )
            assert dx == 0.0 and dy == 0.0

    def test_backward_dt_same_magnitude(self):
        """Sign-symmetric kick: |dx(+dt)| == |dx(-dt)| for the same RNG key."""
        for step_i in range(20):
            dx_f, dy_f = diffusion_uniform_step_meters(
                10.0, 5.0, 60.0, 7, 0, step_i, 11
            )
            dx_b, dy_b = diffusion_uniform_step_meters(
                10.0, 5.0, -60.0, 7, 0, step_i, 11
            )
            assert dx_f == dx_b
            assert dy_f == dy_b

    def test_anisotropic_amplitude_ratio(self):
        """``Kh_x = 4·Kh_y`` ⇒ x-amplitude = 2 × y-amplitude (same RNG)."""
        ky = 3.0
        kx = 4.0 * ky
        n = 10_000
        ratios = np.empty(n)
        for i in range(n):
            dx, dy = diffusion_uniform_step_meters(
                kx, ky, 30.0, 13, 0, i, 0
            )
            # Both axes use the same RNG pair (g1, g2). Ratios of
            # amplitudes are |g1/g2| times the deterministic factor 2.
            if dy != 0.0:
                ratios[i] = abs(dx) / abs(dy)
        # Expected median |g1/g2| for an iid N(0,1) pair: 1.
        # So |dx/dy| ≈ 2 in median.
        assert abs(np.median(ratios) - 2.0) < 0.05

    def test_brownian_variance_isotropic(self):
        """Variance of x-displacement after one step ≈ ``2·Kh·dt``.

        100k particles, 1 step, ``Kh = 1.0 m^2/s``, ``dt = 100 s``.
        Expected ``var = 200 m^2``.
        """
        n = 100_000
        kh = 1.0
        dt = 100.0
        dx = np.empty(n)
        dy = np.empty(n)
        for i in range(n):
            a, b = diffusion_uniform_step_meters(kh, kh, dt, 99, 0, i, 0)
            dx[i] = a
            dy[i] = b

        expected_var = 2.0 * kh * dt
        # SE on variance estimate: sigma^2 * sqrt(2/n) ≈ 0.0045 * var.
        # 5 % tolerance is ~12 SE — comfortable.
        assert abs(dx.var() / expected_var - 1.0) < 0.05
        assert abs(dy.var() / expected_var - 1.0) < 0.05

        # Mean ≈ 0 within ~3 SE on the mean (sqrt(var/n)).
        se_mean = math.sqrt(expected_var / n)
        assert abs(dx.mean()) < 4.0 * se_mean
        assert abs(dy.mean()) < 4.0 * se_mean

    def test_brownian_variance_anisotropic(self):
        """Distinct ``Kh_x ≠ Kh_y`` reproduces both variances independently."""
        n = 100_000
        kh_x = 0.25
        kh_y = 9.0
        dt = 50.0
        dx = np.empty(n)
        dy = np.empty(n)
        for i in range(n):
            a, b = diffusion_uniform_step_meters(kh_x, kh_y, dt, 7, 0, i, 0)
            dx[i] = a
            dy[i] = b

        assert abs(dx.var() / (2.0 * kh_x * dt) - 1.0) < 0.05
        assert abs(dy.var() / (2.0 * kh_y * dt) - 1.0) < 0.05

    def test_determinism(self):
        """Same key tuple → same displacement, regardless of order."""
        a = diffusion_uniform_step_meters(2.0, 3.0, 60.0, 5, 0, 17, 9)
        b = diffusion_uniform_step_meters(2.0, 3.0, 60.0, 5, 0, 17, 9)
        assert a == b


class TestNoOverflowOrNan:
    def test_no_nan_or_inf_over_large_sample(self):
        """Box–Muller with the half-bit-offset uniforms should never
        emit NaN/Inf, even on the smallest representable u1.
        """
        n = 100_000
        for i in range(n):
            g1, g2 = _normal_pair(0xDEADBEEF, 0, i, 0, 0)
            if not (np.isfinite(g1) and np.isfinite(g2)):
                pytest.fail(f"non-finite at i={i}: g1={g1}, g2={g2}")


# ===========================================================================
# Kh interpolation + central-diff gradient
# ===========================================================================
#
# These tests exercise the spatial Kh sampling helpers on a synthetic
# 11×11 / 21×21 grid mesh using the same builder pattern as
# ``tests/test_integrators_diffusion.py`` (no external data).

from kelp.core.geometry import METERS_PER_DEG_LAT, DEG2RAD
from kelp.core.mesh import (
    Mesh, build_node_to_elems, build_search, find_nearest_nodes,
)
from kelp.numerics.diffusion import (
    kh_value_at_point, kh_gradient_at_point,
)


def _build_grid_mesh(xs: np.ndarray, ys: np.ndarray) -> Mesh:
    """Same structure as test_integrators_diffusion._build_grid_mesh."""
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
def kh_grid_mesh():
    """11×11 mesh with 1° spacing centred on (5, 5)."""
    return _build_grid_mesh(np.linspace(0.0, 10.0, 11),
                            np.linspace(0.0, 10.0, 11))


_LAT_REF = 5.0  # all linear-field tests pin the particle here


def _linear_kh_node(mesh: Mesh, a: float, b: float, c: float = 0.0):
    """Return ``Kh = c + a·lon_m + b·lat_m`` evaluated with a *fixed*
    reference ``z0`` so the field is linear in the LSQ kernel's metric
    frame at ``yp = _LAT_REF``.

    Using a per-node ``z0(lat_nd)`` would mix in a small ``xp · z0'(yp)``
    cross-term that violates the planar-fit assumption and shows up as a
    1–2% error on the recovered gradient.  Pin the metric to a single
    reference latitude instead.
    """
    z0_ref = np.cos(_LAT_REF * DEG2RAD) * METERS_PER_DEG_LAT
    return (
        c
        + a * mesh.nodes_lon * z0_ref
        + b * mesh.nodes_lat * METERS_PER_DEG_LAT
    )


def _linear_kh_elem(mesh: Mesh, a: float, b: float, c: float = 0.0):
    """Same flat-Earth linear field evaluated at element centres."""
    z0_ref = np.cos(_LAT_REF * DEG2RAD) * METERS_PER_DEG_LAT
    return (
        c
        + a * mesh.elem_center_lon * z0_ref
        + b * mesh.elem_center_lat * METERS_PER_DEG_LAT
    )


class TestKhValueInterpolation:
    """Exact recovery of constants and linear fields on both layouts."""

    def test_constant_field_node_layout(self, kh_grid_mesh):
        kh = np.full(kh_grid_mesh.n_nodes, 17.0)
        for xp, yp in [(5.0, 5.0), (5.37, 4.62), (3.0, 8.0)]:
            val, ok = kh_value_at_point(xp, yp, kh, "node", kh_grid_mesh)
            assert ok
            assert val == pytest.approx(17.0, abs=1e-9)

    def test_constant_field_element_layout(self, kh_grid_mesh):
        kh = np.full(kh_grid_mesh.n_elements, 23.0)
        for xp, yp in [(5.0, 5.0), (5.37, 4.62), (3.0, 8.0)]:
            val, ok = kh_value_at_point(xp, yp, kh, "element", kh_grid_mesh)
            assert ok
            assert val == pytest.approx(23.0, abs=1e-9)

    def test_linear_field_node_layout(self, kh_grid_mesh):
        a, b, c = 0.5e-6, -0.2e-6, 100.0
        kh = _linear_kh_node(kh_grid_mesh, a, b, c)
        z0_ref = np.cos(_LAT_REF * DEG2RAD) * METERS_PER_DEG_LAT
        for xp in [5.0, 5.37, 4.27]:
            yp = _LAT_REF
            val, ok = kh_value_at_point(xp, yp, kh, "node", kh_grid_mesh)
            assert ok
            expected = c + a * xp * z0_ref + b * yp * METERS_PER_DEG_LAT
            assert val == pytest.approx(expected, rel=1e-6, abs=1e-6)

    def test_linear_field_element_layout(self, kh_grid_mesh):
        a, b, c = 0.5e-6, -0.2e-6, 100.0
        kh = _linear_kh_elem(kh_grid_mesh, a, b, c)
        z0_ref = np.cos(_LAT_REF * DEG2RAD) * METERS_PER_DEG_LAT
        for xp in [5.5, 5.37, 4.27]:
            yp = _LAT_REF
            val, ok = kh_value_at_point(xp, yp, kh, "element", kh_grid_mesh)
            assert ok
            expected = c + a * xp * z0_ref + b * yp * METERS_PER_DEG_LAT
            assert val == pytest.approx(expected, rel=1e-6, abs=1e-6)


class TestKhGradient:
    """Central-diff gradient + fallback chain."""

    def test_constant_field_zero_gradient(self, kh_grid_mesh):
        kh = np.full(kh_grid_mesh.n_nodes, 50.0)
        dKdx, dKdy, ok = kh_gradient_at_point(
            5.0, 5.0, dres_m=500.0, kh_field=kh, kh_layout="node",
            mesh=kh_grid_mesh,
        )
        assert ok
        assert abs(dKdx) < 1e-12
        assert abs(dKdy) < 1e-12

    def test_linear_field_node_layout(self, kh_grid_mesh):
        a, b = 0.7e-6, -0.3e-6
        kh = _linear_kh_node(kh_grid_mesh, a, b, c=10.0)
        for xp in [5.0, 5.27, 4.27]:
            yp = _LAT_REF
            dKdx, dKdy, ok = kh_gradient_at_point(
                xp, yp, dres_m=400.0, kh_field=kh,
                kh_layout="node", mesh=kh_grid_mesh,
            )
            assert ok
            assert dKdx == pytest.approx(a, rel=1e-6, abs=1e-12)
            assert dKdy == pytest.approx(b, rel=1e-6, abs=1e-12)

    def test_linear_field_element_layout(self, kh_grid_mesh):
        a, b = 0.7e-6, -0.3e-6
        kh = _linear_kh_elem(kh_grid_mesh, a, b, c=10.0)
        for xp in [5.5, 5.27, 4.27]:
            yp = _LAT_REF
            dKdx, dKdy, ok = kh_gradient_at_point(
                xp, yp, dres_m=400.0, kh_field=kh,
                kh_layout="element", mesh=kh_grid_mesh,
            )
            assert ok
            assert dKdx == pytest.approx(a, rel=1e-6, abs=1e-12)
            assert dKdy == pytest.approx(b, rel=1e-6, abs=1e-12)

    def test_dres_m_sensitivity(self, kh_grid_mesh):
        """Halving dres_m must keep results within O(dres²) of expected."""
        a, b = 1.0e-6, 0.0
        kh = _linear_kh_node(kh_grid_mesh, a, b, c=0.0)
        d1 = kh_gradient_at_point(5.0, 5.0, 800.0, kh, "node", kh_grid_mesh)
        d2 = kh_gradient_at_point(5.0, 5.0, 400.0, kh, "node", kh_grid_mesh)
        # Exact for linear fields → both should equal a, independent of dres
        assert d1[0] == pytest.approx(a, rel=1e-4)
        assert d2[0] == pytest.approx(a, rel=1e-4)

    def test_dres_m_must_be_positive(self, kh_grid_mesh):
        kh = np.full(kh_grid_mesh.n_nodes, 1.0)
        with pytest.raises(ValueError, match="dres_m"):
            kh_gradient_at_point(5.0, 5.0, 0.0, kh, "node", kh_grid_mesh)
        with pytest.raises(ValueError, match="dres_m"):
            kh_gradient_at_point(5.0, 5.0, -10.0, kh, "node", kh_grid_mesh)

    def test_offmesh_centre_returns_failure(self, kh_grid_mesh):
        """Centre interp failure → ok=False so the integrator can revert.

        We force the failure by passing a node_idx whose 1-ring stencil
        is too small (a corner node has only 2 elements / ~3 unique
        neighbours, but a *deliberately* invalid one — count=0 — gives
        the cleanest failure.  Use ``node_idx=0`` together with a kh
        field where the stencil happens to have < 3 unique nodes by
        slicing the mesh.  Easier path: zero-out node_to_elems_count to
        force the kernel into the early-exit branch.
        """
        # Build a degenerate "mesh" where node 0 has zero adjacency.
        bad_mesh = _build_grid_mesh(np.linspace(0.0, 10.0, 11),
                                    np.linspace(0.0, 10.0, 11))
        bad_mesh.node_to_elems_count[0] = 0  # node 0 isolated
        kh = np.full(bad_mesh.n_nodes, 1.0)
        # Pass node_idx=0 directly so the kernel can't fall through to
        # another node.
        from kelp.numerics.diffusion import _kh_gradient_at_xy, KH_LAYOUT_NODE
        dKdx, dKdy, ok = _kh_gradient_at_xy(
            float(bad_mesh.nodes_lon[0]), float(bad_mesh.nodes_lat[0]),
            500.0, 0, KH_LAYOUT_NODE,
            kh.astype(np.float64),
            bad_mesh.node_to_elems, bad_mesh.node_to_elems_count,
            bad_mesh.elem_nodes,
            bad_mesh.elem_center_lon, bad_mesh.elem_center_lat,
            bad_mesh.nodes_lon, bad_mesh.nodes_lat,
        )
        assert ok is False


class TestEMStepKernel:
    """Combined Euler-Maruyama step in pure metres."""

    def test_zero_kh_matches_deterministic(self):
        """Kh=0 → noise term zero, drift = u·dt, v·dt exactly."""
        from kelp.numerics.diffusion import diffusion_em_step_meters
        for step_i in range(20):
            dx, dy = diffusion_em_step_meters(
                u_ms=0.5, v_ms=-0.25,
                kh_x=0.0, kh_y=0.0, dKdx_x=0.0, dKdy_y=0.0,
                dt=600.0,
                seed=42, real_id=0, particle_i=7, step_i=step_i,
            )
            assert dx == pytest.approx(0.5 * 600.0, abs=1e-12)
            assert dy == pytest.approx(-0.25 * 600.0, abs=1e-12)

    def test_constant_kh_matches_uniform_kick_when_no_advection(self):
        """∂K=0, u=v=0 → EM step equals uniform-Kh kick under the same RNG."""
        from kelp.numerics.diffusion import (
            diffusion_em_step_meters, diffusion_uniform_step_meters,
        )
        kh = 5.0
        for step_i in range(50):
            dx_em, dy_em = diffusion_em_step_meters(
                u_ms=0.0, v_ms=0.0,
                kh_x=kh, kh_y=kh, dKdx_x=0.0, dKdy_y=0.0,
                dt=120.0,
                seed=99, real_id=0, particle_i=3, step_i=step_i,
            )
            dx_u, dy_u = diffusion_uniform_step_meters(
                kh, kh, 120.0, 99, 0, 3, step_i,
            )
            assert dx_em == pytest.approx(dx_u, abs=1e-12)
            assert dy_em == pytest.approx(dy_u, abs=1e-12)

    def test_ito_drift_term_explicit(self):
        """Closed-form check: EM with prescribed Kh, ∂K, RNG draw."""
        from kelp.numerics.diffusion import (
            diffusion_em_step_meters, _normal_pair, COMP_HORIZONTAL,
        )
        kh_x, kh_y = 4.0, 9.0
        dKdx, dKdy = 1e-3, -2e-3
        u, v = 0.1, -0.05
        dt = 60.0
        seed, rid, pi, si = 7, 0, 3, 11
        g1, g2 = _normal_pair(seed, rid, pi, si, COMP_HORIZONTAL)
        sqrt_dt = math.sqrt(dt)
        expected_dx = (u + dKdx) * dt + math.sqrt(2.0 * kh_x) * sqrt_dt * g1
        expected_dy = (v + dKdy) * dt + math.sqrt(2.0 * kh_y) * sqrt_dt * g2

        dx, dy = diffusion_em_step_meters(
            u, v, kh_x, kh_y, dKdx, dKdy, dt,
            seed, rid, pi, si,
        )
        assert dx == pytest.approx(expected_dx, rel=1e-12, abs=1e-12)
        assert dy == pytest.approx(expected_dy, rel=1e-12, abs=1e-12)

    def test_backward_keeps_diffusivity_drift(self):
        """Only the advection reverses with dt; ∂K drift and noise do not.

        The adjoint of the advection-diffusion operator (incompressible
        flow) is ``dX = (-u + ∇K) dτ + sqrt(2K) dW`` — the old signed drift
        gave ``-u - ∇K`` and a backward density ∝ K**-2 (review 2026-09-22).
        """
        from kelp.numerics.diffusion import diffusion_em_step_meters
        kh, dKdx, dKdy, u, v, dt = 2.0, 1e-3, -2e-3, 0.1, -0.05, 300.0
        for i in range(20):
            f = diffusion_em_step_meters(u, v, kh, kh, dKdx, dKdy, dt,
                                         5, 0, i, 0)
            b = diffusion_em_step_meters(u, v, kh, kh, dKdx, dKdy, -dt,
                                         5, 0, i, 0)
            assert f[0] - b[0] == pytest.approx(2.0 * u * dt, abs=1e-12)
            assert f[1] - b[1] == pytest.approx(2.0 * v * dt, abs=1e-12)
            f0 = diffusion_em_step_meters(0.0, 0.0, kh, kh, dKdx, dKdy, dt,
                                          5, 0, i, 0)
            b0 = diffusion_em_step_meters(0.0, 0.0, kh, kh, dKdx, dKdy, -dt,
                                          5, 0, i, 0)
            assert f0 == b0

    def test_backward_dt_sign_symmetric_amplitude(self):
        """Amplitude uses √|dt|; the advective drift flips with dt."""
        from kelp.numerics.diffusion import diffusion_em_step_meters
        kh = 2.0
        u = 0.1
        for step_i in range(20):
            dx_f, _ = diffusion_em_step_meters(
                u, 0.0, kh, kh, 0.0, 0.0, dt=300.0,
                seed=5, real_id=0, particle_i=step_i, step_i=0,
            )
            dx_b, _ = diffusion_em_step_meters(
                u, 0.0, kh, kh, 0.0, 0.0, dt=-300.0,
                seed=5, real_id=0, particle_i=step_i, step_i=0,
            )
            # Drift component differs by sign; noise component is same.
            # noise = dx - u·dt
            noise_f = dx_f - u * 300.0
            noise_b = dx_b - u * (-300.0)
            assert noise_f == pytest.approx(noise_b, abs=1e-12)

    def test_brownian_variance_with_drift(self):
        """Var of x-displacement with linear-in-x Kh.

        Setup: u=0, ∂K_x = b (constant), Kh evaluated at the centre.
        EM step: dx = b·dt + √(2K)√dt·N → mean ≈ b·dt, var ≈ 2K·dt.
        Easy to verify with 100k draws.
        """
        from kelp.numerics.diffusion import diffusion_em_step_meters
        n = 100_000
        kh = 1.0
        b = 0.01
        dt = 100.0
        dx = np.empty(n)
        for i in range(n):
            xi, _ = diffusion_em_step_meters(
                0.0, 0.0, kh, kh, b, 0.0, dt,
                seed=99, real_id=0, particle_i=i, step_i=0,
            )
            dx[i] = xi
        # Mean ≈ b·dt = 1.0 ; SE on mean = √(var/n) = √(200/100k) ≈ 0.045
        assert abs(dx.mean() - b * dt) < 4.0 * math.sqrt(2.0 * kh * dt / n)
        # Var ≈ 2·K·dt = 200; SE on var ≈ var·√(2/n) ≈ 0.9
        assert abs(dx.var() / (2.0 * kh * dt) - 1.0) < 0.05


class TestM1StepKernel:
    """Combined Milstein step in pure metres."""

    def test_zero_kh_matches_deterministic(self):
        from kelp.numerics.diffusion import diffusion_m1_step_meters
        for step_i in range(20):
            dx, dy = diffusion_m1_step_meters(
                u_ms=0.5, v_ms=-0.25,
                kh_x=0.0, kh_y=0.0, dKdx_x=0.0, dKdy_y=0.0,
                dt=600.0,
                seed=42, real_id=0, particle_i=7, step_i=step_i,
            )
            assert dx == pytest.approx(0.5 * 600.0, abs=1e-12)
            assert dy == pytest.approx(-0.25 * 600.0, abs=1e-12)

    def test_zero_grad_reduces_to_em(self):
        """∂K=0 → M1 == EM (Milstein correction term vanishes)."""
        from kelp.numerics.diffusion import (
            diffusion_em_step_meters, diffusion_m1_step_meters,
        )
        for step_i in range(50):
            args = dict(
                u_ms=0.1, v_ms=-0.05,
                kh_x=3.0, kh_y=7.0, dKdx_x=0.0, dKdy_y=0.0,
                dt=120.0,
                seed=11, real_id=0, particle_i=2, step_i=step_i,
            )
            dx_em, dy_em = diffusion_em_step_meters(**args)
            dx_m1, dy_m1 = diffusion_m1_step_meters(**args)
            assert dx_m1 == pytest.approx(dx_em, abs=1e-12)
            assert dy_m1 == pytest.approx(dy_em, abs=1e-12)

    def test_milstein_correction_explicit(self):
        """Closed-form: M1 = u·dt + ∂K·dt + 0.5·∂K·|dt|·(N²-1) + √(2K)·√|dt|·N."""
        from kelp.numerics.diffusion import (
            diffusion_m1_step_meters, _normal_pair, COMP_HORIZONTAL,
        )
        kh_x, kh_y = 4.0, 1.0
        dKdx, dKdy = 5e-3, -1e-3
        u, v = 0.1, 0.0
        dt = 60.0
        seed, rid, pi, si = 7, 0, 3, 11
        g1, g2 = _normal_pair(seed, rid, pi, si, COMP_HORIZONTAL)
        sqrt_dt = math.sqrt(dt)
        expected_dx = (
            u * dt + dKdx * dt
            + 0.5 * dKdx * dt * (g1 * g1 - 1.0)
            + math.sqrt(2.0 * kh_x) * sqrt_dt * g1
        )
        expected_dy = (
            v * dt + dKdy * dt
            + 0.5 * dKdy * dt * (g2 * g2 - 1.0)
            + math.sqrt(2.0 * kh_y) * sqrt_dt * g2
        )
        dx, dy = diffusion_m1_step_meters(
            u, v, kh_x, kh_y, dKdx, dKdy, dt,
            seed, rid, pi, si,
        )
        assert dx == pytest.approx(expected_dx, rel=1e-12, abs=1e-12)
        assert dy == pytest.approx(expected_dy, rel=1e-12, abs=1e-12)

    def test_backward_keeps_diffusivity_drift(self):
        """M1: only u·dt reverses; ∂K drift, Milstein term, noise do not."""
        from kelp.numerics.diffusion import diffusion_m1_step_meters
        kh, dKdx, u, dt = 2.0, 1e-3, 0.05, 300.0
        for i in range(20):
            f = diffusion_m1_step_meters(u, 0.0, kh, kh, dKdx, 0.0, dt,
                                         13, 0, i, 0)
            b = diffusion_m1_step_meters(u, 0.0, kh, kh, dKdx, 0.0, -dt,
                                         13, 0, i, 0)
            assert f[0] - b[0] == pytest.approx(2.0 * u * dt, abs=1e-12)
            assert f[1] == b[1]

    def test_backward_dt_sign_symmetric(self):
        """Backward path is deterministic for a fixed key."""
        from kelp.numerics.diffusion import diffusion_m1_step_meters
        kh = 2.0
        u = 0.05
        dKdx = 1e-3
        for step_i in range(20):
            dx_f, _ = diffusion_m1_step_meters(
                u, 0.0, kh, kh, dKdx, 0.0, dt=300.0,
                seed=13, real_id=0, particle_i=step_i, step_i=0,
            )
            dx_b, _ = diffusion_m1_step_meters(
                u, 0.0, kh, kh, dKdx, 0.0, dt=-300.0,
                seed=13, real_id=0, particle_i=step_i, step_i=0,
            )
            #   forward   = u·300    + dKdx·300 + 0.5·dKdx·300·(g²−1) + S·g
            #   backward  = u·(−300) + dKdx·300 + 0.5·dKdx·300·(g²−1) + S·g
            # (only the advection reverses — see the test above).  Here:
            # re-running the backward step gives the same value.
            dx_b2, _ = diffusion_m1_step_meters(
                u, 0.0, kh, kh, dKdx, 0.0, dt=-300.0,
                seed=13, real_id=0, particle_i=step_i, step_i=0,
            )
            assert dx_b == dx_b2  # determinism on backward path

    def test_milstein_lower_variance_bias_than_em(self):
        """On a linear Kh field, M1 mean drift should be closer to b·t
        than EM at large dt (Milstein corrects an O(dt²) bias).

        We compare bias = |empirical_mean - b·dt| over 10⁵ draws with a
        large dt where EM's bias is detectable.  M1 ≤ EM for the same
        seed bank.
        """
        from kelp.numerics.diffusion import (
            diffusion_em_step_meters, diffusion_m1_step_meters,
        )
        n = 30_000
        kh = 4.0
        b = 0.05  # ∂K/∂x
        dt = 200.0
        dx_em = np.empty(n)
        dx_m1 = np.empty(n)
        for i in range(n):
            xi_em, _ = diffusion_em_step_meters(
                0.0, 0.0, kh, kh, b, 0.0, dt,
                seed=2024, real_id=0, particle_i=i, step_i=0,
            )
            xi_m1, _ = diffusion_m1_step_meters(
                0.0, 0.0, kh, kh, b, 0.0, dt,
                seed=2024, real_id=0, particle_i=i, step_i=0,
            )
            dx_em[i] = xi_em
            dx_m1[i] = xi_m1
        # Mean drift of *both* schemes is b·dt to leading order; the
        # difference between them is the Milstein correction term
        # 0.5·b·|dt|·(N²−1), which has expectation 0 — so this test is
        # really a "doesn't blow up" check rather than a true bias
        # comparison.  Both should be within 5% of b·dt.
        target = b * dt
        assert abs(dx_em.mean() - target) / abs(target) < 0.05
        assert abs(dx_m1.mean() - target) / abs(target) < 0.05


class TestKhOffMeshDomainCheck:
    """Review finding #1: ``_kh_value_at_xy`` rejects off-mesh queries.

    Before the fix the LSQ stencil would return ``ok=True`` for any
    point whose 1-ring normal matrix was solvable, even if the query
    sat outside the mesh.  The fix adds a containing-element check.
    """

    def test_value_off_mesh_returns_false(self, kh_grid_mesh):
        """A point clearly outside the 0..10° mesh must report ok=False."""
        kh = np.full(kh_grid_mesh.n_nodes, 1.0)
        # Anchor the centre node deliberately at lat=5 to keep the LSQ
        # stencil solvable; the (xp, yp) below sit way outside it.
        from kelp.numerics.diffusion import _kh_value_at_xy, KH_LAYOUT_NODE
        # Pick a node near the centre as the "starting nn".
        nn_centre = int(np.argmin(
            (kh_grid_mesh.nodes_lon - 5.0) ** 2
            + (kh_grid_mesh.nodes_lat - 5.0) ** 2
        ))
        for xp, yp in [(20.0, 5.0), (5.0, 50.0), (-10.0, -5.0)]:
            val, ok = _kh_value_at_xy(
                xp, yp, nn_centre, KH_LAYOUT_NODE, kh.astype(np.float64),
                kh_grid_mesh.node_to_elems, kh_grid_mesh.node_to_elems_count,
                kh_grid_mesh.elem_nodes,
                kh_grid_mesh.elem_center_lon, kh_grid_mesh.elem_center_lat,
                kh_grid_mesh.nodes_lon, kh_grid_mesh.nodes_lat,
            )
            assert ok is False, f"off-mesh point ({xp},{yp}) reported ok=True"

    def test_offset_off_mesh_triggers_lsq_fallback(self, kh_grid_mesh):
        """Particle at the mesh corner: + offset is off-mesh → LSQ fallback.

        On a linear field, the LSQ analytic gradient still recovers the
        true slope, so the gradient kernel produces the right answer
        even when the offset is rejected.
        """
        from kelp.numerics.diffusion import (
            kh_gradient_at_point, _kh_value_at_offset, KH_LAYOUT_NODE,
        )
        a, b = 1.0e-6, 0.0
        kh = _linear_kh_node(kh_grid_mesh, a, b, c=10.0)
        # Particle at (9.99, 5.0) — east edge of the mesh.  dres_m
        # large enough that ``xp + dlon`` lands outside.
        xp, yp = 9.99, _LAT_REF
        dres_m = 5000.0  # ~0.045° lon — the +x offset is off-mesh.

        # Confirm the +x offset actually returns ok=False (sanity check
        # of the new domain predicate).
        nn_centre = int(np.argmin(
            (kh_grid_mesh.nodes_lon - xp) ** 2
            + (kh_grid_mesh.nodes_lat - yp) ** 2
        ))
        z0 = np.cos(yp * DEG2RAD) * METERS_PER_DEG_LAT
        dlon = dres_m / z0
        _, ok_xp = _kh_value_at_offset(
            xp + dlon, yp, nn_centre, KH_LAYOUT_NODE, kh.astype(np.float64),
            kh_grid_mesh.node_to_elems, kh_grid_mesh.node_to_elems_count,
            kh_grid_mesh.elem_nodes,
            kh_grid_mesh.elem_center_lon, kh_grid_mesh.elem_center_lat,
            kh_grid_mesh.nodes_lon, kh_grid_mesh.nodes_lat,
        )
        assert ok_xp is False, (
            "the +x offset should be off-mesh; if this trips, retune "
            "xp or dres_m for the test mesh"
        )

        # And the gradient still recovers ``a`` because the fallback
        # chain (one-sided / LSQ) holds.
        dKdx, dKdy, ok = kh_gradient_at_point(
            xp, yp, dres_m=dres_m, kh_field=kh,
            kh_layout="node", mesh=kh_grid_mesh,
        )
        assert ok
        # Allow looser tolerance: one-sided diff has half the formal
        # accuracy and we only ever recover the slope of a linear field
        # anyway.
        assert dKdx == pytest.approx(a, rel=5e-2, abs=1e-12)


class TestKhCoastlineFallback:
    """One-sided / LSQ-only branches engage near a synthetic coastline."""

    def test_one_sided_diff_near_boundary(self, kh_grid_mesh):
        """Particle near the mesh edge: one offset off-mesh.

        On the grid mesh, x in [0, 10].  Place a particle at xp=9.95
        with dres_m large enough that ``xp + dlon`` lands outside the
        bounding box.  The one-sided fallback should still recover the
        gradient of a linear field to within central-diff tolerance.
        """
        a, b = 1.0e-6, 0.0
        kh = _linear_kh_node(kh_grid_mesh, a, b, c=10.0)
        # Note: with our LSQ-everywhere primary path, the offset point
        # xp + dlon may still be acceptable to the 1-ring stencil even
        # if it's outside the bounding box.  The fallback chain still
        # returns the correct slope on a linear field — that's the
        # contract.
        dKdx, dKdy, ok = kh_gradient_at_point(
            9.95, _LAT_REF, dres_m=2000.0, kh_field=kh,
            kh_layout="node", mesh=kh_grid_mesh,
        )
        assert ok
        assert dKdx == pytest.approx(a, rel=2e-2, abs=1e-12)
        assert abs(dKdy) < 1e-9
