"""Tests for the numba-parallel field blend behind VelocityField.lerp.

The full-field lerp is the per-step hot spot of temporal-interp runs on
big meshes (measured ~0.4 s/step on the 23M-element global mesh with
single-threaded numpy).  ``_blend`` replaces the numpy expression with a
numba ``prange`` kernel for large float64 arrays while keeping the exact
per-element arithmetic ``(1-alpha)*a[i] + alpha*b[i]`` — results must be
bit-identical to the numpy expression on every path.
"""
import numpy as np
import pytest

import kelp.core.io as lio
from kelp.core.io import (
    HAS_NUMBA, VelocityField, _blend, _BLEND_MIN_SIZE,
)


def _ref(a, b, alpha):
    """The original numpy expression (per-element arithmetic reference)."""
    return (1.0 - alpha) * a + alpha * b


# ---------------------------------------------------------------------------
# _blend primitive
# ---------------------------------------------------------------------------

class TestBlendPrimitive:
    def test_numba_available_in_this_environment(self):
        # The perf feature must not silently degrade to the numpy path
        # in the supported environments — if numba is genuinely absent
        # this xfails loudly instead of letting every other test pass
        # green via the fallback.
        assert HAS_NUMBA, "numba missing: _blend would silently be numpy-only"

    def test_kernel_actually_taken_for_large_arrays(self, monkeypatch):
        # Guard against the threshold/flag gate silently never selecting
        # the numba branch (the whole optimization would be dead but all
        # value-based tests would still pass via the fallback).
        calls = []
        real_kernel = lio._blend_kernel

        def spy(a, b, alpha, out):
            calls.append(a.size)
            real_kernel(a, b, alpha, out)

        monkeypatch.setattr(lio, "_blend_kernel", spy)
        rng = np.random.default_rng(11)
        big = rng.standard_normal(_BLEND_MIN_SIZE * 2)
        _blend(big, big.copy(), 0.5)
        small = rng.standard_normal(64)
        _blend(small, small.copy(), 0.5)
        assert calls == [big.size]      # taken for big, skipped for small

    def test_shape_mismatch_preserves_numpy_contract(self):
        # Old code raised numpy's broadcast ValueError on incompatible
        # shapes; the bounds-check-free kernel must never see such pairs.
        rng = np.random.default_rng(12)
        a = rng.standard_normal(_BLEND_MIN_SIZE * 2)
        b = rng.standard_normal(a.size - 50)
        with pytest.raises(ValueError):
            _blend(a, b, 0.5)
        # ... and shapes that numpy can broadcast keep broadcasting.
        a2 = rng.standard_normal((1, _BLEND_MIN_SIZE * 2))
        b2 = rng.standard_normal(_BLEND_MIN_SIZE * 2)
        out = _blend(a2, b2, 0.25)
        assert out.shape == (1, b2.size)
        assert np.array_equal(out, _ref(a2, b2, 0.25))

    def test_large_1d_bit_identical(self):
        rng = np.random.default_rng(42)
        n = _BLEND_MIN_SIZE * 4          # comfortably on the numba path
        a = rng.standard_normal(n)
        b = rng.standard_normal(n)
        out = _blend(a, b, 0.3171)
        assert np.array_equal(out, _ref(a, b, 0.3171))

    def test_small_1d_bit_identical(self):
        rng = np.random.default_rng(1)
        a = rng.standard_normal(257)     # below threshold -> numpy path
        b = rng.standard_normal(257)
        out = _blend(a, b, 0.5)
        assert np.array_equal(out, _ref(a, b, 0.5))

    def test_2d_interface_array(self):
        # w is (K, N); the kernel must handle multi-dim contiguous arrays.
        rng = np.random.default_rng(2)
        a = rng.standard_normal((70, max(2, _BLEND_MIN_SIZE // 35)))
        b = rng.standard_normal(a.shape)
        out = _blend(a, b, 0.9)
        assert out.shape == a.shape
        assert np.array_equal(out, _ref(a, b, 0.9))

    def test_alpha_endpoints_exact(self):
        rng = np.random.default_rng(3)
        a = rng.standard_normal(_BLEND_MIN_SIZE * 2)
        b = rng.standard_normal(a.size)
        # alpha=0: (1-0)*a + 0*b == a exactly (b may inject -0.0 only
        # via 0*b which keeps the sum equal to a for finite values)
        assert np.array_equal(_blend(a, b, 0.0), _ref(a, b, 0.0))
        assert np.array_equal(_blend(a, b, 1.0), _ref(a, b, 1.0))

    def test_non_contiguous_falls_back_correctly(self):
        rng = np.random.default_rng(4)
        a_full = rng.standard_normal(_BLEND_MIN_SIZE * 4)
        b_full = rng.standard_normal(a_full.size)
        a, b = a_full[::2], b_full[::2]              # views, not contiguous
        out = _blend(a, b, 0.25)
        assert np.array_equal(out, _ref(a, b, 0.25))

    def test_non_float64_falls_back_to_float64(self):
        # float32 inputs (float32 velocity fields, TODO 5.3): the fallback now
        # computes in float64 — under NEP 50 the naive `(1-a)*a32 + a*b32` stays
        # float32 and would round, diverging from the float64 path; the blended
        # field is a transient, so this float64 promotion is intended.
        rng = np.random.default_rng(5)
        a = rng.standard_normal(_BLEND_MIN_SIZE * 2).astype(np.float32)
        b = rng.standard_normal(a.size).astype(np.float32)
        out = _blend(a, b, 0.75)
        ref = _ref(a.astype(np.float64), b.astype(np.float64), 0.75)
        assert out.dtype == np.float64
        assert np.array_equal(out, ref)

    def test_output_is_fresh_array(self):
        rng = np.random.default_rng(6)
        a = rng.standard_normal(_BLEND_MIN_SIZE * 2)
        b = rng.standard_normal(a.size)
        out = _blend(a, b, 0.5)
        assert out is not a and out is not b
        out[0] = 1e30
        assert a[0] != 1e30 and b[0] != 1e30


# ---------------------------------------------------------------------------
# VelocityField.lerp end-to-end bit-identity vs the numpy reference
# ---------------------------------------------------------------------------

def _make_pair(n_elem, n_node, k=4, with_w=True, with_kh=True,
               with_scalars=True, seed=7):
    rng = np.random.default_rng(seed)

    def f(*shape):
        return rng.standard_normal(shape)

    def mk():
        return VelocityField(
            u=f(n_elem), v=f(n_elem), eta=f(n_node),
            w=f(k, n_node) if with_w else None,
            t=float(rng.standard_normal()),
            kh_zonal=f(n_node) if with_kh else None,
            kh_meridional=f(n_node) if with_kh else None,
            kh_layout="node" if with_kh else None,
            scalars={"temp": f(n_node), "salt": f(k - 1, n_node)}
            if with_scalars else None,
        )

    return mk(), mk()


class TestLerpBitIdentity:
    @pytest.mark.parametrize("alpha", [0.0, 0.123456789, 0.5, 1.0])
    def test_large_fields_match_numpy_reference(self, alpha):
        # sizes above the numba threshold
        n_elem = _BLEND_MIN_SIZE * 2 + 11
        n_node = _BLEND_MIN_SIZE + 13
        va, vb = _make_pair(n_elem, n_node)
        out = va.lerp(vb, alpha)
        assert np.array_equal(out.u, _ref(va.u, vb.u, alpha))
        assert np.array_equal(out.v, _ref(va.v, vb.v, alpha))
        assert np.array_equal(out.eta, _ref(va.eta, vb.eta, alpha))
        assert np.array_equal(out.w, _ref(va.w, vb.w, alpha))
        assert np.array_equal(
            out.kh_zonal, _ref(va.kh_zonal, vb.kh_zonal, alpha))
        assert np.array_equal(
            out.kh_meridional, _ref(va.kh_meridional, vb.kh_meridional, alpha))
        for key in ("temp", "salt"):
            assert np.array_equal(
                out.scalars[key], _ref(va.scalars[key], vb.scalars[key], alpha))
        assert out.t == pytest.approx((1 - alpha) * va.t + alpha * vb.t)

    def test_small_fields_match_numpy_reference(self):
        va, vb = _make_pair(8, 9, with_w=True)
        out = va.lerp(vb, 0.3)
        assert np.array_equal(out.u, _ref(va.u, vb.u, 0.3))
        assert np.array_equal(out.w, _ref(va.w, vb.w, 0.3))

    def test_optional_fields_stay_none(self):
        va, vb = _make_pair(64, 32, with_w=False, with_kh=False,
                            with_scalars=False)
        out = va.lerp(vb, 0.4)
        assert out.w is None and out.kh_zonal is None and out.scalars is None
