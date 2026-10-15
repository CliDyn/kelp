"""LazyLerpField: a VelocityField view that blends on first access.

Temporal-interp runs used to materialize the full blended field every
step; the lazy view makes each consumer pay only for the fields it
actually reads (the fused 2D kernels read none — they take the raw
snapshot pair).  Values must match the eager ``VelocityField.lerp``
bit-for-bit because both go through the same ``_blend``.
"""
import numpy as np
import pytest

from kelp.core.io import LazyLerpField, VelocityField


def _mk(n_elem=64, n_node=32, k=4, with_w=True, with_kh=False,
        with_scalars=False, seed=0):
    rng = np.random.default_rng(seed)

    def f(*s):
        return rng.standard_normal(s)

    return VelocityField(
        u=f(n_elem), v=f(n_elem), eta=f(n_node),
        w=f(k, n_node) if with_w else None, t=1.0,
        kh_zonal=f(n_node) if with_kh else None,
        kh_meridional=f(n_node) if with_kh else None,
        kh_layout="node" if with_kh else None,
        scalars={"temp": f(n_node)} if with_scalars else None,
    )


class TestLazyLerpField:
    def test_values_match_eager_lerp(self):
        va, vb = _mk(seed=1), _mk(seed=2)
        lazy = LazyLerpField(va, vb, 0.3)
        eager = va.lerp(vb, 0.3)
        for name in ("u", "v", "eta", "w"):
            assert np.array_equal(getattr(lazy, name), getattr(eager, name))
        assert lazy.t == eager.t
        assert lazy.uv_layout == eager.uv_layout

    def test_blend_is_lazy_and_cached(self, monkeypatch):
        import kelp.core.io as lio
        calls = []
        real = lio._blend
        monkeypatch.setattr(lio, "_blend",
                            lambda a, b, al: calls.append(1) or real(a, b, al))
        lazy = LazyLerpField(_mk(seed=3), _mk(seed=4), 0.5)
        assert calls == []              # nothing blended at construction
        _ = lazy.eta
        assert len(calls) == 1          # eta only
        _ = lazy.eta
        assert len(calls) == 1          # cached, no re-blend
        _ = lazy.u
        assert len(calls) == 2          # u on demand; v untouched

    def test_w_none_short_circuits_without_blend(self, monkeypatch):
        import kelp.core.io as lio
        calls = []
        real = lio._blend
        monkeypatch.setattr(lio, "_blend",
                            lambda a, b, al: calls.append(1) or real(a, b, al))
        va, vb = _mk(with_w=False, seed=5), _mk(with_w=False, seed=6)
        lazy = LazyLerpField(va, vb, 0.4)
        assert lazy.w is None           # _is_3d() probe must stay free
        assert calls == []

    def test_kh_and_scalars(self):
        va = _mk(with_kh=True, with_scalars=True, seed=7)
        vb = _mk(with_kh=True, with_scalars=True, seed=8)
        lazy, eager = LazyLerpField(va, vb, 0.25), va.lerp(vb, 0.25)
        assert np.array_equal(lazy.kh_zonal, eager.kh_zonal)
        assert np.array_equal(lazy.kh_meridional, eager.kh_meridional)
        assert lazy.kh_layout == eager.kh_layout
        assert np.array_equal(lazy.scalars["temp"], eager.scalars["temp"])

    def test_kh_none_stays_none(self):
        lazy = LazyLerpField(_mk(seed=9), _mk(seed=10), 0.5)
        assert lazy.kh_zonal is None and lazy.kh_layout is None
        assert lazy.scalars is None

    def test_kh_one_sided_stays_none(self):
        # VelocityField.lerp returns kh=None when only one side carries
        # it; the lazy view must do the same (not blend, not crash).
        va = _mk(with_kh=True, seed=18)
        vb = _mk(with_kh=False, seed=19)
        lazy = LazyLerpField(va, vb, 0.5)
        assert lazy.kh_layout is None
        assert lazy.kh_zonal is None and lazy.kh_meridional is None

    def test_lerp_method_chains(self):
        # _pick_kh_eval_vel / _make_stage_vels call vel.lerp(vel_next, a)
        va, vb, vc = _mk(seed=11), _mk(seed=12), _mk(seed=13)
        lazy = LazyLerpField(va, vb, 0.5)
        out = lazy.lerp(vc, 0.5)
        ref = va.lerp(vb, 0.5).lerp(vc, 0.5)
        assert np.array_equal(out.u, ref.u)
        assert np.array_equal(out.eta, ref.eta)

    def test_materialize_equals_eager(self):
        va, vb = _mk(with_kh=True, with_scalars=True, seed=20), \
                 _mk(with_kh=True, with_scalars=True, seed=21)
        m = LazyLerpField(va, vb, 0.75).materialize()
        e = va.lerp(vb, 0.75)
        assert isinstance(m, VelocityField)
        for name in ("u", "v", "eta", "w", "kh_zonal", "kh_meridional"):
            assert np.array_equal(getattr(m, name), getattr(e, name))
        assert np.array_equal(m.scalars["temp"], e.scalars["temp"])

    def test_uv_layout_mismatch_raises_at_construction(self):
        va, vb = _mk(seed=14), _mk(seed=15)
        vb.uv_layout = "node"
        with pytest.raises(ValueError):
            LazyLerpField(va, vb, 0.5)

    def test_raw_pair_exposed(self):
        va, vb = _mk(seed=16), _mk(seed=17)
        lazy = LazyLerpField(va, vb, 0.125)
        assert lazy.vel_a is va and lazy.vel_b is vb
        assert lazy.alpha == 0.125

    def test_unknown_attribute_raises(self):
        lazy = LazyLerpField(_mk(seed=22), _mk(seed=23), 0.5)
        with pytest.raises(AttributeError):
            _ = lazy.no_such_field


class TestPayloadIsLazy:
    def test_build_vel_payload_returns_lazy_between_snapshots(self):
        """Between two snapshots the payload must be a LazyLerpField."""
        import kelp.tracking as tr

        class _StubReader:
            @staticmethod
            def read_velocities(ds, timestep, mesh, mode_cfg,
                                diffusion=None, sample_vars=None):
                n = 8
                base = float(timestep + 1)
                return VelocityField(u=np.full(n, base),
                                     v=np.full(n, -base),
                                     eta=np.zeros(4))

        cache = tr._VelCache()
        file_times = np.array([0.0, 100.0, 200.0])
        bracket = tr._resolve_time_bracket(50.0, file_times, 3, "stop")
        vel, vel_next, dt_to_next, fi = tr._build_vel_payload(
            bracket, cache, _StubReader, None, None, None)
        assert isinstance(vel, LazyLerpField)
        assert vel.alpha == pytest.approx(0.5)
        # values equivalent to the eager path
        assert np.array_equal(vel.u, np.full(8, 1.5))
        assert dt_to_next == pytest.approx(50.0)
        # exact-snapshot queries still return the raw cached field
        bracket0 = tr._resolve_time_bracket(0.0, file_times, 3, "stop")
        vel0, *_ = tr._build_vel_payload(bracket0, cache, _StubReader,
                                         None, None, None)
        assert isinstance(vel0, VelocityField)


class TestFailFastValidation:
    def test_kh_shape_mismatch_raises_at_construction(self):
        # same mesh sizes, only the kh arrays differ between snapshots
        va = _mk(with_kh=True, seed=30)
        vb = _mk(with_kh=True, seed=31)
        vb.kh_zonal = vb.kh_zonal[:-3]
        vb.kh_meridional = vb.kh_meridional[:-3]
        with pytest.raises(ValueError, match="kh_zonal shape"):
            LazyLerpField(va, vb, 0.5)

    def test_kh_layout_mismatch_raises_at_construction(self):
        va = _mk(with_kh=True, seed=32)
        vb = _mk(with_kh=True, seed=33)
        vb.kh_layout = "element"
        with pytest.raises(ValueError, match="kh_layout mismatch"):
            LazyLerpField(va, vb, 0.5)

    def test_scalars_key_mismatch_raises_at_construction(self):
        va = _mk(with_scalars=True, seed=34)
        vb = _mk(with_scalars=True, seed=35)
        vb.scalars = {"salt": vb.scalars["temp"]}
        with pytest.raises(ValueError, match="scalars key mismatch"):
            LazyLerpField(va, vb, 0.5)

    def test_scalars_shape_mismatch_raises_at_construction(self):
        va = _mk(with_scalars=True, seed=36)
        vb = _mk(with_scalars=True, seed=37)
        vb.scalars = {"temp": vb.scalars["temp"][:-3]}
        with pytest.raises(ValueError, match="scalars\\['temp'\\] shape"):
            LazyLerpField(va, vb, 0.5)



class TestCoreFieldValidation:
    def test_uv_shape_mismatch_raises_at_construction(self):
        # The fused raw-pair kernels index BOTH parents without bounds
        # checks — a corrupt bracket must die here, not OOB-read there.
        va, vb = _mk(seed=40), _mk(n_elem=48, seed=41)
        with pytest.raises(ValueError, match="u shape mismatch"):
            LazyLerpField(va, vb, 0.5)

    def test_eta_shape_mismatch_raises_at_construction(self):
        va, vb = _mk(seed=42), _mk(seed=43)
        vb.eta = vb.eta[:-5]
        with pytest.raises(ValueError, match="eta shape mismatch"):
            LazyLerpField(va, vb, 0.5)

    def test_w_shape_mismatch_raises_when_both_present(self):
        va, vb = _mk(seed=44), _mk(k=6, seed=45)
        with pytest.raises(ValueError, match="w shape mismatch"):
            LazyLerpField(va, vb, 0.5)

    def test_one_sided_w_still_allowed(self):
        # lerp semantics: one-sided w -> result w is None, not an error.
        va, vb = _mk(with_w=True, seed=46), _mk(with_w=False, seed=47)
        lazy = LazyLerpField(va, vb, 0.5)
        assert lazy.w is None
