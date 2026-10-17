"""``VelocityField.uv_vertical``: placement of horizontal velocity in depth.

The carrier says where the 3-D u/v data points sit in the vertical:
``"layer"`` (layer centres, piecewise constant in depth — today's meaning
and the default), ``"level"`` (on the levels, FESOM 1.4) or
``"layer_linear"`` (layer-centred data evaluated linearly in depth).  The
constructor checks the 3-D shapes against it, so a dropped or wrong flag
raises instead of mis-placing the data, and every site that rebuilds a
field (lerp, the lazy lerp, PCHIP reconstruction, the shared cache, the
time-aware stage builder) carries it and refuses mixed placements.

``_dl_token`` is the record-check token of the depth-linear path (set by
a later check): it is never an ``__init__`` argument, so any rebuild that
goes through the constructor (``dataclasses.replace``, the shared cache)
drops it; the blends keep it only when every parent holds the SAME object.
"""
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from kelp.core import io as kelp_io
from kelp.core.io import LazyLerpField, VelocityField
from kelp.numerics.integrators import _validate_time_aware_inputs
from kelp.numerics.temporal_reconstruct import reconstruct_pchip
from kelp.tracking import _assemble_vel, _flatten_vel

from tests._bitwise import assert_bitwise

K = 5   # levels (interfaces); K - 1 layers
N = 7   # nodes
E = 9   # elements

PLACEMENTS = ("layer", "level", "layer_linear")


def _field(placement="layer", layout="node", seed=0, t=0.0, token=None,
           u_rows=None):
    """A consistent 3-D field for ``placement`` (u rows K or K-1)."""
    rng = np.random.default_rng(seed)
    if u_rows is None:
        u_rows = K if placement == "level" else K - 1
    n_uv = N if layout == "node" else E
    vel = VelocityField(
        u=rng.standard_normal((u_rows, n_uv)),
        v=rng.standard_normal((u_rows, n_uv)),
        eta=rng.standard_normal(N),
        w=rng.standard_normal((K, N)),
        t=t, uv_layout=layout, uv_vertical=placement,
    )
    if token is not None:
        vel._dl_token = token
    return vel


# -- the value and the shape contract ----------------------------------------

def test_default_is_layer():
    assert kelp_io._VALID_UV_VERTICAL == ("layer", "level", "layer_linear")
    v2 = VelocityField(u=np.zeros(E), v=np.zeros(E), eta=np.zeros(N))
    v3 = VelocityField(u=np.zeros((K - 1, E)), v=np.zeros((K - 1, E)),
                       eta=np.zeros(N), w=np.zeros((K, N)))
    for vel in (v2, v3):
        assert vel.uv_vertical == "layer"
        assert vel._dl_token is None
    lazy = LazyLerpField(v3, v3, 0.5)
    assert lazy.uv_vertical == "layer"
    assert lazy._dl_token is None


@pytest.mark.parametrize("bad", ["linear", "Layer", "levels", "", None])
def test_invalid_value_raises(bad):
    with pytest.raises(ValueError, match="uv_vertical"):
        VelocityField(u=np.zeros((K - 1, N)), v=np.zeros((K - 1, N)),
                      eta=np.zeros(N), w=np.zeros((K, N)), uv_layout="node",
                      uv_vertical=bad)


def test_3d_shape_layer_requires_k_minus_1():
    # (K, N) u with (K, N) w is LEVEL-shaped data: a dropped "level" flag
    # must raise, not be read as layers
    with pytest.raises(ValueError, match="uv_vertical='layer'"):
        _field("layer", u_rows=K)
    with pytest.raises(ValueError, match="uv_vertical='layer'"):
        _field("layer", layout="element", u_rows=K)
    # the consistent shapes construct: node and element layout
    assert _field("layer").u.shape == (K - 1, N)
    assert _field("layer", layout="element").u.shape == (K - 1, E)


def test_3d_shape_level_requires_k_and_node():
    with pytest.raises(ValueError, match="uv_vertical='level'"):
        _field("level", u_rows=K - 1)
    with pytest.raises(ValueError, match="node"):
        _field("level", layout="element")
    assert _field("level").u.shape == (K, N)


def test_layer_linear_requires_node_and_k_minus_1():
    with pytest.raises(ValueError, match="uv_vertical='layer_linear'"):
        _field("layer_linear", u_rows=K)
    with pytest.raises(ValueError, match="node"):
        _field("layer_linear", layout="element")
    assert _field("layer_linear").u.shape == (K - 1, N)


def test_depth_linear_needs_node_layout_without_w():
    # Ruling R8: node layout belongs to the placement, with or without w
    for placement in ("level", "layer_linear"):
        with pytest.raises(ValueError, match="node"):
            VelocityField(u=np.zeros((K - 1, E)), v=np.zeros((K - 1, E)),
                          eta=np.zeros(N), w=None, uv_layout="element",
                          uv_vertical=placement)
    # the layer default stays free of it (FESOM2 3-D reads without w)
    vel = VelocityField(u=np.zeros((K - 1, E)), v=np.zeros((K - 1, E)),
                        eta=np.zeros(N), w=None, uv_layout="element")
    assert vel.uv_vertical == "layer"


def test_2d_field_must_be_layer():
    for placement in ("level", "layer_linear"):
        with pytest.raises(ValueError, match="2-D"):
            VelocityField(u=np.zeros(N), v=np.zeros(N), eta=np.zeros(N),
                          w=None, uv_layout="node", uv_vertical=placement)
        # a 1-D u is 2-D data even when a w rides along
        with pytest.raises(ValueError, match="2-D"):
            VelocityField(u=np.zeros(N), v=np.zeros(N), eta=np.zeros(N),
                          w=np.zeros((K, N)), uv_layout="node",
                          uv_vertical=placement)
    vel = VelocityField(u=np.zeros(N), v=np.zeros(N), eta=np.zeros(N),
                        uv_layout="node", uv_vertical="layer")
    assert vel.uv_vertical == "layer"


# -- propagation through every rebuild ----------------------------------------

def test_lerp_carries_and_checks():
    tok = object()
    a = _field("level", seed=1, t=0.0, token=tok)
    b = _field("level", seed=2, t=10.0, token=tok)
    out = a.lerp(b, 0.25)
    assert out.uv_vertical == "level"
    assert out._dl_token is tok
    # values are the placement-blind blend: same bits as a layer-shaped
    # copy of the same arrays would give
    ref = (1.0 - 0.25) * a.u + 0.25 * b.u
    assert_bitwise(out.u, ref, "lerp u")
    # token kept only when both parents hold the SAME object
    b._dl_token = object()
    assert a.lerp(b, 0.25)._dl_token is None
    b._dl_token = None
    assert a.lerp(b, 0.25)._dl_token is None
    a._dl_token = None
    assert a.lerp(b, 0.25)._dl_token is None
    # mixed placement, same shapes: only the placement differs
    la = _field("layer", seed=3)
    lb = _field("layer_linear", seed=4)
    assert la.u.shape == lb.u.shape
    with pytest.raises(ValueError, match="uv_vertical mismatch"):
        la.lerp(lb, 0.5)
    with pytest.raises(ValueError, match="uv_vertical mismatch"):
        lb.lerp(la, 0.5)
    assert lb.lerp(_field("layer_linear", seed=5), 0.5).uv_vertical \
        == "layer_linear"


def test_lazy_lerp_carries():
    tok = object()
    a = _field("level", seed=1, t=0.0, token=tok)
    b = _field("level", seed=2, t=10.0, token=tok)
    lazy = LazyLerpField(a, b, 0.4)
    assert lazy.uv_vertical == "level"
    assert lazy._dl_token is tok
    m = lazy.materialize()
    assert isinstance(m, VelocityField)
    assert m.uv_vertical == "level"
    assert m._dl_token is tok
    eager = a.lerp(b, 0.4)
    for name in ("u", "v", "eta", "w"):
        assert_bitwise(getattr(m, name), getattr(eager, name), name)
    # different tokens (or one missing) -> None, on the view and the copy
    b._dl_token = object()
    lazy2 = LazyLerpField(a, b, 0.4)
    assert lazy2._dl_token is None
    assert lazy2.materialize()._dl_token is None
    b._dl_token = None
    assert LazyLerpField(a, b, 0.4)._dl_token is None
    # the plain attributes do not go through __getattr__ (which raises for
    # unknown names) and do not blend anything
    fresh = LazyLerpField(a, b, 0.4)
    assert "uv_vertical" in vars(fresh) and "_dl_token" in vars(fresh)
    assert fresh.uv_vertical == "level" and fresh._cache == {}
    # mixed placement, same shapes: raises at construction
    la = _field("layer", seed=3)
    lb = _field("layer_linear", seed=4)
    with pytest.raises(ValueError, match="uv_vertical mismatch"):
        LazyLerpField(la, lb, 0.5)
    with pytest.raises(ValueError, match="uv_vertical mismatch"):
        LazyLerpField(lb, la, 0.5)
    lazy_ll = LazyLerpField(lb, _field("layer_linear", seed=5), 0.5)
    assert lazy_ll.uv_vertical == "layer_linear"
    assert lazy_ll.materialize().uv_vertical == "layer_linear"


_TIMES = (0.0, 10.0, 20.0, 30.0)


def _snaps(placements, token=None):
    return [_field(p, seed=10 + i, t=_TIMES[i], token=token)
            for i, p in enumerate(placements)]


def test_pchip_carries_and_checks():
    tok = object()
    out = reconstruct_pchip(_snaps(["level"] * 4, token=tok), _TIMES, 13.0)
    assert out.uv_vertical == "level"
    assert out._dl_token is tok
    out_ll = reconstruct_pchip(_snaps(["layer_linear"] * 4), _TIMES, 13.0)
    assert out_ll.uv_vertical == "layer_linear"
    assert out_ll._dl_token is None
    # placement does not touch the values: same bits as the layer twin
    out_l = reconstruct_pchip(_snaps(["layer"] * 4), _TIMES, 13.0)
    for name in ("u", "v", "w", "eta"):
        assert_bitwise(getattr(out_ll, name), getattr(out_l, name), name)

    # a mixture at EACH position, same shapes: the placement error comes
    # from reconstruct_pchip itself
    for pos in range(4):
        for base, odd in (("layer", "layer_linear"), ("layer_linear", "layer")):
            placements = [base] * 4
            placements[pos] = odd
            snaps = _snaps(placements)
            assert len({s.u.shape for s in snaps}) == 1
            with pytest.raises(ValueError, match="uv_vertical") as ei:
                reconstruct_pchip(snaps, _TIMES, 13.0)
            assert "shape mismatch" not in str(ei.value)
            # raised by reconstruct_pchip's own placement check (through
            # the shared helper), not by blend() or the constructor
            assert [e.name for e in ei.traceback[-2:]] == [
                "reconstruct_pchip", "_require_same_uv_vertical"]
    # the placement check runs BEFORE the shape checks: a level/layer mix
    # (different u shapes) reports the placement, not the shapes
    snaps = _snaps(["layer", "level", "layer", "layer"])
    with pytest.raises(ValueError, match="uv_vertical") as ei:
        reconstruct_pchip(snaps, _TIMES, 13.0)
    assert "shape mismatch" not in str(ei.value)

    # token kept only when all four share it
    for pos in range(4):
        snaps = _snaps(["level"] * 4, token=tok)
        snaps[pos]._dl_token = object()
        assert reconstruct_pchip(snaps, _TIMES, 13.0)._dl_token is None
        snaps[pos]._dl_token = None
        assert reconstruct_pchip(snaps, _TIMES, 13.0)._dl_token is None


def test_shared_cache_roundtrip():
    for placement in PLACEMENTS:
        vel = _field(placement, seed=7, token=object())
        arrays, meta = _flatten_vel(vel)
        assert meta["uv_vertical"] == placement
        assert all(isinstance(x, np.ndarray) for x in arrays.values())
        out = _assemble_vel(arrays, meta)
        assert out.uv_vertical == placement
        assert out._dl_token is None          # the token never rides along
        for name in ("u", "v", "eta", "w"):
            assert_bitwise(getattr(out, name), getattr(vel, name), name)
    # a meta written without the key (an older publisher) reads as "layer"
    arrays, meta = _flatten_vel(_field("layer", seed=8))
    del meta["uv_vertical"]
    assert _assemble_vel(arrays, meta).uv_vertical == "layer"


def test_replace_resets_dl_token():
    tok = object()
    vel = _field("layer_linear", seed=3, token=tok)
    new_u = vel.u + 1.0
    out = dataclasses.replace(vel, u=new_u)
    assert out._dl_token is None
    assert out.uv_vertical == "layer_linear"
    assert vel._dl_token is tok              # the source keeps its token
    # the field contract: not an __init__ argument, not in repr/compare
    f = {x.name: x for x in dataclasses.fields(VelocityField)}["_dl_token"]
    assert (f.init, f.repr, f.compare, f.default) == (False, False, False,
                                                      None)
    assert "_dl_token" not in repr(vel)
    with pytest.raises(TypeError):
        VelocityField(u=vel.u, v=vel.v, eta=vel.eta, w=vel.w,
                      uv_layout="node", _dl_token=tok)


def _double(placement=None, seed=0):
    rng = np.random.default_rng(seed)
    ns = SimpleNamespace(
        u=rng.standard_normal((K - 1, N)), v=rng.standard_normal((K - 1, N)),
        eta=rng.standard_normal(N), w=rng.standard_normal((K, N)))
    if placement is not None:
        ns.uv_vertical = placement
    return ns


def test_time_aware_inputs_mixed_raise():
    # field-like doubles: _validate_time_aware_inputs reads attributes only
    for pa, pb in (("layer", "layer_linear"), ("layer_linear", "layer")):
        with pytest.raises(ValueError, match="uv_vertical mismatch"):
            _validate_time_aware_inputs(_double(pa, 1), _double(pb, 2),
                                        100.0, 50.0)
    # equal placements pass; a double without the attribute counts as
    # "layer" (the existing error-path doubles)
    _validate_time_aware_inputs(_double("layer_linear", 1),
                                _double("layer_linear", 2), 100.0, 50.0)
    _validate_time_aware_inputs(_double(None, 1), _double("layer", 2),
                                100.0, 50.0)
    with pytest.raises(ValueError, match="uv_vertical mismatch"):
        _validate_time_aware_inputs(_double(None, 1),
                                    _double("layer_linear", 2), 100.0, 50.0)
    # a lazy step field against a real next field
    lazy = LazyLerpField(_field("layer", seed=3), _field("layer", seed=4),
                         0.5)
    with pytest.raises(ValueError, match="uv_vertical mismatch"):
        _validate_time_aware_inputs(lazy, _field("layer_linear", seed=5),
                                    100.0, 50.0)
    assert lazy._cache == {}                  # the probe blended nothing
    _validate_time_aware_inputs(lazy, _field("layer", seed=5), 100.0, 50.0)
