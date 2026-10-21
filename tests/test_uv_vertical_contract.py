"""``VelocityField.uv_vertical``: placement of horizontal velocity in depth.

The carrier says where the 3-D u/v data points sit in the vertical:
``"layer"`` (layer centres, piecewise constant in depth — today's meaning
and the default), ``"level"`` (on the levels, FESOM 1.4; node layout) or
``"layer_linear"`` (layer-centred data evaluated linearly in depth, on
nodes or on elements).  The
constructor checks the 3-D shapes against it, so a dropped or wrong flag
raises instead of mis-placing the data, and every site that rebuilds a
field (lerp, the lazy lerp, PCHIP reconstruction, the shared cache, the
time-aware stage builder) carries it and refuses mixed placements.
"""
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


def _field(placement="layer", layout="node", seed=0, t=0.0, u_rows=None):
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
    return vel


# -- the value and the shape contract ----------------------------------------

def test_default_is_layer():
    assert kelp_io._VALID_UV_VERTICAL == ("layer", "level", "layer_linear")
    v2 = VelocityField(u=np.zeros(E), v=np.zeros(E), eta=np.zeros(N))
    v3 = VelocityField(u=np.zeros((K - 1, E)), v=np.zeros((K - 1, E)),
                       eta=np.zeros(N), w=np.zeros((K, N)))
    for vel in (v2, v3):
        assert vel.uv_vertical == "layer"
    lazy = LazyLerpField(v3, v3, 0.5)
    assert lazy.uv_vertical == "layer"


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


def test_layer_linear_requires_k_minus_1():
    # one row per layer, on nodes and on elements
    for layout, n_uv in (("node", N), ("element", E)):
        with pytest.raises(ValueError, match="uv_vertical='layer_linear'"):
            _field("layer_linear", layout=layout, u_rows=K)
        assert _field("layer_linear", layout=layout).u.shape == (K - 1, n_uv)


def test_level_needs_node_layout_without_w():
    # "level" data sit on the nodes' levels, with or without w
    for rows in (K, K - 1):
        with pytest.raises(ValueError, match="node"):
            VelocityField(u=np.zeros((rows, E)), v=np.zeros((rows, E)),
                          eta=np.zeros(N), w=None, uv_layout="element",
                          uv_vertical="level")
    # "layer_linear" on elements constructs without w too, as the layer
    # default does (FESOM2 3-D reads without w)
    for placement in ("layer", "layer_linear"):
        vel = VelocityField(u=np.zeros((K - 1, E)), v=np.zeros((K - 1, E)),
                            eta=np.zeros(N), w=None, uv_layout="element",
                            uv_vertical=placement)
        assert vel.uv_vertical == placement
    vel = VelocityField(u=np.zeros((K - 1, E)), v=np.zeros((K - 1, E)),
                        eta=np.zeros(N), w=None, uv_layout="element")
    assert vel.uv_vertical == "layer"


def test_layer_linear_element_accepted():
    """An element-layout ``"layer_linear"`` field ``(K-1, E)`` constructs
    and keeps its placement and layout through every rebuild: lerp, the
    lazy lerp, PCHIP reconstruction and the shared cache."""
    a = _field("layer_linear", layout="element", seed=1, t=0.0)
    b = _field("layer_linear", layout="element", seed=2, t=10.0)
    assert (a.uv_layout, a.uv_vertical) == ("element", "layer_linear")
    assert a.u.shape == a.v.shape == (K - 1, E)
    out = a.lerp(b, 0.25)
    assert (out.uv_layout, out.uv_vertical) == ("element", "layer_linear")
    lazy = LazyLerpField(a, b, 0.25)
    assert (lazy.uv_layout, lazy.uv_vertical) == ("element", "layer_linear")
    m = lazy.materialize()
    assert (m.uv_layout, m.uv_vertical) == ("element", "layer_linear")
    assert_bitwise(m.u, out.u, "lazy u")
    snaps = [_field("layer_linear", layout="element", seed=10 + i, t=t)
             for i, t in enumerate(_TIMES)]
    rec = reconstruct_pchip(snaps, _TIMES, 13.0)
    assert (rec.uv_layout, rec.uv_vertical) == ("element", "layer_linear")
    arrays, meta = _flatten_vel(a)
    back = _assemble_vel(arrays, meta)
    assert (back.uv_layout, back.uv_vertical) == ("element", "layer_linear")
    assert_bitwise(back.u, a.u, "shared cache u")
    # mixing it with an element layer field still raises
    with pytest.raises(ValueError, match="uv_vertical mismatch"):
        a.lerp(_field("layer", layout="element", seed=3), 0.5)


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
    a = _field("level", seed=1, t=0.0)
    b = _field("level", seed=2, t=10.0)
    out = a.lerp(b, 0.25)
    assert out.uv_vertical == "level"
    # values are the placement-blind blend: same bits as a layer-shaped
    # copy of the same arrays would give
    ref = (1.0 - 0.25) * a.u + 0.25 * b.u
    assert_bitwise(out.u, ref, "lerp u")
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
    a = _field("level", seed=1, t=0.0)
    b = _field("level", seed=2, t=10.0)
    lazy = LazyLerpField(a, b, 0.4)
    assert lazy.uv_vertical == "level"
    m = lazy.materialize()
    assert isinstance(m, VelocityField)
    assert m.uv_vertical == "level"
    eager = a.lerp(b, 0.4)
    for name in ("u", "v", "eta", "w"):
        assert_bitwise(getattr(m, name), getattr(eager, name), name)
    # the placement is a plain attribute: it does not go through
    # __getattr__ (which raises for unknown names) and blends nothing
    fresh = LazyLerpField(a, b, 0.4)
    assert "uv_vertical" in vars(fresh)
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


def _snaps(placements):
    return [_field(p, seed=10 + i, t=_TIMES[i])
            for i, p in enumerate(placements)]


def test_pchip_carries_and_checks():
    out = reconstruct_pchip(_snaps(["level"] * 4), _TIMES, 13.0)
    assert out.uv_vertical == "level"
    out_ll = reconstruct_pchip(_snaps(["layer_linear"] * 4), _TIMES, 13.0)
    assert out_ll.uv_vertical == "layer_linear"
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


def test_shared_cache_roundtrip():
    for placement in PLACEMENTS:
        vel = _field(placement, seed=7)
        arrays, meta = _flatten_vel(vel)
        assert meta["uv_vertical"] == placement
        assert all(isinstance(x, np.ndarray) for x in arrays.values())
        out = _assemble_vel(arrays, meta)
        assert out.uv_vertical == placement
        for name in ("u", "v", "eta", "w"):
            assert_bitwise(getattr(out, name), getattr(vel, name), name)
    # a meta written without the key (an older publisher) reads as "layer"
    arrays, meta = _flatten_vel(_field("layer", seed=8))
    del meta["uv_vertical"]
    assert _assemble_vel(arrays, meta).uv_vertical == "layer"


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
