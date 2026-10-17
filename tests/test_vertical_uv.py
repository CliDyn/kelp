"""Tests for ``kelp.numerics.vertical_uv``: the depth-linear u/v data-point
table, the per-node last usable point, the bracket/read helpers and the
record check (depth-linear engine, Task 3).

All synthetic: a 4-node z-level mesh (levels ``[0, 10, 20, 30]`` m) and
hand-built node-layout fields.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from kelp.core.io import LazyLerpField, VelocityField
from kelp.core.mesh import Mesh

from tests._bitwise import assert_bitwise

LEVELS = [0.0, 10.0, 20.0, 30.0]


def _mesh(levels=LEVELS, depths=(30.0, 20.0, 10.0, 0.0), *,
          vertical_coord="zlevel", decl=None):
    n = len(depths)
    elem = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = Mesh(
        name="t",
        nodes_lon=np.arange(n, dtype=np.float64),
        nodes_lat=np.zeros(n, dtype=np.float64),
        depth=np.asarray(depths, dtype=np.float64),
        elem_nodes=elem,
        elem_center_lon=np.array([1.0]),
        elem_center_lat=np.array([0.0]),
        node_to_elems=np.zeros((n, 1), dtype=np.int32),
        node_to_elems_count=np.ones(n, dtype=np.int32),
        sigma=np.asarray(levels, dtype=np.float64),
        vertical_coord=vertical_coord,
    )
    if decl is not None:
        mesh.metadata["uv_data_last"] = np.asarray(decl, dtype=np.int32)
    return mesh


def _field(u, v=None, *, placement="level", w=None):
    u = np.asarray(u, dtype=np.float64)
    v = np.zeros_like(u) if v is None else np.asarray(v, dtype=np.float64)
    return VelocityField(u=u, v=v, eta=np.zeros(u.shape[1]), w=w,
                         uv_layout="node", uv_vertical=placement)


def _vu():
    from kelp.numerics import vertical_uv
    return vertical_uv


# ---------------------------------------------------------------------------
# The table: data-point depths and the last usable point
# ---------------------------------------------------------------------------

def test_level_points_and_last():
    vu = _vu()
    mesh = _mesh()
    dp = vu.depth_points(mesh, "level")
    assert_bitwise(dp.p, np.array(LEVELS), "p")
    assert_bitwise(dp.last, np.array([3, 2, 1, 0], dtype=np.int32), "last")
    # cached: the same object on a second call
    assert vu.depth_points(mesh, "level") is dp
    # the cached tables cannot be changed by a consumer
    with pytest.raises(ValueError):
        dp.p[0] = 1.0
    with pytest.raises(ValueError):
        dp.last[0] = 0
    # frozen
    with pytest.raises(dataclasses.FrozenInstanceError):
        dp.p = np.zeros(4)


def test_nonfinite_depth_uses_point_zero():
    """``max({k : lower_k <= NaN} U {0}) = 0``: a node without a bed depth
    reads its first point only (never a possibly dry one)."""
    vu = _vu()
    mesh = _mesh(depths=(np.nan, np.inf, 15.0, -np.inf))
    assert vu.depth_points(mesh, "level").last.tolist() == [0, 3, 1, 0]
    assert vu.depth_points(mesh, "layer_linear").last.tolist() == [0, 2, 0, 0]


def test_level_points_follow_z_levels_sign():
    """An ASCII mesh can carry negative level values; the engine reads
    ``mesh.z_levels`` (``|sigma|``), so the table does too."""
    vu = _vu()
    mesh = _mesh(levels=[-0.0, -10.0, -20.0, -30.0])
    dp = vu.depth_points(mesh, "level")
    assert_bitwise(dp.p, np.abs(mesh.sigma), "p")
    assert_bitwise(dp.last, np.array([3, 2, 1, 0], dtype=np.int32), "last")


def test_level_max_levels_cut():
    vu = _vu()
    mesh = _mesh()
    dp1 = vu.depth_points(mesh, "level")
    mesh.sigma = mesh.sigma[:3]          # what run_tracking does for max_levels
    dp2 = vu.depth_points(mesh, "level")
    assert dp2 is not dp1
    assert_bitwise(dp2.p, np.array([0.0, 10.0, 20.0]), "p")
    assert_bitwise(dp2.last, np.array([2, 2, 1, 0], dtype=np.int32), "last")
    assert dp2.p[dp2.last[0]] == 20.0 < mesh.depth[0]


def test_layer_linear_points_partial_bottom():
    vu = _vu()
    mesh = _mesh(depths=(25.0, 5.0, 30.0, 20.0))
    dp = vu.depth_points(mesh, "layer_linear")
    assert_bitwise(dp.p, np.array([5.0, 15.0, 25.0]), "p")
    # depth 25: lower interfaces 10, 20 <= 25 < 30 -> last layer 1 (the
    # partial bottom layer 20..30 is never read); depth 5 -> 0;
    # depth 30 -> 2; depth 20 -> 1
    assert_bitwise(dp.last, np.array([1, 0, 2, 1], dtype=np.int32), "last")
    # the find_layer rule, node by node, for the deepest layer
    from kelp.numerics.vertical import find_layer
    z = mesh.z_levels
    for i, d in enumerate(mesh.depth):
        assert dp.last[i] == find_layer(z, 1.0e9, d)
    # the two placements are cached side by side
    assert vu.depth_points(mesh, "layer_linear") is dp
    assert vu.depth_points(mesh, "level") is not dp


def test_layer_linear_matches_find_layer_on_many_depths():
    vu = _vu()
    from kelp.numerics.vertical import find_layer
    levels = np.array([0.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0])
    depths = np.array([0.0, 2.0, 5.0, 7.5, 10.0, 24.999, 25.0, 49.0, 50.0,
                       150.0, 200.0, 250.0, 1.0e6])
    mesh = _mesh(levels=levels, depths=depths)
    dp = vu.depth_points(mesh, "layer_linear")
    want = [find_layer(levels, 1.0e9, d) for d in depths]
    assert dp.last.tolist() == want


def test_declared_prefix_caps_last():
    vu = _vu()
    mesh = _mesh(decl=[0, 2, 1, 1])
    dp = vu.depth_points(mesh, "level")
    # geometric [3, 2, 1, 0] capped by the declaration [0, 2, 1, 1]
    assert_bitwise(dp.last, np.array([0, 2, 1, 0], dtype=np.int32), "last")
    # -1 (no data in the column) -> point 0 only
    mesh.metadata["uv_data_last"] = np.array([-1, 5, 1, 1], dtype=np.int32)
    dp2 = vu.depth_points(mesh, "level")
    assert dp2 is not dp
    assert_bitwise(dp2.last, np.array([0, 2, 1, 0], dtype=np.int32), "last")
    # and the record check then refuses a non-finite value at point 0
    u = np.ones((4, 4))
    u[0, 0] = np.nan
    with pytest.raises(ValueError, match=r"node 0, point 0"):
        vu.check_depth_linear_record(_field(u), mesh)


def test_declaration_malformed_refused():
    vu = _vu()
    mesh = _mesh()
    mesh.metadata["uv_data_last"] = np.array([0, 1, 2], dtype=np.int32)
    with pytest.raises(ValueError, match="uv_data_last"):
        vu.depth_points(mesh, "level")
    mesh.metadata["uv_data_last"] = np.array([0.0, 1.0, 2.0, 1.0])
    with pytest.raises(ValueError, match="uv_data_last"):
        vu.depth_points(mesh, "level")
    mesh.metadata["uv_data_last"] = np.array([0, -2, 1, 1], dtype=np.int32)
    with pytest.raises(ValueError, match="uv_data_last"):
        vu.depth_points(mesh, "level")


def test_refuses_layer_and_sigma_mesh():
    vu = _vu()
    with pytest.raises(ValueError, match="layer"):
        vu.depth_points(_mesh(), "layer")
    with pytest.raises(ValueError, match="placement"):
        vu.depth_points(_mesh(), "linear")
    with pytest.raises(ValueError, match="zlevel"):
        vu.depth_points(_mesh(levels=[0.0, -0.5, -1.0],
                              vertical_coord="sigma"), "level")
    # a 2-D mesh (no levels) and a single level for layer data
    with pytest.raises(ValueError, match="levels"):
        vu.depth_points(_mesh(levels=[]), "level")
    with pytest.raises(ValueError, match="levels"):
        vu.depth_points(_mesh(levels=[0.0]), "layer_linear")
    # a level table that is not increasing
    with pytest.raises(ValueError, match="increasing"):
        vu.depth_points(_mesh(levels=[0.0, 20.0, 10.0, 30.0]), "level")
    # the record check refuses a layer field (it is not depth-linear)
    with pytest.raises(ValueError, match="layer"):
        vu.check_depth_linear_record(
            _field(np.ones((3, 4)), placement="layer"), _mesh())
    # a layer field is never checked (and asking does not raise)
    assert not vu.is_checked(_field(np.ones((3, 4)), placement="layer"),
                             _mesh())


# ---------------------------------------------------------------------------
# The bracket and the read
# ---------------------------------------------------------------------------

def _value(vu, p, last_i, f, node, z):
    k, a = vu.uv_bracket(p, last_i, z)
    return vu.uv_at(f, node, k, a)


def test_bracket_ends_and_interior():
    vu = _vu()
    p = np.array(LEVELS)
    f = np.array([[1.0], [3.0], [7.0], [15.0]])
    # above / at the first point -> point 0 directly
    assert vu.uv_bracket(p, 3, -5.0) == (0, -1.0)
    assert vu.uv_bracket(p, 3, 0.0) == (0, -1.0)
    # exactly at an interior point -> that point's value
    assert _value(vu, p, 3, f, 0, 10.0) == 3.0
    assert _value(vu, p, 3, f, 0, 20.0) == 7.0
    # between two points -> exact alpha
    k, a = vu.uv_bracket(p, 3, 12.5)
    assert (k, a) == (1, 0.25)
    assert _value(vu, p, 3, f, 0, 12.5) == 0.75 * 3.0 + 0.25 * 7.0
    # at / below the last usable point -> that point directly
    assert vu.uv_bracket(p, 3, 30.0) == (3, -1.0)
    assert vu.uv_bracket(p, 3, 35.0) == (3, -1.0)
    assert vu.uv_bracket(p, 2, 25.0) == (2, -1.0)
    assert vu.uv_bracket(p, 1, 12.5) == (1, -1.0)
    assert vu.uv_bracket(p, 0, 12.5) == (0, -1.0)
    # degenerate spacing -> the upper end of the bracket read directly
    pd = np.array([0.0, 10.0, 10.0 + 1e-13, 20.0])
    k, a = vu.uv_bracket(pd, 3, 10.0 + 0.5e-13)
    assert a == -1.0 and k == 1


def test_bracket_invariants_random():
    vu = _vu()
    rng = np.random.default_rng(3)
    for _ in range(400):
        m = int(rng.integers(1, 8))
        p = np.cumsum(rng.uniform(0.0, 50.0, m))
        p[0] = rng.uniform(0.0, 5.0)
        p = np.maximum.accumulate(p)
        last_i = int(rng.integers(0, m))
        z = float(rng.uniform(-10.0, p[-1] + 10.0))
        k, a = vu.uv_bracket(p, last_i, z)
        assert 0 <= k <= last_i
        if a >= 0.0:
            assert a < 1.0
            assert k + 1 <= last_i
        else:
            assert a == -1.0


def test_exact_level_and_bed_depths():
    """Review Focus 3: a particle exactly at a data point or exactly at the
    bed reads the stored value bit for bit, with NaN stored at every point
    below the last usable one (+inf at one)."""
    vu = _vu()
    for placement, depths in (("level", (30.0, 20.0, 10.0, 0.0)),
                              ("layer_linear", (25.0, 20.0, 30.0, 5.0))):
        mesh = _mesh(depths=depths)
        dp = vu.depth_points(mesh, placement)
        m = dp.p.size
        rng = np.random.default_rng(11)
        f = rng.normal(size=(m, mesh.n_nodes))
        # signed zeros survive too: at the surface and at an interior point
        # whose lower neighbour is positive (a blend with alpha = 0 would
        # give -0.0 + 0.0 = +0.0)
        f[0, 0] = -0.0
        f[1, 0] = -0.0
        f[2, 0] = 1.5
        for i in range(mesh.n_nodes):
            f[dp.last[i] + 1:, i] = np.nan
        dry = [(k, i) for i in range(mesh.n_nodes)
               for k in range(dp.last[i] + 1, m)]
        assert dry, "fixture needs at least one dry point"
        f[dry[0]] = np.inf
        vel = _field(f, f.copy(), placement=placement)
        vu.check_depth_linear_record(vel, mesh)      # dry points are fine
        for i in range(mesh.n_nodes):
            li = int(dp.last[i])
            for k in range(li + 1):
                got = _value(vu, dp.p, li, f, i, float(dp.p[k]))
                assert_bitwise(np.float64(got), f[k, i], f"{placement} k={k} i={i}")
            got = _value(vu, dp.p, li, f, i, float(mesh.depth[i]))
            assert_bitwise(np.float64(got), f[li, i], f"{placement} bed i={i}")
            # anywhere inside the column: finite, never a dry point
            for z in np.linspace(-5.0, 40.0, 91):
                assert np.isfinite(_value(vu, dp.p, li, f, i, float(z)))


def test_uv_at_never_touches_next_point_on_direct_read():
    vu = _vu()
    f = np.array([[2.5, -1.0]])            # one row only: f[1] does not exist
    assert vu.uv_at.py_func(f, 1, 0, -1.0) == -1.0
    assert vu.uv_at(f, 1, 0, -1.0) == -1.0
    f2 = np.array([[2.0], [4.0]])
    assert vu.uv_at(f2, 0, 0, 0.5) == 3.0


# ---------------------------------------------------------------------------
# The record check
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("comp", ["u", "v"])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_check_record_refuses_nan_in_prefix(comp, bad):
    vu = _vu()
    mesh = _mesh(depths=(30.0, 20.0, 10.0, 0.0))     # last = [3, 2, 1, 0]
    i = 1                                            # last[i] == 2
    u = np.ones((4, 4))
    v = np.ones((4, 4))
    (u if comp == "u" else v)[1, i] = bad
    vel = _field(u, v)
    with pytest.raises(ValueError) as ei:
        vu.check_depth_linear_record(vel, mesh, record="t=7")
    msg = str(ei.value)
    assert f"node {i}," in msg
    assert "point 1 " in msg
    assert f" {comp} " in msg
    assert "t=7" in msg
    assert vel._dl_token is None
    assert not vu.is_checked(vel, mesh)


def test_check_record_reports_first_node():
    vu = _vu()
    mesh = _mesh(depths=(30.0, 20.0, 10.0, 0.0))
    u = np.ones((4, 4))
    u[2, 1] = np.nan
    u[0, 2] = np.nan
    u[3, 0] = np.nan
    with pytest.raises(ValueError, match=r"node 0, point 3 "):
        vu.check_depth_linear_record(_field(u), mesh)
    # float32 fields (readers may hand them back) take the same scan
    u32 = u.astype(np.float32)
    vel32 = VelocityField(u=u32, v=np.ones_like(u32), eta=np.zeros(4),
                          uv_layout="node", uv_vertical="level")
    with pytest.raises(ValueError, match=r"node 0, point 3 "):
        vu.check_depth_linear_record(vel32, mesh)
    u32[3, 0] = u32[2, 1] = u32[0, 2] = 1.0
    vu.check_depth_linear_record(vel32, mesh)
    assert vu.is_checked(vel32, mesh)


def test_check_record_accepts_fill_below_last():
    vu = _vu()
    mesh = _mesh(depths=(30.0, 20.0, 10.0, 0.0))     # last = [3, 2, 1, 0]
    u = np.ones((4, 4))
    v = np.ones((4, 4))
    u[3, 1] = np.nan
    u[2:, 2] = 1.0e30
    v[1:, 3] = np.nan
    v[3, 2] = np.inf
    w = np.full((4, 4), np.nan)                      # w is never checked
    vel = _field(u, v, w=w)
    assert not vu.is_checked(vel, mesh)
    vu.check_depth_linear_record(vel, mesh)
    assert vu.is_checked(vel, mesh)
    assert vel._dl_token is vu.depth_points(mesh, "level")
    # a finite fill value inside the prefix is the reader's business
    u2 = u.copy()
    u2[0, 0] = 1.0e30
    vel2 = _field(u2, v)
    vu.check_depth_linear_record(vel2, mesh)
    assert vu.is_checked(vel2, mesh)


def test_check_record_refuses_wrong_shape():
    vu = _vu()
    mesh = _mesh()
    # a K-row field flagged layer_linear (the table has K-1 points)
    with pytest.raises(ValueError, match="rows"):
        vu.check_depth_linear_record(
            _field(np.ones((4, 4)), placement="layer_linear"), mesh)
    # a field on another mesh's nodes
    with pytest.raises(ValueError, match="nodes"):
        vu.check_depth_linear_record(_field(np.ones((4, 5))), mesh)


def test_token_invalidated():
    vu = _vu()
    mesh_a = _mesh(depths=(30.0, 20.0, 10.0, 0.0))   # last = [3, 2, 1, 0]
    mesh_b = _mesh(depths=(30.0, 20.0, 10.0, 0.0))   # same shapes and values
    u = np.ones((4, 4))
    u[1, 3] = np.nan                                 # below node 3's last (0)
    u[3, 2] = np.nan                                 # below node 2's last (1)
    vel = _field(u)
    vu.check_depth_linear_record(vel, mesh_a)
    assert vu.is_checked(vel, mesh_a)
    # another mesh, even an identical one
    assert not vu.is_checked(vel, mesh_b)

    # max_levels cut
    vu.check_depth_linear_record(vel, mesh_b)
    assert vu.is_checked(vel, mesh_b)
    mesh_b.sigma = mesh_b.sigma[:3]
    assert not vu.is_checked(vel, mesh_b)
    with pytest.raises(ValueError, match="rows"):    # the field is not cut
        vu.check_depth_linear_record(vel, mesh_b)
    vel_cut = _field(u[:3])
    vu.check_depth_linear_record(vel_cut, mesh_b)
    assert vu.is_checked(vel_cut, mesh_b)

    # a new mesh.depth: node 3 deepens, its NaN at point 1 becomes usable
    vu.check_depth_linear_record(vel, mesh_a)
    assert vu.is_checked(vel, mesh_a)
    mesh_a.depth = np.array([30.0, 20.0, 10.0, 30.0])
    assert not vu.is_checked(vel, mesh_a)
    with pytest.raises(ValueError, match=r"node 3, point 1 "):
        vu.check_depth_linear_record(vel, mesh_a)

    # a new uv_data_last: node 2's NaN at point 3 becomes usable
    mesh_c = _mesh(depths=(30.0, 20.0, 30.0, 0.0), decl=[3, 2, 2, 0])
    vel_c = _field(u)
    vu.check_depth_linear_record(vel_c, mesh_c)
    assert vu.is_checked(vel_c, mesh_c)
    mesh_c.metadata["uv_data_last"] = np.array([3, 2, 3, 0], dtype=np.int32)
    assert not vu.is_checked(vel_c, mesh_c)
    with pytest.raises(ValueError, match=r"node 2, point 3 "):
        vu.check_depth_linear_record(vel_c, mesh_c)
    # the declaration removed altogether: also a new table
    vu.check_depth_linear_record(_field(np.ones((4, 4))), mesh_c)
    del mesh_c.metadata["uv_data_last"]
    tok = vu.depth_points(mesh_c, "level")
    assert tok.last.tolist() == [3, 2, 3, 0]


def test_cache_survives_id_reuse():
    vu = _vu()
    mesh = _mesh()
    dp1 = vu.depth_points(mesh, "level")
    entry = mesh._uv_depth_points["level"]
    assert entry.sigma is mesh.sigma and entry.depth is mesh.depth
    assert entry.decl is None and entry.points is dp1
    del entry
    seen = [dp1]
    for _ in range(2):
        mesh.depth = mesh.depth + 0.0      # no other reference to the old one
        dp = vu.depth_points(mesh, "level")
        assert all(dp is not old for old in seen)
        assert_bitwise(dp.last, dp1.last, "last")
        entry = mesh._uv_depth_points["level"]
        assert entry.depth is mesh.depth and entry.sigma is mesh.sigma
        assert entry.points is dp
        del entry
        seen.append(dp)


# ---------------------------------------------------------------------------
# Lazy fields (ruling R2): parents checked, no blend
# ---------------------------------------------------------------------------

def test_check_record_lazy_field_checks_parents_without_blend(monkeypatch):
    vu = _vu()
    import kelp.core.io as kio
    mesh = _mesh()
    a = _field(np.ones((4, 4)))
    b = _field(np.full((4, 4), 2.0))
    lazy = LazyLerpField(a, b, 0.25)
    assert lazy._dl_token is None

    blends = []

    def no_blend(*args, **kwargs):
        blends.append(args)
        raise AssertionError("the record check blended a lazy field")

    monkeypatch.setattr(kio, "_blend", no_blend)
    checked = []
    real = vu._check_components

    def spy(vel, *args, **kwargs):
        checked.append(vel)
        return real(vel, *args, **kwargs)

    monkeypatch.setattr(vu, "_check_components", spy)

    vu.check_depth_linear_record(a, mesh)          # vel_a already checked
    checked.clear()
    assert not vu.is_checked(lazy, mesh)
    vu.check_depth_linear_record(lazy, mesh, record="t=3")
    assert blends == []
    assert lazy._cache == {}
    assert "u" not in vars(lazy) and "v" not in vars(lazy)
    assert [c is b for c in checked] == [True]     # only the unchecked parent
    assert vu.is_checked(a, mesh) and vu.is_checked(b, mesh)
    assert vu.is_checked(lazy, mesh)
    assert lazy._dl_token is vu.depth_points(mesh, "level")

    # a NaN in one parent's prefix: refused, the lazy field stays unchecked
    c = _field(np.ones((4, 4)))
    bad_u = np.ones((4, 4))
    bad_u[2, 0] = np.nan
    d = _field(bad_u)
    lazy2 = LazyLerpField(c, d, 0.5)
    with pytest.raises(ValueError, match=r"node 0, point 2 .*t=9"):
        vu.check_depth_linear_record(lazy2, mesh, record="t=9")
    assert lazy2._dl_token is None
    assert blends == []


# ---------------------------------------------------------------------------
# Mesh: the new cache field does not leak through copies
# ---------------------------------------------------------------------------

def test_mesh_field_declared_and_copies_do_not_share():
    vu = _vu()
    f = {x.name: x for x in dataclasses.fields(Mesh)}["_uv_depth_points"]
    assert f.default is None and f.repr is False and f.compare is False
    mesh = _mesh()
    assert mesh._uv_depth_points is None
    assert "_uv_depth_points" not in repr(mesh)
    dp = vu.depth_points(mesh, "level")

    # dataclasses.replace / copy.copy share the dict object at first; a
    # rebuild on the copy must not change the original's entry
    for clone in (dataclasses.replace(mesh, depth=mesh.depth + 5.0),
                  copy.copy(mesh)):
        clone.depth = clone.depth + 5.0
        dpc = vu.depth_points(clone, "level")
        assert dpc is not dp
        assert vu.depth_points(mesh, "level") is dp
        assert mesh._uv_depth_points["level"].depth is mesh.depth

    # pickling and deep copies carry consistent tables but a field checked
    # on the original is unchecked on the copy (another DepthPoints object)
    vel = _field(np.ones((4, 4)))
    vu.check_depth_linear_record(vel, mesh)
    for clone in (pickle.loads(pickle.dumps(mesh)), copy.deepcopy(mesh)):
        dpc = vu.depth_points(clone, "level")
        assert_bitwise(dpc.p, dp.p, "p")
        assert_bitwise(dpc.last, dp.last, "last")
        assert not vu.is_checked(vel, clone)
        assert vu.is_checked(vel, mesh)


# ---------------------------------------------------------------------------
# Numba: jitted == py_func == NUMBA_DISABLE_JIT
# ---------------------------------------------------------------------------

_P_CASES = [
    [0.0, 10.0, 20.0, 30.0],
    [5.0, 15.0, 25.0],
    [0.0, 10.0, 10.0 + 1e-13, 20.0],
    [2.5],
]
_Z_CASES = [-7.0, 0.0, 2.5, 5.0, 9.999, 10.0, 10.0 + 0.5e-13, 12.5, 15.0,
            19.0, 20.0, 24.0, 25.0, 30.0, 31.0, 1e9]


def _bracket_cases():
    out = []
    for p in _P_CASES:
        for last_i in range(len(p)):
            for z in _Z_CASES:
                out.append((p, last_i, z))
    return out


def _f_for(p):
    m = len(p)
    rng = np.random.default_rng(m)
    return rng.normal(size=(m, 3))


def test_py_func_matches():
    vu = _vu()
    if not hasattr(vu.uv_bracket, "py_func"):
        pytest.skip("numba not available")
    for p, last_i, z in _bracket_cases():
        pa = np.asarray(p, dtype=np.float64)
        kj, aj = vu.uv_bracket(pa, last_i, z)
        kp, ap = vu.uv_bracket.py_func(pa, last_i, z)
        assert int(kj) == int(kp) and float(aj).hex() == float(ap).hex(), \
            (p, last_i, z)
        for f in (_f_for(p), _f_for(p).astype(np.float32)):
            for node in range(f.shape[1]):
                vj = vu.uv_at(f, node, kj, aj)
                vp = vu.uv_at.py_func(f, node, kp, ap)
                assert float(vj).hex() == float(vp).hex(), (p, last_i, z)
    # the record scan
    f = np.ones((4, 70))
    f[2, 65] = np.nan
    f[1, 3] = np.inf
    last = np.full(70, 3, dtype=np.int32)
    last[3] = 0
    out_j = np.empty(70, dtype=np.int64)
    out_p = np.empty(70, dtype=np.int64)
    vu._first_nonfinite_per_node(f, last, out_j)
    vu._first_nonfinite_per_node.py_func(f, last, out_p)
    assert_bitwise(out_j, out_p, "scan")
    assert out_j[65] == 2 and out_j[3] == -1
    assert int((out_j >= 0).sum()) == 1


_SUBPROCESS_SCRIPT = r"""
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import numpy as np
from numba.core import config
assert config.DISABLE_JIT, "NUMBA_DISABLE_JIT not honoured"
import kelp.numerics.vertical_uv as vu
assert str(Path(vu.__file__).resolve()).startswith(
    str(Path(sys.argv[1]).resolve())), vu.__file__
cases = json.loads(sys.argv[2])
rows = []
for p, last_i, z in cases:
    pa = np.asarray(p, dtype=np.float64)
    k, a = vu.uv_bracket(pa, last_i, z)
    m = len(p)
    f = np.random.default_rng(m).normal(size=(m, 3))
    vals = [float(vu.uv_at(f, n, k, a)).hex() for n in range(3)]
    rows.append([int(k), float(a).hex(), vals])
print(json.dumps(rows))
"""


def test_numba_disabled_subprocess_matches(tmp_path):
    vu = _vu()
    cases = _bracket_cases()
    script = tmp_path / "vertical_uv_nojit.py"
    script.write_text(_SUBPROCESS_SCRIPT)
    src = str(Path(__file__).resolve().parents[1] / "src")
    env = dict(os.environ, NUMBA_DISABLE_JIT="1")
    out = subprocess.run(
        [sys.executable, str(script), src, json.dumps(cases)],
        capture_output=True, text=True, timeout=300, env=env)
    assert out.returncode == 0, out.stderr
    rows = json.loads(out.stdout)
    assert len(rows) == len(cases)
    for (p, last_i, z), (k, a_hex, vals) in zip(cases, rows):
        pa = np.asarray(p, dtype=np.float64)
        kj, aj = vu.uv_bracket(pa, last_i, z)
        assert (int(kj), float(aj).hex()) == (k, a_hex), (p, last_i, z)
        f = _f_for(p)
        assert [float(vu.uv_at(f, n, kj, aj)).hex() for n in range(3)] \
            == vals, (p, last_i, z)
