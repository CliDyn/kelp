"""Tests for ``kelp.numerics.vertical_uv``: the depth-linear u/v data-point
table, the per-column last usable point and bed (nodes and elements), the
wet test, the bracket/read helpers and the record check.

All synthetic: a 4-node z-level mesh (levels ``[0, 10, 20, 30]`` m) and
hand-built node-layout fields; a 3-element mesh with a hand-set element
declaration and bed for the element table.
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


def test_declaration_snapshot_at_first_use():
    """The table keeps the declaration by identity, so the engine builds it
    from its own snapshot: an owned, read-only copy that replaces the
    reader's array in ``mesh.metadata``.  A later write into the reader's
    array changes nothing, a write into the snapshot raises, and a new array
    is a new declaration (snapshot in turn)."""
    vu = _vu()
    decl = np.array([0, 2, 1, 1], dtype=np.int32)
    mesh = _mesh()
    mesh.metadata["uv_data_last"] = decl
    dp = vu.depth_points(mesh, "level")
    snap = mesh.metadata["uv_data_last"]
    assert snap is not decl and snap.base is None
    assert snap.dtype == decl.dtype and snap.tolist() == [0, 2, 1, 1]
    assert not snap.flags.writeable and decl.flags.writeable
    assert mesh._uv_depth_points[("level", "node")].decl is snap
    # the reader's own array is no longer the declaration
    decl[:] = 3
    assert vu.depth_points(mesh, "level") is dp
    assert mesh.metadata["uv_data_last"] is snap
    assert snap.tolist() == [0, 2, 1, 1]
    with pytest.raises(ValueError, match="read-only"):
        snap[0] = 3
    # the other placement reuses the snapshot, it does not copy it again
    vu.depth_points(mesh, "layer_linear")
    assert mesh.metadata["uv_data_last"] is snap
    assert mesh._uv_depth_points[("layer_linear", "node")].decl is snap
    # a new array is a new declaration
    mesh.metadata["uv_data_last"] = np.array([3, 2, 1, 0], dtype=np.int32)
    assert vu.depth_points(mesh, "level").last.tolist() == [3, 2, 1, 0]


def test_declaration_snapshot_shared_by_shallow_copies():
    """Two shallow copies made before either built a table share one
    metadata dict: the second reuses the first's snapshot instead of
    replacing it, so alternating lookups settle after one build each."""
    import copy
    vu = _vu()
    mesh = _mesh()
    mesh.metadata["uv_data_last"] = np.array([0, 2, 1, 1], dtype=np.int32)
    a, b = copy.copy(mesh), copy.copy(mesh)
    assert a.metadata is b.metadata
    dp_a = vu.depth_points(a, "level")
    snap = a.metadata["uv_data_last"]
    dp_b = vu.depth_points(b, "level")
    assert b.metadata["uv_data_last"] is snap
    for _ in range(3):
        assert vu.depth_points(a, "level") is dp_a
        assert vu.depth_points(b, "level") is dp_b
    assert a.metadata["uv_data_last"] is snap
    assert dp_b.last.tolist() == dp_a.last.tolist() == [0, 2, 1, 0]


def test_reader_frozen_array_is_copied():
    """A read-only array the READER froze is not an engine snapshot: the
    build copies it like any other declaration."""
    vu = _vu()
    decl = np.array([0, 2, 1, 1], dtype=np.int32)
    decl.setflags(write=False)
    mesh = _mesh()
    mesh.metadata["uv_data_last"] = decl
    vu.depth_points(mesh, "level")
    snap = mesh.metadata["uv_data_last"]
    assert snap is not decl and snap.base is None
    assert vu._SNAPSHOTS.get(id(snap)) is snap
    assert vu._SNAPSHOTS.get(id(decl)) is None


def test_snapshot_registry_forgets_freed_snapshots():
    """A freed snapshot drops out of the registry, so an array that later
    gets its id is never taken for an engine snapshot."""
    import gc
    vu = _vu()
    mesh = _mesh()
    mesh.metadata["uv_data_last"] = np.array([0, 2, 1, 1], dtype=np.int32)
    vu.depth_points(mesh, "level")
    old_id = id(mesh.metadata["uv_data_last"])
    assert vu._SNAPSHOTS.get(old_id) is not None
    del mesh
    gc.collect()
    assert vu._SNAPSHOTS.get(old_id) is None


@pytest.mark.parametrize("alias", ["base", "view"])
def test_declaration_alias_cannot_change_the_table(alias):
    """A write through the reader's base array, or through a writable view
    of its declaration taken before the first build, reaches neither the
    declaration in effect nor a table built later: both come from the
    engine's snapshot.  Surface-only at the build, then "all five levels"
    through the alias, then a rebuild (a new bed array, same values): a new
    record with u = 1 at the surface and a finite fill 1e30 below passes the
    record check judged against the snapshot, and the particle (jitted
    kernels) reads 1.0, never the fill.  The other way round (five levels,
    then the surface only through the alias) the declaration in effect
    keeps the table's five levels."""
    from tests.test_interp_depth_linear import _mesh as _mesh_3x3
    from tests.test_interp_depth_linear import _run, _state
    vu = _vu()

    def declared(value):
        """The declaration and a write through its alias."""
        if alias == "base":
            owner = np.full((2, 9), value, dtype=np.int32)

            def write(v):
                owner[0, :] = v
            return owner[0], write
        decl = np.full(9, value, dtype=np.int32)
        view = decl[:]                 # writable, taken before the build

        def write(v):
            view[:] = v
        return decl, write

    # levels 0..40 m, every bed below the deepest level
    mesh = _mesh_3x3(depths=np.full(9, 200.0))
    decl, write = declared(0)          # surface only
    mesh.metadata["uv_data_last"] = decl
    dp = vu.depth_points(mesh, "level")
    write(4)
    mesh.depth = mesh.depth.copy()     # rebuild from the declaration in effect
    u = np.full((5, 9), 1.0e30)
    u[0] = 1.0
    vel = VelocityField(u=u, v=np.zeros_like(u), eta=np.zeros(9),
                        w=np.zeros((5, 9)), uv_layout="node",
                        uv_vertical="level")
    vu.check_depth_linear_record(vel, mesh)
    u_p, _, ok = _run(mesh, vel, _state([(4.3, 4.2)], 30.0))
    assert ok[0] and u_p[0] == 1.0
    dp2 = vu.depth_points(mesh, "level")
    assert dp2 is not dp and dp2.last.tolist() == [0] * 9
    assert mesh.metadata["uv_data_last"].tolist() == [0] * 9
    # judged against the snapshot: NaN below the surface is never read,
    # NaN at the surface is refused
    u[1:] = np.nan
    vu.check_depth_linear_record(_field(u), mesh)
    u[0, 3] = np.nan
    with pytest.raises(ValueError, match=r"node 3, point 0 "):
        vu.check_depth_linear_record(_field(u), mesh)

    # the reverse: five levels at the build, the surface only through the
    # alias afterwards — the declaration in effect is still the table's
    mesh = _mesh_3x3(depths=np.full(9, 200.0))
    decl, write = declared(4)
    mesh.metadata["uv_data_last"] = decl
    dp = vu.depth_points(mesh, "level")
    write(0)
    assert mesh.metadata["uv_data_last"].tolist() == [4] * 9
    assert vu.depth_points(mesh, "level") is dp
    assert dp.last.tolist() == [4] * 9


@pytest.mark.parametrize("placement", ["level", "layer_linear"])
@pytest.mark.parametrize("how", ["deepcopy", "pickle"])
def test_copied_mesh_revalidates_its_declaration(placement, how):
    """A deep copy or an unpickled mesh (default protocol) carries a NEW,
    WRITEABLE declaration array (numpy drops the read-only flag of the
    snapshot), while its table cache still matches it by identity: the
    first use of the copy rebuilds the table from the current values — an
    in-place write made before it is seen — and installs a new read-only
    snapshot, so a later write into the copy's old array changes nothing
    and a write into the snapshot raises."""
    import copy
    import pickle
    vu = _vu()
    mesh = _mesh(decl=[3, 2, 1, 0])
    dp0 = vu.depth_points(mesh, placement)
    assert not mesh.metadata["uv_data_last"].flags.writeable
    clone = (copy.deepcopy(mesh) if how == "deepcopy"
             else pickle.loads(pickle.dumps(mesh)))
    decl = clone.metadata["uv_data_last"]
    assert decl.flags.writeable                     # numpy's copy semantics
    decl[0] = 0                                     # unseen without the fix
    dp = vu.depth_points(clone, placement)
    assert dp.last[0] == 0 and dp0.last[0] > 0
    snap = clone.metadata["uv_data_last"]
    assert snap is not decl and not snap.flags.writeable
    assert snap.tolist() == [0, 2, 1, 0]
    with pytest.raises(ValueError, match="read-only"):
        snap[1] = 0
    decl[1] = 0                                     # no longer the declaration
    assert vu.depth_points(clone, placement) is dp   # snapshot: cached
    # the original is untouched
    assert vu.depth_points(mesh, placement) is dp0


def test_declared_prefix_caps_last():
    vu = _vu()
    mesh = _mesh(decl=[0, 2, 1, 1])
    dp = vu.depth_points(mesh, "level")
    # geometric [3, 2, 1, 0] capped by the declaration [0, 2, 1, 1]
    assert_bitwise(dp.last, np.array([0, 2, 1, 0], dtype=np.int32), "last")
    # -1 (surface only) -> point 0 only, and it must be finite (below)
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


def test_refuses_layer_placement():
    """A ``"layer"`` field has no table (sigma meshes: see the sigma
    section below, which builds one for ``"layer_linear"``)."""
    vu = _vu()
    with pytest.raises(ValueError, match="layer"):
        vu.depth_points(_mesh(), "layer")
    with pytest.raises(ValueError, match="placement"):
        vu.depth_points(_mesh(), "linear")
    with pytest.raises(ValueError, match="layer"):
        vu.depth_points(_mesh(levels=[1.0, 0.5, 0.0],
                              vertical_coord="sigma"), "layer")
    with pytest.raises(ValueError, match=r"needs a z-level or sigma mesh"):
        vu.depth_points(_mesh(vertical_coord="other"), "level")
    # a 2-D mesh (no levels) and a single level for layer data
    with pytest.raises(ValueError, match="levels"):
        vu.depth_points(_mesh(levels=[]), "level")
    with pytest.raises(ValueError, match="levels"):
        vu.depth_points(_mesh(levels=[0.0]), "layer_linear")
    # a 2-D sigma mesh (the FESOM-C reader's [0.0] without sigma_lev)
    with pytest.raises(ValueError, match=r"needs a 3-D sigma mesh with >= 2 "
                                         r"interfaces in mesh.sigma"):
        vu.depth_points(_mesh(levels=[0.0], vertical_coord="sigma"),
                        "layer_linear")
    # a level table that decreases somewhere
    with pytest.raises(ValueError, match="non-decreasing"):
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
    """A particle exactly at a data point or exactly at the bed reads the
    stored value bit for bit, with NaN stored at every point below the last
    usable one (+inf at one)."""
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


def test_bracket_coincident_points_reads_the_deeper():
    """A depth exactly at two coincident points reads the deeper one (the
    larger index), as the docstring says; just above them the bracket
    ends at the shallower one."""
    vu = _vu()
    p = np.array([0.0, 10.0, 10.0, 20.0])
    assert vu.uv_bracket(p, 3, 10.0) == (2, -1.0)
    k, a = vu.uv_bracket(p, 3, 9.0)
    assert k == 0 and a == 0.9
    k, a = vu.uv_bracket(p, 3, 15.0)
    assert k == 2 and a == 0.5
    # the last usable point is the shallower of the pair: read it directly
    assert vu.uv_bracket(p, 1, 10.0) == (1, -1.0)


def test_bracket_coincident_points_at_the_surface_read_point_zero():
    """The surface exception to the deeper-of-two rule: at or above
    ``p[0]`` the bracket reads point 0 even when point 1 coincides with it;
    just below, the blend starts at point 1 (the deeper of the pair)."""
    vu = _vu()
    p = np.array([0.0, 0.0, 10.0])
    f = np.array([[1.0], [2.0], [4.0]])
    for fn in (vu.uv_bracket, vu.uv_bracket.py_func):
        assert fn(p, 2, 0.0) == (0, -1.0)
        assert fn(p, 2, -3.0) == (0, -1.0)
        k, a = fn(p, 2, 5.0)
        assert k == 1 and a == 0.5
    assert vu.uv_at(f, 0, *vu.uv_bracket(p, 2, 0.0)) == 1.0
    assert vu.uv_at(f, 0, *vu.uv_bracket(p, 2, 5.0)) == 3.0
    # an interior pair keeps the deeper-of-two rule
    p2 = np.array([0.0, 10.0, 10.0, 20.0])
    assert vu.uv_bracket(p2, 3, 10.0) == (2, -1.0)


def test_bracket_nan_depth():
    """A NaN depth (pinned as the docstring states): ``k = last_i - 1``
    with a NaN alpha when ``last_i > 0`` (a NaN value, never a point past
    ``last_i``), point 0 directly when ``last_i == 0``, point ``last_i - 1``
    directly on a degenerate last spacing.  The kernels map a NaN depth to
    ``+inf`` before the bracket: that reads ``last_i`` directly."""
    vu = _vu()
    p = np.array(LEVELS)
    for last_i in (1, 2, 3):
        k, a = vu.uv_bracket(p, last_i, np.nan)
        assert k == last_i - 1 and np.isnan(a), (last_i, k, a)
        f = np.arange(4.0)[:, None]
        assert np.isnan(vu.uv_at(f, 0, k, a))
        assert vu.uv_bracket(p, last_i, np.inf) == (last_i, -1.0)
    assert vu.uv_bracket(p, 0, np.nan) == (0, -1.0)
    assert vu.uv_bracket(p, 0, np.inf) == (0, -1.0)
    pd = np.array([0.0, 10.0, 10.0 + 1e-13])
    assert vu.uv_bracket(pd, 2, np.nan) == (1, -1.0)
    if hasattr(vu.uv_bracket, "py_func"):
        for last_i in (0, 1, 3):
            kj, aj = vu.uv_bracket(p, last_i, np.nan)
            kp, ap = vu.uv_bracket.py_func(p, last_i, np.nan)
            assert int(kj) == int(kp)
            assert float(aj).hex() == float(ap).hex()


def test_table_error_names_first_bad_level_only():
    """A non-finite or decreasing level table is refused naming its size
    and the first offending level, never printing the whole table."""
    vu = _vu()
    levels = list(np.arange(80) * 10.0)
    levels[41] = 395.0                         # shallower than level 40
    with pytest.raises(ValueError) as exc:
        vu.depth_points(_mesh(levels=levels, depths=(30.0,) * 4), "level")
    msg = str(exc.value)
    assert "80 levels, level 41 (395 m) is shallower than level 40 (400 m)" \
        in msg, msg
    assert "790" not in msg and len(msg) < 300
    levels = list(np.arange(80) * 10.0)
    levels[7] = np.nan
    with pytest.raises(ValueError, match=r"finite .*80 levels, level 7 is "
                                         r"nan"):
        vu.depth_points(_mesh(levels=levels, depths=(30.0,) * 4),
                        "layer_linear")


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
    entry = mesh._uv_depth_points[("level", "node")]
    assert entry.sigma is mesh.sigma and entry.depth is mesh.depth
    assert entry.decl is None and entry.points is dp1
    del entry
    seen = [dp1]
    for _ in range(2):
        mesh.depth = mesh.depth + 0.0      # no other reference to the old one
        dp = vu.depth_points(mesh, "level")
        assert all(dp is not old for old in seen)
        assert_bitwise(dp.last, dp1.last, "last")
        entry = mesh._uv_depth_points[("level", "node")]
        assert entry.depth is mesh.depth and entry.sigma is mesh.sigma
        assert entry.points is dp
        del entry
        seen.append(dp)


# ---------------------------------------------------------------------------
# Lazy fields: parents checked, no blend
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
        assert mesh._uv_depth_points[("level", "node")].depth is mesh.depth

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
# The bed of every column; the element table, its record check; uv_wet
# ---------------------------------------------------------------------------

#: levels of the element-table mesh; layer centres [5, 20, 45]
LEVELS_E = [0.0, 10.0, 30.0, 60.0]


def _elem_mesh(levels=LEVELS_E, *, decl=(2, 0, 1), bed=(60.0, 25.0, 30.0),
               depths=(60.0, 60.0, 60.0, 60.0, 60.0)):
    """5 nodes, 3 triangles; the element declaration
    ``metadata["uv_data_last_elem"]`` and bed ``metadata["elem_bed"]`` set
    by hand, as a reader declares them (``None`` leaves one out)."""
    elem = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4]], dtype=np.int32)
    lon = np.array([0.0, 1.0, 0.5, 1.5, 1.0])
    lat = np.array([0.0, 0.0, 1.0, 1.0, 2.0])
    n = lon.size
    mesh = Mesh(
        name="t",
        nodes_lon=lon,
        nodes_lat=lat,
        depth=np.asarray(depths, dtype=np.float64),
        elem_nodes=elem,
        elem_center_lon=lon[elem].mean(axis=1),
        elem_center_lat=lat[elem].mean(axis=1),
        node_to_elems=np.zeros((n, 1), dtype=np.int32),
        node_to_elems_count=np.ones(n, dtype=np.int32),
        sigma=np.asarray(levels, dtype=np.float64),
        vertical_coord="zlevel",
    )
    if decl is not None:
        mesh.metadata["uv_data_last_elem"] = np.asarray(decl, dtype=np.int32)
    if bed is not None:
        mesh.metadata["elem_bed"] = np.asarray(bed, dtype=np.float64)
    return mesh


def _efield(u, v=None, *, w=None):
    """An element-layout ``"layer_linear"`` field on ``_elem_mesh``."""
    u = np.asarray(u, dtype=np.float64)
    v = np.zeros_like(u) if v is None else np.asarray(v, dtype=np.float64)
    return VelocityField(u=u, v=v, eta=np.zeros(5), w=w,
                         uv_layout="element", uv_vertical="layer_linear")


def _etable(vu, mesh):
    return vu.depth_points(mesh, "layer_linear", layout="element")


def test_bed_nodes_is_depth():
    """A node's bed is its physical bed, ``mesh.depth``: a read-only
    float64 copy, the same for both placements, next to ``last``."""
    vu = _vu()
    mesh = _mesh(levels=[0.0, 10.0, 30.0, 60.0], depths=(60.0, 25.0, 10.0))
    dp = vu.depth_points(mesh, "layer_linear")
    assert_bitwise(dp.last, np.array([2, 0, 0], dtype=np.int32), "last")
    assert_bitwise(dp.bed, np.array([60.0, 25.0, 10.0]), "bed")
    assert dp.bed.dtype == np.float64
    assert not np.shares_memory(dp.bed, mesh.depth)
    with pytest.raises(ValueError, match="read-only"):
        dp.bed[0] = 0.0
    assert mesh.depth[0] == 60.0
    dpl = vu.depth_points(mesh, "level")
    assert_bitwise(dpl.bed, np.array([60.0, 25.0, 10.0]), "bed (level)")
    # the explicit node layout is the default one
    assert vu.depth_points(mesh, "layer_linear", layout="node") is dp


def test_elem_table_from_declaration():
    """The element table: ``p`` the layer centres, ``last`` the reader's
    declaration clipped to ``0 .. M-1`` (no bathymetry rule on top: the
    declaration is geometric already), ``bed`` a read-only copy of
    ``metadata["elem_bed"]``; ``"level"`` data are node data."""
    vu = _vu()
    mesh = _elem_mesh()
    dp = _etable(vu, mesh)
    assert_bitwise(dp.p, np.array([5.0, 20.0, 45.0]), "p")
    assert_bitwise(dp.last, np.array([2, 0, 1], dtype=np.int32), "last")
    assert_bitwise(dp.bed, np.array([60.0, 25.0, 30.0]), "bed")
    assert not np.shares_memory(dp.bed, mesh.metadata["elem_bed"])
    for name in ("p", "last", "bed"):
        with pytest.raises(ValueError, match="read-only"):
            getattr(dp, name)[0] = 0
    assert _etable(vu, mesh) is dp
    # the bed does not cap last (wet is decided by the bed, the read by
    # last): a shallower bed array is a new table with the same last
    mesh.metadata["elem_bed"] = np.array([5.0, 5.0, 5.0])
    dp2 = _etable(vu, mesh)
    assert dp2 is not dp
    assert dp2.last.tolist() == [2, 0, 1]
    assert dp2.bed.tolist() == [5.0, 5.0, 5.0]
    # -1 (surface only) -> point 0; past the table -> M-1
    mesh.metadata["uv_data_last_elem"] = np.array([-1, 7, 1], dtype=np.int32)
    assert _etable(vu, mesh).last.tolist() == [0, 2, 1]
    with pytest.raises(ValueError, match="layer_linear"):
        vu.depth_points(mesh, "level", layout="element")
    with pytest.raises(ValueError, match="layout"):
        vu.depth_points(mesh, "layer_linear", layout="nodes")


def test_elem_table_missing_declaration_refused():
    vu = _vu()
    mesh = _elem_mesh(decl=None)
    with pytest.raises(ValueError, match="uv_data_last_elem") as ei:
        _etable(vu, mesh)
    assert "nlevels" in str(ei.value)
    assert mesh._uv_depth_points is None
    # the node table needs neither
    vu.depth_points(mesh, "layer_linear")


def test_elem_table_missing_bed_refused():
    vu = _vu()
    mesh = _elem_mesh(bed=None)
    with pytest.raises(ValueError, match="uv_data_last_elem") as ei:
        _etable(vu, mesh)
    assert "nlevels" in str(ei.value) and "elem_bed" in str(ei.value)
    # the declaration is not snapshotted by a refused build
    assert mesh.metadata["uv_data_last_elem"].flags.writeable


def test_elem_table_cavities_refused():
    """``elem_ulevels`` above 1 anywhere (an ice-shelf cavity) refuses the
    element table, naming the mesh file and the first cavity element, so
    a direct step call meets the refusal the run meets; ``ulevels`` all 1
    builds the table of the mesh without the variable; the node table
    does not look at it."""
    vu = _vu()
    mesh = _elem_mesh()
    mesh.metadata["meshpath"] = "cav/fesom.mesh.diag.nc"
    mesh.metadata["elem_ulevels"] = np.array([1, 3, 1], dtype=np.int32)
    with pytest.raises(ValueError, match=(
            r"^ice-shelf cavities are not supported by mode\.uv_vertical: "
            r"linear on element data yet: mesh 'cav/fesom\.mesh\.diag\.nc' "
            r"has ulevels > 1 at 1 element\(s\) \(first: element 1, "
            r"ulevels 3\)")):
        _etable(vu, mesh)
    assert mesh._uv_depth_points is None
    vu.depth_points(mesh, "layer_linear")
    flat = _elem_mesh()
    flat.metadata["elem_ulevels"] = np.ones(3, dtype=np.int32)
    got, ref = _etable(vu, flat), _etable(vu, _elem_mesh())
    assert_bitwise(got.last, ref.last, "last")
    assert_bitwise(got.bed, ref.bed, "bed")


def test_elem_table_malformed_refused():
    vu = _vu()
    for decl, bad in (([2, 0], "uv_data_last_elem"),
                      ([2.0, 0.0, 1.0], "uv_data_last_elem"),
                      ([2, -2, 1], "uv_data_last_elem")):
        mesh = _elem_mesh(decl=None)
        mesh.metadata["uv_data_last_elem"] = np.asarray(decl)
        with pytest.raises(ValueError, match=bad):
            _etable(vu, mesh)
    mesh = _elem_mesh(bed=(60.0, 25.0))
    with pytest.raises(ValueError, match="elem_bed"):
        _etable(vu, mesh)


def test_bed_independent_of_cut():
    """A ``max_levels`` cut lowers ``last``, never ``bed``: a column stays
    wet down to its physical bed and reads its last kept point below it —
    nodes and elements alike."""
    vu = _vu()
    mesh = _elem_mesh(decl=(2, 2, 0), bed=(60.0, 60.0, 10.0),
                      depths=(60.0, 25.0, 10.0, 60.0, 60.0))
    dpe = _etable(vu, mesh)
    dpn = vu.depth_points(mesh, "layer_linear")
    assert dpe.last.tolist() == [2, 2, 0]
    assert dpn.last.tolist() == [2, 0, 0, 2, 2]
    mesh.sigma = mesh.sigma[:3]          # [0, 10, 30], as max_levels cuts
    dpe2 = _etable(vu, mesh)
    dpn2 = vu.depth_points(mesh, "layer_linear")
    assert dpe2 is not dpe and dpn2 is not dpn
    assert_bitwise(dpe2.last, np.array([1, 1, 0], dtype=np.int32), "last")
    assert_bitwise(dpe2.bed, np.array([60.0, 60.0, 10.0]), "elem bed")
    assert_bitwise(dpn2.last, np.array([1, 0, 0, 1, 1], dtype=np.int32),
                   "node last")
    assert_bitwise(dpn2.bed, np.array([60.0, 25.0, 10.0, 60.0, 60.0]),
                   "node bed")
    # below its last kept point (the centre 20 m) the column is still
    # water, down to its bed at 60 m
    assert dpe2.p[dpe2.last[0]] == 20.0 and dpe2.bed[0] == 60.0
    assert dpn2.p[dpn2.last[0]] == 20.0 and dpn2.bed[0] == 60.0


def test_elem_declaration_snapshot():
    """The element declaration follows the node declaration's rules: an
    owned, read-only snapshot installed in its place at first use (writes
    into the reader's array change nothing, a write into the snapshot
    raises), reused by a rebuild and by a shallow copy, renewed for a new
    array and for a deep-copied or unpickled mesh."""
    vu = _vu()
    decl = np.array([2, 0, 1], dtype=np.int32)
    mesh = _elem_mesh(decl=None)
    mesh.metadata["uv_data_last_elem"] = decl
    dp = _etable(vu, mesh)
    snap = mesh.metadata["uv_data_last_elem"]
    assert snap is not decl and snap.base is None
    assert snap.dtype == decl.dtype and snap.tolist() == [2, 0, 1]
    assert not snap.flags.writeable and decl.flags.writeable
    assert vu._SNAPSHOTS.get(id(snap)) is snap
    assert mesh._uv_depth_points[("layer_linear", "element")].decl is snap
    assert "uv_data_last" not in mesh.metadata     # the node key untouched
    decl[:] = 0
    assert _etable(vu, mesh) is dp
    assert mesh.metadata["uv_data_last_elem"] is snap
    with pytest.raises(ValueError, match="read-only"):
        snap[0] = 0
    # a shallow copy sharing the metadata reuses the snapshot
    other = copy.copy(mesh)
    other._uv_depth_points = None
    assert _etable(vu, other).last.tolist() == [2, 0, 1]
    assert other.metadata["uv_data_last_elem"] is snap
    # a max_levels cut rebuilds from the same snapshot
    mesh.sigma = mesh.sigma[:3]
    dp2 = _etable(vu, mesh)
    assert dp2 is not dp and dp2.last.tolist() == [1, 0, 1]
    assert mesh.metadata["uv_data_last_elem"] is snap
    # a new array is a new declaration
    mesh.metadata["uv_data_last_elem"] = np.array([0, 0, 1], dtype=np.int32)
    assert _etable(vu, mesh).last.tolist() == [0, 0, 1]
    snap = mesh.metadata["uv_data_last_elem"]
    # a deep copy or an unpickled mesh gets a writeable array: revalidated
    for clone in (copy.deepcopy(mesh), pickle.loads(pickle.dumps(mesh))):
        d = clone.metadata["uv_data_last_elem"]
        assert d.flags.writeable
        d[0] = 1
        assert _etable(vu, clone).last.tolist() == [1, 0, 1]
        new = clone.metadata["uv_data_last_elem"]
        assert new is not d and not new.flags.writeable
    assert _etable(vu, mesh).last.tolist() == [0, 0, 1]
    assert mesh.metadata["uv_data_last_elem"] is snap


@pytest.mark.parametrize("comp", ["u", "v"])
def test_elem_record_check_names_element(comp):
    vu = _vu()
    mesh = _elem_mesh()                              # last = [2, 0, 1]
    u = np.ones((3, 3))
    v = np.ones((3, 3))
    (u if comp == "u" else v)[1, 2] = np.nan         # element 2, point 1
    vel = _efield(u, v)
    with pytest.raises(ValueError) as ei:
        vu.check_depth_linear_record(vel, mesh, record="t=4")
    msg = str(ei.value)
    assert "element 2, point 1 " in msg
    assert f" {comp} " in msg and "t=4" in msg
    assert "of this element" in msg
    assert "node" not in msg
    assert vel._dl_token is None
    assert not vu.is_checked(vel, mesh)
    # the column count and the rows, per element
    with pytest.raises(ValueError, match=r"has 4 elements, the mesh has 3 "
                                         r"\(element layout expected\)"):
        vu.check_depth_linear_record(_efield(np.ones((3, 4))), mesh)
    with pytest.raises(ValueError, match=r"has 2 rows"):
        vu.check_depth_linear_record(_efield(np.ones((2, 3))), mesh)


def test_elem_record_check_ignores_below_last():
    vu = _vu()
    mesh = _elem_mesh()                              # last = [2, 0, 1]
    u = np.ones((3, 3))
    v = np.ones((3, 3))
    u[1:, 1] = np.nan                                # below element 1's last
    v[2, 2] = np.inf                                 # below element 2's last
    u[2, 2] = 1.0e30
    vel = _efield(u, v, w=np.full((4, 5), np.nan))   # w is never checked
    assert not vu.is_checked(vel, mesh)
    vu.check_depth_linear_record(vel, mesh)
    assert vu.is_checked(vel, mesh)
    assert vel._dl_token is _etable(vu, mesh)
    assert vel._dl_token is not vu.depth_points(mesh, "layer_linear")
    # a lazy element field: checked through its parents, no blend
    lazy = LazyLerpField(_efield(np.ones((3, 3)), w=np.zeros((4, 5))), vel,
                         0.5)
    vu.check_depth_linear_record(lazy, mesh)
    assert lazy._dl_token is _etable(vu, mesh) and lazy._cache == {}
    # a deeper declaration makes a NaN usable: a new table, unchecked
    mesh.metadata["uv_data_last_elem"] = np.array([2, 1, 1], dtype=np.int32)
    assert not vu.is_checked(vel, mesh)
    with pytest.raises(ValueError, match=r"element 1, point 1 "):
        vu.check_depth_linear_record(vel, mesh)


def test_node_and_elem_tables_cached_separately():
    """One mesh holds a node and an element table side by side; each field
    is checked against its own layout's table, and each table is rebuilt
    only when its own inputs are replaced."""
    vu = _vu()
    mesh = _elem_mesh()
    dpn = vu.depth_points(mesh, "layer_linear")
    dpe = _etable(vu, mesh)
    assert dpn is not dpe
    assert dpn.last.size == mesh.n_nodes and dpe.last.size == mesh.n_elements
    assert set(mesh._uv_depth_points) == {("layer_linear", "node"),
                                          ("layer_linear", "element")}
    assert vu.depth_points(mesh, "layer_linear") is dpn
    assert _etable(vu, mesh) is dpe
    node_vel = _field(np.ones((3, 5)), placement="layer_linear")
    elem_vel = _efield(np.ones((3, 3)))
    vu.check_depth_linear_record(node_vel, mesh)
    vu.check_depth_linear_record(elem_vel, mesh)
    assert node_vel._dl_token is dpn and elem_vel._dl_token is dpe
    assert vu.is_checked(node_vel, mesh) and vu.is_checked(elem_vel, mesh)
    # a new element bed: the element table only
    mesh.metadata["elem_bed"] = np.array([60.0, 25.0, 45.0])
    assert vu.depth_points(mesh, "layer_linear") is dpn
    dpe2 = _etable(vu, mesh)
    assert dpe2 is not dpe and dpe2.bed.tolist() == [60.0, 25.0, 45.0]
    assert vu.is_checked(node_vel, mesh)
    assert not vu.is_checked(elem_vel, mesh)
    # a new mesh.depth or node declaration: the node table only
    vu.check_depth_linear_record(elem_vel, mesh)
    mesh.depth = mesh.depth.copy()
    assert _etable(vu, mesh) is dpe2
    dpn2 = vu.depth_points(mesh, "layer_linear")
    assert dpn2 is not dpn
    mesh.metadata["uv_data_last"] = np.full(5, 1, dtype=np.int32)
    assert _etable(vu, mesh) is dpe2
    assert vu.depth_points(mesh, "layer_linear").last.tolist() == [1] * 5
    assert vu.is_checked(elem_vel, mesh)
    # a cut: both
    mesh.sigma = mesh.sigma[:3]
    assert _etable(vu, mesh) is not dpe2
    assert not vu.is_checked(elem_vel, mesh)


def test_uv_wet():
    """``z <= bed[col]``: at the bed wet, below dry, a NaN depth or bed
    dry, an infinite bed wet down to ``z = inf``; jitted == ``py_func``."""
    vu = _vu()
    bed = np.array([30.0, np.inf, np.nan, 0.0])
    below = float(np.nextafter(30.0, np.inf))
    cases = [(0, 30.0, True), (0, 29.0, True), (0, -5.0, True),
             (0, below, False), (0, np.nan, False), (0, np.inf, False),
             (1, np.inf, True), (1, 1.0e9, True), (1, np.nan, False),
             (2, 0.0, False), (2, np.nan, False),
             (3, 0.0, True), (3, 1.0e-12, False)]
    fns = [vu.uv_wet]
    if hasattr(vu.uv_wet, "py_func"):
        fns.append(vu.uv_wet.py_func)
    for fn in fns:
        for col, z, want in cases:
            got = fn(bed, col, z)
            assert isinstance(got, (bool, np.bool_)), type(got)
            assert bool(got) is want, (fn, col, z)


# ---------------------------------------------------------------------------
# Sigma meshes: the table in the sigma fraction s = z / H
# ---------------------------------------------------------------------------

#: sigma interfaces (1 at the surface, 0 at the bed); layer centres
#: ``1 - (sigma_k + sigma_{k+1}) / 2``
SIGMA = [1.0, 0.75, 0.25, 0.0]
SIGMA_P = np.array([0.125, 0.5, 0.875])


def _sigma_mesh(levels=SIGMA, layout="node"):
    """The node mesh (4 nodes) or the element mesh (3 elements, neither an
    element declaration nor an element bed) on sigma interfaces."""
    if layout == "node":
        return _mesh(levels=levels, vertical_coord="sigma")
    mesh = _elem_mesh(levels=levels, decl=None, bed=None)
    mesh.vertical_coord = "sigma"
    return mesh


def _sigma_field(layout, u, v=None):
    """A ``"layer_linear"`` field of ``layout`` on a sigma mesh."""
    if layout == "node":
        return _field(u, v, placement="layer_linear")
    return _efield(u, v)


@pytest.mark.parametrize("layout", ["node", "element"])
def test_sigma_table_centres(layout):
    """The points are the layer centres in s, every column reads every
    layer (``last = K - 2``) and is wet at every s (``bed = +inf``); no
    declaration and no element bottom is read: a malformed declaration
    is neither validated nor replaced by a snapshot."""
    vu = _vu()
    mesh = _sigma_mesh(layout=layout)
    n_cols = mesh.n_nodes if layout == "node" else mesh.n_elements
    key = "uv_data_last" if layout == "node" else "uv_data_last_elem"
    bad = np.array([0.5, 7.0])                   # wrong dtype and shape
    mesh.metadata[key] = bad
    dp = vu.depth_points(mesh, "layer_linear", layout)
    assert_bitwise(dp.p, SIGMA_P, "p")
    assert_bitwise(dp.last, np.full(n_cols, 2, dtype=np.int32), "last")
    assert_bitwise(dp.bed, np.full(n_cols, np.inf), "bed")
    assert not (dp.p.flags.writeable or dp.last.flags.writeable
                or dp.bed.flags.writeable)
    assert mesh.metadata[key] is bad
    assert vu.depth_points(mesh, "layer_linear", layout) is dp


def test_sigma_ends_read_outer_centre():
    """At the surface (s = 0), at the bed (s = 1) and rounding beyond it a
    column reads its outermost centre directly and is wet there: never a
    dry fallback.  A NaN s, which the kernels map to ``+inf``, is wet and
    reads the bottom centre."""
    vu = _vu()
    dp = vu.depth_points(_sigma_mesh(), "layer_linear")
    for col in range(dp.last.size):
        last = int(dp.last[col])
        assert vu.uv_bracket(dp.p, last, 0.0) == (0, -1.0)
        assert vu.uv_bracket(dp.p, last, 1.0) == (2, -1.0)
        assert vu.uv_bracket(dp.p, last, 1.0 + 1e-15) == (2, -1.0)
        assert vu.uv_bracket(dp.p, last, np.inf) == (2, -1.0)
        # between the first two centres: (0.3125 - 0.125) / 0.375
        assert vu.uv_bracket(dp.p, last, 0.3125) == (0, 0.5)
        for s in (0.0, 1.0, 1.0 + 1e-15, np.inf):
            assert vu.uv_wet(dp.bed, col, s)


def test_sigma_table_follows_kept_levels():
    """``max_levels`` on a sigma mesh keeps the first interfaces from the
    surface (``tracking._apply_max_levels`` cuts ``mesh.sigma``): the table
    is rebuilt from the kept interfaces and every column reads down to
    the last kept centre, as the layer path's ``find_layer_sigma`` stays
    in the last kept layer below it; the record check wants the fields
    cut alike; no limit restores the full table.  A new ``mesh.depth``
    keeps the table (it has no bed rule)."""
    from kelp import tracking
    from kelp.core.config import ModeConfig
    from kelp.numerics.vertical_sigma import find_layer_sigma
    vu = _vu()
    mesh = _sigma_mesh(levels=[1.0, 0.75, 0.5, 0.25, 0.0])
    dp = vu.depth_points(mesh, "layer_linear")
    assert_bitwise(dp.p, np.array([0.125, 0.375, 0.625, 0.875]), "p")
    mesh.depth = mesh.depth + 5.0
    assert vu.depth_points(mesh, "layer_linear") is dp
    tracking._apply_max_levels(
        mesh, ModeConfig(Fdrift3D=True, Fdrift2D=False, max_levels=3))
    assert_bitwise(mesh.sigma, np.array([1.0, 0.75, 0.5]), "kept sigma")
    cut = vu.depth_points(mesh, "layer_linear")
    assert cut is not dp
    assert_bitwise(cut.p, np.array([0.125, 0.375]), "p cut")
    assert_bitwise(cut.last, np.full(4, 1, dtype=np.int32), "last cut")
    assert_bitwise(cut.bed, np.full(4, np.inf), "bed cut")
    h = 40.0
    for s in (0.375, 0.5, 0.75, 1.0):
        assert find_layer_sigma(mesh.sigma, s * h, h) == cut.last[0]
        assert vu.uv_bracket(cut.p, int(cut.last[0]), s) == (1, -1.0)
    with pytest.raises(ValueError, match=r"u has 3 rows, the 'layer_linear' "
                                         r"table of this mesh has 2 points"):
        vu.check_depth_linear_record(
            _field(np.ones((3, 4)), placement="layer_linear"), mesh)
    vu.check_depth_linear_record(
        _field(np.ones((2, 4)), placement="layer_linear"), mesh)
    tracking._apply_max_levels(mesh, ModeConfig(Fdrift3D=True,
                                                Fdrift2D=False))
    back = vu.depth_points(mesh, "layer_linear")
    assert back is not cut
    assert_bitwise(back.p, dp.p, "p restored")
    assert_bitwise(back.last, np.full(4, 3, dtype=np.int32), "last restored")


@pytest.mark.parametrize("comp", ["u", "v"])
@pytest.mark.parametrize("layout", ["node", "element"])
def test_sigma_record_check_all_layers(layout, comp):
    """Every layer of every column is a usable point: a non-finite u or v
    in any layer, the bottom one included (where a z-level column below
    its last point may hold anything), is refused; the message names the
    column, the point and its s."""
    vu = _vu()
    mesh = _sigma_mesh(layout=layout)
    n = mesh.n_nodes if layout == "node" else mesh.n_elements
    col = "node" if layout == "node" else "element"
    good = _sigma_field(layout, np.ones((3, n)), np.ones((3, n)))
    vu.check_depth_linear_record(good, mesh)
    assert vu.is_checked(good, mesh)
    for k, bad in ((2, np.nan), (0, np.inf), (1, -np.inf)):
        u = np.ones((3, n))
        v = np.ones((3, n))
        (u if comp == "u" else v)[k, n - 1] = bad
        fld = _sigma_field(layout, u, v)
        with pytest.raises(ValueError, match=(
                rf"{comp} = {bad!r} is non-finite at {col} {n - 1}, point "
                rf"{k} \(s = {SIGMA_P[k]:g}\).*finite at points 0\.\.2 of "
                rf"this {col} \(every layer of a sigma column; 1 {col}\(s\) "
                r"affected\)")):
            vu.check_depth_linear_record(fld, mesh)
        assert not vu.is_checked(fld, mesh)


def test_sigma_level_placement_refused():
    """Depth-linear u/v on a sigma mesh is layer-centred only: a
    ``"level"`` table, and so a ``"level"`` field's record check, is
    refused, naming the placement; element layout keeps its own
    ``"level"`` refusal.  Nothing is cached."""
    vu = _vu()
    mesh = _sigma_mesh()
    msg = (r"^depth-linear u/v on a sigma mesh \(vertical_coord 'sigma'\) "
           r"is 'layer_linear' only \(layer-centred data, read in the sigma "
           r"fraction s = z / H\), got 'level'$")
    with pytest.raises(ValueError, match=msg):
        vu.depth_points(mesh, "level")
    with pytest.raises(ValueError, match=msg):
        vu.check_depth_linear_record(_field(np.ones((4, 4))), mesh)
    assert mesh._uv_depth_points is None
    with pytest.raises(ValueError, match=r"on element layout is "
                                         r"'layer_linear' only"):
        vu.depth_points(_sigma_mesh(layout="element"), "level", "element")


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
            19.0, 20.0, 24.0, 25.0, 30.0, 31.0, 1e9, float("inf"),
            float("nan")]


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


def _scan_case():
    """A record-scan input over three column blocks (``_SCAN_BLOCK`` = 64):
    non-finite values inside some prefixes, at their last point, and below
    others' last points (ignored), every last from 0 to 3."""
    f = np.ones((4, 150))
    f[2, 65] = np.nan               # inside the prefix (last 3)
    f[1, 3] = np.inf                # below last[3] = 0: ignored
    f[3, 70] = -np.inf              # at the last point
    f[0, 140] = np.nan              # the surface point
    f[3, 141] = np.nan              # below last[141] = 2: ignored
    f[1, 149] = np.nan              # inside, last block
    last = np.full(150, 3, dtype=np.int32)
    last[3] = 0
    last[141] = 2
    last[100:110] = np.arange(10) % 4
    return f, last


#: the scan's answer for ``_scan_case``, by hand
_SCAN_WANT = {65: 2, 70: 3, 140: 0, 149: 1}


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
    f, last = _scan_case()
    out_j = np.empty(f.shape[1], dtype=np.int64)
    out_p = np.empty(f.shape[1], dtype=np.int64)
    vu._first_nonfinite_per_node(f, last, out_j)
    vu._first_nonfinite_per_node.py_func(f, last, out_p)
    assert_bitwise(out_j, out_p, "scan")
    want = np.full(f.shape[1], -1, dtype=np.int64)
    for i, k in _SCAN_WANT.items():
        want[i] = k
    assert_bitwise(out_j, want, "scan by hand")


_SUBPROCESS_SCRIPT = r"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).resolve().parents[0]))
sys.path.insert(0, sys.argv[1])
import numpy as np
from numba.core import config
assert config.DISABLE_JIT, "NUMBA_DISABLE_JIT not honoured"
import kelp.numerics.vertical_uv as vu
assert str(Path(vu.__file__).resolve()).startswith(
    str(Path(sys.argv[1]).resolve())), vu.__file__
import tests.test_vertical_uv as t
rows = []
for p, last_i, z in t._bracket_cases():
    pa = np.asarray(p, dtype=np.float64)
    k, a = vu.uv_bracket(pa, last_i, z)
    f = t._f_for(p)
    vals = [float(vu.uv_at(f, n, k, a)).hex() for n in range(3)]
    rows.append([int(k), float(a).hex(), vals])
f, last = t._scan_case()
out = np.empty(f.shape[1], dtype=np.int64)
vu._first_nonfinite_per_node(f, last, out)
print(json.dumps({"bracket": rows, "scan": out.tolist()}))
"""


def test_numba_disabled_subprocess_matches(tmp_path):
    """Under ``NUMBA_DISABLE_JIT=1`` the bracket, the read and the record
    scan (``_first_nonfinite_per_node``) give the compiled answers."""
    vu = _vu()
    cases = _bracket_cases()
    script = tmp_path / "vertical_uv_nojit.py"
    script.write_text(_SUBPROCESS_SCRIPT)
    src = str(Path(__file__).resolve().parents[1] / "src")
    env = dict(os.environ, NUMBA_DISABLE_JIT="1")
    out = subprocess.run(
        [sys.executable, str(script), src],
        capture_output=True, text=True, timeout=300, env=env)
    assert out.returncode == 0, out.stderr
    res = json.loads(out.stdout)
    rows = res["bracket"]
    assert len(rows) == len(cases)
    for (p, last_i, z), (k, a_hex, vals) in zip(cases, rows):
        pa = np.asarray(p, dtype=np.float64)
        kj, aj = vu.uv_bracket(pa, last_i, z)
        assert (int(kj), float(aj).hex()) == (k, a_hex), (p, last_i, z)
        f = _f_for(p)
        assert [float(vu.uv_at(f, n, kj, aj)).hex() for n in range(3)] \
            == vals, (p, last_i, z)
    f, last = _scan_case()
    out_j = np.empty(f.shape[1], dtype=np.int64)
    vu._first_nonfinite_per_node(f, last, out_j)
    assert_bitwise(np.array(res["scan"], dtype=np.int64), out_j, "scan")
