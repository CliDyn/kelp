"""Precomputed node Cartesian coordinates for the nearest-node search.

Pins (docs/localization.md §3, "Node coordinates come from a table"):
the table equals the OLD in-kernel formula bit for bit; every nearest kernel
returns the same anchor as a frozen copy of its pre-change self, ties
included (true two-node ties: element edge midpoints, plus a duplicated
node); the table follows every hash rebuild; the build-time self-check
detects an ulp-level table error; the wiring through the integrators is
complete (an AST check).
"""
import ast
import pathlib

import numpy as np
import pytest
from numba import njit

import kelp.core.mesh as mesh_mod
from kelp.core.geometry import DEG2RAD
from kelp.core.mesh import (Mesh, build_node_to_elems, build_search,
                            find_nearest_nodes, set_exact_nearest,
                            set_grid_hash_cell, _node_xyz_fill,
                            _node_xyz_selfcheck, _require_node_xyz)


# --------------------------------------------------------------------------
# synthetic meshes
# --------------------------------------------------------------------------

def _grid_mesh(xs, ys, dup_first=False, cell=None, jitter=0.0, seed=3):
    """Regular triangulated grid from 1-D lon/lat axes (degrees).

    ``dup_first`` appends an exact duplicate of node 0 (no element touches
    it): an exact distance tie by construction, so first-found-wins order
    is exercised.  ``cell`` pins an explicit grid-hash cell.  ``jitter``
    perturbs every node by a uniform offset of that size (degrees), so the
    coordinates are not "nice" multiples that every trig implementation
    rounds alike.
    """
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel().astype(np.float64)
    lat = gy.ravel().astype(np.float64)
    if jitter:
        rng = np.random.default_rng(seed)
        lon = lon + rng.uniform(-jitter, jitter, lon.size)
        lat = lat + rng.uniform(-jitter, jitter, lat.size)
    if dup_first:
        lon = np.append(lon, lon[0])
        lat = np.append(lat, lat[0])
    nx, ny = len(xs), len(ys)
    elems = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            n1, n2 = n0 + 1, n0 + nx
            elems.append([n0, n1, n2])
            elems.append([n1, n2 + 1, n2])
    elem_nodes = np.array(elems, dtype=np.int32)
    n2e, n2e_count = build_node_to_elems(lon.size, elem_nodes)
    mesh = Mesh(name="grid", nodes_lon=lon, nodes_lat=lat,
                depth=np.zeros(lon.size), elem_nodes=elem_nodes,
                elem_center_lon=np.array([lon[e].mean() for e in elem_nodes]),
                elem_center_lat=np.array([lat[e].mean() for e in elem_nodes]),
                node_to_elems=n2e, node_to_elems_count=n2e_count)
    build_search(mesh)
    if cell is not None:
        set_grid_hash_cell(mesh, cell)
    return mesh


def _multires_mesh(cell=None):
    """A 2-degree coarse grid (the elements) plus a 0.1-degree node patch
    inside one coarse cell — the multi-resolution case where the widening
    search and the far-rows budget matter."""
    xs = np.arange(-20.0, 20.01, 2.0)
    ys = np.arange(-10.0, 10.01, 2.0)
    base = _grid_mesh(xs, ys)
    fx, fy = np.meshgrid(np.arange(1.0, 3.01, 0.1), np.arange(1.0, 3.01, 0.1))
    lon = np.concatenate([base.nodes_lon, fx.ravel()])
    lat = np.concatenate([base.nodes_lat, fy.ravel()])
    n2e, n2e_count = build_node_to_elems(lon.size, base.elem_nodes)
    mesh = Mesh(name="multires", nodes_lon=lon, nodes_lat=lat,
                depth=np.zeros(lon.size), elem_nodes=base.elem_nodes,
                elem_center_lon=base.elem_center_lon,
                elem_center_lat=base.elem_center_lat,
                node_to_elems=n2e, node_to_elems_count=n2e_count)
    build_search(mesh)
    if cell is not None:
        set_grid_hash_cell(mesh, cell)
    return mesh


def _geometries():
    """(name, mesh) pairs covering every regime the spec names (§5.2), plus
    the ones the review asked for: jittered coordinates (discriminating trig),
    a polar hole above the last node row and a fine cell on the wrap grid
    (empty 3x3 windows -> the widening / seam / full-row branches run)."""
    return [
        ("uniform", _grid_mesh(np.arange(-20.0, 20.01, 0.5),
                               np.arange(-10.0, 10.01, 0.5))),
        ("jittered", _grid_mesh(np.arange(-20.0, 20.01, 0.5),
                                np.arange(-10.0, 10.01, 0.5), jitter=1e-3)),
        # span 358.4 deg in both branches -> cyclic grid; 360 / cell is not
        # an integer -> the seam cell is narrower than the others
        ("wrap_seam", _grid_mesh(np.arange(0.0, 358.5, 1.4),
                                 np.arange(-20.0, 20.01, 1.4))),
        # the same wrap grid with a cell far below the spacing: most 3x3
        # windows are EMPTY -> ring widening, seam-crossing row slices
        ("wrap_seam_fine_cell", _grid_mesh(np.arange(0.0, 358.5, 1.4),
                                           np.arange(-20.0, 20.01, 1.4),
                                           cell=0.3)),
        # span 349 deg -> NOT cyclic, one-cell padding, +-360 images matter
        ("near_global_nonwrap", _grid_mesh(np.arange(0.0, 349.01, 1.0),
                                           np.arange(-10.0, 10.01, 1.0))),
        # rows compress toward the pole; caps reach the pole
        ("polar", _grid_mesh(np.arange(-180.0, 179.9, 3.0),
                             np.arange(60.0, 89.51, 0.5))),
        # nodes stop at 85 N on a 0.5-deg cell: queries 85-90 N have empty
        # windows -> the polar full-row branch of the exact search
        ("polar_hole", _grid_mesh(np.arange(-180.0, 179.9, 3.0),
                                  np.arange(70.0, 85.01, 0.5), cell=0.5)),
        ("multires_auto", _multires_mesh()),
        ("multires_fine_cell", _multires_mesh(cell=0.2)),
        ("ties", _grid_mesh(np.arange(-5.0, 5.01, 0.5),
                            np.arange(-5.0, 5.01, 0.5), dup_first=True)),
        ("small_cell", _grid_mesh(np.arange(-20.0, 20.01, 0.5),
                                  np.arange(-10.0, 10.01, 0.5), cell=0.15)),
    ]


def _edge_midpoints(mesh, cap=20000, seed=11):
    """True two-node ties: the midpoint of every element edge (which of the
    two endpoints wins is decided by the last bit of the distances), capped
    to ``cap`` edges per mesh (random subset, seeded)."""
    v = mesh.elem_nodes[:, :3]
    a = np.concatenate([v[:, 0], v[:, 1], v[:, 2]])
    b = np.concatenate([v[:, 1], v[:, 2], v[:, 0]])
    if a.size > cap:
        pick = np.random.default_rng(seed).choice(a.size, cap, replace=False)
        a, b = a[pick], b[pick]
    lon = 0.5 * (mesh.nodes_lon[a] + mesh.nodes_lon[b])
    lat = 0.5 * (mesh.nodes_lat[a] + mesh.nodes_lat[b])
    return lon, lat


def _queries(mesh, n=4000, seed=0):
    """Random queries over the mesh box widened by 30 degrees (far-outside
    queries included), the edge midpoints (true ties), exact node positions
    and the pole/dateline corners."""
    rng = np.random.default_rng(seed)
    lo0, lo1 = mesh.nodes_lon.min() - 30.0, mesh.nodes_lon.max() + 30.0
    la0 = max(mesh.nodes_lat.min() - 30.0, -90.0)
    la1 = min(mesh.nodes_lat.max() + 30.0, 90.0)
    lon = rng.uniform(lo0, lo1, n)
    lat = rng.uniform(la0, la1, n)
    mlon, mlat = _edge_midpoints(mesh)
    lon = np.concatenate([lon, mlon, mesh.nodes_lon[:50],
                          [-180.0, 180.0, 0.0, 359.99, 179.999, -179.999, 0.0, 0.0]])
    lat = np.concatenate([lat, mlat, mesh.nodes_lat[:50],
                          [0.0, 0.0, 89.9, -89.9, 45.0, -45.0, 90.0, -90.0]])
    lon = ((lon + 180.0) % 360.0) - 180.0
    return lon.astype(np.float64), lat.astype(np.float64)


_GEOMETRIES = _geometries()          # built once per test session
_IDS = lambda v: v if isinstance(v, str) else ""     # noqa: E731


# --------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------

@njit(cache=False)
def _ref_xyz(nodes_lon, nodes_lat):
    """The OLD per-candidate expressions of the scans, verbatim, in a plain
    loop (the same code shape as the fill)."""
    out = np.empty((nodes_lon.shape[0], 3))
    for n in range(nodes_lon.shape[0]):
        lat_n_r = nodes_lat[n] * DEG2RAD
        lon_n_r = nodes_lon[n] * DEG2RAD
        cos_lat_n = np.cos(lat_n_r)
        out[n, 0] = cos_lat_n * np.cos(lon_n_r)
        out[n, 1] = cos_lat_n * np.sin(lon_n_r)
        out[n, 2] = np.sin(lat_n_r)
    return out


class TestNodeXyzTable:

    @pytest.mark.parametrize("name,mesh", _GEOMETRIES, ids=_IDS)
    def test_table_equals_old_kernel_formula_bitwise(self, name, mesh):
        assert mesh._node_xyz is not None
        assert mesh._node_xyz.shape == (mesh.n_nodes, 3)
        assert mesh._node_xyz.dtype == np.float64
        assert mesh._node_xyz.flags["C_CONTIGUOUS"]
        assert np.array_equal(mesh._node_xyz,
                              _ref_xyz(mesh.nodes_lon, mesh.nodes_lat))

    @pytest.mark.parametrize("name,mesh", _GEOMETRIES, ids=_IDS)
    def test_table_equals_the_scans_own_code_shape(self, name, mesh):
        """The build-time self-check on EVERY node: the gather/argmin-shaped
        loop of the old scans evaluates the same floats the table holds."""
        ids = np.arange(mesh.n_nodes, dtype=np.int32)
        mism, best = _node_xyz_selfcheck(ids, mesh.nodes_lon, mesh.nodes_lat,
                                         mesh._node_xyz, float(mesh.nodes_lon[0]),
                                         float(mesh.nodes_lat[0]))
        assert mism == 0
        assert best == 0                     # the query IS node 0

    def test_fill_matches_the_reference_on_random_coordinates(self):
        """The implementation itself, on 500k random (not 'nice') points —
        the table-formula pin above only discriminates on jittered meshes."""
        rng = np.random.default_rng(1)
        lat = rng.uniform(-90.0, 90.0, 500_000)
        lon = rng.uniform(-180.0, 180.0, 500_000)
        ref = _ref_xyz(lon, lat)
        out = np.empty((lon.size, 3))
        _node_xyz_fill(lon, lat, out)
        assert np.array_equal(out, ref)
        # documented observation: numpy's trig is NOT a drop-in (an ulp in a
        # small fraction of values on most builds); skipped where it agrees
        via_numpy = np.column_stack((np.cos(lat * DEG2RAD) * np.cos(lon * DEG2RAD),
                                     np.cos(lat * DEG2RAD) * np.sin(lon * DEG2RAD),
                                     np.sin(lat * DEG2RAD)))
        if np.array_equal(ref, via_numpy):
            pytest.skip("numpy and numba trig agree bitwise on this machine")
        assert np.abs(ref - via_numpy).max() < 1e-15

    def test_selfcheck_detects_an_ulp_perturbation(self):
        mesh = _GEOMETRIES[1][1]                    # jittered
        ids = np.arange(mesh.n_nodes, dtype=np.int32)
        bad = mesh._node_xyz.copy()
        bad[17, 1] = np.nextafter(bad[17, 1], 2.0)
        mism, _ = _node_xyz_selfcheck(ids, mesh.nodes_lon, mesh.nodes_lat, bad,
                                      float(mesh.nodes_lon[0]), float(mesh.nodes_lat[0]))
        assert mism == 1

    def test_build_warns_when_the_fill_and_the_scans_disagree(self, monkeypatch):
        """Simulates a build whose fill loop was lowered to a different math
        library (the Intel SVML case): numpy's trig differs from the scans'
        on jittered coordinates, so the build must warn."""
        def numpy_fill(lon, lat, out):
            out[:, 0] = np.cos(lat * DEG2RAD) * np.cos(lon * DEG2RAD)
            out[:, 1] = np.cos(lat * DEG2RAD) * np.sin(lon * DEG2RAD)
            out[:, 2] = np.sin(lat * DEG2RAD)
        monkeypatch.setattr(mesh_mod, "_node_xyz_fill", numpy_fill)
        monkeypatch.setattr(mesh_mod, "_NODE_XYZ_SELFCHECK_SAMPLE", 10 ** 9)   # every node
        mesh = _grid_mesh(np.arange(-20.0, 20.01, 0.5), np.arange(-10.0, 10.01, 0.5),
                          jitter=1e-3, seed=5)
        ref = _ref_xyz(mesh.nodes_lon, mesh.nodes_lat)
        if np.array_equal(mesh._node_xyz, ref):
            pytest.skip("numpy and numba trig agree bitwise on this machine")
        with pytest.warns(RuntimeWarning, match="node coordinate table differs"):
            build_search(mesh)

    def test_table_uses_raw_longitudes_not_the_folded_branch(self):
        # a dateline mesh is indexed in the [0, 360) branch: cos(lon + 360)
        # is NOT cos(lon) in floating point, so the table must come from the
        # raw node longitudes the kernels read
        mesh = _grid_mesh(np.arange(170.0, 190.01, 0.5),
                          np.arange(-5.0, 5.01, 0.5))
        mesh.nodes_lon[:] = ((mesh.nodes_lon + 180.0) % 360.0) - 180.0   # raw signed lons
        build_search(mesh)
        assert np.array_equal(mesh._node_xyz,
                              _ref_xyz(mesh.nodes_lon, mesh.nodes_lat))

    def test_table_follows_every_rebuild(self):
        mesh = _grid_mesh(np.arange(-5.0, 5.01, 0.5), np.arange(-5.0, 5.01, 0.5))
        before = mesh._node_xyz
        set_grid_hash_cell(mesh, 0.3)
        assert mesh._node_xyz is not before             # rebuilt, not stale
        assert np.array_equal(mesh._node_xyz,
                              _ref_xyz(mesh.nodes_lon, mesh.nodes_lat))
        # a reader that rotates the nodes as points and then rebuilds the
        # search (healpix) must get the rotated table
        mesh.nodes_lon = mesh.nodes_lon + 37.5
        mesh.nodes_lat = mesh.nodes_lat * 0.5 + 10.0
        build_search(mesh)
        assert np.array_equal(mesh._node_xyz,
                              _ref_xyz(mesh.nodes_lon, mesh.nodes_lat))

    def test_every_entry_point_requires_the_table(self):
        from kelp.core.mesh import find_containing_elements_walk
        mesh = _grid_mesh(np.arange(-5.0, 5.01, 0.5), np.arange(-5.0, 5.01, 0.5))
        mesh._node_xyz = None
        with pytest.raises(RuntimeError, match="build_search"):
            _require_node_xyz(mesh)
        with pytest.raises(RuntimeError, match="build_search"):
            find_nearest_nodes(mesh, np.array([0.1]), np.array([0.2]))
        with pytest.raises(RuntimeError, match="build_search"):
            find_containing_elements_walk(mesh, np.array([0.1]), np.array([0.2]),
                                          np.array([0], dtype=np.int32))
        # the integrators' launch sites all go through _require_node_xyz
        # (pinned by the AST test below)


# --------------------------------------------------------------------------
# every kernel returns the OLD anchor, ties included
# --------------------------------------------------------------------------

from kelp.core.mesh import (_grid_hash_nearest_3x3, _grid_hash_nearest_widened,   # noqa: E402
                            _grid_hash_nearest_exact, _grid_hash_nearest_batch)
from tests._ref_nearest_kernels import (_ref_grid_hash_nearest_3x3,             # noqa: E402
                                        _ref_grid_hash_nearest_widened,
                                        _ref_grid_hash_nearest_exact)


def _old_args(m):
    return (m._gh_offsets, m._gh_node_ids, m._gh_cell_size, m._gh_lon_min,
            m._gh_lat_min, m._gh_nx, m._gh_ny, m._gh_wrap_lon,
            m.nodes_lon, m.nodes_lat)


def _new_args(m):
    return _old_args(m) + (m._node_xyz,)


class TestKernelParity:

    @pytest.mark.parametrize("name,mesh", _GEOMETRIES, ids=_IDS)
    def test_single_particle_kernels(self, name, mesh):
        lon, lat = _queries(mesh)
        far = int(mesh._gh_far_rows)
        old, new = _old_args(mesh), _new_args(mesh)
        for lo, la in zip(lon, lat):
            lo, la = float(lo), float(la)
            assert (_grid_hash_nearest_3x3(lo, la, *new)
                    == _ref_grid_hash_nearest_3x3(lo, la, *old))
            assert (_grid_hash_nearest_widened(lo, la, *new)
                    == _ref_grid_hash_nearest_widened(lo, la, *old))
            assert (_grid_hash_nearest_exact(lo, la, *new, far)
                    == _ref_grid_hash_nearest_exact(lo, la, *old, far))

    @pytest.mark.parametrize("name,mesh", _GEOMETRIES, ids=_IDS)
    @pytest.mark.parametrize("exact", [False, True])
    def test_batch_path(self, name, mesh, exact):
        lon, lat = _queries(mesh, seed=7)
        far = int(mesh._gh_far_rows)
        old, new = _old_args(mesh), _new_args(mesh)
        ref = np.array([(_ref_grid_hash_nearest_exact(float(lo), float(la), *old, far)
                         if exact else
                         _ref_grid_hash_nearest_3x3(float(lo), float(la), *old))
                        for lo, la in zip(lon, lat)], dtype=np.int32)
        out = np.empty(lon.size, dtype=np.int32)
        _grid_hash_nearest_batch(lon.size, lon, lat, *new, out, exact, far)
        assert np.array_equal(out, ref)
        # and the public entry point under the guard flag (restored after:
        # the meshes are shared by every test of this module)
        set_exact_nearest(mesh, exact)
        try:
            got = find_nearest_nodes(mesh, lon, lat)
        finally:
            set_exact_nearest(mesh, False)
        if exact:
            assert np.array_equal(got, ref)
        else:
            miss = ref < 0                                 # KD heal only where the 3x3 missed
            assert np.array_equal(got[~miss], ref[~miss])
            assert (got[miss] >= 0).all()

    def test_exact_tie_goes_to_the_first_found_node(self):
        mesh = dict(_GEOMETRIES)["ties"]
        dup = mesh.n_nodes - 1                             # exact duplicate of node 0
        lo, la = float(mesh.nodes_lon[0]), float(mesh.nodes_lat[0])
        assert mesh._node_xyz[0].tolist() == mesh._node_xyz[dup].tolist()
        new = _grid_hash_nearest_3x3(lo, la, *_new_args(mesh))
        old = _ref_grid_hash_nearest_3x3(lo, la, *_old_args(mesh))
        assert new == old and new in (0, dup)
        for d in (1e-7, 1e-4, 0.01):
            assert (_grid_hash_nearest_exact(lo + d, la - d, *_new_args(mesh), int(mesh._gh_far_rows))
                    == _ref_grid_hash_nearest_exact(lo + d, la - d, *_old_args(mesh), int(mesh._gh_far_rows)))

    def test_midpoint_ties_would_expose_an_ulp_wrong_table(self):
        """The detector's teeth: with a table built by numpy's trig instead
        of the scans' expressions, some edge-midpoint ties (exact two-node
        ties on the regular grids) flip — the parity tests would see an
        ulp-wrong table (skipped on a build where numpy agrees bitwise)."""
        geoms = dict(_GEOMETRIES)
        flips = 0
        differing_tables = 0
        for name in ("wrap_seam", "near_global_nonwrap", "uniform"):
            mesh = geoms[name]
            bad = np.column_stack((np.cos(mesh.nodes_lat * DEG2RAD) * np.cos(mesh.nodes_lon * DEG2RAD),
                                   np.cos(mesh.nodes_lat * DEG2RAD) * np.sin(mesh.nodes_lon * DEG2RAD),
                                   np.sin(mesh.nodes_lat * DEG2RAD)))
            if np.array_equal(bad, mesh._node_xyz):
                continue
            differing_tables += 1
            lon, lat = _edge_midpoints(mesh, cap=60000)
            args_bad = _old_args(mesh) + (bad,)
            old = _old_args(mesh)
            flips += sum(_grid_hash_nearest_3x3(float(lo), float(la), *args_bad)
                         != _ref_grid_hash_nearest_3x3(float(lo), float(la), *old)
                         for lo, la in zip(lon, lat))
        if differing_tables == 0:
            pytest.skip("numpy and numba trig agree bitwise on this machine")
        assert flips > 0


# --------------------------------------------------------------------------
# the wiring is complete (source-level, so a forgotten site fails a test
# instead of a Numba typing error deep inside a run)
# --------------------------------------------------------------------------

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "kelp"
_KERNEL_CALLS = {
    "_grid_hash_nearest_3x3", "_grid_hash_nearest_widened",
    "_grid_hash_nearest_exact", "_grid_hash_nearest_batch",
    "_gh_scan_cell", "_gh_scan_row_range", "_gh_scan_row_window",
    "_loop_walk_containing", "_fused_site_anchor", "_rk45_eval",
    "_rk45_kernel_2d",
    "fused_kernel",                 # the generic launch of the 2-D runners
}
_DEF_MARKERS = {"gh_node_ids", "gh_far_rows", "far_rows"}


def _is_launch(node):
    fn = node.func
    name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
    if name is None:
        return False
    # the fused STEP kernels (_fused_<int>_step_<dim>[_variant]) plus the
    # named search kernels; `_fused_3d_bary_field_checks` and
    # `_fused_3d_optional_payload_checks` are Python helpers, not kernels
    if name in _KERNEL_CALLS or (name.startswith("_fused_") and "_step_" in name):
        return True
    # any launch that splats the shared hash-argument tuple is a kernel launch
    return any(isinstance(a, ast.Starred) and isinstance(a.value, ast.Call)
               and getattr(a.value.func, "id", None) == "_mesh_gh_args"
               for a in node.args)


def _carries_table(arg):
    if isinstance(arg, ast.Name) and arg.id == "node_xyz":
        return True
    if isinstance(arg, ast.Attribute) and arg.attr == "_node_xyz":
        return True
    return (isinstance(arg, ast.Call)
            and getattr(arg.func, "id", None) == "_require_node_xyz")


def _wiring_gaps(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    gaps = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            names = [a.arg for a in node.args.args]
            if _DEF_MARKERS & set(names) and "node_xyz" not in names:
                gaps.append(f"{path.name}:{node.lineno} def {node.name}(…) lacks node_xyz")
        elif isinstance(node, ast.Call) and _is_launch(node):
            if not any(_carries_table(a) for a in node.args):
                name = getattr(node.func, "id", None) or getattr(node.func, "attr", "?")
                gaps.append(f"{path.name}:{node.lineno} call {name}(…) lacks node_xyz")
    return gaps


@pytest.mark.parametrize("module", ["core/mesh.py", "numerics/integrators.py"])
def test_every_nearest_kernel_and_launch_passes_node_xyz(module):
    gaps = _wiring_gaps(_SRC / module)
    assert not gaps, "\n".join(gaps)


def test_the_wiring_test_sees_the_generic_launches():
    """The four 2-D runners launch `fused_kernel(...)` through a variable
    with the splatted `_mesh_gh_args(mesh)`: the scanner must count them."""
    src = (_SRC / "numerics/integrators.py").read_text()
    tree = ast.parse(src)
    generic = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
               and getattr(n.func, "id", None) == "fused_kernel"]
    assert len(generic) >= 4
    assert all(_is_launch(n) for n in generic)
