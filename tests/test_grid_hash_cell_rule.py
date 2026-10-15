"""Automatic grid-hash cell by window occupancy (mesh.grid_hash_cell_rule).

Design and calibration: docs/localization.md §3, benchmarks/grid_hash_cell_rule_table.md.
"""
import pathlib

import numpy as np
import pytest

from kelp.core.mesh import (Mesh, build_node_to_elems, build_search, set_grid_hash_cell,
                            set_exact_nearest, find_nearest_nodes, _gh_far_rows_for)
from tests.test_node_xyz import _grid_mesh, _multires_mesh


# --------------------------------------------------------------------------
# Task 1: the build's grid arithmetic, factored out
# --------------------------------------------------------------------------

def _geoms():
    return [
        ("uniform", _grid_mesh(np.arange(-20.0, 20.01, 0.5), np.arange(-10.0, 10.01, 0.5))),
        ("wrap_seam", _grid_mesh(np.arange(0.0, 358.5, 1.4), np.arange(-20.0, 20.01, 1.4))),
        ("near_global_nonwrap", _grid_mesh(np.arange(0.0, 349.01, 1.0), np.arange(-10.0, 10.01, 1.0))),
        ("polar", _grid_mesh(np.arange(-180.0, 179.9, 3.0), np.arange(60.0, 89.51, 0.5))),
        ("dateline", _grid_mesh(np.arange(170.0, 190.01, 0.5), np.arange(-5.0, 5.01, 0.5))),
        ("multires", _multires_mesh()),
    ]


@pytest.mark.parametrize("name", [n for n, _ in _geoms()])
@pytest.mark.parametrize("cell", [None, 0.7, 0.3, 0.123])
def test_grid_helpers_reproduce_the_build(name, cell):
    from kelp.core.mesh import _gh_lon_branch, _gh_grid_geometry, _gh_cell_index
    mesh = dict(_geoms())[name]              # fresh: `None` really tests the auto cell
    if cell is not None:
        set_grid_hash_cell(mesh, cell)
    lon, wrap = _gh_lon_branch(mesh.nodes_lon)
    assert wrap == mesh._gh_wrap_lon
    lon_min, lat_min, nx, ny = _gh_grid_geometry(lon, mesh.nodes_lat, mesh._gh_cell_size, wrap)
    assert (lon_min, lat_min, nx, ny) == (mesh._gh_lon_min, mesh._gh_lat_min, mesh._gh_nx, mesh._gh_ny)
    idx = _gh_cell_index(lon, mesh.nodes_lat, mesh._gh_cell_size, lon_min, lat_min, nx, ny, wrap)
    # the build's CSR: node ids grouped by cell, ascending inside a cell
    counts = np.bincount(idx, minlength=nx * ny)
    assert np.array_equal(np.diff(mesh._gh_offsets), counts)
    order = np.argsort(idx, kind="stable")
    assert np.array_equal(mesh._gh_node_ids, order.astype(np.int32))


# --------------------------------------------------------------------------
# Task 2: the chooser
# --------------------------------------------------------------------------

from kelp.core.mesh import (choose_grid_hash_cell, apply_grid_hash_cell_rule,   # noqa: E402
                            describe_grid_hash, _gh_cell_of)


def _estuary_mesh():
    """FESOM-C-like: a 0.5 x 0.01 deg coastal strip at 0.0005 deg plus a
    1 x 0.5 deg offshore block at 0.05 deg (ratio 100), two disjoint
    triangulations (the transition is not meshed; irrelevant here)."""
    def tri(xs, ys):
        gx, gy = np.meshgrid(xs, ys); lon = gx.ravel(); lat = gy.ravel()
        nx, ny = len(xs), len(ys); el = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                n0 = j * nx + i
                el.append([n0, n0 + 1, n0 + nx]); el.append([n0 + 1, n0 + nx + 1, n0 + nx])
        return lon, lat, np.array(el, dtype=np.int32)
    lon1, lat1, el1 = tri(np.arange(0.0, 0.5001, 0.0005), np.arange(0.0, 0.01001, 0.0005))
    lon2, lat2, el2 = tri(np.arange(0.0, 1.0001, 0.05), np.arange(0.02, 0.5001, 0.05))
    lon = np.concatenate([lon1, lon2]); lat = np.concatenate([lat1, lat2])
    elem = np.concatenate([el1, el2 + lon1.size])
    n2e, n2c = build_node_to_elems(lon.size, elem)
    m = Mesh(name="estuary", nodes_lon=lon, nodes_lat=lat, depth=np.zeros(lon.size),
             elem_nodes=elem, elem_center_lon=lon[elem].mean(1), elem_center_lat=lat[elem].mean(1),
             node_to_elems=n2e, node_to_elems_count=n2c)
    with pytest.warns(RuntimeWarning, match="highly non-uniform"):
        build_search(m)                       # the legacy warning still fires
    m.metadata["n_strip_elems"] = len(el1)
    return m


def _window_occupancy(mesh, lon, lat):
    counts = np.diff(mesh._gh_offsets)
    out = np.empty(lon.size)
    for i in range(lon.size):
        ci, cj = _gh_cell_of(float(lon[i]), float(lat[i]), mesh._gh_cell_size, mesh._gh_lon_min,
                             mesh._gh_lat_min, mesh._gh_nx, mesh._gh_ny, mesh._gh_wrap_lon)
        t = 0
        for dj in (-1, 0, 1):
            cj2 = cj + dj
            if cj2 < 0 or cj2 >= mesh._gh_ny:
                continue
            for di in (-1, 0, 1):
                ci2 = ci + di
                if mesh._gh_wrap_lon:
                    ci2 %= mesh._gh_nx
                elif ci2 < 0 or ci2 >= mesh._gh_nx:
                    continue
                t += counts[cj2 * mesh._gh_nx + ci2]
        out[i] = t
    return out


def _approx_share(mesh, lon, lat):
    """Share of queries whose plain 3x3 scan (guard off) misses the true nearest."""
    from kelp.core.mesh import _grid_hash_nearest_3x3
    set_exact_nearest(mesh, True)
    exact = find_nearest_nodes(mesh, lon, lat)
    args = (mesh._gh_offsets, mesh._gh_node_ids, mesh._gh_cell_size, mesh._gh_lon_min,
            mesh._gh_lat_min, mesh._gh_nx, mesh._gh_ny, mesh._gh_wrap_lon,
            mesh.nodes_lon, mesh.nodes_lat, mesh._node_xyz)
    plain = np.array([_grid_hash_nearest_3x3(float(lo), float(la), *args) for lo, la in zip(lon, lat)])
    return float(np.mean(plain != exact))


class TestChooser:

    def test_uniform_mesh_takes_a_smaller_cell_with_a_sane_window(self):
        m = _grid_mesh(np.arange(-20.0, 20.01, 0.5), np.arange(-10.0, 10.01, 0.5))
        legacy = m._gh_cell_size
        # a quasi-uniform mesh predicts less than the 2x default gain threshold:
        # the reader's cell is kept (calibration 2026-09-22: the search gains
        # ~1.5x there and the step phase nothing measurable)
        keep, rep_keep = choose_grid_hash_cell(m, exact=True)
        assert keep == legacy and "keep the legacy cell" in rep_keep["reason"]
        # the argmin mechanics, with the threshold disabled
        cell, rep = choose_grid_hash_cell(m, exact=True, min_gain=1.0)
        assert cell < legacy and rep["mode"] == "exact" and rep["legacy"] == legacy
        set_grid_hash_cell(m, cell)
        occ = _window_occupancy(m, m.elem_center_lon, m.elem_center_lat)
        assert np.median(occ) >= 1 and np.percentile(occ, 95) < 60
        # the chosen row is the cost minimum of the ladder
        rows = rep["candidates"]
        assert min(rows, key=lambda r: r["mean_cost"])["cell"] == cell

    def test_strict_mode_never_leaves_an_empty_window(self):
        for name, m in _geoms():
            cell, rep = choose_grid_hash_cell(m, exact=False)
            assert rep["mode"] == "approx"
            set_grid_hash_cell(m, cell)
            occ = _window_occupancy(m, m.elem_center_lon, m.elem_center_lat)
            assert (occ > 0).all(), name
            assert np.median(occ) >= 8, name

    def test_polar_mesh_strict_mode_adds_no_approximate_anchor(self):
        m = _grid_mesh(np.arange(-180.0, 179.9, 3.0), np.arange(60.0, 89.51, 0.5))
        lon, lat = m.elem_center_lon, m.elem_center_lat
        base = _approx_share(m, lon, lat)
        cell, rep = choose_grid_hash_cell(m, exact=False)
        rows = {r["cell"]: r for r in rep["candidates"]}
        assert rows[cell]["approx"] <= rows[rep["legacy"]]["approx"] + 0.005
        set_grid_hash_cell(m, cell)
        # the independent measure (plain 3x3 vs exact) agrees
        assert _approx_share(m, lon, lat) <= base + 0.005

    def test_multires_respects_the_far_rows_cap(self):
        m = _multires_mesh()
        for exact in (True, False):
            cell, rep = choose_grid_hash_cell(m, exact=exact)
            assert _gh_far_rows_for(m, cell) <= 64
            assert all(r["far_rows"] <= 64 for r in rep["candidates"])

    def test_estuary_exact_mode_cuts_the_strip_cost_by_an_order(self):
        m = _estuary_mesh()
        n_strip = m.metadata["n_strip_elems"]
        rng = np.random.default_rng(0)
        s = rng.choice(n_strip, 3000, replace=False)
        strip = (m.elem_center_lon[s], m.elem_center_lat[s])
        off = (m.elem_center_lon[n_strip:], m.elem_center_lat[n_strip:])
        legacy = m._gh_cell_size
        occ_legacy = np.median(_window_occupancy(m, *strip))
        set_exact_nearest(m, True)
        ref_s = find_nearest_nodes(m, *strip); ref_o = find_nearest_nodes(m, *off)
        cell, rep = choose_grid_hash_cell(m, exact=True)
        assert cell <= legacy / 8 and _gh_far_rows_for(m, cell) <= 64
        legacy_cost = rep["candidates"][0]["mean_cost"]
        chosen_cost = next(r for r in rep["candidates"] if r["cell"] == cell)["mean_cost"]
        assert chosen_cost * 10 <= legacy_cost
        set_grid_hash_cell(m, cell); set_exact_nearest(m, True)
        assert np.median(_window_occupancy(m, *strip)) * 20 <= occ_legacy
        assert np.array_equal(find_nearest_nodes(m, *strip), ref_s)     # exact, as before
        assert np.array_equal(find_nearest_nodes(m, *off), ref_o)
        m._gh_report = rep
        assert "empty" in describe_grid_hash(m)

    def test_estuary_strict_mode_stops_at_the_last_full_window(self):
        m = _estuary_mesh()
        n_strip = m.metadata["n_strip_elems"]
        legacy = m._gh_cell_size
        cell, rep = choose_grid_hash_cell(m, exact=False)
        assert legacy / 4 <= cell < legacy
        set_grid_hash_cell(m, cell)
        off = (m.elem_center_lon[n_strip:], m.elem_center_lat[n_strip:])
        assert (_window_occupancy(m, *off) > 0).all()

    def test_legacy_rule_is_a_no_op_bitwise(self):
        m = _grid_mesh(np.arange(-5.0, 5.01, 0.5), np.arange(-5.0, 5.01, 0.5))
        snap = (m._gh_offsets.copy(), m._gh_node_ids.copy(), m._gh_cell_size, m._gh_lon_min,
                m._gh_lat_min, m._gh_nx, m._gh_ny, m._node_xyz.copy())
        rep = apply_grid_hash_cell_rule(m, "legacy")
        assert rep["rule"] == "legacy" and m.metadata["grid_hash_cell_rule"] == "legacy"
        assert np.array_equal(m._gh_offsets, snap[0]) and np.array_equal(m._gh_node_ids, snap[1])
        assert (m._gh_cell_size, m._gh_lon_min, m._gh_lat_min, m._gh_nx, m._gh_ny) == snap[2:7]
        assert np.array_equal(m._node_xyz, snap[7])
        assert "grid_hash_cell" not in m.metadata            # nothing pinned

    def test_apply_pins_the_choice_and_survives_a_rebuild(self):
        m = _grid_mesh(np.arange(-20.0, 20.01, 0.5), np.arange(-10.0, 10.01, 0.5))
        set_exact_nearest(m, True)
        rep = apply_grid_hash_cell_rule(m, min_gain=1.0)     # default rule, mode from the flag
        assert rep["mode"] == "exact" and m._gh_cell_size == rep["chosen"] < rep["legacy"]
        assert m.metadata["grid_hash_cell"] == rep["chosen"]
        build_search(m)                                      # a later rebuild keeps the pin
        assert m._gh_cell_size == rep["chosen"]
        assert m._gh_report is rep
        # idempotent: a pinned mesh keeps its choice and its rule
        rep2 = apply_grid_hash_cell_rule(m)
        assert rep2["rule"] == "occupancy" and m._gh_cell_size == rep["chosen"]

    def test_reused_mesh_semantics(self):
        """Review 2026-09-22: the pin is mode-aware and honest on a reused mesh."""
        from kelp.core.mesh import build_grid_hash
        m = _estuary_mesh()
        legacy = m._gh_cell_size
        set_exact_nearest(m, True)
        rep = apply_grid_hash_cell_rule(m)
        assert rep["mode"] == "exact" and rep["chosen"] < legacy
        assert m.metadata["grid_hash_cell_mode"] == "exact"
        # the same rule in the same mode: the earlier report, no second ladder
        assert apply_grid_hash_cell_rule(m) is rep
        # a pinned rebuild (particles_gen -> build_search) keeps cell AND report
        build_grid_hash(m)
        assert m._gh_cell_size == rep["chosen"] and m._gh_report is rep
        # the mode flips (guard off): the strict rule decides afresh from the reader's cell
        set_exact_nearest(m, False)
        rep2 = apply_grid_hash_cell_rule(m)
        assert rep2["mode"] == "approx" and rep2["legacy"] == legacy
        assert rep2["chosen"] == m._gh_cell_size and m.metadata["grid_hash_cell_mode"] == "approx"
        row = next(r for r in rep2["candidates"] if r["cell"] == rep2["chosen"])
        assert row["empty"] == 0.0                      # the strict guarantee holds again
        # legacy after occupancy really restores the reader's cell
        rep3 = apply_grid_hash_cell_rule(m, "legacy")
        assert rep3["rule"] == "legacy" and m._gh_cell_size == legacy
        assert "grid_hash_cell" not in m.metadata and m.metadata["grid_hash_cell_rule"] == "legacy"
        # an explicit pin after a rule is reported as explicit, and describe names the current cell
        set_exact_nearest(m, True)
        apply_grid_hash_cell_rule(m)
        set_grid_hash_cell(m, 0.03)
        assert m.metadata["grid_hash_cell_rule"] == "explicit"
        assert apply_grid_hash_cell_rule(m)["rule"] == "explicit" and m._gh_cell_size == 0.03
        assert "cell 0.03 deg (explicit)" in describe_grid_hash(m)

    def test_describe_never_reports_a_stale_cell(self):
        from kelp.core.mesh import build_grid_hash
        m = _estuary_mesh()
        set_exact_nearest(m, True)
        rep = apply_grid_hash_cell_rule(m)
        assert f"cell {rep['chosen']:.4g} deg by occupancy/exact" in describe_grid_hash(m)
        build_grid_hash(m, cell_size=0.02)                 # an unpinned rebuild at another cell
        assert m._gh_report is None
        assert "cell 0.02 deg" in describe_grid_hash(m)

    def test_without_numba_the_reader_cell_is_kept(self, monkeypatch):
        from kelp.core import mesh as mesh_mod
        m = _estuary_mesh()
        monkeypatch.setattr(mesh_mod, "HAS_NUMBA", False)
        cell, rep = choose_grid_hash_cell(m, exact=True)
        assert cell == m._gh_cell_size and "numba unavailable" in rep["reason"]
        assert rep["candidates"] == [] and rep["sample"] == 0

    def test_non_finite_centres_are_not_sampled(self):
        m = _estuary_mesh()
        ref_cell, ref = choose_grid_hash_cell(m, exact=True)
        m.elem_center_lon[:7] = np.nan
        m.elem_center_lat[7:11] = np.inf
        cell, rep = choose_grid_hash_cell(m, exact=True)
        assert rep["sample"] == ref["sample"] - 11 and cell == ref_cell
        m.elem_center_lon[:] = np.nan
        cell, rep = choose_grid_hash_cell(m, exact=True)
        assert cell == m._gh_cell_size and "no finite element centre" in rep["reason"]

    def test_exact_mode_eligibility_bounds(self):
        """The argmin is taken only inside the regime the proxy was validated
        in: widening share and hash memory (review 2026-09-22: the 11.5 M-node
        mesh's argmin was the cap-limited last candidate at 185 MB)."""
        m = _grid_mesh(np.arange(-20.0, 20.01, 0.5), np.arange(-10.0, 10.01, 0.5))
        legacy = m._gh_cell_size
        cell, rep = choose_grid_hash_cell(m, exact=True, min_gain=1.0)
        rows = rep["candidates"]
        assert rows[0]["eligible"] and all("hash_mb" in r for r in rows)
        assert all(r["unprovable"] <= 0.20 and r["hash_mb"] <= 128.0 for r in rows if r["eligible"])
        # the widening bound: at the uniform 0.5 deg spacing a cell far below it widens
        # every anchor; those rows are ineligible however cheap the proxy says they are
        assert any(not r["eligible"] for r in rows)
        # a hash-memory bound below the argmin's grid moves the choice up the ladder
        row = next(r for r in rows if r["cell"] == cell)
        tight = row["hash_mb"] * 0.9
        cell2, rep2 = choose_grid_hash_cell(m, exact=True, min_gain=1.0, max_hash_mb=tight)
        assert cell2 > cell
        assert all(r["hash_mb"] <= tight for r in rep2["candidates"] if r["eligible"] and r["cell"] != legacy)
        # the legacy row is always eligible, even when it violates the bounds itself
        cell3, rep3 = choose_grid_hash_cell(m, exact=True, min_gain=1.0, max_unprovable=0.0,
                                            max_hash_mb=0.0)
        assert cell3 == legacy and rep3["candidates"][0]["eligible"]

    def test_widening_flag_is_the_searchs_own_certificate(self):
        """`unprovable` == the exact search's verdict on the 3x3 window at that
        cell (review 2026-09-22: the uniform-coverage bound over-counted 7.4 %
        vs 3.8 % on CORE2 and missed seam-cell centres)."""
        from kelp.core.mesh import _anchor_provable_rect, _grid_hash_nearest_3x3
        for name, m in _geoms():
            set_exact_nearest(m, True)
            lon, lat = m.elem_center_lon, m.elem_center_lat
            nn = find_nearest_nodes(m, lon, lat)
            cell, rep = choose_grid_hash_cell(m, exact=True, min_gain=1.0, steps=3)
            for r in rep["candidates"]:
                set_grid_hash_cell(m, r["cell"])
                args = (m._gh_cell_size, m._gh_lon_min, m._gh_lat_min, m._gh_nx, m._gh_ny,
                        m._gh_wrap_lon, m.nodes_lon, m.nodes_lat)
                hit = np.array([_grid_hash_nearest_3x3(
                    float(lo), float(la), m._gh_offsets, m._gh_node_ids, *args[:6],
                    m.nodes_lon, m.nodes_lat, m._node_xyz) for lo, la in zip(lon, lat)])
                widened = np.array([
                    h < 0 or h != n or _anchor_provable_rect(float(lo), float(la), int(h), 1, 1,
                                                             *args) != 0
                    for lo, la, h, n in zip(lon, lat, hit, nn)])
                assert r["unprovable"] == pytest.approx(widened.mean(), abs=1e-12), (name, r["cell"])

    def test_explicit_pin_is_respected(self):
        m = _grid_mesh(np.arange(-20.0, 20.01, 0.5), np.arange(-10.0, 10.01, 0.5))
        set_grid_hash_cell(m, 0.77)
        rep = apply_grid_hash_cell_rule(m)
        assert rep["rule"] == "explicit" and m._gh_cell_size == 0.77

    def test_exact_mode_keeps_legacy_below_the_gain_threshold(self):
        m = _grid_mesh(np.arange(-20.0, 20.01, 0.5), np.arange(-10.0, 10.01, 0.5))
        legacy = m._gh_cell_size
        cell, rep = choose_grid_hash_cell(m, exact=True, min_gain=100.0)
        assert cell == legacy and "keep the legacy cell" in rep["reason"]
        cell2, rep2 = choose_grid_hash_cell(m, exact=True, min_gain=1.0)
        assert cell2 < legacy and "x)" in rep2["reason"]

    def test_choice_is_deterministic(self):
        m = _grid_mesh(np.arange(-20.0, 20.01, 0.5), np.arange(-10.0, 10.01, 0.5))
        a, ra = choose_grid_hash_cell(m, exact=True)
        b, rb = choose_grid_hash_cell(m, exact=True)
        assert a == b and ra["candidates"] == rb["candidates"]

    def test_a_sliver_mesh_whose_legacy_cell_already_exceeds_the_budget_keeps_it(self):
        """One 10-degree sliver edge on a 0.01-degree patch: far_rows > 64 already at the
        legacy cell, so the ladder holds the legacy candidate only — no IndexError, the
        legacy cell is kept in both modes and the report says so."""
        xs = np.arange(0.0, 0.5001, 0.01); ys = np.arange(0.0, 0.5001, 0.01)
        base = _grid_mesh(xs, ys)
        lon = np.append(base.nodes_lon, 10.0); lat = np.append(base.nodes_lat, 0.25)
        far = lon.size - 1
        elem = np.vstack([base.elem_nodes, np.array([[0, 1, far]], dtype=np.int32)])
        n2e, n2c = build_node_to_elems(lon.size, elem)
        m = Mesh(name="sliver", nodes_lon=lon, nodes_lat=lat, depth=np.zeros(lon.size),
                 elem_nodes=elem, elem_center_lon=lon[elem].mean(1), elem_center_lat=lat[elem].mean(1),
                 node_to_elems=n2e, node_to_elems_count=n2c)
        build_search(m)
        assert _gh_far_rows_for(m, m._gh_cell_size) > 64
        for exact in (True, False):
            cell, rep = choose_grid_hash_cell(m, exact=exact)
            assert cell == m._gh_cell_size and len(rep["candidates"]) == 1

    def test_a_mesh_without_elements_keeps_the_legacy_cell(self):
        lon = np.linspace(0.0, 1.0, 50); lat = np.zeros(50)
        elem = np.zeros((0, 3), dtype=np.int32)
        n2e, n2c = build_node_to_elems(lon.size, elem)
        m = Mesh(name="nodes_only", nodes_lon=lon, nodes_lat=lat, depth=np.zeros(50), elem_nodes=elem,
                 elem_center_lon=np.zeros(0), elem_center_lat=np.zeros(0),
                 node_to_elems=n2e, node_to_elems_count=n2c)
        build_search(m)
        cell, rep = choose_grid_hash_cell(m, exact=True)
        assert cell == m._gh_cell_size and rep["candidates"] == [] and "no elements" in rep["reason"]
        assert apply_grid_hash_cell_rule(m)["chosen"] == m._gh_cell_size

    def test_unknown_rule_is_rejected(self):
        m = _grid_mesh(np.arange(-5.0, 5.01, 0.5), np.arange(-5.0, 5.01, 0.5))
        with pytest.raises(ValueError, match="occupancy.*legacy"):
            apply_grid_hash_cell_rule(m, "foo")


# --------------------------------------------------------------------------
# Task 3: a real FESOM-C mesh (southern North Sea, multi-resolution quads)
# --------------------------------------------------------------------------

_FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"


def _sns_mesh():
    z = np.load(_FIXTURES / "fesomc_sns_multi_m8.npz")
    lon, lat, depth, elem = z["lon"], z["lat"], z["depth"], z["elem"]
    tri = elem[:, 3] == elem[:, 0]
    nv = np.where(tri, 3, 4)
    cx = np.array([lon[e[:k]].mean() for e, k in zip(elem, nv)])
    cy = np.array([lat[e[:k]].mean() for e, k in zip(elem, nv)])
    n2e, n2c = build_node_to_elems(lon.size, elem)
    m = Mesh(name="fesomc_sns", nodes_lon=lon, nodes_lat=lat, depth=depth, elem_nodes=elem,
             elem_center_lon=cx, elem_center_lat=cy, node_to_elems=n2e, node_to_elems_count=n2c)
    build_search(m)
    return m


class TestRealFesomCMesh:

    def test_fixture_is_the_north_sea_mesh(self):
        m = _sns_mesh()
        assert m.n_nodes == 43318 and m.n_elements == 42314
        assert 2.0 < m.nodes_lon.min() < 3.0 and 55.0 < m.nodes_lat.max() < 57.0

    def test_exact_mode_choice_and_anchors(self):
        m = _sns_mesh()
        rng = np.random.default_rng(0)
        s = rng.choice(m.n_elements, 20000, replace=False)
        lon, lat = m.elem_center_lon[s], m.elem_center_lat[s]
        set_exact_nearest(m, True)
        ref = find_nearest_nodes(m, lon, lat)
        legacy = m._gh_cell_size
        rep = apply_grid_hash_cell_rule(m)
        # the ladder neighbours of the measured optimum (0.014 / 0.020 / 0.029 / 0.041)
        assert rep["mode"] == "exact" and 0.012 <= rep["chosen"] <= 0.045 < legacy
        chosen_row = next(r for r in rep["candidates"] if r["cell"] == rep["chosen"])
        assert chosen_row["empty"] <= 0.10 and chosen_row["median"] >= 8
        rows = {r["cell"]: r for r in rep["candidates"]}
        assert rows[rep["chosen"]]["mean_cost"] <= rows[legacy]["mean_cost"]
        assert _gh_far_rows_for(m, rep["chosen"]) <= 64
        assert np.array_equal(find_nearest_nodes(m, lon, lat), ref)
        line = describe_grid_hash(m)
        assert "by occupancy/exact" in line and "legacy" in line
        print(line)

    def test_strict_mode_choice(self):
        m = _sns_mesh()
        legacy = m._gh_cell_size
        cell, rep = choose_grid_hash_cell(m, exact=False)
        assert cell < legacy and rep["mode"] == "approx"
        set_grid_hash_cell(m, cell)
        lon, lat = m.elem_center_lon, m.elem_center_lat
        occ = _window_occupancy(m, lon, lat)
        assert (occ > 0).all() and np.median(occ) >= 8
        assert _gh_far_rows_for(m, cell) <= 64


# --------------------------------------------------------------------------
# Task 4: config, run_tracking, metadata
# --------------------------------------------------------------------------

from kelp.core.config import MeshConfig, load_config   # noqa: E402
from kelp import tracking                             # noqa: E402
from tests.test_grid_hash_cell import TestTrackingWiring   # noqa: E402  (its synthetic FESOM-C writer)


class TestConfig:

    def test_default_and_validation(self):
        assert MeshConfig().grid_hash_cell_rule == "occupancy"
        assert MeshConfig(grid_hash_cell_rule="LEGACY").grid_hash_cell_rule == "legacy"
        with pytest.raises(ValueError, match="occupancy.*legacy"):
            MeshConfig(grid_hash_cell_rule="foo")
        assert MeshConfig(grid_hash_cell_rule=None).grid_hash_cell_rule == "occupancy"   # YAML null

    def test_config_and_engine_agree_on_the_rules(self):
        from kelp.core import config as config_mod
        from kelp.core.mesh import GRID_HASH_CELL_RULES
        assert tuple(config_mod._GRID_HASH_CELL_RULES) == tuple(GRID_HASH_CELL_RULES)


def _write_fesomc_nonuniform(tmp_path):
    """A FESOM-C file the rule acts on: a 0.4 deg box (0..4 deg, 121 nodes)
    plus a 0.01 deg patch (2.0..2.3 deg, 961 nodes, 90 % of the elements) —
    two disjoint triangulations, uniform flow.  On the uniform 11x11 fixture
    of tests/test_grid_hash_cell.py exact mode KEEPS the legacy cell, so an
    end-to-end comparison there never exercised a changed cell (review
    2026-09-22)."""
    import xarray as xr

    def tri(xs, ys, base):
        gx, gy = np.meshgrid(xs, ys)
        nx, ny = len(xs), len(ys)
        el = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                n0 = base + j * nx + i + 1                   # 1-based
                el.append([n0, n0 + 1, n0 + nx, n0])
                el.append([n0 + 1, n0 + nx + 1, n0 + nx, n0 + 1])
        return gx.ravel(), gy.ravel(), el
    lon1, lat1, el1 = tri(np.linspace(0.0, 4.0, 11), np.linspace(0.0, 4.0, 11), 0)
    lon2, lat2, el2 = tri(np.linspace(2.0, 2.3, 31), np.linspace(2.0, 2.3, 31), lon1.size)
    lon = np.concatenate([lon1, lon2]); lat = np.concatenate([lat1, lat2])
    nv = np.array(el1 + el2, dtype=np.int32)
    tri0 = nv[:, :3] - 1
    n_t, n_e, n_n = 3, nv.shape[0], lon.size
    nc = tmp_path / "fesomc_nonuniform.nc"
    xr.Dataset({
        "lon": (["node"], lon), "lat": (["node"], lat),
        "depth": (["node"], np.full(n_n, 10.0)),
        "nv": (["nele", "four"], nv),
        "lon_elem": (["nele"], lon[tri0].mean(axis=1)),
        "lat_elem": (["nele"], lat[tri0].mean(axis=1)),
        "sigma_lev": (["sigma"], np.array([0.0, 0.5, 1.0])),
        "u2d": (["time", "nele"], np.full((n_t, n_e), 0.1)),
        "v2d": (["time", "nele"], np.full((n_t, n_e), 0.05)),
        "eta": (["time", "node"], np.zeros((n_t, n_n))),
    }).to_netcdf(nc)
    return nc


# seeds off any grid symmetry (no exact nearest-node ties): three in the patch, one in the box
_NONUNIFORM_SEEDS = "2.0731 2.0817\n2.1543 2.2209\n2.2817 2.0163\n0.7391 3.2887\n"


def _run_with(tmp_path, mesh_lines, nonuniform=False):
    """One synthetic FESOM-C run (the uniform writer of tests/test_grid_hash_cell.py,
    or the non-uniform one above), extra `mesh:` lines injected; returns the
    output path."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    seeds = tmp_path / "p.dat"
    if nonuniform:
        nc = _write_fesomc_nonuniform(tmp_path)
        seeds.write_text(_NONUNIFORM_SEEDS)
    else:
        nc = TestTrackingWiring._write_fesomc(tmp_path)
        seeds.write_text("5.0 5.0\n5.1 5.1\n3.3 6.6\n7.7 2.2\n")
    out = tmp_path / "out.nc"
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"""\
mode: {{Fdrift2D: true, time_dir: 1.0}}
numerics: {{dt: 100.0, integrator: rk2, output_freq: 1}}
mesh:
  type: fesomc
  meshpath: {nc}
{mesh_lines}data:
  ncinput: {nc}
  inicloud: {seeds}
  output_path: {out}
  time_averaging: instant
time: {{start_time_idx: 0, n_steps: 3}}
""")
    tracking.run_tracking(load_config(cfg))
    return out


class TestRunTracking:

    def test_metadata_and_legacy_identity_uniform(self, tmp_path):
        """The uniform fixture: exact mode keeps the reader's cell (predicted
        gain < 2x) — the metadata says so and the runs are byte-identical."""
        import xarray as xr
        out_legacy = _run_with(tmp_path / "legacy", "  grid_hash_cell_rule: legacy\n")
        out_rule = _run_with(tmp_path / "rule", "")
        out_explicit = _run_with(tmp_path / "explicit", "  grid_hash_cell_size: 0.3\n")
        with xr.open_dataset(out_legacy) as a, xr.open_dataset(out_rule) as b, \
                xr.open_dataset(out_explicit) as c:
            assert a.attrs["grid_hash_cell_rule"] == "legacy"
            assert b.attrs["grid_hash_cell_rule"] == "occupancy"
            assert c.attrs["grid_hash_cell_rule"] == "explicit"
            assert float(c.attrs["grid_hash_cell"]) == 0.3
            assert float(b.attrs["grid_hash_cell"]) == float(a.attrs["grid_hash_cell"])
            for k in ("lon", "lat", "status"):
                assert np.array_equal(a[k].values, b[k].values, equal_nan=True), k

    @pytest.mark.parametrize("guard", [True, False])
    def test_changed_cell_end_to_end(self, tmp_path, guard):
        """The non-uniform fixture: the rule CHANGES the cell in both modes,
        the pin survives the seed filter's rebuild, the metadata carries the
        chosen cell, and the trajectories equal the legacy run's bit for bit
        (exact anchors under the guard; no approximate anchor added by the
        strict rule)."""
        import xarray as xr
        g = f"  exact_nearest: {str(guard).lower()}\n"
        out_legacy = _run_with(tmp_path / "legacy", g + "  grid_hash_cell_rule: legacy\n", True)
        out_rule = _run_with(tmp_path / "rule", g, True)
        with xr.open_dataset(out_legacy) as a, xr.open_dataset(out_rule) as b:
            assert a.attrs["grid_hash_cell_rule"] == "legacy"
            assert b.attrs["grid_hash_cell_rule"] == "occupancy"
            # exact mode: the argmin, a few ladder steps down; strict mode: one step
            # (the 10 % coarse-region windows go empty right below it)
            factor = 0.5 if guard else 0.75
            assert float(b.attrs["grid_hash_cell"]) < factor * float(a.attrs["grid_hash_cell"])
            assert int(b.attrs["exact_nearest"]) == int(guard)
            assert (b["status"].values == 1).all()
            for k in ("lon", "lat", "status"):
                assert np.array_equal(a[k].values, b[k].values, equal_nan=True), k

    def test_track_particles_applies_the_rule(self, tmp_path):
        """The functional API: the real call, on the non-uniform mesh."""
        from kelp.readers import get_reader
        nc = _write_fesomc_nonuniform(tmp_path)
        reader = get_reader("fesomc")
        seeds = np.loadtxt(pathlib.Path(tmp_path / "p.dat").write_text(_NONUNIFORM_SEEDS) and
                           tmp_path / "p.dat")
        results = {}
        for rule in ("occupancy", "legacy"):
            mesh = reader.load_mesh(MeshConfig(type="fesomc", meshpath=str(nc)))
            legacy = mesh._gh_cell_size
            ds = reader.open_dataset(str(nc))
            try:
                rec = tracking.track_particles(mesh, ds, reader, seeds[:, 0], seeds[:, 1],
                                               n_steps=2, dt=100.0, grid_hash_cell_rule=rule)
            finally:
                ds.close()
            results[rule] = rec.to_arrays()
            assert mesh.metadata["grid_hash_cell_rule"] == rule
            if rule == "occupancy":
                assert mesh._gh_cell_size < 0.5 * legacy
                assert mesh.metadata["grid_hash_cell"] == mesh._gh_cell_size
                assert mesh.metadata["grid_hash_cell_mode"] == "exact"
            else:
                assert mesh._gh_cell_size == legacy and "grid_hash_cell" not in mesh.metadata
        for k in ("lon", "lat", "status"):
            assert np.array_equal(results["occupancy"][k], results["legacy"][k]), k
