"""Exact nearest-node anchor guard (mesh.exact_nearest).

The 3x3 grid-hash window can be NON-EMPTY and still miss the true nearest node: any node
closer than the found one lies within its angular distance theta of the query, i.e. within
theta in latitude and theta / cos(lat) in longitude, and that can exceed one cell at high
latitude (or with a small explicit cell).  The guard tests that condition per query: when
the window is not conclusive the search widens the window IN-KERNEL until the certificate
proves the best node (`_grid_hash_nearest_exact`) — the batch path and every fused 2-D /
3-D Euler kernel call that same function; the fused 3-D RK2 barycentric kernels flag the
site for the runner's heal through the batch path.  On by default since 2026-09-12;
`mesh.exact_nearest: false` restores the pre-guard anchors bit for bit.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kelp.core.config import MeshConfig  # noqa: E402
from kelp.core.mesh import (build_grid_hash, find_nearest_nodes,  # noqa: E402
                                set_exact_nearest, set_grid_hash_cell,
                                _nearest_provably_exact)
from kelp.core.particles import make_state  # noqa: E402


def _brute_nearest(mesh, lon, lat):
    """True nearest node by 3-D chord distance on the unit sphere."""
    def xyz(lo, la):
        lo, la = np.deg2rad(lo), np.deg2rad(la)
        return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    N = xyz(mesh.nodes_lon, mesh.nodes_lat); Q = xyz(lon, lat)
    return np.array([int(np.argmin(((N - q) ** 2).sum(1))) for q in Q], dtype=np.int32)


def _chord(mesh, idx, lon, lat):
    def xyz(lo, la):
        lo, la = np.deg2rad(lo), np.deg2rad(la)
        return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    return np.sqrt(((xyz(mesh.nodes_lon[idx], mesh.nodes_lat[idx]) - xyz(lon, lat)) ** 2).sum(1))


@pytest.fixture(scope="module")
def coarse_mesh():
    """HEALPix nside 8 dual mesh (768 nodes, ~7 deg spacing): with an explicit 3-deg cell the
    3x3 window is populated almost everywhere but misses the true nearest node for many
    high-latitude queries — the CORE2-rotated-frame situation in miniature."""
    healpy = pytest.importorskip("healpy")
    from kelp.readers.healpix import build_healpix_dual_mesh
    nside = 8; npix = 12 * nside * nside
    lon, lat = healpy.pix2ang(nside, np.arange(npix), nest=True, lonlat=True)
    m = build_healpix_dual_mesh(nside, wet_mask=np.ones(npix, bool),
                                depth=np.full(npix, 3000.0),
                                z_levels=np.array([0.0, 50.0, 500.0, 4000.0]))
    set_grid_hash_cell(m, 3.0)
    return m


def _queries(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    lon = rng.uniform(-180.0, 180.0, n)
    lat = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, n)))
    return lon, lat


class TestProvableCondition:
    def test_close_node_is_provable(self):
        # query at 60N, found node 0.3 deg away, cell 1 deg: theta/cos(60.3) = 0.6 < 1
        assert _nearest_provably_exact(10.0, 60.0, 10.3, 60.0, 1.0)

    def test_far_node_at_high_latitude_is_not_provable(self):
        # query at 85N, found node 0.6 deg away: theta/cos(85.6) = 7.8 deg > cell 1 deg
        assert not _nearest_provably_exact(10.0, 85.0, 10.0, 85.6, 1.0)

    def test_node_beyond_one_cell_is_never_provable(self):
        assert not _nearest_provably_exact(0.0, 0.0, 0.0, 1.5, 1.0)

    def test_coincident_node_is_provable(self):
        assert _nearest_provably_exact(0.0, 89.0, 0.0, 89.0, 0.1)


class TestBatchExactMode:
    def test_default_mode_misses_true_nearest_for_some_queries(self, coarse_mesh):
        lon, lat = _queries()
        got = find_nearest_nodes(coarse_mesh, lon, lat)
        truth = _brute_nearest(coarse_mesh, lon, lat)
        assert (got != truth).sum() > 0          # the symptom the guard fixes

    def test_exact_mode_returns_the_true_nearest_everywhere(self, coarse_mesh):
        lon, lat = _queries()
        set_exact_nearest(coarse_mesh, True)
        try:
            got = find_nearest_nodes(coarse_mesh, lon, lat)
        finally:
            set_exact_nearest(coarse_mesh, False)
        truth = _brute_nearest(coarse_mesh, lon, lat)
        # compare DISTANCES, not indices: an exactly symmetric tie may be broken differently
        # by the window scan and by argmin (review round 2, item 10)
        d_got = _chord(coarse_mesh, got, lon, lat); d_true = _chord(coarse_mesh, truth, lon, lat)
        assert np.allclose(d_got, d_true, rtol=0.0, atol=1e-12)

    def test_exact_mode_changes_only_the_misses(self, coarse_mesh):
        lon, lat = _queries(seed=1)
        base = find_nearest_nodes(coarse_mesh, lon, lat)
        set_exact_nearest(coarse_mesh, True)
        try:
            exact = find_nearest_nodes(coarse_mesh, lon, lat)
        finally:
            set_exact_nearest(coarse_mesh, False)
        truth = _brute_nearest(coarse_mesh, lon, lat)
        same = base == truth
        assert np.array_equal(exact[same], base[same])

    def test_flag_survives_a_hash_rebuild(self, coarse_mesh):
        set_exact_nearest(coarse_mesh, True)
        try:
            build_grid_hash(coarse_mesh)
            assert coarse_mesh.metadata["exact_nearest"] is True
        finally:
            set_exact_nearest(coarse_mesh, False)
        assert not coarse_mesh.metadata.get("exact_nearest", False)


class TestFused3DExact:
    @pytest.mark.parametrize("mode", ["tinterp", "pchip"])
    def test_fused_kernel_matches_unfused_with_exact_anchors(self, coarse_mesh, mode):
        """With the guard on, the fused 3-D kernel flags every stage anchor that the window
        cannot prove and the runner heals it — bitwise parity with the unfused path, whose
        stage anchors come from the (now exact) batch search."""
        from kelp.core.io import VelocityField
        from kelp.numerics.integrators import (_rk2_step_unfused,
                                                    _run_fused_rk2_3d_bary_tinterp,
                                                    W_BARYCENTRIC)
        m = coarse_mesh
        K = m.sigma.size; N = m.n_nodes
        rng = np.random.default_rng(3)
        vel = VelocityField(u=0.5 * rng.standard_normal((K - 1, N)),
                            v=0.5 * rng.standard_normal((K - 1, N)),
                            eta=np.zeros(N), w=1e-4 * rng.standard_normal((K, N)),
                            uv_layout="node")
        n = 1500
        lon = rng.uniform(-179.0, 179.0, n)
        lat = np.degrees(np.arcsin(rng.uniform(-0.999, 0.999, n)))   # incl. polar caps
        st = make_state(lon, lat, rng.uniform(10.0, 2000.0, n))
        st = st.replace(failure_reason=np.zeros(n, dtype=np.int8))
        set_exact_nearest(m, True)
        try:
            st = st.replace(nearest_node=find_nearest_nodes(m, lon, lat))
            if mode == "tinterp":
                got = _run_fused_rk2_3d_bary_tinterp(st, vel, None, None, m, 5400.0, 1.0, W_BARYCENTRIC)
                ref = _rk2_step_unfused(st, vel, m, 5400.0, 1.0, 0, True,
                                        w_interp="barycentric", uv_interp="barycentric")
            else:
                from kelp.numerics.integrators import _run_fused_rk2_3d_bary_pchip
                snaps = [VelocityField(u=0.5 * rng.standard_normal((K - 1, N)), v=0.5 * rng.standard_normal((K - 1, N)),
                                       eta=np.zeros(N), w=1e-4 * rng.standard_normal((K, N)), uv_layout="node") for _ in range(4)]
                recon = (snaps, [0.0, 86400.0, 172800.0, 259200.0], 100000.0, 1.0)
                got = _run_fused_rk2_3d_bary_pchip(st, recon, m, 5400.0, 1.0, W_BARYCENTRIC)
                ref = _rk2_step_unfused(st, snaps[1], m, 5400.0, 1.0, 0, True,
                                        w_interp="barycentric", uv_interp="barycentric", recon=recon)
        finally:
            set_exact_nearest(m, False)
        for name in ("lon", "lat", "z", "status", "nearest_node", "u", "v", "w"):
            a, b = getattr(got, name), getattr(ref, name)
            assert np.array_equal(a, b, equal_nan=True), name

    def test_guard_off_leaves_the_fused_kernel_byte_identical(self, coarse_mesh):
        """Default off: the flag must not change the kernel's result at all."""
        from kelp.core.io import VelocityField
        from kelp.numerics.integrators import _run_fused_rk2_3d_bary_tinterp, W_BARYCENTRIC
        m = coarse_mesh; K = m.sigma.size; N = m.n_nodes
        rng = np.random.default_rng(4)
        vel = VelocityField(u=0.5 * rng.standard_normal((K - 1, N)), v=0.5 * rng.standard_normal((K - 1, N)),
                            eta=np.zeros(N), w=1e-4 * rng.standard_normal((K, N)), uv_layout="node")
        lon, lat = _queries(400, seed=5)
        st = make_state(lon, lat, np.full(400, 100.0)).replace(failure_reason=np.zeros(400, dtype=np.int8))
        st = st.replace(nearest_node=find_nearest_nodes(m, lon, lat))
        a = _run_fused_rk2_3d_bary_tinterp(st, vel, None, None, m, 5400.0, 1.0, W_BARYCENTRIC)
        b = _run_fused_rk2_3d_bary_tinterp(st, vel, None, None, m, 5400.0, 1.0, W_BARYCENTRIC)
        assert np.array_equal(a.lon, b.lon) and np.array_equal(a.nearest_node, b.nearest_node)


class TestConfigAndWiring:
    def test_default_is_on(self):
        assert MeshConfig().exact_nearest is True

    def test_coerces_bool(self):
        assert MeshConfig(exact_nearest=1).exact_nearest is True

    def _run(self, tmp_path, line):
        from kelp.core.config import load_config
        from kelp import tracking
        from kelp.core import mesh as mesh_mod
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "test_grid_hash_cell", Path(__file__).with_name("test_grid_hash_cell.py"))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        nc = mod.TestTrackingWiring._write_fesomc(tmp_path)
        seeds = tmp_path / "p.dat"; seeds.write_text("5.0 5.0\n5.1 5.1\n")
        cfg = tmp_path / "c.yaml"
        cfg.write_text(f"""\
mode: {{Fdrift2D: true, time_dir: 1.0}}
numerics: {{dt: 100.0, integrator: euler, output_freq: 1}}
mesh:
  type: fesomc
  meshpath: {nc}
{line}data:
  ncinput: {nc}
  inicloud: {seeds}
  output_path: {tmp_path / "out.nc"}
  time_averaging: instant
time: {{start_time_idx: 0, n_steps: 2}}
""")
        seen = {}
        real = mesh_mod.set_exact_nearest

        def spy(mesh, flag):
            seen["flag"] = flag; seen["mesh"] = mesh
            return real(mesh, flag)
        mesh_mod.set_exact_nearest = spy
        try:
            tracking.run_tracking(load_config(cfg))
        finally:
            mesh_mod.set_exact_nearest = real
        return seen, tmp_path / "out.nc"

    def test_applied_when_set(self, tmp_path):
        import netCDF4
        seen, out = self._run(tmp_path, "  exact_nearest: true\n")
        assert seen["flag"] is True and seen["mesh"].metadata["exact_nearest"] is True
        with netCDF4.Dataset(out) as ds:
            assert getattr(ds, "exact_nearest", None) in (1, "true", "True", True)

    def test_on_when_unset(self, tmp_path):
        """Default on (2026-09-12): an unset key runs the guard and stamps the metadata; set
        `false` to reproduce pre-guard runs bit for bit."""
        import netCDF4
        seen, out = self._run(tmp_path, "")
        assert seen["flag"] is True and seen["mesh"].metadata["exact_nearest"] is True
        with netCDF4.Dataset(out) as ds:
            assert getattr(ds, "exact_nearest", None) in (1, "true", "True", True)

    def test_off_when_set_false(self, tmp_path):
        import netCDF4
        seen, out = self._run(tmp_path, "  exact_nearest: false\n")
        assert seen["flag"] is False and seen["mesh"].metadata["exact_nearest"] is False
        with netCDF4.Dataset(out) as ds:
            assert int(ds.getncattr("exact_nearest")) == 0      # stamped 0/1 since the default flipped


# ---------------------------------------------------------------------------
# Review round 1 (2026-09-12): the certificate must use the ACTUAL searched
# intervals, and every public path must honour the flag.
# ---------------------------------------------------------------------------
def _mesh_from_points(lon, lat, cell):
    from scipy.spatial import Delaunay
    from kelp.core.mesh import Mesh, build_node_to_elems, build_search
    pts = np.column_stack([lon, lat]); tri = Delaunay(pts).simplices.astype(np.int32)
    m = Mesh(name="pts", nodes_lon=lon.astype(float), nodes_lat=lat.astype(float),
             depth=np.full(lon.size, 100.0), elem_nodes=tri,
             elem_center_lon=lon[tri].mean(1), elem_center_lat=lat[tri].mean(1),
             node_to_elems=None, node_to_elems_count=None, sigma=np.array([]))
    m.node_to_elems, m.node_to_elems_count = build_node_to_elems(lon.size, tri)
    build_search(m); set_grid_hash_cell(m, cell)
    return m


class TestCertificateGeometry:
    def test_wrap_grid_shortened_seam_cell(self):
        """cell = 7 deg -> nx = 52, the last cell is only 3 deg wide: the window across the seam
        covers less than one cell, so a hit there must NOT be certified (review counterexample)."""
        lon = np.r_[5.0, 356.9, np.arange(20.0, 351.0, 5.0)]
        lat = np.r_[0.0, 0.0, np.full(lon.size - 2, 40.0)]
        m = _mesh_from_points(lon, lat, 7.0)
        set_exact_nearest(m, True)
        got = find_nearest_nodes(m, np.array([0.1]), np.array([0.0]))
        assert got[0] == _brute_nearest(m, np.array([0.1]), np.array([0.0]))[0] == 1

    def test_padded_non_wrap_grid_branch_overlap(self):
        """nodes at lon 0..350 (non-wrap grid padded to -20): a query at -10 deg is physically the
        node at 350 deg, which sits at the far end of the grid (review counterexample)."""
        lon = np.r_[np.arange(0.0, 351.0, 10.0), np.arange(0.0, 351.0, 10.0)]
        lat = np.r_[np.zeros(36), np.full(36, 10.0)]
        m = _mesh_from_points(lon, lat, 20.0)
        set_exact_nearest(m, True)
        got = find_nearest_nodes(m, np.array([-10.0]), np.array([0.0]))
        assert got[0] == _brute_nearest(m, np.array([-10.0]), np.array([0.0]))[0]
        assert m.nodes_lon[got[0]] == 350.0

    def test_query_outside_the_grid_is_healed(self):
        rng = np.random.default_rng(7)
        lon = rng.uniform(0.0, 30.0, 300); lat = rng.uniform(60.0, 70.0, 300)
        m = _mesh_from_points(lon, lat, 0.5)
        set_exact_nearest(m, True)
        q_lon = np.array([15.0, -5.0, 35.0]); q_lat = np.array([75.0, 65.0, 55.0])
        assert np.array_equal(find_nearest_nodes(m, q_lon, q_lat), _brute_nearest(m, q_lon, q_lat))


class TestDispatchHonoursTheFlag:
    def _highlat_zlevel_mesh(self):
        from kelp.readers.structured import build_structured_mesh
        lon1d = np.arange(0.0, 40.0, 1.0); lat1d = np.arange(80.0, 88.5, 0.5)
        lon2d, lat2d = np.meshgrid(lon1d, lat1d)
        m = build_structured_mesh(lon2d, lat2d, np.ones_like(lon2d, bool))
        m.sigma = np.array([0.0, 10.0, 25.0, 50.0, 100.0, 250.0])
        m.depth = np.full(m.n_nodes, 300.0)
        set_grid_hash_cell(m, 0.3)     # 1-deg lon spacing > cell: windows miss the true nearest
        return m

    @pytest.mark.parametrize("w_interp", ["nearest", "leastsq"])
    def test_fused_3d_euler_runs_and_heals_when_exact(self, w_interp, monkeypatch):
        """The fused 3-D Euler kernel flags unprovable anchors for the runner's KD heal: with
        the flag on it is USED (not bypassed) and equals the unfused (exact) path bitwise."""
        from kelp.core.io import VelocityField
        import kelp.numerics.integrators as I
        m = self._highlat_zlevel_mesh(); K = m.sigma.size; N, E = m.n_nodes, m.n_elements
        rng = np.random.default_rng(11)
        vel = VelocityField(u=0.3 * rng.standard_normal((K - 1, E)), v=0.3 * rng.standard_normal((K - 1, E)),
                            eta=np.zeros(N), w=1e-3 * rng.standard_normal((K, N)), uv_layout="element")
        n = 1000
        lon = rng.uniform(0.5, 39.0, n); lat = rng.uniform(80.2, 87.8, n)
        st = make_state(lon, lat, rng.uniform(5.0, 200.0, n)).replace(failure_reason=np.zeros(n, dtype=np.int8))
        set_exact_nearest(m, True)
        st = st.replace(nearest_node=find_nearest_nodes(m, lon, lat))
        calls = {"n": 0}; orig = I._run_fused_euler_3d

        def spy(*a, **k):
            calls["n"] += 1
            return orig(*a, **k)
        monkeypatch.setattr(I, "_run_fused_euler_3d", spy)
        got = I.euler_step(st, vel, m, 3600.0, 1.0, w_interp=w_interp, uv_interp="leastsq")
        ref = I._euler_step_unfused(st, vel, m, 3600.0, 1.0, 0, True, w_interp=w_interp, uv_interp="leastsq")
        assert calls["n"] == 1, "the fused 3-D Euler kernel must run under exact_nearest"
        for name in ("lon", "lat", "z", "status", "nearest_node", "u", "v", "w", "failure_reason"):
            assert np.array_equal(getattr(got, name), getattr(ref, name), equal_nan=True), name

    def test_fused_2d_kernels_run_and_heal_when_exact(self, monkeypatch):
        """Dispatch keeps the fused 2-D kernels under the guard (they heal in-kernel + runner)."""
        from kelp.core.io import VelocityField
        import kelp.numerics.integrators as I
        m = self._highlat_zlevel_mesh(); E, N = m.n_elements, m.n_nodes
        rng = np.random.default_rng(12)
        vel = VelocityField(u=0.3 * rng.standard_normal(E), v=0.3 * rng.standard_normal(E), eta=np.zeros(N), w=None, uv_layout="element")
        n = 500
        lon = rng.uniform(0.5, 39.0, n); lat = rng.uniform(80.2, 87.8, n)
        st = make_state(lon, lat).replace(failure_reason=np.zeros(n, dtype=np.int8))
        set_exact_nearest(m, True)
        st = st.replace(nearest_node=find_nearest_nodes(m, lon, lat))
        calls = {}
        for fn in ("_run_fused_2d", "_run_fused_2d_tinterp", "_run_fused_2d_pchip", "_run_fused_euler_2d_pair"):
            orig = getattr(I, fn)

            def spy(*a, _fn=fn, _orig=orig, **k):
                calls[_fn] = calls.get(_fn, 0) + 1
                return _orig(*a, **k)
            monkeypatch.setattr(I, fn, spy)
        for step, unfused in ((I.euler_step, I._euler_step_unfused), (I.rk2_step, I._rk2_step_unfused),
                              (I.rk4_step, I._rk4_step_unfused)):
            got = step(st, vel, m, 3600.0, 1.0, uv_interp="leastsq")
            ref = unfused(st, vel, m, 3600.0, 1.0, 0, False, uv_interp="leastsq")
            for name in ("lon", "lat", "status", "nearest_node", "u", "v", "failure_reason"):
                assert np.array_equal(getattr(got, name), getattr(ref, name), equal_nan=True), (step.__name__, name)
        assert calls.get("_run_fused_2d", 0) == 3, calls

    def test_rk45_runs_exact_under_the_guard(self):
        """rk45's adaptive kernel anchors every stage with the same in-kernel exact search as the
        fused kernels: no rejection; on the unprovable fixture the guard changes some particles,
        on an all-provable mesh it changes nothing."""
        from kelp.core.io import VelocityField
        from kelp.numerics.integrators import rk45_step
        from kelp.readers.structured import build_structured_mesh
        m = self._highlat_zlevel_mesh(); E, N = m.n_elements, m.n_nodes
        rng = np.random.default_rng(13)
        vel = VelocityField(u=0.3 * rng.standard_normal(E), v=0.3 * rng.standard_normal(E), eta=np.zeros(N), w=None, uv_layout="element")
        n = 400
        lon = rng.uniform(0.5, 39.0, n); lat = rng.uniform(80.2, 87.8, n)
        st = make_state(lon, lat).replace(failure_reason=np.zeros(n, dtype=np.int8))
        st = st.replace(nearest_node=find_nearest_nodes(m, lon, lat))
        off = rk45_step(st, vel, m, 3600.0, 1.0, uv_interp="leastsq")
        set_exact_nearest(m, True)
        try:
            on = rk45_step(st, vel, m, 3600.0, 1.0, uv_interp="leastsq")
        finally:
            set_exact_nearest(m, False)
        assert not np.array_equal(off.lon, on.lon)
        # all-provable mesh: identical
        lon1d = np.arange(0.0, 20.0, 0.5); lat1d = np.arange(40.0, 50.0, 0.5)
        lon2d, lat2d = np.meshgrid(lon1d, lat1d)
        m2 = build_structured_mesh(lon2d, lat2d, np.ones_like(lon2d, bool))
        vel2 = VelocityField(u=0.3 * rng.standard_normal(m2.n_elements), v=0.3 * rng.standard_normal(m2.n_elements),
                             eta=np.zeros(m2.n_nodes), w=None, uv_layout="element")
        lon = rng.uniform(1.0, 19.0, n); lat = rng.uniform(41.0, 49.0, n)
        st2 = make_state(lon, lat).replace(failure_reason=np.zeros(n, dtype=np.int8), nearest_node=find_nearest_nodes(m2, lon, lat))
        a = rk45_step(st2, vel2, m2, 3600.0, 1.0, uv_interp="leastsq")
        set_exact_nearest(m2, True)
        try:
            b = rk45_step(st2, vel2, m2, 3600.0, 1.0, uv_interp="leastsq")
        finally:
            set_exact_nearest(m2, False)
        for name in ("lon", "lat", "status", "nearest_node", "u", "v"):
            assert np.array_equal(getattr(a, name), getattr(b, name)), name


class TestReviewRound2:
    def test_rk45_run_tracking_under_the_guard(self, tmp_path):
        """End to end: rk45 + exact_nearest (now the default) runs and stamps the metadata."""
        from kelp.core.config import load_config
        from kelp import tracking
        import importlib.util
        spec = importlib.util.spec_from_file_location("test_grid_hash_cell", Path(__file__).with_name("test_grid_hash_cell.py"))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        nc = mod.TestTrackingWiring._write_fesomc(tmp_path)
        seeds = tmp_path / "p.dat"; seeds.write_text("5.0 5.0\n")
        cfg = tmp_path / "c.yaml"; out = tmp_path / "out.nc"
        cfg.write_text(f"""\
mode: {{Fdrift2D: true, time_dir: 1.0}}
numerics: {{dt: 100.0, integrator: rk45, output_freq: 1}}
mesh: {{type: fesomc, meshpath: {nc}, exact_nearest: true}}
data: {{ncinput: {nc}, inicloud: {seeds}, output_path: {out}, time_averaging: instant}}
time: {{start_time_idx: 0, n_steps: 2}}
""")
        tracking.run_tracking(load_config(cfg))
        import netCDF4
        with netCDF4.Dataset(out) as ds:
            assert int(ds.getncattr("exact_nearest")) == 1

    def test_anchor_certificate_agrees_with_the_cell_fold_west_of_a_non_wrap_grid(self):
        """review round 2 item 1: the certificate must reason about the cell the window actually
        searched (_gh_cell_of folds far-west queries), and remain exact there."""
        rng = np.random.default_rng(23)
        lon = rng.uniform(300.0, 350.0, 400); lat = rng.uniform(70.0, 80.0, 400)
        m = _mesh_from_points(lon, lat, 0.5)
        set_exact_nearest(m, True)
        q_lon = np.array([-12.0, -20.0, 5.0]); q_lat = np.array([75.0, 72.0, 78.0])   # west of the grid, folded
        got = find_nearest_nodes(m, q_lon, q_lat)
        assert np.allclose(_chord(m, got, q_lon, q_lat), _chord(m, _brute_nearest(m, q_lon, q_lat), q_lon, q_lat), atol=1e-12)


def _highlat_2d_mesh():
    """Structured 1 deg x 0.5 deg grid at 80-88 N with a 0.3-deg hash cell: the 3x3 window is
    non-empty but misses the true nearest node for a good fraction of the queries."""
    from kelp.readers.structured import build_structured_mesh
    lon1d = np.arange(0.0, 40.0, 1.0); lat1d = np.arange(80.0, 88.5, 0.5)
    lon2d, lat2d = np.meshgrid(lon1d, lat1d)
    m = build_structured_mesh(lon2d, lat2d, np.ones_like(lon2d, bool))
    m.sigma = np.array([0.0, 10.0, 25.0, 50.0, 100.0, 250.0])
    m.depth = np.full(m.n_nodes, 300.0)
    set_grid_hash_cell(m, 0.3)
    return m


def _elem_field(m, seed, scale=0.3):
    from kelp.core.io import VelocityField
    rng = np.random.default_rng(seed)
    return VelocityField(u=scale * rng.standard_normal(m.n_elements), v=scale * rng.standard_normal(m.n_elements),
                         eta=np.zeros(m.n_nodes), w=None, uv_layout="element")


def _node_field(m, seed, scale=0.5):
    from kelp.core.io import VelocityField
    rng = np.random.default_rng(seed)
    return VelocityField(u=scale * rng.standard_normal(m.n_nodes), v=scale * rng.standard_normal(m.n_nodes),
                         eta=np.zeros(m.n_nodes), w=None, uv_layout="node")


def _seeds_highlat(n, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.5, 39.0, n), rng.uniform(80.2, 87.8, n)


_CMP_2D = ("lon", "lat", "status", "nearest_node", "u", "v", "failure_reason")


def _assert_same(got, ref, names=_CMP_2D):
    for name in names:
        a, b = getattr(got, name), getattr(ref, name)
        assert np.array_equal(a, b, equal_nan=True), name


class TestFused2DExactParity:
    """Every fused 2-D runner, with the guard on, equals its unfused twin bitwise: every
    evaluation site (start, stages, final) anchors through the in-kernel exact search, the same
    function the unfused path's batch nearest-node query runs."""
    DT = 3600.0

    def _state(self, m, n, seed):
        """Interior seeds + seeds hugging the west/north mesh edge (some exit or fail to
        interpolate this step) + 20 % non-ALIVE particles (OUTSIDE / NOT_RELEASED / BEACHED
        must be copied through untouched by both paths)."""
        rng = np.random.default_rng(seed)
        lon, lat = _seeds_highlat(n, seed)
        k = n // 5
        lon[:k] = rng.uniform(0.0005, 0.05, k); lat[k:2 * k] = rng.uniform(87.95, 87.9995, k)
        st = make_state(lon, lat).replace(failure_reason=np.zeros(n, dtype=np.int8))
        status = st.status.copy()
        dead = rng.random(n) < 0.2
        status[dead] = rng.choice([3, 4, 2], dead.sum())
        st = st.replace(status=status)
        return st.replace(nearest_node=find_nearest_nodes(m, lon, lat))

    def test_fixture_has_unprovable_anchors(self):
        """Sanity: with the guard OFF the fused kernel differs from the exact unfused path, so
        the parity tests below really exercise the heal."""
        import kelp.numerics.integrators as I
        m = _highlat_2d_mesh(); vel = _elem_field(m, 1)
        st = self._state(m, 600, 2)
        off = I._run_fused_2d(st, vel, m, self.DT, 1.0, 0, I._fused_rk2_step_2d)
        set_exact_nearest(m, True)
        try:
            st = st.replace(nearest_node=find_nearest_nodes(m, st.lon, st.lat))
            on = I._rk2_step_unfused(st, vel, m, self.DT, 1.0, 0, False, uv_interp="leastsq")
        finally:
            set_exact_nearest(m, False)
        assert not (np.array_equal(off.lon, on.lon) and np.array_equal(off.status, on.status))

    @pytest.mark.parametrize("scheme", ["euler", "rk2", "rk4"])
    @pytest.mark.parametrize("time_dir", [1.0, -1.0])
    def test_steady_leastsq(self, scheme, time_dir):
        import kelp.numerics.integrators as I
        m = _highlat_2d_mesh(); vel = _elem_field(m, 3)
        kern = {"euler": I._fused_euler_step_2d, "rk2": I._fused_rk2_step_2d, "rk4": I._fused_rk4_step_2d}[scheme]
        unf = {"euler": I._euler_step_unfused, "rk2": I._rk2_step_unfused, "rk4": I._rk4_step_unfused}[scheme]
        set_exact_nearest(m, True)
        try:
            st = self._state(m, 800, 4)
            got = I._run_fused_2d(st, vel, m, self.DT, time_dir, 0, kern)
            ref = unf(st, vel, m, self.DT, time_dir, 0, False, uv_interp="leastsq")
        finally:
            set_exact_nearest(m, False)
        assert (got.status == 3).sum() > (st.status == 3).sum(), "fixture must kill some particles"
        _assert_same(got, ref)

    @pytest.mark.parametrize("scheme", ["rk2", "rk4"])
    def test_antimeridian_mesh_leastsq(self, scheme):
        """A mesh straddling +-180 (stored in the [0, 360) branch by the hash): stage positions
        wrapped across the seam must anchor exactly on both sides."""
        import kelp.numerics.integrators as I
        from kelp.readers.structured import build_structured_mesh
        lon1d = ((np.arange(160.0, 200.0, 1.0) + 180.0) % 360.0) - 180.0
        lat1d = np.arange(78.0, 86.5, 0.5)
        lon2d, lat2d = np.meshgrid(lon1d, lat1d)
        m = build_structured_mesh(lon2d, lat2d, np.ones_like(lon2d, bool))
        set_grid_hash_cell(m, 0.3)
        vel = _elem_field(m, 70, scale=1.0)
        rng = np.random.default_rng(71); n = 800
        lon = ((rng.uniform(178.0, 182.0, n) + 180.0) % 360.0) - 180.0; lat = rng.uniform(78.2, 86.2, n)
        kern = I._fused_rk2_step_2d if scheme == "rk2" else I._fused_rk4_step_2d
        unf = I._rk2_step_unfused if scheme == "rk2" else I._rk4_step_unfused
        set_exact_nearest(m, True)
        try:
            st = make_state(lon, lat).replace(failure_reason=np.zeros(n, dtype=np.int8))
            st = st.replace(nearest_node=find_nearest_nodes(m, lon, lat))
            got = I._run_fused_2d(st, vel, m, 6 * self.DT, 1.0, 0, kern)
            ref = unf(st, vel, m, 6 * self.DT, 1.0, 0, False, uv_interp="leastsq")
        finally:
            set_exact_nearest(m, False)
        assert ((got.lon > 179.0) | (got.lon < -179.0)).any()
        _assert_same(got, ref)

    def test_euler_pair_lazy_field(self):
        from kelp.core.io import LazyLerpField
        import kelp.numerics.integrators as I
        m = _highlat_2d_mesh(); va, vb = _elem_field(m, 5), _elem_field(m, 6)
        lazy = LazyLerpField(va, vb, 0.3)
        set_exact_nearest(m, True)
        try:
            st = self._state(m, 800, 7)
            got = I._run_fused_euler_2d_pair(st, lazy, m, self.DT, 1.0, 0)
            ref = I._euler_step_unfused(st, lazy, m, self.DT, 1.0, 0, False, uv_interp="leastsq")
        finally:
            set_exact_nearest(m, False)
        _assert_same(got, ref)

    @pytest.mark.parametrize("scheme", ["rk2", "rk4"])
    @pytest.mark.parametrize("lazy", [False, True])
    def test_tinterp_leastsq(self, scheme, lazy):
        from kelp.core.io import LazyLerpField
        import kelp.numerics.integrators as I
        m = _highlat_2d_mesh(); va, vb = _elem_field(m, 8), _elem_field(m, 9)
        vel = LazyLerpField(va, vb, 0.25) if lazy else va
        kern = I._fused_rk2_step_2d_tinterp if scheme == "rk2" else I._fused_rk4_step_2d_tinterp
        cc = (0.0, 0.5) if scheme == "rk2" else (0.0, 0.5, 0.5, 1.0)
        unf = I._rk2_step_unfused if scheme == "rk2" else I._rk4_step_unfused
        set_exact_nearest(m, True)
        try:
            st = self._state(m, 800, 10)
            got = I._run_fused_2d_tinterp(st, vel, vb, 4 * self.DT, m, self.DT, 1.0, 0, kern, cc)
            ref = unf(st, vel, m, self.DT, 1.0, 0, False, vel_next=vb, dt_to_next=4 * self.DT, uv_interp="leastsq")
        finally:
            set_exact_nearest(m, False)
        _assert_same(got, ref)

    @pytest.mark.parametrize("scheme", ["rk2", "rk4"])
    def test_pchip_leastsq(self, scheme):
        import kelp.numerics.integrators as I
        m = _highlat_2d_mesh()
        snaps = [_elem_field(m, 20 + k) for k in range(4)]
        recon = (snaps, [0.0, 86400.0, 172800.0, 259200.0], 100000.0, 1.0)
        kern = I._fused_rk2_step_2d_pchip if scheme == "rk2" else I._fused_rk4_step_2d_pchip
        cc = (0.0, 0.5) if scheme == "rk2" else (0.0, 0.5, 0.5, 1.0)
        unf = I._rk2_step_unfused if scheme == "rk2" else I._rk4_step_unfused
        set_exact_nearest(m, True)
        try:
            st = self._state(m, 800, 11)
            got = I._run_fused_2d_pchip(st, recon, m, self.DT, 1.0, 0, kern, cc)
            ref = unf(st, snaps[1], m, self.DT, 1.0, 0, False, recon=recon, uv_interp="leastsq")
        finally:
            set_exact_nearest(m, False)
        _assert_same(got, ref)

    @pytest.mark.parametrize("scheme", ["rk2", "rk4"])
    def test_bary_tinterp_node_layout(self, coarse_mesh, scheme):
        """Node-layout barycentric twin on the coarse HEALPix mesh (polar rows + 3-deg cell)."""
        import kelp.numerics.integrators as I
        m = coarse_mesh; va, vb = _node_field(m, 30), _node_field(m, 31)
        kern = I._fused_rk2_step_2d_bary_tinterp if scheme == "rk2" else I._fused_rk4_step_2d_bary_tinterp
        cc = (0.0, 0.5) if scheme == "rk2" else (0.0, 0.5, 0.5, 1.0)
        unf = I._rk2_step_unfused if scheme == "rk2" else I._rk4_step_unfused
        rng = np.random.default_rng(32); n = 1500
        lon = rng.uniform(-179.0, 179.0, n); lat = np.degrees(np.arcsin(rng.uniform(-0.999, 0.999, n)))
        set_exact_nearest(m, True)
        try:
            st = make_state(lon, lat).replace(failure_reason=np.zeros(n, dtype=np.int8))
            st = st.replace(nearest_node=find_nearest_nodes(m, lon, lat))
            got = I._run_fused_2d_bary_tinterp(st, va, vb, 4 * self.DT, m, self.DT, 1.0, 0, kern, cc)
            ref = unf(st, va, m, self.DT, 1.0, 0, False, vel_next=vb, dt_to_next=4 * self.DT, uv_interp="barycentric")
        finally:
            set_exact_nearest(m, False)
        _assert_same(got, ref)


class TestFused3DEulerExact:
    @pytest.mark.parametrize("w_interp", ["nearest", "leastsq"])
    def test_runner_matches_unfused(self, w_interp):
        from kelp.core.io import VelocityField
        import kelp.numerics.integrators as I
        m = _highlat_2d_mesh(); K = m.sigma.size; N, E = m.n_nodes, m.n_elements
        rng = np.random.default_rng(50)
        vel = VelocityField(u=0.3 * rng.standard_normal((K - 1, E)), v=0.3 * rng.standard_normal((K - 1, E)),
                            eta=np.zeros(N), w=1e-3 * rng.standard_normal((K, N)), uv_layout="element")
        lon, lat = _seeds_highlat(800, 51)
        set_exact_nearest(m, True)
        try:
            st = make_state(lon, lat, rng.uniform(5.0, 200.0, 800)).replace(failure_reason=np.zeros(800, dtype=np.int8))
            st = st.replace(nearest_node=find_nearest_nodes(m, lon, lat))
            got = I._run_fused_euler_3d(st, vel, m, 3600.0, 1.0, w_interp=w_interp)
            ref = I._euler_step_unfused(st, vel, m, 3600.0, 1.0, 0, True, w_interp=w_interp, uv_interp="leastsq")
        finally:
            set_exact_nearest(m, False)
        _assert_same(got, ref, _CMP_2D + ("z", "w"))

    def test_guard_off_w_diagnostic_of_dying_particles_matches_unfused(self):
        """Guard-independent parity fix: the fused kernel used to leave vel_w = 0 for a particle
        whose u/v interpolation fails; the unfused path computes W for every alive particle."""
        from kelp.core.io import VelocityField
        import kelp.numerics.integrators as I
        m = _highlat_2d_mesh(); K = m.sigma.size; N, E = m.n_nodes, m.n_elements
        rng = np.random.default_rng(52)
        vel = VelocityField(u=0.3 * rng.standard_normal((K - 1, E)), v=0.3 * rng.standard_normal((K - 1, E)),
                            eta=np.zeros(N), w=1e-3 * rng.standard_normal((K, N)), uv_layout="element")
        n = 600
        lon = rng.uniform(0.0005, 0.05, n); lat = rng.uniform(80.2, 87.8, n)       # west edge: many fail
        st = make_state(lon, lat, rng.uniform(5.0, 200.0, n)).replace(failure_reason=np.zeros(n, dtype=np.int8))
        st = st.replace(nearest_node=find_nearest_nodes(m, lon, lat))
        got = I._run_fused_euler_3d(st, vel, m, 3600.0, 1.0, w_interp="nearest")
        ref = I._euler_step_unfused(st, vel, m, 3600.0, 1.0, 0, True, w_interp="nearest", uv_interp="leastsq")
        assert (got.failure_reason == 1).any(), "fixture must contain interpolation failures"
        assert np.array_equal(got.w, ref.w) and np.array_equal(got.status, ref.status)


class TestInKernelExactSearch:
    """`_grid_hash_nearest_exact`: the single-particle search the fused kernels use under the
    guard — 3x3 first, then a rectangle widened by the certificate's deficits (longitude only
    near a pole) until the best node is provably the nearest.  Must equal brute force."""

    @staticmethod
    def _exact(m, lon, lat):
        from kelp.core.mesh import _grid_hash_nearest_exact
        return np.array([_grid_hash_nearest_exact(
            float(lo), float(la), m._gh_offsets, m._gh_node_ids, m._gh_cell_size, m._gh_lon_min,
            m._gh_lat_min, m._gh_nx, m._gh_ny, m._gh_wrap_lon, m.nodes_lon, m.nodes_lat, m._node_xyz,
            int(m._gh_far_rows)) for lo, la in zip(lon, lat)], dtype=np.int32)

    def _check(self, m, lon, lat):
        """Within the row budget (nearest node closer than ~13 cells) the search is exact;
        farther queries are outside every element and get the plain widened anchor."""
        got = self._exact(m, lon, lat); ref = _brute_nearest(m, lon, lat)
        theta = np.degrees(2.0 * np.arcsin(0.5 * _chord(m, ref, lon, lat)))
        near = theta <= (m._gh_far_rows - 3) * m._gh_cell_size
        bad = near & (got != ref)
        if bad.any():   # allow exact chord ties only
            assert np.allclose(_chord(m, got[bad], lon[bad], lat[bad]),
                               _chord(m, ref[bad], lon[bad], lat[bad]), rtol=0, atol=1e-13)
        assert (got >= 0).all()                   # far queries still get a valid anchor
        return near

    def test_coarse_healpix_incl_poles_and_seam(self, coarse_mesh):
        lon, lat = _queries(4000, seed=80)
        lon = np.r_[lon, 0.0, 179.99, -179.99, 45.0, 10.0]; lat = np.r_[lat, 90.0, -90.0, 89.9, -89.9, 0.0]
        self._check(coarse_mesh, lon, lat)

    def test_highlat_structured_non_wrap(self):
        m = _highlat_2d_mesh(); rng = np.random.default_rng(81)
        lon = rng.uniform(-5.0, 45.0, 3000); lat = rng.uniform(78.0, 90.0, 3000)
        self._check(m, lon, lat)

    def test_wrap_grid_shortened_seam_cell(self):
        lon = np.r_[5.0, 356.9, np.arange(20.0, 351.0, 5.0)]
        lat = np.r_[0.0, 0.0, np.full(lon.size - 2, 40.0)]
        m = _mesh_from_points(lon, lat, 7.0)
        q_lon = np.array([0.1, 358.0, 2.0, 355.0]); q_lat = np.array([0.0, 0.0, 1.0, -1.0])
        self._check(m, q_lon, q_lat)
        assert self._exact(m, q_lon[:1], q_lat[:1])[0] == 1

    def test_padded_non_wrap_branch_overlap(self):
        lon = np.r_[np.arange(0.0, 351.0, 10.0), np.arange(0.0, 351.0, 10.0)]
        lat = np.r_[np.zeros(36), np.full(36, 10.0)]
        m = _mesh_from_points(lon, lat, 20.0)
        q_lon = np.array([-10.0, -4.0, 353.0, 175.0]); q_lat = np.array([0.0, 3.0, 5.0, 12.0])
        self._check(m, q_lon, q_lat)
        assert m.nodes_lon[self._exact(m, q_lon[:1], q_lat[:1])[0]] == 350.0

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_random_geometries_vs_brute_force(self, seed):
        rng = np.random.default_rng(seed)
        for _ in range(4):
            span = rng.choice([30.0, 120.0, 359.0, 360.0]); lat0 = rng.uniform(-89.0, 40.0)
            n = int(rng.integers(50, 600))
            lon = rng.uniform(0.0, span, n) % 360.0; lat = np.clip(lat0 + rng.uniform(0.0, 50.0, n), -89.9, 89.9)
            cell = float(rng.choice([0.5, 2.0, 7.0, 23.0]))
            m = _mesh_from_points(lon, lat, cell)
            q_lon = rng.uniform(-180.0, 180.0, 300); q_lat = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, 300)))
            near = self._check(m, q_lon, q_lat)
            assert near.sum() >= 30, near.sum()

    def test_far_outside_query_is_cheap_and_equals_the_widened_anchor(self):
        """A query farther than the row budget from any node (far outside the mesh, or deep
        inland) cannot be inside an element: the search must not scan the grid for it but
        return the plain widened anchor — the same node on the batch and the fused path."""
        from kelp.core.mesh import _grid_hash_nearest_widened, _grid_hash_nearest_exact
        m = _highlat_2d_mesh()
        args = (m._gh_offsets, m._gh_node_ids, m._gh_cell_size, m._gh_lon_min, m._gh_lat_min,
                m._gh_nx, m._gh_ny, m._gh_wrap_lon, m.nodes_lon, m.nodes_lat, m._node_xyz)
        far = [(150.0, -30.0), (-100.0, 84.0), (20.0, 20.0), (20.0, 60.0)]
        for lo, la in far:
            assert _grid_hash_nearest_exact(lo, la, *args, int(m._gh_far_rows)) == _grid_hash_nearest_widened(lo, la, *args)
        # (the widened anchor differs from the true nearest here, so the equality above
        # proves the shortcut was taken — no timing assertion: too flaky under load)
        assert _grid_hash_nearest_exact(150.0, -30.0, *args, int(m._gh_far_rows)) != _brute_nearest(m, np.array([150.0]), np.array([-30.0]))[0]
        set_exact_nearest(m, True)
        try:
            lon = np.array([lo for lo, _ in far]); lat = np.array([la for _, la in far])
            assert np.array_equal(find_nearest_nodes(m, lon, lat), self._exact(m, lon, lat))
        finally:
            set_exact_nearest(m, False)

    def test_matches_the_batch_exact_path(self, coarse_mesh):
        """Wiring check only: the batch path under the guard calls the same search per query
        (since review round 2) — the load-bearing parity test is the fused-vs-unfused lattice
        test in TestReviewRound2TiesAndBudget."""
        m = coarse_mesh; lon, lat = _queries(3000, seed=82)
        set_exact_nearest(m, True)
        try:
            batch = find_nearest_nodes(m, lon, lat)
        finally:
            set_exact_nearest(m, False)
        assert np.array_equal(self._exact(m, lon, lat), batch)


class TestReviewRound2TiesAndBudget:
    """Round 2 (review, 2026-09-12): exact-tie parity between the in-kernel search and the batch
    path, and frozen positions untouched by the diffusion kick."""

    @staticmethod
    def _lattice():
        from kelp.readers.structured import build_structured_mesh
        lon1d = np.arange(0.0, 20.0, 1.0); lat1d = np.arange(40.0, 60.0, 1.0)
        lon2d, lat2d = np.meshgrid(lon1d, lat1d)
        m = build_structured_mesh(lon2d, lat2d, np.ones_like(lon2d, bool))
        set_grid_hash_cell(m, 0.3)
        qlon = np.repeat(lon1d[:-1] + 0.5, lat1d.size); qlat = np.tile(lat1d, lon1d.size - 1)
        return m, qlon, qlat

    def test_exact_search_and_batch_path_break_ties_identically(self):
        """Particles at exact cell midpoints tie between two nodes bit for bit; the fused
        kernels' in-kernel search and `find_nearest_nodes` under the guard must pick the same."""
        m, qlon, qlat = self._lattice()
        set_exact_nearest(m, True)
        try:
            batch = find_nearest_nodes(m, qlon, qlat)
        finally:
            set_exact_nearest(m, False)
        ker = TestInKernelExactSearch._exact(m, qlon, qlat)
        assert np.array_equal(ker, batch)

    def test_lattice_midpoints_fused_equals_unfused(self):
        import kelp.numerics.integrators as I
        m, qlon, qlat = self._lattice()
        vel = _elem_field(m, 90)
        set_exact_nearest(m, True)
        try:
            st = make_state(qlon, qlat).replace(failure_reason=np.zeros(qlon.size, dtype=np.int8))
            st = st.replace(nearest_node=find_nearest_nodes(m, qlon, qlat))
            got = I._run_fused_2d(st, vel, m, 3600.0, 1.0, 0, I._fused_rk2_step_2d)
            ref = I._rk2_step_unfused(st, vel, m, 3600.0, 1.0, 0, False, uv_interp="leastsq")
        finally:
            set_exact_nearest(m, False)
        _assert_same(got, ref)

    def test_diffusion_kick_keeps_frozen_positions_verbatim(self):
        """A dead particle sitting exactly at lon 180.0 (the one value wrap_lon moves) must come
        out of the split diffusion kick untouched, like it does from the deterministic step."""
        from kelp.numerics.diffusion import DiffusionStepArgs
        import kelp.numerics.integrators as I
        m = _highlat_2d_mesh(); vel = _elem_field(m, 91)
        lon = np.array([5.0, 180.0, 6.0, 100.1, -179.9999, 7.3]); lat = np.array([84.0, 84.0, 84.5, 83.7, 85.2, 86.1])
        st = make_state(lon, lat).replace(failure_reason=np.zeros(6, dtype=np.int8))
        status = st.status.copy(); status[1] = 3; status[3] = 4; status[4] = 2; status[5] = 3
        st = st.replace(status=status, nearest_node=find_nearest_nodes(m, lon, lat))
        args = DiffusionStepArgs(scheme="uniform", kh_x=10.0, kh_y=10.0, seed=1, real_id=0,
                                 step_i=0, composition="split")
        out = I.euler_step(st, vel, m, 600.0, 1.0, uv_interp="leastsq", diffusion=args)
        frozen = [1, 3, 4, 5]
        assert np.array_equal(out.lon[frozen], lon[frozen]) and np.array_equal(out.lat[frozen], lat[frozen])
        assert np.array_equal(out.status[frozen], status[frozen])


class TestReviewRound3:
    """Round 3 (review, 2026-09-12): the search budget must come from the mesh."""

    @staticmethod
    def _multires_mesh(cell):
        """Coarse 1-deg Delaunay lattice with a fine 0.02-deg patch — the multi-resolution mesh
        `mesh.grid_hash_cell_size` exists for; the cell is pinned for the fine patch."""
        from scipy.spatial import Delaunay
        from kelp.core.mesh import Mesh, build_node_to_elems, build_search
        rng = np.random.default_rng(0)
        clo, cla = np.meshgrid(np.arange(0.0, 40.01, 1.0), np.arange(30.0, 50.01, 1.0))
        clo = clo.ravel() + rng.uniform(-0.1, 0.1, clo.size); cla = cla.ravel() + rng.uniform(-0.1, 0.1, cla.size)
        keep = ~((clo > 9.5) & (clo < 12.5) & (cla > 39.5) & (cla < 42.5))
        clo, cla = clo[keep], cla[keep]
        flo, fla = np.meshgrid(np.arange(10.0, 12.001, 0.02), np.arange(40.0, 42.001, 0.02))
        lon = np.r_[clo, flo.ravel()]; lat = np.r_[cla, fla.ravel()]
        tri = Delaunay(np.column_stack([lon, lat])).simplices.astype(np.int32)
        m = Mesh(name="mr", nodes_lon=lon, nodes_lat=lat, depth=np.full(lon.size, 100.0), elem_nodes=tri,
                 elem_center_lon=lon[tri].mean(1), elem_center_lat=lat[tri].mean(1),
                 node_to_elems=None, node_to_elems_count=None, sigma=np.array([]))
        m.node_to_elems, m.node_to_elems_count = build_node_to_elems(lon.size, tri)
        build_search(m); set_grid_hash_cell(m, cell)
        return m

    def test_budget_is_derived_from_the_longest_element_edge(self):
        m = self._multires_mesh(0.02)
        assert m._gh_far_rows >= 50 + 2                    # 1-deg coarse edges / 0.02-deg cell
        m2 = _highlat_2d_mesh()                             # 1 x 0.5 deg quads, cell 0.3
        assert 4 <= m2._gh_far_rows <= 8
        set_grid_hash_cell(m2, 5.0)                         # coarse cell: the floor
        assert m2._gh_far_rows == 4

    def test_multi_resolution_mesh_stays_exact_under_the_guard(self):
        """Queries inside COARSE elements (nearest vertex up to ~50 cells away): the exact
        search and the batch path must return the true nearest, and the particles must be
        contained (a fixed 16-cell budget gave 26 % false OUTSIDE here)."""
        from kelp.core.mesh import find_containing_elements
        m = self._multires_mesh(0.02)
        rng = np.random.default_rng(1)
        q_lon = rng.uniform(25.0, 38.0, 800); q_lat = rng.uniform(40.0, 42.0, 800)
        ref = _brute_nearest(m, q_lon, q_lat)
        got = TestInKernelExactSearch._exact(m, q_lon, q_lat)
        assert np.array_equal(got, ref)
        set_exact_nearest(m, True)
        try:
            batch = find_nearest_nodes(m, q_lon, q_lat)
            _, ok = find_containing_elements(m, q_lon, q_lat, batch)
        finally:
            set_exact_nearest(m, False)
        assert np.array_equal(batch, ref)
        _, ok_ref = find_containing_elements(m, q_lon, q_lat, ref)
        assert np.array_equal(ok, ok_ref) and ok.mean() > 0.95

    def test_land_hole_seeds_get_a_valid_anchor_on_both_paths(self):
        """Seeds inside a hole of a regional mesh (far from every node, inside the bbox): a valid
        anchor, identical on the batch and the in-kernel path, and no scan of the grid beyond
        the budget band (ring growth until the first hit at mid-latitude)."""
        from kelp.readers.structured import build_structured_mesh
        lon1d = np.arange(0.0, 20.0, 0.1); lat1d = np.arange(30.0, 50.0, 0.1)
        lon2d, lat2d = np.meshgrid(lon1d, lat1d)
        wet = ~((lon2d > 5.0) & (lon2d < 15.0) & (lat2d > 35.0) & (lat2d < 45.0))
        m = build_structured_mesh(lon2d, lat2d, wet)
        set_grid_hash_cell(m, 0.05)
        rng = np.random.default_rng(2)
        q_lon = rng.uniform(6.0, 14.0, 300); q_lat = rng.uniform(36.0, 44.0, 300)
        got = TestInKernelExactSearch._exact(m, q_lon, q_lat)
        assert (got >= 0).all()
        set_exact_nearest(m, True)
        try:
            assert np.array_equal(find_nearest_nodes(m, q_lon, q_lat), got)
        finally:
            set_exact_nearest(m, False)


class TestReviewRound4:
    def test_budget_counts_the_closing_edge_of_padded_triangles(self):
        """FESOM-C style 4-slot triangles (-1 padded): the third edge is (slot 2, slot 0)."""
        from kelp.core.mesh import Mesh, build_node_to_elems, build_search, _gh_far_rows_for
        lon = np.array([0.0, 1.5, 3.0]); lat = np.array([0.0, 0.3, 0.0])      # longest edge = slot 2 -> slot 0 (3 deg); the others 1.53
        en = np.array([[0, 1, 2, -1]], dtype=np.int32)
        m = Mesh(name="tri", nodes_lon=lon, nodes_lat=lat, depth=np.zeros(3), elem_nodes=en,
                 elem_center_lon=lon.mean(keepdims=True), elem_center_lat=lat.mean(keepdims=True),
                 node_to_elems=None, node_to_elems_count=None, sigma=np.array([]))
        m.node_to_elems, m.node_to_elems_count = build_node_to_elems(3, en)
        build_search(m)
        assert _gh_far_rows_for(m, 0.5) == int(np.ceil(3.0 / 0.5)) + 2

    def test_far_exit_respects_the_branch_cut_of_a_near_global_grid(self):
        """Non-wrap grid spanning 348 deg with elements ~9 deg: a query near the west edge has its
        nearest node across the branch cut at the east end; the clearance exit must not fire early."""
        lon = np.r_[np.arange(0.0, 349.0, 9.0), np.arange(0.0, 349.0, 9.0), np.arange(0.0, 349.0, 9.0)]
        lat = np.r_[np.full(39, -5.0), np.full(39, 0.0), np.full(39, 5.0)]
        m = _mesh_from_points(lon, lat, 2.0)
        assert not m._gh_wrap_lon
        q_lon = np.array([-3.0, -6.0, -8.0, 351.0]); q_lat = np.array([0.0, 2.0, -3.0, 1.0])
        got = TestInKernelExactSearch._exact(m, q_lon, q_lat); ref = _brute_nearest(m, q_lon, q_lat)
        assert np.array_equal(got, ref)

    def test_fused_3d_rk2_uses_the_in_kernel_search_under_the_guard(self, coarse_mesh, monkeypatch):
        """Round 4 (review): the 3-D RK2 barycentric kernels used the flag/runner-heal rounds under
        the guard and were slower than the unfused path — now they anchor stages in-kernel like
        every other fused kernel: one kernel invocation, no heal round, same result as unfused."""
        from kelp.core.io import VelocityField
        import kelp.numerics.integrators as I
        m = coarse_mesh; K = m.sigma.size; N = m.n_nodes
        rng = np.random.default_rng(140)
        vel = VelocityField(u=0.5 * rng.standard_normal((K - 1, N)), v=0.5 * rng.standard_normal((K - 1, N)),
                            eta=np.zeros(N), w=1e-4 * rng.standard_normal((K, N)), uv_layout="node")
        n = 1500
        lon = rng.uniform(-179.0, 179.0, n); lat = np.degrees(np.arcsin(rng.uniform(-0.999, 0.999, n)))
        st = make_state(lon, lat, rng.uniform(10.0, 2000.0, n)).replace(failure_reason=np.zeros(n, dtype=np.int8))
        calls = {"n": 0}; orig = I._run_fused_3d_with_heal

        def spy(run_kernel, state, mesh, P):
            def rk(*a, **k):
                calls["n"] += 1
                return run_kernel(*a, **k)
            return orig(rk, state, mesh, P)
        monkeypatch.setattr(I, "_run_fused_3d_with_heal", spy)
        set_exact_nearest(m, True)
        try:
            st = st.replace(nearest_node=find_nearest_nodes(m, lon, lat))
            got = I._run_fused_rk2_3d_bary_tinterp(st, vel, None, None, m, 5400.0, 1.0, I.W_BARYCENTRIC)
            ref = I._rk2_step_unfused(st, vel, m, 5400.0, 1.0, 0, True, w_interp="barycentric", uv_interp="barycentric")
        finally:
            set_exact_nearest(m, False)
        assert calls["n"] == 1, calls
        _assert_same(got, ref, _CMP_2D + ("z", "w"))

    def test_longest_edge_is_cached_across_hash_rebuilds(self, monkeypatch):
        """Round 4 (review): the longest element edge is cell-independent — compute it once, not on
        every build_grid_hash (11 s / 3 GB per rebuild on the 23 M-element global mesh)."""
        import kelp.core.mesh as M
        m = _highlat_2d_mesh()
        assert m._gh_max_edge_deg is not None
        edge0 = m._gh_max_edge_deg; rows0 = m._gh_far_rows
        calls = {"n": 0}; orig = M._gh_max_edge_for; orig_rows = M._gh_far_rows_for

        def spy(mesh):
            calls["n"] += 1
            return orig(mesh)
        monkeypatch.setattr(M, "_gh_max_edge_for", spy)
        set_grid_hash_cell(m, 0.1)
        M.build_search(m)
        assert calls["n"] == 0 and m._gh_max_edge_deg == edge0
        assert m._gh_far_rows == orig_rows(m, 0.1) and m._gh_far_rows > rows0

    def test_budget_cap_warns(self, monkeypatch):
        """A longest edge beyond `_GH_FAR_ROWS_MAX` cells voids the in-element guarantee: warn."""
        import warnings
        import kelp.core.mesh as M
        m = _highlat_2d_mesh()
        monkeypatch.setattr(M, "_GH_FAR_ROWS_MAX", 5)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            set_grid_hash_cell(m, 0.1)                      # 1-deg edges / 0.1 = 12 rows > cap
        assert m._gh_far_rows == 5
        assert any("far_rows" in str(w.message) or "budget" in str(w.message) for w in rec)

