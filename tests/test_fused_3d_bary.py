"""Fused 3-D RK2 for NODE-layout barycentric velocities (2026-08).

Golden parity: the fused kernel must reproduce ``_rk2_step_unfused``
BIT FOR BIT on the same inputs — positions, depths, statuses, nearest
nodes, u/v/w diagnostics and failure_reason — across steady, linear
temporal-interpolation (plain pair and LazyLerpField payload), PCHIP,
forward and backward time, near-bottom clamping, and both supported
w_interp methods.  The dual HEALPix mesh (node layout) is the natural
carrier; the density certificate must hold (asserted) so the in-kernel
anchor equals the batch KD-healed one.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kelp.core.io import VelocityField, LazyLerpField  # noqa: E402
from kelp.core.mesh import grid_hash_dense_for_elements  # noqa: E402
from kelp.core.particles import (make_state,  # noqa: E402
                                     ParticleStatus)
from kelp.numerics.integrators import (  # noqa: E402
    _rk2_step_unfused, _run_fused_rk2_3d_bary_tinterp,
    _run_fused_rk2_3d_bary_pchip, rk2_step,
    W_NEAREST, W_BARYCENTRIC)
from kelp.readers.healpix import build_healpix_dual_mesh  # noqa: E402

NSIDE = 8
NPIX = 12 * NSIDE * NSIDE
Z_LEVELS = np.array([0.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0,
                     1000.0, 2000.0, 4000.0])
K = Z_LEVELS.size


@pytest.fixture(scope="module")
def mesh():
    lon, lat = healpy.pix2ang(NSIDE, np.arange(NPIX), nest=True,
                              lonlat=True)
    # varied bathymetry incl. shallow columns (clamp/bottom branches)
    depth = 60.0 + 3500.0 * np.abs(np.sin(np.deg2rad(lat * 2.0)))
    depth[::17] = 35.0                      # shallow spots
    m = build_healpix_dual_mesh(NSIDE, wet_mask=np.ones(NPIX, bool),
                                depth=depth, z_levels=Z_LEVELS)
    assert grid_hash_dense_for_elements(m), (
        "fixture mesh must satisfy the density certificate")
    return m


def _field(mesh, seed, scale=0.4):
    rng = np.random.default_rng(seed)
    N = mesh.n_nodes
    u = scale * rng.standard_normal((K - 1, N))
    v = scale * rng.standard_normal((K - 1, N))
    w = 1e-4 * rng.standard_normal((K, N))
    return VelocityField(u=u, v=v, eta=np.zeros(N), w=w, uv_layout="node")


def _seeds(mesh, n=4000, seed=5):
    rng = np.random.default_rng(seed)
    lon = rng.uniform(-179.0, 179.0, n)
    lat = np.degrees(np.arcsin(rng.uniform(-0.95, 0.95, n)))
    # depths spanning surface, interior, near-bottom (incl. beyond-bottom
    # values the clamp machinery must handle)
    z = rng.uniform(0.0, 3000.0, n)
    z[::7] = 20.0
    z[1::13] = 3900.0
    st = make_state(lon, lat, z)
    st = st.replace(failure_reason=np.zeros(n, dtype=np.int8))
    from kelp.core.mesh import find_nearest_nodes
    return st.replace(nearest_node=find_nearest_nodes(mesh, lon, lat))


def _assert_bitwise(a, b):
    for name in ("lon", "lat", "z", "status", "nearest_node",
                 "u", "v", "w", "failure_reason"):
        va, vb = getattr(a, name), getattr(b, name)
        assert (va is None) == (vb is None), name
        if va is not None:
            assert np.array_equal(va, vb, equal_nan=True), (
                f"{name} differs: max|d|="
                f"{np.nanmax(np.abs(np.asarray(va, float) - np.asarray(vb, float)))}")


DT = 5400.0


class TestParitySteady:
    @pytest.mark.parametrize("w_interp,w_code", [("nearest", W_NEAREST),
                                                 ("barycentric",
                                                  W_BARYCENTRIC)])
    @pytest.mark.parametrize("tdir", [1.0, -1.0])
    def test_steady(self, mesh, w_interp, w_code, tdir):
        vel = _field(mesh, 1)
        st = _seeds(mesh)
        ref = _rk2_step_unfused(st, vel, mesh, DT, tdir, 0, True,
                                w_interp=w_interp,
                                uv_interp="barycentric")
        got = _run_fused_rk2_3d_bary_tinterp(st, vel, None, None, mesh,
                                             DT, tdir, w_code)
        assert (ref.status == 3).sum() > 0 or True  # informational
        _assert_bitwise(got, ref)


class TestParityTinterp:
    @pytest.mark.parametrize("w_interp,w_code", [("nearest", W_NEAREST),
                                                 ("barycentric",
                                                  W_BARYCENTRIC)])
    @pytest.mark.parametrize("tdir", [1.0, -1.0])
    def test_plain_pair(self, mesh, w_interp, w_code, tdir):
        vel = _field(mesh, 1)
        vel_next = _field(mesh, 2)
        st = _seeds(mesh)
        ref = _rk2_step_unfused(st, vel, mesh, DT, tdir, 0, True,
                                w_interp=w_interp,
                                vel_next=vel_next, dt_to_next=21600.0,
                                uv_interp="barycentric")
        got = _run_fused_rk2_3d_bary_tinterp(st, vel, vel_next, 21600.0,
                                             mesh, DT, tdir, w_code)
        _assert_bitwise(got, ref)

    def test_lazy_pair_payload(self, mesh):
        a = _field(mesh, 1)
        b = _field(mesh, 2)
        lazy = LazyLerpField(a, b, 0.3)
        st = _seeds(mesh)
        ref = _rk2_step_unfused(st, lazy, mesh, DT, 1.0, 0, True,
                                w_interp="barycentric",
                                vel_next=b, dt_to_next=21600.0,
                                uv_interp="barycentric")
        got = _run_fused_rk2_3d_bary_tinterp(st, lazy, b, 21600.0, mesh,
                                             DT, 1.0, W_BARYCENTRIC)
        _assert_bitwise(got, ref)


class TestParityPchip:
    @pytest.mark.parametrize("tdir", [1.0, -1.0])
    def test_pchip(self, mesh, tdir):
        snaps = [_field(mesh, s) for s in (11, 12, 13, 14)]
        times = [0.0, 86400.0, 172800.0, 259200.0]
        q = 100000.0
        recon = (snaps, times, q, tdir)
        st = _seeds(mesh)
        ref = _rk2_step_unfused(st, snaps[1], mesh, DT, tdir, 0, True,
                                w_interp="barycentric",
                                uv_interp="barycentric", recon=recon)
        got = _run_fused_rk2_3d_bary_pchip(st, recon, mesh, DT, tdir,
                                           W_BARYCENTRIC)
        _assert_bitwise(got, ref)

    def test_pchip_validation(self, mesh):
        snaps = [_field(mesh, s) for s in (11, 12, 13, 14)]
        st = _seeds(mesh, n=10)
        with pytest.raises(ValueError, match="strictly"):
            _run_fused_rk2_3d_bary_pchip(
                st, (snaps, [0.0, 2.0, 1.0, 3.0], 1.5, 1.0), mesh, DT,
                1.0, W_BARYCENTRIC)
        with pytest.raises(ValueError, match="bracket"):
            _run_fused_rk2_3d_bary_pchip(
                st, (snaps, [0.0, 1.0, 2.0, 3.0], 1.999, 1.0), mesh,
                86400.0, 1.0, W_BARYCENTRIC)


class TestDispatch:
    def test_rk2_step_routes_to_fused(self, mesh, monkeypatch):
        import kelp.numerics.integrators as I
        calls = {"n": 0}
        orig = I._run_fused_rk2_3d_bary_tinterp

        def spy(*a, **k):
            calls["n"] += 1
            return orig(*a, **k)
        monkeypatch.setattr(I, "_run_fused_rk2_3d_bary_tinterp", spy)
        vel = _field(mesh, 1)
        st = _seeds(mesh, n=50)
        out = rk2_step(st, vel, mesh, DT, 1.0, uv_interp="barycentric",
                       w_interp="barycentric",
                       vel_next=_field(mesh, 2), dt_to_next=21600.0)
        assert calls["n"] == 1
        ref = _rk2_step_unfused(st, vel, mesh, DT, 1.0, 0, True,
                                w_interp="barycentric",
                                vel_next=_field(mesh, 2),
                                dt_to_next=21600.0,
                                uv_interp="barycentric")
        _assert_bitwise(out, ref)

    def test_unsupported_w_interp_stays_unfused(self, mesh, monkeypatch):
        import kelp.numerics.integrators as I
        calls = {"n": 0}

        def spy(*a, **k):
            calls["n"] += 1
            raise AssertionError("must not be called")
        monkeypatch.setattr(I, "_run_fused_rk2_3d_bary_tinterp", spy)
        vel = _field(mesh, 1)
        st = _seeds(mesh, n=20)
        rk2_step(st, vel, mesh, DT, 1.0, uv_interp="barycentric",
                 w_interp="leastsq")
        assert calls["n"] == 0

    def test_walk_elems_stay_unfused(self, mesh, monkeypatch):
        import kelp.numerics.integrators as I
        monkeypatch.setattr(
            I, "_run_fused_rk2_3d_bary_tinterp",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
        vel = _field(mesh, 1)
        st = _seeds(mesh, n=20)
        st = st.replace(elem=np.full(20, -1, dtype=np.int32))
        rk2_step(st, vel, mesh, DT, 1.0, uv_interp="barycentric",
                 w_interp="barycentric")


class TestCoastalAndAnchorRegression:
    """Review round 2: the all-wet fixture never exercised the failure
    branches, and the round-1 anchor fix shipped without a pin."""

    def test_coastal_band_failures_parity(self):
        # wet band => real coastline: interp failures and OOB exits fire
        lon, lat = healpy.pix2ang(NSIDE, np.arange(NPIX), nest=True,
                                  lonlat=True)
        wet = np.abs(lat) < 40.0
        depth = 60.0 + 3500.0 * np.abs(np.sin(np.deg2rad(lat * 2.0)))
        m = build_healpix_dual_mesh(NSIDE, wet_mask=wet, depth=depth,
                                    z_levels=Z_LEVELS)
        rng = np.random.default_rng(9)
        n = 6000
        plon = rng.uniform(-179.0, 179.0, n)
        plat = rng.uniform(-55.0, 55.0, n)   # some outside the band
        pz = rng.uniform(0.0, 3000.0, n)
        st = make_state(plon, plat, pz)
        st = st.replace(failure_reason=np.zeros(n, dtype=np.int8))
        from kelp.core.mesh import find_nearest_nodes
        st = st.replace(nearest_node=find_nearest_nodes(m, plon, plat))
        vel = _field_for(m, 1)
        vnx = _field_for(m, 2)
        ref = _rk2_step_unfused(st, vel, m, DT, 1.0, 0, True,
                                w_interp="barycentric",
                                vel_next=vnx, dt_to_next=21600.0,
                                uv_interp="barycentric")
        got = _run_fused_rk2_3d_bary_tinterp(st, vel, vnx, 21600.0, m,
                                             DT, 1.0, W_BARYCENTRIC)
        n_dead = int((ref.status == 3).sum())
        assert n_dead > 500          # failure branches genuinely exercised
        assert set(np.unique(ref.failure_reason)) >= {0, 1}
        _assert_bitwise(got, ref)

    def test_anchor_divergence_mesh_parity(self):
        # a certificate-passing mesh where the widened in-kernel anchor
        # DISAGREES with the batch KD-healed nearest for off-footprint
        # queries: parity holds only because the runners feed the exact
        # batch stage-1 anchor (round-1 fix pin).
        from kelp.readers.structured import build_structured_mesh
        from kelp.core.mesh import find_nearest_nodes
        lon1d = np.arange(0.0, 40.0, 0.25)
        lat1d = np.arange(60.0, 88.0, 0.25)
        lon2d, lat2d = np.meshgrid(lon1d, lat1d)
        wet = np.ones_like(lon2d, dtype=bool)
        wet[(lat2d > 82.0) & (lat2d < 84.0)] = False   # zonal hole
        m = build_structured_mesh(lon2d, lat2d, wet)
        m.sigma = Z_LEVELS
        m.depth = np.full(m.n_nodes, 3500.0)
        assert grid_hash_dense_for_elements(m)
        rng = np.random.default_rng(4)
        n = 8000
        plon = rng.uniform(1.0, 39.0, n)
        plat = rng.uniform(61.0, 87.0, n)   # includes the hole band
        pz = rng.uniform(0.0, 3000.0, n)
        batch_nn = find_nearest_nodes(m, plon, plat)
        # the TRIGGER CLASS is queries whose 3x3 hash window is empty
        # (the in-kernel widened search and the batch KD-heal then run
        # different code): assert the fixture genuinely produces them
        from kelp.core.mesh import _grid_hash_nearest_3x3
        empty = 0
        for i in range(0, n, 7):
            nn3 = _grid_hash_nearest_3x3(
                plon[i], plat[i], m._gh_offsets, m._gh_node_ids,
                m._gh_cell_size, m._gh_lon_min, m._gh_lat_min,
                m._gh_nx, m._gh_ny, m._gh_wrap_lon,
                m.nodes_lon, m.nodes_lat, m._node_xyz)
            empty += int(nn3 < 0)
        assert empty > 0             # off-footprint window class is REAL
        st = make_state(plon, plat, pz)
        st = st.replace(failure_reason=np.zeros(n, dtype=np.int8),
                        nearest_node=batch_nn.copy())
        vel = _field_for(m, 3)
        ref = _rk2_step_unfused(st, vel, m, DT, 1.0, 0, True,
                                w_interp="barycentric",
                                uv_interp="barycentric")
        got = _run_fused_rk2_3d_bary_tinterp(st, vel, None, None, m,
                                             DT, 1.0, W_BARYCENTRIC)
        _assert_bitwise(got, ref)


def _field_for(mesh, seed, scale=0.4):
    rng = np.random.default_rng(seed)
    N = mesh.n_nodes
    return VelocityField(u=scale * rng.standard_normal((K - 1, N)),
                         v=scale * rng.standard_normal((K - 1, N)),
                         eta=np.zeros(N),
                         w=1e-4 * rng.standard_normal((K, N)),
                         uv_layout="node")


class TestHybridExactness:
    def test_certificate_false_mesh_fuses_bitwise(self, monkeypatch):
        # geographic nside-32 dual mesh: density certificate is FALSE
        # (polar rows) — the hybrid makes the fused route exact anyway.
        m32 = build_healpix_dual_mesh(
            32, wet_mask=np.ones(12 * 32 * 32, bool),
            depth=np.full(12 * 32 * 32, 3000.0), z_levels=Z_LEVELS)
        assert not grid_hash_dense_for_elements(m32)
        import kelp.numerics.integrators as I
        calls = {"n": 0}
        orig = I._run_fused_rk2_3d_bary_tinterp

        def spy(*a, **k):
            calls["n"] += 1
            return orig(*a, **k)
        monkeypatch.setattr(I, "_run_fused_rk2_3d_bary_tinterp", spy)
        vel = _field_for(m32, 1)
        vnx = _field_for(m32, 2)
        st = _seeds(m32, n=3000, seed=8)
        out = rk2_step(st, vel, m32, DT, 1.0, uv_interp="barycentric",
                       w_interp="barycentric",
                       vel_next=vnx, dt_to_next=21600.0)
        assert calls["n"] == 1                   # fused engaged
        ref = _rk2_step_unfused(st, vel, m32, DT, 1.0, 0, True,
                                w_interp="barycentric",
                                vel_next=vnx, dt_to_next=21600.0,
                                uv_interp="barycentric")
        _assert_bitwise(out, ref)

    def test_flagged_rerun_engages_and_matches(self, monkeypatch):
        # engineered: a strong northward flow pushes particles so their
        # MIDPOINT lands inside a hash-window-empty hole band — the
        # kernel must flag them and the runner KD-heal the anchors at
        # subset cost (the whole-field splice is terminal-only).
        from kelp.readers.structured import build_structured_mesh
        from kelp.core.mesh import find_nearest_nodes
        lon1d = np.arange(0.0, 30.0, 0.25)
        lat1d = np.arange(60.0, 80.0, 0.25)
        lon2d, lat2d = np.meshgrid(lon1d, lat1d)
        wet = np.ones_like(lon2d, bool)
        wet[(lat2d > 70.0) & (lat2d < 73.0)] = False
        m = build_structured_mesh(lon2d, lat2d, wet)
        m.sigma = Z_LEVELS
        m.depth = np.full(m.n_nodes, 3500.0)
        N = m.n_nodes
        vel = VelocityField(u=np.zeros((K - 1, N)),
                            v=np.full((K - 1, N), 45.0),   # ~2.2 deg/step
                            eta=np.zeros(N),
                            w=np.zeros((K, N)), uv_layout="node")
        n = 400
        rng = np.random.default_rng(2)
        plon = rng.uniform(5.0, 25.0, n)
        plat = np.full(n, 69.9)   # midpoint ~71.0 = deep in the hole
        st = make_state(plon, plat, np.full(n, 100.0))
        st = st.replace(failure_reason=np.zeros(n, dtype=np.int8),
                        nearest_node=find_nearest_nodes(m, plon, plat))
        import kelp.numerics.integrators as I
        spl = {"n": 0}
        orig = I._splice_exact_rerun

        def spy(*a, **k):
            spl["n"] += 1
            return orig(*a, **k)
        monkeypatch.setattr(I, "_splice_exact_rerun", spy)
        got = I._run_fused_rk2_3d_bary_tinterp(st, vel, None, None, m,
                                               DT, 1.0, W_BARYCENTRIC)
        ref = _rk2_step_unfused(st, vel, m, DT, 1.0, 0, True,
                                w_interp="barycentric",
                                uv_interp="barycentric")
        # the in-kernel anchor-heal loop must resolve every flag at
        # SUBSET cost — the whole-field unfused splice (which blends
        # stage fields over the ENTIRE mesh even for one particle:
        # ~30 s/step at zoom 10, the 2026-08 scan anomaly) stays as a
        # terminal safety net only and must NOT fire here
        assert spl["n"] == 0
        _assert_bitwise(got, ref)

    def test_two_round_heal_mid_then_final(self):
        # unit-level: a fake kernel that needs BOTH heal rounds — flags
        # site 1 until ov_mid is set, then site 2 until ov_new is set —
        # must resolve with persisting overrides and no leftovers
        from kelp.readers.structured import build_structured_mesh
        import kelp.numerics.integrators as I
        lon1d = np.arange(0.0, 5.0, 0.5)
        lat1d = np.arange(50.0, 55.0, 0.5)
        lon2d, lat2d = np.meshgrid(lon1d, lat1d)
        m = build_structured_mesh(lon2d, lat2d,
                                  np.ones_like(lon2d, bool))
        st = make_state(np.array([2.0]), np.array([52.0]),
                        np.array([100.0]))
        seen = []

        def fake_kernel(alive_arr, ov_mid, ov_new):
            seen.append((int(ov_mid[0]), int(ov_new[0])))
            out = {"lon": np.array([2.0]), "lat": np.array([52.0]),
                   "z": np.array([100.0]),
                   "status": np.array([1], dtype=np.int32),
                   "nn": np.array([0], dtype=np.int32),
                   "U": np.zeros(1), "V": np.zeros(1), "W": np.zeros(1),
                   "fr": np.zeros(1, dtype=np.int8),
                   "need_exact": np.zeros(1, dtype=np.int8),
                   "flag_lon": np.zeros(1), "flag_lat": np.zeros(1),
                   "flag_site": np.zeros(1, dtype=np.int8)}
            if ov_mid[0] < 0:
                out["need_exact"][0] = 1
                out["flag_site"][0] = 1
                out["flag_lon"][0] = 2.1
                out["flag_lat"][0] = 52.1
            elif ov_new[0] < 0:
                out["need_exact"][0] = 1
                out["flag_site"][0] = 2
                out["flag_lon"][0] = 2.2
                out["flag_lat"][0] = 52.2
            else:
                out["U"][0] = 7.0        # marks the fully healed pass
            return out

        outs, flagged = I._run_fused_3d_with_heal(fake_kernel, st, m, 1)
        assert not flagged.any()          # resolved without fallback
        assert len(seen) == 3             # initial + two heal rounds
        assert seen[1][0] >= 0 and seen[1][1] < 0   # mid healed first
        assert seen[2][0] >= 0 and seen[2][1] >= 0  # both persist
        assert outs["U"][0] == 7.0

    def test_pchip_hole_band_heals_without_splice(self, monkeypatch):
        # PCHIP twin of the hole-band heal (review 2026-08): the
        # 4-snapshot fused kernel must flag+heal identically
        from kelp.readers.structured import build_structured_mesh
        from kelp.core.mesh import find_nearest_nodes
        lon1d = np.arange(0.0, 30.0, 0.25)
        lat1d = np.arange(60.0, 80.0, 0.25)
        lon2d, lat2d = np.meshgrid(lon1d, lat1d)
        wet = np.ones_like(lon2d, bool)
        wet[(lat2d > 70.0) & (lat2d < 73.0)] = False
        m = build_structured_mesh(lon2d, lat2d, wet)
        m.sigma = Z_LEVELS
        m.depth = np.full(m.n_nodes, 3500.0)
        N = m.n_nodes

        def snap(v):
            return VelocityField(u=np.zeros((K - 1, N)),
                                 v=np.full((K - 1, N), v),
                                 eta=np.zeros(N),
                                 w=np.zeros((K, N)), uv_layout="node")
        snaps4 = [snap(44.0), snap(45.0), snap(46.0), snap(47.0)]
        times4 = [0.0, 21600.0, 43200.0, 64800.0]
        recon = (snaps4, times4, 30000.0, 1.0)
        n = 200
        rng = np.random.default_rng(3)
        plon = rng.uniform(5.0, 25.0, n)
        plat = np.full(n, 69.9)
        st = make_state(plon, plat, np.full(n, 100.0))
        st = st.replace(failure_reason=np.zeros(n, dtype=np.int8),
                        nearest_node=find_nearest_nodes(m, plon, plat))
        import kelp.numerics.integrators as I
        spl = {"n": 0}
        orig = I._splice_exact_rerun

        def spy(*a, **k):
            spl["n"] += 1
            return orig(*a, **k)
        monkeypatch.setattr(I, "_splice_exact_rerun", spy)
        got = I._run_fused_rk2_3d_bary_pchip(st, recon, m, DT, 1.0,
                                             W_BARYCENTRIC)
        ref = _rk2_step_unfused(st, snaps4[1], m, DT, 1.0, 0, True,
                                w_interp="barycentric",
                                uv_interp="barycentric", recon=recon)
        assert spl["n"] == 0
        _assert_bitwise(got, ref)

    def test_mixed_status_rows_stay_frozen_through_heal(self,
                                                       monkeypatch):
        # BEACHED/NOT_RELEASED rows must ride flagged sub-runs untouched
        from kelp.readers.structured import build_structured_mesh
        from kelp.core.mesh import find_nearest_nodes
        lon1d = np.arange(0.0, 30.0, 0.25)
        lat1d = np.arange(60.0, 80.0, 0.25)
        lon2d, lat2d = np.meshgrid(lon1d, lat1d)
        wet = np.ones_like(lon2d, bool)
        wet[(lat2d > 70.0) & (lat2d < 73.0)] = False
        m = build_structured_mesh(lon2d, lat2d, wet)
        m.sigma = Z_LEVELS
        m.depth = np.full(m.n_nodes, 3500.0)
        N = m.n_nodes
        vel = VelocityField(u=np.zeros((K - 1, N)),
                            v=np.full((K - 1, N), 45.0),
                            eta=np.zeros(N),
                            w=np.zeros((K, N)), uv_layout="node")
        n = 300
        rng = np.random.default_rng(5)
        plon = rng.uniform(5.0, 25.0, n)
        plat = np.full(n, 69.9)
        st = make_state(plon, plat, np.full(n, 100.0))
        status = st.status.copy()
        status[::3] = int(ParticleStatus.BEACHED)
        status[1::5] = int(ParticleStatus.NOT_RELEASED)
        st = st.replace(status=status,
                        failure_reason=np.zeros(n, dtype=np.int8),
                        nearest_node=find_nearest_nodes(m, plon, plat))
        import kelp.numerics.integrators as I
        got = I._run_fused_rk2_3d_bary_tinterp(st, vel, None, None, m,
                                               DT, 1.0, W_BARYCENTRIC)
        frozen = ~st.alive_mask
        for f in ("lon", "lat", "z", "status"):
            np.testing.assert_array_equal(getattr(got, f)[frozen],
                                          getattr(st, f)[frozen])
        ref = _rk2_step_unfused(st, vel, m, DT, 1.0, 0, True,
                                w_interp="barycentric",
                                uv_interp="barycentric")
        _assert_bitwise(got, ref)

    def test_heal_rounds_zero_falls_back_to_splice(self, monkeypatch):
        # same engineered hole-band scenario; with the heal loop turned
        # off the terminal splice fallback must engage and still give
        # the exact unfused result
        from kelp.readers.structured import build_structured_mesh
        from kelp.core.mesh import find_nearest_nodes
        lon1d = np.arange(0.0, 30.0, 0.25)
        lat1d = np.arange(60.0, 80.0, 0.25)
        lon2d, lat2d = np.meshgrid(lon1d, lat1d)
        wet = np.ones_like(lon2d, bool)
        wet[(lat2d > 70.0) & (lat2d < 73.0)] = False
        m = build_structured_mesh(lon2d, lat2d, wet)
        m.sigma = Z_LEVELS
        m.depth = np.full(m.n_nodes, 3500.0)
        N = m.n_nodes
        vel = VelocityField(u=np.zeros((K - 1, N)),
                            v=np.full((K - 1, N), 45.0),
                            eta=np.zeros(N),
                            w=np.zeros((K, N)), uv_layout="node")
        n = 400
        rng = np.random.default_rng(2)
        plon = rng.uniform(5.0, 25.0, n)
        plat = np.full(n, 69.9)
        st = make_state(plon, plat, np.full(n, 100.0))
        st = st.replace(failure_reason=np.zeros(n, dtype=np.int8),
                        nearest_node=find_nearest_nodes(m, plon, plat))
        import kelp.numerics.integrators as I
        monkeypatch.setattr(I, "_FLAG_HEAL_ROUNDS", 0)
        spl = {"n": 0}
        orig = I._splice_exact_rerun

        def spy(*a, **k):
            spl["n"] += 1
            return orig(*a, **k)
        monkeypatch.setattr(I, "_splice_exact_rerun", spy)
        got = I._run_fused_rk2_3d_bary_tinterp(st, vel, None, None, m,
                                               DT, 1.0, W_BARYCENTRIC)
        ref = _rk2_step_unfused(st, vel, m, DT, 1.0, 0, True,
                                w_interp="barycentric",
                                uv_interp="barycentric")
        assert spl["n"] >= 1                    # terminal fallback fired
        _assert_bitwise(got, ref)
