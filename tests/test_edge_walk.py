"""Edge-walk element tracking (numerics.elem_search: walk).

Search for the containing element only at initialization; afterwards cross
edges incrementally: the most-negative signed-area (barycentric) coordinate
identifies the exit edge, elem_neighbors gives the element on the other
side.  Any non-contained walk outcome falls back to the existing global
machinery, so containment VERDICTS are identical to elem_search: local.
"""
import numpy as np
import pytest

from kelp.core.mesh import (
    Mesh, build_node_to_elems, build_search, find_nearest_nodes,
    find_containing_elements, build_elem_neighbors,
    _walk_containing_element, find_containing_elements_walk,
)


def _mesh_from(lon, lat, elem_nodes):
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    en = np.asarray(elem_nodes, np.int32)
    n2e, n2e_cnt = build_node_to_elems(lon.size, en)
    cl = np.array([lon[e[e >= 0]].mean() for e in en])
    ca = np.array([lat[e[e >= 0]].mean() for e in en])
    m = Mesh(name="test", nodes_lon=lon, nodes_lat=lat,
             depth=np.full(lon.size, 10.0),
             elem_nodes=en, elem_center_lon=cl, elem_center_lat=ca,
             node_to_elems=n2e, node_to_elems_count=n2e_cnt,
             sigma=np.empty(0), vertical_coord="zlevel")
    build_search(m)
    return m


def _tri_grid(ng=6, lon0=0.0, lat0=40.0, step=1.0, cw=False):
    """ng x ng nodes -> 2*(ng-1)^2 triangles, optionally CW-wound."""
    lon1 = lon0 + step * np.arange(ng)
    lat1 = lat0 + step * np.arange(ng)
    LON, LAT = np.meshgrid(lon1, lat1)
    lon = LON.ravel(); lat = LAT.ravel()
    en = []
    for r in range(ng - 1):
        for c in range(ng - 1):
            i = r * ng + c
            t1 = [i, i + 1, i + ng + 1]
            t2 = [i, i + ng + 1, i + ng]
            if cw:
                t1 = t1[::-1]; t2 = t2[::-1]
            en.append(t1 + [-1]); en.append(t2 + [-1])
    return _mesh_from(lon, lat, en)


def _quad_grid(ng=4, lon0=0.0, lat0=40.0, step=1.0):
    lon1 = lon0 + step * np.arange(ng)
    lat1 = lat0 + step * np.arange(ng)
    LON, LAT = np.meshgrid(lon1, lat1)
    lon = LON.ravel(); lat = LAT.ravel()
    en = []
    for r in range(ng - 1):
        for c in range(ng - 1):
            i = r * ng + c
            en.append([i, i + 1, i + ng + 1, i + ng])
    return _mesh_from(lon, lat, en)


def _walk(mesh, lon_p, lat_p, start, max_steps=256):
    nb = build_elem_neighbors(mesh)
    return _walk_containing_element(
        float(lon_p), float(lat_p), int(start),
        mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat, nb, max_steps)


def _global_elem(mesh, lon_p, lat_p):
    nn = find_nearest_nodes(mesh, np.array([lon_p]), np.array([lat_p]))
    e, ok = find_containing_elements(
        mesh, np.array([float(lon_p)]), np.array([float(lat_p)]), nn)
    return int(e[0]), bool(ok[0])


class TestWalkPrimitive:
    def test_stay_in_start_element(self):
        m = _tri_grid()
        e_true, ok = _global_elem(m, 2.3, 42.6)
        assert ok
        e, found = _walk(m, 2.3, 42.6, e_true)
        assert found and e == e_true

    def test_one_hop(self):
        m = _tri_grid()
        # (2.3, 42.6): find its element, then start the walk from any
        # edge-neighbor of it — one hop back.
        e_true, ok = _global_elem(m, 2.3, 42.6)
        nb = build_elem_neighbors(m)
        start = int(nb[e_true][nb[e_true] >= 0][0])
        e, found = _walk(m, 2.3, 42.6, start)
        assert found and e == e_true

    def test_multi_hop_across_mesh(self):
        m = _tri_grid(ng=10)
        e_true, ok = _global_elem(m, 8.4, 48.7)   # far corner
        assert ok
        e, found = _walk(m, 8.4, 48.7, 0)          # start at opposite corner
        assert found and e == e_true

    def test_outside_mesh_returns_not_found(self):
        m = _tri_grid()
        e, found = _walk(m, 20.0, 42.0, 0)
        assert not found

    def test_cw_winding_same_result(self):
        m_ccw = _tri_grid(cw=False)
        m_cw = _tri_grid(cw=True)
        for p in [(2.3, 42.6), (0.2, 40.7), (4.8, 44.2)]:
            e1, f1 = _walk(m_ccw, p[0], p[1], 0)
            e2, f2 = _walk(m_cw, p[0], p[1], 0)
            assert f1 and f2
            # same geometric cell (elements are index-aligned across the
            # two meshes)
            assert e1 == e2

    def test_quads_multi_hop(self):
        m = _quad_grid(ng=5)
        e_true, ok = _global_elem(m, 3.4, 43.6)
        assert ok
        e, found = _walk(m, 3.4, 43.6, 0)
        assert found and e == e_true

    def test_point_on_shared_edge_accepted(self):
        m = _tri_grid()
        # (2.0, 42.5) lies exactly on the vertical grid line lon=2 —
        # a shared element edge.  Must be accepted in SOME adjacent element.
        e, found = _walk(m, 2.0, 42.5, 0)
        assert found
        nn = find_nearest_nodes(m, np.array([2.0]), np.array([42.5]))
        e_ids, in_dom = find_containing_elements(
            m, np.array([2.0]), np.array([42.5]), nn)
        assert in_dom[0]
        # the walk's element must actually contain the point (same accept
        # tolerance class as the fan test): verify via vertex bbox
        verts = m.elem_nodes[e][m.elem_nodes[e] >= 0]
        assert m.nodes_lon[verts].min() - 1e-9 <= 2.0 <= m.nodes_lon[verts].max() + 1e-9
        assert m.nodes_lat[verts].min() - 1e-9 <= 42.5 <= m.nodes_lat[verts].max() + 1e-9

    def test_point_on_vertex_accepted(self):
        m = _tri_grid()
        # exact interior grid node (2, 42)
        e, found = _walk(m, 2.0, 42.0, 0)
        assert found
        verts = m.elem_nodes[e][m.elem_nodes[e] >= 0]
        node_id = int(np.where((m.nodes_lon == 2.0) & (m.nodes_lat == 42.0))[0][0])
        assert node_id in verts

    def test_seam_quad_walk(self):
        # two quads straddling the antimeridian; walk from one into the other
        lon = [179.0, -180.0, -179.0, -179.0, -180.0, 179.0]
        lat = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        en = [[0, 1, 4, 5], [1, 2, 3, 4]]
        m = _mesh_from(lon, lat, en)
        # quad 0 spans lon 179..+-180, quad 1 spans -180..-179; the shared
        # edge sits exactly on the antimeridian.  Walk across it both ways.
        e, found = _walk(m, -179.5, 0.5, 0)   # point in quad 1, start quad 0
        assert found and e == 1
        e, found = _walk(m, 179.5, 0.5, 1)    # point in quad 0, start quad 1
        assert found and e == 0

    def test_fesomc_padded_and_repeated_encodings_agree(self):
        lon = [0.0, 1.0, 0.0, 1.2]
        lat = [0.0, 0.0, 1.0, 1.1]
        en_pad = [[0, 1, 2, -1], [1, 3, 2, -1]]
        en_rep = [[0, 1, 2, 0], [1, 3, 2, 1]]
        m_pad = _mesh_from(lon, lat, en_pad)
        m_rep = _mesh_from(lon, lat, en_rep)
        for p, start in [((0.3, 0.3), 1), ((0.9, 0.7), 0)]:
            e1, f1 = _walk(m_pad, p[0], p[1], start)
            e2, f2 = _walk(m_rep, p[0], p[1], start)
            assert f1 and f2
            assert e1 == e2

    def test_cap_exceeded_not_found(self):
        m = _tri_grid(ng=10)
        # target needs many hops; cap of 1 must give up (found=False)
        e, found = _walk(m, 8.4, 48.7, 0, max_steps=1)
        assert not found

    def test_outside_band_at_boundary_matches_local_verdict(self):
        # review round 1 (2026-07-02): the signed-area accept admits a
        # ~1e-6-normalized OUTSIDE band at boundary edges that the local
        # machinery rejects via its strict bbox prefilter — a life/death
        # verdict divergence, not an on-edge tie.  The walk must give the
        # LOCAL verdict for points just outside the domain.
        m = _quad_grid(ng=3)          # 2x2 quads spanning lon 0..2, lat 40..42
        eps = 1e-9                    # far inside the 1e-6-relative band
        pts = [(1.0, 40.0 - eps),     # just south of the south boundary
               (0.0 - eps, 41.0),     # just west of the west boundary
               (2.0 + eps, 41.5),     # just east
               (1.5, 42.0 + eps)]     # just north
        for lon_p, lat_p in pts:
            e_true, ok_local = _global_elem(m, lon_p, lat_p)
            e_w, in_w = find_containing_elements_walk(
                m, np.array([lon_p]), np.array([lat_p]),
                np.array([0], dtype=np.int32))
            assert bool(in_w[0]) == ok_local, (lon_p, lat_p)
            if ok_local:
                assert e_w[0] == e_true


class TestBatchWalk:
    def test_matches_global_search_random_points(self):
        rng = np.random.default_rng(7)
        for mesh in (_tri_grid(ng=8), _quad_grid(ng=6)):
            P = 300
            lons = rng.uniform(-0.5, 6.5, P)   # includes outside points
            lats = rng.uniform(39.5, 46.5, P)
            starts = rng.integers(0, mesh.n_elements, P).astype(np.int32)
            e_w, in_w = find_containing_elements_walk(mesh, lons, lats, starts)
            nn = find_nearest_nodes(mesh, lons, lats)
            e_g, in_g = find_containing_elements(mesh, lons, lats, nn)
            np.testing.assert_array_equal(in_w, in_g)
            # in general position the containing element is unique
            np.testing.assert_array_equal(e_w[in_w], e_g[in_g])

    def test_invalid_start_falls_back_to_global(self):
        m = _tri_grid()
        lons = np.array([2.3, 0.2])
        lats = np.array([42.6, 40.7])
        starts = np.array([-1, -1], dtype=np.int32)
        e_w, in_w = find_containing_elements_walk(m, lons, lats, starts)
        assert in_w.all()
        nn = find_nearest_nodes(m, lons, lats)
        e_g, _ = find_containing_elements(m, lons, lats, nn)
        np.testing.assert_array_equal(e_w, e_g)

    def test_outside_points_reported_outside(self):
        m = _tri_grid()
        lons = np.array([20.0, 2.3])
        lats = np.array([42.0, 42.6])
        starts = np.array([0, 0], dtype=np.int32)
        e_w, in_w = find_containing_elements_walk(m, lons, lats, starts)
        assert not in_w[0] and e_w[0] == -1
        assert in_w[1]


# ---------------------------------------------------------------------------
# Carried element on ParticleState
# ---------------------------------------------------------------------------

from kelp.core.particles import make_state


class TestParticleStateElem:
    def test_default_none_and_rides_replace(self):
        st = make_state(np.array([1.0]), np.array([2.0]))
        assert st.elem is None
        st2 = st.replace(elem=np.array([5], dtype=np.int32))
        assert st2.elem[0] == 5
        st3 = st2.replace(lon=np.array([1.5]))
        assert st3.elem is st2.elem      # untouched fields ride through

    def test_snapshot_includes_elem_only_when_set(self):
        st = make_state(np.array([1.0]), np.array([2.0]))
        assert "elem" not in st.snapshot()
        st2 = st.replace(elem=np.array([7], dtype=np.int32))
        snap = st2.snapshot()
        assert snap["elem"][0] == 7
        snap["elem"][0] = 9
        assert st2.elem[0] == 7          # snapshot is a copy


class TestElemCheckpointing:
    @pytest.mark.parametrize("suffix", [".nc", ".zarr"])
    def test_checkpoint_roundtrip_elem(self, tmp_path, suffix):
        from kelp.writers import TrajectoryWriter, load_restart
        st = make_state(np.array([1.0, 2.0]), np.array([2.0, 3.0]))
        st = st.replace(elem=np.array([4, -1], dtype=np.int32))
        path = str(tmp_path / f"out{suffix}")
        with TrajectoryWriter(path, n_particles=2) as w:
            w.append(st, 0.0)
            w.write_checkpoint(st, step_i=5, file_idx=1)
        r = load_restart(path)
        assert r["state"].elem is not None
        np.testing.assert_array_equal(r["state"].elem, [4, -1])
        assert r["state"].elem.dtype == np.int32

    @pytest.mark.parametrize("suffix", [".nc", ".zarr"])
    def test_checkpoint_without_elem_stays_clean(self, tmp_path, suffix):
        from kelp.writers import TrajectoryWriter, load_restart
        st = make_state(np.array([1.0]), np.array([2.0]))
        path = str(tmp_path / f"plain{suffix}")
        with TrajectoryWriter(path, n_particles=1) as w:
            w.append(st, 0.0)
            w.write_checkpoint(st, step_i=1, file_idx=0)
        r = load_restart(path)
        assert r["state"].elem is None

    def test_standalone_restart_file_roundtrip_elem(self, tmp_path):
        from kelp.writers import write_restart_file, load_restart
        st = make_state(np.array([1.0]), np.array([2.0]))
        st = st.replace(elem=np.array([11], dtype=np.int32))
        path = str(tmp_path / "restart_step005.nc")
        write_restart_file(path, st, step_i=5, file_idx=2)
        r = load_restart(path)
        np.testing.assert_array_equal(r["state"].elem, [11])


# ---------------------------------------------------------------------------
# elem_search option: config + integrator/tracking wiring
# ---------------------------------------------------------------------------

from kelp.core.io import VelocityField


def _vf_elem(mesh, fn):
    u = fn(mesh.elem_center_lon, mesh.elem_center_lat)
    return VelocityField(u=u, v=0.5 * u, eta=np.zeros(mesh.n_nodes), w=None)


def _seed_elems(mesh, lons, lats):
    nn = find_nearest_nodes(mesh, lons, lats)
    e, ok = find_containing_elements(mesh, lons, lats, nn)
    assert ok.all()
    return e.astype(np.int32)


class TestElemSearchConfig:
    def test_validation(self):
        from kelp.core.config import NumericsConfig
        assert NumericsConfig().elem_search == "local"
        NumericsConfig(elem_search="walk")
        with pytest.raises(ValueError, match="elem_search"):
            NumericsConfig(elem_search="jump")


class TestWalkStepIntegration:
    def test_euler_step_updates_elem(self):
        from kelp.numerics.integrators import euler_step
        m = _tri_grid()
        # strong flow so at least one particle CROSSES into another element
        # (otherwise a stale carried elem passes by accident)
        vf = _vf_elem(m, lambda x, y: np.full(x.shape, 20.0))
        lons = np.array([2.3, 0.7]); lats = np.array([42.6, 43.2])
        st = make_state(lons, lats)
        nn = find_nearest_nodes(m, lons, lats)
        st = st.replace(nearest_node=nn, elem=_seed_elems(m, lons, lats))
        out = euler_step(st, vf, m, 3600.0, 1.0, uv_interp="barycentric")
        assert out.elem is not None
        # the carried element must be the true containing element of the
        # NEW positions
        expected = _seed_elems(m, out.lon, out.lat)
        np.testing.assert_array_equal(out.elem, expected)

    def test_failed_particle_reverts_elem(self):
        from kelp.numerics.integrators import euler_step
        from kelp.core.particles import ParticleStatus
        m = _tri_grid()
        # strong eastward flow: the particle near the east rim exits
        vf = _vf_elem(m, lambda x, y: np.full(x.shape, 500.0))
        lons = np.array([4.9]); lats = np.array([42.5])
        st = make_state(lons, lats)
        nn = find_nearest_nodes(m, lons, lats)
        el0 = _seed_elems(m, lons, lats)
        st = st.replace(nearest_node=nn, elem=el0.copy())
        out = euler_step(st, vf, m, 36000.0, 1.0, uv_interp="barycentric")
        assert out.status[0] == ParticleStatus.OUTSIDE
        assert out.lon[0] == lons[0]           # reverted
        assert out.elem[0] == el0[0]           # elem reverted with it

    @pytest.mark.parametrize("integrator", ["euler", "rk2", "rk4"])
    @pytest.mark.parametrize("uv", ["barycentric", "edge_leastsq"])
    def test_step_walk_equals_local(self, integrator, uv):
        # The walk is a pure fast path: a step with carried elems must land
        # particles EXACTLY where the local-search step does.
        from kelp.numerics.integrators import get_integrator
        m = _tri_grid(ng=8)
        rng = np.random.default_rng(5)
        u_e = rng.uniform(-0.5, 0.5, m.n_elements)
        vf = VelocityField(u=u_e + 1.0, v=u_e.copy(),
                           eta=np.zeros(m.n_nodes), w=None)
        P = 40
        lons = rng.uniform(0.5, 6.5, P)
        lats = rng.uniform(40.5, 46.5, P)
        st = make_state(lons, lats)
        nn = find_nearest_nodes(m, lons, lats)
        st = st.replace(nearest_node=nn)
        step = get_integrator(integrator)
        out_local = step(st, vf, m, 7200.0, 1.0, uv_interp=uv)
        st_walk = st.replace(elem=_seed_elems(m, lons, lats))
        out_walk = step(st_walk, vf, m, 7200.0, 1.0, uv_interp=uv)
        np.testing.assert_array_equal(out_local.lon, out_walk.lon)
        np.testing.assert_array_equal(out_local.lat, out_walk.lat)
        np.testing.assert_array_equal(out_local.status, out_walk.status)


def _make_fesomc_const(tmp_path, uv_dim="nele", u_val=1.0, n_time=6):
    import pandas as pd
    import xarray as xr
    n_nodes, n_elem = 9, 8
    xs = np.array([4.0, 5.0, 6.0]); ys = np.array([4.0, 5.0, 6.0])
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel(); lat = gy.ravel()
    elems = []
    for j in range(2):
        for i in range(2):
            n0 = j * 3 + i
            elems.append([n0 + 1, n0 + 2, n0 + 4, n0 + 1])
            elems.append([n0 + 2, n0 + 5, n0 + 4, n0 + 2])
    nv = np.array(elems, dtype=np.int32)
    en0 = nv[:, :3] - 1
    size = n_nodes if uv_dim == "node" else n_elem
    ds = xr.Dataset({
        "lon": (["node"], lon), "lat": (["node"], lat),
        "depth": (["node"], np.full(9, 10.0)),
        "nv": (["nele", "four"], nv),
        "lon_elem": (["nele"], np.array([lon[e].mean() for e in en0])),
        "lat_elem": (["nele"], np.array([lat[e].mean() for e in en0])),
        "sigma_lev": (["sigma"], np.array([0.0, 0.5, 1.0])),
        "u2d": (["time", uv_dim], np.full((n_time, size), u_val)),
        "v2d": (["time", uv_dim], np.zeros((n_time, size))),
        "eta": (["time", "node"], np.zeros((n_time, 9))),
    }, coords={"time": pd.date_range("2020-01-01", periods=n_time,
                                     freq="3600s")})
    p = tmp_path / f"fesomc_{uv_dim}.nc"
    ds.to_netcdf(p)
    return p


def _open_fesomc(tmp_path, uv_dim="nele"):
    from types import SimpleNamespace
    from kelp.readers import get_reader
    p = _make_fesomc_const(tmp_path, uv_dim=uv_dim)
    r = get_reader("fesomc")
    mesh = r.load_mesh(SimpleNamespace(meshpath=str(p)))
    ds = r.open_dataset(str(p))
    return r, mesh, ds


class TestWalkTracking:
    def _run(self, tmp_path, elem_search, uv="barycentric",
             integrator="rk2", time_dir=1.0, start_idx=0):
        from kelp.core.config import ModeConfig, NumericsConfig
        from kelp.tracking import run_single_realization
        reader, mesh, ds = _open_fesomc(tmp_path)
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=time_dir)
        numerics = NumericsConfig(dt=3600.0, integrator=integrator,
                                  output_freq=1, uv_interp=uv,
                                  elem_search=elem_search)
        init = make_state(np.array([5.0, 4.6]), np.array([5.0, 5.3]))
        rec = run_single_realization(mesh, ds, reader, init, mode, numerics,
                                     start_time_idx=start_idx, n_steps=4)
        ds.close()
        return rec

    @pytest.mark.parametrize("uv", ["barycentric", "edge_leastsq"])
    def test_e2e_walk_equals_local(self, tmp_path, uv):
        rec_l = self._run(tmp_path, "local", uv=uv)
        rec_w = self._run(tmp_path, "walk", uv=uv)
        np.testing.assert_array_equal(np.asarray(rec_l.lons),
                                      np.asarray(rec_w.lons))
        np.testing.assert_array_equal(np.asarray(rec_l.lats),
                                      np.asarray(rec_w.lats))
        np.testing.assert_array_equal(np.asarray(rec_l.statuses),
                                      np.asarray(rec_w.statuses))

    def test_e2e_backward_walk_equals_local(self, tmp_path):
        rec_l = self._run(tmp_path, "local", time_dir=-1.0, start_idx=5)
        rec_w = self._run(tmp_path, "walk", time_dir=-1.0, start_idx=5)
        np.testing.assert_array_equal(np.asarray(rec_l.lons),
                                      np.asarray(rec_w.lons))

    def test_walk_with_fused_leastsq_warns(self, tmp_path):
        pytest.importorskip("numba")
        from kelp.core.config import (
            AppConfig, DataConfig, MeshConfig, ModeConfig, NumericsConfig,
            TimeConfig,
        )
        from kelp.tracking import run_tracking
        nc = _make_fesomc_const(tmp_path)
        parts = tmp_path / "p.dat"
        parts.write_text("5.0 5.0\n")
        out = tmp_path / "out.nc"
        cfg = AppConfig(
            mode=ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0),
            numerics=NumericsConfig(dt=3600.0, integrator="euler",
                                    output_freq=1, elem_search="walk"),
            mesh=MeshConfig(type="fesomc", meshpath=str(nc)),
            data=DataConfig(ncinput=str(nc), inicloud=str(parts),
                            output_path=str(out), time_averaging="instant"),
            time=TimeConfig(start_time_idx=0, n_steps=2),
        )
        with pytest.warns(RuntimeWarning, match="fused"):
            run_tracking(cfg, output_path=str(out))

    def test_track_particles_accepts_elem_search(self, tmp_path):
        from kelp.tracking import track_particles
        reader, mesh, ds = _open_fesomc(tmp_path)
        rec = track_particles(mesh, ds, reader,
                              np.array([5.0]), np.array([5.0]),
                              dt=3600.0, n_steps=2, integrator="rk2",
                              uv_interp="barycentric", elem_search="walk")
        ds.close()
        assert rec.lons[-1][0] > 5.0

    @pytest.mark.parametrize("integrator", ["rk2", "rk4"])
    def test_3d_step_walk_equals_local(self, integrator):
        # 3D plumbing (per-particle layers through _loop_interp_bary_3d,
        # elems= via _interp_uv_at; the rk4 variant pins the el1->el4
        # stage chain) — review round 1, important 2; review round 2 made
        # the rk4 claim real.  Three sequential steps, walk vs local
        # bitwise.
        from kelp.numerics.integrators import get_integrator
        step = get_integrator(integrator)
        m = _tri_grid(ng=8)
        m.sigma = np.array([0.0, 50.0, 100.0])
        m.depth = np.full(m.n_nodes, 100.0)
        rng = np.random.default_rng(13)
        u3 = np.vstack([rng.uniform(0.5, 1.5, m.n_elements),
                        rng.uniform(-1.5, -0.5, m.n_elements)])
        vf = VelocityField(u=u3, v=0.2 * u3, eta=np.zeros(m.n_nodes),
                           w=np.zeros((3, m.n_nodes)))
        P = 20
        lons = rng.uniform(1.0, 6.0, P)
        lats = rng.uniform(41.0, 46.0, P)
        zs = rng.uniform(5.0, 95.0, P)
        st = make_state(lons, lats, zs)
        nn = find_nearest_nodes(m, lons, lats)
        st_l = st.replace(nearest_node=nn)
        st_w = st_l.replace(elem=_seed_elems(m, lons, lats))
        for _ in range(3):
            st_l = step(st_l, vf, m, 3600.0, 1.0,
                        uv_interp="barycentric")
            st_w = step(st_w, vf, m, 3600.0, 1.0,
                        uv_interp="barycentric")
        np.testing.assert_array_equal(st_l.lon, st_w.lon)
        np.testing.assert_array_equal(st_l.lat, st_w.lat)
        np.testing.assert_array_equal(st_l.z, st_w.z)
        np.testing.assert_array_equal(st_l.status, st_w.status)
        assert st_w.elem is not None

    @pytest.mark.parametrize("uv", ["barycentric", "edge_leastsq"])
    def test_quad_step_walk_equals_local_near_edges(self, uv):
        # True-quad step-level coverage incl. near-edge seeds (review
        # round 1, important 4): the walk hint feeds the Wachspress path
        # whose on-edge guards trip the fan fallback.
        from kelp.numerics.integrators import euler_step
        m = _quad_grid(ng=6)
        rng = np.random.default_rng(17)
        u_e = rng.uniform(0.5, 1.5, m.n_elements)
        vf = VelocityField(u=u_e, v=0.3 * u_e,
                           eta=np.zeros(m.n_nodes), w=None)
        # interior points + seeds hugging the internal grid lines
        lons = np.concatenate([rng.uniform(0.5, 4.5, 20),
                               np.array([1.0 + 1e-7, 2.0 - 1e-7, 3.0])])
        lats = np.concatenate([rng.uniform(40.5, 44.5, 20),
                               np.array([41.5, 42.0 + 1e-7, 43.0 - 1e-7])])
        st = make_state(lons, lats)
        nn = find_nearest_nodes(m, lons, lats)
        st_l = st.replace(nearest_node=nn)
        st_w = st_l.replace(elem=_seed_elems(m, lons, lats))
        out_l = euler_step(st_l, vf, m, 3600.0, 1.0, uv_interp=uv)
        out_w = euler_step(st_w, vf, m, 3600.0, 1.0, uv_interp=uv)
        np.testing.assert_array_equal(out_l.lon, out_w.lon)
        np.testing.assert_array_equal(out_l.lat, out_w.lat)
        np.testing.assert_array_equal(out_l.status, out_w.status)

    def test_diffusion_kick_walk_equals_local_bitwise(self, tmp_path):
        # Kick-staleness self-correction is a stated contract (review
        # round 1, important 4b): uniform split diffusion moves particles
        # AFTER the elem update; the next step's walk must re-validate the
        # stale hints and stay bitwise-equal to local.
        from kelp.core.config import (DiffusionConfig, ModeConfig,
                                          NumericsConfig)
        from kelp.tracking import run_single_realization
        reader, mesh, ds = _open_fesomc(tmp_path)
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)
        recs = {}
        for es in ("local", "walk"):
            numerics = NumericsConfig(
                dt=3600.0, integrator="rk2", output_freq=1,
                uv_interp="barycentric", elem_search=es,
                diffusion=DiffusionConfig(enabled=True, scheme="uniform",
                                          kh_zonal=5.0, kh_meridional=5.0,
                                          seed=42))
            init = make_state(np.array([5.0, 4.6]), np.array([5.0, 5.3]))
            recs[es] = run_single_realization(
                mesh, ds, reader, init, mode, numerics,
                start_time_idx=0, n_steps=4)
        ds.close()
        np.testing.assert_array_equal(np.asarray(recs["local"].lons),
                                      np.asarray(recs["walk"].lons))
        np.testing.assert_array_equal(np.asarray(recs["local"].lats),
                                      np.asarray(recs["walk"].lats))

    def test_local_resume_of_walk_checkpoint_bitwise(self, tmp_path):
        # Config is authoritative (review round 1, important 3): resuming
        # a WALK checkpoint under elem_search 'local' must strip the
        # carried elems and continue byte-identically to a pure-local
        # continuous run.
        from kelp.core.config import ModeConfig, NumericsConfig
        from kelp.tracking import run_single_realization
        from kelp.writers import TrajectoryWriter, load_restart
        reader, mesh, ds = _open_fesomc(tmp_path)
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)
        num_local = NumericsConfig(dt=3600.0, integrator="rk2",
                                   output_freq=1, uv_interp="barycentric",
                                   elem_search="local")
        num_walk = NumericsConfig(dt=3600.0, integrator="rk2",
                                  output_freq=1, uv_interp="barycentric",
                                  elem_search="walk")
        init = make_state(np.array([5.0, 4.6]), np.array([5.0, 5.3]))

        rec_pure_local = run_single_realization(
            mesh, ds, reader, init, mode, num_local,
            start_time_idx=0, n_steps=4)

        seg = str(tmp_path / "seg_walk.nc")
        with TrajectoryWriter(seg, n_particles=2) as w:
            run_single_realization(mesh, ds, reader, init, mode, num_walk,
                                   start_time_idx=0, n_steps=2,
                                   writer=w, checkpoint_freq=1)
        r = load_restart(seg)
        assert r["state"].elem is not None
        rec_resumed = run_single_realization(
            mesh, ds, reader, r["state"], mode, num_local,
            start_time_idx=r["file_idx"] + 1, n_steps=2,
            start_step_i=r["step_i"] + 1)
        ds.close()
        np.testing.assert_array_equal(rec_pure_local.lons[-1],
                                      rec_resumed.lons[-1])
        np.testing.assert_array_equal(rec_pure_local.lats[-1],
                                      rec_resumed.lats[-1])

    def test_restart_resume_walk_bitwise(self, tmp_path):
        # segment (checkpoint at step 2) + resume == continuous run
        from kelp.core.config import ModeConfig, NumericsConfig
        from kelp.tracking import run_single_realization
        from kelp.writers import TrajectoryWriter, load_restart
        reader, mesh, ds = _open_fesomc(tmp_path)
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)
        numerics = NumericsConfig(dt=3600.0, integrator="rk2",
                                  output_freq=1, uv_interp="barycentric",
                                  elem_search="walk")
        init = make_state(np.array([5.0, 4.6]), np.array([5.0, 5.3]))

        rec_full = run_single_realization(mesh, ds, reader, init, mode,
                                          numerics, start_time_idx=0,
                                          n_steps=4)

        seg = str(tmp_path / "seg.nc")
        with TrajectoryWriter(seg, n_particles=2) as w:
            run_single_realization(mesh, ds, reader, init, mode, numerics,
                                   start_time_idx=0, n_steps=2,
                                   writer=w, checkpoint_freq=1)
        r = load_restart(seg)
        assert r["state"].elem is not None      # walk state checkpointed
        rec2 = run_single_realization(mesh, ds, reader, r["state"], mode,
                                      numerics,
                                      start_time_idx=r["file_idx"] + 1,
                                      n_steps=2,
                                      start_step_i=r["step_i"] + 1)
        ds.close()
        np.testing.assert_array_equal(rec_full.lons[-1], rec2.lons[-1])
        np.testing.assert_array_equal(rec_full.lats[-1], rec2.lats[-1])
