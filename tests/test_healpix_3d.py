"""HEALPix 3-D tracking (Phase 4): interface convention, W wiring, T/S
sampling, max_levels, self-contained 3-D extracts.

The ng5 conventions verified on the real store (closed
2026-07-06): u/v sit on the K-1 layer mids with an all-NaN padding level K;
``avg_wo`` sits ON the K interfaces (K data levels, surface interface ~ 0,
finite-prefix); the u- and w-family bottoms disagree per cell (separate
remaps), handled by zeroing non-finite velocities and forward-filling
sampled scalars.
"""
import sys
from pathlib import Path

import os

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")
xr = pytest.importorskip("xarray")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kelp.core.config import (AppConfig, DataConfig, MeshConfig,  # noqa: E402
                                  ModeConfig, NumericsConfig, TimeConfig)
from kelp.core.geometry import METERS_PER_DEG_LAT  # noqa: E402
from kelp.readers.healpix import (HealpixReader,  # noqa: E402
                                      _ffill_columns)
from test_healpix_reader import (_make_uv_store, _mesh_cfg,  # noqa: E402
                                 _pix_lonlat, _wet, LEVELS_M, NLEV)

MODE_3D = ModeConfig(Fdrift2D=False, Fdrift3D=True)


def _store_3d(tmp_path, name="3D_t_ocean.nc", **kw):
    kw.setdefault("with_eta", True)
    kw.setdefault("with_w", True)
    return _make_uv_store(tmp_path / name, **kw)


class TestLoadMesh3D:
    def test_levels_required(self, tmp_path):
        p = _store_3d(tmp_path)
        with pytest.raises(ValueError, match="mesh.levels"):
            HealpixReader.load_mesh(_mesh_cfg(p), mode_cfg=MODE_3D)

    def test_sigma_and_bathymetry(self, tmp_path):
        p = _store_3d(tmp_path)
        mesh = HealpixReader.load_mesh(_mesh_cfg(p, levels=LEVELS_M),
                                       mode_cfg=MODE_3D)
        np.testing.assert_array_equal(mesh.sigma, LEVELS_M)
        assert mesh.vertical_coord == "zlevel"
        # default fixture: every wet cell has NLEV-1 finite layers
        np.testing.assert_allclose(mesh.depth, LEVELS_M[NLEV - 1])


class TestReadVelocities3D:
    def test_shapes_padding_and_nan_zeroing(self, tmp_path):
        # mixed column lengths: lat < 0 -> 3 layers, lat >= 0 -> 2 layers
        lon, lat = _pix_lonlat(4, "nest")
        wet = _wet(lat)
        nfin = np.where(wet, np.where(lat < 0, 3, 2), 0)
        p = _store_3d(tmp_path, n_finite_levels=nfin, u_value=0.4,
                      w_value=1e-4)
        mesh = HealpixReader.load_mesh(_mesh_cfg(p, levels=LEVELS_M),
                                       mode_cfg=MODE_3D)
        ds = HealpixReader.open_dataset(p, mode_cfg=MODE_3D)
        vf = HealpixReader.read_velocities(ds, 0, mesh, mode_cfg=MODE_3D)
        n = mesh.n_nodes
        assert vf.u.shape == (NLEV - 1, n)      # K-1 layers, padding gone
        assert vf.w.shape == (NLEV, n)          # K interfaces
        assert np.isfinite(vf.u).all()          # below-bottom zeroed
        assert np.isfinite(vf.w).all()
        shallow = mesh.nodes_lat >= 0           # 2-layer columns
        np.testing.assert_allclose(vf.u[2, shallow], 0.0)   # in the ground
        np.testing.assert_allclose(vf.u[0], 0.4)
        # w on interfaces: a 2-layer column has interfaces 0..2 finite;
        # the file's +1e-4 is UPWARD (FESOM/HEALPix convention) -> the
        # internal positive-down rate is -1e-4
        np.testing.assert_allclose(vf.w[2, shallow], -1e-4)
        np.testing.assert_allclose(vf.w[3, shallow], 0.0)   # below bottom

    def test_w_missing_raises(self, tmp_path):
        p = _store_3d(tmp_path, with_w=False)
        mesh = HealpixReader.load_mesh(_mesh_cfg(p, levels=LEVELS_M),
                                       mode_cfg=MODE_3D)
        ds = HealpixReader.open_dataset(p, mode_cfg=MODE_3D)
        with pytest.raises(KeyError, match="vertical velocity"):
            HealpixReader.read_velocities(ds, 0, mesh, mode_cfg=MODE_3D)

    def test_max_levels_trims_read(self, tmp_path):
        p = _store_3d(tmp_path)
        mesh = HealpixReader.load_mesh(_mesh_cfg(p, levels=LEVELS_M),
                                       mode_cfg=MODE_3D)
        mesh.sigma = mesh.sigma[:3]             # what run_tracking does
        ds = HealpixReader.open_dataset(p, mode_cfg=MODE_3D)
        vf = HealpixReader.read_velocities(ds, 0, mesh, mode_cfg=MODE_3D)
        assert vf.u.shape[0] == 2
        assert vf.w.shape[0] == 3

    def test_scalar_ffill(self, tmp_path):
        # theta = 273 + layer (1-based); columns with 2 layers must carry
        # 274 (their bottom water) below their own bottom, not NaN
        lon, lat = _pix_lonlat(4, "nest")
        wet = _wet(lat)
        nfin = np.where(wet, np.where(lat < 0, 3, 2), 0)
        p = _store_3d(tmp_path, n_finite_levels=nfin, with_theta=True)
        mesh = HealpixReader.load_mesh(_mesh_cfg(p, levels=LEVELS_M),
                                       mode_cfg=MODE_3D)
        ds = HealpixReader.open_dataset(p, mode_cfg=MODE_3D)
        out = HealpixReader.read_scalar_fields(ds, 0, mesh, ["avg_thetao"],
                                               MODE_3D)
        th = out["avg_thetao"]
        assert th.shape == (NLEV - 1, mesh.n_nodes)
        shallow = mesh.nodes_lat >= 0           # 2-layer columns: 274, 275
        np.testing.assert_allclose(th[1, shallow], 275.0)
        np.testing.assert_allclose(th[2, shallow], 275.0)   # forward-filled
        deep = mesh.nodes_lat < 0               # 3-layer columns
        np.testing.assert_allclose(th[2, deep], 276.0)

    def test_ffill_unit(self):
        a = np.array([[np.nan, 1.0], [np.nan, np.nan], [3.0, np.nan]])
        f = _ffill_columns(a)
        np.testing.assert_array_equal(
            f, [[np.nan, 1.0], [np.nan, 1.0], [3.0, 1.0]])


class TestCContiguity3D:
    """Reader contract: every (K, N) field comes out C-contiguous.

    The parallel whole-field time blends (``core.io._blend`` and
    ``temporal_reconstruct._pchip_blend4``) gate on ``c_contiguous``;
    numpy's fancy compaction ``arr[:, idx]`` returns FORTRAN order, which
    silently dropped every healpix 3-D blend to the serial numpy path
    (measured: 25 s/step instead of ~2 s at nside 256, ~1 core busy on a
    16-core allocation).  Values are layout-independent — this pins the
    performance contract only.
    """

    def test_velocities_and_scalars_c_contiguous(self, tmp_path):
        p = _store_3d(tmp_path, with_theta=True)
        mesh = HealpixReader.load_mesh(_mesh_cfg(p, levels=LEVELS_M),
                                       mode_cfg=MODE_3D)
        ds = HealpixReader.open_dataset(p, mode_cfg=MODE_3D)
        vf = HealpixReader.read_velocities(ds, 0, mesh, mode_cfg=MODE_3D)
        for name in ("u", "v", "w"):
            arr = getattr(vf, name)
            assert arr.ndim == 2 and all(s > 1 for s in arr.shape)
            assert arr.flags.c_contiguous, f"{name} not C-contiguous"
        th = HealpixReader.read_scalar_fields(
            ds, 0, mesh, ["avg_thetao"], MODE_3D)["avg_thetao"]
        assert th.ndim == 2 and all(s > 1 for s in th.shape)
        assert th.flags.c_contiguous, "sampled scalar not C-contiguous"


class TestTracking3D:
    def _run(self, tmp_path, store, seeds, sample_vars=(), n_steps=2,
             levels=LEVELS_M, meshpath=None):
        from kelp.tracking import run_tracking
        from kelp.writers import load_trajectories
        seeds_path = tmp_path / "seeds3d.txt"
        with open(seeds_path, "w") as fh:
            for lo, la, z in seeds:
                fh.write(f"{lo} {la} {z}\n")
        out = tmp_path / "traj3d.nc"
        cfg = AppConfig(
            mode=ModeConfig(Fdrift2D=False, Fdrift3D=True,
                            sample_vars=list(sample_vars)),
            numerics=NumericsConfig(uv_interp="barycentric"),
            mesh=MeshConfig(type="healpix", meshpath=str(meshpath or store),
                            levels=levels),
            data=DataConfig(ncinput=str(store), inicloud=str(seeds_path),
                            output_path=str(out), time_averaging="instant"),
            time=TimeConfig(start_time_idx=0, n_steps=n_steps),
        )
        run_tracking(cfg)
        return load_trajectories(str(out))

    def test_analytic_3d_advection(self, tmp_path):
        # constant u + constant w -> analytic lon and z displacement
        # file w is positive UP (FESOM/HEALPix): -1e-4 = downwelling
        store = _store_3d(tmp_path, name="3D_run_ocean.nc", nside=8,
                          u_value=0.4, w_value=-1e-4)
        seeds = [(10.0, 5.0, 15.0), (-40.0, -20.0, 5.0)]
        traj = self._run(tmp_path, store, seeds)
        dt, n = 86400.0, 2
        for i, (lo, la, z0) in enumerate(seeds):
            dlon = n * 0.4 * dt / (np.cos(np.radians(la))
                                   * METERS_PER_DEG_LAT)
            assert abs(traj["lat"][i, -1] - la) < 1e-9
            assert abs(traj["lon"][i, -1] - (lo + dlon)) < 1e-6
            # downwelling 1e-4 m/s: z grows by n*dt*1e-4
            assert abs(traj["z"][i, -1] - (z0 + n * dt * 1e-4)) < 1e-6

    def test_layer_dependent_u_selects_particle_layer(self, tmp_path):
        # constant fields are blind to layer-index bugs: make u differ per
        # LAYER (0.1*(k+1)) and verify each particle advects with ITS
        # layer's velocity (interfaces [0,10,30,70]; w=0 keeps z fixed)
        store = _store_3d(tmp_path, name="3D_lay_ocean.nc", nside=8,
                          w_value=0.0)
        ds = xr.open_dataset(store)
        u = ds["avg_uoe"].values
        for k in range(NLEV):
            fin = np.isfinite(u[:, k, :])
            u[:, k, :][fin] = 0.1 * (k + 1)
        ds["avg_uoe"] = (("time", "level", "value"), u)
        p2 = tmp_path / "3D_lay2_ocean.nc"
        ds.to_netcdf(p2)
        seeds = [(10.0, 5.0, 5.0),     # layer 0 (0-10 m)   -> u = 0.1
                 (10.0, 5.0, 20.0),    # layer 1 (10-30 m)  -> u = 0.2
                 (10.0, 5.0, 50.0)]    # layer 2 (30-70 m)  -> u = 0.3
        traj = self._run(tmp_path, p2, seeds)
        dt, n = 86400.0, 2
        for i, u_k in enumerate((0.1, 0.2, 0.3)):
            dlon = n * u_k * dt / (np.cos(np.radians(5.0))
                                   * METERS_PER_DEG_LAT)
            assert abs(traj["lon"][i, -1] - (10.0 + dlon)) < 1e-6, i

    def test_nonconstant_w_vertical_interp(self, tmp_path):
        # w varies per INTERFACE (1e-4 * k): a particle between interfaces
        # must feel the vertically-interpolated w — constant-w tests are
        # blind to interface indexing (review round 1)
        store = _store_3d(tmp_path, name="3D_wvar_ocean.nc", nside=8,
                          u_value=0.0, w_value=0.0)
        ds = xr.open_dataset(store)
        wv = ds["avg_wo"].values
        for k in range(NLEV):
            fin = np.isfinite(wv[:, k, :])
            wv[:, k, :][fin] = -1e-4 * k      # positive UP: downwelling
        ds["avg_wo"] = (("time", "level", "value"), wv)
        p2 = tmp_path / "3D_wvar2_ocean.nc"
        ds.to_netcdf(p2)
        # z0 = 20 m sits midway between interfaces 10 (k=1) and 30 (k=2):
        # w(20) = 1.5e-4 -> one Euler step dz = 86400 * 1.5e-4 = 12.96 m
        traj = self._run(tmp_path, p2, [(10.0, 5.0, 20.0)], n_steps=1)
        assert abs(traj["z"][0, -1] - (20.0 + 12.96)) < 1e-6

    def test_interior_nan_fails_loud(self, tmp_path):
        # an interior all-NaN layer ABOVE finite data is corruption, not
        # bathymetry — must not be silently zeroed into fake ground
        import kelp.readers.healpix as hx
        store = _store_3d(tmp_path, name="3D_corrupt_ocean.nc", nside=4)
        ds = xr.open_dataset(store)
        u = ds["avg_uoe"].values
        u[:, 0, :] = np.nan                     # kill the SURFACE layer
        ds["avg_uoe"] = (("time", "level", "value"), u)
        p2 = tmp_path / "3D_corrupt2_ocean.nc"
        ds.to_netcdf(p2)
        mesh = HealpixReader.load_mesh(_mesh_cfg(store, levels=LEVELS_M),
                                       mode_cfg=MODE_3D)
        ds2 = HealpixReader.open_dataset(p2, mode_cfg=MODE_3D)
        hx._WARNED.clear()
        with pytest.raises(ValueError, match="BELOW"):
            HealpixReader.read_velocities(ds2, 0, mesh, mode_cfg=MODE_3D)

    def test_near_topography_velocity_bounded(self, tmp_path):
        # a particle whose stencil mixes deep columns (u = 0.4) and
        # shallower zero-filled ones must get a diluted-but-sane velocity
        # (the documented no-flow-at-ground treatment) — never NaN, never
        # amplified, and the particle stays alive
        lon, lat = _pix_lonlat(8, "nest")
        wet = _wet(lat)
        nfin = np.where(wet, np.where(lat < 0, 3, 2), 0)
        store = _store_3d(tmp_path, name="3D_topo_ocean.nc", nside=8,
                          n_finite_levels=nfin, u_value=0.4, w_value=0.0)
        # z = 50 m -> layer 3 (30..70 m): exists only in lat < 0 columns;
        # seed just south of the step so the stencil straddles it
        traj = self._run(tmp_path, store, [(10.0, -4.0, 50.0)])
        assert traj["status"][0, -1] == 1       # ALIVE
        dlon = traj["lon"][0, -1] - 10.0
        dlon_full = 2 * 0.4 * 86400.0 / (np.cos(np.radians(4.0))
                                         * METERS_PER_DEG_LAT)
        assert 0.0 <= dlon <= dlon_full + 1e-9

    def test_sampling_theta_at_layer_midpoint(self, tmp_path):
        # z = 20 m = the exact midpoint of layer 2 (10..30 m) -> sampled
        # theta = 273 + 2 exactly; u = w = 0 keeps the particle there
        store = _store_3d(tmp_path, name="3D_ts_ocean.nc", nside=8,
                          u_value=0.0, w_value=0.0, with_theta=True)
        traj = self._run(tmp_path, store, [(10.0, 5.0, 20.0)],
                         sample_vars=["avg_thetao"])
        assert abs(traj["z"][0, -1] - 20.0) < 1e-9
        np.testing.assert_allclose(traj["avg_thetao"][0], 275.0)


class TestRecommendedPath3D:
    def test_rk2_temporal_interp_sampling(self, tmp_path):
        # the SHIPPED example config path: rk2 + temporal_interp + T/S
        # sampling on a two-store 3-D setup (reviewer coverage gap) —
        # constant fields keep the run analytic even under the lerp
        from kelp.tracking import run_tracking
        from kelp.writers import load_trajectories
        from test_healpix_reader import _make_eta_store
        store = _store_3d(tmp_path, name="3D_rec_ocean.nc", nside=8,
                          u_value=0.4, w_value=-1e-4, with_theta=True,
                          with_eta=False)    # w positive UP: downwelling
        eta = _make_eta_store(tmp_path / "eta_rec.nc", nside=8)
        seeds_path = tmp_path / "seeds_rec.txt"
        seeds_path.write_text("10.0 5.0 20.0\n")
        out = tmp_path / "traj_rec.nc"
        cfg = AppConfig(
            mode=ModeConfig(Fdrift2D=False, Fdrift3D=True,
                            sample_vars=["avg_thetao"]),
            numerics=NumericsConfig(uv_interp="barycentric",
                                    integrator="rk2", dt=43200.0,
                                    temporal_interp=True,
                                    velocity_float32=True),
            mesh=MeshConfig(type="healpix", meshpath=str(store),
                            levels=LEVELS_M),
            data=DataConfig(ncinput=str(store), inicloud=str(seeds_path),
                            ncinput_eta=str(eta), output_path=str(out),
                            time_averaging="instant"),
            time=TimeConfig(start_time_idx=0, n_steps=4),
        )
        run_tracking(cfg)
        traj = load_trajectories(str(out))
        dt, n = 43200.0, 4
        dlon = n * 0.4 * dt / (np.cos(np.radians(5.0))
                               * METERS_PER_DEG_LAT)
        # float32 velocities: relative accuracy ~1e-7
        assert abs(traj["lon"][0, -1] - (10.0 + dlon)) < 1e-4
        assert abs(traj["z"][0, -1] - (20.0 + n * dt * 1e-4)) < 1e-3
        assert np.isfinite(traj["avg_thetao"][0]).all()


class TestExtract3D:
    def test_self_contained_3d_extract(self, tmp_path):
        pytest.importorskip("zarr")
        from extract_healpix_region import extract_region
        src = _store_3d(tmp_path, name="3D_ex_ocean.nc", nside=8,
                        u_value=0.4, w_value=1e-4, with_theta=True)
        out = tmp_path / "sector3d.zarr"
        extract_region({
            "source": {"path": str(src), "levels": list(LEVELS_M)},
            "region": {"bbox": [-30.0, 30.0, -30.0, 30.0],
                       "halo_layers": 2},
            "variables": ["avg_uoe", "avg_von", "avg_wo", "avg_thetao"],
            "output": {"path": str(out), "time_chunk": 2},
        })
        ds = xr.open_zarr(out)
        np.testing.assert_allclose(ds["level_interfaces_m"].values,
                                   LEVELS_M)
        # 3-D run on the extract WITHOUT mesh.levels (self-contained)
        mesh = HealpixReader.load_mesh(_mesh_cfg(out), mode_cfg=MODE_3D)
        np.testing.assert_array_equal(mesh.sigma, LEVELS_M)
        vf = HealpixReader.read_velocities(
            HealpixReader.open_dataset(out, mode_cfg=MODE_3D), 0, mesh,
            mode_cfg=MODE_3D)
        assert vf.w.shape == (NLEV, mesh.n_nodes)
        # conflicting mesh.levels -> loud error
        with pytest.raises(ValueError, match="conflicts"):
            HealpixReader.load_mesh(
                _mesh_cfg(out, levels=[0.0, 5.0, 9.0, 12.0]),
                mode_cfg=MODE_3D)

    def test_tracking_equivalence_3d(self, tmp_path):
        pytest.importorskip("zarr")
        from extract_healpix_region import extract_region
        src = _store_3d(tmp_path, name="3D_eq_ocean.nc", nside=8,
                        u_value=0.4, w_value=1e-4)
        out = tmp_path / "sector3d_eq.zarr"
        extract_region({
            "source": {"path": str(src), "levels": list(LEVELS_M)},
            "region": {"bbox": [-30.0, 30.0, -30.0, 30.0],
                       "halo_layers": 2},
            "variables": ["avg_uoe", "avg_von", "avg_wo"],
            "output": {"path": str(out), "time_chunk": 2},
        })
        runner = TestTracking3D()
        seeds = [(5.0, 5.0, 15.0), (-10.0, -12.0, 25.0)]
        t_dir = runner._run(tmp_path, src, seeds)
        t_ext = runner._run(tmp_path, out, seeds, levels=None,
                            meshpath=out)
        np.testing.assert_array_equal(t_ext["lon"], t_dir["lon"])
        np.testing.assert_array_equal(t_ext["lat"], t_dir["lat"])
        np.testing.assert_array_equal(t_ext["z"], t_dir["z"])


# Set KELP_TEST_NG5_STORE to an ng5 3-D HEALPix store; unset -> skipped.
_NG5_3D_Z7 = os.environ.get("KELP_TEST_NG5_STORE", "")
_NG5_AUX3D = os.environ.get("KELP_TEST_NG5_AUX3D", "")   # the NG5 mesh aux3d.out; unset -> skipped


@pytest.mark.skipif(
    not (_NG5_3D_Z7 and Path(_NG5_3D_Z7).exists()
         and _NG5_AUX3D and Path(_NG5_AUX3D).exists()),
    reason="ng5 store / aux3d not available")
class TestRealNg53D:
    def test_read_3d_snapshot(self):
        pytest.importorskip("gribscan")
        pytest.importorskip("dask")
        mesh = HealpixReader.load_mesh(
            _mesh_cfg(_NG5_3D_Z7, levels=_NG5_AUX3D), mode_cfg=MODE_3D)
        assert mesh.sigma.size == 70
        assert mesh.depth.max() == 6350.0       # deepest interface reached
        ds = HealpixReader.open_dataset(_NG5_3D_Z7, mode_cfg=MODE_3D)
        vf = HealpixReader.read_velocities(ds, 0, mesh, mode_cfg=MODE_3D)
        assert vf.u.shape == (69, mesh.n_nodes)
        assert vf.w.shape == (70, mesh.n_nodes)
        assert np.isfinite(vf.u).all() and np.isfinite(vf.w).all()
        assert np.abs(vf.u).max() < 5.0
        assert np.abs(vf.w).max() < 0.1
        np.testing.assert_allclose(vf.w[0], 0.0, atol=1e-12)  # surface ifc
