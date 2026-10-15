"""HEALPix native reader (Phase 2): mesh-from-store, two-store eta recipe,
carrier handling, compaction, fail-fasts, end-to-end tracking.

Synthetic stores are built in tmp_path with the ng5 conventions pinned in
docs/healpix.md §2: dim ``value``, ``crs`` var with healpix attrs, lon in
[0, 360), land = NaN, u/v on a ``level`` axis (surface = index 0, deepest
level all-NaN padding), ssh in a separate 2-D store.
"""
import os

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")
xr = pytest.importorskip("xarray")

from kelp.core.config import (AppConfig, DataConfig, MeshConfig,
                                  ModeConfig, NumericsConfig, TimeConfig)
from kelp.core.geometry import METERS_PER_DEG_LAT
from kelp.readers import get_reader
from kelp.readers.healpix import (HealpixReader, _derive_eta_sibling,
                                      _is_kerchunk_path, _resolve_levels)

NSIDE = 4
NPIX = 12 * NSIDE * NSIDE
NLEV = 4
NTIME = 3
LEVELS_M = [0.0, 10.0, 30.0, 70.0]


def _pix_lonlat(nside, order):
    lon, lat = healpy.pix2ang(nside, np.arange(12 * nside * nside),
                              nest=(order == "nest"), lonlat=True)
    return np.mod(lon, 360.0), lat


def _wet(lat):
    return lat < 60.0                       # dry northern cap = land


def _u_field(lon, lat):
    return 0.4 + 0.0 * lon                  # constant (analytic advection)


def _make_uv_store(path, *, nside=NSIDE, order="nest", with_crs=True,
                   with_eta=False, with_deptho=False, u_value=None,
                   n_finite_levels=None, extra_vars=(), with_w=False,
                   w_value=0.0, with_theta=False):
    """Write a 3-D-store-shaped dataset: u/v (time, level, value) + land NaN.

    ``n_finite_levels`` — per-pixel count of finite levels (finite-prefix
    columns); default: all wet pixels have NLEV-1 finite levels (the last
    level is all-NaN padding, the ng5 quirk).  ``with_w`` adds ``avg_wo``
    ON the interfaces (a cell with n finite u-layers gets n+1 finite w
    entries — the verified ng5 convention).  ``with_theta`` adds a
    mid-layer scalar ``avg_thetao`` = 273 + layer index (1-based).
    """
    npix = 12 * nside * nside
    lon, lat = _pix_lonlat(nside, order)
    wet = _wet(lat)
    if n_finite_levels is None:
        n_finite_levels = np.where(wet, NLEV - 1, 0)
    u = np.full((NTIME, NLEV, npix), np.nan)
    v = np.full((NTIME, NLEV, npix), np.nan)
    for k in range(NLEV):
        alive = n_finite_levels > k
        u[:, k, alive] = (u_value if u_value is not None
                          else _u_field(lon, lat)[alive])
        v[:, k, alive] = 0.0
    ds = xr.Dataset(
        {
            "avg_uoe": (("time", "level", "value"), u),
            "avg_von": (("time", "level", "value"), v),
        },
        coords={
            "time": np.array(["2020-01-02", "2020-01-03", "2020-01-04"],
                             dtype="datetime64[ns]"),
            "level": np.arange(1, NLEV + 1),
            "lon": ("value", lon),
            "lat": ("value", lat),
        },
    )
    for nm in extra_vars:
        f = np.where(wet, lat * 0.1, np.nan)
        ds[nm] = (("time", "value"), np.repeat(f[None, :], NTIME, axis=0))
    if with_w:
        wv = np.full((NTIME, NLEV, npix), np.nan)
        for k in range(NLEV):
            on_ifc = (n_finite_levels >= k) & (n_finite_levels > 0)
            wv[:, k, on_ifc] = w_value
        ds["avg_wo"] = (("time", "level", "value"), wv)
    if with_theta:
        th = np.full((NTIME, NLEV, npix), np.nan)
        for k in range(NLEV):
            alive = n_finite_levels > k
            th[:, k, alive] = 273.0 + (k + 1)
        ds["avg_thetao"] = (("time", "level", "value"), th)
    if with_eta:
        e = np.where(wet, 0.25, np.nan)
        ds["avg_zos"] = (("time", "value"),
                         np.repeat(e[None, :], NTIME, axis=0))
    if with_deptho:
        ds["deptho"] = ("value", np.where(wet, 123.0, np.nan))
    if with_crs:
        ds["crs"] = ((), np.int32(0),
                     {"grid_mapping_name": "healpix",
                      "healpix_nside": nside, "healpix_order": order})
    ds.to_netcdf(path)
    return path


def _make_eta_store(path, *, nside=NSIDE, order="nest", eta_value=0.25,
                    times=("2020-01-02", "2020-01-03", "2020-01-04")):
    npix = 12 * nside * nside
    _, lat = _pix_lonlat(nside, order)
    e = np.where(_wet(lat), eta_value, np.nan)
    ds = xr.Dataset(
        {"avg_zos": (("time", "value"),
                     np.repeat(e[None, :], len(times), axis=0))},
        coords={"time": np.array(list(times), dtype="datetime64[ns]")},
    )
    ds["crs"] = ((), np.int32(0),
                 {"grid_mapping_name": "healpix",
                  "healpix_nside": nside, "healpix_order": order})
    ds.to_netcdf(path)
    return path


def _mesh_cfg(path, **kw):
    return MeshConfig(type="healpix", meshpath=str(path), **kw)


# ---------------------------------------------------------------------------
# load_mesh
# ---------------------------------------------------------------------------

class TestLoadMesh:
    def test_nside_and_wet_from_store(self, tmp_path):
        p = _make_uv_store(tmp_path / "3D_test_ocean.nc")
        mesh = HealpixReader.load_mesh(_mesh_cfg(p))
        assert mesh.metadata["healpix_nside"] == NSIDE
        assert (mesh.nodes_lat < 60.0).all()          # dry cap excluded
        assert mesh.n_nodes <= int(_wet(_pix_lonlat(NSIDE, "nest")[1]).sum())
        # store order stashed for compaction
        np.testing.assert_array_equal(mesh._healpix_data_idx,
                                      mesh._healpix_node_pix)

    def test_zoom_fallback_without_crs(self, tmp_path):
        p = _make_uv_store(tmp_path / "3D_nocrs.nc", with_crs=False)
        with pytest.raises(ValueError, match="healpix_zoom"):
            HealpixReader.load_mesh(_mesh_cfg(p))
        mesh = HealpixReader.load_mesh(_mesh_cfg(p, healpix_zoom=2))
        assert mesh.metadata["healpix_nside"] == 4    # 2**2

    def test_rotation_wet_pole_rejected(self, tmp_path):
        """mesh.rotation is SUPPORTED since 2026-08-16
        (test_healpix_rotation.py) — but THIS fixture's wet mask (wet =
        lat < 60, so the geographic South Pole region is wet) puts a
        rotated pole in water, and the straddle guard must reject it
        with the actionable message.  (The old blanket em... rotation
        rejection used to fire here for a different reason.)"""
        p = _make_uv_store(tmp_path / "3D_rot.nc")
        with pytest.raises(ValueError, match="rotated pole"):
            HealpixReader.load_mesh(_mesh_cfg(p, rotation="fesom2"))

    def test_ring_store_equals_nest_store(self, tmp_path):
        pn = _make_uv_store(tmp_path / "3D_nest.nc", order="nest")
        pr = _make_uv_store(tmp_path / "3D_ring.nc", order="ring")
        mn = HealpixReader.load_mesh(_mesh_cfg(pn))
        mr = HealpixReader.load_mesh(_mesh_cfg(pr))
        np.testing.assert_array_equal(mn._healpix_node_pix,
                                      mr._healpix_node_pix)
        np.testing.assert_array_equal(mn.elem_nodes, mr.elem_nodes)
        np.testing.assert_allclose(mn.nodes_lon, mr.nodes_lon, atol=1e-12)
        # data compaction differs (store order) but reads the same physics
        dsn = HealpixReader.open_dataset(pn)
        dsr = HealpixReader.open_dataset(pr)
        vn = HealpixReader.read_velocities(dsn, 0, mn)
        vr = HealpixReader.read_velocities(dsr, 0, mr)
        np.testing.assert_allclose(vn.u, vr.u, atol=1e-12)

    def test_crop_bbox(self, tmp_path):
        p = _make_uv_store(tmp_path / "3D_crop.nc")
        mesh = HealpixReader.load_mesh(
            _mesh_cfg(p, crop_bbox=[-40.0, 40.0, -30.0, 30.0]))
        lon = mesh.nodes_lon
        assert ((lon >= -41) & (lon <= 41)).all()
        assert (np.abs(mesh.nodes_lat) <= 31).all()
        # dateline-crossing window (W > E)
        mesh2 = HealpixReader.load_mesh(
            _mesh_cfg(p, crop_bbox=[150.0, -150.0, -30.0, 30.0]))
        lon2 = np.mod(mesh2.nodes_lon, 360.0)
        assert ((lon2 >= 149) | (lon2 <= -149 % 360)).all()

    def test_depth_from_deptho(self, tmp_path):
        p = _make_uv_store(tmp_path / "3D_dep.nc", with_deptho=True)
        mesh = HealpixReader.load_mesh(_mesh_cfg(p))
        np.testing.assert_allclose(mesh.depth, 123.0)

    def test_deptho_refines_wet_mask_sentinel_land(self, tmp_path):
        # a product with NON-NaN (sentinel-filled) land velocities: the
        # deptho > 0 refinement is the only thing that excludes land
        npix = NPIX
        nfin = np.full(npix, NLEV - 1)        # u finite EVERYWHERE
        p = _make_uv_store(tmp_path / "3D_sent.nc", with_deptho=True,
                           n_finite_levels=nfin)
        mesh = HealpixReader.load_mesh(_mesh_cfg(p))
        assert (mesh.nodes_lat < 60.0).all()  # deptho NaN->0 on the cap

    def test_mode_cfg_overrides_reach_wet_mask(self, tmp_path):
        # reviewer finding: load_mesh must honor mode.var_u/var_v (threaded
        # in by run_tracking) — a store with only nonstandard names builds
        # with the override and fails loud without it
        p = _make_uv_store(tmp_path / "3D_names.nc")
        ds = xr.open_dataset(p)
        ds = ds.rename({"avg_uoe": "my_u", "avg_von": "my_v"})
        p2 = tmp_path / "3D_names2.nc"
        ds.to_netcdf(p2)
        with pytest.raises(KeyError, match="no u/v"):
            HealpixReader.load_mesh(_mesh_cfg(p2))
        mesh = HealpixReader.load_mesh(
            _mesh_cfg(p2), mode_cfg=ModeConfig(var_u="my_u", var_v="my_v"))
        assert mesh.n_nodes > 0

    def test_sigma_layer_wet_mask_matches_tracked_level(self, tmp_path):
        # wet extent differs strongly between level 0 and level 1: without
        # mode_cfg threading the mesh uses the SURFACE mask and reading
        # level 1 hits the massive-NaN guard; with it, the mask matches
        lon, lat = _pix_lonlat(NSIDE, "nest")
        wet = _wet(lat)
        nfin = np.where(wet, np.where(lat < 0, 3, 1), 0)
        p = _make_uv_store(tmp_path / "3D_lvlmask.nc", n_finite_levels=nfin,
                           with_eta=True)
        mode = ModeConfig(sigma_layer=1)
        mesh_surf = HealpixReader.load_mesh(_mesh_cfg(p))       # surface mask
        ds = HealpixReader.open_dataset(p)
        with pytest.raises(ValueError, match="non-finite"):
            HealpixReader.read_velocities(ds, 0, mesh_surf, mode_cfg=mode)
        mesh_l1 = HealpixReader.load_mesh(_mesh_cfg(p), mode_cfg=mode)
        assert (mesh_l1.nodes_lat < 0).all()   # only 2-level cells remain
        vf = HealpixReader.read_velocities(ds, 0, mesh_l1, mode_cfg=mode)
        assert np.isfinite(vf.u).all()

    def test_depth_from_levels_finite_prefix(self, tmp_path):
        # per-pixel finite-prefix depth: cells with k finite levels get
        # interface depth LEVELS_M[k]
        lon, lat = _pix_lonlat(NSIDE, "nest")
        wet = _wet(lat)
        nfin = np.where(wet, np.where(lat < 0, 3, 2), 0)
        p = _make_uv_store(tmp_path / "3D_lev.nc", n_finite_levels=nfin)
        mesh = HealpixReader.load_mesh(_mesh_cfg(p, levels=LEVELS_M))
        node_nfin = nfin[mesh._healpix_node_pix]
        np.testing.assert_allclose(
            mesh.depth, np.asarray(LEVELS_M)[node_nfin])
        np.testing.assert_array_equal(mesh.sigma, LEVELS_M)

    def test_crop_bbox_latitude_band(self, tmp_path):
        # review round 2 finding: [-180, 180, S, N] must mean ALL
        # longitudes, not collapse to the lon == -180 meridian
        p = _make_uv_store(tmp_path / "3D_band.nc")
        mesh = HealpixReader.load_mesh(
            _mesh_cfg(p, crop_bbox=[-180.0, 180.0, -20.0, 20.0]))
        assert (np.abs(mesh.nodes_lat) <= 21).all()
        lon_spread = mesh.nodes_lon.max() - mesh.nodes_lon.min()
        assert lon_spread > 300.0            # the full circle survived
        with pytest.raises(ValueError, match="zero-width"):
            HealpixReader.load_mesh(
                _mesh_cfg(p, crop_bbox=[30.0, 30.0, -20.0, 20.0]))

    def test_sigma_layer_out_of_range(self, tmp_path):
        p = _make_uv_store(tmp_path / "3D_oor.nc", with_eta=True)
        mesh = HealpixReader.load_mesh(_mesh_cfg(p))
        ds = HealpixReader.open_dataset(p)
        with pytest.raises(ValueError, match="sigma_layer"):
            HealpixReader.read_velocities(
                ds, 0, mesh, mode_cfg=ModeConfig(sigma_layer=99))

    def test_levels_aux3d_file(self, tmp_path):
        aux = tmp_path / "aux3d.out"
        aux.write_text("4\n0.\n-10.\n-30.\n-70.\n999999\n888888\n")
        arr = _resolve_levels(str(aux))
        np.testing.assert_allclose(arr, [0.0, 10.0, 30.0, 70.0])
        with pytest.raises(ValueError, match="increasing"):
            _resolve_levels([0.0, 10.0, 5.0])


# ---------------------------------------------------------------------------
# open_dataset: carriers, two-store recipe, fail-fasts
# ---------------------------------------------------------------------------

class TestOpenDataset:
    def test_single_store_with_eta(self, tmp_path):
        p = _make_uv_store(tmp_path / "3D_one.nc", with_eta=True)
        ds = HealpixReader.open_dataset(p)
        assert "avg_zos" in ds.variables

    def test_eta_path_merge(self, tmp_path):
        p = _make_uv_store(tmp_path / "3D_main.nc")
        pe = _make_eta_store(tmp_path / "eta_store.nc")
        ds = HealpixReader.open_dataset(p, eta_path=str(pe))
        assert "avg_zos" in ds.variables
        assert ds.sizes["time"] == NTIME

    def test_eta_sibling_autoderived(self, tmp_path):
        p = _make_uv_store(tmp_path / "3D_daily_test_ocean.nc")
        _make_eta_store(tmp_path / "2D_daily_test_ocean.nc")
        assert _derive_eta_sibling(p) is not None
        ds = HealpixReader.open_dataset(p)
        assert "avg_zos" in ds.variables

    def test_no_eta_warns_and_zeroes(self, tmp_path):
        import kelp.readers.healpix as hx
        hx._WARNED.clear()
        p = _make_uv_store(tmp_path / "3D_noeta.nc")
        with pytest.warns(RuntimeWarning, match="eta"):
            ds = HealpixReader.open_dataset(p)
        mesh = HealpixReader.load_mesh(_mesh_cfg(p))
        vf = HealpixReader.read_velocities(ds, 0, mesh)
        np.testing.assert_array_equal(vf.eta, 0.0)

    def test_var_eta_override_lives_in_eta_store(self, tmp_path):
        # regression (caught on the real ng5 store by `kelp info`):
        # mode.var_eta names a variable of the ETA store — its absence
        # from the main store must route to the second store, not KeyError
        p = _make_uv_store(tmp_path / "3D_ovr_eta.nc")
        pe = _make_eta_store(tmp_path / "eta_ovr.nc")
        mode = ModeConfig(var_eta="avg_zos")
        ds = HealpixReader.open_dataset(p, mode_cfg=mode, eta_path=str(pe))
        assert "avg_zos" in ds.variables
        # ...but a var_eta that exists NOWHERE still fails loud
        with pytest.raises(KeyError, match="nope"):
            HealpixReader.open_dataset(p, mode_cfg=ModeConfig(var_eta="nope"),
                                       eta_path=str(pe))
        import kelp.readers.healpix as hx
        hx._WARNED.clear()
        p2 = _make_uv_store(tmp_path / "3D_ovr_none.nc")
        with pytest.raises(KeyError, match="var_eta"):
            HealpixReader.open_dataset(
                p2, mode_cfg=ModeConfig(var_eta="avg_zos"))

    def test_time_axis_mismatch_raises(self, tmp_path):
        p = _make_uv_store(tmp_path / "3D_tm.nc")
        pe = _make_eta_store(tmp_path / "eta_tm.nc",
                             times=("2021-06-01", "2021-06-02",
                                    "2021-06-03"))
        with pytest.raises(ValueError, match="time"):
            HealpixReader.open_dataset(p, eta_path=str(pe))

    def test_v_path_and_w_path_rejected(self, tmp_path):
        # u/v/w always live in ONE healpix store (ng5 3-D layout)
        p = _make_uv_store(tmp_path / "3D_vw.nc", with_eta=True)
        with pytest.raises(ValueError, match="ncinput_v"):
            HealpixReader.open_dataset(p, v_path="x.nc")
        with pytest.raises(ValueError, match="ncinput_w"):
            HealpixReader.open_dataset(p, w_path="w.nc")

    def test_kerchunk_path_detection(self):
        assert _is_kerchunk_path("a/b/store.parq")
        assert _is_kerchunk_path("a/b/refs.json")
        assert not _is_kerchunk_path("a/b/data.zarr")
        assert not _is_kerchunk_path("a/b/data.nc")


# ---------------------------------------------------------------------------
# read_velocities / read_scalar_fields
# ---------------------------------------------------------------------------

class TestReadVelocities:
    def test_values_level_pick_and_eta(self, tmp_path):
        p = _make_uv_store(tmp_path / "3D_val.nc", with_eta=True)
        mesh = HealpixReader.load_mesh(_mesh_cfg(p))
        ds = HealpixReader.open_dataset(p)
        vf = HealpixReader.read_velocities(ds, 0, mesh)
        assert vf.uv_layout == "node"
        np.testing.assert_allclose(vf.u, 0.4)
        np.testing.assert_allclose(vf.v, 0.0)
        np.testing.assert_allclose(vf.eta, 0.25)
        assert vf.w is None

    def test_sigma_layer_picks_level(self, tmp_path):
        # level 0 vs level 1 hold different values -> the pick is visible
        p = _make_uv_store(tmp_path / "3D_lay.nc", with_eta=True)
        ds = xr.open_dataset(p)
        u = ds["avg_uoe"].values
        u[:, 1, :] = np.where(np.isfinite(u[:, 1, :]), 9.9, np.nan)
        ds["avg_uoe"] = (("time", "level", "value"), u)
        p2 = tmp_path / "3D_lay2.nc"
        ds.to_netcdf(p2)
        mesh = HealpixReader.load_mesh(_mesh_cfg(p2))
        ds2 = HealpixReader.open_dataset(p2)
        mode = ModeConfig(sigma_layer=1)
        vf = HealpixReader.read_velocities(ds2, 0, mesh, mode_cfg=mode)
        np.testing.assert_allclose(vf.u, 9.9)

    def test_float32(self, tmp_path):
        p = _make_uv_store(tmp_path / "3D_f32.nc", with_eta=True)
        mesh = HealpixReader.load_mesh(_mesh_cfg(p))
        ds = HealpixReader.open_dataset(p)
        vf = HealpixReader.read_velocities(ds, 0, mesh,
                                           velocity_float32=True)
        assert vf.u.dtype == np.float32
        assert vf.eta.dtype == np.float64

    def test_nan_drift_warns_and_zeroes(self, tmp_path):
        import kelp.readers.healpix as hx
        p = _make_uv_store(tmp_path / "3D_nan.nc", with_eta=True)
        mesh = HealpixReader.load_mesh(_mesh_cfg(p))
        ds = xr.open_dataset(p)
        u = ds["avg_uoe"].values
        wet_pix = mesh._healpix_node_pix
        u[1, 0, wet_pix[5]] = np.nan          # mask drift at t=1
        u[1, 0, wet_pix[6]] = np.inf          # inf must be zeroed too
        ds["avg_uoe"] = (("time", "level", "value"), u)
        p2 = tmp_path / "3D_nan2.nc"
        ds.to_netcdf(p2)
        ds2 = HealpixReader.open_dataset(p2)
        hx._WARNED.clear()
        with pytest.warns(RuntimeWarning, match="mask drift"):
            vf = HealpixReader.read_velocities(ds2, 1, mesh)
        assert vf.u[5] == 0.0
        assert vf.u[6] == 0.0                 # not a huge finite value

    def test_massive_nan_fraction_raises(self, tmp_path):
        # > 5% non-finite wet nodes is not coastal drift — wrong store /
        # level / variable: fail loud instead of tracking on zeros
        p = _make_uv_store(tmp_path / "3D_bignan.nc", with_eta=True)
        mesh = HealpixReader.load_mesh(_mesh_cfg(p))
        ds = xr.open_dataset(p)
        u = ds["avg_uoe"].values
        u[1, 0, mesh._healpix_node_pix[: mesh.n_nodes // 2]] = np.nan
        ds["avg_uoe"] = (("time", "level", "value"), u)
        p2 = tmp_path / "3D_bignan2.nc"
        ds.to_netcdf(p2)
        ds2 = HealpixReader.open_dataset(p2)
        with pytest.raises(ValueError, match="non-finite"):
            HealpixReader.read_velocities(ds2, 1, mesh)

    def test_numeric_time_axes_need_matching_encoding(self):
        from kelp.readers.healpix import _validate_time_axes_match
        a = xr.Dataset(coords={"time": ("time", np.arange(3.0),
                                        {"units": "hours since 2020-01-01"})})
        b = xr.Dataset(coords={"time": ("time", np.arange(3.0),
                                        {"units": "days since 2020-01-01"})})
        with pytest.raises(ValueError, match="encodings"):
            _validate_time_axes_match(a, b, "eta store")
        b2 = xr.Dataset(coords={"time": ("time", np.arange(3.0),
                                         {"units": "hours since 2020-01-01"})})
        _validate_time_axes_match(a, b2, "eta store")   # same encoding: ok

    def test_var_override_missing_raises(self, tmp_path):
        p = _make_uv_store(tmp_path / "3D_ovr.nc", with_eta=True)
        mesh = HealpixReader.load_mesh(_mesh_cfg(p))
        ds = HealpixReader.open_dataset(p)
        with pytest.raises(KeyError, match="nope"):
            HealpixReader.read_velocities(
                ds, 0, mesh, mode_cfg=ModeConfig(var_u="nope"))

    def test_zoom_mismatch_raises(self, tmp_path):
        p4 = _make_uv_store(tmp_path / "3D_z4.nc", with_eta=True)
        p2 = _make_uv_store(tmp_path / "3D_z2.nc", nside=2, with_eta=True)
        mesh = HealpixReader.load_mesh(_mesh_cfg(p4))
        ds2 = HealpixReader.open_dataset(p2)
        with pytest.raises(ValueError, match="zoom"):
            HealpixReader.read_velocities(ds2, 0, mesh)

    def test_em_diffusion_without_kh_vars_reads_clean(self, tmp_path):
        """em/m1 is SUPPORTED since 2026-08-16 (see
        test_healpix_diffusion.py for the Kh wiring).  A store WITHOUT
        Kh variables reads fine with kh None — fesom2 parity; the
        split_rk kick then raises its own clear error at step time."""
        from kelp.core.config import DiffusionConfig
        p = _make_uv_store(tmp_path / "3D_em.nc", with_eta=True)
        mesh = HealpixReader.load_mesh(_mesh_cfg(p))
        ds = HealpixReader.open_dataset(p)
        diff = DiffusionConfig(enabled=True, scheme="em")
        vel = HealpixReader.read_velocities(ds, 0, mesh, diffusion=diff)
        assert vel.kh_zonal is None and vel.kh_meridional is None

    def test_scalar_fields(self, tmp_path):
        p = _make_uv_store(tmp_path / "3D_sc.nc", with_eta=True,
                           extra_vars=("avg_tos",))
        mesh = HealpixReader.load_mesh(_mesh_cfg(p))
        ds = HealpixReader.open_dataset(p)
        out = HealpixReader.read_scalar_fields(ds, 0, mesh, ["avg_tos"])
        lat_nodes = mesh.nodes_lat
        np.testing.assert_allclose(out["avg_tos"], lat_nodes * 0.1,
                                   atol=1e-6)
        with pytest.raises(ValueError, match="missing_var"):
            HealpixReader.read_scalar_fields(ds, 0, mesh, ["missing_var"])

    def test_ring_store_varying_field_per_node(self, tmp_path):
        # reviewer blind spot: the ring==nest test used a CONSTANT u, which
        # a transposed ring->nest compaction would also pass.  A
        # lat-varying field pins the index-map direction per node.
        p = _make_uv_store(tmp_path / "3D_ringvar.nc", order="ring",
                           extra_vars=("avg_tos",))
        mesh = HealpixReader.load_mesh(_mesh_cfg(p))
        ds = HealpixReader.open_dataset(p)
        out = HealpixReader.read_scalar_fields(ds, 0, mesh, ["avg_tos"])
        np.testing.assert_allclose(out["avg_tos"], mesh.nodes_lat * 0.1,
                                   atol=1e-6)

    def test_probe_uv_layout(self):
        assert HealpixReader.probe_uv_layout(None) == "node"


# ---------------------------------------------------------------------------
# registry + end-to-end tracking
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def _config(self, tmp_path, store, seeds, eta_path="", n_steps=2,
                **numerics_kw):
        seeds_path = tmp_path / "seeds.txt"
        with open(seeds_path, "w") as fh:
            for lo, la in seeds:
                fh.write(f"{lo} {la}\n")
        out = tmp_path / "traj.nc"
        return AppConfig(
            mode=ModeConfig(Fdrift2D=True, Fdrift3D=False),
            numerics=NumericsConfig(uv_interp="barycentric", **numerics_kw),
            mesh=MeshConfig(type="healpix", meshpath=str(store)),
            data=DataConfig(ncinput=str(store), inicloud=str(seeds_path),
                            ncinput_eta=str(eta_path) if eta_path else "",
                            output_path=str(out), time_averaging="instant"),
            time=TimeConfig(start_time_idx=0, n_steps=n_steps),
        ), out

    def test_registry(self):
        assert get_reader("healpix") is HealpixReader

    def test_run_tracking_analytic(self, tmp_path):
        from kelp.tracking import run_tracking
        from kelp.writers import load_trajectories
        store = _make_uv_store(tmp_path / "3D_run.nc", with_eta=True,
                               nside=8, u_value=0.4)
        seeds = [(10.0, 5.0), (-120.0, -20.0)]
        config, out = self._config(tmp_path, store, seeds)
        run_tracking(config)
        traj = load_trajectories(str(out))
        lon, lat = traj["lon"], traj["lat"]
        dt = 86400.0                          # daily files, auto-detected
        for i, (lo, la) in enumerate(seeds):
            dlon = 2 * 0.4 * dt / (np.cos(np.radians(la))
                                   * METERS_PER_DEG_LAT)
            assert abs(lat[i, -1] - la) < 1e-9
            assert abs(lon[i, -1] - (lo + dlon)) < 1e-6

    def test_temporal_interp_prefetch_two_store(self, tmp_path):
        # reviewer coverage gap: the lerped/_VelCache/prefetch path on a
        # merged two-store dataset (eta assigned from a second store)
        from kelp.tracking import run_tracking
        from kelp.writers import load_trajectories
        store = _make_uv_store(tmp_path / "3D_ti_ocean.nc", nside=8,
                               u_value=0.4)
        eta = _make_eta_store(tmp_path / "eta_ti.nc", nside=8)
        seeds = [(10.0, 5.0)]
        config, out = self._config(
            tmp_path, store, seeds, eta_path=eta, n_steps=4,
            dt=43200.0, temporal_interp=True, prefetch_size=2)
        run_tracking(config)
        traj = load_trajectories(str(out))
        lo, la = seeds[0]
        # constant u: the snapshot lerp is exact, 4 half-day steps
        dlon = 4 * 0.4 * 43200.0 / (np.cos(np.radians(la))
                                    * METERS_PER_DEG_LAT)
        assert abs(traj["lat"][0, -1] - la) < 1e-9
        assert abs(traj["lon"][0, -1] - (lo + dlon)) < 1e-6

    def test_leastsq_fails_fast(self, tmp_path):
        from kelp.tracking import run_tracking
        store = _make_uv_store(tmp_path / "3D_lsq.nc", with_eta=True,
                               nside=8)
        config, _ = self._config(tmp_path, store, [(10.0, 5.0)])
        config.numerics.uv_interp = "leastsq"
        with pytest.raises(ValueError, match="barycentric|leastsq|node"):
            run_tracking(config)


# ---------------------------------------------------------------------------
# real ng5 kerchunk store (skipped unless KELP_TEST_NG5_STORE points at one)
# ---------------------------------------------------------------------------

# Set KELP_TEST_NG5_STORE to an ng5 3-D HEALPix store; unset -> skipped.
_NG5_3D_Z7 = os.environ.get("KELP_TEST_NG5_STORE", "")


@pytest.mark.skipif(not (_NG5_3D_Z7 and os.path.exists(_NG5_3D_Z7)),
                    reason="ng5 kerchunk store not available (set KELP_TEST_NG5_STORE)")
class TestRealNg5Store:
    def test_open_and_read_surface(self):
        pytest.importorskip("gribscan")
        pytest.importorskip("dask")
        mesh = HealpixReader.load_mesh(_mesh_cfg(_NG5_3D_Z7))
        assert mesh.metadata["healpix_nside"] == 128
        ds = HealpixReader.open_dataset(_NG5_3D_Z7)
        assert "avg_zos" in ds.variables      # auto-derived 2-D sibling
        vf = HealpixReader.read_velocities(ds, 0, mesh)
        assert np.isfinite(vf.u).all()
        assert np.abs(vf.u).max() < 5.0       # sane ocean speeds
        assert np.isfinite(vf.eta).all()
