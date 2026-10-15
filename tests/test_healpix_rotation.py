"""Rotated-frame (pole-safe) tracking on HEALPix meshes (Phase 5b), with the
explicit caution that HEALPix is NOT
FESOM's native grid.

What is DIFFERENT from the fesom2 case, and how the design answers it:

* fesom2's rotated runs read a mesh the MODEL built in that frame; the
  HEALPix remap is geographic east/north on a global pixel grid, so
  rotation here is a pure runtime transform: rotate the pixel-centre
  NODES as points, rotate the velocity VECTORS with per-node tangent
  factors (exact `rotate_vectors` machinery), rebuild the search
  structures on the rotated coordinates, and let the generic tracking
  layer (seed rotation, geographic output, restart frame validation) do
  the rest.  Topology and pixel ids never change.
* the rotated frame has its own two poles: wherever they land, the
  polar pathologies (containment cap, wide-in-lon cells, grid-hash
  degradation) move THERE.  With the FESOM2 angles they land deep in
  Greenland (-40, 75) and East Antarctica (140, -75) — measured on the
  CORE2-z6 twin: wet nodes reach only |rotated lat| 85.3, zero above
  87, zero pole-straddling cells.  The loader GUARDS the bad
  configurations: a wet cell straddling a rotated pole is a hard
  ValueError (its containment chart would lie), wet nodes above 88
  rotated degrees warn once.  An ALL-WET synthetic globe therefore
  deliberately fails — tests carve the rotated caps dry, exactly like
  the real ocean mask does.

The headline pin reuses the fesom2 transpolar construction: a field
CONSTANT in the rotated frame (u_r=0, v_r=V), written to the store as
its geographic image via `unrotate_vectors` per pixel — the tracker
must recover exact rotated-meridian motion (RK2 has zero truncation on
a constant RHS), i.e. a geographic GREAT CIRCLE through the pole region
at machine precision.
"""
import sys
import warnings as _warnings
from pathlib import Path
from types import SimpleNamespace

import os

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kelp.core.config import NumericsConfig  # noqa: E402
from kelp.core.geometry import (METERS_PER_DEG_LAT,  # noqa: E402
                                    R_EARTH_STANDARD)
from kelp.core.particles import ParticleStatus, make_state  # noqa: E402
from kelp.core.rotation import (FESOM2_ANGLES, euler_matrix,  # noqa: E402
                                    rotate_lonlat, unrotate_lonlat,
                                    unrotate_vectors)
from kelp.readers import get_reader  # noqa: E402
from kelp.tracking import run_single_realization  # noqa: E402

from test_healpix_reader import _pix_lonlat  # noqa: E402

NSIDE = 16                       # ~3.7 deg pixels — fast, pole-covering


def _mode():
    return SimpleNamespace(Fdrift2D=True, Fdrift3D=False, time_dir=1.0,
                           sigma_layer=None, var_u=None, var_v=None,
                           var_w=None, var_eta=None, sample_vars=[],
                           max_levels=None)


def _mesh_cfg(path, rotation=None):
    return SimpleNamespace(meshpath=str(path), healpix_zoom=None,
                           crop_bbox=None, levels=None, rotation=rotation,
                           r_earth=None)


def _make_rotated_safe_store(path, *, u_r=0.0, v_r=0.0, nside=NSIDE,
                             cap_deg=86.0):
    """Store whose wet mask keeps the ROTATED caps dry (like a real
    ocean mask does) and whose u/v are the geographic image of a field
    CONSTANT in the rotated frame."""
    npix = 12 * nside * nside
    lon, lat = _pix_lonlat(nside, "nest")
    R = euler_matrix(*FESOM2_ANGLES)
    _, rlat = rotate_lonlat(lon, lat, R)
    wet = np.abs(rlat) < cap_deg
    u_geo, v_geo = unrotate_vectors(np.full(npix, u_r),
                                    np.full(npix, v_r), lon, lat, R)
    u = np.where(wet, u_geo, np.nan)
    v = np.where(wet, v_geo, np.nan)
    ntime = 3
    ds = xr.Dataset(
        {
            "uo": (("time", "cell"), np.repeat(u[None, :], ntime, axis=0)),
            "vo": (("time", "cell"), np.repeat(v[None, :], ntime, axis=0)),
            "ssh": (("time", "cell"),
                    np.where(wet, 0.0, np.nan)[None, :].repeat(ntime, 0)),
        },
        coords={"time": np.arange(ntime) * 86400.0,
                "lon": ("cell", lon), "lat": ("cell", lat)},
    )
    ds["crs"] = ((), np.int32(0), {"grid_mapping_name": "healpix",
                                   "healpix_nside": nside,
                                   "healpix_order": "nest"})
    ds.to_netcdf(path)
    return path


def _load(path, rotation=None):
    reader = get_reader("healpix")
    mesh = reader.load_mesh(_mesh_cfg(path, rotation), _mode())
    ds = reader.open_dataset(str(path))
    return reader, mesh, ds


class TestRotatedMeshBuild:
    def test_none_rotation_byte_identical(self, tmp_path):
        p = _make_rotated_safe_store(tmp_path / "s.nc", u_r=0.1)
        _, m0, ds0 = _load(p)
        ds0.close()
        _, m1, ds1 = _load(p, rotation=None)
        ds1.close()
        assert m1.rotation_angles is None
        np.testing.assert_array_equal(m0.nodes_lon, m1.nodes_lon)
        np.testing.assert_array_equal(m0.nodes_lat, m1.nodes_lat)

    def test_nodes_rotated_as_points_topology_identical(self, tmp_path):
        p = _make_rotated_safe_store(tmp_path / "s.nc", u_r=0.1)
        _, mg, dsg = _load(p)
        dsg.close()
        _, mr, dsr = _load(p, rotation="fesom2")
        dsr.close()
        assert mr.rotation_angles == FESOM2_ANGLES
        R = euler_matrix(*FESOM2_ANGLES)
        exp_lon, exp_lat = rotate_lonlat(mg.nodes_lon, mg.nodes_lat, R)
        np.testing.assert_allclose(mr.nodes_lon, exp_lon, atol=1e-9)
        np.testing.assert_allclose(mr.nodes_lat, exp_lat, atol=1e-9)
        # topology + pixel ids untouched — the same discrete object in
        # another chart
        np.testing.assert_array_equal(mr.elem_nodes, mg.elem_nodes)
        np.testing.assert_array_equal(mr._healpix_node_pix,
                                      mg._healpix_node_pix)
        # node vector factors cached, unit-norm
        c, s = mr._vec_rot_factors["node"]
        np.testing.assert_allclose(np.hypot(c, s), 1.0, atol=1e-12)

    def test_custom_angle_list_accepted(self, tmp_path):
        p = _make_rotated_safe_store(tmp_path / "s.nc", u_r=0.1)
        # the same FESOM2 angles as an explicit list
        _, mr, dsr = _load(p, rotation=(50.0, 15.0, -90.0))
        dsr.close()
        assert mr.rotation_angles == (50.0, 15.0, -90.0)

    def test_search_structures_rebuilt_on_rotated_coords(self, tmp_path):
        from kelp.core.mesh import find_nearest_nodes
        p = _make_rotated_safe_store(tmp_path / "s.nc", u_r=0.1)
        _, mr, dsr = _load(p, rotation="fesom2")
        dsr.close()
        # querying AT a rotated node position returns that node — only
        # true when the grid hash was rebuilt after rotation
        k = 100
        nn = find_nearest_nodes(mr, mr.nodes_lon[k:k + 1],
                                mr.nodes_lat[k:k + 1])
        assert nn[0] == k


class TestGuards:
    def test_all_wet_globe_rejected(self, tmp_path):
        """A wet rotated pole (all-wet globe) means pole-straddling wet
        cells whose containment chart would lie — hard error, with a
        message naming the cure (angles / mask)."""
        p = _make_rotated_safe_store(tmp_path / "s.nc", u_r=0.1,
                                     cap_deg=91.0)      # nothing dry
        with pytest.raises(ValueError, match="rotated pole|straddl"):
            _load(p, rotation="fesom2")

    def test_near_pole_wet_nodes_warn(self, tmp_path):
        """Wet nodes high in the rotated frame (but no straddling cell)
        get the one-time degradation warning."""
        import kelp.readers.healpix as hx
        # nside 64: pixel rings reach 89.27 rotated deg — a cap at 89.1
        # keeps wet nodes ABOVE the 88-deg warn line while leaving the
        # polemost ring dry (no straddling cell -> no hard error)
        p = _make_rotated_safe_store(tmp_path / "s.nc", u_r=0.1,
                                     nside=64, cap_deg=89.1)
        hx._WARNED.clear()
        with pytest.warns(RuntimeWarning, match="ROTATED"):
            _, mr, dsr = _load(p, rotation="fesom2")
            dsr.close()

    def test_real_mask_silent(self, tmp_path):
        """The realistic configuration (caps dry, like the CORE2 twin
        where wet max |rotated lat| = 85.3) loads with no warning."""
        import kelp.readers.healpix as hx
        p = _make_rotated_safe_store(tmp_path / "s.nc", u_r=0.1,
                                     cap_deg=86.0)
        hx._WARNED.clear()     # an earlier test may have burned the key
        with _warnings.catch_warnings():
            # narrow to the reader's own warnings: a blanket error filter
            # trips on netCDF4's import-time RuntimeWarning under
            # targeted invocation (review blocker)
            _warnings.filterwarnings("error", category=RuntimeWarning,
                                     message=".*healpix reader.*")
            _, mr, dsr = _load(p, rotation="fesom2")
            dsr.close()
        assert mr.rotation_angles == FESOM2_ANGLES


class TestVelocityRotation:
    def test_field_constant_in_rotated_frame_reads_constant(self, tmp_path):
        """The store carries the GEOGRAPHIC image of (u_r, v_r) =
        (0.3, -0.2); reading through the rotated mesh must recover
        exactly those rotated components at every node (per-pixel
        rotate_vectors exactness — the point flagged in review as the danger
        zone of a non-native grid)."""
        p = _make_rotated_safe_store(tmp_path / "s.nc", u_r=0.3, v_r=-0.2)
        reader, mr, ds = _load(p, rotation="fesom2")
        vel = reader.read_velocities(ds, 0, mr, _mode())
        ds.close()
        np.testing.assert_allclose(vel.u, 0.3, atol=1e-12)
        np.testing.assert_allclose(vel.v, -0.2, atol=1e-12)

    def test_unrotated_read_unchanged(self, tmp_path):
        p = _make_rotated_safe_store(tmp_path / "s.nc", u_r=0.3, v_r=-0.2)
        reader, mg, ds = _load(p)
        vel = reader.read_velocities(ds, 0, mg, _mode())
        ds.close()
        # geographic components vary node-to-node (the image of a
        # rotated-constant field is NOT constant in the geo frame)
        assert np.std(vel.u) > 0.01


class TestTranspolarTracking:
    def test_transpolar_great_circle_machine_precision(self, tmp_path):
        """Constant rotated-meridional flow: RK2 has ZERO truncation on
        a constant RHS, so the rotated-frame motion is exact and the
        geographic output must follow the great circle through the pole
        region to machine precision (the fesom2 transpolar pin, now on
        a HEALPix mesh — geo seeds at 87.5N cross 89.9N territory that
        used to be the containment-cap death zone)."""
        dt, n_steps = 3600.0, 20
        V = 6.0 * METERS_PER_DEG_LAT / (n_steps * dt)   # 6 deg of lat_r
        p = _make_rotated_safe_store(tmp_path / "s.nc", u_r=0.0, v_r=V)
        reader, mr, ds = _load(p, rotation="fesom2")
        R = euler_matrix(*FESOM2_ANGLES)
        # seed on the rotated meridian through the geo pole, 3 deg below
        # its image (rotated lat 75): crosses geo-pole-adjacent latitudes
        seed_lon, seed_lat = unrotate_lonlat(np.array([-180.0]),
                                             np.array([72.0]), R)
        # track_particles is the USER entry point: it rotates the
        # GEOGRAPHIC seeds into the mesh frame (run_single_realization
        # treats its init_state as already in-frame)
        from kelp.tracking import track_particles
        rec = track_particles(
            mr, ds, reader, seed_lon.copy(), seed_lat.copy(),
            start_time_idx=0, n_steps=n_steps, on_exhausted="cycle",
            dt=dt, integrator="rk2", output_freq=1, mode=_mode(),
            uv_interp="barycentric", temporal_interp=True)
        ds.close()
        arrs = rec.to_arrays()
        assert np.all(arrs["status"][:, 0] == ParticleStatus.ALIVE)
        # analytic: rotated lon fixed at -180, rotated lat 72 + k*0.3 deg
        M = METERS_PER_DEG_LAT
        for k in range(n_steps):
            lat_r_k = 72.0 + (k + 1) * dt * V / M
            elon, elat = unrotate_lonlat(np.array([-180.0]),
                                         np.array([lat_r_k]), R)
            def _u(lo, la):
                lo, la = np.deg2rad(lo), np.deg2rad(la)
                return np.array([np.cos(la) * np.cos(lo),
                                 np.cos(la) * np.sin(lo), np.sin(la)])
            chord = np.linalg.norm(_u(arrs["lon"][k, 0], arrs["lat"][k, 0])
                                   - _u(elon[0], elat[0])) * R_EARTH_STANDARD
            # measured 6.8e-8 m — pin at the advertised machine-precision
            # scale, not the inherited 1-metre slack (review)
            assert chord < 1e-4, (k, chord)
        # the trajectory genuinely visited the deep polar zone
        assert arrs["lat"][:, 0].max() > 87.0


_Z6 = os.path.join(os.environ.get("KELP_TEST_DATA", ""),   # set KELP_TEST_DATA to the twin-store directory
                   "lagdrift_core2_2005_z6.nc")  # kept on purpose: a real name on disk / in the sibling clone, NOT renamed with the package


@pytest.mark.skipif(not (os.environ.get("KELP_TEST_DATA") and Path(_Z6).exists()),
                    reason="CORE2 twin store not available (set KELP_TEST_DATA)")
class TestRealStoreArctic:
    def test_z6_rotated_arctic_tracking(self):
        """The CORE2 twin under rotation='fesom2': loads silently (wet
        max |rotated lat| measured 85.3), and Arctic seeds at 85-89N —
        the previously degraded cold-anchor/cap zone — track 5 daily
        steps alive with geographic output in the Arctic."""
        import kelp.readers.healpix as hx
        reader = get_reader("healpix")
        hx._WARNED.clear()
        with _warnings.catch_warnings():
            _warnings.filterwarnings("error", category=RuntimeWarning,
                                     message=".*healpix reader.*")
            mesh = reader.load_mesh(_mesh_cfg(_Z6, rotation="fesom2"),
                                    _mode())
        ds = reader.open_dataset(_Z6)
        rng = np.random.default_rng(9)
        from kelp.tracking import track_particles
        rec = track_particles(
            mesh, ds, reader,
            rng.uniform(-180.0, 180.0, 12), rng.uniform(85.0, 89.0, 12),
            start_time_idx=0, n_steps=5, dt=86400.0, integrator="rk2",
            output_freq=1, mode=_mode(), uv_interp="barycentric")
        ds.close()
        arrs = rec.to_arrays()
        alive = arrs["status"][-1] == ParticleStatus.ALIVE
        # DISCRIMINATING threshold (review blocker: >= 9 was satisfied
        # by the geographic frame too — measured geo 9/12, rotated
        # 12/12): the rotated frame keeps EVERY seed alive
        assert alive.sum() == 12
        assert np.all(arrs["lat"][-1][alive] > 80.0)

    def test_z6_rotated_passes_density_certificate(self):
        """Side effect worth pinning: in the ROTATED chart the Arctic
        pixels are ordinary-sized, the polar rows sit over dry land, and
        the grid-hash density certificate PASSES (it is denied on the
        geographic frame) — so rotated HEALPix runs get the fused
        barycentric kernels.  A property of this store's geometry, not
        a general healpix contract."""
        from kelp.core.mesh import grid_hash_dense_for_elements
        reader = get_reader("healpix")
        mesh = reader.load_mesh(_mesh_cfg(_Z6, rotation="fesom2"),
                                _mode())
        assert grid_hash_dense_for_elements(mesh) is True


class TestAnisotropicKhRotationWarning:
    """Isotropic Kh is the normal case and is exactly
    frame-invariant (including spatially-varying Kh — the Ito-drift
    gradient is computed natively in the rotated chart and transforms as
    a vector); an ANISOTROPIC pair rides the rotated axes instead of
    geographic ones, so runs must WARN, not just document."""

    def _store_with_kh(self, tmp_path, iso=True):
        p = _make_rotated_safe_store(tmp_path / "s.nc", u_r=0.1)
        with xr.open_dataset(p) as ds0:
            ds0 = ds0.load()
        lon = ds0["lon"].values
        wet = np.isfinite(ds0["uo"].isel(time=0).values)
        kx = np.where(wet, 500.0 + 0.1 * lon, np.nan)
        ky = kx if iso else kx * 0.1
        nt = ds0.sizes["time"]
        ds0["Kh_zonal"] = (("time", "cell"), np.repeat(kx[None], nt, 0))
        ds0["Kh_meridional"] = (("time", "cell"),
                                np.repeat(ky[None], nt, 0))
        ds0.to_netcdf(p, mode="w")
        return p

    def test_rotated_anisotropic_fields_warn(self, tmp_path):
        from kelp.core.config import DiffusionConfig
        import kelp.readers.healpix as hx
        p = self._store_with_kh(tmp_path, iso=False)
        reader, mr, ds = _load(p, rotation="fesom2")
        hx._WARNED.clear()
        diff = DiffusionConfig(enabled=True, scheme="em",
                               composition="split_rk", seed=1)
        with pytest.warns(RuntimeWarning, match="(?i)anisotrop"):
            reader.read_velocities(ds, 0, mr, _mode(), diffusion=diff)
        ds.close()

    def test_rotated_isotropic_fields_silent(self, tmp_path):
        """Spatially-VARYING but isotropic Kh under rotation: the normal
        configuration — no warning (frame-invariant, incl. the drift
        gradient computed in the rotated chart)."""
        from kelp.core.config import DiffusionConfig
        import kelp.readers.healpix as hx
        p = self._store_with_kh(tmp_path, iso=True)
        reader, mr, ds = _load(p, rotation="fesom2")
        hx._WARNED.clear()
        diff = DiffusionConfig(enabled=True, scheme="em",
                               composition="split_rk", seed=1)
        with _warnings.catch_warnings():
            _warnings.filterwarnings("error", category=RuntimeWarning,
                                     message="(?i).*anisotrop.*")
            vel = reader.read_velocities(ds, 0, mr, _mode(),
                                         diffusion=diff)
        ds.close()
        assert vel.kh_zonal is not None

    def test_unrotated_anisotropic_fields_silent(self, tmp_path):
        from kelp.core.config import DiffusionConfig
        import kelp.readers.healpix as hx
        p = self._store_with_kh(tmp_path, iso=False)
        reader, mg, ds = _load(p)                # no rotation
        hx._WARNED.clear()
        diff = DiffusionConfig(enabled=True, scheme="em",
                               composition="split_rk", seed=1)
        with _warnings.catch_warnings():
            _warnings.filterwarnings("error", category=RuntimeWarning,
                                     message="(?i).*anisotrop.*")
            reader.read_velocities(ds, 0, mg, _mode(), diffusion=diff)
        ds.close()

    def test_uniform_scalar_anisotropy_warns_at_run_level(self, tmp_path):
        """The uniform scheme's config SCALARS have the same tensor
        semantics — the tracking layer warns once per run."""
        from kelp.core.config import DiffusionConfig
        p = _make_rotated_safe_store(tmp_path / "s.nc", u_r=0.1)
        reader, mr, ds = _load(p, rotation="fesom2")
        from kelp.tracking import track_particles
        diff = DiffusionConfig(enabled=True, scheme="uniform",
                               kh_zonal=100.0, kh_meridional=10.0, seed=2)
        with pytest.warns(RuntimeWarning, match="(?i)anisotrop"):
            track_particles(mr, ds, reader, np.array([10.0]),
                            np.array([30.0]), start_time_idx=0, n_steps=1,
                            on_exhausted="cycle", dt=3600.0,
                            integrator="rk2", output_freq=1, mode=_mode(),
                            uv_interp="barycentric", diffusion=diff, temporal_interp=True)
        ds.close()

    def test_uniform_scalar_isotropy_silent(self, tmp_path):
        from kelp.core.config import DiffusionConfig
        p = _make_rotated_safe_store(tmp_path / "s.nc", u_r=0.1)
        reader, mr, ds = _load(p, rotation="fesom2")
        from kelp.tracking import track_particles
        diff = DiffusionConfig(enabled=True, scheme="uniform",
                               kh_zonal=100.0, kh_meridional=100.0, seed=2)
        with _warnings.catch_warnings():
            _warnings.filterwarnings("error", category=RuntimeWarning,
                                     message="(?i).*anisotrop.*")
            track_particles(mr, ds, reader, np.array([10.0]),
                            np.array([30.0]), start_time_idx=0, n_steps=1,
                            on_exhausted="cycle", dt=3600.0,
                            integrator="rk2", output_freq=1, mode=_mode(),
                            uv_interp="barycentric", diffusion=diff, temporal_interp=True)
        ds.close()
