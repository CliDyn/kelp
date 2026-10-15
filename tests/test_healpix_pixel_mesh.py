"""HEALPix pixels-as-elements mesh (docs/healpix.md §1b): nodes = deduped pixel
corners, elements = the wet pixels themselves, u/v at pixel centres =
ELEMENT layout -> the fused leastsq kernels apply.

Invariants pinned here:
- all-wet globe: E = npix and V (corner nodes) = npix + 2 (Euler:
  V - E_edges + F = 2 with F = npix, E_edges = 2*npix);
- every element is a quad of 4 distinct nodes, equal-area to near machine
  precision (HEALPix property);
- coastal pixels keep ALL their corners (unlike the dual mesh, where a
  corner cell with a dry member is dropped);
- containment ground truth: healpy.ang2pix says which pixel (= element)
  contains a point;
- the 8 pole-touching pixels are chart-degenerate in the geographic frame:
  kept, listed in metadata, warned about once.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kelp.core.mesh import (build_search, build_elem_areas,  # noqa: E402
                                find_containing_elements,
                                find_nearest_nodes, find_nonconvex_quads)
from kelp.readers.healpix import build_healpix_pixel_mesh  # noqa: E402

NSIDE = 8
NPIX = 12 * NSIDE * NSIDE


def _pix_lonlat(nside, pix):
    lon, lat = healpy.pix2ang(nside, pix, nest=True, lonlat=True)
    return np.asarray(lon), np.asarray(lat)


def _wet_band(lat, lo=-60.0, hi=60.0):
    return (lat > lo) & (lat < hi)


def _contain(mesh, lons, lats):
    nn = find_nearest_nodes(mesh, lons, lats)
    return find_containing_elements(mesh, lons, lats, nn)


@pytest.fixture(scope="module")
def allwet():
    with pytest.warns(RuntimeWarning, match="pole-touching"):
        return build_healpix_pixel_mesh(NSIDE, order="nest")


@pytest.fixture(scope="module")
def band():
    lon, lat = _pix_lonlat(NSIDE, np.arange(NPIX))
    wet = _wet_band(lat)
    return build_healpix_pixel_mesh(NSIDE, order="nest",
                                    wet_mask=wet), wet


class TestBuilderAllWet:
    def test_counts_euler(self, allwet):
        assert allwet.n_elements == NPIX
        assert allwet.n_nodes == NPIX + 2        # Euler on the sphere
        assert allwet.elem_nodes.shape[1] == 4
        assert (allwet.elem_nodes >= 0).all()    # quads everywhere

    def test_no_duplicate_nodes_within_element(self, allwet):
        en = allwet.elem_nodes
        for i in range(4):
            for j in range(i + 1, 4):
                assert not (en[:, i] == en[:, j]).any()

    def test_near_equal_areas(self, allwet):
        # HEALPix pixel BOUNDARIES are not great circles: the quad through
        # the 4 corners only approximates the true (exactly equal-area)
        # pixel, with an O(h^2) geometric mismatch — at nside 8 the quad
        # areas sit within ~3 % of 4*pi/npix (they tighten with nside).
        areas = build_elem_areas(allwet)
        expect = 4.0 * np.pi / NPIX * allwet.r_earth ** 2
        polar = allwet.metadata["healpix_polar_degenerate"]
        keep = np.ones(allwet.n_elements, bool)
        keep[polar] = False
        np.testing.assert_allclose(areas[keep], expect, rtol=0.03)

    def test_nonconvex_only_at_poles(self, allwet):
        bad = find_nonconvex_quads(allwet.elem_nodes, allwet.nodes_lon,
                                   allwet.nodes_lat)
        polar = set(allwet.metadata["healpix_polar_degenerate"].tolist())
        assert set(bad.tolist()) <= polar
        assert len(polar) == 8                   # 4 pixels at each pole

    def test_element_centres_are_pixel_centres(self, allwet):
        lon, lat = _pix_lonlat(NSIDE, np.arange(NPIX))
        np.testing.assert_allclose(allwet.elem_center_lat, lat, atol=1e-9)
        dlon = (allwet.elem_center_lon - ((lon + 180) % 360 - 180)
                + 540) % 360 - 180
        np.testing.assert_allclose(dlon, 0.0, atol=1e-9)


class TestBuilderCoastal:
    def test_coastal_corners_survive(self, band):
        mesh, wet = band
        assert mesh.n_elements == int(wet.sum())
        assert (mesh.elem_nodes >= 0).all()      # full quads at the coast
        bad = find_nonconvex_quads(mesh.elem_nodes, mesh.nodes_lon,
                                   mesh.nodes_lat)
        assert bad.size == 0                     # no poles in the band
        assert mesh.metadata["healpix_polar_degenerate"].size == 0

    def test_depth_aggregated_from_wet_pixels(self):
        lon, lat = _pix_lonlat(NSIDE, np.arange(NPIX))
        wet = _wet_band(lat)
        depth = np.where(wet, 1000.0, 0.0)
        mesh = build_healpix_pixel_mesh(NSIDE, order="nest", wet_mask=wet,
                                        depth=depth)
        assert mesh.depth.shape == (mesh.n_nodes,)
        np.testing.assert_allclose(mesh.depth, 1000.0)


class TestContainmentVsAng2pix:
    def test_containing_element_is_ang2pix(self, band):
        mesh, wet = band
        rng = np.random.default_rng(7)
        qlat = np.degrees(np.arcsin(rng.uniform(-0.7, 0.7, 400)))
        qlon = rng.uniform(-180.0, 180.0, 400)
        true_pix = healpy.ang2pix(NSIDE, qlon, qlat, nest=True, lonlat=True)
        elems, _ = _contain(mesh, qlon, qlat)
        elem_pix = mesh.metadata["healpix_elem_pix"]
        in_wet = wet[true_pix]
        ok = elems >= 0
        assert ok[in_wet].mean() > 0.98
        m = ok & in_wet
        got, true = elem_pix[elems[m]], true_pix[m]
        agree = got == true
        # near-edge points may land in the adjacent quad: the great-circle
        # quad edge sags off the true (non-great-circle) pixel boundary by
        # O(h^2) — every disagreement must be an ADJACENT pixel, never a
        # distant one, and they stay rare.
        assert agree.mean() > 0.98
        for g, t in zip(got[~agree], true[~agree]):
            nbrs = healpy.get_all_neighbours(NSIDE, int(t), nest=True)
            assert int(g) in set(int(x) for x in nbrs), (g, t)

    def test_pixel_centres_contained_in_own_pixel(self, band):
        mesh, wet = band
        pix = np.flatnonzero(wet)[::7]
        lon, lat = _pix_lonlat(NSIDE, pix)
        elems, _ = _contain(mesh, lon, lat)
        assert (elems >= 0).all()
        assert (mesh.metadata["healpix_elem_pix"][elems] == pix).all()


class TestGuards:
    def test_ring_order_supported(self, allwet):
        with pytest.warns(RuntimeWarning, match="pole-touching"):
            m_ring = build_healpix_pixel_mesh(
                NSIDE, order="ring", wet_mask=np.ones(NPIX, bool))
        assert m_ring.n_elements == allwet.n_elements
        assert m_ring.n_nodes == allwet.n_nodes

    def test_no_wet_pixels_raises(self):
        with pytest.raises(ValueError, match="no ocean"):
            build_healpix_pixel_mesh(NSIDE, order="nest",
                                     wet_mask=np.zeros(NPIX, bool))


# ---------------------------------------------------------------------------
# Reader wiring: mesh.healpix_geometry = "pixel"
# ---------------------------------------------------------------------------

from kelp.core.config import MeshConfig, ModeConfig  # noqa: E402
from kelp.readers.healpix import HealpixReader  # noqa: E402
from test_healpix_reader import (_make_uv_store, _mesh_cfg,  # noqa: E402
                                 NSIDE as RD_NSIDE)

MODE_2D = ModeConfig(Fdrift2D=True, Fdrift3D=False)
MODE_3D = ModeConfig(Fdrift2D=False, Fdrift3D=True)


class TestConfigKey:
    def test_default_dual_and_validation(self):
        assert MeshConfig(type="healpix", meshpath="x").healpix_geometry == "dual"
        MeshConfig(type="healpix", meshpath="x", healpix_geometry="pixel")
        with pytest.raises(ValueError, match="healpix_geometry"):
            MeshConfig(type="healpix", meshpath="x",
                       healpix_geometry="voronoi")


class TestLoadMeshPixel:
    def test_pixel_geometry_mesh(self, tmp_path):
        p = _make_uv_store(tmp_path / "3D_t_ocean.nc")
        mesh = HealpixReader.load_mesh(
            _mesh_cfg(p, healpix_geometry="pixel"), mode_cfg=MODE_2D)
        assert mesh.metadata["healpix_geometry"] == "pixel"
        assert mesh.n_elements == mesh.metadata["healpix_elem_pix"].size
        # data compaction targets the ELEMENT pixels
        np.testing.assert_array_equal(mesh._healpix_data_idx,
                                      mesh._healpix_elem_pix)

    def test_3d_not_implemented(self, tmp_path):
        p = _make_uv_store(tmp_path / "3D_t_ocean.nc")
        with pytest.raises(NotImplementedError, match="pixel"):
            HealpixReader.load_mesh(
                _mesh_cfg(p, healpix_geometry="pixel", levels=[0, 10, 20]),
                mode_cfg=MODE_3D)

    def test_probe_reports_element(self, tmp_path):
        import xarray as xr
        p = _make_uv_store(tmp_path / "3D_t_ocean.nc")
        mesh = HealpixReader.load_mesh(
            _mesh_cfg(p, healpix_geometry="pixel"), mode_cfg=MODE_2D)
        ds = HealpixReader.open_dataset(p, mode_cfg=MODE_2D)
        assert HealpixReader.probe_uv_layout(ds, MODE_2D, mesh) == "element"
        mesh_d = HealpixReader.load_mesh(_mesh_cfg(p), mode_cfg=MODE_2D)
        assert HealpixReader.probe_uv_layout(ds, MODE_2D, mesh_d) == "node"


class TestReadVelocitiesPixel:
    def _mesh_ds(self, tmp_path, **cfg_kw):
        p = _make_uv_store(tmp_path / "3D_t_ocean.nc", u_value=0.4)
        mesh = HealpixReader.load_mesh(
            _mesh_cfg(p, healpix_geometry="pixel", **cfg_kw),
            mode_cfg=MODE_2D)
        ds = HealpixReader.open_dataset(p, mode_cfg=MODE_2D)
        return p, mesh, ds

    def test_element_layout_and_values(self, tmp_path):
        _, mesh, ds = self._mesh_ds(tmp_path)
        vel = HealpixReader.read_velocities(ds, 0, mesh, mode_cfg=MODE_2D)
        assert vel.uv_layout == "element"
        assert vel.u.shape == (mesh.n_elements,)
        np.testing.assert_allclose(vel.u, 0.4)
        assert vel.eta.shape == (mesh.n_nodes,)   # corner-aggregated

    def test_eta_aggregated_to_corners(self, tmp_path):
        p = _make_uv_store(tmp_path / "3D_t_ocean.nc", with_eta=True)
        mesh = HealpixReader.load_mesh(
            _mesh_cfg(p, healpix_geometry="pixel"), mode_cfg=MODE_2D)
        ds = HealpixReader.open_dataset(p, mode_cfg=MODE_2D)
        vel = HealpixReader.read_velocities(ds, 0, mesh, mode_cfg=MODE_2D)
        # the fixture writes a constant ssh over wet pixels -> constant
        # at every corner after incident-wet-mean aggregation
        finite = np.isfinite(vel.eta)
        assert finite.all()
        assert np.allclose(vel.eta, vel.eta[0])

    def test_sample_vars_not_implemented(self, tmp_path):
        p = _make_uv_store(tmp_path / "3D_t_ocean.nc", with_theta=True)
        mesh = HealpixReader.load_mesh(
            _mesh_cfg(p, healpix_geometry="pixel"), mode_cfg=MODE_2D)
        ds = HealpixReader.open_dataset(p, mode_cfg=MODE_2D)
        with pytest.raises(NotImplementedError, match="sample"):
            HealpixReader.read_velocities(ds, 0, mesh, mode_cfg=MODE_2D,
                                          sample_vars=["avg_thetao"])

    def test_rotation_uses_element_factors(self, tmp_path):
        from kelp.core.rotation import (FESOM2_ANGLES, euler_matrix,
                                            rotate_vectors)
        import xarray as xr
        p = _make_uv_store(tmp_path / "3D_t_ocean.nc", u_value=0.3)
        # dry BOTH rotated-pole neighbourhoods (the default fixture keeps
        # the whole south wet -> the rotated south pole sits in water and
        # the straddle guard rightly refuses to build)
        with xr.open_dataset(p) as d0:
            d = d0.load()
        lat = np.asarray(d["lat"]) if "lat" in d else None
        import healpy as _hp
        _, plat = _hp.pix2ang(RD_NSIDE, np.arange(12 * RD_NSIDE ** 2),
                              nest=True, lonlat=True)
        cap = np.abs(plat) > 55.0
        for vn in ("avg_uoe", "avg_von"):
            arr = d[vn].values
            arr[..., cap] = np.nan
            d[vn].values = arr
        d.to_netcdf(p, mode="w")
        mesh_g = HealpixReader.load_mesh(
            _mesh_cfg(p, healpix_geometry="pixel"), mode_cfg=MODE_2D)
        geo_lon = mesh_g.elem_center_lon.copy()
        geo_lat = mesh_g.elem_center_lat.copy()
        mesh = HealpixReader.load_mesh(
            _mesh_cfg(p, healpix_geometry="pixel", rotation="fesom2"),
            mode_cfg=MODE_2D)
        assert "element" in mesh._vec_rot_factors
        ds = HealpixReader.open_dataset(p, mode_cfg=MODE_2D)
        vel = HealpixReader.read_velocities(ds, 0, mesh, mode_cfg=MODE_2D)
        R = euler_matrix(*FESOM2_ANGLES)
        ur, vr = rotate_vectors(np.full(geo_lon.size, 0.3),
                                np.zeros(geo_lon.size), geo_lon, geo_lat, R)
        np.testing.assert_allclose(vel.u, ur, atol=1e-12)
        np.testing.assert_allclose(vel.v, vr, atol=1e-12)


class TestTrackingPixelSmoke:
    def test_uniform_east_flow_fused_leastsq(self, tmp_path):
        from kelp.tracking import track_particles
        from kelp.core.geometry import METERS_PER_DEG_LAT
        p = _make_uv_store(tmp_path / "3D_t_ocean.nc", u_value=0.5)
        mesh = HealpixReader.load_mesh(
            _mesh_cfg(p, healpix_geometry="pixel"), mode_cfg=MODE_2D)
        ds = HealpixReader.open_dataset(p, mode_cfg=MODE_2D)
        lon0 = np.array([10.0, -40.0, 120.0])
        lat0 = np.array([5.0, -20.0, 30.0])
        rec = track_particles(mesh, ds, HealpixReader, lon0, lat0,
                              mode=MODE_2D, n_steps=2, dt=3600.0,
                              uv_interp="leastsq", temporal_interp=True)
        arr = rec.to_arrays()          # axes: (obs, particle)
        lon, st = arr["lon"], arr["status"]
        assert (st[-1] == 1).all()
        dlon = lon[-1] - lon0
        expect = 0.5 * 2 * 3600.0 / (
            np.cos(np.deg2rad(lat0)) * METERS_PER_DEG_LAT)
        np.testing.assert_allclose(dlon, expect, rtol=0.02)


class TestReviewFindings:
    """review round 1 findings (2026-08-21), each pinned empirically."""

    def _mini_mesh(self, wetpix, nside=8):
        npix = 12 * nside * nside
        wet = np.zeros(npix, bool)
        wet[wetpix] = True
        return build_healpix_pixel_mesh(nside, wet_mask=wet)

    def test_thin_corner_warning(self):
        target = healpy.ang2pix(8, 45.0, 20.0, nest=True, lonlat=True)
        with pytest.warns(RuntimeWarning, match="singular"):
            self._mini_mesh([target])

    def test_leastsq_dies_edge_leastsq_tracks_on_isolated_pixel(self):
        from kelp.core.io import VelocityField
        from kelp.core.particles import make_state
        from kelp.numerics.integrators import get_integrator
        target = healpy.ang2pix(8, 45.0, 20.0, nest=True, lonlat=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = self._mini_mesh([target])
        lonp, latp = _pix_lonlat(8, np.array([target]))
        lon0 = np.array([((float(lonp[0]) + 180) % 360) - 180])
        lat0 = np.array([float(latp[0])])
        outcomes = {}
        for interp in ("leastsq", "edge_leastsq"):
            vel = VelocityField(u=np.full(m.n_elements, 0.4),
                                v=np.zeros(m.n_elements),
                                eta=np.zeros(m.n_nodes), w=None,
                                uv_layout="element")
            st = make_state(lon0.copy(), lat0.copy())
            st = st.replace(
                nearest_node=find_nearest_nodes(m, st.lon, st.lat))
            step = get_integrator("euler")
            for _ in range(3):
                st = step(st, vel, m, 3600.0, 1.0, uv_interp=interp)
            outcomes[interp] = (int(st.status[0]),
                                float(st.lon[0] - lon0[0]))
        assert outcomes["leastsq"][0] == 3        # singular stencil kills
        assert outcomes["edge_leastsq"][0] == 1   # ghosts keep it alive
        # the isolated pixel is walled on all four sides: the free-slip
        # mirror ghosts (2026-09-23) damp the eastward flow normal to its
        # east/west walls, so the particle creeps instead of crossing
        # (the old copy-ghost carried u = 0.4 through the walls: > 0.03 deg)
        assert 0.0 < outcomes["edge_leastsq"][1] < 0.03

    def test_read_scalar_fields_pixel_not_implemented(self, tmp_path):
        p = _make_uv_store(tmp_path / "3D_t_ocean.nc", with_theta=True)
        mesh = HealpixReader.load_mesh(
            _mesh_cfg(p, healpix_geometry="pixel"), mode_cfg=MODE_2D)
        ds = HealpixReader.open_dataset(p, mode_cfg=MODE_2D)
        with pytest.raises(NotImplementedError, match="pixel"):
            HealpixReader.read_scalar_fields(ds, 0, mesh, ["avg_thetao"],
                                             MODE_2D)

    def test_uniform_diffusion_ignores_kh_vars_in_store(self, tmp_path):
        from kelp.core.config import DiffusionConfig
        p = _make_uv_store(tmp_path / "3D_t_ocean.nc",
                           extra_vars=("Kh_zonal", "Kh_meridional"))
        mesh = HealpixReader.load_mesh(
            _mesh_cfg(p, healpix_geometry="pixel"), mode_cfg=MODE_2D)
        ds = HealpixReader.open_dataset(p, mode_cfg=MODE_2D)
        diff = DiffusionConfig(enabled=True, scheme="uniform",
                               kh_zonal=10.0, kh_meridional=10.0)
        vel = HealpixReader.read_velocities(ds, 0, mesh, mode_cfg=MODE_2D,
                                            diffusion=diff)
        assert vel.kh_zonal is None               # uniform never reads Kh

    def test_em_scheme_pixel_fails_fast_without_vars(self, tmp_path):
        from kelp.core.config import DiffusionConfig
        p = _make_uv_store(tmp_path / "3D_t_ocean.nc")
        mesh = HealpixReader.load_mesh(
            _mesh_cfg(p, healpix_geometry="pixel"), mode_cfg=MODE_2D)
        ds = HealpixReader.open_dataset(p, mode_cfg=MODE_2D)
        diff = DiffusionConfig(enabled=True, scheme="em")
        with pytest.raises(NotImplementedError, match="Kh"):
            HealpixReader.read_velocities(ds, 0, mesh, mode_cfg=MODE_2D,
                                          diffusion=diff)


class TestSingularStencilGuard:
    """Review 2026-08-21 BLOCKER: the absolute det guard (1e-30) never
    fires on a rank-deficient metre-scaled Gram matrix — Cramer divides
    roundoff by roundoff and returns NON-PHYSICAL velocities (measured up
    to 8 m/s on a 0.5 m/s constant field) at corners with 1-2 incident
    wet pixels.  Contract: at ANY query the leastsq interpolation either
    fails cleanly (ok=False) or is accurate — never silent garbage."""

    def test_thin_corner_leastsq_clean_fail_or_exact(self):
        from kelp.core.io import VelocityField
        from kelp.core.particles import make_state
        from kelp.numerics.interpolate import interp_leastsq
        nside = 8
        npix = 12 * nside * nside
        target = healpy.ang2pix(nside, 45.0, 20.0, nest=True, lonlat=True)
        nb = healpy.get_all_neighbours(nside, target, nest=True)
        wet = np.zeros(npix, bool)
        wet[[target, nb[4]]] = True          # 2-pixel strip
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = build_healpix_pixel_mesh(nside, wet_mask=wet)
        vel = VelocityField(u=np.full(m.n_elements, 0.5),
                            v=np.zeros(m.n_elements),
                            eta=np.zeros(m.n_nodes), w=None,
                            uv_layout="element")
        # OFF-CENTRE queries inside the strip (the exact-centre case has
        # det exactly 0 and always failed; off-centre is where the old
        # absolute guard let garbage through)
        clon, clat = _pix_lonlat(nside, np.array([target]))
        clon = ((float(clon[0]) + 180) % 360) - 180
        rng = np.random.default_rng(11)
        qlon = clon + rng.uniform(-2.0, 2.0, 60)
        qlat = float(clat[0]) + rng.uniform(-2.0, 2.0, 60)
        st = make_state(qlon, qlat)
        nn = find_nearest_nodes(m, qlon, qlat)
        uu, vv, ok = interp_leastsq(st, m, vel, nn)
        bad = ok & (np.abs(uu - 0.5) > 1e-6)
        assert not bad.any(), (
            f"{bad.sum()} silent-garbage results, max |u-0.5| = "
            f"{np.abs(uu[ok] - 0.5).max():.3f}")

    def test_well_conditioned_stencil_still_exact(self, band):
        from kelp.core.io import VelocityField
        from kelp.core.particles import make_state
        from kelp.numerics.interpolate import interp_leastsq
        mesh, wet = band
        vel = VelocityField(u=np.full(mesh.n_elements, 0.5),
                            v=np.zeros(mesh.n_elements),
                            eta=np.zeros(mesh.n_nodes), w=None,
                            uv_layout="element")
        rng = np.random.default_rng(12)
        qlat = np.degrees(np.arcsin(rng.uniform(-0.7, 0.7, 200)))
        qlon = rng.uniform(-170.0, 170.0, 200)
        st = make_state(qlon, qlat)
        nn = find_nearest_nodes(mesh, qlon, qlat)
        uu, _, ok = interp_leastsq(st, mesh, vel, nn)
        assert ok.mean() > 0.95
        np.testing.assert_allclose(uu[ok], 0.5, atol=1e-9)


class TestFusedCertificateGate:
    def test_geographic_pixel_mesh_unfuses_with_warning(self):
        # nside 16+ all-wet geographic pixel meshes fail the grid-hash
        # density certificate on their polar rows -> the fused leastsq
        # kernels must step aside (one-shot warning), mirroring the
        # barycentric gate.  (nside 8 happens to pass the certificate.)
        from kelp.core.io import VelocityField
        from kelp.core.particles import make_state
        from kelp.numerics.integrators import get_integrator
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m16 = build_healpix_pixel_mesh(16)
        vel = VelocityField(u=np.full(m16.n_elements, 0.1),
                            v=np.zeros(m16.n_elements),
                            eta=np.zeros(m16.n_nodes), w=None,
                            uv_layout="element")
        st = make_state(np.array([10.0]), np.array([5.0]))
        st = st.replace(nearest_node=find_nearest_nodes(
            m16, st.lon, st.lat))
        with pytest.warns(RuntimeWarning, match="unfused"):
            get_integrator("rk2")(st, vel, m16, 3600.0, 1.0,
                                  uv_interp="leastsq")

    def test_fesom_meshes_not_consulted(self):
        # non-healpix meshes keep the ungated fused path: the gate helper
        # answers True without touching the certificate machinery.
        from kelp.numerics.integrators import _leastsq_fused_anchor_exact
        from test_3d_analytical import _make_mesh
        m = _make_mesh(5.0, 5.0)
        assert _leastsq_fused_anchor_exact(m) is True
        assert getattr(m, "_gh_dense", None) is None  # never computed


class TestRound3Findings:
    def test_tinterp_thin_corner_clean_fail_or_exact(self):
        # round-3: the TINTERP leastsq core had been left on the absolute
        # guard — same contract as TestSingularStencilGuard but through
        # the time-aware path (vel + vel_next).
        from kelp.core.io import VelocityField
        from kelp.core.particles import make_state
        from kelp.numerics.integrators import get_integrator
        nside = 8
        npix = 12 * nside * nside
        target = healpy.ang2pix(nside, 45.0, 20.0, nest=True, lonlat=True)
        nb = healpy.get_all_neighbours(nside, target, nest=True)
        wet = np.zeros(npix, bool)
        wet[[target, nb[4]]] = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = build_healpix_pixel_mesh(nside, wet_mask=wet)

        def vf(val):
            return VelocityField(u=np.full(m.n_elements, val),
                                 v=np.zeros(m.n_elements),
                                 eta=np.zeros(m.n_nodes), w=None,
                                 uv_layout="element")
        clon, clat = _pix_lonlat(nside, np.array([target]))
        lon0 = np.array([((float(clon[0]) + 180) % 360) - 180 + 0.8])
        lat0 = np.array([float(clat[0]) + 0.5])
        st = make_state(lon0, lat0)
        st = st.replace(nearest_node=find_nearest_nodes(m, lon0, lat0))
        out = get_integrator("rk2")(st, vf(0.5), m, 1800.0, 1.0,
                                    uv_interp="leastsq",
                                    vel_next=vf(0.5), dt_to_next=3600.0)
        moved = abs(float(out.lon[0] - lon0[0]))
        # clean failure (revert, OUTSIDE) or an accurate move — never a
        # garbage displacement
        expect = 0.5 * 1800.0 / (np.cos(np.deg2rad(lat0[0])) * 111120.0)
        assert (out.status[0] == 3 and moved < 1e-12) or \
            abs(moved - expect) < 0.1 * expect

    def test_rk45_rejects_certificate_failing_pixel_mesh(self):
        from kelp.core.io import VelocityField
        from kelp.core.particles import make_state
        from kelp.numerics.integrators import get_integrator
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m16 = build_healpix_pixel_mesh(16)
        vel = VelocityField(u=np.full(m16.n_elements, 0.1),
                            v=np.zeros(m16.n_elements),
                            eta=np.zeros(m16.n_nodes), w=None,
                            uv_layout="element")
        st = make_state(np.array([10.0]), np.array([5.0]))
        st = st.replace(nearest_node=find_nearest_nodes(
            m16, st.lon, st.lat))
        with pytest.raises(ValueError, match="density certificate"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                get_integrator("rk45")(st, vel, m16, 3600.0, 1.0,
                                       uv_interp="leastsq")
