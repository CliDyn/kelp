"""Rotated-frame tracking: readers, warnings (Task 2 scope)."""
import warnings as _warnings

import numpy as np
import pytest
import xarray as xr

from kelp.core.config import (
    AppConfig, DataConfig, MeshConfig, ModeConfig, NumericsConfig, TimeConfig,
)
from kelp.core.geometry import METERS_PER_DEG_LAT, R_EARTH_STANDARD
from kelp.core.rotation import (
    FESOM2_ANGLES, euler_matrix, rotate_lonlat, rotate_vectors,
    unrotate_lonlat, unrotate_vectors,
)
from kelp.readers import get_reader


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

#: canonical 3x3 grid connectivity (8 triangles), 0-based, shared by helpers.
_GRID_ELEMS_0 = np.array(
    [[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4],
     [3, 4, 7], [3, 7, 6], [4, 5, 8], [4, 8, 7]], dtype=np.int32)


def _grid_coords(lat_centre=5.0, lon_centre=5.0):
    """3x3 grid centred at (lon_centre, lat_centre); 1-deg spacing."""
    xs = lon_centre + np.array([-1.0, 0.0, 1.0])
    ys = lat_centre + np.array([-1.0, 0.0, 1.0])
    gx, gy = np.meshgrid(xs, ys)
    return gx.ravel().astype(np.float64), gy.ravel().astype(np.float64)


def _circular_mean_centres(lon, lat, elems):
    """Geographic element centres exactly as the reader computes them.

    Mirrors ``fesom2._compute_elem_centres`` (circular mean for longitude,
    arithmetic mean for latitude) so tests pin the carrier convention.
    """
    tri_lon = lon[elems]
    tri_lat = lat[elems]
    lon_rad = np.deg2rad(tri_lon)
    lon_c = np.rad2deg(np.arctan2(
        np.sin(lon_rad).mean(axis=1), np.cos(lon_rad).mean(axis=1)))
    lat_c = tri_lat.mean(axis=1)
    return lon_c, lat_c


def _make_fesom2_diag(tmp_path, lon, lat, elems, *, tag="m"):
    """Write a minimal FESOM2 mesh-diag NetCDF; return its path.

    Carries exactly the variables ``_load_mesh_nc`` requires:
    ``lon``/``lat`` (nod2), ``elements`` (n3, elem, 1-based),
    ``nod_in_elem2D`` (N, nod2, 1-based 0-pad) + ``nod_in_elem2D_num``.
    """
    from kelp.core.mesh import build_node_to_elems
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    elems = np.asarray(elems, dtype=np.int32)
    n_nodes = lon.shape[0]
    elements = (elems.T + 1).astype(np.int32)         # (n3, elem), 1-based
    n2e, n2e_cnt = build_node_to_elems(n_nodes, elems)
    nie = np.where(n2e >= 0, n2e + 1, 0).astype(np.int32).T   # (max_conn, nod2)
    ds = xr.Dataset({
        "lon": (["nod2"], lon), "lat": (["nod2"], lat),
        "elements": (["n3", "elem"], elements),
        "nod_in_elem2D": (["N", "nod2"], nie),
        "nod_in_elem2D_num": (["nod2"], n2e_cnt.astype(np.int32)),
    })
    p = tmp_path / f"fesom2_diag_{tag}.nc"
    ds.to_netcdf(p)
    return p


def _make_fesom2_uv(tmp_path, lon, lat, elems, *, u, v, layout="element",
                    n_time=2, with_w=False, n_layers=2, tag="uv"):
    """Write a FESOM2 velocity NetCDF (single file).

    ``u``/``v`` are broadcast over time (and layers when 3D).  ``layout``
    selects the spatial dim: element centres (``"element"``) or nodes
    (``"node"``).  Returns the file path.
    """
    lon = np.asarray(lon, dtype=np.float64)
    elems = np.asarray(elems, dtype=np.int32)
    n_nodes = lon.shape[0]
    n_elem = elems.shape[0]
    n_items = n_nodes if layout == "node" else n_elem
    spatial = "nod2" if layout == "node" else "elem"
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    data = {}
    if with_w:
        u_arr = np.broadcast_to(u[None, None, :], (n_time, n_layers, n_items)).copy()
        v_arr = np.broadcast_to(v[None, None, :], (n_time, n_layers, n_items)).copy()
        data["u"] = (["time", "nz1", spatial], u_arr)
        data["v"] = (["time", "nz1", spatial], v_arr)
        data["w"] = (["time", "nz", "nod2"], np.zeros((n_time, n_layers + 1, n_nodes)))
    else:
        u_arr = np.broadcast_to(u[None, :], (n_time, n_items)).copy()
        v_arr = np.broadcast_to(v[None, :], (n_time, n_items)).copy()
        data["u"] = (["time", spatial], u_arr)
        data["v"] = (["time", spatial], v_arr)
    data["ssh"] = (["time", "nod2"], np.zeros((n_time, n_nodes)))

    import pandas as pd
    ds = xr.Dataset(data, coords={
        "time": pd.date_range("2020-01-01", periods=n_time, freq="3600s")})
    p = tmp_path / f"fesom2_{tag}.nc"
    ds.to_netcdf(p)
    return p


def _fesomc_ds(tmp_path, lat_centre=5.0, *, tag="m"):
    """FESOM-C-style mesh file on a 3x3 grid centred at lat_centre."""
    lon, lat = _grid_coords(lat_centre=lat_centre)
    elems = []
    for j in range(2):
        for i in range(2):
            n0 = j * 3 + i
            elems.append([n0 + 1, n0 + 2, n0 + 4, n0 + 1])
            elems.append([n0 + 2, n0 + 5, n0 + 4, n0 + 2])
    nv = np.array(elems, dtype=np.int32)
    en0 = nv[:, :3] - 1
    ds = xr.Dataset({
        "lon": (["node"], lon), "lat": (["node"], lat),
        "depth": (["node"], np.full(9, 10.0)),
        "nv": (["nele", "four"], nv),
        "lon_elem": (["nele"], np.array([lon[e].mean() for e in en0])),
        "lat_elem": (["nele"], np.array([lat[e].mean() for e in en0])),
        "sigma_lev": (["sigma"], np.array([0.0, 0.5, 1.0])),
    })
    p = tmp_path / f"fesomc_{tag}.nc"
    ds.to_netcdf(p)
    return p


# ---------------------------------------------------------------------------
# Mesh + velocity rotation
# ---------------------------------------------------------------------------

class TestMeshRotation:
    def test_nodes_and_centres_rotated_as_points(self, tmp_path):
        lon, lat = _grid_coords(lat_centre=45.0)
        p = _make_fesom2_diag(tmp_path, lon, lat, _GRID_ELEMS_0, tag="rot")
        r = get_reader("fesom2")
        mesh = r.load_mesh(MeshConfig(type="fesom2", meshpath=str(p),
                                      rotation="fesom2"))
        R = euler_matrix(*FESOM2_ANGLES)
        exp_lon, exp_lat = rotate_lonlat(lon, lat, R)
        np.testing.assert_allclose(mesh.nodes_lon, exp_lon, atol=1e-12)
        np.testing.assert_allclose(mesh.nodes_lat, exp_lat, atol=1e-12)
        # centres: rotate the GEOGRAPHIC circular-mean centres AS POINTS,
        # NOT the circular mean of the rotated vertices.
        geo_clon, geo_clat = _circular_mean_centres(lon, lat, _GRID_ELEMS_0)
        exp_clon, exp_clat = rotate_lonlat(geo_clon, geo_clat, R)
        np.testing.assert_allclose(mesh.elem_center_lon, exp_clon, atol=1e-12)
        np.testing.assert_allclose(mesh.elem_center_lat, exp_clat, atol=1e-12)
        assert tuple(mesh.rotation_angles) == FESOM2_ANGLES

    def test_no_rotation_byte_identical(self, tmp_path):
        lon, lat = _grid_coords(lat_centre=45.0)
        p = _make_fesom2_diag(tmp_path, lon, lat, _GRID_ELEMS_0, tag="norot")
        r = get_reader("fesom2")
        plain = r.load_mesh(MeshConfig(type="fesom2", meshpath=str(p)))
        none_cfg = r.load_mesh(MeshConfig(type="fesom2", meshpath=str(p),
                                          rotation=None))
        for m in (plain, none_cfg):
            np.testing.assert_array_equal(m.nodes_lon, lon)
            np.testing.assert_array_equal(m.nodes_lat, lat)
            geo_clon, geo_clat = _circular_mean_centres(lon, lat, _GRID_ELEMS_0)
            np.testing.assert_array_equal(m.elem_center_lon, geo_clon)
            np.testing.assert_array_equal(m.elem_center_lat, geo_clat)
            assert m.rotation_angles is None
            assert m._vec_rot_factors is None

    def test_factors_cached_on_mesh(self, tmp_path):
        lon, lat = _grid_coords(lat_centre=45.0)
        p = _make_fesom2_diag(tmp_path, lon, lat, _GRID_ELEMS_0, tag="fac")
        r = get_reader("fesom2")
        mesh = r.load_mesh(MeshConfig(type="fesom2", meshpath=str(p),
                                      rotation="fesom2"))
        assert mesh._vec_rot_factors is not None
        assert set(mesh._vec_rot_factors) == {"element", "node"}
        R = euler_matrix(*FESOM2_ANGLES)
        geo_clon, geo_clat = _circular_mean_centres(lon, lat, _GRID_ELEMS_0)
        carriers = {
            "node": (lon, lat),
            "element": (geo_clon, geo_clat),
        }
        for layout, (clon, clat) in carriers.items():
            c, s = mesh._vec_rot_factors[layout]
            np.testing.assert_allclose(c ** 2 + s ** 2, 1.0, atol=1e-12)
            ec, es = rotate_vectors(np.ones_like(clon), np.zeros_like(clon),
                                    clon, clat, R)
            np.testing.assert_allclose(c, ec, atol=1e-12)
            np.testing.assert_allclose(s, es, atol=1e-12)

    def test_velocities_rotated_element_layout(self, tmp_path):
        lon, lat = _grid_coords(lat_centre=45.0)
        mp = _make_fesom2_diag(tmp_path, lon, lat, _GRID_ELEMS_0, tag="velm")
        r = get_reader("fesom2")
        mesh = r.load_mesh(MeshConfig(type="fesom2", meshpath=str(mp),
                                      rotation="fesom2"))
        n_elem = _GRID_ELEMS_0.shape[0]
        u_raw = np.ones(n_elem)
        v_raw = np.zeros(n_elem)
        dp = _make_fesom2_uv(tmp_path, lon, lat, _GRID_ELEMS_0,
                             u=u_raw, v=v_raw, layout="element", tag="velm")
        ds = r.open_dataset(str(dp))
        vf = r.read_velocities(ds, 0, mesh,
                               ModeConfig(Fdrift2D=True, Fdrift3D=False))
        ds.close()
        R = euler_matrix(*FESOM2_ANGLES)
        geo_clon, geo_clat = _circular_mean_centres(lon, lat, _GRID_ELEMS_0)
        exp_u, exp_v = rotate_vectors(u_raw, v_raw, geo_clon, geo_clat, R)
        np.testing.assert_allclose(vf.u, exp_u, atol=1e-12)
        np.testing.assert_allclose(vf.v, exp_v, atol=1e-12)
        np.testing.assert_allclose(vf.u ** 2 + vf.v ** 2, 1.0, atol=1e-12)

    def test_factor_form_matches_rotate_vectors_random_field(self, tmp_path):
        # COMMITTED SIGN PIN: a reflected factor map passes unit-east and
        # magnitude tests but fails on a general random field.
        lon, lat = _grid_coords(lat_centre=45.0)
        mp = _make_fesom2_diag(tmp_path, lon, lat, _GRID_ELEMS_0, tag="rnd")
        r = get_reader("fesom2")
        mesh = r.load_mesh(MeshConfig(type="fesom2", meshpath=str(mp),
                                      rotation="fesom2"))
        R = euler_matrix(*FESOM2_ANGLES)
        rng = np.random.default_rng(20260606)
        geo_clon, geo_clat = _circular_mean_centres(lon, lat, _GRID_ELEMS_0)
        for layout, (clon, clat), n_items in (
            ("element", (geo_clon, geo_clat), _GRID_ELEMS_0.shape[0]),
            ("node", (lon, lat), lon.shape[0]),
        ):
            u_raw = rng.standard_normal(n_items)
            v_raw = rng.standard_normal(n_items)
            mode = ModeConfig(Fdrift2D=True, Fdrift3D=False)
            if layout == "node":
                mode = ModeConfig(Fdrift2D=True, Fdrift3D=False,
                                  var_u="unod", var_v="vnod")
                import pandas as pd
                ds = xr.Dataset({
                    "unod": (["time", "nod2"],
                             np.broadcast_to(u_raw[None, :], (2, n_items)).copy()),
                    "vnod": (["time", "nod2"],
                             np.broadcast_to(v_raw[None, :], (2, n_items)).copy()),
                    "ssh": (["time", "nod2"], np.zeros((2, lon.shape[0]))),
                }, coords={"time": pd.date_range("2020-01-01", periods=2,
                                                 freq="3600s")})
            else:
                dp = _make_fesom2_uv(tmp_path, lon, lat, _GRID_ELEMS_0,
                                     u=u_raw, v=v_raw, layout="element",
                                     tag="rnd")
                ds = r.open_dataset(str(dp))
            vf = r.read_velocities(ds, 0, mesh, mode)
            if hasattr(ds, "close"):
                ds.close()
            exp_u, exp_v = rotate_vectors(u_raw, v_raw, clon, clat, R)
            np.testing.assert_allclose(vf.u, exp_u, atol=1e-12, err_msg=layout)
            np.testing.assert_allclose(vf.v, exp_v, atol=1e-12, err_msg=layout)

    def test_unit_north_maps_to_minus_s_c(self, tmp_path):
        # column-2 pin: u=0, v=1 reads back as (-s, c).
        lon, lat = _grid_coords(lat_centre=45.0)
        mp = _make_fesom2_diag(tmp_path, lon, lat, _GRID_ELEMS_0, tag="north")
        r = get_reader("fesom2")
        mesh = r.load_mesh(MeshConfig(type="fesom2", meshpath=str(mp),
                                      rotation="fesom2"))
        n_elem = _GRID_ELEMS_0.shape[0]
        dp = _make_fesom2_uv(tmp_path, lon, lat, _GRID_ELEMS_0,
                             u=np.zeros(n_elem), v=np.ones(n_elem),
                             layout="element", tag="north")
        ds = r.open_dataset(str(dp))
        vf = r.read_velocities(ds, 0, mesh,
                               ModeConfig(Fdrift2D=True, Fdrift3D=False))
        ds.close()
        c, s = mesh._vec_rot_factors["element"]
        np.testing.assert_allclose(vf.u, -s, atol=1e-12)
        np.testing.assert_allclose(vf.v, c, atol=1e-12)

    def test_scalars_untouched(self, tmp_path):
        # eta and w pass through the rotation untouched (w only gets the
        # reader's positive-up -> positive-down conversion; this test once
        # passed by accident while that conversion flipped the reference
        # array in place).
        lon, lat = _grid_coords(lat_centre=45.0)
        mp = _make_fesom2_diag(tmp_path, lon, lat, _GRID_ELEMS_0, tag="scal")
        r = get_reader("fesom2")
        mesh = r.load_mesh(MeshConfig(type="fesom2", meshpath=str(mp),
                                      rotation="fesom2"))
        n_nodes = lon.shape[0]
        n_elem = _GRID_ELEMS_0.shape[0]
        import pandas as pd
        rng = np.random.default_rng(7)
        eta_vals = rng.standard_normal(n_nodes)
        u_raw = np.broadcast_to(np.ones(n_elem)[None, None, :], (2, 2, n_elem)).copy()
        w_vals = rng.standard_normal((2, 3, n_nodes))
        ds = xr.Dataset({
            "u": (["time", "nz1", "elem"], u_raw),
            "v": (["time", "nz1", "elem"], np.zeros_like(u_raw)),
            "w": (["time", "nz", "nod2"], w_vals),
            "ssh": (["time", "nod2"],
                    np.broadcast_to(eta_vals[None, :], (2, n_nodes)).copy()),
        }, coords={"time": pd.date_range("2020-01-01", periods=2,
                                         freq="3600s")})
        vf = r.read_velocities(ds, 0, mesh,
                               ModeConfig(Fdrift2D=False, Fdrift3D=True))
        ds.close()
        np.testing.assert_array_equal(vf.eta, eta_vals)
        np.testing.assert_array_equal(vf.w, -w_vals[0])

    def test_3d_velocities_rotated_per_layer(self, tmp_path):
        # 3D (K-1, E) element u/v: the same factors apply to every layer.
        lon, lat = _grid_coords(lat_centre=45.0)
        mp = _make_fesom2_diag(tmp_path, lon, lat, _GRID_ELEMS_0, tag="3d")
        r = get_reader("fesom2")
        mesh = r.load_mesh(MeshConfig(type="fesom2", meshpath=str(mp),
                                      rotation="fesom2"))
        n_elem = _GRID_ELEMS_0.shape[0]
        n_nodes = lon.shape[0]
        # distinct per-layer values so a per-layer mix-up is detectable
        import pandas as pd
        u_l0, u_l1 = 1.0, 0.3
        v_l0, v_l1 = 0.0, 0.7
        u_arr = np.empty((2, 2, n_elem))
        v_arr = np.empty((2, 2, n_elem))
        u_arr[:, 0, :] = u_l0; u_arr[:, 1, :] = u_l1
        v_arr[:, 0, :] = v_l0; v_arr[:, 1, :] = v_l1
        ds = xr.Dataset({
            "u": (["time", "nz1", "elem"], u_arr),
            "v": (["time", "nz1", "elem"], v_arr),
            "w": (["time", "nz", "nod2"], np.zeros((2, 3, n_nodes))),
            "ssh": (["time", "nod2"], np.zeros((2, n_nodes))),
        }, coords={"time": pd.date_range("2020-01-01", periods=2,
                                         freq="3600s")})
        vf = r.read_velocities(ds, 0, mesh,
                               ModeConfig(Fdrift2D=False, Fdrift3D=True))
        ds.close()
        R = euler_matrix(*FESOM2_ANGLES)
        geo_clon, geo_clat = _circular_mean_centres(lon, lat, _GRID_ELEMS_0)
        eu0, ev0 = rotate_vectors(np.full(n_elem, u_l0), np.full(n_elem, v_l0),
                                  geo_clon, geo_clat, R)
        eu1, ev1 = rotate_vectors(np.full(n_elem, u_l1), np.full(n_elem, v_l1),
                                  geo_clon, geo_clat, R)
        assert vf.u.shape == (2, n_elem)
        np.testing.assert_allclose(vf.u[0], eu0, atol=1e-12)
        np.testing.assert_allclose(vf.v[0], ev0, atol=1e-12)
        np.testing.assert_allclose(vf.u[1], eu1, atol=1e-12)
        np.testing.assert_allclose(vf.v[1], ev1, atol=1e-12)


# ---------------------------------------------------------------------------
# Near-pole warnings
# ---------------------------------------------------------------------------

class TestNearPoleWarnings:
    def test_fesom2_near_pole_without_rotation_warns(self, tmp_path):
        lon, lat = _grid_coords(lat_centre=88.0)   # max|lat| = 89
        p = _make_fesom2_diag(tmp_path, lon, lat, _GRID_ELEMS_0, tag="pole")
        r = get_reader("fesom2")
        with pytest.warns(RuntimeWarning, match="rotation"):
            r.load_mesh(MeshConfig(type="fesom2", meshpath=str(p)))

    def test_fesom2_with_rotation_no_warning(self, tmp_path, capsys):
        lon, lat = _grid_coords(lat_centre=88.0)   # max|lat| = 89
        p = _make_fesom2_diag(tmp_path, lon, lat, _GRID_ELEMS_0, tag="poleR")
        r = get_reader("fesom2")
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            r.load_mesh(MeshConfig(type="fesom2", meshpath=str(p),
                                   rotation="fesom2"))
        pole_warns = [w for w in caught
                      if issubclass(w.category, RuntimeWarning)
                      and "pole" in str(w.message).lower()]
        assert not pole_warns, [str(w.message) for w in pole_warns]
        out = capsys.readouterr().out
        assert "rotated frame" in out

    def test_fesom2_midlat_silent(self, tmp_path):
        lon, lat = _grid_coords(lat_centre=60.0)   # max|lat| = 61
        p = _make_fesom2_diag(tmp_path, lon, lat, _GRID_ELEMS_0, tag="mid")
        r = get_reader("fesom2")
        for cfg in (MeshConfig(type="fesom2", meshpath=str(p)),
                    MeshConfig(type="fesom2", meshpath=str(p),
                               rotation="fesom2")):
            with _warnings.catch_warnings(record=True) as caught:
                _warnings.simplefilter("always")
                r.load_mesh(cfg)
            pole_warns = [w for w in caught
                          if issubclass(w.category, RuntimeWarning)
                          and "pole" in str(w.message).lower()]
            assert not pole_warns, [str(w.message) for w in pole_warns]

    def test_fesomc_near_pole_warns_without_rotation_hint(self, tmp_path):
        p = _fesomc_ds(tmp_path, lat_centre=88.0, tag="pole")   # max|lat| = 89
        r = get_reader("fesomc")
        with pytest.warns(RuntimeWarning) as record:
            r.load_mesh(MeshConfig(type="fesomc", meshpath=str(p)))
        pole_msgs = [str(w.message) for w in record
                     if issubclass(w.category, RuntimeWarning)
                     and "pole" in str(w.message).lower()]
        assert pole_msgs, "expected a near-pole RuntimeWarning"
        for msg in pole_msgs:
            assert "rotation" not in msg

    def test_fesomc_rejects_rotation_config(self, tmp_path):
        p = _fesomc_ds(tmp_path, lat_centre=5.0, tag="reject")
        r = get_reader("fesomc")
        with pytest.raises(ValueError, match="rotation"):
            r.load_mesh(MeshConfig(type="fesomc", meshpath=str(p),
                                   rotation="fesom2"))

    def test_exact_pole_node_warns(self, tmp_path):
        # FESOM2 mesh with a node exactly at lat=90, rotation ON.
        lon, lat = _grid_coords(lat_centre=89.0)   # top row at lat = 90
        assert np.isclose(np.abs(lat).max(), 90.0)
        p = _make_fesom2_diag(tmp_path, lon, lat, _GRID_ELEMS_0, tag="exact")
        r = get_reader("fesom2")
        with pytest.warns(RuntimeWarning, match="pole"):
            r.load_mesh(MeshConfig(type="fesom2", meshpath=str(p),
                                   rotation="fesom2"))


# ---------------------------------------------------------------------------
# Tracking layer: seeds in, geographic record out, checkpoint frame
# (Task 3 scope).  Spec sections "Seeds", "Output", "Checkpoints".
# ---------------------------------------------------------------------------

def _midlat_fesom2_files(tmp_path, *, tag, lat_centre=45.0, lon_centre=5.0,
                         u_val=1.0, v_val=0.0, n_time=4):
    """FESOM2 mesh-diag + element-layout uv file on a 3x3 grid.

    ONE physical dataset: constant GEOGRAPHIC eastward flow ``u=u_val``,
    ``v=v_val``.  Returns ``(mesh_path, uv_path, lon, lat)``.  The same
    files are loaded with and without rotation — the reader rotates the
    mesh/vectors when ``MeshConfig.rotation`` is set, so the on-disk data
    is the single geographic source of truth for both runs.
    """
    lon, lat = _grid_coords(lat_centre=lat_centre, lon_centre=lon_centre)
    mp = _make_fesom2_diag(tmp_path, lon, lat, _GRID_ELEMS_0, tag=tag)
    n_elem = _GRID_ELEMS_0.shape[0]
    dp = _make_fesom2_uv(
        tmp_path, lon, lat, _GRID_ELEMS_0,
        u=np.full(n_elem, u_val), v=np.full(n_elem, v_val),
        layout="element", n_time=n_time, tag=tag)
    return mp, dp, lon, lat


class TestTrackingRoundTrip:
    def test_headline_equivalence_midlat(self, tmp_path):
        # THE pin: same physical run with and without rotation.
        #
        # TOLERANCE RATIONALE (measured against THIS tracker, not the
        # analytic review probe).  The continuous flow is chart-
        # invariant; the DISCRETE solution is not.  The review probe
        # (.tmp/chart_probe.py: no mesh, velocity evaluated analytically
        # at each RK stage) measured ~3e-10 deg and a 1e-7 bound — but
        # that probe omits the discrete carrier geometry the REAL tracker
        # has.  MEASURED here (3x3 FESOM2 grid, lat 5, lon 0, RK2):
        # d_coarse = 1.2e-6 deg, resolution-INDEPENDENT (3x3 and 9x9
        # match).  The dominant mechanism is the non-rotation-equivariant
        # CENTROID CARRIER: the reader evaluates element velocities at the
        # geographic circular-mean centroid and rotates that point, but
        # the rotated triangle's own centroid lies elsewhere — an O(1),
        # resolution-independent carrier shift.  The analytic chart-metric
        # term (3e-10) and the leastsq fit error (~1.7e-7) are
        # sub-dominant, so this is NOT a leastsq accuracy artifact.  It
        # still grows with latitude (~10x larger at 45 N).  (Assertion 2
        # below still converges: the carrier-shifted velocity is
        # interpolated and integrated, so the position difference carries
        # the RK time-truncation factor and shrinks with dt — distinct
        # from a constant-offset frame bug at ratio 1.0.)
        #
        # The bound's JOB is to separate this legitimate ~1e-6 chart
        # difference from a FRAME BUG.  A missing/double/wrong un-rotation
        # records the rotated state as geographic -> O(tens of degrees)
        # (measured 41.9 deg for missing-unrotate) AND is dt-INDEPENDENT
        # (a constant rotation offset -> convergence ratio 1.000).  So:
        # Assertion 1: |B - A| < 1e-4 deg (2 orders above the legit 1.2e-6,
        #   6 orders below a 42-deg bug).
        # Assertion 2 (convergence): at dt=1800 (6 steps, same physical
        #   horizon) the difference SHRINKS (measured ratio ~3.5); a
        #   constant-offset frame bug does NOT (ratio 1.0).  Assert
        #   d_fine < d_coarse / 2.5.
        # Assertion 3: vel diagnostics geographic in the OUTPUT -- run B's
        #   recorded u stays ~eastward (v ~ 0) like run A's.
        from kelp.tracking import track_particles

        # 8 snapshots: both runs below cover the SAME 10800 s (the old
        # 4-snapshot fixture stopped the fine run after 4 of its 6 steps,
        # at 7200 s — review 2026-09-22, M08)
        mp, dp, lon, lat = _midlat_fesom2_files(
            tmp_path, tag="head", lat_centre=5.0, lon_centre=0.0, n_time=8)
        reader = get_reader("fesom2")

        def _run(rotation, dt, n_steps):
            mesh = reader.load_mesh(MeshConfig(
                type="fesom2", meshpath=str(mp), rotation=rotation))
            ds = reader.open_dataset(str(dp))
            mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)
            rec = track_particles(
                mesh, ds, reader,
                np.array([0.0]), np.array([5.0]),
                start_time_idx=0, n_steps=n_steps, dt=dt,
                integrator="rk2", output_freq=1, mode=mode, temporal_interp=True,
            )
            ds.close()
            return rec.to_arrays()

        def _diff(rotation_none, rotation_fesom2):
            # max |B - A| over lon AND lat across all record points.
            dlon = np.abs(rotation_fesom2["lon"] - rotation_none["lon"])
            dlat = np.abs(rotation_fesom2["lat"] - rotation_none["lat"])
            return float(max(dlon.max(), dlat.max()))

        a_coarse = _run(None, 3600.0, 3)
        b_coarse = _run("fesom2", 3600.0, 3)
        d_coarse = _diff(a_coarse, b_coarse)

        a_fine = _run(None, 1800.0, 6)
        b_fine = _run("fesom2", 1800.0, 6)
        d_fine = _diff(a_fine, b_fine)

        # Both runs reached the same physical horizon.
        assert a_coarse["lon"].shape[0] == 3 and a_fine["lon"].shape[0] == 6
        assert b_coarse["lon"].shape[0] == 3 and b_fine["lon"].shape[0] == 6
        # The frame difference is a SPATIAL floor — the rotated and the
        # geographic mesh are not the same discrete interpolant (the
        # element-centre carriers do not rotate exactly) — so it does not
        # shrink with dt: measured 1.7336e-7 deg at 3600 s and 1.7354e-7 at
        # 1800 s over the same 10800 s (~2 cm).  The earlier "decreases with
        # dt" assertion passed only because the fine run stopped early.
        # Pin the bound instead: far below a frame bug (a wrong vector
        # rotation moves the particle by metres per hour).
        assert d_coarse < 3e-7, d_coarse
        assert d_fine < 3e-7, d_fine

        # Assertion 3: recorded velocity diagnostics are geographic.
        # Run A (no rotation) is eastward by construction (u=1, v=0); run B
        # must match it in the OUTPUT despite running in the rotated frame.
        # The un-rotated B velocity carries the chart difference (B's
        # interpolated-then-unrotated vector at a slightly chart-shifted
        # position vs A's geographic vector): MEASURED vel_v leak ~1.1e-4,
        # vel_u offset ~1.8e-6.  A FRAME BUG (un-rotation missing) emits the
        # rotated components as geographic -> vel_v ~ 0.17 (O(0.1)), vel_u
        # ~0.98; the 1e-3 bound separates the legit leak from the bug.
        assert "vel_u" in b_coarse and "vel_v" in b_coarse
        np.testing.assert_allclose(b_coarse["vel_u"], a_coarse["vel_u"],
                                   atol=1e-3)
        np.testing.assert_allclose(b_coarse["vel_v"], a_coarse["vel_v"],
                                   atol=1e-3)
        # eastward: u ~ 1, |v| << |u| in the geographic output.
        assert np.all(np.abs(b_coarse["vel_v"]) < 1e-3)
        assert np.all(b_coarse["vel_u"] > 0.5)

    def test_seed_rotation_track_particles(self, tmp_path):
        # track_particles(lon0, lat0) on a rotated mesh: geographic output
        # matches the unrotated reference run to the legitimate chart bound
        # (proves the seed was rotated IN once and the record un-rotated
        # once -- a double/missing rotation displaces the whole trajectory
        # by O(degrees), measured 42 deg, vs the legit ~1e-6 here).  See
        # test_headline_equivalence_midlat for the tolerance derivation.
        from kelp.tracking import track_particles

        mp, dp, lon, lat = _midlat_fesom2_files(
            tmp_path, tag="seed", lat_centre=5.0, lon_centre=0.0)
        reader = get_reader("fesom2")

        def _run(rotation):
            mesh = reader.load_mesh(MeshConfig(
                type="fesom2", meshpath=str(mp), rotation=rotation))
            ds = reader.open_dataset(str(dp))
            mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)
            rec = track_particles(
                mesh, ds, reader,
                np.array([0.0]), np.array([5.0]),
                start_time_idx=0, n_steps=3, dt=3600.0,
                integrator="rk2", output_freq=1, mode=mode,
            )
            ds.close()
            return rec.to_arrays()

        ref = _run(None)
        rot = _run("fesom2")
        # whole trajectory matches the unrotated reference to the chart
        # bound (a frame bug would be O(degrees), 4+ orders larger).
        assert np.abs(rot["lon"] - ref["lon"]).max() < 1e-4
        assert np.abs(rot["lat"] - ref["lat"]).max() < 1e-4

    def test_seed_rotation_run_tracking_inicloud(self, tmp_path):
        # config-driven run_tracking with rotation + an inicloud .dat seed
        # file (GEOGRAPHIC coords): the geographic output matches the
        # rotation=None run to the legitimate chart bound (proves the
        # inicloud seed crossed the geographic->frame boundary exactly
        # once at load).  See test_headline_equivalence_midlat for the
        # tolerance derivation.
        from kelp.tracking import run_tracking
        from kelp.writers import load_trajectories

        mp, dp, lon, lat = _midlat_fesom2_files(
            tmp_path, tag="ini", lat_centre=5.0, lon_centre=0.0)
        parts = tmp_path / "seed.dat"
        parts.write_text("0.0 5.0\n")

        def _cfg(rotation, out):
            return AppConfig(
                mode=ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0),
                numerics=NumericsConfig(dt=3600.0, integrator="rk2",
                                        output_freq=1),
                mesh=MeshConfig(type="fesom2", meshpath=str(mp),
                                rotation=rotation),
                data=DataConfig(ncinput=str(dp), inicloud=str(parts),
                                output_path=str(out),
                                time_averaging="instant"),
                time=TimeConfig(start_time_idx=0, n_steps=3),
            )

        out_n = tmp_path / "out_none.nc"
        out_r = tmp_path / "out_rot.nc"
        run_tracking(_cfg(None, out_n), output_path=str(out_n))
        run_tracking(_cfg("fesom2", out_r), output_path=str(out_r))
        tn = load_trajectories(str(out_n))
        tr = load_trajectories(str(out_r))
        # load_trajectories gives (P, T); the geographic trajectories match
        # to the chart bound (a frame bug would be O(degrees)).
        assert np.abs(tr["lon"] - tn["lon"]).max() < 1e-4
        assert np.abs(tr["lat"] - tn["lat"]).max() < 1e-4

    def test_internal_state_rotated_record_geographic(self, tmp_path):
        # run with writer + checkpoint_freq=1, rotation on:
        # - the WRITER file's lon/lat are geographic (match the
        #   no-rotation reference run);
        # - load_restart(state).lon/lat are the ROTATED frame (equal
        #   rotate_lonlat(geo last record) to 1e-9);
        # - restart dict carries rotation_angles == FESOM2_ANGLES.
        from kelp.tracking import track_particles
        from kelp.writers import (
            TrajectoryWriter, load_restart, load_trajectories,
        )

        mp, dp, lon, lat = _midlat_fesom2_files(
            tmp_path, tag="intern", lat_centre=5.0, lon_centre=0.0)
        reader = get_reader("fesom2")
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)

        # reference (no rotation) geographic trajectory.
        mesh_n = reader.load_mesh(MeshConfig(type="fesom2", meshpath=str(mp)))
        ds_n = reader.open_dataset(str(dp))
        ref = track_particles(
            mesh_n, ds_n, reader, np.array([0.0]), np.array([5.0]),
            start_time_idx=0, n_steps=3, dt=3600.0, integrator="rk2",
            output_freq=1, mode=mode).to_arrays()
        ds_n.close()

        # rotated run, streamed to a writer with per-step checkpoints.
        mesh_r = reader.load_mesh(MeshConfig(
            type="fesom2", meshpath=str(mp), rotation="fesom2"))
        ds_r = reader.open_dataset(str(dp))
        out = tmp_path / "rot_ckpt.nc"
        with TrajectoryWriter(str(out), 1) as w:
            track_particles(
                mesh_r, ds_r, reader, np.array([0.0]), np.array([5.0]),
                start_time_idx=0, n_steps=3, dt=3600.0, integrator="rk2",
                output_freq=1, mode=mode, writer=w, checkpoint_freq=1)
        ds_r.close()

        # 1) written positions are geographic (match the no-rotation run to
        # the legitimate chart bound; see test_headline_equivalence_midlat).
        tr = load_trajectories(str(out))
        assert np.abs(tr["lon"] - ref["lon"].T).max() < 1e-4
        assert np.abs(tr["lat"] - ref["lat"].T).max() < 1e-4

        # 2) the checkpointed (internal) state is in the ROTATED frame.
        restart = load_restart(str(out))
        R = euler_matrix(*FESOM2_ANGLES)
        geo_last_lon = tr["lon"][:, -1]
        geo_last_lat = tr["lat"][:, -1]
        exp_lon, exp_lat = rotate_lonlat(geo_last_lon, geo_last_lat, R)
        np.testing.assert_allclose(restart["state"].lon, exp_lon, atol=1e-9)
        np.testing.assert_allclose(restart["state"].lat, exp_lat, atol=1e-9)

        # 3) the restart dict carries the frame angles.
        assert tuple(restart["rotation_angles"]) == FESOM2_ANGLES

    def test_resume_angle_mismatch_raises(self, tmp_path):
        # _validate_restart_frame raises for (angles vs None),
        # (None vs angles), (angles vs different angles); passes for
        # equal tuples and (None vs None).
        from kelp.tracking import _validate_restart_frame

        class _M:
            def __init__(self, angles):
                self.rotation_angles = angles

        # mismatches raise.
        with pytest.raises(ValueError, match="frame"):
            _validate_restart_frame(FESOM2_ANGLES, _M(None))
        with pytest.raises(ValueError, match="frame"):
            _validate_restart_frame(None, _M(FESOM2_ANGLES))
        with pytest.raises(ValueError, match="frame"):
            _validate_restart_frame((10.0, 20.0, 30.0), _M(FESOM2_ANGLES))
        # matches pass (return None, no raise).
        assert _validate_restart_frame(FESOM2_ANGLES, _M(FESOM2_ANGLES)) is None
        assert _validate_restart_frame(None, _M(None)) is None
        # tuple vs list of the same angles is accepted.
        assert _validate_restart_frame(
            list(FESOM2_ANGLES), _M(FESOM2_ANGLES)) is None

    def test_resume_recipe_continues_in_frame(self, tmp_path):
        # the documented caller-driven recipe: run 2 steps with checkpoint,
        # load_restart, _validate_restart_frame(...), run_single_realization
        # (init_state=restart_state, start_step_i=...) for 2 more steps ->
        # final GEOGRAPHIC position equals the uninterrupted 4-step run
        # (1e-9: same frame, same arithmetic, no chart comparison).
        from kelp.tracking import (
            run_single_realization, track_particles, _validate_restart_frame,
        )
        from kelp.writers import TrajectoryWriter, load_restart

        mp, dp, lon, lat = _midlat_fesom2_files(tmp_path, tag="resume")
        reader = get_reader("fesom2")
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)

        # uninterrupted 4-step rotated run (geographic output).
        mesh = reader.load_mesh(MeshConfig(
            type="fesom2", meshpath=str(mp), rotation="fesom2"))
        ds = reader.open_dataset(str(dp))
        full = track_particles(
            mesh, ds, reader, np.array([5.0]), np.array([45.0]),
            start_time_idx=0, n_steps=4, dt=3600.0, integrator="rk2",
            output_freq=1, mode=mode).to_arrays()
        ds.close()

        # part 1: 2 steps with a checkpoint at the end.
        mesh2 = reader.load_mesh(MeshConfig(
            type="fesom2", meshpath=str(mp), rotation="fesom2"))
        ds2 = reader.open_dataset(str(dp))
        out = tmp_path / "resume.nc"
        with TrajectoryWriter(str(out), 1) as w:
            track_particles(
                mesh2, ds2, reader, np.array([5.0]), np.array([45.0]),
                start_time_idx=0, n_steps=2, dt=3600.0, integrator="rk2",
                output_freq=1, mode=mode, writer=w, checkpoint_freq=2)
        ds2.close()

        # part 2: resume from the checkpoint (already-rotated state -> no
        # re-rotation; run_single_realization is the in-frame engine).
        restart = load_restart(str(out))
        _validate_restart_frame(restart["rotation_angles"], mesh2)
        ds3 = reader.open_dataset(str(dp))
        numerics = NumericsConfig(dt=3600.0, integrator="rk2", output_freq=1)
        rec2 = run_single_realization(
            mesh2, ds3, reader, restart["state"], mode, numerics,
            start_time_idx=restart["file_idx"], n_steps=2,
            start_step_i=restart["step_i"] + 1)
        ds3.close()

        # run_single_realization records through the SAME geographic-view
        # hook (its mesh is rotated), so rec2 is ALREADY geographic -- the
        # resumed leg must reproduce the uninterrupted run's geographic
        # tail to round-trip precision (same frame, same arithmetic, the
        # checkpoint round-trips the rotated state through NetCDF f8).
        rec_arrs = rec2.to_arrays()
        np.testing.assert_allclose(rec_arrs["lon"][-1], full["lon"][-1],
                                   atol=1e-9)
        np.testing.assert_allclose(rec_arrs["lat"][-1], full["lat"][-1],
                                   atol=1e-9)

    def test_writer_metadata_rotation(self, tmp_path):
        # run_tracking with rotation -> nc has rotation_angles attr ==
        # [50, 15, -90]; without -> attr absent.
        import netCDF4
        from kelp.tracking import run_tracking

        mp, dp, lon, lat = _midlat_fesom2_files(tmp_path, tag="meta")
        parts = tmp_path / "seed.dat"
        parts.write_text("5.0 45.0\n")

        def _cfg(rotation, out):
            return AppConfig(
                mode=ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0),
                numerics=NumericsConfig(dt=3600.0, integrator="euler",
                                        output_freq=1),
                mesh=MeshConfig(type="fesom2", meshpath=str(mp),
                                rotation=rotation),
                data=DataConfig(ncinput=str(dp), inicloud=str(parts),
                                output_path=str(out),
                                time_averaging="instant"),
                time=TimeConfig(start_time_idx=0, n_steps=2),
            )

        out_r = tmp_path / "meta_rot.nc"
        run_tracking(_cfg("fesom2", out_r), output_path=str(out_r))
        with netCDF4.Dataset(out_r) as f:
            assert "rotation_angles" in f.ncattrs()
            np.testing.assert_array_equal(
                np.asarray(f.getncattr("rotation_angles"), dtype=float),
                np.array([50.0, 15.0, -90.0]))

        out_n = tmp_path / "meta_none.nc"
        run_tracking(_cfg(None, out_n), output_path=str(out_n))
        with netCDF4.Dataset(out_n) as f:
            assert "rotation_angles" not in f.ncattrs()

    @pytest.mark.parametrize("ext", [".nc", ".zarr"])
    def test_direct_writer_gets_rotation_metadata(self, tmp_path, ext):
        # The DIRECT writer path (track_particles + a caller-constructed
        # TrajectoryWriter, NO run_tracking) must still stamp the frame
        # provenance: rotated run -> rotation_angles attr == [50, 15, -90];
        # geographic run via the SAME path -> attr absent.  Both backends.
        from kelp.tracking import track_particles
        from kelp.writers import TrajectoryWriter

        mp, dp, lon, lat = _midlat_fesom2_files(tmp_path, tag="direct")
        reader = get_reader("fesom2")
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)

        def _run(rotation, out):
            mesh = reader.load_mesh(MeshConfig(
                type="fesom2", meshpath=str(mp), rotation=rotation))
            ds = reader.open_dataset(str(dp))
            with TrajectoryWriter(str(out), 1) as w:
                track_particles(
                    mesh, ds, reader, np.array([5.0]), np.array([45.0]),
                    start_time_idx=0, n_steps=2, dt=3600.0,
                    integrator="euler", output_freq=1, mode=mode, writer=w)
            ds.close()

        def _read_attr(out):
            if ext == ".zarr":
                import zarr
                store = zarr.open_group(str(out), mode="r")
                return store.attrs.get("rotation_angles", None)
            import netCDF4
            with netCDF4.Dataset(out) as f:
                if "rotation_angles" not in f.ncattrs():
                    return None
                return np.asarray(f.getncattr("rotation_angles"), dtype=float)

        out_r = tmp_path / f"direct_rot{ext}"
        _run("fesom2", out_r)
        got = _read_attr(out_r)
        assert got is not None
        np.testing.assert_array_equal(
            np.asarray(got, dtype=float), np.array([50.0, 15.0, -90.0]))

        out_n = tmp_path / f"direct_none{ext}"
        _run(None, out_n)
        assert _read_attr(out_n) is None


# ---------------------------------------------------------------------------
# Polar capability (Task 4 scope): the transpolar great-circle trajectory —
# the whole point of the rotated frame — and node-layout equivalence.
# ---------------------------------------------------------------------------

#: Rotated-frame longitude of the geographic North Pole's image under the
#: FESOM2 Euler matrix.  With FESOM2_ANGLES the pole maps to lat_r = 90 - beta
#: = 75 at this longitude; pinned by ``test_pole_image_geometry`` below so a
#: refactor of euler_matrix can't silently move the carrier meridian.
_POLE_LON_R = -180.0
_POLE_LAT_R = 75.0


def _arbitrary_grid_elems(nx, ny):
    """Two triangles per quad over a row-major (ny x nx) node grid.

    Node index ``j*nx + i`` (row j, col i).  Each interior quad splits into
    a lower-left and upper-right triangle, giving ``2*(nx-1)*(ny-1)``
    triangles — enough connectivity that the interior nodes carry a full
    ring of elements for robust containment/interpolation.
    """
    elems = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            n1 = n0 + 1
            n2 = n0 + nx
            n3 = n2 + 1
            elems.append([n0, n1, n3])
            elems.append([n0, n3, n2])
    return np.array(elems, dtype=np.int32)


def _make_fesom2_uv_node(tmp_path, u_node, v_node, *, tag="uvnode", n_time=60):
    """FESOM2 NODE-layout velocity file (``unod``/``vnod`` on the nod2 dim).

    ``u_node``/``v_node`` are the GEOGRAPHIC components at the mesh nodes,
    broadcast over time.  Returns the file path.  Read with
    ``ModeConfig(var_u="unod", var_v="vnod")`` -> ``uv_layout == "node"``.
    """
    u_node = np.asarray(u_node, dtype=np.float64)
    v_node = np.asarray(v_node, dtype=np.float64)
    n_nodes = u_node.shape[0]
    import pandas as pd
    ds = xr.Dataset({
        "unod": (["time", "nod2"],
                 np.broadcast_to(u_node[None, :], (n_time, n_nodes)).copy()),
        "vnod": (["time", "nod2"],
                 np.broadcast_to(v_node[None, :], (n_time, n_nodes)).copy()),
        "ssh":  (["time", "nod2"], np.zeros((n_time, n_nodes))),
    }, coords={"time": pd.date_range("2020-01-01", periods=n_time,
                                     freq="3600s")})
    p = tmp_path / f"fesom2_{tag}.nc"
    ds.to_netcdf(p)
    return p


def _xyz_chord_m(lon1, lat1, lon2, lat2):
    """3D chord distance (metres) between two lon/lat points on the sphere.

    Uses ``R_EARTH_STANDARD`` (the great-circle reference radius).  The
    straight-line chord avoids any longitude-wrap pathology at the pole,
    where two points an epsilon apart can have arbitrarily different
    longitudes but a tiny chord.
    """
    def _u(lo, la):
        lo = np.deg2rad(lo); la = np.deg2rad(la)
        return np.array([np.cos(la) * np.cos(lo),
                         np.cos(la) * np.sin(lo), np.sin(la)])
    return float(np.linalg.norm(_u(lon1, lat1) - _u(lon2, lat2))
                 * R_EARTH_STANDARD)


def _polar_rotated_grid(R):
    """Build the polar mesh in the ROTATED frame, return geographic coords.

    A regular 13x13 grid spanning rotated lat_r 69..81, lon_r
    ``_POLE_LON_R`` +/- 6 deg at ~1-deg spacing.  The rotated image of the
    geographic North Pole sits at (lon_r=_POLE_LON_R, lat_r=75) — squarely
    inside this patch — so the GEOGRAPHIC mesh (returned here) genuinely
    covers the geographic pole (a node lands exactly on |lat|=90).
    Returns ``(lon_geo, lat_geo, elems)``.
    """
    lat_r_axis = np.arange(69.0, 81.0 + 0.5, 1.0)
    lon_r_axis = _POLE_LON_R + np.arange(-6.0, 6.0 + 0.5, 1.0)
    nx = lon_r_axis.size
    ny = lat_r_axis.size
    gx_r, gy_r = np.meshgrid(lon_r_axis, lat_r_axis)
    lon_geo, lat_geo = unrotate_lonlat(gx_r.ravel(), gy_r.ravel(), R)
    elems = _arbitrary_grid_elems(nx, ny)
    return lon_geo, lat_geo, elems


class TestPolarCapability:
    def test_pole_image_geometry(self, tmp_path):
        # Pin the rotated image of the geographic North Pole: with the
        # FESOM2 Euler angles it sits at lat_r = 90 - beta = 75, lon_r =
        # _POLE_LON_R.  This is the constant the whole class depends on —
        # if euler_matrix is refactored and this moves, the carrier
        # meridian assumption below breaks, so fail loudly HERE.
        R = euler_matrix(*FESOM2_ANGLES)
        lon_r, lat_r = rotate_lonlat(np.array([0.0]), np.array([90.0]), R)
        np.testing.assert_allclose(lat_r[0], _POLE_LAT_R, atol=1e-9)
        np.testing.assert_allclose(lat_r[0], 90.0 - FESOM2_ANGLES[1],
                                   atol=1e-9)
        # longitude is well-defined here (the pole is not a fixed point of
        # this rotation); pin it so _POLE_LON_R stays correct.
        np.testing.assert_allclose(lon_r[0], _POLE_LON_R, atol=1e-9)

    def test_transpolar_meridional_flow_follows_great_circle(self, tmp_path):
        # GEODESIC CASE (review-corrected).  Pure rotated-meridional
        # flow along the rotated meridian through the geographic pole IS a
        # great circle (out-of-plane residual measured 0.0 deg below).
        #
        # The mesh is a regular grid built in the ROTATED frame spanning
        # lat_r 69..81 (the rotated image of the geo pole sits at lat_r=75),
        # un-rotated to geographic for the files — a mesh that genuinely
        # covers the geographic North Pole (a node lands on |lat|=90).
        # Field: u_r = 0, v_r = V (constant); the FILE carries its
        # GEOGRAPHIC images via unrotate_vectors at the geographic element
        # centroids (the reader's circular-mean carriers).  Seed at
        # (_POLE_LON_R, 72) in rotated coords -> geographic seed via
        # unrotate_lonlat; V sized so 20 steps sweep exactly 6 deg of lat_r
        # (72 -> 78, crossing 75 = the geographic pole at step 9).
        #
        # MEASURED (rk2, dt=3600, 20 steps): max 3D chord error to the
        # analytic great circle = 6.8e-08 m (68 nanometres) — i.e. machine
        # precision.  This is EXPECTED to be exact, not merely small: a
        # spatially CONSTANT rotated field makes the rotated-frame motion
        # lat_r += dt*V/M (M = METERS_PER_DEG_LAT) with lon_r fixed, which
        # RK2 integrates with ZERO truncation error (constant RHS).  So the
        # 1.0 m bound below clears by ~7 orders; the dt-FLOOR fallback in
        # the task brief is unused (the constant-field error is a round-off
        # floor that does NOT shrink with dt — that is precisely why the
        # genuine-truncation dt-convergence lives in its own companion test
        # test_transpolar_convergence_with_dt, NOT here).
        from kelp.tracking import track_particles

        R = euler_matrix(*FESOM2_ANGLES)
        lon_geo, lat_geo, elems = _polar_rotated_grid(R)
        # mesh genuinely covers the geographic pole.
        assert np.isclose(np.abs(lat_geo).max(), 90.0, atol=1e-6)

        dt = 3600.0
        n_steps = 20
        V = 6.0 * METERS_PER_DEG_LAT / (n_steps * dt)   # ~9.26 m/s

        # element carriers: geographic circular-mean centroids (reader conv).
        geo_clon, geo_clat = _circular_mean_centres(lon_geo, lat_geo, elems)
        n_elem = elems.shape[0]
        u_geo, v_geo = unrotate_vectors(
            np.zeros(n_elem), np.full(n_elem, V), geo_clon, geo_clat, R)

        mp = _make_fesom2_diag(tmp_path, lon_geo, lat_geo, elems,
                               tag="polargc")
        # element-layout velocity file (carries the geographic image field).
        uvp = _make_fesom2_uv(tmp_path, lon_geo, lat_geo, elems,
                              u=u_geo, v=v_geo, layout="element", n_time=60,
                              tag="polargc")

        reader = get_reader("fesom2")
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", RuntimeWarning)  # no pole warn
            mesh = reader.load_mesh(MeshConfig(
                type="fesom2", meshpath=str(mp), rotation="fesom2"))
        ds = reader.open_dataset(str(uvp))
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)

        seed_lon, seed_lat = unrotate_lonlat(
            np.array([_POLE_LON_R]), np.array([72.0]), R)
        rec = track_particles(
            mesh, ds, reader, seed_lon.copy(), seed_lat.copy(),
            start_time_idx=0, n_steps=n_steps, dt=dt, integrator="rk2",
            output_freq=1, mode=mode).to_arrays()
        ds.close()

        lon_t = rec["lon"][:, 0]     # to_arrays -> (T, P), P = 1
        lat_t = rec["lat"][:, 0]

        # per-record chord error vs the analytic great circle.  Record k is
        # POST-STEP, so it is the position after (k+1) steps -> t=(k+1)*dt.
        max_chord = 0.0
        for k in range(lon_t.shape[0]):
            t = (k + 1) * dt
            latr_k = 72.0 + V * t / METERS_PER_DEG_LAT
            ref_lon, ref_lat = unrotate_lonlat(
                np.array([_POLE_LON_R]), np.array([latr_k]), R)
            max_chord = max(max_chord, _xyz_chord_m(
                lon_t[k], lat_t[k], ref_lon[0], ref_lat[0]))
        # the great circle is followed to machine precision (see docstring).
        assert max_chord < 1.0, max_chord

        # we really crossed the pole region: a recorded geographic |lat|
        # essentially at 90 (the seed's meridian passes exactly through it).
        assert np.abs(lat_t).max() > 89.5, np.abs(lat_t).max()

        # the geographic longitude flips ~180 deg across the crossing: take
        # the records straddling the max-|lat| record and assert the wrapped
        # longitude difference exceeds 150 deg (a meridian crossing the pole
        # reappears on the antipodal meridian).
        imax = int(np.argmax(np.abs(lat_t)))
        assert 0 < imax < lon_t.shape[0] - 1, imax
        dlon = lon_t[imax + 1] - lon_t[imax - 1]
        dlon = ((dlon + 180.0) % 360.0) - 180.0
        assert abs(dlon) > 150.0, (abs(dlon), lon_t[imax - 1], lon_t[imax + 1])

        # recorded geographic vel diagnostics rotate smoothly: |vel| stays
        # ~const (the un-rotated speed equals v_r = V everywhere up to the
        # carrier shift).  MEASURED speed spread < 1e-13 m/s.
        assert "vel_u" in rec and "vel_v" in rec
        spd = np.sqrt(rec["vel_u"][:, 0] ** 2 + rec["vel_v"][:, 0] ** 2)
        np.testing.assert_allclose(spd, V, rtol=1e-9)

    @pytest.mark.parametrize("integrator", ["euler", "rk2"])
    def test_record_velocity_unrotated_at_step_start(self, tmp_path,
                                                     integrator):
        # vel_u/vel_v are the velocity the integrator evaluated at the
        # step's START position; the geographic components of a tangent
        # vector depend on its base point, so the un-rotation must use
        # that start position, not the record (end) position.  Across the
        # geographic pole the two bases differ by ~180 deg of direction.
        from kelp.tracking import track_particles

        R = euler_matrix(*FESOM2_ANGLES)
        lon_geo, lat_geo, elems = _polar_rotated_grid(R)
        dt = 3600.0
        n_steps = 20
        V = 6.0 * METERS_PER_DEG_LAT / (n_steps * dt)
        geo_clon, geo_clat = _circular_mean_centres(lon_geo, lat_geo, elems)
        n_elem = elems.shape[0]
        u_geo, v_geo = unrotate_vectors(
            np.zeros(n_elem), np.full(n_elem, V), geo_clon, geo_clat, R)
        mp = _make_fesom2_diag(tmp_path, lon_geo, lat_geo, elems,
                               tag="polarvel")
        uvp = _make_fesom2_uv(tmp_path, lon_geo, lat_geo, elems,
                              u=u_geo, v=v_geo, layout="element", n_time=60,
                              tag="polarvel")
        reader = get_reader("fesom2")
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", RuntimeWarning)
            mesh = reader.load_mesh(MeshConfig(
                type="fesom2", meshpath=str(mp), rotation="fesom2"))
        ds = reader.open_dataset(str(uvp))
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)
        seed_lon, seed_lat = unrotate_lonlat(
            np.array([_POLE_LON_R]), np.array([72.0]), R)
        rec = track_particles(
            mesh, ds, reader, seed_lon.copy(), seed_lat.copy(),
            start_time_idx=0, n_steps=n_steps, dt=dt, integrator=integrator,
            output_freq=1, mode=mode).to_arrays()
        ds.close()

        lon_t = rec["lon"][:, 0]
        lat_t = rec["lat"][:, 0]
        base_lon = np.concatenate([seed_lon, lon_t[:-1]])
        base_lat = np.concatenate([seed_lat, lat_t[:-1]])
        exp_u, exp_v = unrotate_vectors(
            np.zeros(n_steps), np.full(n_steps, V), base_lon, base_lat, R)
        np.testing.assert_allclose(rec["vel_u"][:, 0], exp_u, atol=1e-9)
        np.testing.assert_allclose(rec["vel_v"][:, 0], exp_v, atol=1e-9)

    def test_rk45_record_velocity_base_is_pre_kick_end(self, tmp_path):
        # rk45 records the velocity of its last accepted substep, i.e. at
        # the step's END position BEFORE the split diffusion kick moves the
        # particle; the un-rotation must use that point, not the kicked
        # record position (PR review 2026-10-14).  The field is constant in
        # the rotated chart (u_r = 0, v_r = V), so the pre-kick end is the
        # previous record moved by V*dt/M in rotated latitude.
        from kelp.core.config import DiffusionConfig
        from kelp.tracking import track_particles

        R = euler_matrix(*FESOM2_ANGLES)
        lon_geo, lat_geo, elems = _polar_rotated_grid(R)
        dt = 3600.0
        n_steps = 6
        V = 6.0 * METERS_PER_DEG_LAT / (20 * dt)
        geo_clon, geo_clat = _circular_mean_centres(lon_geo, lat_geo, elems)
        n_elem = elems.shape[0]
        u_geo, v_geo = unrotate_vectors(
            np.zeros(n_elem), np.full(n_elem, V), geo_clon, geo_clat, R)
        mp = _make_fesom2_diag(tmp_path, lon_geo, lat_geo, elems,
                               tag="polarrk45")
        uvp = _make_fesom2_uv(tmp_path, lon_geo, lat_geo, elems,
                              u=u_geo, v=v_geo, layout="element", n_time=60,
                              tag="polarrk45")
        reader = get_reader("fesom2")
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", RuntimeWarning)
            mesh = reader.load_mesh(MeshConfig(
                type="fesom2", meshpath=str(mp), rotation="fesom2"))
        ds = reader.open_dataset(str(uvp))
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)
        seed_lon, seed_lat = unrotate_lonlat(
            np.array([_POLE_LON_R]), np.array([72.0]), R)
        diff = DiffusionConfig(enabled=True, scheme="uniform",
                               kh_zonal=200.0, kh_meridional=200.0, seed=3)
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", RuntimeWarning)
            rec = track_particles(
                mesh, ds, reader, seed_lon.copy(), seed_lat.copy(),
                start_time_idx=0, n_steps=n_steps, dt=dt, integrator="rk45",
                output_freq=1, mode=mode, diffusion=diff).to_arrays()
        ds.close()

        lon_t = rec["lon"][:, 0]
        lat_t = rec["lat"][:, 0]
        assert np.isfinite(rec["vel_u"][:, 0]).all()
        prev_lon = np.concatenate([seed_lon, lon_t[:-1]])
        prev_lat = np.concatenate([seed_lat, lat_t[:-1]])
        plon_r, plat_r = rotate_lonlat(prev_lon, prev_lat, R)
        base_lon, base_lat = unrotate_lonlat(
            plon_r, plat_r + V * dt / METERS_PER_DEG_LAT, R)
        exp_u, exp_v = unrotate_vectors(
            np.zeros(n_steps), np.full(n_steps, V), base_lon, base_lat, R)
        np.testing.assert_allclose(rec["vel_u"][:, 0], exp_u, atol=1e-9)
        np.testing.assert_allclose(rec["vel_v"][:, 0], exp_v, atol=1e-9)

    def test_transpolar_convergence_with_dt(self, tmp_path):
        # The dt-convergence DIFFERENTIAL DIAGNOSTIC for the transpolar run.
        #
        # The headline test uses a spatially CONSTANT rotated field, whose
        # RK2 truncation error is exactly zero (constant RHS) — so it sits
        # on a round-off floor (~1e-7 m) that does NOT shrink with dt and
        # cannot demonstrate convergence.  To exercise genuine RK2
        # truncation WHILE staying on the exact great circle, this variant
        # uses u_r = 0 everywhere (so the motion is still confined to the
        # fixed rotated meridian lon_r = _POLE_LON_R — a great circle, with
        # out-of-plane residual MEASURED 0.0 deg) but a v_r that varies
        # smoothly with rotated latitude:  v_r(lat_r) = V0*(1 + 0.5*(lat_r -
        # 75)/6).  That makes d(lat_r)/dt = v_r(lat_r)/M genuinely
        # nonlinear, so RK2 carries an O(dt^2) error.
        #
        # The analytic reference integrates that nonlinear ODE with a tight
        # solver (solve_ivp, rtol 1e-12).  MEASURED max chord error:
        # dt=3600 -> 84.3 m, dt=1800 -> 21.3 m, RATIO 3.96 (~4 = quadratic,
        # RK2).  A FRAME BUG (wrong/missing un-rotation) records a rotated
        # offset that is dt-INDEPENDENT -> ratio ~1.0; the > 2.5 assertion
        # separates the two.
        from scipy.integrate import solve_ivp

        from kelp.tracking import track_particles

        R = euler_matrix(*FESOM2_ANGLES)
        lon_geo, lat_geo, elems = _polar_rotated_grid(R)
        n_elem = elems.shape[0]
        M = METERS_PER_DEG_LAT
        lat_r0 = 72.0
        V0 = 6.0 * M / (20 * 3600.0)

        def v_of_latr(latr):
            return V0 * (1.0 + 0.5 * (latr - 75.0) / 6.0)

        # build the field at element centroids using THEIR rotated latitude.
        geo_clon, geo_clat = _circular_mean_centres(lon_geo, lat_geo, elems)
        _clon_r, clat_r = rotate_lonlat(geo_clon, geo_clat, R)
        u_geo, v_geo = unrotate_vectors(
            np.zeros(n_elem), v_of_latr(clat_r), geo_clon, geo_clat, R)

        mp = _make_fesom2_diag(tmp_path, lon_geo, lat_geo, elems,
                               tag="polarconv")
        uvp = _make_fesom2_uv(tmp_path, lon_geo, lat_geo, elems,
                              u=u_geo, v=v_geo, layout="element", n_time=90,
                              tag="polarconv")

        reader = get_reader("fesom2")
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", RuntimeWarning)
            mesh = reader.load_mesh(MeshConfig(
                type="fesom2", meshpath=str(mp), rotation="fesom2"))
        ds = reader.open_dataset(str(uvp))
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)
        seed_lon, seed_lat = unrotate_lonlat(
            np.array([_POLE_LON_R]), np.array([lat_r0]), R)

        def _ref_latr(t):
            sol = solve_ivp(lambda tt, y: v_of_latr(y[0]) / M, (0.0, t),
                            [lat_r0], rtol=1e-12, atol=1e-14)
            return float(sol.y[0, -1])

        def _run(dt_, n_steps_):
            rec = track_particles(
                mesh, ds, reader, seed_lon.copy(), seed_lat.copy(),
                start_time_idx=0, n_steps=n_steps_, dt=dt_, integrator="rk2",
                output_freq=1, mode=mode, temporal_interp=True).to_arrays()
            lon_t = rec["lon"][:, 0]; lat_t = rec["lat"][:, 0]
            mx = 0.0
            for k in range(lon_t.shape[0]):
                t = (k + 1) * dt_
                rlon, rlat = unrotate_lonlat(
                    np.array([_POLE_LON_R]), np.array([_ref_latr(t)]), R)
                mx = max(mx, _xyz_chord_m(lon_t[k], lat_t[k],
                                          rlon[0], rlat[0]))
            return mx, rec

        d_coarse, rec_c = _run(3600.0, 20)
        d_fine, _ = _run(1800.0, 40)
        ds.close()

        # quadratic convergence (RK2) -> ratio ~4; a frame bug is ratio ~1.
        assert d_fine < d_coarse / 2.5, (d_coarse, d_fine)

        # out-of-plane sanity: every record stays on the rotated meridian
        # lon_r = _POLE_LON_R (the great-circle plane).  MEASURED 0.0 deg.
        lon_r_rec, _lat_r_rec = rotate_lonlat(
            rec_c["lon"][:, 0], rec_c["lat"][:, 0], R)
        dlonr = ((lon_r_rec - _POLE_LON_R + 180.0) % 360.0) - 180.0
        assert np.abs(dlonr).max() < 1e-6, np.abs(dlonr).max()

    def test_unrotated_polar_run_is_garbage(self, tmp_path):
        # MOTIVATION PIN: the SAME physical files tracked WITHOUT rotation
        # must fail to follow the great circle.  The geographic mesh
        # straddles the pole (a node at |lat|=90, cos(lat)->0 metric
        # blowup) and cells wrap across it, so the degenerate geographic
        # chart breaks down.  Three legitimate failure modes are accepted
        # (assert at least one):
        #   (a) any particle OUTSIDE / lost (containment failure on a
        #       pole-straddling cell),
        #   (b) the run RAISES (caught and accepted),
        #   (c) max chord deviation > 100x the rotated run's max (garbage
        #       displacement from the blown-up metric).
        # MEASURED here: BOTH (a) (status -> OUTSIDE at step 9) AND (c)
        # (garbage 3.8e5 m vs rotated 6.8e-8 m -> ratio ~5e12).  The
        # near-pole RuntimeWarning that fires on the rotation=None load is
        # explicitly absorbed (it is the EXPECTED warning, not the subject
        # of this test).
        from kelp.core.particles import ParticleStatus
        from kelp.tracking import track_particles

        R = euler_matrix(*FESOM2_ANGLES)
        lon_geo, lat_geo, elems = _polar_rotated_grid(R)
        n_elem = elems.shape[0]
        dt = 3600.0
        n_steps = 20
        V = 6.0 * METERS_PER_DEG_LAT / (n_steps * dt)

        geo_clon, geo_clat = _circular_mean_centres(lon_geo, lat_geo, elems)
        u_geo, v_geo = unrotate_vectors(
            np.zeros(n_elem), np.full(n_elem, V), geo_clon, geo_clat, R)

        mp = _make_fesom2_diag(tmp_path, lon_geo, lat_geo, elems,
                               tag="polargarbage")
        uvp = _make_fesom2_uv(tmp_path, lon_geo, lat_geo, elems,
                              u=u_geo, v=v_geo, layout="element", n_time=60,
                              tag="polargarbage")

        reader = get_reader("fesom2")
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)
        seed_lon, seed_lat = unrotate_lonlat(
            np.array([_POLE_LON_R]), np.array([72.0]), R)

        # reference: the CORRECT rotated run (chord error ~1e-7 m).
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", RuntimeWarning)
            mesh_r = reader.load_mesh(MeshConfig(
                type="fesom2", meshpath=str(mp), rotation="fesom2"))
        ds_r = reader.open_dataset(str(uvp))
        rot = track_particles(
            mesh_r, ds_r, reader, seed_lon.copy(), seed_lat.copy(),
            start_time_idx=0, n_steps=n_steps, dt=dt, integrator="rk2",
            output_freq=1, mode=mode).to_arrays()
        ds_r.close()
        rot_lon = rot["lon"][:, 0]; rot_lat = rot["lat"][:, 0]
        rot_max = 0.0
        for k in range(rot_lon.shape[0]):
            t = (k + 1) * dt
            latr_k = 72.0 + V * t / METERS_PER_DEG_LAT
            rlon, rlat = unrotate_lonlat(
                np.array([_POLE_LON_R]), np.array([latr_k]), R)
            rot_max = max(rot_max, _xyz_chord_m(
                rot_lon[k], rot_lat[k], rlon[0], rlat[0]))

        # the garbage run: load WITHOUT rotation (absorbing the expected
        # near-pole warning) and track in the degenerate geographic chart.
        # SETUP (mesh load + dataset open) lives OUTSIDE the try: a failure
        # here is a construction/fixture bug and MUST propagate, never be
        # silently absorbed as outcome (b).
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", RuntimeWarning)  # near-pole warn
            mesh_n = reader.load_mesh(MeshConfig(
                type="fesom2", meshpath=str(mp)))
        ds_n = reader.open_dataset(str(uvp))

        outcome_lost = False
        outcome_raised = False
        garbage_max = None
        g = None
        try:
            # ONLY the no-rotation tracking call is guarded: the degenerate
            # geographic chart (cos(lat)->0 metric blow-up, pole-straddling
            # cells) may legitimately raise from the drift/containment kernels
            # (e.g. ValueError / ZeroDivisionError / FloatingPointError /
            # numpy LinAlgError) -> outcome (b).  A different exception class
            # would now surface instead of being mislabelled as a pass.
            g = track_particles(
                mesh_n, ds_n, reader, seed_lon.copy(), seed_lat.copy(),
                start_time_idx=0, n_steps=n_steps, dt=dt, integrator="rk2",
                output_freq=1, mode=mode).to_arrays()
        except Exception:        # noqa: BLE001 — any failure is outcome (b)
            outcome_raised = True
        finally:
            ds_n.close()

        if g is not None:
            outcome_lost = bool(np.any(g["status"] != ParticleStatus.ALIVE))
            g_lon = g["lon"][:, 0]; g_lat = g["lat"][:, 0]
            garbage_max = 0.0
            for k in range(g_lon.shape[0]):
                t = (k + 1) * dt
                latr_k = 72.0 + V * t / METERS_PER_DEG_LAT
                rlon, rlat = unrotate_lonlat(
                    np.array([_POLE_LON_R]), np.array([latr_k]), R)
                garbage_max = max(garbage_max, _xyz_chord_m(
                    g_lon[k], g_lat[k], rlon[0], rlat[0]))

        outcome_huge = (garbage_max is not None
                        and garbage_max > 100.0 * max(rot_max, 1e-12))
        assert outcome_lost or outcome_raised or outcome_huge, (
            f"degenerate chart did not fail: lost={outcome_lost} "
            f"raised={outcome_raised} garbage_max={garbage_max} "
            f"rot_max={rot_max}")


class TestNodeLayoutEquivalence:
    def test_headline_equivalence_node_layout(self, tmp_path):
        # The Task-3 headline equivalence pin, re-run with NODE-layout
        # velocities.  Node layout REQUIRES uv_interp="barycentric" (the
        # leastsq drift core indexes element-centre values and the tracking
        # layer raises for node + leastsq).  ONE physical dataset: constant
        # GEOGRAPHIC eastward node field (unod=1, vnod=0); the reader
        # rotates the mesh + node vectors when rotation is set, so the same
        # on-disk file is the single source of truth for both runs.
        #
        # MEASURED (3x3 FESOM2 grid, lat 5, lon 0, rk2, barycentric):
        # d at dt=3600 = 1.3e-6 deg (< 1e-4, the chart bound; a missing/
        # double un-rotation would be O(tens of degrees), dt-INDEPENDENT).
        # Both runs individually reproduce the analytic constant-eastward
        # solution (lat stays 5.0, lon = k*dt/(cos(5)*M)); the ~1e-6
        # difference is the legitimate chart discrepancy.
        #
        # CONVERGENCE ORDER (this is the part that differs from the
        # ELEMENT-layout headline, which converges ~quadratically at ratio
        # ~3.5).  The node + barycentric path is FIRST order: barycentric
        # weights are NOT rotation-equivariant (the rotated triangle has a
        # different spherical shape), so the per-step velocity discrepancy
        # produces a chart difference that decays ~linearly in dt and then
        # bottoms out on a round-off floor.  MEASURED at fixed horizon
        # 10800 s: dt=3600 -> 1.335e-6, 1800 -> 6.543e-7 (ratio 2.04),
        # 900 -> 3.137e-7 (ratio 2.09), 450 -> 1.478e-7 (ratio 2.12),
        # 225 -> 1.329e-7 (floor).  So the convergence ASSERTION here uses
        # the cleanest coarse pair (3600/1800, ratio 2.04) and a > 1.5
        # threshold: it still SHRINKS with dt (separating it from a
        # constant-offset frame bug pinned at ratio exactly 1.0), without
        # falsely claiming the quadratic order the barycentric path does
        # not have.
        from kelp.tracking import track_particles

        lon, lat = _grid_coords(lat_centre=5.0, lon_centre=0.0)
        mp = _make_fesom2_diag(tmp_path, lon, lat, _GRID_ELEMS_0, tag="nodehl")
        u_node = np.ones(lon.shape[0])
        v_node = np.zeros(lon.shape[0])
        dp = _make_fesom2_uv_node(tmp_path, u_node, v_node, tag="nodehl")
        reader = get_reader("fesom2")
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0,
                          var_u="unod", var_v="vnod")

        def _run(rotation, dt, n_steps):
            mesh = reader.load_mesh(MeshConfig(
                type="fesom2", meshpath=str(mp), rotation=rotation))
            ds = reader.open_dataset(str(dp))
            rec = track_particles(
                mesh, ds, reader, np.array([0.0]), np.array([5.0]),
                start_time_idx=0, n_steps=n_steps, dt=dt, integrator="rk2",
                output_freq=1, mode=mode, uv_interp="barycentric")
            ds.close()
            return rec.to_arrays()

        def _diff(a, b):
            dlon = np.abs(b["lon"] - a["lon"])
            dlat = np.abs(b["lat"] - a["lat"])
            return float(max(dlon.max(), dlat.max()))

        # Assertion 1: the legitimate chart difference, far below a bug.
        a_coarse = _run(None, 3600.0, 3)
        b_coarse = _run("fesom2", 3600.0, 3)
        d_coarse = _diff(a_coarse, b_coarse)
        assert d_coarse < 1e-4, d_coarse

        # Assertion 2: a TIGHT magnitude bound on the frame difference.
        # NB (2026-06): the RK2 z0-metric fix (final lon update divides by z0 at
        # the MIDPOINT latitude) made the metric error 2nd-order and shrank
        # d_coarse from ~1.3e-6 to ~1.2e-7 (~11x). What remains is the constant,
        # dt-INDEPENDENT interpolation non-equivariance (rotate->interp !=
        # interp->rotate), which no longer converges — so a convergence-ratio
        # check can no longer tell it from a constant-offset bug. Bound the
        # magnitude instead: it passes the legit ~1.2e-7 residual but fails both
        # a frame bug (O(tens of deg)) and a regression of the RK2 metric (which
        # re-inflates it to ~1.3e-6). RK2's longitude ORDER is pinned separately
        # by TestRk2LongitudeIsSecondOrder.
        assert d_coarse < 5e-7, d_coarse

        # Assertion 3: recorded velocity diagnostics are geographic — run B
        # stays ~eastward (u ~ 1, |v| << 1) like run A, in the OUTPUT,
        # despite running in the rotated frame.
        assert "vel_u" in b_coarse and "vel_v" in b_coarse
        assert np.all(np.abs(b_coarse["vel_v"]) < 1e-3)
        assert np.all(b_coarse["vel_u"] > 0.5)


class TestRk2LongitudeIsSecondOrder:
    """RK2's final longitude update must divide the MIDPOINT velocity by z0 at
    the MIDPOINT latitude (not the start) — that is what makes RK2 longitude 2nd
    order instead of 1st (Euler-level). A constant field makes leastsq exact, so
    ONE rk2 step has a closed form: lon = lon0 + dt*U/z0(lat_mid). Pin that the
    tracker matches z0(MIDPOINT), not the z0(START) bug, on a meridional flow at
    high latitude (z0 changes a lot across the half-step)."""

    def test_rk2_longitude_metric_uses_midpoint(self, tmp_path):
        from kelp.tracking import track_particles
        from kelp.core.geometry import METERS_PER_DEG_LAT as M
        U, V, dt, lat0 = 1.0, 2.0, 43200.0, 60.0
        mp, dp, _, _ = _midlat_fesom2_files(
            tmp_path, tag="rk2mid", lat_centre=lat0, lon_centre=0.0,
            u_val=U, v_val=V, n_time=4)
        reader = get_reader("fesom2")
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)
        mesh = reader.load_mesh(MeshConfig(type="fesom2", meshpath=str(mp)))
        ds = reader.open_dataset(str(dp))
        rec = track_particles(
            mesh, ds, reader, np.array([0.0]), np.array([lat0]),
            start_time_idx=0, n_steps=1, dt=dt,
            integrator="rk2", output_freq=1, mode=mode, temporal_interp=True)
        ds.close()
        lon_new = float(np.asarray(rec.to_arrays()["lon"]).ravel()[-1])

        lat_mid = lat0 + 0.5 * dt * V / M
        lon_fix = dt * U / (np.cos(np.radians(lat_mid)) * M)   # z0(MIDPOINT) — correct
        lon_bug = dt * U / (np.cos(np.radians(lat0)) * M)      # z0(START) — the old bug
        # exact (constant field -> leastsq exact): matches the midpoint metric...
        assert abs(lon_new - lon_fix) < 1e-6, (lon_new, lon_fix)
        # ...and is clearly NOT the start-metric value (they differ by ~0.02 deg).
        assert abs(lon_new - lon_bug) > 1e-3, (lon_new, lon_bug)
