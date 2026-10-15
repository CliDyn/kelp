"""Vertical-velocity sign contract of the readers (review 2026-09-22, M01).

KELP's internal ``VelocityField.w`` is the positive-DOWN rate of the tracked
depth.  The producers write the opposite: FESOM2 ``w`` (``oce_ale.F90``,
``vert_vel_ale``: ``dh/dt + div(hu) + (w^t - w^b) = 0``) and the HEALPix
products (CORE2 twins remapped from it; ng5 ``avg_wo`` "Time-mean upward sea
water velocity") are positive UP.  Before this fix both readers passed the
file value through, so every FESOM2/HEALPix 3-D run moved particles the
wrong way vertically.  Resolution order of the file's convention:
``mode.w_positive`` (up|down) > the variable's CF ``positive`` attribute >
the reader family default (fesom2: up, healpix: up).

The data-gated physics tests read real output where the answer is known
from physics: at 50 m in the equatorial Pacific cold tongue the annual and
monthly mean flow is UPWELLING, so the internal (positive-down) rate must be
negative there.
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kelp.core.config import ModeConfig  # noqa: E402
from kelp.readers import get_reader, resolve_w_sign  # noqa: E402


def _mode_3d(**kw):
    base = dict(Fdrift2D=False, Fdrift3D=True, time_dir=1.0,
                sigma_layer=None, sigma_mean=False, var_u=None,
                var_v=None, var_w=None, var_eta=None, Vpuls=False,
                w_positive="auto")
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def fesom2_mesh(tmp_path):
    from test_readers import _make_fesom2_mesh
    mesh_dir = tmp_path / "mesh"
    mesh_dir.mkdir()
    _make_fesom2_mesh(mesh_dir)
    return get_reader("fesom2").load_mesh(
        SimpleNamespace(meshpath=str(mesh_dir)))


def _fesom2_3d_file(path, w_value, w_attrs=None):
    n_time, n_elem, n_node, n_ifc = 2, 2, 4, 5
    ds = xr.Dataset({
        "u": (["time", "nz1", "elem"], np.zeros((n_time, n_ifc - 1, n_elem))),
        "v": (["time", "nz1", "elem"], np.zeros((n_time, n_ifc - 1, n_elem))),
        "w": (["time", "nz", "nod2"], np.full((n_time, n_ifc, n_node), w_value)),
        "ssh": (["time", "nod2_node"], np.zeros((n_time, n_node))),
    })
    if w_attrs:
        ds["w"].attrs.update(w_attrs)
    ds.to_netcdf(path)
    return path


def _read_w(mesh, path, mode):
    reader = get_reader("fesom2")
    ds = reader.open_dataset(str(path))
    try:
        return reader.read_velocities(ds, 0, mesh, mode).w
    finally:
        ds.close()


class TestResolveWSign:
    def test_family_default_up_flips(self):
        da = xr.DataArray(np.zeros(3))
        assert resolve_w_sign(da, _mode_3d(), "up") == -1.0

    def test_family_default_down_keeps(self):
        da = xr.DataArray(np.zeros(3))
        assert resolve_w_sign(da, _mode_3d(), "down") == 1.0

    def test_cf_attribute_beats_family_default(self):
        da = xr.DataArray(np.zeros(3), attrs={"positive": "Down"})
        assert resolve_w_sign(da, _mode_3d(), "up") == 1.0

    def test_config_beats_attribute(self):
        da = xr.DataArray(np.zeros(3), attrs={"positive": "down"})
        assert resolve_w_sign(da, _mode_3d(w_positive="up"), "up") == -1.0

    def test_mode_without_the_field_uses_auto(self):
        # direct callers pass bare namespaces without w_positive
        mode = SimpleNamespace(Fdrift3D=True)
        assert resolve_w_sign(xr.DataArray(np.zeros(3)), mode, "up") == -1.0

    def test_unknown_attribute_value_falls_back_to_default(self):
        da = xr.DataArray(np.zeros(3), attrs={"positive": "sideways"})
        assert resolve_w_sign(da, _mode_3d(), "up") == -1.0


class TestConfigKey:
    def test_default_auto(self):
        assert ModeConfig().w_positive == "auto"

    def test_normalised_and_validated(self):
        assert ModeConfig(w_positive="UP").w_positive == "up"
        with pytest.raises(ValueError, match="w_positive"):
            ModeConfig(w_positive="upward")


class TestFesom2Reader:
    def test_native_w_is_positive_up(self, fesom2_mesh, tmp_path):
        p = _fesom2_3d_file(tmp_path / "w_up.nc", 0.01)
        w = _read_w(fesom2_mesh, p, _mode_3d())
        np.testing.assert_allclose(w, -0.01)

    def test_positive_down_attribute_is_honoured(self, fesom2_mesh, tmp_path):
        # utils/reconstruct_w.py writes KELP-internal (positive-down) W and
        # stamps positive="down": it must pass through unchanged
        p = _fesom2_3d_file(tmp_path / "w_down.nc", 0.01,
                            {"positive": "down"})
        w = _read_w(fesom2_mesh, p, _mode_3d())
        np.testing.assert_allclose(w, 0.01)

    def test_config_override_down(self, fesom2_mesh, tmp_path):
        # an OLD reconstruct_w file (no attribute) is declared by config
        p = _fesom2_3d_file(tmp_path / "w_old.nc", 0.01)
        w = _read_w(fesom2_mesh, p, _mode_3d(w_positive="down"))
        np.testing.assert_allclose(w, 0.01)

    def test_repeated_read_of_an_in_memory_dataset(self, fesom2_mesh):
        """The conversion must not write into the dataset: on an in-memory
        dataset ``.values`` is a view, and an in-place negation flipped the
        sign back on the second read of the same snapshot."""
        ds = xr.Dataset({
            "u": (["time", "nz1", "elem"], np.zeros((2, 4, 2))),
            "v": (["time", "nz1", "elem"], np.zeros((2, 4, 2))),
            "w": (["time", "nz", "nod2"], np.full((2, 5, 4), 0.01)),
        })
        reader = get_reader("fesom2")
        for _ in range(3):
            w = reader.read_velocities(ds, 0, fesom2_mesh, _mode_3d()).w
            np.testing.assert_allclose(w, -0.01)
        np.testing.assert_allclose(ds["w"].values, 0.01)

    def test_multiyear_family_default(self, tmp_path):
        from kelp.readers.fesom2 import Fesom2Reader, _load_mesh_nc
        from tests._fesom2_synth import write_meshdiag, write_multiyear
        mesh = _load_mesh_nc(write_meshdiag(tmp_path / "m.nc", "ugrid"))
        yl = {1958: 3, 1959: 3}
        for v in ("unod", "vnod"):
            write_multiyear(tmp_path, v, yl, node_layout=True, vertical="nz1")
        write_multiyear(tmp_path, "w", yl, node_layout=True, vertical="nz")
        mode = SimpleNamespace(Fdrift3D=True, var_u="unod", var_v="vnod",
                               var_w="w", var_eta=None, sample_vars=None,
                               max_levels=None, sigma_layer=None,
                               w_positive="auto")
        ds = Fesom2Reader.open_dataset(str(tmp_path), mode_cfg=mode)
        vel = Fesom2Reader.read_velocities(ds, 4, mesh, mode_cfg=mode)
        np.testing.assert_allclose(vel.w, -0.1, rtol=1e-6)


class TestHealpixReader:
    def test_wo_is_positive_up(self, tmp_path):
        pytest.importorskip("healpy")
        from kelp.readers.healpix import HealpixReader
        from test_healpix_reader import _make_uv_store, _mesh_cfg, LEVELS_M
        store = _make_uv_store(tmp_path / "3D_sign_ocean.nc", nside=4,
                               with_eta=True, with_w=True, w_value=2e-4)
        mode = ModeConfig(Fdrift2D=False, Fdrift3D=True)
        mesh = HealpixReader.load_mesh(_mesh_cfg(store, levels=LEVELS_M),
                                       mode_cfg=mode)
        ds = HealpixReader.open_dataset(str(store), mode_cfg=mode)
        vel = HealpixReader.read_velocities(ds, 0, mesh, mode)
        # the fixture's deepest interface of each column is on the bottom;
        # every finite entry was +2e-4 (upward) -> internal -2e-4
        nonzero = vel.w[vel.w != 0.0]
        assert nonzero.size > 0
        np.testing.assert_allclose(nonzero, -2e-4)


class TestTrackingDirection:
    def test_upward_file_w_moves_particle_up(self, tmp_path):
        """End to end: a file w = +1e-4 m/s (upwelling) must DECREASE depth."""
        pytest.importorskip("healpy")
        from test_healpix_3d import TestTracking3D, _store_3d
        store = _store_3d(tmp_path, name="3D_up_ocean.nc", nside=8,
                          u_value=0.0, w_value=1e-4)
        traj = TestTracking3D()._run(tmp_path, store, [(10.0, 5.0, 60.0)],
                                     n_steps=1)
        # one Euler day at 1e-4 m/s = 8.64 m upward
        assert abs(traj["z"][0, -1] - (60.0 - 8.64)) < 1e-6


# ---------------------------------------------------------------------------
# Physics on real output (data-gated): equatorial Pacific upwelling at 50 m
# ---------------------------------------------------------------------------

_NATIVE = os.environ.get("KELP_TEST_NATIVE_DATA", "")
_TWINS = os.environ.get("KELP_TEST_DATA", "")
_EQPAC = (-150.0, -100.0, -2.0, 2.0)   # lon0, lon1, lat0, lat1
_K50 = 6                               # interface index of 50 m (CORE2 table)


def _eqpac_mask(lon, lat):
    lon = (np.asarray(lon) + 180.0) % 360.0 - 180.0
    lo0, lo1, la0, la1 = _EQPAC
    return (lon >= lo0) & (lon <= lo1) & (lat >= la0) & (lat <= la1)


@pytest.mark.slow
@pytest.mark.skipif(
    not (_NATIVE and all(Path(_NATIVE, f).exists() for f in (
        "fesom.mesh.diag.nc", "unod.fesom.1958.nc", "vnod.fesom.1958.nc",
        "w.fesom.1958.nc"))),
    reason="KELP_TEST_NATIVE_DATA (native FESOM2 CORE2 cycle1 output) not set")
def test_native_fesom2_equatorial_upwelling_decreases_depth():
    reader = get_reader("fesom2")
    mesh = reader.load_mesh(SimpleNamespace(
        meshpath=str(Path(_NATIVE, "fesom.mesh.diag.nc"))))
    ds = xr.merge([xr.open_dataset(Path(_NATIVE, f"{v}.fesom.1958.nc"),
                                   chunks={"time": 1})
                   for v in ("unod", "vnod", "w")])
    mode = _mode_3d(var_u="unod", var_v="vnod", var_w="w")
    mask = _eqpac_mask(mesh.nodes_lon, mesh.nodes_lat)
    assert mask.sum() > 1000
    rates = [reader.read_velocities(ds, t, mesh, mode).w[_K50, mask].mean()
             for t in range(0, 31, 3)]
    ds.close()
    # measured on the raw file: +1.3e-5 m/s annual mean (upward) -> the
    # internal positive-down rate is about -1e-5
    assert np.mean(rates) < -2e-6, rates


@pytest.mark.slow
@pytest.mark.skipif(
    not (_TWINS and Path(_TWINS, "lagdrift_core2_2005_z7.nc").exists()),
    reason="KELP_TEST_DATA (CORE2 HEALPix twin stores) not set")
def test_healpix_twin_equatorial_upwelling_decreases_depth():
    pytest.importorskip("healpy")
    from kelp.core.config import MeshConfig
    from kelp.readers.healpix import HealpixReader
    store = str(Path(_TWINS, "lagdrift_core2_2005_z7.nc"))
    mode = ModeConfig(Fdrift2D=False, Fdrift3D=True)
    mesh = HealpixReader.load_mesh(MeshConfig(type="healpix", meshpath=store),
                                   mode_cfg=mode)
    ds = HealpixReader.open_dataset(store, mode_cfg=mode)
    assert abs(mesh.sigma[_K50] - 50.0) < 1e-6
    mask = _eqpac_mask(mesh.nodes_lon, mesh.nodes_lat)
    assert mask.sum() > 50
    rates = [HealpixReader.read_velocities(ds, t, mesh, mode).w[_K50, mask]
             .mean() for t in range(0, 31, 3)]
    assert np.mean(rates) < -2e-6, rates


def test_cf_standard_name_is_honoured():
    up = xr.DataArray(np.zeros(3), attrs={"standard_name": "upward_sea_water_velocity"})
    dn = xr.DataArray(np.zeros(3), attrs={"standard_name": "downward_sea_water_velocity"})
    assert resolve_w_sign(up, _mode_3d(), "down") == -1.0
    assert resolve_w_sign(dn, _mode_3d(), "up") == 1.0


def test_healpix_w_cannot_be_sampled_as_a_scalar(tmp_path):
    pytest.importorskip("healpy")
    from kelp.readers.healpix import HealpixReader
    from test_healpix_reader import _make_uv_store, _mesh_cfg, LEVELS_M
    store = _make_uv_store(tmp_path / "3D_s_ocean.nc", nside=4, with_eta=True,
                           with_w=True, w_value=1e-4)
    mode = ModeConfig(Fdrift2D=False, Fdrift3D=True)
    mesh = HealpixReader.load_mesh(_mesh_cfg(store, levels=LEVELS_M), mode_cfg=mode)
    ds = HealpixReader.open_dataset(str(store), mode_cfg=mode)
    with pytest.raises(ValueError, match="vel_w"):
        HealpixReader.read_scalar_fields(ds, 0, mesh, ["avg_wo"], mode)
