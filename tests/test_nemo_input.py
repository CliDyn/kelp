"""NEMO input robustness, synthetic: ``domain_cfg.nc`` as the mesh file, several
files per grid (glob / list), the explicit ``u_path``, and the loud refusals
(s-coordinates, ice-shelf cavities, non-T ``sample_vars``, T-point 3-D)."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from types import SimpleNamespace

from kelp.core.config import MeshConfig, ModeConfig
from kelp.readers import get_reader
from tests.test_nemo_cgrid import _grid, _widths

K, NY, NX, DZ = 3, 6, 8, 10.0


def _domain_cfg(tmp_path, *, tmask=False, top=1, e3_bad=False, attrs=None, name="domain_cfg.nc"):
    (flon, flat), (tlon, tlat) = _grid(NY, NX)
    e2u, e1v = _widths(flon, flat)
    bot = np.full((NY, NX), K); bot[0, :] = 0; bot[:, 0] = 0      # the first row / column dry
    lvl = np.arange(K)[:, None, None]
    d2 = lambda a: (("y", "x"), a)
    e3 = np.full((K, NY, NX), DZ)
    if e3_bad:
        e3[0] = 4.0                                                # surface cell != e3t_1d
    dv = {"glamf": d2(flon), "gphif": d2(flat), "glamt": d2(tlon), "gphit": d2(tlat),
          "e1t": d2(np.sqrt(e2u * e1v)), "e2t": d2(np.sqrt(e2u * e1v)), "e2u": d2(e2u), "e1v": d2(e1v),
          "e3t_1d": (("z",), np.full(K, DZ)), "e3t_0": (("z", "y", "x"), e3),
          "e3u_0": (("z", "y", "x"), e3.copy()), "e3v_0": (("z", "y", "x"), e3.copy()),
          "bottom_level": d2(bot), "top_level": d2(np.where(bot > 0, top, 0))}
    if tmask:
        dv["tmask"] = (("z", "y", "x"), ((lvl < bot) & (bot > 0)).astype(float))
    xr.Dataset(dv, attrs=attrs or {}).to_netcdf(tmp_path / name)
    return tmp_path / name


def _load(path):
    return get_reader("nemo").load_mesh(MeshConfig(type="nemo", meshpath=str(path), nemo_grid="cgrid"))


def test_domain_cfg_without_tmask_equals_the_mesh_mask_form(tmp_path):
    a = _load(_domain_cfg(tmp_path, tmask=False, name="a.nc"))
    b = _load(_domain_cfg(tmp_path, tmask=True, name="b.nc"))
    assert a.n_elements == b.n_elements and (a._cgrid_nlev_raster == b._cgrid_nlev_raster).all()
    assert a._cgrid_nlev_raster.max() == K and a._cgrid_zw.size == K + 1


@pytest.mark.parametrize("kw,match", [
    (dict(top=2), "ice-shelf cavities"),
    (dict(attrs={"IsfCav": 1}), "ice-shelf cavities"),
    (dict(attrs={"VertCoord": "sco"}), "s-coordinates"),
    (dict(e3_bad=True), "s-coordinates"),
])
def test_domain_cfg_refusals(tmp_path, kw, match):
    with pytest.raises(NotImplementedError, match=match):
        _load(_domain_cfg(tmp_path, **kw))


def _with_misf(tmp_path, wet_val, land_val):
    import xarray as xr
    p = _domain_cfg(tmp_path, tmask=True)
    ds = xr.open_dataset(p).load()
    ds.close()
    m = np.where(np.asarray(ds["bottom_level"]) > 0, wet_val, land_val)
    ds["misf"] = (("y", "x"), m)
    ds["isfdraft"] = (("y", "x"), np.where(m > 0, 0.0, 50.0) * (wet_val == 99))
    p.unlink()
    ds.to_netcdf(p)
    return p


def test_misf_presence_and_land_values_are_not_cavities(tmp_path):
    assert _load(_with_misf(tmp_path, 1, 0)).n_elements > 0      # ORCA05 mesh_mask pattern
    assert _load(_with_misf(tmp_path, 1, 7)).n_elements > 0


def test_misf_above_one_at_wet_columns_is_a_cavity(tmp_path):
    with pytest.raises(NotImplementedError, match="ice-shelf cavities"):
        _load(_with_misf(tmp_path, 2, 0))


# --- several files per grid ---------------------------------------------------
def _files(tmp_path, nt=4, parts=((0, 2), (2, 4))):
    """Per-part grid_T / grid_U / grid_V files plus the unsplit reference."""
    t = pd.date_range("2020-01-01", periods=nt, freq="D")
    tc = {"time_counter": t}
    full = {"T": {"thetao": (("time_counter", "deptht", "y", "x"), np.random.default_rng(0).normal(size=(nt, 1, NY, NX))),
                  "ssh": (("time_counter", "y", "x"), np.zeros((nt, NY, NX)))},
            "U": {"uo": (("time_counter", "depthu", "y", "x"), np.full((nt, 1, NY, NX), 0.1) * np.arange(nt)[:, None, None, None])},
            "V": {"vo": (("time_counter", "depthv", "y", "x"), np.zeros((nt, 1, NY, NX)))}}
    for g, dv in full.items():
        xr.Dataset(dv, coords=tc).to_netcdf(tmp_path / f"ref_grid_{g}.nc")
        for a, b in parts:
            xr.Dataset(dv, coords=tc).isel(time_counter=slice(a, b)).to_netcdf(
                tmp_path / f"run_{a}_{b}_grid_{g}.nc")
    return tmp_path


def test_glob_and_list_join_the_files_along_time(tmp_path):
    _files(tmp_path)
    rd = get_reader("nemo")
    ref = rd.open_dataset(str(tmp_path / "ref_grid_T.nc"))
    mode = ModeConfig(Fdrift3D=False, sample_vars=["thetao"])
    for spec in (str(tmp_path / "run_*_grid_T.nc"),
                 [str(tmp_path / "run_0_2_grid_T.nc"), str(tmp_path / "run_2_4_grid_T.nc")]):
        ds = rd.open_dataset(spec, mode_cfg=mode)
        assert ds.sizes["time"] == 4 and (ds["time"].values == ref["time"].values).all()
        assert (ds["u"].values == ref["u"].values).all()
        assert (ds["thetao"].values == rd.open_dataset(str(tmp_path / "ref_grid_T.nc"), mode_cfg=mode)["thetao"].values).all()


def test_out_of_order_or_overlapping_files_refused(tmp_path):
    _files(tmp_path)
    rd = get_reader("nemo")
    with pytest.raises(ValueError, match="strictly increasing"):
        rd.open_dataset([str(tmp_path / "run_2_4_grid_T.nc"), str(tmp_path / "run_0_2_grid_T.nc")])


def test_u_file_count_mismatch_refused_by_the_pairing_check(tmp_path):
    _files(tmp_path)
    rd = get_reader("nemo")
    with pytest.raises(ValueError, match="time records"):
        rd.open_dataset(str(tmp_path / "run_*_grid_T.nc"), u_path=str(tmp_path / "run_0_2_grid_U.nc"))


def test_explicit_grid_paths_replace_the_token_siblings(tmp_path):
    _files(tmp_path)
    for g in "TUV":
        (tmp_path / f"ref_grid_{g}.nc").rename(tmp_path / f"plain_{g}.nc")
    rd = get_reader("nemo")
    with pytest.raises(FileNotFoundError, match="ncinput_u"):
        rd.open_dataset(str(tmp_path / "plain_T.nc"))
    ds = rd.open_dataset(str(tmp_path / "plain_T.nc"), u_path=str(tmp_path / "plain_U.nc"),
                         v_path=str(tmp_path / "plain_V.nc"))
    assert ds.sizes["time"] == 4


# --- refusals -------------------------------------------------------------------
def test_non_t_sample_var_refused(tmp_path):
    _files(tmp_path)
    rd = get_reader("nemo")
    with pytest.raises(ValueError, match="U-points"):
        rd.open_dataset(str(tmp_path / "ref_grid_T.nc"), mode_cfg=ModeConfig(Fdrift3D=False, sample_vars=["uo"]))


def test_t_point_path_refuses_3d_in_the_reader(tmp_path):
    _files(tmp_path)
    rd = get_reader("nemo")
    mode = ModeConfig(Fdrift3D=True)
    ds = rd.open_dataset(str(tmp_path / "ref_grid_T.nc"), mode_cfg=mode)
    with pytest.raises(NotImplementedError, match="nemo_grid: cgrid"):
        rd.read_velocities(ds, 0, SimpleNamespace(_cgrid=False), mode_cfg=mode)


def test_glob_input_is_fingerprinted_by_its_files(tmp_path):
    from kelp.tracking import _path_fingerprint
    _files(tmp_path)
    fp = _path_fingerprint(str(tmp_path / "run_*_grid_T.nc"))
    assert [f[0] for f in fp] == ["run_0_2_grid_T.nc", "run_2_4_grid_T.nc"]
    assert _path_fingerprint(str(tmp_path / "nope_*.nc")) == ["missing"]
