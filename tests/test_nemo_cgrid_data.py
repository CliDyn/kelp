"""Data-gated (``KELP_TEST_NEMO_DATA``): the exact C-grid path on real NEMO
crops (built by ``utils/fetch_nemo_test_data.py``): partial-cell W against the model's W (ORCA05), z* detection and
halo-free T-pivot layouts (eORCA025), the global topologies (ORCA05 F-pivot,
ORCA2 T-pivot with halos), input robustness.  Skips when the directory or a
file in it is missing."""
import functools
import os
from pathlib import Path

import numpy as np
import pytest

from kelp.core.config import MeshConfig, ModeConfig
from kelp.readers import get_reader

# absolute: a relative value would break the symlinks made in tmp_path
_d = os.path.abspath(os.environ["KELP_TEST_NEMO_DATA"]) if os.environ.get("KELP_TEST_NEMO_DATA") else None
pytestmark = pytest.mark.skipif(not _d, reason="KELP_TEST_NEMO_DATA not set")


def _read(mesh_file, grid_t, var_u, var_v):
    mode = ModeConfig(Fdrift3D=True, var_u=var_u, var_v=var_v)
    rd = get_reader("nemo")
    mesh = rd.load_mesh(MeshConfig(type="nemo", meshpath=str(mesh_file), nemo_grid="cgrid"))
    ds = rd.open_dataset(str(grid_t), mode_cfg=mode)
    rd.prepare(mesh, ds)
    return mesh, ds, rd.read_velocities(ds, 0, mesh, mode_cfg=mode)


def test_orca05_partial_cell_w_equals_model_w():
    d = Path(_d)
    mesh, ds, vel = _read(d / "orca05_equatorial_mesh.nc", d / "orca05_equatorial_grid_T.nc",
                          "vozocrtx", "vomecrty")
    ny, nx = mesh._structured_shape
    import xarray as xr
    with xr.open_dataset(d / "orca05_equatorial_grid_W.nc") as dw:        # the model's W
        wn = next(n for n in ("vovecrtz", "wo", "w") if n in dw.variables)
        wf = -np.nan_to_num(dw[wn].isel({dw[wn].dims[0]: 0}).values.astype(float))   # positive down
    K = wf.shape[0]
    w = vel.w.reshape(K + 1, ny, nx)[:-1].astype(float)
    wet = np.arange(K)[:, None, None] < mesh._nemo_cgrid_nlev_raster
    err = np.abs(w - wf)[wet].max()
    assert np.abs(wf).max() > 1e-5 and err < 5e-6 * np.abs(wf).max()


def test_eorca025_zstar_warns_and_runs():
    d = Path(_d)
    mesh_file = d / "eorca025_equatorial_domain_cfg.nc"
    grid_t = d / "eorca025_equatorial_split" / "eorca025_1m_19760101_19760229_grid_T.nc"
    for f in (mesh_file, grid_t):
        if not f.exists():
            pytest.skip(f"{f.name} missing under KELP_TEST_NEMO_DATA")
    with pytest.warns(RuntimeWarning, match="z\\*/vvl"):
        mesh, ds, vel = _read(mesh_file, grid_t, "uo", "vo")
    assert np.isfinite(vel.w).all()


# --- topology on real grids ----------------------------------------------------
# ORCA05 (F-pivot, halo columns + halo row), eORCA025 (T-pivot, halo-free), ORCA2
# (T-pivot with halos; CMIP5 output, topology only).  Directories are looked up
# next to / above ``KELP_TEST_NEMO_DATA`` (``nemo_orca05/``, ``esgf_orca2/data/``).
def _sib(*parts):
    for base in (Path(_d), Path(_d).parent, Path(_d).parent.parent):
        if (base.joinpath(*parts)).exists():
            return base.joinpath(*parts)
    pytest.skip(f"{'/'.join(parts)} not found near KELP_TEST_NEMO_DATA")


def _mesh(mesh_file, rotation=None):
    return get_reader("nemo").load_mesh(MeshConfig(
        type="nemo", meshpath=str(mesh_file), nemo_grid="cgrid", rotation=rotation))


def _link_eorca025(tmp_path, region):
    """The eORCA025 crops have no ``grid_T`` token in their names: link them."""
    d = Path(_d)
    for src, dst in ((f"eorca025_{region}_T.nc", "x_grid_T.nc"), (f"eorca025_{region}_U.nc", "x_grid_U.nc"),
                     (f"eorca025_{region}_V.nc", "x_grid_V.nc"), (f"eorca025_{region}_W.nc", "x_grid_W.nc")):
        (tmp_path / dst).symlink_to(d / src)
    return d / f"eorca025_{region}_domain_cfg.nc", tmp_path / "x_grid_T.nc"


def _flux_balance_and_special_edges(mesh, ds, mode):
    """Model normal flux balance over every shared edge (relative), and the
    number of edges that join non-neighbouring raster cells (seam / fold)."""
    from kelp.core.mesh import build_elem_neighbors
    vel = get_reader("nemo").read_velocities(ds, 0, mesh, mode_cfg=mode)
    nx = mesh._structured_shape[1]
    nw = int((mesh._nemo_cgrid_nlev > 0).sum())
    ji = mesh._nemo_cgrid_elem_ji[:nw]
    nb = build_elem_neighbors(mesh)[:nw]
    j, i = ji[:, 0], ji[:, 1]
    fi = np.stack([(j - 1) * nx + i, j * nx + i, j * nx + i, j * nx + i - 1], 1)
    uf, vf = vel.u.astype(float), vel.v.astype(float)
    e2u, e1v = mesh._nemo_cgrid_e2u, mesh._nemo_cgrid_e1v
    out = np.stack([-vf[fi[:, 0]] * e1v[fi[:, 0]], uf[fi[:, 1]] * e2u[fi[:, 1]],
                    vf[fi[:, 2]] * e1v[fi[:, 2]], -uf[fi[:, 3]] * e2u[fi[:, 3]]], 1)
    e, k = np.nonzero((nb >= 0) & (nb < nw))
    n = nb[e, k]
    k2 = np.array([int(np.flatnonzero(nb[b] == a)[0]) for a, b in zip(e, n)])
    special = (np.abs(i[e] - i[n]) > 1) | (np.abs(j[e] - j[n]) > 1)
    return np.abs(out[e, k] + out[n, k2]).max() / np.abs(out).max(), int(special.sum())


def test_orca05_full_mesh_counts():
    from kelp.core.mesh import build_elem_neighbors
    mesh = _mesh(_sib("nemo_orca05", "mesh_mask.nc"), "fesom2")
    nw = int((mesh._nemo_cgrid_nlev > 0).sum())               # the wet cells (the ring follows them)
    assert (np.unique(mesh._nemo_cgrid_corners[:nw]).size, nw) == (229958, 224514)
    nb = build_elem_neighbors(mesh)[:nw]
    assert int(((nb >= 0) & (nb < nw)).sum()) // 2 == 443446
    lay = mesh._nemo_cgrid_layout
    assert (lay.fold, lay.halo_cols, lay.halo_row, lay.periodic) == ("F", True, True, True)


def test_orca05_arctic_fold_flux_balance_and_particles_stay_alive(tmp_path):
    from kelp.core.config import AppConfig, DataConfig, NumericsConfig, TimeConfig
    from kelp.core.particles import ParticleStatus
    from kelp.tracking import run_tracking
    from kelp.writers import load_trajectories
    import xarray as xr
    d = Path(_d)
    mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, var_u="vozocrtx", var_v="vomecrty")
    mesh = _mesh(d / "orca05_arctic_mesh.nc", "fesom2")
    lay = mesh._nemo_cgrid_layout
    assert (lay.fold, lay.halo_cols, lay.halo_row) == ("F", True, True) and lay.shape == (71, 722)
    ds = get_reader("nemo").open_dataset(str(d / "orca05_arctic_grid_T.nc"), mode_cfg=mode)
    err, n_special = _flux_balance_and_special_edges(mesh, ds, mode)
    assert err < 1e-6 and n_special > 100            # seam + fold edges
    m = xr.open_dataset(d / "orca05_arctic_mesh.nc")
    lat, lon, wet = (m[v].values.squeeze() for v in ("gphit", "glamt", "tmask"))
    wet = wet[0] if wet.ndim == 3 else wet
    jj, ii = np.nonzero((wet > 0.5) & (lat > 86.0))
    pick = np.linspace(0, jj.size - 1, 20).astype(int)
    seeds = [(float(lon[jj[p], ii[p]]), float(lat[jj[p], ii[p]])) for p in pick]
    sf = tmp_path / "seeds.dat"
    sf.write_text("".join(f"{a:.9f} {b:.9f}\n" for a, b in seeds))
    out = tmp_path / "t.nc"
    cfg = AppConfig(
        mode=mode, numerics=NumericsConfig(dt=3600.0, integrator="rk4", output_freq=24, uv_interp="nemo_cgrid"),
        mesh=MeshConfig(type="nemo", meshpath=str(d / "orca05_arctic_mesh.nc"), nemo_grid="cgrid",
                        rotation="fesom2"),
        data=DataConfig(ncinput=str(d / "orca05_arctic_grid_T.nc"), inicloud=str(sf),
                        output_path=str(out), time_averaging="instant"),
        time=TimeConfig(start_time_idx=0, n_steps=24 * 30, on_exhausted="cycle"))
    run_tracking(cfg, temporal_interp=True)           # sub-steps between the monthly records
    tr = load_trajectories(str(out))
    assert (tr["status"][:, -1] == int(ParticleStatus.ALIVE)).all()


def test_eorca025_arctic_tpivot_halo_free(tmp_path):
    mf, gt = _link_eorca025(tmp_path, "arctic")
    mesh = _mesh(mf, "fesom2")
    lay = mesh._nemo_cgrid_layout
    assert (lay.fold, lay.halo_cols, lay.halo_row, lay.periodic) == ("T", False, False, True)
    assert lay.raw_shape == (70, 1440) and lay.shape == (71, 1442) and mesh._structured_shape == (71, 1442)
    mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, var_u="uo", var_v="vo")
    ds = get_reader("nemo").open_dataset(str(gt), mode_cfg=mode)
    err, n_special = _flux_balance_and_special_edges(mesh, ds, mode)
    assert err < 1e-6 and n_special > 100            # seam + fold edges


def test_eorca025_southern_band_is_halo_free_periodic(tmp_path, capsys):
    mf, _gt = _link_eorca025(tmp_path, "southern")   # stale NFold flag on a crop below the fold
    mesh = _mesh(mf, "fesom2")
    lay = mesh._nemo_cgrid_layout
    assert (lay.fold, lay.halo_cols, lay.periodic, lay.shape) == (None, False, True, (28, 1442))
    assert "north fold the raster does not show" in capsys.readouterr().out


def test_orca05_equatorial_box_has_open_edges(capsys):
    mesh = _mesh(Path(_d) / "orca05_equatorial_mesh.nc")      # stale Iperio / NFold flags
    lay = mesh._nemo_cgrid_layout
    assert (lay.fold, lay.periodic) == (None, False)
    assert "E-W periodicity the raster does not show" in capsys.readouterr().out
    assert "N, E, W" in lay.label()


def test_orca2_cmip5_tpivot_with_halos_topology_only():
    """IPSL-CM5A-LR ORCA2 (T-pivot, halos, no mesh_mask): the layout from the T
    coordinates alone, and the u / v / thetao copies are the canonical ones
    (u, v with the fold sign)."""
    import xarray as xr
    from kelp.readers.nemo._layout import detect_nemo_cgrid_layout, nemo_cgrid_to_canonical
    d = _sib("esgf_orca2", "data")
    th = xr.open_dataset(d / "thetao_t0.nc")
    lay, notes = detect_nemo_cgrid_layout(th["lon"].values, th["lat"].values)
    assert (lay.fold, lay.halo_cols, lay.halo_row) == ("T", True, True) and not notes
    assert lay.shape == (149, 182) and lay.J == 147
    for name, kind, vec in (("thetao_t0.nc", "T", False), ("uo_t0.nc", "U", True), ("vo_t0.nc", "V", True)):
        ds = xr.open_dataset(d / name)
        a = ds[name.split("_")[0]].isel(time=0).values
        a = np.where(np.abs(a) > 1e10, np.nan, a)            # land fill
        nemo_cgrid_to_canonical(a, lay, kind, vector=vec, check=True, name=name)


# --- input robustness: domain_cfg without tmask, files per grid, explicit grid paths ----
def test_eorca025_domain_cfg_without_tmask_gives_the_same_mesh(tmp_path):
    import xarray as xr
    src = Path(_d) / "eorca025_equatorial_domain_cfg.nc"
    with xr.open_dataset(src) as ds:
        ds.drop_vars("tmask").to_netcdf(tmp_path / "domain_cfg.nc")
    a, b = _mesh(src), _mesh(tmp_path / "domain_cfg.nc")
    assert (a.n_nodes, a.n_elements) == (b.n_nodes, b.n_elements)
    assert (a._nemo_cgrid_nlev_raster == b._nemo_cgrid_nlev_raster).all()


def test_eorca025_files_split_in_time_join_along_time():
    d = Path(_d)
    sp = d / "eorca025_equatorial_split"
    rd = get_reader("nemo")
    mode = ModeConfig(Fdrift3D=True, var_u="uo", var_v="vo")
    ds = rd.open_dataset(str(sp / "*_grid_T.nc"), mode_cfg=mode)
    one = rd.open_dataset(str(d / "eorca025_equatorial_split" / "eorca025_1m_19760301_19760331_grid_T.nc"), mode_cfg=mode)
    assert ds.sizes["time"] == 3 and (ds["time"].values[2:] == one["time"].values).all()
    mesh = rd.load_mesh(MeshConfig(type="nemo", meshpath=str(d / "eorca025_equatorial_domain_cfg.nc"),
                                   nemo_grid="cgrid"))
    for t in (0, 2):
        u = rd.read_velocities(ds, t, mesh, mode_cfg=mode).u
        assert np.isfinite(u).all() and np.abs(u).max() > 0
    assert (rd.read_velocities(ds, 2, mesh, mode_cfg=mode).u == rd.read_velocities(one, 0, mesh, mode_cfg=mode).u).all()


def test_eorca025_crops_without_the_grid_token_via_explicit_paths():
    d = Path(_d)
    mode = ModeConfig(Fdrift3D=True, var_u="uo", var_v="vo")
    ds = get_reader("nemo").open_dataset(
        str(d / "eorca025_equatorial_T.nc"), u_path=str(d / "eorca025_equatorial_U.nc"),
        v_path=str(d / "eorca025_equatorial_V.nc"), mode_cfg=mode)
    assert {"u", "v"} <= set(ds.variables)


def _skip_if_files_missing(fn):
    """A data directory that is only partly filled skips, never errors."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except FileNotFoundError as err:
            pytest.skip(f"missing under KELP_TEST_NEMO_DATA: {err.filename or err}")
    return wrapper


for _n, _f in list(globals().items()):
    if _n.startswith("test_") and callable(_f):
        globals()[_n] = _skip_if_files_missing(_f)
