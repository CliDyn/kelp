"""NEMO structured-grid reader (readers/nemo.py, TODO 4.6).

Synthetic NEMO-like output (grid_T/U/V + mesh_mask) — no external data.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from kelp.readers import get_reader
from kelp.core.config import (
    AppConfig, ModeConfig, NumericsConfig, MeshConfig, DataConfig, TimeConfig)


def _make_nemo(tmp_path, ny=6, nx=7, n_time=4, u0=0.5, v0=0.0, land=(),
               lon_skew=0.0, lon_dir=1.0, lat_dir=1.0, u_ramp=0.0,
               u_time_shift_days=0.0):
    import xarray as xr
    import pandas as pd
    x = np.linspace(0.0, 6.0, nx)
    y = np.linspace(0.0, 5.0, ny)
    X, Y = np.meshgrid(x, y)                  # (ny, nx)
    # lon_skew=0, lon_dir=lat_dir=+1 -> geographic-aligned (v1): i=east, j=north.
    # lon_skew>0 tilts the j-axis off north (curvilinear); lon_dir=-1 / lat_dir=-1
    # REVERSE an axis (i->west / j->south) -> the alignment fail-fast must bite
    # (a sign-blind metric would fold the reversed axis onto aligned).
    lon2d = lon_dir * X + lon_skew * Y
    lat2d = lat_dir * Y
    tmask = np.ones((ny, nx), dtype=np.float64)
    for (j, i) in land:
        tmask[j, i] = 0.0
    t = pd.date_range("2020-01-01", periods=n_time, freq="3600s")
    tU = t + pd.Timedelta(days=u_time_shift_days)
    coords = {"time_counter": t}
    coordsU = {"time_counter": tU}

    def _w(name, dvars, coords_=coords):
        xr.Dataset(dvars, coords=coords_).to_netcdf(tmp_path / name)

    # uo varies in x when u_ramp != 0 (non-vacuous node-mapping check)
    uo = np.full((n_time, 1, ny, nx), u0)
    if u_ramp:
        uo = uo + u_ramp * np.arange(nx, dtype=np.float64)[None, None, None, :]
    _w("EXP_grid_T.nc", {
        "ssh": (("time_counter", "y", "x"), np.zeros((n_time, ny, nx))),
        "thetao": (("time_counter", "deptht", "y", "x"),
                   np.full((n_time, 1, ny, nx), 10.0)),
        "nav_lon": (("y", "x"), lon2d), "nav_lat": (("y", "x"), lat2d),
    })
    _w("EXP_grid_U.nc", {
        "uo": (("time_counter", "depthu", "y", "x"), uo),
        "nav_lon": (("y", "x"), lon2d), "nav_lat": (("y", "x"), lat2d),
    }, coords_=coordsU)
    _w("EXP_grid_V.nc", {
        "vo": (("time_counter", "depthv", "y", "x"), np.full((n_time, 1, ny, nx), v0)),
        "nav_lon": (("y", "x"), lon2d), "nav_lat": (("y", "x"), lat2d),
    })
    # mesh_mask: tmask (z, y, x) + coords
    xr.Dataset({
        "tmask": (("z", "y", "x"), tmask[None, :, :]),
        "nav_lon": (("y", "x"), lon2d), "nav_lat": (("y", "x"), lat2d),
    }).to_netcdf(tmp_path / "mesh_mask.nc")
    return str(tmp_path / "EXP_grid_T.nc"), str(tmp_path / "mesh_mask.nc")


def test_load_mesh_builds_quad_mesh(tmp_path):
    _gt, mm = _make_nemo(tmp_path, ny=6, nx=7, land=[(2, 3)])
    reader = get_reader("nemo")
    mesh = reader.load_mesh(SimpleNamespace(meshpath=mm))
    assert mesh.n_nodes == 6 * 7 - 1                 # one land cell dropped
    assert mesh.elem_nodes.shape[1] == 4             # quads
    assert mesh.elem_nodes.shape[0] == (5 * 6) - 4   # 4 quads touch the land cell
    assert mesh._structured_shape == (6, 7)


def test_open_dataset_merges_uvssh(tmp_path):
    gt, _mm = _make_nemo(tmp_path)
    reader = get_reader("nemo")
    mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, sample_vars=["thetao"])
    ds = reader.open_dataset(gt, mode_cfg=mode)
    assert {"u", "v", "ssh", "thetao"} <= set(ds.variables)
    assert "time" in ds.dims and ds.sizes["time"] == 4


def test_read_velocities_node_layout_and_values(tmp_path):
    gt, mm = _make_nemo(tmp_path, u0=0.5, v0=-0.25)
    reader = get_reader("nemo")
    mesh = reader.load_mesh(SimpleNamespace(meshpath=mm))
    mode = ModeConfig(Fdrift2D=True, Fdrift3D=False)
    ds = reader.open_dataset(gt, mode_cfg=mode)
    vel = reader.read_velocities(ds, 0, mesh, mode)
    assert vel.uv_layout == "node"
    assert vel.u.shape == (mesh.n_nodes,)
    # constant faces -> constant T-point velocity (interior + edge both = u0/v0)
    np.testing.assert_allclose(vel.u, 0.5)
    np.testing.assert_allclose(vel.v, -0.25)
    # float32 opt-in
    vel32 = reader.read_velocities(ds, 0, mesh, mode, velocity_float32=True)
    assert vel32.u.dtype == np.float32


def _cfg(tmp_path, gt, mm, uv_interp="barycentric", n_steps=3, lon0=3.0, lat0=2.5):
    seeds = tmp_path / "seeds.dat"
    seeds.write_text(f"{lon0} {lat0}\n")
    out = tmp_path / "traj.nc"
    return AppConfig(
        mode=ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0),
        numerics=NumericsConfig(dt=3600.0, integrator="rk2", output_freq=1,
                                uv_interp=uv_interp),
        mesh=MeshConfig(type="nemo", meshpath=mm),
        data=DataConfig(ncinput=gt, inicloud=str(seeds), output_path=str(out),
                        time_averaging="instant"),
        time=TimeConfig(start_time_idx=0, n_steps=n_steps)), out


def test_end_to_end_advects_by_analytic(tmp_path):
    from kelp.tracking import run_tracking
    from kelp.writers import load_trajectories
    from kelp.core.geometry import METERS_PER_DEG_LAT, DEG2RAD
    gt, mm = _make_nemo(tmp_path, ny=8, nx=9, u0=0.5, v0=0.0)
    cfg, out = _cfg(tmp_path, gt, mm, n_steps=3, lon0=3.0, lat0=2.5)
    run_tracking(cfg)
    tr = load_trajectories(str(out))
    z0 = np.cos(2.5 * DEG2RAD) * METERS_PER_DEG_LAT
    expected_dlon = 3 * 3600.0 * 0.5 / z0          # 3 steps, u=0.5 m/s
    assert tr["lon"][0, -1] == pytest.approx(3.0 + expected_dlon, rel=2e-2)
    assert tr["lat"][0, -1] == pytest.approx(2.5, abs=1e-6)


def test_node_layout_requires_barycentric(tmp_path):
    # leastsq is element-centre math -> node-layout NEMO velocities must fail fast.
    from kelp.tracking import run_tracking
    gt, mm = _make_nemo(tmp_path)
    cfg, _out = _cfg(tmp_path, gt, mm, uv_interp="leastsq")
    with pytest.raises(ValueError, match="barycentric|node"):
        run_tracking(cfg)


# --- v1 fail-fast guards (BLOCKER fixes) ---------------------------------

def test_skewed_grid_refused(tmp_path):
    # A curvilinear grid whose axes are tilted off E/N carries grid-relative
    # u/v that v1 cannot use as east/north -> NotImplementedError at load.
    _gt, mm = _make_nemo(tmp_path, lon_skew=0.1)     # ~5.7deg j-axis tilt
    reader = get_reader("nemo")
    with pytest.raises(NotImplementedError, match="vector rotation|east/north"):
        reader.load_mesh(SimpleNamespace(meshpath=mm))


def test_aligned_grid_accepted(tmp_path):
    # control: lon_skew=0 (E/N-aligned) must NOT trip the alignment guard.
    _gt, mm = _make_nemo(tmp_path, lon_skew=0.0)
    reader = get_reader("nemo")
    mesh = reader.load_mesh(SimpleNamespace(meshpath=mm))
    assert mesh.n_nodes > 0


def test_rotation_rejected(tmp_path):
    # NEMO has no vector-rotation path -> mesh.rotation must be refused.
    _gt, mm = _make_nemo(tmp_path)
    reader = get_reader("nemo")
    with pytest.raises(ValueError, match="rotation"):
        reader.load_mesh(SimpleNamespace(meshpath=mm, rotation="fesom2"))


def test_fdrift3d_refused_open_and_read(tmp_path):
    # v1 is 2-D only: both open_dataset and read_velocities must fail fast in 3D.
    gt, mm = _make_nemo(tmp_path)
    reader = get_reader("nemo")
    mode3d = ModeConfig(Fdrift2D=False, Fdrift3D=True)
    with pytest.raises(NotImplementedError, match="2-D only|Fdrift2D"):
        reader.open_dataset(gt, mode_cfg=mode3d)
    # read_velocities guards independently (covers direct callers / a 2-D ds
    # paired with a 3-D mode_cfg).
    mode2d = ModeConfig(Fdrift2D=True, Fdrift3D=False)
    ds = reader.open_dataset(gt, mode_cfg=mode2d)
    mesh = reader.load_mesh(SimpleNamespace(meshpath=mm))
    with pytest.raises(NotImplementedError, match="2-D only|Fdrift2D"):
        reader.read_velocities(ds, 0, mesh, mode3d)


def test_em_diffusion_refused(tmp_path):
    # spatial-Kh diffusion (em/m1) reads fields not yet mapped to T-points.
    from kelp.core.config import DiffusionConfig
    gt, mm = _make_nemo(tmp_path)
    reader = get_reader("nemo")
    mode = ModeConfig(Fdrift2D=True, Fdrift3D=False)
    ds = reader.open_dataset(gt, mode_cfg=mode)
    mesh = reader.load_mesh(SimpleNamespace(meshpath=mm))
    diff = DiffusionConfig(enabled=True, scheme="em")
    with pytest.raises(NotImplementedError, match="uniform|em/m1"):
        reader.read_velocities(ds, 0, mesh, mode, diffusion=diff)
    # a DISABLED em config must NOT raise (the guard gates on enabled)
    vel = reader.read_velocities(
        ds, 0, mesh, mode, diffusion=DiffusionConfig(enabled=False, scheme="em"))
    assert vel.u.shape == (mesh.n_nodes,)


@pytest.mark.parametrize("kw,name", [
    (dict(lat_dir=-1.0), "j-axis south"),     # latitude decreasing along j
    (dict(lon_dir=-1.0), "i-axis west"),      # longitude decreasing along i
])
def test_reversed_axis_refused(tmp_path, kw, name):
    # An axis that points the WRONG WAY (not just skewed) must be rejected — a
    # sign-blind alignment metric would fold west->east / south->north and read
    # the grid-relative u/v with a flipped sign.
    _gt, mm = _make_nemo(tmp_path, **kw)
    reader = get_reader("nemo")
    with pytest.raises(NotImplementedError, match="vector rotation|east/north"):
        reader.load_mesh(SimpleNamespace(meshpath=mm))


def _write_nemo_time_coords(tmp_path, *, t_coord, uv_coord, ny=5, nx=6, nt=3):
    """Write grid_T/U/V where the time DIMENSION is always present but the time
    COORDINATE is included only where requested — to exercise the all-or-none
    coordinate rule and the integer-time fallback."""
    import xarray as xr
    import pandas as pd
    x = np.linspace(0.0, 6.0, nx); y = np.linspace(0.0, 5.0, ny)
    X, Y = np.meshgrid(x, y)
    tvals = pd.date_range("2020-01-01", periods=nt, freq="3600s")

    def _vars(name, val):
        return {name: (("time_counter", "y", "x"), np.full((nt, ny, nx), val)),
                "nav_lon": (("y", "x"), X), "nav_lat": (("y", "x"), Y)}

    cT = {"time_counter": tvals} if t_coord else {}
    cUV = {"time_counter": tvals} if uv_coord else {}
    xr.Dataset(_vars("ssh", 0.0), coords=cT).to_netcdf(tmp_path / "E_grid_T.nc")
    xr.Dataset(_vars("uo", 0.5), coords=cUV).to_netcdf(tmp_path / "E_grid_U.nc")
    xr.Dataset(_vars("vo", 0.0), coords=cUV).to_netcdf(tmp_path / "E_grid_V.nc")
    return str(tmp_path / "E_grid_T.nc")


def test_uv_missing_time_coord_refused(tmp_path):
    # grid_T has a time coordinate, grid_U/grid_V have only a time dimension:
    # the pairing can't be verified -> refuse rather than graft grid_T's axis.
    gt = _write_nemo_time_coords(tmp_path, t_coord=True, uv_coord=False)
    reader = get_reader("nemo")
    mode = ModeConfig(Fdrift2D=True, Fdrift3D=False)
    with pytest.raises(ValueError, match="time coordinate"):
        reader.open_dataset(gt, mode_cfg=mode)


def test_no_time_coord_gets_integer_axis(tmp_path):
    # no file carries a time coordinate -> merged still gets an integer 'time'
    # coord so ds['time'] exists downstream (a plain arange reads as "no real
    # time", so such a run must set numerics.dt explicitly).
    from kelp.readers import extract_time_info
    gt = _write_nemo_time_coords(tmp_path, t_coord=False, uv_coord=False)
    reader = get_reader("nemo")
    mode = ModeConfig(Fdrift2D=True, Fdrift3D=False)
    ds = reader.open_dataset(gt, mode_cfg=mode)
    assert "time" in ds.coords and ds.sizes["time"] == 3
    assert extract_time_info(ds) == (None, None)


def test_mismatched_uv_time_axis_refused(tmp_path):
    # grid_U on a shifted time axis -> indexing u(t_i) would pair with ssh(t_j);
    # open_dataset must fail fast rather than silently mixing physical times.
    gt, _mm = _make_nemo(tmp_path, u_time_shift_days=1.0)
    reader = get_reader("nemo")
    mode = ModeConfig(Fdrift2D=True, Fdrift3D=False)
    with pytest.raises(ValueError, match="time axis"):
        reader.open_dataset(gt, mode_cfg=mode)


def test_missing_var_override_raises(tmp_path):
    # an explicit var_u override that does not exist must raise, not silently
    # fall back to an alias (wrong velocity product).
    gt, _mm = _make_nemo(tmp_path)
    reader = get_reader("nemo")
    mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, var_u="not_a_real_var")
    with pytest.raises(KeyError, match="not_a_real_var"):
        reader.open_dataset(gt, mode_cfg=mode)


def test_read_velocities_varying_field_node_mapping(tmp_path):
    # Non-vacuous node mapping: uo varies in x (with a land cell present) ->
    # each node's C->T-averaged value must land at the right grid location.
    gt, mm = _make_nemo(tmp_path, ny=6, nx=7, u0=0.0, u_ramp=0.5, land=[(2, 3)])
    reader = get_reader("nemo")
    mesh = reader.load_mesh(SimpleNamespace(meshpath=mm))
    mode = ModeConfig(Fdrift2D=True, Fdrift3D=False)
    ds = reader.open_dataset(gt, mode_cfg=mode)
    vel = reader.read_velocities(ds, 0, mesh, mode)
    assert np.ptp(vel.u) > 0.1                          # genuinely varying
    node_id = mesh._structured_node_id
    # uo(j,i) = 0.5*i  ->  C->T east+west average  u_T(j,i) = 0.25*(2i-1), i>=1
    for (j, i) in [(0, 1), (3, 5), (5, 6)]:
        nd = int(node_id[j, i])
        assert nd >= 0
        assert vel.u[nd] == pytest.approx(0.25 * (2 * i - 1))


def test_barycentric_reproduces_linear_field(tmp_path):
    # Non-vacuous: a spatially-VARYING (linear) node field must be reproduced
    # at an element centroid (the constant-field e2e test could pass even if
    # interpolation were broken; this cannot).
    from kelp.numerics.interpolate import interp_barycentric_single
    from kelp.core.io import VelocityField
    _gt, mm = _make_nemo(tmp_path, ny=6, nx=7)
    mesh = get_reader("nemo").load_mesh(SimpleNamespace(meshpath=mm))
    # u = a + b*lon (varies in space), v = c + d*lat
    a, b, c, d = 0.3, 0.25, -0.1, 0.4
    u_nodes = a + b * mesh.nodes_lon
    v_nodes = c + d * mesh.nodes_lat
    vf = VelocityField(u=u_nodes, v=v_nodes, eta=np.zeros(mesh.n_nodes),
                       w=None, uv_layout="node")
    # sample at the centroid of an interior element (its 4 corner u-values differ)
    e = mesh.elem_nodes.shape[0] // 2
    qlon = float(mesh.elem_center_lon[e]); qlat = float(mesh.elem_center_lat[e])
    corner_u = u_nodes[mesh.elem_nodes[e]]
    assert np.ptp(corner_u) > 1e-3                     # genuinely varying
    u, v, ok = interp_barycentric_single(
        qlon, qlat, 0, mesh, vf, int(mesh.elem_nodes[e, 0]))
    assert ok
    assert u == pytest.approx(a + b * qlon, rel=1e-4)
    assert v == pytest.approx(c + d * qlat, rel=1e-4)


def test_particle_over_land_goes_outside(tmp_path):
    # A particle seeded in a land hole (no quad contains it) must die OUTSIDE,
    # not silently interpolate garbage from surrounding wet nodes.
    from kelp.tracking import run_tracking
    from kelp.writers import load_trajectories
    from kelp.core.particles import ParticleStatus
    # 2x2 land block in the interior -> a hole in the quad mesh
    gt, mm = _make_nemo(tmp_path, ny=8, nx=9,
                        land=[(3, 4), (3, 5), (4, 4), (4, 5)])
    x = np.linspace(0.0, 6.0, 9); y = np.linspace(0.0, 5.0, 8)
    hole_lon = 0.5 * (x[4] + x[5]); hole_lat = 0.5 * (y[3] + y[4])
    cfg, out = _cfg(tmp_path, gt, mm, n_steps=2, lon0=hole_lon, lat0=hole_lat)
    run_tracking(cfg)
    tr = load_trajectories(str(out))
    assert tr["status"][0, -1] == int(ParticleStatus.OUTSIDE)


def test_temporal_interp_runs(tmp_path):
    # temporal_interp blends adjacent snapshots; with constant-in-time fields
    # the analytic displacement is unchanged, but the lazy-lerp payload path
    # (node layout + barycentric) must work end-to-end.
    from kelp.tracking import run_tracking
    from kelp.writers import load_trajectories
    from kelp.core.geometry import METERS_PER_DEG_LAT, DEG2RAD
    gt, mm = _make_nemo(tmp_path, ny=8, nx=9, u0=0.5, v0=0.0)
    cfg, out = _cfg(tmp_path, gt, mm, n_steps=3, lon0=3.0, lat0=2.5)
    run_tracking(cfg, temporal_interp=True)
    tr = load_trajectories(str(out))
    z0 = np.cos(2.5 * DEG2RAD) * METERS_PER_DEG_LAT
    expected_dlon = 3 * 3600.0 * 0.5 / z0
    assert tr["lon"][0, -1] == pytest.approx(3.0 + expected_dlon, rel=2e-2)


def test_sigma_layer_below_the_surface_refused(tmp_path):
    """The v1 reader always takes the surface level (index 0 of every depth
    dimension); a requested deeper layer was ignored without a word
    (audit 2026-10-04)."""
    gt, mm = _make_nemo(tmp_path)
    reader = get_reader("nemo")
    with pytest.raises(NotImplementedError, match="sigma_layer"):
        reader.open_dataset(gt, mode_cfg=ModeConfig(sigma_layer=2))
    mesh = reader.load_mesh(SimpleNamespace(meshpath=mm))
    ds = reader.open_dataset(gt, mode_cfg=ModeConfig(sigma_layer=0))
    try:
        with pytest.raises(NotImplementedError, match="sigma_layer"):
            reader.read_velocities(ds, 0, mesh, ModeConfig(sigma_layer=1))
        reader.read_velocities(ds, 0, mesh, ModeConfig(sigma_layer=0))
    finally:
        ds.close()


def test_sigma_mean_and_scalar_reads_are_guarded_too(tmp_path):
    gt, mm = _make_nemo(tmp_path)
    reader = get_reader("nemo")
    with pytest.raises(NotImplementedError, match="sigma_mean"):
        reader.open_dataset(gt, mode_cfg=ModeConfig(sigma_mean=True))
    mesh = reader.load_mesh(SimpleNamespace(meshpath=mm))
    ds = reader.open_dataset(gt, mode_cfg=ModeConfig())
    try:
        with pytest.raises(NotImplementedError, match="sigma_layer"):
            reader.read_scalar_fields(ds, 0, mesh, ["thetao"],
                                      ModeConfig(sigma_layer=2))
    finally:
        ds.close()
