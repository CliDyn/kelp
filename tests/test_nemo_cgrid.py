"""Exact Arakawa C-grid u/v for NEMO (``mesh.nemo_grid: cgrid`` +
``numerics.uv_interp: cgrid``), synthetic grids only.

Grid frame: F(j,i) sits at (i, j) * d, T(j,i) at (i - 1/2, j - 1/2) * d, the
frame is rotated by ``theta`` (and optionally sheared) and put at the equator,
so the planar metric of the interpolator is accurate to ~1e-4.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from kelp.core.config import (
    AppConfig, BeachingConfig, DataConfig, MeshConfig, ModeConfig,
    NumericsConfig, TimeConfig)
from kelp.core.geometry import METERS_PER_DEG_LAT
from kelp.core.io import VelocityField
from kelp.numerics.interpolate import interp_cgrid_single
from kelp.readers import get_reader
from kelp.readers.structured import build_cgrid_mesh

D = 0.1          # cell size, degrees


def _grid(ny, nx, theta_deg=0.0, shear=0.0):
    """F- and T-point lon/lat of a rotated (optionally sheared) grid."""
    th = np.deg2rad(theta_deg)

    def pos(jj, ii):
        xg = (ii + shear * jj) * D
        yg = jj * D
        return (xg * np.cos(th) - yg * np.sin(th),
                xg * np.sin(th) + yg * np.cos(th))
    jj, ii = np.mgrid[0:ny, 0:nx].astype(float)
    return pos(jj, ii), pos(jj - 0.5, ii - 0.5)


def _edge_m(lon, lat, p, q):
    """Edge vector F[p] -> F[q] in the planar metric (m)."""
    cosl = np.cos(np.deg2rad(0.5 * (lat[p] + lat[q])))
    return ((lon[q] - lon[p]) * cosl * METERS_PER_DEG_LAT,
            (lat[q] - lat[p]) * METERS_PER_DEG_LAT)


def _faces(glamf, gphif, tmask, vx=None, vy=None, psi=None):
    """Face rasters (ny, nx) of a uniform geographic flow (vx, vy), or of the
    discretely non-divergent flow from a streamfunction ``psi`` at F-points
    (flux through a face = difference of psi at its ends)."""
    ny, nx = tmask.shape
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    for j in range(1, ny):
        for i in range(0, nx):
            ex, ey = _edge_m(glamf, gphif, (j - 1, i), (j, i))   # east face of T(j,i)
            L = np.hypot(ex, ey)
            if psi is not None:
                u[j, i] = -(psi[j, i] - psi[j - 1, i]) / L
            else:
                u[j, i] = (vx * ey - vy * ex) / L                # normal to +i
    for j in range(0, ny):
        for i in range(1, nx):
            ex, ey = _edge_m(glamf, gphif, (j, i - 1), (j, i))   # north face of T(j,i)
            L = np.hypot(ex, ey)
            if psi is not None:
                v[j, i] = (psi[j, i] - psi[j, i - 1]) / L
            else:
                v[j, i] = (-vx * ey + vy * ex) / L               # normal to +j
    wu = tmask.astype(bool).copy()
    wu[:, :-1] &= tmask[:, 1:].astype(bool)
    wu[:, -1] = False
    wv = tmask.astype(bool).copy()
    wv[:-1, :] &= tmask[1:, :].astype(bool)
    wv[-1, :] = False
    return np.where(wu, u, 0.0), np.where(wv, v, 0.0)


def _write(tmp_path, ny, nx, theta, tmask, u, v, n_time=3, dt=3600.0,
           shear=0.0, with_f=True):
    import pandas as pd
    import xarray as xr
    (flon, flat), (tlon, tlat) = _grid(ny, nx, theta, shear)
    t = pd.date_range("2020-01-01", periods=n_time, freq=f"{int(dt)}s")
    tc = {"time_counter": t}

    def w(name, dv):
        xr.Dataset(dv, coords=tc).to_netcdf(tmp_path / name)

    def rep(a, dim):
        return np.broadcast_to(a, (n_time, 1, ny, nx)).copy()
    w("EXP_grid_T.nc", {"ssh": (("time_counter", "y", "x"), np.zeros((n_time, ny, nx)))})
    w("EXP_grid_U.nc", {"uo": (("time_counter", "depthu", "y", "x"), rep(u, 0))})
    w("EXP_grid_V.nc", {"vo": (("time_counter", "depthv", "y", "x"), rep(v, 0))})
    mm = {"tmask": (("z", "y", "x"), tmask[None].astype(float)),
          "nav_lon": (("y", "x"), tlon), "nav_lat": (("y", "x"), tlat)}
    if with_f:
        mm.update(glamf=(("y", "x"), flon), gphif=(("y", "x"), flat))
    xr.Dataset(mm).to_netcdf(tmp_path / "mesh_mask.nc")
    return str(tmp_path / "EXP_grid_T.nc"), str(tmp_path / "mesh_mask.nc")


def _cfg(tmp_path, gt, mm, seeds, n_steps, dt=3600.0, uv_interp="cgrid",
         nemo_grid="cgrid", integrator="rk4", **kw):
    sf = tmp_path / "seeds.dat"
    sf.write_text("".join(f"{a:.9f} {b:.9f}\n" for a, b in seeds))
    out = tmp_path / "traj.nc"
    mode = kw.pop("mode", ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0))
    return AppConfig(
        mode=mode,
        numerics=NumericsConfig(dt=dt, integrator=integrator, output_freq=1,
                                uv_interp=uv_interp,
                                **kw.pop("numerics", {})),
        mesh=MeshConfig(type="nemo", meshpath=mm, nemo_grid=nemo_grid),
        data=DataConfig(ncinput=gt, inicloud=str(sf), output_path=str(out),
                        time_averaging="instant"),
        time=TimeConfig(start_time_idx=0, n_steps=n_steps),
        **kw), out


def _nearest(mesh, lon, lat):
    return int(np.argmin((mesh.nodes_lon - lon) ** 2 + (mesh.nodes_lat - lat) ** 2))


# (a) uniform flow is reproduced exactly (rotated, sheared, plain) ------------
@pytest.mark.parametrize("theta,shear", [(0.0, 0.0), (45.0, 0.0), (30.0, 0.4)])
def test_uniform_flow_exact(theta, shear):
    ny, nx = 8, 9
    (flon, flat), _ = _grid(ny, nx, theta, shear)
    tmask = np.ones((ny, nx))
    vx, vy = 0.3, -0.2
    u, v = _faces(flon, flat, np.ones((ny, nx)), vx=vx, vy=vy)
    mesh = build_cgrid_mesh(flon, flat, tmask)
    vel = VelocityField(u=u.ravel(), v=v.ravel(), eta=np.zeros(mesh.n_nodes),
                        uv_layout="cgrid")
    rng = np.random.default_rng(1)
    ji = mesh._cgrid_elem_ji       # the raster's last row/column have a dry far face
    inner = np.flatnonzero((ji[:, 0] < ny - 1) & (ji[:, 1] < nx - 1))
    for e in rng.choice(inner, 12):
        c = mesh._cgrid_corners[e]
        w = rng.dirichlet(np.ones(4) * 3)
        lon, lat = w @ mesh.nodes_lon[c], w @ mesh.nodes_lat[c]
        U, V, ok = interp_cgrid_single(lon, lat, 0, mesh, vel,
                                       _nearest(mesh, lon, lat))
        assert ok
        assert U == pytest.approx(vx, rel=2e-3) and V == pytest.approx(vy, rel=2e-3)


def test_axis_aligned_cell_returns_face_value():
    # rectangle, uW = uE = u0 -> exactly u0 (and linear u gives the linear profile)
    ny, nx = 5, 5
    (flon, flat), _ = _grid(ny, nx)
    mesh = build_cgrid_mesh(flon, flat, np.ones((ny, nx)))
    u = np.zeros((ny, nx)); v = np.zeros((ny, nx))
    u[:, :] = 0.7
    vel = VelocityField(u=u.ravel(), v=v.ravel(), eta=np.zeros(mesh.n_nodes),
                        uv_layout="cgrid")
    U, V, ok = interp_cgrid_single(2.4 * D, 2.6 * D, 0, mesh, vel,
                                   _nearest(mesh, 2.4 * D, 2.6 * D))
    assert ok and U == pytest.approx(0.7, rel=1e-12) and V == pytest.approx(0.0, abs=1e-12)
    u[:, 2] = 0.3                          # uE of T(.,2) smaller: u linear in xi
    vel = VelocityField(u=u.ravel(), v=v.ravel(), eta=np.zeros(mesh.n_nodes),
                        uv_layout="cgrid")
    # T(j,2) spans x in [1, 2] * D; xi = 0.25 -> u = 0.7 + 0.25 * (0.3 - 0.7)
    U, _, ok = interp_cgrid_single(1.25 * D, 2.6 * D, 0, mesh, vel,
                                   _nearest(mesh, 1.25 * D, 2.6 * D))
    assert ok and U == pytest.approx(0.6, rel=2e-3)


# (c) closed basin gyre: never leaves the wet cells ---------------------------
def test_closed_basin_gyre_stays_inside(tmp_path):
    from kelp.tracking import run_tracking
    from kelp.writers import load_trajectories
    from kelp.core.particles import ParticleStatus
    n, theta = 13, 30.0     # odd n-2: the psi peak sits on an F-point, no flat-top cell
    ny = nx = n
    tmask = np.zeros((ny, nx))
    tmask[2:n - 1, 2:n - 1] = 1.0              # dry rim of 2 cells
    (flon, flat), _ = _grid(ny, nx, theta)
    psi = np.zeros((ny, nx))
    ax = np.arange(1, n - 1)
    s = np.sin(np.pi * (ax - 1) / (n - 3))
    psi[1:n - 1, 1:n - 1] = 1.5e4 * np.outer(s, s)
    u, v = _faces(flon, flat, tmask, psi=psi)
    n_steps, dt = 600, 7200.0
    gt, mm = _write(tmp_path, ny, nx, theta, tmask, u, v, n_time=n_steps + 2, dt=dt)
    rng = np.random.default_rng(3)
    gx = rng.uniform(1.3, n - 2.3, 200) * D     # grid-frame seeds in wet region
    gy = rng.uniform(1.3, n - 2.3, 200) * D
    th = np.deg2rad(theta)
    seeds = list(zip(gx * np.cos(th) - gy * np.sin(th),
                     gx * np.sin(th) + gy * np.cos(th)))
    cfg, out = _cfg(tmp_path, gt, mm, seeds, n_steps, dt=dt)
    run_tracking(cfg)
    tr = load_trajectories(str(out))
    assert (tr["status"] == int(ParticleStatus.ALIVE)).all()           # nobody OUTSIDE / failed
    x = tr["lon"] * np.cos(th) + tr["lat"] * np.sin(th)      # back to grid frame
    y = -tr["lon"] * np.sin(th) + tr["lat"] * np.cos(th)
    assert x.min() > 1.0 * D and x.max() < (n - 2) * D
    assert y.min() > 1.0 * D and y.max() < (n - 2) * D
    # they really circulate (not parked): many revolutions of the gyre
    tot = np.abs(np.diff(tr["lon"], axis=1)).sum(axis=1)
    assert tot.min() > 20 * D


# (d) flow into a wall: approach, never cross ---------------------------------
def test_wall_approach_never_crosses(tmp_path):
    from kelp.tracking import run_tracking
    from kelp.writers import load_trajectories
    from kelp.core.particles import ParticleStatus
    ny, nx, theta = 8, 12, 20.0
    tmask = np.zeros((ny, nx)); tmask[2:ny - 1, 2:nx - 2] = 1.0
    (flon, flat), _ = _grid(ny, nx, theta)
    u, v = _faces(flon, flat, tmask, vx=0.0, vy=0.0)
    th = np.deg2rad(theta)
    u, v = _faces(flon, flat, tmask, vx=0.5 * np.cos(th), vy=0.5 * np.sin(th))
    n_steps = 150
    gt, mm = _write(tmp_path, ny, nx, theta, tmask, u, v, n_time=n_steps + 2)
    gx, gy = 2.3 * D, 4.4 * D
    seed = (gx * np.cos(th) - gy * np.sin(th), gx * np.sin(th) + gy * np.cos(th))
    cfg, out = _cfg(tmp_path, gt, mm, [seed], n_steps)
    run_tracking(cfg)
    tr = load_trajectories(str(out))
    assert (tr["status"] == int(ParticleStatus.ALIVE)).all()
    x = tr["lon"][0] * np.cos(th) + tr["lat"][0] * np.sin(th)
    wall = (nx - 3) * D                          # east face of the last wet column
    assert x.max() < wall and x.max() > wall - 0.05 * D


# (e) rotated grid: accepted by cgrid, still refused by the default ----------
def test_rotated_grid_accepted_only_in_cgrid(tmp_path):
    ny = nx = 6
    (flon, flat), _ = _grid(ny, nx, 45.0)
    tmask = np.ones((ny, nx))
    u, v = _faces(flon, flat, tmask, vx=0.1, vy=0.0)
    _gt, mm = _write(tmp_path, ny, nx, 45.0, tmask, u, v)
    reader = get_reader("nemo")
    mesh = reader.load_mesh(SimpleNamespace(meshpath=mm, nemo_grid="cgrid"))
    assert mesh._cgrid and mesh.n_elements == (ny - 1) * (nx - 1)
    with pytest.raises(NotImplementedError, match="east/north"):
        reader.load_mesh(SimpleNamespace(meshpath=mm))          # default tpoint


# (f) refusals ------------------------------------------------------------------
def _fail_case(name, tmp_path):
    ny = nx = 6
    (flon, flat), _ = _grid(ny, nx)
    tmask = np.ones((ny, nx))
    u, v = _faces(flon, flat, tmask, vx=0.1, vy=0.0)
    gt, mm = _write(tmp_path, ny, nx, 0.0, tmask, u, v, with_f=(name != "no_f"))
    seeds = [(2.5 * D, 2.5 * D)]
    kw, extra = {}, {}
    if name == "rk45":
        extra["integrator"] = "rk45"
    elif name == "beaching":
        extra["numerics"] = {"beaching": BeachingConfig(enabled=True)}
    elif name == "sample":
        kw["mode"] = ModeConfig(Fdrift2D=True, Fdrift3D=False, sample_vars=["thetao"])
    elif name == "pchip":
        extra["numerics"] = {"temporal_reconstruction": "pchip"}
    elif name == "bary_on_cgrid":
        extra["uv_interp"] = "barycentric"
    elif name == "cgrid_on_tpoint":
        extra["nemo_grid"] = "tpoint"
    return _cfg(tmp_path, gt, mm, seeds, 1, **extra, **kw)[0]


@pytest.mark.parametrize("name,match", [
    ("no_f", "glamf"), ("rk45", "cgrid|rk45"), ("beaching", "beaching"),
    ("sample", "sampl"), ("pchip", "temporal|cgrid"),
    ("bary_on_cgrid", "cgrid"), ("cgrid_on_tpoint", "cgrid")])
def test_refusals(tmp_path, name, match):
    from kelp.tracking import run_tracking
    with pytest.raises((ValueError, NotImplementedError, KeyError), match=match):
        run_tracking(_fail_case(name, tmp_path), temporal_interp=(name == "pchip"))


def test_default_nemo_grid_is_tpoint():
    assert MeshConfig(type="nemo").nemo_grid == "tpoint"
    with pytest.raises(ValueError, match="nemo_grid"):
        MeshConfig(type="nemo", nemo_grid="x")
    with pytest.raises(ValueError, match="nemo"):
        MeshConfig(type="fesom2", nemo_grid="cgrid")
