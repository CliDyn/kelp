"""Exact Arakawa C-grid u/v for NEMO (``mesh.nemo_grid: cgrid`` +
``numerics.uv_interp: cgrid``), synthetic grids only.

Grid frame: F(j,i) sits at (i, j) * d, T(j,i) at (i - 1/2, j - 1/2) * d, the
frame is rotated by ``theta`` (and optionally sheared) and put at the equator,
so the planar metric of the interpolator is accurate to ~1e-4.
"""
import itertools
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
from tests._cgrid_orca import NC, NY, orca_grid, strip

D = 0.1          # cell size, degrees


def _grid(ny, nx, theta_deg=0.0, shear=0.0, mirror=False, lat0=0.0):
    """F- and T-point lon/lat of a rotated (optionally sheared / mirrored,
    i.e. i pointing west = left-handed) grid whose origin sits at ``lat0``."""
    th = np.deg2rad(theta_deg)

    def pos(jj, ii):
        xg = (ii + shear * jj) * D * (-1.0 if mirror else 1.0)
        yg = jj * D
        return (xg * np.cos(th) - yg * np.sin(th),
                xg * np.sin(th) + yg * np.cos(th) + lat0)
    jj, ii = np.mgrid[0:ny, 0:nx].astype(float)
    return pos(jj, ii), pos(jj - 0.5, ii - 0.5)


def _edge_m(lon, lat, p, q):
    """Edge vector F[p] -> F[q] in the planar metric (m)."""
    cosl = np.cos(np.deg2rad(0.5 * (lat[p] + lat[q])))
    return ((lon[q] - lon[p]) * cosl * METERS_PER_DEG_LAT,
            (lat[q] - lat[p]) * METERS_PER_DEG_LAT)


def _widths(glamf, gphif):
    """Model-style face widths (m): e2u of the east face, e1v of the north face."""
    ny, nx = glamf.shape
    e2u = np.ones((ny, nx)); e1v = np.ones((ny, nx))
    for j in range(ny):
        for i in range(nx):
            if j > 0:
                e2u[j, i] = np.hypot(*_edge_m(glamf, gphif, (j - 1, i), (j, i)))
            if i > 0:
                e1v[j, i] = np.hypot(*_edge_m(glamf, gphif, (j, i - 1), (j, i)))
    return e2u, e1v


def _faces(glamf, gphif, tmask, vx=None, vy=None, psi=None):
    """Face rasters (ny, nx) of a uniform geographic flow (vx, vy), or of the
    discretely non-divergent flow from a streamfunction ``psi`` at F-points
    (flux through a face = difference of psi at its ends)."""
    ny, nx = tmask.shape
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    ei = _edge_m(glamf, gphif, (0, 0), (0, 1))
    ej = _edge_m(glamf, gphif, (0, 0), (1, 0))
    hand = np.sign(ei[0] * ej[1] - ei[1] * ej[0])     # -1: left-handed grid
    for j in range(1, ny):
        for i in range(0, nx):
            ex, ey = _edge_m(glamf, gphif, (j - 1, i), (j, i))   # east face of T(j,i)
            L = np.hypot(ex, ey)
            if psi is not None:
                u[j, i] = -(psi[j, i] - psi[j - 1, i]) / L
            else:
                u[j, i] = hand * (vx * ey - vy * ex) / L        # normal to +i
    for j in range(0, ny):
        for i in range(1, nx):
            ex, ey = _edge_m(glamf, gphif, (j, i - 1), (j, i))   # north face of T(j,i)
            L = np.hypot(ex, ey)
            if psi is not None:
                v[j, i] = (psi[j, i] - psi[j, i - 1]) / L
            else:
                v[j, i] = hand * (-vx * ey + vy * ex) / L       # normal to +j
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
        e2u, e1v = _widths(flon, flat)
        mm.update(e2u=(("y", "x"), e2u), e1v=(("y", "x"), e1v))
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
@pytest.mark.parametrize("theta,shear,mirror", [(0.0, 0.0, False), (45.0, 0.0, False),
                                                (30.0, 0.4, False), (30.0, 0.0, True)])
def test_uniform_flow_exact(theta, shear, mirror):
    ny, nx = 8, 9
    (flon, flat), _ = _grid(ny, nx, theta, shear, mirror)
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
        assert U == pytest.approx(vx, rel=1e-5) and V == pytest.approx(vy, rel=1e-5)


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


# (d2) open edges: the exit through a raster edge is OUTSIDE ------------------
def _exits(tr):
    from kelp.core.particles import FailureReason, ParticleStatus
    assert (tr["status"][:, -1] == int(ParticleStatus.OUTSIDE)).all()
    # a stage leaving the mesh and the end point leaving it differ only in the label
    assert np.isin(tr["failure_reason"][:, -1],
                   [int(FailureReason.OUT_OF_BOUNDS), int(FailureReason.INTERPOLATION_FAILED)]).all()


@pytest.mark.parametrize("vx,vy", [(1, 0), (-1, 0), (0, 1), (0, -1)], ids=["E", "W", "N", "S"])
def test_box_edges_are_open(tmp_path, vx, vy):
    """A raster edge is open where water reaches it: the face flux of the edge is
    read from the raster and the particle leaves (a stage or the end point
    outside the mesh).  No flag, no config key."""
    from kelp.tracking import run_tracking
    from kelp.writers import load_trajectories
    ny = nx = 10
    tmask = np.ones((ny, nx))
    (flon, flat), _ = _grid(ny, nx)
    u, v = _faces(flon, flat, np.ones((ny, nx)), vx=float(vx), vy=float(vy))
    u[:, -1], v[-1, :] = u[:, -2], v[-2, :]               # water reaches the east / north edge
    n_steps, dt = 60, 3600.0
    gt, mm = _write(tmp_path, ny, nx, 0.0, tmask, u, v, n_time=n_steps + 2, dt=dt)
    cfg, out = _cfg(tmp_path, gt, mm, [(5.3 * D, 5.3 * D)], n_steps, dt=dt,
                    mode=ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0,
                                    track_failure_reason=True))
    run_tracking(cfg)
    _exits(load_trajectories(str(out)))


def test_fold_slab_south_edge_is_open(tmp_path):
    """An arctic-style slab (the fold in the raster, the south edge cut): a
    particle crosses the fold southwards, runs on and leaves at the open edge."""
    from kelp.tracking import run_tracking
    from kelp.writers import load_trajectories
    cfg, out, _u, _v = _orca_case(tmp_path, "T", False, vy=-1.5, n_steps=300,
                                  seeds=[(0.3, 0.45)])
    cfg.mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0, track_failure_reason=True)
    run_tracking(cfg, temporal_interp=False)
    tr = load_trajectories(str(out))
    _exits(tr)
    assert tr["lat"][0].min() < -1.0 and np.abs(tr["lon"][0] - 0.3).max() < 5e-3   # crossed y = 0 first


def test_duplicate_nodes_and_wet_pivots_refuse():
    g, flon, flat, tmask, *_ = _orca_fields("F", True)
    with pytest.raises(ValueError, match="pairs of mesh nodes coincide"):
        build_cgrid_mesh(flon, flat, tmask)               # the fold raster taken for a regional one
    from kelp.readers.structured import fold_pivot_cells
    wet = tmask.copy()
    wet[fold_pivot_cells(g.canon)] = 1.0
    with pytest.raises(ValueError, match="fold's grid poles must be on land"):
        build_cgrid_mesh(flon, flat, wet, layout=g.canon)


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


def test_nemo_360d_calendar_alias_decodes(tmp_path):
    """XIOS writes calendar "360d"; cftime only knows "360_day"."""
    import netCDF4
    ny = nx = 5
    (flon, flat), _ = _grid(ny, nx)
    tmask = np.ones((ny, nx))
    u, v = _faces(flon, flat, tmask, vx=0.1, vy=0.0)
    gt, _mm = _write(tmp_path, ny, nx, 0.0, tmask, u, v)
    for f in tmp_path.glob("EXP_grid_*.nc"):
        with netCDF4.Dataset(f, "r+") as nc:
            tv = nc["time_counter"]
            tv.units, tv.calendar = "seconds since 0001-01-01 00:00:00", "360d"
            tv[:] = np.arange(3) * 3600.0
    ds = get_reader("nemo").open_dataset(gt, mode_cfg=ModeConfig(Fdrift2D=True, Fdrift3D=False))
    assert ds["time"].dt.calendar == "360_day"


def _loop_flux(mesh, vel, e, n=10, f=1e-3):
    """Outward flux of the interpolated field around a loop just inside
    element ``e`` in the sphere metric (``U dlat - V cos(lat) dlon``, 5-point
    Gauss on each segment), and the sum of |flux| through the loop."""
    c = mesh._cgrid_corners[e]
    lon, lat = mesh.nodes_lon[c], mesh.nodes_lat[c]
    ctr = np.array([lon.mean(), lat.mean()])
    pts = []
    for a in range(4):
        b = (a + 1) % 4
        for t in np.arange(n) / n:
            pts.append((1 - t) * np.array([lon[a], lat[a]]) + t * np.array([lon[b], lat[b]]))
    pts = ctr + (1 - f) * (np.array(pts) - ctr)
    pts = np.vstack([pts, pts[:1]])
    xs, ws = np.polynomial.legendre.leggauss(5)
    flux = absflux = 0.0
    for p, q in zip(pts[:-1], pts[1:]):
        for x, w in zip(xs, ws):
            m = p + (q - p) * (x + 1) / 2
            U, V, ok = interp_cgrid_single(m[0], m[1], 0, mesh, vel, _nearest(mesh, *m))
            assert ok
            d = w / 2 * (U * (q[1] - p[1]) - V * np.cos(np.deg2rad(m[1])) * (q[0] - p[0]))
            flux += d
            absflux += abs(d)
    return flux, absflux


@pytest.mark.parametrize("lat0", [0.0, 60.0])
def test_model_widths_carry_discrete_continuity(lat0):
    """Fluxes psi-differences / model widths balance in every cell; with the
    model widths the interpolated field is divergence-free whatever the widths
    are (exact on the sphere), without them (plane edge lengths) it is not."""
    ny = nx = 8
    (flon, flat), _ = _grid(ny, nx, 20.0, lat0=lat0)
    rng = np.random.default_rng(5)
    psi = rng.normal(size=(ny, nx)) * 1e4
    e2u, e1v = _widths(flon, flat)
    e2u = e2u * (1.0 + 0.3 * rng.random((ny, nx)))      # widths != plane lengths
    e1v = e1v * (1.0 + 0.3 * rng.random((ny, nx)))
    u = np.zeros((ny, nx)); v = np.zeros((ny, nx))
    u[1:, :] = -(psi[1:, :] - psi[:-1, :]) / e2u[1:, :]
    v[:, 1:] = (psi[:, 1:] - psi[:, :-1]) / e1v[:, 1:]
    res = {}
    for widths in (True, False):
        mesh = build_cgrid_mesh(flon, flat, np.ones((ny, nx)),
                                **(dict(e2u=e2u, e1v=e1v) if widths else {}))
        vel = VelocityField(u=u.ravel(), v=v.ravel(), eta=np.zeros(mesh.n_nodes),
                            uv_layout="cgrid")
        ji = mesh._cgrid_elem_ji
        e = int(np.flatnonzero((ji[:, 0] == 4) & (ji[:, 1] == 4))[0])
        flux, absflux = _loop_flux(mesh, vel, e)
        res[widths] = abs(flux) / absflux
    assert res[True] < 1e-10 and res[False] > 1e-4


# --- global ORCA layouts: E-W seam + T / F-pivot north fold, with / without the halo
# (``tests/_cgrid_orca.py``: a tripolar-like elliptic grid; halo-free files carry
# the NEMO 4.2 topology attributes).  The flow psi = vy * x (northward) has face
# fluxes that are psi differences, canonicalised as vectors, so every copy of u
# and v is what NEMO writes (v antisymmetric on the fold row, a sign across it).
ORCA_LAYOUTS = [("F", True), ("T", True), ("F", False), ("T", False)]
ORCA_IDS = ["F-halo", "T-halo", "F-free", "T-free"]


def _orca_fields(fold, halo, vy=0.5):
    """Canonical-form rasters: F coordinates, tmask, widths, u, v of psi = vy * x."""
    from kelp.readers.structured import cgrid_to_canonical, fold_pivot_cells
    g = orca_grid(fold, halo)
    lay = g.canon
    flon, flat = g.xy_c["F"]
    e2u, e1v = _widths(flon, flat)
    e2u, e1v = cgrid_to_canonical(e2u, lay, "U"), cgrid_to_canonical(e1v, lay, "V")
    psi = vy * flon * METERS_PER_DEG_LAT
    u = np.zeros(flon.shape); v = np.zeros(flon.shape)
    u[1:, :] = -(psi[1:, :] - psi[:-1, :]) / e2u[1:, :]
    v[:, 1:] = (psi[:, 1:] - psi[:, :-1]) / e1v[:, 1:]
    u, v = (cgrid_to_canonical(a, lay, k, vector=True) for a, k in ((u, "U"), (v, "V")))
    tmask = np.ones(flon.shape)
    tmask[fold_pivot_cells(lay)] = 0.0                  # the fold's grid poles are land
    return g, flon, flat, cgrid_to_canonical(tmask, lay, "T"), e2u, e1v, u, v


def _orca_case(tmp_path, fold="F", halo=True, vy=0.5, n_steps=120, dt=1800.0, seeds=(), rotation=None):
    """Mesh file + u/v files of the layout; returns ``cfg, out, u, v`` (canonical)."""
    import pandas as pd
    import xarray as xr
    g, flon, flat, tmask, e2u, e1v, u, v = _orca_fields(fold, halo, vy)
    e2u[0] = np.nan                                     # unused faces may be junk
    raw = {k: strip(a, halo) for k, a in dict(
        e2u=e2u, e1v=e1v, tmask=tmask, glamf=flon, gphif=flat, u=u, v=v).items()}
    t = pd.date_range("2020-01-01", periods=1, freq="h")
    tc = {"time_counter": t}
    ny, nx = g.raw.raw_shape
    xr.Dataset({"ssh": (("time_counter", "y", "x"), np.zeros((1, ny, nx)))},
               coords=tc).to_netcdf(tmp_path / "EXP_grid_T.nc")
    xr.Dataset({"uo": (("time_counter", "depthu", "y", "x"), raw["u"][None, None])},
               coords=tc).to_netcdf(tmp_path / "EXP_grid_U.nc")
    xr.Dataset({"vo": (("time_counter", "depthv", "y", "x"), raw["v"][None, None])},
               coords=tc).to_netcdf(tmp_path / "EXP_grid_V.nc")
    mesh_mask = xr.Dataset(
        {"tmask": (("z", "y", "x"), raw["tmask"][None]),
         "nav_lon": (("y", "x"), g.xy["T"][0]), "nav_lat": (("y", "x"), g.xy["T"][1]),
         "glamf": (("y", "x"), raw["glamf"]), "gphif": (("y", "x"), raw["gphif"]),
         "e2u": (("y", "x"), raw["e2u"]), "e1v": (("y", "x"), raw["e1v"])})
    if not halo:                                        # NEMO >= 4.2 writes the topology flags
        mesh_mask.attrs.update(Iperio=1, Jperio=0, NFold=1, NFtype=fold)
    mesh_mask.to_netcdf(tmp_path / "mesh_mask.nc")
    gt, mm = str(tmp_path / "EXP_grid_T.nc"), str(tmp_path / "mesh_mask.nc")
    cfg, out = _cfg(tmp_path, gt, mm, list(seeds), n_steps, dt=dt)
    cfg.time.on_exhausted = "cycle"                     # one record, held
    cfg.mesh.rotation = rotation
    return cfg, out, u, v


@pytest.mark.parametrize("fold,halo", ORCA_LAYOUTS, ids=ORCA_IDS)
def test_orca_mesh_topology_and_flux_continuity(tmp_path, fold, halo, capsys):
    from kelp.core.mesh import build_elem_neighbors
    from kelp.readers.structured import cgrid_canonical
    cfg, _out, u, v = _orca_case(tmp_path, fold, halo)
    mesh = get_reader("nemo").load_mesh(cfg.mesh)
    lay = mesh._cgrid_layout
    assert (lay.fold, lay.halo_cols, lay.periodic) == (fold, halo, True)
    assert f"NEMO cgrid layout: {lay.label()}" in capsys.readouterr().out      # the log line
    ny, nx = lay.shape
    assert mesh._structured_shape == (NY, NC + 2) and lay.J == NY - 2
    ji = mesh._cgrid_elem_ji
    nb = build_elem_neighbors(mesh)                       # raises if an edge has > 2 cells
    cell = {(int(a), int(b)): e for e, (a, b) in enumerate(ji)}
    for j in range(2, lay.J):                             # seam: W of i=1 is i=NC
        assert nb[cell[(j, 1)], 3] == cell[(j, NC)]
    n_fold = 0
    for (j, i), e in cell.items():                        # fold: N of a fold-row cell is its mirror
        if j == lay.J:
            mj, mi, _ = cgrid_canonical("T", j + 1, i, lay)
            if (int(mj), int(mi)) in cell:
                assert nb[e, 2] == cell[(int(mj), int(mi))]
                n_fold += 1
    assert n_fold > 5
    # model normal flux balances on every shared edge (slots S, E, N, W)
    j, i = ji[:, 0], ji[:, 1]
    fi = np.stack([(j - 1) * nx + i, j * nx + i, j * nx + i, j * nx + i - 1], 1)
    uf, vf = u.ravel(), v.ravel()
    e2u, e1v = mesh._cgrid_e2u, mesh._cgrid_e1v
    out = np.stack([-vf[fi[:, 0]] * e1v[fi[:, 0]], uf[fi[:, 1]] * e2u[fi[:, 1]],
                    vf[fi[:, 2]] * e1v[fi[:, 2]], -uf[fi[:, 3]] * e2u[fi[:, 3]]], 1)
    e, k = np.nonzero(nb >= 0)
    n = nb[e, k]
    k2 = np.array([int(np.flatnonzero(nb[b] == a)[0]) for a, b in zip(e, n)])
    assert np.abs(out[e, k] + out[n, k2]).max() <= 1e-9 * np.abs(out).max()
    # ...and so does the interpolated normal velocity at seam / fold edge midpoints
    vel = VelocityField(u=uf, v=vf, eta=np.zeros(mesh.n_nodes), uv_layout="cgrid")
    c, lo, la = mesh._cgrid_corners, mesh.nodes_lon, mesh.nodes_lat
    special = (np.abs(i[e] - i[n]) > 1) | (np.abs(j[e] - j[n]) > 1)
    assert special.sum() > 10
    for a, s, b, t in zip(e[special], k[special], n[special], k2[special]):
        p, q = c[a, s], c[a, (s + 1) % 4]
        mid = (0.5 * (lo[p] + lo[q]), 0.5 * (la[p] + la[q]))
        nrm = np.array([la[q] - la[p], -(lo[q] - lo[p])])
        va = np.array(interp_cgrid_single(*mid, 0, mesh, vel, int(p), elem=int(a))[:2])
        vb = np.array(interp_cgrid_single(*mid, 0, mesh, vel, int(p), elem=int(b))[:2])
        assert abs((va - vb) @ nrm) <= 1e-9 * np.hypot(*nrm)


def test_orca_particles_cross_seam_and_fold_rotation_invariant(tmp_path):
    """Northward flow: two particles cross the fold segment (|x| < A), one the
    seam (x > A); all stay ALIVE on their streamline x = const, in the
    geographic and in the rotated frame alike, and halo-free input of a pivot
    gives the halo input's trajectories bit for bit."""
    from kelp.tracking import run_tracking
    from kelp.writers import load_trajectories
    from kelp.core.particles import ParticleStatus
    seeds = [(0.3, -0.45), (1.4, -0.45), (-0.5, -0.45)]
    tracks = {}
    for (fold, halo), rotation in itertools.product(ORCA_LAYOUTS, (None, "fesom2")):
        d = tmp_path / f"{fold}{halo}{rotation}"
        d.mkdir()
        cfg, out, _u, _v = _orca_case(d, fold, halo, seeds=seeds, rotation=rotation)
        run_tracking(cfg, temporal_interp=False)
        tr = load_trajectories(str(out))
        assert (tr["status"] == int(ParticleStatus.ALIVE)).all()
        assert (tr["lat"][:, -1] > 0.4).all()                 # crossed y = 0
        assert np.abs(tr["lon"] - np.array(seeds)[:, :1]).max() < 2e-3
        tracks[fold, halo, rotation] = np.stack([tr["lon"], tr["lat"]])
        import xarray as xr
        with xr.open_dataset(out) as ds:                      # the layout, open edges included
            assert f"{fold}-pivot north fold" in ds.attrs["cgrid_layout"]
            assert "unjoined raster edges (open where wet): S" in ds.attrs["cgrid_layout"]
    for fold in "TF":
        np.testing.assert_array_equal(tracks[fold, False, None], tracks[fold, True, None])
        np.testing.assert_allclose(tracks[fold, True, "fesom2"], tracks[fold, True, None], atol=1e-4)


def test_orca_mismatched_copies_refuse(tmp_path):
    """A stale halo column in the u of any record refuses (it would be repaired
    silently otherwise); stale e2u copies refuse at the mesh load."""
    import xarray as xr
    cfg, *_ = _orca_case(tmp_path, "F", True)
    rd = get_reader("nemo")
    mesh = rd.load_mesh(cfg.mesh)
    ds = rd.open_dataset(cfg.data.ncinput, mode_cfg=cfg.mode)
    rd.read_velocities(ds, 0, mesh, mode_cfg=cfg.mode)    # consistent: fine
    ds.close()
    f = tmp_path / "EXP_grid_U.nc"
    with xr.open_dataset(f) as d:
        d = d.load()
    d["uo"][0, 0, 3, 0] += 1e-3                           # the E-W halo column of one face
    d.to_netcdf(tmp_path / "bad_grid_U.nc")
    (tmp_path / "EXP_grid_U.nc").rename(tmp_path / "good_grid_U.nc")
    (tmp_path / "bad_grid_U.nc").rename(f)
    ds = rd.open_dataset(cfg.data.ncinput, mode_cfg=cfg.mode)
    with pytest.raises(ValueError, match="u: 1 halo / fold copies differ"):
        rd.read_velocities(ds, 0, mesh, mode_cfg=cfg.mode)
    ds.close()
    mm = tmp_path / "mesh_mask.nc"
    with xr.open_dataset(mm) as d:
        d = d.load()
    d["e2u"][5, 0] *= 1.001
    d.to_netcdf(tmp_path / "bad_mesh.nc")
    cfg.mesh.meshpath = str(tmp_path / "bad_mesh.nc")
    with pytest.raises(ValueError, match="e2u: 1 halo / fold copies differ"):
        rd.load_mesh(cfg.mesh)


# --- dry ring (one ring of zero-velocity cells around the wet region) --------
def _ring_ji(mesh):
    return {tuple(x) for x in mesh._cgrid_elem_ji[mesh._cgrid_n_wet:].tolist()}


def test_ring_cells_8_neighbourhood():
    ny = nx = 10
    (flon, flat), _ = _grid(ny, nx)
    tmask = np.zeros((ny, nx))
    tmask[4, 4:6] = 1.0
    tmask[7, 7] = 1.0
    m0 = build_cgrid_mesh(flon, flat, tmask, ring=False)
    m1 = build_cgrid_mesh(flon, flat, tmask)
    assert m0._cgrid_n_wet == m1._cgrid_n_wet == 3 and m0.n_elements == 3
    np.testing.assert_array_equal(m1._cgrid_elem_ji[:3], m0._cgrid_elem_ji)   # wet ids unchanged
    np.testing.assert_array_equal(m1._cgrid_nlev, [1, 1, 1] + [0] * (m1.n_elements - 3))
    expect = set()
    for j, i in zip(*np.nonzero(tmask)):
        expect |= {(j + a, i + b) for a in (-1, 0, 1) for b in (-1, 0, 1)}
    expect -= {(j, i) for j, i in zip(*np.nonzero(tmask))}
    assert _ring_ji(m1) == expect and len(_ring_ji(m1)) == m1.n_elements - 3


@pytest.mark.parametrize("fold", "TF")
def test_ring_across_seam_and_fold(fold):
    from kelp.core.mesh import build_elem_neighbors
    from kelp.readers.structured import cgrid_to_canonical, fold_pivot_cells
    g, flon, flat, tmask, _e2u, _e1v, _u, _v = _orca_fields(fold, True)
    J = g.canon.J
    tmask[4, NC] = 0.0                  # dry cell at the seam (col 0 is its halo copy)
    tmask[0, NC] = 0.0
    tmask[J, 4] = 0.0                   # dry cell on the fold row (mirrored in the halo row)
    tmask = cgrid_to_canonical(tmask, g.canon, "T")
    mesh = build_cgrid_mesh(flon, flat, tmask, layout=g.canon)
    ring = [tuple(x) for x in mesh._cgrid_elem_ji[mesh._cgrid_n_wet:].tolist()]
    assert len(ring) == len(set(ring))                   # each canonical cell once
    assert (4, NC) in ring and (J, 4) in ring
    assert all(1 <= i <= NC for _j, i in ring) and all(1 <= j <= J for j, _i in ring)
    piv = fold_pivot_cells(g.canon)
    assert not {(j, i) for j, i in ring if piv[j, i]}
    assert mesh._cgrid_ring_dropped["pivot"] == piv.sum() == (4 if fold == "F" else 2)
    build_elem_neighbors(mesh)                           # raises if an edge has > 2 cells
    # the wet cell east of the seam dry cell reaches it across the halo
    ji = [tuple(x) for x in mesh._cgrid_elem_ji.tolist()]
    nb = build_elem_neighbors(mesh)
    assert nb[ji.index((4, 1)), 3] == ji.index((4, NC))


def _corner_case():
    """Land block j, i >= 6 on a closed raster: node F(5, 5) has 3 wet cells
    and one land cell.  Uniform NE flow (faces into land closed)."""
    ny = nx = 10
    (flon, flat), _ = _grid(ny, nx)
    tmask = np.ones((ny, nx))
    tmask[0, :] = 0.0; tmask[:, 0] = 0.0
    tmask[6:, 6:] = 0.0
    u, v = _faces(flon, flat, tmask, vx=1.0, vy=1.0)
    return flon, flat, tmask, u, v


def _step_corner(ring):
    from kelp.core.particles import make_state
    from kelp.numerics.integrators import rk4_step
    flon, flat, tmask, u, v = _corner_case()
    e2u, e1v = _widths(flon, flat)
    mesh = build_cgrid_mesh(flon, flat, tmask, e2u=e2u, e1v=e1v, ring=ring)
    vel = VelocityField(u=u.ravel(), v=v.ravel(), eta=np.zeros(mesh.n_nodes),
                        uv_layout="cgrid")
    st = make_state([4.9 * D], [4.9 * D])
    st = st.replace(nearest_node=np.array([_nearest(mesh, 4.9 * D, 4.9 * D)], np.int32),
                    failure_reason=np.zeros(1, np.int8))
    dt = 0.3 * D * METERS_PER_DEG_LAT        # 1 m/s: the half step passes the corner by ~0.15 D
    return rk4_step(st, vel, mesh, dt, 1.0, uv_interp="cgrid"), mesh


def test_convex_corner_stage_stays_alive_in_ring():
    from kelp.core.particles import FailureReason, ParticleStatus
    old, _m = _step_corner(ring=False)                    # documents the old failure
    assert old.status[0] == ParticleStatus.OUTSIDE
    assert old.failure_reason[0] == FailureReason.INTERPOLATION_FAILED
    new, _m = _step_corner(ring=True)                     # land = zero velocity
    assert new.status[0] == ParticleStatus.ALIVE
    assert new.lon[0] > 4.9 * D and new.lat[0] > 4.9 * D            # and did move on
