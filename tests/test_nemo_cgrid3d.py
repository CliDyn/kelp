"""3-D exact C-grid (full reference cells): W from continuity, plain RK stepping,
cell-value sampling.  Synthetic closed boxes only."""
import numpy as np

from kelp.core.geometry import METERS_PER_DEG_LAT
from kelp.core.io import VelocityField
from kelp.core.particles import ParticleStatus, make_state
from kelp.numerics.cgrid3d import cgrid3d_step
from kelp.numerics.sampling import sample_field_cgrid
from kelp.readers.structured import (build_cgrid_mesh, cgrid_face_thickness,
                                     cgrid_w_from_continuity)
from tests.test_nemo_cgrid import D, _grid, _widths

K, NY, NX, DZ = 5, 6, 10, 10.0
ZW = DZ * np.arange(K + 1)


def _box(nlev=None):
    """Closed box, wet cells j = 1..NY-2, i = 1..NX-2 (the rest dry), K layers
    (or ``nlev`` (NY, NX) levels per column)."""
    (flon, flat), _ = _grid(NY, NX)
    e2u, e1v = _widths(flon, flat)
    nl = np.full((NY, NX), K) if nlev is None else nlev
    nl = np.where(np.arange(NY)[:, None] * np.arange(NX)[None] > 0, nl, 0)
    nl[-1, :] = 0; nl[:, -1] = 0
    mesh = build_cgrid_mesh(flon, flat, nl > 0, e2u=e2u, e1v=e1v, nlev2d=nl, zw=ZW, depth2d=ZW[nl],
                            area=e2u * e1v)
    return mesh, e2u, e1v, nl


def _w(u, v, e2u, e1v, nl, e3b=None):
    eb = np.full(nl.shape, DZ) if e3b is None else e3b
    e3u, e3v = cgrid_face_thickness(ZW, nl, eb)
    return cgrid_w_from_continuity(u, v, e2u, e1v, e3u, e3v, e2u * e1v, nl)


def _vel(mesh, u, v, w):
    k = u.shape[0]
    return VelocityField(u=u.reshape(k, -1), v=v.reshape(k, -1),
                         w=w.reshape(k + 1, -1), eta=np.zeros(mesh.n_nodes),
                         uv_layout="cgrid")


def test_w_balances_every_cell_and_vanishes_at_the_floor():
    nl = np.full((NY, NX), K); nl[2:4, 3:6] = 2          # a shallow bump
    mesh, e2u, e1v, nl = _box(nl)
    rng = np.random.default_rng(0)
    u = rng.normal(size=(K, NY, NX)); v = rng.normal(size=(K, NY, NX))
    w = _w(u, v, e2u, e1v, nl)
    j, i = np.mgrid[1:NY - 1, 1:NX - 1]
    for k in range(K):
        div = ((u[k, j, i] * e2u[j, i] - u[k, j, i - 1] * e2u[j, i - 1]
                + v[k, j, i] * e1v[j, i] - v[k, j - 1, i] * e1v[j - 1, i]) * DZ
               + (w[k + 1, j, i] - w[k, j, i]) * e2u[j, i] * e1v[j, i])
        wet = k < nl[j, i]
        assert np.abs(div[wet]).max() < 1e-9 * np.abs(u).max() * e2u.max() * DZ
    k = np.arange(K + 1)[:, None, None]
    assert (w[k >= nl[None]] == 0).all()                  # floor and below


def test_uniform_layer_flow_and_land_below_floor():
    mesh, e2u, e1v, nl = _box()
    uk = np.array([0.1, -0.2, 0.3, 0.0, 0.05])
    u = np.broadcast_to(uk[:, None, None], (K, NY, NX)).copy(); v = np.zeros_like(u)
    w = _w(u, v, e2u, e1v, nl)
    vel = _vel(mesh, u, v, w)
    lon0, lat0 = 4.5 * D, 3.5 * D                         # interior, 2 cells from the walls
    dt = 60.0
    for k in (0, 1, 2):
        st = make_state([lon0], [lat0], [ZW[k] + 5.0])
        out = cgrid3d_step(st, vel, mesh, dt, 1.0, "rk4", {})
        assert out.status[0] == ParticleStatus.ALIVE
        np.testing.assert_allclose((out.lon[0] - lon0) * METERS_PER_DEG_LAT, uk[k] * dt,
                                   rtol=2e-3, atol=1e-9)
        assert abs(out.z[0] - (ZW[k] + 5.0)) < 1e-6        # no w in the interior
    # a column with a bump: every level at or below its floor is zero-velocity land
    # (the particle stays alive, stuck)
    nl = np.full((NY, NX), K); nl[4, 5] = 2
    mesh2, e2u, e1v, nl = _box(nl)
    w = _w(u, v, e2u, e1v, nl)
    vel2 = _vel(mesh2, u, v, w)
    st = make_state([lon0, lon0], [lat0, lat0], [ZW[2] + 1.0, ZW[3] + 1.0])
    out = cgrid3d_step(st, vel2, mesh2, dt, 1.0, "euler", {})
    assert out.status[0] == ParticleStatus.ALIVE and abs(out.lon[0] - lon0) < 1e-12
    assert out.status[1] == ParticleStatus.ALIVE and abs(out.lon[1] - lon0) < 1e-12


def test_closed_overturning_cell_keeps_particles_inside():
    mesh, e2u, e1v, nl = _box()
    n = NX - 2                                             # wet columns 1..n
    kk = np.arange(K + 1)[:, None]; ii = np.arange(NX)[None]
    phi = np.sin(np.pi * kk / K) * np.sin(np.pi * (ii - 0.0) / n)   # 0 at i=0, n; k=0, K
    phi[:, n + 1:] = 0.0
    u = np.zeros((K, NY, NX))
    u[:] = ((phi[:-1] - phi[1:]) / (e2u[None, 3] * DZ) * 1.0e4)[:, None, :]   # u east face of T(j,i)
    v = np.zeros_like(u)
    w = _w(u, v, e2u, e1v, nl)
    assert np.abs(w[0]).max() < 1e-12 * np.abs(w).max() + 1e-15        # closed at the lid
    vel = _vel(mesh, u, v, w)
    rng = np.random.default_rng(3)
    P = 40
    st = make_state(rng.uniform(1.1, 8.9, P) * D, rng.uniform(1.1, 4.9, P) * D,
                    rng.uniform(1.0, 49.0, P))
    z0 = st.z.copy()
    lo, hi, dz = [], [], 0.0
    for _ in range(300):
        st = cgrid3d_step(st, vel, mesh, 1.0e3, 1.0, "rk4", {})
        lo.append(st.z.min()); hi.append(st.z.max())
        dz = max(dz, np.abs(st.z - z0).max())
    assert (st.status == ParticleStatus.ALIVE).all()
    assert min(lo) >= 0.0 and max(hi) <= ZW[-1]
    assert st.lon.min() > D and st.lon.max() < (NX - 1) * D     # never leaves the wet box
    assert dz > 5.0                                                                    # and it really overturns


def test_sample_returns_the_cell_value():
    mesh, *_ = _box()
    T = np.arange(K * NY * NX, dtype=float).reshape(K, NY * NX)
    lon = np.array([2.5 * D, 7.3 * D, 2.5 * D, 20 * D]); lat = np.array([2.5 * D, 3.2 * D, 2.5 * D, 0.0])
    z = np.array([25.0, 5.0, 25.0, 5.0])
    nn = np.array([np.argmin((mesh.nodes_lon - a) ** 2 + (mesh.nodes_lat - b) ** 2)
                   for a, b in zip(lon, lat)])
    out = sample_field_cgrid(T, mesh, lon, lat, z, alive=np.array([1, 1, 0, 1], bool),
                             nearest=nn)
    # T(j, i) is centred at ((i - 1/2) D, (j - 1/2) D): cells (3, 3) and (4, 8)
    assert out[0] == T[2, 3 * NX + 3] and out[1] == T[0, 4 * NX + 8]
    assert np.isnan(out[2]) and np.isnan(out[3])               # dead / outside the mesh


def test_partial_bottom_cell_balance_floor_and_particle():
    nl = np.full((NY, NX), K); nl[2:4, 3:6] = 3
    e3b = np.full((NY, NX), DZ); e3b[2:4, 3:6] = 4.0; e3b[3, 6] = 7.0   # partial bottoms
    e3b[:, 6:8] = np.where(nl[:, 6:8] == K, 3.0, e3b[:, 6:8])
    (flon, flat), _ = _grid(NY, NX)
    e2u, e1v = _widths(flon, flat)
    nl = np.where(np.arange(NY)[:, None] * np.arange(NX)[None] > 0, nl, 0)
    nl[-1, :] = 0; nl[:, -1] = 0
    mesh = build_cgrid_mesh(flon, flat, nl > 0, e2u=e2u, e1v=e1v, nlev2d=nl, zw=ZW,
                            e3b=e3b, area=e2u * e1v)
    assert mesh._cgrid_e3b[2 * NX + 3] == 4.0
    rng = np.random.default_rng(1)
    u = rng.normal(size=(K, NY, NX)); v = rng.normal(size=(K, NY, NX))
    e3u, e3v = cgrid_face_thickness(ZW, nl, e3b)
    w = _w(u, v, e2u, e1v, nl, e3b)
    j, i = np.mgrid[1:NY - 1, 1:NX - 1]
    for k in range(K):
        div = ((u[k, j, i] * e2u[j, i] * e3u[k, j, i] - u[k, j, i - 1] * e2u[j, i - 1] * e3u[k, j, i - 1]
                + v[k, j, i] * e1v[j, i] * e3v[k, j, i] - v[k, j - 1, i] * e1v[j - 1, i] * e3v[k, j - 1, i])
               + (w[k + 1, j, i] - w[k, j, i]) * e2u[j, i] * e1v[j, i])
        wet = k < nl[j, i]
        assert np.abs(div[wet]).max() < 1e-9 * np.abs(u).max() * e2u.max() * DZ
    # a face between a partial and a full cell has the smaller thickness
    assert e3u[2, 2, 2] == 4.0 and e3u[2, 2, 5] == 4.0 and e3u[1, 2, 2] == DZ
    # floor at the true depth: a particle in the partial bottom cell of (3, 3) stays above it
    j0, i0 = 3, 4                                         # T(3, 4), nlev 3, e3b 4
    floor = ZW[nl[j0, i0] - 1] + e3b[j0, i0]
    assert floor == 24.0
    vel = _vel(mesh, u * 1e-3, v * 1e-3, w * 1e-3)
    st = make_state([(i0 - 0.5) * D], [(j0 - 0.5) * D], [floor - 0.5])
    z = []
    for _ in range(50):
        st = cgrid3d_step(st, vel, mesh, 60.0, 1.0, "rk4", {})
        z.append(st.z[0])
    assert st.status[0] == ParticleStatus.ALIVE and max(z) <= floor


def test_zstar_thicknesses_warn_and_use_the_reference():
    import warnings
    import pytest
    import xarray as xr
    from kelp.readers.nemo import NemoReader
    mesh, e2u, e1v, nl = _box()
    rng = np.random.default_rng(2)
    u = rng.normal(size=(2, K, NY, NX)); v = rng.normal(size=(2, K, NY, NX))
    ref = np.broadcast_to(np.diff(ZW)[:, None, None], (K, NY, NX))
    dims = ("time", "z", "y", "x")
    ds = xr.Dataset({"u": (dims, u), "v": (dims, v),
                     "e3t_file": (dims, np.stack([ref, 1.05 * ref]))})
    with pytest.warns(RuntimeWarning, match=r"5\.0%.*z\*/vvl"):
        vel = NemoReader._read_cgrid_3d(ds, 0, mesh, np.float64, None)
    np.testing.assert_allclose(vel.w.reshape(K + 1, NY, NX), _w(u[0], v[0], e2u, e1v, nl))
    with warnings.catch_warnings():
        warnings.simplefilter("error")                    # reference thicknesses: silent
        NemoReader._read_cgrid_3d(ds.assign(e3t_file=(dims, np.stack([ref, ref]))), 0,
                                  _box()[0], np.float64, None)
