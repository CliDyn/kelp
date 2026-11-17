"""3-D exact C-grid (full reference cells): W from continuity, plain RK stepping,
cell-value sampling.  Synthetic closed boxes only."""
import numpy as np

from kelp.core.geometry import METERS_PER_DEG_LAT
from kelp.core.io import VelocityField
from kelp.core.particles import ParticleStatus, make_state
from kelp.numerics.cgrid3d import cgrid3d_step
from kelp.numerics.sampling import sample_field_cgrid
from kelp.readers.structured import build_cgrid_mesh, cgrid_w_from_continuity
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
    w = cgrid_w_from_continuity(u, v, e2u, e1v, np.diff(ZW), e2u * e1v, nl)
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
    w = cgrid_w_from_continuity(u, v, e2u, e1v, np.diff(ZW), e2u * e1v, nl)
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
    w = cgrid_w_from_continuity(u, v, e2u, e1v, np.diff(ZW), e2u * e1v, nl)
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
    w = cgrid_w_from_continuity(u, v, e2u, e1v, np.diff(ZW), e2u * e1v, nl)
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
