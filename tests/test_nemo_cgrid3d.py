"""3-D exact C-grid (full reference cells): W from continuity, plain RK stepping,
cell-value sampling.  Synthetic closed boxes only."""
import numpy as np
import pytest

from kelp.core.geometry import METERS_PER_DEG_LAT
from kelp.core.io import VelocityField
from kelp.core.particles import ParticleStatus, make_state
from kelp.numerics.cgrid3d import cgrid3d_step
from kelp.numerics.sampling import sample_field_cgrid
from kelp.readers.structured import (build_cgrid_mesh, cgrid_thickness,
                                     cgrid_w_from_continuity)
from tests.test_nemo_cgrid import D, _grid, _widths

K, NY, NX, DZ = 5, 6, 10, 10.0
ZW = DZ * np.arange(K + 1)


def _zps(nl, e3b):
    """Cell and face thicknesses (K, NY, NX) of the NEMO partial-step rule: a face
    takes the smaller of its two cells' thickness (the last column / row: its own)."""
    e3t = cgrid_thickness(ZW, np.where(nl > 0, nl - 1, -1), e3b)
    e3u, e3v = e3t.copy(), e3t.copy()
    e3u[:, :, :-1] = np.minimum(e3t[:, :, :-1], e3t[:, :, 1:])
    e3v[:, :-1] = np.minimum(e3t[:, :-1], e3t[:, 1:])
    return e3t, e3u, e3v


def _box(nlev=None, e3b=None, partial="shrunk"):
    """Closed box, wet cells j = 1..NY-2, i = 1..NX-2 (the rest dry), K layers
    (or ``nlev`` (NY, NX) levels per column, ``e3b`` the bottom-cell thicknesses)."""
    (flon, flat), _ = _grid(NY, NX)
    e2u, e1v = _widths(flon, flat)
    nl = np.full((NY, NX), K) if nlev is None else nlev
    nl = np.where(np.arange(NY)[:, None] * np.arange(NX)[None] > 0, nl, 0)
    nl[-1, :] = 0; nl[:, -1] = 0
    eb = np.full((NY, NX), DZ) if e3b is None else e3b
    _, e3u, e3v = _zps(nl, eb)
    mesh = build_cgrid_mesh(flon, flat, nl > 0, e2u=e2u, e1v=e1v, nlev2d=nl, zw=ZW, e3b=eb,
                            e3u0=e3u, e3v0=e3v, area=e2u * e1v, partial_cells=partial)
    return mesh, e2u, e1v, nl


def _w(mesh, u, v, e2u, e1v):
    ny, nx = mesh._structured_shape
    e3u, e3v = (cgrid_thickness(ZW, getattr(mesh, f"_cgrid_kb{n}").reshape(ny, nx),
                                getattr(mesh, f"_cgrid_e3{n}").reshape(ny, nx)) for n in "uv")
    return cgrid_w_from_continuity(u, v, e2u, e1v, e3u, e3v, e2u * e1v, mesh._cgrid_nlev_raster)


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
    w = _w(mesh, u, v, e2u, e1v)
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
    w = _w(mesh, u, v, e2u, e1v)
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
    w = _w(mesh2, u, v, e2u, e1v)
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
    w = _w(mesh, u, v, e2u, e1v)
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
    mesh, e2u, e1v, nl = _box(nl, e3b)
    assert mesh._cgrid_e3b[2 * NX + 3] == 4.0
    rng = np.random.default_rng(1)
    u = rng.normal(size=(K, NY, NX)); v = rng.normal(size=(K, NY, NX))
    _, e3u, e3v = _zps(nl, e3b)
    kk = np.arange(K)[:, None, None]                       # land faces carry no flow
    u[:, :, :-1] *= kk < np.minimum(nl[:, :-1], nl[:, 1:]); v[:, :-1] *= kk < np.minimum(nl[:-1], nl[1:])
    w = _w(mesh, u, v, e2u, e1v)
    j, i = np.mgrid[1:NY - 1, 1:NX - 1]
    for k in range(K):
        div = ((u[k, j, i] * e2u[j, i] * e3u[k, j, i] - u[k, j, i - 1] * e2u[j, i - 1] * e3u[k, j, i - 1]
                + v[k, j, i] * e1v[j, i] * e3v[k, j, i] - v[k, j - 1, i] * e1v[j - 1, i] * e3v[k, j - 1, i])
               + (w[k + 1, j, i] - w[k, j, i]) * e2u[j, i] * e1v[j, i])
        wet = k < nl[j, i]
        assert np.abs(div[wet]).max() < 1e-9 * np.abs(u).max() * e2u.max() * DZ
    # a face between a partial and a full cell has the smaller thickness; the
    # mesh holds it as one (level, thickness) pair per face
    assert e3u[2, 2, 2] == 4.0 and e3u[2, 2, 5] == 4.0 and e3u[1, 2, 2] == DZ
    assert mesh._cgrid_kbu[2 * NX + 2] == 2 and mesh._cgrid_e3u[2 * NX + 2] == 4.0
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


def test_crossing_into_a_thinner_cell_stays_in_the_fluid():
    """Deep cell (bottom thickness 10) -> thin neighbour (4): the fractional level
    is kept across the face, so the particle lands inside the neighbour, above
    its floor, and keeps moving (a metres face-map put it below the floor, at rest)."""
    e3b = np.full((NY, NX), DZ); e3b[:, 5:] = 4.0
    mesh, e2u, e1v, nl = _box(None, e3b)
    u = np.zeros((K, NY, NX)); u[K - 1, 1:-1, 1:-2] = 5e-2   # eastward in the bottom layer
    v = np.zeros_like(u)
    vel = _vel(mesh, u, v, _w(mesh, u, v, e2u, e1v))
    z0 = ZW[K - 1] + np.array([7.0, 9.5])       # both below the thin neighbour's floor (in metres)
    st = make_state([2.5 * D] * 2, [2.5 * D] * 2, list(z0))
    for _ in range(1500):
        st = cgrid3d_step(st, vel, mesh, 600.0, 1.0, "rk4", {})
    floor = ZW[K - 1] + 4.0
    assert (st.lon > 4.5 * D).all() and (st.z < floor).all()    # across the face, inside the fluid
    assert (np.abs(st.u) > 0).all()                              # and still moving


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
        NemoReader.prepare(mesh, ds)
    vel = NemoReader._read_cgrid_3d(ds, 0, mesh, np.float64, None)
    fl = mesh._cgrid_face_levels                          # land faces read as zero
    kk = np.arange(K)[:, None, None]
    np.testing.assert_allclose(vel.w.reshape(K + 1, NY, NX),
                               _w(mesh, u[0] * (kk < fl["U"]), v[0] * (kk < fl["V"]), e2u, e1v))
    with warnings.catch_warnings():
        warnings.simplefilter("error")                    # reference thicknesses: silent
        NemoReader.prepare(mesh, ds.assign(e3t_file=(dims, np.stack([ref, ref]))))


def test_porous_changes_metres_only_in_partial_bottom_cells():
    from kelp.core.mesh import find_containing_elements, find_nearest_nodes
    from kelp.numerics.cgrid3d import _level_coord
    nl = np.full((NY, NX), K); nl[2:4, 3:6] = 3
    e3b = np.full((NY, NX), DZ); e3b[2:4, 3:6] = 4.0
    rng = np.random.default_rng(2)
    kk = np.arange(K)[:, None, None]
    u = 1e-3 * rng.normal(size=(K, NY, NX)) * (kk < nl)
    v = 1e-3 * rng.normal(size=(K, NY, NX)) * (kk < nl)
    lon = np.full(3, 3.5 * D); lat = np.full(3, 2.5 * D)   # T(3, 4): nlev 3, e3b 4
    s0 = np.array([0.5, 2.25, 2.75])               # a full cell, then two inside the bottom cell
    res = {}
    for pc, e3 in (("shrunk", 4.0), ("porous", DZ)):
        mesh, e2u, e1v, _ = _box(nl, e3b, pc)
        vel = _vel(mesh, u, v, _w(mesh, u, v, e2u, e1v))
        z0 = np.array([5.0, 20 + 0.25 * e3, 20 + 0.75 * e3])   # the same s in either metres
        st = make_state(lon, lat, z0)
        for _ in range(5):
            st = cgrid3d_step(st, vel, mesh, 60.0, 1.0, "rk4", {})
        s = _level_coord(mesh, *find_containing_elements(mesh, st.lon, st.lat, find_nearest_nodes(mesh, st.lon, st.lat)), z=st.z)
        res[pc] = (st, s)
    (a, sa), (b, sb) = res["shrunk"], res["porous"]
    assert (a.lon == b.lon).all() and (a.lat == b.lat).all()    # dynamics unchanged
    assert np.allclose(sa, sb, atol=1e-12)                      # same s-trajectories
    assert a.z[0] == b.z[0] or abs(a.z[0] - b.z[0]) < 1e-12 * 10  # full cell: same metres
    assert np.allclose(a.z[1:], 20 + (sa[1:] - 2) * 4.0)        # shrunk: top + s*e3t_0
    assert np.allclose(b.z[1:], 20 + (sb[1:] - 2) * DZ)         # porous: top + s*e3t_1d


def test_partial_cells_refused_when_invalid():
    import pytest
    from kelp.core.config import MeshConfig
    with pytest.raises(ValueError, match="nemo_partial_cells"):
        MeshConfig(type="nemo", nemo_grid="cgrid", nemo_partial_cells="bogus")
    with pytest.raises(ValueError, match="nemo_partial_cells"):
        MeshConfig(type="fesom2", nemo_partial_cells="porous")


# --- halo-free fold layouts, 3-D, partial bottom cells on the fold row ----------
@pytest.mark.parametrize("fold", "TF")
def test_halo_free_fold_3d_load_flux_w_and_sampling(tmp_path, fold):
    import warnings

    import pandas as pd
    import xarray as xr

    from kelp.core.config import MeshConfig, ModeConfig
    from kelp.core.mesh import build_elem_neighbors
    from kelp.readers import get_reader
    from kelp.readers.structured import cgrid_to_canonical
    from tests._cgrid_orca import strip
    from tests.test_nemo_cgrid import _orca_fields

    K3 = 3
    zw = DZ * np.arange(K3 + 1)
    g, flon, flat, tm2, e2u, e1v, _u, _v = _orca_fields(fold, False)
    lay = g.canon
    ny, nx = lay.shape
    J = lay.J
    C = lambda a, k, vec=False: cgrid_to_canonical(a, lay, k, vector=vec)
    nl = np.where(tm2 > 0.5, K3, 0)
    nl[J - 1, 3:7] = np.where(nl[J - 1, 3:7] > 0, 2, 0)
    nl[J, 2:6] = np.where(nl[J, 2:6] > 0, 2, 0)                  # partial cells on the fold row
    nl = C(nl, "T").astype(int)
    e3b = np.full((ny, nx), DZ)
    e3b[J - 1, 3:7], e3b[J, 2:6] = 6.0, 4.0
    e3b = C(e3b, "T")
    e3t = cgrid_thickness(zw, np.where(nl > 0, nl - 1, -1), e3b)
    e3u, e3v = e3t.copy(), e3t.copy()
    e3u[:, :, :-1] = np.minimum(e3t[:, :, :-1], e3t[:, :, 1:])
    e3v[:, :-1] = np.minimum(e3t[:, :-1], e3t[:, 1:])
    e3u, e3v = C(e3u, "U"), C(e3v, "V")
    lvl = np.arange(K3)[:, None, None]
    rng = np.random.default_rng(0)
    u, v = rng.normal(size=(2, K3, ny, nx))
    u[:, :, :-1] *= lvl < np.minimum(nl[:, :-1], nl[:, 1:]); u[:, :, -1] = 0
    v[:, :-1] *= lvl < np.minimum(nl[:-1], nl[1:]); v[:, -1] = 0
    u, v = C(u, "U", True), C(v, "V", True)
    jj, ii = np.mgrid[0:ny, 0:nx]
    theta = C(1000.0 * lvl + 10.0 * jj + 0.01 * ii + 0 * lvl, "T")
    e1t = C(np.sqrt(e2u * e1v), "T")
    S = lambda a: strip(a, False)
    tc = {"time_counter": pd.date_range("2020-01-01", periods=2, freq="h")}
    rec = lambda a: np.stack([S(a)] * 2)
    xr.Dataset({"thetao": (("time_counter", "deptht", "y", "x"), rec(theta)),
                "e3t": (("time_counter", "deptht", "y", "x"), rec(e3t))},
               coords=tc).to_netcdf(tmp_path / "E_grid_T.nc")
    xr.Dataset({"uo": (("time_counter", "depthu", "y", "x"), rec(u))}, coords=tc).to_netcdf(tmp_path / "E_grid_U.nc")
    xr.Dataset({"vo": (("time_counter", "depthv", "y", "x"), rec(v))}, coords=tc).to_netcdf(tmp_path / "E_grid_V.nc")
    d3 = lambda a: (("t", "z", "y", "x"), S(a)[None])
    d2 = lambda a: (("y", "x"), S(a))
    xr.Dataset({
        "tmask": d3((lvl < nl).astype(float)), "nav_lon": (("y", "x"), g.xy["T"][0]),
        "nav_lat": (("y", "x"), g.xy["T"][1]), "glamf": d2(flon), "gphif": d2(flat),
        "e2u": d2(e2u), "e1v": d2(e1v), "e1t": d2(e1t), "e2t": d2(e1t),
        "e3t_1d": (("z",), np.full(K3, DZ)), "e3t_0": d3(e3t), "e3u_0": d3(e3u), "e3v_0": d3(e3v)},
        attrs=dict(Iperio=1, Jperio=0, NFold=1, NFtype=fold)).to_netcdf(tmp_path / "mesh_mask.nc")
    rd = get_reader("nemo")
    mesh = rd.load_mesh(MeshConfig(type="nemo", meshpath=str(tmp_path / "mesh_mask.nc"), nemo_grid="cgrid"))
    mode = ModeConfig(Fdrift3D=True, sample_vars=["thetao"])
    ds = rd.open_dataset(str(tmp_path / "E_grid_T.nc"), mode_cfg=mode)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rd.prepare(mesh, ds)
    vel = rd.read_velocities(ds, 0, mesh, mode_cfg=mode, sample_vars=["thetao"])
    U, V, W = vel.u.reshape(K3, ny, nx), vel.v.reshape(K3, ny, nx), vel.w.reshape(K3 + 1, ny, nx)
    np.testing.assert_array_equal(U, u)
    np.testing.assert_array_equal(V, v)
    E3u = cgrid_thickness(zw, mesh._cgrid_kbu.reshape(ny, nx), mesh._cgrid_e3u.reshape(ny, nx))
    E3v = cgrid_thickness(zw, mesh._cgrid_kbv.reshape(ny, nx), mesh._cgrid_e3v.reshape(ny, nx))
    nw = int((mesh._cgrid_nlev > 0).sum())
    j, i = mesh._cgrid_elem_ji[:nw].T
    nb = build_elem_neighbors(mesh)[:nw]
    e2, e1 = mesh._cgrid_e2u.reshape(ny, nx), mesh._cgrid_e1v.reshape(ny, nx)
    area = mesh._cgrid_area.reshape(ny, nx)[j, i]
    e, s = np.nonzero((nb >= 0) & (nb < nw))
    n = nb[e, s]
    s2 = np.array([np.flatnonzero(nb[b] == a)[0] for a, b in zip(e, n)])
    assert ((j[e] >= J - 1) & (np.abs(i[e] - i[n]) > 1)).sum() > 0          # fold edges are in play
    for k in range(K3):
        fu, fv = U[k] * e2 * E3u[k], V[k] * e1 * E3v[k]
        out = np.stack([-fv[j - 1, i], fu[j, i], fv[j, i], -fu[j, i - 1]], 1)
        scale = np.abs(out).max()
        assert np.abs(out[e, s] + out[n, s2]).max() <= 1e-12 * scale        # shared edges balance
        res = out.sum(1) + (W[k + 1, j, i] - W[k, j, i]) * area
        wet = k < mesh._cgrid_nlev_raster[j, i]
        assert np.abs(res[wet]).max() <= 1e-12 * scale                      # per-cell continuity
        got = sample_field_cgrid(vel.scalars["thetao"], mesh, mesh.elem_center_lon[:nw],
                                 mesh.elem_center_lat[:nw], np.full(nw, zw[k] + 0.5))
        np.testing.assert_array_equal(got[wet], theta[k, j, i][wet])        # incl. fold-row cells
        assert np.isnan(got[~wet]).all()


def test_garbage_on_land_faces_is_masked_by_the_tmask_faces():
    """u / v are zeroed on every face the tmask closes (umask / vmask = tmask *
    tmask), whatever the file holds there: same velocities as with zeros."""
    import xarray as xr
    from kelp.readers.nemo import NemoReader
    nl = np.full((NY, NX), K); nl[2:4, 3:6] = 3
    mesh, e2u, e1v, nl = _box(nlev=nl)
    rng = np.random.default_rng(5)
    kk = np.arange(K)[:, None, None]
    fl = mesh._cgrid_face_levels
    dims = ("time", "z", "y", "x")
    clean = [rng.normal(size=(K, NY, NX)) * (kk < fl[k]) for k in "UV"]
    dirty = [c + 7.0 * (kk >= fl[k]) for c, k in zip(clean, "UV")]
    out = [NemoReader._read_cgrid_3d(xr.Dataset({"u": (dims, u[None]), "v": (dims, v[None])}),
                                     0, mesh, np.float64, None) for u, v in (clean, dirty)]
    for f in ("u", "v", "w"):
        np.testing.assert_array_equal(getattr(out[0], f), getattr(out[1], f))


def test_vel_w_is_dz_dt_in_metres_per_second():
    mesh, e2u, e1v, nl = _box()
    u = np.zeros((K, NY, NX)); v = np.zeros_like(u)
    u[0, :, 1:4] = 0.05; u[0, :, 4:8] = -0.05          # convergence in layer 0 -> downwelling
    u[1, :, 1:4] = -0.05; u[1, :, 4:8] = 0.05
    vel = _vel(mesh, u, v, _w(mesh, u, v, e2u, e1v))
    st = make_state([3.5 * D], [2.5 * D], [5.0])
    out = cgrid3d_step(st, vel, mesh, 1.0, 1.0, "euler", {})
    dzdt = (out.z - st.z) / 1.0
    assert dzdt[0] != 0 and abs(out.w[0] - dzdt[0]) < 1e-3 * abs(dzdt[0])


def test_cgrid3d_step_refuses_diffusion_and_reconstruction():
    mesh, *_ = _box()
    z = np.zeros((K, NY, NX))
    vel = _vel(mesh, z, z, np.zeros((K + 1, NY, NX)))
    st = make_state([3.5 * D], [2.5 * D], [5.0])
    for kw in ({"diffusion": object()}, {"recon": object()}):
        with pytest.raises(NotImplementedError):
            cgrid3d_step(st, vel, mesh, 1.0, 1.0, "euler", kw)
