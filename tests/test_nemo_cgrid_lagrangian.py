"""Data-gated (``KELP_TEST_NEMO_DATA``): Lagrangian benchmarks of the C-grid 3-D path on a
56 x 64 ORCA05 window (Denmark Strait / Iceland-Scotland sill, steady monthly record).

1. Transport: transport-weighted particles seeded on the inflow faces of a T-cell box leave it
   through the edges carrying the Eulerian face fluxes (u e2u e3u, v e1v e3v).
2. Uniformity (discrete Liouville): particles uniform per model volume (xi, eta, fractional
   level) stay uniform per cell; the old metre-based vertical placement left bottom cells
   20-26 % over-populated (chi2/dof 1.45), the fractional-level scheme gives 1.01 / 1.02-1.05.
"""
import os
from pathlib import Path

import numpy as np
import pytest

_d = os.path.abspath(os.environ["KELP_TEST_NEMO_DATA"]) if os.environ.get("KELP_TEST_NEMO_DATA") else None
pytestmark = pytest.mark.skipif(not _d, reason="KELP_TEST_NEMO_DATA not set")

# the window = ORCA05 (y, x) = (J0:J0+56, I0:I0+64) of the global 511 x 722 files
J0, I0, NY, NX = 390, 488, 56, 64


def _sib(name):
    for base in (Path(_d), Path(_d).parent, Path(_d).parent.parent):
        if (base / "nemo_orca05" / name).exists():
            return base / "nemo_orca05" / name
    pytest.skip(f"nemo_orca05/{name} not found near KELP_TEST_NEMO_DATA")


@pytest.fixture(scope="module")
def win(tmp_path_factory):
    import xarray as xr
    d = tmp_path_factory.mktemp("denmark")
    sl = dict(y=slice(J0, J0 + NY), x=slice(I0, I0 + NX))
    keep = ["glamt", "glamu", "glamv", "glamf", "gphit", "gphiu", "gphiv", "gphif", "e1t", "e1u", "e1v", "e1f",
            "e2t", "e2u", "e2v", "e2f", "e3t_0", "e3u_0", "e3v_0", "e3w_0", "e3t_1d", "e3w_1d", "gdept_1d",
            "gdepw_1d", "gdept_0", "gdepw_0", "tmask", "umask", "vmask", "fmask", "mbathy"]   # no misf: no cavities
    xr.open_dataset(_sib("mesh_mask_3d.nc"), decode_times=False)[keep].isel(**sl).load().to_netcdf(d / "mesh.nc")
    for g in "TUV":
        xr.open_dataset(_sib(f"ORCA05_1step3d_grid_{g}.nc"), decode_times=False).isel(**sl).load().to_netcdf(d / f"grid_{g}.nc")
    return d


def sq(a):
    return np.squeeze(np.asarray(a, dtype=np.float64))


def _load(d):
    import xarray as xr
    m = xr.open_dataset(d / "mesh.nc", decode_times=False)
    c = {k: sq(m[k].values) for k in ("glamf", "gphif", "glamt", "gphit", "e1t", "e2t", "e2u", "e1v",
                                       "e3t_0", "e3u_0", "e3v_0", "e3t_1d")}
    tm, um, vm = (sq(m[k].values) > 0.5 for k in ("tmask", "umask", "vmask"))
    c.update(tm=tm, e3t=c["e3t_0"], K=tm.shape[0])
    c["zw"] = np.concatenate([[0.0], np.cumsum(c["e3t_1d"])])
    u = sq(xr.open_dataset(d / "grid_U.nc", decode_times=False)["vozocrtx"].values)
    v = sq(xr.open_dataset(d / "grid_V.nc", decode_times=False)["vomecrty"].values)
    c["Tu"] = np.where(um, np.nan_to_num(u), 0.0) * c["e2u"] * c["e3u_0"]       # m3/s through U faces
    c["Tv"] = np.where(vm, np.nan_to_num(v), 0.0) * c["e1v"] * c["e3v_0"]
    return c


def _pt(c, j, i, a, b):
    """lon/lat of the bilinear point (a east, b north) of T cell (j, i) (F corners)."""
    cs = [(j - 1, i - 1), (j - 1, i), (j, i), (j, i - 1)]
    w = [(1 - a) * (1 - b), a * (1 - b), a * b, (1 - a) * b]
    return (sum(wk * c["glamf"][q] for wk, q in zip(w, cs)), sum(wk * c["gphif"][q] for wk, q in zip(w, cs)))


class _Locator:
    """Point -> T cell (j, i) by straight-edge quads on the F corners."""
    def __init__(self, c):
        from scipy.spatial import cKDTree
        self.nx = c["glamt"].shape[1]
        self.tree = cKDTree(np.c_[c["glamt"].ravel(), c["gphit"].ravel()])
        fl, fp = c["glamf"], c["gphif"]
        self.cx = np.stack([np.roll(np.roll(fl, 1, 0), 1, 1), np.roll(fl, 1, 0), fl, np.roll(fl, 1, 1)])
        self.cy = np.stack([np.roll(np.roll(fp, 1, 0), 1, 1), np.roll(fp, 1, 0), fp, np.roll(fp, 1, 1)])

    def locate(self, lon, lat, nc=8):
        _, cand = self.tree.query(np.c_[lon, lat], k=nc)
        J = np.full(lon.shape, -1); I = np.full(lon.shape, -1)
        for q in range(nc):
            jj, ii = np.divmod(cand[:, q], self.nx)
            inside = np.ones(lon.shape, bool)
            for e in range(4):
                x0, y0 = self.cx[e, jj, ii], self.cy[e, jj, ii]
                x1, y1 = self.cx[(e + 1) % 4, jj, ii], self.cy[(e + 1) % 4, jj, ii]
                inside &= ((x1 - x0) * (lat - y0) - (y1 - y0) * (lon - x0)) >= -1e-12
            new = inside & (J < 0)
            J[new] = jj[new]; I[new] = ii[new]
        return J, I


def _run(d, name, seeds, dt, n_steps, out_steps):
    from kelp.core.config import load_config
    from kelp.tracking import run_tracking
    np.savetxt(d / f"{name}.seeds", seeds, fmt="%.9f")
    (d / f"{name}.yaml").write_text(f"""mode: {{Fdrift3D: true, time_dir: 1.0, var_u: vozocrtx, var_v: vomecrty, var_eta: sossheig, track_failure_reason: true}}
numerics: {{dt: {dt}, integrator: rk4, output_freq: {out_steps}, uv_interp: cgrid, temporal_interp: false}}
mesh: {{type: nemo, nemo_grid: cgrid, meshpath: {d}/mesh.nc}}
data: {{ncinput: {d}/grid_T.nc, inicloud: {d}/{name}.seeds, output_path: {d}/{name}.zarr, time_averaging: instant}}
time: {{start_time_idx: 0, n_steps: {n_steps}, on_exhausted: cycle}}
""")
    run_tracking(load_config(str(d / f"{name}.yaml")))
    import xarray as xr
    return xr.open_zarr(d / f"{name}.zarr", consolidated=False, decode_times=False)


# --- 1. transport --------------------------------------------------------------
# box b5 = T cells j 22..26, i 20..24 of the window = ORCA05 j 412..416, i 508..512
B5 = (22, 26, 20, 24)


def _box_faces(c, box):
    j0, j1, i0, i1 = box
    out = {}
    def add(edge, F, jj, ii, sign, cell, a0, b0):
        for k in np.nonzero(F[:, jj, ii] != 0)[0]:
            out.setdefault(edge, []).append((int(k), sign * F[k, jj, ii], cell, a0, b0))
    for j in range(j0, j1 + 1):
        add("W", c["Tu"], j, i0 - 1, +1, (j, i0), 1e-3, None)
        add("E", c["Tu"], j, i1, -1, (j, i1), 1 - 1e-3, None)
    for i in range(i0, i1 + 1):
        add("S", c["Tv"], j0 - 1, i, +1, (j0, i), None, 1e-3)
        add("N", c["Tv"], j1, i, -1, (j1, i), None, 1 - 1e-3)
    return out


def test_box_transport_lagrangian_matches_eulerian_fluxes(win):
    assert (J0 + B5[0], J0 + B5[1], I0 + B5[2], I0 + B5[3]) == (412, 416, 508, 512)
    c = _load(win)
    faces = _box_faces(c, B5)
    eul = {e: -sum(f[1] for f in fs if f[1] < 0) for e, fs in faces.items()}
    tot = sum(f[1] for fs in faces.values() for f in fs if f[1] > 0)
    rng = np.random.default_rng(42)
    lon, lat, z, dT = [], [], [], []
    for e, fs in faces.items():
        for k, T, cell, a0, b0 in fs:
            if T <= 0:
                continue
            n = max(1, int(np.ceil(T / (tot / 5000))))     # weight T/n per particle: sums are exact
            s, xi = rng.uniform(size=n), rng.uniform(size=n)
            for q in range(n):
                x, y = _pt(c, *cell, a0 if a0 is not None else s[q], b0 if b0 is not None else s[q])
                lon.append(x); lat.append(y); z.append(c["zw"][k] + xi[q] * c["e3t"][(k, *cell)]); dT.append(T / n)
    dT = np.array(dT)
    ds = _run(win, "b5", np.c_[lon, lat, z], 3600, 60 * 24, 12)
    st, lo, la = (ds[v].values for v in ("status", "lon", "lat"))
    loc = _Locator(c)
    j0, j1, i0, i1 = B5
    exit_edge = np.full(dT.size, "", dtype="U8")
    for q in range(lo.shape[1]):
        idx = np.nonzero(exit_edge == "")[0]
        if idx.size == 0:
            break
        J, I = loc.locate(lo[idx, q], la[idx, q])
        out = np.where(J < 0, "unloc", "")
        out = np.where((J >= 0) & (J < j0), "S", out)
        out = np.where(J > j1, "N", out)
        out = np.where((J >= j0) & (J <= j1) & (I < i0), "W", out)
        out = np.where((J >= j0) & (J <= j1) & (I > i1), "E", out)
        out = np.where((st[idx, q] != 1) & (out == ""), "dead", out)
        exit_edge[idx[out != ""]] = out[out != ""]
    lag = {e: dT[exit_edge == e].sum() for e in list(eul) + ["dead", "unloc", ""]}
    print({e: (lag[e] / 1e6, eul.get(e, 0) / 1e6) for e in lag})
    assert lag["dead"] == 0 and lag["unloc"] == 0
    assert lag[""] / dT.sum() < 0.02                                         # measured 0.8 % (N=20000)
    assert abs(lag["S"] - eul["S"]) / eul["S"] < 0.01                        # measured 0.34 %
    assert sum(abs(lag[e] - eul[e]) for e in eul) / dT.sum() < 0.02          # measured 0.8 %


# --- 2. uniformity -------------------------------------------------------------
def test_uniform_per_volume_stays_uniform(win):
    c = _load(win)
    N, (bj0, bj1, bi0, bi1), margin = 20000, (20, 31, 24, 35), 3             # "densill" box, 6 x 6 core
    V = c["e1t"][None] * c["e2t"][None] * c["e3t"] * c["tm"]
    mk = np.zeros(V.shape, bool); mk[:, bj0:bj1 + 1, bi0:bi1 + 1] = True
    V = V * mk
    rng = np.random.default_rng(7)
    k, j, i = np.unravel_index(rng.choice(V.size, size=N, p=V.ravel() / V.sum()), V.shape)
    a, b, xi = rng.uniform(size=(3, N))
    ll = np.array([_pt(c, j[n], i[n], a[n], b[n]) for n in range(N)])
    z = c["zw"][k] + xi * c["e3t"][k, j, i]
    ds = _run(win, "uni", np.c_[ll, z], 3600, 72, 72)           # one output at t = 3 d
    lo, la, zz = (ds[v].isel(time=-1).values for v in ("lon", "lat", "z"))
    alive = ds.status.isel(time=-1).values == 1
    J, I = _Locator(c).locate(lo, la)
    ok = (J >= 0) & alive
    Jc, Ic = np.where(ok, J, 0), np.where(ok, I, 0)
    nlev = c["tm"].sum(0)
    kk = np.clip(np.searchsorted(c["zw"], zz, side="right") - 1, 0, np.maximum(nlev[Jc, Ic] - 1, 0))
    ok &= nlev[Jc, Ic] > 0
    cnt = np.zeros(V.shape); np.add.at(cnt, (kk[ok], Jc[ok], Ic[ok]), 1)
    E = N * V / V.sum()
    core = np.zeros(V.shape, bool); core[:, bj0 + margin:bj1 + 1 - margin, bi0 + margin:bi1 + 1 - margin] = True
    m = core & (E > 0)
    chi2 = ((cnt[m] - E[m]) ** 2 / E[m]).sum() / m.sum()
    tm = c["tm"]
    bottom = np.arange(c["K"])[:, None, None] == (nlev - 1)[None]
    ref = np.diff(c["zw"])[:, None, None]
    partial = bottom & (np.abs(c["e3t"] - ref) > 1e-3 * ref)
    ratios = {n: cnt[m & cm].sum() / E[m & cm].sum() for n, cm in (("bottom", bottom), ("partial", partial))}
    print(chi2, ratios)
    assert m.sum() > 500
    assert chi2 < 1.2                                  # measured 1.01 (old metre placement 1.45)
    assert all(0.9 < r < 1.1 for r in ratios.values()) # measured 1.02-1.05 (old 1.20-1.26)
