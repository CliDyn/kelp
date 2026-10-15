"""FESOM-C vertical velocity contract (review 2026-09-22/23; FESOM-C source e3cde90).

FESOM-C writes two vertical velocities (``src/fv_ncoutputmpi.F90``):

* ``w`` = ``Wvel`` (``fv_3D_vel.F90:vert_vel_sigma``): the flux through the
  sigma interfaces per unit area, positive UP, on the nsigma interfaces,
  zero at the surface and at the bottom;
* ``w_cart`` = ``W_n`` (``vert_vel_cart``): a diagnostic at the layer mids
  whose flux term is ``z_mid (Wvel_t - Wvel_b)`` instead of
  ``z_t Wvel_t - z_b Wvel_b`` — it lacks the dia-sigma part of the Cartesian
  velocity, so it cannot drive a tracker and the reader refuses it.

KELP's sigma tracker integrates the depth below the moving surface,
q = (1 - sigma) H, whose material rate on interface k is

    dq/dt = (1 - sigma_k) (eta_t + u_k . grad H) - Wvel_k ,

with u_k the horizontal velocity interpolated linearly in sigma between the
two adjacent layer mids (thickness weights — FESOM-C allows stretched sigma).
Since the review of PR #102 the step integrates the sigma fraction itself,
s = z / H_p with ds/dt = -Wvel / H_p, and places the particle at z = s H_p;
the reader supplies -Wvel (positive down) alone.  That rate is dq/dt above,
written in the coordinate that makes the surface and slope terms exact.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from kelp.core.geometry import METERS_PER_DEG_LAT
from kelp.readers import get_reader

SIG_U = np.linspace(1.0, 0.0, 11)                    # uniform
SIG_S = 1.0 - np.linspace(0.0, 1.0, 11) ** 2         # stretched (fine at the top)


def _grid(lat0=0.0):
    xs, ys = np.array([4.0, 5.0, 6.0]), lat0 + np.array([-1.0, 0.0, 1.0])
    gx, gy = np.meshgrid(xs, ys)
    lon, lat = gx.ravel(), gy.ravel()
    tris = []
    for j in range(2):
        for i in range(2):
            n0 = j * 3 + i
            tris += [[n0, n0 + 1, n0 + 3], [n0 + 1, n0 + 4, n0 + 3]]
    tris = np.array(tris)
    nv = np.column_stack([tris + 1, tris[:, :1] + 1]).astype(np.int32)
    return lon, lat, tris, nv


def _write(path, times, eta, wvel_up, sig=SIG_U, u_elem=None, v_elem=None,
           depth=10.0, with_w_cart=False):
    """FESOM-C file: eta (T, N), wvel_up (T, K, N) on the interfaces."""
    lon, lat, tris, nv = _grid()
    T, N, E, K = len(times), lon.size, tris.shape[0], sig.size
    u = np.zeros((T, E, K - 1)) if u_elem is None else u_elem
    v = np.zeros((T, E, K - 1)) if v_elem is None else v_elem
    dvars = {
        "lon": (["node"], lon), "lat": (["node"], lat),
        "depth": (["node"], np.full(N, depth) if np.isscalar(depth)
                  else np.asarray(depth, float)),
        "nv": (["nele", "four"], nv),
        "lon_elem": (["nele"], lon[tris].mean(1)),
        "lat_elem": (["nele"], lat[tris].mean(1)),
        "sigma_lev": (["sigma"], sig),
        "u": (["time", "nele", "sigmam1"], u),
        "v": (["time", "nele", "sigmam1"], v),
        "w": (["time", "node", "sigma"], np.transpose(wvel_up, (0, 2, 1))),
        "eta": (["time", "node"], eta),
    }
    if with_w_cart:
        dvars["w_cart"] = (["time", "node", "sigma"], np.zeros((T, N, K)))
    ds = xr.Dataset(dvars, coords={"time": ("time", np.asarray(times, float),
                                            {"units": "seconds since 2000-01-01 00:00:00"})})
    ds.to_netcdf(path)
    return path


def _mode(**kw):
    base = dict(Fdrift2D=False, Fdrift3D=True, time_dir=1.0, Vpuls=False,
                var_u=None, var_v=None, var_w=None, var_eta=None,
                sigma_layer=None, sigma_mean=False, max_levels=None,
                w_positive="auto")
    base.update(kw)
    return SimpleNamespace(**base)


def _read(path, t=0, mode=None):
    reader = get_reader("fesomc")
    mesh = reader.load_mesh(SimpleNamespace(meshpath=str(path)))
    ds = reader.open_dataset(str(path))
    vel = reader.read_velocities(ds, t, mesh, mode or _mode())
    ds.close()
    return mesh, vel


class TestReaderConversion:
    def test_dia_sigma_flux_and_sign(self, tmp_path):
        # flat steady surface, no horizontal flow: the rate is -Wvel
        prof = 1e-4 * np.sin(np.pi * (1.0 - SIG_U))          # 0 at both ends
        w = np.broadcast_to(prof[None, :, None], (3, SIG_U.size, 9)).copy()
        p = _write(tmp_path / "flat.nc", [0.0, 100.0, 200.0], np.zeros((3, 9)), w)
        _m, vel = _read(p, t=1)
        np.testing.assert_allclose(vel.w[:, 4], -prof, atol=1e-18)

    def test_config_down_is_honoured(self, tmp_path):
        prof = 1e-4 * np.sin(np.pi * (1.0 - SIG_U))
        w = np.broadcast_to(prof[None, :, None], (2, SIG_U.size, 9)).copy()
        p = _write(tmp_path / "down.nc", [0.0, 100.0], np.zeros((2, 9)), w)
        _m, v_up = _read(p)
        _m, v_dn = _read(p, mode=_mode(w_positive="down"))
        np.testing.assert_allclose(v_dn.w, -v_up.w)

    def test_flux_alone_whatever_the_flow_and_bottom(self, tmp_path):
        """Since the sigma step integrates s = z / H (review of PR #102) the
        reader returns only the dia-sigma flux, positive down: the surface
        motion and the flow along a sloping bottom are carried by the step
        (TestSigmaIntegration), not by W."""
        g, vn, s_eta = 2e-5, 0.3, 1e-6
        lon, lat, tris, _ = _grid()
        depth = 10.0 + g * lat * METERS_PER_DEG_LAT
        eta = np.tile(s_eta * lat * METERS_PER_DEG_LAT, (2, 1))
        v_elem = np.full((2, tris.shape[0], SIG_S.size - 1), vn)
        prof = 1e-4 * np.sin(np.pi * (1.0 - SIG_S))
        w = np.broadcast_to(prof[None, :, None], (2, SIG_S.size, 9)).copy()
        p = _write(tmp_path / "flow.nc", [0.0, 100.0], eta, w, sig=SIG_S,
                   v_elem=v_elem, depth=depth)
        _m, vel = _read(p)
        np.testing.assert_allclose(vel.w, -np.broadcast_to(prof[:, None],
                                                           (SIG_S.size, 9)),
                                   atol=1e-18)

    def test_w_cart_is_refused(self, tmp_path):
        p = _write(tmp_path / "cart.nc", [0.0, 100.0], np.zeros((2, 9)),
                   np.zeros((2, SIG_U.size, 9)), with_w_cart=True)
        with pytest.raises(ValueError, match="w_cart"):
            _read(p, mode=_mode(var_w="w_cart"))


def _run_column(tmp_path, name, times, eta, *, depth=10.0, sig=SIG_U,
                lat=0.0, z0=3.7, integrator="rk4", dt=150.0, n=None,
                direction=1.0, recon="linear", w_interp="leastsq",
                v_elem=None, on_exhausted="stop", start=None, wvel=None,
                diffusion=None, temporal=True, with_times=True):
    """Run one particle through a FESOM-C file (Wvel = 0 unless given);
    return the records (time, lat, z)."""
    from kelp.core.config import NumericsConfig
    from kelp.core.mesh import find_nearest_nodes
    from kelp.core.particles import make_state
    from kelp.readers import extract_time_info
    from kelp.tracking import run_single_realization
    T = len(times)
    p = _write(tmp_path / name, times, eta,
               np.zeros((T, sig.size, 9)) if wvel is None else wvel,
               sig=sig, depth=depth, v_elem=v_elem)
    reader = get_reader("fesomc")
    mesh = reader.load_mesh(SimpleNamespace(meshpath=str(p)))
    ds = reader.open_dataset(str(p))
    file_times, file_dt = extract_time_info(ds)
    st = make_state(np.array([5.0]), np.array([lat]), np.array([z0]))
    st = st.replace(nearest_node=find_nearest_nodes(mesh, st.lon, st.lat))
    if start is None:
        start = 0 if direction > 0 else T - 1
    rec = run_single_realization(
        mesh, ds, reader, st, _mode(time_dir=direction),
        NumericsConfig(dt=dt, integrator=integrator, output_freq=1,
                       temporal_reconstruction=recon, w_interp=w_interp,
                       **({} if diffusion is None
                          else {"diffusion": diffusion})),
        start_time_idx=start, n_steps=n, n_time_total=T,
        on_exhausted=on_exhausted, temporal_interp=temporal,
        file_times=file_times if with_times else None,
        file_dt=file_dt if with_times else None)
    ds.close()
    a = rec.to_arrays()                        # (time, particle)
    return a["time"], a["lat"][:, 0], a["z"][:, 0]


def _eta_tracker(t, times, eta_node, recon):
    """The surface elevation the tracker interpolates at time t."""
    from kelp.numerics.temporal_reconstruct import _pchip_value_scalar
    j = int(np.clip(np.searchsorted(times, t, side="right") - 1, 0,
                    len(times) - 2))
    if recon == "pchip" and 1 <= j <= len(times) - 3:
        return _pchip_value_scalar(*eta_node[j - 1:j + 3], *times[j - 1:j + 3], t)
    return np.interp(t, times, eta_node)


class TestSurfaceMotionInTheStep:
    """Review round 2: a centred-difference eta_t baked into W was not
    the derivative of the eta the tracker interpolates in time, so a particle
    that moves with the sigma surfaces drifted off them between snapshots
    (only full-interval checks telescoped the error away), and cycle runs went
    one-sided at the seam.  The step places the particle at z = s H_p with the
    surface of its own temporal interpolant (sigma-fraction integration since
    the review of PR #102).  Checked at EVERY record, mid-interval included."""

    @pytest.mark.parametrize("integrator", ["euler", "rk2", "rk4"])
    @pytest.mark.parametrize("direction", [1.0, -1.0], ids=["fwd", "bwd"])
    @pytest.mark.parametrize("recon", ["linear", "pchip"])
    def test_particle_keeps_its_sigma_on_a_tide(self, tmp_path, integrator,
                                                direction, recon):
        times = np.arange(9) * 600.0
        tide = 1.5 * np.sin(2 * np.pi * times / 3000.0)   # 5 snapshots/period
        eta = np.tile(tide[:, None], (1, 9))
        H_start = 10.0 + tide[0 if direction > 0 else -1]
        q0 = 0.37 * H_start
        t, _lat, z = _run_column(tmp_path, "tide.nc", times, eta, z0=q0,
                                 integrator=integrator, n=12,
                                 direction=direction, recon=recon)
        # Euler ignores the PCHIP stencil: its surface is the linear one
        eff = "linear" if integrator == "euler" else recon
        H = np.array([10.0 + _eta_tracker(ti, times, tide, eff) for ti in t])
        np.testing.assert_allclose(z / H, 0.37, rtol=1e-12)

    def test_cycle_seam(self, tmp_path):
        times = np.arange(5) * 600.0
        tide = 1.5 * np.sin(2 * np.pi * times / 3000.0)
        eta = np.tile(tide[:, None], (1, 9))
        t, _lat, z = _run_column(tmp_path, "cyc.nc", times, eta,
                                 z0=0.37 * (10.0 + tide[0]), n=40,
                                 on_exhausted="cycle")
        # the tracker's surface is piecewise linear in time, the seam
        # included: at every record z / H is the start value
        assert len(z) == 40
        ratio_at_snapshots = []
        for ti, zi in zip(t, z):
            k = int(round(ti / 600.0))
            if abs(ti - 600.0 * k) < 1e-9:
                ratio_at_snapshots.append(zi / (10.0 + tide[k % 5]))
        assert len(ratio_at_snapshots) >= 8
        np.testing.assert_allclose(ratio_at_snapshots, 0.37, rtol=1e-12)


class TestSigmaFollowingOnASlope:
    """Review round 2: neighbour columns were combined at a fixed
    DEPTH, so on a sloping bottom a particle left its sigma surface even with
    Wvel = 0.  Neighbours are now evaluated at the particle's sigma, with the
    column height interpolated like W — exact for a linear bottom."""

    @pytest.mark.parametrize("w_interp", ["leastsq", "barycentric"])
    def test_flow_along_a_sloping_bottom_keeps_sigma(self, tmp_path, w_interp):
        g, vn, d0 = 5e-5, 0.5, 20.0
        lon, lat, tris, _ = _grid()
        depth = d0 + g * lat * METERS_PER_DEG_LAT        # 27 % per node row
        times = np.arange(8) * 3600.0
        v_elem = np.full((8, tris.shape[0], SIG_U.size - 1), vn)
        y0 = -0.5
        H0 = d0 + g * y0 * METERS_PER_DEG_LAT
        _t, lat_rec, z = _run_column(
            tmp_path, f"slope_{w_interp}.nc", times, np.zeros((8, 9)),
            depth=depth, lat=y0, z0=0.7 * H0, dt=600.0, n=40,
            w_interp=w_interp, v_elem=v_elem)
        H = d0 + g * lat_rec * METERS_PER_DEG_LAT
        np.testing.assert_allclose(z / H, 0.7, rtol=1e-6)


class TestSigmaIntegration:
    """Final review of PR #102: the surface term used the
    start-of-step sigma and position for every stage (first order in dt when
    a dia-sigma flux and a moving surface act together), steps that fell back
    to steady velocity carried no surface motion, and diffusion kicks used the
    start-of-step column.  The sigma step now integrates s = z / H_p itself
    (ds/dt = -Wvel / H_p; z = s H_p(x, t) at every stage and at the end), so
    the combined case has an exact discrete solution."""

    @pytest.mark.parametrize("integrator", ["euler", "rk2", "rk4"])
    @pytest.mark.parametrize("direction", [1.0, -1.0], ids=["fwd", "bwd"])
    @pytest.mark.parametrize("sig", [SIG_U, SIG_S], ids=["uniform", "stretched"])
    def test_tide_slope_flow_and_dia_sigma_flux(self, tmp_path, integrator,
                                                direction, sig):
        g, vn, d0, c = 5e-5, 0.5, 20.0, 1e-5
        lon, lat, tris, _ = _grid()
        depth = d0 + g * lat * METERS_PER_DEG_LAT
        times = np.arange(13) * 600.0
        tide = 1.5 * np.sin(2 * np.pi * times / 3000.0)
        eta = np.tile(tide[:, None], (1, 9))
        H_nodes = depth[None, :] + eta                    # (T, N)
        # uniform dsigma/dt = c: Wvel = c H (positive up) on every interface
        wvel = c * np.broadcast_to(H_nodes[:, None, :], (13, sig.size, 9)).copy()
        wvel[:, 0, :] = 0.0                                # surface: no flux
        wvel[:, -1, :] = 0.0                               # bottom: no flux
        v_elem = np.full((13, tris.shape[0], sig.size - 1), vn)
        y0 = -0.5
        k0 = 0 if direction > 0 else 12
        H0 = d0 + g * y0 * METERS_PER_DEG_LAT + tide[k0]
        s0 = 0.4
        t, lat_rec, z = _run_column(
            tmp_path, "combined.nc", times, eta, depth=depth, sig=sig,
            lat=y0, z0=s0 * H0, integrator=integrator, dt=150.0, n=24,
            direction=direction, v_elem=v_elem, wvel=wvel)
        H = (d0 + g * lat_rec * METERS_PER_DEG_LAT
             + np.array([_eta_tracker(ti, times, tide, "linear") for ti in t]))
        s_exact = s0 - c * (t - times[k0])       # time runs backward too
        np.testing.assert_allclose(z / H, s_exact, rtol=1e-9)

    @pytest.mark.parametrize("integrator", ["euler", "rk2", "rk4"])
    @pytest.mark.parametrize("dt", [400.0, 900.0])
    def test_steady_fallback_steps_keep_sigma(self, tmp_path, integrator, dt):
        """dt not fitting the 600 s file step: steps across a snapshot fall
        back to steady velocity; the column still moves with the surface."""
        times = np.arange(13) * 600.0
        tide = 1.5 * np.sin(2 * np.pi * times / 3000.0)
        eta = np.tile(tide[:, None], (1, 9))
        q0 = 0.37 * (10.0 + tide[0])
        import contextlib
        # RK stages warn about the fallback; Euler has no stages to lose
        ctx = (pytest.warns(RuntimeWarning) if integrator != "euler"
               else contextlib.nullcontext())
        with ctx:
            t, _lat, z = _run_column(tmp_path, f"fallback_{int(dt)}.nc",
                                     times, eta, z0=q0, integrator=integrator,
                                     dt=dt, n=int(6000 // dt))
        H = np.array([10.0 + _eta_tracker(ti, times, tide, "linear")
                      for ti in t])
        np.testing.assert_allclose(z / H, 0.37, rtol=1e-12)

    def test_az_kick_uses_the_end_of_step_column(self, tmp_path):
        """A near-bottom particle on a rising tide must not be reflected at
        the start-of-step bottom (review repro: 7 cm upward in one step)."""
        from kelp.core.config import DiffusionConfig
        times = np.arange(9) * 600.0
        tide = 0.8 * np.sin(2 * np.pi * times / 3000.0)
        eta = np.tile(tide[:, None], (1, 9))
        diff = DiffusionConfig(enabled=True, scheme="uniform", kh_zonal=0.0,
                               kh_meridional=0.0, az_enabled=True,
                               az_const=1e-14, seed=3)
        s0 = 0.975
        t, _lat, z = _run_column(tmp_path, "azbot.nc", times, eta,
                                 z0=s0 * (10.0 + tide[0]), integrator="rk2",
                                 dt=150.0, n=8, diffusion=diff)
        H = np.array([10.0 + _eta_tracker(ti, times, tide, "linear")
                      for ti in t])
        np.testing.assert_allclose(z / H, s0, rtol=1e-6)


class TestSigmaStepOrder:
    """Review of fix/sigma-step: the combined test above is exact
    for any placement of the RK stages (uniform dsigma/dt makes s linear in
    t).  A dsigma/dt linear in s gives an exponential exact solution,
    W/H = a (s - 0.3)  ->  s = 0.3 + (s0 - 0.3) exp(a (t - t0)), on which
    Euler/RK2/RK4 must show orders 1/2/4 — with a tide whose amplitude
    varies in space, a sloping bottom, a flow and an off-node particle."""

    A, SC, S0, G, D0, VN = 2e-4, 0.3, 0.4, 5e-5, 20.0, 0.5
    LON0, LAT0 = 5.3, -0.37

    def _amp(self, lat):
        return 1.0 + 0.5 * lat

    def _case(self, tmp_path, name):
        lon, lat, tris, _ = _grid()
        times = np.arange(9) * 600.0
        depth = self.D0 + self.G * lat * METERS_PER_DEG_LAT
        S = np.sin(2 * np.pi * times / 3000.0)
        eta = self._amp(lat)[None, :] * S[:, None]
        H = depth[None, :] + eta
        g = self.A * ((1.0 - SIG_U) - self.SC)            # W / H on the interfaces
        wvel = -g[None, :, None] * H[:, None, :]            # positive up
        v_elem = np.full((9, tris.shape[0], SIG_U.size - 1), self.VN)
        return times, S, eta, depth, wvel, v_elem

    def _end_error(self, tmp_path, integrator, dt, direction):
        times, S, eta, depth, wvel, v_elem = self._case(tmp_path, "ord")
        k0 = 0 if direction > 0 else 8
        H0 = (self.D0 + self.G * self.LAT0 * METERS_PER_DEG_LAT
              + self._amp(self.LAT0) * S[k0])
        t, lat_rec, z = _run_column(
            tmp_path, f"ord_{integrator}_{int(dt)}_{int(direction)}.nc",
            times, eta, depth=depth, lat=self.LAT0, z0=self.S0 * H0,
            integrator=integrator, dt=dt, n=int(2400 // dt),
            direction=direction, v_elem=v_elem, wvel=wvel)
        H = (self.D0 + self.G * lat_rec[-1] * METERS_PER_DEG_LAT
             + self._amp(lat_rec[-1])
             * np.interp(t[-1], times, S))
        s_exact = self.SC + (self.S0 - self.SC) * np.exp(
            self.A * (t[-1] - times[k0]))
        return abs(z[-1] / H - s_exact)

    @pytest.mark.parametrize("integrator,order", [("euler", 1.0), ("rk2", 2.0),
                                                  ("rk4", 4.0)])
    @pytest.mark.parametrize("direction", [1.0, -1.0], ids=["fwd", "bwd"])
    def test_order_of_convergence(self, tmp_path, integrator, order,
                                  direction):
        e1 = self._end_error(tmp_path, integrator, 300.0, direction)
        e2 = self._end_error(tmp_path, integrator, 150.0, direction)
        observed = np.log2(e1 / e2)
        assert observed > order - 0.3, (integrator, e1, e2, observed)


class TestSigmaKicks:
    """The kicks act in the column the step placed the particle in — the
    tracking layer's end-of-step surface (review of fix/sigma-step: the
    horizontal kick's s-rescale was untested, and a kick that fell back to
    the start surface passed on the steps where the two coincide)."""

    @pytest.mark.parametrize("integrator", ["euler", "rk2", "rk4"])
    @pytest.mark.parametrize("dt", [150.0, 400.0])
    @pytest.mark.parametrize("recon", ["linear", "pchip"])
    def test_horizontal_kick_keeps_sigma(self, tmp_path, integrator, dt,
                                         recon):
        import warnings as _w
        from kelp.core.config import DiffusionConfig
        g, d0 = 5e-5, 20.0
        lon, lat, tris, _ = _grid()
        depth = d0 + g * lat * METERS_PER_DEG_LAT
        times = np.arange(13) * 600.0
        tide = 1.5 * np.sin(2 * np.pi * times / 3000.0)
        eta = np.tile(tide[:, None], (1, 9))
        diff = DiffusionConfig(enabled=True, scheme="uniform", kh_zonal=20.0,
                               kh_meridional=10.0, seed=7)
        y0 = -0.2
        H0 = d0 + g * y0 * METERS_PER_DEG_LAT + tide[0]
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            t, lat_rec, z = _run_column(
                tmp_path, f"khsig_{integrator}_{int(dt)}_{recon}.nc", times,
                eta, depth=depth, lat=y0, z0=0.55 * H0,
                integrator=integrator, dt=dt, n=int(4800 // dt),
                diffusion=diff, recon=recon)
        eff = "linear" if integrator == "euler" else recon
        H = (d0 + g * lat_rec * METERS_PER_DEG_LAT
             + np.array([_eta_tracker(ti, times, tide, eff) for ti in t]))
        assert np.ptp(lat_rec) > 0.0                   # the kicks did move it
        np.testing.assert_allclose(z / H, 0.55, rtol=1e-10)

    @pytest.mark.parametrize("integrator", ["euler", "rk2", "rk4"])
    def test_az_kick_in_the_end_column_on_fallback_steps(self, tmp_path,
                                                         integrator):
        import warnings as _w
        from kelp.core.config import DiffusionConfig
        times = np.arange(13) * 600.0
        tide = 0.8 * np.sin(2 * np.pi * times / 3000.0)
        eta = np.tile(tide[:, None], (1, 9))
        diff = DiffusionConfig(enabled=True, scheme="uniform", kh_zonal=0.0,
                               kh_meridional=0.0, az_enabled=True,
                               az_const=1e-14, seed=3)
        s0 = 0.985
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            t, _lat, z = _run_column(tmp_path, f"azfb_{integrator}.nc",
                                     times, eta, z0=s0 * (10.0 + tide[0]),
                                     integrator=integrator, dt=400.0, n=12,
                                     diffusion=diff)
        H = np.array([10.0 + _eta_tracker(ti, times, tide, "linear")
                      for ti in t])
        np.testing.assert_allclose(z / H, s0, rtol=1e-6)


class TestDiscreteMode:
    """Discrete mode (one record per step): the step ends at the next
    record, whose surface the tracking layer supplies (review of
    fix/sigma-step: no sigma test ran without temporal interpolation).  The
    record written after step k is labelled (k+1)·dt (format 2) and holds the
    END state."""

    @pytest.mark.parametrize("integrator", ["euler", "rk2", "rk4"])
    @pytest.mark.parametrize("with_times", [True, False],
                             ids=["times", "no-times"])
    def test_discrete_mode_keeps_sigma(self, tmp_path, integrator,
                                       with_times):
        times = np.arange(9) * 600.0
        tide = 1.5 * np.sin(2 * np.pi * times / 3000.0)
        eta = np.tile(tide[:, None], (1, 9))
        t, _lat, z = _run_column(
            tmp_path, f"disc_{integrator}_{with_times}.nc", times, eta,
            z0=0.37 * (10.0 + tide[0]), integrator=integrator, dt=600.0,
            n=6, temporal=False, with_times=with_times)
        # format 2 labels the end of the step in seconds (records 600 s
        # apart from record 0): the record the step ends at is t / 600
        k_end = np.rint(np.asarray(t) / 600.0).astype(int)
        H = 10.0 + tide[k_end]
        np.testing.assert_allclose(z / H, 0.37, rtol=1e-12)


class TestPchipWithFallbackSteps:
    @pytest.mark.parametrize("integrator", ["rk2", "rk4"])
    def test_sigma_continuous_across_fallback_steps(self, tmp_path,
                                                    integrator):
        """Review of fix/sigma-step: with PCHIP a step that falls
        back to steady velocity started and ended on the LINEAR surface while
        its neighbours used PCHIP, so s jumped at each fallback boundary
        (2.7 % here).  The step now starts from the column the previous one
        ended in, and a fallback ends on the PCHIP surface where the stencil
        allows."""
        times = np.arange(13) * 600.0
        tide = 1.5 * np.sin(2 * np.pi * times / 3000.0)
        eta = np.tile(tide[:, None], (1, 9))
        q0 = 0.37 * (10.0 + tide[0])
        with pytest.warns(RuntimeWarning):
            t, _lat, z = _run_column(tmp_path, f"pfb_{integrator}.nc", times,
                                     eta, z0=q0, integrator=integrator,
                                     dt=400.0, n=15, recon="pchip")
        H = np.array([10.0 + _eta_tracker(ti, times, tide, "pchip")
                      for ti in t])
        np.testing.assert_allclose(z / H, 0.37, rtol=1e-12)


class TestSigmaRestartIdentity:
    """A resumed sigma run must reproduce the continuous one bit for bit
    (review of fix/sigma-step: carrying the previous step's end surface would
    have broken it — the end-of-step and next-start surfaces are now one
    function of time, supplied by the tracking layer)."""

    @pytest.mark.parametrize("integrator", ["rk2", "rk4"])
    @pytest.mark.parametrize("recon", ["linear", "pchip"])
    @pytest.mark.parametrize("dt", [150.0, 400.0])
    def test_resume_equals_continuous(self, tmp_path, integrator, recon, dt):
        import warnings as _w
        from kelp.core.config import NumericsConfig
        from kelp.core.mesh import find_nearest_nodes
        from kelp.core.particles import make_state
        from kelp.readers import extract_time_info
        from kelp.tracking import run_single_realization
        g, d0, c = 5e-5, 20.0, 1e-5
        lon, lat, tris, _ = _grid()
        depth = d0 + g * lat * METERS_PER_DEG_LAT
        times = np.arange(13) * 600.0
        eta = (1.0 + 0.5 * lat)[None, :] * np.sin(
            2 * np.pi * times / 3000.0)[:, None]
        H = depth[None, :] + eta
        wvel = c * np.broadcast_to(H[:, None, :], (13, SIG_U.size, 9)).copy()
        wvel[:, 0] = wvel[:, -1] = 0.0
        v_elem = np.full((13, tris.shape[0], SIG_U.size - 1), 0.4)
        p = _write(tmp_path / "rs.nc", times, eta, wvel, depth=depth,
                   v_elem=v_elem)
        reader = get_reader("fesomc")
        mesh = reader.load_mesh(SimpleNamespace(meshpath=str(p)))
        ds = reader.open_dataset(str(p))
        ft, fdt = extract_time_info(ds)
        num = NumericsConfig(dt=dt, integrator=integrator, output_freq=1,
                             temporal_reconstruction=recon)

        def run(st, start_idx, n, t_start, step0):
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                return run_single_realization(
                    mesh, ds, reader, st, _mode(), num,
                    start_time_idx=start_idx, n_steps=n, n_time_total=13,
                    temporal_interp=True, file_times=ft, file_dt=fdt,
                    sim_time_start=t_start, start_step_i=step0).to_arrays()

        st0 = make_state(np.array([5.3, 4.6]), np.array([-0.37, 0.2]),
                         np.array([6.0, 12.0]))
        st0 = st0.replace(nearest_node=find_nearest_nodes(mesh, st0.lon,
                                                          st0.lat))
        full = run(st0, 0, 12, 0.0, 0)
        half = run(st0, 0, 6, 0.0, 0)
        mid = make_state(half["lon"][-1], half["lat"][-1], half["z"][-1])
        mid = mid.replace(nearest_node=find_nearest_nodes(mesh, mid.lon,
                                                          mid.lat))
        t6 = 6 * dt
        rest = run(mid, int(t6 // 600.0), 6, t6, 6)
        ds.close()
        for k in ("lon", "lat", "z"):
            np.testing.assert_array_equal(rest[k][-1], full[k][-1], err_msg=k)


class TestMaxLevelsWithoutSigmaInTheDataFile:
    def test_split_mesh_and_data_files(self, tmp_path):
        """Review round 2 (both): with the sigma table only in the mesh file,
        the conversion took the max_levels-trimmed mesh.sigma and crashed on
        broadcasting against the full W."""
        prof = 1e-4 * np.sin(np.pi * (1.0 - SIG_U))
        w = np.broadcast_to(prof[None, :, None], (2, SIG_U.size, 9)).copy()
        full = _write(tmp_path / "full.nc", [0.0, 100.0], np.zeros((2, 9)), w)
        with xr.open_dataset(full) as ds:
            ds.drop_vars("sigma_lev").to_netcdf(tmp_path / "data.nc")
        reader = get_reader("fesomc")
        mesh = reader.load_mesh(SimpleNamespace(meshpath=str(full)))
        mesh.sigma = mesh.sigma[:4]                  # what run_tracking does
        ds = reader.open_dataset(str(tmp_path / "data.nc"))
        vel = reader.read_velocities(ds, 0, mesh, _mode(max_levels=4))
        ds.close()
        assert vel.w.shape == (4, 9)
        np.testing.assert_allclose(vel.w[:, 4], -prof[:4], atol=1e-18)


class TestPrefetchWindow:
    def test_prefetch_window_matches_full_dataset(self, tmp_path):
        """A prefetch window hands the reader a slice of the time axis: the
        conversion must not depend on the neighbouring snapshots."""
        from kelp.tracking import _VelCache
        times = np.arange(9) * 100.0
        eta = np.tile((0.01 * (times / 100.0) ** 2)[:, None], (1, 9))
        p = _write(tmp_path / "tide.nc", times, eta, np.zeros((9, SIG_U.size, 9)))
        reader = get_reader("fesomc")
        mesh = reader.load_mesh(SimpleNamespace(meshpath=str(p)))
        ds = reader.open_dataset(str(p))
        mode = _mode()
        full = [reader.read_velocities(ds, i, mesh, mode).w for i in range(9)]
        for pf in (2, 3, 4):
            cache = _VelCache(prefetch_size=pf, n_time=9)
            for i in range(9):
                np.testing.assert_array_equal(
                    cache.get(i, reader, ds, mesh, mode).w, full[i],
                    err_msg=f"prefetch {pf}, idx {i}")
        ds.close()


class TestMovingColumn:
    @pytest.mark.parametrize("sig", [SIG_U, SIG_S], ids=["uniform", "stretched"])
    def test_particle_stays_on_its_sigma_level(self, tmp_path, sig):
        """Manufactured solution (review M02): H = H0 e^{at}, the flow moves
        with the sigma surfaces (Wvel = 0).  The surface-relative depth obeys
        dq/dt = a q, so q(t) = q0 e^{at}."""
        from kelp.core.config import NumericsConfig
        from kelp.core.mesh import find_nearest_nodes
        from kelp.core.particles import make_state
        from kelp.readers import extract_time_info
        from kelp.tracking import run_single_realization
        a, d, H0, dt, n = 1e-4, 10.0, 10.0, 100.0, 10
        times = np.arange(n + 3) * dt
        H = H0 * np.exp(a * times)
        eta = np.tile((H - d)[:, None], (1, 9))
        p = _write(tmp_path / "column.nc", times, eta,
                   np.zeros((len(times), sig.size, 9)), sig=sig, depth=d)
        reader = get_reader("fesomc")
        mesh = reader.load_mesh(SimpleNamespace(meshpath=str(p)))
        ds = reader.open_dataset(str(p))
        file_times, file_dt = extract_time_info(ds)
        q0 = 0.37 * H0
        state = make_state(np.array([5.0]), np.array([0.0]), np.array([q0]))
        state = state.replace(
            nearest_node=find_nearest_nodes(mesh, state.lon, state.lat))
        rec = run_single_realization(
            mesh, ds, reader, state, _mode(),
            NumericsConfig(dt=dt, integrator="rk4", output_freq=n),
            start_time_idx=0, n_steps=n, n_time_total=len(times),
            temporal_interp=True, file_times=file_times, file_dt=file_dt)
        ds.close()
        z_end = rec.to_arrays()["z"][0, -1]
        assert z_end == pytest.approx(q0 * np.exp(a * n * dt), abs=2e-3)


def test_chain_driver_outputs_carry_the_provenance(tmp_path):
    """Review 2026-09-23: outputs written through run_single_realization
    with a caller's writer (the resumable-chain drivers) had no kelp_version
    and no w_contract, although the docs read 'no w_contract = old engine'."""
    import zarr
    from kelp.core.config import NumericsConfig
    from kelp.core.mesh import find_nearest_nodes
    from kelp.core.particles import make_state
    from kelp.readers import extract_time_info
    from kelp.tracking import run_single_realization
    from kelp.writers import TrajectoryWriter
    times = np.arange(4) * 100.0
    p = _write(tmp_path / "prov.nc", times, np.zeros((4, 9)),
               np.zeros((4, SIG_U.size, 9)))
    reader = get_reader("fesomc")
    mesh = reader.load_mesh(SimpleNamespace(meshpath=str(p)))
    ds = reader.open_dataset(str(p))
    ft, fdt = extract_time_info(ds)
    st = make_state(np.array([5.0]), np.array([0.0]), np.array([3.0]))
    st = st.replace(nearest_node=find_nearest_nodes(mesh, st.lon, st.lat))
    out = str(tmp_path / "seg.zarr")
    w = TrajectoryWriter(out, 1)
    w.open()
    run_single_realization(mesh, ds, reader, st, _mode(),
                           NumericsConfig(dt=100.0, integrator="rk2"),
                           start_time_idx=0, n_steps=2, n_time_total=4,
                           writer=w, temporal_interp=True, file_times=ft,
                           file_dt=fdt)
    w.close()
    ds.close()
    attrs = dict(zarr.open_group(out, mode="r").attrs)
    assert attrs["w_contract"] == "reader-converted-2026-09-23"
    import kelp
    assert str(attrs["kelp_version"]).startswith(kelp.__version__)
    assert attrs["temporal_interp"] == 1
    # review round 2: the W-defining settings too, so the shard
    # concatenator refuses shards that differ in them
    assert attrs["w_interp"] == "leastsq"
    assert attrs["w_positive"] == "auto"
