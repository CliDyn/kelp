"""Tests for PCHIP temporal reconstruction (numerics/temporal_reconstruct.py)."""
import numpy as np
import pytest

from kelp.numerics.temporal_reconstruct import _pchip_blend4

T = (0.0, 10.0, 20.0, 30.0)  # uniform; query inside the central interval [10, 20]


def test_pchip_no_overshoot_on_step():
    # step up then flat: a monotone scheme must NOT overshoot above 1.0
    f0, f1, f2, f3 = (np.array([0.0]), np.array([0.0]),
                      np.array([1.0]), np.array([1.0]))
    for tq in np.linspace(10, 20, 11):
        v = _pchip_blend4(f0, f1, f2, f3, *T, tq)
        assert -1e-12 <= v[0] <= 1.0 + 1e-12


def test_pchip_exact_on_endpoints():
    f = [np.array([2.0]), np.array([5.0]), np.array([4.0]), np.array([9.0])]
    np.testing.assert_allclose(_pchip_blend4(*f, *T, 10.0), [5.0], atol=1e-12)
    np.testing.assert_allclose(_pchip_blend4(*f, *T, 20.0), [4.0], atol=1e-12)


def test_pchip_linear_field_is_linear():
    # samples on a straight line in time -> reconstruction is that line
    line = lambda t: 3.0 + 0.5 * t
    f = [np.array([line(t)]) for t in T]
    for tq in (12.0, 15.0, 18.0):
        np.testing.assert_allclose(
            _pchip_blend4(*f, *T, tq), [line(tq)], atol=1e-10)


def test_pchip_nonuniform_dt():
    t = (0.0, 10.0, 35.0, 50.0)  # non-uniform spacing
    f = [np.array([1.0]), np.array([2.0]), np.array([2.0]), np.array([5.0])]
    v = _pchip_blend4(*f, *t, 20.0)
    # flat secant in [t1, t2] -> monotone scheme stays at 2.0 throughout
    assert 2.0 - 1e-9 <= v[0] <= 2.0 + 1e-9


def test_pchip_vectorized_over_elements():
    # operates element-wise across an array field, not just scalars
    f0 = np.array([0.0, 10.0]); f1 = np.array([0.0, 10.0])
    f2 = np.array([1.0, 12.0]); f3 = np.array([1.0, 12.0])
    v = _pchip_blend4(f0, f1, f2, f3, *T, 15.0)
    assert v.shape == (2,)
    assert 0.0 <= v[0] <= 1.0 and 10.0 <= v[1] <= 12.0


def _vf(scale, layout="element", with_w=False, scalars=None):
    from kelp.core.io import VelocityField
    return VelocityField(
        u=np.array([scale]), v=np.array([2 * scale]), eta=np.array([0.1 * scale]),
        w=(np.array([0.5 * scale]) if with_w else None),
        uv_layout=layout, scalars=scalars)


def test_reconstruct_pchip_builds_field_all_components():
    from kelp.numerics.temporal_reconstruct import reconstruct_pchip
    snaps = [_vf(s, with_w=True, scalars={"temp": np.array([s])})
             for s in (1.0, 2.0, 1.5, 3.0)]
    out = reconstruct_pchip(snaps, list(T), 15.0)
    assert out.u.shape == (1,) and out.w is not None
    assert out.scalars is not None and "temp" in out.scalars
    # at t1 (=10) it must equal snapshot b exactly
    b = reconstruct_pchip(snaps, list(T), 10.0)
    np.testing.assert_allclose(b.u, snaps[1].u, atol=1e-12)
    np.testing.assert_allclose(b.scalars["temp"], snaps[1].scalars["temp"], atol=1e-12)


def test_make_stage_vels_recon_matches_reconstruct_at_stage_times():
    from kelp.numerics.temporal_reconstruct import (
        make_stage_vels_recon, reconstruct_pchip)
    times = [0.0, 3600.0, 7200.0, 10800.0]
    snaps = [_vf(s) for s in (1.0, 2.0, 1.5, 3.0)]
    sim_time, dt = 3600.0, 1800.0  # start at t1, half a bracket
    c = (0.0, 0.5, 0.5, 1.0)
    stages = make_stage_vels_recon(snaps, times, sim_time, dt, +1.0, c)
    assert len(stages) == 4
    for ci, st in zip(c, stages):
        exp = reconstruct_pchip(snaps, times, sim_time + 1.0 * ci * dt)
        np.testing.assert_allclose(st.u, exp.u, atol=1e-12)
        np.testing.assert_allclose(st.v, exp.v, atol=1e-12)


def test_reconstruct_pchip_validates_times(tmp_path=None):
    from kelp.numerics.temporal_reconstruct import reconstruct_pchip
    snaps = [_vf(s) for s in (1.0, 2.0, 1.5, 3.0)]
    # non-strictly-increasing times -> raise (not a silent div-by-zero/NaN)
    with pytest.raises(ValueError, match="strictly increasing"):
        reconstruct_pchip(snaps, [0.0, 10.0, 10.0, 30.0], 10.0)
    # query outside the central interval [t1, t2] -> raise
    with pytest.raises(ValueError, match="central"):
        reconstruct_pchip(snaps, [0.0, 10.0, 20.0, 30.0], 25.0)


def test_make_stage_vels_recon_backward():
    from kelp.numerics.temporal_reconstruct import (
        make_stage_vels_recon, reconstruct_pchip)
    times = [0.0, 3600.0, 7200.0, 10800.0]
    snaps = [_vf(s) for s in (1.0, 2.0, 1.5, 3.0)]
    sim_time, dt = 7200.0, 1800.0  # start at t2, step backward into [t1, t2]
    c = (0.0, 0.5)
    stages = make_stage_vels_recon(snaps, times, sim_time, dt, -1.0, c)
    for ci, st in zip(c, stages):
        exp = reconstruct_pchip(snaps, times, sim_time - 1.0 * ci * dt)
        np.testing.assert_allclose(st.u, exp.u, atol=1e-12)


# --- integrator recon= plumbing (Task 4) -------------------------------------
from tests.test_integrators_time_aware import (  # noqa: E402  reuse fixtures
    _make_mesh, _make_vel, LON0, LAT0)
from kelp.core.particles import make_state
from kelp.core.geometry import METERS_PER_DEG_LAT, DEG2RAD


@pytest.mark.parametrize("step_name", ["rk2", "rk4"])
def test_rk_step_recon_uses_pchip_midpoint(step_name):
    # Spatially-uniform time-varying field: an RK step with recon= must
    # displace by the PCHIP-reconstructed velocity (a constant field
    # interpolates to its own value, so the displacement is analytic).
    from kelp.numerics.integrators import rk2_step, rk4_step
    from kelp.numerics.temporal_reconstruct import (
        make_stage_vels_recon, reconstruct_pchip)
    mesh = _make_mesh()
    times = [0.0, 100.0, 200.0, 300.0]
    snaps = [_make_vel(mesh, t) for t in times]
    sim_time, dt = 100.0, 50.0          # stays inside central interval [100,200]
    state = make_state(np.array([LON0]), np.array([LAT0]))
    step = rk2_step if step_name == "rk2" else rk4_step
    c = (0.0, 0.5) if step_name == "rk2" else (0.0, 0.5, 0.5, 1.0)
    out = step(state, snaps[1], mesh, dt, 1.0,
               recon=(snaps, times, sim_time, 1.0), uv_interp="leastsq")
    # reproduce the RK stage average analytically from the same recon fields
    stage_vels = make_stage_vels_recon(snaps, times, sim_time, dt, 1.0, c)
    if step_name == "rk2":
        u_eff = stage_vels[1].u[0]                       # midpoint
    else:
        u_eff = (stage_vels[0].u[0] + 2 * stage_vels[1].u[0]
                 + 2 * stage_vels[2].u[0] + stage_vels[3].u[0]) / 6.0
    z0 = np.cos(LAT0 * DEG2RAD) * METERS_PER_DEG_LAT
    expected_lon = LON0 + dt * u_eff / z0
    np.testing.assert_allclose(out.lon[0], expected_lon, rtol=1e-7, atol=1e-9)


# --- tracking wiring (Task 5) ------------------------------------------------
def _make_fesomc_timevarying(tmp_path, n_time, dt_seconds, u_of_t):
    """FESOM-C NetCDF with spatially-uniform, time-VARYING u(t), v=0."""
    import pandas as pd
    import xarray as xr
    xs = np.array([4.0, 5.0, 6.0]); ys = np.array([4.0, 5.0, 6.0])
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel().astype(np.float64); lat = gy.ravel().astype(np.float64)
    depth = np.full(9, 10.0, dtype=np.float64)
    elems = []
    for j in range(2):
        for i in range(2):
            n0 = j * 3 + i
            elems.append([n0 + 1, n0 + 2, n0 + 4, n0 + 1])
            elems.append([n0 + 2, n0 + 5, n0 + 4, n0 + 2])
    nv = np.array(elems, dtype=np.int32)
    n_elem = nv.shape[0]
    elem0 = nv[:, :3] - 1
    lon_c = np.array([lon[e].mean() for e in elem0], dtype=np.float64)
    lat_c = np.array([lat[e].mean() for e in elem0], dtype=np.float64)
    sigma_lev = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    uvals = np.array([u_of_t(k * dt_seconds) for k in range(n_time)])
    u2d = np.repeat(uvals[:, None], n_elem, axis=1).astype(np.float64)
    v2d = np.zeros((n_time, n_elem), dtype=np.float64)
    eta = np.zeros((n_time, 9), dtype=np.float64)
    times = pd.date_range("2020-01-01", periods=n_time, freq=f"{int(dt_seconds)}s")
    ds = xr.Dataset({
        "lon": (["node"], lon), "lat": (["node"], lat), "depth": (["node"], depth),
        "nv": (["nele", "four"], nv),
        "lon_elem": (["nele"], lon_c), "lat_elem": (["nele"], lat_c),
        "sigma_lev": (["sigma"], sigma_lev),
        "u2d": (["time", "nele"], u2d), "v2d": (["time", "nele"], v2d),
        "eta": (["time", "node"], eta),
    }, coords={"time": times})
    p = tmp_path / "fesomc_tv.nc"
    ds.to_netcdf(p)
    return p


def test_tracking_pchip_runs_and_differs_from_linear(tmp_path):
    from kelp.readers import get_reader, extract_time_info
    from kelp.tracking import run_single_realization
    from kelp.core.config import ModeConfig, NumericsConfig
    # strongly curved-in-time velocity so PCHIP != linear within a bracket
    u_of_t = lambda t: 1.0 + 0.8 * np.sin(2 * np.pi * t / (8 * 3600.0))
    p = _make_fesomc_timevarying(tmp_path, n_time=8, dt_seconds=3600.0, u_of_t=u_of_t)
    reader = get_reader("fesomc")
    mesh = reader.load_mesh(type("", (), {"meshpath": str(p)})())
    ds = reader.open_dataset(str(p))
    file_times, file_dt = extract_time_info(ds)
    mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)
    init = lambda: make_state(np.array([5.0]), np.array([5.0]))

    def run(recon):
        num = NumericsConfig(dt=file_dt, integrator="rk2", output_freq=1,
                             temporal_reconstruction=recon)
        return run_single_realization(
            mesh, ds, reader, init(), mode, num,
            start_time_idx=1, n_steps=4,  # interior: 4-stencil available
            temporal_interp=True, file_times=file_times, file_dt=file_dt)

    rec_lin = run("linear")
    rec_pchip = run("pchip")
    ds.close()
    # both finite and moved
    assert np.isfinite(rec_pchip.lons[-1][0])
    assert rec_pchip.lons[-1][0] != 5.0
    # PCHIP captures time-curvature the linear blend misses -> different path
    assert abs(rec_pchip.lons[-1][0] - rec_lin.lons[-1][0]) > 1e-7


def test_tracking_pchip_edge_falls_back_to_linear(tmp_path):
    # Starting at the very first interval forces the edge (no idx_a-1) -> the
    # run must complete using the linear fallback (and not crash).
    import warnings
    from kelp.readers import get_reader, extract_time_info
    from kelp.tracking import run_single_realization
    from kelp.core.config import ModeConfig, NumericsConfig
    u_of_t = lambda t: 1.0 + 0.5 * np.sin(2 * np.pi * t / (5 * 3600.0))
    p = _make_fesomc_timevarying(tmp_path, n_time=5, dt_seconds=3600.0, u_of_t=u_of_t)
    reader = get_reader("fesomc")
    mesh = reader.load_mesh(type("", (), {"meshpath": str(p)})())
    ds = reader.open_dataset(str(p))
    file_times, file_dt = extract_time_info(ds)
    mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)
    num = NumericsConfig(dt=file_dt, integrator="rk2", output_freq=1,
                         temporal_reconstruction="pchip")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = run_single_realization(
            mesh, ds, reader, make_state(np.array([5.0]), np.array([5.0])),
            mode, num, start_time_idx=0, n_steps=2,
            temporal_interp=True, file_times=file_times, file_dt=file_dt)
    ds.close()
    assert np.isfinite(rec.lons[-1][0])


# --- convergence / 3D / round-trip (Task 6) ----------------------------------
def _uniform_snaps(mesh, times, U_of_t):
    from kelp.core.io import VelocityField
    ne, nn = mesh.n_elements, mesh.n_nodes
    return [VelocityField(u=np.full(ne, U_of_t(t)), v=np.zeros(ne),
                          eta=np.zeros(nn)) for t in times]


def _integrate_uniform(step, snaps, times, mode, K):
    """Step a single particle through interior brackets n=1..K, dt=file_dt.

    Uniform-in-space field: the leastsq interp returns the field value exactly,
    so the displacement is the RK time-quadrature of U(t) — isolating temporal
    error.  Returns final lon (start lon = 0).
    """
    state = make_state(np.array([0.0]), np.array([0.0]))
    fd = float(times[1] - times[0])
    for n in range(1, K + 1):
        if mode == "pchip":
            recon = (snaps[n - 1:n + 3],
                     [float(times[i]) for i in range(n - 1, n + 3)],
                     float(times[n]), 1.0)
            state = step(state, snaps[n], _M, fd, 1.0,
                         recon=recon, uv_interp="leastsq")
        else:
            state = step(state, snaps[n], _M, fd, 1.0,
                         vel_next=snaps[n + 1], dt_to_next=fd, uv_interp="leastsq")
    return state.lon[0]


_M = _make_mesh()  # equator patch reused across the uniform-field tests


def test_pchip_reconstruction_beats_linear():
    # The core lever: between coarse snapshots of a smooth curved field, the
    # PCHIP cubic reconstructs the true field FAR more accurately than the
    # linear chord.  Dense check inside the central bracket.
    fd = 3600.0
    w = 2 * np.pi / (20 * fd)           # gentle ~18 deg/bracket, away from extrema
    U = lambda t: np.cos(w * t)
    times = np.arange(6) * fd
    f = [np.array([U(t)]) for t in times]
    max_lin = max_pchip = 0.0
    for s in np.linspace(0.0, 1.0, 21):
        tq = times[2] + s * fd
        true = U(tq)
        lin = (1.0 - s) * f[2][0] + s * f[3][0]
        pc = _pchip_blend4(f[1], f[2], f[3], f[4],
                           times[1], times[2], times[3], times[4], tq)[0]
        max_lin = max(max_lin, abs(lin - true))
        max_pchip = max(max_pchip, abs(pc - true))
    assert max_pchip < max_lin / 5.0   # higher-order reconstruction


def test_pchip_lowers_trajectory_error_floor_vs_linear():
    # On a coarsely-sampled curved field, the trajectory error with PCHIP is
    # well below the linear floor; and with LINEAR, RK2 and RK4 coincide
    # (the reconstruction floor caps the order — the motivation for PCHIP).
    from kelp.numerics.integrators import rk2_step, rk4_step
    U0, fd = 0.5, 3600.0
    N, K = 10, 4                        # span [t1, t5] = 1/5 period (no cancel)
    w = 2 * np.pi / (20 * fd)
    U_of_t = lambda t: U0 * np.cos(w * t)
    times = np.arange(N) * fd
    snaps = _uniform_snaps(_M, times, U_of_t)
    z0 = np.cos(LAT0 * DEG2RAD) * METERS_PER_DEG_LAT
    exact = (U0 / w) * (np.sin(w * times[K + 1]) - np.sin(w * times[1])) / z0

    err = {}
    for name, step in (("rk2", rk2_step), ("rk4", rk4_step)):
        for mode in ("linear", "pchip"):
            lon = _integrate_uniform(step, snaps, times, mode, K)
            err[(name, mode)] = abs(lon - exact)

    # linear reconstruction is a straight line in each bracket -> RK2 and RK4
    # integrate it IDENTICALLY: the integrator order collapses onto the
    # reconstruction floor (the motivation for PCHIP).
    assert err[("rk2", "linear")] > 0
    np.testing.assert_allclose(err[("rk4", "linear")], err[("rk2", "linear")],
                               rtol=1e-6)
    # PCHIP gives RK4 a cubic, which Simpson integrates exactly -> RK4 drops
    # well below the linear floor AND below RK2: the orders SEPARATE under
    # PCHIP, where under linear they coincided. (RK2/midpoint benefits less,
    # since its own quadrature error caps the gain.)
    assert err[("rk4", "pchip")] < err[("rk4", "linear")] / 3.0
    assert err[("rk4", "pchip")] < err[("rk2", "pchip")] / 1.5


def test_pchip_equals_linear_for_time_linear_field():
    # PCHIP through collinear-in-time samples reproduces the straight line, so
    # a temporally-linear field gives identical trajectories to the linear path.
    from kelp.numerics.integrators import rk4_step
    fd, N, K = 3600.0, 8, 4
    U_of_t = lambda t: 0.2 + 1e-5 * t          # exactly linear in time
    times = np.arange(N) * fd
    snaps = _uniform_snaps(_M, times, U_of_t)
    lon_lin = _integrate_uniform(rk4_step, snaps, times, "linear", K)
    lon_pchip = _integrate_uniform(rk4_step, snaps, times, "pchip", K)
    np.testing.assert_allclose(lon_pchip, lon_lin, rtol=1e-10, atol=1e-12)


def test_pchip_recon_step_round_trip_single_bracket():
    # one interior bracket forward then backward with PCHIP returns to start
    from kelp.numerics.integrators import rk4_step
    fd = 3600.0
    times = np.arange(6) * fd
    U_of_t = lambda t: 0.4 * np.cos(2 * np.pi * t / (6 * fd))
    snaps = _uniform_snaps(_M, times, U_of_t)
    n = 2  # central bracket [t2, t3], stencil snaps[1..4]
    stencil = (snaps[n - 1:n + 3], [float(times[i]) for i in range(n - 1, n + 3)])
    s0 = make_state(np.array([0.0]), np.array([0.0]))
    fwd = rk4_step(s0, snaps[n], _M, fd, 1.0,
                   recon=(*stencil, float(times[n]), 1.0), uv_interp="leastsq")
    back = rk4_step(fwd, snaps[n + 1], _M, fd, -1.0,
                    recon=(*stencil, float(times[n + 1]), -1.0), uv_interp="leastsq")
    np.testing.assert_allclose(back.lon[0], 0.0, atol=1e-9)


def _uniform_snaps_3d(mesh, times, U_of_t, W_of_t):
    """3D snapshots: spatially- and depth-uniform u(t) (mid-layers) and
    w(t) (interfaces), so the spatial interpolation is exact and the
    trajectory is a pure time-quadrature -> isolates temporal error."""
    from kelp.core.io import VelocityField
    K = mesh.sigma.size                 # 5 interfaces -> 4 mid-layers
    ne, nn = mesh.n_elements, mesh.n_nodes
    out = []
    for t in times:
        u = np.full((K - 1, ne), U_of_t(t))
        v = np.zeros((K - 1, ne))
        w = np.full((K, nn), W_of_t(t))
        out.append(VelocityField(u=u, v=v, eta=np.zeros(nn), w=w))
    return out


def _integrate_uniform_3d(step, snaps, times, mode, K, z_start):
    from kelp.core.particles import make_state as mk
    state = mk(np.array([0.0]), np.array([0.0]), np.array([z_start]))
    fd = float(times[1] - times[0])
    for n in range(1, K + 1):
        if mode == "pchip":
            recon = (snaps[n - 1:n + 3],
                     [float(times[i]) for i in range(n - 1, n + 3)],
                     float(times[n]), 1.0)
            state = step(state, snaps[n], _M, fd, 1.0, recon=recon,
                         uv_interp="leastsq", w_interp="leastsq")
        else:
            state = step(state, snaps[n], _M, fd, 1.0, vel_next=snaps[n + 1],
                         dt_to_next=fd, uv_interp="leastsq", w_interp="leastsq")
    return state.lon[0], state.z[0]


def test_pchip_3d_vertical_strict_convergence():
    # STRICT 3D: the vertical (w-driven) trajectory must converge against the
    # closed-form integral, with the same order story as the horizontal —
    # under linear, RK2==RK4 (vertical order collapses onto the reconstruction
    # floor); under PCHIP, RK4 separates from RK2 and beats the linear floor.
    from kelp.numerics.integrators import rk2_step, rk4_step
    U0, W0, fd = 0.5, 0.005, 3600.0
    N, K = 10, 4
    wang = 2 * np.pi / (20 * fd)         # gentle curvature, away from extrema
    U_of_t = lambda t: U0 * np.cos(wang * t)
    W_of_t = lambda t: W0 * np.cos(wang * t)
    times = np.arange(N) * fd
    snaps = _uniform_snaps_3d(_M, times, U_of_t, W_of_t)
    z_start = 100.0                      # interior: stays within (0, 200) m
    # z is positive-down and updated z += w*dt -> exact = z0 + integral(w)
    exact_z = z_start + (W0 / wang) * (np.sin(wang * times[K + 1])
                                       - np.sin(wang * times[1]))

    zerr = {}
    for name, step in (("rk2", rk2_step), ("rk4", rk4_step)):
        for mode in ("linear", "pchip"):
            _, zf = _integrate_uniform_3d(step, snaps, times, mode, K, z_start)
            zerr[(name, mode)] = abs(zf - exact_z)

    # sanity: the particle stayed interior (no surface/bottom clamp masking it)
    assert 0.0 < exact_z < 200.0
    # vertical order collapses under linear
    assert zerr[("rk2", "linear")] > 0
    np.testing.assert_allclose(zerr[("rk4", "linear")], zerr[("rk2", "linear")],
                               rtol=1e-6)
    # PCHIP: RK4 separates from RK2 in the VERTICAL and beats the linear floor
    assert zerr[("rk4", "pchip")] < zerr[("rk4", "linear")] / 3.0
    assert zerr[("rk4", "pchip")] < zerr[("rk2", "pchip")] / 1.5


def test_pchip_3d_recon_step_runs_and_moves_z():
    # 3D recon path: u/v at mid-layers (K-1, ne), w at interfaces (K, nn).
    from kelp.core.io import VelocityField
    from kelp.numerics.integrators import rk4_step
    from kelp.core.particles import make_state as mk
    K = _M.sigma.size            # 5 interfaces -> 4 mid-layers
    ne, nn = _M.n_elements, _M.n_nodes
    fd = 3600.0
    times = np.arange(6) * fd

    def vf3d(t):
        u = np.full((K - 1, ne), 0.2 * np.cos(2 * np.pi * t / (6 * fd)))
        v = np.zeros((K - 1, ne))
        w = np.full((K, nn), 1e-3 * np.cos(2 * np.pi * t / (6 * fd)))  # m/s downward
        return VelocityField(u=u, v=v, eta=np.zeros(nn), w=w)

    snaps = [vf3d(t) for t in times]
    n = 2
    recon = (snaps[n - 1:n + 3], [float(times[i]) for i in range(n - 1, n + 3)],
             float(times[n]), 1.0)
    st = mk(np.array([0.0]), np.array([0.0]), np.array([50.0]))
    out = rk4_step(st, snaps[n], _M, fd, 1.0, recon=recon,
                   uv_interp="leastsq", w_interp="leastsq")
    assert np.isfinite(out.lon[0]) and np.isfinite(out.z[0])
    assert out.z[0] != 50.0   # vertical velocity moved the particle


# --- coupled non-stationary convergence (approach B) -------------------------
_LONC = 5.0    # shear centre (the fesomc test mesh spans lon 4..6, lat 4..6)
_LAT0 = 5.0


def _make_fesomc_coupled(tmp_path, n_time, dt_seconds, a_of_t):
    """FESOM-C NetCDF, spatially-LINEAR time-varying shear u = a(t)*(lon-_LONC)*z0,
    v=0.  Linear in space => leastsq interpolation is EXACT, and the trajectory
    has the closed form  lon(T) = _LONC + (lon0-_LONC)*exp(integral a dt)
    (since d(lon)/dt = U/z0 = a(t)*(lon-_LONC)).  This is genuinely COUPLED
    (u depends on the moving position) yet analytically known."""
    import pandas as pd
    import xarray as xr
    z0 = np.cos(_LAT0 * DEG2RAD) * METERS_PER_DEG_LAT
    xs = np.array([4.0, 5.0, 6.0]); ys = np.array([4.0, 5.0, 6.0])
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel().astype(np.float64); lat = gy.ravel().astype(np.float64)
    depth = np.full(9, 10.0, dtype=np.float64)
    elems = []
    for j in range(2):
        for i in range(2):
            n0 = j * 3 + i
            elems.append([n0 + 1, n0 + 2, n0 + 4, n0 + 1])
            elems.append([n0 + 2, n0 + 5, n0 + 4, n0 + 2])
    nv = np.array(elems, dtype=np.int32)
    elem0 = nv[:, :3] - 1
    lon_c = np.array([lon[e].mean() for e in elem0], dtype=np.float64)
    lat_c = np.array([lat[e].mean() for e in elem0], dtype=np.float64)
    n_elem = nv.shape[0]
    sigma_lev = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    u2d = np.empty((n_time, n_elem), dtype=np.float64)
    for k in range(n_time):
        u2d[k, :] = a_of_t(k * dt_seconds) * (lon_c - _LONC) * z0
    v2d = np.zeros((n_time, n_elem), dtype=np.float64)
    eta = np.zeros((n_time, 9), dtype=np.float64)
    times = pd.date_range("2020-01-01", periods=n_time, freq=f"{int(dt_seconds)}s")
    ds = xr.Dataset({
        "lon": (["node"], lon), "lat": (["node"], lat), "depth": (["node"], depth),
        "nv": (["nele", "four"], nv),
        "lon_elem": (["nele"], lon_c), "lat_elem": (["nele"], lat_c),
        "sigma_lev": (["sigma"], sigma_lev),
        "u2d": (["time", "nele"], u2d), "v2d": (["time", "nele"], v2d),
        "eta": (["time", "node"], eta),
    }, coords={"time": times})
    p = tmp_path / "fesomc_coupled.nc"
    ds.to_netcdf(p)
    return p


def test_pchip_coupled_nonstationary_floor_below_linear(tmp_path):
    # COUPLED + non-stationary: PCHIP must bring the RK4 trajectory closer to
    # the exact coupled solution than linear does. (For a coupled field the
    # RK order also helps the spatial part, so the claim is "lower temporal
    # floor", not the RK2==RK4 collapse of the uniform case.)
    from kelp.readers import get_reader, extract_time_info
    from kelp.tracking import run_single_realization
    from kelp.core.config import ModeConfig, NumericsConfig
    fd = 3600.0
    wang = 2 * np.pi / (20 * fd)
    a0 = 0.5 * wang                      # integral a over the run stays ~O(0.25)
    a_of_t = lambda t: a0 * np.cos(wang * t)
    n_time = 10
    p = _make_fesomc_coupled(tmp_path, n_time, fd, a_of_t)
    reader = get_reader("fesomc")
    mesh = reader.load_mesh(type("", (), {"meshpath": str(p)})())
    ds = reader.open_dataset(str(p))
    file_times, file_dt = extract_time_info(ds)
    mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)
    lon0 = 5.05
    start_idx, n_steps = 1, 6            # interior brackets [t1..t7]
    t1, t7 = float(file_times[start_idx]), float(file_times[start_idx + n_steps])
    exact_lon = _LONC + (lon0 - _LONC) * np.exp(
        (a0 / wang) * (np.sin(wang * t7) - np.sin(wang * t1)))

    def run(recon):
        num = NumericsConfig(dt=file_dt, integrator="rk4", output_freq=1,
                             temporal_reconstruction=recon)
        rec = run_single_realization(
            mesh, ds, reader, make_state(np.array([lon0]), np.array([_LAT0])),
            mode, num, start_time_idx=start_idx, n_steps=n_steps,
            temporal_interp=True, file_times=file_times, file_dt=file_dt)
        return abs(rec.lons[-1][0] - exact_lon)

    err_lin = run("linear")
    err_pchip = run("pchip")
    ds.close()
    assert err_lin > 0
    assert err_pchip < err_lin / 2.0   # PCHIP lowers the coupled-trajectory error


def test_pchip_3d_round_trip_forward_backward():
    # 3D BACKWARD propagation: forward then backward with PCHIP must return to
    # the start in BOTH lon and z (exercises rk4_step direction=-1 + recon in
    # 3D; RK4 stage symmetry makes each bracket's backward step undo the
    # forward one for a space-uniform field).
    from kelp.numerics.integrators import rk4_step
    from kelp.core.particles import make_state as mk
    U0, W0, fd = 0.5, 0.005, 3600.0
    N, K = 10, 4
    wang = 2 * np.pi / (20 * fd)
    U_of_t = lambda t: U0 * np.cos(wang * t)
    W_of_t = lambda t: W0 * np.cos(wang * t)
    times = np.arange(N) * fd
    snaps = _uniform_snaps_3d(_M, times, U_of_t, W_of_t)

    def stencil(n):  # bracket [t_n, t_{n+1}] -> 4-point snaps[n-1 .. n+2]
        return (snaps[n - 1:n + 3], [float(times[i]) for i in range(n - 1, n + 3)])

    state = mk(np.array([0.0]), np.array([0.0]), np.array([100.0]))
    for n in range(1, K + 1):                       # forward t1 -> t_{K+1}
        s, t = stencil(n)
        state = rk4_step(state, snaps[n], _M, fd, 1.0,
                         recon=(s, t, float(times[n]), 1.0),
                         uv_interp="leastsq", w_interp="leastsq")
    for n in range(K, 0, -1):                        # backward t_{K+1} -> t1
        s, t = stencil(n)
        state = rk4_step(state, snaps[n + 1], _M, fd, -1.0,
                         recon=(s, t, float(times[n + 1]), -1.0),
                         uv_interp="leastsq", w_interp="leastsq")
    np.testing.assert_allclose(state.lon[0], 0.0, atol=1e-7)
    np.testing.assert_allclose(state.z[0], 100.0, atol=1e-6)


def test_pchip_general_nonlinear_coupled_floor():
    # FULLY-GENERAL B: spatially-NONLINEAR, time-dependent Taylor-Green-style
    # field (leastsq is NOT exact for it).  Ground truth = a tracked run with
    # the SAME mesh interpolation but dense time + tiny dt (so its temporal
    # error is negligible and its spatial-interp error matches the coarse runs
    # -> cancels in the comparison).  Claim: at coarse file_dt, RK4 with PCHIP
    # lands closer to that truth than RK4 with linear time blend.
    from kelp.numerics.integrators import rk4_step
    from kelp.core.io import VelocityField
    from kelp.core.particles import make_state as mk
    fd = 3600.0
    wang = 2 * np.pi / (20 * fd)         # gentle temporal curvature
    U0, kk = 3.0, 2 * np.pi / 0.6        # wavelength 0.6 deg, resolved by _M
    g = lambda t: np.cos(wang * t)
    xc, yc, nn = _M.elem_center_lon, _M.elem_center_lat, _M.n_nodes

    def field_vel(t):
        u = U0 * g(t) * np.sin(kk * xc) * np.cos(kk * yc)
        v = -U0 * g(t) * np.cos(kk * xc) * np.sin(kk * yc)
        return VelocityField(u=u, v=v, eta=np.zeros(nn))

    lon0 = lat0 = 0.05
    N = 10
    times = np.arange(N) * fd
    brackets = (2, 3, 4, 5, 6)           # interior; window [t2, t7]
    t_start, t_end = float(times[2]), float(times[brackets[-1] + 1])

    # reference: dense time (fd/16), tiny dt, fields at exact times
    M = 16
    dtr = fd / M
    ref = mk(np.array([lon0]), np.array([lat0]))
    tcur = t_start
    for _ in range(int(round((t_end - t_start) / dtr))):
        ref = rk4_step(ref, field_vel(tcur), _M, dtr, 1.0,
                       vel_next=field_vel(tcur + dtr), dt_to_next=dtr,
                       uv_interp="leastsq")
        tcur += dtr
    rlon, rlat = ref.lon[0], ref.lat[0]

    snaps = [field_vel(t) for t in times]

    def coarse(mode, S=8):
        # SUB-STEP within each coarse bracket (dt = fd/S) so the spatial-RK
        # error is small and the residual is the temporal-reconstruction floor.
        st = mk(np.array([lon0]), np.array([lat0]))
        dt = fd / S
        for n in brackets:
            sten = snaps[n - 1:n + 3]
            tt = [float(times[i]) for i in range(n - 1, n + 3)]
            tn, tn1 = float(times[n]), float(times[n + 1])
            for s in range(S):
                q = tn + s * dt
                if mode == "pchip":
                    st = rk4_step(st, snaps[n], _M, dt, 1.0,
                                  recon=(sten, tt, q, 1.0), uv_interp="leastsq")
                else:
                    a = (q - tn) / (tn1 - tn)
                    vq = (snaps[n] if a <= 0 else snaps[n + 1] if a >= 1
                          else snaps[n].lerp(snaps[n + 1], a))
                    st = rk4_step(st, vq, _M, dt, 1.0, vel_next=snaps[n + 1],
                                  dt_to_next=(tn1 - q), uv_interp="leastsq")
        return st.lon[0], st.lat[0]

    pl, pa = coarse("pchip")
    ll, la = coarse("linear")
    err_p = np.hypot(pl - rlon, pa - rlat)
    err_l = np.hypot(ll - rlon, la - rlat)
    # observed ratio ~5.7x; assert a comfortable >2x so it's robust across
    # machines/numba while still proving PCHIP lowers the coupled floor.
    assert abs(rlon) < 0.15 and abs(rlat) < 0.15   # particle stayed in-patch
    assert err_l > 0
    assert err_p < err_l / 2.0


# --- fused PCHIP kernels (2D) -------------------------------------------------
def test_pchip_scalar_helpers_match_numpy():
    # The @njit scalar twins used inside the fused kernels must reproduce the
    # vectorized NumPy reference (_fc_slope/_pchip_blend4) bit-for-bit, else the
    # fused path would diverge from the unfused reconstruct_pchip path.
    from kelp.numerics.temporal_reconstruct import (
        _fc_slope_scalar, _pchip_value_scalar, _fc_slope, _pchip_blend4)
    rng = np.random.default_rng(0)
    times = (0.0, 10.0, 23.0, 41.0)          # non-uniform spacing
    for _ in range(300):
        f = rng.uniform(-5.0, 5.0, 4)
        tq = rng.uniform(times[1], times[2])
        got = _pchip_value_scalar(float(f[0]), float(f[1]), float(f[2]),
                                  float(f[3]), *times, tq)
        exp = _pchip_blend4(*[np.array([x]) for x in f], *times, tq)[0]
        np.testing.assert_allclose(got, exp, rtol=0.0, atol=1e-12)
    # _fc_slope_scalar across same-sign, sign-change, and zero-secant cases
    for d0, d1 in [(1.0, 2.0), (-1.0, -3.0), (1.0, -1.0), (0.0, 2.0),
                   (2.0, 0.0), (-2.0, 3.0)]:
        got = _fc_slope_scalar(d0, d1, 10.0, 13.0)
        exp = float(_fc_slope(np.array([d0]), np.array([d1]), 10.0, 13.0)[0])
        np.testing.assert_allclose(got, exp, rtol=0.0, atol=1e-12)


def _spatial_snaps(mesh, times, amp_of_t):
    """Element u/v that vary in SPACE (a tilted plane) AND TIME (curved), so
    the leastsq fit is non-trivial and the FC-slope monotone limiter is
    actually exercised — a uniform field would make fused==unfused vacuous."""
    from kelp.core.io import VelocityField
    lon_c, lat_c = mesh.elem_center_lon, mesh.elem_center_lat
    nn = mesh.n_nodes
    snaps = []
    for t in times:
        a = amp_of_t(t)
        u = (0.30 + 0.10 * a) + 0.8 * a * lon_c + 0.25 * lat_c
        v = (-0.20 + 0.05 * a) - 0.40 * lat_c + 0.15 * a * lon_c
        snaps.append(VelocityField(u=u.astype(np.float64),
                                   v=v.astype(np.float64),
                                   eta=np.zeros(nn), w=None))
    return snaps


@pytest.mark.parametrize("step_name,c", [("rk2", (0.0, 0.5)),
                                         ("rk4", (0.0, 0.5, 0.5, 1.0))])
@pytest.mark.parametrize("time_dir", [1.0, -1.0])
def test_fused_pchip_matches_unfused(step_name, c, time_dir, monkeypatch):
    # The fused 4-snapshot cubic kernel must reproduce the unfused
    # make_stage_vels_recon -> reconstruct_pchip -> leastsq path element-for-
    # element, on a spatially- AND temporally-varying field, both directions.
    import kelp.numerics.integrators as I
    from kelp.numerics.integrators import (
        rk2_step, rk4_step, _rk2_step_unfused, _rk4_step_unfused)

    # Spy: prove rk*_step actually routes through the FUSED path (else the test
    # would compare unfused-vs-unfused and pass vacuously).
    calls = {"n": 0}
    real_fused = I._run_fused_2d_pchip

    def spy(*a, **k):
        calls["n"] += 1
        return real_fused(*a, **k)
    monkeypatch.setattr(I, "_run_fused_2d_pchip", spy)

    mesh = _make_mesh()
    times = [0.0, 100.0, 235.0, 360.0]                 # non-uniform
    amp = lambda t: np.sin(0.012 * t) + 0.5 * np.cos(0.017 * t)  # curved in time
    snaps = _spatial_snaps(mesh, times, amp)
    rng = np.random.default_rng(7)
    lon = rng.uniform(-0.10, 0.10, 16)
    lat = rng.uniform(-0.10, 0.10, 16)
    state = make_state(lon, lat)
    dt = 50.0
    sim_time = 110.0 if time_dir > 0 else 225.0        # stages stay in [100, 235]
    recon = (snaps, times, sim_time, time_dir)

    step = rk2_step if step_name == "rk2" else rk4_step
    unfused = _rk2_step_unfused if step_name == "rk2" else _rk4_step_unfused
    out_f = step(state, snaps[1], mesh, dt, time_dir,
                 recon=recon, uv_interp="leastsq")
    assert calls["n"] == 1                             # fused path was taken
    out_u = unfused(state, snaps[1], mesh, dt, time_dir, 0, False,
                    uv_interp="leastsq", recon=recon)

    np.testing.assert_allclose(out_f.lon, out_u.lon, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(out_f.lat, out_u.lat, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(out_f.u, out_u.u, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(out_f.v, out_u.v, rtol=0.0, atol=1e-9)
    np.testing.assert_array_equal(out_f.status, out_u.status)
    # not a trivially-stationary scenario
    assert np.any(np.abs(out_f.lon - lon) > 1e-6)


def test_fused_pchip_validation_rejects_bad_bundle():
    # _run_fused_2d_pchip must raise (contract parity with reconstruct_pchip)
    # on a malformed recon bundle instead of dividing by zero / extrapolating /
    # indexing out of bounds inside Numba.
    from kelp.core.io import VelocityField
    from kelp.numerics.integrators import rk2_step
    mesh = _make_mesh()
    times = [0.0, 100.0, 235.0, 360.0]
    snaps = _spatial_snaps(mesh, times, lambda t: np.sin(0.01 * t))
    state = make_state(np.array([0.02]), np.array([0.0]))

    # (a) stage time outside the central bracket (dt overshoots forward)
    with pytest.raises(ValueError, match="central PCHIP interval"):
        rk2_step(state, snaps[1], mesh, 500.0, 1.0,
                 recon=(snaps, times, 110.0, 1.0), uv_interp="leastsq")
    # (b) non-strictly-increasing (duplicate) times
    with pytest.raises(ValueError, match="strictly increasing"):
        rk2_step(state, snaps[1], mesh, 50.0,  1.0,
                 recon=(snaps, [0.0, 100.0, 100.0, 360.0], 110.0, 1.0),
                 uv_interp="leastsq")
    # (c) recon direction disagrees with the step's time_dir
    with pytest.raises(ValueError, match="disagrees with"):
        rk2_step(state, snaps[1], mesh, 50.0, 1.0,
                 recon=(snaps, times, 110.0, -1.0), uv_interp="leastsq")
    # (d) snapshot u/v shape mismatch across the stencil
    ne = mesh.n_elements
    bad = VelocityField(u=np.full(ne - 1, 0.3), v=np.full(ne - 1, 0.1),
                        eta=np.zeros(mesh.n_nodes), w=None)
    with pytest.raises(ValueError, match=r"shapes .* must all be"):
        rk2_step(state, snaps[1], mesh, 50.0, 1.0,
                 recon=(snaps[:2] + [bad, snaps[3]], times, 110.0, 1.0),
                 uv_interp="leastsq")
    # (e) ALL four snapshots one element short — the per-snapshot shapes AGREE,
    # but none match n_elements, so the kernel would index out of bounds.
    short = [VelocityField(u=np.full(ne - 1, 0.3), v=np.full(ne - 1, 0.1),
                           eta=np.zeros(mesh.n_nodes), w=None) for _ in range(4)]
    with pytest.raises(ValueError, match=r"shapes .* must all be"):
        rk2_step(state, short[1], mesh, 50.0, 1.0,
                 recon=(short, times, 110.0, 1.0), uv_interp="leastsq")
    # (f) node-layout velocity reaching the leastsq fused path (direct caller
    # bypassing the tracking-layer _validate_uv_compat guard) must be rejected.
    nn = mesh.n_nodes
    nodev = [VelocityField(u=np.full(nn, 0.3), v=np.full(nn, 0.1),
                           eta=np.zeros(nn), w=None, uv_layout="node")
             for _ in range(4)]
    with pytest.raises(ValueError, match="element-layout velocity"):
        rk2_step(state, nodev[1], mesh, 50.0, 1.0,
                 recon=(nodev, times, 110.0, 1.0), uv_interp="leastsq")


def test_pchip_recon_not_fused_for_barycentric_or_3d(monkeypatch):
    # Lock the dispatch gating: the fused PCHIP kernel must NOT be reached for
    # barycentric u/v (unfused-only) nor for 3D (is3d gates the whole block).
    import kelp.numerics.integrators as I
    from kelp.numerics.integrators import rk2_step, rk4_step
    from kelp.core.io import VelocityField
    from kelp.core.particles import make_state as mk

    calls = {"n": 0}
    real = I._run_fused_2d_pchip

    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(I, "_run_fused_2d_pchip", spy)

    # (a) barycentric 2D recon -> unfused
    mesh = _make_mesh()
    times = [0.0, 100.0, 235.0, 360.0]
    snaps = _spatial_snaps(mesh, times, lambda t: np.sin(0.01 * t))
    st = make_state(np.array([0.02]), np.array([0.0]))
    out = rk2_step(st, snaps[1], mesh, 50.0, 1.0,
                   recon=(snaps, times, 110.0, 1.0), uv_interp="barycentric")
    assert calls["n"] == 0 and np.isfinite(out.lon[0])

    # (b) 3D recon -> unfused (is3d True)
    K = _M.sigma.size
    ne, nn = _M.n_elements, _M.n_nodes
    fd = 3600.0
    t3 = np.arange(6) * fd

    def vf3d(t):
        u = np.full((K - 1, ne), 0.2 * np.cos(2 * np.pi * t / (6 * fd)))
        v = np.zeros((K - 1, ne))
        w = np.full((K, nn), 1e-3 * np.cos(2 * np.pi * t / (6 * fd)))
        return VelocityField(u=u, v=v, eta=np.zeros(nn), w=w)

    s3 = [vf3d(t) for t in t3]
    n = 2
    recon3 = (s3[n - 1:n + 3], [float(t3[i]) for i in range(n - 1, n + 3)],
              float(t3[n]), 1.0)
    st3 = mk(np.array([0.0]), np.array([0.0]), np.array([50.0]))
    out3 = rk4_step(st3, s3[n], _M, fd, 1.0, recon=recon3,
                    uv_interp="leastsq", w_interp="leastsq")
    assert calls["n"] == 0 and np.isfinite(out3.z[0])


# ---------------------------------------------------------------------------
# Parallel whole-field blend + stage deduplication (TODO 6.7)
# ---------------------------------------------------------------------------

class TestParallelBlendKernel:
    def _rand_fields(self, shape, seed=0, dtype=np.float64):
        rng = np.random.default_rng(seed)
        fs = [rng.standard_normal(shape).astype(dtype) for _ in range(4)]
        # plant zero secants / sign changes so the Fritsch-Carlson
        # zero-slope branch is exercised, not only the harmonic mean
        fs[1][..., ::7] = fs[0][..., ::7]
        return fs

    @pytest.mark.parametrize("stencil,tq", [
        ((0.0, 10.0, 20.0, 30.0), 13.7),      # uniform spacing
        ((0.0, 7.3, 20.9, 24.1), 9.1),        # non-uniform (w1 != w2)
    ])
    def test_kernel_bit_identical_to_numpy_path(self, monkeypatch,
                                                stencil, tq):
        from kelp.numerics import temporal_reconstruct as tr
        # Anti-vacuity guards (mirror test_lerp_blend.py): on a
        # numba-less env this would silently compare numpy-vs-numpy.
        assert tr.HAS_NUMBA, "parity test needs the numba kernel"
        # 2-D shape on purpose: for a 1-D array F-order IS C-order and
        # the reference would ride the kernel too.
        shape = (48, 1500)          # 72000 elements >= the size threshold
        f0, f1, f2, f3 = self._rand_fields(shape)
        calls = []
        orig = tr._pchip_blend4_flat

        def spy(*args):
            calls.append(1)
            return orig(*args)

        monkeypatch.setattr(tr, "_pchip_blend4_flat", spy)
        got = tr._pchip_blend4(f0, f1, f2, f3, *stencil, tq)
        # Fortran-ordered copies carry identical VALUES but are not
        # C-contiguous -> the SAME public function takes the numpy path.
        ref = tr._pchip_blend4(np.asfortranarray(f0), np.asfortranarray(f1),
                               np.asfortranarray(f2), np.asfortranarray(f3),
                               *stencil, tq)
        assert len(calls) == 1      # got hit the kernel; ref took numpy
        assert got.dtype == np.float64 and got.shape == shape
        np.testing.assert_array_equal(got, ref)

    def test_step_level_kernel_path_bitwise(self, monkeypatch):
        """rk4_step + recon through the prange-kernel path must produce
        bitwise-identical trajectories to the numpy path.  Tiny test
        meshes never reach the size gate, so force it with
        _PCHIP_MIN_SIZE=1 (the gate reads the module global at call
        time) and compare against the default run."""
        from tests.test_3d_analytical import (_make_mesh as _mk3d,
                                              _make_vel_3d)
        from kelp.numerics.integrators import rk4_step
        from kelp.numerics import temporal_reconstruct as tr
        from kelp.core.mesh import find_nearest_nodes
        assert tr.HAS_NUMBA

        mesh = _mk3d(5.0, 5.0)
        times = [0.0, 100.0, 200.0, 300.0]
        snaps = [_make_vel_3d(mesh, u=0.1 + 0.05 * i, v=0.02 * i,
                              w=0.001 * i) for i in range(4)]
        st = make_state(np.full(6, 5.0), np.full(6, 5.0),
                        np.linspace(20.0, 150.0, 6))
        st = st.replace(
            nearest_node=find_nearest_nodes(mesh, st.lon, st.lat))
        recon = (snaps, times, 150.0, 1.0)
        ref = rk4_step(st, snaps[1], mesh, 50.0, 1.0, recon=recon,
                       w_interp="leastsq", uv_interp="leastsq")
        monkeypatch.setattr(tr, "_PCHIP_MIN_SIZE", 1)
        got = rk4_step(st, snaps[1], mesh, 50.0, 1.0, recon=recon,
                       w_interp="leastsq", uv_interp="leastsq")
        for attr in ("lon", "lat", "z", "status"):
            np.testing.assert_array_equal(getattr(got, attr),
                                          getattr(ref, attr))

    def test_small_arrays_still_work(self):
        from kelp.numerics.temporal_reconstruct import _pchip_blend4
        small = self._rand_fields((4, 4))
        v = _pchip_blend4(*small, *T, 15.0)
        assert v.dtype == np.float64 and v.shape == (4, 4)

    def test_float32_input_matches_float64(self):
        from kelp.numerics.temporal_reconstruct import _pchip_blend4
        f32 = self._rand_fields((48, 1500), dtype=np.float32)
        got = _pchip_blend4(*f32, *T, 15.0)          # cast → kernel path
        # F-ordered float64 copies force the NUMPY path for the reference
        # (a C-contiguous float64 reference would ride the kernel too and
        # prove nothing about cast/fallback parity — review).
        ref = _pchip_blend4(
            *[np.asfortranarray(x.astype(np.float64)) for x in f32],
            *T, 15.0)
        assert got.dtype == np.float64
        np.testing.assert_array_equal(got, ref)

    def test_numpy_float32_time_scalar_path_independent(self):
        """Time scalars are normalized to float64 BEFORE the path split:
        a np.float32 tq must give bitwise-equal results on the kernel and
        numpy paths (review)."""
        from kelp.numerics.temporal_reconstruct import _pchip_blend4
        f0, f1, f2, f3 = self._rand_fields((48, 1500))
        tq = np.float32(13.7)
        got = _pchip_blend4(f0, f1, f2, f3, *T, tq)   # kernel path
        ref = _pchip_blend4(np.asfortranarray(f0), np.asfortranarray(f1),
                            np.asfortranarray(f2), np.asfortranarray(f3),
                            *T, tq)                    # numpy path
        np.testing.assert_array_equal(got, ref)


class TestStageDedup:
    def _snaps(self):
        return [_vf(s, with_w=True) for s in (1.0, 2.0, 4.0, 8.0)], T

    def test_duplicate_stage_times_share_object(self):
        """RK4's c=(0, .5, .5, 1): the two midpoint stages must be ONE
        reconstruction (whole-field, N-independent cost — TODO 6.7)."""
        from kelp.numerics.temporal_reconstruct import make_stage_vels_recon
        snaps, times = self._snaps()
        vels = make_stage_vels_recon(snaps, times, 12.0, 4.0, 1.0,
                                     (0.0, 0.5, 0.5, 1.0))
        assert vels[1] is vels[2]
        assert vels[0] is not vels[1]

    def test_each_unique_time_reconstructed_once(self, monkeypatch):
        from kelp.numerics import temporal_reconstruct as tr
        calls = []
        orig = tr.reconstruct_pchip

        def counting(snaps, times, tq):
            calls.append(tq)
            return orig(snaps, times, tq)

        monkeypatch.setattr(tr, "reconstruct_pchip", counting)
        snaps, times = self._snaps()
        tr.make_stage_vels_recon(snaps, times, 12.0, 4.0, 1.0,
                                 (0.0, 0.5, 0.5, 1.0))
        assert len(calls) == 3          # unique stage times: 12, 14, 16

    def test_dedup_values_match_direct_reconstruction(self):
        from kelp.numerics.temporal_reconstruct import (
            make_stage_vels_recon, reconstruct_pchip)
        snaps, times = self._snaps()
        vels = make_stage_vels_recon(snaps, times, 12.0, 4.0, 1.0,
                                     (0.0, 0.5, 1.0))
        for c, v in zip((0.0, 0.5, 1.0), vels):
            ref = reconstruct_pchip(snaps, times, 12.0 + c * 4.0)
            np.testing.assert_array_equal(v.u, ref.u)
            np.testing.assert_array_equal(v.eta, ref.eta)
            np.testing.assert_array_equal(v.w, ref.w)
