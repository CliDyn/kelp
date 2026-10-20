"""Depth-linear u/v in the integrators.

Every u/v evaluation of a step reads the depth of its own stage: Euler the
step-entry depth, RK2 the entry depth and then ``z_mid``, RK4 the entry
depth and then ``z2``, ``z3``, ``z4`` (``_interp_uv_at(..., zs=...)``).  The
step functions refuse a depth-linear field they cannot serve at entry
(``_check_depth_linear_entry``) and run the record check on every field
they receive (``vel``, ``vel_next``, each PCHIP snapshot, the raw parents
of a ``LazyLerpField``).  Fused RK2 on node-layout barycentric data runs the
depth-linear twins (``tests/test_fused_3d_depth_linear.py``); the fused
Euler kernel runs element ``leastsq`` fields through its depth-linear twin
(``tests/test_integrators_depth_linear_elements.py``).

All synthetic: the 3x3 test mesh of ``test_3d_analytical`` with levels
``[0, 10, 20, 30, 40]`` m (the fixture of ``test_interp_depth_linear``), and
``u = a + b*z (+ e*t)`` stored at the data points, so every expected
position is a hand-written stage combination.
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from kelp.core.geometry import DEG2RAD, METERS_PER_DEG_LAT, wrap_lon
from kelp.core.io import LazyLerpField, VelocityField
from kelp.core.mesh import find_containing_elements, find_nearest_nodes
from kelp.core.particles import ParticleStatus, make_state
from kelp.numerics import integrators as I
from kelp.numerics import interpolate as ip
from kelp.numerics.vertical_uv import is_checked

from tests._bitwise import assert_bitwise
from tests.test_interp_depth_linear import (
    A_U, A_V, B_U, B_V, LEVELS, P_PTS, SHALLOW_PT, _field, _hand_wall, _lam,
    _mesh,
)

M = METERS_PER_DEG_LAT
N_NODES = 9
K = len(LEVELS)

#: time slope of u and v (m/s per s) for the time-dependent fixtures
E_U, E_V = 2.0e-5, -1.0e-5

#: three particles inside the deep mesh, at depths off the data points
LON0 = np.array([4.6, 5.2, 5.4])
LAT0 = np.array([4.7, 5.3, 4.8])
Z0 = np.array([12.5, 17.5, 22.5])

STEPS = {"euler": I.euler_step, "rk2": I.rk2_step, "rk4": I.rk4_step}
C_RK4 = (0.0, 0.5, 0.5, 1.0)

needs_numba = pytest.mark.skipif(not I.HAS_NUMBA,
                                 reason="fused kernels need numba")


# ---------------------------------------------------------------------------
# Fixtures and hand formulas
# ---------------------------------------------------------------------------

def _deep_mesh():
    """Every node 200 m deep: last point = the deepest one everywhere, so
    ``u = a + b*clip(z)`` at any horizontal position."""
    return _mesh(np.full(N_NODES, 200.0))


def _deep_field(placement, w0=0.0, t=0.0):
    """``u = A_U + B_U*p_k + E_U*t`` at every node and point (v likewise),
    uniform ``w = w0`` on the levels."""
    p = P_PTS[placement]
    col_u = A_U + B_U * p + E_U * t
    col_v = A_V + B_V * p + E_V * t
    return VelocityField(
        u=np.repeat(col_u[:, None], N_NODES, axis=1),
        v=np.repeat(col_v[:, None], N_NODES, axis=1),
        eta=np.zeros(N_NODES), w=np.full((K, N_NODES), float(w0)),
        uv_layout="node", uv_vertical=placement)


def _layer_field(w0=0.0, layout="node", n_elem=8, seed=0):
    """A plain ``"layer"`` field: (K-1, N) node or (K-1, E) element u/v."""
    rng = np.random.default_rng(seed)
    n = N_NODES if layout == "node" else n_elem
    return VelocityField(
        u=rng.uniform(0.1, 0.5, (K - 1, n)),
        v=rng.uniform(-0.3, 0.3, (K - 1, n)),
        eta=np.zeros(N_NODES), w=np.full((K, N_NODES), float(w0)),
        uv_layout=layout)


def _state(lon=LON0, lat=LAT0, z=Z0, *, mesh=None, walk=False):
    st = make_state(np.array(lon, dtype=np.float64),
                    np.array(lat, dtype=np.float64),
                    np.array(z, dtype=np.float64))
    if walk:
        nn = find_nearest_nodes(mesh, st.lon, st.lat)
        e, ok = find_containing_elements(mesh, st.lon, st.lat, nn)
        assert ok.all()
        st = st.replace(elem=np.asarray(e, dtype=np.int32))
    return st


def _clip(z, lo, hi):
    return min(max(z, lo), hi)


def _uv_deep(placement, t_k=(0.0, 0.0, 0.0, 0.0)):
    """Hand u/v of the deep fixture: stage ``k`` at time ``t_k[k]``."""
    p = P_PTS[placement]

    def uv(k, x, y, z):
        zc = _clip(z, p[0], p[-1])
        return (A_U + B_U * zc + E_U * t_k[k],
                A_V + B_V * zc + E_V * t_k[k])
    return uv


def _uv_shallow(mesh, placement, tri=(1, 2, 4)):
    """Hand u/v in the triangle with the 10 m node (the shallow fixture
    of test_interp_depth_linear), the wall rule of ``_hand_wall``: above
    the 10 m bed ``sum_i lam_i * (a + b*clip(z, p[0], p[last_i]))`` —
    per-vertex ends; below it the 10 m node is dry and left out (the
    other two vertices' renormalised blend)."""
    def uv(k, x, y, z):
        lam = _lam(mesh, tri, x, y)
        assert (lam > 0.0).all(), f"stage {k} left triangle {tri}: {lam}"
        return (_hand_wall(mesh, placement, tri, lam, z,
                           lambda nd, zc: A_U + B_U * zc),
                _hand_wall(mesh, placement, tri, lam, z,
                           lambda nd, zc: A_V + B_V * zc))
    return uv


def _hand_euler(lon0, lat0, z0, dt, td, uv):
    u, v = uv(0, lon0, lat0, z0)
    return (dt * td * u / (np.cos(lat0 * DEG2RAD) * M),
            dt * td * v / M, (td * u, td * v))


def _hand_rk2(lon0, lat0, z0, dt, td, w0, uv):
    """Midpoint RK2: stage 2 at ``z0 + dt/2 * W1`` (W = td * w0)."""
    u1, v1 = uv(0, lon0, lat0, z0)
    U1, V1 = td * u1, td * v1
    lon_m = lon0 + 0.5 * dt * U1 / (np.cos(lat0 * DEG2RAD) * M)
    lat_m = lat0 + 0.5 * dt * V1 / M
    z_m = z0 + 0.5 * dt * td * w0
    u2, v2 = uv(1, lon_m, lat_m, z_m)
    U2, V2 = td * u2, td * v2
    return (dt * U2 / (np.cos(lat_m * DEG2RAD) * M), dt * V2 / M, (U1, V1))


def _hand_rk4(lon0, lat0, z0, dt, td, w0, uv):
    """Classical RK4: stages at ``z0``, ``z0 + dt/2 W``, ``z0 + dt/2 W``,
    ``z0 + dt W`` (uniform W = td * w0)."""
    W = td * w0
    u1, v1 = uv(0, lon0, lat0, z0)
    U1, V1 = td * u1, td * v1
    dlon1 = dt * U1 / (np.cos(lat0 * DEG2RAD) * M)
    dlat1 = dt * V1 / M
    lon2, lat2, z2 = lon0 + 0.5 * dlon1, lat0 + 0.5 * dlat1, z0 + 0.5 * dt * W
    u2, v2 = uv(1, lon2, lat2, z2)
    dlon2 = dt * td * u2 / (np.cos(lat2 * DEG2RAD) * M)
    dlat2 = dt * td * v2 / M
    lon3, lat3, z3 = lon0 + 0.5 * dlon2, lat0 + 0.5 * dlat2, z0 + 0.5 * dt * W
    u3, v3 = uv(2, lon3, lat3, z3)
    dlon3 = dt * td * u3 / (np.cos(lat3 * DEG2RAD) * M)
    dlat3 = dt * td * v3 / M
    lon4, lat4, z4 = lon0 + dlon3, lat0 + dlat3, z0 + dt * W
    u4, v4 = uv(3, lon4, lat4, z4)
    dlon4 = dt * td * u4 / (np.cos(lat4 * DEG2RAD) * M)
    dlat4 = dt * td * v4 / M
    return ((dlon1 + 2.0 * dlon2 + 2.0 * dlon3 + dlon4) / 6.0,
            (dlat1 + 2.0 * dlat2 + 2.0 * dlat3 + dlat4) / 6.0, (U1, V1))


HAND = {"euler": lambda lo, la, z, dt, td, w0, uv: _hand_euler(lo, la, z, dt,
                                                               td, uv),
        "rk2": _hand_rk2, "rk4": _hand_rk4}


def _hand_all(integrator, st, dt, td, w0, uv):
    out = [HAND[integrator](st.lon[i], st.lat[i], st.z[i], dt, td, w0, uv)
           for i in range(st.n_particles)]
    dlon = np.array([o[0] for o in out])
    dlat = np.array([o[1] for o in out])
    u1 = np.array([o[2][0] for o in out])
    v1 = np.array([o[2][1] for o in out])
    return dlon, dlat, u1, v1


def _assert_step(out, st, dlon, dlat, rtol=1e-12):
    """Positions equal the hand stage combination: displacement to ``rtol``
    relative (the longitude also allows one rounding of the [-180, 180)
    wrap, which both sides go through)."""
    assert np.all(np.isfinite(out.lon)) and np.all(np.isfinite(out.lat))
    exp_lon = wrap_lon(st.lon + dlon)
    np.testing.assert_allclose(out.lon - st.lon, exp_lon - st.lon,
                               rtol=rtol, atol=4e-14)
    np.testing.assert_allclose(out.lat - st.lat, dlat, rtol=rtol,
                               atol=2e-15)


def _assert_not_step(out, st, dlon, dlat):
    """The w = 0 hand answer is far from the step (stage depth mattered)."""
    assert np.all(np.abs((out.lon - st.lon) - dlon) > 1e-6 * np.abs(dlon))
    assert np.all(np.abs((out.lat - st.lat) - dlat) > 1e-6 * np.abs(dlat))


# ---------------------------------------------------------------------------
# Stage depths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("td", [1.0, -1.0])
@pytest.mark.parametrize("w0", [0.0, 0.004])
@pytest.mark.parametrize("placement", ["level", "layer_linear"])
def test_euler_uses_entry_depth(placement, w0, td):
    """Euler reads u/v at the entry depth whatever ``w`` is:
    ``dlon = dt * td * (a + b*z0) / (cos(lat0) M)``."""
    mesh = _deep_mesh()
    vel = _deep_field(placement, w0)
    st = _state()
    dt = 3600.0
    out = I.euler_step(st, vel, mesh, dt, td, uv_interp="barycentric",
                       w_interp="barycentric")
    dlon, dlat, u1, v1 = _hand_all("euler", st, dt, td, w0,
                                   _uv_deep(placement))
    _assert_step(out, st, dlon, dlat)
    np.testing.assert_allclose(out.u, td * (A_U + B_U * Z0), rtol=1e-13)
    np.testing.assert_allclose(out.v, td * (A_V + B_V * Z0), rtol=1e-13)


def _time_aware(placement, mode, td, w0, dt):
    """(vel, kwargs, stage times) for a steady or a linear-in-time step."""
    if mode == "steady":
        return _deep_field(placement, w0), {}, (0.0,) * 4
    t_vel = 0.0 if td > 0 else 3600.0
    vel = _deep_field(placement, w0, t=t_vel)
    vel_next = _deep_field(placement, w0, t=t_vel + td * 3600.0)
    t_k = tuple(t_vel + td * c * dt for c in C_RK4)
    return vel, dict(vel_next=vel_next, dt_to_next=3600.0), t_k


@pytest.mark.parametrize("walk", [False, True], ids=["scan", "walk"])
@pytest.mark.parametrize("mode", ["steady", "lerp"])
@pytest.mark.parametrize("td", [1.0, -1.0])
@pytest.mark.parametrize("placement", ["level", "layer_linear"])
def test_rk2_stage_depth(placement, td, mode, walk):
    """Stage 2 reads u/v at ``z_mid = z0 + dt/2 * W1``; the same formula
    with ``w = 0`` is a different horizontal answer."""
    mesh = _deep_mesh()
    dt, w0 = 3600.0, 0.004
    vel, kw, t_k = _time_aware(placement, mode, td, w0, dt)
    t_k = t_k[:2]                       # RK2's c = (0, 0.5)
    st = _state(mesh=mesh, walk=walk)
    out = I.rk2_step(st, vel, mesh, dt, td, uv_interp="barycentric",
                     w_interp="barycentric", **kw)
    dlon, dlat, _, _ = _hand_all("rk2", st, dt, td, w0,
                                 _uv_deep(placement, t_k))
    _assert_step(out, st, dlon, dlat)
    dlon0, dlat0, _, _ = _hand_all("rk2", st, dt, td, 0.0,
                                   _uv_deep(placement, t_k))
    _assert_not_step(out, st, dlon0, dlat0)
    np.testing.assert_allclose(out.z, np.clip(Z0 + dt * td * w0, 0.0, 200.0),
                               rtol=1e-14)


@pytest.mark.parametrize("walk", [False, True], ids=["scan", "walk"])
@pytest.mark.parametrize("mode", ["steady", "lerp"])
@pytest.mark.parametrize("td", [1.0, -1.0])
@pytest.mark.parametrize("placement", ["level", "layer_linear"])
def test_rk4_stage_depth(placement, td, mode, walk):
    """RK4 stages 2-4 read u/v at ``z2 = z3 = z0 + dt/2 W`` and
    ``z4 = z0 + dt W``; ``w = 0`` gives a different answer."""
    mesh = _deep_mesh()
    dt, w0 = 3600.0, 0.004
    vel, kw, t_k = _time_aware(placement, mode, td, w0, dt)
    st = _state(mesh=mesh, walk=walk)
    out = I.rk4_step(st, vel, mesh, dt, td, uv_interp="barycentric",
                     w_interp="nearest", **kw)
    dlon, dlat, _, _ = _hand_all("rk4", st, dt, td, w0,
                                 _uv_deep(placement, t_k))
    _assert_step(out, st, dlon, dlat)
    dlon0, dlat0, _, _ = _hand_all("rk4", st, dt, td, 0.0,
                                   _uv_deep(placement, t_k))
    _assert_not_step(out, st, dlon0, dlat0)


def _recon(placement, w0, times, sim_time, td, *, nan_in=None):
    snaps = [_deep_field(placement, w0, t=t) for t in times]
    if nan_in is not None:
        snaps[nan_in].u[1, 3] = np.nan
    return snaps, (snaps, list(times), float(sim_time), int(td))


@pytest.mark.parametrize("placement", ["level", "layer_linear"])
def test_rk4_backward_pchip_stage_depths(placement):
    """Backward RK4 with a 4-snapshot PCHIP bundle.  The
    data are linear in time, which PCHIP reproduces, so stage ``k`` reads
    ``a + b*z_k + e*(t0 - c_k*dt)`` at its own depth."""
    mesh = _deep_mesh()
    dt, td, w0 = 3600.0, -1.0, 0.004
    times = (0.0, 3600.0, 7200.0, 10800.0)
    snaps, recon = _recon(placement, w0, times, 7200.0, td)
    st = _state()
    out = I.rk4_step(st, snaps[2], mesh, dt, td, uv_interp="barycentric",
                     w_interp="barycentric", vel_next=snaps[1],
                     dt_to_next=3600.0, recon=recon)
    t_k = tuple(7200.0 + td * c * dt for c in C_RK4)
    assert t_k == (7200.0, 5400.0, 5400.0, 3600.0)
    dlon, dlat, u1, v1 = _hand_all("rk4", st, dt, td, w0,
                                   _uv_deep(placement, t_k))
    _assert_step(out, st, dlon, dlat)
    np.testing.assert_allclose(out.u, u1, rtol=1e-13)
    dlon0, dlat0, _, _ = _hand_all("rk4", st, dt, td, 0.0,
                                   _uv_deep(placement, t_k))
    _assert_not_step(out, st, dlon0, dlat0)


@pytest.mark.parametrize("end", ["below_bed", "above_surface"])
@pytest.mark.parametrize("integrator", ["rk2", "rk4"])
@pytest.mark.parametrize("placement", ["level", "layer_linear"])
def test_stage_overshoot_clamped(placement, integrator, end):
    """A stage depth below the data (z0 + 60 m) or above the surface
    (z0 - 60 m) — the shallow fixture of test_interp_depth_linear, node 2
    10 m deep, the others 200 m.  Above the surface every vertex reads
    ``p[0]``; at 68 m node 2 is dry and left out (a wall), the deep
    vertices read their last point ``p[last_i]``: finite, by hand, never
    the NaN / +inf stored below ``last``."""
    mesh = _mesh()
    # z0 = 8 m: inside every column of both placements except node 2's
    # layer_linear one (last point 5 m), so the entry stage reads other
    # values than the overshooting stages
    z0, w0 = 8.0, (0.2 if end == "below_bed" else -0.2)
    vel = dataclasses.replace(_field(placement),
                              w=np.full((K, N_NODES), w0))
    st = _state([SHALLOW_PT[0]], [SHALLOW_PT[1]], [z0])
    dt, td = 600.0, 1.0
    z_stage = z0 + 0.5 * dt * w0                    # RK2 z_mid, RK4 z2/z3
    p = P_PTS[placement]
    assert (z_stage > p[-1]) if end == "below_bed" else (z_stage < p[0])
    out = STEPS[integrator](st, vel, mesh, dt, td, uv_interp="barycentric",
                            w_interp="nearest")
    dlon, dlat, _, _ = _hand_all(integrator, st, dt, td, w0,
                                 _uv_shallow(mesh, placement))
    _assert_step(out, st, dlon, dlat)


@pytest.mark.parametrize("integrator", ["euler", "rk2", "rk4"])
def test_unfused_depth_linear_skips_layer_lookup(monkeypatch, integrator):
    """The unfused stages of a depth-linear step read their own depth, so
    no per-stage layer lookup (``_compute_layers``) runs; a layer field of
    the same step still looks its layers up.  (RK2 with ``elem_search:
    walk`` is unfused; Euler with barycentric u/v and RK4 always are.)"""
    mesh = _deep_mesh()
    calls = []
    real = I._compute_layers

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)
    monkeypatch.setattr(I, "_compute_layers", spy)
    kw = dict(uv_interp="barycentric", w_interp="nearest")
    for vel, expect_calls in ((_deep_field("level", 0.004), False),
                              (_layer_field(0.004), True)):
        calls.clear()
        STEPS[integrator](_state(mesh=mesh, walk=True), vel, mesh, 600.0,
                          1.0, **kw)
        assert bool(calls) is expect_calls, (vel.uv_vertical, len(calls))


# ---------------------------------------------------------------------------
# Recorded velocity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("integrator", ["rk2", "rk4"])
def test_recorded_velocity_is_first_stage(integrator):
    """RK2/RK4 record the first stage: the depth-linear value at the
    step-ENTRY position and depth, not a later stage's."""
    mesh = _deep_mesh()
    dt, w0 = 3600.0, 0.004
    vel = _deep_field("level", w0)
    st = _state()
    out = STEPS[integrator](st, vel, mesh, dt, 1.0, uv_interp="barycentric",
                            w_interp="barycentric")
    np.testing.assert_allclose(out.u, A_U + B_U * Z0, rtol=1e-13)
    np.testing.assert_allclose(out.v, A_V + B_V * Z0, rtol=1e-13)
    np.testing.assert_allclose(out.w, np.full(3, w0), rtol=1e-13)
    z_mid = Z0 + 0.5 * dt * w0
    assert np.all(np.abs(out.u - (A_U + B_U * z_mid)) > 0.1)


# ---------------------------------------------------------------------------
# Entry refusals
# ---------------------------------------------------------------------------

_PREFIX = r"^{}_step: depth-linear horizontal velocity \(uv_vertical='level'\) "


@pytest.mark.parametrize("name", ["euler", "rk2", "rk4"])
def test_direct_default_uv_interp_refused(name):
    """A direct call on a NODE-layout depth-linear field with the default
    ``uv_interp="leastsq"`` (or any other non-barycentric interpolator) is
    refused at the step entry, naming the layout (element layout takes
    ``leastsq`` and ``edge_leastsq``: the layout table,
    ``tests/test_integrators_depth_linear_elements.py::test_entry_table``)."""
    mesh = _deep_mesh()
    vel = _deep_field("level", 0.004)
    st = _state()
    with pytest.raises(ValueError,
                       match=_PREFIX.format(name) + r".*uv_interp"):
        STEPS[name](st, vel, mesh, 600.0, 1.0)
    for other in ("leastsq", "edge_leastsq"):
        with pytest.raises(ValueError, match=_PREFIX.format(name)
                           + r"on node layout \(uv_layout='node'\) needs "
                             rf"uv_interp='barycentric', got "
                             rf"uv_interp='{other}'$"):
            STEPS[name](st, vel, mesh, 600.0, 1.0, uv_interp=other,
                        w_interp="barycentric")


@pytest.mark.parametrize("uv_interp", ["leastsq", "barycentric"])
def test_rk45_refuses_depth_linear(uv_interp):
    mesh = _deep_mesh()
    vel = _deep_field("level", 0.004)
    with pytest.raises(ValueError, match=_PREFIX.format("rk45") + r".*rk45"):
        I.rk45_step(_state(), vel, mesh, 600.0, 1.0, uv_interp=uv_interp)


@pytest.mark.parametrize("name", ["euler", "rk2", "rk4", "rk45"])
def test_sigma_mesh_accepted(name):
    """A sigma mesh passes the entry (rk45 still refuses): a
    ``"layer_linear"`` field is read in the sigma fraction ``s = z / H_p``
    of the 200 m columns, so the recorded first-stage velocity is
    ``a + b*s`` at the entry ``s`` (the stage rule in full:
    ``tests/test_integrators_depth_linear_sigma.py``)."""
    mesh = _deep_mesh()
    mesh.vertical_coord = "sigma"
    mesh.sigma = np.linspace(1.0, 0.0, K)       # interfaces s = 0, 1/4 .. 1
    cen = 1.0 - 0.5 * (mesh.sigma[:-1] + mesh.sigma[1:])   # 1/8 .. 7/8
    a_u, b_u, a_v, b_v = 0.2, 0.4, -0.1, 0.3
    vel = VelocityField(
        u=np.repeat((a_u + b_u * cen)[:, None], N_NODES, axis=1),
        v=np.repeat((a_v + b_v * cen)[:, None], N_NODES, axis=1),
        eta=np.zeros(N_NODES), w=np.full((K, N_NODES), 0.004),
        uv_layout="node", uv_vertical="layer_linear")
    st = _state(z=np.array([40.0, 90.0, 130.0]))    # s = 0.2, 0.45, 0.65
    step = dict(STEPS, rk45=I.rk45_step)[name]
    kw = dict(uv_interp="barycentric", w_interp="barycentric")
    if name == "rk45":
        with pytest.raises(ValueError, match=(
                r"^rk45_step: depth-linear horizontal velocity "
                r"\(uv_vertical='layer_linear'\) .*rk45")):
            step(st, vel, mesh, 600.0, 1.0, **kw)
        return
    out = step(st, vel, mesh, 600.0, 1.0, **kw)
    assert np.all(out.status == ParticleStatus.ALIVE)
    np.testing.assert_allclose(out.u, a_u + b_u * st.z / 200.0, rtol=1e-13)
    np.testing.assert_allclose(out.v, a_v + b_v * st.z / 200.0, rtol=1e-13)


def test_check_depth_linear_entry_refusals():
    """Every refusal of the helper, in its order: rk45, the mesh (neither
    z-level nor sigma), a 3-D field, the layout x interpolator table (node
    layout: ``barycentric`` only; element layout: ``leastsq``,
    ``edge_leastsq``, ``barycentric``, and its field then meets the element
    table's rules); then one placement for all fields, whichever side is
    depth-linear."""
    chk = I._check_depth_linear_entry
    mesh = _deep_mesh()
    sigma = _deep_mesh()
    sigma.vertical_coord = "sigma"
    other = _deep_mesh()
    other.vertical_coord = "hybrid"
    level = _deep_field("level", 0.004)
    pre = _PREFIX.format("rk2")
    # rk45 first, whatever else is wrong
    with pytest.raises(ValueError, match=_PREFIX.format("rk45") + r".*rk45"):
        chk(level, other, "leastsq", "rk45")
    with pytest.raises(ValueError, match=pre + (
            r"needs a z-level or sigma mesh \(vertical_coord 'zlevel' or "
            r"'sigma'\), got 'hybrid'$")):
        chk(level, other, "leastsq", "rk2")
    no_w = VelocityField(u=level.u, v=level.v, eta=level.eta, w=None,
                         uv_layout="node", uv_vertical="level")
    with pytest.raises(ValueError, match=pre + r"needs a 3-D field"):
        chk(no_w, mesh, "leastsq", "rk2")
    # element layout passes the table with leastsq, then meets the element
    # table's rules: layer-centred data only
    elem = SimpleNamespace(uv_vertical="level", uv_layout="element",
                           u=level.u, v=level.v, w=level.w, eta=level.eta)
    with pytest.raises(ValueError, match=r"^depth-linear u/v on element "
                       r"layout is 'layer_linear' only"):
        chk(elem, mesh, "leastsq", "rk2")
    with pytest.raises(ValueError, match=pre + r"on element layout "
                       r"\(uv_layout='element'\) needs uv_interp 'leastsq', "
                       r"'edge_leastsq' or 'barycentric', got "
                       r"uv_interp='nearest'"):
        chk(elem, mesh, "nearest", "rk2")
    with pytest.raises(ValueError, match=pre + r"on node layout "
                       r"\(uv_layout='node'\) needs uv_interp="
                       r"'barycentric', got uv_interp='leastsq'"):
        chk(level, mesh, "leastsq", "rk2")
    # mixed placements among the step's fields
    snaps, recon = _recon("level", 0.004, (0.0, 1.0, 2.0, 3.0), 1.0, 1)
    snaps[2] = _deep_field("layer_linear", 0.004)
    recon = (snaps,) + recon[1:]
    with pytest.raises(ValueError, match=r"^rk2_step: uv_vertical mismatch.*"
                       r"recon snapshot 2='layer_linear'"):
        chk(level, mesh, "barycentric", "rk2", recon=recon)
    # a "layer" vel with a depth-linear companion: the same refusal,
    # before anything of the mesh or of the data is touched
    lay = _layer_field(0.004)
    with pytest.raises(ValueError, match=r"^rk4_step: uv_vertical mismatch "
                       r"between the fields of one step: vel='layer', "
                       r"vel_next='level'$"):
        chk(lay, sigma, "leastsq", "rk4", vel_next=level)
    lay_snaps = [_layer_field(0.004, seed=k) for k in range(3)] + [level]
    with pytest.raises(ValueError, match=r"^euler_step: uv_vertical mismatch"
                       r".*recon snapshot 2='layer', "
                       r"recon snapshot 3='level'$"):
        chk(lay, sigma, "leastsq", "euler",
            recon=(lay_snaps, [0.0, 1.0, 2.0, 3.0], 1.0, 1))
    assert chk(lay, sigma, "leastsq", "rk2", vel_next=_layer_field(0.004),
               recon=(lay_snaps[:3] + [_layer_field(0.004)],
                      [0.0, 1.0, 2.0, 3.0], 1.0, 1)) is None
    # accepted: stamps the field
    assert not is_checked(level, mesh)
    assert chk(level, mesh, "barycentric", "rk2") is None
    assert is_checked(level, mesh)


@pytest.mark.parametrize("where", ["vel", "vel_next", "recon0", "recon3",
                                   "lazy_vel_b"])
@pytest.mark.parametrize("name", ["euler", "rk2", "rk4"])
def test_direct_step_refuses_nan_in_prefix(name, where):
    """A NaN at a usable point (node 3, point 1) of ANY field the step
    receives is refused at entry, naming the field — also when it sits only
    in ``vel_next``, in one PCHIP snapshot, or in a lazy field's parent."""
    mesh = _deep_mesh()
    w0 = 0.004
    kw = {}
    if where in ("recon0", "recon3"):
        snaps, recon = _recon("level", w0, (0.0, 3600.0, 7200.0, 10800.0),
                              3600.0, 1, nan_in=int(where[-1]))
        vel = snaps[1]
        kw = dict(vel_next=snaps[2], dt_to_next=3600.0, recon=recon)
        label = f"recon snapshot {where[-1]}"
    elif where == "lazy_vel_b":
        bad = _deep_field("level", w0, t=3600.0)
        bad.u[1, 3] = np.nan
        vel = LazyLerpField(_deep_field("level", w0), bad, 0.25)
        label = r"vel\.vel_b"
    else:
        vel = _deep_field("level", w0)
        nxt = _deep_field("level", w0, t=3600.0)
        (vel if where == "vel" else nxt).u[1, 3] = np.nan
        kw = dict(vel_next=nxt, dt_to_next=3600.0)
        label = where
    with pytest.raises(ValueError, match=(
            rf"non-finite at node 3, point 1 .*in record {label} "
            rf"\({name}_step\)")):
        STEPS[name](_state(), vel, mesh, 1800.0, 1.0,
                    uv_interp="barycentric", w_interp="barycentric", **kw)


def test_entry_record_check_once(monkeypatch):
    """The record check runs once per field and mesh table: a second step
    with the same fields scans nothing."""
    mesh = _deep_mesh()
    calls = []
    real = I.check_depth_linear_record

    def spy(vel, mesh_, *, record=""):
        calls.append(record)
        return real(vel, mesh_, record=record)
    monkeypatch.setattr(I, "check_depth_linear_record", spy)
    vel = _deep_field("level", 0.004)
    nxt = _deep_field("level", 0.004, t=3600.0)
    kw = dict(uv_interp="barycentric", w_interp="nearest", vel_next=nxt,
              dt_to_next=3600.0)
    I.rk4_step(_state(), vel, mesh, 1800.0, 1.0, **kw)
    assert calls == ["vel (rk4_step)", "vel_next (rk4_step)"]
    I.rk4_step(_state(), vel, mesh, 1800.0, 1.0, **kw)
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# Fused 3-D Euler: layer fields only
# ---------------------------------------------------------------------------

class _Taken(Exception):
    pass


def _spies(monkeypatch, names):
    calls = []
    for nm in names:
        def spy(*a, _nm=nm, **k):
            calls.append(_nm)
            raise _Taken(_nm)
        monkeypatch.setattr(I, nm, spy)
    return calls


@needs_numba
def test_euler_fused_gate_placements(monkeypatch):
    """Euler's fused 3-D kernel (leastsq u/v) is gated on the placement on
    its own: ``"layer"`` and ``"layer_linear"`` (element layout: the
    depth-linear twin, ``tests/test_integrators_depth_linear_elements.py``)
    take the runner; with the entry check bypassed, a node-layout
    ``"level"`` field falls to the unfused path, whose leastsq entry
    refuses it."""
    mesh = _deep_mesh()
    calls = _spies(monkeypatch, ("_run_fused_euler_3d",))
    monkeypatch.setattr(I, "_check_depth_linear_entry",
                        lambda *a, **k: None)
    n_elem = mesh.elem_nodes.shape[0]
    lay = _layer_field(0.004, layout="element", n_elem=n_elem)
    lin = dataclasses.replace(lay, uv_vertical="layer_linear")
    for vel in (lay, lin):
        calls.clear()
        with pytest.raises(_Taken):
            I.euler_step(_state(), vel, mesh, 600.0, 1.0)
        assert calls == ["_run_fused_euler_3d"]
    calls.clear()
    with pytest.raises(ValueError, match=r"^interp_leastsq: depth-linear"):
        I.euler_step(_state(), _deep_field("level", 0.004), mesh, 600.0, 1.0)
    assert calls == []


# ---------------------------------------------------------------------------
# Layer runs unchanged
# ---------------------------------------------------------------------------

class _OnlyPlacement:
    """Answers ``uv_vertical = "layer"``; any other attribute read fails."""

    def __init__(self):
        object.__setattr__(self, "_reads", [])

    def __getattribute__(self, name):
        if name == "_reads":
            return object.__getattribute__(self, name)
        object.__getattribute__(self, "_reads").append(name)
        if name == "uv_vertical":
            return "layer"
        raise AssertionError(f"layer entry check read {name!r}")


class _Untouchable:
    def __getattribute__(self, name):
        raise AssertionError(f"layer entry check touched {name!r}")

    def __getitem__(self, i):
        raise AssertionError("layer entry check indexed recon")


@pytest.mark.parametrize("integrator", ["euler", "rk2", "rk4", "rk45"])
def test_entry_check_layer_reads_placements_only(integrator):
    """A ``"layer"`` step with ``"layer"`` companions reads the placement
    of each field it receives — once — and nothing else: no data of any
    field, nothing of the mesh, nothing of the bundle but its snapshots."""
    vel = _OnlyPlacement()
    assert I._check_depth_linear_entry(
        vel, _Untouchable(), "leastsq", integrator) is None
    assert vel._reads == ["uv_vertical"]
    vel, nxt = _OnlyPlacement(), _OnlyPlacement()
    snaps = [_OnlyPlacement() for _ in range(4)]
    recon = (snaps, _Untouchable(), _Untouchable(), _Untouchable())
    assert I._check_depth_linear_entry(
        vel, _Untouchable(), "leastsq", integrator, vel_next=nxt,
        recon=recon) is None
    reads = [probe._reads for probe in [vel, nxt] + snaps]
    assert reads == [["uv_vertical"]] * 6


@pytest.mark.parametrize("uv_interp,layout,with_elems", [
    ("leastsq", "element", False),
    ("edge_leastsq", "element", False), ("edge_leastsq", "element", True),
    ("barycentric", "element", False), ("barycentric", "element", True),
    ("barycentric", "node", False), ("barycentric", "node", True)])
def test_interp_uv_at_zs_ignored_by_layer_interpolators(uv_interp, layout,
                                                        with_elems):
    """``zs`` reaches only depth-linear fields: every layer interpolator
    gives the same bits with any stage depth, NaN included (``elems`` for
    the interpolators that take them)."""
    assert not with_elems or uv_interp in ip.UV_INTERP_USES_ELEMS
    mesh = _deep_mesh()
    vel = _layer_field(layout=layout, n_elem=mesh.elem_nodes.shape[0])
    st = _state()
    lons, lats = st.lon + 0.01, st.lat - 0.01       # the stage positions
    nn = find_nearest_nodes(mesh, lons, lats)
    elems = None
    if with_elems:
        e, _ = find_containing_elements(mesh, lons, lats, nn)
        elems = np.asarray(e, dtype=np.int32)
    fn = ip.get_interpolator(uv_interp)
    mask = np.array([True, True, False])
    layers = np.array([0, 2, 3], dtype=np.int32)
    ref = I._interp_uv_at(fn, st, mesh, vel, lons, lats, mask, nn, 1.0,
                          layers, elems=elems)
    for zs in (np.full(3, np.nan), np.array([3.0, 33.0, 1e9])):
        got = I._interp_uv_at(fn, st, mesh, vel, lons, lats, mask, nn, 1.0,
                              layers, elems=elems, zs=zs)
        for a, b, nm in zip(got, ref, ("U", "V", "ok")):
            assert_bitwise(a, b, nm)


#: golden cases through every changed site: the Euler fused gate, the RK2
#: fused gate (lerp, PCHIP), RK2/RK4 unfused stage depths, the sigma z_mid
_GOLDEN_SUBSET = ["fesom2_elem_euler_fused", "fesom2_elem_rk2_unfused_lerp",
                  "fesom2_elem_rk2_unfused_lerp_bwd",
                  "fesom2_node_rk2_fused_lerp",
                  "fesom2_node_rk2_fused_pchip_bwd",
                  "fesom2_node_rk4_unfused", "fesomc_sigma_rk2",
                  "healpix_3d_rk2"]


@pytest.mark.parametrize("name", _GOLDEN_SUBSET)
def test_layer_runs_unchanged(name, tmp_path):
    """Layer runs through the changed sites equal main's goldens (bitwise
    in the generating environment; the full set is
    ``tests/test_golden_3d_runs.py``)."""
    pytest.importorskip("healpy")
    from tests._gen_golden_3d import run_case
    from tests.test_golden_3d_runs import BITWISE, GOLDEN_DIR, _compare
    _compare(run_case(name, workdir=tmp_path), GOLDEN_DIR / f"{name}.npz",
             BITWISE)
