"""Depth-linear u/v on element layout in the integrators.

The step functions take a depth-linear field by the layout table of
``_check_depth_linear_entry``: node layout with ``uv_interp="barycentric"``
only, element layout (``uv_vertical="layer_linear"``) with ``leastsq``,
``edge_leastsq`` or ``barycentric``; rk45, a mesh neither z-level nor sigma
and 2-D fields stay refused (sigma meshes:
``tests/test_integrators_depth_linear_sigma.py``).  Every 3-D method reads
each stage at its own depth: Euler the
entry depth (fused and unfused), RK2 the entry depth and then ``z_mid``,
RK4 ``z``, ``z2``, ``z3``, ``z4`` (``_interp_uv_at(..., zs=...)``), linear
and PCHIP in time.  The fused 3-D Euler kernel has a depth-linear twin for
element ``leastsq`` (``_fused_euler_step_3d_dl``: the layer kernel with the
layer lookup replaced by ``_interp_leastsq_core_dl`` at the particle depth),
bit for bit the unfused step; element ``"layer"`` fields keep the layer
kernels and never reach depth-linear code.

All synthetic: the 5x5-node mesh of ``test_interp_depth_linear_elements``
(32 triangles, levels ``[0, 10, 30, 60, 100, 150]`` m, layer centres
``[5, 20, 45, 80, 125]``) with the element bottoms set by hand as the
FESOM2 reader declares them.  The hand rule of the node tests,
``u = a + b*clip(z) (+ e*t)``, stored in every element column, gives every
expected position as a hand-written stage combination.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kelp.core.io import LazyLerpField, VelocityField
from kelp.core.mesh import (
    find_containing_elements,
    find_nearest_nodes,
    set_exact_nearest,
)
from kelp.core.particles import ParticleStatus, make_state
from kelp.numerics import integrators as I
from kelp.numerics import interpolate as ip
from kelp.numerics import vertical_uv as vu
from kelp.numerics.temporal_reconstruct import reconstruct_pchip

from tests._bitwise import assert_bitwise
from tests.test_integrators_depth_linear import (
    A_U, A_V, B_U, B_V, C_RK4, E_U, E_V, _assert_not_step, _assert_step,
    _hand_all,
)
from tests.test_interp_depth_linear_elements import (
    CENTRE, CENTRE_PTS, LEVELS, P, _geo_last, _layer_fit, _mesh, _ring,
    _values_at,
)

K = len(LEVELS)
N_ELEM = 32

#: the element interpolators of a depth-linear field
INTERPS = ["leastsq", "edge_leastsq", "barycentric"]

STEPS = {"euler": I.euler_step, "rk2": I.rk2_step, "rk4": I.rk4_step}

#: three particles near interior nodes (6, 17, 13), every stage inside;
#: entry depths so that the stages cross both clip ends: forward the third
#: one sinks below the last centre (125 m), backward the first one rises
#: above the first (5 m)
LON0 = np.array([4.62, 5.13, 5.37])
LAT0 = np.array([4.71, 5.28, 4.84])
Z0 = np.array([12.0, 47.5, 118.0])

#: step and uniform w: a stage half a step on moves 7.2 m
DT = 1800.0
W0 = 0.008

needs_numba = pytest.mark.skipif(not I.HAS_NUMBA,
                                 reason="fused kernels need numba")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _etab(mesh):
    return vu.depth_points(mesh, "layer_linear", "element")


def _elem_field(mesh, fu, fv, w, *, below=np.nan):
    """Element-layout ``"layer_linear"`` field: ``u[k, e] = fu(e, p_k)`` (v
    likewise) at the points ``k <= last_e`` of the mesh's element table,
    ``below`` under them; ``w`` on the nodes' levels."""
    dp = _etab(mesh)
    m = dp.p.size
    u = np.full((m, mesh.n_elements), below, dtype=np.float64)
    v = np.full((m, mesh.n_elements), below, dtype=np.float64)
    for e in range(mesh.n_elements):
        k = int(dp.last[e]) + 1
        u[:k, e] = fu(e, dp.p[:k])
        v[:k, e] = fv(e, dp.p[:k])
    return VelocityField(u=u, v=v, eta=np.zeros(mesh.n_nodes),
                         w=np.asarray(w, dtype=np.float64),
                         uv_layout="element", uv_vertical="layer_linear")


def _g(t, quad=False):
    """The time profile of the fixtures: linear, or quadratic (where a
    PCHIP reconstruction and a linear blend differ)."""
    return t * t / 3600.0 if quad else t


def _rule_field(mesh, w0=W0, t=0.0, quad=False):
    """The hand rule in every element column: ``u = A_U + B_U*p_k +
    E_U*g(t)`` (v likewise), uniform ``w = w0``."""
    tu, tv = E_U * _g(t, quad), E_V * _g(t, quad)
    return _elem_field(mesh, lambda e, p: A_U + B_U * p + tu,
                       lambda e, p: A_V + B_V * p + tv,
                       np.full((K, mesh.n_nodes), float(w0)))


def _uv_rule(t_k=(0.0, 0.0, 0.0, 0.0)):
    """Hand u/v of the rule at stage ``k``: every element 150 m deep, so
    each reads ``a + b*clip(z, p[0], p[-1]) + e*t_k``."""
    def uv(k, x, y, z):
        zc = min(max(z, P[0]), P[-1])
        return (A_U + B_U * zc + E_U * t_k[k],
                A_V + B_V * zc + E_V * t_k[k])
    return uv


def _state(mesh=None, *, walk=False, lon=LON0, lat=LAT0, z=Z0):
    st = make_state(np.array(lon, dtype=np.float64),
                    np.array(lat, dtype=np.float64),
                    np.array(z, dtype=np.float64))
    if walk:
        nn = find_nearest_nodes(mesh, st.lon, st.lat)
        e, ok = find_containing_elements(mesh, st.lon, st.lat, nn)
        assert ok.all()
        st = st.replace(elem=np.asarray(e, dtype=np.int32))
    return st


def _payload(mesh, mode, td):
    """``(vel, kwargs, stage times)``: steady, or a lazy blend between the
    records at 0 s and 3600 s (as the tracking layer builds it) moving
    towards the next record in the direction of the step."""
    if mode == "steady":
        return _rule_field(mesh), {}, (0.0,) * 4
    t_s = 900.0 if td > 0 else 2700.0
    a, b = _rule_field(mesh, t=0.0), _rule_field(mesh, t=3600.0)
    vel = LazyLerpField(a, b, t_s / 3600.0)
    nxt, dt_to_next = (b, 3600.0 - t_s) if td > 0 else (a, t_s)
    t_k = tuple(t_s + td * c * DT for c in C_RK4)
    return vel, dict(vel_next=nxt, dt_to_next=dt_to_next), t_k


class _Count:
    """Call-through spy of a module attribute."""

    def __init__(self, monkeypatch, module, name):
        self.n = 0
        real = getattr(module, name)

        def spy(*a, **k):
            self.n += 1
            return real(*a, **k)
        monkeypatch.setattr(module, name, spy)


# ---------------------------------------------------------------------------
# The hand rule through every method
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("w_interp", ["nearest", "leastsq", "barycentric"])
@pytest.mark.parametrize("mode", ["steady", "lerp"])
@pytest.mark.parametrize("td", [1.0, -1.0])
@pytest.mark.parametrize("uv_interp", INTERPS)
def test_euler_rule(monkeypatch, uv_interp, td, mode, w_interp):
    """Euler reads u/v at the entry position, depth and time:
    ``dlon = dt*td*(a + b*clip(z0) + e*t0) / (cos(lat0) M)``, to 1e-12
    relative; the recorded velocity is that value.  Element ``leastsq``
    with ``w_interp`` nearest / leastsq runs the fused depth-linear twin,
    everything else the unfused step."""
    mesh = _mesh()
    vel, kw, t_k = _payload(mesh, mode, td)
    st = _state()
    fused = _Count(monkeypatch, I, "_fused_euler_step_3d_dl")
    out = I.euler_step(st, vel, mesh, DT, td, uv_interp=uv_interp,
                       w_interp=w_interp, **kw)
    dlon, dlat, u1, v1 = _hand_all("euler", st, DT, td, W0, _uv_rule(t_k))
    _assert_step(out, st, dlon, dlat)
    np.testing.assert_allclose(out.u, u1, rtol=1e-13)
    np.testing.assert_allclose(out.v, v1, rtol=1e-13)
    want = int(I.HAS_NUMBA and uv_interp == "leastsq"
               and w_interp in ("nearest", "leastsq"))
    assert fused.n == want


@pytest.mark.parametrize("walk", [False, True], ids=["scan", "walk"])
@pytest.mark.parametrize("mode", ["steady", "lerp"])
@pytest.mark.parametrize("td", [1.0, -1.0])
@pytest.mark.parametrize("uv_interp", INTERPS)
def test_rk2_rule(uv_interp, td, mode, walk):
    """Stage 2 reads u/v at ``z_mid = z0 + dt/2 * W1`` and ``t + dt/2``
    (clipped to the column's centres), to 1e-12 relative; the same formula
    with ``w = 0`` is a different horizontal answer."""
    mesh = _mesh()
    vel, kw, t_k = _payload(mesh, mode, td)
    st = _state(mesh, walk=walk)
    out = I.rk2_step(st, vel, mesh, DT, td, uv_interp=uv_interp,
                     w_interp="barycentric", **kw)
    dlon, dlat, u1, _ = _hand_all("rk2", st, DT, td, W0, _uv_rule(t_k[:2]))
    _assert_step(out, st, dlon, dlat)
    np.testing.assert_allclose(out.u, u1, rtol=1e-13)
    dlon0, dlat0, _, _ = _hand_all("rk2", st, DT, td, 0.0,
                                   _uv_rule(t_k[:2]))
    _assert_not_step(out, st, dlon0, dlat0)
    np.testing.assert_allclose(out.z, np.clip(Z0 + DT * td * W0, 0.0, 150.0),
                               rtol=1e-14)


@pytest.mark.parametrize("walk", [False, True], ids=["scan", "walk"])
@pytest.mark.parametrize("mode", ["steady", "lerp"])
@pytest.mark.parametrize("td", [1.0, -1.0])
@pytest.mark.parametrize("uv_interp", INTERPS)
def test_rk4_rule(uv_interp, td, mode, walk):
    """RK4 stages 2-4 read u/v at ``z2 = z3 = z0 + dt/2 W``,
    ``z4 = z0 + dt W`` and their times, to 1e-12 relative; ``w = 0`` gives
    a different answer."""
    mesh = _mesh()
    vel, kw, t_k = _payload(mesh, mode, td)
    st = _state(mesh, walk=walk)
    out = I.rk4_step(st, vel, mesh, DT, td, uv_interp=uv_interp,
                     w_interp="nearest", **kw)
    dlon, dlat, u1, _ = _hand_all("rk4", st, DT, td, W0, _uv_rule(t_k))
    _assert_step(out, st, dlon, dlat)
    np.testing.assert_allclose(out.u, u1, rtol=1e-13)
    dlon0, dlat0, _, _ = _hand_all("rk4", st, DT, td, 0.0, _uv_rule(t_k))
    _assert_not_step(out, st, dlon0, dlat0)


#: PCHIP records (s) and the step's start per direction (central bracket)
RECORDS = (0.0, 3600.0, 7200.0, 10800.0)
PCHIP_START = {1.0: 3600.0, -1.0: 7200.0}


@pytest.mark.parametrize("td", [1.0, -1.0])
@pytest.mark.parametrize("uv_interp", INTERPS)
@pytest.mark.parametrize("integrator", ["rk2", "rk4"])
def test_pchip_in_time(integrator, uv_interp, td):
    """``temporal_reconstruction: pchip`` on records quadratic in time
    (``u = a + b*p + e*t**2/3600``), where the 4-snapshot PCHIP differs
    from the linear blend by far more than 1e-6.  The expected stage value
    is computed independently: ``reconstruct_pchip`` of the same records at
    the stage time, then the depth rule (``np.interp`` of that column at the
    clipped stage depth); positions to 1e-12 relative.  The linear-in-time
    answer is far from the step."""
    mesh = _mesh()
    snaps = [_rule_field(mesh, t=t, quad=True) for t in RECORDS]
    t0 = PCHIP_START[td]
    c = C_RK4 if integrator == "rk4" else C_RK4[:2]
    t_k = tuple(t0 + td * ck * DT for ck in c)
    cols, lin = {}, {}
    for t in set(t_k):
        rf = reconstruct_pchip(snaps, RECORDS, t)
        assert np.all(rf.u == rf.u[:, :1]) and np.all(rf.v == rf.v[:, :1])
        cols[t] = (rf.u[:, 0].copy(), rf.v[:, 0].copy())
        a = (t - RECORDS[1]) / (RECORDS[2] - RECORDS[1])
        lin[t] = ((1.0 - a) * snaps[1].u[:, 0] + a * snaps[2].u[:, 0],
                  (1.0 - a) * snaps[1].v[:, 0] + a * snaps[2].v[:, 0])
    mid = t0 + td * 0.5 * DT
    assert np.all(np.abs(cols[mid][0] - lin[mid][0]) > 1e-6)
    assert np.all(np.abs(cols[mid][1] - lin[mid][1]) > 1e-6)

    def by_cols(table):
        def uv(k, x, y, z):
            cu, cv = table[t_k[k]]
            zc = min(max(z, P[0]), P[-1])
            return float(np.interp(zc, P, cu)), float(np.interp(zc, P, cv))
        return uv

    vel, nxt = (snaps[1], snaps[2]) if td > 0 else (snaps[2], snaps[1])
    recon = (snaps, list(RECORDS), t0, int(td))
    st = _state()
    out = STEPS[integrator](st, vel, mesh, DT, td, uv_interp=uv_interp,
                            w_interp="nearest", vel_next=nxt,
                            dt_to_next=3600.0, recon=recon)
    dlon, dlat, u1, _ = _hand_all(integrator, st, DT, td, W0, by_cols(cols))
    _assert_step(out, st, dlon, dlat)
    np.testing.assert_allclose(out.u, u1, rtol=1e-13)
    dlon_l, dlat_l, _, _ = _hand_all(integrator, st, DT, td, W0,
                                     by_cols(lin))
    _assert_not_step(out, st, dlon_l, dlat_l)


# ---------------------------------------------------------------------------
# Fused Euler twin == unfused, bit for bit
# ---------------------------------------------------------------------------

_FIELDS = ("lon", "lat", "z", "status", "nearest_node", "u", "v", "w",
           "failure_reason")

#: element beds of the parity meshes (9 m .. 150 m)
_BEDS = [9.0, 25.0, 30.0, 40.0, 60.0, 90.0, 150.0]


def _parity_mesh(case):
    """Unequal element beds, prefixes declared shorter than the beds allow
    (a wet element below its last point: a partial bottom layer), node beds
    the deepest element around each node (as FESOM2 defines them).

    ``"singular_wall"``: the centre node's ring has 11, 12, 13 wet (150 m)
    with their centres moved onto one parallel and 18, 19, 20 dry (30 m):
    at 80 m three wet members whose fit is singular.  ``"cut"``: the mixed
    mesh after a ``max_levels`` cut to 4 levels (3 points: elements wet
    below 45 m read their last kept point)."""
    rng = np.random.default_rng(61)
    bed = rng.choice(_BEDS, size=N_ELEM)
    if case == "singular_wall":
        bed[[11, 12, 13]] = 150.0
        bed[[18, 19, 20]] = 30.0
    decl = np.minimum(_geo_last(bed), rng.integers(0, len(P), size=N_ELEM))
    if case == "singular_wall":
        decl[[11, 12, 13]] = _geo_last(bed[[11, 12, 13]])
    mesh = _mesh(bed, decl=decl.astype(np.int32))
    mesh.depth = np.array([bed[_ring(mesh, n)].max()
                           for n in range(mesh.n_nodes)])
    if case == "singular_wall":
        lat = mesh.elem_center_lat.copy()
        lat[[11, 12, 13]] = 4.75
        mesh.elem_center_lat = lat
    if case == "cut":
        mesh.sigma = mesh.sigma[:4]
    return mesh


def _parity_field(mesh, seed):
    """Per-element profiles (every element different), NaN below each
    element's last usable point, random w on the levels."""
    rng = np.random.default_rng(seed)
    cu = rng.uniform(0.1, 0.6, N_ELEM)
    cv = rng.uniform(-0.6, -0.1, N_ELEM)
    bu = rng.uniform(-3e-3, 3e-3, N_ELEM)
    bv = rng.uniform(-3e-3, 3e-3, N_ELEM)
    k = mesh.sigma.size
    return _elem_field(mesh, lambda e, p: cu[e] + bu[e] * p,
                       lambda e, p: cv[e] + bv[e] * p,
                       1e-3 * rng.standard_normal((k, mesh.n_nodes)))


def _parity_state(mesh, case):
    """Particles inside and just outside the mesh at depths around the
    beds, below every bed, NaN; two dead.  ``"singular_wall"`` adds the
    centre ring's points at 80 m."""
    rng = np.random.default_rng(67)
    n = 300
    lon = rng.uniform(3.9, 6.1, n)
    lat = rng.uniform(3.9, 6.1, n)
    z = rng.choice([-1.0, 5.0, 12.0, 25.0, 30.0, 33.3, 45.0, 52.5, 60.0,
                    71.0, 90.0, 110.0, 150.0, 400.0, np.nan], size=n)
    if case == "singular_wall":
        pts = np.asarray(CENTRE_PTS)
        lon = np.r_[lon, pts[:, 0]]
        lat = np.r_[lat, pts[:, 1]]
        z = np.r_[z, np.full(len(pts), 80.0)]
    st = make_state(lon, lat, z)
    st.status[[3, 17]] = ParticleStatus.OUTSIDE
    return st.replace(nearest_node=find_nearest_nodes(mesh, st.lon, st.lat),
                      failure_reason=np.zeros(st.n_particles, dtype=np.int8))


def _regimes(mesh, st):
    """Counts of alive particles by regime at the entry (the fused twin's
    only evaluation): every ring element wet, a wall with >= 3 wet, one or
    two wet, none wet; particles whose wet ring has an element read below
    its last usable point (a partial bottom layer or a cut), and particles
    below the deepest kept level with a wet ring element (the cut)."""
    dp = _etab(mesh)
    z_cut = mesh.z_levels[-1]
    nn = find_nearest_nodes(mesh, st.lon, st.lat)
    out = dict(all_wet=0, wall=0, few_wet=0, all_dry=0, below_last=0,
               below_cut=0)
    for i in np.flatnonzero(st.alive_mask):
        zq = np.inf if np.isnan(st.z[i]) else st.z[i]
        ring = _ring(mesh, nn[i])
        wet = [e for e in ring if zq <= dp.bed[e]]
        out["all_wet" if len(wet) == len(ring) else "all_dry" if not wet
            else "few_wet" if len(wet) < 3 else "wall"] += 1
        if any(zq > dp.p[dp.last[e]] for e in wet):
            out["below_last"] += 1
        if wet and zq > z_cut:
            out["below_cut"] += 1
    return out


@needs_numba
@pytest.mark.parametrize("exact", [True, False], ids=["exact", "legacy"])
@pytest.mark.parametrize("w_interp", ["nearest", "leastsq"])
@pytest.mark.parametrize("mode", ["discrete", "linear"])
@pytest.mark.parametrize("td", [1.0, -1.0])
@pytest.mark.parametrize("case", ["mixed", "singular_wall", "cut"])
def test_fused_euler_equals_unfused(monkeypatch, case, td, mode, w_interp,
                                    exact):
    """``euler_step`` on element ``leastsq`` runs the fused depth-linear
    twin, and its state equals ``_euler_step_unfused`` bit for bit (every
    field, dtype and bytes) — discrete records and a lazy blend in time,
    both directions, ``w_interp`` nearest and leastsq, the exact anchor on
    and off — over particles in every regime: dry neighbours, one or two
    wet, the singular-with-a-wall area mean, the all-dry containing
    element, a partial bottom layer, a ``max_levels`` cut, NaN depths and
    particles outside the mesh."""
    mesh = _parity_mesh(case)
    set_exact_nearest(mesh, exact)
    a, b = _parity_field(mesh, 71), _parity_field(mesh, 72)
    if mode == "discrete":
        vel = a
    else:
        vel = LazyLerpField(a, b, 0.35)
    kw = dict(vel_next=b, dt_to_next=7200.0)
    st = _parity_state(mesh, case)
    reg = _regimes(mesh, st)
    if case == "mixed":
        assert reg["below_cut"] == 0
        reg.pop("below_cut")
        assert min(reg.values()) > 0, reg
    elif case == "cut":
        assert reg["below_cut"] > 0 and reg["wall"] > 0, reg
    else:
        dp = _etab(mesh)
        wet = [e for e in _ring(mesh, CENTRE) if 80.0 <= dp.bed[e]]
        assert wet == [11, 12, 13]
        uz, vz = _values_at(mesh, a, 80.0)
        for x, y in CENTRE_PTS:
            _, _, ok = _layer_fit(mesh, x, y, CENTRE, td, uz, vz,
                                  ring=wet)
            assert not ok                  # the wet members' fit: singular
    fused = _Count(monkeypatch, I, "_fused_euler_step_3d_dl")
    layer = _Count(monkeypatch, I, "_fused_euler_step_3d")
    got = I.euler_step(st, vel, mesh, 3600.0, td, uv_interp="leastsq",
                       w_interp=w_interp, **kw)
    assert (fused.n, layer.n) == (1, 0)
    ref = I._euler_step_unfused(st, vel, mesh, 3600.0, td, 0, True,
                                w_interp=w_interp, uv_interp="leastsq",
                                **kw)
    for f in _FIELDS:
        assert_bitwise(getattr(got, f), getattr(ref, f), f)
    # particles failing in the step (outside the mesh) and moving ones
    assert np.any(got.failure_reason != 0)
    assert np.any((got.status == ParticleStatus.ALIVE) & (got.lon != st.lon))


# ---------------------------------------------------------------------------
# The fused runner on its own
# ---------------------------------------------------------------------------

@needs_numba
@pytest.mark.parametrize("where", ["vel", "lazy_vel_b"])
def test_fused_euler_runner_record_check(monkeypatch, where):
    """A direct ``_run_fused_euler_3d`` call checks the field its twin
    reads (a lazy field through its parents): a NaN at a usable point is
    refused, naming the field; a field already checked for the mesh's
    table is not scanned again."""
    mesh = _mesh()
    a, b = _rule_field(mesh), _rule_field(mesh, t=3600.0)
    vel = LazyLerpField(a, b, 0.4) if where == "lazy_vel_b" else a
    bad = b if where == "lazy_vel_b" else a
    label = r"vel\.vel_b" if where == "lazy_vel_b" else "vel"
    keep = bad.u[2, 7]
    bad.u[2, 7] = np.nan
    with pytest.raises(ValueError, match=(
            rf"non-finite at element 7, point 2 .*in record {label} "
            r"\(_run_fused_euler_3d\)")):
        I._run_fused_euler_3d(_state(), vel, mesh, DT, 1.0,
                              w_interp="nearest")
    bad.u[2, 7] = keep
    I._run_fused_euler_3d(_state(), vel, mesh, DT, 1.0, w_interp="nearest")
    scans = []
    real = I.check_depth_linear_record

    def spy(vel_, mesh_, *, record=""):
        scans.append(record)
        return real(vel_, mesh_, record=record)
    monkeypatch.setattr(I, "check_depth_linear_record", spy)
    I._run_fused_euler_3d(_state(), vel, mesh, DT, 1.0, w_interp="nearest")
    assert scans == []


@needs_numba
@pytest.mark.parametrize("uv_interp", INTERPS)
@pytest.mark.parametrize("method", list(STEPS))
def test_cavities_refused_at_direct_entry(method, uv_interp):
    """A step called directly, without ``run_tracking``, refuses an element
    mesh with ice-shelf cavities (``elem_ulevels`` above 1) as the run
    does, before any particle moves."""
    mesh = _mesh()
    mesh.metadata["elem_ulevels"] = np.ones(mesh.n_elements, dtype=np.int32)
    mesh.metadata["elem_ulevels"][5] = 2
    with pytest.raises(ValueError, match=(
            r"^ice-shelf cavities are not supported by mode\.uv_vertical: "
            r"linear on element data yet: .* \(first: element 5, "
            r"ulevels 2\)")):
        STEPS[method](_state(), _rule_field(mesh), mesh, DT, 1.0,
                      uv_interp=uv_interp, w_interp="nearest")


def test_fused_euler_runner_refusals():
    """The runner serves element-layout ``"layer_linear"`` fields only: a
    node-layout depth-linear field is refused (``leastsq`` does not
    evaluate it), and u/v replaced after the check by arrays of another
    shape never reach the kernel."""
    mesh = _mesh()
    n = mesh.n_nodes
    node = VelocityField(u=np.full((K - 1, n), 0.2),
                         v=np.full((K - 1, n), -0.1), eta=np.zeros(n),
                         w=np.zeros((K, n)), uv_layout="node",
                         uv_vertical="layer_linear")
    with pytest.raises(ValueError, match=(
            r"^_run_fused_euler_3d: depth-linear horizontal velocity "
            r"\(uv_vertical='layer_linear'\) on node layout")):
        I._run_fused_euler_3d(_state(), node, mesh, DT, 1.0)
    vel = _rule_field(mesh)
    I._run_fused_euler_3d(_state(), vel, mesh, DT, 1.0)       # stamps
    vel.u = vel.u[:3]
    with pytest.raises(ValueError, match=(
            r"^_run_fused_euler_3d: depth-linear u/v must be \(M, E\) = "
            r"\(5, 32\), got u \(3, 32\)")):
        I._run_fused_euler_3d(_state(), vel, mesh, DT, 1.0)


# ---------------------------------------------------------------------------
# Layer runs unchanged (dispatch)
# ---------------------------------------------------------------------------

#: depth-linear entry points that a layer run must never reach
_DL_FUNCS = ((I, "_fused_euler_step_3d_dl"),
             (ip, "_interp_leastsq_dl"), (ip, "_interp_edge_leastsq_dl"),
             (ip, "_interp_barycentric_dl"),
             (ip, "_loop_interp_leastsq_3d_dl"),
             (ip, "_loop_interp_edge_lsq_3d_dl"),
             (ip, "_loop_interp_bary_elem_dl"),
             (I, "depth_points"), (ip, "depth_points"),
             (vu, "depth_points"), (I, "check_depth_linear_record"),
             (ip, "check_depth_linear_record"), (I, "is_checked"),
             (ip, "is_checked"))

#: the layer kernel each element interpolator runs (unfused); Euler's
#: fused route runs ``_fused_euler_step_3d``
_LAYER_KERNEL = {"leastsq": (ip, "_loop_interp_leastsq_3d"),
                 "edge_leastsq": (ip, "_loop_interp_edge_lsq_3d"),
                 "barycentric": (ip, "_loop_interp_bary_3d")}


def _layer_elem_field(mesh, seed):
    rng = np.random.default_rng(seed)
    return VelocityField(
        u=rng.uniform(0.1, 0.5, (K - 1, N_ELEM)),
        v=rng.uniform(-0.3, 0.3, (K - 1, N_ELEM)),
        eta=np.zeros(mesh.n_nodes),
        w=np.full((K, mesh.n_nodes), 1e-3), uv_layout="element")


def _forbid(monkeypatch):
    def make(name):
        def spy(*a, **k):
            raise AssertionError(f"a layer run reached {name}")
        return spy
    for mod, name in _DL_FUNCS:
        monkeypatch.setattr(mod, name, make(name))


@needs_numba
@pytest.mark.parametrize("mode", ["steady", "lerp", "pchip"])
@pytest.mark.parametrize("uv_interp", INTERPS)
@pytest.mark.parametrize("integrator", ["euler", "rk2", "rk4"])
def test_layer_runs_unchanged(monkeypatch, integrator, uv_interp, mode):
    """Element ``"layer"`` fields run the layer kernels — Euler ``leastsq``
    the fused ``_fused_euler_step_3d``, everything else the interpolators'
    layer loops — and never a depth-linear function, the depth-point table
    or the record check (each replaced by a spy that fails)."""
    if mode == "pchip" and integrator == "euler":
        pytest.skip("Euler ignores the PCHIP bundle")
    mesh = _mesh()
    snaps = [_layer_elem_field(mesh, s) for s in range(4)]
    kw = {}
    vel = snaps[1]
    if mode == "lerp":
        vel = LazyLerpField(snaps[1], snaps[2], 0.25)
        kw = dict(vel_next=snaps[2], dt_to_next=2700.0)
    elif mode == "pchip":
        kw = dict(vel_next=snaps[2], dt_to_next=3600.0,
                  recon=(snaps, list(RECORDS), 3600.0, 1))
    _forbid(monkeypatch)
    fused = _Count(monkeypatch, I, "_fused_euler_step_3d")
    loops = {k: _Count(monkeypatch, *v) for k, v in _LAYER_KERNEL.items()}
    out = STEPS[integrator](_state(), vel, mesh, DT, 1.0,
                            uv_interp=uv_interp, w_interp="nearest", **kw)
    assert np.all(out.status == ParticleStatus.ALIVE)
    if integrator == "euler" and uv_interp == "leastsq":
        assert fused.n == 1 and loops["leastsq"].n == 0
    else:
        n_stage = {"euler": 1, "rk2": 2, "rk4": 4}[integrator]
        assert fused.n == 0 and loops[uv_interp].n == n_stage
    assert sum(c.n for k, c in loops.items() if k != uv_interp) == 0


# ---------------------------------------------------------------------------
# The entry table
# ---------------------------------------------------------------------------

_PRE = (r"^{}_step: depth-linear horizontal velocity "
        r"\(uv_vertical='layer_linear'\) ")


def _node_field(mesh):
    n = mesh.n_nodes
    return VelocityField(u=np.full((K - 1, n), 0.2),
                         v=np.full((K - 1, n), -0.1), eta=np.zeros(n),
                         w=np.full((K, n), 1e-3), uv_layout="node",
                         uv_vertical="layer_linear")


def test_entry_table():
    """``_check_depth_linear_entry`` on a z-level mesh: node layout with
    ``barycentric`` only; element layout with ``leastsq``, ``edge_leastsq``
    and ``barycentric`` (each accepted and record-checked) and nothing
    else; an unknown layout refused; rk45, a 2-D field and a mesh neither
    z-level nor sigma refused for element fields with every interpolator
    (the rk45 message advises euler, rk2 or rk4; for a node field with
    ``uv_interp='barycentric'``); element fields on a mesh without the
    element bottoms refused by the record check."""
    chk = I._check_depth_linear_entry
    mesh = _mesh()
    # node layout: barycentric only
    node = _node_field(mesh)
    for other in ("leastsq", "edge_leastsq"):
        with pytest.raises(ValueError, match=_PRE.format("rk2") + (
                r"on node layout \(uv_layout='node'\) needs "
                rf"uv_interp='barycentric', got uv_interp='{other}'$")):
            chk(node, mesh, other, "rk2")
    assert chk(node, mesh, "barycentric", "rk2") is None
    assert vu.is_checked(node, mesh)
    with pytest.raises(ValueError, match=_PRE.format("rk45") + (
            r"is not supported by the rk45 integrator \(2-D only in this "
            r"version\); use euler, rk2 or rk4 with "
            r"uv_interp='barycentric'$")):
        chk(node, mesh, "barycentric", "rk45")
    # element layout: the three interpolators
    for interp in INTERPS:
        for name in ("euler", "rk2", "rk4"):
            el = _rule_field(mesh)
            assert not vu.is_checked(el, mesh)
            assert chk(el, mesh, interp, name) is None
            assert vu.is_checked(el, mesh)
    el = _rule_field(mesh)
    with pytest.raises(ValueError, match=_PRE.format("euler") + (
            r"on element layout \(uv_layout='element'\) needs uv_interp "
            r"'leastsq', 'edge_leastsq' or 'barycentric', got "
            r"uv_interp='nearest'$")):
        chk(el, mesh, "nearest", "euler")
    odd = SimpleNamespace(uv_vertical="layer_linear", uv_layout="pixel",
                          u=el.u, v=el.v, w=el.w, eta=el.eta)
    with pytest.raises(ValueError, match=_PRE.format("rk4") + (
            r"needs uv_layout 'node' or 'element', got 'pixel'$")):
        chk(odd, mesh, "leastsq", "rk4")
    # still refused for element fields: rk45, 2-D, a mesh neither z-level
    # nor sigma
    other = _mesh()
    other.vertical_coord = "hybrid"
    flat = VelocityField(u=el.u, v=el.v, eta=el.eta, w=None,
                         uv_layout="element", uv_vertical="layer_linear")
    for interp in INTERPS:
        # the advice names the integrators that work, and no interpolator
        # (every one in the element row works)
        with pytest.raises(ValueError, match=_PRE.format("rk45")
                           + r"is not supported by the rk45 integrator "
                             r"\(2-D only in this version\); use euler, "
                             r"rk2 or rk4$"):
            chk(el, mesh, interp, "rk45")
        with pytest.raises(ValueError, match=_PRE.format("rk2")
                           + r"needs a z-level or sigma mesh \(vertical_coord "
                             r"'zlevel' or 'sigma'\), got 'hybrid'$"):
            chk(el, other, interp, "rk2")
        with pytest.raises(ValueError, match=_PRE.format("rk4")
                           + r"needs a 3-D field"):
            chk(flat, mesh, interp, "rk4")
    # the element bottoms: required by the element table
    bare = _mesh()
    del bare.metadata["uv_data_last_elem"]
    del bare.metadata["elem_bed"]
    with pytest.raises(ValueError, match=r"needs the element bottoms"):
        chk(_rule_field(mesh), bare, "leastsq", "rk2")


# ---------------------------------------------------------------------------
# NUMBA_DISABLE_JIT: the pure-Python twin equals the pure-Python unfused step
# ---------------------------------------------------------------------------

_NOJIT_SCRIPT = r"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).resolve().parents[0]))
sys.path.insert(0, sys.argv[1])
import numpy as np
from numba.core import config
assert config.DISABLE_JIT, "NUMBA_DISABLE_JIT not honoured"
import kelp.numerics.integrators as I
assert str(Path(I.__file__).resolve()).startswith(
    str(Path(sys.argv[1]).resolve())), I.__file__
import tests.test_integrators_depth_linear_elements as t
from kelp.core.io import LazyLerpField
launches = []
real = I._fused_euler_step_3d_dl
def spy(*a, **k):
    launches.append(1)
    return real(*a, **k)
I._fused_euler_step_3d_dl = spy
rows = {}
for case in ("mixed", "cut"):
    for mode in ("discrete", "linear"):
        launches.clear()
        mesh = t._parity_mesh(case)
        a, b = t._parity_field(mesh, 71), t._parity_field(mesh, 72)
        vel = a if mode == "discrete" else LazyLerpField(a, b, 0.35)
        st = t._parity_state(mesh, case)
        got = I.euler_step(st, vel, mesh, 3600.0, 1.0, uv_interp="leastsq",
                           w_interp="nearest")
        ref = I._euler_step_unfused(st, vel, mesh, 3600.0, 1.0, 0, True,
                                    w_interp="nearest", uv_interp="leastsq")
        differ = [f for f in t._FIELDS
                  if np.asarray(getattr(got, f)).tobytes()
                  != np.asarray(getattr(ref, f)).tobytes()]
        rows[case + "-" + mode] = [differ, len(launches),
                                   np.nan_to_num(got.lon).tolist()]
print(json.dumps(rows))
"""


@needs_numba
def test_fused_euler_numba_disabled_subprocess(tmp_path):
    """With ``NUMBA_DISABLE_JIT=1`` the twin runs as Python and equals the
    unfused step bit for bit there, and agrees with the compiled twin."""
    script = tmp_path / "fused_euler_dl_nojit.py"
    script.write_text(_NOJIT_SCRIPT)
    src = str(Path(__file__).resolve().parents[1] / "src")
    env = dict(os.environ, NUMBA_DISABLE_JIT="1")
    out = subprocess.run([sys.executable, str(script), src],
                         capture_output=True, text=True, timeout=900,
                         env=env)
    assert out.returncode == 0, out.stderr
    rows = json.loads(out.stdout)
    assert len(rows) == 4
    bad = {k: r[0] for k, r in rows.items() if r[0] or r[1] != 1}
    assert not bad, f"no-JIT fused != unfused (fields) or no twin: {bad}"
    for key, (_, _, lon) in rows.items():
        case, mode = key.split("-")
        mesh = _parity_mesh(case)
        a, b = _parity_field(mesh, 71), _parity_field(mesh, 72)
        vel = a if mode == "discrete" else LazyLerpField(a, b, 0.35)
        got = I.euler_step(_parity_state(mesh, case), vel, mesh, 3600.0, 1.0,
                           uv_interp="leastsq", w_interp="nearest")
        np.testing.assert_allclose(np.array(lon, dtype=float),
                                   np.nan_to_num(got.lon), rtol=0.0,
                                   atol=1e-11)
