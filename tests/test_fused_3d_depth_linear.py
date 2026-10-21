"""Fused 3-D RK2 on depth-linear u/v (``uv_vertical`` "level" / "layer_linear").

The fused node-layout barycentric RK2 kernels have depth-linear twins,
``_fused_rk2_step_3d_bary_tinterp_dl`` and ``_fused_rk2_step_3d_bary_pchip_dl``,
over the cores ``_interp_barycentric_core_tinterp_dl`` and
``_interp_barycentric_core_pchip4_dl``.  Per vertex, each stored
``(point, node)`` value is blended in time FIRST (the lerp, or the 4-snapshot
PCHIP), then read linearly in depth between that vertex's bracketing points
(``uv_bracket`` / ``uv_at`` on the blended values), then combined
barycentrically; stage 1 reads the entry depth, stage 2 ``z_mid``.  The
twins reproduce the unfused depth-linear path (``_rk2_step_unfused`` ->
``interp_barycentric`` -> ``_interp_barycentric_core_dl``) bit for bit,
through the anchor heal and the terminal unfused splice too.  The runners
refuse fields of mixed placement and run the record check on every field
the kernel reads.

Fixtures: the HEALPix dual mesh of ``test_fused_3d_bary`` with random u/v on
the data points and NaN at every point below each node's last usable point
(a read there would surface as a NaN velocity); the hole-band structured
mesh of ``test_fused_3d_bary`` for the heal paths; the 3x3 mesh of
``test_interp_depth_linear`` for the hand-valued PCHIP-order case; a
triangle mesh with a trench for the walls at depth (a vertex dry at the
stage depth left out, the deepest vertex below every bed).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from kelp.core.io import LazyLerpField, VelocityField
from kelp.core.mesh import find_nearest_nodes, set_exact_nearest
from kelp.core.particles import make_state
from kelp.numerics import integrators as I
from kelp.numerics.temporal_reconstruct import _pchip_value_scalar
from kelp.numerics.vertical_uv import depth_points

from tests._bitwise import assert_bitwise

healpy = pytest.importorskip("healpy")

from kelp.readers.healpix import build_healpix_dual_mesh  # noqa: E402
from kelp.readers.structured import build_structured_mesh  # noqa: E402

pytestmark = pytest.mark.skipif(not I.HAS_NUMBA,
                                reason="fused kernels need numba")

NSIDE = 8
NPIX = 12 * NSIDE * NSIDE
Z_LEVELS = np.array([0.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0,
                     1000.0, 2000.0, 4000.0])
K = Z_LEVELS.size
DT = 5400.0
PLACEMENTS = ["level", "layer_linear"]

_FIELDS = ("lon", "lat", "z", "status", "nearest_node", "u", "v", "w",
           "failure_reason")

_LAYER_KERNELS = ("_fused_rk2_step_3d_bary_tinterp",
                  "_fused_rk2_step_3d_bary_pchip")
_DL_KERNELS = ("_fused_rk2_step_3d_bary_tinterp_dl",
               "_fused_rk2_step_3d_bary_pchip_dl")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _healpix_mesh():
    """The HEALPix dual mesh of ``test_fused_3d_bary``: varied bathymetry
    with shallow columns, so ``last`` differs between nodes."""
    lon, lat = healpy.pix2ang(NSIDE, np.arange(NPIX), nest=True,
                              lonlat=True)
    depth = 60.0 + 3500.0 * np.abs(np.sin(np.deg2rad(lat * 2.0)))
    depth[::17] = 35.0
    return build_healpix_dual_mesh(NSIDE, wet_mask=np.ones(NPIX, bool),
                                   depth=depth, z_levels=Z_LEVELS)


@pytest.fixture(scope="module")
def mesh():
    return _healpix_mesh()


def _nan_below_last(arr, dp):
    below = np.arange(dp.p.size)[:, None] > dp.last[None, :]
    arr[below] = np.nan
    return arr


def _dl_field(mesh, placement, seed, scale=0.4):
    """Random u/v on the data points, NaN at every point below ``last``;
    w on the levels."""
    dp = depth_points(mesh, placement)
    rng = np.random.default_rng(seed)
    n = mesh.n_nodes
    u = _nan_below_last(scale * rng.standard_normal((dp.p.size, n)), dp)
    v = _nan_below_last(scale * rng.standard_normal((dp.p.size, n)), dp)
    w = 5e-4 * rng.standard_normal((K, n))
    return VelocityField(u=u, v=v, eta=np.zeros(n), w=w, uv_layout="node",
                         uv_vertical=placement)


def _as_layer(vel, n_rows=None):
    """The same arrays as a ``"layer"`` field (first ``n_rows`` rows)."""
    rows = slice(None) if n_rows is None else slice(0, n_rows)
    return VelocityField(u=vel.u[rows].copy(), v=vel.v[rows].copy(),
                         eta=vel.eta, w=vel.w, uv_layout="node")


def _seeds(mesh, n=3000, seed=5):
    """Depths at the surface, inside, at the 20 m point, below every bed,
    and NaN (read like a depth below the bed on both paths)."""
    rng = np.random.default_rng(seed)
    lon = rng.uniform(-179.0, 179.0, n)
    lat = np.degrees(np.arcsin(rng.uniform(-0.95, 0.95, n)))
    z = rng.uniform(0.0, 3000.0, n)
    z[::7] = 20.0
    z[1::13] = 3900.0
    z[2::29] = np.nan
    z[3::31] = 0.0
    st = make_state(lon, lat, z)
    return st.replace(failure_reason=np.zeros(n, dtype=np.int8),
                      nearest_node=find_nearest_nodes(mesh, lon, lat))


def _assert_same(got, ref):
    for f in _FIELDS:
        assert_bitwise(getattr(got, f), getattr(ref, f), f)


def _assert_no_dry_read(out, st):
    """Every alive particle got finite u/v: nothing below ``last`` read."""
    alive = st.alive_mask & np.isfinite(st.z)
    assert np.isfinite(out.u[alive]).all()
    assert np.isfinite(out.v[alive]).all()


def _spy(monkeypatch, names):
    """Count calls of module-level functions of ``integrators``."""
    calls = {nm: 0 for nm in names}
    for nm in names:
        orig = getattr(I, nm)

        def spy(*a, _nm=nm, _orig=orig, **k):
            calls[_nm] += 1
            return _orig(*a, **k)
        monkeypatch.setattr(I, nm, spy)
    return calls


def _tinterp_payload(mesh, placement, mode):
    """(vel, kwargs) of a steady step (plain or lazy field), a plain-pair or
    a lazy-pair step."""
    a = _dl_field(mesh, placement, 1)
    if mode == "steady":
        return a, {}
    b = _dl_field(mesh, placement, 2)
    if mode == "steady_lazy":
        return LazyLerpField(a, b, 0.3), {}
    if mode == "lerp":
        return a, dict(vel_next=b, dt_to_next=21600.0)
    lazy = LazyLerpField(a, b, 0.3)
    return lazy, dict(vel_next=_dl_field(mesh, placement, 3),
                      dt_to_next=21600.0)


_TIMES = [0.0, 86400.0, 172800.0, 259200.0]


def _pchip_payload(mesh, placement, tdir, seeds=(11, 12, 13, 14)):
    """(vel, kwargs) of a 4-snapshot PCHIP step, forward or backward."""
    snaps = [_dl_field(mesh, placement, s) for s in seeds]
    q = 100000.0 if tdir > 0 else 150000.0
    recon = (snaps, list(_TIMES), q, tdir)
    if tdir > 0:
        return snaps[1], dict(vel_next=snaps[2], dt_to_next=86400.0,
                              recon=recon)
    return snaps[2], dict(vel_next=snaps[1], dt_to_next=86400.0,
                          recon=recon)


# ---------------------------------------------------------------------------
# Fused equals unfused
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("w_interp", ["nearest", "barycentric"])
@pytest.mark.parametrize("tdir", [1.0, -1.0])
@pytest.mark.parametrize("mode", ["steady", "steady_lazy", "lerp", "lazy"])
@pytest.mark.parametrize("placement", PLACEMENTS)
def test_fused_equals_unfused_tinterp(mesh, monkeypatch, placement, mode,
                                      tdir, w_interp):
    """``rk2_step`` takes the fused depth-linear twin and equals the
    unfused depth-linear step bit for bit."""
    vel, kw = _tinterp_payload(mesh, placement, mode)
    st = _seeds(mesh)
    calls = _spy(monkeypatch, _LAYER_KERNELS + _DL_KERNELS)
    got = I.rk2_step(st, vel, mesh, DT, tdir, uv_interp="barycentric",
                     w_interp=w_interp, **kw)
    assert calls == {"_fused_rk2_step_3d_bary_tinterp": 0,
                     "_fused_rk2_step_3d_bary_pchip": 0,
                     "_fused_rk2_step_3d_bary_tinterp_dl": 1,
                     "_fused_rk2_step_3d_bary_pchip_dl": 0}
    ref = I._rk2_step_unfused(st, vel, mesh, DT, tdir, 0, True,
                              w_interp=w_interp, uv_interp="barycentric",
                              **kw)
    _assert_same(got, ref)
    _assert_no_dry_read(got, st)
    assert int((got.status == 1).sum()) > 2000     # most particles moved


@pytest.mark.parametrize("w_interp", ["nearest", "barycentric"])
@pytest.mark.parametrize("tdir", [1.0, -1.0])
@pytest.mark.parametrize("placement", PLACEMENTS)
def test_fused_equals_unfused_pchip(mesh, monkeypatch, placement, tdir,
                                    w_interp):
    """The PCHIP twin, forward and backward, through ``rk2_step`` and
    through the runner itself."""
    vel, kw = _pchip_payload(mesh, placement, tdir)
    st = _seeds(mesh)
    calls = _spy(monkeypatch, _LAYER_KERNELS + _DL_KERNELS)
    got = I.rk2_step(st, vel, mesh, DT, tdir, uv_interp="barycentric",
                     w_interp=w_interp, **kw)
    assert calls == {"_fused_rk2_step_3d_bary_tinterp": 0,
                     "_fused_rk2_step_3d_bary_pchip": 0,
                     "_fused_rk2_step_3d_bary_tinterp_dl": 0,
                     "_fused_rk2_step_3d_bary_pchip_dl": 1}
    ref = I._rk2_step_unfused(st, vel, mesh, DT, tdir, 0, True,
                              w_interp=w_interp, uv_interp="barycentric",
                              recon=kw["recon"])
    _assert_same(got, ref)
    _assert_no_dry_read(got, st)
    w_code = I.W_NEAREST if w_interp == "nearest" else I.W_BARYCENTRIC
    direct = I._run_fused_rk2_3d_bary_pchip(st, kw["recon"], mesh, DT, tdir,
                                           w_code)
    _assert_same(direct, ref)


# ---------------------------------------------------------------------------
# Walls at depth: a trench
# ---------------------------------------------------------------------------

TRENCH_LEVELS = np.array([0.0, 50.0, 100.0, 200.0, 300.0, 500.0, 800.0,
                          1200.0, 2000.0, 3000.0])


def _trench_mesh():
    """A triangle mesh of 0.2 deg spacing over 4..6 E, 4..6 N with a trench
    along 5 E: the nodes on 5 E are 2000 m deep, those on 4.8 and 5.2 E
    800 m, the rest 300 m.  The triangles beside the trench have vertices
    dry at depths between their beds."""
    from tests.test_3d_analytical import _make_mesh
    m = _make_mesh(5.0, 5.0, span=1.0, nx=11, ny=11,
                   z_levels=TRENCH_LEVELS, bottom_depth=300.0)
    off = np.abs(m.nodes_lon - 5.0)
    m.depth = np.where(off < 0.1, 2000.0, np.where(off < 0.3, 800.0, 300.0))
    return m


def _trench_off(n, c):
    """Per-node offsets of the trench field (m/s): the vertices differ."""
    return c * np.sin(1.7 * np.arange(n))


def _trench_uv_hand(m, t):
    """``value(nd, zc)`` of u and v of :func:`_trench_field` at time t."""
    ou, ov = _trench_off(m.n_nodes, 0.05), _trench_off(m.n_nodes, -0.03)
    return ((lambda nd, zc: 0.1 + 1e-4 * zc + ou[nd] + 2e-6 * t),
            (lambda nd, zc: -0.05 - 5e-5 * zc + ov[nd] + 1e-6 * t))


def _trench_field(m, placement, t):
    """u/v linear in depth and time with a different offset per node, NaN
    below each node's last usable point; w = 0.02 m/s (the stage depths
    move by 36 and 72 m in a step)."""
    dp = depth_points(m, placement)
    n = m.n_nodes
    fu, fv = _trench_uv_hand(m, t)
    u = np.array([[fu(nd, pk) for nd in range(n)] for pk in dp.p])
    v = np.array([[fv(nd, pk) for nd in range(n)] for pk in dp.p])
    w = np.full((TRENCH_LEVELS.size, n), 0.02)
    return VelocityField(u=_nan_below_last(u, dp), v=_nan_below_last(v, dp),
                         eta=np.zeros(n), w=w, uv_layout="node",
                         uv_vertical=placement)


def _trench_seeds(m, n=400, seed=9):
    """Seeds beside the trench at depths between the beds, on the beds,
    below every bed and NaN."""
    rng = np.random.default_rng(seed)
    lon = rng.uniform(4.65, 5.35, n)
    lat = rng.uniform(4.2, 5.8, n)
    z = rng.uniform(0.0, 2600.0, n)
    z[::11] = 800.0
    z[1::11] = 2000.0
    z[2::13] = 2500.0
    z[3::17] = np.nan
    st = make_state(lon, lat, z)
    return st.replace(failure_reason=np.zeros(n, dtype=np.int8),
                      nearest_node=find_nearest_nodes(m, lon, lat))


@pytest.mark.parametrize("w_interp", ["nearest", "barycentric"])
@pytest.mark.parametrize("tdir", [1.0, -1.0])
@pytest.mark.parametrize("mode", ["lerp", "pchip"])
@pytest.mark.parametrize("placement", PLACEMENTS)
def test_fused_equals_unfused_with_walls(monkeypatch, placement, mode, tdir,
                                         w_interp):
    """The trench mesh: the fused RK2 twin (linear in time and PCHIP)
    equals the unfused step bit for bit where vertices are dry at the
    stage depths, and the recorded first-stage velocity is the wall rule
    by hand (``tests/test_interp_depth_linear._hand_wall``): dry vertices
    left out and the weights renormalised; below every bed the deepest
    vertex at its last point."""
    from kelp.core.mesh import _find_containing_element_local
    from tests.test_interp_depth_linear import _hand_wall, _lam
    m = _trench_mesh()
    times = [0.0, 21600.0, 43200.0, 64800.0]
    snaps = [_trench_field(m, placement, t) for t in times]
    if mode == "lerp":
        a, b = (1, 2) if tdir > 0 else (2, 1)
        vel, kw = snaps[a], dict(vel_next=snaps[b], dt_to_next=21600.0)
        t1 = times[a]
    else:
        q = 30000.0 if tdir > 0 else 35000.0
        vel = snaps[1] if tdir > 0 else snaps[2]
        kw = dict(vel_next=snaps[2] if tdir > 0 else snaps[1],
                  dt_to_next=21600.0, recon=(snaps, times, q, tdir))
        t1 = q
    st = _trench_seeds(m)
    calls = _spy(monkeypatch, _LAYER_KERNELS + _DL_KERNELS)
    got = I.rk2_step(st, vel, m, 3600.0, tdir, uv_interp="barycentric",
                     w_interp=w_interp, **kw)
    want = _DL_KERNELS[1] if mode == "pchip" else _DL_KERNELS[0]
    assert calls[want] == 1 and sum(calls.values()) == 1, calls
    ref_kw = dict(recon=kw["recon"]) if mode == "pchip" else kw
    ref = I._rk2_step_unfused(st, vel, m, 3600.0, tdir, 0, True,
                              w_interp=w_interp, uv_interp="barycentric",
                              **ref_kw)
    _assert_same(got, ref)
    assert (got.status == 1).all()
    # the first stage by hand, and the fixture does reach the walls
    fu, fv = _trench_uv_hand(m, t1)
    n_wall = n_dry = n_wet = 0
    for i in range(st.n_particles):
        x, y, z = st.lon[i], st.lat[i], st.z[i]
        e, inside = _find_containing_element_local(
            x, y, int(st.nearest_node[i]), m.node_to_elems,
            m.node_to_elems_count, m.elem_nodes, m.nodes_lon, m.nodes_lat)
        assert inside
        tri = [int(nd) for nd in m.elem_nodes[e]]
        lam = _lam(m, tri, x, y)
        wet = (np.inf if np.isnan(z) else z) <= m.depth[tri]
        n_wet += wet.all()
        n_dry += not wet.any()
        n_wall += wet.any() and not wet.all()
        hu = _hand_wall(m, placement, tri, lam, z, fu)
        hv = _hand_wall(m, placement, tri, lam, z, fv)
        # the vertex values are up to ~0.4 m/s: an absolute bound too, for
        # a result near zero (cancellation between vertices)
        np.testing.assert_allclose(got.u[i], tdir * hu, rtol=1e-12,
                                   atol=1e-14, err_msg=f"particle {i}")
        np.testing.assert_allclose(got.v[i], tdir * hv, rtol=1e-12,
                                   atol=1e-14, err_msg=f"particle {i}")
    # 136 / 178 / 86 of the 400 seeds (rng seed 9)
    assert n_wall >= 100 and n_dry >= 100 and n_wet >= 60, (n_wall, n_dry,
                                                           n_wet)


def _as_float32(vel):
    return VelocityField(u=vel.u.astype(np.float32), v=vel.v.astype(np.float32),
                         eta=vel.eta, w=vel.w, uv_layout="node",
                         uv_vertical=vel.uv_vertical)


@pytest.mark.parametrize("mode", ["steady", "lazy", "pchip"])
def test_fused_equals_unfused_float32(mesh, monkeypatch, mode):
    """float32 u/v (cached snapshots may be float32): the twins widen each
    stored value before any arithmetic, as the unfused ``_blend`` /
    ``reconstruct_pchip`` / ``uv_at`` do — still bit for bit."""
    if mode == "pchip":
        vel, kw = _pchip_payload(mesh, "level", 1.0)
        snaps = [_as_float32(s) for s in kw["recon"][0]]
        vel = snaps[1]
        kw = dict(recon=(snaps,) + tuple(kw["recon"][1:]))
    elif mode == "lazy":
        a = _as_float32(_dl_field(mesh, "layer_linear", 1))
        b = _as_float32(_dl_field(mesh, "layer_linear", 2))
        vel = LazyLerpField(a, b, 0.3)
        kw = dict(vel_next=_as_float32(_dl_field(mesh, "layer_linear", 3)),
                  dt_to_next=21600.0)
    else:
        vel, kw = _as_float32(_dl_field(mesh, "level", 1)), {}
    st = _seeds(mesh, n=1000)
    calls = _spy(monkeypatch, _DL_KERNELS)
    got = I.rk2_step(st, vel, mesh, DT, 1.0, uv_interp="barycentric",
                     w_interp="barycentric", **kw)
    assert sum(calls.values()) == 1
    ref = I._rk2_step_unfused(st, vel, mesh, DT, 1.0, 0, True,
                              w_interp="barycentric",
                              uv_interp="barycentric", **kw)
    _assert_same(got, ref)
    assert got.u.dtype == np.float64


_STEP_SPIES = (_LAYER_KERNELS + _DL_KERNELS
               + ("_rk2_step_unfused", "_rk4_step_unfused",
                  "_euler_step_unfused"))


def _mixed_pchip(mesh, mix):
    """(vel, recon) of an inconsistent PCHIP step, same shapes on both
    sides: a ``"layer"`` vel with ``"layer_linear"`` snapshots, or the
    reverse."""
    _, kw = _pchip_payload(mesh, "layer_linear", 1.0)
    snaps = kw["recon"][0]
    rest = tuple(kw["recon"][1:])
    if mix == "layer_vel":
        return _as_layer(snaps[1]), (snaps,) + rest
    return snaps[1], ([_as_layer(s_) for s_ in snaps],) + rest


@pytest.mark.parametrize("route", ["rk2_step", "rk4_step", "runner"])
@pytest.mark.parametrize("mix", ["layer_vel", "layer_snapshots"])
def test_mixed_vel_and_snapshots_refused(mesh, monkeypatch, mix, route):
    """A step whose ``vel`` and PCHIP snapshots differ in placement is
    inconsistent input, refused whichever side is depth-linear: at the
    entry of ``rk2_step`` (the fused route) and ``rk4_step`` (unfused), and
    by the fused runner called directly with the step's ``vel``.  Nothing
    is integrated."""
    vel, recon = _mixed_pchip(mesh, mix)
    pv, ps = (("layer", "layer_linear") if mix == "layer_vel"
              else ("layer_linear", "layer"))
    st = _seeds(mesh, n=50)
    calls = _spy(monkeypatch, _STEP_SPIES)
    if route == "runner":
        with pytest.raises(ValueError, match=(
                r"^_run_fused_rk2_3d_bary_pchip: uv_vertical mismatch "
                r"between the fields the fused kernel reads: "
                rf"vel='{pv}', recon snapshot 0='{ps}', .*"
                rf"recon snapshot 3='{ps}'$")):
            I._run_fused_rk2_3d_bary_pchip(st, recon, mesh, DT, 1.0,
                                           I.W_BARYCENTRIC, vel=vel)
    else:
        step = getattr(I, route)
        with pytest.raises(ValueError, match=(
                rf"^{route}: uv_vertical mismatch between the fields of one "
                rf"step: vel='{pv}', recon snapshot 0='{ps}', .*"
                rf"recon snapshot 3='{ps}'$")):
            step(st, vel, mesh, DT, 1.0, uv_interp="barycentric",
                 w_interp="barycentric", recon=recon)
    assert all(c == 0 for c in calls.values()), calls


@pytest.mark.parametrize("route", ["euler_step", "rk2_step", "rk4_step"])
@pytest.mark.parametrize("mix", ["layer_vel", "layer_next"])
def test_mixed_vel_and_next_refused(mesh, monkeypatch, mix, route):
    """The same for ``vel`` / ``vel_next`` (same shapes), both directions,
    at the entry of every step function."""
    dl = _dl_field(mesh, "layer_linear", 1)
    lay = _as_layer(_dl_field(mesh, "layer_linear", 2))
    vel, nxt = (lay, dl) if mix == "layer_vel" else (dl, lay)
    calls = _spy(monkeypatch, _STEP_SPIES)
    with pytest.raises(ValueError, match=(
            rf"^{route}: uv_vertical mismatch between the fields of one "
            rf"step: vel='{vel.uv_vertical}', "
            rf"vel_next='{nxt.uv_vertical}'$")):
        getattr(I, route)(_seeds(mesh, n=50), vel, mesh, DT, 1.0,
                          uv_interp="barycentric", w_interp="barycentric",
                          vel_next=nxt, dt_to_next=21600.0)
    assert all(c == 0 for c in calls.values()), calls


# ---------------------------------------------------------------------------
# Time first, then vertical (the non-commuting fixture)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tdir", [1.0, -1.0])
def test_pchip_time_then_vertical(monkeypatch, tdir):
    """Equally spaced snapshots; the upper level holds ``[0, 1, 1, 1]``, the
    lower ``[0, 1, 2, 3]``; read at their vertical midpoint and at the
    temporal midpoint of the central interval.  Time first gives
    ``(1 + 1.5) / 2 = 1.25`` exactly (fused and unfused); vertical first
    would give PCHIP of ``[0, 1, 1.5, 2]`` = 1.2708333333333333."""
    from tests.test_interp_depth_linear import _mesh
    m = _mesh(np.full(9, 200.0))                 # levels 0, 10, .., 40 m
    times = [0.0, 3600.0, 7200.0, 10800.0]
    upper = [0.0, 1.0, 1.0, 1.0]                 # level 1 (10 m)
    lower = [0.0, 1.0, 2.0, 3.0]                 # level 2 (20 m)
    snaps = []
    for j in range(4):
        col = np.array([upper[j], upper[j], lower[j], lower[j], lower[j]])
        u = np.repeat(col[:, None], 9, axis=1)
        snaps.append(VelocityField(u=u, v=0.5 * u, eta=np.zeros(9),
                                   w=np.zeros((5, 9)), uv_layout="node",
                                   uv_vertical="level"))
    recon = (snaps, times, 5400.0, tdir)
    st = make_state(np.array([4.6, 5.3, 5.0]), np.array([4.7, 5.2, 5.0]),
                    np.full(3, 15.0))
    calls = _spy(monkeypatch, _DL_KERNELS)
    got = I.rk2_step(st, snaps[1], m, 600.0, tdir, uv_interp="barycentric",
                     w_interp="nearest", recon=recon)
    assert calls["_fused_rk2_step_3d_bary_pchip_dl"] == 1
    ref = I._rk2_step_unfused(st, snaps[1], m, 600.0, tdir, 0, True,
                              w_interp="nearest", uv_interp="barycentric",
                              recon=recon)
    vertical_first = _pchip_value_scalar.py_func(
        0.0, 1.0, 1.5, 2.0, *times, 5400.0)
    assert abs(vertical_first - 1.2708333333333333) < 1e-15   # 61/48
    for out in (got, ref):
        assert_bitwise(out.u, np.full(3, tdir * 1.25), "u")
        assert_bitwise(out.v, np.full(3, tdir * 0.625), "v")
        assert np.all(out.u != tdir * 1.2708333333333333)
        assert np.all(out.u != tdir * vertical_first)


# ---------------------------------------------------------------------------
# Anchor heal and terminal splice keep the placement
# ---------------------------------------------------------------------------

def _hole_mesh():
    """The hole-band structured mesh of ``test_fused_3d_bary``: no wet node
    between 70 and 73 N, so a stage anchor there has an empty hash window."""
    lon1d = np.arange(0.0, 30.0, 0.25)
    lat1d = np.arange(60.0, 80.0, 0.25)
    lon2d, lat2d = np.meshgrid(lon1d, lat1d)
    wet = np.ones_like(lon2d, bool)
    wet[(lat2d > 70.0) & (lat2d < 73.0)] = False
    m = build_structured_mesh(lon2d, lat2d, wet)
    m.sigma = Z_LEVELS
    m.depth = np.full(m.n_nodes, 3500.0)
    return m


def _hole_field(m, placement, v0):
    """Northward ``v0 + 1e-3 p`` (about 1 deg per half step) and
    ``u = 0.2 + 1e-4 p`` on the data points, NaN below ``last``."""
    dp = depth_points(m, placement)
    n = m.n_nodes
    u = _nan_below_last(np.repeat((0.2 + 1e-4 * dp.p)[:, None], n, axis=1),
                        dp)
    v = _nan_below_last(np.repeat((v0 + 1e-3 * dp.p)[:, None], n, axis=1),
                        dp)
    return VelocityField(u=u, v=v, eta=np.zeros(n),
                         w=np.full((K, n), 2e-3), uv_layout="node",
                         uv_vertical=placement)


def _hole_seeds(m):
    """Half at 69.9 N (the midpoint lands in the hole: flagged), half at
    65 N (never flagged), at depths between data points and above every
    node's last point (where a layer reading gives other values)."""
    n = 200
    rng = np.random.default_rng(2)
    plon = rng.uniform(5.0, 25.0, n)
    plat = np.where(np.arange(n) % 2 == 0, 69.9, 65.0)
    pz = np.resize(np.array([7.0, 130.0, 700.0, 1300.0, 40.0]), n)
    st = make_state(plon, plat, pz)
    return st.replace(failure_reason=np.zeros(n, dtype=np.int8),
                      nearest_node=find_nearest_nodes(m, plon, plat))


def _hole_run(m, placement, mode, st):
    """(fused runner result, unfused reference, kernel name, payload)."""
    if mode == "tinterp":
        vel = _hole_field(m, placement, 45.0)
        got = I._run_fused_rk2_3d_bary_tinterp(st, vel, None, None, m, DT,
                                               1.0, I.W_BARYCENTRIC)
        ref = I._rk2_step_unfused(st, vel, m, DT, 1.0, 0, True,
                                  w_interp="barycentric",
                                  uv_interp="barycentric")
        return got, ref, _DL_KERNELS[0], vel
    snaps = [_hole_field(m, placement, v0) for v0 in (44.0, 45.0, 46.0, 47.0)]
    recon = (snaps, [0.0, 21600.0, 43200.0, 64800.0], 30000.0, 1.0)
    got = I._run_fused_rk2_3d_bary_pchip(st, recon, m, DT, 1.0,
                                         I.W_BARYCENTRIC)
    ref = I._rk2_step_unfused(st, snaps[1], m, DT, 1.0, 0, True,
                              w_interp="barycentric",
                              uv_interp="barycentric", recon=recon)
    return got, ref, _DL_KERNELS[1], recon


@pytest.mark.parametrize("mode", ["tinterp", "pchip"])
@pytest.mark.parametrize("placement", PLACEMENTS)
def test_fused_heal_keeps_placement(monkeypatch, placement, mode):
    """Particles whose midpoint anchor has an empty hash window are flagged
    and re-run through the SAME depth-linear twin with the healed anchor
    (no unfused splice); the result equals the unfused depth-linear step."""
    m = _hole_mesh()
    st = _hole_seeds(m)
    calls = _spy(monkeypatch, _DL_KERNELS + _LAYER_KERNELS
                 + ("_splice_exact_rerun",))
    got, ref, kern, _ = _hole_run(m, placement, mode, st)
    assert calls[kern] >= 2                  # first pass + a heal round
    assert calls["_splice_exact_rerun"] == 0
    assert all(calls[k] == 0 for k in _LAYER_KERNELS)
    _assert_same(got, ref)
    assert int((got.status == 1).sum()) >= 90    # the 65 N half moved


@pytest.mark.parametrize("mode", ["tinterp", "pchip"])
@pytest.mark.parametrize("placement", PLACEMENTS)
def test_exact_nearest_parity(monkeypatch, placement, mode):
    """With the exact nearest-node guard on (``exact_nearest``), the twins
    resolve the empty-window stage anchors with their in-kernel exact
    search: one launch (no flag, no heal round, no splice — the same seeds
    flag without the guard, see the heal test) and the unfused step bit
    for bit."""
    m = _hole_mesh()
    set_exact_nearest(m, True)
    st = _hole_seeds(m)
    calls = _spy(monkeypatch, _DL_KERNELS + _LAYER_KERNELS
                 + ("_splice_exact_rerun",))
    got, ref, kern, _ = _hole_run(m, placement, mode, st)
    assert calls[kern] == 1
    assert calls["_splice_exact_rerun"] == 0
    assert all(calls[k] == 0 for k in _LAYER_KERNELS)
    _assert_same(got, ref)
    assert int((got.status == 1).sum()) >= 90


@pytest.mark.parametrize("mode", ["tinterp", "pchip"])
@pytest.mark.parametrize("placement", PLACEMENTS)
def test_terminal_splice_keeps_placement(monkeypatch, placement, mode):
    """With the heal rounds off, the flagged particles are spliced from the
    unfused re-run, which reads the field with its own placement: they
    equal the unfused depth-linear step, not a layer re-run."""
    m = _hole_mesh()
    st = _hole_seeds(m)
    monkeypatch.setattr(I, "_FLAG_HEAL_ROUNDS", 0)
    seen = []
    orig = I._splice_exact_rerun

    def spy(kernel_out, state, flagged, unfused_fn):
        seen.append(flagged.copy())
        return orig(kernel_out, state, flagged, unfused_fn)
    monkeypatch.setattr(I, "_splice_exact_rerun", spy)
    got, ref, _, payload = _hole_run(m, placement, mode, st)
    assert len(seen) == 1
    flagged = seen[0]
    assert flagged.sum() >= 50
    _assert_same(got, ref)
    # a layer reading of the same arrays gives other recorded velocities
    # on the spliced particles: the splice kept the placement
    rows = K - 1
    if mode == "tinterp":
        lay = I._rk2_step_unfused(st, _as_layer(payload, rows), m, DT, 1.0, 0,
                                  True, w_interp="barycentric",
                                  uv_interp="barycentric")
    else:
        snaps = [_as_layer(s, rows) for s in payload[0]]
        lay = I._rk2_step_unfused(st, snaps[1], m, DT, 1.0, 0, True,
                                  w_interp="barycentric",
                                  uv_interp="barycentric",
                                  recon=(snaps,) + tuple(payload[1:]))
    assert np.all(got.u[flagged] != lay.u[flagged])


# ---------------------------------------------------------------------------
# Placement and record checks in the runners
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("j", [0, 1, 2, 3])
def test_pchip_snapshot_placement_mixed_raises(mesh, monkeypatch, j):
    """A same-shaped ``"layer"`` snapshot among ``"layer_linear"`` ones, at
    each of the four positions, is refused by the runner before any kernel
    runs."""
    _, kw = _pchip_payload(mesh, "layer_linear", 1.0)
    snaps = list(kw["recon"][0])
    snaps[j] = _as_layer(snaps[j])
    assert snaps[j].u.shape == snaps[(j + 1) % 4].u.shape
    recon = (snaps,) + tuple(kw["recon"][1:])
    calls = _spy(monkeypatch, _LAYER_KERNELS + _DL_KERNELS)
    with pytest.raises(ValueError, match=(
            r"^_run_fused_rk2_3d_bary_pchip: uv_vertical mismatch.*"
            rf"recon snapshot {j}='layer'")):
        I._run_fused_rk2_3d_bary_pchip(_seeds(mesh, n=50), recon, mesh, DT,
                                       1.0, I.W_BARYCENTRIC)
    assert all(c == 0 for c in calls.values())


_TINTERP_MIX = {
    "vel_next": "vel='layer_linear', vel_next='layer'",
    "vel": "vel='layer', vel_next='layer_linear'",
    "lazy_parent": ("vel.vel_a='layer_linear', vel.vel_b='layer', "
                    "vel='layer_linear', vel_next='layer_linear'"),
    "steady_lazy_parent": ("vel.vel_a='layer_linear', vel.vel_b='layer', "
                           "vel='layer_linear'"),
}


@pytest.mark.parametrize("case", sorted(_TINTERP_MIX))
def test_tinterp_placement_mixed_raises(mesh, monkeypatch, case):
    """The tinterp runner refuses a ``"layer"`` field among the depth-linear
    ones it reads: ``vel``, the target ``vel_next`` or a lazy ``vel``'s raw
    parent (a parent swapped after construction, direct callers only); the
    message is the runner's and names every field with its placement."""
    dl = _dl_field(mesh, "layer_linear", 1)
    lay = _as_layer(_dl_field(mesh, "layer_linear", 2))
    kw = dict(vel_next=_dl_field(mesh, "layer_linear", 3),
              dt_to_next=21600.0)
    if case == "vel_next":
        vel, kw["vel_next"] = dl, lay
    elif case == "vel":
        vel = lay
    else:
        vel = LazyLerpField(dl, _dl_field(mesh, "layer_linear", 4), 0.3)
        vel.vel_b = lay
        if case == "steady_lazy_parent":
            kw = dict(vel_next=None, dt_to_next=None)
    calls = _spy(monkeypatch, _LAYER_KERNELS + _DL_KERNELS)
    msg = ("_run_fused_rk2_3d_bary_tinterp: uv_vertical mismatch between "
           "the fields the fused kernel reads: " + _TINTERP_MIX[case])
    with pytest.raises(ValueError, match="^" + re.escape(msg) + "$"):
        I._run_fused_rk2_3d_bary_tinterp(_seeds(mesh, n=50), vel,
                                         kw["vel_next"], kw["dt_to_next"],
                                         mesh, DT, 1.0, I.W_BARYCENTRIC)
    assert all(c == 0 for c in calls.values())


def test_field_checks_by_placement():
    """``_fused_3d_bary_field_checks`` wants u/v ``(K, N)`` for ``"level"``
    and ``(K-1, N)`` otherwise; w is ``(K, N)`` always.  The ``"layer"``
    messages are the historical ones."""
    from tests.test_interp_depth_linear import _mesh
    m = _mesh()                                      # K = 5, N = 9
    chk = I._fused_3d_bary_field_checks
    kn, kn1 = np.zeros((5, 9)), np.zeros((4, 9))
    with pytest.raises(ValueError) as e:
        chk(m, [("u_a", kn)], [("w_a", kn)])
    assert str(e.value) == ("fused 3-D barycentric kernel: u_a must be "
                            "(K-1, N) = (4, 9), got (5, 9)")
    with pytest.raises(ValueError) as e:
        chk(m, [("u_a", kn1)], [("w_a", kn1)])
    assert str(e.value) == ("fused 3-D barycentric kernel: w_a must be "
                            "(K, N) = (5, 9), got (4, 9)")
    with pytest.raises(ValueError) as e:
        chk(m, [("u_a", None)], [("w_a", kn)], uv_vertical="layer")
    assert str(e.value) == ("fused 3-D barycentric kernel: u_a must be "
                            "(K-1, N) = (4, 9), got None")
    with pytest.raises(ValueError) as e:
        chk(m, [("u0", kn1)], [("w0", kn)], uv_vertical="level")
    assert str(e.value) == ("fused 3-D barycentric kernel: u0 must be "
                            "(K, N) = (5, 9), got (4, 9)")
    with pytest.raises(ValueError) as e:
        chk(m, [("v_t", kn)], [("w_t", kn)], uv_vertical="layer_linear")
    assert str(e.value) == ("fused 3-D barycentric kernel: v_t must be "
                            "(K-1, N) = (4, 9), got (5, 9)")
    with pytest.raises(ValueError) as e:
        chk(m, [("u_a", kn)], [("w_a", kn1)], uv_vertical="level")
    assert str(e.value) == ("fused 3-D barycentric kernel: w_a must be "
                            "(K, N) = (5, 9), got (4, 9)")
    assert chk(m, [("u_a", kn)], [("w_a", kn)], uv_vertical="level") is None
    assert chk(m, [("u_a", kn1)], [("w_a", kn)],
               uv_vertical="layer_linear") is None
    assert chk(m, [("u_a", kn1)], [("w_a", kn)]) is None


# ---------------------------------------------------------------------------
# NUMBA_DISABLE_JIT: the pure-Python twins equal the pure-Python unfused path
# ---------------------------------------------------------------------------

def _to_float32(vel, kw):
    """The same payload with float32 u/v everywhere (lazy fields rebuilt
    over float32 parents; one object stays one object)."""
    conv = {}

    def f32(x):
        if id(x) not in conv:
            if isinstance(x, LazyLerpField):
                conv[id(x)] = LazyLerpField(f32(x.vel_a), f32(x.vel_b),
                                            x.alpha)
            else:
                conv[id(x)] = _as_float32(x)
        return conv[id(x)]
    out = dict(kw)
    if kw.get("vel_next") is not None:
        out["vel_next"] = f32(kw["vel_next"])
    if "recon" in kw:
        out["recon"] = ([f32(s_) for s_ in kw["recon"][0]],) \
            + tuple(kw["recon"][1:])
    return f32(vel), out


#: no-JIT parity cases: (placement, payload, dtype); the PCHIP one backward
NOJIT_CASES = [(pl, mode, dt) for pl in PLACEMENTS
               for mode in ("lerp", "lazy", "pchip")
               for dt in ("float64", "float32")]


def _nojit_case(mesh, placement, mode, dtype):
    """(vel, kwargs, time_dir) of one no-JIT parity case."""
    if mode == "pchip":
        vel, kw = _pchip_payload(mesh, placement, -1.0)
        tdir = -1.0
    else:
        vel, kw = _tinterp_payload(mesh, placement, mode)
        tdir = 1.0
    if dtype == "float32":
        vel, kw = _to_float32(vel, kw)
    return vel, kw, tdir


def _same_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (a.dtype == b.dtype and a.shape == b.shape
            and a.tobytes() == b.tobytes())


_SUBPROCESS_SCRIPT = r"""
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
import tests.test_fused_3d_depth_linear as t
launches = []
for nm in t._DL_KERNELS:
    def spy(*a, _orig=getattr(I, nm), **k):
        launches.append(1)
        return _orig(*a, **k)
    setattr(I, nm, spy)
mesh = t._healpix_mesh()
st = t._seeds(mesh, n=40)
rows = {}
for placement, mode, dtype in t.NOJIT_CASES:
    launches.clear()
    vel, kw, tdir = t._nojit_case(mesh, placement, mode, dtype)
    got = I.rk2_step(st, vel, mesh, t.DT, tdir, uv_interp="barycentric",
                     w_interp="barycentric", **kw)
    if mode == "pchip":
        kw = dict(recon=kw["recon"])
    ref = I._rk2_step_unfused(st, vel, mesh, t.DT, tdir, 0, True,
                              w_interp="barycentric",
                              uv_interp="barycentric", **kw)
    differ = [f for f in t._FIELDS
              if not t._same_bits(getattr(got, f), getattr(ref, f))]
    rows["-".join((placement, mode, dtype))] = [
        differ, len(launches), got.lon.tolist(), got.u.tolist()]
print(json.dumps(rows))
"""


def test_numba_disabled_subprocess_matches(mesh, tmp_path):
    """With ``NUMBA_DISABLE_JIT=1`` the twins run as Python and equal the
    unfused step bit for bit there (dtype, shape and bytes) — float64 and
    float32 fields, plain and lazy lerp payloads and PCHIP, both
    placements — and agree with the compiled twins."""
    script = tmp_path / "fused_dl_nojit.py"
    script.write_text(_SUBPROCESS_SCRIPT)
    src = str(Path(__file__).resolve().parents[1] / "src")
    env = dict(os.environ, NUMBA_DISABLE_JIT="1")
    out = subprocess.run([sys.executable, str(script), src],
                         capture_output=True, text=True, timeout=900,
                         env=env)
    assert out.returncode == 0, out.stderr
    rows = json.loads(out.stdout)
    assert len(rows) == len(NOJIT_CASES) == 12
    st = _seeds(mesh, n=40)
    bad = {k: r[0] for k, r in rows.items() if r[0] or r[1] != 1}
    assert not bad, f"no-JIT fused != unfused (fields) or no twin: {bad}"
    for placement, mode, dtype in NOJIT_CASES:
        _, _, lon, u = rows["-".join((placement, mode, dtype))]
        vel, kw, tdir = _nojit_case(mesh, placement, mode, dtype)
        got = I.rk2_step(st, vel, mesh, DT, tdir, uv_interp="barycentric",
                         w_interp="barycentric", **kw)
        np.testing.assert_allclose(np.array(lon, dtype=float), got.lon,
                                   rtol=0.0, atol=1e-11)
        np.testing.assert_allclose(np.array(u, dtype=float), got.u,
                                   rtol=1e-12, atol=1e-15)
