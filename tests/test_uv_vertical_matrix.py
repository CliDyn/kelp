"""Depth-linear horizontal velocity: the path matrix.

Every supported path of a depth-linear u/v field (``VelocityField``
``uv_vertical`` ``"level"`` / ``"layer_linear"``, or the run option
``mode.uv_vertical: linear``) takes one step, or a short run, and is
compared with a hand value; every unsupported combination is refused with
a named error.  One parametrised test per row:

====================================================  =====================
row                                                   expectation
====================================================  =====================
rk2 / rk4 x w_interp x {steady, lerp, PCHIP} x        rule (RK2 fused where
placement x time_dir x elem_search x Numba on / off   the gate allows, else
                                                      unfused)
euler x the same (PCHIP configured)                   rule, LINEAR in time
rk2 fused: anchor heal, terminal splice               rule
rk45 x depth-linear                                   refused
uv_interp leastsq / edge_leastsq x depth-linear       refused on node u/v
split uniform Kh + az_const diffusion                 rule (advection part)
em / m1 in split, combined, split_rk (3-D)            refused
beaching with resuspension                            rule for alive
                                                      particles, frozen ones
                                                      untouched
``_sample_record_fields``: a level and a layer        bitwise equal
container with the same scalars and eta
prefetch {1, 3} x shared cache {off, on}              rule, placement kept
2-D run with the option                               refused
====================================================  =====================

The rule: ``u = a + b z (+ e t)`` stored at the data points of the 3x3
mesh of ``tests/test_interp_depth_linear`` (three shallow nodes, NaN below
each node's last usable point, ``+inf`` in one column), so a stage at
``(x, y, z, t)`` reads ``sum_i lam_i (a + b clip(z, p[0], p[last_i])) +
e t`` over the vertices wet at ``z`` (``z <= bed``; a dry vertex is a wall,
left out, the other weights renormalised); ``w`` is uniform, so every W
interpolator gives the same stage depths.  The run-level rows use the
FESOM2 fixture of ``tests/test_uv_vertical_option``.

The ``Numba off`` column runs each family's cells in ONE subprocess with
``NUMBA_DISABLE_JIT=1`` and checks them against the same hand values.
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kelp.core.io import VelocityField
from kelp.core.mesh import find_containing_elements, find_nearest_nodes
from kelp.core.particles import ParticleStatus, make_state
from kelp.numerics import integrators as I
from kelp.numerics import temporal_reconstruct as TR
from kelp.numerics.diffusion import DiffusionStepArgs
from kelp.numerics.vertical_uv import depth_points, is_checked

from tests._bitwise import assert_bitwise
from tests._gen_golden_3d import _payload
from tests.test_integrators_depth_linear import (
    C_RK4, E_U, E_V, K, M, N_NODES, STEPS, _assert_step, _hand_all,
    _hand_euler, _hand_rk2,
)
from tests.test_interp_depth_linear import (
    A_U, A_V, B_U, B_V, _field, _hand_wall, _lam, _mesh,
)

PLACEMENTS = ("level", "layer_linear")
W_INTERPS = ("nearest", "leastsq", "barycentric", "idw")
MODES = ("steady", "lerp", "pchip")
TDS = (1.0, -1.0)
SEARCHES = ("local", "walk")

#: step length (s) and uniform w (m/s): stage depths move 5.4 / 10.8 m
DT_STEP = 1800.0
W0 = 0.006
#: snapshot times (s); a step starts at 4500 s (forward) or 6300 s
#: (backward), inside the bracket [3600, 7200], so the lerp and PCHIP
#: payloads carry a lazy blend, as the tracking layer builds them
RECORDS = (0.0, 3600.0, 7200.0, 10800.0)
T_START = {1.0: 4500.0, -1.0: 6300.0}
#: Euler's snapshots are QUADRATIC in time (``q * t**2 / 3600``), so a
#: linear blend in time and a PCHIP reconstruction give different values
Q_U, Q_V = 3.0e-6, -2.0e-6

#: deep cell (every vertex 200 m), the 10 m node's triangle, the 25 m
#: node's cell: depths between points, above and below shallow vertices
SEEDS = ((4.3, 4.2, 12.5), (5.6, 4.3, 8.0), (5.7, 5.6, 22.0))

_STATE_FIELDS = ("lon", "lat", "z", "u", "v", "w", "status", "nearest_node")

_DL_FUSED = ("_fused_rk2_step_3d_bary_tinterp_dl",
             "_fused_rk2_step_3d_bary_pchip_dl")
_LAYER_FUSED = ("_fused_rk2_step_3d_bary_tinterp",
                "_fused_rk2_step_3d_bary_pchip", "_run_fused_euler_3d")
_UNFUSED = ("_euler_step_unfused", "_rk2_step_unfused", "_rk4_step_unfused")


def _td_id(td):
    return "fwd" if td > 0 else "bwd"


# ---------------------------------------------------------------------------
# The hand rule on the 3x3 mesh
# ---------------------------------------------------------------------------

def _tri_at(mesh, x, y):
    """The triangle containing ``(x, y)`` and its barycentric weights."""
    for row in mesh.elem_nodes:
        tri = tuple(int(n) for n in row if n >= 0)[:3]
        lam = _lam(mesh, tri, x, y)
        if (lam >= -1e-12).all():
            return tri, lam
    raise AssertionError(f"({x}, {y}) is outside the mesh")


def _uv_rule(mesh, placement, terms):
    """Hand u/v at stage ``k``: the per-vertex clipped depth rule with
    walls at depth (``_hand_wall``: a vertex whose bed lies above the stage
    depth is left out, the other weights renormalised — the 10 m node of
    seed 2 and the 25 m node of seed 3 below their beds) plus the stage's
    time term ``terms[k] = (tu, tv)``."""
    def uv(k, x, y, z):
        tri, lam = _tri_at(mesh, x, y)
        return (_hand_wall(mesh, placement, tri, lam, z,
                           lambda nd, zc: A_U + B_U * zc) + terms[k][0],
                _hand_wall(mesh, placement, tri, lam, z,
                           lambda nd, zc: A_V + B_V * zc) + terms[k][1])
    return uv


def _g(t, quad):
    return t * t / 3600.0 if quad else t


def _snap(placement, t, quad=False):
    """The fixture field at time ``t``: ``_field`` (NaN below last, +inf
    in one column) plus the time term, uniform ``w = W0``."""
    f = _field(placement)
    eu, ev = (Q_U, Q_V) if quad else (E_U, E_V)
    g = _g(t, quad)
    return dataclasses.replace(f, u=f.u + eu * g, v=f.v + ev * g,
                               w=np.full((K, N_NODES), W0))


def _cell_payload(mode, placement, td, quad=False):
    """``(vel, kwargs, terms)`` of one step: ``terms[k]`` is the hand time
    term of RK4 stage ``k`` (stage times ``s + td c_k dt``).  Linear data
    are reproduced exactly by the lerp and by PCHIP; quadratic data (Euler)
    get the LINEAR blend of the bracket at ``s``."""
    eu, ev = (Q_U, Q_V) if quad else (E_U, E_V)
    if mode == "steady":
        g = _g(3600.0, quad)
        return _snap(placement, 3600.0, quad), {}, [(eu * g, ev * g)] * 4
    s = T_START[td]
    snaps = [_snap(placement, t, quad) for t in RECORDS]
    vel, nxt, dtn = _payload(snaps[1], snaps[2], RECORDS[1], RECORDS[2], s,
                             td)
    kw = dict(vel_next=nxt, dt_to_next=dtn)
    if mode == "pchip":
        kw["recon"] = (snaps, list(RECORDS), s, int(td))
    if quad:
        a = (s - RECORDS[1]) / (RECORDS[2] - RECORDS[1])
        g = (1.0 - a) * _g(RECORDS[1], True) + a * _g(RECORDS[2], True)
        return vel, kw, [(eu * g, ev * g)] * 4
    return vel, kw, [(eu * (s + td * c * DT_STEP), ev * (s + td * c * DT_STEP))
                     for c in C_RK4]


def _seed_state(mesh, search):
    lon, lat, z = (np.array(c, dtype=np.float64) for c in zip(*SEEDS))
    st = make_state(lon, lat, z)
    if search == "walk":
        nn = find_nearest_nodes(mesh, st.lon, st.lat)
        e, ok = find_containing_elements(mesh, st.lon, st.lat, nn)
        assert ok.all()
        st = st.replace(elem=np.asarray(e, dtype=np.int32))
    return st


@contextlib.contextmanager
def _route_spies():
    """Count the launches of the step routes (module attributes of the
    integrators and ``make_stage_vels_recon``), restored afterwards."""
    calls = {}
    saved = []
    targets = [(I, nm) for nm in _DL_FUSED + _LAYER_FUSED + _UNFUSED]
    targets.append((TR, "make_stage_vels_recon"))
    for mod, nm in targets:
        orig = getattr(mod, nm)
        calls[nm] = 0

        def spy(*a, _nm=nm, _orig=orig, **k):
            calls[_nm] += 1
            return _orig(*a, **k)
        saved.append((mod, nm, orig))
        setattr(mod, nm, spy)
    try:
        yield calls
    finally:
        for mod, nm, orig in saved:
            setattr(mod, nm, orig)


#: a step cell: (integrator, w_interp, mode, placement, time_dir, search)
def _cell_id(cell):
    integ, w, mode, placement, td, search = cell
    return "-".join((integ, w, mode, placement, _td_id(td), search))


def _run_step_cell(cell):
    """One step of ``cell``: the state fields and the route counts."""
    integ, w_interp, mode, placement, td, search = cell
    mesh = _mesh()
    vel, kw, _ = _cell_payload(mode, placement, td, quad=integ == "euler")
    st = _seed_state(mesh, search)
    with _route_spies() as calls:
        out = STEPS[integ](st, vel, mesh, DT_STEP, td,
                           uv_interp="barycentric", w_interp=w_interp, **kw)
    res = {f: np.asarray(getattr(out, f)).tolist() for f in _STATE_FIELDS}
    res["calls"] = dict(calls)
    return res


def _fused_expected(integ, w_interp, search):
    """The fused 3-D RK2 gate for these cells: node layout, barycentric
    u/v and z-level always hold; walk (state.elem) and leastsq/idw W keep
    the unfused path; Euler and RK4 are always unfused here."""
    return (integ == "rk2" and search == "local"
            and w_interp in ("nearest", "barycentric"))


def _check_step_cell(cell, res):
    """The hand rule and the route of one step cell; raises
    AssertionError naming the cell."""
    integ, w_interp, mode, placement, td, search = cell
    name = _cell_id(cell)
    mesh = _mesh()
    _, _, terms = _cell_payload(mode, placement, td, quad=integ == "euler")
    st = _seed_state(mesh, search)
    out = SimpleNamespace(**{f: np.asarray(res[f]) for f in _STATE_FIELDS})
    dlon, dlat, u1, v1 = _hand_all(integ, st, DT_STEP, td, W0,
                                   _uv_rule(mesh, placement, terms))
    assert (out.status == ParticleStatus.ALIVE).all(), (name, out.status)
    _assert_step(out, st, dlon, dlat)
    np.testing.assert_allclose(out.u, u1, rtol=1e-12, atol=1e-15,
                               err_msg=name)
    np.testing.assert_allclose(out.v, v1, rtol=1e-12, atol=1e-15,
                               err_msg=name)
    np.testing.assert_allclose(out.w, td * W0, rtol=1e-12, err_msg=name)
    # depth: z0 + dt W, clamped to the surface and the bed of the end
    # node (the hand nearest node: shortest chord on the unit sphere)
    lam_r, phi_r = np.deg2rad(out.lon), np.deg2rad(out.lat)
    xyz_p = np.stack([np.cos(phi_r) * np.cos(lam_r),
                      np.cos(phi_r) * np.sin(lam_r), np.sin(phi_r)], 1)
    ln, pn = np.deg2rad(mesh.nodes_lon), np.deg2rad(mesh.nodes_lat)
    xyz_n = np.stack([np.cos(pn) * np.cos(ln), np.cos(pn) * np.sin(ln),
                      np.sin(pn)], 1)
    nn = np.argmin(((xyz_p[:, None, :] - xyz_n[None]) ** 2).sum(-1), 1)
    assert_bitwise(out.nearest_node.astype(np.int64), nn.astype(np.int64),
                   f"{name} nearest node")
    z_hand = np.clip(st.z + td * W0 * DT_STEP, 0.0, mesh.depth[nn])
    np.testing.assert_allclose(out.z, z_hand, rtol=1e-13, atol=1e-12,
                               err_msg=name)
    # the route
    calls = res["calls"]
    assert all(calls[k] == 0 for k in _LAYER_FUSED), (name, calls)
    if _fused_expected(integ, w_interp, search):
        want = _DL_FUSED[1] if mode == "pchip" else _DL_FUSED[0]
        assert calls[want] == 1, (name, calls)
        assert sum(calls[k] for k in _DL_FUSED) == 1, (name, calls)
        assert sum(calls[k] for k in _UNFUSED) == 0, (name, calls)
    else:
        assert sum(calls[k] for k in _DL_FUSED) == 0, (name, calls)
        assert calls[f"_{integ}_step_unfused"] == 1, (name, calls)
    # PCHIP: the multistage unfused steps build their stages from the
    # bundle; Euler never reads it (one stage at the step start)
    recon_used = (mode == "pchip" and integ != "euler"
                  and not _fused_expected(integ, w_interp, search))
    assert (calls["make_stage_vels_recon"] > 0) == recon_used, (name, calls)


#: row 1: rk2 / rk4.  The full product of the route axes: every cell costs
#: milliseconds, and each axis changes what runs — w_interp and
#: elem_search decide fused vs unfused RK2 (and the W kernel inside the
#: fused twin), the time mode picks the kernel or stage builder, the
#: placement the table and the field-shape branch, time_dir the payload
#: bracket and the sign.
RK_CELLS = [(i, w, m, p, td, s) for i in ("rk2", "rk4") for w in W_INTERPS
            for m in MODES for p in PLACEMENTS for td in TDS
            for s in SEARCHES]
#: row 2: Euler, the same axes (Euler is always unfused here; w_interp
#: still picks the W kernel, elem_search the elements hint).
EULER_CELLS = [("euler", w, m, p, td, s) for w in W_INTERPS for m in MODES
               for p in PLACEMENTS for td in TDS for s in SEARCHES]


#: the unfused PCHIP stages reconstruct whole fields, the fixture's +inf
#: below a last point included (inf - inf: numpy warns; that slot is
#: never read — the cells check it)
_PCHIP_INF = pytest.mark.filterwarnings(
    "ignore:invalid value encountered in subtract:RuntimeWarning")


@_PCHIP_INF
@pytest.mark.parametrize("cell", RK_CELLS, ids=[_cell_id(c)
                                                for c in RK_CELLS])
def test_rk_rule(cell):
    """RK2 / RK4: one step equals the hand stage combination, through the
    fused depth-linear twin where the gate allows, else the unfused
    stages."""
    _check_step_cell(cell, _run_step_cell(cell))


@pytest.mark.parametrize("cell", EULER_CELLS, ids=[_cell_id(c)
                                                   for c in EULER_CELLS])
def test_euler_rule(cell):
    """Euler: one step at the entry depth and the step-start time; with a
    PCHIP bundle passed the velocity is still the LINEAR blend of the
    bracket (Euler never reads the bundle), checked on snapshots that are
    quadratic in time."""
    _check_step_cell(cell, _run_step_cell(cell))


def test_euler_quadratic_snapshots_tell_pchip_from_linear():
    """The quadratic snapshots of the Euler cells are far from linear in
    time: PCHIP at the step start differs from the linear blend by much
    more than the cells' tolerance, so those cells can tell them apart."""
    g = [_g(t, True) for t in RECORDS]
    for td, s in T_START.items():
        a = (s - RECORDS[1]) / (RECORDS[2] - RECORDS[1])
        lin = (1.0 - a) * g[1] + a * g[2]
        pch = TR._pchip_value_scalar(g[0], g[1], g[2], g[3], *RECORDS, s)
        dlon = DT_STEP * Q_U * abs(pch - lin) / M
        assert dlon > 1e-6, (td, lin, pch)


# ---------------------------------------------------------------------------
# Row 2, run level: Euler with temporal_reconstruction "pchip"
# ---------------------------------------------------------------------------

#: records 0..5, daily; u = A + B z + Q2 r**2 at the layer centres (r =
#: record index): dyadic, exact in float32, quadratic in time
Q2 = 2.0 ** -10


def _option():
    """The FESOM2 fixture module (imported lazily: it needs xarray,
    netCDF4 and healpy)."""
    pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    pytest.importorskip("healpy")
    from tests import test_uv_vertical_option as o
    return o


def _quad_u_values(o):
    def u_vals(year, k):
        recs = [r for r, y in enumerate(o.REC_YEAR) if y == year]
        col = np.array([o.A + o.B * o.Z_MID[k] + Q2 * r * r for r in recs])
        return np.repeat(col[:, None], 4, axis=1)
    return u_vals


#: run seeds (4-node FESOM2 square, every bed 30 m): depths above, between
#: and below the layer centres 5 and 20 m; mid-square, so six quarter-day
#: steps (about 0.12 deg) stay inside in both directions
RUN_SEEDS = [(0.45, 0.50, 3.0), (0.50, 0.40, 12.5), (0.42, 0.60, 26.0)]
RUN_STEPS = 6


def _euler_run_cell(work, w_interp, td, search):
    """A run_single_realization with PCHIP configured (Euler, dt = 1/4
    record, the steps start between records): ``(lon (step, particle),
    the recon argument of every step call)``."""
    o = _option()
    from kelp import tracking
    from kelp.core.config import NumericsConfig
    from kelp.readers import extract_time_info
    cfg = o._fesom2_cfg(work, seeds=RUN_SEEDS, u_values=_quad_u_values(o),
                        uv_vertical="linear", time_dir=td)
    reader, mesh, ds = o._fesom2_env(cfg)
    file_times, file_dt = extract_time_info(ds)
    numerics = NumericsConfig(
        dt=o.DT / 4.0, integrator="euler", uv_interp="barycentric",
        w_interp=w_interp, temporal_interp=True,
        temporal_reconstruction="pchip", elem_search=search)
    seen = []
    real = I.euler_step

    def spy(state, vel, mesh_, dt, time_dir, **kw):
        seen.append(kw.get("recon") is not None)
        return real(state, vel, mesh_, dt, time_dir, **kw)
    old = tracking.get_integrator
    tracking.get_integrator = lambda name: spy
    try:
        rec = tracking.run_single_realization(
            mesh, ds, reader, o._state(mesh, RUN_SEEDS), cfg.mode, numerics,
            start_time_idx=1 if td > 0 else 4, n_steps=RUN_STEPS,
            temporal_interp=True, file_times=file_times, file_dt=file_dt)
    finally:
        tracking.get_integrator = old
        ds.close()
    return {"lon": rec.to_arrays()["lon"].tolist(), "recon_seen": seen}


def _euler_run_hand(o, td, pchip=False):
    """Hand positions ``(step, particle)``: u at the step start, the time
    term linear (or PCHIP) between the bracketing records."""
    g = [float(r * r) for r in range(6)]
    t0 = 1.0 if td > 0 else 4.0
    rows, x = [], np.array([s[0] for s in RUN_SEEDS])
    for j in range(RUN_STEPS):
        s = t0 + td * 0.25 * j                       # in records
        r = int(np.floor(s)) if td > 0 else int(np.ceil(s)) - 1
        r = min(max(r, 0), 4)
        a = s - r
        if pchip and 1 <= r <= 3:
            gs = TR._pchip_value_scalar(g[r - 1], g[r], g[r + 1], g[r + 2],
                                        r - 1.0, float(r), r + 1.0, r + 2.0,
                                        s)
        else:
            gs = (1.0 - a) * g[r] + a * g[r + 1]
        u = np.array([o.A + o.B * min(max(z, 5.0), 20.0) + Q2 * gs
                      for _, _, z in RUN_SEEDS])
        x = x + td * np.array([o._dlon(ui, la) / 4.0
                               for ui, (_, la, _) in zip(u, RUN_SEEDS)])
        rows.append(x.copy())
    return np.array(rows)


def _check_euler_run_cell(cell, res, o):
    w_interp, td, search = cell
    name = f"euler-run-{w_interp}-{_td_id(td)}-{search}"
    lon = np.asarray(res["lon"])
    assert res["recon_seen"] == [False] * RUN_STEPS, (name, res)
    lin = _euler_run_hand(o, td)
    np.testing.assert_allclose(lon, lin, rtol=0.0, atol=1e-11, err_msg=name)
    # PCHIP in time would have moved them elsewhere
    pch = _euler_run_hand(o, td, pchip=True)
    assert np.abs(pch - lin).max() > 1e-6, name


#: placement: "layer_linear" only — a run marks the reader's layer fields so,
#: and no reader of this version delivers "level" fields; the tracking payload
#: (a lazy blend of cached records) never looks at the placement, and the
#: step-level Euler cells above cover "level"
EULER_RUN_CELLS = [(w, td, s) for w in W_INTERPS for td in TDS
                   for s in SEARCHES]


@pytest.fixture(scope="module")
def quad_work(tmp_path_factory):
    """The FESOM2 dataset with quadratic-in-time records, written once."""
    o = _option()
    work = tmp_path_factory.mktemp("matrix_quad")
    o._fesom2_cfg(work, seeds=RUN_SEEDS, u_values=_quad_u_values(o))
    return work


@pytest.mark.parametrize("cell", EULER_RUN_CELLS, ids=[
    f"{w}-{_td_id(td)}-{s}" for w, td, s in EULER_RUN_CELLS])
def test_euler_pchip_configured_is_linear_in_time(quad_work, cell):
    """``numerics.temporal_reconstruction: pchip`` with Euler: the tracking
    layer builds no PCHIP stencil for a one-stage integrator, so every
    step reads the LINEAR blend of its bracket (steps start between
    records; the records are quadratic in time)."""
    o = _option()
    w_interp, td, search = cell
    _check_euler_run_cell(cell, _euler_run_cell(quad_work, w_interp, td,
                                                search), o)


def test_pchip_stencil_built_for_rk2(quad_work):
    """The same run with RK2 does get the bundle (the contrast of the
    Euler cells)."""
    o = _option()
    from kelp import tracking
    from kelp.core.config import NumericsConfig
    from kelp.readers import extract_time_info
    cfg = o._fesom2_cfg(quad_work, seeds=RUN_SEEDS,
                        u_values=_quad_u_values(o))
    reader, mesh, ds = o._fesom2_env(cfg)
    file_times, file_dt = extract_time_info(ds)
    numerics = NumericsConfig(
        dt=o.DT / 4.0, integrator="rk2", uv_interp="barycentric",
        w_interp="barycentric", temporal_interp=True,
        temporal_reconstruction="pchip")
    seen = []
    real = I.rk2_step

    def spy(state, vel, mesh_, dt, time_dir, **kw):
        seen.append(kw.get("recon") is not None)
        return real(state, vel, mesh_, dt, time_dir, **kw)
    old = tracking.get_integrator
    tracking.get_integrator = lambda name: spy
    try:
        tracking.run_single_realization(
            mesh, ds, reader, o._state(mesh, RUN_SEEDS), cfg.mode, numerics,
            start_time_idx=1, n_steps=4, temporal_interp=True,
            file_times=file_times, file_dt=file_dt)
    finally:
        tracking.get_integrator = old
        ds.close()
    assert seen == [True] * 4


# ---------------------------------------------------------------------------
# Numba off: one subprocess per family
# ---------------------------------------------------------------------------

_SUBPROCESS_SCRIPT = r"""
import contextlib, io, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).resolve().parents[0]))
sys.path.insert(0, sys.argv[1])
from numba.core import config
assert config.DISABLE_JIT, "NUMBA_DISABLE_JIT not honoured"
import kelp.numerics.integrators as I
assert str(Path(I.__file__).resolve()).startswith(
    str(Path(sys.argv[1]).resolve())), I.__file__
import tests.test_uv_vertical_matrix as t
with contextlib.redirect_stdout(io.StringIO()):      # reader progress lines
    rows = t._family_results(sys.argv[2], sys.argv[3])
print(json.dumps(rows))
"""


def _family_results(family, work):
    """Every cell of ``family`` run in this process (the subprocess entry
    point): ``{cell id: result}``."""
    if family == "rk":
        return {_cell_id(c): _run_step_cell(c) for c in RK_CELLS}
    if family == "euler":
        out = {_cell_id(c): _run_step_cell(c) for c in EULER_CELLS}
        for w, td, s in EULER_RUN_CELLS:
            out[f"run-{w}-{_td_id(td)}-{s}"] = _euler_run_cell(
                Path(work), w, td, s)
        return out
    raise ValueError(family)


@pytest.mark.parametrize("family", ["rk", "euler"])
def test_numba_off_subprocess(family, quad_work, tmp_path):
    """The ``Numba off`` column: the family's cells under
    ``NUMBA_DISABLE_JIT=1`` (the kernels run as Python; the fused RK2
    twins too) meet the same hand values and take the same routes."""
    o = _option() if family == "euler" else None
    script = tmp_path / "matrix_nojit.py"
    script.write_text(_SUBPROCESS_SCRIPT)
    src = str(Path(__file__).resolve().parents[1] / "src")
    env = dict(os.environ, NUMBA_DISABLE_JIT="1")
    out = subprocess.run(
        [sys.executable, str(script), src, family, str(quad_work)],
        capture_output=True, text=True, timeout=900, env=env)
    assert out.returncode == 0, out.stderr[-4000:]
    rows = json.loads(out.stdout)
    failures = []
    cells = RK_CELLS if family == "rk" else EULER_CELLS
    for c in cells:
        try:
            _check_step_cell(c, rows[_cell_id(c)])
        except AssertionError as exc:
            failures.append(f"{_cell_id(c)}: {exc}")
    if family == "euler":
        for c in EULER_RUN_CELLS:
            w, td, s = c
            try:
                _check_euler_run_cell(c, rows[f"run-{w}-{_td_id(td)}-{s}"], o)
            except AssertionError as exc:
                failures.append(f"run {c}: {exc}")
    assert not failures, "\n".join(failures[:10])
    expected = len(cells) + (len(EULER_RUN_CELLS) if family == "euler"
                             else 0)
    assert len(rows) == expected


# ---------------------------------------------------------------------------
# Row 3: fused RK2, anchor heal and terminal splice
# ---------------------------------------------------------------------------

def _fused():
    pytest.importorskip("healpy")
    from tests import test_fused_3d_depth_linear as f
    return f


def _band_mesh(f):
    """The structured mesh of the hole-band fixture, but with ONE row of
    tall elements spanning 70 -> 73 N instead of the hole: no node there,
    so a stage anchor in the band has an empty hash window (flagged, then
    healed or spliced), while the band is still inside the mesh and the
    particle stays ALIVE."""
    from kelp.readers.structured import build_structured_mesh
    lon1d = np.arange(0.0, 30.0, 0.25)
    lat1d = np.concatenate([np.arange(60.0, 70.0 + 1e-9, 0.25),
                            np.arange(73.0, 80.0, 0.25)])
    lon2d, lat2d = np.meshgrid(lon1d, lat1d)
    m = build_structured_mesh(lon2d, lat2d, np.ones_like(lon2d, bool))
    m.sigma = f.Z_LEVELS
    m.depth = np.full(m.n_nodes, 3500.0)
    return m


@pytest.mark.parametrize("route", ["heal", "splice"])
@pytest.mark.parametrize("mode", ["tinterp", "pchip"])
@pytest.mark.parametrize("placement", PLACEMENTS)
def test_rk2_fused_heal_and_splice_rule(monkeypatch, placement, mode, route):
    """The hole-band fixture of ``test_fused_3d_depth_linear`` on the band
    mesh: half the particles have both stage anchors in the band.  With
    the heal rounds they re-run through the same depth-linear twin; with
    the heal rounds off they are spliced from the unfused step.  Either
    way every particle ends ALIVE at the hand RK2 position of
    ``u = 0.2 + 1e-4 z``, ``v = v0(t) + 1e-3 z`` (every node 3500 m deep:
    one clip for all vertices).  Forward only, barycentric W only (the
    fixture's): the flag/heal handshake reads neither time_dir nor the W
    method (row 1 runs the fused twins backward and with both W kernels)."""
    f = _fused()
    m = _band_mesh(f)
    st = f._hole_seeds(m)
    if route == "splice":
        monkeypatch.setattr(I, "_FLAG_HEAL_ROUNDS", 0)
    calls = f._spy(monkeypatch, f._DL_KERNELS + f._LAYER_KERNELS
                   + ("_splice_exact_rerun",))
    got, ref, kern, _ = f._hole_run(m, placement, mode, st)
    assert all(calls[k] == 0 for k in f._LAYER_KERNELS), calls
    if route == "heal":
        assert calls[kern] >= 2 and calls["_splice_exact_rerun"] == 0, calls
    else:
        assert calls[kern] == 1 and calls["_splice_exact_rerun"] == 1, calls
    dp = depth_points(m, placement)
    p0, pl = float(dp.p[0]), float(dp.p[dp.last.max()])
    assert (dp.last == dp.last.max()).all()
    t_k = [30000.0 + c * f.DT for c in C_RK4]

    def uv(k, x, y, z):
        zc = min(max(z, p0), pl)
        v0 = 45.0 if mode == "tinterp" else 44.0 + t_k[k] / 21600.0
        return 0.2 + 1e-4 * zc, v0 + 1e-3 * zc
    assert (got.status == ParticleStatus.ALIVE).all()
    flagged_band = st.lat > 69.0
    assert flagged_band.sum() >= 90
    assert (got.lat[flagged_band] > 70.5).all()
    assert (got.lat[flagged_band] < 72.9).all()
    for i in range(st.n_particles):
        dlon, dlat, (u1, v1) = _hand_rk2(st.lon[i], st.lat[i], st.z[i], f.DT,
                                         1.0, 2e-3, uv)
        np.testing.assert_allclose(got.lon[i] - st.lon[i], dlon, rtol=1e-12)
        np.testing.assert_allclose(got.lat[i] - st.lat[i], dlat, rtol=1e-12)
        np.testing.assert_allclose(got.u[i], u1, rtol=1e-13)
        np.testing.assert_allclose(got.v[i], v1, rtol=1e-13)
    np.testing.assert_allclose(got.z, st.z + f.DT * 2e-3, rtol=1e-14)


# ---------------------------------------------------------------------------
# Row 4: rk45 refuses depth-linear fields
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("uv_interp", ["barycentric", "leastsq"])
@pytest.mark.parametrize("placement", PLACEMENTS)
def test_rk45_refuses(placement, uv_interp, mode):
    """``rk45_step`` refuses a depth-linear field first, whatever the
    interpolator and the time payload (``_check_depth_linear_entry``)."""
    mesh = _mesh()
    vel, kw, _ = _cell_payload(mode, placement, 1.0)
    st = _seed_state(mesh, "local")
    with pytest.raises(ValueError, match=r"^rk45_step: depth-linear "
                       rf"horizontal velocity \(uv_vertical='{placement}'\) "
                       r"is not supported by the rk45 integrator"):
        I.rk45_step(st, vel, mesh, DT_STEP, 1.0, uv_interp=uv_interp,
                    w_interp="barycentric", **kw)


def test_rk45_run_with_option_refused_before_output(tmp_path):
    """A run asking for rk45 and the option is refused before any output
    (rk45 is 2-D only)."""
    o = _option()
    cfg = o._fesom2_cfg(tmp_path, integrator="rk45")
    with pytest.raises(NotImplementedError, match="2-D only"):
        o._run(cfg)
    assert not Path(cfg.data.output_path).exists()


# ---------------------------------------------------------------------------
# Row 5: leastsq / edge_leastsq u/v refuse depth-linear fields
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("integrator", ["euler", "rk2", "rk4"])
@pytest.mark.parametrize("uv_interp", ["leastsq", "edge_leastsq"])
@pytest.mark.parametrize("placement", PLACEMENTS)
def test_non_barycentric_uv_interp_refused(placement, uv_interp, integrator):
    mesh = _mesh()
    vel, kw, _ = _cell_payload("lerp", placement, 1.0)
    st = _seed_state(mesh, "local")
    with pytest.raises(ValueError, match=rf"^{integrator}_step: depth-linear "
                       r"horizontal velocity .* needs "
                       r"uv_interp='barycentric', got "
                       rf"uv_interp='{uv_interp}'"):
        STEPS[integrator](st, vel, mesh, DT_STEP, 1.0, uv_interp=uv_interp,
                          w_interp="barycentric", **kw)


@pytest.mark.parametrize("uv_interp", ["leastsq", "edge_leastsq"])
def test_non_barycentric_uv_interp_refused_by_the_option(tmp_path,
                                                         uv_interp):
    """The option with a non-barycentric interpolator on node-layout u/v
    is refused at the pre-writer layout probe (no output).  Right after the
    mesh is loaded the layout is not known yet, and both interpolators are
    in the layout table (element layout)."""
    o = _option()
    from kelp import tracking
    tracking._check_uv_vertical_option(
        "linear", fdrift3d=True, mesh=o._zmesh(), uv_interp=uv_interp,
        reader=None)
    cfg = o._fesom2_cfg(tmp_path, uv_interp=uv_interp)
    with pytest.raises(ValueError, match="node-based velocities .* require "
                                         "numerics.uv_interp: 'barycentric'"):
        o._run(cfg)
    assert not Path(cfg.data.output_path).exists()


# ---------------------------------------------------------------------------
# Row 6: split uniform Kh and az_const diffusion: the advection part
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kick", ["kh", "az", "kh+az"])
@pytest.mark.parametrize("integrator", ["euler", "rk2", "rk4"])
@pytest.mark.parametrize("placement", PLACEMENTS)
def test_split_uniform_diffusion_advection_part(placement, integrator, kick):
    """A split-composition step with uniform Kh and/or a constant Az is
    the advection step (the hand rule) followed by the kicks, bit for bit:
    the depth-linear advection inside a diffusion step is the rule."""
    mesh = _mesh()
    td = 1.0
    vel, kw, terms = _cell_payload("lerp", placement, td,
                                   quad=integrator == "euler")
    st = _seed_state(mesh, "local")
    kh = 25.0 if "kh" in kick else 0.0
    args = DiffusionStepArgs(
        scheme="uniform", kh_x=kh, kh_y=kh, seed=2026, real_id=0, step_i=3,
        composition="split", az_enabled="az" in kick,
        az_const=2e-3 if "az" in kick else 0.0)
    kws = dict(uv_interp="barycentric", w_interp="barycentric", **kw)
    adv = STEPS[integrator](st, vel, mesh, DT_STEP, td, **kws)
    dlon, dlat, u1, v1 = _hand_all(integrator, st, DT_STEP, td, W0,
                                   _uv_rule(mesh, placement, terms))
    _assert_step(adv, st, dlon, dlat)
    got = STEPS[integrator](st, vel, mesh, DT_STEP, td, diffusion=args,
                            **kws)
    want = I._apply_split_diffusion_kick(st, adv, mesh, DT_STEP, args, vel)
    want = I._apply_split_az_kick(st, want, mesh, DT_STEP, args, vel,
                                  time_dir=td)
    for f in _STATE_FIELDS:
        assert_bitwise(getattr(got, f), getattr(want, f), f)
    # the kicks did move the particles
    if kh:
        assert np.all(got.lon != adv.lon)
    if "az" in kick:
        assert np.any(got.z != adv.z)


# ---------------------------------------------------------------------------
# Row 7: em / m1 in every composition refuse in 3-D
# ---------------------------------------------------------------------------

#: (integrator, composition) -> the guard that fires and its message
_EM_GUARDS = {
    ("euler", "split"): "with composition='split' is unsupported",
    ("euler", "combined"): "is 2D-only",
    ("euler", "split_rk"): "with composition='split' is unsupported",
    ("rk2", "split"): "requires composition='split_rk'",
    ("rk2", "combined"): "requires composition='split_rk'",
    ("rk2", "split_rk"): "'split_rk' is 2D-only",
    ("rk4", "split"): "requires composition='split_rk'",
    ("rk4", "combined"): "requires composition='split_rk'",
    ("rk4", "split_rk"): "'split_rk' is 2D-only",
}


@pytest.mark.parametrize("scheme", ["em", "m1"])
@pytest.mark.parametrize("integ_comp", sorted(_EM_GUARDS),
                         ids=lambda ic: "-".join(ic))
@pytest.mark.parametrize("placement", PLACEMENTS)
def test_em_m1_refused_in_3d(placement, integ_comp, scheme):
    """Variable-Kh schemes stay 2-D: on a depth-linear 3-D field each
    composition hits its existing guard (the entry check lets the field
    through first)."""
    integrator, comp = integ_comp
    mesh = _mesh()
    vel, kw, _ = _cell_payload("lerp", placement, 1.0)
    st = _seed_state(mesh, "local")
    args = DiffusionStepArgs(scheme=scheme, kh_x=0.0, kh_y=0.0, seed=1,
                             real_id=0, step_i=0, composition=comp)
    with pytest.raises(NotImplementedError,
                       match=_EM_GUARDS[(integrator, comp)]):
        STEPS[integrator](st, vel, mesh, DT_STEP, 1.0, diffusion=args,
                          uv_interp="barycentric", w_interp="barycentric",
                          **kw)
    # with az on, every em/m1 shape is refused before anything else
    with pytest.raises(ValueError, match="composes with scheme='uniform'"):
        STEPS[integrator](st, vel, mesh, DT_STEP, 1.0,
                          diffusion=args._replace(az_enabled=True,
                                                  az_const=1e-3),
                          uv_interp="barycentric", w_interp="barycentric",
                          **kw)


# ---------------------------------------------------------------------------
# Row 8: beaching with resuspension
# ---------------------------------------------------------------------------

#: node 1 of the 4-node square is 10 m deep (the others 30 m): its layer
#: table keeps one centre (5 m), the others two (5, 20 m)
_BEACH_LAST = np.array([1, 0, 1, 1])
_BEACH_P = np.array([5.0, 20.0])
#: three seeds nearest node 1 (they beach), two elsewhere
_BEACH_SEEDS = [(0.55, 0.15, 7.0), (0.6, 0.3, 4.0), (0.7, 0.1, 9.0),
                (0.15, 0.6, 12.0), (0.3, 0.8, 25.0)]


def _beach_uv(o, mesh, terms):
    def uv(k, x, y, z):
        tri, lam = _tri_at(mesh, x, y)
        zc = np.array([min(max(z, _BEACH_P[0]), _BEACH_P[_BEACH_LAST[i]])
                       for i in tri])
        return float(lam @ (o.A + o.B * zc)) + terms[k], 0.0
    return uv


@pytest.mark.parametrize("integrator", ["euler", "rk2"])
def test_beaching_resuspension_rule(tmp_path, monkeypatch, integrator):
    """Depth beaching (node 1, 10 m, is shallower than ``depth_min``
    except at high tide) with auto-or-stochastic resuspension, option on:
    a particle ALIVE at the start of a step moves by the hand rule
    (per-vertex clip at the 10 m node, the year term of the records), a
    BEACHED one keeps its position bit for bit; the particles near node 1
    beach, are resuspended by the tide and move again."""
    from kelp.core.config import BeachingConfig
    from kelp.writers import load_trajectories
    o = _option()
    monkeypatch.setattr(o.synth, "ZBAR_N_BOTTOM_NEG",
                        np.array([-30.0, -10.0, -30.0, -30.0]))
    n_steps = 5
    # a tide: the 10 m node is wet (H > 12.1 m) only in records 2 and 3
    data = tmp_path / "data"
    o._write_fesom2(data)
    eta = {1958: np.array([[0.0], [0.0], [4.0]]),
           1959: np.array([[4.0], [0.0], [0.0]])}
    o.synth.write_multiyear(data, "ssh", o.YEARS, vertical=None,
                            values=lambda y, k: eta[y], continuous_time=True)
    cfg = o._fesom2_cfg(tmp_path, seeds=_BEACH_SEEDS, n_steps=n_steps,
                        integrator=integrator, data=data)
    cfg.numerics.beaching = BeachingConfig(
        enabled=True, criteria=["depth"], depth_min=12.0, use_eta=True,
        resuspension="auto_or_stochastic", t_resuspend=o.DT, seed=11)
    o._run(cfg)
    traj = load_trajectories(cfg.data.output_path)
    reader, mesh, ds = o._fesom2_env(cfg)
    ds.close()
    np.testing.assert_array_equal(
        depth_points(mesh, "layer_linear").last, _BEACH_LAST)
    year_term = [o.C * (y - o.Y0) for y in o.REC_YEAR]
    prev = {k: np.array([s[i] for s in _BEACH_SEEDS])
            for i, k in enumerate(("lon", "lat", "z"))}
    prev_status = np.full(len(_BEACH_SEEDS), ParticleStatus.ALIVE)
    moved_after_beaching = frozen = 0
    was_beached = np.zeros(len(_BEACH_SEEDS), bool)
    for j in range(n_steps):
        # stage 1 at record j; RK2's stage 2 at the midpoint of the step
        # (the lerp of records j and j + 1)
        uv = _beach_uv(o, mesh, [year_term[j],
                                 0.5 * (year_term[j] + year_term[j + 1])])
        for i in range(len(_BEACH_SEEDS)):
            x0, y0, z0 = (prev[k][i] for k in ("lon", "lat", "z"))
            x1, y1, z1 = (traj[k][i, j] for k in ("lon", "lat", "z"))
            if prev_status[i] == ParticleStatus.ALIVE:
                if integrator == "rk2":
                    dlon = _hand_rk2(x0, y0, z0, o.DT, 1.0, 0.0, uv)[0]
                else:
                    dlon = _hand_euler(x0, y0, z0, o.DT, 1.0, uv)[0]
                np.testing.assert_allclose(x1 - x0, dlon, rtol=1e-12,
                                           atol=1e-14,
                                           err_msg=f"step {j} particle {i}")
                assert (y1, z1) == (y0, z0), (j, i)
                if was_beached[i]:
                    moved_after_beaching += 1
            else:
                assert prev_status[i] == ParticleStatus.BEACHED, (j, i)
                assert (x1, y1, z1) == (x0, y0, z0), (j, i)
                frozen += 1
        prev = {k: traj[k][:, j].copy() for k in ("lon", "lat", "z")}
        prev_status = traj["status"][:, j].copy()
        was_beached |= prev_status == ParticleStatus.BEACHED
    assert was_beached[:3].all() and not was_beached[3:].any()
    assert moved_after_beaching > 0 and frozen > 0


# ---------------------------------------------------------------------------
# Row 9: record-point sampling reads scalars and eta only
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sample_interp",
                         ["nearest", "leastsq", "barycentric", "idw"])
def test_sample_record_fields_ignores_uv_placement(sample_interp):
    """``_sample_record_fields`` on the SAME state with two valid
    containers of the same scalars, eta and w — a ``"level"`` field (u/v
    on K rows) and a ``"layer"`` field (dummy u/v on K-1 rows) — returns
    the same samples bit for bit: it reads the scalars and eta only."""
    from kelp import tracking
    mesh = _mesh()
    n = mesh.n_nodes
    rng = np.random.default_rng(31)
    eta = rng.normal(scale=0.3, size=n)
    w = rng.normal(scale=1e-3, size=(K, n))
    scalars = {"t_lev": rng.normal(size=(K, n)),
               "s_mid": rng.normal(size=(K - 1, n)),
               "ssh2": rng.normal(size=n)}
    level = VelocityField(u=rng.normal(size=(K, n)), v=rng.normal(size=(K, n)),
                          eta=eta, w=w, uv_layout="node",
                          uv_vertical="level", scalars=dict(scalars))
    layer = VelocityField(u=np.full((K - 1, n), 9.0),
                          v=np.full((K - 1, n), -9.0), eta=eta, w=w,
                          uv_layout="node", scalars=dict(scalars))
    lon = np.array([4.3, 5.6, 5.7, 4.9, 5.2])
    lat = np.array([4.2, 4.3, 5.6, 5.1, 4.5])
    z = np.array([12.5, 8.0, 22.0, 0.0, 33.0])
    st = make_state(lon, lat, z)
    st = st.replace(nearest_node=find_nearest_nodes(mesh, lon, lat))
    status = st.status.copy()
    status[3] = ParticleStatus.BEACHED
    st = st.replace(status=status)
    a = tracking._sample_record_fields(st, mesh, level, sample_interp)
    b = tracking._sample_record_fields(st, mesh, layer, sample_interp)
    assert sorted(a) == sorted(b) == sorted(scalars)
    for k in scalars:
        assert_bitwise(a[k], b[k], k)
        assert np.isfinite(a[k][[0, 1, 2, 4]]).all() and np.isnan(a[k][3])


# ---------------------------------------------------------------------------
# Row 10: prefetch windows x the shared snapshot cache
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shared", [False, True], ids=["local", "shared"])
@pytest.mark.parametrize("prefetch", [1, 3])
def test_prefetch_and_shared_cache_rule(tmp_path, monkeypatch, prefetch,
                                        shared):
    """RK2 with the next record, option on: every step function call
    receives ``"layer_linear"`` fields checked for the run's mesh (the
    read, the prefetch window and the shared cache keep the placement),
    and the positions are the hand sums of ``dt (u(r) + u(r + 1)) / 2``.
    With the shared cache a prefetch of 3 is forced to 1 (warned)."""
    o = _option()
    from kelp import tracking
    cfg = o._fesom2_cfg(tmp_path, integrator="rk2", prefetch=prefetch,
                        start=1, n_steps=4)
    if shared:
        shm = tmp_path / "shm"
        shm.mkdir()
        cfg.numerics.shared_fields_dir = str(shm)
        cfg.numerics.shared_fields_generation = f"matrix-{prefetch}"
    seen = []

    def spy(state, vel, mesh, dt, time_dir, **kw):
        vn = kw.get("vel_next")
        seen.append((vel.uv_vertical, vn.uv_vertical, is_checked(vel, mesh),
                     is_checked(vn, mesh)))
        return I.rk2_step(state, vel, mesh, dt, time_dir, **kw)
    monkeypatch.setattr(tracking, "get_integrator", lambda name: spy)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        o._run(cfg)
    msgs = [str(w.message) for w in rec]
    assert not any("shared_fields disabled" in m for m in msgs), msgs
    forced = any("prefetch_size is forced to 1" in m for m in msgs)
    assert forced == (shared and prefetch > 1), msgs
    assert seen == [("layer_linear", "layer_linear", True, True)] * 4
    from kelp.writers import load_trajectories
    traj = load_trajectories(cfg.data.output_path)
    hand = []
    for lo, la, z in o.SEEDS:
        x, row = lo, []
        for r in (1, 2, 3, 4):
            u = 0.5 * (o._u_lin(o.REC_YEAR[r], z)
                       + o._u_lin(o.REC_YEAR[r + 1], z))
            x = x + o._dlon(u, la)
            row.append(x)
        hand.append(row)
    np.testing.assert_allclose(traj["lon"], np.array(hand), rtol=0,
                               atol=1e-11)
    assert o._global_attrs(cfg.data.output_path)["uv_vertical"] == "linear"


# ---------------------------------------------------------------------------
# Row 11: a 2-D run with the option
# ---------------------------------------------------------------------------

def test_2d_run_with_option_refused():
    """The option needs 3-D tracking: refused by the configuration, by
    the run-entry check (a mode object that skipped the configuration),
    and by the cache marking of a 2-D field."""
    from kelp import tracking
    from kelp.core.config import ModeConfig
    with pytest.raises(ValueError, match=r"needs 3-D tracking "
                                         r"\(mode.Fdrift3D: true\)"):
        ModeConfig(Fdrift2D=True, Fdrift3D=False, uv_vertical="linear")
    with pytest.raises(ValueError, match="needs 3-D tracking"):
        tracking._check_uv_vertical_option(
            "linear", fdrift3d=False, mesh=_mesh(), uv_interp="barycentric",
            reader=None)
    n = N_NODES
    flat = VelocityField(u=np.ones(n), v=np.zeros(n), eta=np.zeros(n),
                         uv_layout="node")
    with pytest.raises(ValueError, match="needs a 3-D field"):
        tracking._mark_depth_linear(flat, "linear", _mesh(), record="r")


def test_2d_direct_entry_refused_before_any_read():
    """``run_single_realization`` with a 2-D mode carrying the option (a
    mode object built without ``ModeConfig``) is refused before the reader
    is asked for anything."""
    from kelp import tracking
    from kelp.core.config import NumericsConfig

    class Never:
        def read_velocities(self, *a, **k):
            raise AssertionError("read before the refusal")
    mesh = _mesh()
    mode = SimpleNamespace(Fdrift2D=True, Fdrift3D=False, time_dir=1.0,
                           uv_vertical="linear", max_levels=None)
    st = make_state(np.array([5.0]), np.array([5.0]), np.array([0.0]))
    with pytest.raises(ValueError, match="needs 3-D tracking"):
        tracking.run_single_realization(
            mesh, None, Never(), st, mode,
            NumericsConfig(dt=DT_STEP, uv_interp="barycentric"),
            start_time_idx=0, n_steps=1, n_time_total=2)
