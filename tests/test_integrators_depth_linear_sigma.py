"""Depth-linear u/v on FESOM-C sigma meshes in the integrators.

On a sigma mesh the depth-linear coordinate is the sigma fraction of the
particle's column, ``s = clip(z / H_p, 0, 1)`` (``H_p = max(depth + eta,
Dmin)`` at the particle, interpolated like W: ``integrators._sigma_col_H``),
and the data points are the layer centres in ``s``
(``vertical_uv.depth_points``).  Every unfused stage hands the u/v
interpolator the ``s`` of its own depth, position and time
(``integrators._sigma_frac_for_uv``): Euler ``z / H_0``, RK2 ``z / H_1``
and then ``z_mid / H_2``, RK4 ``z_k / H_k`` at each of its four stages.  W,
the layers for W and the integration of ``s`` keep metres.  No fused
kernel serves sigma 3-D.

All synthetic: FESOM-C files written as in ``tests/test_fesomc_w_contract``
(its 3x3-node grid of 8 triangles; element u/v ``(time, nele, sigmam1)``)
and read by the FESOM-C reader, with a bed linear in lon and lat (columns of
unequal height), a dia-sigma W linear in ``s`` (nonzero everywhere, so
``s`` changes within a step) and ``u = a + b*s (+ e*t)`` in every column
(``Rule``).  The column height, W and u/v are then exact at any position
(every interpolator reproduces a plane; the particles of the exact tests
stay in the two elements without a coast edge), so every expected position
is a hand-written stage combination (``_hand``): each stage's ``s`` from
its own position, time and ``H_p``.  A spy on the interpolators
(``_Reads``) records the coordinate every stage hands them.
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("xarray")

from kelp.core.geometry import DEG2RAD, METERS_PER_DEG_LAT  # noqa: E402
from kelp.core.io import LazyLerpField, VelocityField  # noqa: E402
from kelp.core.mesh import (  # noqa: E402
    find_containing_elements,
    find_nearest_nodes,
)
from kelp.core.particles import ParticleStatus, make_state  # noqa: E402
from kelp.numerics import integrators as I  # noqa: E402
from kelp.numerics import interpolate as ip  # noqa: E402
from kelp.numerics import vertical_uv as vu  # noqa: E402
from kelp.numerics.temporal_reconstruct import reconstruct_pchip  # noqa: E402
from kelp.readers import get_reader  # noqa: E402

from tests._bitwise import assert_bitwise  # noqa: E402
from tests.test_fesomc_w_contract import (  # noqa: E402
    SIG_S, SIG_U, _grid, _mode, _write)

M = METERS_PER_DEG_LAT
N_NODES, N_ELEM = 9, 8
DT = 1800.0
DMIN, Z0B = 0.5, 0.1
C_RK4 = (0.0, 0.5, 0.5, 1.0)
STEPS = {"euler": I.euler_step, "rk2": I.rk2_step, "rk4": I.rk4_step}
N_STAGE = {"euler": 1, "rk2": 2, "rk4": 4}

#: (layout, uv_interp) of the depth-linear fields
CASES = [("element", "leastsq"), ("element", "edge_leastsq"),
         ("element", "barycentric"), ("node", "barycentric")]
CASE_IDS = ["leastsq", "edge_leastsq", "barycentric-element",
            "barycentric-node"]

#: dia-sigma W (positive down) at the interface ``s_k``: ``W0 (0.5 + s_k)``
W0 = 1.2e-3
#: three particles in elements 1 and 6 (no coast edge), at sigma fractions
#: ``S0`` of their start columns: the third crosses the last centre (0.905)
#: in its first half step forward
LON0 = np.array([4.62, 5.13, 5.30])
LAT0 = np.array([-0.29, 0.28, 0.55])
S0 = np.array([0.30, 0.50, 0.87])
#: the same particles where no stage reaches a clip end
S0_FREE = np.array([0.30, 0.50, 0.70])

_PRE = (r"^{}_step: depth-linear horizontal velocity "
        r"\(uv_vertical='layer_linear'\) ")

needs_numba = pytest.mark.skipif(not I.HAS_NUMBA,
                                 reason="the layer kernels are counted")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _centres(sig):
    """Layer centres in s, ``1 - (sigma_k + sigma_{k+1}) / 2``."""
    sig = np.asarray(sig, dtype=np.float64)
    return 1.0 - 0.5 * (sig[:-1] + sig[1:])


@dataclasses.dataclass(frozen=True)
class Geo:
    """Bed ``d0 + gx (lon - 5) + gy lat`` (m) and surface
    ``(et + ey lat) t + eq (t / 3600)**2`` (m; t in s)."""
    d0: float = 30.0
    gx: float = 6.0
    gy: float = 8.0
    et: float = 0.0
    ey: float = 0.0
    eq: float = 0.0

    def bed(self, lon, lat):
        return self.d0 + self.gx * (lon - 5.0) + self.gy * lat

    def eta(self, lon, lat, t):
        return ((self.et + self.ey * lat) * t
                + self.eq * (t / 3600.0) ** 2 + 0.0 * lon)


@dataclasses.dataclass(frozen=True)
class Rule:
    """``u = au + bu s + eu g(t)``, ``v = av + bv s + ev g(t)`` (m/s; ``s``
    read clipped to the outer centres; ``g(t) = t``, or ``t**2 / 3600``
    with ``quad``, where a PCHIP reconstruction and a linear blend
    differ)."""
    au: float = 0.2
    bu: float = 0.4
    eu: float = 2.0e-5
    av: float = -0.1
    bv: float = 0.3
    ev: float = -1.0e-5
    quad: bool = False

    def _g(self, t):
        return t * t / 3600.0 if self.quad else t

    def cols(self, cen, t):
        return (self.au + self.bu * cen + self.eu * self._g(t),
                self.av + self.bv * cen + self.ev * self._g(t))

    def at(self, cen, s, t):
        sc = min(max(s, cen[0]), cen[-1])
        return (self.au + self.bu * sc + self.eu * self._g(t),
                self.av + self.bv * sc + self.ev * self._g(t))


def _fesomc(tmp_path, *, geo=Geo(), rule=Rule(), times=(0.0, 3600.0),
            w0=W0, sig=SIG_S, cols=None):
    """Write a FESOM-C file and read every record: ``(mesh, records)``.

    ``cols(t) -> (cu, cv)``, each ``(E, K-1)``, replaces the rule's
    columns."""
    lon, lat, _tris, _nv = _grid()
    sig = np.asarray(sig, dtype=np.float64)
    k = sig.size
    cen = _centres(sig)
    if cols is None:
        def cols(t):
            cu, cv = rule.cols(cen, t)
            return (np.broadcast_to(cu, (N_ELEM, k - 1)),
                    np.broadcast_to(cv, (N_ELEM, k - 1)))
    w_down = w0 * (0.5 + (1.0 - sig))
    w_up = -np.broadcast_to(w_down[None, :, None],
                            (len(times), k, N_NODES)).copy()
    eta = np.array([geo.eta(lon, lat, t) for t in times])
    uv = [cols(t) for t in times]
    path = _write(tmp_path / "fesomc.nc", list(times), eta, w_up, sig=sig,
                  u_elem=np.array([c[0] for c in uv]),
                  v_elem=np.array([c[1] for c in uv]),
                  depth=geo.bed(lon, lat))
    reader = get_reader("fesomc")
    mesh = reader.load_mesh(SimpleNamespace(meshpath=str(path)))
    ds = reader.open_dataset(str(path))
    try:
        recs = [reader.read_velocities(ds, i, mesh, _mode())
                for i in range(len(times))]
    finally:
        ds.close()
    assert mesh.vertical_coord == "sigma"
    assert recs[0].u.shape == (k - 1, N_ELEM)
    return mesh, recs


def _as(rec, layout, placement="layer_linear", node_cols=None):
    """The record as a field of ``placement``: element layout as read, or
    the same columns on the nodes (``node_cols = (cu, cv)``, ``(K-1, N)``,
    replaces them)."""
    if layout == "element":
        return dataclasses.replace(rec, uv_vertical=placement)
    if node_cols is None:
        node_cols = (np.repeat(rec.u[:, :1], N_NODES, axis=1),
                     np.repeat(rec.v[:, :1], N_NODES, axis=1))
    return VelocityField(u=np.array(node_cols[0]), v=np.array(node_cols[1]),
                         eta=rec.eta, w=rec.w, uv_layout="node",
                         uv_vertical=placement)


def _state(geo, t=0.0, *, s0=S0, lon=LON0, lat=LAT0, mesh=None, walk=False):
    """Particles at the sigma fractions ``s0`` of their columns at ``t``."""
    h = np.maximum(geo.bed(lon, lat) + geo.eta(lon, lat, t), DMIN)
    st = make_state(np.array(lon, dtype=np.float64),
                    np.array(lat, dtype=np.float64),
                    np.asarray(s0, dtype=np.float64) * h)
    if walk:
        nn = find_nearest_nodes(mesh, st.lon, st.lat)
        e, ok = find_containing_elements(mesh, st.lon, st.lat, nn)
        assert ok.all()
        st = st.replace(elem=np.asarray(e, dtype=np.int32))
    return st


def _payload(mesh, recs, geo, layout, mode, td):
    """``(vel, kwargs, u/v stage times, surface stage times, end time)``.

    ``steady``: the record at 0 s.  ``lerp``: a lazy blend at ``t_s``
    between the records at 0 s and 3600 s (as the tracking layer builds
    it), moving towards the record in the direction of the step.
    ``supplied``: the steady record with the start and end surfaces the
    tracking layer supplies (``sigma_eta_start`` / ``sigma_eta_end`` at
    ``t_s`` and ``t_s + td dt``)."""
    lin = [_as(r, layout) for r in recs]
    if mode == "steady":
        return lin[0], {}, (0.0,) * 4, (0.0,) * 4, 0.0
    t_s = 900.0 if td > 0 else 2700.0
    t_k = tuple(t_s + td * c * DT for c in C_RK4)
    t_end = t_s + td * DT
    if mode == "lerp":
        a, b = lin
        vel = LazyLerpField(a, b, t_s / 3600.0)
        nxt, dt_to_next = (b, 3600.0 - t_s) if td > 0 else (a, t_s)
        return (vel, dict(vel_next=nxt, dt_to_next=dt_to_next), t_k, t_k,
                t_end)
    assert mode == "supplied"
    lon, lat = mesh.nodes_lon, mesh.nodes_lat
    kw = dict(sigma_eta_start=geo.eta(lon, lat, t_s),
              sigma_eta_end=geo.eta(lon, lat, t_end))
    return lin[0], kw, (0.0,) * 4, t_k, t_end


class _Reads:
    """Call-through spy of every u/v interpolation of a step: the
    coordinate it receives (``state.z`` of its alive particles, NaN
    elsewhere) and its result."""

    def __init__(self, monkeypatch):
        self.z, self.uv = [], []
        real = I.get_interpolator

        def make(name, bc="freeslip"):
            fn = real(name, bc)

            def spy(state, *a, **k):
                self.z.append(np.where(state.alive_mask, state.z, np.nan))
                out = fn(state, *a, **k)
                self.uv.append(out)
                return out
            return spy
        monkeypatch.setattr(I, "get_interpolator", make)


# ---------------------------------------------------------------------------
# The hand rule
# ---------------------------------------------------------------------------

def _clip(x, lo, hi):
    return min(max(x, lo), hi)


def _hand(integrator, st, *, geo, rule=Rule(), sig=SIG_S, td=1.0, dt=DT,
          tu=(0.0,) * 4, te=(0.0,) * 4, t_end=0.0, surf=None, read_uv=None,
          w0=W0, w_hi=1.0, variant="stage"):
    """The sigma step by hand: ``(lon, lat, z, u1, v1, reads)``.

    ``H(x, y, t) = max(bed + eta, Dmin)`` (exact for every ``w_interp``:
    a plane; ``surf(x, y, t)`` replaces ``geo.eta``),
    ``W = td w0 (0.5 + clip(z / H, 0, w_hi))`` (``w_hi < 1``: the last
    interface a ``max_levels`` cut keeps); stage ``k`` reads u/v at the time
    ``tu[k]`` (``read_uv(k, s)`` replaces the rule) and its column at the
    surface time ``te[k]``; the end of the step places ``s_new`` in the
    column at ``t_end`` and clamps it (``[0, H - z0b_min]``).  ``reads[k]``
    is the ``s`` stage ``k`` hands its u/v interpolator: ``z_k / H_k``
    (``variant="stage"``, the rule), the entry ``z_0 / H_1`` at every stage
    (``"entry_s"``), or ``z_k / H_1``, the stage depth in the ENTRY column
    (``"entry_H"``).  The integrator places stage ``k`` at
    ``z_k = s_k H_k``, so the rule's read is the integrated ``s_k`` itself;
    the column of a stage moves it only through the rate ``W / H``."""
    cen = _centres(sig)
    eta = geo.eta if surf is None else surf
    if read_uv is None:
        def read_uv(k, s):
            return rule.at(cen, s, tu[k])

    def col(x, y, t):
        return max(geo.bed(x, y) + eta(x, y, t), DMIN)

    def wv(z, h):
        return td * w0 * (0.5 + _clip(z / h, 0.0, w_hi))

    def uv(k, s):
        u, v = read_uv(k, s)
        return td * u, td * v

    def move(u, v, y):
        return dt * u / (np.cos(y * DEG2RAD) * M), dt * v / M

    res = []
    for x0, y0, z0 in zip(st.lon, st.lat, st.z):
        h1 = col(x0, y0, te[0])

        def read(z, h):
            s = {"stage": z / h, "entry_s": z0 / h1,
                 "entry_H": z / h1}[variant]
            return _clip(s, 0.0, 1.0)

        reads = [read(z0, h1)]
        u1, v1 = uv(0, reads[0])
        w1 = wv(z0, h1)
        if integrator == "euler":
            dlon, dlat = move(u1, v1, y0)
            s_new = (z0 + w1 * dt) / h1
        elif integrator == "rk2":
            hx, hy = move(u1, v1, y0)
            xm, ym = x0 + 0.5 * hx, y0 + 0.5 * hy
            s_mid = (z0 + w1 * 0.5 * dt) / h1
            h2 = col(xm, ym, te[1])
            zm = s_mid * h2
            reads.append(read(zm, h2))
            u2, v2 = uv(1, reads[1])
            dlon, dlat = move(u2, v2, ym)
            s_new = z0 / h1 + dt * wv(zm, h2) / h2
        else:
            s0 = z0 / h1
            d = [move(u1, v1, y0)]
            f = [w1 / h1]
            for k, c in ((1, 0.5), (2, 0.5), (3, 1.0)):
                xk, yk = x0 + c * d[-1][0], y0 + c * d[-1][1]
                hk = col(xk, yk, te[k])
                zk = (s0 + c * dt * f[-1]) * hk
                reads.append(read(zk, hk))
                d.append(move(*uv(k, reads[-1]), yk))
                f.append(wv(zk, hk) / hk)
            dlon = (d[0][0] + 2.0 * d[1][0] + 2.0 * d[2][0] + d[3][0]) / 6.0
            dlat = (d[0][1] + 2.0 * d[1][1] + 2.0 * d[2][1] + d[3][1]) / 6.0
            s_new = s0 + dt * (f[0] + 2.0 * f[1] + 2.0 * f[2] + f[3]) / 6.0
        xn, yn = x0 + dlon, y0 + dlat
        h_end = col(xn, yn, t_end)
        zn = _clip(s_new * h_end, 0.0, max(h_end - Z0B, 0.0))
        res.append((xn, yn, zn, u1, v1, reads))
    lon, lat, z, u1, v1 = (np.array([r[i] for r in res]) for i in range(5))
    reads = np.array([r[5] for r in res]).T              # (stage, particle)
    return lon, lat, z, u1, v1, reads


def _assert_hand(out, st, hand, rtol=1e-9):
    """Every particle alive; displacement and depth equal the hand values to
    ``rtol`` relative; the recorded velocity is the first stage's."""
    lon, lat, z, u1, v1, _ = hand
    assert np.all(out.status == ParticleStatus.ALIVE)
    np.testing.assert_allclose(out.lon - st.lon, lon - st.lon, rtol=rtol,
                               atol=0.0)
    np.testing.assert_allclose(out.lat - st.lat, lat - st.lat, rtol=rtol,
                               atol=0.0)
    np.testing.assert_allclose(out.z, z, rtol=rtol, atol=0.0)
    np.testing.assert_allclose(out.u, u1, rtol=1e-12)
    np.testing.assert_allclose(out.v, v1, rtol=1e-12)


def _assert_far(out, st, hand, rel=1e-6):
    """The step's zonal displacement is far from another hand answer (u > 0
    everywhere): the coordinate that answer reads differs from the rule's."""
    got = out.lon - st.lon
    assert np.all(np.abs(got - (hand[0] - st.lon)) > rel * np.abs(got)), (
        np.abs(got - (hand[0] - st.lon)) / np.abs(got))


# ---------------------------------------------------------------------------
# The stage coordinate
# ---------------------------------------------------------------------------

def test_sigma_frac_for_uv_is_the_scalar_rule():
    """``_sigma_frac_for_uv`` is ``vertical_sigma._sigma_frac`` element for
    element, bit for bit: ``clip(z / H, 0, 1)``, a NaN kept NaN (the
    depth-linear kernels read it as ``+inf``: the bottom centre), a signed
    zero kept."""
    from kelp.numerics.vertical_sigma import _sigma_frac
    z = np.array([-3.0, -0.0, 0.0, 0.1, 7.5, 30.0, 30.000001, 1e300,
                  np.inf, -np.inf, np.nan, 12.0])
    h = np.array([30.0, 30.0, 30.0, 0.5, 30.0, 30.0, 30.0, 30.0, 30.0,
                  30.0, 30.0, np.nan])
    got = I._sigma_frac_for_uv(z, h)
    want = np.array([_sigma_frac(a, b) for a, b in zip(z, h)])
    assert got.dtype == np.float64 and got.shape == z.shape
    assert_bitwise(got, want, "s")
    assert got[0] == 0.0 and got[6] == 1.0 and np.isnan(got[-2:]).all()


# ---------------------------------------------------------------------------
# A field linear in s: exact, stage by stage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("walk", [False, True], ids=["scan", "walk"])
@pytest.mark.parametrize("mode", ["steady", "lerp"])
@pytest.mark.parametrize("td", [1.0, -1.0])
@pytest.mark.parametrize("layout,uv_interp", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("integrator", list(STEPS))
def test_linear_in_s_exact(tmp_path, integrator, layout, uv_interp, td,
                           mode, walk):
    """``u = a + b*s (+ e*t)`` with W crossing the centres: the positions
    and depths equal the hand stage combination, each stage reading
    ``s = clip(z_k / H_k)`` at its own position and time, to 1e-9 relative
    (steady, and a lazy blend between two records with u varying in time;
    the elements the walk tracks reach ``edge_leastsq`` and element
    ``barycentric``).  The answer that reads the entry ``s`` at every
    stage is far from the step."""
    geo = Geo()
    mesh, recs = _fesomc(tmp_path, geo=geo)
    vel, kw, tu, te, t_end = _payload(mesh, recs, geo, layout, mode, td)
    st = _state(geo, te[0], mesh=mesh, walk=walk)
    out = STEPS[integrator](st, vel, mesh, DT, td, uv_interp=uv_interp,
                            w_interp="barycentric", **kw)
    hand = _hand(integrator, st, geo=geo, td=td, tu=tu, te=te, t_end=t_end)
    _assert_hand(out, st, hand)
    if integrator != "euler":
        _assert_far(out, st, _hand(integrator, st, geo=geo, td=td, tu=tu,
                                   te=te, t_end=t_end, variant="entry_s"))


#: moving surface: ``eta = (4e-4 + 2e-4 lat) t`` (1.4 m per hour at the
#: equator, tilted)
GEO_MOVING = Geo(et=4.0e-4, ey=2.0e-4)


@pytest.mark.parametrize("mode", ["lerp", "supplied"])
@pytest.mark.parametrize("td", [1.0, -1.0])
@pytest.mark.parametrize("layout,uv_interp", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("integrator", list(STEPS))
def test_moving_surface_linear_in_s_exact(tmp_path, integrator, layout,
                                          uv_interp, td, mode):
    """``eta`` linear in time (and tilted): each stage's ``H_p`` is the
    column at its own time — the blend of the two records, or the blend of
    the surfaces the tracking layer supplies on a steady step — so the
    positions equal the hand values to 1e-9 relative; the answer that reads
    each stage's depth in the ENTRY column (the surface of the step's
    start) is far from it (RK2, RK4)."""
    geo = GEO_MOVING
    mesh, recs = _fesomc(tmp_path, geo=geo)
    vel, kw, tu, te, t_end = _payload(mesh, recs, geo, layout, mode, td)
    st = _state(geo, te[0], s0=S0_FREE)
    out = STEPS[integrator](st, vel, mesh, DT, td, uv_interp=uv_interp,
                            w_interp="barycentric", **kw)
    args = dict(geo=geo, td=td, tu=tu, te=te, t_end=t_end)
    _assert_hand(out, st, _hand(integrator, st, **args))
    if integrator != "euler":
        _assert_far(out, st, _hand(integrator, st, variant="entry_H",
                                   **args))


#: a steep bed (4 .. 56 m) and a flow up the slope
GEO_STEEP = Geo(d0=30.0, gx=10.0, gy=16.0)
RULE_UPSLOPE = Rule(au=0.1, bu=0.3, av=0.5, bv=0.4)


@pytest.mark.parametrize("layout,uv_interp", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("integrator", list(STEPS))
def test_sloping_bed_linear_in_s_exact(tmp_path, integrator, layout,
                                       uv_interp):
    """A steep bed crossed by the flow (the column changes by 0.4 .. 0.9 %
    within a step; ``w_interp: leastsq``, the step default, for W and
    ``H_p``):
    the positions equal the hand values to 1e-9 relative; the answer that
    reads each stage's depth in the ENTRY column is far from it (RK2,
    RK4)."""
    geo, rule = GEO_STEEP, RULE_UPSLOPE
    mesh, recs = _fesomc(tmp_path, geo=geo, rule=rule)
    vel = _as(recs[0], layout)
    st = _state(geo, s0=S0_FREE)
    out = STEPS[integrator](st, vel, mesh, DT, 1.0, uv_interp=uv_interp,
                            w_interp="leastsq")
    _assert_hand(out, st, _hand(integrator, st, geo=geo, rule=rule))
    if integrator != "euler":
        _assert_far(out, st, _hand(integrator, st, geo=geo, rule=rule,
                                   variant="entry_H"))


@pytest.mark.parametrize("td", [1.0, -1.0])
@pytest.mark.parametrize("layout,uv_interp", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("integrator", ["rk2", "rk4"])
def test_stage_s_not_entry_s(monkeypatch, tmp_path, integrator, layout,
                             uv_interp, td):
    """A moving surface over a sloping bed with W: the entry ``s``, the
    stage depth in the entry column and the stage's own ``s`` all differ.
    Every u/v interpolation of the step receives the stage's own ``s`` (to
    1e-12 relative, the hand ``reads``); the positions equal that answer to
    1e-9 relative and are far from both other answers."""
    geo = GEO_MOVING
    mesh, recs = _fesomc(tmp_path, geo=geo)
    vel, kw, tu, te, t_end = _payload(mesh, recs, geo, layout, "lerp", td)
    st = _state(geo, te[0], s0=S0_FREE)
    reads = _Reads(monkeypatch)
    out = STEPS[integrator](st, vel, mesh, DT, td, uv_interp=uv_interp,
                            w_interp="barycentric", **kw)
    args = dict(geo=geo, td=td, tu=tu, te=te, t_end=t_end)
    hand = _hand(integrator, st, **args)
    assert len(reads.z) == N_STAGE[integrator]
    np.testing.assert_allclose(np.array(reads.z), hand[5], rtol=1e-12)
    assert np.all((hand[5] > 0.01) & (hand[5] < 0.9))
    _assert_hand(out, st, hand)
    for variant in ("entry_s", "entry_H"):
        _assert_far(out, st, _hand(integrator, st, variant=variant, **args))


@pytest.mark.parametrize("mode", ["lerp", "supplied", "pchip"])
@pytest.mark.parametrize("integrator", list(STEPS))
def test_steps_do_not_scan_records(monkeypatch, tmp_path, integrator, mode):
    """The steps on a sigma mesh trust their fields: neither the entry nor
    a stage field (a blend of the records, the field seen with another
    surface, a PCHIP reconstruction at the stage time) runs a record scan
    (a run checks every record once, where it stores it)."""
    geo = GEO_MOVING
    if mode == "pchip":
        mesh, recs = _fesomc(tmp_path, geo=geo, times=RECORDS)
        snaps = [_as(r, "element") for r in recs]
        vel, te = snaps[1], (RECORDS[1],)
        kw = dict(vel_next=snaps[2], dt_to_next=3600.0,
                  recon=(snaps, list(RECORDS), RECORDS[1], 1))
    else:
        mesh, recs = _fesomc(tmp_path, geo=geo)
        vel, kw, _tu, te, _t_end = _payload(mesh, recs, geo, "element",
                                            mode, 1.0)
    def no_scan(*a, **k):
        raise AssertionError("a step scanned a record")
    monkeypatch.setattr(vu, "_check_components", no_scan)
    st = _state(geo, te[0])
    out = STEPS[integrator](st, vel, mesh, DT, 1.0, uv_interp="leastsq",
                            w_interp="barycentric", **kw)
    assert np.all(out.status == ParticleStatus.ALIVE)


# ---------------------------------------------------------------------------
# Thin water: H_p at Dmin
# ---------------------------------------------------------------------------

#: every node 0.1 .. 0.4 m deep: H_p = Dmin = 0.5 m everywhere
GEO_THIN = Geo(d0=0.25, gx=0.05, gy=0.1)
W0_THIN = 2.0e-4


@pytest.mark.parametrize("td", [1.0, -1.0])
@pytest.mark.parametrize("layout,uv_interp", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("integrator", list(STEPS))
def test_dmin_column(monkeypatch, tmp_path, integrator, layout, uv_interp,
                     td):
    """Columns thinner than ``Dmin``: ``H_p = Dmin``, W drives the RK
    stages out of the column (below the bed forward, above the surface
    backward).  Every u/v interpolation receives ``s`` in ``[0, 1]`` — the
    stage beyond the bed reads ``s = 1``, the one above the surface
    ``s = 0`` (the outer centres' values); the positions equal the hand
    values to 1e-9 relative, and the clamp leaves the particle inside the
    column (``0 <= z <= Dmin - z0b_min``, ``s`` in ``[0, 1]``)."""
    geo = GEO_THIN
    assert np.all(geo.bed(*_grid()[:2]) < DMIN)
    mesh, recs = _fesomc(tmp_path, geo=geo, w0=W0_THIN)
    vel = _as(recs[0], layout)
    st = _state(geo, s0=np.array([0.2, 0.5, 0.76]))
    np.testing.assert_allclose(st.z, [0.1, 0.25, 0.38], rtol=1e-15)
    reads = _Reads(monkeypatch)
    out = STEPS[integrator](st, vel, mesh, DT, td, uv_interp=uv_interp,
                            w_interp="barycentric")
    hand = _hand(integrator, st, geo=geo, td=td, w0=W0_THIN)
    _assert_hand(out, st, hand)
    got = np.array(reads.z)
    np.testing.assert_allclose(got, hand[5], rtol=1e-12)
    assert np.all((got >= 0.0) & (got <= 1.0))
    if integrator != "euler":
        assert np.any(got[1:] == (1.0 if td > 0 else 0.0))
    assert np.all((out.z >= 0.0) & (out.z <= DMIN - Z0B))
    assert np.all((out.z / DMIN >= 0.0) & (out.z / DMIN <= 1.0))


# ---------------------------------------------------------------------------
# PCHIP in time
# ---------------------------------------------------------------------------

#: PCHIP records (s) and the step's start per direction (central bracket)
RECORDS = (0.0, 3600.0, 7200.0, 10800.0)
PCHIP_START = {1.0: 3600.0, -1.0: 7200.0}
#: a surface quadratic in time (0.5 m at 1 h, 4.5 m at 3 h)
GEO_QUAD = Geo(eq=0.5)


@pytest.mark.parametrize("td", [1.0, -1.0])
@pytest.mark.parametrize("layout,uv_interp", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("integrator", ["rk2", "rk4"])
def test_pchip_in_time(tmp_path, integrator, layout, uv_interp, td):
    """``temporal_reconstruction: pchip`` on records quadratic in time (u/v
    and the surface), where the 4-record PCHIP differs from the linear
    blend.  The expected stage value is computed independently:
    ``reconstruct_pchip`` of the same records at the stage time — u/v by
    ``np.interp`` of that column at the clipped stage ``s``, the stage
    column from that surface; positions to 1e-9 relative.  The answer with
    the linear blend of the central records is far from the step."""
    geo, rule = GEO_QUAD, Rule(quad=True)
    mesh, recs = _fesomc(tmp_path, geo=geo, rule=rule, times=RECORDS)
    snaps = [_as(r, layout) for r in recs]
    cen = _centres(SIG_S)
    t0 = PCHIP_START[td]
    t_k = tuple(t0 + td * c * DT for c in C_RK4)
    t_end = t0 + td * DT
    tab_p, tab_l = {}, {}
    for t in set(t_k) | {t_end}:
        rf = reconstruct_pchip(snaps, RECORDS, t)
        assert np.all(rf.u == rf.u[:, :1]) and np.all(rf.v == rf.v[:, :1])
        assert np.all(rf.eta == rf.eta[0])
        tab_p[t] = (rf.u[:, 0].copy(), rf.v[:, 0].copy(), float(rf.eta[0]))
        a = (t - RECORDS[1]) / (RECORDS[2] - RECORDS[1])
        tab_l[t] = tuple((1.0 - a) * x1 + a * x2 for x1, x2 in (
            (snaps[1].u[:, 0], snaps[2].u[:, 0]),
            (snaps[1].v[:, 0], snaps[2].v[:, 0]),
            (float(recs[1].eta[0]), float(recs[2].eta[0]))))
    mid = t0 + td * 0.5 * DT
    assert np.all(np.abs(tab_p[mid][0] - tab_l[mid][0]) > 1e-3)
    assert abs(tab_p[mid][2] - tab_l[mid][2]) > 1e-2

    def by(tab):
        def read_uv(k, s):
            cu, cv, _ = tab[t_k[k]]
            sc = min(max(s, cen[0]), cen[-1])
            return float(np.interp(sc, cen, cu)), float(np.interp(sc, cen, cv))
        return dict(read_uv=read_uv, surf=lambda x, y, t: tab[t][2])

    vel, nxt = (snaps[1], snaps[2]) if td > 0 else (snaps[2], snaps[1])
    recon = (snaps, list(RECORDS), t0, int(td))
    st = _state(geo, t0)
    out = STEPS[integrator](st, vel, mesh, DT, td, uv_interp=uv_interp,
                            w_interp="barycentric", vel_next=nxt,
                            dt_to_next=3600.0, recon=recon)
    args = dict(geo=geo, td=td, te=t_k, t_end=t_end)
    _assert_hand(out, st, _hand(integrator, st, **by(tab_p), **args))
    _assert_far(out, st, _hand(integrator, st, **by(tab_l), **args))


# ---------------------------------------------------------------------------
# A depth-constant field: the layer answer
# ---------------------------------------------------------------------------

#: ulp of max|c| allowed between two velocities of the same position: the
#: plane fits 8, element barycentric 4, node barycentric 2
VEL_ULP = {"leastsq": 8, "edge_leastsq": 8, "barycentric-element": 4,
           "barycentric-node": 2}
#: the binade of every constant of the depth-constant field (m/s)
C_BINADE = (2.0 ** -5, 2.0 ** -4)
#: four particles at sigma fractions ``S_C`` (uniform sigma table, centres
#: 0.05 .. 0.95, slow W): the first and the last stay beyond the outer
#: centres at every stage (every member reads a stored value), the middle
#: two between centres
LON_C = np.array([4.62, 5.13, 5.30, 5.37])
LAT_C = np.array([-0.29, 0.28, 0.55, -0.16])
S_C = np.array([0.02, 0.33, 0.61, 0.97])
EXACT_C = np.array([True, False, False, True])
W0_C = 1.0e-4


def _const_cols(n, t):
    """Per-column constants, varying from column to column and in time,
    within ``C_BINADE`` (0.036 .. 0.058 m/s)."""
    j = np.arange(n, dtype=np.float64)
    r = t / 3600.0
    return (0.047 + 0.01 * np.sin(1.7 * j + 0.3) + 0.001 * r * np.cos(0.9 * j),
            0.047 + 0.01 * np.cos(1.3 * j) - 0.001 * r * np.sin(0.7 * j))


@pytest.mark.parametrize("mode", ["steady", "lerp"])
@pytest.mark.parametrize("td", [1.0, -1.0])
@pytest.mark.parametrize("layout,uv_interp", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("integrator", list(STEPS))
def test_depth_constant_same_as_layer(monkeypatch, tmp_path, integrator,
                                      layout, uv_interp, td, mode):
    """u, v independent of ``s`` (per-column constants, varying from column
    to column and in time) with W: the depth-linear step equals the layer
    step — bit for bit (positions, depth, recorded and every stage's
    velocity) for the particles whose members all read stored values at
    every stage (beyond the outer centres), elsewhere every position and
    depth within 4 ulp and every stage velocity within the interpolator's
    ulp bound of ``max|c|`` (the constants lie in one binade, so that ulp is
    the field's).

    Measured (2026-10-20): every position and depth of every case bit for
    bit the layer step's; per integrator and interpolator (both directions,
    both modes) up to 6 of the 32 .. 128 stage velocities differ, by at most
    2 ulp of ``max|c|`` (``leastsq``, node ``barycentric``), 1
    (``edge_leastsq``, element ``barycentric``)."""
    def cols(t):
        cu, cv = _const_cols(N_ELEM, t)
        return (np.repeat(cu[:, None], SIG_U.size - 1, axis=1),
                np.repeat(cv[:, None], SIG_U.size - 1, axis=1))
    geo = Geo()
    mesh, recs = _fesomc(tmp_path, geo=geo, w0=W0_C, sig=SIG_U, cols=cols)
    runs = {}
    for placement in ("layer", "layer_linear"):
        fields = []
        for r, t in zip(recs, (0.0, 3600.0)):
            nc = None
            if layout == "node":
                cu, cv = _const_cols(N_NODES, t)
                nc = (np.repeat(cu[None, :], SIG_U.size - 1, axis=0),
                      np.repeat(cv[None, :], SIG_U.size - 1, axis=0))
            fields.append(_as(r, layout, placement, node_cols=nc))
        if mode == "steady":
            vel, kw, t0 = fields[0], {}, 0.0
        else:
            t0 = 900.0 if td > 0 else 2700.0
            vel = LazyLerpField(fields[0], fields[1], t0 / 3600.0)
            kw = (dict(vel_next=fields[1], dt_to_next=3600.0 - t0) if td > 0
                  else dict(vel_next=fields[0], dt_to_next=t0))
        st = _state(geo, t0, s0=S_C, lon=LON_C, lat=LAT_C)
        reads = _Reads(monkeypatch)
        out = STEPS[integrator](st, vel, mesh, DT, td, uv_interp=uv_interp,
                                w_interp="barycentric", **kw)
        monkeypatch.undo()
        assert np.all(out.status == ParticleStatus.ALIVE)
        runs[placement] = (out, reads)
    (lin, lin_r), (lay, lay_r) = runs["layer_linear"], runs["layer"]
    lin_uv, lay_uv = lin_r.uv, lay_r.uv
    # the exact particles stay beyond the outer centres at every stage, the
    # others between them
    cen = _centres(SIG_U)
    s = np.array(lin_r.z)
    assert np.all((s[:, EXACT_C] < cen[0]) | (s[:, EXACT_C] > cen[-1]))
    assert np.all((s[:, ~EXACT_C] > cen[0]) & (s[:, ~EXACT_C] < cen[-1]))
    for k in ("lon", "lat", "z", "u", "v"):
        assert_bitwise(getattr(lin, k)[EXACT_C], getattr(lay, k)[EXACT_C], k)
    for k in ("lon", "lat", "z"):
        a, b = getattr(lin, k), getattr(lay, k)
        assert np.all(np.abs(a - b) <= 4 * np.spacing(np.abs(b))), k
    ulp = np.spacing(C_BINADE[0])
    tol = VEL_ULP[uv_interp if uv_interp != "barycentric"
                  else f"barycentric-{layout}"] * ulp
    assert len(lin_uv) == len(lay_uv) == N_STAGE[integrator]
    for (ul, vl, okl), (uc, vc, okc) in zip(lin_uv, lay_uv):
        assert okl.all() and okc.all()
        for a, b, nm in ((ul, uc, "U"), (vl, vc, "V")):
            assert_bitwise(a[EXACT_C], b[EXACT_C], nm)
            assert np.all(np.abs(a - b) <= tol), (nm, np.abs(a - b) / ulp)


# ---------------------------------------------------------------------------
# Layer runs unchanged (dispatch)
# ---------------------------------------------------------------------------

#: depth-linear entry points that a layer run must never reach
_DL_FUNCS = ((I, "_sigma_frac_for_uv"), (I, "_fused_euler_step_3d_dl"),
             (ip, "_interp_leastsq_dl"), (ip, "_interp_edge_leastsq_dl"),
             (ip, "_interp_barycentric_dl"),
             (ip, "_loop_interp_leastsq_3d_dl"),
             (ip, "_loop_interp_edge_lsq_3d_dl"),
             (ip, "_loop_interp_bary_elem_dl"), (ip, "_loop_interp_bary_dl"),
             (I, "depth_points"), (ip, "depth_points"),
             (vu, "depth_points"), (vu, "check_depth_linear_record"),
             (vu, "_check_components"))

#: the layer kernel each interpolator runs (unfused)
_LAYER_KERNEL = {"leastsq": (ip, "_loop_interp_leastsq_3d"),
                 "edge_leastsq": (ip, "_loop_interp_edge_lsq_3d"),
                 "barycentric": (ip, "_loop_interp_bary_3d")}


class _Count:
    """Call-through spy of a module attribute."""

    def __init__(self, monkeypatch, module, name):
        self.n = 0
        real = getattr(module, name)

        def spy(*a, **k):
            self.n += 1
            return real(*a, **k)
        monkeypatch.setattr(module, name, spy)


def _forbid(monkeypatch):
    def make(name):
        def spy(*a, **k):
            raise AssertionError(f"a layer run reached {name}")
        return spy
    for mod, name in _DL_FUNCS:
        monkeypatch.setattr(mod, name, make(name))


@needs_numba
@pytest.mark.parametrize("mode", ["steady", "lerp", "pchip"])
@pytest.mark.parametrize("layout,uv_interp", CASES, ids=CASE_IDS)
@pytest.mark.parametrize("integrator", list(STEPS))
def test_layer_runs_unchanged(monkeypatch, tmp_path, integrator, layout,
                              uv_interp, mode):
    """``"layer"`` fields on a sigma mesh run the layer kernels — one layer
    lookup (``_compute_layers``) and one layer interpolation per stage, no
    fused kernel — and never the sigma fraction of the depth-linear path, a
    depth-linear function, the depth-point table or the record check (each
    replaced by a spy that fails)."""
    if mode == "pchip" and integrator == "euler":
        pytest.skip("Euler ignores the PCHIP bundle")
    geo = GEO_MOVING
    mesh, recs = _fesomc(tmp_path, geo=geo, times=RECORDS)
    snaps = [_as(r, layout, "layer") for r in recs]
    kw = {}
    vel = snaps[1]
    if mode == "lerp":
        vel = LazyLerpField(snaps[1], snaps[2], 0.25)
        kw = dict(vel_next=snaps[2], dt_to_next=2700.0)
    elif mode == "pchip":
        kw = dict(vel_next=snaps[2], dt_to_next=3600.0,
                  recon=(snaps, list(RECORDS), 3600.0, 1))
    st = _state(geo, 3600.0)
    _forbid(monkeypatch)
    layers = _Count(monkeypatch, I, "_compute_layers")
    fused = _Count(monkeypatch, I, "_fused_euler_step_3d")
    loops = {k: _Count(monkeypatch, *v) for k, v in _LAYER_KERNEL.items()}
    out = STEPS[integrator](st, vel, mesh, DT, 1.0, uv_interp=uv_interp,
                            w_interp="barycentric", **kw)
    assert np.all(out.status == ParticleStatus.ALIVE)
    n = N_STAGE[integrator]
    assert fused.n == 0 and layers.n == n and loops[uv_interp].n == n
    assert sum(c.n for k, c in loops.items() if k != uv_interp) == 0


# ---------------------------------------------------------------------------
# Still refused on sigma
# ---------------------------------------------------------------------------

def test_rk45_and_2d_still_refused(tmp_path):
    """On a sigma mesh the entry still refuses rk45 (every layout and
    interpolator) and a field without ``w``; ``"level"`` data (one row per
    interface) are refused by the sigma table, which serves layer-centred
    data only."""
    mesh, recs = _fesomc(tmp_path)
    st = _state(Geo())
    for layout, interp in CASES:
        vel = _as(recs[0], layout)
        with pytest.raises(ValueError, match=_PRE.format("rk45")
                           + r"is not supported by the rk45 integrator"):
            I.rk45_step(st, vel, mesh, DT, 1.0, uv_interp=interp)
        flat = VelocityField(u=vel.u, v=vel.v, eta=vel.eta, w=None,
                             uv_layout=layout, uv_vertical="layer_linear")
        for name, step in STEPS.items():
            with pytest.raises(ValueError, match=_PRE.format(name)
                               + r"needs a 3-D field"):
                step(st, flat, mesh, DT, 1.0, uv_interp=interp,
                     w_interp="barycentric")
    k = SIG_S.size
    lev = VelocityField(u=np.full((k, N_NODES), 0.2),
                        v=np.full((k, N_NODES), -0.1), eta=recs[0].eta,
                        w=recs[0].w, uv_layout="node", uv_vertical="level")
    for name, step in STEPS.items():
        with pytest.raises(ValueError, match=(
                r"^depth-linear u/v on a sigma mesh \(vertical_coord "
                r"'sigma'\) is 'layer_linear' only .*got 'level'$")):
            step(st, lev, mesh, DT, 1.0, uv_interp="barycentric",
                 w_interp="barycentric")
