"""mode.uv_vertical: linear on FESOM-C sigma meshes through the runs.

The option reaches FESOM-C ``u``/``v`` (element layout: ``uv_interp``
``leastsq``, ``edge_leastsq`` or ``barycentric``; node layout:
``barycentric``) through ``run_tracking`` and the direct entry points: the
velocity cache marks each record ``"layer_linear"`` (prefetch windows and
the shared snapshot cache included), the output says
``uv_vertical = "linear"``, and every stage reads u/v at its own sigma
fraction ``s = clip(z / H_p, 0, 1)`` — with temporal interpolation (linear
or PCHIP in time) from the records blended at its own time.  A sigma mesh
needs no element bottoms (every layer of every column is wet), so element
data run on it; restarts resume bit for bit and refuse the other rule, a
``max_levels`` cut keeps the layers it keeps (below the last kept centre
the value of that centre, as the layer rule reads it), and the diffusion
compositions, the Az kick and multi-realization runs keep the option.

Fixture (all synthetic): the FESOM-C file of
``tests/test_integrators_depth_linear_sigma`` (its 3x3-node grid of 8
triangles, element ``u``/``v`` ``(time, nele, sigmam1)``, stretched sigma)
written for ``N_REC`` records ``DT`` apart, plus the same columns on the
nodes (``unode``/``vnode``, ``(time, node, sigmam1)``): a bed linear in lon
and lat (columns 16 .. 44 m, unequal), a surface rising in time and tilted
(``GEO``), a dia-sigma W ``W0 (0.5 + s)`` (positive down, nonzero
everywhere, so ``s`` changes within every step) and ``u = a + b s + e t``
in every column (``RULE``).  The particles of the exact tests stay in the
two elements without a coast edge, where every interpolator reproduces the
columns exactly, so every position is the hand stage combination of
``_hand`` (each stage's ``s`` from its own position, time and column),
step after step: discrete records ``DT`` apart and ``dt = DT``, so a step
starts on a record and its RK stages blend the next one.  The temporal
interpolation test writes the surface and u/v quadratic in time instead
(``GEO_QUAD``, ``RULE_QUAD``), where the linear blend and the PCHIP of the
records differ, and steps ``DT / 2``.
"""
from __future__ import annotations

import contextlib
import dataclasses
import io
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

xr = pytest.importorskip("xarray")
netCDF4 = pytest.importorskip("netCDF4")

from kelp import tracking  # noqa: E402
from kelp.core.config import (AppConfig, DataConfig,  # noqa: E402
                              DiffusionConfig, MeshConfig, ModeConfig,
                              NumericsConfig, TimeConfig)
from kelp.core.geometry import (DEG2RAD, METERS_PER_DEG_LAT,  # noqa: E402
                                wrap_lon)
from kelp.core.mesh import (find_containing_elements,  # noqa: E402
                            find_nearest_nodes)
from kelp.core.particles import ParticleStatus, make_state  # noqa: E402
from kelp.numerics.vertical_uv import is_checked  # noqa: E402
from kelp.readers import extract_time_info, get_reader  # noqa: E402
from kelp.writers import (TrajectoryWriter, load_restart,  # noqa: E402
                          load_trajectories)
from tests._bitwise import assert_bitwise  # noqa: E402
from tests.test_fesomc_w_contract import SIG_S  # noqa: E402
from tests.test_integrators_depth_linear_sigma import (  # noqa: E402
    C_RK4, CASE_IDS, CASES, DMIN, N_NODES, W0, Geo, Rule, _centres, _fesomc,
    _hand)
from tests.test_uv_vertical_elements_run import (  # noqa: E402
    _FIELDS, _assert_metres, _assert_same_run, _global_attrs)

# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------

DT = 1800.0
N_REC = 7
N_STEPS = 4
#: a surface rising in time and tilted (1.4 m per hour at the equator)
GEO = Geo(et=4.0e-4, ey=2.0e-4)
RULE = Rule()
INTEGRATORS = ["euler", "rk2", "rk4"]

#: in elements 1 and 6 (no coast edge) for the whole run, at the sigma
#: fractions ``S_SEED`` of their columns at the start
LON0 = (4.62, 5.13, 5.30)
LAT0 = (-0.29, 0.28, 0.55)
S_SEED = (0.12, 0.30, 0.45)


def _file(work, *, geo=GEO, rule=RULE, sig=SIG_S, w0=W0, n_rec=N_REC):
    """The FESOM-C file under ``work``: element ``u``/``v`` and the same
    columns on the nodes (``unode``/``vnode``)."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    times = tuple(r * DT for r in range(n_rec))
    with contextlib.redirect_stdout(io.StringIO()):
        _fesomc(work, geo=geo, rule=rule, times=times, w0=w0, sig=sig)
    path = work / "fesomc.nc"
    cen = _centres(sig)
    with netCDF4.Dataset(str(path), "a") as nc:
        for name, comp in (("unode", 0), ("vnode", 1)):
            var = nc.createVariable(name, "f8", ("time", "node", "sigmam1"))
            var[:] = np.array([np.broadcast_to(rule.cols(cen, t)[comp],
                                               (N_NODES, sig.size - 1))
                               for t in times])
    return path


def _seeds(lon=LON0, lat=LAT0, s=S_SEED, *, geo=GEO, t=0.0):
    """``(lon, lat, z)`` at the sigma fractions ``s`` of the columns at
    ``t``."""
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    h = np.maximum(geo.bed(lon, lat) + geo.eta(lon, lat, t), DMIN)
    return list(zip(lon, lat, np.asarray(s, float) * h))


def _mode(layout="element", **kw):
    base = dict(Fdrift2D=False, Fdrift3D=True, uv_vertical="linear")
    if layout == "node":
        base.update(var_u="unode", var_v="vnode")
    base.update(kw)
    return ModeConfig(**base)


def _cfg(path, *, layout="element", uv_interp="leastsq", integrator="rk2",
         seeds=None, n_steps=N_STEPS, start=0, out="traj.nc", num_runs=1,
         dt=DT, **kw):
    """A run_tracking config on the file ``path``; ``kw`` goes to
    NumericsConfig when it names one of its fields, else to the mode."""
    work = Path(path).parent
    seeds = _seeds() if seeds is None else seeds
    seeds_path = work / f"seeds_{out}.dat"
    seeds_path.write_text("".join(f"{lo:.17g} {la:.17g} {z:.17g}\n"
                                  for lo, la, z in seeds))
    num_names = {f.name for f in dataclasses.fields(NumericsConfig)}
    num_kw = {k: v for k, v in kw.items() if k in num_names}
    mode_kw = {k: v for k, v in kw.items() if k not in num_names}
    num_kw.setdefault("w_interp", "barycentric")
    return AppConfig(
        mode=_mode(layout, **mode_kw),
        numerics=NumericsConfig(dt=dt, integrator=integrator,
                                uv_interp=uv_interp, **num_kw),
        mesh=MeshConfig(type="fesomc", meshpath=str(path)),
        data=DataConfig(ncinput=str(path), inicloud=str(seeds_path),
                        output_path=str(work / out),
                        time_averaging="instant"),
        time=TimeConfig(start_time_idx=start, n_steps=n_steps,
                        num_runs=num_runs),
    )


def _run(cfg):
    with contextlib.redirect_stdout(io.StringIO()):
        return tracking.run_tracking(cfg)


def _hand_run(integrator, seeds, n_steps, *, start=0, sig=SIG_S, w_hi=1.0):
    """Positions and depths after each step, ``(particle, step)``: the
    hand step of ``tests/test_integrators_depth_linear_sigma`` applied step
    after step, step ``i`` from record ``start + i`` (its stages at
    ``t_i + c DT``, its end in the column at ``t_i + DT``); ``sig`` the
    levels in use, ``w_hi`` the sigma fraction below which W stays at its
    value there (the last kept interface of a cut)."""
    lon, lat, z = (np.array(c, dtype=np.float64) for c in zip(*seeds))
    rows = []
    for i in range(n_steps):
        t_s = (start + i) * DT
        t_k = tuple(t_s + c * DT for c in C_RK4)
        lon, lat, z, *_ = _hand(
            integrator, SimpleNamespace(lon=lon, lat=lat, z=z), geo=GEO,
            rule=RULE, sig=sig, dt=DT, tu=t_k, te=t_k, t_end=t_s + DT,
            w_hi=w_hi)
        lon = np.asarray(wrap_lon(lon), dtype=np.float64)
        rows.append((lon, lat, z))
    return tuple(np.array([r[j] for r in rows]).T for j in range(3))


def _gap(a, b):
    """Horizontal distance (m) between two runs' positions."""
    return np.hypot((a["lon"] - b["lon"]) * METERS_PER_DEG_LAT
                    * np.cos(b["lat"] * DEG2RAD),
                    (a["lat"] - b["lat"]) * METERS_PER_DEG_LAT)


def _state(mesh, seeds):
    lon, lat, z = (np.array(c, dtype=np.float64) for c in zip(*seeds))
    st = make_state(lon, lat, z)
    return st.replace(nearest_node=find_nearest_nodes(mesh, st.lon, st.lat))


def _open(path, layout="element"):
    reader = get_reader("fesomc")
    with contextlib.redirect_stdout(io.StringIO()):
        mesh = reader.load_mesh(MeshConfig(type="fesomc", meshpath=str(path)))
    ds = reader.open_dataset(str(path), mode_cfg=_mode(layout))
    return reader, mesh, ds


class _CountingReader:
    """The FESOM-C reader with its reads counted."""

    def __init__(self):
        self.base = get_reader("fesomc")
        self.reads = 0

    def probe_uv_layout(self, ds, mode_cfg, mesh):
        return self.base.probe_uv_layout(ds, mode_cfg, mesh)

    def read_velocities(self, *a, **k):
        self.reads += 1
        return self.base.read_velocities(*a, **k)


# ---------------------------------------------------------------------------
# The option on FESOM-C data: the hand rule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("integrator", INTEGRATORS)
@pytest.mark.parametrize("layout,uv_interp", CASES, ids=CASE_IDS)
def test_run_tracking_linear_fesomc(tmp_path, layout, uv_interp, integrator):
    """Element and node layout x euler/rk2/rk4 through ``run_tracking``:
    every position and depth equals the hand stage combination step after
    step to 1e-9 m (each stage's ``s`` from its own position, time and
    column); the output says ``uv_vertical = "linear"``.  The layer rule
    (option off) moves every particle elsewhere and writes no attribute."""
    path = _file(tmp_path)
    cfg = _cfg(path, layout=layout, uv_interp=uv_interp,
               integrator=integrator)
    _run(cfg)
    t = load_trajectories(cfg.data.output_path)
    assert (t["status"] == ParticleStatus.ALIVE).all()
    h_lon, h_lat, h_z = _hand_run(integrator, _seeds(), N_STEPS)
    _assert_metres(t["lon"], t["lat"], h_lon, h_lat)
    assert np.all(np.abs(t["z"] - h_z) <= 1e-9), np.abs(t["z"] - h_z).max()
    # W moves every particle down through the layers (s from 0.12 .. 0.45
    # to 0.35 .. 0.69 in four steps), none reaches an outer centre
    assert np.all(np.diff(t["z"], axis=1) > 0.5)
    assert _global_attrs(cfg.data.output_path)["uv_vertical"] == "linear"
    off = _cfg(path, layout=layout, uv_interp=uv_interp,
               integrator=integrator, uv_vertical="layer", out="off.nc")
    _run(off)
    t_off = load_trajectories(off.data.output_path)
    # the two rules part and meet again as the particles sink through the
    # layers: 17 .. 72 m apart at their farthest
    gap = np.max(_gap(t_off, t), axis=1)
    assert np.all(gap > 10.0), gap
    assert "uv_vertical" not in _global_attrs(off.data.output_path)


# ---------------------------------------------------------------------------
# Temporal interpolation
# ---------------------------------------------------------------------------

#: a surface and u/v quadratic in time, where the PCHIP of the records and
#: their linear blend differ (the surface 0.5 m at record 2, 3.1 m at
#: record 5); a surface level in space has the same PCHIP at every node, so
#: the hand column is exact
GEO_QUAD = Geo(eq=0.5)
RULE_QUAD = Rule(quad=True)
#: from record 2 to record 5 (of 0 .. 6) in steps of DT / 2: every step's
#: interval has its four PCHIP records, so no step falls back to the linear
#: blend and nothing warns
TI_START, TI_STEPS = 2, 6


def _in_time(mode, rec, t):
    """Records ``rec`` (``DT`` apart from 0; any trailing shape) at ``t``
    as the run blends them: linearly between the two around ``t``
    (``"lerp"``), or by their PCHIP (``"pchip"``: SciPy's over every
    record, which between records 1 and ``N_REC - 2`` has interior slopes
    only, so it is the 4-record stencil of the run)."""
    times = np.arange(len(rec)) * DT
    if mode == "pchip":
        return PchipInterpolator(times, rec, axis=0)(t)
    i = min(int(t // DT), len(rec) - 2)
    a = (t - times[i]) / DT
    return (1.0 - a) * rec[i] + a * rec[i + 1]


def _hand_run_in_time(integrator, seeds, mode, *, dt=DT / 2,
                      start=TI_START, n_steps=TI_STEPS):
    """:func:`_hand_run` on the file of ``GEO_QUAD`` and ``RULE_QUAD`` with
    temporal interpolation: each stage's field is first blended in time at
    the stage's time (:func:`_in_time` of the u/v columns at the layer
    centres and of the surface), then read at the stage's own ``s``
    (``np.interp`` of the columns at ``s`` clipped to the outer
    centres)."""
    cen = _centres(SIG_S)
    times = np.arange(N_REC) * DT
    cols = np.array([np.column_stack(RULE_QUAD.cols(cen, t))
                     for t in times])                 # (record, K-1, 2)
    eta = np.array([GEO_QUAD.eta(0.0, 0.0, t) for t in times])

    def surf(x, y, t):
        return float(_in_time(mode, eta, t))
    lon, lat, z = (np.array(c, dtype=np.float64) for c in zip(*seeds))
    rows = []
    for i in range(n_steps):
        t_s = start * DT + i * dt
        t_k = tuple(t_s + c * dt for c in C_RK4)

        def read_uv(k, s, t_k=t_k):
            cu, cv = _in_time(mode, cols, t_k[k]).T
            sc = min(max(s, cen[0]), cen[-1])
            return float(np.interp(sc, cen, cu)), float(np.interp(sc, cen,
                                                                  cv))
        lon, lat, z, *_ = _hand(
            integrator, SimpleNamespace(lon=lon, lat=lat, z=z), geo=GEO_QUAD,
            sig=SIG_S, dt=dt, te=t_k, t_end=t_s + dt, surf=surf,
            read_uv=read_uv)
        lon = np.asarray(wrap_lon(lon), dtype=np.float64)
        rows.append((lon, lat, z))
    return tuple(np.array([r[j] for r in rows]).T for j in range(3))


@pytest.mark.parametrize("mode", ["lerp", "pchip"])
def test_temporal_interp_linear_sigma(tmp_path, mode):
    """Temporal interpolation on FESOM-C element data under the option
    (``leastsq``, rk2 and rk4, ``dt`` half the record spacing, from record
    2 over three records: no step falls back, nothing warns), a surface and
    u/v quadratic in time, W nonzero: every stage reads the records blended
    at its own time — linearly, or by the PCHIP of the four records
    (``numerics.temporal_reconstruction: pchip``) — at its own ``s``, from
    its own ``H_p``.  Every position and depth equals that hand rule to
    1e-9 m and is far from the other blend's; the output says
    ``uv_vertical = "linear"``."""
    path = _file(tmp_path, geo=GEO_QUAD, rule=RULE_QUAD)
    seeds = _seeds(geo=GEO_QUAD, t=TI_START * DT)
    recon = "pchip" if mode == "pchip" else "linear"
    other = "lerp" if mode == "pchip" else "pchip"
    for integrator in ("rk2", "rk4"):
        cfg = _cfg(path, integrator=integrator, seeds=seeds, dt=DT / 2,
                   start=TI_START, n_steps=TI_STEPS, temporal_interp=True,
                   temporal_reconstruction=recon, out=f"{integrator}.nc")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _run(cfg)
        runtime = [str(w.message) for w in caught
                   if issubclass(w.category, RuntimeWarning)]
        assert not runtime, runtime
        t = load_trajectories(cfg.data.output_path)
        assert (t["status"] == ParticleStatus.ALIVE).all()
        h_lon, h_lat, h_z = _hand_run_in_time(integrator, seeds, mode)
        _assert_metres(t["lon"], t["lat"], h_lon, h_lat)
        assert np.all(np.abs(t["z"] - h_z) <= 1e-9), np.abs(t["z"] - h_z).max()
        # the other blend's hand rule: 18 .. 21 m apart at the farthest
        o_lon, o_lat, _ = _hand_run_in_time(integrator, seeds, other)
        gap = np.max(_gap(t, {"lon": o_lon, "lat": o_lat}), axis=1)
        assert np.all(gap > 5.0), gap
        attrs = _global_attrs(cfg.data.output_path)
        assert attrs["uv_vertical"] == "linear"
        assert attrs["temporal_reconstruction"] == recon


# ---------------------------------------------------------------------------
# Caches, element search, restarts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("layout,uv_interp", CASES, ids=CASE_IDS)
def test_prefetch_and_shared_cache_sigma(tmp_path, monkeypatch, layout,
                                         uv_interp):
    """Prefetch windows of 3 and the shared snapshot cache each give the
    plain run bit for bit; a second shared cache attaches the published
    record without reading it, marked ``"layer_linear"`` and checked, bit
    for bit the local read."""
    from kelp.core.shared_fields import SharedFieldCache
    path = _file(tmp_path)
    kw = dict(layout=layout, uv_interp=uv_interp, integrator="rk2")
    base = _cfg(path, out="base.nc", **kw)
    _run(base)
    loads = []
    orig = tracking._VelCache._load_window

    def counting(self, start, *a, **k):
        loads.append(start)
        return orig(self, start, *a, **k)
    monkeypatch.setattr(tracking._VelCache, "_load_window", counting)
    pre = _cfg(path, prefetch_size=3, out="pre.nc", **kw)
    _run(pre)
    # the run reads records 0 .. 4: two windows, records 0 .. 2 and 3 .. 5
    assert loads == [0, 3], loads
    shm = tmp_path / "shm"
    shm.mkdir()
    shared = _cfg(path, shared_fields_dir=str(shm),
                  shared_fields_generation="sigma", out="shared.nc", **kw)
    attached = []
    orig_get = tracking._VelCache._shared_get

    def shared_get(self, file_idx, *a, **k):
        attached.append(file_idx)
        return orig_get(self, file_idx, *a, **k)
    monkeypatch.setattr(tracking._VelCache, "_shared_get", shared_get)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _run(shared)
    assert not any("shared_fields" in str(w.message) for w in rec)
    assert sorted(set(attached)) == list(range(N_STEPS + 1))
    for cfg in (pre, shared):
        _assert_same_run(cfg.data.output_path, base.data.output_path)
    # the attached record
    reader, mesh, ds = _open(path, layout)
    r2 = _CountingReader()
    mode = _mode(layout)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        c1 = tracking._VelCache(shared=SharedFieldCache(shm / "c", "tagS"))
        c1.get(3, reader, ds, mesh, mode)
        c2 = tracking._VelCache(shared=SharedFieldCache(shm / "c", "tagS"))
        v2 = c2.get(3, r2, ds, mesh, mode)
    assert r2.reads == 0
    assert not any("shared_fields" in str(w.message) for w in rec)
    local = tracking._VelCache().get(3, reader, ds, mesh, mode)
    for v in (v2, local):
        assert v.uv_layout == layout and v.uv_vertical == "layer_linear"
        assert is_checked(v, mesh)
    for f in ("u", "v", "w", "eta"):
        assert_bitwise(getattr(v2, f), getattr(local, f), f)
    ds.close()


#: a fast flow (1.2 .. 2 m/s east) that moves the particles across
#: element edges, coastal ones included
RULE_FAST = Rule(au=1.2, bu=0.6, av=0.3, bv=-0.4)
#: near element edges (an interior diagonal, the centre node, a coast edge)
WALK_SEEDS = _seeds(lon=(4.93, 4.97, 5.62, 4.30, 5.80),
                    lat=(-0.31, 0.02, -0.55, 0.93, 0.15),
                    s=(0.10, 0.35, 0.55, 0.25, 0.70))


@pytest.mark.parametrize("bc", ["freeslip", "noslip"])
@pytest.mark.parametrize("integrator", INTEGRATORS)
def test_walk_equals_local_sigma(tmp_path, integrator, bc):
    """``elem_search: walk`` with ``edge_leastsq`` on FESOM-C element data
    under the option, particles crossing element edges (coastal elements
    included): bit for bit the run with the per-step search."""
    path = _file(tmp_path, rule=RULE_FAST)
    runs = {}
    for search in ("local", "walk"):
        cfg = _cfg(path, uv_interp="edge_leastsq", integrator=integrator,
                   edge_leastsq_bc=bc, elem_search=search, seeds=WALK_SEEDS,
                   n_steps=5, out=f"{search}.nc")
        _run(cfg)
        runs[search] = cfg.data.output_path
    t = load_trajectories(runs["walk"])
    assert np.isfinite(t["lon"]).all()
    assert (t["status"] == ParticleStatus.ALIVE).all()
    # the particles cross element edges (the walk has edges to cross): at
    # least three times from record to record, one particle into a coastal
    # element (7; with the free-slip ghosts there, back and in again)
    _, mesh, ds = _open(path)
    ds.close()
    lon0, lat0 = (np.array(c) for c in list(zip(*WALK_SEEDS))[:2])
    lon = np.column_stack([lon0, t["lon"]])
    lat = np.column_stack([lat0, t["lat"]])
    e = np.column_stack([find_containing_elements(
        mesh, lon[:, j], lat[:, j],
        find_nearest_nodes(mesh, lon[:, j], lat[:, j]))[0]
        for j in range(lon.shape[1])])
    assert np.sum(e[:, 1:] != e[:, :-1]) >= 3, e
    assert np.any(e == 7)
    assert _global_attrs(runs["walk"])["elem_search"] == "walk"
    assert _global_attrs(runs["walk"])["uv_vertical"] == "linear"
    _assert_same_run(runs["walk"], runs["local"])


@pytest.mark.parametrize("layout,uv_interp", CASES, ids=CASE_IDS)
def test_restart_with_option_sigma(tmp_path, layout, uv_interp):
    """A run cut at a checkpoint and resumed under the option equals the
    continuous run bit for bit (positions, depths, velocities, record
    times, the final restart); the checkpoint records ``"linear"`` and
    resuming it without the option is refused."""
    path = _file(tmp_path)
    reader, mesh, ds = _open(path, layout)
    file_times, file_dt = extract_time_info(ds)
    numerics = NumericsConfig(dt=DT, integrator="rk2", uv_interp=uv_interp,
                              w_interp="barycentric")
    mode = _mode(layout)

    def run(name, state, start, n, start_step_i):
        out = str(tmp_path / f"{name}.nc")
        w = TrajectoryWriter(out, state.n_particles).open()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                tracking.run_single_realization(
                    mesh, ds, reader, state, mode, numerics,
                    start_time_idx=start, n_steps=n, writer=w,
                    checkpoint_freq=2, file_times=file_times,
                    file_dt=file_dt, start_step_i=start_step_i)
        finally:
            w.close()
        return out
    init = _state(mesh, _seeds())
    whole = run("whole", init, 0, 4, 0)
    a = run("a", init, 0, 2, 0)
    r = load_restart(a)
    assert r["uv_vertical"] == "linear"
    tracking._validate_restart_uv_vertical(r["uv_vertical"], mode)
    b = run("b", r["state"], r["file_idx"] + 1, 2, r["step_i"] + 1)
    tw, ta, tb = (load_trajectories(p) for p in (whole, a, b))
    assert_bitwise(np.concatenate([ta["time"], tb["time"]]), tw["time"],
                   "time")
    for k in _FIELDS:
        assert_bitwise(np.concatenate([ta[k], tb[k]], axis=1), tw[k], k)
    # the continuous run is the hand run
    h_lon, h_lat, h_z = _hand_run("rk2", _seeds(), 4)
    _assert_metres(tw["lon"], tw["lat"], h_lon, h_lat)
    assert np.all(np.abs(tw["z"] - h_z) <= 1e-9)
    rw, rb = load_restart(whole), load_restart(b)
    assert (rb["step_i"], rb["file_idx"]) == (rw["step_i"], rw["file_idx"])
    for k in ("lon", "lat", "z"):
        assert_bitwise(getattr(rb["state"], k), getattr(rw["state"], k), k)
    with pytest.raises(ValueError,
                       match="written with mode.uv_vertical 'linear', this "
                             "run has 'layer'"):
        tracking._validate_restart_uv_vertical(
            r["uv_vertical"], _mode(layout, uv_vertical="layer"))
    ds.close()


# ---------------------------------------------------------------------------
# max_levels, diffusion, realizations
# ---------------------------------------------------------------------------

#: ``max_levels: 7`` keeps the interfaces s = 0 .. 0.36 of ``SIG_S``
#: (centres 0.005 .. 0.305)
CUT = 7
#: two particles above the last kept centre, two below it for the whole run
CUT_S = (0.10, 0.22, 0.45, 0.60)
CUT_LON = (4.62, 4.75, 5.13, 5.30)
CUT_LAT = (-0.29, -0.12, 0.28, 0.55)


@pytest.mark.parametrize("integrator", ["euler", "rk2"])
@pytest.mark.parametrize("layout,uv_interp",
                         [CASES[0], CASES[3]], ids=[CASE_IDS[0], CASE_IDS[3]])
def test_max_levels_sigma_run(tmp_path, layout, uv_interp, integrator):
    """``max_levels: 7`` on a sigma mesh: the depth-linear table follows the
    kept levels — every position equals the hand rule on the kept table
    (u/v clipped to the last kept centre, W at the last kept interface
    below it) to 1e-9 m; the particles below the last kept centre at every
    stage read what the layer rule reads there, bit for bit; uncut, they
    read their own depth and go elsewhere."""
    path = _file(tmp_path)
    seeds = _seeds(lon=CUT_LON, lat=CUT_LAT, s=CUT_S)
    kw = dict(layout=layout, uv_interp=uv_interp, integrator=integrator,
              seeds=seeds)
    cut = _cfg(path, max_levels=CUT, out="cut.nc", **kw)
    _run(cut)
    t = load_trajectories(cut.data.output_path)
    assert (t["status"] == ParticleStatus.ALIVE).all()
    sig = SIG_S[:CUT]
    h_lon, h_lat, h_z = _hand_run(integrator, seeds, N_STEPS, sig=sig,
                                  w_hi=1.0 - sig[-1])
    _assert_metres(t["lon"], t["lat"], h_lon, h_lat)
    assert np.all(np.abs(t["z"] - h_z) <= 1e-9)
    deep = np.array(CUT_S) > _centres(sig)[-1]
    layer = _cfg(path, max_levels=CUT, uv_vertical="layer", out="layer.nc",
                 **kw)
    _run(layer)
    t_layer = load_trajectories(layer.data.output_path)
    for k in ("lon", "lat", "z"):
        assert_bitwise(t[k][deep], t_layer[k][deep], k)
    gap = np.max(_gap(t_layer, t), axis=1)
    assert np.all(gap[~deep] > 5.0), gap
    full = _cfg(path, out="full.nc", **kw)
    _run(full)
    t_full = load_trajectories(full.data.output_path)
    gap = np.max(_gap(t_full, t), axis=1)
    assert np.all(gap[deep] > 50.0), gap


def _diffusion(seed, *, kh=0.0, az=0.0):
    return DiffusionConfig(enabled=True, scheme="uniform", kh_zonal=kh,
                           kh_meridional=kh, seed=seed, composition="split",
                           az_enabled=az > 0.0, az_const=az)


@pytest.mark.parametrize("layout,uv_interp", CASES, ids=CASE_IDS)
def test_split_uniform_kh_with_option_sigma(tmp_path, layout, uv_interp):
    """Split uniform Kh on FESOM-C data with the option: the same seed gives
    the same run bit for bit, another seed another run; the kicks move the
    particles off the advected run; every particle stays alive."""
    path = _file(tmp_path)
    kw = dict(layout=layout, uv_interp=uv_interp, integrator="rk2")
    paths = []
    for name, seed in (("a", 11), ("b", 11), ("c", 12)):
        cfg = _cfg(path, diffusion=_diffusion(seed, kh=5.0),
                   out=f"{name}.nc", **kw)
        _run(cfg)
        paths.append(cfg.data.output_path)
    plain = _cfg(path, out="plain.nc", **kw)
    _run(plain)
    _assert_same_run(paths[1], paths[0])
    ta, tc = load_trajectories(paths[0]), load_trajectories(paths[2])
    tp = load_trajectories(plain.data.output_path)
    assert (ta["status"] == ParticleStatus.ALIVE).all()
    assert not np.array_equal(ta["lon"], tc["lon"])
    assert np.all(_gap(ta, tp)[:, -1] > 1.0)
    assert _global_attrs(paths[0])["uv_vertical"] == "linear"
    _assert_kicked_run_advects_linear(path, kw, tp, _diffusion(11, kh=1e-14))


def _assert_kicked_run_advects_linear(path, kw, tp, diffusion):
    """With a vanishing coefficient the kicked run is the plain linear run
    (``tp``), far from the plain layer run: the step that composes the kick
    advects by the depth-linear rule, not by the layer one."""
    tiny = _cfg(path, diffusion=diffusion, out="tiny.nc", **kw)
    _run(tiny)
    layer = _cfg(path, uv_vertical="layer", out="layer.nc", **kw)
    _run(layer)
    tt = load_trajectories(tiny.data.output_path)
    tl = load_trajectories(layer.data.output_path)
    near = max(np.max(_gap(tt, tp)), np.max(np.abs(tt["z"] - tp["z"])))
    far = np.max(_gap(tl, tp), axis=1)
    assert near < 1e-3, near
    assert np.all(far > 1.0), far


@pytest.mark.parametrize("layout,uv_interp", CASES, ids=CASE_IDS)
def test_az_kick_with_option_sigma(tmp_path, layout, uv_interp):
    """The Az kick (``az_const``) on FESOM-C data with the option: the same
    seed gives the same run bit for bit, another seed other depths; the
    kick moves the depths off the advected run; every particle stays
    alive inside its column."""
    path = _file(tmp_path)
    kw = dict(layout=layout, uv_interp=uv_interp, integrator="rk2")
    paths = []
    for name, seed in (("a", 5), ("b", 5), ("c", 6)):
        cfg = _cfg(path, diffusion=_diffusion(seed, az=1.0e-3),
                   out=f"{name}.nc", **kw)
        _run(cfg)
        paths.append(cfg.data.output_path)
    plain = _cfg(path, out="plain.nc", **kw)
    _run(plain)
    _assert_same_run(paths[1], paths[0])
    ta, tc = load_trajectories(paths[0]), load_trajectories(paths[2])
    tp = load_trajectories(plain.data.output_path)
    assert (ta["status"] == ParticleStatus.ALIVE).all()
    assert not np.array_equal(ta["z"], tc["z"])
    assert np.all(np.abs(ta["z"] - tp["z"])[:, -1] > 1e-3)
    h = (GEO.bed(ta["lon"], ta["lat"])
         + GEO.eta(ta["lon"], ta["lat"], (np.arange(N_STEPS) + 1) * DT))
    assert np.all((ta["z"] >= 0.0) & (ta["z"] <= h))
    assert _global_attrs(paths[0])["uv_vertical"] == "linear"
    _assert_kicked_run_advects_linear(path, kw, tp, _diffusion(5, az=1e-14))


@pytest.mark.parametrize("layout,uv_interp", CASES, ids=CASE_IDS)
def test_multi_realization_sigma(tmp_path, layout, uv_interp):
    """Two realizations (``num_runs: 2``): each output is the single run,
    bit for bit, and says ``uv_vertical = "linear"``."""
    path = _file(tmp_path)
    kw = dict(layout=layout, uv_interp=uv_interp, integrator="rk2")
    single = _cfg(path, out="single.nc", **kw)
    _run(single)
    multi = _cfg(path, num_runs=2, out="multi.nc", **kw)
    _run(multi)
    for r in range(2):
        out = tmp_path / f"multi_r{r}.nc"
        _assert_same_run(out, single.data.output_path)
        assert _global_attrs(out)["uv_vertical"] == "linear"
