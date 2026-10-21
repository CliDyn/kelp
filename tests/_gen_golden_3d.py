"""Golden 3-D trajectories, frozen on ``main`` before the depth-linear work.

Each case runs a small synthetic 3-D experiment through one engine path and
returns its trajectories.  ``tests/test_golden_3d_runs.py`` re-runs every case
and compares the result with the stored ``.npz`` bit for bit (dtype, shape,
raw bytes), so a change that moves an existing 3-D run by one bit fails there.
The files were generated once on ``main`` (3607596), before any engine change.
Bits hold only in the generating environment: each directory carries its
``ENV.json`` fingerprint (``tests/_bitwise.py``), and elsewhere the test
compares floats to a tight tolerance (integers and strings stay exact).
The cases build their data with fixture helpers of other test files
(``tests/test_3d_analytical.py``, ``tests/test_fused_3d_bary.py``,
``tests/test_fesomc_w_contract.py``, ``tests/test_healpix_reader.py``,
``tests/_fesom2_synth.py``); the goldens depend on those helpers' default
arguments, and changing those defaults moves the goldens.

Two sets: ``tests/data/golden_3d`` (Numba JIT) and
``tests/data/golden_3d_nojit`` (``NUMBA_DISABLE_JIT=1``: the pure-Python
kernels round differently).
Regenerate only for a change that is MEANT to move existing runs, from the
repository root, each set in its own process (own ``NUMBA_CACHE_DIR``)::

    python tests/_gen_golden_3d.py --write tests/data/golden_3d
    NUMBA_DISABLE_JIT=1 python tests/_gen_golden_3d.py --write tests/data/golden_3d_nojit

Every ``--write`` also (re)writes ``ENV.json`` in that directory: regenerate
a set in one environment only.

What a case returns:

* step-level cases (the integrators called directly, the payload built like
  ``tracking._build_vel_payload``): ``lon, lat, z, status, failure_reason,
  u, v, w`` of the state after every step, ``(step, particle)``;
* run-level cases (``run_single_realization`` with a trajectory writer): the
  same keys as written, ``(particle, record)``, plus ``sample__<name>`` for
  sampled scalars;
* ``run_tracking_fesom2_multiyear_node``: every variable of the output file
  under its own name, ``dims__<var>`` (dimension names), and every attribute
  as a 0-d string, ``attr__<name>`` (global) or ``attr__<var>__<name>``.
  ``attr__kelp_version`` names the commit; the test checks it against
  ``tracking._kelp_version()`` instead of the stored value.

Every RK2 case also runs backward in time as ``<name>_bwd``.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, Optional

if __name__ == "__main__":
    # CLI only (a spawned interpreter): import THIS checkout's kelp (an
    # installed kelp may point at another checkout), find the ``tests.*``
    # fixture helpers, and cap numba's threads like tests/conftest.py, all
    # before numba loads.  Imported by pytest, this module changes neither
    # sys.path nor the environment.
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "src"), str(_ROOT)]
    os.environ.setdefault("NUMBA_NUM_THREADS",
                          str(min(8, os.cpu_count() or 8)))

import numpy as np  # noqa: E402

from kelp.core.io import LazyLerpField, VelocityField  # noqa: E402
from kelp.core.mesh import find_nearest_nodes  # noqa: E402
from kelp.core.particles import make_state  # noqa: E402
from kelp.numerics.diffusion import DiffusionStepArgs  # noqa: E402
from kelp.numerics.integrators import (  # noqa: E402
    euler_step, rk2_step, rk4_step)
from tests._bitwise import write_env_fingerprint  # noqa: E402

STATE_KEYS = ("lon", "lat", "z", "status", "failure_reason", "u", "v", "w")

CASES: Dict[str, Callable[[], Dict[str, np.ndarray]]] = {}

# parent of the run-level cases' scratch directories (set by ``run_case``;
# None = ``tempfile``'s default, i.e. ``$TMPDIR``)
_WORKDIR: Optional[Path] = None


def _case(name: str, *, backward: bool = False):
    """Register ``fn(direction)`` as ``name`` (forward) and, when
    ``backward``, as ``name + "_bwd"`` (time_dir = -1)."""
    def deco(fn):
        CASES[name] = lambda: fn(1.0)
        if backward:
            CASES[name + "_bwd"] = lambda: fn(-1.0)
        return fn
    return deco


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _scratch():
    """A scratch directory for one run-level case, under ``_WORKDIR``."""
    return tempfile.TemporaryDirectory(prefix="kelp_golden_", dir=_WORKDIR)


def _c(a) -> np.ndarray:
    """A C-contiguous float64 copy (broadcast results included)."""
    return np.ascontiguousarray(np.array(a, dtype=np.float64))


def _init_state(mesh, lon, lat, z):
    st = make_state(np.asarray(lon, dtype=np.float64),
                    np.asarray(lat, dtype=np.float64),
                    np.asarray(z, dtype=np.float64))
    return st.replace(
        nearest_node=find_nearest_nodes(mesh, st.lon, st.lat),
        failure_reason=np.zeros(st.n_particles, dtype=np.int8))


def _payload(a, b, ta, tb, s, direction):
    """(vel, vel_next, dt_to_next) at time ``s`` in the bracket [ta, tb],
    as ``tracking._build_vel_payload`` builds them."""
    alpha = (s - ta) / (tb - ta)
    if abs(alpha) < 1e-12:
        vel = a
    elif abs(alpha - 1.0) < 1e-12:
        vel = b
    else:
        vel = LazyLerpField(a, b, alpha)
    if direction > 0:
        return vel, b, tb - s
    return vel, a, s - ta


def _run_steps(step_fn, state, mesh, dt, direction, n_steps, per_step, **kw):
    """Take ``n_steps`` steps; ``per_step(i)`` returns the step's ``vel``
    and payload keywords.  Returns the state after every step as
    ``(step, particle)`` arrays."""
    rows = {k: [] for k in STATE_KEYS}
    for i in range(n_steps):
        p = per_step(i)
        vel = p.pop("vel")
        state = step_fn(state, vel, mesh, dt, direction, **p, **kw)
        for k in STATE_KEYS:
            a = getattr(state, k)
            if a is None:
                raise RuntimeError(f"step {i}: state.{k} is None")
            rows[k].append(np.array(a, copy=True))
    return {k: np.stack(v) for k, v in rows.items()}


def _run_written(run, n_particles, workdir, sample_names=()):
    """Run ``run(writer)`` with a NetCDF trajectory writer; return the
    written ``lon, lat, z, status, failure_reason, u, v, w`` (+
    ``sample__<name>``), ``(particle, record)``."""
    from kelp.writers import TrajectoryWriter, load_trajectories
    path = str(Path(workdir) / "traj.nc")
    writer = TrajectoryWriter(path, n_particles)
    writer.open()
    try:
        run(writer)
    finally:
        writer.close()
    traj = load_trajectories(path)
    names = {"lon": "lon", "lat": "lat", "z": "z", "status": "status",
             "failure_reason": "failure_reason",
             "u": "vel_u", "v": "vel_v", "w": "vel_w"}
    names.update({f"sample__{s}": s for s in sample_names})
    return {k: traj[src] for k, src in names.items()}


# ---------------------------------------------------------------------------
# FESOM2-style element-layout mesh (tests/test_3d_analytical._make_mesh)
# ---------------------------------------------------------------------------

ELEM_LEVELS = [0.0, 10.0, 25.0, 50.0, 100.0, 200.0]
ELEM_DT = 3600.0
ELEM_STEPS = 8


def _elem_mesh():
    from tests.test_3d_analytical import _make_mesh
    mesh = _make_mesh(5.0, 5.0, span=1.0, nx=5, ny=5, z_levels=ELEM_LEVELS,
                      bottom_depth=200.0)
    # bathymetry 60 m (west edge) .. 200 m (east edge), one shallow node
    mesh.depth[:] = 60.0 + 70.0 * (mesh.nodes_lon - 4.0)
    mesh.depth[12] = 35.0                     # the centre node (5, 5)
    return mesh


def _elem_field(mesh, s):
    """Element u/v (K-1, E) varying with layer and position, node w (K, N)
    (positive down, zero at the surface); ``s`` shifts the snapshot."""
    k = np.arange(mesh.sigma.size - 1, dtype=np.float64)[:, None]
    x = (mesh.elem_center_lon - 5.0)[None, :]
    y = (mesh.elem_center_lat - 5.0)[None, :]
    u = (0.35 - 0.04 * k) * (1.0 + 0.4 * x) + 0.15 * y + 0.05 * s
    v = 0.12 * (1.0 + 0.2 * k) * np.sin(2.0 * x) - 0.08 * y * (1.0 + s)
    zk = (mesh.sigma / mesh.sigma[-1])[:, None]
    xn = (mesh.nodes_lon - 5.0)[None, :]
    w = 1.5e-3 * np.sin(np.pi * zk) * (1.0 + 0.5 * xn) * (1.0 - 0.3 * s)
    return VelocityField(u=_c(u), v=_c(v), eta=np.zeros(mesh.n_nodes),
                         w=_c(w))


def _elem_seeds(mesh):
    i = np.arange(32, dtype=np.float64)
    lon = 4.1 + 1.85 * ((i * 0.618034) % 1.0)
    lat = 4.1 + 1.8 * ((i * 0.381966 + 0.13) % 1.0)
    lon[-4:] = [5.96, 5.975, 5.99, 5.998]    # leave through the east edge
    depths = np.array([0.0, 5.0, 10.0, 17.5, 25.0, 37.0, 50.0, 75.0,
                       100.0, 150.0, 199.0, 260.0])
    z = depths[np.arange(32) % depths.size]
    return _init_state(mesh, lon, lat, z)


@_case("fesom2_elem_euler_fused")
def _elem_euler_fused(direction):
    """Element layout, leastsq u/v and W, steady field: the fused 3-D Euler
    kernel."""
    mesh = _elem_mesh()
    vel = _elem_field(mesh, 0.0)
    return _run_steps(euler_step, _elem_seeds(mesh), mesh, ELEM_DT,
                      direction, ELEM_STEPS, lambda i: {"vel": vel},
                      w_interp="leastsq")


@_case("fesom2_elem_rk2_unfused_lerp", backward=True)
def _elem_rk2_unfused_lerp(direction):
    """Element layout, leastsq u/v and W, two snapshots blended in time:
    the unfused 3-D RK2 path."""
    mesh = _elem_mesh()
    a, b = _elem_field(mesh, 0.0), _elem_field(mesh, 1.0)
    T = ELEM_STEPS * ELEM_DT
    t0 = 0.0 if direction > 0 else T

    def per_step(i):
        vel, nxt, dtn = _payload(a, b, 0.0, T, t0 + direction * i * ELEM_DT,
                                 direction)
        return {"vel": vel, "vel_next": nxt, "dt_to_next": dtn}
    return _run_steps(rk2_step, _elem_seeds(mesh), mesh, ELEM_DT, direction,
                      ELEM_STEPS, per_step, w_interp="leastsq")


# ---------------------------------------------------------------------------
# node-layout HEALPix dual mesh (tests/test_fused_3d_bary.py)
# ---------------------------------------------------------------------------

NODE_DT = 21600.0
NODE_STEPS = 8
NODE_P = 48
NODE_KW = {"uv_interp": "barycentric", "w_interp": "barycentric"}


def _dual_mesh():
    """The module fixture ``mesh`` of tests/test_fused_3d_bary.py, rebuilt
    (a pytest fixture cannot be called directly)."""
    import healpy
    from kelp.readers.healpix import build_healpix_dual_mesh
    from tests.test_fused_3d_bary import NSIDE, NPIX, Z_LEVELS
    lon, lat = healpy.pix2ang(NSIDE, np.arange(NPIX), nest=True, lonlat=True)
    depth = 60.0 + 3500.0 * np.abs(np.sin(np.deg2rad(lat * 2.0)))
    depth[::17] = 35.0
    return build_healpix_dual_mesh(NSIDE, wet_mask=np.ones(NPIX, bool),
                                   depth=depth, z_levels=Z_LEVELS)


def _node_fields(mesh, *seeds):
    from tests.test_fused_3d_bary import _field
    return [_field(mesh, s) for s in seeds]


def _node_seeds(mesh):
    from tests.test_fused_3d_bary import _seeds
    return _seeds(mesh, n=NODE_P)


def _node_lerp_steps(step_fn, mesh, a, b, direction, extra=None):
    """Node snapshots ``a`` and ``b`` bracket the whole run, blended in
    time; ``extra(i)`` adds payload keywords (diffusion)."""
    T = NODE_STEPS * NODE_DT
    t0 = 0.0 if direction > 0 else T

    def per_step(i):
        vel, nxt, dtn = _payload(a, b, 0.0, T, t0 + direction * i * NODE_DT,
                                 direction)
        p = {"vel": vel, "vel_next": nxt, "dt_to_next": dtn}
        if extra is not None:
            p.update(extra(i))
        return p
    return _run_steps(step_fn, _node_seeds(mesh), mesh, NODE_DT, direction,
                      NODE_STEPS, per_step, **NODE_KW)


@_case("fesom2_node_rk2_fused_lerp", backward=True)
def _node_rk2_fused_lerp(direction):
    """Node layout, barycentric u/v and W, linear time blend: the fused
    3-D barycentric RK2 kernel."""
    mesh = _dual_mesh()
    return _node_lerp_steps(rk2_step, mesh, *_node_fields(mesh, 1, 2),
                            direction)


@_case("fesom2_node_rk2_fused_pchip", backward=True)
def _node_rk2_fused_pchip(direction):
    """Node layout, 4-snapshot PCHIP in time: the fused 3-D barycentric
    RK2 PCHIP kernel; the run stays in the central bracket."""
    mesh = _dual_mesh()
    snaps = _node_fields(mesh, 11, 12, 13, 14)
    T = NODE_STEPS * NODE_DT
    times = [0.0, T, 2.0 * T, 3.0 * T]
    t0 = T if direction > 0 else 2.0 * T

    def per_step(i):
        q = t0 + direction * i * NODE_DT
        vel, nxt, dtn = _payload(snaps[1], snaps[2], T, 2.0 * T, q,
                                 direction)
        return {"vel": vel, "vel_next": nxt, "dt_to_next": dtn,
                "recon": (snaps, times, q, int(direction))}
    return _run_steps(rk2_step, _node_seeds(mesh), mesh, NODE_DT, direction,
                      NODE_STEPS, per_step, **NODE_KW)


@_case("fesom2_node_rk4_unfused")
def _node_rk4_unfused(direction):
    """Node layout, barycentric u/v and W, linear time blend: the unfused
    3-D RK4 path (time-aware stages)."""
    mesh = _dual_mesh()
    return _node_lerp_steps(rk4_step, mesh, *_node_fields(mesh, 21, 22),
                            direction)


@_case("split_uniform_diffusion_rk2", backward=True)
def _split_uniform_diffusion_rk2(direction):
    """Fused node RK2 with the split uniform horizontal Kh kick
    (anisotropic, counter-RNG keyed by the step)."""
    mesh = _dual_mesh()

    def extra(i):
        return {"diffusion": DiffusionStepArgs(
            scheme="uniform", kh_x=50.0, kh_y=20.0, seed=2026, real_id=0,
            step_i=i, composition="split")}
    return _node_lerp_steps(rk2_step, mesh, *_node_fields(mesh, 31, 32),
                            direction, extra=extra)


@_case("az_const_euler")
def _az_const_euler(direction):
    """Unfused node Euler (barycentric u/v) with the vertical Az kick
    (constant Az, reflective column ends)."""
    mesh = _dual_mesh()
    (vel,) = _node_fields(mesh, 41)

    def per_step(i):
        return {"vel": vel, "diffusion": DiffusionStepArgs(
            scheme="uniform", kh_x=0.0, kh_y=0.0, seed=99, real_id=0,
            step_i=i, composition="split", az_enabled=True, az_const=5e-3)}
    return _run_steps(euler_step, _node_seeds(mesh), mesh, NODE_DT,
                      direction, NODE_STEPS, per_step, **NODE_KW)


# ---------------------------------------------------------------------------
# run_single_realization on an in-memory FESOM2 node dataset
# ---------------------------------------------------------------------------

RUN_LEVELS = [0.0, 10.0, 25.0, 50.0, 100.0]
RUN_FILE_DT = 3600.0
RUN_DT = 1800.0


def _run_mesh(depth):
    from tests.test_3d_analytical import _make_mesh
    mesh = _make_mesh(5.0, 5.0, span=1.0, nx=5, ny=5, z_levels=RUN_LEVELS,
                      bottom_depth=100.0)
    mesh.depth[:] = depth(mesh.nodes_lon, mesh.nodes_lat)
    return mesh


def _fesom2_node_ds(mesh, n_time, eta, scalars=False):
    """FESOM2-shaped node dataset: unod/vnod (time, nz1, nod2), w (time,
    nz, nod2; file convention positive up), ssh (time, nod2)."""
    import xarray as xr
    K, N = mesh.sigma.size, mesh.n_nodes
    t = np.arange(n_time, dtype=np.float64)[:, None, None]
    k = np.arange(K - 1, dtype=np.float64)[None, :, None]
    x = (mesh.nodes_lon - 5.0)[None, None, :]
    y = (mesh.nodes_lat - 5.0)[None, None, :]
    zk = (mesh.sigma / mesh.sigma[-1])[None, :, None]
    unod = (0.25 - 0.03 * k) * (1.0 + 0.3 * x) + 0.05 * y + 0.02 * np.sin(t)
    vnod = 0.1 * (1.0 + 0.2 * k) * np.cos(2.0 * x) - 0.03 * y * (1.0 + 0.1 * t)
    w = 1e-3 * np.sin(np.pi * zk) * (1.0 + 0.4 * x) * np.cos(0.5 * t)
    shp_u, shp_w = (n_time, K - 1, N), (n_time, K, N)
    dvars = {
        "unod": (("time", "nz1", "nod2"), _c(np.broadcast_to(unod, shp_u))),
        "vnod": (("time", "nz1", "nod2"), _c(np.broadcast_to(vnod, shp_u))),
        "w": (("time", "nz", "nod2"), _c(np.broadcast_to(w, shp_w))),
        "ssh": (("time", "nod2"),
                _c(np.broadcast_to(np.asarray(eta)[:, None], (n_time, N)))),
    }
    if scalars:
        z_ifc = mesh.sigma[None, :, None]
        z_mid = (0.5 * (mesh.sigma[:-1] + mesh.sigma[1:]))[None, :, None]
        s_lev = 10.0 + 0.1 * z_ifc + 2.0 * x - y + 0.5 * t
        s_mid = 35.0 - 0.02 * z_mid + x * y + 0.1 * t
        dvars["s_lev"] = (("time", "nz", "nod2"),
                          _c(np.broadcast_to(s_lev, shp_w)))
        dvars["s_mid"] = (("time", "nz1", "nod2"),
                          _c(np.broadcast_to(s_mid, shp_u)))
    return xr.Dataset(dvars, coords={"time": np.arange(n_time) * RUN_FILE_DT})


def _run_fesom2(mesh, ds, init, direction, numerics, n_steps, workdir,
                sample_vars=()):
    from kelp.core.config import ModeConfig
    from kelp.readers import get_reader
    from kelp.tracking import run_single_realization
    n_time = ds.sizes["time"]
    mode = ModeConfig(Fdrift2D=False, Fdrift3D=True, time_dir=direction,
                      var_u="unod", var_v="vnod", var_w="w", var_eta="ssh")
    file_times = np.arange(n_time) * RUN_FILE_DT

    def run(writer):
        run_single_realization(
            mesh, ds, get_reader("fesom2"), init, mode, numerics,
            start_time_idx=0 if direction > 0 else n_time - 1,
            n_steps=n_steps, n_time_total=n_time, writer=writer,
            temporal_interp=True, file_times=file_times,
            file_dt=RUN_FILE_DT, sample_vars=list(sample_vars) or None)
    return _run_written(run, init.n_particles, workdir, sample_vars)


@_case("beaching_resuspension_rk2", backward=True)
def _beaching_resuspension_rk2(direction):
    """A shallow strip falls dry on a falling tide and rewets: depth +
    stochastic beaching, auto-or-stochastic resuspension, fused node RK2."""
    from kelp.core.config import BeachingConfig, NumericsConfig

    def depth(lon, lat):
        d = np.full(lon.shape, 90.0)
        d[(lon == 5.5) & (lat >= 4.5) & (lat <= 5.5)] = 1.0   # the strip
        return d
    mesh = _run_mesh(depth)
    eta = [0.0, -0.45, -0.85, -0.85, -0.45, 0.0, 0.3, 0.0]
    ds = _fesom2_node_ds(mesh, len(eta), eta)
    lon = np.concatenate([np.repeat([5.22, 5.3, 5.45, 5.6], 3),
                          np.repeat([4.3, 4.7], 4), [5.97, 5.985] * 2])
    lat = np.concatenate([np.tile([4.6, 5.0, 5.4], 4),
                          np.tile([4.3, 4.8, 5.2, 5.7], 2),
                          [4.4, 4.4, 5.6, 5.6]])
    z = np.concatenate([np.tile([0.2, 0.5, 0.8], 4),
                        np.tile([5.0, 30.0, 60.0, 85.0], 2), [20.0] * 4])
    beach = BeachingConfig(
        enabled=True, criteria=["depth", "stochastic"], depth_min=0.3,
        use_eta=True, t_beach=10800.0, resuspension="auto_or_stochastic",
        t_resuspend=7200.0, min_wet_steps=1, seed=4242)
    numerics = NumericsConfig(dt=RUN_DT, integrator="rk2",
                              uv_interp="barycentric",
                              w_interp="barycentric", temporal_interp=True,
                              beaching=beach)
    with _scratch() as d:
        return _run_fesom2(mesh, ds, _init_state(mesh, lon, lat, z),
                           direction, numerics, 14, d)


@_case("scalar_sampling_rk2", backward=True)
def _scalar_sampling_rk2(direction):
    """Record-point sampling of an interface (K, N) and a mid-layer
    (K-1, N) scalar along node RK2 trajectories; leastsq W keeps the step
    on the unfused barycentric path (time-aware stages)."""
    from kelp.core.config import NumericsConfig
    mesh = _run_mesh(lambda lon, lat: 60.0 + 20.0 * (lon - 4.0))
    ds = _fesom2_node_ds(mesh, 5, np.zeros(5), scalars=True)
    i = np.arange(24, dtype=np.float64)
    lon = 4.15 + 1.7 * ((i * 0.618034) % 1.0)
    lat = 4.15 + 1.7 * ((i * 0.381966 + 0.29) % 1.0)
    z = np.array([0.0, 4.0, 10.0, 17.5, 25.0, 33.0, 50.0, 64.0, 79.0,
                  99.0, 12.5, 40.0] * 2)
    numerics = NumericsConfig(dt=RUN_DT, integrator="rk2",
                              uv_interp="barycentric",
                              w_interp="leastsq", temporal_interp=True)
    with _scratch() as d:
        return _run_fesom2(mesh, ds, _init_state(mesh, lon, lat, z),
                           direction, numerics, 8, d,
                           sample_vars=("s_lev", "s_mid"))


# ---------------------------------------------------------------------------
# reader-path cases: the stored u/v must vary with depth
# ---------------------------------------------------------------------------

LAYER_CONTRAST = 1e-3          # m/s, minimum |u/v difference| between layers


def _check_layer_contrast(case, u, v):
    """Stored u and v (layer axis -2, NaN padding allowed) differ by more
    than ``LAYER_CONTRAST`` between every two adjacent finite layers of every
    column: a change of the vertical u/v interpolation moves the run."""
    for comp, a in (("u", u), ("v", v)):
        d = np.abs(np.diff(np.asarray(a, dtype=np.float64), axis=-2))
        d = d[np.isfinite(d)]
        if d.size == 0 or d.min() <= LAYER_CONTRAST:
            raise RuntimeError(
                f"{case}: stored {comp} does not vary between adjacent layers "
                f"(min |difference| {d.min() if d.size else None})")


def _check_two_layers_visited(case, z_start, z_rec, alive, levels):
    """At least one particle visits two layers (its release depth and its
    ALIVE records), so the layer contrast is actually sampled."""
    levels = np.asarray(levels, dtype=np.float64)
    n_layers = []
    for p in range(len(z_start)):
        zp = np.concatenate([[z_start[p]], z_rec[p][alive[p]]])
        k = np.clip(np.searchsorted(levels, zp, side="right") - 1,
                    0, levels.size - 2)
        n_layers.append(np.unique(k).size)
    if max(n_layers) < 2:
        raise RuntimeError(f"{case}: no particle visits two layers")


# ---------------------------------------------------------------------------
# FESOM-C sigma file (tests/test_fesomc_w_contract._write)
# ---------------------------------------------------------------------------

@_case("fesomc_sigma_rk2", backward=True)
def _fesomc_sigma_rk2(direction):
    """FESOM-C sigma coordinates: stretched sigma, sloping bottom, a tide,
    a dia-sigma flux and layered element u/v; RK2 with a linear time
    blend (the sigma-fraction step)."""
    from kelp.core.config import NumericsConfig
    from kelp.readers import extract_time_info, get_reader
    from kelp.tracking import run_single_realization
    from types import SimpleNamespace
    from tests.test_fesomc_w_contract import SIG_S, _grid, _mode, _write
    lon_n, lat_n, tris, _nv = _grid()
    sig = SIG_S
    K, E = sig.size, tris.shape[0]
    n_time = 7
    times = np.arange(n_time) * 900.0
    tide = 0.8 * np.sin(2.0 * np.pi * times / 5400.0)
    eta = tide[:, None] * (1.0 + 0.1 * (lon_n - 5.0))[None, :]
    depth = 12.0 + 6.0 * (lat_n + 1.0) + 2.0 * (lon_n - 4.0)
    H = depth[None, :] + eta                                    # (T, N)
    wvel = 2e-5 * H[:, None, :] * np.sin(np.pi * (1.0 - sig))[None, :, None]
    wvel[:, 0, :] = 0.0                                # no flux at the surface
    wvel[:, -1, :] = 0.0                               # ... and the bottom
    kk = ((np.arange(K - 1) + 0.5) / (K - 1))[None, None, :]
    lonc = lon_n[tris].mean(1)[None, :, None]
    latc = lat_n[tris].mean(1)[None, :, None]
    ph = 2.0 * np.pi * times[:, None, None] / 5400.0
    u_e = 0.25 * (1.0 - 0.5 * kk) * (1.0 + 0.3 * (lonc - 5.0)) \
        + 0.05 * np.cos(ph)
    v_e = 0.15 * (1.0 + 0.4 * kk) * (1.0 + 0.2 * latc) - 0.04 * np.sin(ph)
    shp = (n_time, E, K - 1)
    # nine around the centre node (a corner node's stencil is singular on
    # this 3 x 3 grid), two leaving through the east edge
    lon = np.concatenate([np.repeat([4.6, 5.0, 5.4], 3), [5.996, 5.998]])
    lat = np.concatenate([np.tile([-0.4, 0.0, 0.4], 3), [0.3, -0.4]])
    frac = np.concatenate([np.tile([0.1, 0.4, 0.8], 3), [0.5, 0.95]])
    h0 = 12.0 + 6.0 * (lat + 1.0) + 2.0 * (lon - 4.0)
    numerics = NumericsConfig(dt=450.0, integrator="rk2", output_freq=1,
                              temporal_reconstruction="linear",
                              w_interp="leastsq")
    with _scratch() as d:
        p = _write(Path(d) / "fesomc.nc", times, eta, _c(wvel), sig=sig,
                   u_elem=_c(np.broadcast_to(u_e, shp)),
                   v_elem=_c(np.broadcast_to(v_e, shp)), depth=depth)
        reader = get_reader("fesomc")
        with contextlib.redirect_stdout(io.StringIO()):     # load summary
            mesh = reader.load_mesh(SimpleNamespace(meshpath=str(p)))
        ds = reader.open_dataset(str(p))
        file_times, file_dt = extract_time_info(ds)
        init = _init_state(mesh, lon, lat, frac * h0)

        def run(writer):
            run_single_realization(
                mesh, ds, reader, init, _mode(time_dir=direction), numerics,
                start_time_idx=0 if direction > 0 else n_time - 1,
                n_steps=12, n_time_total=n_time, writer=writer,
                temporal_interp=True, file_times=file_times,
                file_dt=file_dt)
        try:
            return _run_written(run, init.n_particles, d)
        finally:
            ds.close()


# ---------------------------------------------------------------------------
# HEALPix store (tests/test_healpix_reader._make_uv_store, with_w=True)
# ---------------------------------------------------------------------------

def _rewrite_healpix_uv(store, plon, plat):
    """Overwrite the store's u/v (``avg_uoe``/``avg_von``, (time, level,
    value)) with values that vary by level, cell and record; the NaN padding
    below each column's finite prefix stays exactly as written."""
    import netCDF4
    with netCDF4.Dataset(str(store), "r+") as nc:
        nc.set_auto_mask(False)
        u = nc.variables["avg_uoe"]
        if u.dimensions != ("time", "level", "value"):
            raise RuntimeError(f"unexpected u layout {u.dimensions}")
        nt, nl, npix = u.shape
        t = np.arange(nt, dtype=np.float64)[:, None, None]
        k = np.arange(nl, dtype=np.float64)[None, :, None]
        x = (np.asarray(plon) / 360.0)[None, None, :]
        y = (np.asarray(plat) / 60.0)[None, None, :]
        new = {"avg_uoe": 0.4 - 0.12 * k + 0.03 * y + 0.01 * t,
               "avg_von": 0.05 - 0.05 * k + 0.02 * x - 0.005 * t}
        for name, val in new.items():
            var = nc.variables[name]
            old = var[:]
            var[:] = np.where(np.isfinite(old), val, old)
            if not np.array_equal(np.isfinite(var[:]), np.isfinite(old)):
                raise RuntimeError(f"{name}: the NaN padding moved")


def _read_healpix_uv(store):
    import netCDF4
    with netCDF4.Dataset(str(store)) as nc:
        nc.set_auto_mask(False)
        return nc.variables["avg_uoe"][:], nc.variables["avg_von"][:]


@_case("healpix_3d_rk2", backward=True)
def _healpix_3d_rk2(direction):
    """HEALPix native reader: columns of 1, 2 and 3 finite layers (padding
    and below-bottom zeroing), u/v varying by level, cell and record (the
    store rewritten in place), w on the interfaces, float32 velocities,
    fused node RK2 with a linear time blend."""
    from kelp.core.config import ModeConfig, NumericsConfig
    from kelp.readers import extract_time_info
    from kelp.readers.healpix import HealpixReader
    from kelp.tracking import run_single_realization
    from tests.test_healpix_reader import (LEVELS_M, NTIME, _make_uv_store,
                                           _mesh_cfg, _pix_lonlat, _wet)
    nside = 8
    plon, plat = _pix_lonlat(nside, "nest")
    nfin = np.where(_wet(plat), np.where(plat < 0, 3, 2), 0)
    nfin[::5] = np.where(_wet(plat[::5]), 1, 0)
    lon = np.repeat([-150.0, -60.0, 0.0, 45.0, 120.0, 170.0], 4)
    lat = np.tile([-8.0, -2.0, 3.0, 51.0], 6)   # wet rows end at 54.3
    # off the layer centres (5, 20, 50 m) and the levels (0, 10, 30, 70 m)
    z = np.array([6.3, 15.0, 25.0, 29.5, 35.0, 47.6, 65.0, 69.0] * 3)
    mode = ModeConfig(Fdrift2D=False, Fdrift3D=True, time_dir=direction)
    numerics = NumericsConfig(dt=21600.0, integrator="rk2",
                              uv_interp="barycentric",
                              w_interp="barycentric", temporal_interp=True,
                              velocity_float32=True)
    with _scratch() as d:
        store = _make_uv_store(Path(d) / "3D_golden_ocean.nc", nside=nside,
                               with_eta=True, with_w=True,
                               n_finite_levels=nfin, w_value=1e-4)
        _rewrite_healpix_uv(store, plon, plat)
        _check_layer_contrast("healpix_3d_rk2", *_read_healpix_uv(store))
        mesh = HealpixReader.load_mesh(_mesh_cfg(store, levels=LEVELS_M),
                                       mode_cfg=mode)
        ds = HealpixReader.open_dataset(str(store), mode_cfg=mode)
        file_times, file_dt = extract_time_info(ds)
        init = _init_state(mesh, lon, lat, z)

        def run(writer):
            run_single_realization(
                mesh, ds, HealpixReader, init, mode, numerics,
                start_time_idx=0 if direction > 0 else NTIME - 1,
                n_steps=8, n_time_total=NTIME, writer=writer,
                temporal_interp=True, file_times=file_times,
                file_dt=file_dt)
        try:
            out = _run_written(run, init.n_particles, d)
        finally:
            ds.close()
    from kelp.core.particles import ParticleStatus
    _check_two_layers_visited("healpix_3d_rk2", z, out["z"],
                              out["status"] == ParticleStatus.ALIVE, LEVELS_M)
    return out


# ---------------------------------------------------------------------------
# run_tracking end to end: multi-year FESOM2 files (tests/_fesom2_synth)
# ---------------------------------------------------------------------------

# write_multiyear writes 0.1 into every value.  The run rewrites unod, vnod,
# temp and w in place with fields that vary by layer (0-10, 10-30 m), node
# and year: u/v differ between the layers (a change of the vertical u/v
# interpolation moves the run), and w is small and upward (the file
# convention), growing with depth on the interfaces (0, 10, 30 m), so the
# particles rise a few metres and stay below the surface.
MULTIYEAR_YEAR = {1958: 1.0, 1959: 1.25}
MULTIYEAR_W_UP = np.array([1.5e-5, 2.25e-5, 3.0e-5])


def _multiyear_field(var, year):
    """(levels, nodes) values of ``var`` in ``year``."""
    from tests._fesom2_synth import LAT_DEG, LON_DEG
    f = MULTIYEAR_YEAR[year]
    x, y = LON_DEG[None, :], LAT_DEG[None, :]
    if var == "unod":
        return (np.array([0.06, 0.02])[:, None] + 0.01 * x - 0.008 * y) * f
    if var == "vnod":
        return (np.array([0.03, -0.02])[:, None] + 0.006 * x + 0.004 * y) * f
    if var == "temp":
        return (np.array([12.0, 6.0])[:, None] + 0.5 * x - 0.3 * y
                + 0.5 * (year - 1958))
    if var == "w":
        return np.broadcast_to(MULTIYEAR_W_UP[:, None] * f, (3, x.size))
    raise ValueError(var)


def _rewrite_multiyear(data, years):
    """Overwrite unod, vnod, temp and w in the ``<var>.fesom.<year>.nc``
    files in place (every record of a year gets the same values)."""
    import netCDF4
    for var, vdim in (("unod", "nz1"), ("vnod", "nz1"), ("temp", "nz1"),
                      ("w", "nz")):
        for year in years:
            path = str(data / f"{var}.fesom.{year}.nc")
            with netCDF4.Dataset(path, "r+") as nc:
                v = nc.variables[var]
                if v.dimensions != ("time", vdim, "nod2"):
                    raise RuntimeError(
                        f"unexpected {var} layout {v.dimensions}")
                v[:] = np.broadcast_to(_multiyear_field(var, year)[None],
                                       v.shape).astype(v.dtype)


def _read_multiyear_uv(data, years):
    import netCDF4
    out = []
    for var in ("unod", "vnod"):
        parts = []
        for year in years:
            with netCDF4.Dataset(str(data / f"{var}.fesom.{year}.nc")) as nc:
                nc.set_auto_mask(False)
                parts.append(nc.variables[var][:])
        out.append(np.concatenate(parts))
    return out


def _multiyear_config(workdir):
    """A node-layout multi-year FESOM2 run crossing the 1958/1959 file
    boundary: prefetch windows of 2, one sampled scalar, failure reasons
    tracked, daily means (``time_averaging: mean``); the files of
    ``write_multiyear`` with unod, vnod, temp and w rewritten
    (``_rewrite_multiyear``)."""
    from kelp.core.config import (AppConfig, DataConfig, MeshConfig,
                                  ModeConfig, NumericsConfig, TimeConfig)
    from tests._fesom2_synth import write_meshdiag, write_multiyear
    workdir = Path(workdir)
    data = workdir / "data"
    data.mkdir()
    years = {1958: 365, 1959: 2}          # contiguous daily records
    for var in ("unod", "vnod", "temp"):
        write_multiyear(data, var, years, node_layout=True, vertical="nz1")
    write_multiyear(data, "w", years, node_layout=True, vertical="nz")
    write_multiyear(data, "ssh", years, node_layout=True, vertical=None)
    _rewrite_multiyear(data, years)
    _check_layer_contrast("run_tracking_fesom2_multiyear_node",
                          *_read_multiyear_uv(data, years))
    mesh_path = write_meshdiag(workdir / "fesom.mesh.diag.nc", "ugrid")
    seeds = workdir / "seeds.dat"
    # depths off the levels (0, 10, 30 m) and the layer centres (5, 20 m);
    # the last seed leaves through the corner on purpose (failure_reason)
    seeds.write_text("".join(f"{lo} {la} {z}\n" for lo, la, z in [
        (0.2, 0.3, 12.3), (0.5, 0.5, 17.6), (0.7, 0.2, 24.1),
        (0.3, 0.8, 27.4), (0.72, 0.55, 8.7), (0.98, 0.97, 14.2)]))
    return AppConfig(
        mode=ModeConfig(Fdrift2D=False, Fdrift3D=True, var_u="unod",
                        var_v="vnod", var_w="w", var_eta="ssh",
                        sample_vars=["temp"], track_failure_reason=True),
        numerics=NumericsConfig(dt=21600.0, integrator="rk2",
                                uv_interp="barycentric",
                                w_interp="barycentric", temporal_interp=True,
                                prefetch_size=2),
        mesh=MeshConfig(type="fesom2", meshpath=str(mesh_path)),
        data=DataConfig(ncinput=str(data), inicloud=str(seeds),
                        output_path=str(workdir / "traj.nc"),
                        time_averaging="mean"),
        time=TimeConfig(start_time_idx=362, n_steps=12),
    )


def _read_output_file(path):
    """Every variable of a trajectory file (load_trajectories, plus the raw
    netCDF4 read of any other variable), its dimension names and every
    attribute as a 0-d string."""
    import netCDF4
    from kelp.writers import load_trajectories
    out = dict(load_trajectories(str(path)))
    with netCDF4.Dataset(str(path)) as nc:
        nc.set_auto_mask(False)
        for name in nc.ncattrs():
            out[f"attr__{name}"] = np.array(str(nc.getncattr(name)))
        for vname, var in nc.variables.items():
            if vname not in out:
                out[vname] = np.array(var[:])
            out[f"dims__{vname}"] = np.array(" ".join(var.dimensions))
            for name in var.ncattrs():
                out[f"attr__{vname}__{name}"] = np.array(
                    str(var.getncattr(name)))
    return out


@_case("run_tracking_fesom2_multiyear_node")
def _run_tracking_fesom2_multiyear_node(direction):
    """``run_tracking`` end to end on ``_multiyear_config``: the whole
    output file, attributes included."""
    from kelp.tracking import run_tracking
    with _scratch() as d:
        cfg = _multiyear_config(d)
        with contextlib.redirect_stdout(io.StringIO()):
            run_tracking(cfg)
        out = _read_output_file(cfg.data.output_path)
    _check_multiyear_depths(out)
    return out


def _check_multiyear_depths(out):
    """Every recorded ALIVE depth lies strictly between the surface and the
    bed, every particle alive through the run moved > 1 m vertically, and
    one particle visits both layers."""
    from kelp.core.particles import ParticleStatus
    from tests._fesom2_synth import ZBAR_N_BOTTOM_NEG, ZBAR_NEG
    bed = float(np.abs(ZBAR_N_BOTTOM_NEG).min())
    alive = out["status"] == ParticleStatus.ALIVE
    _check_two_layers_visited("run_tracking_fesom2_multiyear_node",
                              out["release_z"], out["z"], alive,
                              np.abs(ZBAR_NEG))
    z = out["z"][alive]
    if not (z.size and np.all((z > 0.0) & (z < bed))):
        raise RuntimeError(f"multi-year case: alive z outside (0, {bed}): "
                           f"[{z.min()}, {z.max()}]")
    full = alive.all(axis=1)
    zr = np.column_stack([out["release_z"], out["z"]])[full]
    if not full.any() or np.min(zr.max(axis=1) - zr.min(axis=1)) <= 1.0:
        raise RuntimeError("multi-year case: a particle alive through the "
                           "run moved <= 1 m vertically")


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------

def run_case(name: str, workdir=None) -> Dict[str, np.ndarray]:
    """Run one case; every value is a numpy array (no object arrays).

    ``workdir`` holds the scratch files of a run-level case (removed after
    the case); None leaves them to ``tempfile``'s default (``$TMPDIR``)."""
    global _WORKDIR
    saved = _WORKDIR
    if workdir is not None:
        Path(workdir).mkdir(parents=True, exist_ok=True)
    _WORKDIR = None if workdir is None else Path(workdir)
    try:
        out = CASES[name]()
    finally:
        _WORKDIR = saved
    res = {}
    for k, v in out.items():
        a = np.asarray(v)
        if a.dtype == object:
            raise TypeError(f"{name}: {k} is an object array")
        res[k] = a
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", required=True, metavar="DIR",
                    help="directory for <case>.npz")
    ap.add_argument("--case", action="append", choices=sorted(CASES),
                    help="case to run (repeatable; default: all)")
    ap.add_argument("--workdir", metavar="DIR", default=None,
                    help="parent of the scratch files (default: $TMPDIR)")
    args = ap.parse_args(argv)
    outdir = Path(args.write)
    outdir.mkdir(parents=True, exist_ok=True)
    for name in args.case or sorted(CASES):
        t0 = time.perf_counter()
        res = run_case(name, workdir=args.workdir)
        path = outdir / f"{name}.npz"
        np.savez_compressed(path, **res)
        print(f"{name:40s} {time.perf_counter() - t0:7.2f} s "
              f"{path.stat().st_size / 1024:7.1f} kB")
    write_env_fingerprint(outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
