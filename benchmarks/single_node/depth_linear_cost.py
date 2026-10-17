#!/usr/bin/env python3
"""Step cost of depth-linear u/v (mode.uv_vertical: linear), fused 3-D RK2.

One fused node-barycentric 3-D RK2 step with a linear time blend (the payload
the tracking layer builds between two records: a lazy blend at the step
start and the next record), timed with the SAME u/v arrays read two ways:
``"layer"`` (piecewise constant in depth, ``_fused_rk2_step_3d_bary_tinterp``)
and ``"layer_linear"`` (linear in depth between the layer centres,
``_fused_rk2_step_3d_bary_tinterp_dl``).  Self-contained: a synthetic
HEALPix dual mesh with varied bathymetry (so the last usable point varies
from node to node), random u/v/w, particles at random positions and depths.

Method: the thread count is fixed before Numba loads; one warm-up step per
placement (compilation, the once-per-field record check); then ``--steps``
timed steps per placement, interleaved (layer, linear, layer, ...) so slow
drifts of the machine hit both; the median of each placement and their
ratio are reported with the machine, the commit and the thread count.  Each
timed step starts from the same state (the same work every time) and builds
a new lazy blend from the cached, already-checked records, as the tracking
layer does every step.

Usage (an otherwise idle node)::

    python benchmarks/single_node/depth_linear_cost.py \\
        [--nside 64] [--particles 100000] [--steps 15] [--threads N]
        [--payload lerp|pchip]

The method and the measured rows:
``benchmarks/single_node/depth_linear_cost.md``.
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: the level table of ``run_bench.py`` (21 layers down to 4000 m)
K_INTERFACES = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0,
                200.0, 300.0, 400.0, 500.0, 700.0, 900.0, 1200.0,
                1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0]
DT = 21600.0          # s, the step of run_bench.py
RECORD_DT = 86400.0   # s between the records
ALPHA0 = 0.25         # the step starts a quarter into the bracket


def _commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short",
                              "HEAD"], capture_output=True, text=True,
                             timeout=10)
        sha = out.stdout.strip() or "unknown"
        dirty = subprocess.run(["git", "-C", str(REPO), "status",
                                "--porcelain", "--", "src"],
                               capture_output=True, text=True, timeout=10)
        return sha + ("+dirty-src" if dirty.stdout.strip() else "")
    except Exception:
        return "unknown"


def _mesh(nside):
    import healpy
    import numpy as np
    from kelp.readers.healpix import build_healpix_dual_mesh
    npix = 12 * nside * nside
    lon, lat = healpy.pix2ang(nside, np.arange(npix), nest=True, lonlat=True)
    depth = 60.0 + 3500.0 * np.abs(np.sin(np.deg2rad(lat * 2.0)))
    depth[::17] = 35.0                       # shallow columns everywhere
    return build_healpix_dual_mesh(nside, wet_mask=np.ones(npix, bool),
                                   depth=depth,
                                   z_levels=np.asarray(K_INTERFACES))


def _records(mesh, n, seed):
    """``n`` layer records (u/v on the K-1 layers, NaN below each node's
    last usable layer centre, w on the K levels) and their layer_linear
    twins over the SAME arrays."""
    import dataclasses
    import numpy as np
    from kelp.core.io import VelocityField
    from kelp.numerics.vertical_uv import depth_points
    k = len(K_INTERFACES)
    nn = mesh.n_nodes
    lay, lin = [], []
    for r in range(n):
        rng = np.random.default_rng(seed + r)
        u = 0.3 * rng.standard_normal((k - 1, nn))
        v = 0.3 * rng.standard_normal((k - 1, nn))
        w = 2e-4 * rng.standard_normal((k, nn))
        f = VelocityField(u=u, v=v, eta=np.zeros(nn), w=w, uv_layout="node")
        if r == 0:
            dp = depth_points(mesh, "layer_linear")
            below = np.arange(k - 1)[:, None] > dp.last[None, :]
        # dry slots as a reader leaves them for layer runs (zeros) — the
        # depth-linear read never touches them
        u[below] = 0.0
        v[below] = 0.0
        lay.append(f)
        lin.append(dataclasses.replace(f, uv_vertical="layer_linear"))
    return lay, lin


def _state(mesh, n, seed):
    import numpy as np
    from kelp.core.mesh import find_nearest_nodes
    from kelp.core.particles import make_state
    rng = np.random.default_rng(seed)
    lon = rng.uniform(-180.0, 180.0, n)
    lat = np.degrees(np.arcsin(rng.uniform(-0.97, 0.97, n)))
    z = rng.uniform(0.0, 3000.0, n)
    st = make_state(lon, lat, z)
    return st.replace(failure_reason=np.zeros(n, dtype=np.int8),
                      nearest_node=find_nearest_nodes(mesh, lon, lat))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nside", type=int, default=64)
    ap.add_argument("--particles", type=int, default=100_000)
    ap.add_argument("--steps", type=int, default=15,
                    help="timed steps per placement (median reported)")
    ap.add_argument("--threads", type=int, default=None,
                    help="numba threads (default: NUMBA_NUM_THREADS, else "
                         "the CPU count)")
    ap.add_argument("--payload", choices=("lerp", "pchip"), default="lerp",
                    help="lerp: two records blended linearly in time (the "
                         "default); pchip: the 4-record PCHIP twins")
    args = ap.parse_args(argv)

    threads = args.threads or int(os.environ.get("NUMBA_NUM_THREADS", 0)) \
        or (os.cpu_count() or 4)
    os.environ["NUMBA_NUM_THREADS"] = str(threads)   # before numba loads
    sys.path.insert(0, str(REPO / "src"))
    import numba
    import numpy as np
    import kelp.numerics.integrators as I
    from kelp.core.io import LazyLerpField
    numba.set_num_threads(threads)

    t_build = time.perf_counter()
    mesh = _mesh(args.nside)
    lay, lin = _records(mesh, 4, seed=11)
    st = _state(mesh, args.particles, seed=5)
    build_s = time.perf_counter() - t_build
    times = [i * RECORD_DT for i in range(4)]
    q = times[1] + ALPHA0 * RECORD_DT

    def payload(recs):
        """(vel, kwargs) of one step, built like the tracking layer: a new
        lazy blend at the step start from the cached records."""
        vel = LazyLerpField(recs[1], recs[2], ALPHA0)
        kw = dict(vel_next=recs[2], dt_to_next=(1.0 - ALPHA0) * RECORD_DT)
        if args.payload == "pchip":
            kw["recon"] = (recs, times, q, 1)
        return vel, kw

    kernels = {"layer": ("_fused_rk2_step_3d_bary_tinterp",
                         "_fused_rk2_step_3d_bary_pchip"),
               "layer_linear": ("_fused_rk2_step_3d_bary_tinterp_dl",
                                "_fused_rk2_step_3d_bary_pchip_dl")}

    def step(recs):
        vel, kw = payload(recs)
        return I.rk2_step(st, vel, mesh, DT, 1.0, uv_interp="barycentric",
                          w_interp="barycentric", **kw)

    # warm-up: compile, check each record once, and prove the route
    alive = {}
    for name, recs in (("layer", lay), ("layer_linear", lin)):
        want = kernels[name][1 if args.payload == "pchip" else 0]
        orig = getattr(I, want)
        hits = []

        def spy(*a, _orig=orig, **k):
            hits.append(1)
            return _orig(*a, **k)
        setattr(I, want, spy)
        try:
            out = step(recs)
        finally:
            setattr(I, want, orig)
        if not hits:
            sys.exit(f"{name}: the fused kernel {want} was not launched")
        alive[name] = int((out.status == 1).sum())

    timings = {"layer": [], "layer_linear": []}
    for _ in range(args.steps):
        for name, recs in (("layer", lay), ("layer_linear", lin)):
            t0 = time.perf_counter()
            step(recs)
            timings[name].append(time.perf_counter() - t0)
    med = {k: float(np.median(v)) for k, v in timings.items()}

    print("============ depth-linear u/v: fused 3-D RK2 step ============")
    print(f"machine    : {platform.platform()}")
    print(f"cpu        : {platform.processor() or platform.machine()}  "
          f"(os.cpu_count {os.cpu_count()})")
    print(f"python     : {platform.python_version()}  numpy "
          f"{np.__version__}  numba {numba.__version__}")
    print(f"commit     : {_commit()}")
    print(f"threads    : {threads} (numba.get_num_threads "
          f"{numba.get_num_threads()})")
    print(f"workload   : HEALPix dual nside {args.nside}, {mesh.n_nodes:,} "
          f"nodes, {len(K_INTERFACES)} levels, {args.particles:,} "
          f"particles, rk2 {args.payload} (alpha0 {ALPHA0}), "
          f"barycentric u/v and W, dt {DT:g} s")
    print(f"alive after the step: layer {alive['layer']:,}, "
          f"layer_linear {alive['layer_linear']:,}")
    print(f"timed steps: {args.steps} per placement, interleaved, median")
    for name in ("layer", "layer_linear"):
        v = np.asarray(timings[name]) * 1e3
        print(f"  {name:13s}: median {med[name] * 1e3:9.3f} ms/step   "
              f"(min {v.min():.3f}, max {v.max():.3f})")
    print(f"  ratio layer_linear / layer: "
          f"{med['layer_linear'] / med['layer']:.3f}")
    print(f"setup (mesh, fields, seeds): {build_s:.1f} s")
    print("==============================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
