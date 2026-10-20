#!/usr/bin/env python3
"""Step cost of depth-linear u/v (mode.uv_vertical: linear), 3-D steps.

One 3-D step timed with the SAME u/v arrays read two ways: ``"layer"``
(piecewise constant in depth) and ``"layer_linear"`` (linear in depth
between the layer centres, a stencil member dry at the particle depth a
wall).  Two legs:

- node layout (``--layout node``, the default): one fused node-barycentric
  3-D RK2 step (``_fused_rk2_step_3d_bary_tinterp`` against its twin
  ``_fused_rk2_step_3d_bary_tinterp_dl``; the PCHIP twins with
  ``--payload pchip``) on a synthetic HEALPix dual mesh;
- element layout (``--layout element``): element ``leastsq`` u/v on the
  same mesh with every quad split into two triangles (a FESOM2-like
  triangular mesh) and FESOM2-like element bottoms, either one fused 3-D
  Euler step (``--integrator euler``: ``_fused_euler_step_3d`` against its
  twin ``_fused_euler_step_3d_dl``) or one unfused 3-D RK2 step
  (``--integrator rk2``: ``_rk2_step_unfused`` with the vectorised
  ``interp_leastsq``, layer core against ``_interp_leastsq_core_dl``).

The payload is the one the tracking layer builds between two records (a
lazy blend at the step start and the next record); ``--payload pchip``
adds the 4-record PCHIP bundle (RK2 only).  Self-contained: varied
bathymetry (so the last usable point varies from column to column), random
u/v/w, particles at random positions.

Method: the thread count is fixed before Numba loads; one warm-up step per
placement (compilation, the once-per-field record check, a check that the
expected kernel or step path ran); then ``--steps`` timed steps per
placement, interleaved (layer, linear, layer, ...) so slow drifts of the
machine hit both; the median of each placement and their ratio are
reported with the machine, the commit and the thread count.  Each timed
step starts from the same state (the same work every time) and builds a
new lazy blend from the cached, already-checked records, as the tracking
layer does every step.

Usage (an otherwise idle node)::

    python benchmarks/single_node/depth_linear_cost.py \\
        [--layout node|element] [--integrator rk2|euler] \\
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


def _elem_mesh(nside):
    """The dual mesh with every quad split along its 0-2 diagonal: a
    triangular mesh (about six triangles per node, as FESOM2's) with
    FESOM2-like element bottoms.

    An element's bed is the first level at or below the mean bathymetry of
    its three vertices (``nlevels`` of the mesh diagnostic, at most the
    last level), a node's depth the deepest bed of its elements (FESOM2's
    node levels: the maximum over the elements); the element declaration
    comes from the FESOM2 reader's own rule (``_elem_level_metadata`` on
    that ``nlevels``)."""
    import numpy as np
    from kelp.core.mesh import Mesh, build_node_to_elems, build_search
    from kelp.readers.fesom2 import _elem_level_metadata
    dual = _mesh(nside)
    en = np.asarray(dual.elem_nodes)
    is_tri = en[:, 3] < 0
    quads = en[~is_tri]
    tri = np.ascontiguousarray(np.concatenate(
        [en[is_tri, :3], quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]]),
        dtype=np.int32)
    lon = dual.nodes_lon
    lat = dual.nodes_lat
    # element centres: the normalized mean of the vertex unit vectors
    lr, pr = np.deg2rad(lon), np.deg2rad(lat)
    xyz = np.stack((np.cos(pr) * np.cos(lr), np.cos(pr) * np.sin(lr),
                    np.sin(pr)), axis=1)
    c = xyz[tri].sum(axis=1)
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    c_lon = np.ascontiguousarray(np.degrees(np.arctan2(c[:, 1], c[:, 0])))
    c_lat = np.ascontiguousarray(np.degrees(np.arcsin(np.clip(c[:, 2],
                                                              -1.0, 1.0))))
    z = np.asarray(K_INTERFACES, dtype=np.float64)
    mean_d = np.asarray(dual.depth)[tri].mean(axis=1)
    nlevels = np.minimum(np.searchsorted(z, mean_d, side="left"),
                         z.size - 1) + 1
    meta = _elem_level_metadata(
        lambda name: nlevels if name == "nlevels" else None, z, tri.shape[0])
    node_depth = np.zeros(lon.size)
    np.maximum.at(node_depth, tri.ravel(),
                  np.repeat(meta["elem_bed"], 3))
    n2e, cnt = build_node_to_elems(lon.size, tri)
    mesh = Mesh(name="healpix-tri", nodes_lon=lon, nodes_lat=lat,
                depth=node_depth, elem_nodes=tri, elem_center_lon=c_lon,
                elem_center_lat=c_lat, node_to_elems=n2e,
                node_to_elems_count=cnt, sigma=z.copy(),
                vertical_coord="zlevel", metadata=dict(meta))
    build_search(mesh)
    return mesh


def _records(mesh, n, seed, layout="node"):
    """``n`` layer records (u/v on the K-1 layers, zero below each
    column's last usable layer centre, w on the K levels) and their
    layer_linear twins over the SAME arrays."""
    import dataclasses
    import numpy as np
    from kelp.core.io import VelocityField
    from kelp.numerics.vertical_uv import depth_points
    k = len(K_INTERFACES)
    nn = mesh.n_nodes
    nc = nn if layout == "node" else mesh.n_elements
    lay, lin = [], []
    for r in range(n):
        rng = np.random.default_rng(seed + r)
        u = 0.3 * rng.standard_normal((k - 1, nc))
        v = 0.3 * rng.standard_normal((k - 1, nc))
        w = 2e-4 * rng.standard_normal((k, nn))
        f = VelocityField(u=u, v=v, eta=np.zeros(nn), w=w, uv_layout=layout)
        if r == 0:
            dp = depth_points(mesh, "layer_linear", layout)
            below = np.arange(k - 1)[:, None] > dp.last[None, :]
        # dry slots as a reader leaves them for layer runs (zeros) — the
        # depth-linear read never touches them
        u[below] = 0.0
        v[below] = 0.0
        lay.append(f)
        lin.append(dataclasses.replace(f, uv_vertical="layer_linear"))
    return lay, lin


def _state(mesh, n, seed, in_column=False):
    """Particles at random positions (|lat| < 76 deg); depths uniform in
    0-3000 m, or (``in_column``) uniform between the surface and the
    nearest node's depth, where the end-of-step clamp keeps them."""
    import numpy as np
    from kelp.core.mesh import find_nearest_nodes
    from kelp.core.particles import make_state
    rng = np.random.default_rng(seed)
    lon = rng.uniform(-180.0, 180.0, n)
    lat = np.degrees(np.arcsin(rng.uniform(-0.97, 0.97, n)))
    z = rng.uniform(0.0, 3000.0, n)
    nn = find_nearest_nodes(mesh, lon, lat)
    if in_column:
        z = (z / 3000.0) * np.asarray(mesh.depth)[nn]
    st = make_state(lon, lat, z)
    return st.replace(failure_reason=np.zeros(n, dtype=np.int8),
                      nearest_node=nn)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--layout", choices=("node", "element"), default="node",
                    help="node: barycentric u/v on the HEALPix dual mesh "
                         "(the default); element: leastsq u/v on its "
                         "triangulated twin")
    ap.add_argument("--integrator", choices=("rk2", "euler"), default="rk2",
                    help="rk2 (the default; node layout: the fused kernel, "
                         "element layout: the unfused step) or euler "
                         "(element layout only: the fused 3-D kernel)")
    ap.add_argument("--nside", type=int, default=64)
    ap.add_argument("--particles", type=int, default=100_000)
    ap.add_argument("--steps", type=int, default=15,
                    help="timed steps per placement (median reported)")
    ap.add_argument("--threads", type=int, default=None,
                    help="numba threads (default: NUMBA_NUM_THREADS, else "
                         "the CPU count)")
    ap.add_argument("--payload", choices=("lerp", "pchip"), default="lerp",
                    help="lerp: two records blended linearly in time (the "
                         "default); pchip: the 4-record PCHIP bundle (rk2 "
                         "only)")
    args = ap.parse_args(argv)
    if args.layout == "node" and args.integrator != "rk2":
        ap.error("--layout node measures the fused 3-D RK2 kernel: "
                 "--integrator rk2 only")
    if args.integrator == "euler" and args.payload != "lerp":
        ap.error("--integrator euler reads the step-start field only: "
                 "--payload lerp only")

    threads = args.threads or int(os.environ.get("NUMBA_NUM_THREADS", 0)) \
        or (os.cpu_count() or 4)
    os.environ["NUMBA_NUM_THREADS"] = str(threads)   # before numba loads
    sys.path.insert(0, str(REPO / "src"))
    import numba
    import numpy as np
    import kelp.numerics.integrators as I
    import kelp.numerics.interpolate as IP
    from kelp.core.io import LazyLerpField
    numba.set_num_threads(threads)

    element = args.layout == "element"
    t_build = time.perf_counter()
    if element:
        mesh = _elem_mesh(args.nside)
        lay, lin = _records(mesh, 4, seed=11, layout="element")
        st = _state(mesh, args.particles, seed=5, in_column=True)
    else:
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

    # the functions a warm-up step must reach: (module, name) per placement
    if not element:
        route = "fused 3-D RK2, node barycentric"
        want = {"layer": [(I, ("_fused_rk2_step_3d_bary_pchip"
                               if args.payload == "pchip"
                               else "_fused_rk2_step_3d_bary_tinterp"))],
                "layer_linear": [(I, ("_fused_rk2_step_3d_bary_pchip_dl"
                                      if args.payload == "pchip"
                                      else "_fused_rk2_step_3d_bary_"
                                           "tinterp_dl"))]}

        def step(recs):
            vel, kw = payload(recs)
            return I.rk2_step(st, vel, mesh, DT, 1.0,
                              uv_interp="barycentric",
                              w_interp="barycentric", **kw)
    elif args.integrator == "euler":
        route = "fused 3-D Euler, element leastsq"
        want = {"layer": [(I, "_fused_euler_step_3d")],
                "layer_linear": [(I, "_fused_euler_step_3d_dl")]}

        def step(recs):
            vel, kw = payload(recs)
            return I.euler_step(st, vel, mesh, DT, 1.0, uv_interp="leastsq",
                                w_interp="leastsq", **kw)
    else:
        route = "unfused 3-D RK2, element leastsq"
        want = {"layer": [(I, "_rk2_step_unfused"),
                          (IP, "_loop_interp_leastsq_3d")],
                "layer_linear": [(I, "_rk2_step_unfused"),
                                 (IP, "_loop_interp_leastsq_3d_dl")]}

        def step(recs):
            vel, kw = payload(recs)
            return I.rk2_step(st, vel, mesh, DT, 1.0, uv_interp="leastsq",
                              w_interp="leastsq", **kw)

    # warm-up: compile, check each record once, and prove the route
    alive = {}
    for name, recs in (("layer", lay), ("layer_linear", lin)):
        hits = {}
        saved = []
        for mod, fn in want[name]:
            orig = getattr(mod, fn)
            saved.append((mod, fn, orig))

            def spy(*a, _orig=orig, _fn=fn, **k):
                hits[_fn] = hits.get(_fn, 0) + 1
                return _orig(*a, **k)
            setattr(mod, fn, spy)
        try:
            out = step(recs)
        finally:
            for mod, fn, orig in saved:
                setattr(mod, fn, orig)
        missing = [fn for _, fn in want[name] if not hits.get(fn)]
        if missing:
            sys.exit(f"{name}: not reached: {', '.join(missing)}")
        alive[name] = int((out.status == 1).sum())

    timings = {"layer": [], "layer_linear": []}
    for _ in range(args.steps):
        for name, recs in (("layer", lay), ("layer_linear", lin)):
            t0 = time.perf_counter()
            step(recs)
            timings[name].append(time.perf_counter() - t0)
    med = {k: float(np.median(v)) for k, v in timings.items()}

    if element:
        geom = (f"HEALPix dual nside {args.nside} split into triangles, "
                f"{mesh.n_nodes:,} nodes, {mesh.n_elements:,} elements")
        uvw = "element leastsq u/v, leastsq W"
        depths = "particle depths within the nearest node's column"
    else:
        geom = f"HEALPix dual nside {args.nside}, {mesh.n_nodes:,} nodes"
        uvw = "barycentric u/v and W"
        depths = "particle depths 0-3000 m"
    print(f"============ depth-linear u/v: {route} step ============")
    print(f"machine    : {platform.platform()}")
    print(f"cpu        : {platform.processor() or platform.machine()}  "
          f"(os.cpu_count {os.cpu_count()})")
    print(f"python     : {platform.python_version()}  numpy "
          f"{np.__version__}  numba {numba.__version__}")
    print(f"commit     : {_commit()}")
    print(f"threads    : {threads} (numba.get_num_threads "
          f"{numba.get_num_threads()})")
    print(f"workload   : {geom}, {len(K_INTERFACES)} levels, "
          f"{args.particles:,} particles, {args.integrator} {args.payload} "
          f"(alpha0 {ALPHA0}), {uvw}, dt {DT:g} s, {depths}")
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
    print("=" * (36 + len(route) + 7))
    return 0


if __name__ == "__main__":
    sys.exit(main())
