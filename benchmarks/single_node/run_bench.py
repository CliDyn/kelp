#!/usr/bin/env python3
"""Portable single-node tracking benchmark (cross-machine comparison).

Self-contained: builds a synthetic HEALPix 3-D velocity store (no
external data), seeds particles at every wet pixel on two depth
horizons, runs the SAME fused RK2 + temporal-interpolation +
barycentric configuration the reference E2 scaling experiment uses, and reports the
same metric — the steady tracking-phase step time (mean of the final
five steps) plus thread-microseconds per particle — alongside the
machine description. Run it on any machine (Apple-silicon Mac,
laptop, HPC node) to put that machine on the same comparison axis.

The synthetic field is a rotated-frame zonal flow (constant 0.3 m/s in
the frame whose poles sit on the dry polar caps) with a small vertical
velocity and ±5 % temporal modulation, on a wet mask |rotated lat| <=
80 deg — dynamically boring on purpose: the benchmark measures the
ENGINE (gathers, stencils, time blends), not the ocean.

Usage:
    python benchmarks/single_node/run_bench.py [--zoom 8] [--steps 8]
        [--threads N] [--workdir DIR] [--keep]

zoom 8 -> 786k pixels, ~590k wet nodes, ~1.2M particles, ~1.1 GB
store, a few minutes end to end. Use --zoom 7 for a quick check.
NUMBA_NUM_THREADS (or --threads) controls the thread count; on Apple
silicon set it to the PERFORMANCE core count for the headline number.
"""
from __future__ import annotations

import argparse
import io
import os
import platform
import re
import shutil
import sys
import tempfile
import time
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

K_INTERFACES = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0,
                200.0, 300.0, 400.0, 500.0, 700.0, 900.0, 1200.0,
                1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0]
U_ROT = 0.3          # m/s, zonal in the rotated frame
W_AMP = 1.0e-5       # m/s
WET_CAP_DEG = 80.0   # |rotated lat| above this = dry (pole guard)
N_SNAPSHOTS = 3      # covers 8 x 6h steps with a temporal bracket


def build_store(store: Path, zoom: int) -> None:
    import healpy as hp
    import numpy as np
    import zarr
    from kelp.core.rotation import (FESOM2_ANGLES, euler_matrix,
                                        rotate_lonlat, unrotate_vectors)

    nside = 1 << zoom
    npix = hp.nside2npix(nside)
    lon, lat = hp.pix2ang(nside, np.arange(npix), nest=True, lonlat=True)
    R = euler_matrix(*FESOM2_ANGLES)
    _, rlat = rotate_lonlat(lon, lat, R)
    wet = np.abs(rlat) <= WET_CAP_DEG
    # geographic image of a constant rotated-frame zonal flow
    u_geo, v_geo = unrotate_vectors(np.full(npix, U_ROT),
                                    np.zeros(npix), lon, lat, R)
    k = len(K_INTERFACES)
    root = zarr.open_group(str(store), mode="w")
    t = root.create_array("time", shape=(N_SNAPSHOTS,), dtype="int64",
                          dimension_names=("time",))
    t[:] = np.arange(N_SNAPSHOTS) * 86_400
    t.attrs.update({"units": "seconds since 2020-01-01",
                    "calendar": "proleptic_gregorian",
                    "_ARRAY_DIMENSIONS": ["time"]})
    lv = root.create_array("level", shape=(k,), dtype="int64",
                           dimension_names=("level",))
    lv[:] = np.arange(1, k + 1)
    lv.attrs["_ARRAY_DIMENSIONS"] = ["level"]
    li = root.create_array("level_interfaces_m", shape=(k,),
                           dtype="float64",
                           dimension_names=("level_interface",))
    li[:] = np.asarray(K_INTERFACES)
    li.attrs["_ARRAY_DIMENSIONS"] = ["level_interface"]
    crs = root.create_array("crs", shape=(1,), dtype="float32",
                            dimension_names=("crs",))
    crs[:] = 0.0
    crs.attrs.update({"grid_mapping_name": "healpix",
                      "healpix_nside": int(nside),
                      "healpix_order": "nest",
                      "_ARRAY_DIMENSIONS": ["crs"]})
    # depth-independent u/v (K-1 mids on a K-level axis, last = padding),
    # w on the K interfaces; land = NaN; +-5% temporal modulation
    for name, base in (("uo", u_geo), ("vo", v_geo), ("wo", None)):
        arr = root.create_array(
            name, shape=(N_SNAPSHOTS, k, npix), dtype="float32",
            chunks=(1, k, min(npix, 1 << 20)),
            dimension_names=("time", "level", "value"))
        arr.attrs["_ARRAY_DIMENSIONS"] = ["time", "level", "value"]
        for ti in range(N_SNAPSHOTS):
            mod = 1.0 + 0.05 * ti
            plane = np.full((k, npix), np.nan, dtype=np.float32)
            if base is None:                     # w: K interfaces
                prof = W_AMP * np.sin(
                    np.pi * np.arange(k) / max(k - 1, 1))
                plane[:, wet] = (mod * prof)[:, None].astype(np.float32)
            else:                                # u/v: K-1 mids + pad
                plane[: k - 1, wet] = np.float32(mod) * base[
                    wet].astype(np.float32)[None, :]
            arr[ti] = plane
    root.attrs["title"] = ("kelp synthetic benchmark store "
                           f"(zoom {zoom}) — engine timing only")


def write_seeds(store: Path, seeds: Path, zoom: int) -> int:
    import numpy as np
    from kelp.core.config import MeshConfig, ModeConfig
    from kelp.readers.healpix import HealpixReader

    mesh = HealpixReader.load_mesh(
        MeshConfig(type="healpix", meshpath=str(store),
                   levels=K_INTERFACES),
        mode_cfg=ModeConfig(Fdrift2D=False, Fdrift3D=True))
    lon, lat = mesh.nodes_lon, mesh.nodes_lat
    rows = [np.column_stack([lon, lat, np.full(lon.size, d)])
            for d in (100.0, 500.0)]
    allr = np.vstack(rows)
    np.savetxt(seeds, allr, fmt="%.6f %.6f %.1f")
    return allr.shape[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--zoom", type=int, default=8)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--threads", type=int, default=None,
                    help="numba threads (default: NUMBA_NUM_THREADS or "
                         "physical core count)")
    ap.add_argument("--workdir", default=None,
                    help="where store/seeds/output live (default: a "
                         "temp dir; pass a dir to reuse the store)")
    ap.add_argument("--keep", action="store_true",
                    help="keep the workdir (implied with --workdir)")
    args = ap.parse_args(argv)

    threads = args.threads or int(os.environ.get(
        "NUMBA_NUM_THREADS", 0)) or (os.cpu_count() or 4)
    os.environ["NUMBA_NUM_THREADS"] = str(threads)

    work = Path(args.workdir).resolve() if args.workdir else Path(
        tempfile.mkdtemp(prefix="kelp_bench_",
                         dir=str(REPO / ".tmp")))
    work.mkdir(parents=True, exist_ok=True)
    store = work / f"bench_z{args.zoom}.zarr"
    seeds = work / f"bench_seeds_z{args.zoom}.dat"
    out = work / f"bench_out_z{args.zoom}.zarr"

    t0 = time.time()
    if not (store / "zarr.json").exists() and \
            not (store / ".zgroup").exists():
        print(f"building synthetic store (zoom {args.zoom}) ...",
              flush=True)
        build_store(store, args.zoom)
    if not seeds.exists():
        print("seeding ...", flush=True)
        n = write_seeds(store, seeds, args.zoom)
    else:
        n = sum(1 for _ in open(seeds))
    print(f"store: {store.name}  particles: {n:,}  threads: {threads}",
          flush=True)

    cfg = work / "bench.yaml"
    cfg.write_text(f"""\
mode:
  Fdrift2D: false
  Fdrift3D: true
  time_dir: 1.0
numerics:
  dt: 21600.0
  integrator: rk2
  temporal_interp: true
  uv_interp: barycentric
  w_interp: barycentric
  output_freq: {args.steps}
mesh:
  type: healpix
  meshpath: {store}
  rotation: fesom2
  levels: {K_INTERFACES}
data:
  ncinput: {store}
  inicloud: {seeds}
  output_path: {out}
time:
  start_time_idx: 0
  n_steps: {args.steps}
""")
    from kelp.core.config import load_config
    from kelp.tracking import run_tracking

    shutil.rmtree(out, ignore_errors=True)
    buf = io.StringIO()

    class Tee(io.TextIOBase):
        def write(self, s):
            buf.write(s)
            sys.__stdout__.write(s)
            return len(s)

    with redirect_stdout(Tee()):
        run_tracking(load_config(cfg), output_path=str(out),
                     verbose=True)
    steps = [float(m) / 1000.0 for m in
             re.findall(r"\(io=[\d.]+ step=([\d.]+)", buf.getvalue())]
    if len(steps) < 3:
        sys.exit("benchmark produced too few step timings")
    tail = steps[-5:] if len(steps) >= 5 else steps[1:]
    steady = sum(tail) / len(tail)
    per_particle_us = steady / n * 1e6
    thread_us = per_particle_us * threads

    print("\n================ kelp bench result ================")
    print(f"machine   : {platform.platform()}")
    print(f"cpu       : {platform.processor() or platform.machine()}")
    print(f"python    : {platform.python_version()}  "
          f"threads: {threads}")
    print(f"workload  : zoom {args.zoom}, {n:,} particles, "
          f"{args.steps} steps, rk2+tinterp+barycentric, rotated frame")
    print(f"steady step               : {steady:.3f} s")
    print(f"per-particle              : {per_particle_us:.3f} us")
    print(f"thread-us per particle    : {thread_us:.1f}")
    print("compare machines by running THIS SAME benchmark on each "
          "(same zoom/steps);\nthe reference real-data range "
          "(20-36 thread-us) is a different, heavier workload.")
    print(f"total wall (incl. build)  : {time.time() - t0:.0f} s")
    print("=======================================================")
    if not args.keep and not args.workdir:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
