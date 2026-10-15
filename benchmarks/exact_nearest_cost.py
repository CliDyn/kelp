#!/usr/bin/env python3
"""Cost of ``mesh.exact_nearest`` inside the fused kernels (docs/performance.md numbers).

One fused step, guard off vs on, next to the unfused path under the guard, on synthetic
meshes chosen to span the anchor geometry: every anchor provable (mid-latitude, automatic
cell), half provable (80–88 N, automatic cell), a cell far below the node spacing, a HEALPix
polar band (89–89.9 N: the cap of every anchor reaches the pole), and seeds far outside a
regional grid (the whole grid is scanned once, the particle dies).  Run on an otherwise idle
node: ``NUMBA_NUM_THREADS=8 python benchmarks/exact_nearest_cost.py [--particles N]``.
"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from kelp.core.io import VelocityField
from kelp.core.mesh import find_nearest_nodes, set_exact_nearest, set_grid_hash_cell, _grid_hash_nearest_batch
from kelp.core.particles import make_state
from kelp.readers.structured import build_structured_mesh
import kelp.numerics.integrators as I

ap = argparse.ArgumentParser(); ap.add_argument("--particles", type=int, default=400_000)
ap.add_argument("--nside", type=int, default=512); ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--cell-rule", choices=("legacy", "occupancy"), default="legacy",
                help="grid-hash cell: the reader's auto cell (legacy) or the run's occupancy rule, applied in "
                     "exact mode before the timings (meshes built with an explicit cell keep it)")
a = ap.parse_args(); P = a.particles


def structured(lo0, lo1, la0, la1, step, cell=None):
    lon1d = np.arange(lo0, lo1, step); lat1d = np.arange(la0, la1, step)
    lo, la = np.meshgrid(lon1d, lat1d)
    m = build_structured_mesh(lo, la, np.ones_like(lo, bool))
    m.depth = np.full(m.n_nodes, 300.0); m.sigma = np.array([])
    if cell:
        set_grid_hash_cell(m, cell)
    return m


def healpix(nside, land_cap_deg=0.0):
    """All-wet dual mesh, or one with a circular land cap of the given radius at (20 E, 40 N)."""
    import healpy
    from kelp.readers.healpix import build_healpix_dual_mesh
    npix = 12 * nside * nside
    wet = np.ones(npix, bool)
    if land_cap_deg > 0:
        vec = healpy.ang2vec(20.0, 40.0, lonlat=True)
        wet[healpy.query_disc(nside, vec, np.radians(land_cap_deg), nest=True)] = False
    return build_healpix_dual_mesh(nside, wet_mask=wet, depth=np.full(npix, 3000.0),
                                   z_levels=np.array([0.0, 50.0, 500.0, 4000.0]))


def best_of(fn, n):
    fn(); t = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
    return 1e3 * min(t)


def provable_share(m, lon, lat):
    out = np.empty(lon.size, np.int32)
    _grid_hash_nearest_batch(lon.size, lon, lat, m._gh_offsets, m._gh_node_ids, m._gh_cell_size, m._gh_lon_min,
                             m._gh_lat_min, m._gh_nx, m._gh_ny, m._gh_wrap_lon, m.nodes_lon, m.nodes_lat, m._node_xyz, out, False,
                             int(m._gh_far_rows))
    from kelp.core.mesh import _anchor_provable
    ok = np.array([o >= 0 and _anchor_provable(float(lo), float(la), int(o), m._gh_cell_size, m._gh_lon_min, m._gh_lat_min,
                                               m._gh_nx, m._gh_ny, m._gh_wrap_lon, m.nodes_lon, m.nodes_lat)
                   for lo, la, o in zip(lon[:20000], lat[:20000], out[:20000])])
    return 100.0 * ok.mean()


def run(name, m, lon, lat, layout, scheme):
    rng = np.random.default_rng(0)
    if layout == "element":
        vel = VelocityField(u=0.2 * rng.standard_normal(m.n_elements), v=0.2 * rng.standard_normal(m.n_elements),
                            eta=np.zeros(m.n_nodes), w=None, uv_layout="element"); vb = None
    else:
        vel = VelocityField(u=0.2 * rng.standard_normal(m.n_nodes), v=0.2 * rng.standard_normal(m.n_nodes),
                            eta=np.zeros(m.n_nodes), w=None, uv_layout="node")
        vb = VelocityField(u=0.2 * rng.standard_normal(m.n_nodes), v=0.2 * rng.standard_normal(m.n_nodes),
                           eta=np.zeros(m.n_nodes), w=None, uv_layout="node")
    st = make_state(lon, lat).replace(failure_reason=np.zeros(lon.size, np.int8))
    if a.cell_rule == "occupancy":
        from kelp.core.mesh import apply_grid_hash_cell_rule, describe_grid_hash
        set_exact_nearest(m, True)                 # the run applies the rule in the guard's mode
        apply_grid_hash_cell_rule(m, "occupancy")
        print(f"\n  {describe_grid_hash(m)}", flush=True)
    print(f"\n{name}: nodes {m.n_nodes}, cell {m._gh_cell_size:.3g}, provable 3x3 anchors {provable_share(m, lon, lat):.1f} %", flush=True)
    res = {}
    for guard in (False, True):
        set_exact_nearest(m, guard)
        s = st.replace(nearest_node=find_nearest_nodes(m, lon, lat))
        if layout == "element":
            kern = I._fused_rk2_step_2d if scheme == "rk2" else I._fused_rk4_step_2d
            unf = I._rk2_step_unfused if scheme == "rk2" else I._rk4_step_unfused
            fu = best_of(lambda: I._run_fused_2d(s, vel, m, 600.0, 1.0, 0, kern), a.reps)
            un = best_of(lambda: unf(s, vel, m, 600.0, 1.0, 0, False, uv_interp="leastsq"), a.reps)
        else:
            if scheme == "rk2":
                kern = I._fused_rk2_step_2d_bary_tinterp; cc = (0.0, 0.5); unf = I._rk2_step_unfused
            else:
                kern = I._fused_rk4_step_2d_bary_tinterp; cc = (0.0, 0.5, 0.5, 1.0); unf = I._rk4_step_unfused
            fu = best_of(lambda: I._run_fused_2d_bary_tinterp(s, vel, vb, 2400.0, m, 600.0, 1.0, 0, kern, cc), a.reps)
            un = best_of(lambda: unf(s, vel, m, 600.0, 1.0, 0, False, vel_next=vb, dt_to_next=2400.0,
                                     uv_interp="barycentric"), a.reps)
        res[guard] = (fu, un)
        set_exact_nearest(m, False)
    (f0, u0), (f1, u1) = res[False], res[True]
    print(f"  fused: off {f0:7.1f} ms  on {f1:7.1f} ms (x{f1 / f0:.2f})   unfused under the guard {u1:7.1f} ms  (fused/unfused x{u1 / f1:.2f})", flush=True)


r = np.random.default_rng(1)
m = structured(0, 40, 30, 50, 0.25)
run("structured mid-lat, auto cell, RK4 leastsq", m, r.uniform(1, 39, P), r.uniform(31, 49, P), "element", "rk4")
m = structured(0, 40, 80, 88, 0.5)
run("structured 80-88 N, auto cell, RK4 leastsq", m, r.uniform(1, 39, P), r.uniform(80.2, 87.8, P), "element", "rk4")
m = structured(0, 40, 80, 88, 0.5, cell=0.3)
run("structured 80-88 N, cell 0.3 (3x below spacing), RK4 leastsq", m, r.uniform(1, 39, P), r.uniform(80.2, 87.8, P), "element", "rk4")
m = structured(0, 20, 30, 50, 0.1, cell=0.05)
run("regional 20x20 deg, cell 0.05, seeds FAR OUTSIDE (die this step), RK2 leastsq", m,
    r.uniform(150, 170, P), r.uniform(-40, -20, P), "element", "rk2")
try:
    hp = healpix(a.nside)
    run(f"HEALPix nside {a.nside} (global wrap grid), 80-88 N, RK4 barycentric tinterp", hp,
        r.uniform(-180, 180, P), r.uniform(80, 88, P), "node", "rk4")
    run(f"HEALPix nside {a.nside}, POLAR BAND 89-89.9 N, RK4 barycentric tinterp", hp,
        r.uniform(-180, 180, P), r.uniform(89.0, 89.9, P), "node", "rk4")
    run(f"HEALPix nside {a.nside}, global uniform seeds, RK4 barycentric tinterp", hp,
        r.uniform(-180, 180, P), np.degrees(np.arcsin(r.uniform(-1, 1, P))), "node", "rk4")
    # 3-D RK2 barycentric (its own fused kernel; stage anchors in-kernel under the guard since round 4)
    K = hp.sigma.size; N3 = hp.n_nodes; rng3 = np.random.default_rng(9)
    v3 = VelocityField(u=0.2 * rng3.standard_normal((K - 1, N3)), v=0.2 * rng3.standard_normal((K - 1, N3)),
                       eta=np.zeros(N3), w=1e-4 * rng3.standard_normal((K, N3)), uv_layout="node")
    lon3 = r.uniform(-180, 180, P); lat3 = r.uniform(80, 88, P)
    st3 = make_state(lon3, lat3, r.uniform(10.0, 400.0, P)).replace(failure_reason=np.zeros(P, np.int8))
    res = {}
    for guard in (False, True):
        set_exact_nearest(hp, guard)
        s3 = st3.replace(nearest_node=find_nearest_nodes(hp, lon3, lat3))
        fu = best_of(lambda: I._run_fused_rk2_3d_bary_tinterp(s3, v3, None, None, hp, 600.0, 1.0, I.W_BARYCENTRIC), a.reps)
        un = best_of(lambda: I._rk2_step_unfused(s3, v3, hp, 600.0, 1.0, 0, True, w_interp="barycentric",
                                                 uv_interp="barycentric"), a.reps)
        res[guard] = (fu, un)
        set_exact_nearest(hp, False)
    (f0, u0), (f1, u1) = res[False], res[True]
    print(f"\nHEALPix nside {a.nside}, 80-88 N, 3-D RK2 barycentric (fused 3-D kernel): nodes {N3}, K {K}", flush=True)
    print(f"  fused: off {f0:7.1f} ms  on {f1:7.1f} ms (x{f1 / f0:.2f})   unfused under the guard {u1:7.1f} ms  (fused/unfused x{u1 / f1:.2f})", flush=True)
    del hp
    hp = healpix(a.nside, land_cap_deg=20.0)
    run(f"HEALPix nside {a.nside} with a 20-deg LAND CAP, seeds INSIDE the cap (die this step), RK2 barycentric tinterp", hp,
        20.0 + r.uniform(-12, 12, P), 40.0 + r.uniform(-12, 12, P), "node", "rk2")
except ImportError:
    print("healpy not available: HEALPix cases skipped")
