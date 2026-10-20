# depth_linear_cost — what `mode.uv_vertical: linear` costs per step

The depth-linear horizontal velocity (`docs/vertical.md` §2.1) reads two data
points per stencil member (a vertex, or an element) instead of one, searches
the member's column for the bracket and tests the member against its bed (a
member dry at the particle depth is a wall). `depth_linear_cost.py` measures
what that costs in the hot path: one 3-D step timed with the SAME u/v arrays
read two ways, `"layer"` and `"layer_linear"`. Three legs:

- **node layout** (`--layout node`, the default): one fused node-barycentric
  3-D RK2 step with a linear time blend — `_fused_rk2_step_3d_bary_tinterp`
  against `_fused_rk2_step_3d_bary_tinterp_dl` (`--payload pchip`: the PCHIP
  twins);
- **element layout** (`--layout element`): element `leastsq` u/v, either one
  fused 3-D Euler step (`--integrator euler`: `_fused_euler_step_3d` against
  `_fused_euler_step_3d_dl`) or one unfused 3-D RK2 step (`--integrator rk2`:
  `_rk2_step_unfused` with the vectorised `interp_leastsq`, the layer core
  against `_interp_leastsq_core_dl`; `--payload pchip`: with the 4-record PCHIP
  bundle);
- **sigma** (`--vertical sigma`, `--integrator rk2`, either layout): the same
  horizontal meshes as a FESOM-C-like sigma mesh, one unfused 3-D RK2 step
  (`_rk2_step_unfused`; node `barycentric` or element `leastsq` u/v, the layer
  kernels against their depth-linear twins, which read u/v in the sigma
  fraction `s = z / H_p` of each stage, `_sigma_frac_for_uv`; no fused kernel
  serves sigma meshes in either rule).

## Method

- Synthetic and self-contained: a HEALPix dual mesh (all wet, NSIDE 64 →
  49,152 nodes by default) with varied bathymetry (`60 + 3500·|sin 2φ|` m, every
  17th column 35 m, so each node's last usable layer differs), the 22-level table
  of `run_bench.py` (21 layers down to 4000 m), random u/v (σ = 0.3 m/s) and w
  (σ = 2·10⁻⁴ m/s), zeros in the dry slots below each column's last usable layer
  centre (as a reader leaves them).
- Element leg: the same mesh with every quad split along one diagonal into two
  triangles (98,300 triangles at NSIDE 64, six per node on average, as on a
  FESOM2 mesh), and FESOM2-like bottoms: an element's bed is the first level at
  or below the mean bathymetry of its vertices (its `nlevels`), declared by the
  FESOM2 reader's own rule; a node's depth is the deepest bed of its elements.
  u/v live on the elements.
- Particles at random positions (uniform on the sphere, |lat| < 76°). Node leg:
  depths 0–3000 m, so some lie below their column's bed and read its last
  point. Element leg: depths uniform between the surface and the nearest node's
  depth, where the end-of-step clamp keeps them; at NSIDE 64 about 16 % of them
  then have a dry element in their `leastsq` ring and about 4 % fewer than three
  wet ones (the area-mean fallback).
- Sigma leg: the dual mesh (node layout) or its triangulated twin (element
  layout) with `vertical_coord = "sigma"`, a 22-interface sigma table finer at
  the top (`σ_k = 1 − (k/21)^1.5`), the node bathymetry above as the bed and a
  surface of ±0.5 m varying in space and from record to record; every layer of
  every column holds water (random u/v everywhere, no dry slots, no walls at
  depth). W and the column height `H_p` by the layout's own interpolator
  (`barycentric` for nodes, `leastsq` for elements). Particle depths uniform
  between the surface and the nearest node's bathymetry, so the sigma
  fractions spread over the whole column.
- `dt = 21600 s`, the step starting a quarter into a one-day record bracket: a
  lazy blend of two records plus the next record (`--payload pchip`: the
  4-record PCHIP bundle, RK2 only — Euler reads the step-start field).
- The thread count is fixed before Numba loads (`--threads`, else
  `NUMBA_NUM_THREADS`, else the CPU count). Pass `--threads` equal to the
  physical core count: the default `os.cpu_count()` counts SMT threads. One warm-up step per placement
  (compilation; the record check, which runs once per record and mesh; a check
  that the expected kernel or step path ran). Then `--steps` timed steps per
  placement, interleaved (layer, linear, layer, …), each from the same state,
  each with a new lazy blend over the cached records, as the tracking layer
  builds it every step. Reported: the median per placement and the ratio
  `layer_linear / layer`, with the machine, the commit (`+dirty-src` when `src/`
  has uncommitted changes) and the thread count.
- Not included: reading the records and the once-per-record check (I/O-side
  costs, see `docs/performance.md` §1).

```bash
# node layout, fused 3-D RK2 (linear and PCHIP in time)
python benchmarks/single_node/depth_linear_cost.py \
    --nside 64 --particles 100000 --steps 15 --threads 128 [--payload pchip]
# element layout, leastsq: fused 3-D Euler; unfused 3-D RK2 (linear and PCHIP)
python benchmarks/single_node/depth_linear_cost.py --layout element --integrator euler \
    --nside 64 --particles 100000 --steps 15 --threads 128
python benchmarks/single_node/depth_linear_cost.py --layout element --integrator rk2 \
    --nside 64 --particles 100000 --steps 15 --threads 128 [--payload pchip]
# sigma mesh (FESOM-C-like), unfused 3-D RK2: element leastsq and node barycentric
python benchmarks/single_node/depth_linear_cost.py --vertical sigma --layout element \
    --integrator rk2 --nside 64 --particles 100000 --steps 15 --threads 128 [--payload pchip]
python benchmarks/single_node/depth_linear_cost.py --vertical sigma --layout node \
    --integrator rk2 --nside 64 --particles 100000 --steps 15 --threads 128 [--payload pchip]
```

Run it on an otherwise idle node; `--nside 16 --particles 1000 --steps 3` is a
smoke test of the script, not a measurement.

## Rows

An otherwise idle HPC compute node (exclusive allocation), Python 3.13.13,
NumPy 2.4.6, Numba 0.65.1. Setup 13.6 s and 1.8 s (fd4e869, node lerp / pchip),
17.3 s, 3.0 s and 2.7 s (3136b19, element Euler / RK2 lerp / RK2 pchip, the first
incl. compilation), 1.5 s and 1.8 s (3136b19, node lerp / pchip). At 3136b19, with
the walls at depth, the cost of the depth rule is 6 % of the fused element Euler
step, 3 % of the unfused element RK2 step with a linear time blend and 10 % of the
fused node RK2 step; with the PCHIP payload it is within the run-to-run spread
(ratios 0.97 element, 1.00 node: no measurable cost). The element workload had a
dry element in the `leastsq` ring for 16.1 % of the particles and fewer than three
wet ones for 4.2 % (counted from the script's own mesh and seeds, the same day).
The two fd4e869 rows were measured before the walls at depth (a vertex whose bed
lay above the particle was then read at its own last usable point, not left out).
At 4067870, on the FESOM-C-like sigma mesh (setup 14.1 s incl. compilation, then
1.7–2.1 s), the cost of the depth rule in `s` per unfused RK2 step is within the
run-to-run spread for element `leastsq` with a linear time blend (ratio 0.98),
3 % with PCHIP, and 3 % / 4 % for node `barycentric` (linear / PCHIP).

| machine | CPU | threads | leg | NSIDE / nodes | particles | payload | layer [ms/step] | layer_linear [ms/step] | ratio | commit | measured |
|---|---|---|---|---|---|---|---|---|---|---|---|
| HPC compute node | 2× AMD EPYC 7763 (128 cores) | 128 | node, fused RK2 | 64 / 49,152 | 100,000 | lerp | 9.774 (9.665–10.939) | 10.183 (10.092–11.101) | 1.042 | fd4e869 | 2026-10-19 |
| HPC compute node | 2× AMD EPYC 7763 (128 cores) | 128 | node, fused RK2 | 64 / 49,152 | 100,000 | pchip | 15.985 (15.944–16.451) | 17.254 (17.209–17.644) | 1.079 | fd4e869 | 2026-10-19 |
| HPC compute node | 2× AMD EPYC 7763 (128 cores) | 128 | element leastsq, fused Euler | 64 / 98,300 elements | 100,000 | lerp | 7.298 (5.466–8.777) | 7.754 (5.904–8.884) | 1.063 | 3136b19 | 2026-10-20 |
| HPC compute node | 2× AMD EPYC 7763 (128 cores) | 128 | element leastsq, unfused RK2 | 64 / 98,300 elements | 100,000 | lerp | 24.181 (23.691–26.483) | 24.792 (24.525–26.023) | 1.025 | 3136b19 | 2026-10-20 |
| HPC compute node | 2× AMD EPYC 7763 (128 cores) | 128 | element leastsq, unfused RK2 | 64 / 98,300 elements | 100,000 | pchip | 37.210 (35.913–39.294) | 36.184 (35.650–37.320) | 0.972 | 3136b19 | 2026-10-20 |
| HPC compute node | 2× AMD EPYC 7763 (128 cores) | 128 | node, fused RK2 | 64 / 49,152 | 100,000 | lerp | 9.588 (9.456–10.206) | 10.517 (10.459–10.590) | 1.097 | 3136b19 | 2026-10-20 |
| HPC compute node | 2× AMD EPYC 7763 (128 cores) | 128 | node, fused RK2 | 64 / 49,152 | 100,000 | pchip | 12.512 (12.484–16.679) | 12.448 (12.404–17.992) | 0.995 | 3136b19 | 2026-10-20 |
| HPC compute node | 2× AMD EPYC 7763 (128 cores) | 128 | sigma, element leastsq, unfused RK2 | 64 / 98,300 elements | 100,000 | lerp | 32.538 (31.635–36.572) | 31.918 (31.004–37.175) | 0.981 | 4067870 | 2026-10-20 |
| HPC compute node | 2× AMD EPYC 7763 (128 cores) | 128 | sigma, element leastsq, unfused RK2 | 64 / 98,300 elements | 100,000 | pchip | 36.565 (36.109–39.490) | 37.688 (37.545–39.161) | 1.031 | 4067870 | 2026-10-20 |
| HPC compute node | 2× AMD EPYC 7763 (128 cores) | 128 | sigma, node barycentric, unfused RK2 | 64 / 49,152 | 100,000 | lerp | 36.508 (35.789–38.455) | 37.513 (36.915–37.971) | 1.028 | 4067870 | 2026-10-20 |
| HPC compute node | 2× AMD EPYC 7763 (128 cores) | 128 | sigma, node barycentric, unfused RK2 | 64 / 49,152 | 100,000 | pchip | 34.491 (34.329–36.637) | 35.712 (35.481–36.076) | 1.035 | 4067870 | 2026-10-20 |

Median of 15 interleaved steps per placement (min–max in brackets).
