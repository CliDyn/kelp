# Grid-hash cell rule — calibration and verification table (2026-09-22)

HPC compute node (AMD EPYC 7763), the project's conda environment, one process. "exact search" = the
batch nearest search in exact mode (`find_nearest_nodes` under
`mesh.exact_nearest: true`) on sampled element centres; anchors are identical
at every cell (the guard), so only the time differs. The scan-cost proxy is
`choose_grid_hash_cell`'s per-query estimate (nodes + cells + 20 per
certificate evaluation); "widening" is the exact search's own certificate on
the 3x3 window (the review round replaced the uniform-coverage bound, which
over-counted: CORE2 7.4 % -> 3.8 % at 0.69 deg). Scripts (transient, in the
feature worktree's `.tmp/`): `core2_calib.py`, `sns_calib.py`, `big_calib.py`,
`hpx_gains.py`, `run_recalib.sh`; the numbers are what matters.

## The two calibrated classes and the threshold

| mesh | nodes | legacy cell | proxy argmin (eligible) | predicted gain | measured on the search | rule (`min_gain` 2.75) |
|---|---|---|---|---|---|---|
| HEALPix CORE2 twin z6 | 35 k | 2.62 deg | 0.93 deg | 2.03 | (benchmark rows: slower at 80-88 N, nothing on global seeds) | keeps legacy |
| HEALPix CORE2 twin z7 | 140 k | 1.31 deg | 0.46 deg | 2.04 | idem | keeps legacy |
| HEALPix CORE2 twin z8 | 562 k | 0.66 deg | 0.23 deg | 2.06 | idem | keeps legacy |
| CORE2 `fesom.mesh.diag.nc` | 127 k | 1.38 deg | 0.49 deg | 2.16 | 1.5-1.6x (0.62 -> 0.42 us/pt, 1 thr); step phase of a global daily run unchanged | keeps legacy |
| FESOM-C North Sea `sns_multi_m8` | 43 k | 0.058 deg | 0.014 deg | 3.57 | 1.7x (0.59 -> 0.35 us/pt, 1 thr) | takes 0.014 deg |
| FESOM2 global 11.5 M | 11.5 M | 0.145 deg | 0.051 deg | 4.90 | 2.4-2.6x here (7.7 -> 3.2 us/pt 1 thr; 1.07 -> 0.40 at 8 thr); 1.67x in the reviewer's run | takes 0.051 deg |
| synthetic estuary (0.0005 deg strip + 0.05 deg offshore) | 21 k | 0.05 deg | 0.0016 deg | >> | 26x on the strip | takes 0.0016 deg |

The quasi-uniform class tops out at CORE2's 2.16, the structure-resolving
class starts at the North Sea mesh's 3.57: `min_gain = 2.75` sits mid-gap.
The proxy's own error across meshes and hardware is up to ~2x (measured /
proxy 0.45-1.85 per mille), which is why only a large predicted gain is
trusted. CORE2 is the marginal quasi-uniform case: its Arctic refinement is
where an explicit 0.5 deg cell measured -28 % per step on the Beaufort run
(2026-09-13); the rule samples the whole mesh and keeps 1.38 deg.

## CORE2 (126 858 nodes, global, legacy cell 1.383 deg) — real certificate flag

| cell (deg) | far_rows | proxy cost | window median / p95 | empty | widening | exact search us/pt (8 thr / 1 thr) |
|---|---|---|---|---|---|---|
| 1.383 (legacy) | 4 | 110.5 | 68 / 189 | 0.0 % | 1.0 % | 0.111 / 0.62 |
| 0.978 | 4 | 72.3 | 35 / 96 | 0.0 % | 2.0 % | 0.099 / 0.50 |
| 0.691 | 5 | 53.4 | 18 / 49 | 0.0 % | 3.8 % | 0.083 / 0.44 |
| 0.489 (argmin) | 6 | 51.3 | 9 / 25 | 0.1 % | 18.1 % | 0.078 / 0.42 |
| 0.346 | 7 | 60.6 | 5 / 13 | 2.3 % | 41.5 % | 0.088 / 0.47 |
| 0.245 | 9 | 79.2 | 3 / 7 | 17.1 % | 63.4 % | 0.114 / 0.57 |
| 0.173 | 12 | 107.4 | 1 / 4 | 36.1 % | 89.4 % | 0.169 / 0.79 |
| 0.122 | 16 | 144.3 | 0 / 2 | 58.6 % | 99.6 % | 0.198 / 1.06 |
| 0.086 | 21 | 195.1 | 0 / 1 | 84.1 % | 100 % | 0.282 / 1.89 |

The measured optimum is flat over 0.49-0.69 deg; the argmin lands on it.
Predicted gain 110.5 / 51.3 = 2.15 < 2.75 -> legacy kept.

**Tracking verification at 0.69 deg** (2-D RK2 leastsq, temporal
interpolation, dt = 1 day, 60 steps, 200 000 seeds spread over the ocean,
8 threads; the earlier calibration's choice):

| cell | step phase mean ms | alive at step 60 | trajectories vs legacy |
|---|---|---|---|
| 1.383 deg (legacy) | 93-97 | 184 479 | - |
| 0.691 deg | 89-108 (mean 97) | 184 479 | lon/lat/status **bitwise identical** |

The same comparison with a MONTHLY dt (11 steps, at 0.346 deg) had 6 of
200 000 trajectories differ: RK2 stages carried 14-61 deg off the mesh by the
monthly displacement reach the far-rows budget, where the anchor is "best
node seen" by contract and hence cell-dependent (`docs/localization.md` §3).

## FESOM-C North Sea `sns_multi_m8` (43 k nodes, quads, legacy 0.0578 deg)

| cell (deg) | proxy cost | window median | empty | widening | exact search us/pt (8 thr / 1 thr) |
|---|---|---|---|---|---|
| 0.0578 (legacy) | 193 | 207 | 0.0 % | 0.0 % | 0.145 / 0.59 |
| 0.0408 | 113 | 104 | 0.0 % | 0.8 % | 0.071 / 0.42 |
| 0.0289 | 77 | 52 | 0.0 % | 10.7 % | 0.088 / 0.36 |
| 0.0204 | 61 | 26 | 0.2 % | 15.0 % | 0.072 / 0.34 |
| **0.0144 (chosen)** | **54** | 13 | 8.5 % | 18.5 % | 0.074 / 0.35 |
| 0.0102 | 60 | 6 | 14.7 % | 26.1 % | 0.082 / 0.41 |
| 0.0072 | 95 | 3 | 18.2 % | 82.4 % | 0.134 / 0.68 |
| 0.0051 | 133 | 2 | 23.0 % | 98.7 % | 0.146 / 0.89 |

Predicted gain 193 / 54 = 3.57; measured optimum flat over 0.0144-0.029 deg.

## FESOM2 global 11.5 M nodes / 23 M elements (legacy 0.1451 deg) — the reviewer's blocker

| cell (deg) | far_rows | hash cells | offsets MB | proxy cost | window median / p95 | empty | widening | exact search us/pt (8 thr / 1 thr) | eligible |
|---|---|---|---|---|---|---|---|---|---|
| 0.1451 (legacy) | 5 | 2.9 M | 11.6 | 495 | 483 / 784 | 0.0 % | 3.3 % | 1.07 / 7.69 | yes |
| 0.1026 | 6 | 5.8 M | 23.1 | 268 | 244 / 395 | 0.4 % | 6.1 % | 0.76 / 6.26 | yes |
| 0.0726 | 8 | 11.6 M | 46.2 | 155 | 123 / 199 | 3.4 % | 8.3 % | 0.57 / 4.38 | yes |
| **0.0513 (chosen)** | 10 | 23.1 M | 92.4 | **101** | 62 / 100 | 4.0 % | 12.7 % | **0.40 / 3.19** | yes |
| 0.0363 | 13 | 46.2 M | 184.7 | 80 | 31 / 51 | 4.1 % | 21.0 % | 0.33 / 2.56 | no (widening, memory) |

Unbounded, the argmin was 0.0363 deg: the last candidate the 50 M-cell cap
allowed, 185 MB of hash offsets per process (a sharded campaign multiplies
it), predicted 5.85x. The reviewer measured that cell 18 % SLOWER than
0.0513 deg (1.41x vs 1.67x over legacy, best-of-4); this run measured it
faster (3.1x vs 2.6x) — the node was shared, and out-of-cache offsets and
node gathers dominate at this size, which the proxy cannot see. Either way
the proxy over-predicts here (5.85x vs 1.4-3.1x), so exact mode is bounded
to the regime it was validated in: widening <= 20 % and offsets <= 128 MB.
The chooser took 2.9 s (8 threads) after the stats kernel went parallel
(15 s serial).

## `benchmarks/exact_nearest_cost.py --cell-rule occupancy` (400 k particles, 8 thr)

Under the first, unbounded calibration the rule halved the cell on every
quasi-uniform row and the fused step got SLOWER where anchors widen: HEALPix
nside 512 at 80-88 N +17-24 %, the 20 deg land-cap seeds x3.5, structured
mid-latitude within noise. Re-run after the review round (real certificate
flag, bounds, `min_gain = 2.75`): the HEALPix nside-512 rows predict 1.63-1.64x,
the structured mid-latitude mesh 1.68x, the structured 80-88 N mesh has no
eligible candidate (every anchor widens at every cell) — every row keeps the
legacy cell and the benchmark's numbers are the `docs/performance.md` §1 ones.
