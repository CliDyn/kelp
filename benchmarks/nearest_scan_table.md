# Node coordinate table for the nearest-node search — before / after

The node-table change vs the code before it, on an HPC login node (AMD EPYC,
256 cores), the project's conda environment, numba 0.65.1,
2026-09-22. Every row below was measured with the two code trees on the same
inputs, one run at a time (no overlap). All outputs are bit-identical (last
section).

## 1. Batch nearest search + containment, CORE2 (127k nodes), 1M interior points

`find_nearest_nodes` under `exact_nearest` (the certificate and the widening
included) then `find_containing_elements` with those anchors; automatic cell
1.38°; `OMP_NUM_THREADS=1`. The anchor checksum (sum of node ids) is identical
on both sides: 62823726787.

| threads | nearest, main | nearest, branch | ratio | containment, main | branch |
|---|---|---|---|---|---|
| 1 | 3333 ns/pt | 647 ns/pt | 5.2× | 1201 ns/pt | 1147 ns/pt |
| 8 | 437 ns/pt | 88 ns/pt | 5.0× | 172 ns/pt | 148 ns/pt |

Localization (nearest + containment) per point: 4534 → 1794 ns at 1 thread
(2.5×), 609 → 236 ns at 8 threads (2.6×). The pure 3×3 scan measured in the
spec (no certificate) was 3166 → 499 ns (6.3×). Re-measured after the review
round (078e5c5: guard helper at every launch, build-time self-check): 639 /
82 ns/pt, same checksum.

## 2. `benchmarks/exact_nearest_cost.py` (400k particles, 8 threads, one fused step)

Milliseconds per step; "on" = `exact_nearest: true` (the default), "unfused" =
the unfused path under the guard.

| mesh / configuration | main: off / on / unfused | branch: off / on / unfused | on-ratio |
|---|---|---|---|
| structured mid-lat, auto cell, RK4 leastsq (100 % provable) | 361 / 395 / 464 | 115 / 157 / 230 | 2.5× |
| structured 80–88 N, auto cell, RK4 leastsq (49 % provable) | 318 / 541 / 632 | 102 / 201 / 278 | 2.7× |
| structured 80–88 N, cell 0.3 (3× below spacing), RK4 leastsq | 113 / 476 / 574 | 81 / 371 / 455 | 1.3× |
| regional 20×20°, cell 0.05, seeds far outside (die) | 10.0 / 10.9 / 40.5 | 8.7 / 10.4 / 37.9 | 1.05× |
| HEALPix nside 512, 80–88 N, RK4 bary tinterp (37 % provable) | 521 / 645 / 798 | 439 / 504 / 634 | 1.3× |
| HEALPix nside 512, polar band 89–89.9 N, RK4 bary tinterp | 637 / 2619 / 2845 | 531 / 855 / 961 | 3.1× |
| HEALPix nside 512, global uniform seeds, RK4 bary tinterp | 956 / 998 / 1721 | 554 / 541 / 1019 | 1.8× |
| HEALPix nside 512, 80–88 N, 3-D RK2 bary (fused 3-D kernel) | 456 / 472 / 515 | 398 / 385 / 406 | 1.2× |
| HEALPix nside 512, 20° land cap, seeds inside (die) | 1211 / 1343 / 1293 | 932 / 993 / 1033 | 1.4× |

The gain is largest where the scans hold many candidates (the automatic cell
on a multi-resolution mesh; the polar band, whose widened rectangles scan
whole rows) and smallest where the step is dominated by other work (3-D
kernels, dying seeds).

## 3. A tracking phase, CORE2 2-D, 200k particles, 11 monthly RK2 leastsq steps

`kelp run --verbose`, element velocities `u`/`v` of 1960, fused 2-D leastsq
RK2 kernel, `exact_nearest: true`, 8 threads. The `step=` phase of the
steady-state steps (7, 9, 11; the others include I/O and the record flush):

| | step 7 | step 9 | step 11 | mean |
|---|---|---|---|---|
| main | 237 ms | 193 ms | 183 ms | 204 ms |
| branch | 104 ms | 96 ms | 88 ms | 96 ms |
| ratio | | | | 2.1× |

I/O and record phases are unchanged (io ≈ 0.45 s, rec ≈ 0.9–1.1 s per
recording step, both sides).

## 4. Bitwise identity

The two tracking outputs (`out_core2_2d_{main,branch}.zarr`, 200000 × 11)
compare `np.array_equal` (NaN-equal) on `lon`, `lat` and `status`: all three
`True`; 142726 particles alive at the end on both sides. The full test suite
on the branch: 2032 passed, 35 skipped (main baseline 1993 passed, 35
skipped, same 16 slow deselected), incl. the golden trajectories of
`tests/test_regression.py` and the kernel-by-kernel parity pins of
`tests/test_node_xyz.py`.

Scripts: `.tmp/bench_nearest.py`, `.tmp/run_bench_chain.sh`,
`.tmp/bench_core2_2d.yaml` in the worktree (transient, not committed).
