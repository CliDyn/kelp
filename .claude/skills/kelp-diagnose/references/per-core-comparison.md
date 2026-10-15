# Per-core comparison across tools or engine versions

Read after node 1 of the tree, before any per-core number or any sentence about "the algorithm".
Everything below assumes the phase split of node 1 is done.

**Pin the WHOLE process** — `CPU=$(python -c 'import os; print(min(os.sched_getaffinity(0)))')`,
`taskset -c $CPU …` (never a hard-coded 0, which may lie outside the allocation, and
never a range such as `0-$(nproc)`; `CPU` comes from plain `python`, not from
`taskset`) — reader decompression and writer codec threads included; KELP prints no
thread count itself, so run it through one process that prints first:
`taskset -c $CPU python -c 'import numba, os, sys; print("threads", numba.get_num_threads(), "cpus", os.sched_getaffinity(0), flush=True); from kelp.cli.main import main; sys.exit(main(sys.argv[1:]))' run config.yaml -v`;
`export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1`
(one assignment each — a chained `A=B=C=1` sets only `A`, to the string `B=C=1`)
for both codes (numpy's BLAS threads inflated one tool's single-core cost 1.4×);
the log header prints `numba.get_num_threads()` and `os.sched_getaffinity(0)` from
inside the process. On a partition that sizes CPUs by memory (the interactive
one), `--cpus-per-task` is NOT the thread count: "1-thread" legs ran with 5 and 9,
and `SLURM_CPUS_PER_TASK` inherits the inflated count, so
`NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK` runs 5–9 threads — write `NUMBA_NUM_THREADS=1`
literally.
Then fill this table (unknown = `?`, never blank) before any sentence about "the
algorithm":
```
| code | searches per stage (nearest / u,v element / W element) | passes over particles | cell reused for u,v,w | final domain check |
| KELP 3-D RK4 unfused | 1 / 1 / 1, all cold, × 4 stages | 4 + 1 final | no | yes |
| other tool | | | | |
```
(Another KELP path — fused RK2, 2-D — gets its own row, read from the code.)
With identical numerics, KELP's 3-D RK4 (unfused path: nearest node + u/v element + a separate W
element, all cold, at every stage, plus a final domain check) cost about 3× more per
particle-step on one core than a tracker that searches once per stage from the
previous cell and reuses it for u, v and w (one past measurement on another setup —
re-measure; never quote it as this run's gap, nor as a confirmation of a reported
one). Then profile (7) and A/B one knob at a time on one pinned core → 12.
Legs to prepare, each pinned as above:
```
1. KELP baseline, same config.
2. The same leg + `py-spy dump` ×3 — always, not only on an anomaly.
3. A/B `elem_search: walk`.
4. A/B `integrator: rk2` (it has a fused 3-D kernel).
5. The other tool, pinned, with the same four variables (a template if its command is unknown).
```
Legs 3–4 are the baseline YAML with ONE key under `numerics:` changed (`elem_search: walk`,
`integrator: rk2`) — `yaml.safe_load` → set → `safe_dump`, never retyped; `load_config` the result and
assert that key and `dt` against the baseline.
