---
name: kelp-bench
description: Use when asked to benchmark KELP on a specific machine (laptop, workstation, HPC node), to produce a cross-machine comparison row, or to check whether this machine's tracking speed matches the reference rows in benchmarks/single_node/README.md.
---

# Single-node tracking benchmark on this machine

Goal: one table row per machine — `machine, CPU, threads, particles,
steady step [s], us/particle, thread-us/particle` — produced by
`benchmarks/single_node/run_bench.py` (self-contained synthetic HEALPix store; no
data downloads; its README carries the reference rows). The metric is the script's
own steady-step convention, printed in its result block, on a fixed synthetic
workload — it is not a substitute for the log-median rule the run skills use on
production logs.

## Pinning (every row, both for KELP and for any tool compared with it)

- `export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` for a
  single-core row (numpy's BLAS threads inflated one single-core number 1.4×); a
  single-core row also runs the whole process under `taskset -c <cpu>`.
- The thread count in the row is printed from INSIDE the process
  (`numba.get_num_threads()`, `len(os.sched_getaffinity(0))`), never taken from
  `--cpus-per-task` (the interactive partition raised it to 5 and 9 for "1-thread" legs).
- Re-measuring a published number: regenerate the published configs with an asserted
  path-only difference (same seeds, stores, metric); anything else is a new
  experiment with its own name.

## Steps

1. Environment: Python >= 3.11; `pip install -e .` from the repo root; check that
   `healpy zarr xarray netCDF4 numba` import (wheels exist for x86-64 and arm64).
2. Warm run: `python benchmarks/single_node/run_bench.py --zoom 7 --workdir .tmp/bench`
   (fast sanity; fills the JIT cache).
3. Headline run: `python benchmarks/single_node/run_bench.py --zoom 8 --workdir .tmp/bench`
   twice — once with `--threads <physical cores>` and once with all logical cores.
   On a hybrid CPU the first run uses the performance cores only (Apple silicon:
   `sysctl hw.perflevel0.physicalcpu`; Linux: `lscpu`). Pass `--workdir .tmp/bench`
   so the ~300 MB zoom-8 store is built once. Report the better steady step; note both.
4. Report: paste the result block the script prints verbatim + fill the table row;
   check the particle count in the block against the expected workload (zoom 8
   ≈ 1.55 M, zoom 7 ≈ 387 k) — a wrong count means a reused seed file, and the row
   is invalid. Peak memory: run under `/usr/bin/time -v` (Linux) or
   `/usr/bin/time -l` (macOS) if asked.
5. Rules: compare only same-bench rows (same zoom, same steps) — never against
   real-data runs (heavier workload); keep zoom/steps identical across machines;
   if the README has no reference row for a comparable machine, say so rather
   than inventing one. Running the benchmark on a cluster node is a compute job
   and needs the run owner's explicit yes for that command.
