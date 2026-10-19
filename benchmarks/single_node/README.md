# single_node — portable single-node tracking benchmark

One command, no external data: builds a synthetic HEALPix 3-D velocity
store, seeds ~1M particles, runs the same fused RK2 + temporal
interpolation + barycentric configuration as the reference scaling experiment
(3-D, rotated frame, z-levels), and prints the steady tracking-phase step time
and thread-microseconds per particle (the step time × threads ÷ particles — the
machine-independent way to compare a laptop with a compute node).

```bash
python -m venv .venv && source .venv/bin/activate      # or conda
pip install -e '.[healpix,zarr]'                       # from the repo root
python benchmarks/single_node/run_bench.py             # zoom 8, ~minutes
python benchmarks/single_node/run_bench.py --zoom 7    # quick check
```

- `--threads N` (or `NUMBA_NUM_THREADS`) — on Apple silicon use the
  PERFORMANCE core count for the headline number, then optionally all
  cores for a second row.
- `--workdir DIR` reuses the built store between runs (recommended when
  scanning thread counts); default is a temp dir under `.tmp/`.
- Compare machines with the SAME zoom/steps settings of the SAME
  benchmark. The synthetic field is dynamically boring on purpose —
  this measures the engine (gathers, stencils, time blends), not the
  ocean; absolute numbers are lighter than real-data runs.
- Read the particle count in the result block against the expected
  workload (zoom 8 ≈ 1.55 M, zoom 7 ≈ 387 k) before a row goes into a
  table: until 2026-09-14 the seed file was shared across zooms in one
  workdir, and two early rows silently ran the wrong particle set.

## Rows so far (8 steps, steady step = mean of the last 5 steps, shipped engine with the exact-nearest guard)

| machine | CPU | zoom | threads | particles | steady step [s] | µs/particle | thread-µs/particle | measured |
|---|---|---|---|---|---|---|---|---|
| HPC compute node | 2× AMD EPYC 7763 (128 cores, 256 GB) | 8 | 128 | 1,548,988 | 0.241 | 0.156 | 19.9 | 2026-09-14 |
| HPC compute node | same | 8 | 64 | 1,548,988 | 0.424 | 0.274 | 17.5 | same |
| HPC compute node | same | 8 | 32 | 1,548,988 | 0.803 | 0.519 | 16.6 | same |
| HPC compute node | same | 8 | 16 | 1,548,988 | 1.518 | 0.980 | 15.7 | same |
| HPC compute node | same | 7 | 128 | 387,244 | 0.058 | 0.150 | 19.2 | same |
| Laptop | Apple M3 Pro 11-core (5P+6E), 18 GB | 8 | 5 (P-cores) | 1,548,988 | 1.694 | 1.094 | 5.5 | 2026-09-14 |
| Laptop | same | 8 | 11 (all) | 1,548,988 | 1.228 | 0.792 | 8.7 | same |

Peak RSS: laptop 2.4 GB (11 threads), HPC node 1.96 GB (128 threads). The
guard costs +3–6 % on this workload (2026-09-12 rows with the guard off:
0.228/0.410/0.772/1.464 s — superseded; that run's zoom-7 row was invalid,
it ran on the zoom-8 seed file).

## Other measurements in this folder

- `depth_linear_cost.py` — the step cost of `mode.uv_vertical: linear` (the
  fused 3-D RK2 step with the same u/v arrays read per layer and linearly in
  depth); method and rows in [depth_linear_cost.md](depth_linear_cost.md).
- [fesom14_lr_cost.md](fesom14_lr_cost.md) — the FESOM 1.4 reader on the
  published LR output: measured I/O (mesh, dataset open, record reads), the
  step cost pending.
