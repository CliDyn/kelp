# depth_linear_cost — what `mode.uv_vertical: linear` costs per step

The depth-linear horizontal velocity (`docs/vertical.md` §2.1) reads two data
points per vertex instead of one and searches the vertex's column for the
bracket. `depth_linear_cost.py` measures what that costs in the hot path: one
fused node-barycentric 3-D RK2 step with a linear time blend — the payload the
tracking layer builds between two records — timed with the SAME u/v arrays read
two ways, `"layer"` (`_fused_rk2_step_3d_bary_tinterp`) and `"layer_linear"`
(`_fused_rk2_step_3d_bary_tinterp_dl`).

## Method

- Synthetic and self-contained: a HEALPix dual mesh (all wet, NSIDE 64 →
  49,152 nodes by default) with varied bathymetry (`60 + 3500·|sin 2φ|` m, every
  17th column 35 m, so each node's last usable layer differs), the 22-level table
  of `run_bench.py` (21 layers down to 4000 m), random u/v (σ = 0.3 m/s) and w
  (σ = 2·10⁻⁴ m/s), zeros in the dry slots below each column's last usable layer
  centre (as a reader leaves them).
- Particles at random positions (uniform on the sphere, |lat| < 76°) and depths
  (0–3000 m, so some lie below their column's bed and read its last point).
- `dt = 21600 s`, the step starting a quarter into a one-day record bracket: a
  lazy blend of two records plus the next record (`--payload pchip`: the 4-record
  PCHIP twins instead).
- The thread count is fixed before Numba loads (`--threads`, else
  `NUMBA_NUM_THREADS`, else the CPU count). Pass `--threads` equal to the
  physical core count: the default `os.cpu_count()` counts SMT threads. One warm-up step per placement
  (compilation; the record check, which runs once per record and mesh; a check
  that the expected fused kernel was launched). Then `--steps` timed steps per
  placement, interleaved (layer, linear, layer, …), each from the same state,
  each with a new lazy blend over the cached records, as the tracking layer
  builds it every step. Reported: the median per placement and the ratio
  `layer_linear / layer`, with the machine, the commit (`+dirty-src` when `src/`
  has uncommitted changes) and the thread count.
- Not included: reading the records and the once-per-record check (I/O-side
  costs, see `docs/performance.md` §1).

```bash
python benchmarks/single_node/depth_linear_cost.py \
    --nside 64 --particles 100000 --steps 15 --threads 128
```

Run it on an otherwise idle node; `--nside 16 --particles 1000 --steps 3` is a
smoke test of the script, not a measurement.

## Rows

An otherwise idle HPC compute node (exclusive allocation), Python 3.13.13,
NumPy 2.4.6, Numba 0.65.1; setup 13.6 s (lerp, incl. compilation) and 1.8 s
(pchip). The cost of the depth rule is 4 % of the step with a linear time
blend and 8 % with the PCHIP payload.

| machine | CPU | threads | NSIDE / nodes | particles | payload | layer [ms/step] | layer_linear [ms/step] | ratio | commit | measured |
|---|---|---|---|---|---|---|---|---|---|---|
| HPC compute node | 2× AMD EPYC 7763 (128 cores) | 128 | 64 / 49,152 | 100,000 | lerp | 9.774 (9.665–10.939) | 10.183 (10.092–11.101) | 1.042 | fd4e869 | 2026-10-19 |
| HPC compute node | 2× AMD EPYC 7763 (128 cores) | 128 | 64 / 49,152 | 100,000 | pchip | 15.985 (15.944–16.451) | 17.254 (17.209–17.644) | 1.079 | fd4e869 | 2026-10-19 |

Median of 15 interleaved steps per placement (min–max in brackets).
