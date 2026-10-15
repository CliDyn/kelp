# Performance, the shared snapshot cache and sharding

Read when choosing threads, processes per node, the node-shared snapshot cache (`shared_fields_dir`)
or a sharded multi-node layout.

## Contents
- Performance note: 3D RK stage interpolation is parallel
- Sharding: how a memory-bandwidth-bound run uses a whole node (and many)

## Performance note: 3D RK stage interpolation is parallel
**3D RK2/RK4 use the *unfused multi-pass* path, but each stage's u/v
interpolation now runs through the parallel vectorised kernels** (it was a serial
per-particle Python loop until 2026-06, which capped 3D throughput on
million-particle runs). Measured on a 10 M-particle 3D probe (rk2, node-layout
barycentric, global mesh, 128 threads): steady step **70.7 s → 15.9 s (~4.5×)**,
CPU 284% → 662%, trajectories bit-identical. The kernel is
**memory-bandwidth-bound**, so it does NOT scale to all 128 cores — ~16–32 threads
saturate it (docs/performance.md §2). **The FUSED 3-D RK2 barycentric kernel
(node-layout, z-level, w_interp nearest/barycentric — the HEALPix 3-D
config) is different (2026-08-22)**: it scales to the full node (×6.2 from
16 → 128 threads measured at 6M particles) and its steady step is
nearly mesh-size-independent (0.63–0.91 s/step for 6.0 M particles from 9.4 k
to 35.5 M wet nodes, re-measured 2026-09-24; docs/performance.md §2). Size 3D compute from a **load-probe on the real
mesh**, not from core count. **Verify before quoting** — perf is a moving target.

**Shared snapshot cache — decide from this table before anything else:**

| forcing + layout | `numerics.shared_fields_dir` |
|---|---|
| kerchunk/HEALPix snapshot read > 1 s, ≥ 2 processes per node | YES — the generator writes `shared_fields_dir` = `<node-local root>/kelp_cache_<run_tag>` (the root passed to the generator as an argument: the partition's node-local in-memory filesystem, the same path on every node — never `$HOME`, never a shared filesystem; config values do not expand `$VARS`) AND `prefetch_size: 1` into every shard config; both asserted in the config check (campaign-templates.md). The cache forces `prefetch_size: 1`; `temporal_reconstruction: pchip` still works with it (the cache keeps the 4-snapshot stencil). The node-local placement is expected to be the fast one but was never measured — every measurement put the cache on the shared filesystem: nearly free within one node, slow across nodes |
| per-year NetCDF, a regional Zarr crop, or one process per node | NO — use `prefetch_size` instead (a Lustre cache made z11 5.5 s vs 1.22 s/step; yearly NetCDF went to 8.9–15 s/step) |

**Big-mesh multi-process recipe (2026-08-22, in order):** sort seeds
(`utils/sort_seeds.py`) → shard (`utils/split_seeds.py`) → run one process
per shard → merge (`utils/concat_trajectories.py`). **The shared snapshot
cache (`numerics.shared_fields_dir`) is for EXPENSIVE single snapshots read
by many co-resident processes** (HEALPix kerchunk stores: 1×128 1.19 s →
8×16+shared 0.86 s per step): point every shard at ONE directory on node
tmpfs — one process reads each snapshot, the rest attach the same bytes
read-only. It is poison in two other cases, both measured: across nodes via
the shared filesystem (a Lustre cache made z11 2×64 5.5 s vs 1.22 s per step), and on
per-year NetCDF forcing (it forces `prefetch_size: 1`; a single-day read
from a lazily concatenated multi-decade dataset costs 10–19 s, so a
full-mesh leg went to 8.9–15.2 s per step, while the 8-day prefetch
amortises it to 0.8 s) — there the prefetch is the lever and a regional
Zarr crop (0.06 s/step input) beats both; fields that fit in RAM need no
cache. Budget the layout: per-process floor ~19 GB at zoom 9 (catalog +
xarray + numba, replicated per process) and the transient read/rotate PEAK
(zoom 10: 58.9 GB vs 15 GB steady) set node feasibility — the floor and the
peak are two different numbers; take the peak from the probe's `sacct MaxRSS`
and write "peak: not yet measured" until you have it, never a guess. Copy this table into
`PLAN.md` and fill the value column ONLY from the named source:
```
| memory item        | value             | source                                   |
|--------------------|-------------------|------------------------------------------|
| per-process floor  | <TBD> GB          | probe log, steady state (`sacct AveRSS`) |
| transient peak     | NOT YET MEASURED  | probe `sacct -j <id> --format=MaxRSS`    |
| processes per node | <TBD>             | (node RAM − 10 % headroom) / peak         |
```
`velocity_float32: true` is a MEMORY knob, not a speed knob
(15–20 % slower per step). SLURM traps that cost us OOM rounds:
`--hint=nomultithread` makes SLURM allocate memory per PHYSICAL CPU
(≈117.5 GB = HALF a 256-GB node even with `--exclusive`) — always add
`--mem=0`; when benchmarking shards, quote the steady step (median rule
above) with ALL siblings running (staggered starts create a flattering
"solo window" for early finishers), and remember MaxRSS double-counts
mmap'd shared pages — judge node fit by unique bytes per node. Speed
numbers are comparable only inside one identical exclusive allocation
(the scan runs its matrix sequentially in one node). Throughput comes from
NODES (2026-09-24, zoom 9, 6.0 M particles, slowest process: 1/2/4/8/16 nodes
0.486/0.222/0.148/0.094/0.050 s, ×9.7 on 16 nodes); splitting a node into
processes shortens the fused step by 22 % (16×8 vs 1×128; 17 % at 8×16) and was worth 3.4× on
the memory-bound unfused 3-D path (2026-08: 1×128 3.87 s → 32×4 1.13 s slowest shard).

**Runs that are unfused anyway: consider `numerics.elem_search: walk`.** It
replaces the per-step containing-element search with FESOM-style edge crossing
(seeded once at release, checkpointed, verdict-identical — any uncertain walk
falls back to the old search): measured ~3.6× faster containment and ~1.3× faster
rk2 steps at 200k particles on the unfused path. That covers element-layout
`barycentric`, `edge_leastsq`, steady barycentric and the unfused 3-D paths.
Keep `local` for node-layout `barycentric` with a next snapshot (HEALPix, NEMO,
FESOM2 `unod`/`vnod`, discrete or interpolated): those runs take the fused
barycentric kernels, and `walk` switches them off. It does nothing for fused
`leastsq` runs (a one-time RuntimeWarning says so).

## Sharding: how a memory-bandwidth-bound run uses a whole node (and many)

One process saturates a node's memory bus at ~16-32 threads — more cores
only queue. The scaling lever is SHARDING (particles are independent):

0. `python utils/sort_seeds.py seeds.dat --nside <mesh-scale>` — spatial
   sort for cache locality (+17% measured on the fused 3-D step); do it
   BEFORE sharding so every shard is spatially compact.
1. `python utils/split_seeds.py seeds_sorted.dat --shards S --outdir
   shards/` — contiguous, order-preserving, deterministic.
2. Run each shard as its OWN process with its OWN config (same everything,
   different `inicloud`/`output_path`). Within one exclusive node, pin one
   process per NUMA domain so each gets its own memory channels:
   prefer explicit `numactl --cpunodebind=$k --membind=$k <run>` per
   shard with `NUMBA_NUM_THREADS=$((cores/S))` (srun's
   `--cpu-bind=cores --distribution=block:block` does NOT by itself
   guarantee one task per NUMA domain — verify the achieved binding with
   `--cpu-bind=verbose` or hwloc before trusting scan numbers). Across
   nodes: an array job, one or more shards per node — linear by
   construction.
3. `python utils/concat_trajectories.py merged.zarr shards_out/*.zarr` —
   for `n_pulses=1` tracking the merged store is DATA-identical to the
   unsharded run (every variable's values bit for bit; container chunk
   layout may differ; pinned by tests/test_sharding.py), stochastic runs
   included. Zarr only. The merge refuses shards passed out of order.

Seeds: keep `diffusion.seed` and `beaching.seed` EQUAL across shards. Each
shard file starts with a `# kelp particle_offset N` header (written by the
splitter; an engine older than 2026-10-04 ignores it and draws per-shard
streams), and the engine keys every random draw by the global index
`N + i`. A shard built another way sets `data.particle_offset` instead; a
chain driver passes `particle_offset=` itself (the leg driver in
[campaign-templates.md](campaign-templates.md) does). Re-splitting a shard keeps
its offset (into a NEW `--outdir`: the splitter refuses to overwrite shard files); sorting
a shard file is refused (sort the full file, then split).
Do not shard `n_pulses > 1` configs through the SEED file (per-shard seed-set
replication changes the particle ordering); shard a pulsed run through its
RESTART instead, **by seed block** — those shards are not contiguous blocks of
the release, so no offset can key them like the unsharded run: give each such
shard its OWN `diffusion.seed`/`beaching.seed` (the run stays statistically
valid, not bitwise equal). Shard k holds seeds
`[k·n_seed/S, (k+1)·n_seed/S)` of EVERY pulse — the repo ships no restart
splitter; this is the whole of one (tested against the writer API; the last
shard absorbs the remainder), run once per shard k:
```python
import numpy as np
from kelp.writers import load_restart, write_restart_file
r = load_restart(SRC); st = r["state"]
sid = np.arange(st.n_particles) % n_seed                          # seed index of every particle, every pulse
keep = (sid >= k * n_seed // S) & (sid < (k + 1) * n_seed // S)   # shard k = one seed block of EVERY pulse
sub = st.subset(keep)                    # EVERY per-particle array, release_lon/lat/z included; never a hand-written field list
wet = None if r["wet_streak"] is None else r["wet_streak"][keep]
write_restart_file(DST, sub, r["step_i"], r["file_idx"], wet_streak=wet,
                   rotation_angles=r["rotation_angles"], release_step=sub.release_step)
```
A hand-written field list misses fields added later (it left the release positions of output
format 2 unsliced, and `write_restart_file` refuses such a state with a length `ValueError`).
Shards by seed block are what a real sharded
pulsed run would hold — contiguous particle blocks of a pulsed run are not
representative (early pulses are mostly dead) and alive-only balancing
leaves a 1.7× slowest-vs-mean tail; balance on alive AND allocated counts.
Checkpoints are per-shard; resume each shard independently before merging, and
merge these shards with `concat_trajectories.py --independent-shards` (they are not
blocks of one release keyed like it, and each has its own seed: without the option the
merge refuses them).
A shard's own particle index is `pulse*n_seed_shard + seed_local`, and a merged
sharded store is shard-major, not the pulse-major order the cohort formula
`pulse = i // n_seed` assumes — keep each shard's seed-block offset next to it
and rebuild cohort ids from the offsets after merging.
