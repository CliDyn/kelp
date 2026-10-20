# Performance, sizing & operations

> **Covers:** what each configuration choice costs — per-step time, memory,
> I/O, thread scaling, output volume — and how to size a run from it.
> **Does not cover:** what the keys mean (see `configuration.md`) or why a
> particular run is slow (see the diagnosis path in the `kelp-diagnose` skill).
> **Code:** `src/kelp/tracking.py:run_single_realization`, `src/kelp/core/io.py:LazyLerpField`,
> `src/kelp/writers.py:TrajectoryWriter`; measured rows in `benchmarks/`.
> **Verified at:** see `docs/doc_map.yaml` (stamped via check_doc_sync --bless).

How much each configuration choice *costs* and how long a run will take — the complement to
`configuration.md` (what each key does). Example numbers below were measured on a **FESOM2 core2
global mesh** (126,858 nodes), **2-D surface**, on a DKRZ compute node; the absolute values are
hardware/mesh-specific, but the *structure* (linear kernel, memory-bound threading, mesh-set I/O)
is general.

## 1. Cost model — where the time goes
Two facts (in the source) explain the whole cost structure:
- **Interpolation is local** (containing element / nearest-node ring; O(1) grid-hash localization)
  → **kernel cost ∝ number of ALIVE tracers**, independent of mesh size or where the tracers are.
- **The snapshot read is full-field** (`ds[var].isel(time=t).values` loads all nodes) → **I/O ∝
  mesh/field size and number of days, NOT tracer count or location**.

Per step: `t_step ≈ t_IO(fixed) + per_particle · N`. Example (global, daily snapshots, rk2+linear):
**~1.8 s/step** at 127k tracers / 16 threads — of which I/O ≈ 1.0 s (amortized daily read) and kernel
≈ 0.8 s; i.e. **~11 min per simulated year** (a 2026-06 campaign measurement, before the exact-nearest
guard and the node coordinate table; the log is not kept in this repository). Per-particle kernel (per step): euler ≈ **1.1 µs**,
rk2 ≈ **6.6 µs**, rk4 ≈ **22 µs**; `pchip` temporal reconstruction is **≈ free at scale** (a fixed
≈ 40 ms/step, swamped by N).

**Depth-linear u/v** (`mode.uv_vertical: linear`, `vertical.md` §2.1): each stencil member (a
vertex, or an element) reads two data points instead of one, plus a bracket search over its
column and a wet test against its bed, and every record is checked once for finite values down
to each column's last usable point (one parallel pass over the u/v arrays per record read). The
step-cost ratio against the layer rule (same arrays) is measured by
`benchmarks/single_node/depth_linear_cost.py`: node layout, fused 3-D RK2, 1.04 with a linear
time blend and 1.08 with PCHIP (128 threads, 1e5 particles, measured before the walls at depth;
rows in `benchmarks/single_node/depth_linear_cost.md`). The element-layout leg of the script
(`--layout element`: element `leastsq`, the fused 3-D Euler kernel and the unfused RK2 step)
has no measured row yet.

**Run-time rule of thumb** (global, 1-day step, ≥16 threads, rk2): `T ≈ n_days × (1.0 s + 6.6 µs × N)`.
Below ~130k tracers the run is **I/O-bound** (tracer count barely matters); above, it grows linearly
in N.

**Nearest-node anchor, cell size and the exact guard (2026-09-12).** The 3×3 hash-window
scan is a linear pass over the nodes in the window, twice per RK2 step; the reader's automatic cell
(mesh-wide mean spacing — the `legacy` rule since 2026-09-22, see below) leaves 100–135 nodes per window at the refined Arctic of CORE2 and
the scan took 25–45 % of the tracking phase there (`mesh.grid_hash_cell_size: 0.5` → 17 nodes,
daily step 1.13 → 0.75 s on one node) — but a cell below the local spacing makes the anchor
approximate (0.1 % of the Beaufort particles, 0.16 % of the statuses changed in 91 days), see
`localization.md`. `mesh.exact_nearest` (on by default since 2026-09-12) keeps exact anchors inside every fused
kernel (the in-kernel exact search in every fused kernel, the 3-D RK2 barycentric kernels included;
their flag + KD-heal handshake now serves only the guard-off empty-window case; since 2026-09-12 no path
is bypassed). Measured with
`benchmarks/exact_nearest_cost.py` (400 k particles, 8 threads, one fused step; the current
rows, re-measured 2026-09-22 with the node coordinate table, are in
`benchmarks/nearest_scan_table.md` §2 and supersede the 2026-09-12 set): about 0 % for a global
HEALPix nside-512 release and for the fused 3-D RK2 barycentric kernel; +15 % for HEALPix
80–88° N; +37 % with every anchor provable on a structured mid-latitude mesh; ×2 where half the
anchors need a widened window (structured 80–88° N, automatic cell); +61 % in the polar band
89–90° N, where every cap reaches the pole (0.02 % of a global release); ×4.6 with a cell three
times below the node spacing; +7–20 % for seeds that die this step (far outside the mesh, or inside
a 20° land cap: the search budget and its clearance exit; that one-time cost grows as
(budget/cell)² cells per such seed — on CORE2 at the automatic cell the guarded seed load is
faster than the old KD heal, on a mesh whose longest edge spans hundreds of cells it is about a
millisecond per far seed). In every case the fused path stays 1.04–3.6× faster than the unfused
path under the guard. On the real CORE2
Beaufort run (2026-09-13, one model year from the middle, 2.4 M alive, 32 × 4 on one node) the guard
cost +1 % at the automatic cell (0.005 % of the anchors widened; 405 of 2.4 M trajectories differ
from the guard-off run after a year, one status of 7.9 M) and made the 0.5° cell exact — bit for
bit the automatic-cell result — at 28 % less per step (mean daily step over the shards 0.71 → 0.51 s; the slowest shard, which
sets the wall time, 0.93 → 0.62 s: 3.8 instead of 5.7 min per model year). History (three review rounds, 2026-09-12): a first version
healed the 2-D kernels through runner rounds like the 3-D RK2 kernels and was MEASURED slower
than the unfused path at high latitude (one round per unprovable site); the in-kernel search
replaced it; its first cut scanned whole grids for far-outside seeds (a minute per step at
400 k seeds) and cell by cell across polar rows (×15 in the polar band); a fixed 16-row budget
then broke exactness on multi-resolution meshes (coarse elements spanning 50 cells of the fine
patch's cell) and still scanned the whole latitude band for seeds inside a land hole (×27–64);
the row slices, the mesh-derived budget and the clearance exit fixed those.
**Since 2026-09-22 the run chooses the cell itself** (`mesh.grid_hash_cell_rule: occupancy`,
the default): candidate cells are scored on sampled element centres — window occupancy, empty
windows, the widening share, an estimate of the nodes and cells the exact search scans — and
the cheapest safe cell is taken (guard on: the scan-cost minimum among the candidates with
≤ 20 % widening and ≤ 128 MB of hash offsets, and only when it predicts ≥ 2.75× — quasi-uniform
meshes predict 2.0–2.2× and keep the reader's cell; guard off: the smallest cell with no empty
window and no added approximate anchor); the choice is printed
(`grid hash: cell … by occupancy/exact (legacy …)`) and stamped in the output metadata
(`grid_hash_cell`), `legacy` restores the mean-spacing cell bit for bit. Measured with the
exact search (anchors identical at every cell; `benchmarks/grid_hash_cell_rule_table.md`):
CORE2 keeps 1.38° (a global daily-step run gains nothing measurable at 0.69°; an Arctic release
still wants the explicit 0.5° of the paragraph above); the FESOM-C North Sea mesh `sns_multi_m8`
0.058° → 0.014° (window 207 → 13 nodes, search 1.7×); the 11.5 M-node global FESOM2 mesh 0.145° →
0.051° (window 483 → 62, search 1.7–2.6×, 92 MB of hash offsets per process instead of 12); a
synthetic estuary (0.0005° coastal strip, 0.05° offshore) 0.05° → 0.0016° with the strip's scan
cost ÷ 30 (`docs/localization.md` §3). The chooser itself costs one exact batch search of
≤ 200 k centres plus 13 grid passes: ~1.5 s on CORE2, ~3 s on the 11.5 M-node mesh (8 threads),
once per run. `utils/check_grid_hash_cell.py` remains the tool for an explicit
candidate and for the seed positions (the rule samples the mesh, not the seeds).

## 2. Scaling (measured)
- **Re-measurement on the current engine (2026-09-24, commit 4b8b354; 128-core nodes, 2 × AMD EPYC 7763,
  256 GB; nextGEMS ng5 HEALPix ocean stores — zooms 7 and 9 from the archive, the other zooms
  resampled from them — fused 3-D RK2 barycentric, 6-hour steps; "step" = the
  median over read-free steps of the tracking phase of the SLOWEST process):**
  - *Processes on one node* (zoom 9, 6.0 M particles, 128 threads in total, node-shared snapshots):
    1 × 128 0.576 s, 2 × 64 0.540 s, 4 × 32 0.489 s, 8 × 16 0.478 s, 16 × 8 0.447 s — splitting the node
    into processes helps up to 16 processes (−22 %).
  - *Nodes* (zoom 9, 6.0 M particles, 8 × 16 per node, each process reads its own snapshots): 1, 2, 4,
    8, 16 nodes → 0.486, 0.222, 0.148, 0.094, 0.050 s (speed-up 9.7 on 16 nodes with the slowest
    process, 12.2 with the mean over processes; at 16 nodes each process holds only 47 000 particles).
  - *Mesh size* (6.0 M particles, one 128-thread process, zoom 5–11 = 9.4 k–35.5 M wet surface nodes):
    0.63–0.91 s at every zoom — a 3765-fold larger mesh changes the step by at most 1.45×; the grid-hash
    window holds a median of 57–65 nodes on all seven meshes, so the variation is not the search.
  - *Particle count* (zoom 10, one 128-thread process): 0.25 M → 24 M particles = 0.042 → 2.55 s,
    log-log slope 0.88 (0.144 µs per particle-step at 3 M, 0.106 µs at 24 M).
  - *Zoom 11, one 128-thread process per node* (a process needs ~146 GiB, so one per 256 GB node):
    weak scaling at 6.0 M particles per node 0.76 → 0.85 s from 1 to 16 nodes (90 % efficiency,
    96.2 M particles on 16 nodes, 113 × 10⁶ particle-steps/s); strong scaling of 24.1 M particles
    12.6× on 16 nodes. A zoom-11 snapshot read costs ~146 s per process.
  - *Large releases without output* (zoom 11, 3–4 steps): 10⁹ particles on 16 nodes 7.6 s per step
    (132 × 10⁶ particle-steps/s, 122 GiB per node); 10¹⁰ on 16 × 512 GB nodes 73 s (136 × 10⁶/s,
    166 GiB per node); on one 1 TB node 10⁹ → 109 s (230 GiB) and 4 × 10⁹ → 451 s (735 GiB).
    Throughput does not degrade with release size (the memory line is in §3).
  - *A full year with input and output* (zoom 9, 6.0 M, 1460 six-hour steps, one 1 × 128 process,
    float64): 2 h 12 min; mean step 5.33 s = input 2.94 s + tracking 0.62 s + output 1.78 s (7.1 s per
    record, every 4th step) — input and output are 88 % of the step, so they, not the kernel, set the
    wall time of a run that writes trajectories.
- **t_step is linear in tracer count** (kernel ∝ N), mesh-agnostic per particle.
- **Scheme cost ladder**: euler 1× < rk2 ~6× ≈ rk2+pchip < rk4 ~21× ≈ rk4+pchip. **rk4 is the
  expensive knob; pchip is ~free** → `rk2+pchip` is the cost-effective high-time-accuracy choice.
- **Sub-stepping** (`dt < file_dt`, temporal interpolation): t_step is dt-independent; the per-day
  cost `t_day = t_step × (file_dt/dt)` grows linearly with the number of sub-steps. The pchip
  premium does not grow with finer dt.
- **Threads — scaling depends on mode, scheme and dtype** (re-measured 2026-06-30 with
  `velocity_float32`, global core2 + N50 crop, via the standalone `scalability` tool):
  - **2-D** scales steeply: rk2+linear **~×38** (global) / ×45 (crop), rk4+pchip **~×24** / ×28 over
    1→128 threads. (float32 halves memory traffic on the bandwidth-bound kernel; the older **float64**
    numbers of one campaign were ~×3.6 / ×1.9 — the "more than 8 cores are wasted" rule was a float64-2D
    statement and does **not** hold for float32.)
  - **3-D rk2+linear**: **~×5.6** (global) / ×12.6 (crop) — the parallelized RK stage interpolation
    working (it was serial before 2026-06).
  - **3-D rk2/rk4 + pchip**: the old fixed **~5–6.6 s/step** reconstruction floor (independent of
    tracer count, only ~×1.5 thread scaling) was removed 2026-08-15 (TODO 6.7): the whole-field
    PCHIP blend now runs through a parallel Numba kernel (bit-identical to the numpy path), and
    RK4's duplicate midpoint stage is reconstructed once instead of twice (4 → 3 reconstructions).
    Measured on a core2-size field (47 × 244k): one blend 953 ms → 46 ms at 8 threads (×20) /
    20 ms at 32 threads (×47); a full RK4 stage set (u+v+w+eta) ≈ 275 ms at 8 threads / 136 ms
    at 32 — reconstruction is no longer the 3-D pchip bottleneck. (2-D pchip stays fused:
    ~40 ms, scales ×24+.)
  - **Guidance**: with float32 velocities 2-D keeps gaining out to 128 threads (×38 above); the
    UNFUSED 3-D paths (linear AND pchip) saturate around ~16–32 threads; the fused 3-D RK2
    barycentric kernel scales on (see the E2 update in §3). A regional crop scales better (its
    field fits in cache).
- **The I/O floor is set by mesh size, not tracer count or location.** Releasing tracers in a small
  box on the GLOBAL mesh does NOT cut I/O (the whole field is still read). **Extracting the region
  cuts the read by ~2 orders of magnitude** AND improves threading — the decisive optimization for a
  spatially-confined study.

## 3. Sizing recipes
- **Quick global picture**: `rk2 + linear`, `dt: 86400`, all nodes → ~11 min/yr. (euler is cheaper
  per step but degrades long trajectories, and on the global mesh the I/O floor hides the saving.)
- **Accuracy in strong currents** (equatorial / western-boundary jets — where the step changes
  trajectories by 10²–10³ km): finer dt and/or `pchip`. **rk2+pchip** is the sweet spot (pchip
  ~free; avoid rk4's ×3.4). A 1-day step is already adequate in calm interiors.
- **Localized / zoom study**: **extract the region** with `utils/extract_fesom_region.py` (one year,
  only the roles you need — e.g. u/v/ssh for 2-D). ~140× less I/O + better core scaling.
- **Continuous / staggered release**: `time.n_pulses` + `pulse_interval` re-release the seed set.
  Frozen (not-yet-released) particles cost **nothing** (kernel skips them; cost ∝ ALIVE). Caveat: the
  writer stores the **full pre-allocated array** (incl. NaN not-released) every record → `n_pulses`
  inflates output size and write time.
- **T/S sampling** (`sample_vars: [temp, salt]`) **≈ doubles wall time** (two extra 3-D field reads
  per snapshot; the sampling kernel itself is negligible). Omit unless you need T/S along tracks.
- **Threads**: always set `NUMBA_NUM_THREADS` (else numba grabs every logical CPU and small kernels
  run far slower). The sweet spot is mode-dependent (see §2): with float32 velocities, 2-D and
  3-D-linear runs keep gaining out to ~16–32 threads (2-D all the way to 128). 3-D pchip runs used
  to be reconstruction-bound; since the parallel reconstruction (§2) they follow the same ~16–32
  guidance (the blend itself measured ×20 at 8 threads / ×47 at 32).
- **Scaling update (2026-08-22, E2 scans).** The FUSED 3-D RK2
  barycentric kernel is NOT bandwidth-capped: measured ×6.2 going 16 → 128
  threads (zoom-9 HEALPix, 6M particles, one process), and after the
  flag-heal fix (the hybrid-exactness fallback used to charge
  whole-mesh stage blends for even ONE flagged particle) the steady step is
  **mesh-size-independent**: 6M 3-D particles cost 0.99/1.45/1.22 s/step at
  128 threads on 2.2M-, 8.9M- and 35.5M-node meshes (20–36 thread-µs per
  particle; superseded by the 2026-09-24 re-measurement in §2: 0.64/0.64/0.91 s). The 16–32-thread saturation guidance below remains TRUE for the
  UNFUSED paths (sigma meshes, RK4-3D, non-barycentric interpolators).
  Related measured facts: `velocity_float32` halves RAM but costs ~15–20 %
  step time (gather-convert); multi-node sharding: see §2 (the old "×4.7 at
  512 cores, ~83 %" mixed a 4-node speed-up on a read-contaminated worst-step
  metric with an 8-node efficiency); `numerics.shared_fields_dir` (node-shared
  snapshots) removes the per-process field replication that otherwise caps
  process co-residency (a zoom-10 8×16 leg that was impossible unshared
  completed with it). SLURM traps that cost us real OOM rounds: with
  `--hint=nomultithread` SLURM allocates memory PER PHYSICAL CPU (940M×128 ≈
  117.5 GB — HALF the node, even with `--exclusive`; always add `--mem=0`),
  and when benchmarking co-resident shards, quote steady last-5-step times
  with ALL siblings running (a shard stepping while its staggered siblings
  still load shows a flattering "solo window"); MaxRSS double-counts mmap'd
  shared pages — judge node fit by the sum of unique bytes, not per-process
  RSS. **Where to put `shared_fields_dir` (measured 2026-08-22):** within
  ONE node it is nearly free and saves the replication (zoom-9 8×16: RSS
  16→10 GB/proc, step unchanged); ACROSS nodes a cache on the shared (Lustre) filesystem makes
  remote nodes fault mmap pages into their page cache DURING stepping —
  zoom-9 2/4-node legs ran ~0.2–0.3 s/step slower than plain per-process
  reads, and a zoom-11 2×64 leg stepped 5.5 s vs 1.2 s single. Recipe:
  multi-node runs whose fields FIT per-process RAM → shard without the
  cache; big-mesh multi-process runs that DON'T fit → shared cache per
  node (node-local tmpfs — expected to avoid the fault cost, NOT measured:
  every measurement so far put the cache on the shared filesystem), or accept the
  Lustre-fault cost as the price of fitting at all.
- **Sharding = the linear-scaling lever (2026-08).** The UNFUSED tracking
  paths are memory-bandwidth-bound (~16-32 threads saturate one node's bus;
  more cores only queue); on the fused path sharding still pays across
  NODES and against the per-process memory floor (~15–19 GB of catalogue +
  runtime per process — see shared fields above). Particles are
  independent, so split the seed file into
  contiguous shards (`utils/split_seeds.py`), run each shard as its OWN
  process — per NUMA domain within a node (each socket has its own memory
  channels: 2-8 pinned processes/node ≈ multiply the effective bandwidth) and
  across nodes (strictly linear) — then merge outputs with
  `utils/concat_trajectories.py`. For DETERMINISTIC `n_pulses=1` runs the
  merged store is **data-identical** to the unsharded run (every variable's
  values bit for bit; container chunk layout may differ; pinned by
  tests/test_sharding.py) — stochastic runs included since 2026-10-04: each
  shard file starts with a `# kelp particle_offset N` header and the engine
  keys every random draw (diffusion, Az, stochastic beaching/resuspension) by
  the global particle index `N + i`, so all shards use the SAME seeds. In-node
  pattern: prefer explicit `numactl --cpunodebind=$k --membind=$k` per shard
  (srun's `--cpu-bind=cores --distribution=block:block` does not by itself
  guarantee one task per NUMA domain — VERIFY the achieved binding via
  `--cpu-bind=verbose`/hwloc before trusting scan numbers) with
  `NUMBA_NUM_THREADS=cores/S` per shard.
- **Memory of a huge release (2026-09-24, zoom-11 HEALPix, 3-D RK2 barycentric, no
  output):** per process ≈ 61 GiB + 181 bytes per particle once the particles dominate
  — the line through two one-node points (10⁹: 230 GiB; 4 × 10⁹: 735 GiB) predicted
  166.7 GiB for 625 M particles per process, measured 166.2 GiB. Below about 0.4 × 10⁹
  particles per process the snapshot read sets the peak instead (122–146 GiB), so a small
  probe's peak is the read, not the particles. Another mesh, integrator or dtype needs
  its own two-point fit (the recipe: `kelp-experiment` step 5).
- **Spatial seed sorting (2026-08):** `utils/sort_seeds.py` reorders a seed
  file by (HEALPix NEST cell, depth) so consecutive particles share velocity
  cache lines — measured +17 % on the fused 3-D step from the horizontal key
  alone at 1M particles. One-time preprocessing; sort FIRST, then shard
  (every shard becomes spatially compact too). Initial sortedness decays as
  particles disperse over months; an in-run periodic re-sort is staged.
- **Whole-field time blends need C-contiguous float64 fields** (`core.io._blend`, `_pchip_blend4`):
  anything else falls back to serial numpy. Readers now emit C-order (2026-08 fix — Fortran-ordered
  compaction cost ~10x/step on healpix 3-D; a one-time `RuntimeWarning` fires if a large float64
  field ever misses the fast path on memory order again). Caveat: `numerics.velocity_float32: true`
  re-disables the parallel **linear** lerp by dtype (measured 0.05 s → 0.25 s per core2-size blend;
  the 4-snapshot PCHIP blend upcasts first and keeps its fast path) — on a big temporal-interp run
  weigh the halved snapshot RAM against serial per-step lerps.

## 4. Output / writers
- **Backend by extension**: `.zarr` (Zarr v3 — **lock-free, readable mid-run; the flushed chunks
  survive a crash** — records are buffered and written up to 256 at a time, so the buffered tail is
  lost) vs `.nc` (NetCDF4 — HDF5-locked, unreadable by other processes until close unless you set
  `numerics.output_flush_every: N`). For a long run you want to watch live → use `.zarr`.
- The Zarr writer block-buffers records and flushes one adaptive-width obs-chunk at a time, so a
  chunk-boundary flush compresses/writes each chunk once (a mid-buffer checkpoint or the final
  partial buffer may rewrite the trailing chunk — pre-existing behavior) — streaming append is fast
  even for millions of particles. Above 500k particles the trajectory axis is additionally split
  into 500k blocks (TODO 5.4, 2026-08): chunks stay ~4 MB and reading a compact particle subset
  touches only its blocks; runs at or below 500k keep the historical single-block layout.
- **Output size ∝ tracers × records.** e.g. 1.5 M particles × 730 records (`output_freq 5`, 10 yr)
  ≈ 17 GB. Choose `output_freq` for the cadence you need; `checkpoint_freq` ≈ yearly is plenty for
  restart safety on multi-hour runs.

## 5. SLURM operations
- **Partition**: a whole **exclusive** node gives clean, repeatable timing; a **shared** node can run
  ~10 % slower from co-tenant memory contention on the memory-bound kernel (fine for production, not
  for benchmarks). One node is enough: the kernel scales within a single node's core count (up to
  128 threads), and how far each mode actually benefits is set out in §2.
- **Judge progress by the OUTPUT growing** — `stat -c %s out.nc`, or the Zarr record count from
  `time/zarr.json` shape — **NOT** by the slurm log (often block-buffered / empty mid-run) or
  `sacct TotalCPU` (under-reports `--wrap` jobs). To tell *running vs hung*, check CPU on the node
  (`top`/`sstat`).
- **"Looks hung" ≠ hung**: a run can sit with an empty log while numba **JIT-compiles**, and a
  bad/slow node can stretch a heavy unfused compile to tens of minutes — **re-run on another node
  before concluding a code bug** (an exclusive node avoids it).
- Pin the in-tree clone on every invocation: `PYTHONPATH=<repo>/src`; `OMP_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`.

## 6. Accuracy notes (so performance choices are informed)
- **Daily-mean velocity + 1-day step is adequate for the global / calm picture** (finer dt barely
  changes it), but in **strong-current zones trajectories diverge 10²–10³ km** with step/scheme — the
  step matters there. **Do not summarize a spatially-heterogeneous divergence by a global median**
  (calm interiors dominate it and hide the energetic-zone behaviour).
- **The integrator is reversible**: a clean short same-scheme surface round trip returns to sub-grid
  distance; large decadal round-trip divergence is **Lagrangian chaos + irreversible boundary
  clamping**, not a bug. Isolate with a clean short round trip before blaming "chaos".
- **Forward passive tracking on daily means misses episodic/convective sinking**; the
  **backward/inverse** problem (`time_dir: -1.0`, seed the product) recovers it.

## 7. Config quick-gotchas (full reference: `configuration.md`)
- `unod`/`vnod` node velocities **require** `uv_interp: barycentric`; regional extractions rename
  them, so use `var_u: u` / `var_v: v` there.
- A **global** run needs `mesh.rotation: fesom2` (avoids the lon/lat singularity at the true North
  Pole; irrelevant for a Southern-Ocean-only seed set).
- **2-D surface**: `Fdrift2D: true` + `sigma_layer: 0`. A per-step read then loads only that level;
  `max_levels: 2` trims `mesh.sigma` and sampled 3-D scalars, and with `prefetch_size > 1` the window
  loads every variable at full depth — extract the surface level when the read dominates.
- `prefetch_size` > 1 amortizes the snapshot reads (≥ 4 recommended with pchip; any value is
  correct); each window loads every variable of the dataset at full depth.
