---
name: kelp-experiment
description: Use when starting a KELP experiment that is big (millions of particles, several processes or nodes), long (longer than one job's wall clock, so a resumable chain), on a large mesh that may need regional cropping, or a speed and memory showcase without output; also when planning probes, shards, restarts or a campaign folder. For one ready run use kelp-run.
---

# Setting up a new KELP experiment (big / long / big-mesh)

This is the **campaign layer** above the two run skills — use it to decide the
*order of operations* and avoid the traps that bite when an experiment is large,
long, or on a big mesh. It does not replace them:

- **kelp-run** — turns one described task into **seeds + config + command**.
- **kelp-config** — the **YAML** mechanics for one config.
- **`docs/performance.md`** — measured cost model, big-mesh I/O knobs, SLURM ops.
- **`docs/io_and_time.md` §6** — the resumable-chain restart mechanics.

Keep the methodology here **generic**: concrete paths, mesh names, and data
locations belong in the experiment's own folder/config, never in this skill.

**Reference files (read the one the step points to):**
- [references/campaign-templates.md](references/campaign-templates.md) — folder files, job scripts, chain driver, config check, generator.
- [references/huge-release.md](references/huge-release.md) — ≥ 10⁸ particles per process, showcases without output.
- [references/performance-and-sharding.md](references/performance-and-sharding.md) — threads, shared snapshot cache, sharding.
- [references/grid-hash-cell.md](references/grid-hash-cell.md) — the nearest-node cell on non-uniform meshes.
- [references/probes-and-timing.md](references/probes-and-timing.md) — benchmark legs on a running campaign.

## The order that avoids rework

Each step exists because skipping it cost a rerun at least once.

1. **Isolate the experiment from the dev code.** Give the experiment its own
   folder OUTSIDE the repo *and its own KELP clone* (or a worktree off main
   when a main-bound fix must be extracted), and write
   the folder's `AGENTS.md` AND `REPORT.md` (its four headings, empty) FIRST from the
   templates in [references/campaign-templates.md](references/campaign-templates.md); the handback `SUMMARY.md` ends with the
   literal line `Launch gate: nothing was run; say "yes" to run the smoke test.`
   Every step below is a SLURM job unless it is pure file generation — `kelp
   run`, `kelp info` on a million-node mesh and `check_grid_hash_cell.py` are
   jobs, not login-node work. Then run that clone via
   `PYTHONPATH=<clone>/src` so the development repository can keep changing
   without disturbing a multi-week run. Pin the clone to a known commit; pull a
   newer one only deliberately. (A resumable chain re-imports KELP every job,
   so a mid-run pull lets *later* jobs pick up a fix — e.g. a speed-up — without
   restarting.) The editable `pip install -e` in the environment stays pointed at
   the dev repo; `PYTHONPATH` wins over it per job. Print `kelp.__file__` and
   the clone's commit at job start so the log says which copy ran; wipe the
   clone's `src/**/__pycache__` after every `git pull` (stale numba kernels).
   Every run whose numbers are reported — campaign job, laptop run, one-off
   sweep script — records the engine it ran (commit + dirty flag) in its result
   folder: for a `kelp run` output the store's `kelp_version` (version + commit, no
   dirty flag — the job prints `git -C <clone> status --porcelain --untracked-files=no`
   at start), an `engine.json` for a script (the `kelp-run` "Comparing two settings" block
   writes it). A hash cited in a README or a report is on main or carries a
   tag: tag a branch that will be rebased BEFORE its hash is cited (a local
   tag; pushing it is the owner's call). A README or pin engine line is pasted
   from what the run recorded, after the run, in the same commit as its results.
   **Configs are build artefacts:** write a generator script, regenerate, never
   hand-edit or overwrite an old run's config/store (a protocol change = a new
   run tag); validate every generated config with `load_config` on the login
   node before any `sbatch` — a parse error must never be discovered by SLURM.
   Paths in the YAML are absolute (relative ones resolve against the config
   file's directory, not the working directory — a first smoke run died
   on that). Seeds are deterministic and content-addressed by filename (a
   changed seed set = a new file), and a low alive fraction on a fresh seed set
   is the seeds (land, shelf shallower than the release depth), not the engine.
   **Reproducing a published run:** spend ten minutes searching the shared
   filesystem for the producing program and its seed files (the author's project
   space) and ask the author BEFORE georeferencing anything off a figure
   — one setup was digitised off a published figure while the original code,
   seed files and reference trajectories sat on the same cluster; days lost,
   everything re-staged.

2. **Crop the mesh + forcing to the region of interest (big meshes).** Cropping
   to Zarr is the memory lever that makes 3D / multi-snapshot runs fit in RAM and
   cuts per-step I/O. Rules learned:
   - Extract the velocity **family the run will interpolate** — node velocities
     (`unod`/`vnod`/`w`) for node-layout/barycentric runs, not element `u`/`v`.
     Mixing cadences (daily node vs monthly element) fails fast — pick one family.
   - Know what the extractor reads and writes: ONE input file per role (the
     first `u*`/`v*`/`w*`/`ssh*` match of a directory, or the per-role paths of its
     config) — so one extract covers the records of that file; it does NOT
     concatenate years. It writes `<role>.region.zarr` (velocities renamed to their
     role, e.g. `u`/`v`/`w`) plus `fesom.mesh.diag.region.zarr` — NOT the multi-year
     `<var>.fesom.<YYYY>.nc` layout. For several years, cut each year (or a
     pre-concatenated file) and concatenate the extracts along time before the run.
     Point `data.ncinput` at the extract and set `mode.var_u: u` / `var_v: v` (or
     `ncinput_v`/`ncinput_w` per store); make the store's time chunk equal to the
     run's `prefetch_size`.
   - Run the crop on a **compute/shared** partition, never the login node.
   - **Size the crop for the run's duration; the plan carries both lines.** An open crop
     boundary kills every particle that reaches it, so the box holds the region the particles
     can reach in that time (a regional crop can lose all its surface particles within a few
     years):
     ```
     CROP: <box, or "full mesh"> — reach in <duration>: <path length at a typical 5 cm/s ≈ 1,600 km per year, an upper bound; the basin the release connects to>; the box holds that reach, or the run is shortened, or it goes on the full mesh
     ROTATION: <the box or the expected drift reaches poleward of 60° → mesh.rotation: fesom2 | not needed: <reason>>
     ```
     The drift sets the box, and the box sets the rotation: a box cut south of a pole to
     avoid `mesh.rotation` caps the experiment instead.
   - **Validate the crop equals the full mesh** (bit-identical on a shared subset)
     before trusting it.
   - Know what the crop buys: **input ×10–14, tracking only ×1.2–1.5** (measured,
     full mesh vs crop: 0.7–0.85 s vs 0.06 s per step per process on input);
     once tracking is sharded the input term dominates, so crop for I/O. Going
     the other way (crop → full, e.g. continuing a run on the full mesh) the
     input term gets 10–14× MORE expensive: re-probe the shard and thread layout
     on the full mesh before sizing `--time`; the crop's numbers do not transfer.
     `CHECKS.md` carries this row verbatim:
     `| crop → full | input ×10–14, tracking ×1.2–1.5 | re-probe layout before sizing --time |`
   - Symlink only the years a smoke/probe needs into its own directory so the
     multi-year reader does not assemble every year of a multi-decade archive.

3. **Let extraction FINISH before you read it.** The multi-year reader opens the
   dataset **once** at job start and assembles whatever years exist *then* — it
   will not pick up years written later, and a store still being written is
   partial. Launch only when **every year × every variable is complete**. For
   smoke tests, point at a single *finished* year so you don't race the writer.

4. **Fail-fast probe → smoke → scale, in that order.** Run the info probe first
   (mesh loads; the multi-year store assembles into **one monotonically
   increasing time axis**; `n_steps` matches the assembled length; the velocity
   **layout is compatible with the interpolator**). Then a short run. *Then*
   scale. Reading the startup summary catches 90% of misconfigurations for free.
   **3-D: settle the SIGN of the vertical velocity against physics on the real
   forcing before the first leg** (the `kelp-run` check: equatorial upwelling, run on a
   config that points at the global SOURCE files — a regional crop has no equator) —
   every FESOM2/HEALPix 3-D campaign ran with the wrong sign until the readers
   converted it; a synthetic test in our own convention cannot catch this.
   Record the forcing inventory (record count, time span, variables, levels,
   provenance) in the campaign README before writing the experiment description.
   **Record 0 of a model archive is a cold start.** Before writing
   `time.start_time_idx`, print the zero fraction of `eta` at records 0 / mid /
   end; make the chosen index a GENERATOR parameter and assert it
   (`assert c.time.start_time_idx > 0`). Skipping one record is not spin-up — an
   archive whose forcing carries a periodic signal needs hours of records to
   develop; start micro configs and production well past that point.
   **Paired A/B arms on two mesh variants of the same domain**: the two arms'
   configs must differ in EXACTLY one path token (`…_a.nc` → `…_b.nc`);
   take `mesh.meshpath` and `data.ncinput` from the SAME file per arm and `diff`
   the two YAMLs before submitting — the layout checks match by dimension NAME
   and will pair one arm's mesh with the other's velocities silently and run to
   completion (the two meshes differed by three nodes). If the mesh genuinely lives
   in a separate file, assert its node count against the data file. Seeds in
   lon/lat, never node ids; align the arms by record index after asserting equal `time` values
   and `time` `units` (format 2: the same instants).
   **After generating configs, run the config check (references/campaign-templates.md) immediately,
   on every config** — a failing assert is the expected outcome the first time;
   an assert that can pass while the key is unset (`is None or …`) is not a
   check; a `grep` is not a check (a key in the wrong section is silently
   ignored and the grep still finds it).

5. **Load-probe at the TARGET particle count before committing compute.** Measure
   peak **RAM**, **per-step** wall time, **per-record output size**, and — easy to
   miss — **CPU utilisation** (parallel efficiency). Read the verbose per-step
   **phase breakdown** (`io=` vs `step=`), not just the total: a healthy fused 2-D
   step is **µs per particle** — a `step=` phase at ~ms per particle means a
   search structure degenerated (see the non-uniform-mesh trap below), and
   extrapolating it would blow the wall limit ~100×. Do not trust extrapolation
   from a tiny run. Be explicit about what the probe measures: a probe on the
   *full* mesh over-estimates a *cropped* production run, so treat it as an upper
   bound, not the number. Probe on the FULL seed set — all shards, every process, 8 steps, from its OWN
   generated config (`n_steps: 8`, `output_path` under `probe_out/`, everything
   else identical — there is no CLI flag that limits steps, so a probe pointed at
   the production config runs the whole campaign); a 10 k-particle probe measures
   nothing about node feasibility (its RSS is the floor, not the peak) — it gives
   the wall-clock budget AND the RAM per process. **The steady step is the
   MEDIAN of `step=` over steps ≥ 2 with `io < 0.5 s`, of the worst process** —
   never "the mean of the last 5 steps" (that convention caught a snapshot-read
   step and inflated a reported number: 1.187 → 0.997 s clean). When reads
   are NOT rare (a kerchunk store at ~10 s every k-th step), the filtered median
   no longer sizes the leg: size `--time` from the cost per MODEL DAY =
   (Σ io + Σ step over one full read cycle) / (days per cycle), and quote the
   median only as the kernel cross-check. Kernel cost
   is proportional to ALIVE particles: for a pulsed run probe from a mid-run
   restart, not from the seed file (references/probes-and-timing.md).
   On an EXCLUSIVE partition (a job gets all 128 cores whatever
   you ask) request `--cpus-per-task=128`, `NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK`
   and confirm with `sacct -j <id> --format=AllocCPUS`.
   **Thread count is measured, never assumed — a thread-scan leg:** one job, three
   `NUMBA_NUM_THREADS` values (16 / 64 / 128), ~250 steps each, ~3 minutes total;
   record all three in the ledger (2-D FESOM-C 1.2 M nodes measured 417 / 148 /
   114 ms per step — the 16-thread sweet spot of small meshes does not hold).
   Record `MaxRSS` from `sacct`: a 2-D run on a regional mesh peaks below 1 GB
   whatever the input file size, so do not tune `--mem` on an exclusive
   partition. (The "MaxRSS double-counts mmap'd pages" caveat applies only to
   multi-process shared-cache legs.)
   **Processes per node and the node-shared snapshot cache** (`shared_fields_dir`): decide from the
   table in [references/performance-and-sharding.md](references/performance-and-sharding.md) before the probe.
   **Huge releases (≥ 10⁸ particles per process) and showcases without output:** the memory
   fit, the tracking-only driver, the steady-step extractor and the numbered job layout are in
   [references/huge-release.md](references/huge-release.md) — read it before sizing such a job.

6. **Checkpoints are OPT-IN — enable them on EVERY leg that could touch a wall
   limit.** `numerics.checkpoint_freq` defaults to **0 = off**, and the end-of-run
   checkpoint only happens if the run *finishes* — so a job that hits its
   time limit without `checkpoint_freq` set loses **everything** (the output
   store has no `restart_*` block, so `load_restart` raises `ValueError`; a format-1 store keeps
   `restart_valid` 0 — there is nothing to resume from). Set
   `checkpoint_freq` (e.g. a few hundred steps ≈ tens of model days) in the
   config *generator* before the first submit, not after the first timeout;
   the cost is one buffered-flush per interval, negligible. When sizing
   `--time`, budget from the load-probe per-step number **with slack** — and
   still enable checkpoints, because the probe is not a guarantee.

7. **Outlives one wall-clock? Use a dependency CHAIN, not a time prediction.** You
   cannot know when the wall will hit, and a wall-killed job cannot resubmit
   itself — so do not try. Submit N jobs with `--dependency=afterany` (the next
   starts even when the previous was killed), each **resuming from the latest
   restart**; write a restart **frequently** (e.g. every 1/4 model-year) to bound
   the work lost on a kill; **concat the segments** at the end. Surplus jobs
   no-op once the run is complete. **Verify restart pickup is bit-for-bit** on the
   *real* setup (3D + rotation + node-layout, whatever applies) **before**
   launching the chain — resuming and running straight through must match exactly;
   the check is `resume_check.sh` from references/campaign-templates.md, a script in the
   run folder whose last line goes into the README.
   Per-step cost is flat in `dt`, so wall time is LINEAR in substeps: halving
   `dt` doubles the run. Request `--time` = 1.3 × YOUR OWN probe-based estimate,
   rounded up, never above the partition cap (the ceiling is `<cap − 15 min>`:
   11:45:00 under a 12 h cap, 07:45:00 under an 8 h cap — read the cap
   from `sinfo -o '%P %l'`, never copy a number from this text). If the deepest rung still does not fit, shorten the
   horizon — never the margin — or move to a longer-limit partition IF this
   cluster has one (`sinfo -o '%P %l'`; some shared partitions allow 7 days, some
   clusters nothing beyond 12 h); more cores do not help a wall-clock-bound leg. Cancel a doomed leg early (checkpointless past half its wall clock,
   or superseded by a fix) instead of burning the hours. Every submit gets a
   ledger row (job id, config, partition, elapsed, state) and the campaign
   folder gets a `sacct` snapshot the day a job finishes — `sacct` ages out.

8. **Size the output deliberately.** Output volume ≈ `(n_steps / output_freq) ×
   allocated particles × 66 B` uncompressed — the writer stores EVERY allocated
   particle at every record, dead and not-yet-released included (per record:
   lon/lat/z/status + `vel_*` + samples; a compressed Zarr store measured ~37 B;
   `output_freq` counts STEPS, not days — at `dt = 6 h` "every 5 days" is
   `output_freq: 20`). At millions of particles a fine `output_freq`
   is **hundreds of GB to terabytes** — choose it against your storage. The `vel_*`
   diagnostics are ≈ 40 % of the volume but there is NO config key to drop
   them — the levers are `output_freq` and post-processing.

## Traps we have actually hit (so you don't)

| Symptom / risk | Cause | Do this |
|---|---|---|
| `KeyError` opening a multi-year native-name store | reader's broad glob renamed `unod`→`u` because `mode_cfg` wasn't passed | pass `mode_cfg` to `open_dataset` so it assembles by **exact** variable names |
| run truncates early / wrong years | started before extraction finished; the reader opens once at start | wait until **all years × vars** complete before launch |
| `OUTSIDE` storm / fails at step 0 | node-layout velocities with `leastsq`, or element-vs-node mismatch | node layout ⇒ `uv_interp: barycentric` (leastsq is element-centre and fails fast) |
| metric blows up near the pole | geographic lon/lat singularity (`cos(lat)→0`) | `mesh.rotation: fesom2` (FESOM2); output stays geographic |
| rotated vs unrotated trajectories diverge | usually Lagrangian **chaos** amplifying a discretisation-level seed (verified once by dt convergence) | call it chaos only after a short-leg check passes — both frames converge as `dt` shrinks and the ensemble statistics agree; otherwise keep a frame-bug hypothesis; trust the **statistics**, not individual paths |
| Zarr write: "chunks overlap multiple dask chunks" | dask chunks not aligned to the zarr encoding chunks | clear inherited `encoding['chunks']` and **rechunk uniformly** before `to_zarr` |
| chain stalls after a wall-kill | tried to self-resubmit / predict the wall | `afterany` dependency chain + frequent restart instead |
| job timed out → whole run lost: no `restart_*` block, `load_restart` raises `ValueError` (format 1: `restart_valid=0`) | `checkpoint_freq` left at its default **0 = off** (the final checkpoint needs the run to finish) | set `numerics.checkpoint_freq` in the config generator **before the first submit**; resubmit longer legs with `--dependency=afterany:<old>` so the dead job and the rerun never write the same store concurrently |
| validated, then it still drifted | trusted the test suite / small run | re-validate **convergence and restart bit-for-bit on the real setup** before scaling |
| a multi-shard `srun` step lost its slow shards mid-run | srun's default kills the step 600 s after the FIRST task exits | `srun --wait=0` on every multi-process step with unequal shard runtimes (cost of forgetting: one wasted leg) |
| later probe legs started from a later state / found the run already finished | the probe driver wrote its end restart back into the shared `restarts/` dir | probe drivers write to `<segment_dir>/restart_out/`; the canonical restart dir is READ-ONLY for probes |
| "z8 stuck in the mesh build" for hours | block-buffered stdout — the steps had been running from the start | `python -u` + `stdbuf -oL`; judge progress by the growing output store |
| a node OOMs when 8 processes start together | simultaneous daily reads by every shard | stagger shard starts (`sleep $((SLURM_LOCALID * STAGGER_S))`), isolate legs, put risky legs last |
| a variable set before `srun … bash -c '…'` is empty inside the task | not exported — `bash -c` starts a new shell | `export VAR` (or pass it on the command line); `echo` the task command once first; names built from `SLURM_PROCID` (cache dir, shard, log) stay INSIDE the single-quoted task string |
| a resubmitted job still ran the OLD script | the edit command had failed (its own assertion) and the job was resubmitted anyway | after editing a job script, `grep` the changed line before `sbatch`; a failed edit means nothing changed |
| a queued (`afterok`) job would have produced a superseded product | the choice changed after it was submitted | re-read every pending dependent job's command against the current choice before it starts; cancelling a pending job costs nothing |
| a "re-measurement" is not comparable with the published numbers | configs rewritten by hand, seeds or stores changed | regenerate the published configs with an asserted path-only diff (output and cache paths); same seeds, stores and metric; anything else is a new experiment with its own name |
| a rerun of the same leg overwrote its logs | per-process log names without the job id | include `$SLURM_JOB_ID` in every per-process log name |
| a multi-process step crashes with `NRT_adapt_ndarray_to_python: 'descr' is NULL` | concurrent compiles into one numba cache | `export NUMBA_CACHE_DIR=<campaign>/.tmp/numba_${SLURM_JOB_ID}_${SLURM_PROCID}` INSIDE the task string, or pre-warm with one short run |
| 30-min startup stall, no output | a bad/slow node (the same job compiled instantly elsewhere) | propose the identical job on another node/partition as the first step, before deep-diving, and submit it after the owner's yes (one minute of queue) |
| restarts replayed on another mesh start from garbage anchors | restart files carry mesh-specific `nearest_node` and `restart_elem` ids | on any mesh swap (crop ↔ full) recompute `nearest_node` (`find_nearest_nodes`) and clear the carried `elem`; a restart holds the INTERNAL (rotated) frame, trajectory output the geographic one — seed a new run from the output converted to an `inicloud` file when the frames differ (`kelp-config`) |
| a harness bug is found late (e.g. the `--hint=nomultithread` OOMs) | conclusions drawn from contaminated legs | LIST the conclusions the bug touched and re-run them; tag contaminated metric rows `pre_fix` rather than deleting |
| `step=` phase ~**ms per particle** (io= small), run ~100× over budget | **highly NON-UNIFORM mesh** (resolution spanning orders of magnitude, e.g. structure-resolving 1.4 m → 1.4 km): `build_grid_hash` auto cell size bottoms out at its **0.05° floor**, one cell over the refined zone holds 10⁴–10⁵ nodes, and the in-kernel 3×3 nearest scan goes linear | set `mesh.grid_hash_cell_size` (degrees) in the config — `run_tracking` applies it reader-agnostically and pins it against later rebuilds; the loader also **warns with a suggested value** when it detects the case. See [references/grid-hash-cell.md](references/grid-hash-cell.md) for how to pick and validate it; the milder, warning-free version of the same problem (a regionally refined mesh, 100+ nodes per window, 25–45 % of the tracking phase) is caught by `utils/check_grid_hash_cell.py` |

## Convergence ladders (discrete event statistics)

Design the ladder before any data exists, by these rules, not by a threshold:
- Start AT the forcing cadence (never coarser) and halve down at least 4 rungs;
  expect to need 8–16× below the cadence (a measured ladder converged only ~16×
  below its forcing cadence).
- Size the ladder by bracketing, not by a threshold: run at least 4 rungs and
  report the rung where the published statistic stops moving. The tolerance that
  CALLS convergence is written down before the table is read (`kelp-analysis`,
  "Reading a dt ladder"); it never decides how many rungs to run. A measured ladder
  was non-monotone — two intermediate rungs were WORSE than the coarsest — so a
  two-rung criterion would have declared it converged at the coarsest rung.
- Name the event mechanism in the report only once measured (speed at the
  event, zero-velocity element census, the control arm) and fix one analysis horizon
  first — the reading rules are in `kelp-analysis` ("Reading a dt ladder").
- Wall time is linear in substeps: size the deepest rung against the cap (step 7).

## Campaign skeleton (copy these — they are the slots agents skip when left as prose)

The templates — `AGENTS.md`, the job-script preamble, the multi-shard `srun` block, the
dependency chain, the resumable leg driver `leg_driver.py`, the config check, the config
generator, the steady-step extractor, `LEDGER.md` and `REPORT.md` — are in
[references/campaign-templates.md](references/campaign-templates.md); copy them verbatim.

**Files the campaign folder contains before the first submit — tick each in the handback:**
`[ ] AGENTS.md  [ ] README.md  [ ] PLAN.md (its "## Expected throughput" section holds this row
verbatim: `| nodes ×2.2/×3.3/×5.2/×9.7 on 2/4/8/16 (slowest process) | intra-node split 16×8: step −22 % (fused kernel) | step ∝ mesh size ⇒ BUG → kelp-diagnose |`
and the memory table from step 5)  [ ] SUMMARY.md (every decision → the rule quoted or "own
judgement"; last line the launch gate)  [ ] LEDGER.md (empty table)  [ ] REPORT.md — written NOW
with the four headings of its template verbatim (a skeleton "to add later" is written 1 time in 3)
[ ] the generator  [ ] the seeds script  [ ] `leg_driver.py`  [ ] the sbatch scripts (preamble +
srun block + chain loop copied verbatim).

## Worked flow (generic, no paths)
Idea (region, 2D/3D, N particles, M years) → isolate folder + own clone → crop the
region to Zarr (right velocity family, native names, validate vs full) → **wait
for extraction to finish** → info probe → short smoke run → **load-probe at N**
(RAM / per-step / output size / CPU%) → set `output_freq` for storage → resumable
**dependency-chain** driver (restart every ~1/4 yr, resume from latest) → **smoke
the chain** (resume bit-for-bit) → submit the chain (**gated on the user's go**) →
concat segments → analyse in the experiment folder → **fill `REPORT.md`'s two
mandatory headings while the job ids are still on disk**.
