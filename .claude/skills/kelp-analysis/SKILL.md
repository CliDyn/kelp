---
name: kelp-analysis
description: Use when reading or computing anything from KELP trajectory output (a .nc or .zarr store, or a directory of part_*.zarr chain segments) — presence or +N-year maps, cohort or region fractions, section crossings and other events, comparisons of two runs, reversibility, dt ladders, cost or timing numbers from run logs — or when such a read is slow or a statistic looks wrong.
---

# Reading KELP trajectory output

This skill is about reading the trajectory file a run *wrote* (not the velocity
forcing). Everything here is measured/verified, not guessed.

**Reference files (read the one a section points to):**
- [references/events-and-crossings.md](references/events-and-crossings.md) — section crossings, event counts, event age.
- [references/timing-and-cost.md](references/timing-and-cost.md) — ranges, per-event costs, store or re-run.
- [references/reversibility.md](references/reversibility.md) — forward → backward round trips.
- [references/dt-ladder.md](references/dt-ladder.md) — convergence of an event statistic over dt.

## The one rule: read COLUMN-major (by time), never row-major (by particle)

**Why — how the writer chunks the file.** The Zarr output (`writers.py`) stores
each `(trajectory, time)` variable (format 1: `(trajectory, obs)`) with

    chunks = (traj_chunk, obs_chunk)
    traj_chunk = n_particles if n_particles <= 500_000 else 500_000   # TODO 5.4, 2026-08-16
    obs_chunk  = clamp( 4_000_000 // (traj_chunk * 8 bytes), 1, 256 )

**Outputs written BEFORE 2026-08-16** (and any run ≤ 500k particles): the
particle (trajectory) axis is a single chunk — it is never split. Only
the record axis (`time`; format 1 `obs`) is chunked, and at large particle counts `obs_chunk` hits
its floor of 1. So **one chunk = one full time column of ALL particles**
(e.g. ~7.9 M particles → a 63 MB chunk per time step) — everything below
applies fully.

**Outputs written after that, with > 500k particles**: the trajectory axis is
split into 500k-particle blocks (~4 MB chunks). A particle SUBSET then only
reads the blocks it touches — up to blocks-count× less I/O for compact
cohorts — but a column read still touches all blocks, so the column-major
rule below stays the right default; the block layout just stops punishing
subset reads. Check `ds["lon"].encoding["chunks"]` (or `zarr .chunks`) to see
which layout a given store has.

**Consequence (measured on the real 7.9 M-particle output):**

- Selecting a subset of particles — `ds["lon"].isel(trajectory=[...])` — **cannot
  read less than the whole column.** Zarr reads the entire 63 MB column for each
  time and discards the rest. **Fancy particle indexing buys ZERO I/O.**
- Reading 5000 scattered particles over a time window took **1.76 s**; reading
  the *whole* column (all particles) took **1.82 s** — identical. ~101 ms per
  time column (63 MB). The histogram + matplotlib for that same data ≈ **0.25 s**
  — i.e. compute is noise; **time is ~100% reading.**
- Therefore, **on single-block stores** (`chunks[0] == n_particles` — all
  pre-2026-08-16 outputs and every run ≤ 500k): **read each time column once**
  (`isel(time=k)`, or a contiguous `time=slice(...)` block; format 1: `obs`), pull **all**
  particles, and index the ones you want **in memory**. Fan one column out to
  *all* consumers — every cohort, every product — in a **single pass**. Never
  loop per-cohort (that re-reads the same columns N times), and never
  fancy-index particles hoping to "read less" — on THIS layout it buys zero.
  Inside the block loop the masks are VECTORISED over the block
  (`inside = (status_blk == ALIVE) & in_box(lon_blk, lat_blk)`; `np.add.at` /
  `bincount` by cohort id) — never a `for i in range(n_particles)` loop inside
  the column loop (7.9 M × 4 600 columns is 3.6 × 10¹⁰ Python iterations).
- **On split stores** (`chunks[0] == 500_000 < n_particles`): a COMPACT cohort
  (ids inside one or a few 500k blocks) genuinely reads less — slice the
  trajectory axis by whole blocks (`isel(trajectory=slice(b*500_000,
  (b+1)*500_000))`) and subset in memory within the block. A SCATTERED cohort
  still touches all blocks per column — fall back to the single-pass
  column-major pattern above; it remains the right default for global
  statistics either way.

```python
# ONE pass over the time window; every consumer is fed per column.
REC = ds["lon"].dims[1]                                 # record dimension: "time" (format 2) or "obs" (format 1)
for b0 in range(klo, khi, block):                       # block of record columns
    lon = ds["lon"].isel({REC: slice(b0, b0 + block)}).values   # (P, nb) — reads nb chunks
    lat = ds["lat"].isel({REC: slice(b0, b0 + block)}).values
    st  = ds["status"].isel({REC: slice(b0, b0 + block)}).values
    for j in range(lon.shape[1]):                       # loop over COLUMNS, never over particles
        k = b0 + j
        at_tau = (k_tau == k)                           # (P,) particles whose own +τ record is THIS column
        inside = (st[:, j] == ALIVE) & in_box(lon[:, j], lat[:, j])
        n_alive_tau  += np.bincount(cohort_id[at_tau], minlength=n_cohorts)
        n_inside_tau += np.bincount(cohort_id[at_tau & inside], minlength=n_cohorts)
        presence[decade_of[k]] += np.bincount(cell_of(lon[:, j], lat[:, j])[inside], minlength=n_cells)
```

Pick `block` so `n_particles * block * 8 * (#vars)` is a few GB (≈ 16–24 columns
at ~8 M particles, 4 variables).

This chunk shape is **good** for streaming writes (a whole chunk is compressed
once) and for whole-timeslice reads (`isel(time=k)` = exactly one chunk on a
single-block store, one chunk per 500k block on a split store — how the
statistics tables read). It is only a mismatch for *per-trajectory* reads on
single-block stores, which is why you read column-major there. (The
particle-axis chunking shipped 2026-08-16 — TODO 5.4; existing outputs keep
their old layout, do NOT re-chunk them.)

## Output may be a DIRECTORY of resumable-chain segments

A long run writes `part_*.zarr` segments (one per checkpoint chunk) plus a
sibling `<name>_restarts/` directory. To read the whole run, concatenate along
the record dimension (`time`; format 1 `obs`) — every segment of ONE format and
ONE `time` `units`, asserted:

- **Skip the segment still being written** (live-safe, partial run): keep
  segments whose start step ≤ the latest restart step (read precisely from the
  `_restarts` dir); without it, drop the newest segment.
- **A FINISHED archive that was merged or copied needs its restart marker** (a
  `_restarts` dir holding, or linking to, the final restart): without it the live-safe
  rule above silently drops the last segment (a merged multi-year archive lost its
  final segment this way). Before any statistic, assert that the last record is the run end:
  `assert abs(time[-1] - t_run_end) < cadence` (cadence = the record spacing's magnitude).
- **A merge of shards is validated against an independent run**, per particle: the
  correlation of a per-particle quantity (distance from the seed after a few days)
  with the same seeds' values in another run of the same setup (> 0.95 when the order
  is right, ~0 when it is shuffled) plus bitwise spot checks against the shards — not
  an absolute threshold picked in advance (real coastal currents exceeded a 1° limit).
  Gate the statistics job on the validation job with `afterok` — once, when the merge is
  made; an archive already recorded as validated needs only the last-record assert.
  The 0.95 gate is for shards of ONE setup. Two runs of different setups (another mesh, dt
  or forcing) are a comparison, not a merge: print the aligned correlation beside a shuffled
  control — the order is right when the aligned value is far above the shuffled one (≈ 0):
  ```python
  perm = np.random.default_rng(0).permutation(len(b)); print(f"aligned {np.corrcoef(a, b)[0, 1]:.3f} | shuffled {np.corrcoef(a, b[perm])[0, 1]:.3f}")
  ```
- **Concatenate cheaply** — and an empty `_restarts/` beside a FINISHED merged archive is
  the trap above: load every segment, then assert the end:

  ```python
  parts = [xr.open_zarr(p, consolidated=False, decode_times=False) for p in seg_paths]   # time in seconds, units kept in .attrs
  fmt = {int(p.attrs.get("kelp_output_format", 1)) for p in parts}; units = {p["time"].attrs.get("units") for p in parts}
  assert len(fmt) == 1 and len(units) == 1, f"segments mix formats {fmt} or epochs {units}"   # never merge across an engine upgrade
  ds = xr.concat(parts, dim=parts[0]["lon"].dims[1], data_vars="minimal", coords="minimal",   # "time" (format 2) / "obs" (format 1)
                 compat="override", join="override")                                          # release_* (no record dim) kept once
  cadence = float(abs(np.median(np.diff(ds["time"].values))))   # a magnitude: a backward run's labels count down
  ap.add_argument("--t-run-end", type=float, required=True)   # the `time` of the run's last record in seconds, from its config (formula below); no default: a guessed end hides the missing segment
  t_run_end = args.t_run_end                                    # never read from the archive; the check below is an assert, never a warning
  assert abs(float(ds["time"][-1]) - t_run_end) < cadence, "last segment missing"   # a FINISHED run; a live partial read stamps PARTIAL instead
  ```

  A check on the final gap or on monotonic time cannot see a missing LAST segment;
  only the run's own end, given from outside the archive, can.

  Without `override`, `concat` compares/loads the full `n_particles`-long
  `trajectory` coordinate **once per segment** — at millions of particles ×
  hundreds of segments that **OOMs and is slow** (verified failure). The
  trajectory coordinate is identical across segments (same pre-allocated particle
  set), so `override` is correct.

## What the arrays mean (enough to read correctly)

Authoritative description: `docs/io_and_time.md` §5–§6. Which format: `ds.attrs["kelp_output_format"] == 2`
(release 1.0 on); no such attribute and a record dimension `obs` = format 1 (every older store).

- `lon, lat, z, status` have shape `(trajectory, time)`; `time(time)` and `trajectory(trajectory)`
  are coordinate variables. `trajectory` holds GLOBAL ids (`particle_offset + i` in a shard).
- `z` is depth, **positive-down metres** — take `abs()` if you only want magnitude.
- `status`: **ALIVE = 1, BEACHED = 2, OUTSIDE = 3, NOT_RELEASED = 4.** Filter to
  `status == ALIVE` for presence maps / live paths. Positions of not-yet-released particles are
  NaN; dead and beached particles keep their frozen last position.
- `time`: seconds since the forcing's first RAW record, at the END of the recorded step, in BOTH
  time modes; `units = "seconds since <that record>"` + `calendar`. xarray decodes it to dates by
  default (cftime objects for `noleap`/`360_day`; `ds.sel(time="2005-03-01")` works); for
  arithmetic in seconds open with `decode_times=False` (`units` then sits in `.attrs`;
  `load_trajectories`, netCDF4 and zarr return raw seconds anyway). A forcing without a calendar
  epoch gives `units = "seconds"` and a `time_origin` attribute: seconds, no dates. Under
  `on_exhausted: cycle` the clock does not wrap (`forcing_cycle_period` attribute).
  - The record after global step `g` is `T_g0 + (g + 1)·dt·time_dir`, `T_g0` = the start of the
    whole run in those seconds: the `start_date` instant (interpolated mode), else the time of
    record `start_time_idx` (its window centre under `time_averaging: mean`), in both modes.
    So the run's last record time is `T_g0 + n_last·dt·time_dir`, `n_last` = the last multiple of
    `output_freq` ≤ `n_steps`. Cadence = `median(diff(time))`, seconds.
- **Release: `release_lon`, `release_lat` (geographic), `release_z` (m, positive down),
  `release_time` (the instant the particle starts moving), dims `(trajectory,)`** — the origin of
  every displacement and every age; the first record is one output interval AFTER release, so
  never use it, and no seed file is needed. Every chain segment carries the same values.
  `release_time` is the schedule rounded to `dt` and never NaN (a particle never activated has
  one after the last record); the positions are NaN only after a resume from a pre-format-2 restart.
- `vel_u, vel_v, vel_w` = the FLOW velocity (same sign forward and backward) the integrator evaluated
  at the step's START position and time (`rk45`: at the end of its last accepted substep); NaN unless the particle was ALIVE through the whole step
  (dying record, dead, beached, not released); no `vel_w` in 2-D. Sampled fields (`sample_vars`)
  are at the record's instant (discrete mode: the forcing record at the end of the step).
- `restart_*` variables exist only after a first checkpoint (`load_restart` raises `ValueError`
  without a valid one) — a generic reader skips them.
- xarray loads `lat`, `lon`, `z` as COORDINATES of `status` and the sampled fields (their
  `coordinates` attribute): `.load()` on `status` reads the positions too — on a big store take
  `.values` of the one variable, or open with `decode_coords=False`.
- **Format 1** (no `kelp_output_format`, dim `obs`): `time` has no epoch (`"seconds or file
  index"`) — interpolated mode: seconds since the forcing's first raw record (dates = that
  timestamp + seconds); DISCRETE mode: the forcing RECORD INDEX of the step's start, not seconds
  (last record `start_time_idx + (n_last − 1)·time_dir`). No `release_*`: the release position
  is the seed file's row (by `sid` in a pulsed run), the release record the first non-NaN
  record. `vel_w` = 0 in 2-D, the death record keeps a velocity
  (0 afterwards), backward runs store the time-reversed velocity; discrete sampled fields lag one
  record; `restart_*` always present (9.97e36 NetCDF / 0 Zarr without a checkpoint — read them
  only when `restart_valid == 1`).
- Backend by extension: `.nc` (NetCDF4 = HDF5) holds an exclusive lock and is
  **unreadable by other processes until the run closes it**; `.zarr` is
  lock-free and **readable mid-run**. Prefer `.zarr` for live inspection.

## Run the read on a real node

The column reads move tens of GB even when the final figure is small. The plan's
first line is the command that runs it:
```
srun --partition=<CPU partition> --cpus-per-task=8 --mem=64G python stats.py …
```
never the login node (a login-node OOM kill loses buffered stdout and looks like a
silent failure).

Open our Zarr stores with `consolidated=False` (they carry no consolidated
metadata; the default emits a warning per open).

## Statistics recipes (what a campaign got wrong at least once)

**Defaults and checks.** Which tables to write, how an event is defined and which
uncertainty to quote are DEFAULTS below. When the owner asks for another definition, a
different statistic or another interval, compute what was asked, reuse the reading
pattern and the masks, and add one line on how it differs from the default and why the
default exists, if that matters for the number. The CHECKS hold for any statistic:
column-major reading; the status mask the product needs (live positions and both
records of a crossing: `status == ALIVE`; state counts keep every status; `ALIVE|BEACHED`
is the beaching sensitivity); time from each particle's own release; a denominator on
every number; cohorts the run does not cover are dropped or censored; for a claim about
the whole run, an `assert` (never a warning) that the last record matches the run's own
end — a live partial product instead stamps `data through <date> — PARTIAL`.
The handback has a section `Defaults not used / checks skipped`, one line each:
`<default> — owner's choice: <what>` or `<check> — <condition not met>`.

- **Hand over only numbers the script printed**: quote them from its pasted output.
  Before the run, each is a `<file: column>` placeholder naming one value the script
  writes — never an illustrative value, a remembered one, or a peak read off a map.
  A sentence with a comparison word (dominate, rare, peak, stronger, earlier) is a
  claim: it gets its `<file: column>` placeholder too, or it is cut. Interpretation (a
  mechanism, "dominated", "most", "earlier") goes to a separate "to discuss" list in the
  handback, not into the caption file. Hedging ("may", "consistent with") does not turn a claim
  into a non-claim; a number typed into a string, or a unit changed by hand, counts as typed.
- **Every handed-over number gets a provenance row** in a table the script writes, five fields,
  `NOT RECORDED` where the source does not say; a number worked out interactively is first saved
  as a script. The engine commit comes from the data's own metadata (run store attributes, the
  metrics file, the ledger), never from the clone that reads it:
  ```python
  PROV_COLS = ["value", "producing_script_at_commit", "file_key_or_column", "command", "engine_commit_of_data"]   # one row per handed-over number
  ```
- **Round once; average raw values; read the table back.** Round at print time, once, from the
  stored full-precision value (a value already rounded is never rounded again). Means and ratios
  come from raw values (Σ passed / Σ items, or unrounded per-row values), never from a rounded
  column. Every table file is read back in the same script, by VALUES — an unquoted comma in a
  label shifts the columns and keeps the shape:
  ```python
  import io
  df.to_csv(out_csv, index=False)        # pandas / csv quote labels with commas; a "# data through …" line above the header is fine
  body = "".join(l for l in open(out_csv) if not l.startswith("#"))      # stamp lines above the header; a "#" inside a cell stays
  pd.testing.assert_frame_equal(pd.read_csv(io.StringIO(body), dtype=str, keep_default_na=False),
                                df.astype(str).where(df.notna(), "").reset_index(drop=True), check_names=False)   # numeric, text, mixed, wide (unstack/crosstab) tables; NA = empty cell
  ```

### Cohorts are arithmetic, not bookkeeping

- **Pulsed release** (`time.n_pulses`): pulses are contiguous id blocks in release
  order, so particle `i` has `pulse = i // n_seed`, `seed = i % n_seed`
  (`n_seed` = rows of the seed file):
  ```python
  P = n_seed * n_pulses; i = np.arange(P); pid = i // n_seed; sid = i % n_seed   # vectorised, never per particle
  # a store stamped independent_shards = 1 is SHARD-major: apply the formula per shard block
  # (attribute shard_counts; each shard is pulse-major over its own seed block), or read
  # each particle's release_time (format 1: its first non-NaN record)
  z0 = ds["release_z"].values   # per particle (format 2); format 1: seeds[sid, 2] — seed metadata by sid, never by the particle id
  ```
  Decade/season come from the pulse's release
  date (`release_time`), the depth band from the release depth (`release_z`; format 1:
  seed-file column 3, the same in every pulse). No cohort table is needed; carry seed metadata (`kind`,
  `pair_of`) next to the seed file and **exclude illustrative seeds** from every
  statistic — when such metadata exists; without it the handback says "no seed metadata:
  none excluded" and the script runs (never a required file, never a refused extra column).
- **Release** of a particle = `release_time` / `release_lon/lat/z` of the store (format 2);
  format 1: its first non-NaN record (dormant `NOT_RELEASED` positions are written as NaN —
  up to one record interval after the true release); a restart file gives the step directly
  (`restart_release_step`).
- **Derive the time axis from the data, never from the cadence.** A resumed
  chain written before 2026-09-22 has *segment-local* record cadence: one chain wrote
  18 records per 91-step segment at 5 days with a **6-day interval at every one of its
  hundreds of seams** (since then the cadence follows the global step; the rule below
  holds for both).
  `k = k0 + years·365/cadence` is wrong there; use
  `np.searchsorted(time, t_release + tau)` per particle (`time` and `release_time` in seconds:
  `decode_times=False`; a backward run counts DOWN — with `sgn = np.sign(time[-1] - time[0])`,
  `np.searchsorted(sgn * time, sgn * t_release + tau)`), and print the REALISED lag
  next to every table (median and range of `sgn * (time[k] - t_release) - tau`). The calendar
  anchor is `time.attrs["units"]` + `calendar` (format 2; to dates:
  `cftime.num2date(t, units, calendar)`); format 1 — see "What the arrays mean".

### Two products, three denominators

- **Presence integral vs snapshot — the biggest correctness trap.** `PROB`
  accumulates density at *every* record in each particle's 0→τ window and is
  dominated by slow early residence near the source ("everyone stays home");
  `PROB10` takes *one point per particle* at exactly +τ from its own release.
  They gave opposite conclusions on the same run. Keep
  both, label them, never use the presence integral for a dispersal claim.
- **State the denominator on every number.** `export_frac = N_exited / N_born`,
  `region_frac = n_R / N_born` ("where they go": dead and exited particles are rows of their own, so
  the rows add to 1), `alive_frac = N_alive / N_born`; the share among survivors `n_R / N_alive` goes
  beside it, labelled "of survivors".
  Born-only denominators mean later pulses never retro-change an earlier cohort.
  A particle that has not yet reached lead time τ is **censored**
  (Kaplan–Meier), not "retained"; drop cohorts whose +τ window the run does not
  cover — check the availability matrix before a table row (the cohorts released
  within τ of the end of the run once showed a spurious 8–9 % in every band
  because only their earliest releases had reached +τ and the rest were
  silently counted as not-retained).
- **Mask by status, always.** A dead (`OUTSIDE`) particle keeps its frozen last
  valid position — inside the box, it counts as "inside" unless you require
  `status == ALIVE`. `BEACHED` is the one reversible status (resuspension);
  report `ALIVE` and, when beaching was on, `ALIVE|BEACHED` as a sensitivity.
- **Before any table row, fill the lines that apply to the statistic in the plan**
  (`n/a — <reason>` for one that does not apply):
  ```
  N_born = <…>;  N_alive = <…>;  export_frac = N_exited/N_born;  region_frac = n_R/N_born (n_R/N_alive beside it, "of survivors")
  AVAILABILITY: cohorts whose latest release + tau exceeds the last record are DROPPED, not averaged
  CENSORED: a particle not yet at tau is censored (Kaplan–Meier), never "retained"
  LOOP BOUND: k_end = max over cohorts of (k0 + tau), NOT n_records of the first cohort
  UNCERTAINTY: <method> — default the spread across pulses; a requested binomial interval is given, named, with the shared-field caveat
  AGE HORIZON: H = <yr> — releases with release + H ≤ last record, events with age ≤ H; tail beyond H from releases followed <H_TAIL> yr
  DATA THROUGH: <YYYY-MM-DD> — PARTIAL until the chain is finished
  ```
  (the last cohort silently never reached its +τ once).
- **Default uncertainty on a cohort fraction: the spread across its pulses** (or a
  block bootstrap over pulses): the same seeds repeat in every pulse and share one
  velocity field, so particles within a pulse are not independent samples. If the owner
  or a reviewer asks for a binomial interval, give it, name the method (Wilson,
  Clopper–Pearson) and state in one line that it is too narrow for that reason; the
  pulse spread may go beside it as additional information.
- **Two rates in a pulsed run with outflow:** "particles/s" is ambiguous —
  quote allocated AND active particle-steps (one pulsed run: 43 % of allocated
  particle-steps were dead; max alive 57 % of allocated).
- **Events versus delayed snapshots; section crossings.** An event count is on the EVENT time
  axis, a +τ map on the RELEASE axis — every table column and caption names the decade as
  release or crossing decade. The rules, the whole crossing loop (masks, interpolation, release
  age, the default every/first tables, calendar columns), the age link to a +τ map, the AGE of
  an event in a pulsed run (one horizon H, the tail beyond H) and a calendar-year event series
  (age cut, share removed per year) are in
  [references/events-and-crossings.md](references/events-and-crossings.md) — its two blocks after
  the loop are the code for both. By default every count table keeps `dir` (S and N are not
  summed), and the script prints — the caption states — that an excursion shorter than the
  record interval is missed.
- Bin on an equal-area grid, or divide lon/lat counts by the true cell area,
  at high latitude (a 1° cell shrinks ×3 across 70–80 °N); the mesh-element
  alternative is `compute_presence_density_mesh(..., normalize="per_area")`.

### Comparing two runs or two methods

- **A run invalidated by a bug is superseded, not compared.** The report gives
  the corrected run's numbers only; the old-vs-new comparison goes to a separate
  review-only file, and the report does not mention the old archive.
- **Statuses bitwise, positions statistically.** Bitwise equality is the proof
  for a "same results, faster" change (sharding, layouts, `exact_nearest`);
  anything else is compared on the **common survivors** with the survival
  difference reported separately. "Statistically equivalent" = three numbers:
  survival, a displacement quantile pair (median + p90), a presence-map overlap
  (`edge_leastsq` vs `leastsq` on one regional run: alive 31,409/31,414, median
  349.6/348.4 km, overlap 0.991 at 2° bins).
- **Ensemble statistics of two runs** (mean squared displacement, dispersion, displacement
  quantiles, FSLE pairs) are compared over the particles alive in BOTH runs at that record; each
  run's own-survivor value is at most a labelled sensitivity — its ratio measures who died, not
  how particles spread. Survival stays per run; the survival DIFFERENCE between the runs is its own
  number beside the ratio. The displacement origin is
  the release position `release_lon/lat` of each store (format 1: the seed file; the first
  record is one output interval after release). A
  statistic averaged over scales or classes is quoted with its range and the count per class.
  ```python
  sgn = np.sign(time[-1] - time[0])                                               # -1 in a backward run: its labels count down
  k = int(np.searchsorted(sgn * time, sgn * t_release + tau))                     # lead record from the time axis (seconds)
  common = np.logical_and.reduce([st[r][:, k] == ALIVE for r in runs])            # alive in every compared run
  print({r: int((st[r][:, k] == ALIVE).sum()) for r in runs}, "common:", int(common.sum()))   # beside every ratio
  msd = {r: np.mean(gc_km(rel_lon[r][common], rel_lat[r][common], lon[r][common, k], lat[r][common, k]) ** 2) for r in runs}   # origin = ds_r["release_lon"/"release_lat"].values
  surv = {r: float((st[r][:, k] == ALIVE).mean()) for r in runs}; print("survival", surv, f"difference {100 * (surv[runs[1]] - surv[runs[0]]):+.2f} pp")   # one release; pulsed: divide by the particles released by record k
  ```
  The owner may ask for each run's own-survivor value (an earlier table, FSLE pairs per run): give it,
  with the counts per run and the label "own-survivor sets — includes who died"; the common-set value
  goes beside it. A number the owner asks for is never held back for a check the owner did not ask for.
- **Trajectory errors:** median + p90 + p99 + max, normalized TWICE — by path
  length and by the local grid cell (median element edge); **cut by dynamical regime** — a global median hides the effect
  (a dt comparison whose global median was kilometres hid a regional median
  ten times larger and a tail of thousands of kilometres). Publish a per-trajectory error
  (paired particles of two runs) only next to its chaos floor (perturbed-seed twin) and protocol
  floor (same scheme, other frame). The ensemble ratio above is reported on its own — a
  perturbed-seed twin's ratio may go beside it as its noise level, never instead of it.
- Hold the output axis fixed across a swept parameter (`output_freq = 86400/dt`)
  so differences are element-wise; a plateau under refinement = another error
  source. On-edge ties are a bounded class: barycentric µm (max 1.2 mm/yr),
  leastsq-family metres (onset ~11 m, end max 472 m) — not a bug.
- Individual decadal trajectories are not reversible or comparable across any
  code/grid change (Lagrangian chaos); use ensembles.
- First-passage events need a dwell threshold (e.g. ≥ 30 d below the depth)
  and exits are attributed to the last in-domain gateway, not the bare status.

### Testing reversibility and reading a dt ladder

- **Reversibility** (forward → backward round trip): short deterministic legs, a calibrated floor,
  a cost slot — [references/reversibility.md](references/reversibility.md).
- **A dt ladder** (convergence of an event statistic): transcribe the rungs verbatim, state the
  tolerance before reading, one MECHANISM line — [references/dt-ladder.md](references/dt-ladder.md).

### Timing numbers from run logs

Steady step = the **median** of `step=` over steps ≥ 2 with `io < 0.5 s`, of the
worst process; never "mean of the last 5 steps" (it caught a snapshot-read step
and inflated a reported number: 1.187 → 0.997 s clean). Tag contaminated rows
in the metrics file (`pre_fix`) instead of deleting them; logs → metrics table →
figures as three scripts so any reported number regenerates with one command.

- **Ranges, per-event costs, the store-or-rerun break-even and phase regressions:**
  [references/timing-and-cost.md](references/timing-and-cost.md).

### Provenance and partial runs (output contract)

Every table and figure this plan produces carries the literal line
`data through <YYYY-MM-DD> — PARTIAL` until the chain is finished. A value used only for a stamp
(the data-through date, the forcing's first timestamp, a released count in a title) is optional:
read it from the store or its config, or stamp `NOT GIVEN` — it never stops the script. Three more
obligations, each in its own shape in the PLAN:
```
[ ] chain-smoke: the pipeline was run end-to-end on <path of a chain-smoke trajectory> BEFORE real output existed
| provenance | counts/histograms: ALL particles | per-particle tracks: SUBSAMPLE (cache_every = <N>) |
Domain review of this PLAN (not only the code): <name>, <YYYY-MM-DD>
```
never infer the statistical sample size from the cached tracks.

- A cache's per-particle tracks are a **subsample** (`cache_every`); its
  counts/histograms are all-particle — never infer the statistical sample size
  from the spaghetti density. Before re-deriving a suspected shortcut, check
  provenance by counts (the "2 M subsample" worry was wrong; the caches were
  all-particle).
- Build the analysis **partial-safe from day one**: skip the segment being
  written, stamp every table/figure `data through <date>` + `PARTIAL`. Smoke
  the whole pipeline on a chain-smoke trajectory before real output exists.
- Get a **domain review of the analysis plan** (not only the code); keep the
  old product as an appendix when the question changes (e.g. an Eulerian
  "where at τ" product superseded by event-based fates).
