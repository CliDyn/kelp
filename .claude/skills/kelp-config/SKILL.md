---
name: kelp-config
description: Use when a KELP YAML run configuration is written, edited, reviewed or debugged — mesh and velocity files, variable names, 2-D or 3-D, backward tracking, diffusion, beaching, sampling, the time window — or when KELP opened the wrong file or variable, ignored a key, or failed at startup on a config value, even if the user never says config.
---

# Writing a KELP run config

The authoritative option-by-option reference is **`docs/configuration.md`** in
this repository — read it before answering detailed questions about any key
(defaults, validation, file-discovery rules). This skill is the workflow on
top of it. For the *cost* of each choice — run time, thread count, I/O, the
output-writer (Zarr vs NetCDF), and sizing recipes — see **`docs/performance.md`**.
If this config is one piece of a **big / long / multi-job experiment** (millions
of particles, a run that outlives one allocation, or a mesh needing regional
cropping), set it up with the **kelp-experiment** skill — it owns the
campaign order-of-operations and the setup pitfalls.

## Core principles

1. **Pin files explicitly.** The house preference for this project: a config
   should state outright which files will be opened, not rely on directory
   discovery. Use `data.ncinput` + `ncinput_v` + `ncinput_w` pointing at
   concrete files. FESOM2 directory mode (`ncinput: some/dir/`) silently
   takes the *first* lexicographically-sorted `u*`/`v*`/`w*`/`ssh*` match and
   merges every other readable NetCDF/Zarr in the directory — prefix globs
   mean `velocity.nc` matches `v*` and `wind.nc` matches `w*`. The directory is
   the right input in two cases: the multi-year FESOM2 layout `<var>.fesom.<YYYY>.nc`
   read by EXACT names (`mode.var_u`/`var_v`/… set; `kelp-run`, "Multi-year FESOM2
   output"), and `sample_vars` from per-variable files (temp/salt) — then warn to keep
   the directory clean. The prefix-glob risk applies to every other directory.
2. **There is no typo protection.** Unknown or misspelled keys (including
   wrong case, e.g. `Fdrift2d`) are silently dropped and the default is used.
   After writing a config, re-check every key against `docs/configuration.md`.
3. **Make the implicit explicit.** Always set `data.time_averaging`. FESOM2
   output is time-averaged at EVERY cadence, daily included: write
   `time_averaging: mean` (its timestamps are the window END). `instant` is for
   true snapshot archives only (FESOM-C); if you cannot prove the file holds
   snapshots, write `mean`. Omitting it only produces a guess and a warning. Prefer
   explicit `numerics.dt` and `mode.var_u/var_v/var_w` when the
   dataset uses non-default names; remember FESOM2 silently runs *without*
   vertical velocity if the W name doesn't resolve. **But do not declare an
   optional variable the store lacks**: setting `var_eta: ssh` against a
   z-level store without ssh turns an optional input into a hard dependency
   (startup error), while leaving it unset gives eta = 0, which z-level
   advection ignores.
4. **Configs are build artefacts.** For any campaign (a sweep, a chain, more
   than two configs) write a generator script and regenerate; never hand-edit
   a generated YAML or overwrite an old run's config — a protocol change is a
   new run tag. **Validate every generated config with `load_config` before
   `sbatch`** and assert the resolved fields equal the intended matrix (a
   dropped key silently becomes its default — 16 identical runs that look
   fine); a parse error must never be discovered by SLURM. Use absolute paths
   in generated configs, and write floats with a dot AND a signed exponent
   (`1.0e-5`, `100.0`) — PyYAML reads `1e-5` as a STRING, so a numeric sweep
   axis can arrive as text; assert the resolved types, not only the values.

## Workflow

1. **Interview / inspect.** Establish: model family (FESOM2 → z-levels,
   FESOM-C → sigma), 2D or 3D, forward or backward, mesh path, velocity
   file(s), particle seed file, run length, output destination. If the data
   is on disk, look at it (`ncdump -h`, `ds.data_vars`) instead of guessing
   variable names and time cadence.
2. **Draft the config** from the template below. All five sections must be
   present; an intentionally empty one must be written `section: {}` (a bare
   `section:` crashes). Relative paths resolve relative to the *config file's
   directory*, not the working directory.
3. **Choose the time regime deliberately.**
   - Discrete mode (`temporal_interp: false`): one file snapshot per step;
     set `dt` equal to the file step (or omit `dt` for auto-detect).
   - Sub-stepping (`temporal_interp: true`): `dt` < file step, velocity
     interpolated linearly between snapshots.
   - `n_steps` counts *integration* steps (dt-sized), not file records.
   - Short forcing + long run → `on_exhausted: cycle`.
4. **Check cross-key constraints** (full list in `docs/configuration.md`):
   - `var_u: unod`/`vnod` (node-layout velocities), HEALPix and NEMO require
     `numerics.uv_interp: barycentric` — `leastsq`/`edge_leastsq` fail at
     startup (element-centre math). FESOM-C `u2d`/`v2d` and FESOM2 `u`/`v` are
     ELEMENT layout: keep `leastsq` (default, fused); decide from the variable's
     dimension (`elem` vs `nod2`), never from the name.
   - 3-D horizontal diffusion: only `scheme: uniform` works; `em`/`m1` raise (vertical
     Az mixing, next bullet, is separate and supported).
   - Vertical mixing = `diffusion.az_enabled: true` + `az_const` (m²/s; or
     `var_az` on FESOM2 z-levels): 3-D only, requires `enabled: true`, rejected
     with `em`/`m1`; `az_const: 0` with no `var_az` is silently dormant.
   - `em`/`m1` + `rk2`/`rk4` requires explicit `composition: split_rk`.
   - `em`/`m1` + `euler` (auto-`combined`) requires `uv_interp: leastsq` —
     `barycentric`/`edge_leastsq` fail at startup, before the writer.
   - `elem_search: walk` + `uv_interp: leastsq` warns once (the fused
     leastsq kernels keep their own search) — the walk pays off with
     `barycentric`/`edge_leastsq` on 2-D element layout; on 3-D node-layout
     `rk2` it DISABLES the fused kernel — keep `local` there.
   - 3-D node-layout + `rk2`: set `w_interp: barycentric` (or `nearest`)
     explicitly — the default `leastsq` silently takes the slow unfused path.
   - 3-D file whose vertical velocity has no `positive` attribute: write
     `mode.w_positive` explicitly, with the regional-mean number that decided it in a
     comment (equatorial upwelling; see `kelp-run` step 4).
   - `max_levels` is a RAM lever that inserts an artificial bottom; a depth
     study keeps it `null` (full column).
   - Multi-year exact layout works for `.zarr` stores too (the tail must be
     exactly `<YYYY>.nc` or `<YYYY>.zarr`); anything else falls back to the
     first-sorted prefix glob silently.
   - **A continuation, or a restart moved to another mesh** (crop ↔ full): anchors, frames,
     provenance and the input cost — [references/continuation.md](references/continuation.md).
   - `beaching.use_eta: true` requires an ssh/eta/zos variable in the data.
   - **Pulsed release window — computed and ASSERTED, never eyeballed:**
     `n_steps = ((n_pulses − 1) · pulse_interval + drift_after_the_LAST_pulse) / dt`.
     `drift_after_the_LAST_pulse` = the tracking length of ONE realisation ("30 days
     per realisation" → 30 d), NOT the run length minus the release span. Worked:
     10 pulses / 3600 s / 30 d / dt 600 → last pulse at step 54, 4,320 steps of
     drift after it, `n_steps = 4,374` (4,320 alone is one realisation's drift,
     not the run). Paste into `validate.py` (`DRIFT_S = 30 * 86400`; `FILE_DT` = the
     record spacing in s; `START_IDX` = `c.time.start_time_idx`, or the record a
     `start_date` resolves to):
     ```python
     import math
     exp = ((c.time.n_pulses - 1) * c.time.pulse_interval + DRIFT_S) / c.numerics.dt
     assert c.time.n_steps == int(exp), (c.time.n_steps, exp)
     # forcing spans it: n_steps counts dt-steps, not records
     need = (math.ceil(c.time.n_steps * c.numerics.dt / FILE_DT) + 1   # interpolated: the span + the closing bracket record
             if c.numerics.temporal_interp else c.time.n_steps)        # discrete: one record per step
     if c.mode.time_dir > 0:
         assert n_file_records >= START_IDX + need, (n_file_records, START_IDX, need)
     else:                                     # backward: the records BEFORE the start
         assert START_IDX + 1 >= need, (START_IDX, need)
     ```
   - `start_date` must hit a file timestamp exactly unless
     `temporal_interp: true`; with `time_averaging: mean` give the raw
     end-of-window timestamp from the file, not the midpoint.
   - `checkpoint_freq` does nothing without `data.output_path`. It only ever
     keeps the **latest** checkpoint (resume from where the run stopped). A run
     too long for one compute job's wall clock instead needs the driver-level
     **resumable-chain** pattern (`write_restart_file` → segment outputs +
     numbered kept restarts + resubmit + concat) — that is a small run script,
     not a config key; see `docs/running.md` and `docs/io_and_time.md` §6.
5. **Verify, don't assume — and never launch.** `load_config` and `kelp info`
   are free (the mesh loads, `ncinput` opens, the time range prints — it does not
   merge `ncinput_v`/`ncinput_w`). `kelp run` — a smoke run included — and
   `sbatch` spend compute and need the user's explicit yes: PROPOSE the smoke
   run (`n_steps: 5`, `-v`; read the startup lines for the resolved dt, file
   step, time-averaging shift, diffusion composition, files used) in your
   handback, do not execute it. The handback ends with the literal line
   `Launch gate: nothing was run; say "yes" to run the smoke test.`

## Template (explicit-files style)

```yaml
mode:
  Fdrift2D: true          # 3D: Fdrift2D: false, Fdrift3D: true
  time_dir: 1.0           # -1.0 = backward
  # var_u: u              # set when the dataset uses non-default names
  # var_v: v
  # var_w: w              # 3D: MUST resolve — FESOM2 runs silently without W otherwise
  w_positive: auto        # 3-D, file without a `positive` attribute: replace auto by up/down from the kelp-run step-4 sign check (w_sign.py); command.sh refuses auto
  # var_eta: ssh          # ONLY if the store has it: declaring it makes it required
  # max_levels: null      # 3D: null = full column; N inserts an artificial bottom at interface N
  track_failure_reason: true  # per-particle cause of death (crop edge vs interpolation failure); cheap
  # sample_vars: [temp, salt]   # node-based scalars sampled along tracks

numerics:
  dt: 3600.0              # seconds; omit for auto-detect from the file
  integrator: rk2         # euler / rk2 / rk4
  prefetch_size: 8        # >= 2 keeps reads overlapped on multi-year / 3-D input; pchip needs >= 4; shared_fields_dir forces 1
  output_freq: 1           # record every N steps
  # STORAGE = n_steps / output_freq × n_seeds × n_pulses × 66 B = <TBD> GB vs QUOTA <TBD> GB  ← compute it HERE; raise output_freq ONLY if the brief leaves the cadence free — a cadence the brief fixes is never coarsened silently: put both numbers in the launch gate for the owner; no key drops vel_* (strip in post-processing)
  temporal_interp: false  # false = DISCRETE: one step per file record, dt MUST equal the file step and n_steps counts RECORDS; true → dt may be smaller than the file step
  checkpoint_freq: 100    # > 0 when the run may outlive its job or is costly to repeat (needs data.output_path); a run of minutes may keep 0 = off
  # uv_interp: leastsq       # leastsq (element-layout) / barycentric (node-layout) / edge_leastsq (element + edge neighbors, FESOM-style)
  # w_interp: leastsq        # 3D node-layout + rk2: set barycentric/nearest explicitly to engage the fused 3-D kernel
  # edge_leastsq_bc: freeslip # edge_leastsq coastal ghosts: freeslip / noslip
  # elem_search: local       # walk = track the containing element across edges (identical results, faster; pair with barycentric/edge_leastsq)
  # periodic_lon: [0.0, 10.0] # idealized channels ONLY: post-step fold into the window; the MESH must carry a ghost band past the window (fold adds no seam connectivity); window strictly inside (-180,180); rejects mesh.rotation; periodic_lat likewise for doubly-periodic boxes
  # output_flush_every: 0  # NetCDF only: buffer N records then close/reopen so the file is readable MID-RUN (0 = held open until close; .zarr is already lock-free)
  # big-mesh I/O (default off = byte-identical; halve memory + bulk-read):
  velocity_float32: false  # a MEMORY knob only — the step is 15–20 % SLOWER; never set it to make a run faster
  # NOTE mesh.rotation + diffusion: use ISOTROPIC Kh (kh_zonal == kh_meridional; equal fields for em/m1) — an unequal pair is applied along the ROTATED axes and the run warns (docs/coordinate_frame.md §3). Spatially-varying-but-isotropic Kh is fine.
  # EXAMPLE ONLY (add Kh only when the task asks for horizontal spreading; vertical mixing alone = kh 0.0):
  # diffusion: {enabled: true, scheme: uniform, kh_zonal: 100.0,
  #             kh_meridional: 100.0, seed: 42}

mesh:
  type: fesom2            # fesom2 / fesomc
  meshpath: path/to/fesom.mesh.diag.nc
  # rotation: fesom2      # default when the release or the expected drift reaches poleward of 60 °N/S (FESOM2/HEALPix): the lon/lat metric degenerates near the pole; output stays geographic
  # grid_hash_cell_size: 0.5   # speed only: nodes per nearest-node hash window — pick with utils/check_grid_hash_cell.py
  # exact_nearest: true        # default true (exact anchor for any cell); false + grid_hash_cell_rule: legacy = pre-2026-09-14 anchors bit for bit

data:
  ncinput:    path/to/u.fesom.1958.nc     # pin files explicitly
  ncinput_v:  path/to/v.fesom.1958.nc
  # ncinput_w: path/to/w.fesom.1958.nc    # 3D only
  inicloud:   particles.dat                # lon lat [depth] [release_sec]
  output_path: trajectories.zarr           # .zarr or .nc; empty = in-memory
  time_averaging: mean    # ALWAYS set: mean (averages) / instant (snapshots)

time:
  start_date: "2005-01-01"   # temporal_interp: the run starts at this instant; discrete: must equal a RAW file timestamp (list them: `ncdump -t -v time FILE`; `kelp info` shows only the first and last)
  # start_time_idx: 0     # alternative ONLY when you have read the index off the file's time axis; a multi-year directory starts at its FIRST year — index 0 is NOT your date
  n_steps: 720
  on_exhausted: stop      # cycle = wrap around (periodic forcing)
  num_runs: 1
  # n_pulses: 5           # delayed release: re-release the seed set 5 times ...
  # pulse_interval: 86400 #   ... pulse_interval seconds apart (total = seeds*n_pulses)
```

Delayed/staggered release (one place, many times → a pulsed source): set
`time.n_pulses` (>1) + `time.pulse_interval` (seconds) to re-release the whole
seed set as timed pulses, and/or give each seed an individual birth time via an
optional 4th `inicloud` column `release_sec` (offset in seconds from the run
start). The two add up; `n_pulses=1` + no column = off (byte-identical). Frozen
particles are written as NaN with status `not_released` until they activate.
Size the window: `n_steps · dt` must cover the LAST release time plus the drift
you want after it, or late pulses never activate (the run warns).

## Beaching on a coastal mesh

Counting strandings (which criterion can fire on this mesh, the default block, coast versus open sea) and the
"revert and retry" setup, with the rim and seed counts for `validate.py`:
[references/beaching.md](references/beaching.md).

## Output contract (what a config task hands back)

1. `config.yaml` — generated or hand-written, all five sections, keys spelled per
   `docs/configuration.md`, absolute paths, floats as decimals.
2. For a generated or campaign config (a sweep, a pulsed release, a continuation):
   `validate.py` — a FILE in the run folder (not a block inside a `.md`), ending in
   `print("VALIDATED")`, run once before the handback and its printed output pasted
   into `SUMMARY.md`; a `<…>` placeholder left in any NUMERIC or DATE key is a
   failed handback (paths may stay placeholders, numbers may not): `load_config` + resolved-TYPE
   asserts on every numeric field you set + the pulsed-window assert + the
   forcing-span assert + the time-regime assert
   `assert c.numerics.temporal_interp or c.numerics.dt == FILE_DT, "discrete mode: dt must equal the file step"`
   + (coastal beaching) the rim/seed block of references/beaching.md + the storage assert
   `gb = c.time.n_steps / c.numerics.output_freq * n_seeds * c.time.n_pulses * 66 / 1e9;
   assert gb < QUOTA_GB, f"{gb:.0f} GB of trajectory output vs quota {QUOTA_GB} GB — raise output_freq only if the brief leaves the cadence free, else ask the owner"`.
   For a CONTINUATION of a finished run, `validate.py` is required and also asserts
   against the original config:
   ```python
   c = load_config(p); o = load_config(ORIGINAL)          # the finished run's config
   assert c.mode.Fdrift3D == o.mode.Fdrift3D              # Fdrift3D is NOT implied by Fdrift2D: false
   assert c.numerics.diffusion.enabled == o.numerics.diffusion.enabled
   assert c.numerics.diffusion.seed == o.numerics.diffusion.seed
   assert isinstance(c.numerics.dt, float) and isinstance(c.time.start_time_idx, int)
   assert c.mesh.rotation == o.mesh.rotation and c.data.time_averaging == o.data.time_averaging
   ```
3. `SUMMARY.md` — one line per decision with the rule that drove it (quoted) or
   "own judgement"; its last line is the launch-gate line below (a gate stated
   only in chat is invisible to whoever reads the folder later).
4. The last line of the handback AND of `SUMMARY.md`, as a literal code line — it
   states what did NOT happen; reproduce it verbatim, never replace it with a
   result line ("smoke test successful" is a launch, not a gate):
   ```
   Launch gate: nothing was run; say "yes" to run the smoke test.
   ```

## Review checklist (when checking an existing config)

- Every key spelled exactly as in `docs/configuration.md`? (silent-drop risk)
- All five sections present, empty ones as `{}`?
- Files pinned explicitly, or is directory discovery intentional and the
  directory clean?
- `time_averaging` set explicitly?
- dt ↔ `temporal_interp` consistent (discrete mode wants dt = file step)?
- 3D: W variable actually resolves (`var_w` or `w`/`wo`/`wvel` for FESOM2,
  `w` = `Wvel` for FESOM-C — `w_cart` is refused), and its sign convention
  is the file's: `mode.w_positive` written, with the sign-check number in a
  comment (`auto` falls back to the family default when the file has no
  `positive` attribute, and that fallback is not a decision; `down` for
  `utils/reconstruct_w.py` output written before 2026-09-23), `max_levels` ≥ 2?
- Cross-key constraints from step 4 above?
- Paths correct *relative to the config file location*? (absolute in generated configs)
- Generated by a script and re-validated with `load_config` (resolved fields =
  intended values AND types)? No optional variable declared that the store lacks?
- Pulsed: `n_steps` computed from the formula above; `checkpoint_freq > 0`;
  `track_failure_reason: true`? Beaching on a coastal mesh: rim counted, gates
  consistent, `zone_depth` a plain decimal?
