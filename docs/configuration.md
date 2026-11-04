# KELP configuration reference

> **Covers:** every option of the run-config YAML (all five sections plus the
> `diffusion`/`beaching` sub-configs), the input-file discovery rules, and the
> common pitfalls.
> **Does not cover:** how the options are consumed at run time (see
> `io_and_time.md`, `time_integration.md`, and the subsystem docs); writing a
> config from a task description (see the `kelp-config` agent skill).
> **Code:** `src/kelp/core/config.py:load_config`,
> `src/kelp/core/config.py:AppConfig`.
> **Verified at:** see `docs/doc_map.yaml` (stamped via check_doc_sync --bless).

This document describes **every option** of the YAML configuration file consumed by
`kelp run config.yaml`, `kelp info config.yaml`, and
`kelp.tracking.run_tracking(load_config(...))`. It also documents exactly
**which input files get opened** for each setting, so a config can be written to
pin files explicitly instead of relying on automatic discovery.

Source of truth: `src/kelp/core/config.py` (schema and validation),
`src/kelp/readers/fesom2.py`, `src/kelp/readers/fesomc.py` and
`src/kelp/readers/fesom14/reader.py` (file discovery and variable resolution),
`src/kelp/tracking.py` (how the values are consumed). A copy-paste template lives in `config.yaml` at the repo root;
runnable examples in `examples/config*.yaml`.

---

## 1. How the config is loaded

```yaml
mode:     { ... }   # tracking mode, variable names, sampling
numerics: { ... }   # dt, integrator, interpolation, diffusion, beaching
mesh:     { ... }   # mesh type and path
data:     { ... }   # input/output files
time:     { ... }   # time window, realizations
```

Rules enforced by `load_config` (verified against the code):

1. **All five sections must be present.** A missing section raises
   `ValueError: Config missing sections: [...]`.
2. **A present-but-empty section must be written `{}`.** Writing a bare
   `mode:` with nothing under it parses to `None` and crashes with
   `AttributeError` — write `mode: {}` instead.
3. **Unknown keys are silently ignored — including misspelled or
   wrong-case keys.** `Fdrift2d: true` (lowercase `d`) is *not* an error; it is
   dropped and the default `Fdrift2D: true` is used. There is no typo
   protection: double-check key spelling against this document.
4. **Relative paths are resolved relative to the config file's directory**,
   not the current working directory. `~` is expanded. Applies to
   `mesh.meshpath`, `mesh.levels` (when it is a path), `data.ncinput`,
   `data.ncinput_v`, `data.ncinput_w`, `data.ncinput_eta`, `data.inicloud`,
   `data.output_path`. `numerics.shared_fields_dir` is NOT resolved: a
   relative value is taken from the current working directory (`~` is
   expanded, since 2026-11).
5. Validation of values (allowed enum strings, ranges) happens at load time in
   each section's `__post_init__`; invalid values raise `ValueError`
   immediately, before any data is opened. Since 2026-10-04 this also covers
   `mode.time_dir`, `numerics.integrator`, `numerics.w_interp`,
   `data.time_averaging`, `time.on_exhausted`, `time.duration_days` without
   `start_date`, `mode.sigma_layer` together with `mode.sigma_mean`, quoted
   strings where a list is expected, and the grid-hash cell bound. Checked
   when the run starts or when first used: `mesh.type` (the reader registry),
   `n_pulses`/`pulse_interval`, and combinations that need the dataset (the
   velocity layout, the time axis, the presence of variables).
   An empty section (`mode:` with nothing under it) loads with its defaults;
   a section that is not a mapping (`mode: false`) is refused.

---

## 2. `mode` — tracking mode and variable names

| Key | Default | Meaning |
|---|---|---|
| `Fdrift2D` | `true` | 2-D surface tracking (horizontal only). |
| `Fdrift3D` | `false` | 3-D tracking with vertical velocity W. Set `Fdrift2D: false` when enabling. FESOM2 = z-levels, FESOM-C = sigma coordinates (chosen automatically by mesh type); FESOM 1.4 = z-levels with u, v and w ON the levels, always read linearly in depth (docs/vertical.md §2.1). |
| `time_dir` | `1.0` | `1.0` = forward in time, `-1.0` = backward. Any other value is refused at load (it would multiply every velocity while the clock advances by its sign only). |
| `Vpuls` | `false` | FESOM-C 2-D only: add pulsation velocity `u2d_puls`/`v2d_puls` to the mean flow, when those variables exist in the dataset. |
| `var_u`, `var_v` | `null` (auto) | Override the u/v velocity variable names. Auto-resolution: **FESOM2** → `u`/`v` (when `ncinput` is a single file or Zarr store, the reader first renames the aliases `ua`→`u`, `u_vel`→`u`, `va`→`v`, `v_vel`→`v`; directory mode does **not** apply these renames); **FESOM-C 2-D** → `u2d`/`v2d` if present, else `u`/`v`; **FESOM-C 3-D** → `u`/`v`. Set `var_u: unod`, `var_v: vnod` to use FESOM2 node-projected velocities (see `numerics.uv_interp` — node layout requires `barycentric`). **FESOM 1.4** → `uo`/`vo`: the files are grouped by the variable they hold, whatever their names (§5.1). |
| `var_w` | `null` (auto) | Vertical velocity name, 3-D only. Auto: FESOM2 tries `w`, `wo`, `wvel`; FESOM-C uses `w` (the model's `Wvel`: the positive-up flux through the sigma interfaces; the tracker integrates the particle's sigma fraction with it, `ds/dt = −Wvel/H` — docs/vertical.md §3); FESOM-C's `w_cart` is refused (it lacks the dia-sigma part of the vertical velocity). FESOM-C raises `KeyError` when missing. **FESOM2 with a single file or a non-yearly directory silently proceeds without W when the variable is absent — even when `var_w` was set explicitly but misspelled** — so double-check the name for 3-D runs (the multi-year `<var>.fesom.<year>.nc` layout raises instead). Both readers reject a *mis-staggered* W (a layer-based variable on `nz1`/`sigmam1`, or an interface count disagreeing with the mesh vertical table) with `ValueError` — W must live on the interfaces; FESOM2 also accepts spatial-first storage (`(time, nod2, nz)`) and orients it by dim name like u/v. **FESOM 1.4** → `wo` (required in 3-D; positive up unless its attributes say otherwise). |
| `var_eta` | `null` (auto) | Sea-surface-height name. Auto: FESOM2 tries `ssh`, `eta`, `zos`; FESOM-C uses `eta`; FESOM 1.4 `zos`. **Missing eta is not an error** — zeros are substituted (FESOM-C 3-D additionally warns once; FESOM 1.4 prints one line, and `beaching.use_eta: true` is then refused before the output exists; a `var_eta` set explicitly but in no file is refused). |
| `sigma_layer` | `null` → layer 0 | 2-D mode with layered (3-D) input: which vertical layer to track in (0 = surface). For FESOM2 this is a depth-level index; for FESOM-C a sigma-layer index; for FESOM 1.4 a 0-based index of a kept level (refused when it is not one) — at a node whose bed lies above that level u/v are 0, while a sampled scalar takes the column's last value above the bed. |
| `sigma_mean` | `false` | FESOM-C and FESOM 1.4, 2-D only: instead of picking one layer, use the thickness-weighted vertical mean (FESOM 1.4: from the surface to each node's bed, trapezoids between the levels). Setting both `sigma_layer` and `sigma_mean` is refused at load (the velocity and the sampled scalars resolved the pair differently). |
| `max_levels` | `null` (all) | Keep only the first N vertical *interfaces* from the surface down (N−1 layers). Saves RAM on deep meshes. Intended for 3-D runs (must be ≥ 2 there — enforced only in 3-D mode); when set it trims `mesh.sigma` and all sampled scalar fields consistently in 2-D layered runs too. |
| `w_positive` | `auto` | Sign convention of the vertical velocity **in the file** (3-D only): `up`, `down` or `auto`. KELP integrates a positive-down depth, so a positive-up field is negated at read time. `auto` uses the variable's CF `positive` attribute when present, else its CF `standard_name` (`upward_…` / `downward_…`), else the reader's producer default: `fesom2`, `fesom14`, `healpix` and `fesomc` = `up` (FESOM2 `w`, FESOM 1.4 `wo`, the HEALPix twins' `wo` and ng5 `avg_wo` are all positive upward). `utils/reconstruct_w.py` writes positive-down W and stamps `positive = "down"`; a reconstructed file from before 2026-09-23 has no attribute and needs `w_positive: down`. See docs/vertical.md §3. |
| `uv_vertical` | `layer` | How the horizontal velocity of a 3-D run varies in depth: `layer` = piecewise constant per layer (every vertex or element of the stencil reads the row of the particle's layer); `linear` = linear in depth between the layer centres, the value of the first centre above it and of each column's last wet centre below (no extrapolation; a partial bottom layer is never read); a member of the interpolation stencil (a vertex, or an element) whose bed lies above the particle is a wall, left out as the coast is (`edge_leastsq`: a boundary edge with the `edge_leastsq_bc` ghost) — docs/vertical.md §2.1. `linear` needs 3-D tracking (`Fdrift3D`; refused at load otherwise), a z-level mesh (FESOM2, HEALPix) or a FESOM-C sigma mesh (`mesh.type: fesomc`: there u/v are linear in the sigma fraction `s = z / H_p` between the sigma layer centres `1 − (σ_k + σ_{k+1}) / 2`, each RK stage's `s` from its own column, and every column is wet to the bed — no walls at depth, no element bottoms needed; docs/vertical.md §2.1, *Sigma columns*), `integrator` `euler`, `rk2` or `rk4` (rk45 is 2-D only) and a `numerics.uv_interp` that evaluates the u/v layout: node-layout u/v (FESOM2 `unod`/`vnod`, the HEALPix dual geometry, FESOM-C u/v on the nodes) need `barycentric`; element-layout u/v (FESOM2 and FESOM-C `u`/`v`) take `leastsq`, `edge_leastsq` or `barycentric`; FESOM2 element u/v need each element's bottom from the mesh diagnostic — `nlevels` or `zbar_e_bottom` in the `fesom.mesh.diag.nc` (or in the subset mesh, NetCDF or Zarr, written by `utils/extract_fesom_region.py`, which keeps them; an ASCII mesh directory has neither, and a region extracted before the tool kept them must be extracted again) — and a mesh without ice-shelf cavities (`ulevels > 1` anywhere is refused for now). Each record read is checked once: a non-finite u or v above a column's last usable centre refuses the run, naming node or element, point and record (with `linear` the HEALPix reader checks the raw record before it zeroes missing values, so surface-mask drift is refused, not zeroed; it needs the mesh loaded with the run's mode config, which declares the u data prefix — `run_tracking` and the CLI do). A reader whose u/v sit on the levels (`mesh.type: fesom14`) is always linear in depth and refuses the option. Every refusal of a combination comes before the output file exists; a bad record is refused when it is read. **Proof that it was on: the output's global attribute `uv_vertical = "linear"`** (absent with `layer`, so option-off files are byte-identical) — a misspelled key is dropped silently (§1). Part of the shared-cache namespace (`mode` is). A run handed fields already marked depth-linear with the option off is refused. **The rule cannot change mid-trajectory:** a checkpoint records it (`load_restart` returns it as `uv_vertical`; resume with the same value, checked by `tracking._validate_restart_uv_vertical`). |
| `track_failure_reason` | `false` | Opt into per-particle death diagnostics: allocates the int8 `failure_reason` array on the seeds — every integrator path (fused and unfused) stamps INTERPOLATION_FAILED / OUT_OF_BOUNDS on the step that kills a particle, the writer emits the `failure_reason` output variable, and checkpoints carry it. A beaching remap / resuspension clears the code (reversible, not a death). |
| `sample_vars` | `[]` | Dataset variable names (node-based, e.g. `temp`, `salt`) to sample along trajectories and write into the output file. A single string is accepted and coerced to a one-element list. Empty list = sampling fully off. The variables are probe-read at startup, so a missing or element-based variable fails immediately with `ValueError`. Interpolation method: `numerics.sample_interp`. |

## 3. `numerics` — time step, integrator, interpolation

| Key | Default | Meaning |
|---|---|---|
| `dt` | `null` (auto) | Integration time step in **seconds**. When omitted: set to `file_dt` (one step per file snapshot); if `file_dt` itself is undetectable (irregular spacing), the median interval between records is used instead. When nothing is detectable, startup fails with a clear error. With `temporal_interp: true`, `dt` may be smaller than the file step (sub-stepping). With `temporal_interp: false` (discrete mode) each tracking step reads the *next file index* while the particles move by `dt`, so a `dt` that differs from the file step by more than `max(0.1 s, 1e-3·file_dt)` is **refused at startup** (ValueError, since 2026-09-24; before, the run went on with the forcing at another speed than the particles). Irregular record times (calendar months, a gap) fit no single `dt` and are refused in discrete mode too — use `temporal_interp: true`. A single record is not checked. The check also runs for direct `run_single_realization` / `track_particles` callers, reading the time axis from the dataset when none is passed. |
| `file_dt` | `null` (auto) | Time between consecutive file records, in seconds. Auto-detected from the file `time` coordinate (datetime64, cftime, a numeric CF `<unit> since <epoch>` axis, or a numeric axis whose `units` is exactly one of the day/hour/minute/second/millisecond/microsecond tokens, e.g. `days`, `h`, `ms`; an unknown unit is not guessed). A plain 0,1,2,… INTEGER index without units is rejected as "not real time"; any other numeric axis without a `units` attribute is read as **seconds** — set `file_dt` or add units when it is not. Detection requires near-regular spacing (0.1 % tolerance); irregular spacing yields `file_dt = None` and the median is used only for display/`dt` auto-detection. Set explicitly when the file has no usable time coordinate — `file_times` are then reconstructed as a uniform grid `0, file_dt, 2·file_dt, …`. When the file HAS a decoded axis with increasing stamps, the discrete-mode `dt` check uses that axis (its spacing, and its irregularity), not `file_dt`. |
| `integrator` | `euler` | `euler` / `rk2` / `rk4` / `rk45` (validated at load). RK stages are time-aware: they interpolate velocity to the sub-step times when file-time information is available. `rk45` (TODO 2.3) is the adaptive Dormand–Prince 5(4): each particle substeps inside the fixed tracking `dt` under a metre error tolerance — useful in strongly inhomogeneous flows. **v1 limits (all rejected at startup, never silent):** 2-D only, `uv_interp: leastsq` + element-layout velocities only, no `temporal_reconstruction: pchip`, no `elem_search: walk`, no `em`/`m1` diffusion (uniform/split composes normally). See `time_integration.md` §7. |
| `rk45_tol_m` | `1.0` | `rk45` only: absolute per-substep position tolerance in metres (> 0). Controls the substep count — the error estimate is the embedded 4th/5th-order difference converted to metres. |
| `rk45_h_min` | `null` (auto) | `rk45` only: minimum substep in seconds; `null` → `dt · 1e-4`. When the controller hits it, the substep is force-accepted above tolerance and the run warns once. |
| `output_freq` | `1` | Record trajectory positions every N integration steps. |
| `temporal_interp` | `false` | `true` = linearly interpolate velocity between the two bracketing file snapshots at every step; allows `dt < file_dt`. `false` = discrete mode, one file snapshot per step. CLI flag `--temporal-interp` overrides. |
| `temporal_reconstruction` | `linear` | How velocity is blended in time between file snapshots. `linear` = the 2-snapshot lerp (default, byte-identical to prior behaviour). `pchip` = a 4-snapshot monotone-cubic (Fritsch–Carlson) reconstruction so RK4 separates from RK2 at coarse `file_dt` (otherwise both collapse to the O(file_dt²) linear floor). Active **only** with `temporal_interp: true` (else a one-time warning that it is inert). 2D least-squares uses fused Numba kernels (the 4-snapshot cubic is blended inline per element), and so does 3-D RK2 on node-layout `barycentric` z-level meshes (`w_interp` nearest/barycentric); the other 3D, node-layout and `barycentric` cases use the unfused RK path. Falls back to linear (one-time warning) at the first/last interval where fewer than 4 snapshots exist. No overshoot; works in 3D. |
| `velocity_float32` | `false` | Keep velocity `u`/`v`/`w` in **float32** (their native FESOM2 on-disk dtype) instead of upcasting to float64 — **halves velocity-snapshot memory and read/decode cost** on big meshes. Bit-identical to the default float64 when the store holds float32 (all per-particle arithmetic runs in float64 accumulators; a float32 value is its exact float64 upcast); a float64 store is rounded to float32 on read. TODO 5.3. (Caveats: rotated FESOM2 rounds the rotated u/v at ~float32 eps; FESOM-C 2D `sigma_mean`/`Vpuls` reduce in float64 then downcast — both negligible, default path unaffected.) |
| `prefetch_size` | `1` | **Batch read-ahead**: read this many consecutive velocity snapshots per disk read (one `isel(time=slice).load()`) instead of one-per-step. The main I/O lever on big meshes where reading a snapshot (~seconds) dwarfs a tracking step (~ms). `1` = per-timestep, byte-identical. The window load reads EVERY variable of the opened dataset at full depth and in the on-disk dtype for `prefetch_size` records (the level pick, `max_levels` and the float32 cast happen afterwards, per record), so peak RAM ≈ `prefetch_size` × that raw window plus the processed snapshots — size so that fits. With `temporal_reconstruction: pchip`, `>= 4` is recommended (one window covers the 4-snapshot stencil); smaller values are correct, just more reads. Disabled (per-timestep reads) under `time.on_exhausted: cycle` and under `shared_fields_dir`. TODO 5.2. |
| `shared_fields_dir` | `null` | **Node-shared snapshot cache** (the memory lever for SHARDED runs): one co-resident process reads and publishes each processed velocity snapshot; the other shard processes attach the SAME bytes as read-only memory maps — one field copy per node instead of one per process, one store read instead of S, and no aligned read spikes at model-day boundaries. Point it at a node-local tmpfs (one writer per node; expected fastest, not yet measured) or at a run directory on the shared filesystem (one writer for the whole multi-node run; each node still shares pages via its page cache — mind the extra disk writes). All shards of one run must use the SAME directory and differ only in seed/output paths (the namespace tag is a hash of the fields that change what is read — `mesh`, `mode`, the four `ncinput*` paths, `velocity_float32`, the diffusion field names, the beaching dry-mask variable — plus — since 2026-09-23 — the size and modification time of the mesh and data files, or of a data directory's top-level entries, and the engine version, so a store rewritten at the same path gets a fresh namespace; a rewrite that keeps every stat needs a new directory). Off by default (byte-identical); any error in the shared layer falls back to plain local reads with one warning; forces `prefetch_size: 1`. Outputs are bitwise identical with the cache on or off. A snapshot carries its `uv_vertical` placement (an attached field is checked in its own process, like a record it read); `mode.uv_vertical` is part of `mode`, so runs with and without the option never share a namespace. |
| `shared_fields_generation` | `null` | Namespace generation of the shared snapshot cache (2026-09-23): shards of one run must share it, separate runs never reuse each other's snapshots. `null` = `$KELP_SHARED_FIELDS_GENERATION`, else the SLURM array id (array tasks share it) or job id, plus the srun step id when set (the tasks of one `srun` share it, two sequential steps do not) — so a chain leg or a rerun after rewriting a store never reads stale snapshots (the size/mtime fingerprint of the data paths cannot see an in-place rewrite of a nested Zarr chunk). Outside SLURM with neither set, the cache is not used (local reads, one warning naming this key). Two runs started one after the other in the same batch step without `srun` share the SLURM value — give each its own generation. Each process registers itself in the namespace; a new run removes the namespaces of earlier runs whose processes have all ended (a process killed on another node keeps its namespace — clear the directory after a campaign). |
| `checkpoint_freq` | `0` (off) | Write a restart checkpoint into the output file every N steps (overwritten each time). Requires `data.output_path` (or `-o`). CLI flag `--checkpoint-freq` overrides. Resume recipe: `writers.load_restart` → validate the frame (`tracking._validate_restart_frame`) and the `mode.uv_vertical` rule (`tracking._validate_restart_uv_vertical`: the checkpoint's rule, `load_restart(...)["uv_vertical"]`, `None` = `layer`, must equal the run's) → `run_single_realization` (no config-level auto-resume). |
| `output_flush_every` | `0` (off) | **NetCDF only.** Buffer N record points in memory, then open the file in append mode, write the batch, and close it — so the file holds no HDF5 write lock between flushes and another process can open it read-only **mid-run**. `0` (default) keeps the file open the whole run with `sync()` per record (the original behaviour — on-disk but locked, unreadable until close). Output is **byte-identical** either way; only the on-disk flush cadence differs. `N=1` closes after every record; larger N amortises the open/close cost (cheaper on Lustre). The **Zarr** backend ignores this — Zarr is lock-free and readable mid-run, but it buffers records in memory and writes them a whole chunk at a time (up to 256 records), so a reader sees, and a killed run keeps, only the flushed chunks. CLI flag `--flush-every` overrides. |
| `uv_interp` | `leastsq` | Horizontal u/v interpolation: `leastsq` (least-squares fit over the elements around the nearest node — the default, fastest, fused Numba kernels), `barycentric` (containing-element vertex blend) or `edge_leastsq` (FESOM-style fit over the containing element + its edge-sharing neighbors, boundary ghosts per `edge_leastsq_bc`). **`barycentric` on node-layout velocity gets its own fused RK2/RK4 kernels** whenever a next snapshot is available (discrete and `temporal_interp` runs alike; 2-D, no `walk`, no PCHIP; with `mesh.exact_nearest: false` the mesh must also pass the grid-hash density certificate — HEALPix polar meshes then fall back with a one-time warning); 3-D RK2 on z-level meshes has a fused barycentric kernel too (`w_interp` nearest/barycentric, linear or PCHIP in time); other non-default combinations take the slower unfused path. **`leastsq` and `edge_leastsq` are element-centre math**: combining them with node-layout velocities (`unod`/`vnod`) fails at startup with `ValueError` — use `barycentric` there. Formulas: `docs/interpolation.md` §1; fused-bary details: `docs/time_integration.md` §5. |
| `edge_leastsq_bc` | `freeslip` | Boundary ghost value for `uv_interp: edge_leastsq` only (inert otherwise): a boundary edge contributes the element centroid mirrored across it, carrying `freeslip` (the mirror image of the element's velocity — tangential component kept, normal component reversed; flow along the coast stays exact, flow into it is damped) or `noslip` (−u_own — fitted velocity ~0 at the coast edge). **`noslip` caveat**: the tilted plane overshoots the ambient speed on the interior side of wall elements (~+60 %) and reverses sign near two-wall corners — see `docs/interpolation.md` §1.2. |
| `elem_search` | `local` | Containing-element search: `local` (per-step nearest-node scan, the default) or `walk` (search **once at release**, then track the element by crossing edges — FESOM-style; carried on `ParticleState.elem`, checkpoint-safe). Verdicts identical to `local` (any non-contained walk falls back to the global machinery; near-edge accepts re-verified with the exact local predicate); measured ~3.6× faster containment, ~1.3× faster rk2+barycentric steps. **Unfused paths only**: combining with `uv_interp: leastsq` warns once (the fused kernels keep the grid-hash search). **Trade-off on 2-D node-layout runs**: carried elems also disable the fused barycentric kernels (≈×9 on big meshes vs walk's ~×1.3) — there, prefer `local`; `walk` still pays off for element-layout `barycentric`/`edge_leastsq` and 3-D. Details: `docs/localization.md` §7. |
| `w_interp` | `leastsq` | 3-D only: horizontal combination of vertical velocity W: `nearest` / `leastsq` / `barycentric` / `idw` (validated at load, also in 2-D). The fast fused 3-D Euler kernel supports `nearest`/`leastsq` only; the other methods fall back to the unfused path. |
| `sample_interp` | `leastsq` | Interpolation for `mode.sample_vars` scalars: `nearest` / `leastsq` / `barycentric` / `idw`. |
| `Dmin` | `0.5` | Minimum water-column height in metres — **sigma meshes only** (column geometry, the sigma clamp, sigma sampling); the z-level clamp does not use it. |
| `z0b_min` | `0.1` | Bottom roughness length in metres — how far above the bottom particles are clamped, **sigma meshes only** (the z-level clamp stops at the nearest node's bottom depth). |
| `Dch` | `1.0` | **Currently unused** (reserved; a legacy nearest-node search radius in degrees). Setting it has no effect. |
| `periodic_lon` | `null` (off) | **Periodic boundary window** `[lo, hi]` (degrees) for idealized regional domains (channels, ridges): after each step, ALIVE particles outside `[lo, hi)` are folded back in (post-step, before beaching/record; seeds in the ghost band fold before the first step). **Ghost-band contract**: the fold creates no seam connectivity — the MESH must extend beyond the window by at least one step's maximum displacement on each periodic side (build the mesh with a ghost copy of the opposite edge's columns), or the particle dies OUTSIDE in the gap before the fold can catch it. In-window particles are bitwise untouched; wrapped ones get a fresh nearest-node anchor (and a reset carried element under `elem_search: walk` — the next walk re-seeds via its fallback). Validated: two finite numbers, `hi > lo`, window STRICTLY inside `(-180, 180)` (tracking normalizes longitudes to that branch after every displacement — a window, or a ghost band, touching ±180 lets that wrap relocate crossings and teleport the fold; build the channel away from the dateline, and a mesh straddling ±180 is rejected at startup for the same reason). A mesh stored in the 0..360 branch works — express its window in `[-180, 180)` terms (e.g. nodes at 349..356° take `periodic_lon: [-10, -5]`; the startup checks unwrap the mesh around the window centre). Incompatible with `mesh.rotation` (windows act on MESH-frame coordinates — `ValueError`). Startup sanity (pre-writer in `run_tracking`, repeated with warnings at the run level): a window not overlapping the mesh extent raises; a mesh flush with a window side warns once (crossings on that side die before the fold). **Analysis caveat**: folded trajectories jump by the window span at the seam — unwrap (`np.unwrap`-style over the window) before computing displacements or speeds. In 3-D the fold moves only lon/lat; the ghost band is assumed to carry the same bathymetry (`z` is re-clamped by the next step). Writer metadata `periodic_lon` only when set. Not needed for global meshes — those wrap at ±180° natively. |
| `periodic_lat` | `null` (off) | Same fold for latitude, window STRICTLY inside `(-90, 90)` (`clamp_lat` would swallow a pole overshoot before the fold could see it) — only meaningful for idealized planar-style domains (a doubly-periodic box); real spherical geometry has no periodic latitude. |
| `diffusion` | (disabled) | Stochastic horizontal diffusion — see §3.1. |
| `beaching` | (disabled) | Beaching/resuspension model — see §3.2. |

### 3.1 `numerics.diffusion`

`enabled: false` (default) is byte-identical to no diffusion at all.

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Master switch. |
| `scheme` | `uniform` | `uniform` (constant Kh from this config), `em` (Euler–Maruyama with Itô drift correction, spatial Kh from the dataset), `m1` (Milstein; on spatially varying Kh it warns once — cross-derivative terms are not implemented, `em` is safer for general 2-D Kh). |
| `kh_zonal`, `kh_meridional` | `0.0` | Diffusivities in m²/s, **`uniform` scheme only**. Must be ≥ 0. |
| `var_kh_zonal`, `var_kh_meridional` | `null` (auto) | Dataset variable names for spatial Kh (`em`/`m1`). Auto-detection tries, in order: `Kh_zonal`, `kh_zonal`, `kh_x`, `Kh` (zonal) and `Kh_meridional`, `kh_meridional`, `kh_y` (meridional). An explicitly set name that is missing raises `KeyError`; with auto-detection and no match, diffusion Kh is simply absent. Node or element layout is auto-detected from dim names; both components must agree. NaN/Inf or negative values are rejected. **HEALPix carve-out**: the variables live on the FULL pixel axis of the same store and are compacted onto the wet nodes like u/v (`kh_layout` is always `node`); non-finite Kh at wet nodes is ZEROED with a one-time warning (it also perturbs the surrounding LSQ stencil — see docs/healpix.md §5b) instead of raising; negative Kh is rejected as everywhere. Under `mesh.rotation` use ISOTROPIC Kh (equal fields) — an unequal pair is applied along the ROTATED axes and warns once (docs/coordinate_frame.md §3). |
| `dres_m` | `null` (auto) | Central-difference offset in metres for Kh gradients. `null` → mesh-derived default (median element edge length, clipped to [100 m, 50 km]). |
| `seed` | `0` | Counter-based RNG seed, range `[0, 2⁶⁴)`. Fully deterministic and restart-safe. CLI `--seed` overrides. |
| `composition` | `split` | How the random kick combines with the deterministic step. **Resolved automatically — only one combination needs a manual value:** `uniform` + any integrator → `split`; `em`/`m1` + `euler` → `combined`; `em`/`m1` + `rk2`/`rk4` → you must explicitly set `composition: split_rk` (a pragmatic extension, not a rigorous SDE scheme), otherwise startup raises `NotImplementedError`. Any other override is rejected. |
| `kh_eval_position` | `end` | `split_rk` only: sample Kh at the `start` or `end` (post-RK position) of the step. |
| `az_enabled` | `false` | Vertical Az diffusion (TODO 2.2b): a split kick `Δz = ∂Az/∂z·|dt| + √(2·Az·|dt|)·ξ` with the mandatory Itô drift (kept in backward runs: the adjoint process) and **reflective** surface/bottom boundaries, applied after the deterministic step. Requires `enabled: true`, and is rejected together with `scheme: em`/`m1` — both checked at config load regardless of source (v1; for a vertical-only run leave the Kh values at 0). **3-D runs only** (2-D raises at startup when a source is set). Uses its own counter-RNG component, so results stay thread- and restart-independent. See `diffusion.md` §7. |
| `az_const` | `0.0` | Constant vertical diffusivity in m²/s (≥ 0), used when `var_az` is unset. Works with every reader and both vertical coordinate systems. `az_enabled: true` with `az_const: 0` and no `var_az` is fully dormant — byte-identical output, no guards, no extra attrs. |
| `var_az` | `null` | Dataset variable name for a 3-D Az field (e.g. FESOM2 `Av`). **FESOM2 z-level runs only (v1)** — other readers and sigma meshes raise at startup. Read node-based at the nz interfaces (same layout and `max_levels` trim as `w`); non-finite entries are zeroed; a mid-layer (`nz1`) variable is rejected from metadata before the output file is created. Multi-year exact-layout directories assemble `<var_az>.fesom.<year>.nc` into the family (unlike the em/m1 Kh variables, which still need single-year/pre-concatenated input). |

3-D limitation: `combined` and `split_rk` raise `NotImplementedError` in 3-D
runs; only `uniform` (split) horizontal diffusion works in 3-D — plus the
vertical `az_*` kick above, which is 3-D-only.

### 3.2 `numerics.beaching`

`enabled: false` (default) is byte-identical to no beaching. When disabled,
the remaining keys are not even validated.

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Master switch. |
| `criteria` | `[depth]` | Non-empty subset of `depth`, `coastal_element`, `on_failure`, `stochastic`; OR-combined. A single string is coerced to a list. |
| `depth_min` | `0.3` | Metres: beach a particle when the raw (unclamped) column height `H` drops below this. Must be > 0. |
| `use_eta` | `false` | Include the per-timestep sea level in `H = depth + eta` (tidal drying). **Requires an ssh/eta/zos (or `mode.var_eta`) variable in the dataset** — otherwise startup fails with `ValueError` (the FESOM2 reader would silently substitute zeros and hide all drying; for `fesom14` the message names the `zos` files to add to `data.ncinput`). |
| `dry_mask_var` | `null` | Node variable name: value > 0.5 means dry, ORed into the depth criterion. Probe-read at startup (missing → `ValueError`). Read internally each step; appears in output only if also listed in `mode.sample_vars`. With `mesh.type: fesom14` (whose reader opens only the variables a run consumes) the mask is opened as one more of them, like a sampled variable (2-D, or 3-D like the velocities) with the records of u; a mask no file holds is refused when the data are opened, naming this key. |
| `zone_depth` | `null` → `2·depth_min` | Stochastic-beaching zone: water shallower than this counts as "near shore". |
| `t_beach` | `86400.0` | Seconds: stochastic criterion timescale, `P = 1 − exp(−|dt|/t_beach)` per step inside the zone. Must be > 0 when `stochastic` is enabled. |
| `resuspension` | `never` | `never` / `auto` (release when re-wetted) / `stochastic` (probabilistic release, always re-wet-gated) / `auto_or_stochastic`. |
| `rewet_hysteresis` | `0.1` | Metres above `depth_min` required to count as re-wetted (anti-flicker). |
| `min_wet_steps` | `1` | Consecutive wet steps required before auto release. Must be ≥ 1. |
| `t_resuspend` | `null` | Seconds: stochastic-release timescale. Required (> 0) for `stochastic` / `auto_or_stochastic`. |
| `seed` | `null` | RNG seed for beaching decisions; `null` → reuse `numerics.diffusion.seed`. |

## 4. `mesh` — mesh source

| Key | Default | Meaning |
|---|---|---|
| `type` | `fesomc` | Reader to use: `fesomc`, `fesom2`, `fesom14`, `nemo`, or `healpix`. Decides mesh format, variable-name conventions, vertical coordinate (`fesom2` and `fesom14` → z-levels, `fesomc` → sigma), and the input-file discovery rules of §5.1. **`fesom14`** reads FESOM 1.4 output (docs/dev/readers.md): FESOM 1.4 is a P1 model whose u, v, w, T and S share the 3-D nodes ON the z-levels, so u/v/w come back on the levels and every 3-D run is linear in depth between them (docs/vertical.md §2.1) — combined horizontally with the triangle's barycentric weights, a prism-wise approximation of FESOM 1.4's own linear interpolation on tetrahedra (the same on the levels and along each node's column). Two data forms, detected from the dimensions: level-wise `(time, <level>, <horizontal>)` files (level `depth`, `lev` or `level`; horizontal `ncells`, `nodes_2d` or `nodes_3d` — the model's level-wise writer keeps `nodes_3d` for a dimension of the surface-node length) (e.g. the published CMIP6 output) and the raw writer's vectors over the 3-D nodes `(time, nodes_3d)`, which need the ASCII mesh. The level-wise form is tested on the published CMIP6 output; the raw form and the ASCII mesh are checked against FESOM 1.4's description of its formats and by a round trip on real geometry (the published mesh and fields written in those forms give bit-identical trajectories), not against real raw output. Node-layout velocities ⇒ **`numerics.uv_interp: barycentric` required**. The reader resolves what time each record represents (§5, `data.time_stamp`); `data.time_averaging` is refused. Refused before the output exists: `em`/`m1` diffusion (the output has no Kh fields; `uniform` works), `beaching.use_eta: true` without a sea surface; `diffusion.var_az` is FESOM2-only (`az_const` works). See `examples/config_fesom14_levelwise.yaml`. **`nemo`** (TODO 4.6, **v1: 2-D, E/N-aligned grids only**) maps a structured NEMO C-grid onto an unstructured quad mesh (nodes = wet T-points); `meshpath` = `mesh_mask.nc`/`domain_cfg.nc`, `ncinput` = the `*_grid_T.nc` file (grid_U/grid_V siblings auto-derived), and **`numerics.uv_interp: barycentric` is required** (NEMO velocities are node-layout). NEMO `grid_U`/`grid_V` are along the local grid axes, so v1 uses them as geographic east/north only when the grid axes align with E/N (regular lat/lon or pre-rotated output): a grid whose axes deviate > 2° from E/N is **rejected** (`NotImplementedError`) — vector rotation for true curvilinear/ORCA grids is a staged follow-up. v1 also fails fast on 3-D mode (`Fdrift3D`), on `em`/`m1` spatial diffusion, and on `mesh.rotation`. See `examples/config_nemo.yaml`. **`healpix`** (docs/healpix.md; 2-D and 3-D) tracks directly on HEALPix products (nextGEMS/EERIE/DestinE): the mesh is derived FROM the data store (dual mesh: wet pixel centres = nodes, corner cells = elements; wet = finite u∧v at t=0 on the tracked level — `mode.var_u`/`var_v`/`sigma_layer` are honoured at mesh build; land must be NaN-encoded, or carry a `deptho`-like variable which refines the mask), so `meshpath` points at the store carrying the velocities. Node-layout ⇒ **`numerics.uv_interp: barycentric` required**. Store carriers: plain Zarr, NetCDF, or kerchunk `*.parq`/`*.parquet`/`*.json` reference stores (GRIB-chunked ones need `gribscan` + `dask` in the env; healpy is always required). **3-D** (`Fdrift3D`) requires `mesh.levels` (GRIB level axes are bare indices; extracts carry their own table): u/v = K−1 layer mids, `avg_wo` ON the K interfaces (verified ng5 convention; positive UP — "upward sea water velocity" — negated by the reader into the tracker's positive-down depth rate), T/S sampling via `mode.sample_vars` (ng5 `avg_thetao` is **Kelvin**). `mesh.rotation` supported since 2026-08-16 (pure runtime transform with rotated-pole guards — docs/healpix.md §3); `em`/`m1` spatial Kh is supported on pixels since 2026-08-16 (split_rk compositions — docs/healpix.md §5b). Polar caveat (geographic frame): points poleward of the first pixel ring are unreachable and cold-anchor searches degrade ≳85° — use `mesh.rotation` for polar work (moves the pathology onto land), or crop the caps out (`crop_bbox`). See `examples/config_healpix_2d.yaml` / `config_healpix_3d.yaml`. |
| `meshpath` | `""` (required) | Path to the mesh. **`fesom2`** accepts three forms, auto-detected: a `.zarr` store (as produced by `utils/extract_fesom_region.py`), a single `.nc` file (`fesom.mesh.diag.nc` — recommended for big meshes, loads ~5 s for 11.5M nodes; needs `lon`, `lat`, `elements`, `nod_in_elem2D`, `nod_in_elem2D_num`, optionally `zbar_n_bottom`, `nz`, `elem_area`, and the element levels `nlevels` / `zbar_e_bottom` / `ulevels`, from which the reader declares each element's bottom for `mode.uv_vertical: linear` on element u/v (read when present, by the `.nc` and the Zarr loader; a variable read there that is not one finite value per element, or holds levels outside `1..K`, refuses the mesh at load); the Zarr form needs only `lon`/`lat`/`elements` — adjacency and element centres are rebuilt when absent), or a directory with ASCII `nod2d.out` + `elem2d.out` (optionally `depth.out` and `aux3d.out`/`depth_levels.txt`/`nlvls.out` for depth levels). **`fesomc`** accepts a single NetCDF (or Zarr) with `lon`, `lat`, `nv` (connectivity), optionally `depth`, `lon_elem`/`lat_elem`, `sigma_lev` (or 1-D `sigma`), and optional area variables. **`fesom14`** accepts a FESOM 1.4 **ASCII mesh directory** (it holds `nod3d.out`; `nod2d.out`, `elem2d.out`, `nod3d.out` and `aux3d.out` — FESOM 1.4's `aux3d.out`, not FESOM2's — are read and cross-checked; the levels come from `nod3d.out`) or **level-wise data** — a NetCDF file, or a directory whose first `*.nc` file (name order) holding `mode.var_u` (default `uo`) is used — whose files hold the node coordinates `lon`/`lat` and the cell polygons `lon_bnds`/`lat_bnds`: the triangles are recovered exactly from the polygons, the levels come from the depth coordinate, the wet mask from the first u record, and a `deptho` variable (from the first file holding it, e.g. the published `Ofx` file in the same directory) must agree with that mask within 0.01 m — without it, data written as 0 (not as a fill value) below the bed cannot be detected. Either way `mesh.depth` is the depth of each column's last wet level, exactly. Meshes with ice-shelf cavities are refused. |
| `r_earth` | `null` (family default) | Earth radius in metres for mesh-derived element areas. `null` → FESOM2 and FESOM 1.4 `6 367 500`, FESOM-C `6 400 000`. Only the spherical-excess area *fallback* uses it (when the mesh file carries no area variable); set it only to match a model built with a non-default radius. Must be > 0 when set. |
| `grid_hash_cell_size` | `null` (auto) | Cell size in **degrees** of the uniform grid hash behind every nearest-node query. `null` → the run chooses the cell itself (`grid_hash_cell_rule`, next row; the reader builds the legacy cell `sqrt(domain_area / n_nodes) · 2`, floored at 0.05°, which assumes a **quasi-uniform** mesh). Set it explicitly when the mesh resolution spans orders of magnitude and the rule keeps the legacy cell, or for a release confined to a refined region: there the floor can leave 10⁴–10⁵ nodes in one cell, and the in-kernel 3×3 nearest scan degrades to a linear search (measured: ~2.7 ms per particle per step, a ~100× run-time blow-up). `build_grid_hash` detects that case and emits a `RuntimeWarning` naming this key with a suggested value. **Trade-off** with `mesh.exact_nearest: false` (under the default guard the anchor stays exact, and a cell far below the spacing costs widened scans instead): a smaller cell is faster, but where the local node spacing exceeds ~one cell the in-kernel (fused-path) anchor becomes *approximate* — it may return a near-nearest node instead of the nearest, which perturbs the interpolation stencil (positions are still evaluated exactly). Validate by running ~1 model day at both the candidate cell and a provably-exact (larger) one: statuses must be identical and the divergence confined to the coarse-spacing zones. The value is pinned in `mesh.metadata` and survives any later search rebuild. Must be in (0, 45) when set. `python utils/check_grid_hash_cell.py config.yaml` remains the tool for explicit candidates and seed positions: it reports, at the seeds and at the element centres of the whole mesh, the nodes per 3×3 window and the approximate anchors (true nearest node outside the window) for the auto cell and candidates, and recommends the smallest cell with none. A smaller cell shortens the scan (CORE2 Arctic: 100 nodes per window under the auto cell, 17 at 0.5°, 25–45 % of the tracking phase) but, with the guard off, is exact only where the local node spacing in mesh-frame degrees stays below the cell — at 0.5° the Beaufort run had 2.7 % approximate anchors domain-wide and 0.16 % of the statuses changed over 91 days. |
| `grid_hash_cell_rule` | `occupancy` | How the automatic cell is chosen when `grid_hash_cell_size` is not set (**2026-09-22**). `occupancy`: after `load_mesh`, `run_tracking` / `track_particles` / `kelp info` measure a ladder of candidate cells (the reader's auto cell divided by √2 up to 12 times) on ≤ 200 000 sampled element centres — window occupancy, empty windows, the share of anchors the exact search must widen (its own certificate), the share of anchors a plain 3×3 scan would get wrong — and pick one by the guard mode: with `exact_nearest: true` the cell with the smallest estimated scan cost (nodes evaluated + cells visited + certificate evaluations per query; an empty window only costs the widening and stays exact) among the candidates with a widening share ≤ 20 % and hash offsets ≤ 128 MB per process (the regime the estimate was validated in), with `exact_nearest: false` the smallest cell with no empty window, median occupancy ≥ 8, no more approximate anchors than the auto cell (+0.5 points) and the same memory bound. Hard limits: the far-rows budget ≤ 64 and the 50 M-cell cap. The choice is deterministic (fixed sample), pinned like an explicit cell (rebuilds keep it) and printed as one line (`grid hash: cell … by occupancy/exact (legacy …): window median …`). `legacy`: keep the reader's cell — bit-identical to runs before 2026-09-22. The RESOLVED cell and the rule are always written to the output metadata (`grid_hash_cell`, `grid_hash_cell_rule` = occupancy / legacy / explicit): to repeat an old run exactly, pin `grid_hash_cell_size` to its metadata value, or set `legacy`. The sample is the mesh, not the seeds: a release confined to a region much coarser than the rest of the mesh may still prefer an explicit cell (`utils/check_grid_hash_cell.py` probes the seed positions). In exact mode the rule leaves the legacy cell only when the estimate predicts ≥ 2.75× on the scan cost — the mid-gap between the calibrated classes: quasi-uniform meshes (HEALPix twins 2.0–2.1×, CORE2 2.2×) keep the reader's cell, because there a smaller cell buys ~1.6× on the search in isolation, nothing measurable on the step of a global release, and costs at high latitude (+17–24 % at 80–88 °N, ×3.5 for seeds inside a land cap); structure-resolving meshes act: FESOM-C `sns_multi_m8` 0.058° → 0.014° (window 207 → 13 nodes, search 1.7×), the 11.5 M-node global FESOM2 mesh 0.145° → 0.051° (window 483 → 62, search 1.7–2.6×, 92 MB of hash offsets instead of 12), a synthetic estuary (0.0005° strip / 0.05° offshore) 0.05° → 0.0016° (×26 on the strip). A release confined to a refined region of a quasi-uniform mesh (the CORE2 Arctic: −28 % per step at 0.5°) still wants the explicit cell. Calibration table: `benchmarks/grid_hash_cell_rule_table.md`. |
| `exact_nearest` | `true` | Exact nearest-node anchor guard. The 3×3 grid-hash window can be non-empty and still miss the true nearest node (any closer node lies in the spherical cap of the found node's angular distance θ: θ in latitude, asin(sin θ / cos lat) in longitude — more than one cell at high latitude or with a small explicit cell); the anchor is then only near-nearest, the interpolation stencil differs and the trajectory diverges (measured: CORE2 Beaufort at `grid_hash_cell_size: 0.5`, 2.7 % of the element centres, 0.16 % of the statuses changed over 91 days). With `true`, every batch query and every fused kernel's evaluation site whose window cannot prove its hit runs the in-kernel exact search (the searched window widens until the certificate proves the anchor; only the polar rows near a pole); the 3-D RK2 barycentric kernels run the same in-kernel search (their flag-and-heal handshake serves only the guard-off empty-window case) — results are bitwise those of the unfused (exact) path. Exact for every anchor whose nearest node is within the mesh-derived budget (the longest element edge in cells + 2, so every point inside an element); a query farther than that is outside every element and keeps the best node seen. `rk45` anchors its adaptive substeps with the same in-kernel exact search. **Default `true` since 2026-09-12** — set `false` to reproduce earlier runs bit for bit (the approximate in-window anchor) TOGETHER WITH `grid_hash_cell_rule: legacy` (or the old run's pinned cell): since 2026-09-22 the default rule may change the cell in this mode too, and a different cell means different approximate anchors; the cost when on is the widened scan of the unprovable anchors only (0.005 % of the anchors at the auto cell, 0.1 % at 0.5° on the Beaufort run): measured with `benchmarks/exact_nearest_cost.py` (400 k particles, 8 threads, one fused step): +8 % with every anchor provable (structured mid-latitude mesh, automatic cell), +4 % for a global HEALPix nside-512 release (99 % provable), +25–75 % where half the anchors need a widened window (80–88° N, automatic cell; the fused 3-D RK2 kernel there: no measurable cost), ×4.5 with a cell three times below the node spacing, ×4.1 in the polar band 89–90° N (every cap reaches the pole; 0.02 % of a global release), +10 % for seeds far outside the mesh and +6 % for seeds inside a 20° land cap (both die this step); in every case the fused path stays 1.0–1.7× faster than the unfused one under the guard. Always written to the output metadata as `exact_nearest` (1 or 0). |
| `rotation` | `null` | Coordinate-frame rotation for pole-safe tracking, **FESOM2 + HEALPix + FESOM 1.4** (`fesomc` rejects it with `ValueError`). `null` = geographic frame, byte-identical to older versions. `"fesom2"` = the standard FESOM Euler angles (50, 15, −90) — computational pole over Greenland; use this for Arctic/transpolar tracking. A 3-list `[α, β, γ]` of finite angles in degrees is also accepted. Particle seeds are given in geographic coordinates and rotated in automatically; **output is always geographic**. A warning suggests `rotation: fesom2` when the mesh reaches beyond ±88° latitude without rotation. On HEALPix (2026-08-16) it is a pure runtime transform with rotated-pole guards (a wet cell straddling a rotated pole raises, naming where the poles landed) — docs/healpix.md §3. |
| `input_rotation` | `null` | **fesom14 only** (refused at load for every other `mesh.type`). The Euler angles `[alpha, beta, gamma]` in degrees — as in the model run's namelist — of the rotated frame a FESOM 1.4 **ASCII mesh** is stored in; the reader rotates the node coordinates back to geographic on load (convention of `core.rotation.euler_matrix`, as for `rotation`). Refused for a mesh from cell polygons (the data files' coordinates are geographic). Never guessed: a rotated mesh looks like a geographic one, so the reader warns when a global ASCII mesh (> 350° of longitude) without a node north of 85° N is loaded without it (a geographic FESOM mesh reaches the Arctic; a rotated one has its poles on land). The velocities of FESOM 1.4 mean output are written geographic. Independent of `rotation` (the frame KELP tracks in). |
| `healpix_zoom` | `null` | **healpix only.** Fallback zoom level (`nside = 2**zoom`) for stores that carry no healpix `crs` attributes (`grid_mapping_name: healpix` + `healpix_nside`). When the attrs are present this key is not needed. Integer ≥ 0. |
| `healpix_geometry` | `"dual"` | **healpix only.** Mesh mapping of the pixelation: `"dual"` = nodes are pixel centres (node-layout u/v ⇒ `uv_interp: barycentric` required); `"pixel"` = elements are the pixels themselves, nodes are pixel corners (element-layout u/v ⇒ the fused `leastsq` kernels and `edge_leastsq` apply). `"pixel"` is **2-D only in v1**: `sample_vars`, spatial Kh (em/m1) and `Fdrift3D` fail fast. See docs/healpix.md §1b. |
| `crop_bbox` | `null` | **healpix only.** Regional crop applied at mesh build: a pixel-centre test against `[W, E, S, N]` (degrees; `W > E` = a window crossing the dateline). A raw span `E − W ≥ 360` (e.g. `[-180, 180, S, N]`) means **all longitudes** (a pure latitude band); `W` and `E` that name the *same* meridian after wrapping (e.g. `[180, -180]` or `[350, -10]`) are ambiguous (0°- or 360°-wide) and are rejected with a pointer to the `span ≥ 360` form. The memory lever for zoom ≥ 10 — particles die at the crop edge like at any mesh boundary. |
| `levels` | `null` | **healpix only.** Vertical interface depths for stores whose level axis is bare indices (GRIB kerchunk streams, e.g. ng5): a FESOM `aux3d.out` path (line 1 = K, then K interface depths; the node-bathymetry tail is ignored) **or** an inline list of metres. In 2-D it also provides per-cell bathymetry (deepest finite level of the u column → interface depth), which the beaching depth criteria need; without it (and without a `deptho` variable in the store) `mesh.depth` is all zeros. Values are `abs()`-ed and must be strictly increasing. |

## 5. `data` — input and output files

| Key | Default | Meaning |
|---|---|---|
| `ncinput` | `""` (required) | Velocity input: a single NetCDF file, a Zarr store, or (FESOM2 and FESOM 1.4) a **directory** of per-variable files; FESOM 1.4 also takes a glob pattern — see §5.1 for exactly what gets opened. |
| `ncinput_v` | `""` | Separate v-velocity file, for datasets where u and v live in different files. Empty = v is in `ncinput`. FESOM 1.4: more files, a directory or a glob, whose files simply join those of `ncinput`. |
| `ncinput_w` | `""` | Separate vertical-velocity file (3-D). Empty = no W or W is in `ncinput`. FESOM 1.4: as `ncinput_v`. |
| `ncinput_eta` | `""` | **healpix only** (other readers reject it). Separate ssh store for products whose velocity store carries no sea-surface height — the ng5 recipe: u/v from the 3-D store, `avg_zos` from the 2-D store. Empty = ssh is in `ncinput`, or a `3D`→`2D` filename sibling is auto-derived; if neither exists the run proceeds with eta = 0 (one warning). Both stores must share the exact same time axis (validated). |
| `inicloud` | `""` (required) | Initial particle positions: whitespace-separated text, one particle per row, `lon lat [depth] [release_sec]` in degrees (+ metres positive down; + seconds), `#` comments allowed. Depth column is optional (0 when absent). The optional 4th column is a **delayed-release offset in seconds** from the run start (default 0 = released immediately); see `time.n_pulses` / `time.pulse_interval`. Coordinates are geographic even for rotated-frame runs. |
| `particle_offset` | `null` | Global index of the seed file's first particle in the full release, for a run over one SHARD of it (2026-10-04). Every random draw (diffusion, the vertical Az kick, stochastic beaching and resuspension) is keyed by `particle_offset + i`, so a sharded run with ONE common seed merges into exactly the unsharded run. `null` = the `# kelp particle_offset N` header that `utils/split_seeds.py` writes into every shard file, else 0. A value that disagrees with the header is refused, as is an offset above 0 with `time.n_pulses > 1` (a pulsed release replicates the seed set and cannot be sharded). An integer in [0, 2⁶³). Any comment before the data that mentions `particle_offset` or `particle-offset` (any case) must be exactly that header with ASCII digits (a near miss such as `particle_offset: 5` is refused, never read as 0; prose like "particle offset from the coast" is not affected). Stamped as the output attribute `particle_offset`, next to `random_draws` (0/1: whether the run draws random numbers at all). |
| `output_path` | `""` (in-memory) | Trajectory output. Backend chosen **by extension**: `.zarr` → Zarr v3, anything else → NetCDF4 (use `.nc`; a wrong extension is not rejected, it just writes NetCDF). Empty = keep trajectories in memory only (returned as `TrajectoryRecord` objects). The file follows output format 2 (`kelp_output_format = 2`: dimensions `trajectory`/`time`, CF calendar time, `release_*` variables) — layout in `io_and_time.md` §5. With `time.num_runs > 1`, each realization gets a suffixed file: `name_r0.nc`, `name_r1.nc`, … CLI `-o/--output` overrides. **Reading output mid-run:** a NetCDF4 file is HDF5-locked and unreadable by other processes until the run closes it (and a hard-killed run leaves it un-closed) — use `.zarr` (lock-free, readable as it grows) or `numerics.output_flush_every` (buffered NetCDF) to inspect partial results during a long run. |
| `time_averaging` | `null` (warn + guess) | Whether velocity records are time-averages (`"mean"`, e.g. FESOM2 monthly means) or instantaneous snapshots (`"instant"`, typical FESOM-C). `"mean"` shifts every file timestamp from the *end* of its averaging window to the window *midpoint* before any time interpolation. **Always set this explicitly** — when omitted, KELP warns and guesses (`fesom2` and `healpix` → `mean`, otherwise `instant`). Any other value raises `ValueError`. **`fesom14`: leave it unset** — that reader resolves what time each record represents itself (next rows), nothing shifts its time axis again, and a set value is refused when the run starts; the output then records `time_averaging = "resolved_by_reader"`. |
| `time_stamp` | `null` | **fesom14 only** (refused when the run starts for every other `mesh.type`, as are the next two keys). Where each record's time stamp sits in its averaging window: `start`, `centre` or `end`, or `instant` for snapshots. Given together with `time_window` or not at all (refused at load otherwise). Needed only for files that do not say what their stamps mean; when set, the two decide for every file, and a file's own time bounds or `output_schedule` are overridden with a warning naming the first record where they disagree. Without them the reader takes, per variable and file, the time bounds (`time.attrs["bounds"]`, e.g. the published CMIP6 files) or the raw writer's `output_schedule` attribute (stamps at the window ends, checked against the schedule), else refuses with a message naming these keys. Each record then represents the CENTRE of its window (a mean) or its stamp (a snapshot), the window is never estimated from the spacing of the stamps, and every record the source says must exist must be there (a gap — a missing record, month or file — refuses the run). All variables of a run must have the same records. See `io_and_time.md` §3. |
| `time_window` | `null` | **fesom14 only.** The averaging window of each record (with `instant`: the snapshot cadence): `calendar_month`, `calendar_year`, `calendar_day`, or a fixed duration `"<number> <s\|min\|h\|d>"` (e.g. `"5 d"`); windows are computed exactly in the files' calendar. With a calendar window, a stamp must sit on a boundary of its window (`start`, `end`, `instant`) or at its centre (`centre`); any other stamp is refused (naming the first such record; see `time_offset`). |
| `time_offset` | `null` | **fesom14 only.** A fixed shift `"<±number> <s\|min\|h\|d>"` (e.g. `"+12 h"`) added to the raw stamps, the bounds and the epoch of every file before anything else; allowed alone. For raw-writer output of a job that started at a time of day other than 00:00: its stamps all sit early by that time of day, which the files cannot show, so the schedule check refuses them until the offset is given (from 0 up to, not including, 24 h there). |

### 5.1 Exactly which files get opened (`ncinput` discovery rules)

This is where implicit behaviour lives — read carefully, or avoid it entirely
by pointing `ncinput`/`ncinput_v`/`ncinput_w` at explicit files.

**Zarr detection** (everywhere a path is opened): a path is treated as Zarr if
it ends in `.zarr` or is a directory containing a `.zattrs` (Zarr v2) or
`zarr.json` (Zarr v3) file; otherwise NetCDF.

**FESOM-C (`mesh.type: fesomc`)** — no directory scanning at all:

- `ncinput` must be one file (or Zarr store).
- `ncinput_v` / `ncinput_w`, when set, must exist (`FileNotFoundError`
  otherwise) and are merged (`xr.merge(compat="override")`; on duplicate
  variable names the **first** dataset — `ncinput` — wins). Each must have a
  time dimension and cover every record of `ncinput` (extra records are
  dropped; compared as instants, so another epoch or unit, or a coarser
  cadence, is refused — a merge by label would pad the unmatched records with
  NaN; since 2026-10-04).
- Variables are then resolved by name inside the merged dataset (see
  `mode.var_*` in §2).

**HEALPix (`mesh.type: healpix`)** — no directory scanning; three carriers,
detected from the path:

- `*.parq` / `*.parquet` / `*.json` → a **kerchunk reference store** opened
  via fsspec (`reference::` protocol, dask-backed); a lazy `import gribscan`
  registers the GRIB codec for GRIB-chunked products (ng5), and KELP
  installs a guard that maps null reference rows (absent chunks) to
  missing-chunk semantics. Anything else follows the usual Zarr/NetCDF
  detection.
- Variables resolve by `mode.var_*` override first, then aliases
  (`avg_uoe`/`uo`/`u`, `avg_von`/`vo`/`v`, `avg_zos`/`zos`/`ssh`).
- If `ncinput` lacks an ssh variable: `ncinput_eta` is opened when set, else
  a `3D`→`2D` filename sibling is tried, else eta = 0 (one warning). The two
  stores must share the exact time axis.
- `ncinput_v` is rejected (u/v always live in one healpix store);
  `ncinput_w` is rejected too (W lives in the same store as u/v).
- **Regional extracts** made by `utils/extract_healpix_region.py` are
  consumed transparently (the reader finds `healpix_index` inside): point
  both `meshpath` and `ncinput` at the extract; no `ncinput_eta` (ssh was
  merged at extraction); `crop_bbox` composes as a further crop.

**FESOM 1.4 (`mesh.type: fesom14`)** — files are grouped by what they hold,
never by their names:

- `ncinput` (and `ncinput_v` / `ncinput_w`, whose files simply join) is a
  NetCDF file, a directory (its `*.nc` files, name order; hidden files
  skipped) or a glob pattern (e.g. `path/to/<member>/Omon/*/gn/*/*.nc` for a
  CMIP6 directory tree); every file must be readable NetCDF, its header is
  read, and the files holding each variable the run consumes are taken
  together — whatever the names (`<var>_<runid>_<YYYYMMDD>.nc` of the raw
  writer, the CMIP6 names, others). Other variables and files are ignored.
- Consumed: `mode.var_u` / `var_v` (`uo` / `vo`), in 3-D `var_w` (`wo`), the
  sea-surface height `var_eta` (`zos`) when some file holds it, every
  `mode.sample_vars` entry (`thetao`, `so`, …) and, with beaching enabled,
  `beaching.dry_mask_var`. A required variable that no file holds is refused,
  listing the variables the files hold.
- Each variable's time axis is decoded file by file (each file's own epoch and
  calendar) and resolved (`data.time_stamp` row of §5); then every consumed
  variable must have exactly the records of u — the same instants and windows.
  The default raw output writes `zos` daily and the velocities monthly, which
  is refused (the message suggests leaving `zos` out of `ncinput`).
- Refused with a message: FESOM 1.4 restart files (they hold `wpot` or
  `iter`: snapshots with u/v in the rotated model frame) and the older
  branch's `<runid>.<year>.oce.mean.nc` mean output (its writer is not part of
  FESOM 1.4, so neither its frame nor its time axis can be checked); files or
  variables of different layouts, levels or node coordinates; and a variable
  named `lon` or `lat` that is not a node coordinate the reader can check —
  in a file of a consumed variable, `lon` and `lat` must be one-dimensional
  along one horizontal dimension (`ncells`, `nodes_2d` or `nodes_3d`), both
  of them and along the same one (cell bounds such as `lon_bnds` are not
  affected); write them so or remove them from the file.
- The checks against the mesh (sizes, the level axis, node coordinates, the
  wet mask of record 0 of every variable) run right after, before any output
  file exists. Node coordinates count from any file that carries them, along
  any of its horizontal dimensions: next to vectors over the 3-D nodes, the
  `lon`/`lat` over the 2-D nodes of a `zos` file, or of the velocity files
  themselves, are compared with the mesh too. The tolerance is 1e-6°, or two
  spacings of the coordinates' own floating type at their magnitude when that
  is more (float32 coordinates near 180°: 3.05e-5°, still far below the
  distance between two nodes).

**FESOM2 (`mesh.type: fesom2`), `ncinput` is a file or Zarr store:**

- The file is opened directly; aliases `ua`/`u_vel` → `u`, `va`/`v_vel` → `v`
  are renamed when the standard name is absent.
- `ncinput_v`, when set and `v` is not already present: its `v` variable (or,
  failing that, its **first data variable, whatever it is**) becomes `v`.
- `ncinput_w`, when set and `w` absent: same rule (`w` or first data variable).
- An attached `ncinput_v` / `ncinput_w` is matched to `ncinput` record by
  record, by instant (datetime64 and standard-calendar cftime compare equal;
  numeric axes compare in their units): it may hold more records than
  `ncinput` (only `ncinput`'s are used; with dates even a finer cadence, e.g. a daily W next
  to a 2-day extract, sub-sampled with a warning — next to time MEANS over a
  longer window those are single-instant values, not means). For numeric time axes without an epoch a record of it
  inside `ncinput`'s span that is not one of `ncinput`'s records means
  another origin or unit and is refused; repeated stamps match only when both
  files repeat them identically; a file without a time dimension is refused.
  `ncinput_v` must cover every record of `ncinput`; `ncinput_w` may cover a
  contiguous block of them (a `utils/reconstruct_w.py` run with
  `time_range`; a gap inside its own range is another cadence and is
  refused), and a run that needs an uncovered record is refused when it reads
  it — a step also reads the next record (PCHIP the next two), so keep a
  margin. The NaN fill of the uncovered records is lazy (no extra memory).
  In a 2-D run (mode `Fdrift3D: false`) an `ncinput_w` that does not fit is
  skipped with a warning, since W is never read.
  Before 2026-10-04 xarray attached these files by label without a check, so a
  file on other stamps — a W written by `utils/reconstruct_w.py`, which then
  carried relative seconds — came out all-NaN; such W files must be
  regenerated (the tool now writes its source's own time coordinate).

**FESOM2, `ncinput` is a directory** — automatic discovery, in this order:

1. `u*.nc` (else `u*.zarr`) and `v*.nc` (else `v*.zarr`), each list sorted
   lexicographically — **only the first file of each list is used**. Missing
   u or v files → `FileNotFoundError`. Inside the chosen file, variable `u`
   (resp. `v`) is taken if present, else the first data variable.
2. Optionally `w*` and `ssh*`, same "first sorted match, named variable or
   first variable" rule. Each is matched to the chosen `u` file record by
   record (same rules as for `ncinput_v`/`ncinput_w` above; since 2026-10-04 —
   the assembly aligns by label and used to stretch the axis with NaN
   records): `v` must cover every `u` record; `w` may cover a contiguous
   block of them, and a `w*` file that does not fit is refused in 3-D (and
   when the reader gets no `mode` to tell) and skipped with a warning in 2-D
   (the prefix glob also catches files such as `wind_stress.nc`); an `ssh*`
   file that does not fit is skipped with a warning, and one that covers only a
   contiguous block of the velocity records is attached with a warning naming the
   records it lacks (they are NaN: `beaching.use_eta` sees no surface there).
   `data.ncinput_v` / `ncinput_w` are ignored (with a warning) when
   `ncinput` is a directory.
3. **Every remaining** `*.nc`/`*.zarr` file in the directory is then opened
   and merged so that `mode.sample_vars` can reach per-variable files
   (`temp.fesom.1958.nc`, `salt…`), with two guards: files whose *name*
   contains `mesh` (case-insensitive) are skipped, and only variables whose
   dims are all within the known FESOM2 set
   `{time, nz1, nz, nod2, elem, node, sigma, depth, level}` are merged
   (anything on an alien grid is silently dropped). Unreadable files are
   skipped with a `RuntimeWarning`, and so are variables that do not cover
   every record of the `u` file (another cadence or other stamps): they
   cannot be sampled along the trajectories, and merging them would pad the
   velocities with NaN records.

Pitfalls of directory mode (all verified against the glob logic):

- The patterns are **prefix globs**: `velocity.nc` matches `v*`,
  `wind_stress.nc` matches `w*`, `uv_combined.nc` matches `u*`. A stray file
  with an unlucky name silently becomes your velocity input.
- **Prefix-glob dirs read one file per variable.** Loosely-named files such as
  `u.1958.nc`, `u.1959.nc` → only `u.1958.nc` (sorted first) is read; no
  concatenation. Concatenate beforehand if you need a longer span.
- **Exact multi-year layout DOES concatenate** (FESOM2 only). When files follow
  the model's `<var>.fesom.<YYYY>.nc` naming and the run passes its `mode_cfg`,
  the reader recognises the per-variable/per-year layout and concatenates **all**
  years lazily onto one time axis (driven by the exact
  `var_u`/`var_v`/`var_w`/`var_eta`/`sample_vars` names; needs `dask`). See
  `io_and_time.md` §2 "Multi-year directories" for the trigger, cadence
  fail-fast, and limitations.
- The merge in step 3 means *every* readable NetCDF/Zarr in the directory can
  contribute variables. Keep run directories clean, or use explicit file
  paths.

**Recommended explicit style** — no discovery, every file pinned:

```yaml
data:
  ncinput:    path/to/u.fesom.1958.nc
  ncinput_v:  path/to/v.fesom.1958.nc
  ncinput_w:  path/to/w.fesom.1958.nc     # 3-D only
  time_averaging: mean
```

(The single remaining implicit choice is the variable *name* inside each
file — pin those too with `mode.var_u/var_v/var_w/var_eta` if non-standard.
With explicit files there is no way to attach extra `sample_vars` files;
sampling variables must live in one of the three files, or use directory
mode with a clean directory.)

`kelp info config.yaml` prints the mesh statistics and the time range of
whatever was actually opened — use it to verify a config before running. It
opens the data as a run does, through the same helper: `ncinput` with
`ncinput_v`/`ncinput_w` (since 2026-10-19; before, `ncinput` only), `ncinput_eta`,
the diffusion block when it reads a `var_az` field (FESOM2), the `data.time_*`
stamp options and, for `fesom14` with beaching enabled, `beaching.dry_mask_var`
(the eta path, `var_az` and the cut below since 2026-11-04); it cuts
`mode.max_levels` from the levels (the `sigma:` line shows the file's count)
and runs the reader's checks against the mesh on the kept levels; for `fesom14` it also prints
what resolved the record times and the first and last instant.

## 6. `time` — time window and realizations

| Key | Default | Meaning |
|---|---|---|
| `start_time_idx` | `0` | First file time index to read. |
| `n_steps` | `null` | Number of integration steps (**the preferred way** to set run length). Required unless `end_time_idx` or `start_date` + `duration_days` is given. In temporal-interp mode these are `dt`-sized sub-steps, not file steps. |
| `end_time_idx` | `null` | Legacy: last file index; converted to `n_steps = end_time_idx − start_time_idx` (in every entry point since 2026-10-04; the direct API added one step before). `n_steps` wins if both are set. Quirk kept for backward compatibility: with `time_dir: -1.0` (and no `n_steps`/`start_date`), tracking *starts* at `end_time_idx`. With `start_date` and neither `n_steps` nor `duration_days` it is refused at load (`kelp run` and `kelp info`): the length would ignore the record `start_date` resolves to — set `duration_days` or `n_steps`. |
| `start_date` | `null` | ISO datetime string (e.g. `"2020-01-15T00:00"`) resolved against the file time coordinate; overrides `start_time_idx`. Requires an interpretable time coordinate. Must match a file timestamp **exactly** unless `temporal_interp: true`, in which case the run may start between snapshots; outside the file range → error. Note: the match is done against the **raw timestamps stored in the file** — with `time_averaging: mean`, give the end-of-window timestamp as written in the file, not the window midpoint. With `temporal_interp: true` the run clock starts at the requested instant itself, exact stamp or not; in discrete mode the run starts at the record (under `mean` its window centre). Under `mean` the last stamp lies past the last window centre and keeps the record start (with a warning), and `start_time_idx: k` starts at the record's window centre, half a record before `start_date: <stamp k>`. **`fesom14`**: the file time coordinate is the reader's resolved instants — the window centres of means — so a daily mean sits at 12:00 and a discrete-mode `start_date` at 00:00 matches no record: use `temporal_interp: true` or a 12:00 start. Monthly means are a calendar month apart (irregular), which discrete mode refuses: they need `temporal_interp: true`. `kelp info` prints the first and last resolved instant. |
| `duration_days` | `null` | With `start_date`: run length in days (a finite number > 0), converted to `n_steps = duration_days · 86400 / dt` (it then overrides `n_steps`). Without `start_date` it is refused at load — set `n_steps`. |
| `on_exhausted` | `stop` | What happens when file time indices run out before `n_steps` is reached (validated at load, case-sensitive): `stop` (halt) or `cycle` (wrap around to index 0 — periodic forcing; useful to spin long runs from a short or single-snapshot file). |
| `num_runs` | `1` | Number of realizations. Each gets its own output file (`_rN` suffix) and its own diffusion realization id (different random kicks, same seed). |
| `run_interval` | `0` | File-time-index shift between consecutive realizations (realization k starts at `start_time_idx + k·run_interval`). Under `on_exhausted: cycle` a start index outside the records wraps by whole forcing periods (index `n_time + 1` starts one period after record 1, index `-1` one period before the last record). Under `stop` every start must be a record: a start outside `[0, n_time)` refuses the run before the first realization, naming `num_runs`, `run_interval`, the index and the range. With `start_date` and temporal interpolation each realization starts at the date shifted by the record times between the start indices, so a date between records can start a later realization past the last record although its index is a record: refused the same way under `stop`, naming `start_date`, `num_runs`, `run_interval`, the start time and the record times. |
| `n_pulses` | `1` | **Delayed/staggered release.** Re-release the whole seed set this many times (pulses) at the *same* coordinates, `pulse_interval` seconds apart: pulse `p` is released `p·pulse_interval` after the run start. Total particles = `n_pulses × (seeds in inicloud)`. `1` (default) = single release. Composes additively with the per-particle `release_sec` seed column (pulse `p` adds `p·pulse_interval` to each seed's own offset). Release time is an offset from the run start (→ step index `round(sec/dt)`), so it behaves identically forward and backward. Frozen particles are written as NaN with status `not_released` until they activate; the output's `release_time` holds each particle's rounded release instant (`io_and_time.md` §5.3). `1` + `pulse_interval=0` + no seed column → byte-identical to no feature. |
| `pulse_interval` | `0.0` | Seconds between release pulses (see `n_pulses`). Must be `> 0` when `n_pulses > 1`. |

## 7. Command-line overrides

`kelp run config.yaml` accepts overrides that win over the YAML values:

| Flag | Overrides |
|---|---|
| `-o / --output PATH` | `data.output_path` |
| `--checkpoint-freq N` | `numerics.checkpoint_freq` |
| `--temporal-interp` | `numerics.temporal_interp` (only forces *on*; absence keeps the YAML value) |
| `--seed N` | `numerics.diffusion.seed` (re-validated) |
| `--flush-every N` | `numerics.output_flush_every` (NetCDF only; re-validated) |
| `-v / --verbose` | progress + per-step timing |

## 8. Quick checklist for writing a config

1. Pick `mesh.type` (`fesom2`, `fesom14`, `fesomc`, `healpix` or `nemo`) — this fixes the
   vertical coordinate, the default variable names, and the file-discovery rules.
2. Point `mesh.meshpath` at the mesh; prefer NetCDF mesh-diag or Zarr for
   large FESOM2 meshes.
3. Pin input files explicitly (`ncinput` + `ncinput_v`/`ncinput_w`) unless you
   deliberately want FESOM2 directory discovery — then keep the directory
   clean and remember: first sorted `u*`/`v*`/`w*`/`ssh*` match only.
4. Set `data.time_averaging` explicitly (`mean` for averaged output, `instant`
   for snapshots) — never rely on the guess. **Except `fesom14`**: leave it
   unset; the reader resolves the record times from the files (time bounds or
   the raw writer's schedule), and `data.time_stamp` + `data.time_window` say
   what the stamps mean when the files cannot.
5. 3-D run (FESOM2 `unod`/`vnod` or `u`/`v`, HEALPix, FESOM-C)?
   Decide `mode.uv_vertical`: `layer` (default) or `linear` in depth (node
   u/v: `uv_interp: barycentric`; element u/v: `leastsq`, `edge_leastsq` or
   `barycentric`, on FESOM2 with a mesh diagnostic that has `nlevels` or
   `zbar_e_bottom`; on FESOM-C linear in the sigma fraction); check the
   output's `uv_vertical` attribute.
   (`fesom14`: its u/v are on the levels and always linear in depth — leave the
   key unset.)
6. Decide discrete vs temporal-interp: `dt == file_dt` with
   `temporal_interp: false`, or `dt < file_dt` with `temporal_interp: true`.
7. Set the window: `start_time_idx` (or `start_date`) + `n_steps` (or
   `duration_days`), and `on_exhausted`.
8. Give `data.output_path` (`.nc` or `.zarr`) for anything longer than a toy
   run; add `checkpoint_freq` for long runs (resumable restart). If you want to
   **watch the output as it grows** or the run might be interrupted, prefer
   `.zarr` (lock-free, readable mid-run; records appear a chunk of up to 256 at a time) or `numerics.output_flush_every: N`
   (buffered NetCDF) — a plain `.nc` is HDF5-locked and unreadable until close.
9. Verify with `kelp info config.yaml` (mesh loads, dataset opens, time
   range is what you expect), then a short `n_steps` smoke run with
   `--verbose`.
