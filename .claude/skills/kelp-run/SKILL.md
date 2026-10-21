---
name: kelp-run
description: Use when the user describes a KELP tracking experiment to set up or start — a release location or region, forward or backward, surface or 3-D, diffusion, beaching, temperature or salinity along the tracks, a time window — on a laptop or on SLURM, even without the word config; also for the first run on a dataset not used before. For the YAML alone use kelp-config; for a campaign of many jobs use kelp-experiment.
---

# Setting up and launching a KELP run from a task

This skill turns a described experiment into a runnable setup: **seeds → config
→ run command**. It builds on, and does not duplicate:

- **`docs/architecture.md`** — the pipeline (config → reader → mesh → loop →
  writer) and the three data structures. Read it to know what each choice maps
  to at run time.
- **`docs/configuration.md`** — the authoritative option-by-option reference
  (defaults, validation, file-discovery rules, pitfalls). Every config key
  decision is grounded there.
- **`docs/performance.md`** — measured cost model, scaling, thread count, I/O,
  output-writer choice (Zarr vs NetCDF), sizing recipes, and SLURM run
  operations. Read it to set a realistic `dt`/scheme/tracer-count and to answer
  *how long will it take / how many threads / should I extract a region*.
- **the `kelp-config` skill** — the config-writing mechanics (explicit-file
  pinning, `time_averaging`, cross-key constraints). Follow it when you write
  the YAML; this skill is the layer above that also decides the knobs, makes
  the seeds, and launches.

**Checks apply where their condition holds.** Each pre-launch check below names its
condition (3-D, a dataset not used before, a release at depth, a large or 3-D run, a
mesh refined where the particles are), read literally — a store nobody has run on yet is "a
dataset not used before", 2-D or 3-D, production or test data. A check whose condition does not
hold is skipped. The handback has a section `Defaults not used / checks skipped`, one line each:
`<default> — owner's choice: <what>` or `<check> — <condition not met>`. The launch
gate (ask, then wait for an explicit yes) holds for every run.

**Reference files and scripts (read the one a step points to):**
- [references/data-and-io.md](references/data-and-io.md) — big meshes, regional subsets, multi-year directories, I/O knobs, PCHIP.
- [references/seeds.md](references/seeds.md) — other seed shapes, sorting, release over time.
- [references/grid-hash-cell.md](references/grid-hash-cell.md) — the optional nearest-node cell check.
- `scripts/w_sign.py`, `scripts/record0.py` — the vertical-velocity sign and record-0 checks (step 4).

## Workflow

1. **Read the task for the five decisions** (ask only if genuinely ambiguous —
   prefer sensible defaults and state them):

   | From the task… | …decide |
   |---|---|
   | model / data on disk (FESOM2 global/regional vs FESOM-C coastal vs FESOM 1.4, e.g. the CMIP6 AWI-CM-1-1 `uo`/`vo`/`wo` output) | `mesh.type` (`fesom2` / `fesomc` / `fesom14`), the mesh + data paths |
   | "surface" / "at depth" / "3D" | `mode.Fdrift2D` vs `Fdrift3D` (+ `max_levels`) |
   | "backtrack" / "where did it come from" | `mode.time_dir: -1.0` |
   | "with turbulent spreading / eddy diffusion" | `numerics.diffusion` (scheme below) |
   | "temperature/salinity along the path" | `mode.sample_vars: [temp, salt]` |
   | "how many strand / wash up / reach the coast" | `numerics.beaching` — the counting setup in kelp-config `references/beaching.md` (what fires on THIS mesh, the seeds in rim elements, coast versus open sea) |
   | a release point/region + count | the **seeds** (step 2) |
   | "released over time / every N days / a continuous source" | `time.n_pulses` + `pulse_interval` (or a per-seed `release_sec` column) — see step 2 |
   | a duration + cadence | `dt`, `n_steps`, `temporal_interp`, `on_exhausted` |
   | which velocity layout the data has | ELEMENT layout (FESOM2 `u`/`v`, FESOM-C `u2d`/`v2d`/`u`/`v` on the element dim) → `uv_interp: leastsq` (the default, fused; it reproduces the legacy Fortran tracker's scheme); NODE layout (FESOM2 `unod`/`vnod`, HEALPix pixels, NEMO T-points) → `barycentric` (required). Decide from the variable's DIMENSION in `ncdump -h` (`elem` vs `nod2`/`node`), never from the name (`kelp info` prints mesh sizes and the first/last time stamp, not the layout). On 2-D node layout pair `barycentric` with `elem_search: local` — `walk` disables the ~×9 fused barycentric kernel |
   | node-layout data (`unod`/`vnod`, NEMO) or a coastal boundary-condition study | `numerics.uv_interp` (`barycentric` for node layout; `edge_leastsq` + `edge_leastsq_bc` for FESOM-style elemental interpolation) — `elem_search`: `local` on node layout (the 2-D fused barycentric kernels and the 3-D RK2 fused kernel need it); `walk` (identical results, faster containment) pays off on the unfused element-layout paths (`barycentric`/`edge_leastsq` on element layout) |
   | an idealized periodic channel / doubly-periodic box | `numerics.periodic_lon` / `periodic_lat` `[lo, hi]` — the mesh must carry a ghost band past the window (docs/configuration.md; incompatible with `mesh.rotation`; window strictly inside ±180/±90) |
   | "how deep do they get" / subduction / any depth statistic | `mode.max_levels: null` (full column) — `max_levels` is a RAM lever that inserts an artificial bottom and CLAMPS exactly the quantity measured; never trim it "to save time" (it saves none) |
   | 3-D, and the vertical structure of u/v matters (shear between layers, a coarse vertical grid, paths near the bed or in the thermocline) | by default `mode.uv_vertical: layer` (constant per layer); `linear` (between the layer centres) when the owner wants it or the vertical shear matters — measured on a 10-year Arctic run: individual paths move ~10 km in a year, the cloud's statistics stay within a few percent, and on a grid of half the levels the linear rule errs about a third as much as the layer rule. Mechanics and refusals: `kelp-config` (`fesom14` is always linear) |
   | 3-D node-layout (`unod`/`vnod`, HEALPix) with `rk2` | set `numerics.w_interp: barycentric` (or `nearest`) EXPLICITLY — the config default `leastsq` silently bypasses the fused 3-D RK2 kernel (×1.3 and full-node scaling); keep `elem_search: local` there |
   | "vertical mixing" / "how deep" with turbulence | `diffusion.az_enabled: true` + `az_const` (3-D only, composes with `uniform`/split; `var_az` field on FESOM2 z-levels) — a second experiment after the deterministic one, never in a dt comparison |
   | 3-D with a vertical velocity from a dataset or reader not used before | check the SIGN against physics before the first campaign: the mean `w` of the FILE over the equatorial Pacific (2°S–2°N, 150–100°W) at ~50 m is upwelling, ≈ +1e-5 m/s if the file is positive up; KELP's internal `w` is positive DOWN and the readers convert by `mode.w_positive` (`auto` = the CF `positive` attribute, then `standard_name` upward_/downward_, else the reader family default) — decide `mode.w_positive` from that number, then write it (the family default is not a decision); for a regional crop run the check on a config that points at the global SOURCE files (the crop has no equator); a synthetic test in our own convention cannot catch a wrong sign |
   | any grid-hash or exactness key | `grid_hash_cell_size` and `exact_nearest` live under `mesh:`, never `numerics:` — a mis-sectioned key is silently ignored and `grep` still finds it; verify with `load_config` (see the kelp-experiment "config check"); set the cell to the number `utils/check_grid_hash_cell.py` prints for THIS mesh, never to another mesh's number (structure-resolving ≈ 0.005–0.01°, CORE2 ≈ 0.5°) |
   | any dataset not used before (record 0 may be the initial state of a cold start) | `time.start_time_idx` AFTER spin-up — print the zero fraction of `eta` at records 0/mid/end (periodic forcing needs hours of records); skipping one record is not spin-up |
   | paired A/B arms on two mesh variants of the same domain | `mesh.meshpath` and `data.ncinput` from the SAME file per arm (layout checks match by dimension name and pair meshes silently); seeds in lon/lat, never node ids |
   | the task names variables (`w_cart`, `Wvel`, `u2d`, `sigma_lev`) | that is the FESOM-C family (sigma) — `mesh.type: fesomc`, 3-D tracks with its `w` (`Wvel`; `w_cart` is refused); FESOM2 is `w`/`wo`/`wvel` on z-levels. Reconcile the reported names against the family BEFORE writing the config; a regional FESOM subset may carry `u`/`v` where the global run has `unod`/`vnod` — read `ncdump -h`, never assume |

   **The mesh usually lives apart from the velocity data — do not assume they
   are co-located.** In FESOM, the mesh diagnostic file (`fesom.mesh.diag*.nc`,
   or a mesh directory / Zarr store) and the model output (`u.fesom.YYYY.nc`,
   `v…`, `w…`, `temp…`, `salt…`) are normally in *different* directories.
   `mesh.meshpath` points at the mesh; `data.ncinput` at the velocity data (a
   single file or a directory of per-variable files). Pin both explicitly and
   confirm each path with the user rather than guessing a shared parent.

   **Big meshes, regional subsets, multi-year FESOM2 directories, the I/O knobs (float32,
   prefetch) and PCHIP time accuracy:** [references/data-and-io.md](references/data-and-io.md).

2. **Make the seeds** (`inicloud` is a text file: `lon lat [depth]`, one
   particle per row). Write `make_seeds.py` as ONE runnable script — not
   fragments — and verify it compiles:
   ```python
   import numpy as np
   from kelp.core.config import load_config
   from kelp.readers import get_reader
   from kelp.core.particles_gen import generate_around_point
   from kelp.core.mesh import find_nearest_nodes, find_containing_elements
   from kelp.core.rotation import euler_matrix, rotate_lonlat, unrotate_lonlat
   cfg = load_config("config.yaml"); reader = get_reader(cfg.mesh.type)
   mesh = reader.load_mesh(cfg.mesh, mode_cfg=cfg.mode) if cfg.mesh.type == "healpix" else reader.load_mesh(cfg.mesh)
   R = euler_matrix(*mesh.rotation_angles) if mesh.rotation_angles is not None else None   # mesh.rotation set → the mesh lives in a rotated frame
   lon0, lat0 = (np.atleast_1d(v)[0] for v in rotate_lonlat(np.array([LON0]), np.array([LAT0]), R)) if R is not None else (LON0, LAT0)
   st = generate_around_point(mesh, lon0, lat0, n=N_OVER, radius_km=R_KM, seed=42)  # MESH-frame coordinates; over-seed ~1.3×
   nn = find_nearest_nodes(mesh, st.lon, st.lat)
   elem, inside = find_containing_elements(mesh, st.lon, st.lat, nn)   # off-mesh / land → False
   keep = inside & (mesh.depth[nn] > DEPTH_M + 50.0)                    # release at depth; a surface release drops the depth term
   lon, lat = st.lon[keep][:N], st.lat[keep][:N]
   if R is not None: lon, lat = unrotate_lonlat(lon, lat, R)            # the seed file is GEOGRAPHIC
   print(f"kept {keep.sum()} of {len(keep)}; writing {len(lon)}")
   np.savetxt("seeds.dat", np.column_stack([lon, lat, np.full(len(lon), DEPTH_M)]), fmt="%.8f %.8f %.3f")
   ```
   Then `python -m py_compile make_seeds.py` and run it. For a large or 3-D run, also
   `python utils/sort_seeds.py seeds.dat -o seeds_sorted.dat` and point
   `data.inicloud` at the SORTED file (+17 % on the fused 3-D step); a small 2-D run
   may keep the unsorted file. A lon/lat box instead of a cloud:
   `rng = np.random.default_rng(42); lon = rng.uniform(W, E, N_OVER); lat = rng.uniform(S, N, N_OVER)`
   in place of `generate_around_point`; on a rotated mesh test the box in the mesh frame
   (`mlon, mlat = rotate_lonlat(lon, lat, R)`, then `find_nearest_nodes`/`find_containing_elements`
   on `mlon, mlat`) and save the geographic `lon, lat` of the kept points.
   Seeds are always **geographic** lon/lat, even for rotated-frame runs (the
   tracker rotates them in and writes geographic output back) — but a mesh loaded
   with `mesh.rotation` holds MESH-frame coordinates, so every helper and search above
   works in that frame: rotate the centre (or box) in, and the kept points back out.
   - **Filter seeds by containment AND bathymetry before saving** — the helpers
     check horizontal containment only. A seed released below the local bottom
     is clamped to the bottom on step 0 and then reported as "subducted"; a
     seed on a shelf shallower than the release depth dies as a coastal loss.
     Always drop seeds off the mesh or on land and report the count. For a release
     at depth, also keep only seeds whose nearest-node `mesh.depth` exceeds the
     release depth by a margin (+50 m); a surface release keeps shelf seeds unless the
     owner excludes them (on a store without a depth table `mesh.depth` is 0 and this
     filter would drop every seed). A low
     alive fraction on a fresh seed set is the seeds, not the engine: measure the
     attrition on the target mesh (land, shelf, coast) before blaming the tracker;
     coastal loss is physics, step-size independent, and a regional crop's open
     boundary kills like a coastline.
   - **Other seed shapes** (a cloud around a point, seeds at depth, a regular grid), **spatial
     sorting** for large or 3-D runs and **release over time** (pulses, a per-seed release column):
     [references/seeds.md](references/seeds.md).

3. **Write the config** following the `kelp-config` skill and
   `configuration.md`. Decision guide for the knobs that need judgement:

   - **A store already in use:** by default copy the old config's store keys as
     active lines (`mesh.type`, `meshpath` — on HEALPix the same store as
     `data.ncinput` —, `uv_interp: barycentric`, `var_u`/`var_v`) and change only the
     date, the seeds and the duration. Discrete mode keeps `dt` = the file step; a key
     left as a comment is unset.

   - **Integrator:** `rk2` is the sensible default (good accuracy/cost). Use
     `euler` only for a quick look or to match a reference; `rk4` when time
     accuracy matters and `dt` is well below the file step — by default together with
     `temporal_reconstruction: pchip`, because under the linear time blend rk4
     collapses onto rk2 (an owner who wants rk4 with the linear blend gets it, with
     that note). Rule of thumb: on CORE2-class meshes the temporal error of `rk2 + linear`
     sits BELOW the grid error, so rk4+pchip buys nothing there and costs ×5.5
     wall on the unfused 3-D path; pchip pays on finer meshes. Best on analytic
     tests ≠ best on real fields: validate a method choice on a rough real field
     at the production horizon.
   - **Comparing two settings (dt, integrator, interpolator):** one identical
     filtered seed file, `temporal_interp: true` on BOTH legs (discrete mode
     reads one file record per step and refuses a `dt` unlike the record spacing,
     so it cannot compare step sizes), the same
     `grid_hash_cell_size`, diffusion OFF (the counter RNG is keyed by step
     index — a finer dt draws different noise), `output_freq` chosen so both
     legs record at the same wall-clock cadence; then compare on the common
     survivors and by regime, never by one global median (the
     `kelp-analysis` skill has the recipe). Sub-forcing-timescale physics
     (episodic sinking invisible to daily means) cannot be recovered by
     integrator refinement — change the question, not the step. A SCRIPT that
     sweeps a setting with `track_particles` (a convergence table, a laptop
     sweep) runs each leg like this and writes the engine it ran next to its
     result table — a hash cited in a README or a report must be on main or
     carry a tag, so a branch that will be rebased gets a local tag first
     (pushing it is the owner's call):
     ```python
     rec = track_particles(mesh, ds, reader, lon0, lat0, n_steps=int(round(T / dt)), dt=dt, integrator=INTEGRATOR,
                           temporal_interp=True, file_times=file_times, file_dt=file_dt, output_freq=int(round(RECORD_S / dt)))
     A = rec.to_arrays()   # (T, P), time first; time = seconds since the forcing's first raw record, at the END of each recorded step; record 0 = after output_freq steps — the release is A["release_lon"/"release_lat"/"release_z"/"release_time"] (P,)
     kdir = os.path.dirname(os.path.abspath(kelp.__file__))
     git = lambda *a: subprocess.run(["git", "-C", kdir, *a], capture_output=True, text=True, check=True).stdout.strip()
     engine = {"kelp_file": kelp.__file__, "kelp_version": getattr(kelp, "__version__", "unknown"), "commit": git("rev-parse", "HEAD"),
               "dirty": bool(git("status", "--porcelain", "--untracked-files=no")), "tags": git("tag", "--points-at", "HEAD").split()}
     # a dirty or untagged HEAD is recorded and warned about, never refused: the tag is needed before the hash is CITED
     json.dump(engine, open(os.path.join(out_dir, "engine.json"), "w"), indent=2)   # next to the result table; a README engine line is pasted from it, after the run
     ```
   - **Time regime:** if the forcing has many snapshots and you want smooth
     sub-step motion, `temporal_interp: true` with `dt` < file step. If you want
     one step per snapshot, `temporal_interp: false` with `dt` = file step. A
     long run on short/periodic forcing → `on_exhausted: cycle`.
   - **Diffusion:** none → omit. Constant eddy diffusivity → `scheme: uniform`
     with `kh_zonal`/`kh_meridional` (m²/s) and a `seed`. Spatially-varying `Kh`
     from the dataset → `scheme: em` (Euler–Maruyama, the safe general choice);
     `m1` only for separable `Kh`. **3-D horizontal diffusion: only `uniform` works**
     (vertical Az mixing is separate: `diffusion.az_enabled`, `kelp-config`).
     **Under `mesh.rotation` use ISOTROPIC Kh** (equal scalars/fields; an
     unequal pair rides the rotated axes and the run warns — spatial
     variation is fine as long as the two components are equal).
     `em`/`m1` with `rk2`/`rk4` needs `composition: split_rk`.
   - **Grid-hash cell — an optional check since 2026-09-22** (the run picks the cell and stamps
     it in the output metadata): for a release in a refined region of a non-uniform mesh or an
     explicit cell, see [references/grid-hash-cell.md](references/grid-hash-cell.md).
   - **Release or expected drift poleward of 60 °N/S (FESOM2/HEALPix):** by default add
     `mesh.rotation: fesom2` so the lon/lat metric does not degenerate — a 72 °N
     Nordic Seas release qualifies, a 60 °N Iceland Basin release is the
     boundary case (set it); output stays geographic.
   - **Output:** `.nc` (NetCDF) or `.zarr`; long runs → add `checkpoint_freq`
     (resumable restart). **If you want to watch a long run's output as it
     grows, or it might be interrupted:** a plain `.nc` is HDF5-locked and
     unreadable by other processes until it closes — use `.zarr` (lock-free,
     readable mid-run) or `numerics.output_flush_every: N` (buffered NetCDF,
     readable between flushes; e.g. 10). Both are byte-identical to the default
     and roughly the same speed (Zarr block-buffers; NetCDF buffered ≈ +3%).
     Velocities (`vel_u/vel_v`, `vel_w` in 3-D only; NaN unless the particle was ALIVE through
     the step), the release (`release_lon/lat/z/time`) and any `sample_vars` ride along
     automatically; `time` carries the forcing's epoch and calendar (output format 2 — reading it:
     `kelp-analysis`). Turn on `mode.track_failure_reason` for any run with
     losses (crop edge vs interpolation failure). **Bottom contact is a clamp,
     not a death**: a particle on the sea floor stays `ALIVE` at the clamped
     depth (`SURFACE_HIT`/`BOTTOM_HIT` are reserved, unimplemented) — detect
     it from `z` vs local bathymetry, never from `status`.
   - **Run longer than one job's wall clock** (a multi-decade run vs. an 8-hour
     compute limit): `checkpoint_freq` alone resumes from the *latest* embedded
     checkpoint, but a chained run wants a **kept history** of restarts it
     manages itself. The building block is `kelp.writers.write_restart_file`
     (a standalone, numbered NetCDF restart per checkpoint, read back by the same
     `load_restart`). The **resumable-chain** pattern around it: run in chunks,
     write one trajectory segment + one numbered restart per chunk, resume from
     the latest restart, chain the legs with `--dependency=afterany` (a wall-killed
     job cannot resubmit itself), then concatenate the segments
     — a no-change resume is bit-for-bit identical to one uninterrupted run. This
     lives in a small run driver, not in the CLI; see
     [`docs/io_and_time.md`](../../../docs/io_and_time.md) §6 for the recipe.

4. **Verify before the real run** (cheap):
   ```bash
   python <clone>/.claude/skills/kelp-run/scripts/w_sign.py path/to/config.yaml    # 3-D, new dataset: the SIGN of the file's w — write the number in PLAN.md
   python <clone>/.claude/skills/kelp-run/scripts/record0.py path/to/config.yaml   # a dataset not used before: ssh zero fraction at records 0 / mid / end; all-zero at 0 = cold start → start after spin-up (skipping one record is not spin-up)
   kelp info config.yaml                 # mesh loads, data opens, time range
   python <clone>/utils/check_grid_hash_cell.py path/to/config.yaml   # OPTIONAL since 2026-09-22: the run picks the cell itself (mesh.grid_hash_cell_rule: occupancy, printed as `grid hash: cell ...`); use it for explicit candidates or to probe the SEED positions
   kelp run config.yaml -v               # then a short n_steps smoke
   ```
   Read the startup summary lines (resolved `dt`, file step, `time_averaging`
   shift, diffusion composition, which files/areas were used). **3-D: confirm
   W was actually read** — FESOM2 proceeds SILENTLY without vertical velocity
   when the W name does not resolve (even a misspelled explicit `var_w`); pin
   `mode.var_w` and check that `vel_w` in the smoke output exists and is non-zero where finite
   (`np.nanmax(np.abs(...))`; NaN = not ALIVE through the step) and `z` changes. Checklist
   lines for the handback, each with its condition:
   - [ ] (3-D, a vertical velocity not used before) w sign: mean w = ____ m/s → `w_positive: ____`
   - [ ] (a dataset not used before) record-0 zero fraction = ____ → `start_time_idx: ____`
   The config keeps `w_positive: auto` until the sign is measured — a `____` in config.yaml fails
   `load_config`, and make_seeds.py and w_sign.py read that file.

   When a condition holds, `command.sh` stops (`exit 1`) until that blank is filled —
   the full run is never the first command of such a script; when neither holds
   (a 2-D run, a store already in use), no blank gate: the run line stays live behind
   the owner's yes and the handback says why each check was skipped:
   ```bash
   # only the lines whose condition holds
   grep -Eq '^ *w_positive: *(up|down)' config.yaml || { echo "decide mode.w_positive with w_sign.py first"; exit 1; }   # 3-D, a vertical velocity not used before
   grep -Eq '^ *w_interp: *(barycentric|nearest)' config.yaml || { echo "3-D node layout + rk2: set numerics.w_interp"; exit 1; }   # 3-D node layout with rk2
   START_IDX=____                                                                                                        # a dataset not used before: from record0.py
   if [[ "$START_IDX" == "____" ]]; then echo "run the record-0 check first"; exit 1; fi
   ```
   Both checks are scripts next to this skill, run as they are (each loads the mesh like
   `kelp info` — a job on a big mesh): `python <clone>/.claude/skills/kelp-run/scripts/w_sign.py path/to/config.yaml`
   and the same for `record0.py`. In command.sh `w_sign.py` runs uncommented before the grep, and
   `w_positive` is written only from its printed line — "the parent model is positive up" is the
   family default, which the gate cannot tell from a measurement. The leading minus in
   `w_sign.py` turns KELP's positive-down w back into the file's w, and 50 m at 2° S–2° N is where
   the sign is unambiguous (upwelling) — not the release depth; on a regional store without that
   box it stops and asks for a region of known sign.

5. **Hand over the run command — but do not start it yourself.**
   Prepare everything and *show* the exact command; then **ask a direct yes/no
   question and wait for a direct answer before launching anything.** This holds
   for **both** targets — a local run and a SLURM job are equally gated. A run
   described inside the task ("run it on 128 CPU", "launch this", "go") is **not**
   consent; you must still ask "shall I run/submit this now?" and get an explicit
   "yes". Starting a run (`kelp run …`, `NUMBA_NUM_THREADS=… python -m …`,
   `sbatch …`) or stopping one (`scancel …`) is the gated action; writing the
   files and showing the command is not.
   The pre-launch checks go in the file the task names (`CHECKS.md`), in order, and
   its first command is `kelp info config.yaml`. The handback ends with this
   literal line — it states what did NOT happen; reproduce it verbatim, never replace
   it with a result line:
   ```
   Launch gate: nothing was run; say "yes" to run the smoke test.
   ```
   - **Laptop / interactive:** `NUMBA_NUM_THREADS=<cores> kelp run config.yaml -o out.nc -v`
   - **SLURM (HPC):** a small sbatch — one timed run, with
     `OMP/MKL_NUM_THREADS=1` so Numba parallelism is what runs, and
     `NUMBA_NUM_THREADS` set in ONE place (never only via SLURM directives;
     a 256-logical-CPU node's default makes small kernels 25–50× slower):
     ```bash
     #SBATCH --partition=<compute partition> --account=<acct> --nodes=1 --exclusive --mem=0 --time=02:00:00
     #SBATCH --output=%x_%j.log --error=%x_%j.err        # job-id-keyed logs: two runs never overwrite each other
     export NUMBA_NUM_THREADS=128 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1   # see the thread rule below
     /usr/bin/time -v python -u -m kelp.cli.main run config.yaml -v  # -v prints the per-step Timing summary; -u = unbuffered log
     ```
     In a batch job the launch line is ALWAYS `python -u -m kelp.cli.main run …`
     — a bare `kelp run` in an sbatch loses the buffered log on a wall kill.
     **Thread rule:** 128 (= physical cores; not the 256 logical) only for the
     FUSED paths (2-D leastsq/bary, 3-D RK2 node-layout barycentric with
     `w_interp` set); the UNFUSED 3-D paths (rk4, sigma, `w_interp: leastsq`,
     `walk`) are memory-bandwidth-bound and saturate at 16–32 threads — more
     cores only queue, so use the shared partition with 16–32 CPUs, or shard the particles
     into several processes (the `kelp-experiment` skill). A 20 k-particle
     run is I/O-bound at any thread count. **Partition:** the shared partition is
     the default for agent-driven legs — there memory is granted PER CPU (read
     the per-CPU default and the wall limit from `scontrol show partition
     <name>`), so `--cpus-per-task` is the RAM lever as well as the thread
     lever; an exclusive full node only for timing-sensitive or full-node work,
     with `--mem=0` always (`--hint=nomultithread` otherwise halves the node's
     memory). Extraction on the shared partition, tracking on the compute
     partition; ≤ 5 concurrent jobs, one node.

## Output to give the user

State the decisions you made and why (one line each), then hand over: the seed
file, the config (or its path), and the exact command. Keep each run
**self-contained in its own directory** (seeds + config + output + log together)
so experiments do not clobber each other.

## Worked example (intent → setup)

Task: *"backtrack 500 floats from the Lofoten eddy at 3°E 70°N, 3D, sampling
temperature, over the past 60 days on the Arctic monthly means."*
→ FESOM2 3D (`Fdrift3D: true`, `max_levels: null` — trim only if RAM forces it),
`mesh.rotation: fesom2` (poleward of 60°), `time_dir: -1.0`,
`sample_vars: [temp]`, `temporal_interp: true` (12 monthly snapshots),
`dt: 86400`, `n_steps: 60`, `time_averaging: mean`; seeds = the rotate-in /
unrotate-out recipe of step 2 at (3, 70), n=500, plus a depth column if a
release depth is given; output `.nc`; command = the SLURM sbatch at 16 threads
(500 particles do not fill a node — the thread scan decides). Verify with
`kelp info` first.

Task: *"release 200 particles near 5°E 60°N every day for a week and track them
forward for a month."*
→ FESOM2 2D, `time_dir: 1.0`, `temporal_interp: true`, `dt: 3600`; seeds =
`generate_around_point` at (5, 60), n=200 (one pulse's worth); delayed release
via `time.n_pulses: 7`, `pulse_interval: 86400` (→ 1400 particles, a pulse per
day; pulses on days 0…6, last release at day 6); `n_steps` set so `n_steps·dt`
covers the last release (day 6) + 30 days of drift = 36 d, i.e. `n_steps = ((7 − 1)·86400 + 30·86400)/3600 = 864`. Frozen particles read
as NaN until their day. Ready template (continuous pulsed release):
`examples/config_global_2d_10yr.yaml`.
