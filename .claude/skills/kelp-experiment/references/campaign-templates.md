# Campaign templates (copy these verbatim, then fill the slots)

The files and scripts every campaign folder needs: AGENTS.md, the job-script preamble, the multi-shard
srun block, the dependency chain, the resumable leg driver, the config check, the config generator,
the steady-step extractor, LEDGER.md and REPORT.md.

## Contents
- `AGENTS.md`, the first file of every campaign folder
- Job-script preamble
- Multi-node, multi-shard leg (the `srun` block)
- Chain submission (sequential legs)
- `leg_driver.py` — the resumable chain driver (the CLI cannot resume); `resume_check.sh` — resume equals continuous, bit for bit
- Config check
- Generator (`gen_configs.py`)
- Steady-step extractor
- `LEDGER.md`
- `REPORT.md`

**`AGENTS.md`, the first file of every campaign folder:**
```markdown
# AGENTS.md — <campaign>  (rules first, data gotchas second)
1. No heavy work on the login node; every heavy step is a SLURM job.
2. No `sbatch`/`srun`/`kelp run` without the run owner's explicit yes for THAT command.
3. Read-only foreign data: <dir> (belongs to <owner>) — never write there.
4. Transient files: ./.tmp/ inside the campaign folder only — never the system
   temp directories.
5. Engine: clone <path> @ <commit> via PYTHONPATH; env <name>; branch <name>.
## Data gotchas   (fill as found: cadence, record 0, variable names, meshes)
## Status log / Next session starts here
```

**Job-script preamble (verbatim in every sbatch; the `#SBATCH` lines come first — sbatch ignores every
directive after the first command, `set -e` included):**
```bash
#SBATCH --partition=<THIS cluster's partition> --account=<THIS cluster's account>   # never carried over from another cluster
export PYTHONPATH=<clone>/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-<n>} NUMBA_CACHE_DIR=<campaign>/.tmp/numba_$SLURM_JOB_ID   # per job = ONE process only; a multi-task srun step sets it per SLURM_PROCID INSIDE the task (concurrent compiles into one cache crash)
python -c "import kelp; print('engine', kelp.__file__)"; git -C <clone> rev-parse --short HEAD
git -C <clone> status --porcelain --untracked-files=no   # empty = clean; the store's kelp_version carries no dirty flag
python -u -m kelp.cli.main run <config> -v > logs/<leg>_$SLURM_JOB_ID.log 2>&1
```

**Multi-node, multi-shard leg — shards are SRUN TASKS, never background jobs
(a `for … &` loop runs the whole allocation on ONE node). `$SLURM_PROCID`/`$SLURM_LOCALID`
stay inside the SINGLE quotes: the task expands them, not the submitting shell — pre-expanded,
every task runs shard 0. The `numactl` prefix is valid ONLY when `PER_NODE` equals the node's
NUMA domain count (`lscpu | awk '/^NUMA node\(s\)/{print $3}'`); otherwise delete it and rely
on `--cpu-bind=cores,verbose`, recording the achieved binding from the probe log:**
```bash
export NODES PER_NODE LEG RESTART_EVERY LEG_BUDGET_S   # sbatch-set variables reach the tasks ONLY if exported (srun propagates the environment)
srun --wait=0 --ntasks=$((NODES*PER_NODE)) --ntasks-per-node=$PER_NODE \
     --cpus-per-task=$((128/PER_NODE)) --cpu-bind=cores,verbose \
     bash -c 'export NUMBA_CACHE_DIR=<campaign>/.tmp/numba_${SLURM_JOB_ID}_${SLURM_PROCID} NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK; printf -v K %03d $SLURM_PROCID; sleep $((SLURM_LOCALID*10)); numactl --cpunodebind=$SLURM_LOCALID \
       --membind=$SLURM_LOCALID python -u -m kelp.cli.main run \
       configs/shard_$K.yaml -v > logs/leg${LEG}_p${SLURM_PROCID}_$SLURM_JOB_ID.log 2>&1'
```

**Chain submission (legs are SEQUENTIAL: never `--array`, never a self-`sbatch` at
the end of a job — a wall-killed job cannot resubmit itself):**
```bash
PREV=""; for LEG in $(seq 1 $N_LEGS); do
  DEP=${PREV:+--dependency=afterany:$PREV}
  PREV=$(sbatch --parsable $DEP --export=ALL,LEG=$LEG chain.sbatch); echo "| $PREV | leg $LEG | … |" >> LEDGER.md
done
```

**The CLI cannot resume — a chain leg runs THIS driver, never `kelp run`** (a leg
that calls the CLI re-runs leg 1 forever):
```python
# leg_driver.py  <config> <segment_dir> <restart_dir> <restart_every_steps> <leg_budget_s>
import glob, inspect, os, re, sys, time, warnings
import numpy as np
from kelp.core.config import load_config
from kelp.core.mesh import set_grid_hash_cell, set_exact_nearest, apply_grid_hash_cell_rule
from kelp.readers import get_reader, extract_time_info, _time_option_kwargs, _dry_mask_kwargs, _prepare_dataset, _READER_RESOLVES_TIME
from kelp.tracking import run_single_realization, _apply_max_levels, _shift_to_midpoints, _validate_restart_frame, _validate_restart_uv_vertical, _shared_fields_tag, _resolve_particle_offset, _resolve_start_date
from kelp.writers import TrajectoryWriter, load_restart, write_restart_file
cfg = load_config(sys.argv[1]); SEG, RDIR, EVERY, BUDGET = sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5]); T0 = time.time()
os.makedirs(SEG, exist_ok=True); os.makedirs(RDIR, exist_ok=True)   # a missing restart dir fails AFTER the first chunk, and the next leg re-runs it
reader = get_reader(cfg.mesh.type)
time_kw = _time_option_kwargs(cfg)   # the fesom14 stamp options; refuses data.time_averaging there (the reader resolves the record times)
poff = _resolve_particle_offset(cfg)   # the shard's place in the release: keys its random draws (header or data.particle_offset)
mesh = reader.load_mesh(cfg.mesh, mode_cfg=cfg.mode) if "mode_cfg" in inspect.signature(reader.load_mesh).parameters else reader.load_mesh(cfg.mesh)   # healpix, fesom14
# --- what run_tracking does after load_mesh; the CLI would do it for you, a driver must not skip it ---
set_exact_nearest(mesh, bool(cfg.mesh.exact_nearest))                 # before the cell rule, whose mode depends on it
if cfg.mesh.grid_hash_cell_size is not None: set_grid_hash_cell(mesh, cfg.mesh.grid_hash_cell_size); CELL_RULE = "explicit"
else: CELL_RULE = apply_grid_hash_cell_rule(mesh, cfg.mesh.grid_hash_cell_rule)["rule"]   # the automatic cell, as run_tracking chooses it
_apply_max_levels(mesh, cfg.mode)   # the level cut as run_tracking makes it (keeps the full table in metadata)
d = cfg.numerics.diffusion; az_family = cfg.mesh.type == "fesom2" and d.enabled and d.az_enabled and d.var_az
ds = reader.open_dataset(cfg.data.ncinput, v_path=cfg.data.ncinput_v or None, w_path=cfg.data.ncinput_w or None, mode_cfg=cfg.mode,
                         **({"eta_path": cfg.data.ncinput_eta} if getattr(cfg.data, "ncinput_eta", "") else {}),
                         **({"diffusion": d} if az_family else {}),   # multi-year FESOM2: assembles var_az too
                         **time_kw, **_dry_mask_kwargs(cfg))          # fesom14: stamp options + the beaching dry mask (empty elsewhere)
_prepare_dataset(reader, mesh, ds)   # the reader's dataset check, before the writer exists (run_single_realization repeats it)
user_sv = list(cfg.mode.sample_vars or []); b = cfg.numerics.beaching     # sampled variables + the internal dry mask
read_sv = user_sv + ([b.dry_mask_var] if b.enabled and b.dry_mask_var and b.dry_mask_var not in user_sv else [])
n_time_total = ds.sizes["time"]; ft, file_dt = extract_time_info(ds)
ft = _shift_to_midpoints(ft) if cfg.mesh.type not in _READER_RESOLVES_TIME and cfg.data.time_averaging == "mean" else ft   # fesom14 records already sit at their window centres
shared = None
if cfg.numerics.shared_fields_dir:                      # the node-local cache, exactly as run_tracking builds it
    from kelp.core.shared_fields import SharedFieldCache
    try: shared = SharedFieldCache(cfg.numerics.shared_fields_dir, _shared_fields_tag(cfg))
    except Exception as exc: warnings.warn(f"shared_fields disabled: {exc!r}", RuntimeWarning)
dt = float(cfg.numerics.dt)   # must be explicit (the generator writes it; the config check asserts it)
start_idx, t_start = _resolve_start_date(ds, cfg.time.start_date, ft, cfg.numerics.temporal_interp)   # START_DATE exactly as run_tracking (and `kelp run` on this config) resolves it
assert cfg.mode.time_dir == 1.0, "this driver is forward-only: resume_point does not walk the axis backward"
assert cfg.time.on_exhausted == "stop", "this driver does not resume cycled forcing: resume_point clamps at the last record"
def resume_point(gstep):
    sim_t = t_start + gstep * dt
    if not cfg.numerics.temporal_interp:               # discrete: one record per step, so the record IS start + global step
        return start_idx + int(gstep), sim_t           # (a time search repeats a record when the stamps jitter within the dt check; past the end the run stops at once)
    fidx = int(np.searchsorted(ft, sim_t, side="right") - 1)   # temporal_interp: the bracket of that sim time
    return fidx, sim_t
files = glob.glob(os.path.join(RDIR, "restart_step*.nc"))
if files:                                    # RESUME: state from the newest numbered restart
    r = load_restart(max(files, key=lambda f: int(re.search(r"step(\d+)", os.path.basename(f)).group(1))))
    _validate_restart_frame(r["rotation_angles"], mesh)
    _validate_restart_uv_vertical(r["uv_vertical"], cfg.mode)   # one mode.uv_vertical rule per trajectory
    if (r["particle_offset"] or 0) != poff: sys.exit(f"restart keyed with particle_offset {r['particle_offset']}, config gives {poff}")   # None = an older driver, keyed from 0
    state, gstep, wet = r["state"], r["step_i"] + 1, r["wet_streak"]
else:                                        # FRESH START: the three calls run_tracking makes, verbatim
    from kelp.core.release import read_seeds_with_release, build_release_state
    from kelp.tracking import _maybe_rotate_seeds
    lon, lat, z, rel = read_seeds_with_release(cfg.data.inicloud)
    state = build_release_state(lon, lat, z, rel, n_pulses=cfg.time.n_pulses,
                                pulse_interval=cfg.time.pulse_interval, dt=dt, start_step_i=0)
    state, gstep, wet = _maybe_rotate_seeds(state, mesh), 0, None
    if cfg.mode.track_failure_reason:                  # run_tracking allocates it; without it no failure_reason is written
        state = state.replace(failure_reason=np.zeros(state.n_particles, dtype=np.int8))
angles = mesh.rotation_angles
while gstep < cfg.time.n_steps and time.time() - T0 < BUDGET:   # BUDGET = wall − 20 min: stop on the budget, never let SLURM kill a chunk
    chunk = min(EVERY, cfg.time.n_steps - gstep); seg = os.path.join(SEG, f"part_{gstep:08d}.zarr")
    fidx, sim_t = resume_point(gstep)
    meta = {"integrator": cfg.numerics.integrator, "dt": dt, "mode_3d": int(bool(cfg.mode.Fdrift3D)),
            "time_averaging": cfg.data.time_averaging, "exact_nearest": int(bool(cfg.mesh.exact_nearest)),
            "grid_hash_cell": float(mesh._gh_cell_size), "grid_hash_cell_rule": CELL_RULE,
            **({"n_pulses": int(cfg.time.n_pulses)} if cfg.time.n_pulses > 1 else {}),   # lets the merge refuse pulsed shards
            **({"diffusion_enabled": 1, "diffusion_seed": int(d.seed)} if d.enabled else {}),   # lets the merge check seeds
            **({"beaching_enabled": 1, "beaching_criteria": ",".join(b.criteria), "beaching_resuspension": b.resuspension,
                "beaching_seed": int(b.seed if b.seed is not None else d.seed)} if b.enabled else {}),
            **({"rotation_angles": list(angles)} if angles else {})}   # run_tracking's provenance; run_single_realization adds the rest
    w = TrajectoryWriter(seg, state.n_particles, metadata=meta); w.open()
    try:
        run_single_realization(mesh, ds, reader, state, cfg.mode, cfg.numerics, start_time_idx=fidx, n_steps=chunk,
                               n_time_total=n_time_total, on_exhausted=cfg.time.on_exhausted, verbose=True, writer=w,
                               shared_cache=shared, checkpoint_freq=chunk, temporal_interp=cfg.numerics.temporal_interp,
                               file_times=ft, file_dt=file_dt, sim_time_start=sim_t, start_step_i=gstep, start_wet_streak=wet,
                               sample_vars=read_sv, sample_interp=cfg.numerics.sample_interp, output_sample_vars=user_sv,
                               particle_offset=poff)
    finally:
        w.close()
    try: rr = load_restart(seg)
    except ValueError: print("FORCING EXHAUSTED before step", gstep, flush=True); break   # no step ran: n_steps exceeds the forcing
    state, wet = rr["state"], rr["wet_streak"]
    rfile = os.path.join(RDIR, f"restart_step{rr['step_i']:08d}.nc")           # numbered, never overwritten
    write_restart_file(rfile + ".tmp", state, rr["step_i"], rr["file_idx"], wet_streak=wet,
                       rotation_angles=angles, release_step=state.release_step, particle_offset=poff,
                       uv_vertical=cfg.mode.uv_vertical); os.replace(rfile + ".tmp", rfile)
    exhausted = rr["step_i"] + 1 < gstep + chunk                                   # the forcing ended inside this chunk
    gstep = rr["step_i"] + 1
    if exhausted: print("FORCING EXHAUSTED at step", gstep, flush=True); break
print("LEG COMPLETE" if gstep >= cfg.time.n_steps else "LEG STOPPED", gstep, flush=True)
```
The fresh-start branch is the seed loader + pulse replication + `_maybe_rotate_seeds` exactly as
`run_tracking` calls them — never re-implement the seed rotation by hand. Output format 2 needs no
line here: the release positions ride in the state through `load_restart` → `write_restart_file`,
so every segment writes the same `release_*`, and the `time` labels equal the uncut run's because
every leg passes the same shifted `ft` and the global step. The per-leg `chain.sbatch`
calls `python -u leg_driver.py configs/shard_$K.yaml segments/p$K restarts/p$K $RESTART_EVERY $LEG_BUDGET_S`
inside the `srun` block above, in place of `python -u -m kelp.cli.main run …`; `LEG_BUDGET_S`
= the job's `--time` minus 20 minutes, in seconds. The resume point is computed from the GLOBAL
step: in discrete mode the record is `start_idx + gstep` (one record per step — a `searchsorted` on
`t_start + gstep·dt` lands one record early whenever the stamps sit a fraction of a second off the
`dt` grid, which the discrete-dt check allows, and that leg repeats a record); with
`temporal_interp` the bracket of `sim_time = t_start + gstep·dt` by `searchsorted` (`dt < file_dt`
has no one-record-per-step index). A stronger reader found the first version of this block lacked the grid-hash cell,
the exact-anchor guard, `max_levels`, the shared cache and the budget stop — every one a
silent divergence from the CLI run; keep those lines. What `run_tracking` does and this
driver does NOT: part of the writer metadata (`diffusion_*`, `beaching_*`, sampled-variable
units; the driver writes the rest, and `run_single_realization` stamps `particle_offset`,
`random_draws`, `kelp_version` and the time treatment itself) and the startup fail-fasts (`use_eta` without ssh, a missing sample variable) — run one
`kelp info` / short CLI smoke on the same config first, and add the metadata if the
analysis reads it.

**Resume equals continuous — the check is this script in the run folder**, run once on a compute
node before the chain is submitted (a PLAN line saying "compare bit for bit" is not the check):
```bash
# resume_check.sh <config> <N>  — leg 1 + resume vs one continuous run of 2N steps, bit for bit; N = a multiple of output_freq
set -euo pipefail
CFG=$1; N=$2; W=.tmp/resume_check
rm -rf "$W"; mkdir -p "$W"                                    # this check's own scratch folder, nothing else
python - "$CFG" "$W" "$N" <<'EOF'
import sys, yaml
y = yaml.safe_load(open(sys.argv[1])); t = y.setdefault("time", {}); n = int(sys.argv[3])
t.pop("end_time_idx", None); t.pop("duration_days", None)
for name, steps in (("smoke_1.yaml", n), ("smoke_2.yaml", 2 * n)):   # the leg driver stops at n_steps
    t["n_steps"] = steps; yaml.safe_dump(y, open(f"{sys.argv[2]}/{name}", "w"))
EOF
python -u leg_driver.py "$W/smoke_2.yaml" "$W/cont" "$W/cont_r" $((2 * N)) 1e9   # continuous: one chunk of 2N steps
python -u leg_driver.py "$W/smoke_1.yaml" "$W/res" "$W/res_r" "$N" 1e9           # leg 1: N steps, numbered restart
python -u leg_driver.py "$W/smoke_2.yaml" "$W/res" "$W/res_r" "$N" 1e9           # leg 2: resumes from that restart, N more
python - "$W" <<'EOF'
import glob, sys
import numpy as np, xarray as xr
W = sys.argv[1]; op = lambda p: xr.open_zarr(p, consolidated=False, decode_times=False)   # time in seconds, units in .attrs
cont = op(glob.glob(f"{W}/cont/part_*.zarr")[0]); segs = [op(p) for p in sorted(glob.glob(f"{W}/res/part_*.zarr"))]
assert len({int(s.attrs.get("kelp_output_format", 1)) for s in segs + [cont]}) == 1 and len({s["time"].attrs.get("units") for s in segs + [cont]}) == 1
REC = cont["lon"].dims[1]                          # "time" (output format 2) or "obs" (format 1)
res = xr.concat(segs, dim=REC, data_vars="minimal", coords="minimal", compat="override", join="override")
assert res.sizes[REC] == cont.sizes[REC] > 0, (dict(res.sizes), dict(cont.sizes))
for v in ("time", "lon", "lat", "z", "status"):
    assert np.array_equal(np.asarray(cont[v]), np.asarray(res[v]), equal_nan=True), f"{v} differs after the resume"
REL = [v for v in ("release_lon", "release_lat", "release_z", "release_time") if v in cont]   # format 2 only
for s in segs:                                     # every segment carries the release of the whole run
    for v in REL:
        assert np.array_equal(np.asarray(cont[v]), np.asarray(s[v]), equal_nan=True), f"{v} differs in a segment"
print(f"RESUME == CONTINUOUS (bit for bit): {res.sizes[REC]} records, {res.sizes['trajectory']} particles")
EOF
```
Its last line goes into the campaign README; a failure stops the chain until it is understood.

**Config check — run immediately after generating, on every config (a failing
assert is the expected first outcome; soften nothing):**
```python
from kelp.core.config import load_config; import os
NODE_TMPFS = NODE_TMPFS_ROOT                          # the same root the generator was given
c = load_config(p)
if CELL is None:                             # the automatic cell (occupancy rule, the default since 2026-09-22)
    assert c.mesh.grid_hash_cell_size is None and c.mesh.grid_hash_cell_rule == "occupancy"
else:                                         # an explicit cell chosen with check_grid_hash_cell.py
    assert isinstance(c.mesh.grid_hash_cell_size, float) and c.mesh.grid_hash_cell_size == CELL
assert c.mesh.exact_nearest is True
assert os.path.exists(c.data.inicloud) and os.path.isdir(os.path.dirname(c.data.output_path))   # load_config makes paths absolute; check they exist
assert isinstance(c.numerics.dt, float) and c.time.n_steps == N_STEPS
assert c.time.start_date == START_DATE and START_IDX > 0   # START_IDX = the record of START_DATE (generator); spin-up skipped
assert (c.numerics.shared_fields_dir is not None) == USE_SHARED_CACHE   # the table's verdict, in the config
if USE_SHARED_CACHE:
    assert c.numerics.prefetch_size == 1 and NODE_TMPFS and c.numerics.shared_fields_dir.startswith(NODE_TMPFS)  # node-local root only (NODE_TMPFS as in the generator)
if IS_HEALPIX_3D:                            # raw/kerchunk store: the vertical table is mesh.levels, NOT mode.max_levels
    assert c.mesh.levels is not None, "generator must write mesh.levels (aux3d path or inline list); extracts embed it"
```
The generator's YAML template therefore carries the line `levels: <aux3d path or [0.0, 5.0, …]>`
under `mesh:` for HEALPix 3-D — as a written key, not as a comment.

**Generator — the config dict every shard config is written from (copy; the keys
that decide the campaign live HERE, not in a comment of the config check):**
```python
# gen_configs.py <node_tmpfs_root>  (writes configs/shard_<kkk>.yaml; run BEFORE the config check)
import os, sys, yaml, numpy as np
from kelp.readers import get_reader
# START_IDX: the RAW record of START_DATE on the forcing axis, resolved once here
_ax = get_reader("healpix").open_dataset(STORE)["time"].values
START_IDX = int(np.flatnonzero(_ax == np.datetime64(START_DATE))[0])   # IndexError = START_DATE is not a record
NODE_TMPFS = sys.argv[1] if len(sys.argv) > 1 else ""       # the node-local in-memory root of the target partition, same path on every node
assert not USE_SHARED_CACHE or (os.path.isabs(NODE_TMPFS) and not NODE_TMPFS.startswith(os.path.expanduser("~"))), \
    "shared cache: pass the node-local root (absolute, not $HOME, not a shared filesystem); config values do not expand $VARS"
for k in range(N_SHARDS):
    cfg = {
      "mode": {"Fdrift2D": False, "Fdrift3D": True, "time_dir": 1.0, "max_levels": None,
               "var_u": VAR_U, "var_v": VAR_V, "var_w": VAR_W, "track_failure_reason": True},
      "numerics": {"dt": DT, "integrator": "rk2", "temporal_interp": True, "output_freq": OUTPUT_FREQ,
                   "checkpoint_freq": CHECKPOINT_FREQ, "uv_interp": "barycentric", "w_interp": "barycentric",
                   "elem_search": "local", "prefetch_size": 1 if USE_SHARED_CACHE else 8,
                   **({"shared_fields_dir": f"{NODE_TMPFS}/kelp_cache_{RUN_TAG}"} if USE_SHARED_CACHE else {})},
      "mesh": {"type": "healpix", "meshpath": STORE, "rotation": "fesom2", "exact_nearest": True,
               "grid_hash_cell_size": CELL,                      # None = the automatic cell; a float only when check_grid_hash_cell.py chose one
               **({"levels": LEVELS} if IS_HEALPIX_3D else {})}, # aux3d path or inline list; extracts embed it
      "data": {"ncinput": STORE, "ncinput_eta": ETA_STORE, "inicloud": f"{SEEDS}/seeds_shard{k:03d}.dat",   # split_seeds.py names
               "output_path": f"{OUT}/shard_{k:03d}.zarr", "time_averaging": "mean"},
      "time": {"start_date": START_DATE, "n_steps": N_STEPS, "on_exhausted": "stop"},   # the leg driver and `kelp run` resolve it the same way
    }
    yaml.safe_dump(cfg, open(f"configs/shard_{k:03d}.yaml", "w"), sort_keys=False)
```
Shard file names are zero-padded the SAME way (three digits) in the generator, the seeds
split (`utils/split_seeds.py` writes `seeds_shard<kkk>.dat`) and the `srun` line
(`printf -v K %03d $SLURM_PROCID` inside the task) — the seam between two correct
blocks is where legs died.

**Steady-step extractor (median of read-free steps, worst process):**
```bash
for f in logs/<leg>_p*_$JOB.log; do grep -oE "\(io=[0-9.]+ step=[0-9.]+" $f | awk -F'[= ]' 'NR>1 && $2<500 {print $4/1000}' | sort -g | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; done | sort -g | tail -1   # seconds; io=/step= are ms in the log
```

**`LEDGER.md`, created before the first submit, one row per submit:**
```markdown
| job id | leg | config | partition | cores/threads | node | elapsed | state | note |
|---|---|---|---|---|---|---|---|---|
```
`sacct -j <ids> -o JobID,Elapsed,State,MaxRSS,AllocCPUS > logs/sacct_<date>.txt` the day a job ends.

**`REPORT.md`, created EMPTY at campaign start, filled as results land (never delete a heading):**
```markdown
# <campaign> — report
## Results
## Provenance of every number      (table/figure → script → output store → job id)
## Experiments not performed       (each with what it would cost to run)
## Job ledger + sacct snapshot
```
