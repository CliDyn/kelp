# KELP

**Kinematic Engine for Lagrangian Particles.**

Lagrangian particle tracking quantifies transport pathways, connectivity and
dispersion from the velocity field of an ocean model. Applying it to
high-resolution simulations raises two difficulties: mesh geometry and data
volume. Global models now resolve kilometres on unstructured meshes of tens of
millions of nodes. Regional models resolve metres on mixed meshes of triangles
and quadrilaterals. Their output is distributed on the native mesh, or remapped
to a common grid such as the equal-area HEALPix grid.

KELP is a particle tracker for such output, written in Python with compiled
kernels. It reads triangular meshes, mixed meshes of triangles and
quadrilaterals, among them those of the FESOM family, and HEALPix stores. One
set of compiled time-stepping kernels serves all of them, with fixed depth levels or
terrain-following sigma layers in the vertical. Trajectories are integrated
forward or backward in time, in two or three dimensions, with an optional random
walk. Its random draws are keyed to the particle and the time step, so
stochastic trajectories are reproducible bit for bit across thread counts and
restarts.

For scale: six million particles on the HEALPix remap of a kilometre-scale model
(about 13 km pixels, 2.2 million wet surface nodes) with 69 layers, one model year at a six-hour step, took about two
hours on a single 128-core node with input and output included.

## How we actually use it: we tell an agent

We run KELP almost entirely through an LLM coding agent, and the repository is
built for that. A session usually looks like this. Start the agent in a clone of
this repository. Say in ordinary words what you want and where the data is. Let
it work.

> Release 50 000 particles across the Fram Strait on 1 March, track them
> backward for 90 days in 3D, sample temperature and salinity along the way,
> and write the output into a new folder beside the model output.

The agent writes the seed file, writes the YAML configuration, checks it against
the data you pointed it at, and gives you the launch command or the batch job.
Afterwards it reads the output, draws the figures and answers questions about
them. You stay in the conversation instead of in the file format.

What makes this work is not the model but seven **skills** under
[`.claude/skills/`](.claude/skills/). A skill is a plain Markdown instruction
file that tells an agent how to do one part of the job properly, including the
traps we have already walked into.

| skill | what the agent does with it |
|---|---|
| `kelp-run` | turn a described experiment into seeds, a config and the exact launch command |
| `kelp-config` | write or review one configuration file, option by option |
| `kelp-experiment` | the campaign layer for a big or long run: isolate the setup, crop a region, validate before scaling, run as a resumable chain across wall-clock limits |
| `kelp-analysis` | read trajectory output at the chunking the writer used; cohort statistics, denominators, censoring |
| `kelp-visualize` | figures, montages and videos from trajectory output |
| `kelp-diagnose` | a run is slow, silent or scales oddly: what to check, in order, before touching code or config |
| `kelp-bench` | one comparable benchmark row for the machine you are on |

The skills are written for [Claude Code](https://claude.com/claude-code), which
picks them up from `.claude/skills/` by itself. They are ordinary Markdown, so
another agent can simply be pointed at them. Whichever agent you use, have it
read [`CLAUDE.md`](CLAUDE.md) first: that file is the map of the package,
written for an agent rather than for a human reader.

## Driving it yourself

None of the above is required, and the agent has no private entry point. A run
is described entirely by one YAML file, and two commands do everything:

```bash
kelp info config.yaml      # mesh stats and time range, no tracking
kelp run  config.yaml      # run the tracking
```

A minimal, valid 2D config (complete real ones are in [`examples/`](examples/)):

```yaml
mode:
  Fdrift2D: true                 # 2D surface tracking
numerics:
  integrator: rk2                # euler / rk2 / rk4
  temporal_interp: true          # blend velocity between file snapshots
mesh:
  type: fesom2                   # fesom2 / fesomc / healpix
  meshpath: path/to/fesom.mesh.diag.nc   # the mesh — usually a separate
data:                                      # directory from the velocity data
  ncinput:  path/to/fesom2_output         # file or directory of <var>.fesom.<year>.nc
  inicloud: particles.dat                  # seeds: "lon lat [depth] [release_sec]"
  output_path: trajectories.nc             # .nc or .zarr; empty = keep in RAM
  time_averaging: mean                     # mean (averaged output) / instant
time:
  start_time_idx: 0
  n_steps: 240
```

Every feature is a key in this file; only resuming a run that outlives one job
needs a small Python driver (below).
The complete list of keys, defaults and validation is in the
[Config file reference](#config-file-reference) below and, with prose and
file-discovery rules, in [`docs/configuration.md`](docs/configuration.md). How to
drive a run (time window, temporal interpolation, backward tracking, delayed
release, diffusion, beaching, scalar sampling, output and restart) is in
[`docs/running.md`](docs/running.md).

### Running from Python

The same config runs from Python. Without `data.output_path` the result is the
in-memory trajectories; with it the trajectories stream to disk and the returned
records stay empty:

```python
from kelp.core.config import load_config
from kelp.tracking import run_tracking

config = load_config("config.yaml")
records = run_tracking(config, verbose=True)
arrs = records[0].to_arrays()      # {"lon": (T, P), "lat": (T, P), "status": ..., "time": (T,) s,
                                   #  "release_lon": (P,), ...}
                                   # (load_trajectories() reads streamed output back as (P, T))
```

The output file is a CF trajectory file (output format 2, global attribute
`kelp_output_format = 2`): dimensions `trajectory` and `time`; `time` in seconds
since the first record of the forcing, with that record's date as epoch and the
forcing's calendar, so xarray decodes it to dates (`xr.open_dataset(path)`,
then `ds.sel(time="2005-03-01")`); the release position and release time of
every particle (`release_lon`, `release_lat`, `release_z`, `release_time`); and
velocities that are NaN wherever a particle was not alive through the step.
Files written by earlier versions (record dimension `obs`) stay readable. The
layout is described in [`docs/io_and_time.md`](docs/io_and_time.md) §5.

Read streamed output back with `kelp.writers.load_trajectories(path)` (or
xarray) and resume from a checkpoint with `kelp.writers.load_restart(path)`. For a run
that outlives a single job's wall clock, `kelp.writers.write_restart_file`
drops a standalone, numbered restart per checkpoint — the building block for a
resumable-chain driver (segments + numbered restarts + resubmit + concat,
bit-for-bit identical to one uninterrupted run); see
[`docs/io_and_time.md`](docs/io_and_time.md) §6.

### CLI overrides

The config is the source of truth; a few flags override single keys for one run:
`-o/--output` (`data.output_path`), `--temporal-interp`
(`numerics.temporal_interp`), `--checkpoint-freq` (`numerics.checkpoint_freq`),
`--seed` (`numerics.diffusion.seed`), `--flush-every`
(`numerics.output_flush_every`), `-v/--verbose`.

## Getting it to run

You need the dependencies either way: numpy, scipy, xarray, netCDF4, dask,
numba and PyYAML, plus `zarr` for Zarr output and `matplotlib` with `cartopy`
for the plotting helpers. On a many-core machine, install numba through conda so
it gets the TBB threading backend (`conda install numba tbb -c conda-forge`);
[`docs/performance.md`](docs/performance.md) covers thread counts and batch
system settings.

Installing KELP **itself** is optional, because the package lives under `src/`
and nothing is generated at install time.

- Editable install from a clone, which is what we do. The code keeps living in
  the clone you edit, and you get the `kelp` command on your path:

  ```bash
  pip install -e .                  # core
  pip install -e '.[viz,zarr]'      # plotting, Zarr I/O
  pip install -e '.[healpix]'       # HEALPix meshes, kerchunk reference stores
  pip install -e '.[kerchunk]'      # GRIB-backed kerchunk stores (needs the eccodes library)
  pip install -e '.[dev]'           # tests
  ```

- Or no install at all, pointing Python at the sources:

  ```bash
  PYTHONPATH=src python -m kelp.cli.main run config.yaml
  ```

There is no package on PyPI. `pip install git+https://github.com/CliDyn/kelp`
works without a clone, but it gives you the library alone, without the agent
skills, the documentation and the example configurations, which is most of what
makes the thing usable.

## Documentation

- **[`CLAUDE.md`](CLAUDE.md)** — the package as an agent needs to see it:
  module contracts, data shapes, conventions, what is and is not implemented.
- **[`docs/`](docs/)** — the topic documentation. Start at
  [`docs/architecture.md`](docs/architecture.md) for the data flow and the core
  data structures. The full YAML reference is
  [`docs/configuration.md`](docs/configuration.md); a guide to running is
  [`docs/running.md`](docs/running.md).
- **[`examples/`](examples/)** — complete configurations from real runs, with
  the paths genericised.

## Config file reference

Every option, grouped by section, with its default. Relative paths resolve
against the **config file's** directory (not the working directory); `~` expands.
Unknown/misspelled keys are silently ignored. All five sections must be present
(an empty one as `section: {}`). Full prose, validation and the input-file
discovery rules: [`docs/configuration.md`](docs/configuration.md); a copy-paste
annotated template: [`config.yaml`](config.yaml).

### `mode`

| Key | Default | Meaning |
|-----|---------|---------|
| `Fdrift2D` | `true` | 2D surface tracking |
| `Fdrift3D` | `false` | 3D tracking (also set `Fdrift2D: false`) |
| `time_dir` | `1.0` | `+1.0` forward, `-1.0` backward (anything else is refused) |
| `Vpuls` | `false` | FESOM-C 2D pulsation velocity |
| `var_u`, `var_v`, `var_w`, `var_eta` | `null` (auto) | velocity / SSH variable-name overrides |
| `sigma_layer` | `null` | 2D-in-3D-data: track this layer index (`0` = surface) |
| `sigma_mean` | `false` | FESOM-C: thickness-weighted vertical mean instead |
| `max_levels` | `null` (all) | 3D: keep only the first N vertical interfaces from the surface |
| `w_positive` | `auto` | sign of the vertical velocity in the FILE: `up` / `down` / `auto` (CF attributes, else the reader default: FESOM2, FESOM-C and HEALPix `up`) |
| `sample_vars` | `[]` | node scalars sampled along trajectories (e.g. `[temp, salt]`) |
| `track_failure_reason` | `false` | record WHY each particle died (`failure_reason` output variable) |

### `numerics`

| Key | Default | Meaning |
|-----|---------|---------|
| `dt` | `null` (auto) | integration step (s); auto = file step |
| `file_dt` | `null` (auto) | file time step (s); set only when the file has no usable time axis |
| `integrator` | `euler` | `euler` / `rk2` / `rk4` / `rk45` (adaptive Dormand–Prince; 2D + `leastsq` only) |
| `rk45_tol_m` | `1.0` | `rk45` only: per-substep position tolerance (m) |
| `rk45_h_min` | `null` (`dt·1e-4`) | `rk45` only: minimum substep (s) |
| `output_freq` | `1` | record every N steps |
| `output_flush_every` | `0` | NetCDF only: buffer N records then flush so the file is readable mid-run (`0` = keep open). Zarr is always lock-free |
| `temporal_interp` | `false` | linearly blend velocity between file snapshots (then `dt` may be < file step) |
| `temporal_reconstruction` | `linear` | `linear` / `pchip` (4-snapshot monotone cubic; lets RK4 beat RK2 at coarse cadence) |
| `uv_interp` | `leastsq` | u/v interpolation: `leastsq` (element-layout only) / `barycentric` (node-layout, e.g. FESOM2 `unod`/`vnod` or HEALPix) / `edge_leastsq` (FESOM-style: containing element + edge neighbors; element-layout only) |
| `edge_leastsq_bc` | `freeslip` | `edge_leastsq` coastal ghosts: `freeslip` / `noslip` (inert for other methods) |
| `elem_search` | `local` | containing-element search: `local` (per-step scan) / `walk` (search once at release, then cross edges; identical results; faster on unfused runs — element-layout `barycentric`, `edge_leastsq` — but it switches off the fused node-layout barycentric kernels) |
| `w_interp` | `leastsq` | 3D W: `nearest` / `leastsq` / `barycentric` / `idw` |
| `sample_interp` | `leastsq` | `sample_vars`: `nearest` / `leastsq` / `barycentric` / `idw` |
| `velocity_float32` | `false` | keep u/v/w in float32 to halve snapshot memory / I/O |
| `prefetch_size` | `1` | bulk-read N velocity snapshots per disk read (≥ 4 recommended with `pchip`) |
| `shared_fields_dir` | `null` | node-shared snapshot cache for sharded runs (one read per node; see docs/configuration.md) |
| `shared_fields_generation` | `null` | namespace generation of that cache (default: the batch job) |
| `Dmin` | `0.5` | minimum water-column height (m); sigma meshes only |
| `z0b_min` | `0.1` | bottom clamp distance (m); sigma meshes only |
| `Dch` | `1.0` | nearest-node search radius (deg); reserved, currently unused |
| `checkpoint_freq` | `0` | write a restart checkpoint every N steps (needs `output_path`) |
| `periodic_lon` | `null` | idealized channels: `[lo, hi]` post-step fold window (mesh needs a ghost band past it; strictly inside ±180; see docs/configuration.md) |
| `periodic_lat` | `null` | same for latitude (doubly-periodic boxes; strictly inside ±90) |
| `diffusion` | (off) | stochastic diffusion block, below |
| `beaching` | (off) | beaching block, below |

`numerics.diffusion` (`enabled: false` → byte-identical to no diffusion):

| Key | Default | Meaning |
|-----|---------|---------|
| `enabled` | `false` | turn diffusion on |
| `scheme` | `uniform` | `uniform` (scalar Kh) / `em` / `m1` (spatial Kh from the dataset) |
| `kh_zonal`, `kh_meridional` | `0.0` | m²/s, `uniform` only |
| `var_kh_zonal`, `var_kh_meridional` | `null` | dataset Kh variable names (`em`/`m1`) |
| `dres_m` | `null` (mesh-derived) | central-difference offset (m) for Kh gradients |
| `seed` | `0` | counter-RNG seed (reproducible, thread- and restart-independent) |
| `composition` | `split` | `split` / `combined` / `split_rk` (auto-resolved; `split_rk` required for `em`/`m1` + RK2/RK4) |
| `kh_eval_position` | `end` | `split_rk` only: sample Kh at `start` or `end` of the step |
| `az_enabled` | `false` | vertical Az diffusion (3D only): Itô-drift kick with reflective surface/bottom boundaries |
| `az_const` | `0.0` | constant vertical diffusivity (m²/s), used when `var_az` unset |
| `var_az` | `null` | dataset Az variable, e.g. FESOM2 `Av` (FESOM2 z-level runs only) |

`numerics.beaching` (`enabled: false` → byte-identical to no beaching):

| Key | Default | Meaning |
|-----|---------|---------|
| `enabled` | `false` | turn beaching on |
| `criteria` | `[depth]` | subset of `depth` / `coastal_element` / `on_failure` / `stochastic` (OR-combined) |
| `depth_min` | `0.3` | m; beach when raw column height < `depth_min` |
| `use_eta` | `false` | add per-timestep SSH to depth (needs an ssh/eta/zos variable) |
| `dry_mask_var` | `null` | node variable; `> 0.5` = dry |
| `zone_depth` | `null` (`2·depth_min`) | stochastic-zone height (m) |
| `t_beach` | `86400.0` | s; stochastic beaching timescale |
| `resuspension` | `never` | `never` / `auto` / `stochastic` / `auto_or_stochastic` |
| `rewet_hysteresis` | `0.1` | m above `depth_min` for auto release |
| `min_wet_steps` | `1` | wet steps required before resuspension |
| `t_resuspend` | `null` | s; stochastic resuspension timescale |
| `seed` | `null` (`diffusion.seed`) | counter-RNG seed for beaching draws |

### `mesh`

| Key | Default | Meaning |
|-----|---------|---------|
| `type` | `fesomc` | **required**: `fesomc` / `fesom2` / `nemo` / `healpix` |
| `meshpath` | `""` | **required**: mesh-diag `.nc` / `.zarr` / ASCII dir (FESOM2); single NetCDF (FESOM-C); `mesh_mask.nc`/`domain_cfg.nc` (NEMO); the u/v data store itself (HEALPix — the mesh is derived from the data) |
| `r_earth` | `null` (family default) | Earth radius (m); set only to match a non-default model build |
| `rotation` | `null` | FESOM2 + HEALPix: pole-safe rotated frame — `null` / `fesom2` / `[α, β, γ]` (deg) |
| `grid_hash_cell_size` | `null` (rule) | nearest-node hash cell (deg); `null` = chosen by `grid_hash_cell_rule` |
| `grid_hash_cell_rule` | `occupancy` | `occupancy` (measured choice) / `legacy` (the reader's mean-spacing cell) |
| `exact_nearest` | `true` | exact nearest-node anchors for any cell |
| `healpix_geometry` | `dual` | HEALPix only: `dual` (nodes at pixel centres) / `pixel` (pixels as elements; 2D) |
| `healpix_zoom` | `null` | HEALPix only: zoom level (nside = 2^zoom), fallback when the store has no `crs` attributes |
| `crop_bbox` | `null` | HEALPix only: `[W, E, S, N]` (deg) regional pixel crop; `W > E` = dateline window |
| `levels` | `null` | HEALPix: vertical interfaces — an aux3d file path or an inline list of metres (required in 3D for raw GRIB stores; in 2D it sets the node depths when the store has no `deptho`) |

### `data`

| Key | Default | Meaning |
|-----|---------|---------|
| `ncinput` | `""` | **required**: velocity file or directory (u-file if split) |
| `ncinput_v` | `""` | separate v file |
| `ncinput_w` | `""` | separate w file (3D) |
| `ncinput_eta` | `""` | separate SSH store (HEALPix two-store recipes) |
| `inicloud` | `""` | **required**: seeds, one `lon lat [depth] [release_sec]` per line |
| `particle_offset` | `null` | global index of the file's first particle in a sharded release (default: the shard file's header, else 0) |
| `output_path` | `""` | `.nc` → NetCDF, `.zarr` → Zarr; empty = in-memory only |
| `time_averaging` | `null` (warns) | `mean` (time-averaged output; timestamps shifted to window midpoints) / `instant` |

### `time`

| Key | Default | Meaning |
|-----|---------|---------|
| `start_time_idx` | `0` | first file time index |
| `n_steps` | `null` | number of tracking steps (**preferred** way to set the window) |
| `end_time_idx` | `null` | legacy alternative → `n_steps = end - start` |
| `start_date` | `null` | ISO datetime, overrides `start_time_idx` (between snapshots needs `temporal_interp`) |
| `duration_days` | `null` | with `start_date` → `n_steps = days·86400/dt` |
| `on_exhausted` | `stop` | `stop` at the file boundary / `cycle` (periodic forcing) |
| `num_runs` | `1` | realizations; when `> 1`, output files get `_r0`, `_r1`, … (a single run keeps the plain name) |
| `run_interval` | `0` | file-index shift between realizations |
| `n_pulses` | `1` | delayed release: re-release the seed set N times (same coords) |
| `pulse_interval` | `0.0` | seconds between pulses |

## Performance

The per-step hot loops are Numba-parallel (`prange`) with fused kernels, and on
large meshes velocity fields are blended lazily, which keeps
temporal-interpolation runs fast. For the cost model, thread
scaling, sizing recipes and SLURM operation see
[`docs/performance.md`](docs/performance.md); for how performance and accuracy
are measured see [`docs/validation_and_performance.md`](docs/validation_and_performance.md).

> **Large meshes, small regions.** KELP runs directly on a full global mesh,
> so you do not have to subset the data. If you only need a region, cut it out
> first with [`utils/extract_fesom_region.py`](utils/README.md) (writes a Zarr
> subset the FESOM2 reader loads directly), which makes the mesh load and per-step
> I/O much lighter. Two `numerics` knobs cut the dominant I/O cost (both default
> off → byte-identical): `velocity_float32` and `prefetch_size`.

Preprocessing tools live in [`utils/`](utils/README.md): regional subsetting
(`extract_fesom_region.py`) and offline vertical-velocity reconstruction
(`reconstruct_w.py`). Plotting and animation helpers are in
[`src/kelp/visualization/`](src/kelp/visualization/README.md).

## Tests

```bash
pip install -e '.[dev]'
python -m pytest tests/        # synthetic data and small committed fixtures; the slow suites (Parcels
                               # comparisons, data-gated HEALPix twin / W sign) are deselected
```

See [`docs/validation_and_performance.md`](docs/validation_and_performance.md) for
the validation campaign and the `-m slow` cross-checks against Parcels.

## Features

- **2D and 3D tracking**, forward or backward, on FESOM2 meshes (fixed depth
  levels), FESOM-C meshes (terrain-following sigma layers) and HEALPix stores
  (next bullet).
- **HEALPix reader** — tracks directly on HEALPix-gridded output (e.g. nextGEMS
  dissemination stores; Zarr, NetCDF or kerchunk references), 2D and 3D: the
  mesh is built from the data store itself as the exact corner dual of the
  pixel grid (node-layout velocities → `uv_interp: barycentric`), with regional
  pixel extraction (`utils/extract_healpix_region.py`) that reproduces the
  full-store trajectories bit-for-bit inside the extracted sector. Because the
  same simulation can be tracked on its native mesh and on HEALPix with
  identical schemes and releases, the difference between the two isolates the
  effect of the remapping: for a uniform global release the 30-day median
  separation was 4.4, 1.4 and 0.39 km at pixel spacings of 102, 51 and
  25.5 km.
- **Time integrators** Euler / RK2 / RK4, with time-aware stages and an optional
  4-snapshot PCHIP temporal reconstruction; plus an adaptive RK45
  (Dormand–Prince, per-particle substepping under a metre error tolerance —
  2D runs).
- **Interpolation** chosen per run: least-squares plane fit, barycentric /
  Wachspress (product form — exact on element edges and vertices), or the
  FESOM-style edge-neighbor least squares (`edge_leastsq`, with freeslip /
  noslip coastal boundary ghosts) for u/v; four schemes (nearest,
  least-squares, barycentric, inverse-distance) for W and sampled scalars.
- **Incremental element tracking** (`numerics.elem_search: walk`) — the
  containing element is searched once at release and then tracked by crossing
  edges (FESOM-style), with identical results to the per-step search and
  ~3.6× faster containment on the paths that need it.
- **Periodic boundary windows** (`numerics.periodic_lon` / `periodic_lat`) for
  idealized channel and doubly-periodic domains: particles fold back into the
  window after each step (the mesh carries a ghost band past the window; off =
  byte-identical).
- **Stochastic diffusion** — horizontal: constant or spatially-varying Kh via
  Euler–Maruyama / Milstein, with the Itô drift correction; vertical (3D): an
  Az random walk with the Visser drift and reflective surface/bottom
  boundaries, from a constant or a model field (FESOM2 `Av`); counter-based
  RNG, so results are reproducible regardless of thread count.
- **Beaching and resuspension** by several combinable criteria.
- **Pole-safe tracking** — an optional rotated frame moves the lon/lat
  singularity off the ocean for Arctic / transpolar runs; output is geographic.
- **Scalar sampling along trajectories** — temperature, salinity, or any
  node-based field, recorded at each particle.
- **Delayed / staggered release** — pulsed or continuous sources; not-yet-released
  particles are frozen (written as gaps) until their time.
- **Streaming output** to NetCDF or Zarr with restart checkpoints; both readable
  mid-run (Zarr is lock-free; NetCDF via `numerics.output_flush_every`). The
  files follow the CF trajectory conventions: calendar time, release positions
  and times, and the same attributes in NetCDF and Zarr.
  Standalone numbered restart files (`write_restart_file`) support resumable-chain
  runs that span a wall-clock limit, bit-for-bit identical to one continuous run.

There is also a **structured-grid foundation** (`readers/structured.py`,
`readers/nemo.py`) that maps a logically rectangular C-grid onto a quad mesh.
It is early work: two-dimensional only, restricted to grids whose axes line up
with east and north, and not part of what we run day to day. It refuses rather
than guesses whenever those limits are exceeded. See
[`CLAUDE.md`](CLAUDE.md) for its exact scope before relying on it.

## Licence and citation

KELP is released under the BSD 3-Clause licence (`LICENSE`).

If KELP contributes to work you publish, please cite it. `CITATION.cff` carries
the machine-readable record that GitHub's "Cite this repository" button reads,
and `CHANGELOG.md` records what changed in each release.
