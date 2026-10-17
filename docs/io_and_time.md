# KELP input/output and time control

> **Covers:** how the tracker finds and opens its input data, how it interprets the time axis of that data (the time step, time-averaging, a calendar start date, and cycling), the small cache that avoids re-reading the same data, and how trajectories and restart checkpoints are written and read back — including the authoritative description of the trajectory output format (format 2, §5).
> **Does not cover:** the configuration keys that set the data paths and the time window — see `configuration.md` §5 and §6; how velocity is *blended* between two snapshots in time — see `time_integration.md`; how scalar fields are sampled along trajectories — see `field_sampling.md`.
> **Code:** `src/kelp/readers/__init__.py:get_reader`, `src/kelp/readers/__init__.py:extract_time_info`, `src/kelp/writers.py:TrajectoryWriter` (authoritative implementation).
> **Verified at:** see `docs/doc_map.yaml` (stamped via check_doc_sync --bless).

This document explains the machinery that surrounds the actual particle-stepping
loop: the part that turns files on disk into velocity fields in memory, decides
what time each of those fields represents, and writes the resulting particle
paths back out to disk. It is the plumbing of a run. The companion overview is
`architecture.md`, which shows where these pieces sit relative to the stepping
loop; the time *blending* between two snapshots (how a velocity halfway between
two file records is computed) lives in `time_integration.md`.

A few terms used throughout. A **snapshot** (or **record**) is the velocity
field stored for one instant in the input file — one slice along the file's time
axis. A **reader** is a small module that knows how one ocean model writes its
mesh and velocity, and translates that model's file conventions into the
tracker's internal data structures. A **trajectory** is the recorded path of one
particle: its position written out once per output step.

## §1 Readers and the model registry

Different ocean models store their meshes and velocities differently, so the
tracker keeps a small **registry** of model-specific readers. `get_reader(name)`
looks a reader up by model name — currently `"fesom2"`, `"fesomc"`, `"nemo"`,
and `"healpix"` — and
auto-imports the matching module the first time it is asked for. An unknown name
raises a clear error listing the readers that are available.

Every reader exposes the same four-method interface, so the tracking loop never
needs to know which model it is reading:

- `load_mesh(mesh_cfg)` builds the mesh (node coordinates, element
  connectivity, depths, vertical grid) from the mesh source named in the config.
- `open_dataset(path)` opens the velocity data and returns it as an xarray
  dataset (see §2).
- `read_velocities(ds, timestep, mesh, mode)` reads one snapshot and returns the
  velocity field for that instant.
- `read_scalar_fields(ds, timestep, mesh, varnames, mode_cfg)` (optional) reads
  named node-based scalar fields for trajectory sampling — see `field_sampling.md`.

Whether a given path is a NetCDF file or a Zarr store is decided by
`_detect_engine`: a `.zarr` suffix, or a directory that contains a `.zattrs`
(Zarr v2) or `zarr.json` (Zarr v3) marker file, is treated as Zarr; everything
else is treated as NetCDF. (NetCDF
and Zarr are two on-disk formats for the same kind of gridded array data; the
tracker reads both transparently.)

## §2 Finding the data files

`open_dataset` (the FESOM2 reader's version is the richest) accepts the input
data in three shapes:

- **A single file or Zarr store** that already holds the velocity variables
  (`u`, `v`, and optionally `ssh` and `w`) together. Common alias names for the
  velocity variables are renamed to the standard `u`/`v` on the way in.
- **Separate velocity files** named explicitly — a `v`-file and/or `w`-file
  passed alongside the `u`-file — which are merged into one dataset. Each is
  matched to the `u`-file record by record, by instant
  (`readers._align_onto_time_axis`, 2026-10-04): extra records outside the
  `u`-file's span are dropped; with dates a finer cadence is sub-sampled onto
  the `u` records with a warning (numeric axes without an epoch refuse it); the
  `v`-file must cover every `u` record; a `w`-file may cover a block of them (a
  reconstructed `time_range`; a `u` record missing inside its own range is
  another cadence and refused), and an uncovered record is refused when a run
  reads it — look-ahead reads of the read-ahead window are not. In a 2-D run a
  `w`-file that does not fit is skipped with a warning. Without this, xarray
  attaches a variable by its time labels and pads every unmatched record with
  NaN.
- **A directory of split per-variable files**, which is how raw model output and
  the region-extraction utility both store their data: one variable per file.
  The reader takes the first alphabetically-sorted match of `u*` and of `v*`,
  and optionally the first `w*` and `ssh*`. It then merges in *any other*
  per-variable file in the directory (for example `temp*`, `salt*`) so that
  trajectory sampling can reach those fields. Two guards keep this safe: files
  with `mesh` in the name (mesh-diagnostic stores) are skipped, and variables
  whose dimensions do not match the known model grid are skipped silently rather
  than risking a dimension-alignment error. The `v*`, `w*` and `ssh*` files are
  matched to the `u*` file's records the same way; a `w*` file that does not
  fit is refused in 3-D and skipped with a warning in 2-D, and an `ssh*` or
  extra file that does not fit is skipped with a warning, since merging it
  would stretch the time axis with NaN records.

Because the directory case picks files by a **prefix glob** (`u*`, `v*`, …) and
takes the *first sorted match*, it is possible to point it at a directory where
the first `u*` file is not the one you meant — the classic example being a
`uice*` (sea-ice velocity) file sorting ahead of the ocean `u*` file. The full
discovery rules, the exact globs, and the pitfalls to watch for are written out
in `configuration.md` §5.1; this section is only the summary. When a run picks
up an unexpected file or variable, that section is the place to look.

### Multi-year directories and mesh-diagnostic conventions (FESOM2)

Two newer capabilities extend the directory case:

- **Multi-year reading.** Standard FESOM2 output stores one file *per variable
  per year* (`<var>.fesom.<year>.nc`, e.g. 1958–2020). When `open_dataset`
  receives the run's `mode_cfg` and EITHER configured velocity variable's
  **exact** glob `<name>.fesom.<YYYY>.nc` matches at least one yearly file
  (single-year directories included), the reader assembles the
  variable family named by the config (`var_u`/`var_v`/`var_w`/`var_eta` and any
  `sample_vars`) and concatenates every year **lazily** with `xr.open_mfdataset`
  + `xr.merge`. The exact-token glob (not a `u*` prefix) means `u` no longer
  accidentally matches `uice`/`unod`/`uwind`. Each file's own
  `seconds since <year>-01-01` epoch decodes independently and the records order
  into one monotonic absolute axis spanning all years (leap years included);
  only the per-step slice is ever read into memory. If the selected variables
  disagree on cadence (for example monthly element `u` at 12 records/year versus
  daily node `unod` at 365/year) the merge fails fast with a clear error rather
  than silently misaligning; non-consecutive year tokens raise a `RuntimeWarning`
  (not an error). A non-exact directory (e.g. `u.nc`/`v.nc`) or a single combined
  file keeps the original prefix-glob behaviour. This path needs `dask` installed.
  *Limitation:* multi-year mode assembles only the velocity/eta/`sample_vars`
  family plus `diffusion.var_az` (fesom2, active Az kick) —
  `beaching.dry_mask_var` and spatial-diffusion Kh fields are not yet
  globbed across years (use single-year or pre-concatenated input for those).
- **Mesh-diagnostic conventions.** The NetCDF mesh loader is convention-agnostic:
  it reads coordinates, element connectivity, and vertical levels by *variable
  name* across the known FESOM2 diag layouts — `lon`/`lat` + `elements` + `nz`;
  the older core2 form with packed `nodes` (in radians) + `elem` + `zbar`
  (negative-down); and the UGRID form with `face_nodes`. Connectivity is oriented
  to a single `(elements, 3)` shape and made 0-based, and depths are taken
  positive-down regardless of the file's sign convention.

## §3 The time axis

Reading the data is only half the job; the tracker also has to know *what time*
each snapshot represents, so it can advance particles by the right amount and
stamp the output with real timestamps.

### Interpreting the time coordinate

`extract_time_info(ds)` turns the dataset's time coordinate into two things: an
array of times in **seconds since the first record**, and a single regular
spacing `file_dt` (the seconds between consecutive records). It understands the
common ways a model stores time — calendar dates (`datetime64`), climate-model
calendars (`cftime` objects), and plain numbers carrying a `units` attribute:
a CF epoch (`hours since 2001-01-01`, with its `calendar`) goes through
xarray's CF decoder, a bare unit (`h`, `hours`, `d`, `min`, `s`, `ms`, `us`, …)
is scaled by an exact token table, and an unknown unit is not guessed. A
numeric axis with **no** `units` attribute at all is read as seconds (except
the bare integer index below) — set `numerics.file_dt` or add units when it is
not.

Two guards keep it honest. A coordinate that is just an auto-generated integer
index (0, 1, 2, …) **without time units** is rejected as *not real time* — it
carries no physical duration, so the tracker refuses to invent one (an integer
axis whose `units` say `hours since …` is a real axis). And `file_dt` is only
returned when the spacing between records is regular to within a small tolerance
(about 0.1%); irregular spacing yields `file_dt = None`, signalling that no single
file step exists. `cftime` axes are differenced in their own calendar (noleap,
360_day, …), and `datetime64` keeps sub-second steps.

When `temporal_interp` is requested and no physical axis can be decoded (and no
`numerics.file_dt` declares the interval), the run stops before the output is
created. Until 2026-09-23 it fell back silently to one snapshot per step — with an
explicit `dt` that changed the physics without a word.

### The simulation step versus the file step

There are two distinct time steps, and keeping them apart matters:

- **`file_dt`** is the spacing of the input data — how far apart the snapshots
  on disk are.
- **`dt`** is the simulation step — how far the integrator advances particles in
  one move. When `dt` is left unset in the config it is auto-detected from
  `file_dt`.

In **discrete mode** the two are the same: one file snapshot per simulation step.
In **temporal-interpolation mode** `dt` may be *smaller* than `file_dt`, and the
tracker blends adjacent snapshots to produce a velocity at each sub-step time.
That blending — and the accuracy caveat that comes with linear interpolation
between snapshots — is the subject of `time_integration.md`. In discrete mode a
`dt` unlike the file step, or irregular record times, are refused at startup
(`tracking._check_discrete_dt`);
in temporal-interpolation mode no check makes `dt` divide the file step, and a
step that straddles a snapshot runs on a steady velocity (one warning per
realization).

### Time-averaged output and the midpoint shift

Some models write **time-averaged** output: each record is the mean over an
averaging window (a day, a month), and the file stamps that record with the time
at the *end* of the window. But the velocity that average best represents is the
one at the *middle* of the window. So when `data.time_averaging` is set to
`"mean"`, `_shift_to_midpoints` moves every timestamp back to the midpoint of its
window — record `i` is re-timed to halfway between records `i-1` and `i`, and the
very first record is extrapolated back by half the first interval. Setting
`time_averaging` to `"instant"` (snapshots that already represent a single
instant) skips the shift. If the key is left unset, the tracker warns and guesses
a sensible default for the model family. The config keys are in
`configuration.md` §6.

The shift moves the time *values*, not the origin they are measured from: the
times stay seconds since the first **raw** record, so the first window centre is
a negative number (half an interval before that record). The trajectory output
uses the same origin as its epoch (§5.2), so its dates are the window centres.

### Starting from a calendar date

Instead of a numeric index into the file, a run can name an ISO **`start_date`**
(for example `2020-01-15T00:00`). `_resolve_start_date` matches that date against
the **raw** file timestamps — the original on-disk times, *not* the
midpoint-shifted ones — and returns the corresponding file index. If the
requested date falls exactly on a snapshot, the run starts there. If it falls
*between* two snapshots, that only works with temporal interpolation enabled
(so the tracker can construct a velocity at the in-between instant); otherwise it
raises with an explanatory message. With temporal interpolation the run clock
starts at the requested instant itself, exact stamp or not; under
`time_averaging: mean` the velocity at that instant is interpolated between the
window centres, and the last stamp — past the last window centre, with nothing
to interpolate towards — keeps the record start (with a warning). In discrete mode an exact stamp starts its
record, whose time under `mean` is the window centre. Until 2026-10-04 an exact
stamp started at the window centre in both modes — half a record before the
requested date — while a date one second later started at the date. A date outside the file's time range is also
rejected. The date is built in the axis's own calendar — the ISO components are
read without Gregorian validation, so `2001-02-30` is a valid start on a
360-day axis — and an undecoded numeric CF axis is decoded with its `units` and
`calendar` first. The output's first record is labelled one step after the
start (§5.2): with temporal interpolation at `start_date + dt · time_dir`, in
discrete mode at the start record's time `+ dt · time_dir`.

### Running off the end of the data: stop or cycle

When the simulation wants a snapshot beyond the last one in the file,
`on_exhausted` decides what happens. `"stop"` (the default) ends the run at the
file boundary. `"cycle"` wraps around to the beginning, mapping the requested
index back into range with a modulo against the total number of records
(`% n_time_total`) — useful for driving a long integration from a short,
repeating sequence of snapshots (an idealised or climatological forcing). The
time written to the trajectory output does not wrap: it is the run's own clock,
so past the first forcing period its dates lie beyond the end of the forcing
(§5.2). A run
can also be repeated as several independent **realizations** (`num_runs`), each
offset in time by `run_interval`; these are mainly relevant to stochastic
diffusion and are described in `configuration.md` §6.

## §4 The velocity cache

Reading a snapshot off disk is expensive, especially on a large mesh. The same
snapshot is often needed on several consecutive steps — both endpoints of a time
bracket are reused as the window slides, and a *cycling* run over a handful of
records re-reads the same few snapshots over and over. `_VelCache` avoids that
waste: it holds velocity fields keyed by file index, and `read_velocities` is
only called for an index that is not already in hand. After each step the cache
evicts everything except the indices still needed: the two snapshots a step
can touch, the four of a PCHIP stencil, and — with `prefetch_size > 1` — the
read-ahead window not yet consumed. This is purely a performance shortcut —
it changes nothing about the result, only how many times the disk is read.

Every field the cache stores — a single read, each record of a prefetch window,
a snapshot of the shared cache — first passes `mode.uv_vertical`
(`tracking._mark_depth_linear`): with the default `layer` it is stored as read;
with `linear` it is stored as a shallow copy marked depth-linear and checked
(u/v finite down to each column's last usable point, `vertical.md` §2.1), so a
bad record is refused when it is first read, named by its global time index.

## §5 Writing trajectories (output format 2)

`TrajectoryWriter` is a **streaming** writer: it writes the output as the run
goes rather than holding the whole run in memory, so the memory footprint stays
flat regardless of how long the run is. NetCDF appends each record (or, with
`output_flush_every: N`, each batch of N); Zarr buffers records in memory and
writes one whole chunk of records (up to 256) at a time, so a reader sees — and
a killed run keeps — only the flushed chunks; checkpoints and `close()` flush the
buffer. The on-disk format is chosen by the
output file's extension — `.nc` writes compressed NetCDF4 with an unlimited
record dimension, `.zarr` writes a chunked Zarr store. A Zarr
store chunks each `(trajectory, time)` variable `(traj_chunk, obs_chunk)`:
runs up to 500 000 particles keep the whole trajectory axis in one block
(the historical layout), larger runs split it into 500 000-particle blocks
so that a chunk stays near ~4 MB and reading a COMPACT subset of
particles touches only its blocks instead of full columns (scattered
subsets may still touch all blocks — TODO 5.4); the record width of a chunk
adapts to the block size. Both back ends share
one API: `open()`, `append(state, time_value, sampled)` once per output step,
`write_checkpoint(...)` for restarts (§6), and `close()`; it also works as a
context manager. Three setters complete the schema; `run_single_realization`
calls them once at its start: `set_time_units(...)` (the epoch and calendar of
`time`, §5.2), `write_release(...)` (the release variables, §5.3) and
`set_trajectory_ids(...)` (the particle identifiers of a shard).

**Output format 2.** Every file this version writes carries the global
attribute `kelp_output_format = 2`. A file without that attribute is a
**format-1** file, written by an engine before release 1.0; §5.9 says how the two
differ and how old files are read. Format 2 follows the CF conventions for
trajectories (CF = Climate and Forecast metadata conventions; a "trajectory" is
one of its discrete sampling geometries), so xarray, ncview and other CF tools
read the time as dates and know which variables are positions.

### 5.1 Layout

The file has two dimensions. **`trajectory`** has one entry per particle and a
fixed length. **`time`** has one entry per record and grows as the run writes
(unlimited in NetCDF, resized in Zarr). Every per-record variable has the
dimensions `(trajectory, time)`, so one row is the track of one particle.
`trajectory(trajectory)` and `time(time)` are **coordinate variables** — a
variable with the same name as its dimension, which tools use to label that
axis.

| variable | dimensions | type | present | units and main attributes |
|---|---|---|---|---|
| `trajectory` | `(trajectory)` | int32 | always | `cf_role = "trajectory_id"`, `long_name = "particle ID"`; values `particle_offset + 0 … P−1` (the global index of the particle in the whole release, also in a shard); int32, so a release holds at most 2³¹−1 particles (the writer and the shard merge refuse larger IDs) |
| `time` | `(time)` | float64 | always | seconds; the time at the end of the recorded step; epoch, calendar and the no-epoch case in §5.2 |
| `lon` | `(trajectory, time)` | float64 | always | `standard_name = "longitude"`, `units = "degrees_east"`; fill NaN |
| `lat` | `(trajectory, time)` | float64 | always | `standard_name = "latitude"`, `units = "degrees_north"`; fill NaN |
| `z` | `(trajectory, time)` | float64 | always | `standard_name = "depth"`, `units = "m"`, `positive = "down"`; 0 in 2-D; fill NaN |
| `status` | `(trajectory, time)` | int32 | always | `flag_values = 1 2 3 4`, `flag_meanings = "alive beached outside not_released"`, `coordinates = "lat lon z"` |
| `failure_reason` | `(trajectory, time)` | int8 | `mode.track_failure_reason: true` | `flag_values = 0 … 4` (none, interpolation_failed, out_of_bounds, surface_hit, bottom_hit), `coordinates = "lat lon z"`; fill 0 (none) |
| `vel_u`, `vel_v` | `(trajectory, time)` | float64 | when a record carries velocities (from the first record of a tracking run) | `units = "m s-1"`, a `comment` on their meaning (§5.4), no `coordinates`; fill NaN |
| `vel_w` | `(trajectory, time)` | float64 | 3-D runs only | as `vel_u`; positive down |
| sampled fields (`temp`, …) | `(trajectory, time)` | float64 | `mode.sample_vars` | the source variable's `units` and `long_name`, `coordinates = "lat lon z"`, a `comment` on the sampling instant (§5.5); fill NaN |
| `release_lon` | `(trajectory)` | float64 | written by `run_single_realization` | `units = "degrees_east"`; fill NaN |
| `release_lat` | `(trajectory)` | float64 | as above | `units = "degrees_north"`; fill NaN |
| `release_z` | `(trajectory)` | float64 | as above | `units = "m"`, depth positive down; fill NaN |
| `release_time` | `(trajectory)` | float64 | as above | the `units` and `calendar` of `time`; never NaN |
| `restart_*` | `(trajectory)` or one value | — | only after the first checkpoint | §6 |

The release variables have a `long_name` and `units` but no `standard_name`.
The global attributes are
`Conventions = "CF-1.8"`, `featureType = "trajectory"`, `kelp_output_format = 2`
(an int32 in NetCDF, a JSON integer in Zarr), `history`, and the provenance
attributes of §5.8. Both back ends write the same attributes.

**Fill values.** A **fill value** is the number a variable holds where nothing
was written. It is NaN for `lon/lat/z`, the velocities, the sampled fields and
the release variables, and 0 (none) for `failure_reason`. The positions of a
particle that is not yet released are written as NaN (its `status` stays 4,
not_released). The writer creates `failure_reason` and the sampled fields
**lazily**, the first time a record carries them, so a run without the feature
has no empty variable. The velocities are created when the run starts
(`vel_u`, `vel_v`; `vel_w` in a 3-D run), so every file of a run — each chain
segment, each shard — has the same variables, even a segment resumed when
every particle is already dead, which records no velocity at all (a direct
`TrajectoryWriter` user without `declare_velocity` still gets them at the first
record that carries a velocity). A lazily created variable reads its
fill value in the records before it existed, and a later record that does not
carry it writes the fill value too. (In format 1 a NetCDF velocity variable
that missed a record read 9.97e36 there — the NetCDF default fill.)

### 5.2 Time

**Values.** `time` holds seconds in both time modes. The origin `T0` is the
first **raw** record of the forcing time axis (`ds["time"][0]`), the origin
`extract_time_info` measures from (§3). Under `time_averaging: mean` the run
reads midpoint-shifted times; the labels follow them, but the origin does not
move, so the dates are the window centres.

**The forcing's own clock.** KELP keeps the clock of the forcing and converts
no time zones: the output dates repeat the forcing's timestamps. CF time and
the model files carry no zone; if a hand-built dataset gives timestamps with a
time zone (Python datetimes with `tzinfo`), the zone is dropped and the epoch
repeats their wall-clock value (`03:00+03:00` → `seconds since … 03:00:00`).
Convert such timestamps yourself if the output must be in UTC.

**One formula for both modes.** The record written after global step `g` (the
step counter that starts at 0 with the run and keeps counting across restarts)
holds

```
time = T_g0 + (g + 1) · dt · direction
```

where `direction` is `time_dir` (+1 forward, −1 backward) and `T_g0` is the
time at which global step 0 started — for a resumed call (`start_step_i > 0`),
the start of the whole run, not of this segment:

- **interpolated mode** (`temporal_interp: true`):
  `T_g0 = t_first − start_step_i · dt · direction`, where `t_first`, the time
  at which this call's first step starts, is `sim_time_start` when it is given
  (a `start_date`, or a chain driver), otherwise `file_times[start_time_idx]`;
- **discrete mode**: the time of the record where global step 0 started,
  `T_g0 = file_times[start_time_idx − start_step_i · direction]` (a single
  cycled record repeats with period `dt`: `file_times[0] + idx · dt`), on the axis
  the run reads — `run_tracking` passes the midpoint-shifted axis under
  `mean`; a direct caller who passes no `file_times` gets the dataset's own,
  unshifted axis. Anchoring on the run's first record rather than this
  segment's keeps the labels independent of the chain cuts even when the
  record spacing differs slightly from `dt` (§3 allows a small tolerance).
  `sim_time_start` is ignored in discrete mode, because the reads follow
  `start_time_idx` and the label must too. With `on_exhausted: cycle` an index
  beyond the last record counts whole forcing periods. Without a usable axis
  `T_g0 = (start_time_idx − start_step_i · direction) · dt`.

So a label is the time at the **end** of the recorded step, and backward runs
count down. Example: hourly snapshots (`time_averaging: instant`) whose first
record is 2005-01-01 00:00; a forward run from `start_time_idx = 2` with
`dt = 3600` s has `T_g0` = 7 200 s and writes its first record at
`T_g0 + dt` = 10 800 s, which reads as 2005-01-01 03:00.

The labels do not depend on how a run is cut into chain segments, because they
are computed from the global step and the start of the whole run (up to the
rounding of a `t_first` that a driver computes; the resume recipe is in §6).
The internal clock of the interpolated mode, which picks the snapshots to
blend, is unchanged, so positions are bit-identical to format 1; only the label
uses the formula.

In discrete mode the label is the particle clock: it advances by `dt` per step,
not by the stamp of the record that was read. `_check_discrete_dt` admits a
record spacing up to `max(0.1 s, 1e-3·file_dt)` away from `dt`, so the label can
drift from the record stamps by that much per step. The in-memory result of a
run without an output file (`TrajectoryRecord.times`) carries the same seconds.

**Attributes of `time`.**

- **Forcing with a calendar epoch** (dates, cftime objects, or numbers with
  `units = "<unit> since <date>"`): `standard_name = "time"`, `axis = "T"`,
  `long_name = "time at the end of the recorded step"`,
  `units = "seconds since YYYY-MM-DD hh:mm:ss[.f]"` (the first raw record; the
  year has four digits; a fraction of a second is written only when it is not
  zero), and `calendar` = the forcing's calendar: the one xarray kept from the
  forcing file, the calendar of cftime objects (for example `noleap`,
  `360_day`), `proleptic_gregorian` for plain `datetime64` dates, `standard`
  for Python datetimes, or the `calendar` attribute of a numeric axis
  (default `standard`). The helper is `readers.time_axis_epoch(ds)`.
- **Forcing without a calendar epoch** (a bare unit such as `hours`, an index
  axis with `numerics.file_dt`, or no usable axis): `units = "seconds"`,
  `long_name`, and `time_origin = "first record of the forcing time axis (the
  forcing has no calendar epoch)"`. There is no `standard_name` and no `axis`,
  because CF requires a reference date for those. The values are seconds from
  the first record all the same.
- **`on_exhausted: cycle`**: `time` is the run's own clock and does not wrap, so
  after the first forcing period the dates lie beyond the end of the forcing.
  The attribute `forcing_cycle_period` gives the period in seconds (the span of
  the forcing records the run uses plus one record interval; `n_time_total · dt` without an
  axis or with fewer than two records), and a `comment` says that the forcing repeats with that period.

A `TrajectoryWriter` used directly, without `set_time_units`, writes
`units = "seconds"` and no `standard_name`.

### 5.3 Release variables

A record is written at the end of a step, so even a particle released at the
start of the run first appears one step later; the release position itself is
not in the records. The four release variables keep it, with dimension
`(trajectory)`:

- **`release_lon`, `release_lat`, `release_z`** — the position of each particle
  when the run starts at global step 0: the seed position, after the periodic
  fold of `numerics.periodic_lon/lat` when that is on. A particle with a
  delayed release keeps its seed position. Longitude and latitude are
  **geographic**, un-rotated on a rotated mesh like every record
  (`coordinate_frame.md` §4); `release_z` is the depth in metres, positive down.
- **`release_time`** — the instant the particle starts moving:

  ```
  release_time = T_g0 + max(r, 0) · dt · direction
  ```

  with `r` the particle's release step (0 without a delayed release). A
  particle is activated at the start of global step `r`, before the integrator
  runs, so this is the start of its first step. `r = round(release_sec / dt)`
  (`core/release.py`), so `release_time` is the **rounded** schedule, not the
  raw offset from the seed file. A particle that is never activated within the
  run still gets its scheduled `release_time`, which then lies after the last
  record (backward: before it). `release_time` is never NaN and has the
  `units` and `calendar` of `time`.

The release positions travel with the particle state (`ParticleState.release_lon`,
`release_lat`, `release_z`, held in the mesh frame like `lon`/`lat`) and every
checkpoint stores them (§6). `run_single_realization` writes the release
variables at the start of every call, so each segment of a resumed chain
carries the same values and is self-contained; a chain driver needs no change
for that. The positions are NaN only when they are unknown: a run resumed from
a restart written before format 2 (one `RuntimeWarning` says so). The
displacement of a particle from its release point is `lon − release_lon` and
`lat − release_lat`; do not use the first record for that.

### 5.4 Velocities

`vel_u`, `vel_v` and `vel_w` are the velocity the integrator evaluated for the
step that **ends** at this record, at the step's **start** position and time
(for Euler, RK2 and RK4 the evaluation at the start of the step; for RK45 the
velocity at the accepted end of its last substep, `time_integration.md` §7).
They belong to the start of the step, not to the recorded position, so they
have no `coordinates` attribute. `vel_u` and
`vel_v` are east and north components in m/s, un-rotated to geographic on a
rotated mesh. `vel_w` is the depth rate in m/s, positive down; on a sigma mesh
it is the dia-sigma flux, positive down, not the rate of `z`
(`vertical.md` §3). All three are FLOW velocities, with the same sign in
forward and backward runs: the integrators carry `time_dir` times the velocity
(the direction the particle moves), and the record path removes that factor.
Format 1 wrote the time-reversed value in backward runs.

**The NaN rule.** A velocity is written only where the particle was **ALIVE
through the whole step**: ALIVE after the release activation and before the
step, and still ALIVE after the step and the beaching hook. Everywhere else the
value is NaN. This covers the record in which a particle dies or strands,
every later record of a beached or outside particle, a particle that is not yet
released, and a particle resuspended during this step (the kernels skipped it).
The helper is `core.release.mask_inactive_velocity`; it masks the output copy
only, never the particle state.

A 2-D run has **no `vel_w` variable** (format 1 wrote zeros). A tracking run
creates `vel_u`/`vel_v` (and `vel_w` in 3-D) when it starts (§5.1). The combined
em/m1 diffusion step (with Euler) evaluates no stored velocity, so in such a run
they hold NaN throughout.

### 5.5 Sampled fields

Each field listed in `mode.sample_vars` becomes a `(trajectory, time)` float64
variable with the source variable's `units` and `long_name`. The value is the
field at the particle's recorded position and, as far as the forcing allows, at
the record's time (`field_sampling.md` §4):

- **interpolated mode**: the field is interpolated in time (linearly) to the
  record time. At the end of the forcing, where that is not possible, the
  start-of-step field is used.
- **discrete mode**: the field comes from the forcing record at the **end** of
  the step — the next record, the one the label points at — so its instant
  equals `time`. Only when there is no next record (the last step of a `stop`
  run at the end of the forcing; backward, at the first record) the
  start-of-step record is used. Format 1 used the start-of-step record in every
  discrete step, one record behind the label.

The variable's `comment` states these rules. A particle that is not ALIVE at
the record gets NaN.

### 5.6 CF conventions and the one deviation

The file states `Conventions = "CF-1.8"` and `featureType = "trajectory"`. The
`coordinates = "lat lon z"` attribute on `status`, `failure_reason` and the
sampled fields tells CF tools that those values sit at the recorded position
(the time is the coordinate variable `time`). The velocities carry no such
attribute (§5.4). `trajectory` has `cf_role = "trajectory_id"`.

One deviation is known and deliberate: CF (§9.3) wants an unlimited dimension
to be the outer one, while here the record dimension `time` is the inner one,
`(trajectory, time)`. That keeps one particle's track contiguous on disk, the
layout every trajectory tool reads fastest.

One choice is worth stating: all tracks share the coordinate variable
`time(time)`, CF's orthogonal multidimensional representation (§9.3.1), which
CF allows when every feature has the same element coordinates — here every
particle is recorded at the same instants. CF's trajectory examples use a
per-track `time(trajectory, obs)` instead, so a strict feature-type checker may
remark on the shared coordinate; the values are what the per-track form would
hold for every particle.

### 5.7 Reading the output

xarray reads both back ends and decodes `time` into dates — cftime objects for
calendars such as `noleap` or `360_day` — and indexes by them:

```python
import xarray as xr
ds = xr.open_dataset("trajectories.nc")                       # NetCDF
ds = xr.open_zarr("trajectories.zarr", consolidated=False)    # Zarr
ds.sel(time="2005-03-01")                     # the records of that day, by date
dx = ds["lon"] - ds["release_lon"]            # displacement from the release
                                              # (wrap into [-180, 180) at the dateline)
raw = xr.open_dataset("trajectories.nc", decode_times=False)
raw["time"].values                            # seconds, for time arithmetic
```

- For arithmetic in seconds, open with `decode_times=False`;
  `load_trajectories`, `netCDF4` and `zarr` read the raw seconds anyway.
- Because of the `coordinates` attribute, xarray loads `lat`, `lon` and `z` as
  coordinates of `status` and the sampled fields, and `.load()` on one of them
  reads the positions too. On a big store take `.values` of the variable you
  need, or open with `decode_coords=False`.
- Segments of a resumed chain join along `time`. All segments must have the
  same `kelp_output_format` and the same `time` `units`; then

  ```python
  ds = xr.concat([xr.open_zarr(p, consolidated=False) for p in parts],
                 dim="time", data_vars="minimal", coords="minimal",
                 compat="override", join="override")
  ```

  keeps the `release_*` variables once.

`load_trajectories(path)` returns plain arrays: `lon`, `lat`, `z`, `status` as
`(P, T)`, `time` `(T,)` in seconds, `trajectory` `(P,)`, plus `failure_reason`,
`vel_*`, the sampled fields and the four `release_*` arrays `(P,)` when they
are in the file. It reads format-1 and format-2 files and returns NaN where a
value was never written. `visualization.TrajectoryDataset` reads through it.

If no output path is given the writer is simply not created, and the run keeps
its results in an in-memory record instead (`TrajectoryRecord`, whose
`to_arrays()` returns `(T, P)` arrays with the same names, times in seconds and
the `release_*` arrays) — convenient for short runs and interactive use, where
writing a file would be overkill.

### 5.8 Provenance attributes

`run_tracking` stamps the settings that define the
trajectories as global attributes of the output. Always: `integrator`, `dt`,
`time_dir`, `mesh_type`, `start_time_idx`, `n_steps`, `mode_3d`,
`temporal_interp` (0/1), `temporal_reconstruction`, `time_averaging` (the
resolved value), `exact_nearest`, `grid_hash_cell`, `grid_hash_cell_rule`,
`particle_offset` (the shard offset of the random draws, 0 for an unsharded run),
`random_draws` (0/1: whether the run draws random numbers at all) and
`kelp_version` (package version plus the commit, read from `.git` without a
`git` binary); in 3-D also `w_interp`, `w_positive` and `w_contract`
(`reader-converted-2026-09-23` — an output without it was written by an engine
that used FESOM2/HEALPix W unconverted); option-dependent attributes (diffusion,
beaching, rotation, `uv_vertical = "linear"` for `mode.uv_vertical: linear`, …)
only when the option is on. The sharding concatenator
(`utils/concat_trajectories.py`) compares every attribute and refuses shards
that differ (until 2026-09-23 two shards run with linear and PCHIP temporal
reconstruction merged without complaint). It also refuses shards that differ in
`kelp_output_format`, in the name of the record dimension (`obs` or `time`), or
in the `units` or `calendar` of `time`; it merges format-1 shards as before.

### 5.9 Format-1 files

Every output written before format 2 is a format-1 file. It is recognised by
the missing `kelp_output_format` attribute and by its record dimension, which
is called **`obs`** instead of `time`. Format 1 differs in these points:

- `time(obs)` is not a coordinate variable and has no epoch
  (`units = "seconds or file index"`): in discrete mode it holds the index of
  the forcing record the step started from, in interpolated mode seconds from
  the first record;
- there are no `release_*` variables;
- a 2-D run has a `vel_w` filled with zeros, and a particle's velocity is
  written in the record in which it dies or strands and is 0 afterwards;
- in discrete mode the sampled fields come from the start-of-step record;
- the `restart_*` variables always exist, holding 9.97e36 (NetCDF) or 0 (Zarr)
  when no checkpoint was written;
- there are no `featureType`, `cf_role` or `positive` attributes, and the Zarr
  arrays carry no units.

Format-1 files stay readable: `load_trajectories`, `load_restart`,
`utils/concat_trajectories.py` (format-1 shards merge as before) and
`visualization.TrajectoryDataset` handle both formats. There is no converter.
Analysis code that must read both checks `"kelp_output_format" in ds.attrs` or
`"obs" in ds.dims`.

## §6 Restart checkpoints

A long run can be interrupted and resumed. `write_checkpoint` saves a **single**
restart record — the full particle state plus the bookkeeping needed to continue
— and **overwrites** it on each call, so the file never accumulates a history of
checkpoints; it always holds just the latest. What it stores: the particle state
(positions, status, nearest node, and failure reason if present), the release
positions (`restart_release_lon/lat/z`, in the mesh frame like the positions),
the simulation step index `step_i`, the file index `file_idx` the run had
reached, the beaching wet-step counter `wet_streak` (only when beaching is
active), the release steps of a delayed release (`restart_release_step`), the
carried containing elements of an edge-walk run (`restart_elem`), and the
mesh's coordinate-frame rotation angles (only for a rotated run). The
non-applicable pieces are simply not written.

**The restart block is created by the first checkpoint.** An output written
without checkpoints has no `restart_*` variables at all, and `load_restart` on
it raises `ValueError` ("No valid restart checkpoint"). (Format 1 created the
block when the file was opened, so a file without a checkpoint held 9.97e36 in
NetCDF and 0 in Zarr there.) Every checkpoint lists the optional arrays it
wrote in the attribute `restart_optional`, and `load_restart` reads only those,
so a rewrite without, say, `failure_reason` never returns the array an earlier
checkpoint left; a geographic checkpoint also removes rotation angles an earlier
one left. Every per-particle array is checked against the particle count
before the validity flag is cleared: a wrong-length call raises and leaves the
previous checkpoint valid.

`load_restart(path)` returns all of that as a dictionary, or raises if the file
holds no valid checkpoint; the state it returns carries `release_lon`,
`release_lat` and `release_z` when the restart has them. The resume recipe is
three steps: load the restart, **validate the coordinate frame** against the
mesh you are about to run on, then hand the state to `run_single_realization`
with `start_step_i` set to continue from where it left off and the time
position of the continuation:

```python
r = load_restart(path)
_validate_restart_frame(r["rotation_angles"], mesh)   # see coordinate_frame.md
_validate_restart_uv_vertical(r["uv_vertical"], mode)  # mode.uv_vertical unchanged
n = r["step_i"] + 1          # steps done, counted from a first call with start_step_i = 0
run_single_realization(..., init_state=r["state"],    # carries the release positions
                       start_step_i=n,
                       start_time_idx=start0 + n * int(time_dir),   # discrete mode; not wrapped
                       file_times=file_times,                  # the SAME axis as the first call
                       sim_time_start=t0 + n * dt * time_dir,  # interpolated mode
                       start_wet_streak=r["wet_streak"], ...)
```

The checkpoint stores no simulation time: the resumed call must pass the time
position of the continuation, or the forcing replays from the original start.
Here `start0` is the original start index and `t0` the original start time in
seconds since the first record (with temporal interpolation, as
`tests/test_restart_record_schedule.py` does). The config stores `time_dir` as
a float, hence `int(time_dir)`; `run_single_realization` accepts a whole-valued
float index and refuses a fractional one. Three rules keep the output
`time` labels of the segments equal to those of the uninterrupted run (§5.2):

- **Pass the same `file_times`** the first call used — under
  `time_averaging: mean` the midpoint-shifted axis that `run_tracking` builds
  (`extract_time_info` and then `tracking._shift_to_midpoints`). A segment
  given another axis does not continue the run: its labels — and, with
  temporal interpolation, the forcing times it reads — are off, by half a
  record interval for the raw axis.
- **Do not wrap the start index.** With `on_exhausted: cycle` in discrete mode,
  pass the unwrapped `start0 + n · time_dir`, even when it is past the last
  record; the reads wrap by themselves, and the label counts the forcing
  periods from that index. In discrete mode the `file_idx` that a checkpoint
  stores is that unwrapped index (format 1 stored the wrapped one, which
  labelled a resumed segment whole forcing periods too early), so
  `start_time_idx = r["file_idx"] + time_dir` is right too. With temporal
  interpolation `sim_time_start` is likewise the unwrapped `t0 + n · dt · time_dir`.
- **Open the same forcing.** The epoch of `time` is the forcing's first raw
  record; a segment that opens a forcing with another first record gets other
  `units`, so its seconds count from another origin (§5.7 asks for one `units`
  across the segments).

The release positions need no extra step: they ride in the state, so every
segment writes the same `release_*` variables. A driver that splits a restart
state into parts — for example a pulsed run split into independent shards
through its restart — slices it with `ParticleState.subset(index)`, which
slices every per-particle array, the release positions included. Slicing the
fields one by one misses fields added later; `write_release` and
`write_restart_file` refuse arrays whose length differs from the particle
count.

A chain over one SHARD of a release passes `particle_offset=` too (the
shard's first global particle index, as `run_tracking` resolves it from
`data.particle_offset` or the shard file's header —
`tracking._resolve_particle_offset(cfg)`): random draws are keyed by the
global index, and every segment output carries the `particle_offset`
attribute the sharding concatenator checks. `load_restart` returns the
offset a restart was written with (`r["particle_offset"]`, from the embedded
checkpoint's output attribute or from `write_restart_file(...,
particle_offset=)`), so a driver can refuse a mismatch.

The frame check refuses to resume a checkpoint on a mesh in a different
coordinate frame — see `coordinate_frame.md` for why that matters. The
`uv_vertical` check refuses to resume under another `mode.uv_vertical`
(`vertical.md` §2.1): a trajectory would mix the piecewise-constant and the
depth-linear horizontal velocity. `load_restart` returns the rule a checkpoint
was written with (`r["uv_vertical"]`: the trajectory file's `uv_vertical`
attribute, or `write_restart_file(..., uv_vertical=)`'s
`restart_uv_vertical`; `None` = `layer`, so checkpoints of runs without the
option carry nothing new). A driver that writes several segments through ONE
writer gets the same protection without a call: `run_single_realization`
refuses a segment whose rule differs from the one the writer already carries
(its `uv_vertical` attribute, or `layer` when it holds records without it),
before anything is written. The reason a
restart reproduces the original run *bit for bit* (rather than merely
approximately) is that the stochastic diffusion draws are keyed by the **global**
step index, not by anything that resets on resume: continuing from step `N`
produces exactly the random numbers the uninterrupted run would have produced at
step `N`. That counter-based randomness is described in `diffusion.md`.
The record cadence (`output_freq`) and the embedded-checkpoint cadence
(`checkpoint_freq`) follow the same global step index: a record is written
after global step `g` when `(g + 1) % output_freq == 0`, so a run resumed in
chunks writes exactly the records of the uninterrupted run. (Until 2026-09-23
the record counter restarted at every call: a chain of 91-step chunks with
`output_freq = 5` wrote a 6-step gap at every chunk boundary.)

### Standalone restart files and the resumable-chain pattern

`write_checkpoint` keeps just the *latest* restart record, embedded in the
trajectory output and overwritten each call. That is all a simple resume needs.
A run that must survive a hard time limit — say an 8-hour compute job continued
across many submissions — wants something more: a **kept history** of restarts it
manages itself. `write_restart_file(path, state, step_i, file_idx, ...)` is the
module-level function for that. It writes a self-contained NetCDF restart with the
same `restart_*` schema (so `load_restart` reads it back identically), but as its
own file rather than embedded in an output — so a driver can drop **one numbered
restart per checkpoint** (`restart_step00012345.nc`) into a dedicated directory
and never overwrite the earlier ones.

Both kinds are safe against an interrupted write. The embedded checkpoint keeps
a validity flag, `restart_valid`. The first checkpoint creates that flag before
any other `restart_*` variable and writes 0 into it (NetCDF: synced to disk),
then writes the payload, then sets the flag to 1. Every later checkpoint clears
the flag to 0 first and sets it to 1 last. So a process killed mid-checkpoint —
the first one included — leaves a file that `load_restart` refuses instead of a
mixture of two steps (detection, not a rollback: the previous checkpoint is
gone once its arrays are overwritten). `load_restart` treats a missing flag, or
a flag that was created but never written, as invalid. `write_restart_file`
writes to a temporary sibling and renames it into place after a clean close, so
the previous file at `path` survives a failed write. (The clear-first order
dates from 2026-09-23; since format 2 the flag is created lazily, and creating
it first keeps the first checkpoint covered.)

> **Engine upgrade inside a chain (2026-09-12):** `mesh.exact_nearest` is on by default and
> restarts carry no record of it. A chain started before the upgrade resumes its next segment
> with exact anchors unless the config pins `exact_nearest: false`; the output metadata attribute
> `exact_nearest` (0/1, always written now) says which engine wrote a segment.

The **resumable-chain** pattern built on this (it lives in a run driver, not in
the library) runs a long experiment as a sequence of bounded jobs:

1. **Resume or start.** Find the newest `restart_step*.nc` in the restart dir;
   if one exists, `load_restart` it, validate its frame, and continue from
   `step_i + 1`. Otherwise build the seeds and start fresh.
2. **Run in chunks.** Advance `restart_every` steps at a time. Each chunk writes
   one trajectory **segment** (`part_<step>.zarr`) *and* one numbered standalone
   restart file. Picking up the chunk-end state from the segment's embedded
   checkpoint (`load_restart(segment)`) feeds the next chunk.
3. **Stop on the job budget.** After a configured amount of model time (sized
   conservatively inside the wall-clock limit), stop — and, if the full run is
   not yet done, submit the next job, which resumes from the latest restart.
4. **Concatenate.** When the whole run is complete, stitch the per-chunk segments
   along the time axis into one trajectory store for analysis (§5.7: one
   `kelp_output_format`, one `time` `units`; the `release_*` variables are
   kept once).

Because the chunk boundary carries the full state forward and the randomness is
global-step-keyed, the chained run is **bit-for-bit identical** to one
uninterrupted run — the same property §6 relies on, now spanning job
resubmissions. Restart files are never auto-deleted; the driver (and you) manage
their cleanup. To change a parameter mid-run, edit the config before the next
job: the driver restores only the *state* from the restart and reads everything
else from the (possibly edited) config.

## §7 What is called when

The pieces above fire at three different rhythms over a run.

**Once, at startup.** `get_reader(model)` selects the reader; `load_mesh` and
`open_dataset` build the mesh and open the data (§1–§2). `extract_time_info`
reads the time axis into seconds and derives `file_dt`, and `dt` is auto-detected
if unset (§3). If `time_averaging` is `"mean"`, the timestamps are shifted to
window midpoints. If a `start_date` is given, it is resolved to a starting file
index against the raw timestamps.

**Every simulation step.** The step asks `_VelCache.get` for the velocity
snapshot(s) it needs; in temporal-interpolation mode a small helper blends two
adjacent snapshots to the exact step time (§4). The integrator then advances the
particles — see `time_integration.md`.

At the start of `run_single_realization` the writer gets the epoch and calendar
of `time` (`set_time_units`), the release variables (`write_release`) and, for
a shard, the global particle identifiers (`set_trajectory_ids`) (§5).

**Every output step.** `writer.append(state, time, sampled)` writes one
record per particle, labelled with the time at the end of the step (§5.2);
velocities of particles that were not alive through the step are written as
NaN (§5.4).

**Every `checkpoint_freq` steps.** `writer.write_checkpoint(...)` overwrites the
single restart record so the run can be resumed (§6).

The whole sequence, including where the stepping loop sits, is laid out in
`architecture.md`.
