# Running a tracking experiment

> **Covers:** how to drive a run from the config — the time window, temporal
> interpolation, direction, output and restart, delayed release, and where each
> optional subsystem (diffusion, beaching, sampling, rotated frame) is configured.
> **Does not cover:** the exhaustive per-key reference and defaults (see
> `configuration.md`) or the implementation mechanics (see the subsystem docs
> linked below). For turning a described experiment into a ready config + seeds +
> launch command, use the `kelp-run` agent skill.
> **Code:** `src/kelp/tracking.py:run_tracking`, `src/kelp/core/config.py:TimeConfig`,
> `src/kelp/cli/main.py:main`.
> **Verified at:** see `docs/doc_map.yaml` (stamped via check_doc_sync --bless).

A run is one YAML config plus one command:

```bash
kelp run config.yaml
```

Everything below is a key in that config. The complete option list with defaults
is in the README's *Config file reference* and, with full prose, in
[`configuration.md`](configuration.md). The run pipeline itself is in
[`architecture.md`](architecture.md).

## Time step (`numerics.dt`)

- **Auto** (omit `dt`): KELP reads the file time coordinate and uses the
  interval between records.
- **Explicit** (`dt: 1800.0`): in discrete mode `dt` must equal the file step —
  every step reads the next record while the particles move by `dt`, so a
  different value is refused at startup. With `temporal_interp: true`, `dt` may be
  **smaller** than the file step; a step that straddles a snapshot then runs on a
  steady velocity, so choose a `dt` that divides it.

Mechanics of the time axis (auto-detection, `file_dt`, the time-averaging
midpoint shift) are in [`io_and_time.md`](io_and_time.md).

## Temporal interpolation (`numerics.temporal_interp`)

By default each tracking step uses one file snapshot (discrete mode). Set
`temporal_interp: true` to linearly blend velocity between adjacent snapshots, so
`dt` can be smaller than the file step for smoother trajectories.

Linear-in-time blending caps the time order at O(dt²) regardless of RK order. To
let RK4 separate from RK2 at coarse file cadence, set
`numerics.temporal_reconstruction: pchip` (a 4-snapshot monotone cubic; with
`prefetch_size > 1`, a value ≥ 4 covers its stencil in one read). Default `linear` is byte-identical. Full treatment:
[`time_integration.md`](time_integration.md).

## The tracking window

| Set | Effect |
|-----|--------|
| `time.start_time_idx` | first file time index (default `0`) |
| `time.n_steps` | number of tracking steps (**preferred**) |
| `time.end_time_idx` | legacy: → `n_steps = end - start` (refused together with `start_date`: use `duration_days` or `n_steps`) |
| `time.start_date` + `time.duration_days` | start at an ISO datetime; `n_steps = days·86400/dt` |

`start_date` matches the raw file timestamps; starting **between** snapshots
requires `temporal_interp: true` (else it raises with that suggestion).

## Direction

`mode.time_dir: 1.0` forward, `-1.0` backward. Release times (below) are offsets
from the run start, so they behave the same in both directions.

## When the file runs out (`time.on_exhausted`)

`stop` (default) halts at the file boundary; `cycle` wraps around for periodic
forcing.

## Time-averaged input (`data.time_averaging`)

Models like FESOM2 write time-averaged output (e.g. monthly means) timestamped at
the **end** of the averaging window. Set `time_averaging: mean` to shift each
timestamp to the window midpoint; `instant` for snapshots. If unset, KELP
warns and suggests a default by mesh type (FESOM2 and HEALPix → `mean`, FESOM-C and NEMO → `instant`).
FESOM 1.4 (`mesh.type: fesom14`) is the exception: its reader resolves what time each record
represents (the window centre of a mean, from the files' time bounds or the raw output schedule, or
from `data.time_stamp` + `data.time_window`), so leave `time_averaging` unset — it is refused there
(`io_and_time.md` §3).

## Multiple realizations (`time.num_runs`)

`num_runs` repeats tracking with the same seeds, shifting `start_time_idx` by
`run_interval` each time (ensemble over forcing windows). When `num_runs > 1`,
output files get an `_r0`, `_r1`, … suffix; a single run keeps the plain name.
Under `on_exhausted: cycle` a start past either end of the records wraps by
whole forcing periods; under `stop` a schedule whose start (index, or the
`start_date`-shifted time) leaves the records is refused before any output.

## Delayed / staggered release

Release tracers from the same place(s) at different times in one run/output file
— a pulsed or continuous source. All particles are pre-allocated and **frozen**
(status `not_released`, written as NaN gaps) until their release time. The
output records each particle's release position and release instant
(`release_lon/lat/z`, `release_time`; see [`io_and_time.md`](io_and_time.md)
§5.3). Two ways, which compose additively:

- **Repeated pulses** (config): `time.n_pulses` re-releases the whole seed set,
  `time.pulse_interval` seconds apart. With `n` seeds this makes `n × n_pulses`
  particles.
- **Per-particle release time** (seed file): a 4th column `release_sec` — an
  offset in seconds from the run start — per seed line.

```yaml
time:
  n_pulses: 5            # release the seed set 5 times ...
  pulse_interval: 86400  # ... one day apart (seconds)
```

The schedule is saved in restart checkpoints. Defaults (`n_pulses: 1`,
`pulse_interval: 0`, no 4th column) are byte-identical to no delayed release.

## Optional subsystems

Each is off by default and byte-identical when off. Keys and defaults are in the
README reference and [`configuration.md`](configuration.md); the mechanics are in:

- **Stochastic diffusion** (`numerics.diffusion`) — [`diffusion.md`](diffusion.md).
- **Beaching / resuspension** (`numerics.beaching`) — [`beaching.md`](beaching.md).
- **Scalar sampling** (`mode.sample_vars`, `numerics.sample_interp`) —
  [`field_sampling.md`](field_sampling.md).
- **Pole-safe rotated frame** (`mesh.rotation`, FESOM2 + HEALPix) —
  [`coordinate_frame.md`](coordinate_frame.md). Output is always geographic.
- **u/v interpolation method** (`numerics.uv_interp`: `leastsq` /
  `barycentric` / `edge_leastsq`, plus `edge_leastsq_bc` coastal ghosts) —
  [`interpolation.md`](interpolation.md) §1 (the edge-neighbor method is
  §1.2).
- **Edge-walk element tracking** (`numerics.elem_search: walk` — search once
  at release, then cross edges; identical results, faster) —
  [`localization.md`](localization.md) §7.
- **Horizontal velocity linear in depth** (`mode.uv_vertical: linear`, 3-D
  runs on z-level meshes and on FESOM-C sigma meshes, where it is linear in
  the sigma fraction: node-layout u/v with `uv_interp: barycentric`,
  element-layout u/v with `leastsq`, `edge_leastsq` or `barycentric` — FESOM2
  on a mesh diagnostic with `nlevels` or `zbar_e_bottom`, FESOM-C as it is;
  the output then has the global attribute `uv_vertical = "linear"`) —
  [`vertical.md`](vertical.md) §2.1.

## Output and restart

Set `data.output_path` to stream trajectories to disk (backend by extension:
`.nc` → NetCDF, `.zarr` → Zarr; empty = keep in RAM). Enable periodic restart
checkpoints with `numerics.checkpoint_freq: N` (written into the same file).

```python
from kelp.writers import load_trajectories, load_restart
data    = load_trajectories("trajectories.nc")  # lon/lat/z/status (P, T), time (s), vel_*,
                                                # sampled vars, release_* (P,)
restart = load_restart("trajectories.nc")        # state, step_i, file_idx, uv_vertical, ...

import xarray as xr                             # CF trajectory file (output format 2):
ds = xr.open_dataset("trajectories.nc")         # time decodes to dates; ds.sel(time=...)
```

The file follows output format 2 (`kelp_output_format = 2`): dimensions
`trajectory` and `time`, `time` in seconds since the first forcing record with
that record's date as CF epoch and the forcing's calendar, release positions
and times per particle, velocities NaN where a particle was not alive through
the step. Files written by earlier engines (record dimension `obs`) stay
readable. The full layout is in [`io_and_time.md`](io_and_time.md) §5.

NetCDF holds a write lock while open (unreadable mid-run, and a hard-killed run
leaves it unclosed); set `numerics.output_flush_every: N` for periodic
close/append so it stays readable. Zarr is lock-free. The writer/restart
mechanics and the resume recipe are in [`io_and_time.md`](io_and_time.md).

For a long run that must survive a wall-clock limit (e.g. an 8-hour compute job
continued across many submissions), the embedded checkpoint is not enough on its
own — you want a kept history of restarts. `write_restart_file(...)` writes a
**standalone, numbered** restart file per checkpoint (readable by the same
`load_restart`), the building block for a **resumable-chain** driver: run in
chunks, write one trajectory segment + one numbered restart per chunk, resume from
the latest restart, resubmit until done, then concatenate the segments. A no-change
resume is bit-for-bit identical to one uninterrupted run. The pattern is described
in [`io_and_time.md`](io_and_time.md) §6.

## Big meshes

On a full global mesh, two `numerics` knobs cut the dominant I/O cost (both
default off → byte-identical for float32 stores): `velocity_float32: true`
(float32 u/v/w, half the memory of each processed snapshot) and
`prefetch_size: N` (read N snapshots per disk read). The prefetch window itself
is loaded in the on-disk dtype with every variable of the dataset at full depth,
so float32 does not shrink the window — size `N` against that raw window. To track only
a region, pre-cut it with `utils/extract_fesom_region.py`
([`utils/README.md`](../utils/README.md)). Sizing guidance:
[`performance.md`](performance.md).
