---
name: kelp-nemo
description: Use when KELP is to track particles on NEMO output (ORCA family, regional NEMO configs, GYRE, crops of them) — adapting the model files to a config (mesh_mask vs domain_cfg, which variables, file names without a grid_T token, halo / fold / crop detection, partial cells, z* output, 360-day calendars) or when a NEMO run is refused, opens the wrong variable or loses particles at the edge.
---

# Adapting NEMO output to a KELP run

Authoritative references: **`docs/nemo.md`** (what the C-grid path does, layouts,
refusals, validation) and **`docs/configuration.md`** (keys `mesh.nemo_grid`,
`nemo_partial_cells`, `data.ncinput`, `ncinput_u/v/w`). This skill is the
adaptation workflow. The run itself follows **kelp-run**; the YAML mechanics
**kelp-config**.

Use `mesh.nemo_grid: cgrid` (exact C-grid, 2-D and 3-D) for any real NEMO
output. The default `tpoint` path is 2-D, E/N-aligned grids only.
`numerics.uv_interp` follows from `cgrid`: do not set another interpolator.

## Input checklist

1. **Mesh file** (`mesh.meshpath`): a `mesh_mask.nc` or a `domain_cfg.nc`.
   Needed: `glamf`, `gphif`, `tmask` (a `domain_cfg` without `tmask`: the mask is
   built from `bottom_level`), the T coordinates (`glamt/gphit` or
   `nav_lon/nav_lat`), `e2u`, `e1v`. 3-D also: `e3t_1d`, `e1t`, `e2t`, and
   `e3t_0`, `e3u_0`, `e3v_0` (partial cells; without them full reference
   cells are assumed — wrong on zps output, check `VertCoord` / `ncdump -h`).
2. **Velocity files**: u in grid_U, v in grid_V; T/S and `ssh` in grid_T. W is
   NOT read: the reader computes it from continuity (equal to the model's W
   on ORCA05); a grid_W file is optional and only compared in tests.
3. **File names**: the reader finds grid_U/V by swapping the `grid_T` token in
   the `data.ncinput` name. Files without that token (crops, renamed
   files): set `data.ncinput_u`, `ncinput_v` explicitly. Every one of the four
   keys takes a file, a glob or a list; several files per grid join along
   time (each file may carry its own epoch; the joined axis must be strictly
   increasing).
4. **Halo / fold / crop**: nothing to set. The layout is detected from the
   coordinates and printed at load and by `kelp info` (`cgrid_layout`): read that line.
   A crop (the file's `Iperio/NFold/NFtype` attributes claim more than the raster
   has) gets OPEN edges: a particle that reaches one is absorbed and ends
   `OUT_OF_BOUNDS` or `INTERPOLATION_FAILED`; count both as exits.
   Land-processor-eliminated output (zero-filled coordinate tiles) is refused:
   rebuild the global files (`rebuild_nemo`) first.
5. **Vertical**: `zco`/`zps` only. s-coordinates and ice-shelf cavities are
   refused. Partial cells: `mesh.nemo_partial_cells: shrunk` (NEMO <= 4.2) or
   `porous` (NEMO 5); only the depth in metres differs. z*/vvl output (a
   time-varying `e3t` in grid_T): a warning with the deviation, fixed `e3*_0`
   geometry is used (transports off by ~eta/H) — decide whether that is
   acceptable; there is no real z* yet.
6. **Seeds**: `lon lat depth_m` with depth positive down (3-D). A seed in
   land or below the floor stays put with zero velocity: check the seed
   cells with the mesh's `tmask` first.
7. **Arctic / transpolar**: add `mesh.rotation: fesom2` (cells degenerate above
   ~88 N; the reader warns).

## Variable names → config keys

| NEMO name(s) | what | kelp |
|---|---|---|
| `uo`, `vozocrtx`, `uos` (grid_U) | zonal velocity | auto alias; `mode.var_u` for others |
| `vo`, `vomecrty`, `vos` (grid_V) | meridional velocity | auto alias; `mode.var_v` |
| `wo`, `vovecrtz` (grid_W) | model W | not used (continuity W); optional file |
| `ssh`, `zos`, `sossheig`, `sshn` | free surface | auto alias; `mode.var_eta` |
| `votemper`, `thetao`, `thetao_con`, `toce` | temperature | `mode.sample_vars` — EXACT name in grid_T, no alias |
| `vosaline`, `so`, `so_abs`, `soce` | salinity | `mode.sample_vars` — exact name |
| `e3t`, `e3t_0`, `e3t_1d`, `e3u_0`, `e3v_0` | layer thicknesses | mesh file (`e3t` in grid_T: z* check only) |
| `glamf`, `gphif`, `glamt`, `gphit`, `nav_lon`, `nav_lat` | F / T coordinates | mesh file; F coordinates are required |
| `e2u`, `e1v`, `e1t`, `e2t` | model face / cell widths | mesh file |
| `tmask`, `bottom_level`, `top_level` | wet mask | mesh file (`mesh_mask` / `domain_cfg`) |
| `time_counter` | time axis | automatic; record times via `data.time_averaging` or `time_stamp`/`time_window`/`time_offset` |

`sample_vars` accept T-point fields only (U/V/W/F-point variables refuse) and
return the T-cell value, NaN on land.

## Pitfalls

- **Land values**: files carry 0 or NaN on land; both are fine, the closed
  faces are zeroed from `tmask`. Do not pre-fill or mask the files.
- **Calendars**: XIOS spellings `360d`, `365d`, `366d` are accepted and mapped
  to `360_day`, `noleap`, `all_leap`. A 360-day run starts with the model's
  dates: use `time.start_date` in the same calendar, and a record step that is
  a whole number of days (`time_averaging`: monthly means are stamped at
  month end, `mean` for window-end stamps; a start-stamped store needs
  `time_stamp: start`).
- **Stale halos** (older NEMO, copies in `e3t_0`, u, v): a halo or fold copy that
  differs from its canonical point refuses the mesh or the record; it means a
  wrong grid file for the output, not a kelp problem.
- **A `sample_vars` typo** is caught at the startup probe, not silently dropped.
- **Refused combinations**: rk45, `temporal_reconstruction: pchip`, em/m1,
  any diffusion in 3-D, beaching, `elem_search: walk` in 3-D,
  `Fdrift3D` with `tpoint`. Euler / RK2 / RK4 only.
- **Particles on land are not handled**: filter post hoc (the final or any
  position in a dry column / below the floor; velocity stays zero).
- **Discrete mode** (`temporal_interp: false`): `dt` must equal the record step.
  Use `temporal_interp: true` to sub-step.
- A run cost: ORCA05 global 3-D, ~9000 particles: ~14 ms per RK4 step on 4
  threads (`docs/nemo.md` §8; measure on your machine with `kelp-bench`).

## Minimal config (2-D surface, ORCA output)

```yaml
mode:
  Fdrift2D: true
  # sample_vars: [thetao, so]
numerics: {dt: 3600.0, integrator: rk4, output_freq: 24, temporal_interp: true}
mesh:
  type: nemo
  nemo_grid: cgrid
  meshpath: /path/mesh_mask.nc          # or domain_cfg.nc
  # rotation: fesom2                    # Arctic
data:
  ncinput: /path/EXP_*_grid_T_*.nc      # grid_U / grid_V siblings by the grid_T token
  # ncinput_u: /path/EXP_U_*.nc         # when the names lack the token
  # ncinput_v: /path/EXP_V_*.nc
  inicloud: seeds.dat
  output_path: ./.tmp/nemo_tracks.zarr
  time_averaging: instant               # or mean for window-end means
time: {start_time_idx: 0, n_steps: 240, on_exhausted: stop}
```

For 3-D set `mode.Fdrift3D: true` (and drop `Fdrift2D`), add
`mesh.nemo_partial_cells`, seed depths; the template is
`examples/config_nemo_cgrid_3d.yaml`. Check `kelp info config.yaml` before the
run: layout line, number of cells, first and last record time.
