# Changelog

All notable changes to KELP are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/). Public history starts with 0.9.0;
the tracker was developed privately from 2026-06.

## [Unreleased]

### Added — horizontal velocity linear in depth (`mode.uv_vertical`)

- New option `mode.uv_vertical`: `layer` (default, unchanged — the horizontal
  velocity is constant within each layer) or `linear` — linear in depth between
  the layer centres, the value of the first centre above it and of each column's
  last wet centre below it, never extrapolated; a partial bottom layer is never
  read, and every vertex of the interpolation stencil is read in its own column
  (`docs/vertical.md` §2.1; a vertex whose bed lies above the particle is a
  wall: next section). For 3-D z-level runs with node-layout u/v (FESOM2
  `unod`/`vnod`, the HEALPix dual geometry), `numerics.uv_interp: barycentric`
  and `euler`, `rk2` or `rk4`, steady, linearly interpolated or PCHIP in time
  (each stored value is blended in time first, then read in depth). Refused before
  the output file is written: 2-D runs, other u/v interpolators, `rk45` (FESOM2
  element-layout u/v and FESOM-C sigma meshes: the next sections).
- Each record a run reads with the option is checked: a non-finite `u` or `v`
  above a column's last usable layer centre stops the run with the node, the
  layer and the record named. The HEALPix reader checks the raw record before it
  zeroes missing values (so a drifting surface mask is refused, not zeroed) and
  declares where each column's u data end (`mesh.metadata["uv_data_last"]`).
- Output files of runs with the option carry the global attribute
  `uv_vertical = "linear"` — the proof the option was on (a misspelled key is
  ignored silently, like every unknown key). Runs without it are bit for bit
  unchanged and carry no new attribute.
- The rule cannot change inside one trajectory: `load_restart` returns the rule
  a checkpoint was written with (`uv_vertical`, `None` = `layer`;
  `write_restart_file` takes `uv_vertical=`), `tracking._validate_restart_uv_vertical`
  refuses a resume under the other rule (the resume recipe calls it next to the
  frame check), and `run_single_realization` refuses a segment whose rule
  differs from the one its writer already carries — the first segment binds its
  rule to the writer before any write, in memory only, so a segment that wrote
  only a checkpoint counts too; `open()` creates a new file and starts unbound.
  `TrajectoryWriter.global_attr` returns a global attribute written through the
  writer, `bound_rule`/`bind_rule` the rules bound to it.
- Engine: `VelocityField.uv_vertical` (`"layer"`, `"layer_linear"`, `"level"`)
  records where u/v sit in the vertical and travels through every time blend and
  the shared snapshot cache; the new module `numerics/vertical_uv.py` holds the
  data-point table, the bracket and the record check. Readers may set
  `native_uv_vertical = "level"` (u/v on the levels, always linear in depth) and
  provide a hook `prepare(mesh, ds)`; the FESOM 1.4 reader (next section) uses
  both.
  `prepare` runs at every entry: right after `open_dataset` (before any output
  file exists) and again at the entry of each realization, after the
  `max_levels` cut — 1 + R calls for R realizations, one per direct
  `run_single_realization` — so mesh facts a dataset declares (`uv_data_last`)
  are always those of the dataset, mesh, levels and reader in use; the engine
  keeps no record of earlier calls. The engine takes a snapshot of the
  declaration at first use (an owned, read-only copy that replaces the
  reader's array in `mesh.metadata`): assign a new array to change it —
  writing into the reader's old array changes nothing, writing into the
  snapshot raises.
- The cost of the option is measured by
  `benchmarks/single_node/depth_linear_cost.py` (128 threads, 1e5 particles,
  with the walls at depth of the next section): +10 % per fused 3-D node RK2
  step with a linear time blend, no measurable cost with PCHIP; element
  `leastsq`: +6 % per fused 3-D Euler step, +3 % per unfused 3-D RK2 step,
  no measurable cost with PCHIP.

### Added — depth-linear u/v on FESOM2 element layout; walls at depth

- `mode.uv_vertical: linear` works on FESOM2 element-layout u/v (`u`/`v` on
  the triangles) with `numerics.uv_interp` `leastsq` (the default),
  `edge_leastsq` (either `edge_leastsq_bc`) or `barycentric`, and `euler`,
  `rk2` or `rk4`, steady, linearly interpolated or PCHIP in time, with
  `elem_search: walk`, the split uniform and Az kicks, prefetch windows, the
  shared snapshot cache, checkpoints and restarts, multi-realization runs and
  `max_levels`. Each element is read linearly in depth between its own layer
  centres, down to its own last wet layer (`docs/vertical.md` §2.1). The fused
  3-D Euler kernel has a depth-linear twin (bit for bit the unfused step);
  RK2 and RK4 on element data run unfused, as in layer mode.
- **Walls at depth, for nodes too** — a change of the node-layout rule of the
  previous section, which is in no release yet: a member of the
  interpolation stencil (a vertex, or an element) whose bed lies above the
  particle is a wall, like the coast. It is left out of the barycentric sum
  (the weights of the wet vertices renormalised, a negative one — a particle
  just outside an edge — counted as zero) or of the `leastsq` fit; for
  `edge_leastsq` its edge is a boundary edge with the `edge_leastsq_bc` ghost;
  for element `barycentric` it is left out of each vertex mean. Not a zero
  velocity, which would brake every particle along the wall, and no longer the
  shallow vertex's bottom value read below its bed (the earlier node rule),
  which could speed a particle up. When the wet members cannot support the
  interpolator, their area-weighted mean (`leastsq`, `edge_leastsq`) or the
  equal-weight mean of the remaining vertices (`barycentric`, when their
  weights vanish); when none is wet, the containing element (node layout: the
  vertex with the largest bed) at its last usable point. With every member wet
  the result is the layer interpolator's arithmetic. The velocity jumps where
  a particle crosses the depth of a neighbour's bed, and near the bed the two
  modes differ by more than the vertical interpolation: layer mode reads a
  shallow neighbour's stored zeros, linear mode leaves that neighbour out.
- Element bottoms: the FESOM2 reader reads `nlevels`, `zbar_e_bottom` and
  `ulevels` from the mesh diagnostic (NetCDF or Zarr) when present and
  declares each element's bed and last wet layer in `mesh.metadata`
  (`elem_bed`, `uv_data_last_elem`; `elem_ulevels`). A level variable the
  reader uses (`zbar_e_bottom`, else `nlevels`; and `ulevels`) that is not one
  finite value per element, or holds levels outside `1..K`, refuses the mesh at
  load, in layer runs too, naming the file and the fix.
  `utils/extract_fesom_region.py` keeps `nlevels`, `ulevels`, `zbar_e_bottom`
  and `nlevels_nod2D` in the subset mesh when the source has them; a subset
  written before must be extracted again to use the option on element data.
- Refused before the output file exists, naming the mesh file: element data on
  a mesh without element bottoms (a diagnostic with neither `nlevels` nor
  `zbar_e_bottom`, an ASCII mesh directory, an older subset), meshes with
  ice-shelf cavities (`ulevels > 1` anywhere; not supported yet), and an
  interpolator outside the table (node layout: `barycentric`; element layout:
  `leastsq`, `edge_leastsq`, `barycentric`). Each element record a run reads is
  checked like a node record: a non-finite `u` or `v` at a usable layer refuses
  the run, naming the element, the layer and the record. Runs without the
  option are bit for bit unchanged.
- Engine: `VelocityField(uv_vertical="layer_linear")` accepts element layout;
  `numerics.vertical_uv.depth_points(mesh, placement, layout)` builds one table
  per placement and layout, with the physical bed of every column
  (`DepthPoints.bed`, the wet test `uv_wet`). Analytic tests through the public
  run path (`tests/test_depth_linear_analytic.py`, with a mutation pass): a
  depth-constant field gives the layer answer, a field linear in depth is
  exact, a smooth profile converges at second order in the layer thickness
  (layer mode at first), and a flow along a trench keeps its speed at the
  trench walls.

### Added — depth-linear u/v on FESOM-C sigma meshes

- `mode.uv_vertical: linear` works on FESOM-C sigma meshes (`mesh.type:
  fesomc`): element u/v (`u`/`v`) with `leastsq`, `edge_leastsq` or
  `barycentric`, node u/v with `barycentric`; `euler`, `rk2` or `rk4`, steady,
  linearly interpolated or PCHIP in time, with the same companions as on FESOM2
  (edge walk, kicks, caches, restarts, multi-realization runs, `max_levels`).
  u/v are read linearly in the sigma fraction `s = z / H_p` of the particle's
  column between the sigma layer centres `1 − (σ_k + σ_{k+1}) / 2`, each RK
  stage at the `s` of its own depth and column height; every column is wet to
  the bed (no walls at depth, no element bottoms needed), and every layer of
  every record read must be finite (`docs/vertical.md` §2.1). Exact for a field
  linear in `s`; on a sloping bed a field linear in `z` is not claimed. The
  sigma path stays unfused, as in layer mode; runs without the option are bit
  for bit unchanged. Analytic tests through `run_tracking` (the layer answer
  for a depth-constant field, second order in the sigma layer thickness for a
  smooth profile, layer mode first) with sigma mutants in their mutation pass;
  `benchmarks/single_node/depth_linear_cost.py --vertical sigma` measures the
  cost per unfused 3-D RK2 step (128 threads, 1e5 particles): element
  `leastsq` no measurable cost with a linear time blend, +3 % with PCHIP;
  node `barycentric` +3 % and +4 %.

### Added — FESOM 1.4 reader (`mesh.type: fesom14`)

- A reader for FESOM 1.4 output. The mesh comes from a FESOM 1.4 ASCII mesh
  directory (`nod2d.out`, `elem2d.out`, `nod3d.out`, `aux3d.out`, cross-checked
  on load) or from level-wise data files that hold the node cell polygons
  (`lon_bnds`/`lat_bnds`): the triangles are recovered exactly from the
  polygons, and a `deptho` variable is checked against the wet mask. A mesh
  stored in the model's rotated frame is rotated back with
  `mesh.input_rotation` (the Euler angles of the run; refused for other mesh
  types); a global ASCII mesh without a node north of 85° N draws a warning
  when the angles are missing.
- Data in either form FESOM 1.4 writes: level-wise `(time, depth, node)` files
  (e.g. the published CMIP6 output) and the raw writer's vectors over the 3-D
  nodes `(time, nodes_3d)`, mapped through the ASCII mesh's index table. Files
  are grouped by the variables they hold, not by their names; only the
  variables a run consumes are opened (the velocities, `zos`, the
  `mode.sample_vars` and, with beaching enabled, `beaching.dry_mask_var`). u, v and w come back on the levels and
  are always read linearly in depth (`uv_vertical = "level"`): a prism-wise
  approximation (barycentric on the triangle, linear in depth per vertex) of
  FESOM 1.4's own interpolation on tetrahedra. Every record of every variable
  is checked against the mesh's static wet mask on every level before anything
  is changed; values below the bed are then 0 (u, v, w) or the column's last
  value (sampled scalars). 2-D tracking takes one level (`mode.sigma_layer`),
  the depth mean (`mode.sigma_mean`) or the surface.
- Time: each file is decoded with its own epoch and calendar, and the reader
  resolves what each record represents — the centre of its averaging window
  for a mean, the stamp for a snapshot — from the files' time bounds, the raw
  writer's `output_schedule` (checked against the stamps) or the new keys
  `data.time_stamp` + `data.time_window`; `data.time_offset` shifts the raw
  stamps first (a raw job that started mid-day). A window is never estimated
  and a gap in the records refuses the run. For this reader `run_tracking`
  applies no midpoint shift, refuses `data.time_averaging`, and writes
  `time_averaging = "resolved_by_reader"`; the three new keys are refused for
  every other mesh type.
- Refused before the output file exists: em/m1 diffusion (FESOM 1.4 output
  has no Kh fields) and `beaching.use_eta: true` without a sea-surface
  height in the files. The offline W reconstruction (`w_reconstruct.py`,
  `utils/reconstruct_w.py`) refuses FESOM 1.4 meshes and data on the levels:
  FESOM 1.4 output holds `wo` itself.
- Validation: the level-wise form against the published CMIP6 AWI-CM-1-1-LR
  output (data-gated `tests/test_fesom14_published_lr.py`, directory from
  `KELP_TEST_FESOM14_DATA`): mesh recovery, the absolute time axis, first-step
  velocities against hand values, the w sign, PCHIP in time. The raw form and
  the ASCII mesh are checked against FESOM 1.4's description of its formats
  and by a round trip on real geometry (the published mesh and fields written
  in those forms give bit-identical trajectories), not against real raw
  output. Example: `examples/config_fesom14_levelwise.yaml`; I/O cost on that
  mesh: `benchmarks/single_node/fesom14_lr_cost.md`.
- `cftime` is a declared runtime dependency (it was installed with `netCDF4`
  already).

### Changed

- `kelp info` opens the data as a run does, for every reader: `data.ncinput`
  together with `data.ncinput_v` / `data.ncinput_w` (before, `ncinput` only),
  the stamp options, and the reader's `prepare` checks against the mesh. It no
  longer imports the tracking loop: the helpers it shares with a run
  (`_prepare_dataset`, `_time_option_kwargs`, `_describe_record_times`) moved
  from `tracking.py` to `readers/__init__.py`, and `tracking` imports them
  under the same names.
- `VelocityField` checks the rows of a 3-D `u`/`v` against `w`: layer data must
  carry `K-1` rows (one per layer). A FESOM2 dataset whose u/v carry `K` rows
  (on the levels) now raises in the constructor instead of running on the first
  `K-1` rows.
- `mode.max_levels` keeps the uncut level table (`mesh.metadata["sigma_full"]`)
  and cuts from it, so a mesh used again with another limit gets that limit;
  `run_single_realization` re-applies the limit to a mesh cut this way before
  (also for runs without the option).
- A HEALPix 3-D mesh load (dual geometry) reads the first record of `u` once
  more, down every column, to declare the u data prefix; 2-D loads do not.

### Documentation

- The per-module notes of `CLAUDE.md` (the module details and the import graph)
  moved, word for word, to `docs/dev/`: an index (`docs/dev/README.md`) and nine
  notes files. `CLAUDE.md` shrinks from about 140 kB to 23 kB and names the notes
  files; agents read them on demand. No code change.

## [1.0.0] - 2026-10-14

Release 1.0 fixes the trajectory output format (format 2, below): from this
version on, a change to the file schema is a breaking change. The persistent
identifier of the archived release is added to `CITATION.cff` once the release
has been archived.

### Changed — BREAKING: trajectory output format 2

The trajectory file is a public contract from release 1.0 on, so its schema was
fixed before the release. Every file now carries the global attribute
`kelp_output_format = 2`; a file without it was written by an earlier engine
(format 1). Particle positions are unchanged bit for bit; labels, attributes
and masks change. The full description is in `docs/io_and_time.md` §5.

- **Record dimension renamed `obs` → `time`.** Every per-record variable is
  `(trajectory, time)`, and `time(time)` and `trajectory(trajectory)` are
  coordinate variables. `release_lon`, `release_lat`, `release_z` and
  `release_time` are reserved names, so a sampled field cannot use them
  (`obs` stays reserved).
- **`time` has a calendar epoch and holds seconds in both time modes.**
  Discrete mode used to write the index of the forcing record. The record
  written after global step `g` now holds
  `T_g0 + (g + 1) · dt · time_dir`, the time at the end of the step, in
  seconds since the first raw record of the forcing (`T_g0` is the start of
  the run on that clock). `units` is `seconds since <first record>` with the
  forcing's `calendar` (for example `noleap` or `360_day`), so xarray decodes
  the dates and `ds.sel(time=...)` works; arithmetic in seconds needs
  `decode_times=False`. A forcing without a calendar epoch gives
  `units = "seconds"` and a `time_origin` attribute. With
  `on_exhausted: cycle` the time keeps counting past the end of the forcing
  (attribute `forcing_cycle_period`). The labels do not depend on how a run is
  cut into chain segments.
- **Release positions and times are written.** `release_lon`/`release_lat`
  (geographic), `release_z` (m, positive down) and `release_time` (the instant
  the particle starts moving, from the rounded release schedule; never NaN)
  have the dimension `(trajectory)`. A particle's first record lies at least one
  step after its release, so before this change the release position was lost. The particle
  state carries the positions (`ParticleState.release_lon/lat/z`), and
  checkpoints and standalone restart files store them
  (`restart_release_lon/lat/z`), so every segment of a resumed chain writes
  the same values without a change to the driver. A run resumed from an older
  restart writes NaN positions and warns once. `ParticleState.subset(index)`
  slices every per-particle array, these included; drivers that split a state
  should use it instead of slicing fields one by one.
- **Velocities are NaN unless the particle was alive through the whole step.**
  `vel_u/vel_v/vel_w` used to carry a value in the record in which a particle
  died or stranded and 0 afterwards; now that record, dead, stranded and
  not-yet-released particles, and a particle resuspended during the step are
  NaN (`core.release.mask_inactive_velocity`). The attribute `comment` states
  the meaning: the velocity of the step that ends at the record, at the step's
  start position and time (RK45: at its end).
- **Velocities are flow velocities in backward runs too.** A backward run
  wrote `time_dir` times the velocity (an eastward current read negative); the
  record now removes the factor, so `vel_u/vel_v/vel_w` have the same sign in
  both directions.
- **No `vel_w` in 2-D.** A 2-D run used to write a `vel_w` of zeros; the
  variable is now absent. The in-memory `TrajectoryRecord` no longer invents a
  zero `w` either and keeps its velocity lists aligned with the positions.
- **Fill values.** Positions, velocities, sampled fields and the release
  variables use NaN as fill value on both back ends; `failure_reason` uses 0.
  A NetCDF velocity variable that missed a record read back 9.97e36 there; a
  lazily created variable now reads NaN in every record that did not carry it.
- **Sampled fields in discrete mode are taken at the record's time.** They come
  from the forcing record at the end of the step, so their instant equals
  `time`; only the last step of a `stop` run without a next record uses the
  start-of-step record. Before, every discrete-mode sample was one record
  behind the label.
- **The restart block is created by the first checkpoint.** A file without
  checkpoints has no `restart_*` variables (it used to hold 9.97e36 in NetCDF
  or 0 in Zarr). The first checkpoint creates `restart_valid` first and writes
  0 before any other restart variable, so an interrupted first checkpoint
  reads as invalid; `load_restart` also refuses a flag that was never written.
- **CF trajectory attributes on both back ends.** `featureType = "trajectory"`,
  `cf_role = "trajectory_id"` on `trajectory` (values `particle_offset +
  0 … P−1`, the global particle index also in a shard), `standard_name` and
  `units` on `lon`/`lat`/`z` (`z`: `positive = "down"`), `coordinates =
  "lat lon z"` on `status`, `failure_reason` and the sampled fields, and the
  same attributes in Zarr as in NetCDF. Known deviation: the record dimension
  is the inner one, so that each track is contiguous on disk.
- **Format-1 files stay readable.** `load_trajectories` (it also returns the
  `release_*` arrays when present), `load_restart`, `visualization.TrajectoryDataset`
  and `utils/concat_trajectories.py` read both formats. The concatenator now
  refuses shards that differ in `kelp_output_format`, in the record dimension
  name or in the `units`/`calendar` of `time`; it merges format-1 shards as
  before. There is no converter for old files. Analysis code that must read
  both formats can check `"obs" in ds.dims`.

### Fixed

- **Rotated mesh: `vel_u`/`vel_v` were turned to geographic components at the
  wrong point.** A velocity's east/north components depend on where the vector
  sits; the record un-rotated the velocity evaluated at the step's start at the
  step's END position, which turned it by the frame's change of direction along
  the step (about 0.4 % of the speed at 80 °N over a daily step; up to 180° for a
  step across the geographic pole). The record now un-rotates at the start
  position (`rk45`, which evaluates at the end, keeps the end). Positions were
  never affected; geographic meshes are unchanged.
- **`rk45` in a rotated mesh with diffusion un-rotated `vel_u`/`vel_v` at the
  kicked position.** `rk45` evaluates the velocity at the end of the step,
  before the diffusion kick; the record now uses that point.
- **Interpolated forcing read records past `n_time_total`.** A direct caller
  passing an axis longer than `n_time_total` got a cycle period and labels from
  the records the run uses, but brackets from the whole axis (`cycle` read
  records beyond the period, `stop` ran past it). One active axis now serves
  the brackets, the exhaustion check, the period and the labels; `run_tracking`
  was not affected (it passes the whole axis).
- **A shard or segment whose particles were all dead had no velocity
  variables**, and `concat_trajectories` refused to merge it with the other
  shards. A tracking run now creates `vel_u`/`vel_v` (and `vel_w` in 3-D) when
  it starts (`TrajectoryWriter.declare_velocity`), so every file of a run has
  the same variables; `concat_trajectories` fills NaN velocities for such a
  format-2 shard written before this change (none of its particles ever alive).
- **The kelp-analysis "last segment" check failed on every backward archive**
  (the record spacing is negative there); it compares with the spacing's
  magnitude now.
- **A single cycled forcing record ignored the start index.** With one record
  and `on_exhausted: cycle`, every start (and every realization shifted by
  `run_interval`) was labelled from that record's time; record `k` is now
  `t0 + k·dt`, the period the file states.
- **A float start index broke a resumed run.** The resume recipe computes
  `start0 + n * time_dir` with the config's float `time_dir`; xarray refused the
  float index. `run_single_realization` now accepts a whole-valued float index
  and refuses a fractional one, and the recipe uses `int(time_dir)`.
- **`utils/extract_fesom_region.py` wrote only the last time step to NetCDF.**
  The streaming writer appended every step after the first with xarray's
  `to_netcdf(mode="a")`, which overwrites a variable of the same name instead
  of extending the unlimited time dimension: a 10-day crop came out as one
  record (day 10), silently. Steps are now appended with netCDF4 along the time
  dimension, and the first write fixes the time encoding (float seconds) so
  sub-daily steps are not truncated. The `combined` layout had the same defect
  and is fixed the same way. Zarr output kept every step but stored sub-daily
  steps in the unit xarray chose from the first value (6-hourly stamps came
  out 6 days apart); the first write now fixes float seconds there too. A
  packed (scale/offset) integer source no longer turns missing values into a
  finite number on appended steps. Found by the real-data runs and the review
  of the output-format work (`tests/test_extract_region_stream.py`).

- **BREAKING — vertical velocity sign (FESOM2 and HEALPix, 3-D).** FESOM2
  writes `w` positive **upward** (`oce_ale.F90:vert_vel_ale`), and so do the
  HEALPix products derived from it (`wo`; ng5 `avg_wo`, "upward sea water
  velocity"). KELP integrates a positive-down depth and passed the file value
  through unchanged, so every FESOM2/HEALPix 3-D run moved particles
  vertically in the wrong direction. The readers now negate it. The file's
  convention is resolved from the new `mode.w_positive` (`auto` | `up` |
  `down`), then the variable's CF `positive` attribute, then its CF
  `standard_name` (`upward_…` / `downward_…`), then the reader's producer
  default (`up`). Listing the vertical velocity under `mode.sample_vars` is
  refused where the scalar path would mis-stagger it (HEALPix, FESOM-C
  `w_cart`): `vel_w` records W at the particle.
  The recorded `vel_w` is the tracker's positive-down rate, so its sign in
  FESOM2/HEALPix 3-D outputs is opposite to that of older outputs (which
  recorded the file's positive-up value); a FESOM2 `w` listed in
  `mode.sample_vars` is sampled as a scalar with the file's sign.
  `utils/reconstruct_w.py` writes positive-down W
  and now stamps `positive = "down"`; files it wrote before this change need
  `mode.w_positive: down`. 3-D trajectories from earlier versions are not
  reproducible and should be rerun.
- **Coastal FESOM 3-D vertical velocity.** The reader used FESOM-C's `w_cart`
  as a positive-down interface field and dropped the motion of the sea surface,
  although the sigma tracker integrates the depth below that moving surface —
  and `w_cart` itself lacks the dia-sigma part of the vertical velocity. KELP
  now tracks with FESOM-C's `w` (`Wvel`, the flux through the sigma
  interfaces) and integrates the particle's sigma fraction, `ds/dt = −Wvel/H`,
  placing it at `z = s·H` in its water column at every stage: the moving
  surface and the flow along a sloping bottom are carried exactly, for any
  time interpolation. Vertical velocity is sampled along the sigma surfaces,
  and the diffusion kicks keep the sigma level (horizontal) or reflect in the
  end-of-step column (vertical). `w_cart` is refused with an explanation. On
  sigma meshes the recorded `vel_w` is the dia-sigma flux. Coastal 3-D
  trajectories change; 2-D runs are unaffected.
- Backward runs with a spatially varying diffusivity (`em`/`m1` and the
  vertical `Az` kick): the diffusivity-gradient drift was multiplied by the
  signed time step, so a backward ensemble drifted into low-diffusivity water
  (closed column: density ∝ K⁻² instead of uniform). The drift belongs to the
  diffusion and now uses `|dt|` like the noise; only the advection reverses.
  Forward runs are unchanged.
- Vertical `Az` drift: the gradient was a central difference over half a
  layer, which straddles the next interface, so it was not the derivative of
  the piecewise-linear `Az` used for the noise and the uniform (well-mixed)
  density was not stationary even for vanishing `dt` on non-linear profiles.
  It is now the slope of the segment at the particle depth.
- `edge_leastsq_bc: freeslip` used a ghost that copied the element's whole
  velocity — a zero-gradient condition that let the reconstructed flow cross
  the coast. The ghost is now the mirror image (tangential kept, normal
  reversed). Runs with `uv_interp: edge_leastsq` change near coasts.
- Random walk on the sphere: horizontal kicks were applied on the
  longitude–latitude chart, which lacks the metric drift `−K tan(lat)/R`
  (equilibrium uniform in latitude instead of area). Kicks now move along
  the great circle (rotation-invariant for isotropic `K`: the uniform area
  density is kept exactly; regular at the poles);
  `Kx ≠ Ky` adds `(Kx − Ky) tan(lat)/R`. Diffusion runs change slightly,
  most at high latitude.
- Checkpoints: the embedded checkpoint rewrote its arrays under a
  `restart_valid` flag left at 1 from the previous checkpoint, so a write
  interrupted between two arrays loaded as a valid mixture of two steps. The
  flag is now cleared first and set last; `write_restart_file` writes a
  temporary file and renames it into place, so a failed write keeps the
  previous restart.
- Time axis: `cftime` axes (e.g. noleap) and integer axes with time units were
  not decoded, and sub-second `datetime64` steps were truncated; a run that
  requested `temporal_interp` then fell back silently to one snapshot per
  step. The axes are decoded now (CF epochs through xarray's decoder, bare units
  such as `h`, `d`, `ms` by an exact token table), `start_date` is resolved in
  the axis calendar (e.g. `2001-02-30` on a 360-day axis), and requested
  interpolation without a usable axis (and without `numerics.file_dt`) stops
  before the output is created.
- Offline W reconstruction (`w_reconstruct`): the bottom correction read the
  residual at the global last interface (so `w(H) ≠ 0` on shallow nodes) and
  re-filled masked below-bottom interfaces. It now uses the local bottom and
  masks after the correction.
- Beaching: the dry mask beached particles but did not stop their
  resuspension — a permanently masked node with a large nominal depth released
  its particle every step. Masked (and unlocated) nodes are never wet now.
- Output provenance: `temporal_interp`, `temporal_reconstruction`,
  `time_averaging` and `kelp_version` (version + commit), and in 3-D `w_interp`,
  `w_positive` and `w_contract`, are now always written to the output metadata
  (drivers that call `run_single_realization` with their own writer get all of
  them except `time_averaging`, which only `run_tracking` resolves),
  so the sharding concatenator refuses shards that differ in them.
- Shared snapshot cache (`numerics.shared_fields_dir`): the namespace hashed the
  configuration and paths only, so data rewritten at the same path was served
  from the old cache. It now includes the size and modification time of the
  mesh and data files (and of a data directory's top-level entries), the
  engine version and the run generation (new `numerics.shared_fields_generation`,
  default: the SLURM array/job id and srun step; outside SLURM without it the
  cache is not used), so separate runs never share snapshots. A new run
  removes the namespaces of earlier runs whose processes have all ended.
- Record and checkpoint cadence across restarts: `run_single_realization`
  counted `output_freq` with a per-call counter, so a resumed chunk started
  a new record phase — a chain of 91-step chunks with `output_freq: 5` wrote
  a 6-step gap at every boundary (252 such gaps in a 63-year run). Records and
  embedded checkpoints now follow the global step index; runs that are not
  resumed are unchanged.
- Separately stored fields attached by label: a W file written by
  `utils/reconstruct_w.py` carried its time axis as relative seconds without
  units, and the FESOM2 reader attaches `data.ncinput_w` (and `ncinput_v`, and
  the per-variable files of an input directory, and FESOM-C's separate v/w
  files) onto the velocity dataset by its time labels — no stamp matched, and
  the attached W was NaN at every record. The tool now writes its source's own
  time coordinate (units and calendar kept) and refuses separate v/ssh/ALE-w
  inputs on other records. The readers match an attached file to the velocity
  file record by record, as instants: it may hold more records (with dates a
  finer cadence is sub-sampled onto the velocity records with a warning), a W
  may cover a block of the records (a `time_range` sub-run; FESOM2 single-file
  and directory inputs; a run that needs an uncovered record is refused at
  that record, look-ahead reads are not), and a coarser cadence, another
  epoch or unit, a v file missing records (FESOM-C: any attached file missing
  records), a file without a time dimension, or stamps repeated on one side
  only are refused instead of padded with NaN; in a 2-D run a W file that does
  not fit is skipped with a warning. The FESOM2 multi-year layout keeps its
  exact merge. In an input directory a non-fitting `w*` file is refused in
  3-D and skipped with a warning in 2-D, a non-fitting `ssh*` or extra file is
  skipped with a warning (one covering only a block of the records is attached
  with a warning naming the NaN records), and `ncinput_v`/`ncinput_w` next to a directory
  `ncinput` (which ignores them) now warn. W files written by the tool before
  this change must be regenerated.
- Configuration values that passed the load-time checks silently are now
  refused when the file is loaded: `mode.time_dir` other than 1 or -1 (the
  value multiplied every velocity while the clock advanced by its sign only,
  so `2.0` doubled every displacement), an unknown `numerics.integrator`
  (checked only after the output file had been created) or `w_interp`,
  `data.time_averaging` other than `mean`/`instant`, `time.on_exhausted` other
  than `stop`/`cycle` (the `start_date` + `duration_days` path ran `Cycle` as
  `stop`), `time.duration_days` without `start_date` (dropped silently),
  `mode.sigma_layer` together with `mode.sigma_mean` (FESOM-C took the mean for
  the velocity and the layer for sampled scalars), a quoted string where a list
  is expected (`periodic_lon: "12"` became `(1.0, 2.0)`), a boolean grid-hash
  cell, and a cell of 45 degrees or more (accepted, then refused by the next
  rebuild). An empty configuration section now loads with its defaults
  instead of raising `AttributeError`.
- Discrete-mode `dt` check: a configured `numerics.file_dt` replaced the
  decoded record axis in the comparison, so calendar-month stamps with
  `file_dt: 2592000` (or a `file_dt` contradicting an hourly axis) passed and
  the forcing ran off the particle clock. The decoded axis now decides; the
  configured `file_dt` counts only for a file without a usable time axis (or
  with all stamps equal; one repeated stamp is irregular and refused), also
  for direct callers of `run_single_realization`.
- `start_date` with `time_averaging: mean` and temporal interpolation: an
  exact stamp started at its record's window centre — half a record before
  the requested date — while a date one second later started at the date.
  With temporal interpolation the run now starts at the requested instant,
  exact stamp or not (the last stamp of averaged records, past the last window
  centre, keeps its record start with a warning); discrete mode still starts
  the record, and `instant` axes are unchanged.
- Zarr v3 stores without a `.zarr` suffix (they carry `zarr.json`, not
  `.zattrs`) were opened with the NetCDF engine; they are now recognised.
- Zarr trajectory output now carries the CF `flag_values`/`flag_meanings` of
  `status` and `failure_reason`, as the NetCDF output always did.
- The Milstein variable-diffusivity warning was keyed by the diffusivity
  arrays' identity and re-issued at every step of a temporal-interpolation run;
  it is now issued once per realization.
- Seed generation (`generate_grid`, `generate_around_point`) rebuilt the whole
  grid hash on every call (it tested the lazily built KD-tree); it now keeps an
  existing hash, including a pinned or rule-chosen cell.
- The NEMO reader's 3-D refusal now names the key that decides
  (`mode.Fdrift3D: false`) instead of the inert `mode.Fdrift2D`, and a
  `mode.sigma_layer` other than 0 is refused: the v1 reader always reads the
  surface level and ignored the setting silently.
- A sharded run with stochastic diffusion or stochastic beaching now equals
  the unsharded run bit for bit. `utils/split_seeds.py` writes a
  `# kelp particle_offset N` header into each shard file, and the engine keys
  every random draw by the particle's global index (new key
  `data.particle_offset`, keyword `particle_offset=` of
  `run_single_realization` and `track_particles`, output attribute
  `particle_offset`). Use the same seed in every shard. The merge tool
  refuses shards whose offsets do not follow the merge order unless they
  declare `random_draws = 0` (then it warns); `split_seeds.py` refuses to
  overwrite existing shard files (its cleanup removes only the names it
  writes, never the input, and never writes through a link); shards that are
  independent by design merge with `--independent-shards`, which refuses
  shards sharing any one seed over the same particle indices and records each
  shard's count, offset and seeds; stochastic shards without any offset
  (written before this change) merge with a warning; a comment that mentions `particle_offset` but is not
  the exact header is refused instead of read as 0. `write_restart_file(particle_offset=)` stores
  the offset and `load_restart` returns it for chain drivers; re-splitting a
  shard keeps its offset, and sorting a shard file is refused. An unknown
  reader name now lists every known reader.
- When every particle of a run (or shard) had died, the step returned the
  last velocities, so `vel_*` kept stale values where a run with live
  particles elsewhere recorded 0; the all-dead step now reports 0.
- The legacy `time.end_time_idx` meant `end − start` steps in `run_tracking`
  but `end − start + 1` in `run_single_realization` and `track_particles`; it
  now means `end − start` everywhere.
- `docs/check_doc_sync.py` ran its drift query in the current directory and
  reported "all docs in sync" whenever it was started outside the repository
  root; it now queries the repository (`--root`), and `--bless` stamps that
  repository's HEAD.

### Changed

- Grid-hash cell: the automatic cell is now chosen by measurement
  (`mesh.grid_hash_cell_rule: occupancy`, the default) — a ladder of
  candidate cells is scored on sampled element centres (window occupancy,
  empty windows, widening share, scan cost) and the cheapest safe cell is
  taken — under the exact guard among the candidates with ≤ 20 % widening
  and ≤ 128 MB of hash offsets, and only when it predicts ≥ 2.75× on the
  scan (quasi-uniform meshes — HEALPix, CORE2 — keep the legacy cell;
  measured: the FESOM-C North Sea mesh 0.058° → 0.014°, search 1.7×; the
  11.5 M-node global mesh 0.145° → 0.051°, search 1.7–2.6×; a 100:1 estuary
  0.05° → 0.0016°, ×26 on the search); `legacy` keeps the pre-2026-09-22
  mean-spacing cell bit for bit.
  The resolved cell and the rule are written to the output metadata
  (`grid_hash_cell`, `grid_hash_cell_rule`); `kelp run --verbose` and
  `kelp info` print the choice. Runs are exact under `exact_nearest` at
  any cell; only exact ties (scan order), anchors beyond the far-rows budget
  (a query outside every element) and `exact_nearest: false` runs can differ
  from the legacy cell — pin `grid_hash_cell_size` from an old run's metadata
  to repeat it, and add `grid_hash_cell_rule: legacy` to `exact_nearest:
  false` when reproducing a pre-guard run.
- Nearest-node search: node coordinates are converted to unit-sphere
  Cartesian once when the grid hash is built (`Mesh._node_xyz`) instead of
  per candidate per query; anchors and trajectories are bit-identical, the
  scan is ~6× faster (CORE2, automatic cell). Wipe `src/**/__pycache__`
  after updating (inlined kernels).

## [0.9.0] - 2026-09-19

The first public version: the whole tracker, published so that it can be read,
used and cited. Version 1.0.0 will follow and will be the one archived with a
persistent identifier; that identifier is minted only when a release is
archived, so no version before it can carry one.

### Features
- 2-D surface tracking on FESOM2 and FESOM-C meshes and on HEALPix grids;
  Euler, RK2, RK4 and adaptive RK45 integrators.
- 3-D tracking on z-levels (FESOM2, HEALPix) and sigma coordinates (FESOM-C);
  offline reconstruction of the vertical velocity from u, v and ssh.
- Temporal velocity interpolation (linear, 4-snapshot monotone cubic) and
  rotated-frame (pole-safe) tracking with geographic output.
- Stochastic diffusion (uniform, Euler–Maruyama, Milstein; vertical Az kick with
  reflective boundaries), beaching and resuspension, delayed and pulsed release,
  scalar-field sampling along trajectories, per-particle failure diagnostics.
- Streaming NetCDF and Zarr output readable mid-run, restart checkpoints and the
  resumable-chain pattern for runs longer than one compute job.
- Numba kernels: the tracking loop runs 15–19× faster than the Fortran reference.
- Agent skills for run set-up, configuration, experiments, analysis,
  visualization, diagnosis and benchmarking (`.claude/skills/`).
- An experimental structured C-grid reader that maps a logically rectangular
  grid onto a quad mesh. Two-dimensional, limited to grids whose axes line up
  with east and north, and it refuses rather than guesses outside those limits.
