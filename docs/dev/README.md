# Developer notes — index

Level 1 of the developer guide. `CLAUDE.md` (level 0) is read in every session;
this index and the notes files (level 2) are read on demand:

- **before changing code or tests of a module** — read its entry below, then every
  notes file the entry names;
- **during a run**, when a fine detail of a module matters (a default, a fail-fast,
  a metadata attribute, a known trap).

The notes files are the module notes of `CLAUDE.md` as of v1.0.0, moved verbatim.
They refer to each other: a module's contract is sometimes written down under
another module's heading. Each entry below names the module's own notes file and
the main other places. To find every mention of a module or a function, search
the notes: `grep -n '<module or function name>' docs/dev/*.md`.
The topic documentation for users is one level up, in `docs/` (entry point
[../architecture.md](../architecture.md)).

## Notes files

| file | modules |
|---|---|
| [config.md](config.md) | `core/config.py`; also the grid-hash cell rule and the exact nearest-node guard (internals of `core/mesh.py` and the fused kernels) and `core/shared_fields.py` |
| [mesh.md](mesh.md) | `core/geometry.py`, `core/mesh.py` |
| [core.md](core.md) | `core/particles.py`, `core/release.py`, `core/particles_gen.py`, `core/io.py`, `core/rotation.py` |
| [writers.md](writers.md) | `writers.py` |
| [numerics.md](numerics.md) | `numerics/interpolate.py`, `numerics/vertical.py`, `numerics/vertical_sigma.py`, `numerics/diffusion.py`, `numerics/beaching.py`, `numerics/sampling.py`, `w_reconstruct.py` |
| [integrators.md](integrators.md) | `numerics/integrators.py`, `numerics/temporal_reconstruct.py` |
| [tracking.md](tracking.md) | `tracking.py`, `cli/main.py` |
| [readers.md](readers.md) | `readers/` |
| [visualization.md](visualization.md) | `visualization/` |

## Modules

Paths are below `src/kelp/`.

### core/

- **`core/config.py`** — YAML → `AppConfig` (sections `mode`, `numerics`, `mesh`,
  `data`, `time`), validated at load; unknown keys are ignored silently. Several
  keys reach far beyond parsing: the grid-hash cell rule and the exact
  nearest-node guard are applied by `run_tracking`, so a driver that calls
  `load_mesh` directly applies them itself. Notes: [config.md](config.md).
- **`core/geometry.py`** — constants and spherical helpers. `METERS_PER_DEG_LAT =
  111120.0` (Parcels' convention, deliberately not Fortran's 111 300);
  `great_circle_distance` is the spherical law of cosines, not Vincenty. Notes:
  [mesh.md](mesh.md).
- **`core/mesh.py`** — the `Mesh` dataclass (`elem_nodes` 0-based and −1-padded,
  `depth` positive down), the grid-hash nearest-node search (KD-tree built lazily),
  containment and vertex weights, lazy caches (areas, coastal flags, neighbours),
  the edge walk. Traps: `build_node_to_elems(n_nodes, elem_nodes)` takes `n_nodes`
  first; `_grid_hash_nearest_exact` is inlined into the fused kernels of
  `integrators.py`, and Numba invalidates only on the caller's file — wipe
  `src/**/__pycache__` after editing `mesh.py`. Notes: [mesh.md](mesh.md); the cell
  rule and the exact guard in [config.md](config.md) (`MeshConfig.grid_hash_cell_rule`,
  `MeshConfig.exact_nearest`); the in-kernel exact search as the fused kernels use it
  in [integrators.md](integrators.md); the local projection, the vertex weights and
  the seam-safe projection origin as the interpolators use them in
  [numerics.md](numerics.md).
- **`core/particles.py`** — `ParticleState` (immutable: `.replace()` returns a new
  state), `ParticleStatus` (ALIVE, BEACHED, OUTSIDE, NOT_RELEASED),
  `FailureReason`. A new optional per-particle array goes into `_OPTIONAL_ARRAYS`;
  split a state with `.subset()`, never field by field. Notes: [core.md](core.md).
- **`core/particles_gen.py`** — seeding: a regular grid over the mesh, or a
  metric-scaled circle around a point; raises when no candidate lies inside the
  mesh. Notes: [core.md](core.md); the grid-hash cell pin that survives search
  rebuilds: [mesh.md](mesh.md) (the filter builds the search only when the mesh has
  none, and keeps an existing one).
- **`core/release.py`** — delayed and staggered release: particles are allocated
  up front as `NOT_RELEASED` and frozen until their release step; output masking
  of dormant particles and inactive velocities. The trivial schedule returns a
  plain state (byte-identical runs). Notes: [core.md](core.md); the pulse keys in
  [config.md](config.md) (`TimeConfig`); the seed-file header `# kelp particle_offset N`
  and its checks in [numerics.md](numerics.md) (diffusion, "Shards").
- **`core/io.py`** — `VelocityField` (u/v on elements or nodes, `w` positive down
  at interfaces, optional Kh, Az and sampled scalars) and `LazyLerpField`, which
  blends a field between two snapshots only when it is first read. Notes:
  [core.md](core.md); the cost of the eager blend it replaced: [tracking.md](tracking.md)
  (temporal-interp mode).
- **`core/rotation.py`** — Euler rotated-frame primitives (`v_rot = R @ v_geo`;
  `"fesom2"` = (50, 15, −90)); plain numpy, run once per mesh load or snapshot,
  never in particle loops. Notes: [core.md](core.md); how tracking uses the frame:
  [tracking.md](tracking.md).
- **`core/shared_fields.py`** — node-shared snapshot cache: one process reads and
  publishes each snapshot, sibling shard processes map the same bytes read-only;
  any error falls back to local reads. Notes: only in [config.md](config.md),
  bullet `NumericsConfig.shared_fields_dir`.

### numerics/

- **`numerics/interpolate.py`** — u/v interpolation registry: `leastsq` (default;
  least-squares fit centred on the particle), `barycentric`, `edge_leastsq`; Numba
  kernels with a Python fallback. Node-layout u/v need `barycentric` (the fail-fast
  is in [tracking.md](tracking.md)). Notes: [numerics.md](numerics.md); the
  time-blending kernel twins the fused integrators call: [integrators.md](integrators.md).
- **`numerics/vertical.py`** — z-level layers, depth clamping and W interpolation
  (vertical first, then `nearest`, `leastsq`, `barycentric` or `idw` across the
  neighbours, bottom-aware). Notes: [numerics.md](numerics.md); the `w_interp`
  validation against its method map: [config.md](config.md).
- **`numerics/vertical_sigma.py`** — sigma-coordinate (FESOM-C) column geometry,
  layers, and W sampled along the particle's sigma level. Notes:
  [numerics.md](numerics.md); the sigma step itself in [readers.md](readers.md)
  (the `fesomc.py` 3-D W bullet).
- **`numerics/temporal_reconstruct.py`** — 4-snapshot PCHIP (Fritsch–Carlson) blend
  in time, opt-in via `numerics.temporal_reconstruction: pchip`. Notes: inside the
  integrators section, [integrators.md](integrators.md).
- **`numerics/diffusion.py`** — stochastic diffusion: the counter-based RNG (every
  draw fixed by seed, realization, particle, step and component — independent of
  thread order and restarts), the uniform, Euler–Maruyama and Milstein kernels, Kh
  interpolation and gradients, the vertical Az kick. Trap: the particle key is the
  original state index plus the shard offset, never the alive-compacted index.
  Notes: [numerics.md](numerics.md); `DiffusionConfig` in [config.md](config.md); how
  the Kh and Az fields are read in [readers.md](readers.md) and carried in
  [core.md](core.md) (`VelocityField`); the `--seed` override in
  [tracking.md](tracking.md) (`cli/main.py`).
- **`numerics/beaching.py`** — beaching decisions (four combinable criteria,
  resuspension) in plain numpy; the status changes are applied by the hook in
  `tracking.py`. Notes: [numerics.md](numerics.md); the hook in
  [tracking.md](tracking.md); the options in [config.md](config.md); the dry mask in
  multi-year FESOM2 input in [readers.md](readers.md).
- **`numerics/sampling.py`** — scalar fields (temperature, salinity, …) sampled at
  particle positions for the output; direction-free (backward equals forward).
  Notes: [numerics.md](numerics.md); when samples are taken: [tracking.md](tracking.md).
- **`numerics/integrators.py`** — Euler, RK2, RK4, RK45: fused Numba kernels and
  unfused paths, failure handling (`OUTSIDE` + `failure_reason`), diffusion
  composition, time-aware stages. `step(...)` returns a new state. Notes:
  [integrators.md](integrators.md); also: the sigma step (integration of the
  particle's sigma fraction) in [readers.md](readers.md) (the `fesomc.py` 3-D W
  bullet); the M1 warning, the all-dead early exit and the sigma bypass of the
  fused 3-D Euler kernel in [numerics.md](numerics.md); the exact anchor search inside
  the fused kernels in [config.md](config.md) (`MeshConfig.exact_nearest`) and
  [mesh.md](mesh.md); the stage helpers, the lazy payload and the failure-reason
  merge in [core.md](core.md) (`core/io.py`, `core/particles.py`); the edge walk in
  [tracking.md](tracking.md).

### readers/

- **`readers/__init__.py`** — the registry (`get_reader("fesomc" | "fesom2" | "nemo" |
  "healpix")`) and shared helpers: engine detection, the time axis and its epoch,
  the sign rule for W in the file. Notes: [readers.md](readers.md); the W sign from
  `mode.w_positive` in [config.md](config.md); the time units the output takes from
  the axis in [tracking.md](tracking.md) and [writers.md](writers.md).
- **`readers/fesom2.py`** — FESOM2: NetCDF mesh diagnostics in several
  conventions, ASCII or Zarr meshes; single files, directories, multi-year
  `<var>.fesom.<year>.nc`; W is turned positive down by `resolve_w_sign` —
  negated by default, since FESOM2 writes it positive up, unless `mode.w_positive`
  or the file's CF metadata say otherwise (e.g. a reconstructed W, stamped
  `positive="down"`). Notes: [readers.md](readers.md); the sign precedence and how
  `mesh.rotation` is resolved in [config.md](config.md) (`ModeConfig.w_positive`,
  `MeshConfig.rotation`).
- **`readers/fesomc.py`** — FESOM-C: one NetCDF on sigma coordinates; 3-D W =
  −`Wvel` by default (the sign through `resolve_w_sign`, as for FESOM2); rejects
  `mesh.rotation`. Notes: [readers.md](readers.md).
- **`readers/healpix.py`** — HEALPix stores (Zarr, NetCDF, kerchunk): the mesh is
  built from the data (dual or pixel geometry; the dual geometry has node-layout
  u/v and needs `uv_interp: barycentric`), regional extracts, rotation, 2-D and
  3-D. Notes: [readers.md](readers.md); the pixel geometry, `crop_bbox`, `levels`,
  `healpix_zoom`, `ncinput_eta` and the rotation in [config.md](config.md)
  (`MeshConfig.rotation` and the healpix-only keys bullet); topic doc
  [../healpix.md](../healpix.md).
- **`readers/structured.py`** — maps a logically rectangular grid onto a quad
  `Mesh` (a quad of four neighbouring cell centres is kept only when all four are
  wet; the nodes are the wet cell centres that belong to at least one such quad).
  Notes: [readers.md](readers.md).
- **`readers/nemo.py`** — NEMO adapter, version 1: 2-D, grids aligned with east and
  north only (others are refused); node-layout u/v. Notes: [readers.md](readers.md).

### Top level

- **`tracking.py`** — orchestration: `run_tracking` (config → mesh → reader →
  realizations → writer), `run_single_realization`, `track_particles`; the record
  clock, the lazy temporal-interp payload, the beaching, sampling, rotation and
  edge-walk hooks, fail-fast checks before the writer exists. Notes:
  [tracking.md](tracking.md); also: periodic windows (`NumericsConfig.periodic_lon`),
  the `start_date` resolution (`TimeConfig`), the shared-fields tag
  (`NumericsConfig.shared_fields_dir`) and the grid-hash / exact-guard calls of
  `run_tracking` in [config.md](config.md); the time bracket and the PCHIP stencil in
  [integrators.md](integrators.md); the lazy payload in [core.md](core.md)
  (`core/io.py`); the `comment` the sampled variables get in [writers.md](writers.md);
  the surface elevation of the sigma step and the cache's rule for an all-NaN W
  record in [readers.md](readers.md).
- **`writers.py`** — trajectory output, format 2 (CF trajectory; NetCDF4 or Zarr by
  file extension), checkpoints and standalone restart files. A NetCDF file is
  locked while open; `numerics.output_flush_every` makes it readable mid-run. Notes:
  [writers.md](writers.md).
- **`w_reconstruct.py`** — offline W from u, v and ssh (FESOM2 z-level only),
  positive down. Notes: [numerics.md](numerics.md).
- **`cli/main.py`** — the `kelp run|info` console script; `--temporal-interp` and
  `--checkpoint-freq` default to None so the YAML values pass through. Notes:
  [tracking.md](tracking.md); what `kelp info` applies and prints (the grid-hash
  cell) in [config.md](config.md) and [mesh.md](mesh.md).

### visualization/

Matplotlib helpers, no core module depends on them. Notes (what the package
exports): [visualization.md](visualization.md). Per file (module contents, not all
of them exported):

- **`visualization/core.py`** — `TrajectoryDataset` (load and access trajectory
  files), `display_coords`. Notes: [visualization.md](visualization.md).
- **`visualization/plot_2d.py`** — `plot_mesh`, `plot_mesh_nodes`, `plot_field`,
  `plot_trajectories`, `plot_diagnostics`. Notes: [visualization.md](visualization.md).
- **`visualization/plot_3d.py`** — `plot_mesh_3d`, `plot_trajectories_3d`
  (pyvista/vtk when available, else matplotlib), `build_pyvista_mesh`. Notes: [visualization.md](visualization.md).
- **`visualization/animation.py`** — `TrajectoryAnimator`. Notes: [visualization.md](visualization.md).
- **`visualization/stats.py`** — `compute_presence_probability`,
  `compute_presence_density_mesh`, `compute_status_ratios`. Notes: [visualization.md](visualization.md).

## Dependency graph (imports)

Generated from the import statements (2026-10-04); "lazy" = imported inside a
function. Third-party: numpy everywhere, numba optional in the kernels, xarray
in the readers, netCDF4/zarr in the writer.

```
core/config.py            ← standalone (yaml)
core/geometry.py          ← standalone
core/io.py                ← standalone (VelocityField, LazyLerpField)
core/rotation.py          ← standalone
core/shared_fields.py     ← standalone (mmap/flock snapshot cache)
core/particles.py         ← standalone (lazy: particles_gen)
core/mesh.py              ← geometry
core/particles_gen.py     ← geometry, mesh, particles
core/release.py           ← particles
writers.py                ← particles (netCDF4 / zarr)
w_reconstruct.py          ← geometry
numerics/temporal_reconstruct.py ← io
numerics/interpolate.py   ← geometry, io, mesh, particles, temporal_reconstruct
numerics/vertical.py      ← mesh, temporal_reconstruct
numerics/vertical_sigma.py ← mesh
numerics/diffusion.py     ← geometry, mesh, interpolate
numerics/beaching.py      ← particles, diffusion
numerics/sampling.py      ← mesh, vertical, vertical_sigma
numerics/integrators.py   ← geometry, mesh, particles, diffusion, interpolate, vertical
                            (lazy: io, temporal_reconstruct, vertical_sigma)
readers/__init__.py       ← standalone (registry + shared helpers)
readers/fesom2.py, fesomc.py ← config, io, mesh, readers helpers (fesom2 lazy: rotation)
readers/healpix.py        ← geometry, mesh, readers helpers (lazy: io, rotation, Kh name helpers)
readers/structured.py     ← mesh
readers/nemo.py           ← io, mesh, readers helpers, structured
tracking.py               ← config, geometry, io, mesh, particles, release, diffusion,
                            integrators, readers (lazy: writers, shared_fields, beaching,
                            sampling, temporal_reconstruct, rotation, readers.fesom2, __version__)
visualization/*           ← core.geometry, core.mesh, core.particles (lazy: rotation)
cli/main.py               ← __version__ (lazy per subcommand: config, mesh, readers, tracking)
```

`interpolate.py` re-exports `VelocityField` from `core.io` (some tests import it
from there); the readers import it from `core.io`.
