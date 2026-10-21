# utils

Standalone tools, kept out of the `kelp` package. The three data tools are
driven by their own YAML config (separate from the run config: copy the matching
`*.example.yaml`, edit the paths, and run); the others take command-line
arguments (`--help`).

| Tool | Driven by | What it does |
|------|-----------|--------------|
| `extract_fesom_region.py` | YAML (`--config`) | Cut a region out of a large FESOM2 mesh + variable files → a smaller NetCDF/Zarr subset that the FESOM2 reader loads directly. |
| `extract_healpix_region.py` | YAML (positional) | Cut a pixel subset out of a HEALPix store → one self-describing Zarr the HEALPix reader consumes transparently (docs/healpix.md §4). |
| `reconstruct_w.py` | YAML (`--config`) | Reconstruct Cartesian vertical velocity W from u, v, ssh when the model output has no W. |
| `check_grid_hash_cell.py` | a run config | Report nodes per grid-hash window and widened anchors for candidate cells, at the seeds and the element centres. |
| `sort_seeds.py` | arguments | Sort a seed file by (HEALPix NEST cell, depth) for memory locality — sort first, then shard. |
| `split_seeds.py` | arguments | Split a seed file into contiguous shards; each shard file carries `# kelp particle_offset N`. |
| `concat_trajectories.py` | arguments | Merge shard Zarr outputs in shard order; equals the unsharded run bit for bit. Reads output format 2 and format 1; refuses mixed formats (see below). |

---

## `extract_fesom_region.py` — regional subsetting

The memory lever for big meshes: extract a region once, then track on the small
subset (lighter mesh load and per-step I/O). Variables on **nodes** or
**elements** are auto-detected by matching mesh sizes; output is streamed
timestep-by-timestep (bounded memory).

```bash
cp utils/extract_fesom_region.example.yaml my_extract.yaml   # edit paths + region
python utils/extract_fesom_region.py --config my_extract.yaml
```

Config sections (full annotated example: `extract_fesom_region.example.yaml`):

- **`source`** — `mesh` (FESOM2 diag NetCDF, required) and the variables to cut.
  Velocity as `u_path`/`v_path` (+ optional `w_path`/`ssh_path`) or an `input`
  directory of `u*`/`v*`/… ; any extras under `extra_vars` (node- or
  element-based). Common name aliases (`temp`/`thetao`, `salt`/`so`) are applied.
- **`region`** — `mode: bbox` with a nested `bbox: {min_lon, max_lon, min_lat,
  max_lat}`; or `center_radius` with `center: {lon, lat}` + `radius_km`; or
  `polygon` with `polygon_path`. `halo_layers` adds boundary element rings (1–2
  avoids edge artifacts).
- **`time`** — index-based `start` / `stop` / `step`.
- **`output`** — `dir`, `format: netcdf|zarr`, `layout: split|combined` (split =
  one file per variable, recommended for KELP), optional `zarr_chunks`.
- **`metadata`** — optional provenance written into output attributes.

Point KELP at the result:

```yaml
mesh: { type: fesom2, meshpath: path/to/output_subset/fesom.mesh.diag.region.nc }
data: { ncinput: path/to/output_subset }
```

The subset mesh keeps the level variables of the source diagnostic when it has
them: `nlevels`, `ulevels` and `zbar_e_bottom` for the kept elements and
`nlevels_nod2D` for the kept nodes, values and signs as in the source. The FESOM2
reader declares each element's bottom from `nlevels` / `zbar_e_bottom`; depth-linear
u/v on element data (`mode.uv_vertical: linear`) needs them, so a subset written
before the tool kept them must be extracted again.

The individual `source`/`region`/`output` settings can also be given as CLI flags
(`--mesh`, `--bbox`, `--output-dir`, …); run with `-h` to list them.

**Limitations:** no dateline-crossing regions; polygons are simple (GeoJSON
`Polygon`); time selection is index-based only.

---

## `reconstruct_w.py` — vertical-velocity reconstruction

Reconstruct Cartesian W (positive-down) from FESOM2 horizontal velocities and
SSH when the output has no W (or only `w_ALE`). FESOM 1.4 inputs are refused before
anything is loaded (an ASCII mesh directory holding `nod3d.out`; u over the 3-D nodes
or on levels starting at the surface): their u/v lie on the levels, and the output
holds `wo`. Two methods:

- **`divergence`** — integrate continuity from the surface down (weak-form
  shape-function divergence on the triangular mesh).
- **`ale`** — heuristic `w_cart ≈ w_ALE − (1 − D/H)·ssh_t` (fast, approximate).

```bash
cp utils/reconstruct_w.example.yaml my_w_config.yaml          # edit paths
python utils/reconstruct_w.py my_w_config.yaml                # or --method divergence|ale|auto
```

Config keys (full example: `reconstruct_w.example.yaml`): `source` (mesh + u/v/ssh),
`output`, `method` (`auto`/`divergence`/`ale`), `correct_bottom`, `use_nod_area`,
`max_levels`, `time_range`.

The output (`w.reconstructed.nc` / `.zarr`) is read by the FESOM2 reader via
`data.ncinput_w`:

```yaml
data:
  ncinput:   path/to/u.nc
  ncinput_v: path/to/v.nc
  ncinput_w: path/to/w.reconstructed.nc
```

---

## `concat_trajectories.py` — merging shards

```bash
python utils/concat_trajectories.py merged.zarr shard000.zarr shard001.zarr ...
```

Merges the Zarr outputs of a sharded release along the particle axis, in shard
order (`--help` lists the options and the safety checks). It reads both
trajectory output formats (`docs/io_and_time.md` §5):

- **Format 2** (global attribute `kelp_output_format = 2`, record dimension
  `time`): the merged store keeps the record dimension `time`, the `time`
  `units` and `calendar`, and the `release_*` variables (one value per
  particle, concatenated like the particles). Float arrays keep NaN as fill
  value, and the merged `trajectory` holds the global particle index, starting
  at the first shard's `particle_offset`. A shard written without velocity
  variables because none of its particles was ever alive (before 2026-10-14 a
  run created them only at the first evaluated velocity) gets NaN velocities,
  as format 2 records them for particles that are not alive.
- **Format 1** (no `kelp_output_format`, record dimension `obs`): shards merge
  as before, including the zero back-fill of `vel_w` for a 2-D shard whose
  particles were never released.
- **Refused:** shards that differ in `kelp_output_format`, in the name of the
  record dimension (`obs` against `time`), or in the `units` or `calendar` of
  `time` — the merge compares the time values, and equal numbers on different
  epochs are not the same instants. Every other run-defining attribute must
  agree as well.
