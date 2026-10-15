# Data layout, big meshes and I/O knobs

Read when the mesh is large, the region small, the output is multi-year FESOM2, snapshot reads dominate,
or time accuracy on a coarse file cadence matters.

**Large global mesh, small region of interest?** The tracker runs fine
directly on the full mesh (the search/interpolation handle tens of millions
of nodes), so subsetting is optional. But if the task only needs a region,
you can offer to cut a regional subset to Zarr first with
`utils/extract_fesom_region.py` (bounding box / circle / polygon) and point
`mesh.meshpath` + `data.ncinput` at the subset — it makes the mesh load and
the per-step I/O lighter (and on very large 3D meshes it is what makes the
4-snapshot reconstruction below fit in RAM).

**Big-mesh I/O knobs** (`numerics`, both default off → byte-identical; the
I/O lever when reading a snapshot dwarfs a step): `velocity_float32: true`
keeps `u`/`v`/`w` float32 (halves snapshot memory/read; a MEMORY knob — the
step itself is 15–20 % slower), and
`prefetch_size: N` bulk-reads N snapshots per disk read instead of
one-per-step. Pair them (float32 halves the prefetch window's RAM); size N so
~N snapshots fit in memory (global 3D u/v/w float32 ≈ 0.28 GB/snapshot →
N≈8 ≈ 2.2 GB). With `temporal_reconstruction: pchip` set `prefetch_size >= 4` —
which is why pchip and the shared snapshot cache (`shared_fields_dir`, forces
`prefetch_size: 1`) exclude each other, and pchip also leaves the fused 3-D
RK2 kernel: on a kerchunk/HEALPix multi-process campaign take the cache and
buy time accuracy with `dt = file_dt/4` instead.

**Multi-year FESOM2 output** (one file per variable per year,
`<var>.fesom.<year>.nc`) is concatenated automatically across all years when
`data.ncinput` is the directory — point `mode.var_u`/`var_v`/`var_w`/`var_eta`
and `sample_vars` at the *exact* variable names (e.g. node velocities
`unod`/`vnod`/`w`). Keep the chosen family's cadence consistent (mixing
monthly element `u`/`v` with daily node `unod` fails fast). Needs `dask`. A
ready 3D template is `examples/config_weddell_3d.yaml` — copy it,
set the paths, the start record (not the cold-start record 0) and `checkpoint_freq`,
and keep its `w_interp: barycentric` (the fused 3-D RK2 kernel needs nearest or
barycentric W).

**Time accuracy on coarse file cadence:** with `temporal_interp: true` you may
also set `numerics.temporal_reconstruction: pchip` — a 4-snapshot monotone-
cubic time blend so RK4 separates from RK2 instead of collapsing to the linear
`O(file_dt²)` floor. Default `linear` is byte-identical; `pchip` uses the
unfused RK path and holds 4 snapshots in memory.
