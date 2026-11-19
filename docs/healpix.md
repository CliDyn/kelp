# HEALPix support (dual mesh + native reader)

> **Covers:** how a HEALPix product (nextGEMS / EERIE / DestinE) becomes a
> trackable KELP `Mesh`, and how the native reader feeds it — store
> carriers, the two-store ng5 recipe, config keys, polar limitations.
> **Does not cover:** the interpolation and time-stepping math the mesh feeds
> (see `velocity_interpolation.md`, `time_integration.md`), the grid-hash search
> (see `localization.md`) or the run configuration in general (see
> `configuration.md`).
>
> **Code:** `src/kelp/readers/healpix.py:build_healpix_dual_mesh`,
> `src/kelp/readers/healpix.py:HealpixReader`,
> `src/kelp/readers/healpix.py:_open_healpix_store`.
>
> **History:** phases 1–5 (dual mesh, reader, extracts, 3-D, validation)
> shipped 2026-07/08.
> **Verified at:** see `docs/doc_map.yaml` (stamped via check_doc_sync --bless).

## 1. The dual mesh (Phase 1)

HEALPix provides ONE data location — pixel centres on iso-latitude rings,
12·nside² equal-area curvilinear-quad pixels. `build_healpix_dual_mesh`
maps that onto the node/element structure the tracker uses:

- **nodes = wet pixel centres** → u/v/eta/scalars are node-based natively
  (`uv_layout="node"`, so a run REQUIRES `numerics.uv_interp: barycentric`);
- **elements = dual cells of the pixel-corner points**: the polygon of the
  (3 or 4) pixel centres incident to one corner. Corner grouping is
  **integer-exact** from `healpy.get_all_neighbours` slots — edge-sharing
  neighbours are the diagonal slots SW/NW/NE/SE of the rhombic pixel, the
  compass slots W/N/E/S touch only at a corner and are `-1` exactly at the
  8 valence-3 special vertices (→ triangle). Every incident wet pixel
  generates the same cell; the copy owned by the smallest member id is
  kept, and elements are sorted by (owner, corner) so the mesh is
  byte-identical for any `block_size`.
- A corner cell is kept only when ALL incident pixels are wet (dry pixel →
  coastline, the `structured.py` rule); wet pixels in no element (orphans)
  are dropped.

All-wet-globe laws (pinned by `tests/test_healpix_mesh.py`, self-checked at
build): exactly **8 triangles** at any nside, `E = npix + 2`, node valence
4, zero boundary edges, Euler characteristic 2, spherical-excess areas
summing to 4πR². Build cost: nside 512 ≈ 12 s / 3.6 GB peak; nside 1024 ≈
48 s / 12.8 GB (element-centre + convexity passes dominate). Zoom 11
runs whole-globe on a 256 GB node (~146 GiB per process, the 2026-09-24
scaling legs, `performance.md` §2); `mesh.crop_bbox` is the memory lever
below that.

`_healpix_node_pix` (int64 NEST pixel id per node) + `_healpix_nside` are
stashed on the mesh; the reader compacts store arrays to nodes with
`_healpix_data_idx` (the same ids in the STORE's ordering — ring stores are
handled by one index map computed at load).

### 1b. The pixel mesh (`mesh.healpix_geometry: "pixel"` — v1 2-D)

The alternative mapping inverts the roles: **elements = the wet pixels
themselves** (always 4-vertex quads), **nodes = deduplicated pixel
corners**. Pixel-centre data (u/v) then rides the **element layout**, so the
fused least-squares kernels and `edge_leastsq` apply — no `barycentric`
requirement. Corner identity is integer-exact (the sorted set of incident
pixel ids from `get_all_neighbours`; valence-3 corners carry one `-1`
identically from every owner); coordinates come from `healpy.boundaries`
with an empirically derived slot permutation that is asserted uniform
across pixels and cross-checked between incident owners. Unlike the dual
mesh, **coastal corners survive** (a corner adjacent to land is still a
vertex of its wet pixel), so the tracked domain extends to the physical
pixel edge. The flip side (2026-08 review): a corner with **< 3 incident
wet pixels** (isolated cells, 1–2-pixel channels — ~3 % of z7 corner
nodes) gives the default `leastsq` a rank-deficient stencil; since the
scale-relative singularity guard those anchors fail CLEANLY (previously
they could return non-physical velocities), so under plain `leastsq`
particles die there — **use `edge_leastsq` for coastal-heavy domains**
(mirror ghosts keep the fit well-posed; exact on the analytic tests).
`eta` and `depth` are aggregated centre→corner (mean over incident wet
pixels).

Two geometric facts to know (pinned by `tests/test_healpix_pixel_mesh.py`):
HEALPix pixel boundaries are **not great circles**, so the corner quad
approximates the true pixel — near an edge the containing quad can
disagree with `healpy.ang2pix` by one **adjacent** pixel. Quad areas sit
within ~3 % of `4π/npix` at nside 8 and plateau at ~1.1 % for nside ≥ 32
(a fixed shape mismatch, not O(h²) vanishing); the 8 polar-cap quads are
a constant −10 % at every nside; and the 8 pole-touching pixels are chart-degenerate in
the geographic frame (kept, listed in
`metadata["healpix_polar_degenerate"]`, warned once — `mesh.rotation`
resolves them like every other polar pathology; the rotation path computes
tangent factors for the **element** carrier).

**v1 scope:** 2-D only; `sample_vars`, spatial Kh (em/m1) and 3-D fail
fast with actionable messages (the corner-node aggregation those paths
need is staged). Method comparison vs the dual mapping (2026-08-21, analytic
solid-body + CORE2-twin z7 mini + FULL-YEAR z7 legs, post det-guard
fix, conditioned on the common survivor set to remove composition
bias): `pixel+leastsq` is statistically equal to `dual+barycentric` in
accuracy against the native truth (year medians 5.21 vs 4.94 km, +5 %)
at ~1.8× the speed (fused PCHIP kernels apply on the element layout:
182 vs 326 ms/step at 160k particles) and better coastal coverage
(85.3 % vs 82.6 % alive after a year). `pixel+edge_leastsq` keeps the
MOST particles alive (86.3 %) but pays a real accuracy price on rough
real fields — year median 8.37 km, ~70 % above the other two — despite
being the most accurate on the smooth analytic flow (40 m vs 46 m over
2 days); prefer it only where coastal coverage outweighs trajectory
accuracy. Dual-vs-pixel divergence (0.9 km median over the year) stays
well below the remap-vs-native error.

## 2. The native reader (Phases 2+4 — 2-D and 3-D, direct whole-store mode)

`@register("healpix")`. The mesh is derived FROM the data store
(`mesh.meshpath` points at the store carrying u/v): nside/ordering from the
`crs` attrs (`mesh.healpix_zoom` fallback), wet mask = finite u∧v at t=0
**on the tracked level** (`run_tracking` threads `mode_cfg` into
`load_mesh` — signature-inspected, other readers keep the plain call — so
`mode.var_u`/`var_v` overrides and the `mode.sigma_layer` pick shape the
mask the run actually tracks on; direct `load_mesh(mesh_cfg)` callers get
alias resolution + surface). Land is assumed **NaN-encoded** (ng5/CMIP
convention); a `deptho`-like variable additionally refines the mask
(`wet &= deptho > 0` — covers sentinel-filled land) and provides
bathymetry, else deepest-finite-level × `mesh.levels`, else zeros.
Optional `mesh.crop_bbox [W,E,S,N]` pixel-centre crop (W > E = across the
dateline; a span ≥ 360° = a pure latitude band).

**Store carriers** (`_open_healpix_store`): plain Zarr, NetCDF, and
kerchunk reference stores (`*.parq`/`*.parquet`/`*.json`) opened via
fsspec + `engine="zarr"` with dask; GRIB-chunked products (ng5) need
`gribscan` (codec registered by a lazy import). A process-wide fsspec
guard maps null reference rows (absent chunks, e.g. the all-empty deepest
u/v level of ng5) to `FileNotFoundError` → zarr fill semantics
(`_install_reference_na_shim`; fsspec/pandas-3 bug, drop when fixed
upstream).

**Two-store recipe (ng5):** the 2-D store has no ocean u/v, so
`data.ncinput` = the 3-D store (u/v at the surface level; `mode.sigma_layer`
picks another) and ssh comes from `data.ncinput_eta` (auto-derived
`3D`→`2D` filename sibling when unset). Time axes must match exactly.
Variable names: `mode.var_*` overrides first (set-but-absent → `KeyError`,
never a silent fallback), then aliases `avg_uoe`/`uo`/`u` etc.

**Where the stamps sit.** A HEALPix store of daily or monthly means stamps
each record somewhere in its averaging window, and where depends on the tool
that wrote the store, not on the model. A store written from GRIB by a
polytope or earthkit download carries the GRIB reference time, which is the
START of the averaging window: the mean over day D is stamped D 00:00, a
monthly mean the first day of its month at 00:00. A gribscan-indexed store of
the same GRIB messages carries the validity time, the END of the window: the
mean over day D is stamped D+1 00:00. When unsure, read one GRIB message of
the source: its `stepRange` gives the averaging interval (`0-24` for a daily
mean) and its validity date and time give the end of that interval; compare
them with the stamp the store shows for the same record. The settings:

- stamps at the window end (gribscan): `data.time_averaging: mean`, as before
  (also the guess when the key is left out);
- stamps at the window start, daily means: `data.time_stamp: start` with
  `data.time_window: calendar_day`, and no `time_averaging` (refused next to
  them); each record then sits at 12:00 of its day, so a discrete-mode
  `time.start_date` names 12:00;
- stamps at the window start, monthly means: `data.time_stamp: start` with
  `data.time_window: calendar_month` and `numerics.temporal_interp: true`
  (months differ in length); with `time.on_exhausted: cycle` such records
  repeat after the sum of the months, 365 d for a common year
  (`io_and_time.md` §3).

A wrong choice does not fail: under `mean`, daily means stamped at the window
start are placed 24 h early and monthly means a month early (a January mean in
mid-December), and the statistical validation of §5 cannot see a uniform
offset of the time axis. With the two-store recipe the options apply to the
joined dataset, and a refusal names both stores. `kelp info` prints the first
and last resolved instant; compare them with a record whose window you know.
The options are described in `configuration.md` §5 and `io_and_time.md` §3.

**3-D** (`mode.Fdrift3D`, Phase 4): the vertical conventions were
verified on the real ng5 store — u/v sit on the **K−1 layer mids** (the
K-th GRIB level is all-NaN padding), `avg_wo` sits **ON the K
interfaces** (K data levels, surface interface w ≈ 0, finite-prefix
columns), positive **up** like the FESOM2 `w` it derives from (ng5
`avg_wo` is "Time-mean upward sea water velocity"), negated by the reader
into the tracker's positive-down depth rate (`docs/vertical.md` §3). GRIB level axes carry only indices ⇒ 3-D **requires
`mesh.levels`** (aux3d path or inline metres) unless the store carries
its own `level_interfaces_m` table (an extract written with a table
does); a store's own table wins, and a `mesh.levels` that differs from it
is refused at load. `mode.max_levels`
trims both the mesh interfaces and the per-step read. Below-bottom (and
family-mask-mismatch) NaNs in u/v/w are **zeroed** (no flow in the
ground — the ng5 u- and w-family bottoms disagree per cell, separate
remaps); the standard no-flow-at-ground boundary treatment: a near-topography
particle's horizontal stencil can include a shallower neighbour's zero
at its layer (only W interpolation is bottom-aware by design; the
dilution is bounded-tested). An interior NaN *above* finite data is NOT
bathymetry — the finite-prefix shape is validated once per variable and
violations fail loud. `mode.max_levels` keeps `mesh.depth` from the FULL
bathymetry while truncating the interfaces (particles below the last
kept interface ride the deepest kept layer — the same semantics as
FESOM2 `max_levels`); sampled 3-D scalars are **forward-filled** down
each column so the sampler's bottom clamp reads bottom-water properties,
never NaN.
T/S sampling: `mode.sample_vars: [avg_thetao, avg_so]` — **thetao is in
Kelvin** (ng5 GRIB convention), written through unchanged.

**The vertical axis of each variable** (2026-11). The reader finds the
vertical dim of every variable on its own: a dim whose lower-cased name is
`level`, `lev`, `depth`, `deptht`, `z` or `height`, or starts with
`level`; a variable with two such dims is refused. w may therefore sit on
its own axis — w on its own `level_w` axis next to the layer variables on
`level` — as well as on the axis of u and v. In a 3-D run every vertical
axis is checked **strictly against the FULL interface table** (K entries:
the store's `level_interfaces_m`, else `mesh.levels`, taken before any
`mode.max_levels` cut, which keeps cutting the interfaces and the
per-step read as before):

| variables | accepted entries on the vertical axis |
|---|---|
| w (`mode.var_w`, else the w aliases) | exactly K (the interfaces) |
| u, v and every scalar read in 3-D (`mode.sample_vars`, `beaching.dry_mask_var`) | K−1 (the layer mids), or K with the K-th level empty at every tracked node column |

Any non-NaN value counts as data, `0.0` and `inf` included; the tracked
node columns are the store cells the mesh carries, so stray values on
untracked land cells never refuse a run. Layouts that pass: layer
variables on `lev` (K−1) with w on `level` (K); w on its own `level_w`
axis (K) next to the layer variables on `level` (K−1); a padded store
with u, v and w on one `level` axis of K entries whose last level is
empty for u and v. Anything else is refused with the numbers. Before
2026-11 the reader only checked that an axis was long enough and read
its first entries, so a table one interface short (the layer mid-depths
given instead of the interface depths) paired each w level with the
depth of another interface and dropped the deepest layer of u and v
without a message. Velocities on the interfaces (u and v with data in
their K-th level) are not supported: the rule refuses them.

**Fail-fasts:** the pixel geometry (§1b) refuses 3-D, `sample_vars` and
`em`/`m1` spatial Kh (`NotImplementedError`; the dual mesh supports all
three), `ncinput_v`/`ncinput_w` → `ValueError` (u/v/w always live in ONE
healpix store), zoom mismatch between mesh and data store → `ValueError`,
3-D without `mesh.levels` → `ValueError`, missing w in 3-D → `KeyError`,
more than 5 % of the wet nodes with non-finite u/v at a record →
`ValueError` (the wrong store, variable or level, not mask drift).
`mesh.rotation` is supported (§3).
The vertical axes of a 3-D run (2026-11): w not on exactly K entries
(checked first, so a table one interface short is reported for w), a
layer variable on neither K−1 nor K entries, a layer variable on K
entries with data in its last level, a variable with two vertical dims,
and — for u, v and w — a missing variable (`KeyError`), no vertical dim
or a cell count other than the mesh's store → `ValueError`; the messages
name the variable, its axis, the lengths and the table's source. Where
they are refused:

- u, v and w at the dataset's first record, **before any output file
  exists and in `kelp info`**: the reader's check-only `prepare` hook,
  which the engine calls with the run's mode (`run_tracking` right after
  the open, each realization at its entry, `kelp info` once), so `kelp
  info` refuses the same u/v/w configurations as `kelp run`, with the same
  messages (before 2026-11 a missing w or an axis without a vertical dim
  stopped the run at step 0, after the output file was created);
- sampled scalars and the beaching dry mask in the run's sampling and
  dry-mask probes, at the start record, also before any output exists
  (`kelp info` reads no scalars);
- every later record at its read: a K-entry layer variable whose last
  level fills only in a later record stops the run at that record, after
  the output exists (with masks static by product convention, a store
  defect). A prefetch look-ahead record that fails is not cached and is
  refused only if the run reaches it; the message names it by its time
  stamp (the index counts from the window's start).

Extracts cut by the extractor before version 1.1 are the one exception to
the empty-last-level rule (§4, "Extracts written by the earlier
extractor").

**NaN handling:** masks are static by product convention; a NaN at a wet
node in a later timestep (mask drift vs the t=0 mask) is zeroed with a
one-time warning, up to 5 % of the wet nodes (above that the read fails,
see the fail-fasts). Eta land masks differ from the u/v family on ~2% of
coastal cells (ng5) — eta NaNs at wet nodes are zeroed silently.

**Depth-linear u/v** (`mode.uv_vertical: linear`, 2026-10; the rule in
`vertical.md` §2.1). The zeroing above is the layer path's no-flow-at-ground
treatment; a depth-linear read instead reads every wet vertex in its own
column, down to the column's last usable point (a vertex whose bed lies above
the particle is a wall, left out of the sum), so it must know where each
column's data end and must never blend a zero the reader wrote into a
profile. Two mechanisms see to that:

- *The declaration.* A 3-D load of a depth-linear run on the dual geometry
  (`load_mesh` with a `mode_cfg` that has `Fdrift3D` and `uv_vertical:
  linear`) reads the leading finite count of `u` down every store column at
  t=0 and declares it per node:
  `mesh.metadata["uv_data_last"]` (int32, mesh-node order through the store
  index the data compaction uses, `_healpix_data_idx`; the count minus one,
  so `-1` = no u in the column — never a mesh node, the wet mask drops it).
  It caps each node's last usable point, so a bathymetry (`deptho`) deeper
  than the u data never points the table at a zeroed slot (u- and w-family
  bottoms disagree per cell on ng5). A 2-D or a layer load, or one without
  `mode_cfg`, skips this extra pass over the first record and declares
  nothing; a run with the option on such a mesh is refused at its first
  record ("load the mesh with the run's mode config": `run_tracking` and the
  CLI do). The declaration comes from the first record
  of the MESH store (`mesh.meshpath`), not of `data.ncinput`: a data store
  whose u prefix is shorter is refused by the raw-record check below, one
  whose prefix is longer is capped to the mesh store's prefix — point
  `mesh.meshpath` at a store of the run's own family.
- *The raw-record check.* With the option on, `read_velocities` checks each
  RAW record against the table (`check_depth_linear_record`) before the u/v
  rotation mix (§3: after the mix a NaN in one component would be NaN in
  both, and the refusal would name the wrong one) and before the zeroing: a
  non-finite u or v inside a column's usable prefix refuses the run, naming
  component, node, point and the record (index in the dataset read — a
  prefetch window counts from its own start — and time stamp). Surface-mask
  drift is therefore refused, not zeroed (the CORE2 twin has none). With the
  option off nothing changes.

## 3. Polar tracking — the rotated frame (shipped 2026-08-16) and residual limits

> **Measured cost of skipping rotation (2026-08-22, E2 diagnostics):** on
> global upsampled stress meshes, 50k-particle 8-step probes stepped in
> ~28 ms rotated but **27 s (zoom 10) / 133 s (zoom 11) per step
> unrotated** — the geographic-frame anchor degradation below is not a
> corner case at scale; treat `mesh.rotation` as mandatory for global 3-D
> HEALPix runs.

1. **Geometric containment cap** — the mesh covers the poles, but the
   lon/lat containment chart rejects points poleward of the first pixel
   ring, `arcsin(1 − 1/(3·nside²))`: 84.15° at nside 8, 89.91° at nside
   512. Fail-safe (OUTSIDE), never a wrong position.
2. **Cold-anchor search degradation** — the grid hash bins by raw lon/lat,
   so from-scratch nearest-node queries near the poles can anchor far from
   the true nearest (measured: ~6% false-OUTSIDE at |lat| 85° for random
   points at nside 512). With `mesh.exact_nearest: false`, empty-window
   misses fall back to the KD-tree inside `find_nearest_nodes`; under the
   default guard the batch search is exact (see localization.md §3/§4). Seeding and
   global re-searches only — per-step tracking updates anchors
   topologically.

For the same polar-sparsity reason a GEOGRAPHIC-frame HEALPix mesh
**fails the grid-hash density certificate**
(`grid_hash_dense_for_elements`). That matters only with
`mesh.exact_nearest: false`: the fused node-layout barycentric RK2/RK4
kernels then stay OFF (one-time `RuntimeWarning`; tracking uses the unfused
exact-anchor path — see `time_integration.md` §5). Under the default guard
the fused kernels search exactly in-kernel and run on every HEALPix mesh.

**`mesh.rotation` is now supported on HEALPix** (2026-08-16), with the
explicit caution that this is not FESOM's native grid.
Unlike fesom2 — whose model built its mesh in the rotated frame — the
HEALPix remap carries geographic east/north on a global pixel grid, so
rotation here is a pure runtime transform: node coordinates rotate AS
POINTS, velocities get exact per-node tangent factors
(`rotate_vectors`), the search structures are rebuilt on the rotated
chart, and the generic tracking layer does the rest (geographic seeds
in, geographic output, frame-validated restarts). Topology and pixel
ids never change; the rotated field read-back is exact to 1e-12
(pinned), and the transpolar great-circle test reproduces the fesom2
machine-precision result on a HEALPix mesh.

**The caution, engineered in**: the rotated frame has its own two
poles, and the polar pathologies move THERE. The loader therefore
guards the dangerous configurations — a wet element straddling a
rotated pole is a hard `ValueError` (its containment chart would lie);
wet nodes above 88° rotated latitude warn once. With the `"fesom2"`
preset the poles land deep in Greenland (−40, 75) and East Antarctica
(140, −75): measured on the CORE2 twin, wet nodes stay below |rotated
lat| 85.3° — silent, safe. An all-wet synthetic globe is deliberately
rejected; pick angles whose poles fall on land.

Side effect worth knowing: on the rotated CORE2-z6 mesh the density
certificate PASSES (Arctic pixels are ordinary-sized in the rotated
chart, the sparse polar rows sit over dry land), so with
`mesh.exact_nearest: false` rotated HEALPix runs get the **fused
barycentric kernels** the geographic frame is denied (under the default
guard both frames run them). Arctic tracking at 85–89°N — the old cold-anchor/cap death
zone — runs alive (`tests/test_healpix_rotation.py`, real-store leg).

Residual limits without rotation: avoid releasing near the poles, or
crop the caps away with `mesh.crop_bbox`.

## 4. Regional extracts (Phase 3)

`utils/extract_healpix_region.py` (YAML-driven CLI, see
`utils/extract_healpix_example.yaml`) cuts a **pixel-index subset** out of
any carrier — a bbox test on pixel centres plus `region.halo_layers`
rings of topological dilation (default 1: the dual mesh strictly inside
the bbox is then IDENTICAL to the whole-store build; tracking on an
extract is bitwise-equal for particles that stay inside — pinned by
`tests/test_healpix_extract.py`). The output is one plain Zarr,
**self-describing** (`healpix_index` = sorted source NEST ids, a `crs`
var, provenance attrs) and **NEST-canonical** regardless of the source
ordering; ssh is merged in at extraction (`source.eta_path` or the
`3D`→`2D` sibling), so an extract needs no `ncinput_eta` downstream.
Source encodings are stripped before the write (kerchunk-GRIB variables
carry the gribscan read codec, which zarr rejects as a write codec).

The reader consumes extracts transparently: `healpix_index` present →
wet mask/depth scatter onto the full pixel space, node→cell compaction
via `searchsorted`, `mesh.crop_bbox` composes as a further crop.
Extracts are the memory lever for zoom ≥ 10 3-D and PCHIP (4 snapshots).

**Vertical axes** (extractor version 1.1, 2026-11). The extractor follows
the reader's convention (§2) and its precedence for the table:

- *The table* is the source's own `level_interfaces_m` when it carries
  one, else `source.levels`; a `source.levels` that differs from the
  store's table is refused. The extract carries the table (cut with the
  levels), so a 3-D run on it needs no `mesh.levels`. (Before 1.1 a
  store's own table was dropped unless `source.levels` repeated it.)
- *w* is `source.var_w` when set, else the first w alias among the
  extracted variables; every other extracted variable with a vertical dim
  is a layer variable. A `source.var_w` the store does not hold is
  refused; with an explicit `variables:` list it must be listed there,
  without one it joins the default list (u, v and the ssh found).
- *Checked before anything is cut*, whenever there is a table: w on
  exactly K entries (K = the size of the full source table), every layer
  variable on K−1 or K; a selected w without a vertical dim is refused
  too. A wrong table, or a layer variable on the interfaces, is refused
  here, before a cut could hide it.
- *`region.max_levels: M` with a table* keeps `min(M, K)` interfaces: w
  keeps that many entries, every layer variable one fewer, the table that
  many. A layer variable on K entries always loses its K-th level in the
  cut, so that level must be empty at the exported cells (halo included)
  in every exported record; otherwise the extraction is refused, naming
  the variable and the source record.
- *The renamed w axis.* When w shares its vertical dim with a layer
  variable (a padded store with u, v and w on one `level` axis), the cut
  gives them different lengths, which one dim cannot hold: w's dim is
  renamed to the first unused name of `level_w`, `level_w_1`, …, together
  with its dim coordinate; any other coordinate of w on that dim becomes
  an unused `<name>_w`, ….
- *The bed.* When the cut keeps fewer interfaces than the source has and
  the source has no depth variable, the extract carries `deptho`: the bed
  the reader computes on the whole store, from the finite levels of the
  uncut u at the first exported record and the full table. A run on the
  extract then has the source's bed, as `mode.max_levels: M` on the whole
  store does, instead of a bed stopped at the deepest kept interface. A
  source depth variable travels with the extract as before.
- *`region.max_levels: 1`* is a surface extract for 2-D runs: every
  vertical axis keeps one entry, and no table is resolved, checked or
  written, whatever table the source has (a printed note says so; a 3-D
  extract needs `max_levels` ≥ 2). No bed is computed; a source depth
  variable is carried as before. `max_levels` below 1 is refused.
- *Without a table*, `max_levels: M` keeps the first M entries of every
  vertical axis, as before. Such a depth-cut extract serves 2-D runs; in
  3-D it runs only where u's M-th level is empty at every tracked column
  (a padded store with K = M), since otherwise no table satisfies both
  the w rule and the empty last level: **a depth-cut 3-D extract needs
  the table at extraction.** An uncut extract without a table runs in 3-D
  with a matching `mesh.levels`, as before.
- `--dry-run` makes every metadata check but reads no data, and says when
  the last-level check was not run.
- *2-D runs on a re-extracted region:* with a table and `max_levels: M`
  a layer variable now has M−1 entries, so a `mode.sigma_layer: M−1` that
  worked on a table-backed extract of the earlier extractor is out of
  range (the largest index is M−2).

**Extracts written by the earlier extractor.** Every extract records the
tool that wrote it in its `tool` attribute: `extract_healpix_region v1.0`
before version 1.1, `extract_healpix_region v1.1` now. The v1.0 cut kept
M entries on every vertical axis, so a layer variable of a 3-D extract
cut with `region.max_levels` holds one level more than the M interfaces
of its table bound: the real layer below the deepest kept interface. The
reader runs such an extract (a dataset with `healpix_index` whose `tool`
is exactly `extract_healpix_region v1.0`) as earlier versions did, bit
for bit: its layer variables are read as the K−1 layers the table bounds
and the last layer is not used — the same layers as `mode.max_levels: M`
on the whole store — and, when the extract carries no depth variable, the
bed is placed no deeper than the deepest table entry. A `RuntimeWarning`
says so, once per process for each store and set of variable names; it
names the variables and the unused layer. It comes from the records the
checks read anyway: record 0 of the dataset for u and v (the `prepare`
hook, so `kelp info` shows it too), the start record for a sampled scalar
or the dry mask (the probes). The per-step reads of an old extract never
read the last level, so a variable whose last level carries data only in
a later record is read as before, without a warning. An old extract whose
last level is empty at the tracked columns in those records reads as a
padded store, and an uncut old extract as any store, both without the
warning; every other rule (w on exactly K entries, the layer lengths, one
vertical dim) applies to old extracts as to any store. To use the unused
layer and the true bed, **extract the region again from the whole store**
(the extractor refuses an extract as its input) with the current
extractor and a `region.max_levels` of at least one more than the old
extract's table (K + 1). The same layout without the v1.0 attribute — a
raw store, or a v1.1 extract cut without a table — is refused.

## 5. Validation against the native FESOM twin (Phase 5, 2026-08-16)

The CORE2 twin dataset
(described by a dataset README kept with the stores, not in this
repository: the
native CORE2 run's 2005 output remapped to nside 64/128/256 combined
stores with daily node `uo/vo/wo/ssh/temp/salt`) closed the Phase-5
validation. Same 200 seeds, 30 daily RK2 steps, North Atlantic
(38–50°N), native FESOM2 vs the HEALPix remaps — all runs
**deterministic** (verified bit-identical across 1–64 numba threads),
numbers measured 2026-08-16 with throwaway probe scripts (not in this
repository) and frozen into
`tests/test_healpix_native_twin.py` (`@slow`, data-gated). The frozen
bounds carry ≥1.5× headroom against *reseeding the particle set* (5
independent draws measured), not just against the single realization:

| grid | nodes | median displacement | median paired separation vs native |
|---|---|---|---|
| native CORE2 | 126 858 | 187.1 km | — |
| z6 (nside 64, ~102 km) | 34 988 | 188.2 km (ratio 1.006) | 5.5 km |
| z7 (nside 128, ~51 km) | 140 325 | 187.2 km (ratio 1.000) | 1.7 km |
| z8 (nside 256, ~26 km) | 562 136 | 186.6 km (ratio 0.997) | 0.5 km |

Mean drift vectors agree to 0.004–0.010° on a ~2° drift; every particle
stays alive on every grid; the **zoom ordering** (z8 < z7 < z6
separation) is pinned — it is the assertion that catches a fault
affecting only a finer grid, which the other pins are blind to. The
suite also pins **time-axis equality** between the stores and the
native output: the statistical bounds alone would NOT catch a uniform
one-day index offset (measured: it moves the z7 separation only
1.7 → 7.1 km, inside the bound), so the axis assertion carries that
whole bug class. Fast
contract tests for the same stores (mesh counts, 2-D/3-D reads, surface
interface w ≈ 0, Celsius/psu ranges, 5-day 2-D/3-D runs, T/S sampling,
the element-sourced monthly stores) live in
`tests/test_healpix_real_core2.py` (data-gated, seconds).

**Node- vs element-sourced velocities** (the twin ships both): January
surface u/v from the element-monthly remap correlate 0.996/0.987 with
the month-averaged node-daily remap, rms ~1.1 cm/s, bias ~1e-4 — the
remap operator commutes with time averaging, so the residual is purely
FESOM's element-vs-node velocity representation (pinned in the same
suite).

**Per-step cost** (10k particles, RK2 + barycentric, daily discrete
steps incl. per-step I/O, 8 threads, a cluster login node): z6 2-D
19 ms, z7 2-D 24 ms; z6 3-D 232 ms, z7 3-D 659 ms (47-layer u/v +
48-interface w reads dominate); native 2-D 204 ms (the native daily
`unod` files carry 47 levels, so even a surface run pays the 3-D read).
Mesh-from-store build: z6 1.2 s, z7 ~10 s cold / 0.5 s warm.

## 5b. What still waits

**em/m1 spatial diffusion on pixels shipped (2026-08-16)**: put
`Kh_zonal`/`Kh_meridional` (or the `diffusion.var_kh_*` names) into the
SAME store on the pixel axis — the reader compacts them onto the wet
nodes exactly like u/v (`kh_layout: node`), wet-NaN Kh is zeroed with a
one-time warning, and the em/m1 kick runs through the layout-agnostic
`split_rk` composition (RK2/RK4; opt in via
`diffusion.composition: split_rk`). The `combined` mode (em/m1 + Euler)
remains impossible on healpix — its drift kernel is element-centre
least-squares: the `uv_interp != "leastsq"` guard of
`_apply_combined_em_step` fires first, with the node-layout guard right
behind it. Negative Kh is rejected (shared contract); non-finite Kh at
wet nodes is zeroed with a one-time warning — note the zero also enters
the LSQ plane fit of the surrounding 1-rings (measured ~−11% on the
neighbours plus a spurious outward ∂K drift), so a large bad fraction
means the wrong store, not tolerable drift. **Regional extracts do NOT
carry Kh by default** — list `Kh_zonal`/`Kh_meridional` explicitly
under the extractor's `variables:` for an em/m1 extract (§4). Under `mesh.rotation`, use ISOTROPIC Kh (equal
fields): an anisotropic pair is geographic-zonal/meridional by meaning
but is applied along the ROTATED axes (the declared frame behaviour,
same as fesom2 — see docs/coordinate_frame.md §3; an unequal pair under
rotation warns once at read time). The CORE2
twin stores carry no Kh variables, so the suite
(`tests/test_healpix_diffusion.py`) validates the wiring on synthetic
stores, including a temporal-interp run with time-varying Kh and an
extract round-trip.

The method comparison dual-mesh barycentric vs pixels-as-elements leastsq
is implemented (`mesh.healpix_geometry: pixel`, §1b reports it). Still deferred: the
built-mesh cache (zoom-9 direct build is 12 s; the extractor removes
the main cost driver), and multi-year concatenated stores
(one-year stores + the concatenation recipe are documented in the twin
dataset's AGENTS.md).
