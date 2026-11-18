# NEMO support (exact Arakawa C-grid path)

> **Covers:** tracking on native NEMO output (ORCA family, regional crops, GYRE-like boxes) with `mesh.nemo_grid: cgrid`: the scheme, the supported grid layouts, the input files, the vertical treatment, sampling, refusals, validation and known limitations.
> **Does not cover:** the option reference (see `configuration.md`, keys `nemo_grid`, `nemo_partial_cells`, `ncinput_u/v`), the interpolation formulas (see `velocity_interpolation.md` §8), NEMO file adaptation recipes (skill `kelp-nemo`), the default T-point path (`nemo_grid: tpoint`, 2-D only, E/N-aligned grids; `docs/dev/readers.md`).
> **Code:** `src/kelp/readers/nemo/reader.py:NemoReader`, `src/kelp/readers/nemo/_mesh.py:build_nemo_cgrid_mesh`, `src/kelp/readers/nemo/_layout.py:detect_nemo_cgrid_layout`, `src/kelp/numerics/nemo_cgrid.py:interp_nemo_cgrid3d`, `src/kelp/numerics/nemo_cgrid.py:nemo_cgrid3d_step`.
> **Verified at:** see `docs/doc_map.yaml` (stamped via check_doc_sync --bless).

## 1. What the path does

`mesh.nemo_grid: cgrid` tracks particles on the model's own staggered grid with
the flux scheme of Delandmeter and van Sebille (2019). Nothing is averaged
onto T-points and no vector rotation is needed:

- **Mesh.** Nodes are the F-points, elements are the wet T-cells with their four
  F-point corners. u lives on the east face and v on the north face of a T-cell.
- **Interpolation.** Inside a cell the particle moves in index space,
  `dxi/dt = ((1-xi) U0 + xi U1) / |detJ|` (and likewise for `eta`), with the
  fluxes `U = u * e2u (* e3u)`, `V = v * e1v (* e3v)` of the model's own face
  widths. The flow through a closed face is exactly zero and the discrete
  continuity of the model is reproduced (`velocity_interpolation.md` §8).
- **Land.** Land has zero velocity. The mesh carries **one ring of dry cells**
  around the wet region; a particle that enters the ring or a column below its
  floor stays there, ALIVE, with zero velocity. Beyond the ring a stage
  fails. **Particles on land are not handled in the code**: filter them post
  hoc (e.g. by the mesh's wet mask at the final position, or by a zero
  displacement over a record). Horizontal diffusion (`diffusion.scheme:
  uniform`, 2-D) does not see the closed faces: a kick into the dry ring leaves
  the particle there, ALIVE, and later kicks move it on; a kick past the ring
  is undone and the particle ends OUTSIDE (`OUT_OF_BOUNDS`). On FESOM meshes
  the coast is the mesh edge, so a kick across it ends the particle at once;
  here the first land cell is open to the kick.
- **Frames.** The Jacobian is taken in whatever frame the mesh nodes are in, so
  `mesh.rotation` (e.g. `fesom2`) works for the Arctic without any vector
  rotation. Above about 88 N the lon/lat cell geometry degenerates and the
  reader warns; use the rotation there.
- **Integrators.** Euler, RK2, RK4, steady or time-aware (linear blend of the
  snapshots, `numerics.temporal_interp`). 2-D and 3-D.

`numerics.uv_interp` follows from `nemo_grid: cgrid` (see `configuration.md`).

## 2. Supported layouts

The layout is read from the coordinates (exact identities between the T and F
points), never from a key. Every raster is brought to one canonical (halo)
form, so the seam and the fold are interior edges and particles cross them.

| layout | halo columns / row | seen with | validated on |
|---|---|---|---|
| global, F-point-pivot fold | yes | ORCA05, ORCA1 | ORCA05 (2-D and 3-D, real data) |
| global, T-point-pivot fold, halo-free (NEMO >= 4.2, `Iperio=1`) | no | eORCA025 and other ORCA2/025/12 and eORCA output | eORCA025 northern rows and southern band (real data) |
| global, T-point-pivot fold with halos (NEMO <= 4.0) | yes | ORCA2 | ORCA2 CMIP5 (IPSL-CM5A-LR): topology only (no `mesh_mask`, no W) |
| global, F-point-pivot fold, halo-free | no | | synthetic tests only |
| regional box, periodic channel, crops of any of the above | no | GYRE, AGRIF-free regional configs, extracts | GYRE-like synthetic grids; ORCA05 / eORCA025 crops |

A crop (a raster narrower or shorter than its parent file) is detected when the
file's `Iperio`/`NFold`/`NFtype` flags claim more than the raster shows. The
raster edges are then **open**: where water reaches an edge a particle leaves
the domain and is absorbed. It ends OUTSIDE with the label
`OUT_OF_BOUNDS`, or `INTERPOLATION_FAILED` when an RK stage crossed the edge
first (so count both as an exit). The layout and the open edges are printed at
load and by `kelp info`, and stored as the output attribute `nemo_cgrid_layout`.

Refused at load: a partial-width raster whose fold identities hold (a crop
symmetric about a pivot), zero-filled mesh tiles (land-processor elimination),
a pole row, a wet cell at a fold pivot, coincident nodes, a contradiction
between the flags and the raster, and any halo or fold copy of u, v (every
record), `tmask`, `e1t/e2t`, `e2u/e1v`, `e3t_0/e3u_0/e3v_0` that differs from its
canonical point (stale halos, a wrong grid file). T/S samples, bathymetry and
the `e3t` of the z*/vvl check are canonicalised silently.

## 3. Inputs

`mesh.meshpath` is a `mesh_mask.nc` or a `domain_cfg.nc`:

- always `glamf`, `gphif`, `tmask` (a `domain_cfg` without `tmask`: the mask is
  `level <= bottom_level`), the T coordinates `glamt/gphit` or
  `nav_lon/nav_lat` (layout detection), `e2u`, `e1v` (face widths; absent in 2-D: plane
  lengths, logged; required in 3-D);
- for 3-D also `e3t_1d`, `e1t`, `e2t`, and `e3t_0`, `e3u_0`, `e3v_0` for
  partial cells (absent: full reference cells; a zps vertical grid without them is refused by a 3-D run).

The model output is `data.ncinput` (the grid_T file(s)), with the grid_U /
grid_V siblings derived by swapping the `grid_T` token in the file
name. Files without that token take `ncinput_u`, `ncinput_v` explicitly
(`ncinput_w` is refused: grid_W is never read). Each of these keys takes a file, a glob (sorted; no match
refuses) or a list; several files per grid are joined along time (each file may
have its own time epoch; the joined axis must be strictly increasing; grid_U/V
must share grid_T's time axis). Velocity names `uo`/`vozocrtx`, `vo`/`vomecrty`
are found by alias, others by `mode.var_u`/`var_v`; `sample_vars` take exact
names. Record times are set as for every reader. NEMO/XIOS stamps a mean at the
CENTRE of its averaging window (`time_counter` == `time_centered`), so for means
use `time_stamp: centre` with `time_window` (`calendar_month`, `"5 d"`, ...);
`time_averaging: mean` assumes window-end stamps and would place the records
half a window early. Snapshots: `time_averaging: instant`, or `time_stamp:
instant` with the cadence as `time_window`.

Land values in the velocity files (0 or NaN) are irrelevant: the faces a
`tmask` closes are zeroed from the mask, never from the file.

## 4. Vertical (3-D, `mode.Fdrift3D: true`)

- **Levels.** z-coordinates (`VertCoord` zco/zps). Layer interfaces are the
  cumulative `e3t_1d`; the depth is positive down, in metres.
- **Fractional level coordinate.** The step runs in `s = k + zeta` (level and
  the fraction within the containing cell). A particle crossing a side face keeps
  `s`, so a crossing from a thick cell into a thinner neighbour cannot land below
  its floor. Metres are a diagnostic of `s` and jump at a face between cells of
  different bottom thickness. Seeds are converted metres to `s` with the start
  cell.
- **Partial cells (zps).** The bottom cell has the thickness `e3t_0` there;
  faces use `e3u_0`/`e3v_0` (checked against the zps min rule, else refused).
  `mesh.nemo_partial_cells` chooses the depth in metres of the fractional level in
  a partial cell: `shrunk` (NEMO <= 4.2, default; the cell spans
  `[top, top + e3t_0]`) or `porous` (NEMO 5; the full reference layer). Only metres
  differ; fluxes, W, lon and lat are identical. `porous` is tested synthetically
  only.
- **W.** Computed by the reader from continuity of the partial-cell fluxes
  `u e2u e3u`, `v e1v e3v` (integrated up from 0 at the floor); it equals the
  model's W to float32 rounding on ORCA05. A `grid_W` file is never read (`ncinput_w` is
  refused); the data-gated tests open it themselves to compare. W is linear in the
  fractional level inside a cell. The output `vel_w` is the physical dz/dt in m/s
  (the level rate times the start cell's thickness).
- **Surface.** `s` is clamped at 0 (the continuity W leaves a small surface
  residual, as the model's free surface does); never at the bottom.
- **z\*/vvl output.** If `e3t` in grid_T differs from `e3t_0` the reader warns
  with the deviation and uses the fixed `e3*_0` geometry: the field stays
  exactly non-divergent, transports are off by about eta/H. eORCA025 is such
  a case (up to 3 % in the Arctic).

## 5. Sampling

`mode.sample_vars` returns the **T-cell value** of the containing cell and
layer (2-D: the surface layer), NaN outside, on land and for dead particles.
There is no interpolation between cells. U/V/W/F-point variables are refused.
Names are exact (`thetao`, `thetao_con`, `so`, `so_abs`, `votemper`, ...).

## 6. Refusals

| what | message source |
|---|---|
| s-coordinates (`VertCoord` other than zco/zps, `e3t_0 != e3t_1d` above the bottom cell) | reader |
| ice-shelf cavities (`top_level > 1`, `misf > 1`, `isfdraft > 0` at wet columns, `IsfCav`) | reader |
| `sample_vars` on non-T points | reader |
| `Fdrift3D` with `nemo_grid: tpoint` | reader / tracking |
| `rk45`, `temporal_reconstruction` other than linear, `em`/`m1` diffusion | `tracking._refuse_nemo_cgrid_unsupported` |
| any diffusion in 3-D (2-D: uniform only), beaching | same |
| `elem_search: walk` in 3-D | same |
| `uv_interp` other than `nemo_cgrid` with `nemo_grid: cgrid` | `load_config` |

Out-of-spec configurations are refused at startup with a clear message. A
warning is given when the run stays well-defined but is less accurate, e.g.
z*/vvl output (the fixed reference thicknesses `e3*_0` are used, the transports
are off by about eta/H), cells above about 88 N without `mesh.rotation` (§1),
and an unset `data.time_averaging` (`instant` is assumed, `running.md`).

## 7. Validation

- **W against the model.** The recomputed W equals the file's W to float32
  rounding on ORCA05 over every cell of the crop (data-gated test), including
  partial cells.
- **Lagrangian transport benchmark** (ORCA05 window at the Denmark Strait /
  Iceland-Scotland sill, a 5x5-column box, 60 d, dt 3600 s, 20 055
  transport-weighted particles seeded on the inflow faces): the transport leaving
  the box through each edge matches the Eulerian face fluxes; the southern edge
  within 0.34 %, the sum of the deviations 0.8 % of the inflow (which is the
  weight still inside the box); no particle lost; no dt dependence (dt 900 s:
  0.28 %, 0.8 %). The data-gated test repeats it with about 5000 particles:
  0.31 % and 0.95 %.
- **Volume uniformity** (discrete Liouville; particles uniform per volume
  in `(xi, eta, s)` after 3 d): chi2 per degree of freedom 1.01 on the sill window.
  A scheme placing the vertical in metres gives 1.45 with 20-26 % too many
  particles in the bottom cells.
- **Topology** is exercised on real data: ORCA05 F-pivot with halos (model flux
  `u e2u` / `v e1v` balances to 0 on all 443 446 shared edges), eORCA025
  halo-free T-pivot, ORCA2 with halos (topology only).
- All the synthetic suites run without data; the data-gated ones read
  `KELP_TEST_NEMO_DATA` (`tests/test_nemo_cgrid_data.py`,
  `tests/test_nemo_cgrid_lagrangian.py`), built as described under "Test data".

A uniform flow comes out multiplied by `e2u / (dlat * 111120 m)`, i.e. +0.07 %
with the NEMO radius of 6 371 229 m. Particle paths can still cut a convex land
corner within about `v*dt/2` of it in an RK stage (rare; a smaller dt does not
remove it).

### Test data

```bash
python utils/fetch_nemo_test_data.py all --verify     # default --out ./.tmp/data/nemo_crops
export KELP_TEST_NEMO_DATA=$PWD/.tmp/data/nemo_crops
python utils/fetch_nemo_test_data.py all --dry-run    # list the source URLs only
```

The script downloads the public sources and writes the crops in the layout the
tests expect; existing files are skipped, a failed download prints the URL and
the target path for a manual download.

| source | where | licence |
|---|---|---|
| ORCA05 / FOCI1.14 | Zenodo 3755721 (the year-2011 files, about 3.8 GB, kept in `<out>/_sources/orca05/`; `--drop-sources` removes them) | CC-BY-4.0 |
| eORCA025 (NOC SE-NEMO JRA55 v1) | NOC S3 zarr store, read remotely (crops only) | unstated: local testing only, do not redistribute the crops |
| ORCA2 (IPSL-CM5A-LR, CMIP5) | ESGF / CEDA, OPeNDAP for the `*_t0.nc` records, HTTPS for the fixed fields | [CMIP5 terms of use](https://pcmdi.llnl.gov/mips/cmip5/terms-of-use.html): unrestricted use, acknowledge the providers, pass the terms on when redistributing |

Extra dependencies: `zarr>=3`, `fsspec`, `aiohttp` (eORCA025) and a netCDF4
library with DAP support (ORCA2). See `utils/README.md`.

## 8. Known limitations and follow-ups

- Real z\* (time-varying geometry, `d(e3)/dt` in the continuity, moving layer
  search); the fixed-geometry warning above is the current behaviour.
- Eddy-induced (bolus) velocity as an option: only the file's resolved
  velocity is advected.
- Diffusion in 3-D and beaching on the C-grid.
- Mirror horizontal diffusion kicks at land faces (no-normal-flux coast).
- In-kernel time blend for time-aware runs (the blend is a whole-field blend
  per stage) and region crops of the rasters in the loop.
- AGRIF nests, s-coordinates, ice-shelf cavities, ORCA grids whose land
  processors were eliminated.
- A halo-free F-pivot layout has synthetic tests only.
- Cost: ORCA05 global 3-D with 8926 particles ~ 14 ms per RK4 step on 4
  threads; about 45 % of the step is containment and nearest-node searches.
