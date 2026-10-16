# CLAUDE.md — Developer Guide for LLM agents

This file describes the `kelp` package internals so an LLM can continue development without re-reading every source file. It records **non-obvious** facts: conventions, cross-module contracts, gotchas. Plain API details live in the docstrings — read the module when you need them.

## What this project is

A Python Lagrangian particle tracker for unstructured ocean meshes, reimplementing Fortran tracer models. Status:

- **2D surface tracking** — complete; validated against the Fortran reference (forward + backward, < 1 mm) while both used 111 300 m per degree — since the switch to 111 120 the comparison test is skipped (~11.6 m systematic offset after 2 h at 1 m/s, see the Fortran section).
- **3D vertical tracking** — FESOM2 z-levels (direct W input) and FESOM-C sigma coordinates.
- **Stochastic diffusion** — uniform Kh / Euler-Maruyama / Milstein, wired into Euler/RK2/RK4 via three composition modes (split / combined / split_rk), counter-based RNG; **vertical Az** (TODO 2.2b, 3-D only) — split kick with mandatory Itô drift (Visser 1997) + reflective boundaries, `az_const` everywhere / `var_az` field FESOM2 z-level, `COMP_VERTICAL=1`.
- **Time-aware RK2/RK4** — stages lerp velocity to substep times on non-autonomous flows; opt-in 4-snapshot PCHIP in time (`numerics.temporal_reconstruction`); adaptive **RK45** (Dormand–Prince, 2-D element leastsq).
- **u/v interpolation** — `leastsq` (default), `barycentric` (node layout: HEALPix, NEMO, FESOM2 `unod`/`vnod`), `edge_leastsq` (FESOM-style).
- **HEALPix** reader (2-D + 3-D, dual or pixel geometry, regional extracts) and **NEMO** (2-D, E/N-aligned grids) besides FESOM2 and FESOM-C.
- **Offline W reconstruction** (`w_reconstruct.py`) from u, v, ssh.
- **Per-particle FailureReason diagnostics** — why each particle died.
- **Beaching model** (TODO 2.2) — 4 combinable criteria (depth/coastal_element/on_failure/stochastic), tidal resuspension (never/auto/stochastic/auto_or_stochastic), Fortran fix_* freeze via the alive-mask.
- **Field sampling along trajectories** (TODO 2.4) — T/S/any node-based scalar sampled at record points, written to trajectory output; standalone `sample_field` API.
- **Rotated-frame tracking** (FESOM2 + HEALPix, opt-in `mesh.rotation`) — Euler-rotated sphere (default `None` byte-identical; `"fesom2"` = (50,15,−90)); transpolar-capable, geographic output, frame-validated restart.
- **FESOM-style elemental interpolation + element tracking** (reviewer items, 2026-07) — `uv_interp: edge_leastsq` (LSQ over the containing element + its edge neighbors; boundary ghosts per `edge_leastsq_bc: freeslip|noslip`); `elem_search: walk` (containing element searched ONCE at release, then tracked across edges; verdict-identical to the per-step scan, unfused paths, ~3.6× faster containment); quad vertex weights = guard-free product-form Wachspress (convex quads are a mesh contract).
- Streaming trajectory output (NetCDF **and Zarr**, both readable mid-run — Zarr is lock-free, NetCDF via `numerics.output_flush_every`; **output format 2**: CF trajectory layout with a calendar `time`, release positions/times, velocities NaN outside ALIVE steps) with restart checkpoints, temporal velocity interpolation, installable package (`pip install -e .`), `kelp` CLI (`run` / `info`).
- Performance: measured rows live in `benchmarks/` and `docs/performance.md` (the old "15–19× faster than Fortran" figure has no measurement in this repository).

## Running an experiment (start here for a new run)

To turn a described experiment into a runnable setup, do NOT re-derive
capabilities from the source — the mechanics are already written down:

- **Skill `kelp-run`** (`.claude/skills/kelp-run/SKILL.md`) — the entry
  point: a described task (release location, forward/backward, surface/3D,
  diffusion, beaching, T/S sampling, time window) → **seeds + config + run
  command**. **Skill `kelp-config`** — the YAML-writing mechanics.
  **Skill `kelp-experiment`** — the campaign layer for a **big / long /
  big-mesh** experiment (millions of particles, runs that outlive one allocation,
  region cropping): the order-of-operations (isolate clone → crop+validate →
  wait-for-extraction → probe → load-probe → resumable dependency chain) and the
  setup pitfalls to avoid. The other skills: **`kelp-analysis`** (reading
  trajectory output at the writer's chunking; cohort statistics, denominators,
  censoring, timing numbers), **`kelp-visualize`** (figures, montages,
  videos; the figure contract), **`kelp-diagnose`** (a run slower than
  documented, silent, or scaling oddly — the diagnosis path BEFORE any code or
  config change), and **`kelp-bench`** (the portable benchmark row per
  machine).
- **`docs/dev/README.md`** — the per-module notes, when a run needs a fine detail
  of a module (a default, a fail-fast, a known trap); see "Module notes" below.
- **`docs/configuration.md`** — authoritative option-by-option reference
  (defaults, validation, file-discovery rules, pitfalls). **`docs/architecture.md`**
  — the config→reader→mesh→loop→writer pipeline.
- **Ready templates** — `examples/config_*.yaml` (genericized real runs):
  `config_global_2d_10yr.yaml` (2D, multi-year, node velocities, pulsed release,
  rotated frame), `config_global_2d_1yr.yaml` (simpler), `config_weddell_3d.yaml`
  (3D, W + T/S sampling), `config_nemo.yaml` (structured C-grid),
  `config_healpix_2d.yaml` / `config_healpix_3d.yaml` (HEALPix stores). Copy and
  set the paths.
- **Tools** — `utils/extract_fesom_region.py` (crop a region/years → Zarr; the
  memory lever for big meshes so 3D / 4-snapshot reconstruction fits in RAM);
  `utils/reconstruct_w.py` (offline W when a dataset lacks vertical velocity);
  the full list is in `utils/README.md`.
- **Mesh ≠ data**: the mesh diagnostic (`fesom.mesh.diag*.nc` / dir / Zarr) and
  the model output normally live in *different* directories — set both.
- **Data shapes seen in practice** (no fixed locations — set paths per run):
  *FESOM2* multi-year output is one file per variable per year
  (`<var>.fesom.<year>.nc`); a single run cadence must be consistent, so pick
  one family — node velocities `unod`/`vnod`/`w` (+ `temp`/`salt`/`ssh`) are
  often **daily**, while element `u`/`v` may be **monthly**; the mesh is a
  FESOM2 `fesom.mesh.diag.nc` (core2/`.comp`/UGRID conventions auto-detected).
  *FESOM-C* is a single coastal NetCDF (sigma coordinates).
  *HEALPix twins* (validation-grade): combined one-year stores with daily node
  `uo/vo/wo/ssh/temp/salt`, self-contained incl. the vertical table; the tests that
  use them read the directory from `KELP_TEST_DATA` (see `tests/test_healpix_real_core2.py`).

## Repository layout

**`src/` layout** — `pip install -e .` from the repo root makes `kelp` importable without sys.path hacks (tests fall back to inserting `src/` via `conftest.py`).

### Documentation map (read before diving into source)

Topic docs live in `docs/`; `docs/architecture.md` is the entry point
(full data flow + per-step call sequence + core data structures). Each
doc's header names the authoritative `file:symbol` anchors.
`docs/doc_map.yaml` is the canonical subsystem ↔ code ↔ doc map; run
`python docs/check_doc_sync.py` to see which docs have code changed under
them since they were last verified.
The per-module developer notes are in `docs/dev/` (see "Module notes" below).

```
kelp/                     ← repo root
├── pyproject.toml            ← packaging + pytest config + [project.scripts] + extras
├── environment.yml, requirements*.txt, CITATION.cff, .zenodo.json, LICENSE
├── README.md / CLAUDE.md / AGENTS.md / CHANGELOG.md
├── config.yaml               ← annotated copy-paste config template (root)
├── docs/                     ← topic docs; architecture.md = entry point, configuration.md =
│                                every option, interpolation.md = formula reference;
│                                doc_map.yaml + check_doc_sync.py = doc ↔ code sync
│   └── dev/                  ← developer notes per module, read on demand (index: dev/README.md)
├── manual/                   ← LaTeX user manual (make -C manual pdf)
├── benchmarks/               ← exact_nearest_cost.py, nearest_scan_table.md,
│                                grid_hash_cell_rule_table.md, single_node/ (the kelp-bench row)
├── gallery/                  ← figure helpers for the README gallery
├── .claude/skills/kelp-*/    ← agent skills (run, config, experiment, analysis, visualize,
│                                diagnose, bench)
├── src/kelp/
│   ├── core/                 ← primitives, no domain logic
│   │   ├── geometry.py       ← spherical math, physical constants
│   │   ├── mesh.py           ← Mesh dataclass, grid-hash/KD-tree search, containment, edge walk
│   │   ├── particles.py      ← ParticleState, ParticleStatus, FailureReason
│   │   ├── particles_gen.py  ← grid + around-point seeding
│   │   ├── release.py        ← delayed/staggered release (TODO 2.5), seed-file readers
│   │   ├── io.py             ← VelocityField container + lerp, LazyLerpField
│   │   ├── rotation.py       ← Euler rotated-frame primitives (pole-safe tracking)
│   │   ├── shared_fields.py  ← node-shared snapshot cache (TODO 6.9)
│   │   └── config.py         ← YAML → AppConfig
│   ├── numerics/             ← math kernels (Numba hot paths)
│   │   ├── interpolate.py    ← u/v interpolation: leastsq, barycentric, edge_leastsq
│   │   ├── integrators.py    ← Euler / RK2 / RK4 / RK45 (2D + 3D), fused kernels,
│   │   │                        diffusion + time-aware dispatch
│   │   ├── temporal_reconstruct.py ← 4-snapshot PCHIP in time
│   │   ├── diffusion.py      ← counter-RNG, uniform/EM/M1 SDE kernels, Kh interp, Az kick
│   │   ├── beaching.py       ← beaching criteria + resuspension (TODO 2.2)
│   │   ├── sampling.py       ← scalar field sampling at particle positions (TODO 2.4)
│   │   ├── vertical.py       ← z-level layer/W helpers
│   │   └── vertical_sigma.py ← sigma layer/W helpers (FESOM-C)
│   ├── readers/              ← registry: get_reader("fesomc" | "fesom2" | "nemo" | "healpix"); NetCDF + Zarr + kerchunk
│   │                            (structured.py = curvilinear→quad-Mesh foundation; nemo.py = NEMO adapter, 2-D;
│   │                             healpix.py = HEALPix dual/pixel mesh builder + native reader, 2-D and 3-D — docs/healpix.md)
│   ├── visualization/        ← matplotlib 2D/3D plots, animation, stats
│   ├── tracking.py           ← orchestration: run_tracking, run_single_realization, track_particles
│   ├── writers.py            ← trajectory writer (NetCDF4 / Zarr v3) + checkpoints + restart files
│   ├── w_reconstruct.py      ← offline W reconstruction (FESOM2 z-level only)
│   └── cli/main.py           ← `kelp run|info`, argparse subcommands
├── examples/                 ← runnable YAML configs only (genericized real runs):
│      config_global_2d_10yr.yaml, config_global_2d_1yr.yaml,
│      config_weddell_3d.yaml, config_nemo.yaml, config_healpix_2d.yaml,
│      config_healpix_3d.yaml
├── utils/                    ← README.md (what each tool is for),
│                               extract_fesom_region.py (region → Zarr subset),
│                               extract_healpix_region.py (HEALPix pixel subset → self-describing Zarr),
│                               reconstruct_w.py (offline W; YAML-driven),
│                               check_grid_hash_cell.py (window occupancy / exactness of a cell
│                               for a config's seeds and mesh),
│                               sort_seeds.py (spatial (NEST cell, depth) seed sort for cache
│                               locality — sort FIRST, then shard),
│                               split_seeds.py + concat_trajectories.py (particle sharding:
│                               contiguous shards → parallel runs → order-preserving merge,
│                               bit-identical to the unsharded run incl. stochastic runs — each
│                               shard file carries `# kelp particle_offset N`), + example yamls
└── tests/                    ← pytest suite; one file per module + feature suites
       (test_vertical.py, test_vertical_sigma.py, test_w_reconstruct.py,
        test_w_spatial_interp.py, test_3d_analytical.py, test_analytical_3d_benchmarks.py,
        test_downwelling_psi.py, test_sigma_time_aware_eta.py, test_fesomc_sigma_medi.py,
        test_diffusion.py, test_integrators_diffusion.py, test_tracking_diffusion.py,
        test_sampling.py, test_beaching.py, test_sharding.py, test_exact_nearest.py,
        test_integrators_time_aware.py, test_temporal_interp.py, test_failure_reason.py,
        test_regression.py, test_real_mesh.py, test_visualization.py, test_cli.py,
        test_parcels_comparison.py, test_parcels_diffusion_parity.py,
        test_parcels_diffusion_comparison.py, test_healpix_*.py, ...)
```

## Module notes (`docs/dev/`)

The per-module notes — contracts, data shapes, traps, measured costs — and the
import graph live in `docs/dev/`. They are read on demand and never imported into
this file:

- **before changing code or tests of a module**: read
  [`docs/dev/README.md`](docs/dev/README.md) (one entry per module: what it does,
  its main trap, its notes file and the main other places), then those notes, and
  find every other mention with `grep -n '<module or function>' docs/dev/*.md`;
- **during a run**: open them when a fine detail of a module matters (a default,
  a fail-fast, a metadata attribute, a known trap).

| notes file | modules (below `src/kelp/`) |
|---|---|
| [`docs/dev/config.md`](docs/dev/config.md) | `core/config.py`; also the grid-hash cell rule and the exact nearest-node guard (internals of `core/mesh.py` and the fused kernels) and `core/shared_fields.py` |
| [`docs/dev/mesh.md`](docs/dev/mesh.md) | `core/geometry.py`, `core/mesh.py` |
| [`docs/dev/core.md`](docs/dev/core.md) | `core/particles.py`, `core/release.py`, `core/particles_gen.py`, `core/io.py`, `core/rotation.py` |
| [`docs/dev/writers.md`](docs/dev/writers.md) | `writers.py` |
| [`docs/dev/numerics.md`](docs/dev/numerics.md) | `numerics/interpolate.py`, `numerics/vertical.py`, `numerics/vertical_sigma.py`, `numerics/diffusion.py`, `numerics/beaching.py`, `numerics/sampling.py`, `w_reconstruct.py` |
| [`docs/dev/integrators.md`](docs/dev/integrators.md) | `numerics/integrators.py`, `numerics/temporal_reconstruct.py` |
| [`docs/dev/tracking.md`](docs/dev/tracking.md) | `tracking.py`, `cli/main.py` |
| [`docs/dev/readers.md`](docs/dev/readers.md) | `readers/` |
| [`docs/dev/visualization.md`](docs/dev/visualization.md) | `visualization/` |

## Test patterns

Tests use synthetic data built in `tmp_path` fixtures or small committed fixtures under `tests/data/` (e.g. `test_real_mesh.py`, `test_regression.py`) — no external data needed (exceptions: the `@slow` suites, and the data-gated HEALPix suites — see `KELP_TEST_DATA` in their docstrings).

- Canonical mesh: 3×3 grid (9 nodes, 8 triangles) centred at (5°, 5°) — interior node gets ~6 surrounding elements for robust interpolation.
- Constant velocity fields so displacement is analytic: `Δlon ≈ U·dt / (cos(lat)·METERS_PER_DEG_LAT)`.
- Output format 2 is pinned by `tests/test_output_format.py` (layout, CF attributes, time units per calendar, the record clock in both modes incl. backward/cycle/chain cuts, release variables, velocity NaN rule, lazy restart block, format-1 reads, concat refusals).

```bash
python -m pytest tests/          # default: 2687 collected (2652 pass, 35 data/plot skips as of 2026-10-14), 18 slow deselected
python -m pytest tests/ -m slow  # Parcels comparison + data-gated HEALPix-twin / W-sign suites
```

`@slow` tests live in four files: `test_parcels_comparison.py` (kinematic benchmarks vs v4 reference CSVs) and `test_parcels_diffusion_comparison.py` (cross-system moment comparison vs v4 reference) need a sibling Parcels repo (its benchmark outputs directory, see the test docstrings; own venv); `test_healpix_native_twin.py` and `test_w_sign_contract.py` need the HEALPix twin / native data (`KELP_TEST_DATA`, `KELP_TEST_NATIVE_DATA`). `test_parcels_diffusion_parity.py` (Parcels' two diffusion benchmarks re-run natively: Brownian moments, spatially-varying Kh skewness) is NOT slow and runs in the default suite.


**Validation suites get a mutation/honesty pass** (HEALPix Phase 5, 2026-08): after pinning measured tolerances, perturb the code or data in ~10 deliberate ways — including a UNIFORM time-axis offset, which statistics-only pins cannot see — and check that each pin fails; pin the ORDERING too (finer resolution → closer to the reference), the claim a broken remap would violate.

**Numba cache and concurrency (2026-09-12):** two processes compiling the SAME kernels into one on-disk cache (the suite plus a benchmark, a reviewer subagent, or a second worktree sharing `src/**/__pycache__`) corrupt each other's index and crash with `NRT_adapt_ndarray_to_python: 'descr' is NULL` or stale-callee results — give every concurrent process its own `NUMBA_CACHE_DIR` (under `./.tmp/`), and wipe `src/**/__pycache__` after editing a callee that is inlined across files (mesh.py → integrators.py). Same rule for campaign clones after a `git pull`.

## Fortran reference

The original Fortran tracker, its bug reports, and the comparison/benchmark
scripts are not part of this repository. The facts worth keeping:

Validation (10 particles, 10 steps, dt=3600 s, Euler): forward and backward both < 0.6 mm vs Fortran with the Fortran constant `SLM = 111300`; after the switch to 111 120 the parity test (`tests/test_regression.py::test_forward_euler_vs_fortran`) is skipped — a systematic ~11.6 m offset after 2 h at 1 m/s. Two Fortran bugs were found and fixed in the Python port: the **a11 summation bug**, and **catastrophic cancellation** in the absolute-coordinate Sarrus determinant — fixed by centering the least-squares fit on the particle (see docs/interpolation.md §1 and docs/velocity_interpolation.md §2).

| Fortran | Python |
|---------|--------|
| `SLM = 111300.0` | `METERS_PER_DEG_LAT = 111120.0` (Parcels convention; parity broken intentionally) |
| `z0 = dcos(yp*rad)*SLM` | `np.cos(yp * DEG2RAD) * METERS_PER_DEG_LAT` |
| `st_lon = xp + dt*tam_tr(i)*UU/z0` | `lon_new = lon + dt * UU / z0` |
| `elem2D_nodes(4,elem2D)` | `mesh.elem_nodes` (transposed, 0-based) |
| `nod_in_elem2D` | `mesh.node_to_elems` |
| `sFB` (+1/-1) | `mode.time_dir` (config) / `time_dir` argument |
| `tam_tr` (0/1 activation) | `state.alive_mask` |

## Remaining work

`TODO N.N` markers in the code and in this file are stable labels of the
project's internal backlog items, not links to a file in the repository.

- **Vertical beaching (v2)**: `FailureReason.SURFACE_HIT`/`BOTTOM_HIT` codes stay reserved — horizontal beaching (TODO 2.2) shipped; bottom/surface contact in 3-D did not.
- **3D diffusion**: combined and split_rk modes raise NotImplementedError in 3D. Of the four originally planned 3-D diffusion items, **vertical Kz shipped** as the Az kick (TODO 2.2b, incl. its reflective vertical boundaries); still open: anisotropic Kh tensor, adaptive dres_m, horizontal reflective boundaries. Az v1 gaps (staged): no horizontal Az interpolation (nearest-node column), no Az field on sigma meshes (const only), Az not PCHIP-reconstructed in time.
- **Sampling on sigma meshes**: mid-layer `(K-1, N)` scalars (T/S) are sampled per neighbour at the particle's DEPTH (bottom-aware), not along the sigma surfaces like W and the u/v layer (review of fix/sigma-step) — make `sampling._loop_sample_mid_sigma` sigma-following with the method's `H_p`.
- **Output**: no Fortran-compatible ASCII `.dat` writer; multi-realization runs write one file per realization (`_rN`), no single-file mode.
- **W reconstruction**: edge-flux divergence (exact FESOM2 match, needs edge connectivity) and proper z-star layer distribution for the ALE correction not done. (The divergence kernels' antimeridian fragility was gate-fixed 2026-07, same idiom as the leastsq cores — no raw-lon-delta consumers remain.)
- **Structured / NEMO grids** (TODO 4.6): v1 shipped (`readers/structured.py` + `readers/nemo.py`) is **2-D, E/N-aligned grids only** — NEMO `grid_U`/`grid_V` are along the local grid axes, so a grid whose axes deviate > 2° from E/N (skewed/curvilinear/global ORCA) is rejected. Staged follow-ups, in priority order: **(1) vector rotation** — rotate grid-relative u/v into geographic E/N (from `gcos`/`gsin` or the coordinates), the gate that lifts the E/N-only restriction and is the prerequisite for everything below (a *fully reversed* axis — i→west / j→south, caught by the same signed-angle metric — additionally needs an index/winding flip, not just a rotation angle); **(2) global ORCA** — north-fold seam dedup + east-west periodicity (the foundation grows `periodic_x`/`north_fold`); **(3) 3-D** — `deptht`/`e3t`/partial cells, W on interfaces → `VelocityField.w (K, N)`, per-node bottom depth from bathy, exact `e1t*e2t` areas; **(4) offline converter** `utils/nemo_to_mesh.py` (same foundation, writes a FESOM-C-format mesh); **(5)** em/m1 diffusion onto T-points (fails fast now) + validate `beaching.dry_mask_var` (read via the scalar infra but **untested/unguarded** on NEMO — not a fail-fast); **(6) ROMS/MOM6 adapters** (same `structured.py`); exact C-grid scheme (Delandmeter & van Sebille 2019) = a separate interpolator (C→T averaging loses divergence — coastal streamlines may slightly cut land).

## Conventions

1. **Numba for hot paths** — no Jax, no PyTorch. Always with a graceful Python fallback.
2. **Type hints + NumPy-style docstrings** on all public functions.
3. **Tests for every module** — synthetic data in `tmp_path`, no external files.
4. **Immutable state** — integrators return a new `ParticleState` via `.replace()`, never mutate.
5. **Reader interface** — `load_mesh` / `open_dataset` / `read_velocities` + `@register("name")`.
6. **Transient files go to `./.tmp/`** (gitignored) — never to system temp directories.

## Documentation sync

- **Docs sync:** after a behaviour change > ~50 LOC, run `python
  docs/check_doc_sync.py`; update any doc it flags as drifted and re-stamp
  with `--bless <subsystem>`. New subsystem behaviour → add/extend its
  `docs/doc_map.yaml` entry.
