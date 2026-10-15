# KELP gallery — reusable trajectory visualization

A catalog of **proven** techniques for figures and videos from KELP trajectory
output, plus the cross-cutting lessons that make them fast and correct. They were
developed on multi-decade, multi-million-particle campaign runs and apply to any
KELP run (the input is always the `(trajectory, obs)` `lon/lat/z/status` +
`time` output, or a compact cache derived from it).

> **Status:** this README is the *technique catalog*. The importable library module
> `src/kelp/visualization/` (`TrajectoryDataset` / `plot_mesh` / `plot_trajectories`
> / `plot_trajectories_3d` / `plot_field` / `compute_presence_probability` /
> `compute_presence_density_mesh` / `TrajectoryAnimator`) is the code layer; cleaned
> generic generators for the heavier products land in this folder as they are
> written. The `kelp-visualize` skill is the on-demand entry point to all this;
> reading is covered by the `kelp-analysis` skill (column-major!).

---

## Which technique for which goal

| Goal | Building block | Key technique |
|---|---|---|
| **Presence & +N yr snapshot maps** (density) | `compute_presence_probability` (uniform grid) / `compute_presence_density_mesh` + `plot_field` (mesh elements); the snapshot = each particle's record at k_release + τ (searchsorted, see `kelp-analysis`) fed to the same binning | Two DIFFERENT products — presence-integral vs release+N-yr snapshot. A FIXED colour scale when panels are compared (an exploratory look may use per-panel scales, labelled). |
| **Regional zoom spaghetti** (trajectory lines) | `plot_trajectories` on a regional extent (it colours by absolute depth or time, linear norm; compute depth-deviation colours yourself) | Polar/regional extent; unwrap longitudes to 0–360 for windows crossing the dateline (0→120W via 180); colour = depth deviation (non-linear norm), dark ocean by default. |
| **2D comet-tail video** (particles appear, trailing tails) | a per-frame `LineCollection` over a trailing window (`TrajectoryAnimator` is a fixed-colour, non-projected starting point) | Comet tails; colour by **depth deviation** (single release) or **tracer age** (pulsed release); optional orthographic globe / camera fly-path / spin. The pre-projection speed trick. |
| **3D eddy / vortex video** | `Line3DCollection` tails coloured by depth | One release, fixed or rotating camera; filter by release depth to isolate the deep core. |
| **Static 3D eddy** | a `Line3DCollection` coloured by depth (`plot_trajectories_3d` draws one fixed colour) | 3D depth-coloured trajectory lines; drop the surface band to unmask the deep spiral. |
| **2D pinwheel / vortex-by-angle** | scatter of positions coloured by initial angle around the gyre centre | The clearest "is it rotating" view. |
| **Density grid / volume render** | 3D presence histogram → VTK / pyvista | Headless volume renders read fuzzy — prefer crisp top-down or the line/tail views. |
| **Where similar periods differ** (decades, seasons, runs) | `figure_helpers.anomaly_vs_mean` + `symmetric_log_norm`; `box_fraction(counts)` per panel → CSV (never on the anomaly) | Anomaly = each period minus the mean over the periods; ONE symmetric-log norm and colour bar for all panels; rows = release bands, columns = periods. Absolute maps of similar periods hide the differences. |
| **Spaghetti with a zoom inset** (a gateway, a shelf) | `figure_helpers.preproject_segments` + `set_view_from_points` | Pre-project the tracks once for the main map AND the inset; the inset's view = projected bounding box of the points it must show; denser subsample in the inset. |
| **Several lines in one panel** (time series by depth, by decade) | `figure_helpers.lightness_spaced_colors` | Colours at evenly spaced CIELAB lightness of a perceptual sequential map — distinct in greyscale and for colour-blind readers. |
| **Mesh + trajectories, 2D/3D, status stats** | `plot_mesh`, `plot_trajectories`, `plot_mesh_3d`, `compute_status_ratios`, `plot_diagnostics` | Importable, publication-ready; reads `load_trajectories(...)` + a `Mesh`. |

The engine behind large-run figures is **extract → cache(.npz) → render**: read the
trajectory store *column-major* once, fan every obs column out to all cohorts, cache
compact float32 arrays, render from the cache (`--render-only` = no store re-read).
This extract→cache→render split is the pattern to copy for any large run.

---

## Cross-cutting techniques (the hard-won lessons)

**Speed — the one that matters (×20), for videos AND static line maps.** The
bottleneck is cartopy RE-projecting every `LineCollection` segment on every draw
(~14 s/frame at 720p; a static spaghetti map with a zoom inset took 17–30 min). **Fix: project
all lon/lat to the axes projection ONCE** (`ax.projection.transform_points`), then
draw in plain projected coords with **no** `transform=PlateCarree` → ~0.7 s/frame.
Also drop `cfeature.OCEAN` (use `ax.set_facecolor`) and use 110 m coastlines. Use
`imageio_ffmpeg`'s bundled binary (clusters often have no system ffmpeg). A 60 s
720p video ≈ ~15 min; HD ≈ ~30–40 min.

**Comet tails.** Each particle drags a trailing window (tail length in years);
colour by depth-deviation OR tracer age; fade alpha along the tail (recent opaque,
old faint). For pulsed releases colour-by-**age** cures the flicker (a fresh pulse
is the darkest colour, so it can't flash). Pre-roll the read window by `tail-1`
records so frame 0 already shows full tails.

**Colour ↔ sign ↔ label MUST agree** (a real trap). `DZ = current_depth −
release_depth`: DZ>0 = **sank**, DZ<0 = **rose**. `cmap="RdBu"` → high=blue, low=red
→ **blue=sank, red=rose**; the colourbar label must say the same. **Change the cmap
and the label together, never one alone.** Verify with `cmap(0)`/`cmap(1)`.

**Colour scales.** Non-linear near 0 so small deviations show: spaghetti uses
`BoundaryNorm(DEV_LEVELS=±10/30/50/70/100/150/200)`; comet video uses a symmetric
`FuncNorm` (sqrt) for a smooth "fast colour near 0". For density maps compared across panels use a **FIXED**
`PowerNorm(0.45, vmin=0, vmax=0.01)` + `extend="max"` so panels are comparable
across decades/depths — per-figure autoscale hides the cross-panel signal (an
exploratory look at each panel may use its own scale, labelled as such).

**Statistical fields — two different products, don't conflate.** `PROB` = presence
integrated over each particle's 0→N yr window (dominated by early near-source
residence → "everyone stays home"). `PROB10` = each particle's position at exactly
+N yr from its OWN release (one point/particle, survivors only → "where they end
up"). Keep BOTH, label them.

**Drafts first.** Iterate layout and colours at a low track density (every 400th
particle: seconds per render), render the chosen density once at the end, and print
per-stage timings (load, main map, insets, save).

**Palette against the background.** Aim for ≥ 2 : 1 contrast of every palette bin
against the ocean colour — a dark ocean hides the dark end of a diverging track palette,
a white ocean hides its pale centre; compute it per bin (WCAG contrast), do not judge by
eye. On white, raise line weight (≥ 0.55 pt) and alpha (≥ 0.9) before changing the
palette. When the palette is undecided, offer 4–6 variants as one
`figure_helpers.contact_sheet` and let the reader pick. The script prints
`figure_helpers.palette_contrast(cmap, OCEAN)` and names the weak bins; it asserts only
on a palette it chose itself — an owner-named palette is drawn as named, the weak bins
reported as advice.

**Numbers in captions come from tables, not from the map.** The render script writes
the numbers a caption quotes (box fractions, event counts) to a CSV next to the figure;
a statement read off a map by eye is a hypothesis until the table confirms it. Keep a
`CAPTION_<name>.md` next to each figure, written by the render script itself
(`figure_helpers.write_caption`: what it replaces, regenerate command, data, the engine commit
that produced the data, the render script's own commit, caption).
Never delete a superseded figure: new renders go to a new dated folder, and the script
builds the old-vs-new side-by-side for review (`figure_helpers.old_vs_new`).

**Presentation.** Top-down crisp views read
better than fuzzy 3D volumes; decimate drawn vertices + `set_rasterized(True)` for
heavy line sets. Put **released (all)** vs **shown (sample)** counts on videos, and
for pulsed runs note "releases only in yr 0–N".

**Memory / rendering gotchas (bite on big runs).**
- **Column-major read** of the trajectory Zarr, chunked `(trajectory block ≤ 500 000, obs block ≤ 256)` by the writer (older stores: one trajectory block); `xr.concat`
  segments with `compat="override", coords/data_vars="minimal", join="override"`.
- The npz cache stores a **subsample** of per-particle tracks (`--cache-every`), but
  the density and snapshot arrays and the counts are all-particle; denser spaghetti/3D needs a smaller `cache-every`.
- **Lazy cache load**: load only the cohort keys you need
  (`z[f"{season}__{band}__X"]`), not the whole cache — a dense cache is ~8 GB and
  `load_cache` OOMs pulling all cohorts.
- **Montage/many figures: one unit per fresh process** (matplotlib/cartopy leak → OOM
  ~30 figs); loop in the shell so each process frees memory.

**Camera (3D / globe).** Fixed view is fine and clearest for a first pass; slow
azimuth rotation (`--rotate`) or an orthographic globe + camera fly-path for hero
shots. In 3D, exaggerate the vertical (depth ≪ horizontal in metres) and set
`set_box_aspect`.

**Pace a video** by frames, not just fps: 1 data-column/frame is the slowest smooth
motion (sub-column needs interpolation); more frames over the same drift = slower.

---

## Adapting to a new run (the input contract)

Every generator you write consumes one of two things:

1. **The trajectory output directly** — `(trajectory, obs)` `lon/lat/z/status` + a
   `time (obs,)` axis, from a `.zarr`/`.nc` or a directory of resumable-chain
   `part_*.zarr` segments (read column-major — see `kelp-analysis`). A video
   generator can read the store directly with a trailing-window read.
2. **A compact cache** (`.npz`) — extract once, render many.

To reuse on a new run: point `--traj`/`--cache` (and mesh/seed paths) at your data;
adjust the **cohort/label loader** (`pulse = traj_id // n_seed` for a pulsed release)
to your release scheme, or drop cohorts entirely for a single-release run. The
drawing, colour, speed, and camera code is dataset-agnostic — only the *loader* is
run-specific. When you generalize a generator (generic input, no campaign-specific
cohorts), drop the cleaned version here in `gallery/` and update this table.

## References
- `src/kelp/visualization/README.md` — the importable package module.
- skills: `kelp-analysis` (reading), `kelp-run`/`-config`/`-experiment` (runs).
