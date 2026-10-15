---
name: kelp-visualize
description: Use when making a figure, animation or video from KELP trajectory output or its caches — track or spaghetti maps, comet-tail or 3-D eddy videos, presence or +N-year maps, differences between decades or periods, zoom insets, montages, figure captions — or when colours, palettes, contrast or labels need changing, or a render is too slow.
---

# Visualizing KELP trajectory output

**Entry points — read these first:** `gallery/README.md` in the KELP repo is the
catalog of techniques (which building block for which goal + the cross-cutting
lessons); the importable library is `src/kelp/visualization/` (`TrajectoryDataset`,
`plot_mesh`, `plot_trajectories`, `plot_trajectories_3d`, `plot_field`,
`compute_presence_probability`, `compute_presence_density_mesh`, `TrajectoryAnimator`;
its README documents them). Reading the trajectory data efficiently is the
`kelp-analysis` skill (**column-major** — non-negotiable).

Don't reinvent these — the correctness and speed lessons below are already baked in.
Start from the library function or the recipe closest to the goal, point it at your
data, adjust only the loader.

**Defaults and checks.** Palettes, backgrounds, line weights, layouts, the colour
variable and the panel set below are DEFAULTS that worked for past figures. When the
owner names one of them, draw what the owner named. The CHECKS hold whatever the style:
where the label gives colours a meaning (blue = shallower, red = more), the colour
map's ends match it — a bare `assert` in code (never wrapped in try/except or turned into a
warning), whoever chose the palette; the colour variable
is defined correctly (age of THAT particle; DZ = current − release depth); a +N-yr map
states released / surviving; panels meant to be compared share one scale; caption
numbers come from a table the script wrote; a superseded figure is kept. A check the
handback names cites the script line that does it (`render_tracks.py:101`).
The handback has a section `Defaults not used / checks skipped`, one line each:
`<default> — owner's choice: <what>` or `<check> — <condition not met>`. When the owner named a
style the section is never empty (e.g. `dark-ocean recipe (RdBu_r cut, #0f1113, 0.5 pt) — owner's
choice: balance on #d9d9d9, 0.4 pt`).

**Reference files (read the one a bullet points to):**
- [references/palette.md](references/palette.md) — palettes against the ocean, contact sheets, line colours.
- [references/spaghetti-and-insets.md](references/spaghetti-and-insets.md) — track maps with a zoom panel.

## Which building block for which goal

- **Comet-tail video** (particles appear at release, drag a fading tail): a per-frame
  `LineCollection` over a trailing window (tail length in records), alpha fading along
  the tail, colour = tracer age (pulsed release) or depth deviation (single release);
  the pre-projection trick below. `TrajectoryAnimator` is a minimal starting point
  (fixed-colour tails, plain matplotlib, NO pre-projection and NO age/DZ colouring —
  add both yourself).
- **3D eddy / vortex video**: `Line3DCollection` tails coloured by depth, one release,
  fixed or slowly rotating camera; filter by release depth to isolate the deep core;
  exaggerate the vertical (`set_box_aspect`).
- **Static 3D eddy**: depth-coloured lines from a `Line3DCollection` (as in the video
  recipe above) — the library's `plot_trajectories_3d` draws every path in one fixed
  colour; drop the surface band to unmask the deep spiral.
- **Regional-zoom spaghetti**: `plot_trajectories` on a regional extent (it colours
  by absolute depth or time with a linear norm and does no pre-projection — for
  depth DEVIATION with a non-linear norm compute the segment colours yourself);
  unwrap longitudes to 0–360 for windows that cross the dateline; a dark ocean is the tested default (a background the owner
  names replaces it, and the palette check below reports against that one).
- **Pinwheel / rotation view**: top-down dots coloured by each particle's initial
  angle around the gyre centre — the clearest "is it rotating" view.
- **Presence / +N-yr snapshot maps + montages**: `compute_presence_probability`
  (uniform grid) or `compute_presence_density_mesh(ds, mesh)` (particle density on
  the mesh ELEMENTS themselves, not a uniform grid — plot it with `plot_field`;
  `normalize="per_area"` for a per-m² density); montage = one unit per fresh
  process, one shared norm.
- **Any node/element scalar as coloured mesh faces**: `plot_field(mesh, values)`.
- **Where similar periods differ** (decades, seasons, runs that look alike as
  absolute maps): by default an ANOMALY montage — each period minus the mean over the periods,
  symmetric-log norm (`SymLogNorm`, linthresh ≈ the smallest density you care about),
  ONE diverging colour bar for all panels, rows = release bands, columns = periods;
  write each panel's box/gate fractions to a CSV next to the figure, computed on the
  raw particle COUNTS of that period (`box_fraction(H, …)` = its share of the period's
  particles; not the per-area density you plot), never on the anomaly (an anomaly sums to about zero; `box_fraction` refuses it).
  `gallery/figure_helpers.py::anomaly_vs_mean`. If the owner asks for absolute maps or
  another layout, draw that; box fractions still come from raw counts.
- **Spaghetti with a zoom inset** (a gateway, a shelf): the inset's view comes from the
  projected bounding box of the region it must show, never from a lon/lat box — the recipe and
  its code are in [references/spaghetti-and-insets.md](references/spaghetti-and-insets.md).

Engine pattern to copy for any large run: **extract → cache(.npz) → render** — read
the trajectory store column-major ONCE, fan every record column out to all cohorts, cache
compact float32 arrays (per-particle tracks as a SUBSAMPLE, `--cache-every N`; counts
all-particle), render from the cache. The render scripts are yours to write (the repo
ships the library, not the renderers); every one's usage block carries both flags,
verbatim:
```
python render_<product>.py --cache-dir <cache_dir>                # extract + cache + render
python render_<product>.py --cache-dir <cache_dir> --render-only  # re-render from the cache, no store re-read
python render_<product>.py --cache-dir <cache_dir> --render-only --compare-old <old.pdf>  # + old-vs-new review PNG
git -C <campaign> add analysis/render_<product>.py && git -C <campaign> commit -m "render_<product>"   # before the FINAL render (drafts: --allow-dirty)
```

## Must-know techniques (details in gallery/README.md)

- **Speed ×20, videos AND static line maps:** the bottleneck is cartopy
  re-projecting every LineCollection segment. **Pre-project lon/lat to the axes
  projection ONCE** (`gallery/figure_helpers.py::preproject_segments` — one call per panel:
  `segs, vals = figure_helpers.preproject_segments(ax.projection, lon, lat, values=dz)`; never
  `transform_points` per segment or per track in a Python loop), then draw with no
  `transform=PlateCarree`; drop `cfeature.OCEAN` (use `set_facecolor`), 110 m
  coastlines (50 m in a zoom inset). A static spaghetti map with an inset took
  17–30 min without it, minutes with it. Video: imageio_ffmpeg's bundled binary; pace
  by frames (1 data-col/frame = slowest smooth).
- **Drafts first, one final render.** Every layout or colour iteration runs at a low
  track density (every 400th particle — seconds); the chosen density (e.g. every 10th)
  is rendered once, at the end; one invocation renders one density (`--stride`), so an
  iteration never re-renders the dense final. Every render prints its per-stage timings; the inset
  is its own stage, drawn after both main maps, never inside the panel loop:
  ```python
  t = [("start", time.perf_counter())]; stamp = lambda k: t.append((k, time.perf_counter()))
  # load the cache ...                     stamp("load")
  # draw both main maps ...                stamp("main map")
  # draw the inset ...                     stamp("inset")
  # savefig ...                            stamp("save")
  print(" · ".join(f"{k} {b - a:.1f} s" for (_, a), (k, b) in zip(t, t[1:])))
  ```
  Paste the three code lines unchanged and move only the four `stamp()` calls: each
  printed duration is named after the stamp that ends it. A layout change that takes
  minutes means pre-projection is missing.
  Densities count PARTICLES, not cache rows: on a cache that already holds every Nth
  particle, the draft stride is 400/N (40 for every 10th) and the final stride is 1.
- **The colour variable follows the release scheme by default:** a PULSED run
  (`n_pulses > 1`) is coloured by tracer AGE — a fresh pulse is the darkest colour and
  cannot flash; depth deviation (DZ) is the default for SINGLE-release runs. If the owner
  names the colour variable, use it. Whatever is coloured is defined per particle: age
  is `sgn·(t − t_release[i])` of THAT particle, `sgn = np.sign(time[-1] − time[0])` (+1 forward,
  −1 backward: a backward run's labels count down, so age stays positive), `t_release[i]` =
  `release_time` of the store (format 1: its first non-NaN record; never the pulse index, never
  the release year — those colour a pulse, not its drift); NaN before the release.
- **Colour ↔ sign ↔ label must agree:** `DZ = current − release depth` (`release_z`; format 1:
  the seed depth); DZ>0 = sank,
  DZ<0 = rose. `RdBu` goes red(0) → blue(1) (blue=sank, red=rose); `RdBu_r` is the
  reverse — do not guess, put the check in the code, not in a comment:
  `assert cmap(1.0)[2] > cmap(0.0)[2], "label says blue=sank but cmap is reversed"`
  (`[2]` is the BLUE channel of the RGBA tuple; `[3]` is alpha and always 1);
  **change cmap and label together**. The assert follows YOUR label: for "blue =
  shallower" (DZ < 0 blue, `RdBu_r`) it is mirrored —
  `assert cmap(0.0)[2] > cmap(1.0)[2], "label says blue=shallower but cmap is reversed"`.
  cmocean `balance` runs blue (0) → red (1), like `RdBu_r`; for any other map the assert
  reads `cmap(0)`/`cmap(1)` in code — never write the direction from memory (viridis and
  magma run dark → light). A count or density map labelled only by its units needs no
  ends assert. Pass cmocean maps as `cmocean.cm.<name>` — the string `"cmo.<name>"`
  works only after `import cmocean`.
  Non-linear scale near 0 (`BoundaryNorm`/`FuncNorm`).
- **Palette against the background.** Aim for ≥ 2 : 1 contrast of every bin against the
  ocean (≥ 2.5 : 1 for thin lines). Lines are checked against EVERY background they cross —
  the axes face and each shaded span (`axvspan`, `fill_between`); the weakest pair is printed.
  The script PRINTS the contrast of the map it draws against
  the ocean it draws and names the weak bins; it asserts only on a palette the skill chose —
  an owner-named palette is drawn as named, its weak bins reported as advice; an owner-named
  ocean with your own palette keeps a recipe's cut and its assert (the dark cut clears
  2.5 : 1 on dark navy too). The colour ↔ label assert above stays in
  every script. Recipes, the code block, contact sheets for undecided palettes and
  colour-blind-safe line colours: [references/palette.md](references/palette.md).
- **Two DIFFERENT density products, don't conflate:** `PROB` (presence integrated
  over 0→N yr — "stays home") vs `PROB10` (position at +N yr from each release —
  "where they end up"). Keep both, label them, use a FIXED colour scale when panels
  are compared.
- **Memory:** project only the frame's time window, never the whole trajectory
  array per frame (a render OOM'd at 2 GB/frame; windowing took it 5.5 → 3 GB);
  column-major Zarr read; the npz cache stores a track **subsample**
  (counts are all-particle); **lazy-load only the cohort keys you need** (a dense
  cache OOMs `load_cache`); render montages **one unit per fresh process** (matplotlib
  leak); a montage whose panels are compared shares ONE norm and ONE colourbar (no
  per-panel autoscale there); when the owner asks for per-panel scales, each panel gets
  its own colour bar and the suptitle says "own colour scale per panel". Its rows ×
  columns equal the requested count.
- **Execution target, by the render's COST (default; the owner may name another):**
  a render expected to take > 2 min (dense spaghetti, > 100 frames, big montages) runs
  on a CPU node — the usage block's first line is
  `srun --partition=<CPU partition> --cpus-per-task=8 --mem=32G bash render_montage.sh …`
  (a CPU node: `--gres`/`--gpus` buy nothing here); 1:400 drafts and ONE small gridded
  figure may render on the login node; a batch of variants counts its TOTAL time
  (6 variants × 90 s = 9 min → `srun`). `commands.sh` types the dense final once,
  uncommented, behind `srun`, with `--compare-old <old figure>` when it replaces one.
- **Labels:** put released (all) vs shown (sample) counts on videos; for pulsed runs
  note "releases only in yr 0–N" (the calendar-axis comet empties at the end because
  each track is only stored for its own window — artifact, not physics). The slot,
  in the frame loop, and the writer:
  ```python
  ax.text(0.01, 0.99, f"released {n_released:,} (all) · shown {n_shown:,} (sample) · releases only in {y0}–{y1}",
          transform=ax.transAxes, va="top", color="w")
  writer = imageio.get_writer(out_mp4, fps=fps, codec="libx264", ffmpeg_params=["-pix_fmt", "yuv420p"])  # imageio_ffmpeg's bundled binary
  ```
- **Figure contract (defaults for a publication or presentation figure; an exploratory
  look for the owner keeps only the checks — denominator, sign, labels):** one claim
  per figure; panel arithmetic first — decades × bands must equal the requested
  count (4 × 2 = 8, not 6); a POOLED dimension ("all seasons") is not a facet
  (1 decade × 2 bands, all seasons pooled = 2 panels); make ONE cut, name it in the PLAN and in the filename;
  every compared panel shares projection, extent, norm and colourbar; the panel
  title is written in the CODE (a title described in the plan and absent from the
  script reaches the report without its denominator) and carries the fields that
  apply — always the denominator (released / surviving for a +N-yr snapshot, released /
  shown for tracks) and data through / PARTIAL; season, band and τ when the figure has
  them. A snapshot panel:
  ```python
  ax.set_title(f"{season} · {band_m[0]}–{band_m[1]} m · τ = +{tau_yr} yr · "
               f"released {n_released:,} / surviving {n_surviving:,} · data through {data_through}{' — PARTIAL' if partial else ''}")
  ```
  (released N / surviving N — a +N-yr snapshot is survivors-only, so two
  similar-looking panels can hide a 60 % loss); gates/regions labelled on the map. A run fact
  the owner did not give (particles released, record interval, event definition) is read from
  the store or the run's files, or stamped `NOT GIVEN` in the caption; it never gates the final
  render. `data_through` from the store: `ds["time"][-1].dt.strftime("%Y-%m-%d").item()` (output
  format 2 decodes `time` to dates in the forcing's calendar; ages and intervals in seconds need
  `decode_times=False`); `units = "seconds"` has no dates — the forcing's first raw timestamp +
  seconds, or `NOT GIVEN`. A format-1 store (no `kelp_output_format`) has no epoch either, and its
  `time` is seconds only when the run interpolated in time (global attribute `temporal_interp = 1`);
  a DISCRETE format-1 run stores the forcing RECORD INDEX of the step's start — the date is the
  forcing's timestamp at that index (kelp-analysis "What the arrays mean"), else `NOT GIVEN`. Avoid
  in a figure that compares panels: averaging seasons without saying so, tiny decade×τ
  grids as a headline, grey spaghetti, normalise-to-own-max (an exploratory look the
  owner asked for may use its own scales, labelled as such). An AXIS LABEL is the quantity
  and its unit only; the kind of a time axis (release or event time) and the statistic's
  definition (first crossing, mean, sign convention, an age cut, partial decades) go once into
  the caption. An owner who asks for longer labels gets them:
  ```python
  ax.set_xlabel("year"); ax.set_ylabel("crossings per year")   # quantity + unit; axis kind and definitions → caption
  TOTAL_ON = "caption"   # caption | twin axis | own panel | component axis (only when the owner says SAME axis, or the total is the claim; say why)
  # "the total too / in the figure" → twin axis or own panel. The drawing branches on TOTAL_ON — no assert or raise pins one value.
  ```
  A pooled total drawn on the axis of its components squeezes them when it is several times
  larger — `TOTAL_ON` decides where it goes. A figure is a pure function of ONE named cache directory:
  stamp the cache path, the ENGINE commit that produced the data (run store attributes or
  ledger, never the clone that renders) and the render script's own commit with a dirty flag
  into the caption (`figure_helpers.write_caption` writes the stamp); commit the render script
  BEFORE the final render, and keep it next to the cache
  (`render_figs.py --cache-dir`), so the figure regenerates a year later.
- **Captions and superseded figures.** Every number or comparison in a caption comes from a
  table the render script wrote (formatted from its rows, never typed); the script writes
  `CAPTION_<name>.md` itself (`figure_helpers.write_caption`); a superseded figure is kept —
  new renders go to a new dated folder and the script builds the old-vs-new PNG
  (`figure_helpers.old_vs_new`).
- **A queued render job is re-read before it starts:** its command must match the
  product chosen since it was submitted; cancelling a pending job costs nothing.

## Adapting to a new run
Generators consume either the trajectory output directly (`(trajectory, time)`
`lon/lat/z/status` + `time` + `release_*`; format 1: `(trajectory, obs)`, no `release_*`) or a
compact cache. Point `--traj`/`--cache` at your
data; the drawing/colour/speed/camera code is dataset-agnostic — only the cohort
**loader** is run-specific (`pulse = traj_id // n_seed` for a pulsed release; drop it
for a single-release run). Generic helpers for the recipes above live in
`gallery/figure_helpers.py` (anomaly vs mean, lightness-spaced colours, pre-projected
tracks, inset view from points, palette contact sheet), imported as
`sys.path.insert(0, os.path.join(REPO, "gallery")); import figure_helpers` — if the
import fails, stop and report it; never write a stand-in for a helper (a stand-in
check that always passes is worse than none). Default import path:
`REPO = os.environ.get("KELP_REPO", "<absolute clone path>")`, never a path counted
from `__file__` parents. When data loading is stubbed, stub only the loader: the draw
calls and `savefig` stay live, so every draft writes its PNG. When you clean a generator
into generic-input form, put it in `gallery/` and update the catalog table.
