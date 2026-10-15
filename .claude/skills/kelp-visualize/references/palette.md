# Palettes against the background, contact sheets, line colours

Read when choosing or checking a palette or a background, offering palette variants, or picking N
line colours. The colour ↔ sign ↔ label assert of the skill stays in every script.

- **Palette against the background — both ends and the centre.** Aim for ≥ 2 : 1
  contrast of every palette bin against the ocean colour (≥ 2.5 : 1 for thin
  lines): a dark ocean hides the dark end of a diverging track palette, a
  white ocean hides its pale centre — compute the contrast (WCAG) per bin, do not judge
  by eye. When the owner has not chosen a style, start from these recipes, which passed
  for a depth-deviation track palette (`RdBu_r`, 13 bins):
  DARK = neutral near-black ocean (`#0f1113`; navy's blue cast swallows blue tracks),
  ramp sampled 0.12–0.40 and 0.60–0.86 (the darkest 12 % per side dropped), a light
  centre bin (`#f7f7f7`) for |dz| < 10 m, lines 0.5 pt, alpha 0.95; WHITE = ramp
  0–0.27 and 0.73–1.0, grey centre (`#6e6e6e`), lines ≥ 0.55 pt, alpha 1.0 (raise
  weight and alpha before touching the palette; boost chroma in OKLCh, never HSV).
  Filled maps too: an anomaly map's zero centre and a density map's lowest bins (few or
  no particles) may fade into the ocean; the top bins should not. The script prints the
  contrast of the colour map it DRAWS against the ocean it DRAWS and names the weak
  bins; it asserts only on a palette the skill chose (a recipe here). Only the contrast
  assert depends on who chose the palette; the colour ↔ label assert of the skill stays in every
  script. A palette you picked yourself (neither the owner's nor a recipe here): print the
  contrast and change your own background or map until the top bins reach 2 : 1 — on a
  quick look a print, not an assert:
  ```python
  STYLE_FROM_SKILL = True                                 # False only when the owner named the palette itself
  OCEAN = "#0f1113"; ax.set_facecolor(OCEAN)              # the colour you draw is the colour you check
  base = plt.get_cmap("RdBu_r")                           # dark-ocean recipe: the darkest 12 % per side dropped
  cmap = ListedColormap(np.vstack([base(np.linspace(0.12, 0.40, 6)), to_rgba("#f7f7f7"),
                                   base(np.linspace(0.60, 0.86, 6))]))
  norm = BoundaryNorm([-200, -150, -100, -70, -50, -30, -10, 10, 30, 50, 70, 100, 150, 200], cmap.N)   # 14 edges = 13 bins = 13 colours
  c = figure_helpers.palette_contrast(cmap, OCEAN)        # the cmap passed to the draw call
  print("palette contrast vs ocean:", " ".join(f"{x:.1f}" for x in c), "| weak bins (< 2.5):", np.nonzero(c < 2.5)[0].tolist())
  # the colour ↔ label assert of the skill stays, whoever chose the palette
  if STYLE_FROM_SKILL:
      assert c.min() >= 2.5, "a bin of the skill's recipe vanishes into the ocean"
  ```
  An owner-named ocean does not name the ramp: with the owner's ocean and your own
  palette, keep a recipe's cut (on a dark ocean 0.12–0.40 / 0.60–0.86 with a light
  centre) and say so. An owner-named palette is drawn as named, uncut; its weak bins go into the handback as advice (never into the caption)
  (for example a truncated variant beside it), never as a silent swap. Changing only the
  ocean never rescues a dark end: on any dark ocean the full RdBu_r ends measure about
  1.4 : 1 (on black 1.6). WHITE-ocean recipe: `OCEAN = "#ffffff"`, the ramp sampled
  0–0.27 and 0.73–1.0 with the grey centre `#6e6e6e`, checked against 2.0. A filled
  anomaly map on white checks its two wings only (`lo=0, hi=0.27` and `lo=0.73, hi=1`);
  the band between them is the zero centre that may fade.
  When the owner says a version "was better before", name what the old one did right
  (background lightness? hue?) before changing anything.
- **When the palette is undecided, offer it as ONE contact sheet** (4–6 variants of the
  same figure, same data, labelled; `figure_helpers.contact_sheet`) and let the owner
  pick, rather than iterating one palette at a time; when the owner has named the
  palette, draw that one (an optional variant may go beside it). Every variant keeps one sign convention and label (never a map
  beside its `_r` twin), is written to its own `<name>__<variant>.png`, and is rendered
  in its own process. One label means one hue pair: a "red = more" label allows red–blue
  variants only; for other hues write each variant's label from its own end colours.
- **N line colours, colour-blind safe:** sample a perceptual sequential map
  (`cmo.deep`, `cmc.batlow`) at N points of evenly spaced CIELAB lightness L*, not
  at evenly spaced map positions (`gallery/figure_helpers.py::lightness_spaced_colors`);
  the lines stay distinct in greyscale. Lines are checked against EVERY background they cross
  — the axes face and each shaded span (`axvspan`, `fill_between`); the weakest pair decides
  (a light line that passes on white can fall below 2 : 1 on a grey band). Lighter shading or a
  darker light end of the ramp fixes it:
  ```python
  for bg in (FACE, SHADE):                                   # every background a line crosses
      c = figure_helpers.palette_contrast(ListedColormap(line_colors), bg, n=len(line_colors))
      print(f"line contrast vs {bg}: " + " ".join(f"{x:.2f}" for x in c) + f" | weakest {c.min():.2f}")
      if STYLE_FROM_SKILL:
          assert c.min() >= 2.5, f"a line fades on {bg}"
  ```
