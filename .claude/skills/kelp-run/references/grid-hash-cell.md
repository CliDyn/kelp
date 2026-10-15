# The grid-hash cell check before a run

Optional since 2026-09-22 (the run chooses the cell and stamps it in the output metadata). Read when the
release sits in a refined region of a non-uniform mesh or an explicit cell is wanted. The campaign-level
recipe is in the kelp-experiment skill (references/grid-hash-cell.md).

- **Grid-hash cell — an optional check since 2026-09-22** (the run picks the cell
  and stamps it in the output metadata). Run it when the release sits in a refined
  region of a non-uniform mesh or an explicit cell is wanted; quasi-uniform meshes
  (CORE2, the HEALPix twins) keep the automatic cell.
  The nearest-node search scans a 3×3 window of hash cells twice per RK2 step.
  Two failure modes, opposite directions: a cell too LARGE for the local node
  spacing (the auto cell comes from the mesh-wide mean spacing, so a mesh
  refined where the particles are — a regional refinement, a coastal focus — gets 100+
  nodes per window and the scan takes 25–45 % of the tracking phase, with no
  loader warning); a cell too SMALL leaves the TRUE nearest node outside the
  window for some positions ("approximate anchor": the interpolation stencil
  changes and those trajectories diverge — a window can be non-empty and still
  miss it, especially at high rotated latitude where a degree of longitude
  shrinks). Run, from any directory:
  ```bash
  python <clone>/utils/check_grid_hash_cell.py path/to/config.yaml   # seeds = data.inicloud; add --cells ... to widen (optional since 2026-09-22: the run chooses the cell itself and stamps it in the metadata)
  ```
  It probes the seeds AND the element centres of the whole mesh (where the
  particles may drift) and reports the nodes per window and the share of
  "approximate anchors" (true nearest node outside the window) for the auto
  cell and candidates. **Since 2026-09-14 `mesh.exact_nearest` is on by
  default** (docs/localization.md): every anchor is exact whatever the cell —
  an approximate window hit only costs a widened in-kernel search — so the
  choice is about SPEED only: the tool recommends the smallest cell with no
  empty window, ≥ 8 nodes per window and at most 10 % widened anchors
  (`--allow-approx` overrides the share). Put the value in the config as
  `mesh.grid_hash_cell_size` (degrees). Measured on a global mesh refined in the release region:
  the auto cell leaves 100 nodes per window and 0.005 % widened anchors, 0.5°
  leaves 17 per window and 0.1 % widened — the daily step 1.13 → 0.75 s on one
  node (2026-09-12, guard off; with the guard the 0.5° result is exact and
  equals the auto-cell result bit for bit). Runs that must reproduce a
  pre-2026-09-14 output bit for bit set `mesh.exact_nearest: false` AND
  `mesh.grid_hash_cell_rule: legacy` (or pin the old run's `grid_hash_cell` from its
  metadata) — since 2026-09-22 the default rule may change the cell too.
