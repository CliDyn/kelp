# Developer notes — visualization (`visualization/`)

> **Covers:** the matplotlib helpers in `visualization/` and what they export.
> **Does not cover:** figure recipes and the figure contract (the `kelp-visualize` skill, `.claude/skills/kelp-visualize/`).
> **Code:** `src/kelp/visualization/core.py:TrajectoryDataset`, `src/kelp/visualization/plot_2d.py:plot_field`, `src/kelp/visualization/stats.py:compute_presence_density_mesh` (authoritative implementation).
> **Verified at:** see `docs/doc_map.yaml` (stamped via check_doc_sync --bless).

Developer notes, level 2 — index: [README.md](README.md). The text below is the module notes of `CLAUDE.md` as of v1.0.0, moved verbatim; "above", "below" or "the config section" in it may point into another file of this folder.

## visualization/
Matplotlib-based helpers, no core deps on them. Exports: `TrajectoryDataset` (load/access trajectory files), `plot_mesh`, `plot_mesh_nodes`, `plot_trajectories` (2D), `plot_diagnostics`, `plot_mesh_3d`, `plot_trajectories_3d` (pyvista/vtk when available, else matplotlib), `TrajectoryAnimator`, `compute_presence_probability`, `compute_status_ratios`, `plot_field` (scalar per node/element as colored mesh faces — node+all-tri = Gouraud tripcolor, else flat PolyCollection with node→vertex-mean; dateline-straddling faces masked from display; rotated meshes drawn geographic via `display_coords`), `compute_presence_density_mesh` (trajectory samples → containing ELEMENT via the real containment machinery — the unstructured twin of the uniform-grid presence probability; `normalize: count|probability|per_area`; geographic samples rotated into a rotated mesh's frame before containment).
- **Endpoints and diagnostics on format-2 output (2026-11)**: `TrajectoryDataset` keeps the optional `release_lon` / `release_lat` of the data dict (sliced by `filter_status`); `plot_trajectories`' start marker is the release position where finite, else the particle's first finite record (a delayed release records NaN before it), the end marker its last finite record (`plot_2d._endpoints`). `plot_diagnostics` wraps each `dlon` to [-180, 180) before the speed and takes the time axis and the speed intervals from `ds.time` when present (`|diff|`, so backward runs count down), `dt` only without record times.
