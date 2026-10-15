# Continuing a finished run, or moving a restart to another mesh

Read when a run continues after it finished, or a restart is reused on another mesh (crop ↔ full).

- A restart written on one mesh cannot seed a run on another (crop ↔ full):
  `nearest_node`/`restart_elem` are indices into the mesh that wrote them.
  Re-derive the anchors on the new mesh, or convert the last valid record
  of the (geographic) trajectory output into an `inicloud` file. Under
  `mesh.rotation` the RESTART stores the internal ROTATED state while the
  trajectory OUTPUT is un-rotated at record time: re-anchor a restart
  against the rotated mesh as-is — never un-rotate it, never feed rotated
  restart coordinates into a seed file; only the OUTPUT record is geographic
  and safe as an `inicloud`. The re-anchoring is this line in `prepare_restart.py`:
  ```python
  nn = find_nearest_nodes(mesh_new, st.lon, st.lat)   # MESH-frame coords as stored: never un-rotated, never via an inicloud
  st = st.replace(nearest_node=nn, elem=None)          # elem re-seeded by the run (elem_search: local)
  ```
  `exact_nearest` lives under `mesh:` only — a copy under `numerics:` is
  silently dropped by the loader. A continuation never goes through
  `kelp run`: the CLI has no resume, and `data.inicloud` can never point at
  a restart `.nc` — the chain driver loads it with `load_restart` (kelp-experiment,
  `references/campaign-templates.md`). Record both sides' provenance before resuming:
  the finished run's `exact_nearest` and `rotation_angles` writer-metadata
  attributes and its engine commit, and the same for the continuation — a
  flipped `exact_nearest` changes anchors, a different commit changes
  kernels; either makes "same run, longer" false for a second reason. Crop →
  full multiplies the INPUT cost ×10–14 (tracking ×1.2–1.5): re-probe the
  layout before sizing `--time`; `CHECKS.md` carries this row verbatim:
  `| crop → full | input ×10–14, tracking ×1.2–1.5 | re-probe layout before sizing --time |`
