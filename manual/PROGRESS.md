# KELP manual — build progress / plan of record

Project: `manual/` — a `book`-class LaTeX reference for the KELP
engine (numerics + source linkage). Build: `make -C manual` (pins the
TeX Live on PATH or `make TEXBIN=<texlive bin dir>`; runs `latexmk`).

Style: peer-reviewed scientific register inside a navigable book structure.
Diagrams: TikZ schematics for methods + reused method/validation PNGs from
`../plots`. Scope: numerics + code reference (no experiment results).

## Status — v2 (2026-09-24)

All chapters rewritten or checked against the engine at the September 2026 state (exact nearest-node
guard, cell rule by measurement, node table, edge least squares, edge walk, W sign contract, sigma
s-integration, sphere kicks and adjoint drift, vertical Az, RK45, HEALPix and NEMO readers, shared snapshot
cache, resumable chains, the 2026-09 performance measurements) and a new chapter 18 on the agent skills.
`main.pdf`: 106 pages, 0 overfull boxes, all cross-references and citations resolved; 27 TikZ figures,
2 pgfplots charts, 1 reused result PNG.

- [x] Front matter: title, abstract, how-to-use, notation & conventions
- [x] Ch 1 — Introduction
- [x] Ch 2 — Solver architecture & the per-step pipeline  *(pipeline TikZ)*
- [x] Ch 3 — The unstructured mesh  *(mesh, column, mesh sources; nod_in_elem2D rebuild; convexity)*
- [x] Ch 4 — Particle localization  *(grid hash, exact certificate, containment, edge walk)*
- [x] Ch 5 — Horizontal velocity interpolation  *(barycentric, leastsq, edge leastsq, Wachspress)*
- [x] Ch 6 — Vertical structure & W interpolation  *(W sign, column, sigma-following W, sigma step)*
- [x] Ch 7 — Scalar sampling
- [x] Ch 8 — Coordinate frame & rotation  *(incl. HEALPix)*
- [x] Ch 9 — Time-integration schemes  *(RK4 stages, RK45)*
- [x] Ch 10 — Temporal reconstruction  *(linear vs PCHIP timeline)*
- [x] Ch 11 — Fused vs unfused kernels  *(kernel table)*
- [x] Ch 12 — Stochastic diffusion  *(sphere kicks, vertical Az)*
- [x] Ch 13 — Particle lifecycle & boundary conditions  *(state machine, periodic windows, beaching)*
- [x] Ch 14 — Forcing input & time control  *(shared cache figure)*
- [x] Ch 15 — Trajectory output & restart  *(chunking, resumable chain figure)*
- [x] Ch 16 — Verification & validation
- [x] Ch 17 — Computational cost & performance  *(budget, layouts, mesh/particle/node scaling)*
- [x] Ch 18 — Agent skills  *(routing figure)*
- [x] App A — Configuration key reference (full tables + command line)
- [x] App B — Notation & symbols
- [x] App C — Source-code map
- [x] App D — Glossary

## 1.1 additions (2026-10-21): FESOM 1.4 reader, depth-linear horizontal velocity

Methods and code linkage only (no experiment results); 2 new TikZ figures, 5 new tables.

- [x] Front matter, Ch 1 — FESOM 1.4 in the FESOM family (a P1 model, fields on the levels at the nodes), the scope and the test count
- [x] Ch 3 — §3.5 FESOM2 element bottoms (`nlevels`/`zbar_e_bottom` → `elem_bed`, `uv_data_last_elem`; partial cells; the region tool); §3.7 the FESOM 1.4 mesh (ASCII directory, `input_rotation`, triangles from cell polygons *(TikZ)*, `deptho`, cavities planned)
- [x] Ch 5 — Common properties: depth first, then the horizontal combination; the wall rule per scheme; FESOM 1.4 in the layout table
- [x] Ch 6 — §6.4 Horizontal velocity in depth: layers or linear (placements, the data-point table, bracket and blend, walls at depth per interpolator and the fallbacks, time and stages, sigma meshes in `s`, FESOM 1.4 prism-wise, the record check, refusals) *(TikZ: column stencil with a wall)*
- [x] Ch 11 — the depth-linear fused twins (one paragraph)
- [x] Ch 14 — the FESOM 1.4 reader in the reader layer, the `prepare` hook; record times resolved by the reader and why `data.time_averaging` is refused
- [x] Ch 15 — the `uv_vertical` attribute; one vertical rule per trajectory (checkpoint, resume, writer)
- [x] Ch 16 — §16.3 the depth-linear analytic suite, its sigma counterparts and mutation pass; the published FESOM 1.4 data checks
- [x] Ch 17 — the cost of the depth rule per 3-D step (table from `benchmarks/single_node/depth_linear_cost.md`)
- [x] App A — `mode.uv_vertical`, `mesh.type: fesom14`, `mesh.input_rotation`, `data.time_stamp`/`time_window`/`time_offset`, the fesom14 note on `data.time_averaging`
- [x] App B, App D — the new symbols and glossary entries
- [x] App C — the new source files and a symbol table for the depth-linear rule and the FESOM 1.4 reader

## Bundled result figures (`figs/`, self-contained)
The manual embeds one reused result PNG, `revtest_roundtrip.png` (Ch 16, a June 2026 measurement,
labelled as such); every other figure is inline TikZ or pgfplots.
