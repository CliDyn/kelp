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

## Bundled result figures (`figs/`, self-contained)
The manual embeds one reused result PNG, `revtest_roundtrip.png` (Ch 16, a June 2026 measurement,
labelled as such); every other figure is inline TikZ or pgfplots.
