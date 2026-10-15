# KELP — Numerical Methods and Implementation: Reference Manual

A code-grounded, book-style LaTeX reference for the KELP engine: every
algorithm and numeric is stated as the discrete formula **as implemented**, each
claim linked to the source (`file:function`), with a purpose-built schematic for
each non-trivial method. Scope is the engine's numerics and code linkage (no
experiment results; engineering measurements of speed and memory are in chapter 17), plus a
final part on the agent skills (chapter 18). The built `main.pdf` is ~106 pages.

## Build

```
make          # runs latexmk -> main.pdf (TeX Live on PATH, or make TEXBIN=<texlive bin dir>)
make clean    # remove LaTeX build artifacts
```

Requires TeX Live 2025 on PATH, or pass its binary directory as `TEXBIN` (equivalent to
the TeX module your cluster provides). Hold the bar at **0 errors, 0
overfull boxes, all references and citations resolved**.

## Layout

- `main.tex` — document root (parts → `\include`d chapters); `preamble.tex`,
  `notation.tex`, `style/` — packages, symbols, and the shared TikZ conventions.
- `frontmatter/`, `chapters/` (01–18), `appendices/` (A–D) — one file each.
- `figs/` — the one reused result PNG (`revtest_roundtrip.png`); every other figure is inline TikZ
  or pgfplots.
  `\graphicspath` is `{figs/}`, so the manual builds standalone.
- `refs.bib` — bibliography. `PROGRESS.md` — chapter checklist / plan of record.

## Updating the manual

The **code is the source of truth**. When the engine changes, re-verify the
affected chapter against the current source and its `docs/` topic doc, update the
formula and the `file:function` anchor, and rebuild clean. `AUTHORING.md` records
the full methodology used to produce this manual (parallel read-only code
exploration, document structure, scientific-article style, and the LaTeX
engineering pitfalls); follow it for any substantial revision or to regenerate a
chapter from scratch.
