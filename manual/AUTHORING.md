# Project Reference Manual — Authoring Template & Methodology

A reusable instruction for producing a **code-grounded, book-style technical reference
manual** for any software project in this workspace. Hand this whole file to an agent as a
brief, or follow it yourself. It is deliberately project-agnostic: replace the bracketed
`<…>` placeholders and it applies to any codebase here (numerical model, data pipeline,
library, …).

It captures three things you asked for: **the request** (what to produce), **how to search
and walk the code**, and **how to write it** (including style).

---

## 0. When to use / inputs

Use this when you want a navigable, publication-quality reference that explains a project's
algorithms or numerics with formulas, ties every claim to the source code, and includes
purpose-built explanatory diagrams. Before starting, locate: the project root and source
tree, the project's own docs (`README`, `CLAUDE.md`, `docs/`, any subsystem↔code map or
experiment registry), the test suite, and any existing result figures worth reusing.

---

## 1. The deliverable (what to produce)

A modular LaTeX **book** that:

- explains every algorithm/numeric with the exact formulas **as implemented**;
- links each implementation claim to the source (`file:function` or `file:line`) through a
  short **"Source"** callout, so manual and code read side by side;
- has a purpose-built **schematic** for each non-trivial method, showing the geometry/data
  flow and **exactly which values enter the computation**;
- is built for **random access**: parts/chapters, table of contents, cross-references, an
  index, a front-matter "How to use this manual", and a source-code map;
- is **readable by a newcomer**: terminology is allowed but defined at first use; no terse,
  context-free statements that explain nothing;
- **mentions the critical tests** (e.g. comparison against a reference implementation,
  conservation/convergence checks) by name, *without* enumerating the full suite — most
  tests guard trivial invariants and are not individually informative;
- explicitly **flags where the project's own docs diverge from the code** (a second
  **"Code vs. documentation"** callout);
- **reuses existing figures** for validation/performance where relevant.

Default scope is the **engine/methods + code linkage**; keep experiment results out unless
the requester asks for them. State the scope decision up front.

---

## 2. Process & discipline (do **not** write it all at once)

1. **Orient.** Read the project's meta-docs and skim the source tree to learn the
   subsystems and the order in which data flows through them.
2. **Lock framing with ~3 multiple-choice questions** before investing effort:
   (a) scope — methods-only vs methods + results;
   (b) diagrams — hand-built schematics, reused figures, or both;
   (c) output — one modular book vs single file vs separate documents.
3. **Plan the structure and get it approved.** Propose the full part/chapter outline —
   for each chapter: its purpose, key formulas, the diagram it will carry, and its code
   anchors. Do not write prose until the structure is approved.
4. **Fill incrementally.** Write 1–2 chapters at a time in dependency order; **build after
   every batch**; keep the log clean. Track status in a `PROGRESS.md` with a checkbox per
   chapter and the fill order.
5. **Stay correctable.** Report after each batch so the requester can redirect early; fold
   any correction into the established pattern *before* producing more, so the fix
   propagates.

---

## 3. How to search and walk the code (the **source is the source of truth**)

- **Read the project's own docs first** (README, CLAUDE.md, `docs/`, any subsystem↔code
  map, registry). They are cheap orientation but never the final authority.
- **Decompose by subsystem and explore in parallel.** Dispatch **read-only** exploration
  agents, one per area (e.g. data/mesh loading; spatial methods; time integration;
  I/O & lifecycle; tests). Running them in parallel keeps raw file dumps out of your
  context — you keep the *synthesized, cited findings*, not the source.
- **Demand exactness from each explorer:** file paths + line numbers, function/class names,
  the **actual formula/algorithm as coded** (not the docs' wording), config-key→code
  mappings, sign/unit conventions, clamping rules, and degenerate-case fallbacks — and,
  for anything you intend to draw, the **precise stencil/data flow** (which entities supply
  which values and how they are combined).
- **Verify docs against code and record divergences.** Whenever the project's docs state a
  formula, default, or behavior, confirm it in the source and **flag every mismatch**,
  quoting doc-line vs code-line. These divergences are among the most valuable content in
  the manual.
- **Probe variants explicitly.** For every axis of variation (file formats, element/grid
  types, coordinate systems, compute backends, forward/backward modes, …), ask each
  explorer: *what differs across variants, and what is shared?* Reviewers care about
  exactly this, and it is where docs are most often wrong.
- **Structured returns, not prose.** Subagents return citations + formulas; the
  orchestrator synthesizes and writes only facts that trace to something actually read.
  Every formula and every `file:function` in the manual must be grounded.

### Reusable subagent prompt (issue one per subsystem)

> Read-only investigation of the SOURCE CODE (the code is the source of truth) for
> **`<subsystem>`** under `<repo>/src`. Your output returns to an orchestrating agent
> writing a manual with formulas, diagrams, and code links — return raw structured
> markdown with exact formulas (LaTeX-ish), `file:line` citations, and function names; no
> preamble, no "I found that…". For each item give the discrete formula **as implemented**
> (signs, units, clamps, degenerate-case fallbacks), the config keys that select it, and —
> for interpolation/geometry — the exact stencil (which nodes/cells/levels supply values
> and how they are weighted) so a schematic can be drawn. Then read the project's own docs
> for this subsystem and list **every** place the docs diverge from the code, quoting both.
> Read the actual source; do not guess.

---

## 4. Document structure (generic skeleton)

- **Front matter** — title; abstract; **"How to use this manual"** (the random-access
  guide); **"Notation & conventions"** (symbols, sign/unit conventions, the core domain
  objects and where data lives).
- **Part I — Foundations** — the problem statement (the governing equation or contract);
  the **architecture & per-step pipeline** (the map everything else hangs off) and the
  principal data structures.
- **Parts II…N — subsystems in pipeline order** — one chapter per subsystem, following the
  order data flows through the system (e.g. *locate → interpolate → integrate → apply
  constraints → record*). Group related chapters into parts.
- **Cross-cutting part** — input/configuration; output/persistence; **verification &
  validation**; **performance/cost**.
- **Appendices** — exhaustive **configuration/parameter reference** (tables); **notation &
  symbols**; **source-code map** (subsystem → `file:symbol` → manual section); **glossary**.

Each chapter is self-contained enough to be read on its own; the front-matter guide, the
code map, and the config appendix should let a reader reach any topic in about two lookups.

---

## 5. How to write it — style (peer-reviewed scientific article)

**Register & tone.** Formal, objective, restrained. No hype, no marketing, no exclamations.
Express judgment only through calibrated qualifiers ("non-trivial", "substantial",
"modest"), never emotion.

**Voice.** First-person-plural "we" for your own actions and claims ("We document…",
"We emphasize…"). Passive voice for procedures where the actor is unimportant ("the array
is padded with −1").

**Structure of prose.** One main claim per paragraph, stated first, then supported with
specifics (often "For instance," / "Similarly,"). Substantial paragraphs, not choppy
one-liners.

**Evidence.** Support claims with concrete numbers and units; give exact values and ranges
("RMSE reaches 5–13 °C", not "errors are large"). Reference and **narrate every figure and
table** in the prose ("(Figure 3.1)", "Table A.1 lists…").

**Calibration.** Match strength to evidence (suggests, indicates, likely, tends to). State
scope limits explicitly ("We characterize what the engine computes, not its physical
realism").

**Relation to prior work / external tools.** State what is known, the gap, and the
contribution; cite in author–year form, grouping multiple citations.

**Connectives.** Guide the reader: However, Notably, Importantly, Critically, In fact.

**Syntax.** Medium-length sentences with subordinate and parenthetical clauses; em-dashes
for asides, colons to introduce elaboration, parentheses for brief clarifications. Define
technical terms inline at first use. **No contractions** ("do not", "cannot").

**Tense.** Present perfect for prior work, past tense for what you did, present tense for
general truths and for interpreting findings.

**Per-chapter pattern.** Open with a claim-first overview of what the component does and
where it sits in the pipeline → develop the governing equations and the **discrete
formulae as implemented** → a **schematic** → a **"Source"** callout to the code. Add a
**"Code vs. documentation"** callout wherever the project's docs diverge from the source.

---

## 6. Diagrams

- **Schematics** (hand-built, e.g. TikZ): one per non-trivial method. Show the geometry or
  data flow and **exactly which values enter the computation** — for an interpolation, draw
  the cell, its nodes, the moving entity, and the weights, and label which value sits where.
  Adopt **one consistent visual convention** across all figures and state it once (e.g.
  node = filled circle, cell centre = square, level boundary = line, the moving entity =
  star).
- **Reused figures**: embed the project's existing **validation/performance** plots and
  describe each in the prose per the style guide. Keep experiment-result figures out unless
  results are in scope. View a figure before captioning it, so the description is accurate.

---

## 7. LaTeX engineering

- **Class**: `book` (parts/chapters); **modular** — `\include` one file per chapter, and
  use `\includeonly{…}` for fast single-chapter rebuilds while drafting.
- **Packages**: `amsmath`/`mathtools`, `siunitx`; `tikz` (+ libraries `calc, positioning,
  arrows.meta, shapes.geometric, patterns, decorations.*`), `pgfplots`; `listings`
  (dependency-free — prefer over `minted`, which needs shell-escape + Pygments);
  `booktabs` + `longtable` + `tabularx`; `graphicx` with `\graphicspath` pointed at the
  existing-figures directory; `hyperref` **then** `cleveref`; `natbib` (author–year);
  `makeidx`.
- **Macros**: a shared `notation.tex` (symbols); a `\codeanchor{…}` callout rendering a
  left-bar **"Source."** note; a `\codenote{…}` callout rendering a **"Code vs.
  documentation."** note; a TikZ style file fixing the visual convention.
- **Build**: a `Makefile` that pins the toolchain (hardcode the `texlive` `bin` path, or
  `module load` it) and runs `latexmk -pdf`. **Build after every chapter** and hold the bar
  at **0 errors, 0 overfull boxes, no undefined references or citations.**
- **Common pitfalls (observed in practice):**
  - `\degree` is **not** a valid bare math macro under modern `siunitx` — use `^{\circ}` in
    math (it is fine inside a `\SI{}{…\degree}` unit argument).
  - TikZ `calc` cannot divide a coordinate (`($(C)/3$)` fails) — write `($0.333*(C)+…$)`.
  - `longtable` columns need explicit wrap widths (`p{0.xx\linewidth}`) or long cells
    overflow; reword slash-joined option lists ("a/b/c") into comma lists so they break;
    a single benign "infinite glue shrinkage" note on a multi-page table is ignorable.
  - Underscores inside `\texttt`/key columns must be escaped (`var\_u`).
  - Overwriting a pre-created stub file with the editor requires reading it first.

---

## 8. Kickoff prompt (paste to an agent; fill the brackets)

> Produce a code-grounded, book-style LaTeX reference manual for the project at `<path>`,
> documenting its `<numerics/algorithms>` with formulas and linking every claim to the
> source (`file:function`). Treat the **code as the source of truth** and flag where the
> project's own docs diverge from it. Include a purpose-built schematic for each non-trivial
> method (show the geometry/data and which values enter the stencil), reuse the existing
> validation/performance figures, and mention the critical tests by name without listing
> the suite. Write in a peer-reviewed scientific-article register. **First** explore the
> code with parallel read-only subagents and **propose the chapter structure for approval**;
> **then** fill it one chapter at a time, building after each and keeping the log clean
> (0 errors, 0 overfull, refs resolved). Track progress in `PROGRESS.md`.

---

## 9. Definition of done

- Every chapter follows the pattern (overview → formulae as implemented → schematic →
  Source callout).
- Every formula and every `file:function` traces to source that was actually read.
- Docs/code divergences are flagged with the "Code vs. documentation" callout.
- The build is clean: 0 errors, 0 overfull boxes, all references and citations resolved.
- The front-matter guide, the code map, and the configuration appendix make any topic
  reachable in about two lookups.
