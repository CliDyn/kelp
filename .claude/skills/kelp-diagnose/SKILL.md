---
name: kelp-diagnose
description: Use when a KELP run is slower than the documented cost, silent or apparently hung, scales anomalously with mesh size, shows a step cost in milliseconds per particle, or when a proposed speed fix (elem_search walk, a smaller grid-hash cell, exact_nearest off, a cache wipe) needs vetting — BEFORE changing code or config, and before any "wall" or "scaling limit" goes into a report.
---

# Diagnosing a slow, silent or anomalously scaling KELP run

Every node below is a measurement plus a decision rule. Quote a measured number
from `docs/performance.md` or CLAUDE.md, or label a number
"unmeasured" — never invent thresholds. Write BOTH outcomes of every
measurement. Never write a tool's output — not as an assumption, not as an
"example": copy its column names and leave the values blank; an example
reading becomes an invented cause and, quoted with a source, a fabricated
record. A performance anomaly is a **bug hypothesis until diagnosed**:
a "×24 zoom-10 wall" once was a fallback code path and nearly entered a
report as physics.

## Start here: symptom → path

| Symptom | Path | Hard rule at the branch |
|---|---|---|
| Same config, same N, the step cost moved with MESH SIZE (`io=` small) | 1 → 2 → 5 → 6 → 7 → 12 | **Node 3 is SKIPPED on a quasi-uniform mesh** (HEALPix, CORE2, structured): the auto cell is 2× the mean node spacing (`max(0.05°, 2·sqrt(area/N))`), so the 3×3 window holds the same few dozen nodes at every zoom — a zoom change cannot change occupancy (the floor bites only below ~0.025° spacing). The density certificate gates the 2-D barycentric kernels only; the fused 3-D RK2 kernel has none. Do not run `check_grid_hash_cell.py` here — not as a diagnostic, not "to confirm the rule", not "revisited" after the profile: it is harmless and it is still a detour into the wrong hypothesis. The zoom-10 "wall" was `_splice_exact_rerun`, found by 5→6→7. |
| MILLISECONDS per particle per step on a multi-resolution / structure-resolving mesh | 1 → 2 → 3 → 4 → 12 | Node 3 first. `walk` cannot fix an anchor cost; a smaller cell is bit-identical with `mesh.exact_nearest: true` (the default). |
| Silent job, empty log, ~2–3 cores busy on a many-core node | 1 → 8 → 9 → 10 (→ 11 if the signature outlives the compile) | Judge by the output store; resubmit elsewhere BEFORE any deep-dive and leave the original running (it is the control); never kill, never signal, never attach `strace`; never wipe the numba cache for silence — a stale cache crashes, it never stalls (wipe only for NRT/boxing errors or after editing an inlined callee). |
| A `prange` kernel on top of the profile, ~1 core busy | 7 → 11 | Memory order / dtype at the reader boundary. |
| Another tracker, or another KELP version, is N× faster PER CORE | 1 → 13 → 7 → 12 | No per-core number without the pinning evidence in each log header; list the work per stage of both paths before blaming "the algorithm". A leg whose header shows `cpus=` > 1 or `time -v` > 110 % CPU is multi-threaded: it supports neither parity nor a gap, in either direction. |
| A phase (`io=`, `rec=`) got slower between two runs of the same setup | 14 | Not a regression until split per event; the mean alone decides nothing. |

If you find yourself at node 3 on a HEALPix or CORE2 mesh, you have left the
path — go back to the table ("to confirm", "revisited" and "harmless, metadata
only" included).

## The tree

1. **Phase split first.** In a multi-process layout with `shared_fields_dir`, first
   `grep -l "shared_fields disabled" <all process logs>`: a process that fell back
   reads its own snapshots, so that leg's memory and `io=` numbers are mixed (3 of 16
   processes fell back once, 16 × 8 on one node). Then `grep -E "io=|step=|rec="` on the `-v` log (or a 3–5-step
   probe with `python -u`). `step=` dominant → 2. `io=` dominant → the reader
   branch (snapshot cache, prefetch, `shared_fields_dir` placement, float32
   gather). `rec=` dominant → the writer branch (chunking, `output_freq`).
   Nothing printed at all → 8.
2. **Normalise, then classify the SYMPTOM before the number.** `step_ms /
   n_alive`. References (docs/performance.md §1–2), per particle per step: 2-D
   at 16 threads euler ≈ 1.1, rk2 ≈ 6.6, rk4 ≈ 22 wall-µs; fused 3-D RK2 at 128
   threads 0.63–0.91 s per step for 6.0 M particles on zoom 5–11 meshes
   (2026-09-24) = 0.10–0.15 wall-µs = 12–19 thread-µs (divide thread-µs by the
   thread count before comparing with `step_ms / n_alive`). Same N and
   config but the cost moved with the MESH → 5 (never 3). Milliseconds per
   particle on a multi-resolution mesh → 3. Microseconds but above the reference
   → 5.
3. **Grid-hash cell — multi-resolution meshes only.** Precondition: the loader
   warned `grid hash: auto cell size … leaves up to N nodes` (a warning about the
   reader's legacy cell — the run's own choice is the `grid hash: cell … by
   occupancy/…` line), or the mesh spans decades of resolution; otherwise → 4.
   `python utils/check_grid_hash_cell.py config.yaml` prints one table per query
   set (`-- query set: seeds`, then the element centres) — `cell (deg) | nodes per
   3x3 window: median | p90 | max | empty windows | approx. anchors` — then one
   `recommendation:` line (`set mesh.grid_hash_cell_size: X (…)`, or `auto cell …
   is fine` when the auto cell's median is ≤ 30 nodes) and `the run would use:`;
   those lines are all you may quote. 10⁴–10⁵ nodes per cell =
   the 0.05° floor (a structure-resolving mesh, 1.4 m–1.4 km: 36,000 nodes, 2.7 ms per
   particle) → set `mesh.grid_hash_cell_size` to the recommended
   cell (the smallest with no empty window, a median ≥ 8 nodes and, under the
   guard, at most 10 % widened anchors), keep `mesh.exact_nearest: true`. The AUTO
   cell's anchor is exact (slow, not wrong); only a cell below the local spacing
   can be approximate, and the guard makes that exact too — old and new runs must
   be bit-identical except at exact ties and for queries beyond the search budget. (100–135 nodes
   per window on a regionally refined global mesh is a 25–45 % tuning, not this symptom.)
   Re-probe → 12.
4. **The `walk` gate** (before anyone proposes it): `grep -E "uv_interp|elem_search"
   config.yaml`. `walk` removes the per-step CONTAINMENT search, not the
   nearest-node ANCHOR that `leastsq` fits its plane over; it pairs only with
   `barycentric` (node or element layout) / `edge_leastsq` (element only), runs on
   the unfused paths (drops the fused kernels) and with `leastsq` warns once and
   does nothing. Anchor cost (`_grid_hash_nearest*` on top in 7) → back to 3 on
   a multi-resolution mesh ONLY (on HEALPix/CORE2 an anchor frame on top is not
   occupancy — node 7); containment cost (`_find_containing_element*` on top) →
   walk is legitimate.
   Eliminated as a fix for exactly such a slowdown once, for both reasons.
5. **Small-N correctness probe — same mesh, other commit.** Same config, same
   MESH, 50–100 particles, 3 steps, on the current commit and on a commit known
   good for this mesh (the one that produced the reference leg, or `main`
   before the suspect change); `np.array_equal` on lon/lat/z/status. Differs →
   a correctness bug, stop performance work. Identical → 6. Never compare two
   MESHES bit for bit: zoom 9 and zoom 10 differ legitimately (a remapped twin
   validates against its native mesh to a distance over a month, never to
   identity), so a cross-mesh difference is not a bug signal and a cross-mesh
   "identical" proves nothing.
6. **N-scan.** 3 steps at N = 10², 10³, 10⁴, 10⁵ particles, same config. `step=`
   FLAT in N → a fixed per-step path (whole-field lerp/recon, the unfused splice,
   a KD heal) → 7. `step=` ∝ N → per-particle cost → 3 (multi-resolution mesh
   only) or 11.
7. **py-spy the live process** (`py-spy dump --pid
   <pid>` three times, or `py-spy top` — py-spy from the run's own environment,
   its `bin/`; it is rarely on `PATH` in a batch job). Name the top frame. A fallback on top
   (`_splice_exact_rerun`, a whole-field `lerp`/`reconstruct_pchip`,
   `find_nearest_nodes` heal) → count the fraction of particles that triggers it:
   a whole-grid cost for a tiny fraction is a DEFECT (zoom-10: ~30 s/step for ONE
   flagged particle; fixed by re-running just the flagged subset) → fix →
   12. A `prange` kernel on top with ~1 core busy → 11. A fused kernel on top
   with every core busy and the cost still moved with the mesh contradicts
   performance.md §2 (the steady fused 3-D step is nearly mesh-size-independent:
   6.0 M particles, 0.63–0.91 s at 128 threads from 9.4 k to 35.5 M wet nodes,
   2026-09-24) → diff the two configs (`max_levels`/K, `w_interp`, `uv_interp`,
   `temporal_reconstruction`, `elem_search`); identical → a NEW anomaly: write
   the N-scan and the three dumps into the campaign's report and stop — it is not a
   "scaling characteristic" and it does not go into a publication.
   `_grid_hash_nearest*` on top on a quasi-uniform mesh is NOT occupancy (the
   window holds the same few dozen nodes at every zoom): it is the exact-search
   widening in a latitude band (performance.md §1: +25–75 % at 80–88 °N, ×4.1
   at 89–90 °N, 0.02 % of a global release) or off-mesh seeds — name the band
   and the fraction → 12; a smaller cell cannot fix it.
8. **Silent job — is the store growing?** `stat -c %s out.nc`, or the record count
   in `out.zarr/time/zarr.json`, twice, minutes apart. Growing → the log is
   block-buffered, the run is healthy — **STOP, nothing to diagnose** (`python
   -u` next time). A running job's buffer cannot be flushed from outside: never
   signal the process (Python has no USR1 handler; `kill -USR1` terminates it),
   never attach `strace`. Not growing, or no store yet → 9.
9. **CPU signature on the node** (`top -H`, `ps -o %cpu,etime -p <pid>`). A running
   `prange` kernel on a 128-thread allocation shows ≥ 80 % × cores (~10,000 %).
   ~100–300 % (2–3 cores) = a numba COMPILE (user-space LLVM, few syscalls) or a
   serial path — never the parallel kernel, not a "memory-bound kernel" → 10.
   Kernel-level CPU → the step is merely slow → 1. ~0 % = truly blocked (I/O,
   lock, dead reader) → `py-spy dump`. Never kill on "no syscalls".
10. **Resubmit FIRST, then the numba checklist.** Propose the identical job on
    another node/partition as the first step and submit it once the owner says yes
    (like every launch); leave the original running — it is the control for the
    comparison. A cold compile is ~50 s; the 30-min case
    (2026-06) was a bad node — the same job compiled instantly elsewhere. Same
    stall elsewhere → in order:
    - `NUMBA_WARNINGS=1` (object mode = ×100 slower);
    - `NUMBA_DISABLE_JIT=1` on a 50-particle repro (a Python error surfaces as a
      clear traceback);
    - `cache=True` present on the hot kernels;
    - count the fused combinations being compiled (integrator × tinterp/pchip ×
      w_interp × uv_interp) and BISECT those knobs — a compile that outlives
      ~50 s on a healthy node is never "expected, just wait";
    - same stall elsewhere AND py-spy shows a KELP `prange` frame, not LLVM
      → 11.
11. **Serial-fallback / memory-order check.** After any fancy-index compaction or
    transpose at a reader boundary, check `arr.flags['C_CONTIGUOUS']` and dtype:
    Fortran-ordered or float32 fields make the `prange` blends (`_blend`,
    `_pchip_blend4`) fall back to serial numpy — 1.08 of 16 cores busy, values
    bit-identical, "only the speed lied" (measured: 25 s/step at
    zoom 8). Fix at the boundary with `np.ascontiguousarray` + a contiguity test.
12. **Prove it and record it.** Rerun a real leg with and without the change;
    `np.array_equal` on lon/lat/z/status — no `allclose`, no tolerance, no
    "close" category.
    Bit-identical by contract: a smaller cell with the guard on, a contiguity
    fix, a subset re-run, a resubmit. The ONE documented difference is
    guard OFF vs ON (405 of 2.4 M trajectories after a model year, one status of
    7.9 M — performance.md §1); it is not a tolerance for anything else. Any
    other difference means the fix is wrong. Write the ELIMINATED hypotheses
    with their evidence into the campaign's report (zoom-10: empty
    windows 0.00 %, splice storm, cache size, bus contention — all ruled out
    before the cause); every "same results, faster" change ships with an
    identity test.
13. **Per-core comparison hygiene (across tools or engine versions).** First node 1:
    split the phases and note the `step=` share only. The per-particle-step comparison
    waits for the pinned legs; never compute a ratio, per-thread or per-core number from
    an unpinned leg — not even a "naive" one. A tool that preloads its fields has no
    per-step input, so wall-clock totals are not a kernel comparison. The pinning recipe (whole process on one
    affinity CPU, BLAS threads 1 for both codes, an in-process thread print), the work-per-stage
    table with the KELP row filled, and the five legs to prepare are in
    [references/per-core-comparison.md](references/per-core-comparison.md); fill the table and
    prepare the legs before any sentence about "the algorithm", then profile (7) → 12.
14. **Phase regression between two runs.** Split the phase per EVENT: `rec=` on the
    record steps, `io=` on the snapshot-read steps; for both runs print median, p90,
    p99 and max per event, and `git log <old>..<new> --` the code path of that phase.
    Median shifted AND code changed → a code regression: bisect. Median unchanged, tail
    grown → environment (file-system load). Median shifted, code unchanged → run the
    two commits back to back on one node (an A/B leg) before claiming anything. One
    year run: output median 6.6 vs 4.8 s per record, tail to 45 s, writer code
    unchanged — cause not established; it was reported as open, not as a regression.
