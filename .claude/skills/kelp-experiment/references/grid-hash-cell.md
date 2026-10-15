# Grid-hash cell on non-uniform meshes

Read when the release sits in a refined region of a non-uniform mesh, the loader warns about the auto
cell, or `step=` costs milliseconds per particle.

**Since 2026-09-22 the run picks the cell itself** (`mesh.grid_hash_cell_rule:
occupancy`, the default; the choice is printed as `grid hash: cell ...` and stamped
in the output metadata) — the check below is for an EXPLICIT cell and for the seed
positions (the rule samples the mesh, not the seeds): `python
<clone>/utils/check_grid_hash_cell.py path/to/config.yaml` prints, for the auto
cell and candidates, the nodes per 3×3 window and the share of anchors whose
window does not hold the true nearest node, at the seeds and at the element
centres of the whole mesh. The ordinary case is a mesh refined where the
particles are (a global mesh refined regionally, a coastal focus): the auto
cell leaves 100+ nodes per window WITHOUT any loader warning and the window scan
took 25–45 % of the tracking phase of a regional run; 0.5° gave the daily
step 1.13 → 0.75 s on one node. **Since 2026-09-14 `mesh.exact_nearest` is on by default**
(docs/localization.md): an anchor whose window cannot prove it is resolved by an
exact in-kernel search, so a small cell no longer trades exactness for speed —
the cell 0.5° result equals the auto-cell result bit for bit, and the tool's
"approximate anchor" share is the widening COST (it tolerates 10 % by default).
Before the guard, that run at 0.5° had 2.7 % approximate anchors and 0.16 % of
the statuses changed over 91 days; reproduce such pre-guard outputs with
`mesh.exact_nearest: false` plus `mesh.grid_hash_cell_rule: legacy` (or the old run's
pinned `grid_hash_cell`). The extreme case below is the same mechanism at
the 0.05° floor.

The nearest-node grid hash assumes quasi-uniform meshes. On a mesh whose
resolution spans orders of magnitude (coastal/structure-resolving), the auto
cell (`max(0.05°, sqrt(area/N)·2)`) is wrong in BOTH directions, so choose it
from the data (found on a mesh resolving metre-scale structures, 1.4 m–1.4 km in
one domain, where the floor made a day-long micro-run take 12 min instead of 20 s):

1. **Diagnose**: the loader warns about the READER's legacy cell (`grid hash: auto
   cell size … leaves up to N nodes in one cell`) with a suggested value, before the
   run's occupancy rule replaces that cell — read the rule's own line (`grid hash:
   cell … by occupancy/exact`) to see what the run actually uses.
   To choose deliberately, sample the node nearest-neighbour spacing (cKDTree on
   a ~30k-node sample) and histogram nodes-per-cell at candidate cell sizes.
   The runtime symptom is in the verbose run log: `step=` ~ms per particle.
2. **Too big a cell** (the auto floor): 10⁴–10⁵ nodes per cell over refined
   zones → linear scans, ~100× run-time blowup.
3. **Too small a cell**: where local spacing exceeds one cell the 3×3
   window does not hold the true nearest node. With `mesh.exact_nearest`
   (the default) the engine widens the search in-kernel until it is provably
   the nearest — exact, at the cost of that widening (proportional to the
   share of such anchors); with the guard off the anchor is silently
   *near*-nearest (a stencil perturbation, not a failure).
4. **Validate the choice**: run ~1 model day at the real particle count with
   the candidate cell AND with a cell large enough to be provably exact;
   with the guard the two must come out **bit-identical** (the anchor no
   longer depends on the cell) except at exact ties (two equidistant nodes: the
   scan order breaks them) and for queries beyond the search budget (far outside
   the mesh, where the anchor is the best node seen) — any other difference is a
   bug, report it.
5. Pick the fastest cell that passes 4 (nodes per window vs widened share).
   `elem_search: walk` is NOT the fix for this cost (hypothesis raised and
   eliminated once: walk removes the containment search, not the
   nearest-node anchor `leastsq` needs, and it changes the scheme) — see
   `kelp-diagnose` node 4.
   Set it as `mesh.grid_hash_cell_size` (config), not in code: the run pins it
   in `mesh.metadata` so a later `build_search` (e.g. from seed generation)
   cannot silently revert to the pathological auto size. Scale of the answer: a
   structure-resolving mesh (metres to kilometres in one domain) lands at
   0.005–0.01°; the 0.5° figure above is a CORE2-class GLOBAL mesh — never guess
   from another mesh's number. The tool's only input is a config: generate a
   BOOTSTRAP config with no cell set, run `check_grid_hash_cell.py` on it, then
   regenerate every production config with the printed cell (or leave the cell
   unset: the run's rule chooses it and the metadata records it) (the validator
   asserts the bootstrap has none and the production configs have it). Symptom in the log: `step=` ÷
   alive particles ≥ 1 ms.
