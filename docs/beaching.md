# KELP beaching and resuspension

> **Covers:** the beaching model — how a particle near the shore can strand (become BEACHED) and later re-float, the four stranding criteria, the stochastic stranding and release timescales, and how the post-step hook turns those decisions into status changes.
> **Does not cover:** the YAML keys that switch beaching on and tune it (see `configuration.md` §3.2); the deterministic advection step that moves particles before the hook runs (see `time_integration.md`); how a particle's containing element and the coastal-element flags are computed (see `localization.md` and the mesh notes).
> **Code:** `src/kelp/numerics/beaching.py:evaluate_beaching`, `src/kelp/tracking.py:_apply_beaching_hook` (authoritative implementation).
> **Verified at:** see `docs/doc_map.yaml` (stamped via check_doc_sync --bless).

A real floating tracer — a piece of plastic, an oil droplet, a larva — does not
always stay in the water. When the flow carries it close to land it can wash up
on a beach, lodge in a tidal flat, or snag on a coastline and stop moving. Later,
a rising tide or a change in the current can lift it off again and return it to
the water. KELP models this with a **beaching layer**: a set of rules,
applied once per tracking step, that can freeze a particle in place (mark it
*beached*) and later release it (mark it *alive* again). This document explains
what the layer decides and when, leaving the configuration keys to
`configuration.md` §3.2.

The beaching layer is **off by default**. When it is disabled the tracking loop
takes a different branch entirely and the output is byte-for-byte identical to a
run with no beaching at all. Everything below applies only when beaching is
switched on in the configuration.


## 1. What beaching is

Every particle carries a **status**, a small integer (`ParticleStatus` in
`src/kelp/core/particles.py`). Three of its values matter for beaching (a
fourth, `NOT_RELEASED`, is used only by delayed release and never produced or
consumed here):

- **ALIVE** — the particle is in the water and the integrator moves it each step.
- **BEACHED** — the particle has stranded; the integrator no longer moves it, but
  it is not gone and may re-float later.
- **OUTSIDE** — the particle has left the model domain (for example, it crossed an
  open boundary or the velocity interpolation failed); it is permanently dead.

OUTSIDE is *sticky*: once a particle goes OUTSIDE it never comes back.
**BEACHED is the only status that can flip back and forth** — a beached particle
can return to ALIVE when the water comes back. (NOT_RELEASED, the waiting state of
a delayed release, turns ALIVE once, at its release step; `io_and_time.md`.) This single reversible transition is the whole point of
the beaching layer.

Beaching is a **post-step decision layer**, not part of the numerical integrator.
The integrator advances every ALIVE particle by one time step using the resolved
velocity field (and, if enabled, the diffusion kick). Only *after* that motion is
complete does the beaching hook look at where each particle ended up and decide
whether to strand it or release it. Because a BEACHED particle is no longer in the
set of ALIVE particles the integrator advances, freezing a particle costs nothing:
it simply stops being moved. Re-floating it is just as cheap — it rejoins the ALIVE
set and the integrator picks it up again on the next step.


## 2. The four stranding criteria

A configuration chooses any subset of four criteria. They are **OR-combined**: an
ALIVE particle strands this step if *any* of the chosen criteria fires. The four
criteria are evaluated inside `evaluate_beaching`.

Throughout, the key quantity is the **raw column height** at the particle's
nearest mesh node:

```
H_raw = mesh.depth[node]                  (when sea-surface height is not used)
H_raw = mesh.depth[node] + eta[node]      (when use_eta is on)
```

Here `mesh.depth` is the still-water bottom depth (positive, in metres) and `eta`
is the sea-surface height for this time step. `H_raw` is therefore how much water
sits over the bottom right now. Two details matter:

- **`H_raw` is left raw and unclamped.** Elsewhere in the code the column geometry
  is clamped to a minimum thickness (`Dmin`) so that vertical interpolation stays
  well-behaved. Beaching deliberately does *not* clamp: a tidal flat that has just
  gone dry has a near-zero or even negative `H_raw`, and that drying is exactly what
  the beaching test needs to see. Clamping would hide it.
- **The sea-surface height `eta` comes only from the per-time-step velocity
  payload, never from the mesh.** Each step brings a fresh `eta`, so the column
  height rises and falls with the tide. A particle can be afloat one step and
  stranded the next purely because the water dropped beneath it.

The four criteria:

**`depth`** — strand the particle when the water over it is too shallow:
`H_raw < depth_min`. This is the basic "ran aground" test. An optional dry-mask
field can be supplied (a node-based field where a value above 0.5 marks a dry
node); when present it is OR-ed into the depth test, so a particle over an
explicitly dry node strands regardless of `H_raw` — and the same mask vetoes
resuspension (§3): a masked node is never wet, whatever its nominal depth
(until 2026-09-23 a permanently masked deep node released its particle every
step).

**`coastal_element`** — strand the particle when the mesh element it currently sits
in is a *boundary* (coastal) element. The mesh marks an element as coastal if one
of its edges is shared by no other element — that edge lies on the coastline or on
an island shore (see `build_boundary_info` in the mesh code, and `localization.md`).
This criterion captures "drifted into the shore cell" independently of how deep the
water there happens to be.

**`on_failure`** — reinterpret a death as a stranding. When the integrator kills a
particle this step (ALIVE → OUTSIDE) because it left the domain or the interpolation
failed, this criterion asks: did it die *at the shore* rather than out in the open?
If the dead particle's frozen anchor sits in shallow water or in a coastal element,
it is reclassified as BEACHED instead of lost. A particle that left through a deep,
open boundary stays OUTSIDE. The exact context test is described in §5.

**`stochastic`** — strand the particle *probabilistically* while it is anywhere
within a near-shore zone, even if it has not yet run aground. This represents the
fact that wash-up is a chance event, not a hard threshold. Within the zone, each
ALIVE particle strands this step with probability

$$P_\text{beach} = 1 - \exp\!\left(-\frac{|\Delta t|}{t_\text{beach}}\right)$$

where $\Delta t$ is the time step and $t_\text{beach}$ is a stranding timescale (the
mean time to wash up while in the zone). The probability uses the absolute value of
the step, so backward-in-time runs behave the same as forward ones. The **near-shore
zone** is

```
H_raw < zone_depth        OR        the particle's element is coastal
```

where `zone_depth` defaults to twice `depth_min` if not set. The coastal half of the
zone is active whenever the mesh coastal-flag topology has been built — which means
probabilistic stranding *near the coast* works even if the hard `coastal_element`
criterion is itself switched off.


## 3. Resuspension (re-floating)

A BEACHED particle may return to the water. The resuspension policy chooses how:

- **`never`** — beaching is permanent; a stranded particle stays put forever.
- **`auto`** — release the particle as soon as it is properly re-wetted and has
  *stayed* wet long enough (the anti-flicker condition below).
- **`stochastic`** — once re-wetted, release the particle with a chance each step.
- **`auto_or_stochastic`** — both of the above can fire; whichever triggers first
  wins.

Whatever the policy, **release is always water-gated**: a particle cannot re-float
without water over it. "Re-wetted" means the column has risen comfortably above the
stranding threshold:

```
H_raw > depth_min + rewet_hysteresis   and   the node is not marked dry
```

(a node the dry mask marks dry, or a particle without a nearest node, is never
wet).

The small extra margin `rewet_hysteresis` is an **anti-flicker** band. Without it, a
particle sitting right at `depth_min` would beach and release on alternate steps as
`H_raw` jittered by a hair across the threshold. By requiring the water to rise a
clear margin *above* `depth_min` before release, while stranding still triggers at
`depth_min`, the two thresholds are separated and the rapid on/off flicker is
suppressed. This is the standard hysteresis pattern: the level that turns something
on differs from the level that turns it off.

To guard against a single transient wet step, automatic release also requires the
water to have *persisted*. A per-particle counter, **`wet_streak`**, counts how many
consecutive steps a beached particle has been re-wetted. Each step the counter is
incremented for beached particles that are currently re-wetted and reset to zero for
those that are not. Automatic release fires only when

```
wet_streak >= min_wet_steps
```

so with `min_wet_steps` greater than one a particle must stay wet for several steps
before it lifts off. The counter is carried across the whole run and saved in
restart checkpoints, so a resumed run continues counting exactly where it left off.
It is reset to zero whenever a particle is released or is no longer beached.

Stochastic release applies, to the *re-wetted* beached particles only, the same
kind of memoryless probability as stochastic stranding:

$$P_\text{release} = 1 - \exp\!\left(-\frac{|\Delta t|}{t_\text{resuspend}}\right)$$

with $t_\text{resuspend}$ the mean time to re-float once back in the water. Because
the stochastic branch iterates over `beached & rewet` only, the water gate applies
here too: a stranded particle still sitting on a dry flat draws no release chance at
all.


**failure_reason interplay** (2026-08-16): the `on_failure` remap turns a
freshly-killed OUTSIDE particle into a *reversible* BEACHED one — when the
run tracks `failure_reason`, the code the integrator just stamped is
CLEARED on the remap (and again on resuspension), so the field keeps its
cause-of-death meaning; a later real death re-stamps it.

## 4. Reproducible randomness

The two probabilistic decisions — stochastic stranding and stochastic release — draw
random numbers from the **same counter-based generator** used by the diffusion model
(see `diffusion.md` for the full mechanism). A counter-based generator does not keep
any running state; instead, each draw is computed directly from a tuple of integers
by a hash function (`_philox_uint64` — despite the name a keyed SplitMix64,
not Philox; see `diffusion.md`). The tuple here is

```
(seed, realization id, original particle index, global step index, component id)
```

with a distinct *component id* for each kind of draw — `COMP_BEACH = 10` for
stranding and `COMP_RESUSP = 11` for release — so the two never collide and neither
collides with the diffusion draws.

Two points are essential for correctness:

- **The particle index in the key is the *original* index**, not a compacted
  position among the currently-alive particles. As particles die or beach, the live
  set shrinks, but each particle keeps the index it was born with. Keying on the
  original index pins every particle's random sequence to that particle regardless of
  how the live set is reshuffled — so the result does not depend on thread ordering
  and is identical on replay. A run over one shard of a larger release adds the
  shard's first global index (`data.particle_offset` or the seed file's
  `# kelp particle_offset N` header), so a particle draws the same numbers in the
  shard as in the unsharded run (`diffusion.md` §4).
- **Each draw is turned into a uniform value in `[0, 1)` with a 53-bit
  construction**, `(h >> 11) / 2**53`, rather than the naive `h / 2**64`. The naive
  form can round *up* to exactly `1.0` for the largest hash values, which would break
  the half-open `[0, 1)` contract and, for example, never let a probability-1 event
  fail. The 53-bit form (the function `_u01`) stays strictly below one.

The beaching seed is taken from the beaching configuration, falling back to the
diffusion seed when not set. Because the step index in the key is the *global* step
counter, a run resumed from a checkpoint reproduces the same draws it would have made
without the interruption — the randomness is restart-bitwise.


## 5. The post-step hook — what runs when

All of the above is orchestrated by one function, `_apply_beaching_hook`, called once
per tracking step. Its position in the step is fixed: it runs **after** the integrator
has moved the ALIVE particles (and after the periodic fold, when
`numerics.periodic_lon/lat` is configured — the hook always sees in-window
positions) and **before** the simulation clock advances to the next
step. The deterministic motion is already done; the velocity payload (and its `eta`)
for this step is still in hand. (For the surrounding loop, see `architecture.md` §2.)

The hook does its work in two phases and then applies the results:

**Phase 1 — the `on_failure` remap.** The hook looks for particles the integrator
just killed this step (their status went ALIVE → OUTSIDE; the integrator has already
reverted their position to a frozen in-domain anchor). For each such particle it
checks the *context* of that anchor:

```
H_raw < zone_depth        OR        the anchor's containing element is coastal
```

If the context is coastal — shallow or in a boundary element — the particle is
remapped OUTSIDE → BEACHED. Otherwise it stays OUTSIDE. The test is on anchor
*context* rather than on the failure code by design: the gate predates the failure
codes of the fused kernels (every path records them since 2026-08, and they are
opt-in), and the shoreline context is what distinguishes a wash-up from a genuine
exit through an open boundary, whatever the code.

**Phase 2 — criteria and resuspension.** The hook then calls `evaluate_beaching` on
the (possibly remapped) state. That function applies the four stranding criteria of
§2 to the ALIVE particles and the resuspension policy of §3 to the BEACHED ones, and
returns two boolean arrays — `to_beach` (ALIVE particles to strand) and `to_release`
(BEACHED particles to re-float) — together with the updated `wet_streak` counter.

A clean division of labour runs through both phases: **`evaluate_beaching` only
*decides*** — it computes the boolean arrays and never touches particle status. **The
hook *applies*** the decisions, writing BEACHED into the `to_beach` rows and ALIVE
into the `to_release` rows. The `wet_streak` counter is loop-carried by the tracking
loop and written into restart checkpoints.

Two practical notes:

- **The coastal-flag topology is built only when it is needed** — that is, only when
  `coastal_element` or `stochastic` is among the chosen criteria. The containing-element
  lookup that the criteria need is, in turn, computed only for the particles that
  actually consume it (the ALIVE, BEACHED, and just-failed ones), so a run with most
  particles long dead does not pay to re-locate them.
- **Same-step flicker is accepted.** A particle remapped OUTSIDE → BEACHED in Phase 1
  enters Phase 2 already BEACHED, and if its anchor is wet and `min_wet_steps` is 1 it
  may release in the very same step — a grazing contact that beaches and re-floats at
  once. This is intended first-version behaviour, not a bug.
