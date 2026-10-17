# KELP time integration

> **Covers:** how one time step advances a particle — the Euler / RK2 / RK4 / RK45 integrators, the time-aware Runge-Kutta stages on a time-varying flow, and the discrete vs temporal-interpolation velocity modes (including the lazy two-snapshot blend).
> **Does not cover:** how the `u`/`v` value at a point is computed in space (see `velocity_interpolation.md`); vertical motion and the `W` velocity (see `vertical.md`); reading snapshots from disk and the time bookkeeping (see `io_and_time.md`); the diffusion kick (see `diffusion.md`).
> **Code:** `src/kelp/numerics/integrators.py:euler_step`, `src/kelp/numerics/integrators.py:rk2_step`, `src/kelp/numerics/integrators.py:get_integrator`.
> **Verified at:** see `docs/doc_map.yaml` (stamped via check_doc_sync --bless).

This document explains the heart of the tracking loop: taking a particle where
it is now and computing where it is one step later. Each step asks the velocity
field for the flow at the particle's position (the subject of
`velocity_interpolation.md`), then *integrates* that velocity over a short time
interval to produce a new position. The integration scheme — how many times the
velocity is sampled within the step, and where — is what this document covers,
together with the two ways the run supplies a velocity field to the integrator
and how a multi-stage scheme handles a flow that is itself changing in time.

The companion overview is `architecture.md` §2, which places this step in the
per-step call sequence (velocity → step → beaching → record). This document
zooms into the "step" box: the contract every integrator obeys, the four
integrators themselves, and the time-handling that wraps them.


## 1. The step contract

Every integrator is a single function with the same signature:

```python
step(state, vel, mesh, dt, time_dir, layer=0, **kwargs) -> ParticleState
```

It takes the current `ParticleState`, a velocity field `vel` for this moment,
the `Mesh`, the step size `dt` in seconds, and a direction sign `time_dir`. It
returns a **new** `ParticleState`. The integrators never mutate their input;
each builds the result with `.replace()`, leaving the previous state intact (the
beaching hook relies on this to compare a particle's status before and after the
step — see `beaching.md`). `dt` is always a positive number of seconds; the
*direction* of integration is carried separately by `time_dir`, which is `+1`
for forward tracking and `-1` for backward tracking. Backward tracking needs no
special branching anywhere in the integrator: the interpolated velocity is
returned already multiplied by `time_dir`, so a negative sign simply reverses
the flow.

The horizontal position update is the same first-order displacement in every
scheme; the schemes differ only in *which* velocity they feed into it. A
velocity `(UU, VV)` in metres per second moves a particle over `dt` seconds by

$$
\mathrm{lon} \mathrel{+}= \frac{dt \cdot UU}{z_0},
\qquad
\mathrm{lat} \mathrel{+}= \frac{dt \cdot VV}{M_\mathrm{lat}},
$$

where $M_\mathrm{lat}$ is the number of metres per degree of latitude (the
constant `METERS_PER_DEG_LAT`, aliased `_SLM` inside the Numba kernels) and
`z_0` converts metres of eastward motion into degrees of longitude at the
particle's latitude:

$$
z_0 = \cos(\mathrm{lat}\cdot\mathrm{DEG2RAD})\cdot M_\mathrm{lat}.
$$

The `cos(lat)` factor accounts for meridians converging toward the poles: a
metre of eastward motion spans more longitude the farther you are from the
equator. In 3-D runs the vertical position also advances, `z += W·dt`, using the
vertical velocity `W` described in `vertical.md`. After the move, the new
position is checked against the mesh; a particle that has left the usable domain
or failed velocity interpolation is **reverted** to its pre-step position and
its status is set to OUTSIDE, and — when the run opted into the tracking —
`failure_reason` records *why* (interpolation failure vs out of bounds) on
EVERY path: the unfused ones and, since 2026-08-16, all fused kernels.

The integrator to use is named in the config (`numerics.integrator`) and
resolved through a small registry. `get_integrator(name)`
(`src/kelp/numerics/integrators.py:get_integrator`) maps the string to the
step function via the `INTEGRATORS` dict — `"euler"` → `euler_step`, `"rk2"` →
`rk2_step`, `"rk4"` → `rk4_step`, `"rk45"` → `rk45_step` (§7) — and raises a `ValueError` listing the valid
names for anything else. The rest of the code holds the resulting function as
`step_fn` and never name-checks the scheme again.


## 2. The fixed-step integrators

Euler, RK2 and RK4 are members of the Runge-Kutta family, differing in how many times
they sample the velocity field within a single step and how they combine those
samples. More samples buy a higher *order of accuracy* — how fast the
per-step error shrinks as `dt` is made smaller — at the cost of more work per
step. Each velocity sample is one full velocity interpolation plus the
localization that precedes it (finding the nearest node and, for some methods,
the containing element); those costs are described in `velocity_interpolation.md`
and `localization.md`.

**Euler** (`src/kelp/numerics/integrators.py:euler_step`) is the simplest:
one velocity sample at the particle's current position, one displacement. It is
first-order accurate — halving `dt` roughly halves the per-step error. It is the
cheapest scheme; its fused 3-D kernel serves z-level leastsq runs (§5).

**RK2**, the midpoint method (`src/kelp/numerics/integrators.py:rk2_step`),
takes two samples. The first is a half-step Euler move from the start position
to a provisional **midpoint**; the second samples the velocity *at that
midpoint* and uses it to take the full step from the original position. By
correcting the direction with a mid-step estimate, RK2 is second-order
accurate — halving `dt` cuts the error by about a factor of four.

**RK4**, classical fourth-order Runge-Kutta
(`src/kelp/numerics/integrators.py:rk4_step`), takes four samples at
staggered positions (start, two midpoints, and the provisional endpoint) and
combines them in the weighted average `(k_1 + 2k_2 + 2k_3 + k_4)/6`. It is
fourth-order accurate in space and the most expensive of the three fixed-step schemes, but for a
given accuracy it can use a much larger `dt` than Euler. (On a time-varying
flow, the *time* accuracy is capped lower than the spatial order — see §4.)


## 3. Discrete vs temporal-interpolation modes

Before the integrator can run, the loop must hand it a velocity field for the
current moment. There are two ways to produce one, selected by the
`temporal_interp` flag, and the choice sets the effective time step.

**Discrete mode** (the default) advances exactly one file snapshot per tracking
step. Step `i` reads the velocity at file index `start + i·direction`, and the
payload is a plain `VelocityField` straight from the reader, while the particles
move by the configured `dt`. The two clocks agree only when `dt` equals the file
spacing, so a `dt` unlike it is refused at startup (`tracking._check_discrete_dt`,
since 2026-09-24, in `run_tracking` and in `run_single_realization`, so direct
callers cannot bypass it: a regular axis allows `max(0.1 s, 1e-3·file_dt)` of
jitter; irregular record times — calendar months, a gap — fit no single `dt` and
need interpolated mode; a single record is not checked). This is the
mode to use when the tracking step is meant to match the model output cadence.

**Temporal-interpolation mode** (`temporal_interp: true`) lets the tracking `dt`
be smaller than the file spacing. The simulation keeps its own clock in seconds.
Each step resolves which two adjacent file snapshots **bracket** the current
time and builds a velocity field blended between them. Two helpers in
`tracking.py` do this:

- `_resolve_time_bracket` is pure index/time arithmetic. Given the current
  simulation time it returns the two bracketing snapshot indices, their
  timestamps, and the (possibly wrapped) query time — no file reads, no
  allocation. It also handles the end-of-range and cycling behaviour
  (`on_exhausted`) and the subtle case of a backward step that starts exactly on
  a snapshot, which must bracket the interval *below* it rather than degenerate.
- `_build_vel_payload` turns that bracket into the actual payload. It reads the
  two snapshots through the snapshot cache, computes the blend fraction `alpha`
  from where the query time sits in the interval, and returns the blended
  field together with the time-aware extras (`vel_next`, `dt_to_next`) described
  in §4. When the query lands exactly on a snapshot (`alpha = 0` or `1`) it
  returns that raw snapshot; when it lands strictly between, it returns a lazy
  blend (§5).

The time-coordinate handling, the snapshot cache, and the cycling rules are the
subject of `io_and_time.md`; this document only needs the fact that one mode
gives the integrator a single snapshot and the other gives it a time-blended
field.


## 4. Time-aware Runge-Kutta stages

A multi-stage scheme samples the velocity at intermediate *substep times*, not
just at the start of the step. On a flow that does not change in time this is
harmless — every stage sees the same field. But ocean velocity *does* change
between snapshots, and a midpoint stage of RK2 conceptually wants the velocity
at the position the particle will be at time `t + 0.5·dt`, evaluated using the
flow *as it is at that later time*. Feeding every stage the same start-of-step
field would make the scheme first-order in time no matter how many spatial
stages it has, wasting the higher-order machinery.

So the integrators are **time-aware**. Each stage `k` evaluates the velocity at
its own substep time `t + c_k·dt`, where the `c_k` are the standard
Runge-Kutta stage fractions:

- Euler: `(0)` — single stage, time-awareness is a no-op;
- RK2: `(0, 0.5)`;
- RK4: `(0, 0.5, 0.5, 1.0)`.

The velocity at a substep time comes from blending toward the *next* snapshot.
The tracking layer passes the integrator two extra arguments: `vel_next`, the
raw snapshot the step is moving **toward**, and `dt_to_next`, the positive
number of seconds from the current time to that snapshot. The direction matters:
for forward tracking `vel_next` is the *next* file snapshot, but for backward
tracking it is the *previous* one — the snapshot the motion is heading into. The
tracking layer hands over the direction-correct neighbour on both the
temporal-interpolation and the discrete path, so even in discrete mode the
Runge-Kutta stages get a proper time-order rather than a frozen field.

Each stage then blends the current field toward `vel_next` by the fraction

$$
\alpha_k = \frac{c_k \cdot dt}{dt_{\text{to next}}},
$$

which samples the flow at `t + c_k·dt` forward (and `t - c_k·dt` backward), the
times the trajectory actually visits. There is one constraint: a single step may
not reach past the next snapshot, i.e. `dt <= dt_to_next`. The integrator
itself refuses a step that would (`ValueError` from the time-aware helpers);
the tracking loop never sends one: when the step would pass the bracket (a start
between snapshots, the end of the range) it **downgrades** that step to steady
mode (every stage reuses the start field) and emits one `RuntimeWarning` per
realization rather than extrapolating beyond the bracket. Euler accepts
`vel_next`/`dt_to_next` but its velocity ignores them, since it has only the
single `c_0 = 0` stage (on sigma meshes they still give the column's surface at
the end of the step).

**Convergence caveat.** The blend between two snapshots is *linear* in time. A
linear interpolation introduces a time-truncation error of order `dt²`
regardless of the spatial order of the scheme, so RK4's fourth-order accuracy
applies only to the spatial part — the *time* accuracy is capped at second
order. To make the time error small, use a tracking `dt` much smaller than the
snapshot spacing (`dt << dt_to_next`), or supply model output with a denser time
cadence.

### PCHIP temporal reconstruction (raising the time-order)

Setting `numerics.temporal_reconstruction: pchip` replaces the two-snapshot
linear blend with a **four-snapshot monotone cubic** (PCHIP: Fritsch–Carlson
monotonicity with the Fritsch–Butland weighted-harmonic slopes) in
time. Instead of a straight line between the two bracketing snapshots, each RK
stage value is read from a cubic fitted through the four snapshots
`(t_{n-1}, t_n, t_{n+1}, t_{n+2})` surrounding the step, evaluated at the stage
time `t + c_k·dt`. The cubic is *monotone* — it introduces no new extrema
between snapshots, component by component. That is shape preservation per
component only: a linear relation that every snapshot satisfies (a zero
divergence, a fixed ratio between components) is in general not kept between
them, the reconstruction does not commute with a rotation of the vector
components, and for time-averaged archives it does not reproduce the window
means (the linear blend keeps linear relations between snapshots, but it has the
same limitation for window means).

This lifts the time-reconstruction floor: a cubic is integrated exactly by
RK4's Simpson-type quadrature, so at coarse cadence RK4 **separates from** (drops
below) RK2 instead of collapsing onto the same `O(file_dt²)` floor. It is
opt-in and byte-identical when left at the default `linear`; it is active only
in temporal-interpolation mode. Mechanically, the tracking layer resolves a
four-snapshot stencil around the step (interior intervals only — the first and
last interval of the file range, and the cycle seam, fall back to linear with a
one-time warning) and hands the `recon=` bundle to `integrators.rk2_step`/`rk4_step`.
**2-D least-squares runs take a fused four-snapshot-cubic kernel**
(`_fused_rk{2,4}_step_2d_pchip` → `_interp_leastsq_core_pchip4`): the Fritsch–Carlson
cubic is blended inline per element via the `@njit` scalar twins
`_fc_slope_scalar`/`_pchip_value_scalar`, so no full reconstructed field is
materialized per stage. 3-D, node-layout and barycentric runs instead take the
**unfused** path, which routes stage construction through
`temporal_reconstruct.make_stage_vels_recon`; the fused two-snapshot *linear*
kernels of §5 bake the linear blend and are still bypassed when `recon` is
supplied. The fused and unfused PCHIP results agree element-for-element (pinned
by `tests/test_temporal_reconstruction.py::test_fused_pchip_matches_unfused`),
and the fused wrapper re-validates the bundle (four finite strictly-increasing
times, `direction` matching `time_dir`, equal snapshot shapes, every stage time
inside the central bracket) for parity with `reconstruct_pchip`, which raises on
the same. The blend is per-array-element, so it works unchanged in 3D
(`u`/`v`/`w`/`eta` and sampled scalars; Kh is not reconstructed).

The unfused path's whole-field blends are themselves parallel (TODO 6.7,
2026-08-15): large C-contiguous fields go through a Numba `prange` kernel
built on the same scalar twins as the fused 2-D kernels — bit-identical to
the numpy expression, which everything else (small fields, no numba,
**non-C-contiguous or non-float64 fields**) still uses. A large float64
field that misses the fast path *only* on memory order now emits a
one-time `RuntimeWarning` — that case is always a reader handing back a
Fortran-ordered/strided array (it silently cost ~10x per step on a real
healpix 3-D run, 2026-08, before the readers were fixed to emit C-order) —
and `make_stage_vels_recon` reconstructs each **unique** stage time
once, sharing the field object between RK4's two midpoint stages
(`c = 0.5, 0.5`). This removed the ~5–6.6 s/step tracer-count-independent
reconstruction floor of 3-D PCHIP runs (see `performance.md` §2 for the
measured numbers). Authoritative
code: `src/kelp/numerics/temporal_reconstruct.py`.


## 5. Fused kernels and the lazy blend

For 2-D tracking the inner work of a step — find the nearest node, interpolate
the velocity, displace the particle, check it is still inside the domain — is
collapsed into a single compiled Numba kernel that runs over all particles in
one pass, in parallel. There is a fused kernel for each integrator and each
time mode: a steady variant (`_fused_euler_step_2d`, `_fused_rk2_step_2d`,
`_fused_rk4_step_2d`), a time-aware "tinterp" variant for the two-snapshot
linear blend (`_fused_euler_step_2d_tinterp`, `_fused_rk2_step_2d_tinterp`,
`_fused_rk4_step_2d_tinterp`), and a "pchip" variant for the four-snapshot
monotone-cubic blend (`_fused_rk2_step_2d_pchip`, `_fused_rk4_step_2d_pchip`;
§4 — Euler has no PCHIP kernel, the tracking layer never builds a stencil for
single-stage Euler), and a barycentric tinterp variant for **node-layout**
velocity (`_fused_rk2_step_2d_bary_tinterp`, `_fused_rk4_step_2d_bary_tinterp`;
below). In 3-D there is a fused **Euler** kernel for z-level
leastsq runs (`_run_fused_euler_3d`) and a fused **RK2 barycentric** kernel for
z-level node-layout runs with `w_interp` nearest/barycentric, in a linear-blend
and a PCHIP variant (`_fused_rk2_step_3d_bary_tinterp`,
`_fused_rk2_step_3d_bary_pchip`). This is a performance structure only: the
fused kernels compute the same result as the unfused path without materializing
intermediate per-particle arrays. Under the default `mesh.exact_nearest: true`
every fused kernel anchors each evaluation site through the exact in-kernel
search (`core/mesh.py:_grid_hash_nearest_exact`), the same function the batch
path calls, so fused and unfused agree bit for bit on any mesh. With the guard
off, a stage whose query lands in an **empty 3×3 grid-hash window** gets a
ring-widened *approximate* anchor where the unfused batch path KD-heals to the
exact one (`core/mesh.py:_grid_hash_nearest_widened` vs `find_nearest_nodes`);
on quasi-uniform FESOM meshes in-domain windows are never empty, and the
barycentric family below turns that precondition into a checked gate.

**Which paths are fused.** A run takes the fused path only when its math matches
what the kernel bakes in. The least-squares kernels are eligible only when
`uv_interp == "leastsq"` on element-layout velocity. **Node-layout barycentric
with a next snapshot supplied** (FESOM2 `unod`/`vnod`, NEMO C→T — both discrete
mode, which passes `vel_next` whenever the file times are known, and
`temporal_interp` runs) has its own fused RK2/RK4 twin since 2026-08: each stage
blends the two snapshots inline **per vertex**
(`_interp_barycentric_core_tinterp`, the nested two-level blend of the
leastsq tinterp core on top of the untouched barycentric geometry), so the
per-stage whole-node-field lerp disappears — on a 4M-node mesh an RK4 step drops
from ~83 ms to ~9 ms at 8 threads (~×9, and the fused cost is mesh-size
independent). It runs only when ALL of: 2-D, node layout, `vel_next` supplied,
no PCHIP bundle, no walk-carried elements (`state.elem is None`), and — **only
with `mesh.exact_nearest: false`** — **the grid-hash density certificate holds**
(under the default guard the kernels search exactly in-kernel and fuse on any
mesh): `core/mesh.py:grid_hash_dense_for_elements`, computed once per mesh: every hash
cell touched by an element bbox has a nonempty 3×3 window, under which the
in-kernel anchor is provably bit-identical to the batch KD-healed one for every
containable query, so fusing cannot change any value or life/death verdict
(parity is pinned **bitwise** in `tests/test_fused_bary_tinterp.py`). A mesh
that fails the certificate — HEALPix polar rows, where pixel lon-width grows as
1/cos(lat) while hash cells are uniform in degrees, measured to divert anchors
and kill high-latitude particles if fused — stays on the unfused exact path
with a one-time `RuntimeWarning`. Element-layout
barycentric, steady barycentric, walk runs, `edge_leastsq`, the 3-D paths
other than the two fused 3-D kernels above, and all sigma-coordinate 3-D paths
fall through to the
**unfused** dispatcher instead, which evaluates the stages through the
interpolator registry (and, for the multi-stage schemes, materializes one
velocity field per stage via `_make_stage_vels`
(`src/kelp/numerics/integrators.py:_make_stage_vels`), one `lerp` blend per
stage time). The dispatch that picks fused-steady, fused-tinterp, or unfused is
described in §6. On sigma meshes the stages advance the particle's sigma
fraction `s = z/H_p` (`ds/dt = W/H_p`) and place it at `z = s·H_p` with the
stage's own surface elevation, so the time interpolation of `eta` (linear,
PCHIP, cycle seam) and the vertical motion it implies agree exactly; Euler
takes its end-of-step `eta` from `vel_next`, and a step that fell back to
steady velocity from the tracking layer (docs/vertical.md §3).

"Unfused" means **multi-pass, not single-threaded**: each stage's u/v
interpolation still runs through the registry's vectorised
`@njit(parallel=True)` kernels. The first stage reads the step-entry state
directly; later stages (RK2 midpoint, the four RK4 stages) evaluate at the
*stage* positions, which `_interp_uv_at`
(`src/kelp/numerics/integrators.py:_interp_uv_at`) feeds to the same
vectorised interpolator via a shallow stage state (`state.replace` with the
stage `lon`/`lat` and an `alive_mask` selecting the particles that reached the
stage). So a 3-D RK2/RK4 step is parallel across particles and scales with
`NUMBA_NUM_THREADS` — it just runs as separate interpolation/displacement/W/
clamp passes rather than one fused kernel. (Earlier this stage interpolation
was a serial per-particle Python loop, which capped 3-D RK throughput on
million-particle runs; the result is bit-identical, only parallel.) The stage
state also carries the stage DEPTH (`_interp_uv_at(..., zs=)`): a depth-linear
`u`/`v` (`mode.uv_vertical: linear`, `vertical.md` §2.1) is read at each
stage's own depth — Euler at the entry depth, RK2 at `z` and `z_mid`, RK4 at
`z`, `z_2`, `z_3`, `z_4` — and such a stage looks no layer up; layer
interpolators never read the depth, so layer runs are unchanged.

**The lazy two-snapshot blend.** In temporal-interpolation mode strictly between
snapshots, `_build_vel_payload` returns a `LazyLerpField`
(`src/kelp/core/io.py:LazyLerpField`) rather than an eagerly blended field. A
`LazyLerpField` is a view over the two bracketing snapshots (`vel_a`, `vel_b`)
and the blend fraction `alpha`; it blends each field only when something reads
it, and caches the result. The blended values are bit-identical to the eager
`VelocityField.lerp`. The point is that a step which touches only a few fields
pays only for those — and the fused tinterp kernels go further: they read the
**raw** snapshot pair and never materialize the blended `u`/`v` at all.

The fused tinterp kernels
(`_run_fused_2d_tinterp`,
`src/kelp/numerics/integrators.py:_run_fused_2d_tinterp`, is the dispatcher)
receive the raw pair `(vel_a, vel_b)`, the bracket fraction `alpha0` from the
`LazyLerpField`, and separately the stage target `vel_next`. Per element they
apply a **nested** two-level blend: first reconstruct the start-of-step field
from the bracket pair, then blend that toward the stage target by the stage
fraction. Written out for the eastward component at element `e`:

$$
u^0_e = (1-\alpha_0)\,u_{a,e} + \alpha_0\,u_{b,e},
\qquad
u_e = (1-\alpha_k)\,u^0_e + \alpha_k\,u_{t,e},
$$

where `(u_a, u_b)` is the bracket pair, `u_t` is the stage target (`vel_next`),
and `alpha_k` is the per-stage fraction from §4. The first line reproduces the
materialized start-of-step field bitwise; the second is the ordinary stage
blend. When the payload is a plain field rather than a lazy view — a raw
snapshot, or an already-materialized blend — the caller passes `alpha0 = 0`,
which collapses the first line to `u_{a,e}` and recovers the original
single-blend arithmetic exactly. So the lazy machinery changes *when* the blend
happens, never the numbers it produces.


## 6. What is called when

The dispatch lives in `run_single_realization` and the integrator functions
themselves. The loop resolves `step_fn = get_integrator(numerics.integrator)`
once, then calls it each step as
`step_fn(state, vel, mesh, dt, time_dir, ..., vel_next=..., dt_to_next=...)`,
passing the direction-correct `vel_next` and `dt_to_next` whenever the time
information is available (in both discrete and temporal-interpolation modes).

Inside the step function the gate selects a path from a small set of facts —
the integrator, whether the run is 3-D, the vertical coordinate, and
`uv_interp`:

- **Euler** uses the fused 3-D kernel for a z-level mesh when
  `w_interp` is `nearest`/`leastsq` and `uv_interp == "leastsq"`; the fused 2-D
  kernel otherwise (the pair-aware tinterp kernel when the payload is a
  `LazyLerpField`, the steady kernel for a plain field); and the unfused path
  for barycentric/edge_leastsq/node-layout velocity or sigma 3-D. Its single
  stage means `vel_next`/`dt_to_next` never change the result.
- **RK2 and RK4** use a fused 2-D kernel when the run is 2-D and
  `uv_interp == "leastsq"`: the fused **pchip** kernel (via `_run_fused_2d_pchip`)
  when a PCHIP `recon=` bundle is supplied, otherwise the steady fused kernel
  when the step is not time-aware, or the fused **tinterp** (two-snapshot linear)
  kernel (via `_run_fused_2d_tinterp`) when it is. **Node-layout
  `uv_interp == "barycentric"`** additionally gets the fused **bary-tinterp**
  kernel (via `_run_fused_2d_bary_tinterp`) when the step is time-aware, no
  PCHIP bundle or walk-carried elements are present, and (with
  `mesh.exact_nearest: false` only) the mesh passes the grid-hash density
  certificate (§5).
- **RK2 in 3-D** uses the fused barycentric kernel on a z-level mesh with
  node-layout velocity, `uv_interp == "barycentric"`, `w_interp` nearest or
  barycentric and no walk-carried elements — the PCHIP variant when a `recon=`
  bundle is supplied, else the linear/steady variant; a depth-linear field
  runs their `_dl` twins (`_fused_rk2_step_3d_bary_{tinterp,pchip}_dl`: per
  vertex the time blend first, then the vertical rule, then the vertex sum;
  the anchor heal and the terminal splice keep the placement). Everything else — the
  other 3-D runs (RK4 3-D, leastsq or idw `W`, sigma coordinates),
  element-layout barycentric, steady 2-D barycentric, edge_leastsq,
  certificate-denied meshes with the guard off — takes the unfused multi-stage path through `_make_stage_vels` /
  `make_stage_vels_recon` and the interpolator registry.

Before any of this, every step function runs `_check_depth_linear_entry`: a
depth-linear field is refused for `rk45`, a sigma mesh, a field without `w`, a
non-node layout and a `uv_interp` other than `barycentric`; the fields of one
step (`vel`, `vel_next`, each PCHIP snapshot) must share one placement, and each
passes the record check once. For a layer field it reads only the placement
attributes, so layer runs keep their cost and their bits.

After the deterministic step returns, the stochastic diffusion kick (if enabled)
is applied as a separate stage; the schemes, composition modes, and the
restrictions in 3-D are the subject of `diffusion.md`. In every path the
integrator returns a fresh `ParticleState`, and only the particles that
interpolated and stayed inside the domain are advanced. Non-alive particles get
zero velocity diagnostics on every path, including the early exit of a step in
which no particle is alive (until 2026-10-04 that exit kept the last values).


## 7. The adaptive integrator (`rk45`)

`integrator: rk45` (TODO 2.3) is a Dormand–Prince 5(4) scheme with
**per-particle adaptive substepping inside the fixed tracking step**: the
tracking loop still advances by `dt` (records, snapshot brackets, the
diffusion RNG keying and the beaching hook all keep their cadence), but
within each step every particle takes its own sequence of substeps `h`,
controlled by the embedded 4th/5th-order error estimate converted to
metres and compared against `numerics.rk45_tol_m`. A rejected substep is
retried with a smaller `h` (`h ← h·clip(0.9·(tol/err)^{1/5}, 0.2, 5)`);
the last substep is clipped to land exactly on `dt`; the 7th stage of an
accepted substep is reused as the next substep's first (FSAL). Substeps
sample velocity in time through the same nested two-snapshot blend as the
fused RK kernels (§4–§5) — per element, at each particle's own stage
times — so temporal interpolation works without materializing any stage
field. `h` never drops below `rk45_h_min` (default `dt·1e-4`): a clamped
substep is force-accepted and the run warns once, so the step always
terminates.

Failure semantics are the unfused ones, with one adaptive twist: a
FAILED stage evaluation (empty stencil, off-hash probe) is treated as a
rejected attempt and retried at `h·0.2` — a too-large trial can probe
past a coast where a smaller substep is perfectly valid — and only a
failure at the `h_min` floor (or the substep cap, which the wrapper
sizes to exceed a full `dt`-worth of floor-clamped substeps) is
terminal: the particle then reverts to its pre-step position with
`OUTSIDE` and a `failure_reason` code. An off-domain FINAL position
reverts the same way. Intermediate stage positions are wrap/clamp
normalized but not domain-checked (same as the rk2/rk4 stages), and the
`h_min` warning counts only substeps that were genuinely forced through
ABOVE the tolerance.

v1 scope — everything else is rejected at startup, never silently: 2-D
runs only, element-layout velocities with `uv_interp: leastsq`, linear
temporal interpolation only (no `pchip`), no `elem_search: walk`, no
`em`/`m1` diffusion (the uniform split kick composes after the full step
like for rk2/rk4). Authoritative code:
The recorded `vel_u`/`vel_v` diagnostics are the velocity at the accepted
**end** of the last substep (the FSAL stage); the fixed-step integrators
record their first stage, at the start of the step, so the diagnostic is
integrator-dependent (as it always was). The output's NaN rule for particles
not alive through the step applies to both (`io_and_time.md` §5.4). Authoritative code:
`src/kelp/numerics/integrators.py:rk45_step` (kernel
`_rk45_kernel_2d`); tests: `tests/test_rk45.py`.
