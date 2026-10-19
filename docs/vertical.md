# KELP vertical motion (3-D tracking)

> **Covers:** how a particle moves up and down in three-dimensional tracking — the two vertical coordinate systems (fixed z-levels and terrain-following sigma), finding which vertical layer a particle sits in, how the horizontal velocity is read in depth (per layer, or linearly in depth with `mode.uv_vertical: linear`), interpolating the vertical velocity `W`, and clamping the depth to the water column.
> **Does not cover:** the horizontal combination of the `u`/`v` vertex values (see `velocity_interpolation.md`); how the vertical displacement `z += W·dt` is applied within a step (see `time_integration.md`); the algebraic formulas for the vertical profile and the four horizontal-combine methods (see `interpolation.md` §2–§3).
> **Code:** `src/kelp/numerics/vertical.py:interp_w_spatial`, `src/kelp/numerics/vertical_sigma.py:interp_w_sigma_spatial`, `src/kelp/numerics/vertical.py:clamp_depth`, `src/kelp/numerics/vertical_uv.py:depth_points`, `src/kelp/numerics/vertical_uv.py:uv_bracket`.
> **Verified at:** see `docs/doc_map.yaml` (stamped via check_doc_sync --bless).

In two-dimensional tracking a particle only ever has a longitude and a latitude;
its depth never changes. In three-dimensional tracking the particle also carries
a depth `z` (in metres, positive downward — `z = 0` is the surface, larger `z`
is deeper), and each step moves that depth up or down according to the model's
vertical velocity `W`. This document explains the machinery behind that vertical
motion: how the model's vertical layers are laid out, which layer a given
particle depth falls in, how `W` is interpolated to the particle, and how the
depth is held inside the valid water column.

The companion overview is `architecture.md`. The horizontal half of the same
step — turning position into `(u, v)` — is `velocity_interpolation.md`, and the
integrator that ties horizontal and vertical motion together into one time step
is `time_integration.md`.


## 1. Two vertical coordinate systems

The ocean models KELP tracks in store their fields on vertical layers, but
the two model families describe those layers differently. The mesh records which
description applies in `mesh.vertical_coord`, and the integrators dispatch on
that string to choose the right backend.

**z-level (`vertical_coord == "zlevel"`).** Here the vertical layers have
**fixed physical depths** that are the same everywhere in the domain. The
interface depths are stored directly in `mesh.sigma` as metres
(positive-down), for example `[0, 10, 20, 50, ...]`. The local bottom is a
separate per-node depth in `mesh.depth`, so two columns can use different numbers
of the shared levels but the level depths themselves never move. This is the
arrangement used by the FESOM2 z-level model.

**sigma (`vertical_coord == "sigma"`).** Here the layers are
**terrain-following**: each water column is split into the same number of layers,
but their physical depths stretch and shrink with the local water depth and the
moving free surface. `mesh.sigma` no longer holds metres — it holds the
dimensionless sigma values of the interfaces (for this model family, `1` at the
surface descending to `0` at the bottom). The physical depth of an interface in a
given column is recovered on the fly from the column thickness. With node depth
`d`, free-surface elevation (sea-surface height) `\eta`, and a minimum thickness
`D_\mathrm{min}` that guards against degenerate near-dry columns,

$$ H = \max(d + \eta,\; D_\mathrm{min}) $$

is the local column thickness, and an interface at sigma value `\sigma` sits at
physical depth

$$ z = (1 - \sigma)\, H. $$

So `\sigma = 1` maps to `z = 0` (surface) and `\sigma = 0` maps to `z = H`
(bottom). These two relations are the whole sigma geometry; they are implemented
as `column_depth` (`src/kelp/numerics/vertical_sigma.py:column_depth`) and
`sigma_interfaces_to_z` (`src/kelp/numerics/vertical_sigma.py:sigma_interfaces_to_z`).
This is the arrangement used by the FESOM-C sigma model. The key consequence is
that a sigma column's geometry depends on `\eta`, which changes every snapshot —
so layer finding and clamping in the sigma backend always take the live
free-surface elevation, whereas the z-level backend needs only the fixed levels
and the per-node bottom.

In both systems the vertical velocity field `W` is **node-based and stored at the
`K` interfaces**: `VelocityField.w` has shape `(K, N)` for `K` interfaces and `N`
nodes (it is `None` in 2-D runs). The particle's depth, by contrast, is always a
physical depth in metres regardless of which coordinate system the mesh uses —
the sigma transform is applied locally, inside the vertical kernels, and never
leaks into the particle state.


## 2. Finding the layer

Before `W` can be read, the model needs to know which pair of interfaces a
particle's depth sits between — its **mid-layer** index. With `K` interfaces
there are `K-1` mid-layers; mid-layer `k` spans the depth range from interface
`k` down to interface `k+1`.

For z-levels this is a direct bracket search against the fixed level depths.
`find_layer` (`src/kelp/numerics/vertical.py:find_layer`) returns the index
`k` such that `z_levels[k] <= z < z_levels[k+1]`, clamping a particle at or above
the surface to layer `0` and a particle at or below the local bottom to the
deepest valid layer. The deepest valid layer is itself capped by the per-node
bottom depth, so a particle never lands in a layer that lies entirely below the
sea floor at its location. `find_layers`
(`src/kelp/numerics/vertical.py:find_layers`) is the vectorised wrapper that
does this for every alive particle at once (dead particles get layer `0`).

For sigma the same idea applies, but the interface depths must first be
reconstructed for the particle's own water column from the live geometry of §1.
`find_layer_sigma` (`src/kelp/numerics/vertical_sigma.py:find_layer_sigma`)
computes each interface depth inline as `(1 - sigma_lev[k]) * H` and then
brackets the particle against those, with the same surface/bottom clamping.
`find_layers_sigma` (`src/kelp/numerics/vertical_sigma.py:find_layers_sigma`)
is the vectorised wrapper; because the column thickness `H = max(depth + eta,
Dmin)` depends on the free surface, it is given the per-node bottom depth and the
current `\eta` field as well. The integrators pass the particle's own column
`H_p` (`H_particle=`, interpolated at the particle like W, §3) instead of the
nearest node's; direct callers without it get the nearest node's column.

### 2.1 The horizontal velocity in depth: per layer, or linear

The horizontal velocity of a 3-D z-level run sits at the **layer centres**
(FESOM2 `unod`/`vnod` and `u`/`v`, the HEALPix `uo`/`vo`: `K-1` rows). By
default it is **piecewise constant in depth**: every vertex of the
interpolation stencil reads the row of the particle's layer (the layer of §2,
found in the nearest node's column), so a particle crossing a level sees the
velocity jump. With `mode.uv_vertical: linear` the same data are read
**linearly in depth** between the layer centres instead. Inside the engine the
choice is a property of the field, `VelocityField.uv_vertical`:

| `uv_vertical` | rows of `u`/`v` | data points `p` (positive down) | how it gets there |
|---|---|---|---|
| `"layer"` (default) | `K-1` | — (one value per layer) | every reader; the option off |
| `"layer_linear"` | `K-1` | the layer centres `(z_k + z_{k+1}) / 2` | `mode.uv_vertical: linear`: the tracking layer marks the fields it reads |
| `"level"` | `K` | the levels `\|mesh.sigma\|` | a reader whose u/v sit on the levels (`native_uv_vertical = "level"`); always linear in depth, the option is refused there |

`depth_points(mesh, placement)` (`src/kelp/numerics/vertical_uv.py`) builds
the table once per mesh and placement: the points `p[0..M-1]` and, per node,
the **last usable point**

`last_i = max(0, min(M-1, max({k : lower_k <= depth_i} ∪ {0}), D_i))`

with `lower_k = p_k` for level data and `lower_k = z_{k+1}` — the layer's
LOWER level — for layer data. That is `find_layer`'s rule (§2): a partial
bottom layer, whose lower level lies below the bed, is never read. `D_i` is a
reader's declaration of the last finite point of its own data column
(`mesh.metadata["uv_data_last"]`, the HEALPix reader, `healpix.md`); `-1`
there means "the surface only", and the surface value must then be finite.
Infinite bed depths follow the formula as written: `+inf` keeps every point
(`last_i = M-1`, then capped by `D_i`), `-inf` keeps the surface point only. A
NaN bed depth compares false with every `lower_k`, so the set is empty and
`last_i = 0` (the code sets it explicitly: a sorted search would place the
NaN after the last point). The table is cached on the
mesh and rebuilt whenever `mesh.sigma`, `mesh.depth` or the declaration is
replaced (a `max_levels` cut included; an in-place write is not seen — the
engine therefore builds the table from its own read-only snapshot of the
declaration, which replaces the reader's array in `mesh.metadata`: writing
into the reader's array afterwards changes nothing, writing into the snapshot
raises, and a new array declares another prefix; a copied or unpickled mesh,
whose snapshot numpy returns writeable, gets a new snapshot on first use).

The rule at one vertex `i` and particle depth `z`:

- the depth is clipped to the vertex's own column, `z_i = max(p[0],
  min(z, p[last_i]))`: **above the first point and below the last usable point
  the value of that point is used** — constant ends, no extrapolation;
- between two points it is the linear blend `(1 − α) f_k + α f_{k+1}`; exactly
  on a point the point is read directly (of two coincident points, the deeper
  one — except at the surface end: at or above `p[0]` point 0 is read even
  when point 1 coincides with it, e.g. `p = [0, 0, 10]`, `z = 0` reads point
  0);
- points below `last_i` are **never read**, not even multiplied by zero, so
  dry or fill values there are harmless. Points `<= last_i` must be finite: the
  record check (`check_depth_linear_record`) refuses a record with a
  non-finite `u` or `v` there, naming the component, node, point and record,
  once per field and mesh table.

Each vertex clips to **its own** bottom — the depth-linear counterpart of the
bottom-aware neighbour clamp of W (§3) — so a shallow vertex contributes its
deepest usable value while its deep neighbours are read at the particle's
depth. A NaN particle depth reads like a depth below the bed (each vertex its
last point), the layer path's outcome (`find_layer` puts a NaN depth in the
deepest valid layer). In time-interpolated runs each stored value is blended
in time FIRST, then read in depth, then combined horizontally
(`velocity_interpolation.md` §7); for the 4-snapshot PCHIP the order matters.
Every Runge–Kutta stage reads the horizontal velocity at its own stage depth
(Euler: the step-entry depth; RK2: `z` and `z_mid`; RK4: `z`, `z_2`, `z_3`,
`z_4`), unclamped (the per-vertex clip above handles a stage that overshoots
the column).

The rule reproduces the layer answer where it should: data uniform in depth
give the layer result bit for bit wherever every vertex reads a data point
directly (on a point, or clipped at either end), and between points each
vertex value `(1 − α) c + α c` is within 1 ulp of `c`
(`tests/test_interp_depth_linear.py`).

Scope: 3-D tracking on z-level meshes, node-layout u/v (FESOM2
`unod`/`vnod`, the HEALPix dual geometry) and `numerics.uv_interp:
barycentric` — the only interpolator that evaluates depth-linear fields — with
`euler`, `rk2` or `rk4`; the configuration keys and refusals are in
`configuration.md` §2. A FESOM2 mesh read from ASCII files without
`depth.out` has depth 0 everywhere: depth-linear data then collapse to the
surface point, as `find_layer` puts layer data in layer 0.


## 3. Vertical velocity `W` interpolation

**What `W` means inside KELP, and what the files contain.** The tracker
integrates `z += W·dt` with `z` the positive-down depth, so the internal
`VelocityField.w` is the **positive-down rate of the tracked depth**, on the
`K` interfaces. Ocean models write the opposite sign, and each reader
converts once, at read time (`src/kelp/readers/__init__.py:resolve_w_sign`):

| Producer (reader) | Variable | File convention | Conversion |
|---|---|---|---|
| FESOM2 (`fesom2`) | `w` on `nz` | positive **up**; zstar: the velocity *through* the moving interfaces (`oce_ale.F90:vert_vel_ale`) | negated |
| HEALPix CORE2 twins (`healpix`) | `wo` | positive up (remapped FESOM2 `w`) | negated |
| ng5 IFS-FESOM (`healpix`) | `avg_wo` | positive up ("Time-mean upward sea water velocity"; +2·10⁻⁵ m/s at 50 m in the equatorial Pacific) | negated |
| FESOM-C (`fesomc`) | `w` (= `Wvel`) on `sigma` | positive **up**; the flux through the **sigma interfaces** per unit area, zero at the surface and the bottom (`fv_3D_vel.F90:vert_vel_sigma`) | negated; the step integrates the sigma fraction with it (below); `w_cart` is refused |
| `utils/reconstruct_w.py` output | `w` | positive **down** (KELP internal), stamped `positive = "down"` | none |

The file's convention is resolved as `mode.w_positive` (`up`/`down`) >
the variable's CF `positive` attribute > its CF `standard_name`
(`upward_…`/`downward_…`) > the reader's producer default. A W
file written by `utils/reconstruct_w.py` before 2026-09-23 carries no
attribute; read it with `mode.w_positive: down`. Until 2026-09-23 no reader
converted: every FESOM2/HEALPix 3-D run moved particles vertically in the
wrong direction (the synthetic tests used KELP's own convention and could
not see it; `tests/test_w_sign_contract.py` now checks the sign against the
equatorial upwelling in the real CORE2 output and its HEALPix twin).

*Coastal FESOM (sigma).* FESOM-C writes the sigma-coordinate vertical
velocity `w` (`Wvel`): the flux through the sigma interfaces per unit area,
positive up, zero at the surface and the bottom. With `q = (1 − σ) H` the
depth below the moving surface and `H = d + η`, the particle's sigma level
obeys `dσ/dt = Wvel/H`, i.e. its sigma fraction `s = q/H = 1 − σ` obeys

`ds/dt = −Wvel / H` .

(Written as a depth rate this is `dq/dt = (1 − σ)(η_t + u·∇H) − Wvel`; the
sigma form carries the moving surface and the flow along a sloping bottom
without any derivative of `η` or `H`.) The reader supplies `W = −Wvel` on the
interfaces (sign per `mode.w_positive`, default up). The integrators
(`numerics/integrators.py`, `_euler/_rk2/_rk4_step_unfused`) advance `s` with
`ds/dt = W/H_p` at every RK stage and place the particle at
`z = s·H_p(x_stage, t_stage)`, and at the end at `z = s_new·H_p(x_new, t_end)`;
`H_p = max(d + η, D_min)` is the column height interpolated at the particle
by the same horizontal method as W (`numerics/vertical_sigma.py:
sigma_column_height`; exact for a linear bottom with `leastsq` and
`barycentric`, the nearest node's column with `nearest`), and `η` at each
stage is the step's own temporal interpolant (linear, PCHIP, across the cycle
seam). The surface at the start and at the end of every step comes from
one function of time in the tracking layer (`tracking.py:_eta_at_time`: the
PCHIP value where a 4-snapshot stencil exists and PCHIP is on, else the
linear payload blend; in discrete mode the end is the next record), so the
end of one step and the start of the next are bit-identical — `s` is
continuous across steps even where consecutive steps interpolate `η`
differently (a steady-fallback step, `dt` not fitting the file step, between
PCHIP steps blends the surface linearly between its start and end), and a
resumed run reproduces the continuous one. The layer of the horizontal velocity, the depth clamp and the
diffusion kicks use the same `H_p` (the horizontal kick keeps `s`, the Az kick
reflects in the end-of-step column). With a linear time interpolant, a linear
bottom and a uniform `σ̇` the discrete solution is exact
(`tests/test_fesomc_w_contract.py::TestSigmaIntegration`: tide + slope +
flow + flux, Euler/RK2/RK4, forward/backward, uniform and stretched σ, to
1e-9 relative); with a `σ̇` linear in `s` (an exponential exact solution) the
schemes converge at orders 1/2/4 (`TestSigmaStepOrder`).

W is sampled **along the sigma surfaces**: every neighbour column of the
horizontal stencil is evaluated at the particle's σ, `z_j = (z/H_p)·H_j`
(`numerics/vertical_sigma.py`); sampling at the particle's depth mixed sigma
levels on a sloping bottom. FESOM-C's other diagnostic, `w_cart`
(`vert_vel_cart`), is **refused**: its flux term is `z_mid (Wvel_t − Wvel_b)`
instead of `z_t Wvel_t − z_b Wvel_b`, so it lacks the dia-sigma part of the
Cartesian velocity. On sigma meshes the recorded `vel_w` is `W = −Wvel`, the
dia-sigma flux positive down, not the rate of `z` (the flow value: the same sign in forward and
backward runs, like `vel_u`/`vel_v`). As for every recorded velocity (output format 2,
`io_and_time.md` §5.4), it is the value evaluated at the start of the step that
ends at the record, NaN wherever the particle was not alive through that step,
and a 2-D run writes no `vel_w` variable at all.

History (2026-09-23): the reader used `w_cart` as a positive-down interface
field (a manufactured moving column drifted 1.1 m in 1000 s); the first fix
baked a centred time difference of `η` into W (4–6 % off the sigma level
between snapshots on a coarsely sampled tide, 20 % at the cycle seam); the
second applied the surface motion as a start-of-step depth scaling (first
order in `dt` with a dia-sigma flux, lost on steady-fallback steps, kicks in
the start-of-step column); integrating `s` removes all three.

*zstar.* FESOM2's `w` is not the Cartesian vertical velocity: interfaces
sit at fixed zstar coordinate `s` and move with the sea surface, and `w` is
the flow through them. The particle's zstar coordinate obeys
`ds/dt = w·H/(H+η)`; KELP integrates `w` itself on the fixed reference levels
(where the horizontal velocities also live), i.e. it drops the factor
`H/(H+η)` — a relative error `η/H` in the vertical RATE (about 10⁻⁴ in the deep
ocean, 1–2 % on a 50–100 m shelf; the regional crops carry no `ssh`, so the
factor cannot be applied there). Integrated over a trajectory this is at most
`η/H` times the particle's total vertical excursion, and it can accumulate
when `w` and `η` are correlated. Separately, the recorded depth is the zstar
(reference) depth, which differs from the physical depth by `η·(1 − z/H)` at
any instant (at most |η|, under a metre in CORE2).

Interpolating `W` to the particle is a **two-stage** operation: first a vertical
interpolation that collapses the `(K, N)` interface field to one value per node
at the particle's depth, then a horizontal combine that blends those node values
into a single value at the particle's longitude and latitude.

**Stage 1 — vertical, per node.** At each node involved, `W` is linearly
interpolated between the two interfaces that bracket the particle depth (the
layer found in §2), with hard clamping to the surface and bottom interface (no
extrapolation past the ends of the column). The z-level kernel is
`interp_w_linear` (`src/kelp/numerics/vertical.py:interp_w_linear`, single
particle `_interp_w_linear_single`); the sigma kernel is `interp_w_sigma_linear`
(`src/kelp/numerics/vertical_sigma.py:interp_w_sigma_linear`), which
reconstructs the interface depths from `H` exactly as in §1 before bracketing.
The formula for this linear blend is in `interpolation.md` §2 and is not repeated
here.

On z-level meshes the `leastsq` and `idw` kernels do NOT repeat the
bracket search per neighbour (TODO 6.4, 2026-08-15): the particle's own
bracket `(k, alpha)` is computed once (`_w_bracket`) and reused for every
neighbour deep enough to be unclamped; only neighbours whose bottom is
above the particle re-bracket at their own bottom depth. Bit-identical to
the per-neighbour evaluation (pinned against frozen reference copies in
`tests/test_w_spatial_interp.py`); measured ~×1.2–1.4 on the K=100
column regime the search cost scales with, ~×1.1 at K=48.

**Stage 2 — horizontal combine.** The per-node `W` values from the 1-ring
stencil around the nearest node — the centre node plus the unique nodes of the
elements touching it, built by `_collect_neighbour_nodes`
(`src/kelp/numerics/vertical.py:_collect_neighbour_nodes`) — are combined by
one of four methods, selected with the configuration key `numerics.w_interp`:

- **`nearest`** — use the vertical profile at the nearest node only (it does not
  enter the horizontal stencil at all);
- **`leastsq`** *(the default)* — a least-squares plane fit over the 1-ring node
  values;
- **`barycentric`** — locate the 1-ring element that contains the particle and
  blend its vertex `W` values;
- **`idw`** — inverse-distance-squared weighting over the 1-ring nodes.

These are dispatched by integer flag inside the kernels — the constants are
`W_NEAREST = 0`, `W_LEASTSQ = 1`, `W_BARYCENTRIC = 2`, `W_IDW = 3` — so the hot
loop never branches on a string. The dispatcher is `interp_w_spatial`
(`src/kelp/numerics/vertical.py:interp_w_spatial`) for z-levels and
`interp_w_sigma_spatial`
(`src/kelp/numerics/vertical_sigma.py:interp_w_sigma_spatial`) for sigma; the
per-method formulas are in `interpolation.md` §3 and are not re-derived here.

**Bottom-aware per-neighbour clamping.** When stage 1 samples the vertical
profile at a *neighbour* node, the evaluation depth is clamped to that
neighbour's own bottom: `z_eval = min(z, node_bottom)`. Without this, a particle
near the sea floor could pull a below-bottom `W` from a shallower neighbour into
the horizontal blend. Clamping each neighbour to its own column keeps the
contribution physical. Every method also degrades gracefully — if its stencil is
unusable (too few neighbours, a singular fit matrix, or no containing element),
it falls back to the nearest-node bottom-aware profile value, so the result is
always a finite `W` and never NaN.


## 4. Depth clamping

After the vertical displacement has been applied the particle's new depth must be
held inside the water column: not above the surface, not below the bottom. This
clamp runs **after** the step has moved the depth (see `time_integration.md` for
where in the step sequence it sits), against the per-node bottom at the
particle's nearest node.

For z-levels, `clamp_depth` (`src/kelp/numerics/vertical.py:clamp_depth`)
clamps the depth into `[z_levels[0], node_depths[nearest_node]]` — the surface
interface (typically `0` m) at the top and the local bottom depth at the bottom.

For sigma, `clamp_depth_sigma`
(`src/kelp/numerics/vertical_sigma.py:clamp_depth_sigma`) clamps against the
**live** column. The surface boundary is fixed at `z = 0`; the bottom boundary is
the local thickness `H = max(depth + eta, Dmin)` minus a small offset `z0b_min`,
which keeps the particle strictly above the bed rather than exactly on it.
Because `H` carries the free-surface elevation, the sigma clamp follows a moving
sea surface and a drying/wetting column from one snapshot to the next. The
integrators clamp in the particle's end-of-step column `H_p` (`H_particle=`),
the same column the sigma step placed it in (§3).

Dead particles pass through both clamps unchanged.


## 5. What is called when

For a 3-D step the integrator detects three-dimensionality automatically (the
velocity field carries a `W` and the horizontal velocity is layered) and runs the
vertical pipeline around the horizontal update. Every stage dispatches on
`mesh.vertical_coord`, so the same call sequence serves both coordinate systems:

1. **Find the layer** — `find_layers` (z-level) or `find_layers_sigma` (sigma),
   giving each particle its mid-layer index.
2. **Interpolate `W`** — `interp_w_spatial` (z-level) or `interp_w_sigma_spatial`
   (sigma), with the horizontal method taken from `numerics.w_interp`.
3. **Advance the depth** — `z += W·dt` (the integrator's vertical update; see
   `time_integration.md`).
4. **Clamp the depth** — `clamp_depth` (z-level) or `clamp_depth_sigma` (sigma).

A depth-linear horizontal velocity (§2.1) skips step 1 for `u`/`v`: each
stage hands its depth to the interpolator instead, and no stage looks a layer
up (W brackets the depth itself either way).

There is one performance fork worth knowing. For z-level meshes, two
combinations run **fused** 3-D kernels that do the whole horizontal-plus-
vertical update in one compiled pass: the Euler integrator with
element-layout `leastsq` u/v and `w_interp` `nearest`/`leastsq`; and —
since 2026-08 — the **RK2 integrator with NODE-layout `barycentric` u/v**
and `w_interp` `nearest`/`barycentric`, in steady, linear
temporal-interpolation and 4-snapshot PCHIP modes alike (the vertical
module provides time-blended scalar twins of the W singles for it:
`_interp_w_{linear,bary}_single_{tinterp,pchip4}`, bit-for-bit the
arithmetic of interpolating the materialized stage field; a depth-linear
field runs the `_dl` twins of the RK2 kernels, `time_integration.md`).  Everything
else takes the **unfused** path: sigma meshes always (their column
geometry is rebuilt per column from `\eta`), `idw` (and `leastsq` for
the RK2-barycentric kernel) horizontal-`W`, and RK4 in 3-D.  The fork is
purely about speed — the paths are BIT-IDENTICAL (pinned by
`tests/test_fused_3d_bary.py`), with a hybrid guarantee: the rare
particle whose stage anchor falls into an empty grid-hash window is
re-run through the exact unfused machinery automatically.  The
integrator side of this split is covered in `time_integration.md`.
