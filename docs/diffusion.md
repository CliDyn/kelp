# KELP stochastic diffusion

> **Covers:** how a random "kick" is added to each particle to represent turbulent (sub-grid) horizontal mixing — the random-walk model, the three schemes (`uniform` / `em` / `m1`), the deterministic drift correction that variable-diffusivity schemes need, the counter-based random-number generator, and how the kick composes with the deterministic integrator.
> **Does not cover:** the deterministic advection step itself (see `time_integration.md`); how the diffusivity field `Kh` and its gradient are sampled in space (see `interpolation.md` §5); the YAML keys that turn diffusion on and configure it (see `configuration.md` §3.1).
> **Code:** `src/kelp/numerics/diffusion.py:diffusion_uniform_step_meters`, `src/kelp/numerics/diffusion.py:diffusion_em_step_meters`, `src/kelp/numerics/diffusion.py:_philox_uint64` (authoritative implementation).
> **Verified at:** see `docs/doc_map.yaml` (stamped via check_doc_sync --bless).

A Lagrangian particle moved by nothing but the resolved velocity field traces a
single smooth streamline. Real ocean tracers do not: on top of the large-scale
flow there is turbulent stirring at scales the model grid does not resolve.
KELP represents that unresolved stirring as a **random displacement** added
to each particle every step — a random walk superimposed on the deterministic
advection. This document explains the physics of that kick, the three numerical
schemes that produce it, the correction term the spatially-varying schemes need
to stay physically faithful, and the reproducible random-number machinery behind
it. When diffusion is switched off (the default) none of this runs and the output
is byte-for-byte identical to a pure-advection run.


## 1. Why particles get a random kick

The velocity field read from a model output is an *average* over each grid cell
and each output interval. Motions smaller or faster than that — eddies,
filaments, turbulent overturning — are smoothed away. Yet those motions do real
work: they disperse a patch of dye, mix a plume, spread a population of larvae.
The standard way to put that dispersion back into a particle model is to add, at
every step, a small random displacement drawn from a normal distribution. This
turns each trajectory into a *realization* of a random walk.

The justification is a correspondence between the random walk and a partial
differential equation. If a large ensemble of particles is released and each one
is advected by the resolved flow plus an independent random kick of the right
size, then the **density** of those particles evolves according to the
advection-diffusion equation — the same equation a concentration field would
obey. A single particle is one noisy sample; the physically meaningful quantity
is the statistics over many particles. The concentration at a place and time is
recovered by counting how many particles land there across the ensemble.

The kick size is fixed by one requirement: the spread of the particle cloud must
grow at the rate the diffusion coefficient prescribes. For a diffusivity `Kh`
(units of square metres per second), the variance of an initially-point cloud
grows linearly in time,

$$
\sigma^2 = 2\,K_h\,t .
$$

Every scheme below is built so that its per-step random displacement reproduces
exactly this variance growth. The factor of two and the square-root structure of
the kick (next section) both come straight from this relation.


## 2. The displacement

Diffusion is computed as a displacement in **metres** on the local tangent
plane (`dx` east, `dy` north) and applied **along the great circle** with that
initial tangent — the exponential map of the sphere, radius
`R = METERS_PER_DEG_LAT / DEG2RAD` (the tracker's metric), implemented in
`sphere_displace` / `sphere_displace_arrays` (`numerics/diffusion.py`). A pure
northward kick of `dy` metres still moves exactly `dy / METERS_PER_DEG_LAT`
degrees. Until 2026-09-23 the kick was applied on the longitude–latitude chart
(`dx / (cos(lat)·METERS_PER_DEG_LAT)` degrees of longitude, `dy /
METERS_PER_DEG_LAT` of latitude); that walk lacks the metric drift
`−K tan(lat)/R` of the diffusion operator on the sphere (its generator on
`sin(lat)` is half the correct one, its equilibrium is uniform in latitude, not
in area). For an isotropic constant `K` the great-circle kick is rotation-invariant, so it keeps
the uniform area density exactly (it is a weak first-order step in time, not the
exact heat kernel); it does not depend on the coordinate frame and passes through a pole. A diagonal
tensor with `Kx ≠ Ky` gets the extra northward drift `(Kx − Ky) tan(lat)/R · |dt|`
(`anisotropic_metric_drift_m`; latitude capped at 89.9°, where east/north
anisotropy stops being defined). In the combined Euler scheme the diffusive
part moves along the great circle from the start position and the advection
keeps the chart step of the deterministic update; the deterministic
integrators themselves are unchanged (`time_integration.md` §1).

The full per-step horizontal update for a particle is the deterministic move plus
the random kick. Writing the deterministic velocity as `u` and the random
standard-normal draws as `R` (mean zero, variance one):

$$
\Delta x = u\,\Delta t \;+\; R\,\sqrt{2\,K_h\,|\Delta t|} .
$$

The deterministic part scales with `dt`; the random part scales with the *square
root* of `dt` — that is what makes the variance grow linearly in time
(`(\sqrt{2K_h\,\Delta t})^2 = 2K_h\,\Delta t`), matching §1. Diffusion is
**anisotropic**: the zonal and meridional directions carry separate diffusivities
(`kh_zonal`, `kh_meridional`), so the two axes get independent kicks with their
own amplitudes and their own normal draws.

Two sign rules make the kick correct under both forward and backward integration:

- The **advection** (the deterministic velocity term) uses the *signed* `dt` —
  positive when tracking forward in time, negative when tracking backward — so
  advection
  reverses cleanly.
- The **noise** amplitude uses `√|dt|`, the magnitude only. Random spreading is
  irreversible: a cloud spreads going forward *and* going backward, so the kick
  magnitude must not change sign. This is why the schemes below take `|dt|` under
  the square root while the advection keeps the signed `dt`.
- The **diffusivity-gradient drift** (`∂K`, the Itô correction of the
  variable-`Kh` schemes and of the vertical `Az` kick) is part of the diffusion,
  not of the advection, so it also uses `|dt|`. Backward tracking with
  diffusion follows the adjoint of the advection–diffusion equation, which
  for incompressible flow is `dX = (−u + ∇K) dτ + √(2K) dW`: the advection
  reverses, the drift and the spreading do not. (Until 2026-09-23 the drift
  used the signed `dt`; a backward ensemble then drifted into low-diffusivity
  water — stationary density ∝ `K⁻²` in a closed column instead of uniform.)
  This is still backward advection with Fickian spreading, not the
  conditional time reversal of a stochastic process (Thygesen 2011); the
  backward cloud is not a validated source-probability distribution.

The metre-space kernels (`diffusion_uniform_step_meters` and the EM/M1 kernels)
return `(dx, dy)` in metres; the integrator moves the particle along the great circle.


## 3. Three schemes

KELP offers three diffusion schemes, selected by `numerics.diffusion.scheme`
(`configuration.md` §3.1). They differ in whether the diffusivity is a single
constant or a spatially-varying field, and — when it varies — in how carefully
they correct for that variation.

### `uniform` — constant diffusivity, plain Brownian kick

The diffusivity is a single pair of scalars taken from the config
(`kh_zonal`, `kh_meridional`), constant everywhere. The displacement is the pure
Brownian increment with no drift term, implemented by
`diffusion_uniform_step_meters`:

$$
\Delta x = \sqrt{2\,K_x}\,\sqrt{|\Delta t|}\,N_1,
\qquad
\Delta y = \sqrt{2\,K_y}\,\sqrt{|\Delta t|}\,N_2,
$$

where `(N_1, N_2)` is an independent standard-normal pair. Because `Kh` is
constant there is no spatial gradient to worry about, so this scheme adds nothing
but the random kick. It is the only scheme available in 3D.

### `em` — Euler-Maruyama, with the drift correction for varying `Kh`

When the diffusivity varies in space — read from the dataset as a field rather
than a config scalar — a naive random walk produces a physical artefact. A pure
random kick whose *amplitude* depends on the local `Kh` spends more time, and
takes shorter steps, in low-diffusivity regions, so particles spuriously pile up
where `Kh` is small. That is not what real diffusion does: Fick's law moves tracer
*down* the diffusivity gradient, away from the bias, leaving a uniform tracer
field uniform.

The fix is a deterministic **drift correction** equal to the gradient of the
diffusivity, `∂Kh/∂x`. Added to the advecting velocity, it exactly cancels the
spurious accumulation and restores Fickian behaviour. This is the Euler-Maruyama
discretization of the stochastic differential equation

$$
dx = \bigl(u + \partial_x K_x\bigr)\,dt \;+\; \sqrt{2\,K_x}\;dW,
$$

with the analogous equation in `y`. Here `dW` is the Wiener increment, realized
as `N·√|dt|`. The implementation is `diffusion_em_step_meters`: the
advection `u` multiplies the signed `dt`, the drift `∂Kh/∂x` multiplies `|dt|`,
and the `√(2Kh)·√|dt|·N` noise uses `|dt|`. The value of `Kh` and the gradient `∂Kh/∂x` are sampled from the dataset
field at the particle position — a least-squares plane fit for the value and a
central difference at `±dres_m` for the gradient, with a fallback chain when an
offset point falls off the mesh. Those sampling formulas live in
`interpolation.md` §5.

### `m1` — Milstein, one order higher (with a caveat)

The Milstein scheme (`diffusion_m1_step_meters`) adds the next term in the
stochastic Taylor expansion — a quadratic-variation correction proportional to
`∂Kh · (N² − 1)` — on top of the Euler-Maruyama step:

$$
\Delta x = u\,\Delta t + \partial_x K_x\,|\Delta t|
         + \tfrac{1}{2}\,\partial_x K_x\,|\Delta t|\,(N_1^2 - 1)
         + \sqrt{2\,K_x}\,\sqrt{|\Delta t|}\,N_1 .
$$

When `∂Kh = 0` this reduces exactly to Euler-Maruyama. The extra term is only a
rigorous higher-order improvement for a **separable** diagonal diffusivity
(`Kx = Kx(x)`, `Ky = Ky(y)`): the cross-derivative iterated-integral terms that a
general two-dimensional varying `Kh` would require are not implemented. When `m1`
is used with a genuinely spatially-varying field the code emits a single
`RuntimeWarning` per realization and process (keyed by seed, realization and scheme, so a
fresh blended `Kh` array every step does not repeat it, and a second run in the same
process with the same seed stays silent) to flag the limitation. For general 2-D `Kh`,
`em` is the safer choice.


## 4. Counter-based random numbers (reproducible randomness)

A naive Monte Carlo run pulls draws from a single global generator advanced one
sample at a time. That breaks two things KELP needs: parallel execution
(threads racing for the next draw give run-to-run-different results) and restart
(resuming from a checkpoint cannot replay the exact stream a continuous run would
have produced). KELP sidesteps both by using a **counter-based** generator:
there is no evolving state at all. Every draw is a pure hash of a key tuple

$$
(\text{seed},\ \text{real\_id},\ \text{particle\_i},\ \text{step\_i},\ \text{component}).
$$

The hash is `_philox_uint64` — a five-key, order-sensitive mixer built from
SplitMix64 rounds (swapping any two keys changes the output). Despite the
name, the algorithm is **not** Philox: the function is counter-based in the
spirit of Salmon et al. (2011) — from whom the name stuck — but the mixing
itself is a keyed SplitMix64 chain (Steele, Lea & Flood 2014). Its 64-bit result
is turned into a pair of independent standard normals by `_normal_pair`, which
splits the bits into two uniforms and applies the Box-Muller transform. The
horizontal kick consumes that pair under the `COMP_HORIZONTAL` component slot: one
normal for the zonal axis, one for the meridional, drawn together and pairwise
independent.

Because a draw depends only on its key and not on any history, the stream has two
properties that matter:

- **Bit-identical regardless of thread count and order.** Any thread can compute
  any particle's kick at any step in any order and get the same number, because
  the number is a function of the key alone.
- **Restart-safe.** Resuming at `step_i = N` reproduces exactly the draws a single
  continuous run would have made, because step `N`'s key does not depend on step
  `N−1` ever having run.

**The trap.** `particle_i` in the key must be the particle's **original** index in
the `ParticleState` arrays — not the compacted index it gets after dead particles
are squeezed out of an alive-only loop. Using the compaction index would change a
surviving particle's key the moment any earlier particle died, silently altering
its random stream mid-run. The loop kernels are written to pass the stable
original index.

**Shards.** A run over one contiguous block of a larger release keys its draws by
the GLOBAL index `pid_offset + i` (`DiffusionStepArgs.pid_offset`,
`BeachingStepArgs.pid_offset`; resolved by `run_tracking` from
`data.particle_offset` or the `# kelp particle_offset N` header that
`utils/split_seeds.py` writes into every shard file; `run_single_realization`
and `track_particles` take `particle_offset=`). With one common seed, a sharded
run merged by `utils/concat_trajectories.py` equals the unsharded run bit for
bit (`tests/test_sharding.py::TestStochasticShards`, 2026-10-04). A pulsed
release (`time.n_pulses > 1`) replicates the seed set and cannot be sharded:
an offset above 0 with `n_pulses > 1` is refused. The merge refuses shards whose
offsets do not follow the merge order when they draw random numbers (output
attribute `random_draws`, stamped by `run_single_realization`, so chain drivers
carry it too) and only warns for deterministic shards. Shards that are independent by
design — a pulsed run split through its restart, one seed per shard — merge with
`independent_shards=True` / `--independent-shards` (stamped on the merged store).

Two knobs control the ensemble. The `seed` (`numerics.diffusion.seed`) makes a run
reproducible: same seed, same trajectories. The `real_id` distinguishes
realizations: when `num_runs > 1`, each realization gets a different `real_id`, so
the same seed produces a family of **independent** ensembles (different random
kicks, same reproducible recipe).


## 5. Composition with the integrator

Diffusion does not run on its own; its kick has to be combined with the
deterministic step from the integrator (`time_integration.md`). How they combine
is the *composition*, set by `numerics.diffusion.composition` but normally left to
auto-resolution by `_resolve_composition` in `tracking.py`. There are three modes,
each matched to a scheme/integrator pairing:

- **`split`** — used for the `uniform` scheme with *any* integrator. The
  integrator runs its full deterministic step first; then the dispatch helper
  `_apply_split_diffusion_kick` adds the Brownian kick on top and re-validates the
  result. This is the simplest composition: deterministic move, then noise.

- **`combined`** — used for `em`/`m1` with the Euler integrator. The SDE step
  *replaces* the deterministic Euler step entirely: `_apply_combined_em_step`
  evaluates velocity, the diffusivity value, and its gradient at the start
  position and assembles the whole drift-plus-noise displacement in one kernel.
  Combined mode is **2D only — it raises `NotImplementedError` in 3D**.

- **`split_rk`** — used for `em`/`m1` with RK2 or RK4, and it must be opted into
  explicitly (any other composition for this pairing is rejected). The integrator
  runs its deterministic Runge-Kutta step, then `_apply_split_rk_em_kick` adds the
  SDE kick (drift correction plus noise) afterwards. The `kh_eval_position` key
  chooses whether the diffusivity is sampled at the `start` of the step or at the
  post-RK `end` position. This is a pragmatic extension rather than a rigorous SDE
  scheme, so it emits a one-time `RuntimeWarning` per realization; it is also 2D
  only and **raises `NotImplementedError` in 3D**.

`_resolve_composition` enforces these pairings automatically: `uniform → split`;
`em`/`m1` + Euler → `combined`; `em`/`m1` + RK → the user must set `split_rk`. The
resolved value (not the raw config value) is what gets recorded in the output
metadata. The configuration keys are documented in `configuration.md` §3.1.


## 6. What is called when

With diffusion enabled, each tracking step does the following, in order:

1. The integrator computes the deterministic move (`split` and `split_rk` modes)
   or sets up to fold the SDE drift into the move (`combined` mode).

2. The composition-specific helper runs:
   - `_apply_split_diffusion_kick` for `split`,
   - `_apply_combined_em_step` for `combined`,
   - `_apply_split_rk_em_kick` for `split_rk`.

3. Inside that helper, the random kick is drawn from the counter-based generator
   via `_normal_pair(seed, real_id, particle_i, step_i, COMP_HORIZONTAL)`, keyed
   by the particle's **original** index, and the metre-space kernel from §2/§3
   computes the displacement. For `em`/`m1` the helper first samples the
   diffusivity value and gradient at the relevant position (`interpolation.md` §5).

4. The displaced position is **re-validated** against the mesh. A particle that
   the kick pushed off the domain (interpolation failed or it left the mesh) is
   reverted to its position at the start of the step (before the deterministic
   step, not only before the kick) and marked `OUTSIDE` — the same
   revert-on-failure policy the deterministic step uses (`time_integration.md`).

5. In 3D (only `uniform`/`split`), after the horizontal kick the depth `z` is
   **re-clamped** against the new local water column, because the horizontal move
   may have carried the particle over a different bottom depth.

6. When the vertical Az kick is enabled (§7), `_apply_split_az_kick` runs last:
   it moves only `z`, inside the local column, so it needs no re-validation.

When diffusion is disabled, step 1 is the only thing that runs and the rest of
this document is dormant — the output is identical to a pure-advection run.

## 7. Vertical Az diffusion (`az_*`, 3-D runs)

Vertical turbulent mixing is modelled as a separate split kick applied after
the deterministic step and after the horizontal kick, in positive-down depth
coordinates:

```
Δz = ∂Az/∂z · |dt|  +  sqrt(2 · Az · |dt|) · ξ,      ξ ~ N(0, 1)
```

The first term is the **Itô drift correction** and it is not optional: without
it particles artificially accumulate wherever Az is small (Visser 1997) — the
stationary density of the drift-free walk is `p ∝ 1/Az`, while the physical
answer for a passive tracer in a closed column is uniform. The well-mixed
condition test (`tests/test_az_diffusion.py::TestApplySplitAzKick::
test_well_mixed_condition`) pins exactly this. The sign convention is friendly:
under the z-up → depth flip both the derivative and the axis change sign, so
the drift is `+∂Az/∂depth` in KELP's positive-down coordinates too. As in
the horizontal kernels, the drift and the noise both use `|dt|`, so a backward
run keeps the drift and the uniform equilibrium (the well-mixed test runs in
both directions).

**Az sources.** Two, selected per run:

- `az_const` (m²/s): a constant, works with every reader and both vertical
  coordinate systems (the gradient is zero, only the column bounds matter);
- `var_az` (dataset variable name, e.g. FESOM2 `Av`): a full 3-D field,
  **v1 = FESOM2 z-level runs only**. The reader loads it as a node-based
  `(K, N)` array on the nz interfaces — the same layout and `max_levels`
  trimming as `w` — zero-fills non-finite (below-bottom) entries, rejects
  mid-layer-shaped variables, and the field is time-blended exactly like the
  other fields (`VelocityField.az`, lazily in temporal-interp runs). Any
  other reader, or a sigma mesh, with `var_az` set fails at startup.

**Evaluation at the particle.** Az and its gradient come from the
**nearest-node column** (no horizontal interpolation in v1): the value is
linear in z between interface values with the evaluation depth bottom-clamped
to the node bottom; the gradient is the slope of that same linear segment
(on an interface: the segment above it) — the derivative of the interpolant
whose value sets the noise (`az_column_value_gradient`). The well-mixed
condition needs exactly this identity: the half-layer central difference used
until 2026-09-23 straddled the neighbouring interface (at 0.75 m on
Az = 1, 2, 1 m²/s at 0, 1, 2 m it gave 0.5 instead of 1.0), and the uniform
density was then not stationary even for vanishing dt. With the exact slope
the generator conserves the uniform density; the finite-dt Euler step with
reflection keeps an O(dt) bias, as before.

**Reflective boundaries.** The kicked depth is folded back into the local
column by specular reflection (`reflect_depth`) — never clamped: clamping
would pile particles onto the boundaries and break the well-mixed
equilibrium. The modulo fold handles kicks of any size (multiple bounces),
and the bounds mirror the deterministic clamps exactly:

| mesh | column |
|---|---|
| z-level | `[z_levels[0], depth[nearest node]]` (as `clamp_depth`) |
| sigma | `[0, H_p − z0b_min]`, `H_p = max(depth + eta, Dmin)` interpolated at the particle like W with the END-of-step surface (as the step's clamp, `clamp_depth_sigma(H_particle=)`); the nearest node's column when called without the step's column |

**Randomness.** One normal per particle per step from the same counter-based
generator (§4), on its own component slot `COMP_VERTICAL = 1` — independent
of the horizontal pair by construction, keyed by the original particle index
and the global step index, so parallel- and restart-safety carry over
unchanged.

**Configuration** (`numerics.diffusion`, see `configuration.md` §3.1):

| key | default | meaning |
|---|---|---|
| `az_enabled` | `false` | master switch for the vertical kick |
| `az_const` | `0.0` | constant Az (m²/s ≥ 0), used when `var_az` is unset |
| `var_az` | `null` | dataset Az variable (FESOM2 z-level runs only, v1) |

`az_enabled: false` (default) is byte-identical to the previous behaviour;
`az_enabled: true` with a zero source (`az_const: 0`, no `var_az`) is fully
dormant — no kick, no startup guards, no extra output attributes.
Fail-fasts. Two fire at config LOAD, regardless of source: `az_enabled`
under a disabled diffusion block (`enabled: false` — the block would
silently no-op; for a vertical-only run set `enabled: true` and leave the
Kh values at 0), and `az_enabled` with `scheme: em|m1` (v1 composes with
the `uniform`/split path only). With an actual source, startup
additionally rejects `az_enabled` on a 2-D run, and a `var_az` that is
missing from the dataset or lives on the `nz1` mid-layers (checked from
metadata BEFORE the output file is created). A run that configured `var_az` also refuses
to fall back to `az_const` if a velocity payload ever arrives without the
field. Multi-year FESOM2 datasets are supported: the exact per-variable
layout assembles `<var_az>.fesom.<year>.nc` (e.g. cycle1 `Av`) into the
family alongside u/v/w/ssh. Limitations (v1, staged): no horizontal
interpolation of Az, no Az field on sigma meshes, Az is not
PCHIP-reconstructed in time (linear blend only, like Kh).
