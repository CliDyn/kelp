# KELP validation and performance

> **Covers:** how the tracker's correctness is established — analytical
> known-answer benchmarks, statistical and conservation checks, and
> determinism/regression guardrails — and how its performance is characterised
> in absolute terms (throughput, per-step cost, thread scaling).
> **Does not cover:** the methods being validated themselves (see
> `architecture.md`, `time_integration.md`, `diffusion.md`, `vertical.md`);
> configuration keys (see `configuration.md`).
> **Code:** `tests/test_3d_analytical.py`, `tests/test_diffusion.py`,
> `tests/test_integrators_time_aware.py`.
> **Verified at:** see `docs/doc_map.yaml` (stamped via check_doc_sync --bless).

This document is the answer to two questions an agent or a reviewer will ask
before trusting a trajectory: *how do we know the numbers are right?* and *how
fast and how big can a run be?* The first half describes the validation
strategy — what kinds of correctness the test suite establishes and how. The
second half describes how a run's performance is measured. Neither half repeats
the methods themselves; those live in the subsystem documents linked above.
This is about the evidence that they work, and the yardsticks for how well.


## 1. Validation philosophy

Correctness in KELP is established against **known answers**, not by
inspecting plots and judging them plausible. Every claim the test suite makes
falls into one of three families, and each family is anchored to something
that can be computed independently of the code under test:

1. **Closed-form analytical trajectories.** For a handful of simple flows the
   exact particle path is known on paper — a constant velocity gives a straight
   line, a rotational flow gives a circle, a known time-dependent velocity gives
   an integrable path. The tracker is run on that flow and its output is
   compared to the formula. This is the strongest kind of test: there is a
   single right answer and the code either lands on it (to within the
   integrator's expected truncation error) or it does not.

2. **Statistical and conserved quantities.** Some behaviour has no single right
   trajectory but a right *distribution* or a right *invariant*. A cloud of
   diffusing particles must spread at a predictable rate; a particle in a steady
   incompressible flow must conserve its streamfunction. These are checked over
   an ensemble or along a path rather than point-by-point.

3. **Determinism and regression.** A great deal of the code exists in two forms
   — a readable reference path and an optimised "fused" path — and the optimised
   path is only allowed to exist because it produces *bit-identical* output.
   The same applies to restarting a run from a checkpoint and to the
   random-number generator. These tests do not check physics; they check that
   refactoring and parallelism never silently changed an answer.

A practical property of the whole suite is that it is **synthetic-data-driven**.
Almost every test builds its own small mesh and velocity field in a temporary
directory fixture — the canonical case is a 3×3 node grid of triangles centred
at (5°, 5°) with constant or analytically prescribed velocity — so the suite
runs anywhere with no external files to download. The real-mesh smoke test
(`tests/test_real_mesh.py`) exercises the readers against small committed
fixtures cut from genuine model meshes (it skips only if those files are
missing); the suites that need external model data — the HEALPix twin and
W-sign suites — read their location from `KELP_TEST_DATA` and skip without it.


## 2. Analytical benchmarks

The analytical tests pin the *accuracy order* of the integrators: how fast the
error shrinks as the time step `dt` is reduced. They come in three flavours,
corresponding to the three things a flow can do.

**Constant velocity (linear trajectory).**
`tests/test_3d_analytical.py` builds local meshes at a range of geographic
locations — the equator, the prime meridian, the dateline, near the pole — and
prescribes a constant velocity. With constant flow the displacement per step is
exactly `Δlon ≈ U·dt / (cos(lat)·METERS_PER_DEG_LAT)` in the horizontal and
`W·dt` in the vertical, so the particle's position after any number of steps is
a closed form. Running this at several latitudes catches metric-term mistakes
(the `cos(lat)` longitude scaling) and confirms all three integrators reproduce
a straight line exactly, which they must, since a first-order scheme is already
exact on a constant field.

**Time-dependent velocity (the time-aware test).**
A constant field cannot distinguish a first-order from a fourth-order
integrator. `tests/test_integrators_time_aware.py` therefore uses a spatially
uniform but **time-dependent** velocity whose exact trajectory is integrable,
and compares each integrator's error against that closed-form solution. This is
where the Runge–Kutta stages earn their keep: when the integrator is told about
the *next* velocity snapshot (the time-aware mode described in
`time_integration.md`), the RK2 and RK4 stages sample the flow at the correct
intermediate times and the error against the exact path collapses by more than
an order of magnitude relative to the steady-velocity step; when they are not
told, every scheme degrades to first order in time. The assertions are ratios:
the time-aware error must fall below the steady-step error by a factor of 10
(RK4) and 5 (RK2). An important ceiling is pinned elsewhere, in
`tests/test_temporal_reconstruction.py`: because the velocity between two
stored snapshots is reconstructed by **linear interpolation in time**, the
time-truncation error can be no better than `O(dt²)` no matter how high the
spatial RK order is — halving the step cuts the RK4 error by a factor of about
four, not sixteen. The fix is not a better integrator but denser snapshots, `dt`
much smaller than the snapshot spacing, or the PCHIP reconstruction, under which
RK4 separates from RK2.

In short, these tests demonstrate that the time-aware stages pay off by large
factors, while flagging that the *time* order is capped by the snapshot
interpolation. See `time_integration.md` for the stage
mechanics behind this.

**Rotational and 3-D flows with closed-form solutions.**
`tests/test_analytical_3d_benchmarks.py` and the 3-D cases in
`tests/test_3d_analytical.py` extend the same idea to flows where the path
curves and to vertical motion. A solid-body rotation traces a circle of known
radius and period, so the integrator's tendency to spiral inward or outward
(its amplitude error) is directly measurable; the 3-D cases additionally check
that vertical displacement and the layer/interface bookkeeping behave as the
geometry predicts.


## 3. Statistical and conservation checks

Some of the physics has no single correct trajectory, only a correct
*statistic* or a correct *invariant*. These tests run an ensemble, or run one
path and watch a quantity that should not change.

**Diffusion — Brownian variance growth.**
The random-walk diffusion model (see `diffusion.md`) is correct when a cloud of
particles spreads at exactly the rate the diffusivity prescribes. For a constant
horizontal diffusivity `Kh`, the displacement variance after time `t` must grow
as

$$\sigma^2 = 2\,K_h\,t.$$

`tests/test_diffusion.py` checks precisely this: it draws the per-step
displacement for a large ensemble (10⁵ particles), and asserts that the sample
variance of each component matches `2·Kh·dt` to within a few percent — a
tolerance set from the statistical error of a variance estimate over that many
samples — while the sample mean stays at zero (an unbiased kick). The
anisotropic version uses different `Kh` in the two directions and checks that
each component reproduces its own variance independently, confirming the two
axes do not leak into each other.

**Diffusion — spatially varying diffusivity.**
When `Kh` varies in space, a naive random walk would pile particles up in the
low-diffusivity regions — a well-known spurious accumulation. The
variable-diffusivity schemes carry a deterministic drift-correction term
precisely to prevent this, and the tests verify that the resulting particle
distribution has the predicted shape with no false accumulation. Several of
these distribution-shape and moment test cases were adapted from Parcels to
cross-check the results against an independent implementation of the same
stochastic model. Those cross-checks are marked slow (they need a sibling
reference setup), so they are deselected from the default run.

**Vertical streamfunction conservation.**
`tests/test_downwelling_psi.py` exercises a steady two-dimensional overturning
(downwelling) cell in the vertical. Because the flow is steady and
incompressible, a streamfunction ψ exists and is constant along every particle
path — pathlines coincide with streamlines. The test integrates particles
through the cell and asserts that ψ evaluated at each particle stays bounded
near its starting value, i.e. that the vertical-velocity interpolation
(see `vertical.md`) does not inject or destroy mass. It additionally confirms
the smoother interpolation methods conserve ψ about as well as the nearest-
node fallback (least squares within 10 % of it on the coarse mesh; barycentric
strictly better than nearest and idw on the fine one), turning a physical
conservation law into a numerical ranking.


## 4. Determinism and regression

The third family of tests protects answers across change. They are what makes it
safe to optimise: any rewrite that alters a trajectory by even one bit fails
loudly.

**Fused vs unfused bit-identity.**
Every integrator exists as a readable, dispatcher-style "unfused" path and an
optimised, single-kernel "fused" path (see `time_integration.md`). The fused
path is only justified if it is numerically indistinguishable from the
reference. Regression tests run both on the same inputs and assert the
trajectories are **bit-identical**, which also nails down summation order — a
subtle source of last-bit drift when a loop is reorganised for speed.

**Restart reproducibility.**
A run can be checkpointed and resumed (see `io_and_time.md`). The contract is
that resuming from step *N* and continuing produces exactly the same
trajectory, bit-for-bit, as a single uninterrupted run. This is tested directly
and is non-trivial because it requires every piece of evolving state — particle
positions, the wet-streak counter for beaching, and the random-number stream —
to be captured and restored faithfully.

**Counter-based RNG determinism.**
The diffusion kick uses a *counter-based* random-number generator: every draw is
a pure function of `(seed, realization_id, particle_index, step_index,
component)` rather than a running stream (see `diffusion.md`). The tests in
`tests/test_diffusion.py` verify the consequences of that design — identical
inputs give identical draws; and changing any one key changes the draw. From
that follows, by construction, the most important property: the result is
**independent of thread count and execution order**, because each draw depends
only on its key. No test varies the thread count; the property was measured on
the HEALPix twin (bit-identical across 1–64 Numba threads, `healpix.md`). This
is what lets diffusion run on many threads and still reproduce the
single-threaded answer exactly. The same property is what makes restart reproducibility above possible:
resuming at `step_index = N` regenerates the identical continuation.

Together these guardrails are why the performance work in the next section is
allowed to exist — the optimised and parallel paths are held to the standard of
producing the reference answer unchanged.


## 5. Performance characterisation

Performance is reported in **absolute terms** — how many particle-updates per
second, how long a step takes, how peak memory grows, and how the work scales
with the number of threads. There is no comparison to any other tool; the
numbers stand on their own as a description of what a run will cost.

**What is measured.** Performance is characterised by a few quantities:

- **Wall time** — the total elapsed time of a run, what a user actually waits.
- **Particle-loop time** — the time spent purely in the per-step particle loop,
  measured separately from the one-time startup.
- **Per-step cost** — the loop time per step, in milliseconds; `--verbose`
  breaks it into the orchestration phases `io` (snapshot read and blend),
  `step` (the integrator, incl. its nearest-node and containment searches),
  `beach` (when beaching is on) and `rec` (record writing) — the phases inside
  a fused kernel are not separable.
- **Throughput** — particle-updates per second, derived from the above.
- **Peak memory** — the high-water resident set, which scales with mesh size and
  particle count and determines whether a run fits on a node.

`kelp run --verbose` reports per-step milliseconds and a first-25%/last-25%
slowdown comparison; the absolute cost model, per-particle kernel costs, and
sizing recipes live in `performance.md`.

**Why startup is separated from the loop.** A run has a large fixed cost paid
exactly once: loading and indexing the mesh (building the search structure for
an 11-million-node global mesh takes seconds) and compiling the Numba kernels on
their first call (the JIT warm-up). On a short run with few particles this fixed
cost *dominates the wall clock* and would completely mask the per-step loop
behaviour — a 100-particle benchmark could spend almost all of its wall time in
mesh load and report a meaningless "throughput". Isolating the loop time
(the tracking-phase step time, with the first few steps dropped as warm-up —
`benchmarks/single_node/run_bench.py` reports the mean of the final five) measures the
thing that actually grows with particle count and step count, which is the
quantity that matters for scaling up a campaign. Timing should warm the JIT
before measuring and skip the initial steps to keep startup out of the measured
loop.

**Thread scaling.** The particle loop is the parallel hot path — a Numba
`prange` over particles — so scaling is characterised by sweeping the thread
count (`NUMBA_NUM_THREADS`) and recording loop time at each setting. The loop time falls as
threads are added (more particles processed in parallel); the **per-record
output writing does not scale** with threads, since it is a serial,
I/O-bound append to the trajectory file. The practical consequence is that on a
write-heavy configuration (frequent output records) the writer can become the
bottleneck even as the compute loop keeps getting cheaper, so throughput is best
read together with the output cadence, not in isolation.

The specific numbers — milliseconds per step on a given CPU, particles per
second at each thread count, peak RSS — depend on the machine, the mesh, and the
thread allocation; the measured rows kept in this repository are in
`performance.md` and `benchmarks/`, the full campaign logs live with the
campaigns. What lives here is the *methodology*: the quantities
above, the separation of startup from loop, and the thread sweep.

### Where a step spends its time

A step does two interpolations, and they cost very differently:

- **Spatial interpolation** — evaluating the velocity *at each particle's
  position* (the least-squares plane fit over the elements around the nearest
  node). This is irreducibly **per particle**: a particle sits at an arbitrary
  continuous point, not at a mesh node, so the interpolation *is* the act of
  evaluating there — it cannot be precomputed once for the whole field. Its cost
  scales with the particle count.
- **Temporal interpolation** — blending the velocity field between two file
  snapshots to get the field "now". This is a **whole-field** operation: on a
  global mesh it touches tens of millions of elements and nodes, independent of
  how many particles there are.

Which one dominates depends on the ratio of particle count to mesh size. With a
*small* cloud on a *large* mesh, materialising the entire blended field every
step — only to read a few thousand values out of tens of millions — is the
overwhelming cost; the per-particle work is milliseconds. With a *dense* cloud
that covers the mesh (a particle in most elements), the per-particle spatial
fits dominate and the whole-field blend is amortised over many reads.

KELP blends the field **lazily**: each field component is interpolated only
when something reads it, and the fused horizontal kernels blend just the elements
a particle actually touches, inline, rather than building the whole field first.
This automatically picks the right strategy — it blends almost nothing for a
small cloud, and degrades to roughly the materialised cost (but parallel) for a
dense one. The change is exact: trajectories are bit-identical with and without
it.

The effect, measured on one HPC node at 128 threads (loop time only, same code
otherwise, results bit-identical) — illustrative, not a fixed benchmark, and not
recorded in this repository beyond this table:

| configuration | particles | regime | loop speedup |
|---|---|---|---|
| Arctic 2D, full global mesh, 2400 steps | 1 000 | field-blend bound | ~85× |
| Arctic 2D (Fram / pole), full mesh, 1 year | 1 000 | field-blend bound | ~10–13× |
| Arctic 3D, full mesh, T/S sampled | 1 000 | field-blend bound | ~8× |
| Svalbard 3D regional, T/S sampled | 1 000 | field-blend bound | ~7× |
| Arctic 2D, whole-ocean seeding | 1 000 000 | particle-loop bound | ~1.2× |
| coastal regional meshes (small) | 1 000 | small mesh | ~1.0–1.3× |

The pattern is the point, not the exact factors: the speedup is large exactly
where the whole-field temporal blend dominated the step (a few particles on a big
mesh with temporal interpolation), and near one where the per-particle loop
dominates (a million particles) or the mesh is small enough that blending it is
already cheap.


## 6. Running the checks

The correctness suite is fast and self-contained:

```bash
python -m pytest tests/          # synthetic data, no external files needed
```

The default selection runs the analytical, statistical, determinism, and
regression tests described above. The slow diffusion cross-checks need a sibling
reference setup and are run separately:

```bash
python -m pytest tests/ -m slow
```

The real-mesh smoke test (`tests/test_real_mesh.py`) runs on committed fixtures;
the data-gated suites skip without `KELP_TEST_DATA`, so a clean checkout still
gets a green run.

For run timing, use `kelp run --verbose`, which prints per-step milliseconds
and a first-25%/last-25% slowdown comparison. The cost model, thread-scaling
guidance, and sizing recipes are in `performance.md`.
