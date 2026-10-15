# KELP field sampling

> **Covers:** recording the value of a node-based scalar field (temperature, salinity, or any other) at each particle's position along its trajectory — the shape-dispatched vertical handling, the four horizontal interpolation methods, and where in the step the value is read.
> **Does not cover:** the config keys that switch sampling on (`mode.sample_vars`) and pick the method (`numerics.sample_interp`) — see `configuration.md`; the per-method horizontal weight formulas — see `interpolation.md` §4; the two-stage `W` interpolation that interface-located fields reuse — see `vertical.md`.
> **Code:** `src/kelp/numerics/sampling.py:sample_field`, `src/kelp/tracking.py:_sample_record_fields` (authoritative implementation).
> **Verified at:** see `docs/doc_map.yaml` (stamped via check_doc_sync --bless).

This document explains how the tracker records the value of an ocean field —
temperature, salinity, or any other scalar quantity stored on the mesh — at the
place where each particle happens to be, at every point along its path. The
result is a column of numbers written into the trajectory output file next to
the particle's longitude, latitude and depth, so that afterwards you can ask not
only *where did the particle go* but *what water did it travel through*.

A **scalar field** here means a single number defined at every mesh node (in
contrast to the velocity, which is a vector). "Node-based" means the field's
values live at the mesh vertices — the points whose coordinates are
`mesh.nodes_lon` / `mesh.nodes_lat`. The companion overview is
`architecture.md`; the interpolation mathematics this feature reuses are in
`interpolation.md` §4 and `vertical.md`.

## §1 What it does

Sampling records the value of a dataset scalar — for example the model's
`temp` (temperature) or `salt` (salinity) field — at each particle's position,
once per output step, and writes it into the trajectory file alongside the
positions. After a run you get, for every recorded time, a temperature (or
salinity, or whatever you asked for) for every particle, lined up with that
particle's coordinates at the same instant.

The feature is **off by default.** When no scalar variables are requested the
whole path is dormant: nothing is read, nothing is interpolated, and the output
is byte-for-byte identical to a run without sampling. It is switched on by
naming the dataset variables to sample (`mode.sample_vars`) and choosing the
horizontal method (`numerics.sample_interp`); both keys live in
`configuration.md`.

Sampling is also available directly as a standalone function,
`sample_field(...)` in `src/kelp/numerics/sampling.py`, for use outside a
tracking run — for instance to evaluate a field at an arbitrary set of points.
The tracking integration described below is a thin wrapper around it.

## §2 Shape-dispatched vertical handling

`sample_field` looks at the *shape* of the field array to decide how — or
whether — to resolve the vertical (depth) dimension before doing the horizontal
blend. A field has one value per node; depending on the model it may also carry
a vertical axis, and the position of that axis relative to the model's vertical
grid determines the rule. Throughout, `N` is the number of mesh nodes and `K`
is the number of vertical *interfaces* (the layer boundaries), so there are
`K-1` *layers* between them.

- **`(N,)` — purely horizontal.** A single value per node, no depth axis. The
  particle's depth is irrelevant; only the horizontal blend (§3) runs. This is
  the 2D case.

- **`(K-1, N)` — mid-layer field.** One value per *layer* per node, conceptually
  located at the midpoint depth of each layer. For each neighbour node the
  value is first interpolated *linearly* in the vertical between the two
  bracketing layer midpoints, clamped to the topmost / bottommost midpoint when
  the particle sits above or below the column's mid-layer range. The clamp is
  **bottom-aware per neighbour**: each neighbour node's evaluation depth is
  capped at that node's own sea-floor depth, so a shallow neighbour never
  contributes a value from below its own bottom. Only after this per-node
  vertical resolve does the horizontal blend (§3) run on the resulting node
  values.

- **`(K, N)` — interface field.** One value per *interface* per node, located at
  the layer boundaries — the same vertical layout as the vertical velocity `W`.
  This case is delegated to the existing `W` interface-interpolation machinery
  (`interp_w_spatial` for z-level meshes, `interp_w_sigma_spatial` for sigma
  meshes), reusing its two-stage scheme described in `vertical.md` — on sigma
  meshes that scheme samples the neighbour columns at the particle's σ since
  2026-09-23 (the mid-layer case above still evaluates each neighbour at the
  particle's depth, bottom-aware). Because the
  `W` kernels were built for an integration that has a direction, they take a
  `time_dir` sign argument; scalar sampling hardcodes `time_dir=+1.0`. Scalars
  are **direction-free** — the temperature at a point does not depend on whether
  the model is being integrated forwards or backwards in time — so backward
  sampling produces exactly the same value as forward sampling.

For sigma-coordinate meshes (FESOM-C) the depth of each layer or interface
depends on the local water-column thickness, which in turn depends on the
free-surface elevation. So **sigma meshes require the `eta` (sea-surface
height) field** to be supplied for any 3D (`(K-1, N)` or `(K, N)`) sampling;
`sample_field` raises a `ValueError` if it is missing. The column geometry is
then computed per neighbour node as `H = max(depth + eta, Dmin)` — `Dmin` is a
`sample_field` keyword, passed the run's `numerics.Dmin` by the record sampler
(until 2026-09-24 it was a hard-coded 0.5 m) — and the interface/midpoint depths
follow from the sigma levels. See
`vertical.md` for the sigma column geometry and `project_vertical_coords` for
which model uses which.

## §3 Horizontal combine — the four methods

Once each neighbour node carries a single (possibly vertically-resolved) value,
`_combine_horizontal` blends those node values into one number at the exact
particle position. It works over the **1-ring** of the particle's nearest node
— that node plus every node sharing an element with it — and offers the same
four methods used elsewhere in the tracker for velocity and `W`:

- **`nearest`** — the value at the nearest node only, with no spatial blending.
- **`leastsq`** (the default) — a planar least-squares fit over the 1-ring,
  evaluated at the particle. The fit is done in metric coordinates *centered on
  the particle* to avoid loss of precision; the value at the particle is then
  simply the fit's constant term.
- **`barycentric`** — locate the element of the 1-ring that contains the
  particle and blend its vertices (classical barycentric weights for triangles,
  Wachspress weights for quadrilaterals).
- **`idw`** — inverse-distance-squared weighting of the 1-ring node values;
  exact when the particle sits on a node.

The full weight formulas for all four are in `interpolation.md` §4. Note that
`_combine_horizontal` deliberately **re-implements** them rather than calling
the `W` per-method helpers: those helpers fuse the vertical interpolation and
the `time_dir` sign into one kernel and so cannot serve a direction-free,
already-vertically-resolved scalar.

The combine is **never-NaN by construction.** Whenever a method cannot produce a
result for the local stencil — too few neighbours for the plane fit, a singular
normal matrix, or no containing element found — it falls back to the
nearest-node value rather than emitting `NaN`. The only ways a sampled value is
`NaN` are a *dead particle* (one whose status is not ALIVE), a particle with
no valid nearest node (`nearest < 0`), or a nearest node whose own source
value is fill (below).

Source-field `NaN`s at stencil nodes — decoded fill values at the coast or
below the sea floor — are **filtered out of the blends** (2026-08-16):
`leastsq` skips non-finite neighbours and requires at least three finite
points (otherwise the nearest fallback), `idw` skips them, and a containing
element with a non-finite vertex sends `barycentric` to the nearest
fallback. The nearest value itself is returned raw — when the particle's own
nearest node carries fill, the honest sample is that fill value. A fully
finite stencil takes bit-identically the same arithmetic as before the
filtering. Two edges: a particle sitting exactly ON a fill node under `idw`
gets that raw fill (blending the remaining ring would invent water on a dry
spot), and interface-shaped `(K, N)` sampled fields — which delegate to the
unfiltered W kernels — still propagate a poisoned neighbour.

## §4 When it is sampled (the record instant)

Sampling does **not** run on every integration step. It runs only at the
**record points** — the steps whose state is actually written to the trajectory
output — via `_sample_record_fields`. At a record point it uses the
particle's *post-step* position and its *post-step* nearest node
(`state.nearest_node`), so the value is read where the particle ended up.

The record is labelled with the time at the **end** of the step
(`io_and_time.md` §5.2), and the field is read, as far as the forcing allows,
at that same instant. Which snapshot of the field is read depends on the time
mode (see `io_and_time.md` for the two modes):

- **Discrete mode** (one file timestep per tracking step). The scalar is read
  from the forcing record at the *end* of the step — the next record in the
  direction of the run, already in the velocity cache because the step uses it
  as its target — so the sampled instant equals the recorded `time`. Only when
  there is no next record (the last step of an `on_exhausted: stop` run that
  reaches the end of the forcing, or, backward, the first record) the
  *start-of-step* record is used, one record behind the label; the sampled
  variable's `comment` attribute says so. Until output format 2 every
  discrete-mode sample came from the start-of-step record (a lag of one
  record). The velocity diagnostics `vel_u`/`vel_v`/`vel_w` still belong to
  the start of the step (`io_and_time.md` §5.4).

- **Temporal-interpolation mode** (velocity blended between snapshots, allowing
  a sub-file timestep). The scalars *and* `eta` are **re-interpolated in time**
  to the exact post-step simulation time — the timestamp that is actually
  recorded — so the field value lines up with the recorded position with no
  lag. At the very end of the time range, when the time bracket for that
  instant cannot be resolved, it falls back to the in-hand (start-of-step)
  field.

In both modes the eta used for sigma column geometry comes from the *same*
field snapshot the scalars themselves came from, never from the mesh, keeping
the column geometry self-consistent with the sampled values.

## §5 Reading the fields

The raw fields come from each reader's `read_scalar_fields` method
(`src/kelp/readers/fesom2.py`, `src/kelp/readers/fesomc.py`,
`src/kelp/readers/healpix.py`, `src/kelp/readers/nemo.py`; the HEALPix pixel
geometry refuses sampling in v1). It reads
the named node-based variables for one timestep and trims the vertical
dimension so that it lines up with the mesh's vertical grid (`mesh.sigma`):

- FESOM2 trims an `nz1`-dimensioned variable to `K-1` layers and an
  `nz`-dimensioned variable to `K` interfaces; in 2D mode it picks a single
  level just as the velocities do.
- FESOM-C trims its sigma layer / interface dimensions the same way, and in 2D
  mode collapses the vertical axis to a single value per node with a
  node-aware reduction (layer pick, thickness-weighted mean, or surface).

The orientation is decided by the *dimension name*, not by the array shape, so
a field that happens to be square (number of layers equal to number of nodes)
is never silently transposed. A variable that is **missing** from the dataset,
or that turns out to be **element-based** rather than node-based, raises a
`ValueError`. This check runs once at the very start of `run_tracking` — before
any tracking I/O or output file is created — so a typo in a variable name fails
fast rather than partway through a long run.

Once read, the fields ride along on the velocity payload as
`VelocityField.scalars`, a `{variable_name: array}` dictionary. When two
snapshots are blended in time, the scalars are interpolated only when both sides
carry the identical set of keys (a key or shape mismatch raises). See
`io_and_time.md` for the velocity-field container and the time blending.

## §6 What is called when

At each record point, with sampling switched on:

1. `_sample_record_fields(state, mesh, vel, sample_interp, Dmin)` is invoked with the
   post-step particle state and the resolved (discrete or re-interpolated)
   field snapshot `vel`.
2. For each scalar variable in `vel.scalars`, it calls

   ```
   sample_field(values, mesh, state.lon, state.lat,
                zs=state.z if values.ndim == 2 else None,
                nearest=state.nearest_node,
                alive=(state.status == ALIVE),
                eta=vel.eta,
                method=sample_interp,
                Dmin=Dmin)          # numerics.Dmin; used on sigma meshes only
   ```

   passing depth `zs` only for 3D fields, reusing the post-step nearest node,
   and supplying the same field's `eta` for sigma column geometry.
3. `sample_field` dispatches on the field shape (§2), resolves the vertical
   dimension where present, and runs the horizontal combine (§3), returning one
   value per particle (`NaN` for dead/invalid particles).
4. The resulting `{variable_name: (P,) array}` is written into the trajectory
   output as a per-particle, per-record column next to the positions: a
   `(trajectory, time)` float64 variable with NaN as fill value, the source
   variable's `units` and `long_name`, `coordinates = "lat lon z"` and a
   `comment` on the sampling instant (`io_and_time.md` §5.5).
