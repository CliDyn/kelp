# KELP velocity interpolation

> **Covers:** how the horizontal `u`/`v` velocity at a particle is computed once its nearest node and containing element are known — the default least-squares plane fit (and why it is particle-centered), the barycentric/Wachspress vertex blend, and node- vs element-based velocity layouts.
> **Does not cover:** finding the nearest node / containing element (see `localization.md`); the full weight formulas and normal-equation derivations (see `interpolation.md` §1, §1.1, §7); the vertical `W` velocity (see `vertical.md`); where the velocity layout is detected on input (see `io_and_time.md`, `configuration.md`).
> **Code:** `src/kelp/numerics/interpolate.py:_interp_leastsq_core`, `src/kelp/numerics/interpolate.py:_interp_barycentric_core`, `src/kelp/numerics/interpolate.py:_interp_barycentric_core_dl`, `src/kelp/numerics/interpolate.py:get_interpolator`.
> **Verified at:** see `docs/doc_map.yaml` (stamped via check_doc_sync --bless).

This document explains how the tracker turns a particle's position into a
velocity vector `(u, v)` in metres per second. By the time this step runs,
*localization* (the subject of `localization.md`) has already supplied the two
facts it needs: the **nearest mesh node** and, for the `barycentric` and
`edge_leastsq` methods, the **containing element**. From there the model fields — discrete velocities stored
on the mesh — are blended into a single value at the exact particle location.
That blend is the interpolation described here; the integrator then uses
`(u, v)` to advance the particle (see `time_integration.md`).

The companion overview is `architecture.md`. This document adds the call-flow,
the input/output contracts (including the success flag that decides whether a
particle survives the step), and the rules for *when* each method is used. For
the algebraic derivations — the normal equations, the Wachspress formula, the
polygon weights — link out to `interpolation.md` rather than expecting them
here.


Both methods start from the particle's **nearest node** delivered by the grid hash
(`localization.md`). That anchor is exact only where the window provably contains the true
nearest node; where it does not (high latitude of a rotated frame, a small explicit cell) the
least-squares ring and the containment start differ from the exact-anchor run — a stencil
perturbation, not a failure — unless `mesh.exact_nearest` is on (2026-09-12).

## 1. Three methods, one registry

There are three horizontal velocity interpolators, and the configuration key
`numerics.uv_interp` chooses between them:

- **`leastsq`** (the default) — a least-squares plane fitted to the velocities
  of the elements surrounding the nearest node (§2).
- **`barycentric`** — a vertex blend over the single element that contains the
  particle, using barycentric weights on triangles and Wachspress weights on
  quadrilaterals (§3).
- **`edge_leastsq`** — the FESOM-style elemental fit: the same least-squares
  plane, but over the containing element and its edge-sharing neighbors (3 for
  a triangle, 4 for a quad), with boundary edges replaced by mirrored ghost
  points carrying the `numerics.edge_leastsq_bc` lateral boundary condition
  (`freeslip` default / `noslip`). Formula-level details:
  `interpolation.md` §1.2.

All are reached through a small registry rather than by name-checking in the
hot loop. `get_interpolator(name, edge_leastsq_bc="freeslip")`
(`src/kelp/numerics/interpolate.py:get_interpolator`) maps the config string
to the vectorised function — `"leastsq"` → `interp_leastsq`, `"barycentric"` →
`interp_barycentric`, `"edge_leastsq"` → `interp_edge_leastsq` (a
`functools.partial` binding the non-default boundary condition when asked) —
and raises a `ValueError` listing the valid names if the
string is anything else. Each method also has a single-particle twin, resolved
by `get_interpolator_single(name)`
(`src/kelp/numerics/interpolate.py:get_interpolator_single` →
`interp_leastsq_single` / `interp_barycentric_single` /
`interp_edge_leastsq_single`); these are kept as the single-point API (tests
and direct callers) — the 3-D Runge-Kutta stage loops nowadays evaluate their
stages through the vectorised `_interp_uv_at` shim instead of a per-particle
loop. The registry contract is simply: a name resolves to a
function with a fixed signature, and the rest of the code never needs to know
which method is active.


## 2. Least-squares plane fit (default)

The default method (`src/kelp/numerics/interpolate.py:_interp_leastsq_core`)
fits a **plane** to the velocity samples around the particle and reads the
velocity off that plane. The element-centre velocities of the ring of elements
touching the nearest node (typically about six around an interior node, found
via `node_to_elems` — see `localization.md` §2) are the samples; the model is

$$
u(x, y) = a_0 + a_1\,dx + a_2\,dy,
$$

and the same fit is done independently for `v`. Solving for the three
coefficients `(a0, a1, a2)` is a 3×3 linear least-squares problem; the code
builds the symmetric normal matrix and right-hand side in one pass over the ring
elements, then solves with the Sarrus determinant and Cramer's rule. The full
normal-equation assembly is written out in `interpolation.md` §1.

### Particle-centered coordinates

The key implementation choice is that the element-centre coordinates are
expressed **relative to the particle**, not in absolute degrees. Each element
centre is first scaled from degrees to approximate metres with a per-latitude
metric factor

$$
z_0 = \cos(y_p\cdot\mathrm{DEG2RAD})\cdot \mathrm{METERS\_PER\_DEG\_LAT},
$$

and then shifted so the particle sits at the origin:

$$
e_x = \mathrm{lon}_{\mathrm{elem}}\cdot z_0 - x_p\,z_0,
\qquad
e_y = \mathrm{lat}_{\mathrm{elem}}\cdot \mathrm{METERS\_PER\_DEG\_LAT} - y_p\,\mathrm{METERS\_PER\_DEG\_LAT}.
$$

Because the particle is the origin, `dx = dy = 0` *at the particle*, so the
interpolated velocity is simply the constant term:

$$
u = a_0, \qquad v = a_0^{\,V}.
$$

There is no separate evaluation step — fitting the plane and reading the value
are the same operation.

This centering is not a convenience; it is what keeps the fit numerically
sound. There exists a legacy absolute-coordinate variant of this kernel, kept
in the module only for comparison, that builds the same normal matrix from raw
`lon·z0` / `lat·metric` values. Those values are large
(hundreds of kilometres from the equator and prime meridian), and the entries of
the normal matrix then differ by many orders of magnitude. Computing the
determinant of such a matrix loses almost all significant digits to
**catastrophic cancellation** — large nearly-equal products subtracted from one
another — and the resulting velocity is corrupted. Centering on the particle
makes every coordinate a small displacement (metres, not thousands of
kilometres), the matrix entries become comparable in size, and the determinant
is computed accurately. The centered kernel above is the one actually used; the
absolute-coordinate variant is retained only as a documented reference and is
never called in a tracking run.

### Success flag and failure

`_interp_leastsq_core` returns `(u, v, ok)`. The `ok` flag is `False` in two
cases: the nearest node has fewer than one adjacent element (no samples to fit),
or the normal matrix is effectively singular — `|det| < 1e-30`, or the
scale-relative `|det| < 1e-12 · |a11·a22·a33|`, which is the test that actually
fires on a rank-deficient ring (the Gram entries are in metres², so roundoff
alone gives determinants of 1e6 and more; `interpolation.md` §2). When `ok` is `False`, the
caller treats the particle as having left the usable domain: its position is
reverted to the pre-step value and its status is set to OUTSIDE (the same
handling localization applies on a containment miss). A successful fit returns
the velocity already multiplied by the `time_dir` sign, so backward tracking
(`time_dir = -1`) reverses the flow with no extra branching.


## 3. Barycentric / Wachspress vertex blend

The alternative method
(`src/kelp/numerics/interpolate.py:_interp_barycentric_core`) does not fit a
plane. It locates the single element that **contains** the particle (via
`_find_containing_element_local`, see `localization.md` §5), reads a velocity
value at each of that element's vertices, and blends them with weights that
depend on where the particle sits inside the cell.

**The weights** come from `mesh._polygon_weights`
(`src/kelp/core/mesh.py:_polygon_weights`), the same dispatcher localization
uses for its containment-test geometry:

- a **triangle** (`nv == 3`) uses classical barycentric weights
  (`_barycentric_weights_tri`);
- a **quadrilateral** (`nv == 4`) uses **product-form Wachspress** weights
  (`src/kelp/core/mesh.py:_wachspress_weights_quad`) — a smooth,
  vertex-order-invariant generalisation of barycentric coordinates in which all
  four vertices contribute; the form is division-free, so it is valid on edges
  and vertices too (no guards, no fallback — convex quads are a mesh
  contract).

The weight formulas are derived in `interpolation.md` §1.1 and §7.

**The vertex values** depend on the velocity layout (§4):

- **node layout** — the velocity is stored at nodes, so the vertex value is a
  direct lookup;
- **element layout** — the velocity is stored at element centres, so the value
  *at* a vertex is reconstructed as the **area-weighted average** of the
  elements sharing that vertex (`node_to_elems`, weighted by element areas from
  `build_elem_areas`).

The blend is computed on a local metric plane built around the **element
centroid** (not the particle, as in §2). Vertex and particle longitudes are
unwrapped relative to the first vertex so a dateline-straddling cell does not
collapse its projection origin, and coordinates are scaled with `mesh.r_earth`.
The radius only sets the units of the projection: the barycentric/Wachspress
weights are scale-invariant (they are ratios of areas), so the returned velocity
does not depend on `r_earth`.

**Exactness.** The blend reproduces a constant velocity field exactly in both
layouts. It is exact for a *linear* field only in the node layout; in the
element layout the vertex reconstruction (area-averaging neighbouring element
centres) smooths the field, so linear gradients are not reproduced exactly.

As with the least-squares method, the core returns `(u, v, ok)`; `ok` is `False`
when the particle has no containing element, when the cell has fewer than three
distinct vertices, or when the weight computation degenerates — and the caller
again reverts the particle to OUTSIDE.


## 4. Node vs element velocity layout

FESOM-family models can store horizontal velocity in two places on the mesh, and
`VelocityField.uv_layout` records which one a given run uses:

- **element layout** (the FESOM default) — `u` and `v` live at element centres,
  one value per element. This is the natural input for the least-squares fit,
  which is built around element-centre samples.
- **node layout** — `u` and `v` live at mesh nodes (for example the `unod` /
  `vnod` fields), one value per node.

How each interpolator handles the two:

- **`leastsq`** is *element-centre math*: it fits a plane to the velocities of
  the elements around the nearest node. It has no meaning for node-stored
  velocity.
- **`barycentric`** handles both, branching on `uv_layout`: node values are read
  directly at the cell vertices, element values are area-averaged onto the
  vertices first (§3).

Because of this, `leastsq` on node-layout velocity is **rejected fail-fast**.
The tracking layer validates the combination (`_validate_uv_compat`): node
layout with `uv_interp = "leastsq"` raises a `ValueError` before any tracking
work, with a message directing the user to `uv_interp: barycentric`. The check
runs twice — once as a metadata-only probe before the output writer is created,
and once as a step-0 backstop inside the realization loop for callers that
bypass the probe. Where and how the layout is detected from the input files is
covered in `io_and_time.md` and `configuration.md`.

A **depth-linear** field (`uv_vertical` `"level"` or `"layer_linear"`, §7) is
node layout by contract and is evaluated by `barycentric` only: `leastsq`,
`edge_leastsq`, the nearest-element read and every single-particle entry
refuse it with a `ValueError` before reading any data.


## 5. What is called when

Inside the integrator hot loop, the chosen interpolator is invoked once per
substep, per alive particle (dead particles are skipped — see `localization.md`
§6). The call-flow depends on whether a fused Numba kernel is in play:

- **Fused 2-D kernels** (Euler / RK2 / RK4 for 2-D tracking, and the fused 3-D
  Euler kernel) **inline** `_interp_leastsq_core` directly. Two time-aware
  variants inline a sibling core that fits the same least-squares plane but
  blends the per-element value in time first: `_interp_leastsq_core_tinterp` (the
  two-snapshot linear blend) and `_interp_leastsq_core_pchip4` (the four-snapshot
  monotone-cubic PCHIP blend; see `time_integration.md` §4). The leastsq kernel
  math *is* the least-squares fit, so those kernels are only eligible when
  `uv_interp == "leastsq"` on element-layout velocity.
  **Exception (2026-08): node-layout `barycentric` with a next snapshot supplied**
  takes its own fused RK2/RK4 kernels (`_fused_rk{2,4}_step_2d_bary_tinterp` →
  `_interp_barycentric_core_tinterp`, the same nested time blend applied per
  VERTEX on the untouched barycentric geometry); with `mesh.exact_nearest:
  false` they are gated additionally on the grid-hash density certificate
  (`grid_hash_dense_for_elements` — see `time_integration.md` §5; under the
  default guard they search exactly in-kernel and fuse on any mesh). 3-D RK2
  on z-level meshes has its own fused barycentric kernel (linear or PCHIP in
  time). Element-layout barycentric, steady runs, `edge_leastsq`, walk-carried
  elements and, with the guard off, certificate-denied meshes (HEALPix polar
  rows) take the **unfused dispatcher path** instead.
- **Unfused dispatcher path** calls the vectorised registry function —
  `get_interpolator(uv_interp)` → `interp_leastsq`, `interp_barycentric` or
  `interp_edge_leastsq` —
  over the alive subset of particles. This is the path used for the non-default
  methods outside the fused-bary exception above, and the RK2/RK4 and sigma 3-D
  paths.
- **3-D Runge-Kutta stages** after the first evaluate at per-particle sample
  points fed through `_interp_uv_at` to the same vectorised (parallel) registry
  kernels via a shallow stage state — multi-pass, but not per-particle serial
  (the historical one-particle-at-a-time stage loop was parallelized in 2026-08;
  the single-particle twins from `get_interpolator_single` remain for the
  diffusion-kick revalidation and direct callers).

In every case the interpolator returns `(u, v, ok)` per particle; `ok = False`
sends that particle OUTSIDE as described in §2 and §3, and the integrator
advances only the particles that interpolated successfully.

## 6. 3-D layer-indexed barycentric cores (2026-08)

The fused 3-D RK2 kernel interpolates node-layout u/v with layer-indexed
twins of the barycentric time-blend cores:
`_interp_barycentric_core_tinterp_3d` (nested two-level linear blend per
vertex, endpoint shortcuts identical to `_make_stage_vels`) and
`_interp_barycentric_core_pchip4_3d` (per-vertex 4-snapshot
Fritsch–Carlson cubic via `_pchip_value_scalar`).  Geometry — containment
search, compacted vertex ring, centroid-anchored dateline-safe metric
projection, `_polygon_weights` — is byte-for-byte the untouched 2-D
core's; the only difference is that every vertex read happens at the
particle's mid-layer (`u[layer, node]`), because u/v are layer-piecewise
in the vertical. A depth-linear field (§7) is read linearly in depth instead,
by the `_dl` twins of these cores.

## 7. Depth-linear horizontal velocity (`uv_vertical`, 2026-10)

With `mode.uv_vertical: linear` the layer-centred u/v are linear in depth
between the layer centres (a reader with level data, `uv_vertical =
"level"`, between the levels); the rule — the data points, each node's last
usable point, constant ends, never a point below the bed — is in
`vertical.md` §2.1. `interp_barycentric` hands such a field to
`_interp_barycentric_dl` → `_loop_interp_bary_dl` →
`_interp_barycentric_core_dl`, which reads each particle at its own depth
`state.z` (the `layer` argument is ignored). The geometry — containment,
the elements hint of the edge walk, the compacted vertex ring, the
centroid-anchored projection, `_polygon_weights` — is the node-layout
`_interp_barycentric_core`'s; only the vertex value differs: at vertex `nd`,
`(k, α) = uv_bracket(p, last[nd], z)` and `u_nd = uv_at(u, nd, k, α)`
(`src/kelp/numerics/vertical_uv.py`). The record check
(`check_depth_linear_record`) runs once per field and mesh table before the
first read — a lazy blend is checked through its two parents, before anything
is blended.

The fused 3-D RK2 kernels have depth-linear twins of their time-blend cores,
`_interp_barycentric_core_tinterp_dl` and `_interp_barycentric_core_pchip4_dl`.
Per vertex they blend every stored value they need in **time first** (the
same expressions and endpoint shortcuts as the layer twins;
`_tinterp_point_uv` widens each stored value to float64 before the blend),
**then** apply the vertical rule to the blended values, **then** combine the
vertices — bit for bit the unfused path, which materializes the time blend
before `_interp_barycentric_core_dl`. PCHIP is not linear, so the order is
part of the contract (the vertical-first result differs).

Exactness: depth-uniform depth-linear data reproduce the layer answer bit for
bit wherever every vertex reads a data point directly (the depth on a point,
or clipped at either end, a NaN depth included); between points each vertex
value `(1 − α)·c + α·c` is within 1 ulp of `c`, and the combined velocity
within 2 ulp of the largest vertex value (a few more ulp of the result itself
when vertex values of opposite sign cancel)
(`tests/test_interp_depth_linear.py::test_depth_uniform_field_reproduces_layer_answer`).
