# Interpolation methods in KELP

> **Covers:** the formula-level reference for every interpolation KELP
> performs — horizontal `u`/`v`, vertical `W`, scalar sampling, `Kh`, temporal,
> and polygon vertex weights — each cross-checked against the implementing
> function.
> **Does not cover:** which method to choose / the call-flow (see the subsystem
> docs `velocity_interpolation.md`, `vertical.md`, `field_sampling.md`,
> `diffusion.md`); config keys (see `configuration.md`).
> **Code:** `src/kelp/numerics/interpolate.py`,
> `src/kelp/numerics/vertical.py`, `src/kelp/numerics/vertical_uv.py`,
> `src/kelp/core/mesh.py`.
> **Verified at:** see `docs/doc_map.yaml` (stamped via check_doc_sync --bless).

This document is the formula-level reference for every interpolation the tracker
performs. It describes **the code**, not the textbook: each method below is
cross-checked against the live function it documents, and every claim links down
to the `module.function` that implements it. Function references use dotted paths
relative to the `kelp` package (e.g. `numerics.interpolate._interp_leastsq_core`).

KELP interpolates several distinct quantities, each with its own method
selection and code path:

- **Horizontal velocity `u, v`** at the particle — one of three methods selected
  with `numerics.uv_interp`: a least-squares plane fit over the surrounding
  element centres (`leastsq`, default), a containing-element vertex blend
  (`barycentric`), or a FESOM-style fit over the containing element and its
  edge neighbours (`edge_leastsq`).
- **Vertical velocity `W`** — a vertical linear profile between interfaces,
  combined horizontally by one of four methods (`numerics.w_interp`).
- **Sampled scalar fields** along trajectories — the same four horizontal methods
  (`numerics.sample_interp`), with a direction-free vertical profile.
- **Diffusivity `Kh`** (em/m1 schemes) — a least-squares plane fit; the value
  lookup is gated by mesh containment (the gradient builds on gated values, §5).
- **Velocity in time** — linear interpolation between file snapshots
  (`numerics.temporal_interp`).
- **Polygon vertex weights** — the shared primitive under barycentric `W` and
  scalar sampling: classical barycentric on triangles, Wachspress on quads.

All horizontal fits work in a **local metric plane** (degrees scaled to
approximate metres), with coordinates centred on the particle to avoid
catastrophic cancellation in the normal-equations determinant (§1).
The longitude scale is `z0 = cos(lat) · METERS_PER_DEG_LAT` with
`METERS_PER_DEG_LAT = 111120` m/deg (the nautical-mile convention,
1852 m × 60), the latitude scale is `METERS_PER_DEG_LAT` directly.

| Config key | Default | Quantity | Methods | Code (function) |
|---|---|---|---|---|
| `numerics.uv_interp` | `leastsq` | horizontal `u, v` | `leastsq` / `barycentric` | `numerics.interpolate._interp_leastsq_core`, `._interp_barycentric_core` |
| `numerics.w_interp` | `leastsq` | horizontal combine for `W` | `nearest` / `leastsq` / `barycentric` / `idw` | `numerics.vertical.interp_w_spatial` |
| `numerics.sample_interp` | `leastsq` | sampled scalar fields | same four | `numerics.sampling.sample_field` |
| `numerics.temporal_interp` | `false` | velocity in time | linear lerp | `core.io.VelocityField.lerp` |
| (diffusion `em` / `m1`) | — | `Kh` value + gradient | LSQ plane + central diff | `numerics.diffusion._kh_value_at_xy`, `._kh_gradient_at_xy` |
| `mesh.rotation` | `None` | coordinate frame | `None` / `fesom2` / `[α,β,γ]` | `core.rotation` |

---

## Coordinate frame (rotated-pole tracking)

Every interpolation above runs in a **local metric plane** built from the
particle longitude and latitude, with the longitude scale
`z0 = cos(lat)·METERS_PER_DEG_LAT`. Near a geographic pole this metric
degenerates: a step `Δlon = dt·U / (cos φ · M)` blows up as `cos φ → 0`, and the
eastward/northward `u, v` decomposition winds around the pole, so a single mesh
cell can span a huge longitude range. The default geographic tracker is therefore
unusable in the immediate neighbourhood of either pole.

### The opt-in rotated frame

The mesh config key `mesh.rotation` moves the whole computation into a **rotated
sphere** whose poles sit far from any ocean point:

- `None` *(default)* — no rotation, the geographic frame. The default path is
  **byte-identical** to before this feature (a geographic run produces the same
  output bytes, including global attributes).
- `"fesom2"` — the standard FESOM Euler angles **(50, 15, −90)**, which place the
  computational pole over Greenland. This is FESOM2's *own* rotated frame: with
  these angles the Euler matrix reconstructs the model's `elem_area` to machine
  precision (validated against `fesom.mesh.diag.nc`).
- `[α, β, γ]` — a custom 3-list of Euler angles in degrees.

It is supported on FESOM2, HEALPix and FESOM 1.4 (fesomc/nemo reject it). The FESOM-C reader **rejects** `mesh.rotation` with a
`ValueError` (FESOM-C is a coastal model with no near-pole use case and no
vector-rotation path). **Both** readers emit a one-time `RuntimeWarning` when the
mesh reaches `max|lat| > 88°` while rotation is OFF; the FESOM2 message recommends
`mesh.rotation: fesom2`, the FESOM-C message does not (there is no rotation to
recommend). The angles are validated in `core.config.MeshConfig.__post_init__`,
resolved to a concrete tuple by `readers.fesom2._resolve_rotation_angles`, and
stored on the loaded mesh as `core.mesh.Mesh.rotation_angles` (`None` = geographic).

### What rotates and what does not

The rotation is applied **once at mesh load / velocity read**, never inside the
particle loop, by `readers.fesom2._apply_mesh_rotation / readers.healpix._apply_rotation`. The per-carrier
vector-rotation factors are computed **first** from the *geographic* carriers (they
need the geographic tangent basis), then the stored coordinates are rotated.

| Quantity | Treatment under rotation |
|---|---|
| Mesh node coords, element centres | rotated **as points**¹ (`rotate_lonlat`) |
| Velocity `u, v` | rotated as **tangent vectors** via per-carrier 2×2 factors² |
| User seed positions (in) | rotated into the mesh frame at the two entry points |
| Output positions, `vel_u`, `vel_v` (out) | un-rotated back to **geographic** |
| `W`, `eta`, `Kh`, sampled scalars, areas | **frame-invariant** — passed through untouched |

¹ Element centres are rotated **as points**, *not recomputed* from the rotated
nodes: the circular-mean centroid carrier is **not rotation-equivariant** (the
rotated triangle's own centroid lies elsewhere), so re-deriving it would change the
discrete object. This footnote is the dominant mechanism behind the accuracy note
below.

² The tangent-plane image of a proper rotation is a pure 2-D rotation. The factor
pair `(c, s)` comes from rotating unit-east at the carrier (element centres for the
`element` layout, nodes for the `node` layout, stored in
`core.mesh.Mesh._vec_rot_factors`); unit-north maps to `(−s, c)`, so
`read_velocities` applies `u' = c·u − s·v`, `v' = s·u + c·v` (broadcast across the
layer axis in 3D). Scalars (`W`, `eta`, `Kh`) are frame-invariant and pass through.
Anisotropic uniform diffusion (`kh_zonal != kh_meridional`) applies its ellipse
along the **rotated** frame's axes (Kh is declared frame-invariant by design; the
common isotropic case is exactly rotation-invariant).

### Geographic output, rotated internal state

**Output is always geographic.** The internal `ParticleState` and the checkpoints
live in the rotated frame; at every record point `tracking._geo_view_for_record`
un-rotates positions and the `vel_u`/`vel_v` diagnostics onto a **copy** (the
internal state stays rotated, so integration is unaffected). Sampled scalars are
*not* un-rotated — they are frame-invariant and sampled on the rotated mesh.
`sample_vars` is for true scalars; a node-based **vector** component (e.g. `unod`)
listed there would be sampled in the rotated frame and is **not** un-rotated.
Visualization helpers un-rotate mesh coordinates for display, so plots are
geographic too. Seeds
flow the other way: `tracking._maybe_rotate_seeds` rotates user/geographic seed
positions into the mesh frame at the two user entry points (`run_tracking` seed
load and `track_particles`); `run_single_realization` treats its `init_state` as
already in-frame (restart states arrive rotated).

Checkpoints store the frame as the scalar attributes
`restart_rotation_alpha/beta/gamma` (both NetCDF and Zarr backends);
`writers.load_restart` returns them as `rotation_angles`. A resume **validates**
the frame: the caller-driven recipe runs `tracking._validate_restart_frame`, which
raises a `ValueError` if the checkpoint frame differs from the current
`mesh.rotation`. The trajectory file also carries a `rotation_angles` global
attribute (only when rotation is on — geographic runs stay byte-identical).

### Accuracy note (measured)

With rotation **ON**, a mid-latitude run differs from the geographic run by a tiny
amount — **~1.2 × 10⁻⁶ degrees** at lat 5°N (3 steps, dt = 3600 s, RK2, element
layout). This is a **legitimate discrete chart difference**, not a bug:

- The dominant mechanism is the **non-rotation-equivariant circular-mean centroid
  carrier** (the footnote above): the reader evaluates element velocities at the
  geographic centroid and rotates that point, but the rotated triangle's centroid
  lies elsewhere — an O(1), resolution-independent carrier shift. The analytic
  chart-metric term (~3 × 10⁻¹⁰) and the leastsq fit error (~1.7 × 10⁻⁷) are
  sub-dominant.
- It **converges with `dt`**: the carrier-shifted velocity is interpolated and
  integrated, so the position difference carries the RK time-truncation factor.
  The **element** layout converges roughly **quadratically** (measured ratio ~3.5
  halving `dt`); the **node** layout (`unod`/`vnod`, which requires
  `uv_interp: barycentric`) converges only **first order** (ratio ~2.0), because
  barycentric weights are themselves not rotation-equivariant.
- A genuine **frame bug** — a missing, doubled, or wrong un-rotation — would be
  **O(tens of degrees)** (measured 41.9° for a missing un-rotate) and
  **`dt`-independent** (a constant rotation offset, convergence ratio exactly 1.0).
  The order-of-magnitude gap is how the test suite separates the two.

Near the pole the trade is decisive: the geographic tracker **fails entirely**,
while the rotated tracker follows great circles to **sub-metre accuracy**. The
transpolar test seeds a pure rotated-meridional flow through a mesh that genuinely
covers the geographic North Pole (a node lands on `|lat| = 90`); the geographic
longitude flips ~180° across the crossing, and the maximum 3-D chord error to the
analytic great circle is **6.8 × 10⁻⁸ m** (68 nanometres — a spatially constant
rotated field makes the rotated-frame motion exact under RK2).

**Code:** primitives `core.rotation` (`FESOM2_ANGLES`, `euler_matrix`,
`rotate_lonlat` / `unrotate_lonlat`, `rotate_vectors` / `unrotate_vectors` — the
vector functions take **geographic** positions); mesh-load application
`readers.fesom2._apply_mesh_rotation / readers.healpix._apply_rotation` / `_resolve_rotation_angles` /
`_warn_near_pole_fesom2`; per-step seed/record/resume handling
`tracking._maybe_rotate_seeds`, `._geo_view_for_record`, `._validate_restart_frame`.

---

## 1. Horizontal velocity (`u`, `v`)

The horizontal velocity at a particle is reconstructed by one of three methods,
selected with `numerics.uv_interp` (default `leastsq`): the **least-squares plane
fit** over element centres (below), the default path; the
**barycentric** containing-element vertex blend (§1.1); or the **edge-neighbor
least-squares** fit over the containing element and its edge-sharing neighbors
(§1.2, FESOM-style). All run in 2D and 3D (in
3D each particle uses its own vertical layer to slice the field first). The
registry `numerics.interpolate.get_interpolator(name)` resolves the config name to
the vectorised interpolator and `get_interpolator_single` to the single-particle
twin (the single-point API for tests and direct callers; RK stage loops use the
vectorised form via `_interp_uv_at`); an unknown name raises with the valid list.

### Least-squares plane fit (`leastsq`, default)

The least-squares method fits a plane to the **element-centre** velocities of the
elements that ring the particle's nearest mesh node. It is exact for any linear
velocity field over the stencil and is the default reference implementation,
left byte-for-byte untouched by the interpolation-registry work.

For each element `e` in the 1-ring of the nearest node, with centred metric
coordinates `Δx_e = lon_e·z0 − x_p`, `Δy_e = lat_e·SLM − y_p`, the fit minimises

$$ \min_{a_0,a_1,a_2} \sum_{e \in \mathrm{ring}} \bigl(a_0 + a_1\,\Delta x_e + a_2\,\Delta y_e - f_e\bigr)^2 . $$

Because the coordinates are centred on the particle (`Δx = Δy = 0` at the
particle), the interpolated value is simply the constant term:

$$ f(\text{particle}) = a_0 . $$

The normal equations form a symmetric `3×3` system accumulated in a single pass
over the ring; it is solved by **Sarrus' rule** for the determinant and
**Cramer's rule** for `a_0` (the off-diagonal cofactors are computed but only the
first row is needed for the value). A near-singular matrix returns
`ok = False` via a **two-stage guard** (2026-08): `|det| < 1e-30` for the
exact-zero case OR the scale-relative test
`|det| < 1e-12 · |a11·a22·a33|` — the Gram entries are metric (metres²),
so a rank-deficient stencil's pure-roundoff determinant is ~1e6…1e24 and
an absolute threshold alone never fires (Cramer then divides noise by
noise; measured 8 m/s out of a 0.5 m/s constant field at corners with
two incident elements). Well-posed stencils keep
`det/(a11·a22·a33) ≳ 1e-3`; rank-deficient ones sit at machine epsilon.
The same guard protects the tinterp/PCHIP/edge twins, the W plane fits
(z-level and sigma) and the Kh plane fits;
and the integrator marks that particle `INTERPOLATION_FAILED` and reverts its
position.

Centring the coordinates on the particle is what makes the determinant
well-conditioned: a legacy absolute-coordinate variant of the same fit (kept
only for comparison) suffers catastrophic cancellation in the Sarrus
determinant — element-centre longitudes near the dateline are `O(10⁷)` in
metres, so the squared sums in the normal matrix lose precision. Centring the
stencil on the particle keeps every coordinate small and the determinant
well-scaled.

The `time_dir` factor (`+1` forward, `−1` backward) multiplies each element
velocity inside the fit, so backward tracking reverses the velocity field.

**Antimeridian.** Each stencil element's longitude delta carries a **gated
wrap** (applied only when the raw centre-minus-particle difference exceeds
180°): a stencil that straddles ±180° is corrected by one full turn, while
every off-seam stencil keeps the historical arithmetic **bit-for-bit** — the
Fortran-parity contract now holds everywhere except at the seam, where the
raw arithmetic placed stencil elements ~360° away and corrupted the fit
(fixed 2026-07; pinned by translation-invariance tests in
`tests/test_leastsq_seam.py`). The same gate lives in the time-aware and
PCHIP twins; the legacy `_interp_leastsq_core_fortran` comparison kernel is
deliberately left untouched.

**Code:** `numerics.interpolate._interp_leastsq_core` (steady), and the
time-aware variant `numerics.interpolate._interp_leastsq_core_tinterp` which
blends two snapshots inline per element
(`u_e = (1−α)·u_a + α·u_b`) before accumulating the normal equations. The
vectorised wrapper is `numerics.interpolate.interp_leastsq`.

### 1.1 Barycentric vertex blend (`barycentric`)

The barycentric method finds the **single element that contains the particle**
and blends that element's vertex velocity values with the polygon vertex weights
of §7. Containment is the same 1-ring scan (with the 2-ring fallback for
jittered meshes) used everywhere else
(`core.mesh._find_containing_element_local`); the vertex weights come from the
same dispatcher `core.mesh._polygon_weights` — classical barycentric on
triangles, guard-free product-form Wachspress on quads (§7). The
element's vertices are projected to a local metric plane around the **element
centroid** (dateline-safe `Δlon`, `cos(centroid lat)` scaling, `R = mesh.r_earth`; the
longitude origin is anchored on the first vertex so an element across ±180° keeps its
shape) before the weights are formed.

The vertex *values* that are blended come from one of two sources, depending on
where the file stores `u, v`:

- **Element layout** (`(E,)` 2D / `(K-1, E)` 3D, FESOM's element-centre default).
  Each vertex value is the **area-weighted average** of the velocities of the
  elements that share that vertex,

  $$u_n = \frac{\sum_{e \ni n} A_e\, u_e}{\sum_{e \ni n} A_e},$$

  where `elems(n)` is `mesh.node_to_elems[n, :count]` and `A_e` is the element
  area from `core.mesh.build_elem_areas`. **By default the areas are read from
  the model files**: the FESOM2 mesh diag always carries `elem_area`, and a
  FESOM-C NetCDF supplies them whenever the model wrote them out — bit-parity
  with the model build, pole-correct. The **spherical-excess computation is only
  the fallback** for area-less sources (e.g. ASCII meshes): each area is then
  **spherical excess × `mesh.r_earth²`** (the area on the unit sphere as a sum of
  fan-triangle spherical excesses, scaled by the radius), which is **pole-safe and
  frame-invariant** — unlike the old planar `cos(lat)` shoelace, which collapsed
  polar-cap areas to near zero. The per-family
  radius defaults are FESOM2 `6_367_500` m and FESOM-C `6_400_000` m (set by the
  reader; overridable via `mesh.r_earth`). This is FESOM's own element→node
  projection convention, correct on graded meshes; **areas enter only as ratios**,
  so the radius choice cancels here. A vertex whose surrounding areas sum to zero
  makes the interpolation fail (`ok = False`) for that particle.
- **Node layout** (`(N,)` / `(K-1, N)`, e.g. FESOM2 `unod`/`vnod`). The vertex
  value is a **direct lookup** `u[n]` — no averaging step.

The blend is `U = Σ_k w_k u_{n_k}`, `V = Σ_k w_k v_{n_k}`, then multiplied by the
`time_dir` sign (the same `+1` forward / `−1` backward convention the leastsq
kernel bakes).

**Exactness.** Constant fields are reproduced exactly in **both** layouts.
Globally **linear** fields are reproduced exactly in the **node** layout only
(the vertex values are exact and the weights reproduce affine functions); the
**element** layout is **smoothing by construction** — the area-weighted vertex
projection is not exact for a globally linear field (this is a declared property,
not a bug).

**Global continuity.** Because both weight families (barycentric and Wachspress)
reduce to linear interpolation along a shared edge in the two shared vertices,
the barycentric field is **continuous across element edges**. This contrasts with
`leastsq`, whose value jumps where the nearest node — and therefore the fitting
stencil — changes.

**Edge behaviour (stricter than leastsq).** Barycentric **requires** containment:
a particle with no containing element in the 1-ring (2-ring) returns `ok = False`
and the integrator reverts it to `OUTSIDE` with `INTERPOLATION_FAILED`.
`leastsq` needs only a nearest node, so it can succeed for a
particle lying slightly **outside** the mesh.

**Performance.** Barycentric is a non-default method and mostly runs on the
**unfused** integrator paths (like the non-default `W` methods); the fused
leastsq 2D and fused 3D Euler Numba kernels bake the leastsq plane fit and are
bypassed whenever `uv_interp != "leastsq"`. **Exception (2026-08): 2-D
node-layout barycentric with a next snapshot supplied** (discrete runs with a
known file-time axis and `temporal_interp` runs alike) takes its own fused
RK2/RK4 kernels — `_interp_barycentric_core_tinterp` is the untouched
barycentric geometry with the leastsq-tinterp nested two-level time blend
applied **per vertex** (bracket pair + `alpha0` from a `LazyLerpField`, stage
target `vel_next`; `alpha0 == 0` and the stage-endpoint alphas 0/1 collapse to
the direct value exactly, same non-finite-propagation rationale as the leastsq
twin and `_make_stage_vels`). With `mesh.exact_nearest: false` fusing is
additionally gated on the grid-hash density certificate
(`grid_hash_dense_for_elements`), under which the in-kernel nearest anchor is
bit-identical to the batch KD-healed one; under the default guard every fused
kernel searches exactly in-kernel and the gate passes on any mesh. Element
layout, steady runs, 2-D PCHIP bundles, walk-carried elements and (guard off)
certificate-denied meshes (HEALPix polar rows) stay unfused (see
docs/time_integration.md §5). Choosing a non-default method is not warned
about, by the same convention as the non-default `W` methods — but a
certificate DENIAL (guard off) on an otherwise-eligible run does warn once per mesh
(`RuntimeWarning`), so the silent perf downgrade is visible. Combined em/m1 diffusion (the
only composition that REPLACES the Euler step) evaluates drift with the leastsq
core and fails fast when `uv_interp != "leastsq"` or velocities are node-based.

**Code:** `numerics.interpolate._interp_barycentric_core` (single point),
time-aware node-layout twin `_interp_barycentric_core_tinterp`,
vectorised wrappers `numerics.interpolate.interp_barycentric` (2D) /
`_loop_interp_bary` / `_loop_interp_bary_3d`, single-particle entry
`interp_barycentric_single`; element areas `core.mesh.build_elem_areas`
(model-provided when present, else spherical-excess × `mesh.r_earth²` — pole-safe,
frame-invariant; `-1`-padding and FESOM-C repeated-first-vertex safe, lazy `(E,)`
cache).

### Node-based velocity input

`barycentric` also unlocks a new **input regime**: velocities already projected
onto **nodes** rather than element centres (FESOM2 `unod`/`vnod`).
`core.io.VelocityField.uv_layout` (`"element"`, the default, or `"node"`) tags
which mesh location `u, v` live on; it is validated in `__post_init__` and a
layout mismatch between two snapshots raises in `.lerp`. Readers **auto-detect**
the layout per variable from the dim names (node hints `node`/`nod2d`/… vs
element hints `nele`/`elem`/`elements`/…, with a shape fallback against
`mesh.n_nodes`/`n_elements`); `u` and `v` must agree or the read raises. FESOM2
honours `mode.var_u` / `mode.var_v` (e.g. `var_u: unod`, `var_v: vnod`) before
the default `"u"`/`"v"` names; FESOM-C already had `var_u`/`var_v` and gains the
same detection (its node-layout layered fields use a node-aware vertical
reduction, since the element-hardcoded reducer cannot serve node fields).

The method×layout rule is **fail-fast, no silent fallback**: node-layout
velocities with `uv_interp: leastsq` (or `edge_leastsq`) raise a `ValueError` at
startup (both are element-centre math) with an actionable message pointing at
`uv_interp: barycentric`. The check fires in two places — a **metadata-only**
pre-writer probe in `run_tracking` (`reader.probe_uv_layout`, which inspects
dims/sizes and never loads data) and a **step-0 backstop** in
`run_single_realization` (covers readers without the probe and direct callers).

### 1.2 Edge-neighbor least squares (`edge_leastsq`)

The FESOM-style elemental interpolation: instead of fitting over the (up to
~8-element) ring around the nearest **node**, the plane is fitted over the
**containing element and the elements sharing an edge with it** — 3 neighbors
for a triangle, 4 for a quad, so the stencil is always 4 (tri) or 5 (quad)
points. The adjacency table is `core.mesh.build_elem_neighbors` (lazy
`(E, max_verts)` int32 cache, slot `k` = neighbor across edge
`(verts[k], verts[k+1])` of the compacted vertex ring, `-1` = boundary edge;
this is FESOM's own `elem_neighbors` array reconstructed from the
connectivity).

The fit itself is the same particle-centred metric normal system as `leastsq`
(§ above) — value = the constant term `a_0` — with one difference: the
longitude deltas are **wrapped** (`((Δlon + 180) mod 360) − 180`), so the
stencil is dateline-safe by construction.

**Boundary treatment.** A boundary edge has no neighbor element; the missing
stencil row is replaced by a **ghost point**: the containing element's centroid
mirrored across that edge (reflection in the local metric plane), carrying the
lateral boundary condition selected by `numerics.edge_leastsq_bc`:

- `freeslip` (default) — ghost value = the **mirror image** of `u_own` across the
  edge: the tangential component is kept, the normal component reversed
  (`u_g = 2(u·t̂)t̂ − u` with `t̂` the edge tangent in the metric plane). A flow
  parallel to the wall is reproduced exactly; a flow into the wall is damped
  towards zero normal velocity at the wall (a soft constraint of the
  over-determined plane fit, not an exact zero — e.g. |v_n| < 25 % of the
  ambient normal speed at the wall midpoint in the pinned test). Until
  2026-09-23 the ghost copied the whole vector `+u_own` — a zero-gradient
  condition that let the reconstructed flow cross the coast.
- `noslip` — ghost value `−u_own`: the fitted plane passes through ~zero at the
  boundary edge, decaying the interpolated velocity toward the coast. By
  construction this is **not** constant-exact in boundary elements — it imposes
  the wall condition instead. **Known artifacts of the single-plane fit** (pinned
  by `test_noslip_overshoot_and_corner_reversal_characterized`): tilting the
  plane to hit ~0 at the wall makes it **overshoot the ambient speed on the
  interior side of the same element** (~+60 % for one wall edge), and in a
  **corner cell with two wall edges the plane crosses zero inside the cell**, so
  velocities near the corner point *against* the ambient flow. If these are
  unacceptable for your use case, prefer `freeslip` (tangential flow kept, only
  the normal component damped).

**Exactness.** Linear fields are reproduced exactly in **interior** elements
(the stencil points then lie on one plane; both 4- and 5-point fits are
consistent). In boundary elements exactness degrades to the chosen boundary
condition (see above). Like `leastsq` — and unlike `barycentric` — the fitted
field is **discontinuous across element edges** (the stencil switches with the
containing element rather than with the nearest node).

**Edge behaviour.** Like `barycentric`, the method **requires containment**
(same `_find_containing_element_local` scan); a particle with no containing
element returns `ok = False` → `OUTSIDE` / `INTERPOLATION_FAILED` on the
unfused path. Element layout only: node-layout velocities fail fast (see the
method×layout rule above).

**Performance.** Non-default method → **unfused integrator paths only** (the
fused kernels bake leastsq math), like `barycentric`. Per evaluation the fit
itself is cheaper than `leastsq` (4–5 points vs the node ring), but the
containing-element search dominates; the edge-walk element tracking
(`numerics.elem_search: walk`, shipped 2026-07 — search once at release, then
cross edges incrementally; `localization.md` §7) is what turns this into the
cheap path. Combined em/m1 diffusion fails fast when
`uv_interp != "leastsq"`, which includes this method.

**Code:** `numerics.interpolate._interp_edge_leastsq_core` (single point),
vectorised wrapper `interp_edge_leastsq` (+ `_loop_interp_edge_lsq` /
`_loop_interp_edge_lsq_3d`), single-particle entry
`interp_edge_leastsq_single`; adjacency `core.mesh.build_elem_neighbors`.

### 1.3 Depth-linear `u`/`v` in the vertical (`mode.uv_vertical: linear`)

By default a 3-D layer-centred `u`/`v` is piecewise constant in depth: every
vertex reads the row of the particle's layer. A depth-linear field
(`VelocityField.uv_vertical` `"layer_linear"` — what `mode.uv_vertical: linear`
makes of layer data — or `"level"`) is evaluated by `barycentric` only (§1.1),
each vertex `i` at the particle depth `z` (positive-down metres):

$$ z_i = \max\big(p_0,\ \min(z,\ p_{\ell_i})\big), \qquad
f_i(z) = (1-\alpha)\,f_{k,i} + \alpha\,f_{k+1,i}, \qquad
\alpha = \frac{z_i - p_k}{p_{k+1} - p_k}, $$

with `p_k <= z_i < p_{k+1}`, and `f_i = f_{k,i}` read directly when `z_i` is on
a point or at an end (of two coincident points the deeper — except at
the surface end: at or above `p_0` point 0 is read even when point 1
coincides with it; a spacing `< 1e-12 m` reads `p_k`, the shallower). The data points `p` are the layer
centres `(z_k + z_{k+1})/2` (`"layer_linear"`, `K-1` points) or the levels
(`"level"`, `K` points), and `ℓ_i` is the node's last usable point,

$$ \ell_i = \max\Big(0,\ \min\big(M-1,\ \max(\{k : \mathrm{lower}_k \le d_i\} \cup \{0\}),\ D_i\big)\Big), $$

`lower_k = p_k` for levels, `z_{k+1}` (the layer's lower level) for layers, `d_i`
the node's bed depth and `D_i` an optional reader declaration (`vertical.md`
§2.1). The velocity is then the barycentric blend of the vertex values with the
weights of §1.1, `U = time_dir · Σ_i λ_i f_i(z)`. Points below `ℓ_i` are never
read. In time-interpolated runs each stored value `f_{k,i}` is first blended in
time (linear or PCHIP, §6), then this rule applies.

**Code:** `numerics.vertical_uv.depth_points` (`p`, `ℓ`), `uv_bracket`
(`k`, `α`), `uv_at`; the vertex loop `numerics.interpolate._interp_barycentric_core_dl`
and its fused time-blend twins `_interp_barycentric_core_{tinterp,pchip4}_dl`.

## 2. Vertical `W` profile — linear between interfaces

`W` is stored at vertical interfaces (`(K, N)`, node-based). Before any
horizontal combination, the value at a node is obtained by **linear
interpolation between the two interfaces that bracket the particle depth**, with
hard clamping to the surface and bottom interface (no vertical extrapolation):

$$ W(z) = (1-\alpha)\,W_k + \alpha\,W_{k+1}, \qquad \alpha = \frac{z - z_k}{z_{k+1} - z_k}, $$

where `z_k ≤ z < z_{k+1}` are the bracketing interface depths
(positive-down metres). If `z` is at or above the top interface it returns `W_0`;
at or below the bottom interface it returns `W_{K-1}`; a degenerate layer
(`Δz < 1e-12`) returns the value at the shallower interface of the bracket, `W_k`.

When this profile is sampled at a *neighbour* node during a horizontal combine
(§3), the evaluation depth is **bottom-aware**: `z_eval = min(z, node_bottom)`,
so a shallow neighbour never contributes a spurious below-bottom `W`.

The W kernels bake the integration sign into the result: the returned value is
multiplied by `time_dir`. This is why scalar sampling (§4), which must be
direction-free, does **not** reuse these kernels directly but interpolates the
profile separately with `time_dir = +1`.

**Code:** z-level profile `numerics.vertical._interp_w_linear_single`.
For sigma meshes the interfaces are reconstructed per water column from the local
geometry: column thickness `H = max(depth + eta, D_min)`
(`numerics.vertical_sigma.column_depth`), interface depths
`z = (1 − σ)·H`
(`numerics.vertical_sigma.sigma_interfaces_to_z`), and the same linear profile in
`numerics.vertical_sigma._interp_w_sigma_linear_single`.

## 3. Horizontal combine for `W` — the four methods (`numerics.w_interp`)

After the vertical profile (§2) gives a `W` value per node, the value at the
particle's horizontal position is combined from the 1-ring stencil by one of four
methods selected with `numerics.w_interp` (default `leastsq`):

- **`nearest`** — the vertical linear profile evaluated at the **nearest node**
  only (bottom-aware). Note this is *not* a nearest-interface lookup: it is
  vertically linear, horizontally nearest-node (it dispatches to the plain
  `_loop_interp_w_linear` loop).
- **`leastsq`** *(default)* — a least-squares plane fit over the 1-ring
  **nodes** (centre node plus the unique nodes of its elements), using the same
  centred-metric normal-equation form as §1, value `= a_0`. This stencil is the
  ring of nodes, in contrast to §1's ring of element centres.
- **`barycentric`** — find the element of the 1-ring that contains the particle
  (fan-triangulation point-in-polygon test) and blend its vertex `W` values with
  the vertex weights of §7 (Wachspress for quads, barycentric for triangles).
- **`idw`** — inverse-distance-squared weighting over the 1-ring nodes,
  $w_i = 1/d_i^2$, exact at a node (a coincident neighbour, `d < 1` m, returns its
  value directly).

**Fallback chain (never NaN):** every method degrades gracefully to the
nearest-node bottom-aware profile value when its stencil is unusable —
`leastsq`/`idw` when the ring has fewer than 3 (resp. 1) neighbours or the normal
matrix is singular, `barycentric` when no containing element is found in the
1-ring. The fallback is always a finite `W`.

**Code:** dispatcher `numerics.vertical.interp_w_spatial` with per-method kernels
`_interp_w_leastsq_single`, `_interp_w_barycentric_single`,
`_interp_w_idw_single`. The sigma twins are
`numerics.vertical_sigma.interp_w_sigma_spatial` and its `_interp_w_sigma_*_single`
kernels. They sample **along the sigma surfaces** (2026-09-23): every
neighbour column `j` is evaluated at the particle's σ, `z_j = (z/H_p)·H_j`
with `H_j = max(depth + eta, D_min)` and `H_p` the column height interpolated
at the particle by the same method (`_H_leastsq_single`, `_H_bary_single`,
`_H_idw_single`; exact for a linear bottom with `leastsq` and `barycentric`),
because a sigma model's fields are smooth along its sigma surfaces, not along
levels of constant depth (docs/vertical.md §3). `nearest` and every fallback
use the nearest column at `min(z, H)` as before; the same `H_p`
(`sigma_column_height`) is the particle's column in the sigma integrators
(`s = z/H_p`, docs/vertical.md §3).

## 4. Scalar sampling along trajectories (`numerics.sample_interp`)

Arbitrary node-based scalar fields (`(N,)` 2D, `(K-1, N)` mid-layer, or `(K, N)`
interface) are sampled at particle positions with the **same four horizontal
methods** as §3, selected by `numerics.sample_interp` (default `leastsq`). The
horizontal combine reuses the identical maths (nearest / centred-LSQ plane /
containing-element Wachspress-or-barycentric / inverse-distance-squared).

The vertical handling differs by field shape:

- **Mid-layer fields `(K-1, N)`** are interpolated **linearly between layer
  midpoints** with clamping to the first/last midpoint (no extrapolation):
  `z_mid[m] = ½(z_m + z_{m+1})` for z-level, or
  `z_mid[m] = (1 − ½(σ_m + σ_{m+1}))·H` per column for sigma. Per-neighbour
  bottom-aware clamping (`z_eval = min(z, node_bottom)`) is applied before the
  vertical interp.
- **Interface fields `(K, N)`** delegate to the `W` machinery of §3.

Sampling is **direction-free**: it represents a physical scalar, not a flux, so
every delegation hardcodes `time_dir = +1.0` — backward and forward runs sample
identically (pinned by test).

**NaN contract** (updated 2026-08-16): `NaN` is produced for dead or
uninitialised particles (`alive = False` or `nearest < 0`), and — as an
honest missing-data answer — when the particle's own nearest node carries a
fill value (or the particle sits exactly ON a fill node under `idw`).
Non-finite values at OTHER stencil nodes are filtered out of the blends:
`leastsq` skips them and needs ≥ 3 finite points (else the nearest
fallback), `idw` skips them, and a containing element with a non-finite
vertex sends `barycentric` to the nearest fallback. All-finite stencils are
bit-identical to the pre-filter arithmetic. Limitation: interface-shaped
`(K, N)` sampled fields delegate to the W kernels, which do not filter —
there a poisoned neighbour still propagates (interface scalars are rare).

**Code:** `numerics.sampling.sample_field`, horizontal kernel
`numerics.sampling._combine_horizontal`, vertical kernel
`numerics.sampling._interp_mid_linear_single`, sigma dispatch
`numerics.sampling._sample_sigma_3d`.

## 5. Diffusion `Kh` fields (em / m1 schemes)

The `Kh` pair is read in the layout the dataset stores it — `(N,)` on nodes or
`(E,)` on elements, recorded as `VelocityField.kh_layout`. **Readers never average
one layout into the other**; the layout picks the stencil below, and both
components must be present together.

The spatially-varying-diffusivity schemes (`em`, `m1`) need both the **value** of
`Kh` and its **gradient** at the particle. Both are computed from a least-squares
plane fit. The **value** lookup (`_kh_value_at_xy`) is containment-gated so off-mesh
query points are rejected rather than silently extrapolated; `_kh_gradient_at_xy`
gates only its `±dres_m` offset samples (each via the value helper) and *assumes an
in-domain centre*. Production always evaluates the value first, so the particle
position is already validated by the time the gradient runs.

**Value.** `_kh_value_at_xy` first runs a containment test
(`_find_containing_element_local`, a 1-ring scan with a 2-ring fallback for
irregular meshes): if the point is not inside a real (wet) element near the node,
it returns `ok = False`. Otherwise it evaluates a **least-squares plane fit on the
1-ring for both layouts** — node-layout fits over the ring nodes
(`_kh_lsq_node_planar`), element-layout fits over the surrounding element centres
(`_kh_lsq_elem_planar`), the same stencil `u, v` use. **The element layout is not
a barycentric path** — both layouts are LSQ. The interpolated value is the
centred-coordinate constant term `a_0`.

**Gradient.** `_kh_gradient_at_xy` uses **central differences** at `±dres_m`
(metres, converted to `Δlon = dres_m / z0`, `Δlat = dres_m / SLM`):

$$ \partial_x K \approx \frac{K(x+\delta) - K(x-\delta)}{2\delta}, \qquad \delta = \texttt{dres\_m}, $$

with a documented per-axis fallback chain:

1. both offsets in-domain → the central difference above;
2. exactly one offset in-domain → a one-sided difference against the centre value;
3. neither offset in-domain → the **analytic LSQ-plane gradient** at the centre
   (`a_1`, `a_2` from the same fit, always available for an in-domain particle);
4. the centre fit itself fails → return `ok = False`, so the integrator reverts
   the step and marks the particle `OUTSIDE`.

`dres_m` defaults to the median element edge length clipped to `[100 m, 50 km]`.

**Code:** `numerics.diffusion._kh_lsq_node_planar`,
`._kh_lsq_elem_planar`, `._kh_value_at_xy`, `._kh_gradient_at_xy` (offset sampling
via `._kh_value_at_offset`). Public test wrappers:
`numerics.diffusion.kh_value_at_point` and
`numerics.diffusion.kh_gradient_at_point`.

## 6. Temporal interpolation (`numerics.temporal_interp`)

When `numerics.temporal_interp` is enabled, velocity (and `W`, `eta`, and any
loaded `Kh` / scalar fields) is **linearly interpolated between the two file
snapshots** that bracket the simulation time:

$$ f(t) = (1-\alpha)\,f_a + \alpha\,f_b, \qquad \alpha = \frac{t - t_a}{t_b - t_a} . $$

`VelocityField.lerp` blends all carried fields with `β = 1 − α`; it raises on a
layout or shape mismatch between the two `Kh` fields and returns `kh = None` when
only one side carries `Kh`.

Two derived uses of the same lerp:

- **RK stage-time lerp.** On non-autonomous flows, RK stages must evaluate the
  velocity at the substep time `t + c_k·dt`. Each stage materialises
  `vel.lerp(vel_next, α_k)` with stage fraction
  $\alpha_k = c_k\,\dfrac{dt}{dt_{\text{to\_next}}}$ (e.g. `c = (0, ½, ½, 1)` for
  RK4). `α_k = 0` reuses the start field, `α_k = 1` reuses `vel_next`.
- **Sampling record-time re-lerp.** Output scalars are sampled from the velocity
  payload after it has been lerped to the record (post-step) time, so sampled
  diagnostics are consistent with the recorded position.

**Accuracy cap.** Linear interpolation in time caps the temporal truncation at
**O(dt²) regardless of RK order** — a higher-order RK scheme cannot beat the
piecewise-linear-in-time field. Use `dt ≪ dt_to_next` or a denser file cadence to
shrink the time-truncation error.

**Code:** `core.io.VelocityField.lerp`, `tracking._build_vel_payload` (builds the
bracketed payload `vel_a.lerp(vel_b, α)`), `numerics.integrators._make_stage_vels`
(RK stage fields), `numerics.integrators._pick_kh_eval_vel` (Kh re-lerp to
`sim_time + dt` for split-RK diffusion).

## 7. Containment and polygon vertex weights

This is the shared geometric primitive under the barycentric `W` combine (§3) and
barycentric scalar sampling (§4). It splits cleanly into a **containment test**
(is the point inside the element?) and a **weight computation** (how much does
each vertex contribute?).

**Containment.** A point-in-convex-polygon test by **fan triangulation**: the
polygon is split into sub-triangles `(0, t+1, t+2)`, and the point is inside if it
lies in any sub-triangle (barycentric weights all `≥ _BARY_EPS`, with
`_BARY_EPS = −1e-6` a small negative tolerance admitting edge points). This test
is unchanged by the Wachspress work — only the returned weights changed. The local
plane is centred on the vertex mean, `core.mesh._seam_safe_lon0`: the raw mean
longitude, or — for an element whose raw longitude span exceeds 180°, i.e. across
the dateline — the first vertex plus the mean wrapped offset (until 2026-09-24 the
raw mean put a quad split two-and-two across ±180° half a world away, and the W,
sigma-column and sampling kernels fell back to the nearest node there).

**Weights — triangles** use the classical barycentric coordinates

$$ \lambda_1 = \frac{(y_2-y_3)(x-x_3) + (x_3-x_2)(y-y_3)}{\det}, \quad \lambda_2 = \frac{(y_3-y_1)(x-x_3) + (x_1-x_3)(y-y_3)}{\det}, \quad \lambda_3 = 1 - \lambda_1 - \lambda_2, $$

with `det = (y_2−y_3)(x_1−x_3) + (x_3−x_2)(y_1−y_3)`; a degenerate triangle
(`|det| < _AREA_TOL = 1e-10`) returns the sentinel `[-1, -1, -1]`
(`core.mesh._barycentric_weights_tri`).

**Weights — quads** use **Wachspress coordinates in product form**:

$$ w_i = C_i \, A_{i+1}\, A_{i+2}, \qquad \lambda_i = \frac{w_i}{\sum_j w_j}, $$

where `A_j` is the (doubled) signed area of the triangle `(p, v_j, v_{j+1})` and
`C_i` is the (doubled) signed area of the corner triangle
`(v_{i-1}, v_i, v_{i+1})`, indices cyclic. This is the classical
`w_i = C_i / (A_{i-1} A_i)` multiplied through by `Π_j A_j`: **division-free**,
so there is no on-edge singularity — on an edge the weights reduce exactly to
linear interpolation along it, at a vertex to that vertex's indicator. Winding
needs no handling at all: each term carries three sign-flipping factors, so
clockwise input negates every weight uniformly and the normalisation cancels
it. Properties: **partition of unity** (`Σλ_i = 1`), **linear reproduction**
(`Σλ_i v_i = p`, exact for affine fields), **vertex-order invariance**, and a
**smooth** variation inside the cell and through its edges. There are **no
guards and no fallback**: convexity and non-degeneracy of quads are a *mesh
contract* — FESOM quads are always convex (reviewer guidance, 2026-07)
(`core.mesh._wachspress_weights_quad`). Quantitatively, the contract requires
corners bounded away from flat by far more than the ~1e-6 normalized
containment tolerance: the denominator's zero set first approaches the cell
at an outward distance of roughly (worst normalized corner cross-product) ×
(cell size), so for real meshes (corner quality O(1)) the margin over the
acceptance band is ~6 orders of magnitude. The contract is **checked at
load** (`core.mesh.validate_quad_convexity`, called by the FESOM-C reader
and `build_structured_mesh`): a malformed file fails loud with the offending
element ids instead of producing NaN weights mid-run (~0.4 s per million
quads, one-time).

**Dispatcher.** `core.mesh._polygon_weights(px, py, x_n, y_n, nv)` routes
`nv == 3` to the untouched triangle primitive (pure-tri meshes stay
bit-for-bit identical; its degenerate-triangle sentinel passes through) and
`nv == 4` to product-form Wachspress. Callers guarantee `nv ∈ {3, 4}` —
compacted FESOM element rings have no other arity, and no other geometry is
provided for. `core.mesh._polygon_contains` returns these dispatcher weights
when the containment test succeeds, `None` otherwise.

**History.** Quad weights went from fan triangulation (pre 2026-06-05) to
guarded division-form Wachspress with a fan fallback on edge/degenerate/
non-convex trips (2026-06-05) to the guard-free product form above (2026-07,
reviewer remark 3: the meshes have no really-bad triangles and quads are
always convex — the generality was removed). On-edge values are unchanged in
substance (both fan and Wachspress reduce to linear edge interpolation);
interior quad values match the division form to rounding. **Triangle results
are bit-identical throughout.**

## 8. What is *not* an interpolation

Two lookups are deliberately raw — no spatial blending — and are listed here so
they are not mistaken for interpolations:

- **Beaching column height.** The drying / re-wetting test reads
  `H_raw = mesh.depth[nn] + eta[nn]` at the **nearest node**, unclamped (the
  sigma machinery clamps columns to `D_min` for geometry, but drying detection
  must see the raw near-zero/negative column on tidal flats). This is a direct
  nearest-node value, not an interpolation
  (`numerics.beaching.evaluate_beaching`).
- **Nearest-node search.** Locating the nearest mesh node to a particle is a
  spatial-index query (the Numba grid hash, with a lazily-built scipy KD-tree as
  fallback), not an interpolation of any field
  (`core.mesh.build_search`, `core.mesh.find_nearest_nodes`).
