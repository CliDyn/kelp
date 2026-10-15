# KELP coordinate frame

> **Covers:** the optional Euler-rotated tracking frame for FESOM2 and HEALPix — why it is needed near the pole, what rotates (mesh points and velocity vectors) and what does not (scalar fields), and the contract that seeds go in as geographic coordinates and trajectories come out as geographic coordinates.
> **Does not cover:** the `mesh.rotation` config key and its accepted values (see `configuration.md` §4); horizontal velocity interpolation (see `velocity_interpolation.md`); the full prose walk-through of the design trade-offs (see `interpolation.md` "Coordinate frame (rotated-pole tracking)").
> **Code:** `src/kelp/core/rotation.py:euler_matrix`, `src/kelp/readers/fesom2.py:_apply_mesh_rotation`, `src/kelp/tracking.py:_geo_view_for_record`.
> **Verified at:** see `docs/doc_map.yaml` (stamped via check_doc_sync --bless).

Tracking a particle in longitude and latitude works well almost everywhere on
the sphere, but it breaks down close to the geographic poles. KELP can side-
step that breakdown by doing the whole computation in a **rotated coordinate
frame**: it spins the sphere so that the awkward point — the pole — sits over
land, far from any water a particle could reach, and then tracks in those
rotated coordinates. The feature is **opt-in**, **FESOM2 + HEALPix (fesomc/nemo reject)**, and completely
invisible from the outside: you still hand in geographic seed positions and you
still get geographic trajectories back. This document explains why the rotation
helps, the exact mathematics of the rotation, which quantities are transformed
and which are left alone, and where in the run each transformation happens.

The feature is switched on by the `mesh.rotation` configuration key, whose
accepted values are documented in `configuration.md` §4. The default (`None`)
means no rotation: the geographic path is taken and the output is byte-for-byte
identical to a build that has no rotation code at all. Everything below
describes what happens when rotation *is* requested.


## 1. Why rotate

A particle's horizontal step is taken in longitude and latitude. To convert a
velocity in metres per second into a change in longitude, the tracker divides by
the east–west metric

```
z0 = cos(lat) · METERS_PER_DEG_LAT
```

which is the number of metres spanned by one degree of longitude at that
latitude. As a particle approaches the pole this factor shrinks toward zero: the
meridians (lines of constant longitude) crowd together and finally all meet at
the pole. Dividing the velocity by a quantity that is heading to zero makes the
longitude update *ill-conditioned* — tiny velocity errors turn into enormous,
meaningless swings in longitude, and exactly at the pole the longitude of a
point is undefined altogether. Latitude/longitude is simply a poor chart for the
neighbourhood of a pole.

The cure is to choose a *different* chart whose pole lies somewhere the
particles never go. A rigid rotation of the sphere does exactly that: it leaves
all the geometry intact (distances, areas, the flow itself) but relabels every
point with new longitude/latitude values measured from a rotated axis. If the
rotated axis is aimed so that the computational pole lands on solid ground, the
singular point is on land, and every ocean point — including the Arctic and any
trajectory that crosses straight over the geographic pole from one side to the
other — sits at a comfortable, well-behaved rotated latitude.

KELP uses the standard FESOM frame for this, named
`FESOM2_ANGLES = (50, 15, −90)` (degrees) in `core/rotation.py`. With these
angles the computational pole sits on Greenland. This is not an arbitrary
choice: it is FESOM2's *own* rotated frame, the one the model's grid is defined
in, and using it lets the rotated mesh reproduce the model's stored element
areas to machine precision. For an Arctic or trans-polar run, rotating into this
frame is what makes the tracking well-behaved. The feature is FESOM2 + HEALPix (fesomc/nemo reject); the
FESOM-C reader rejects `mesh.rotation` outright, because it has no path for
rotating its velocity vectors.


## 2. The rotation, mathematically

A rotation of the sphere is captured by a single 3×3 matrix built from three
Euler angles. `euler_matrix(α, β, γ)`
(`src/kelp/core/rotation.py:euler_matrix`) returns that matrix `R`. The
convention throughout is that `R` acts on a point's 3-D unit vector to send the
**geographic** vector to its **rotated-frame** image:

$$
\mathbf{v}_{\text{rot}} = R \, \mathbf{v}_{\text{geo}},
\qquad
\mathbf{v}_{\text{geo}} = R^{\top} \, \mathbf{v}_{\text{rot}},
$$

so the inverse rotation is just the transpose `R.T` (a rotation matrix is
orthogonal, so its inverse equals its transpose).

**Point maps.** To rotate a *position* given as longitude/latitude, the code
converts it to a 3-D unit vector on the sphere, multiplies by `R` (or `R.T` for
the inverse), and converts back to longitude/latitude, wrapping longitude to
`[−180, 180)`. These are `rotate_lonlat` and `unrotate_lonlat`. A position is
a point on the sphere, so this is the whole story for positions.

**Tangent-vector maps.** A velocity is not a point; it is a *tangent vector* —
an arrow lying flat against the sphere's surface, with an eastward component `u`
and a northward component `v`. Rotating the sphere also rotates that little
arrow, and because the rotation is a proper (orientation-preserving) rotation,
its effect on the arrow *at any given point* is a pure 2-D rotation in the local
tangent plane:

$$
\begin{pmatrix} u' \\ v' \end{pmatrix}
=
\begin{pmatrix} c & -s \\ s & c \end{pmatrix}
\begin{pmatrix} u \\ v \end{pmatrix},
\qquad
\begin{aligned}
u' &= c \, u - s \, v, \\
v' &= s \, u + c \, v,
\end{aligned}
$$

where `(c, s)` are the cosine and sine of the local rotation angle at that
point. The factors `(c, s)` are obtained once, by feeding a unit eastward vector
(`u = 1, v = 0`) through the full vector transform at the geographic position:
its rotated components are exactly `(c, s)`. (Feeding unit-north through gives
`(−s, c)`, the second column of the matrix, confirming it is a rotation.)
Because a rotation preserves length, `u' ² + v' ² = u² + v²`: the speed of the
flow is unchanged, only its direction is re-expressed in the rotated frame's
east/north axes.

The function `rotate_vectors(u, v, lon_geo, lat_geo, R)` does the full transform
and takes the **geographic** positions of the vectors; it computes the rotated
images of those positions internally and returns the rotated-frame components
there. `unrotate_vectors` is the exact inverse. These are vectorised NumPy
routines run **once** per mesh load and once per velocity snapshot — never
inside the per-particle loop, so they need no Numba acceleration.


## 3. What rotates and what does not

Not every quantity in a run is a position or a tangent vector, and the
distinction decides whether it is transformed.

**At mesh load** (`src/kelp/readers/fesom2.py:_apply_mesh_rotation`) the
order matters and is deliberate:

1. **Vector factors first.** The per-carrier 2×2 rotation factors `(c, s)` from
   §2 are computed *before* anything is moved, by rotating a unit-east vector at
   the **geographic** node coordinates and again at the **geographic** element-
   centre coordinates. They must be taken at the geographic positions because
   that is where the local east/north basis is defined. The factors are stored
   on the mesh, keyed `"node"` and `"element"`, in `Mesh._vec_rot_factors`, so
   each velocity layout has the correct pre-computed map.
2. **Then the coordinates are rotated as points.** The node longitudes/latitudes
   and the element-centre longitudes/latitudes are passed through
   `rotate_lonlat` and replaced by their rotated images. Crucially the element
   centres are rotated **as points**, not recomputed from the rotated nodes: the
   stored centre is a circular mean of its nodes, and a circular mean is *not*
   rotation-equivariant (the centroid of the rotated triangle is not the
   rotation of the original centroid). Re-deriving it would silently shift the
   centres, so the code moves the existing centre as a point and leaves it
   consistent with how it was built.

After this, the mesh's stored coordinates live entirely in the rotated frame,
and the nearest-node search structures (the grid hash and, lazily, the KD-tree)
are built on those **rotated** coordinates — so localization runs natively in
the frame the particles are tracked in.

**At each velocity snapshot** (`read_velocities`) the eastward/northward
velocity `u, v` are rotated by the pre-computed 2×2 factor map for the field's
layout (node or element):

```
u, v = c * u - s * v,  s * u + c * v
```

For a 3-D field shaped `(K−1, X)` the carrier factors `(c, s)` of shape `(X,)`
broadcast across the vertical-layer axis, since the spatial axis is last.

**Scalar fields pass through untouched.** The vertical velocity `W` and the
sea-surface height `eta` are **frame-invariant**: they are *not* tangent
vectors with an east/north direction, so a rotation of the horizontal axes
does not change them, and each carries straight through the reader unchanged.

**`Kh` is subtler — the pair is a tensor, not a scalar.** Each `Kh` VALUE is
a scalar and rides its point unchanged (the fields are never rotated), and
the common **isotropic** case (`kh_zonal == kh_meridional`, spatial
variation included) is *exactly* frame-invariant: the tensor is `k·I` in
every frame, and the em/m1 Itô-drift gradient is computed natively in the
tracking chart, where it transforms as a proper vector. An **unequal** pair,
however, is a diagonal diffusion tensor whose axes are the FRAME's
east/north: under `mesh.rotation` the same two numbers are applied along the
ROTATED axes, so the mixing ellipse turns with the frame instead of staying
geographic — the declared behaviour (a correct treatment needs the full
tensor rotation with off-diagonal, correlated kicks, which the kernels do
not implement). Since 2026-08-18 an unequal pair under rotation emits a
one-time `RuntimeWarning` — reader-side for the em/m1 fields, run-level for
the uniform scheme's scalars. Use isotropic `Kh` under rotation.


## 4. Seeds in, geographic out

The frame is an internal detail. The contract at the boundary is simple:
**you give geographic coordinates and you receive geographic coordinates.**

**Seeds in.** Particle seed positions are always supplied in ordinary
geographic longitude/latitude. They are rotated *into* the mesh frame by
`_maybe_rotate_seeds` (`src/kelp/tracking.py:_maybe_rotate_seeds`) at the
**two** user entry points — the seed load inside `run_tracking` and the
functional `track_particles` API. When the mesh is geographic this function is a
no-op and returns the same state object unchanged. Once seeds are rotated in,
the entire internal particle state — every position the integrator touches —
lives in the rotated frame for the whole run. (The per-realization driver
`run_single_realization` does *not* re-rotate: it treats its incoming state as
already in-frame, which is also why a restart state, saved in-frame, resumes
correctly.)

**Geographic out.** At every point where a trajectory is recorded,
`_geo_view_for_record` (`src/kelp/tracking.py:_geo_view_for_record`)
produces a **geographic copy** of the state for the writer. It un-rotates the
positions back to geographic longitude/latitude, and — when velocity
diagnostics are present — also un-rotates the recorded `vel_u`/`vel_v` vectors
back to geographic east/north components, so the velocities written alongside
the trajectory are in the frame the user expects. A tangent vector's east/north
components depend on the point it is attached to, so the velocities are
un-rotated at the step's **start** position, where the integrators evaluated
them (`rk45`: at the end of its last substep, before any diffusion kick); until 1.0 they were un-rotated at the
recorded end position, which turned each vector by the frame's change of
direction along the step (about 0.4 % at 80 °N over a daily step, up to 180°
for a step across the geographic pole). This is done on a *copy*: the
live internal state stays rotated, so the next step continues in-frame. When the
mesh is geographic the function returns the same object, preserving byte-
identity. **The output is always geographic.** Sampled scalar fields (T, S, and
any other node-based scalar) are written **as is**, because they are frame-
invariant (§3) and have nothing to un-rotate; so is `vel_w`, a vertical rate.

**Release positions.** The particle state also carries each particle's release
position (`ParticleState.release_lon/lat/z`, output format 2), taken from the
in-frame seeds when the run starts, so it lives in the rotated frame like every
other position and is checkpointed in that frame. The output variables
`release_lon` and `release_lat` are un-rotated to geographic with the same rule
as the records (`tracking._release_arrays`), so a release position and the
first record of the same particle are in the same frame
(`io_and_time.md` §5.3).

**Restarts.** A checkpoint stores the rotation angles it was written under. On
resume, `_validate_restart_frame` (`src/kelp/tracking.py:_validate_restart_frame`)
compares the checkpoint's frame against the mesh's current frame and raises a
`ValueError` on any mismatch — you cannot accidentally continue a rotated run in
the geographic frame (or vice versa), because the saved positions would be
meaningless in the wrong chart.


## 5. What is called when

Putting the pieces in sequence, here is where each transformation happens across
a run. The companion overview of the whole pipeline is `architecture.md`; this
is the rotation-specific slice of it.

**At mesh load (once):**

1. The reader resolves the requested frame with `_resolve_rotation_angles`,
   turning `None` / `"fesom2"` / `[α, β, γ]` into a concrete angle tuple or
   `None`.
2. If a frame was requested, `_apply_mesh_rotation` computes the per-carrier
   vector factors **first** (from the geographic carriers), then rotates the
   node and element-centre coordinates **as points** (§3). The factors are
   stored on the mesh; the angles are stored as `Mesh.rotation_angles`.
3. The nearest-node search structures are built on the **rotated** coordinates.

**At each tracking step:**

4. `read_velocities` rotates `u, v` into the frame with the stored 2×2 factor
   map and passes the scalars (`W`, `eta`, `Kh`) through unchanged.
5. The tracker — localization, interpolation, integration, diffusion — runs
   **entirely in-frame**, on rotated coordinates and rotated velocities. No
   rotation code runs inside the hot per-particle loop.

**At each record point:**

6. `_geo_view_for_record` un-rotates positions and the `vel_u`/`vel_v`
   diagnostics onto a **copy** for the writer; the internal state stays rotated
   (§4). The release positions are un-rotated once, when
   `run_single_realization` writes the `release_*` variables at its start.

When `Mesh.rotation_angles` is `None`, every one of these hooks is dormant: each
returns the same object it was given, the search structures are built on the
unmodified geographic coordinates, and the run is byte-for-byte identical to one
with no rotation code present at all.
