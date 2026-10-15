# Offline reconstruction of the vertical velocity W (FESOM2 z-levels)

> **Covers:** how a vertical velocity is rebuilt offline from archived `u`, `v`
> and `ssh` when a FESOM2 z-level archive carries no `w`.
> **Does not cover:** how `w` is used once it exists (see `vertical.md`), or the
> configuration keys that point a run at it (see `configuration.md` §4).
> **Authoritative code:** `src/kelp/w_reconstruct.py:reconstruct_w`,
> `src/kelp/w_reconstruct.py:compare_variants`; CLI `utils/reconstruct_w.py`
> (YAML-driven, see `utils/reconstruct_w.example.yaml`).
> **Verified at:** see docs/doc_map.yaml (stamped via check_doc_sync --bless).

Some FESOM2 archives carry no vertical velocity. This module rebuilds a node-based
`w` on the vertical interfaces from the archived `u`, `v` and `ssh`, in exactly the
layout the tracker consumes (`VelocityField.w`, shape `(K, N)`, positive-down). It is
an offline approximation of what the model computes online; §4 states how far from
the model's own algorithm it is.

## 1. Sign convention

Three conventions meet here and must not be mixed:

| | typical reference derivation | KELP (`numerics/vertical.py`, this module) |
|---|---|---|
| z direction | z up, depth D ≥ 0 down | z positive-down (`mesh.sigma` holds depths) |
| w > 0 means | upward | downward (consistent with z) |
| surface condition | w_surf = +∂ssh/∂t | w_surf = −∂ssh/∂t |

FESOM2 archives store their depth axes (`zbar`, `zbar_n_bottom`) as negative
numbers growing downward; the reader takes their absolute value, so everything
downstream of it — including this module — is positive-down.

Everything below is **positive-down**, as in the tracker: `w > 0` is
downwelling, a rising sea surface gives a negative surface velocity. FESOM2's own
`w` output is positive **up** (`oce_ale.F90:vert_vel_ale`); the readers negate it
(`docs/vertical.md` §3). The reconstructed field is written positive-down and the
output variable carries `positive = "down"`, so the readers pass it through
unchanged; a file written before 2026-09-23 lacks that attribute and needs
`mode.w_positive: down`.

## 2. Continuity and the vertical integral

With z increasing downward and `w > 0` = downwelling, continuity reads

```
∂u/∂x + ∂v/∂y + ∂w/∂z = 0 .
```

Integrating from the surface (z = 0) down to depth D:

```
w(D) = w_surf − ∫₀ᴰ div_h(z') dz' ,        w_surf = −∂(ssh)/∂t .
```

**Variant A (divergence).** On the discrete interfaces k + ½ with layer thicknesses
Δz_j taken from the z-levels:

```
ssh_t   = (ssh(t_{i+1}) − ssh(t_{i−1})) / (t_{i+1} − t_{i−1})     # real intervals, not a nominal step
w_surf  = −ssh_t
w(k+½)  = w_surf − Σ_{j=1..k} div_h(j) · Δz_j
```

The raw integral does not vanish at the bottom (the offline divergence is only
approximate). The residual is taken at the node's **local bottom** `H`,
`R = w_raw(H)` (linear between the two interfaces around `H`, so a partial bottom
layer counts), before any masking, and removed linearly over the column above it:

```
w_corr(D) = w_raw(D) − (D/H) · R   for D ≤ H      →   w_corr(H) = 0 ,
```

interfaces deeper than `H` are set to zero, and the inactive-level mask
(`nlevels_nod`) is applied after the correction. Until 2026-09-23 the residual
came from the global last interface when `nlevels_nod` was absent (so `w(H) ≠ 0`
on shallow nodes) and the correction re-filled masked below-bottom interfaces.
The correction adds a uniform divergence `R/H` to the column that the horizontal
velocities do not have: it closes the bottom condition, it does not make the
field conservative.

**Variant B (ALE heuristic).** When the archive carries the model's ALE vertical
velocity `w_ALE` (the velocity relative to the moving layer interfaces of a z-star
vertical coordinate), a Cartesian estimate is

```
w_cart(D) ≈ w_ALE(D) − (1 − D/H) · ssh_t .
```

This is a **heuristic**, not the model's exact z-star kinematics: in the model not
every layer stretches with the free surface (the layers near the bottom stay fixed),
the distribution depends on the node's number of active levels rather than on the
full depth H, and the linear profile `(1 − D/H)` is a coarse stand-in. It gives the
order of magnitude, not a tracking-grade field.

**Method selection.** `method: divergence` computes Variant A only; `method: ale`
Variant B only (an explicit user choice); `method: auto` always computes Variant A
and, when `w_ALE` is available, Variant B as well, reporting the comparison
diagnostics of `compare_variants`: the absolute difference between the two
variants, and `chi_ALE` = |(1 − D/H)·ssh_t| / max(|w_ALE|, 1e-8), the size of the
ALE correction relative to `w_ALE` itself (max and mean). A mean at or above 0.5 means the
heuristic correction is as large as the field it corrects: the library entry point
`reconstruct_w` prints the diagnostics and emits a warning there. The CLI calls the
variant kernels and `compare_variants` directly and prints the same numbers without
that warning, for the FIRST output step only (later steps are not compared). Either way the user sees the disagreement instead of silently getting
the worse estimate.

## 3. SSH tendency

The centred difference above uses the **real** time intervals of the archive
(`extract_time_info` gives the times in seconds), so monthly means with 28–31-day
spacing are differenced correctly. At the first and last record a one-sided
difference is used.

## 4. Horizontal divergence: what the model does and what this module does

**FESOM2 online.** The model loops over mesh edges, forms the flux
`(v·dx − u·dy) · h_k` with the layer thickness `h_k` (depth- and ssh-dependent) and
the edge cross-vectors, scatters the fluxes to the nodes, accumulates them vertically
and divides by a **depth-dependent** control-volume area `area(nz, n)` (with an extra
volume correction in cavity configurations). That needs the edge geometry, the
per-node and per-element level counts and the per-level node areas from the mesh
diagnostic file.

**Tier 1: weak-form divergence on shape functions.** The library default
(`mesh_extras=None`); only the `Mesh` is needed. For node n, element e contributes (integration by parts supplies the sign)

```
δ_n   = −[ u_e · (y_next − y_prev) + v_e · (x_prev − x_next) ] / 2
div_n = Σ_e δ_n / A_cluster_n ,        A_cluster_n = Σ_e |A_e| / 3 ,
```

with the vertices of e ordered around the element and (x, y) in metres on a local
flat metric (`lon · cos(lat_ref) · METERS_PER_DEG_LAT`, `lat · METERS_PER_DEG_LAT`).
Differences from the model: the surface cluster area replaces `area(nz, n)`; the
z-level thickness Δz replaces the layer thickness `h_k`; the metric is flat, not
spherical. Accuracy degrades near steep slopes, in cavities and over large domains
(more than about 10°); for a regional extract of a few degrees and monthly-mean
fields it is adequate.

**Tier 2: depth-dependent areas and level masks.** This is what the CLI does by
default (`use_nod_area: true`): when it reads the mesh diagnostic directly it uses `nod_area(nz, nod2)` as the denominator and
the per-element active-level range (`ulevels(e)` … `nlevels(e) − 1`) as a mask, so
an element contributes only on the levels where it is wet; outside that range its
`u_e`, `v_e` are taken as zero. This gives the correct bottom mask on slopes. The
numerator is still the shape-function estimate: without the edge cross-vectors the
edge-flux form of the model is not reproduced, and that remains the main
approximation.

## 5. Output

`reconstruct_w(u, v, eta_prev, eta_next, dt_real, mesh, ...)` returns `w` on the
interfaces, `(K, N)`, node-based, positive-down — the layout of `VelocityField.w`.
The CLI writes it to `output.dir` / `output.filename` (default
`./w.reconstructed.nc`), with the interfaces on the `nz` dimension and the
velocity file's own time coordinate (values, units and calendar of the
processed records) — the layout `data.ncinput_w` expects in a run
configuration. The reader attaches the W file by its time stamps and refuses
one on another record axis; a `time_range` sub-run attaches to its full source
file, and a run that reaches a record outside the reconstructed range is
refused at that record (a step also reads the next record, PCHIP the next
two — reconstruct a margin). The tool refuses separate `v`, `ssh` or ALE-`w` inputs
that are not on the `u` file's records (it pairs them by record position). (A W
carrying the mid-layer dimension `nz1` is rejected by the FESOM2 reader. Files
written before 2026-10-04 carry relative seconds instead of the source's stamps
and must be regenerated.) The tests (`tests/test_w_reconstruct.py`)
pin the sign convention (rising ssh → negative surface w), the bottom closure,
Tier 1 on synthetic meshes and the Variant A/B comparison diagnostics. Tier 2's
depth-dependent denominator and active-level mask are so far exercised only with
unit areas.
