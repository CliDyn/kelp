# Beaching on a coastal mesh

Two setups: counting strandings ("how many wash up, and where") and the "revert and retry" tracker.

## Counting strandings

1. **Print what can fire on this mesh** before choosing criteria (`mesh` from the reader's `load_mesh`, `c` the loaded
   config, `lon, lat` the seeds as written):
   ```python
   from kelp.core.mesh import build_boundary_info, find_nearest_nodes, find_containing_elements
   build_boundary_info(mesh); rim = mesh._elem_is_coastal          # elements that own a boundary edge: coast AND open sea
   ce, ok = find_containing_elements(mesh, lon, lat, find_nearest_nodes(mesh, lon, lat))
   b = c.numerics.beaching; zone = b.zone_depth if b.zone_depth is not None else 2 * b.depth_min
   print(f"shallowest node {mesh.depth.min():.2f} m | on_failure zone_depth {zone:g} m | rim elements {int(rim.sum())} | "
         f"seeds outside the mesh {int((~ok).sum())} | seeds in rim elements {int(rim[ce[ok]].sum())} of {len(lon)}")
   ```

   | criterion | a particle becomes BEACHED when | trap |
   |---|---|---|
   | `on_failure` | the integrator kills it (its step leaves the water) and the node it stops at is shallower than `zone_depth` (default 2·`depth_min` = 0.6 m); a rim element counts too, but only when `coastal_element` or `stochastic` is also listed | alone with the default `zone_depth`, nothing beaches where the shallowest node is deeper than 0.6 m |
   | `coastal_element` | it is inside any rim element | the rim includes the open-sea boundary; a seed in a rim element beaches on the first step |
   | `depth` | its node is shallower than `depth_min` (0.3 m) | needs drying nodes (or `use_eta` with an ssh variable) |

   `failure_reason` does not separate the coast from the open sea: a step that leaves the mesh is `OUT_OF_BOUNDS` on
   either side.

2. **Default setup** — a particle strands where its step would leave the water, and stays there:
   ```yaml
   mode:
     track_failure_reason: true
   numerics:
     beaching:
       enabled: true
       criteria: [on_failure]
       zone_depth: 1000000000.0   # plain decimal: every integrator death is remapped, open-sea exits included (step 3 splits them)
       resuspension: never        # washed up stays washed up: the count at the end is every particle that reached the coast
   ```
   The same counts come without the beaching model: beaching off, every `OUTSIDE` particle classified by step 3. The plan
   says which of the two it uses.

3. **Coast or open sea, by location.** KELP's boundary is every edge owned by one element; the mesh carries no
   open-boundary flag, so the split is made in the analysis. The block draws the rim (`rim.png`); read off it the two
   corners of the outer loop where the open boundary starts and ends and one point on it, put them in `OPEN_ARC`, run
   again and look at `rim_check.png` (open sea red) before trusting the split. The plan names the three points:
   ```python
   import numpy as np
   from collections import Counter, defaultdict
   from scipy.spatial import cKDTree
   cnt = Counter()
   for ring in mesh.elem_nodes:
       r = [int(v) for v in ring if v >= 0]
       r = [v for i, v in enumerate(r) if i == 0 or v != r[i - 1]]
       if len(r) > 1 and r[-1] == r[0]:
           r = r[:-1]
       cnt.update(tuple(sorted(e)) for e in zip(r, r[1:] + r[:1]))
   bnd = [e for e, n in cnt.items() if n == 1]                     # boundary edges (coast and open sea)
   nbr = defaultdict(list)
   for a, b in bnd:
       nbr[a].append(b); nbr[b].append(a)
   loops, seen = [], set()
   for start in nbr:
       if start in seen:
           continue
       loop, prev, cur = [start], None, start
       seen.add(start)
       while True:
           nxt = [n for n in nbr[cur] if n != prev and (n not in seen or (n == start and len(loop) > 2))]
           if not nxt or nxt[0] == start:
               break
           prev, cur = cur, nxt[0]; loop.append(cur); seen.add(cur)
       loops.append(loop)
   outer = max(loops, key=len)                                      # the other loops are islands: coast
   import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
   fig, ax = plt.subplots(figsize=(7, 7))
   for a, b in bnd:
       ax.plot(mesh.nodes_lon[[a, b]], mesh.nodes_lat[[a, b]], "k-", lw=0.5)
   ax.grid(True); fig.savefig("rim.png", dpi=120)
   OPEN_ARC = [START_CORNER, POINT_ON_OPEN_BOUNDARY, END_CORNER]     # three (lon, lat) tuples read off rim.png of THIS mesh
   q = lambda lon, lat: np.c_[np.asarray(lon) * np.cos(np.deg2rad(lat)), lat]
   at = cKDTree(q(mesh.nodes_lon[outer], mesh.nodes_lat[outer]))
   ia, iv, ib = (int(at.query(q([p[0]], [p[1]]))[1][0]) for p in OPEN_ARC)
   m = len(outer); fwd = lambda i, j: (j - i) % m
   arc = [(ia + k) % m for k in range(fwd(ia, ib) + 1)] if fwd(ia, iv) <= fwd(ia, ib) else \
         [(ib + k) % m for k in range(fwd(ib, ia) + 1)]           # the way round that passes the middle point
   on_arc = {outer[i] for i in arc}
   is_open = np.array([a in on_arc and b in on_arc for a, b in bnd])
   mid_lon = np.array([(mesh.nodes_lon[a] + mesh.nodes_lon[b]) / 2 for a, b in bnd])
   mid_lat = np.array([(mesh.nodes_lat[a] + mesh.nodes_lat[b]) / 2 for a, b in bnd])
   edges = cKDTree(q(mid_lon, mid_lat))
   for (a, b), o in zip(bnd, is_open):
       ax.plot(mesh.nodes_lon[[a, b]], mesh.nodes_lat[[a, b]], "r-" if o else "k-", lw=0.8 if o else 0.5)
   fig.savefig("rim_check.png", dpi=120)
   print(f"boundary edges {len(bnd)} in {len(loops)} loops; open-sea edges {int(is_open.sum())} between {OPEN_ARC[0]} and {OPEN_ARC[2]}")
   ```

4. **The count**, all over the released number; the stop position of a stranded particle is its last record:
   ```python
   st = ds["status"].isel(time=-1).values                           # a format-1 store (no kelp_output_format attribute): obs=-1
   lon_end, lat_end = ds["lon"].isel(time=-1).values, ds["lat"].isel(time=-1).values
   stuck = (st == 2) | (st == 3)                                     # BEACHED; OUTSIDE when the beaching model is off
   k = edges.query(q(lon_end[stuck], lat_end[stuck]))[1]            # the nearest boundary edge of each stop
   coast = ~is_open[k]
   print(f"released {st.size:,} | washed up on the coast {int(coast.sum()):,} | left through the open sea {int((~coast).sum()):,} | "
         f"afloat {int((st == 1).sum()):,}")
   ```
   Where: group the coast stops by stretch (named segments, or distance along the coast) and map them.

When the owner names another definition (contact with a rim element = `coastal_element`; a distance to the coastline;
resuspension with the tide), use it; the printout of step 1 and the split of step 3 stay.

## The "revert and retry" setup

A Fortran-style tracker that reverts a particle leaving its element and retries
later is this block, copied literally (values are the gates, not examples):
```yaml
  beaching:
    enabled: true
    criteria: [on_failure]        # on_failure and NOTHING ELSE in this list — `depth` traps at the coast
    zone_depth: 1000000000.0      # plain decimal; gates the on_failure remap: catches EVERY failure, open-boundary exits included (intended)
    resuspension: auto
    depth_min: 0.000001           # release gate is depth_min + rewet_hysteresis ...
    rewet_hysteresis: -1000000.0  # ... = -1e6 here, so a depth-0 rim node still releases
    min_wet_steps: 1
```
- Why the gates: release needs `H_raw > depth_min + rewet_hysteresis`; with the
  defaults (0.3 + 0.1) a particle beached at a `depth == 0` node NEVER re-wets —
  a permanent trap, the opposite of a retry. `resuspension: stochastic` is
  water-gated by the same expression (and needs `t_resuspend > 0`), so it is not
  an escape hatch — it needs the same open gates.
- Count the rim and the seeds; paste into `validate.py`:
  ```python
  n_rim = int((mesh.depth <= 0).sum())                       # the trap population
  nn = find_nearest_nodes(mesh, lons, lats)
  n_on_rim = int((mesh.depth[nn] <= 0).sum())                # seeds beached from step 1 until rewetted
  assert c.numerics.beaching.depth_min + c.numerics.beaching.rewet_hysteresis < 0
  ```
- `zone_depth` written as `1e9`/`1.0e9` is a STRING to PyYAML and the kernel
  crashes late with `ufunc 'less' … StrDType`; assert `isinstance(..., float)`.
- Never add `depth` to the criteria for this setup (`depth_min` would beach
  every particle over shallower water). Verified on a micro run: a drifter at
  the coast keeps ALIVE with a frozen position and moves again when the flow
  turns; the hook's cost was negligible (re-measure on your run).
