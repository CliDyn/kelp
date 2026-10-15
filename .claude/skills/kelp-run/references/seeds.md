# Seed shapes, sorting and delayed release

Read when the seeds are not a plain box or cloud at one depth, when the run is large or 3-D (sorting),
or when particles are released over time.

- A cloud around a point → the in-repo helper (geographic mesh; on a mesh loaded with
  `mesh.rotation` use the rotate-in / unrotate-out recipe of SKILL.md step 2 — the helper
  works in the mesh frame):
  ```python
  from kelp.core.particles_gen import generate_around_point
  st = generate_around_point(mesh, lon0, lat0, n=1000, radius_km=80, seed=42)
  st.save("seeds.dat")            # writes "lon lat" (+ depth if any non-zero)
  ```
- **At depth** (for 3D vertical-distribution runs) the helper seeds at the
  surface, so add a depth column yourself — the writer/loader treats column 3
  as depth in metres, positive-down:
  ```python
  import numpy as np
  lonlat = np.loadtxt("surface_cloud.dat")           # (N,2)
  depth  = np.random.default_rng(42).uniform(10, 1000, len(lonlat))
  np.savetxt("seeds.dat", np.column_stack([lonlat, depth]),
             fmt="%.8f %.8f %.3f")
  ```
- A regular grid over the mesh → `generate_grid(mesh, nx, ny, margin)` (mesh-frame extents and
  coordinates: unrotate before saving on a rotated mesh).

- **Sort seeds spatially** (`utils/sort_seeds.py`) before a big or 3-D run:
  +17 % on the fused 3-D step (shuffled seeds cost +6 %); sort FIRST, then
  shard.
- **Released over time (delayed/staggered)?** Two ways, which compose:
  - *Regular pulses* (same place, fixed cadence) → **config only, no seed
    work**: set `time.n_pulses` + `time.pulse_interval` (seconds). The seed
    set is re-released `n_pulses` times; total particles = seeds ×
    `n_pulses`. E.g. `n_pulses: 5, pulse_interval: 86400` = once a day for
    5 days.
  - *Individual birth times* → add a **4th seed column** `release_sec`
    (offset in seconds from the run start; column 3 is depth, so include a
    depth even if 0):
    ```python
    import numpy as np
    lonlat = np.loadtxt("cloud.dat")                  # (N,2)
    depth  = np.zeros(len(lonlat))                    # column 3 (metres), 0 = surface
    rel    = my_release_seconds_per_particle          # column 4: each particle's own offset (s)
    np.savetxt("seeds.dat", np.column_stack([lonlat, depth, rel]),
               fmt="%.8f %.8f %.3f %.1f")
    ```
    (For a regular cadence at the same place, prefer the `n_pulses` config
    keys above — no seed work. Use the column for irregular/individual times.)
  Frozen particles are written as NaN (status `not_released`) until they
  activate. **Size the window:** `n_steps · dt` must cover the LAST release
  time **plus** the drift you want to see after it — otherwise late pulses
  never activate (the run warns) or barely move.
