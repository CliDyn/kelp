# Developer notes — index

<!-- module index: written by hand above this line -->

## Dependency graph (imports)

Generated from the import statements (2026-10-04); "lazy" = imported inside a
function. Third-party: numpy everywhere, numba optional in the kernels, xarray
in the readers, netCDF4/zarr in the writer.

```
core/config.py            ← standalone (yaml)
core/geometry.py          ← standalone
core/io.py                ← standalone (VelocityField, LazyLerpField)
core/rotation.py          ← standalone
core/shared_fields.py     ← standalone (mmap/flock snapshot cache)
core/particles.py         ← standalone (lazy: particles_gen)
core/mesh.py              ← geometry
core/particles_gen.py     ← geometry, mesh, particles
core/release.py           ← particles
writers.py                ← particles (netCDF4 / zarr)
w_reconstruct.py          ← geometry
numerics/temporal_reconstruct.py ← io
numerics/interpolate.py   ← geometry, io, mesh, particles, temporal_reconstruct
numerics/vertical.py      ← mesh, temporal_reconstruct
numerics/vertical_sigma.py ← mesh
numerics/diffusion.py     ← geometry, mesh, interpolate
numerics/beaching.py      ← particles, diffusion
numerics/sampling.py      ← mesh, vertical, vertical_sigma
numerics/integrators.py   ← geometry, mesh, particles, diffusion, interpolate, vertical
                            (lazy: io, temporal_reconstruct, vertical_sigma)
readers/__init__.py       ← standalone (registry + shared helpers)
readers/fesom2.py, fesomc.py ← config, io, mesh, readers helpers (fesom2 lazy: rotation)
readers/healpix.py        ← geometry, mesh, readers helpers (lazy: io, rotation, Kh name helpers)
readers/structured.py     ← mesh
readers/nemo.py           ← io, mesh, readers helpers, structured
tracking.py               ← config, geometry, io, mesh, particles, release, diffusion,
                            integrators, readers (lazy: writers, shared_fields, beaching,
                            sampling, temporal_reconstruct, rotation, readers.fesom2, __version__)
visualization/*           ← core.geometry, core.mesh, core.particles (lazy: rotation)
cli/main.py               ← __version__ (lazy per subcommand: config, mesh, readers, tracking)
```

`interpolate.py` re-exports `VelocityField` from `core.io` (some tests import it
from there); the readers import it from `core.io`.
