# KELP Visualization Module

A publication-ready visualization and analysis module for the `kelp` particle tracker. This module provides a set of tools to load tracking datasets along with their computational mesh and render trajectories and statistics in both 2D and 3D.

## Features

- **2D Cartopy Mapping (`plot_2d.py`)**: Geographically accurate rendering of unstructured meshes and particle paths.
- **3D Renderings (`plot_3d.py`)**: Depth-aware visualization of the mesh bathymetry and vertical trajectory movement (using PyVista or Matplotlib).
- **Statistical Gridding (`stats.py`)**: Fast 2D histogram tools to compute presence probabilities and beaching ratios.
- **Animations (`animation.py`)**: Export tracking iterations out to `.mp4` or `.gif`.

## Dependencies

The visualization module relies on several optional dependencies:
```bash
pip install matplotlib cartopy imageio pyvista
```
*(These are already included automatically if you use the main repository's updated `requirements.txt` or `environment.yml`)*

## Quick Start Example

A complete end-to-end example combining 2D plots, 3D plots, and statistics:

```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from kelp.readers import get_reader
from kelp.writers import load_trajectories
from kelp.visualization import TrajectoryDataset, plot_mesh, plot_trajectories

# 1. Load the mesh
reader = get_reader("fesomc")
# In standard scripts this is usually wrapped in a SimpleNamespace or config dict
from types import SimpleNamespace
mesh = reader.load_mesh(SimpleNamespace(meshpath="mesh.nc"))

# 2. Load trajectories output by kelp
data = load_trajectories("trajectories.nc")

# 3. Create a unified dataset wrapper
ds = TrajectoryDataset(mesh, data)

# 4. Plot 2D trajectories colored by depth
fig = plt.figure(figsize=(10, 8))
proj = ccrs.PlateCarree()
ax = fig.add_subplot(1, 1, 1, projection=proj)

plot_mesh(ds.mesh, ax=ax, projection=proj, alpha=0.3)
plot_trajectories(ds, ax=ax, projection=proj, color_by="depth")
plt.show()
```

## Creating Statistics

```python
from kelp.visualization.stats import compute_presence_probability, compute_status_ratios
import numpy as np

# Compute a uniform grid spanning the bounding box
min_x, max_x, min_y, max_y = ds.bounding_box
lon_bins = np.linspace(min_x, max_x, 50)
lat_bins = np.linspace(min_y, max_y, 50)

# Compute chance of finding a particle in each bin
probability = compute_presence_probability(ds, lon_bins, lat_bins)

# See what fraction of particles ended up beached vs stayed alive
ratios = compute_status_ratios(ds)
print("ALIVE:", ratios[1])
print("BEACHED:", ratios[2])
```

## Further functions

- `plot_field(mesh, values, layout='auto', ...)` — a node or element scalar drawn as
  coloured mesh faces (Gouraud on all-triangle node fields, flat faces otherwise;
  dateline-straddling faces masked; rotated meshes drawn geographic).
- `plot_mesh_nodes(...)` — the mesh nodes as points.
- `plot_trajectories(ds, color_by=..., cmap=..., projection=...)` — 2-D paths coloured
  by absolute depth or time with a linear norm (no pre-projection of the data).
- `plot_trajectories_3d(ds, plotter)` — 3-D paths in one fixed colour (PyVista when
  available, else a Matplotlib 3-D axes).
- `plot_diagnostics(ds, dt=...)` — per-particle panels of lon(t), lat(t), speed(t).
- `compute_presence_density_mesh(ds, mesh, normalize='probability')` — particle
  presence per mesh ELEMENT (`count`, `probability` or `per_area`), the unstructured
  twin of `compute_presence_probability`.

## Creating Animations

```python
from kelp.visualization.animation import TrajectoryAnimator

animator = TrajectoryAnimator(ds, tail_length=20)
animator.save("tracking.mp4", fps=10)
```
