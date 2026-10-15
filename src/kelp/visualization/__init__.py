"""
kelp.visualization

A publication-ready visualization and analysis module for the kelp particle tracker.
"""
from .core import TrajectoryDataset
from .plot_2d import (plot_mesh, plot_mesh_nodes, plot_trajectories,
                      plot_diagnostics, plot_field)
from .plot_3d import plot_mesh_3d, plot_trajectories_3d
from .stats import (compute_presence_probability, compute_status_ratios,
                    compute_presence_density_mesh)
from .animation import TrajectoryAnimator

__all__ = [
    "TrajectoryDataset",
    "plot_mesh",
    "plot_mesh_nodes",
    "plot_trajectories",
    "plot_diagnostics",
    "plot_mesh_3d",
    "plot_trajectories_3d",
    "plot_field",
    "compute_presence_probability",
    "compute_status_ratios",
    "compute_presence_density_mesh",
    "TrajectoryAnimator"
]
