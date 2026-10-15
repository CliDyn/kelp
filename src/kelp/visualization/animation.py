"""
Animation module for Lagrangian trajectories.
"""
from typing import Optional
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
except ImportError:
    plt, FuncAnimation = None, None

try:
    import cartopy.crs as ccrs
except ImportError:
    ccrs = None

from .core import TrajectoryDataset
from .plot_2d import plot_mesh


class TrajectoryAnimator:
    """
    Creates and saves animations of particle trajectories over time.
    """
    
    def __init__(
        self, 
        ds: TrajectoryDataset, 
        figsize: tuple = (10, 8),
        step_stride: int = 1,
        tail_length: int = 5,
        projection: Optional['ccrs.Projection'] = None
    ):
        """
        Initialize the Animator.
        
        Parameters
        ----------
        ds : TrajectoryDataset
            Data to animate.
        figsize : tuple
            Figure size.
        step_stride : int
            Number of data steps to jump per animation frame (for speed).
        tail_length : int
            Number of previous positions to draw as a vanishing "tail".
        projection : ccrs.Projection
            Cartopy projection map.
        """
        if plt is None:
            raise ImportError("matplotlib and cartopy are required for animations.")
            
        self.ds = ds
        self.step_stride = step_stride
        self.tail_length = tail_length
        self.frames = list(range(0, ds.n_steps, step_stride))
        
        # Setup Figure
        self.fig = plt.figure(figsize=figsize)
        if projection is not None:
            self.ax = self.fig.add_subplot(1, 1, 1, projection=projection)
            self.transform = ccrs.PlateCarree()
        else:
            self.ax = self.fig.add_subplot(1, 1, 1)
            self.transform = None
            
        self._setup_background()
        
        # Scatter object for current positions
        self.scatter = None
        # Line objects for tail
        self.lines = [None] * ds.n_particles
        
    def _setup_background(self):
        """Draw the static mesh background and set limits."""
        plot_mesh(self.ds.mesh, ax=self.ax, alpha=0.3, projection=None)
        
        min_x, max_x, min_y, max_y = self.ds.bounding_box
        pad_x = max((max_x - min_x) * 0.1, 1.0)
        pad_y = max((max_y - min_y) * 0.1, 1.0)
        
        if hasattr(self.ax, 'set_extent') and ccrs is not None:
            self.ax.set_extent([min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y], crs=ccrs.PlateCarree())
        else:
            self.ax.set_xlim(min_x - pad_x, max_x + pad_x)
            self.ax.set_ylim(min_y - pad_y, max_y + pad_y)

    def _init_func(self):
        """Init function for FuncAnimation."""
        if self.scatter is None:
            self.scatter = self.ax.scatter(
                [], [], s=10, c='red', zorder=3,
                transform=self.transform
            )
            for p in range(self.ds.n_particles):
                self.lines[p], = self.ax.plot(
                    [], [], color='red', alpha=0.5, linewidth=1.0,
                    transform=self.transform, zorder=2
                )
                
        return [self.scatter] + self.lines

    def _update_frame(self, frame_idx: int):
        """Update visually for frame index."""
        t_current = frame_idx
        t_start = max(0, t_current - self.tail_length * self.step_stride)
        
        # Current points
        curr_lon = self.ds.lon[:, t_current]
        curr_lat = self.ds.lat[:, t_current]
        self.scatter.set_offsets(np.column_stack((curr_lon, curr_lat)))
        
        # Update strings/tails
        for p in range(self.ds.n_particles):
            self.lines[p].set_data(
                self.ds.lon[p, t_start:t_current+1],
                self.ds.lat[p, t_start:t_current+1]
            )
            
        self.ax.set_title(f"Time Step: {t_current}")
        return [self.scatter] + self.lines

    def animate(self):
        """Render to an internal FuncAnimation object."""
        self.anim = FuncAnimation(
            self.fig, self._update_frame, frames=self.frames,
            init_func=self._init_func, blit=True
        )

    def save(self, filename: str, fps: int = 15):
        """
        Save the animation to disk.
        
        Parameters
        ----------
        filename : str
            File to save to, e.g., 'trajectory.mp4' or 'trajectory.gif'
        fps : int
            Frames per second.
        """
        if not hasattr(self, 'anim'):
            self.animate()
            
        self.anim.save(filename, fps=fps)
        plt.close(self.fig)
