"""
Core data structures for the visualization module.
Provides a `TrajectoryDataset` class that encapsulates outputs from the tracking run.
"""
from typing import Dict, Optional, Tuple, Union
import numpy as np

from kelp.core.mesh import Mesh


def display_coords(mesh: Mesh) -> Tuple[np.ndarray, np.ndarray]:
    """Mesh node coordinates in the GEOGRAPHIC frame, for display.

    Tracking may run in a rotated computational frame (``mesh.rotation_angles``
    set, ``mesh.nodes_lon/lat`` stored rotated), but all trajectory OUTPUT —
    and hence every plot overlaid on it — is geographic.  This helper returns
    the node coordinates un-rotated to geographic so a rotated mesh draws in
    the same frame as the trajectories.  For a geographic mesh
    (``rotation_angles is None``) it returns the stored arrays unchanged.

    Parameters
    ----------
    mesh : Mesh
        The mesh whose node coordinates to display.

    Returns
    -------
    lon, lat : np.ndarray
        Geographic-frame longitudes and latitudes (degrees).
    """
    if getattr(mesh, "rotation_angles", None) is None:
        return mesh.nodes_lon, mesh.nodes_lat
    from kelp.core.rotation import euler_matrix, unrotate_lonlat
    R = euler_matrix(*mesh.rotation_angles)
    return unrotate_lonlat(mesh.nodes_lon, mesh.nodes_lat, R)


class TrajectoryDataset:
    """
    A wrapper around trajectory outputs and the underlying computational mesh.
    
    This class is designed to handle `(P, T)` trajectory arrays (where P is the number
    of particles and T is the number of time steps) along with a `(T,)` time array.
    """
    def __init__(self, mesh: Mesh, data: Dict[str, np.ndarray]):
        """
        Initialize the TrajectoryDataset.
        
        Parameters
        ----------
        mesh : Mesh
            The unstructured grid used for tracking.
        data : dict
            A dictionary containing:
            - "lon": ndarray of shape (P, T), longitudes in degrees
            - "lat": ndarray of shape (P, T), latitudes in degrees
            - "z": ndarray of shape (P, T), depths (positive-down) in meters
            - "status": ndarray of shape (P, T), integer status codes
            - "time": ndarray of shape (T,), timestamps in seconds or similar
            - "release_lon", "release_lat" (optional): ndarray of shape (P,),
              the release positions (output format 2)
        """
        self.mesh = mesh
        
        # Extract fields
        self.lon = data.get("lon")
        self.lat = data.get("lat")
        self.z = data.get("z", np.zeros_like(self.lon) if self.lon is not None else None)
        self.status = data.get("status")
        self.time = data.get("time")
        self.release_lon = data.get("release_lon")
        self.release_lat = data.get("release_lat")
        
        # Validate shapes
        if self.lon is not None and self.lat is not None:
            if self.lon.shape != self.lat.shape:
                raise ValueError("lon and lat arrays must have the same shape.")
                
            # Check for (T, P) orientation from .to_arrays() and transpose to (P, T)
            if self.time is not None and self.lon.ndim == 2 and self.lon.shape[0] == len(self.time) and self.lon.shape[0] != self.lon.shape[1]:
                self.lon = self.lon.T
                self.lat = self.lat.T
                if self.z is not None: self.z = self.z.T
                if self.status is not None: self.status = self.status.T
                
            self.n_particles, self.n_steps = self.lon.shape
            
            if self.z is not None and self.z.shape != self.lon.shape:
                raise ValueError(f"z shape {self.z.shape} does not match lon shape {self.lon.shape}")
            if self.status is not None and self.status.shape != self.lon.shape:
                raise ValueError(f"status shape {self.status.shape} does not match lon shape {self.lon.shape}")
            if self.time is not None and len(self.time) != self.n_steps:
                raise ValueError(f"time length {len(self.time)} does not match n_steps {self.n_steps}")
        else:
            self.n_particles = 0
            self.n_steps = 0
            
    def get_step(self, step_idx: int) -> Dict[str, np.ndarray]:
        """
        Extract the particle state vectors for a specific timestep.
        
        Parameters
        ----------
        step_idx : int
            The timestep index (0 to T-1).
            
        Returns
        -------
        dict
            Dictionary containing 1D arrays ("lon", "lat", "z", "status", "time")
            representing the particle states at `step_idx`.
        """
        if step_idx < -self.n_steps or step_idx >= self.n_steps:
            raise IndexError(f"step_idx {step_idx} out of bounds for {self.n_steps} steps.")
            
        return {
            "lon": self.lon[:, step_idx] if self.lon is not None else None,
            "lat": self.lat[:, step_idx] if self.lat is not None else None,
            "z": self.z[:, step_idx] if self.z is not None else None,
            "status": self.status[:, step_idx] if self.status is not None else None,
            "time": self.time[step_idx] if self.time is not None else None
        }
        
    def filter_status(self, status_code: int) -> 'TrajectoryDataset':
        """
        Return a new dataset containing ONLY particles that have the specified
        status code at any point in their trajectory, OR just slice by those
        particles? Since status changes over time, this method filters the
        particle dimension based on the final status.
        
        Parameters
        ----------
        status_code : int
            The particle status to filter by (e.g., 1 for ALIVE, 2 for BEACHED, 3 for OUTSIDE).
            
        Returns
        -------
        TrajectoryDataset
            A dataset retaining only particles ending with the given status code.
        """
        if self.status is None:
            return self
            
        # Check final status of each particle
        final_status = self.status[:, -1]
        keep_mask = (final_status == status_code)
        
        new_data = {
            "time": self.time,
            "lon": self.lon[keep_mask, :],
            "lat": self.lat[keep_mask, :],
            "z": self.z[keep_mask, :],
            "status": self.status[keep_mask, :]
        }
        for name in ("release_lon", "release_lat"):
            if getattr(self, name) is not None:
                new_data[name] = getattr(self, name)[keep_mask]
        return TrajectoryDataset(self.mesh, new_data)
        
    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        """
        Get the bounding box of all trajectories safely.
        
        Returns
        -------
        tuple
            (min_lon, max_lon, min_lat, max_lat)
        """
        if self.lon is None or self.lat is None or self.n_particles == 0:
            return (-180.0, 180.0, -90.0, 90.0)
            
        return (
            float(np.nanmin(self.lon)), float(np.nanmax(self.lon)),
            float(np.nanmin(self.lat)), float(np.nanmax(self.lat))
        )
