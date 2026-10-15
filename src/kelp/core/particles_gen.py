"""Particle seeding utilities.

Generate initial particle distributions on an unstructured mesh.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .geometry import DEG2RAD, METERS_PER_DEG_LAT
from .mesh import Mesh, build_search, find_nearest_nodes, find_containing_elements
from .particles import ParticleState, make_state


def generate_grid(mesh: Mesh, nx: int = 5, ny: int = 5,
                  margin: float = 0.0) -> ParticleState:
    """Generate particles on a regular lon/lat grid, keeping only in-mesh points.

    Parameters
    ----------
    mesh : Mesh
        The mesh to seed particles in.  Must have ``build_search`` called.
    nx, ny : int
        Number of grid points in longitude and latitude directions.
    margin : float
        Fraction of the mesh extent to shrink inward (0.0 = full extent,
        0.05 = 5% margin on each side).

    Returns
    -------
    ParticleState
        Only particles that lie inside a mesh element.
    """
    lon_min, lon_max = mesh.nodes_lon.min(), mesh.nodes_lon.max()
    lat_min, lat_max = mesh.nodes_lat.min(), mesh.nodes_lat.max()

    dx = (lon_max - lon_min) * margin
    dy = (lat_max - lat_min) * margin
    lons = np.linspace(lon_min + dx, lon_max - dx, nx)
    lats = np.linspace(lat_min + dy, lat_max - dy, ny)
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    grid_lon = grid_lon.ravel()
    grid_lat = grid_lat.ravel()

    return _filter_to_mesh(mesh, grid_lon, grid_lat)


def generate_around_point(mesh: Mesh, lon0: float, lat0: float,
                          n: int = 20, radius_km: float = 5.0,
                          seed: Optional[int] = None) -> ParticleState:
    """Generate random particles around a point, keeping only in-mesh points.

    Particles are uniformly distributed in a circle of given radius.

    Parameters
    ----------
    mesh : Mesh
        The mesh to seed particles in.  Must have ``build_search`` called.
    lon0, lat0 : float
        Centre point in degrees.
    n : int
        Number of candidate particles to generate (some may be filtered out
        if they fall outside the mesh).
    radius_km : float
        Radius of the circle in kilometres.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    ParticleState
        Only particles that lie inside a mesh element.

    Notes
    -----
    On a rotated-frame mesh (``mesh.rotation_angles`` set) ``lon0``/``lat0``
    are interpreted in the MESH frame; rotate user/geographic coordinates
    first (the tracking layer does this for config-provided seeds).
    """
    rng = np.random.default_rng(seed)

    # uniform distribution in a circle: sqrt for uniform area coverage
    r = np.sqrt(rng.uniform(0, 1, size=n)) * radius_km * 1000.0  # metres
    theta = rng.uniform(0, 2 * np.pi, size=n)

    dx_m = r * np.cos(theta)
    dy_m = r * np.sin(theta)

    z0 = np.cos(lat0 * DEG2RAD) * METERS_PER_DEG_LAT
    dlons = dx_m / z0
    dlats = dy_m / METERS_PER_DEG_LAT

    lons = lon0 + dlons
    lats = lat0 + dlats

    return _filter_to_mesh(mesh, lons, lats)


def _filter_to_mesh(mesh: Mesh, lons: np.ndarray,
                    lats: np.ndarray) -> ParticleState:
    """Keep only particles that are inside a mesh element.

    Parameters
    ----------
    mesh : Mesh
        The search structure is built on first use; an existing grid hash
        (incl. a pinned or rule-chosen cell) is kept.
    lons, lats : np.ndarray
        Candidate particle positions.

    Returns
    -------
    ParticleState
    """
    # The KD-tree is built lazily (only for the batch heal), so testing it
    # rebuilt the whole grid hash on every call; test the hash itself.
    if mesh._gh_offsets is None:
        build_search(mesh)

    nn = find_nearest_nodes(mesh, lons, lats)
    _, in_domain = find_containing_elements(mesh, lons, lats, nn)
    mask = in_domain.astype(bool)

    if not mask.any():
        raise ValueError("No candidate particles fall inside the mesh domain.")

    return make_state(lons[mask], lats[mask])


__all__ = [
    "generate_grid",
    "generate_around_point",
]
