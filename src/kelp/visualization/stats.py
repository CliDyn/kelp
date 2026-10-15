"""
Statistical operations on trajectory data.
Computes probability density on uniform grids and on the mesh faces.
"""
from typing import Tuple, Dict
import numpy as np

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

from .core import TrajectoryDataset
from kelp.core.mesh import (Mesh, build_search, build_elem_areas,
                                find_nearest_nodes,
                                find_containing_elements)
from kelp.core.particles import ParticleStatus


def compute_presence_probability(
    ds: TrajectoryDataset,
    lon_bins: np.ndarray,
    lat_bins: np.ndarray
) -> np.ndarray:
    """
    Computes the probability of finding a particle in a 2D uniform discrete grid.
    
    This essentially maps scattered trajectories to a dense Cartesian array.
    
    Parameters
    ----------
    ds : TrajectoryDataset
        The tracked trajectory data.
    lon_bins : np.ndarray
        1D array of longitude bin edges.
    lat_bins : np.ndarray
        1D array of latitude bin edges.
        
    Returns
    -------
    np.ndarray
        2D array of shape (len(lon_bins)-1, len(lat_bins)-1) representing probability density.
    """
    if ds.n_particles == 0 or ds.lon is None:
        return np.zeros((len(lon_bins) - 1, len(lat_bins) - 1))
        
    # Flatten the (P, T) arrays to (P*T)
    lons = ds.lon.flatten()
    lats = ds.lat.flatten()
    
    # Filter NaNs if any (out-of-domain might be stored as nan)
    valid = ~(np.isnan(lons) | np.isnan(lats))
    
    # Exclude dead/beached particles to avoid heavy histogram biases
    if ds.status is not None:
        valid &= (ds.status.flatten() == ParticleStatus.ALIVE)
        
    # We use numpy histogram2d
    # Note: numpy histogram2d returns (H, xedges, yedges)
    H, _, _ = np.histogram2d(
        lons[valid], 
        lats[valid], 
        bins=[lon_bins, lat_bins]
    )
    
    # Convert density counts to probability per cell
    total_valid = np.sum(valid)
    if total_valid > 0:
        prob = H / total_valid
    else:
        prob = H
        
    return prob


def compute_presence_density_mesh(
    ds: TrajectoryDataset,
    mesh: Mesh,
    normalize: str = "probability",
    alive_only: bool = True,
    return_stats: bool = False,
):
    """Particle presence per mesh ELEMENT — the unstructured twin of
    :func:`compute_presence_probability`.

    Every recorded ``(lon, lat)`` sample is mapped onto its containing
    element with the real containment machinery (nearest node → local
    element scan), so the statistic lives on the mesh faces themselves
    instead of a uniform overlay grid.  Plot it with
    :func:`~kelp.visualization.plot_2d.plot_field`.

    Trajectory output is always GEOGRAPHIC; for a rotated-frame mesh
    (``mesh.rotation_angles`` set) the samples are rotated into the mesh
    frame before containment.

    Parameters
    ----------
    ds : TrajectoryDataset
        The tracked trajectory data (``(P, T)`` arrays).
    mesh : Mesh
        The mesh whose elements to bin into (its search structures are
        built on demand).
    normalize : str
        ``'count'`` — raw sample counts per element;
        ``'probability'`` — counts / total number of VALID samples
        (finite, and ALIVE when ``alive_only``), mirroring the
        uniform-grid twin (off-mesh samples stay in the denominator);
        ``'per_area'`` — probability divided by the element area in m²
        (`build_elem_areas`) → a surface probability density.
    alive_only : bool
        Count only samples whose status is ALIVE (default, like the
        uniform-grid twin).  Silently a no-op when the dataset carries
        no status array.  ``False`` counts every finite sample.
    return_stats : bool
        Also return ``{"total_valid", "in_mesh"}`` sample counts, so a
        large off-mesh fraction (coastal deaths, wrong mesh) is
        distinguishable from "particles left the domain".

    Returns
    -------
    np.ndarray
        ``(E,)`` float64 — the per-element statistic.  With
        ``return_stats=True``: ``(array, stats_dict)``.
    """
    if normalize not in ("count", "probability", "per_area"):
        raise ValueError(
            f"normalize must be 'count'|'probability'|'per_area', "
            f"got {normalize!r}"
        )
    n_elems = mesh.n_elements
    out = np.zeros(n_elems, dtype=np.float64)
    empty_stats = {"total_valid": 0, "in_mesh": 0}
    if ds.lon is None or ds.n_particles == 0 or ds.lon.size == 0:
        return (out, empty_stats) if return_stats else out

    lons = np.asarray(ds.lon, dtype=np.float64).ravel()
    lats = np.asarray(ds.lat, dtype=np.float64).ravel()
    valid = np.isfinite(lons) & np.isfinite(lats)
    if alive_only and ds.status is not None:
        valid &= (np.asarray(ds.status).ravel() == ParticleStatus.ALIVE)
    total_valid = int(valid.sum())
    if total_valid == 0:
        return (out, empty_stats) if return_stats else out
    lons = lons[valid]
    lats = lats[valid]

    # Trajectory output is geographic; containment runs in the mesh's
    # own (possibly rotated) frame.
    if mesh.rotation_angles is not None:
        from kelp.core.rotation import euler_matrix, rotate_lonlat
        R = euler_matrix(*mesh.rotation_angles)
        lons, lats = rotate_lonlat(lons, lats, R)

    if mesh._gh_offsets is None and mesh._node_tree is None:
        build_search(mesh)

    nn = find_nearest_nodes(mesh, lons, lats)
    elem_ids, in_domain = find_containing_elements(mesh, lons, lats, nn)
    hits = elem_ids[in_domain]
    counts = np.bincount(hits, minlength=n_elems).astype(np.float64)
    stats = {"total_valid": total_valid, "in_mesh": int(in_domain.sum())}

    if normalize == "count":
        return (counts, stats) if return_stats else counts
    prob = counts / total_valid
    if normalize == "probability":
        return (prob, stats) if return_stats else prob
    areas = build_elem_areas(mesh)
    # Degenerate (zero-area) elements: 0 density, not NaN/warning spam.
    dens = np.divide(prob, areas, out=np.zeros_like(prob),
                     where=areas > 0)
    return (dens, stats) if return_stats else dens


def compute_status_ratios(ds: TrajectoryDataset) -> Dict[int, float]:
    """
    Compute the fraction of particles ending in each status state.
    
    Parameters
    ----------
    ds : TrajectoryDataset
        The trajectory data.
        
    Returns
    -------
    dict
        Mapping of status_code to fraction (0.0 to 1.0).
    """
    if ds.status is None or ds.n_particles == 0:
        return {}
        
    final_status = ds.status[:, -1]
    unique, counts = np.unique(final_status, return_counts=True)
    fractions = counts / ds.n_particles
    
    return dict(zip(unique, fractions))
