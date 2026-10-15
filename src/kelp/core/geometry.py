"""Spherical geometry utilities and physical constants."""
from __future__ import annotations

import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Physical constants
#
# NOTE: METERS_PER_DEG_LAT (SLM) and DEG2RAD are duplicated as literals
# inside @njit kernels in interpolate.py, vertical.py, and integrators.py
# because Numba cannot reliably capture module-level imports.  If you change
# a value here, grep for the literal and update the kernels too.
# ---------------------------------------------------------------------------
# NOTE: As of the Parcels comparison work, METERS_PER_DEG_LAT was changed from
# the Fortran value 111300 to 111120 (= 1852 m/nm × 60 nm/deg, nautical-mile
# convention used by Parcels and most oceanographic tooling). This breaks
# bit-identical parity with the original Fortran reference tracker
# — tests that rely on that parity are marked appropriately.
#
# IMPORTANT for Numba @njit(cache=True): if you change this value, clear the
# Numba cache so that kernels that capture this constant as a global are
# rebuilt:
#     find . -name "__pycache__" -type d ! -path "*/.venv/*" -exec rm -rf {} +
# Kernels in interpolate.py and integrators.py capture this via import;
# kernels in vertical.py inline the literal value (grep for SLM = 111120).
METERS_PER_DEG_LAT = 111120.0    # m/deg = 1852 × 60 (nautical mile convention)
R_EARTH = 6_400_000.0            # legacy Fortran value; do not use in new code
R_EARTH_STANDARD = 6_371_000.0   # standard mean Earth radius — use this in new code
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def wrap_lon(lon: np.ndarray) -> np.ndarray:
    """Wrap longitude into [-180, 180).

    Parameters
    ----------
    lon : array_like
        Longitude in degrees.

    Returns
    -------
    np.ndarray
        Wrapped longitude.
    """
    return ((np.asarray(lon, dtype=np.float64) + 180.0) % 360.0) - 180.0


def clamp_lat(lat: np.ndarray) -> np.ndarray:
    """Clamp latitude to [-90, 90].

    Parameters
    ----------
    lat : array_like
        Latitude in degrees.

    Returns
    -------
    np.ndarray
        Clamped latitude.
    """
    return np.clip(np.asarray(lat, dtype=np.float64), -90.0, 90.0)


def lonlat_to_xyz(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """Convert lon/lat (degrees) to unit-sphere Cartesian coordinates.

    Parameters
    ----------
    lon_deg, lat_deg : array_like
        Longitude and latitude in degrees.

    Returns
    -------
    np.ndarray
        Shape (..., 3) Cartesian coordinates.
    """
    lon = np.asarray(lon_deg, dtype=np.float64) * DEG2RAD
    lat = np.asarray(lat_deg, dtype=np.float64) * DEG2RAD
    cos_lat = np.cos(lat)
    return np.stack([cos_lat * np.cos(lon),
                     cos_lat * np.sin(lon),
                     np.sin(lat)], axis=-1)


def xyz_to_lonlat(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert unit-sphere Cartesian to lon/lat (degrees).

    Parameters
    ----------
    xyz : np.ndarray
        Shape (..., 3) Cartesian coordinates.

    Returns
    -------
    lon_deg, lat_deg : np.ndarray
        Longitude and latitude in degrees.
    """
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    lon = np.arctan2(y, x) * RAD2DEG
    lat = np.arcsin(np.clip(z, -1.0, 1.0)) * RAD2DEG
    return lon, lat


# ---------------------------------------------------------------------------
# Metric / velocity conversions
# ---------------------------------------------------------------------------

def metric_z0(lat_deg: np.ndarray) -> np.ndarray:
    """Compute metric z0 = cos(lat)*METERS_PER_DEG_LAT — meters per degree longitude.

    This is the metric factor used throughout to convert
    between longitude degrees and meters.

    Parameters
    ----------
    lat_deg : array_like
        Latitude in degrees.

    Returns
    -------
    np.ndarray
        z0 in meters.
    """
    return np.cos(np.asarray(lat_deg, dtype=np.float64) * DEG2RAD) * METERS_PER_DEG_LAT


def angular_rates(u_ms: np.ndarray, v_ms: np.ndarray,
                  lat_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert velocity (m/s) to angular rates (deg/s).

    Uses metric: dlon/dt = U / z0,  dlat/dt = V / METERS_PER_DEG_LAT
    where z0 = cos(lat)*METERS_PER_DEG_LAT.

    Parameters
    ----------
    u_ms : array_like
        Eastward velocity in m/s.
    v_ms : array_like
        Northward velocity in m/s.
    lat_deg : array_like
        Latitude in degrees.

    Returns
    -------
    dlon_dt, dlat_dt : np.ndarray
        Angular rates in degrees per second.
    """
    z0 = metric_z0(lat_deg)
    # Avoid division by zero near poles
    z0_safe = np.where(np.abs(z0) < 1e-10, 1e-10, z0)
    dlon_dt = np.asarray(u_ms, dtype=np.float64) / z0_safe
    dlat_dt = np.asarray(v_ms, dtype=np.float64) / METERS_PER_DEG_LAT
    return dlon_dt, dlat_dt


def great_circle_distance(lon1: np.ndarray, lat1: np.ndarray,
                          lon2: np.ndarray, lat2: np.ndarray,
                          radius: float = R_EARTH_STANDARD) -> np.ndarray:
    """Compute great-circle distance between two points.

    Parameters
    ----------
    lon1, lat1 : array_like
        First point lon/lat in degrees.
    lon2, lat2 : array_like
        Second point lon/lat in degrees.
    radius : float
        Earth radius in meters.

    Returns
    -------
    np.ndarray
        Distance in meters.
    """
    xyz1 = lonlat_to_xyz(lon1, lat1)
    xyz2 = lonlat_to_xyz(lon2, lat2)
    cos_d = np.clip(np.sum(xyz1 * xyz2, axis=-1), -1.0, 1.0)
    return np.arccos(cos_d) * radius


# ---------------------------------------------------------------------------
# Dateline-safe longitude differences
# ---------------------------------------------------------------------------

def delta_lon(lon: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Compute dateline-safe longitude difference: lon - ref.

    Parameters
    ----------
    lon, ref : array_like
        Longitudes in degrees.

    Returns
    -------
    np.ndarray
        Difference in [-180, 180).
    """
    d = np.asarray(lon, dtype=np.float64) - np.asarray(ref, dtype=np.float64)
    return ((d + 180.0) % 360.0) - 180.0


__all__ = [
    "METERS_PER_DEG_LAT", "R_EARTH", "R_EARTH_STANDARD", "DEG2RAD", "RAD2DEG",
    "wrap_lon", "clamp_lat",
    "lonlat_to_xyz", "xyz_to_lonlat",
    "metric_z0", "angular_rates",
    "great_circle_distance", "delta_lon",
]
