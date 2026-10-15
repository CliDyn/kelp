"""Rotated-sphere coordinate frame (FESOM convention).

The Euler matrix below reproduces FESOM2's own rotated frame: with the
standard angles (50, 15, -90) it reconstructs the model's ``elem_area``
to machine precision (validated 2026-06-06 against fesom.mesh.diag.nc).
Convention: ``v_rot = R @ v_geo`` on 3D unit vectors; inverse via ``R.T``.

All functions are pure vectorised numpy — they run once per mesh load /
velocity snapshot, never inside particle loops, so no numba is needed.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

#: Standard FESOM2 Euler angles (degrees): the computational pole sits on
#: Greenland, far from any ocean point.
FESOM2_ANGLES: Tuple[float, float, float] = (50.0, 15.0, -90.0)


def euler_matrix(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """FESOM rotation matrix for Euler angles in degrees.

    Returns the (3, 3) matrix R with ``v_rot = R @ v_geo``.
    """
    al, be, ga = np.deg2rad([alpha, beta, gamma])
    ca, sa = np.cos(al), np.sin(al)
    cb, sb = np.cos(be), np.sin(be)
    cg, sg = np.cos(ga), np.sin(ga)
    return np.array([
        [cg * ca - sg * cb * sa,  cg * sa + sg * cb * ca, sg * sb],
        [-sg * ca - cg * cb * sa, -sg * sa + cg * cb * ca, cg * sb],
        [sb * sa,                 -sb * ca,                cb],
    ])


def _to_xyz(lon_deg, lat_deg):
    lo = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    la = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    cl = np.cos(la)
    return np.stack([cl * np.cos(lo), cl * np.sin(lo), np.sin(la)], axis=-1)


def _to_lonlat(xyz):
    lon = np.rad2deg(np.arctan2(xyz[..., 1], xyz[..., 0]))
    lat = np.rad2deg(np.arcsin(np.clip(xyz[..., 2], -1.0, 1.0)))
    lon = ((lon + 180.0) % 360.0) - 180.0      # wrap to [-180, 180)
    return lon, lat


def rotate_lonlat(lon_deg: np.ndarray, lat_deg: np.ndarray,
                  R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Geographic lon/lat (degrees) -> rotated-frame lon/lat (degrees).

    Parameters
    ----------
    lon_deg, lat_deg : array_like
        Geographic longitude and latitude in degrees.
    R : np.ndarray
        Shape (3, 3) Euler rotation matrix, ``v_rot = R @ v_geo`` (see
        :func:`euler_matrix`).

    Returns
    -------
    lon_rot, lat_rot : np.ndarray
        Rotated-frame longitude (wrapped to [-180, 180)) and latitude in
        degrees.
    """
    return _to_lonlat(_to_xyz(lon_deg, lat_deg) @ R.T)


def unrotate_lonlat(lon_deg: np.ndarray, lat_deg: np.ndarray,
                    R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rotated-frame lon/lat (degrees) -> geographic lon/lat (degrees).

    Parameters
    ----------
    lon_deg, lat_deg : array_like
        Rotated-frame longitude and latitude in degrees.
    R : np.ndarray
        Shape (3, 3) Euler rotation matrix, ``v_rot = R @ v_geo`` (see
        :func:`euler_matrix`); the inverse rotation ``R.T`` is applied
        internally.

    Returns
    -------
    lon_geo, lat_geo : np.ndarray
        Geographic longitude (wrapped to [-180, 180)) and latitude in
        degrees.
    """
    return _to_lonlat(_to_xyz(lon_deg, lat_deg) @ R)


def _east_north(lon_deg, lat_deg):
    """Local unit basis vectors (east, north) as 3D vectors.

    At |lat| = 90 the basis is the limit at the stored longitude
    (deterministic; the spec's exact-pole convention).
    """
    lo = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    la = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    e = np.stack([-np.sin(lo), np.cos(lo), np.zeros_like(lo)], axis=-1)
    n = np.stack([-np.sin(la) * np.cos(lo),
                  -np.sin(la) * np.sin(lo),
                  np.cos(la)], axis=-1)
    return e, n


def rotate_vectors(u: np.ndarray, v: np.ndarray,
                   lon_geo: np.ndarray, lat_geo: np.ndarray,
                   R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate tangent vectors from the geographic frame into the rotated frame.

    The vectors are given at GEOGRAPHIC positions; their rotated-frame
    east/north components are returned at the rotated images of those
    positions (computed internally).  The transform is magnitude-preserving.

    Parameters
    ----------
    u, v : array_like
        Eastward and northward velocity components (any units) at the
        geographic positions ``(lon_geo, lat_geo)``.
    lon_geo, lat_geo : array_like
        Geographic longitude and latitude of the vectors in degrees.
    R : np.ndarray
        Shape (3, 3) Euler rotation matrix, ``v_rot = R @ v_geo`` (see
        :func:`euler_matrix`).

    Returns
    -------
    u_rot, v_rot : np.ndarray
        Eastward and northward components in the rotated frame, at the
        rotated images of the input positions.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    e_g, n_g = _east_north(lon_geo, lat_geo)
    vec3 = u[..., None] * e_g + v[..., None] * n_g
    vec3_r = vec3 @ R.T
    lon_r, lat_r = rotate_lonlat(lon_geo, lat_geo, R)
    e_r, n_r = _east_north(lon_r, lat_r)
    return (vec3_r * e_r).sum(-1), (vec3_r * n_r).sum(-1)


def unrotate_vectors(u_r: np.ndarray, v_r: np.ndarray,
                     lon_geo: np.ndarray, lat_geo: np.ndarray,
                     R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`rotate_vectors`: rotated-frame -> geographic vectors.

    The rotated-frame components are taken at the rotated images of the
    given GEOGRAPHIC positions and mapped back to geographic east/north
    components at those positions.  The transform is magnitude-preserving.

    Parameters
    ----------
    u_r, v_r : array_like
        Eastward and northward velocity components (any units) in the
        rotated frame, at the rotated images of ``(lon_geo, lat_geo)``.
    lon_geo, lat_geo : array_like
        Geographic longitude and latitude in degrees (the positions at
        which the geographic components are returned).
    R : np.ndarray
        Shape (3, 3) Euler rotation matrix, ``v_rot = R @ v_geo`` (see
        :func:`euler_matrix`); the inverse rotation ``R`` (transpose of
        ``R.T``) is applied internally.

    Returns
    -------
    u_geo, v_geo : np.ndarray
        Eastward and northward components in the geographic frame at
        ``(lon_geo, lat_geo)``.
    """
    u_r = np.asarray(u_r, dtype=np.float64)
    v_r = np.asarray(v_r, dtype=np.float64)
    lon_r, lat_r = rotate_lonlat(lon_geo, lat_geo, R)
    e_r, n_r = _east_north(lon_r, lat_r)
    vec3_r = u_r[..., None] * e_r + v_r[..., None] * n_r
    vec3_g = vec3_r @ R
    e_g, n_g = _east_north(lon_geo, lat_geo)
    return (vec3_g * e_g).sum(-1), (vec3_g * n_g).sum(-1)


__all__ = [
    "FESOM2_ANGLES", "euler_matrix",
    "rotate_lonlat", "unrotate_lonlat",
    "rotate_vectors", "unrotate_vectors",
]
