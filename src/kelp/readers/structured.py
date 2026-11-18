"""Structured / curvilinear grid -> kelp unstructured Mesh (foundation).

Maps a logically-rectangular (curvilinear or regular lat/lon) ocean grid onto
the unstructured ``Mesh`` the tracker already uses, so the whole existing stack
(search, integrators, diffusion, beaching, sampling, rotation, float32/prefetch)
works with no changes:

- **nodes = wet cell-centre (T) points**;
- **elements = quads of 4 logically-adjacent T-points** kept only when all four
  corners are wet (a land corner drops the quad -> coastline, exactly like an
  unstructured-mesh boundary).

This is model-agnostic; per-model adapters (NEMO first, then ROMS / MOM / plain
regular lat-lon) read their files into ``(lon2d, lat2d, wet_mask, u, v, ...)``
and feed this builder.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..core.mesh import (Mesh, build_node_to_elems, build_search,
                         validate_quad_convexity)


def build_structured_mesh(lon2d, lat2d, wet_mask, *,
                          depth2d=None, z_levels=None,
                          name: str = "structured",
                          r_earth: Optional[float] = None) -> Mesh:
    """Build a quad :class:`Mesh` from a logically-rectangular grid.

    Parameters
    ----------
    lon2d, lat2d : array_like, shape (ny, nx)
        Curvilinear cell-centre (T-point) coordinates in degrees.
    wet_mask : array_like (ny, nx), bool
        True = ocean T-cell, False = land (dropped).
    depth2d : array_like (ny, nx) or None
        Bottom depth per T-cell (m, sign-agnostic -> abs); 0 when None.
    z_levels : array_like or None
        Vertical interface depths (m) for 3D; empty (2D) when None.
    name : str
    r_earth : float or None
        Earth radius (m) for mesh-derived areas; reader-family default when None.

    Returns
    -------
    Mesh
        Quad mesh (``elem_nodes`` shape ``(E, 4)``); also carries
        ``_structured_shape = (ny, nx)`` and ``_structured_node_id`` (the
        ``(ny, nx) -> node id`` map, ``-1`` on land) for C-grid -> T mapping.
    """
    lon2d = np.asarray(lon2d, dtype=np.float64)
    lat2d = np.asarray(lat2d, dtype=np.float64)
    if lon2d.ndim != 2 or lon2d.shape != lat2d.shape:
        raise ValueError(
            f"lon2d/lat2d must be 2-D (y, x) of equal shape, got "
            f"{lon2d.shape} and {lat2d.shape}")
    ny, nx = lon2d.shape
    if ny < 2 or nx < 2:
        raise ValueError(f"structured grid must be at least 2x2, got {(ny, nx)}")
    wet = np.asarray(wet_mask, dtype=bool)
    if wet.shape != (ny, nx):
        raise ValueError(f"wet_mask shape {wet.shape} != grid {(ny, nx)}")

    if not wet.any():
        raise ValueError("wet_mask has no ocean cells")

    # --- elements = quads of 4 adjacent T-points, kept iff all 4 are wet ---
    # corners (j,i) (j,i+1) (j+1,i+1) (j+1,i) -> consistent winding (a simple
    # quad on a curvilinear grid), matching the Wachspress weight contract.
    blk = wet[:-1, :-1] & wet[:-1, 1:] & wet[1:, 1:] & wet[1:, :-1]
    if not blk.any():
        raise ValueError("structured mesh has no fully-wet quads")
    bj, bi = np.nonzero(blk)
    raster = np.arange(ny * nx, dtype=np.int64).reshape(ny, nx)
    quad_raster = np.stack([raster[bj, bi], raster[bj, bi + 1],
                            raster[bj + 1, bi + 1], raster[bj + 1, bi]], axis=1)

    # --- nodes = ONLY T-points referenced by >=1 kept quad ---
    # A wet cell with no fully-wet quad (1-cell channel, ragged coast, island)
    # would be a dead-end node: if it is the nearest node, containment fails
    # immediately -> a false OUTSIDE.  Excluding such orphans is correct (they
    # are untrackable: no element to interpolate in).
    referenced = np.unique(quad_raster)            # sorted full-raster ids
    n_nodes = int(referenced.size)
    remap = np.full(ny * nx, -1, dtype=np.int64)
    remap[referenced] = np.arange(n_nodes, dtype=np.int64)
    elem_nodes = remap[quad_raster].astype(np.int32)
    node_id = remap.reshape(ny, nx)                # (ny, nx) -> node id (-1 land/orphan)

    nodes_lon = np.ascontiguousarray(lon2d.reshape(-1)[referenced])
    nodes_lat = np.ascontiguousarray(lat2d.reshape(-1)[referenced])
    if depth2d is not None:
        depth = np.ascontiguousarray(
            np.abs(np.asarray(depth2d, dtype=np.float64)).reshape(-1)[referenced])
    else:
        depth = np.zeros(n_nodes, dtype=np.float64)

    # element centres = mean of the 4 corner T-points (dateline-safe: unwrap to
    # the first corner before averaging, then re-wrap to [-180, 180)).
    elon = nodes_lon[elem_nodes]            # (E, 4)
    elat = nodes_lat[elem_nodes]
    ref = elon[:, [0]]
    elon_unw = ref + ((elon - ref + 180.0) % 360.0 - 180.0)
    elem_center_lon = ((elon_unw.mean(axis=1) + 180.0) % 360.0) - 180.0
    elem_center_lat = elat.mean(axis=1)

    # Load-time mesh-contract check (2026-07): strongly sheared curvilinear
    # cells can go non-convex; the guard-free product-form Wachspress
    # weights would then produce garbage silently — fail loud instead.
    validate_quad_convexity(elem_nodes, nodes_lon, nodes_lat,
                            source=f"structured mesh '{name}'")

    n2e, n2e_count = build_node_to_elems(n_nodes, elem_nodes)
    sigma = (np.abs(np.asarray(z_levels, dtype=np.float64))
             if z_levels is not None else np.array([], dtype=np.float64))
    kw = {} if r_earth is None else {"r_earth": r_earth}
    mesh = Mesh(
        name=name, nodes_lon=nodes_lon, nodes_lat=nodes_lat, depth=depth,
        elem_nodes=elem_nodes,
        elem_center_lon=np.ascontiguousarray(elem_center_lon),
        elem_center_lat=np.ascontiguousarray(elem_center_lat),
        node_to_elems=n2e, node_to_elems_count=n2e_count,
        sigma=sigma, vertical_coord="zlevel", **kw)
    # Stash the structured layout for the C-grid -> T-point velocity mapping.
    mesh._structured_shape = (ny, nx)
    mesh._structured_node_id = node_id
    build_search(mesh)
    return mesh


def c_grid_to_t(u_faces, v_faces, node_id) -> Tuple[np.ndarray, np.ndarray]:
    """Average Arakawa-C-grid face velocities onto wet T-points (-> node layout).

    NEMO convention: ``u(j,i)`` sits on the EAST face of T-cell ``(j,i)`` (between
    ``T(j,i)`` and ``T(j,i+1)``); ``v(j,i)`` on the NORTH face.  The T-point value
    is the mean of its two bracketing faces; the west/south domain-edge column/row
    keeps the single available face.  Land faces are expected to be 0 (model
    mask), so a wet cell next to a land face is halved — the standard A-grid
    reduction (loses exact C-grid divergence; documented).

    Parameters
    ----------
    u_faces, v_faces : array_like (ny, nx)
        C-grid face velocities (float; computed in float64).
    node_id : ndarray (ny, nx)
        ``mesh._structured_node_id`` — wet cell -> node id, ``-1`` on land.

    Returns
    -------
    (u_t, v_t) : ndarray (n_nodes,) float64
        Node-indexed T-point velocities.
    """
    # Land faces in real model output are masked with NaN / a decoded
    # _FillValue, not 0 — averaging those would poison coastal wet T-points
    # (NaN / huge).  Replace non-finite faces with 0 (the standard masked-face
    # treatment); a wet cell next to a land face is then halved (documented).
    u_faces = np.nan_to_num(np.asarray(u_faces, dtype=np.float64),
                            nan=0.0, posinf=0.0, neginf=0.0)
    v_faces = np.nan_to_num(np.asarray(v_faces, dtype=np.float64),
                            nan=0.0, posinf=0.0, neginf=0.0)
    ny, nx = node_id.shape
    if u_faces.shape != (ny, nx) or v_faces.shape != (ny, nx):
        raise ValueError(
            f"u/v face shapes {u_faces.shape}/{v_faces.shape} != grid {(ny, nx)}")
    u_t = u_faces.copy()
    u_t[:, 1:] = 0.5 * (u_faces[:, :-1] + u_faces[:, 1:])   # west+east faces
    v_t = v_faces.copy()
    v_t[1:, :] = 0.5 * (v_faces[:-1, :] + v_faces[1:, :])   # south+north faces

    wet = node_id >= 0
    ids = node_id[wet]
    u_out = np.empty(int(wet.sum()), dtype=np.float64)
    v_out = np.empty_like(u_out)
    u_out[ids] = u_t[wet]
    v_out[ids] = v_t[wet]
    return u_out, v_out


def field_to_nodes(field2d, node_id) -> np.ndarray:
    """Map a ``(ny, nx)`` T-point field onto the wet-node array (``(n_nodes,)``).

    For T-point scalars (eta, temperature, …) that are already cell-centred — no
    staggering, just the wet-cell -> node-id compaction.
    """
    field2d = np.asarray(field2d, dtype=np.float64)
    if field2d.shape != node_id.shape:
        raise ValueError(
            f"field shape {field2d.shape} != grid {node_id.shape}")
    wet = node_id >= 0
    out = np.empty(int(wet.sum()), dtype=np.float64)
    out[node_id[wet]] = field2d[wet]
    return out


__all__ = ["build_structured_mesh", "c_grid_to_t", "field_to_nodes"]
