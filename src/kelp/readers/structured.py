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
                         find_nonconvex_quads, validate_quad_convexity)


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


def _unit_xyz(lon_deg, lat_deg):
    lo = np.deg2rad(lon_deg); la = np.deg2rad(lat_deg)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo),
                     np.sin(la)], axis=-1)


#: global layouts :func:`build_cgrid_mesh` handles (``orca=``)
CGRID_ORCA_LAYOUTS = (None, "ew", "fpivot")
#: chord distance (unit sphere, ~6 m) under which two raster points are the
#: same physical point (the halo / fold copies are exact in model output)
SAME_POINT_CHORD = 1e-6


def _orca_canonical_f(ny, nx, orca):
    """``(ny*nx,)`` map of every F raster index to its canonical copy.

    ``"ew"``: east-west halo columns (``col 0 == col nx-2``, ``col nx-1 ==
    col 1``): F column 0 -> nx-2.  ``"fpivot"`` adds the F-point-pivot north
    fold (``T(ny-1, i) == T(ny-2, nx-1-i)``): on the fold row ``ny-2``,
    ``F(ny-2, k) == F(ny-2, nx-2-k)`` -> ``min(k, nx-2-k)`` first, then the
    halo map.  Only rows ``0..ny-2`` and columns ``0..nx-2`` are referenced
    by the unique cells, so the map is complete for them.
    """
    jj, ii = np.mgrid[0:ny, 0:nx]
    ii = ii.copy()
    if orca == "fpivot":
        row = jj == ny - 2
        ii[row] = np.minimum(ii[row], np.maximum(nx - 2 - ii[row], 0))
    ii[ii == 0] = nx - 2
    return (jj * nx + ii).reshape(-1)


def build_cgrid_mesh(glamf, gphif, tmask2d, *, depth2d=None, e2u=None, e1v=None,
                     orca: Optional[str] = None, ring: bool = True,
                     name: str = "nemo",
                     r_earth: Optional[float] = None) -> Mesh:
    """Build the exact-C-grid quad :class:`Mesh`: nodes = F-points, elements =
    wet T-cells (the cell a ``numerics.uv_interp: cgrid`` particle lives in).

    NEMO convention: ``T(j,i)`` is the cell centre, ``u(j,i)`` sits on its EAST
    face, ``v(j,i)`` on its NORTH face and ``F(j,i)`` at its NE corner.  The
    element of ``T(j,i)`` (``1 <= j < ny``, ``1 <= i < nx``, wet) has the
    corners ``SW=F(j-1,i-1)``, ``SE=F(j-1,i)``, ``NE=F(j,i)``, ``NW=F(j,i-1)``;
    nodes are the F-points referenced by at least one kept element.

    Global ORCA (``orca``): the halo copies are dropped (cells ``i`` in
    ``1..nx-2``; with ``"fpivot"`` also the north halo row, cells ``j <=
    ny-2``) and the duplicated F-points are merged by index (verified against
    the coordinates), so the seam and the fold become interior edges.  Every
    cell still reads its OWN faces: the halo / fold copies of u and v equal
    their partners (with the fold's sign flip built into the raster), so no
    face needs a remap or a sign.

    Parameters
    ----------
    glamf, gphif : array_like (ny, nx)
        F-point longitude / latitude (deg) — in the frame the mesh lives in
        (the reader rotates them for ``mesh.rotation``).
    tmask2d : array_like (ny, nx)
        Surface T-point land-sea mask (truthy = wet).
    depth2d : array_like (ny, nx) or None
        T-point bottom depth (m); a node takes the max over its wet neighbours
        (and over its merged halo / fold copies).
    e2u, e1v : array_like (ny, nx) or None
        Model widths of the u faces (e2u) and v faces (e1v), m.  Both given:
        the interpolator forms fluxes ``u*e2u`` / ``v*e1v`` (the model's own
        continuity); otherwise it uses plane edge lengths.  Only the faces of
        wet cells must be positive and finite.
    orca : None, "ew" or "fpivot"
        Global layout (``readers.nemo._detect_orca_layout``): ``None`` =
        closed raster (edges are walls), ``"ew"`` = east-west halo columns,
        ``"fpivot"`` = halo columns + F-point-pivot north fold.
    ring : bool
        Append ONE ring of dry cells (8-neighbourhood of the wet region,
        across the ORCA halo / fold) as zero-velocity elements after the wet
        ones (default; land = zero velocity, as in Parcels).  A stage or end
        point in the ring reads zero velocity (the particle stays ALIVE);
        beyond the ring a stage still fails.  ``ring=False`` = wet cells only.  Ring cells
        that are non-convex or sit at a fold pivot are dropped (counted in
        ``_cgrid_ring_dropped``).

    Returns
    -------
    Mesh
        Quad mesh; stashes ``_structured_shape=(ny, nx)``, ``_cgrid=True``,
        ``_cgrid_orca``, ``_cgrid_elem_ji (E, 2)`` (the T-cell ``(j, i)`` of
        each element) and ``_cgrid_corners (E, 4)`` (node ids, SW/SE/NE/NW —
        the interpolator reads this, not ``elem_nodes``).  ``elem_nodes`` is
        the same quad (containment is winding-agnostic).  Also
        ``_cgrid_n_wet`` (the first ``n_wet`` elements are wet cells, the
        rest ring cells), ``_cgrid_nlev (E,) int32`` (1 wet / 0 ring).
    """
    if orca not in CGRID_ORCA_LAYOUTS:
        raise ValueError(f"orca must be one of {CGRID_ORCA_LAYOUTS}, got {orca!r}")
    lon2d = np.asarray(glamf, dtype=np.float64)
    lat2d = np.asarray(gphif, dtype=np.float64)
    if lon2d.ndim != 2 or lon2d.shape != lat2d.shape:
        raise ValueError(
            f"glamf/gphif must be 2-D (y, x) of equal shape, got "
            f"{lon2d.shape} and {lat2d.shape}")
    ny, nx = lon2d.shape
    wet = np.asarray(tmask2d) > 0.5
    if wet.shape != (ny, nx):
        raise ValueError(f"tmask shape {wet.shape} != grid {(ny, nx)}")
    wet = wet.copy()
    wet[0, :] = False                 # no F corners south / west of the raster
    wet[:, 0] = False
    if orca is not None:
        wet[:, nx - 1] = False        # east halo column (copy of column 1)
    if orca == "fpivot":
        wet[ny - 1, :] = False        # north halo row (mirror of row ny-2)
        piv = [(nx - 2) // 2, (nx - 2) // 2 + 1, 1, nx - 2]
        if wet[ny - 2, piv].any():
            raise ValueError(
                "cgrid ORCA mesh: a wet T-cell touches a north-fold pivot "
                f"(row {ny - 2}, columns {piv}); the fold's grid poles must "
                "be on land")
    if not wet.any():
        raise ValueError("cgrid mesh has no wet T-cells")
    ej, ei = np.nonzero(wet)
    raster = np.arange(ny * nx, dtype=np.int64).reshape(ny, nx)
    quad_raster = np.stack([raster[ej - 1, ei - 1], raster[ej - 1, ei],
                            raster[ej, ei], raster[ej, ei - 1]], axis=1)
    flat_lon = lon2d.reshape(-1)
    flat_lat = lat2d.reshape(-1)
    if orca is not None:
        canon = _orca_canonical_f(ny, nx, orca)
        used = np.unique(quad_raster)
        chord = np.linalg.norm(_unit_xyz(flat_lon[used], flat_lat[used])
                               - _unit_xyz(flat_lon[canon[used]],
                                           flat_lat[canon[used]]), axis=-1)
        if chord.max() > SAME_POINT_CHORD:
            k = int(used[np.argmax(chord)])
            raise ValueError(
                f"cgrid ORCA mesh ({orca}): F{divmod(k, nx)} and its copy "
                f"F{divmod(int(canon[k]), nx)} are {chord.max():.2e} apart "
                "(unit sphere): not the halo / fold layout")
        quad_raster = canon[quad_raster]
    else:
        canon = None
    n_wet = int(ej.size)
    n_ring = 0
    ring_dropped = {"pivot": 0, "nonconvex": 0, "chord": 0}
    if ring:
        # wetness of every raster cell, halo / fold copies taking their
        # canonical partner's value (so the seam / fold neighbourhoods are
        # visible without index arithmetic)
        jj, ii = np.mgrid[0:ny, 0:nx]
        cj, ci = jj.copy(), ii.copy()
        if orca == "fpivot":
            top = cj == ny - 1
            ci[top] = nx - 1 - ci[top]
            cj[top] = ny - 2
        if orca is not None:
            ci[ci == 0] = nx - 2
            ci[ci == nx - 1] = 1
        wet_full = wet[cj, ci]
        pd_ = np.pad(wet_full, 1)
        dil = np.zeros_like(wet_full)
        for dj in range(3):
            for di in range(3):
                dil |= pd_[dj:dj + ny, di:di + nx]
        rj, ri = np.nonzero(dil & ~wet_full)
        rj, ri = cj[rj, ri], ci[rj, ri]
        keep = (rj >= 1) & (ri >= 1)
        if orca is not None:
            keep &= ri <= nx - 2
            if orca == "fpivot":
                keep &= rj <= ny - 2
        rj, ri = rj[keep], ri[keep]
        rj, ri = np.unique(np.stack([rj, ri], 1), axis=0).T
        if orca == "fpivot":
            piv = [(nx - 2) // 2, (nx - 2) // 2 + 1, 1, nx - 2]
            keep = ~((rj == ny - 2) & np.isin(ri, piv))
            ring_dropped["pivot"] = int((~keep).sum())
            rj, ri = rj[keep], ri[keep]
        rq = np.stack([raster[rj - 1, ri - 1], raster[rj - 1, ri],
                       raster[rj, ri], raster[rj, ri - 1]], axis=1)
        if canon is not None:
            ch = np.linalg.norm(
                _unit_xyz(flat_lon[rq], flat_lat[rq])
                - _unit_xyz(flat_lon[canon[rq]], flat_lat[canon[rq]]),
                axis=-1).max(axis=1)
            keep = ch <= SAME_POINT_CHORD
            ring_dropped["chord"] = int((~keep).sum())
            rj, ri, rq = rj[keep], ri[keep], canon[rq[keep]]
        bad = find_nonconvex_quads(rq, flat_lon, flat_lat)
        keep = np.ones(rq.shape[0], dtype=bool)
        keep[bad] = False
        ring_dropped["nonconvex"] = int(bad.size)
        rj, ri, rq = rj[keep], ri[keep], rq[keep]
        n_ring = int(rj.size)
        ej = np.concatenate([ej, rj]); ei = np.concatenate([ei, ri])
        quad_raster = np.concatenate([quad_raster, rq], axis=0)
    referenced = np.unique(quad_raster)
    n_nodes = int(referenced.size)
    remap = np.full(ny * nx, -1, dtype=np.int64)
    remap[referenced] = np.arange(n_nodes, dtype=np.int64)
    corners = remap[quad_raster].astype(np.int32)

    nodes_lon = np.ascontiguousarray(flat_lon[referenced])
    nodes_lat = np.ascontiguousarray(flat_lat[referenced])
    depth = np.zeros(n_nodes, dtype=np.float64)
    if depth2d is not None:
        d = np.abs(np.asarray(depth2d, dtype=np.float64))
        d = np.where(np.asarray(tmask2d) > 0.5, np.nan_to_num(d), 0.0)
        dp = np.zeros((ny + 1, nx + 1))
        dp[:ny, :nx] = d                       # F(j,i): T(j,i)...T(j+1,i+1)
        dm = np.maximum.reduce([dp[:ny, :nx], dp[1:, :nx], dp[:ny, 1:], dp[1:, 1:]])
        dm = dm.reshape(-1)
        if canon is not None:                  # max over merged copies
            acc = np.zeros(ny * nx)
            np.maximum.at(acc, canon, dm)
            dm = acc
        depth = np.ascontiguousarray(dm[referenced])

    elon = nodes_lon[corners]
    elat = nodes_lat[corners]
    ref = elon[:, [0]]
    elon_unw = ref + ((elon - ref + 180.0) % 360.0 - 180.0)
    elem_center_lon = ((elon_unw.mean(axis=1) + 180.0) % 360.0) - 180.0
    elem_center_lat = elat.mean(axis=1)

    validate_quad_convexity(corners, nodes_lon, nodes_lat,
                            source=f"cgrid mesh '{name}'")
    n2e, n2e_count = build_node_to_elems(n_nodes, corners)
    kw = {} if r_earth is None else {"r_earth": r_earth}
    mesh = Mesh(
        name=name, nodes_lon=nodes_lon, nodes_lat=nodes_lat, depth=depth,
        elem_nodes=corners.copy(),
        elem_center_lon=np.ascontiguousarray(elem_center_lon),
        elem_center_lat=np.ascontiguousarray(elem_center_lat),
        node_to_elems=n2e, node_to_elems_count=n2e_count,
        sigma=np.array([], dtype=np.float64), vertical_coord="zlevel", **kw)
    mesh._structured_shape = (ny, nx)
    mesh._cgrid = True
    mesh._cgrid_orca = orca
    mesh._cgrid_elem_ji = np.stack([ej, ei], axis=1).astype(np.int64)
    mesh._cgrid_corners = corners
    mesh._cgrid_n_wet = n_wet
    mesh._cgrid_nlev = np.r_[np.ones(n_wet), np.zeros(n_ring)].astype(np.int32)
    mesh._cgrid_ring_dropped = dict(ring_dropped)
    if (e2u is None) != (e1v is None):
        raise ValueError("pass both e2u and e1v, or neither")
    # the faces the kept cells read: W/E (u at (j, i-1), (j, i)), S/N (v at
    # (j-1, i), (j, i)) -- land-eliminated / NaN-filled widths elsewhere load
    wj, wi = ej[:n_wet], ei[:n_wet]           # ring cells read no face
    faces = {"_cgrid_e2u": np.r_[wj * nx + wi - 1, wj * nx + wi],
             "_cgrid_e1v": np.r_[(wj - 1) * nx + wi, wj * nx + wi]}
    for nm, a in (("_cgrid_e2u", e2u), ("_cgrid_e1v", e1v)):
        if a is None:
            setattr(mesh, nm, np.empty(0, dtype=np.float64))
        else:
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (ny, nx):
                raise ValueError(f"{nm[1:]} shape {a.shape} != grid {(ny, nx)}")
            a = a.reshape(-1)
            f = a[faces[nm]]
            if not (np.isfinite(f).all() and (f > 0).all()):
                raise ValueError(
                    f"{nm[1:]} must be positive and finite on the faces of "
                    f"the wet cells ({int((~(np.isfinite(f) & (f > 0))).sum())} "
                    "are not)")
            setattr(mesh, nm, np.ascontiguousarray(a))
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


__all__ = ["build_structured_mesh", "build_cgrid_mesh", "c_grid_to_t", "field_to_nodes"]
