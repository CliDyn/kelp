"""The exact C-grid quad mesh of a NEMO raster (``mesh.nemo_grid: cgrid``)."""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from ...core.mesh import (Mesh, build_node_to_elems, build_search,
                          find_nonconvex_quads, validate_quad_convexity)
from ._layout import (SAME_POINT_CHORD, NemoCgridLayout, _unit_xyz, nemo_cgrid_canonical,
                      nemo_fold_pivot_cells)
from ._vertical import _partial_level


def build_nemo_cgrid_mesh(glamf, gphif, tmask2d, *, depth2d=None, e2u=None, e1v=None,
                     layout: Optional[NemoCgridLayout] = None,
                     nlev2d=None, zw=None, area=None, e3b=None, e3u0=None, e3v0=None,
                     name: str = "nemo", partial_cells: str = "shrunk",
                     r_earth: Optional[float] = None) -> Mesh:
    """Build the exact-C-grid quad :class:`Mesh`: nodes = F-points, elements =
    wet T-cells (the cell a ``numerics.uv_interp: nemo_cgrid`` particle lives in).

    NEMO convention: ``T(j,i)`` is the cell centre, ``u(j,i)`` sits on its EAST
    face, ``v(j,i)`` on its NORTH face and ``F(j,i)`` at its NE corner.  The
    element of ``T(j,i)`` (``1 <= j < ny``, ``1 <= i < nx``, wet) has the
    corners ``SW=F(j-1,i-1)``, ``SE=F(j-1,i)``, ``NE=F(j,i)``, ``NW=F(j,i-1)``;
    nodes are the F-points referenced by at least one kept element.

    Global ORCA (``layout``): the rasters are in the CANONICAL form of
    :class:`NemoCgridLayout` (E-W halo columns, north halo row, every copy filled
    from its canonical point, vectors with the fold sign).  Only the canonical
    cells are kept (a halo column or the mirrored half of the fold row is the
    same cell as its canonical point), and the corners are the canonical F
    points, so the seam and the fold become interior edges; every cell reads its
    OWN faces, which the filled copies make equal to their partners'.

    Parameters
    ----------
    glamf, gphif : array_like (ny, nx)
        F-point longitude / latitude (deg) — in the frame the mesh lives in
        (the reader rotates them for ``mesh.rotation``).
    tmask2d : array_like (ny, nx)
        Surface T-point land-sea mask (truthy = wet).
    depth2d : array_like (ny, nx) or None
        T-point bottom depth (m); a node takes the max over its wet neighbours
        (the copies are filled: canonical).
    e2u, e1v : array_like (ny, nx) or None
        Model widths of the u faces (e2u) and v faces (e1v), m.  Both given:
        the interpolator forms fluxes ``u*e2u`` / ``v*e1v`` (the model's own
        continuity); otherwise it uses plane edge lengths.  Only the faces of
        wet cells must be positive and finite.
    layout : NemoCgridLayout or None
        The raster layout (:func:`._layout.detect_nemo_cgrid_layout`); the arrays
        must have ``layout.shape``.  ``None`` = a closed regional raster (its
        edges are walls or open water: nothing is merged).
    nlev2d, zw, area, e3b, e3u0, e3v0 : array_like or None
        3-D (z-level, partial bottom cells): ``nlev2d (ny, nx)`` wet levels per
        column, ``zw (K+1,)`` the reference layer interfaces (positive down),
        ``area (ny, nx)`` the T-cell area ``e1t*e2t`` and ``e3b (ny, nx)`` the
        thickness of each column's bottom wet cell (default: the reference
        layer = full cells); the floor is ``zw[nlev-1] + e3b`` (also the node
        depth when ``depth2d`` is None).  ``e3u0`` / ``e3v0 (K, ny, nx)`` are
        the model's face thicknesses (default: the reference layer); only a
        face's bottom level differs from it (zps), so the mesh keeps one level
        index and one thickness per face (``_nemo_cgrid_kbu/_e3u``, ``_kbv/_e3v``,
        ``-1`` = no partial level).  ``mesh.sigma`` becomes ``zw`` and
        ``_nemo_cgrid_nlev`` the wet levels (ring cells 0; every level at or below a
        column's floor is zero-velocity land).

    Returns
    -------
    Mesh
        Quad mesh; stashes ``_structured_shape=(ny, nx)``, ``_nemo_cgrid=True``,
        ``_nemo_cgrid_layout``, ``_nemo_cgrid_elem_ji (E, 2)`` (the T-cell ``(j, i)`` of
        each element) and ``_nemo_cgrid_corners (E, 4)`` (node ids, SW/SE/NE/NW —
        the interpolator reads this, not ``elem_nodes``).  ``elem_nodes`` is
        the same quad (containment is winding-agnostic).  Also
        ``_nemo_cgrid_nlev (E,) int32`` (the wet levels: 1 for a wet 2-D cell; the
        wet elements come first, the ring cells (0) after them).

    Notes
    -----
    The mesh always carries ONE ring of dry cells (8-neighbourhood of the wet
    region, across the seam and the fold) as zero-velocity elements after the
    wet ones (land = zero velocity, as in Parcels): a stage or end point in the
    ring reads zero velocity (the particle stays ALIVE); beyond the ring a
    stage still fails.  Ring cells that are non-convex or sit at a fold pivot
    are dropped.
    """
    lon2d = np.asarray(glamf, dtype=np.float64)
    lat2d = np.asarray(gphif, dtype=np.float64)
    if lon2d.ndim != 2 or lon2d.shape != lat2d.shape:
        raise ValueError(
            f"glamf/gphif must be 2-D (y, x) of equal shape, got "
            f"{lon2d.shape} and {lat2d.shape}")
    ny, nx = lon2d.shape
    if layout is None:
        layout = NemoCgridLayout((ny, nx))
    if layout.shape != (ny, nx):
        raise ValueError(f"the rasters are {(ny, nx)}, the layout's canonical shape is {layout.shape}")
    wet = np.asarray(tmask2d) > 0.5
    if wet.shape != (ny, nx):
        raise ValueError(f"tmask shape {wet.shape} != grid {(ny, nx)}")
    # a cell is a canonical T point with an F corner south and west of it
    jj, ii = np.mgrid[0:ny, 0:nx]
    cj, ci, _ = nemo_cgrid_canonical("T", jj, ii, layout)
    piv = nemo_fold_pivot_cells(layout)
    own = (cj == jj) & (ci == ii) & (jj >= 1) & (ii >= 1)
    if (wet & own & piv).any():
        raise ValueError(
            "NEMO C-grid mesh: a wet T-cell sits at a north-fold pivot "
            f"(cells {np.argwhere(wet & piv).tolist()[:4]}); the fold's grid poles "
            "must be on land")
    wet = wet & own
    if not wet.any():
        raise ValueError("NEMO C-grid mesh has no wet T-cells")
    flat_lon = lon2d.reshape(-1)
    flat_lat = lat2d.reshape(-1)

    def quads(j, i):
        """Canonical flat F indices of the corners SW, SE, NE, NW of cells (j, i)."""
        out = []
        for dj, di in ((-1, -1), (-1, 0), (0, 0), (0, -1)):
            fj, fi, _ = nemo_cgrid_canonical("F", j + dj, i + di, layout)
            out.append(fj * nx + fi)
        return np.stack(out, axis=1)

    ej, ei = np.nonzero(wet)
    quad_raster = quads(ej, ei)
    n_wet = int(ej.size)
    # a raster point is wet when its canonical cell is (the seam / fold
    # neighbourhoods are then seen without index arithmetic)
    wet_full = wet[cj, ci]
    pd_ = np.pad(wet_full, 1)
    dil = np.zeros_like(wet_full)
    for dj in range(3):
        for di in range(3):
            dil |= pd_[dj:dj + ny, di:di + nx]
    rj, ri = np.nonzero(dil & ~wet_full)
    rj, ri = cj[rj, ri], ci[rj, ri]
    keep = (rj >= 1) & (ri >= 1)
    rj, ri = np.unique(np.stack([rj[keep], ri[keep]], 1), axis=0).T
    keep = ~piv[rj, ri]
    rj, ri = rj[keep], ri[keep]
    rq = quads(rj, ri)
    bad = find_nonconvex_quads(rq, flat_lon, flat_lat)
    keep = np.ones(rq.shape[0], dtype=bool)
    keep[bad] = False
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
    if zw is not None:
        nf = np.asarray(nlev2d, np.int32)
        eb = (np.diff(np.asarray(zw, np.float64))[np.maximum(nf - 1, 0)] if e3b is None
              else np.asarray(e3b, np.float64))
        eb = np.where(nf > 0, eb, 0.0)
        if depth2d is None:
            depth2d = np.where(nf > 0, np.asarray(zw, np.float64)[np.maximum(nf - 1, 0)] + eb, 0.0)
    depth = np.zeros(n_nodes, dtype=np.float64)
    if depth2d is not None:
        d = np.abs(np.asarray(depth2d, dtype=np.float64))
        d = np.where(np.asarray(tmask2d) > 0.5, np.nan_to_num(d), 0.0)
        dp = np.zeros((ny + 1, nx + 1))
        dp[:ny, :nx] = d                       # F(j,i): T(j,i)...T(j+1,i+1)
        dm = np.maximum.reduce([dp[:ny, :nx], dp[1:, :nx], dp[:ny, 1:], dp[1:, 1:]])
        dm = dm.reshape(-1)
        depth = np.ascontiguousarray(dm[referenced])

    elon = nodes_lon[corners]
    elat = nodes_lat[corners]
    ref = elon[:, [0]]
    elon_unw = ref + ((elon - ref + 180.0) % 360.0 - 180.0)
    elem_center_lon = ((elon_unw.mean(axis=1) + 180.0) % 360.0) - 180.0
    elem_center_lat = elat.mean(axis=1)

    pairs = cKDTree(_unit_xyz(nodes_lon, nodes_lat)).query_pairs(SAME_POINT_CHORD)
    if pairs:
        a_, b_ = min(pairs)
        raise ValueError(
            f"NEMO C-grid mesh: {len(pairs)} pairs of mesh nodes coincide (e.g. "
            f"({nodes_lon[a_]:.4f}, {nodes_lat[a_]:.4f})): the layout does not merge them "
            "(a crop through the fold that is not full width?)")
    validate_quad_convexity(corners, nodes_lon, nodes_lat,
                            source=f"NEMO C-grid mesh '{name}'")
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
    mesh._nemo_cgrid = True
    mesh._nemo_cgrid_layout = layout
    if partial_cells not in ("shrunk", "porous"):
        raise ValueError(f"partial_cells must be 'shrunk' or 'porous', got {partial_cells!r}")
    mesh._nemo_cgrid_partial = partial_cells
    mesh._nemo_cgrid_elem_ji = np.stack([ej, ei], axis=1).astype(np.int64)
    mesh._nemo_cgrid_corners = corners
    nl = (np.ones(n_wet) if nlev2d is None
          else np.asarray(nlev2d)[ej[:n_wet], ei[:n_wet]])
    mesh._nemo_cgrid_nlev = np.r_[nl, np.zeros(n_ring)].astype(np.int32)
    mesh._nemo_cgrid_nlev_raster = np.zeros((ny, nx), dtype=np.int32)
    mesh._nemo_cgrid_nlev_raster[ej[:n_wet], ei[:n_wet]] = nl
    # wet levels of the u / v faces (NEMO umask / vmask = tmask * tmask) from the
    # raw tmask -- the cells without an F-point quad (first row / column) are
    # still water for their faces; a raster edge is open (no outer cell)
    lev = (np.asarray(nlev2d) if nlev2d is not None else np.asarray(tmask2d) > 0.5).astype(np.int32)
    fu, fv = lev.copy(), lev.copy()
    fu[:, :-1] = np.minimum(lev[:, :-1], lev[:, 1:])
    fv[:-1] = np.minimum(lev[:-1], lev[1:])
    mesh._nemo_cgrid_face_levels = {"U": fu, "V": fv}
    mesh._nemo_cgrid_zw = np.empty(0) if zw is None else np.asarray(zw, np.float64)
    mesh._nemo_cgrid_area = (np.empty(0) if area is None
                        else np.ascontiguousarray(area, np.float64).reshape(-1))
    mesh._nemo_cgrid_e3b = np.empty(0)
    for nm in ("u", "v"):
        setattr(mesh, f"_nemo_cgrid_kb{nm}", np.empty(0, np.int32))
        setattr(mesh, f"_nemo_cgrid_e3{nm}", np.empty(0))
    if zw is not None:
        mesh.sigma = mesh._nemo_cgrid_zw
        mesh._nemo_cgrid_e3b = np.ascontiguousarray(eb).reshape(-1)
        for nm, e3 in (("u", e3u0), ("v", e3v0)):
            kb, th = _partial_level(mesh._nemo_cgrid_zw, e3, (ny, nx))
            setattr(mesh, f"_nemo_cgrid_kb{nm}", kb)
            setattr(mesh, f"_nemo_cgrid_e3{nm}", th)
    if (e2u is None) != (e1v is None):
        raise ValueError("pass both e2u and e1v, or neither")
    # the faces the kept cells read: W/E (u at (j, i-1), (j, i)), S/N (v at
    # (j-1, i), (j, i)) -- land-eliminated / NaN-filled widths elsewhere load
    wj, wi = ej[:n_wet], ei[:n_wet]           # ring cells read no face
    faces = {"_nemo_cgrid_e2u": np.r_[wj * nx + wi - 1, wj * nx + wi],
             "_nemo_cgrid_e1v": np.r_[(wj - 1) * nx + wi, wj * nx + wi]}
    for nm, a in (("_nemo_cgrid_e2u", e2u), ("_nemo_cgrid_e1v", e1v)):
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
