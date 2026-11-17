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

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

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


# --- C-grid raster layouts: E-W periodicity and the north fold -----------------
# NEMO's global grids repeat points: E-W halo columns (col 0 == col nx-2, col
# nx-1 == col 1) and, north of the last real row, a fold that maps every point
# to its image under a 180 degree rotation about the fold centre (T-point pivot:
# ORCA2 / ORCA025 / ORCA12; F-point pivot: ORCA05 / ORCA1).  The reader brings
# every raster to ONE canonical form -- NEMO's halo form: halo columns when
# periodic, the halo row above the fold row when folded, every copy overwritten
# from its canonical point (the vector sign flips across the fold) -- so the
# kernels, the thickness / W helpers and the ring read each face as a plain
# raster neighbour, at the seam and across the fold alike.
_KIND_OFFSET = {"T": (0, 0), "U": (0, 1), "V": (1, 0), "F": (1, 1)}   # (oy, ox) on the doubled lattice


@dataclass(frozen=True)
class CgridLayout:
    """What a raw ``(ny, nx)`` NEMO raster contains and its canonical form.

    ``halo_cols``: the input carries the E-W halo columns; ``halo_row``: it
    carries the north halo row above the fold row; ``periodic``: east-west
    periodic (always with ``halo_cols`` or ``fold``); ``fold``: ``None``,
    ``"T"`` or ``"F"`` (the pivot type).  The canonical raster has the halo
    columns (padded when absent) and the halo row (padded when absent);
    ``J`` is its fold row (the last real T row)."""
    raw_shape: Tuple[int, int]
    halo_cols: bool = False
    halo_row: bool = False
    periodic: bool = False
    fold: Optional[str] = None

    def __post_init__(self):
        if self.fold not in (None, "T", "F"):
            raise ValueError(f"fold must be None, 'T' or 'F', got {self.fold!r}")
        if (self.halo_cols or self.fold) and not self.periodic:
            raise ValueError("halo columns and a fold imply E-W periodicity")
        if self.halo_row and not self.fold:
            raise ValueError("a halo row needs a fold")

    @property
    def pad_cols(self) -> bool:
        return self.periodic and not self.halo_cols

    @property
    def pad_row(self) -> bool:
        return self.fold is not None and not self.halo_row

    @property
    def shape(self) -> Tuple[int, int]:
        ny, nx = self.raw_shape
        return ny + self.pad_row, nx + 2 * self.pad_cols

    @property
    def c0(self) -> int:           # first interior column
        return 1 if self.periodic else 0

    @property
    def ni(self) -> int:           # periodic width (0 = not periodic)
        return self.shape[1] - 2 if self.periodic else 0

    @property
    def J(self) -> int:            # fold row = last real T row
        return self.shape[0] - 2 if self.fold else self.shape[0] - 1

    def label(self) -> str:
        """One line for the log, ``kelp info`` and the output metadata."""
        parts = [f"{self.fold}-pivot north fold" + (" (halo row)" if self.halo_row else " (halo-free, padded)")
                 if self.fold else "no fold"]
        if self.periodic:
            parts.append("E-W periodic" + (" (halo columns)" if self.halo_cols else " (halo-free, padded)"))
        edges = "S" + ("" if self.fold else ", N") + ("" if self.periodic else ", E, W")
        shape = ("" if self.shape == tuple(self.raw_shape) else
                 f"; {self.raw_shape[0]}x{self.raw_shape[1]} -> {self.shape[0]}x{self.shape[1]}")
        return f"{', '.join(parts)}; unjoined raster edges (open where wet): {edges}{shape}"


def _centre(layout: CgridLayout, kind: str):
    """Fold centre ``(Y2, X2)``: the mirror of ``kind`` point ``(j, i)`` is
    ``(Y2 - j - oy, X2 - i - ox)`` (columns modulo the period)."""
    f = layout.fold == "F"
    return 2 * layout.J + f, 2 * layout.c0 + layout.ni - f


def _mirror(kind: str, j, i, layout: CgridLayout):
    oy, ox = _KIND_OFFSET[kind]
    Y2, X2 = _centre(layout, kind)
    c0, ni = layout.c0, layout.ni
    return Y2 - j - oy, c0 + ((X2 - i - ox) - c0) % ni


def cgrid_canonical(kind: str, j, i, layout: CgridLayout):
    """Canonical ``(j, i, sign)`` of the canonical-raster points ``(j, i)`` of
    ``kind`` (``"T"``, ``"U"``, ``"V"``, ``"F"``): the E-W wrap, then the fold
    (a point beyond the fold line, or on it with the larger column, maps to its
    mirror image, ``sign = -1``: vectors flip there).  Idempotent."""
    j = np.asarray(j, dtype=np.int64)
    i = np.asarray(i, dtype=np.int64)
    sign = np.ones(j.shape)
    if layout.periodic:
        i = layout.c0 + (i - layout.c0) % layout.ni
    if layout.fold:
        oy, _ = _KIND_OFFSET[kind]
        mj, mi = _mirror(kind, j, i, layout)
        beyond = (2 * j + oy > _centre(layout, kind)[0]) | (
            (2 * j + oy == _centre(layout, kind)[0]) & (mi < i))
        j, i = np.where(beyond, mj, j), np.where(beyond, mi, i)
        sign = np.where(beyond, -1.0, 1.0)
    return j, i, sign


@lru_cache(maxsize=64)
def _copy_sets(layout: CgridLayout, kind: str):
    """``(dst, src, sign, present)``: flat canonical indices of every copy point
    (E-W halo columns, the fold row's mirrored half, the halo row), the flat
    index of its canonical point, the vector sign, and the mask of canonical
    points the raw raster carries."""
    nr, nc = layout.shape
    ny, nx = layout.raw_shape
    cand = np.zeros((nr, nc), dtype=bool)
    if layout.periodic:
        cand[:, [0, nc - 1]] = True
    if layout.fold:
        cand[layout.J:, :] = True
    jj, ii = np.nonzero(cand)
    cj, ci, s = cgrid_canonical(kind, jj, ii, layout)
    if ((cj < 0) | (cj >= nr) | (ci < 0) | (ci >= nc)).any():
        raise ValueError("the raster is too small for its fold / halo layout")
    moved = (cj != jj) | (ci != ii)
    present = np.zeros((nr, nc), dtype=bool)
    pc = int(layout.pad_cols)
    present[:ny, pc:pc + nx] = True
    return ((jj * nc + ii)[moved], (cj * nc + ci)[moved], s[moved], present.reshape(-1))


def cgrid_to_canonical(a, layout: CgridLayout, kind: str, *, vector: bool = False,
                       check: bool = False, name: str = "raster") -> np.ndarray:
    """Canonical float64 raster of ``a (..., ny, nx)`` (NaN / inf -> 0): padded
    to the halo form, every copy overwritten from its canonical point (``vector``:
    times the fold sign -- only u and v).  ``check=True`` first compares the
    copies the raw raster carries with their canonical points (exact: NEMO
    writes bit copies) and raises ``ValueError`` naming ``name`` on a mismatch
    -- a stale halo, a wrong layout or the wrong grid file."""
    a = np.asarray(a)
    ny, nx = layout.raw_shape
    if a.ndim < 2 or a.shape[-2:] != (ny, nx):
        raise ValueError(f"{name}: raster shape {a.shape[-2:]} != the mesh grid {(ny, nx)}")
    nr, nc = layout.shape
    pc = int(layout.pad_cols)
    out = np.zeros(a.shape[:-2] + (nr, nc))
    out[..., :ny, pc:pc + nx] = a
    np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    flat = out.reshape(a.shape[:-2] + (-1,))
    dst, src, sgn, present = _copy_sets(layout, kind)
    if not vector:
        sgn = np.ones_like(sgn)
    if check:
        k = present[dst]
        bad = flat[..., dst[k]] != flat[..., src[k]] * sgn[k]
        if bad.any():
            first = np.argwhere(bad)[0]
            d = int(dst[k][first[-1]])
            diff = np.abs(flat[..., dst[k]] - flat[..., src[k]] * sgn[k])
            raise ValueError(
                f"{name}: {int(bad.sum())} halo / fold copies differ from their "
                f"canonical points (max |diff| {float(diff.max()):.3g}, first at "
                f"{tuple(int(x) for x in first[:-1]) + divmod(d, nc)} of the "
                f"canonical raster): stale copies, a wrong layout, or a field "
                f"from another grid file")
    flat[..., dst] = flat[..., src] * sgn
    return out


def fold_pivot_cells(layout: CgridLayout) -> np.ndarray:
    """Canonical-raster mask of the cells at a fold pivot (their mirror image is
    themselves): the self-mapped T cells (T-point pivot, degenerate quads) or the
    cells with a self-mapped F corner (F-point pivot).  Must be land."""
    piv = np.zeros(layout.shape, dtype=bool)
    if not layout.fold:
        return piv
    J, c0, ni = layout.J, layout.c0, layout.ni
    i = np.arange(c0, c0 + ni)
    mj, mi = _mirror(layout.fold, J, i, layout)
    hit = i[(mj == J) & (mi == i)]
    for k in hit:
        for c in ((k,) if layout.fold == "T" else (k, k + 1)):
            piv[J, c0 + (c - c0) % ni] = True
    return piv


def _unit_xyz(lon_deg, lat_deg):
    lo = np.deg2rad(lon_deg); la = np.deg2rad(lat_deg)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo),
                     np.sin(la)], axis=-1)


#: chord distance (unit sphere, ~6 m) under which two raster points are the
#: same physical point (the halo / fold copies are exact in model output)
SAME_POINT_CHORD = 1e-6


def build_cgrid_mesh(glamf, gphif, tmask2d, *, depth2d=None, e2u=None, e1v=None,
                     layout: Optional[CgridLayout] = None, ring: bool = True,
                     nlev2d=None, zw=None, area=None, e3b=None, e3u0=None, e3v0=None,
                     name: str = "nemo", partial_cells: str = "shrunk",
                     r_earth: Optional[float] = None) -> Mesh:
    """Build the exact-C-grid quad :class:`Mesh`: nodes = F-points, elements =
    wet T-cells (the cell a ``numerics.uv_interp: cgrid`` particle lives in).

    NEMO convention: ``T(j,i)`` is the cell centre, ``u(j,i)`` sits on its EAST
    face, ``v(j,i)`` on its NORTH face and ``F(j,i)`` at its NE corner.  The
    element of ``T(j,i)`` (``1 <= j < ny``, ``1 <= i < nx``, wet) has the
    corners ``SW=F(j-1,i-1)``, ``SE=F(j-1,i)``, ``NE=F(j,i)``, ``NW=F(j,i-1)``;
    nodes are the F-points referenced by at least one kept element.

    Global ORCA (``layout``): the rasters are in the CANONICAL form of
    :class:`CgridLayout` (E-W halo columns, north halo row, every copy filled
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
    layout : CgridLayout or None
        The raster layout (``readers.nemo.detect_cgrid_layout``); the arrays
        must have ``layout.shape``.  ``None`` = a closed regional raster (its
        edges are walls or open water: nothing is merged).
    ring : bool
        Append ONE ring of dry cells (8-neighbourhood of the wet region,
        across the seam and the fold) as zero-velocity elements after the wet
        ones (default; land = zero velocity, as in Parcels).  A stage or end
        point in the ring reads zero velocity (the particle stays ALIVE);
        beyond the ring a stage still fails.  ``ring=False`` = wet cells only.  Ring cells
        that are non-convex or sit at a fold pivot are dropped (counted in
        ``_cgrid_ring_dropped``).

    nlev2d, zw, area, e3b, e3u0, e3v0 : array_like or None
        3-D (z-level, partial bottom cells): ``nlev2d (ny, nx)`` wet levels per
        column, ``zw (K+1,)`` the reference layer interfaces (positive down),
        ``area (ny, nx)`` the T-cell area ``e1t*e2t`` and ``e3b (ny, nx)`` the
        thickness of each column's bottom wet cell (default: the reference
        layer = full cells); the floor is ``zw[nlev-1] + e3b`` (also the node
        depth when ``depth2d`` is None).  ``e3u0`` / ``e3v0 (K, ny, nx)`` are
        the model's face thicknesses (default: the reference layer); only a
        face's bottom level differs from it (zps), so the mesh keeps one level
        index and one thickness per face (``_cgrid_kbu/_e3u``, ``_kbv/_e3v``,
        ``-1`` = no partial level).  ``mesh.sigma`` becomes ``zw`` and
        ``_cgrid_nlev`` the wet levels (ring cells 0; every level at or below a
        column's floor is zero-velocity land).

    Returns
    -------
    Mesh
        Quad mesh; stashes ``_structured_shape=(ny, nx)``, ``_cgrid=True``,
        ``_cgrid_layout``, ``_cgrid_elem_ji (E, 2)`` (the T-cell ``(j, i)`` of
        each element) and ``_cgrid_corners (E, 4)`` (node ids, SW/SE/NE/NW —
        the interpolator reads this, not ``elem_nodes``).  ``elem_nodes`` is
        the same quad (containment is winding-agnostic).  Also
        ``_cgrid_n_wet`` (the first ``n_wet`` elements are wet cells, the
        rest ring cells), ``_cgrid_nlev (E,) int32`` (1 wet / 0 ring).
    """
    lon2d = np.asarray(glamf, dtype=np.float64)
    lat2d = np.asarray(gphif, dtype=np.float64)
    if lon2d.ndim != 2 or lon2d.shape != lat2d.shape:
        raise ValueError(
            f"glamf/gphif must be 2-D (y, x) of equal shape, got "
            f"{lon2d.shape} and {lat2d.shape}")
    ny, nx = lon2d.shape
    if layout is None:
        layout = CgridLayout((ny, nx))
    if layout.shape != (ny, nx):
        raise ValueError(f"the rasters are {(ny, nx)}, the layout's canonical shape is {layout.shape}")
    wet = np.asarray(tmask2d) > 0.5
    if wet.shape != (ny, nx):
        raise ValueError(f"tmask shape {wet.shape} != grid {(ny, nx)}")
    # a cell is a canonical T point with an F corner south and west of it
    jj, ii = np.mgrid[0:ny, 0:nx]
    cj, ci, _ = cgrid_canonical("T", jj, ii, layout)
    piv = fold_pivot_cells(layout)
    own = (cj == jj) & (ci == ii) & (jj >= 1) & (ii >= 1)
    if (wet & own & piv).any():
        raise ValueError(
            "cgrid mesh: a wet T-cell sits at a north-fold pivot "
            f"(cells {np.argwhere(wet & piv).tolist()[:4]}); the fold's grid poles "
            "must be on land")
    wet = wet & own
    if not wet.any():
        raise ValueError("cgrid mesh has no wet T-cells")
    flat_lon = lon2d.reshape(-1)
    flat_lat = lat2d.reshape(-1)

    def quads(j, i):
        """Canonical flat F indices of the corners SW, SE, NE, NW of cells (j, i)."""
        out = []
        for dj, di in ((-1, -1), (-1, 0), (0, 0), (0, -1)):
            fj, fi, _ = cgrid_canonical("F", j + dj, i + di, layout)
            out.append(fj * nx + fi)
        return np.stack(out, axis=1)

    ej, ei = np.nonzero(wet)
    quad_raster = quads(ej, ei)
    n_wet = int(ej.size)
    n_ring = 0
    ring_dropped = {"pivot": 0, "nonconvex": 0}
    if ring:
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
        ring_dropped["pivot"] = int((~keep).sum())
        rj, ri = rj[keep], ri[keep]
        rq = quads(rj, ri)
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
            f"cgrid mesh: {len(pairs)} pairs of mesh nodes coincide (e.g. "
            f"({nodes_lon[a_]:.4f}, {nodes_lat[a_]:.4f})): the layout does not merge them "
            "(a crop through the fold that is not full width?)")
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
    mesh._cgrid_layout = layout
    if partial_cells not in ("shrunk", "porous"):
        raise ValueError(f"partial_cells must be 'shrunk' or 'porous', got {partial_cells!r}")
    mesh._cgrid_partial = partial_cells
    mesh._cgrid_elem_ji = np.stack([ej, ei], axis=1).astype(np.int64)
    mesh._cgrid_corners = corners
    mesh._cgrid_n_wet = n_wet
    nl = (np.ones(n_wet) if nlev2d is None
          else np.asarray(nlev2d)[ej[:n_wet], ei[:n_wet]])
    mesh._cgrid_nlev = np.r_[nl, np.zeros(n_ring)].astype(np.int32)
    mesh._cgrid_nlev_raster = np.zeros((ny, nx), dtype=np.int32)
    mesh._cgrid_nlev_raster[ej[:n_wet], ei[:n_wet]] = nl
    mesh._cgrid_zw = np.empty(0) if zw is None else np.asarray(zw, np.float64)
    mesh._cgrid_area = (np.empty(0) if area is None
                        else np.ascontiguousarray(area, np.float64).reshape(-1))
    mesh._cgrid_e3b = np.empty(0)
    for nm in ("u", "v"):
        setattr(mesh, f"_cgrid_kb{nm}", np.empty(0, np.int32))
        setattr(mesh, f"_cgrid_e3{nm}", np.empty(0))
    if zw is not None:
        mesh.sigma = mesh._cgrid_zw
        mesh._cgrid_e3b = np.ascontiguousarray(eb).reshape(-1)
        for nm, e3 in (("u", e3u0), ("v", e3v0)):
            kb, th = _partial_level(mesh._cgrid_zw, e3, (ny, nx))
            setattr(mesh, f"_cgrid_kb{nm}", kb)
            setattr(mesh, f"_cgrid_e3{nm}", th)
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


def _partial_level(zw, e3, shape):
    """Compact form of a zps thickness field ``e3 (K, ny, nx)``: per column the
    first level whose thickness differs from the reference layer and that
    thickness -> flat ``(kb int32, th)``; ``kb = -1`` where none does (also
    for ``e3 is None``: pure reference layers).  Only a face's bottom wet level
    can differ (zps; the reader checks it); levels below it are land."""
    n = shape[0] * shape[1]
    if e3 is None:
        return np.full(n, -1, np.int32), np.zeros(n)
    e3 = np.asarray(e3, np.float64)
    ref = np.diff(zw)[:, None, None]
    diff = np.abs(e3 - ref) > 1e-4 * ref
    kb = np.where(diff.any(0), diff.argmax(0), -1)
    th = np.take_along_axis(e3, np.maximum(kb, 0)[None], 0)[0]
    return kb.astype(np.int32).reshape(-1), np.where(kb >= 0, th, 0.0).reshape(-1)


def cgrid_thickness(zw, kb, e3):
    """Thickness field ``(K, ny, nx)``: the reference layers, level ``kb[j, i]``
    replaced by ``e3[j, i]`` where ``kb >= 0`` (cells: ``kb = nlev - 1`` and the
    bottom thickness; faces: the mesh's ``_cgrid_kbu/_e3u`` or ``_kbv/_e3v``)."""
    K = len(zw) - 1
    out = np.broadcast_to(np.diff(zw)[:, None, None], (K,) + kb.shape).copy()
    jj, ii = np.nonzero(kb >= 0)
    out[kb[jj, ii], jj, ii] = e3[jj, ii]
    return out


def cgrid_w_from_continuity(u, v, e2u, e1v, e3u, e3v, area, nlev):
    """Vertical velocity (positive down) from the continuity of the face
    fluxes ``u*e2u*e3u`` and ``v*e1v*e3v`` (``e3u``/``e3v`` ``(K, ny, nx)``
    from :func:`cgrid_thickness`),
    integrated upward from ``w = 0`` at the sea floor.

    ``u``, ``v`` ``(K, ny, nx)`` (land 0); ``nlev (ny, nx)`` wet levels.
    Returns ``(K+1, ny, nx)``: the discrete field is exactly non-divergent in
    every cell (surface ``w`` is the residual).
    """
    fu = u * e2u * e3u
    fv = v * e1v * e3v
    H = fu + fv
    H[:, :, 1:] -= fu[:, :, :-1]
    H[:, 1:, :] -= fv[:, :-1, :]
    H *= np.arange(u.shape[0])[:, None, None] < nlev
    w = np.zeros((u.shape[0] + 1,) + u.shape[1:])
    w[:-1] = np.cumsum(H[::-1], axis=0)[::-1] / area
    return w


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


__all__ = ["build_structured_mesh", "build_cgrid_mesh", "cgrid_w_from_continuity",
           "cgrid_thickness", "c_grid_to_t", "field_to_nodes"]
