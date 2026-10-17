"""HEALPix grid -> kelp unstructured Mesh (dual-mesh builder).

The HEALPix dual-mesh construction (docs/healpix.md §1): map the single
data location HEALPix provides (pixel centres) onto the node/element
structure the tracker uses:

- **nodes = wet pixel centres** (``healpy.pix2ang``) — u/v/eta/scalars are
  then node-based natively (``uv_layout="node"``, run with
  ``uv_interp: barycentric``);
- **elements = dual cells of the pixel-corner points**: every corner point
  of the HEALPix tessellation is shared by exactly 4 pixels (3 at the 8
  special vertices of the base tessellation); the polygon of the incident
  pixel CENTRES around one corner is one element (quad, or triangle at the
  special corners), kept only when ALL incident pixels are wet — a dry
  pixel drops the corner cell, which is exactly the coastline rule of
  ``readers/structured.py``.  Wet pixels referenced by no element (isolated
  cells) are dropped like structured.py orphans: they would be dead-end
  nodes producing false OUTSIDE verdicts.

Corner grouping is **integer-exact** — no float corner dedup, no
tangent-plane angle sort.  From ``healpy.get_all_neighbours(nside, p,
nest=True)`` (slot order SW, W, NW, N, NE, E, SE, S) the four corner cells
of the rhombic pixel ``p`` are::

    N corner: (p, NE, N, NW)      E corner: (p, SE, E, NE)
    S corner: (p, SW, S, SE)      W corner: (p, NW, W, SW)

each listed in cyclic order around the corner point.  The edge-sharing
neighbours of a HEALPix pixel are the diagonal slots SW/NW/NE/SE; the
compass slots W/N/E/S touch ``p`` only at a corner and are ``-1`` exactly
at the 8 valence-3 special vertices (healpy contract) — an absent diagonal
turns that corner cell into a triangle, a *dry* diagonal drops it.  Every
incident wet pixel generates the same cell; only the copy whose owner
(first member) is the smallest member id is kept — exact dedup, and the
final (owner, corner) sort makes the mesh independent of ``block_size``.
Seam- and pole-free by construction: healpy resolves all face-boundary
adjacency internally, so no per-face index bookkeeping exists here at all.

The builder is data-source-agnostic — plain Zarr, kerchunk reference
stores and NetCDF healpix products all reduce to ``nside + wet mask``.
The ``@register("healpix")`` reader below feeds it (whole stores and
regional extracts, 2-D and 3-D, em/m1 Kh on pixels, ``mesh.rotation``).

Known polar limitations of the GEOGRAPHIC frame (docs/healpix.md §3 —
solved by ``mesh.rotation``, shipped 2026-08-16, which moves both
pathologies onto the rotated poles / dry land; without rotation avoid
releasing or tracking within a few degrees of the poles, or crop them
out via wet_mask):

1. **Geometric containment cap**: the mesh covers the poles, but the
   lon/lat containment chart cannot accept points poleward of the first
   pixel ring (the polar cap quad's bbox tops out at its vertices'
   latitude ``arcsin(1 - 1/(3*nside^2))``) — a ~half-pixel cap: 84.15deg
   at nside 8, 89.91deg at nside 512.  Fail-safe: such points read as
   OUTSIDE, never as a wrong position.
2. **Cold-anchor search degradation**: the grid hash bins by raw lon/lat
   degrees, so near the poles (ring longitude gaps of 90deg/ring) a
   from-scratch nearest-node query can anchor far from the true nearest
   — measured at nside 512: false-OUTSIDE verdicts for ~6% of random
   points at |lat| 85deg, growing toward the cap.  Empty-window misses
   (formerly ``-1`` anchors) fall back to the exact KD-tree inside
   ``find_nearest_nodes``.  Affects SEEDING and global re-searches only:
   per-step tracking updates anchors topologically (warm anchors), so
   advected particles are unaffected until the geometric cap itself.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

from ..core.geometry import DEG2RAD, wrap_lon
from ..core.mesh import (Mesh, build_node_to_elems, build_search,
                         find_nonconvex_quads, validate_quad_convexity)
from . import register, _detect_engine, resolve_w_sign


def _require_healpy():
    """Import healpy lazily so kelp stays importable without it."""
    try:
        import healpy
    except ImportError as exc:  # pragma: no cover - env-dependent message
        raise ImportError(
            "healpy is required for HEALPix mesh support; install it into "
            "the run environment (e.g. `conda install -c conda-forge "
            "healpy`)"
        ) from exc
    return healpy


def build_healpix_dual_mesh(nside: int, *,
                            order: str = "nest",
                            wet_mask: Optional[np.ndarray] = None,
                            depth: Optional[np.ndarray] = None,
                            z_levels: Optional[np.ndarray] = None,
                            name: str = "healpix",
                            r_earth: Optional[float] = None,
                            block_size: int = 1 << 20) -> Mesh:
    """Build the dual :class:`Mesh` (centres = nodes) for a HEALPix grid.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter (power of 2); ``npix = 12 * nside**2``.
    order : str
        Index ordering of the INPUT arrays (``wet_mask``/``depth``):
        ``"nest"`` or ``"ring"``.  Internally everything is NEST; ring
        inputs are reordered on entry and all pixel ids stashed on the
        mesh are NEST ids.
    wet_mask : array_like (npix,) of bool or None
        True = ocean pixel, False = land (dropped).  None = all wet
        (water world — the synthetic-test configuration).
    depth : array_like (npix,) or None
        Bottom depth per pixel (m, sign-agnostic -> abs); 0 when None.
    z_levels : array_like or None
        Vertical interface depths (m) for 3D; empty (2D) when None.
    name : str
    r_earth : float or None
        Earth radius (m) for mesh-derived areas; ``Mesh`` default
        (``R_EARTH_STANDARD``) when None.  Enters only the spherical-
        excess element areas, which downstream are consumed as ratios.
    block_size : int
        Pixels per neighbour-query block — bounds the memory of the
        neighbour/candidate pass only (multi-GB at zoom 10+ otherwise).
        The element-centre and convexity passes are NOT chunked and
        dominate peak memory at high zoom (measured: nside 512 = 3.6 GB,
        nside 1024 = 12.8 GB peak); zoom >= 11 needs a regional wet_mask.
        The result is byte-identical for any value.

    Returns
    -------
    Mesh
        Mixed tri/quad mesh (``elem_nodes`` shape ``(E, 4)``, triangles
        padded with ``-1``); carries ``_healpix_nside`` and
        ``_healpix_node_pix`` (``(N,)`` int64 NEST pixel id per node) for
        the reader-side data compaction, plus ``metadata`` counts.
        Search structures are built; areas/adjacency stay lazy.
    """
    hp = _require_healpy()
    nside = int(nside)
    if not hp.isnsideok(nside, nest=True):
        raise ValueError(
            f"invalid HEALPix nside {nside}: must be a positive power of 2")
    npix = int(hp.nside2npix(nside))
    if npix > np.iinfo(np.int32).max:
        # Mesh connectivity (elem_nodes, node_to_elems, element ids) is
        # int32 across the whole machinery — refuse to build a mesh whose
        # node/element counts could overflow it (nside >= 16384).
        raise ValueError(
            f"nside {nside} gives npix={npix:,}, exceeding the int32 "
            "connectivity contract of the Mesh machinery")

    if order not in ("nest", "ring"):
        raise ValueError(f"order must be 'nest' or 'ring', got {order!r}")

    if wet_mask is None:
        wet = np.ones(npix, dtype=bool)
    else:
        wet = np.asarray(wet_mask, dtype=bool)
        if wet.shape != (npix,):
            raise ValueError(
                f"wet_mask shape {wet.shape} != ({npix},) for nside {nside}")
    dep = None
    if depth is not None:
        dep = np.abs(np.asarray(depth, dtype=np.float64))
        if dep.shape != (npix,):
            raise ValueError(
                f"depth shape {dep.shape} != ({npix},) for nside {nside}")
    if order == "ring":
        # value_nest[p] = value_ring[ring_index_of(p)]
        ring_of_nest = hp.nest2ring(nside, np.arange(npix, dtype=np.int64))
        wet = wet[ring_of_nest]
        if dep is not None:
            dep = dep[ring_of_nest]

    wet_pix = np.flatnonzero(wet).astype(np.int64)
    if wet_pix.size == 0:
        raise ValueError("HEALPix wet mask has no ocean pixels")

    # Sentinel slot: fancy-indexing wet_ext with a healpy -1 ("no such
    # neighbour") lands on the appended False — absent reads as dry.
    wet_ext = np.concatenate([wet, np.zeros(1, dtype=bool)])

    kept_rows = []
    kept_keys = []
    bs = max(int(block_size), 1)
    for start in range(0, wet_pix.size, bs):
        blk = wet_pix[start:start + bs]
        nb = hp.get_all_neighbours(nside, blk, nest=True).astype(np.int64)
        n_sw, n_w, n_nw, n_n, n_ne, n_e, n_se, n_s = nb
        # Four corner cells per pixel, members in cyclic order around the
        # corner point: (owner, edge-neighbour, diagonal, edge-neighbour).
        cand = np.concatenate([
            np.stack([blk, n_ne, n_n, n_nw], axis=1),   # k=0: north corner
            np.stack([blk, n_se, n_e, n_ne], axis=1),   # k=1: east corner
            np.stack([blk, n_sw, n_s, n_se], axis=1),   # k=2: south corner
            np.stack([blk, n_nw, n_w, n_sw], axis=1),   # k=3: west corner
        ], axis=0)
        k = np.repeat(np.arange(4, dtype=np.int64), blk.size)
        edge1, diag, edge2 = cand[:, 1], cand[:, 2], cand[:, 3]
        # All incident pixels wet; an ABSENT diagonal (valence-3 corner)
        # is fine (triangle), a DRY one drops the cell.
        keep = wet_ext[edge1] & wet_ext[edge2] & (wet_ext[diag] | (diag < 0))
        # Exact dedup: every incident wet pixel generates this cell once;
        # keep only the copy owned by the smallest member id.
        members = np.where(cand >= 0, cand, np.iinfo(np.int64).max)
        keep &= cand[:, 0] == members.min(axis=1)
        kept_rows.append(cand[keep])
        kept_keys.append((cand[keep, 0] << 2) | k[keep])

    rows = np.concatenate(kept_rows, axis=0)
    if rows.shape[0] == 0:
        raise ValueError(
            "HEALPix dual mesh has no elements — every wet pixel is "
            "isolated (no corner has all incident pixels wet)")
    # Canonical element order (owner pixel, corner index): the mesh is
    # byte-identical for any block_size.
    keys = np.concatenate(kept_keys)
    rows = rows[np.argsort(keys, kind="stable")]

    # Valence-3 corner: absent diagonal -> triangle (drop the -1 slot,
    # cyclic order preserved), stored FESOM-C style as [a, b, c, -1].
    tri = rows[:, 2] < 0
    rows[tri, 2] = rows[tri, 3]
    rows[tri, 3] = -1
    n_triangles = int(tri.sum())

    # Paranoia guards, O(E): a healpy adjacency inconsistency would
    # otherwise build a silently-broken mesh (the invariant tests catch
    # this too, but only for the meshes they run on).
    for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        dup = (rows[:, i] == rows[:, j]) & (rows[:, i] >= 0)
        if dup.any():
            raise ValueError(
                f"degenerate HEALPix corner cell at nside {nside}: pixel "
                f"repeated within element (first: members "
                f"{rows[np.flatnonzero(dup)[0]].tolist()})")
    srt = np.sort(np.where(rows >= 0, rows, np.iinfo(np.int64).max), axis=1)
    ss = srt[np.lexsort(srt.T)]
    if ss.shape[0] > 1 and bool(np.all(ss[1:] == ss[:-1], axis=1).any()):
        raise ValueError(
            f"duplicate HEALPix dual cell at nside {nside}: two corners "
            "produced the same member set")
    if wet_mask is None and rows.shape[0] != npix + 2:
        # Euler's formula fixes the all-wet element count exactly (the 8
        # valence-3 corners force E = npix + 2); a shortfall means a corner
        # cell was silently dropped (healpy adjacency inconsistency) —
        # extends the invariant the tests pin at small nside to every
        # all-wet build.
        raise ValueError(
            f"all-wet HEALPix dual mesh at nside {nside} has "
            f"{rows.shape[0]} elements, expected npix + 2 = {npix + 2}")

    # --- nodes = wet pixels referenced by >= 1 element (orphans dropped) --
    node_pix = np.unique(rows[rows >= 0])            # sorted NEST ids, int64
    n_nodes = int(node_pix.size)
    remap = np.full(npix, -1, dtype=np.int32)
    remap[node_pix] = np.arange(n_nodes, dtype=np.int32)
    elem_nodes = np.where(rows >= 0,
                          remap[np.clip(rows, 0, npix - 1)],
                          np.int32(-1)).astype(np.int32)

    lon_pix, lat_pix = hp.pix2ang(nside, node_pix, nest=True, lonlat=True)
    nodes_lon = np.ascontiguousarray(
        wrap_lon(np.asarray(lon_pix, dtype=np.float64)))
    nodes_lat = np.ascontiguousarray(np.asarray(lat_pix, dtype=np.float64))
    node_depth = (np.ascontiguousarray(dep[node_pix]) if dep is not None
                  else np.zeros(n_nodes, dtype=np.float64))

    # Element centres: normalized mean of the member unit vectors —
    # dateline- and pole-safe (the polar cap cell gets the pole itself).
    lon_r = nodes_lon * DEG2RAD
    lat_r = nodes_lat * DEG2RAD
    cos_lat = np.cos(lat_r)
    node_xyz = np.stack((cos_lat * np.cos(lon_r),
                         cos_lat * np.sin(lon_r),
                         np.sin(lat_r)), axis=1)      # (N, 3)
    valid = elem_nodes >= 0
    V = node_xyz[np.where(valid, elem_nodes, 0)] * valid[:, :, None]
    C = V.sum(axis=1)
    nrm = np.linalg.norm(C, axis=1, keepdims=True)
    if not np.all(nrm > 1e-9):
        # unreachable for compact HEALPix corner cells (members span at
        # most a couple of pixel radii) — guards the normalization against
        # ever emitting NaN centres silently
        raise ValueError("degenerate HEALPix element centre (zero mean "
                         "vector) — corner cell members are not compact")
    C /= nrm
    elem_center_lon = np.ascontiguousarray(
        wrap_lon(np.degrees(np.arctan2(C[:, 1], C[:, 0]))))
    elem_center_lat = np.ascontiguousarray(
        np.degrees(np.arcsin(np.clip(C[:, 2], -1.0, 1.0))))

    # Load-time mesh contract (guard-free Wachspress weights downstream):
    # near-corner distorted quads must stay convex — fail loud otherwise
    # (escape hatch, if a real product ever trips this: split into tris).
    validate_quad_convexity(elem_nodes, nodes_lon, nodes_lat,
                            source=f"healpix dual mesh '{name}' "
                                   f"(nside={nside})")

    n2e, n2e_count = build_node_to_elems(n_nodes, elem_nodes)
    sigma = (np.abs(np.asarray(z_levels, dtype=np.float64))
             if z_levels is not None else np.array([], dtype=np.float64))
    kw = {} if r_earth is None else {"r_earth": r_earth}
    mesh = Mesh(
        name=name, nodes_lon=nodes_lon, nodes_lat=nodes_lat,
        depth=node_depth, elem_nodes=elem_nodes,
        elem_center_lon=elem_center_lon, elem_center_lat=elem_center_lat,
        node_to_elems=n2e, node_to_elems_count=n2e_count,
        sigma=sigma, vertical_coord="zlevel",
        metadata={"n_triangles": n_triangles,
                  "n_quads": int(rows.shape[0]) - n_triangles,
                  "healpix_nside": nside,
                  "healpix_npix": npix,
                  "healpix_order": "nest"},
        **kw)
    # Stash the pixel mapping for the reader-side data compaction
    # (node i carries pixel _healpix_node_pix[i], NEST at _healpix_nside).
    mesh._healpix_nside = nside
    mesh._healpix_node_pix = node_pix
    build_search(mesh)
    return mesh


#: |lat| above which a corner node makes its quad chart-degenerate in the
#: geographic frame (the pole point has no single longitude; the 8
#: pole-touching pixels can only be tracked under ``mesh.rotation``).
_POLE_CORNER_LAT = 89.9


def build_healpix_pixel_mesh(nside: int, *,
                             order: str = "nest",
                             wet_mask: Optional[np.ndarray] = None,
                             depth: Optional[np.ndarray] = None,
                             z_levels: Optional[np.ndarray] = None,
                             name: str = "healpix_pixel",
                             r_earth: Optional[float] = None) -> Mesh:
    """Build the pixels-as-elements :class:`Mesh` (docs/healpix.md §1b).

    Nodes = deduplicated pixel CORNERS, elements = the wet pixels
    themselves (always 4-vertex quads), element centres = the exact pixel
    centres — so pixel-centre data (u/v) rides the ELEMENT layout and the
    fused least-squares kernels apply (``uv_interp: leastsq`` /
    ``edge_leastsq``).  Corner identity is integer-exact: a corner is the
    sorted set of its (up to 4) incident pixel ids from
    ``healpy.get_all_neighbours`` — no floating-point deduplication.
    Unlike the dual mesh, COASTAL corners survive (a corner adjacent to
    land is still a geometric vertex of its wet pixel).

    Same input conventions as :func:`build_healpix_dual_mesh`
    (``order``/``wet_mask``/``depth``/``z_levels``/``r_earth``).  In the
    geographic frame the 8 pole-touching pixels are chart-degenerate (the
    pole corner has no single longitude): they are kept, listed in
    ``mesh.metadata['healpix_polar_degenerate']`` and warned about once —
    use ``mesh.rotation`` to track near the poles (the convexity check
    projects through 3-D unit vectors and is rotation-invariant, so no
    re-validation is needed after rotation).
    """
    hp = _require_healpy()
    nside = int(nside)                    # numpy ints overflow the guard
    if not hp.isnsideok(nside, nest=True):
        raise ValueError(f"invalid HEALPix nside {nside} (power of 2 > 0)")
    npix = 12 * nside * nside
    if npix + 8 > np.iinfo(np.int32).max:
        raise ValueError(
            f"nside {nside} gives npix={npix:,}, exceeding the int32 "
            "connectivity contract of the Mesh machinery")
    if order not in ("nest", "ring"):
        raise ValueError(f"order must be 'nest' or 'ring', got {order!r}")

    if wet_mask is None:
        wet = np.ones(npix, dtype=bool)
    else:
        wet = np.asarray(wet_mask, dtype=bool)
        if wet.shape != (npix,):
            raise ValueError(
                f"wet_mask shape {wet.shape} != ({npix},) for nside {nside}")
    dep = None
    if depth is not None:
        dep = np.abs(np.asarray(depth, dtype=np.float64))
        if dep.shape != (npix,):
            raise ValueError(
                f"depth shape {dep.shape} != ({npix},) for nside {nside}")
    if order == "ring":
        ring_of_nest = hp.nest2ring(nside, np.arange(npix, dtype=np.int64))
        wet = wet[ring_of_nest]
        if dep is not None:
            dep = dep[ring_of_nest]

    wet_pix = np.flatnonzero(wet).astype(np.int64)
    if wet_pix.size == 0:
        raise ValueError("HEALPix wet mask has no ocean pixels")
    n_wet = int(wet_pix.size)

    # --- corner member sets, cyclic per pixel: N, E, S, W ----------------
    nb = hp.get_all_neighbours(nside, wet_pix, nest=True).astype(np.int64)
    n_sw, n_w, n_nw, n_n, n_ne, n_e, n_se, n_s = nb
    cand = np.concatenate([
        np.stack([wet_pix, n_ne, n_n, n_nw], axis=1),   # k=0: north
        np.stack([wet_pix, n_se, n_e, n_ne], axis=1),   # k=1: east
        np.stack([wet_pix, n_sw, n_s, n_se], axis=1),   # k=2: south
        np.stack([wet_pix, n_nw, n_w, n_sw], axis=1),   # k=3: west
    ], axis=0)                                          # (4*n_wet, 4)

    # Integer-exact identity: sorted member multiset (valence-3 corners
    # carry one -1, identically from every incident pixel).
    key = np.sort(cand, axis=1)
    uniq, first_idx, inverse = np.unique(
        key, axis=0, return_index=True, return_inverse=True)
    n_nodes = int(uniq.shape[0])
    if n_nodes + 8 > np.iinfo(np.int32).max:
        raise ValueError("HEALPix pixel mesh node count exceeds int32")
    elem_nodes = np.ascontiguousarray(
        inverse.reshape(4, n_wet).T.astype(np.int32))   # cyclic N,E,S,W

    # --- corner coordinates from healpy.boundaries -----------------------
    # boundaries() returns 4 corner vectors per pixel in a fixed intrinsic
    # order; the slot->column permutation is derived EMPIRICALLY from the
    # member-set geometry and asserted identical across a sample (a healpy
    # convention change would fail loudly here, never mis-place corners).
    bnd = np.asarray(hp.boundaries(nside, wet_pix, step=1, nest=True))
    if bnd.ndim == 2:                                    # single-pixel form
        bnd = bnd[None, :, :]
    # approximate corner direction per (row, slot) = mean of member centres.
    # Sample SPREAD over the wet set plus representatives of every base
    # face present — the first-64 window covers a single face at
    # nside >= 8 and could hide a per-face convention (2026-08 review).
    smp = np.unique(np.linspace(0, n_wet - 1, num=min(64, n_wet),
                                dtype=np.int64))
    face = wet_pix // (nside * nside)
    for f in np.unique(face):
        smp = np.union1d(smp, np.flatnonzero(face == f)[:3])
    perm = None
    for i in smp:
        row_perm = np.empty(4, dtype=np.int64)
        for k in range(4):
            mem = cand[k * n_wet + i]
            mem = mem[mem >= 0]
            vx, vy, vz = hp.pix2vec(nside, mem, nest=True)
            d = np.array([np.sum(vx), np.sum(vy), np.sum(vz)])
            d /= np.linalg.norm(d)
            row_perm[k] = int(np.argmax(d @ bnd[i]))
        if sorted(row_perm.tolist()) != [0, 1, 2, 3]:
            raise ValueError(
                "healpy.boundaries corner matching is not a bijection "
                f"for pixel {int(wet_pix[i])} — unexpected convention")
        if perm is None:
            perm = row_perm
        elif not np.array_equal(perm, row_perm):
            raise ValueError(
                "healpy.boundaries corner order is not uniform across "
                "pixels — convention changed, refusing to build")
    rep_row = first_idx % n_wet
    rep_slot = first_idx // n_wet
    node_xyz = bnd[rep_row, :, perm[rep_slot]]           # (n_nodes, 3)
    nodes_lon = np.ascontiguousarray(wrap_lon(
        np.degrees(np.arctan2(node_xyz[:, 1], node_xyz[:, 0]))))
    nodes_lat = np.ascontiguousarray(np.degrees(np.arcsin(
        np.clip(node_xyz[:, 2] / np.linalg.norm(node_xyz, axis=1),
                -1.0, 1.0))))

    # cross-owner consistency: a sampled corner's vector must also appear
    # among ANOTHER incident wet pixel's boundary corners (healpy corners
    # are shared points; a mismatch means mis-matched slots).
    wet_row_of_pix = np.full(npix, -1, dtype=np.int64)
    wet_row_of_pix[wet_pix] = np.arange(n_wet)
    rng_idx = np.linspace(0, n_nodes - 1, num=min(200, n_nodes),
                          dtype=np.int64)
    for c in rng_idx:
        members = uniq[c]
        others = members[(members >= 0)
                         & (members != wet_pix[rep_row[c]])]
        others = others[wet[others]] if others.size else others
        if others.size == 0:
            continue
        j = wet_row_of_pix[others[0]]
        chord = np.linalg.norm(bnd[j].T - node_xyz[c], axis=1).min()
        if chord > 1e-9:
            raise ValueError(
                f"HEALPix pixel-mesh corner {int(c)} disagrees between "
                f"incident pixels (chord {chord:.2e}) — boundary corner "
                "matching is broken")

    # --- per-node depth: mean over incident WET pixels --------------------
    if dep is not None:
        node_depth = np.zeros(n_nodes, dtype=np.float64)
        cnt = np.zeros(n_nodes, dtype=np.int64)
        flat = elem_nodes.ravel()
        np.add.at(node_depth, flat, np.repeat(dep[wet_pix], 4))
        np.add.at(cnt, flat, 1)
        node_depth /= np.maximum(cnt, 1)
    else:
        node_depth = np.zeros(n_nodes, dtype=np.float64)

    # --- element centres = exact pixel centres (the data points) ----------
    lon_c, lat_c = hp.pix2ang(nside, wet_pix, nest=True, lonlat=True)
    elem_center_lon = np.ascontiguousarray(
        wrap_lon(np.asarray(lon_c, dtype=np.float64)))
    elem_center_lat = np.ascontiguousarray(
        np.asarray(lat_c, dtype=np.float64))

    # --- chart-degeneracy policy at the poles -----------------------------
    polar_node = np.abs(nodes_lat) > _POLE_CORNER_LAT
    elem_polar = polar_node[elem_nodes].any(axis=1)
    bad = find_nonconvex_quads(elem_nodes, nodes_lon, nodes_lat)
    bad_nonpolar = bad[~elem_polar[bad]] if bad.size else bad
    if bad_nonpolar.size:
        head = ", ".join(str(int(b)) for b in bad_nonpolar[:10])
        raise ValueError(
            f"healpix pixel mesh '{name}' (nside={nside}): "
            f"{bad_nonpolar.size} non-convex quad(s) away from the poles "
            f"(first: {head}) — mesh contract violated")
    if elem_polar.any():
        warnings.warn(
            f"healpix pixel mesh '{name}': {int(elem_polar.sum())} "
            "pole-touching pixel(s) are chart-degenerate in the "
            "geographic frame (the pole corner has no single longitude); "
            "containment and weights are unreliable there — use "
            "mesh.rotation to track near the poles.",
            RuntimeWarning, stacklevel=2)

    n2e, n2e_count = build_node_to_elems(n_nodes, elem_nodes)
    # Thin-corner diagnostic (2026-08 review): a corner with < 3 incident
    # wet pixels (isolated cells, 1-2-pixel-wide channels) gives the
    # default 'leastsq' interpolation a SINGULAR plane-fit stencil — a
    # particle whose nearest corner is thin dies as an interpolation
    # failure.  'edge_leastsq' stays well-posed there (mirror ghosts).
    n_thin = int((n2e_count < 3).sum())
    if n_thin:
        warnings.warn(
            f"healpix pixel mesh '{name}': {n_thin} corner node(s) have "
            "< 3 incident wet pixels (isolated cells / narrow channels) — "
            "uv_interp 'leastsq' has a singular stencil at those corners "
            "and kills particles there; use 'edge_leastsq' for "
            "coastal-heavy domains.",
            RuntimeWarning, stacklevel=2)
    sigma = (np.abs(np.asarray(z_levels, dtype=np.float64))
             if z_levels is not None else np.array([], dtype=np.float64))
    kw = {} if r_earth is None else {"r_earth": r_earth}
    mesh = Mesh(
        name=name, nodes_lon=nodes_lon, nodes_lat=nodes_lat,
        depth=node_depth, elem_nodes=elem_nodes,
        elem_center_lon=elem_center_lon, elem_center_lat=elem_center_lat,
        node_to_elems=n2e, node_to_elems_count=n2e_count,
        sigma=sigma, vertical_coord="zlevel",
        metadata={"n_triangles": 0,
                  "n_quads": n_wet,
                  "healpix_nside": nside,
                  "healpix_npix": npix,
                  "healpix_order": "nest",
                  "healpix_geometry": "pixel",
                  "healpix_elem_pix": wet_pix,
                  "healpix_thin_corners": n_thin,
                  "healpix_polar_degenerate":
                      np.flatnonzero(elem_polar).astype(np.int64)},
        **kw)
    mesh._healpix_nside = nside
    mesh._healpix_elem_pix = wet_pix
    build_search(mesh)
    return mesh


# ---------------------------------------------------------------------------
# Native HEALPix reader (Phase 2: 2-D surface, direct whole-store mode)
# ---------------------------------------------------------------------------
#
# Three store carriers (docs/healpix.md §2): plain Zarr, NetCDF, and
# kerchunk *reference* stores (``*.parq`` / ``*.parquet`` / ``*.json``) whose
# chunks may be GRIB messages (ng5-production) — opened via plain
# xarray + fsspec (+ a lazy ``import gribscan`` to register the
# ``gribscan.rawgrib`` numcodecs codec; needs ``dask``).  The ng5 recipe
# needs TWO stores per run — the 2-D store has no ocean u/v, so u/v come
# from the 3-D store (surface level) and ssh from the 2-D store: point
# ``data.ncinput`` at the 3-D store and ``data.ncinput_eta`` at the 2-D one
# (auto-derived by swapping a ``3D`` filename token to ``2D`` when unset).

_U_ALIASES = ("avg_uoe", "uo", "uos", "u")
_V_ALIASES = ("avg_von", "vo", "vos", "v")
_W_ALIASES = ("avg_wo", "wo", "w", "wvel")
_ETA_ALIASES = ("avg_zos", "zos", "ssh", "eta", "sossheig")
_DEPTH_ALIASES = ("deptho", "bathymetry", "bathy_meter")
#: extractor-written interface table (metres) — makes extracts fully
#: self-contained for 3-D (docs/healpix.md §4: mesh.levels is only needed when
#: pointing at a raw GRIB store directly)
_LEVELS_VAR = "level_interfaces_m"

#: level-block height for 3-D reads (`_compact_3d`): caps the transient
#: full-sphere float64 materialization at this many levels per read.
#: 10 divides our extractor/upsampler Zarr level chunking exactly, and
#: kerchunked GRIB stores are single-level-chunked — no read
#: amplification either way.
_LEVEL_BLOCK = 10
#: vertical dim names seen on healpix products (GRIB level axis, CMIP depth)
_LEVEL_DIMS = ("level", "lev", "depth", "deptht", "z", "height")

_KERCHUNK_SUFFIXES = {".parq", ".parquet", ".json"}

#: one-time-warning registry (per process), keyed by (kind, detail)
_WARNED: set = set()

_NA_SHIM_INSTALLED = False


def _warn_once(key, message):
    if key in _WARNED:
        return
    _WARNED.add(key)
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def _install_reference_na_shim():
    """Guard fsspec's ReferenceFileSystem against null reference rows.

    fsspec 2026.6.0 + pandas 3: an ABSENT chunk (null reference row in a
    kerchunk parquet store — e.g. the all-empty deepest u/v level of the
    ng5 stores) surfaces as pandas ``NA`` and crashes ``_cat_common``.
    Correct missing-chunk semantics: raise ``FileNotFoundError`` so zarr
    fills the chunk with ``_FillValue``.  Idempotent process-wide patch;
    drop once fixed upstream (verified against the Phase-0 probes).
    """
    global _NA_SHIM_INSTALLED
    if _NA_SHIM_INSTALLED:
        return
    import fsspec.implementations.reference as _ref

    orig = _ref.ReferenceFileSystem._cat_common
    if getattr(orig, "__wrapped__", None) is not None:
        # already wrapped (module reloaded / second kelp import path)
        _NA_SHIM_INSTALLED = True
        return

    def _cat_common_safe(self, path, start=None, end=None):
        part_or_url, s, e = orig(self, path, start=start, end=end)
        if not isinstance(part_or_url, (str, bytes)):
            raise FileNotFoundError(path)
        return part_or_url, s, e

    _cat_common_safe.__wrapped__ = orig
    _ref.ReferenceFileSystem._cat_common = _cat_common_safe
    _NA_SHIM_INSTALLED = True


def _is_kerchunk_path(path) -> bool:
    return Path(str(path)).suffix.lower() in _KERCHUNK_SUFFIXES


def _open_healpix_store(path):
    """Open one healpix store: kerchunk reference | plain Zarr | NetCDF."""
    import xarray as xr

    p = str(path)
    if _is_kerchunk_path(p):
        _install_reference_na_shim()
        try:
            import gribscan  # noqa: F401 — registers the rawgrib codec
        except ImportError:
            _warn_once(("gribscan", None),
                       "gribscan is not installed — kerchunk stores whose "
                       "chunks are GRIB messages (e.g. ng5-production) will "
                       "fail with an unknown-codec error; plain-chunk "
                       "kerchunk stores are unaffected.")
        import fsspec
        fs = fsspec.filesystem("reference", fo=p, remote_protocol="file",
                               lazy=True)
        return xr.open_dataset(fs.get_mapper(""), engine="zarr",
                               consolidated=False, chunks={})
    return xr.open_dataset(p, engine=_detect_engine(p))


def _first(ds, aliases: Sequence[str], cfg_name: Optional[str] = None):
    """Variable-name resolution: explicit override (must exist) else aliases.

    NEMO-reader pattern: a set-but-absent override raises ``KeyError`` — a
    typoed ``var_u`` must never silently fall back to a different product.
    """
    if cfg_name:
        if cfg_name in ds.variables:
            return cfg_name
        raise KeyError(
            f"healpix reader: configured variable {cfg_name!r} not found in "
            f"the store (present: {sorted(ds.data_vars)[:12]}…)")
    for nm in aliases:
        if nm in ds.variables:
            return nm
    return None


def _first_soft(ds, aliases: Sequence[str], cfg_name: Optional[str] = None):
    """Like :func:`_first`, but a set-but-absent override returns None.

    Needed where a variable may legitimately live in ANOTHER store of the
    run (the two-store eta recipe): ``mode.var_eta`` names a variable of
    the ETA store, so its absence from the main store must mean "keep
    looking", not KeyError.  The hard enforcement happens where the
    variable is finally required.
    """
    if cfg_name:
        return cfg_name if cfg_name in ds.variables else None
    for nm in aliases:
        if nm in ds.variables:
            return nm
    return None


def _find_crs(ds):
    """Return ``(nside, order)`` from healpix ``crs`` attrs, or (None, None)."""
    for name in list(ds.variables) + ["crs"]:
        if name not in ds.variables:
            continue
        attrs = ds[name].attrs
        if str(attrs.get("grid_mapping_name", "")).lower() != "healpix":
            continue
        nside = attrs.get("healpix_nside")
        if nside is None:
            continue
        order = str(attrs.get("healpix_order", "nest")).lower()
        return int(nside), order
    return None, None


def _pick_level(da, mode_cfg):
    """Select one vertical level of a (…, level, cell) variable (2-D mode).

    ``mode.sigma_layer`` is a 0-BASED POSITIONAL index into the level axis
    (the established FESOM-C semantics: 0 = first = surface — true for the
    ng5 GRIB axis and CF depth axes, which are surface-first).  Default =
    index 0.  Bounds-checked so a bad config fails with a clear message
    instead of an opaque IndexError deep in xarray.
    """
    for d in da.dims:
        if str(d).lower() in _LEVEL_DIMS:
            layer = getattr(mode_cfg, "sigma_layer", None) if mode_cfg else None
            k = int(layer) if layer is not None else 0
            n = int(da.sizes[d])
            if not (0 <= k < n):
                raise ValueError(
                    f"mode.sigma_layer={k} is out of range for the "
                    f"{d!r} axis of {da.name!r} (size {n}); it is a "
                    "0-based index, 0 = surface.")
            return da.isel({d: k})
    return da


def _resolve_levels(levels: Union[None, str, List[float]]):
    """``mesh.levels`` -> (K,) positive-down interface depths, or None.

    A string is a FESOM ``aux3d.out`` path: line 1 = K, then K interface
    depths (only that table is read — the node-bathymetry tail of a full
    aux3d is ignored).  A list is inline depths in metres.  Values are
    ``abs()``-ed and must be strictly increasing.
    """
    if levels is None:
        return None
    if isinstance(levels, str):
        with open(levels) as fh:
            k = int(fh.readline().split()[0])
            vals = [float(fh.readline().split()[0]) for _ in range(k)]
    else:
        vals = [float(v) for v in levels]
    arr = np.abs(np.asarray(vals, dtype=np.float64))
    if arr.size < 2 or not np.all(np.diff(arr) > 0):
        raise ValueError(
            "mesh.levels must give >= 2 strictly increasing interface "
            f"depths after abs(); got {arr[:5]}…")
    return arr


def _derive_eta_sibling(path) -> Optional[str]:
    """Derive the 2-D store sibling by swapping a ``3D`` filename token.

    ng5 layout: ``3D_daily_healpix512_ocean.parq`` ↔
    ``2D_daily_healpix512_ocean.parq``.  Basename-only swap (NEMO
    ``_sibling`` pattern); returns None when nothing exists.
    """
    p = Path(str(path))
    name = p.name
    for src, tgt in (("3D", "2D"), ("3d", "2d")):
        if src in name:
            cand = p.with_name(name.replace(src, tgt, 1))
            if cand.name != name and cand.exists():
                return str(cand)
    return None


def _bbox_wet_mask(hp, nside: int, order: str, bbox) -> np.ndarray:
    """Pixel-centre test against ``[W, E, S, N]`` (W > E = dateline crop)."""
    west, east, south, north = (float(b) for b in bbox)
    npix = hp.nside2npix(nside)
    lon, lat = hp.pix2ang(nside, np.arange(npix), nest=(order == "nest"),
                          lonlat=True)
    lat_ok = (lat >= south) & (lat <= north)
    if east - west >= 360.0:
        # a latitude-band crop like [-180, 180, S, N]: the naive modulo
        # arithmetic below would collapse it to the lon == W meridian
        return lat_ok
    lon = np.mod(lon, 360.0)
    w = np.mod(west, 360.0)
    e = np.mod(east, 360.0)
    if w == e:
        raise ValueError(
            f"mesh.crop_bbox W={west} and E={east} describe a zero-width "
            "longitude window (equal after wrapping); use a span >= 360 "
            "for all longitudes.")
    if w < e:
        in_lon = (lon >= w) & (lon <= e)
    else:                                   # crosses the dateline/seam
        in_lon = (lon >= w) | (lon <= e)
    return in_lon & lat_ok


def _validate_time_axes_match(ds_a, ds_b, what: str):
    """Both stores must share one time axis (count + exact timestamps)."""
    na = ds_a.sizes.get("time")
    nb = ds_b.sizes.get("time")
    if na != nb:
        raise ValueError(
            f"healpix reader: {what} has {nb} time records but the main "
            f"store has {na}; both stores must be on the same time axis so "
            "a timestep index refers to one physical time.")
    ta = ds_a["time"].values if "time" in ds_a.variables else None
    tb = ds_b["time"].values if "time" in ds_b.variables else None
    if (ta is None) != (tb is None):
        raise ValueError(
            f"healpix reader: {what} and the main store disagree on whether "
            "a time coordinate is present — their pairing cannot be "
            "verified.")
    if ta is not None:
        # Raw NUMERIC axes are only comparable when their encodings agree:
        # equal numbers with different units/calendar/epoch would pair
        # different physical times by index.  (Decoded datetime64/cftime
        # axes compare directly.)
        num_a = np.issubdtype(np.asarray(ta).dtype, np.number)
        num_b = np.issubdtype(np.asarray(tb).dtype, np.number)
        if num_a or num_b:
            enc = []
            for d in (ds_a, ds_b):
                t = d["time"]
                enc.append((t.attrs.get("units") or t.encoding.get("units"),
                            t.attrs.get("calendar")
                            or t.encoding.get("calendar")))
            if num_a != num_b or enc[0] != enc[1]:
                raise ValueError(
                    f"healpix reader: {what} and the main store carry "
                    "numeric time axes with different encodings "
                    f"({enc[0]} vs {enc[1]}) — their pairing cannot be "
                    "verified.")
        if not np.array_equal(ta, tb):
            raise ValueError(
                f"healpix reader: {what} timestamps differ from the main "
                "store's; both stores must be written on the same time "
                "axis.")


def _compact_to_nodes(da, mesh, name: str, mode_cfg=None) -> np.ndarray:
    """One (…, cell) DataArray at one timestep -> (N,) node values."""
    da = _pick_level(da, mode_cfg)
    full = np.asarray(da.values, dtype=np.float64).reshape(-1)
    ncells = getattr(mesh, "_healpix_data_ncells",
                     mesh.metadata["healpix_npix"])
    if full.size != ncells:
        raise ValueError(
            f"healpix reader: variable {name!r} has {full.size} cells but "
            f"the mesh was built for a {ncells}-cell store (nside "
            f"{mesh.metadata['healpix_nside']}) — the data store and "
            "mesh.meshpath disagree on the zoom level or on being the "
            "same extract.")
    return full[mesh._healpix_data_idx]


def _compact_3d(da, mesh, name: str, n_keep: int,
                check_prefix: bool = False) -> np.ndarray:
    """One (level, cell) DataArray at one timestep -> (n_keep, N) values.

    Slices the first ``n_keep`` entries of the level axis (found by name),
    orients the level axis FIRST (the kernels' contract), and compacts the
    cell axis to nodes.  ``check_prefix=True`` validates (once per
    variable name per process) that finite values form a PREFIX down every
    column — an interior NaN stripe is store corruption, not bathymetry,
    and must not be silently zeroed into fake ground.
    """
    lev_dim = next((d for d in da.dims if str(d).lower() in _LEVEL_DIMS),
                   None)
    if lev_dim is None:
        raise ValueError(
            f"healpix reader: 3-D variable {name!r} has no level dim "
            f"(dims {da.dims}; known level names {sorted(_LEVEL_DIMS)}).")
    if int(da.sizes[lev_dim]) < n_keep:
        raise ValueError(
            f"healpix reader: variable {name!r} has {da.sizes[lev_dim]} "
            f"levels but the mesh interfaces need {n_keep} — mesh.levels "
            "and the store's level axis disagree.")
    da = da.isel({lev_dim: slice(0, n_keep)})
    cell_dim = next(d for d in da.dims if d != lev_dim)
    da = da.transpose(lev_dim, cell_dim)
    ncells = getattr(mesh, "_healpix_data_ncells",
                     mesh.metadata["healpix_npix"])
    if int(da.sizes[cell_dim]) != ncells:
        raise ValueError(
            f"healpix reader: variable {name!r} has "
            f"{int(da.sizes[cell_dim])} cells but the mesh was built "
            f"for a {ncells}-cell store.")
    # np.take gathers directly in C order — the fancy form ``arr[:, idx]``
    # returns a FORTRAN-ordered array (the parallel whole-field time blends
    # gate on C-contiguity, so it silently cost ~10x the step at nside 256),
    # and ascontiguousarray on top of it would pay a second full copy.
    # take(mode='raise') matches fancy indexing for negative and
    # out-of-range indices (2026-08 review, measured 4x + half the
    # transient at nside 256).
    # The read is LEVEL-BLOCKED: materializing the whole (n_keep, ncells)
    # float64 sphere before the gather peaked at ~28 GB per variable on a
    # zoom-11 store (OOM on a 256-GB node even single-process); reading
    # _LEVEL_BLOCK levels at a time caps the transient at block/n_keep of
    # that, and the values are bit-identical — np.take per block equals
    # one np.take of the whole array, dtypes and operation order
    # unchanged.  A contiguous row slice of a C array is C-contiguous, so
    # ``out=`` writes in place with no extra copy.
    idx = mesh._healpix_data_idx
    # bounds validated ONCE here so the per-block take can use
    # mode="clip": with mode="raise" numpy buffers the result even when
    # ``out=`` is given (an extra (block, N) temp — the thing this loop
    # exists to avoid); with in-range indices "clip" is value-identical
    # and writes into ``out`` directly (review 2026-08).
    if idx.ndim != 1 or (idx.size and
                         (int(idx.min()) < 0 or int(idx.max()) >= ncells)):
        raise ValueError(
            f"healpix reader: node->cell index map is invalid for "
            f"variable {name!r} (ncells={ncells}) — mesh/store mismatch.")
    out = np.empty((n_keep, int(idx.size)), dtype=np.float64)
    for k0 in range(0, n_keep, _LEVEL_BLOCK):
        k1 = min(k0 + _LEVEL_BLOCK, n_keep)
        blk = np.asarray(
            da.isel({lev_dim: slice(k0, k1)}).values, dtype=np.float64)
        np.take(blk, idx, axis=1, out=out[k0:k1], mode="clip")
        del blk
    if check_prefix and ("prefix", name) not in _WARNED:
        # FIRST-READ structural gate (once per variable per process,
        # like the drift warning): scoped to the TRACKED node columns —
        # untracked land cells with stray values must not kill a run.
        _WARNED.add(("prefix", name))
        fin = np.isfinite(out)
        # a column violates the prefix shape iff a finite value appears
        # BELOW a non-finite one
        below_gap = (~fin[:-1] & fin[1:]).any(axis=0)
        if below_gap.any():
            raise ValueError(
                f"healpix reader: variable {name!r} has "
                f"{int(below_gap.sum())} node column(s) with finite data "
                "BELOW a non-finite level — not a finite-prefix "
                "bathymetry column; the store is corrupt or on an "
                "unexpected vertical layout.")
    return out


def _finite_prefix_per_cell(da, lev_dim) -> Optional[np.ndarray]:
    """Leading finite count down every column of a ``(level, cell)``
    DataArray at one record: int64 ``(ncells,)`` in store order, or None
    when the variable has other dims too.  Read ``_LEVEL_BLOCK`` levels at
    a time (like :func:`_compact_3d`), so no whole-sphere array is held."""
    if da.ndim != 2:
        return None
    cell_dim = next(d for d in da.dims if d != lev_dim)
    n_lev = int(da.sizes[lev_dim])
    run = np.ones(int(da.sizes[cell_dim]), dtype=bool)
    count = np.zeros(run.size, dtype=np.int64)
    for k0 in range(0, n_lev, _LEVEL_BLOCK):
        k1 = min(k0 + _LEVEL_BLOCK, n_lev)
        blk = np.isfinite(np.asarray(
            da.isel({lev_dim: slice(k0, k1)}).transpose(lev_dim,
                                                         cell_dim).values))
        for row in blk:
            run &= row
            count += run
        if not run.any():
            break
    return count


def _record_label(ds, timestep) -> str:
    """Names one record of ``ds`` in a refusal: its index in ``ds`` (a
    prefetch window counts from its own start) and its time stamp."""
    label = f"time index {timestep} of the dataset read"
    try:
        t = ds["time"].values[timestep]
    except Exception:
        return label
    return f"{label} (time {str(t)[:19]})"


def _ffill_columns(arr: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs DOWN each column of a (K, N) array (returns a copy).

    Extends the deepest finite value to the bottom — the clamp-consistent
    treatment for sampled scalars (T/S keep bottom-water properties below
    the variable's own last finite level; the ng5 u- and w-family bottoms
    disagree per cell, so sampled columns can be shorter than the mesh
    bathymetry).  Leading (surface) NaNs stay NaN.
    """
    mask = np.isfinite(arr)
    idx = np.where(mask, np.arange(arr.shape[0])[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    filled = arr[idx, np.arange(arr.shape[1])[None, :]]
    seen = np.maximum.accumulate(mask, axis=0)
    filled[~seen] = np.nan
    # Defence-in-depth: the broadcast fancy index above PRESERVES the input
    # order (C in -> C out), and _compact_3d now always feeds C — so this is
    # a no-op today; it pins the helper's own C-order contract for any
    # future F-ordered caller (parallel time blends gate on C-contiguity).
    return np.ascontiguousarray(filled)


def _rotate_uv_blocked(u, v, c_n, s_n):
    """Tangent rotation (u' = c·u − s·v, v' = s·u + c·v), level-blocked.

    The whole-array expression built two full-field float64 temporaries
    per component (~40 GB on a zoom-11 mesh — the read-path OOM peak
    after the blocked-read fix moved here); blocking caps the temporaries
    at ``_LEVEL_BLOCK`` rows while keeping the exact per-element
    operation order (multiply, multiply, subtract/add) — bit-identical.
    1-D (surface) fields take the plain expression: they are one row.
    """
    if u.ndim == 1:
        return c_n * u - s_n * v, s_n * u + c_n * v
    # outputs in the PROMOTED dtype, exactly like the literal expression
    # would produce — with float32 inputs np.empty_like would instead
    # round each product before the add/subtract (review 2026-08)
    dt = np.result_type(u, v, c_n, s_n)
    un = np.empty(u.shape, dtype=dt)
    vn = np.empty(v.shape, dtype=dt)
    for k0 in range(0, u.shape[0], _LEVEL_BLOCK):
        k1 = min(k0 + _LEVEL_BLOCK, u.shape[0])
        np.multiply(c_n, u[k0:k1], out=un[k0:k1])
        un[k0:k1] -= s_n * v[k0:k1]
        np.multiply(s_n, u[k0:k1], out=vn[k0:k1])
        vn[k0:k1] += c_n * v[k0:k1]
    return un, vn


@register("healpix")
class HealpixReader:
    """Reader for HEALPix-gridded ocean products (2-D surface, Phase 2)."""

    @staticmethod
    def load_mesh(mesh_cfg, mode_cfg=None) -> Mesh:
        """Build the dual mesh from the store at ``mesh_cfg.meshpath``.

        ``mode_cfg`` (threaded in by ``run_tracking``/the CLI when the
        reader's signature accepts it) supplies the ``var_u``/``var_v``
        overrides and the ``sigma_layer`` pick, so the wet mask is built
        from the SAME variable and level the run will track on.  Direct
        callers may omit it (alias resolution, surface level).
        """
        hp = _require_healpy()
        path = str(mesh_cfg.meshpath)
        ds = _open_healpix_store(path)
        try:
            nside, order = _find_crs(ds)
            if nside is None:
                zoom = getattr(mesh_cfg, "healpix_zoom", None)
                if zoom is None:
                    raise ValueError(
                        f"healpix store {path!r} carries no healpix crs "
                        "attrs (grid_mapping_name='healpix' + "
                        "healpix_nside); set mesh.healpix_zoom explicitly.")
                nside, order = 1 << int(zoom), "nest"
            if order not in ("nest", "ring"):
                raise ValueError(
                    f"healpix store {path!r}: unsupported ordering "
                    f"{order!r} (nest or ring).")

            cfg = mode_cfg
            u_name = _first(ds, _U_ALIASES,
                            getattr(cfg, "var_u", None) if cfg else None)
            v_name = _first(ds, _V_ALIASES,
                            getattr(cfg, "var_v", None) if cfg else None)
            if u_name is None or v_name is None:
                raise KeyError(
                    f"healpix store {path!r}: no u/v variable found "
                    f"(looked for {_U_ALIASES} / {_V_ALIASES}, plus any "
                    f"mode.var_u/var_v override; present: "
                    f"{sorted(ds.data_vars)[:12]}…).  Point mesh.meshpath "
                    "at the store that carries the velocities.")

            # wet mask = finite u AND v at t=0 on the TRACKED level
            # (mode.sigma_layer; surface by default) — masks are static in
            # time by product convention; families can differ per variable
            # (eta NaNs at u-wet coastal cells are zeroed at read).  Land
            # is assumed NaN-encoded (ng5/CMIP convention); a deptho-like
            # variable additionally refines the mask below.
            u0 = ds[u_name]
            v0 = ds[v_name]
            if "time" in u0.dims:
                u0 = u0.isel(time=0)
            if "time" in v0.dims:
                v0 = v0.isel(time=0)
            u0s = np.asarray(_pick_level(u0, cfg).values,
                             dtype=np.float64).reshape(-1)
            v0s = np.asarray(_pick_level(v0, cfg).values,
                             dtype=np.float64).reshape(-1)
            npix = int(hp.nside2npix(nside))

            # Regional EXTRACT (utils/extract_healpix_region.py): the cell
            # axis holds only a pixel subset; ``healpix_index`` carries the
            # source NEST ids (sorted).  Extracts are NEST-canonical.
            sub_idx = None
            if "healpix_index" in ds.variables:
                hi = ds["healpix_index"]
                # the cell axis is the LAST dim of the velocity variable
                # (the layout convention of every healpix product seen);
                # membership of ANY u-dim would accept healpix_index(time)
                if hi.ndim != 1 or hi.dims[0] != ds[u_name].dims[-1]:
                    raise ValueError(
                        f"healpix extract {path!r}: healpix_index must be "
                        f"1-D on the velocity CELL dim "
                        f"{ds[u_name].dims[-1]!r} (got dims {hi.dims}).")
                sub_idx = np.asarray(hi.values, dtype=np.int64).reshape(-1)
                if order != "nest":
                    raise ValueError(
                        f"healpix extract {path!r} declares ordering "
                        f"{order!r} — extracts are NEST-canonical by "
                        "construction; re-extract with the current tool.")
                if (sub_idx.size == 0
                        or sub_idx[0] < 0 or sub_idx[-1] >= npix
                        or (sub_idx.size > 1
                            and np.any(np.diff(sub_idx) <= 0))):
                    raise ValueError(
                        f"healpix extract {path!r}: healpix_index must be "
                        "non-empty, strictly increasing NEST ids within "
                        f"[0, npix={npix}).")
            ncells = int(sub_idx.size) if sub_idx is not None else npix
            if u0s.size != ncells:
                raise ValueError(
                    f"healpix store {path!r}: {u_name!r} has {u0s.size} "
                    f"cells, expected {ncells} (npix for nside {nside}, "
                    "or the healpix_index length of an extract) — the "
                    "store and the mesh zoom/extract disagree.")
            wet_cells = np.isfinite(u0s) & np.isfinite(v0s)
            if sub_idx is not None:
                wet = np.zeros(npix, dtype=bool)
                wet[sub_idx] = wet_cells
            else:
                wet = wet_cells

            bbox = getattr(mesh_cfg, "crop_bbox", None)
            if bbox is not None:
                # composes with an extract (a further crop of the sector)
                wet &= _bbox_wet_mask(hp, nside, order, bbox)

            # u/v data prefix for depth-linear runs (mode.uv_vertical): the
            # leading finite u count of every store column at t=0, read
            # BEFORE the bathymetry choice below — a deptho deeper than the
            # u prefix would otherwise point the depth-linear table at a
            # slot the reader zeroes.  Mapped to mesh-node order once the
            # mesh exists; declared for 3-D dual-geometry meshes only (a
            # mode_cfg with Fdrift3D: a 2-D load, or one without mode_cfg,
            # never pays this extra read of the first record; layer runs
            # never read the declaration).
            nfin_u = None
            lev_dim_u = next((d for d in u0.dims
                              if str(d).lower() in _LEVEL_DIMS), None)
            if (lev_dim_u is not None
                    and (getattr(mesh_cfg, "healpix_geometry", "dual")
                         or "dual") == "dual"
                    and cfg is not None and getattr(cfg, "Fdrift3D", False)):
                nfin_u = _finite_prefix_per_cell(u0, lev_dim_u)

            # bathymetry: a depth variable if the product carries one, else
            # deepest-finite-level x mesh.levels (finite-prefix columns),
            # else zeros (fine for 2-D; beaching depth criteria need one).
            levels = _resolve_levels(getattr(mesh_cfg, "levels", None))
            if _LEVELS_VAR in ds.variables:
                # an extract carries its interface table (self-contained)
                store_lev = np.abs(np.asarray(
                    ds[_LEVELS_VAR].values, dtype=np.float64)).reshape(-1)
                if levels is not None and not np.array_equal(levels,
                                                             store_lev):
                    raise ValueError(
                        f"healpix store {path!r} carries an interface "
                        f"table ({store_lev.size} levels) that conflicts "
                        "with mesh.levels — drop the config key or "
                        "re-extract.")
                levels = store_lev
            if (cfg is not None and getattr(cfg, "Fdrift3D", False)
                    and levels is None):
                raise ValueError(
                    "healpix 3-D mode needs the vertical interface depths: "
                    "set mesh.levels (a FESOM aux3d.out path or an inline "
                    "list of metres) — GRIB level axes carry only indices.")
            dep = None
            d_name = _first(ds, _DEPTH_ALIASES)
            if d_name is not None:
                dd = ds[d_name]
                if "time" in dd.dims:
                    dd = dd.isel(time=0)
                extra = [d for d in dd.dims
                         if int(dd.sizes[d]) != ncells
                         and int(dd.sizes[d]) != 1]
                if extra:
                    raise ValueError(
                        f"healpix store {path!r}: depth variable "
                        f"{d_name!r} carries non-cell dims {extra} — a "
                        "bathymetry must be a per-cell 1-D field.")
                dep = np.abs(np.nan_to_num(
                    np.asarray(dd.values, dtype=np.float64),
                    nan=0.0)).reshape(-1)
                if dep.size != ncells:
                    raise ValueError(
                        f"healpix store {path!r}: depth variable "
                        f"{d_name!r} has {dep.size} cells, expected "
                        f"{ncells}.")
                if sub_idx is not None:
                    full = np.zeros(npix, dtype=np.float64)
                    full[sub_idx] = dep
                    dep = full
                # docs/healpix.md §4: a real bathymetry refines the wet mask — a
                # sentinel-filled land (non-NaN u) is still excluded
                wet &= dep > 0.0
            elif levels is not None:
                lev_dim = next((d for d in u0.dims
                                if str(d).lower() in _LEVEL_DIMS), None)
                if lev_dim is not None:
                    # finite-prefix column length, reduced over the NAMED
                    # level dim (axis order varies between products)
                    nfin = np.isfinite(u0).sum(dim=lev_dim).values
                    nfin = np.minimum(nfin.reshape(-1), levels.size - 1)
                    dep = levels[nfin]
                    if sub_idx is not None:
                        full = np.zeros(npix, dtype=np.float64)
                        full[sub_idx] = dep
                        dep = full

            geometry = getattr(mesh_cfg, "healpix_geometry", "dual") or "dual"
            if geometry not in ("dual", "pixel"):
                raise ValueError(
                    f"mesh.healpix_geometry: unknown value {geometry!r} "
                    "(use 'dual' or 'pixel')")
            if geometry == "pixel":
                if mode_cfg is not None and getattr(mode_cfg, "Fdrift3D",
                                                    False):
                    raise NotImplementedError(
                        "mesh.healpix_geometry='pixel' is 2-D only in v1 "
                        "(w/eta/scalars live at pixel CENTRES; the "
                        "corner-node aggregation the 3-D kernels need is "
                        "staged) — use the default 'dual' geometry for "
                        "3-D runs.")
                mesh = build_healpix_pixel_mesh(
                    nside, order=order, wet_mask=wet, depth=dep,
                    z_levels=levels, name="healpix_pixel",
                    r_earth=getattr(mesh_cfg, "r_earth", None))
                carrier_pix = mesh._healpix_elem_pix
            else:
                mesh = build_healpix_dual_mesh(
                    nside, order=order, wet_mask=wet, depth=dep,
                    z_levels=levels, name="healpix",
                    r_earth=getattr(mesh_cfg, "r_earth", None))
                carrier_pix = mesh._healpix_node_pix
            # store-order index for data compaction (carrier -> position
            # on the STORE's cell axis; carrier pixels are always NEST:
            # nodes for the dual mapping, elements for the pixel mapping)
            if sub_idx is not None:
                # extract: position within the (sorted) healpix_index —
                # every carrier pixel is in the subset by construction of
                # the wet-mask scatter; VERIFY anyway (defense against a
                # store whose index disagrees with its own data ordering)
                pos = np.searchsorted(sub_idx, carrier_pix)
                if (np.any(pos >= sub_idx.size)
                        or np.any(sub_idx[np.minimum(pos, sub_idx.size - 1)]
                                  != carrier_pix)):
                    raise ValueError(
                        f"healpix extract {path!r}: mesh carrier pixels "
                        "are not all present in healpix_index — the "
                        "extract is internally inconsistent.")
                mesh._healpix_data_idx = pos
            elif order == "ring":
                mesh._healpix_data_idx = hp.nest2ring(nside, carrier_pix)
            else:
                mesh._healpix_data_idx = carrier_pix
            mesh._healpix_data_order = order
            mesh._healpix_data_ncells = ncells
            if (nfin_u is not None and geometry == "dual"
                    and nfin_u.size == ncells):
                # mesh-node order through the same store index the data
                # compaction uses; -1 = no u in the column (never on a mesh
                # node: land is dropped by the wet mask)
                mesh.metadata["uv_data_last"] = (
                    nfin_u[mesh._healpix_data_idx] - 1).astype(np.int32)
        finally:
            ds.close()
        HealpixReader._apply_rotation(mesh, mesh_cfg)
        return mesh

    @staticmethod
    def _apply_rotation(mesh, mesh_cfg):
        """Rotate the (geographic) dual mesh into a pole-safe frame.

        HEALPix is NOT a model-native rotated grid (the fesom2 case) —
        the remap carries geographic east/north components on a global
        pixel grid — so rotation here is a pure runtime transform:
        per-node tangent vector factors from the GEOGRAPHIC pixel
        centres (exact ``rotate_vectors`` machinery), then nodes and
        element centres rotated AS POINTS, then the search structures
        REBUILT on the rotated coordinates (the grid hash built during
        the mesh construction indexed the geographic chart).  Topology
        and pixel ids never change.

        The rotated frame has its own two poles, and wherever they land
        the polar pathologies move THERE — so this GUARDS the dangerous
        configurations instead of silently degrading:

        * a wet element straddling a rotated pole (wrapped-lon span
          > 180° in the rotated chart) would get a lying containment
          chart → hard ``ValueError`` (pick angles whose poles land on
          land, as the FESOM2 defaults do: Greenland / East Antarctica —
          measured on the CORE2 twin, wet nodes stay below |rotated
          lat| 85.3°);
        * wet nodes above 88° rotated latitude → one-time
          ``RuntimeWarning`` (grid-hash cold-anchor degradation zone).
        """
        rotation = getattr(mesh_cfg, "rotation", None)
        if rotation is None:
            return
        from ..core.rotation import (FESOM2_ANGLES, euler_matrix,
                                     rotate_lonlat, rotate_vectors)
        if isinstance(rotation, str):
            if rotation != "fesom2":
                raise ValueError(
                    f"mesh.rotation: unknown preset {rotation!r} "
                    "(use 'fesom2', a 3-list of Euler angles, or null).")
            angles = FESOM2_ANGLES
        else:
            try:
                angles = tuple(float(a) for a in rotation)
            except TypeError:
                raise ValueError(
                    f"mesh.rotation must be 'fesom2', a 3-list of Euler "
                    f"angles, or null — got {rotation!r}") from None
            if len(angles) != 3:
                raise ValueError(
                    f"mesh.rotation must be 3 Euler angles, got {rotation!r}")
        R = euler_matrix(*angles)

        lon_g, lat_g = mesh.nodes_lon, mesh.nodes_lat
        rlon, rlat = rotate_lonlat(lon_g, lat_g, R)

        # -- guards on the ROTATED geometry (the pathology moved there) --
        en = mesh.elem_nodes
        vmask = en >= 0
        safe = np.where(vmask, en, en[:, [0]])
        lf = rlon[safe]
        d = ((lf - lf[:, [0]] + 180.0) % 360.0) - 180.0
        span = (np.where(vmask, d, 0.0).max(axis=1)
                - np.where(vmask, d, 0.0).min(axis=1))
        if (span > 180.0).any():
            from ..core.rotation import unrotate_lonlat
            pl, pb = unrotate_lonlat(np.array([0.0, 0.0]),
                                     np.array([90.0, -90.0]), R)
            raise ValueError(
                f"mesh.rotation {angles}: {(span > 180.0).sum()} wet "
                "element(s) straddle a rotated pole — their containment "
                "chart would misbehave. These angles put the rotated "
                f"poles at geographic ({pl[0]:.1f}, {pb[0]:.1f}) and "
                f"({pl[1]:.1f}, {pb[1]:.1f}); choose angles whose poles "
                "fall on DRY land (the 'fesom2' preset: Greenland / "
                "East Antarctica).")
        hi = np.abs(rlat) > 88.0
        if hi.any():
            _warn_once(("rot-near-pole", angles),
                       f"healpix reader: {int(hi.sum())} wet node(s) sit "
                       f"above 88 deg ROTATED latitude under "
                       f"mesh.rotation {angles} — nearest-node search "
                       "degrades there (the polar caveat moved to the "
                       "rotated poles).")

        # -- vector factors FIRST (need the geographic carriers) --
        c_node, s_node = rotate_vectors(
            np.ones_like(lon_g), np.zeros_like(lon_g), lon_g, lat_g, R)
        mesh._vec_rot_factors = {"node": (c_node, s_node)}
        if mesh.metadata.get("healpix_geometry") == "pixel":
            # pixel geometry: u/v ride the ELEMENTS (pixel centres) —
            # factors from the GEOGRAPHIC element centres, before the
            # point rotation below overwrites them.
            ce_lon, ce_lat = mesh.elem_center_lon, mesh.elem_center_lat
            c_el, s_el = rotate_vectors(
                np.ones_like(ce_lon), np.zeros_like(ce_lon),
                ce_lon, ce_lat, R)
            mesh._vec_rot_factors["element"] = (c_el, s_el)

        # -- rotate stored coordinates as points, rebuild search --
        mesh.nodes_lon, mesh.nodes_lat = rlon, rlat
        mesh.elem_center_lon, mesh.elem_center_lat = rotate_lonlat(
            mesh.elem_center_lon, mesh.elem_center_lat, R)
        mesh.rotation_angles = angles
        if mesh.metadata.get("healpix_geometry") == "pixel":
            # the polar-degeneracy list was computed on the GEOGRAPHIC
            # chart; recompute it on the rotated one (with poles on land
            # it is normally empty) so the metadata never goes stale.
            pol = np.abs(mesh.nodes_lat) > _POLE_CORNER_LAT
            mesh.metadata["healpix_polar_degenerate"] = np.flatnonzero(
                pol[mesh.elem_nodes].any(axis=1)).astype(np.int64)
        build_search(mesh)          # grid hash indexed the geo chart
        mesh._node_tree = None      # lazy KD rebuilds on the new coords
        # NO convexity re-validation: find_nonconvex_quads projects via
        # 3-D unit vectors onto the tangent plane at the centroid —
        # exactly rotation-invariant (review-verified), and the pass
        # dominates build memory at high zoom (~9 s / 1.2 GB at nside
        # 1024).  The builder already validated the same geometry.
        print(f"    rotated frame: alpha/beta/gamma = {angles} "
              "(pole-safe)")

    @staticmethod
    def open_dataset(path, v_path=None, w_path=None, mode_cfg=None,
                     eta_path=None):
        """Open the velocity store; attach ssh from a second store if needed.

        ``eta_path`` (config ``data.ncinput_eta``) points at the store
        carrying ssh when the velocity store lacks it (the ng5 recipe);
        unset, a ``3D``→``2D`` filename sibling is tried automatically.
        """
        if v_path:
            raise ValueError(
                "healpix reader: data.ncinput_v is not used — u and v live "
                "in one healpix store; use data.ncinput_eta for a separate "
                "ssh store.")
        if w_path:
            raise ValueError(
                "healpix reader: data.ncinput_w is not used — w lives in "
                "the same 3-D healpix store as u/v (ng5 layout).")
        ds = _open_healpix_store(path)
        cfg = mode_cfg
        var_eta = getattr(cfg, "var_eta", None) if cfg else None
        # SOFT probe of the main store: a set-but-absent var_eta means
        # "look in the eta store" (the two-store recipe), not KeyError.
        eta_name = _first_soft(ds, _ETA_ALIASES, var_eta)
        if eta_name is not None:
            return ds
        eta_p = eta_path or _derive_eta_sibling(path)
        if eta_p is None:
            if var_eta:
                raise KeyError(
                    f"healpix reader: mode.var_eta={var_eta!r} is not in "
                    f"the store and no eta store was found "
                    "(data.ncinput_eta unset, no 2D filename sibling).")
            _warn_once(("no-eta", str(path)),
                       "healpix reader: no ssh variable in the store and no "
                       "eta store found (data.ncinput_eta unset, no 2D "
                       "sibling) — eta = 0 for the whole run.")
            return ds
        ds_eta = _open_healpix_store(eta_p)
        try:
            eta_name = _first(ds_eta, _ETA_ALIASES,
                              getattr(cfg, "var_eta", None) if cfg else None)
            if eta_name is None:
                raise KeyError(
                    f"healpix reader: eta store {eta_p!r} has no ssh "
                    f"variable (looked for {_ETA_ALIASES}; present: "
                    f"{sorted(ds_eta.data_vars)[:12]}…).")
            _validate_time_axes_match(ds, ds_eta, f"eta store {eta_p!r}")
        except Exception:
            ds_eta.close()
            ds.close()
            raise
        # On success BOTH stores stay open deliberately: the returned
        # dataset reads the assigned eta variable lazily from ds_eta's
        # backing store for the whole run (xarray's file manager owns the
        # handles; closing the returned ds releases its own store only).
        return ds.assign({eta_name: ds_eta[eta_name]})

    @staticmethod
    def probe_uv_layout(ds, mode_cfg=None, mesh=None) -> str:
        # Layout follows the mesh mapping: dual geometry carries pixel
        # centres as NODES (run needs uv_interp: barycentric); pixel
        # geometry carries them as ELEMENTS (the fused leastsq kernels
        # apply).  Surfacing this lets run_tracking fail fast before the
        # writer exists.
        if (mesh is not None and getattr(mesh, "metadata", None)
                and mesh.metadata.get("healpix_geometry") == "pixel"):
            return "element"
        return "node"

    @staticmethod
    def _read_kh_pixels(ds, timestep, mesh, diffusion, cfg):
        """Pixel-space Kh fields for em/m1 → compacted NODE arrays.

        The shared ``_read_kh_fields`` helper detects layout by array
        SIZE, which cannot work here: healpix stores carry variables on
        the FULL pixel space (``value``/``cell`` = npix), not the
        compacted wet-node count — so Kh takes the same compaction path
        as u/v/eta.  Returns ``(kh_x, kh_y, "node")`` or ``None`` (no
        diffusion / disabled / variables absent — absent
        mirrors fesom2: the split_rk kick then raises its own clear
        ValueError).  Non-finite Kh at wet nodes (mask drift between the
        Kh family and the t=0 u∧v mask) is zeroed with a one-time
        warning — NOTE the zero is not local: it enters the LSQ plane
        fit of every 1-ring touching that node, pulling the interpolated
        Kh down by ~1/ring_size and adding a spurious outward ∂K drift
        there (review-measured −11% on the neighbours) — acceptable for
        isolated drift cells, but a large bad fraction means the wrong
        store.  Negative Kh is rejected (same contract as the shared
        ``_read_kh_fields``: a negative diffusivity silently drops the
        noise while the Itô drift keeps acting).
        """
        from . import _resolve_kh_var_name, _DEFAULT_KH_X_NAMES, \
            _DEFAULT_KH_Y_NAMES
        if diffusion is None or not getattr(diffusion, "enabled", False):
            return None
        name_x = _resolve_kh_var_name(
            ds, getattr(diffusion, "var_kh_zonal", None),
            _DEFAULT_KH_X_NAMES)
        name_y = _resolve_kh_var_name(
            ds, getattr(diffusion, "var_kh_meridional", None),
            _DEFAULT_KH_Y_NAMES)
        if name_x is None or name_y is None:
            return None
        out = []
        for nm in (name_x, name_y):
            da = ds[nm]
            if "time" in da.dims:
                da = da.isel(time=timestep)
            arr = _compact_to_nodes(da, mesh, nm, cfg)
            # NaN = mask drift (zero + warn); ±Inf is NOT drift — it is
            # a broken field and must not slip past the sign check
            # below by being zeroed first (round-2 review).
            if np.isinf(arr).any():
                raise ValueError(
                    f"healpix reader: Kh field {nm!r} contains "
                    "±Inf values — broken store, not mask drift.")
            bad = np.isnan(arr)
            if bad.any():
                _warn_once(("nan-kh", nm),
                           f"healpix reader: {int(bad.sum())} wet "
                           f"node(s) carry non-finite Kh ({nm!r}) — "
                           "zeroed (this also perturbs the Kh stencil "
                           "of the surrounding 1-rings; see docs).")
                arr = np.nan_to_num(arr, nan=0.0)
            if (arr < 0).any():
                raise ValueError(
                    f"healpix reader: Kh fields must be non-negative; "
                    f"min({nm!r}) = {float(arr.min())} (a negative "
                    "diffusivity silently drops the noise term while "
                    "the Ito drift keeps acting).")
            out.append(arr.astype(np.float64, copy=False))
        if (getattr(mesh, "rotation_angles", None) is not None
                and not np.array_equal(out[0], out[1])):
            _warn_once(
                ("aniso-kh-rot", name_x, name_y,
                 tuple(mesh.rotation_angles)),
                f"healpix reader: ANISOTROPIC Kh ({name_x!r} != "
                f"{name_y!r}) under mesh.rotation — the pair is applied "
                "along the ROTATED axes, not geographic east/north (the "
                "diffusion ellipse turns with the frame). Use isotropic "
                "Kh (equal fields) under rotation; see "
                "docs/coordinate_frame.md §3.")
        return out[0], out[1], "node"

    @staticmethod
    def read_velocities(ds, timestep, mesh, mode_cfg=None, diffusion=None,
                        sample_vars=None, velocity_float32=False):
        from ..core.io import VelocityField

        cfg = mode_cfg
        is_3d = cfg is not None and getattr(cfg, "Fdrift3D", False)
        geometry = (mesh.metadata.get("healpix_geometry", "dual")
                    if getattr(mesh, "metadata", None) else "dual")
        if geometry == "pixel":
            # v1 scope guards (load_mesh already blocks 3-D; this is the
            # direct-caller backstop).
            if is_3d:
                raise NotImplementedError(
                    "healpix pixel geometry is 2-D only in v1")
            if sample_vars:
                raise NotImplementedError(
                    "healpix pixel geometry v1 does not support "
                    "sample_vars (sampled scalars live at pixel centres; "
                    "the corner-node sampling path is staged) — use the "
                    "dual geometry for T/S sampling.")
            if (diffusion is not None
                    and getattr(diffusion, "enabled", False)
                    and getattr(diffusion, "scheme", "uniform")
                    in ("em", "m1")):
                # early + variable-independent: fail before any Kh read
                # (and before run_tracking creates a writer).
                raise NotImplementedError(
                    "healpix pixel geometry v1 does not support spatial "
                    "Kh fields (em/m1 diffusion) — Kh compaction targets "
                    "pixel centres, which are elements here; use the "
                    "dual geometry or uniform Kh.")
        u_name = _first(ds, _U_ALIASES,
                        getattr(cfg, "var_u", None) if cfg else None)
        v_name = _first(ds, _V_ALIASES,
                        getattr(cfg, "var_v", None) if cfg else None)
        if u_name is None or v_name is None:
            raise KeyError(
                f"healpix reader: no u/v variable in the dataset (looked "
                f"for {_U_ALIASES} / {_V_ALIASES}).")

        w = None
        if is_3d:
            # mesh.sigma = K interfaces (run_tracking has already applied
            # mode.max_levels to it) => read K-1 u/v layers + K w levels.
            # ng5: the store's level axis has K entries for u/v too — the
            # deepest is all-NaN padding and is dropped by the K-1 slice.
            k_ifc = int(mesh.sigma.size)
            if k_ifc < 2:
                raise ValueError(
                    "healpix reader: 3-D read needs >= 2 vertical "
                    "interfaces on the mesh — set mesh.levels AND load "
                    "the mesh with mode_cfg (run_tracking/the CLI do; a "
                    "direct load_mesh(mesh_cfg) call builds a 2-D mesh).")
            u = _compact_3d(ds[u_name].isel(time=timestep), mesh, u_name,
                            k_ifc - 1, check_prefix=True)
            v = _compact_3d(ds[v_name].isel(time=timestep), mesh, v_name,
                            k_ifc - 1, check_prefix=True)
            w_name = _first(ds, _W_ALIASES,
                            getattr(cfg, "var_w", None) if cfg else None)
            if w_name is None:
                raise KeyError(
                    f"healpix reader: 3-D mode needs a vertical velocity "
                    f"(looked for {_W_ALIASES}, or set mode.var_w); "
                    "point data.ncinput at the 3-D store.")
            # wo sits ON the K interfaces (verified on ng5: 70 data
            # levels vs u's 69, surface interface ~ 0).  Positive UP like
            # the FESOM2 w it comes from (ng5 avg_wo = "Time-mean upward
            # sea water velocity", +2e-5 m/s at 50 m in the equatorial
            # Pacific; the CORE2 twins are remapped FESOM2 w) -> negated
            # into the tracker's positive-down depth rate.
            w = _compact_3d(ds[w_name].isel(time=timestep), mesh, w_name,
                            k_ifc, check_prefix=True)
            if resolve_w_sign(ds[w_name], cfg, "up") < 0.0:
                np.negative(w, out=w)
            surf_bad = ~(np.isfinite(u[0]) & np.isfinite(v[0]))
        else:
            u = _compact_to_nodes(ds[u_name].isel(time=timestep), mesh,
                                  u_name, cfg)
            v = _compact_to_nodes(ds[v_name].isel(time=timestep), mesh,
                                  v_name, cfg)
            surf_bad = ~(np.isfinite(u) & np.isfinite(v))

        # Rotated frame: the store carries GEOGRAPHIC east/north — apply
        # the per-node tangent map (u' = c·u − s·v, v' = s·u + c·v);
        # (N,) factors broadcast over (K−1, N) in 3-D.  A NaN in EITHER
        # component makes BOTH NaN after the mix, so the per-component
        # zeroing below zeroes the pair as a unit under rotation (in the
        # geographic frame a finite-u/NaN-v cell keeps its finite u —
        # a deliberate frame-dependent difference: mixed components are
        # meaningless once one leg is missing).  The surf_bad
        # conjunction mask is invariant under the mix.  w/eta/scalars
        # are scalars — frame-invariant, untouched.
        vd = np.float32 if velocity_float32 else np.float64
        if w is not None:
            # w skips the rotation mix — drop it to the velocity dtype
            # NOW so its float64 copy is gone before the mix temporaries
            # (values identical to casting at the return; a finite-f8
            # value beyond float32 range would become inf and be zeroed
            # below instead of returned as inf — safer, and impossible
            # for physical velocities)
            w = w.astype(vd, copy=False)
        if (is_3d and geometry == "dual"
                and getattr(cfg, "uv_vertical", "layer") == "linear"):
            # depth-linear layer u/v: validate the RAW record against the
            # depth-point table (bathymetry, kept levels, the declared u
            # prefix) before the zeroing below turns a missing value into
            # data — and before the rotation mix, which makes a NaN in one
            # component NaN in both, so the refusal names the component
            # that carries it; refused naming node, point and record
            from types import SimpleNamespace
            from ..numerics.vertical_uv import check_depth_linear_record
            check_depth_linear_record(
                SimpleNamespace(u=u, v=v, uv_vertical="layer_linear",
                                _dl_token=None),
                mesh, record=_record_label(ds, timestep))

        if getattr(mesh, "rotation_angles", None) is not None:
            carrier = "element" if geometry == "pixel" else "node"
            c_n, s_n = mesh._vec_rot_factors[carrier]
            u, v = _rotate_uv_blocked(u, v, c_n, s_n)
        # the mix itself stays float64; u/v drop to the velocity dtype
        # right after it (bit-equal to the historical cast-at-return)
        u = u.astype(vd, copy=False)
        v = v.astype(vd, copy=False)

        if surf_bad.any():
            # masks are static by convention; a few non-finite wet nodes
            # on the TRACKED (surface) layer mean the t=0-derived mask has
            # drifted — zeroed below.  A LARGE bad fraction is not drift,
            # it is the wrong store/level/variable — fail loud instead of
            # silently tracking on zeros.
            frac = surf_bad.mean()
            if frac > 0.05:
                raise ValueError(
                    f"healpix reader: {frac:.1%} of wet nodes carry "
                    f"non-finite u/v at timestep {timestep} — this is not "
                    "coastal mask drift; check the store, variable names "
                    "and level selection.")
            _warn_once(("nan-uv", u_name),
                       f"healpix reader: {int(surf_bad.sum())} wet node(s) "
                       f"carry non-finite u/v at timestep {timestep} (mask "
                       "drift vs the t=0 wet mask) — zeroed.")
        # zero ALL non-finite velocities: below-bottom levels are NaN in
        # healpix products (finite-prefix columns), and the u- vs w-family
        # bottoms disagree per cell (ng5 remap quirk) — zero flow in the
        # ground is the correct clamped behaviour.
        u[~np.isfinite(u)] = 0.0
        v[~np.isfinite(v)] = 0.0
        if w is not None:
            w[~np.isfinite(w)] = 0.0

        eta = np.zeros(mesh.n_nodes, dtype=np.float64)
        # soft: eta is optional here — open_dataset already enforced a set
        # var_eta (merged store, or KeyError when nowhere to be found)
        eta_name = _first_soft(ds, _ETA_ALIASES,
                               getattr(cfg, "var_eta", None) if cfg else None)
        if eta_name is not None:
            eta_c = _compact_to_nodes(ds[eta_name].isel(time=timestep),
                                      mesh, eta_name, cfg)
            # eta land masks differ from the u/v family on ~2% coastal
            # cells (element- vs node-sourced remap) -> zero, never NaN.
            eta_c = np.nan_to_num(eta_c, nan=0.0, posinf=0.0, neginf=0.0)
            if geometry == "pixel":
                # ssh compacts at pixel CENTRES (= elements); the
                # VelocityField contract wants node-based eta -> mean over
                # the incident wet pixels at every corner node.
                acc = np.zeros(mesh.n_nodes, dtype=np.float64)
                cnt = np.zeros(mesh.n_nodes, dtype=np.int64)
                flat = mesh.elem_nodes.ravel()
                np.add.at(acc, flat, np.repeat(eta_c, 4))
                np.add.at(cnt, flat, 1)
                eta = acc / np.maximum(cnt, 1)
            else:
                eta = eta_c

        scalars = None
        if sample_vars:
            scalars = HealpixReader.read_scalar_fields(
                ds, timestep, mesh, sample_vars, mode_cfg)

        # Pixel geometry never reads spatial Kh: em/m1 already failed
        # fast above (variable-independent), and a uniform run must not
        # be touched by Kh-named store variables (2026-08 review; the
        # DUAL geometry keeps its pinned attach-on-uniform behaviour).
        kh = (None if geometry == "pixel"
              else HealpixReader._read_kh_pixels(ds, timestep, mesh,
                                                 diffusion, cfg))
        kh_kwargs = {}
        if kh is not None:
            kh_kwargs = dict(kh_zonal=kh[0], kh_meridional=kh[1],
                             kh_layout=kh[2])

        vd = np.float32 if velocity_float32 else np.float64
        return VelocityField(
            u=u.astype(vd, copy=False), v=v.astype(vd, copy=False),
            eta=eta, w=w.astype(vd, copy=False) if w is not None else None,
            uv_layout="element" if geometry == "pixel" else "node",
            scalars=scalars, **kh_kwargs)

    @staticmethod
    def read_scalar_fields(ds, timestep, mesh, varnames, mode_cfg=None):
        if (getattr(mesh, "metadata", None)
                and mesh.metadata.get("healpix_geometry") == "pixel"):
            # scalars compact at pixel CENTRES = elements here, which
            # would violate the node-field scalar contract downstream
            # (sampling, beaching dry_mask) — fail fast (2026-08 review).
            raise NotImplementedError(
                "healpix pixel geometry v1 does not support node-based "
                "scalar reads (sample_vars / beaching.dry_mask_var) — "
                "scalars live at pixel centres; use the dual geometry.")
        is_3d = mode_cfg is not None and getattr(mode_cfg, "Fdrift3D",
                                                 False)
        out = {}
        for nm in varnames:
            if nm in _W_ALIASES or (cfg_w := getattr(mode_cfg, "var_w", None)) == nm:
                # the generic scalar path treats a level field as K-1 layer
                # mids: W (K interfaces) would be truncated and shifted
                # (review 2026-09-23) — the vel_w output already records it
                raise ValueError(
                    f"healpix reader: {nm!r} is the vertical velocity — it "
                    "cannot be sampled as a scalar (the scalar path is for "
                    "layer fields); the trajectory output records W at the "
                    "particle as vel_w.")
            if nm not in ds.variables:
                raise ValueError(
                    f"healpix reader: sampled variable {nm!r} not in the "
                    f"store (present: {sorted(ds.data_vars)[:12]}…).")
            da = ds[nm].isel(time=timestep)
            has_lev = any(str(d).lower() in _LEVEL_DIMS for d in da.dims)
            if is_3d and has_lev:
                # mid-layer scalar (K-1, N), forward-filled down each
                # column: the sampler's bottom clamp then reads bottom-
                # water properties instead of NaN where the scalar's own
                # bottom is shallower than the u-derived bathymetry.
                # (Interface-located sampled scalars — K levels — are not
                # supported; every known healpix T/S product is mid-layer.)
                arr = _compact_3d(da, mesh, nm, int(mesh.sigma.size) - 1)
                out[nm] = _ffill_columns(arr)
            else:
                out[nm] = _compact_to_nodes(da, mesh, nm, mode_cfg)
        return out


__all__ = ["build_healpix_dual_mesh", "HealpixReader"]
