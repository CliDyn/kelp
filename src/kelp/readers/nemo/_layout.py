"""NEMO raster layouts: E-W periodicity, the north fold and their detection.

:class:`NemoCgridLayout` says what a raw ``(ny, nx)`` raster holds;
:func:`nemo_cgrid_to_canonical` brings a raster to its canonical (halo) form
(the numpy ``lbc_lnk``, optionally checking the copies the file carries);
:func:`detect_nemo_cgrid_layout` classifies a mesh file's coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np


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
class NemoCgridLayout:
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


def _centre(layout: NemoCgridLayout, kind: str):
    """Fold centre ``(Y2, X2)``: the mirror of ``kind`` point ``(j, i)`` is
    ``(Y2 - j - oy, X2 - i - ox)`` (columns modulo the period)."""
    f = layout.fold == "F"
    return 2 * layout.J + f, 2 * layout.c0 + layout.ni - f


def _mirror(kind: str, j, i, layout: NemoCgridLayout):
    oy, ox = _KIND_OFFSET[kind]
    Y2, X2 = _centre(layout, kind)
    c0, ni = layout.c0, layout.ni
    return Y2 - j - oy, c0 + ((X2 - i - ox) - c0) % ni


def nemo_cgrid_canonical(kind: str, j, i, layout: NemoCgridLayout
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonical point of canonical-raster points.

    The E-W wrap is applied first, then the fold: a point beyond the fold line,
    or on it with the larger column, maps to its mirror image.  Idempotent.

    Parameters
    ----------
    kind : {"T", "U", "V", "F"}
        Grid point type of the indices.
    j, i : array_like of int
        Row / column indices in the canonical (halo) raster of ``layout``.
    layout : NemoCgridLayout
        The raster layout.

    Returns
    -------
    j, i : ndarray of int64
        Indices of the canonical points.
    sign : ndarray of float
        ``-1`` where the point was mirrored at the fold (vector components flip
        there), else ``+1``.
    """
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
def _copy_sets(layout: NemoCgridLayout, kind: str):
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
    cj, ci, s = nemo_cgrid_canonical(kind, jj, ii, layout)
    if ((cj < 0) | (cj >= nr) | (ci < 0) | (ci >= nc)).any():
        raise ValueError("the raster is too small for its fold / halo layout")
    moved = (cj != jj) | (ci != ii)
    present = np.zeros((nr, nc), dtype=bool)
    pc = int(layout.pad_cols)
    present[:ny, pc:pc + nx] = True
    return ((jj * nc + ii)[moved], (cj * nc + ci)[moved], s[moved], present.reshape(-1))


def nemo_cgrid_to_canonical(a, layout: NemoCgridLayout, kind: str, *, vector: bool = False,
                       check: bool = False, name: str = "raster") -> np.ndarray:
    """Bring a raw raster to the canonical (halo) form of ``layout``.

    The raster is padded to ``layout.shape``, NaN / inf become 0, and every
    halo / fold copy is overwritten from its canonical point.

    Parameters
    ----------
    a : array_like (..., ny, nx)
        Raw raster of ``layout.raw_shape`` on the trailing axes.
    layout : NemoCgridLayout
        The raster layout.
    kind : {"T", "U", "V", "F"}
        Grid point type of ``a``.
    vector : bool
        Multiply the copies by the fold sign (u and v components only).
    check : bool
        First compare the copies the raw raster carries with their canonical
        points (exact: NEMO writes bit copies).
    name : str
        Name used in error messages.

    Returns
    -------
    ndarray (..., nr, nc), float64
        The canonical raster.

    Raises
    ------
    ValueError
        The trailing shape is not ``layout.raw_shape``, or (``check``) a copy
        differs from its canonical point: a stale halo, a wrong layout or a
        field from another grid file.
    """
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


def nemo_fold_pivot_cells(layout: NemoCgridLayout) -> np.ndarray:
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


def _read_layout_flags(ds) -> dict:
    """NEMO 4.2 topology attributes (global attributes or scalar variables):
    ``Iperio`` / ``Jperio`` / ``NFold`` (ints) and ``NFtype`` (``"T"`` / ``"F"``).
    Only a cross-check of what the coordinates show (NEMO <= 4.0 mesh files
    carry none of them)."""
    out = {}
    for k in ("Iperio", "Jperio", "NFold", "NFtype"):
        v = ds.attrs.get(k)
        if v is None and k in ds.variables and ds[k].size == 1:
            v = ds[k].values.reshape(-1)[0]
        if v is None:
            continue
        if isinstance(v, bytes):
            v = v.decode()
        out[k] = str(v).strip().upper() if k == "NFtype" else int(v)
    return out


#: neighbouring columns this many median spacings apart across the wrap still
#: count as one periodic raster
_WRAP_SPACINGS = 1.5


def _wrap_ratio(xyz) -> float:
    """Median distance from the last to the first column over the median column
    spacing, on the rows below the fold region."""
    ny = xyz.shape[0]
    rows = slice(0, max(1, ny - 3))
    wrap = np.linalg.norm(xyz[rows, -1] - xyz[rows, 0], axis=-1)
    step = np.median(np.linalg.norm(xyz[rows, 1:] - xyz[rows, :-1], axis=-1), axis=1)
    ok = step > 0
    return float(np.median(wrap[ok] / step[ok])) if ok.any() else np.inf


def detect_nemo_cgrid_layout(tlon, tlat, flon=None, flat=None, flags=None
                        ) -> Tuple[NemoCgridLayout, List[str]]:
    """Classify the raw raster for ``mesh.nemo_grid: cgrid``.

    The coordinates decide what the raster contains: the E-W halo columns, the
    north halo row and the fold are identities between points (the copies NEMO's
    ``lbc_lnk`` writes), checked exactly for the T points and the F points of
    the real rows (the halo-row F is not reliable in ORCA05; F optional), no thresholds.
    The nine candidates ``{halo columns, none} x {no fold, T / F fold, with /
    without the halo row}`` are tried; the passing one with the most copies wins
    (no-halo and halo F-pivot share a fold centre, so the count is the
    tie-break); equal counts refuse.  ``flags`` (:func:`_read_layout_flags`) only
    cross-check: a contradiction refuses; flags that claim MORE than the raster
    shows (all of our crops inherit the parent file's) make it a crop -- the
    missing topology is an open edge, noted in ``notes``.  Periodicity without
    halo columns (halo-free fold or ``Iperio``) must also pass the wrap-gap test
    (last -> first column within 1.5 median spacings): fold identities on a
    partial-width crop symmetric about a pivot would otherwise join its edges.
    Also refuses zero-filled tiles (land-processor elimination) and a pole row.

    Parameters
    ----------
    tlon, tlat : array_like (ny, nx)
        T-point longitude / latitude (deg) of the raw raster.
    flon, flat : array_like (ny, nx) or None
        F-point coordinates (optional; they sharpen the fold test).
    flags : dict or None
        The file's NEMO 4.2 attributes ``Iperio`` / ``NFold`` / ``NFtype``.

    Returns
    -------
    layout : NemoCgridLayout
        Halo columns, fold type and the open edges of the raster.
    notes : list of str
        Messages for the load log (e.g. flags that claim more than the raster
        shows).

    Raises
    ------
    ValueError
        Zero-filled tiles, a pole row, ambiguous candidates, a partial-width
        raster whose fold identities hold, or flags that contradict the
        coordinates.
    """
    flags = flags or {}
    tlon, tlat = (np.asarray(a, dtype=np.float64) for a in (tlon, tlat))
    ny, nx = raw = tlon.shape
    have_f = flon is not None
    if have_f:
        flon, flat = (np.asarray(a, dtype=np.float64) for a in (flon, flat))
    if tlat.shape != raw or (have_f and (flon.shape != raw or flat.shape != raw)):
        raise ValueError("glamt/gphit and glamf/gphif must share one (ny, nx) shape")
    if not all(np.isfinite(a).all() for a in (tlon, tlat) + ((flon, flat) if have_f else ())):
        raise ValueError("the mesh file's glamt/gphit/glamf/gphif hold non-finite values: "
                         "the layout cannot be detected")
    def _zero_block(lon, lat):        # a tile is a block; one point (0E, 0N) and its halo copy are not
        z = (lon == 0) & (lat == 0)
        return bool((z[:, 1:] & z[:, :-1]).any() or (z[1:] & z[:-1]).any())
    if _zero_block(tlon, tlat) or (have_f and _zero_block(flon, flat)):
        raise ValueError(
            "the mesh file has zero-filled tiles (land-processor elimination when "
            "it was assembled): regenerate it without land elimination or use domain_cfg.nc")
    if flags.get("Jperio", 0) == 1:
        raise NotImplementedError("mesh.nemo_grid: cgrid: N-S periodic grids (Jperio = 1) are not supported")
    xt, xf = _unit_xyz(tlon, tlat), (_unit_xyz(flon, flat) if have_f else None)
    if np.linalg.norm(xt[-1] - xt[-1, :1], axis=-1).max() <= SAME_POINT_CHORD:
        raise ValueError("the top row of T points collapses to one point (a pole row): "
                         "crop below it")
    notes = []
    if ny < 4 or nx < 4:
        return NemoCgridLayout(raw), notes

    def copies(lay):
        """Number of copy pairs compared, or None when one does not coincide."""
        n = 0
        for kind, xyz in (("T", xt), ("F", xf)):
            if xyz is None:                                # no F coordinates (CMIP output)
                continue
            dst, src, _s, present = _copy_sets(lay, kind)
            nc, pc = lay.shape[1], int(lay.pad_cols)
            k = present[dst]
            if kind == "F" and lay.fold:
                k &= dst // nc <= lay.J
            d, s_ = dst[k], src[k]
            a = xyz[d // nc, d % nc - pc]
            b = xyz[s_ // nc, s_ % nc - pc]
            if d.size and np.linalg.norm(a - b, axis=-1).max() > SAME_POINT_CHORD:
                return None
            n += int(d.size)
        return n

    found = []
    for halo_cols in (True, False):
        for fold, halo_row in ((None, False), ("T", True), ("T", False), ("F", True), ("F", False)):
            if fold is None and not halo_cols:
                continue                                   # implies no copy at all
            lay = NemoCgridLayout(raw, halo_cols, halo_row, True, fold)
            n = copies(lay)
            if n:
                found.append((n, lay))
    found.sort(key=lambda t: -t[0])
    if len(found) > 1 and found[0][0] == found[1][0]:
        raise ValueError("the raster matches two layouts equally well: "
                         f"{found[0][1].label()} / {found[1][1].label()}")
    iperio, nfold, nftype = flags.get("Iperio"), flags.get("NFold"), flags.get("NFtype")
    ratio = _wrap_ratio(xt)
    if found:
        lay = found[0][1]
        if lay.fold and (nfold == 0 or nftype not in (None, lay.fold)):
            raise ValueError(f"the coordinates show a {lay.fold}-pivot fold, the file's flags say "
                             f"NFold={nfold}, NFtype={nftype!r}")
        if iperio == 0:
            raise ValueError("the coordinates show E-W periodicity, the file's flag says Iperio=0")
        if not lay.halo_cols and ratio > _WRAP_SPACINGS:
            raise ValueError(
                f"the fold identities hold, but the raster is not full width (the last column is "
                f"{ratio:.1f} grid spacings from the first): a partial-width crop through the fold "
                "cannot be joined; crop the full width")
        if not lay.fold and nfold == 1:
            notes.append("the flags claim a north fold the raster does not show: treated as a crop "
                         "(the north edge is open)")
        return lay, notes
    looks_periodic = ratio <= _WRAP_SPACINGS
    if iperio == 1:
        if looks_periodic:
            if nfold == 1:
                notes.append("the flags claim a north fold the raster does not show: treated as a crop "
                             "(the north edge is open)")
            return NemoCgridLayout(raw, False, False, True, None), notes
        notes.append("the flags claim E-W periodicity the raster does not show (not full width): "
                     "treated as a crop (the east and west edges are open)")
    elif looks_periodic and iperio is None:
        raise ValueError(
            "the raster looks E-W periodic without halo columns (the last column is "
            f"{ratio:.1f} grid spacings from the first) but the file says nothing: add Iperio=1 "
            "(and NFold / NFtype for a fold) to the mesh file's attributes, or crop it")
    return NemoCgridLayout(raw), notes
