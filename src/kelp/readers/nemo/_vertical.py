"""NEMO z-level vertical grid: partial-step checks, layer thicknesses, W.

The checks refuse what the C-grid path does not support (s-coordinates,
ice-shelf cavities, face thicknesses off the partial-step rule);
:func:`nemo_cgrid_thickness` expands the compact one-partial-level-per-point
form; :func:`nemo_cgrid_w_from_continuity` integrates the face fluxes into W.
"""
from __future__ import annotations

import numpy as np


def _refuse_unsupported_vertical(ds, path) -> None:
    """z-levels with partial bottom cells only: s-coordinates and ice-shelf
    cavities are refused (the file's attributes, ``top_level``, ``misf``)."""
    vc = ds.attrs.get("VertCoord")
    if vc is not None and str(vc).strip().lower() not in ("zco", "zps"):
        raise NotImplementedError(
            f"mesh file {path!r}: VertCoord = {str(vc)!r}; only z-levels "
            "(zco, zps) are supported, not s-coordinates")
    # only real cavities: the variables count at WET columns (ORCA05 mesh_mask
    # carries misf = 1 and isfdraft = 0 there, and 0 on land)
    if "tmask" in ds.variables:
        tm = np.asarray(ds["tmask"]).squeeze()
        wet = (tm > 0.5).any(0) if tm.ndim == 3 else tm > 0.5
    elif "bottom_level" in ds.variables:
        wet = np.asarray(ds["bottom_level"]).squeeze() > 0
    else:
        wet = None
    cav = int(np.asarray(ds.attrs.get("IsfCav", 0)).reshape(-1)[0]) != 0
    for name, bad in (("misf", lambda a: a > 1), ("top_level", lambda a: a > 1),
                      ("isfdraft", lambda a: a > 0)):
        if cav or name not in ds.variables:
            continue
        a = np.asarray(ds[name]).squeeze()
        cav = bool((bad(a) if wet is None or a.shape != wet.shape else bad(a) & wet).any())
    if cav:
        raise NotImplementedError(
            f"mesh file {path!r} has ice-shelf cavities (IsfCav / misf > 1 / "
            "top_level > 1 / isfdraft > 0 at wet columns); they are not "
            "supported (levels must start at the surface)")


def _check_zps_rule(zw, nlev, e3b, e3u0, e3v0):
    """Load-time consistency check: the model's face thickness at a face's
    bottom level must be the NEMO partial-step rule, the smaller of its two
    cells' thickness there (``e3u(j,i) = min(e3t(j,i), e3t(j,i+1))``; ``v``
    likewise towards ``j+1``).  Interior faces of wet cells only (the crop's
    last column / row has no neighbour cell here; the mesh keeps the file's
    value there).  The kernel uses the file's e3u_0 / e3v_0, not this rule."""
    ref = np.diff(zw)

    bad = 0.0
    for e3f, sl_a, sl_b in ((e3u0, np.s_[:, :-1], np.s_[:, 1:]),
                            (e3v0, np.s_[:-1, :], np.s_[1:, :])):
        na, nb = nlev[sl_a], nlev[sl_b]
        kb = np.minimum(na, nb) - 1
        wet = kb >= 0
        k = np.maximum(kb, 0)
        rule = np.minimum(np.where(na - 1 == k, e3b[sl_a], ref[k]),
                          np.where(nb - 1 == k, e3b[sl_b], ref[k]))
        got = np.take_along_axis(e3f[(slice(None),) + sl_a], k[None], 0)[0]
        bad = max(bad, float(np.max(np.abs(got - rule)[wet] / rule[wet], initial=0.0)))
        above = np.arange(len(ref))[:, None, None] < kb[None]      # wet levels above the bottom
        bad = max(bad, float(np.max(np.abs(e3f[(slice(None),) + sl_a] / ref[:, None, None] - 1)
                                    * above, initial=0.0)))
    if bad > 1e-4:
        raise ValueError(
            "mesh.nemo_grid: cgrid 3-D: e3u_0/e3v_0 differ from the partial-step "
            f"rule min(e3t_0 of the two cells) by {bad:.2e} (relative); only "
            "z-levels with partial steps (zps) are supported")


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


def nemo_cgrid_thickness(zw, kb, e3) -> np.ndarray:
    """Layer thicknesses with partial levels.

    Parameters
    ----------
    zw : array_like (K+1,)
        Reference layer interfaces (m, positive down).
    kb : ndarray of int (ny, nx)
        The one partial level per point, ``-1`` = none (cells: ``nlev - 1``;
        faces: the mesh's ``_nemo_cgrid_kbu`` / ``_nemo_cgrid_kbv``).
    e3 : ndarray (ny, nx)
        Thickness of that level (cells: the bottom thickness; faces:
        ``_nemo_cgrid_e3u`` / ``_nemo_cgrid_e3v``).

    Returns
    -------
    ndarray (K, ny, nx)
        The reference layers, with level ``kb[j, i]`` replaced by ``e3[j, i]``
        where ``kb >= 0``.
    """
    K = len(zw) - 1
    out = np.broadcast_to(np.diff(zw)[:, None, None], (K,) + kb.shape).copy()
    jj, ii = np.nonzero(kb >= 0)
    out[kb[jj, ii], jj, ii] = e3[jj, ii]
    return out


def nemo_cgrid_w_from_continuity(u, v, e2u, e1v, e3u, e3v, area, nlev) -> np.ndarray:
    """Vertical velocity (positive down) from the continuity of the face fluxes.

    The fluxes ``u*e2u*e3u`` and ``v*e1v*e3v`` are integrated upward from
    ``w = 0`` at the sea floor.

    Parameters
    ----------
    u, v : ndarray (K, ny, nx)
        Face velocities (m/s, land 0) on the east / north faces of each cell.
    e2u, e1v : ndarray (ny, nx)
        Face widths (m).
    e3u, e3v : ndarray (K, ny, nx)
        Face thicknesses (m), from :func:`nemo_cgrid_thickness`.
    area : ndarray (ny, nx)
        T-cell area ``e1t*e2t`` (m^2).
    nlev : ndarray of int (ny, nx)
        Wet levels per column.

    Returns
    -------
    ndarray (K+1, ny, nx)
        ``w`` (m/s) on the layer interfaces; the discrete field is exactly
        non-divergent in every cell (the surface value is the residual).
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
