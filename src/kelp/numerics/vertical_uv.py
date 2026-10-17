"""Depth-linear horizontal velocity: data points, last usable point, bracket.

A depth-linear u/v field (``VelocityField.uv_vertical`` ``"level"`` or
``"layer_linear"``) is evaluated linearly in depth between data points
``p[0..M-1]`` (positive-down metres) of a z-level mesh:

- ``"level"``: ``p`` = the kept levels, ``|mesh.sigma|`` (``M = K``);
- ``"layer_linear"``: ``p`` = the layer centres ``(z_k + z_{k+1}) / 2`` of
  the kept levels (``M = K - 1``).

Per node ``i`` the last usable point is

``last_i = max(0, min(M-1, max({k : lower_k <= depth_i} U {0}), D_i))``

with ``lower_k = p[k]`` for level data and ``lower_k = z_{k+1}`` (the
layer's lower level: the :func:`kelp.numerics.vertical.find_layer` rule, so
a partial bottom layer is never read) for layer data, and ``D_i`` the
reader's declaration ``mesh.metadata["uv_data_last"][i]`` when present
(the last finite point of the field's own vertical axis).  ``-1`` declares
the surface only: point 0 is the only one read, and the record check
requires it to be finite — a column with no data at all must not be a mesh
node, or must carry a finite surface value.  A non-finite ``depth_i``
gives ``last_i = 0``.

Above ``p[0]`` and below ``p[last_i]`` the value of that point is used (no
extrapolation); in between, the two bracketing points are blended.  The
read never touches a point below ``last_i``, so dry or fill values there
are never used, not even multiplied by zero.

The table is built lazily per mesh and placement (:func:`depth_points`) and
cached in ``mesh._uv_depth_points``; an entry is valid while
``mesh.sigma``, ``mesh.depth`` and ``mesh.metadata["uv_data_last"]`` are
the very objects it was built from (identity, never ``id()``), so a
``max_levels`` cut or a new depth or declaration rebuilds it.  Replace
those arrays to change them; an in-place write is not detected.

The record check (:func:`check_depth_linear_record`) refuses a field whose
u or v is non-finite at a point ``<= last_i`` and stamps a passing field
with the table object (``vel._dl_token``); :func:`is_checked` compares by
identity, so a field checked on another mesh or before a rebuild counts as
unchecked.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:  # pragma: no cover — local dev fallback
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def prange(*args, **kwargs):
        return range(*args)

from ..core.io import LazyLerpField
from ..core.mesh import Mesh

#: placements this module serves ("layer" is the piecewise-constant path)
_DL_PLACEMENTS = ("level", "layer_linear")

#: columns per block of the record scan (one block = one prange task; the
#: inner loop walks a contiguous row segment of the block)
_SCAN_BLOCK = 64


@dataclass(frozen=True, eq=False)
class DepthPoints:
    """Depth-linear data points of one mesh and placement.

    Attributes
    ----------
    p : np.ndarray
        Data-point depths, float64 ``(M,)``, positive down, non-decreasing.
        Read-only.
    last : np.ndarray
        Index of the last usable data point per node, int32 ``(N,)``, in
        ``0 .. M-1``.  Read-only.

    Notes
    -----
    ``eq=False``: a ``DepthPoints`` object is a token (record checks
    compare it by identity), so equality and hashing are by identity.
    """
    p: np.ndarray
    last: np.ndarray


@dataclass(frozen=True, eq=False)
class _DepthPointsEntry:
    """Cache entry: strong references to the inputs and the table."""
    sigma: object
    depth: object
    decl: object
    points: DepthPoints


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

def _build_depth_points(sigma, depth, decl, placement: str,
                        n_nodes: int) -> DepthPoints:
    z = np.abs(np.asarray(sigma, dtype=np.float64))   # == mesh.z_levels
    need = 1 if placement == "level" else 2
    if z.ndim != 1 or z.size < need:
        raise ValueError(
            f"depth-linear u/v ({placement!r}) needs a 3-D z-level mesh with "
            f">= {need} levels in mesh.sigma, got shape {z.shape}")
    if not np.all(np.isfinite(z)) or np.any(np.diff(z) < 0.0):
        raise ValueError(
            f"depth-linear u/v ({placement!r}) needs finite, increasing "
            f"level depths |mesh.sigma|, got {z.tolist()}")
    if placement == "level":
        p = z.copy()
        lower = p
    else:
        p = 0.5 * (z[:-1] + z[1:])
        lower = z[1:]
    m = p.size
    d = np.asarray(depth, dtype=np.float64)
    if d.shape != (n_nodes,):
        raise ValueError(
            f"depth-linear u/v: mesh.depth has shape {d.shape}, expected "
            f"({n_nodes},)")
    # number of lower_k <= depth_i (lower is non-decreasing), minus one;
    # NaN sorts last in searchsorted, the rule wants it unmatched
    geo = np.searchsorted(lower, d, side="right").astype(np.int64) - 1
    geo[np.isnan(d)] = 0
    last = np.minimum(geo, m - 1)
    if decl is not None:
        dl = np.asarray(decl)
        if dl.shape != (n_nodes,) or not np.issubdtype(dl.dtype, np.integer):
            raise ValueError(
                "mesh.metadata['uv_data_last'] must be an integer array of "
                f"shape ({n_nodes},), got dtype {dl.dtype} shape {dl.shape}")
        if dl.size and int(dl.min()) < -1:
            raise ValueError(
                "mesh.metadata['uv_data_last'] must be >= -1 (-1 = surface "
                "only: point 0 is read and must be finite), got minimum "
                f"{int(dl.min())}")
        last = np.minimum(last, dl.astype(np.int64))
    last = np.maximum(last, 0).astype(np.int32)
    p.setflags(write=False)
    last.setflags(write=False)
    return DepthPoints(p=p, last=last)


def depth_points(mesh: Mesh, placement: str) -> DepthPoints:
    """The depth-linear data-point table of ``mesh`` for ``placement``.

    Parameters
    ----------
    mesh : Mesh
        A z-level mesh (``vertical_coord == "zlevel"``) with its level
        depths in ``mesh.sigma`` (the engine reads ``|mesh.sigma|``, i.e.
        ``mesh.z_levels``) and bed depths in ``mesh.depth``; optional
        reader declaration ``mesh.metadata["uv_data_last"]`` (int ``(N,)``;
        ``-1`` = surface only, whose value must be finite).
    placement : str
        ``"level"`` or ``"layer_linear"``.

    Returns
    -------
    DepthPoints
        Cached in ``mesh._uv_depth_points``: the same object while
        ``mesh.sigma``, ``mesh.depth`` and the declaration are the same
        objects; rebuilt (a new object) when any of them is replaced.

    Raises
    ------
    ValueError
        ``placement`` is ``"layer"`` or unknown, the mesh is not z-level,
        the level table is too short, non-finite or not increasing, or the
        depth or the declaration has the wrong shape or type.
    """
    if placement not in _DL_PLACEMENTS:
        raise ValueError(
            f"depth_points: placement must be one of {_DL_PLACEMENTS}, got "
            f"{placement!r} (a 'layer' field is piecewise constant in depth "
            "and has no depth-linear table)")
    if mesh.vertical_coord != "zlevel":
        raise ValueError(
            f"depth-linear u/v ({placement!r}) needs a z-level mesh "
            f"(vertical_coord 'zlevel'), got {mesh.vertical_coord!r}")
    sigma = mesh.sigma
    depth = mesh.depth
    decl = (mesh.metadata or {}).get("uv_data_last")
    cache = mesh._uv_depth_points
    if cache:
        entry = cache.get(placement)
        if (entry is not None and entry.sigma is sigma
                and entry.depth is depth and entry.decl is decl):
            return entry.points
    points = _build_depth_points(sigma, depth, decl, placement, mesh.n_nodes)
    # copy-on-write: a shallow copy of the mesh (dataclasses.replace,
    # copy.copy) shares the dict object; never mutate it in place
    new_cache = dict(cache or {})
    new_cache[placement] = _DepthPointsEntry(sigma, depth, decl, points)
    mesh._uv_depth_points = new_cache
    return points


# ---------------------------------------------------------------------------
# Bracket and read (njit)
# ---------------------------------------------------------------------------

@njit(cache=True)
def uv_bracket(p: np.ndarray, last_i: int, z: float):
    """Vertical bracket ``(k, alpha)`` of depth ``z`` in ``p[0..last_i]``.

    The depth is clipped first, ``z_i = max(p[0], min(z, p[last_i]))``.
    ``alpha < 0`` (``-1.0``) means "read point ``k`` directly": at or above
    ``p[0]``, at or below ``p[last_i]``, exactly at a point, or a degenerate
    spacing (< 1e-12 m).  Otherwise ``0 < alpha < 1`` blends points ``k``
    and ``k + 1`` with ``k + 1 <= last_i``.  A NaN ``z`` gives
    ``(last_i - 1, NaN)`` (a NaN value, still never past ``last_i``), as
    :func:`kelp.numerics.vertical._w_bracket` does.

    Parameters
    ----------
    p : np.ndarray
        Data-point depths, float64 ``(M,)``, non-decreasing.
    last_i : int
        The node's last usable point, ``0 <= last_i <= M-1``.
    z : float
        Depth, metres, positive down.

    Returns
    -------
    tuple of (int, float)
    """
    li = int(last_i)
    if li <= 0:
        return 0, -1.0
    zc = z
    if zc > p[li]:
        zc = p[li]
    if zc < p[0]:
        zc = p[0]
    if zc <= p[0]:
        return 0, -1.0
    if zc >= p[li]:
        return li, -1.0
    k = li - 1
    for j in range(li):
        if zc < p[j + 1]:
            k = j
            break
    dz = p[k + 1] - p[k]
    if dz < 1e-12:
        return k, -1.0
    alpha = (zc - p[k]) / dz
    if alpha <= 0.0:
        return k, -1.0
    if alpha >= 1.0:
        return k + 1, -1.0
    return k, alpha


@njit(cache=True)
def uv_at(f: np.ndarray, node: int, k: int, alpha: float) -> float:
    """Value of ``f`` (``(M, N)``) at node ``node`` and bracket ``(k, alpha)``.

    ``alpha < 0`` reads ``f[k, node]`` only — ``f[k + 1]`` is never
    touched; otherwise ``(1 - alpha) * f[k] + alpha * f[k + 1]``.  The
    result is float64 (float32 data is widened first).
    """
    a = float(f[k, node])
    if alpha < 0.0:
        return a
    return (1.0 - alpha) * a + alpha * float(f[k + 1, node])


# ---------------------------------------------------------------------------
# The record check
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _first_nonfinite_per_node(f, last, out):
    """``out[i]`` = the first point ``k <= last[i]`` where ``f[k, i]`` is
    non-finite, else ``-1``.  Blocks of columns run in parallel; each block
    walks contiguous row segments."""
    n = f.shape[1]
    nb = (n + _SCAN_BLOCK - 1) // _SCAN_BLOCK
    for b in prange(nb):
        i0 = b * _SCAN_BLOCK
        i1 = min(n, i0 + _SCAN_BLOCK)
        kmax = -1
        for i in range(i0, i1):
            out[i] = -1
            if last[i] > kmax:
                kmax = last[i]
        for k in range(kmax + 1):
            for i in range(i0, i1):
                if out[i] < 0 and k <= last[i]:
                    if not np.isfinite(f[k, i]):
                        out[i] = k


def _check_components(vel, dp: DepthPoints, placement: str,
                      record: str) -> None:
    """Shape and finiteness of ``vel.u`` / ``vel.v`` at points ``<= last``."""
    m = dp.p.size
    n = dp.last.size
    rec = f" in record {record}" if record else ""
    for comp in ("u", "v"):
        f = getattr(vel, comp)
        shape = getattr(f, "shape", None)
        if shape is None or len(shape) != 2:
            raise ValueError(
                f"check_depth_linear_record: {comp} of a {placement!r} field "
                f"must be 2-D (points, nodes), got shape {shape}")
        if shape[1] != n:
            raise ValueError(
                f"check_depth_linear_record: {comp} has {shape[1]} nodes, the "
                f"mesh has {n} (node layout expected)")
        if shape[0] != m:
            raise ValueError(
                f"check_depth_linear_record: {comp} has {shape[0]} rows, the "
                f"{placement!r} table of this mesh has {m} points (levels "
                "and fields must be cut to the same depth, e.g. by max_levels)")
        out = np.empty(n, dtype=np.int64)
        _first_nonfinite_per_node(f, dp.last, out)
        bad = np.flatnonzero(out >= 0)
        if bad.size:
            i = int(bad[0])
            k = int(out[i])
            raise ValueError(
                f"check_depth_linear_record: {comp} = {float(f[k, i])!r} is "
                f"non-finite at node {i}, point {k} (depth {dp.p[k]:g} m)"
                f"{rec}; a {placement!r} field must be finite at points "
                f"0..{int(dp.last[i])} of this node (down to its last usable "
                f"point: bed, kept levels and declared data prefix; "
                f"{bad.size} node(s) affected)")


def check_depth_linear_record(vel, mesh: Mesh, *, record: str = "") -> None:
    """Validate a depth-linear field for ``mesh`` and stamp it as checked.

    Refuses (``ValueError`` naming node, point, component and ``record``) a
    u or v value that is non-finite at a point ``<= last[i]``; points below
    ``last[i]`` may hold anything (NaN, fill values).  ``w`` is never
    checked, and a finite fill value inside the prefix is the reader's
    business (readers mask their fill values).  On success sets
    ``vel._dl_token = depth_points(mesh, vel.uv_vertical)``.

    A :class:`~kelp.core.io.LazyLerpField` is checked through its parents
    ``vel_a`` / ``vel_b`` (each skipped when already checked) without
    blending anything; its own token is set only when both parents then
    hold the identical token object.

    Parameters
    ----------
    vel : VelocityField or LazyLerpField
        A ``"level"`` or ``"layer_linear"`` field with u/v of shape
        ``(M, N)`` (``M`` = the table's points, ``N`` = mesh nodes).
    mesh : Mesh
        The z-level mesh the field is evaluated on.
    record : str, optional
        Names the record in the message (e.g. ``"time index 2"``).

    Raises
    ------
    ValueError
        A ``"layer"`` field, a non-z-level mesh, a shape mismatch, or a
        non-finite value at a usable point.
    """
    dp = depth_points(mesh, vel.uv_vertical)
    if isinstance(vel, LazyLerpField):
        for parent in (vel.vel_a, vel.vel_b):
            if not is_checked(parent, mesh):
                check_depth_linear_record(parent, mesh, record=record)
        if vel.vel_a._dl_token is dp and vel.vel_b._dl_token is dp:
            vel._dl_token = dp
        return
    _check_components(vel, dp, vel.uv_vertical, record)
    vel._dl_token = dp


def is_checked(vel, mesh: Mesh) -> bool:
    """True when ``vel`` passed the record check for ``mesh``'s current table.

    ``vel._dl_token is depth_points(mesh, vel.uv_vertical)``; a field that
    was never stamped (every ``"layer"`` field) answers ``False`` without
    building a table.  Never blends a lazy field.
    """
    tok = getattr(vel, "_dl_token", None)
    if tok is None:
        return False
    return tok is depth_points(mesh, vel.uv_vertical)


__all__ = [
    "DepthPoints",
    "depth_points",
    "uv_bracket",
    "uv_at",
    "check_depth_linear_record",
    "is_checked",
]
