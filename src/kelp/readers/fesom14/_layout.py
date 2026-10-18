"""FESOM 1.4 data layouts: form detection, gather, static wet masks.

FESOM 1.4 writes a 3-D field in one of two layouts:

- **raw** ``(time, nodes_3d)``: one vector over all 3-D nodes per record,
  numbered level-major (the surface level holds every 2-D node in 2-D order,
  then come the wet nodes of the next level, and so on), so a record has no
  dry slots. :func:`gather_nodes3d` maps it to ``(K, N)`` through the mesh's
  index table ``(N, K)`` (``-1`` = dry).
- **level-wise** ``(time, <depth>, <horizontal>)``: one value per level and
  2-D node; dry slots hold a fill value (``1e30`` in the model's level-wise
  writer, NaN once a ``_FillValue`` is decoded). The horizontal dimension is
  named ``ncells``, ``nodes_2d`` or ``nodes_3d`` (the model's level-wise
  writer keeps the name ``nodes_3d`` for a dimension of length nod2D).

Every value read from either layout passes through ONE validity predicate,
:func:`is_data`: a value is data iff it is finite and ``|x| < 1e29``. A NaN and
a raw ``1e30`` fill are both missing, wherever they occur.

The wet mask of a node column is static: built once (from the index table, or
from the first record with :func:`wet_mask_levelwise`), and every record of
every variable is compared against it with :func:`check_record`. Sizes are not
checked here: comparing the dimensions with the mesh is the reader's job.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import xarray as xr

__all__ = ["FILL_THRESHOLD", "TIME_DIM", "DEPTH_DIMS", "HORIZONTAL_DIMS",
           "is_data", "detect_form", "gather_nodes3d", "wet_mask_levelwise",
           "check_record"]

#: a value with ``|x| >= FILL_THRESHOLD`` is a fill value, not data
FILL_THRESHOLD = 1e29

#: the record dimension of both layouts
TIME_DIM = "time"

#: accepted names of the level dimension of the level-wise layout
DEPTH_DIMS: Tuple[str, ...] = ("depth", "lev", "level")

#: accepted names of the horizontal (2-D node) dimension of the level-wise layout
HORIZONTAL_DIMS: Tuple[str, ...] = ("ncells", "nodes_2d", "nodes_3d")

#: how a missing value is described in messages (FILL_THRESHOLD written out)
_MISSING_TEXT = "NaN, an infinity or a fill value with |x| >= 1e29"


def is_data(x: np.ndarray) -> np.ndarray:
    """Return where *x* holds data: finite and ``|x| < 1e29``.

    The one validity predicate of the FESOM 1.4 reader. A NaN (decoded
    ``_FillValue``), an infinity and a raw fill such as ``1e30`` are all
    "missing".

    Parameters
    ----------
    x : array_like
        Values of any shape.

    Returns
    -------
    numpy.ndarray of bool
        Same shape as *x*; True where the value is data.
    """
    a = np.asarray(x)
    with np.errstate(invalid="ignore"):
        return np.isfinite(a) & (np.abs(a) < FILL_THRESHOLD)


def detect_form(da: xr.DataArray) -> str:
    """Detect the layout of a FESOM 1.4 3-D variable from its dimensions.

    Parameters
    ----------
    da : xarray.DataArray
        One 3-D variable as opened from a file.

    Returns
    -------
    str
        ``"nodes3d"`` for dimensions exactly ``(time, nodes_3d)`` (one
        vector over all 3-D nodes, no depth dimension); ``"levelwise"`` for
        ``(time, <depth>, <horizontal>)`` in this order, with ``<depth>`` in
        :data:`DEPTH_DIMS` and ``<horizontal>`` in :data:`HORIZONTAL_DIMS`.

    Raises
    ------
    ValueError
        For any other set or order of dimensions (a 2-D field, a missing or
        differently named time dimension, a transposed level-wise array,
        an extra dimension, ...). The message names the variable and its
        dimensions. Sizes are not checked here.
    """
    dims = tuple(da.dims)
    if dims == (TIME_DIM, "nodes_3d"):
        return "nodes3d"
    if (len(dims) == 3 and dims[0] == TIME_DIM and dims[1] in DEPTH_DIMS
            and dims[2] in HORIZONTAL_DIMS):
        return "levelwise"
    name = da.name if da.name is not None else "<unnamed>"
    raise ValueError(
        f"fesom14: variable {name!r} has dimensions {dims}, which is not a "
        f"FESOM 1.4 3-D data layout. Expected either ('{TIME_DIM}', "
        f"'nodes_3d') (one vector over all 3-D nodes) or ('{TIME_DIM}', "
        f"<depth>, <horizontal>) in this order, with <depth> one of "
        f"{DEPTH_DIMS} and <horizontal> one of {HORIZONTAL_DIMS}.")


def gather_nodes3d(vec: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Map one raw record over the 3-D nodes to ``(K, N)`` levels x nodes.

    Parameters
    ----------
    vec : array_like, shape (n3d,)
        One record of a ``(time, nodes_3d)`` variable, in the 3-D node order
        of the mesh.
    idx : array_like of int, shape (N, K)
        The mesh's index table: ``idx[n, k]`` is the 0-based 3-D node at
        level ``k`` below 2-D node ``n``, ``-1`` where the slot is dry.

    Returns
    -------
    numpy.ndarray, shape (K, N), C-contiguous
        ``out[k, n] = vec[idx[n, k]]`` on wet slots and NaN on dry slots.
        Floating input keeps its dtype; other input is cast to float64.
        Values are copied as they are: a fill value in a wet slot stays in
        the output for :func:`check_record` to refuse.

    Raises
    ------
    ValueError
        If *vec* is not 1-D, *idx* is not a 2-D integer table, or an entry
        of *idx* lies outside ``[-1, vec.size - 1]``.
    """
    vec = np.asarray(vec)
    idx = np.asarray(idx)
    if vec.ndim != 1:
        raise ValueError(
            f"fesom14: a raw record must be a 1-D (nodes_3d,) vector; got "
            f"shape {vec.shape}")
    if idx.ndim != 2 or not np.issubdtype(idx.dtype, np.integer):
        raise ValueError(
            f"fesom14: the index table must be a 2-D (N, K) integer array; "
            f"got shape {idx.shape}, dtype {idx.dtype}")
    if idx.size:
        lo, hi = int(idx.min()), int(idx.max())
        if lo < -1 or hi >= vec.size:
            raise ValueError(
                f"fesom14: index table out of range for a record of "
                f"{vec.size} 3-D nodes: entries must lie in "
                f"[-1, {vec.size - 1}] (-1 = dry); found min {lo}, max {hi}")
    dtype = vec.dtype if np.issubdtype(vec.dtype, np.floating) else np.float64
    n_nodes, n_levels = idx.shape
    out = np.full((n_levels, n_nodes), np.nan, dtype=dtype)
    wet = idx >= 0
    out.T[wet] = vec[idx[wet]]
    return out


def _first_node_level(sel: np.ndarray) -> Tuple[int, int]:
    """Return ``(node, level)`` of the first True slot, lowest node first."""
    n = int(np.argmax(sel.any(axis=0)))
    k = int(np.argmax(sel[:, n]))
    return n, k


def wet_mask_levelwise(arr_kn: np.ndarray) -> np.ndarray:
    """Build the static wet mask from one level-wise record.

    Parameters
    ----------
    arr_kn : array_like, shape (K, N)
        One record, levels x nodes (decoded or raw: NaN and ``1e30`` count
        alike as missing).

    Returns
    -------
    numpy.ndarray of bool, shape (K, N), C-contiguous
        :func:`is_data` of the record.

    Raises
    ------
    ValueError
        If the record is not 2-D; if a column's data is not a prefix (a
        missing level with data again below it); or if a column has no data
        at all (every node of a FESOM 1.4 mesh is wet at the surface). The
        message names the first such node and level.
    """
    arr = np.asarray(arr_kn)
    if arr.ndim != 2:
        raise ValueError(
            f"fesom14: a level-wise record must be 2-D (K, N) levels x "
            f"nodes; got shape {arr.shape}")
    wet = np.ascontiguousarray(is_data(arr))
    holes = wet[1:] & ~wet[:-1]          # data at k+1 below a missing level k
    if holes.any():
        cols = holes.any(axis=0)
        n = int(np.argmax(cols))
        k_gap = int(np.argmax(~wet[:, n]))
        k_data = k_gap + 1 + int(np.argmax(wet[k_gap + 1:, n]))
        raise ValueError(
            f"fesom14: the data of node {n} is not a prefix of its column: "
            f"no data at level {k_gap} ({_MISSING_TEXT}) but data again at "
            f"level {k_data} (0-based indices; {int(cols.sum())} column(s) "
            f"with such a hole). A column must hold data from the surface "
            f"down to its bottom level and nothing below.")
    empty = ~wet.any(axis=0)
    if empty.any():
        n = int(np.argmax(empty))
        raise ValueError(
            f"fesom14: node {n} has no data at any level, not even at "
            f"level 0 (0-based; {int(empty.sum())} such node(s)); every "
            f"node of a FESOM 1.4 mesh is wet at the surface.")
    return wet


def check_record(arr_kn: np.ndarray, wet: np.ndarray, *, name: str,
                 record: str) -> None:
    """Check one ``(K, N)`` record against the static wet mask.

    Parameters
    ----------
    arr_kn : array_like, shape (K, N)
        One record, levels x nodes, before any zeroing or trimming.
    wet : array_like of bool, shape (K, N)
        The static wet mask of the mesh.
    name : str
        The variable's name, for the message.
    record : str
        The record's label (its instant), for the message.

    Raises
    ------
    ValueError
        If the shapes differ; if a wet slot holds no data (NaN, an infinity
        or a fill value with ``|x| >= 1e29``); or if a dry slot holds data.
        The message names the variable, the record, and the first offending
        node and level (lowest node first, 0-based), with the count.
    """
    arr = np.asarray(arr_kn)
    wet = np.asarray(wet, dtype=bool)
    if arr.ndim != 2 or arr.shape != wet.shape:
        raise ValueError(
            f"fesom14: variable {name!r}, record {record}: shape {arr.shape} "
            f"does not match the (K, N) shape {wet.shape} of the static wet "
            f"mask")
    data = is_data(arr)
    if np.array_equal(data, wet):
        return
    missing = wet & ~data
    extra = data & ~wet
    n_missing = int(missing.sum())
    n_extra = int(extra.sum())
    rule = ("Every record must hold data in exactly the wet slots of the "
            "static mask, which is fixed once per mesh; a record whose bottom "
            "moved or that carries gaps cannot be read.")
    if n_missing:
        n, k = _first_node_level(missing)
        also = (f"; also {n_extra} dry slot(s) with data" if n_extra else "")
        raise ValueError(
            f"fesom14: variable {name!r}, record {record}: no data at node "
            f"{n}, level {k} (0-based), a wet slot of the static mask "
            f"({_MISSING_TEXT}; {n_missing} wet slot(s) without data in this "
            f"record{also}). {rule}")
    n, k = _first_node_level(extra)
    raise ValueError(
        f"fesom14: variable {name!r}, record {record}: data ({float(arr[k, n]):g}) "
        f"at node {n}, level {k} (0-based), a dry slot of the static mask "
        f"({n_extra} dry slot(s) with data in this record). {rule}")
