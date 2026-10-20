"""Depth-linear horizontal velocity: data points, last usable point, bed,
bracket.

A depth-linear u/v field (``VelocityField.uv_vertical`` ``"level"`` or
``"layer_linear"``) is evaluated linearly in depth between data points
``p[0..M-1]`` (positive-down metres) of a z-level mesh (a sigma mesh:
in the sigma fraction, below):

- ``"level"``: ``p`` = the kept levels, ``|mesh.sigma|`` (``M = K``);
- ``"layer_linear"``: ``p`` = the layer centres ``(z_k + z_{k+1}) / 2`` of
  the kept levels (``M = K - 1``).

The table (:class:`DepthPoints`) has one column per u/v carrier: per node
for node-layout fields, per element for element-layout fields
(``"layer_linear"`` only; ``"level"`` data sit on nodes).  Each column has
a last usable point ``last`` (the deepest point it may be read at) and a
``bed``, its physical bed depth: a column is WET at depth ``z`` while
``z <= bed`` (:func:`uv_wet`), and a wet column below ``p[last]`` (a
partial bottom layer, a ``max_levels`` cut, a declared prefix that ends
above the bed) reads its last usable point.

Per node ``i`` the last usable point is

``last_i = max(0, min(M-1, max({k : lower_k <= depth_i} U {0}), D_i))``

with ``lower_k = p[k]`` for level data and ``lower_k = z_{k+1}`` (the
layer's lower level: the :func:`kelp.numerics.vertical.find_layer` rule, so
a partial bottom layer is never read) for layer data, and ``D_i`` the
reader's declaration ``mesh.metadata["uv_data_last"][i]`` when present
(the last finite point of the field's own vertical axis).  ``-1`` declares
the surface only: point 0 is the only one read, and the record check
requires it to be finite — a column with no data at all must not be a mesh
node, or must carry a finite surface value.  Infinite bed depths follow
the formula as written: ``+inf`` keeps every point (``last_i = M-1``, then
capped by ``D_i``), ``-inf`` the surface point only.  A NaN ``depth_i``
compares false with every ``lower_k``, so ``last_i = 0`` (set explicitly:
a sorted search would place the NaN after the last point).  A node's bed
is ``mesh.depth[i]``, the bed the end-of-step depth clamp uses.

Per element ``e`` there is no bathymetry rule: the reader DECLARES the
element's last wet layer, ``mesh.metadata["uv_data_last_elem"][e]`` (int
``(E,)``, ``>= -1``; geometric already: a layer counts when its lower
level is at or above the element's bed), and its bed,
``mesh.metadata["elem_bed"][e]`` (float ``(E,)``, positive down);
``last_e = max(0, min(M-1, D_e))``.  Without either of the two an element
table is refused (the FESOM2 reader sets both from the mesh diagnostic's
``nlevels`` or ``zbar_e_bottom``).

On a sigma mesh (``vertical_coord == "sigma"``, FESOM-C) the coordinate
is the sigma fraction ``s = z / H`` of the particle's column (0 at the
surface, 1 at the bed; ``H`` the column height at the particle, which the
integrators compute for each stage at its own position and time,
``integrators._sigma_col_H``, and hand the interpolators as
``clip(z / H, 0, 1)``, ``integrators._sigma_frac_for_uv``) instead of the
depth, and only ``"layer_linear"``
is served, on either layout: ``p`` = the layer centres
``c_k = 1 - (sigma_k + sigma_{k+1}) / 2`` of ``mesh.sigma`` (sigma 1 at
the surface, 0 at the bed: interfaces at ``s_k = 1 - sigma_k``, the
convention of :func:`kelp.numerics.vertical_sigma.find_layer_sigma`).
Every layer of every column holds water, so nothing else is read (no
declaration, no element bottom, no bathymetry rule): ``last = M - 1``
for every column and ``bed = +inf``, wet at every ``s`` (a NaN ``s``,
which the kernels map to ``+inf``, reads the bottom centre).  The record
check then wants every layer of every column finite, and the cached
table is valid while ``mesh.sigma`` is the very object it was built from.

Above ``p[0]`` and below ``p[last_i]`` the value of that point is used (no
extrapolation); in between, the two bracketing points are blended.  The
read never touches a point below ``last_i``, so dry or fill values there
are never used, not even multiplied by zero.

The table is built lazily per mesh, placement and layout
(:func:`depth_points`) and cached in ``mesh._uv_depth_points``; an entry
is valid while ``mesh.sigma``, the bed array (``mesh.depth`` for nodes,
``mesh.metadata["elem_bed"]`` for elements) and the declaration
(``mesh.metadata["uv_data_last"]`` / ``["uv_data_last_elem"]``) are the
very objects it was built from (identity, never ``id()``), so a
``max_levels`` cut or a new bed or declaration rebuilds it.  Replace
those arrays to change them; an in-place write is not detected.  The
engine therefore takes a SNAPSHOT of the declaration at first use: the
table is built from an owned, read-only copy, which replaces the reader's
array in the metadata — the declaration in effect is exactly what the
table was built from, and no array or view the reader still holds can
change it (writing into the snapshot raises; assign a new array to
declare another prefix).  A copied or unpickled mesh (whose snapshot
numpy returns writeable) gets a new snapshot, validated again, on first
use.  The table keeps its own read-only copy of the bed.

The record check (:func:`check_depth_linear_record`) refuses a field whose
u or v is non-finite at a point ``<= last`` of a column (a node or an
element, by the field's ``uv_layout``) and stamps a passing field with the
table object (``vel._dl_token``); :func:`is_checked` compares by identity,
so a field checked on another mesh or before a rebuild counts as
unchecked.
"""
from __future__ import annotations

import weakref
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

#: u/v layouts a table serves: one column per node or per element
_DL_LAYOUTS = ("node", "element")

#: the reader's declaration of the last usable point, per layout
_DECL_KEYS = {"node": "uv_data_last", "element": "uv_data_last_elem"}

#: the element bed depths (float (E,), positive down), set by the reader
_ELEM_BED_KEY = "elem_bed"

#: columns per block of the record scan (one block = one prange task; the
#: inner loop walks a contiguous row segment of the block)
_SCAN_BLOCK = 64


@dataclass(frozen=True, eq=False)
class DepthPoints:
    """Depth-linear data points of one mesh, placement and layout.

    A column is a node (node layout) or an element (element layout).

    Attributes
    ----------
    p : np.ndarray
        Data-point depths, float64 ``(M,)``, positive down, non-decreasing;
        on a sigma mesh the layer centres in the sigma fraction ``s``
        (0 at the surface, 1 at the bed).  Read-only.
    last : np.ndarray
        Index of the last usable data point per column, int32 ``(C,)``, in
        ``0 .. M-1`` (``M-1`` for every column of a sigma mesh).
        Read-only.
    bed : np.ndarray
        Physical bed depth per column, float64 ``(C,)``, positive down: a
        copy of ``mesh.depth`` (nodes) or of ``mesh.metadata["elem_bed"]``
        (elements), never changed by a ``max_levels`` cut; ``+inf`` for
        every column of a sigma mesh (wet at every ``s``).  A column is
        wet at depth ``z`` while ``z <= bed`` (:func:`uv_wet`).
        Read-only.

    Notes
    -----
    ``eq=False``: a ``DepthPoints`` object is a token (record checks
    compare it by identity), so equality and hashing are by identity.
    """
    p: np.ndarray
    last: np.ndarray
    bed: np.ndarray


@dataclass(frozen=True, eq=False)
class _DepthPointsEntry:
    """Cache entry: strong references to the inputs and the table.

    ``depth`` is the bed array the table was built from: ``mesh.depth``
    (nodes) or ``mesh.metadata["elem_bed"]`` (elements); ``depth`` and
    ``decl`` are ``None`` for a sigma table, which reads neither.
    """
    sigma: object
    depth: object
    decl: object
    points: DepthPoints


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

# every declaration snapshot this process made, by id; a freed snapshot
# drops out, so a reused id never matches.  Reuse then does not depend on
# whose cache holds the snapshot: two shallow copies of a mesh (one shared
# metadata dict) reuse each other's instead of replacing it on every lookup
_SNAPSHOTS: "weakref.WeakValueDictionary[int, np.ndarray]" = (
    weakref.WeakValueDictionary())


def _snapshot_declaration(decl):
    """The declaration a table is built from: an owned, read-only copy.

    ``decl`` itself when it already is an engine snapshot that is still
    read-only (the other placement's, this placement's before a
    ``max_levels`` cut, or a shallow copy's of the same mesh); otherwise
    ``np.array(decl, copy=True)`` (dtype kept, validated by the caller),
    frozen before anyone else holds it, so no earlier view or base array
    of the reader's can write into it.  ``None`` (no declaration) stays
    ``None``.
    """
    if decl is None:
        return None
    if (isinstance(decl, np.ndarray) and not decl.flags.writeable
            and _SNAPSHOTS.get(id(decl)) is decl):
        return decl
    snap = np.array(decl, copy=True)
    snap.setflags(write=False)
    _SNAPSHOTS[id(snap)] = snap
    return snap


def _build_depth_points(sigma, depth, decl, placement: str, n_cols: int,
                        layout: str = "node",
                        vertical_coord: str = "zlevel") -> DepthPoints:
    """Build and validate the table of one placement and layout.

    ``depth`` is the bed array of the columns (``mesh.depth`` for nodes,
    ``mesh.metadata["elem_bed"]`` for elements), ``n_cols`` their number.
    ``decl`` is the engine's snapshot of the declaration
    (:func:`_snapshot_declaration`) or ``None`` (nodes only); it is
    validated here and nothing of it is kept but its values in ``last``.
    On a sigma mesh (``vertical_coord == "sigma"``, ``"layer_linear"``
    only) ``depth`` and ``decl`` are not read: the points are the layer
    centres in the sigma fraction, every column reads every layer and is
    wet at every ``s``.
    Raises the ``ValueError`` documented in :func:`depth_points`.
    """
    if vertical_coord == "sigma":
        # interfaces at s_k = 1 - sigma_k (find_layer_sigma's convention:
        # sigma 1 at the surface, 0 at the bed); centres 1 - (sigma_k +
        # sigma_{k+1}) / 2; every layer of every column holds water
        sig = np.asarray(sigma, dtype=np.float64)
        if sig.ndim != 1 or sig.size < 2:
            raise ValueError(
                f"depth-linear u/v ({placement!r}) needs a 3-D sigma mesh "
                f"with >= 2 interfaces in mesh.sigma, got shape {sig.shape}")
        p = 1.0 - 0.5 * (sig[:-1] + sig[1:])
        last = np.full(n_cols, p.size - 1, dtype=np.int32)
        bed = np.full(n_cols, np.inf, dtype=np.float64)
        p.setflags(write=False)
        last.setflags(write=False)
        bed.setflags(write=False)
        return DepthPoints(p=p, last=last, bed=bed)
    z = np.abs(np.asarray(sigma, dtype=np.float64))   # == mesh.z_levels
    need = 1 if placement == "level" else 2
    if z.ndim != 1 or z.size < need:
        raise ValueError(
            f"depth-linear u/v ({placement!r}) needs a 3-D z-level mesh with "
            f">= {need} levels in mesh.sigma, got shape {z.shape}")
    # the messages name the first offending level, never the whole table
    # (a global mesh has ~50-100 levels)
    nonfin = np.flatnonzero(~np.isfinite(z))
    if nonfin.size:
        i = int(nonfin[0])
        raise ValueError(
            f"depth-linear u/v ({placement!r}) needs finite level depths "
            f"|mesh.sigma|: {z.size} levels, level {i} is {float(z[i])!r}")
    down = np.flatnonzero(np.diff(z) < 0.0)
    if down.size:
        i = int(down[0])
        raise ValueError(
            f"depth-linear u/v ({placement!r}) needs non-decreasing level "
            f"depths |mesh.sigma|: {z.size} levels, level {i + 1} "
            f"({float(z[i + 1]):g} m) is shallower than level {i} "
            f"({float(z[i]):g} m)")
    if placement == "level":
        p = z.copy()
        lower = p
    else:
        p = 0.5 * (z[:-1] + z[1:])
        lower = z[1:]
    m = p.size
    d = np.asarray(depth, dtype=np.float64)
    if layout == "node":
        if d.shape != (n_cols,):
            raise ValueError(
                f"depth-linear u/v: mesh.depth has shape {d.shape}, expected "
                f"({n_cols},)")
        # number of lower_k <= depth_i (lower is non-decreasing), minus
        # one; NaN sorts last in searchsorted, the rule wants it unmatched
        geo = np.searchsorted(lower, d, side="right").astype(np.int64) - 1
        geo[np.isnan(d)] = 0
        last = np.minimum(geo, m - 1)
    else:
        if d.shape != (n_cols,):
            raise ValueError(
                f"depth-linear u/v: mesh.metadata['{_ELEM_BED_KEY}'] has "
                f"shape {d.shape}, expected ({n_cols},) (one bed depth per "
                "element, positive down)")
        # the declaration alone caps an element (it is geometric already)
        last = np.full(n_cols, m - 1, dtype=np.int64)
    key = _DECL_KEYS[layout]
    if decl is not None:
        dl = np.asarray(decl)
        if dl.shape != (n_cols,) or not np.issubdtype(dl.dtype, np.integer):
            raise ValueError(
                f"mesh.metadata['{key}'] must be an integer array of "
                f"shape ({n_cols},), got dtype {dl.dtype} shape {dl.shape}")
        if dl.size and int(dl.min()) < -1:
            raise ValueError(
                f"mesh.metadata['{key}'] must be >= -1 (-1 = surface "
                "only: point 0 is read and must be finite), got minimum "
                f"{int(dl.min())}")
        last = np.minimum(last, dl.astype(np.int64))
    last = np.maximum(last, 0).astype(np.int32)
    bed = np.array(d, dtype=np.float64, copy=True)
    p.setflags(write=False)
    last.setflags(write=False)
    bed.setflags(write=False)
    return DepthPoints(p=p, last=last, bed=bed)


def _missing_elem_bottoms(mesh: Mesh, no_decl: bool,
                          no_bed: bool) -> ValueError:
    """The refusal of an element table without the reader's bottoms."""
    missing = [repr(k) for k, gone in ((_DECL_KEYS["element"], no_decl),
                                       (_ELEM_BED_KEY, no_bed)) if gone]
    meshpath = (mesh.metadata or {}).get("meshpath")
    where = f"mesh {meshpath!r}" if meshpath else "this mesh"
    return ValueError(
        "depth-linear u/v on element layout needs the element bottoms "
        f"mesh.metadata['{_DECL_KEYS['element']}'] (each element's last "
        f"wet layer, int (E,)) and mesh.metadata['{_ELEM_BED_KEY}'] (its "
        "bed depth, float (E,), positive down), which the FESOM2 reader "
        "declares from the mesh diagnostic's nlevels or zbar_e_bottom "
        f"(the run's full fesom.mesh.diag.nc); {where} has no "
        + " and no ".join(missing))


def refuse_elem_cavities(mesh: Mesh) -> None:
    """Refuse depth-linear u/v on element layout under ice-shelf cavities.

    A cavity is ``mesh.metadata["elem_ulevels"]`` (the FESOM2 reader's
    ``ulevels``) above 1 anywhere: the column starts below the surface,
    which the element table does not represent yet.  Called by every
    element table build and by tracking before the writer opens.

    Raises
    ------
    ValueError
        Naming the mesh file (``mesh.metadata["meshpath"]``), the number
        of cavity elements and the first one.
    """
    md = mesh.metadata or {}
    ulevels = md.get("elem_ulevels")
    if ulevels is None:
        return
    ulevels = np.asarray(ulevels)
    cav = np.flatnonzero(ulevels > 1)
    if cav.size:
        meshpath = md.get("meshpath")
        where = f"mesh {meshpath!r}" if meshpath else "this mesh"
        raise ValueError(
            "ice-shelf cavities are not supported by mode.uv_vertical: "
            f"linear on element data yet: {where} has ulevels > 1 at "
            f"{cav.size} element(s) (first: element {int(cav[0])}, "
            f"ulevels {int(ulevels[cav[0]])}); run without the option")


def depth_points(mesh: Mesh, placement: str,
                 layout: str = "node") -> DepthPoints:
    """The depth-linear data-point table of ``mesh`` for ``placement``.

    Parameters
    ----------
    mesh : Mesh
        A z-level mesh (``vertical_coord == "zlevel"``) with its level
        depths in ``mesh.sigma`` (the engine reads ``|mesh.sigma|``, i.e.
        ``mesh.z_levels``).  Node layout: bed depths in ``mesh.depth``;
        optional reader declaration ``mesh.metadata["uv_data_last"]`` (int
        ``(N,)``; ``-1`` = surface only, whose value must be finite).
        Element layout: the reader's declaration
        ``mesh.metadata["uv_data_last_elem"]`` (int ``(E,)``, ``>= -1``,
        the element's last wet layer) and bed depths
        ``mesh.metadata["elem_bed"]`` (float ``(E,)``, positive down), both
        required.  Or a sigma mesh (``vertical_coord == "sigma"``) with
        its sigma interfaces in ``mesh.sigma`` (1 at the surface, 0 at the
        bed), either layout, nothing else read: ``p`` = the layer centres
        in the sigma fraction ``s``, ``last = M - 1`` and ``bed = +inf``
        for every column.
    placement : str
        ``"level"`` or ``"layer_linear"`` (element layout and sigma
        meshes: ``"layer_linear"`` only).
    layout : str, optional
        ``"node"`` (default: one column per node) or ``"element"`` (one
        column per element) — the ``uv_layout`` of the fields the table
        serves.

    Returns
    -------
    DepthPoints
        Cached in ``mesh._uv_depth_points`` per ``(placement, layout)``:
        the same object while ``mesh.sigma``, the bed array
        (``mesh.depth`` / ``mesh.metadata["elem_bed"]``) and the
        declaration are the same objects; rebuilt (a new object) when any
        of them is replaced.  The node and the element table of a mesh
        are independent: a new ``mesh.depth`` rebuilds only the node
        table, a new ``elem_bed`` only the element table.  A sigma table
        reads no bed and no declaration: it is the same object while
        ``mesh.sigma`` is.

    Notes
    -----
    A build takes the declaration's snapshot (an owned, read-only copy;
    one that already is an engine snapshot, still read-only, is reused —
    the other placement's, an earlier build's before a ``max_levels`` cut,
    or a shallow copy's of the mesh), validates it, builds from it, and
    installs it in ``mesh.metadata["uv_data_last"]`` (element layout:
    ``["uv_data_last_elem"]``) in place of the reader's array: the
    declaration in effect is exactly what the table was built from, and a
    write into the reader's array, its base or any view of it after that
    changes nothing (assign a new array to declare another prefix).  A
    cache entry counts only while that snapshot is still read-only, so a
    deep-copied or unpickled mesh (numpy returns the copy writeable) gets
    a new snapshot, validated again, on first use.  The bed is copied into
    the table (``DepthPoints.bed``), never snapshotted in the metadata.

    Raises
    ------
    ValueError
        ``placement`` is ``"layer"`` or unknown, ``layout`` is unknown,
        ``"level"`` on element layout or on a sigma mesh, the mesh is
        neither z-level nor sigma, a sigma mesh has fewer than 2
        interfaces, the level table is too short, non-finite or not
        increasing, the depth, the element bed or the declaration has the
        wrong shape or type, an element table lacks the declaration or
        the element bed (the message names both keys and ``nlevels``), or
        the mesh has ice-shelf cavities (:func:`refuse_elem_cavities`).
        A sigma table is refused for its placement, layout and number of
        interfaces only.
    """
    if placement not in _DL_PLACEMENTS:
        raise ValueError(
            f"depth_points: placement must be one of {_DL_PLACEMENTS}, got "
            f"{placement!r} (a 'layer' field is piecewise constant in depth "
            "and has no depth-linear table)")
    if layout not in _DL_LAYOUTS:
        raise ValueError(
            f"depth_points: layout must be one of {_DL_LAYOUTS}, got "
            f"{layout!r}")
    if layout == "element" and placement != "layer_linear":
        raise ValueError(
            "depth-linear u/v on element layout is 'layer_linear' only "
            f"(layer-centred data), got {placement!r}; 'level' data sit on "
            "the nodes' levels")
    coord = mesh.vertical_coord
    if coord == "sigma":
        if placement != "layer_linear":
            raise ValueError(
                "depth-linear u/v on a sigma mesh (vertical_coord 'sigma') "
                "is 'layer_linear' only (layer-centred data, read in the "
                f"sigma fraction s = z / H), got {placement!r}")
    elif coord != "zlevel":
        raise ValueError(
            f"depth-linear u/v ({placement!r}) needs a z-level or sigma "
            f"mesh (vertical_coord 'zlevel' or 'sigma'), got {coord!r}")
    sigma = mesh.sigma
    meta = mesh.metadata or {}
    decl_key = _DECL_KEYS[layout]
    # a sigma column is wet in every layer: no declaration, no bed (and no
    # element bottoms on element layout) is read
    decl = None if coord == "sigma" else meta.get(decl_key)
    if coord == "sigma":
        depth = None
        n_cols = mesh.n_nodes if layout == "node" else mesh.n_elements
    elif layout == "node":
        depth = mesh.depth
        n_cols = mesh.n_nodes
    else:
        depth = meta.get(_ELEM_BED_KEY)
        if decl is None or depth is None:
            raise _missing_elem_bottoms(mesh, decl is None, depth is None)
        n_cols = mesh.n_elements
    key = (placement, layout)
    cache = mesh._uv_depth_points
    if cache:
        entry = cache.get(key)
        # a WRITEABLE declaration is not the frozen snapshot the entry was
        # built from, even when identical: a deep copy or an unpickled mesh
        # gets a new, writeable array (numpy drops the flag) that still
        # matches its cache entry by identity — rebuild from its values
        if (entry is not None and entry.sigma is sigma
                and entry.depth is depth and entry.decl is decl
                and not (isinstance(decl, np.ndarray)
                         and decl.flags.writeable)):
            return entry.points
    if layout == "element" and coord == "zlevel":
        refuse_elem_cavities(mesh)
    snap = _snapshot_declaration(decl)
    points = _build_depth_points(sigma, depth, snap, placement, n_cols,
                                 layout, coord)
    if snap is not decl:
        # the declaration in effect is what the table was built from.  A
        # shallow copy of the mesh shares this dict and then sees the
        # snapshot too: equal values, so its next lookup rebuilds its own
        # table (its entry names the replaced object) with the same prefix
        mesh.metadata[decl_key] = snap
    # copy-on-write: a shallow copy of the mesh (dataclasses.replace,
    # copy.copy) shares the dict object; never mutate it in place
    new_cache = dict(cache or {})
    new_cache[key] = _DepthPointsEntry(sigma, depth, snap, points)
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
    and ``k + 1`` with ``k + 1 <= last_i``.

    Two coincident points (a repeated level): a depth exactly at them reads
    the DEEPER one (the larger index), e.g. ``p = [0, 10, 10, 20]``,
    ``z = 10`` gives ``(2, -1.0)``; the shallower one is the end of the
    bracket above, reached only as a limit.  Except at the surface end: a
    depth at or above ``p[0]`` reads point 0 even when point 1 coincides
    with it (``p = [0, 0, 10]``, ``z = 0`` gives ``(0, -1.0)``).

    A NaN ``z`` gives ``k = last_i - 1`` with a NaN ``alpha`` when
    ``last_i > 0`` (a NaN value through :func:`uv_at`, still never past
    ``last_i``; a degenerate last spacing reads point ``last_i - 1``
    directly instead), as :func:`kelp.numerics.vertical._w_bracket` does,
    and ``(0, -1.0)`` (the surface point, read directly) when
    ``last_i == 0``.  The interpolation kernels never pass a NaN: they map
    a NaN particle depth to ``+inf`` first, the layer path's reading of a
    NaN depth, which :func:`uv_wet` makes dry wherever the bed is finite
    (a column it leaves wet reads its last point directly,
    ``(last_i, -1.0)``).

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


@njit(cache=True)
def uv_wet(bed: np.ndarray, col: int, z: float) -> bool:
    """True when column ``col`` holds water at depth ``z``: ``z <= bed[col]``.

    ``bed`` is :attr:`DepthPoints.bed` (positive down).  A column is wet
    down to and including its bed; a wet column below ``p[last]`` reads
    its last usable point (:func:`uv_bracket`).  A NaN ``z`` or a NaN bed
    compares false: dry.  An infinite bed is wet at every depth, ``+inf``
    included (the interpolation kernels map a NaN particle depth to
    ``+inf`` before this test: dry wherever the bed is finite).
    """
    return z <= bed[col]


# ---------------------------------------------------------------------------
# The record check
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _first_nonfinite_per_node(f, last, out):
    """``out[i]`` = the first point ``k <= last[i]`` where ``f[k, i]`` is
    non-finite, else ``-1`` (``i`` a column: a node or an element).  Blocks
    of columns run in parallel; each block walks contiguous row
    segments."""
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


def _check_components(vel, dp: DepthPoints, placement: str, record: str,
                      layout: str = "node",
                      sigma_mesh: bool = False) -> None:
    """Shape and finiteness of ``vel.u`` / ``vel.v`` at points ``<= last``.

    ``layout`` names the columns in the messages: ``"node"`` or
    ``"element"``; ``sigma_mesh`` (a sigma mesh's table) names a point by
    its sigma fraction ``s`` instead of its depth.
    """
    m = dp.p.size
    n = dp.last.size
    col = "element" if layout == "element" else "node"
    if sigma_mesh:
        why = "every layer of a sigma column"
    else:
        why = "down to its last usable point: " + (
            "declared last wet layer and kept levels" if col == "element"
            else "bed, kept levels and declared data prefix")
    rec = f" in record {record}" if record else ""
    for comp in ("u", "v"):
        f = getattr(vel, comp)
        shape = getattr(f, "shape", None)
        if shape is None or len(shape) != 2:
            raise ValueError(
                f"check_depth_linear_record: {comp} of a {placement!r} field "
                f"must be 2-D (points, {col}s), got shape {shape}")
        if shape[1] != n:
            raise ValueError(
                f"check_depth_linear_record: {comp} has {shape[1]} {col}s, "
                f"the mesh has {n} ({col} layout expected)")
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
            at = (f"s = {dp.p[k]:g}" if sigma_mesh
                  else f"depth {dp.p[k]:g} m")
            raise ValueError(
                f"check_depth_linear_record: {comp} = {float(f[k, i])!r} is "
                f"non-finite at {col} {i}, point {k} ({at}){rec}; a "
                f"{placement!r} field must be finite at points "
                f"0..{int(dp.last[i])} of this {col} ({why}; {bad.size} "
                f"{col}(s) affected)")


def _layout_of(vel) -> str:
    """The table layout of ``vel``: its ``uv_layout``; a field-like object
    without one counts as element layout, the default of
    :class:`~kelp.core.io.VelocityField` (a reader's raw-record check
    states its layout)."""
    return getattr(vel, "uv_layout", "element")


def check_depth_linear_record(vel, mesh: Mesh, *, record: str = "") -> None:
    """Validate a depth-linear field for ``mesh`` and stamp it as checked.

    Refuses (``ValueError`` naming node or element, point, component and
    ``record``) a u or v value that is non-finite at a point ``<= last``
    of its column; points below ``last`` may hold anything (NaN, fill
    values; on a sigma mesh ``last`` is the bottom layer of every column,
    so every layer must be finite).  ``w`` is never checked, and a finite
    fill value inside the prefix is the reader's business (readers mask
    their fill values).  On
    success sets ``vel._dl_token = depth_points(mesh, vel.uv_vertical,
    vel.uv_layout)``: a node-layout field is checked against the node
    table, an element-layout field against the element table.

    A :class:`~kelp.core.io.LazyLerpField` is checked through its parents
    ``vel_a`` / ``vel_b`` (each skipped when already checked) without
    blending anything; its own token is set only when both parents then
    hold the identical token object.

    Parameters
    ----------
    vel : VelocityField or LazyLerpField
        A ``"level"`` or ``"layer_linear"`` field with u/v of shape
        ``(M, C)`` (``M`` = the table's points, ``C`` = mesh nodes for
        ``uv_layout="node"``, mesh elements for ``"element"``).  A
        field-like object without ``uv_layout`` counts as element layout.
    mesh : Mesh
        The z-level or sigma mesh the field is evaluated on.
    record : str, optional
        Names the record in the message (e.g. ``"time index 2"``).

    Raises
    ------
    ValueError
        A ``"layer"`` field, a mesh neither z-level nor sigma, a
        ``"level"`` field on a sigma mesh, a z-level mesh without the
        element bottoms (element layout), a shape mismatch, or a
        non-finite value at a usable point.
    """
    layout = _layout_of(vel)
    dp = depth_points(mesh, vel.uv_vertical, layout)
    if isinstance(vel, LazyLerpField):
        for parent in (vel.vel_a, vel.vel_b):
            if not is_checked(parent, mesh):
                check_depth_linear_record(parent, mesh, record=record)
        if vel.vel_a._dl_token is dp and vel.vel_b._dl_token is dp:
            vel._dl_token = dp
        return
    _check_components(vel, dp, vel.uv_vertical, record, layout,
                      mesh.vertical_coord == "sigma")
    vel._dl_token = dp


def is_checked(vel, mesh: Mesh) -> bool:
    """True when ``vel`` passed the record check for ``mesh``'s current table.

    ``vel._dl_token is depth_points(mesh, vel.uv_vertical, vel.uv_layout)``;
    a field that was never stamped (every ``"layer"`` field) answers
    ``False`` without building a table.  Never blends a lazy field.
    """
    tok = getattr(vel, "_dl_token", None)
    if tok is None:
        return False
    return tok is depth_points(mesh, vel.uv_vertical, _layout_of(vel))


__all__ = [
    "DepthPoints",
    "depth_points",
    "uv_bracket",
    "uv_at",
    "uv_wet",
    "check_depth_linear_record",
    "is_checked",
]
