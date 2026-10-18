"""Read and cross-validate the ASCII mesh files of a FESOM 1.4 mesh directory.

A FESOM 1.4 mesh directory holds plain-text files meant for Fortran
list-directed input. This module reads the four that KELP needs:

``nod2d.out``
    The number of 2-D nodes ``N``, then one record per node: its index,
    longitude and latitude (degrees) and a flag (0 interior, 1 coast,
    2 open boundary).
``elem2d.out``
    The number of triangles ``E``, then the three 1-based node indices of each
    triangle.
``nod3d.out``
    The number of 3-D nodes, then one record per 3-D node: its index,
    longitude, latitude, ``z`` in metres (0 at the surface, negative down) and
    a flag. The 3-D nodes are numbered level-major: every 2-D node at the
    surface in 2-D order first, then the wet nodes of each deeper level in
    increasing 2-D order.
``aux3d.out``
    Four parts in a row: ``max_num_layers``; ``nod3D_below_nod2D``, which
    gives, for each 2-D node in turn, ``max_num_layers`` slots holding the
    1-based 3-D node at each level of its column, or a value ``<= 0``
    (normally -999) when the slot is dry; ``nod2D_corresp_to_nod3D``, the
    1-based 2-D node of every 3-D node; and a block about 3-D elements, which
    is not read.

Each file is read as one stream of whitespace-separated numbers, the way a
list-directed read sees it, so how the records are spread over lines does
not matter. The number of values must still match the counts the headers
give, which catches a record with a missing or an extra column. Fortran
``D`` exponents (``1.0D+01``, ``-2.5d-3``) read like ``E`` exponents; a value
written with one is still refused where the file holds an integer (a count,
an index or a flag), since Fortran writes integers without exponents.

The level table comes from ``nod3d.out``: walking its ``z`` values in file
order, every change of value starts a new level; the depths are stored
positive down, the first exactly 0. Before anything is returned, the index
table, the 2-D node of every 3-D node and the 3-D node depths are checked
against each other (see :func:`read_ascii_mesh`). A failed check raises a
``ValueError`` that names the file and the first offending index.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NoReturn

import numpy as np

NOD2D = "nod2d.out"
ELEM2D = "elem2d.out"
NOD3D = "nod3d.out"
AUX3D = "aux3d.out"
MESH_FILES = (NOD2D, ELEM2D, NOD3D, AUX3D)

# largest magnitude at which every integer is exact in float64
_EXACT_INT = float(2 ** 53)

# Fortran double-precision exponent letters, read as E
_D_TO_E = bytes.maketrans(b"Dd", b"EE")

# the bytes that separate numbers (C isspace; also what bytes.split uses)
_BLANK = np.zeros(256, dtype=bool)
_BLANK[list(b" \t\n\r\x0b\x0c")] = True


@dataclass(frozen=True)
class AsciiMesh:
    """A FESOM 1.4 mesh read from its ASCII files.

    Attributes
    ----------
    lon, lat : numpy.ndarray, shape (N,), float64
        Coordinates of the 2-D nodes in degrees, exactly as in ``nod2d.out``
        (no longitude normalisation, no rotation).
    coast : numpy.ndarray, shape (N,), int8
        The flag column of ``nod2d.out``.
    tri : numpy.ndarray, shape (E, 3), int32
        The 0-based node indices of each triangle, in file order.
    levels : numpy.ndarray, shape (K,), float64
        Depth of each level in metres, positive down, strictly increasing,
        ``levels[0] == 0.0``. ``K`` is the number of levels that hold a 3-D
        node; trailing slots of ``aux3d.out`` that are dry everywhere are not
        counted.
    idx : numpy.ndarray, shape (N, K), int64
        ``idx[n, k]`` is the 0-based 3-D node (its position in ``nod3d.out``)
        of 2-D node ``n`` at level ``k``, or -1 when that slot is dry. The wet
        slots of each row come first, and ``idx[:, 0]`` is wet everywhere.
    n3d : int
        The number of 3-D nodes.
    """

    lon: np.ndarray
    lat: np.ndarray
    coast: np.ndarray
    tri: np.ndarray
    levels: np.ndarray
    idx: np.ndarray
    n3d: int


def read_ascii_mesh(meshdir: str | os.PathLike) -> AsciiMesh:
    """Read a FESOM 1.4 ASCII mesh and check that its files agree.

    Parameters
    ----------
    meshdir : str or os.PathLike
        Directory holding ``nod2d.out``, ``elem2d.out``, ``nod3d.out`` and
        ``aux3d.out``.

    Returns
    -------
    AsciiMesh
        The 2-D nodes and triangles (0-based), the level table and the
        ``(N, K)`` table of 3-D node indices.

    Raises
    ------
    FileNotFoundError
        If ``meshdir`` lacks one of the four files.
    ValueError
        If a file cannot be read as a stream of numbers, its length disagrees
        with its header, or the files disagree. The checks, in order: every
        triangle corner is a node of ``nod2d.out``; the first 3-D node lies
        at z = 0 and each new level is deeper than the one before (otherwise
        ``nod3d.out`` is not level-major); every slot of
        ``nod3D_below_nod2D`` is ``<= 0`` (dry) or a 3-D node, and every entry
        of ``nod2D_corresp_to_nod3D`` is a 2-D node; the wet slots of each
        column come first, without gaps, and include the surface slot; every
        3-D node fills exactly one slot; each 3-D node's slot belongs to the
        2-D node ``nod2D_corresp_to_nod3D`` gives it, and lies at the level
        of its ``z`` in ``nod3d.out``; within each level the 3-D nodes follow
        increasing 2-D order. The message names the file and the first
        offending index (1-based, as in the files) and says how to fix it.
    """
    meshdir = Path(meshdir)
    missing = [name for name in MESH_FILES if not (meshdir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"FESOM 1.4 mesh directory {meshdir} lacks {', '.join(missing)}; "
            f"an ASCII mesh needs all of {', '.join(MESH_FILES)} in one directory"
        )

    lon, lat, coast = _read_nod2d(meshdir / NOD2D)
    n2d = lon.size
    tri = _read_elem2d(meshdir / ELEM2D, n2d)
    z3d = _read_nod3d_z(meshdir / NOD3D)
    n3d = z3d.size
    levels, level_of = _level_table(z3d, meshdir)
    max_layers, table, corresp = _read_aux3d(meshdir / AUX3D, n2d, n3d)
    idx = _index_table(table, corresp, z3d, levels, level_of, max_layers, meshdir)
    return AsciiMesh(
        lon=lon,
        lat=lat,
        coast=coast,
        tri=tri,
        levels=levels,
        idx=idx,
        n3d=int(n3d),
    )


# ---------------------------------------------------------------------------
# Token streams
# ---------------------------------------------------------------------------


def _read_numbers(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """All whitespace-separated numbers of a file, in order, as float64.

    Returns the values and, per value, whether its token was written with a
    Fortran ``D`` exponent (read as ``E``).
    """
    raw = path.read_bytes()
    if not raw or raw.isspace():
        # (numpy parses a whitespace-only string as [-1.0])
        raise ValueError(
            f"{path.name}: the file is empty; write the mesh file again from "
            f"the mesh generator ({path})"
        )
    values = _parse(raw)
    if values is not None:
        # (zeros are not touched in memory until written)
        return values, np.zeros(values.size, dtype=bool)
    if b"D" in raw or b"d" in raw:
        values = _parse(raw.translate(_D_TO_E))
        if values is not None:
            return values, _d_exponent_tokens(raw, values.size, path)
    position, token = _first_non_number(raw)
    where = (f"number {position} of the file, {token!r}, is not a number"
             if token is not None else "the file holds something that is not a number")
    raise ValueError(
        f"{path.name}: {where}; the mesh files hold integers and decimal "
        "numbers (with a decimal point, not a comma) separated by blanks or "
        f"line breaks ({path})"
    )


def _parse(raw: bytes) -> np.ndarray | None:
    """The numbers of ``raw`` as float64, or None if a token is not a number."""
    with warnings.catch_warnings():
        # older numpy only warns and returns the numbers before a bad token
        warnings.simplefilter("error", DeprecationWarning)
        try:
            return np.fromstring(raw, dtype=np.float64, sep=" ")
        except (ValueError, DeprecationWarning):
            return None


def _d_exponent_tokens(raw: bytes, n_values: int, path: Path) -> np.ndarray:
    """Per value of ``raw``: does its token contain a ``D`` or ``d``?"""
    buf = np.frombuffer(raw, dtype=np.uint8)
    blank = _BLANK[buf]
    start = ~blank
    start[1:] &= blank[:-1]
    starts = np.flatnonzero(start)
    if starts.size != n_values:
        raise ValueError(
            f"{path.name}: the file splits into {starts.size} blank-separated "
            f"tokens, but {n_values} numbers were read from it; remove unusual "
            f"control characters from the file ({path})"
        )
    d_at = np.flatnonzero((buf == ord("D")) | (buf == ord("d")))
    dexp = np.zeros(n_values, dtype=bool)
    dexp[np.searchsorted(starts, d_at, side="right") - 1] = True
    return dexp


def _first_non_number(raw: bytes) -> tuple[int | None, str | None]:
    """1-based position and text of the first token that is not a number."""
    for position, token in enumerate(raw.split(), start=1):
        text = token.translate(_D_TO_E).decode("latin-1")
        try:
            float(text)
        except ValueError:
            return position, token.decode("latin-1")
        if "_" in text:  # float() accepts digit separators, numpy does not
            return position, token.decode("latin-1")
    return None, None


def _header(values: np.ndarray, dexp: np.ndarray, path: Path, what: str) -> int:
    """The leading count of a file: a positive integer."""
    first = values[0]
    if dexp[0]:
        _refuse_d_exponent(path, f"the first number, {what},", first)
    if not (np.isfinite(first) and first == np.trunc(first) and 1 <= first < _EXACT_INT):
        raise ValueError(
            f"{path.name}: the first number must be {what}, a positive "
            f"integer, but it is {float(first)!r}; check that this is a FESOM 1.4 "
            f"{path.name} ({path})"
        )
    return int(first)


def _records(
    values: np.ndarray, dexp: np.ndarray, path: Path, count: int, layout: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """The ``count`` records after the header as a (count, width) array, and
    the same view of the D-exponent flags."""
    width = len(layout)
    expected = 1 + count * width
    if values.size != expected:
        raise ValueError(
            f"{path.name}: the header gives {count} records of {width} numbers "
            f"({', '.join(layout)}), {expected} numbers with the header, but the "
            f"file holds {values.size}; check that every record has exactly "
            f"{width} numbers and that the header count is right ({path})"
        )
    return values[1:].reshape(count, width), dexp[1:].reshape(count, width)


def _integers(
    values: np.ndarray, dexp: np.ndarray, path: Path, name: Callable[[int], str]
) -> np.ndarray:
    """``values`` as int64; refuses the first value that is not an integer or
    was written with a ``D`` exponent.

    ``name(i)`` describes the position of the flat index ``i``.
    """
    finite = np.isfinite(values)
    bad = ~finite
    bad[finite] = (values[finite] != np.trunc(values[finite])) | (
        np.abs(values[finite]) >= _EXACT_INT)
    bad |= dexp
    if bad.any():
        i = int(np.argmax(bad))
        if dexp[i]:
            _refuse_d_exponent(path, name(i), values[i])
        raise ValueError(
            f"{path.name}: {name(i)} is {float(values[i])!r}, which is not an integer; "
            f"this position of the file holds an integer index or flag, so the "
            f"file is corrupt or its records are shifted ({path})"
        )
    return values.astype(np.int64)


def _refuse_d_exponent(path: Path, what: str, value: float) -> NoReturn:
    raise ValueError(
        f"{path.name}: {what} is written with a Fortran D exponent (it reads as "
        f"{float(value)!r}), but this position of the file holds an integer, "
        f"which Fortran writes without an exponent; the file is corrupt or its "
        f"records are shifted ({path})"
    )


# ---------------------------------------------------------------------------
# The four files
# ---------------------------------------------------------------------------


def _read_nod2d(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values, dexp = _read_numbers(path)
    n2d = _header(values, dexp, path, "the number of 2-D nodes")
    recs, drecs = _records(values, dexp, path, n2d, ("index", "lon", "lat", "flag"))
    lon = np.ascontiguousarray(recs[:, 1])
    lat = np.ascontiguousarray(recs[:, 2])
    bad = ~(np.isfinite(lon) & np.isfinite(lat))
    if bad.any():
        n = int(np.argmax(bad))
        raise ValueError(
            f"{path.name}: 2-D node {n + 1} has a non-finite coordinate "
            f"(lon {float(lon[n])!r}, lat {float(lat[n])!r}); every node needs finite "
            f"coordinates in degrees ({path})"
        )
    flag = _integers(recs[:, 3], drecs[:, 3], path, lambda n: f"the flag of 2-D node {n + 1}")
    info = np.iinfo(np.int8)
    bad = (flag < info.min) | (flag > info.max)
    if bad.any():
        n = int(np.argmax(bad))
        raise ValueError(
            f"{path.name}: the flag of 2-D node {n + 1} is {flag[n]}, outside "
            f"{info.min}..{info.max}; FESOM 1.4 node flags are small codes "
            f"(0 interior, 1 coast, 2 open boundary) ({path})"
        )
    return lon, lat, flag.astype(np.int8)


def _read_elem2d(path: Path, n2d: int) -> np.ndarray:
    values, dexp = _read_numbers(path)
    n_elem = _header(values, dexp, path, "the number of triangles")
    recs, drecs = _records(values, dexp, path, n_elem, ("node 1", "node 2", "node 3"))
    nodes = _integers(
        recs.ravel(), drecs.ravel(), path,
        lambda i: f"corner {i % 3 + 1} of element {i // 3 + 1}",
    ).reshape(n_elem, 3)
    bad = (nodes < 1) | (nodes > n2d)
    if bad.any():
        e, c = np.unravel_index(int(np.argmax(bad)), bad.shape)
        raise ValueError(
            f"{path.name}: element {e + 1} has node {nodes[e, c]} at corner "
            f"{c + 1}, outside 1..{n2d}, the 2-D nodes of {NOD2D}; the corners "
            f"are 1-based indices into {NOD2D} - check that both files belong "
            f"to the same mesh ({path})"
        )
    return (nodes - 1).astype(np.int32)


def _read_nod3d_z(path: Path) -> np.ndarray:
    """The z column of ``nod3d.out`` (metres, negative down)."""
    values, dexp = _read_numbers(path)
    n3d = _header(values, dexp, path, "the number of 3-D nodes")
    recs, _ = _records(values, dexp, path, n3d, ("index", "lon", "lat", "z", "flag"))
    z = np.ascontiguousarray(recs[:, 3])
    bad = ~np.isfinite(z)
    if bad.any():
        j = int(np.argmax(bad))
        raise ValueError(
            f"{path.name}: 3-D node {j + 1} has z = {float(z[j])!r}; every 3-D node "
            f"needs a finite z in metres ({path})"
        )
    return z


def _level_table(z3d: np.ndarray, meshdir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Levels (positive down) from the z values in file order, and the level
    of each 3-D node. Refuses a first node below the surface and a level that
    is not deeper than the one before it."""
    if z3d[0] != 0.0:
        raise ValueError(
            f"{NOD3D}: 3-D node 1 has z = {z3d[0]:g} m, but the first 3-D node "
            f"of a FESOM 1.4 mesh lies at the surface, z = 0; the file must "
            f"list the surface nodes first (level-major order) ({meshdir})"
        )
    new_level = np.empty(z3d.size, dtype=bool)
    new_level[0] = True
    new_level[1:] = z3d[1:] != z3d[:-1]
    starts = np.flatnonzero(new_level)
    levels = -z3d[starts] + 0.0  # + 0.0 turns -0.0 into 0.0
    bad = np.flatnonzero(np.diff(levels) <= 0.0)
    if bad.size:
        k = int(bad[0]) + 1
        j = int(starts[k])
        raise ValueError(
            f"{NOD3D} is not level-major: 3-D node {j + 1} has z = "
            f"{z3d[j]:g} m, not deeper than the level before it (z = "
            f"{z3d[j - 1]:g} m). FESOM 1.4 lists every surface node first and "
            f"then each deeper level in turn, so z only goes down through the "
            f"file; a file ordered column by column must be renumbered "
            f"level by level, together with {AUX3D} ({meshdir})"
        )
    level_of = np.cumsum(new_level) - 1
    return levels, level_of


def _read_aux3d(path: Path, n2d: int, n3d: int) -> tuple[int, np.ndarray, np.ndarray]:
    """``max_num_layers``, the (N, max_num_layers) slot table and the 2-D node
    of every 3-D node (both 1-based, as in the file). The trailing block about
    3-D elements is not interpreted."""
    values, dexp = _read_numbers(path)
    max_layers = _header(values, dexp, path, "max_num_layers")
    n_table = n2d * max_layers
    need = 1 + n_table + n3d
    if values.size < need:
        raise ValueError(
            f"{path.name}: the file holds {values.size} numbers, fewer than the "
            f"{need} its first three parts need: max_num_layers ({max_layers}), "
            f"nod3D_below_nod2D ({n2d} 2-D nodes x {max_layers} slots) and "
            f"nod2D_corresp_to_nod3D ({n3d} 3-D nodes); check that {path.name} "
            f"belongs to the same mesh as {NOD2D} and {NOD3D} ({path})"
        )
    table = _integers(
        values[1:1 + n_table], dexp[1:1 + n_table], path,
        lambda i: (f"nod3D_below_nod2D for 2-D node {i // max_layers + 1}, "
                   f"slot {i % max_layers + 1}"),
    ).reshape(n2d, max_layers)
    corresp = _integers(
        values[1 + n_table:need], dexp[1 + n_table:need], path,
        lambda j: f"nod2D_corresp_to_nod3D for 3-D node {j + 1}",
    )
    return max_layers, table, corresp


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def _index_table(
    table: np.ndarray,
    corresp: np.ndarray,
    z3d: np.ndarray,
    levels: np.ndarray,
    level_of: np.ndarray,
    max_layers: int,
    meshdir: Path,
) -> np.ndarray:
    """Check the slot table against ``nod2D_corresp_to_nod3D`` and the depths
    of ``nod3d.out``; return the 0-based (N, K) index table, -1 = dry."""
    n2d = table.shape[0]
    n3d = z3d.size
    n_levels = levels.size
    same_mesh = (f"check that {AUX3D}, {NOD2D} and {NOD3D} belong to the same "
                 f"mesh ({meshdir})")

    # every slot is dry (<= 0) or a 3-D node
    bad = table > n3d
    if bad.any():
        n, k = np.unravel_index(int(np.argmax(bad)), bad.shape)
        raise ValueError(
            f"{AUX3D}: nod3D_below_nod2D for 2-D node {n + 1}, slot {k + 1} "
            f"is {table[n, k]}, but {NOD3D} has only {n3d} 3-D nodes; a slot "
            f"holds a 1-based 3-D node (1..{n3d}) or a value <= 0 when it is "
            f"dry - {same_mesh}"
        )
    # every 3-D node belongs to a 2-D node
    bad = (corresp < 1) | (corresp > n2d)
    if bad.any():
        j = int(np.argmax(bad))
        raise ValueError(
            f"{AUX3D}: nod2D_corresp_to_nod3D gives 2-D node {corresp[j]} for "
            f"3-D node {j + 1}, outside 1..{n2d}, the 2-D nodes of {NOD2D} - "
            f"{same_mesh}"
        )

    wet = table > 0
    # wet slots come first in each column
    gap = wet[:, 1:] & ~wet[:, :-1]
    if gap.any():
        n, k = np.unravel_index(int(np.argmax(gap)), gap.shape)
        raise ValueError(
            f"{AUX3D}: in nod3D_below_nod2D, 2-D node {n + 1} has 3-D node "
            f"{table[n, k + 1]} in slot {k + 2} below its dry slot {k + 1}; the "
            f"wet slots of a column must come first, without gaps. Check that "
            f"the table is written node by node, {max_layers} slots per node "
            f"({meshdir})"
        )
    empty = ~wet[:, 0]
    if empty.any():
        n = int(np.argmax(empty))
        raise ValueError(
            f"{AUX3D}: in nod3D_below_nod2D, 2-D node {n + 1} has no 3-D node "
            f"at all (its surface slot is {table[n, 0]}); every 2-D node of a "
            f"FESOM 1.4 mesh has at least a surface node - {same_mesh}"
        )

    # every 3-D node fills exactly one slot
    wet_n, wet_k = np.nonzero(wet)  # node-major order
    wet_j = table[wet_n, wet_k] - 1
    count = np.bincount(wet_j, minlength=n3d)
    dup = np.flatnonzero(count > 1)
    if dup.size:
        j = int(dup[0])
        s1, s2 = np.flatnonzero(wet_j == j)[:2]
        raise ValueError(
            f"{AUX3D}: 3-D node {j + 1} fills two slots of nod3D_below_nod2D, "
            f"2-D node {wet_n[s1] + 1} slot {wet_k[s1] + 1} and 2-D node "
            f"{wet_n[s2] + 1} slot {wet_k[s2] + 1}; each 3-D node belongs to "
            f"exactly one slot - {same_mesh}"
        )
    lost = np.flatnonzero(count == 0)
    if lost.size:
        j = int(lost[0])
        raise ValueError(
            f"{AUX3D}: 3-D node {j + 1} of {NOD3D} fills no slot of "
            f"nod3D_below_nod2D; each 3-D node belongs to exactly one slot "
            f"({NOD3D} has {n_levels} levels and max_num_layers is "
            f"{max_layers}) - {same_mesh}"
        )
    owner = np.empty(n3d, dtype=np.int64)
    slot = np.empty(n3d, dtype=np.int64)
    owner[wet_j] = wet_n
    slot[wet_j] = wet_k

    # the two parts of aux3d.out agree
    bad = corresp - 1 != owner
    if bad.any():
        j = int(np.argmax(bad))
        raise ValueError(
            f"{AUX3D}: nod2D_corresp_to_nod3D gives 2-D node {corresp[j]} for "
            f"3-D node {j + 1}, but nod3D_below_nod2D places 3-D node {j + 1} "
            f"under 2-D node {owner[j] + 1} (slot {slot[j] + 1}); the two parts "
            f"of {AUX3D} must agree - {same_mesh}"
        )

    # a slot's level is the level of its 3-D node's z
    bad = level_of != slot
    if bad.any():
        j = int(np.argmax(bad))
        k = int(slot[j])
        at = (f"the level at {levels[k]:g} m" if k < n_levels
              else f"beyond the {n_levels} levels of {NOD3D}")
        raise ValueError(
            f"{NOD3D} and {AUX3D} disagree on the depth of 3-D node {j + 1}: "
            f"{AUX3D} places it in slot {k + 1} of 2-D node {owner[j] + 1} "
            f"({at}), but {NOD3D} gives it z = {z3d[j]:g} m (level "
            f"{level_of[j] + 1}); slot k of a column must hold the column's "
            f"3-D node at the k-th level of {NOD3D} - {same_mesh}"
        )

    # within a level, the 3-D nodes follow increasing 2-D order
    bad = (level_of[1:] == level_of[:-1]) & (owner[1:] <= owner[:-1])
    if bad.any():
        j = int(np.argmax(bad)) + 1
        raise ValueError(
            f"{NOD3D} is not level-major: 3-D node {j + 1} (2-D node "
            f"{owner[j] + 1}) follows 3-D node {j} (2-D node {owner[j - 1] + 1}) "
            f"on the level at {levels[level_of[j]]:g} m. FESOM 1.4 numbers the "
            f"wet nodes of each level in increasing 2-D order; renumber "
            f"{NOD3D} and {AUX3D} together ({meshdir})"
        )

    # every wet slot lies above n_levels (checked above), so the cut drops
    # only dry slots
    return np.where(wet, table - 1, -1)[:, :n_levels].astype(np.int64)
