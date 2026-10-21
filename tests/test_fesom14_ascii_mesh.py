"""FESOM 1.4 ASCII mesh reader (``kelp.readers.fesom14._ascii_mesh``).

The fixture is a hand-written five-node mesh: three triangles, four levels
(0, 10, 25, 50 m), column lengths 4, 3, 2, 3, 1 and ``max_num_layers = 6``
(two trailing slots that are dry everywhere). Its 3-D nodes are numbered
level-major: the five surface nodes first, then the wet nodes of each deeper
level in increasing 2-D order. The expected index table below is written by
hand from that rule, not computed by the reader's logic.

Every refusal must be a ``ValueError`` whose message names the file and the
first offending index.
"""
from __future__ import annotations

import copy
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from kelp.readers.fesom14._ascii_mesh import AsciiMesh, read_ascii_mesh

# ---------------------------------------------------------------------------
# The hand-written mesh
# ---------------------------------------------------------------------------

#: nod2d.out records: index, lon, lat, coast flag
NOD2D = [
    (1, 10.0, 50.0, 1),
    (2, 11.0, 50.0, 0),
    (3, 10.5, 51.0, 0),
    (4, 11.5, 51.0, 0),
    (5, 12.0, 50.0, 1),
]

#: elem2d.out records: three 1-based node indices per triangle
ELEM2D = [
    (1, 2, 3),
    (2, 4, 3),
    (2, 5, 4),
]

#: nod3d.out records, level-major: index, lon, lat, z (negative down), flag
NOD3D = [
    # level 1, z = 0: every 2-D node
    (1, 10.0, 50.0, 0.0, 0),
    (2, 11.0, 50.0, 0.0, 0),
    (3, 10.5, 51.0, 0.0, 0),
    (4, 11.5, 51.0, 0.0, 0),
    (5, 12.0, 50.0, 0.0, 0),
    # level 2, z = -10: 2-D nodes 1, 2, 3, 4
    (6, 10.0, 50.0, -10.0, 0),
    (7, 11.0, 50.0, -10.0, 0),
    (8, 10.5, 51.0, -10.0, 0),
    (9, 11.5, 51.0, -10.0, 0),
    # level 3, z = -25: 2-D nodes 1, 2, 4
    (10, 10.0, 50.0, -25.0, 0),
    (11, 11.0, 50.0, -25.0, 0),
    (12, 11.5, 51.0, -25.0, 0),
    # level 4, z = -50: 2-D node 1
    (13, 10.0, 50.0, -50.0, 0),
]

MAX_NUM_LAYERS = 6

#: aux3d.out part 2, nod3D_below_nod2D: one row per 2-D node, 1-based 3-D
#: node per slot, -999 = dry
AUX_TABLE = [
    [1, 6, 10, 13, -999, -999],
    [2, 7, 11, -999, -999, -999],
    [3, 8, -999, -999, -999, -999],
    [4, 9, 12, -999, -999, -999],
    [5, -999, -999, -999, -999, -999],
]

#: aux3d.out part 3, nod2D_corresp_to_nod3D: the 2-D node of each 3-D node
AUX_CORRESP = [1, 2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 4, 1]

#: aux3d.out part 4, elem2D_corresp_to_elem3D: a dummy, never interpreted
AUX_ELEM_DUMMY = [1, 1, 2, 2, 3, 3, 3]

#: expected (N, K) index table: 0-based 3-D node, -1 = dry
EXPECTED_IDX = np.array(
    [
        [0, 5, 9, 12],
        [1, 6, 10, -1],
        [2, 7, -1, -1],
        [3, 8, 11, -1],
        [4, -1, -1, -1],
    ],
    dtype=np.int64,
)
EXPECTED_LEVELS = np.array([0.0, 10.0, 25.0, 50.0])
EXPECTED_TRI = np.array([[0, 1, 2], [1, 3, 2], [1, 4, 3]], dtype=np.int32)


def _fixture() -> dict:
    """A deep copy of the hand mesh, for tests that mutate it."""
    return copy.deepcopy(
        {
            "nod2d": NOD2D,
            "elem2d": ELEM2D,
            "nod3d": NOD3D,
            "max_num_layers": MAX_NUM_LAYERS,
            "table": AUX_TABLE,
            "corresp": AUX_CORRESP,
            "elem_dummy": AUX_ELEM_DUMMY,
        }
    )


def _fmt(v) -> str:
    return f"{v:.6f}" if isinstance(v, float) else str(v)


def _fmt_d(v) -> str:
    """Floats with Fortran D exponents, upper case for positive values and
    lower case for negative ones; everything else unchanged."""
    if not isinstance(v, float):
        return str(v)
    return f"{v:.7E}".replace("E", "D" if v >= 0 else "d")


def _file_tokens(fx: dict, fmt=_fmt) -> dict[str, list[list[str]]]:
    """Each file as a list of records, each record a list of tokens."""
    aux = [[str(fx["max_num_layers"])]]
    aux += [[str(v)] for row in fx["table"] for v in row]  # one per line
    aux += [[str(v)] for v in fx["corresp"]]
    aux += [[str(v)] for v in fx["elem_dummy"]]
    return {
        "nod2d.out": [[str(len(fx["nod2d"]))]]
        + [[fmt(v) for v in rec] for rec in fx["nod2d"]],
        "elem2d.out": [[str(len(fx["elem2d"]))]]
        + [[fmt(v) for v in rec] for rec in fx["elem2d"]],
        "nod3d.out": [[str(len(fx["nod3d"]))]]
        + [[fmt(v) for v in rec] for rec in fx["nod3d"]],
        "aux3d.out": aux,
    }


def _fmt_repr(v) -> str:
    """Floats written with all their digits (for values like -1e-12)."""
    return repr(v) if isinstance(v, float) else str(v)


def _write(meshdir: Path, fx: dict | None = None, fmt=_fmt) -> Path:
    """Write the four files the way FESOM 1.4 meshes are laid out."""
    meshdir.mkdir(parents=True, exist_ok=True)
    for name, records in _file_tokens(fx or _fixture(), fmt).items():
        text = "".join(" ".join(rec) + "\n" for rec in records)
        (meshdir / name).write_text(text)
    return meshdir


def _write_reflowed(meshdir: Path, fx: dict | None = None, fmt=_fmt) -> Path:
    """Write the same numbers ignoring record boundaries: several numbers per
    line, records split across lines, tabs, blank lines, CRLF endings."""
    meshdir.mkdir(parents=True, exist_ok=True)
    for i, (name, records) in enumerate(_file_tokens(fx or _fixture(), fmt).items()):
        tokens = [t for rec in records for t in rec]
        per_line = 3 + 2 * i  # 3, 5, 7, 9 numbers per line, cutting records
        lines = [
            "\t ".join(tokens[j:j + per_line])
            for j in range(0, len(tokens), per_line)
        ]
        newline = "\r\n" if name == "nod3d.out" else "\n"
        text = ("  " + newline + newline).join(lines) + "   " + newline + newline
        (meshdir / name).write_text(text, newline="")
    return meshdir


def _refusal(tmp_path: Path, fx: dict) -> str:
    with pytest.raises(ValueError) as info:
        read_ascii_mesh(_write(tmp_path / "mesh", fx))
    return str(info.value)


def _assert_mesh_equal(a: AsciiMesh, b: AsciiMesh) -> None:
    for f in dataclasses.fields(AsciiMesh):
        va, vb = getattr(a, f.name), getattr(b, f.name)
        if isinstance(va, np.ndarray):
            assert va.dtype == vb.dtype, f.name
            np.testing.assert_array_equal(va, vb, err_msg=f.name)
        else:
            assert va == vb, f.name


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def test_reads_hand_mesh(tmp_path):
    m = read_ascii_mesh(_write(tmp_path / "mesh"))

    assert m.lon.dtype == np.float64 and m.lat.dtype == np.float64
    np.testing.assert_array_equal(m.lon, [10.0, 11.0, 10.5, 11.5, 12.0])
    np.testing.assert_array_equal(m.lat, [50.0, 50.0, 51.0, 51.0, 50.0])
    assert m.coast.dtype == np.int8
    np.testing.assert_array_equal(m.coast, [1, 0, 0, 0, 1])

    assert m.tri.dtype == np.int32 and m.tri.shape == (3, 3)
    np.testing.assert_array_equal(m.tri, EXPECTED_TRI)

    assert m.levels.dtype == np.float64
    np.testing.assert_array_equal(m.levels, EXPECTED_LEVELS)
    assert m.levels[0] == 0.0 and not np.signbit(m.levels[0])  # not -0.0

    # K = 4 levels with a wet node; the two trailing table slots are cut
    assert m.idx.dtype == np.int64 and m.idx.shape == (5, 4)
    np.testing.assert_array_equal(m.idx, EXPECTED_IDX)
    assert m.n3d == 13 and isinstance(m.n3d, int)

    with pytest.raises(dataclasses.FrozenInstanceError):
        m.n3d = 14  # type: ignore[misc]


def test_accepts_str_path(tmp_path):
    d = _write(tmp_path / "mesh")
    _assert_mesh_equal(read_ascii_mesh(str(d)), read_ascii_mesh(d))


def test_list_directed_tokens(tmp_path):
    """Fortran list-directed reads see a stream of numbers: the same numbers
    laid out differently over lines give the same mesh."""
    ref = read_ascii_mesh(_write(tmp_path / "lines"))
    flowed_dir = _write_reflowed(tmp_path / "flowed")
    # the reflowed files really do cut records across lines
    assert (flowed_dir / "nod2d.out").read_text().splitlines()[0].split() == [
        "5", "1", "10.000000"]
    assert "\r\n" in (flowed_dir / "nod3d.out").read_bytes().decode()
    _assert_mesh_equal(read_ascii_mesh(flowed_dir), ref)


def test_d_exponents_read_like_e(tmp_path):
    """Fortran D exponents (either case) read as E exponents: a reflowed mesh
    written with them gives the same mesh as the plain one."""
    ref = read_ascii_mesh(_write(tmp_path / "plain"))
    d = _write_reflowed(tmp_path / "dexp", fmt=_fmt_d)
    nod3d = (d / "nod3d.out").read_text()
    assert "1.0000000D+01" in nod3d and "-2.5000000d+01" in nod3d
    assert "5.0000000D+01" in (d / "nod2d.out").read_text()
    _assert_mesh_equal(read_ascii_mesh(d), ref)


def test_zero_is_dry(tmp_path):
    """A slot <= 0 is dry, whatever the negative value: 0 and -1 as well as -999."""
    fx = _fixture()
    fx["table"][2][2:] = [0, -1, 0, -999]
    fx["table"][4][1:] = [0, 0, 0, 0, 0]
    m = read_ascii_mesh(_write(tmp_path / "mesh", fx))
    np.testing.assert_array_equal(m.idx, EXPECTED_IDX)


def test_elem3d_block_is_skipped(tmp_path):
    """The trailing elem2D_corresp_to_elem3D block is never parsed: any
    length, including none, and any content read the same mesh."""
    ref = read_ascii_mesh(_write(tmp_path / "ref"))
    for n, dummy in enumerate(([], [7] * 40, ["end", "#", "1,5", "x"])):
        fx = _fixture()
        fx["elem_dummy"] = dummy
        _assert_mesh_equal(read_ascii_mesh(_write(tmp_path / f"m{n}", fx)), ref)


@pytest.mark.parametrize("chunk", [1, 2, 3, 7])
def test_aux3d_cut_across_chunks(tmp_path, monkeypatch, chunk):
    """The end of the needed values is found by scanning aux3d.out in chunks;
    tokens cut by a chunk boundary are counted once. With tiny chunks every
    boundary case occurs, and a non-numeric elem3D block right after the
    needed values shows where the cut fell."""
    import kelp.readers.fesom14._ascii_mesh as am

    ref = read_ascii_mesh(_write(tmp_path / "ref"))
    fx = _fixture()
    fx["elem_dummy"] = ["end", "x"]
    d = _write_reflowed(tmp_path / "mesh", fx)
    monkeypatch.setattr(am, "_CHUNK", chunk)
    _assert_mesh_equal(read_ascii_mesh(d), ref)


def test_max_num_layers_equal_to_k(tmp_path):
    """No trailing dry slot: max_num_layers equals the number of levels."""
    fx = _fixture()
    fx["max_num_layers"] = 4
    fx["table"] = [row[:4] for row in fx["table"]]
    m = read_ascii_mesh(_write(tmp_path / "mesh", fx))
    np.testing.assert_array_equal(m.idx, EXPECTED_IDX)
    np.testing.assert_array_equal(m.levels, EXPECTED_LEVELS)


def test_coast_flag_open_boundary(tmp_path):
    """Flag 2 (open boundary) is kept as it is."""
    fx = _fixture()
    fx["nod2d"][4] = (5, 12.0, 50.0, 2)
    m = read_ascii_mesh(_write(tmp_path / "mesh", fx))
    np.testing.assert_array_equal(m.coast, [1, 0, 0, 0, 2])
    assert m.coast.dtype == np.int8


@pytest.mark.parametrize("z0", [-1e-12, 5e-7, -1e-6])
def test_surface_tolerance(tmp_path, z0):
    """A surface written within 1e-6 m of 0 (the same value for every surface
    node) is the surface: levels[0] is exactly +0.0."""
    fx = _fixture()
    for j in range(1, 6):
        _set_z(fx, j, z0)
    m = read_ascii_mesh(_write(tmp_path / "mesh", fx, fmt=_fmt_repr))
    assert m.levels[0] == 0.0 and not np.signbit(m.levels[0])
    np.testing.assert_array_equal(m.levels, EXPECTED_LEVELS)
    np.testing.assert_array_equal(m.idx, EXPECTED_IDX)


def test_integer_fields_accept_integral_spellings(tmp_path):
    """An integer field takes any number whose value is an integer: +7, 7.0,
    7E0, 7D0, 0.7D+01; the mesh is the same as with plain integers."""
    ref = read_ascii_mesh(_write(tmp_path / "ref"))
    fx = _fixture()
    _set_table(fx, 2, 2, "0.7D+01")
    _set_table(fx, 1, 1, "+1")
    _set_table(fx, 3, 3, "-999.0")
    _set_corresp(fx, 4, "4E0")
    _set_corresp(fx, 9, "4d0")
    _set_elem(fx, 1, 2, "2.0")
    fx["nod2d"][2] = (3, 10.5, 51.0, "0D0")
    fx["max_num_layers"] = "6.0"
    d = _write(tmp_path / "mesh", fx)
    elem = (d / "elem2d.out").read_text()
    (d / "elem2d.out").write_text("+3.0E0" + elem[1:])  # the header
    _assert_mesh_equal(read_ascii_mesh(d), ref)


# ---------------------------------------------------------------------------
# Refusals: each names the file and the first offending index (1-based)
# ---------------------------------------------------------------------------


def _set_table(fx, node, slot, value):  # 1-based node and slot
    fx["table"][node - 1][slot - 1] = value


def _set_corresp(fx, node3d, value):  # 1-based 3-D node
    fx["corresp"][node3d - 1] = value


def _set_elem(fx, elem, corner, value):  # 1-based element and corner
    rec = list(fx["elem2d"][elem - 1])
    rec[corner - 1] = value
    fx["elem2d"][elem - 1] = tuple(rec)


def _set_z(fx, node3d, z):  # 1-based 3-D node
    rec = list(fx["nod3d"][node3d - 1])
    rec[3] = z
    fx["nod3d"][node3d - 1] = tuple(rec)


@pytest.mark.parametrize(
    "mutate, needles",
    [
        # a table entry past nod3D (13)
        (lambda fx: _set_table(fx, 2, 4, 14),
         ["aux3d.out", "2-D node 2", "slot 4", "is 14", "1..13"]),
        # two out-of-range entries: the first in node-major order is named
        (lambda fx: (_set_table(fx, 4, 5, 99), _set_table(fx, 3, 6, 20)),
         ["aux3d.out", "2-D node 3", "slot 6", "is 20", "1..13"]),
        # a 2-D node past nod2D (5) in nod2D_corresp_to_nod3D
        (lambda fx: _set_corresp(fx, 9, 6),
         ["aux3d.out", "3-D node 9", "2-D node 6", "outside 1..5"]),
        (lambda fx: _set_corresp(fx, 4, 0),
         ["aux3d.out", "3-D node 4", "2-D node 0", "outside 1..5"]),
        # a triangle corner past nod2D, or 0 (the file is 1-based)
        (lambda fx: _set_elem(fx, 2, 3, 6),
         ["elem2d.out", "element 2", "node 6", "corner 3", "outside 1..5"]),
        (lambda fx: _set_elem(fx, 3, 1, 0),
         ["elem2d.out", "element 3", "node 0", "corner 1", "outside 1..5"]),
    ],
)
def test_refuses_out_of_range_index(tmp_path, mutate, needles):
    fx = _fixture()
    mutate(fx)
    msg = _refusal(tmp_path, fx)
    for needle in needles:
        assert needle in msg, (needle, msg)


def test_refuses_duplicate_3d_node(tmp_path):
    fx = _fixture()
    _set_table(fx, 5, 2, 7)  # 3-D node 7 is already 2-D node 2, slot 2
    msg = _refusal(tmp_path, fx)
    for needle in ("aux3d.out", "3-D node 7", "two slots", "2-D node 2 slot 2",
                   "2-D node 5 slot 2"):
        assert needle in msg, (needle, msg)


def test_refuses_missing_3d_node(tmp_path):
    fx = _fixture()
    _set_table(fx, 1, 4, -999)  # 3-D node 13 loses its only slot
    msg = _refusal(tmp_path, fx)
    for needle in ("aux3d.out", "3-D node 13", "nod3d.out", "fills no slot"):
        assert needle in msg, (needle, msg)


def test_refuses_gap_in_column(tmp_path):
    fx = _fixture()
    fx["table"][2] = [3, -999, 8, -999, -999, -999]  # 2-D node 3: dry slot 2, wet slot 3
    msg = _refusal(tmp_path, fx)
    for needle in ("aux3d.out", "2-D node 3", "slot 3", "dry slot 2", "without gaps"):
        assert needle in msg, (needle, msg)


def test_refuses_empty_column(tmp_path):
    fx = _fixture()
    _set_table(fx, 5, 1, -999)  # 2-D node 5 loses its surface node
    msg = _refusal(tmp_path, fx)
    for needle in ("aux3d.out", "2-D node 5", "no 3-D node at all"):
        assert needle in msg, (needle, msg)


def test_refuses_wrong_corresp(tmp_path):
    fx = _fixture()
    _set_corresp(fx, 8, 4)  # 3-D node 8 is below 2-D node 3, not 4
    _set_corresp(fx, 11, 1)  # a later one is not named
    msg = _refusal(tmp_path, fx)
    for needle in ("aux3d.out", "3-D node 8", "under 2-D node 3", "gives 2-D node 4",
                   "must agree"):
        assert needle in msg, (needle, msg)
    assert "3-D node 11" not in msg


@pytest.mark.parametrize(
    "node3d, z, needles",
    [
        # still four monotonic levels, but node 9 now sits at 25 m while the
        # table puts it in slot 2 (10 m)
        (9, -25.0, ["depth of 3-D node 9", "slot 2", "2-D node 4", "z = -25.0 m",
                    "ice-shelf cavities"]),
        # the last surface node moved to 10 m: the second level starts one early
        (5, -10.0, ["depth of 3-D node 5", "slot 1", "2-D node 5", "z = -10.0 m",
                    "top wet node of 2-D node 5", "not supported"]),
        # the first node is not at the surface
        (1, -5.0, ["3-D node 1", "z = -5.0 m", "at the surface"]),
        # just outside the 1e-6 m surface tolerance
        (1, -2e-6, ["3-D node 1", "z = -2e-06 m", "at the surface"]),
        # the last node of the second level written with a rounding error: it
        # starts a level of its own, so its slot (2) no longer matches
        (9, -10.000001, ["depth of 3-D node 9", "slot 2", "z = -10.000001 m",
                         "differ only by rounding"]),
        # the deepest node moved up to 25 m: nod3d.out has three levels, but
        # the table still puts it in slot 4
        (13, -25.0, ["depth of 3-D node 13", "slot 4", "beyond the 3 levels"]),
    ],
)
def test_refuses_wrong_z(tmp_path, node3d, z, needles):
    fx = _fixture()
    _set_z(fx, node3d, z)
    msg = _refusal(tmp_path, fx)
    assert "nod3d.out" in msg, msg
    for needle in needles:
        assert needle in msg, (needle, msg)


@pytest.mark.parametrize(
    "zs, needles",
    [
        # 3-D node 7 in the middle of the second level
        ({7: -10.000001}, ["3-D node 8", "z = -10.0 m", "z = -10.000001 m"]),
        # the surface level, within the surface tolerance but not written the
        # same for every node: node 5 lies above the nodes before it
        ({1: -5e-7, 2: -5e-7, 3: -5e-7, 4: -5e-7, 5: -1e-7},
         ["3-D node 5", "z = -1e-07 m", "z = -5e-07 m"]),
    ],
)
def test_refuses_rounding_inside_level(tmp_path, zs, needles):
    """A rounding error inside a level breaks the level walk; the message
    shows both values in full and names the cause."""
    fx = _fixture()
    for j, z in zs.items():
        _set_z(fx, j, z)
    with pytest.raises(ValueError) as info:
        read_ascii_mesh(_write(tmp_path / "mesh", fx, fmt=_fmt_repr))
    msg = str(info.value)
    for needle in ["nod3d.out is not level-major", *needles, "differ only by rounding"]:
        assert needle in msg, (needle, msg)


def test_refuses_positive_down_z(tmp_path):
    """Depths written positive down are named as such, not as a level-order
    problem."""
    fx = _fixture()
    for j in range(6, 14):
        _set_z(fx, j, -fx["nod3d"][j - 1][3])
    msg = _refusal(tmp_path, fx)
    for needle in ("nod3d.out", "3-D node 6", "z = 10.0 m", "above the surface",
                   "negative down"):
        assert needle in msg, (needle, msg)


def test_refuses_cavity_mesh(tmp_path):
    """An ice-shelf cavity: 2-D node 5's column starts at the second level
    (10 m). Numbered level-major and consistent in every file, it is refused
    as unsupported, naming the column's top wet node."""
    fx = _fixture()
    fx["nod3d"] = [
        (1, 10.0, 50.0, 0.0, 0),
        (2, 11.0, 50.0, 0.0, 0),
        (3, 10.5, 51.0, 0.0, 0),
        (4, 11.5, 51.0, 0.0, 0),
        (5, 10.0, 50.0, -10.0, 0),
        (6, 11.0, 50.0, -10.0, 0),
        (7, 10.5, 51.0, -10.0, 0),
        (8, 11.5, 51.0, -10.0, 0),
        (9, 12.0, 50.0, -10.0, 0),  # 2-D node 5, under the ice shelf
        (10, 10.0, 50.0, -25.0, 0),
        (11, 11.0, 50.0, -25.0, 0),
        (12, 11.5, 51.0, -25.0, 0),
        (13, 10.0, 50.0, -50.0, 0),
    ]
    fx["table"] = [
        [1, 5, 10, 13, -999, -999],
        [2, 6, 11, -999, -999, -999],
        [3, 7, -999, -999, -999, -999],
        [4, 8, 12, -999, -999, -999],
        [9, -999, -999, -999, -999, -999],
    ]
    fx["corresp"] = [1, 2, 3, 4, 1, 2, 3, 4, 5, 1, 2, 4, 1]
    msg = _refusal(tmp_path, fx)
    for needle in ("nod3d.out", "3-D node 9", "slot 1", "2-D node 5", "z = -10.0 m",
                   "top wet node of 2-D node 5", "ice-shelf cavities",
                   "not supported"):
        assert needle in msg, (needle, msg)


def _column_major(fx: dict) -> None:
    """Renumber the hand mesh column by column, consistently in all files."""
    lonlat = {n: (lon, lat) for n, lon, lat, _ in fx["nod2d"]}
    depths = [0.0, -10.0, -25.0, -50.0]
    lengths = [4, 3, 2, 3, 1]
    nod3d, table, corresp = [], [], []
    for node, length in enumerate(lengths, start=1):
        row = []
        for k in range(length):
            nod3d.append((len(nod3d) + 1, *lonlat[node], depths[k], 0))
            row.append(len(nod3d))
            corresp.append(node)
        table.append(row + [-999] * (MAX_NUM_LAYERS - length))
    fx["nod3d"], fx["table"], fx["corresp"] = nod3d, table, corresp


def _swap_level2(fx: dict) -> None:
    """List the second level's nodes as 2-D nodes 2, 1, 3, 4 (consistently)."""
    a, b = fx["nod3d"][5], fx["nod3d"][6]  # 3-D nodes 6 (2-D 1) and 7 (2-D 2)
    fx["nod3d"][5] = (6, *b[1:])
    fx["nod3d"][6] = (7, *a[1:])
    _set_table(fx, 1, 2, 7)
    _set_table(fx, 2, 2, 6)
    _set_corresp(fx, 6, 2)
    _set_corresp(fx, 7, 1)


@pytest.mark.parametrize(
    "renumber, needles",
    [
        # z goes back to 0 at 3-D node 5 (the second column starts)
        (_column_major, ["3-D node 5", "not deeper"]),
        # levels in order, but 2-D order broken inside the second level
        (_swap_level2, ["3-D node 7 (2-D node 1)", "increasing 2-D order"]),
    ],
)
def test_refuses_not_level_major(tmp_path, renumber, needles):
    fx = _fixture()
    renumber(fx)
    msg = _refusal(tmp_path, fx)
    for n in ["nod3d.out is not level-major", *needles]:
        assert n in msg, (n, msg)


# ---------------------------------------------------------------------------
# Refusals of the token stream itself
# ---------------------------------------------------------------------------


def test_refuses_wrong_token_count(tmp_path):
    """An extra column cannot be read as a stream of shifted records."""
    d = _write(tmp_path / "mesh")
    lines = (d / "nod2d.out").read_text().splitlines()
    (d / "nod2d.out").write_text(
        "\n".join([lines[0]] + [ln + " 7" for ln in lines[1:]]) + "\n")
    with pytest.raises(ValueError) as info:
        read_ascii_mesh(d)
    msg = str(info.value)
    assert "nod2d.out" in msg and "holds 26" in msg and "21 numbers" in msg, msg


def test_refuses_short_aux3d(tmp_path):
    fx = _fixture()
    fx["corresp"] = fx["corresp"][:-2]
    fx["elem_dummy"] = []
    msg = _refusal(tmp_path, fx)
    assert "aux3d.out" in msg and "fewer than the 44" in msg, msg


@pytest.mark.parametrize("token", ["-25,000000", "-25.0Q+00", "-25.0DD+00"])
def test_refuses_non_number(tmp_path, token):
    d = _write(tmp_path / "mesh")
    text = (d / "nod3d.out").read_text().replace("-25.000000", token, 1)
    (d / "nod3d.out").write_text(text)
    with pytest.raises(ValueError) as info:
        read_ascii_mesh(d)
    msg = str(info.value)
    assert "nod3d.out" in msg and repr(token) in msg and "not a number" in msg, msg


@pytest.mark.parametrize("token", ["11.5", "1.15D+01", "1.15e1"])
def test_refuses_non_integer_index(tmp_path, token):
    """A value that is not an integer is refused in an integer field, however
    it is spelt."""
    fx = _fixture()
    _set_table(fx, 2, 3, token)
    msg = _refusal(tmp_path, fx)
    for needle in ("aux3d.out", "2-D node 2", "slot 3", "is 11.5", "not an integer"):
        assert needle in msg, (needle, msg)


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda fx: _set_table(fx, 4, 2, "nine"), "'nine'"),
        # the last value reading needs: the parse reaches it
        (lambda fx: _set_corresp(fx, 13, "x1"), "'x1'"),
    ],
)
def test_refuses_non_number_in_aux3d(tmp_path, mutate, needle):
    fx = _fixture()
    mutate(fx)
    msg = _refusal(tmp_path, fx)
    for n in ("aux3d.out", needle, "not a number"):
        assert n in msg, (n, msg)


def test_refuses_empty_file(tmp_path):
    d = _write(tmp_path / "mesh")
    (d / "elem2d.out").write_text(" \n\n")
    with pytest.raises(ValueError, match="elem2d.out: the file is empty"):
        read_ascii_mesh(d)


def test_missing_file(tmp_path):
    d = _write(tmp_path / "mesh")
    (d / "aux3d.out").unlink()
    with pytest.raises(FileNotFoundError, match="aux3d.out"):
        read_ascii_mesh(d)
