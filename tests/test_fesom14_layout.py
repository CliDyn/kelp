"""FESOM 1.4 data layouts (readers/fesom14/_layout.py).

Form detection from dimensions, the gather of a raw ``(nodes_3d,)`` vector
into ``(K, N)`` through the index table, the static wet mask of level-wise
data, and the per-record check against that mask. Synthetic arrays only.

The index table below is the hand-written table of the small ASCII test mesh:
levels 0, 10, 25, 50 m, column lengths 4, 3, 2, 3, 1, 3-D nodes numbered
level-major (level 0 = all 2-D nodes in 2-D order, then level 1, ...),
0-based, -1 = dry; 13 3-D nodes.
"""
import numpy as np
import pytest
import xarray as xr

from kelp.readers.fesom14._layout import (
    check_record, detect_form, gather_nodes3d, is_data, wet_mask_levelwise)

IDX = np.array([[0, 5, 9, 12],
                [1, 6, 10, -1],
                [2, 7, -1, -1],
                [3, 8, 11, -1],
                [4, -1, -1, -1]])
N3D = 13
COLUMN_LENGTHS = (4, 3, 2, 3, 1)

#: expected (K, N) wet mask of the table: column n wet on levels < its length
WET = np.array([[k < L for L in COLUMN_LENGTHS] for k in range(4)])


def _da(dims, sizes=None):
    sizes = sizes or {}
    shape = tuple(sizes.get(d, 3) for d in dims)
    return xr.DataArray(np.zeros(shape, dtype=np.float32), dims=dims)


def _levelwise(fill=np.nan, dtype=np.float64):
    """A (K, N) record shaped like the table: data on wet slots, *fill* below."""
    arr = np.full((4, 5), fill, dtype=dtype)
    vals = (np.arange(20, dtype=dtype).reshape(4, 5) - 7.0) * 0.1
    arr[WET] = vals[WET]
    return arr


# --------------------------------------------------------------------------
# is_data: the one validity predicate
# --------------------------------------------------------------------------

def test_is_data_predicate():
    x = np.array([0.0, -5.0, 9.9e28, -9.9e28, 1e29, -1e29, 1e30, -1e30,
                  np.nan, np.inf, -np.inf])
    want = [True, True, True, True, False, False, False, False,
            False, False, False]
    np.testing.assert_array_equal(is_data(x), want)
    # the raw float32 fill and a decoded NaN are both "missing"
    f32 = np.array([1e30, np.nan, 0.25], dtype=np.float32)
    np.testing.assert_array_equal(is_data(f32), [False, False, True])


# --------------------------------------------------------------------------
# detect_form
# --------------------------------------------------------------------------

def test_detect_forms():
    assert detect_form(_da(("time", "nodes_3d"))) == "nodes3d"
    # sizes are not checked here (that needs the mesh)
    assert detect_form(_da(("time", "nodes_3d"), {"nodes_3d": 7})) == "nodes3d"
    for horiz in ("ncells", "nodes_2d", "nodes_3d"):
        assert detect_form(_da(("time", "depth", horiz))) == "levelwise"


@pytest.mark.parametrize("dims", [
    ("time", "nodes_2d"),                    # a 2-D field, not a 3-D form
    ("time", "ncells"),
    ("nodes_3d",),                           # no time dimension
    ("depth", "ncells"),
    ("T", "nodes_3d"),                       # time dimension not named 'time'
    ("nodes_3d", "time"),                    # wrong order
    ("time", "ncells", "depth"),             # depth after the horizontal dim
    ("depth", "time", "ncells"),
    ("time", "depth", "elem"),               # unknown horizontal dimension
    ("time", "zz", "ncells"),                # unknown depth dimension
    ("time", "depth", "nodes_3d", "x"),      # extra dimension
    ("time", "depth", "ncells", "ncells2"),
])
def test_detect_form_refuses_other_layouts(dims):
    with pytest.raises(ValueError, match="layout"):
        detect_form(_da(dims))


def test_detect_form_names_variable_and_dims():
    da = _da(("time", "ncells", "depth")).rename("uo")
    with pytest.raises(ValueError) as exc:
        detect_form(da)
    msg = str(exc.value)
    assert "uo" in msg and "('time', 'ncells', 'depth')" in msg


# --------------------------------------------------------------------------
# gather_nodes3d
# --------------------------------------------------------------------------

def test_gather_hand_values():
    vec = 100.0 + np.arange(N3D)
    nan = np.nan
    want = np.array([[100., 101., 102., 103., 104.],
                     [105., 106., 107., 108., nan],
                     [109., 110., nan, 111., nan],
                     [112., nan, nan, nan, nan]])
    got = gather_nodes3d(vec, IDX)
    assert got.shape == (4, 5)
    assert got.flags.c_contiguous
    assert got.dtype == np.float64
    np.testing.assert_array_equal(got, want)        # NaN == NaN in this helper
    # every 3-D node lands exactly once, in its own (level, node) slot
    assert np.count_nonzero(np.isfinite(got)) == N3D
    # the mask of a gathered record is the table's wet mask
    np.testing.assert_array_equal(wet_mask_levelwise(got), IDX.T >= 0)
    np.testing.assert_array_equal(IDX.T >= 0, WET)


def test_gather_keeps_float32_and_casts_ints():
    v32 = np.arange(N3D, dtype=np.float32)
    g32 = gather_nodes3d(v32, IDX)
    assert g32.dtype == np.float32
    assert np.isnan(g32[3, 1])
    gi = gather_nodes3d(np.arange(N3D), IDX)
    assert gi.dtype == np.float64
    assert np.isnan(gi[1, 4]) and gi[2, 3] == 11.0


def test_gather_fills_raw_fill_values_are_kept_as_values():
    """The gather does not judge values: a raw 1e30 in a wet slot stays 1e30
    (the record check refuses it), dry slots become NaN."""
    vec = np.zeros(N3D)
    vec[7] = 1e30
    got = gather_nodes3d(vec, IDX)
    assert got[1, 2] == 1e30
    with pytest.raises(ValueError, match="node 2, level 1"):
        check_record(got, IDX.T >= 0, name="uo", record="r0")


@pytest.mark.parametrize("bad_idx, match", [
    (np.where(IDX == 12, 13, IDX), "out of range"),    # beyond the vector
    (np.where(IDX == 12, -2, IDX), "out of range"),    # below -1
    (IDX[0], "2-D"),
])
def test_gather_refuses_bad_table(bad_idx, match):
    with pytest.raises(ValueError, match=match):
        gather_nodes3d(np.zeros(N3D), bad_idx)


def test_gather_refuses_non_vector():
    with pytest.raises(ValueError, match="1-D"):
        gather_nodes3d(np.zeros((2, N3D)), IDX)


# --------------------------------------------------------------------------
# wet_mask_levelwise
# --------------------------------------------------------------------------

def test_wet_mask_prefix():
    arr = _levelwise()
    arr[0, 4] = 0.0          # no-slip zero at the surface is data, not missing
    wet = wet_mask_levelwise(arr)
    assert wet.dtype == np.bool_
    assert wet.shape == (4, 5)
    assert wet.flags.c_contiguous
    np.testing.assert_array_equal(wet, WET)


@pytest.mark.parametrize("fill", [np.nan, 1e30, -1e30])
def test_wet_hole_refused(fill):
    arr = _levelwise()
    arr[1, 3] = fill          # node 3: data at 0, missing at 1, data at 2
    with pytest.raises(ValueError) as exc:
        wet_mask_levelwise(arr)
    msg = str(exc.value)
    assert "node 3" in msg and "level 1" in msg and "prefix" in msg


def test_column_without_data_refused():
    arr = _levelwise()
    arr[0, 4] = np.nan        # node 4: its only level is missing
    with pytest.raises(ValueError, match="node 4"):
        wet_mask_levelwise(arr)


def test_wet_mask_refuses_non_2d():
    with pytest.raises(ValueError, match=r"\(K, N\)"):
        wet_mask_levelwise(np.zeros(5))


def test_nan_and_1e30_equivalent_below_bottom():
    as_nan = _levelwise(np.nan)
    as_fill = _levelwise(1e30)
    as_fill32 = _levelwise(1e30, dtype=np.float32)
    mixed = _levelwise(np.nan)
    mixed[3, 1:] = 1e30
    masks = [wet_mask_levelwise(a) for a in (as_nan, as_fill, as_fill32, mixed)]
    for m in masks:
        np.testing.assert_array_equal(m, WET)
    for a in (as_nan, as_fill, as_fill32, mixed):
        check_record(a, WET, name="uo", record="1951-01-16")   # accepted alike


# --------------------------------------------------------------------------
# check_record
# --------------------------------------------------------------------------

@pytest.mark.parametrize("fill", [np.nan, 1e30, np.inf])
def test_check_record_missing_in_wet_slot(fill):
    arr = _levelwise()
    arr[2, 3] = fill          # wet slot (node 3 has 3 levels)
    with pytest.raises(ValueError) as exc:
        check_record(arr, WET, name="vo", record="1951-02-15 00:00:00")
    msg = str(exc.value)
    for part in ("vo", "1951-02-15 00:00:00", "node 3", "level 2", "wet"):
        assert part in msg, (part, msg)


def test_check_record_data_in_dry_slot_refused():
    arr = _levelwise()
    arr[2, 1] = 0.0           # node 1 has 3 levels: level 2 is wet, fine
    check_record(arr, WET, name="thetao", record="r7")
    arr[3, 1] = 0.0           # level 3 of node 1 is dry: a finite value there
    with pytest.raises(ValueError) as exc:
        check_record(arr, WET, name="thetao", record="r7")
    msg = str(exc.value)
    for part in ("thetao", "r7", "node 1", "level 3", "dry"):
        assert part in msg, (part, msg)


def test_check_record_reports_first_node_and_count():
    arr = _levelwise()
    arr[1, 3] = np.nan
    arr[0, 2] = np.nan
    arr[3, 0] = 1e30
    with pytest.raises(ValueError) as exc:
        check_record(arr, WET, name="uo", record="r0")
    msg = str(exc.value)
    assert "node 0, level 3" in msg
    assert "3 wet slot(s) without data" in msg


def test_check_record_refuses_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        check_record(np.zeros((3, 5)), WET, name="uo", record="r0")


def test_mask_drift_refused():
    """The mask is fixed by the first record; a later record whose bottom
    moves (deeper or shallower in one column) is refused, one whose values
    change on the same slots is accepted."""
    rec0 = _levelwise()
    wet = wet_mask_levelwise(rec0)

    same_slots = rec0 * 3.0 - 1.0
    check_record(same_slots, wet, name="uo", record="t1")

    deeper = _levelwise()
    deeper[2, 2] = 0.5        # node 2 gains a level
    with pytest.raises(ValueError, match="node 2, level 2"):
        check_record(deeper, wet, name="uo", record="t2")

    shallower = _levelwise()
    shallower[2, 3] = np.nan  # node 3 loses its last level
    with pytest.raises(ValueError, match="node 3, level 2"):
        check_record(shallower, wet, name="uo", record="t3")

    # the drifted record is itself a valid prefix: only the comparison sees it
    np.testing.assert_array_equal(
        wet_mask_levelwise(shallower)[:, 3], [True, True, False, False])
