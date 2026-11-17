"""Layout detection for ``mesh.nemo_grid: cgrid`` (``readers.nemo.detect_cgrid_layout``)."""
import numpy as np
import pytest

from kelp.readers.nemo import detect_cgrid_layout
from kelp.readers.structured import CgridLayout
from tests._cgrid_orca import NC, NY, orca_grid

LAYOUTS = [("T", True), ("T", False), ("F", True), ("F", False)]


def _detect(xy, flags=None):
    return detect_cgrid_layout(*xy["T"], *xy["F"], flags)


@pytest.mark.parametrize("fold,halo", LAYOUTS)
def test_detects_all_four_layouts_from_the_coordinates_alone(fold, halo):
    g = orca_grid(fold, halo)
    lay, notes = _detect(g.xy)
    assert lay == g.raw and not notes
    assert lay.shape == g.canon.shape and lay.fold == fold and lay.halo_cols == halo
    lay2, _ = _detect(g.xy, {"Iperio": 1, "NFold": 1, "NFtype": fold})   # flags agree
    assert lay2 == lay


def _regional(ny=8, nx=10):
    j, i = np.mgrid[0:ny, 0:nx].astype(float)
    return {"T": (i * 0.1, j * 0.1), "F": ((i + .5) * 0.1, (j + .5) * 0.1)}


def _global_latlon(halo, ny=12, nc=20):
    lat1 = np.linspace(-60.0, 80.0, ny)
    lon1 = (np.arange(nc + 2) - 1) * 360.0 / nc if halo else (np.arange(nc) + 0.5) * 360.0 / nc
    lon, lat = np.meshgrid(lon1, lat1)
    lon = ((lon + 180.0) % 360.0) - 180.0
    return {"T": (lon, lat), "F": (lon + 360.0 / nc / 2, lat + 0.5)}


def test_regular_rasters_without_a_fold():
    lay, notes = _detect(_regional())
    assert lay == CgridLayout((8, 10)) and not notes                      # regional: all edges open
    lay, _ = _detect(_global_latlon(halo=True))
    assert (lay.halo_cols, lay.periodic, lay.fold) == (True, True, None)
    with pytest.raises(ValueError, match="add Iperio=1"):                  # NEMO >= 4.2, flags absent
        _detect(_global_latlon(halo=False))
    lay, _ = _detect(_global_latlon(halo=False), {"Iperio": 1})
    assert lay == CgridLayout((12, 20), False, False, True, None) and lay.shape == (12, 22)
    lay, _ = _detect(_global_latlon(halo=False), {"Iperio": 0})           # trusted
    assert lay == CgridLayout((12, 20))


def test_stale_flags_make_a_crop_not_a_refusal():
    g = orca_grid("F", True)
    below = {k: (a[:5], b[:5]) for k, (a, b) in g.xy.items()}              # rows below the fold
    lay, notes = _detect(below, {"Iperio": 1, "NFold": 1, "NFtype": "F"})
    assert lay == CgridLayout((5, NC + 2), True, False, True, None)
    assert "north fold" in notes[0] and "open" in notes[0]
    box = _regional()
    lay, notes = _detect(box, {"Iperio": 1, "NFold": 1, "NFtype": "T"})   # box crop of a global file
    assert lay == CgridLayout((8, 10)) and "east and west edges are open" in notes[0]


def test_contradictions_refuse():
    g = orca_grid("F", True)
    with pytest.raises(ValueError, match="NFtype='T'"):
        _detect(g.xy, {"NFold": 1, "NFtype": "T"})
    with pytest.raises(ValueError, match="NFold=0"):
        _detect(g.xy, {"NFold": 0})
    with pytest.raises(ValueError, match="Iperio=0"):
        _detect(g.xy, {"Iperio": 0})
    with pytest.raises(NotImplementedError, match="Jperio"):
        _detect(g.xy, {"Jperio": 1})


@pytest.mark.parametrize("fold", "TF")
def test_partial_width_crop_symmetric_about_a_pivot_is_refused(fold):
    """The fold identities hold exactly for a window centred on a pivot, which
    would join the crop's west and east edges."""
    g = orca_grid(fold, False)
    ni = NC
    refused = 0
    for lo in range(1, ni // 2):
        win = slice(lo, ni - lo)                       # symmetric about the raw centre
        xy = {k: (a[:, win], b[:, win]) for k, (a, b) in g.xy.items()}
        try:
            lay, _ = _detect(xy)
        except ValueError as e:
            assert "not full width" in str(e)
            refused += 1
        else:
            assert not lay.fold                        # never a fold on a partial raster
    assert refused > 0


def test_zero_tiles_and_pole_row_are_refused():
    g = orca_grid("T", True)
    xy = {k: (a.copy(), b.copy()) for k, (a, b) in g.xy.items()}
    xy["T"][0][3, 4:6] = xy["T"][1][3, 4:6] = 0.0
    with pytest.raises(ValueError, match="zero-filled tiles"):
        _detect(xy)
    pole = _global_latlon(halo=True)
    pole["T"][1][-1, :] = 90.0
    pole["T"][0][-1, :] = 0.0
    with pytest.raises(ValueError, match="pole row"):
        _detect(pole)


def test_pole_guard_needs_the_whole_row_to_collapse():
    """Neighbouring top-row points coincide near the pivots of real ORCA grids."""
    xy = _regional()
    for a in xy["T"]:
        a[-1, 5] = a[-1, 6]
    assert _detect(xy)[0] == CgridLayout((8, 10))
