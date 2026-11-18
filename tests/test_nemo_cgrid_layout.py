"""C-grid raster layouts (E-W halo, T / F pivot north fold): the canonical map,
the numpy lbc and the copy check (``readers.nemo._layout``), no grids needed."""
import numpy as np
import pytest

from kelp.readers.nemo._layout import (
    NemoCgridLayout, nemo_cgrid_canonical, nemo_cgrid_to_canonical, nemo_fold_pivot_cells)

NI, NY = 12, 8          # canonical halo form: 14 columns, fold row 6, halo row 7


def _layouts():
    for fold in ("T", "F"):
        yield NemoCgridLayout((NY, NI + 2), True, True, True, fold)       # NEMO <= 4.0 output
        yield NemoCgridLayout((NY - 1, NI), False, False, True, fold)     # NEMO >= 4.2, halo-free


def _canon_layout(fold):
    return NemoCgridLayout((NY, NI + 2), True, True, True, fold)


@pytest.mark.parametrize("fold", ["T", "F"])
@pytest.mark.parametrize("kind", "TUVF")
def test_canonical_is_idempotent_and_flips_only_beyond_the_fold(fold, kind):
    lay = _canon_layout(fold)
    jj, ii = np.mgrid[0:NY, 0:NI + 2]
    cj, ci, s = nemo_cgrid_canonical(kind, jj, ii, lay)
    cj2, ci2, s2 = nemo_cgrid_canonical(kind, cj, ci, lay)
    assert (cj2 == cj).all() and (ci2 == ci).all() and (s2 == 1).all()
    assert ((cj <= lay.J) & (ci >= 1) & (ci <= NI)).all()
    assert (s[(jj < lay.J) & (ii >= 1) & (ii <= NI)] == 1).all()
    assert (s[jj == lay.J + 1] == -1).all()


def test_canonical_maps_match_the_nemo_tables():
    J = NY - 2
    nx = NI + 2
    t = _canon_layout("T")
    i = np.arange(NI // 2 + 1, nx - 1)
    np.testing.assert_array_equal(nemo_cgrid_canonical("T", J, i, t)[1], nx - i)         # T(J,i) = T(J,nx-i)
    cj, ci, s = nemo_cgrid_canonical("V", J, np.arange(2, nx - 1), t)
    assert (cj == J - 1).all() and (ci == nx - np.arange(2, nx - 1)).all() and (s == -1).all()
    f = _canon_layout("F")
    cj, ci, s = nemo_cgrid_canonical("T", J + 1, np.arange(1, nx - 1), f)                # T(J+1,i) = T(J, nx-1-i)
    assert (cj == J).all() and (ci == nx - 1 - np.arange(1, nx - 1)).all()
    cj, ci, s = nemo_cgrid_canonical("U", J + 1, np.arange(1, nx - 2), f)                # U(J+1,i) = -U(J, nx-2-i)
    assert (cj == J).all() and (ci == nx - 2 - np.arange(1, nx - 2)).all() and (s == -1).all()


@pytest.mark.parametrize("fold", ["T", "F"])
def test_halo_and_halo_free_input_give_the_same_canonical_raster(fold):
    rng = np.random.default_rng(0)
    full, free = (_canon_layout(fold), NemoCgridLayout((NY - 1, NI), False, False, True, fold))
    for kind, vec in (("T", False), ("U", True), ("V", True), ("F", False)):
        a = nemo_cgrid_to_canonical(rng.standard_normal((3, NY, NI + 2)), full, kind, vector=vec)
        np.testing.assert_array_equal(nemo_cgrid_to_canonical(a, full, kind, vector=vec, check=True), a)
        raw = a[:, :NY - 1, 1:NI + 1]                 # strip the halo columns and the halo row
        assert free.shape == (NY, NI + 2)
        b = nemo_cgrid_to_canonical(raw, free, kind, vector=vec, check=True, name=kind)
        np.testing.assert_array_equal(b, a)


def test_vector_sign_only_for_vectors():
    lay = _canon_layout("T")
    a = np.arange(1, (NY * (NI + 2)) + 1, dtype=float).reshape(NY, NI + 2)
    for vec, sgn in ((False, 1.0), (True, -1.0)):
        b = nemo_cgrid_to_canonical(a, lay, "U", vector=vec)
        cj, ci, _ = nemo_cgrid_canonical("U", NY - 1, 3, lay)
        assert b[NY - 1, 3] == sgn * b[cj, ci]


def test_copy_check_refuses_a_stale_copy_and_names_the_raster():
    lay = _canon_layout("F")
    rng = np.random.default_rng(1)
    a = nemo_cgrid_to_canonical(rng.standard_normal((2, NY, NI + 2)), lay, "U", vector=True)
    a[1, 3, 0] += 1e-3                                # stale E-W halo column
    with pytest.raises(ValueError, match=r"u record: 1 halo / fold copies differ.*max \|diff\| 0\.001"):
        nemo_cgrid_to_canonical(a, lay, "U", vector=True, check=True, name="u record")
    a = nemo_cgrid_to_canonical(a, lay, "U", vector=True)  # silently repaired without the check
    a[0, NY - 1, 4] *= -1                             # a missing sign flip across the fold
    with pytest.raises(ValueError, match="differ from their canonical"):
        nemo_cgrid_to_canonical(a, lay, "U", vector=True, check=True)
    with pytest.raises(ValueError, match="raster shape"):
        nemo_cgrid_to_canonical(np.zeros((NY, NI)), lay, "U")


def test_nan_is_zero_and_copies_follow():
    lay = _canon_layout("T")
    a = np.full((NY, NI + 2), np.nan)
    a[3, 1] = 2.0
    b = nemo_cgrid_to_canonical(a, lay, "T")
    assert b[3, NI + 1] == 2.0 and b[3, 2] == 0.0 and np.isfinite(b).all()


@pytest.mark.parametrize("fold,cells", [("T", [(NY - 2, 1), (NY - 2, NI // 2 + 1)]),
                                        ("F", [(NY - 2, NI // 2), (NY - 2, NI // 2 + 1),
                                               (NY - 2, 1), (NY - 2, NI)])])
def test_fold_pivot_cells(fold, cells):
    piv = nemo_fold_pivot_cells(_canon_layout(fold))
    assert {tuple(x) for x in np.argwhere(piv)} == set(cells)
    assert not nemo_fold_pivot_cells(NemoCgridLayout((NY, NI + 2), True, False, True, None)).any()


def test_layout_shapes_label_and_validation():
    lay = NemoCgridLayout((70, 1440), False, False, True, "T")
    assert lay.shape == (71, 1442) and (lay.c0, lay.ni, lay.J) == (1, 1440, 69)
    assert "T-pivot north fold" in lay.label() and "70x1440 -> 71x1442" in lay.label()
    assert "unjoined raster edges (open where wet): S" in lay.label()
    reg = NemoCgridLayout((10, 12))
    assert reg.shape == (10, 12) and "N, E, W" in reg.label()
    with pytest.raises(ValueError):
        NemoCgridLayout((5, 5), fold="T")                 # a fold needs periodicity
    with pytest.raises(ValueError):
        NemoCgridLayout((5, 5), True, False, False)
