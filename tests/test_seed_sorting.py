"""Spatial seed sorting (utils/sort_seeds.py): reorder a seed file so
particles adjacent in the array are adjacent on the sphere — consecutive
particles then share cache lines of the velocity fields (measured +17%
on the fused 3-D step from horizontal sorting alone; more with the depth
key). A one-time preprocessing step; composes with split_seeds (sort
THEN shard -> each shard is also spatially compact).

Contracts: rows are permuted only (values verbatim); sort key = healpix
NEST cell at the requested zoom, ties broken by depth column when
present (stable); comments dropped like split_seeds; deterministic.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "utils"))

from sort_seeds import sort_seeds  # noqa: E402


class TestSortSeeds:
    def test_rows_permuted_verbatim_and_spatially_ordered(self, tmp_path):
        rng = np.random.default_rng(3)
        n = 500
        lon = rng.uniform(-180, 180, n)
        lat = np.degrees(np.arcsin(rng.uniform(-0.99, 0.99, n)))
        dep = rng.uniform(0, 2000, n)
        p = tmp_path / "seeds.dat"
        np.savetxt(p, np.column_stack([lon, lat, dep]), fmt="%.7f")
        out = sort_seeds(p, nside=32)
        orig = sorted(p.read_text().split())
        got = sorted(Path(out).read_text().split())
        assert got == orig                      # permutation, verbatim
        arr = np.loadtxt(out)
        pix = healpy.ang2pix(32, arr[:, 0], arr[:, 1], nest=True,
                             lonlat=True)
        assert (np.diff(pix) >= 0).all()        # NEST-sorted

    def test_depth_secondary_key(self, tmp_path):
        p = tmp_path / "s.dat"
        # two seeds in the SAME pixel, depths out of order
        p.write_text("10.0 5.0 500.0\n10.0 5.0 100.0\n")
        out = sort_seeds(p, nside=8)
        arr = np.loadtxt(out)
        assert arr[0, 2] == 100.0 and arr[1, 2] == 500.0

    def test_two_column_file(self, tmp_path):
        p = tmp_path / "s.dat"
        p.write_text("100.0 5.0\n-100.0 5.0\n0.0 5.0\n# c\n")
        out = sort_seeds(p, nside=8)
        arr = np.loadtxt(out, ndmin=2)
        assert arr.shape == (3, 2)

    def test_inline_comment_rows(self, tmp_path):
        p = tmp_path / "s.dat"
        p.write_text("100.0 5.0 10.0  # note\n-100.0 5.0 20.0\n")
        out = sort_seeds(p, nside=8)
        txt = Path(out).read_text()
        assert "# note" in txt                  # original line preserved
        assert np.loadtxt(out, comments="#", ndmin=2).shape == (2, 3)

    def test_output_overwriting_input_rejected(self, tmp_path):
        p = tmp_path / "s.dat"
        p.write_text("1.0 2.0\n3.0 4.0\n")
        with pytest.raises(ValueError, match="overwrite"):
            sort_seeds(p, nside=8, out_path=p)

    def test_deterministic(self, tmp_path):
        rng = np.random.default_rng(1)
        p = tmp_path / "s.dat"
        np.savetxt(p, rng.uniform(-60, 60, (50, 2)), fmt="%.6f")
        a = Path(sort_seeds(p, nside=16)).read_text()
        b = Path(sort_seeds(p, nside=16)).read_text()
        assert a == b
