"""Level-blocked 3-D compaction (`_LEVEL_BLOCK`) — bitwise parity pin.

The 3-D read used to materialize the FULL (level, npix) float64 array
before compacting to wet nodes (~28 GB per variable at zoom 11, OOM on a
256-GB node even single-process).  The blocked read caps the transient
at a few levels while the returned values stay bit-identical: np.take
applied per level block equals one np.take of the whole array, and the
dtype/order of every operation is unchanged.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kelp.core.config import ModeConfig  # noqa: E402
import kelp.readers.healpix as hxr  # noqa: E402
from test_healpix_reader import (_make_uv_store, _mesh_cfg,  # noqa: E402
                                 _pix_lonlat, _wet, LEVELS_M, NLEV)

MODE_3D = ModeConfig(Fdrift2D=False, Fdrift3D=True)


def _read(tmp_path, monkeypatch, block):
    """One 3-D read of the mixed-depth fixture at a given block height."""
    monkeypatch.setattr(hxr, "_LEVEL_BLOCK", block)
    hxr._WARNED.clear()                      # re-arm once-per-name gates
    lon, lat = _pix_lonlat(4, "nest")
    nfin = np.where(_wet(lat), np.where(lat < 0, 3, 2), 0)
    p = _make_uv_store(tmp_path / f"3D_b{block}_ocean.nc", with_eta=True,
                       with_w=True, n_finite_levels=nfin, u_value=0.4,
                       w_value=1e-4)
    mesh = hxr.HealpixReader.load_mesh(_mesh_cfg(p, levels=LEVELS_M),
                                       mode_cfg=MODE_3D)
    ds = hxr.HealpixReader.open_dataset(p, mode_cfg=MODE_3D)
    return hxr.HealpixReader.read_velocities(ds, 0, mesh,
                                             mode_cfg=MODE_3D)


class TestBlockedReadParity:
    def test_block_constant_exists(self):
        assert isinstance(hxr._LEVEL_BLOCK, int) and hxr._LEVEL_BLOCK >= 1

    @pytest.mark.parametrize("block", [1, 2, 1000])
    def test_bitwise_parity_across_block_sizes(self, tmp_path,
                                               monkeypatch, block):
        # block=2 exercises a partial tail block (NLEV-1=3 u-layers);
        # block=1000 is the single-block (old whole-array) path
        ref = _read(tmp_path, monkeypatch, 10**9)
        got = _read(tmp_path, monkeypatch, block)
        for name in ("u", "v", "w"):
            a, b = getattr(ref, name), getattr(got, name)
            assert a.dtype == b.dtype and a.shape == b.shape
            np.testing.assert_array_equal(a, b)
            assert b.flags.c_contiguous

    def test_prefix_gate_still_fires_blocked(self, tmp_path, monkeypatch):
        # an interior NaN stripe (finite BELOW non-finite) must still be
        # caught when the corruption sits in a later block
        import xarray as xr
        monkeypatch.setattr(hxr, "_LEVEL_BLOCK", 1)
        hxr._WARNED.clear()
        lon, lat = _pix_lonlat(4, "nest")
        nfin = np.where(_wet(lat), 3, 0)
        p = _make_uv_store(tmp_path / "3D_g_ocean.nc", with_eta=True,
                           with_w=True, n_finite_levels=nfin)
        # corrupt one wet column at an INTERIOR level (not the surface:
        # the wet mask keys on level 0, a surface NaN would just delist
        # the column) -> finite BELOW non-finite; with block=1 the gap
        # level and the finite level land in different blocks
        with xr.open_dataset(p) as src:
            ds = src.load()
        wet_pix = int(np.flatnonzero(np.isfinite(
            ds["avg_uoe"].values[0, 0])).ravel()[0])
        for vn in ("avg_uoe", "avg_von"):
            ds[vn].values[0, 1, wet_pix] = np.nan
        p.unlink()
        ds.to_netcdf(p)
        mesh = hxr.HealpixReader.load_mesh(_mesh_cfg(p, levels=LEVELS_M),
                                           mode_cfg=MODE_3D)
        ds = hxr.HealpixReader.open_dataset(p, mode_cfg=MODE_3D)
        with pytest.raises(ValueError, match="BELOW a non-finite"):
            hxr.HealpixReader.read_velocities(ds, 0, mesh,
                                              mode_cfg=MODE_3D)
