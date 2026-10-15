"""Blocked rotation mix + early dtype cast (`_rotate_uv_blocked`) pins.

At zoom 11 the whole-array rotation mix ``c*u - s*v`` built ~40 GB of
float64 temporaries on top of float64 u/v/w held to the end of the read
— the OOM peak after the blocked-read fix moved it there.  The mix now
runs level-block by level-block into preallocated outputs (float64
arithmetic, so bits are unchanged), and u/v/w drop to the configured
velocity dtype as soon as each is final instead of at the return.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kelp.core.config import ModeConfig  # noqa: E402
from kelp.core.rotation import (FESOM2_ANGLES, euler_matrix,  # noqa: E402
                                    rotate_lonlat)
import kelp.readers.healpix as hxr  # noqa: E402
from test_healpix_reader import (_make_uv_store, _mesh_cfg,  # noqa: E402
                                 _pix_lonlat, LEVELS_M)

MODE_3D = ModeConfig(Fdrift2D=False, Fdrift3D=True)


def _rotated_read(tmp_path, name, *, float32, block=None, monkeypatch=None):
    # wet mask in ROTATED latitude: the rotated caps must stay dry or
    # load_mesh's pole-in-wet-element guard (correctly) refuses the mesh
    lon, lat = _pix_lonlat(4, "nest")
    _, rlat = rotate_lonlat(lon, lat, euler_matrix(*FESOM2_ANGLES))
    nfin = np.where(np.abs(rlat) < 75.0,
                    np.where(rlat < 0, 3, 2), 0)
    p = _make_uv_store(tmp_path / name, with_eta=True, with_w=True,
                       n_finite_levels=nfin, u_value=0.37, w_value=2e-4)
    if block is not None:
        monkeypatch.setattr(hxr, "_LEVEL_BLOCK", block)
    mesh = hxr.HealpixReader.load_mesh(
        _mesh_cfg(p, levels=LEVELS_M, rotation="fesom2"), mode_cfg=MODE_3D)
    ds = hxr.HealpixReader.open_dataset(p, mode_cfg=MODE_3D)
    return hxr.HealpixReader.read_velocities(
        ds, 0, mesh, mode_cfg=MODE_3D, velocity_float32=float32)


class TestBlockedMix:
    def test_blocked_mix_helper_exists(self):
        assert callable(getattr(hxr, "_rotate_uv_blocked", None))

    @pytest.mark.parametrize("block", [1, 2, 1000])
    def test_rotated_f64_parity_across_blocks(self, tmp_path,
                                              monkeypatch, block):
        ref = _rotated_read(tmp_path, "3D_ref_ocean.nc", float32=False,
                            block=10**9, monkeypatch=monkeypatch)
        got = _rotated_read(tmp_path, f"3D_b{block}_ocean.nc",
                            float32=False, block=block,
                            monkeypatch=monkeypatch)
        for nm in ("u", "v", "w"):
            np.testing.assert_array_equal(getattr(ref, nm),
                                          getattr(got, nm))
            assert getattr(got, nm).flags.c_contiguous

    def test_blocked_equals_literal_expression_bitwise(self, monkeypatch):
        # direct pin against the OLD expression, bit views: signed zeros,
        # NaN payloads and denormals must survive identically
        monkeypatch.setattr(hxr, "_LEVEL_BLOCK", 2)
        rng = np.random.default_rng(7)
        u = rng.standard_normal((5, 33))
        v = rng.standard_normal((5, 33))
        u[0, 0] = -0.0
        v[0, 1] = np.nan
        u[1, 2] = 5e-324                       # denormal
        c = np.cos(rng.uniform(0, 2 * np.pi, 33))
        s = np.sin(rng.uniform(0, 2 * np.pi, 33))
        un, vn = hxr._rotate_uv_blocked(u, v, c, s)
        eu, ev = c * u - s * v, s * u + c * v
        np.testing.assert_array_equal(un.view(np.uint64),
                                      eu.view(np.uint64))
        np.testing.assert_array_equal(vn.view(np.uint64),
                                      ev.view(np.uint64))
        # float32 inputs must promote exactly like the literal expression
        u32, v32 = u.astype(np.float32), v.astype(np.float32)
        un32, vn32 = hxr._rotate_uv_blocked(u32, v32, c, s)
        np.testing.assert_array_equal(
            un32.view(np.uint64), (c * u32 - s * v32).view(np.uint64))
        assert un32.dtype == np.float64        # promoted, not rounded

    def test_f32_equals_f64_mix_then_cast(self, tmp_path):
        # the early cast must not change precision: the mix stays f64,
        # so the f32 read == astype(f32) of the f64 read, bit for bit
        f64 = _rotated_read(tmp_path, "3D_c64_ocean.nc", float32=False)
        f32 = _rotated_read(tmp_path, "3D_c32_ocean.nc", float32=True)
        for nm in ("u", "v", "w"):
            a64, a32 = getattr(f64, nm), getattr(f32, nm)
            assert a32.dtype == np.float32
            np.testing.assert_array_equal(
                a64.astype(np.float32), a32)
            assert np.isfinite(a32).all()      # ground stays zeroed
