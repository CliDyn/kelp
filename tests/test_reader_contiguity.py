"""C-contiguity contract of reader-produced fields (performance pin).

The parallel whole-field time blends (``core.io._blend`` and
``numerics.temporal_reconstruct._pchip_blend4``) gate on
``flags.c_contiguous``: any reader path handing back FORTRAN-ordered or
strided >=2-D fields silently drops every per-step blend to the serial
numpy expression (measured ~10x per-step cost on a healpix nside-256
3-D run, ~1 busy core of 16).  Values are layout-independent — these
tests pin the *performance* contract for every producer the 2026-08
reviews found delivering F-order, plus the one-time kernel warning that
makes future regressions of this class self-announcing.
"""
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

xr = pytest.importorskip("xarray")

from kelp.core.config import DiffusionConfig, ModeConfig  # noqa: E402
from test_3d_analytical import _make_mesh  # noqa: E402

_MODE_3D = ModeConfig(Fdrift2D=False, Fdrift3D=True)


# ---------------------------------------------------------------------------
# FESOM2: spatial-first storage (time, space, level) — the reader supports it
# by dim-name orientation; the transpose view must not leak F-order out.
# ---------------------------------------------------------------------------

class TestFesom2SpatialFirst:
    K, N, E, T = 5, 9, 8, 2
    Z = [0.0, 10.0, 25.0, 50.0, 100.0]

    def _mesh(self):
        return _make_mesh(5.0, 5.0, z_levels=self.Z, bottom_depth=100.0)

    def _ds(self, with_az=False):
        # float64 payloads on purpose: asarray(dtype=float64) then makes NO
        # copy, so a transpose view would survive to the VelocityField.
        data = {
            "u": (("time", "elem", "nz1"),
                  np.ones((self.T, self.E, self.K - 1))),
            "v": (("time", "elem", "nz1"),
                  np.full((self.T, self.E, self.K - 1), 0.5)),
            "w": (("time", "nod2", "nz"),
                  np.full((self.T, self.N, self.K), 0.01)),
            "ssh": (("time", "nod2"), np.zeros((self.T, self.N))),
        }
        if with_az:
            data["Av"] = (("time", "nod2", "nz"),
                          np.full((self.T, self.N, self.K), 2.0e-3))
        return xr.Dataset(data)

    def test_uvw_oriented_and_c_contiguous(self):
        from kelp.readers.fesom2 import Fesom2Reader
        vel = Fesom2Reader.read_velocities(self._ds(), 0, self._mesh(),
                                           _MODE_3D)
        assert vel.u.shape == (self.K - 1, self.E)
        assert vel.v.shape == (self.K - 1, self.E)
        # w stored (time, nod2, nz) must come back vert-first like u/v
        assert vel.w.shape == (self.K, self.N)
        np.testing.assert_allclose(vel.w, -0.01)    # positive-up file
        for name in ("u", "v", "w"):
            arr = getattr(vel, name)
            assert all(s > 1 for s in arr.shape)
            assert arr.flags.c_contiguous, f"{name} not C-contiguous"

    def test_misstaggered_w_rejected(self):
        # A LAYER-staggered w (nz1, K-1 levels) must be rejected, both
        # storage orders — silently accepting it yields wrong vertical
        # velocities (FESOM-C has this guard; FESOM2 lacked it).
        from kelp.readers.fesom2 import Fesom2Reader
        for dims, shape in ((("time", "nz1", "nod2"),
                             (self.T, self.K - 1, self.N)),
                            (("time", "nod2", "nz1"),
                             (self.T, self.N, self.K - 1))):
            ds = self._ds()
            ds = ds.drop_vars("w")
            ds["w"] = (dims, np.full(shape, 0.01))
            with pytest.raises(ValueError, match="interface"):
                Fesom2Reader.read_velocities(ds, 0, self._mesh(), _MODE_3D)

    def test_wrong_interface_count_rejected(self):
        # An interface-named dim carrying the wrong level count must not
        # slip through (mirrors the az validator).
        from kelp.readers.fesom2 import Fesom2Reader
        ds = self._ds()
        ds = ds.drop_vars("w")
        ds["w"] = (("time", "nz", "nod2"),
                   np.full((self.T, self.K + 3, self.N), 0.01))
        with pytest.raises(ValueError, match="interface"):
            Fesom2Reader.read_velocities(ds, 0, self._mesh(), _MODE_3D)

    def test_az_c_contiguous(self):
        from kelp.readers.fesom2 import Fesom2Reader
        diff = DiffusionConfig(enabled=True, az_enabled=True, var_az="Av")
        vel = Fesom2Reader.read_velocities(self._ds(with_az=True), 0,
                                           self._mesh(), _MODE_3D,
                                           diffusion=diff)
        assert vel.az.shape == (self.K, self.N)
        assert vel.az.flags.c_contiguous, "az not C-contiguous"


# ---------------------------------------------------------------------------
# FESOM-C: (node, sigma) storage normalized by transposing — the
# ``np.asarray(arr.T)`` view must not leak F-order out.
# ---------------------------------------------------------------------------

class TestFesomC3DContiguity:
    def test_uvw_c_contiguous(self, tmp_path):
        from test_readers import _make_fesomc_3d
        from kelp.readers import get_reader
        nc = tmp_path / "fesomc_3d.nc"
        _make_fesomc_3d(nc)
        reader = get_reader("fesomc")
        mesh = reader.load_mesh(SimpleNamespace(meshpath=str(nc)))
        ds = reader.open_dataset(str(nc))
        mode = SimpleNamespace(
            Fdrift2D=False, Fdrift3D=True, Vpuls=False,
            var_u=None, var_v=None, var_w=None, var_eta=None,
            sigma_layer=None, sigma_mean=False, max_levels=None,
        )
        vel = reader.read_velocities(ds, 0, mesh, mode)
        ds.close()
        for name in ("u", "v", "w"):
            arr = getattr(vel, name)
            assert all(s > 1 for s in arr.shape)
            assert arr.flags.c_contiguous, f"{name} not C-contiguous"


# ---------------------------------------------------------------------------
# Shared Kh loader: a non-leading time dim leaves a strided 1-D view.
# ---------------------------------------------------------------------------

class TestKhLoaderContiguity:
    def test_kh_strided_source_comes_back_contiguous(self):
        from kelp.readers import _read_kh_fields
        N, T = 9, 3
        mesh = _make_mesh(5.0, 5.0)
        assert mesh.n_nodes == N
        # float64 + time LAST: isel(time=0) yields a strided view and
        # asarray(dtype=float64) keeps it (no copy).
        ds = xr.Dataset({
            "Kh_zonal": (("nod2", "time"),
                         np.full((N, T), 10.0)),
            "Kh_meridional": (("nod2", "time"),
                              np.full((N, T), 10.0)),
        })
        diff = DiffusionConfig(enabled=True, scheme="em")
        kh = _read_kh_fields(ds, 0, mesh, diff)
        assert kh is not None
        kh_x, kh_y, layout = kh
        assert layout == "node"
        for nm, arr in (("kh_x", kh_x), ("kh_y", kh_y)):
            assert arr.flags.c_contiguous, f"{nm} not C-contiguous"


# ---------------------------------------------------------------------------
# Kernel self-announcement: a large float64 same-shape field missing the
# fast path ONLY on contiguity warns once — this class of regression must
# surface by running, not by profiling.
# ---------------------------------------------------------------------------

class TestBlendContiguityWarning:
    def _big_pair_f_order(self):
        a = np.asfortranarray(np.ones((4, 40_000)))
        b = np.asfortranarray(np.full((4, 40_000), 2.0))
        return a, b

    def test_blend_warns_once_on_f_order(self):
        import kelp.core.io as io_mod
        if not io_mod.HAS_NUMBA:
            pytest.skip("warning is tied to the numba fast path")
        a, b = self._big_pair_f_order()
        io_mod._CONTIG_WARNED.clear()
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            out = io_mod._blend(a, b, 0.5)
        np.testing.assert_allclose(out, 1.5)
        msgs = [w for w in rec if "contiguous" in str(w.message)]
        assert len(msgs) == 1
        # second call: silent (once per process)
        with warnings.catch_warnings(record=True) as rec2:
            warnings.simplefilter("always")
            io_mod._blend(a, b, 0.5)
        assert not [w for w in rec2 if "contiguous" in str(w.message)]

    def test_blend_silent_on_c_order_and_small(self):
        import kelp.core.io as io_mod
        if not io_mod.HAS_NUMBA:
            pytest.skip("warning is tied to the numba fast path")
        io_mod._CONTIG_WARNED.clear()
        big_c = np.ones((4, 40_000))
        small_f = np.asfortranarray(np.ones((4, 40)))
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            io_mod._blend(big_c, np.ones((4, 40_000)), 0.5)
            io_mod._blend(small_f, np.asfortranarray(np.ones((4, 40))), 0.5)
        assert not [w for w in rec if "contiguous" in str(w.message)]

    def test_pchip_blend4_warns_once_on_f_order(self):
        import kelp.numerics.temporal_reconstruct as tr
        if not tr.HAS_NUMBA:
            pytest.skip("warning is tied to the numba fast path")
        fs = [np.asfortranarray(np.full((4, 40_000), float(k)))
              for k in range(4)]
        tr._CONTIG_WARNED.clear()
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            out = tr._pchip_blend4(*fs, 0.0, 1.0, 2.0, 3.0, 1.5)
        np.testing.assert_allclose(out, 1.5)
        msgs = [w for w in rec if "contiguous" in str(w.message)]
        assert len(msgs) == 1
        with warnings.catch_warnings(record=True) as rec2:
            warnings.simplefilter("always")
            tr._pchip_blend4(*fs, 0.0, 1.0, 2.0, 3.0, 1.5)
        assert not [w for w in rec2 if "contiguous" in str(w.message)]

    def test_pchip_blend4_silent_on_c_order(self):
        import kelp.numerics.temporal_reconstruct as tr
        if not tr.HAS_NUMBA:
            pytest.skip("warning is tied to the numba fast path")
        tr._CONTIG_WARNED.clear()
        fs = [np.full((4, 40_000), float(k)) for k in range(4)]
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            tr._pchip_blend4(*fs, 0.0, 1.0, 2.0, 3.0, 1.5)
        assert not [w for w in rec if "contiguous" in str(w.message)]


# ---------------------------------------------------------------------------
# _ffill_columns defence-in-depth: even for a FORTRAN-ordered input the
# result must come out C (the healpix caller now feeds it C, but the
# helper's own contract should not depend on that).
# ---------------------------------------------------------------------------

def test_ffill_columns_forces_c_order():
    from kelp.readers.healpix import _ffill_columns
    a = np.asfortranarray(np.arange(12.0).reshape(3, 4))
    a[0, 1] = np.nan
    f = _ffill_columns(a)
    assert f.flags.c_contiguous


class TestSamplingSingularGuard:
    """Same class as the interpolate det-guard fix (2026-08): sampling's
    own leastsq fit had an ABSOLUTE det threshold; at a thin nearest-node
    ring the fit 'succeeds' on roundoff and the sampled value can leave
    the field's range entirely — the contract is fall-back-to-nearest."""

    def test_thin_ring_sample_stays_in_range(self):
        healpy = pytest.importorskip("healpy")
        from kelp.readers.healpix import build_healpix_pixel_mesh
        from kelp.numerics.sampling import sample_field
        import warnings as _w
        nside = 8
        npix = 12 * nside * nside
        target = healpy.ang2pix(nside, 45.0, 20.0, nest=True, lonlat=True)
        nb = healpy.get_all_neighbours(nside, target, nest=True)
        wet = np.zeros(npix, bool)
        wet[[target, nb[4]]] = True
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            m = build_healpix_pixel_mesh(nside, wet_mask=wet)
        # node-based linear field (sampling wants node values)
        vals = 5.0 + 0.1 * m.nodes_lat
        rng = np.random.default_rng(3)
        lons = 45.0 + rng.uniform(-2, 2, 40)
        lats = 20.0 + rng.uniform(-2, 2, 40)
        out = sample_field(vals, m, lons, lats, method="leastsq")
        fin = np.isfinite(out)
        assert fin.any()
        lo, hi = vals.min() - 0.5, vals.max() + 0.5
        assert ((out[fin] >= lo) & (out[fin] <= hi)).all(), (
            f"sampled values escaped the field range: "
            f"[{out[fin].min():.2f}, {out[fin].max():.2f}] vs [{lo}, {hi}]")
