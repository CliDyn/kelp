"""em/m1 spatially-varying Kh diffusion on HEALPix pixels (Phase 5b).

The split_rk composition (em/m1 + RK2/RK4) never depended on the u/v
layout — the Kh kick machinery reads only `vel.kh_zonal/kh_meridional/
kh_layout` and generic mesh structures — so lifting the reader's blanket
em/m1 rejection needs exactly one new piece: reading the pixel-space Kh
variables and compacting them onto the wet-pixel NODES (the same
compaction u/v/eta use).  The em/m1+Euler "combined" mode stays
impossible on healpix: `_apply_combined_em_step`'s `uv_interp !=
"leastsq"` guard fires first (and the node-layout guard right behind
it) — pinned below.

Synthetic stores only: the CORE2 twin stores carry no Kh variables
(AGENTS.md), so the real-data legs live in test_healpix_real_core2.py
without diffusion.  Restart-bitwise properties of the counter-RNG are
already pinned generically in the diffusion suites — nothing healpix-
specific to re-pin there.
"""
import sys
import warnings as _w
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))

from kelp.core.config import DiffusionConfig, NumericsConfig  # noqa: E402
from kelp.core.particles import ParticleStatus, make_state  # noqa: E402
from kelp.readers import get_reader  # noqa: E402
from kelp.tracking import run_single_realization  # noqa: E402
import kelp.readers.healpix as hx  # noqa: E402

from test_healpix_reader import (_make_uv_store, _pix_lonlat,  # noqa: E402
                                 _wet, NSIDE)

KH0 = 5000.0        # m^2/s — big enough for km-scale kicks over days


def _kh_field(lon, lat):
    # varies in BOTH axes (a pure-lat field would be blind to
    # longitudinal permutations within a HEALPix ring)
    return KH0 + 50.0 * lat + 0.5 * lon


def _make_store_with_kh(path, *, nan_wet_kh=0, neg_wet_kh=0,
                        time_factor=0.0, **kw):
    """uv store + node (pixel-space) Kh_zonal/Kh_meridional fields.

    ``nan_wet_kh`` / ``neg_wet_kh`` — poison this many WET pixels' Kh
    with NaN (zero-with-warning leg) / a negative value (rejection leg).
    ``time_factor`` — scale Kh by ``(1 + time_factor*t_index)`` so the
    temporal-interp lerp has something to blend.
    """
    _make_uv_store(path, **kw)
    lon, lat = _pix_lonlat(NSIDE, "nest")
    wet = _wet(lat)
    kh = np.where(wet, _kh_field(lon, lat), np.nan)
    if nan_wet_kh:
        kh[np.where(wet)[0][:nan_wet_kh]] = np.nan
    if neg_wet_kh:
        kh[np.where(wet)[0][:neg_wet_kh]] = -3000.0
    with xr.open_dataset(path) as ds:
        ds = ds.load()
    ntime = ds.sizes["time"]
    scale = 1.0 + time_factor * np.arange(ntime)
    ds["Kh_zonal"] = (("time", "value"), kh[None, :] * scale[:, None])
    ds["Kh_meridional"] = (("time", "value"),
                           (kh * 0.5)[None, :] * scale[:, None])
    ds.to_netcdf(path, mode="w")
    return path


def _mode():
    return SimpleNamespace(Fdrift2D=True, Fdrift3D=False, time_dir=1.0,
                           sigma_layer=None, var_u=None, var_v=None,
                           var_w=None, var_eta=None, sample_vars=[],
                           max_levels=None)


def _mesh_cfg(path):
    return SimpleNamespace(meshpath=str(path), healpix_zoom=None,
                           crop_bbox=None, levels=None, rotation=None,
                           r_earth=None)


def _open(path):
    reader = get_reader("healpix")
    mesh = reader.load_mesh(_mesh_cfg(path), _mode())
    ds = reader.open_dataset(str(path))
    return reader, mesh, ds


def _diff(scheme="em", composition="split_rk", seed=11):
    return DiffusionConfig(enabled=True, scheme=scheme,
                           composition=composition, seed=seed)


class TestReaderKh:
    def test_em_read_attaches_node_kh(self, tmp_path):
        p = _make_store_with_kh(tmp_path / "s.nc")
        reader, mesh, ds = _open(p)
        vel = reader.read_velocities(ds, 0, mesh, _mode(),
                                     diffusion=_diff())
        ds.close()
        assert vel.kh_layout == "node"
        assert vel.kh_zonal.shape == (mesh.n_nodes,)
        assert vel.kh_meridional.shape == (mesh.n_nodes,)
        # values land on the right pixels (compaction parity with u/v);
        # the field varies in lon AND lat, so a within-ring permutation
        # would fail this too.  The store writes lon in [0, 360) while
        # the mesh stores [-180, 180) — evaluate in the store's branch.
        expect = _kh_field(np.mod(mesh.nodes_lon, 360.0),
                           mesh.nodes_lat)
        np.testing.assert_allclose(vel.kh_zonal, expect, rtol=1e-12)
        np.testing.assert_allclose(vel.kh_meridional, 0.5 * expect,
                                   rtol=1e-12)

    def test_disabled_reads_no_kh_uniform_attaches(self, tmp_path):
        """fesom2 parity: the shared helper reads Kh whenever diffusion
        is ENABLED and the variables exist (uniform simply never
        consumes them) — only disabled/None skips the read."""
        p = _make_store_with_kh(tmp_path / "s.nc")
        reader, mesh, ds = _open(p)
        v0 = reader.read_velocities(ds, 0, mesh, _mode(), diffusion=None)
        assert v0.kh_zonal is None
        v1 = reader.read_velocities(
            ds, 0, mesh, _mode(),
            diffusion=DiffusionConfig(enabled=True, scheme="uniform",
                                      kh_zonal=10.0, kh_meridional=10.0,
                                      seed=1))
        ds.close()
        assert v1.kh_zonal is not None        # attached, ignored by split

    def test_missing_kh_vars_yield_none(self, tmp_path):
        """No Kh variables in the store: reader parity with fesom2 —
        kh stays None and the split_rk kick raises its own clear
        ValueError at step time."""
        p = tmp_path / "nokh.nc"
        _make_uv_store(p)
        reader, mesh, ds = _open(p)
        vel = reader.read_velocities(ds, 0, mesh, _mode(),
                                     diffusion=_diff())
        ds.close()
        assert vel.kh_zonal is None

    def test_wet_nan_kh_zeroed_with_warning(self, tmp_path):
        p = _make_store_with_kh(tmp_path / "s.nc", nan_wet_kh=3)
        reader, mesh, ds = _open(p)
        hx._WARNED.clear()          # the warn key is var-NAME-scoped —
        # another test in this process may have burned it (house idiom)
        with pytest.warns(RuntimeWarning, match="Kh"):
            vel = reader.read_velocities(ds, 0, mesh, _mode(),
                                         diffusion=_diff())
        ds.close()
        assert np.isfinite(vel.kh_zonal).all()
        assert (vel.kh_zonal == 0.0).sum() >= 3

    def test_inf_kh_rejected(self, tmp_path):
        """±Inf is a broken field, not mask drift — it must raise, not
        be zeroed into silence (round-2: -Inf slipped past the sign
        check by being zeroed first)."""
        p = _make_store_with_kh(tmp_path / "s.nc")
        with xr.open_dataset(p) as ds0:
            ds0 = ds0.load()
        kh = ds0["Kh_zonal"].values
        wet_cols = np.where(np.isfinite(kh[0]))[0]
        kh[:, wet_cols[0]] = -np.inf
        ds0["Kh_zonal"] = (("time", "value"), kh)
        ds0.to_netcdf(p, mode="w")
        reader, mesh, ds = _open(p)
        with pytest.raises(ValueError, match="Inf"):
            reader.read_velocities(ds, 0, mesh, _mode(),
                                   diffusion=_diff())
        ds.close()

    def test_negative_kh_rejected(self, tmp_path):
        """Same contract as the shared _read_kh_fields: a negative
        diffusivity silently drops the noise term while the Ito drift
        keeps acting — reject, never clamp."""
        p = _make_store_with_kh(tmp_path / "s.nc", neg_wet_kh=5)
        reader, mesh, ds = _open(p)
        with pytest.raises(ValueError, match="non-negative"):
            reader.read_velocities(ds, 0, mesh, _mode(),
                                   diffusion=_diff())
        ds.close()


class TestTrackingEmOnPixels:
    def _run(self, path, diffusion, n_steps=2, seeds=24, **rsr_kw):
        reader, mesh, ds = _open(path)
        rng = np.random.default_rng(5)
        st = make_state(rng.uniform(10.0, 60.0, seeds),
                        rng.uniform(20.0, 45.0, seeds))
        num = NumericsConfig(dt=rsr_kw.pop("dt", 86400.0),
                             integrator="rk2", output_freq=1,
                             uv_interp="barycentric",
                             diffusion=diffusion if diffusion else
                             DiffusionConfig())
        rec = run_single_realization(mesh, ds, reader, st, _mode(), num,
                                     0, n_steps=n_steps, n_time_total=3,
                                     on_exhausted="cycle", **rsr_kw)
        ds.close()
        return rec.to_arrays()

    def test_em_split_rk_runs_and_disperses(self, tmp_path):
        p = _make_store_with_kh(tmp_path / "s.nc")
        with _w.catch_warnings():
            _w.simplefilter("ignore", RuntimeWarning)   # one-time split-RK warn
            a_diff = self._run(p, _diff(seed=11))
            a_none = self._run(p, None)
        alive = a_diff["status"][-1] == ParticleStatus.ALIVE
        assert alive.sum() >= 20
        # kick magnitude must be CONSISTENT with Kh: sigma per step =
        # sqrt(2*K*dt) with K ~ 5-7e3 m2/s, dt = 86400 s -> ~33-35 km;
        # over 2 steps ~0.3-0.5 deg at these latitudes (measured median
        # 0.38 deg).  The two-sided band catches unit errors both ways
        # (a 100x-small kick fails the floor, a 100x-large one the
        # ceiling) — review 2026-08-16.
        dev = np.hypot(a_diff["lon"][-1] - a_none["lon"][-1],
                       a_diff["lat"][-1] - a_none["lat"][-1])
        med = np.median(dev[alive])
        assert 0.05 < med < 2.0, med

    def test_em_with_temporal_interp_and_time_varying_kh(self, tmp_path):
        """The combination a production run actually uses: sub-file-step
        dt + LazyLerpField payloads lerping a TIME-VARYING Kh."""
        p = _make_store_with_kh(tmp_path / "s.nc", time_factor=0.2)
        file_times = np.array([0.0, 86400.0, 172800.0])
        with _w.catch_warnings():
            _w.simplefilter("ignore", RuntimeWarning)
            a1 = self._run(p, _diff(seed=21), n_steps=4, dt=43200.0,
                           temporal_interp=True, file_times=file_times)
            a2 = self._run(p, _diff(seed=21), n_steps=4, dt=43200.0,
                           temporal_interp=True, file_times=file_times)
            a0 = self._run(p, None, n_steps=4, dt=43200.0,
                           temporal_interp=True, file_times=file_times)
        alive = a1["status"][-1] == ParticleStatus.ALIVE
        assert alive.sum() >= 20
        dev = np.hypot(a1["lon"][-1] - a0["lon"][-1],
                       a1["lat"][-1] - a0["lat"][-1])
        assert 0.05 < np.median(dev[alive]) < 2.0
        np.testing.assert_array_equal(a1["lon"], a2["lon"])   # bitwise

    def test_same_seed_bitwise_reproducible(self, tmp_path):
        p = _make_store_with_kh(tmp_path / "s.nc")
        with _w.catch_warnings():
            _w.simplefilter("ignore", RuntimeWarning)
            a1 = self._run(p, _diff(seed=42))
            a2 = self._run(p, _diff(seed=42))
            a3 = self._run(p, _diff(seed=43))
        np.testing.assert_array_equal(a1["lon"], a2["lon"])
        np.testing.assert_array_equal(a1["lat"], a2["lat"])
        assert (a1["lon"] != a3["lon"]).any()   # seed matters

    def test_m1_split_rk_emits_phase_c_warning(self, tmp_path):
        """Pin the SPLIT_RK warning specifically — a bare
        pytest.warns(RuntimeWarning) is satisfied by the store's
        unrelated eta=0 warning (review: the old smoke pinned nothing)."""
        p = _make_store_with_kh(tmp_path / "s.nc")
        with pytest.warns(RuntimeWarning, match="split_rk"):
            a = self._run(p, _diff(scheme="m1", seed=7))
        assert (a["status"][-1] == ParticleStatus.ALIVE).sum() >= 20

    def test_em_euler_combined_still_rejected(self, tmp_path):
        """combined (em+Euler) is impossible on healpix: the
        `uv_interp != "leastsq"` guard of _apply_combined_em_step fires
        first (the node-layout guard sits right behind it)."""
        p = _make_store_with_kh(tmp_path / "s.nc")
        reader, mesh, ds = _open(p)
        st = make_state(np.array([30.0]), np.array([30.0]))
        num = NumericsConfig(dt=86400.0, integrator="euler",
                             output_freq=1, uv_interp="barycentric",
                             diffusion=_diff(composition="combined"))
        with pytest.raises(ValueError, match="leastsq"):
            run_single_realization(mesh, ds, reader, st, _mode(), num, 0,
                                   n_steps=1, n_time_total=3,
                                   on_exhausted="cycle")
        ds.close()


class TestExtractCarriesKh:
    def test_extract_with_kh_listed_round_trips(self, tmp_path):
        """Regional extracts carry Kh ONLY when listed under
        `variables:` (the extractor's default family does not include
        it — documented in docs/healpix.md).  Listed, the extract feeds
        em/m1 exactly like the whole store."""
        from extract_healpix_region import extract_region
        src = _make_store_with_kh(tmp_path / "src.nc", with_eta=True)
        out = tmp_path / "sector.zarr"
        extract_region({
            "source": {"path": str(src)},
            "region": {"bbox": [0.0, 90.0, 0.0, 55.0], "halo_layers": 1},
            "output": {"path": str(out), "time_chunk": 2},
            "metadata": {"subset_name": "t", "description": "test"},
            "variables": ["avg_uoe", "avg_von", "avg_zos",
                          "Kh_zonal", "Kh_meridional"],
        })
        reader, mesh, ds = _open(out)
        vel = reader.read_velocities(ds, 0, mesh, _mode(),
                                     diffusion=_diff())
        ds.close()
        assert vel.kh_layout == "node"
        expect = _kh_field(np.mod(mesh.nodes_lon, 360.0),
                           mesh.nodes_lat)
        np.testing.assert_allclose(vel.kh_zonal, expect, rtol=1e-12)
