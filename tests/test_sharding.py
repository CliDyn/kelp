"""Particle sharding: split a seed file into contiguous shards, run each
shard as an independent job, concatenate the outputs — the linear-scaling
lever for memory-bandwidth-bound runs (several processes with their own
memory buses instead of one saturated one).

Contracts pinned here:
- ``utils/split_seeds.py``: contiguous block split (order-preserving),
  deterministic, 2/3/4-column seed files pass through verbatim, a meta
  sidecar records offsets;
- ``utils/concat_trajectories.py``: shard Zarr outputs concatenate along
  the trajectory axis in shard order == original seed order; every
  ``(trajectory, obs)`` variable is carried generically; the trajectory
  coordinate is renumbered globally; mismatched time axes fail loud;
- END-TO-END: a deterministic run split into 2 shards and concatenated is
  DATA-IDENTICAL to the unsharded run (every variable's values bit for
  bit; container chunk layout may differ).
  Runs WITH stochastic diffusion are bit-identical too: each shard file
  carries its particle_offset, and the engine keys every random draw by the
  global particle index (TestStochasticShards, 2026-10-04).
"""
import subprocess
import sys
from pathlib import Path

import warnings

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")
zarr = pytest.importorskip("zarr")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "utils"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from split_seeds import split_seeds  # noqa: E402
from concat_trajectories import concat_trajectories  # noqa: E402

from kelp.core.config import (AppConfig, DataConfig, MeshConfig,  # noqa: E402
                                  ModeConfig, NumericsConfig, TimeConfig)
from test_healpix_reader import _make_uv_store  # noqa: E402


class TestSplit:
    def _seedfile(self, tmp_path, n=10, cols=3):
        rng = np.random.default_rng(1)
        arr = np.column_stack([rng.uniform(-170, 170, n),
                               rng.uniform(-60, 60, n),
                               rng.uniform(0, 100, n),
                               rng.uniform(0, 3600, n)])[:, :cols]
        p = tmp_path / "seeds.dat"
        np.savetxt(p, arr, fmt="%.6f")
        return p, arr

    @pytest.mark.parametrize("cols", [2, 3, 4])
    def test_contiguous_order_preserving(self, tmp_path, cols):
        p, arr = self._seedfile(tmp_path, n=10, cols=cols)
        files = split_seeds(p, 3, outdir=tmp_path / "shards")
        assert len(files) == 3
        back = np.vstack([np.loadtxt(f, ndmin=2) for f in files])
        np.testing.assert_array_equal(back, np.loadtxt(p, ndmin=2))
        sizes = [np.loadtxt(f, ndmin=2).shape[0] for f in files]
        assert sum(sizes) == 10 and max(sizes) - min(sizes) <= 1

    def test_meta_offsets(self, tmp_path):
        p, _ = self._seedfile(tmp_path, n=11)
        files = split_seeds(p, 4, outdir=tmp_path / "sh")
        meta = np.load(tmp_path / "sh" / "shards_meta.npz")
        assert meta["n_total"] == 11
        assert list(meta["offsets"]) == [0, 3, 6, 9]
        assert list(meta["counts"]) == [3, 3, 3, 2]

    def test_comments_and_blanks_dropped(self, tmp_path):
        p = tmp_path / "seeds.dat"
        p.write_text("# header comment\n1.0 2.0\n\n3.0 4.0\r\n"
                     "# tail\n5.0 6.0\n")
        files = split_seeds(p, 3, outdir=tmp_path / "sh")
        rows = [np.loadtxt(f, ndmin=2) for f in files]
        assert [r.shape[0] for r in rows] == [1, 1, 1]
        np.testing.assert_array_equal(np.vstack(rows),
                                      [[1, 2], [3, 4], [5, 6]])

    def test_more_shards_than_rows_raises(self, tmp_path):
        p, _ = self._seedfile(tmp_path, n=2)
        with pytest.raises(ValueError, match="shards"):
            split_seeds(p, 5, outdir=tmp_path / "sh")


class TestEndToEndBitwise:
    def _cfg(self, store, seeds_path, out):
        return AppConfig(
            mode=ModeConfig(Fdrift2D=True, Fdrift3D=False),
            numerics=NumericsConfig(uv_interp="barycentric",
                                    integrator="rk2",
                                    temporal_interp=True),
            mesh=MeshConfig(type="healpix", meshpath=str(store)),
            data=DataConfig(ncinput=str(store), inicloud=str(seeds_path),
                            output_path=str(out),
                            time_averaging="instant"),
            time=TimeConfig(start_time_idx=0, n_steps=2),
        )

    def test_two_shards_equal_whole(self, tmp_path):
        from kelp.tracking import run_tracking
        store = _make_uv_store(tmp_path / "3D_t_ocean.nc", u_value=0.3)
        rng = np.random.default_rng(7)
        n = 30
        lon = rng.uniform(-170, 170, n)
        lat = np.degrees(np.arcsin(rng.uniform(-0.6, 0.6, n)))
        seeds = tmp_path / "seeds.dat"
        np.savetxt(seeds, np.column_stack([lon, lat]), fmt="%.6f")

        whole = tmp_path / "whole.zarr"
        run_tracking(self._cfg(store, seeds, whole))

        files = split_seeds(seeds, 2, outdir=tmp_path / "sh")
        shard_outs = []
        for k, f in enumerate(files):
            out = tmp_path / f"shard{k}.zarr"
            run_tracking(self._cfg(store, f, out))
            shard_outs.append(out)

        merged = tmp_path / "merged.zarr"
        concat_trajectories(shard_outs, merged)

        gw = zarr.open_group(str(whole), mode="r")
        gm = zarr.open_group(str(merged), mode="r")
        for name in gw.array_keys():
            if name.startswith("restart_"):
                continue          # per-shard restart state is not merged
            aw = np.asarray(gw[name])
            am = np.asarray(gm[name])
            if name == "trajectory":
                np.testing.assert_array_equal(am, np.arange(n))
                continue
            assert np.array_equal(aw, am, equal_nan=True), name

    def test_mismatched_variable_sets_raise(self, tmp_path):
        # the writer creates every per-particle variable a config will
        # ever have at step 0, so a set difference between shards means
        # the configs differed — that must be an ERROR, never a silent
        # NaN merge (review 2026-08).
        from kelp.tracking import run_tracking
        import shutil
        store = _make_uv_store(tmp_path / "3D_t_ocean.nc", u_value=0.3)
        seeds = tmp_path / "s.dat"
        np.savetxt(seeds, np.array([[10.0, 5.0], [12.0, 6.0]]),
                   fmt="%.6f")
        files = split_seeds(seeds, 2, outdir=tmp_path / "sh")
        outs = []
        for k, f in enumerate(files):
            o = tmp_path / f"s{k}.zarr"
            run_tracking(self._cfg(store, f, o))
            outs.append(o)
        shutil.rmtree(outs[1] / "vel_u", ignore_errors=True)
        with pytest.raises(ValueError, match="variable set"):
            concat_trajectories(outs, tmp_path / "m.zarr")

    @pytest.mark.parametrize("field,values", [
        ("temporal_reconstruction", ("linear", "pchip")),
    ])
    def test_mixed_physics_shards_refused(self, tmp_path, field, values):
        """Review 2026-09-22 (F07): the writer did not stamp
        temporal_reconstruction (nor time_averaging), so shards that differ
        in it passed the attribute check and merged into one file with two
        different trajectory physics."""
        from kelp.tracking import run_tracking
        store = _make_uv_store(tmp_path / "3D_t_ocean.nc", u_value=0.3)
        seeds = tmp_path / "s.dat"
        np.savetxt(seeds, np.array([[10.0, 5.0], [12.0, 6.0]]),
                   fmt="%.6f")
        files = split_seeds(seeds, 2, outdir=tmp_path / "sh")
        outs = []
        for k, (f, val) in enumerate(zip(files, values)):
            cfg = self._cfg(store, f, tmp_path / f"s{k}.zarr")
            if field == "time_averaging":
                cfg.data.time_averaging = val
            else:
                cfg.numerics.temporal_reconstruction = val
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                run_tracking(cfg)
            outs.append(tmp_path / f"s{k}.zarr")
        with pytest.raises(ValueError, match=field):
            concat_trajectories(outs, tmp_path / "m.zarr")

    def test_merged_opens_in_xarray(self, tmp_path):
        import xarray as xr
        from kelp.tracking import run_tracking
        store = _make_uv_store(tmp_path / "3D_t_ocean.nc", u_value=0.3)
        seeds = tmp_path / "s.dat"
        np.savetxt(seeds, np.array([[10.0, 5.0], [12.0, 6.0]]),
                   fmt="%.6f")
        files = split_seeds(seeds, 2, outdir=tmp_path / "sh")
        outs = []
        for k, f in enumerate(files):
            o = tmp_path / f"s{k}.zarr"
            run_tracking(self._cfg(store, f, o))
            outs.append(o)
        merged = tmp_path / "m.zarr"
        concat_trajectories(outs, merged)
        ds = xr.open_zarr(str(merged), consolidated=False)
        assert ds["lon"].dims == ("trajectory", "time")   # format 2
        assert ds["time"].dims == ("time",)
        ds.close()

    def test_all_dormant_shard_vel_backfill_data_identical(self, tmp_path):
        # A shard whose particles stay NOT_RELEASED the whole run records no
        # velocity (integrators early-exit with no alive particles).  Until
        # 2026-10-14 its vel_* were never created and concat back-filled
        # them; now the run declares them up front (NaN) — either way the
        # merge must equal the unsharded run, which records dormants as NaN.
        from kelp.tracking import run_tracking
        store = _make_uv_store(tmp_path / "3D_t_ocean.nc", u_value=0.3)
        seeds = tmp_path / "s.dat"
        # lon lat depth release_sec; second particle releases far beyond
        # the 2-step run -> dormant forever
        seeds.write_text("10.0 5.0 0.0 0.0\n12.0 6.0 0.0 8640000.0\n")
        whole = tmp_path / "whole.zarr"
        run_tracking(self._cfg(store, seeds, whole))
        files = split_seeds(seeds, 2, outdir=tmp_path / "sh")
        outs = []
        for k, f in enumerate(files):
            o = tmp_path / f"d{k}.zarr"
            run_tracking(self._cfg(store, f, o))
            outs.append(o)
        g1 = zarr.open_group(str(outs[1]), mode="r")
        assert (g1["status"][:] == 4).all()          # case is real
        assert np.isnan(g1["vel_u"][:]).all()        # declared, never written
        merged = tmp_path / "m.zarr"
        concat_trajectories(outs, merged)
        gw = zarr.open_group(str(whole), mode="r")
        gm = zarr.open_group(str(merged), mode="r")
        for name in gw.array_keys():
            if name.startswith("restart_") or name == "trajectory":
                continue
            assert np.array_equal(np.asarray(gw[name]),
                                  np.asarray(gm[name]),
                                  equal_nan=True), name

    def test_pulsed_store_refused(self, tmp_path):
        from kelp.tracking import run_tracking
        store = _make_uv_store(tmp_path / "3D_t_ocean.nc", u_value=0.3)
        seeds = tmp_path / "s.dat"
        np.savetxt(seeds, np.array([[10.0, 5.0], [12.0, 6.0]]),
                   fmt="%.6f")
        out = tmp_path / "p.zarr"
        cfg = self._cfg(store, seeds, out)
        cfg.time.n_pulses = 2
        cfg.time.pulse_interval = 86400.0
        run_tracking(cfg)
        with pytest.raises(ValueError, match="n_pulses"):
            concat_trajectories([out], tmp_path / "m.zarr")

    def test_output_aliasing_input_rejected(self, tmp_path):
        d = tmp_path / "a.zarr"
        d.mkdir()
        (d / "zarr.json").write_text("{}")
        with pytest.raises(ValueError, match="aliases"):
            concat_trajectories([d], d)

    def test_glob_order_and_duplicates_rejected(self, tmp_path):
        paths = []
        for k in (0, 1, 10, 2):
            d = tmp_path / f"shard{k}.zarr"
            d.mkdir()
            paths.append(d)
        with pytest.raises(ValueError, match="ascending"):
            concat_trajectories(paths, tmp_path / "m.zarr")
        with pytest.raises(ValueError, match="duplicate"):
            concat_trajectories([paths[0], paths[0]],
                                tmp_path / "m2.zarr")

    def test_netcdf_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="Zarr"):
            concat_trajectories([tmp_path / "a.nc"], tmp_path / "m.zarr")

    def test_mismatched_time_axes_raise(self, tmp_path):
        from kelp.tracking import run_tracking
        store = _make_uv_store(tmp_path / "3D_t_ocean.nc", u_value=0.3)
        seeds = tmp_path / "s.dat"
        np.savetxt(seeds, np.array([[10.0, 5.0], [12.0, 6.0]]),
                   fmt="%.6f")
        files = split_seeds(seeds, 2, outdir=tmp_path / "sh")
        o1, o2 = tmp_path / "a.zarr", tmp_path / "b.zarr"
        run_tracking(self._cfg(store, files[0], o1))
        cfg2 = self._cfg(store, files[1], o2)
        cfg2.time.n_steps = 1                     # different time axis
        run_tracking(cfg2)
        with pytest.raises(ValueError, match="time"):
            concat_trajectories([o1, o2], tmp_path / "m.zarr")


class TestCLI:
    def test_split_cli(self, tmp_path):
        p = tmp_path / "seeds.dat"
        np.savetxt(p, np.random.default_rng(0).uniform(0, 1, (9, 3)),
                   fmt="%.6f")
        r = subprocess.run(
            [sys.executable, str(REPO / "utils" / "split_seeds.py"),
             str(p), "--shards", "3", "--outdir", str(tmp_path / "o")],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert len(list((tmp_path / "o").glob("seeds_shard*.dat"))) == 3


class TestStochasticShards:
    """Option (a), decided 2026-10-04: a shard keys its random draws with the
    GLOBAL particle index (its offset in the unsharded release + the local
    index), so a stochastic run split into shards and merged equals the
    unsharded run bit for bit with ONE common seed."""

    def _cfg(self, store, seeds_path, out, **data_kw):
        from kelp.core.config import DiffusionConfig
        return AppConfig(
            mode=ModeConfig(Fdrift2D=True, Fdrift3D=False),
            numerics=NumericsConfig(
                uv_interp="barycentric", integrator="rk2",
                temporal_interp=True,
                diffusion=DiffusionConfig(enabled=True, scheme="uniform",
                                          kh_zonal=50.0, kh_meridional=20.0,
                                          seed=3)),
            mesh=MeshConfig(type="healpix", meshpath=str(store)),
            data=DataConfig(ncinput=str(store), inicloud=str(seeds_path),
                            output_path=str(out), time_averaging="instant",
                            **data_kw),
            time=TimeConfig(start_time_idx=0, n_steps=2),
        )

    def _seeds(self, tmp_path, n=30):
        rng = np.random.default_rng(7)
        lon = rng.uniform(-170, 170, n)
        lat = np.degrees(np.arcsin(rng.uniform(-0.6, 0.6, n)))
        p = tmp_path / "seeds.dat"
        np.savetxt(p, np.column_stack([lon, lat]), fmt="%.6f")
        return p

    def test_two_stochastic_shards_equal_whole(self, tmp_path):
        from kelp.tracking import run_tracking
        store = _make_uv_store(tmp_path / "3D_t_ocean.nc", u_value=0.3)
        seeds = self._seeds(tmp_path)
        whole = tmp_path / "whole.zarr"
        run_tracking(self._cfg(store, seeds, whole))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            files = split_seeds(seeds, 2, outdir=tmp_path / "sh")
        outs = []
        for k, f in enumerate(files):
            out = tmp_path / f"shard{k}.zarr"
            run_tracking(self._cfg(store, f, out))
            outs.append(out)
        merged = tmp_path / "merged.zarr"
        concat_trajectories(outs, merged)
        gw = zarr.open_group(str(whole), mode="r")
        gm = zarr.open_group(str(merged), mode="r")
        moved = np.asarray(gw["lon"])[:, -1] - np.asarray(gw["lon"])[:, 0]
        assert np.isfinite(moved).all()
        for name in gw.array_keys():
            if name.startswith("restart_") or name == "trajectory":
                continue
            assert np.array_equal(np.asarray(gw[name]), np.asarray(gm[name]),
                                  equal_nan=True), name

    def test_shard_files_carry_their_offset(self, tmp_path):
        from kelp.core.release import read_particle_offset
        seeds = self._seeds(tmp_path, n=11)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            files = split_seeds(seeds, 4, outdir=tmp_path / "sh")
        assert [read_particle_offset(f) for f in files] == [0, 3, 6, 9]
        assert read_particle_offset(seeds) is None

    def test_offset_lands_in_the_output_metadata(self, tmp_path):
        from kelp.tracking import run_tracking
        store = _make_uv_store(tmp_path / "3D_t_ocean.nc", u_value=0.3)
        seeds = self._seeds(tmp_path, n=6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            files = split_seeds(seeds, 2, outdir=tmp_path / "sh")
        out = tmp_path / "s1.zarr"
        run_tracking(self._cfg(store, files[1], out))
        assert zarr.open_group(str(out), mode="r").attrs["particle_offset"] == 3

    def test_config_offset_conflicting_with_the_seed_header_is_refused(self, tmp_path):
        from kelp.tracking import run_tracking
        store = _make_uv_store(tmp_path / "3D_t_ocean.nc", u_value=0.3)
        seeds = self._seeds(tmp_path, n=6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            files = split_seeds(seeds, 2, outdir=tmp_path / "sh")
        with pytest.raises(ValueError, match="particle_offset"):
            run_tracking(self._cfg(store, files[1], tmp_path / "x.zarr",
                                   particle_offset=5))

    def test_concat_refuses_shards_out_of_offset_order(self, tmp_path):
        from kelp.tracking import run_tracking
        store = _make_uv_store(tmp_path / "3D_t_ocean.nc", u_value=0.3)
        seeds = self._seeds(tmp_path, n=6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            files = split_seeds(seeds, 2, outdir=tmp_path / "sh")
        outs = []
        for k, f in enumerate(files):
            o = tmp_path / f"s{k}.zarr"
            run_tracking(self._cfg(store, f, o))
            outs.append(o)
        with pytest.raises(ValueError, match="particle_offset"):
            concat_trajectories(outs[::-1], tmp_path / "m.zarr",
                                check_order=False)

    def test_pulsed_release_with_an_offset_is_refused(self, tmp_path):
        from kelp.tracking import run_tracking
        store = _make_uv_store(tmp_path / "3D_t_ocean.nc", u_value=0.3)
        seeds = self._seeds(tmp_path, n=6)
        cfg = self._cfg(store, seeds, tmp_path / "p.zarr", particle_offset=3)
        cfg.time.n_pulses = 2
        cfg.time.pulse_interval = 3600.0
        with pytest.raises(ValueError, match="particle_offset"):
            run_tracking(cfg)


class TestShardOffsetRoundTwo:
    """Review round of the shard offset (2026-10-04)."""

    def _cfg(self, store, seeds_path, out, *, beach=False, diffusion=True,
             **data_kw):
        from kelp.core.config import BeachingConfig, DiffusionConfig
        num = dict(uv_interp="barycentric", integrator="rk2",
                   temporal_interp=True)
        if diffusion:
            num["diffusion"] = DiffusionConfig(
                enabled=True, scheme="uniform", kh_zonal=50.0,
                kh_meridional=20.0, seed=3)
        if beach:
            num["beaching"] = BeachingConfig(
                enabled=True, criteria=["stochastic"], depth_min=0.3,
                t_beach=1.5 * 86400.0, seed=11)       # daily store: ~half strand per step
        return AppConfig(
            mode=ModeConfig(Fdrift2D=True, Fdrift3D=False),
            numerics=NumericsConfig(**num),
            mesh=MeshConfig(type="healpix", meshpath=str(store)),
            data=DataConfig(ncinput=str(store), inicloud=str(seeds_path),
                            output_path=str(out), time_averaging="instant",
                            **data_kw),
            time=TimeConfig(start_time_idx=0, n_steps=2),
        )

    def _seeds(self, tmp_path, n=30, name="seeds.dat"):
        rng = np.random.default_rng(7)
        lon = rng.uniform(-170, 170, n)
        lat = np.degrees(np.arcsin(rng.uniform(-0.6, 0.6, n)))
        p = tmp_path / name
        np.savetxt(p, np.column_stack([lon, lat]), fmt="%.6f")
        return p

    def _split(self, seeds, k, outdir):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return split_seeds(seeds, k, outdir=outdir)

    def test_stochastic_beaching_shards_equal_whole(self, tmp_path):
        from kelp.tracking import run_tracking
        store = _make_uv_store(tmp_path / "3D_t_ocean.nc", u_value=0.3)
        seeds = self._seeds(tmp_path)
        whole = tmp_path / "whole.zarr"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run_tracking(self._cfg(store, seeds, whole, beach=True,
                                   diffusion=False))
        gw = zarr.open_group(str(whole), mode="r")
        st = np.asarray(gw["status"])[:, -1]
        assert 0 < (st == 2).sum() < st.size        # the draws decide
        outs = []
        for k, f in enumerate(self._split(seeds, 3, tmp_path / "sh")):
            o = tmp_path / f"shard{k}.zarr"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                run_tracking(self._cfg(store, f, o, beach=True,
                                       diffusion=False))
            outs.append(o)
        merged = tmp_path / "merged.zarr"
        concat_trajectories(outs, merged)
        gm = zarr.open_group(str(merged), mode="r")
        for name in gw.array_keys():
            if name.startswith("restart_") or name == "trajectory":
                continue
            assert np.array_equal(np.asarray(gw[name]), np.asarray(gm[name]),
                                  equal_nan=True), name

    def test_resplitting_a_shard_keeps_its_offset(self, tmp_path):
        from kelp.core.release import read_particle_offset
        seeds = self._seeds(tmp_path, n=12)
        first = self._split(seeds, 2, tmp_path / "a")
        again = self._split(first[1], 3, tmp_path / "b")
        assert [read_particle_offset(f) for f in again] == [6, 8, 10]
        meta = np.load(tmp_path / "b" / "shards_meta.npz")
        assert list(meta["offsets"]) == [6, 8, 10]

    def test_sorting_a_shard_is_refused(self, tmp_path):
        from sort_seeds import sort_seeds
        seeds = self._seeds(tmp_path, n=8)
        shard = self._split(seeds, 2, tmp_path / "a")[1]
        with pytest.raises(ValueError, match="shard"):
            sort_seeds(shard)

    def test_hand_split_deterministic_shards_merge_with_a_warning(self, tmp_path):
        from kelp.tracking import run_tracking
        store = _make_uv_store(tmp_path / "3D_t_ocean.nc", u_value=0.3)
        seeds = self._seeds(tmp_path, n=10)
        rows = np.loadtxt(seeds)
        outs = []
        for k, (lo, hi) in enumerate(((0, 4), (4, 10))):
            p = tmp_path / f"hand{k}.dat"
            np.savetxt(p, rows[lo:hi], fmt="%.6f")      # no header
            o = tmp_path / f"hand{k}.zarr"
            run_tracking(self._cfg(store, p, o, diffusion=False))
            outs.append(o)
        with pytest.warns(RuntimeWarning, match="no '# kelp particle_offset' header"):
            concat_trajectories(outs, tmp_path / "m.zarr")

    def test_hand_split_stochastic_shards_are_refused(self, tmp_path):
        from kelp.tracking import run_tracking
        store = _make_uv_store(tmp_path / "3D_t_ocean.nc", u_value=0.3)
        seeds = self._seeds(tmp_path, n=10)
        rows = np.loadtxt(seeds)
        outs = []
        for k, (lo, hi) in enumerate(((0, 4), (4, 10))):
            p = tmp_path / f"hand{k}.dat"
            np.savetxt(p, rows[lo:hi], fmt="%.6f")
            o = tmp_path / f"hand{k}.zarr"
            run_tracking(self._cfg(store, p, o))
            outs.append(o)
        with pytest.raises(ValueError, match="draw random numbers"):
            concat_trajectories(outs, tmp_path / "m.zarr")

    def test_pulsed_release_with_offset_zero_is_allowed(self, tmp_path):
        from kelp.tracking import _resolve_particle_offset
        seeds = self._seeds(tmp_path, n=4)
        cfg = self._cfg("unused", seeds, tmp_path / "p.zarr", particle_offset=0)
        cfg.time.n_pulses = 2
        assert _resolve_particle_offset(cfg) == 0


class TestParticleOffsetParsing:
    def _write(self, tmp_path, text, name="s.dat"):
        p = tmp_path / name
        p.write_text(text)
        return p

    def test_prose_about_the_offset_is_refused_not_read_as_zero(self, tmp_path):
        from kelp.core.release import read_particle_offset
        p = self._write(tmp_path, "# kelp particle_offsets are explained below\n1 2\n")
        with pytest.raises(ValueError, match="malformed"):
            read_particle_offset(p)

    def test_an_unrelated_comment_is_ignored(self, tmp_path):
        from kelp.core.release import read_particle_offset
        p = self._write(tmp_path, "# seeds from the 2026 survey, kelp run 3\n1 2\n")
        assert read_particle_offset(p) is None

    def test_case_of_the_tag_word_is_free(self, tmp_path):
        from kelp.core.release import read_particle_offset
        assert read_particle_offset(self._write(tmp_path, "# KELP particle_offset 5\n1 2\n")) == 5

    def test_trailing_comment_and_bom(self, tmp_path):
        from kelp.core.release import read_particle_offset
        p = tmp_path / "s.dat"
        import codecs
        p.write_bytes(codecs.BOM_UTF8 + "# kelp particle_offset 7  # shard 3\n1 2\n".encode("utf-8"))
        assert read_particle_offset(p) == 7

    @pytest.mark.parametrize("text", [
        "# kelp particle_offset -3\n1 2\n",
        "# kelp particle_offset 2.5\n1 2\n",
        "# kelp particle_offset 3 4\n1 2\n",
        "# kelp particle_offset 1\n# kelp particle_offset 2\n1 2\n",
        "# kelp particle_offset: 5\n1 2\n",
        "# kelp particle_offset=5\n1 2\n",
        "# kelp particle_offset is the first index\n1 2\n",
        "# particle_offset 5\n1 2\n",
        "# kelp: particle_offset 5\n1 2\n",
        "# kelp particle-offset 5\n1 2\n",
        "# kelp particle_offset +5\n1 2\n",
        "# kelp particle_offset 1_000\n1 2\n",
        "# kelp particle_offset 9223372036854775808\n1 2\n",
    ])
    def test_malformed_headers_are_refused(self, tmp_path, text):
        from kelp.core.release import read_particle_offset
        with pytest.raises(ValueError):
            read_particle_offset(self._write(tmp_path, text))

    def test_header_after_data_is_ignored(self, tmp_path):
        from kelp.core.release import read_particle_offset
        p = self._write(tmp_path, "1 2\n# kelp particle_offset 5\n3 4\n")
        assert read_particle_offset(p) is None

    @pytest.mark.parametrize("value,ok", [
        (3, True), (np.int64(4), True), ("12", True), (0, True),
        (-1, False), (True, False), (2.0, False), ("x", False), ("-2", False),
        (2 ** 63, False), ("++3", False),
    ])
    def test_config_key(self, value, ok):
        if ok:
            assert DataConfig(particle_offset=value).particle_offset == int(value)
        else:
            with pytest.raises(ValueError, match="particle_offset"):
                DataConfig(particle_offset=value)

    def test_entry_point_refuses_a_negative_offset(self, tmp_path):
        from kelp.tracking import run_single_realization
        with pytest.raises(ValueError, match="particle_offset"):
            run_single_realization(None, None, None, None, ModeConfig(),
                                   NumericsConfig(), 0, n_steps=1,
                                   particle_offset=-3)


class TestRestartCarriesTheOffset:
    def test_standalone_restart_file(self, tmp_path):
        from kelp.core.particles import make_state
        from kelp.writers import load_restart, write_restart_file
        st = make_state(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        p = str(tmp_path / "r.nc")
        write_restart_file(p, st, 5, 6, particle_offset=40)
        assert load_restart(p)["particle_offset"] == 40
        q = str(tmp_path / "r0.nc")
        write_restart_file(q, st, 5, 6)
        assert load_restart(q)["particle_offset"] is None

    def test_embedded_checkpoint(self, tmp_path):
        from kelp.tracking import run_tracking
        from kelp.writers import load_restart
        store = _make_uv_store(tmp_path / "3D_t_ocean.nc", u_value=0.3)
        rng = np.random.default_rng(1)
        seeds = tmp_path / "s.dat"
        np.savetxt(seeds, np.column_stack([rng.uniform(-90, 90, 6),
                                           rng.uniform(-30, 30, 6)]), fmt="%.6f")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = split_seeds(seeds, 2, outdir=tmp_path / "sh")[1]
        out = tmp_path / "o.zarr"
        cfg = AppConfig(
            mode=ModeConfig(Fdrift2D=True, Fdrift3D=False),
            numerics=NumericsConfig(uv_interp="barycentric", integrator="rk2",
                                    temporal_interp=True, checkpoint_freq=2),
            mesh=MeshConfig(type="healpix", meshpath=str(store)),
            data=DataConfig(ncinput=str(store), inicloud=str(f),
                            output_path=str(out), time_averaging="instant"),
            time=TimeConfig(start_time_idx=0, n_steps=2))
        run_tracking(cfg)
        assert load_restart(str(out))["particle_offset"] == 3


class TestShardOffsetRoundThree:
    """Review round 2 of the shard offset (2026-10-04)."""

    def _cfg(self, store, seeds_path, out, *, diffusion):
        from kelp.core.config import DiffusionConfig
        num = dict(uv_interp="barycentric", integrator="rk2", temporal_interp=True)
        if diffusion:
            num["diffusion"] = DiffusionConfig(enabled=True, scheme="uniform",
                                               kh_zonal=50.0, kh_meridional=20.0,
                                               seed=3)
        return AppConfig(
            mode=ModeConfig(Fdrift2D=True, Fdrift3D=False),
            numerics=NumericsConfig(**num),
            mesh=MeshConfig(type="healpix", meshpath=str(store)),
            data=DataConfig(ncinput=str(store), inicloud=str(seeds_path),
                            output_path=str(out), time_averaging="instant"),
            time=TimeConfig(start_time_idx=0, n_steps=2))

    def _hand_split(self, tmp_path, store, diffusion):
        from kelp.tracking import run_tracking
        rng = np.random.default_rng(7)
        rows = np.column_stack([rng.uniform(-170, 170, 10),
                                np.degrees(np.arcsin(rng.uniform(-0.6, 0.6, 10)))])
        outs = []
        for k, (lo, hi) in enumerate(((0, 4), (4, 10))):
            p = tmp_path / f"hand{k}.dat"
            np.savetxt(p, rows[lo:hi], fmt="%.6f")          # no header
            o = tmp_path / f"hand{k}.zarr"
            run_tracking(self._cfg(store, p, o, diffusion=diffusion))
            outs.append(o)
        return outs

    def test_every_output_says_whether_it_draws_random_numbers(self, tmp_path):
        store = _make_uv_store(tmp_path / "3D_t_ocean.nc", u_value=0.3)
        for diff, flag in ((False, 0), (True, 1)):
            sub = tmp_path / f"d{int(diff)}"
            sub.mkdir()
            outs = self._hand_split(sub, store, diff)
            assert zarr.open_group(str(outs[0]), mode="r").attrs["random_draws"] == flag

    def test_driver_style_stochastic_shards_are_refused(self, tmp_path):
        """A chain driver writes no run_tracking metadata (no diffusion_*):
        the refusal must still fire, from the random_draws flag."""
        store = _make_uv_store(tmp_path / "3D_t_ocean.nc", u_value=0.3)
        outs = self._hand_split(tmp_path, store, True)
        for o in outs:
            g = zarr.open_group(str(o), mode="r+")
            for k in [k for k in g.attrs if k.startswith("diffusion_")]:
                del g.attrs[k]
        with pytest.raises(ValueError, match="draw random numbers"):
            concat_trajectories(outs, tmp_path / "m.zarr")

    def test_splitter_refuses_to_overwrite_shards(self, tmp_path):
        rng = np.random.default_rng(1)
        seeds = tmp_path / "seeds.dat"
        np.savetxt(seeds, rng.uniform(0, 10, (12, 2)), fmt="%.6f")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            files = split_seeds(seeds, 3, outdir=tmp_path / "sh")
            with pytest.raises(ValueError, match="already holds shard files"):
                split_seeds(seeds, 3, outdir=tmp_path / "sh")
            with pytest.raises(ValueError, match="overwrite|already holds"):
                split_seeds(files[2], 3)            # in place, next to its siblings
            again = split_seeds(seeds, 2, outdir=tmp_path / "sh", overwrite=True)
        assert sorted(p.name for p in (tmp_path / "sh").glob("seeds_shard*.dat")) == \
            [p.name for p in again]


class TestShardOffsetRoundFour:
    """Review round 3 of the shard offset (2026-10-04)."""

    HEADERS = [
        "# kelp particle_offset 5", "# KELP particle_offset 5",
        "# kelp particle_offset 7  # shard 3", "# kelp particle_offset: 5",
        "# kelp particle_offset=5", "# particle_offset 5", "# kelp: particle_offset 5",
        "# kelp particle-offset 5", "# kelp particle_offset +5",
        "# kelp particle_offset 1_000", "# kelp particle_offset -0",
        "# kelp particle_offset 9223372036854775807",
        "# kelp particle_offset 9223372036854775808",
        "# kelp particle_offsets are explained", "# seeds, kelp run 3",
        "# lon lat -- particle offset from the coast 2 km",
        "# kelp particle_offset \u0665", "# kelp particle offset 500",
    ]

    def _verdict(self, fn, path):
        try:
            return fn(path)
        except ValueError:
            return "refused"

    def test_the_fallback_parsers_agree_with_the_engine(self, tmp_path, monkeypatch):
        import importlib
        from kelp.core.release import read_particle_offset
        import split_seeds as ss
        import sort_seeds as so
        for k, h in enumerate(self.HEADERS):
            p = tmp_path / f"h{k}.dat"
            p.write_text(h + "\n1 2\n")
            core = self._verdict(read_particle_offset, p)
            monkeypatch.setitem(__import__("sys").modules, "kelp.core.release", None)
            try:
                for mod in (ss, so):
                    assert self._verdict(mod._read_particle_offset, p) == core, (h, mod.__name__)
            finally:
                monkeypatch.undo()
                importlib.import_module("kelp.core.release")

    def test_force_never_deletes_the_input_or_foreign_files(self, tmp_path):
        rng = np.random.default_rng(3)
        d = tmp_path / "sh"
        d.mkdir()
        full = d / "seeds_shards.dat"              # matches a loose glob, not ours
        np.savetxt(full, rng.uniform(0, 10, (12, 2)), fmt="%.6f")
        extra = d / "seeds_shard_extra.dat"
        extra.write_text("1 2\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            files = split_seeds(full, 4, outdir=d)
            split_seeds(full, 2, outdir=d, overwrite=True)
            with pytest.raises(ValueError, match="sibling shards"):
                split_seeds(files[0], 2, outdir=d, overwrite=True)   # in place, even with force
        assert full.exists() and extra.exists()
        assert sorted(p.name for p in d.glob("seeds_shard[0-9]*.dat")) == \
            ["seeds_shard000.dat", "seeds_shard001.dat"]

    def test_independent_shards_merge_only_on_request(self, tmp_path):
        from kelp.core.config import DiffusionConfig
        from kelp.tracking import run_tracking
        store = _make_uv_store(tmp_path / "3D_t_ocean.nc", u_value=0.3)
        rng = np.random.default_rng(7)
        rows = np.column_stack([rng.uniform(-170, 170, 10),
                                np.degrees(np.arcsin(rng.uniform(-0.6, 0.6, 10)))])
        outs = []
        for k, (lo, hi) in enumerate(((0, 4), (4, 10))):
            p = tmp_path / f"ind{k}.dat"
            np.savetxt(p, rows[lo:hi], fmt="%.6f")
            o = tmp_path / f"ind{k}.zarr"
            run_tracking(AppConfig(
                mode=ModeConfig(Fdrift2D=True, Fdrift3D=False),
                numerics=NumericsConfig(
                    uv_interp="barycentric", integrator="rk2", temporal_interp=True,
                    diffusion=DiffusionConfig(enabled=True, scheme="uniform",
                                              kh_zonal=50.0, kh_meridional=20.0,
                                              seed=100 + k)),        # own seed per shard
                mesh=MeshConfig(type="healpix", meshpath=str(store)),
                data=DataConfig(ncinput=str(store), inicloud=str(p),
                                output_path=str(o), time_averaging="instant"),
                time=TimeConfig(start_time_idx=0, n_steps=2)))
            outs.append(o)
        with pytest.raises(ValueError, match="independent"):
            concat_trajectories(outs, tmp_path / "m.zarr")
        with pytest.warns(RuntimeWarning, match="independent"):
            concat_trajectories(outs, tmp_path / "m2.zarr", independent_shards=True)
        g = zarr.open_group(str(tmp_path / "m2.zarr"), mode="r")
        assert g.attrs["independent_shards"] == 1
        assert g["lon"].shape[0] == 10
        assert g.attrs["shard_counts"] == [4, 6]
        assert g.attrs["shard_seeds"] == [[100, None], [101, None]]
        assert "particle_offset" not in g.attrs and "diffusion_seed" not in g.attrs
        # pulsed shards (the restart-split recipe): refused without the option,
        # merged with a warning with it
        for o in outs:
            zarr.open_group(str(o), mode="r+").attrs["n_pulses"] = 2
        with pytest.raises(ValueError, match="n_pulses"):
            concat_trajectories(outs, tmp_path / "m3.zarr")
        with pytest.warns(RuntimeWarning, match="n_pulses"):
            concat_trajectories(outs, tmp_path / "m4.zarr", independent_shards=True)


@pytest.mark.parametrize("criteria,resusp,expected", [
    (["depth"], "never", False), (["stochastic"], "never", True),
    (["depth"], "stochastic", True), (["depth"], "auto_or_stochastic", True),
    (["depth"], "auto", False),
])
def test_random_draws_flag_follows_the_beaching_settings(criteria, resusp, expected):
    from kelp.core.config import BeachingConfig
    from kelp.tracking import _draws_random_numbers
    b = BeachingConfig(enabled=True, criteria=criteria, resuspension=resusp,
                       t_beach=86400.0, t_resuspend=86400.0)
    assert _draws_random_numbers(NumericsConfig(beaching=b)) is expected


class TestShardOffsetRoundFive:
    """Review round 4 of the shard offset (2026-10-04)."""

    def test_independent_shards_with_one_seed_are_refused(self, tmp_path):
        from kelp.core.config import DiffusionConfig
        from kelp.tracking import run_tracking
        store = _make_uv_store(tmp_path / "3D_t_ocean.nc", u_value=0.3)
        rows = np.array([[10.0, 5.0], [20.0, 6.0]])
        outs = []
        for k in range(2):
            p = tmp_path / f"same{k}.dat"
            np.savetxt(p, rows, fmt="%.6f")            # the same rows, the same seed
            o = tmp_path / f"same{k}.zarr"
            run_tracking(AppConfig(
                mode=ModeConfig(Fdrift2D=True, Fdrift3D=False),
                numerics=NumericsConfig(
                    uv_interp="barycentric", integrator="rk2", temporal_interp=True,
                    diffusion=DiffusionConfig(enabled=True, scheme="uniform",
                                              kh_zonal=50.0, kh_meridional=20.0, seed=100)),
                mesh=MeshConfig(type="healpix", meshpath=str(store)),
                data=DataConfig(ncinput=str(store), inicloud=str(p),
                                output_path=str(o), time_averaging="instant"),
                time=TimeConfig(start_time_idx=0, n_steps=2)))
            outs.append(o)
        with pytest.raises(ValueError, match="same diffusion_seed"):
            concat_trajectories(outs, tmp_path / "m.zarr", independent_shards=True)

    def test_splitter_never_writes_through_links(self, tmp_path):
        import os
        rng = np.random.default_rng(5)
        seeds = tmp_path / "seeds.dat"
        np.savetxt(seeds, rng.uniform(0, 10, (12, 2)), fmt="%.6f")
        out = tmp_path / "sh"
        out.mkdir()
        foreign = tmp_path / "important.txt"
        foreign.write_text("keep me\n")
        os.symlink(foreign, out / "seeds_shard000.dat")       # a target is a link
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            split_seeds(seeds, 2, outdir=out, overwrite=True)
        assert foreign.read_text() == "keep me\n"
        assert not (out / "seeds_shard000.dat").is_symlink()
        # the input hard-linked under a target name is refused, not truncated
        out2 = tmp_path / "sh2"
        out2.mkdir()
        os.link(seeds, out2 / "seeds_shard001.dat")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="input|sibling"):
                split_seeds(seeds, 2, outdir=out2, overwrite=True)
        assert len(np.loadtxt(seeds)) == 12
        # a symlink to a shard, re-split in that directory, is an in-place re-split
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            files = split_seeds(seeds, 4, outdir=tmp_path / "sh3")
        link = tmp_path / "sh3" / "myshard.dat"
        os.symlink(files[2], link)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="sibling"):
                split_seeds(link, 2, outdir=tmp_path / "sh3", overwrite=True)
        assert all(f.exists() for f in files)


class TestShardOffsetFinal:
    """Leads of the final review round, 2026-10-04."""

    def _store(self, path, n, attrs):
        g = zarr.open_group(str(path), mode="w")
        g.create_array("lon", data=np.zeros((n, 2)))
        g.create_array("lat", data=np.zeros((n, 2)))
        g.create_array("time", data=np.array([0.0, 1.0]))
        g.attrs.update(attrs)
        return path

    def test_one_shared_seed_is_enough_to_refuse(self, tmp_path):
        base = {"random_draws": 1, "diffusion_enabled": 1, "beaching_enabled": 1,
                "beaching_criteria": "stochastic", "particle_offset": 0}
        a = self._store(tmp_path / "a0.zarr", 3, {**base, "diffusion_seed": 7, "beaching_seed": 1})
        b = self._store(tmp_path / "a1.zarr", 3, {**base, "diffusion_seed": 7, "beaching_seed": 2})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="diffusion_seed 7"):
                concat_trajectories([a, b], tmp_path / "m.zarr", independent_shards=True)

    def test_offsetless_stochastic_shards_warn(self, tmp_path):
        a = self._store(tmp_path / "o0.zarr", 2, {"diffusion_enabled": 1, "diffusion_seed": 1})
        b = self._store(tmp_path / "o1.zarr", 2, {"diffusion_enabled": 1, "diffusion_seed": 1})
        with pytest.warns(RuntimeWarning, match="no particle_offset"):
            concat_trajectories([a, b], tmp_path / "m.zarr")

    def test_sidecar_is_not_written_through_a_link(self, tmp_path):
        import os
        seeds = tmp_path / "s.dat"
        np.savetxt(seeds, np.random.default_rng(0).uniform(0, 1, (6, 2)), fmt="%.6f")
        out = tmp_path / "sh"
        out.mkdir()
        foreign = tmp_path / "keep.npz"
        foreign.write_bytes(b"keep")
        os.symlink(foreign, out / "shards_meta.npz")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            split_seeds(seeds, 2, outdir=out, overwrite=True)
        assert foreign.read_bytes() == b"keep"


def test_a_deterministic_beaching_seed_does_not_block_independent_shards(tmp_path):
    """Final review round: run_tracking stamps beaching_seed whenever
    beaching is on; a shared seed of a deterministic beaching setup keys
    nothing and must not refuse the merge."""
    def store(path, dseed):
        g = zarr.open_group(str(path), mode="w")
        g.create_array("lon", data=np.zeros((2, 2)))
        g.create_array("lat", data=np.zeros((2, 2)))
        g.create_array("time", data=np.array([0.0, 1.0]))
        g.attrs.update({"random_draws": 1, "diffusion_enabled": 1, "diffusion_seed": dseed,
                        "beaching_enabled": 1, "beaching_criteria": "depth",
                        "beaching_resuspension": "auto", "beaching_seed": 5,
                        "particle_offset": 0})
        return path
    a, b = store(tmp_path / "d0.zarr", 1), store(tmp_path / "d1.zarr", 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        concat_trajectories([a, b], tmp_path / "m.zarr", independent_shards=True)


def test_a_spaced_header_is_refused_not_read_as_zero(tmp_path):
    from kelp.core.release import read_particle_offset
    p = tmp_path / "s.dat"
    p.write_text("# kelp particle offset 500\n1 2\n")
    with pytest.raises(ValueError, match="malformed"):
        read_particle_offset(p)
