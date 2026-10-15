"""Phase 5b — statistical validation of HEALPix tracking against the
NATIVE CORE2 twin (docs/healpix.md §5; the twin stores are described by the
AGENTS.md shipped next to them).

Same 200 seeds, 30 daily RK2 steps of 2005 in the North Atlantic, run on
the native FESOM2 output and on its HEALPix remaps (z6 ~102 km,
z7 ~51 km, z8 ~26 km — the last matching CORE2's ~27 km regional
resolution).  Individual trajectories are chaotic — the pins are
STATISTICS.

Honesty of the frozen bounds (review 2026-08-16): the runs are
DETERMINISTIC (verified bit-identical across NUMBA_NUM_THREADS
1..64 — every prange writes per-particle slots, no reductions), so the
measured values reproduce exactly:

    displacement-median ratio to native: z6 1.006, z7 1.000, z8 0.997;
    median paired final-position separation: z6 5.5, z7 1.7, z8 0.5 km;
    mean drift vectors agree to 0.004-0.010 deg on a ~2 deg drift;
    alive 200/200 everywhere.

Against RESAMPLING the seed set (5 independent draws measured), the
spreads are: ratio dev up to 0.033 (z6) / 0.013 (z7); median sep
5.4-7.4 (z6) / 1.7-2.3 km (z7); drift dev up to 0.036 deg; ordering and
alive robust 5/5.  The bounds below keep >=1.5x headroom against that
spread — NOT the naive measured-value margins, which for the drift pin
would be only ~1.4x.

The suite is deliberately paired with test_time_axes_aligned: a uniform
+1-day time-index offset moves the z7 separation only 1.7 -> 7.1 km
(inside the bound) and shifts both grids together, so the statistical
pins alone are blind to it — the axis-equality assertion is what
catches that whole bug class.

@slow (four 30-day runs, ~25 s warm) + gated on KELP_TEST_DATA / KELP_TEST_NATIVE_DATA.
"""
from pathlib import Path
from types import SimpleNamespace

import os

import numpy as np
import pytest

# Set KELP_TEST_DATA to the directory of the CORE2 HEALPix twin stores and
# KELP_TEST_NATIVE_DATA to the native FESOM2 CORE2 output; unset -> skipped.
_DATA = os.environ.get("KELP_TEST_DATA", "")
_Z6 = f"{_DATA}/lagdrift_core2_2005_z6.nc"  # kept on purpose: a real name on disk / in the sibling clone, NOT renamed with the package
_Z7 = f"{_DATA}/lagdrift_core2_2005_z7.nc"
_Z8 = f"{_DATA}/lagdrift_core2_2005_z8.nc"
_NATIVE = os.environ.get("KELP_TEST_NATIVE_DATA", "")
_NATIVE_FILES = ("unod.fesom.2005.nc", "vnod.fesom.2005.nc",
                 "ssh.fesom.2005.nc", "fesom.mesh.diag.nc")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (_DATA and _NATIVE and Path(_Z6).exists() and Path(_Z7).exists()
             and Path(_Z8).exists()
             and all(Path(f"{_NATIVE}/{f}").exists()
                     for f in _NATIVE_FILES)),
        reason="CORE2 native+healpix twin data not available"),
]

from kelp.core.particles import ParticleStatus, make_state
from kelp.readers import get_reader
from kelp.tracking import run_single_realization
from kelp.core.config import NumericsConfig

N_SEEDS = 200
N_DAYS = 30
_rng = np.random.default_rng(2005)
SEED_LON = _rng.uniform(-38.0, -22.0, N_SEEDS)
SEED_LAT = _rng.uniform(38.0, 50.0, N_SEEDS)


def _disp_km(arrs):
    return np.hypot(
        (arrs["lon"][-1] - arrs["lon"][0])
        * np.cos(np.deg2rad(arrs["lat"][0])),
        arrs["lat"][-1] - arrs["lat"][0]) * 111.12


def _native_dir(base) -> Path:
    """Symlink ONLY the 2005 files so the multiyear glob stays cheap."""
    nd = Path(base) / "native2005"
    nd.mkdir(exist_ok=True)
    for f in _NATIVE_FILES:
        dst = nd / f
        if not dst.exists():
            dst.symlink_to(f"{_NATIVE}/{f}")
    return nd


def _native_mode():
    return SimpleNamespace(Fdrift2D=True, Fdrift3D=False, time_dir=1.0,
                           sigma_layer=None, var_u="unod", var_v="vnod",
                           var_w=None, var_eta=None, sample_vars=[],
                           max_levels=None, Vpuls=False)


def _run_healpix(store):
    reader = get_reader("healpix")
    mode = SimpleNamespace(Fdrift2D=True, Fdrift3D=False, time_dir=1.0,
                           sigma_layer=None, var_u=None, var_v=None,
                           var_w=None, var_eta=None, sample_vars=[],
                           max_levels=None)
    mesh = reader.load_mesh(
        SimpleNamespace(meshpath=store, healpix_zoom=None, crop_bbox=None,
                        levels=None, rotation=None, r_earth=None), mode)
    ds = reader.open_dataset(store)
    num = NumericsConfig(dt=86400.0, integrator="rk2", output_freq=1,
                         uv_interp="barycentric")
    rec = run_single_realization(
        mesh, ds, reader, make_state(SEED_LON.copy(), SEED_LAT.copy()),
        mode, num, 0, n_steps=N_DAYS, n_time_total=365)
    ds.close()
    return rec.to_arrays()


def _run_native(tmp_dir):
    nd = _native_dir(tmp_dir)
    reader = get_reader("fesom2")
    mode = _native_mode()
    mesh = reader.load_mesh(SimpleNamespace(
        meshpath=str(nd / "fesom.mesh.diag.nc"), rotation=None,
        r_earth=None))
    ds = reader.open_dataset(str(nd), mode_cfg=mode)
    num = NumericsConfig(dt=86400.0, integrator="rk2", output_freq=1,
                         uv_interp="barycentric")
    rec = run_single_realization(
        mesh, ds, reader, make_state(SEED_LON.copy(), SEED_LAT.copy()),
        mode, num, 0, n_steps=N_DAYS, n_time_total=365)
    ds.close()
    return rec.to_arrays()


@pytest.fixture(scope="module")
def twin_runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("native_twin")
    return {"native": _run_native(tmp),
            "z6": _run_healpix(_Z6),
            "z7": _run_healpix(_Z7),
            "z8": _run_healpix(_Z8)}


class TestNativeTwinStatistics:
    def test_time_axes_aligned(self, tmp_path):
        """Index t means the same calendar instant on every grid.  The
        statistical pins below are BLIND to a uniform time-index offset
        (measured: a +1-day shift moves the z7 median separation only
        1.7 -> 7.1 km, still inside its bound, and shifts both zooms
        together so the ordering holds) — this axis-equality assertion
        is what catches that entire bug class."""
        import xarray as xr
        nd = _native_dir(tmp_path)
        nat_reader = get_reader("fesom2")
        ds_nat = nat_reader.open_dataset(str(nd), mode_cfg=_native_mode())
        t_nat = np.asarray(ds_nat["time"].values)
        for store in (_Z6, _Z7, _Z8):
            with xr.open_dataset(store) as d:
                assert np.array_equal(np.asarray(d["time"].values),
                                      t_nat), store
        ds_nat.close()

    def test_all_alive_everywhere(self, twin_runs):
        for name, arrs in twin_runs.items():
            alive = arrs["status"][-1] == ParticleStatus.ALIVE
            # measured 200/200 on every grid (and 5/5 seed redraws)
            assert alive.sum() >= 0.95 * N_SEEDS, name

    def test_displacement_median_matches_native(self, twin_runs):
        d_nat = np.median(_disp_km(twin_runs["native"]))
        assert d_nat > 50.0                      # the flow genuinely moves
        # measured devs: z6 0.006, z7 0.0003, z8 0.003; seed-resampling
        # spread up to 0.033 (z6) / 0.013 (z7).  Bounds keep >=1.5x
        # headroom against resampling while catching a +1-day z7 shift
        # (dev 0.028), 5% velocity bias (0.037-0.046) and an off-by-one
        # vertical level (0.075) — review 2026-08-16 mutation matrix.
        for name, tol in (("z6", 0.06), ("z7", 0.02), ("z8", 0.02)):
            ratio = np.median(_disp_km(twin_runs[name])) / d_nat
            assert abs(ratio - 1.0) < tol, (name, ratio)

    def _median_sep(self, twin_runs, name):
        nat, hp = twin_runs["native"], twin_runs[name]
        both = ((nat["status"][-1] == ParticleStatus.ALIVE)
                & (hp["status"][-1] == ParticleStatus.ALIVE))
        sep = np.hypot(
            (hp["lon"][-1] - nat["lon"][-1])
            * np.cos(np.deg2rad(nat["lat"][-1])),
            hp["lat"][-1] - nat["lat"][-1]) * 111.12
        return float(np.median(sep[both]))

    def test_paired_separation_bounded(self, twin_runs):
        # measured medians: z6 5.5, z7 1.7, z8 0.5 km (30 days of
        # chaos); seed-resampling ranges 5.4-7.4 / 1.7-2.3 km (z6/z7)
        assert self._median_sep(twin_runs, "z6") < 30.0
        assert self._median_sep(twin_runs, "z7") < 15.0
        assert self._median_sep(twin_runs, "z8") < 5.0

    def test_zoom_ordering(self, twin_runs):
        """Finer zoom tracks the native run strictly closer.  This is
        the assertion that catches a fault affecting ONLY a finer grid —
        the other pins are blind to it (verified: z7 alone shifted one
        day passes all four, only the ordering fires).  Robust to seed
        choice: held 5/5 redraws with a stable ~3x gap."""
        s6 = self._median_sep(twin_runs, "z6")
        s7 = self._median_sep(twin_runs, "z7")
        s8 = self._median_sep(twin_runs, "z8")
        assert s8 < s7 < s6

    def test_mean_drift_vector_matches(self, twin_runs):
        nat = twin_runs["native"]
        nat_alive = nat["status"][-1] == ParticleStatus.ALIVE
        for name in ("z6", "z7", "z8"):
            hp = twin_runs[name]
            m = nat_alive & (hp["status"][-1] == ParticleStatus.ALIVE)
            # SAME subset on both sides (review: different populations
            # would misattribute a data revision to the reader)
            dn = (np.mean(nat["lon"][-1][m] - nat["lon"][0][m]),
                  np.mean(nat["lat"][-1][m] - nat["lat"][0][m]))
            dh = (np.mean(hp["lon"][-1][m] - hp["lon"][0][m]),
                  np.mean(hp["lat"][-1][m] - hp["lat"][0][m]))
            # measured agreement 0.004-0.010 deg; seed-resampling moves
            # it up to ~0.036 deg — 0.10 keeps ~2.8x headroom against
            # resampling (the old 0.05 had only ~1.4x)
            assert abs(dh[0] - dn[0]) < 0.10, name
            assert abs(dh[1] - dn[1]) < 0.10, name
