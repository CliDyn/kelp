"""Configured runs with the exact cycle period (``time.on_exhausted: cycle``).

Twelve monthly means of one ``noleap`` year repeat after 365 d, with a seam of
31 d from the December record to the January record of the next pass (the
legacy rule, the span of the records plus the last interval, gives 364.5 d).
A ``cycle`` run on the twelve records is compared with "the doubled
reference": a ``stop`` run on the same records written twice along time, the
second copy stamped one period later, with the same seeds, ``dt`` and number
of steps.  Discrete mode (one record per step) cannot follow windows of
unequal length and is refused before the output exists.

Fixture: the FESOM 1.4 grid of ``tests/test_fesom14_tracking.py`` (level-wise
files with cell polygons, the mesh recovered from them); monthly means with
time bounds at the month edges and stamps at the window centres, calendar
``noleap``; ``uo = 2e-4 * (1 + k % n_months)`` m/s for record ``k``, uniform
in space, ``vo = wo = 0``.  All data is synthetic, written in ``tmp_path``.
The names of the exact period are imported inside the tests, so the file
also runs on code that lacks them.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("netCDF4")
pytest.importorskip("cftime")

from kelp import tracking  # noqa: E402
from kelp.core.config import MeshConfig, ModeConfig, NumericsConfig, TimeConfig  # noqa: E402
from kelp.core.mesh import find_nearest_nodes  # noqa: E402
from kelp.core.particles import make_state  # noqa: E402
from kelp.readers.fesom14 import Fesom14Reader  # noqa: E402
from kelp.writers import load_trajectories  # noqa: E402

from tests.test_fesom14_tracking import (DAY, LAT, LAT_B, LEVELS, LON,  # noqa: E402
                                         LON_B, SEEDS, WET, _cfg, _quiet,
                                         _run, _write_nc)

#: month edges of one noleap year, days since 2001-01-01
EDGES = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365],
                 dtype=np.float64)


def _month_time_vars(n_months: int, copies: int):
    """Time and bounds of ``n_months`` monthly means from January 2001,
    written ``copies`` times, each copy stamped the sum of the windows
    later."""
    total = EDGES[n_months]
    lo = np.concatenate([EDGES[:n_months] + c * total for c in range(copies)])
    hi = np.concatenate([EDGES[1:n_months + 1] + c * total
                         for c in range(copies)])
    attrs = {"units": "days since 2001-01-01", "calendar": "noleap",
             "standard_name": "time", "bounds": "time_bnds"}
    return ([("time", ("time",), (lo + hi) / 2.0, attrs),
             ("time_bnds", ("time", "bnds"), np.stack([lo, hi], axis=1),
              {})], {"bnds": 2})


def _monthly_data(d: Path, n_months: int = 12, copies: int = 1,
                  tracer: bool = False) -> Path:
    """Level-wise ``uo``, ``vo``, ``wo`` (the u file holds the polygons) and,
    with ``tracer``, ``tracer(time, ncells) = k % n_months``."""
    if d.exists():
        return d
    d.mkdir(parents=True)
    n = n_months * copies
    k = np.arange(n)
    tvars, tdims = _month_time_vars(n_months, copies)
    for var in ("uo", "vo", "wo"):
        rec = 2e-4 * (1 + k % n_months) if var == "uo" else np.zeros(n)
        vals = np.where(WET[None], rec[:, None, None]
                        + 0.0 * LON[None, None, :], np.nan)
        attrs = {"_FillValue": 1e20}
        if var == "wo":
            attrs["standard_name"] = "upward_sea_water_velocity"
        variables = tvars + [
            ("depth", ("depth",), LEVELS, {"units": "m", "positive": "down"}),
            ("lon", ("ncells",), LON, {"standard_name": "longitude",
                                       "bounds": "lon_bnds"}),
            ("lat", ("ncells",), LAT, {"standard_name": "latitude",
                                       "bounds": "lat_bnds"}),
            (var, ("time", "depth", "ncells"), vals, attrs),
        ]
        dims = {"time": None, **tdims, "depth": LEVELS.size,
                "ncells": LON.size}
        if var == "uo":
            dims["vertices"] = LON_B.shape[1]
            variables += [("lon_bnds", ("ncells", "vertices"), LON_B, {}),
                          ("lat_bnds", ("ncells", "vertices"), LAT_B, {})]
        _write_nc(d / f"{var}.nc", dims, variables)
    if tracer:
        values = np.repeat((k % n_months).astype(np.float64)[:, None],
                           LON.size, axis=1)
        _write_nc(d / "tracer.nc", {"time": None, **tdims, "ncells": LON.size},
                  tvars + [("tracer", ("time", "ncells"), values,
                            {"_FillValue": 1e20, "units": "1"})])
    return d


def _year_cfg(tmp_path: Path, data: Path, out: str, *, n_steps: int,
              on_exhausted: str, start_time_idx: int = 0, **kw):
    """The FESOM 1.4 run of ``_cfg`` (3-D RK2, dt = 1 d, lerp in time) from
    record ``start_time_idx``."""
    cfg = _cfg(tmp_path, data=data, out=out, **kw)
    cfg.time = TimeConfig(start_time_idx=start_time_idx, n_steps=n_steps,
                          on_exhausted=on_exhausted)
    return cfg


def _period(path) -> float:
    with xr.open_dataset(path, decode_times=False) as ds:
        return float(ds["time"].attrs["forcing_cycle_period"])


def _assert_doubled_reference(got, ref):
    for key in ("lon", "lat", "z"):
        assert np.allclose(got[key], ref[key], rtol=0, atol=1e-9), (
            key, np.max(np.abs(got[key] - ref[key])))
    assert np.array_equal(got["status"], ref["status"])


def _m_d(period: float, seam: float) -> str:
    return ("time.on_exhausted: cycle with numerics.temporal_interp: false is "
            "refused: the records' averaging windows differ in length, so the "
            f"forcing repeats after {period / DAY:g} d with a seam of "
            f"{seam / DAY:g} d between the last and the first record, which a "
            "discrete run (one record per step of dt) cannot follow. Set "
            "numerics.temporal_interp: true")


def test_fesom14_monthly_cycle_equals_the_records_written_twice(tmp_path):
    """450 daily steps on twelve monthly means: through the seam December ->
    January and into the second pass, as on the records written twice."""
    year = _monthly_data(tmp_path / "year")
    twice = _monthly_data(tmp_path / "twice", copies=2)
    cfg = _year_cfg(tmp_path, year, "cycle.nc", n_steps=450,
                    on_exhausted="cycle")
    ref = _year_cfg(tmp_path, twice, "ref.nc", n_steps=450,
                    on_exhausted="stop")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _run(cfg)
    _run(ref)
    got, want = (load_trajectories(c.data.output_path) for c in (cfg, ref))
    _assert_doubled_reference(got, want)
    assert np.array_equal(got["time"], want["time"])
    assert _period(cfg.data.output_path) == 365 * DAY
    assert np.ptp(got["lon"]) > 0.1                       # it moves
    assert not [w for w in caught if "last interval" in str(w.message)]


def test_fesom14_backward_cycle_across_the_seam(tmp_path):
    """60 daily steps backward from the first record: the run enters the
    seam bracket from its top and goes on in the previous pass."""
    year = _monthly_data(tmp_path / "year")
    twice = _monthly_data(tmp_path / "twice", copies=2)
    back = {"time_dir": -1.0}
    cfg = _year_cfg(tmp_path, year, "cycle.nc", n_steps=60,
                    on_exhausted="cycle", mode=back)
    ref = _year_cfg(tmp_path, twice, "ref.nc", n_steps=60,
                    on_exhausted="stop", start_time_idx=12, mode=back)
    _run(cfg)
    _run(ref)
    got, want = (load_trajectories(c.data.output_path) for c in (cfg, ref))
    _assert_doubled_reference(got, want)
    t_cycle, t_ref = got["time"], want["time"]
    assert np.array_equal(t_ref - t_cycle, np.full_like(t_ref, 365 * DAY))


@pytest.mark.parametrize("n_months, dt_days, period_days", [
    (3, 29.5, 90), (12, 30.0, 365)], ids=["jan_mar", "year"])
def test_discrete_cycle_on_unequal_windows_is_refused(tmp_path, n_months,
                                                       dt_days, period_days):
    """Discrete mode with ``cycle`` on windows of unequal length: refused
    before the output exists (January-March are spaced evenly, 29.5 d, so
    only this refusal stops them; on the year it comes before the refusal
    of irregular record times)."""
    data = _monthly_data(tmp_path / "data", n_months=n_months)
    num = {"temporal_interp": False, "dt": dt_days * DAY}
    cfg = _year_cfg(tmp_path, data, "cycle.nc", n_steps=2,
                    on_exhausted="cycle", numerics=num)
    with pytest.raises(ValueError) as info:
        _run(cfg)
    assert str(info.value) == _m_d(period_days * DAY, 31 * DAY)
    assert not Path(cfg.data.output_path).exists()
    if n_months != 3:
        return

    # the same records under stop: no cycle, no refusal
    cfg = _year_cfg(tmp_path, data, "stop.nc", n_steps=2,
                    on_exhausted="stop", numerics=num)
    _run(cfg)
    assert load_trajectories(cfg.data.output_path)["lon"].shape == (2, 2)

    # a direct caller without file_times meets the same refusal
    mode = ModeConfig(Fdrift2D=False, Fdrift3D=True)
    mesh = _quiet(Fesom14Reader.load_mesh,
                  MeshConfig(type="fesom14", meshpath=str(data)), mode_cfg=mode)
    ds = _quiet(Fesom14Reader.open_dataset, str(data), mode_cfg=mode)
    try:
        numerics = NumericsConfig(dt=dt_days * DAY, integrator="rk2",
                                  uv_interp="barycentric",
                                  w_interp="barycentric",
                                  temporal_interp=False, output_freq=1)
        lon, lat, z = (np.array(c, dtype=np.float64) for c in zip(*SEEDS))
        init = make_state(lon, lat, z)
        init = init.replace(nearest_node=find_nearest_nodes(mesh, lon, lat))
        with pytest.raises(ValueError) as info:
            _quiet(tracking.run_single_realization,
                   mesh, ds, Fesom14Reader, init, mode, numerics,
                   start_time_idx=0, n_steps=2,
                   n_time_total=ds.sizes["time"], on_exhausted="cycle",
                   temporal_interp=False)
        assert str(info.value) == _m_d(90 * DAY, 31 * DAY)
    finally:
        ds.close()
