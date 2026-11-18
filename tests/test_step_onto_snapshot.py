"""A time-interpolated RK step that ends on a record keeps the time
interpolation, whatever the roundoff of the run clock.

With a ``dt`` that is not a whole number of seconds the run clock
(``sim_time += dt``) carries roundoff, so a step that ends exactly on a record
in exact arithmetic ends a few nanoseconds before or after it.  Before the
fix, a step that ended after the record exceeded the remaining interval by
more than the absolute 1 ns tolerance and fell back to steady velocity for
the whole step; a step that ended before it left the next step a bracket a
few nanoseconds long, so that step fell back.  Each costs metres per step.
Now a step within 5e-10 of ``dt`` of the record ends on it, and a step that
starts within that band before a record starts on it (half of the 1e-9 band
``reconstruct_pchip`` accepts, so a PCHIP step that ends on a record stays
inside it).  The resolver's band, the sigma surface and a step that passes the
record by more than the band are pinned too.

Fixture: the FESOM 1.4 monthly means of ``tests/test_exact_cycle_runs.py``
(uniform ``uo = 2e-4 * (1 + k)`` m/s for record ``k``, ``vo = wo = 0``,
``noleap`` months, stamps at the window centres), so the eastward
displacement of an RK2 run with linear time interpolation is the integral of
a piecewise linear velocity, known exactly.  The numbers of steps per
interval are ones whose steps fell back before the fix: forward 7 (the last
step, onto the last record, overshot it), forward 11 (step 11 started a hair
before record 1), backward 22 (step 21, onto record 1, overshot it), and 13
backward across the cycle seam (the case of the final review of the exact
cycle period).  All data is synthetic, written in ``tmp_path``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("netCDF4")
pytest.importorskip("cftime")

import xarray as xr  # noqa: E402

from kelp.core.geometry import METERS_PER_DEG_LAT  # noqa: E402
from kelp.tracking import _resolve_time_bracket  # noqa: E402
from kelp.writers import load_trajectories  # noqa: E402

from tests.test_exact_cycle_runs import (EDGES,  # noqa: E402
                                         _assert_doubled_reference,
                                         _monthly_data, _sigma_cfg,
                                         _sigma_file, _year_cfg)
from tests.test_fesom14_tracking import DAY, SEEDS, _run  # noqa: E402

#: the first two intervals between the record centres (Jan-Feb, Feb-Mar)
INTERVAL = 29.5 * DAY
#: the seam of the noleap cycle, December centre -> January centre
SEAM = 31 * DAY


def _u(k):
    return 2e-4 * (1 + k)


def _east_m(traj) -> np.ndarray:
    """Eastward displacement of each particle at the last record, m
    (``load_trajectories`` arrays are particle x time)."""
    lat0 = np.array([s[1] for s in SEEDS])
    lon0 = np.array([s[0] for s in SEEDS])
    return ((traj["lon"][:, -1] - lon0) * METERS_PER_DEG_LAT
            * np.cos(np.deg2rad(lat0)))


def _run_quiet(cfg):
    """Run; return the fallback warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _run(cfg)
    return [str(w.message) for w in caught
            if "fall back to steady" in str(w.message)]


@pytest.mark.parametrize("direction, n", [(1, 7), (1, 11), (-1, 22)])
def test_steps_onto_records_keep_time_interpolation(tmp_path, direction, n):
    """2n steps of dt = 29.5 d / n across two intervals, stop mode."""
    year = _monthly_data(tmp_path / "year")
    start = 0 if direction > 0 else 2
    cfg = _year_cfg(tmp_path, year, "run.nc", n_steps=2 * n,
                    on_exhausted="stop", start_time_idx=start,
                    mode={"time_dir": float(direction)},
                    numerics={"dt": INTERVAL / n})
    fallbacks = _run_quiet(cfg)
    assert not fallbacks, fallbacks
    east = _east_m(load_trajectories(cfg.data.output_path))
    exact = direction * INTERVAL * ((_u(0) + _u(1)) / 2 + (_u(1) + _u(2)) / 2)
    assert np.allclose(east, exact, rtol=0, atol=1e-3), (east, exact)


def test_backward_cycle_onto_the_seam_end_keeps_time_interpolation(tmp_path):
    """13 steps of dt = 31 d / 13 backward from January through the seam:
    the cycle run, the run on the records written twice and the cycle run
    started one pass later agree, and the displacement is exact."""
    year = _monthly_data(tmp_path / "year")
    twice = _monthly_data(tmp_path / "twice", copies=2)
    kw = dict(n_steps=13, mode={"time_dir": -1.0},
              numerics={"dt": SEAM / 13})
    cyc = _year_cfg(tmp_path, year, "cycle.nc", on_exhausted="cycle", **kw)
    ref = _year_cfg(tmp_path, twice, "ref.nc", on_exhausted="stop",
                    start_time_idx=12, **kw)
    later = _year_cfg(tmp_path, year, "later.nc", on_exhausted="cycle",
                      start_time_idx=12, **kw)
    fallbacks = [_run_quiet(c) for c in (cyc, ref, later)]
    got, want, shifted = (load_trajectories(c.data.output_path)
                          for c in (cyc, ref, later))
    exact = -SEAM * (_u(11) + _u(0)) / 2
    assert fallbacks == [[], [], []], fallbacks
    _assert_doubled_reference(got, want)
    _assert_doubled_reference(shifted, want)
    assert np.allclose(_east_m(got), exact, rtol=0, atol=1e-3), (
        _east_m(got), exact)


def test_pchip_step_onto_a_record_keeps_the_reconstruction(tmp_path):
    """Backward 22 steps per interval with 4-snapshot PCHIP in time: the step
    onto record 1 passes it by the clock's roundoff, inside the band
    ``reconstruct_pchip`` accepts, and keeps the reconstruction."""
    year = _monthly_data(tmp_path / "year")
    cfg = _year_cfg(tmp_path, year, "pchip.nc", n_steps=44,
                    on_exhausted="stop", start_time_idx=2,
                    mode={"time_dir": -1.0},
                    numerics={"dt": INTERVAL / 22,
                              "temporal_reconstruction": "pchip"})
    assert not _run_quiet(cfg)
    assert np.isfinite(load_trajectories(cfg.data.output_path)["lon"]).all()


# --- the bracket resolver's band, the sigma surface, PCHIP's band ---------

#: record centres of the noleap months (``EDGES`` of the exact-cycle tests), s
_FT = ((EDGES[:-1] + EDGES[1:]) / 2) * DAY
_YEAR = (365 * DAY, 31 * DAY)


def test_bracket_band_backward_shift_beyond_the_old_microsecond():
    """Backward, 50 us above record 3: inside a band of 200 us the query is
    on the record (the interval below it, position at its top); the old
    fixed band of 1 us left it in the interval above."""
    q = _FT[3] + 5e-5
    got = _resolve_time_bracket(q, _FT, 12, "stop", direction=-1,
                                cycle=None, snap=2e-4)
    assert got == (2, 3, _FT[2], _FT[3], _FT[3])
    assert _resolve_time_bracket(q, _FT, 12, "stop", direction=-1,
                                 cycle=None, snap=0.0) == (
        3, 4, _FT[3], _FT[4], q)


@pytest.mark.parametrize("on_exhausted, cycle, q, want", [
    # stop: 100 us before the last record = on it (the collapsed bracket)
    ("stop", None, _FT[-1] - 1e-4, (11, 11, _FT[11], _FT[11], _FT[11])),
    # exact cycle: 100 us before the end of the seam = record 0 of the next
    # pass, position at the bottom of [0, 1]
    ("cycle", _YEAR, _FT[0] + 365 * DAY - 1e-4,
     (0, 1, _FT[0], _FT[1], _FT[0])),
    # mid-axis
    ("stop", None, _FT[4] - 1e-4, (4, 5, _FT[4], _FT[5], _FT[4])),
])
def test_bracket_band_forward_shift(on_exhausted, cycle, q, want):
    assert _resolve_time_bracket(q, _FT, 12, on_exhausted, direction=1,
                                 cycle=cycle, snap=2e-4) == want


def test_bracket_band_leaves_queries_outside_it_alone():
    for q in (_FT[4] - 1e-3, _FT[4] + 1e-3, _FT[7] - 3e-4):
        for d in (1, -1):
            assert (_resolve_time_bracket(q, _FT, 12, "stop", direction=d,
                                          cycle=None, snap=2e-4)
                    == _resolve_time_bracket(q, _FT, 12, "stop", direction=d,
                                             cycle=None, snap=0.0)), (q, d)


def test_sigma_steps_onto_records_keep_time_interpolation(tmp_path):
    """FESOM-C sigma run, records 1800 s apart, dt = 1800 s / 7: before the
    fix step 28 started 2e-12 s before record 4 and fell back."""
    path = _sigma_file(tmp_path / "sig", n_rec=7)
    cfg = _sigma_cfg(path, n_steps=42, dt=1800.0 / 7, temporal_interp=True)
    assert not _run_quiet(cfg)
    assert np.isfinite(load_trajectories(cfg.data.output_path)["z"]).all()


def test_pchip_step_longer_than_its_record_interval_falls_back(tmp_path):
    """One RK4 step of 10800.0000108 s from a record on records 10800 s apart,
    PCHIP in time: the step passes the next record by 1.08e-5 s, beyond the
    band, and falls back to steady velocity with the warning, as before the
    fix — it must not end on the record, where its last stage would lie
    outside the 1e-9 band reconstruct_pchip accepts around its interval."""
    src = _sigma_file(tmp_path / "src", n_rec=7)
    with xr.open_dataset(src, decode_times=False) as d:
        one = d.load()
    one = one.assign_coords(time=("time", np.arange(7) * 10800.0,
                                  {"units": "seconds since 2001-01-01",
                                   "calendar": "noleap"}))
    (tmp_path / "run").mkdir()
    path = tmp_path / "run" / "fesomc.nc"
    one.to_netcdf(path)
    for start, d in ((1, 1.0), (3, -1.0)):
        cfg = _sigma_cfg(path, start=start, n_steps=1, dt=10800.0000108,
                         out=f"p{start}.nc", integrator="rk4",
                         temporal_interp=True,
                         temporal_reconstruction="pchip", time_dir=d)
        fallbacks = _run_quiet(cfg)
        assert len(fallbacks) == 1 and "step 0" in fallbacks[0], fallbacks
