"""A time-interpolated RK step that ends on a record keeps the time
interpolation, whatever the roundoff of the run clock.

With a ``dt`` that is not a whole number of seconds the run clock
(``sim_time += dt``) carries roundoff, so a step that ends exactly on a record
in exact arithmetic ends a few nanoseconds before or after it.  Before the
fix, a step that ended after the record exceeded the remaining interval by
more than the absolute 1 ns tolerance and fell back to steady velocity for
the whole step; a step that ended before it left the next step a bracket a
few nanoseconds long, so that step fell back.  Each costs metres per step.
Now a step within 1e-9 of ``dt`` of the record ends on it, and a step that
starts within that band before a record starts on it.

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

from kelp.core.geometry import METERS_PER_DEG_LAT  # noqa: E402
from kelp.writers import load_trajectories  # noqa: E402

from tests.test_exact_cycle_runs import (_assert_doubled_reference,  # noqa: E402
                                         _monthly_data, _year_cfg)
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
