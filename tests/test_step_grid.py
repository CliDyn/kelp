"""RK2/RK4/RK45 with time interpolation: every record the run crosses lies
on the step grid.

A time-interpolated RK step takes one pair of velocity fields: the field at
its start and the record ahead.  A step that crosses a record without ending
on it would need the next pair for its later stages; it runs on the steady
velocity of its start instead.  The run therefore refuses, when it starts, a
``dt`` whose steps do not land on every record they reach: each must be a
whole number of steps from the start.  Euler has a single stage and is not
refused.

A step that reaches a record to within the clock's roundoff ends exactly on
it and the run clock is set to the record, so the roundoff of
``sim_time += dt`` never accumulates past a record: a run with a ``dt`` that
is not a whole number of seconds, decades along the time axis, keeps the
time interpolation on every step.  All data is synthetic, in ``tmp_path``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("netCDF4")

from kelp.core.geometry import METERS_PER_DEG_LAT  # noqa: E402
from kelp.readers import get_reader  # noqa: E402
from kelp.tracking import _check_step_grid, track_particles  # noqa: E402

from tests.test_temporal_interp import _make_fesomc_timed  # noqa: E402

HOUR = 3600.0
DAY = 86400.0


def _setup(tmp_path, n_time=6, spacing=HOUR, u_val=0.01):
    path = str(_make_fesomc_timed(tmp_path, n_time=n_time,
                                  dt_seconds=spacing, u_val=u_val).resolve())
    reader = get_reader("fesomc")
    mesh = reader.load_mesh(type("M", (), {"meshpath": path}))
    ds = reader.open_dataset(path)
    return reader, mesh, ds


def _track(reader, mesh, ds, file_times, dt, n_steps, integrator="rk4",
           **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rec = track_particles(
            mesh, ds, reader, lon0=np.array([5.0]), lat0=np.array([5.0]),
            n_steps=n_steps, dt=dt, time_dir=kw.pop("time_dir", 1.0),
            integrator=integrator, file_times=np.asarray(file_times, float),
            temporal_interp=kw.pop("temporal_interp", True), **kw)
    fallbacks = [str(w.message) for w in caught
                 if "fall back to steady" in str(w.message)]
    return rec, fallbacks


# --- the startup refusal -----------------------------------------------------

@pytest.mark.parametrize("integrator", ["rk2", "rk4", "rk45"])
def test_a_step_that_misses_the_records_is_refused(tmp_path, integrator):
    """Hourly records, dt = M2/64 = 698.66 s: refused before any step, with
    the record interval and the steps that fit it named."""
    reader, mesh, ds = _setup(tmp_path)
    try:
        with pytest.raises(ValueError, match=r"numerics\.dt") as e:
            _track(reader, mesh, ds, np.arange(6) * HOUR,
                   dt=12.4206012 * HOUR / 64, n_steps=20,
                   integrator=integrator, start_time_idx=0)
        msg = str(e.value)
        assert "3600" in msg and "720" in msg and "euler" in msg, msg
    finally:
        ds.close()


def test_a_step_longer_than_the_record_interval_is_refused(tmp_path):
    """Records 300 s apart, dt = 900 s: every step would cross two."""
    reader, mesh, ds = _setup(tmp_path, spacing=300.0)
    try:
        with pytest.raises(ValueError, match=r"numerics\.dt"):
            _track(reader, mesh, ds, np.arange(6) * 300.0, dt=900.0,
                   n_steps=3, start_time_idx=0)
    finally:
        ds.close()


def test_euler_is_not_refused(tmp_path):
    reader, mesh, ds = _setup(tmp_path)
    try:
        rec, _ = _track(reader, mesh, ds, np.arange(6) * HOUR,
                        dt=12.4206012 * HOUR / 64, n_steps=20,
                        integrator="euler", start_time_idx=0)
        assert np.isfinite(rec.to_arrays()["lon"]).all()
    finally:
        ds.close()


def _ok(**kw):
    base = dict(integrator="rk4", temporal_interp=True, n_time_total=None,
                on_exhausted="stop", cycle=None, direction=1)
    base.update(kw)
    if base["n_time_total"] is None:
        base["n_time_total"] = len(base["file_times"])
    _check_step_grid(**base)


def test_a_start_between_records_must_be_on_the_step_grid():
    """A start half an hour after a record: dt = 600 s reaches the next record
    in 3 steps; dt = 720 s does not (2.5 steps) and is refused."""
    ft = np.arange(6) * HOUR
    _ok(file_times=ft, dt=600.0, sim_time=1800.0, n_steps=12)
    with pytest.raises(ValueError, match="start"):
        _ok(file_times=ft, dt=720.0, sim_time=1800.0, n_steps=10)


def test_monthly_means_need_a_step_dividing_the_half_day():
    """noleap monthly means stamped at the window centres lie 29.5 to 31 days
    apart: 12 h fits every interval, 1 day does not."""
    edges = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    ft = (edges[:-1] + edges[1:]) / 2 * DAY
    _ok(file_times=ft, dt=12 * HOUR, sim_time=ft[0], n_steps=600)
    with pytest.raises(ValueError, match=r"numerics\.dt"):
        _ok(file_times=ft, dt=DAY, sim_time=ft[0], n_steps=300)


def test_only_the_records_the_run_crosses_count():
    """An irregular gap after the end of the run does not refuse it."""
    ft = np.array([0.0, 3600.0, 7200.0, 10000.0])
    _ok(file_times=ft, dt=600.0, sim_time=0.0, n_steps=12)       # ends at 7200
    with pytest.raises(ValueError):
        _ok(file_times=ft, dt=600.0, sim_time=0.0, n_steps=17)   # crosses 10000


def test_backward_runs_and_the_cycle_seam():
    ft = np.arange(4) * HOUR
    _ok(file_times=ft, dt=900.0, sim_time=ft[3], n_steps=12, direction=-1)
    # an exact cycle whose seam (2700 s) is no multiple of dt = 1800 s:
    # refused once the run reaches the seam, not before
    cyc = (ft[-1] - ft[0] + 2700.0, 2700.0)
    _ok(file_times=ft, dt=1800.0, sim_time=0.0, n_steps=6,
        on_exhausted="cycle", cycle=cyc)
    with pytest.raises(ValueError):
        _ok(file_times=ft, dt=1800.0, sim_time=0.0, n_steps=8,
            on_exhausted="cycle", cycle=cyc)


def test_not_checked_without_time_interpolation_or_for_euler():
    ft = np.arange(4) * HOUR
    _ok(file_times=ft, dt=1000.0, sim_time=0.0, n_steps=9, integrator="euler")
    _ok(file_times=ft, dt=1000.0, sim_time=0.0, n_steps=9,
        temporal_interp=False)


# --- the clock lands on the records -------------------------------------------

def test_a_fractional_step_decades_along_the_axis_keeps_interpolating(tmp_path):
    """Records 1 h apart starting 30 years after the first stamp, dt = 1 h/7:
    the roundoff of ``sim_time += dt`` at 9.5e8 s exceeded the old band after a
    few steps and steps onto records fell back to steady velocity.  Each step
    onto a record now ends on it: no fallback, and the displacement is the
    exact integral of the piecewise linear velocity."""
    n_time = 7
    reader, mesh, ds = _setup(tmp_path, n_time=n_time)
    t0 = 30 * 365 * DAY
    ft = np.r_[0.0, t0 + np.arange(n_time - 1) * HOUR]
    try:
        rec, fb = _track(reader, mesh, ds, ft, dt=HOUR / 7, n_steps=7 * 5,
                         start_time_idx=1)
        assert not fb, fb
        lon = rec.to_arrays()["lon"][-1, 0]
        east = (lon - 5.0) * METERS_PER_DEG_LAT * np.cos(np.deg2rad(5.0))
        u = 0.01 * (1.0 + np.arange(n_time))
        exact = HOUR * np.sum((u[1:6] + u[2:7]) / 2)
        assert abs(east - exact) < 1e-3, (east, exact)
    finally:
        ds.close()


def test_a_step_missing_the_interval_by_less_than_the_band_drifts_off():
    """dt = 3600.018 s on hourly records misses each record by 0.018 s, less
    than the snapshot band (0.036 s): the records are counted from the start,
    so the run is accepted while the drift stays inside the band (two records)
    and refused beyond it -- the output's record times (start + n dt) would
    drift off the records by 0.018 s a record."""
    ft = np.arange(6) * HOUR
    _ok(file_times=ft, dt=HOUR * (1 + 5e-6), sim_time=0.0, n_steps=2)
    with pytest.raises(ValueError, match="from the start"):
        _ok(file_times=ft, dt=HOUR * (1 + 5e-6), sim_time=0.0, n_steps=3)


def test_the_clock_is_set_to_each_record(tmp_path):
    """Records 1 s apart 30 years along the axis, dt = 1/7 s: each
    ``sim_time += dt`` rounds up by 5e-8 s there, so after 200 steps without
    a record the clock would sit a whole snapshot band (1e-5 s) past the
    record and the step onto it would fall back.  The clock is set to each
    record it reaches: 245 steps across 35 records keep the time
    interpolation, and the displacement is the exact integral."""
    n_time = 37
    reader, mesh, ds = _setup(tmp_path, n_time=n_time, spacing=1.0)
    t0 = 30 * 365 * DAY
    ft = np.r_[0.0, t0 + np.arange(n_time - 1) * 1.0]
    try:
        rec, fb = _track(reader, mesh, ds, ft, dt=1.0 / 7, n_steps=7 * 35,
                         start_time_idx=1)
        assert not fb, fb
        lon = rec.to_arrays()["lon"][-1, 0]
        east = (lon - 5.0) * METERS_PER_DEG_LAT * np.cos(np.deg2rad(5.0))
        u = 0.01 * (1.0 + np.arange(n_time))
        exact = np.sum((u[1:36] + u[2:37]) / 2)
        assert abs(east - exact) < 1e-4, (east, exact)
    finally:
        ds.close()


# --- review round 1 (finder + verifier) ------------------------------------

def test_a_start_and_a_step_each_inside_the_band_but_together_outside():
    """A start 0.03 s before an hourly record and dt = 3600.05 s: each record
    is within the band (0.036 s) of the step grid counted from the start, but
    the run starts ON the record (the resolver moves it there) and the step
    from it misses the next record by 0.05 s, so it would fall back: refused,
    naming the two records."""
    ft = np.arange(4) * HOUR
    with pytest.raises(ValueError, match=r"records at 3600 s and 7200 s"):
        _ok(file_times=ft, dt=3600.05, sim_time=3599.97, n_steps=1)


def test_the_message_names_dt_with_its_digits():
    ft = np.arange(100) * HOUR
    with pytest.raises(ValueError, match=r"numerics\.dt = 3600\.002 s "):
        _ok(file_times=ft, dt=3600.002, sim_time=0.0, n_steps=25)


def test_a_later_realization_off_the_grid_is_refused_before_any_output(
        tmp_path):
    """Two realizations on noleap monthly means, one record apart, daily
    steps: the first (July -> August centre, 31 d) fits, the second
    (August -> September, 30.5 d) does not.  run_tracking refuses before it
    writes anything, not after the first realization."""
    from pathlib import Path

    from tests.test_exact_cycle_runs import _monthly_data, _timed
    from tests.test_fesom14_tracking import _cfg, _run
    data = _monthly_data(tmp_path / "data")
    cfg = _timed(_cfg(tmp_path, data=data, out="multi.nc"),
                 start_time_idx=6, n_steps=31, on_exhausted="stop",
                 num_runs=2, run_interval=1)
    with pytest.raises(ValueError, match=r"numerics\.dt"):
        _run(cfg)
    out = Path(cfg.data.output_path)
    assert not list(out.parent.glob(out.stem + "*")), list(out.parent.iterdir())


def test_euler_keeps_its_own_clock(tmp_path):
    """Euler is not checked and its steps are not snapped onto the records:
    dt = 720.001 s on hourly records (every fifth step passes a record by
    0.005 s, inside the RK band) gives exactly Euler's sum
    dt * sum(u(n dt)) for u(t) = 0.01 (1 + t / 3600)."""
    reader, mesh, ds = _setup(tmp_path)
    dt, n = 720.001, 20
    try:
        rec, _ = _track(reader, mesh, ds, np.arange(6) * HOUR, dt=dt,
                        n_steps=n, integrator="euler", start_time_idx=0)
        lon = rec.to_arrays()["lon"][-1, 0]
        east = (lon - 5.0) * METERS_PER_DEG_LAT * np.cos(np.deg2rad(5.0))
        exact = dt * np.sum(0.01 * (1.0 + np.arange(n) * dt / HOUR))
        assert abs(east - exact) < 1e-5, (east - exact)
    finally:
        ds.close()


def test_discrete_rk_on_fractional_record_spacing_keeps_the_next_record(
        tmp_path):
    """Discrete mode, RK2, records 3600.1 s apart and dt = 3600.1 s: some
    intervals of the float axis are 1.5e-12 s shorter than dt.  Such a step
    takes the next record as before (the old 5e-10 dt tolerance of discrete
    mode), not the steady velocity."""
    reader, mesh, ds = _setup(tmp_path)
    ft = np.arange(6, dtype=np.float64) * 3600.1
    assert (np.diff(ft) < 3600.1).any()
    try:
        _, fb = _track(reader, mesh, ds, ft, dt=3600.1, n_steps=5,
                       integrator="rk2", start_time_idx=0,
                       temporal_interp=False)
        assert not fb, fb
    finally:
        ds.close()


def test_the_end_on_a_record_across_the_cycle_wrap():
    """Backward from record 0 under cycle the bracket is the seam interval
    at the top of the pass (its start one period above the folded clock);
    forward a clock just below the period end is moved onto record 0 of the
    next pass.  The step end lands on the record next to the clock, not a
    period away."""
    from kelp.tracking import _resolve_time_bracket, _step_end_on_record
    ft = np.arange(4) * HOUR                          # legacy period 4 h
    b = _resolve_time_bracket(0.0, ft, 4, "cycle", direction=-1, cycle=None,
                              snap=0.036)
    assert _step_end_on_record(0.0, b[4], HOUR, -1, ft, 4, "cycle",
                               None) == -HOUR
    q = 4 * HOUR - 0.01
    b = _resolve_time_bracket(q, ft, 4, "cycle", direction=1, cycle=None,
                              snap=0.036)
    assert _step_end_on_record(q, b[4], HOUR, 1, ft, 4, "cycle",
                               None) == 5 * HOUR
