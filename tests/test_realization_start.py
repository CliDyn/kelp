"""Start time of each realization in a multi-realization run.

``time.num_runs`` realizations start ``time.run_interval`` records apart, so
a later start index may leave the records of the forcing.  Under
``on_exhausted: cycle`` it wraps by whole forcing periods (the start time is
the wrapped record time plus or minus the period); under ``stop`` the run is
refused before the first realization reads or writes anything.  Before, an
interpolated run read ``file_times[idx]`` directly: an IndexError past the
end, and a negative index silently started at the end of the axis without
the period; a discrete ``stop`` run gave a realization of zero steps without
a word.
"""
from __future__ import annotations

import numpy as np
import pytest

from kelp.core.config import (
    AppConfig, DataConfig, MeshConfig, ModeConfig, NumericsConfig, TimeConfig,
)
from kelp.readers import get_reader
from kelp.tracking import run_tracking, track_particles

from tests.test_temporal_interp import _make_fesomc_timed

N_TIME = 4
FILE_DT = 3600.0
PERIOD = N_TIME * FILE_DT          # regular axis: span + last interval


def _cfg(tmp_path, *, start, run_interval, on_exhausted, temporal,
         num_runs=2, out="", start_date=None):
    nc = _make_fesomc_timed(tmp_path, n_time=N_TIME, dt_seconds=FILE_DT,
                            u_val=0.1)
    seeds = tmp_path / "s.dat"
    seeds.write_text("5.0 5.0\n")
    return AppConfig(
        mode=ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0),
        numerics=NumericsConfig(dt=(1800.0 if temporal else FILE_DT),
                                integrator="euler", output_freq=1,
                                temporal_interp=temporal),
        mesh=MeshConfig(type="fesomc", meshpath=str(nc)),
        data=DataConfig(ncinput=str(nc), inicloud=str(seeds),
                        output_path=out, time_averaging="instant"),
        time=TimeConfig(start_time_idx=start, n_steps=1,
                        num_runs=num_runs, run_interval=run_interval,
                        on_exhausted=on_exhausted, start_date=start_date))


@pytest.mark.parametrize("start, run_interval, expected", [
    # realization 1 at index 5 = record 1 of the second period
    (2, 3, PERIOD + 1 * FILE_DT),
    # realization 1 at index -1 = record 3 of the period before
    (0, -1, -PERIOD + 3 * FILE_DT),
])
def test_cycle_wraps_the_start_of_a_later_realization(tmp_path, start,
                                                      run_interval, expected):
    """Interpolated mode: the start time is the wrapped record time with the
    period offset (old: IndexError at index 5; index -1 started at the last
    record, 10800 s, without the period)."""
    records = run_tracking(_cfg(tmp_path, start=start,
                                run_interval=run_interval,
                                on_exhausted="cycle", temporal=True))
    assert len(records) == 2
    assert records[0].release_time[0] == start * FILE_DT
    assert records[1].release_time[0] == expected
    # the record clock follows the start: one step of dt after it
    assert records[1].times[0] == expected + 1800.0


@pytest.mark.parametrize("temporal", [True, False])
@pytest.mark.parametrize("start, run_interval, bad", [(2, 2, 4), (0, -1, -1)])
def test_stop_refuses_a_realization_start_off_the_records(
        tmp_path, temporal, start, run_interval, bad):
    """Under ``stop`` a start index outside the records is refused before
    the first realization runs (old: interpolated mode ran realization 0,
    then an IndexError or a silent start at the end of the axis; discrete
    mode wrote a realization of zero steps)."""
    out = tmp_path / "traj.nc"
    cfg = _cfg(tmp_path, start=start, run_interval=run_interval,
               on_exhausted="stop", temporal=temporal, out=str(out))
    with pytest.raises(ValueError) as exc:
        run_tracking(cfg)
    msg = str(exc.value)
    for part in ("time.num_runs = 2", f"time.run_interval = {run_interval}",
                 f"time index {bad}", f"[0, {N_TIME})", "on_exhausted"):
        assert part in msg, (part, msg)
    # nothing ran: no realization output exists
    assert not (tmp_path / "traj_r0.nc").exists()
    assert not (tmp_path / "traj_r1.nc").exists()


@pytest.mark.parametrize("temporal", [True, False])
def test_stop_names_the_start_index_of_a_single_run(tmp_path, temporal):
    """A single run starting off the records names ``time.start_time_idx``,
    the knob to change, not the realization schedule."""
    cfg = _cfg(tmp_path, start=N_TIME, run_interval=0, on_exhausted="stop",
               temporal=temporal, num_runs=1)
    with pytest.raises(ValueError) as exc:
        run_tracking(cfg)
    msg = str(exc.value)
    for part in ("time.start_time_idx", f"time index {N_TIME}",
                 f"[0, {N_TIME})"):
        assert part in msg, (part, msg)
    assert "num_runs" not in msg, msg


def test_stop_accepts_every_start_on_the_records(tmp_path):
    """The last start on the last record still runs (the bound is
    exclusive at ``n_time_total`` only)."""
    records = run_tracking(_cfg(tmp_path, start=0, run_interval=N_TIME - 1,
                                on_exhausted="stop", temporal=False))
    assert len(records) == 2


def test_stop_refuses_a_start_date_realization_past_the_last_record(tmp_path):
    """With ``time.start_date`` between records (00:30) a realization
    starts at the date shifted by the record times: realization 1 of
    run_interval 3 starts at index 3, a record, but at 03:30, past the last
    record (03:00).  Refused before anything is written (old: realization 0
    ran and realization 1 wrote no record, without a word)."""
    out = tmp_path / "traj.nc"
    cfg = _cfg(tmp_path, start=0, run_interval=3, on_exhausted="stop",
               temporal=True, out=str(out), start_date="2020-01-01T00:30")
    with pytest.raises(ValueError) as exc:
        run_tracking(cfg)
    msg = str(exc.value)
    for part in ("time.start_date", "time.num_runs = 2",
                 "time.run_interval = 3", "12600 s", "[0, 10800] s",
                 "on_exhausted: cycle"):
        assert part in msg, (part, msg)
    assert not (tmp_path / "traj_r0.nc").exists()
    assert not (tmp_path / "traj_r1.nc").exists()


@pytest.mark.parametrize("start_date, run_interval", [
    ("2020-01-01T00:30", 2),       # realization 1 at 02:30
    ("2020-01-01T00:00", 3),       # realization 1 on the last record, 03:00
])
def test_stop_runs_a_start_date_realization_on_the_records(tmp_path,
                                                           start_date,
                                                           run_interval):
    """A realization start inside the records, the last record included,
    still runs."""
    t0 = 1800.0 if start_date.endswith("00:30") else 0.0
    records = run_tracking(_cfg(tmp_path, start=0, run_interval=run_interval,
                                on_exhausted="stop", temporal=True,
                                start_date=start_date))
    assert len(records) == 2
    assert records[0].release_time[0] == t0
    assert records[1].release_time[0] == t0 + run_interval * FILE_DT
    assert records[1].times[0] == t0 + run_interval * FILE_DT + 1800.0


@pytest.mark.parametrize("start", [N_TIME, -1])
def test_direct_interpolated_call_refuses_a_start_off_the_records(tmp_path,
                                                                  start):
    """``track_particles`` (``run_single_realization``) with temporal
    interpolation under ``stop``: refused with the index and the range (old:
    IndexError past the end, a silent start at the last record for -1)."""
    nc = str(_make_fesomc_timed(tmp_path, n_time=N_TIME, dt_seconds=FILE_DT,
                                u_val=0.1))
    reader = get_reader("fesomc")
    mesh = reader.load_mesh(type("M", (), {"meshpath": nc}))
    ds = reader.open_dataset(nc)
    try:
        with pytest.raises(ValueError) as exc:
            track_particles(mesh, ds, reader, lon0=np.array([5.0]),
                            lat0=np.array([5.0]), start_time_idx=start,
                            n_steps=1, dt=1800.0, integrator="euler",
                            temporal_interp=True,
                            file_times=np.arange(N_TIME) * FILE_DT)
    finally:
        ds.close()
    msg = str(exc.value)
    assert f"start_time_idx {start}" in msg and f"[0, {N_TIME})" in msg, msg


def _single_record_cfg(tmp_path, *, temporal, time_dir):
    nc = _make_fesomc_timed(tmp_path, n_time=1, dt_seconds=FILE_DT,
                            u_val=0.1)
    seeds = tmp_path / "s.dat"
    seeds.write_text("5.0 5.0\n")
    return AppConfig(
        mode=ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=time_dir),
        numerics=NumericsConfig(dt=FILE_DT, file_dt=FILE_DT,
                                integrator="euler", output_freq=1,
                                temporal_interp=temporal),
        mesh=MeshConfig(type="fesomc", meshpath=str(nc)),
        data=DataConfig(ncinput=str(nc), inicloud=str(seeds),
                        output_path="", time_averaging="instant"),
        time=TimeConfig(start_time_idx=0, n_steps=1, num_runs=2,
                        run_interval=1, on_exhausted="cycle"))


@pytest.mark.parametrize("time_dir", [1.0, -1.0])
def test_cycle_single_record_realization_starts_one_period_later(tmp_path,
                                                                 time_dir):
    """One record cycled repeats with the period dt (the output's
    ``forcing_cycle_period``): realization 1 of run_interval 1 starts dt
    after realization 0 in interpolated mode as in discrete mode (old
    interpolated: 1 s, a synthetic period of the time bracket)."""
    out = {}
    for temporal in (True, False):
        sub = tmp_path / ("t" if temporal else "d")
        sub.mkdir()
        recs = run_tracking(_single_record_cfg(sub, temporal=temporal,
                                               time_dir=time_dir))
        out[temporal] = (recs[1].release_time[0], recs[1].times[0])
    assert out[True][0] == FILE_DT, out
    assert out[True] == out[False], out


def test_stop_start_date_check_follows_an_irregular_axis(tmp_path):
    """The check advances its record index with the loop: on records at
    0, 3, 4 and 5 h (the shape of monthly means) with start_date 00:30 and
    run_interval 1 the starts are 00:30, 03:30 and 04:30 — all on the
    records, so the run goes ahead (a check that kept adding the first
    interval would reach 06:30 and refuse)."""
    import pandas as pd
    import xarray as xr

    nc0 = _make_fesomc_timed(tmp_path, n_time=4, dt_seconds=FILE_DT,
                             u_val=0.1)
    with xr.open_dataset(nc0) as raw:
        ds = raw.load()
    ds = ds.assign_coords(time=pd.to_datetime(
        ["2020-01-01T00:00", "2020-01-01T03:00", "2020-01-01T04:00",
         "2020-01-01T05:00"]))
    nc = tmp_path / "irregular.nc"
    ds.to_netcdf(nc)
    seeds = tmp_path / "s.dat"
    seeds.write_text("5.0 5.0\n")
    cfg = AppConfig(
        mode=ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0),
        numerics=NumericsConfig(dt=1200.0, integrator="euler", output_freq=1,
                                temporal_interp=True),
        mesh=MeshConfig(type="fesomc", meshpath=str(nc)),
        data=DataConfig(ncinput=str(nc), inicloud=str(seeds), output_path="",
                        time_averaging="instant"),
        time=TimeConfig(start_time_idx=0, n_steps=1, num_runs=3,
                        run_interval=1, on_exhausted="stop",
                        start_date="2020-01-01T00:30"))
    records = run_tracking(cfg)
    assert [r.release_time[0] for r in records] == [1800.0, 12600.0, 16200.0]
