"""Discrete mode refuses a step that differs from the file step.

In discrete mode every step reads the next record while the particles move
by the configured ``dt``: with ``dt`` = 1800 s on hourly files the forcing
ran twice as fast as the particles.  The run only printed a notice in its
verbose summary, and that notice claimed the effective step was the file
step, which is not what the integrator did.  ``run_tracking`` now refuses the
combination before any output exists, naming both remedies.
"""
import numpy as np
import pytest

from kelp.core.config import (
    AppConfig, ModeConfig, NumericsConfig, MeshConfig, DataConfig, TimeConfig)


def _cfg(tmp_path, dt, temporal):
    from tests.test_tracking import _make_fesomc_3x3_with_time
    nc = _make_fesomc_3x3_with_time(tmp_path, n_time=6, dt_seconds=3600.0,
                                    u_val=1.0, v_val=0.5)
    seeds = tmp_path / "s.dat"
    seeds.write_text("5.0 5.0\n")
    return AppConfig(
        mode=ModeConfig(Fdrift2D=True, Fdrift3D=False),
        numerics=NumericsConfig(dt=dt, integrator="rk2", output_freq=1,
                                temporal_interp=temporal),
        mesh=MeshConfig(type="fesomc", meshpath=str(nc)),
        data=DataConfig(ncinput=str(nc), inicloud=str(seeds),
                        output_path=str(tmp_path / "out.nc"),
                        time_averaging="instant"),
        time=TimeConfig(start_time_idx=0, n_steps=2))


def test_discrete_mode_with_dt_unequal_file_step_refuses(tmp_path):
    from kelp.tracking import run_tracking
    cfg = _cfg(tmp_path, dt=1800.0, temporal=False)
    with pytest.raises(ValueError, match="temporal_interp"):
        run_tracking(cfg)
    assert not (tmp_path / "out.nc").exists()


def test_discrete_mode_with_dt_equal_file_step_runs(tmp_path):
    from kelp.tracking import run_tracking
    run_tracking(_cfg(tmp_path, dt=3600.0, temporal=False))
    assert (tmp_path / "out.nc").exists()


def test_interpolated_mode_with_smaller_dt_runs(tmp_path):
    from kelp.tracking import run_tracking
    run_tracking(_cfg(tmp_path, dt=1800.0, temporal=True))
    assert (tmp_path / "out.nc").exists()


def _sr_env(tmp_path):
    from types import SimpleNamespace
    from kelp.readers import get_reader, extract_time_info
    from tests.test_tracking import _make_fesomc_3x3_with_time
    nc = _make_fesomc_3x3_with_time(tmp_path, n_time=6, dt_seconds=3600.0,
                                    u_val=1.0, v_val=0.5)
    reader = get_reader("fesomc")
    mesh = reader.load_mesh(SimpleNamespace(meshpath=str(nc)))
    ds = reader.open_dataset(str(nc))
    times, fdt = extract_time_info(ds)
    return mesh, ds, reader, times, fdt


def test_run_single_realization_refuses_too(tmp_path):
    """The functional entry points (run_single_realization, track_particles,
    resumable-chain drivers) must not bypass the refusal."""
    from kelp.core.config import ModeConfig, NumericsConfig
    from kelp.core.particles import make_state
    from kelp.tracking import run_single_realization
    mesh, ds, reader, times, fdt = _sr_env(tmp_path)
    with pytest.raises(ValueError, match="discrete mode"):
        run_single_realization(
            mesh, ds, reader, make_state(np.array([5.0]), np.array([5.0])),
            ModeConfig(), NumericsConfig(dt=900.0, integrator="rk2"),
            start_time_idx=0, n_steps=2, n_time_total=6,
            temporal_interp=False, file_times=times, file_dt=fdt)
    ds.close()


def test_track_particles_refuses_its_default_dt_on_hourly_files(tmp_path):
    from kelp.tracking import track_particles
    mesh, ds, reader, times, fdt = _sr_env(tmp_path)
    with pytest.raises(ValueError, match="discrete mode"):
        track_particles(mesh, ds, reader, np.array([5.0]), np.array([5.0]),
                        n_steps=2, file_times=times, file_dt=fdt)
    ds.close()


def test_irregular_axis_is_refused_in_discrete_mode():
    """Calendar months (28-31 d) or a daily axis with a gap: no single dt
    matches every record, so every dt misregisters some records (31-d steps
    stretch a year of months to 372 d).  Interpolated mode handles them."""
    from kelp.tracking import _check_discrete_dt
    months = np.cumsum([0, 31, 28, 31, 30, 31]) * 86400.0
    for dt in (86400.0, 30 * 86400.0, 31 * 86400.0):
        with pytest.raises(ValueError, match="irregular"):
            _check_discrete_dt(dt, False, None, months)
    gap = np.array([0.0, 1.0, 2.0, 4.0, 5.0]) * 86400.0
    with pytest.raises(ValueError, match="irregular"):
        _check_discrete_dt(86400.0, False, None, gap)


def test_regular_axis_given_only_as_record_times():
    from kelp.tracking import _check_discrete_dt
    hours = np.arange(5) * 3600.0
    _check_discrete_dt(3600.0, False, None, hours)
    with pytest.raises(ValueError, match="discrete mode"):
        _check_discrete_dt(1800.0, False, None, hours)


def test_single_record_is_never_checked():
    """One snapshot has no record spacing: a steady or cycled single field
    runs with any dt, also when numerics.file_dt is set."""
    from kelp.tracking import _check_discrete_dt
    _check_discrete_dt(600.0, False, 86400.0, None, n_records=1)
    _check_discrete_dt(600.0, False, None, np.array([0.0]))


def test_track_particles_refuses_without_explicit_record_times(tmp_path):
    """The functional API passes no record times by default; the check reads
    the axis from the dataset itself."""
    from kelp.tracking import track_particles
    mesh, ds, reader, times, fdt = _sr_env(tmp_path)
    with pytest.raises(ValueError, match="discrete mode"):
        track_particles(mesh, ds, reader, np.array([5.0]), np.array([5.0]),
                        n_steps=2)                            # default dt 900 s
    ds.close()


def test_regular_axis_tolerates_stamp_jitter():
    from kelp.tracking import _check_discrete_dt
    _check_discrete_dt(86400.0, False, 86437.5, None)            # 0.04 %: fine
    with pytest.raises(ValueError, match="discrete mode"):
        _check_discrete_dt(86400.0, False, 86400.0 * 1.01, None)


def test_interpolated_mode_is_never_checked():
    from kelp.tracking import _check_discrete_dt
    _check_discrete_dt(60.0, True, 86400.0, None)


def test_interpolated_mode_reads_the_axis_from_the_dataset(tmp_path):
    """What the refusal tells a direct caller to do must work as told:
    temporal_interp=True without explicit record times reads the axis from
    the dataset (it raised 'no usable time axis' before)."""
    from kelp.tracking import track_particles
    mesh, ds, reader, times, fdt = _sr_env(tmp_path)
    rec = track_particles(mesh, ds, reader, np.array([5.0]), np.array([5.0]),
                          n_steps=2, dt=900.0, temporal_interp=True)
    ds.close()
    assert rec.to_arrays()["lon"].shape[0] == 2


def test_configured_file_dt_does_not_hide_an_irregular_axis():
    """numerics.file_dt is the spacing of a file WITHOUT a usable time
    axis.  When the axis decodes, its own spacing decides: monthly stamps
    with file_dt = 30 d passed the check and ran the forcing up to ~3 % off
    the particle clock (audit 2026-10-04, F-039)."""
    from kelp.tracking import _check_discrete_dt
    months = np.cumsum([0, 31, 28, 31, 30, 31]) * 86400.0
    with pytest.raises(ValueError, match="irregular"):
        _check_discrete_dt(30 * 86400.0, False, 30 * 86400.0, months)


def test_configured_file_dt_contradicting_the_axis_is_refused():
    """dt = file_dt = 1800 s on an hourly axis: the forcing would run twice
    as fast as the particles although dt 'matches' the configured step."""
    from kelp.tracking import _check_discrete_dt
    hours = np.arange(5) * 3600.0
    with pytest.raises(ValueError, match="discrete mode"):
        _check_discrete_dt(1800.0, False, 1800.0, hours)
    _check_discrete_dt(3600.0, False, 3600.0, hours)        # consistent: fine


def test_axis_without_positive_spacing_leaves_the_decision_to_file_dt():
    """extract_time_info calls an axis with repeated stamps unusable and the
    docs send such files to numerics.file_dt; the check must follow."""
    from kelp.tracking import _check_discrete_dt
    flat = np.zeros(4)
    _check_discrete_dt(86400.0, False, 86400.0, flat)
    with pytest.raises(ValueError, match="discrete mode"):
        _check_discrete_dt(3600.0, False, 86400.0, flat)


def test_direct_caller_with_only_file_dt_is_checked_against_the_axis(tmp_path):
    """run_single_realization read the axis from the dataset only when
    neither file_times nor file_dt was given: dt = file_dt = 1800 s on an
    hourly file passed."""
    from kelp.core.config import ModeConfig, NumericsConfig
    from kelp.core.particles import make_state
    from kelp.tracking import run_single_realization
    mesh, ds, reader, times, fdt = _sr_env(tmp_path)
    with pytest.raises(ValueError, match="discrete mode"):
        run_single_realization(
            mesh, ds, reader, make_state(np.array([5.0]), np.array([5.0])),
            ModeConfig(), NumericsConfig(dt=1800.0, integrator="rk2"),
            start_time_idx=0, n_steps=2, n_time_total=6,
            temporal_interp=False, file_times=None, file_dt=1800.0)
    ds.close()


def test_one_repeated_stamp_is_irregular_not_a_file_dt_case():
    """Only an axis with NO positive spacing goes to numerics.file_dt; one
    repeated stamp in a daily axis is irregular (review round 2)."""
    from kelp.tracking import _check_discrete_dt
    days = np.array([0.0, 1.0, 1.0, 2.0, 3.0]) * 86400.0
    with pytest.raises(ValueError, match="irregular"):
        _check_discrete_dt(3600.0, False, 3600.0, days)
