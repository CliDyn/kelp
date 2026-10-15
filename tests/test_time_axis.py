"""Time-axis decoding and the temporal-interpolation guard (review 2026-09-22, F03).

``extract_time_info`` returned ``(None, None)`` for a cftime noleap axis and
for an integer axis with physical units, and truncated sub-second datetime64
steps; ``run_single_realization`` then ran DISCRETELY although temporal
interpolation was requested (``use_tinterp = temporal_interp and file_times
is not None``) — with an explicit ``dt`` that silently changed the physics.
"""
import numpy as np
import pytest
import xarray as xr

from kelp.readers import extract_time_info


def _ds(time_values, **attrs):
    t = xr.DataArray(time_values, dims=("time",), attrs=attrs)
    return xr.Dataset({"u": (("time",), np.zeros(len(time_values)))},
                      coords={"time": t})


def test_cftime_noleap_axis():
    cftime = pytest.importorskip("cftime")
    vals = np.array([cftime.DatetimeNoLeap(2001, 2, 27 + k) if k < 2 else
                     cftime.DatetimeNoLeap(2001, 3, 1) for k in range(3)],
                    dtype=object)
    secs, dt = extract_time_info(_ds(vals))
    np.testing.assert_allclose(secs, [0.0, 86400.0, 172800.0])
    assert dt == 86400.0


def test_integer_axis_with_units_is_physical():
    secs, dt = extract_time_info(
        _ds(np.array([0, 1, 2]), units="hours since 2001-01-01 00:00:00"))
    np.testing.assert_allclose(secs, [0.0, 3600.0, 7200.0])
    assert dt == 3600.0


def test_integer_index_without_units_is_not_time():
    assert extract_time_info(_ds(np.array([0, 1, 2]))) == (None, None)


def test_subsecond_datetime64_kept():
    t0 = np.datetime64("2001-01-01T00:00:00.000", "ns")
    vals = t0 + np.array([0, 500, 1000], dtype="timedelta64[ms]")
    secs, dt = extract_time_info(_ds(vals))
    np.testing.assert_allclose(secs, [0.0, 0.5, 1.0])
    assert dt == pytest.approx(0.5)


def test_temporal_interp_without_axis_refuses(tmp_path):
    """Requested interpolation + no usable axis -> ValueError, not a silent
    discrete run."""
    from types import SimpleNamespace
    from kelp.core.config import ModeConfig, NumericsConfig
    from kelp.core.particles import make_state
    from kelp.readers import get_reader
    from kelp.tracking import run_single_realization
    from tests.test_tracking import _make_fesomc_3x3
    nc, _m, _mo = _make_fesomc_3x3(tmp_path, n_time=4)   # index-only axis
    reader = get_reader("fesomc")
    mesh = reader.load_mesh(SimpleNamespace(meshpath=str(nc)))
    ds = reader.open_dataset(str(nc))
    times, _ = extract_time_info(ds)
    assert times is None
    with pytest.raises(ValueError, match="time axis"):
        run_single_realization(
            mesh, ds, reader, make_state(np.array([5.0]), np.array([5.0])),
            ModeConfig(), NumericsConfig(dt=60.0, integrator="rk2"),
            start_time_idx=0, n_steps=2, n_time_total=4,
            temporal_interp=True, file_times=times)
    ds.close()


def _cftime_ds():
    cftime = pytest.importorskip("cftime")
    vals = np.array([cftime.DatetimeNoLeap(2001, 2, 27),
                     cftime.DatetimeNoLeap(2001, 2, 28),
                     cftime.DatetimeNoLeap(2001, 3, 1)], dtype=object)
    return _ds(vals)


def test_start_date_on_a_noleap_axis_exact():
    """Review 2026-09-23: decoding cftime axes made _resolve_start_date
    compare cftime with datetime64 -> TypeError."""
    from kelp.tracking import _resolve_start_date
    ds = _cftime_ds()
    times, _ = extract_time_info(ds)
    idx, t = _resolve_start_date(ds, "2001-02-28", times, temporal_interp=False)
    assert idx == 1 and t == 86400.0


def test_start_date_on_a_noleap_axis_between_snapshots():
    from kelp.tracking import _resolve_start_date
    ds = _cftime_ds()
    times, _ = extract_time_info(ds)
    idx, t = _resolve_start_date(ds, "2001-02-28T12:00", times, temporal_interp=True)
    assert idx == 1 and t == pytest.approx(1.5 * 86400.0)


def test_numeric_axis_with_millisecond_units():
    """Review 2026-09-23: 'milliseconds since' passed the 'second'
    substring test and was read as seconds."""
    secs, dt = extract_time_info(
        _ds(np.array([0, 1000, 2000]), units="milliseconds since 2001-01-01"))
    np.testing.assert_allclose(secs, [0.0, 1.0, 2.0])
    assert dt == pytest.approx(1.0)


def test_track_particles_temporal_interp_with_file_dt_only(tmp_path):
    """Review 2026-09-23: a caller that supplies file_dt but no file_times
    got the error that told it to set file_dt."""
    from types import SimpleNamespace
    from kelp.core.config import ModeConfig
    from kelp.readers import get_reader
    from kelp.tracking import track_particles
    from tests.test_tracking import _make_fesomc_3x3
    nc, _m, _mo = _make_fesomc_3x3(tmp_path, n_time=4)
    reader = get_reader("fesomc")
    mesh = reader.load_mesh(SimpleNamespace(meshpath=str(nc)))
    ds = reader.open_dataset(str(nc))
    rec = track_particles(mesh, ds, reader, np.array([5.0]), np.array([5.0]),
                          start_time_idx=0, n_steps=2, dt=1800.0,
                          mode=ModeConfig(), temporal_interp=True,
                          file_dt=3600.0)
    ds.close()
    assert len(rec.to_arrays()["time"]) == 2


@pytest.mark.parametrize("unit,scale", [("h", 3600.0), ("d", 86400.0),
                                        ("min", 60.0), ("s", 1.0),
                                        ("ms", 1e-3), ("us", 1e-6)])
def test_integer_axis_with_a_short_bare_unit_is_physical(unit, scale):
    """Review round 2: the index-axis guard matched only long unit
    substrings, so [0, 1, 2] with units "h" was dropped as an index."""
    secs, dt = extract_time_info(_ds(np.array([0, 1, 2]), units=unit))
    np.testing.assert_allclose(secs, [0.0, scale, 2 * scale])
    assert dt == pytest.approx(scale)


def test_start_date_on_a_360_day_axis():
    """Review round 2: '2001-02-30' exists in the 360_day calendar
    but was parsed as a Gregorian pandas.Timestamp first and rejected."""
    cftime = pytest.importorskip("cftime")
    from kelp.tracking import _resolve_start_date
    vals = np.array([cftime.Datetime360Day(2001, 2, 29 + k) for k in range(2)]
                    + [cftime.Datetime360Day(2001, 3, 1)], dtype=object)
    ds = _ds(vals)
    times, _ = extract_time_info(ds)
    idx, t = _resolve_start_date(ds, "2001-02-30", times, temporal_interp=False)
    assert idx == 1 and t == 86400.0


def test_start_date_on_an_undecoded_numeric_axis():
    """A numeric CF axis left undecoded (360_day units) must be decoded with
    its units and calendar, not subtracted as datetimes."""
    pytest.importorskip("cftime")
    from kelp.tracking import _resolve_start_date
    ds = _ds(np.array([58.0, 59.0, 60.0]),
             units="days since 2001-01-01", calendar="360_day")
    times, _ = extract_time_info(ds)
    np.testing.assert_allclose(times, [0.0, 86400.0, 172800.0])
    # day 59 after 2001-01-01 in a 360-day year = 2001-02-30
    idx, t = _resolve_start_date(ds, "2001-02-30T12:00", times,
                                 temporal_interp=True)
    assert idx == 1 and t == pytest.approx(1.5 * 86400.0)


# --- start_date under time_averaging: mean (audit 2026-10-04, F-038) --------

def _daily_mean_axis(n=4):
    import pandas as pd
    from kelp.tracking import _shift_to_midpoints
    ds = _ds(pd.date_range("2020-01-01", periods=n, freq="1D").values)
    raw, _ = extract_time_info(ds)
    return ds, raw, _shift_to_midpoints(raw)


def test_start_date_with_interpolation_is_the_physical_instant():
    """With temporal interpolation the run starts at the requested instant,
    exact stamp or not: an exact stamp used to start at its record's window
    centre (half a record early) and one second later at the literal date —
    12 h apart on daily means."""
    from kelp.tracking import _resolve_start_date
    ds, raw, shifted = _daily_mean_axis()
    idx, t_exact = _resolve_start_date(ds, "2020-01-03T00:00:00", shifted, True)
    _, t_after = _resolve_start_date(ds, "2020-01-03T00:00:01", shifted, True)
    _, t_before = _resolve_start_date(ds, "2020-01-02T23:59:59", shifted, True)
    assert t_exact == raw[2] == 2 * 86400.0
    assert t_after - t_exact == 1.0 and t_exact - t_before == 1.0
    # the record bracket below the instant: the Jan-3 mean (centre 1.5 d)
    assert idx == 2 and shifted[idx] <= t_exact < shifted[idx + 1]


def test_start_date_in_discrete_mode_starts_at_the_record():
    """Discrete mode reads whole records: an exact stamp starts its record."""
    from kelp.tracking import _resolve_start_date
    ds, raw, shifted = _daily_mean_axis()
    assert _resolve_start_date(ds, "2020-01-03", shifted, False) == (2, shifted[2])


def test_start_date_between_stamps_is_the_literal_instant():
    from kelp.tracking import _resolve_start_date
    ds, raw, shifted = _daily_mean_axis()
    idx, t = _resolve_start_date(ds, "2020-01-02T12:00", shifted, True)
    assert t == 1.5 * 86400.0 and idx == 2      # the Jan-2 window centre


def test_start_date_on_an_instant_axis_is_unchanged():
    from kelp.tracking import _resolve_start_date
    ds, raw, _ = _daily_mean_axis()
    idx, t = _resolve_start_date(ds, "2020-01-02T12:00", raw, True)
    assert idx == 1 and t == 1.5 * 86400.0          # bit for bit, as before
    assert _resolve_start_date(ds, "2020-01-03", raw, True) == (2, raw[2])


def test_start_date_at_the_last_mean_stamp_starts_the_last_record():
    """Past the last window centre there is nothing to interpolate towards:
    the exact last stamp keeps its old start (the last record's centre),
    with a warning, instead of being refused (review round 2)."""
    from kelp.tracking import _resolve_start_date
    ds, raw, shifted = _daily_mean_axis()
    with pytest.warns(RuntimeWarning, match="window centre"):
        idx, t = _resolve_start_date(ds, "2020-01-04", shifted, True)
    assert idx == 3 and t == shifted[3]
    with pytest.raises(ValueError, match="outside"):
        _resolve_start_date(ds, "2020-01-03T20:00", shifted, True)
