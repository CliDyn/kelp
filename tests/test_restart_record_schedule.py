"""Record and checkpoint cadence across restarts (review 2026-09-22, F04).

``run_single_realization`` counted records with a per-call counter, so every
resumed chunk restarted the ``output_freq`` phase at zero.  A chain of
91-step chunks with ``output_freq=5`` (the Beaufort Gyre run) therefore wrote
a 6-day gap at every chunk boundary instead of the 5-day cadence it was
configured for, and an analysis that assumed a uniform cadence picked the
"+10 years" record about 40 days late.  The cadence must follow the GLOBAL
step index, so a chunked run records exactly what the uninterrupted run does.
"""
import numpy as np

from kelp.core.particles import make_state


def _fesomc_env(tmp_path, n_time):
    from types import SimpleNamespace
    from kelp.readers import get_reader
    from kelp.core.config import ModeConfig
    from tests.test_tracking import _make_fesomc_3x3_with_time
    # a real (datetime64, hourly) time axis: with an index-only axis the
    # temporal-interpolation leg would silently run discrete
    nc_path = _make_fesomc_3x3_with_time(tmp_path, n_time=n_time,
                                         dt_seconds=3600.0, u_val=1.0,
                                         v_val=0.0)
    reader = get_reader("fesomc")
    mesh = reader.load_mesh(SimpleNamespace(meshpath=str(nc_path)))
    ds = reader.open_dataset(str(nc_path))
    mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)
    return mesh, ds, reader, mode


def _run_chunks(tmp_path, tag, chunks, output_freq, n_time, dt=3600.0,
                temporal_interp=False):
    """Run ``sum(chunks)`` steps split into restart chunks; return the
    concatenated record times, lons and the per-chunk record counts."""
    from kelp.core.config import NumericsConfig
    from kelp.readers import extract_time_info
    from kelp.tracking import run_single_realization
    from kelp.writers import TrajectoryWriter, load_restart, load_trajectories
    (tmp_path / tag).mkdir()
    mesh, ds, reader, mode = _fesomc_env(tmp_path / tag, n_time)
    num = NumericsConfig(dt=dt, integrator="euler", output_freq=output_freq)
    file_times, file_dt = extract_time_info(ds)
    state = make_state(np.array([5.0, 5.1]), np.array([5.0, 5.0]))
    times, lons, counts = [], [], []
    step0 = 0
    for c, n in enumerate(chunks):
        p = str(tmp_path / f"{tag}_{c}.nc")
        w = TrajectoryWriter(p, state.n_particles)
        w.open()
        kw = {}
        if temporal_interp:
            kw = dict(temporal_interp=True, file_times=file_times,
                      file_dt=file_dt, sim_time_start=step0 * dt)
        run_single_realization(mesh, ds, reader, state, mode, num,
                               start_time_idx=step0, n_steps=n,
                               n_time_total=n_time, writer=w,
                               checkpoint_freq=n, start_step_i=step0, **kw)
        w.close()
        tr = load_trajectories(p)
        counts.append(len(tr["time"]))
        times.extend(tr["time"].tolist())
        lons.append(tr["lon"])
        r = load_restart(p)
        state = r["state"]
        step0 = r["step_i"] + 1
    lons = np.concatenate(lons, axis=1) if lons else np.empty((2, 0))
    return np.asarray(times), lons, counts


def test_discrete_chunks_record_like_the_uninterrupted_run(tmp_path):
    t_a, lon_a, _ = _run_chunks(tmp_path, "a", [6], output_freq=3, n_time=12)
    t_b, lon_b, counts = _run_chunks(tmp_path, "b", [2, 4], output_freq=3,
                                     n_time=12)
    assert len(t_a) == 2
    np.testing.assert_array_equal(t_b, t_a)
    np.testing.assert_array_equal(lon_b, lon_a)
    assert counts == [0, 2]


def test_uneven_chunks_keep_a_uniform_cadence(tmp_path):
    # the Beaufort shape in miniature: chunk length not a multiple of the
    # record interval
    t_a, lon_a, _ = _run_chunks(tmp_path, "a", [14], output_freq=5,
                                n_time=20, temporal_interp=True)
    t_b, lon_b, _ = _run_chunks(tmp_path, "b", [4, 4, 4, 2], output_freq=5,
                                n_time=20, temporal_interp=True)
    np.testing.assert_allclose(t_b, t_a)
    np.testing.assert_array_equal(lon_b, lon_a)
    assert np.allclose(np.diff(t_b), 5 * 3600.0)
