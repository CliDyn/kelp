"""Batch read-ahead / prefetch (TODO 5.2).

prefetch_size > 1 bulk-loads a window of consecutive velocity snapshots in ONE
read; default 1 is per-timestep and byte-identical.  Reading from the loaded
window gives identical values, so trajectories match prefetch_size=1 bit-for-bit.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from kelp.tracking import _VelCache
from kelp.readers import get_reader
from kelp.core.config import (
    AppConfig, ModeConfig, NumericsConfig, MeshConfig, DataConfig, TimeConfig)


def _fesomc_env(tmp_path, n_time=10, u_val=1.0, v_val=0.5):
    from tests.test_tracking import _make_fesomc_3x3
    nc, _m, _mo = _make_fesomc_3x3(tmp_path, n_time=n_time, u_val=u_val, v_val=v_val)
    reader = get_reader("fesomc")
    mesh = reader.load_mesh(SimpleNamespace(meshpath=str(nc)))
    ds = reader.open_dataset(str(nc))
    mode = ModeConfig(Fdrift2D=True, Fdrift3D=False)
    return nc, reader, mesh, ds, mode


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
def test_config_default_and_validation():
    assert NumericsConfig().prefetch_size == 1
    for bad in (0, -1, 1.9, "4", True):
        with pytest.raises(ValueError, match="prefetch_size"):
            NumericsConfig(prefetch_size=bad)


def _make_timevarying_fesomc(tmp_path, n_time=12):
    """FESOM-C 3x3 NetCDF with per-snapshot-DISTINCT velocity (u2d[t] varies in
    time), so a wrong window/local-index mapping changes the result — the
    constant-field fixtures cannot catch an i-lo off-by-one.  Built on the
    *timed* helper (a datetime time coord) so temporal_interp / pchip actually
    activate when used in a run (without a time coord run_tracking falls back to
    the discrete path)."""
    import xarray as xr
    from tests.test_tracking import _make_fesomc_3x3_with_time
    nc = _make_fesomc_3x3_with_time(tmp_path, n_time=n_time, dt_seconds=3600.0,
                                    u_val=1.0, v_val=0.0)
    ds = xr.open_dataset(nc)
    ds.load()
    ne = ds.sizes["nele"]
    t = np.arange(n_time, dtype=np.float64)
    ds["u2d"] = (("time", "nele"), 0.1 * (t[:, None] + 1.0) * np.ones((n_time, ne)))
    ds["v2d"] = (("time", "nele"), -0.05 * (t[:, None] + 1.0) * np.ones((n_time, ne)))
    out = tmp_path / "timevarying.nc"
    ds.to_netcdf(out)
    ds.close()
    return out


# --------------------------------------------------------------------------
# windowing / batching (unit, on the cache directly)
# --------------------------------------------------------------------------
import math


@pytest.mark.parametrize("direction", [1, -1])
def test_prefetch_batches_window_loads(tmp_path, direction):
    # Mirror the PRODUCTION discrete loop (get(file_idx) AND get(next_idx) each
    # step), so the measured batch factor is the real ~pf-1 (the next_idx of a
    # window's last index is a fresh-window miss), not an optimistic pf.
    _nc, reader, mesh, ds, mode = _fesomc_env(tmp_path, n_time=12)
    pf, steps = 4, 8
    cache = _VelCache(prefetch_size=pf, direction=direction, n_time=12)
    start = 1 if direction > 0 else 10
    for k in range(steps):
        i = start + direction * k
        cache.get(i, reader, ds, mesh, mode)
        cache.get(i + direction, reader, ds, mesh, mode)   # next_idx
        cache.evict_except({i, i + direction})
    # Batching works (far fewer than the 2*steps gets) at ~steps/(pf-1).
    assert cache._n_window_loads < steps
    assert cache._n_window_loads <= math.ceil(steps / (pf - 1)) + 1
    ds.close()


def test_prefetch_values_match_single_timevarying(tmp_path):
    # With per-snapshot-DISTINCT velocity, the window-loaded field must equal the
    # per-timestep read at the SAME global index — pins the i-lo local mapping
    # (a constant-field fixture cannot catch an off-by-one here).
    nc = _make_timevarying_fesomc(tmp_path, n_time=8)
    reader = get_reader("fesomc")
    mesh = reader.load_mesh(SimpleNamespace(meshpath=str(nc)))
    ds = reader.open_dataset(str(nc))
    mode = ModeConfig(Fdrift2D=True, Fdrift3D=False)
    single = _VelCache(prefetch_size=1, n_time=8)
    batch = _VelCache(prefetch_size=4, direction=1, n_time=8)
    seen = []
    for i in range(6):
        a = single.get(i, reader, ds, mesh, mode)
        b = batch.get(i, reader, ds, mesh, mode)
        np.testing.assert_array_equal(a.u, b.u)
        np.testing.assert_array_equal(a.v, b.v)
        seen.append(float(a.u.flat[0]))
    assert len(set(seen)) == len(seen)   # values really do vary per index
    ds.close()


# --------------------------------------------------------------------------
# run-level bit-identical
# --------------------------------------------------------------------------
def _load(out):
    from kelp.writers import load_trajectories
    return load_trajectories(str(out))


def _assert_identical(a, b):
    for k in ("lon", "lat", "z"):
        np.testing.assert_array_equal(a[k], b[k], err_msg=f"{k} differs")
    np.testing.assert_array_equal(a["status"], b["status"])


def _run(cfg):
    from kelp.tracking import run_tracking
    run_tracking(cfg)


@pytest.mark.parametrize("temporal", [False, True])
@pytest.mark.parametrize("time_dir", [1.0, -1.0])
def test_run_bit_identical_prefetch_2d(tmp_path, temporal, time_dir):
    if temporal:
        from tests.test_tracking import _make_fesomc_3x3_with_time
        nc = _make_fesomc_3x3_with_time(tmp_path, n_time=12, dt_seconds=3600.0,
                                        u_val=1.0, v_val=0.5)
        dt = 1800.0
    else:
        from tests.test_tracking import _make_fesomc_3x3
        nc, _m, _mo = _make_fesomc_3x3(tmp_path, n_time=12, u_val=1.0, v_val=0.5)
        dt = 3600.0
    seeds = tmp_path / "s.dat"
    seeds.write_text("5.0 5.0\n4.7 5.1\n")
    start = 0 if time_dir > 0 else 10

    def cfg(out, pf):
        return AppConfig(
            mode=ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=time_dir),
            numerics=NumericsConfig(dt=dt, integrator="rk2", output_freq=1,
                                    temporal_interp=temporal, prefetch_size=pf),
            mesh=MeshConfig(type="fesomc", meshpath=str(nc)),
            data=DataConfig(ncinput=str(nc), inicloud=str(seeds),
                            output_path=str(out), time_averaging="instant"),
            time=TimeConfig(start_time_idx=start, n_steps=6))

    a = tmp_path / "a.nc"
    b = tmp_path / "b.nc"
    _run(cfg(a, 1))
    _run(cfg(b, 4))
    _assert_identical(_load(a), _load(b))


def test_run_bit_identical_prefetch_timevarying(tmp_path):
    # The non-vacuous headline: per-snapshot-distinct fields + temporal_interp,
    # so a wrong window/local-index mapping would change the trajectory.
    # prefetch=4 must match prefetch=1, and the run must actually move.
    nc = _make_timevarying_fesomc(tmp_path, n_time=12)
    seeds = tmp_path / "s.dat"
    seeds.write_text("5.0 5.0\n4.7 5.1\n")

    def cfg(out, pf):
        return AppConfig(
            mode=ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0),
            numerics=NumericsConfig(dt=1800.0, integrator="rk4", output_freq=1,
                                    temporal_interp=True, prefetch_size=pf),
            mesh=MeshConfig(type="fesomc", meshpath=str(nc)),
            data=DataConfig(ncinput=str(nc), inicloud=str(seeds),
                            output_path=str(out), time_averaging="instant"),
            time=TimeConfig(start_time_idx=0, n_steps=10))

    a = tmp_path / "a.nc"
    b = tmp_path / "b.nc"
    _run(cfg(a, 1))
    _run(cfg(b, 4))
    ta = _load(a)
    _assert_identical(ta, _load(b))
    assert np.nanmax(np.abs(ta["lon"][:, -1] - 5.0)) > 1e-6   # actually moved


def test_run_bit_identical_prefetch_pchip(tmp_path):
    # prefetch + temporal_reconstruction=pchip: the 4-snapshot stencil must
    # survive eviction (keep_extra ∪ window) and match prefetch=1.
    nc = _make_timevarying_fesomc(tmp_path, n_time=12)
    seeds = tmp_path / "s.dat"
    seeds.write_text("5.0 5.0\n")

    def cfg(out, pf):
        return AppConfig(
            mode=ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0),
            numerics=NumericsConfig(dt=1200.0, integrator="rk4", output_freq=1,
                                    temporal_interp=True,
                                    temporal_reconstruction="pchip",
                                    prefetch_size=pf),
            mesh=MeshConfig(type="fesomc", meshpath=str(nc)),
            data=DataConfig(ncinput=str(nc), inicloud=str(seeds),
                            output_path=str(out), time_averaging="instant"),
            time=TimeConfig(start_time_idx=2, n_steps=9))

    a = tmp_path / "a.nc"
    b = tmp_path / "b.nc"
    _run(cfg(a, 1))
    _run(cfg(b, 6))
    _assert_identical(_load(a), _load(b))


def test_run_bit_identical_prefetch_cycle(tmp_path):
    # on_exhausted=cycle auto-disables prefetch (avoids a seam I/O regression);
    # the run must still be correct and match the per-timestep run.
    from tests.test_tracking import _make_fesomc_3x3
    nc, _m, _mo = _make_fesomc_3x3(tmp_path, n_time=4, u_val=1.0, v_val=0.25)
    seeds = tmp_path / "s.dat"
    seeds.write_text("5.0 5.0\n")

    def cfg(out, pf):
        return AppConfig(
            mode=ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0),
            numerics=NumericsConfig(dt=3600.0, integrator="euler", output_freq=1,
                                    prefetch_size=pf),
            mesh=MeshConfig(type="fesomc", meshpath=str(nc)),
            data=DataConfig(ncinput=str(nc), inicloud=str(seeds),
                            output_path=str(out), time_averaging="instant"),
            time=TimeConfig(start_time_idx=0, n_steps=10, on_exhausted="cycle"))

    a = tmp_path / "a.nc"
    b = tmp_path / "b.nc"
    _run(cfg(a, 1))
    _run(cfg(b, 3))
    _assert_identical(_load(a), _load(b))


def test_run_bit_identical_prefetch_3d(tmp_path):
    from tests.test_tracking import _make_fesom2_3x3_3d
    from kelp.tracking import run_single_realization
    from kelp.core.particles import make_state
    from kelp.readers import extract_time_info

    mesh_dir, vel_path, mesh_cfg = _make_fesom2_3x3_3d(
        tmp_path, n_time=6, u_val=0.5, v_val=0.25, w_val=0.0625)
    reader = get_reader("fesom2")
    mesh = reader.load_mesh(mesh_cfg)
    ds = reader.open_dataset(str(vel_path))
    file_times, file_dt = extract_time_info(ds)
    if file_times is None:
        # index-only fixture axis: declare the interval, or the temporal
        # path is refused (it used to run discrete silently — F03)
        file_dt = 100.0
        file_times = np.arange(ds.sizes["time"], dtype=np.float64) * file_dt
    mode = ModeConfig(Fdrift2D=False, Fdrift3D=True, time_dir=1.0)
    init = make_state(np.array([5.0]), np.array([5.0]), np.array([20.0]))

    def run(pf):
        num = NumericsConfig(dt=(file_dt / 2.0 if file_dt else 50.0),
                             integrator="rk2", prefetch_size=pf)
        return run_single_realization(
            mesh, ds, reader, init, mode, num, 0, n_steps=4, n_time_total=6,
            temporal_interp=True, file_times=file_times, file_dt=file_dt,
        ).to_arrays()

    a, b = run(1), run(4)
    for k in ("lon", "lat", "z"):
        np.testing.assert_array_equal(a[k], b[k], err_msg=f"{k} differs")
