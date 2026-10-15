"""float32 velocity fields (TODO 5.3).

Keeping u/v/w in float32 (native FESOM2 dtype) halves snapshot memory/IO.
Opt-in via numerics.velocity_float32; default float64 (byte-identical).

Because all per-particle arithmetic runs in float64 accumulators, computing on
a float32 field equals computing on its (exact) float64 upcast — so on a
non-rotated mesh the float32 run is bit-identical to the float64 run whenever the
stored values are exactly float32-representable (always true for real FESOM2,
whose fields are float32 on disk).  The fixtures here use float32-exact values
(1.0, 0.5, 0.0625, …) so the bit-identical assertion holds against the float64
synthetic NetCDF.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from kelp.core.io import _blend
from kelp.readers import get_reader
from kelp.core.config import (
    AppConfig, ModeConfig, NumericsConfig, MeshConfig, DataConfig, TimeConfig)


def test_config_default_false():
    assert NumericsConfig().velocity_float32 is False


def test_reader_float32_dtype_and_values(tmp_path):
    from tests.test_tracking import _make_fesomc_3x3
    nc, _m, _mo = _make_fesomc_3x3(tmp_path, n_time=3, u_val=0.5, v_val=-0.25)
    reader = get_reader("fesomc")
    mesh = reader.load_mesh(SimpleNamespace(meshpath=str(nc)))
    ds = reader.open_dataset(str(nc))
    mode = ModeConfig(Fdrift2D=True, Fdrift3D=False)
    v64 = reader.read_velocities(ds, 0, mesh, mode)
    v32 = reader.read_velocities(ds, 0, mesh, mode, velocity_float32=True)
    assert v64.u.dtype == np.float64 and v64.v.dtype == np.float64
    assert v32.u.dtype == np.float32 and v32.v.dtype == np.float32
    # float32 value == float64 value (values are float32-exact here)
    np.testing.assert_array_equal(v32.u.astype(np.float64), v64.u)
    np.testing.assert_array_equal(v32.v.astype(np.float64), v64.v)


def test_blend_float32_promotes_to_float64_and_matches():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(5000).astype(np.float32)
    b = rng.standard_normal(5000).astype(np.float32)
    out32 = _blend(a, b, 0.3)
    out64 = _blend(a.astype(np.float64), b.astype(np.float64), 0.3)
    assert out32.dtype == np.float64          # numpy promotion on the fallback
    np.testing.assert_array_equal(out32, out64)


def _load(out):
    from kelp.writers import load_trajectories
    return load_trajectories(str(out))


def _assert_traj_identical(a, b):
    for k in ("lon", "lat", "z"):
        np.testing.assert_array_equal(a[k], b[k], err_msg=f"{k} differs")
    np.testing.assert_array_equal(a["status"], b["status"])


def _run(cfg):
    from kelp.tracking import run_tracking
    run_tracking(cfg)


@pytest.mark.parametrize("temporal", [False, True])
def test_run_bit_identical_2d(tmp_path, temporal):
    # FESOM-C 2D, discrete and temporal-interp: float32 == float64 (bit-for-bit).
    if temporal:
        from tests.test_tracking import _make_fesomc_3x3_with_time
        nc = _make_fesomc_3x3_with_time(tmp_path, n_time=8, dt_seconds=3600.0,
                                        u_val=1.0, v_val=0.5)
        dt = 1800.0
    else:
        from tests.test_tracking import _make_fesomc_3x3
        nc, _m, _mo = _make_fesomc_3x3(tmp_path, n_time=8, u_val=1.0, v_val=0.5)
        dt = 3600.0
    seeds = tmp_path / "s.dat"
    seeds.write_text("5.0 5.0\n4.6 5.2\n")

    def cfg(out, f32):
        return AppConfig(
            mode=ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0),
            numerics=NumericsConfig(dt=dt, integrator="rk2", output_freq=1,
                                    temporal_interp=temporal,
                                    velocity_float32=f32),
            mesh=MeshConfig(type="fesomc", meshpath=str(nc)),
            data=DataConfig(ncinput=str(nc), inicloud=str(seeds),
                            output_path=str(out), time_averaging="instant"),
            time=TimeConfig(start_time_idx=0, n_steps=6))

    a = tmp_path / "a.nc"
    b = tmp_path / "b.nc"
    _run(cfg(a, False))
    _run(cfg(b, True))
    _assert_traj_identical(_load(a), _load(b))


def test_run_bit_identical_2d_temporal_unfused(tmp_path):
    # temporal-interp + barycentric forces the UNFUSED path, which materializes
    # each stage via _make_stage_vels -> lerp -> _blend.  This is the case the
    # NEP-50 float32 _blend rounding would break; pins the float64-blend fix.
    from tests.test_tracking import _make_fesomc_3x3_with_time
    nc = _make_fesomc_3x3_with_time(tmp_path, n_time=8, dt_seconds=3600.0,
                                    u_val=1.0, v_val=0.5)
    seeds = tmp_path / "s.dat"
    seeds.write_text("5.0 5.0\n")

    def cfg(out, f32):
        return AppConfig(
            mode=ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0),
            numerics=NumericsConfig(dt=1800.0, integrator="rk2", output_freq=1,
                                    temporal_interp=True, uv_interp="barycentric",
                                    velocity_float32=f32),
            mesh=MeshConfig(type="fesomc", meshpath=str(nc)),
            data=DataConfig(ncinput=str(nc), inicloud=str(seeds),
                            output_path=str(out), time_averaging="instant"),
            time=TimeConfig(start_time_idx=0, n_steps=6))

    a = tmp_path / "a.nc"
    b = tmp_path / "b.nc"
    _run(cfg(a, False))
    _run(cfg(b, True))
    _assert_traj_identical(_load(a), _load(b))


def test_run_bit_identical_3d(tmp_path):
    # FESOM2 3D (exercises w): float32 == float64 via run_single_realization,
    # comparing the in-memory records.  float32-exact velocities (w = 1/16).
    from tests.test_tracking import _make_fesom2_3x3_3d
    from kelp.tracking import run_single_realization
    from kelp.core.particles import make_state

    mesh_dir, vel_path, mesh_cfg = _make_fesom2_3x3_3d(
        tmp_path, n_time=4, u_val=0.5, v_val=0.25, w_val=0.0625)
    reader = get_reader("fesom2")
    mesh = reader.load_mesh(mesh_cfg)
    ds = reader.open_dataset(str(vel_path))
    mode = ModeConfig(Fdrift2D=False, Fdrift3D=True, time_dir=1.0)
    init = make_state(np.array([5.0]), np.array([5.0]), np.array([20.0]))

    def run(f32):
        num = NumericsConfig(dt=100.0, integrator="rk2", velocity_float32=f32)
        return run_single_realization(
            mesh, ds, reader, init, mode, num, 0, n_steps=3, n_time_total=4
        ).to_arrays()

    a, b = run(False), run(True)
    for k in ("lon", "lat", "z"):
        np.testing.assert_array_equal(a[k], b[k], err_msg=f"{k} differs")


@pytest.mark.parametrize("uv_interp", ["leastsq", "barycentric"])
def test_run_bit_identical_pchip(tmp_path, uv_interp):
    # float32 + temporal_interp + PCHIP: the reconstruction must compute in
    # float64 (leastsq -> fused _pchip_value_scalar; barycentric -> unfused
    # reconstruct_pchip/_pchip_blend4).  Both rounded in float32 before the fix.
    from tests.test_tracking import _make_fesomc_3x3_with_time
    nc = _make_fesomc_3x3_with_time(tmp_path, n_time=10, dt_seconds=3600.0,
                                    u_val=1.0, v_val=0.5)
    seeds = tmp_path / "s.dat"
    seeds.write_text("5.0 5.0\n")

    def cfg(out, f32):
        return AppConfig(
            mode=ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0),
            numerics=NumericsConfig(dt=1200.0, integrator="rk4", output_freq=1,
                                    temporal_interp=True,
                                    temporal_reconstruction="pchip",
                                    uv_interp=uv_interp, velocity_float32=f32),
            mesh=MeshConfig(type="fesomc", meshpath=str(nc)),
            data=DataConfig(ncinput=str(nc), inicloud=str(seeds),
                            output_path=str(out), time_averaging="instant"),
            time=TimeConfig(start_time_idx=2, n_steps=9))

    a = tmp_path / "a.nc"
    b = tmp_path / "b.nc"
    _run(cfg(a, False))
    _run(cfg(b, True))
    _assert_traj_identical(_load(a), _load(b))


def test_run_bit_identical_float32_on_disk(tmp_path):
    # The headline case: velocity stored as float32 ON DISK (real FESOM2) with
    # NON-round values.  Both modes read the same float32 value (float64-mode
    # upcasts losslessly) and compute in float64 -> bit-identical regardless of
    # value.  Pins the feature's reason to exist (not just float32-exact values).
    import xarray as xr
    from tests.test_tracking import _make_fesomc_3x3
    nc, _m, _mo = _make_fesomc_3x3(tmp_path, n_time=8, u_val=0.37, v_val=-0.21)
    ds = xr.open_dataset(nc)
    ds.load()
    ds["u2d"] = ds["u2d"].astype(np.float32)
    ds["v2d"] = ds["v2d"].astype(np.float32)
    nc32 = tmp_path / "f32disk.nc"
    ds.to_netcdf(nc32)
    ds.close()
    seeds = tmp_path / "s.dat"
    seeds.write_text("5.0 5.0\n4.7 5.1\n")

    def cfg(out, f32):
        return AppConfig(
            mode=ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0),
            numerics=NumericsConfig(dt=3600.0, integrator="rk2", output_freq=1,
                                    velocity_float32=f32),
            mesh=MeshConfig(type="fesomc", meshpath=str(nc32)),
            data=DataConfig(ncinput=str(nc32), inicloud=str(seeds),
                            output_path=str(out), time_averaging="instant"),
            time=TimeConfig(start_time_idx=0, n_steps=6))

    a = tmp_path / "a.nc"
    b = tmp_path / "b.nc"
    _run(cfg(a, False))
    _run(cfg(b, True))
    _assert_traj_identical(_load(a), _load(b))


def test_run_bit_identical_3d_temporal(tmp_path):
    # 3D + temporal-interp exercises the unfused stage materialization
    # (_make_stage_vels -> lerp -> _blend) for u/v AND w; pins the _blend fix
    # in 3D.  float32-exact velocities (w = 1/16).
    from tests.test_tracking import _make_fesom2_3x3_3d
    from kelp.tracking import run_single_realization
    from kelp.core.particles import make_state
    from kelp.readers import extract_time_info

    mesh_dir, vel_path, mesh_cfg = _make_fesom2_3x3_3d(
        tmp_path, n_time=5, u_val=0.5, v_val=0.25, w_val=0.0625)
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

    def run(f32):
        num = NumericsConfig(dt=(file_dt / 2.0 if file_dt else 50.0),
                             integrator="rk2", velocity_float32=f32)
        return run_single_realization(
            mesh, ds, reader, init, mode, num, 0, n_steps=4, n_time_total=5,
            temporal_interp=True, file_times=file_times, file_dt=file_dt,
        ).to_arrays()

    a, b = run(False), run(True)
    for k in ("lon", "lat", "z"):
        np.testing.assert_array_equal(a[k], b[k], err_msg=f"{k} differs")


def test_run_default_is_float64(tmp_path):
    # A default run keeps cached velocity float64 and writes no metadata key.
    from tests.test_tracking import _make_fesomc_3x3
    from kelp.readers import get_reader as gr
    nc, _m, _mo = _make_fesomc_3x3(tmp_path, n_time=3, u_val=1.0, v_val=0.0)
    reader = gr("fesomc")
    mesh = reader.load_mesh(SimpleNamespace(meshpath=str(nc)))
    ds = reader.open_dataset(str(nc))
    v = reader.read_velocities(ds, 0, mesh, ModeConfig(Fdrift2D=True))
    assert v.u.dtype == np.float64
