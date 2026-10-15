"""Delayed / staggered particle release (TODO 2.5).

Pre-allocate all particles and FREEZE (status NOT_RELEASED) the not-yet-released
ones until their release step.
"""
import numpy as np
import pytest

from kelp.core.release import (
    build_release_state, activate_released, mask_dormant_for_record,
    read_seeds_with_release,
)
from kelp.core.particles import ParticleStatus, make_state
from kelp.core.io import VelocityField
from kelp.numerics.integrators import euler_step
from tests.test_integrators_time_aware import _make_mesh, LON0, LAT0


# --------------------------------------------------------------------------
# build_release_state
# --------------------------------------------------------------------------
def test_build_trivial_is_plain_state():
    # n_pulses=1 + no release column -> byte-identical to a normal seed load.
    s = build_release_state(np.array([1.0, 2.0]), np.array([0.0, 0.0]), dt=3600.0)
    assert s.release_step is None
    assert (s.status == ParticleStatus.ALIVE).all()


def test_build_pulses():
    s = build_release_state(np.array([1.0, 2.0]), np.array([0.0, 0.0]),
                            n_pulses=3, pulse_interval=7200.0, dt=3600.0)
    assert s.n_particles == 6                       # 3 pulses x 2 seeds
    assert list(s.release_step) == [0, 0, 2, 2, 4, 4]
    assert int((s.status == ParticleStatus.ALIVE).sum()) == 2          # pulse 0
    assert int((s.status == ParticleStatus.NOT_RELEASED).sum()) == 4
    # pulses replicate the SAME coordinates
    np.testing.assert_array_equal(s.lon, np.tile([1.0, 2.0], 3))


def test_build_per_particle_column_plus_pulses_compose():
    # base release [0, 3600] sec; 2 pulses, interval 7200; dt 3600 ->
    # pulse 0 steps [0, 1], pulse 1 steps [2, 3].
    s = build_release_state(np.array([1.0, 2.0]), np.array([0.0, 0.0]),
                            release_sec=np.array([0.0, 3600.0]),
                            n_pulses=2, pulse_interval=7200.0, dt=3600.0)
    assert list(s.release_step) == [0, 1, 2, 3]


def test_build_per_particle_only():
    s = build_release_state(np.array([1.0, 2.0]), np.array([0.0, 0.0]),
                            release_sec=np.array([0.0, 7200.0]), dt=3600.0)
    assert list(s.release_step) == [0, 2]
    assert s.status[0] == ParticleStatus.ALIVE
    assert s.status[1] == ParticleStatus.NOT_RELEASED


def test_build_validation():
    with pytest.raises(ValueError, match=">= 0"):
        build_release_state([1.0], [0.0], release_sec=[-1.0], dt=3600.0)
    with pytest.raises(ValueError, match="n_pulses"):
        build_release_state([1.0], [0.0], n_pulses=0, dt=3600.0)
    with pytest.raises(ValueError, match="pulse_interval"):
        build_release_state([1.0], [0.0], n_pulses=2, pulse_interval=0.0, dt=3600.0)
    with pytest.raises(ValueError, match="positive dt"):
        build_release_state([1.0], [0.0], release_sec=[3600.0], dt=0.0)


def test_build_non_multiple_warns_and_rounds():
    with pytest.warns(RuntimeWarning, match="whole multiple"):
        s = build_release_state([1.0], [0.0], release_sec=[5400.0], dt=3600.0)  # 1.5 step
    assert list(s.release_step) == [2]                # rounds to nearest


# --------------------------------------------------------------------------
# activate_released
# --------------------------------------------------------------------------
def test_activate_none_release_step_is_same_object():
    s = make_state(np.array([1.0]), np.array([0.0]))
    assert activate_released(s, 5) is s


def test_activate_flips_only_at_due_step():
    s = build_release_state([1.0, 2.0], [0.0, 0.0],
                            n_pulses=2, pulse_interval=7200.0, dt=3600.0)  # steps 0,0,2,2
    assert activate_released(s, 1) is s               # nothing new due at step 1
    s2 = activate_released(s, 2)
    assert int((s2.status == ParticleStatus.ALIVE).sum()) == 4
    # original untouched (immutable)
    assert int((s.status == ParticleStatus.ALIVE).sum()) == 2


# --------------------------------------------------------------------------
# mask_dormant_for_record
# --------------------------------------------------------------------------
def test_mask_no_dormant_is_same_object():
    s = make_state(np.array([1.0]), np.array([0.0]))
    rec, samp = mask_dormant_for_record(s, None)
    assert rec is s and samp is None


def test_mask_nans_dormant_positions_and_sampled():
    s = build_release_state([1.0, 2.0], [0.0, 0.0],
                            n_pulses=2, pulse_interval=7200.0, dt=3600.0)
    sampled = {"temp": np.array([10.0, 20.0, 30.0, 40.0])}
    rec, samp = mask_dormant_for_record(s, sampled)
    dormant = s.status == ParticleStatus.NOT_RELEASED
    assert np.isnan(rec.lon[dormant]).all() and np.isnan(rec.lat[dormant]).all()
    assert not np.isnan(rec.lon[~dormant]).any()
    assert np.isnan(samp["temp"][dormant]).all()
    assert not np.isnan(samp["temp"][~dormant]).any()
    # status still records NOT_RELEASED; original coords intact.
    np.testing.assert_array_equal(rec.status, s.status)
    assert not np.isnan(s.lon).any()


# --------------------------------------------------------------------------
# read_seeds_with_release
# --------------------------------------------------------------------------
def test_read_seeds_3col_and_4col(tmp_path):
    p3 = tmp_path / "s3.dat"
    p3.write_text("5 5\n6 6\n")
    lon, lat, z, rel = read_seeds_with_release(str(p3))
    assert list(lon) == [5, 6] and list(lat) == [5, 6]
    assert (z == 0).all() and (rel == 0).all()

    p4 = tmp_path / "s4.dat"
    p4.write_text("5 5 0 0\n6 6 0 7200\n")
    lon, lat, z, rel = read_seeds_with_release(str(p4))
    assert list(rel) == [0, 7200]


# --------------------------------------------------------------------------
# Mechanism equivalence: a delayed particle, after release, follows the SAME
# path as a normal particle (constant-in-time velocity), forward AND backward.
# --------------------------------------------------------------------------
def _const_vel(mesh, u=0.5, v=0.1):
    return VelocityField(u=np.full(mesh.n_elements, u),
                         v=np.full(mesh.n_elements, v),
                         eta=np.zeros(mesh.n_nodes), w=None)


@pytest.mark.parametrize("time_dir", [1.0, -1.0])
def test_delayed_matches_normal_constant_velocity(time_dir):
    mesh = _make_mesh()
    vel = _const_vel(mesh)
    dt, K, M = 1800.0, 3, 4

    # Reference: a normal particle advected M steps.
    sn = make_state(np.array([LON0]), np.array([LAT0]))
    for _ in range(M):
        sn = euler_step(sn, vel, mesh, dt, time_dir)

    # Delayed: release at step K, run K+M steps (frozen K, then M moves).
    sd = build_release_state(np.array([LON0]), np.array([LAT0]),
                             release_sec=np.array([K * dt]), dt=dt)
    assert int(sd.release_step[0]) == K
    for step_i in range(K + M):
        sd = activate_released(sd, step_i)        # pre-step hook (mirrors loop)
        if step_i < K:
            assert sd.status[0] == ParticleStatus.NOT_RELEASED
            assert sd.lon[0] == LON0 and sd.lat[0] == LAT0   # frozen at seed
        sd = euler_step(sd, vel, mesh, dt, time_dir)

    assert sd.status[0] == ParticleStatus.ALIVE
    np.testing.assert_allclose(sd.lon[0], sn.lon[0], atol=1e-9)
    np.testing.assert_allclose(sd.lat[0], sn.lat[0], atol=1e-9)


# --------------------------------------------------------------------------
# Restart: checkpoint round-trip preserves release_step + NOT_RELEASED status.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ext", [".nc", ".zarr"])
def test_checkpoint_roundtrip_release_step(tmp_path, ext):
    from kelp.writers import TrajectoryWriter, load_restart
    s = build_release_state([1.0, 2.0], [0.0, 0.0],
                            n_pulses=2, pulse_interval=7200.0, dt=3600.0)
    path = str(tmp_path / f"traj{ext}")
    w = TrajectoryWriter(path, s.n_particles, {})
    w.open()
    w.append(s, 0.0)
    w.write_checkpoint(s, 3, 3, release_step=s.release_step)
    w.close()

    r = load_restart(path)
    assert r["release_step"] is not None
    np.testing.assert_array_equal(r["release_step"], s.release_step)
    np.testing.assert_array_equal(r["state"].release_step, s.release_step)
    np.testing.assert_array_equal(r["state"].status, s.status)


def test_checkpoint_no_release_step_absent(tmp_path):
    # A normal run writes no restart_release_step; load_restart returns None.
    from kelp.writers import TrajectoryWriter, load_restart
    s = make_state(np.array([1.0]), np.array([0.0]))
    path = str(tmp_path / "plain.nc")
    w = TrajectoryWriter(path, s.n_particles, {})
    w.open()
    w.append(s, 0.0)
    w.write_checkpoint(s, 0, 0)
    w.close()
    assert load_restart(path)["release_step"] is None


# --------------------------------------------------------------------------
# End-to-end via run_tracking: pulses -> NaN-before-release + path equivalence.
# --------------------------------------------------------------------------
def test_end_to_end_pulses_nan_and_equivalence(tmp_path):
    from kelp.core.config import (
        AppConfig, ModeConfig, NumericsConfig, MeshConfig, DataConfig, TimeConfig)
    from kelp.tracking import run_tracking
    from kelp.writers import load_trajectories
    from tests.test_tracking import _make_fesomc_3x3

    nc_path, _mesh_cfg, _mode_cfg = _make_fesomc_3x3(
        tmp_path, n_time=8, u_val=1.0, v_val=0.0)
    seeds = tmp_path / "seeds.dat"
    seeds.write_text("5.0 5.0\n")
    out = tmp_path / "traj.nc"
    dt, K, M = 3600.0, 2, 3

    cfg = AppConfig(
        mode=ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0),
        numerics=NumericsConfig(dt=dt, integrator="euler", output_freq=1),
        mesh=MeshConfig(type="fesomc", meshpath=str(nc_path)),
        data=DataConfig(ncinput=str(nc_path), inicloud=str(seeds),
                        output_path=str(out), time_averaging="instant"),
        time=TimeConfig(start_time_idx=0, n_steps=K + M,
                        n_pulses=2, pulse_interval=K * dt),
    )
    run_tracking(cfg)

    tr = load_trajectories(str(out))
    lon, lat, status = tr["lon"], tr["lat"], tr["status"]
    assert lon.shape[0] == 2                          # 2 pulses x 1 seed

    # pulse 0: real positions throughout, ALIVE.
    assert not np.isnan(lon[0]).any()
    assert (status[0] == ParticleStatus.ALIVE).all()

    # pulse 1: frozen (NaN + NOT_RELEASED) for the first K obs, real after.
    assert np.isnan(lon[1, :K]).all()
    assert (status[1, :K] == ParticleStatus.NOT_RELEASED).all()
    assert not np.isnan(lon[1, K:]).any()
    assert (status[1, K:] == ParticleStatus.ALIVE).all()

    # Equivalence (constant velocity): pulse 1's post-release path == pulse 0's
    # first M moves, element-for-element.
    np.testing.assert_allclose(lon[1, K:], lon[0, :M], atol=1e-6)
    np.testing.assert_allclose(lat[1, K:], lat[0, :M], atol=1e-6)


def test_build_rejects_non_finite():
    with pytest.raises(ValueError, match="finite"):
        build_release_state([1.0, 2.0], [0.0, 0.0],
                            release_sec=[0.0, np.nan], dt=3600.0)
    with pytest.raises(ValueError, match="finite"):
        build_release_state([1.0, 2.0], [0.0, 0.0],
                            release_sec=[0.0, np.inf], dt=3600.0)
    with pytest.raises(ValueError, match="finite"):
        build_release_state([1.0], [0.0], n_pulses=2,
                            pulse_interval=np.inf, dt=3600.0)


# --------------------------------------------------------------------------
# Run-level restart + early-break checkpoint (validates the BLOCKER fixes).
# --------------------------------------------------------------------------
def _fesomc_env(tmp_path, n_time=10, u_val=1.0):
    from types import SimpleNamespace
    from kelp.readers import get_reader
    from kelp.core.config import ModeConfig
    from tests.test_tracking import _make_fesomc_3x3
    nc_path, _m, _mo = _make_fesomc_3x3(tmp_path, n_time=n_time, u_val=u_val, v_val=0.0)
    reader = get_reader("fesomc")
    mesh = reader.load_mesh(SimpleNamespace(meshpath=str(nc_path)))
    ds = reader.open_dataset(str(nc_path))
    mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)
    return mesh, ds, reader, mode


def test_run_level_restart_resumes_dormant(tmp_path):
    # Interrupt a run mid-dormancy, resume from the checkpoint, and require the
    # resumed run to match the uninterrupted one — i.e. release_step survives
    # the checkpoint and the dormant particle still activates at its step.
    from kelp.tracking import run_single_realization
    from kelp.writers import TrajectoryWriter, load_restart, load_trajectories
    from kelp.core.config import NumericsConfig
    mesh, ds, reader, mode = _fesomc_env(tmp_path, n_time=10)
    dt, K, N, J = 3600.0, 4, 8, 2          # release@4; interrupt@2 (still frozen)
    num = NumericsConfig(dt=dt, integrator="euler")
    init = build_release_state(np.array([5.0]), np.array([5.0]),
                               release_sec=np.array([K * dt]), dt=dt)

    pa = str(tmp_path / "a.nc")
    wa = TrajectoryWriter(pa, init.n_particles); wa.open()
    run_single_realization(mesh, ds, reader, init, mode, num, 0, n_steps=N,
                           n_time_total=10, writer=wa)
    wa.close()
    final_a = load_trajectories(pa)["lon"][0, -1]

    pb = str(tmp_path / "b.nc")
    wb = TrajectoryWriter(pb, init.n_particles); wb.open()
    run_single_realization(mesh, ds, reader, init, mode, num, 0, n_steps=J,
                           n_time_total=10, writer=wb, checkpoint_freq=1)
    wb.close()
    r = load_restart(pb)
    assert r["release_step"] is not None
    assert r["state"].release_step is not None      # BLOCKER 1: schedule survived

    pc = str(tmp_path / "c.nc")
    wc = TrajectoryWriter(pc, init.n_particles); wc.open()
    run_single_realization(mesh, ds, reader, r["state"], mode, num,
                           start_time_idx=J, n_steps=N - J, n_time_total=10,
                           writer=wc, start_step_i=r["step_i"] + 1)
    wc.close()
    final_c = load_trajectories(pc)["lon"][0, -1]
    assert not np.isnan(final_c)                     # actually activated & moved
    np.testing.assert_allclose(final_c, final_a, atol=1e-9)


def test_early_break_checkpoint_stamps_last_executed_step(tmp_path):
    # n_steps far beyond the available snapshots -> the loop breaks early
    # (on_exhausted='stop').  The final checkpoint must stamp the LAST EXECUTED
    # step, not start_step_i+n_steps-1 (which would make a resume skip steps).
    from kelp.tracking import run_single_realization
    from kelp.writers import TrajectoryWriter, load_restart
    from kelp.core.config import NumericsConfig
    mesh, ds, reader, mode = _fesomc_env(tmp_path, n_time=4)
    num = NumericsConfig(dt=3600.0, integrator="euler")
    init = make_state(np.array([5.0]), np.array([5.0]))
    p = str(tmp_path / "brk.nc")
    w = TrajectoryWriter(p, init.n_particles); w.open()
    run_single_realization(mesh, ds, reader, init, mode, num, 0, n_steps=20,
                           n_time_total=4, on_exhausted="stop",
                           writer=w, checkpoint_freq=1)
    w.close()
    r = load_restart(p)
    assert r["step_i"] == 3            # last executable file_idx (0..3), NOT 19


def test_release_beyond_nsteps_warns_and_all_nan(tmp_path):
    from kelp.core.config import (
        AppConfig, ModeConfig, NumericsConfig, MeshConfig, DataConfig, TimeConfig)
    from kelp.tracking import run_tracking
    from kelp.writers import load_trajectories
    from tests.test_tracking import _make_fesomc_3x3
    nc_path, _m, _mo = _make_fesomc_3x3(tmp_path, n_time=8, u_val=1.0, v_val=0.0)
    seeds = tmp_path / "s.dat"
    seeds.write_text("5.0 5.0\n")
    out = tmp_path / "t.nc"
    dt = 3600.0
    cfg = AppConfig(
        mode=ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0),
        numerics=NumericsConfig(dt=dt, integrator="euler", output_freq=1),
        mesh=MeshConfig(type="fesomc", meshpath=str(nc_path)),
        data=DataConfig(ncinput=str(nc_path), inicloud=str(seeds),
                        output_path=str(out), time_averaging="instant"),
        # pulse 1 releases at step 10 > n_steps 3 -> never activates.
        time=TimeConfig(start_time_idx=0, n_steps=3, n_pulses=2,
                        pulse_interval=10 * dt),
    )
    with pytest.warns(RuntimeWarning, match="never activate"):
        run_tracking(cfg)
    tr = load_trajectories(str(out))
    assert np.isnan(tr["lon"][1]).all()
    assert (tr["status"][1] == ParticleStatus.NOT_RELEASED).all()


def test_delayed_with_diffusion_reproducible_and_active(tmp_path):
    # Delayed release + stochastic diffusion: two identical runs are
    # bit-reproducible (counter-RNG keyed on the global step), and the diffused
    # path actually differs from the deterministic one (diffusion is active).
    from kelp.core.config import (
        AppConfig, ModeConfig, NumericsConfig, MeshConfig, DataConfig,
        TimeConfig, DiffusionConfig)
    from kelp.tracking import run_tracking
    from kelp.writers import load_trajectories
    from tests.test_tracking import _make_fesomc_3x3
    nc_path, _m, _mo = _make_fesomc_3x3(tmp_path, n_time=10, u_val=0.5, v_val=0.0)
    seeds = tmp_path / "s.dat"
    seeds.write_text("5.0 5.0\n")
    dt = 3600.0

    def _cfg(out, diffusion):
        return AppConfig(
            mode=ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0),
            numerics=NumericsConfig(dt=dt, integrator="euler", output_freq=1,
                                    diffusion=diffusion),
            mesh=MeshConfig(type="fesomc", meshpath=str(nc_path)),
            data=DataConfig(ncinput=str(nc_path), inicloud=str(seeds),
                            output_path=str(out), time_averaging="instant"),
            time=TimeConfig(start_time_idx=0, n_steps=6, n_pulses=2,
                            pulse_interval=2 * dt),
        )

    diff = dict(enabled=True, scheme="uniform", kh_zonal=20.0,
                kh_meridional=20.0, seed=123)
    o1 = tmp_path / "d1.nc"
    o2 = tmp_path / "d2.nc"
    o0 = tmp_path / "d0.nc"
    run_tracking(_cfg(o1, DiffusionConfig(**diff)))
    run_tracking(_cfg(o2, DiffusionConfig(**diff)))
    run_tracking(_cfg(o0, DiffusionConfig()))          # diffusion off

    t1, t2, t0 = (load_trajectories(str(o))["lon"] for o in (o1, o2, o0))
    # reproducible (ignore the NaN dormant region via equal_nan)
    np.testing.assert_array_equal(np.isnan(t1), np.isnan(t2))
    np.testing.assert_allclose(t1[~np.isnan(t1)], t2[~np.isnan(t2)], atol=0.0)
    # diffusion actually perturbed the released path (pulse 0, last obs)
    assert abs(float(t1[0, -1]) - float(t0[0, -1])) > 0.0


def test_end_to_end_off_is_normal_run(tmp_path):
    # n_pulses=1, no release column -> no NOT_RELEASED, no NaN, 1 trajectory.
    from kelp.core.config import (
        AppConfig, ModeConfig, NumericsConfig, MeshConfig, DataConfig, TimeConfig)
    from kelp.tracking import run_tracking
    from kelp.writers import load_trajectories
    from tests.test_tracking import _make_fesomc_3x3

    nc_path, _m, _mo = _make_fesomc_3x3(tmp_path, n_time=8, u_val=1.0, v_val=0.0)
    seeds = tmp_path / "seeds.dat"
    seeds.write_text("5.0 5.0\n")
    out = tmp_path / "traj.nc"
    cfg = AppConfig(
        mode=ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0),
        numerics=NumericsConfig(dt=3600.0, integrator="euler", output_freq=1),
        mesh=MeshConfig(type="fesomc", meshpath=str(nc_path)),
        data=DataConfig(ncinput=str(nc_path), inicloud=str(seeds),
                        output_path=str(out), time_averaging="instant"),
        time=TimeConfig(start_time_idx=0, n_steps=3),
    )
    run_tracking(cfg)
    tr = load_trajectories(str(out))
    assert tr["lon"].shape[0] == 1
    assert not np.isnan(tr["lon"]).any()
    assert (tr["status"] != ParticleStatus.NOT_RELEASED).all()
