"""Tests for the buffered / close-reopen NetCDF writer mode (flush_every).

The motivation: NetCDF4 = HDF5 holds an exclusive write lock while the file
is open, so a streaming writer's output is unreadable by other processes
until the run ends.  ``flush_every=N`` buffers N record points, then opens
the file in append mode, writes the batch, and closes it — releasing the lock
between flushes so the file is readable mid-run.  Output must stay identical
to streaming mode.
"""
from __future__ import annotations

import numpy as np
import pytest

from kelp.core.particles import make_state, ParticleStatus
from kelp.core.config import NumericsConfig
from kelp.writers import TrajectoryWriter, load_restart, load_trajectories


def _make_states(n_obs, n_particles=3, with_vel=False, with_failure=False):
    """A deterministic sequence of distinct states."""
    states = []
    for t in range(n_obs):
        lon = np.array([1.0, 2.0, 3.0])[:n_particles] + 0.1 * t
        lat = np.array([10.0, 20.0, 30.0])[:n_particles] + 0.1 * t
        s = make_state(lon, lat)
        s.z[:] = 0.5 * t
        if with_vel:
            s.u = np.full(n_particles, 0.1 * t + 0.01)
            s.v = np.full(n_particles, -0.2 * t - 0.02)
            s.w = np.full(n_particles, 0.001 * t)
        if with_failure:
            s.failure_reason = np.zeros(n_particles, dtype=np.int8)
        states.append(s)
    return states


def _write_run(path, states, flush_every, sampled_seq=None):
    """Write a sequence through a writer; return loaded trajectory data."""
    with TrajectoryWriter(path, n_particles=states[0].lon.size,
                          flush_every=flush_every) as w:
        for i, s in enumerate(states):
            samp = None if sampled_seq is None else sampled_seq[i]
            w.append(s, float(i * 100), sampled=samp)
    return load_trajectories(path)


# ===================================================================
# Data identity: buffered == streaming
# ===================================================================

class TestBufferedMatchesStreaming:
    @pytest.mark.parametrize("flush_every", [1, 2, 3, 7])
    def test_core_fields_identical(self, tmp_path, flush_every):
        states = _make_states(5)
        ref = _write_run(str(tmp_path / "stream.nc"), states, flush_every=0)
        buf = _write_run(str(tmp_path / "buf.nc"), states,
                         flush_every=flush_every)
        for key in ("lon", "lat", "z", "status", "time"):
            np.testing.assert_array_equal(buf[key], ref[key],
                                          err_msg=f"mismatch in {key}")

    def test_velocity_and_failure_identical(self, tmp_path):
        states = _make_states(5, with_vel=True, with_failure=True)
        ref = _write_run(str(tmp_path / "stream.nc"), states, flush_every=0)
        buf = _write_run(str(tmp_path / "buf.nc"), states, flush_every=2)
        for key in ("vel_u", "vel_v", "vel_w", "failure_reason"):
            np.testing.assert_array_equal(buf[key], ref[key],
                                          err_msg=f"mismatch in {key}")

    def test_sampled_fields_identical(self, tmp_path):
        states = _make_states(4)
        sampled = [{"temp": np.full(3, 10.0 + t),
                    "salt": np.full(3, 35.0 - t)} for t in range(4)]
        ref = _write_run(str(tmp_path / "stream.nc"), states,
                         flush_every=0, sampled_seq=sampled)
        buf = _write_run(str(tmp_path / "buf.nc"), states,
                         flush_every=2, sampled_seq=sampled)
        for key in ("temp", "salt"):
            np.testing.assert_array_equal(buf[key], ref[key],
                                          err_msg=f"mismatch in {key}")


# ===================================================================
# Lazy variables first appearing AFTER a flush (back-fill across reopen)
# ===================================================================

class TestLazyVarsAcrossFlush:
    """The riskiest interaction: a velocity / sampled variable that first
    appears at obs_idx > 0, when the file has already been flushed and closed,
    must be created in the re-opened "a"-mode dataset and back-fill the earlier
    obs columns with the fill value — identical to streaming mode."""

    def test_sampled_var_first_appears_after_flush(self, tmp_path):
        states = _make_states(5)
        # 'temp' absent for obs 0,1 (flushed at flush_every=2), appears at 2.
        sampled = [None, None,
                   {"temp": np.full(3, 12.0)},
                   {"temp": np.full(3, 13.0)},
                   {"temp": np.full(3, 14.0)}]
        ref = _write_run(str(tmp_path / "stream.nc"), states,
                         flush_every=0, sampled_seq=sampled)
        buf = _write_run(str(tmp_path / "buf.nc"), states,
                         flush_every=2, sampled_seq=sampled)
        assert buf["temp"].shape == ref["temp"].shape == (3, 5)
        # NaN back-fill on the first two obs, values after — exactly streaming.
        np.testing.assert_array_equal(buf["temp"], ref["temp"])
        assert np.isnan(buf["temp"][:, :2]).all()
        assert not np.isnan(buf["temp"][:, 2:]).any()

    def test_velocity_first_appears_after_flush(self, tmp_path):
        # vel set only from obs 3 onward — created on a re-opened dataset.
        states = []
        for t in range(5):
            s = make_state(np.array([1.0, 2.0, 3.0]) + 0.1 * t,
                           np.array([10.0, 20.0, 30.0]) + 0.1 * t)
            if t >= 3:
                s.u = np.full(3, 0.5)
                s.v = np.full(3, -0.5)
                s.w = np.full(3, 0.01)
            states.append(s)
        ref = _write_run(str(tmp_path / "stream.nc"), states, flush_every=0)
        buf = _write_run(str(tmp_path / "buf.nc"), states, flush_every=2)
        np.testing.assert_array_equal(buf["vel_u"], ref["vel_u"])
        np.testing.assert_array_equal(buf["vel_v"], ref["vel_v"])


# ===================================================================
# Buffer copy-safety (aliasing) — the reason append() does .copy()
# ===================================================================

class TestBufferAliasing:
    def test_reused_mutated_state_arrays_are_snapshotted(self, tmp_path):
        """Reuse ONE state object and mutate its arrays in place between
        appends.  Buffered mode must have copied each record, so the file
        holds the per-step values, not the final mutated arrays."""
        path = str(tmp_path / "out.nc")
        s = make_state(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        sampled = {"temp": np.array([10.0, 20.0])}
        with TrajectoryWriter(path, n_particles=2, flush_every=10) as w:
            for t in range(4):
                w.append(s, float(t), sampled=sampled)
                # mutate the SAME arrays in place for the next step
                s.lon[:] += 1.0
                s.lat[:] += 1.0
                sampled["temp"][:] += 1.0
        data = load_trajectories(path)
        # lon should be [1,2,3,4] for particle 0, not all 4.0 (final value)
        np.testing.assert_array_equal(data["lon"][0, :], [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(data["temp"][0, :],
                                      [10.0, 11.0, 12.0, 13.0])


# ===================================================================
# Mid-run readability (the whole point of the feature)
# ===================================================================

class TestMidRunReadable:
    def test_flushed_records_readable_before_close(self, tmp_path):
        """After flushes, the file is closed on disk and another reader can
        open it WHILE the writer object is still alive (not closed)."""
        path = str(tmp_path / "out.nc")
        states = _make_states(5)
        w = TrajectoryWriter(path, n_particles=3, flush_every=2)
        w.open()
        for i, s in enumerate(states):
            w.append(s, float(i * 100))
        # 5 appended with flush_every=2 → flushes at 2 and 4; the 5th is
        # still buffered.  The writer is NOT closed yet.
        assert w.n_obs == 5  # counts buffered too
        data = load_trajectories(path)  # reader opens the unlocked file
        assert data["time"].shape == (4,)  # only the 4 flushed are on disk
        np.testing.assert_array_equal(data["time"], [0.0, 100.0, 200.0, 300.0])

        w.close()  # flushes the tail
        data2 = load_trajectories(path)
        assert data2["time"].shape == (5,)
        np.testing.assert_array_equal(
            data2["time"], [0.0, 100.0, 200.0, 300.0, 400.0])

    def test_empty_file_readable_after_open(self, tmp_path):
        """In buffered mode open() creates + closes the file (schema only)."""
        path = str(tmp_path / "out.nc")
        w = TrajectoryWriter(path, n_particles=3, flush_every=4)
        w.open()
        data = load_trajectories(path)  # readable immediately, 0 obs
        assert data["time"].shape == (0,)
        assert data["trajectory"].shape == (3,)
        w.close()


# ===================================================================
# Tail flush + n_obs bookkeeping
# ===================================================================

class TestFlushBookkeeping:
    def test_tail_flushed_on_close(self, tmp_path):
        states = _make_states(4)
        data = _write_run(str(tmp_path / "out.nc"), states, flush_every=3)
        assert data["time"].shape == (4,)  # 3 flushed + 1 tail

    def test_n_obs_counts_buffered(self, tmp_path):
        path = str(tmp_path / "out.nc")
        states = _make_states(3)
        with TrajectoryWriter(path, n_particles=3, flush_every=10) as w:
            for i, s in enumerate(states):
                w.append(s, float(i))
                assert w.n_obs == i + 1  # buffered records counted


# ===================================================================
# Checkpoints in buffered mode
# ===================================================================

class TestBufferedCheckpoint:
    def test_checkpoint_roundtrip(self, tmp_path):
        path = str(tmp_path / "out.nc")
        states = _make_states(5)
        with TrajectoryWriter(path, n_particles=3, flush_every=2) as w:
            for i, s in enumerate(states):
                w.append(s, float(i * 100))
                if i == 2:
                    w.write_checkpoint(s, step_i=2, file_idx=7)
        # All observations present...
        data = load_trajectories(path)
        assert data["time"].shape == (5,)
        # ...and the checkpoint is valid and round-trips.
        r = load_restart(path)
        assert r["step_i"] == 2
        assert r["file_idx"] == 7
        np.testing.assert_array_equal(r["state"].lon, states[2].lon)
        np.testing.assert_array_equal(r["state"].status, states[2].status)

    def test_checkpoint_at_flush_boundary(self, tmp_path):
        """Checkpoint right when a flush has just emptied the buffer."""
        path = str(tmp_path / "out.nc")
        states = _make_states(6)
        with TrajectoryWriter(path, n_particles=3, flush_every=2) as w:
            for i, s in enumerate(states):
                w.append(s, float(i * 100))
                if i == 3:  # i==3 → buffer just flushed at len 2 (idx 2,3)
                    w.write_checkpoint(s, step_i=3, file_idx=11)
        data = load_trajectories(path)
        assert data["time"].shape == (6,)
        r = load_restart(path)
        assert r["step_i"] == 3 and r["file_idx"] == 11
        np.testing.assert_array_equal(r["state"].lon, states[3].lon)

    def test_multiple_checkpoints_no_double_create(self, tmp_path):
        """Two checkpoints across reopens must not double-create restart_* vars
        and the latest one wins."""
        path = str(tmp_path / "out.nc")
        states = _make_states(6)
        with TrajectoryWriter(path, n_particles=3, flush_every=2) as w:
            for i, s in enumerate(states):
                w.append(s, float(i * 100))
                if i in (1, 4):
                    w.write_checkpoint(s, step_i=i, file_idx=i + 100)
        data = load_trajectories(path)
        assert data["time"].shape == (6,)
        r = load_restart(path)  # latest checkpoint wins
        assert r["step_i"] == 4 and r["file_idx"] == 104
        np.testing.assert_array_equal(r["state"].lon, states[4].lon)


# ===================================================================
# Zarr ignores flush_every (already lock-free)
# ===================================================================

class TestZarrIgnoresFlushEvery:
    def test_zarr_with_flush_every(self, tmp_path):
        zarr = pytest.importorskip("zarr")  # noqa: F841
        path = str(tmp_path / "out.zarr")
        states = _make_states(4)
        with TrajectoryWriter(path, n_particles=3, flush_every=5) as w:
            for i, s in enumerate(states):
                w.append(s, float(i))
        data = load_trajectories(path)
        assert data["time"].shape == (4,)


# ===================================================================
# Zarr block-buffering + adaptive chunk width (writer perf fix)
# ===================================================================

class TestZarrChunking:
    def test_adaptive_obs_chunk_width(self):
        """Huge particle counts shrink the obs-chunk (so a chunk is not
        hundreds of MB); tiny meshes stay at the cap."""
        from kelp.writers import _ZarrBackend
        tiny = _ZarrBackend("x.zarr", 100, None)
        huge = _ZarrBackend("x.zarr", 200_000, None)
        assert tiny._obs_chunk == _ZarrBackend.OBS_CHUNK_MAX
        assert 1 <= huge._obs_chunk < _ZarrBackend.OBS_CHUNK_MAX

    def test_block_buffer_identity_vs_netcdf(self, tmp_path):
        """Block-buffered Zarr (multi-block + partial tail) reproduces the
        NetCDF streaming data exactly."""
        pytest.importorskip("zarr")
        states = _make_states(7)
        ref = _write_run(str(tmp_path / "s.nc"), states, flush_every=0)
        zp = str(tmp_path / "o.zarr")
        w = TrajectoryWriter(zp, n_particles=3)
        w._backend._obs_chunk = 3   # 7 records -> blocks 3 + 3 + tail 1
        with w:
            for i, s in enumerate(states):
                w.append(s, float(i * 100))
        zd = load_trajectories(zp)
        for k in ("lon", "lat", "z", "status", "time"):
            np.testing.assert_array_equal(zd[k], ref[k], err_msg=k)

    def test_sampled_var_after_chunk_boundary(self, tmp_path):
        """A sampled var first appearing in a LATER block back-fills the
        already-written obs with NaN."""
        pytest.importorskip("zarr")
        states = _make_states(6)
        sampled = [None, None, None,
                   {"temp": np.full(3, 9.0)},
                   {"temp": np.full(3, 8.0)},
                   {"temp": np.full(3, 7.0)}]
        zp = str(tmp_path / "o.zarr")
        w = TrajectoryWriter(zp, n_particles=3)
        w._backend._obs_chunk = 3   # 'temp' appears at obs 3 = a fresh block
        with w:
            for i, s in enumerate(states):
                w.append(s, float(i), sampled=sampled[i])
        zd = load_trajectories(zp)
        assert zd["temp"].shape == (3, 6)
        assert np.isnan(zd["temp"][:, :3]).all()
        np.testing.assert_array_equal(zd["temp"][:, 3], np.full(3, 9.0))


# ===================================================================
# Zarr block path — identity across ALL fields + lazy-var edge cases
# ===================================================================

class TestZarrBlockFields:
    def test_all_fields_identity_vs_netcdf(self, tmp_path):
        """vel + failure_reason + sampled through the Zarr block path match
        the NetCDF streaming writer exactly (multi-block + tail)."""
        pytest.importorskip("zarr")
        states = _make_states(7, with_vel=True, with_failure=True)
        sampled = [{"temp": np.full(3, 10.0 + t)} for t in range(7)]
        ref = _write_run(str(tmp_path / "s.nc"), states,
                         flush_every=0, sampled_seq=sampled)
        zp = str(tmp_path / "o.zarr")
        w = TrajectoryWriter(zp, n_particles=3)
        w._backend._obs_chunk = 3
        with w:
            for i, s in enumerate(states):
                w.append(s, float(i * 100), sampled=sampled[i])
        zd = load_trajectories(zp)
        for k in ("lon", "lat", "z", "status", "time",
                  "vel_u", "vel_v", "vel_w", "failure_reason", "temp"):
            np.testing.assert_array_equal(zd[k], ref[k], err_msg=k)

    def test_sampled_var_appears_mid_block(self, tmp_path):
        """'temp' first appears at obs 4 = the MIDDLE of the 2nd block
        (chunk 3 → blocks [0,1,2],[3,4,5]); back-fill to _flushed + within-block
        NaN fill must combine correctly."""
        pytest.importorskip("zarr")
        states = _make_states(6)
        sampled = [None, None, None, None,
                   {"temp": np.full(3, 5.0)}, {"temp": np.full(3, 6.0)}]
        zp = str(tmp_path / "o.zarr")
        w = TrajectoryWriter(zp, n_particles=3)
        w._backend._obs_chunk = 3
        with w:
            for i, s in enumerate(states):
                w.append(s, float(i), sampled=sampled[i])
        zd = load_trajectories(zp)
        assert zd["temp"].shape == (3, 6)
        assert np.isnan(zd["temp"][:, :4]).all()
        np.testing.assert_array_equal(zd["temp"][:, 4], np.full(3, 5.0))

    def test_trailing_gap_lazy_var_padded_on_close(self, tmp_path):
        """A vel present only in the FIRST block then absent must be padded to
        the full obs width at close (Zarr arrays are independently shaped)."""
        pytest.importorskip("zarr")
        states = []
        for t in range(6):
            s = make_state(np.array([1.0, 2.0, 3.0]) + 0.1 * t,
                           np.array([10.0, 20.0, 30.0]) + 0.1 * t)
            if t < 3:
                s.u = np.full(3, 0.5)
                s.v = np.full(3, -0.5)
                s.w = np.full(3, 0.0)
            states.append(s)
        zp = str(tmp_path / "o.zarr")
        w = TrajectoryWriter(zp, n_particles=3)
        w._backend._obs_chunk = 3
        with w:
            for i, s in enumerate(states):
                w.append(s, float(i))
        zd = load_trajectories(zp)
        assert zd["vel_u"].shape == (3, 6)        # padded, obs-aligned
        np.testing.assert_array_equal(zd["vel_u"][:, :3], 0.5)
        # fill: NaN since output format 2 (format 1 padded with 0)
        assert np.isnan(zd["vel_u"][:, 3:]).all()


# ===================================================================
# Not-open guard + reuse reset (review round 2)
# ===================================================================

class TestNotOpenGuard:
    @pytest.mark.parametrize("path,kw", [
        ("out.nc", {}), ("out.nc", {"flush_every": 2}), ("out.zarr", {})])
    def test_append_before_open_raises(self, tmp_path, path, kw):
        if path.endswith(".zarr"):
            pytest.importorskip("zarr")
        w = TrajectoryWriter(str(tmp_path / path), n_particles=2, **kw)
        s = make_state(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        with pytest.raises(RuntimeError, match="not open"):
            w.append(s, 0.0)

    @pytest.mark.parametrize("path,kw", [
        ("out.nc", {}), ("out.nc", {"flush_every": 2}), ("out.zarr", {})])
    def test_append_after_close_raises(self, tmp_path, path, kw):
        if path.endswith(".zarr"):
            pytest.importorskip("zarr")
        w = TrajectoryWriter(str(tmp_path / path), n_particles=2, **kw)
        s = make_state(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        with w:
            w.append(s, 0.0)
        with pytest.raises(RuntimeError, match="not open"):
            w.append(s, 1.0)

    def test_reopen_resets_state(self, tmp_path):
        """Reusing one writer object across open()s starts a fresh file at obs
        0 (no stale _flushed/_obs_idx/_obs_buf)."""
        pytest.importorskip("zarr")
        zp = str(tmp_path / "o.zarr")
        w = TrajectoryWriter(zp, n_particles=2)
        with w:
            for i in range(3):
                w.append(make_state(np.array([1.0, 2.0]),
                                    np.array([3.0, 4.0])), float(i))
        w.open()  # fresh file on the SAME object
        w.append(make_state(np.array([9.0, 9.0]), np.array([9.0, 9.0])), 0.0)
        w.close()
        data = load_trajectories(zp)
        assert data["time"].shape == (1,)   # reset — not 4


# ===================================================================
# Config validation
# ===================================================================

class TestConfigValidation:
    def test_default_is_zero(self):
        assert NumericsConfig().output_flush_every == 0

    @pytest.mark.parametrize("bad", [-1, True, 1.5, "4"])
    def test_invalid_rejected(self, bad):
        with pytest.raises(ValueError, match="output_flush_every"):
            NumericsConfig(output_flush_every=bad)

    def test_valid_accepted(self):
        assert NumericsConfig(output_flush_every=100).output_flush_every == 100
