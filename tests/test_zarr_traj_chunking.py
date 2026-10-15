"""Zarr trajectory-axis chunking for large particle counts (TODO 5.4).

With many particles the adaptive obs width bottoms out at 1 and a chunk
is the WHOLE particle column (~63 MB at 7.9M particles) — fine for
streaming and time-slice reads, terrible for particle-subset reads.
Above ``TRAJ_CHUNK_MAX`` the trajectory axis is split into blocks and
the obs width is derived from the BLOCK size, so a chunk stays near the
target bytes and a particle subset touches only its blocks.  Small runs
keep the old single-block layout byte-identically.
"""
import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from kelp.core.particles import make_state
from kelp.writers import (TrajectoryWriter, _ZarrBackend,
                              load_restart, load_trajectories)


def _states(n_obs, P, rich=False):
    """Deterministic distinct states; ``rich`` adds vel diagnostics,
    failure_reason, and a varying status so the lazy int/float vars all
    cross the split blocks."""
    out = []
    for t in range(n_obs):
        s = make_state(np.linspace(0.0, 1.0, P) + 0.01 * t,
                       np.linspace(10.0, 11.0, P))
        s.z[:] = 0.5 * t
        if rich:
            s.u = np.arange(P, dtype=np.float64) + 0.1 * t
            s.v = -np.arange(P, dtype=np.float64) - 0.2 * t
            s.w = np.full(P, 0.001 * t)
            s.failure_reason = ((np.arange(P) + t) % 3).astype(np.int8)
            s.status[t % P] = 3          # OUTSIDE — varies per record
        out.append(s)
    return out


def _write(path, states, sampled_seq=None, checkpoint_at=None):
    P = states[0].lon.size
    with TrajectoryWriter(str(path), n_particles=P) as w:
        for i, s in enumerate(states):
            samp = None if sampled_seq is None else sampled_seq[i]
            w.append(s, float(i * 100), sampled=samp)
            if checkpoint_at is not None and i == checkpoint_at:
                w.write_checkpoint(s, step_i=i, file_idx=i)
    return load_trajectories(str(path))


class TestTrajAxisChunking:
    def test_small_run_keeps_single_block(self, tmp_path):
        p = tmp_path / "small.zarr"
        _write(p, _states(5, 6))
        g = zarr.open_group(str(p), mode="r")
        assert g["lon"].chunks[0] == 6          # trajectory axis unsplit

    def test_large_run_splits_trajectory_axis(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_ZarrBackend, "TRAJ_CHUNK_MAX", 4)
        p = tmp_path / "big.zarr"
        sampled = [{"temp": np.full(10, float(t))} for t in range(5)]
        _write(p, _states(5, 10, rich=True), sampled)
        g = zarr.open_group(str(p), mode="r")
        # every (trajectory, obs) variable — core, lazy velocity/failure,
        # and sampled — must carry the split block (a revert of any one
        # create_array to (self._np, ...) fails here)
        for name in ("lon", "lat", "z", "status", "vel_u", "vel_v",
                     "vel_w", "failure_reason", "temp"):
            assert g[name].chunks[0] == 4, name

    def test_split_output_bitwise_equal_unsplit(self, tmp_path, monkeypatch):
        """Rich run (vel + failure_reason + varying status + sampled var
        + a MID-RUN checkpoint flush) — split blocks of 3 over 10
        particles (ragged last block) must be value-identical to the
        historical single-block layout, key for key."""
        states = _states(7, 10, rich=True)
        sampled = [{"temp": np.full(10, float(t))} for t in range(7)]
        ref = _write(tmp_path / "ref.zarr", states, sampled,
                     checkpoint_at=3)
        monkeypatch.setattr(_ZarrBackend, "TRAJ_CHUNK_MAX", 3)
        got = _write(tmp_path / "split.zarr", states, sampled,
                     checkpoint_at=3)
        assert set(got) == set(ref)
        assert "vel_u" in ref and "failure_reason" in ref
        for k in ref:
            np.testing.assert_array_equal(
                np.asarray(got[k]), np.asarray(ref[k]), err_msg=k)

    def test_late_sampled_var_backfills_across_blocks(self, tmp_path,
                                                      monkeypatch):
        """checkpoint_at=2 forces a FLUSH before ``salt`` first appears,
        so the on-disk resize backfill (not just in-buffer NaN stacking)
        runs across the split blocks (round-2 review: without the
        checkpoint everything flushed once at close and the backfill
        branch never executed)."""
        monkeypatch.setattr(_ZarrBackend, "TRAJ_CHUNK_MAX", 3)
        states = _states(6, 10)
        sampled = ([None] * 3
                   + [{"salt": np.full(10, 5.0 + t)} for t in range(3)])
        out = _write(tmp_path / "late.zarr", states, sampled,
                     checkpoint_at=2)
        salt = np.asarray(out["salt"])
        assert np.all(np.isnan(salt[:, :3]))    # NaN-backfilled columns
        for c, v in ((3, 5.0), (4, 6.0), (5, 7.0)):
            np.testing.assert_array_equal(salt[:, c], np.full(10, v))

    def test_checkpoint_restart_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_ZarrBackend, "TRAJ_CHUNK_MAX", 3)
        states = _states(6, 10)
        _write(tmp_path / "c.zarr", states, checkpoint_at=4)
        rst = load_restart(str(tmp_path / "c.zarr"))
        np.testing.assert_array_equal(rst["state"].lon, states[4].lon)
        np.testing.assert_array_equal(rst["state"].lat, states[4].lat)
        np.testing.assert_array_equal(rst["state"].status,
                                      states[4].status)
        assert rst["step_i"] == 4
        assert rst["file_idx"] == 4

    def test_obs_chunk_derived_from_block_size(self, tmp_path, monkeypatch):
        """The obs width must come from the BLOCK's byte size, not the
        full particle count — otherwise splitting the axis would leave
        obs at 1 even for tiny blocks."""
        monkeypatch.setattr(_ZarrBackend, "TRAJ_CHUNK_MAX", 4)
        monkeypatch.setattr(_ZarrBackend, "TARGET_CHUNK_BYTES", 4 * 8 * 5)
        p = tmp_path / "o.zarr"
        _write(p, _states(3, 10))
        g = zarr.open_group(str(p), mode="r")
        assert g["lon"].chunks == (4, 5)
