"""Tests for kelp.writers — TrajectoryWriter, load_restart, load_trajectories."""
from __future__ import annotations

import numpy as np
import pytest

from kelp.core.particles import make_state, ParticleState, ParticleStatus
from kelp.writers import (TrajectoryWriter, load_restart, load_trajectories,
                              write_restart_file)


# ===================================================================
# TrajectoryWriter basics
# ===================================================================

class TestTrajectoryWriter:
    def test_open_close(self, tmp_path):
        """Writer creates a valid NetCDF file."""
        path = str(tmp_path / "out.nc")
        writer = TrajectoryWriter(path, n_particles=3)
        writer.open()
        assert writer.n_obs == 0
        writer.close()

        data = load_trajectories(path)
        assert data["trajectory"].shape == (3,)
        assert data["time"].shape == (0,)

    def test_context_manager(self, tmp_path):
        """Writer works as context manager."""
        path = str(tmp_path / "out.nc")
        with TrajectoryWriter(path, n_particles=2) as w:
            state = make_state(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
            w.append(state, 0.0)
            assert w.n_obs == 1

        data = load_trajectories(path)
        assert data["lon"].shape == (2, 1)  # (P, T)

    def test_append_multiple(self, tmp_path):
        """Multiple appends produce correct shape."""
        path = str(tmp_path / "out.nc")
        with TrajectoryWriter(path, n_particles=2) as w:
            s1 = make_state(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
            s2 = make_state(np.array([1.1, 2.1]), np.array([3.1, 4.1]))
            s3 = make_state(np.array([1.2, 2.2]), np.array([3.2, 4.2]))
            w.append(s1, 0.0)
            w.append(s2, 100.0)
            w.append(s3, 200.0)
            assert w.n_obs == 3

        data = load_trajectories(path)
        assert data["lon"].shape == (2, 3)   # (P, T)
        assert data["time"].shape == (3,)
        np.testing.assert_allclose(data["time"], [0.0, 100.0, 200.0])
        np.testing.assert_allclose(data["lon"][0, :], [1.0, 1.1, 1.2])
        np.testing.assert_allclose(data["lat"][1, :], [4.0, 4.1, 4.2])

    def test_status_written(self, tmp_path):
        """Status codes are recorded correctly."""
        path = str(tmp_path / "out.nc")
        with TrajectoryWriter(path, n_particles=2) as w:
            state = make_state(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
            state.status[1] = ParticleStatus.OUTSIDE
            w.append(state, 0.0)

        data = load_trajectories(path)
        assert data["status"][0, 0] == ParticleStatus.ALIVE
        assert data["status"][1, 0] == ParticleStatus.OUTSIDE

    def test_z_written(self, tmp_path):
        """Depth values are recorded."""
        path = str(tmp_path / "out.nc")
        with TrajectoryWriter(path, n_particles=2) as w:
            state = make_state(
                np.array([1.0, 2.0]),
                np.array([3.0, 4.0]),
                np.array([10.0, 20.0]),
            )
            w.append(state, 0.0)

        data = load_trajectories(path)
        np.testing.assert_allclose(data["z"][:, 0], [10.0, 20.0])

    def test_metadata(self, tmp_path):
        """Global attributes are stored."""
        import netCDF4 as nc4
        path = str(tmp_path / "out.nc")
        with TrajectoryWriter(path, n_particles=1, metadata={"integrator": "rk4", "dt": 900.0}):
            pass

        with nc4.Dataset(path, "r") as ds:
            assert ds.getncattr("integrator") == "rk4"
            assert ds.getncattr("dt") == 900.0

    def test_not_open_raises(self, tmp_path):
        """Appending without open() raises RuntimeError."""
        w = TrajectoryWriter(str(tmp_path / "never_opened.nc"), n_particles=1)
        state = make_state(np.array([1.0]), np.array([1.0]))
        with pytest.raises((RuntimeError, AttributeError)):
            w.append(state, 0.0)


# ===================================================================
# Zarr backend
# ===================================================================

class TestZarrWriter:
    def test_zarr_roundtrip(self, tmp_path):
        """Write and read zarr trajectory store."""
        path = str(tmp_path / "out.zarr")
        s1 = make_state(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        s2 = make_state(np.array([1.1, 2.1]), np.array([3.1, 4.1]))

        with TrajectoryWriter(path, n_particles=2) as w:
            w.append(s1, 0.0)
            w.append(s2, 100.0)
            assert w.n_obs == 2

        data = load_trajectories(path)
        assert data["lon"].shape == (2, 2)
        np.testing.assert_allclose(data["time"], [0.0, 100.0])
        np.testing.assert_allclose(data["lon"][0, :], [1.0, 1.1])

    def test_zarr_checkpoint(self, tmp_path):
        """Zarr checkpoint roundtrip."""
        path = str(tmp_path / "ckpt.zarr")
        state = make_state(np.array([10.5]), np.array([50.1]))
        state = state.replace(nearest_node=np.array([42], dtype=np.int32))

        with TrajectoryWriter(path, n_particles=1) as w:
            w.append(state, 0.0)
            w.write_checkpoint(state, step_i=7, file_idx=3)

        restart = load_restart(path)
        assert restart["step_i"] == 7
        assert restart["file_idx"] == 3
        np.testing.assert_allclose(restart["state"].lon, [10.5])
        np.testing.assert_array_equal(restart["state"].nearest_node, [42])

    def test_zarr_no_checkpoint_raises(self, tmp_path):
        """Loading restart from zarr without checkpoint raises ValueError."""
        path = str(tmp_path / "no_ckpt.zarr")
        with TrajectoryWriter(path, n_particles=1):
            pass
        with pytest.raises(ValueError, match="No valid restart"):
            load_restart(path)

    def test_zarr_status(self, tmp_path):
        """Status codes written correctly to zarr."""
        path = str(tmp_path / "status.zarr")
        state = make_state(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        state.status[1] = ParticleStatus.OUTSIDE

        with TrajectoryWriter(path, n_particles=2) as w:
            w.append(state, 0.0)

        data = load_trajectories(path)
        assert data["status"][0, 0] == ParticleStatus.ALIVE
        assert data["status"][1, 0] == ParticleStatus.OUTSIDE


# ===================================================================
# Restart checkpoint (NetCDF)
# ===================================================================

class TestCheckpoint:
    def test_checkpoint_roundtrip(self, tmp_path):
        """Write checkpoint, load it, verify state matches."""
        path = str(tmp_path / "ckpt.nc")
        state = make_state(
            np.array([10.5, 20.3]),
            np.array([50.1, 60.2]),
            np.array([5.0, 15.0]),
        )
        state.status[1] = ParticleStatus.BEACHED
        state = state.replace(nearest_node=np.array([42, 99], dtype=np.int32))

        with TrajectoryWriter(path, n_particles=2) as w:
            w.append(state, 0.0)
            w.write_checkpoint(state, step_i=7, file_idx=3)

        restart = load_restart(path)
        assert restart["step_i"] == 7
        assert restart["file_idx"] == 3
        np.testing.assert_allclose(restart["state"].lon, [10.5, 20.3])
        np.testing.assert_allclose(restart["state"].lat, [50.1, 60.2])
        np.testing.assert_allclose(restart["state"].z, [5.0, 15.0])
        assert restart["state"].status[0] == ParticleStatus.ALIVE
        assert restart["state"].status[1] == ParticleStatus.BEACHED
        np.testing.assert_array_equal(restart["state"].nearest_node, [42, 99])

    def test_checkpoint_overwrite(self, tmp_path):
        """Second checkpoint overwrites first."""
        path = str(tmp_path / "ckpt2.nc")
        s1 = make_state(np.array([1.0]), np.array([2.0]))
        s2 = make_state(np.array([3.0]), np.array([4.0]))

        with TrajectoryWriter(path, n_particles=1) as w:
            w.append(s1, 0.0)
            w.write_checkpoint(s1, step_i=0, file_idx=0)
            w.write_checkpoint(s2, step_i=5, file_idx=5)

        restart = load_restart(path)
        assert restart["step_i"] == 5
        np.testing.assert_allclose(restart["state"].lon, [3.0])

    def test_no_checkpoint_raises(self, tmp_path):
        """Loading restart from file without checkpoint raises ValueError."""
        path = str(tmp_path / "no_ckpt.nc")
        with TrajectoryWriter(path, n_particles=1):
            pass

        with pytest.raises(ValueError, match="No valid restart"):
            load_restart(path)


# ===================================================================
# load_trajectories
# ===================================================================

class TestLoadTrajectories:
    def test_roundtrip(self, tmp_path):
        """Write then read full trajectory data."""
        path = str(tmp_path / "traj.nc")
        s1 = make_state(np.array([5.0, 6.0, 7.0]), np.array([50.0, 51.0, 52.0]))
        s2 = make_state(np.array([5.1, 6.1, 7.1]), np.array([50.1, 51.1, 52.1]))

        with TrajectoryWriter(path, n_particles=3) as w:
            w.append(s1, 0.0)
            w.append(s2, 3600.0)

        data = load_trajectories(path)
        assert data["lon"].shape == (3, 2)
        assert data["lat"].shape == (3, 2)
        assert data["z"].shape == (3, 2)
        assert data["status"].shape == (3, 2)
        assert data["time"].shape == (2,)
        assert data["trajectory"].shape == (3,)
        np.testing.assert_array_equal(data["trajectory"], [0, 1, 2])
        np.testing.assert_allclose(data["lon"][0, :], [5.0, 5.1])
        np.testing.assert_allclose(data["time"], [0.0, 3600.0])


class TestStandaloneRestartFile:
    def test_write_restart_file_roundtrip(self, tmp_path):
        """write_restart_file -> load_restart recovers the full state from a
        self-contained restart file (not embedded in a trajectory output)."""
        P = 8
        s = make_state(np.linspace(0, 7, P), np.linspace(40, 47, P),
                       np.linspace(5, 500, P))
        s = s.replace(nearest_node=np.arange(P, dtype=np.int32))
        wet = (np.arange(P) % 3).astype(np.int32)
        p = str(tmp_path / "restart_step000042.nc")
        write_restart_file(p, s, step_i=42, file_idx=7, wet_streak=wet,
                           rotation_angles=(50.0, 15.0, -90.0))
        r = load_restart(p)
        assert r["step_i"] == 42 and r["file_idx"] == 7
        np.testing.assert_array_equal(r["state"].lon, s.lon)
        np.testing.assert_array_equal(r["state"].lat, s.lat)
        np.testing.assert_array_equal(r["state"].z, s.z)
        np.testing.assert_array_equal(r["state"].nearest_node, s.nearest_node)
        np.testing.assert_array_equal(r["wet_streak"], wet)
        assert r["rotation_angles"] == (50.0, 15.0, -90.0)
