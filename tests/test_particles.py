"""Tests for kelp.core.particles."""
import os
import tempfile

import numpy as np
import pytest

from kelp.core.particles import (
    ParticleStatus, ParticleState,
    status_to_string, count_by_status,
    make_state, load_particles,
)


# ---------------------------------------------------------------------------
# Status utilities
# ---------------------------------------------------------------------------

class TestStatus:
    def test_values(self):
        assert int(ParticleStatus.ALIVE) == 1
        assert int(ParticleStatus.BEACHED) == 2
        assert int(ParticleStatus.OUTSIDE) == 3

    def test_status_to_string(self):
        assert status_to_string(1) == "alive"
        assert status_to_string(2) == "beached"
        assert status_to_string(3) == "outside"
        assert "unknown" in status_to_string(99)

    def test_count_by_status(self):
        status = np.array([1, 1, 2, 3, 1], dtype=np.int32)
        counts = count_by_status(status)
        assert counts["alive"] == 3
        assert counts["beached"] == 1
        assert counts["outside"] == 1
        assert counts["total"] == 5


# ---------------------------------------------------------------------------
# make_state
# ---------------------------------------------------------------------------

class TestMakeState:
    def test_2d(self):
        state = make_state(np.array([10.0, 20.0]), np.array([50.0, 55.0]))
        assert state.n_particles == 2
        np.testing.assert_array_equal(state.lon, [10.0, 20.0])
        np.testing.assert_array_equal(state.lat, [50.0, 55.0])
        np.testing.assert_array_equal(state.z, [0.0, 0.0])
        assert np.all(state.status == ParticleStatus.ALIVE)
        assert np.all(state.nearest_node == -1)

    def test_3d(self):
        state = make_state(
            np.array([10.0]), np.array([50.0]), np.array([5.0])
        )
        assert state.z[0] == 5.0

    def test_alive_mask(self):
        state = make_state(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
        state.status[1] = ParticleStatus.BEACHED
        mask = state.alive_mask
        np.testing.assert_array_equal(mask, [True, False, True])


# ---------------------------------------------------------------------------
# replace
# ---------------------------------------------------------------------------

class TestReplace:
    def test_replace_lon(self):
        state = make_state(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        new_lon = np.array([10.0, 20.0])
        new_state = state.replace(lon=new_lon)
        np.testing.assert_array_equal(new_state.lon, [10.0, 20.0])
        # original unchanged
        np.testing.assert_array_equal(state.lon, [1.0, 2.0])
        # lat carried over
        np.testing.assert_array_equal(new_state.lat, [3.0, 4.0])


# ---------------------------------------------------------------------------
# snapshot
# ---------------------------------------------------------------------------

class TestSnapshot:
    def test_snapshot_is_copy(self):
        state = make_state(np.array([1.0]), np.array([2.0]))
        snap = state.snapshot()
        assert "lon" in snap
        assert "lat" in snap
        assert "z" in snap
        assert "status" in snap
        # Mutating snapshot doesn't affect original
        snap["lon"][0] = 999.0
        assert state.lon[0] == 1.0


# ---------------------------------------------------------------------------
# load_particles
# ---------------------------------------------------------------------------

class TestLoadParticles:
    def test_load_2d(self):
        fd, path = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, "w") as f:
            f.write("10.5 54.3\n11.2 55.1\n12.0 53.8\n")
        try:
            state = load_particles(path)
            assert state.n_particles == 3
            assert state.lon[0] == pytest.approx(10.5)
            assert state.lat[1] == pytest.approx(55.1)
            np.testing.assert_array_equal(state.z, [0.0, 0.0, 0.0])
        finally:
            os.unlink(path)

    def test_load_3d(self):
        fd, path = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, "w") as f:
            f.write("10.5 54.3 5.0\n11.2 55.1 10.0\n")
        try:
            state = load_particles(path)
            assert state.n_particles == 2
            assert state.z[0] == pytest.approx(5.0)
            assert state.z[1] == pytest.approx(10.0)
        finally:
            os.unlink(path)

    def test_load_with_comments(self):
        fd, path = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, "w") as f:
            f.write("# lon lat depth\n10.5 54.3 0.0\n# another comment\n11.2 55.1 0.0\n")
        try:
            state = load_particles(path)
            assert state.n_particles == 2
        finally:
            os.unlink(path)

    def test_load_single_particle(self):
        fd, path = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, "w") as f:
            f.write("10.5 54.3\n")
        try:
            state = load_particles(path)
            assert state.n_particles == 1
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# ParticleState.save / .load classmethods
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_roundtrip_2d(self, tmp_path):
        state = make_state(np.array([10.5, 11.2]), np.array([54.3, 55.1]))
        path = str(tmp_path / "particles.dat")
        state.save(path)
        loaded = ParticleState.load(path)
        assert loaded.n_particles == 2
        np.testing.assert_allclose(loaded.lon, state.lon)
        np.testing.assert_allclose(loaded.lat, state.lat)
        np.testing.assert_array_equal(loaded.z, [0.0, 0.0])

    def test_roundtrip_3d(self, tmp_path):
        state = make_state(
            np.array([10.5, 11.2]),
            np.array([54.3, 55.1]),
            np.array([5.0, 10.0]),
        )
        path = str(tmp_path / "particles_3d.dat")
        state.save(path)
        loaded = ParticleState.load(path)
        assert loaded.n_particles == 2
        np.testing.assert_allclose(loaded.z, [5.0, 10.0])

    def test_save_has_header(self, tmp_path):
        state = make_state(np.array([1.0]), np.array([2.0]))
        path = str(tmp_path / "p.dat")
        state.save(path)
        with open(path) as f:
            header = f.readline()
        assert header.startswith("#")


# ---------------------------------------------------------------------------
# ParticleState.from_grid / .from_around_point
# ---------------------------------------------------------------------------

class TestClassmethodGenerators:
    @staticmethod
    def _make_simple_mesh():
        from kelp.core.mesh import Mesh, build_search, build_node_to_elems
        xs = np.array([4.0, 5.0, 6.0])
        ys = np.array([4.0, 5.0, 6.0])
        gx, gy = np.meshgrid(xs, ys)
        lon = gx.ravel().astype(np.float64)
        lat = gy.ravel().astype(np.float64)
        elems = []
        for j in range(2):
            for i in range(2):
                n0 = j * 3 + i
                elems.append([n0, n0 + 1, n0 + 3, -1])
                elems.append([n0 + 1, n0 + 4, n0 + 3, -1])
        elem_nodes = np.array(elems, dtype=np.int32)
        elem_nodes_tri = elem_nodes[:, :3]
        lon_c = np.array([lon[e].mean() for e in elem_nodes_tri], dtype=np.float64)
        lat_c = np.array([lat[e].mean() for e in elem_nodes_tri], dtype=np.float64)
        n2e, cnt = build_node_to_elems(len(lon), elem_nodes)
        mesh = Mesh(
            name="test", nodes_lon=lon, nodes_lat=lat,
            depth=np.full(len(lon), 10.0), elem_nodes=elem_nodes,
            elem_center_lon=lon_c, elem_center_lat=lat_c,
            node_to_elems=n2e, node_to_elems_count=cnt,
            sigma=np.array([], dtype=np.float64),
        )
        build_search(mesh)
        return mesh

    def test_from_grid(self):
        mesh = self._make_simple_mesh()
        state = ParticleState.from_grid(mesh, nx=3, ny=3)
        assert state.n_particles > 0
        assert np.all(state.lon >= 4.0)
        assert np.all(state.lon <= 6.0)

    def test_from_around_point(self):
        mesh = self._make_simple_mesh()
        state = ParticleState.from_around_point(mesh, 5.0, 5.0, n=50,
                                                 radius_km=50.0, seed=42)
        assert state.n_particles > 0
