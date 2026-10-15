"""Explicit grid-hash cell size (mesh.grid_hash_cell_size).

The auto cell (``sqrt(area/N)*2``, floored at 0.05 deg) assumes a
quasi-uniform mesh.  On a mesh whose resolution spans orders of
magnitude one cell can hold thousands of nodes, degrading the in-kernel
3x3 nearest scan to a linear search.  These tests pin the escape hatch:
config validation, the pin surviving rebuilds, precedence, the
pathological-occupancy warning, and query correctness afterwards.
"""
import numpy as np
import pytest

from kelp.core.config import MeshConfig
from kelp.core.mesh import (GRID_HASH_WARN_OCCUPANCY, Mesh,
                                build_grid_hash, build_node_to_elems,
                                build_search, find_nearest_nodes,
                                set_grid_hash_cell)


def _point_mesh(lon, lat):
    """Minimal Mesh carrying a node cloud (hash/nearest tests only)."""
    lon = np.asarray(lon, dtype=np.float64).ravel()
    lat = np.asarray(lat, dtype=np.float64).ravel()
    elem_nodes = np.array([[0, 1, 2]], dtype=np.int32)
    n2e, n2e_count = build_node_to_elems(lon.size, elem_nodes)
    mesh = Mesh(
        name="points", nodes_lon=lon, nodes_lat=lat,
        depth=np.zeros(lon.size), elem_nodes=elem_nodes,
        elem_center_lon=np.array([lon[:3].mean()]),
        elem_center_lat=np.array([lat[:3].mean()]),
        node_to_elems=n2e, node_to_elems_count=n2e_count,
    )
    return mesh


#: centre of the dense cluster — deliberately INSIDE one auto cell
#: (boundaries fall on 5.00/54.00 here), so the occupancy test does not
#: quietly weaken by having the cluster split across four cells.
DENSE_LON, DENSE_LAT, DENSE_HALF = 5.02, 54.02, 0.005


def _nonuniform_mesh(n_dense=6000):
    """Dense cluster (~1e-4 deg spacing) inside a coarse box.

    Mimics a structure-resolving coastal mesh: the auto cell hits its
    0.05-deg floor and swallows the whole cluster.
    """
    rng = np.random.default_rng(7)
    dense_lon = DENSE_LON + rng.uniform(-DENSE_HALF, DENSE_HALF, n_dense)
    dense_lat = DENSE_LAT + rng.uniform(-DENSE_HALF, DENSE_HALF, n_dense)
    coarse_lon, coarse_lat = np.meshgrid(np.linspace(4.5, 5.5, 12),
                                         np.linspace(53.5, 54.5, 12))
    return _point_mesh(np.concatenate([dense_lon, coarse_lon.ravel()]),
                       np.concatenate([dense_lat, coarse_lat.ravel()]))


def _nearest_bruteforce(mesh, lons, lats):
    """Oracle: argmin of 3-D cartesian distance on the unit sphere —
    the metric the grid-hash kernels themselves use."""
    def xyz(lo, la):
        lo_r, la_r = np.radians(lo), np.radians(la)
        return np.column_stack((np.cos(la_r) * np.cos(lo_r),
                                np.cos(la_r) * np.sin(lo_r),
                                np.sin(la_r)))
    q = xyz(np.asarray(lons), np.asarray(lats))
    n = xyz(mesh.nodes_lon, mesh.nodes_lat)
    d2 = ((q[:, None, :] - n[None, :, :]) ** 2).sum(axis=2)
    return d2.argmin(axis=1)


def _grid_hash_warnings(records):
    return [w for w in records
            if issubclass(w.category, RuntimeWarning)
            and "grid hash" in str(w.message)]


class TestConfigValidation:
    def test_default_is_none(self):
        assert MeshConfig().grid_hash_cell_size is None

    @pytest.mark.parametrize("bad", [0.0, -0.01, float("nan"),
                                     float("inf"), "abc"])
    def test_rejects_invalid(self, bad):
        with pytest.raises(ValueError, match="grid_hash_cell_size"):
            MeshConfig(grid_hash_cell_size=bad)

    def test_coerces_to_float(self):
        cfg = MeshConfig(grid_hash_cell_size="0.005")
        assert isinstance(cfg.grid_hash_cell_size, float)
        assert cfg.grid_hash_cell_size == pytest.approx(0.005)


class TestPinning:
    def test_pin_survives_rebuild(self):
        mesh = _nonuniform_mesh()
        set_grid_hash_cell(mesh, 0.002)
        assert mesh._gh_cell_size == pytest.approx(0.002)
        # Any later rebuild (e.g. particles_gen._filter_to_mesh) must
        # NOT revert to the auto size.
        build_search(mesh)
        assert mesh._gh_cell_size == pytest.approx(0.002)
        build_grid_hash(mesh)
        assert mesh._gh_cell_size == pytest.approx(0.002)

    def test_explicit_argument_wins_over_pin(self):
        mesh = _nonuniform_mesh()
        set_grid_hash_cell(mesh, 0.002)
        build_grid_hash(mesh, cell_size=0.01)
        assert mesh._gh_cell_size == pytest.approx(0.01)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
    def test_rejects_invalid_cell(self, bad):
        with pytest.raises(ValueError, match="cell_size"):
            set_grid_hash_cell(_nonuniform_mesh(), bad)

    def test_unpinned_mesh_keeps_auto_size(self):
        mesh = _nonuniform_mesh()
        with pytest.warns(RuntimeWarning):
            build_grid_hash(mesh)
        assert mesh._gh_cell_size == pytest.approx(0.05)  # the auto floor

    def test_density_certificate_invalidated(self):
        mesh = _nonuniform_mesh()
        mesh._gh_dense_elems = True          # stale cache sentinel
        set_grid_hash_cell(mesh, 0.002)
        assert mesh._gh_dense_elems is None


class TestOccupancyWarning:
    def test_warns_on_pathological_mesh(self):
        mesh = _nonuniform_mesh()
        with pytest.warns(RuntimeWarning, match="grid_hash_cell_size") as rec:
            build_grid_hash(mesh)
        msg = str(_grid_hash_warnings(rec)[0].message)
        assert "nodes in one cell" in msg
        # the suggestion must be smaller than the auto cell it replaces
        suggested = float(msg.split("try ~")[1].split(")")[0])
        assert 0 < suggested < mesh._gh_cell_size

    def test_silent_on_uniform_mesh(self, recwarn):
        lon, lat = np.meshgrid(np.linspace(0, 10, 60), np.linspace(0, 10, 60))
        build_grid_hash(_point_mesh(lon, lat))
        assert not _grid_hash_warnings(recwarn)

    def test_silent_when_cell_is_explicit(self, recwarn):
        # An explicit (or pinned) cell is the user's decision — even a
        # deliberately huge one must not nag.
        mesh = _nonuniform_mesh()
        build_grid_hash(mesh, cell_size=0.05)
        set_grid_hash_cell(mesh, 0.05)
        assert not _grid_hash_warnings(recwarn)

    def test_threshold_constant_is_sane(self):
        assert GRID_HASH_WARN_OCCUPANCY >= 100


class TestTrackingWiring:
    """`run_tracking` must apply the config key reader-agnostically."""

    @staticmethod
    def _write_fesomc(tmp_path):
        """Minimal synthetic FESOM-C file (11x11 nodes, uniform flow)."""
        import xarray as xr
        nx = 11
        gx, gy = np.meshgrid(np.linspace(0.0, 10.0, nx),
                             np.linspace(0.0, 10.0, nx))
        lon, lat = gx.ravel(), gy.ravel()
        elems = []
        for j in range(nx - 1):
            for i in range(nx - 1):
                n0 = j * nx + i
                elems.append([n0 + 1, n0 + 2, n0 + nx + 1, n0 + 1])
                elems.append([n0 + 2, n0 + nx + 2, n0 + nx + 1, n0 + 2])
        nv = np.array(elems, dtype=np.int32)
        tri0 = nv[:, :3] - 1
        n_t, n_e, n_n = 3, nv.shape[0], lon.size
        nc = tmp_path / "fesomc.nc"
        xr.Dataset({
            "lon": (["node"], lon), "lat": (["node"], lat),
            "depth": (["node"], np.full(n_n, 10.0)),
            "nv": (["nele", "four"], nv),
            "lon_elem": (["nele"], lon[tri0].mean(axis=1)),
            "lat_elem": (["nele"], lat[tri0].mean(axis=1)),
            "sigma_lev": (["sigma"], np.array([0.0, 0.5, 1.0])),
            "u2d": (["time", "nele"], np.full((n_t, n_e), 0.1)),
            "v2d": (["time", "nele"], np.zeros((n_t, n_e))),
            "eta": (["time", "node"], np.zeros((n_t, n_n))),
        }).to_netcdf(nc)
        return nc

    def _run(self, tmp_path, cell_line):
        from kelp.core.config import load_config
        from kelp import tracking

        nc = self._write_fesomc(tmp_path)
        seeds = tmp_path / "p.dat"
        seeds.write_text("5.0 5.0\n5.1 5.1\n")
        cfg = tmp_path / "c.yaml"
        cfg.write_text(f"""\
mode: {{Fdrift2D: true, time_dir: 1.0}}
numerics: {{dt: 100.0, integrator: euler, output_freq: 1}}
mesh:
  type: fesomc
  meshpath: {nc}
{cell_line}data:
  ncinput: {nc}
  inicloud: {seeds}
  output_path: {tmp_path / "out.nc"}
  time_averaging: instant
time: {{start_time_idx: 0, n_steps: 2}}
""")
        seen = {}
        from kelp.core import mesh as mesh_mod
        real = mesh_mod.set_grid_hash_cell

        def spy(mesh, cell_size):
            seen["cell"] = cell_size
            seen["mesh"] = mesh
            return real(mesh, cell_size)

        mesh_mod.set_grid_hash_cell = spy
        try:
            tracking.run_tracking(load_config(cfg))
        finally:
            mesh_mod.set_grid_hash_cell = real
        return seen

    def test_applied_when_set(self, tmp_path):
        seen = self._run(tmp_path, "  grid_hash_cell_size: 0.2\n")
        assert seen["cell"] == pytest.approx(0.2)
        assert seen["mesh"]._gh_cell_size == pytest.approx(0.2)
        assert seen["mesh"].metadata["grid_hash_cell"] == pytest.approx(0.2)

    def test_dormant_when_unset_with_the_legacy_rule(self, tmp_path):
        # no explicit cell + the legacy rule: nothing is pinned (pre-2026-09-22
        # behaviour); the default occupancy rule pins its own choice through
        # the same set_grid_hash_cell (tests/test_grid_hash_cell_rule.py)
        assert self._run(tmp_path, "  grid_hash_cell_rule: legacy\n") == {}

    def test_occupancy_rule_pins_at_most_the_legacy_cell(self, tmp_path):
        import xarray as xr
        from kelp.readers import get_reader
        from kelp.core.config import MeshConfig
        seen = self._run(tmp_path, "")
        legacy = get_reader("fesomc").load_mesh(
            MeshConfig(type="fesomc", meshpath=str(tmp_path / "fesomc.nc")))._gh_cell_size
        with xr.open_dataset(tmp_path / "out.nc") as ds:
            assert ds.attrs["grid_hash_cell_rule"] == "occupancy"
            assert 0 < float(ds.attrs["grid_hash_cell"]) <= legacy
            if seen:                          # the chooser pinned a smaller cell
                assert seen["cell"] < legacy
                assert seen["mesh"].metadata["grid_hash_cell"] == seen["cell"]
                assert float(ds.attrs["grid_hash_cell"]) == seen["cell"]
            else:                             # it kept the reader's cell (this uniform fixture)
                assert float(ds.attrs["grid_hash_cell"]) == legacy


class TestQueriesStayCorrect:
    def test_nearest_matches_bruteforce_after_pinning(self):
        mesh = _nonuniform_mesh()
        set_grid_hash_cell(mesh, 0.004)
        rng = np.random.default_rng(11)
        qlon = DENSE_LON + rng.uniform(-DENSE_HALF, DENSE_HALF, 200)
        qlat = DENSE_LAT + rng.uniform(-DENSE_HALF, DENSE_HALF, 200)
        assert np.array_equal(find_nearest_nodes(mesh, qlon, qlat),
                              _nearest_bruteforce(mesh, qlon, qlat))
