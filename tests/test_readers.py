"""Tests for kelp.readers — FESOM-C and FESOM2 readers.

Creates synthetic NetCDF and ASCII mesh data for testing.
"""
from __future__ import annotations


import numpy as np
import pytest
import xarray as xr


from kelp.readers import get_reader, extract_time_info


# ---------------------------------------------------------------------------
# Helper: create synthetic FESOM-C NetCDF file
# ---------------------------------------------------------------------------

def _make_fesomc_nc(path: Path, n_time=3, n_elem=2, n_node=4):
    """Create a minimal FESOM-C-style NetCDF file.

    4 nodes forming 2 triangles::

        0---1
        |\\  |
        | \\|
        2---3
    """
    lon = np.array([4.0, 5.0, 4.0, 5.0], dtype=np.float64)
    lat = np.array([5.0, 5.0, 4.0, 4.0], dtype=np.float64)
    depth = np.array([10.0, 12.0, 8.0, 15.0], dtype=np.float64)

    # nv: 1-based, shape (nele, 4). Triangles padded with node0+1.
    nv = np.array([
        [1, 2, 4, 1],   # tri: 0-1-3 (1-based)
        [1, 4, 3, 1],   # tri: 0-3-2 (1-based)
    ], dtype=np.int32)

    lon_elem = np.array([
        (4.0 + 5.0 + 5.0) / 3,
        (4.0 + 5.0 + 4.0) / 3,
    ], dtype=np.float64)
    lat_elem = np.array([
        (5.0 + 5.0 + 4.0) / 3,
        (5.0 + 4.0 + 4.0) / 3,
    ], dtype=np.float64)

    sigma_lev = np.array([0.0, 0.5, 1.0], dtype=np.float64)

    # Velocity: constant 1.0 m/s east per timestep
    u2d = np.ones((n_time, n_elem), dtype=np.float64) * 1.5
    v2d = np.ones((n_time, n_elem), dtype=np.float64) * 0.5
    eta = np.zeros((n_time, n_node), dtype=np.float64)

    ds = xr.Dataset(
        {
            "lon": (["node"], lon),
            "lat": (["node"], lat),
            "depth": (["node"], depth),
            "nv": (["nele", "four"], nv),
            "lon_elem": (["nele"], lon_elem),
            "lat_elem": (["nele"], lat_elem),
            "sigma_lev": (["sigma"], sigma_lev),
            "u2d": (["time", "nele"], u2d),
            "v2d": (["time", "nele"], v2d),
            "eta": (["time", "node"], eta),
        },
    )
    ds.to_netcdf(path)
    return ds


# ---------------------------------------------------------------------------
# Helper: create synthetic FESOM2 ASCII mesh + NetCDF velocity
# ---------------------------------------------------------------------------

def _make_fesom2_mesh(mesh_dir: Path, n_node=4, n_elem=2):
    """Create minimal FESOM2 ASCII mesh.

    Same 4-node, 2-triangle geometry as FESOM-C test.
    """
    mesh_dir.mkdir(parents=True, exist_ok=True)

    lon = [4.0, 5.0, 4.0, 5.0]
    lat = [5.0, 5.0, 4.0, 4.0]

    # nod2d.out: <n_nodes> then id lon lat flag
    with open(mesh_dir / "nod2d.out", "w") as f:
        f.write(f"{n_node}\n")
        for i in range(n_node):
            f.write(f"{i + 1} {lon[i]:.6f} {lat[i]:.6f} 0\n")

    # elem2d.out: <n_elements> then n1 n2 n3 (1-based)
    elems = [[1, 2, 4], [1, 4, 3]]
    with open(mesh_dir / "elem2d.out", "w") as f:
        f.write(f"{n_elem}\n")
        for e in elems:
            f.write(f"{e[0]} {e[1]} {e[2]}\n")

    # depth.out (optional)
    with open(mesh_dir / "depth.out", "w") as f:
        for d in [10.0, 12.0, 8.0, 15.0]:
            f.write(f"{d}\n")


def _make_fesom2_nc(path: Path, n_time=3, n_elem=2, n_node=4):
    """Create a minimal FESOM2-style velocity NetCDF file."""
    u = np.ones((n_time, n_elem), dtype=np.float64) * 2.0
    v = np.ones((n_time, n_elem), dtype=np.float64) * -0.5
    ssh = np.zeros((n_time, n_node), dtype=np.float64)

    ds = xr.Dataset(
        {
            "u": (["time", "nod2"], u),
            "v": (["time", "nod2"], v),
            "ssh": (["time", "nod2_node"], ssh),
        },
    )
    ds.to_netcdf(path)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fesomc_nc(tmp_path):
    """Create synthetic FESOM-C NetCDF and return its path."""
    p = tmp_path / "fesomc_test.nc"
    _make_fesomc_nc(p)
    return p


@pytest.fixture
def fesom2_dir(tmp_path):
    """Create synthetic FESOM2 mesh dir + velocity file; return (mesh_dir, vel_path)."""
    mesh_dir = tmp_path / "fesom2_mesh"
    _make_fesom2_mesh(mesh_dir)
    vel_path = tmp_path / "fesom2_vel.nc"
    _make_fesom2_nc(vel_path)
    return mesh_dir, vel_path


# ===================================================================
# Registry tests
# ===================================================================

class TestRegistry:
    def test_fesomc_registered(self):
        reader = get_reader("fesomc")
        assert hasattr(reader, "load_mesh")
        assert hasattr(reader, "open_dataset")
        assert hasattr(reader, "read_velocities")

    def test_fesom2_registered(self):
        reader = get_reader("fesom2")
        assert hasattr(reader, "load_mesh")
        assert hasattr(reader, "open_dataset")
        assert hasattr(reader, "read_velocities")

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="No reader registered"):
            get_reader("nonexistent_model")


# ===================================================================
# FESOM-C reader tests
# ===================================================================

class TestFesomCReader:
    def test_load_mesh(self, fesomc_nc):
        from types import SimpleNamespace
        cfg = SimpleNamespace(meshpath=str(fesomc_nc))
        reader = get_reader("fesomc")
        mesh = reader.load_mesh(cfg)

        assert mesh.name == "fesomc"
        assert mesh.n_nodes == 4
        assert mesh.n_elements == 2
        assert mesh._gh_offsets is not None  # search built

    def test_mesh_coordinates(self, fesomc_nc):
        from types import SimpleNamespace
        cfg = SimpleNamespace(meshpath=str(fesomc_nc))
        mesh = get_reader("fesomc").load_mesh(cfg)

        assert mesh.nodes_lon[0] == pytest.approx(4.0)
        assert mesh.nodes_lat[0] == pytest.approx(5.0)
        assert mesh.depth[0] == pytest.approx(10.0)

    def test_mesh_connectivity(self, fesomc_nc):
        from types import SimpleNamespace
        cfg = SimpleNamespace(meshpath=str(fesomc_nc))
        mesh = get_reader("fesomc").load_mesh(cfg)

        # Check that elements are valid triangles (3 unique nodes each)
        for e in range(mesh.n_elements):
            valid = mesh.elem_valid_nodes(e)
            assert len(valid) == 3

    def test_read_velocities_2d(self, fesomc_nc):
        from types import SimpleNamespace
        cfg = SimpleNamespace(meshpath=str(fesomc_nc))
        mode = SimpleNamespace(Fdrift3D=False, Fdrift2D=True, Vpuls=False)

        reader = get_reader("fesomc")
        mesh = reader.load_mesh(cfg)
        ds = reader.open_dataset(fesomc_nc)
        vel = reader.read_velocities(ds, 0, mesh, mode)
        ds.close()

        assert vel.u.shape == (2,)
        assert vel.v.shape == (2,)
        assert vel.eta.shape == (4,)
        np.testing.assert_allclose(vel.u, 1.5)
        np.testing.assert_allclose(vel.v, 0.5)

    def test_read_velocities_different_timestep(self, fesomc_nc):
        from types import SimpleNamespace
        cfg = SimpleNamespace(meshpath=str(fesomc_nc))
        mode = SimpleNamespace(Fdrift3D=False, Fdrift2D=True, Vpuls=False)

        reader = get_reader("fesomc")
        mesh = reader.load_mesh(cfg)
        ds = reader.open_dataset(fesomc_nc)
        vel_0 = reader.read_velocities(ds, 0, mesh, mode)
        vel_1 = reader.read_velocities(ds, 1, mesh, mode)
        ds.close()

        # Both timesteps have same constant velocity in our test data
        np.testing.assert_allclose(vel_0.u, vel_1.u)

    def test_missing_file_raises(self, tmp_path):
        from types import SimpleNamespace
        cfg = SimpleNamespace(meshpath=str(tmp_path / "missing.nc"))
        with pytest.raises(FileNotFoundError):
            get_reader("fesomc").load_mesh(cfg)


# ===================================================================
# FESOM-C reader: split files, custom varnames, 3D→2D reduction
# ===================================================================

def _make_fesomc_nc_custom_vars(path, u_name="uu", v_name="vv", eta_name="ssh",
                                 n_time=3, n_elem=2, n_node=4):
    """FESOM-C NetCDF with non-standard variable names."""
    lon = np.array([4.0, 5.0, 4.0, 5.0], dtype=np.float64)
    lat = np.array([5.0, 5.0, 4.0, 4.0], dtype=np.float64)
    depth = np.array([10.0, 12.0, 8.0, 15.0], dtype=np.float64)
    nv = np.array([[1, 2, 4, 1], [1, 4, 3, 1]], dtype=np.int32)
    lon_elem = np.array([(4+5+5)/3, (4+5+4)/3], dtype=np.float64)
    lat_elem = np.array([(5+5+4)/3, (5+4+4)/3], dtype=np.float64)
    sigma_lev = np.array([0.0, 0.5, 1.0], dtype=np.float64)

    u = np.ones((n_time, n_elem), dtype=np.float64) * 1.5
    v = np.ones((n_time, n_elem), dtype=np.float64) * 0.5
    eta = np.zeros((n_time, n_node), dtype=np.float64) + 0.1

    ds = xr.Dataset({
        "lon": (["node"], lon),
        "lat": (["node"], lat),
        "depth": (["node"], depth),
        "nv": (["nele", "four"], nv),
        "lon_elem": (["nele"], lon_elem),
        "lat_elem": (["nele"], lat_elem),
        "sigma_lev": (["sigma"], sigma_lev),
        u_name: (["time", "nele"], u),
        v_name: (["time", "nele"], v),
        eta_name: (["time", "node"], eta),
    })
    ds.to_netcdf(path)


def _make_fesomc_split_uv(u_path, v_path, n_time=3, n_elem=2, n_node=4):
    """Create separate u and v NetCDF files (no mesh vars in either)."""
    lon = np.array([4.0, 5.0, 4.0, 5.0], dtype=np.float64)
    lat = np.array([5.0, 5.0, 4.0, 4.0], dtype=np.float64)
    nv = np.array([[1, 2, 4, 1], [1, 4, 3, 1]], dtype=np.int32)
    lon_elem = np.array([(4+5+5)/3, (4+5+4)/3], dtype=np.float64)
    lat_elem = np.array([(5+5+4)/3, (5+4+4)/3], dtype=np.float64)
    sigma_lev = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    depth = np.array([10.0, 12.0, 8.0, 15.0], dtype=np.float64)

    # u file has mesh + u variable
    u = np.ones((n_time, n_elem), dtype=np.float64) * 2.0
    ds_u = xr.Dataset({
        "lon": (["node"], lon), "lat": (["node"], lat),
        "depth": (["node"], depth),
        "nv": (["nele", "four"], nv),
        "lon_elem": (["nele"], lon_elem), "lat_elem": (["nele"], lat_elem),
        "sigma_lev": (["sigma"], sigma_lev),
        "u": (["time", "nele"], u),
    })
    ds_u.to_netcdf(u_path)

    # v file has just v variable
    v = np.ones((n_time, n_elem), dtype=np.float64) * -0.3
    ds_v = xr.Dataset({
        "v": (["time", "nele"], v),
    })
    ds_v.to_netcdf(v_path)


def _make_fesomc_3d(path, n_time=2, n_elem=2, n_node=4, n_sigma=3):
    """FESOM-C NetCDF with 3D velocity (nele, sigmam1)."""
    lon = np.array([4.0, 5.0, 4.0, 5.0], dtype=np.float64)
    lat = np.array([5.0, 5.0, 4.0, 4.0], dtype=np.float64)
    depth = np.array([10.0, 12.0, 8.0, 15.0], dtype=np.float64)
    nv = np.array([[1, 2, 4, 1], [1, 4, 3, 1]], dtype=np.int32)
    lon_elem = np.array([(4+5+5)/3, (4+5+4)/3], dtype=np.float64)
    lat_elem = np.array([(5+5+4)/3, (5+4+4)/3], dtype=np.float64)
    sigma_lev = np.array([1.0, 0.5, 0.0], dtype=np.float64)
    n_layers = n_sigma - 1  # 2 layers

    # u shape (time, nele, sigmam1): each layer has distinct value
    u = np.zeros((n_time, n_elem, n_layers), dtype=np.float64)
    u[:, :, 0] = 1.0   # surface layer
    u[:, :, 1] = 3.0   # bottom layer
    v = np.zeros((n_time, n_elem, n_layers), dtype=np.float64)
    v[:, :, 0] = 0.5
    v[:, :, 1] = 1.5
    w_cart = np.zeros((n_time, n_node, n_sigma), dtype=np.float64)
    for k in range(n_sigma):
        w_cart[:, :, k] = 10.0 + k
    eta = np.zeros((n_time, n_node), dtype=np.float64)

    ds = xr.Dataset({
        "lon": (["node"], lon),
        "lat": (["node"], lat),
        "depth": (["node"], depth),
        "nv": (["nele", "four"], nv),
        "lon_elem": (["nele"], lon_elem),
        "lat_elem": (["nele"], lat_elem),
        "sigma_lev": (["sigma"], sigma_lev),
        "u": (["time", "nele", "sigmam1"], u),
        "v": (["time", "nele", "sigmam1"], v),
        "w": (["time", "node", "sigma"], w_cart),       # FESOM-C Wvel
        "eta": (["time", "node"], eta),
    })
    ds.to_netcdf(path)


class TestFesomCReaderExtended:
    """Tests for split files, custom variable names, and 3D→2D reduction."""

    def test_read_velocities_custom_varnames(self, tmp_path):
        from types import SimpleNamespace
        nc = tmp_path / "custom.nc"
        _make_fesomc_nc_custom_vars(nc, u_name="uu", v_name="vv", eta_name="ssh")

        reader = get_reader("fesomc")
        mesh = reader.load_mesh(SimpleNamespace(meshpath=str(nc)))
        ds = reader.open_dataset(str(nc))
        mode = SimpleNamespace(Fdrift3D=False, Fdrift2D=True, Vpuls=False,
                               var_u="uu", var_v="vv", var_w=None, var_eta="ssh",
                               sigma_layer=None, sigma_mean=False)
        vel = reader.read_velocities(ds, 0, mesh, mode)
        ds.close()

        np.testing.assert_allclose(vel.u, 1.5)
        np.testing.assert_allclose(vel.v, 0.5)
        np.testing.assert_allclose(vel.eta, 0.1)

    def test_load_mesh_sets_vertical_coord_sigma(self, tmp_path):
        from types import SimpleNamespace
        nc = tmp_path / "fesomc_3d.nc"
        _make_fesomc_3d(nc)

        reader = get_reader("fesomc")
        mesh = reader.load_mesh(SimpleNamespace(meshpath=str(nc)))

        assert mesh.vertical_coord == "sigma"
        assert mesh.metadata["sigma_transform"] == "linear_fesomc"

    def test_open_dataset_split_files(self, tmp_path):
        u_nc = tmp_path / "u_data.nc"
        v_nc = tmp_path / "v_data.nc"
        _make_fesomc_split_uv(u_nc, v_nc)

        reader = get_reader("fesomc")
        ds = reader.open_dataset(str(u_nc), v_path=str(v_nc))
        assert "u" in ds
        assert "v" in ds
        np.testing.assert_allclose(ds["u"].isel(time=0).values, 2.0)
        np.testing.assert_allclose(ds["v"].isel(time=0).values, -0.3)
        ds.close()

    def test_split_files_read_velocities(self, tmp_path):
        from types import SimpleNamespace
        u_nc = tmp_path / "u_data.nc"
        v_nc = tmp_path / "v_data.nc"
        _make_fesomc_split_uv(u_nc, v_nc)

        reader = get_reader("fesomc")
        mesh = reader.load_mesh(SimpleNamespace(meshpath=str(u_nc)))
        ds = reader.open_dataset(str(u_nc), v_path=str(v_nc))
        mode = SimpleNamespace(Fdrift3D=False, Fdrift2D=True, Vpuls=False,
                               var_u="u", var_v="v", var_w=None, var_eta=None,
                               sigma_layer=None, sigma_mean=False)
        vel = reader.read_velocities(ds, 0, mesh, mode)
        ds.close()

        np.testing.assert_allclose(vel.u, 2.0)
        np.testing.assert_allclose(vel.v, -0.3)
        # No eta in files → zeros
        np.testing.assert_allclose(vel.eta, 0.0)

    def test_split_files_missing_v_raises(self, tmp_path):
        u_nc = tmp_path / "u_data.nc"
        _make_fesomc_split_uv(u_nc, tmp_path / "v_data.nc")

        reader = get_reader("fesomc")
        with pytest.raises(FileNotFoundError, match="V file not found"):
            reader.open_dataset(str(u_nc), v_path=str(tmp_path / "missing_v.nc"))

    def test_open_dataset_split_files_with_w(self, tmp_path):
        u_nc = tmp_path / "u_data.nc"
        v_nc = tmp_path / "v_data.nc"
        w_nc = tmp_path / "w_data.nc"
        _make_fesomc_split_uv(u_nc, v_nc)

        xr.Dataset({
            "w_cart": (["time", "node", "sigma"],
                        np.ones((3, 4, 3), dtype=np.float64) * 0.25),
        }).to_netcdf(w_nc)

        reader = get_reader("fesomc")
        ds = reader.open_dataset(str(u_nc), v_path=str(v_nc), w_path=str(w_nc))
        assert "u" in ds
        assert "v" in ds
        assert "w_cart" in ds
        np.testing.assert_allclose(ds["w_cart"].isel(time=0).values, 0.25)
        ds.close()

    def test_read_velocities_3d_layer_select(self, tmp_path):
        from types import SimpleNamespace
        nc = tmp_path / "fesomc_3d.nc"
        _make_fesomc_3d(nc)

        reader = get_reader("fesomc")
        mesh = reader.load_mesh(SimpleNamespace(meshpath=str(nc)))
        ds = reader.open_dataset(str(nc))

        # Select layer 1 (bottom)
        mode = SimpleNamespace(Fdrift3D=False, Fdrift2D=True, Vpuls=False,
                               var_u="u", var_v="v", var_w=None, var_eta=None,
                               sigma_layer=1, sigma_mean=False)
        vel = reader.read_velocities(ds, 0, mesh, mode)
        ds.close()

        assert vel.u.shape == (2,)
        np.testing.assert_allclose(vel.u, 3.0)
        np.testing.assert_allclose(vel.v, 1.5)

    def test_read_velocities_3d_default_surface(self, tmp_path):
        from types import SimpleNamespace
        nc = tmp_path / "fesomc_3d.nc"
        _make_fesomc_3d(nc)

        reader = get_reader("fesomc")
        mesh = reader.load_mesh(SimpleNamespace(meshpath=str(nc)))
        ds = reader.open_dataset(str(nc))

        # No sigma_layer or sigma_mean → default to layer 0 (surface)
        mode = SimpleNamespace(Fdrift3D=False, Fdrift2D=True, Vpuls=False,
                               var_u="u", var_v="v", var_w=None, var_eta=None,
                               sigma_layer=None, sigma_mean=False)
        vel = reader.read_velocities(ds, 0, mesh, mode)
        ds.close()

        np.testing.assert_allclose(vel.u, 1.0)
        np.testing.assert_allclose(vel.v, 0.5)

    def test_read_velocities_3d_sigma_mean(self, tmp_path):
        from types import SimpleNamespace
        nc = tmp_path / "fesomc_3d.nc"
        _make_fesomc_3d(nc)

        reader = get_reader("fesomc")
        mesh = reader.load_mesh(SimpleNamespace(meshpath=str(nc)))
        ds = reader.open_dataset(str(nc))

        mode = SimpleNamespace(Fdrift3D=False, Fdrift2D=True, Vpuls=False,
                               var_u="u", var_v="v", var_w=None, var_eta=None,
                               sigma_layer=None, sigma_mean=True)
        vel = reader.read_velocities(ds, 0, mesh, mode)
        ds.close()

        # sigma = [0.0, 0.5, 1.0] → dz = [0.5, 0.5] → weights = [0.5, 0.5]
        # u: 0.5*1.0 + 0.5*3.0 = 2.0
        # v: 0.5*0.5 + 0.5*1.5 = 1.0
        assert vel.u.shape == (2,)
        np.testing.assert_allclose(vel.u, 2.0)
        np.testing.assert_allclose(vel.v, 1.0)

    def test_missing_eta_returns_zeros(self, tmp_path):
        from types import SimpleNamespace
        # Create a file with u and v but no eta
        nc = tmp_path / "no_eta.nc"
        lon = np.array([4.0, 5.0, 4.0, 5.0], dtype=np.float64)
        lat = np.array([5.0, 5.0, 4.0, 4.0], dtype=np.float64)
        depth = np.array([10.0, 12.0, 8.0, 15.0], dtype=np.float64)
        nv = np.array([[1, 2, 4, 1], [1, 4, 3, 1]], dtype=np.int32)
        lon_elem = np.array([(4+5+5)/3, (4+5+4)/3], dtype=np.float64)
        lat_elem = np.array([(5+5+4)/3, (5+4+4)/3], dtype=np.float64)
        sigma_lev = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        ds = xr.Dataset({
            "lon": (["node"], lon), "lat": (["node"], lat),
            "depth": (["node"], depth),
            "nv": (["nele", "four"], nv),
            "lon_elem": (["nele"], lon_elem), "lat_elem": (["nele"], lat_elem),
            "sigma_lev": (["sigma"], sigma_lev),
            "u": (["time", "nele"], np.ones((2, 2), dtype=np.float64)),
            "v": (["time", "nele"], np.ones((2, 2), dtype=np.float64) * 0.5),
        })
        ds.to_netcdf(nc)

        reader = get_reader("fesomc")
        mesh = reader.load_mesh(SimpleNamespace(meshpath=str(nc)))
        ds = reader.open_dataset(str(nc))
        mode = SimpleNamespace(Fdrift3D=False, Fdrift2D=True, Vpuls=False,
                               var_u="u", var_v="v", var_w=None, var_eta=None,
                               sigma_layer=None, sigma_mean=False)
        vel = reader.read_velocities(ds, 0, mesh, mode)
        ds.close()

        np.testing.assert_allclose(vel.eta, 0.0)

    def test_read_velocities_3d_uses_wvel_and_normalizes_axes(self, tmp_path):
        from types import SimpleNamespace
        nc = tmp_path / "fesomc_3d.nc"
        _make_fesomc_3d(nc)

        reader = get_reader("fesomc")
        mesh = reader.load_mesh(SimpleNamespace(meshpath=str(nc)))
        ds = reader.open_dataset(str(nc))
        mode = SimpleNamespace(
            Fdrift2D=False, Fdrift3D=True, Vpuls=False,
            var_u=None, var_v=None, var_w=None, var_eta=None,
            sigma_layer=None, sigma_mean=False, max_levels=None,
        )
        vel = reader.read_velocities(ds, 0, mesh, mode)
        ds.close()

        assert vel.u.shape == (2, 2)
        assert vel.v.shape == (2, 2)
        assert vel.w is not None
        assert vel.w.shape == (3, 4)
        np.testing.assert_allclose(vel.u[0], 1.0)
        np.testing.assert_allclose(vel.u[1], 3.0)
        # w = Wvel (positive-up flux through the interfaces: 10, 11, 12) ->
        # surface-relative depth rate (1-sigma)(eta_t + u.grad H) - Wvel:
        # the surface interface (sigma 1) is exactly -Wvel; the others add
        # the stretching term over the sloping bottom
        np.testing.assert_allclose(vel.w[0], -10.0)
        assert np.all(np.isfinite(vel.w))

    def test_read_velocities_3d_rejects_misstaggered_w(self, tmp_path):
        """A W variable stored on sigmam1 (layers) instead of sigma
        (interfaces) must be rejected, even if dim names look OK."""
        from types import SimpleNamespace
        nc = tmp_path / "fesomc_3d_bad_w.nc"
        _make_fesomc_3d(nc)

        ds_tmp = xr.open_dataset(nc)
        ds_fix = ds_tmp.load()
        ds_tmp.close()
        # Replace w (time, node, sigma=3) with wrong stagger (time, node, sigmam1=2)
        bad_w = xr.DataArray(
            np.zeros((2, 4, 2), dtype=np.float64),
            dims=["time", "node", "sigmam1"],   # wrong: sigmam1 for W
        )
        ds_fix = ds_fix.drop_vars("w")
        ds_fix["w"] = bad_w
        ds_fix.to_netcdf(nc, mode="w")

        reader = get_reader("fesomc")
        mesh = reader.load_mesh(SimpleNamespace(meshpath=str(nc)))
        ds = reader.open_dataset(str(nc))
        mode = SimpleNamespace(
            Fdrift2D=False, Fdrift3D=True, Vpuls=False,
            var_u=None, var_v=None, var_w=None, var_eta=None,
            sigma_layer=None, sigma_mean=False, max_levels=None,
        )
        with pytest.raises(ValueError, match="vertical stagger"):
            reader.read_velocities(ds, 0, mesh, mode)
        ds.close()

    def test_read_velocities_3d_missing_eta_warns_once(self, tmp_path):
        from types import SimpleNamespace
        import warnings

        nc = tmp_path / "fesomc_3d_no_eta.nc"
        _make_fesomc_3d(nc)
        ds_tmp = xr.open_dataset(nc)
        ds_no_eta = ds_tmp.drop_vars("eta").load()
        ds_tmp.close()
        ds_no_eta.to_netcdf(nc, mode="w")

        reader = get_reader("fesomc")
        mesh = reader.load_mesh(SimpleNamespace(meshpath=str(nc)))
        ds = reader.open_dataset(str(nc))
        mode = SimpleNamespace(
            Fdrift2D=False, Fdrift3D=True, Vpuls=False,
            var_u=None, var_v=None, var_w=None, var_eta=None,
            sigma_layer=None, sigma_mean=False, max_levels=None,
        )

        with pytest.warns(RuntimeWarning, match="eta is missing"):
            vel = reader.read_velocities(ds, 0, mesh, mode)
        np.testing.assert_allclose(vel.eta, 0.0)

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            _ = reader.read_velocities(ds, 1, mesh, mode)
        assert rec == []
        ds.close()

    def test_read_velocities_3d_max_levels_uses_interface_count(self, tmp_path):
        from types import SimpleNamespace
        nc = tmp_path / "fesomc_3d.nc"
        _make_fesomc_3d(nc)

        reader = get_reader("fesomc")
        mesh = reader.load_mesh(SimpleNamespace(meshpath=str(nc)))
        ds = reader.open_dataset(str(nc))
        mode = SimpleNamespace(
            Fdrift2D=False, Fdrift3D=True, Vpuls=False,
            var_u=None, var_v=None, var_w=None, var_eta=None,
            sigma_layer=None, sigma_mean=False, max_levels=2,
        )
        vel = reader.read_velocities(ds, 0, mesh, mode)
        ds.close()

        assert vel.u.shape == (1, 2)
        assert vel.v.shape == (1, 2)
        assert vel.w.shape == (2, 4)
        np.testing.assert_allclose(vel.u[0], 1.0)
        np.testing.assert_allclose(vel.w[0], -10.0)      # surface: -Wvel
        assert np.all(np.isfinite(vel.w[1]))

    def test_read_velocities_3d_max_levels_with_trimmed_mesh(self, tmp_path):
        """Blocker regression: tracking.py trims mesh.sigma before calling
        read_velocities.  The reader must normalize u/v/w against the
        full sigma count from the file, not the trimmed mesh.sigma."""
        from types import SimpleNamespace
        nc = tmp_path / "fesomc_3d.nc"
        _make_fesomc_3d(nc)  # sigma_lev has 3 interfaces, 2 layers

        reader = get_reader("fesomc")
        mesh = reader.load_mesh(SimpleNamespace(meshpath=str(nc)))

        # Simulate tracking.py trimming mesh.sigma before reader call
        assert mesh.sigma.shape[0] == 3  # full
        mesh.sigma = mesh.sigma[:2]       # trimmed to 2 interfaces

        ds = reader.open_dataset(str(nc))
        mode = SimpleNamespace(
            Fdrift2D=False, Fdrift3D=True, Vpuls=False,
            var_u=None, var_v=None, var_w=None, var_eta=None,
            sigma_layer=None, sigma_mean=False, max_levels=2,
        )
        # Before the fix this would raise a shape mismatch because
        # normalize was called with the trimmed count (2) against a
        # full-file array with 3 interfaces / 2 layers.
        vel = reader.read_velocities(ds, 0, mesh, mode)
        ds.close()

        assert vel.u.shape == (1, 2)
        assert vel.w.shape == (2, 4)

    def test_read_velocities_3d_without_sigma_lev_uses_dim_names(self, tmp_path):
        """Review CRITICAL regression: when sigma_lev is absent (split files),
        the reader must locate the vertical axis by named dim, NOT by
        max(w_da.shape). The default fixture has n_node=4 and n_sigma=3,
        so max(4,3)=4 would mis-validate against the node count.
        In real datasets n_node (~135k) is far larger than n_sigma (~10),
        making the bug catastrophic."""
        from types import SimpleNamespace
        nc = tmp_path / "fesomc_3d_with_siglev.nc"
        _make_fesomc_3d(nc)

        ds_tmp = xr.open_dataset(nc)
        ds_fix = ds_tmp.load()
        ds_tmp.close()
        # Drop sigma_lev from the velocity dataset (simulates split files
        # where sigma_lev lives in the mesh file only).
        ds_fix = ds_fix.drop_vars("sigma_lev")
        nc_no_sl = tmp_path / "fesomc_3d_no_siglev.nc"
        ds_fix.to_netcdf(nc_no_sl)

        reader = get_reader("fesomc")
        mesh = reader.load_mesh(SimpleNamespace(meshpath=str(nc)))
        ds = reader.open_dataset(str(nc_no_sl))
        mode = SimpleNamespace(
            Fdrift2D=False, Fdrift3D=True, Vpuls=False,
            var_u=None, var_v=None, var_w=None, var_eta=None,
            sigma_layer=None, sigma_mean=False, max_levels=None,
        )
        # Before the fix, max(w_da.shape) = max(4, 3) = 4 → tries to
        # validate against n_sigma_full=4 instead of 3 → ValueError.
        vel = reader.read_velocities(ds, 0, mesh, mode)
        ds.close()

        assert vel.w.shape == (3, 4)   # (n_sigma, n_nodes)
        assert vel.u.shape == (2, 2)   # (n_layers, n_elements)

    def test_read_velocities_3d_max_levels_still_rejects_misstagger(self, tmp_path):
        """max_levels must not mask a mis-staggered field — validation
        against the FULL sigma count catches e.g. w on sigmam1."""
        from types import SimpleNamespace
        nc = tmp_path / "fesomc_3d_bad_w2.nc"
        _make_fesomc_3d(nc)

        ds_tmp = xr.open_dataset(nc)
        ds_fix = ds_tmp.load()
        ds_tmp.close()
        # Replace w (time,node,sigma=3) with wrong stagger (time,node,sigmam1=2)
        bad_w = xr.DataArray(
            np.zeros((2, 4, 2), dtype=np.float64),
            dims=["time", "node", "sigmam1"],
        )
        ds_fix = ds_fix.drop_vars("w")
        ds_fix["w"] = bad_w
        ds_fix.to_netcdf(nc, mode="w")

        reader = get_reader("fesomc")
        mesh = reader.load_mesh(SimpleNamespace(meshpath=str(nc)))
        mesh.sigma = mesh.sigma[:2]  # trim like tracking.py would

        ds = reader.open_dataset(str(nc))
        mode = SimpleNamespace(
            Fdrift2D=False, Fdrift3D=True, Vpuls=False,
            var_u=None, var_v=None, var_w=None, var_eta=None,
            sigma_layer=None, sigma_mean=False, max_levels=2,
        )
        with pytest.raises(ValueError, match="vertical stagger"):
            reader.read_velocities(ds, 0, mesh, mode)
        ds.close()


# ===================================================================
# FESOM2 reader tests
# ===================================================================

class TestFesom2Reader:
    def test_load_mesh(self, fesom2_dir):
        from types import SimpleNamespace
        mesh_dir, _ = fesom2_dir
        cfg = SimpleNamespace(meshpath=str(mesh_dir))
        reader = get_reader("fesom2")
        mesh = reader.load_mesh(cfg)

        assert mesh.name == "fesom2"
        assert mesh.n_nodes == 4
        assert mesh.n_elements == 2
        assert mesh.vertical_coord == "zlevel"
        assert mesh._gh_offsets is not None

    def test_mesh_coordinates(self, fesom2_dir):
        from types import SimpleNamespace
        mesh_dir, _ = fesom2_dir
        cfg = SimpleNamespace(meshpath=str(mesh_dir))
        mesh = get_reader("fesom2").load_mesh(cfg)

        assert mesh.nodes_lon[0] == pytest.approx(4.0)
        assert mesh.nodes_lat[0] == pytest.approx(5.0)
        assert mesh.depth[0] == pytest.approx(10.0)

    def test_mesh_triangles(self, fesom2_dir):
        from types import SimpleNamespace
        mesh_dir, _ = fesom2_dir
        cfg = SimpleNamespace(meshpath=str(mesh_dir))
        mesh = get_reader("fesom2").load_mesh(cfg)

        # All FESOM2 elements are triangles (max 3 verts)
        assert mesh.elem_nodes.shape[1] == 3
        for e in range(mesh.n_elements):
            assert mesh.elem_is_triangle(e)

    def test_elem_centres(self, fesom2_dir):
        from types import SimpleNamespace
        mesh_dir, _ = fesom2_dir
        cfg = SimpleNamespace(meshpath=str(mesh_dir))
        mesh = get_reader("fesom2").load_mesh(cfg)

        # Element 0 is nodes 0,1,3 → (4,5,5), (5,5,4) → centre ~ (4.67, 4.67)
        assert mesh.elem_center_lon[0] == pytest.approx(4.667, abs=0.01)
        assert mesh.elem_center_lat[0] == pytest.approx(4.667, abs=0.01)

    def test_open_dataset_single_file(self, fesom2_dir):
        _, vel_path = fesom2_dir
        reader = get_reader("fesom2")
        ds = reader.open_dataset(str(vel_path))
        assert "u" in ds
        assert "v" in ds
        ds.close()

    def test_read_velocities(self, fesom2_dir):
        from types import SimpleNamespace
        mesh_dir, vel_path = fesom2_dir
        cfg = SimpleNamespace(meshpath=str(mesh_dir))
        reader = get_reader("fesom2")
        mesh = reader.load_mesh(cfg)
        ds = reader.open_dataset(str(vel_path))
        vel = reader.read_velocities(ds, 0, mesh)
        ds.close()

        assert vel.u.shape == (2,)
        assert vel.v.shape == (2,)
        assert vel.eta.shape == (4,)
        np.testing.assert_allclose(vel.u, 2.0)
        np.testing.assert_allclose(vel.v, -0.5)

    def test_missing_dir_raises(self, tmp_path):
        from types import SimpleNamespace
        cfg = SimpleNamespace(meshpath=str(tmp_path / "missing_dir"))
        with pytest.raises(FileNotFoundError):
            get_reader("fesom2").load_mesh(cfg)

    def test_node_to_elems(self, fesom2_dir):
        """Node-to-element adjacency should be populated."""
        from types import SimpleNamespace
        mesh_dir, _ = fesom2_dir
        cfg = SimpleNamespace(meshpath=str(mesh_dir))
        mesh = get_reader("fesom2").load_mesh(cfg)

        # Node 0 (top-left) appears in both triangles
        count_0 = mesh.node_to_elems_count[0]
        assert count_0 == 2

    def test_read_velocities_3d(self, fesom2_dir, tmp_path):
        """3D mode: read_velocities returns W and full-depth u/v arrays."""
        from types import SimpleNamespace

        mesh_dir, _ = fesom2_dir
        cfg = SimpleNamespace(meshpath=str(mesh_dir))
        reader = get_reader("fesom2")
        mesh = reader.load_mesh(cfg)

        # Create 3D velocity file: u(time, nz1, elem), v(time, nz1, elem), w(time, nz, nod2)
        n_time, n_elem, n_node = 2, 2, 4
        n_z_levels = 5   # 5 level interfaces
        n_mid = n_z_levels - 1  # 4 mid-layers for u/v

        u = np.ones((n_time, n_mid, n_elem), dtype=np.float64) * 1.5
        v = np.ones((n_time, n_mid, n_elem), dtype=np.float64) * -0.3
        w = np.ones((n_time, n_z_levels, n_node), dtype=np.float64) * 0.01
        ssh = np.zeros((n_time, n_node), dtype=np.float64)

        ds = xr.Dataset({
            "u": (["time", "nz1", "elem"], u),
            "v": (["time", "nz1", "elem"], v),
            "w": (["time", "nz", "nod2"], w),
            "ssh": (["time", "nod2_node"], ssh),
        })
        vel_path = tmp_path / "fesom2_3d_vel.nc"
        ds.to_netcdf(vel_path)

        ds_open = reader.open_dataset(str(vel_path))
        mode_3d = SimpleNamespace(
            Fdrift2D=False, Fdrift3D=True, time_dir=1.0,
            sigma_layer=None, sigma_mean=False, var_u=None,
            var_v=None, var_w=None, var_eta=None, Vpuls=False,
        )
        vel = reader.read_velocities(ds_open, 0, mesh, mode_3d)
        ds_open.close()

        # u/v should keep depth dimension: (n_mid, n_elem)
        assert vel.u.shape == (n_mid, n_elem)
        assert vel.v.shape == (n_mid, n_elem)
        np.testing.assert_allclose(vel.u, 1.5)
        np.testing.assert_allclose(vel.v, -0.3)

        # W should be present: (n_z_levels, n_node); FESOM2 w is positive
        # UP -> the internal positive-down rate is negated
        assert vel.w is not None
        assert vel.w.shape == (n_z_levels, n_node)
        np.testing.assert_allclose(vel.w, -0.01)

    def test_read_velocities_3d_max_levels_uses_interface_count(self, fesom2_dir, tmp_path):
        """max_levels in 3D mode means the number of interfaces to keep."""
        from types import SimpleNamespace

        mesh_dir, _ = fesom2_dir
        cfg = SimpleNamespace(meshpath=str(mesh_dir))
        reader = get_reader("fesom2")
        mesh = reader.load_mesh(cfg)

        n_time, n_elem, n_node = 2, 2, 4
        n_z_levels = 5
        n_mid = n_z_levels - 1

        u = np.ones((n_time, n_mid, n_elem), dtype=np.float64)
        v = np.ones((n_time, n_mid, n_elem), dtype=np.float64) * 2.0
        w = np.ones((n_time, n_z_levels, n_node), dtype=np.float64) * 3.0
        ssh = np.zeros((n_time, n_node), dtype=np.float64)

        ds = xr.Dataset({
            "u": (["time", "nz1", "elem"], u),
            "v": (["time", "nz1", "elem"], v),
            "w": (["time", "nz", "nod2"], w),
            "ssh": (["time", "nod2_node"], ssh),
        })
        vel_path = tmp_path / "fesom2_3d_max_levels.nc"
        ds.to_netcdf(vel_path)

        ds_open = reader.open_dataset(str(vel_path))
        mode_3d = SimpleNamespace(
            Fdrift2D=False, Fdrift3D=True, time_dir=1.0,
            sigma_layer=None, sigma_mean=False, var_u=None,
            var_v=None, var_w=None, var_eta=None, Vpuls=False,
            max_levels=3,
        )
        vel = reader.read_velocities(ds_open, 0, mesh, mode_3d)
        ds_open.close()

        assert vel.u.shape == (2, n_elem)
        assert vel.v.shape == (2, n_elem)
        assert vel.w.shape == (3, n_node)
        np.testing.assert_allclose(vel.u, 1.0)
        np.testing.assert_allclose(vel.v, 2.0)
        np.testing.assert_allclose(vel.w, -3.0)     # positive-up file


# ===================================================================
# Diffusivity (Kh) reader exposure
# ===================================================================

def _make_fesomc_with_kh(path, layout, n_time=2, n_elem=2, n_node=4,
                         var_x="Kh_zonal", var_y="Kh_meridional"):
    """FESOM-C-style NetCDF that also carries Kh on nodes or elements."""
    lon = np.array([4.0, 5.0, 4.0, 5.0], dtype=np.float64)
    lat = np.array([5.0, 5.0, 4.0, 4.0], dtype=np.float64)
    depth = np.array([10.0, 12.0, 8.0, 15.0], dtype=np.float64)
    nv = np.array(
        [[1, 2, 4, 1], [1, 4, 3, 1]], dtype=np.int32
    )
    lon_elem = np.array([(4.0 + 5.0 + 5.0) / 3,
                         (4.0 + 5.0 + 4.0) / 3], dtype=np.float64)
    lat_elem = np.array([(5.0 + 5.0 + 4.0) / 3,
                         (5.0 + 4.0 + 4.0) / 3], dtype=np.float64)
    sigma_lev = np.array([0.0, 0.5, 1.0], dtype=np.float64)

    u2d = np.ones((n_time, n_elem), dtype=np.float64) * 1.5
    v2d = np.ones((n_time, n_elem), dtype=np.float64) * 0.5
    eta = np.zeros((n_time, n_node), dtype=np.float64)

    if layout == "node":
        kh_dim = "node"
        kh_x = np.full(n_node, 7.0, dtype=np.float64)
        kh_y = np.full(n_node, 9.0, dtype=np.float64)
    elif layout == "element":
        kh_dim = "nele"
        kh_x = np.full(n_elem, 7.0, dtype=np.float64)
        kh_y = np.full(n_elem, 9.0, dtype=np.float64)
    else:
        raise ValueError(f"unknown layout {layout!r}")

    ds = xr.Dataset(
        {
            "lon": (["node"], lon),
            "lat": (["node"], lat),
            "depth": (["node"], depth),
            "nv": (["nele", "four"], nv),
            "lon_elem": (["nele"], lon_elem),
            "lat_elem": (["nele"], lat_elem),
            "sigma_lev": (["sigma"], sigma_lev),
            "u2d": (["time", "nele"], u2d),
            "v2d": (["time", "nele"], v2d),
            "eta": (["time", "node"], eta),
            var_x: ([kh_dim], kh_x),
            var_y: ([kh_dim], kh_y),
        },
    )
    ds.to_netcdf(path)


def _diff_cfg(enabled=True, scheme="em",
              var_kh_zonal=None, var_kh_meridional=None):
    """Build a DiffusionConfig the readers can consume."""
    from kelp.core.config import DiffusionConfig
    return DiffusionConfig(
        enabled=enabled,
        scheme=scheme,
        var_kh_zonal=var_kh_zonal,
        var_kh_meridional=var_kh_meridional,
    )


class TestFesomCKhReader:
    """FESOM-C reader exposes Kh as-is when DiffusionConfig is enabled."""

    def test_no_kh_when_disabled(self, tmp_path):
        from types import SimpleNamespace
        p = tmp_path / "fesomc_kh.nc"
        _make_fesomc_with_kh(p, layout="node")
        cfg = SimpleNamespace(meshpath=str(p))
        mode = SimpleNamespace(Fdrift3D=False, Fdrift2D=True, Vpuls=False)

        reader = get_reader("fesomc")
        mesh = reader.load_mesh(cfg)
        ds = reader.open_dataset(p)
        try:
            vel = reader.read_velocities(ds, 0, mesh, mode,
                                         diffusion=_diff_cfg(enabled=False))
            assert vel.kh_zonal is None
            assert vel.kh_layout is None
        finally:
            ds.close()

    def test_no_kh_when_diffusion_none(self, tmp_path):
        """No diffusion arg → no Kh, even if dataset has the variables."""
        from types import SimpleNamespace
        p = tmp_path / "fesomc_kh.nc"
        _make_fesomc_with_kh(p, layout="node")
        cfg = SimpleNamespace(meshpath=str(p))
        mode = SimpleNamespace(Fdrift3D=False, Fdrift2D=True, Vpuls=False)

        reader = get_reader("fesomc")
        mesh = reader.load_mesh(cfg)
        ds = reader.open_dataset(p)
        try:
            vel = reader.read_velocities(ds, 0, mesh, mode)
            assert vel.kh_zonal is None
        finally:
            ds.close()

    def test_kh_node_layout(self, tmp_path):
        from types import SimpleNamespace
        p = tmp_path / "fesomc_kh.nc"
        _make_fesomc_with_kh(p, layout="node")
        cfg = SimpleNamespace(meshpath=str(p))
        mode = SimpleNamespace(Fdrift3D=False, Fdrift2D=True, Vpuls=False)

        reader = get_reader("fesomc")
        mesh = reader.load_mesh(cfg)
        ds = reader.open_dataset(p)
        try:
            vel = reader.read_velocities(ds, 0, mesh, mode,
                                         diffusion=_diff_cfg())
        finally:
            ds.close()
        assert vel.kh_layout == "node"
        assert vel.kh_zonal.shape == (4,)
        np.testing.assert_allclose(vel.kh_zonal, 7.0)
        np.testing.assert_allclose(vel.kh_meridional, 9.0)

    def test_kh_element_layout(self, tmp_path):
        from types import SimpleNamespace
        p = tmp_path / "fesomc_kh.nc"
        _make_fesomc_with_kh(p, layout="element")
        cfg = SimpleNamespace(meshpath=str(p))
        mode = SimpleNamespace(Fdrift3D=False, Fdrift2D=True, Vpuls=False)

        reader = get_reader("fesomc")
        mesh = reader.load_mesh(cfg)
        ds = reader.open_dataset(p)
        try:
            vel = reader.read_velocities(ds, 0, mesh, mode,
                                         diffusion=_diff_cfg())
        finally:
            ds.close()
        assert vel.kh_layout == "element"
        assert vel.kh_zonal.shape == (2,)

    def test_custom_var_names(self, tmp_path):
        from types import SimpleNamespace
        p = tmp_path / "fesomc_kh.nc"
        _make_fesomc_with_kh(p, layout="node",
                             var_x="diffH_x", var_y="diffH_y")
        cfg = SimpleNamespace(meshpath=str(p))
        mode = SimpleNamespace(Fdrift3D=False, Fdrift2D=True, Vpuls=False)

        reader = get_reader("fesomc")
        mesh = reader.load_mesh(cfg)
        ds = reader.open_dataset(p)
        try:
            vel = reader.read_velocities(
                ds, 0, mesh, mode,
                diffusion=_diff_cfg(var_kh_zonal="diffH_x",
                                    var_kh_meridional="diffH_y"),
            )
        finally:
            ds.close()
        assert vel.kh_layout == "node"
        np.testing.assert_allclose(vel.kh_zonal, 7.0)

    def test_explicit_var_missing_raises(self, tmp_path):
        from types import SimpleNamespace
        p = tmp_path / "fesomc_kh.nc"
        _make_fesomc_with_kh(p, layout="node")
        cfg = SimpleNamespace(meshpath=str(p))
        mode = SimpleNamespace(Fdrift3D=False, Fdrift2D=True, Vpuls=False)

        reader = get_reader("fesomc")
        mesh = reader.load_mesh(cfg)
        ds = reader.open_dataset(p)
        try:
            with pytest.raises(KeyError, match="absent_var"):
                reader.read_velocities(
                    ds, 0, mesh, mode,
                    diffusion=_diff_cfg(var_kh_zonal="absent_var",
                                        var_kh_meridional="Kh_meridional"),
                )
        finally:
            ds.close()

    def test_layout_mismatch_raises(self, tmp_path):
        """Kh_zonal on nodes, Kh_meridional on elements → reader raises."""
        from types import SimpleNamespace
        p = tmp_path / "fesomc_mixed.nc"

        lon = np.array([4.0, 5.0, 4.0, 5.0], dtype=np.float64)
        lat = np.array([5.0, 5.0, 4.0, 4.0], dtype=np.float64)
        depth = np.array([10.0, 12.0, 8.0, 15.0], dtype=np.float64)
        nv = np.array([[1, 2, 4, 1], [1, 4, 3, 1]], dtype=np.int32)
        lon_elem = np.array([4.667, 4.333], dtype=np.float64)
        lat_elem = np.array([4.667, 4.333], dtype=np.float64)
        sigma_lev = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        u2d = np.ones((2, 2), dtype=np.float64) * 1.5
        v2d = np.ones((2, 2), dtype=np.float64) * 0.5
        eta = np.zeros((2, 4), dtype=np.float64)
        kh_x_node = np.full(4, 7.0, dtype=np.float64)
        kh_y_elem = np.full(2, 9.0, dtype=np.float64)

        ds_in = xr.Dataset({
            "lon": (["node"], lon),
            "lat": (["node"], lat),
            "depth": (["node"], depth),
            "nv": (["nele", "four"], nv),
            "lon_elem": (["nele"], lon_elem),
            "lat_elem": (["nele"], lat_elem),
            "sigma_lev": (["sigma"], sigma_lev),
            "u2d": (["time", "nele"], u2d),
            "v2d": (["time", "nele"], v2d),
            "eta": (["time", "node"], eta),
            "Kh_zonal": (["node"], kh_x_node),
            "Kh_meridional": (["nele"], kh_y_elem),
        })
        ds_in.to_netcdf(p)

        cfg = SimpleNamespace(meshpath=str(p))
        mode = SimpleNamespace(Fdrift3D=False, Fdrift2D=True, Vpuls=False)
        reader = get_reader("fesomc")
        mesh = reader.load_mesh(cfg)
        ds = reader.open_dataset(p)
        try:
            with pytest.raises(ValueError, match="layout mismatch"):
                reader.read_velocities(ds, 0, mesh, mode,
                                       diffusion=_diff_cfg())
        finally:
            ds.close()

    def test_negative_kh_raises(self, tmp_path):
        """Negative diffusivity is unphysical and rejected at read time."""
        from types import SimpleNamespace
        p = tmp_path / "fesomc_negkh.nc"

        lon = np.array([4.0, 5.0, 4.0, 5.0], dtype=np.float64)
        lat = np.array([5.0, 5.0, 4.0, 4.0], dtype=np.float64)
        depth = np.array([10.0, 12.0, 8.0, 15.0], dtype=np.float64)
        nv = np.array([[1, 2, 4, 1], [1, 4, 3, 1]], dtype=np.int32)
        lon_elem = np.array([4.667, 4.333], dtype=np.float64)
        lat_elem = np.array([4.667, 4.333], dtype=np.float64)
        sigma_lev = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        u2d = np.ones((2, 2), dtype=np.float64) * 1.5
        v2d = np.ones((2, 2), dtype=np.float64) * 0.5
        eta = np.zeros((2, 4), dtype=np.float64)
        kh_x = np.array([-1.0, 7.0, 7.0, 7.0], dtype=np.float64)
        kh_y = np.full(4, 9.0, dtype=np.float64)

        xr.Dataset({
            "lon": (["node"], lon),
            "lat": (["node"], lat),
            "depth": (["node"], depth),
            "nv": (["nele", "four"], nv),
            "lon_elem": (["nele"], lon_elem),
            "lat_elem": (["nele"], lat_elem),
            "sigma_lev": (["sigma"], sigma_lev),
            "u2d": (["time", "nele"], u2d),
            "v2d": (["time", "nele"], v2d),
            "eta": (["time", "node"], eta),
            "Kh_zonal": (["node"], kh_x),
            "Kh_meridional": (["node"], kh_y),
        }).to_netcdf(p)

        cfg = SimpleNamespace(meshpath=str(p))
        mode = SimpleNamespace(Fdrift3D=False, Fdrift2D=True, Vpuls=False)
        reader = get_reader("fesomc")
        mesh = reader.load_mesh(cfg)
        ds = reader.open_dataset(p)
        try:
            with pytest.raises(ValueError, match="non-negative"):
                reader.read_velocities(ds, 0, mesh, mode,
                                       diffusion=_diff_cfg())
        finally:
            ds.close()

    def test_nan_kh_raises(self, tmp_path):
        """NaN in Kh field is rejected before it can poison the SDE step."""
        from types import SimpleNamespace
        p = tmp_path / "fesomc_nan.nc"
        lon = np.array([4.0, 5.0, 4.0, 5.0], dtype=np.float64)
        lat = np.array([5.0, 5.0, 4.0, 4.0], dtype=np.float64)
        depth = np.array([10.0, 12.0, 8.0, 15.0], dtype=np.float64)
        nv = np.array([[1, 2, 4, 1], [1, 4, 3, 1]], dtype=np.int32)
        lon_elem = np.array([4.667, 4.333], dtype=np.float64)
        lat_elem = np.array([4.667, 4.333], dtype=np.float64)
        sigma_lev = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        u2d = np.ones((2, 2), dtype=np.float64) * 1.5
        v2d = np.ones((2, 2), dtype=np.float64) * 0.5
        eta = np.zeros((2, 4), dtype=np.float64)
        kh_x = np.array([np.nan, 7.0, 7.0, 7.0], dtype=np.float64)
        kh_y = np.full(4, 9.0, dtype=np.float64)

        xr.Dataset({
            "lon": (["node"], lon),
            "lat": (["node"], lat),
            "depth": (["node"], depth),
            "nv": (["nele", "four"], nv),
            "lon_elem": (["nele"], lon_elem),
            "lat_elem": (["nele"], lat_elem),
            "sigma_lev": (["sigma"], sigma_lev),
            "u2d": (["time", "nele"], u2d),
            "v2d": (["time", "nele"], v2d),
            "eta": (["time", "node"], eta),
            "Kh_zonal": (["node"], kh_x),
            "Kh_meridional": (["node"], kh_y),
        }).to_netcdf(p)

        cfg = SimpleNamespace(meshpath=str(p))
        mode = SimpleNamespace(Fdrift3D=False, Fdrift2D=True, Vpuls=False)
        reader = get_reader("fesomc")
        mesh = reader.load_mesh(cfg)
        ds = reader.open_dataset(p)
        try:
            with pytest.raises(ValueError, match="finite"):
                reader.read_velocities(ds, 0, mesh, mode,
                                       diffusion=_diff_cfg())
        finally:
            ds.close()

    def test_extra_dim_raises(self, tmp_path):
        """A 2-D Kh array (e.g. ``(nz, node)``) is rejected even if dim
        names trip the layout detector — the kernels assume 1-D."""
        from types import SimpleNamespace
        p = tmp_path / "fesomc_extra_dim.nc"
        lon = np.array([4.0, 5.0, 4.0, 5.0], dtype=np.float64)
        lat = np.array([5.0, 5.0, 4.0, 4.0], dtype=np.float64)
        depth = np.array([10.0, 12.0, 8.0, 15.0], dtype=np.float64)
        nv = np.array([[1, 2, 4, 1], [1, 4, 3, 1]], dtype=np.int32)
        lon_elem = np.array([4.667, 4.333], dtype=np.float64)
        lat_elem = np.array([4.667, 4.333], dtype=np.float64)
        sigma_lev = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        u2d = np.ones((2, 2), dtype=np.float64) * 1.5
        v2d = np.ones((2, 2), dtype=np.float64) * 0.5
        eta = np.zeros((2, 4), dtype=np.float64)
        # 2-D Kh: 3 layers × 4 nodes — passes dim-name layout detection
        # because "node" still appears in dims, but the kernels need 1-D.
        kh_x_3d = np.full((3, 4), 7.0, dtype=np.float64)
        kh_y_3d = np.full((3, 4), 9.0, dtype=np.float64)

        xr.Dataset({
            "lon": (["node"], lon),
            "lat": (["node"], lat),
            "depth": (["node"], depth),
            "nv": (["nele", "four"], nv),
            "lon_elem": (["nele"], lon_elem),
            "lat_elem": (["nele"], lat_elem),
            "sigma_lev": (["sigma"], sigma_lev),
            "u2d": (["time", "nele"], u2d),
            "v2d": (["time", "nele"], v2d),
            "eta": (["time", "node"], eta),
            "Kh_zonal": (["sigma", "node"], kh_x_3d),
            "Kh_meridional": (["sigma", "node"], kh_y_3d),
        }).to_netcdf(p)

        cfg = SimpleNamespace(meshpath=str(p))
        mode = SimpleNamespace(Fdrift3D=False, Fdrift2D=True, Vpuls=False)
        reader = get_reader("fesomc")
        mesh = reader.load_mesh(cfg)
        ds = reader.open_dataset(p)
        try:
            with pytest.raises(ValueError, match="1-D"):
                reader.read_velocities(ds, 0, mesh, mode,
                                       diffusion=_diff_cfg())
        finally:
            ds.close()

    def test_size_mismatch_raises(self, tmp_path):
        """Kh array whose size doesn't match the inferred layout is rejected.

        We bypass the reader's xarray-enforced dim-size constraint and call
        ``_read_kh_fields`` against a manually-constructed dataset where the
        layout-detector has to fall back to dim-name hints (``nele``) but
        the actual array length is wrong (4 instead of n_elements=2).
        """
        from types import SimpleNamespace
        from kelp.readers import _read_kh_fields
        p = tmp_path / "fesomc_size_mismatch.nc"
        _make_fesomc_with_kh(p, layout="node")  # mesh has 4 nodes, 2 elems

        cfg = SimpleNamespace(meshpath=str(p))
        reader = get_reader("fesomc")
        mesh = reader.load_mesh(cfg)
        # Build a Dataset where dim "nele" has size 4 (mismatched with
        # mesh.n_elements=2) so the dim-name heuristic says "element"
        # but the size validator catches it.
        bad_ds = xr.Dataset({
            "Kh_zonal": (["nele"], np.full(4, 7.0, dtype=np.float64)),
            "Kh_meridional": (["nele"], np.full(4, 9.0, dtype=np.float64)),
        })
        with pytest.raises(ValueError, match="size 4 but layout"):
            _read_kh_fields(bad_ds, 0, mesh, _diff_cfg())


def _make_fesom2_with_kh(mesh_dir, vel_path, layout,
                         n_time=2, n_elem=2, n_node=4):
    """FESOM2 ASCII mesh + velocity NetCDF that includes Kh."""
    _make_fesom2_mesh(mesh_dir, n_node=n_node, n_elem=n_elem)

    u = np.ones((n_time, n_node), dtype=np.float64) * 2.0
    v = np.ones((n_time, n_node), dtype=np.float64) * -0.5
    ssh = np.zeros((n_time, n_node), dtype=np.float64)

    if layout == "node":
        kh_dim = "nod2"
        kh_x = np.full(n_node, 7.0, dtype=np.float64)
        kh_y = np.full(n_node, 9.0, dtype=np.float64)
    else:
        kh_dim = "elem"
        kh_x = np.full(n_elem, 7.0, dtype=np.float64)
        kh_y = np.full(n_elem, 9.0, dtype=np.float64)

    ds = xr.Dataset({
        "u": (["time", "nod2"], u),
        "v": (["time", "nod2"], v),
        "ssh": (["time", "nod2"], ssh),
        "Kh_zonal": ([kh_dim], kh_x),
        "Kh_meridional": ([kh_dim], kh_y),
    })
    ds.to_netcdf(vel_path)


class TestFesom2KhReader:
    def test_kh_node_layout(self, tmp_path):
        from types import SimpleNamespace
        mesh_dir = tmp_path / "mesh"
        vel_path = tmp_path / "vel.nc"
        _make_fesom2_with_kh(mesh_dir, vel_path, layout="node")

        cfg = SimpleNamespace(meshpath=str(mesh_dir))
        mode = SimpleNamespace(Fdrift3D=False, Fdrift2D=True,
                               sigma_layer=None, var_w=None)
        reader = get_reader("fesom2")
        mesh = reader.load_mesh(cfg)
        ds = reader.open_dataset(str(vel_path))
        try:
            vel = reader.read_velocities(ds, 0, mesh, mode,
                                         diffusion=_diff_cfg())
        finally:
            ds.close()
        assert vel.kh_layout == "node"
        np.testing.assert_allclose(vel.kh_zonal, 7.0)

    def test_kh_element_layout(self, tmp_path):
        from types import SimpleNamespace
        mesh_dir = tmp_path / "mesh"
        vel_path = tmp_path / "vel.nc"
        _make_fesom2_with_kh(mesh_dir, vel_path, layout="element")

        cfg = SimpleNamespace(meshpath=str(mesh_dir))
        mode = SimpleNamespace(Fdrift3D=False, Fdrift2D=True,
                               sigma_layer=None, var_w=None)
        reader = get_reader("fesom2")
        mesh = reader.load_mesh(cfg)
        ds = reader.open_dataset(str(vel_path))
        try:
            vel = reader.read_velocities(ds, 0, mesh, mode,
                                         diffusion=_diff_cfg())
        finally:
            ds.close()
        assert vel.kh_layout == "element"
        np.testing.assert_allclose(vel.kh_zonal, 7.0)

    def test_disabled_diffusion(self, tmp_path):
        from types import SimpleNamespace
        mesh_dir = tmp_path / "mesh"
        vel_path = tmp_path / "vel.nc"
        _make_fesom2_with_kh(mesh_dir, vel_path, layout="node")

        cfg = SimpleNamespace(meshpath=str(mesh_dir))
        mode = SimpleNamespace(Fdrift3D=False, Fdrift2D=True,
                               sigma_layer=None, var_w=None)
        reader = get_reader("fesom2")
        mesh = reader.load_mesh(cfg)
        ds = reader.open_dataset(str(vel_path))
        try:
            vel = reader.read_velocities(ds, 0, mesh, mode,
                                         diffusion=_diff_cfg(enabled=False))
        finally:
            ds.close()
        assert vel.kh_layout is None


# ===================================================================
# extract_time_info tests
# ===================================================================

class TestExtractTimeInfo:
    def test_datetime64_regular(self, tmp_path):
        """Regular datetime64 time coordinate → correct dt in seconds."""
        import pandas as pd
        times = pd.date_range("2020-01-01", periods=5, freq="6h")
        ds = xr.Dataset({"x": (["time"], np.zeros(5))}, coords={"time": times})
        secs, file_dt = extract_time_info(ds)
        assert file_dt == pytest.approx(6 * 3600.0)
        assert secs[0] == pytest.approx(0.0)
        assert secs[-1] == pytest.approx(4 * 6 * 3600.0)

    def test_numeric_seconds(self, tmp_path):
        """Numeric time in seconds."""
        ds = xr.Dataset(
            {"x": (["time"], np.zeros(4))},
            coords={"time": np.array([0.0, 1800.0, 3600.0, 5400.0])},
        )
        secs, file_dt = extract_time_info(ds)
        assert file_dt == pytest.approx(1800.0)

    def test_numeric_hours(self, tmp_path):
        """Numeric time with 'hours' units attribute."""
        time_var = xr.Variable("time", np.array([0.0, 6.0, 12.0]),
                               attrs={"units": "hours since 2020-01-01"})
        ds = xr.Dataset({"x": (["time"], np.zeros(3))}, coords={"time": time_var})
        secs, file_dt = extract_time_info(ds)
        assert file_dt == pytest.approx(6 * 3600.0)

    def test_single_timestep(self):
        """Single timestep → dt is None."""
        ds = xr.Dataset(
            {"x": (["time"], np.zeros(1))},
            coords={"time": np.array([0.0])},
        )
        secs, file_dt = extract_time_info(ds)
        assert secs is not None
        assert file_dt is None

    def test_no_time_dim(self):
        """Dataset without time → (None, None)."""
        ds = xr.Dataset({"x": (["node"], np.zeros(3))})
        secs, file_dt = extract_time_info(ds)
        assert secs is None
        assert file_dt is None

    def test_irregular_spacing(self):
        """Irregular spacing → dt is None."""
        ds = xr.Dataset(
            {"x": (["time"], np.zeros(3))},
            coords={"time": np.array([0.0, 100.0, 500.0])},
        )
        secs, file_dt = extract_time_info(ds)
        assert secs is not None
        assert file_dt is None


# ===================================================================
# Zarr input tests
# ===================================================================

def _make_fesom2_mesh_zarr(zarr_path, n_node=4, n_elem=2):
    """Create a FESOM2-style mesh Zarr store (same layout as extract_fesom_region).

    Same 4-node, 2-triangle geometry as the other test helpers.
    """
    lon = np.array([4.0, 5.0, 4.0, 5.0], dtype=np.float64)
    lat = np.array([5.0, 5.0, 4.0, 4.0], dtype=np.float64)

    # elements: (n3, elem) = (3, 2), 1-based (FESOM2 convention)
    elements = np.array([[1, 1], [2, 4], [4, 3]], dtype=np.int32)

    # node_to_elems: (N=2, nod2=4), 1-based, 0 = padding
    n2e = np.array([
        [1, 1, 2, 1],
        [2, 0, 0, 2],
    ], dtype=np.int32)
    n2e_count = np.array([2, 1, 1, 2], dtype=np.int32)

    depth = np.array([-10.0, -12.0, -8.0, -15.0], dtype=np.float64)

    lon_c = np.array([
        (4.0 + 5.0 + 5.0) / 3,
        (4.0 + 5.0 + 4.0) / 3,
    ], dtype=np.float64)
    lat_c = np.array([
        (5.0 + 5.0 + 4.0) / 3,
        (5.0 + 4.0 + 4.0) / 3,
    ], dtype=np.float64)

    ds = xr.Dataset({
        "lon": (["nod2"], lon),
        "lat": (["nod2"], lat),
        "elements": (["n3", "elem"], elements),
        "nod_in_elem2D": (["N", "nod2"], n2e),
        "nod_in_elem2D_num": (["nod2"], n2e_count),
        "zbar_n_bottom": (["nod2"], depth),
        "elem_center_lon": (["elem"], lon_c),
        "elem_center_lat": (["elem"], lat_c),
    })
    ds.to_zarr(str(zarr_path), mode="w")


def _make_fesom2_vel_zarr(zarr_path, n_time=3, n_elem=2, n_node=4):
    """Create a FESOM2-style velocity Zarr store."""
    u = np.ones((n_time, n_elem), dtype=np.float64) * 2.0
    v = np.ones((n_time, n_elem), dtype=np.float64) * -0.5
    ssh = np.zeros((n_time, n_node), dtype=np.float64)
    ds = xr.Dataset({
        "u": (["time", "nod2"], u),
        "v": (["time", "nod2"], v),
        "ssh": (["time", "nod2_node"], ssh),
    })
    ds.to_zarr(str(zarr_path), mode="w")


def _make_fesomc_zarr(zarr_path, n_time=3, n_elem=2, n_node=4):
    """Create a FESOM-C-style Zarr store (mesh + velocity combined)."""
    lon = np.array([4.0, 5.0, 4.0, 5.0], dtype=np.float64)
    lat = np.array([5.0, 5.0, 4.0, 4.0], dtype=np.float64)
    depth = np.array([10.0, 12.0, 8.0, 15.0], dtype=np.float64)

    nv = np.array([
        [1, 2, 4, 1],
        [1, 4, 3, 1],
    ], dtype=np.int32)

    lon_elem = np.array([
        (4.0 + 5.0 + 5.0) / 3,
        (4.0 + 5.0 + 4.0) / 3,
    ], dtype=np.float64)
    lat_elem = np.array([
        (5.0 + 5.0 + 4.0) / 3,
        (5.0 + 4.0 + 4.0) / 3,
    ], dtype=np.float64)

    sigma_lev = np.array([0.0, 0.5, 1.0], dtype=np.float64)

    u2d = np.ones((n_time, n_elem), dtype=np.float64) * 1.5
    v2d = np.ones((n_time, n_elem), dtype=np.float64) * 0.5
    eta = np.zeros((n_time, n_node), dtype=np.float64)

    ds = xr.Dataset({
        "lon": (["node"], lon),
        "lat": (["node"], lat),
        "depth": (["node"], depth),
        "nv": (["nele", "four"], nv),
        "lon_elem": (["nele"], lon_elem),
        "lat_elem": (["nele"], lat_elem),
        "sigma_lev": (["sigma"], sigma_lev),
        "u2d": (["time", "nele"], u2d),
        "v2d": (["time", "nele"], v2d),
        "eta": (["time", "node"], eta),
    })
    ds.to_zarr(str(zarr_path), mode="w")


class TestZarrInputFesom2:
    """Test FESOM2 reader with Zarr input."""

    def test_load_mesh_zarr(self, tmp_path):
        from types import SimpleNamespace
        zarr_path = tmp_path / "mesh.zarr"
        _make_fesom2_mesh_zarr(zarr_path)
        cfg = SimpleNamespace(meshpath=str(zarr_path))
        mesh = get_reader("fesom2").load_mesh(cfg)

        assert mesh.name == "fesom2"
        assert mesh.n_nodes == 4
        assert mesh.n_elements == 2
        assert mesh._gh_offsets is not None
        assert mesh.nodes_lon[0] == pytest.approx(4.0)
        assert mesh.depth[0] == pytest.approx(10.0)

    def test_load_mesh_zarr_elem_centres(self, tmp_path):
        from types import SimpleNamespace
        zarr_path = tmp_path / "mesh.zarr"
        _make_fesom2_mesh_zarr(zarr_path)
        cfg = SimpleNamespace(meshpath=str(zarr_path))
        mesh = get_reader("fesom2").load_mesh(cfg)

        expected_lon = (4.0 + 5.0 + 5.0) / 3
        assert mesh.elem_center_lon[0] == pytest.approx(expected_lon, abs=0.01)

    def test_open_dataset_zarr_single(self, tmp_path):
        zarr_path = tmp_path / "vel.zarr"
        _make_fesom2_vel_zarr(zarr_path)
        ds = get_reader("fesom2").open_dataset(str(zarr_path))

        assert "u" in ds
        assert "v" in ds
        assert ds.sizes["time"] == 3

    def test_open_dataset_zarr_directory(self, tmp_path):
        """Directory containing u*.zarr and v*.zarr split stores."""
        vel_dir = tmp_path / "velocity"
        vel_dir.mkdir()

        u = np.ones((3, 2), dtype=np.float64)
        v = np.ones((3, 2), dtype=np.float64) * 0.5
        xr.Dataset({"u": (["time", "nod2"], u)}).to_zarr(
            str(vel_dir / "u.region.zarr"), mode="w")
        xr.Dataset({"v": (["time", "nod2"], v)}).to_zarr(
            str(vel_dir / "v.region.zarr"), mode="w")

        ds = get_reader("fesom2").open_dataset(str(vel_dir))
        assert "u" in ds
        assert "v" in ds

    def test_open_dataset_zarr_directory_extra_scalars(self, tmp_path):
        """Split-zarr dir (extract_fesom_region layout): temp/salt stores are
        merged into the dataset for sample_vars; the mesh diag is skipped."""
        vel_dir = tmp_path / "region"
        vel_dir.mkdir()

        n_t, n_n = 3, 2
        xr.Dataset({"u": (["time", "nz1", "nod2"],
                          np.ones((n_t, 2, n_n)))}).to_zarr(
            str(vel_dir / "u.region.zarr"), mode="w")
        xr.Dataset({"v": (["time", "nz1", "nod2"],
                          np.ones((n_t, 2, n_n)) * 0.5)}).to_zarr(
            str(vel_dir / "v.region.zarr"), mode="w")
        xr.Dataset({"w": (["time", "nz", "nod2"],
                          np.ones((n_t, 3, n_n)) * 0.1)}).to_zarr(
            str(vel_dir / "w.region.zarr"), mode="w")
        xr.Dataset({"temp": (["time", "nz1", "nod2"],
                             np.ones((n_t, 2, n_n)) * 4.0)}).to_zarr(
            str(vel_dir / "temp.region.zarr"), mode="w")
        xr.Dataset({"salt": (["time", "nz1", "nod2"],
                             np.ones((n_t, 2, n_n)) * 35.0)}).to_zarr(
            str(vel_dir / "salt.region.zarr"), mode="w")
        # Mesh diag in the same dir must NOT leak into the dataset
        xr.Dataset({"elements": (["three", "elem"],
                                 np.ones((3, 5), dtype=np.int32))}).to_zarr(
            str(vel_dir / "fesom.mesh.diag.region.zarr"), mode="w")

        ds = get_reader("fesom2").open_dataset(str(vel_dir))
        assert "u" in ds and "v" in ds and "w" in ds
        assert "temp" in ds
        assert "salt" in ds
        assert "elements" not in ds
        np.testing.assert_allclose(ds["temp"].isel(time=0).values, 4.0)
        np.testing.assert_allclose(ds["salt"].isel(time=0).values, 35.0)

    def test_open_dataset_nc_directory_extra_scalars(self, tmp_path):
        """Split-NetCDF dir (raw FESOM2 output layout): per-variable
        temp*/salt* files merge; mesh diag and alien-grid files are skipped."""
        vel_dir = tmp_path / "year"
        vel_dir.mkdir()

        n_t, n_e, n_n = 2, 3, 4
        xr.Dataset({"u": (["time", "nz1", "elem"],
                          np.ones((n_t, 2, n_e)))}).to_netcdf(
            vel_dir / "u.fesom.2018.nc")
        xr.Dataset({"v": (["time", "nz1", "elem"],
                          np.ones((n_t, 2, n_e)) * 0.5)}).to_netcdf(
            vel_dir / "v.fesom.2018.nc")
        xr.Dataset({"temp": (["time", "nz1", "nod2"],
                             np.ones((n_t, 2, n_n)) * 4.0)}).to_netcdf(
            vel_dir / "temp.fesom.2018.nc")
        xr.Dataset({"salt": (["time", "nz1", "nod2"],
                             np.ones((n_t, 2, n_n)) * 35.0)}).to_netcdf(
            vel_dir / "salt.fesom.2018.nc")
        # mesh diag (skipped by name) + alien-dims file (skipped by dims)
        xr.Dataset({"nod_area": (["nz", "nod2"],
                                 np.ones((3, n_n)))}).to_netcdf(
            vel_dir / "fesom.mesh.diag.nc")
        xr.Dataset({"junk": (["weird_dim"], np.arange(7.0))}).to_netcdf(
            vel_dir / "stationlist.nc")

        ds = get_reader("fesom2").open_dataset(str(vel_dir))
        assert "u" in ds and "v" in ds
        assert "temp" in ds and "salt" in ds
        assert "nod_area" not in ds
        assert "junk" not in ds
        np.testing.assert_allclose(ds["salt"].isel(time=1).values, 35.0)

    def test_read_velocities_zarr(self, tmp_path):
        from types import SimpleNamespace
        # Mesh
        zarr_mesh = tmp_path / "mesh.zarr"
        _make_fesom2_mesh_zarr(zarr_mesh)
        cfg = SimpleNamespace(meshpath=str(zarr_mesh))
        mesh = get_reader("fesom2").load_mesh(cfg)

        # Velocity
        zarr_vel = tmp_path / "vel.zarr"
        _make_fesom2_vel_zarr(zarr_vel)
        ds = get_reader("fesom2").open_dataset(str(zarr_vel))

        mode = SimpleNamespace(Fdrift3D=False, sigma_layer=None,
                               sigma_mean=False, sFB=1)
        vel = get_reader("fesom2").read_velocities(ds, 0, mesh, mode)

        assert vel.u.shape == (2,)
        assert vel.u[0] == pytest.approx(2.0)
        assert vel.v[0] == pytest.approx(-0.5)


class TestZarrInputFesomC:
    """Test FESOM-C reader with Zarr input."""

    def test_load_mesh_zarr(self, tmp_path):
        from types import SimpleNamespace
        zarr_path = tmp_path / "fesomc.zarr"
        _make_fesomc_zarr(zarr_path)
        cfg = SimpleNamespace(meshpath=str(zarr_path))
        mesh = get_reader("fesomc").load_mesh(cfg)

        assert mesh.name == "fesomc"
        assert mesh.n_nodes == 4
        assert mesh.n_elements == 2
        assert mesh._gh_offsets is not None

    def test_open_dataset_zarr(self, tmp_path):
        zarr_path = tmp_path / "fesomc.zarr"
        _make_fesomc_zarr(zarr_path)
        ds = get_reader("fesomc").open_dataset(str(zarr_path))

        assert "u2d" in ds
        assert "v2d" in ds
        assert ds.sizes["time"] == 3

    def test_read_velocities_zarr(self, tmp_path):
        from types import SimpleNamespace
        zarr_path = tmp_path / "fesomc.zarr"
        _make_fesomc_zarr(zarr_path)

        cfg_mesh = SimpleNamespace(meshpath=str(zarr_path))
        mesh = get_reader("fesomc").load_mesh(cfg_mesh)

        ds = get_reader("fesomc").open_dataset(str(zarr_path))

        mode = SimpleNamespace(
            Fdrift3D=False, sigma_layer=None, sigma_mean=False,
            sFB=1, var_u=None, var_v=None, var_w=None, var_eta=None,
            Vpuls=0,
        )
        vel = get_reader("fesomc").read_velocities(ds, 0, mesh, mode)

        assert vel.u[0] == pytest.approx(1.5)
        assert vel.v[0] == pytest.approx(0.5)
