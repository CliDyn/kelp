"""Tests for field sampling (TODO 2.4): transport, core sampler, integration."""
import numpy as np
import pytest

from kelp.core.io import VelocityField
from kelp.core.mesh import Mesh, build_search, build_node_to_elems


def _vf(uval, scalars=None):
    return VelocityField(
        u=np.full(4, uval), v=np.zeros(4), eta=np.zeros(9), scalars=scalars,
    )


class TestVelocityFieldScalars:
    def test_default_none(self):
        assert _vf(1.0).scalars is None

    def test_empty_dict_normalized_to_none(self):
        assert _vf(1.0, scalars={}).scalars is None

    def test_lerp_both_sides(self):
        a = _vf(0.0, scalars={"temp": np.zeros(9)})
        b = _vf(1.0, scalars={"temp": np.full(9, 2.0)})
        mid = a.lerp(b, 0.5)
        np.testing.assert_allclose(mid.scalars["temp"], 1.0)

    def test_lerp_one_side_returns_none(self):
        a = _vf(0.0, scalars={"temp": np.zeros(9)})
        b = _vf(1.0)
        assert a.lerp(b, 0.5).scalars is None

    def test_lerp_key_mismatch_raises(self):
        a = _vf(0.0, scalars={"temp": np.zeros(9)})
        b = _vf(1.0, scalars={"salt": np.zeros(9)})
        with pytest.raises(ValueError, match="scalars"):
            a.lerp(b, 0.5)

    def test_lerp_shape_mismatch_raises(self):
        a = _vf(0.0, scalars={"temp": np.zeros(9)})
        b = _vf(1.0, scalars={"temp": np.zeros((2, 9))})
        with pytest.raises(ValueError, match="scalars"):
            a.lerp(b, 0.5)

    def test_lerp_alpha_boundaries(self):
        a = _vf(0.0, scalars={"temp": np.zeros(9)})
        b = _vf(1.0, scalars={"temp": np.full(9, 4.0)})
        np.testing.assert_allclose(a.lerp(b, 0.0).scalars["temp"], 0.0)
        np.testing.assert_allclose(a.lerp(b, 1.0).scalars["temp"], 4.0)


from kelp.numerics.sampling import sample_field

METHODS = ["nearest", "leastsq", "barycentric", "idw"]


def _grid_mesh(xs, ys):
    # identical to tests/test_mesh.py::_grid_mesh — regular triangulated grid
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel().astype(np.float64)
    lat = gy.ravel().astype(np.float64)
    nx, ny = len(xs), len(ys)
    elems = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            n1, n2 = n0 + 1, n0 + nx
            elems.append([n0, n1, n2])
            elems.append([n1, n2 + 1, n2])
    elem_nodes = np.array(elems, dtype=np.int32)
    n2e, n2e_count = build_node_to_elems(nx * ny, elem_nodes)
    mesh = Mesh(name="grid", nodes_lon=lon, nodes_lat=lat,
                depth=np.full(nx * ny, 50.0), elem_nodes=elem_nodes,
                elem_center_lon=np.array([lon[e].mean() for e in elem_nodes]),
                elem_center_lat=np.array([lat[e].mean() for e in elem_nodes]),
                node_to_elems=n2e, node_to_elems_count=n2e_count)
    build_search(mesh)
    return mesh


class TestSample2D:
    @pytest.fixture(scope="class")
    def mesh(self):
        return _grid_mesh(np.linspace(0, 10, 11), np.linspace(0, 10, 11))

    @pytest.mark.parametrize("method", METHODS)
    def test_constant_field_exact(self, mesh, method):
        vals = np.full(mesh.n_nodes, 7.5)
        out = sample_field(vals, mesh, np.array([4.3]), np.array([5.6]),
                           method=method)
        np.testing.assert_allclose(out, 7.5, rtol=1e-12)

    @pytest.mark.parametrize("method", ["leastsq", "barycentric"])
    def test_linear_field_exact(self, mesh, method):
        a, b, c = 2.0, 0.3, -0.7
        vals = a + b * mesh.nodes_lon + c * mesh.nodes_lat
        lons = np.array([4.3, 7.9, 1.2])
        lats = np.array([5.6, 2.4, 8.8])
        out = sample_field(vals, mesh, lons, lats, method=method)
        np.testing.assert_allclose(out, a + b * lons + c * lats, rtol=1e-9)

    @pytest.mark.parametrize("method", METHODS)
    def test_exact_at_nodes(self, mesh, method):
        vals = np.arange(mesh.n_nodes, dtype=np.float64)
        n = 60
        out = sample_field(vals, mesh, mesh.nodes_lon[[n]], mesh.nodes_lat[[n]],
                           method=method)
        np.testing.assert_allclose(out, vals[n], rtol=1e-9)

    def test_nearest_negative_gives_nan(self, mesh):
        vals = np.zeros(mesh.n_nodes)
        out = sample_field(vals, mesh, np.array([5.0]), np.array([5.0]),
                           nearest=np.array([-1], dtype=np.int32))
        assert np.isnan(out[0])

    def test_dead_particle_gives_nan(self, mesh):
        vals = np.zeros(mesh.n_nodes)
        out = sample_field(vals, mesh, np.array([5.0, 5.0]), np.array([5.0, 5.0]),
                           alive=np.array([True, False]))
        assert not np.isnan(out[0]) and np.isnan(out[1])

    def test_unknown_method_raises(self, mesh):
        with pytest.raises(ValueError, match="method"):
            sample_field(np.zeros(mesh.n_nodes), mesh,
                         np.array([5.0]), np.array([5.0]), method="cubic")

    def test_wrong_size_raises(self, mesh):
        with pytest.raises(ValueError, match="node"):
            sample_field(np.zeros(mesh.n_nodes + 3), mesh,
                         np.array([5.0]), np.array([5.0]))


def _quad_mesh_2d():
    """Pure-QUAD 2x2-cell 2D mesh: 9 nodes, 4 quad elements (E,4) CCW.

    Equator-centred (cos(lat) ~ 1) so the local metric projection is
    near-isotropic.
    """
    xs = np.array([4.0, 5.0, 6.0])
    ys = np.array([-1.0, 0.0, 1.0])
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel().astype(np.float64)
    lat = gy.ravel().astype(np.float64)
    elems = []
    for j in range(2):
        for i in range(2):
            n_sw = j * 3 + i
            elems.append([n_sw, n_sw + 1, n_sw + 4, n_sw + 3])  # SW SE NE NW
    elem_nodes = np.array(elems, dtype=np.int32)
    n2e, n2e_count = build_node_to_elems(9, elem_nodes)
    mesh = Mesh(name="quad2d", nodes_lon=lon, nodes_lat=lat,
                depth=np.full(9, 50.0), elem_nodes=elem_nodes,
                elem_center_lon=np.array([lon[e].mean() for e in elem_nodes]),
                elem_center_lat=np.array([lat[e].mean() for e in elem_nodes]),
                node_to_elems=n2e, node_to_elems_count=n2e_count)
    build_search(mesh)
    return mesh


def _fan_weights_ref(px, py, vx, vy, nv):
    """In-test fan-triangulation reference — the pre-2026-07 fallback
    behaviour (the production `_fan_weights` was removed with the guard
    chain; quads now always take product-form Wachspress)."""
    for t in range(nv - 2):
        ix = (0, t + 1, t + 2)
        ax, ay = vx[ix[0]], vy[ix[0]]
        bx, by = vx[ix[1]], vy[ix[1]]
        cx, cy = vx[ix[2]], vy[ix[2]]
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-10:
            continue
        w1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / den
        w2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / den
        w3 = 1.0 - w1 - w2
        if w1 >= -1e-6 and w2 >= -1e-6 and w3 >= -1e-6:
            out = np.zeros(nv)
            out[ix[0]] += w1
            out[ix[1]] += w2
            out[ix[2]] += w3
            return out
    return np.full(nv, -1.0)


def _wachspress_expected_2d(mesh, elem, lon_p, lat_p, node_vals):
    """Replicate the sampling barycentric projection and return the
    in-test Wachspress-blended value for an interior probe."""
    from kelp.core.mesh import _polygon_weights
    DEG2R = 0.017453292519943295
    R = 6371000.0
    verts = mesh.elem_nodes[elem]
    lon0 = mesh.nodes_lon[verts].mean()
    lat0 = mesh.nodes_lat[verts].mean()
    cos_lat0 = np.cos(lat0 * DEG2R)
    dlon = ((mesh.nodes_lon[verts] - lon0 + 180.0) % 360.0) - 180.0
    vx = dlon * DEG2R * R * cos_lat0
    vy = (mesh.nodes_lat[verts] - lat0) * DEG2R * R
    dlon_p = ((lon_p - lon0 + 180.0) % 360.0) - 180.0
    px = dlon_p * DEG2R * R * cos_lat0
    py = (lat_p - lat0) * DEG2R * R
    wts = _polygon_weights(px, py, vx, vy, 4)
    return float(np.dot(wts, node_vals[verts])), (px, py, vx, vy, verts)


class TestSample2DQuadWachspress:
    """Sampling barycentric on a pure-quad mesh: bilinear field expectation
    computed in-test from the Wachspress weights."""

    def test_bilinear_field_matches_wachspress(self):
        mesh = _quad_mesh_2d()
        node_vals = mesh.nodes_lon * mesh.nodes_lat   # bilinear, non-affine
        lon_p, lat_p = 4.7, -0.2   # interior of element 0 (lower-left quad)
        out = sample_field(node_vals, mesh, np.array([lon_p]),
                           np.array([lat_p]), method="barycentric")
        expected, (px, py, vx, vy, verts) = _wachspress_expected_2d(
            mesh, 0, lon_p, lat_p, node_vals)
        np.testing.assert_allclose(out[0], expected, rtol=1e-9)
        # Must DIFFER from the fan result (proving the change is live).
        fan_val = float(np.dot(_fan_weights_ref(px, py, vx, vy, 4),
                               node_vals[verts]))
        assert not np.isclose(out[0], fan_val, rtol=1e-6)

    def test_linear_field_exact(self):
        # GREEN before and after (linear reproduced by fan and Wachspress).
        mesh = _quad_mesh_2d()
        a, b, c = 1.0, 2.0, -0.5
        node_vals = a + b * mesh.nodes_lon + c * mesh.nodes_lat
        lon_p, lat_p = 5.6, 0.3
        out = sample_field(node_vals, mesh, np.array([lon_p]),
                           np.array([lat_p]), method="barycentric")
        np.testing.assert_allclose(out[0], a + b * lon_p + c * lat_p,
                                   rtol=1e-9)


class TestSample3DZlevel:
    @pytest.fixture(scope="class")
    def mesh(self):
        m = _grid_mesh(np.linspace(0, 10, 11), np.linspace(0, 10, 11))
        m.sigma = np.array([0.0, 10.0, 20.0, 40.0])  # K=4 interfaces
        return m

    def test_midlayer_vertical_linear(self, mesh):
        mids = np.array([5.0, 15.0, 30.0])
        vals = np.repeat((2.0 * mids)[:, None], mesh.n_nodes, axis=1)
        out = sample_field(vals, mesh, np.array([5.0]), np.array([5.0]),
                           zs=np.array([10.0]), method="nearest")
        np.testing.assert_allclose(out, 20.0, rtol=1e-12)

    def test_midlayer_clamped_above_first(self, mesh):
        mids = np.array([5.0, 15.0, 30.0])
        vals = np.repeat((2.0 * mids)[:, None], mesh.n_nodes, axis=1)
        out = sample_field(vals, mesh, np.array([5.0]), np.array([5.0]),
                           zs=np.array([0.5]), method="nearest")
        np.testing.assert_allclose(out, 10.0, rtol=1e-12)

    def test_midlayer_clamped_below_last(self, mesh):
        mids = np.array([5.0, 15.0, 30.0])
        vals = np.repeat((2.0 * mids)[:, None], mesh.n_nodes, axis=1)
        out = sample_field(vals, mesh, np.array([5.0]), np.array([5.0]),
                           zs=np.array([39.0]), method="nearest")
        np.testing.assert_allclose(out, 60.0, rtol=1e-12)

    def test_interface_field_no_time_dir_flip(self, mesh):
        K = mesh.sigma.size
        vals = np.repeat(np.arange(K, dtype=np.float64)[:, None],
                         mesh.n_nodes, axis=1)
        kw = dict(zs=np.array([10.0]), method="leastsq")
        out = sample_field(vals, mesh, np.array([5.0]), np.array([5.0]), **kw)
        assert out[0] > 0
        np.testing.assert_allclose(out, 1.0, rtol=1e-9)

    @pytest.mark.parametrize("method", ["leastsq", "barycentric", "idw"])
    def test_midlayer_horizontal_linear(self, mesh, method):
        base = 1.0 + 0.2 * mesh.nodes_lon - 0.1 * mesh.nodes_lat
        vals = np.repeat(base[None, :], 3, axis=0)
        out = sample_field(vals, mesh, np.array([4.3]), np.array([6.7]),
                           zs=np.array([12.0]), method=method)
        expected = 1.0 + 0.2 * 4.3 - 0.1 * 6.7
        tol = 1e-9 if method != "idw" else 5e-2
        np.testing.assert_allclose(out, expected, rtol=tol)

    def test_bottom_aware_neighbour_clamp(self):
        # Linear-in-z field T = 2*z on midpoints (5, 15, 30).  One shallow
        # node right next to the query point: bottom at 12 m.  Sampling at
        # z=25 with method="idw" blends neighbours; the shallow neighbour
        # must contribute T(min(25, 12)) = T(12) = 24, not extrapolated
        # T(25) = 50.  With the clamp removed, the blended value rises —
        # this test pins the clamp.
        #
        # Exact-at-node trick: query EXACTLY at the shallow node so that
        # idw (which is exact at coincident nodes) returns precisely that
        # node's vertically-interpolated value, with no horizontal blend to
        # reason about.  Node 60 = grid point (5, 5); make it the shallow one.
        mesh = _grid_mesh(np.linspace(0, 10, 11), np.linspace(0, 10, 11))
        mesh.sigma = np.array([0.0, 10.0, 20.0, 40.0])  # K=4 -> mids 5,15,30
        mesh.depth[:] = 50.0
        shallow = 60  # node at (lon=5, lat=5)
        assert mesh.nodes_lon[shallow] == 5.0 and mesh.nodes_lat[shallow] == 5.0
        mesh.depth[shallow] = 12.0

        mids = np.array([5.0, 15.0, 30.0])
        vals = np.repeat((2.0 * mids)[:, None], mesh.n_nodes, axis=1)
        out = sample_field(
            vals, mesh,
            mesh.nodes_lon[[shallow]], mesh.nodes_lat[[shallow]],
            zs=np.array([25.0]), method="idw",
        )
        # z_eval = min(25, 12) = 12 -> between mids 5 and 15:
        #   alpha = (12-5)/10 = 0.7 -> 0.3*10 + 0.7*30 = 24.0
        # Clamp-removed mutant: z=25 between 15 and 30 -> 50.0
        np.testing.assert_allclose(out, 24.0, rtol=1e-12)


class TestSampleSigma:
    @pytest.fixture(scope="class")
    def mesh(self):
        m = _grid_mesh(np.linspace(0, 10, 11), np.linspace(0, 10, 11))
        m.sigma = np.array([1.0, 0.75, 0.4, 0.0])   # K=4 sigma interfaces
        m.vertical_coord = "sigma"
        m.depth = np.full(m.n_nodes, 20.0)
        return m

    def test_sigma_midlayer_follows_eta(self, mesh):
        eta = np.full(mesh.n_nodes, 2.0)            # H = 22
        sig = mesh.sigma
        sig_mid = 0.5 * (sig[:-1] + sig[1:])
        H = 22.0
        z_mid = (1.0 - sig_mid) * H
        vals = np.repeat((3.0 * z_mid)[:, None], mesh.n_nodes, axis=1)
        out = sample_field(vals, mesh, np.array([5.0]), np.array([5.0]),
                           zs=np.array([10.0]), eta=eta, method="nearest")
        np.testing.assert_allclose(out, 30.0, rtol=1e-9)

    def test_sigma_requires_eta(self, mesh):
        vals = np.zeros((3, mesh.n_nodes))
        with pytest.raises(ValueError, match="eta"):
            sample_field(vals, mesh, np.array([5.0]), np.array([5.0]),
                         zs=np.array([10.0]))

    def test_sigma_shallow_column_geometry(self):
        # Shallow-column correctness, NOT a clamp mutation-pin.  The sigma
        # mid-layer ``min(z, H)`` clamp is mathematically SHADOWED by the
        # last-midpoint clamp of ``_interp_mid_linear_single``: per-column
        # midpoints z_mid = (1-sig_mid)*H are always < H (sig_mid>0), so any
        # z >= H lands beyond the last midpoint and clamps to the bottom
        # layer with or without the min(z, H) guard.  A pin test is therefore
        # impossible (see the comment at the sigma clamp in sampling.py).
        #
        # Instead pin the per-node COLUMN GEOMETRY: the same field column,
        # queried at z=4 m, yields different sampled values at a node with
        # H=6 vs a node with H=20, because the sigma midpoints scale with the
        # local column depth H = depth + eta.  Query exactly at each node and
        # use idw (exact at nodes) so no horizontal blend muddies the result.
        mesh = _grid_mesh(np.linspace(0, 10, 11), np.linspace(0, 10, 11))
        mesh.sigma = np.array([1.0, 0.75, 0.4, 0.0])
        mesh.vertical_coord = "sigma"
        mesh.depth = np.full(mesh.n_nodes, 20.0)
        eta = np.zeros(mesh.n_nodes)               # H = depth

        deep = 49   # (lon=5, lat=4) -> H = 20
        shallow = 60  # (lon=5, lat=5) -> H = 6 after override
        assert mesh.nodes_lon[deep] == 5.0 and mesh.nodes_lat[deep] == 4.0
        assert mesh.nodes_lon[shallow] == 5.0 and mesh.nodes_lat[shallow] == 5.0
        mesh.depth[shallow] = 6.0

        # Field column built for an H=20 column: T(z) = 3*z at its midpoints.
        sig = mesh.sigma
        sig_mid = 0.5 * (sig[:-1] + sig[1:])        # [0.875, 0.575, 0.2]
        z_mid20 = (1.0 - sig_mid) * 20.0            # [2.5, 8.5, 16.0]
        vals = np.repeat((3.0 * z_mid20)[:, None], mesh.n_nodes, axis=1)

        out = sample_field(
            vals, mesh,
            mesh.nodes_lon[[deep, shallow]], mesh.nodes_lat[[deep, shallow]],
            zs=np.array([4.0, 4.0]), eta=eta, method="idw",
        )
        # Deep (H=20): midpoints [2.5, 8.5, 16.0]; z=4 between 2.5 and 8.5 ->
        #   the field is exactly linear here, so T = 3*4 = 12.0.
        # Shallow (H=6): midpoints (1-sig_mid)*6 = [0.75, 2.55, 4.8];
        #   z=4 between 2.55 and 4.8 -> alpha=(4-2.55)/2.25 -> blend -> 40.0.
        np.testing.assert_allclose(out[0], 12.0, rtol=1e-9)
        np.testing.assert_allclose(out[1], 40.0, rtol=1e-9)
        assert not np.isclose(out[0], out[1])


# ===================================================================
# Reader-side scalar loading (TODO 2.4, Task 4)
# ===================================================================

import xarray as xr
from kelp.readers import get_reader

# Reuse the canonical FESOM-C 3D fixture builder (house pattern: cross-test
# imports, cf. tests/test_failure_reason.py importing from
# tests.test_integrators_diffusion).
from tests.test_readers import _make_fesomc_3d


def _fesom2_ds(n_nodes, K, n_time=2):
    """Minimal FESOM2-style dataset: u/v on elements, temp on nz1, salt 2D."""
    nz1 = K - 1
    # 180: arbitrary element count — u/v not validated against the mesh by
    # read_scalar_fields (only the scalar node-vars are).
    rng = np.random.default_rng(3)
    return xr.Dataset(
        {
            "u": (("time", "nz1", "elem"), rng.normal(size=(n_time, nz1, 180)).astype(np.float32)),
            "v": (("time", "nz1", "elem"), rng.normal(size=(n_time, nz1, 180)).astype(np.float32)),
            "ssh": (("time", "nod2"), np.zeros((n_time, n_nodes), np.float32)),
            "w": (("time", "nz", "nod2"), np.zeros((n_time, K, n_nodes), np.float32)),
            "temp": (("time", "nz1", "nod2"),
                     np.arange(n_time * nz1 * n_nodes, dtype=np.float64)
                     .reshape(n_time, nz1, n_nodes)),
            "salt": (("time", "nod2"),
                     np.full((n_time, n_nodes), 35.0)),
        },
        coords={"time": np.arange(n_time)},
    )


class TestReadScalarFieldsFesom2:
    def setup_method(self):
        self.mesh = _grid_mesh(np.linspace(0, 10, 11), np.linspace(0, 10, 11))
        self.mesh.sigma = np.array([0.0, 10.0, 20.0, 40.0])  # K=4
        self.reader = get_reader("fesom2")
        self.ds = _fesom2_ds(self.mesh.n_nodes, K=4)

    def _mode(self, **kw):
        from kelp.core.config import ModeConfig
        kw.setdefault("Fdrift2D", False)
        kw.setdefault("Fdrift3D", True)
        return ModeConfig(**kw)

    def test_3d_layer_var_shape(self):
        out = self.reader.read_scalar_fields(
            self.ds, 0, self.mesh, ["temp"], self._mode())
        assert out["temp"].shape == (3, self.mesh.n_nodes)
        assert out["temp"].dtype == np.float64

    def test_2d_var_shape(self):
        out = self.reader.read_scalar_fields(
            self.ds, 1, self.mesh, ["salt"], self._mode())
        assert out["salt"].shape == (self.mesh.n_nodes,)

    def test_missing_var_raises(self):
        with pytest.raises(ValueError, match="dens"):
            self.reader.read_scalar_fields(
                self.ds, 0, self.mesh, ["dens"], self._mode())

    def test_element_sized_var_raises(self):
        ds = self.ds.assign(bad=(("time", "elem"), np.zeros((2, 180))))
        with pytest.raises(ValueError, match="bad"):
            self.reader.read_scalar_fields(ds, 0, self.mesh, ["bad"], self._mode())

    def test_2d_mode_picks_level(self):
        mode = self._mode(Fdrift2D=True, Fdrift3D=False, sigma_layer=1)
        out = self.reader.read_scalar_fields(self.ds, 0, self.mesh, ["temp"], mode)
        assert out["temp"].shape == (self.mesh.n_nodes,)
        np.testing.assert_array_equal(
            out["temp"], self.ds["temp"].isel(time=0, nz1=1).values)

    def test_2d_mode_default_surface(self):
        mode = self._mode(Fdrift2D=True, Fdrift3D=False)   # sigma_layer=None
        out = self.reader.read_scalar_fields(self.ds, 0, self.mesh, ["temp"], mode)
        np.testing.assert_array_equal(
            out["temp"], self.ds["temp"].isel(time=0, nz1=0).values)

    def test_max_levels_trims_layers(self):
        mode = self._mode(max_levels=3)          # keep 3 interfaces -> 2 layers
        mesh = self.mesh
        mesh.sigma = mesh.sigma[:3]
        out = self.reader.read_scalar_fields(self.ds, 0, mesh, ["temp"], mode)
        assert out["temp"].shape == (2, mesh.n_nodes)

    def test_interface_var_kept_at_K(self):
        ds = self.ds.assign(
            wflux=(("time", "nz", "nod2"), np.ones((2, 4, self.mesh.n_nodes))))
        out = self.reader.read_scalar_fields(ds, 0, self.mesh, ["wflux"],
                                             self._mode())
        assert out["wflux"].shape == (4, self.mesh.n_nodes)

    def test_square_field_node_first_oriented_by_dims(self):
        # N == n_layers makes the shape heuristic ambiguous: a 2x2-node mesh
        # (N=4) with nz1=4 stored (nod2, nz1) must still come back (4, N)
        # with layers on axis 0.
        mesh = _grid_mesh(np.linspace(0, 1, 2), np.linspace(0, 1, 2))  # N=4
        mesh.sigma = np.array([0., 10., 20., 30., 40.])                 # K=5, nz1=4
        vals = np.arange(4 * 4, dtype=np.float64).reshape(4, 4)        # (nod2, nz1)
        ds = xr.Dataset({"temp": (("time", "nod2", "nz1"),
                                  vals[None, :, :])},
                        coords={"time": [0]})
        out = self.reader.read_scalar_fields(ds, 0, mesh, ["temp"], self._mode())
        # layer k of node n must equal vals[n, k] -> oriented (nz1, nod2)
        np.testing.assert_array_equal(out["temp"], vals.T)

    def test_max_levels_trims_without_mesh_sigma_trim(self):
        # B2: direct callers may pass an UNTRIMMED mesh.sigma (K=4) but set
        # max_levels=3 on the mode.  Scalars must trim reader-side to stay
        # consistent with read_velocities (which trims u/v/w by max_levels
        # directly).  Without touching mesh.sigma: 3 interfaces -> 2 layers.
        mode = self._mode(max_levels=3)
        out = self.reader.read_scalar_fields(
            self.ds, 0, self.mesh, ["temp"], mode)
        assert out["temp"].shape == (2, self.mesh.n_nodes)

    @pytest.mark.parametrize("vdim", ["depth", "level"])
    @pytest.mark.parametrize("trim_mesh", [True, False])
    def test_max_levels_depth_dim_layer_field_by_pre_trim_size(
            self, vdim, trim_mesh):
        # A mid-layer field on a non-canonical vertical dim (K-1 = 3 layers
        # next to the interface dim nz = 4) under max_levels=3 keeps 2
        # layers.  Before 2026-11 the size was compared with the TRIMMED
        # interface count (3 >= 3), so the field passed as an interface
        # field: 3 rows, half a layer shifted.
        N = self.mesh.n_nodes
        vals = np.arange(2 * 3 * N, dtype=np.float64).reshape(2, 3, N)
        iface = np.arange(2 * 4 * N, dtype=np.float64).reshape(2, 4, N)
        other = "level" if vdim == "depth" else "depth"
        ds = self.ds.assign(tlay=(("time", vdim, "nod2"), vals),
                            tifc=(("time", other, "nod2"), iface))
        mesh = self.mesh
        if trim_mesh:
            mesh.sigma = mesh.sigma[:3]
        out = self.reader.read_scalar_fields(
            ds, 1, mesh, ["tlay", "tifc"], self._mode(max_levels=3))
        assert out["tlay"].shape == (2, N)
        np.testing.assert_array_equal(out["tlay"], vals[1, :2])
        # an interface field on the other name stays an interface field
        assert out["tifc"].shape == (3, N)
        np.testing.assert_array_equal(out["tifc"], iface[1, :3])

    def test_read_velocities_attaches_scalars(self):
        vel = self.reader.read_velocities(
            self.ds, 0, self.mesh, self._mode(), sample_vars=["temp", "salt"])
        assert set(vel.scalars) == {"temp", "salt"}

    def test_read_velocities_default_no_scalars(self):
        vel = self.reader.read_velocities(self.ds, 0, self.mesh, self._mode())
        assert vel.scalars is None


# -------------------------------------------------------------------
# FESOM-C reader (sigma coordinate)
# -------------------------------------------------------------------

def _sigma_grid_mesh():
    """11x11 grid mesh with K=4 sigma interfaces (decreasing 1->0)."""
    m = _grid_mesh(np.linspace(0, 10, 11), np.linspace(0, 10, 11))
    m.sigma = np.array([1.0, 0.75, 0.4, 0.0])   # K=4
    m.vertical_coord = "sigma"
    m.depth = np.full(m.n_nodes, 20.0)
    return m


def _fesomc_scalar_ds(n_nodes, K, n_time=2):
    """FESOM-C-style dataset: temp on (time, sigma_lay, node), salt 2D."""
    n_lay = K - 1
    rng = np.random.default_rng(7)
    return xr.Dataset(
        {
            "temp": (("time", "sigma_lay", "node"),
                     rng.normal(size=(n_time, n_lay, n_nodes)).astype(np.float64)),
            "salt": (("time", "node"),
                     np.full((n_time, n_nodes), 33.0)),
        },
        coords={"time": np.arange(n_time)},
    )


def _make_fesomc_3d_with_scalar(path, n_time=2, n_elem=2, n_node=4, n_sigma=3):
    """FESOM-C 3D NetCDF (u/v on sigmam1, w = Wvel on sigma) + a temp scalar.

    Thin wrapper over tests/test_readers.py::_make_fesomc_3d (which writes the
    base file), reopening it to append a node-based layer scalar
    ``temp(time, sigmam1, node)`` so read_velocities can attach it via
    ``sample_vars``.
    """
    _make_fesomc_3d(path, n_time=n_time, n_elem=n_elem, n_node=n_node,
                    n_sigma=n_sigma)
    n_layers = n_sigma - 1
    temp = np.arange(n_time * n_layers * n_node, dtype=np.float64).reshape(
        n_time, n_layers, n_node)
    with xr.open_dataset(path) as ds_in:
        ds = ds_in.load()
    ds["temp"] = (("time", "sigmam1", "node"), temp)
    ds.to_netcdf(path, mode="w")


def _make_fesomc_sigma_e2e(path, n_time=3, eta_val=2.0, depth_val=10.0):
    """11x11 FESOM-C 3D sigma NetCDF tuned for an end-to-end sampling check.

    Mirrors _make_fesomc_sampling_ds (loadable 11x11 FESOM-C mesh) but in 3D:
    u/v on ``(time, nele, sigmam1)``, ``w_cart`` on ``(time, node, sigma)``,
    and a node-based layer scalar ``temp(time, sigmam1, node)``.  An 11x11
    grid gives a rock-solid interior (vs the degenerate 2-element fixture),
    so a particle at (5, 5) = node 60 stays well away from any element edge.

    Tuned so the sampled value is analytic:
      * UNIFORM node depth  -> H = depth + eta identical at every node, so the
        horizontal blend of a horizontally-constant scalar is a no-op;
      * CONSTANT nonzero ``eta`` -> exercises H = depth + eta (not eta=0);
      * u = v = w_cart = 0 -> the particle parks at a fixed (lon, lat, z);
      * ``temp[t, layer, node] = layer * 10`` -> vertically distinct,
        horizontally constant.
    """
    from pathlib import Path
    xs = np.linspace(0.0, 10.0, 11)
    ys = np.linspace(0.0, 10.0, 11)
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel().astype(np.float64)
    lat = gy.ravel().astype(np.float64)
    n_node = lon.size
    depth = np.full(n_node, depth_val, dtype=np.float64)
    elems = []
    nx = ny = 11
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            n1, n2, n3 = n0 + 1, n0 + nx, n0 + nx + 1
            elems.append([n0 + 1, n1 + 1, n2 + 1, n0 + 1])
            elems.append([n1 + 1, n3 + 1, n2 + 1, n1 + 1])
    nv = np.array(elems, dtype=np.int32)
    n_elem = nv.shape[0]
    elem_nodes_0 = nv[:, :3] - 1
    lon_c = np.array([lon[e].mean() for e in elem_nodes_0], dtype=np.float64)
    lat_c = np.array([lat[e].mean() for e in elem_nodes_0], dtype=np.float64)
    sigma_lev = np.array([1.0, 0.5, 0.0], dtype=np.float64)   # K=3, 2 layers
    n_sigma = sigma_lev.size
    n_layers = n_sigma - 1

    u = np.zeros((n_time, n_elem, n_layers), dtype=np.float64)
    v = np.zeros((n_time, n_elem, n_layers), dtype=np.float64)
    w_cart = np.zeros((n_time, n_node, n_sigma), dtype=np.float64)
    eta = np.full((n_time, n_node), eta_val, dtype=np.float64)
    # temp[t, layer, node] = layer * 10  (broadcast across time + node).
    layer_vals = np.arange(n_layers, dtype=np.float64) * 10.0
    temp = np.broadcast_to(
        layer_vals[None, :, None], (n_time, n_layers, n_node)
    ).copy()

    ds = xr.Dataset({
        "lon": (["node"], lon),
        "lat": (["node"], lat),
        "depth": (["node"], depth),
        "nv": (["nele", "four"], nv),
        "lon_elem": (["nele"], lon_c),
        "lat_elem": (["nele"], lat_c),
        "sigma_lev": (["sigma"], sigma_lev),
        "u": (["time", "nele", "sigmam1"], u),
        "v": (["time", "nele", "sigmam1"], v),
        "w": (["time", "node", "sigma"], w_cart),       # FESOM-C Wvel
        "eta": (["time", "node"], eta),
        "temp": (["time", "sigmam1", "node"], temp),
    })
    ds.to_netcdf(Path(path))


class TestReadScalarFieldsFesomc:
    def setup_method(self):
        self.mesh = _sigma_grid_mesh()                # K=4
        self.reader = get_reader("fesomc")
        self.ds = _fesomc_scalar_ds(self.mesh.n_nodes, K=4)

    def _mode(self, **kw):
        from kelp.core.config import ModeConfig
        kw.setdefault("Fdrift2D", False)
        kw.setdefault("Fdrift3D", True)
        return ModeConfig(**kw)

    def test_3d_shape(self):
        out = self.reader.read_scalar_fields(
            self.ds, 0, self.mesh, ["temp"], self._mode())
        assert out["temp"].shape == (3, self.mesh.n_nodes)
        assert out["temp"].dtype == np.float64

    def test_2d_mode_layer_pick(self):
        mode = self._mode(Fdrift2D=True, Fdrift3D=False, sigma_layer=1)
        out = self.reader.read_scalar_fields(self.ds, 0, self.mesh, ["temp"], mode)
        assert out["temp"].shape == (self.mesh.n_nodes,)
        np.testing.assert_array_equal(
            out["temp"], self.ds["temp"].isel(time=0, sigma_lay=1).values)

    def test_2d_mode_thickness_mean(self):
        mode = self._mode(Fdrift2D=True, Fdrift3D=False, sigma_mean=True)
        out = self.reader.read_scalar_fields(self.ds, 0, self.mesh, ["temp"], mode)
        vals = self.ds["temp"].isel(time=0).values          # (K-1, N)
        weights = np.abs(np.diff(self.mesh.sigma))
        expected = np.average(vals, axis=0, weights=weights)
        np.testing.assert_allclose(out["temp"], expected, rtol=1e-12)

    def test_2d_mode_default_surface(self):
        # _reduce_to_2d default (sigma_layer=None, sigma_mean=False) -> row 0
        mode = self._mode(Fdrift2D=True, Fdrift3D=False)
        out = self.reader.read_scalar_fields(self.ds, 0, self.mesh, ["temp"], mode)
        np.testing.assert_array_equal(
            out["temp"], self.ds["temp"].isel(time=0, sigma_lay=0).values)

    def test_missing_raises(self):
        with pytest.raises(ValueError, match="dens"):
            self.reader.read_scalar_fields(
                self.ds, 0, self.mesh, ["dens"], self._mode())

    def test_max_levels_trims_without_mesh_sigma_trim(self):
        # B2: untrimmed mesh.sigma (K=4) + max_levels=3 -> reader-side trim to
        # 2 layers, consistent with read_velocities, without touching
        # mesh.sigma.  temp is a layer field on sigma_lay (K-1=3 layers).
        mode = self._mode(max_levels=3)
        out = self.reader.read_scalar_fields(
            self.ds, 0, self.mesh, ["temp"], mode)
        assert out["temp"].shape == (2, self.mesh.n_nodes)

    @pytest.mark.parametrize("vdim", ["depth", "level"])
    @pytest.mark.parametrize("trim_mesh", [True, False])
    def test_max_levels_depth_dim_layer_field_by_pre_trim_size(
            self, vdim, trim_mesh):
        # A mid-layer field on a non-canonical vertical dim: K_ds - 1 = 3
        # layers next to the dataset's interface dim sigma (K_ds = 4).
        # Under max_levels=3 it keeps 2 layers; before 2026-11 it was
        # compared with the TRIMMED interface count and kept 3 rows.
        N = self.mesh.n_nodes
        vals = np.arange(2 * 3 * N, dtype=np.float64).reshape(2, 3, N)
        iface = np.arange(2 * 4 * N, dtype=np.float64).reshape(2, 4, N)
        other = "level" if vdim == "depth" else "depth"
        ds = self.ds.assign(sigma_lev=(("sigma",), self.mesh.sigma.copy()),
                            tlay=(("time", vdim, "node"), vals),
                            tifc=(("time", other, "node"), iface))
        mesh = self.mesh
        if trim_mesh:
            mesh.sigma = mesh.sigma[:3]
        out = self.reader.read_scalar_fields(
            ds, 1, mesh, ["tlay", "tifc"], self._mode(max_levels=3))
        assert out["tlay"].shape == (2, N)
        np.testing.assert_array_equal(out["tlay"], vals[1, :2])
        assert out["tifc"].shape == (3, N)
        np.testing.assert_array_equal(out["tifc"], iface[1, :3])

    def test_2d_mode_thickness_mean_of_interface_field(self):
        # sigma_mean on an interface field (K rows): adjacent interfaces are
        # averaged to layers, then weighted by the layer thickness.  Before
        # 2026-11 np.average got K values with K-1 weights and raised.
        N = self.mesh.n_nodes
        rng = np.random.default_rng(11)
        vals = rng.normal(size=(2, 4, N))
        ds = self.ds.assign(kv=(("time", "sigma", "node"), vals))
        mode = self._mode(Fdrift2D=True, Fdrift3D=False, sigma_mean=True)
        out = self.reader.read_scalar_fields(ds, 0, self.mesh, ["kv"], mode)
        layers = 0.5 * (vals[0, :-1] + vals[0, 1:])
        expected = np.average(layers, axis=0,
                              weights=np.abs(np.diff(self.mesh.sigma)))
        assert out["kv"].shape == (N,)
        np.testing.assert_allclose(out["kv"], expected, rtol=1e-12)

    def test_read_velocities_attaches(self, tmp_path):
        from types import SimpleNamespace
        nc = tmp_path / "fesomc_3d_scalar.nc"
        _make_fesomc_3d_with_scalar(nc)
        mesh = self.reader.load_mesh(SimpleNamespace(meshpath=str(nc)))
        ds = self.reader.open_dataset(str(nc))
        mode = SimpleNamespace(
            Fdrift2D=False, Fdrift3D=True, Vpuls=False,
            var_u="u", var_v="v", var_w="w", var_eta="eta",
            sigma_layer=None, sigma_mean=False, max_levels=None,
        )
        vel = self.reader.read_velocities(ds, 0, mesh, mode, sample_vars=["temp"])
        ds.close()
        assert set(vel.scalars) == {"temp"}
        assert vel.scalars["temp"].shape == (mesh.sigma.size - 1, mesh.n_nodes)


# ===================================================================
# Writer-side sampled-variable support (TODO 2.4, Task 5)
# ===================================================================

from kelp.writers import TrajectoryWriter, load_trajectories
from kelp.core.particles import make_state


class TestWriterSampled:
    def _state(self, P=3):
        return make_state(np.zeros(P), np.zeros(P))

    @pytest.mark.parametrize("ext", [".nc", ".zarr"])
    def test_roundtrip(self, tmp_path, ext):
        path = str(tmp_path / f"traj{ext}")
        st = self._state()
        attrs = {"temp": {"units": "degC", "long_name": "temperature"}}
        with TrajectoryWriter(path, 3, sample_var_attrs=attrs) as w:
            w.append(st, 0.0, sampled={"temp": np.array([1.0, 2.0, np.nan])})
            w.append(st, 1.0, sampled={"temp": np.array([4.0, 5.0, 6.0])})
        out = load_trajectories(path)
        assert out["temp"].shape == (3, 2)
        np.testing.assert_allclose(out["temp"][0], [1.0, 4.0])
        np.testing.assert_allclose(out["temp"][1], [2.0, 5.0])
        assert np.isnan(out["temp"][2, 0])
        assert out["temp"][2, 1] == 6.0

    def test_reserved_name_raises(self, tmp_path):
        with TrajectoryWriter(str(tmp_path / "t.nc"), 3) as w:
            with pytest.raises(ValueError, match="reserved"):
                w.append(self._state(), 0.0, sampled={"obs": np.zeros(3)})

    def test_restart_prefix_raises(self, tmp_path):
        with TrajectoryWriter(str(tmp_path / "t.nc"), 3) as w:
            with pytest.raises(ValueError, match="reserved"):
                w.append(self._state(), 0.0, sampled={"restart_temp": np.zeros(3)})

    def test_no_sampled_no_extra_vars(self, tmp_path):
        path = str(tmp_path / "t.nc")
        with TrajectoryWriter(path, 3) as w:
            w.append(self._state(), 0.0)
        import netCDF4 as nc4
        with nc4.Dataset(path) as ds:
            assert "temp" not in ds.variables

    @pytest.mark.parametrize("ext", [".nc", ".zarr"])
    def test_attrs_written(self, tmp_path, ext):
        path = str(tmp_path / f"t{ext}")
        attrs = {"temp": {"units": "degC", "long_name": "temperature"}}
        with TrajectoryWriter(path, 2, sample_var_attrs=attrs) as w:
            # First step has no sampling so the sampled var is created LATE
            # (obs_idx>0) and back-fill-resized — for Zarr this exercises the
            # path where a resize on the live handle clobbers attrs written on
            # a transient handle (regression guard).
            w.append(self._state(2), 0.0)
            w.append(self._state(2), 1.0, sampled={"temp": np.zeros(2)})
        if ext == ".nc":
            import netCDF4 as nc4
            with nc4.Dataset(path) as ds:
                assert ds.variables["temp"].units == "degC"
                assert ds.variables["temp"].long_name == "temperature"
        else:
            import zarr
            a = zarr.open_group(path)["temp"]
            assert a.attrs["units"] == "degC"
            assert a.attrs["long_name"] == "temperature"

    def test_checkpoint_does_not_store_sampled(self, tmp_path):
        path = str(tmp_path / "t.nc")
        with TrajectoryWriter(path, 2) as w:
            st = self._state(2)
            w.append(st, 0.0, sampled={"temp": np.zeros(2)})
            w.write_checkpoint(st, 0, 0)
        import netCDF4 as nc4
        with nc4.Dataset(path) as ds:
            restart_vars = [v for v in ds.variables if v.startswith("restart_")]
            assert "restart_temp" not in restart_vars

    @pytest.mark.parametrize("ext", [".nc", ".zarr"])
    def test_late_sampled_backfills_nan(self, tmp_path, ext):
        path = str(tmp_path / f"t{ext}")
        with TrajectoryWriter(path, 2) as w:
            st = self._state(2)
            w.append(st, 0.0)                                    # no sampling yet
            w.append(st, 1.0, sampled={"temp": np.array([7.0, 8.0])})
        out = load_trajectories(path)
        assert out["temp"].shape == (2, 2)
        assert np.isnan(out["temp"][0, 0]) and out["temp"][0, 1] == 7.0

    def test_dormant_kwargs_byte_identical(self, tmp_path):
        # Spec item 14: the dormant sampled-variable plumbing must add NOTHING
        # to a file written without it.  Write two .nc files that differ ONLY
        # in whether the sampled kwargs are mentioned:
        #   (a) TrajectoryWriter(path, P)         + append(state, t)
        #   (b) TrajectoryWriter(path2, P,         + append(state, t, sampled=None)
        #       sample_var_attrs=None)
        # and assert the variable-name set and per-variable values are
        # byte-for-byte identical.  Pins that passing sample_var_attrs=None /
        # sampled=None is a true no-op.
        import netCDF4 as nc4

        P = 3
        st = self._state(P)
        path_a = str(tmp_path / "a.nc")
        path_b = str(tmp_path / "b.nc")

        with TrajectoryWriter(path_a, P) as w:
            w.append(st, 0.0)
            w.append(st, 1.0)
        with TrajectoryWriter(path_b, P, sample_var_attrs=None) as w:
            w.append(st, 0.0, sampled=None)
            w.append(st, 1.0, sampled=None)

        def _filled(x):
            # Float vars may carry masked fill (none expected on this fixed
            # schema, but be safe); int vars (status, nearest_node, restart_*)
            # cannot take a NaN fill -> use a dtype-appropriate sentinel.
            fv = np.nan if np.issubdtype(x.dtype, np.floating) else -1
            return np.ma.filled(x, fv)

        with nc4.Dataset(path_a) as da, nc4.Dataset(path_b) as db:
            assert set(da.variables) == set(db.variables)
            for name in da.variables:
                va = _filled(da.variables[name][:])
                vb = _filled(db.variables[name][:])
                np.testing.assert_array_equal(
                    va, vb, err_msg=f"variable {name!r} differs")


class TestTrajectoryRecordSampled:
    def test_to_arrays_includes_sampled(self):
        from kelp.tracking import TrajectoryRecord
        rec = TrajectoryRecord()
        st = make_state(np.zeros(2), np.zeros(2))
        rec.append(st, 0, sampled={"temp": np.array([1.0, 2.0])})
        rec.append(st, 1, sampled={"temp": np.array([3.0, 4.0])})
        arrs = rec.to_arrays()
        assert arrs["temp"].shape == (2, 2)
        np.testing.assert_allclose(arrs["temp"][:, 0], [1.0, 3.0])

    def test_append_without_sampled_still_works(self):
        from kelp.tracking import TrajectoryRecord
        rec = TrajectoryRecord()
        st = make_state(np.zeros(2), np.zeros(2))
        rec.append(st, 0)
        assert "temp" not in rec.to_arrays()


# ===================================================================
# Tracking-layer integration (TODO 2.4, Task 6)
# ===================================================================


class TestTrackingSampling:
    """End-to-end: run_single_realization with sample_vars."""

    def _setup(self, K=4, n_time=4):
        mesh = _grid_mesh(np.linspace(0, 10, 11), np.linspace(0, 10, 11))
        mesh.sigma = np.array([0.0, 10.0, 20.0, 40.0])[:K]
        reader = get_reader("fesom2")
        ds = _fesom2_ds(mesh.n_nodes, K=K, n_time=n_time)
        for t in range(n_time):
            ds["temp"][t, :, :] = float(t)   # temp constant in space, = t
        return mesh, reader, ds

    def _run(self, mesh, reader, ds, n_steps=2, **kw):
        from kelp.tracking import run_single_realization
        from kelp.core.config import ModeConfig, NumericsConfig
        mode = ModeConfig(Fdrift2D=False, Fdrift3D=True)
        numerics = NumericsConfig(dt=3600.0, integrator="euler")
        init = make_state(np.array([5.0]), np.array([5.0]), np.array([12.0]))
        return run_single_realization(
            mesh, ds, reader, init, mode, numerics,
            start_time_idx=0, n_steps=n_steps,
            sample_vars=["temp"], **kw)

    def test_discrete_mode_samples_end_of_step_record(self):
        # output format 2: the record after a step from record k holds the
        # field of record k+1 — the instant of its label (format 1 sampled
        # the start-of-step record k, one record early)
        mesh, reader, ds = self._setup()
        rec = self._run(mesh, reader, ds)
        arrs = rec.to_arrays()
        np.testing.assert_allclose(arrs["temp"][:, 0], [1.0, 2.0])

    def test_temporal_interp_samples_at_record_time(self):
        mesh, reader, ds = self._setup()
        file_times = np.arange(4) * 7200.0
        rec = self._run(mesh, reader, ds, temporal_interp=True,
                        file_times=file_times, file_dt=7200.0)
        arrs = rec.to_arrays()
        # dt=3600, file_dt=7200: post-step times 3600 (alpha=.5) and 7200 (1.0)
        np.testing.assert_allclose(arrs["temp"][:, 0], [0.5, 1.0])

    def test_dormant_when_unset(self):
        mesh, reader, ds = self._setup()
        from kelp.tracking import run_single_realization
        from kelp.core.config import ModeConfig, NumericsConfig
        mode = ModeConfig(Fdrift2D=False, Fdrift3D=True)
        numerics = NumericsConfig(dt=3600.0, integrator="euler")
        init = make_state(np.array([5.0]), np.array([5.0]), np.array([12.0]))
        rec = run_single_realization(mesh, ds, reader, init, mode, numerics,
                                     start_time_idx=0, n_steps=2)
        assert "temp" not in rec.to_arrays()

    def test_backward_run_not_flipped(self):
        mesh, reader, ds = self._setup()
        from kelp.tracking import run_single_realization
        from kelp.core.config import ModeConfig, NumericsConfig
        mode = ModeConfig(Fdrift2D=False, Fdrift3D=True, time_dir=-1.0)
        numerics = NumericsConfig(dt=3600.0, integrator="euler")
        init = make_state(np.array([5.0]), np.array([5.0]), np.array([12.0]))
        rec = run_single_realization(mesh, ds, reader, init, mode, numerics,
                                     start_time_idx=2, n_steps=2,
                                     sample_vars=["temp"])
        arrs = rec.to_arrays()
        # backward steps from records 2 and 1 end at records 1 and 0 (format
        # 2 samples the end of the step); the values are not sign-flipped
        np.testing.assert_allclose(arrs["temp"][:, 0], [1.0, 0.0])

    def test_space_binding_temp_tracks_lon(self):
        # Spec item 15 (space-binding half): with temp varying LINEARLY in lon
        # (temp[t,k,:] = nodes_lon) and a particle advected by a CONSTANT u,
        # the sampled temp series must equal the particle's own recorded lon
        # trajectory (leastsq is exact for linear fields).  This pins that the
        # sampler reads the field at the *post-step horizontal position*, not a
        # stale/start position.  temp is time-independent here, so the discrete
        # start-of-step snapshot carries the same field at every step.
        mesh, reader, ds = self._setup()
        # _fesom2_ds gives u/v RANDOM on 180 elements; replace with a constant
        # u sized to the actual mesh element count so displacement is
        # deterministic and no element index goes out of bounds.
        E = mesh.n_elements
        K = mesh.sigma.size
        ds = ds.drop_vars(["u", "v"])
        ds["u"] = (("time", "nz1", "elem"),
                   np.full((ds.sizes["time"], K - 1, E), 0.5, np.float32))
        ds["v"] = (("time", "nz1", "elem"),
                   np.zeros((ds.sizes["time"], K - 1, E), np.float32))
        # temp(t, k, node) = nodes_lon (linear in lon, constant in z and time).
        for t in range(ds.sizes["time"]):
            for k in range(K - 1):
                ds["temp"][t, k, :] = mesh.nodes_lon
        rec = self._run(mesh, reader, ds, n_steps=2)
        arrs = rec.to_arrays()
        # Sampled temp at each obs == recorded lon at that obs (same rec arrays).
        np.testing.assert_allclose(
            arrs["temp"][:, 0], arrs["lon"][:, 0], rtol=1e-6)
        # Sanity: the particle actually moved (non-degenerate, space-binding).
        assert arrs["lon"][1, 0] > arrs["lon"][0, 0] + 1e-4

    def test_sigma_3d_e2e_eta_and_column_geometry(self):
        # Spec item 15 (sigma e2e half): drive run_single_realization on a
        # FESOM-C SIGMA mesh with NONZERO eta and a vertically-varying temp
        # field, then verify the sampled value matches the sigma column
        # geometry computed in-test.  Pins TWO things end-to-end:
        #   (1) eta reaches the sampler through the tracking layer, and
        #   (2) sigma column geometry (H = depth + eta) is applied.
        from types import SimpleNamespace
        import tempfile
        import os
        from kelp.tracking import run_single_realization
        from kelp.core.config import ModeConfig, NumericsConfig

        tmpdir = tempfile.mkdtemp()
        nc = os.path.join(tmpdir, "fesomc_sigma_e2e.nc")
        # eta = 2.0 (constant), uniform depth = 10 -> H = 12 everywhere so the
        # column geometry is identical at every node and horizontal blending of
        # the (horizontally-constant) temp is a no-op.  temp[layer] = layer*10
        # (vertically distinct).  u=v=w=0 keeps the particle parked at a fixed
        # (lon, lat, z) on the 11x11 mesh interior.
        eta_val = 2.0
        depth_val = 10.0
        _make_fesomc_sigma_e2e(nc, eta_val=eta_val, depth_val=depth_val)

        reader = get_reader("fesomc")
        mesh = reader.load_mesh(SimpleNamespace(meshpath=nc))
        ds = reader.open_dataset(nc)

        mode = ModeConfig(Fdrift2D=False, Fdrift3D=True)
        numerics = NumericsConfig(dt=3600.0, integrator="euler",
                                  sample_interp="nearest")
        z0 = 6.0
        # (5, 5) = node 60, deep interior of the 11x11 grid.
        init = make_state(np.array([5.0]), np.array([5.0]), np.array([z0]))
        rec = run_single_realization(
            mesh, ds, reader, init, mode, numerics,
            start_time_idx=0, n_steps=2, sample_vars=["temp"])
        ds.close()
        arrs = rec.to_arrays()

        # EXPECTED (replicate _interp_mid_linear_single per recorded obs):
        #   H        = depth + eta
        #   sig_mid  = 0.5*(sigma[m] + sigma[m+1])
        #   z_mid[m] = (1 - sig_mid[m]) * H        (positive-down midpoints)
        #   vertical-linear interp of temp_layer at the recorded z, clamped.
        sig = mesh.sigma                              # [1.0, 0.5, 0.0]
        sig_mid = 0.5 * (sig[:-1] + sig[1:])          # [0.75, 0.25]
        H = depth_val + eta_val                       # 12.0
        z_mid = (1.0 - sig_mid) * H                   # [3.0, 9.0]
        temp_layer = np.arange(sig.size - 1) * 10.0   # [0.0, 10.0]

        def _expect(z):
            if z <= z_mid[0]:
                return temp_layer[0]
            if z >= z_mid[-1]:
                return temp_layer[-1]
            k = int(np.searchsorted(z_mid, z, side="right") - 1)
            a = (z - z_mid[k]) / (z_mid[k + 1] - z_mid[k])
            return (1.0 - a) * temp_layer[k] + a * temp_layer[k + 1]

        zrec = arrs["z"][:, 0]
        expected = np.array([_expect(z) for z in zrec])
        np.testing.assert_allclose(arrs["temp"][:, 0], expected, rtol=1e-9)
        # eta-binding sanity: with eta=0 (H=10) the same z would give a
        # DIFFERENT value, so a nonzero result that matches H=12 pins eta.
        z_mid0 = (1.0 - sig_mid) * depth_val          # [2.5, 7.5]
        a0 = (z0 - z_mid0[0]) / (z_mid0[1] - z_mid0[0])
        eta0_val = (1.0 - a0) * temp_layer[0] + a0 * temp_layer[1]  # 7.0
        assert not np.isclose(expected[0], eta0_val)


def _make_fesomc_sampling_ds(tmp_path):
    """Synthetic 11x11 FESOM-C NetCDF (loadable mesh + 2D velocity) with a
    node-based ``temp(time, node)`` scalar for sampling.

    Mirrors tests/test_cli.py::_make_fesomc_minimal and appends ``temp``.
    """
    from pathlib import Path
    xs = np.linspace(0.0, 10.0, 11)
    ys = np.linspace(0.0, 10.0, 11)
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel().astype(np.float64)
    lat = gy.ravel().astype(np.float64)
    n_nodes = lon.size
    depth = np.full(n_nodes, 10.0, dtype=np.float64)
    elems = []
    nx = ny = 11
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            n1 = n0 + 1
            n2 = n0 + nx
            n3 = n2 + 1
            elems.append([n0 + 1, n1 + 1, n2 + 1, n0 + 1])
            elems.append([n1 + 1, n3 + 1, n2 + 1, n1 + 1])
    nv = np.array(elems, dtype=np.int32)
    n_elem = nv.shape[0]
    elem_nodes_0 = nv[:, :3] - 1
    lon_c = np.array([lon[e].mean() for e in elem_nodes_0], dtype=np.float64)
    lat_c = np.array([lat[e].mean() for e in elem_nodes_0], dtype=np.float64)
    sigma_lev = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    n_time = 3
    u2d = np.full((n_time, n_elem), 0.1, dtype=np.float64)
    v2d = np.zeros((n_time, n_elem), dtype=np.float64)
    eta = np.zeros((n_time, n_nodes), dtype=np.float64)
    temp = np.tile(np.arange(n_nodes, dtype=np.float64), (n_time, 1))
    ds = xr.Dataset({
        "lon": (["node"], lon),
        "lat": (["node"], lat),
        "depth": (["node"], depth),
        "nv": (["nele", "four"], nv),
        "lon_elem": (["nele"], lon_c),
        "lat_elem": (["nele"], lat_c),
        "sigma_lev": (["sigma"], sigma_lev),
        "u2d": (["time", "nele"], u2d),
        "v2d": (["time", "nele"], v2d),
        "eta": (["time", "node"], eta),
        "temp": (["time", "node"], temp),
    })
    nc_path = Path(tmp_path) / "fesomc_sampling.nc"
    ds.to_netcdf(nc_path)
    return nc_path


class TestSamplingIntegrationEdges:
    """Cycle, restart-resume, and multi-realization sampling semantics."""

    def _setup(self, K=4, n_time=4):
        mesh = _grid_mesh(np.linspace(0, 10, 11), np.linspace(0, 10, 11))
        mesh.sigma = np.array([0.0, 10.0, 20.0, 40.0])[:K]
        reader = get_reader("fesom2")
        ds = _fesom2_ds(mesh.n_nodes, K=K, n_time=n_time)
        for t in range(n_time):
            ds["temp"][t, :, :] = float(t)   # temp constant in space, = t
        return mesh, reader, ds

    def test_cycle_mode_samples_end_of_step_record(self):
        from kelp.tracking import run_single_realization
        from kelp.core.config import ModeConfig, NumericsConfig
        mesh, reader, ds = self._setup(n_time=4)
        mode = ModeConfig(Fdrift2D=False, Fdrift3D=True)
        numerics = NumericsConfig(dt=3600.0, integrator="euler")
        init = make_state(np.array([5.0]), np.array([5.0]), np.array([12.0]))
        # 6 discrete steps over 4 snapshots with wrap: the steps start at
        # records 0,1,2,3,0,1 and END at 1,2,3,0,1,2 -> format 2 samples the
        # end-of-step record (temp[t]=t), wrapped like the reads.
        rec = run_single_realization(
            mesh, ds, reader, init, mode, numerics,
            start_time_idx=0, n_steps=6, n_time_total=4,
            on_exhausted="cycle", sample_vars=["temp"])
        arrs = rec.to_arrays()
        np.testing.assert_allclose(arrs["temp"][:, 0], [1.0, 2.0, 3.0, 0.0, 1.0, 2.0])

    def test_restart_resume_sampled_series_continuous(self, tmp_path):
        # PATTERN: tests/test_tracking_diffusion.py::TestRestartBitwiseAtTrackingLayer
        from kelp.tracking import run_single_realization
        from kelp.writers import TrajectoryWriter, load_restart, load_trajectories
        from kelp.core.config import ModeConfig, NumericsConfig

        mesh, reader, ds = self._setup(n_time=4)
        mode = ModeConfig(Fdrift2D=False, Fdrift3D=True)
        numerics = NumericsConfig(dt=3600.0, integrator="euler", output_freq=1)
        init = make_state(np.array([5.0]), np.array([5.0]), np.array([12.0]))

        # Continuous: 4 steps from records 0..3 -> format 2 samples the
        # end-of-step records 1,2,3 and, for the last step (no record 4),
        # the start record 3 -> temp [1,2,3,3].
        cont_path = str(tmp_path / "cont.nc")
        with TrajectoryWriter(cont_path, 1) as w:
            run_single_realization(
                mesh, ds, reader, init, mode, numerics,
                start_time_idx=0, n_steps=4, n_time_total=4,
                writer=w, sample_vars=["temp"])

        # First leg: 2 steps with a checkpoint, then resume the remaining 2.
        split_path = str(tmp_path / "split.nc")
        with TrajectoryWriter(split_path, 1) as w:
            run_single_realization(
                mesh, ds, reader, init, mode, numerics,
                start_time_idx=0, n_steps=2, n_time_total=4,
                writer=w, checkpoint_freq=2, sample_vars=["temp"])
        rs = load_restart(split_path)
        resume_path = str(tmp_path / "resume.nc")
        with TrajectoryWriter(resume_path, 1) as w:
            run_single_realization(
                mesh, ds, reader, rs["state"], mode, numerics,
                start_time_idx=rs["file_idx"] + 1, n_steps=2, n_time_total=4,
                writer=w, start_step_i=rs["step_i"] + 1, sample_vars=["temp"])
        ds.close()

        cont = load_trajectories(cont_path)
        resumed = load_trajectories(resume_path)
        # Resumed file's sampled obs == continuous run's tail (obs 2,3).
        np.testing.assert_allclose(resumed["temp"][0], cont["temp"][0, 2:])
        np.testing.assert_allclose(resumed["temp"][0], [3.0, 3.0])

    def test_multi_realization_files_carry_sampled(self, tmp_path):
        # PATTERN: tests/test_cli.py config builders + num_runs realizations.
        from kelp.core.config import load_config
        from kelp.tracking import run_tracking
        from kelp.writers import load_trajectories

        nc_path = _make_fesomc_sampling_ds(tmp_path)
        particles = tmp_path / "p.dat"
        particles.write_text("5.0 5.0\n5.0 5.0\n5.0 5.0\n")
        out = tmp_path / "traj.nc"
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(f"""\
mode:
  Fdrift2D: true
  time_dir: 1.0
  sample_vars: [temp]
numerics:
  dt: 100.0
  integrator: euler
  output_freq: 1
  sample_interp: leastsq
mesh:
  type: fesomc
  meshpath: {nc_path}
data:
  ncinput: {nc_path}
  inicloud: {particles}
  output_path: {out}
  time_averaging: instant
time:
  start_time_idx: 0
  n_steps: 2
  num_runs: 2
""")
        cfg = load_config(str(cfg_path))
        run_tracking(cfg)
        for ire in range(2):
            f = tmp_path / f"traj_r{ire}.nc"
            assert f.exists()
            arrs = load_trajectories(str(f))
            assert "temp" in arrs
            assert np.isfinite(arrs["temp"]).all()


# ---------------------------------------------------------------------------
# NaN finite-filtering in the horizontal stencils (review finding 2026-06-05)
# ---------------------------------------------------------------------------

class TestNaNStencilFiltering:
    """One non-finite neighbour (decoded _FillValue at the coast/bottom)
    must not poison leastsq/idw/barycentric to NaN: non-finite neighbours
    are skipped, with minimum-point thresholds (leastsq >= 3, idw >= 1;
    a barycentric vertex NaN falls back to the nearest value).  The
    nearest value itself is returned raw — if the particle's own node is
    dry, the honest answer is that node's fill."""

    @pytest.fixture(scope="class")
    def mesh(self):
        return _grid_mesh(np.linspace(0, 10, 11), np.linspace(0, 10, 11))

    def _poison(self, mesh, lon, lat, k_far=1):
        """Linear field with the k_far-th 1-ring neighbour of the query
        point's nearest node set to NaN (never the nearest itself)."""
        from kelp.core.mesh import find_nearest_nodes
        from kelp.numerics.vertical import _collect_neighbour_nodes
        a, b, c = 2.0, 0.3, -0.7
        vals = a + b * mesh.nodes_lon + c * mesh.nodes_lat
        nn = int(find_nearest_nodes(mesh, np.array([lon]),
                                    np.array([lat]))[0])
        buf, cnt = _collect_neighbour_nodes(
            nn, mesh.node_to_elems, mesh.node_to_elems_count,
            mesh.elem_nodes)
        vals[buf[k_far]] = np.nan          # a ring member != nearest
        return vals, (a, b, c), int(buf[k_far])

    def test_leastsq_survives_one_nan_neighbour(self, mesh):
        lon, lat = 4.3, 5.6
        vals, (a, b, c), _ = self._poison(mesh, lon, lat)
        out = sample_field(vals, mesh, np.array([lon]), np.array([lat]),
                           method="leastsq")
        # plane fit of the REMAINING finite ring of a linear field is
        # still exact
        np.testing.assert_allclose(out[0], a + b * lon + c * lat,
                                   rtol=1e-9)

    def test_idw_survives_one_nan_neighbour(self, mesh):
        lon, lat = 4.3, 5.6
        vals, _, poisoned = self._poison(mesh, lon, lat)
        vals[np.isfinite(vals)] = 7.5      # constant among the survivors
        out = sample_field(vals, mesh, np.array([lon]), np.array([lat]),
                           method="idw")
        np.testing.assert_allclose(out[0], 7.5, rtol=1e-12)

    def test_barycentric_nan_vertex_falls_back_to_nearest(self, mesh):
        from kelp.core.mesh import find_nearest_nodes
        lon, lat = 4.3, 5.6
        a, b, c = 2.0, 0.3, -0.7
        vals = a + b * mesh.nodes_lon + c * mesh.nodes_lat
        # poison EVERY ring member except the nearest node -> whatever
        # element contains the point has a NaN vertex
        from kelp.numerics.vertical import _collect_neighbour_nodes
        nn = int(find_nearest_nodes(mesh, np.array([lon]),
                                    np.array([lat]))[0])
        buf, cnt = _collect_neighbour_nodes(
            nn, mesh.node_to_elems, mesh.node_to_elems_count,
            mesh.elem_nodes)
        for i in range(1, cnt):
            vals[buf[i]] = np.nan
        out = sample_field(vals, mesh, np.array([lon]), np.array([lat]),
                           method="barycentric")
        np.testing.assert_allclose(out[0], vals[nn], rtol=1e-12)

    def test_leastsq_below_threshold_falls_back(self, mesh):
        """< 3 finite ring members -> nearest-value fallback."""
        from kelp.core.mesh import find_nearest_nodes
        from kelp.numerics.vertical import _collect_neighbour_nodes
        lon, lat = 4.3, 5.6
        vals = np.full(mesh.n_nodes, 3.25)
        nn = int(find_nearest_nodes(mesh, np.array([lon]),
                                    np.array([lat]))[0])
        buf, cnt = _collect_neighbour_nodes(
            nn, mesh.node_to_elems, mesh.node_to_elems_count,
            mesh.elem_nodes)
        for i in range(2, cnt):            # keep nearest + one other
            vals[buf[i]] = np.nan
        out = sample_field(vals, mesh, np.array([lon]), np.array([lat]),
                           method="leastsq")
        np.testing.assert_allclose(out[0], 3.25, rtol=1e-12)

    def test_nearest_returns_raw_even_if_nan(self, mesh):
        from kelp.core.mesh import find_nearest_nodes
        lon, lat = 4.3, 5.6
        vals = np.full(mesh.n_nodes, 1.0)
        nn = int(find_nearest_nodes(mesh, np.array([lon]),
                                    np.array([lat]))[0])
        vals[nn] = np.nan
        out = sample_field(vals, mesh, np.array([lon]), np.array([lat]),
                           method="nearest")
        assert np.isnan(out[0])

    def test_idw_coincident_with_nan_node_returns_raw(self, mesh):
        """A particle sitting EXACTLY on a fill node must get that raw
        fill, not a finite blend of the remaining ring (review round 2:
        blending would invent water on a dry spot)."""
        vals = np.full(mesh.n_nodes, 7.0)
        # node at (4, 5) — grid nodes are integer lon/lat
        from kelp.core.mesh import find_nearest_nodes
        nn = int(find_nearest_nodes(mesh, np.array([4.0]),
                                    np.array([5.0]))[0])
        vals[nn] = np.nan
        out = sample_field(vals, mesh, np.array([4.0]), np.array([5.0]),
                           method="idw")
        assert np.isnan(out[0])
