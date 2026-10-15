"""Field plotting on the mesh (plot_field) and particle density on mesh
faces (compute_presence_density_mesh) — the TODO 3.5 tails.

plot_field renders a scalar per NODE or per ELEMENT as colored mesh
faces (node scalars on an all-triangle mesh use smooth Gouraud
tripcolor; anything else flat PolyCollection).  The density statistic
maps trajectory samples onto mesh ELEMENTS via the real containment
machinery — the unstructured twin of the uniform-grid
compute_presence_probability.
"""
import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from kelp.core.mesh import Mesh, build_node_to_elems, build_search
from kelp.core.particles import ParticleStatus
from kelp.core.rotation import euler_matrix, rotate_lonlat
from kelp.visualization.core import TrajectoryDataset
from kelp.visualization.plot_2d import plot_field
from kelp.visualization.stats import compute_presence_density_mesh


def _tri_mesh(nx=5, span=1.0, lon0=0.0):
    xs = lon0 + np.linspace(0.0, span, nx)
    ys = np.linspace(0.0, span, nx)
    gx, gy = np.meshgrid(xs, ys)
    lon = (((gx.ravel() + 180.0) % 360.0) - 180.0).astype(np.float64)
    lat = gy.ravel().astype(np.float64)
    elems = []
    for j in range(nx - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            elems.append([n0, n0 + 1, n0 + nx])
            elems.append([n0 + 1, n0 + nx + 1, n0 + nx])
    elem_nodes = np.array(elems, dtype=np.int32)
    n2e, cnt = build_node_to_elems(nx * nx, elem_nodes)

    def _ecl(e):
        ref = lon[e[0]]
        d = ((lon[e] - ref + 180.0) % 360.0) - 180.0
        return ((ref + d.mean() + 180.0) % 360.0) - 180.0

    mesh = Mesh(name="tri", nodes_lon=lon, nodes_lat=lat,
                depth=np.full(nx * nx, 50.0), elem_nodes=elem_nodes,
                elem_center_lon=np.array([_ecl(e) for e in elem_nodes]),
                elem_center_lat=np.array([lat[e].mean() for e in elem_nodes]),
                node_to_elems=n2e, node_to_elems_count=cnt)
    build_search(mesh)
    return mesh


def _quad_mesh(nx=4):
    xs = np.linspace(0.0, 1.0, nx)
    gx, gy = np.meshgrid(xs, xs)
    lon = gx.ravel().astype(np.float64)
    lat = gy.ravel().astype(np.float64)
    elems = []
    for j in range(nx - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            elems.append([n0, n0 + 1, n0 + nx + 1, n0 + nx])
    elem_nodes = np.array(elems, dtype=np.int32)
    n2e, cnt = build_node_to_elems(nx * nx, elem_nodes)
    mesh = Mesh(name="quad", nodes_lon=lon, nodes_lat=lat,
                depth=np.full(nx * nx, 50.0), elem_nodes=elem_nodes,
                elem_center_lon=np.array([lon[e].mean() for e in elem_nodes]),
                elem_center_lat=np.array([lat[e].mean() for e in elem_nodes]),
                node_to_elems=n2e, node_to_elems_count=cnt)
    build_search(mesh)
    return mesh


DEG2RAD_ = np.pi / 180.0


def _band_tri_mesh(n_lon=72, n_lat=3, lat0=40.0, lat1=50.0):
    """GLOBAL cyclic tri band (includes the seam column): no contiguous
    350-degree lon branch exists, so the display keeps raw coordinates
    and masks the over-wide seam faces."""
    lons = -180.0 + 360.0 * np.arange(n_lon) / n_lon
    lats = np.linspace(lat0, lat1, n_lat)
    gx, gy = np.meshgrid(lons, lats)
    lon = gx.ravel().astype(np.float64)
    lat = gy.ravel().astype(np.float64)
    elems = []
    for j in range(n_lat - 1):
        for i in range(n_lon):
            i2 = (i + 1) % n_lon
            n0 = j * n_lon + i
            n1 = j * n_lon + i2
            n2 = (j + 1) * n_lon + i
            n3 = (j + 1) * n_lon + i2
            elems.append([n0, n1, n2])
            elems.append([n1, n3, n2])
    elem_nodes = np.array(elems, dtype=np.int32)
    n2e, cnt = build_node_to_elems(lon.size, elem_nodes)
    mesh = Mesh(name="band", nodes_lon=lon, nodes_lat=lat,
                depth=np.full(lon.size, 100.0), elem_nodes=elem_nodes,
                elem_center_lon=np.zeros(len(elems)),
                elem_center_lat=np.array(
                    [lat[e].mean() for e in elem_nodes]),
                node_to_elems=n2e, node_to_elems_count=cnt)
    build_search(mesh)
    return mesh


def _collections_of(ax):
    return [c for c in ax.collections]


class TestPlotField:
    def teardown_method(self):
        plt.close("all")

    def test_element_values_polycollection(self):
        mesh = _tri_mesh()
        vals = np.arange(mesh.n_elements, dtype=float)
        fig, ax = plt.subplots()
        out = plot_field(mesh, vals, ax=ax, add_colorbar=False)
        assert out is ax
        pcs = [c for c in _collections_of(ax) if isinstance(c, PolyCollection)]
        assert len(pcs) == 1
        arr = np.asarray(pcs[0].get_array())
        np.testing.assert_allclose(arr, vals)

    def test_node_values_tri_mesh_gouraud(self):
        """All-triangle mesh + node scalar -> smooth tripcolor whose
        array is the NODE values (a flat fallback would carry per-FACE
        values instead — the discriminating property)."""
        mesh = _tri_mesh()
        vals = mesh.nodes_lon + 2.0 * mesh.nodes_lat
        fig, ax = plt.subplots()
        out = plot_field(mesh, vals, ax=ax, add_colorbar=False)
        assert out is ax
        cols = _collections_of(ax)
        assert len(cols) == 1
        assert not isinstance(cols[0], PolyCollection)   # not the flat path
        arr = np.asarray(cols[0].get_array()).ravel()
        assert arr.size == mesh.n_nodes                  # gouraud = node array
        np.testing.assert_allclose(arr, vals)

    def test_node_values_quad_mesh_flat_fallback(self):
        """Quads cannot Gouraud: node scalar -> per-face vertex mean."""
        mesh = _quad_mesh()
        vals = np.arange(mesh.n_nodes, dtype=float)
        fig, ax = plt.subplots()
        plot_field(mesh, vals, ax=ax, add_colorbar=False)
        pcs = [c for c in _collections_of(ax) if isinstance(c, PolyCollection)]
        assert len(pcs) == 1
        arr = np.asarray(pcs[0].get_array())
        expected = np.array([vals[[int(n) for n in e if n >= 0]].mean()
                             for e in mesh.elem_nodes])
        np.testing.assert_allclose(arr, expected)

    def test_gouraud_on_quads_raises(self):
        mesh = _quad_mesh()
        vals = np.zeros(mesh.n_nodes)
        with pytest.raises(ValueError, match="gouraud"):
            plot_field(mesh, vals, shading="gouraud", add_colorbar=False)

    def test_wrong_size_raises(self):
        mesh = _tri_mesh()
        with pytest.raises(ValueError, match="values"):
            plot_field(mesh, np.zeros(7), add_colorbar=False)

    def test_colorbar_added(self):
        mesh = _tri_mesh()
        vals = np.arange(mesh.n_elements, dtype=float)
        fig, ax = plt.subplots()
        plot_field(mesh, vals, ax=ax, add_colorbar=True, cbar_label="speed")
        assert len(ax.figure.axes) == 2          # main axes + colorbar

    def test_compact_dateline_mesh_draws_contiguous(self):
        """A COMPACT regional mesh straddling +/-180 unwraps to its own
        branch: every face drawn, tight extent around lon 180 (no
        world-wide smear, nothing masked)."""
        mesh = _tri_mesh(lon0=179.5, span=1.0)   # crosses the dateline
        vals = np.arange(mesh.n_elements, dtype=float)
        fig, ax = plt.subplots()
        plot_field(mesh, vals, ax=ax, add_colorbar=False)
        pcs = [c for c in _collections_of(ax) if isinstance(c, PolyCollection)]
        assert len(pcs[0].get_paths()) == mesh.n_elements
        verts = np.concatenate([p.vertices for p in pcs[0].get_paths()])
        assert 179.0 < verts[:, 0].min() and verts[:, 0].max() < 181.0
        lo, hi = ax.get_xlim()
        assert hi - lo < 5.0                     # tight, not world-wide

    def test_global_band_seam_faces_masked_flat(self):
        """On a GLOBAL mesh (no contiguous branch) the seam-column faces
        are masked from the flat display — exactly those, no more."""
        mesh = _band_tri_mesh(n_lat=3)
        vals = np.arange(mesh.n_elements, dtype=float)
        fig, ax = plt.subplots()
        plot_field(mesh, vals, ax=ax, add_colorbar=False)
        pcs = [c for c in _collections_of(ax) if isinstance(c, PolyCollection)]
        n_seam = 2 * 2                           # 2 tris x (n_lat-1) row pairs
        assert len(pcs[0].get_paths()) == mesh.n_elements - n_seam

    def test_global_band_gouraud_masks_seam_triangles(self):
        """Gouraud path on a global all-tri band: node array intact,
        seam triangles masked via the Triangulation."""
        mesh = _band_tri_mesh(n_lat=3)
        vals = np.cos(mesh.nodes_lon * DEG2RAD_) + mesh.nodes_lat
        fig, ax = plt.subplots()
        plot_field(mesh, vals, ax=ax, add_colorbar=False)
        cols = _collections_of(ax)
        assert len(cols) == 1 and not isinstance(cols[0], PolyCollection)
        arr = np.asarray(cols[0].get_array()).ravel()
        assert arr.size == mesh.n_nodes

    def test_masked_input_values_stay_masked(self):
        """A MASKED (finite) node value must not be painted: the faces
        touching it come out masked in the flat fallback."""
        mesh = _quad_mesh()
        vals = np.ma.asarray(np.zeros(mesh.n_nodes))
        vals[5] = 7.0
        vals[5] = np.ma.masked
        fig, ax = plt.subplots()
        plot_field(mesh, vals, ax=ax, add_colorbar=False)
        pcs = [c for c in _collections_of(ax) if isinstance(c, PolyCollection)]
        arr = pcs[0].get_array()
        touching = [i for i, (k, f) in enumerate(
            [(k, [int(n) for n in e if n >= 0])
             for k, e in enumerate(mesh.elem_nodes)]) if 5 in f]
        assert np.ma.getmaskarray(arr)[touching].all()
        others = [i for i in range(len(arr)) if i not in touching]
        assert not np.ma.getmaskarray(arr)[others].any()

    def test_2d_values_rejected(self):
        mesh = _tri_mesh()
        vals = np.zeros((mesh.n_elements, 1))
        with pytest.raises(ValueError, match="1-D"):
            plot_field(mesh, vals, add_colorbar=False)

    def test_layout_validation(self):
        mesh = _tri_mesh()
        with pytest.raises(ValueError, match="layout"):
            plot_field(mesh, np.zeros(mesh.n_elements), layout="banana",
                       add_colorbar=False)
        with pytest.raises(ValueError, match="expected"):
            plot_field(mesh, np.zeros(mesh.n_elements), layout="node",
                       add_colorbar=False)

    def test_ambiguous_size_requires_explicit_layout(self):
        mesh = _band_tri_mesh(n_lat=2)           # N == E by construction
        assert mesh.n_nodes == mesh.n_elements
        vals = np.zeros(mesh.n_nodes)
        with pytest.raises(ValueError, match="layout='node'"):
            plot_field(mesh, vals, add_colorbar=False)
        fig, ax = plt.subplots()
        out = plot_field(mesh, vals, layout="element", ax=ax,
                         add_colorbar=False)
        assert out is ax

    def test_vmin_vmax_and_norm(self):
        import matplotlib.colors as mcolors
        mesh = _tri_mesh()
        vals = np.arange(mesh.n_elements, dtype=float) + 1.0
        fig, ax = plt.subplots()
        plot_field(mesh, vals, ax=ax, vmin=2.0, vmax=9.0,
                   add_colorbar=False)
        pcs = [c for c in _collections_of(ax) if isinstance(c, PolyCollection)]
        assert pcs[0].get_clim() == (2.0, 9.0)
        fig2, ax2 = plt.subplots()
        ln = mcolors.LogNorm(vmin=1.0, vmax=vals.max())
        plot_field(mesh, vals, ax=ax2, norm=ln, add_colorbar=False)
        pcs2 = [c for c in _collections_of(ax2)
                if isinstance(c, PolyCollection)]
        assert pcs2[0].norm is ln
        with pytest.raises(ValueError, match="norm"):
            plot_field(mesh, vals, norm=ln, vmin=0.0, add_colorbar=False)

    def test_flat_shading_on_tri_mesh(self):
        """shading='flat' forces the PolyCollection path even where
        Gouraud is possible."""
        mesh = _tri_mesh()
        vals = np.zeros(mesh.n_nodes)
        fig, ax = plt.subplots()
        plot_field(mesh, vals, shading="flat", ax=ax, add_colorbar=False)
        pcs = [c for c in _collections_of(ax) if isinstance(c, PolyCollection)]
        assert len(pcs) == 1

    def test_degenerate_and_fesomc_encodings(self):
        """FESOM-C encodings + degenerate rows: [a,b,c,-1] and [a,b,c,a]
        are triangles (node mean over 3 verts, not 4), an all-(-1) row
        and a 2-vertex row are skipped without crashing, and element
        values stay aligned with the surviving faces."""
        lon = np.array([0.0, 1.0, 0.0, 1.0, 2.0, 2.0])
        lat = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
        elem_nodes = np.array([
            [0, 1, 2, -1],       # tri, -1 pad
            [1, 3, 2, 1],        # tri, repeated first vertex
            [1, 4, 5, 3],        # true quad
            [-1, -1, -1, -1],    # degenerate: skipped
            [0, 1, -1, -1],      # degenerate 2-vertex: skipped
        ], dtype=np.int32)
        n2e, cnt = build_node_to_elems(6, elem_nodes)
        mesh = Mesh(name="mix", nodes_lon=lon, nodes_lat=lat,
                    depth=np.full(6, 10.0), elem_nodes=elem_nodes,
                    elem_center_lon=np.zeros(5),
                    elem_center_lat=np.zeros(5),
                    node_to_elems=n2e, node_to_elems_count=cnt)
        node_vals = np.arange(6, dtype=float)
        fig, ax = plt.subplots()
        plot_field(mesh, node_vals, ax=ax, add_colorbar=False)
        pcs = [c for c in _collections_of(ax) if isinstance(c, PolyCollection)]
        arr = np.asarray(pcs[0].get_array())
        assert len(arr) == 3                     # two degenerates skipped
        np.testing.assert_allclose(
            arr, [node_vals[[0, 1, 2]].mean(),   # 3 verts, not 4
                  node_vals[[1, 3, 2]].mean(),
                  node_vals[[1, 4, 5, 3]].mean()])
        # element values align with surviving faces too
        fig2, ax2 = plt.subplots()
        evals = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        plot_field(mesh, evals, ax=ax2, add_colorbar=False)
        pcs2 = [c for c in _collections_of(ax2)
                if isinstance(c, PolyCollection)]
        np.testing.assert_allclose(np.asarray(pcs2[0].get_array()),
                                   [10.0, 20.0, 30.0])

    def test_cartopy_axes_smoke(self):
        pytest.importorskip("cartopy")
        mesh = _tri_mesh()
        vals = np.arange(mesh.n_elements, dtype=float)
        ax = plot_field(mesh, vals, add_colorbar=True, cbar_label="x")
        assert hasattr(ax, "coastlines")         # GeoAxes path taken

    def test_rotated_mesh_displays_geographic(self):
        """Rotated-frame mesh draws at geographic positions
        (display_coords), matching the trajectory overlay."""
        mesh = _tri_mesh()
        R = euler_matrix(50.0, 15.0, -90.0)
        rlon, rlat = rotate_lonlat(mesh.nodes_lon, mesh.nodes_lat, R)
        rot = Mesh(name="rot", nodes_lon=rlon, nodes_lat=rlat,
                   depth=mesh.depth, elem_nodes=mesh.elem_nodes,
                   elem_center_lon=mesh.elem_center_lon,
                   elem_center_lat=mesh.elem_center_lat,
                   node_to_elems=mesh.node_to_elems,
                   node_to_elems_count=mesh.node_to_elems_count,
                   rotation_angles=(50.0, 15.0, -90.0))
        vals = np.arange(rot.n_elements, dtype=float)
        fig, ax = plt.subplots()
        plot_field(rot, vals, ax=ax, add_colorbar=False)
        pcs = [c for c in _collections_of(ax) if isinstance(c, PolyCollection)]
        verts = np.concatenate([p.vertices for p in pcs[0].get_paths()])
        # geographic frame of this mesh is lon in [0,1], lat in [0,1]
        assert verts[:, 0].min() > -1.0 and verts[:, 0].max() < 2.0
        assert verts[:, 1].min() > -1.0 and verts[:, 1].max() < 2.0


def _make_ds(lon, lat, status=None):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if status is None:
        status = np.full(lon.shape, int(ParticleStatus.ALIVE), dtype=np.int32)
    z = np.zeros_like(lon)
    time = np.arange(lon.shape[1], dtype=float)
    return TrajectoryDataset(None, {"lon": lon, "lat": lat, "z": z,
                                    "status": status, "time": time})


class TestPresenceDensityMesh:
    def test_counts_land_in_containing_element(self):
        mesh = _tri_mesh()
        # all samples inside element 0's area: tiny cluster near its centroid
        cx = mesh.elem_center_lon[0]
        cy = mesh.elem_center_lat[0]
        lon = np.full((3, 4), cx) + np.linspace(-1e-3, 1e-3, 4)
        lat = np.full((3, 4), cy) + np.linspace(-1e-3, 1e-3, 4)[None, :]
        ds = _make_ds(lon, lat)
        dens = compute_presence_density_mesh(ds, mesh, normalize="count")
        assert dens.shape == (mesh.n_elements,)
        assert dens.sum() == 12
        assert dens[0] == 12

    def test_probability_normalization(self):
        mesh = _tri_mesh()
        cx, cy = mesh.elem_center_lon[3], mesh.elem_center_lat[3]
        ds = _make_ds(np.full((2, 5), cx), np.full((2, 5), cy))
        prob = compute_presence_density_mesh(ds, mesh,
                                             normalize="probability")
        assert prob[3] == pytest.approx(1.0)
        assert prob.sum() == pytest.approx(1.0)

    def test_per_area_normalization(self):
        mesh = _tri_mesh()
        cx, cy = mesh.elem_center_lon[3], mesh.elem_center_lat[3]
        ds = _make_ds(np.full((2, 5), cx), np.full((2, 5), cy))
        prob = compute_presence_density_mesh(ds, mesh,
                                             normalize="probability")
        dens = compute_presence_density_mesh(ds, mesh, normalize="per_area")
        from kelp.core.mesh import build_elem_areas
        areas = build_elem_areas(mesh)
        np.testing.assert_allclose(dens, prob / areas)

    def test_dead_and_nan_samples_excluded(self):
        mesh = _tri_mesh()
        cx, cy = mesh.elem_center_lon[0], mesh.elem_center_lat[0]
        lon = np.array([[cx, cx, np.nan]])
        lat = np.array([[cy, cy, np.nan]])
        status = np.array([[int(ParticleStatus.ALIVE),
                            int(ParticleStatus.OUTSIDE),
                            int(ParticleStatus.ALIVE)]], dtype=np.int32)
        ds = _make_ds(lon, lat, status)
        cnt = compute_presence_density_mesh(ds, mesh, normalize="count")
        assert cnt.sum() == 1                    # only the live finite sample
        cnt_all = compute_presence_density_mesh(ds, mesh, normalize="count",
                                                alive_only=False)
        assert cnt_all.sum() == 2                # NaN still excluded

    def test_off_mesh_samples_ignored(self):
        mesh = _tri_mesh()
        ds = _make_ds(np.array([[10.0, mesh.elem_center_lon[0]]]),
                      np.array([[10.0, mesh.elem_center_lat[0]]]))
        cnt = compute_presence_density_mesh(ds, mesh, normalize="count")
        assert cnt.sum() == 1
        # probability keeps ALL valid samples in the denominator (mirror
        # of the uniform-grid twin): 1 in-mesh hit / 2 valid samples
        prob = compute_presence_density_mesh(ds, mesh,
                                             normalize="probability")
        assert prob.sum() == pytest.approx(0.5)

    def test_rotated_mesh_takes_geographic_trajectories(self):
        """Trajectory output is GEOGRAPHIC; a rotated-frame mesh must
        rotate the samples into its own frame before containment."""
        mesh = _tri_mesh()
        cx, cy = mesh.elem_center_lon[5], mesh.elem_center_lat[5]
        ds = _make_ds(np.full((1, 6), cx), np.full((1, 6), cy))
        ref = compute_presence_density_mesh(ds, mesh, normalize="count")

        R = euler_matrix(50.0, 15.0, -90.0)
        rlon, rlat = rotate_lonlat(mesh.nodes_lon, mesh.nodes_lat, R)
        recl, recla = rotate_lonlat(mesh.elem_center_lon,
                                    mesh.elem_center_lat, R)
        rot = Mesh(name="rot", nodes_lon=rlon, nodes_lat=rlat,
                   depth=mesh.depth, elem_nodes=mesh.elem_nodes,
                   elem_center_lon=recl, elem_center_lat=recla,
                   node_to_elems=mesh.node_to_elems,
                   node_to_elems_count=mesh.node_to_elems_count,
                   rotation_angles=(50.0, 15.0, -90.0))
        build_search(rot)
        # SAME geographic samples -> same per-element counts
        out = compute_presence_density_mesh(ds, rot, normalize="count")
        np.testing.assert_array_equal(out, ref)

    def test_empty_dataset(self):
        mesh = _tri_mesh()
        ds = _make_ds(np.zeros((0, 0)), np.zeros((0, 0)),
                      status=np.zeros((0, 0), dtype=np.int32))
        out = compute_presence_density_mesh(ds, mesh, normalize="count")
        assert out.shape == (mesh.n_elements,)
        assert out.sum() == 0

    def test_bad_normalize_raises(self):
        mesh = _tri_mesh()
        ds = _make_ds(np.zeros((1, 1)), np.zeros((1, 1)))
        with pytest.raises(ValueError, match="normalize"):
            compute_presence_density_mesh(ds, mesh, normalize="banana")

    def test_return_stats_reports_off_mesh_fraction(self):
        """Off-mesh losses are distinguishable from domain exits."""
        mesh = _tri_mesh()
        ds = _make_ds(np.array([[10.0, mesh.elem_center_lon[0]]]),
                      np.array([[10.0, mesh.elem_center_lat[0]]]))
        cnt, stats = compute_presence_density_mesh(
            ds, mesh, normalize="count", return_stats=True)
        assert stats == {"total_valid": 2, "in_mesh": 1}
        assert cnt.sum() == 1
        # empty dataset returns zeroed stats, same tuple shape
        ds0 = _make_ds(np.zeros((0, 0)), np.zeros((0, 0)),
                       status=np.zeros((0, 0), dtype=np.int32))
        out0, stats0 = compute_presence_density_mesh(
            ds0, mesh, normalize="count", return_stats=True)
        assert stats0 == {"total_valid": 0, "in_mesh": 0}

    def test_per_area_zero_area_element_guarded(self):
        """A degenerate (zero-area) element yields 0 density, not NaN."""
        mesh = _tri_mesh()
        from kelp.core.mesh import build_elem_areas
        areas = build_elem_areas(mesh)
        areas[1] = 0.0                      # poison the lazy cache
        cx, cy = mesh.elem_center_lon[0], mesh.elem_center_lat[0]
        ds = _make_ds(np.full((1, 3), cx), np.full((1, 3), cy))
        dens = compute_presence_density_mesh(ds, mesh,
                                             normalize="per_area")
        assert np.isfinite(dens).all()
        assert dens[1] == 0.0
