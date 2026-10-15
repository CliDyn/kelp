"""Load-time quad convexity check (mesh contract made loud).

The product-form Wachspress weights assume convex, non-degenerate quads (a
MESH CONTRACT since 2026-07 — a violating quad silently produces NaN or
garbage weights).  ``find_nonconvex_quads`` makes the contract checkable in
O(E) vectorized time; the FESOM-C reader and ``build_structured_mesh`` run
it at load and fail loud with the offending element ids.
"""
import numpy as np
import pytest

from kelp.core.mesh import find_nonconvex_quads


def _elem(rows):
    return np.asarray(rows, dtype=np.int32)


class TestFindNonconvexQuads:
    def test_convex_quads_pass(self):
        lon = np.array([0.0, 1.0, 1.1, -0.1, 2.0, 3.0, 3.0, 2.0])
        lat = np.array([0.0, 0.1, 1.0, 0.9, 0.0, 0.0, 1.0, 1.0])
        en = _elem([[0, 1, 2, 3], [4, 5, 6, 7]])
        bad = find_nonconvex_quads(en, lon, lat)
        assert bad.size == 0

    def test_chevron_flagged(self):
        # vertex 2 pulled inside -> non-convex corner
        lon = np.array([0.0, 1.0, 0.4, 0.0])
        lat = np.array([0.0, 0.0, 0.4, 1.0])
        en = _elem([[0, 1, 2, 3]])
        bad = find_nonconvex_quads(en, lon, lat)
        np.testing.assert_array_equal(bad, [0])

    def test_collinear_degenerate_flagged(self):
        # all four vertices on one line -> zero-area corners
        lon = np.array([0.0, 1.0, 2.0, 3.0])
        lat = np.array([0.0, 0.0, 0.0, 0.0])
        en = _elem([[0, 1, 2, 3]])
        bad = find_nonconvex_quads(en, lon, lat)
        np.testing.assert_array_equal(bad, [0])

    def test_bowtie_flagged(self):
        # self-intersecting vertex order (0,1 swapped with 2,3 crossing)
        lon = np.array([0.0, 1.0, 0.0, 1.0])
        lat = np.array([0.0, 0.0, 1.0, 1.0])
        en = _elem([[0, 1, 2, 3]])   # square vertices in bowtie order
        bad = find_nonconvex_quads(en, lon, lat)
        np.testing.assert_array_equal(bad, [0])

    def test_triangles_skipped(self):
        # -1-padded and repeated-first-vertex tri encodings are NOT quads
        lon = np.array([0.0, 1.0, 0.5])
        lat = np.array([0.0, 0.0, 1.0])
        en = _elem([[0, 1, 2, -1], [0, 1, 2, 0]])
        bad = find_nonconvex_quads(en, lon, lat)
        assert bad.size == 0

    def test_pure_tri_mesh_fast_path(self):
        lon = np.array([0.0, 1.0, 0.5])
        lat = np.array([0.0, 0.0, 1.0])
        en = _elem([[0, 1, 2]])       # max_verts == 3 -> nothing to check
        bad = find_nonconvex_quads(en, lon, lat)
        assert bad.size == 0

    def test_cw_convex_quad_passes(self):
        lon = np.array([0.0, 0.0, 1.0, 1.0])
        lat = np.array([0.0, 1.0, 1.0, 0.0])   # CW winding
        en = _elem([[0, 1, 2, 3]])
        assert find_nonconvex_quads(en, lon, lat).size == 0

    def test_seam_straddling_convex_quad_passes(self):
        # convex quad across +-180: raw lon deltas would look wild
        lon = np.array([179.5, -179.5, -179.5, 179.5])
        lat = np.array([0.0, 0.0, 1.0, 1.0])
        en = _elem([[0, 1, 2, 3]])
        assert find_nonconvex_quads(en, lon, lat).size == 0

    def test_duplicate_vertex_patterns_skipped(self):
        # review 2026-07-05: lock down pairwise duplicates beyond the
        # repeated-FIRST-vertex encoding — [0,1,0,2] and [0,1,2,1] are
        # degenerate rows, NOT quads; the distinctness test must skip them.
        lon = np.array([0.0, 1.0, 0.5])
        lat = np.array([0.0, 0.0, 1.0])
        en = _elem([[0, 1, 0, 2], [0, 1, 2, 1]])
        assert find_nonconvex_quads(en, lon, lat).size == 0

    def test_polar_cap_quad_passes(self):
        # A convex quad AROUND the north pole: vertex longitudes span 270
        # deg (0/90/180/270) — a lon/lat chart false-flags this; the
        # tangent-plane projection must pass it.
        lon = np.array([0.0, 90.0, 180.0, -90.0])
        lat = np.array([89.5, 89.5, 89.5, 89.5])
        en = _elem([[0, 1, 2, 3]])
        assert find_nonconvex_quads(en, lon, lat).size == 0

    def test_near_pole_chevron_still_flagged(self):
        # Non-convexity must remain detectable at high latitude: vertex 3
        # pulled INSIDE the triangle of the other three (placed toward
        # vertex 1's direction, short radius) — a chevron on the polar
        # tangent plane.  (A short vertex on its OWN ray is just a convex
        # kite — that case is the polar-cap test above.)
        lon = np.array([0.0, 90.0, 180.0, 90.0])
        lat = np.array([89.5, 89.5, 89.5, 89.9])
        en = _elem([[0, 1, 2, 3]])
        bad = find_nonconvex_quads(en, lon, lat)
        np.testing.assert_array_equal(bad, [0])

    def test_mixed_mesh_reports_only_bad_ids(self):
        lon = np.array([0.0, 1.0, 1.0, 0.0,     # square (good)
                        2.0, 3.0, 2.4, 2.0])    # chevron (bad)
        lat = np.array([0.0, 0.0, 1.0, 1.0,
                        0.0, 0.0, 0.4, 1.0])
        en = _elem([[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 2, -1]])
        bad = find_nonconvex_quads(en, lon, lat)
        np.testing.assert_array_equal(bad, [1])


class TestReaderIntegration:
    def _fesomc_file(self, tmp_path, nv_rows, lon, lat):
        import pandas as pd
        import xarray as xr
        n_nodes = len(lon)
        nv = np.asarray(nv_rows, dtype=np.int32)   # 1-based, FESOM-C style
        n_elem = nv.shape[0]
        ds = xr.Dataset({
            "lon": (["node"], np.asarray(lon, float)),
            "lat": (["node"], np.asarray(lat, float)),
            "depth": (["node"], np.full(n_nodes, 10.0)),
            "nv": (["nele", "four"], nv),
            "sigma_lev": (["sigma"], np.array([0.0, 1.0])),
            "u2d": (["time", "nele"], np.zeros((2, n_elem))),
            "v2d": (["time", "nele"], np.zeros((2, n_elem))),
            "eta": (["time", "node"], np.zeros((2, n_nodes))),
        }, coords={"time": pd.date_range("2020-01-01", periods=2,
                                         freq="3600s")})
        p = tmp_path / "fesomc_conv.nc"
        ds.to_netcdf(p)
        return p

    def test_fesomc_nonconvex_quad_raises(self, tmp_path):
        from types import SimpleNamespace
        from kelp.readers import get_reader
        lon = [0.0, 1.0, 0.4, 0.0]
        lat = [0.0, 0.0, 0.4, 1.0]
        p = self._fesomc_file(tmp_path, [[1, 2, 3, 4]], lon, lat)
        with pytest.raises(ValueError, match="convex"):
            get_reader("fesomc").load_mesh(SimpleNamespace(meshpath=str(p)))

    def test_fesomc_convex_quads_and_tris_load(self, tmp_path):
        from types import SimpleNamespace
        from kelp.readers import get_reader
        lon = [0.0, 1.0, 1.0, 0.0, 2.0]
        lat = [0.0, 0.0, 1.0, 1.0, 0.5]
        # one convex quad + one tri (repeated first vertex)
        p = self._fesomc_file(tmp_path, [[1, 2, 3, 4], [2, 5, 3, 2]],
                              lon, lat)
        mesh = get_reader("fesomc").load_mesh(
            SimpleNamespace(meshpath=str(p)))
        assert mesh.n_elements == 2

    def test_structured_nonconvex_cell_raises(self):
        from kelp.readers.structured import build_structured_mesh
        # 2x2 T-points, one corner pulled INSIDE the quad -> chevron
        lon2d = np.array([[0.0, 1.0], [0.0, 0.45]])
        lat2d = np.array([[0.0, 0.0], [1.0, 0.45]])
        wet = np.ones((2, 2), dtype=bool)
        with pytest.raises(ValueError, match="convex"):
            build_structured_mesh(lon2d, lat2d, wet)

    def test_structured_regular_grid_loads(self):
        from kelp.readers.structured import build_structured_mesh
        lon2d, lat2d = np.meshgrid(np.arange(3.0), np.arange(3.0))
        wet = np.ones((3, 3), dtype=bool)
        mesh = build_structured_mesh(lon2d, lat2d, wet)
        assert mesh.n_elements == 4
