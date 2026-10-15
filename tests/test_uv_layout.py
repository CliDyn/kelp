"""VelocityField.uv_layout + reader-side node/element detection."""
import numpy as np
import pytest
import xarray as xr

from kelp.numerics.interpolate import VelocityField


def _vf(n=4, layout="element", **kw):
    return VelocityField(u=np.zeros(n), v=np.zeros(n),
                         eta=np.zeros(9), w=None, uv_layout=layout, **kw)


class TestUvLayoutField:
    def test_default_is_element(self):
        vf = VelocityField(u=np.zeros(4), v=np.zeros(4),
                           eta=np.zeros(9), w=None)
        assert vf.uv_layout == "element"

    def test_invalid_layout_raises(self):
        with pytest.raises(ValueError, match="uv_layout"):
            _vf(layout="faces")

    def test_lerp_layout_mismatch_raises(self):
        a = _vf(layout="element"); b = _vf(layout="node")
        with pytest.raises(ValueError, match="uv_layout"):
            a.lerp(b, 0.5)

    def test_lerp_preserves_layout(self):
        a = _vf(layout="node"); b = _vf(layout="node")
        assert a.lerp(b, 0.5).uv_layout == "node"


# --- readers ---------------------------------------------------------------

def _fesomc_ds(tmp_path, uv_dim, n_nodes=9, n_elem=8):
    """FESOM-C-style file; uv_dim selects the spatial dim of u2d/v2d."""
    import pandas as pd
    xs = np.array([4.0, 5.0, 6.0]); ys = np.array([4.0, 5.0, 6.0])
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel(); lat = gy.ravel()
    elems = []
    for j in range(2):
        for i in range(2):
            n0 = j * 3 + i
            elems.append([n0 + 1, n0 + 2, n0 + 4, n0 + 1])
            elems.append([n0 + 2, n0 + 5, n0 + 4, n0 + 2])
    nv = np.array(elems, dtype=np.int32)
    en0 = nv[:, :3] - 1
    size = n_nodes if uv_dim == "node" else n_elem
    ds = xr.Dataset({
        "lon": (["node"], lon), "lat": (["node"], lat),
        "depth": (["node"], np.full(9, 10.0)),
        "nv": (["nele", "four"], nv),
        "lon_elem": (["nele"], np.array([lon[e].mean() for e in en0])),
        "lat_elem": (["nele"], np.array([lat[e].mean() for e in en0])),
        "sigma_lev": (["sigma"], np.array([0.0, 0.5, 1.0])),
        "u2d": (["time", uv_dim], np.ones((3, size))),
        "v2d": (["time", uv_dim], np.zeros((3, size))),
        "eta": (["time", "node"], np.zeros((3, 9))),
    }, coords={"time": pd.date_range("2020-01-01", periods=3, freq="3600s")})
    p = tmp_path / f"fesomc_{uv_dim}.nc"
    ds.to_netcdf(p)
    return p


class TestReaderLayoutDetection:
    def test_fesomc_element_default(self, tmp_path):
        # mode_cfg=None would AttributeError on Fdrift3D/Vpuls dereference
        # in the FESOM-C reader — always pass an explicit 2D ModeConfig.
        from kelp.core.config import ModeConfig
        from kelp.readers import get_reader
        p = _fesomc_ds(tmp_path, "nele")
        r = get_reader("fesomc")
        mesh = r.load_mesh(type("", (), {"meshpath": str(p)})())
        ds = r.open_dataset(str(p))
        vf = r.read_velocities(ds, 0, mesh,
                               ModeConfig(Fdrift2D=True, Fdrift3D=False))
        ds.close()
        assert vf.uv_layout == "element"

    def test_fesomc_node_velocities_detected(self, tmp_path):
        from kelp.core.config import ModeConfig
        from kelp.readers import get_reader
        p = _fesomc_ds(tmp_path, "node")
        r = get_reader("fesomc")
        mesh = r.load_mesh(type("", (), {"meshpath": str(p)})())
        ds = r.open_dataset(str(p))
        vf = r.read_velocities(ds, 0, mesh,
                               ModeConfig(Fdrift2D=True, Fdrift3D=False))
        ds.close()
        assert vf.uv_layout == "node"
        assert vf.u.shape == (9,)

    def test_fesom2_honours_var_u_var_v(self, tmp_path):
        # FESOM2-style: default u/v on elem dim; unod/vnod on nod2 dim.
        from kelp.core.config import ModeConfig
        from kelp.readers import get_reader
        import pandas as pd
        n_nod, n_el = 9, 8
        ds = xr.Dataset({
            "u":    (["time", "elem"], np.full((2, n_el), 2.0)),
            "v":    (["time", "elem"], np.zeros((2, n_el))),
            "unod": (["time", "nod2"], np.full((2, n_nod), 3.0)),
            "vnod": (["time", "nod2"], np.zeros((2, n_nod))),
            "ssh":  (["time", "nod2"], np.zeros((2, n_nod))),
        }, coords={"time": pd.date_range("2020-01-01", periods=2,
                                         freq="3600s")})
        # minimal Mesh matching the counts (PACKAGE-form import:
        # tests/__init__.py exists, bare import fails at collection)
        from tests.test_uv_registry import _mesh_from
        lonn = [4, 5, 6, 4, 5, 6, 4, 5, 6]; latn = [4, 4, 4, 5, 5, 5, 6, 6, 6]
        en = [[0, 1, 4, -1], [0, 4, 3, -1], [1, 2, 5, -1], [1, 5, 4, -1],
              [3, 4, 7, -1], [3, 7, 6, -1], [4, 5, 8, -1], [4, 8, 7, -1]]
        mesh = _mesh_from(lonn, latn, en)
        r = get_reader("fesom2")
        mode_el = ModeConfig(Fdrift2D=True, Fdrift3D=False)
        vf_el = r.read_velocities(ds, 0, mesh, mode_el)
        assert vf_el.uv_layout == "element" and vf_el.u[0] == 2.0
        mode_nd = ModeConfig(Fdrift2D=True, Fdrift3D=False,
                             var_u="unod", var_v="vnod")
        vf_nd = r.read_velocities(ds, 0, mesh, mode_nd)
        assert vf_nd.uv_layout == "node" and vf_nd.u[0] == 3.0

    def test_uv_layout_disagreement_raises(self, tmp_path):
        from kelp.core.config import ModeConfig
        from kelp.readers import get_reader
        import pandas as pd
        ds = xr.Dataset({
            "u": (["time", "elem"], np.ones((1, 8))),
            "v": (["time", "nod2"], np.zeros((1, 9))),
        }, coords={"time": pd.date_range("2020-01-01", periods=1,
                                         freq="3600s")})
        from tests.test_uv_registry import _mesh_from
        lonn = [4, 5, 6, 4, 5, 6, 4, 5, 6]; latn = [4, 4, 4, 5, 5, 5, 6, 6, 6]
        en = [[0, 1, 4, -1], [0, 4, 3, -1], [1, 2, 5, -1], [1, 5, 4, -1],
              [3, 4, 7, -1], [3, 7, 6, -1], [4, 5, 8, -1], [4, 8, 7, -1]]
        mesh = _mesh_from(lonn, latn, en)
        r = get_reader("fesom2")
        with pytest.raises(ValueError, match="layout"):
            r.read_velocities(ds, 0, mesh, ModeConfig(Fdrift2D=True,
                                                      Fdrift3D=False))

    def test_fesomc_3d_node_velocities_survive(self, tmp_path):
        # 3D node-layout (K-1, N): the 3D path must validate against
        # n_nodes, not n_elements (review round 2)
        from kelp.core.config import ModeConfig
        from kelp.readers import get_reader
        import pandas as pd
        # build like _fesomc_ds but with a layered node u/v named u, v
        xs = np.array([4.0, 5.0, 6.0]); ys = np.array([4.0, 5.0, 6.0])
        gx, gy = np.meshgrid(xs, ys)
        lon = gx.ravel(); lat = gy.ravel()
        elems = []
        for j in range(2):
            for i in range(2):
                n0 = j * 3 + i
                elems.append([n0 + 1, n0 + 2, n0 + 4, n0 + 1])
                elems.append([n0 + 2, n0 + 5, n0 + 4, n0 + 2])
        nv = np.array(elems, dtype=np.int32)
        en0 = nv[:, :3] - 1
        ds = xr.Dataset({
            "lon": (["node"], lon), "lat": (["node"], lat),
            "depth": (["node"], np.full(9, 10.0)),
            "nv": (["nele", "four"], nv),
            "lon_elem": (["nele"], np.array([lon[e].mean() for e in en0])),
            "lat_elem": (["nele"], np.array([lat[e].mean() for e in en0])),
            "sigma_lev": (["sigma"], np.array([0.0, 0.5, 1.0])),
            "u": (["time", "sigmam1", "node"], np.ones((2, 2, 9))),
            "v": (["time", "sigmam1", "node"], np.zeros((2, 2, 9))),
            "w": (["time", "sigma", "node"], np.zeros((2, 3, 9))),
            "eta": (["time", "node"], np.zeros((2, 9))),
        }, coords={"time": pd.date_range("2020-01-01", periods=2,
                                         freq="3600s")})
        p = tmp_path / "fesomc_3d_node.nc"
        ds.to_netcdf(p)
        r = get_reader("fesomc")
        mesh = r.load_mesh(type("", (), {"meshpath": str(p)})())
        dso = r.open_dataset(str(p))
        vf = r.read_velocities(dso, 0, mesh,
                               ModeConfig(Fdrift2D=False, Fdrift3D=True))
        dso.close()
        assert vf.uv_layout == "node"
        assert vf.u.shape == (2, 9)      # (K-1, N) intact

    def test_fesom2_3d_node_velocities_both_orientations(self, tmp_path):
        # FESOM2 3D node u/v stored either (time, nz1, nod2) or
        # (time, nod2, nz1): the reader must orient to (K-1, N) by dim
        # NAME so the kernel sees vertical-first (BLOCKER, review).  Layer
        # k carries value k+1 so a transpose is detectable.
        from kelp.core.config import ModeConfig
        from kelp.core.mesh import (
            Mesh, build_node_to_elems, build_search,
        )
        from kelp.readers import get_reader
        import pandas as pd
        lonn = np.array([4, 5, 6, 4, 5, 6, 4, 5, 6], dtype=np.float64)
        latn = np.array([4, 4, 4, 5, 5, 5, 6, 6, 6], dtype=np.float64)
        en = np.array(
            [[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4],
             [3, 4, 7], [3, 7, 6], [4, 5, 8], [4, 8, 7]], dtype=np.int32)
        cl = np.array([lonn[e].mean() for e in en], dtype=np.float64)
        ca = np.array([latn[e].mean() for e in en], dtype=np.float64)
        n2e, n2e_cnt = build_node_to_elems(9, en)
        # K=3 interfaces -> K-1=2 layers, matches the nz1 dim below.
        mesh = Mesh(name="f2", nodes_lon=lonn, nodes_lat=latn,
                    depth=np.full(9, 10.0), elem_nodes=en,
                    elem_center_lon=cl, elem_center_lat=ca,
                    node_to_elems=n2e, node_to_elems_count=n2e_cnt,
                    sigma=np.array([0.0, 5.0, 10.0]),
                    vertical_coord="zlevel")
        build_search(mesh)

        # layer 0 -> 1.0, layer 1 -> 2.0 on every node
        layer_vals = np.array([1.0, 2.0])
        u_vert_first = np.broadcast_to(
            layer_vals[None, :, None], (2, 2, 9)).copy()
        r = get_reader("fesom2")
        mode3d = ModeConfig(Fdrift2D=False, Fdrift3D=True,
                            var_u="unod", var_v="vnod")
        for orient, dims, arr in (
            ("vert_first", ("time", "nz1", "nod2"), u_vert_first),
            ("node_first", ("time", "nod2", "nz1"),
             np.swapaxes(u_vert_first, 1, 2).copy()),
        ):
            ds = xr.Dataset({
                "unod": (list(dims), arr),
                "vnod": (list(dims), np.zeros_like(arr)),
                "w": (["time", "nz", "nod2"], np.zeros((2, 3, 9))),
                "ssh": (["time", "nod2"], np.zeros((2, 9))),
            }, coords={"time": pd.date_range("2020-01-01", periods=2,
                                             freq="3600s")})
            vf = r.read_velocities(ds, 0, mesh, mode3d)
            assert vf.uv_layout == "node", orient
            assert vf.u.shape == (2, 9), orient        # (K-1, N) intact
            # layer index 1 on node 0 must be 2.0 for BOTH orientations
            assert vf.u[1, 0] == 2.0, orient

    def test_fesomc_layered_node_reduced_to_2d_both_orientations(self, tmp_path):
        # 2D mode + layered node var: node-aware reduction, BOTH axis
        # orders (sigma-first and node-first) must give the layer pick
        from kelp.core.config import ModeConfig
        from kelp.readers import get_reader
        import pandas as pd
        xs = np.array([4.0, 5.0, 6.0]); ys = np.array([4.0, 5.0, 6.0])
        gx, gy = np.meshgrid(xs, ys)
        lon = gx.ravel(); lat = gy.ravel()
        elems = []
        for j in range(2):
            for i in range(2):
                n0 = j * 3 + i
                elems.append([n0 + 1, n0 + 2, n0 + 4, n0 + 1])
                elems.append([n0 + 2, n0 + 5, n0 + 4, n0 + 2])
        nv = np.array(elems, dtype=np.int32)
        en0 = nv[:, :3] - 1
        layer_vals = np.array([1.0, 9.0])     # layer 0 -> 1.0, layer 1 -> 9.0
        u_sig_first = np.broadcast_to(
            layer_vals[None, :, None], (2, 2, 9)).copy()
        for orient, dims, arr in (
            ("sig_first", ("time", "sigmam1", "node"), u_sig_first),
            ("node_first", ("time", "node", "sigmam1"),
             np.swapaxes(u_sig_first, 1, 2).copy()),
        ):
            ds = xr.Dataset({
                "lon": (["node"], lon), "lat": (["node"], lat),
                "depth": (["node"], np.full(9, 10.0)),
                "nv": (["nele", "four"], nv),
                "lon_elem": (["nele"],
                             np.array([lon[e].mean() for e in en0])),
                "lat_elem": (["nele"],
                             np.array([lat[e].mean() for e in en0])),
                "sigma_lev": (["sigma"], np.array([0.0, 0.5, 1.0])),
                "u2d": (list(dims), arr),
                "v2d": (list(dims), np.zeros_like(arr)),
                "eta": (["time", "node"], np.zeros((2, 9))),
            }, coords={"time": pd.date_range("2020-01-01", periods=2,
                                             freq="3600s")})
            p = tmp_path / f"fesomc_layered_{orient}.nc"
            ds.to_netcdf(p)
            r = get_reader("fesomc")
            mesh = r.load_mesh(type("", (), {"meshpath": str(p)})())
            dso = r.open_dataset(str(p))
            vf = r.read_velocities(
                dso, 0, mesh,
                ModeConfig(Fdrift2D=True, Fdrift3D=False, sigma_layer=1))
            dso.close()
            assert vf.uv_layout == "node", orient
            np.testing.assert_allclose(vf.u, 9.0, err_msg=orient)


class TestProbeUvLayout:
    def test_probe_matches_read_2d(self, tmp_path):
        from kelp.core.config import ModeConfig
        from kelp.readers import get_reader
        for uv_dim, expected in (("nele", "element"), ("node", "node")):
            p = _fesomc_ds(tmp_path, uv_dim)
            r = get_reader("fesomc")
            mesh = r.load_mesh(type("", (), {"meshpath": str(p)})())
            ds = r.open_dataset(str(p))
            mode = ModeConfig(Fdrift2D=True, Fdrift3D=False)
            assert r.probe_uv_layout(ds, mode, mesh) == expected
            ds.close()

    def test_probe_uses_3d_name_resolution(self, tmp_path):
        # dataset carries BOTH a node-based 2D diagnostic u2d AND the
        # real element-based 3D u: in 3D mode the probe must classify
        # from "u" (element), not "u2d" (review round 2)
        from kelp.core.config import ModeConfig
        from kelp.readers import get_reader
        import pandas as pd
        xs = np.array([4.0, 5.0, 6.0]); ys = np.array([4.0, 5.0, 6.0])
        gx, gy = np.meshgrid(xs, ys)
        lon = gx.ravel(); lat = gy.ravel()
        elems = []
        for j in range(2):
            for i in range(2):
                n0 = j * 3 + i
                elems.append([n0 + 1, n0 + 2, n0 + 4, n0 + 1])
                elems.append([n0 + 2, n0 + 5, n0 + 4, n0 + 2])
        nv = np.array(elems, dtype=np.int32)
        en0 = nv[:, :3] - 1
        ds = xr.Dataset({
            "lon": (["node"], lon), "lat": (["node"], lat),
            "depth": (["node"], np.full(9, 10.0)),
            "nv": (["nele", "four"], nv),
            "lon_elem": (["nele"], np.array([lon[e].mean() for e in en0])),
            "lat_elem": (["nele"], np.array([lat[e].mean() for e in en0])),
            "sigma_lev": (["sigma"], np.array([0.0, 0.5, 1.0])),
            "u2d": (["time", "node"], np.ones((2, 9))),
            "v2d": (["time", "node"], np.zeros((2, 9))),
            "u": (["time", "sigmam1", "nele"], np.ones((2, 2, 8))),
            "v": (["time", "sigmam1", "nele"], np.zeros((2, 2, 8))),
            "eta": (["time", "node"], np.zeros((2, 9))),
        }, coords={"time": pd.date_range("2020-01-01", periods=2,
                                         freq="3600s")})
        p = tmp_path / "fesomc_dual.nc"
        ds.to_netcdf(p)
        r = get_reader("fesomc")
        mesh = r.load_mesh(type("", (), {"meshpath": str(p)})())
        dso = r.open_dataset(str(p))
        mode3d = ModeConfig(Fdrift2D=False, Fdrift3D=True)
        mode2d = ModeConfig(Fdrift2D=True, Fdrift3D=False)
        assert r.probe_uv_layout(dso, mode3d, mesh) == "element"
        assert r.probe_uv_layout(dso, mode2d, mesh) == "node"
        dso.close()

    def test_probe_is_metadata_only(self, tmp_path, monkeypatch):
        # the probe must never load data: poison .values
        from kelp.core.config import ModeConfig
        from kelp.readers import get_reader
        p = _fesomc_ds(tmp_path, "node")
        r = get_reader("fesomc")
        mesh = r.load_mesh(type("", (), {"meshpath": str(p)})())
        ds = r.open_dataset(str(p))

        def _boom(self):
            raise AssertionError("probe touched .values (data load)")

        monkeypatch.setattr(xr.DataArray, "values",
                            property(_boom), raising=True)
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False)
        assert r.probe_uv_layout(ds, mode, mesh) == "node"
        ds.close()


# --- model-provided areas + per-family r_earth (addendum deliverables 1-2) --

def _fesomc_area_ds(tmp_path, *, with_areas=False, elem_val=7.0,
                    node_val=5.0, elem_time=None, tag=None):
    """FESOM-C-style mesh file; optionally carries element + node areas.

    The area VALUES are deliberately constant sentinels that DIFFER from
    what build_elem_areas would compute, so a passing precedence test
    proves the file value was returned (not recomputed).

    ``elem_time`` adds a leading ``time`` dim of that size to ``elem_area``
    (so the variable is rank-2, shape ``(elem_time, nele)``): ``elem_time=1``
    is a squeezable singleton that must still be accepted as ``(E,)``;
    ``elem_time>1`` is a genuine extra axis that must be SKIPPED (fallback).
    """
    xs = np.array([4.0, 5.0, 6.0]); ys = np.array([4.0, 5.0, 6.0])
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel(); lat = gy.ravel()
    elems = []
    for j in range(2):
        for i in range(2):
            n0 = j * 3 + i
            elems.append([n0 + 1, n0 + 2, n0 + 4, n0 + 1])
            elems.append([n0 + 2, n0 + 5, n0 + 4, n0 + 2])
    nv = np.array(elems, dtype=np.int32)
    en0 = nv[:, :3] - 1
    n_elem = nv.shape[0]
    data = {
        "lon": (["node"], lon), "lat": (["node"], lat),
        "depth": (["node"], np.full(9, 10.0)),
        "nv": (["nele", "four"], nv),
        "lon_elem": (["nele"], np.array([lon[e].mean() for e in en0])),
        "lat_elem": (["nele"], np.array([lat[e].mean() for e in en0])),
        "sigma_lev": (["sigma"], np.array([0.0, 0.5, 1.0])),
    }
    if with_areas:
        if elem_time is not None:
            data["elem_area"] = (
                ["time", "nele"], np.full((elem_time, n_elem), elem_val))
        else:
            data["elem_area"] = (["nele"], np.full(n_elem, elem_val))
        data["area"] = (["node"], np.full(9, node_val))
    ds = xr.Dataset(data)
    suffix = tag if tag is not None else str(with_areas)
    p = tmp_path / f"fesomc_area_{suffix}.nc"
    ds.to_netcdf(p)
    return p, n_elem


def _fesom2_diag_data(*, with_area=False, elem_val=7.0, area_time=None):
    """Build the dict + n_elem for a minimal FESOM2 mesh-diag dataset.

    Carries the variables ``_load_mesh_nc`` / ``_load_mesh_zarr`` need
    (lon/lat/elements/nod_in_elem2D/_num) plus an optional ``elem_area``
    sentinel.  ``area_time`` adds a leading ``time`` dim of that size to
    ``elem_area`` so the read site sees a rank-2 variable (``area_time>1``
    is a genuine extra axis that must be SKIPPED -> fallback).
    """
    from kelp.core.mesh import build_node_to_elems
    lonn = np.array([4, 5, 6, 4, 5, 6, 4, 5, 6], dtype=np.float64)
    latn = np.array([4, 4, 4, 5, 5, 5, 6, 6, 6], dtype=np.float64)
    en = np.array(
        [[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4],
         [3, 4, 7], [3, 7, 6], [4, 5, 8], [4, 8, 7]], dtype=np.int32)
    n_elem = en.shape[0]
    # elements stored (n3, elem), 1-based
    elements = (en.T + 1).astype(np.int32)
    n2e, n2e_cnt = build_node_to_elems(9, en)
    # nod_in_elem2D stored (N, nod2) 1-based, 0=pad -> transpose of (nod2, N)
    nie = np.where(n2e >= 0, n2e + 1, 0).astype(np.int32).T  # (max_conn, nod2)
    data = {
        "lon": (["nod2"], lonn), "lat": (["nod2"], latn),
        "elements": (["n3", "elem"], elements),
        "nod_in_elem2D": (["N", "nod2"], nie),
        "nod_in_elem2D_num": (["nod2"], n2e_cnt.astype(np.int32)),
    }
    if with_area:
        if area_time is not None:
            data["elem_area"] = (
                ["time", "elem"], np.full((area_time, n_elem), elem_val))
        else:
            data["elem_area"] = (["elem"], np.full(n_elem, elem_val))
    return data, n_elem


def _fesom2_diag_ds(tmp_path, *, with_area=False, elem_val=7.0,
                    area_time=None, tag=None):
    """Minimal FESOM2 mesh-diag NetCDF (NetCDF-diag branch of load_mesh)."""
    data, n_elem = _fesom2_diag_data(
        with_area=with_area, elem_val=elem_val, area_time=area_time)
    ds = xr.Dataset(data)
    suffix = tag if tag is not None else str(with_area)
    p = tmp_path / f"fesom2_diag_{suffix}.nc"
    ds.to_netcdf(p)
    return p, n_elem


def _fesom2_diag_zarr(tmp_path, *, with_area=False, elem_val=7.0,
                      area_time=None, tag=None):
    """Minimal FESOM2 mesh-diag Zarr store (Zarr branch of load_mesh)."""
    data, n_elem = _fesom2_diag_data(
        with_area=with_area, elem_val=elem_val, area_time=area_time)
    ds = xr.Dataset(data)
    suffix = tag if tag is not None else str(with_area)
    p = tmp_path / f"fesom2_diag_{suffix}.zarr"
    ds.to_zarr(p)
    return p, n_elem


class TestModelProvidedAreas:
    def test_fesomc_reads_element_and_node_areas(self, tmp_path):
        # Precedence pin: file areas (sentinel 7.0 / 5.0) are returned
        # untouched by build_elem_areas; mesh._node_areas holds the file
        # node values.  Spherical recomputation would NOT give 7.0.
        from kelp.core.mesh import build_elem_areas
        from kelp.readers import get_reader
        p, n_elem = _fesomc_area_ds(tmp_path, with_areas=True,
                                    elem_val=7.0, node_val=5.0)
        r = get_reader("fesomc")
        mesh = r.load_mesh(type("", (), {"meshpath": str(p)})())
        # the lazy cache must already hold the file element areas
        assert mesh._elem_areas is not None
        A = build_elem_areas(mesh)
        assert np.array_equal(A, np.full(n_elem, 7.0))
        assert mesh._node_areas is not None
        assert np.array_equal(mesh._node_areas, np.full(9, 5.0))

    def test_fesomc_no_areas_computes_fallback(self, tmp_path):
        # Without area vars the fallback runs and gives real (non-sentinel)
        # areas; _node_areas stays None.
        from kelp.core.mesh import build_elem_areas
        from kelp.readers import get_reader
        p, n_elem = _fesomc_area_ds(tmp_path, with_areas=False)
        r = get_reader("fesomc")
        mesh = r.load_mesh(type("", (), {"meshpath": str(p)})())
        assert mesh._elem_areas is None      # not pre-filled
        A = build_elem_areas(mesh)
        assert A.shape == (n_elem,)
        assert np.all(A > 0.0)
        assert not np.allclose(A, 7.0)
        assert mesh._node_areas is None

    def test_fesom2_reads_elem_area(self, tmp_path):
        # FESOM2 NetCDF-diag branch reads elem_area; precedence pinned by
        # the 7.0 sentinel.
        from kelp.core.mesh import build_elem_areas
        from kelp.readers import get_reader
        p, n_elem = _fesom2_diag_ds(tmp_path, with_area=True, elem_val=7.0)
        r = get_reader("fesom2")
        mesh = r.load_mesh(type("", (), {"meshpath": str(p)})())
        assert mesh._elem_areas is not None
        A = build_elem_areas(mesh)
        assert np.array_equal(A, np.full(n_elem, 7.0))

    def test_fesom2_no_elem_area_computes_fallback(self, tmp_path):
        from kelp.core.mesh import build_elem_areas
        from kelp.readers import get_reader
        p, n_elem = _fesom2_diag_ds(tmp_path, with_area=False)
        r = get_reader("fesom2")
        mesh = r.load_mesh(type("", (), {"meshpath": str(p)})())
        assert mesh._elem_areas is None
        A = build_elem_areas(mesh)
        assert A.shape == (n_elem,) and np.all(A > 0.0)

    # -- rank-safe candidate acceptance (Fix 1 + Fix 2) ----------------------

    def test_fesomc_elem_area_time1_squeezed_accepted(self, tmp_path):
        # elem_area(time=1, nele): total size == E so the OLD .size-only
        # guard accepts it as (1, E).  Must squeeze to (E,).  RED before
        # the fix: mesh._elem_areas.shape == (1, E).
        from kelp.core.mesh import build_elem_areas
        from kelp.readers import get_reader
        p, n_elem = _fesomc_area_ds(tmp_path, with_areas=True, elem_val=7.0,
                                    node_val=5.0, elem_time=1, tag="t1")
        r = get_reader("fesomc")
        mesh = r.load_mesh(type("", (), {"meshpath": str(p)})())
        assert mesh._elem_areas is not None
        assert mesh._elem_areas.shape == (n_elem,)
        A = build_elem_areas(mesh)
        assert np.array_equal(A, np.full(n_elem, 7.0))

    def test_fesomc_elem_area_time3_skipped_fallback(self, tmp_path):
        # elem_area(time=3, nele): a genuine extra axis (size 3E != E).
        # Must be SKIPPED -> the spherical fallback runs (NOT the 7.0
        # sentinel).  Already skipped by the old size guard; pin it stays so.
        from kelp.core.mesh import build_elem_areas
        from kelp.readers import get_reader
        p, n_elem = _fesomc_area_ds(tmp_path, with_areas=True, elem_val=7.0,
                                    node_val=5.0, elem_time=3, tag="t3")
        r = get_reader("fesomc")
        mesh = r.load_mesh(type("", (), {"meshpath": str(p)})())
        assert mesh._elem_areas is None      # not pre-filled from file
        A = build_elem_areas(mesh)
        assert A.shape == (n_elem,)
        assert not np.allclose(A, 7.0)       # fallback, not file values
        # node area is a clean (N,) var here -> still accepted
        assert mesh._node_areas is not None
        assert np.array_equal(mesh._node_areas, np.full(9, 5.0))

    def test_fesomc_area_buffers_are_owned(self, tmp_path):
        # Fix 3: the cached arrays must OWN their buffers (returned by
        # reference to consumers) — base is None.
        from kelp.readers import get_reader
        p, n_elem = _fesomc_area_ds(tmp_path, with_areas=True, elem_val=7.0,
                                    node_val=5.0, tag="owned")
        r = get_reader("fesomc")
        mesh = r.load_mesh(type("", (), {"meshpath": str(p)})())
        assert mesh._elem_areas.base is None
        assert mesh._node_areas.base is None

    def test_fesom2_diag_wrong_shape_elem_area_fallback(self, tmp_path):
        # FESOM2 NetCDF-diag: elem_area(time=2, elem) is rank-2 with a
        # non-singleton extra axis -> must fall through to the spherical
        # fallback (file values NOT returned).  RED before the fix:
        # _load_mesh_nc stored the (2, E) array unguarded.
        from kelp.core.mesh import build_elem_areas
        from kelp.readers import get_reader
        p, n_elem = _fesom2_diag_ds(tmp_path, with_area=True, elem_val=7.0,
                                    area_time=2, tag="t2")
        r = get_reader("fesom2")
        mesh = r.load_mesh(type("", (), {"meshpath": str(p)})())
        assert mesh._elem_areas is None
        A = build_elem_areas(mesh)
        assert A.shape == (n_elem,)
        assert not np.allclose(A, 7.0)

    def test_fesom2_diag_elem_area_time1_squeezed_accepted(self, tmp_path):
        # elem_area(time=1, elem): squeezable singleton -> accepted as (E,).
        from kelp.core.mesh import build_elem_areas
        from kelp.readers import get_reader
        p, n_elem = _fesom2_diag_ds(tmp_path, with_area=True, elem_val=7.0,
                                    area_time=1, tag="f2t1")
        r = get_reader("fesom2")
        mesh = r.load_mesh(type("", (), {"meshpath": str(p)})())
        assert mesh._elem_areas is not None
        assert mesh._elem_areas.shape == (n_elem,)
        A = build_elem_areas(mesh)
        assert np.array_equal(A, np.full(n_elem, 7.0))

    def test_fesom2_diag_elem_area_owned(self, tmp_path):
        # Fix 3: FESOM2 NetCDF-diag area buffer is owned.
        from kelp.readers import get_reader
        p, n_elem = _fesom2_diag_ds(tmp_path, with_area=True, elem_val=7.0,
                                    tag="f2owned")
        r = get_reader("fesom2")
        mesh = r.load_mesh(type("", (), {"meshpath": str(p)})())
        assert mesh._elem_areas.base is None

    def test_fesom2_zarr_wrong_shape_elem_area_fallback(self, tmp_path):
        # FESOM2 Zarr branch: same rank guard on the (2, E) elem_area.
        from kelp.core.mesh import build_elem_areas
        from kelp.readers import get_reader
        p, n_elem = _fesom2_diag_zarr(tmp_path, with_area=True, elem_val=7.0,
                                      area_time=2, tag="zt2")
        r = get_reader("fesom2")
        mesh = r.load_mesh(type("", (), {"meshpath": str(p)})())
        assert mesh._elem_areas is None
        A = build_elem_areas(mesh)
        assert A.shape == (n_elem,)
        assert not np.allclose(A, 7.0)

    def test_fesom2_zarr_elem_area_accepted_and_owned(self, tmp_path):
        # FESOM2 Zarr branch: clean (E,) elem_area accepted, buffer owned.
        from kelp.core.mesh import build_elem_areas
        from kelp.readers import get_reader
        p, n_elem = _fesom2_diag_zarr(tmp_path, with_area=True, elem_val=7.0,
                                      tag="zclean")
        r = get_reader("fesom2")
        mesh = r.load_mesh(type("", (), {"meshpath": str(p)})())
        assert mesh._elem_areas is not None
        assert mesh._elem_areas.shape == (n_elem,)
        assert mesh._elem_areas.base is None
        A = build_elem_areas(mesh)
        assert np.array_equal(A, np.full(n_elem, 7.0))

    # -- FESOM-C area-source print (Fix 4) -----------------------------------

    def test_fesomc_prints_both_area_sources(self, tmp_path, capsys):
        from kelp.readers import get_reader
        p, _ = _fesomc_area_ds(tmp_path, with_areas=True, tag="printboth")
        r = get_reader("fesomc")
        r.load_mesh(type("", (), {"meshpath": str(p)})())
        out = capsys.readouterr().out
        assert "element areas: from file" in out
        assert "node areas: from file" in out

    def test_fesomc_prints_fallback_when_no_areas(self, tmp_path, capsys):
        from kelp.readers import get_reader
        p, _ = _fesomc_area_ds(tmp_path, with_areas=False, tag="printnone")
        r = get_reader("fesomc")
        r.load_mesh(type("", (), {"meshpath": str(p)})())
        out = capsys.readouterr().out
        assert "element areas: not in file -> spherical fallback" in out


class TestREarthPlumbing:
    def test_fesomc_default_r_earth(self, tmp_path):
        from kelp.readers import get_reader
        p, _ = _fesomc_area_ds(tmp_path, with_areas=False)
        r = get_reader("fesomc")
        mesh = r.load_mesh(type("", (), {"meshpath": str(p)})())
        assert mesh.r_earth == 6_400_000.0

    def test_fesom2_default_r_earth(self, tmp_path):
        from kelp.readers import get_reader
        p, _ = _fesom2_diag_ds(tmp_path, with_area=False)
        r = get_reader("fesom2")
        mesh = r.load_mesh(type("", (), {"meshpath": str(p)})())
        assert mesh.r_earth == 6_367_500.0

    def test_explicit_r_earth_flows_through(self, tmp_path):
        from kelp.core.config import MeshConfig
        from kelp.readers import get_reader
        p, _ = _fesomc_area_ds(tmp_path, with_areas=False)
        r = get_reader("fesomc")
        cfg = MeshConfig(type="fesomc", meshpath=str(p), r_earth=6_300_000.0)
        mesh = r.load_mesh(cfg)
        assert mesh.r_earth == 6_300_000.0

    def test_meshconfig_r_earth_default_none(self):
        from kelp.core.config import MeshConfig
        assert MeshConfig().r_earth is None

    def test_meshconfig_invalid_r_earth_raises(self):
        from kelp.core.config import MeshConfig
        with pytest.raises(ValueError, match="r_earth"):
            MeshConfig(r_earth=0.0)
        with pytest.raises(ValueError, match="r_earth"):
            MeshConfig(r_earth=-1.0)
