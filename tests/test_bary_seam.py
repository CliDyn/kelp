"""Dateline safety of the barycentric W, sigma-column and sampling kernels.

Five barycentric kernels projected an element onto a local plane around the
RAW arithmetic mean of its vertex longitudes: ``_interp_w_barycentric_single``
and its time-blended twins ``_interp_w_bary_single_tinterp`` /
``_interp_w_bary_single_pchip4`` (numerics/vertical.py), ``_bary_locate``
(numerics/vertical_sigma.py) and the barycentric branch of
``sampling._combine_horizontal``.  For a quadrilateral split two-and-two
across +-180 deg the raw mean lands 180 deg away from the element, the
wrapped vertex offsets split around +-180, the element projects to a
360-deg-wide shape, the containment test fails and the value falls back
silently to the nearest node's.  The u/v barycentric kernel and
``core.mesh._project_to_local_xy`` already anchor the origin on the first
vertex for such elements.

The tests pin the seam via translation invariance: the same quad mesh and a
field linear in (longitude offset, latitude) interpolate EXACTLY (Wachspress
reproduces linear fields) at the seam and 160 deg away.
"""
import numpy as np
import pytest

from kelp.core.mesh import Mesh, build_node_to_elems, build_search, find_nearest_nodes
from kelp.numerics.sampling import sample_field
from kelp.numerics.vertical import (
    interp_w_spatial,
    _interp_w_bary_single_tinterp,
    _interp_w_bary_single_pchip4,
)
from kelp.numerics.vertical_sigma import interp_w_sigma_spatial


def _wrap(x):
    return ((np.asarray(x, dtype=np.float64) + 180.0) % 360.0) - 180.0


def _quad_mesh_at(lon0, vertical_coord="zlevel"):
    """3x3 nodes, four quads; node columns at lon0, lon0+1, lon0+2 (wrapped),
    rows at lat 10, 11, 12.  With lon0 = 179.5 the left quads are split
    two-and-two across the dateline."""
    lons1 = _wrap(lon0 + np.arange(3.0))
    lats1 = 10.0 + np.arange(3.0)
    LON, LAT = np.meshgrid(lons1, lats1)
    lon = LON.ravel()
    lat = LAT.ravel()
    en = []
    for r in range(2):
        for c in range(2):
            i = r * 3 + c
            en.append([i, i + 1, i + 4, i + 3])
    en = np.asarray(en, np.int32)
    n2e, n2e_cnt = build_node_to_elems(lon.size, en)
    cl = np.empty(en.shape[0])
    ca = np.empty(en.shape[0])
    for e, row in enumerate(en):
        ref = lon[row[0]]
        cl[e] = _wrap(ref + np.mean(_wrap(lon[row] - ref)))
        ca[e] = lat[row].mean()
    if vertical_coord == "zlevel":
        sigma = np.array([0.0, 50.0, 100.0])
    else:
        sigma = np.array([1.0, 0.5, 0.0])
    m = Mesh(name="seamquad", nodes_lon=lon, nodes_lat=lat,
             depth=np.full(lon.size, 100.0),
             elem_nodes=en, elem_center_lon=cl, elem_center_lat=ca,
             node_to_elems=n2e, node_to_elems_count=n2e_cnt,
             sigma=sigma, vertical_coord=vertical_coord)
    build_search(m)
    return m


def _linear(mesh, lon0, a=0.3, b=2.0, c=-0.7):
    """Node field linear in the wrapped longitude offset from lon0 and in lat."""
    off = _wrap(mesh.nodes_lon - lon0)
    return a + b * off + c * (mesh.nodes_lat - 10.0)


# (longitude offset, latitude) inside the LEFT quads, which straddle the
# dateline when lon0 = 179.5
_QUERIES = [(0.3, 10.6), (0.7, 11.4), (0.5, 10.5), (0.2, 11.8)]
_SEAM = 179.5
_AWAY = 19.5


def _queries(lon0):
    lons = _wrap(np.array([lon0 + q[0] for q in _QUERIES]))
    lats = np.array([q[1] for q in _QUERIES])
    return lons, lats


def _expected(a=0.3, b=2.0, c=-0.7):
    return np.array([a + b * q[0] + c * (q[1] - 10.0) for q in _QUERIES])


@pytest.mark.parametrize("lon0", [_AWAY, _SEAM])
def test_w_barycentric_reproduces_linear_field_across_the_dateline(lon0):
    m = _quad_mesh_at(lon0)
    col = _linear(m, lon0)
    w = np.vstack([col, col, col])            # constant in depth
    lons, lats = _queries(lon0)
    nn = find_nearest_nodes(m, lons, lats)
    z = np.full(lons.size, 30.0)
    alive = np.ones(lons.size, dtype=bool)
    got = interp_w_spatial(w, m.sigma, z, lons, lats, nn, alive,
                           time_dir=1.0, method="barycentric", mesh=m,
                           node_depths=m.depth)
    np.testing.assert_allclose(got, _expected(), rtol=0, atol=1e-9)


@pytest.mark.parametrize("lon0", [_AWAY, _SEAM])
def test_w_barycentric_time_blended_twins_across_the_dateline(lon0):
    m = _quad_mesh_at(lon0)
    col = _linear(m, lon0)
    w = np.ascontiguousarray(np.vstack([col, col, col]))
    lons, lats = _queries(lon0)
    nn = find_nearest_nodes(m, lons, lats)
    exp = _expected()
    for i in range(lons.size):
        v, ok = _interp_w_bary_single_tinterp(
            lons[i], lats[i], 30.0, int(nn[i]), w, w, w, 0.25, 0.5,
            m.sigma, 1.0, m.depth, m.nodes_lon, m.nodes_lat,
            m.node_to_elems, m.node_to_elems_count, m.elem_nodes)
        assert ok
        assert v == pytest.approx(exp[i], abs=1e-9)
        v, ok = _interp_w_bary_single_pchip4(
            lons[i], lats[i], 30.0, int(nn[i]), w, w, w, w,
            0.0, 1.0, 2.0, 3.0, 1.5,
            m.sigma, 1.0, m.depth, m.nodes_lon, m.nodes_lat,
            m.node_to_elems, m.node_to_elems_count, m.elem_nodes)
        assert ok
        assert v == pytest.approx(exp[i], abs=1e-9)


@pytest.mark.parametrize("lon0", [_AWAY, _SEAM])
def test_sigma_w_barycentric_across_the_dateline(lon0):
    m = _quad_mesh_at(lon0, vertical_coord="sigma")
    col = _linear(m, lon0)
    w = np.vstack([col, col, col])
    lons, lats = _queries(lon0)
    nn = find_nearest_nodes(m, lons, lats)
    z = np.full(lons.size, 30.0)
    alive = np.ones(lons.size, dtype=bool)
    eta = np.zeros(m.nodes_lon.size)
    got = interp_w_sigma_spatial(w, m.sigma, z, lons, lats, nn, alive,
                                 m.depth, eta, Dmin=0.5, time_dir=1.0,
                                 method="barycentric", mesh=m)
    np.testing.assert_allclose(got, _expected(), rtol=0, atol=1e-9)


@pytest.mark.parametrize("lon0", [_AWAY, _SEAM])
def test_sampling_barycentric_across_the_dateline(lon0):
    m = _quad_mesh_at(lon0)
    vals = _linear(m, lon0)
    lons, lats = _queries(lon0)
    got = sample_field(vals, m, lons, lats, method="barycentric")
    np.testing.assert_allclose(got, _expected(), rtol=0, atol=1e-9)


@pytest.mark.parametrize("lon0", [_AWAY, _SEAM])
def test_fused_3d_rk2_vertical_step_across_the_dateline(lon0, monkeypatch):
    """Through rk2_step: node-layout 3-D barycentric on a z-level mesh takes
    the FUSED kernel (asserted), whose W sites call the time-blended twins.
    With u = v = 0 and W constant in depth the particle stays put and moves
    by W(x) dt exactly -- also on a quad split across the dateline."""
    from kelp.core.io import VelocityField
    from kelp.core.particles import make_state
    import kelp.numerics.integrators as integ

    calls = []
    real = integ._run_fused_rk2_3d_bary_tinterp

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(integ, "_run_fused_rk2_3d_bary_tinterp", spy)
    m = _quad_mesh_at(lon0)
    n = m.nodes_lon.size
    col = _linear(m, lon0) * 1e-3                    # m/s, positive down
    w = np.ascontiguousarray(np.vstack([col, col, col]))
    vel = VelocityField(u=np.zeros((2, n)), v=np.zeros((2, n)),
                        eta=np.zeros(n), w=w, uv_layout="node")
    lons, lats = _queries(lon0)
    st = make_state(lons, lats, np.full(lons.size, 30.0))
    st = st.replace(nearest_node=find_nearest_nodes(m, lons, lats))
    dt = 1000.0
    out = integ.rk2_step(st, vel, m, dt, 1.0, uv_interp="barycentric",
                         w_interp="barycentric")
    assert calls, "the fused 3-D RK2 kernel was not used"
    np.testing.assert_allclose(out.z, 30.0 + _expected() * 1e-3 * dt,
                               rtol=0, atol=1e-9)
