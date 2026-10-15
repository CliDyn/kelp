"""Antimeridian (dateline) safety of the default leastsq u/v cores.

The three production leastsq cores (steady / tinterp / pchip4) used RAW
longitude differences: a particle at +179.9 with a stencil element centred
at -179.9 saw it ~360 deg away, corrupting the fit for stencils that
straddle the seam.  Fixed 2026-07 with a GATED correction (applied only
when |raw dlon| > 180), so off-seam arithmetic stays bit-identical — the
existing golden/regression pins are the off-seam proof; these tests pin
the seam via translation invariance: the same mesh + field built at the
seam and 30 deg away must interpolate identically.

The legacy Fortran-parity kernel (_interp_leastsq_core_fortran) is
deliberately NOT fixed — its purpose is bitwise comparison with Fortran.
"""
import numpy as np
import pytest

from kelp.core.io import VelocityField
from kelp.core.mesh import (
    Mesh, build_node_to_elems, build_search, find_nearest_nodes,
)
from kelp.core.particles import make_state, ParticleStatus
from kelp.numerics.interpolate import (
    interp_leastsq, _interp_leastsq_core_tinterp, _interp_leastsq_core_pchip4,
)


def _wrap(x):
    return ((x + 180.0) % 360.0) - 180.0


def _grid_mesh_at(lon0):
    """3x3-node, 8-tri mesh whose lon column spacing is 1 deg starting at
    lon0 (built with wrapped node lons); lats 0..2."""
    lons1 = _wrap(lon0 + np.arange(3.0))
    lats1 = np.arange(0.0, 3.0)
    LON, LAT = np.meshgrid(lons1, lats1)
    lon = LON.ravel(); lat = LAT.ravel()
    en = []
    for r in range(2):
        for c in range(2):
            i = r * 3 + c
            en.append([i, i + 1, i + 4, -1])
            en.append([i, i + 4, i + 3, -1])
    en = np.asarray(en, np.int32)
    n2e, n2e_cnt = build_node_to_elems(lon.size, en)
    # anchored (seam-aware) element centers
    cl = np.empty(en.shape[0]); ca = np.empty(en.shape[0])
    for e, row in enumerate(en):
        v = row[row >= 0]
        ref = lon[v[0]]
        cl[e] = _wrap(ref + np.mean(_wrap(lon[v] - ref)))
        ca[e] = lat[v].mean()
    m = Mesh(name="seam", nodes_lon=lon, nodes_lat=lat,
             depth=np.full(lon.size, 10.0),
             elem_nodes=en, elem_center_lon=cl, elem_center_lat=ca,
             node_to_elems=n2e, node_to_elems_count=n2e_cnt,
             sigma=np.empty(0), vertical_coord="zlevel")
    build_search(m)
    return m


def _linear_field(mesh, lon0, slope=3.0, offset=0.2):
    """Element field linear in the WRAPPED offset from lon0."""
    off = _wrap(np.asarray(mesh.elem_center_lon) - lon0)
    u = offset + slope * off
    return u


# Query points as offsets from lon0 (the mesh spans lon0..lon0+2, lat 0..2)
_QUERIES = [(0.9, 1.05), (1.1, 0.8), (1.4, 1.5), (0.6, 1.2)]


class TestSteadyCoreSeam:
    def test_translation_invariance_quadratic_field(self):
        # A QUADRATIC field is NOT reproduced exactly by the LSQ plane, so
        # the interpolated value depends on the exact stencil geometry —
        # this catches a seam/offset anchor-node or stencil mismatch that
        # an exactly-linear field would hide (review 2026-07-03, minor 3).
        m_seam = _grid_mesh_at(179.0)
        m_off = _grid_mesh_at(-151.0)
        for time_dir in (1.0, -1.0):
            for dx, dy in _QUERIES:
                vals = {}
                for tag, m, lon0 in (("s", m_seam, 179.0),
                                     ("o", m_off, -151.0)):
                    off = _wrap(np.asarray(m.elem_center_lon) - lon0)
                    u = 0.2 + 3.0 * off + 0.8 * off ** 2
                    vf = VelocityField(u=u, v=2 * u, eta=np.zeros(9),
                                       w=None)
                    q = _wrap(lon0 + dx)
                    st = make_state(np.array([q]), np.array([dy]))
                    nn = find_nearest_nodes(m, st.lon, st.lat)
                    U, V, ok = interp_leastsq(st, m, vf, nn, time_dir)
                    assert ok[0], (dx, dy, time_dir)
                    vals[tag] = (U[0], V[0])
                assert vals["s"][0] == pytest.approx(vals["o"][0],
                                                     rel=1e-9), (dx, dy)
                assert vals["s"][1] == pytest.approx(vals["o"][1],
                                                     rel=1e-9), (dx, dy)

    def test_translation_invariance_linear_field(self):
        # Mesh + linear field built AT the seam (spanning 179..181, i.e.
        # crossing +-180) and 30 deg away: interp_leastsq must return
        # identical values at corresponding points.  RED before the gated
        # wrap: seam stencils saw ~360-deg deltas.
        m_seam = _grid_mesh_at(179.0)
        m_off = _grid_mesh_at(-151.0)
        u_seam = _linear_field(m_seam, 179.0)
        u_off = _linear_field(m_off, -151.0)
        vf_seam = VelocityField(u=u_seam, v=2 * u_seam,
                                eta=np.zeros(9), w=None)
        vf_off = VelocityField(u=u_off, v=2 * u_off,
                               eta=np.zeros(9), w=None)
        for dx, dy in _QUERIES:
            q_s, q_o = _wrap(179.0 + dx), _wrap(-151.0 + dx)
            st_s = make_state(np.array([q_s]), np.array([dy]))
            st_o = make_state(np.array([q_o]), np.array([dy]))
            nn_s = find_nearest_nodes(m_seam, st_s.lon, st_s.lat)
            nn_o = find_nearest_nodes(m_off, st_o.lon, st_o.lat)
            U_s, V_s, ok_s = interp_leastsq(st_s, m_seam, vf_seam, nn_s)
            U_o, V_o, ok_o = interp_leastsq(st_o, m_off, vf_off, nn_o)
            assert ok_s[0] and ok_o[0], (dx, dy)
            assert U_s[0] == pytest.approx(U_o[0], rel=1e-9), (dx, dy)
            assert V_s[0] == pytest.approx(V_o[0], rel=1e-9), (dx, dy)
            # and the value is the exact linear-field value
            expected = 0.2 + 3.0 * dx
            assert U_s[0] == pytest.approx(expected, rel=1e-9), (dx, dy)

    def test_fused_euler_step_seam_matches_offset(self):
        # The fused 2D Euler kernel inlines the same core: a step on the
        # seam mesh must displace exactly like the 30-deg-shifted twin.
        pytest.importorskip("numba")
        from kelp.numerics.integrators import euler_step
        m_seam = _grid_mesh_at(179.0)
        m_off = _grid_mesh_at(-151.0)
        u_seam = _linear_field(m_seam, 179.0)
        u_off = _linear_field(m_off, -151.0)
        vf_seam = VelocityField(u=u_seam, v=0.5 * u_seam,
                                eta=np.zeros(9), w=None)
        vf_off = VelocityField(u=u_off, v=0.5 * u_off,
                               eta=np.zeros(9), w=None)
        dxs = np.array([q[0] for q in _QUERIES])
        dys = np.array([q[1] for q in _QUERIES])
        st_s = make_state(_wrap(179.0 + dxs), dys.copy())
        st_o = make_state(_wrap(-151.0 + dxs), dys.copy())
        st_s = st_s.replace(nearest_node=find_nearest_nodes(
            m_seam, st_s.lon, st_s.lat))
        st_o = st_o.replace(nearest_node=find_nearest_nodes(
            m_off, st_o.lon, st_o.lat))
        out_s = euler_step(st_s, vf_seam, m_seam, 600.0, 1.0)
        out_o = euler_step(st_o, vf_off, m_off, 600.0, 1.0)
        assert (out_s.status == ParticleStatus.ALIVE).all()
        assert (out_o.status == ParticleStatus.ALIVE).all()
        np.testing.assert_allclose(_wrap(out_s.lon - out_o.lon - 330.0), 0.0,
                                   atol=1e-9)
        np.testing.assert_allclose(out_s.lat, out_o.lat, atol=1e-9)


class TestTimeAwareCoresSeam:
    def _core_args(self, mesh, q_lon, q_lat):
        nn = int(find_nearest_nodes(mesh, np.array([q_lon]),
                                    np.array([q_lat]))[0])
        return nn, (mesh.node_to_elems_count, mesh.node_to_elems,
                    mesh.elem_center_lon, mesh.elem_center_lat)

    def test_tinterp_core_translation_invariance(self):
        m_seam = _grid_mesh_at(179.0)
        m_off = _grid_mesh_at(-151.0)
        out = {}
        for tag, m, lon0 in (("s", m_seam, 179.0), ("o", m_off, -151.0)):
            u_a = _linear_field(m, lon0, slope=3.0)
            u_b = _linear_field(m, lon0, slope=5.0)   # different at t_b
            q_lon, q_lat = _wrap(lon0 + 0.9), 1.05
            nn, marrs = self._core_args(m, q_lon, q_lat)
            out[tag] = _interp_leastsq_core_tinterp(
                q_lon, q_lat, nn, 1.0, 0.3, 0.2,
                *marrs, u_a, 2 * u_a, u_b, 2 * u_b, u_b, 2 * u_b)
        assert out["s"][2] and out["o"][2]
        assert out["s"][0] == pytest.approx(out["o"][0], rel=1e-9)
        assert out["s"][1] == pytest.approx(out["o"][1], rel=1e-9)

    def test_pchip4_core_translation_invariance(self):
        m_seam = _grid_mesh_at(179.0)
        m_off = _grid_mesh_at(-151.0)
        out = {}
        for tag, m, lon0 in (("s", m_seam, 179.0), ("o", m_off, -151.0)):
            snaps = [_linear_field(m, lon0, slope=s)
                     for s in (2.0, 3.0, 4.5, 5.0)]
            q_lon, q_lat = _wrap(lon0 + 0.9), 1.05
            nn, marrs = self._core_args(m, q_lon, q_lat)
            out[tag] = _interp_leastsq_core_pchip4(
                q_lon, q_lat, nn, 1.0,
                0.0, 1.0, 2.0, 3.0, 1.5,
                *marrs,
                snaps[0], 2 * snaps[0], snaps[1], 2 * snaps[1],
                snaps[2], 2 * snaps[2], snaps[3], 2 * snaps[3])
        assert out["s"][2] and out["o"][2]
        assert out["s"][0] == pytest.approx(out["o"][0], rel=1e-9)
        assert out["s"][1] == pytest.approx(out["o"][1], rel=1e-9)
