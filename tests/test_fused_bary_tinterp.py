"""Fused 2D barycentric + time-interp kernels for NODE-layout velocities
(candidate 10).

Node velocities (FESOM2 unod/vnod, NEMO C->T) must use
uv_interp=barycentric, which routed every RK stage through the unfused
path — and temporal interpolation there materializes a blend of the
WHOLE node field per stage (`_make_stage_vels` -> lerp).  The fused
kernels blend the two snapshots inline per VERTEX instead (the exact
nested arithmetic of the leastsq tinterp kernels), so cost scales with
particles, not mesh size.

Fusing is gated on the grid-hash density certificate
(`grid_hash_dense_for_elements`): under it the in-kernel widened
nearest-node anchor is bit-identical to the batch KD-healed one, so
parity below is asserted BITWISE, not within a tolerance.
"""
import numpy as np
import pytest

from kelp.core.io import LazyLerpField, VelocityField
from kelp.core.mesh import (Mesh, build_grid_hash, build_node_to_elems,
                                build_search, find_nearest_nodes,
                                grid_hash_dense_for_elements,
                                _element_contains, _gh_cell_of)
from kelp.core.particles import (FailureReason, ParticleState,
                                     ParticleStatus, make_state)


def _wrap_band_mesh(n_lon=72, lat0=40.0, lat1=60.0, n_lat=9):
    """Cyclic global band (dateline seam INCLUDED): exercises the wrap
    legs of the grid hash, the certificate dilation (np.roll) and the
    wrapped element-bbox raster."""
    lons = -180.0 + 360.0 * np.arange(n_lon) / n_lon
    lats = np.linspace(lat0, lat1, n_lat)
    gx, gy = np.meshgrid(lons, lats)
    lon = gx.ravel().astype(np.float64)
    lat = gy.ravel().astype(np.float64)
    elems = []
    for j in range(n_lat - 1):
        for i in range(n_lon):
            i2 = (i + 1) % n_lon          # seam column connects back
            n0 = j * n_lon + i
            n1 = j * n_lon + i2
            n2 = (j + 1) * n_lon + i
            n3 = (j + 1) * n_lon + i2
            elems.append([n0, n1, n2])
            elems.append([n1, n3, n2])
    elem_nodes = np.array(elems, dtype=np.int32)
    n2e, cnt = build_node_to_elems(lon.size, elem_nodes)

    def _ecl(e):
        ref = lon[e[0]]
        d = ((lon[e] - ref + 180.0) % 360.0) - 180.0
        return ((ref + d.mean() + 180.0) % 360.0) - 180.0

    mesh = Mesh(name="band", nodes_lon=lon, nodes_lat=lat,
                depth=np.full(lon.size, 100.0), elem_nodes=elem_nodes,
                elem_center_lon=np.array([_ecl(e) for e in elem_nodes]),
                elem_center_lat=np.array([lat[e].mean() for e in elem_nodes]),
                node_to_elems=n2e, node_to_elems_count=cnt)
    build_search(mesh)
    assert mesh._gh_wrap_lon                     # cyclic grid engaged
    return mesh


def _node_mesh(nx=9, span=0.2):
    xs = np.linspace(-span, span, nx)
    ys = np.linspace(-span, span, nx)
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel().astype(np.float64)
    lat = gy.ravel().astype(np.float64)
    elems = []
    for j in range(nx - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            elems.append([n0, n0 + 1, n0 + nx])
            elems.append([n0 + 1, n0 + nx + 1, n0 + nx])
    elem_nodes = np.array(elems, dtype=np.int32)
    n2e, cnt = build_node_to_elems(nx * nx, elem_nodes)
    mesh = Mesh(name="node2d", nodes_lon=lon, nodes_lat=lat,
                depth=np.full(nx * nx, 100.0), elem_nodes=elem_nodes,
                elem_center_lon=np.array([lon[e].mean() for e in elem_nodes]),
                elem_center_lat=np.array([lat[e].mean() for e in elem_nodes]),
                node_to_elems=n2e, node_to_elems_count=cnt)
    build_search(mesh)
    return mesh


def _quad_mesh(nx=7, span=0.18):
    """Structured convex-QUAD mesh (Wachspress weight branch)."""
    xs = np.linspace(-span, span, nx)
    ys = np.linspace(-span, span, nx)
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel().astype(np.float64)
    lat = gy.ravel().astype(np.float64)
    elems = []
    for j in range(nx - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            elems.append([n0, n0 + 1, n0 + nx + 1, n0 + nx])  # CCW quad
    elem_nodes = np.array(elems, dtype=np.int32)
    n2e, cnt = build_node_to_elems(nx * nx, elem_nodes)
    mesh = Mesh(name="quad2d", nodes_lon=lon, nodes_lat=lat,
                depth=np.full(nx * nx, 100.0), elem_nodes=elem_nodes,
                elem_center_lon=np.array(
                    [lon[e].mean() for e in elem_nodes]),
                elem_center_lat=np.array(
                    [lat[e].mean() for e in elem_nodes]),
                node_to_elems=n2e, node_to_elems_count=cnt)
    build_search(mesh)
    return mesh


def _node_vel(mesh, t):
    """Spatially- and temporally-varying NODE-layout field."""
    lon, lat = mesh.nodes_lon, mesh.nodes_lat
    u = 0.15 + 0.4 * lon + 0.2 * lat + 0.05 * np.sin(0.011 * t)
    v = -0.05 + 0.3 * lat - 0.1 * lon + 0.03 * np.cos(0.017 * t)
    return VelocityField(u=u, v=v, eta=np.zeros(mesh.n_nodes),
                         uv_layout="node")


def _seeds(n=16, seed=3):
    rng = np.random.default_rng(seed)
    return make_state(rng.uniform(-0.12, 0.12, n),
                      rng.uniform(-0.12, 0.12, n))


def _assert_state_equal(out_f, out_u):
    """Bitwise fused/unfused parity (positions, status, anchors, K1)."""
    np.testing.assert_array_equal(out_f.lon, out_u.lon)
    np.testing.assert_array_equal(out_f.lat, out_u.lat)
    np.testing.assert_array_equal(out_f.status, out_u.status)
    np.testing.assert_array_equal(out_f.nearest_node, out_u.nearest_node)
    np.testing.assert_array_equal(out_f.u, out_u.u)
    np.testing.assert_array_equal(out_f.v, out_u.v)


class TestFusedBaryTinterpParity:
    @pytest.mark.parametrize("step_name", ["rk2", "rk4"])
    @pytest.mark.parametrize("time_dir", [1.0, -1.0])
    def test_matches_unfused(self, step_name, time_dir, monkeypatch):
        import kelp.numerics.integrators as I
        from kelp.numerics.integrators import (
            rk2_step, rk4_step, _rk2_step_unfused, _rk4_step_unfused)
        calls = {"n": 0}
        real = I._run_fused_2d_bary_tinterp

        def spy(*a, **k):
            calls["n"] += 1
            return real(*a, **k)

        monkeypatch.setattr(I, "_run_fused_2d_bary_tinterp", spy)
        mesh = _node_mesh()
        vel = _node_vel(mesh, 0.0)
        vel_next = _node_vel(mesh, 600.0)
        st = _seeds()
        dt = 120.0
        step = rk2_step if step_name == "rk2" else rk4_step
        unf = _rk2_step_unfused if step_name == "rk2" else _rk4_step_unfused
        out_f = step(st, vel, mesh, dt, time_dir,
                     vel_next=vel_next, dt_to_next=600.0,
                     uv_interp="barycentric")
        assert calls["n"] == 1              # fused path actually taken
        out_u = unf(st, vel, mesh, dt, time_dir, 0, False,
                    uv_interp="barycentric",
                    vel_next=vel_next, dt_to_next=600.0)
        _assert_state_equal(out_f, out_u)

    @pytest.mark.parametrize("step_name", ["rk2", "rk4"])
    def test_quad_mesh_matches_unfused(self, step_name, monkeypatch):
        """Wachspress (quad) weight branch through the fused core."""
        import kelp.numerics.integrators as I
        from kelp.numerics.integrators import (
            rk2_step, rk4_step, _rk2_step_unfused, _rk4_step_unfused)
        calls = {"n": 0}
        real = I._run_fused_2d_bary_tinterp

        def spy(*a, **k):
            calls["n"] += 1
            return real(*a, **k)

        monkeypatch.setattr(I, "_run_fused_2d_bary_tinterp", spy)
        mesh = _quad_mesh()
        vel = _node_vel(mesh, 0.0)
        vel_next = _node_vel(mesh, 600.0)
        st = _seeds(n=12, seed=7)
        step = rk2_step if step_name == "rk2" else rk4_step
        unf = _rk2_step_unfused if step_name == "rk2" else _rk4_step_unfused
        out_f = step(st, vel, mesh, 120.0, 1.0,
                     vel_next=vel_next, dt_to_next=600.0,
                     uv_interp="barycentric")
        assert calls["n"] == 1
        out_u = unf(st, vel, mesh, 120.0, 1.0, 0, False,
                    uv_interp="barycentric",
                    vel_next=vel_next, dt_to_next=600.0)
        _assert_state_equal(out_f, out_u)

    def test_lazylerp_pair_matches_unfused(self, monkeypatch):
        """LazyLerpField base (alpha0 > 0) + stage target."""
        import kelp.numerics.integrators as I
        from kelp.numerics.integrators import rk2_step, _rk2_step_unfused
        calls = {"n": 0}
        real = I._run_fused_2d_bary_tinterp

        def spy(*a, **k):
            calls["n"] += 1
            return real(*a, **k)

        monkeypatch.setattr(I, "_run_fused_2d_bary_tinterp", spy)
        mesh = _node_mesh()
        a, b = _node_vel(mesh, 0.0), _node_vel(mesh, 600.0)
        lazy = LazyLerpField(a, b, 0.4)
        st = _seeds()
        out_f = rk2_step(st, lazy, mesh, 120.0, 1.0,
                         vel_next=b, dt_to_next=360.0,
                         uv_interp="barycentric")
        assert calls["n"] == 1
        out_u = _rk2_step_unfused(st, lazy, mesh, 120.0, 1.0, 0, False,
                                  uv_interp="barycentric",
                                  vel_next=b, dt_to_next=360.0)
        _assert_state_equal(out_f, out_u)

    @pytest.mark.parametrize("step_name", ["rk2", "rk4"])
    def test_backward_bracket_contract(self, step_name):
        """The REAL backward contract: vel_next is the LOWER snapshot
        (the one the step moves toward), both for a raw field and for a
        LazyLerpField bracket pair."""
        from kelp.numerics.integrators import (
            rk2_step, rk4_step, _rk2_step_unfused, _rk4_step_unfused)
        mesh = _node_mesh()
        A, B = _node_vel(mesh, 0.0), _node_vel(mesh, 600.0)
        st = _seeds()
        step = rk2_step if step_name == "rk2" else rk4_step
        unf = _rk2_step_unfused if step_name == "rk2" else _rk4_step_unfused
        # raw field sitting AT the upper snapshot, moving down to A
        out_f = step(st, B, mesh, 120.0, -1.0, vel_next=A, dt_to_next=600.0,
                     uv_interp="barycentric")
        out_u = unf(st, B, mesh, 120.0, -1.0, 0, False,
                    uv_interp="barycentric", vel_next=A, dt_to_next=600.0)
        _assert_state_equal(out_f, out_u)
        # LazyLerp bracket pair at alpha0=0.6 inside [A, B], moving to A
        lazy = LazyLerpField(A, B, 0.6)
        out_f = step(st, lazy, mesh, 120.0, -1.0, vel_next=A,
                     dt_to_next=360.0, uv_interp="barycentric")
        out_u = unf(st, lazy, mesh, 120.0, -1.0, 0, False,
                    uv_interp="barycentric", vel_next=A, dt_to_next=360.0)
        _assert_state_equal(out_f, out_u)


class TestRoutingGates:
    def _spy(self, monkeypatch):
        import kelp.numerics.integrators as I
        calls = {"n": 0}
        real = I._run_fused_2d_bary_tinterp

        def spy(*a, **k):
            calls["n"] += 1
            return real(*a, **k)

        monkeypatch.setattr(I, "_run_fused_2d_bary_tinterp", spy)
        return calls

    def test_element_layout_stays_unfused(self, monkeypatch):
        from kelp.numerics.integrators import rk2_step
        calls = self._spy(monkeypatch)
        mesh = _node_mesh()
        ne = mesh.n_elements
        vel = VelocityField(u=np.full(ne, 0.1), v=np.zeros(ne),
                            eta=np.zeros(mesh.n_nodes))
        rk2_step(_seeds(), vel, mesh, 120.0, 1.0,
                 vel_next=vel, dt_to_next=600.0, uv_interp="barycentric")
        assert calls["n"] == 0

    def test_steady_stays_unfused(self, monkeypatch):
        from kelp.numerics.integrators import rk2_step
        calls = self._spy(monkeypatch)
        mesh = _node_mesh()
        rk2_step(_seeds(), _node_vel(mesh, 0.0), mesh, 120.0, 1.0,
                 uv_interp="barycentric")
        assert calls["n"] == 0

    def test_walk_elems_stay_unfused(self, monkeypatch):
        from kelp.numerics.integrators import rk2_step
        calls = self._spy(monkeypatch)
        mesh = _node_mesh()
        st = _seeds()
        st = st.replace(elem=np.zeros(st.n_particles, dtype=np.int32))
        rk2_step(st, _node_vel(mesh, 0.0), mesh, 120.0, 1.0,
                 vel_next=_node_vel(mesh, 600.0), dt_to_next=600.0,
                 uv_interp="barycentric")
        assert calls["n"] == 0

    def test_recon_stays_unfused(self, monkeypatch):
        """PCHIP bundle keeps the unfused path even though tracking passes
        vel_next ALONGSIDE recon (the production shape) — without vel_next
        the time_aware clause would short-circuit first and this test
        would be vacuous."""
        from kelp.numerics.integrators import rk2_step
        calls = self._spy(monkeypatch)
        mesh = _node_mesh()
        snaps = [_node_vel(mesh, t) for t in (0.0, 600.0, 1200.0, 1800.0)]
        rk2_step(_seeds(), snaps[1], mesh, 120.0, 1.0,
                 vel_next=snaps[2], dt_to_next=600.0,
                 recon=(snaps, [0.0, 600.0, 1200.0, 1800.0], 700.0, 1.0),
                 uv_interp="barycentric")
        assert calls["n"] == 0

    def test_wrapper_rejects_element_layout_vel_next(self):
        from kelp.numerics.integrators import _run_fused_2d_bary_tinterp
        from kelp.numerics.integrators import _fused_rk2_step_2d_bary_tinterp
        mesh = _node_mesh()
        vel = _node_vel(mesh, 0.0)
        # element-layout target whose array SIZE happens to match the node
        # count — the shape check upstream cannot catch it, only the
        # layout guard can
        bad = VelocityField(u=np.zeros(mesh.n_nodes),
                            v=np.zeros(mesh.n_nodes),
                            eta=np.zeros(mesh.n_nodes),
                            uv_layout="element")
        with pytest.raises(ValueError, match="node-layout vel_next"):
            _run_fused_2d_bary_tinterp(
                _seeds(), vel, bad, 600.0, mesh, 120.0, 1.0, 0,
                _fused_rk2_step_2d_bary_tinterp, c_coeffs=(0.0, 0.5))


class TestAnchorCertificateGate:
    """Grid-hash density certificate: fused only where the in-kernel
    anchor is provably exact (HEALPix polar rows are the real deny case;
    here a synthetic sparse mesh reproduces the geometry)."""

    def _sparse_mesh(self):
        """Dense node cluster + one huge far element: the big element's
        interior cells have empty 3x3 windows -> certificate denied."""
        nx = 20
        xs = np.linspace(0.0, 1.0, nx)
        gx, gy = np.meshgrid(xs, xs)
        lon = gx.ravel()
        lat = gy.ravel()
        elems = []
        for j in range(nx - 1):
            for i in range(nx - 1):
                n0 = j * nx + i
                elems.append([n0, n0 + 1, n0 + nx])
                elems.append([n0 + 1, n0 + nx + 1, n0 + nx])
        n_grid = nx * nx
        lon = np.concatenate([lon, [10.0, 14.0, 10.0]])
        lat = np.concatenate([lat, [0.0, 4.0, 4.0]])
        elems.append([n_grid, n_grid + 1, n_grid + 2])
        elem_nodes = np.array(elems, dtype=np.int32)
        n2e, cnt = build_node_to_elems(lon.size, elem_nodes)
        mesh = Mesh(name="sparse", nodes_lon=lon, nodes_lat=lat,
                    depth=np.full(lon.size, 100.0), elem_nodes=elem_nodes,
                    elem_center_lon=np.array(
                        [lon[e].mean() for e in elem_nodes]),
                    elem_center_lat=np.array(
                        [lat[e].mean() for e in elem_nodes]),
                    node_to_elems=n2e, node_to_elems_count=cnt)
        build_search(mesh)
        return mesh

    def test_dense_mesh_certified(self):
        mesh = _node_mesh()
        assert grid_hash_dense_for_elements(mesh) is True
        assert mesh._gh_dense_elems is True          # cached

    def test_sparse_mesh_denied(self):
        mesh = self._sparse_mesh()
        assert grid_hash_dense_for_elements(mesh) is False

    def test_rebuild_invalidates_cache(self):
        mesh = _node_mesh()
        grid_hash_dense_for_elements(mesh)
        assert mesh._gh_dense_elems is not None
        build_grid_hash(mesh)
        assert mesh._gh_dense_elems is None

    def test_denied_mesh_stays_unfused_with_warning(self, monkeypatch):
        import kelp.numerics.integrators as I
        from kelp.numerics.integrators import rk2_step
        calls = {"n": 0}
        real = I._run_fused_2d_bary_tinterp

        def spy(*a, **k):
            calls["n"] += 1
            return real(*a, **k)

        monkeypatch.setattr(I, "_run_fused_2d_bary_tinterp", spy)
        mesh = self._sparse_mesh()
        lonn, latn = mesh.nodes_lon, mesh.nodes_lat
        vel = VelocityField(u=np.full(lonn.size, 0.1),
                            v=np.zeros(lonn.size),
                            eta=np.zeros(lonn.size), uv_layout="node")
        st = make_state(np.array([0.5, 0.52]), np.array([0.5, 0.48]))
        with pytest.warns(RuntimeWarning, match="fused kernels"):
            out = rk2_step(st, vel, mesh, 60.0, 1.0,
                           vel_next=vel, dt_to_next=600.0,
                           uv_interp="barycentric")
        assert calls["n"] == 0                       # unfused path taken
        assert np.all(out.status == ParticleStatus.ALIVE)
        # warning is one-time per mesh
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error")
            rk2_step(st, vel, mesh, 60.0, 1.0,
                     vel_next=vel, dt_to_next=600.0,
                     uv_interp="barycentric")
        assert calls["n"] == 0


class TestFailureReasonStamps:
    def test_fast_flow_all_carry_codes(self):
        from kelp.numerics.integrators import rk2_step
        mesh = _node_mesh()
        vel = VelocityField(u=np.full(mesh.n_nodes, 500.0),
                            v=np.zeros(mesh.n_nodes),
                            eta=np.zeros(mesh.n_nodes), uv_layout="node")
        st = _seeds(n=6)
        st = st.replace(failure_reason=np.zeros(6, dtype=np.int8))
        out = rk2_step(st, vel, mesh, 3600.0, 1.0,
                       vel_next=vel, dt_to_next=3600.0,
                       uv_interp="barycentric")
        assert np.all(out.status == ParticleStatus.OUTSIDE)
        assert np.all(out.failure_reason != FailureReason.NONE)

    def test_oob_stamped_exactly(self):
        """Endpoint exits the hull while every STAGE stays inside ->
        the OOB leg specifically (not INTERPOLATION_FAILED)."""
        from kelp.numerics.integrators import rk2_step
        mesh = _node_mesh()          # hull at lon = +/-0.2
        dt = 3600.0
        # displacement per full step = 0.022 deg: midpoint 0.196 (inside),
        # endpoint 0.207 (outside)
        u = 0.022 * 111120.0 / dt
        vel = VelocityField(u=np.full(mesh.n_nodes, u),
                            v=np.zeros(mesh.n_nodes),
                            eta=np.zeros(mesh.n_nodes), uv_layout="node")
        st = make_state(np.array([0.185]), np.array([0.0]))
        st = st.replace(failure_reason=np.zeros(1, dtype=np.int8))
        out = rk2_step(st, vel, mesh, dt, 1.0,
                       vel_next=vel, dt_to_next=dt,
                       uv_interp="barycentric")
        assert out.status[0] == ParticleStatus.OUTSIDE
        assert out.failure_reason[0] == FailureReason.OUT_OF_BOUNDS
        assert out.lon[0] == 0.185                   # reverted

    def test_interp_failed_stamped(self):
        """Seed placed just outside the mesh hull: stage-1 containment
        fails -> INTERPOLATION_FAILED through the fused bary kernel."""
        from kelp.numerics.integrators import rk2_step
        mesh = _node_mesh()
        vel = _node_vel(mesh, 0.0)
        st = make_state(np.array([0.35]), np.array([0.0]))  # off-hull
        st = st.replace(failure_reason=np.zeros(1, dtype=np.int8))
        out = rk2_step(st, vel, mesh, 120.0, 1.0,
                       vel_next=_node_vel(mesh, 600.0), dt_to_next=600.0,
                       uv_interp="barycentric")
        assert out.status[0] == ParticleStatus.OUTSIDE
        assert out.failure_reason[0] == FailureReason.INTERPOLATION_FAILED


class TestNonFiniteEndpointCollapse:
    """The blend must NOT multiply unused arrays: 0*Inf would poison
    stages whose alpha sits exactly on an endpoint (K1 always has
    alpha == 0), and the alpha0 == 0 collapse likewise must never read
    bracket-B.  Pinned at core level with Inf planted in the unused
    array."""

    def _setup(self):
        mesh = _node_mesh()
        xp, yp = 0.01, 0.02
        nn = int(find_nearest_nodes(mesh, np.array([xp]),
                                    np.array([yp]))[0])
        fin_a = _node_vel(mesh, 0.0)
        fin_t = _node_vel(mesh, 600.0)
        return mesh, xp, yp, nn, fin_a, fin_t

    def _core(self, mesh, xp, yp, nn, alpha0, alpha, ua, va, ub, vb, ut, vt):
        from kelp.numerics.interpolate import (
            _interp_barycentric_core_tinterp)
        return _interp_barycentric_core_tinterp(
            xp, yp, nn, 1.0, alpha0, alpha,
            mesh.node_to_elems, mesh.node_to_elems_count,
            mesh.elem_nodes, mesh.nodes_lon, mesh.nodes_lat,
            ua, va, ub, vb, ut, vt, mesh.r_earth)

    def test_alpha0_zero_never_reads_bracket_b(self):
        mesh, xp, yp, nn, A, T = self._setup()
        inf = np.full(mesh.n_nodes, np.inf)
        u1, v1, ok1 = self._core(mesh, xp, yp, nn, 0.0, 0.4,
                                 A.u, A.v, inf, inf, T.u, T.v)
        u2, v2, ok2 = self._core(mesh, xp, yp, nn, 0.0, 0.4,
                                 A.u, A.v, A.u, A.v, T.u, T.v)
        assert ok1 and ok2
        assert np.isfinite(u1) and (u1, v1) == (u2, v2)

    def test_alpha_zero_never_reads_target(self):
        mesh, xp, yp, nn, A, T = self._setup()
        inf = np.full(mesh.n_nodes, np.inf)
        u1, v1, ok1 = self._core(mesh, xp, yp, nn, 0.0, 0.0,
                                 A.u, A.v, A.u, A.v, inf, inf)
        u2, v2, ok2 = self._core(mesh, xp, yp, nn, 0.0, 0.0,
                                 A.u, A.v, A.u, A.v, T.u, T.v)
        assert ok1 and ok2
        assert np.isfinite(u1) and (u1, v1) == (u2, v2)

    def test_alpha_one_never_reads_bracket(self):
        mesh, xp, yp, nn, A, T = self._setup()
        inf = np.full(mesh.n_nodes, np.inf)
        u1, v1, ok1 = self._core(mesh, xp, yp, nn, 0.3, 1.0,
                                 inf, inf, inf, inf, T.u, T.v)
        u2, v2, ok2 = self._core(mesh, xp, yp, nn, 0.0, 0.0,
                                 T.u, T.v, T.u, T.v, A.u, A.v)
        assert ok1 and ok2
        assert np.isfinite(u1) and (u1, v1) == (u2, v2)

    def test_near_endpoint_thresholds_match_make_stage_vels(self):
        """alpha just inside 1e-12 of an endpoint takes the same object
        shortcut as `_make_stage_vels` (< 1e-12 -> bracket, > 1-1e-12 ->
        target) — bitwise parity with the unfused path even for freak
        dt/dt_to_next ratios, and no 0*Inf from the unused array."""
        mesh, xp, yp, nn, A, T = self._setup()
        inf = np.full(mesh.n_nodes, np.inf)
        u_lo, v_lo, ok_lo = self._core(mesh, xp, yp, nn, 0.0, 5e-13,
                                       A.u, A.v, A.u, A.v, inf, inf)
        u_a0, v_a0, _ = self._core(mesh, xp, yp, nn, 0.0, 0.0,
                                   A.u, A.v, A.u, A.v, T.u, T.v)
        assert ok_lo and np.isfinite(u_lo) and (u_lo, v_lo) == (u_a0, v_a0)
        u_hi, v_hi, ok_hi = self._core(mesh, xp, yp, nn, 0.3, 1.0 - 5e-13,
                                       inf, inf, inf, inf, T.u, T.v)
        u_a1, v_a1, _ = self._core(mesh, xp, yp, nn, 0.0, 0.0,
                                   T.u, T.v, T.u, T.v, A.u, A.v)
        assert ok_hi and np.isfinite(u_hi) and (u_hi, v_hi) == (u_a1, v_a1)


class TestFailureParity:
    def test_rk4_failed_particle_bitwise_parity(self):
        """A particle that actually FAILS mid-step must come out bitwise
        identical from the fused and unfused RK4 paths — including
        nearest_node (round-2 review: unfused RK4 used to echo the stale
        incoming value, the fused kernel writes the fresh start anchor)."""
        from kelp.numerics.integrators import rk4_step, _rk4_step_unfused
        mesh = _node_mesh()
        dt = 3600.0
        u = 0.022 * 111120.0 / dt        # endpoint exits, stages stay in
        vel = VelocityField(u=np.full(mesh.n_nodes, u),
                            v=np.zeros(mesh.n_nodes),
                            eta=np.zeros(mesh.n_nodes), uv_layout="node")
        # one survivor + one OOB failure + one off-hull INTERP failure
        st = make_state(np.array([0.0, 0.185, 0.35]),
                        np.array([0.0, 0.0, 0.0]))
        st = st.replace(failure_reason=np.zeros(3, dtype=np.int8))
        out_f = rk4_step(st, vel, mesh, dt, 1.0,
                         vel_next=vel, dt_to_next=dt,
                         uv_interp="barycentric")
        out_u = _rk4_step_unfused(st, vel, mesh, dt, 1.0, 0, False,
                                  uv_interp="barycentric",
                                  vel_next=vel, dt_to_next=dt)
        assert out_f.status[1] == ParticleStatus.OUTSIDE
        assert out_f.status[2] == ParticleStatus.OUTSIDE
        assert out_f.status[0] == ParticleStatus.ALIVE
        _assert_state_equal(out_f, out_u)
        np.testing.assert_array_equal(out_f.failure_reason,
                                      out_u.failure_reason)


class TestWrapMeshParity:
    @pytest.mark.parametrize("step_name", ["rk2", "rk4"])
    def test_dateline_band_certified_and_bitwise(self, step_name,
                                                 monkeypatch):
        """Cyclic global band: the certificate's wrap legs (np.roll
        dilation, n_i + nx arc) actually run, the mesh certifies, and
        fused == unfused bitwise for particles crossing the seam."""
        import kelp.numerics.integrators as I
        from kelp.numerics.integrators import (
            rk2_step, rk4_step, _rk2_step_unfused, _rk4_step_unfused)
        calls = {"n": 0}
        real = I._run_fused_2d_bary_tinterp

        def spy(*a, **k):
            calls["n"] += 1
            return real(*a, **k)

        monkeypatch.setattr(I, "_run_fused_2d_bary_tinterp", spy)
        mesh = _wrap_band_mesh()
        assert grid_hash_dense_for_elements(mesh) is True
        lonn, latn = mesh.nodes_lon, mesh.nodes_lat
        u = 5.0 + 0.01 * latn
        v = 0.2 * np.sin(lonn * 0.02)
        vel = VelocityField(u=u, v=v, eta=np.zeros(lonn.size),
                            uv_layout="node")
        vel2 = VelocityField(u=u * 1.1, v=v * 0.9,
                             eta=np.zeros(lonn.size), uv_layout="node")
        # seeds straddling the dateline seam
        st = make_state(np.array([179.5, -179.5, 178.0, -178.0, 0.0]),
                        np.array([45.0, 50.0, 55.0, 42.0, 48.0]))
        step = rk2_step if step_name == "rk2" else rk4_step
        unf = _rk2_step_unfused if step_name == "rk2" else _rk4_step_unfused
        out_f = step(st, vel, mesh, 3600.0, 1.0,
                     vel_next=vel2, dt_to_next=86400.0,
                     uv_interp="barycentric")
        assert calls["n"] == 1
        out_u = unf(st, vel, mesh, 3600.0, 1.0, 0, False,
                    uv_interp="barycentric",
                    vel_next=vel2, dt_to_next=86400.0)
        assert np.all(out_f.status == ParticleStatus.ALIVE)
        _assert_state_equal(out_f, out_u)


class TestCertificateInvariant:
    def test_contains_accepted_points_map_into_raster(self):
        """SOUNDNESS INVARIANT pin: any point `_element_contains` accepts
        maps (via `_gh_cell_of`) into the cell range the certificate
        rasters for that element — the first-valid-vertex wrapped-delta
        bbox.  Re-anchoring the prefilter (e.g. on the centroid) without
        the raster would break this and silently unsound the
        certificate."""
        rng = np.random.default_rng(11)
        for mesh in (_node_mesh(), _wrap_band_mesh()):
            cs = mesh._gh_cell_size
            lon_min, lat_min = mesh._gh_lon_min, mesh._gh_lat_min
            nx, ny = mesh._gh_nx, mesh._gh_ny
            wrap = mesh._gh_wrap_lon
            checked = 0
            for e in range(0, mesh.n_elements,
                           max(1, mesh.n_elements // 40)):
                verts = [int(n) for n in mesh.elem_nodes[e] if n >= 0]
                vl = mesh.nodes_lon[verts]
                vt = mesh.nodes_lat[verts]
                ref = vl[0]
                dl = ((vl - ref + 180.0) % 360.0) - 180.0
                # raster range exactly as _gh_elements_covered computes it
                ci_w, cj_s = _gh_cell_of(ref + dl.min(), vt.min(), cs,
                                         lon_min, lat_min, nx, ny, wrap)
                ci_e, cj_n = _gh_cell_of(ref + dl.max(), vt.max(), cs,
                                         lon_min, lat_min, nx, ny, wrap)
                n_i = ci_e - ci_w
                if n_i < 0:
                    n_i += nx
                cols = {(ci_w + t) % nx for t in range(n_i + 1)}
                # random convex combinations of the vertices (in wrapped
                # delta space, dateline-safe)
                w = rng.random((30, len(verts)))
                w /= w.sum(axis=1, keepdims=True)
                pl = ref + w @ dl
                pl = ((pl + 180.0) % 360.0) - 180.0
                pt = w @ vt
                for k in range(30):
                    if not _element_contains(e, pl[k], pt[k],
                                             mesh.elem_nodes,
                                             mesh.nodes_lon,
                                             mesh.nodes_lat):
                        continue
                    ci, cj = _gh_cell_of(pl[k], pt[k], cs, lon_min,
                                         lat_min, nx, ny, wrap)
                    assert ci in cols and cj_s <= cj <= cj_n, (
                        f"element {e}: accepted point ({pl[k]}, {pt[k]}) "
                        f"maps to cell ({ci},{cj}) outside raster")
                    checked += 1
            assert checked > 100          # the pin is not vacuous
