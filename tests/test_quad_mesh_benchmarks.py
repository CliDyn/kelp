"""Physical-scenario mega-tests on QUAD meshes (FESOM-C / structured
style) — the quad twin of test_analytical_3d_benchmarks.py.

Scope (review 2026-08-16): the Wachspress weight MATH is already
unit-covered on random non-rectangular quads in test_polygon_weights.py;
this file adds the INTEGRATOR-LEVEL scenarios — containment
(`_element_contains` quad branch, incl. the polygon reject path on
sheared quads), the fused 3-D Euler kernel's in-kernel quad containment
(pinned by a call counter, not just claimed), the W-barycentric quad
branch made OBSERVABLE by a horizontally-linear W field (a wrong weight
vector breaks linear precision — a horizontally-constant W would accept
any normalized weights), element-layout barycentric u/v on quads, the
FESOM-C connectivity sanitizer, and advection ACROSS a mixed
tri-as-quad band.  Note the z-level + quad + element-layout u/v
combination is hand-built (readers produce sigma quads / node-layout
z-level) — the kernel paths are shared, the entry point is synthetic.
"""
import numpy as np
import pytest

from kelp.core.geometry import DEG2RAD, METERS_PER_DEG_LAT
from kelp.core.mesh import (Mesh, build_node_to_elems, build_search,
                                find_nearest_nodes,
                                find_containing_elements,
                                validate_quad_convexity,
                                _element_contains)
from kelp.core.io import VelocityField
from kelp.core.particles import ParticleStatus, make_state
from kelp.numerics.integrators import euler_step, rk2_step, rk4_step
from kelp.numerics.vertical import interp_w_spatial
from kelp.readers.fesomc import _sanitize_connectivity


def _build_quad_patch_mesh(
    center_lon: float = 10.0,
    center_lat: float = 45.0,
    span: float = 0.3,
    nx: int = 81,
    ny: int = 81,
    z_levels=None,
    bottom_depth: float = 200.0,
    mixed: bool = False,
    shear: float = 0.0,
):
    """Structured convex-quad patch (CCW), max_verts = 4.

    ``shear`` tilts the columns (x += shear·(y−y₀) + curvature term), so
    quads become generic convex trapezoids whose bbox is STRICTLY larger
    than the polygon — the containment reject path then has to do real
    work (an axis-aligned rectangle is decided by the bbox prefilter
    alone).  Convexity is enforced by ``validate_quad_convexity``.

    ``mixed=True`` splits every quad of the middle row band into two
    triangles, alternately encoded FESOM-C-style as ``[a, b, c, -1]``
    and ``[a, b, c, a]`` — a valid tessellation exercising the padded
    and repeated-vertex 4-column encodings alongside true quads.  (The
    ``[a,b,c,a]`` form is what `_sanitize_connectivity` would REMOVE at
    load; carrying it here deliberately tests the defensive in-kernel
    compaction, not a reader-produced layout.)
    """
    if z_levels is None:
        z_levels = np.linspace(0.0, bottom_depth, 21)
    xs = np.linspace(center_lon - span, center_lon + span, nx)
    ys = np.linspace(center_lat - span, center_lat + span, ny)
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel().astype(np.float64)
    lat = gy.ravel().astype(np.float64)
    if shear != 0.0:
        dy = lat - center_lat
        lon = lon + shear * dy + 0.3 * shear * dy * dy / max(span, 1e-9)
    n_nodes = lon.shape[0]

    elems = []
    j_mid = (ny - 1) // 2
    flip = True
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            quad = [n0, n0 + 1, n0 + nx + 1, n0 + nx]     # CCW convex
            if mixed and j == j_mid:
                a, b, c, d = quad
                if flip:
                    elems.append([a, b, c, -1])           # padded tri
                    elems.append([a, c, d, a])            # repeated-vertex tri
                else:
                    elems.append([a, b, c, a])
                    elems.append([a, c, d, -1])
                flip = not flip
            else:
                elems.append(quad)
    elem_nodes = np.array(elems, dtype=np.int32)

    def _center(e):
        nds = [int(n) for n in e if n >= 0]
        if len(nds) == 4 and nds[3] == nds[0]:
            nds = nds[:3]
        return lon[nds].mean(), lat[nds].mean()

    centers = np.array([_center(e) for e in elem_nodes])
    n2e, cnt = build_node_to_elems(n_nodes, elem_nodes)
    mesh = Mesh(
        name="quad_patch",
        nodes_lon=lon, nodes_lat=lat,
        depth=np.full(n_nodes, bottom_depth, dtype=np.float64),
        elem_nodes=elem_nodes,
        elem_center_lon=centers[:, 0], elem_center_lat=centers[:, 1],
        node_to_elems=n2e, node_to_elems_count=cnt,
        sigma=np.array(z_levels, dtype=np.float64),
    )
    build_search(mesh)
    # the convexity contract quads rely on (reader load-time check)
    validate_quad_convexity(elem_nodes, lon, lat, source="test_patch")
    return mesh


def _local_xy(lon, lat, lon0, lat0):
    cos0 = np.cos(lat0 * DEG2RAD)
    return ((lon - lon0) * cos0 * METERS_PER_DEG_LAT,
            (lat - lat0) * METERS_PER_DEG_LAT)


def _xy_to_lonlat(x, y, lon0, lat0):
    cos0 = np.cos(lat0 * DEG2RAD)
    return (lon0 + x / (cos0 * METERS_PER_DEG_LAT),
            lat0 + y / METERS_PER_DEG_LAT)


# =====================================================================
# 1. Linear incompressible strain on quads (3-D, z-level)
# =====================================================================

class TestLinearStrainQuad:
    """u=ax, v=by, w=-(a+b)z on a pure-quad z-level mesh."""

    LON0, LAT0 = 10.0, 45.0
    A, B = 2e-5, 1e-5

    @pytest.fixture
    def setup(self):
        mesh = _build_quad_patch_mesh(self.LON0, self.LAT0,
                                      span=0.3, nx=121, ny=121)
        z_levels = np.abs(mesh.sigma)
        K, ne, nn = z_levels.shape[0], mesh.n_elements, mesh.n_nodes
        x_e, y_e = _local_xy(mesh.elem_center_lon, mesh.elem_center_lat,
                             self.LON0, self.LAT0)
        u3 = np.tile(self.A * x_e, (K - 1, 1))
        v3 = np.tile(self.B * y_e, (K - 1, 1))
        w3 = np.tile((-(self.A + self.B) * z_levels)[:, None], (1, nn))
        vel = VelocityField(u=u3, v=v3, eta=np.zeros(nn), w=w3)
        rng = np.random.default_rng(99)
        state = make_state(self.LON0 + rng.uniform(0.005, 0.015, 10),
                           self.LAT0 + rng.uniform(0.005, 0.015, 10),
                           np.full(10, 50.0))
        return mesh, vel, state

    def _analytical(self, lon0, lat0, z0, t):
        x0, y0 = _local_xy(lon0, lat0, self.LON0, self.LAT0)
        lon_t, lat_t = _xy_to_lonlat(x0 * np.exp(self.A * t),
                                     y0 * np.exp(self.B * t),
                                     self.LON0, self.LAT0)
        return lon_t, lat_t, z0 * np.exp(-(self.A + self.B) * t)

    def test_trajectory_accuracy_euler_fused(self, setup, monkeypatch):
        """Fused 3-D Euler on quads — the fused routing is PINNED by a
        call counter (a silent fall to the unfused path would produce
        the same trajectory and hide a dispatch regression)."""
        import kelp.numerics.integrators as I
        calls = {"n": 0}
        real = I._run_fused_euler_3d

        def spy(*a, **k):
            calls["n"] += 1
            return real(*a, **k)

        monkeypatch.setattr(I, "_run_fused_euler_3d", spy)
        mesh, vel, state = setup
        lon0, lat0, z0 = state.lon.copy(), state.lat.copy(), state.z.copy()
        dt, n = 60.0, 100
        for _ in range(n):
            state = euler_step(state, vel, mesh, dt, time_dir=1.0,
                               w_interp="leastsq")
        assert calls["n"] == n                    # fused path every step
        assert np.all(state.status == ParticleStatus.ALIVE)
        lon_a, lat_a, z_a = self._analytical(lon0, lat0, z0, dt * n)
        # measured errors: lon 8.8e-7, lat 2.7e-7 deg, z 6.8e-3 m
        np.testing.assert_allclose(state.lon, lon_a, rtol=0, atol=2e-6)
        np.testing.assert_allclose(state.lat, lat_a, rtol=0, atol=2e-6)
        np.testing.assert_allclose(state.z, z_a, rtol=0, atol=8e-3)

    def test_trajectory_accuracy_rk4_bary_w(self, setup):
        """Unfused RK4 with W = barycentric: the quad Wachspress branch
        of `_interp_w_barycentric_single` runs every stage.  (The value
        pin for WRONG weights lives in TestWBarycentricQuadLinear — this
        W is horizontally constant so any normalized weights reproduce
        it; here the pins are the trajectory + tight z/lat.)"""
        mesh, vel, state = setup
        lon0, lat0, z0 = state.lon.copy(), state.lat.copy(), state.z.copy()
        dt, n = 300.0, 20
        for _ in range(n):
            state = rk4_step(state, vel, mesh, dt, time_dir=1.0,
                             w_interp="barycentric")
        assert np.all(state.status == ParticleStatus.ALIVE)
        lon_a, lat_a, z_a = self._analytical(lon0, lat0, z0, dt * n)
        # measured errors: lon 3.5e-7 deg, lat 2.1e-14 deg, z 4.1e-10 m
        np.testing.assert_allclose(state.lon, lon_a, rtol=0, atol=2e-6)
        np.testing.assert_allclose(state.lat, lat_a, rtol=0, atol=1e-12)
        np.testing.assert_allclose(state.z, z_a, rtol=0, atol=1e-8)

    def test_all_w_methods_smoke(self, setup):
        """All 4 W methods agree on a horizontally-constant W (smoke —
        the DISCRIMINATING quad-weight pin is
        TestWBarycentricQuadLinear); every particle must stay alive,
        else reverted positions agree trivially."""
        mesh, vel, state = setup
        res = {}
        for m in ["nearest", "leastsq", "barycentric", "idw"]:
            s = state.replace()
            for _ in range(50):
                s = euler_step(s, vel, mesh, 60.0, time_dir=1.0, w_interp=m)
            assert np.all(s.status == ParticleStatus.ALIVE), m
            res[m] = s
        ref = res["leastsq"]
        for m in ["nearest", "barycentric", "idw"]:
            assert np.max(np.abs(res[m].z - ref.z)) < 0.1


# =====================================================================
# 2. W barycentric on quads: linear precision (the weight-sensitive pin)
# =====================================================================

class TestWBarycentricQuadLinear:
    """A horizontally-LINEAR W makes the quad Wachspress weights
    observable: any wrong weight vector (uniform 0.25, mass on one
    vertex, permuted) breaks linear precision, where the correct
    product-form weights reproduce the field exactly.  Sheared quads —
    the bbox alone cannot answer containment."""

    LON0, LAT0 = 10.0, 45.0
    W_A, W_B = 3e-4, 5e-3        # w = W_A + W_B * (lon - LON0)

    def _setup(self):
        mesh = _build_quad_patch_mesh(self.LON0, self.LAT0,
                                      span=0.1, nx=41, ny=41,
                                      shear=0.25,
                                      z_levels=np.linspace(0.0, 100.0, 11))
        K = mesh.sigma.shape[0]
        w_field = np.tile(
            self.W_A + self.W_B * (mesh.nodes_lon - self.LON0),
            (K, 1))
        rng = np.random.default_rng(21)
        P = 60
        lons = self.LON0 + rng.uniform(-0.06, 0.06, P)
        lats = self.LAT0 + rng.uniform(-0.06, 0.06, P)
        zs = rng.uniform(5.0, 95.0, P)
        nn = find_nearest_nodes(mesh, lons, lats)
        alive = np.ones(P, dtype=bool)
        return mesh, w_field, lons, lats, zs, nn, alive

    def test_linear_field_reproduced_exactly(self):
        mesh, w_field, lons, lats, zs, nn, alive = self._setup()
        got = interp_w_spatial(
            w_field, np.abs(mesh.sigma), zs, lons, lats, nn, alive,
            time_dir=1.0, method="barycentric", mesh=mesh)
        expected = self.W_A + self.W_B * (lons - self.LON0)
        # Wachspress linear precision in the projected plane; the only
        # residual is the metric curvature across a 0.005-deg cell.
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-7)

    def test_field_discriminates_wrong_weights(self):
        """The SAME field separates the methods (nearest is piecewise
        constant), proving a wrong weight vector could not hide: a
        uniform-0.25 blend would land at the cell mean, orders of
        magnitude outside the linear-exactness pin above."""
        mesh, w_field, lons, lats, zs, nn, alive = self._setup()
        bary = interp_w_spatial(
            w_field, np.abs(mesh.sigma), zs, lons, lats, nn, alive,
            time_dir=1.0, method="barycentric", mesh=mesh)
        near = interp_w_spatial(
            w_field, np.abs(mesh.sigma), zs, lons, lats, nn, alive,
            time_dir=1.0, method="nearest", mesh=mesh)
        sep = np.max(np.abs(bary - near))
        # nearest snaps to the node value: typical offset ~ W_B * (half
        # cell) ~ 1.2e-5; require a robustly nonzero separation, far
        # above the 1e-7 exactness tolerance.
        assert sep > 2e-6


# =====================================================================
# 3. Solid-body rotation on quads (2-D) — leastsq AND barycentric u/v
# =====================================================================

class TestSolidRotationQuad:
    """u = -Ω·y, v = Ω·x: circular orbits, radius conserved."""

    LON0, LAT0 = 10.0, 45.0
    OMEGA = 2.0 * np.pi / 43200.0          # one turn / 12 h

    @pytest.fixture
    def setup(self):
        mesh = _build_quad_patch_mesh(self.LON0, self.LAT0,
                                      span=0.3, nx=121, ny=121,
                                      z_levels=np.array([0.0, 100.0]))
        x_e, y_e = _local_xy(mesh.elem_center_lon, mesh.elem_center_lat,
                             self.LON0, self.LAT0)
        vel = VelocityField(u=-self.OMEGA * y_e, v=self.OMEGA * x_e,
                            eta=np.zeros(mesh.n_nodes))
        state = make_state(np.array([self.LON0 + 0.05, self.LON0 - 0.03]),
                           np.array([self.LAT0, self.LAT0 + 0.04]))
        return mesh, vel, state

    @pytest.mark.parametrize("uv_interp", ["leastsq", "barycentric"])
    def test_quarter_rotation(self, setup, uv_interp):
        mesh, vel, state = setup
        x0, y0 = _local_xy(state.lon.copy(), state.lat.copy(),
                           self.LON0, self.LAT0)
        T4 = 0.25 * 2.0 * np.pi / self.OMEGA
        n = 108
        dt = T4 / n
        s = state.replace()
        for _ in range(n):
            s = rk4_step(s, vel, mesh, dt, time_dir=1.0,
                         uv_interp=uv_interp)
        assert np.all(s.status == ParticleStatus.ALIVE)
        # quarter turn: (x, y) -> (-y, x)
        lon_a, lat_a = _xy_to_lonlat(-y0, x0, self.LON0, self.LAT0)
        # Measured error 2.06e-5 deg (leastsq) / 2.04e-5 (barycentric) —
        # spatial-interpolation floor, dt-independent.  Symmetric honest
        # tolerance with ~2.4x headroom, rtol=0 so the number is real.
        np.testing.assert_allclose(s.lon, lon_a, rtol=0, atol=5e-5)
        np.testing.assert_allclose(s.lat, lat_a, rtol=0, atol=5e-5)

    @pytest.mark.parametrize("uv_interp", ["leastsq", "barycentric"])
    def test_radius_conserved(self, setup, uv_interp):
        mesh, vel, state = setup
        x0, y0 = _local_xy(state.lon.copy(), state.lat.copy(),
                           self.LON0, self.LAT0)
        r0 = np.hypot(x0, y0)
        s = state.replace()
        for _ in range(120):
            s = rk4_step(s, vel, mesh, 120.0, time_dir=1.0,
                         uv_interp=uv_interp)
        x1, y1 = _local_xy(s.lon, s.lat, self.LON0, self.LAT0)
        r1 = np.hypot(x1, y1)
        # measured drift 3.4e-4 relative (both interpolators)
        np.testing.assert_allclose(r1, r0, rtol=2e-3)


# =====================================================================
# 4. Convergent downwelling on quads (3-D): z grows exponentially
# =====================================================================

class TestDownwellingQuad:
    """u = -c·x, v = -c·y, w = +2c·z (positive-down: downwelling)."""

    LON0, LAT0 = 10.0, 45.0
    C = 1.5e-5

    @pytest.fixture
    def setup(self):
        mesh = _build_quad_patch_mesh(self.LON0, self.LAT0,
                                      span=0.3, nx=121, ny=121,
                                      bottom_depth=400.0,
                                      z_levels=np.linspace(0.0, 400.0, 41))
        z_levels = np.abs(mesh.sigma)
        K, nn = z_levels.shape[0], mesh.n_nodes
        x_e, y_e = _local_xy(mesh.elem_center_lon, mesh.elem_center_lat,
                             self.LON0, self.LAT0)
        u3 = np.tile(-self.C * x_e, (K - 1, 1))
        v3 = np.tile(-self.C * y_e, (K - 1, 1))
        w3 = np.tile((2.0 * self.C * z_levels)[:, None], (1, nn))
        vel = VelocityField(u=u3, v=v3, eta=np.zeros(nn), w=w3)
        rng = np.random.default_rng(7)
        state = make_state(self.LON0 + rng.uniform(-0.05, 0.05, 8),
                           self.LAT0 + rng.uniform(-0.05, 0.05, 8),
                           np.full(8, 20.0))
        return mesh, vel, state

    def test_sinks_and_contracts(self, setup):
        mesh, vel, state = setup
        x0, y0 = _local_xy(state.lon.copy(), state.lat.copy(),
                           self.LON0, self.LAT0)
        z0 = state.z.copy()
        dt, n = 120.0, 100
        s = state.replace()
        for _ in range(n):
            s = rk2_step(s, vel, mesh, dt, time_dir=1.0,
                         w_interp="barycentric")
        assert np.all(s.status == ParticleStatus.ALIVE)
        t = dt * n
        x1, y1 = _local_xy(s.lon, s.lat, self.LON0, self.LAT0)
        np.testing.assert_allclose(x1, x0 * np.exp(-self.C * t), atol=8.0)
        np.testing.assert_allclose(y1, y0 * np.exp(-self.C * t), atol=8.0)
        np.testing.assert_allclose(s.z, z0 * np.exp(2.0 * self.C * t),
                                   rtol=2e-3)
        assert np.all(s.z > z0)               # genuinely downwelling


# =====================================================================
# 5. Mixed tri-as-quad tessellation: advection ACROSS the encodings
# =====================================================================

class TestMixedEncodingAdvection:
    """Uniform DIAGONAL flow drives every particle from the pure-quad
    region south of the band, THROUGH the row of FESOM-C-encoded
    triangles ([a,b,c,-1] / [a,b,c,a]), and out into quads again —
    exercising the pad/repeated-vertex compaction inside containment
    and the vertex-weight dispatch at both quad↔tri interfaces."""

    LON0, LAT0 = 10.0, 45.0
    U0 = 0.4
    V0 = 0.12                    # ~0.052 deg north over the run

    @pytest.fixture
    def setup(self):
        mesh = _build_quad_patch_mesh(self.LON0, self.LAT0,
                                      span=0.3, nx=81, ny=81, mixed=True,
                                      z_levels=np.array([0.0, 100.0]))
        ne = mesh.n_elements
        vel = VelocityField(u=np.full(ne, self.U0),
                            v=np.full(ne, self.V0),
                            eta=np.zeros(mesh.n_nodes))
        # all seeds strictly SOUTH of the band (row spans LAT0..LAT0+0.0075)
        state = make_state(
            np.full(6, self.LON0 - 0.15),
            self.LAT0 - 0.03 + np.linspace(0.0, 0.02, 6),
        )
        return mesh, vel, state

    @pytest.mark.parametrize("uv_interp", ["leastsq", "barycentric"])
    def test_diagonal_advection_exact_and_crosses(self, setup, uv_interp):
        """Constant field ⇒ displacement exact for any consistent
        interpolation; AND every particle is observed INSIDE a
        tri-encoded element at least once (anti-vacuity: the band is
        genuinely traversed, both encodings)."""
        mesh, vel, state = setup
        enc = mesh.elem_nodes
        is_pad = (enc == -1).any(axis=1)
        is_rep = (enc[:, 3] == enc[:, 0]) & (enc[:, 0] >= 0)
        assert is_pad.any() and is_rep.any()
        lon0 = state.lon.copy()
        lat0 = state.lat.copy()
        dt, n = 600.0, 80
        s = state.replace()
        in_tri_ever = np.zeros(s.n_particles, dtype=bool)
        seen_pad = seen_rep = False
        for _ in range(n):
            s = rk2_step(s, vel, mesh, dt, time_dir=1.0,
                         uv_interp=uv_interp)
            nn = find_nearest_nodes(mesh, s.lon, s.lat)
            eids, ind = find_containing_elements(mesh, s.lon, s.lat, nn)
            hit = eids[ind]
            tri_hit = is_pad[hit] | is_rep[hit]
            in_tri_ever[np.where(ind)[0][tri_hit]] = True
            seen_pad |= bool(is_pad[hit].any())
            seen_rep |= bool(is_rep[hit].any())
        assert np.all(s.status == ParticleStatus.ALIVE)
        assert in_tri_ever.all()          # every particle crossed the band
        assert seen_pad and seen_rep      # both encodings visited
        # exact diagonal displacement (metres, spherical step metric)
        t = dt * n
        got_dx = (s.lon - lon0) * np.cos(lat0 * DEG2RAD) * METERS_PER_DEG_LAT
        got_dy = (s.lat - lat0) * METERS_PER_DEG_LAT
        np.testing.assert_allclose(got_dy, self.V0 * t, rtol=1e-6)
        # zonal metres accumulate along drifting lat: cos(lat) shrinks
        # slightly northward — allow that O(dlat*tan(lat)) effect
        np.testing.assert_allclose(got_dx, self.U0 * t, rtol=2e-3)


# =====================================================================
# 6. FESOM-C connectivity sanitizer (1-based padded nv)
# =====================================================================

class TestSanitizeConnectivity:
    def test_one_based_quad(self):
        out = _sanitize_connectivity(np.array([[1, 2, 5, 4]]))
        np.testing.assert_array_equal(out, [[0, 1, 4, 3]])
        assert out.dtype == np.int32

    def test_zero_padding_is_fill(self):
        out = _sanitize_connectivity(np.array([[1, 2, 3, 0]]))
        np.testing.assert_array_equal(out, [[0, 1, 2, -1]])

    def test_negative_padding_is_fill(self):
        out = _sanitize_connectivity(np.array([[1, 2, 3, -1]]))
        np.testing.assert_array_equal(out, [[0, 1, 2, -1]])

    def test_repeated_vertex_deduped(self):
        out = _sanitize_connectivity(np.array([[1, 2, 3, 1]]))
        np.testing.assert_array_equal(out, [[0, 1, 2, -1]])

    def test_mid_row_duplicate_deduped_left_packed(self):
        out = _sanitize_connectivity(np.array([[4, 4, 7, 9]]))
        np.testing.assert_array_equal(out, [[3, 6, 8, -1]])

    def test_all_pad_row(self):
        out = _sanitize_connectivity(np.array([[0, 0, 0, 0]]))
        np.testing.assert_array_equal(out, [[-1, -1, -1, -1]])

    def test_three_column_input(self):
        """FESOM tri connectivity (3 columns) passes through untouched
        apart from the 1-based shift."""
        out = _sanitize_connectivity(np.array([[1, 2, 3], [4, 5, 6]]))
        np.testing.assert_array_equal(out, [[0, 1, 2], [3, 4, 5]])

    def test_row_collapsing_below_three_nodes(self):
        """A row deduping to 2 unique nodes yields [a, b, -1, -1] —
        downstream containment treats it as n_valid < 3 (never contains
        anything: a silently-dead element, the documented behaviour for
        malformed input)."""
        out = _sanitize_connectivity(np.array([[3, 5, 3, 5]]))
        np.testing.assert_array_equal(out, [[2, 4, -1, -1]])

    def test_mixed_rows(self):
        out = _sanitize_connectivity(np.array([
            [1, 2, 5, 4],
            [2, 3, 6, 2],
            [3, 6, 5, 0],
        ]))
        np.testing.assert_array_equal(out, [
            [0, 1, 4, 3],
            [1, 2, 5, -1],
            [2, 5, 4, -1],
        ])


# =====================================================================
# 7. Quad containment on SHEARED quads (the polygon test does the work)
# =====================================================================

class TestQuadContainment:
    """On sheared (non-rectangular) convex quads the bbox prefilter is
    strictly larger than the polygon, so the fan containment test — not
    the bbox — decides every verdict probed here."""

    def _mesh(self):
        return _build_quad_patch_mesh(span=0.1, nx=6, ny=6, shear=0.35,
                                      z_levels=np.array([0.0, 10.0]))

    def _quad(self, mesh, e):
        nds = [int(n) for n in mesh.elem_nodes[e]]
        return nds, mesh.nodes_lon[nds], mesh.nodes_lat[nds]

    def test_inside_outside_edge_vertex(self):
        mesh = self._mesh()
        e = 12
        _, lc, tc = self._quad(mesh, e)
        cx, cy = lc.mean(), tc.mean()
        assert _element_contains(e, cx, cy, mesh.elem_nodes,
                                 mesh.nodes_lon, mesh.nodes_lat)
        # on an edge midpoint and exactly on a vertex: contained
        mx, my = (lc[0] + lc[1]) / 2, (tc[0] + tc[1]) / 2
        assert _element_contains(e, mx, my, mesh.elem_nodes,
                                 mesh.nodes_lon, mesh.nodes_lat)
        assert _element_contains(e, lc[0], tc[0], mesh.elem_nodes,
                                 mesh.nodes_lon, mesh.nodes_lat)
        # far outside: not contained
        assert not _element_contains(e, cx + 1.0, cy, mesh.elem_nodes,
                                     mesh.nodes_lon, mesh.nodes_lat)

    def test_near_miss_inside_bbox_outside_quad(self):
        """The discriminating probe rectangles cannot provide: a point
        INSIDE the quad's bbox but OUTSIDE the sheared polygon must be
        rejected by the polygon test itself."""
        mesh = self._mesh()
        e = 12
        _, lc, tc = self._quad(mesh, e)
        # midpoint of the slanted 'left' edge (vertices 3 -> 0), pushed
        # slightly outward along -x: still inside the bbox (the shear
        # tilts the edge into the bbox interior), outside the polygon
        mx, my = (lc[3] + lc[0]) / 2, (tc[3] + tc[0]) / 2
        px = mx - 2e-4
        assert lc.min() < px < lc.max()          # inside the bbox
        assert not _element_contains(e, px, my, mesh.elem_nodes,
                                     mesh.nodes_lon, mesh.nodes_lat)

    def test_winding_invariance(self):
        """A CW-wound copy of a convex quad answers containment
        identically to the CCW original."""
        mesh = self._mesh()
        e = 12
        rev = mesh.elem_nodes.copy()
        rev[e] = rev[e][::-1]                 # CW winding
        rng = np.random.default_rng(3)
        _, lc, tc = self._quad(mesh, e)
        for _ in range(50):
            px = rng.uniform(lc.min() - 0.01, lc.max() + 0.01)
            py = rng.uniform(tc.min() - 0.01, tc.max() + 0.01)
            a = _element_contains(e, px, py, mesh.elem_nodes,
                                  mesh.nodes_lon, mesh.nodes_lat)
            b = _element_contains(e, px, py, rev,
                                  mesh.nodes_lon, mesh.nodes_lat)
            assert a == b

    def test_matches_fan_triangulation_reference(self):
        """Quad containment == union of the two fan triangles (the exact
        geometric definition for a convex quad), sampled densely on the
        SHEARED quad.  Points within an epsilon shell of a polygon EDGE
        (distance < 1e-7 deg, where tolerance conventions may
        legitimately differ) are excluded from the comparison."""
        mesh = self._mesh()
        e = 7
        _, lc, tc = self._quad(mesh, e)

        def _in_tri(px, py, i0, i1, i2):
            x0, y0 = lc[i0], tc[i0]
            x1, y1 = lc[i1], tc[i1]
            x2, y2 = lc[i2], tc[i2]
            d = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            wa = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / d
            wb = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / d
            wc = 1.0 - wa - wb
            eps = -1e-12
            return wa >= eps and wb >= eps and wc >= eps

        def _edge_dist(px, py):
            dmin = np.inf
            for k in range(4):
                x0, y0 = lc[k], tc[k]
                x1, y1 = lc[(k + 1) % 4], tc[(k + 1) % 4]
                ex, ey = x1 - x0, y1 - y0
                tpar = ((px - x0) * ex + (py - y0) * ey) / (ex * ex + ey * ey)
                tpar = min(max(tpar, 0.0), 1.0)
                dmin = min(dmin, np.hypot(px - (x0 + tpar * ex),
                                          py - (y0 + tpar * ey)))
            return dmin

        rng = np.random.default_rng(5)
        n_checked = 0
        for _ in range(500):
            px = rng.uniform(lc.min() - 0.005, lc.max() + 0.005)
            py = rng.uniform(tc.min() - 0.005, tc.max() + 0.005)
            if _edge_dist(px, py) < 1e-7:
                continue                        # epsilon shell excluded
            ref = _in_tri(px, py, 0, 1, 2) or _in_tri(px, py, 0, 2, 3)
            got = _element_contains(e, px, py, mesh.elem_nodes,
                                    mesh.nodes_lon, mesh.nodes_lat)
            assert ref == got, (px, py)
            n_checked += 1
        assert n_checked > 300                  # anti-vacuity
