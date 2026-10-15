"""Analytical 3D Lagrangian benchmark tests.

Six test cases with known analytical trajectories to validate
the 3D particle tracker on unstructured meshes.

All tests use a small lon/lat patch to minimize spherical geometry effects.
Analytical trajectories computed in local Cartesian, converted to lon/lat
via fixed cos(lat₀).
"""
from __future__ import annotations

import numpy as np
import pytest

from kelp.numerics.integrators import euler_step
from kelp.numerics.interpolate import VelocityField
from kelp.core.mesh import Mesh, build_node_to_elems, build_search
from kelp.core.particles import make_state, ParticleStatus
from kelp.core.geometry import METERS_PER_DEG_LAT, DEG2RAD


# =====================================================================
# Shared mesh builder
# =====================================================================

def _build_patch_mesh(
    center_lon: float = 10.0,
    center_lat: float = 45.0,
    span: float = 0.1,
    nx: int = 21,
    ny: int = 21,
    z_levels=None,
    bottom_depth: float = 200.0,
    jitter: float = 0.0,
    seed: int = 42,
):
    """Build a small triangular mesh patch centred at (center_lon, center_lat).

    Parameters
    ----------
    span : float
        Half-extent in degrees (mesh covers ±span).
    jitter : float
        Fraction of grid spacing for random node perturbation.
    """
    if z_levels is None:
        z_levels = np.linspace(0.0, bottom_depth, 21)

    xs = np.linspace(center_lon - span, center_lon + span, nx)
    ys = np.linspace(center_lat - span, center_lat + span, ny)
    gx, gy = np.meshgrid(xs, ys)
    lon = gx.ravel().astype(np.float64)
    lat = gy.ravel().astype(np.float64)
    n_nodes = lon.shape[0]
    depth = np.full(n_nodes, bottom_depth, dtype=np.float64)

    if jitter > 0:
        rng = np.random.default_rng(seed)
        dx = (xs[1] - xs[0]) * jitter
        dy = (ys[1] - ys[0]) * jitter
        for i in range(n_nodes):
            ix, iy = i % nx, i // nx
            if 0 < ix < nx - 1 and 0 < iy < ny - 1:
                lon[i] += rng.uniform(-dx, dx)
                lat[i] += rng.uniform(-dy, dy)

    elems = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            n0 = j * nx + i
            elems.append([n0, n0 + 1, n0 + nx])
            elems.append([n0 + 1, n0 + nx + 1, n0 + nx])
    elem_nodes = np.array(elems, dtype=np.int32)

    # Validate mesh: check for degenerate elements (near-zero area)
    # and ensure consistent orientation (all positive area = CCW)
    if jitter > 0:
        for e in range(elem_nodes.shape[0]):
            n0, n1, n2 = elem_nodes[e]
            # Signed area via cross product
            ax, ay = lon[n1] - lon[n0], lat[n1] - lat[n0]
            bx, by = lon[n2] - lon[n0], lat[n2] - lat[n0]
            area2 = ax * by - ay * bx
            if abs(area2) < 1e-16:
                raise ValueError(f"Degenerate element {e}: nodes {n0},{n1},{n2}, "
                                 f"area={area2:.2e}")
            # If orientation flipped by jitter, swap two vertices to fix
            if area2 < 0:
                elem_nodes[e, 1], elem_nodes[e, 2] = elem_nodes[e, 2], elem_nodes[e, 1]

    lon_c = np.array([lon[e].mean() for e in elem_nodes], dtype=np.float64)
    lat_c = np.array([lat[e].mean() for e in elem_nodes], dtype=np.float64)
    n2e, n2e_count = build_node_to_elems(n_nodes, elem_nodes)

    mesh = Mesh(
        name="test_patch",
        nodes_lon=lon, nodes_lat=lat, depth=depth,
        elem_nodes=elem_nodes,
        elem_center_lon=lon_c, elem_center_lat=lat_c,
        node_to_elems=n2e, node_to_elems_count=n2e_count,
        sigma=np.array(z_levels, dtype=np.float64),
    )
    build_search(mesh)
    return mesh


def _local_xy(lon, lat, lon0, lat0):
    """Convert lon/lat to local Cartesian (metres) around (lon0, lat0)."""
    cos_lat0 = np.cos(lat0 * DEG2RAD)
    x = (lon - lon0) * cos_lat0 * METERS_PER_DEG_LAT
    y = (lat - lat0) * METERS_PER_DEG_LAT
    return x, y


def _xy_to_lonlat(x, y, lon0, lat0):
    """Convert local Cartesian (metres) back to lon/lat."""
    cos_lat0 = np.cos(lat0 * DEG2RAD)
    lon = lon0 + x / (cos_lat0 * METERS_PER_DEG_LAT)
    lat = lat0 + y / METERS_PER_DEG_LAT
    return lon, lat


# =====================================================================
# Test 1: Linear Incompressible Strain
# =====================================================================

class TestLinearStrain:
    """u=ax, v=by, w=-(a+b)z — exponential trajectories."""

    LON0 = 10.0
    LAT0 = 45.0
    A = 2e-5   # s⁻¹
    B = 1e-5   # s⁻¹

    @pytest.fixture
    def setup(self):
        mesh = _build_patch_mesh(self.LON0, self.LAT0, span=0.3, nx=161, ny=161)
        z_levels = np.abs(mesh.sigma)
        K = z_levels.shape[0]
        ne = mesh.n_elements
        nn = mesh.n_nodes
        cos_lat0 = np.cos(self.LAT0 * DEG2RAD)

        # u, v on element centres
        x_e = (mesh.elem_center_lon - self.LON0) * cos_lat0 * METERS_PER_DEG_LAT
        y_e = (mesh.elem_center_lat - self.LAT0) * METERS_PER_DEG_LAT

        u_3d = np.zeros((K - 1, ne), dtype=np.float64)
        v_3d = np.zeros((K - 1, ne), dtype=np.float64)
        for k in range(K - 1):
            u_3d[k, :] = self.A * x_e
            v_3d[k, :] = self.B * y_e

        # w on nodes at z-level interfaces: w = -(a+b)*z
        w_3d = np.zeros((K, nn), dtype=np.float64)
        for k in range(K):
            w_3d[k, :] = -(self.A + self.B) * z_levels[k]

        vel = VelocityField(
            u=u_3d, v=v_3d,
            eta=np.zeros(nn, dtype=np.float64),
            w=w_3d,
        )

        # Seed 10 particles slightly offset from centre
        rng = np.random.default_rng(99)
        lons = self.LON0 + rng.uniform(0.005, 0.015, 10)
        lats = self.LAT0 + rng.uniform(0.005, 0.015, 10)
        z0 = np.full(10, 50.0)
        state = make_state(lons, lats, z0)
        return mesh, vel, state

    def _analytical(self, lon_init, lat_init, z_init, t):
        """Compute analytical position at time t."""
        x0, y0 = _local_xy(lon_init, lat_init, self.LON0, self.LAT0)
        x_t = x0 * np.exp(self.A * t)
        y_t = y0 * np.exp(self.B * t)
        z_t = z_init * np.exp(-(self.A + self.B) * t)
        lon_t, lat_t = _xy_to_lonlat(x_t, y_t, self.LON0, self.LAT0)
        return lon_t, lat_t, z_t

    def test_trajectory_accuracy(self, setup):
        """Numerical trajectories match analytical within Euler error."""
        mesh, vel, state = setup
        lon_init = state.lon.copy()
        lat_init = state.lat.copy()
        z_init = state.z.copy()

        dt = 60.0
        n_steps = 100
        for _ in range(n_steps):
            state = euler_step(state, vel, mesh, dt, time_dir=1.0,
                               w_interp="leastsq")

        assert np.all(state.status == ParticleStatus.ALIVE)

        t = dt * n_steps
        lon_a, lat_a, z_a = self._analytical(lon_init, lat_init, z_init, t)

        dlon = state.lon - lon_a
        dlat = state.lat - lat_a
        dz = state.z - z_a
        cos_lat0 = np.cos(self.LAT0 * DEG2RAD)

        print(f"\n  Linear strain: dt={dt}s, {n_steps} steps, T={t:.0f}s")
        print(f"  |Δlon| max={np.max(np.abs(dlon)):.2e}°  "
              f"({np.max(np.abs(dlon))*cos_lat0*METERS_PER_DEG_LAT:.3f} m)")
        print(f"  |Δlat| max={np.max(np.abs(dlat)):.2e}°  "
              f"({np.max(np.abs(dlat))*METERS_PER_DEG_LAT:.3f} m)")
        print(f"  |Δz|   max={np.max(np.abs(dz)):.4f} m  "
              f"rel={np.max(np.abs(dz)/np.abs(z_a)):.2e}")

        np.testing.assert_allclose(state.lon, lon_a, atol=2e-6,
                                   err_msg="lon mismatch")
        np.testing.assert_allclose(state.lat, lat_a, atol=2e-6,
                                   err_msg="lat mismatch")
        np.testing.assert_allclose(state.z, z_a, atol=0.008,
                                   err_msg="z mismatch")

    def test_z_decreases(self, setup):
        """With a,b > 0, z should decrease (particles rise)."""
        mesh, vel, state = setup
        z_init = state.z.copy()
        for _ in range(50):
            state = euler_step(state, vel, mesh, 60.0, time_dir=1.0,
                               w_interp="leastsq")
        # w = -(a+b)*z < 0 for z > 0 → z decreases (particles rise)
        assert np.all(state.z < z_init)

    def test_all_methods_close(self, setup):
        """All 4 W interp methods should give similar results."""
        mesh, vel, state = setup
        results = {}
        for method in ["nearest", "leastsq", "barycentric", "idw"]:
            s = state.replace()  # copy
            for _ in range(50):
                s = euler_step(s, vel, mesh, 60.0, time_dir=1.0,
                               w_interp=method)
            results[method] = s

        ref = results["leastsq"]
        for method in ["nearest", "barycentric", "idw"]:
            dz = np.max(np.abs(results[method].z - ref.z))
            assert dz < 0.1, f"{method} vs leastsq: ΔZ max = {dz:.4f}m"


# =====================================================================
# Test 3: Burgers Vortex
# =====================================================================

class TestBurgersVortex:
    """Burgers vortex: radial convergence + axial stretching + azimuthal swirl.

    Analytical: r(t) = r₀·exp(-αt/2), z(t) = z₀·exp(αt).
    θ(t) involves Ei — we skip it and check only r and z.
    """

    LON0 = 10.0
    LAT0 = 45.0
    ALPHA = 5e-5       # s⁻¹, strain rate
    GAMMA = 1.5e4      # m²/s, circulation
    RC = 5000.0         # m, core radius

    @pytest.fixture
    def setup(self):
        mesh = _build_patch_mesh(self.LON0, self.LAT0, span=0.3, nx=161, ny=161)
        z_levels = np.abs(mesh.sigma)
        K = z_levels.shape[0]
        ne = mesh.n_elements
        nn = mesh.n_nodes
        cos_lat0 = np.cos(self.LAT0 * DEG2RAD)

        # Element centres in local Cartesian
        x_e = (mesh.elem_center_lon - self.LON0) * cos_lat0 * METERS_PER_DEG_LAT
        y_e = (mesh.elem_center_lat - self.LAT0) * METERS_PER_DEG_LAT
        r_e = np.sqrt(x_e**2 + y_e**2)

        # Burgers velocity on element centres
        # u_r = -(α/2)r, u_θ = Γ/(2πr)(1-exp(-r²/rc²))
        # Cartesian: u = u_r·cos(θ) - u_θ·sin(θ), v = u_r·sin(θ) + u_θ·cos(θ)
        u_3d = np.zeros((K - 1, ne), dtype=np.float64)
        v_3d = np.zeros((K - 1, ne), dtype=np.float64)

        for k in range(K - 1):
            for e in range(ne):
                x, y = x_e[e], y_e[e]
                r = r_e[e]
                # Radial component
                ur = -self.ALPHA / 2 * r
                # Azimuthal: (1-exp(-r²/rc²))/r → r/rc² as r→0
                if r > 1.0:  # > 1 metre
                    utheta = self.GAMMA / (2 * np.pi * r) * (
                        1 - np.exp(-r**2 / self.RC**2)
                    )
                else:
                    utheta = self.GAMMA / (2 * np.pi * self.RC**2) * r

                # Convert (ur, utheta) → (u, v)
                if r > 1e-10:
                    cos_t, sin_t = x / r, y / r
                else:
                    cos_t, sin_t = 1.0, 0.0
                u_3d[k, e] = ur * cos_t - utheta * sin_t
                v_3d[k, e] = ur * sin_t + utheta * cos_t

        # w = α·z on nodes (positive-down: stretching downward)
        w_3d = np.zeros((K, nn), dtype=np.float64)
        for k in range(K):
            w_3d[k, :] = self.ALPHA * z_levels[k]

        vel = VelocityField(
            u=u_3d, v=v_3d,
            eta=np.zeros(nn, dtype=np.float64),
            w=w_3d,
        )

        # Seed 5 particles at r=5km from centre, z=10m (shallow to avoid bottom)
        r_m = 5000.0
        cos_lat0_val = np.cos(self.LAT0 * DEG2RAD)
        angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        lons = self.LON0 + (r_m * np.cos(angles)) / (cos_lat0_val * METERS_PER_DEG_LAT)
        lats = self.LAT0 + (r_m * np.sin(angles)) / METERS_PER_DEG_LAT
        z0 = np.full(5, 10.0)  # shallow start
        state = make_state(lons, lats, z0)
        return mesh, vel, state

    def test_radial_convergence(self, setup):
        """r(t) should follow r₀·exp(-αt/2)."""
        mesh, vel, state = setup
        cos_lat0 = np.cos(self.LAT0 * DEG2RAD)

        x0, y0 = _local_xy(state.lon, state.lat, self.LON0, self.LAT0)
        r0 = np.sqrt(x0**2 + y0**2)

        dt = 60.0
        n_steps = 200  # 200 min
        for _ in range(n_steps):
            state = euler_step(state, vel, mesh, dt, time_dir=1.0,
                               w_interp="leastsq")

        assert np.all(state.status == ParticleStatus.ALIVE)

        t = dt * n_steps
        r_analytical = r0 * np.exp(-self.ALPHA * t / 2)

        x_f, y_f = _local_xy(state.lon, state.lat, self.LON0, self.LAT0)
        r_f = np.sqrt(x_f**2 + y_f**2)

        dr = np.abs(r_f - r_analytical)
        print(f"\n  Burgers vortex radial: dt={dt}s, {n_steps} steps, T={t:.0f}s")
        print(f"  r₀={r0.mean():.0f}m → r_analytical={r_analytical.mean():.0f}m, "
              f"r_numerical={r_f.mean():.0f}m")
        print(f"  |Δr| max={np.max(dr):.1f}m  rel={np.max(dr/r_analytical):.2e}")

        # Euler error: allow 0.5% relative
        np.testing.assert_allclose(r_f, r_analytical, rtol=0.005,
                                   err_msg="radial convergence mismatch")

    def test_vertical_stretching(self, setup):
        """z(t) should follow z₀·exp(αt)."""
        mesh, vel, state = setup
        z0 = state.z.copy()

        dt = 60.0
        n_steps = 200
        for _ in range(n_steps):
            state = euler_step(state, vel, mesh, dt, time_dir=1.0,
                               w_interp="leastsq")

        assert np.all(state.status == ParticleStatus.ALIVE)

        t = dt * n_steps
        z_analytical = z0 * np.exp(self.ALPHA * t)

        dz = np.abs(state.z - z_analytical)
        print(f"\n  Burgers vortex vertical: dt={dt}s, {n_steps} steps, T={t:.0f}s")
        print(f"  z₀={z0.mean():.1f}m → z_analytical={z_analytical.mean():.2f}m, "
              f"z_numerical={state.z.mean():.2f}m")
        print(f"  |Δz| max={np.max(dz):.4f}m  rel={np.max(dz/z_analytical):.2e}")

        np.testing.assert_allclose(state.z, z_analytical, rtol=0.0015,
                                   err_msg="vertical stretching mismatch")


# =====================================================================
# Test 2: Solid-Body Rotation + Vertical Drift (Helical)
# =====================================================================

class TestHelicalRotation:
    """u=-Ωy, v=Ωx, w=W₀ — circular orbit + linear z drift."""

    LON0 = 10.0
    LAT0 = 45.0
    OMEGA = 2 * np.pi / (24 * 3600)  # one rotation per day
    W0 = 0.001  # 1 mm/s downward

    @pytest.fixture
    def setup(self):
        mesh = _build_patch_mesh(self.LON0, self.LAT0, span=0.3, nx=161, ny=161)
        z_levels = np.abs(mesh.sigma)
        K = z_levels.shape[0]
        ne = mesh.n_elements
        nn = mesh.n_nodes
        cos_lat0 = np.cos(self.LAT0 * DEG2RAD)

        # u = -Ω·y, v = Ω·x on element centres
        x_e = (mesh.elem_center_lon - self.LON0) * cos_lat0 * METERS_PER_DEG_LAT
        y_e = (mesh.elem_center_lat - self.LAT0) * METERS_PER_DEG_LAT

        u_3d = np.zeros((K - 1, ne), dtype=np.float64)
        v_3d = np.zeros((K - 1, ne), dtype=np.float64)
        for k in range(K - 1):
            u_3d[k, :] = -self.OMEGA * y_e
            v_3d[k, :] = self.OMEGA * x_e

        # w = W₀ constant on all nodes
        w_3d = np.full((K, nn), self.W0, dtype=np.float64)

        vel = VelocityField(
            u=u_3d, v=v_3d,
            eta=np.zeros(nn, dtype=np.float64),
            w=w_3d,
        )

        # Seed 5 particles on circular orbit of radius ~2 km
        r_m = 2000.0  # 2 km radius in metres
        cos_lat0_val = np.cos(self.LAT0 * DEG2RAD)
        angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        # Scale lon by 1/cos(lat) so orbit is circular in metres
        lons = self.LON0 + (r_m * np.cos(angles)) / (cos_lat0_val * METERS_PER_DEG_LAT)
        lats = self.LAT0 + (r_m * np.sin(angles)) / METERS_PER_DEG_LAT
        z0 = np.full(5, 20.0)
        state = make_state(lons, lats, z0)
        return mesh, vel, state

    def _analytical(self, lon_init, lat_init, z_init, t):
        """Analytical helical trajectory."""
        x0, y0 = _local_xy(lon_init, lat_init, self.LON0, self.LAT0)
        x_t = x0 * np.cos(self.OMEGA * t) - y0 * np.sin(self.OMEGA * t)
        y_t = x0 * np.sin(self.OMEGA * t) + y0 * np.cos(self.OMEGA * t)
        z_t = z_init + self.W0 * t
        lon_t, lat_t = _xy_to_lonlat(x_t, y_t, self.LON0, self.LAT0)
        return lon_t, lat_t, z_t

    def test_quarter_rotation(self, setup):
        """After 6 hours (quarter rotation), check position."""
        mesh, vel, state = setup
        lon_init = state.lon.copy()
        lat_init = state.lat.copy()
        z_init = state.z.copy()

        dt = 60.0   # 1 min (smaller dt for rotation accuracy)
        t_total = 6 * 3600  # 6 hours
        n_steps = int(t_total / dt)

        for _ in range(n_steps):
            state = euler_step(state, vel, mesh, dt, time_dir=1.0,
                               w_interp="leastsq")

        assert np.all(state.status == ParticleStatus.ALIVE)

        lon_a, lat_a, z_a = self._analytical(lon_init, lat_init, z_init, t_total)

        dlon = state.lon - lon_a
        dlat = state.lat - lat_a
        dz = state.z - z_a
        cos_lat0 = np.cos(self.LAT0 * DEG2RAD)

        # Radial error
        x_n, y_n = _local_xy(state.lon, state.lat, self.LON0, self.LAT0)
        x_a, y_a = _local_xy(lon_a, lat_a, self.LON0, self.LAT0)
        dr = np.sqrt((x_n - x_a)**2 + (y_n - y_a)**2)

        print(f"\n  Helical rotation: dt={dt}s, {n_steps} steps, T={t_total/3600:.1f}h")
        print(f"  |Δlon| max={np.max(np.abs(dlon)):.2e}°  "
              f"({np.max(np.abs(dlon))*cos_lat0*METERS_PER_DEG_LAT:.1f} m)")
        print(f"  |Δlat| max={np.max(np.abs(dlat)):.2e}°  "
              f"({np.max(np.abs(dlat))*METERS_PER_DEG_LAT:.1f} m)")
        print(f"  |Δr|   max={np.max(dr):.1f} m")
        print(f"  |Δz|   max={np.max(np.abs(dz)):.4f} m")

        # Euler with dt=60s over 6h — tight thresholds
        np.testing.assert_allclose(state.lon, lon_a, atol=1e-4,
                                   err_msg="lon after quarter rotation")
        np.testing.assert_allclose(state.lat, lat_a, atol=1e-4,
                                   err_msg="lat after quarter rotation")
        np.testing.assert_allclose(state.z, z_a, atol=0.5,
                                   err_msg="z after quarter rotation")

    def test_z_increases(self, setup):
        """With W₀ > 0, z should increase (particles sink)."""
        mesh, vel, state = setup
        z_init = state.z.copy()
        for _ in range(100):
            state = euler_step(state, vel, mesh, 60.0, time_dir=1.0,
                               w_interp="leastsq")
        assert np.all(state.z > z_init)

    def test_radius_conserved(self, setup):
        """Orbital radius should be approximately conserved."""
        mesh, vel, state = setup
        x0, y0 = _local_xy(state.lon, state.lat, self.LON0, self.LAT0)
        r0 = np.sqrt(x0**2 + y0**2)

        for _ in range(360):  # 360 × 60s = 6h
            state = euler_step(state, vel, mesh, 60.0, time_dir=1.0,
                               w_interp="leastsq")

        x_f, y_f = _local_xy(state.lon, state.lat, self.LON0, self.LAT0)
        r_f = np.sqrt(x_f**2 + y_f**2)

        # Radius within 0.5% with dt=60s
        np.testing.assert_allclose(r_f, r0, rtol=0.005,
                                   err_msg="orbital radius not conserved")


# =====================================================================
# Test 4: Inertial Circles
# =====================================================================

class TestInertialCircles:
    """u=V·sin(ft), v=V·cos(ft), w=0 — inertial oscillation.

    Spatially uniform, time-dependent.  Trajectory:
    x(t) = x₀ + (V/f)·(1 - cos(ft))
    y(t) = y₀ + (V/f)·sin(ft)
    z(t) = z₀
    """

    LON0 = 10.0
    LAT0 = 45.0
    V = 0.3          # m/s
    F = 1e-4         # s⁻¹, Coriolis parameter at ~45°N

    @pytest.fixture
    def setup(self):
        mesh = _build_patch_mesh(self.LON0, self.LAT0, span=0.3, nx=161, ny=161)
        return mesh

    def _make_vel(self, mesh, t):
        """Build velocity field at time t: u=V·sin(ft), v=V·cos(ft), w=0."""
        z_levels = np.abs(mesh.sigma)
        K = z_levels.shape[0]
        ne = mesh.n_elements
        nn = mesh.n_nodes

        u_val = self.V * np.sin(self.F * t)
        v_val = self.V * np.cos(self.F * t)

        u_3d = np.full((K - 1, ne), u_val, dtype=np.float64)
        v_3d = np.full((K - 1, ne), v_val, dtype=np.float64)
        w_3d = np.zeros((K, nn), dtype=np.float64)

        return VelocityField(
            u=u_3d, v=v_3d,
            eta=np.zeros(nn, dtype=np.float64),
            w=w_3d,
        )

    def _analytical(self, lon0, lat0, z0, t):
        """Analytical inertial circle position at time t."""
        cos_lat0 = np.cos(self.LAT0 * DEG2RAD)
        R = self.V / self.F  # inertial radius in metres

        dx = R * (1 - np.cos(self.F * t))  # metres
        dy = R * np.sin(self.F * t)

        lon_t = lon0 + dx / (cos_lat0 * METERS_PER_DEG_LAT)
        lat_t = lat0 + dy / METERS_PER_DEG_LAT
        return lon_t, lat_t, z0

    def test_inertial_circle(self, setup):
        """Particle traces inertial circle over one period."""
        mesh = setup
        lons = np.array([self.LON0])
        lats = np.array([self.LAT0])
        z0 = np.array([50.0])
        state = make_state(lons, lats, z0)

        dt = 60.0
        T_period = 2 * np.pi / self.F  # ~17.5 hours
        n_steps = int(T_period / dt)

        for step in range(n_steps):
            t = step * dt
            vel = self._make_vel(mesh, t)
            state = euler_step(state, vel, mesh, dt, time_dir=1.0,
                               w_interp="leastsq")

        assert state.status[0] == ParticleStatus.ALIVE

        t_final = n_steps * dt
        lon_a, lat_a, z_a = self._analytical(lons[0], lats[0], z0[0], t_final)

        dlon = abs(state.lon[0] - lon_a)
        dlat = abs(state.lat[0] - lat_a)
        cos_lat0 = np.cos(self.LAT0 * DEG2RAD)
        dr = np.sqrt((dlon * cos_lat0 * METERS_PER_DEG_LAT)**2 +
                     (dlat * METERS_PER_DEG_LAT)**2)
        R = self.V / self.F

        print(f"\n  Inertial circle: dt={dt}s, {n_steps} steps, "
              f"T={t_final/3600:.1f}h (period={T_period/3600:.1f}h)")
        print(f"  Inertial radius R={R:.0f}m")
        print(f"  After one period: |Δr|={dr:.1f}m  "
              f"(rel={dr/R:.4f} of R)")
        print(f"  |Δz|={abs(state.z[0] - z0[0]):.4f}m")

        # After one full period, should return close to start
        # Euler accumulates ~1-2% error over one rotation
        assert dr < R * 0.01, f"Closure error {dr:.1f}m > 1% of R={R:.0f}m"
        assert abs(state.z[0] - z0[0]) < 0.01, "z should not change (w=0)"

    def test_max_displacement(self, setup):
        """Maximum displacement should be ~2R (diameter of circle)."""
        mesh = setup
        state = make_state(
            np.array([self.LON0]), np.array([self.LAT0]), np.array([50.0])
        )
        cos_lat0 = np.cos(self.LAT0 * DEG2RAD)
        R = self.V / self.F

        dt = 60.0
        max_dist = 0.0
        n_steps = int(np.pi / self.F / dt)  # half period (max displacement)

        for step in range(n_steps):
            vel = self._make_vel(mesh, step * dt)
            state = euler_step(state, vel, mesh, dt, time_dir=1.0,
                               w_interp="leastsq")
            dx = (state.lon[0] - self.LON0) * cos_lat0 * METERS_PER_DEG_LAT
            dy = (state.lat[0] - self.LAT0) * METERS_PER_DEG_LAT
            dist = np.sqrt(dx**2 + dy**2)
            max_dist = max(max_dist, dist)

        print(f"\n  Max displacement: {max_dist:.0f}m (expected ~2R={2*R:.0f}m)")

        # Max displacement should be close to 2R (diameter)
        assert abs(max_dist - 2 * R) < R * 0.03, \
            f"Max displacement {max_dist:.0f}m not close to 2R={2*R:.0f}m"


# =====================================================================
# Test 5: Ekman Spiral
# =====================================================================

class TestEkmanSpiral:
    """u(z), v(z) from Ekman theory, w=0 — straight-line drift at each depth.

    u(z) = V₀·exp(-z/D)·cos(-z/D)
    v(z) = V₀·exp(-z/D)·sin(-z/D)
    where z is positive-down depth and D is Ekman depth.

    Trajectory: x(t)=x₀+u(z₀)t, y(t)=y₀+v(z₀)t, z(t)=z₀.
    Velocity direction rotates clockwise with depth (Northern Hemisphere).
    """

    LON0 = 10.0
    LAT0 = 45.0
    V0 = 0.3     # m/s surface current
    D_EKMAN = 50.0  # Ekman depth in metres

    @pytest.fixture
    def setup(self):
        mesh = _build_patch_mesh(self.LON0, self.LAT0, span=0.3, nx=161, ny=161)
        z_levels = np.abs(mesh.sigma)
        K = z_levels.shape[0]
        ne = mesh.n_elements
        nn = mesh.n_nodes

        # u, v on elements — depends only on z (mid-layer depth)
        u_3d = np.zeros((K - 1, ne), dtype=np.float64)
        v_3d = np.zeros((K - 1, ne), dtype=np.float64)
        for k in range(K - 1):
            z_mid = 0.5 * (z_levels[k] + z_levels[k + 1])
            u_val = self.V0 * np.exp(-z_mid / self.D_EKMAN) * np.cos(-z_mid / self.D_EKMAN)
            v_val = self.V0 * np.exp(-z_mid / self.D_EKMAN) * np.sin(-z_mid / self.D_EKMAN)
            u_3d[k, :] = u_val
            v_3d[k, :] = v_val

        # w = 0
        w_3d = np.zeros((K, nn), dtype=np.float64)

        vel = VelocityField(
            u=u_3d, v=v_3d,
            eta=np.zeros(nn, dtype=np.float64),
            w=w_3d,
        )

        # Seed particles at mid-layer depths (avoid z-level boundaries)
        depths = np.array([5.0, 25.0, 55.0, 95.0, 145.0])
        n_p = len(depths)
        lons = np.full(n_p, self.LON0)
        lats = np.full(n_p, self.LAT0)
        state = make_state(lons, lats, depths)
        return mesh, vel, state, depths

    def _analytical(self, z0, t):
        """Analytical displacement (metres) at depth z0 over time t."""
        u = self.V0 * np.exp(-z0 / self.D_EKMAN) * np.cos(-z0 / self.D_EKMAN)
        v = self.V0 * np.exp(-z0 / self.D_EKMAN) * np.sin(-z0 / self.D_EKMAN)
        return u * t, v * t  # dx, dy in metres

    def test_ekman_drift(self, setup):
        """Particles at different depths drift in Ekman-rotated directions."""
        mesh, vel, state, depths = setup
        lon_init = state.lon.copy()
        lat_init = state.lat.copy()
        cos_lat0 = np.cos(self.LAT0 * DEG2RAD)

        dt = 60.0
        n_steps = 200  # ~3.3 hours
        for _ in range(n_steps):
            state = euler_step(state, vel, mesh, dt, time_dir=1.0,
                               w_interp="leastsq")

        assert np.all(state.status == ParticleStatus.ALIVE)

        t = dt * n_steps
        print(f"\n  Ekman spiral: dt={dt}s, {n_steps} steps, T={t/3600:.1f}h")

        for i, z0 in enumerate(depths):
            dx_a, dy_a = self._analytical(z0, t)
            dx_n = (state.lon[i] - lon_init[i]) * cos_lat0 * METERS_PER_DEG_LAT
            dy_n = (state.lat[i] - lat_init[i]) * METERS_PER_DEG_LAT

            err = np.sqrt((dx_n - dx_a)**2 + (dy_n - dy_a)**2)
            speed = np.sqrt(dx_a**2 + dy_a**2) / t
            angle = np.degrees(np.arctan2(dy_a, dx_a))

            print(f"  z={z0:5.0f}m: speed={speed:.3f}m/s  "
                  f"dir={angle:+6.1f}°  |Δr|={err:.2f}m")

            # Allow 0.5% of total displacement or 0.5m, whichever larger
            disp = np.sqrt(dx_a**2 + dy_a**2)
            threshold = max(disp * 0.005, 0.5)
            assert err < threshold, (
                f"z={z0}m: displacement error {err:.2f}m > threshold {threshold:.2f}m"
            )

    def test_z_unchanged(self, setup):
        """z should remain constant (w=0)."""
        mesh, vel, state, depths = setup
        z_init = state.z.copy()
        for _ in range(100):
            state = euler_step(state, vel, mesh, 60.0, time_dir=1.0,
                               w_interp="leastsq")
        np.testing.assert_allclose(state.z, z_init, atol=0.001,
                                   err_msg="z changed with w=0")

    def test_velocity_rotates_with_depth(self, setup):
        """Surface current direction vs deep current — should be rotated."""
        _, _, _, depths = setup
        # Analytical directions
        angles = []
        for z in depths:
            u = self.V0 * np.exp(-z / self.D_EKMAN) * np.cos(-z / self.D_EKMAN)
            v = self.V0 * np.exp(-z / self.D_EKMAN) * np.sin(-z / self.D_EKMAN)
            angles.append(np.degrees(np.arctan2(v, u)))

        # Surface (z=5m) should be roughly 0° (eastward)
        # Deep (z=150m) should be rotated ~-170° (nearly opposite, very weak)
        rotation = angles[-1] - angles[0]
        print(f"\n  Ekman rotation: surface dir={angles[0]:.1f}°, "
              f"deep dir={angles[-1]:.1f}°, rotation={rotation:.1f}°")
        # At least 90° rotation between surface and 150m
        assert abs(rotation) > 90, f"Not enough rotation: {rotation:.1f}°"


# =====================================================================
# Test 6: Ekman Pumping (Cellular Flow)
# =====================================================================

class TestEkmanPumping:
    """Cellular flow with w(x,z) — tests horizontal W interpolation.

    u(x,z) = U₀·sin(kx)·sin(2πz/H)
    w(x,z) = -(kHU₀)/(2π)·cos(kx)·[1 - cos(2πz/H)]
    v = 0

    div(u) = 0.  ψ = (HU₀)/(2π)·sin(kx)·[1 - cos(2πz/H)] = const along
    trajectories.  This is the key test for horizontal W spatial interpolation
    since w depends on BOTH x and z.
    """

    LON0 = 10.0
    LAT0 = 45.0
    U0 = 0.2     # m/s
    H = 200.0    # m
    K_WAVE = 2 * np.pi / 8_000  # wavelength ~8 km (short → tests W interp quality)

    def _psi(self, lon, z):
        """Streamfunction ψ(x, z)."""
        cos_lat0 = np.cos(self.LAT0 * DEG2RAD)
        x = (lon - self.LON0) * cos_lat0 * METERS_PER_DEG_LAT
        return (self.H * self.U0 / (2 * np.pi)) * np.sin(self.K_WAVE * x) * (
            1 - np.cos(2 * np.pi * z / self.H)
        )

    @pytest.fixture
    def setup(self):
        mesh = _build_patch_mesh(self.LON0, self.LAT0, span=0.3, nx=161, ny=161,
                                 bottom_depth=self.H)
        z_levels = np.abs(mesh.sigma)
        K = z_levels.shape[0]
        ne = mesh.n_elements
        nn = mesh.n_nodes
        cos_lat0 = np.cos(self.LAT0 * DEG2RAD)

        # Element centres in metres
        x_e = (mesh.elem_center_lon - self.LON0) * cos_lat0 * METERS_PER_DEG_LAT
        # Node positions in metres
        x_n = (mesh.nodes_lon - self.LON0) * cos_lat0 * METERS_PER_DEG_LAT

        # u on elements: u = U₀·sin(kx)·sin(2πz/H)
        u_3d = np.zeros((K - 1, ne), dtype=np.float64)
        v_3d = np.zeros((K - 1, ne), dtype=np.float64)
        for k in range(K - 1):
            z_mid = 0.5 * (z_levels[k] + z_levels[k + 1])
            u_3d[k, :] = self.U0 * np.sin(self.K_WAVE * x_e) * np.sin(
                2 * np.pi * z_mid / self.H
            )

        # w on nodes: w = -(kHU₀)/(2π)·cos(kx)·[1 - cos(2πz/H)]
        w_coeff = -self.K_WAVE * self.H * self.U0 / (2 * np.pi)
        w_3d = np.zeros((K, nn), dtype=np.float64)
        for k in range(K):
            w_3d[k, :] = w_coeff * np.cos(self.K_WAVE * x_n) * (
                1 - np.cos(2 * np.pi * z_levels[k] / self.H)
            )

        vel = VelocityField(
            u=u_3d, v=v_3d,
            eta=np.zeros(nn, dtype=np.float64),
            w=w_3d,
        )

        # Seed particles at mid-depth, offset from cell centre
        # x ≈ 7-8 km (inside one cell), z = 50-100m (mid-depth, strong W)
        rng = np.random.default_rng(77)
        n_p = 10
        x_offset_m = rng.uniform(5000, 10000, n_p)  # 5-10 km from centre
        lons = self.LON0 + x_offset_m / (cos_lat0 * METERS_PER_DEG_LAT)
        lats = np.full(n_p, self.LAT0)
        z0 = rng.uniform(40, 120, n_p)  # mid-depth range
        state = make_state(lons, lats, z0)
        return mesh, vel, state

    def test_psi_conservation(self, setup):
        """ψ should be conserved along trajectories."""
        mesh, vel, state = setup
        psi0 = self._psi(state.lon, state.z)

        dt = 60.0
        n_steps = 200  # ~3.3 hours
        for _ in range(n_steps):
            state = euler_step(state, vel, mesh, dt, time_dir=1.0,
                               w_interp="leastsq")

        assert np.all(state.status == ParticleStatus.ALIVE)

        psi_f = self._psi(state.lon, state.z)
        drift = np.abs(psi_f - psi0)
        scale = np.maximum(np.abs(psi0), 1e-12)
        rel = drift / scale

        print(f"\n  Ekman pumping ψ conservation: dt={dt}s, {n_steps} steps")
        print(f"  |Δψ| rel mean={np.mean(rel):.4%}  max={np.max(rel):.4%}")
        print(f"  z range: {state.z.min():.1f}–{state.z.max():.1f}m")

        # ψ relative drift should be < 3% (200 steps with 8km wavelength)
        assert np.max(rel) < 0.03, (
            f"ψ rel drift {np.max(rel):.4f} > 3%"
        )

    @pytest.mark.parametrize("method", ["nearest", "leastsq", "barycentric", "idw"])
    def test_psi_all_methods(self, setup, method):
        """All W interp methods should conserve ψ reasonably."""
        mesh, vel, state = setup
        psi0 = self._psi(state.lon, state.z)

        dt = 60.0
        n_steps = 100
        for _ in range(n_steps):
            state = euler_step(state, vel, mesh, dt, time_dir=1.0,
                               w_interp=method)

        alive = state.status == ParticleStatus.ALIVE
        if not np.any(alive):
            pytest.skip(f"No alive particles with method={method}")

        psi_f = self._psi(state.lon, state.z)
        drift = np.abs(psi_f[alive] - psi0[alive])
        scale = np.maximum(np.abs(psi0[alive]), 1e-12)
        rel = drift / scale

        print(f"\n  Ekman pumping [{method}]: ψ rel mean={np.mean(rel):.4%}  "
              f"max={np.max(rel):.4%}")

        assert np.max(rel) < 0.05, (
            f"method={method}: ψ rel drift {np.max(rel):.4f} > 5%"
        )

    def test_w_depends_on_x(self, setup):
        """W should vary across particles at same depth but different x."""
        mesh, vel, state = setup
        # This test verifies the field setup: w should differ between particles
        # because they're at different x positions
        from kelp.numerics.vertical import interp_w_linear
        z_levels = np.abs(mesh.sigma)
        nn = np.zeros(state.n_particles, dtype=np.int32)
        from kelp.core.mesh import find_nearest_nodes
        nn[:] = find_nearest_nodes(mesh, state.lon, state.lat)
        alive = state.alive_mask

        W = interp_w_linear(vel.w, z_levels, state.z, nn, alive, 1.0)
        w_range = np.max(W) - np.min(W)
        print(f"\n  W range across particles: {w_range:.6f} m/s "
              f"(min={np.min(W):.6f}, max={np.max(W):.6f})")
        assert w_range > 1e-5, "W should vary with x position"
