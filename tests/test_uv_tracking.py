"""uv_interp wiring: fail-fast, fused-path preservation, e2e, metadata.

Wiring of ``numerics.uv_interp`` ("leastsq" default / "barycentric")
through the integrators and the tracking layer (Task 4).  The fused
Numba kernels and the leastsq core stay untouched; uv_interp only adds
dispatch (kwargs, gates, registry lookups) in the unfused paths.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import netCDF4

from kelp.core.config import (
    AppConfig, DataConfig, MeshConfig, ModeConfig, NumericsConfig, TimeConfig,
)
from kelp.core.geometry import DEG2RAD, METERS_PER_DEG_LAT
from kelp.core.io import VelocityField
from kelp.core.mesh import (
    Mesh, build_node_to_elems, build_search, find_nearest_nodes,
)
from kelp.core.particles import (
    FailureReason, ParticleState, ParticleStatus, make_state,
)
from kelp.numerics.diffusion import DiffusionStepArgs
from kelp.numerics.integrators import euler_step
from kelp.readers import get_reader
from kelp.tracking import run_single_realization


# ---------------------------------------------------------------------------
# Synthetic FESOM-C dataset (3x3 grid, 9 nodes, 8 triangles, centred 5,5).
#
# Adapted from tests/test_uv_layout.py::_fesomc_ds — adds a configurable
# constant ``u_val`` (over time and space) with ``v=0`` so a constant-
# latitude closed-form displacement holds.  ``uv_dim`` selects whether
# u2d/v2d sit on "nele" (element layout, 8 items) or "node" (9 items);
# the reader classifies layout by matching the spatial dim size to
# n_elements vs n_nodes (9 != 8 keeps the two unambiguous).
# ---------------------------------------------------------------------------

def _make_fesomc_const(tmp_path, uv_dim="nele", u_val=1.0, n_time=4):
    import pandas as pd
    n_nodes, n_elem = 9, 8
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
        "u2d": (["time", uv_dim], np.full((n_time, size), u_val)),
        "v2d": (["time", uv_dim], np.zeros((n_time, size))),
        "eta": (["time", "node"], np.zeros((n_time, 9))),
    }, coords={"time": pd.date_range("2020-01-01", periods=n_time,
                                     freq="3600s")})
    p = tmp_path / f"fesomc_{uv_dim}_{u_val}.nc"
    ds.to_netcdf(p)
    return p


def _open(tmp_path, uv_dim, u_val=1.0):
    """Build the FESOM-C file and return (reader, mesh, open_dataset)."""
    p = _make_fesomc_const(tmp_path, uv_dim=uv_dim, u_val=u_val)
    r = get_reader("fesomc")
    mesh = r.load_mesh(SimpleNamespace(meshpath=str(p)))
    ds = r.open_dataset(str(p))
    return r, mesh, ds


# ---------------------------------------------------------------------------
# Spatially-VARYING node-layout field (linear in lon), constant in time.
#
# Single source of truth for the field formula so the in-Python
# VelocityField and the on-disk u2d(node) cannot drift apart.  The field
# varies in space (so stage-2's interpolation method actually matters) but
# is identical at every time index (so temporal lerp / vel_next is a no-op
# and the closed-form midpoint composition stays exact).
# ---------------------------------------------------------------------------

def _U_LIN(lon):
    """u as a function of longitude: u = 0.2 + 0.8*(lon - 4.0), v = 0."""
    return 0.2 + 0.8 * (np.asarray(lon, dtype=np.float64) - 4.0)


def _node_linear_field(mesh):
    """Node-layout VelocityField with u linear in lon (``_U_LIN``), v = 0.

    Matches the on-disk u2d(node) built by ``_make_fesomc_linear_node``
    node-for-node: the FESOM-C reader keeps file node order in
    ``mesh.nodes_lon``, so ``u[i]`` here equals ``u2d[t, i]`` there.
    """
    from kelp.numerics.interpolate import VelocityField
    u = _U_LIN(mesh.nodes_lon)
    return VelocityField(u=u, v=np.zeros_like(u),
                         eta=np.zeros(mesh.n_nodes), w=None,
                         uv_layout="node")


def _make_fesomc_linear_node(tmp_path, n_time=4):
    """FESOM-C file whose u2d(node) is linear in lon (``_U_LIN``), v2d = 0.

    Same dataset structure as ``_make_fesomc_const`` with ``uv_dim="node"``,
    but the constant ``u_val`` is replaced by the per-node ``_U_LIN(lon)``
    (broadcast identically across all time steps).
    """
    import pandas as pd
    n_nodes, n_elem = 9, 8
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
    u_node = _U_LIN(lon)                         # (9,), linear in lon
    u2d = np.broadcast_to(u_node, (n_time, n_nodes)).copy()
    ds = xr.Dataset({
        "lon": (["node"], lon), "lat": (["node"], lat),
        "depth": (["node"], np.full(9, 10.0)),
        "nv": (["nele", "four"], nv),
        "lon_elem": (["nele"], np.array([lon[e].mean() for e in en0])),
        "lat_elem": (["nele"], np.array([lat[e].mean() for e in en0])),
        "sigma_lev": (["sigma"], np.array([0.0, 0.5, 1.0])),
        "u2d": (["time", "node"], u2d),
        "v2d": (["time", "node"], np.zeros((n_time, n_nodes))),
        "eta": (["time", "node"], np.zeros((n_time, 9))),
    }, coords={"time": pd.date_range("2020-01-01", periods=n_time,
                                     freq="3600s")})
    p = tmp_path / "fesomc_node_linear.nc"
    ds.to_netcdf(p)
    return p


def _open_linear(tmp_path):
    """Build the linear-in-lon FESOM-C file; return (reader, mesh, dataset)."""
    p = _make_fesomc_linear_node(tmp_path)
    r = get_reader("fesomc")
    mesh = r.load_mesh(SimpleNamespace(meshpath=str(p)))
    ds = r.open_dataset(str(p))
    return r, mesh, ds


class TestUvInterpWiring:
    def test_node_plus_leastsq_fails_fast(self, tmp_path):
        reader, mesh, ds = _open(tmp_path, "node")
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False)
        numerics = NumericsConfig(dt=600.0, integrator="euler",
                                  output_freq=1)        # default leastsq
        init = make_state(np.array([5.0]), np.array([5.0]))
        with pytest.raises(ValueError, match="barycentric"):
            run_single_realization(mesh, ds, reader, init, mode, numerics,
                                   start_time_idx=0, n_steps=2, temporal_interp=True)
        ds.close()

    def test_default_run_bypasses_python_wrapper(self, tmp_path, monkeypatch):
        # A correctly-FUSED default 2D run calls neither the integrators-
        # module leastsq binding NOR the registry entries; a default run
        # mis-routed through the unfused registry path increments the
        # leastsq spy; a barycentric run must hit the barycentric spy.
        pytest.importorskip("numba")
        import kelp.numerics.integrators as integ
        import kelp.numerics.interpolate as interp_mod
        calls = {"leastsq": 0, "bary": 0}
        real_lsq = interp_mod.interp_leastsq
        real_bary = interp_mod.interp_barycentric

        def spy_lsq(*a, **k):
            calls["leastsq"] += 1
            return real_lsq(*a, **k)

        def spy_bary(*a, **k):
            calls["bary"] += 1
            return real_bary(*a, **k)

        monkeypatch.setattr(integ, "interp_leastsq", spy_lsq)
        monkeypatch.setitem(interp_mod._UV_INTERPOLATORS, "leastsq", spy_lsq)
        monkeypatch.setitem(interp_mod._UV_INTERPOLATORS,
                            "barycentric", spy_bary)

        reader, mesh, ds = _open(tmp_path, "nele")
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False)
        init = make_state(np.array([5.0]), np.array([5.0]))

        num_default = NumericsConfig(dt=600.0, integrator="euler",
                                     output_freq=1)
        run_single_realization(mesh, ds, reader, init, mode, num_default,
                               start_time_idx=0, n_steps=2, temporal_interp=True)
        assert calls["leastsq"] == 0       # fused kernel, wrappers bypassed

        num_bary = NumericsConfig(dt=600.0, integrator="euler",
                                  output_freq=1, uv_interp="barycentric")
        run_single_realization(mesh, ds, reader, init, mode, num_bary,
                               start_time_idx=0, n_steps=2, temporal_interp=True)
        ds.close()
        assert calls["bary"] >= 1          # unfused path used the registry

    def test_e2e_element_barycentric_matches_analytic(self, tmp_path,
                                                       monkeypatch):
        import kelp.numerics.interpolate as interp_mod
        calls = {"bary": 0}
        real_bary = interp_mod.interp_barycentric

        def spy_bary(*a, **k):
            calls["bary"] += 1
            return real_bary(*a, **k)

        monkeypatch.setitem(interp_mod._UV_INTERPOLATORS,
                            "barycentric", spy_bary)
        monkeypatch.setitem(interp_mod._UV_INTERPOLATORS_SINGLE,
                            "barycentric", interp_mod.interp_barycentric_single)

        reader, mesh, ds = _open(tmp_path, "nele", u_val=1.0)
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False)
        numerics = NumericsConfig(dt=3600.0, integrator="rk2",
                                  output_freq=1, uv_interp="barycentric")
        init = make_state(np.array([5.0]), np.array([5.0]))
        rec = run_single_realization(mesh, ds, reader, init, mode, numerics,
                                     start_time_idx=0, n_steps=3)
        ds.close()
        # On a constant field leastsq and barycentric agree numerically;
        # the spy proves the barycentric path is actually exercised so this
        # test is not vacuous before wiring exists.
        assert calls["bary"] >= 1
        z0 = np.cos(5.0 * DEG2RAD) * METERS_PER_DEG_LAT
        np.testing.assert_allclose(rec.lons[-1][0] - 5.0,
                                   3 * 3600.0 * 1.0 / z0, rtol=1e-12)
        np.testing.assert_allclose(rec.lats[-1][0], 5.0, atol=1e-12)

    def test_e2e_node_barycentric_matches_analytic(self, tmp_path,
                                                    monkeypatch):
        import kelp.numerics.interpolate as interp_mod
        calls = {"bary": 0}
        real_bary = interp_mod.interp_barycentric

        def spy_bary(*a, **k):
            calls["bary"] += 1
            return real_bary(*a, **k)

        monkeypatch.setitem(interp_mod._UV_INTERPOLATORS,
                            "barycentric", spy_bary)
        monkeypatch.setitem(interp_mod._UV_INTERPOLATORS_SINGLE,
                            "barycentric", interp_mod.interp_barycentric_single)

        reader, mesh, ds = _open(tmp_path, "node", u_val=1.0)
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False)
        numerics = NumericsConfig(dt=3600.0, integrator="rk2",
                                  output_freq=1, uv_interp="barycentric")
        init = make_state(np.array([5.0]), np.array([5.0]))
        rec = run_single_realization(mesh, ds, reader, init, mode, numerics,
                                     start_time_idx=0, n_steps=3)
        ds.close()
        assert calls["bary"] >= 1
        z0 = np.cos(5.0 * DEG2RAD) * METERS_PER_DEG_LAT
        np.testing.assert_allclose(rec.lons[-1][0] - 5.0,
                                   3 * 3600.0 * 1.0 / z0, rtol=1e-12)

    def test_rk2_both_stages_use_configured_method(self, tmp_path):
        # Spatially-VARYING node-layout field through rk2 + barycentric:
        # the expected displacement is composed IN-TEST from two explicit
        # interp_barycentric_single calls (midpoint method by hand), so a
        # stage-2 falling back to another interpolator changes the value
        # (on node-layout data leastsq would even read node values by
        # element index).  Kills the stage-2 dispatch mutant that the
        # constant-field e2e tests cannot see.
        from kelp.numerics.interpolate import interp_barycentric_single
        from kelp.core.mesh import find_nearest_nodes
        reader, mesh, ds = _open_linear(tmp_path)
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False)
        numerics = NumericsConfig(dt=3600.0, integrator="rk2",
                                  output_freq=1, uv_interp="barycentric")
        lon0, lat0 = 5.0, 5.0
        init = make_state(np.array([lon0]), np.array([lat0]))
        rec = run_single_realization(mesh, ds, reader, init, mode, numerics,
                                     start_time_idx=0, n_steps=1)
        ds.close()

        # Manual midpoint (RK2) composition mirroring _rk2_step_unfused
        # (src/kelp/numerics/integrators.py).  The field is constant in
        # time, so vel_k1 == vel_k2 == vel and the in-Python field below is
        # exactly what the integrator sees (no temporal lerp).
        #
        # Mirrored arithmetic (_rk2_step_unfused):
        #   stage 1: U1,V1 = interp at start (lon0, lat0)
        #   midpoint: lon_mid,lat_mid = _apply_displacement(.., 0.5*dt)
        #   stage 2: U2,V2 = interp_single at (lon_mid, lat_mid)
        #   final:   lon_new,lat_new = _apply_displacement(.., dt, z0_lat=lat_mid)
        # The base position stays (state.lon, state.lat); the lat update uses the
        # constant METERS_PER_DEG_LAT, and the lon update divides U2 by z0 at the
        # MIDPOINT latitude (proper RK2 — z0_lat=lat_mid).
        # _z0 clamps cos*M to >= 1.0 m/deg; at lat0=5 deg that clamp is inactive,
        # so the bare cos formula below is bit-identical to _z0(lat_mid).
        # wrap_lon/clamp_lat are identities at (5, 5).  (This field has v=0 so
        # lat_mid==lat0; the midpoint-vs-start distinction is exercised by
        # test_rotated_frame.TestRk2LongitudeIsSecondOrder.)
        dt = 3600.0
        nn0 = int(find_nearest_nodes(mesh, np.array([lon0]),
                                     np.array([lat0]))[0])
        vf = _node_linear_field(mesh)               # same field as in ds
        u1, v1, ok1 = interp_barycentric_single(lon0, lat0, 0, mesh, vf, nn0)
        assert ok1
        z0_a = np.cos(lat0 * DEG2RAD) * METERS_PER_DEG_LAT    # == _z0(lat0)
        lon_mid = lon0 + 0.5 * dt * u1 / z0_a
        lat_mid = lat0 + 0.5 * dt * v1 / METERS_PER_DEG_LAT
        nn_mid = int(find_nearest_nodes(mesh, np.array([lon_mid]),
                                        np.array([lat_mid]))[0])
        u2, v2, ok2 = interp_barycentric_single(lon_mid, lat_mid, 0,
                                                mesh, vf, nn_mid)
        assert ok2
        z0_b = np.cos(lat_mid * DEG2RAD) * METERS_PER_DEG_LAT  # MIDPOINT lat (RK2)
        lon_expect = lon0 + dt * u2 / z0_b
        lat_expect = lat0 + dt * v2 / METERS_PER_DEG_LAT

        np.testing.assert_allclose(rec.lons[-1][0], lon_expect, rtol=1e-10)
        np.testing.assert_allclose(rec.lats[-1][0], lat_expect, rtol=1e-10)

    def test_outside_particle_gets_outside_status(self, tmp_path):
        # near the +lon edge with strong u: exits the mesh within 3 steps.
        # TrajectoryRecord does not snapshot failure_reason (only status),
        # so the OUTSIDE status is the observable signal the run records.
        reader, mesh, ds = _open(tmp_path, "nele", u_val=30.0)
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False)
        numerics = NumericsConfig(dt=3600.0, integrator="euler",
                                  output_freq=1, uv_interp="barycentric")
        init = make_state(np.array([5.9]), np.array([5.0]))
        rec = run_single_realization(mesh, ds, reader, init, mode, numerics,
                                     start_time_idx=0, n_steps=3)
        ds.close()
        assert rec.statuses[-1][0] == ParticleStatus.OUTSIDE

    def test_track_particles_accepts_uv_interp(self, tmp_path):
        from kelp.tracking import track_particles
        reader, mesh, ds = _open(tmp_path, "node", u_val=1.0)
        rec = track_particles(mesh, ds, reader,
                              np.array([5.0]), np.array([5.0]),
                              dt=3600.0, n_steps=2, integrator="euler",
                              uv_interp="barycentric")
        ds.close()
        assert rec.lons[-1][0] > 5.0


class TestRunTrackingFailFastAndMetadata:
    """run_tracking pre-writer probe + writer metadata.

    The project's config is AppConfig dataclasses (there is no raw-yaml
    'particles' section); particle positions ride a .dat file via
    ``data.inicloud``.  Mirrors tests/test_tracking_diffusion.py.
    """

    @staticmethod
    def _write_particles(path, n=1, lon=5.0, lat=5.0):
        path.write_text("\n".join(f"{lon} {lat}" for _ in range(n)) + "\n")
        return path

    def _config(self, nc_path, particles_path, out_path, uv_interp="leastsq"):
        return AppConfig(
            mode=ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0),
            numerics=NumericsConfig(dt=3600.0, integrator="euler",
                                    output_freq=1, uv_interp=uv_interp),
            mesh=MeshConfig(type="fesomc", meshpath=str(nc_path)),
            data=DataConfig(ncinput=str(nc_path),
                            inicloud=str(particles_path),
                            output_path=str(out_path),
                            time_averaging="instant"),
            time=TimeConfig(start_time_idx=0, n_steps=2),
        )

    def test_node_file_fails_before_writer_creation(self, tmp_path):
        # default-named node velocities + default leastsq: the metadata
        # probe must raise BEFORE any output file is created
        from kelp.tracking import run_tracking
        nc = _make_fesomc_const(tmp_path, uv_dim="node")
        parts = self._write_particles(tmp_path / "p.dat")
        out = tmp_path / "out.nc"
        cfg = self._config(nc, parts, out, uv_interp="leastsq")
        with pytest.raises(ValueError, match="barycentric"):
            run_tracking(cfg, output_path=str(out))
        assert not out.exists()           # writer never created

    def test_metadata_only_when_nondefault(self, tmp_path):
        from kelp.tracking import run_tracking
        nc = _make_fesomc_const(tmp_path, uv_dim="nele")
        parts = self._write_particles(tmp_path / "p.dat")
        # default run: no uv_interp attr
        out_d = tmp_path / "out_default.nc"
        run_tracking(self._config(nc, parts, out_d, uv_interp="leastsq"),
                     output_path=str(out_d))
        with netCDF4.Dataset(out_d) as f:
            assert "uv_interp" not in f.ncattrs()
        # barycentric run: attr present
        out_b = tmp_path / "out_bary.nc"
        run_tracking(self._config(nc, parts, out_b, uv_interp="barycentric"),
                     output_path=str(out_b))
        with netCDF4.Dataset(out_b) as f:
            assert f.getncattr("uv_interp") == "barycentric"


# ---------------------------------------------------------------------------
# Direct euler_step guards (no tracking layer) — these paths are reachable
# only by callers that bypass run_single_realization's step-0 backstop.
# ---------------------------------------------------------------------------

def _grid_mesh_direct():
    """3x3-node, 8-tri mesh around (5,5) WITH search structures.

    The interior node at (5,5) is surrounded by elements so a particle
    seeded there has a containing element and a valid LSQ stencil.
    """
    lonn = np.array([4, 5, 6, 4, 5, 6, 4, 5, 6], dtype=np.float64)
    latn = np.array([4, 4, 4, 5, 5, 5, 6, 6, 6], dtype=np.float64)
    en = np.array(
        [[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4],
         [3, 4, 7], [3, 7, 6], [4, 5, 8], [4, 8, 7]],
        dtype=np.int32,
    )
    cl = np.array([lonn[e].mean() for e in en], dtype=np.float64)
    ca = np.array([latn[e].mean() for e in en], dtype=np.float64)
    n2e, n2e_cnt = build_node_to_elems(lonn.size, en)
    m = Mesh(name="direct", nodes_lon=lonn, nodes_lat=latn,
             depth=np.full(lonn.size, 10.0),
             elem_nodes=en, elem_center_lon=cl, elem_center_lat=ca,
             node_to_elems=n2e, node_to_elems_count=n2e_cnt,
             sigma=np.empty(0), vertical_coord="zlevel")
    build_search(m)
    return m


def _em_combined_args():
    return DiffusionStepArgs(
        scheme="em", kh_x=0.0, kh_y=0.0,
        seed=7, real_id=0, step_i=0,
        composition="combined", dres_m=1000.0,
    )


def _vf_element_kh(mesh, kh=2.0, u=0.1, v=0.0):
    """Element-layout u/v carrying node-based Kh (em/m1 path requires it)."""
    ne = mesh.n_elements
    nn = mesh.n_nodes
    return VelocityField(
        u=np.full(ne, u, dtype=np.float64),
        v=np.full(ne, v, dtype=np.float64),
        eta=np.zeros(nn, dtype=np.float64),
        kh_zonal=np.full(nn, kh, dtype=np.float64),
        kh_meridional=np.full(nn, kh, dtype=np.float64),
        kh_layout="node",
    )


def _vf_node_kh(mesh, kh=2.0, u=0.1, v=0.0):
    """Node-layout u/v (uv_layout='node') carrying node-based Kh."""
    nn = mesh.n_nodes
    return VelocityField(
        u=np.full(nn, u, dtype=np.float64),
        v=np.full(nn, v, dtype=np.float64),
        eta=np.zeros(nn, dtype=np.float64),
        kh_zonal=np.full(nn, kh, dtype=np.float64),
        kh_meridional=np.full(nn, kh, dtype=np.float64),
        kh_layout="node",
        uv_layout="node",
    )


class TestCombinedDiffusionGuards:
    """combined em/m1 (replaces the Euler step) must reject barycentric u/v
    interpolation and node-layout velocities — the drift kernel is the
    element-centre leastsq core (BLOCKER, both reviewers; spec 2026-06-06).
    """

    def test_combined_em_rejects_barycentric_method(self):
        mesh = _grid_mesh_direct()
        vel = _vf_element_kh(mesh)
        state = make_state(np.array([5.0]), np.array([5.0]))
        with pytest.raises(ValueError, match="barycentric|uv_interp"):
            euler_step(state, vel, mesh, dt=300.0, time_dir=1.0,
                       diffusion=_em_combined_args(),
                       uv_interp="barycentric")

    def test_combined_em_rejects_node_layout_velocities(self):
        # uv_interp="leastsq" so the METHOD guard passes and the LAYOUT
        # guard fires — this is exactly the dangerous silent case (node
        # arrays indexed by element id) that the tracking step-0 guard
        # would catch but a direct euler_step caller bypasses.
        mesh = _grid_mesh_direct()
        vel = _vf_node_kh(mesh)
        state = make_state(np.array([5.0]), np.array([5.0]))
        with pytest.raises(ValueError, match="node-based|node-layout"):
            euler_step(state, vel, mesh, dt=300.0, time_dir=1.0,
                       diffusion=_em_combined_args(),
                       uv_interp="leastsq")


class TestBarycentricFailureReasonDirect:
    """Spec Testing item 9: a barycentric euler_step on a particle with no
    containing element must mark it OUTSIDE with INTERPOLATION_FAILED.
    """

    def test_outside_particle_records_interpolation_failed(self):
        mesh = _grid_mesh_direct()
        ne = mesh.n_elements
        vel = VelocityField(
            u=np.full(ne, 0.1, dtype=np.float64),
            v=np.zeros(ne, dtype=np.float64),
            eta=np.zeros(mesh.n_nodes, dtype=np.float64),
        )
        # A point far outside the 4..6 deg mesh extent: nearest_node is
        # valid but no element contains it, so barycentric interp fails.
        # Pre-allocate failure_reason (int8 zeros) to opt into tracking —
        # the unfused path only stamps codes when the field is present.
        nn = find_nearest_nodes(mesh, np.array([50.0]), np.array([50.0]))
        state = ParticleState(
            lon=np.array([50.0], dtype=np.float64),
            lat=np.array([50.0], dtype=np.float64),
            z=np.zeros(1, dtype=np.float64),
            status=np.full(1, ParticleStatus.ALIVE, dtype=np.int32),
            nearest_node=nn.astype(np.int32),
            failure_reason=np.zeros(1, dtype=np.int8),
        )
        out = euler_step(state, vel, mesh, dt=300.0, time_dir=1.0,
                         uv_interp="barycentric")
        assert out.status[0] == ParticleStatus.OUTSIDE
        assert out.failure_reason is not None
        assert out.failure_reason[0] == FailureReason.INTERPOLATION_FAILED
