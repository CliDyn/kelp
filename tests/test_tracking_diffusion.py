"""End-to-end tracking tests with split-mode diffusion enabled.

Covers ``run_tracking`` propagation of ``DiffusionConfig``
into the integrator, auto-derived ``realization_id``, NetCDF writer
metadata, ``track_particles`` functional API, and CLI ``--seed``
override.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

import netCDF4

from kelp.cli.main import main as cli_main
from kelp.core.config import (
    AppConfig, DataConfig, DiffusionConfig, MeshConfig, ModeConfig,
    NumericsConfig, TimeConfig,
)
from kelp.core.particles import make_state
from kelp.readers import get_reader
from kelp.tracking import run_single_realization, run_tracking, track_particles


# ---------------------------------------------------------------------------
# Synthetic FESOM-C dataset on an 11×11 grid centred at (5°, 5°).
#
# The wider mesh (10° × 10°) keeps random walks inside the domain over a
# handful of steps even with Kh of order 1 m^2/s and dt of order 100 s.
# ---------------------------------------------------------------------------

def _make_fesomc_wide(tmp_path: Path, n_time: int = 4,
                      u_val: float = 0.0, v_val: float = 0.0):
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
            elems.append([n0 + 1, n1 + 1, n2 + 1, n0 + 1])  # 1-based padded
            elems.append([n1 + 1, n3 + 1, n2 + 1, n1 + 1])
    nv = np.array(elems, dtype=np.int32)
    n_elem = nv.shape[0]

    elem_nodes_0 = nv[:, :3] - 1
    lon_c = np.array([lon[e].mean() for e in elem_nodes_0], dtype=np.float64)
    lat_c = np.array([lat[e].mean() for e in elem_nodes_0], dtype=np.float64)

    sigma_lev = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    u2d = np.full((n_time, n_elem), u_val, dtype=np.float64)
    v2d = np.full((n_time, n_elem), v_val, dtype=np.float64)
    eta = np.zeros((n_time, n_nodes), dtype=np.float64)

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
    })
    nc_path = tmp_path / "fesomc_diff.nc"
    ds.to_netcdf(nc_path)

    mesh_cfg = SimpleNamespace(meshpath=str(nc_path))
    return nc_path, mesh_cfg


def _build_app_config(nc_path: Path, particles_path: Path, output_path: Path,
                      *, seed: int = 0, kh: float = 0.5, num_runs: int = 1,
                      scheme: str = "uniform", composition: str = "split",
                      var_kh_zonal=None, var_kh_meridional=None,
                      dres_m=None):
    """Compose an AppConfig that points at the synthetic dataset."""
    return AppConfig(
        mode=ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0),
        numerics=NumericsConfig(
            dt=100.0, integrator="euler", output_freq=1,
            diffusion=DiffusionConfig(
                enabled=True, scheme=scheme,
                kh_zonal=kh, kh_meridional=kh, seed=seed,
                composition=composition,
                var_kh_zonal=var_kh_zonal,
                var_kh_meridional=var_kh_meridional,
                dres_m=dres_m,
            ),
        ),
        mesh=MeshConfig(type="fesomc", meshpath=str(nc_path)),
        data=DataConfig(
            ncinput=str(nc_path),
            inicloud=str(particles_path),
            output_path=str(output_path),
            time_averaging="instant",
        ),
        time=TimeConfig(start_time_idx=0, n_steps=3, num_runs=num_runs),
    )


def _make_fesomc_wide_with_kh(tmp_path: Path, n_time: int = 4,
                               u_val: float = 0.0, v_val: float = 0.0,
                               kh_const: float = 1.0):
    """Same as ``_make_fesomc_wide`` but adds node-based Kh variables.

    Used by the end-to-end tests to verify the run_tracking path
    propagates the diffusion config through readers + integrator on a
    real synthetic dataset.
    """
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

    u2d = np.full((n_time, n_elem), u_val, dtype=np.float64)
    v2d = np.full((n_time, n_elem), v_val, dtype=np.float64)
    eta = np.zeros((n_time, n_nodes), dtype=np.float64)
    kh_zonal = np.full(n_nodes, kh_const, dtype=np.float64)
    kh_meridional = np.full(n_nodes, kh_const, dtype=np.float64)

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
        "Kh_zonal": (["node"], kh_zonal),
        "Kh_meridional": (["node"], kh_meridional),
    })
    nc_path = tmp_path / "fesomc_diff_kh.nc"
    ds.to_netcdf(nc_path)
    return nc_path


def _write_particles(path: Path, n: int = 8, lon: float = 5.0,
                     lat: float = 5.0) -> Path:
    lines = "\n".join(f"{lon} {lat}" for _ in range(n)) + "\n"
    path.write_text(lines)
    return path


# ---------------------------------------------------------------------------
# Reproducibility: same seed → bitwise-identical trajectory
# ---------------------------------------------------------------------------

class TestSeedReproducibility:
    def test_run_tracking_same_seed_reproduces(self, tmp_path):
        nc_path, _mesh_cfg = _make_fesomc_wide(tmp_path)
        particles = _write_particles(tmp_path / "p.dat")

        out_a = tmp_path / "out_a.nc"
        out_b = tmp_path / "out_b.nc"
        cfg_a = _build_app_config(nc_path, particles, out_a, seed=42, kh=0.5)
        cfg_b = _build_app_config(nc_path, particles, out_b, seed=42, kh=0.5)
        run_tracking(cfg_a, verbose=False)
        run_tracking(cfg_b, verbose=False)

        with netCDF4.Dataset(out_a) as ds_a, netCDF4.Dataset(out_b) as ds_b:
            np.testing.assert_array_equal(ds_a["lon"][:], ds_b["lon"][:])
            np.testing.assert_array_equal(ds_a["lat"][:], ds_b["lat"][:])

    def test_different_seeds_diverge(self, tmp_path):
        nc_path, _ = _make_fesomc_wide(tmp_path)
        particles = _write_particles(tmp_path / "p.dat", n=20)

        out_a = tmp_path / "out_a.nc"
        out_b = tmp_path / "out_b.nc"
        cfg_a = _build_app_config(nc_path, particles, out_a, seed=1, kh=0.5)
        cfg_b = _build_app_config(nc_path, particles, out_b, seed=2, kh=0.5)
        run_tracking(cfg_a, verbose=False)
        run_tracking(cfg_b, verbose=False)

        with netCDF4.Dataset(out_a) as ds_a, netCDF4.Dataset(out_b) as ds_b:
            assert not np.array_equal(ds_a["lon"][:], ds_b["lon"][:])
            assert not np.array_equal(ds_a["lat"][:], ds_b["lat"][:])


class TestMultiRealization:
    """Different ``realization_id`` ⇒ uncorrelated trajectories within a
    single ``run_tracking`` call, even when ``seed`` is the same."""

    def test_runs_diverge_with_same_seed(self, tmp_path):
        nc_path, _ = _make_fesomc_wide(tmp_path)
        particles = _write_particles(tmp_path / "p.dat", n=10)
        out = tmp_path / "multi.nc"
        cfg = _build_app_config(nc_path, particles, out,
                                seed=0, kh=0.3, num_runs=2)
        run_tracking(cfg, verbose=False)

        # run_tracking writes one file per realization with suffix _rN
        f0 = tmp_path / "multi_r0.nc"
        f1 = tmp_path / "multi_r1.nc"
        assert f0.exists() and f1.exists()
        with netCDF4.Dataset(f0) as ds0, netCDF4.Dataset(f1) as ds1:
            assert not np.array_equal(ds0["lon"][:], ds1["lon"][:])


# ---------------------------------------------------------------------------
# Writer metadata exposes diffusion fields when enabled.
# ---------------------------------------------------------------------------

class TestWriterMetadata:
    def test_diffusion_metadata_present(self, tmp_path):
        nc_path, _ = _make_fesomc_wide(tmp_path)
        particles = _write_particles(tmp_path / "p.dat")
        out = tmp_path / "meta.nc"
        cfg = _build_app_config(nc_path, particles, out, seed=7, kh=0.5)
        run_tracking(cfg, verbose=False)

        with netCDF4.Dataset(out) as ds:
            attrs = {k: ds.getncattr(k) for k in ds.ncattrs()}
            assert int(attrs.get("diffusion_enabled", 0)) == 1
            assert attrs["diffusion_scheme"] == "uniform"
            assert float(attrs["diffusion_kh_zonal"]) == 0.5
            assert int(attrs["diffusion_seed"]) == 7
            assert int(attrs["diffusion_realization_id"]) == 0

    def test_metadata_absent_when_diffusion_disabled(self, tmp_path):
        nc_path, _ = _make_fesomc_wide(tmp_path, u_val=0.5)
        particles = _write_particles(tmp_path / "p.dat")
        out = tmp_path / "meta_off.nc"
        cfg = _build_app_config(nc_path, particles, out)
        cfg.numerics.diffusion = DiffusionConfig(enabled=False)
        run_tracking(cfg, verbose=False)

        with netCDF4.Dataset(out) as ds:
            attrs = {k: ds.getncattr(k) for k in ds.ncattrs()}
            assert "diffusion_enabled" not in attrs
            assert "diffusion_scheme" not in attrs


# ---------------------------------------------------------------------------
# Functional API: track_particles with explicit DiffusionConfig
# ---------------------------------------------------------------------------

class TestTrackParticlesAPI:
    def test_diffusion_passed_through(self, tmp_path):
        nc_path, _ = _make_fesomc_wide(tmp_path)
        reader = get_reader("fesomc")
        mesh_cfg = SimpleNamespace(meshpath=str(nc_path))
        mesh = reader.load_mesh(mesh_cfg)
        ds = reader.open_dataset(str(nc_path))

        diff = DiffusionConfig(enabled=True, scheme="uniform",
                               kh_zonal=0.4, kh_meridional=0.4, seed=11)
        rec_a = track_particles(
            mesh, ds, reader,
            lon0=np.full(8, 5.0), lat0=np.full(8, 5.0),
            n_steps=3, dt=100.0, time_dir=1.0,
            diffusion=diff, realization_id=0,
        )
        # Same input but different realization_id → different trajectory.
        rec_b = track_particles(
            mesh, ds, reader,
            lon0=np.full(8, 5.0), lat0=np.full(8, 5.0),
            n_steps=3, dt=100.0, time_dir=1.0,
            diffusion=diff, realization_id=1,
        )
        ds.close()

        a = rec_a.to_arrays()["lon"]
        b = rec_b.to_arrays()["lon"]
        assert a.shape == b.shape
        assert not np.array_equal(a, b)

    def test_disabled_diffusion_matches_baseline(self, tmp_path):
        """Passing a default-disabled DiffusionConfig must reproduce the
        pre-A5 behaviour bit-for-bit."""
        nc_path, _ = _make_fesomc_wide(tmp_path, u_val=0.4)
        reader = get_reader("fesomc")
        mesh_cfg = SimpleNamespace(meshpath=str(nc_path))
        mesh = reader.load_mesh(mesh_cfg)
        ds = reader.open_dataset(str(nc_path))

        rec_baseline = track_particles(
            mesh, ds, reader,
            lon0=np.full(4, 5.0), lat0=np.full(4, 5.0),
            n_steps=3, dt=100.0, time_dir=1.0,
        )
        rec_off = track_particles(
            mesh, ds, reader,
            lon0=np.full(4, 5.0), lat0=np.full(4, 5.0),
            n_steps=3, dt=100.0, time_dir=1.0,
            diffusion=DiffusionConfig(enabled=False),
        )
        ds.close()

        np.testing.assert_array_equal(
            rec_baseline.to_arrays()["lon"],
            rec_off.to_arrays()["lon"],
        )


# ---------------------------------------------------------------------------
# CLI integration: --seed overrides config.numerics.diffusion.seed
# ---------------------------------------------------------------------------

class TestRestartBitwiseAtTrackingLayer:
    """Resume via ``start_step_i`` reproduces a continuous run bit-for-bit.

    The orchestration layer must offset the diffusion step index when
    resuming from a checkpoint, otherwise the post-resume RNG keys
    collide with steps 0..N-1 of the original run.
    """

    def test_resume_from_midpoint(self, tmp_path):
        from kelp.tracking import run_single_realization
        from kelp.readers import get_reader

        nc_path, _ = _make_fesomc_wide(tmp_path, n_time=8)
        reader = get_reader("fesomc")
        mesh_cfg = SimpleNamespace(meshpath=str(nc_path))
        mesh = reader.load_mesh(mesh_cfg)
        ds = reader.open_dataset(str(nc_path))
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)
        diff = DiffusionConfig(enabled=True, scheme="uniform",
                               kh_zonal=0.5, kh_meridional=0.5, seed=99)
        numerics = NumericsConfig(
            dt=100.0, integrator="euler", output_freq=1, diffusion=diff,
        )
        init = make_state(np.full(8, 5.0), np.full(8, 5.0))

        # Continuous: 6 steps in one go.
        rec_cont = run_single_realization(
            mesh, ds, reader, init, mode, numerics,
            start_time_idx=0, n_steps=6, n_time_total=8,
        )

        # Split: 3 steps, then 3 more starting from start_step_i=3.
        rec_a = run_single_realization(
            mesh, ds, reader, init, mode, numerics,
            start_time_idx=0, n_steps=3, n_time_total=8,
        )
        # Take the final state from rec_a (last snapshot in trajectory).
        last_arrs = rec_a.to_arrays()
        from kelp.core.particles import ParticleState
        mid_state = ParticleState(
            lon=last_arrs["lon"][-1].copy(),
            lat=last_arrs["lat"][-1].copy(),
            z=last_arrs["z"][-1].copy(),
            status=last_arrs["status"][-1].astype(np.int8).copy(),
            nearest_node=np.full(init.n_particles, -1, dtype=np.int32),
        )
        rec_b = run_single_realization(
            mesh, ds, reader, mid_state, mode, numerics,
            start_time_idx=3, n_steps=3, n_time_total=8,
            start_step_i=3,
        )
        ds.close()

        cont = rec_cont.to_arrays()
        b = rec_b.to_arrays()
        # The continuous run's last snapshot must equal the split run's
        # last snapshot bit-for-bit.
        np.testing.assert_array_equal(cont["lon"][-1], b["lon"][-1])
        np.testing.assert_array_equal(cont["lat"][-1], b["lat"][-1])

    def test_resume_without_offset_diverges(self, tmp_path):
        """Without ``start_step_i``, resume replays RNG steps 0..N-1
        and the trajectory diverges from the continuous run.  This is
        the regression the fix prevents.
        """
        from kelp.tracking import run_single_realization
        from kelp.readers import get_reader

        nc_path, _ = _make_fesomc_wide(tmp_path, n_time=8)
        reader = get_reader("fesomc")
        mesh_cfg = SimpleNamespace(meshpath=str(nc_path))
        mesh = reader.load_mesh(mesh_cfg)
        ds = reader.open_dataset(str(nc_path))
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=1.0)
        diff = DiffusionConfig(enabled=True, scheme="uniform",
                               kh_zonal=0.5, kh_meridional=0.5, seed=99)
        numerics = NumericsConfig(
            dt=100.0, integrator="euler", output_freq=1, diffusion=diff,
        )
        init = make_state(np.full(8, 5.0), np.full(8, 5.0))

        rec_cont = run_single_realization(
            mesh, ds, reader, init, mode, numerics,
            start_time_idx=0, n_steps=6, n_time_total=8,
        )
        rec_a = run_single_realization(
            mesh, ds, reader, init, mode, numerics,
            start_time_idx=0, n_steps=3, n_time_total=8,
        )
        last_arrs = rec_a.to_arrays()
        from kelp.core.particles import ParticleState
        mid_state = ParticleState(
            lon=last_arrs["lon"][-1].copy(),
            lat=last_arrs["lat"][-1].copy(),
            z=last_arrs["z"][-1].copy(),
            status=last_arrs["status"][-1].astype(np.int8).copy(),
            nearest_node=np.full(init.n_particles, -1, dtype=np.int32),
        )
        # Forget to pass start_step_i — RNG keys collide with 0..2.
        rec_naive = run_single_realization(
            mesh, ds, reader, mid_state, mode, numerics,
            start_time_idx=3, n_steps=3, n_time_total=8,
        )
        ds.close()

        cont = rec_cont.to_arrays()
        naive = rec_naive.to_arrays()
        # Without the offset, post-resume positions deviate from the
        # continuous run.
        assert not np.array_equal(cont["lon"][-1], naive["lon"][-1])


class TestCLISeedOverride:
    def _write_yaml(self, tmp_path: Path, nc_path: Path, particles: Path,
                    output: Path, seed: int) -> Path:
        tmp_path.mkdir(parents=True, exist_ok=True)
        yaml_text = f"""\
mode:
  Fdrift2D: true
  time_dir: 1.0
numerics:
  dt: 100.0
  integrator: euler
  output_freq: 1
  diffusion:
    enabled: true
    scheme: uniform
    kh_zonal: 0.5
    kh_meridional: 0.5
    seed: {seed}
mesh:
  type: fesomc
  meshpath: {nc_path}
data:
  ncinput: {nc_path}
  inicloud: {particles}
  output_path: {output}
  time_averaging: instant
time:
  start_time_idx: 0
  n_steps: 3
"""
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml_text)
        return path

    def test_negative_seed_rejected_by_cli(self, tmp_path):
        """``--seed -1`` must trigger ``DiffusionConfig`` validation
        (raised through ``dataclasses.replace``), not silently wrap
        through the ``np.uint64(...)`` cast inside the kick kernel.
        """
        nc_path, _ = _make_fesomc_wide(tmp_path)
        particles = _write_particles(tmp_path / "p.dat")
        out = tmp_path / "neg.nc"
        cfg = self._write_yaml(tmp_path / "neg", nc_path, particles, out, seed=1)
        with pytest.raises(ValueError, match="seed"):
            cli_main(["run", str(cfg), "--seed", "-1"])

    def test_seed_override_changes_output(self, tmp_path):
        nc_path, _ = _make_fesomc_wide(tmp_path)
        particles = _write_particles(tmp_path / "p.dat", n=10)
        out_default = tmp_path / "out_default.nc"
        out_override = tmp_path / "out_override.nc"

        cfg_default = self._write_yaml(tmp_path / "default", nc_path,
                                       particles, out_default, seed=1)
        cfg_override = self._write_yaml(tmp_path / "override", nc_path,
                                        particles, out_override, seed=1)

        # Run twice: once using config seed=1, once with --seed 7 overriding.
        rc1 = cli_main(["run", str(cfg_default)])
        assert rc1 == 0
        rc2 = cli_main(["run", str(cfg_override), "--seed", "7"])
        assert rc2 == 0

        with netCDF4.Dataset(out_default) as a, \
             netCDF4.Dataset(out_override) as b:
            assert not np.array_equal(a["lon"][:], b["lon"][:])
            # And the override's seed appears in the metadata.
            assert int(b.getncattr("diffusion_seed")) == 7


# ---------------------------------------------------------------------------
# Variable-Kh end-to-end through run_tracking
# ---------------------------------------------------------------------------


class TestPhaseBEndToEnd:
    """``run_tracking`` with em/m1 + Euler on a dataset that exposes Kh."""

    def test_em_combined_runs(self, tmp_path):
        """A full run with scheme='em' + Euler should complete without error.

        Also pins review finding #6: writer metadata records the
        *resolved* composition, not the user-set default.
        """
        nc_path = _make_fesomc_wide_with_kh(tmp_path, kh_const=2.0)
        particles = _write_particles(tmp_path / "p.dat", n=12)
        out_path = tmp_path / "traj.nc"

        cfg = _build_app_config(
            nc_path, particles, out_path, seed=42,
            scheme="em", composition="split",  # auto-resolves to "combined"
            dres_m=2000.0, kh=0.0,  # kh_zonal/_meridional ignored for em
        )
        run_tracking(cfg, verbose=False)
        # File written → metadata reflects 'em' AND the auto-resolved
        # composition 'combined' (not the user's "split" default).
        with netCDF4.Dataset(out_path) as ds:
            assert ds.getncattr("diffusion_scheme") == "em"
            assert ds.getncattr("diffusion_composition") == "combined"

    def test_m1_combined_runs(self, tmp_path):
        nc_path = _make_fesomc_wide_with_kh(tmp_path, kh_const=2.0)
        particles = _write_particles(tmp_path / "p.dat", n=12)
        out_path = tmp_path / "traj.nc"

        cfg = _build_app_config(
            nc_path, particles, out_path, seed=42,
            scheme="m1", dres_m=2000.0,
        )
        run_tracking(cfg, verbose=False)
        with netCDF4.Dataset(out_path) as ds:
            assert ds.getncattr("diffusion_scheme") == "m1"

    def test_em_on_rk2_raises(self, tmp_path):
        """em + RK2 must fail at the orchestration layer, naming the opt-in."""
        nc_path = _make_fesomc_wide_with_kh(tmp_path)
        particles = _write_particles(tmp_path / "p.dat", n=4)
        out_path = tmp_path / "traj.nc"

        cfg = _build_app_config(
            nc_path, particles, out_path, scheme="em", dres_m=1000.0,
        )
        # Override integrator to rk2 (the helper sets Euler).
        cfg.numerics.integrator = "rk2"
        with pytest.raises(NotImplementedError, match="split_rk"):
            run_tracking(cfg, verbose=False)

    def test_em_without_kh_in_dataset_raises(self, tmp_path):
        """em scheme + dataset without Kh variables → clear runtime error."""
        # Use the original helper (no Kh) so the reader returns kh=None.
        nc_path, _ = _make_fesomc_wide(tmp_path)
        particles = _write_particles(tmp_path / "p.dat", n=4)
        out_path = tmp_path / "traj.nc"

        cfg = _build_app_config(
            nc_path, particles, out_path, scheme="em", dres_m=1000.0,
        )
        with pytest.raises(ValueError, match="kh_zonal"):
            run_tracking(cfg, verbose=False)

    def test_em_split_rk_on_rk4_runs(self, tmp_path):
        """Full run with em + RK4 + composition='split_rk'."""
        nc_path = _make_fesomc_wide_with_kh(tmp_path, kh_const=2.0)
        particles = _write_particles(tmp_path / "p.dat", n=12)
        out_path = tmp_path / "traj.nc"

        cfg = _build_app_config(
            nc_path, particles, out_path, seed=42,
            scheme="em", composition="split_rk",
            dres_m=2000.0,
        )
        cfg.numerics.integrator = "rk4"
        # Suppress the once-per-realization split-RK warning.
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore", RuntimeWarning)
            run_tracking(cfg, verbose=False)
        with netCDF4.Dataset(out_path) as ds:
            assert ds.getncattr("diffusion_scheme") == "em"
            assert ds.getncattr("diffusion_composition") == "split_rk"

    def test_em_split_rk_default_composition_on_rk_raises(self, tmp_path):
        """em + RK with default composition='split' → orchestration raises."""
        nc_path = _make_fesomc_wide_with_kh(tmp_path, kh_const=1.0)
        particles = _write_particles(tmp_path / "p.dat", n=4)
        out_path = tmp_path / "traj.nc"
        cfg = _build_app_config(
            nc_path, particles, out_path, scheme="em", dres_m=1000.0,
        )
        cfg.numerics.integrator = "rk2"
        # composition is the default "split" — must error pointing at split_rk.
        with pytest.raises(NotImplementedError, match="split_rk"):
            run_tracking(cfg, verbose=False)

    def test_dres_m_default_from_mesh(self, tmp_path):
        """Without explicit dres_m, tracking layer derives it from the mesh.

        Run two configs — one with explicit dres_m, one without — and
        check that both complete successfully (the auto-derived value
        is never zero, otherwise the integrator would raise).
        """
        nc_path = _make_fesomc_wide_with_kh(tmp_path, kh_const=1.0)
        particles = _write_particles(tmp_path / "p.dat", n=4)

        out_a = tmp_path / "ta.nc"
        cfg_a = _build_app_config(
            nc_path, particles, out_a, scheme="em", dres_m=1500.0,
        )
        run_tracking(cfg_a, verbose=False)

        out_b = tmp_path / "tb.nc"
        cfg_b = _build_app_config(
            nc_path, particles, out_b, scheme="em", dres_m=None,
        )
        run_tracking(cfg_b, verbose=False)
        # Both runs produced output files.
        assert out_a.exists()
        assert out_b.exists()
