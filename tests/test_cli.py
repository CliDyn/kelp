"""Smoke tests for the ``kelp`` CLI.

Covers ``kelp info`` (mesh stats + time range) and a basic
``kelp run`` invocation with a synthetic FESOM-C dataset.  The
diffusion-specific seed-override and validation tests live in
``test_tracking_diffusion.py::TestCLISeedOverride``.
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import netCDF4
import numpy as np
import pytest
import xarray as xr

from kelp.cli.main import main as cli_main


def _make_fesomc_minimal(tmp_path: Path) -> Path:
    """Synthetic 11×11 FESOM-C dataset, no diffusion fields needed."""
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
    n_time = 3
    u2d = np.full((n_time, n_elem), 0.1, dtype=np.float64)
    v2d = np.zeros((n_time, n_elem), dtype=np.float64)
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
    nc_path = tmp_path / "fesomc.nc"
    ds.to_netcdf(nc_path)
    return nc_path


def _write_particles(path: Path, n: int = 6, lon: float = 5.0,
                     lat: float = 5.0) -> Path:
    path.write_text("\n".join(f"{lon} {lat}" for _ in range(n)) + "\n")
    return path


def _write_config(yaml_path: Path, nc_path: Path,
                  particles: Path, output: Path) -> Path:
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(f"""\
mode:
  Fdrift2D: true
  time_dir: 1.0
numerics:
  dt: 100.0
  integrator: euler
  output_freq: 1
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
""")
    return yaml_path


class TestCliVersionAndHelp:
    def test_version_prints_and_exits_clean(self):
        with pytest.raises(SystemExit) as exc_info:
            cli_main(["--version"])
        assert exc_info.value.code == 0

    def test_no_subcommand_errors(self):
        """``kelp`` with no subcommand → argparse error (exit 2)."""
        with pytest.raises(SystemExit) as exc_info:
            cli_main([])
        # argparse exits 2 on missing required subparser.
        assert exc_info.value.code == 2


class TestCliInfo:
    def test_info_smoke(self, tmp_path):
        """``kelp info config.yaml`` prints mesh stats + time range."""
        nc_path = _make_fesomc_minimal(tmp_path)
        particles = _write_particles(tmp_path / "p.dat")
        out = tmp_path / "out.nc"
        cfg = _write_config(tmp_path / "cfg.yaml", nc_path, particles, out)

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli_main(["info", str(cfg)])
        assert rc == 0
        text = buf.getvalue()
        assert "Mesh type: fesomc" in text
        assert "nodes:" in text
        assert "elements:" in text
        # Time range comes from the dataset open path — pin this so a
        # future regression on the ``ncinput`` attr name (review catch
        # this commit) gets caught.
        assert "Time range:" in text
        # the grid-hash cell the run would use, with its window statistics
        assert "grid hash: cell" in text
        assert "(reader auto)" not in text

    def test_info_imports_no_tracking_loop(self, tmp_path):
        """``kelp info`` takes what it shares with a run (the reader hook
        ``prepare``, the stamp options, the record-times line) from the
        reader registry: in a fresh interpreter it neither imports
        ``kelp.tracking`` nor the integrators."""
        nc_path = _make_fesomc_minimal(tmp_path)
        particles = _write_particles(tmp_path / "p.dat")
        cfg = _write_config(tmp_path / "cfg.yaml", nc_path, particles,
                            tmp_path / "out.nc")
        code = ("import sys\n"
                "from kelp.cli.main import main\n"
                f"rc = main(['info', {str(cfg)!r}])\n"
                "heavy = [m for m in ('kelp.tracking', "
                "'kelp.numerics.integrators') if m in sys.modules]\n"
                "print('RC', rc, 'IMPORTED', heavy)\n")
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
        assert out.returncode == 0, out.stderr
        assert "RC 0 IMPORTED []" in out.stdout, (out.stdout, out.stderr)

    def test_info_labels_an_explicit_cell(self, tmp_path):
        """Review 2026-09-22: an explicit grid_hash_cell_size was printed as
        '(reader auto)'."""
        nc_path = _make_fesomc_minimal(tmp_path)
        particles = _write_particles(tmp_path / "p.dat")
        cfg = _write_config(tmp_path / "cfg.yaml", nc_path, particles, tmp_path / "out.nc")
        cfg.write_text(cfg.read_text().replace("  type: fesomc\n",
                                               "  type: fesomc\n  grid_hash_cell_size: 0.33\n"))
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli_main(["info", str(cfg)])
        assert rc == 0
        assert "grid hash: cell 0.33 deg (explicit)" in buf.getvalue()

    def test_info_missing_config_returns_2(self, tmp_path):
        rc = cli_main(["info", str(tmp_path / "no-such-file.yaml")])
        assert rc == 2


class TestCliRun:
    def test_run_smoke_writes_output(self, tmp_path):
        """``kelp run config.yaml -o out.nc`` produces a valid NetCDF."""
        nc_path = _make_fesomc_minimal(tmp_path)
        particles = _write_particles(tmp_path / "p.dat", n=10)
        out = tmp_path / "traj.nc"
        cfg = _write_config(tmp_path / "cfg.yaml", nc_path, particles, out)

        rc = cli_main(["run", str(cfg)])
        assert rc == 0
        assert out.exists()
        with netCDF4.Dataset(out) as ds:
            # 10 particles × at least 1 obs slot.
            assert ds.dimensions["trajectory"].size == 10
            assert ds.dimensions["time"].size >= 1   # format 2 (was obs)

    def test_run_missing_config_returns_2(self, tmp_path):
        rc = cli_main(["run", str(tmp_path / "no-such-file.yaml")])
        assert rc == 2

    def test_run_output_override(self, tmp_path):
        """``-o`` flag overrides the YAML's output_path."""
        nc_path = _make_fesomc_minimal(tmp_path)
        particles = _write_particles(tmp_path / "p.dat", n=4)
        yaml_out = tmp_path / "yaml_out.nc"
        cli_out = tmp_path / "cli_out.nc"
        cfg = _write_config(tmp_path / "cfg.yaml", nc_path, particles, yaml_out)

        rc = cli_main(["run", str(cfg), "-o", str(cli_out)])
        assert rc == 0
        assert cli_out.exists()
        # The CLI override path is what gets written; the YAML default
        # path may also be created by the loader, so we don't assert on
        # its absence — only that the override path is real.

    def test_run_respects_yaml_temporal_interp(self, tmp_path):
        """review CLI-#1 regression: when ``--temporal-interp`` is *not*
        passed, the YAML's ``numerics.temporal_interp: true`` must still
        take effect.  Pre-fix the CLI's argparse default of False
        silently overrode the YAML, disabling temporal interpolation
        whenever the user ran the CLI without the flag.

        The two modes are observably different on the writer's time
        axis: discrete writes the file index (0, 1, 2, …) while
        temporal-interp writes sim_time in seconds (dt, 2·dt, …).  We
        pick a config where ``dt < file_dt`` so the values can't
        coincide.
        """
        import pandas as pd

        # Build a fixture with an explicit datetime time coord so
        # extract_time_info gives a real file_dt different from dt.
        xs = np.linspace(0.0, 10.0, 11)
        ys = np.linspace(0.0, 10.0, 11)
        gx, gy = np.meshgrid(xs, ys)
        lon = gx.ravel().astype(np.float64)
        lat = gy.ravel().astype(np.float64)
        n_nodes = lon.size
        depth = np.full(n_nodes, 10.0)
        elems = []
        for j in range(10):
            for i in range(10):
                n0 = j * 11 + i
                elems.append([n0 + 1, n0 + 2, n0 + 12, n0 + 1])
                elems.append([n0 + 2, n0 + 13, n0 + 12, n0 + 2])
        nv = np.array(elems, dtype=np.int32)
        n_elem = nv.shape[0]
        elem_nodes_0 = nv[:, :3] - 1
        lon_c = np.array([lon[e].mean() for e in elem_nodes_0])
        lat_c = np.array([lat[e].mean() for e in elem_nodes_0])
        sigma_lev = np.array([0.0, 0.5, 1.0])
        n_time = 4
        times = pd.date_range("2020-01-01", periods=n_time, freq="600s")
        u2d = np.full((n_time, n_elem), 0.5)  # 0.5 m/s east
        v2d = np.zeros((n_time, n_elem))
        eta = np.zeros((n_time, n_nodes))
        nc_path = tmp_path / "mesh_ti.nc"
        xr.Dataset({
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
        }, coords={"time": times}).to_netcdf(nc_path)

        particles = _write_particles(tmp_path / "p.dat", n=4)
        out = tmp_path / "ti.nc"
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(f"""\
mode:
  Fdrift2D: true
numerics:
  dt: 100.0
  integrator: euler
  output_freq: 1
  temporal_interp: true
mesh:
  type: fesomc
  meshpath: {nc_path}
data:
  ncinput: {nc_path}
  inicloud: {particles}
  output_path: {out}
  time_averaging: instant
time:
  start_time_idx: 0
  n_steps: 4
""")
        rc = cli_main(["run", str(cfg)])
        assert rc == 0

        # Observable distinction:
        #   discrete       → time axis = file indices [0, 1, 2, 3]
        #   temporal_interp → time axis = sim_time secs [100, 200, 300, 400]
        with netCDF4.Dataset(out) as ds:
            t_vals = list(ds["time"][:])
        # If the CLI default leaked through, we'd see [0, 1, 2, 3].
        # With YAML respected, we see sim_time values matching dt·step.
        assert t_vals[0] == pytest.approx(100.0), (
            f"time axis starts at {t_vals[0]}; expected 100.0 "
            "(temporal_interp mode).  CLI default may be overriding "
            "YAML again."
        )
        assert t_vals == pytest.approx([100.0, 200.0, 300.0, 400.0])

    def test_run_respects_yaml_checkpoint_freq(self, tmp_path):
        """review CLI-#1 regression: when ``--checkpoint-freq`` is *not*
        passed, the YAML's ``numerics.checkpoint_freq`` must take effect.
        Pre-fix the CLI's argparse default of 0 always overrode the
        YAML, disabling checkpointing.
        """
        nc_path = _make_fesomc_minimal(tmp_path)
        particles = _write_particles(tmp_path / "p.dat", n=4)
        out = tmp_path / "ck.nc"

        cfg = tmp_path / "cfg.yaml"
        cfg.write_text(f"""\
mode:
  Fdrift2D: true
  time_dir: 1.0
numerics:
  dt: 100.0
  integrator: euler
  output_freq: 1
  checkpoint_freq: 1
mesh:
  type: fesomc
  meshpath: {nc_path}
data:
  ncinput: {nc_path}
  inicloud: {particles}
  output_path: {out}
  time_averaging: instant
time:
  start_time_idx: 0
  n_steps: 3
""")
        rc = cli_main(["run", str(cfg)])
        assert rc == 0
        # With checkpoint_freq=1 every step writes a restart; the
        # writer stores it as `restart_*` arrays.
        with netCDF4.Dataset(out) as ds:
            assert "restart_lon" in ds.variables, (
                "YAML checkpoint_freq=1 should have caused the writer "
                "to record a restart; CLI default 0 was leaking through."
            )

    def test_info_mesh_load_failure_returns_nonzero(self, tmp_path):
        """review CLI-#2 round-2: ``info`` must NOT bubble tracebacks
        when the mesh path is broken — it must surface to stderr and
        return non-zero, same as the dataset and missing-config paths.
        """
        particles = _write_particles(tmp_path / "p.dat", n=4)
        out = tmp_path / "out.nc"
        cfg = tmp_path / "cfg.yaml"
        bad_mesh = tmp_path / "no-such-mesh.nc"
        cfg.write_text(f"""\
mode:
  Fdrift2D: true
numerics:
  dt: 100.0
mesh:
  type: fesomc
  meshpath: {bad_mesh}
data:
  ncinput: {bad_mesh}
  inicloud: {particles}
  output_path: {out}
  time_averaging: instant
time:
  start_time_idx: 0
  n_steps: 3
""")
        rc = cli_main(["info", str(cfg)])
        assert rc != 0, (
            "info must return non-zero on mesh-load failure, not "
            "bubble a traceback"
        )

    def test_info_dataset_open_failure_returns_nonzero(self, tmp_path):
        """review CLI-#2: ``info`` must NOT silently return 0 when the
        dataset path is unreadable — that hides broken configs from
        CI."""
        nc_path = _make_fesomc_minimal(tmp_path)
        particles = _write_particles(tmp_path / "p.dat", n=4)
        out = tmp_path / "out.nc"

        # Build a config whose mesh exists but whose ncinput is a
        # nonexistent path.  The mesh-load path succeeds, but the
        # dataset open fails.
        cfg = tmp_path / "cfg.yaml"
        bad_data = tmp_path / "no-such-data.nc"
        cfg.write_text(f"""\
mode:
  Fdrift2D: true
numerics:
  dt: 100.0
mesh:
  type: fesomc
  meshpath: {nc_path}
data:
  ncinput: {bad_data}
  inicloud: {particles}
  output_path: {out}
  time_averaging: instant
time:
  start_time_idx: 0
  n_steps: 3
""")
        rc = cli_main(["info", str(cfg)])
        assert rc != 0, (
            "info must return non-zero when the dataset can't be "
            "opened, so CI / shell scripts can detect the failure"
        )
