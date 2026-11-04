"""Tests of ``utils/reconstruct_w.py`` on synthetic FESOM2 files.

- The dimension order of u, v, the ALE w and ``nod_area`` as stored in the
  files: oriented by dimension NAME (vertical axis first), so the same data
  in either order gives the same W.
- The bottom residual the tool prints: read at each node's own bottom from
  the raw integral before the level mask.
"""
from __future__ import annotations

import importlib.util
import pathlib
import re
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from tests._fesom2_synth import write_meshdiag

N_NODE, N_ELEM = 4, 2                     # tests/_fesom2_synth.py geometry
ZBAR = [0.0, -10.0, -30.0, -60.0]         # 4 interfaces, 3 layers
N_IFC = len(ZBAR)
N_TIME = 4


def _load_cli():
    script = (pathlib.Path(__file__).resolve().parents[1]
              / "utils" / "reconstruct_w.py")
    spec = importlib.util.spec_from_file_location("reconstruct_w_cli_u", script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _times(n=N_TIME):
    return pd.date_range("2005-01-01", periods=n, freq="1D")


def _mesh(tmp_path, *, area_order=None, tag="m", **kw):
    """Mesh diag of the 4-node square on ZBAR; ``area_order`` adds a
    ``nod_area`` stored ``("nz", "nod2")`` or ``("nod2", "nz")`` (the same
    values, not symmetric, so a swapped read changes them)."""
    path = tmp_path / f"{tag}.mesh.diag.nc"
    write_meshdiag(path, "comp", zbar=ZBAR, **kw)
    if area_order is None:
        return path
    with xr.open_dataset(path) as ds:
        ds = ds.load()
    area = (1.0e8 * (1.0 + 0.1 * np.arange(N_IFC)[:, None])
            * (1.0 + 0.3 * np.arange(N_NODE)[None, :]))      # (nz, nod2)
    if area_order == ("nod2", "nz"):
        area = area.T
    ds["nod_area"] = (area_order, area)
    out = tmp_path / f"{tag}_area.mesh.diag.nc"
    ds.to_netcdf(out)
    return out


def _uv_ssh(tmp_path, order, *, tag="uv"):
    """u/v on ``(time, nz1, elem)`` or ``(time, elem, nz1)`` — the same
    values — and a varying ssh on ``(time, nod2)``."""
    rng = np.random.default_rng(0)
    u = 0.1 + 0.05 * rng.standard_normal((N_TIME, N_IFC - 1, N_ELEM))
    v = -0.05 + 0.05 * rng.standard_normal((N_TIME, N_IFC - 1, N_ELEM))
    dims = ("time", "nz1", "elem")
    if order == ("elem", "nz1"):
        u, v = u.transpose(0, 2, 1), v.transpose(0, 2, 1)
        dims = ("time", "elem", "nz1")
    ssh = 0.01 * rng.standard_normal((N_TIME, N_NODE))
    ds = xr.Dataset({"u": (dims, u), "v": (dims, v),
                     "ssh": (("time", "nod2"), ssh)},
                    coords={"time": _times()})
    path = tmp_path / f"{tag}.nc"
    ds.to_netcdf(path)
    return path


def _w_ale(tmp_path, order, *, tag="wale"):
    """FESOM2's ALE w (positive up) on ``(time, nz, nod2)`` or
    ``(time, nod2, nz)`` — the same values."""
    rng = np.random.default_rng(1)
    w = 1e-5 * rng.standard_normal((N_TIME, N_IFC, N_NODE))
    dims = ("time", "nz", "nod2")
    if order == ("nod2", "nz"):
        w = w.transpose(0, 2, 1)
        dims = ("time", "nod2", "nz")
    path = tmp_path / f"{tag}.nc"
    xr.Dataset({"w": (dims, w)}, coords={"time": _times()}).to_netcdf(path)
    return path


def _run(cli, tmp_path, mesh, uv, *, method, use_nod_area, w_ale=None,
         name="w.nc"):
    cfg = cli.ReconstructConfig(
        source=cli.SourceConfig(meshpath=str(mesh), u_path=str(uv),
                                v_path=str(uv), ssh_path=str(uv),
                                w_ale_path=None if w_ale is None else str(w_ale)),
        output=cli.OutputConfig(dir=str(tmp_path), format="netcdf",
                                filename=name),
        method=method, use_nod_area=use_nod_area,
    )
    cli.run(cfg)
    with xr.open_dataset(tmp_path / name) as ds:
        return ds["w"].values.copy()


class TestDimensionOrder:
    """u/v, the ALE w and nod_area were taken with ``.values`` as stored:
    u/v on ``(time, elem, nz1)`` went in as ``(elem, nz1)`` and a
    ``(nod2, nz)`` nod_area or ALE w was read transposed — a wrong W with
    no error."""

    @pytest.mark.parametrize("which", ["uv", "nod_area", "w_ale"])
    def test_same_data_in_either_order_gives_the_same_w(self, tmp_path, which):
        cli = _load_cli()
        normal = {"uv": ("nz1", "elem"), "nod_area": ("nz", "nod2"),
                  "w_ale": ("nz", "nod2")}
        swapped = {"uv": ("elem", "nz1"), "nod_area": ("nod2", "nz"),
                   "w_ale": ("nod2", "nz")}
        bottoms = [-60.0, -45.0, -60.0, -30.0]
        out = {}
        for tag, orders in (("a", normal), ("b", dict(normal, **{which: swapped[which]}))):
            sub = tmp_path / tag
            sub.mkdir()
            mesh = _mesh(sub, zbar_n_bottom=bottoms,
                         area_order=orders["nod_area"] if which == "nod_area" else None)
            uv = _uv_ssh(sub, orders["uv"])
            w_ale = _w_ale(sub, orders["w_ale"]) if which == "w_ale" else None
            out[tag] = _run(cli, sub, mesh, uv,
                            method="ale" if which == "w_ale" else "divergence",
                            use_nod_area=which == "nod_area", w_ale=w_ale)
        assert out["a"].shape == (N_TIME, N_IFC, N_NODE)
        assert np.abs(out["a"]).max() > 0
        np.testing.assert_allclose(out["b"], out["a"], rtol=1e-12, atol=1e-20)


class TestBottomResidual:
    """The printed residual was ``max |w_raw[-1, :]|`` after the
    ``nlevels_nod2D`` mask, which zeroes every interface below the node's
    last level, so it printed 0 whatever the residual."""

    def test_helper_reads_each_node_at_its_own_bottom(self):
        cli = _load_cli()
        z = np.array([0.0, 10.0, 20.0])
        w_raw = np.array([[0.0, 0.0, 0.0, 0.0, 0.0],
                          [1.0, 2.0, 3.0, 4.0, 5.0],
                          [3.0, 4.0, 9.0, 8.0, 7.0]])
        depth = np.array([15.0, 20.0, 50.0, 0.0, 10.0])
        # node 0: 2.0 at 15 m; node 1: 4.0 at 20 m; node 4: 5.0 at 10 m;
        # node 2 below the cut and node 3 without a bottom are not counted
        assert cli._bottom_residual(w_raw, z, depth) == pytest.approx(5.0)
        assert cli._bottom_residual(-w_raw, z, depth) == pytest.approx(5.0)
        assert cli._bottom_residual(w_raw, z, np.array([50.0, 0.0])) == 0.0

    def test_printed_residual_survives_the_level_mask(self, tmp_path, capsys):
        cli = _load_cli()
        # bottom at 45 m, between the interfaces at 30 and 60 m: the
        # residual reads both, and nlevels_nod2D = 3 masks the one at 60 m,
        # so the masked integral would give half of it (5e-7)
        mesh = _mesh(tmp_path, zbar_n_bottom=[-45.0] * N_NODE,
                     nlevels_nod2D=[3] * N_NODE)
        # u = v = 0 and ssh rising 0.0864 m a day: w_raw = -1e-6 m/s in
        # every column, so the residual at the bottom is 1e-6
        ds = xr.Dataset(
            {"u": (("time", "nz1", "elem"), np.zeros((N_TIME, N_IFC - 1, N_ELEM))),
             "v": (("time", "nz1", "elem"), np.zeros((N_TIME, N_IFC - 1, N_ELEM))),
             "ssh": (("time", "nod2"),
                     0.0864 * np.arange(N_TIME)[:, None] * np.ones((1, N_NODE)))},
            coords={"time": _times()})
        uv = tmp_path / "uv.nc"
        ds.to_netcdf(uv)
        _run(cli, tmp_path, mesh, uv, method="divergence", use_nod_area=True)
        out = capsys.readouterr().out
        steps = re.findall(r"\|w_bottom\|=(\S+)", out)
        assert len(steps) == N_TIME
        assert all(float(s) == pytest.approx(1e-6, rel=1e-6) for s in steps), out
        summary = re.search(r"Bottom residual \(before correction\):\s+max=(\S+)",
                            out)
        assert float(summary.group(1)) == pytest.approx(1e-6, rel=1e-6), out
