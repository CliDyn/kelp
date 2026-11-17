"""Data-gated (``KELP_TEST_NEMO_DATA``): partial-cell W against the model's W on
the ORCA05 equatorial crop, and z* detection on the eORCA025 crop."""
import os
from pathlib import Path

import numpy as np
import pytest

from kelp.core.config import MeshConfig, ModeConfig
from kelp.readers import get_reader

_d = os.environ.get("KELP_TEST_NEMO_DATA")
pytestmark = pytest.mark.skipif(not _d, reason="KELP_TEST_NEMO_DATA not set")


def _read(mesh_file, grid_t, var_u, var_v):
    mode = ModeConfig(Fdrift3D=True, var_u=var_u, var_v=var_v)
    rd = get_reader("nemo")
    mesh = rd.load_mesh(MeshConfig(type="nemo", meshpath=str(mesh_file), nemo_grid="cgrid"))
    ds = rd.open_dataset(str(grid_t), mode_cfg=mode)
    rd.prepare(mesh, ds)
    return mesh, ds, rd.read_velocities(ds, 0, mesh, mode_cfg=mode)


def test_orca05_partial_cell_w_equals_model_w():
    d = Path(_d)
    mesh, ds, vel = _read(d / "orca05_equatorial_mesh.nc", d / "orca05_equatorial_grid_T.nc",
                          "vozocrtx", "vomecrty")
    ny, nx = mesh._structured_shape
    wf = -np.nan_to_num(ds["w_file"].isel(time=0).values.astype(float))   # positive down
    K = wf.shape[0]
    w = vel.w.reshape(K + 1, ny, nx)[:-1].astype(float)
    wet = np.arange(K)[:, None, None] < mesh._cgrid_nlev_raster
    err = np.abs(w - wf)[wet].max()
    assert np.abs(wf).max() > 1e-5 and err < 5e-6 * np.abs(wf).max()


def test_eorca025_zstar_warns_and_runs():
    d = Path(_d)
    with pytest.warns(RuntimeWarning, match="z\\*/vvl"):
        mesh, ds, vel = _read(d / "eorca025_equatorial_domain_cfg.nc",
                              d / "eorca025_equatorial_split" / "eorca025_1m_19760101_19760229_grid_T.nc",
                              "uo", "vo")
    assert np.isfinite(vel.w).all()
