"""utils/extract_fesom_region.py streams a regional subset one time step at a
time.  The NetCDF branch appended every step after the first with
``to_netcdf(path, mode="a")`` — which in xarray OVERWRITES a variable of the
same name instead of extending the unlimited dimension, so the written file
kept only the LAST time step, silently (found by the output-format real runs,
2026-10-13: a 10-day crop came out as one record, day 10)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from extract_fesom_region import _LazySubset, _write_lazy_subset  # noqa: E402


def _source(n_time=5, n_nodes=7, n_lev=3):
    t = np.array("1958-01-01T23:15", dtype="datetime64[ns]") + \
        np.arange(n_time) * np.timedelta64(1, "D")
    data = (np.arange(n_time)[:, None, None] * 100.0
            + np.arange(n_lev)[None, :, None] * 10.0
            + np.arange(n_nodes)[None, None, :])
    return xr.DataArray(data, dims=("time", "nz1", "nod2"),
                        coords={"time": t}, name="unod",
                        attrs={"units": "m/s"})


@pytest.mark.parametrize("fmt", ["netcdf", "zarr"])
def test_every_time_step_is_written(tmp_path, fmt):
    src = _source()
    sidx = np.array([1, 3, 4])
    lazy = _LazySubset(role="u", source_da=src, spatial_dim="nod2",
                       spatial_idx=sidx, time_dim="time",
                       out_attrs=dict(src.attrs), out_name="unod")
    path = tmp_path / ("u.region.nc" if fmt == "netcdf" else "u.region.zarr")
    _write_lazy_subset(lazy, path, fmt, {"title": "t"}, progress=False)
    out = (xr.open_dataset(path) if fmt == "netcdf"
           else xr.open_zarr(path, consolidated=False))
    assert out.sizes["time"] == 5
    np.testing.assert_array_equal(out["time"].values, src["time"].values)
    np.testing.assert_array_equal(out["unod"].values,
                                  src.values[:, :, sidx])


def test_sub_daily_steps_keep_their_times(tmp_path):
    """The first write fixes the time encoding (float seconds), so later
    6-hourly steps are not truncated to whole days of an integer axis."""
    src = _source(n_time=4)
    src = src.assign_coords(time=src["time"].values[0]
                            + np.arange(4) * np.timedelta64(6, "h"))
    lazy = _LazySubset(role="u", source_da=src, spatial_dim="nod2",
                       spatial_idx=np.array([0, 2]), time_dim="time",
                       out_attrs={}, out_name="unod")
    path = tmp_path / "u6h.region.nc"
    _write_lazy_subset(lazy, path, "netcdf", {}, progress=False)
    out = xr.open_dataset(path)
    np.testing.assert_array_equal(out["time"].values, src["time"].values)


# --- review 2026-10-13: combined layout, packed sources, Zarr sub-daily

def _lazy(da, sidx, name=None):
    return _LazySubset(role=name or da.name, source_da=da, spatial_dim="nod2",
                       spatial_idx=np.asarray(sidx), time_dim="time",
                       out_attrs=dict(da.attrs), out_name=name or da.name)


@pytest.mark.parametrize("fmt", ["netcdf", "zarr"])
def test_combined_layout_keeps_every_step(tmp_path, fmt):
    from extract_fesom_region import (write_outputs, SubsetConfig,
                                      OutputConfig, RegionSpec, TimeConfig,
                                      SourceConfig)
    n_time, n_nodes = 5, 7
    t = np.array("2000-01-01", dtype="datetime64[ns]") + \
        np.arange(n_time) * np.timedelta64(6, "h")
    u = xr.DataArray(np.arange(n_time)[:, None] * 100.0
                     + np.arange(n_nodes)[None, :],
                     dims=("time", "nod2"), coords={"time": t}, name="u")
    v = -u.rename("v")
    sidx = np.array([1, 3])
    cfg = SubsetConfig(source=SourceConfig(mesh="x"), region=RegionSpec(),
                       time=TimeConfig(),
                       output=OutputConfig(dir=str(tmp_path), format=fmt,
                                           layout="combined",
                                           write_manifest=False))
    mesh_ds = xr.Dataset({"lon": ("nod2", np.arange(2.0))})
    write_outputs(cfg, mesh_ds, {"u": _lazy(u, sidx), "v": _lazy(v, sidx)},
                  tmp_path, {})
    path = next(p for p in tmp_path.iterdir() if p.name.startswith("velocity"))
    out = (xr.open_dataset(path) if fmt == "netcdf"
           else xr.open_zarr(path, consolidated=False))
    assert out.sizes["time"] == n_time
    np.testing.assert_array_equal(out["time"].values, t)
    np.testing.assert_array_equal(out["u"].values, u.values[:, sidx])
    np.testing.assert_array_equal(out["v"].values, v.values[:, sidx])


def test_packed_source_keeps_missing_values(tmp_path):
    n_time, n_nodes = 5, 7
    t = np.array("2000-01-01", dtype="datetime64[ns]") + \
        np.arange(n_time) * np.timedelta64(1, "D")
    data = np.arange(n_time)[:, None] * 1.5 + np.arange(n_nodes)[None, :] * 0.25
    data[2, 3] = np.nan
    data[4, 1] = np.nan
    src_nc = tmp_path / "packed.nc"
    xr.DataArray(data, dims=("time", "nod2"), coords={"time": t},
                 name="unod").to_dataset().to_netcdf(
        src_nc, encoding={"unod": {"dtype": "int16", "scale_factor": 0.25,
                                   "add_offset": 0.0, "_FillValue": -32768}})
    src = xr.open_dataset(src_nc)["unod"]
    sidx = np.array([0, 1, 3, 4])
    _write_lazy_subset(_lazy(src, sidx), tmp_path / "B.nc", "netcdf", {},
                       progress=False)
    out = xr.open_dataset(tmp_path / "B.nc")["unod"].values
    exp = src.values[:, sidx]
    np.testing.assert_array_equal(np.isnan(out), np.isnan(exp))
    np.testing.assert_allclose(np.nan_to_num(out), np.nan_to_num(exp))


def test_zarr_sub_daily_steps_keep_their_times(tmp_path):
    src = _source(n_time=4)
    src = src.assign_coords(time=src["time"].values[0]
                            + np.arange(4) * np.timedelta64(6, "h"))
    _write_lazy_subset(_lazy(src, [0, 2], name="unod"),
                       tmp_path / "u6h.region.zarr", "zarr", {}, progress=False)
    out = xr.open_zarr(tmp_path / "u6h.region.zarr", consolidated=False)
    np.testing.assert_array_equal(out["time"].values, src["time"].values)
