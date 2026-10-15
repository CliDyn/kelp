"""HEALPix regional extractor (Phase 3): self-describing subsets, the
reader's extract mode, and the §4b equivalence requirement — tracking on
an extract == tracking on the whole store, up to sector-leavers.
"""
import sys
from pathlib import Path

import os

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")
xr = pytest.importorskip("xarray")
pytest.importorskip("zarr")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_healpix_region import extract_region  # noqa: E402

from kelp.core.config import (AppConfig, DataConfig, MeshConfig,  # noqa: E402
                                  ModeConfig, NumericsConfig, TimeConfig)
from kelp.readers.healpix import HealpixReader  # noqa: E402
from test_healpix_reader import (_make_eta_store, _make_uv_store,  # noqa: E402
                                 _mesh_cfg, _pix_lonlat, _wet)

BBOX = [-40.0, 40.0, -40.0, 40.0]


def _extract_cfg(src, out, *, bbox=None, halo=1, variables=None,
                 eta_path=None, **extra):
    cfg = {
        "source": {"path": str(src)},
        "region": {"bbox": bbox or BBOX, "halo_layers": halo},
        "output": {"path": str(out), "time_chunk": 2},
        "metadata": {"subset_name": "t", "description": "test"},
    }
    if variables:
        cfg["variables"] = list(variables)
    if eta_path:
        cfg["source"]["eta_path"] = str(eta_path)
    cfg.update(extra)
    return cfg


class TestExtractStore:
    def test_structure_and_attrs(self, tmp_path):
        src = _make_uv_store(tmp_path / "3D_src_ocean.nc", with_eta=True)
        out = tmp_path / "sector.zarr"
        extract_region(_extract_cfg(src, out))
        ds = xr.open_zarr(out)
        idx = ds["healpix_index"].values
        assert idx.dtype == np.int64
        assert (np.diff(idx) > 0).all()          # sorted, unique
        assert ds.attrs["grid_type"] == "healpix"
        assert ds.attrs["healpix_order"] == "nest"
        assert ds.attrs["halo_layers"] == 1
        assert ds.attrs["n_cells"] == idx.size
        assert ds["crs"].attrs["grid_mapping_name"] == "healpix"
        assert "avg_uoe" in ds and "avg_von" in ds and "avg_zos" in ds
        # every kept pixel centre is inside bbox+halo (halo = 1 ring)
        lon, lat = healpy.pix2ang(4, idx, nest=True, lonlat=True)
        lon = ((lon + 180.0) % 360.0) - 180.0
        assert (np.abs(lat) < 65.0).all()        # bbox lat 40 + halo pixels

    def test_eta_merged_from_sibling(self, tmp_path):
        src = _make_uv_store(tmp_path / "3D_daily_x_ocean.nc")
        _make_eta_store(tmp_path / "2D_daily_x_ocean.nc")
        out = tmp_path / "sector_eta.zarr"
        extract_region(_extract_cfg(src, out))
        ds = xr.open_zarr(out)
        assert "avg_zos" in ds.variables

    def test_time_and_level_slice(self, tmp_path):
        src = _make_uv_store(tmp_path / "3D_slice_ocean.nc", with_eta=True)
        out = tmp_path / "sector_sl.zarr"
        cfg = _extract_cfg(src, out)
        cfg["time"] = {"start": 1, "stop": 3}
        cfg["levels"] = 2
        extract_region(cfg)
        ds = xr.open_zarr(out)
        assert ds.sizes["time"] == 2
        assert ds.sizes["level"] == 2

    def test_ring_source_normalized_to_nest(self, tmp_path):
        src = _make_uv_store(tmp_path / "3D_ringsrc.nc", order="ring",
                             extra_vars=("avg_tos",), with_eta=True)
        out = tmp_path / "sector_ring.zarr"
        extract_region(_extract_cfg(
            src, out, variables=["avg_uoe", "avg_von", "avg_tos"]))
        ds = xr.open_zarr(out)
        idx = ds["healpix_index"].values
        # per-cell physics must sit at the NEST-canonical positions:
        # avg_tos = 0.1 * lat(pixel)
        _, lat_nest = healpy.pix2ang(4, idx, nest=True, lonlat=True)
        got = ds["avg_tos"].isel(time=0).values
        np.testing.assert_allclose(got, lat_nest * 0.1, atol=1e-6)

    def test_refuses_overwrite_and_double_extract(self, tmp_path):
        src = _make_uv_store(tmp_path / "3D_ow_ocean.nc", with_eta=True)
        out = tmp_path / "sector_ow.zarr"
        extract_region(_extract_cfg(src, out))
        with pytest.raises(FileExistsError, match="force"):
            extract_region(_extract_cfg(src, out))
        extract_region(_extract_cfg(src, out), force=True)   # ok
        with pytest.raises(ValueError, match="already an extract"):
            extract_region(_extract_cfg(out, tmp_path / "x.zarr"))

    def test_explicit_variables_require_uv(self, tmp_path):
        # an extract without velocities is not reader-consumable
        src = _make_uv_store(tmp_path / "3D_nouv_ocean.nc", with_eta=True,
                             extra_vars=("avg_tos",))
        with pytest.raises(ValueError, match="u/v pair"):
            extract_region(_extract_cfg(src, tmp_path / "n.zarr",
                                        variables=["avg_tos"]))

    def test_time_start_zero_and_provenance(self, tmp_path):
        src = _make_uv_store(tmp_path / "3D_t0_ocean.nc", with_eta=True)
        out = tmp_path / "sector_t0.zarr"
        cfg = _extract_cfg(src, out)
        cfg["time"] = {"start": 0, "stop": 2}   # 0 must NOT read as null
        extract_region(cfg)
        ds = xr.open_zarr(out)
        assert ds.sizes["time"] == 2
        assert list(ds.attrs["time_slice"]) == [0, 2, -1]
        assert ds.attrs["eta_merged"]
        assert ds.attrs["eta_source"] == "source:avg_zos"

    def test_empty_region_raises(self, tmp_path):
        src = _make_uv_store(tmp_path / "3D_empty_ocean.nc", with_eta=True)
        with pytest.raises(ValueError, match="no pixels"):
            extract_region(_extract_cfg(src, tmp_path / "e.zarr",
                                        bbox=[10.0, 11.0, 89.5, 89.9],
                                        halo=0))


class TestReaderExtractMode:
    def test_load_mesh_from_extract(self, tmp_path):
        src = _make_uv_store(tmp_path / "3D_rm_ocean.nc", with_eta=True)
        out = tmp_path / "sector_rm.zarr"
        extract_region(_extract_cfg(src, out))
        mesh = HealpixReader.load_mesh(_mesh_cfg(out))
        assert mesh.metadata["healpix_nside"] == 4
        assert mesh.n_nodes > 0
        # all nodes inside bbox+halo, all wet
        assert (np.abs(mesh.nodes_lat) < 65.0).all()
        ds = HealpixReader.open_dataset(out)
        vf = HealpixReader.read_velocities(ds, 0, mesh)
        assert np.isfinite(vf.u).all()
        np.testing.assert_allclose(vf.eta, 0.25)

    def test_extract_mesh_matches_direct_in_core(self, tmp_path):
        # §4b halo requirement: the dual mesh strictly inside the bbox is
        # IDENTICAL to the whole-store build (element member sets match)
        src = _make_uv_store(tmp_path / "3D_core_ocean.nc", with_eta=True)
        out = tmp_path / "sector_core.zarr"
        extract_region(_extract_cfg(src, out, halo=2))
        m_dir = HealpixReader.load_mesh(_mesh_cfg(src))
        m_ext = HealpixReader.load_mesh(_mesh_cfg(out))

        def _core_elems(mesh):
            elems = set()
            for e in range(mesh.n_elements):
                verts = mesh.elem_valid_nodes(e)
                pix = mesh._healpix_node_pix[verts]
                lon, lat = healpy.pix2ang(4, pix, nest=True, lonlat=True)
                lon = ((lon + 180.0) % 360.0) - 180.0
                if (np.abs(lon) < 35.0).all() and (np.abs(lat) < 35.0).all():
                    elems.add(tuple(sorted(int(p) for p in pix)))
            return elems

        assert _core_elems(m_ext) == _core_elems(m_dir)

    def test_extract_reader_varying_field_per_node(self, tmp_path):
        # reviewer blind spot: constant fields cannot catch a permuted
        # searchsorted compaction — pin the extract-mode read with a
        # lat-varying scalar, on a RING source for good measure
        src = _make_uv_store(tmp_path / "3D_var_ocean.nc", order="ring",
                             with_eta=True, extra_vars=("avg_tos",))
        out = tmp_path / "sector_var.zarr"
        extract_region(_extract_cfg(
            src, out, variables=["avg_uoe", "avg_von", "avg_tos"]))
        mesh = HealpixReader.load_mesh(_mesh_cfg(out))
        ds = HealpixReader.open_dataset(out)
        got = HealpixReader.read_scalar_fields(ds, 0, mesh, ["avg_tos"])
        np.testing.assert_allclose(got["avg_tos"], mesh.nodes_lat * 0.1,
                                   atol=1e-6)

    def test_crop_bbox_composes_on_extract(self, tmp_path):
        src = _make_uv_store(tmp_path / "3D_comp_ocean.nc", with_eta=True)
        out = tmp_path / "sector_comp.zarr"
        extract_region(_extract_cfg(src, out))          # bbox ±40
        mesh = HealpixReader.load_mesh(
            _mesh_cfg(out, crop_bbox=[-40.0, 40.0, -15.0, 15.0]))
        assert (np.abs(mesh.nodes_lat) <= 16).all()     # further crop

    def test_deptho_travels_with_extract(self, tmp_path):
        src = _make_uv_store(tmp_path / "3D_dep_ocean.nc", with_eta=True,
                             with_deptho=True)
        out = tmp_path / "sector_dep.zarr"
        extract_region(_extract_cfg(src, out))          # default variables
        ds = xr.open_zarr(out)
        assert "deptho" in ds.variables                 # auto-included
        mesh = HealpixReader.load_mesh(_mesh_cfg(out))
        np.testing.assert_allclose(mesh.depth, 123.0)

    def test_corrupt_healpix_index_raises(self, tmp_path):
        src = _make_uv_store(tmp_path / "3D_bad_ocean.nc", with_eta=True)
        out = tmp_path / "sector_bad.zarr"
        extract_region(_extract_cfg(src, out))
        ds = xr.open_zarr(out).load()
        idx = ds["healpix_index"].values.copy()
        idx[1] = idx[0]                          # not strictly increasing
        ds["healpix_index"] = (ds["healpix_index"].dims[0], idx)
        import shutil
        shutil.rmtree(out)
        ds.to_zarr(out)
        with pytest.raises(ValueError, match="strictly increasing"):
            HealpixReader.load_mesh(_mesh_cfg(out))


# Set KELP_TEST_NG5_STORE to an ng5 3-D HEALPix store; unset -> skipped.
_NG5_3D_Z7 = os.environ.get("KELP_TEST_NG5_STORE", "")


@pytest.mark.skipif(not (_NG5_3D_Z7 and Path(_NG5_3D_Z7).exists()),
                    reason="ng5 kerchunk store not available (set KELP_TEST_NG5_STORE)")
class TestRealNg5Extract:
    def test_extract_and_roundtrip(self, tmp_path):
        # pins the encoding-strip: kerchunk-GRIB variables carry the
        # gribscan read codec in .encoding, which zarr rejects at write
        pytest.importorskip("gribscan")
        pytest.importorskip("dask")
        out = tmp_path / "weddell_z7.zarr"
        extract_region({
            "source": {"path": _NG5_3D_Z7},
            "region": {"bbox": [-70.0, 20.0, -75.0, -50.0],
                       "halo_layers": 1, "max_levels": 1},
            "variables": ["avg_uoe", "avg_von"],
            "time": {"start": 0, "stop": 3},
            "output": {"path": str(out), "time_chunk": 3},
        })
        ds = xr.open_zarr(out)
        assert "avg_zos" in ds.variables       # auto-merged 2-D sibling
        mesh = HealpixReader.load_mesh(_mesh_cfg(out))
        vf = HealpixReader.read_velocities(
            HealpixReader.open_dataset(out), 0, mesh)
        assert np.isfinite(vf.u).all()
        assert np.abs(vf.u).max() < 5.0


class TestEquivalence:
    def test_tracking_extract_equals_direct(self, tmp_path):
        # §4b: same seeds on the extract and on the whole store —
        # identical trajectories for particles that never leave the sector
        from kelp.tracking import run_tracking
        from kelp.writers import load_trajectories
        src = _make_uv_store(tmp_path / "3D_eq_ocean.nc", nside=8,
                             with_eta=True, u_value=0.4)
        out = tmp_path / "sector_eq.zarr"
        extract_region(_extract_cfg(src, out, bbox=[-30.0, 30.0,
                                                    -30.0, 30.0], halo=2))
        seeds_path = tmp_path / "seeds.txt"
        seeds_path.write_text("5.0 5.0\n-10.0 -12.0\n0.0 20.0\n")

        def _run(store, tag):
            outp = tmp_path / f"traj_{tag}.nc"
            cfg = AppConfig(
                mode=ModeConfig(),
                numerics=NumericsConfig(uv_interp="barycentric"),
                mesh=MeshConfig(type="healpix", meshpath=str(store)),
                data=DataConfig(ncinput=str(store),
                                inicloud=str(seeds_path),
                                output_path=str(outp),
                                time_averaging="instant"),
                time=TimeConfig(start_time_idx=0, n_steps=2),
            )
            run_tracking(cfg)
            return load_trajectories(str(outp))

        t_dir = _run(src, "direct")
        t_ext = _run(out, "extract")
        # interior particles: bitwise-equal positions
        np.testing.assert_array_equal(t_ext["lon"], t_dir["lon"])
        np.testing.assert_array_equal(t_ext["lat"], t_dir["lat"])
        np.testing.assert_array_equal(t_ext["status"], t_dir["status"])
