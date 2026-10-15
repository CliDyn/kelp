"""Small engine defects found by the documentation audit of 2026-10-04."""
import warnings

import numpy as np
import pytest


# --- F-047: Zarr v3 stores without a .zarr suffix ---------------------------

def test_detect_engine_recognises_a_zarr_v3_store_without_suffix(tmp_path):
    """Zarr v3 writes `zarr.json` and no `.zattrs`: a suffix-less store was
    handed to the NetCDF engine."""
    import xarray as xr
    from kelp.readers import _detect_engine
    store = tmp_path / "store_nosuffix"
    xr.Dataset({"u": (("time",), np.zeros(3))}).to_zarr(store, zarr_format=3)
    assert (store / "zarr.json").exists()
    assert _detect_engine(store) == "zarr"


# --- F-048: CF flag attributes on the Zarr failure_reason ---------------------

def test_zarr_failure_reason_carries_cf_flags(tmp_path):
    import zarr
    from kelp.core.particles import make_state
    from kelp.writers import TrajectoryWriter
    st = make_state(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    st = st.replace(failure_reason=np.zeros(2, dtype=np.int8))
    path = tmp_path / "t.zarr"
    with TrajectoryWriter(str(path), 2) as w:
        w.append(st, 0.0)
    grp = zarr.open_group(str(path), mode="r")
    attrs = dict(grp["failure_reason"].attrs)
    assert "flag_values" in attrs and "flag_meanings" in attrs
    assert "interpolation_failed" in attrs["flag_meanings"].lower()
    status = dict(grp["status"].attrs)
    assert status.get("flag_meanings") == "alive beached outside not_released"


# --- F-020: the Milstein variable-Kh warning, once per realization ----------

def test_m1_variable_kh_warning_is_issued_once_per_realization():
    """The one-shot key was the arrays' id: in temporal-interp mode each
    step blends a new Kh array, so the warning was re-issued every step
    (hidden only by Python's default once-per-location filter)."""
    from types import SimpleNamespace
    from kelp.numerics.diffusion import DiffusionStepArgs
    from kelp.numerics import integrators as it
    it._M1_VARIABLE_KH_WARNED.clear()
    args = DiffusionStepArgs(scheme="m1", kh_x=0.0, kh_y=0.0, seed=7,
                             real_id=0, step_i=0, composition="combined",
                             dres_m=1000.0, kh_eval_position="end")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        for step in range(3):           # a fresh blended Kh array each step
            vel = SimpleNamespace(kh_zonal=np.linspace(1.0, 2.0, 5) + step,
                                  kh_meridional=np.linspace(1.0, 2.0, 5))
            it._maybe_warn_m1_variable_kh(args, vel)
    hits = [w for w in rec if "separable" in str(w.message)]
    assert len(hits) == 1


# --- F-042: seed filtering must not rebuild the grid hash -------------------

def test_seed_generation_keeps_the_existing_grid_hash():
    """_filter_to_mesh tested the lazily built KD-tree, which is always
    None after build_search, so every generate_* call rebuilt the hash."""
    from kelp.core.mesh import Mesh, build_node_to_elems, build_search
    from kelp.core.particles_gen import generate_around_point
    lon, lat = np.meshgrid(np.linspace(0, 2, 5), np.linspace(0, 2, 5))
    lon, lat = lon.ravel(), lat.ravel()
    tris = []
    for j in range(4):
        for i in range(4):
            a = j * 5 + i
            tris += [[a, a + 1, a + 6], [a, a + 6, a + 5]]
    en = np.array(tris, dtype=np.int32)
    n2e, cnt = build_node_to_elems(lon.size, en)
    mesh = Mesh(name="g", nodes_lon=lon, nodes_lat=lat, depth=np.ones(lon.size),
                elem_nodes=en, elem_center_lon=lon[en].mean(1),
                elem_center_lat=lat[en].mean(1), node_to_elems=n2e,
                node_to_elems_count=cnt)
    build_search(mesh)
    offsets = mesh._gh_offsets
    generate_around_point(mesh, 1.0, 1.0, n=20, radius_km=30.0, seed=1)
    generate_around_point(mesh, 1.0, 1.0, n=20, radius_km=30.0, seed=2)
    assert mesh._gh_offsets is offsets


# --- F-046: the NEMO reader's advice names the key that decides ---------------

def test_nemo_3d_refusal_names_fdrift3d(tmp_path):
    from types import SimpleNamespace
    from kelp.readers import get_reader
    import xarray as xr
    t = tmp_path / "x_grid_T.nc"
    xr.Dataset({"ssh": (("time_counter", "y", "x"), np.zeros((1, 2, 2)))}).to_netcdf(t)
    mode = SimpleNamespace(Fdrift2D=True, Fdrift3D=True, var_u=None, var_v=None,
                           var_w=None, var_eta=None, sample_vars=[])
    with pytest.raises(NotImplementedError, match="Fdrift3D: false"):
        get_reader("nemo").open_dataset(str(t), mode_cfg=mode)
