# tests/test_fesom2_multiyear.py
import numpy as np
import pytest
import resource
from types import SimpleNamespace
from kelp.readers.fesom2 import Fesom2Reader
from kelp.readers import extract_time_info
from tests._fesom2_synth import write_multiyear

def _mode(**kw):
    base = dict(Fdrift3D=True, var_u="unod", var_v="vnod", var_w="w",
                var_eta="ssh", sample_vars=["temp"], max_levels=None,
                sigma_layer=None)
    base.update(kw); return SimpleNamespace(**base)

def test_multiyear_concat_axis_and_lazy(tmp_path):
    # Full-year lengths so consecutive per-year files (each "seconds since
    # <year>") are contiguous daily and the merged cadence stays regular;
    # tiny per-year lengths would leave ~362-day gaps at year boundaries and
    # make extract_time_info return file_dt=None.
    yl = {1958: 365, 1959: 365, 1960: 366}
    for v in ("unod", "vnod"):
        write_multiyear(tmp_path, v, yl, node_layout=True, vertical="nz1")
    write_multiyear(tmp_path, "w", yl, node_layout=True, vertical="nz")
    write_multiyear(tmp_path, "temp", yl, node_layout=True, vertical="nz1")
    write_multiyear(tmp_path, "ssh", yl, node_layout=True, vertical=None)
    # decoy files that the OLD broad glob would wrongly grab:
    write_multiyear(tmp_path, "uice", {1958: 365}, node_layout=True, vertical=None)

    ds = Fesom2Reader.open_dataset(str(tmp_path), mode_cfg=_mode())
    # exactly the requested family, no uice pollution
    assert set(ds.data_vars) >= {"unod", "vnod", "w", "temp", "ssh"}
    assert "uice" not in ds.data_vars
    # virtual time axis spans all years, monotonic
    assert ds.sizes["time"] == 1096
    times, file_dt = extract_time_info(ds)
    assert times is not None and np.all(np.diff(times) > 0)
    assert file_dt == 86400.0
    # lazy: arrays are dask-backed, not eagerly materialised
    assert "dask" in type(ds["unod"].data).__module__


def test_mixed_cadence_raises(tmp_path):
    # monthly u (1/file) vs daily w (3/file) -> join="exact" mismatch
    write_multiyear(tmp_path, "u", {1958: 1, 1959: 1}, node_layout=False, vertical="nz1")
    write_multiyear(tmp_path, "v", {1958: 1, 1959: 1}, node_layout=False, vertical="nz1")
    write_multiyear(tmp_path, "w", {1958: 3, 1959: 3}, node_layout=True, vertical="nz")
    # eta optional (no ssh file written) so the cadence mismatch is what raises.
    mode = _mode(var_u="u", var_v="v", var_w="w", var_eta=None, sample_vars=None)
    with pytest.raises(ValueError, match="cadence|exact|align"):
        Fesom2Reader.open_dataset(str(tmp_path), mode_cfg=mode)


def test_single_year_exact_layout_no_broad_glob_pollution(tmp_path):
    # FIX A: a single-YEAR exact-layout directory (each var = exactly 1 file)
    # must use the precise <name>.fesom.*.nc path, NOT fall through to the
    # old broad u*/v* glob (which grabs uice under key 'u' and renames node
    # vars -> KeyError 'unod').
    yl = {1958: 3}
    for v in ("unod", "vnod"):
        write_multiyear(tmp_path, v, yl, node_layout=True, vertical="nz1")
    write_multiyear(tmp_path, "w", yl, node_layout=True, vertical="nz")
    write_multiyear(tmp_path, "temp", yl, node_layout=True, vertical="nz1")
    write_multiyear(tmp_path, "ssh", yl, node_layout=True, vertical=None)
    # decoy the OLD broad glob would wrongly grab under key 'u':
    write_multiyear(tmp_path, "uice", yl, node_layout=True, vertical=None)

    mode = _mode()
    ds = Fesom2Reader.open_dataset(str(tmp_path), mode_cfg=mode)
    assert "unod" in ds.data_vars
    assert "uice" not in ds.data_vars
    # and the velocity actually reads (node layout, (K-1, N))
    from kelp.readers.fesom2 import _load_mesh_nc
    from tests._fesom2_synth import write_meshdiag
    mesh = _load_mesh_nc(write_meshdiag(tmp_path / "m.nc", "ugrid"))
    vel = Fesom2Reader.read_velocities(ds, 0, mesh, mode_cfg=mode,
                                       sample_vars=["temp"])
    assert vel.u.shape[1] == mesh.n_nodes
    assert vel.w is not None


def test_missing_required_var_w_raises(tmp_path):
    # FIX B: a configured var_w with zero matching files must raise (else the
    # integrators silently auto-detect a 2D run).
    yl = {1958: 3}
    for v in ("unod", "vnod"):
        write_multiyear(tmp_path, v, yl, node_layout=True, vertical="nz1")
    write_multiyear(tmp_path, "ssh", yl, node_layout=True, vertical=None)
    mode = _mode(var_w="nosuchw", sample_vars=None)
    with pytest.raises(ValueError, match="nosuchw"):
        Fesom2Reader.open_dataset(str(tmp_path), mode_cfg=mode)


def test_eta_alias_zos_picked_up(tmp_path):
    # FIX B: var_eta unset, eta stored as zos.fesom.*.nc (no ssh) -> the merged
    # multi-year dataset must expose the eta field via the alias list.
    yl = {1958: 365, 1959: 365}
    for v in ("unod", "vnod"):
        write_multiyear(tmp_path, v, yl, node_layout=True, vertical="nz1")
    write_multiyear(tmp_path, "w", yl, node_layout=True, vertical="nz")
    write_multiyear(tmp_path, "zos", yl, node_layout=True, vertical=None)
    mode = _mode(var_eta=None, sample_vars=None)
    ds = Fesom2Reader.open_dataset(str(tmp_path), mode_cfg=mode)
    assert "zos" in ds.data_vars


def test_time_axis_overlap_raises(tmp_path):
    # FIX C: two files whose decoded times overlap/duplicate -> ValueError.
    yl = {1958: 3, 1959: 3}
    for v in ("unod", "vnod"):
        # both files decode against 1958 -> identical absolute timestamps
        write_multiyear(tmp_path, v, yl, node_layout=True, vertical="nz1",
                        units_years={1959: 1958})
    write_multiyear(tmp_path, "w", yl, node_layout=True, vertical="nz",
                    units_years={1959: 1958})
    mode = _mode(var_eta=None, sample_vars=None)
    with pytest.raises(ValueError, match="time|increas|duplicat|overlap"):
        Fesom2Reader.open_dataset(str(tmp_path), mode_cfg=mode)


def test_year_gap_warns(tmp_path):
    # FIX C: non-consecutive years (1958, 1960; no 1959) -> RuntimeWarning,
    # not a raise.
    yl = {1958: 365, 1960: 366}
    for v in ("unod", "vnod"):
        write_multiyear(tmp_path, v, yl, node_layout=True, vertical="nz1")
    write_multiyear(tmp_path, "w", yl, node_layout=True, vertical="nz")
    mode = _mode(var_eta=None, sample_vars=None)
    with pytest.warns(RuntimeWarning, match="1959|gap|consecutive"):
        Fesom2Reader.open_dataset(str(tmp_path), mode_cfg=mode)


def test_validate_merged_time_handles_cftime_and_uint():
    # FIX C (review round 1): cftime/object and unsigned-integer time axes must
    # not crash the strict-increasing check (np.diff vs scalar 0 would).
    import xarray as xr
    from kelp.readers.fesom2 import _validate_merged_time
    cftime = pytest.importorskip("cftime")
    inc = [cftime.DatetimeNoLeap(1958, 1, d) for d in (1, 2, 3)]
    _validate_merged_time(
        xr.Dataset(coords={"time": ("time", np.array(inc, dtype=object))}))
    dec = [cftime.DatetimeNoLeap(1958, 1, d) for d in (3, 2, 1)]
    with pytest.raises(ValueError, match="increas"):
        _validate_merged_time(
            xr.Dataset(coords={"time": ("time", np.array(dec, dtype=object))}))
    with pytest.raises(ValueError, match="increas"):
        _validate_merged_time(xr.Dataset(
            coords={"time": ("time", np.array([3, 2, 1], dtype=np.uint32))}))


def test_partial_exact_layout_missing_v_raises(tmp_path):
    # FIX A (review round 1): a dir with unod.fesom.* but NO vnod.fesom.* is an
    # exact-layout dir with a broken/missing v -> must enter exact mode and
    # raise the required-var error, NOT fall through to the broad glob (which
    # would grab vice under key 'v').
    yl = {1958: 365}
    write_multiyear(tmp_path, "unod", yl, node_layout=True, vertical="nz1")
    write_multiyear(tmp_path, "vice", yl, node_layout=True, vertical=None)  # decoy
    write_multiyear(tmp_path, "w", yl, node_layout=True, vertical="nz")
    mode = _mode(var_eta=None, sample_vars=None)
    with pytest.raises(ValueError, match="vnod"):
        Fesom2Reader.open_dataset(str(tmp_path), mode_cfg=mode)


def test_malformed_year_token_not_exact_layout(tmp_path):
    # FIX A (review round 1): u.fesom.mean.nc has a non-year token -> must NOT
    # trip the exact-layout path; with no real yearly velocity files the dir
    # falls through to the broad glob.
    from kelp.readers.fesom2 import _fesom2_var_files
    yl_bad = {}  # write a single non-year file by hand
    write_multiyear(tmp_path, "u", {1958: 1}, node_layout=False, vertical=None)
    # rename to a non-year token
    (tmp_path / "u.fesom.1958.nc").rename(tmp_path / "u.fesom.mean.nc")
    assert _fesom2_var_files(tmp_path, "u") == []   # year filter rejects it
    # review round 2: a year token with a TRAILING extra segment must also be
    # rejected (tail must be EXACTLY "<YYYY>.nc", not "<YYYY>.extra.nc").
    (tmp_path / "u.fesom.mean.nc").rename(tmp_path / "u.fesom.2020.extra.nc")
    assert _fesom2_var_files(tmp_path, "u") == []
    from kelp.readers.fesom2 import _parse_year
    assert _parse_year(tmp_path / "u.fesom.2020.extra.nc") is None
    assert _parse_year(tmp_path / "u.fesom.2020.nc") == 2020


def test_var_w_alias_wo_in_exact_layout(tmp_path):
    # FIX A (review round 1): 3D with var_w unset must accept wo.fesom.*.nc via
    # the w/wo/wvel alias list (mirrors read_velocities).
    yl = {1958: 365, 1959: 365}
    for v in ("unod", "vnod"):
        write_multiyear(tmp_path, v, yl, node_layout=True, vertical="nz1")
    write_multiyear(tmp_path, "wo", yl, node_layout=True, vertical="nz")
    mode = _mode(var_w=None, var_eta=None, sample_vars=None)
    ds = Fesom2Reader.open_dataset(str(tmp_path), mode_cfg=mode)
    assert "wo" in ds.data_vars


def test_read_velocities_and_scalar_from_multiyear(tmp_path):
    from kelp.readers.fesom2 import _load_mesh_nc
    from tests._fesom2_synth import write_meshdiag
    mesh = _load_mesh_nc(write_meshdiag(tmp_path / "m.nc", "ugrid"))
    yl = {1958: 3, 1959: 3}
    for v in ("unod", "vnod"):
        write_multiyear(tmp_path, v, yl, node_layout=True, vertical="nz1")
    write_multiyear(tmp_path, "w", yl, node_layout=True, vertical="nz")
    write_multiyear(tmp_path, "temp", yl, node_layout=True, vertical="nz1")
    write_multiyear(tmp_path, "ssh", yl, node_layout=True, vertical=None)
    mode = _mode(max_levels=None)
    ds = Fesom2Reader.open_dataset(str(tmp_path), mode_cfg=mode)
    # global index 4 lives in the 2nd year file
    vel = Fesom2Reader.read_velocities(ds, 4, mesh, mode_cfg=mode,
                                       sample_vars=["temp"])
    assert vel.u.shape[1] == mesh.n_nodes      # (K-1, N), node layout
    assert vel.w is not None
    assert vel.scalars is not None and "temp" in vel.scalars
