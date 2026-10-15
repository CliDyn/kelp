"""Attached forcing files must share the velocity file's record axis.

The readers attach a separate v / w file (``data.ncinput_v`` / ``ncinput_w``)
and the per-variable files of a directory onto the velocity dataset.  xarray
aligns such an assignment on the ``time`` INDEX, so a file whose stamps differ
(another epoch, numeric seconds without units, another cadence, a sub-range)
used to be padded with NaN without a word: a W written by
``utils/reconstruct_w.py`` (numeric seconds since its first record) attached
to a FESOM2 u file (decoded dates) came out all-NaN (audit 2026-10-04, F-059 /
F-060).  These tests pin the two halves of the fix: the tool writes the
source's own time coordinate, and the readers refuse a mismatched record axis
instead of padding it.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from kelp.readers import get_reader
from tests._fesom2_synth import write_meshdiag

N_NODE, N_ELEM = 4, 2          # tests/_fesom2_synth.py geometry
N_IFC = 3                      # interfaces 0 / 10 / 30 m


def _load_cli():
    script = (pathlib.Path(__file__).resolve().parents[1]
              / "utils" / "reconstruct_w.py")
    spec = importlib.util.spec_from_file_location("reconstruct_w_cli_t", script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _days(n, start="2005-01-01", freq="1D"):
    return pd.date_range(start, periods=n, freq=freq)


def _write_uv_ssh(path, times, *, eta_amp=0.01):
    """3-D FESOM2-style u/v on (time, nz1, elem) and ssh on (time, nod2)."""
    n = len(times)
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {
            "u": (("time", "nz1", "elem"), 0.1 + 0.01 * rng.standard_normal((n, N_IFC - 1, N_ELEM))),
            "v": (("time", "nz1", "elem"), -0.05 + 0.01 * rng.standard_normal((n, N_IFC - 1, N_ELEM))),
            "ssh": (("time", "nod2"), eta_amp * np.sin(np.arange(n))[:, None] * np.ones((n, N_NODE))),
        },
        coords={"time": times},
    )
    ds.to_netcdf(path)
    return path


def _mode3d(**kw):
    from kelp.core.config import ModeConfig
    return ModeConfig(Fdrift2D=False, Fdrift3D=True, **kw)


def _write_w(path, time_coord, n=None):
    n = len(time_coord) if n is None else n
    coords = {} if time_coord is None else {"time": time_coord}
    ds = xr.Dataset(
        {"w": (("time", "nz", "nod2"), np.full((n, N_IFC, N_NODE), 1e-5))},
        coords=coords,
    )
    ds["w"].attrs["positive"] = "down"
    ds.to_netcdf(path)
    return path


# ---------------------------------------------------------------------------
# 1. The reconstruction tool writes the source's time axis
# ---------------------------------------------------------------------------

class TestReconstructedWAttaches:
    def test_reconstructed_w_attaches_to_its_source_u_file(self, tmp_path):
        """reconstruct_w → data.ncinput_w → finite W at every record."""
        cli = _load_cli()
        mesh_nc = write_meshdiag(tmp_path / "fesom.mesh.diag.nc", "comp")
        uv = _write_uv_ssh(tmp_path / "uvssh.nc", _days(4))
        cfg = cli.ReconstructConfig(
            source=cli.SourceConfig(meshpath=str(mesh_nc), u_path=str(uv),
                                    v_path=str(uv), ssh_path=str(uv)),
            output=cli.OutputConfig(dir=str(tmp_path), format="netcdf",
                                    filename="w.nc"),
            method="divergence", use_nod_area=False,
        )
        cli.run(cfg)

        ds = get_reader("fesom2").open_dataset(str(uv), w_path=str(tmp_path / "w.nc"))
        try:
            assert ds.sizes["time"] == 4
            w = ds["w"].values
            assert np.isfinite(w).all(), "attached W is NaN — record axes did not match"
        finally:
            ds.close()

    def test_output_keeps_source_time_encoding(self, tmp_path):
        """The written time axis carries the source's units and calendar, so
        any reader decodes it to the same instants as the velocity file."""
        cli = _load_cli()
        mesh_nc = write_meshdiag(tmp_path / "fesom.mesh.diag.nc", "comp")
        uv = tmp_path / "uvssh.nc"
        _write_uv_ssh(uv, _days(4))
        # re-encode the source time with an explicit noleap calendar
        with xr.open_dataset(uv) as src:
            src = src.load()
        src["time"].encoding.update(units="days since 2000-01-01",
                                    calendar="noleap")
        uv2 = tmp_path / "uvssh_noleap.nc"
        src.to_netcdf(uv2)
        cfg = cli.ReconstructConfig(
            source=cli.SourceConfig(meshpath=str(mesh_nc), u_path=str(uv2),
                                    v_path=str(uv2), ssh_path=str(uv2)),
            output=cli.OutputConfig(dir=str(tmp_path), format="netcdf",
                                    filename="w.nc"),
            method="divergence", use_nod_area=False,
        )
        cli.run(cfg)
        with xr.open_dataset(tmp_path / "w.nc", decode_times=False) as raw_w, \
                xr.open_dataset(uv2, decode_times=False) as raw_u:
            assert raw_w["time"].attrs.get("units") == raw_u["time"].attrs.get("units")
            assert raw_w["time"].attrs.get("calendar") == raw_u["time"].attrs.get("calendar")
            np.testing.assert_array_equal(raw_w["time"].values, raw_u["time"].values)

    def test_time_range_subset_keeps_its_own_stamps(self, tmp_path):
        """A ``time_range`` sub-run writes the stamps of THOSE records."""
        cli = _load_cli()
        mesh_nc = write_meshdiag(tmp_path / "fesom.mesh.diag.nc", "comp")
        uv = _write_uv_ssh(tmp_path / "uvssh.nc", _days(5))
        cfg = cli.ReconstructConfig(
            source=cli.SourceConfig(meshpath=str(mesh_nc), u_path=str(uv),
                                    v_path=str(uv), ssh_path=str(uv)),
            output=cli.OutputConfig(dir=str(tmp_path), format="netcdf",
                                    filename="w.nc"),
            method="divergence", use_nod_area=False, time_range=[1, 4],
        )
        cli.run(cfg)
        with xr.open_dataset(tmp_path / "w.nc") as w, xr.open_dataset(uv) as u:
            np.testing.assert_array_equal(w["time"].values,
                                          u["time"].values[1:4])


# ---------------------------------------------------------------------------
# 2. FESOM2 reader: attached files must share the record axis
# ---------------------------------------------------------------------------

class TestFesom2AttachedTimeAxis:
    def test_w_path_with_numeric_seconds_raises(self, tmp_path):
        uv = _write_uv_ssh(tmp_path / "uv.nc", _days(3))
        wf = _write_w(tmp_path / "w.nc", np.array([0.0, 86400.0, 172800.0]))
        with pytest.raises(ValueError, match="time"):
            get_reader("fesom2").open_dataset(str(uv), w_path=str(wf))

    def test_w_path_on_foreign_stamps_raises(self, tmp_path):
        """No stamp of the W file is a stamp of the u file."""
        uv = _write_uv_ssh(tmp_path / "uv.nc", _days(3))
        wf = _write_w(tmp_path / "w.nc", _days(3, start="2006-01-01"))
        with pytest.raises(ValueError, match="time"):
            get_reader("fesom2").open_dataset(str(uv), w_path=str(wf))

    def test_v_path_with_other_stamps_raises(self, tmp_path):
        u = tmp_path / "u.nc"
        xr.Dataset({"u": (("time", "elem"), np.ones((3, N_ELEM)))},
                   coords={"time": _days(3)}).to_netcdf(u)
        v = tmp_path / "v.nc"
        xr.Dataset({"v": (("time", "elem"), np.ones((3, N_ELEM)))},
                   coords={"time": _days(3, start="2006-01-01")}).to_netcdf(v)
        with pytest.raises(ValueError, match="time"):
            get_reader("fesom2").open_dataset(str(u), v_path=str(v))

    def test_w_path_same_axis_attaches(self, tmp_path):
        uv = _write_uv_ssh(tmp_path / "uv.nc", _days(3))
        wf = _write_w(tmp_path / "w.nc", _days(3))
        ds = get_reader("fesom2").open_dataset(str(uv), w_path=str(wf))
        try:
            assert np.isfinite(ds["w"].values).all()
        finally:
            ds.close()

    def test_w_path_without_time_index_attaches_positionally(self, tmp_path):
        """No time coordinate in the W file: nothing to mislabel, the records
        pair up by position (count still checked)."""
        uv = _write_uv_ssh(tmp_path / "uv.nc", _days(3))
        wf = _write_w(tmp_path / "w.nc", None, n=3)
        ds = get_reader("fesom2").open_dataset(str(uv), w_path=str(wf))
        try:
            assert np.isfinite(ds["w"].values).all()
        finally:
            ds.close()

    def test_directory_w_with_other_cadence_raises(self, tmp_path):
        d = tmp_path / "run"
        d.mkdir()
        xr.Dataset({"u": (("time", "elem"), np.ones((4, N_ELEM)))},
                   coords={"time": _days(4)}).to_netcdf(d / "u.nc")
        xr.Dataset({"v": (("time", "elem"), np.ones((4, N_ELEM)))},
                   coords={"time": _days(4)}).to_netcdf(d / "v.nc")
        xr.Dataset({"w": (("time", "nz", "nod2"), np.ones((2, N_IFC, N_NODE)))},
                   coords={"time": _days(2, freq="MS", start="2005-02-01")}).to_netcdf(d / "w.nc")
        with pytest.raises(ValueError, match="time"):
            get_reader("fesom2").open_dataset(str(d), mode_cfg=_mode3d())

    def test_directory_extra_file_on_other_axis_is_skipped(self, tmp_path):
        """A per-variable extra (temp*, salt*) on another cadence cannot be
        sampled consistently; it is skipped with a warning instead of
        stretching the velocity axis with NaN records."""
        d = tmp_path / "run"
        d.mkdir()
        for name in ("u", "v"):
            xr.Dataset({name: (("time", "elem"), np.ones((4, N_ELEM)))},
                       coords={"time": _days(4)}).to_netcdf(d / f"{name}.nc")
        xr.Dataset({"temp": (("time", "nod2"), np.ones((2, N_NODE)))},
                   coords={"time": _days(2, freq="MS")}).to_netcdf(d / "temp.nc")
        with pytest.warns(RuntimeWarning, match="temp"):
            ds = get_reader("fesom2").open_dataset(str(d))
        try:
            assert ds.sizes["time"] == 4
            assert "temp" not in ds
            assert np.isfinite(ds["u"].values).all()
        finally:
            ds.close()


# ---------------------------------------------------------------------------
# 3. FESOM-C reader: same rule for its separate v / w files
# ---------------------------------------------------------------------------

class TestFesomCAttachedTimeAxis:
    def test_w_path_with_other_stamps_raises(self, tmp_path):
        u = tmp_path / "u.nc"
        xr.Dataset({"u": (("time", "nele"), np.ones((3, N_ELEM)))},
                   coords={"time": _days(3)}).to_netcdf(u)
        wf = tmp_path / "w.nc"
        xr.Dataset({"w": (("time", "node", "sigma"), np.ones((3, N_NODE, 3)))},
                   coords={"time": np.array([0.0, 3600.0, 7200.0])}).to_netcdf(wf)
        with pytest.raises(ValueError, match="time"):
            get_reader("fesomc").open_dataset(str(u), w_path=str(wf))

    def test_v_path_same_axis_merges(self, tmp_path):
        u = tmp_path / "u.nc"
        xr.Dataset({"u": (("time", "nele"), np.ones((3, N_ELEM)))},
                   coords={"time": _days(3)}).to_netcdf(u)
        v = tmp_path / "v.nc"
        xr.Dataset({"v": (("time", "nele"), np.ones((3, N_ELEM)))},
                   coords={"time": _days(3)}).to_netcdf(v)
        ds = get_reader("fesomc").open_dataset(str(u), v_path=str(v))
        try:
            assert ds.sizes["time"] == 3
            assert np.isfinite(ds["v"].values).all()
        finally:
            ds.close()


# ---------------------------------------------------------------------------
# 4. Review round 1: what the equality-only check refused or let through
# ---------------------------------------------------------------------------

class TestAttachRules:
    def test_cftime_and_datetime64_with_the_same_instants_match(self):
        """A standard-calendar cftime axis and a datetime64 axis holding the
        same instants (NetCDF decodes both the same way, so this arises in
        memory): xarray aligns them, the check must accept them too."""
        from kelp.readers import _time_axis_mismatch
        a = xr.DataArray(np.ones(3), dims="time", coords={"time": _days(3)})
        cf = xr.date_range("2005-01-01", periods=3, freq="1D",
                           calendar="standard", use_cftime=True)
        b = xr.DataArray(np.ones(3), dims="time", coords={"time": cf})
        assert _time_axis_mismatch(a, b) is None
        c = xr.DataArray(np.ones(3), dims="time",
                         coords={"time": _days(3, start="2005-01-02")})
        assert _time_axis_mismatch(a, c) is not None

    def test_numeric_axes_in_other_units_are_refused(self, tmp_path):
        """Equal numbers in other units are other instants."""
        u = tmp_path / "u.nc"
        xr.Dataset({"u": (("time", "elem"), np.ones((2, N_ELEM))),
                    "v": (("time", "elem"), np.ones((2, N_ELEM)))},
                   coords={"time": ("time", [0.0, 1.0], {"units": "days"})}
                   ).to_netcdf(u)
        wf = tmp_path / "w.nc"
        xr.Dataset({"w": (("time", "nz", "nod2"), np.ones((2, N_IFC, N_NODE)))},
                   coords={"time": ("time", [0.0, 1.0], {"units": "hours"})}
                   ).to_netcdf(wf)
        with pytest.raises(ValueError, match="time"):
            get_reader("fesom2").open_dataset(str(u), w_path=str(wf))

    def test_w_covering_more_records_attaches_on_the_u_records(self, tmp_path):
        """A full-year W with a time-cropped u: every u stamp is in the W
        file, and the u records get exactly their W records."""
        uv = _write_uv_ssh(tmp_path / "uv.nc", _days(5, start="2005-01-04"))
        wf = tmp_path / "w.nc"
        w = np.arange(10, dtype=float)[:, None, None] * np.ones((10, N_IFC, N_NODE))
        xr.Dataset({"w": (("time", "nz", "nod2"), w)},
                   coords={"time": _days(10)}).to_netcdf(wf)
        ds = get_reader("fesom2").open_dataset(str(uv), w_path=str(wf))
        try:
            assert ds.sizes["time"] == 5
            np.testing.assert_array_equal(ds["w"].values[:, 0, 0], [3, 4, 5, 6, 7])
        finally:
            ds.close()

    def test_v_file_missing_some_u_records_is_refused(self, tmp_path):
        u = tmp_path / "u.nc"
        xr.Dataset({"u": (("time", "elem"), np.ones((3, N_ELEM)))},
                   coords={"time": _days(3)}).to_netcdf(u)
        v = tmp_path / "v.nc"
        xr.Dataset({"v": (("time", "elem"), np.ones((2, N_ELEM)))},
                   coords={"time": _days(2)}).to_netcdf(v)
        with pytest.raises(ValueError, match="time"):
            get_reader("fesom2").open_dataset(str(u), v_path=str(v))

    def test_w_file_without_a_time_dimension_is_refused(self, tmp_path):
        uv = _write_uv_ssh(tmp_path / "uv.nc", _days(3))
        wf = tmp_path / "w.nc"
        xr.Dataset({"w": (("nz", "nod2"), np.ones((N_IFC, N_NODE)))}).to_netcdf(wf)
        with pytest.raises(ValueError, match="time"):
            get_reader("fesom2").open_dataset(str(uv), w_path=str(wf))

    def test_time_range_w_attaches_and_refuses_uncovered_records(self, tmp_path):
        """A reconstruct_w time_range sub-run attaches to its own source u
        file; a record outside the reconstructed range is refused when the
        run asks for it (instead of a NaN vertical velocity)."""
        cli = _load_cli()
        mesh_nc = write_meshdiag(tmp_path / "fesom.mesh.diag.nc", "comp")
        uv = _write_uv_ssh(tmp_path / "uvssh.nc", _days(5))
        cfg = cli.ReconstructConfig(
            source=cli.SourceConfig(meshpath=str(mesh_nc), u_path=str(uv),
                                    v_path=str(uv), ssh_path=str(uv)),
            output=cli.OutputConfig(dir=str(tmp_path), format="netcdf",
                                    filename="w.nc"),
            method="divergence", use_nod_area=False, time_range=[1, 4],
        )
        cli.run(cfg)
        reader = get_reader("fesom2")
        mesh = reader.load_mesh(SimpleNamespace(meshpath=str(mesh_nc)))
        ds = reader.open_dataset(str(uv), w_path=str(tmp_path / "w.nc"))
        try:
            mode = _mode3d()
            for t in (1, 2, 3):
                vel = reader.read_velocities(ds, t, mesh, mode)
                assert np.isfinite(vel.w).all()
            for t in (0, 4):
                with pytest.raises(ValueError, match="record"):
                    reader.read_velocities(ds, t, mesh, mode)
        finally:
            ds.close()

    def test_directory_w_on_other_axis_is_skipped_in_2d(self, tmp_path):
        """A 2-D run never reads W: a w* file on another axis (or a stray
        wind_stress.nc caught by the w* glob) is skipped with a warning."""
        d = tmp_path / "run"
        d.mkdir()
        for name in ("u", "v"):
            xr.Dataset({name: (("time", "elem"), np.ones((4, N_ELEM)))},
                       coords={"time": _days(4)}).to_netcdf(d / f"{name}.nc")
        xr.Dataset({"w": (("time", "nz", "nod2"), np.ones((2, N_IFC, N_NODE)))},
                   coords={"time": _days(2, freq="MS", start="2005-02-01")}
                   ).to_netcdf(d / "w.nc")
        from kelp.core.config import ModeConfig
        with pytest.warns(RuntimeWarning, match="w.nc"):
            ds = get_reader("fesom2").open_dataset(str(d), mode_cfg=ModeConfig())
        try:
            assert ds.sizes["time"] == 4 and "w" not in ds
        finally:
            ds.close()

    def test_directory_input_ignores_ncinput_w_with_a_warning(self, tmp_path):
        d = tmp_path / "run"
        d.mkdir()
        for name in ("u", "v"):
            xr.Dataset({name: (("time", "elem"), np.ones((2, N_ELEM)))},
                       coords={"time": _days(2)}).to_netcdf(d / f"{name}.nc")
        wf = _write_w(tmp_path / "w_extra.nc", _days(2))
        with pytest.warns(RuntimeWarning, match="ignored"):
            ds = get_reader("fesom2").open_dataset(str(d), w_path=str(wf))
        ds.close()

    def test_reconstruct_w_refuses_an_ssh_file_on_other_records(self, tmp_path):
        """The tool pairs its inputs by record position; an ssh file on
        other stamps would give a wrong W stamped with u's axis."""
        cli = _load_cli()
        mesh_nc = write_meshdiag(tmp_path / "fesom.mesh.diag.nc", "comp")
        uv = _write_uv_ssh(tmp_path / "uv.nc", _days(4))
        ssh = tmp_path / "ssh.nc"
        xr.Dataset({"ssh": (("time", "nod2"), np.zeros((4, N_NODE)))},
                   coords={"time": _days(4, start="2005-01-02")}).to_netcdf(ssh)
        cfg = cli.ReconstructConfig(
            source=cli.SourceConfig(meshpath=str(mesh_nc), u_path=str(uv),
                                    v_path=str(uv), ssh_path=str(ssh)),
            output=cli.OutputConfig(dir=str(tmp_path), format="netcdf",
                                    filename="w.nc"),
            method="divergence", use_nod_area=False,
        )
        with pytest.raises(ValueError, match="ssh"):
            cli.run(cfg)


# ---------------------------------------------------------------------------
# 5. Review round 2
# ---------------------------------------------------------------------------

def _da(times, values=None, **coord_attrs):
    n = len(times)
    vals = np.arange(n, dtype=float) + 100.0 if values is None else values
    t = xr.DataArray(np.asarray(times), dims="time", attrs=coord_attrs)
    return xr.DataArray(vals, dims="time", coords={"time": t})


class TestMatchingRound2:
    def test_identical_axes_with_repeated_stamps_pair_by_position(self):
        """Concatenated segments with an overlap repeat a stamp; an identical
        axis on the other file must pair record by record, not take the
        first of the repeated records twice."""
        from kelp.readers import _align_onto_time_axis, _require_same_time_axis
        t = np.array(["2005-01-01", "2005-01-01", "2005-01-02"], dtype="datetime64[ns]")
        base, other = _da(t), _da(t)
        out = _align_onto_time_axis(base, other, "w", "u")
        np.testing.assert_array_equal(out.values, [100.0, 101.0, 102.0])
        _require_same_time_axis(base, other, "w", "u")          # no refusal

    def test_repeated_stamps_on_one_side_only_are_refused(self):
        from kelp.readers import _align_onto_time_axis
        tb = np.array(["2005-01-01", "2005-01-01", "2005-01-02"], dtype="datetime64[ns]")
        to = np.array(["2005-01-01", "2005-01-02", "2005-01-03"], dtype="datetime64[ns]")
        with pytest.raises(ValueError, match="repeated"):
            _align_onto_time_axis(_da(tb), _da(to), "w", "u", allow_missing=True)

    def test_empty_attached_file_is_a_clean_refusal(self):
        from kelp.readers import _align_onto_time_axis
        with pytest.raises(ValueError, match="no records"):
            _align_onto_time_axis(_da(_days(3).values), _da(_days(0).values), "w", "u")

    def test_bare_units_compare_in_rounded_seconds(self):
        """0.1 s and 100 ms are one instant; float scaling must not split them."""
        from kelp.readers import _time_axis_mismatch
        a = _da(np.array([0.1, 0.2, 0.3]), units="s")
        b = _da(np.array([100.0, 200.0, 300.0]), units="ms")
        assert _time_axis_mismatch(a, b) is None

    def test_coarser_w_with_holes_inside_its_range_is_refused(self):
        """A daily W next to 6-hourly u shares every 4th stamp; the records
        in between are another cadence, not the edge of a time_range."""
        from kelp.readers import _align_onto_time_axis
        u = _da(pd.date_range("2005-01-01", periods=9, freq="6h").values)
        w = _da(pd.date_range("2005-01-01", periods=3, freq="1D").values)
        with pytest.raises(ValueError, match="cadence"):
            _align_onto_time_axis(u, w, "w", "u", allow_missing=True)

    def test_finer_datetime_superset_attaches_on_the_u_records(self):
        """A strided u extract (every 2nd day) next to the daily W: dates are
        unambiguous instants, the u records are taken."""
        from kelp.readers import _align_onto_time_axis
        u = _da(pd.date_range("2005-01-01", periods=3, freq="2D").values)
        w = _da(pd.date_range("2005-01-01", periods=5, freq="1D").values)
        # sub-sampling a finer cadence is accepted, but not silently: a
        # time-mean u next to a daily W would take the W of one day
        with pytest.warns(RuntimeWarning, match="2 of its records"):
            out = _align_onto_time_axis(u, w, "w", "u")
        np.testing.assert_array_equal(out.values, [100.0, 102.0, 104.0])


class TestReaderRound2:
    def test_partial_w_stays_lazy(self, tmp_path):
        """Filling the uncovered records must not read the whole W into RAM."""
        uv = _write_uv_ssh(tmp_path / "uv.nc", _days(5))
        wf = _write_w(tmp_path / "w.nc", _days(3, start="2005-01-02"))
        ds = get_reader("fesom2").open_dataset(str(uv), w_path=str(wf))
        try:
            assert ds["w"].chunks is not None
            w = ds["w"].values
            assert np.isnan(w[0]).all() and np.isnan(w[4]).all()
            assert np.isfinite(w[1:4]).all()
        finally:
            ds.close()

    def test_prefetch_does_not_refuse_an_uncovered_look_ahead_record(self, tmp_path):
        from kelp.tracking import _VelCache
        mesh_nc = write_meshdiag(tmp_path / "fesom.mesh.diag.nc", "comp")
        uv = _write_uv_ssh(tmp_path / "uv.nc", _days(6))
        wf = _write_w(tmp_path / "w.nc", _days(3, start="2005-01-02"))   # records 1..3
        reader = get_reader("fesom2")
        mesh = reader.load_mesh(SimpleNamespace(meshpath=str(mesh_nc)))
        ds = reader.open_dataset(str(uv), w_path=str(wf))
        try:
            cache = _VelCache(prefetch_size=4, n_time=6)
            vel = cache.get(2, reader, ds, mesh, _mode3d())   # window 2..5
            assert np.isfinite(vel.w).all()
            with pytest.raises(ValueError, match="record"):
                cache.get(5, reader, ds, mesh, _mode3d())
        finally:
            ds.close()

    def test_directory_w_without_mode_is_refused_not_skipped(self, tmp_path):
        """Without mode_cfg the reader cannot know the run is 2-D."""
        d = tmp_path / "run"
        d.mkdir()
        for name in ("u", "v"):
            xr.Dataset({name: (("time", "elem"), np.ones((4, N_ELEM)))},
                       coords={"time": _days(4)}).to_netcdf(d / f"{name}.nc")
        xr.Dataset({"w": (("time", "nz", "nod2"), np.ones((2, N_IFC, N_NODE)))},
                   coords={"time": _days(2, freq="MS", start="2005-02-01")}
                   ).to_netcdf(d / "w.nc")
        with pytest.raises(ValueError, match="time"):
            get_reader("fesom2").open_dataset(str(d))

    def test_empty_extra_file_in_a_directory_is_skipped(self, tmp_path):
        d = tmp_path / "run"
        d.mkdir()
        for name in ("u", "v"):
            xr.Dataset({name: (("time", "elem"), np.ones((4, N_ELEM)))},
                       coords={"time": _days(4)}).to_netcdf(d / f"{name}.nc")
        xr.Dataset({"temp": (("time", "nod2"), np.ones((0, N_NODE)))},
                   coords={"time": _days(0)}).to_netcdf(d / "temp.nc")
        with pytest.warns(RuntimeWarning, match="temp"):
            ds = get_reader("fesom2").open_dataset(str(d))
        try:
            assert "temp" not in ds and ds.sizes["time"] == 4
        finally:
            ds.close()

    def test_zarr_store_with_a_w_file_attaches_without_a_warning(self, tmp_path):
        import warnings as _w
        store = tmp_path / "uv.zarr"
        xr.open_dataset(_write_uv_ssh(tmp_path / "uv.nc", _days(3))).to_zarr(store)
        wf = _write_w(tmp_path / "w.nc", _days(3))
        with _w.catch_warnings():
            _w.simplefilter("error", RuntimeWarning)
            ds = get_reader("fesom2").open_dataset(str(store), w_path=str(wf))
        try:
            assert np.isfinite(ds["w"].values).all()
        finally:
            ds.close()

    def test_fesomc_static_v_file_is_refused(self, tmp_path):
        u = tmp_path / "u.nc"
        xr.Dataset({"u": (("time", "nele"), np.ones((3, N_ELEM)))},
                   coords={"time": _days(3)}).to_netcdf(u)
        v = tmp_path / "v.nc"
        xr.Dataset({"v": (("nele",), np.ones(N_ELEM))}).to_netcdf(v)
        with pytest.raises(ValueError, match="time"):
            get_reader("fesomc").open_dataset(str(u), v_path=str(v))


class TestFinalRound:
    def test_directory_ssh_covering_part_of_the_records_is_kept(self, tmp_path):
        """A contiguous partial ssh (or extra) field keeps its values on the
        records it covers, as the label attach did, instead of being
        dropped (final review)."""
        d = tmp_path / "run"
        d.mkdir()
        for name in ("u", "v"):
            xr.Dataset({name: (("time", "elem"), np.ones((4, N_ELEM)))},
                       coords={"time": _days(4)}).to_netcdf(d / f"{name}.nc")
        xr.Dataset({"ssh": (("time", "nod2"), np.full((2, N_NODE), 1.5))},
                   coords={"time": _days(2, start="2005-01-02")}).to_netcdf(d / "ssh.nc")
        from kelp.core.config import ModeConfig
        ds = get_reader("fesom2").open_dataset(str(d), mode_cfg=ModeConfig())
        try:
            assert ds.sizes["time"] == 4 and "ssh" in ds
            s = ds["ssh"].values
            np.testing.assert_array_equal(s[1:3], 1.5)
            assert np.isnan(s[0]).all() and np.isnan(s[3]).all()
        finally:
            ds.close()

    def test_dates_of_an_attached_file_are_kept_when_u_has_none(self, tmp_path):
        """u without a time coordinate, v with dates: the records pair by
        position and the dataset keeps v's dates (its only clock)."""
        from kelp.readers import extract_time_info
        u = tmp_path / "u.nc"
        xr.Dataset({"u": (("time", "elem"), np.ones((3, N_ELEM)))}).to_netcdf(u)
        v = tmp_path / "v.nc"
        xr.Dataset({"v": (("time", "elem"), np.ones((3, N_ELEM)))},
                   coords={"time": _days(3)}).to_netcdf(v)
        ds = get_reader("fesom2").open_dataset(str(u), v_path=str(v))
        try:
            times, fdt = extract_time_info(ds)
            assert fdt == 86400.0 and times is not None
        finally:
            ds.close()


class TestFinalRound2:
    def test_single_file_w_that_does_not_fit_is_skipped_in_2d(self, tmp_path):
        """A 2-D run never reads W: a stale ncinput_w kept from a 3-D config
        is skipped with a warning, as in the directory layout."""
        from kelp.core.config import ModeConfig
        uv = _write_uv_ssh(tmp_path / "uv.nc", _days(3))
        wf = _write_w(tmp_path / "w.nc", np.array([0.0, 86400.0, 172800.0]))
        with pytest.warns(RuntimeWarning, match="w.nc"):
            ds = get_reader("fesom2").open_dataset(str(uv), w_path=str(wf),
                                                   mode_cfg=ModeConfig())
        try:
            assert "w" not in ds
        finally:
            ds.close()

    def test_identical_axes_with_a_nat_pair_by_position(self):
        from kelp.readers import _time_axis_mismatch
        t = np.array(["2005-01-01", "NaT", "2005-01-03"], dtype="datetime64[ns]")
        assert _time_axis_mismatch(_da(t), _da(t)) is None


def test_a_partly_covering_ssh_file_warns_with_the_missing_records(tmp_path):
    """Final review round (2026-10-04): an ssh* file covering only some of the
    velocity records was attached with NaN silently."""
    import warnings as _w
    import xarray as xr
    from kelp.readers import _align_onto_time_axis
    t = np.array(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"], dtype="datetime64[ns]")
    u = xr.DataArray(np.zeros((4, 3)), dims=("time", "nod2"), coords={"time": t})
    ssh = xr.DataArray(np.ones((2, 3)), dims=("time", "nod2"), coords={"time": t[2:]})
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        out = _align_onto_time_axis(u, ssh, "ssh.nc", allow_missing=True, warn_missing=True)
    assert any("records 0, 1" in str(r.message) for r in rec)
    assert np.isnan(np.asarray(out.isel(time=0))).all()
