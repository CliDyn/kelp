"""Trajectory output format 2 (release 1.0) — the writer side.

Described in docs/io_and_time.md §5 and §6: the lazy, crash-safe restart
block that carries the release positions, and the format-2 schema (record
dimension ``time``, CF trajectory attributes, release variables, NaN fills,
independent lazy velocity variables) on both backends.
"""
import numpy as np
import pytest
import xarray as xr


def _state(P=3):
    from kelp.core.particles import make_state
    return make_state(np.linspace(5.0, 5.2, P), np.full(P, 5.0))


@pytest.fixture(params=[".nc", ".zarr", ".nc-buffered"])
def out_path(request, tmp_path):
    ext = request.param
    flush = 2 if ext == ".nc-buffered" else 0
    path = tmp_path / ("out.nc" if ext.startswith(".nc") else "out.zarr")
    return str(path), flush


def _open_any(path, **kw):
    return (xr.open_zarr(path, consolidated=False, **kw)
            if path.endswith(".zarr") else xr.open_dataset(path, **kw))


class _FailingVariable:
    """A netCDF4 variable whose writes raise."""

    def __init__(self, var, exc):
        self._var, self._exc = var, exc

    def __setitem__(self, key, val):
        raise self._exc("killed")

    def __getattr__(self, name):
        return getattr(self._var, name)


class _FailingVariables(dict):
    """``Dataset.variables`` with the writes into one variable failing."""

    def __init__(self, variables, write, exc):
        super().__init__(variables)
        self._write, self._exc = write, exc

    def __getitem__(self, name):
        var = super().__getitem__(name)
        return _FailingVariable(var, self._exc) if name == self._write else var


class _FailingDataset:
    """An open netCDF4 dataset that fails at creating the variable
    ``create`` or at writing into the variable ``write``.

    The failure is put around the object, not into netCDF4's classes: some
    builds refuse that ("cannot set 'createVariable' attribute of immutable
    type", netCDF4 1.7.4 on Python 3.11)."""

    def __init__(self, ds, create, write, exc):
        object.__setattr__(self, "_nc", (ds, create, write, exc))

    def createVariable(self, name, *a, **k):
        ds, create, _, exc = self._nc
        if name == create:
            raise exc("killed")
        return ds.createVariable(name, *a, **k)

    @property
    def variables(self):
        ds, _, write, exc = self._nc
        return _FailingVariables(ds.variables, write, exc)

    def __getattr__(self, name):
        return getattr(self._nc[0], name)

    def __setattr__(self, name, value):
        setattr(self._nc[0], name, value)


def netcdf_fails_at(monkeypatch, *, create=None, write=None,
                    exc=RuntimeError):
    """Make every netCDF4 dataset opened from now on fail at ``create`` or
    ``write``.  Call it BEFORE ``TrajectoryWriter.open()``: without
    buffering the writer keeps the dataset it opens there."""
    import netCDF4
    real = netCDF4.Dataset

    def opening(*a, **k):
        return _FailingDataset(real(*a, **k), create, write, exc)

    monkeypatch.setattr(netCDF4, "Dataset", opening)


# ===================================================================
# Task 2 — lazy, crash-safe restart block; release positions ride it
# ===================================================================

class TestRestartBlock:
    def test_absent_without_checkpoint(self, out_path):
        from kelp.writers import TrajectoryWriter, load_restart
        path, flush = out_path
        st = _state()
        with TrajectoryWriter(path, 3, flush_every=flush) as w:
            w.append(st, 0.0)
        ds = _open_any(path)
        assert not [v for v in ds.variables if v.startswith("restart_")]
        with pytest.raises(ValueError, match="No valid restart"):
            load_restart(path)

    def test_release_carried_through_checkpoint(self, out_path):
        from kelp.writers import TrajectoryWriter, load_restart
        path, flush = out_path
        st = _state().replace(release_lon=np.array([1.0, 2.0, 3.0]),
                              release_lat=np.array([4.0, 5.0, 6.0]),
                              release_z=np.array([0.0, 1.0, 2.0]))
        with TrajectoryWriter(path, 3, flush_every=flush) as w:
            w.append(st, 0.0)
            w.write_checkpoint(st, 0, 0)
        r = load_restart(path)
        np.testing.assert_array_equal(r["state"].release_lon, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(r["state"].release_lat, [4.0, 5.0, 6.0])
        np.testing.assert_array_equal(r["state"].release_z, [0.0, 1.0, 2.0])

    def test_no_release_in_state_gives_none(self, out_path):
        from kelp.writers import TrajectoryWriter, load_restart
        path, flush = out_path
        with TrajectoryWriter(path, 3, flush_every=flush) as w:
            w.write_checkpoint(_state(), 0, 0)
        st = load_restart(path)["state"]
        assert st.release_lon is None and st.release_z is None

    def test_interrupted_first_checkpoint_is_invalid(self, out_path,
                                                     monkeypatch):
        """Kill the FIRST checkpoint in the middle of its payload (zarr
        accepts a wrong-length assignment silently, so the failure is
        injected): the flag must already exist and read 0."""
        import netCDF4
        import zarr
        from kelp.writers import TrajectoryWriter, load_restart
        path, flush = out_path

        class Boom(RuntimeError):
            pass

        orig_z = zarr.Array.__setitem__

        def z_set(self, key, val):
            if self.basename == "restart_nearest_node":
                raise Boom()
            return orig_z(self, key, val)

        st = _state()
        if not path.endswith(".zarr"):
            netcdf_fails_at(monkeypatch, write="restart_nearest_node",
                            exc=Boom)
        w = TrajectoryWriter(path, 3, flush_every=flush).open()
        w.append(st, 0.0)
        if path.endswith(".zarr"):
            monkeypatch.setattr(zarr.Array, "__setitem__", z_set)
        with pytest.raises(Boom):
            w.write_checkpoint(st, 0, 0)
        monkeypatch.undo()
        w.close()
        with pytest.raises(ValueError, match="No valid restart"):
            load_restart(path)
        if path.endswith(".zarr"):
            flag = zarr.open_group(path, mode="r")["restart_valid"][0]
        else:
            with netCDF4.Dataset(path) as ds:
                flag = ds.variables["restart_valid"][0]
        assert int(np.ma.filled(flag, -1)) == 0

    def test_second_checkpoint_after_lazy_block(self, out_path):
        """The block is created once; a later checkpoint overwrites it."""
        from kelp.writers import TrajectoryWriter, load_restart
        path, flush = out_path
        a = _state()
        b = a.replace(lon=a.lon + 1.0, release_lon=np.full(3, 7.0),
                      release_lat=np.full(3, 8.0), release_z=np.zeros(3))
        with TrajectoryWriter(path, 3, flush_every=flush) as w:
            w.append(a, 0.0)
            w.write_checkpoint(a, 0, 0)
            w.append(b, 1.0)
            w.write_checkpoint(b, 1, 1)
        r = load_restart(path)
        assert r["step_i"] == 1
        np.testing.assert_array_equal(r["state"].lon, b.lon)
        np.testing.assert_array_equal(r["state"].release_lat, np.full(3, 8.0))

    def test_standalone_restart_file_carries_release(self, tmp_path):
        from kelp.writers import write_restart_file, load_restart
        st = _state().replace(release_lon=np.ones(3), release_lat=np.ones(3),
                              release_z=np.zeros(3))
        p = str(tmp_path / "r.nc")
        write_restart_file(p, st, 4, 2)
        r = load_restart(p)
        np.testing.assert_array_equal(r["state"].release_lat, np.ones(3))
        np.testing.assert_array_equal(r["state"].release_z, np.zeros(3))
        # the standalone file lists its optional arrays like a checkpoint
        import netCDF4
        with netCDF4.Dataset(p) as ds:
            assert set(ds.getncattr("restart_optional").split()) == {
                "restart_release_lon", "restart_release_lat",
                "restart_release_z"}

    def test_unwritten_netcdf_flag_is_invalid(self, tmp_path):
        import netCDF4
        from kelp.writers import load_restart
        p = str(tmp_path / "masked.nc")
        with netCDF4.Dataset(p, "w") as ds:
            ds.createDimension("trajectory", 2)
            ds.createVariable("restart_valid", "i4")          # never written
            for n in ("restart_lon", "restart_lat", "restart_z"):
                ds.createVariable(n, "f8", ("trajectory",))[:] = 0.0
        with pytest.raises(ValueError, match="No valid restart"):
            load_restart(p)

    def test_restart_file_length_check(self, tmp_path):
        from kelp.writers import write_restart_file
        st = _state().replace(release_lon=np.ones(5), release_lat=np.ones(5),
                              release_z=np.zeros(5))
        p = tmp_path / "r.nc"
        with pytest.raises(ValueError, match="length"):
            write_restart_file(str(p), st, 0, 0)
        assert list(tmp_path.iterdir()) == []      # no file, no temporary

    def test_restart_file_length_check_wet_streak(self, tmp_path):
        from kelp.writers import write_restart_file
        with pytest.raises(ValueError, match="length"):
            write_restart_file(str(tmp_path / "r.nc"), _state(), 0, 0,
                               wet_streak=np.zeros(2, np.int32))

    def test_subset_state_restart_roundtrip(self, tmp_path):
        """The pulsed restart-split recipe: a subset state writes and reads
        back with its release arrays sliced like the positions."""
        from kelp.writers import write_restart_file, load_restart
        st = _state(P=4).replace(release_lon=np.arange(4.0),
                                 release_lat=np.arange(4.0) + 10,
                                 release_z=np.zeros(4))
        p = str(tmp_path / "r.nc")
        write_restart_file(p, st.subset(np.array([1, 3])), 0, 0)
        r = load_restart(p)["state"]
        np.testing.assert_array_equal(r.release_lon, [1.0, 3.0])
        np.testing.assert_array_equal(r.lon, st.lon[[1, 3]])


# ===================================================================
# Task 3 — format-2 schema
# ===================================================================

class TestSchema:
    def test_layout_and_cf_attrs(self, out_path):
        from kelp.writers import TrajectoryWriter
        path, flush = out_path
        st = _state()
        with TrajectoryWriter(path, 3, flush_every=flush) as w:
            w.set_time_units("seconds since 2005-01-01 23:15:00",
                             "proleptic_gregorian")
            w.append(st, 3600.0)
            w.append(st, 7200.0)
        ds = _open_any(path)
        assert ds.attrs["featureType"] == "trajectory"
        assert ds.attrs["Conventions"] == "CF-1.8"
        assert int(ds.attrs["kelp_output_format"]) == 2
        assert ds["lon"].dims == ("trajectory", "time")
        assert "obs" not in ds.dims
        assert ds["trajectory"].attrs["cf_role"] == "trajectory_id"
        assert ds["z"].attrs["positive"] == "down"
        assert str(ds["time"].values[0]).startswith("2005-01-02T00:15")
        assert ds.sel(time="2005-01-02T01:15").lon.shape == (3,)
        kw = dict(decode_times=False, decode_coords=False)  # keep `coordinates` in attrs
        raw = _open_any(path, **kw)
        assert raw["status"].attrs["coordinates"] == "lat lon z"
        assert raw["time"].attrs["standard_name"] == "time"
        assert raw["time"].attrs["axis"] == "T"
        assert raw["time"].attrs["calendar"] == "proleptic_gregorian"
        assert raw["lon"].attrs["units"] == "degrees_east"
        assert raw["lat"].attrs["standard_name"] == "latitude"
        assert raw["z"].attrs["standard_name"] == "depth"
        assert raw["z"].attrs["units"] == "m"

    def test_format_version_attribute_is_int32_netcdf(self, tmp_path):
        import netCDF4
        from kelp.writers import OUTPUT_FORMAT, TrajectoryWriter
        assert OUTPUT_FORMAT == 2
        p = str(tmp_path / "x.nc")
        with TrajectoryWriter(p, 3):
            pass
        with netCDF4.Dataset(p) as ds:
            v = ds.getncattr("kelp_output_format")
            assert np.asarray(v).dtype == np.int32 and int(v) == 2

    def test_default_time_units_without_setter(self, out_path):
        from kelp.writers import TrajectoryWriter
        path, flush = out_path
        with TrajectoryWriter(path, 3, flush_every=flush) as w:
            w.append(_state(), 5.0)
        raw = _open_any(path, decode_times=False)
        assert raw["time"].attrs["units"] == "seconds"
        assert "standard_name" not in raw["time"].attrs
        assert float(raw["time"][0]) == 5.0

    def test_no_epoch_units(self, out_path):
        from kelp.readers import NO_EPOCH_NOTE
        from kelp.writers import TrajectoryWriter
        path, flush = out_path
        with TrajectoryWriter(path, 3, flush_every=flush) as w:
            w.set_time_units("seconds", origin_note=NO_EPOCH_NOTE)
            w.write_release(np.zeros(3), np.zeros(3), np.zeros(3),
                            np.zeros(3))
            w.append(_state(), 10.0)
        ds = _open_any(path)
        assert ds["time"].attrs["time_origin"] == NO_EPOCH_NOTE
        # release_time shares the units AND the origin note of time
        assert ds["release_time"].attrs["time_origin"] == NO_EPOCH_NOTE
        assert "standard_name" not in ds["time"].attrs
        assert float(ds["time"][0]) == 10.0

    def test_cycle_period_attribute(self, out_path):
        from kelp.writers import TrajectoryWriter
        path, flush = out_path
        with TrajectoryWriter(path, 3, flush_every=flush) as w:
            w.set_time_units("seconds since 2001-01-01 00:00:00", "standard",
                             cycle_period=10800.0)
            w.append(_state(), 3600.0)
        raw = _open_any(path, decode_times=False)
        assert float(raw["time"].attrs["forcing_cycle_period"]) == 10800.0
        assert "cycle" in raw["time"].attrs["comment"]

    def test_2d_has_no_vel_w_and_late_vel_aligned(self, out_path):
        from kelp.writers import TrajectoryWriter, load_trajectories
        path, flush = out_path
        st = _state()
        with TrajectoryWriter(path, 3, flush_every=flush) as w:
            w.append(st, 1.0)                                  # no velocity yet
            w.append(st.replace(u=np.ones(3), v=np.full(3, 2.0)), 2.0)
            w.append(st, 3.0)                                  # skipped again
        tr = load_trajectories(path)
        assert "vel_w" not in tr
        assert np.isnan(tr["vel_u"][:, 0]).all()
        np.testing.assert_array_equal(tr["vel_v"][:, 1], [2.0, 2.0, 2.0])
        assert np.isnan(tr["vel_u"][:, 2]).all()
        # through xarray too: the record written before vel_u existed must
        # read NaN, not the netCDF default fill 9.97e36 (spec §1)
        xu = _open_any(path)["vel_u"].values
        assert np.isnan(xu[:, [0, 2]]).all()
        np.testing.assert_array_equal(xu[:, 1], 1.0)

    def test_vel_w_created_independently(self, out_path):
        """3-D: ``vel_w`` appears after ``vel_u/vel_v`` and reads NaN where
        no record carried it."""
        from kelp.writers import TrajectoryWriter, load_trajectories
        path, flush = out_path
        st = _state().replace(u=np.ones(3), v=np.ones(3))
        with TrajectoryWriter(path, 3, flush_every=flush) as w:
            w.append(st, 1.0)
            w.append(st.replace(w=np.full(3, 0.5)), 2.0)
            w.append(st, 3.0)
        tr = load_trajectories(path)
        assert tr["vel_w"].shape == (3, 3)
        assert np.isnan(tr["vel_w"][:, [0, 2]]).all()
        np.testing.assert_array_equal(tr["vel_w"][:, 1], 0.5)
        np.testing.assert_array_equal(tr["vel_u"], np.ones((3, 3)))

    def test_velocity_attrs_and_fill(self, out_path):
        from kelp.writers import TrajectoryWriter
        path, flush = out_path
        st = _state().replace(u=np.ones(3), v=np.ones(3), w=np.ones(3))
        with TrajectoryWriter(path, 3, flush_every=flush) as w:
            w.append(st, 1.0)
        raw = _open_any(path, decode_times=False, decode_coords=False)
        for name in ("vel_u", "vel_v", "vel_w"):
            assert raw[name].attrs["units"] == "m s-1"
            assert "comment" in raw[name].attrs
            assert "coordinates" not in raw[name].attrs
        assert "positive down" in raw["vel_w"].attrs["long_name"]

    def test_sampled_var_skipped_on_a_record_reads_aligned(self, out_path):
        from kelp.writers import TrajectoryWriter, load_trajectories
        path, flush = out_path
        st = _state()
        with TrajectoryWriter(path, 3, flush_every=flush) as w:
            w.append(st, 1.0, sampled={"temp": np.full(3, 1.0)})
            w.append(st, 2.0)
            w.append(st, 3.0, sampled={"temp": np.full(3, 3.0)})
        t = load_trajectories(path)["temp"]
        np.testing.assert_array_equal(t[:, 0], 1.0)
        assert np.isnan(t[:, 1]).all()
        np.testing.assert_array_equal(t[:, 2], 3.0)

    def test_sampled_var_attrs(self, out_path):
        from kelp.writers import TrajectoryWriter
        path, flush = out_path
        with TrajectoryWriter(path, 3, flush_every=flush,
                              sample_var_attrs={"temp": {"units": "degC"}}) as w:
            w.append(_state(), 1.0, sampled={"temp": np.full(3, 1.0)})
        raw = _open_any(path, decode_times=False, decode_coords=False)
        a = raw["temp"].attrs
        assert a["units"] == "degC"
        assert a["coordinates"] == "lat lon z"
        assert "start-of-step" in a["comment"]

    def test_failure_reason_zero_when_absent_after_creation(self, out_path):
        from kelp.writers import TrajectoryWriter, load_trajectories
        path, flush = out_path
        st = _state()
        with TrajectoryWriter(path, 3, flush_every=flush) as w:
            w.append(st.replace(failure_reason=np.array([1, 2, 0], np.int8)),
                     1.0)
            w.append(st, 2.0)                       # no failure_reason
        fr = load_trajectories(path)["failure_reason"]
        np.testing.assert_array_equal(fr[:, 0], [1, 2, 0])
        np.testing.assert_array_equal(fr[:, 1], [0, 0, 0])
        raw = _open_any(path, decode_times=False, decode_coords=False)
        assert raw["failure_reason"].attrs["coordinates"] == "lat lon z"

    def test_release_vars_and_trajectory_ids(self, out_path):
        from kelp.writers import TrajectoryWriter, load_trajectories
        path, flush = out_path
        with TrajectoryWriter(path, 3, flush_every=flush) as w:
            w.set_time_units("seconds since 2001-01-01 00:00:00", "noleap")
            w.set_trajectory_ids(np.arange(10, 13))
            w.write_release(np.array([1.0, 2.0, 3.0]), np.zeros(3),
                            np.zeros(3), np.array([0.0, 3600.0, 7200.0]))
            w.append(_state(), 3600.0)
        tr = load_trajectories(path)
        np.testing.assert_array_equal(tr["trajectory"], [10, 11, 12])
        np.testing.assert_array_equal(tr["release_time"], [0.0, 3600.0, 7200.0])
        np.testing.assert_array_equal(tr["release_lon"], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(tr["release_z"], np.zeros(3))
        ds = _open_any(path)
        assert str(ds["release_time"].values[1]).startswith("2001-01-01 01:00")
        raw = _open_any(path, decode_times=False)
        assert np.isfinite(raw["release_time"]).all()
        assert raw["release_time"].attrs["calendar"] == "noleap"
        assert raw["release_lon"].attrs["units"] == "degrees_east"
        assert "standard_name" not in raw["release_lon"].attrs

    def test_set_time_units_after_release_updates_release_time(self, out_path):
        from kelp.writers import TrajectoryWriter
        path, flush = out_path
        with TrajectoryWriter(path, 3, flush_every=flush) as w:
            w.write_release(np.ones(3), np.ones(3), np.zeros(3), np.zeros(3))
            w.set_time_units("seconds since 2001-01-01 00:00:00", "360_day")
            w.append(_state(), 3600.0)
        raw = _open_any(path, decode_times=False)
        assert raw["release_time"].attrs["units"] == \
            "seconds since 2001-01-01 00:00:00"
        assert raw["release_time"].attrs["calendar"] == "360_day"

    @pytest.mark.parametrize("method", ["write_release", "set_trajectory_ids"])
    def test_setter_length_checks(self, out_path, method):
        from kelp.writers import TrajectoryWriter
        path, flush = out_path
        with TrajectoryWriter(path, 3, flush_every=flush) as w:
            with pytest.raises(ValueError, match="length"):
                if method == "write_release":
                    w.write_release(np.ones(4), np.ones(3), np.zeros(3),
                                    np.zeros(3))
                else:
                    w.set_trajectory_ids(np.arange(2))

    def test_time_attrs_survive_zarr_flushes(self, tmp_path):
        from kelp.writers import TrajectoryWriter
        path = str(tmp_path / "many.zarr")
        w = TrajectoryWriter(path, 3).open()
        w.set_time_units("seconds since 2005-01-01 00:00:00", "standard")
        for k in range(600):                 # several obs-chunk flushes
            w.append(_state(), float(k))
        w.close()
        import zarr
        g = zarr.open_group(path, mode="r")
        assert g["time"].attrs["units"] == "seconds since 2005-01-01 00:00:00"
        assert g["time"].metadata.dimension_names == ("time",)
        assert g["time"].shape == (600,)

    def test_reserved_release_names(self, tmp_path):
        from kelp.writers import TrajectoryWriter
        with TrajectoryWriter(str(tmp_path / "x.nc"), 3) as w:
            with pytest.raises(ValueError, match="reserved"):
                w.append(_state(), 0.0, sampled={"release_time": np.zeros(3)})

    def test_big_store_blocks_with_late_variable(self, tmp_path, monkeypatch):
        from kelp import writers
        monkeypatch.setattr(writers._ZarrBackend, "TRAJ_CHUNK_MAX", 4)
        path = str(tmp_path / "big.zarr")
        st = _state(P=10)
        with writers.TrajectoryWriter(path, 10) as w:
            w.append(st, 1.0)
            w.write_checkpoint(st, 0, 0)        # flush: the next block resizes
            w.append(st.replace(u=np.ones(10), v=np.ones(10)), 2.0)
        tr = writers.load_trajectories(path)
        assert tr["vel_u"].shape == (10, 2)
        assert np.isnan(tr["vel_u"][:, 0]).all()
        np.testing.assert_array_equal(tr["vel_u"][:, 1], np.ones(10))
        import zarr
        g = zarr.open_group(path, mode="r")
        assert g["vel_u"].chunks[0] == 4
        assert g["vel_u"].metadata.dimension_names == ("trajectory", "time")

    def test_format1_file_still_reads(self, tmp_path):
        """A format-1 NetCDF (obs dimension, no epoch) through load_trajectories."""
        import netCDF4
        from kelp.writers import load_trajectories
        p = str(tmp_path / "old.nc")
        with netCDF4.Dataset(p, "w") as ds:
            ds.createDimension("trajectory", 2)
            ds.createDimension("obs", None)
            ds.createVariable("trajectory", "i4", ("trajectory",))[:] = [0, 1]
            ds.createVariable("time", "f8", ("obs",))[:] = [0.0, 1.0]
            for n in ("lon", "lat", "z"):
                ds.createVariable(n, "f8", ("trajectory", "obs"))[:] = np.ones((2, 2))
            ds.createVariable("status", "i4", ("trajectory", "obs"))[:] = np.ones((2, 2))
            ds.createVariable("temp", "f8", ("trajectory", "obs"))[:] = np.ones((2, 2))
        tr = load_trajectories(p)
        assert tr["lon"].shape == (2, 2) and "temp" in tr

    def test_format1_zarr_still_reads(self, tmp_path):
        import zarr
        from kelp.writers import load_trajectories
        p = str(tmp_path / "old.zarr")
        g = zarr.open_group(p, mode="w")
        g.create_array("trajectory", shape=(2,), dtype="i4",
                       dimension_names=("trajectory",))[:] = [0, 1]
        g.create_array("time", shape=(2,), dtype="f8",
                       dimension_names=("obs",))[:] = [0.0, 1.0]
        for n in ("lon", "lat", "z", "vel_u", "vel_v", "vel_w"):
            g.create_array(n, shape=(2, 2), dtype="f8",
                           dimension_names=("trajectory", "obs"))[:] = 1.0
        g.create_array("status", shape=(2, 2), dtype="i4",
                       dimension_names=("trajectory", "obs"))[:] = 1
        tr = load_trajectories(p)
        assert tr["lon"].shape == (2, 2) and tr["vel_w"].shape == (2, 2)
        assert "release_lon" not in tr
