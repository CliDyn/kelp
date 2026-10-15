"""Trajectory output format 2 (release 1.0) — docs/io_and_time.md §5."""
import numpy as np
import pytest
import xarray as xr

cftime = pytest.importorskip("cftime")


class TestTimeAxisEpoch:
    def _ds(self, values, attrs=None):
        return xr.Dataset(coords={"time": ("time", values, attrs or {})})

    def test_datetime64(self):
        from kelp.readers import time_axis_epoch
        t = np.array(["2005-01-01T23:15", "2005-01-02T23:15"],
                     dtype="datetime64[ns]")
        units, cal = time_axis_epoch(self._ds(t))
        assert units == "seconds since 2005-01-01 23:15:00"
        assert cal in ("proleptic_gregorian", "standard")

    def test_datetime64_fraction_kept(self):
        from kelp.readers import time_axis_epoch
        t = np.array(["2005-01-01T00:00:00.5"], dtype="datetime64[ns]")
        assert time_axis_epoch(self._ds(t))[0] == \
            "seconds since 2005-01-01 00:00:00.5"

    @pytest.mark.parametrize("unit", ["D", "s", "ms", "us", "ns"])
    def test_datetime64_every_unit(self, unit):
        from kelp.readers import time_axis_epoch
        t = np.array(["2005-03-01"], dtype=f"datetime64[{unit}]")
        assert time_axis_epoch(self._ds(t))[0] == \
            "seconds since 2005-03-01 00:00:00"

    @pytest.mark.parametrize("cls,cal", [(cftime.DatetimeNoLeap, "noleap"),
                                         (cftime.Datetime360Day, "360_day")])
    def test_cftime(self, cls, cal):
        from kelp.readers import time_axis_epoch
        vals = np.array([cls(2001, 2, 28, 12), cls(2001, 3, 1, 12)])
        units, c = time_axis_epoch(self._ds(vals))
        assert units == "seconds since 2001-02-28 12:00:00"
        assert c == cal

    def test_numeric_since_with_calendar(self):
        from kelp.readers import time_axis_epoch
        ds = xr.Dataset(coords={"time": ("time", np.array([10.0, 11.0]),
                                          {"units": "days since 2000-01-01",
                                           "calendar": "noleap"})})
        assert time_axis_epoch(ds) == ("seconds since 2000-01-11 00:00:00",
                                       "noleap")

    @pytest.mark.parametrize("vals,attrs", [
        (np.array([0.0, 1.0]), {"units": "hours"}),   # bare unit
        (np.arange(3), {}),                           # index axis
    ])
    def test_no_epoch(self, vals, attrs):
        from kelp.readers import time_axis_epoch
        assert time_axis_epoch(self._ds(vals, attrs)) is None

    def test_no_time_dim(self):
        from kelp.readers import time_axis_epoch
        assert time_axis_epoch(xr.Dataset()) is None

    @pytest.mark.parametrize("units_cal", [
        ("seconds since 2005-01-01 23:15:00", "proleptic_gregorian"),
        ("seconds since 2001-02-28 12:00:00", "noleap"),
        ("seconds since 2001-02-28 12:00:00", "360_day"),
    ])
    def test_units_decode_back_in_xarray(self, units_cal):
        units, cal = units_cal
        ds = xr.Dataset(coords={"time": ("time", np.array([0.0, 3600.0]),
                                          {"units": units, "calendar": cal})})
        dec = xr.decode_cf(ds)["time"].values
        assert str(dec[0]).replace("T", " ").startswith(
            units.split("since ")[1])

    def test_netcdf_roundtrip_first_record(self, tmp_path):
        """The epoch of a CF file opened by xarray is its first raw record,
        and the units string decodes back to that instant."""
        from kelp.readers import time_axis_epoch
        p = tmp_path / "t.nc"
        t = np.array(["2005-01-01T23:15", "2005-01-02T23:15"],
                     dtype="datetime64[ns]")
        xr.Dataset({"u": ("time", [1.0, 2.0])},
                   coords={"time": t}).to_netcdf(p)
        units, cal = time_axis_epoch(xr.open_dataset(p))
        back = xr.decode_cf(xr.Dataset(coords={"time": (
            "time", [0.0], {"units": units, "calendar": cal})}))
        assert back["time"].values[0] == t[0]


def _state(P=3):
    from kelp.core.particles import make_state
    return make_state(np.linspace(5.0, 5.2, P), np.full(P, 5.0))


class TestVelocityMask:
    def test_mask_rules(self):
        from kelp.core.particles import ParticleStatus as S
        from kelp.core.release import mask_inactive_velocity
        st = _state(P=4).replace(
            status=np.array([S.ALIVE, S.OUTSIDE, S.BEACHED, S.ALIVE],
                            np.int32),
            u=np.ones(4), v=np.ones(4))
        prev = np.array([S.ALIVE, S.ALIVE, S.ALIVE, S.BEACHED], np.int32)
        out = mask_inactive_velocity(st, prev)
        np.testing.assert_array_equal(np.isnan(out.u),
                                      [False, True, True, True])
        np.testing.assert_array_equal(np.isnan(out.v),
                                      [False, True, True, True])
        assert out.w is None and st.u[1] == 1.0       # copy, not in place

    def test_same_object_when_nothing_to_mask(self):
        from kelp.core.release import mask_inactive_velocity
        st = _state().replace(u=np.ones(3), v=np.ones(3))
        assert mask_inactive_velocity(st, st.status) is st
        bare = _state()
        assert mask_inactive_velocity(bare, bare.status) is bare


class TestTrajectoryRecordFormat2:
    def test_alignment_and_no_fabricated_w(self):
        from kelp.tracking import TrajectoryRecord
        rec = TrajectoryRecord()
        st = _state()
        rec.append(st, 1.0)                                    # no velocity
        rec.append(st.replace(u=np.ones(3), v=np.ones(3)), 2.0)
        rec.append(st, 3.0)                                    # none again
        a = rec.to_arrays()
        assert a["vel_u"].shape == (3, 3)
        assert np.isnan(a["vel_u"][0]).all() and np.isnan(a["vel_v"][2]).all()
        np.testing.assert_array_equal(a["vel_u"][1], 1.0)
        assert "vel_w" not in a
        np.testing.assert_array_equal(a["time"], [1.0, 2.0, 3.0])

    def test_w_lists_aligned_when_w_appears_late(self):
        from kelp.tracking import TrajectoryRecord
        rec = TrajectoryRecord()
        st = _state().replace(u=np.ones(3), v=np.ones(3))
        rec.append(st, 1.0)
        rec.append(st.replace(w=np.full(3, 2.0)), 2.0)
        a = rec.to_arrays()
        assert a["vel_w"].shape == (2, 3)
        assert np.isnan(a["vel_w"][0]).all()

    def test_release(self):
        from kelp.tracking import TrajectoryRecord
        rec = TrajectoryRecord()
        assert rec.release_lon is None
        rec.append(_state(), 1.0)
        rec.set_release(np.ones(3), np.ones(3), np.zeros(3), np.zeros(3))
        a = rec.to_arrays()
        for k in ("release_lon", "release_lat", "release_z", "release_time"):
            assert a[k].shape == (3,)


class TestTimeAxisEpochRobust:
    """Inputs ``extract_time_info`` tolerates must not crash the epoch helper
    (it runs after the writer is open) — review of the plan, 2026-10-13."""

    @pytest.mark.parametrize("attrs", [
        {"units": "days since 0000-01-01", "calendar": "standard"},
        {"units": "days since 2000-13-01"},
        {"units": "days since 2000-01-01", "calendar": "no_such_calendar"},
    ])
    def test_undecodable_numeric_axis_gives_none(self, attrs):
        from kelp.readers import time_axis_epoch
        ds = xr.Dataset(coords={"time": ("time", np.array([1.0, 2.0]), attrs)})
        assert time_axis_epoch(ds) is None

    def test_nat_first_record_gives_none(self):
        from kelp.readers import time_axis_epoch
        t = np.array(["NaT", "2005-01-01"], dtype="datetime64[ns]")
        ds = xr.Dataset(coords={"time": ("time", t)})
        assert time_axis_epoch(ds) is None

    @pytest.mark.parametrize("unit,text", [("M", "2005-03-01 00:00:00"),
                                           ("Y", "2005-01-01 00:00:00"),
                                           ("h", "2005-03-01 06:00:00"),
                                           ("m", "2005-03-01 06:30:00")])
    def test_coarse_units(self, unit, text):
        from kelp.readers import time_axis_epoch
        src = {"M": "2005-03", "Y": "2005", "h": "2005-03-01T06",
               "m": "2005-03-01T06:30"}[unit]
        t = np.array([src], dtype=f"datetime64[{unit}]")
        ds = xr.Dataset(coords={"time": ("time", t)})
        assert time_axis_epoch(ds)[0] == f"seconds since {text}"


class TestParticleStateRelease:
    def test_release_fields_ride_replace_and_snapshot(self):
        st = _state().replace(release_lon=np.ones(3), release_lat=np.ones(3),
                              release_z=np.zeros(3))
        moved = st.replace(lon=st.lon + 1.0)
        np.testing.assert_array_equal(moved.release_lon, np.ones(3))
        snap = moved.snapshot()
        assert "release_lon" in snap and snap["release_lon"] is not st.release_lon

    def test_subset_slices_every_per_particle_array(self):
        P = 5
        st = _state(P=P).replace(
            u=np.arange(P, dtype=float), v=np.arange(P, dtype=float),
            w=np.arange(P, dtype=float),
            failure_reason=np.arange(P, dtype=np.int8),
            release_step=np.arange(P, dtype=np.int32),
            elem=np.arange(P, dtype=np.int32),
            release_lon=np.arange(P, dtype=float),
            release_lat=np.arange(P, dtype=float),
            release_z=np.arange(P, dtype=float))
        sub = st.subset(np.array([1, 3]))
        for name in ("lon", "lat", "z", "status", "nearest_node", "u", "v",
                     "w", "failure_reason", "release_step", "elem",
                     "release_lon", "release_lat", "release_z"):
            np.testing.assert_array_equal(getattr(sub, name),
                                          getattr(st, name)[[1, 3]])
        assert _state().subset(slice(0, 2)).release_lon is None


# ---------------------------------------------------------------------------
# run_single_realization end to end (Task 5)
# ---------------------------------------------------------------------------

def _env(tmp_path, n_time=8, calendar=None, eta_ramp=False):
    from types import SimpleNamespace
    from kelp.readers import get_reader
    from kelp.core.config import ModeConfig
    from tests.test_tracking import _make_fesomc_3x3_with_time
    nc = _make_fesomc_3x3_with_time(tmp_path, n_time=n_time, dt_seconds=3600.0,
                                    u_val=1.0, v_val=0.0)
    if calendar or eta_ramp:
        d = xr.open_dataset(nc).load()
        if calendar:   # the axis as cftime from 2001-02-28 00:00
            cls = {"noleap": cftime.DatetimeNoLeap,
                   "360_day": cftime.Datetime360Day}[calendar]
            d = d.assign_coords(time=[cls(2001, 2, 28, h)
                                      for h in range(n_time)])
        if eta_ramp:   # eta = record index, to see WHICH record is sampled
            d["eta"] = d["eta"] * 0.0 + np.arange(n_time)[:, None]
        d.close()
        nc = tmp_path / "fesomc_variant.nc"
        d.to_netcdf(nc)
    reader = get_reader("fesomc")
    mesh = reader.load_mesh(SimpleNamespace(meshpath=str(nc)))
    ds = reader.open_dataset(str(nc))
    return mesh, ds, reader, ModeConfig(Fdrift2D=True, Fdrift3D=False,
                                        time_dir=1.0)


def _run(tmp_path, name, *, state=None, n_steps=4, start=0, start_step_i=0,
         temporal_interp=False, time_dir=1.0, on_exhausted="stop", n_time=8,
         checkpoint_freq=0, calendar=None, ext=".nc", sim_time_start=None,
         sample_vars=None, eta_ramp=False, writer=True):
    """One run_single_realization call on the 3x3 FESOM-C mesh (hourly
    datetime64 axis from 2020-01-01, u = 1 m/s); returns the output path,
    or the TrajectoryRecord when ``writer=False``."""
    from kelp.core.config import NumericsConfig
    from kelp.readers import extract_time_info
    from kelp.tracking import run_single_realization
    from kelp.writers import TrajectoryWriter
    sub = tmp_path / name
    sub.mkdir()
    mesh, ds, reader, mode = _env(sub, n_time=n_time, calendar=calendar,
                                  eta_ramp=eta_ramp)
    mode.time_dir = time_dir
    ft, fdt = extract_time_info(ds)
    st = state if state is not None else _state()
    kw = (dict(temporal_interp=True, file_times=ft, file_dt=fdt)
          if temporal_interp else {})
    if sim_time_start is not None:
        kw["sim_time_start"] = sim_time_start
    if sample_vars:
        kw.update(sample_vars=list(sample_vars),
                  output_sample_vars=list(sample_vars))
    path = str(sub / f"out{ext}")
    w = TrajectoryWriter(path, st.n_particles).open() if writer else None
    rec = run_single_realization(
        mesh, ds, reader, st, mode,
        NumericsConfig(dt=3600.0, integrator="euler"),
        start_time_idx=start, n_steps=n_steps, n_time_total=ds.sizes["time"],
        writer=w, on_exhausted=on_exhausted, start_step_i=start_step_i,
        checkpoint_freq=checkpoint_freq, **kw)
    if w is not None:
        w.close()
        return path
    return rec


class TestRecordClockInMemory:
    """Labels and release arrays through TrajectoryRecord (no writer)."""

    def test_discrete_equals_interpolated_seconds(self, tmp_path):
        a = _run(tmp_path, "d", writer=False).to_arrays()["time"]
        b = _run(tmp_path, "i", writer=False,
                 temporal_interp=True).to_arrays()["time"]
        np.testing.assert_array_equal(a, [3600.0, 7200.0, 10800.0, 14400.0])
        np.testing.assert_array_equal(a, b)

    def test_backward_and_cycle(self, tmp_path):
        t = _run(tmp_path, "b", start=5, time_dir=-1.0,
                 writer=False).to_arrays()["time"]
        np.testing.assert_array_equal(t, [14400.0, 10800.0, 7200.0, 3600.0])
        t = _run(tmp_path, "c", n_steps=5, n_time=3, on_exhausted="cycle",
                 writer=False).to_arrays()["time"]
        np.testing.assert_array_equal(t, 3600.0 * np.arange(1, 6))

    def test_release_in_record(self, tmp_path):
        from kelp.core.release import build_release_state
        st = build_release_state(np.array([5.0, 5.1]), np.array([5.0, 5.0]),
                                 np.zeros(2), np.array([0.0, 7200.0]),
                                 n_pulses=1, pulse_interval=0.0, dt=3600.0,
                                 start_step_i=0)
        a = _run(tmp_path, "dl", state=st, start=1, writer=False).to_arrays()
        np.testing.assert_array_equal(a["release_time"], [3600.0, 10800.0])
        np.testing.assert_array_equal(a["release_lon"], [5.0, 5.1])
        assert np.isnan(a["lon"][0, 1]) and np.isnan(a["vel_u"][0, 1])
        assert "vel_w" not in a

    def test_discrete_sampling_reads_the_end_of_step_record(self, tmp_path):
        a = _run(tmp_path, "s", start=2, n_steps=3, n_time=6, eta_ramp=True,
                 sample_vars=["eta"], writer=False).to_arrays()
        # record after step g holds the field of record start + g + 1
        np.testing.assert_allclose(a["eta"][:, 0], [3.0, 4.0, 5.0])

    def test_discrete_sampling_last_step_without_next_record(self, tmp_path):
        a = _run(tmp_path, "s2", start=3, n_steps=3, n_time=6, eta_ramp=True,
                 sample_vars=["eta"], writer=False).to_arrays()
        # steps read records 3, 4, 5; record 6 does not exist → start field
        np.testing.assert_allclose(a["eta"][:, 0], [4.0, 5.0, 5.0])


EXTS = [".nc", ".zarr"]


def _open(path, **kw):
    return (xr.open_zarr(path, consolidated=False, **kw)
            if path.endswith(".zarr") else xr.open_dataset(path, **kw))


@pytest.mark.parametrize("ext", EXTS)
class TestRecordClockInFile:
    def test_labels_and_dates(self, tmp_path, ext):
        from kelp.writers import load_trajectories
        p = _run(tmp_path, "e", start=2, ext=ext)
        np.testing.assert_array_equal(load_trajectories(p)["time"],
                                      3600.0 * np.arange(3, 7))
        ds = _open(p)
        assert str(ds["time"].values[0]).startswith("2020-01-01T03:00")
        assert ds.sel(time="2020-01-01T04:00")["lon"].shape == (3,)
        assert int(ds.attrs["kelp_output_format"]) == 2

    def test_cycle_period_attribute(self, tmp_path, ext):
        p = _run(tmp_path, "c", n_steps=5, n_time=3, on_exhausted="cycle",
                 ext=ext)
        attrs = _open(p, decode_times=False)["time"].attrs
        assert float(attrs["forcing_cycle_period"]) == 3 * 3600.0

    def test_chain_labels_and_release_equal_uncut_run(self, tmp_path, ext):
        from kelp.writers import load_trajectories, load_restart
        whole = load_trajectories(_run(tmp_path, "w", n_steps=6, ext=ext,
                                       temporal_interp=True))
        p1 = _run(tmp_path, "s1", n_steps=2, temporal_interp=True,
                  checkpoint_freq=2, ext=ext)
        r = load_restart(p1)
        p2 = _run(tmp_path, "s2", n_steps=4, temporal_interp=True, ext=ext,
                  state=r["state"], start_step_i=r["step_i"] + 1,
                  sim_time_start=(r["step_i"] + 1) * 3600.0)
        a, b = load_trajectories(p1), load_trajectories(p2)
        np.testing.assert_array_equal(
            np.concatenate([a["time"], b["time"]]), whole["time"])
        np.testing.assert_array_equal(
            np.concatenate([a["lon"], b["lon"]], axis=1), whole["lon"])
        for k in ("release_lon", "release_lat", "release_z", "release_time"):
            np.testing.assert_array_equal(b[k], whole[k])

    @pytest.mark.parametrize("calendar", ["noleap", "360_day"])
    def test_cftime_forcing(self, tmp_path, ext, calendar):
        p = _run(tmp_path, "cf", calendar=calendar, ext=ext)
        raw = _open(p, decode_times=False)
        assert raw["time"].attrs["calendar"] == calendar
        assert np.isfinite(raw["release_time"].values).all()
        assert str(_open(p)["time"].values[0]).startswith("2001-02-28 01:00")


@pytest.mark.parametrize("ext", EXTS)
class TestReleaseInFile:
    def test_delayed_release(self, tmp_path, ext):
        from kelp.core.release import build_release_state
        from kelp.writers import load_trajectories
        st = build_release_state(np.array([5.0, 5.1]), np.array([5.0, 5.0]),
                                 np.zeros(2), np.array([0.0, 7200.0]),
                                 n_pulses=1, pulse_interval=0.0, dt=3600.0,
                                 start_step_i=0)
        tr = load_trajectories(_run(tmp_path, "dl", state=st, start=1,
                                    ext=ext))
        np.testing.assert_array_equal(tr["release_time"], [3600.0, 10800.0])
        np.testing.assert_array_equal(tr["release_lon"], [5.0, 5.1])
        assert np.isnan(tr["lon"][1, 0]) and np.isnan(tr["vel_u"][1, 0])

    def test_pulses(self, tmp_path, ext):
        from kelp.core.release import build_release_state
        from kelp.writers import load_trajectories
        st = build_release_state(np.array([5.0, 5.1]), np.array([5.0, 5.0]),
                                 np.zeros(2), None, n_pulses=2,
                                 pulse_interval=3600.0, dt=3600.0,
                                 start_step_i=0)
        tr = load_trajectories(_run(tmp_path, "pl", state=st, ext=ext))
        np.testing.assert_array_equal(tr["release_time"],
                                      [0.0, 0.0, 3600.0, 3600.0])

    def test_all_dormant_at_first_record(self, tmp_path, ext):
        from kelp.core.release import build_release_state
        from kelp.writers import load_trajectories
        st = build_release_state(np.array([5.0, 5.1]), np.array([5.0, 5.0]),
                                 np.zeros(2), np.array([7200.0, 7200.0]),
                                 n_pulses=1, pulse_interval=0.0, dt=3600.0,
                                 start_step_i=0)
        tr = load_trajectories(_run(tmp_path, "ad", state=st, ext=ext))
        assert tr["vel_u"].shape == (2, 4)
        assert np.isnan(tr["vel_u"][:, :2]).all()
        assert np.isfinite(tr["vel_u"][:, 2:]).all()

    def test_pre_format2_restart(self, tmp_path, ext):
        from kelp.writers import load_trajectories
        with pytest.warns(RuntimeWarning, match="release"):
            p = _run(tmp_path, "old", start=2, start_step_i=2, ext=ext)
        tr = load_trajectories(p)
        assert np.isnan(tr["release_lon"]).all()
        np.testing.assert_array_equal(tr["release_time"], np.zeros(3))
        assert np.isfinite(_open(p, decode_times=False)["release_time"]
                           .values).all()


@pytest.mark.parametrize("ext", EXTS)
def test_dying_record_nan_velocity_no_vel_w(tmp_path, ext):
    from kelp.writers import load_trajectories
    st = _state(P=2).replace(lon=np.array([5.0, 5.99]))
    tr = load_trajectories(_run(tmp_path, "die", state=st, ext=ext))
    assert "vel_w" not in tr
    out = tr["status"][1] == 3
    assert out.any() and np.isnan(tr["vel_u"][1][out]).all()
    assert np.isfinite(tr["vel_u"][0]).all()


def test_chain_recipe_of_the_analysis_skill(tmp_path):
    """The kelp-analysis chain merge on format-2 Zarr segments."""
    from kelp.writers import load_restart, load_trajectories
    whole = load_trajectories(_run(tmp_path, "w", n_steps=6, ext=".zarr"))
    p1 = _run(tmp_path, "a", n_steps=2, checkpoint_freq=2, ext=".zarr")
    r = load_restart(p1)
    p2 = _run(tmp_path, "b", n_steps=4, ext=".zarr", state=r["state"],
              start=r["step_i"] + 1, start_step_i=r["step_i"] + 1)
    parts = [xr.open_zarr(p, consolidated=False) for p in (p1, p2)]
    ds = xr.concat(parts, dim="time", data_vars="minimal", coords="minimal",
                   compat="override", join="override")
    np.testing.assert_array_equal(ds["lon"].values, whole["lon"])
    assert ds["release_lon"].dims == ("trajectory",)
    np.testing.assert_array_equal(ds["release_lon"].values,
                                  whole["release_lon"])


def test_rotated_mesh_release_is_geographic(tmp_path):
    from kelp.core.config import MeshConfig, ModeConfig
    from kelp.readers import get_reader
    from kelp.tracking import track_particles
    from tests.test_rotated_frame import _midlat_fesom2_files
    mp, dp, _, _ = _midlat_fesom2_files(tmp_path, tag="rot", lat_centre=5.0,
                                        lon_centre=0.0)
    reader = get_reader("fesom2")
    mesh = reader.load_mesh(MeshConfig(type="fesom2", meshpath=str(mp),
                                       rotation="fesom2"))
    ds = reader.open_dataset(str(dp))
    rec = track_particles(mesh, ds, reader, np.array([0.0, 0.2]),
                          np.array([5.0, 5.1]), start_time_idx=0, n_steps=2,
                          dt=3600.0, integrator="rk2",
                          mode=ModeConfig(Fdrift2D=True, Fdrift3D=False,
                                          time_dir=1.0))
    a = rec.to_arrays()
    np.testing.assert_allclose(a["release_lon"], [0.0, 0.2], atol=1e-9)
    np.testing.assert_allclose(a["release_lat"], [5.0, 5.1], atol=1e-9)


def test_start_date_through_run_tracking(tmp_path):
    from kelp.core.config import load_config
    from kelp.tracking import run_tracking
    from tests.test_tracking import _make_fesomc_3x3_with_time
    nc = _make_fesomc_3x3_with_time(tmp_path, n_time=8, dt_seconds=3600.0)
    seeds = tmp_path / "seeds.dat"
    seeds.write_text("5.0 5.0\n5.1 5.0\n")
    out = tmp_path / "sd.nc"
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"""\
mode:
  Fdrift2D: true
  Fdrift3D: false
  time_dir: 1
numerics:
  integrator: euler
  dt: 3600
  temporal_interp: true
mesh:
  type: fesomc
  meshpath: {nc}
data:
  ncinput: {nc}
  inicloud: {seeds}
  time_averaging: instant
time:
  start_date: "2020-01-01T02:30:00"
  duration_days: 0.125
""")
    run_tracking(load_config(str(cfg)), output_path=str(out))
    ds = xr.open_dataset(out)
    assert str(ds["time"].values[0]).startswith("2020-01-01T03:30")
    np.testing.assert_array_equal(
        xr.open_dataset(out, decode_times=False)["release_time"].values,
        [9000.0, 9000.0])


# ---------------------------------------------------------------------------
# utils/concat_trajectories.py (Task 6)
# ---------------------------------------------------------------------------

def _concat():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
    from concat_trajectories import concat_trajectories
    return concat_trajectories


class TestConcatFormat2:
    def _shards(self, tmp_path, units=("seconds since 2001-01-01 00:00:00",) * 2,
                offset0=0):
        from kelp.writers import TrajectoryWriter
        out = []
        for k, u in enumerate(units):
            p = str(tmp_path / f"shard{k}.zarr")
            with TrajectoryWriter(p, 2,
                                  metadata={"particle_offset": offset0 + 2 * k,
                                            "random_draws": 0}) as w:
                w.set_time_units(u, "standard")
                w.set_trajectory_ids(offset0 + 2 * k + np.arange(2))
                w.write_release(np.full(2, float(k)), np.zeros(2),
                                np.zeros(2), np.zeros(2))
                w.append(_state(P=2).replace(u=np.ones(2), v=np.ones(2)),
                         3600.0)
            out.append(p)
        return out

    def test_merge_keeps_release_time_attrs_and_fill(self, tmp_path):
        import zarr
        m = _concat()(self._shards(tmp_path), tmp_path / "m.zarr")
        ds = xr.open_zarr(str(m), consolidated=False)
        np.testing.assert_array_equal(ds["release_lon"].values, [0, 0, 1, 1])
        assert ds["lon"].dims == ("trajectory", "time")
        assert str(ds["time"].values[0]).startswith("2001-01-01T01:00")
        assert np.isnan(zarr.open_group(str(m), mode="r")["lon"].fill_value)

    def test_global_ids_from_the_first_offset(self, tmp_path):
        m = _concat()(self._shards(tmp_path, offset0=100), tmp_path / "m.zarr")
        ids = xr.open_zarr(str(m), consolidated=False)["trajectory"].values
        np.testing.assert_array_equal(ids, [100, 101, 102, 103])

    def test_differing_time_units_refused(self, tmp_path):
        s = self._shards(tmp_path, units=("seconds since 2001-01-01 00:00:00",
                                          "seconds since 2002-01-01 00:00:00"))
        with pytest.raises(ValueError, match="time units"):
            _concat()(s, tmp_path / "m.zarr")

    def test_format1_next_to_format2_refused(self, tmp_path):
        s = self._shards(tmp_path)
        old = _format1_zarr_shard(tmp_path / "old1.zarr", dormant=False,
                                  times=(3600.0,))   # same values, old layout
        with pytest.raises(ValueError,
                           match="record dimension|kelp_output_format"):
            _concat()([s[0], old], tmp_path / "m.zarr", check_order=False)


def _format1_zarr_shard(path, *, dormant, n=2, times=(0.0, 1.0),
                        with_vel=True):
    """A Zarr store laid out like a format-1 writer output (dimension obs,
    no epoch, no kelp_output_format, 2-D zero vel_w)."""
    import zarr
    T = len(times)
    g = zarr.open_group(str(path), mode="w")
    g.attrs.update({"mode_3d": 0, "random_draws": 0})
    g.create_array("trajectory", shape=(n,), dtype="i4",
                   dimension_names=("trajectory",))[:] = np.arange(n)
    g.create_array("time", shape=(T,), dtype="f8",
                   dimension_names=("obs",))[:] = np.asarray(times)
    for name in ("lon", "lat", "z"):
        g.create_array(name, shape=(n, T), dtype="f8",
                       dimension_names=("trajectory", "obs"))[:] = 1.0
    g.create_array("status", shape=(n, T), dtype="i4",
                   dimension_names=("trajectory", "obs"))[:] = 4 if dormant else 1
    if with_vel and not dormant:
        for name in ("vel_u", "vel_v", "vel_w"):
            g.create_array(name, shape=(n, T), dtype="f8",
                           dimension_names=("trajectory", "obs"))[:] = (
                0.0 if name == "vel_w" else 2.0)
    return str(path)


def test_format1_shards_still_merge_with_legacy_vel_w(tmp_path):
    a = _format1_zarr_shard(tmp_path / "s0.zarr", dormant=False)
    b = _format1_zarr_shard(tmp_path / "s1.zarr", dormant=True)
    m = _concat()([a, b], tmp_path / "m.zarr")
    ds = xr.open_zarr(str(m), consolidated=False)
    assert ds["lon"].dims == ("trajectory", "obs")
    np.testing.assert_array_equal(ds["vel_w"].values[2:], 0.0)  # legacy 0
    assert np.isnan(ds["vel_u"].values[2:]).all()


@pytest.mark.parametrize("integrator", ["euler", "rk2", "rk4"])
def test_backward_run_records_the_physical_velocity(tmp_path, integrator):
    """Output format 2 writes the flow velocity, not the time-reversed
    tracking velocity: an eastward 1 m/s current reads +1 in a backward run
    (owner's decision, 2026-10-13)."""
    from kelp.core.config import NumericsConfig
    from kelp.tracking import run_single_realization
    from kelp.writers import TrajectoryWriter, load_trajectories
    for td, start in ((1.0, 0), (-1.0, 5)):
        sub = tmp_path / f"{integrator}_{int(td)}"
        sub.mkdir()
        mesh, ds, reader, mode = _env(sub)
        mode.time_dir = td
        path = str(sub / "o.nc")
        w = TrajectoryWriter(path, 3).open()
        rec = run_single_realization(
            mesh, ds, reader, _state(), mode,
            NumericsConfig(dt=3600.0, integrator=integrator),
            start_time_idx=start, n_steps=2, n_time_total=ds.sizes["time"],
            writer=w)
        w.close()
        np.testing.assert_allclose(load_trajectories(path)["vel_u"], 1.0)
        rec = run_single_realization(
            mesh, ds, reader, _state(), mode,
            NumericsConfig(dt=3600.0, integrator=integrator),
            start_time_idx=start, n_steps=2, n_time_total=ds.sizes["time"])
        np.testing.assert_allclose(rec.to_arrays()["vel_u"], 1.0)


# ---------------------------------------------------------------------------
# Code review round 1 (2026-10-13)
# ---------------------------------------------------------------------------

def _env_spacing(tmp_path, n_time, spacing_ms):
    """3x3 FESOM-C mesh whose records are ``spacing_ms`` milliseconds apart."""
    from types import SimpleNamespace
    import pandas as pd
    from kelp.readers import get_reader
    from kelp.core.config import ModeConfig
    from tests.test_tracking import _make_fesomc_3x3_with_time
    nc = _make_fesomc_3x3_with_time(tmp_path, n_time=n_time,
                                    dt_seconds=3600.0, u_val=1.0, v_val=0.0)
    d = xr.open_dataset(nc).load()
    d = d.assign_coords(time=pd.date_range("2020-01-01", periods=n_time,
                                           freq=f"{int(spacing_ms)}ms"))
    d.close()
    nc2 = tmp_path / "spaced.nc"
    d.to_netcdf(nc2)
    reader = get_reader("fesomc")
    mesh = reader.load_mesh(SimpleNamespace(meshpath=str(nc2)))
    return mesh, reader.open_dataset(str(nc2)), reader, ModeConfig(
        Fdrift2D=True, Fdrift3D=False, time_dir=1.0)


@pytest.mark.parametrize("n_time,spacing_ms,on_exhausted", [
    (8, 3_600_500, "stop"),     # spacing 0.5 s off dt: tolerated, drifts
    (1, 3_600_000, "cycle"),    # a single cycled record
])
def test_discrete_labels_do_not_depend_on_chain_cuts(tmp_path, n_time,
                                                     spacing_ms, on_exhausted):
    """Labels anchor on the record where global step 0 started, not on the
    resumed segment's own start record (review 2026-10-13)."""
    from kelp.core.config import NumericsConfig
    from kelp.readers import extract_time_info
    from kelp.tracking import run_single_realization
    num = NumericsConfig(dt=3600.0, integrator="euler")

    def run(sub, state, start, step0, n):
        mesh, ds, reader, mode = _env_spacing(sub, n_time, spacing_ms)
        ft, _ = extract_time_info(ds)
        return run_single_realization(
            mesh, ds, reader, state, mode, num, start_time_idx=start,
            n_steps=n, n_time_total=n_time, file_times=ft,
            on_exhausted=on_exhausted, start_step_i=step0)

    (tmp_path / "w").mkdir()
    whole = run(tmp_path / "w", _state(), 0, 0, 6).to_arrays()
    (tmp_path / "a").mkdir()
    a = run(tmp_path / "a", _state(), 0, 0, 2)
    (tmp_path / "b").mkdir()
    st = _state().replace(lon=a.lons[-1], lat=a.lats[-1],
                          release_lon=a.release_lon, release_lat=a.release_lat,
                          release_z=a.release_z)
    b = run(tmp_path / "b", st, 2, 2, 4)      # unwrapped start index 0 + 2
    t = np.concatenate([a.to_arrays()["time"], b.to_arrays()["time"]])
    np.testing.assert_array_equal(t, whole["time"])
    np.testing.assert_array_equal(b.to_arrays()["release_time"],
                                  whole["release_time"])


@pytest.mark.parametrize("ext,flush", [(".nc", 0), (".nc", 2), (".zarr", 0)])
def test_late_failure_reason_reads_none_for_earlier_records(tmp_path, ext,
                                                            flush):
    from kelp.writers import TrajectoryWriter, load_trajectories
    path = str(tmp_path / f"fr{ext}")
    st = _state()
    with TrajectoryWriter(path, 3, flush_every=flush) as w:
        w.append(st, 1.0)
        w.append(st.replace(failure_reason=np.array([0, 2, 1], np.int8)), 2.0)
    fr = load_trajectories(path)["failure_reason"]
    np.testing.assert_array_equal(fr[:, 0], 0)
    np.testing.assert_array_equal(fr[:, 1], [0, 2, 1])


@pytest.mark.parametrize("ext,flush", [(".nc", 0), (".nc", 2), (".zarr", 0)])
def test_append_and_checkpoint_refuse_wrong_lengths(tmp_path, ext, flush):
    from kelp.writers import TrajectoryWriter, load_restart
    path = str(tmp_path / f"len{ext}")
    good = _state()
    w = TrajectoryWriter(path, 3, flush_every=flush).open()
    w.append(good, 1.0)
    w.write_checkpoint(good, 0, 0)
    for bad in (good.replace(lon=np.zeros(5)),
                good.replace(u=np.zeros(2), v=np.zeros(2)),
                good.replace(failure_reason=np.zeros(4, np.int8))):
        with pytest.raises(ValueError, match="length"):
            w.append(bad, 2.0)
    with pytest.raises(ValueError, match="length"):
        w.append(good, 2.0, sampled={"temp": np.zeros(7)})
    with pytest.raises(ValueError, match="length"):
        w.write_checkpoint(good.replace(nearest_node=np.zeros(9, np.int32)),
                           1, 1)
    with pytest.raises(ValueError, match="length"):
        w.write_checkpoint(good, 1, 1, wet_streak=np.zeros(2, np.int32))
    w.close()
    r = load_restart(path)          # the previous checkpoint is intact
    assert r["step_i"] == 0


@pytest.mark.parametrize("ext", [".nc", ".zarr"])
def test_checkpoint_rewrite_drops_absent_optional_fields(tmp_path, ext):
    from kelp.writers import TrajectoryWriter, load_restart
    path = str(tmp_path / f"opt{ext}")
    full = _state().replace(failure_reason=np.ones(3, np.int8),
                            elem=np.arange(3, dtype=np.int32),
                            release_step=np.arange(3, dtype=np.int32))
    bare = _state()
    with TrajectoryWriter(path, 3) as w:
        w.write_checkpoint(full, 0, 0, wet_streak=np.ones(3, np.int32),
                           rotation_angles=(50.0, 15.0, -90.0))
        w.write_checkpoint(bare, 1, 1)
    r = load_restart(path)
    assert r["step_i"] == 1
    assert r["state"].failure_reason is None and r["state"].elem is None
    assert r["release_step"] is None and r["wet_streak"] is None
    assert r["rotation_angles"] is None


def test_trajectory_ids_beyond_int32_refused(tmp_path):
    from kelp.writers import TrajectoryWriter
    with TrajectoryWriter(str(tmp_path / "ids.nc"), 3) as w:
        with pytest.raises(ValueError, match="int32"):
            w.set_trajectory_ids(np.int64(2**31 - 2) + np.arange(3))
        with pytest.raises(ValueError, match="integ"):
            w.set_trajectory_ids(np.array([0.5, 1.0, 2.0]))


# ---------------------------------------------------------------------------
# Code review, round 1 (2026-10-13)
# ---------------------------------------------------------------------------

def test_discrete_cycle_resume_from_the_checkpoint_index(tmp_path):
    """A checkpoint stores the UNWRAPPED record index, so a cycle-mode resume
    from ``file_idx + 1`` labels like the uncut run."""
    from kelp.writers import load_restart, load_trajectories
    whole = load_trajectories(_run(tmp_path, "w", n_steps=7, n_time=3,
                                   on_exhausted="cycle"))
    p1 = _run(tmp_path, "a", n_steps=4, n_time=3, on_exhausted="cycle",
              checkpoint_freq=4)
    r = load_restart(p1)
    assert r["file_idx"] == 3                    # unwrapped (3 % 3 == 0)
    p2 = _run(tmp_path, "b", n_steps=3, n_time=3, on_exhausted="cycle",
              state=r["state"], start=r["file_idx"] + 1,
              start_step_i=r["step_i"] + 1)
    a, b = load_trajectories(p1), load_trajectories(p2)
    np.testing.assert_array_equal(np.concatenate([a["time"], b["time"]]),
                                  whole["time"])
    np.testing.assert_array_equal(b["release_time"], whole["release_time"])
    np.testing.assert_array_equal(np.concatenate([a["lon"], b["lon"]], 1),
                                  whole["lon"])


def test_epoch_kept_when_the_run_uses_part_of_the_axis(tmp_path):
    """A direct caller may pass n_time_total below the axis length."""
    from kelp.core.config import NumericsConfig
    from kelp.readers import extract_time_info
    from kelp.tracking import run_single_realization
    from kelp.writers import TrajectoryWriter
    mesh, ds, reader, mode = _env(tmp_path, n_time=8)
    ft, _ = extract_time_info(ds)
    p = str(tmp_path / "part.nc")
    with TrajectoryWriter(p, 3) as w:
        run_single_realization(mesh, ds, reader, _state(), mode,
                               NumericsConfig(dt=3600.0, integrator="euler"),
                               start_time_idx=0, n_steps=3, n_time_total=5,
                               file_times=ft, writer=w)
    units = xr.open_dataset(p, decode_times=False)["time"].attrs["units"]
    assert units == "seconds since 2020-01-01 00:00:00"


def test_write_release_refuses_non_finite_times(tmp_path):
    from kelp.writers import TrajectoryWriter
    with TrajectoryWriter(str(tmp_path / "r.nc"), 3) as w:
        w.set_time_units("seconds since 2001-01-01 00:00:00", "noleap")
        with pytest.raises(ValueError, match="finite"):
            w.write_release(np.zeros(3), np.zeros(3), np.zeros(3),
                            np.array([0.0, np.nan, 3600.0]))


def test_backward_delayed_release_time(tmp_path):
    from kelp.core.release import build_release_state
    st = build_release_state(np.array([5.0, 5.1]), np.array([5.0, 5.0]),
                             np.zeros(2), np.array([0.0, 7200.0]),
                             n_pulses=1, pulse_interval=0.0, dt=3600.0,
                             start_step_i=0)
    a = _run(tmp_path, "bd", state=st, start=6, time_dir=-1.0,
             writer=False).to_arrays()
    # start record 6 = 21600 s; the second particle starts 2 steps later,
    # i.e. 7200 s EARLIER in a backward run
    np.testing.assert_array_equal(a["release_time"], [21600.0, 14400.0])
    np.testing.assert_array_equal(a["time"], [18000.0, 14400.0, 10800.0,
                                              7200.0])


def test_release_position_is_the_folded_seed(tmp_path):
    """Release positions are taken after the init periodic fold."""
    from kelp.core.config import NumericsConfig
    from kelp.tracking import run_single_realization
    mesh, ds, reader, mode = _env(tmp_path)
    num = NumericsConfig(dt=3600.0, integrator="euler",
                         periodic_lon=(4.5, 5.5))
    st = _state(P=2).replace(lon=np.array([5.0, 5.75]))   # 5.75 -> 4.75
    rec = run_single_realization(mesh, ds, reader, st, mode, num,
                                 start_time_idx=0, n_steps=1,
                                 n_time_total=ds.sizes["time"])
    np.testing.assert_allclose(rec.release_lon, [5.0, 4.75])


@pytest.mark.parametrize("ext", [".nc", ".zarr"])
def test_first_checkpoint_killed_while_creating_the_block(tmp_path, ext,
                                                        monkeypatch):
    """The validity flag is created (as 0) BEFORE the payload variables: a
    kill while the block is being created leaves an invalid checkpoint."""
    import zarr
    from kelp.writers import TrajectoryWriter, load_restart
    from tests.test_output_format_writer import netcdf_fails_at
    path = str(tmp_path / f"blk{ext}")
    orig_z = zarr.Group.create_array

    def z_create(self, name, *a, **k):
        if name == "restart_lon":
            raise RuntimeError("killed")
        return orig_z(self, name, *a, **k)

    if ext == ".nc":
        netcdf_fails_at(monkeypatch, create="restart_lon")
    w = TrajectoryWriter(path, 3).open()
    w.append(_state(), 0.0)
    if ext == ".zarr":
        monkeypatch.setattr(zarr.Group, "create_array", z_create)
    with pytest.raises(RuntimeError, match="killed"):
        w.write_checkpoint(_state(), 0, 0)
    monkeypatch.undo()
    w.close()
    with pytest.raises(ValueError, match="No valid restart"):
        load_restart(path)


@pytest.mark.parametrize("integrator", ["euler", "rk2"])
def test_backward_3d_vel_w_is_the_flow_velocity(tmp_path, integrator):
    """vel_w keeps the flow sign in a backward 3-D run (z-level)."""
    from types import SimpleNamespace
    from kelp.core.config import ModeConfig, NumericsConfig
    from kelp.readers import get_reader
    from kelp.tracking import run_single_realization
    from kelp.core.particles import make_state
    from tests.test_tracking import _make_fesom2_3x3_3d
    out = {}
    for td, start in ((1.0, 0), (-1.0, 2)):
        sub = tmp_path / f"{integrator}{int(td)}"
        sub.mkdir()
        res = _make_fesom2_3x3_3d(sub, n_time=3, u_val=0.5, v_val=0.0,
                                  w_val=0.005)
        mesh_path, data_path = res[0], res[1]
        reader = get_reader("fesom2")
        mesh = reader.load_mesh(SimpleNamespace(meshpath=str(mesh_path),
                                                rotation=None, r_earth=None))
        mode = ModeConfig(Fdrift2D=False, Fdrift3D=True, time_dir=td)
        ds = reader.open_dataset(str(data_path), mode_cfg=mode)
        rec = run_single_realization(
            mesh, ds, reader, make_state(np.array([5.0]), np.array([5.0]),
                                         np.array([20.0])), mode,
            NumericsConfig(dt=3600.0, integrator=integrator),
            start_time_idx=start, n_steps=1, n_time_total=3)
        out[td] = rec.to_arrays()["vel_w"][0, 0]
    assert np.sign(out[1.0]) == np.sign(out[-1.0]) != 0


# ---------------------------------------------------------------------------
# Code review, round 2 (2026-10-13)
# ---------------------------------------------------------------------------

def test_cycle_clock_uses_only_the_run_records(tmp_path):
    """A forcing axis longer than n_time_total: the cycle period comes from
    the records the run uses, as the reads do."""
    from kelp.core.config import NumericsConfig
    from kelp.readers import extract_time_info
    from kelp.tracking import run_single_realization
    mesh, ds, reader, mode = _env(tmp_path, n_time=8)
    ft, _ = extract_time_info(ds)
    num = NumericsConfig(dt=3600.0, integrator="euler")
    out = {}
    for tag, axis in (("full", ft), ("cut", ft[:5])):
        rec = run_single_realization(mesh, ds, reader, _state(), mode, num,
                                     start_time_idx=6, n_steps=3,
                                     n_time_total=5, file_times=axis,
                                     on_exhausted="cycle")
        out[tag] = rec.to_arrays()["time"]
    np.testing.assert_array_equal(out["full"], out["cut"])
    np.testing.assert_array_equal(out["cut"], [25200.0, 28800.0, 32400.0])


@pytest.mark.parametrize("on_exhausted", ["cycle", "stop"])
def test_interpolated_forcing_uses_only_the_run_records(tmp_path,
                                                        on_exhausted):
    """Interpolated mode with an axis longer than n_time_total: the forcing
    clock cycles (or stops) on the records the run uses, like the labels
    and the cycle period (PR review 2026-10-14).  eta = record index, so
    the sampled eta says which record the run read."""
    from kelp.core.config import NumericsConfig
    from kelp.readers import extract_time_info
    from kelp.tracking import run_single_realization
    mesh, ds, reader, mode = _env(tmp_path, n_time=8, eta_ramp=True)
    ft, fdt = extract_time_info(ds)
    num = NumericsConfig(dt=3600.0, integrator="euler")
    out = {}
    for tag, axis in (("full", ft), ("cut", ft[:5])):
        rec = run_single_realization(
            mesh, ds, reader, _state(), mode, num, start_time_idx=2,
            n_steps=6, n_time_total=5, temporal_interp=True,
            file_times=axis, file_dt=fdt, on_exhausted=on_exhausted,
            sample_vars=["eta"], output_sample_vars=["eta"])
        out[tag] = rec.to_arrays()["eta"][:, 0]
    np.testing.assert_array_equal(out["full"], out["cut"])
    if on_exhausted == "cycle":
        # sim_time 3 h .. 8 h on a 5-h period: records 3, 4, 0, 1, 2, 3
        np.testing.assert_allclose(out["cut"], [3.0, 4.0, 0.0, 1.0, 2.0, 3.0])


@pytest.mark.parametrize("ext,flush", [(".nc", 0), (".nc", 2), (".zarr", 0)])
def test_append_refuses_non_finite_times(tmp_path, ext, flush):
    from kelp.writers import TrajectoryWriter
    with TrajectoryWriter(str(tmp_path / f"t{ext}"), 3,
                          flush_every=flush) as w:
        w.append(_state(), 3600.0)
        for bad in (np.nan, np.inf):
            with pytest.raises(ValueError, match="finite"):
                w.append(_state(), bad)


@pytest.mark.parametrize("ext", [".nc", ".zarr"])
def test_partial_release_triplet_refused(tmp_path, ext):
    from kelp.writers import TrajectoryWriter, write_restart_file
    st = _state().replace(release_lat=np.ones(3), release_z=np.zeros(3))
    with TrajectoryWriter(str(tmp_path / f"p{ext}"), 3) as w:
        with pytest.raises(ValueError, match="together"):
            w.write_checkpoint(st, 0, 0)
    with pytest.raises(ValueError, match="together"):
        write_restart_file(str(tmp_path / "r.nc"), st, 0, 0)


def test_periodic_checkpoints_store_the_unwrapped_index(tmp_path,
                                                        monkeypatch):
    from kelp.writers import TrajectoryWriter
    seen = []
    orig = TrajectoryWriter.write_checkpoint

    def spy(self, state, step_i, file_idx, *a, **k):
        seen.append((int(step_i), int(file_idx)))
        return orig(self, state, step_i, file_idx, *a, **k)

    monkeypatch.setattr(TrajectoryWriter, "write_checkpoint", spy)
    _run(tmp_path, "ck", n_steps=5, n_time=3, on_exhausted="cycle",
         checkpoint_freq=2)
    assert seen == [(1, 1), (3, 3), (4, 4)]     # periodic 1, 3; final 4


def test_buffered_set_trajectory_ids_flushes_first(tmp_path):
    import netCDF4
    from kelp.writers import TrajectoryWriter
    p = str(tmp_path / "ids.nc")
    w = TrajectoryWriter(p, 3, flush_every=10).open()
    for k in range(3):
        w.append(_state(), 3600.0 * (k + 1))
    w.set_trajectory_ids(np.arange(5, 8))
    with netCDF4.Dataset(p) as ds:             # readable between flushes
        assert len(ds.dimensions["time"]) == 3
        np.testing.assert_array_equal(ds["trajectory"][:], [5, 6, 7])
    w.close()


def test_concat_refuses_ids_beyond_int32(tmp_path):
    from kelp.writers import TrajectoryWriter
    base = 2**31 - 2
    shards = []
    for k in range(2):
        p = str(tmp_path / f"shard{k}.zarr")
        with TrajectoryWriter(p, 2, metadata={"particle_offset": base + 2 * k,
                                              "random_draws": 0}) as w:
            w.set_time_units("seconds since 2001-01-01 00:00:00", "standard")
            w.write_release(np.zeros(2), np.zeros(2), np.zeros(2),
                            np.zeros(2))
            w.append(_state(P=2), 3600.0)
        shards.append(p)
    with pytest.raises(ValueError, match="exceed"):
        _concat()(shards, tmp_path / "m.zarr")


@pytest.mark.parametrize("time_dir", [1.0, -1.0])
def test_resume_recipe_float_start_index(tmp_path, time_dir):
    """The §6 resume recipe computes ``start0 + n * time_dir`` with the
    config's FLOAT ``time_dir``: a float start index resumes exactly like
    the integer one, and the next checkpoint stores an integer index."""
    from kelp.writers import load_restart, load_trajectories
    start0 = 0 if time_dir > 0 else 7
    whole = load_trajectories(_run(tmp_path, "w", n_steps=6, start=start0,
                                   time_dir=time_dir))
    p1 = _run(tmp_path, "a", n_steps=3, start=start0, time_dir=time_dir,
              checkpoint_freq=3)
    r = load_restart(p1)
    n = r["step_i"] + 1
    p2 = _run(tmp_path, "b", n_steps=3, state=r["state"], time_dir=time_dir,
              start=start0 + n * time_dir, start_step_i=n, checkpoint_freq=3)
    a, b = load_trajectories(p1), load_trajectories(p2)
    np.testing.assert_array_equal(np.concatenate([a["time"], b["time"]]),
                                  whole["time"])
    np.testing.assert_array_equal(np.concatenate([a["lon"], b["lon"]], 1),
                                  whole["lon"])
    r2 = load_restart(p2)
    assert r2["file_idx"] == start0 + 5 * int(time_dir)
    assert isinstance(r2["file_idx"], int)


def test_fractional_start_index_refused(tmp_path):
    with pytest.raises(ValueError, match="whole record index"):
        _run(tmp_path, "f", start=2.5, n_steps=1)


# ---------------------------------------------------------------------------
# PR review (2026-10-14)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ext,flush", [(".nc", 0), (".nc", 2), (".zarr", 0)])
def test_declare_velocity_creates_the_variables_before_any_record(
        tmp_path, ext, flush):
    from kelp.writers import TrajectoryWriter
    p = str(tmp_path / f"d{ext}")
    with TrajectoryWriter(p, 3, flush_every=flush) as w:
        w.declare_velocity(w=False)
        w.append(_state(), 3600.0)            # no velocity on the state
        w.append(_state(), 7200.0)
    ds = _open(p, decode_times=False)
    assert ds["vel_u"].dims == ("trajectory", "time")
    assert np.isnan(ds["vel_u"].values).all() and np.isnan(ds["vel_v"].values).all()
    assert "vel_w" not in ds
    p3 = str(tmp_path / f"d3{ext}")
    with TrajectoryWriter(p3, 3, flush_every=flush) as w:
        w.declare_velocity(w=True)
        w.append(_state(), 3600.0)
    assert np.isnan(_open(p3, decode_times=False)["vel_w"].values).all()


def _dead(P=3):
    from kelp.core.particles import ParticleStatus
    st = _state(P)
    return st.replace(status=np.full(P, int(ParticleStatus.OUTSIDE),
                                     dtype=st.status.dtype))


@pytest.mark.parametrize("ext", EXTS)
def test_segment_with_every_particle_dead_has_velocity_vars(tmp_path, ext):
    """A segment resumed when every particle is already dead records no
    velocity; the run still declares vel_u/vel_v (NaN), so every segment
    and shard of a run has the same variable set."""
    ds = _open(_run(tmp_path, "dead", state=_dead(), ext=ext),
               decode_times=False)
    assert np.isnan(ds["vel_u"].values).all() and np.isnan(ds["vel_v"].values).all()
    assert "vel_w" not in ds


def test_shard_with_every_particle_dead_merges(tmp_path):
    a = _run(tmp_path, "alive", ext=".zarr")
    b = _run(tmp_path, "dead", state=_dead(), ext=".zarr")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")    # shards without particle offsets
        m = _concat()([a, b], tmp_path / "m.zarr", check_order=False)
    ds = xr.open_zarr(str(m), consolidated=False, decode_times=False)
    assert np.isfinite(ds["vel_u"].values[:3]).all()
    assert np.isnan(ds["vel_u"].values[3:]).all()



def _f2_shards_without_declared_velocity(tmp_path, second_status):
    """Two format-2 Zarr shards as a writer without declare_velocity leaves
    them: the first records a velocity, the second (status ``second_status``
    at its only record) records none, so it has no vel_* at all."""
    from kelp.writers import TrajectoryWriter
    paths = []
    for k in range(2):
        p = str(tmp_path / f"old{k}.zarr")
        with TrajectoryWriter(p, 2, metadata={"particle_offset": 2 * k,
                                              "random_draws": 0}) as w:
            w.set_time_units("seconds since 2001-01-01 00:00:00", "standard")
            w.set_trajectory_ids(2 * k + np.arange(2))
            w.write_release(np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2))
            st = _state(P=2)
            if k == 0:
                st = st.replace(u=np.ones(2), v=np.ones(2))
            else:
                st = st.replace(status=np.full(2, second_status,
                                               dtype=st.status.dtype))
            w.append(st, 3600.0)
        paths.append(p)
    return paths


def test_older_format2_shard_never_alive_without_velocity_merges(tmp_path):
    """A format-2 shard written before velocities were declared up front,
    whose particles are never ALIVE, merges with NaN velocities — what
    format 2 records for particles that are not alive."""
    from kelp.core.particles import ParticleStatus
    paths = _f2_shards_without_declared_velocity(
        tmp_path, int(ParticleStatus.OUTSIDE))
    m = _concat()(paths, tmp_path / "m.zarr")
    ds = xr.open_zarr(str(m), consolidated=False, decode_times=False)
    np.testing.assert_array_equal(ds["vel_u"].values[:2, 0], [1.0, 1.0])
    assert np.isnan(ds["vel_u"].values[2:]).all()


def test_format2_shard_with_alive_particles_without_velocity_refused(tmp_path):
    from kelp.core.particles import ParticleStatus
    paths = _f2_shards_without_declared_velocity(
        tmp_path, int(ParticleStatus.ALIVE))
    with pytest.raises(ValueError, match="variable set"):
        _concat()(paths, tmp_path / "m.zarr")


# ---------------------------------------------------------------------------
# Final PR review (2026-10-14)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("start", [0, 3])
def test_single_record_cycle_clock_counts_from_the_start_index(tmp_path,
                                                                start):
    """One forcing record cycled with period dt: record ``k`` of the
    unwrapped axis is ``t0 + k·dt``, so the start index moves the release
    and the labels (it used to be ignored: every start read t0)."""
    a = _run(tmp_path, f"one{start}", n_time=1, start=start, n_steps=2,
             on_exhausted="cycle", writer=False).to_arrays()
    t0 = 3600.0 * start
    np.testing.assert_array_equal(a["release_time"], [t0] * 3)
    np.testing.assert_array_equal(a["time"], [t0 + 3600.0, t0 + 7200.0])


def test_time_zone_of_the_forcing_is_not_carried(tmp_path):
    """KELP keeps the forcing's own clock: a time zone on the timestamps is
    dropped, the epoch repeats their wall-clock value (docs §5.2)."""
    import datetime as _dt
    from kelp.readers import extract_time_info, time_axis_epoch
    tz = _dt.timezone(_dt.timedelta(hours=3))
    t = np.array([_dt.datetime(2001, 1, 1, 3, tzinfo=tz),
                  _dt.datetime(2001, 1, 1, 4, tzinfo=tz)], dtype=object)
    ds = xr.Dataset(coords={"time": ("time", t)})
    assert time_axis_epoch(ds) == ("seconds since 2001-01-01 03:00:00",
                                   "standard")
    np.testing.assert_array_equal(extract_time_info(ds)[0], [0.0, 3600.0])
