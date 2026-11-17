"""Stamp options in configured runs of a reader other than FESOM 1.4.

``data.time_stamp`` / ``data.time_window`` (and ``data.time_offset``) say what
each record's time stamp means; for a FESOM2 run the opened dataset's time
axis is resolved from them once, at the open, and the run then applies no
midpoint shift.  The acceptance test: one field written with its stamps at
the start, the centre and the end of the same daily windows, each read with
the matching options, gives the trajectories and output times of the
centre-stamped field read as snapshots.  Also pinned: no midpoint shift and
no default warning with the options, the one ``Record times`` line, ``kelp
info`` printing the resolved first and last instants, and the default
``time_averaging`` warning pointing to the options, and the output
attributes: each set key as a string, none without the keys, and the shard
concat refusing shards with different options.  The other consumers of the
record times: a cftime (``noleap``) axis, a backward run, PCHIP in time,
prefetched reads, two realizations, ``time.start_date`` matched against the
resolved instants, calendar-month windows (irregular: discrete mode and the
cycle refused), a run resumed from a checkpoint on the resolved axis, and a
HEALPix two-store run (``data.ncinput_eta``).

Fixture (all synthetic, ``tests/_fesom2_synth``): the 17 x 17 node grid of
``tests/test_depth_linear_analytic.py`` with six levels, node u, v constant
in depth and varying from record to record, six daily records restamped to
seconds since 1958-01-01.
"""
import contextlib, dataclasses, io, sys, warnings
from datetime import datetime, timedelta
from pathlib import Path
import cftime
import numpy as np, pytest, xarray as xr, yaml
from kelp import tracking
from kelp.cli.main import main as cli_main
from kelp.core.config import (AppConfig, DataConfig, MeshConfig, ModeConfig,
                              NumericsConfig, TimeConfig)
from kelp.core.mesh import find_nearest_nodes
from kelp.core.particles import make_state
from kelp.readers import _open_run_dataset, extract_time_info, get_reader
from kelp.writers import TrajectoryWriter, load_restart, load_trajectories
from tests._bitwise import assert_bitwise
from tests.test_depth_linear_analytic import (
    CONST_SEEDS, LEVELS_C, NC, _cfg, _col_beds, _const_field, _elem_centres,
    _quadrant_beds, _write_data, _write_mesh)
from tests.test_uv_vertical_elements_run import _global_attrs, _load_mesh

REPO = Path(__file__).resolve().parents[1]
DAY = 86400.0
N_REC = 6
K = np.arange(N_REC, dtype=np.float64)
#: stamps (seconds since 1958-01-01) of the SAME daily windows
STAMPS = {"start": K * DAY, "centre": K * DAY + DAY / 2, "end": (K + 1) * DAY}
UNITS = "seconds since 1958-01-01 0:0:0"
SEEDS = [s[:3] for s in CONST_SEEDS]


def _restamp(data_dir, seconds, calendar="standard"):
    """Rewrite the time coordinate of every file of a synthetic FESOM2 set."""
    for f in sorted(Path(data_dir).glob("*.nc")):
        with xr.open_dataset(f, decode_times=False) as d:
            d = d.load()
        d = d.assign_coords(time=("time", np.asarray(seconds, dtype=np.float64),
                                  {"units": UNITS, "calendar": calendar}))
        tmp = f.with_suffix(".tmp")
        d.to_netcdf(tmp)
        tmp.replace(f)


def _world(tmp_path):
    """Mesh (17 x 17 nodes, six levels) and node u, v that vary from record
    to record, so a shift of the time axis moves the particles."""
    elem_bed = _quadrant_beds(*_elem_centres(NC, NC))
    mesh_path = _write_mesh(tmp_path / "fesom.mesh.diag.nc", NC, NC, LEVELS_C,
                            elem_bed)
    mesh = _load_mesh(mesh_path)
    u, v = _const_field(mesh, "node", _col_beds(mesh, "node", elem_bed), N_REC)
    return mesh_path, mesh, u, v


def _case(tmp_path, world, name, seconds, *, calendar="standard", interp=True,
          ext="nc", time=None, numerics=None, mode=None, n_rec=None,
          **data_kw):
    """Write one data set stamped ``seconds``, run it, return what a test reads.

    ``numerics`` may override ``n_steps`` / ``dt`` / ``temporal_interp`` (one
    merged dict: passing them twice to ``_cfg`` is a TypeError); ``n_rec``
    writes only the first records of the field."""
    mesh_path, mesh, u, v = world
    if n_rec is not None:
        u, v = u[:n_rec], v[:n_rec]
    data = _write_data(tmp_path / f"data_{name}", mesh, "node", u, v)
    if seconds is not None:
        _restamp(data, seconds, calendar)
    num = {"n_steps": 8 if interp else 4, "dt": DAY / 4 if interp else DAY,
           "temporal_interp": interp, **(numerics or {})}
    cfg = _cfg(mesh_path, data, SEEDS, out=f"{name}.{ext}",
               uv_interp="barycentric", integrator="rk2", uv_vertical="layer",
               **num)
    data_kw.setdefault("time_averaging", None)
    cfg.data = dataclasses.replace(cfg.data, **data_kw)
    for k, val in (time or {}).items():
        setattr(cfg.time, k, val)
    for k, val in (mode or {}).items():
        setattr(cfg.mode, k, val)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        tracking.run_tracking(cfg)
    return cfg, out.getvalue()


def _time_of(path):
    with xr.open_dataset(path, decode_times=False) as d:
        t = d["time"]
        return t.values.copy(), t.attrs["units"], t.attrs["calendar"]


def _same_run(a, b):
    ta, tb = load_trajectories(str(a)), load_trajectories(str(b))
    for k in ("lon", "lat", "z"):
        assert_bitwise(ta[k], tb[k], k)
    assert _time_of(a)[1:] == _time_of(b)[1:]
    assert_bitwise(_time_of(a)[0], _time_of(b)[0], "time")


# ---------------------------------------------------------------------------
# the acceptance test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("interp", [True, False], ids=["interp", "discrete"])
@pytest.mark.parametrize("window", ["calendar_day", "1 d"])
def test_fesom2_convention_invariance(tmp_path, window, interp):
    """The acceptance test: one field stamped at the window start, centre and
    end, each with the matching options, and the centre-stamped field read as
    snapshots (no options) give the same trajectories and output times."""
    world = _world(tmp_path)
    # a start INSIDE the records' coverage: a run
    # from record 0 follows the records, so a uniform shift of the axis
    # moves only the output epoch, not the particles
    start = {"start_date": "1958-01-02T06:00:00" if interp
             else "1958-01-02T12:00:00"}          # discrete: on a record
    ref, _ = _case(tmp_path, world, "ref", STAMPS["centre"], interp=interp,
                   time=start, time_averaging="instant")
    assert _time_of(ref.data.output_path)[1:] == (
        "seconds since 1958-01-01 12:00:00", "standard")
    for stamp in ("start", "centre", "end"):
        cfg, _ = _case(tmp_path, world, stamp, STAMPS[stamp], interp=interp,
                       time=start, time_stamp=stamp, time_window=window)
        _same_run(cfg.data.output_path, ref.data.output_path)
        assert _global_attrs(cfg.data.output_path)["time_averaging"] == \
            "resolved_from_options"
    # the test discriminates: the start-stamped field read as snapshots
    # (records 12 h early) from the same absolute start is another run
    early, _ = _case(tmp_path, world, "early", STAMPS["start"], interp=True,
                     time={"start_date": "1958-01-02T06:00:00"},
                     time_averaging="instant")
    ref_i, _ = _case(tmp_path, world, "ref_i", STAMPS["centre"], interp=True,
                     time={"start_date": "1958-01-02T06:00:00"},
                     time_averaging="instant")
    assert not np.array_equal(
        load_trajectories(early.data.output_path)["lon"],
        load_trajectories(ref_i.data.output_path)["lon"])


# ---------------------------------------------------------------------------
# what the run prints and warns
# ---------------------------------------------------------------------------

def test_no_midpoint_shift_no_warning_one_line(tmp_path, monkeypatch):
    """With the options the run neither shifts to window midpoints nor
    warns about ``data.time_averaging``; it prints one ``Record times`` line
    naming the options and the resolved first and last instants."""
    def _no_shift(file_times):
        raise AssertionError("_shift_to_midpoints called with the options")

    monkeypatch.setattr(tracking, "_shift_to_midpoints", _no_shift)
    world = _world(tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, out = _case(tmp_path, world, "opts", STAMPS["start"],
                       time_stamp="start", time_window="calendar_day")
    averaging = [str(w.message) for w in caught
                 if "time_averaging" in str(w.message)]
    assert averaging == [], averaging
    lines = [ln for ln in out.splitlines() if "Record times" in ln]
    assert len(lines) == 1, out
    line = lines[0]
    for part in ("resolved from data.time_stamp=start, "
                 "data.time_window=calendar_day",
                 "1958-01-01T12:00:00", "1958-01-06T12:00:00"):
        assert part in line, (part, line)


def test_kelp_info_prints_the_resolved_axis(tmp_path, capsys):
    """``kelp info`` resolves the axis as the run does and prints the first
    and last instants of the run's ``Record times`` line."""
    mesh_path, mesh, u, v = _world(tmp_path)
    data = _write_data(tmp_path / "data_info", mesh, "node", u, v)
    _restamp(data, STAMPS["start"])
    cfg = {"mode": {"Fdrift2D": False, "Fdrift3D": True, "var_u": "u",
                    "var_v": "v", "var_w": "w", "var_eta": "ssh"},
           "numerics": {"uv_interp": "barycentric"},
           "time": {"n_steps": 1},
           "mesh": {"type": "fesom2", "meshpath": str(mesh_path)},
           "data": {"ncinput": str(data), "time_stamp": "start",
                    "time_window": "calendar_day"}}
    path = tmp_path / "info.yaml"
    path.write_text(yaml.safe_dump(cfg))
    capsys.readouterr()
    rc = cli_main(["info", str(path)])
    cap = capsys.readouterr()
    assert rc == 0, (cap.out, cap.err)
    lines = [ln for ln in cap.out.splitlines()
             if "record times resolved from data.time_stamp=start, "
                "data.time_window=calendar_day" in ln]
    assert len(lines) == 1, cap.out
    assert "1958-01-01T12:00:00" in lines[0], lines[0]
    assert "1958-01-06T12:00:00" in lines[0], lines[0]


def test_default_warning_points_to_the_stamp_options(tmp_path):
    """Without ``data.time_averaging`` and without the options a FESOM2 run
    warns once, still assumes 'mean', and points to ``data.time_stamp`` for
    stamps that are not the end of the averaging window."""
    world = _world(tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _case(tmp_path, world, "default", STAMPS["centre"])
    msgs = [str(w.message) for w in caught
            if "data.time_averaging is not set" in str(w.message)]
    assert len(msgs) == 1, msgs
    assert "data.time_stamp" in msgs[0], msgs[0]
    assert "assuming 'mean'" in msgs[0], msgs[0]


# ---------------------------------------------------------------------------
# output attributes
# ---------------------------------------------------------------------------

def test_attributes_with_the_options(tmp_path):
    """Each set key is written as a string attribute, a key not set is not
    written, and ``time_averaging`` says what resolved the record times."""
    world = _world(tmp_path)
    cfg, _ = _case(tmp_path, world, "opts", STAMPS["start"],
                   time_stamp="start", time_window="calendar_day")
    attrs = _global_attrs(cfg.data.output_path)
    assert attrs["time_stamp"] == "start"
    assert attrs["time_window"] == "calendar_day"
    assert "time_offset" not in attrs
    assert attrs["time_averaging"] == "resolved_from_options"


def test_offset_run_equals_reference_and_stamps_three_keys(tmp_path):
    """Daily means stamped 23:15, a model step before the end of their day
    (as ``write_multiyear`` writes them), read with ``end`` /
    ``calendar_day`` and ``+45 min`` sit at 12:00 of their day: the run
    equals the centre-stamped field read as snapshots, and the output names
    all three keys."""
    world = _world(tmp_path)
    start = {"start_date": "1958-01-02T06:00:00"}
    ref, _ = _case(tmp_path, world, "ref", STAMPS["centre"], time=start,
                   time_averaging="instant")
    cfg, out = _case(tmp_path, world, "late", None, time=start,
                     time_stamp="end", time_window="calendar_day",
                     time_offset="+45 min")
    _same_run(cfg.data.output_path, ref.data.output_path)
    assert "data.time_offset=+45 min" in out, out
    attrs = _global_attrs(cfg.data.output_path)
    assert attrs["time_offset"] == "+45 min"
    assert attrs["time_stamp"] == "end"
    assert attrs["time_window"] == "calendar_day"
    assert attrs["time_averaging"] == "resolved_from_options"


def test_no_new_attribute_without_the_keys(tmp_path):
    """A run without the keys writes none of them (never ``"None"``) and
    keeps its ``time_averaging``."""
    world = _world(tmp_path)
    ref, _ = _case(tmp_path, world, "ref", STAMPS["centre"],
                   time_averaging="instant")
    attrs = _global_attrs(ref.data.output_path)
    assert {"time_stamp", "time_window", "time_offset"}.isdisjoint(attrs), \
        attrs
    assert attrs["time_averaging"] == "instant"


def test_concat_refuses_shards_with_different_time_stamp(tmp_path):
    """Two shards with the same resolved record times but different stamp
    options do not merge: the concat compares every run attribute."""
    pytest.importorskip("zarr")
    utils = str(REPO / "utils")
    if utils not in sys.path:
        sys.path.insert(0, utils)
    from concat_trajectories import concat_trajectories
    world = _world(tmp_path)
    a, _ = _case(tmp_path, world, "sa", STAMPS["start"], ext="zarr",
                 time_stamp="start", time_window="calendar_day")
    b, _ = _case(tmp_path, world, "sb", STAMPS["end"], ext="zarr",
                 time_stamp="end", time_window="calendar_day")
    with pytest.raises(ValueError, match="time_stamp"):
        concat_trajectories([Path(a.data.output_path),
                             Path(b.data.output_path)], tmp_path / "m.zarr")


# ---------------------------------------------------------------------------
# the other consumers of the record times
# ---------------------------------------------------------------------------

def _first_record(path):
    """The first output record time, decoded with the file's own units and
    calendar."""
    t, units, calendar = _time_of(path)
    return cftime.num2date(t[0], units, calendar,
                           only_use_cftime_datetimes=False)


def test_cftime_axis_run(tmp_path):
    """On a ``noleap`` axis (decoded to cftime) the start-stamped field read
    with ``start`` / ``calendar_day`` equals the centre-stamped field read as
    snapshots, from a start inside the records' coverage."""
    world = _world(tmp_path)
    start = {"start_date": "1958-01-02T06:00:00"}
    ref, _ = _case(tmp_path, world, "ref", STAMPS["centre"], calendar="noleap",
                   time=start, time_averaging="instant")
    cfg, _ = _case(tmp_path, world, "opts", STAMPS["start"], calendar="noleap",
                   time=start, time_stamp="start", time_window="calendar_day")
    _same_run(cfg.data.output_path, ref.data.output_path)
    assert _time_of(cfg.data.output_path)[1:] == (
        "seconds since 1958-01-01 12:00:00", "noleap")


#: extra settings of the reference and the ``end`` / ``1 d`` run
VARIANTS = {
    "backward": {"mode": {"time_dir": -1.0}, "time": {"start_time_idx": 5}},
    "pchip": {"numerics": {"temporal_reconstruction": "pchip"}},
    "prefetch": {"numerics": {"prefetch_size": 3}},
    "two_realizations": {"time": {"num_runs": 2, "run_interval": 1}},
}


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_run_variants(tmp_path, variant):
    """A backward run, PCHIP in time, prefetched reads and two realizations
    see the resolved record times: the end-stamped field read with ``end`` /
    ``1 d`` equals the centre-stamped field read as snapshots, in every
    output file.  These runs start on a record, so a uniform shift of the
    axis shows in the output ``time`` units, not in the positions."""
    world = _world(tmp_path)
    extra = VARIANTS[variant]
    ref, _ = _case(tmp_path, world, "ref", STAMPS["centre"],
                   time_averaging="instant", **extra)
    cfg, _ = _case(tmp_path, world, "opts", STAMPS["end"], time_stamp="end",
                   time_window="1 d", **extra)
    pairs = [(Path(cfg.data.output_path), Path(ref.data.output_path))]
    if variant == "two_realizations":
        pairs = [(p.with_stem(f"{p.stem}_r{i}"), q.with_stem(f"{q.stem}_r{i}"))
                 for p, q in pairs for i in range(2)]
    for path, ref_path in pairs:
        assert path.exists() and ref_path.exists(), (path, ref_path)
        _same_run(path, ref_path)
        assert _time_of(path)[1:] == (
            "seconds since 1958-01-01 12:00:00", "standard")
    if variant == "two_realizations":
        # the second realization starts one record (one day) later
        assert _first_record(pairs[1][0]) - _first_record(pairs[0][0]) == \
            timedelta(days=1)
        assert not np.array_equal(load_trajectories(str(pairs[0][0]))["lon"],
                                  load_trajectories(str(pairs[1][0]))["lon"])


def test_start_date_meets_the_resolved_instants(tmp_path):
    """``time.start_date`` is matched against the resolved instants: daily
    means stamped at the start of their day sit at 12:00, so a discrete run
    must start at 12:00; a start at 00:00 needs ``temporal_interp``."""
    world = _world(tmp_path)
    opts = {"time_stamp": "start", "time_window": "calendar_day"}
    cfg, _ = _case(tmp_path, world, "d12", STAMPS["start"], interp=False,
                   time={"start_date": "1958-01-02T12:00"}, **opts)
    # the first record is written after the first step of dt = 1 day
    assert _first_record(cfg.data.output_path) == datetime(1958, 1, 3, 12)
    with pytest.raises(ValueError, match="does not match any file timestep"):
        _case(tmp_path, world, "d00", STAMPS["start"], interp=False,
              time={"start_date": "1958-01-02T00:00"}, **opts)
    cfg, _ = _case(tmp_path, world, "i00", STAMPS["start"], interp=True,
                   time={"start_date": "1958-01-02T00:00"}, **opts)
    # dt = 6 h with temporal_interp
    assert _first_record(cfg.data.output_path) == datetime(1958, 1, 2, 6)


def test_calendar_months_need_temporal_interp(tmp_path):
    """Monthly means stamped at the start of their month sit at the month
    centres (15.5, 45.0, 74.5, 105.0 d: spacings 29.5, 29.5, 30.5 d): a
    discrete run is refused as irregular, an interpolated run runs from the
    first centre, and ``time.on_exhausted: cycle`` is refused before any
    output exists."""
    world = _world(tmp_path)
    months = np.array([0.0, 31.0, 59.0, 90.0]) * DAY
    opts = {"time_stamp": "start", "time_window": "calendar_month"}
    with pytest.raises(ValueError, match="irregular") as info:
        _case(tmp_path, world, "mdisc", months, n_rec=4, interp=False,
              numerics={"dt": 30 * DAY}, **opts)
    # the spacings of the resolved centres, not of the raw stamps (31, 28,
    # 31 d)
    assert "2.5488e+06..2.6352e+06 s" in str(info.value), str(info.value)
    cfg, _ = _case(tmp_path, world, "minterp", months, n_rec=4,
                   numerics={"dt": DAY, "n_steps": 4}, **opts)
    assert _time_of(cfg.data.output_path)[1] == \
        "seconds since 1958-01-16 12:00:00"
    with pytest.raises(ValueError, match=r"time\.on_exhausted"):
        _case(tmp_path, world, "mcycle", months, n_rec=4,
              numerics={"dt": DAY, "n_steps": 4},
              time={"on_exhausted": "cycle"}, **opts)
    assert not (tmp_path / "mcycle.nc").exists()


def test_resumed_chain_on_the_resolved_axis_equals_continuous(tmp_path):
    """A resume (``load_restart`` -> ``run_single_realization``) on a
    dataset opened through ``_open_run_dataset`` with the stamp options
    equals the continuous run bit for bit, and the output's time axis starts
    at the resolved first instant."""
    mesh_path, mesh, u, v = _world(tmp_path)
    data = _write_data(tmp_path / "data_chain", mesh, "node", u, v)
    _restamp(data, STAMPS["start"])
    cfg = _cfg(mesh_path, data, SEEDS, out="chain.nc",
               uv_interp="barycentric", integrator="rk2", uv_vertical="layer",
               dt=DAY, temporal_interp=True)
    cfg.data = dataclasses.replace(cfg.data, time_averaging=None,
                                   time_stamp="start",
                                   time_window="calendar_day")
    reader = get_reader("fesom2")
    ds = _open_run_dataset(reader, mesh, cfg)
    ft, fdt = extract_time_info(ds)

    def run(name, state, n, step0):
        path = str(tmp_path / f"{name}.nc")
        w = TrajectoryWriter(path, state.n_particles).open()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                tracking.run_single_realization(
                    mesh, ds, reader, state, cfg.mode, cfg.numerics,
                    start_time_idx=step0, n_steps=n,
                    n_time_total=ds.sizes["time"], writer=w,
                    checkpoint_freq=2, temporal_interp=True, file_times=ft,
                    file_dt=fdt, sim_time_start=step0 * DAY,
                    start_step_i=step0)
        finally:
            w.close()
        return path

    lon, lat, z = (np.array(c, dtype=np.float64) for c in zip(*SEEDS))
    init = make_state(lon, lat, z)
    init = init.replace(nearest_node=find_nearest_nodes(mesh, lon, lat))
    try:
        whole = run("whole", init, 4, 0)
        a = run("a", init, 2, 0)
        r = load_restart(a)
        b = run("b", r["state"], 2, r["step_i"] + 1)
    finally:
        ds.close()
    tw, ta, tb = (load_trajectories(p) for p in (whole, a, b))
    assert_bitwise(np.concatenate([ta["time"], tb["time"]]), tw["time"],
                   "time")
    for k in ("lon", "lat", "z", "status"):
        assert_bitwise(np.concatenate([ta[k], tb[k]], axis=1), tw[k], k)
    assert np.ptp(tw["lon"]) > 0.01                    # it moves
    # the values are seconds since the first record; the units carry the
    # resolved first instant (a uniform shift shows only there)
    assert tw["time"][0] == DAY
    assert _time_of(whole)[1:] == ("seconds since 1958-01-01 12:00:00",
                                   "standard")


def test_healpix_two_store_convention_invariance(tmp_path):
    """A HEALPix run with a separate surface-elevation store: the options
    apply to the merged dataset, and the field stamped at the start, the
    centre and the end of the same daily windows, each with the matching
    options, gives the run of the centre-stamped field read as snapshots."""
    pytest.importorskip("healpy")
    # that module skips at import without healpy: imported here, not at the
    # top of this file
    from tests.test_healpix_reader import (LEVELS_M, _make_eta_store,
                                           _make_uv_store)
    src = _make_uv_store(tmp_path / "src.nc", nside=8, with_w=True,
                         w_value=-1e-4)
    with xr.open_dataset(src) as d:
        base = d.load()
    base["avg_uoe"] = base["avg_uoe"] * xr.DataArray([1.0, 1.5, 2.5],
                                                     dims="time")
    stamps = {
        "start": ["2020-01-02T00:00", "2020-01-03T00:00", "2020-01-04T00:00"],
        "centre": ["2020-01-02T12:00", "2020-01-03T12:00", "2020-01-04T12:00"],
        "end": ["2020-01-03T00:00", "2020-01-04T00:00", "2020-01-05T00:00"],
    }
    seeds = tmp_path / "seeds_hp.txt"
    seeds.write_text("10.0 5.0 15.0\n-40.0 -20.0 5.0\n")

    def run(kind, name, **data_kw):
        store = tmp_path / f"3D_{kind}_ocean.nc"
        if not store.exists():
            times = np.array(stamps[kind], dtype="datetime64[ns]")
            base.drop_vars("time").assign_coords(time=times).to_netcdf(store)
            _make_eta_store(tmp_path / f"eta_{kind}.nc", nside=8,
                            times=tuple(stamps[kind]))
        out = tmp_path / f"{name}.nc"
        cfg = AppConfig(
            mode=ModeConfig(Fdrift2D=False, Fdrift3D=True),
            numerics=NumericsConfig(uv_interp="barycentric",
                                    integrator="rk2", dt=21600.0,
                                    temporal_interp=True,
                                    velocity_float32=True),
            mesh=MeshConfig(type="healpix", meshpath=str(store),
                            levels=LEVELS_M),
            data=DataConfig(ncinput=str(store), inicloud=str(seeds),
                            ncinput_eta=str(tmp_path / f"eta_{kind}.nc"),
                            output_path=str(out), **data_kw),
            time=TimeConfig(n_steps=6, start_date="2020-01-02T18:00:00"),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            tracking.run_tracking(cfg)
        return out

    ref = run("centre", "ref", time_averaging="instant")
    t = load_trajectories(str(ref))
    assert np.ptp(t["lon"][0]) > 0.0 and np.ptp(t["z"][0]) > 0.0  # it moves
    for kind in ("start", "centre", "end"):
        out = run(kind, f"opts_{kind}", time_stamp=kind,
                  time_window="calendar_day")
        _same_run(out, ref)
        assert _global_attrs(out)["time_averaging"] == "resolved_from_options"
