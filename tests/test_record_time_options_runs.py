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
concat refusing shards with different options.

Fixture (all synthetic, ``tests/_fesom2_synth``): the 17 x 17 node grid of
``tests/test_depth_linear_analytic.py`` with six levels, node u, v constant
in depth and varying from record to record, six daily records restamped to
seconds since 1958-01-01.
"""
import contextlib, dataclasses, io, sys, warnings
from pathlib import Path
import numpy as np, pytest, xarray as xr, yaml
from kelp import tracking
from kelp.cli.main import main as cli_main
from kelp.writers import load_trajectories
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
    merged dict: passing them twice to ``_cfg`` is a TypeError — plan review
    r1, agy F1); ``n_rec`` writes only the first records of the field."""
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
    # a start INSIDE the records' coverage (plan review r1, codex 1): a run
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
    cfg, _ = _case(tmp_path, world, "late", None, time=start,
                   time_stamp="end", time_window="calendar_day",
                   time_offset="+45 min")
    _same_run(cfg.data.output_path, ref.data.output_path)
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
