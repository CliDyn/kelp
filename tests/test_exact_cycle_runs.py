"""Configured runs with the exact cycle period (``time.on_exhausted: cycle``).

Twelve monthly means of one ``noleap`` year repeat after 365 d, with a seam of
31 d from the December record to the January record of the next pass (the
legacy rule, the span of the records plus the last interval, gives 364.5 d).
A ``cycle`` run on the twelve records is compared with "the doubled
reference": a ``stop`` run on the same records written twice along time, the
second copy stamped one period later, with the same seeds, ``dt`` and number
of steps.  Discrete mode (one record per step) cannot follow windows of
unequal length and is refused before the output exists.

Fixture: the FESOM 1.4 grid of ``tests/test_fesom14_tracking.py`` (level-wise
files with cell polygons, the mesh recovered from them); monthly means with
time bounds at the month edges and stamps at the window centres, calendar
``noleap``; ``uo = 2e-4 * (1 + k % n_months)`` m/s for record ``k``, uniform
in space, ``vo = wo = 0``.  The runs of a reader other than FESOM 1.4 use the
FESOM2 fixture of ``tests/test_record_time_options_runs.py`` (17 x 17 nodes,
six levels, node u, v constant in depth and varying in space and from record
to record) with twelve monthly means given by ``data.time_stamp`` and
``data.time_window: calendar_month``; snapshots on calendar windows stay
refused, for every reader; and a ``cycle`` run on irregularly spaced records
without known windows warns that its period is the legacy one.  The other
consumers of the period follow the seam too: a sampled scalar
(``mode.sample_vars``), the surface elevation of a FESOM-C sigma run (the
file of ``tests/test_uv_vertical_sigma_run.py`` restamped to the month
starts), the start of a later realization (with and without
``time.start_date``), a run resumed from a checkpoint after the seam,
prefetched reads and PCHIP in time.  Equal windows and axes without known
windows keep the legacy period, and their runs are meant to be compared
bit for bit with the code before the exact period.  All data is
synthetic, written in ``tmp_path``.  The names of the exact period are
imported inside the tests, so the file also runs on code that lacks them.
"""
from __future__ import annotations

import dataclasses
import inspect
import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

pytest.importorskip("netCDF4")
pytest.importorskip("cftime")

from kelp import tracking  # noqa: E402
from kelp.core.config import MeshConfig, ModeConfig, NumericsConfig, TimeConfig  # noqa: E402
from kelp.core.mesh import find_nearest_nodes  # noqa: E402
from kelp.core.particles import make_state  # noqa: E402
from kelp.readers import extract_time_info, get_reader  # noqa: E402
from kelp.readers.fesom14 import Fesom14Reader  # noqa: E402
from kelp.writers import (TrajectoryWriter, load_restart,  # noqa: E402
                          load_trajectories)

from kelp.cli.main import main as cli_main  # noqa: E402

from tests._bitwise import assert_bitwise  # noqa: E402
from tests.test_depth_linear_analytic import (LEVELS_C, NC,  # noqa: E402
                                              _col_beds, _const_field,
                                              _elem_centres, _quadrant_beds,
                                              _write_data, _write_mesh)
from tests.test_fesom14_tracking import (DAY, LAT, LAT_B, LEVELS, LON,  # noqa: E402
                                         LON_B, SEEDS, WET, _cfg, _data,
                                         _quiet, _run, _write_nc)
from tests.test_record_time_options_runs import (STAMPS, _case,  # noqa: E402
                                                 _restamp, _same_run, _world)
from tests.test_uv_vertical_elements_run import _load_mesh  # noqa: E402
from tests.test_uv_vertical_sigma_run import DT as SIGMA_DT  # noqa: E402
from tests.test_uv_vertical_sigma_run import _cfg as _sigma_cfg  # noqa: E402
from tests.test_uv_vertical_sigma_run import _file as _sigma_file  # noqa: E402

#: month edges of one noleap year, days since 2001-01-01
EDGES = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365],
                 dtype=np.float64)


def _month_time_vars(n_months: int, copies: int):
    """Time and bounds of ``n_months`` monthly means from January 2001,
    written ``copies`` times, each copy stamped the sum of the windows
    later."""
    total = EDGES[n_months]
    lo = np.concatenate([EDGES[:n_months] + c * total for c in range(copies)])
    hi = np.concatenate([EDGES[1:n_months + 1] + c * total
                         for c in range(copies)])
    attrs = {"units": "days since 2001-01-01", "calendar": "noleap",
             "standard_name": "time", "bounds": "time_bnds"}
    return ([("time", ("time",), (lo + hi) / 2.0, attrs),
             ("time_bnds", ("time", "bnds"), np.stack([lo, hi], axis=1),
              {})], {"bnds": 2})


def _monthly_data(d: Path, n_months: int = 12, copies: int = 1,
                  tracer: bool = False) -> Path:
    """Level-wise ``uo``, ``vo``, ``wo`` (the u file holds the polygons) and,
    with ``tracer``, ``tracer(time, ncells) = k % n_months``."""
    if d.exists():
        return d
    d.mkdir(parents=True)
    n = n_months * copies
    k = np.arange(n)
    tvars, tdims = _month_time_vars(n_months, copies)
    for var in ("uo", "vo", "wo"):
        rec = 2e-4 * (1 + k % n_months) if var == "uo" else np.zeros(n)
        vals = np.where(WET[None], rec[:, None, None]
                        + 0.0 * LON[None, None, :], np.nan)
        attrs = {"_FillValue": 1e20}
        if var == "wo":
            attrs["standard_name"] = "upward_sea_water_velocity"
        variables = tvars + [
            ("depth", ("depth",), LEVELS, {"units": "m", "positive": "down"}),
            ("lon", ("ncells",), LON, {"standard_name": "longitude",
                                       "bounds": "lon_bnds"}),
            ("lat", ("ncells",), LAT, {"standard_name": "latitude",
                                       "bounds": "lat_bnds"}),
            (var, ("time", "depth", "ncells"), vals, attrs),
        ]
        dims = {"time": None, **tdims, "depth": LEVELS.size,
                "ncells": LON.size}
        if var == "uo":
            dims["vertices"] = LON_B.shape[1]
            variables += [("lon_bnds", ("ncells", "vertices"), LON_B, {}),
                          ("lat_bnds", ("ncells", "vertices"), LAT_B, {})]
        _write_nc(d / f"{var}.nc", dims, variables)
    if tracer:
        values = np.repeat((k % n_months).astype(np.float64)[:, None],
                           LON.size, axis=1)
        _write_nc(d / "tracer.nc", {"time": None, **tdims, "ncells": LON.size},
                  tvars + [("tracer", ("time", "ncells"), values,
                            {"_FillValue": 1e20, "units": "1"})])
    return d


def _year_cfg(tmp_path: Path, data: Path, out: str, *, n_steps: int,
              on_exhausted: str, start_time_idx: int = 0, **kw):
    """The FESOM 1.4 run of ``_cfg`` (3-D RK2, dt = 1 d, lerp in time) from
    record ``start_time_idx``."""
    cfg = _cfg(tmp_path, data=data, out=out, **kw)
    cfg.time = TimeConfig(start_time_idx=start_time_idx, n_steps=n_steps,
                          on_exhausted=on_exhausted)
    return cfg


def _period(path) -> float:
    with xr.open_dataset(path, decode_times=False) as ds:
        return float(ds["time"].attrs["forcing_cycle_period"])


def _assert_doubled_reference(got, ref):
    for key in ("lon", "lat", "z"):
        assert np.allclose(got[key], ref[key], rtol=0, atol=1e-9), (
            key, np.max(np.abs(got[key] - ref[key])))
    assert np.array_equal(got["status"], ref["status"])


def _m_d(period: float, seam: float) -> str:
    return ("time.on_exhausted: cycle with numerics.temporal_interp: false is "
            "refused: the records' averaging windows differ in length, so the "
            f"forcing repeats after {period / DAY:g} d with a seam of "
            f"{seam / DAY:g} d between the last and the first record, which a "
            "discrete run (one record per step of dt) cannot follow. Set "
            "numerics.temporal_interp: true")


def test_fesom14_monthly_cycle_equals_the_records_written_twice(tmp_path):
    """450 daily steps on twelve monthly means: through the seam December ->
    January and into the second pass, as on the records written twice."""
    year = _monthly_data(tmp_path / "year")
    twice = _monthly_data(tmp_path / "twice", copies=2)
    cfg = _year_cfg(tmp_path, year, "cycle.nc", n_steps=450,
                    on_exhausted="cycle")
    ref = _year_cfg(tmp_path, twice, "ref.nc", n_steps=450,
                    on_exhausted="stop")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _run(cfg)
    _run(ref)
    got, want = (load_trajectories(c.data.output_path) for c in (cfg, ref))
    _assert_doubled_reference(got, want)
    assert np.array_equal(got["time"], want["time"])
    assert _period(cfg.data.output_path) == 365 * DAY
    assert np.ptp(got["lon"]) > 0.1                       # it moves
    assert not [w for w in caught if "last interval" in str(w.message)]


def test_fesom14_backward_cycle_across_the_seam(tmp_path):
    """60 daily steps backward from the first record: the run enters the
    seam bracket from its top and goes on in the previous pass."""
    year = _monthly_data(tmp_path / "year")
    twice = _monthly_data(tmp_path / "twice", copies=2)
    back = {"time_dir": -1.0}
    cfg = _year_cfg(tmp_path, year, "cycle.nc", n_steps=60,
                    on_exhausted="cycle", mode=back)
    ref = _year_cfg(tmp_path, twice, "ref.nc", n_steps=60,
                    on_exhausted="stop", start_time_idx=12, mode=back)
    _run(cfg)
    _run(ref)
    got, want = (load_trajectories(c.data.output_path) for c in (cfg, ref))
    _assert_doubled_reference(got, want)
    t_cycle, t_ref = got["time"], want["time"]
    assert np.array_equal(t_ref - t_cycle, np.full_like(t_ref, 365 * DAY))


@pytest.mark.parametrize("n_months, dt_days, period_days", [
    (3, 29.5, 90), (12, 30.0, 365)], ids=["jan_mar", "year"])
def test_discrete_cycle_on_unequal_windows_is_refused(tmp_path, n_months,
                                                       dt_days, period_days):
    """Discrete mode with ``cycle`` on windows of unequal length: refused
    before the output exists (January-March are spaced evenly, 29.5 d, so
    only this refusal stops them; on the year it comes before the refusal
    of irregular record times)."""
    data = _monthly_data(tmp_path / "data", n_months=n_months)
    num = {"temporal_interp": False, "dt": dt_days * DAY}
    cfg = _year_cfg(tmp_path, data, "cycle.nc", n_steps=2,
                    on_exhausted="cycle", numerics=num)
    with pytest.raises(ValueError) as info:
        _run(cfg)
    assert str(info.value) == _m_d(period_days * DAY, 31 * DAY)
    assert not Path(cfg.data.output_path).exists()

    if n_months == 3:
        # the same records under stop: no cycle, no refusal
        cfg = _year_cfg(tmp_path, data, "stop.nc", n_steps=2,
                        on_exhausted="stop", numerics=num)
        _run(cfg)
        assert load_trajectories(cfg.data.output_path)["lon"].shape == (2, 2)

    # a direct caller without file_times meets the same refusal (on the
    # year ahead of the refusal of irregular record times there too)
    mode = ModeConfig(Fdrift2D=False, Fdrift3D=True)
    mesh = _quiet(Fesom14Reader.load_mesh,
                  MeshConfig(type="fesom14", meshpath=str(data)), mode_cfg=mode)
    ds = _quiet(Fesom14Reader.open_dataset, str(data), mode_cfg=mode)
    try:
        numerics = NumericsConfig(dt=dt_days * DAY, integrator="rk2",
                                  uv_interp="barycentric",
                                  w_interp="barycentric",
                                  temporal_interp=False, output_freq=1)
        lon, lat, z = (np.array(c, dtype=np.float64) for c in zip(*SEEDS))
        init = make_state(lon, lat, z)
        init = init.replace(nearest_node=find_nearest_nodes(mesh, lon, lat))
        with pytest.raises(ValueError) as info:
            _quiet(tracking.run_single_realization,
                   mesh, ds, Fesom14Reader, init, mode, numerics,
                   start_time_idx=0, n_steps=2,
                   n_time_total=ds.sizes["time"], on_exhausted="cycle",
                   temporal_interp=False)
        assert str(info.value) == _m_d(period_days * DAY, 31 * DAY)
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# a reader other than FESOM 1.4: monthly means given by the stamp options
# ---------------------------------------------------------------------------

#: the month edges of one noleap year in days (the FESOM2 runs count them
#: from 1958-01-01)
MONTH_EDGES = EDGES


def _year_world(tmp_path: Path, copies: int = 1):
    """The FESOM2 world of ``_world`` with twelve records of a slower field
    (at most ~7e-4 m/s, a quarter of a degree in 450 d), written ``copies``
    times along time.  The mesh file is written once per ``tmp_path``."""
    elem_bed = _quadrant_beds(*_elem_centres(NC, NC))
    mesh_path = tmp_path / "fesom.mesh.diag.nc"
    if not mesh_path.exists():
        _write_mesh(mesh_path, NC, NC, LEVELS_C, elem_bed)
    mesh = _load_mesh(mesh_path)
    beds = _col_beds(mesh, "node", elem_bed)
    u, v = (0.01 * a for a in _const_field(mesh, "node", beds, 12))
    if copies == 2:
        u, v = np.concatenate([u, u]), np.concatenate([v, v])
    return mesh_path, mesh, u, v


def _month_stamps(stamp: str, copies: int = 1) -> np.ndarray:
    """Seconds since 1958-01-01 (``noleap``) of the twelve months stamped at
    their ``start``, ``centre`` or ``end``; each further copy 365 d later."""
    lo, hi = MONTH_EDGES[:-1], MONTH_EDGES[1:]
    one = {"start": lo, "centre": (lo + hi) / 2.0, "end": hi}[stamp]
    return np.concatenate([one + 365.0 * c for c in range(copies)]) * DAY


def _last_interval(caught):
    return [w for w in caught if "last interval" in str(w.message)]


def _m_w(dmin: float, dmax: float, period: float) -> str:
    return (f"time.on_exhausted: cycle on irregularly spaced records "
            f"({dmin / DAY:g}..{dmax / DAY:g} d between records): the forcing "
            "period is taken as the span of the records plus the last "
            f"interval ({period / DAY:g} d), which is exact only for equally "
            "spaced records. For means over calendar months or years set "
            "data.time_stamp and data.time_window, so that the period is the "
            "sum of the averaging windows (the FESOM 1.4 reader resolves its "
            "windows from the files)")


def test_monthly_means_cycle_for_each_stamp_convention(tmp_path):
    """Twelve monthly means stamped at the start, the centre or the end of
    their month, each read with the matching options and ``cycle``: the same
    run bit for bit, 450 daily steps through the seam December -> January,
    equal to the doubled reference; the period is 365 d (the legacy rule:
    364.5 d) and no legacy-period warning is given."""
    world = _year_world(tmp_path)
    twice = _year_world(tmp_path, copies=2)
    num = {"dt": DAY, "n_steps": 450}
    out = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for stamp in ("start", "centre", "end"):
            cfg, _ = _case(tmp_path, world, f"m_{stamp}",
                           _month_stamps(stamp), calendar="noleap", n_rec=12,
                           time={"on_exhausted": "cycle"}, numerics=num,
                           time_stamp=stamp, time_window="calendar_month")
            out[stamp] = cfg.data.output_path
    ref, _ = _case(tmp_path, twice, "m_ref", _month_stamps("centre", 2),
                   calendar="noleap", n_rec=24, numerics=num,
                   time_stamp="centre", time_window="calendar_month")
    for stamp in ("start", "end"):
        _same_run(out[stamp], out["centre"])
    got = load_trajectories(out["centre"])
    want = load_trajectories(ref.data.output_path)
    _assert_doubled_reference(got, want)
    assert np.array_equal(got["time"], want["time"])
    assert np.ptp(got["lon"]) > 0.01                      # it moves
    for stamp in out:
        assert _period(out[stamp]) == 365 * DAY, stamp
    assert not _last_interval(caught)


def test_kelp_info_accepts_means_and_refuses_snapshots(tmp_path, capsys):
    """``kelp info`` applies the run's cycle rule before the mesh is read:
    monthly means (``start``) with ``cycle`` pass, snapshots (``instant``)
    on calendar months are refused."""
    mesh_path, mesh, u, v = _world(tmp_path)
    data = _write_data(tmp_path / "data_info", mesh, "node", u, v)
    _restamp(data, MONTH_EDGES[:6] * DAY, "noleap")
    for stamp, rc_want in (("start", 0), ("instant", 1)):
        cfg = {"mode": {"Fdrift2D": False, "Fdrift3D": True, "var_u": "u",
                        "var_v": "v", "var_w": "w", "var_eta": "ssh"},
               "numerics": {"uv_interp": "barycentric"},
               "time": {"n_steps": 1, "on_exhausted": "cycle"},
               "mesh": {"type": "fesom2", "meshpath": str(mesh_path)},
               "data": {"ncinput": str(data), "time_stamp": stamp,
                        "time_window": "calendar_month"}}
        path = tmp_path / f"info_{stamp}.yaml"
        path.write_text(yaml.safe_dump(cfg))
        capsys.readouterr()
        rc = cli_main(["info", str(path)])
        cap = capsys.readouterr()
        assert rc == rc_want, (stamp, cap.out, cap.err)
        if stamp == "instant":
            assert "data.time_stamp: instant" in cap.err, cap.err


def test_irregular_legacy_axis_cycle_warns(tmp_path):
    """A ``cycle`` run on irregularly spaced records without known windows
    keeps the legacy period (span plus the last interval) and warns once,
    pointing to ``data.time_stamp`` / ``data.time_window``: (i) twelve
    monthly means stamped at the month ends under ``time_averaging: mean``
    (period 363 d), (ii) five snapshots on 1 January 1959-1963 (1826 d).
    No warning for (iii) six equally spaced daily records, (iv) case (i)
    under ``stop``, (v) two records (one interval is always regular)."""
    days = _world(tmp_path)
    year = _year_world(tmp_path)          # reuses the mesh file of _world
    cyc = {"on_exhausted": "cycle"}
    num = {"dt": DAY, "n_steps": 4}

    def _warned(name, world, seconds, **kw):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg, _ = _case(tmp_path, world, name, seconds, **kw)
        return cfg, _last_interval(caught)

    # (i) monthly END stamps shifted to midpoints: 28..31 d apart
    cfg, got = _warned("i_end", year, MONTH_EDGES[1:] * DAY,
                       calendar="noleap", n_rec=12, time=cyc, numerics=num,
                       time_averaging="mean")
    assert len(got) == 1, [str(w.message) for w in got]
    assert issubclass(got[0].category, UserWarning)
    msg = str(got[0].message)
    for part in ("plus the last interval (363 d)", "data.time_stamp",
                 "data.time_window"):
        assert part in msg, (part, msg)
    assert msg == _m_w(28 * DAY, 31 * DAY, 363 * DAY)
    assert _period(cfg.data.output_path) == 363 * DAY

    # (ii) yearly snapshots across the leap year 1960: 365..366 d apart
    years = np.array([365.0, 730.0, 1096.0, 1461.0, 1826.0]) * DAY
    cfg, got = _warned("ii_years", days, years, calendar="standard", n_rec=5,
                       time=cyc, time_averaging="instant")
    assert len(got) == 1, [str(w.message) for w in got]
    assert "(1826 d)" in str(got[0].message), str(got[0].message)
    assert str(got[0].message) == _m_w(365 * DAY, 366 * DAY, 1826 * DAY)

    # (iii) six equally spaced daily records
    _, got = _warned("iii_daily", days, np.arange(6.0) * DAY, n_rec=6,
                     time=cyc, time_averaging="instant")
    assert not got, [str(w.message) for w in got]

    # (iv) case (i) under stop
    _, got = _warned("iv_stop", year, MONTH_EDGES[1:] * DAY,
                     calendar="noleap", n_rec=12, numerics=num,
                     time_averaging="mean")
    assert not got, [str(w.message) for w in got]

    # (v) two records 31 d apart
    _, got = _warned("v_two", days, np.array([0.0, 31.0]) * DAY, n_rec=2,
                     time=cyc, time_averaging="instant")
    assert not got, [str(w.message) for w in got]


# ---------------------------------------------------------------------------
# the other consumers of the period
# ---------------------------------------------------------------------------

def _timed(cfg, **time_kw):
    """``cfg`` with its time block replaced by ``TimeConfig(**time_kw)``."""
    cfg.time = TimeConfig(**time_kw)
    return cfg


def test_sampled_scalar_follows_the_exact_seam(tmp_path):
    """A sampled scalar (``tracer = k`` for record ``k``, uniform in space)
    is blended in the seam bracket December -> January (31 d): from the
    December centre two steps of 15.5 d reach 1 January, the middle of the
    seam (11 / 2 = 5.5; the legacy 30.5 d bracket gives 5.41), and the
    January centre of the next pass (0; the legacy period of 364.5 d puts
    it half a day into January: 0.017)."""
    data = _monthly_data(tmp_path / "data", tracer=True)
    cfg = _timed(_cfg(tmp_path, data=data, out="sampled.nc",
                      mode={"sample_vars": ["tracer"]},
                      numerics={"dt": 15.5 * DAY}),
                 start_date="2001-12-16T12:00", n_steps=2,
                 on_exhausted="cycle")
    _run(cfg)
    tracer = load_trajectories(cfg.data.output_path)["tracer"]
    assert tracer.shape == (len(SEEDS), 2)
    assert tracer[:, 0] == pytest.approx(5.5, abs=1e-9), tracer
    assert tracer[:, 1] == pytest.approx(0.0, abs=1e-9), tracer


def _sigma_months(work: Path, copies: int) -> Path:
    """The FESOM-C file of ``tests/test_uv_vertical_sigma_run.py`` with
    twelve records (a surface rising from record to record) stamped at the
    twelve month starts of 2001 (``noleap``), written ``copies`` times along
    time, each copy 365 d later."""
    src = _sigma_file(work / "src", n_rec=12)
    with xr.open_dataset(src, decode_times=False) as d:
        one = d.load()
    attrs = {"units": "seconds since 2001-01-01", "calendar": "noleap"}
    parts = [one.assign_coords(time=("time", (EDGES[:12] + 365.0 * c) * DAY,
                                     attrs))
             for c in range(copies)]
    # the mesh arrays are data variables without time: "minimal" keeps them
    # out of the concatenation (a time axis on them fails the mesh loader)
    out = parts[0] if copies == 1 else xr.concat(
        parts, dim="time", data_vars="minimal", coords="minimal",
        compat="equals")
    path = work / "fesomc.nc"
    out.to_netcdf(path)
    return path


def _sigma_month_cfg(path: Path, out: str, on_exhausted: str):
    """Monthly means stamped at the month starts, temporal interpolation,
    eight steps of the fixture's ``DT`` from record 11 (December)."""
    cfg = _sigma_cfg(path, start=11, n_steps=8, dt=SIGMA_DT, out=out,
                     temporal_interp=True)
    cfg.data = dataclasses.replace(cfg.data, time_averaging=None,
                                   time_stamp="start",
                                   time_window="calendar_month")
    cfg.time.on_exhausted = on_exhausted
    return cfg


def test_sigma_eta_follows_the_exact_seam(tmp_path):
    """FESOM-C sigma 3-D from the December centre: the surface elevation
    that turns the sigma fractions into depths is blended in the seam
    bracket December -> January (31 d), as on the records written twice."""
    year = _sigma_months(tmp_path / "year", copies=1)
    twice = _sigma_months(tmp_path / "twice", copies=2)
    cfg = _sigma_month_cfg(year, "cycle.nc", "cycle")
    ref = _sigma_month_cfg(twice, "ref.nc", "stop")
    _run(cfg)
    _run(ref)
    got, want = (load_trajectories(c.data.output_path) for c in (cfg, ref))
    _assert_doubled_reference(got, want)
    assert np.array_equal(got["time"], want["time"])
    assert np.ptp(got["z"], axis=1).min() > 1e-3          # every z moves


@pytest.mark.parametrize("start_date", [None, "2001-02-01T00:00"],
                         ids=["index", "start_date"])
def test_realizations_start_whole_exact_periods_apart(tmp_path, start_date):
    """Two realizations twelve records apart start one exact period (365 d)
    apart and follow the same forcing; thirteen records apart (index start)
    the second starts at the February centre of the next pass (394.5 d)."""
    data = _monthly_data(tmp_path / "data")

    def _pair(name, run_interval):
        cfg = _timed(_cfg(tmp_path, data=data, out=f"{name}.nc"),
                     start_date=start_date, n_steps=40, on_exhausted="cycle",
                     num_runs=2, run_interval=run_interval)
        _run(cfg)
        out = Path(cfg.data.output_path)
        return [load_trajectories(str(out.with_stem(f"{out.stem}_r{i}")))
                for i in range(2)]

    r0, r1 = _pair("real", 12)
    t0, t1 = r0["time"], r1["time"]
    assert np.array_equal(t1 - t0, np.full_like(t0, 365 * DAY)), t1 - t0
    for key in ("lon", "lat", "z"):
        assert np.allclose(r1[key], r0[key], rtol=0, atol=1e-9), (
            key, np.max(np.abs(r1[key] - r0[key])))
    if start_date is None:
        r0, r1 = _pair("real13", 13)
        assert r1["time"][0] - r0["time"][0] == 394.5 * DAY, (
            r1["time"][0] - r0["time"][0])


def test_resume_across_the_seam_equals_the_uncut_run(tmp_path):
    """A run from 320 d (after the November centre) cut at a checkpoint
    after the seam (370 d) and resumed equals the uncut run bit for bit;
    both segments carry the exact period, and the uncut run equals the
    doubled reference from the same start."""
    mode = ModeConfig(Fdrift2D=False, Fdrift3D=True)
    num = NumericsConfig(dt=DAY, integrator="rk2", uv_interp="barycentric",
                         w_interp="barycentric", temporal_interp=True,
                         output_freq=1)
    t_start = 320 * DAY

    def _open(data):
        mesh = _quiet(Fesom14Reader.load_mesh,
                      MeshConfig(type="fesom14", meshpath=str(data)),
                      mode_cfg=mode)
        return mesh, _quiet(Fesom14Reader.open_dataset, str(data),
                            mode_cfg=mode)

    def _go(mesh, ds, name, state, n, step0, idx0, on_exhausted):
        ft, fdt = extract_time_info(ds)
        path = str(tmp_path / f"{name}.nc")
        w = TrajectoryWriter(path, state.n_particles).open()
        try:
            _quiet(tracking.run_single_realization,
                   mesh, ds, Fesom14Reader, state, mode, num,
                   start_time_idx=idx0, n_steps=n,
                   n_time_total=ds.sizes["time"], on_exhausted=on_exhausted,
                   writer=w, checkpoint_freq=10, temporal_interp=True,
                   file_times=ft, file_dt=fdt,
                   sim_time_start=t_start + step0 * DAY, start_step_i=step0)
        finally:
            w.close()
        return path

    mesh, ds = _open(_monthly_data(tmp_path / "year"))
    mesh2, ds2 = _open(_monthly_data(tmp_path / "twice", copies=2))
    lon, lat, z = (np.array(c, dtype=np.float64) for c in zip(*SEEDS))
    init = make_state(lon, lat, z)
    init = init.replace(nearest_node=find_nearest_nodes(mesh, lon, lat))
    try:
        whole = _go(mesh, ds, "whole", init, 60, 0, 10, "cycle")
        a = _go(mesh, ds, "a", init, 50, 0, 10, "cycle")
        r = load_restart(a)
        n = r["step_i"] + 1
        assert n == 50
        b = _go(mesh, ds, "b", r["state"], 10, n, r["file_idx"], "cycle")
        ref = _go(mesh2, ds2, "ref", init, 60, 0, 10, "stop")
    finally:
        ds.close()
        ds2.close()
    tw, ta, tb = (load_trajectories(p) for p in (whole, a, b))
    assert_bitwise(np.concatenate([ta["time"], tb["time"]]), tw["time"],
                   "time")
    for k in ("lon", "lat", "z", "status"):
        assert_bitwise(np.concatenate([ta[k], tb[k]], axis=1), tw[k], k)
    assert _period(whole) == 365 * DAY
    assert _period(b) == 365 * DAY
    want = load_trajectories(ref)
    _assert_doubled_reference(tw, want)
    assert np.array_equal(tw["time"], want["time"])
    assert np.ptp(tw["lon"]) > 0.01                       # it moves


@pytest.mark.parametrize("variant", ["prefetch", "pchip"])
def test_run_variants(tmp_path, monkeypatch, variant):
    """Prefetched reads: under ``cycle`` the prefetch is forced to one
    record, so the window path is pinned on the doubled ``stop`` reference
    (its windows carry the averaging windows) — the same run as without
    prefetch.  PCHIP in time with ``cycle``: the run goes through the seam
    (where the bracket falls back to linear) with every record finite and
    the exact period."""
    num = {"dt": DAY, "n_steps": 450}
    if variant == "prefetch":
        twice = _year_world(tmp_path, copies=2)
        kw = dict(calendar="noleap", n_rec=24, time_stamp="centre",
                  time_window="calendar_month")
        ref, _ = _case(tmp_path, twice, "ref", _month_stamps("centre", 2),
                       numerics=num, **kw)
        reader = get_reader("fesom2")
        read = reader.read_velocities
        windows = []

        def _spy(ds, timestep, *args, **kwargs):
            if ds.sizes["time"] <= 3:                     # a prefetch window
                windows.append(set(ds.coords))
            return read(ds, timestep, *args, **kwargs)

        with monkeypatch.context() as m:
            m.setattr(reader, "read_velocities", staticmethod(_spy))
            got, _ = _case(tmp_path, twice, "prefetch",
                           _month_stamps("centre", 2),
                           numerics={**num, "prefetch_size": 3}, **kw)
        _same_run(got.data.output_path, ref.data.output_path)
        from kelp.readers import TIME_WINDOWS
        assert windows and all(TIME_WINDOWS in c for c in windows), windows
        return
    world = _year_world(tmp_path)
    cfg, _ = _case(tmp_path, world, "pchip", _month_stamps("start"),
                   calendar="noleap", n_rec=12, time={"on_exhausted": "cycle"},
                   numerics={**num, "temporal_reconstruction": "pchip"},
                   time_stamp="start", time_window="calendar_month")
    got = load_trajectories(cfg.data.output_path)
    assert got["time"].shape == (450,)
    for key in ("time", "lon", "lat", "z"):
        assert np.isfinite(got[key]).all(), key
    assert _period(cfg.data.output_path) == 365 * DAY


def test_start_date_before_the_first_centre_is_refused(tmp_path):
    """1 January with monthly means lies before the first record time (the
    January centre): refused as a date outside the record times, as before
    the exact period (a cycled run starts at the first centre or later)."""
    data = _monthly_data(tmp_path / "data")
    cfg = _timed(_cfg(tmp_path, data=data, out="early.nc"),
                 start_date="2001-01-01T00:00", n_steps=2,
                 on_exhausted="cycle")
    with pytest.raises(ValueError) as info:
        _run(cfg)
    assert "start_date" in str(info.value), str(info.value)
    assert not Path(cfg.data.output_path).exists()


# ---------------------------------------------------------------------------
# equal windows and axes without known windows: the legacy period
# ---------------------------------------------------------------------------

#: records 3600.1 s apart (seconds since 1958-01-01)
FRAC = np.arange(6, dtype=np.float64) * 3600.1


def _legacy_run(tmp_path, case):
    """Run one legacy case; return its config."""
    cyc = {"on_exhausted": "cycle"}
    if case == "fesom14_daily":
        cfg = _timed(_cfg(tmp_path, data=_data(tmp_path / "daily"),
                          out=f"{case}.nc"), n_steps=9, on_exhausted="cycle")
        _run(cfg)
        return cfg
    if case == "mean_monthly":
        cfg, _ = _case(tmp_path, _year_world(tmp_path), case,
                       MONTH_EDGES[1:] * DAY, calendar="noleap", n_rec=12,
                       time=cyc, numerics={"dt": DAY, "n_steps": 400},
                       time_averaging="mean")
        return cfg
    world = _world(tmp_path)
    six_h = {"dt": DAY / 4, "n_steps": 32}
    if case == "day":
        cfg, _ = _case(tmp_path, world, case, STAMPS["start"], time=cyc,
                       numerics=six_h, time_stamp="start",
                       time_window="calendar_day")
    elif case == "frac_interp":
        cfg, _ = _case(tmp_path, world, case, FRAC, time=cyc,
                       numerics={"dt": 900.0, "n_steps": 32},
                       time_stamp="start", time_window="3600.1 s")
    elif case == "frac_discrete_back":
        cfg, _ = _case(tmp_path, world, case, FRAC, interp=False,
                       time={"on_exhausted": "cycle", "start_time_idx": 5},
                       numerics={"dt": 3600.1, "n_steps": 8},
                       mode={"time_dir": -1.0}, time_stamp="start",
                       time_window="3600.1 s")
    elif case == "mean_daily":
        cfg, _ = _case(tmp_path, world, case, STAMPS["end"], time=cyc,
                       numerics=six_h, time_averaging="mean")
    else:
        raise AssertionError(case)
    return cfg


@pytest.mark.parametrize("case", ["day", "frac_interp", "frac_discrete_back",
                                  "fesom14_daily", "mean_daily",
                                  "mean_monthly"])
def test_equal_windows_and_legacy_axes_keep_the_legacy_period(tmp_path, case):
    """Windows of equal length (calendar days, 3600.1 s, FESOM 1.4 daily
    bounds) and axes without known windows (``time_averaging: mean``, daily
    and monthly) keep the legacy period, the span of the run's records plus
    the last interval, computed as before; the dataset opened as the run
    opens it gives ``exact_cycle(...) is None``.  The runs come first and
    the new names are imported after them, so these outputs can be compared
    bit for bit with the code before the exact period."""
    cfg = _legacy_run(tmp_path, case)
    out = cfg.data.output_path
    # the run completes, and it is longer than one period (it wraps)
    assert load_trajectories(out)["time"].shape == (cfg.time.n_steps,)
    assert cfg.time.n_steps * cfg.numerics.dt > _period(out)
    from kelp.readers import _open_run_dataset, exact_cycle
    reader = get_reader(cfg.mesh.type)
    kw = ({"mode_cfg": cfg.mode}
          if "mode_cfg" in inspect.signature(reader.load_mesh).parameters
          else {})
    mesh = _quiet(reader.load_mesh, cfg.mesh, **kw)
    ds = _quiet(_open_run_dataset, reader, mesh, cfg)
    try:
        ft, _ = extract_time_info(ds)
        n = ds.sizes["time"]
        if case.startswith("mean_"):
            ft = tracking._shift_to_midpoints(ft)
        assert exact_cycle(ds, ft, n) is None
    finally:
        ds.close()
    assert len(ft) == n
    assert _period(out) == float(ft[-1] - ft[0] + (ft[-1] - ft[-2]))


def test_declared_file_dt_keeps_the_legacy_rule(tmp_path):
    """A direct caller that passes ``file_dt`` without ``file_times`` declares
    equally spaced records: the run synthesises that axis and cycles it with
    the legacy rule, whatever windows the dataset carries (the windows belong
    to the decoded axis, not to the declared one; a seam taken from them and
    the declared span would be meaningless, even negative)."""
    from kelp.readers import TIME_WINDOWS

    mode = ModeConfig(Fdrift2D=False, Fdrift3D=True)
    num = NumericsConfig(dt=DAY, integrator="rk2", uv_interp="barycentric",
                         w_interp="barycentric", temporal_interp=True,
                         output_freq=1)
    data = _monthly_data(tmp_path / "data")
    mesh = _quiet(Fesom14Reader.load_mesh,
                  MeshConfig(type="fesom14", meshpath=str(data)), mode_cfg=mode)
    ds = _quiet(Fesom14Reader.open_dataset, str(data), mode_cfg=mode)
    lon, lat, z = (np.array(c, dtype=np.float64) for c in zip(*SEEDS))
    init = make_state(lon, lat, z)
    init = init.replace(nearest_node=find_nearest_nodes(mesh, lon, lat))

    def _go(dset, name):
        path = str(tmp_path / f"{name}.nc")
        w = TrajectoryWriter(path, init.n_particles).open()
        try:
            _quiet(tracking.run_single_realization,
                   mesh, dset, Fesom14Reader, init, mode, num,
                   start_time_idx=0, n_steps=400,
                   n_time_total=dset.sizes["time"], on_exhausted="cycle",
                   writer=w, temporal_interp=True, file_dt=30 * DAY)
        finally:
            w.close()
        return path

    try:
        assert TIME_WINDOWS in ds.coords
        with_windows = _go(ds, "with_windows")
        without = _go(ds.drop_vars(TIME_WINDOWS), "without")
    finally:
        ds.close()
    a, b = load_trajectories(with_windows), load_trajectories(without)
    for k in ("time", "lon", "lat", "z", "status"):
        assert_bitwise(a[k], b[k], k)
    # twelve declared 30 d intervals: the legacy period of that axis
    assert _period(with_windows) == _period(without) == 360 * DAY
