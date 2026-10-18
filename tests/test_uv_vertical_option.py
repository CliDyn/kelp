"""mode.uv_vertical: layer-centred u/v evaluated linearly in depth.

The option (``"layer"``, the default, or ``"linear"``) reaches every read of
a run: the velocity cache marks each layer field ``"layer_linear"`` and
validates the record (prefetch windows and the shared snapshot cache
included); the HEALPix reader declares its u/v data prefix
(``mesh.metadata["uv_data_last"]``) and checks its raw records before it
zeroes them; ``max_levels`` cuts the mesh on every entry point of a
depth-linear run; the reader hook ``prepare(mesh, ds)`` runs whenever the
mesh is not prepared for this dataset and level table (a dataset switch or a
``max_levels`` re-cut prepares again); the output records the option
(``uv_vertical = "linear"``).  With the option off every run stays
bit-identical.

Fixtures: the 4-node FESOM2 square of ``tests/_fesom2_synth`` (levels 0, 10,
30 m, layer centres 5 and 20 m, bed 30 m) with u = A + B·z + C·(year −
1958) stored at the layer centres (dyadic, exact in float32), v = w = 0;
the synthetic HEALPix stores of ``tests/test_healpix_reader`` (levels 0,
10, 30, 70 m, layer centres 5, 20 and 50 m) with u = 0.1, 0.2, 0.3 m/s per
layer.  With w = 0 and u uniform in space, every position is a hand sum.
"""
from __future__ import annotations

import contextlib
import dataclasses
import io
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
netCDF4 = pytest.importorskip("netCDF4")

from kelp import tracking  # noqa: E402
from kelp.core.config import (AppConfig, DataConfig, MeshConfig,  # noqa: E402
                              ModeConfig, NumericsConfig, TimeConfig,
                              load_config)
from kelp.core.geometry import DEG2RAD, METERS_PER_DEG_LAT  # noqa: E402
from kelp.core.io import VelocityField  # noqa: E402
from kelp.core.particles import ParticleStatus, make_state  # noqa: E402
from kelp.numerics.vertical_uv import depth_points, is_checked  # noqa: E402
from kelp.readers import extract_time_info, get_reader  # noqa: E402
from kelp.writers import (TrajectoryWriter, load_restart,  # noqa: E402
                          load_trajectories)
from tests import _fesom2_synth as synth  # noqa: E402
from tests._bitwise import assert_bitwise  # noqa: E402

# ---------------------------------------------------------------------------
# FESOM2 fixture: u = A + B·z + C·(year - Y0) at the layer centres
# ---------------------------------------------------------------------------

A, B, C = 2.0 ** -4, 2.0 ** -9, 2.0 ** -6
Y0 = 1958
DT = 86400.0
Z_MID = (5.0, 20.0)                     # layer centres of levels 0, 10, 30 m
YEARS = {1958: 3, 1959: 3}               # records 0-2 and 3-5, daily
REC_YEAR = [1958, 1958, 1958, 1959, 1959, 1959]
# off the layer centres; 3 m above the first centre, 26 m below the last
SEEDS = [(0.10, 0.50, 3.0), (0.15, 0.40, 8.0), (0.12, 0.60, 12.5),
         (0.20, 0.30, 17.0), (0.08, 0.55, 26.0)]


def _u_layer(year, k):
    return A + B * Z_MID[k] + C * (year - Y0)


def _u_lin(year, z, centres=Z_MID):
    """The depth-linear rule for u = A + B·z: constant above the first and
    below the last centre."""
    return A + B * min(max(z, centres[0]), centres[-1]) + C * (year - Y0)


def _u_step(year, z):
    """The layer rule (option off): the value of the particle's layer."""
    return _u_layer(year, 0 if z < 10.0 else 1)


def _dlon(u, lat):
    return DT * u / (METERS_PER_DEG_LAT * np.cos(lat * DEG2RAD))


def _hand_euler(seeds, recs, u_of):
    """Euler positions after each step, ``(particle, step)``."""
    rows = []
    for lo, la, z in seeds:
        x, row = lo, []
        for r in recs:
            x = x + _dlon(u_of(REC_YEAR[r], z), la)
            row.append(x)
        rows.append(row)
    return np.array(rows)


def _write_fesom2(data, *, u_values=None, years=YEARS):
    data.mkdir(parents=True, exist_ok=True)
    zero = lambda y, k: 0.0                              # noqa: E731
    synth.write_multiyear(data, "unod", years, vertical="nz1",
                          values=u_values or _u_layer, continuous_time=True)
    synth.write_multiyear(data, "vnod", years, vertical="nz1", values=zero,
                          continuous_time=True)
    synth.write_multiyear(data, "w", years, vertical="nz", values=zero,
                          continuous_time=True)
    synth.write_multiyear(data, "ssh", years, vertical=None, values=zero,
                          continuous_time=True)


def _fesom2_mode(**kw):
    base = dict(Fdrift2D=False, Fdrift3D=True, var_u="unod", var_v="vnod",
                var_w="w", var_eta="ssh", uv_vertical="linear")
    base.update(kw)
    return ModeConfig(**base)


def _fesom2_cfg(work, *, seeds=SEEDS, n_steps=5, start=0, integrator="euler",
                prefetch=1, uv_interp="barycentric", u_values=None,
                out="traj.nc", data=None, num_runs=1, **mode_kw):
    """A run_tracking config on the FESOM2 fixture (written once per
    ``work``); discrete records, dt = 1 day."""
    work = Path(work)
    if data is None:
        data = work / "data"
        if not data.exists():
            _write_fesom2(data, u_values=u_values)
    mesh_path = work / "fesom.mesh.diag.nc"
    if not mesh_path.exists():
        synth.write_meshdiag(mesh_path, "ugrid")
    seeds_path = work / f"seeds_{out}.dat"
    seeds_path.write_text("".join(f"{lo} {la} {z}\n" for lo, la, z in seeds))
    return AppConfig(
        mode=_fesom2_mode(**mode_kw),
        numerics=NumericsConfig(dt=DT, integrator=integrator,
                                uv_interp=uv_interp, w_interp="barycentric",
                                prefetch_size=prefetch),
        mesh=MeshConfig(type="fesom2", meshpath=str(mesh_path)),
        data=DataConfig(ncinput=str(data), inicloud=str(seeds_path),
                        output_path=str(work / out),
                        time_averaging="instant"),
        time=TimeConfig(start_time_idx=start, n_steps=n_steps,
                        num_runs=num_runs),
    )


def _run(cfg):
    with contextlib.redirect_stdout(io.StringIO()):
        return tracking.run_tracking(cfg)


def _global_attrs(path):
    if str(path).endswith(".zarr"):
        import zarr
        return dict(zarr.open_group(str(path), mode="r").attrs)
    with netCDF4.Dataset(str(path)) as nc:
        return {k: nc.getncattr(k) for k in nc.ncattrs()}


#: writer configurations: streaming NetCDF, buffered NetCDF, Zarr
WRITER_KINDS = {"nc": (".nc", 0), "nc_buffered": (".nc", 2),
                "zarr": (".zarr", 0)}


def _writer(tmp_path, name, kind, n):
    ext, flush = WRITER_KINDS[kind]
    if ext == ".zarr":
        pytest.importorskip("zarr")
    return TrajectoryWriter(str(tmp_path / f"{name}{ext}"), n,
                            flush_every=flush)


def _fesom2_env(cfg):
    reader = get_reader("fesom2")
    mesh = reader.load_mesh(cfg.mesh)
    ds = reader.open_dataset(cfg.data.ncinput, mode_cfg=cfg.mode)
    return reader, mesh, ds


def _state(mesh, seeds):
    lon, lat, z = (np.array(c, dtype=np.float64) for c in zip(*seeds))
    st = make_state(lon, lat, z)
    from kelp.core.mesh import find_nearest_nodes
    return st.replace(nearest_node=find_nearest_nodes(mesh, st.lon, st.lat))


def _rsr(mesh, ds, reader, mode, seeds, *, n_steps, start=0,
         integrator="euler", **kw):
    numerics = NumericsConfig(dt=DT, integrator=integrator,
                              uv_interp="barycentric",
                              w_interp="barycentric")
    rec = tracking.run_single_realization(
        mesh, ds, reader, _state(mesh, seeds), mode, numerics,
        start_time_idx=start, n_steps=n_steps, **kw)
    return rec.to_arrays()


# ---------------------------------------------------------------------------
# HEALPix fixture: u = 0.1, 0.2, 0.3 m/s on the layers centred at 5, 20, 50 m
# ---------------------------------------------------------------------------

HP_U = (0.1, 0.2, 0.3)
HP_CENTRES = (5.0, 20.0, 50.0)


def _hp_u(z, centres=HP_CENTRES):
    """Depth-linear u of the HEALPix fixture at depth z."""
    return float(np.interp(z, centres, HP_U[:len(centres)]))


def _hp_store(path, *, nside=8, n_fin=None, deptho=False, w_full=False,
              order="nest", nan_at=None):
    """A HEALPix store with u per layer (HP_U), v = 0, w = 0.

    ``n_fin``: finite u layers per store cell (default 3 on wet cells);
    ``deptho``: add a 123 m bathymetry (deeper than every column);
    ``w_full``: w finite on all four interfaces of a wet cell (a w prefix
    longer than u's); ``nan_at``: ``(record, layer, cell, value)`` planted
    into u after the rewrite.
    """
    from tests.test_healpix_reader import (NLEV, _make_uv_store,
                                           _pix_lonlat, _wet)
    lon, lat = _pix_lonlat(nside, order)
    wet = _wet(lat)
    if n_fin is None:
        n_fin = np.where(wet, NLEV - 1, 0)
    raw = Path(str(path) + ".raw.nc")
    _make_uv_store(raw, nside=nside, order=order, with_eta=True,
                   with_w=True, w_value=0.0, n_finite_levels=n_fin,
                   with_deptho=deptho)
    ds = xr.open_dataset(raw).load()
    ds.close()
    u = ds["avg_uoe"].values.copy()
    for k in range(NLEV):
        fin = np.isfinite(u[:, k, :])
        u[:, k, :][fin] = HP_U[k] if k < len(HP_U) else 0.4
    if nan_at is not None:
        rec, k, cell, value = nan_at
        u[rec, k, cell] = value
    ds["avg_uoe"] = (("time", "level", "value"), u)
    if w_full:
        wv = ds["avg_wo"].values.copy()
        wv[:, :, wet] = 0.0
        ds["avg_wo"] = (("time", "level", "value"), wv)
    ds.to_netcdf(path)
    return Path(path)


def _hp_mesh_cfg(store, levels="default"):
    from tests.test_healpix_reader import LEVELS_M
    lev = LEVELS_M if levels == "default" else levels
    return MeshConfig(type="healpix", meshpath=str(store), levels=lev)


def _hp_mode(**kw):
    base = dict(Fdrift2D=False, Fdrift3D=True, uv_vertical="linear")
    base.update(kw)
    return ModeConfig(**base)


def _hp_cfg(work, store, seeds, *, n_steps=2, prefetch=1, integrator="euler",
            out="hp.nc", **mode_kw):
    work = Path(work)
    seeds_path = work / f"seeds_{out}.dat"
    seeds_path.write_text("".join(f"{lo} {la} {z}\n" for lo, la, z in seeds))
    return AppConfig(
        mode=_hp_mode(**mode_kw),
        numerics=NumericsConfig(dt=DT, integrator=integrator,
                                uv_interp="barycentric",
                                w_interp="barycentric",
                                prefetch_size=prefetch),
        mesh=_hp_mesh_cfg(store),
        data=DataConfig(ncinput=str(store), inicloud=str(seeds_path),
                        output_path=str(work / out),
                        time_averaging="instant"),
        time=TimeConfig(start_time_idx=0, n_steps=n_steps),
    )


def _hp_hand(seeds, n_steps, u_of):
    return np.array([[lo + (j + 1) * _dlon(u_of(z), la)
                      for j in range(n_steps)] for lo, la, z in seeds])


def _store_cell_of_nodes(mesh, nside, order):
    """The store cell of every mesh node, from the node's coordinates."""
    import healpy
    return healpy.ang2pix(nside, mesh.nodes_lon, mesh.nodes_lat,
                          nest=(order == "nest"), lonlat=True)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def test_yaml_key_read_validated_and_namespaced(tmp_path, monkeypatch):
    """``mode.uv_vertical`` is read from YAML (default "layer"), validated,
    and part of the shared-cache namespace (a run with the option never
    attaches a run without it)."""
    def cfg(extra, fdrift3d="true"):
        p = tmp_path / f"c{len(list(tmp_path.iterdir()))}.yaml"
        p.write_text(f"""
mode:
  Fdrift2D: false
  Fdrift3D: {fdrift3d}{extra}
numerics:
  dt: 3600.0
  uv_interp: barycentric
mesh:
  type: fesom2
  meshpath: mesh
data:
  ncinput: data
  inicloud: seeds
time:
  n_steps: 2
""")
        return load_config(str(p))
    assert cfg("").mode.uv_vertical == "layer"
    assert cfg("\n  uv_vertical: layer").mode.uv_vertical == "layer"
    on = cfg("\n  uv_vertical: linear")
    assert on.mode.uv_vertical == "linear"
    with pytest.raises(ValueError, match="uv_vertical"):
        cfg("\n  uv_vertical: cubic")
    with pytest.raises(ValueError, match="Fdrift3D"):
        cfg("\n  uv_vertical: linear", fdrift3d="false")
    monkeypatch.setenv("KELP_SHARED_FIELDS_GENERATION", "one-run")
    assert (tracking._shared_fields_tag(on)
            != tracking._shared_fields_tag(cfg("")))


# ---------------------------------------------------------------------------
# option off: unchanged
# ---------------------------------------------------------------------------

def test_option_off_bitwise(tmp_path):
    """The golden multi-year FESOM2 run, with the key absent and with
    ``uv_vertical: layer``: both equal the stored golden file, neither
    carries a ``uv_vertical`` attribute."""
    from tests._gen_golden_3d import _multiyear_config, _read_output_file
    from tests.test_golden_3d_runs import BITWISE, GOLDEN_DIR, _compare
    golden = GOLDEN_DIR / "run_tracking_fesom2_multiyear_node.npz"
    outs = []
    for name in ("absent", "layer"):
        d = tmp_path / name
        d.mkdir()
        cfg = _multiyear_config(d)
        if name == "layer":
            cfg.mode = ModeConfig(**dict(dataclasses.asdict(cfg.mode),
                                         uv_vertical="layer"))
        _run(cfg)
        out = _read_output_file(cfg.data.output_path)
        assert "attr__uv_vertical" not in out
        _compare(out, golden, BITWISE)
        outs.append(out)
    for k in outs[0]:
        if not k.startswith(("attr__", "dims__")):
            assert_bitwise(outs[1][k], outs[0][k], k)


# ---------------------------------------------------------------------------
# option on: FESOM2 node layout
# ---------------------------------------------------------------------------

def test_option_on_fesom2_node_run(tmp_path):
    """Euler, one record per step, a year file crossed at step 3: every
    position is the hand sum of dt·u(record, z0) with the depth-linear u;
    the layer rule gives other positions; the output says
    ``uv_vertical = "linear"``."""
    cfg = _fesom2_cfg(tmp_path)
    _run(cfg)
    traj = load_trajectories(cfg.data.output_path)
    hand = _hand_euler(SEEDS, range(5), _u_lin)
    np.testing.assert_allclose(traj["lon"], hand, rtol=0, atol=1e-11)
    np.testing.assert_array_equal(
        traj["lat"], np.repeat([[s[1]] for s in SEEDS], 5, axis=1))
    np.testing.assert_array_equal(
        traj["z"], np.repeat([[s[2]] for s in SEEDS], 5, axis=1))
    assert _global_attrs(cfg.data.output_path)["uv_vertical"] == "linear"
    # the option matters: the layer rule moves the off-centre seeds elsewhere
    off = _fesom2_cfg(tmp_path, uv_vertical="layer", out="off.nc")
    _run(off)
    t_off = load_trajectories(off.data.output_path)
    np.testing.assert_allclose(t_off["lon"],
                               _hand_euler(SEEDS, range(5), _u_step),
                               rtol=0, atol=1e-11)
    assert not np.allclose(t_off["lon"][1:4], traj["lon"][1:4], atol=1e-6)
    assert "uv_vertical" not in _global_attrs(off.data.output_path)


def test_option_survives_prefetch_window_and_year(tmp_path, monkeypatch):
    """Prefetch windows of 3 (one window spans the 1958/1959 files), RK2
    with the next record: every step function call receives
    ``"layer_linear"`` fields, the records of the right year, and the
    positions are the hand sums of dt·(u(rec) + u(rec+1))/2."""
    from kelp.numerics import integrators
    cfg = _fesom2_cfg(tmp_path, integrator="rk2", prefetch=3, start=1,
                      n_steps=4)
    loads = []
    orig = tracking._VelCache._load_window

    def counting(self, start, *a, **k):
        loads.append(start)
        return orig(self, start, *a, **k)
    monkeypatch.setattr(tracking._VelCache, "_load_window", counting)
    seen = []

    def spy(state, vel, mesh, dt, time_dir, **kw):
        vn = kw.get("vel_next")
        seen.append((vel.uv_vertical, getattr(vn, "uv_vertical", None),
                     float(vel.u[0, 0]),
                     None if vn is None else float(vn.u[0, 0])))
        return integrators.rk2_step(state, vel, mesh, dt, time_dir, **kw)
    monkeypatch.setattr(tracking, "get_integrator", lambda name: spy)
    _run(cfg)
    assert len(loads) > 0
    assert len(seen) == 4
    recs = (1, 2, 3, 4)
    for (pv, pn, u_vel, u_next), r in zip(seen, recs):
        assert pv == "layer_linear" and pn == "layer_linear"
        assert u_vel == _u_layer(REC_YEAR[r], 0)
        assert u_next == _u_layer(REC_YEAR[r + 1], 0)
    traj = load_trajectories(cfg.data.output_path)
    hand = []
    for lo, la, z in SEEDS:
        x, row = lo, []
        for r in recs:
            u = 0.5 * (_u_lin(REC_YEAR[r], z) + _u_lin(REC_YEAR[r + 1], z))
            x = x + _dlon(u, la)
            row.append(x)
        hand.append(row)
    np.testing.assert_allclose(traj["lon"], np.array(hand), rtol=0,
                               atol=1e-11)


def test_option_survives_shared_cache(tmp_path):
    """Shared snapshot cache, prefetch 1: the second cache attaches without
    calling its reader, its field is ``"layer_linear"`` and checked, bit for
    bit the field of a local read; no fallback warning."""
    from kelp.core.shared_fields import SharedFieldCache
    cfg = _fesom2_cfg(tmp_path)
    reader, mesh, ds = _fesom2_env(cfg)

    class Counting:
        def __init__(self):
            self.calls = 0

        def read_velocities(self, *a, **k):
            self.calls += 1
            return reader.read_velocities(*a, **k)
    shm = tmp_path / "shm"
    shm.mkdir()
    r1, r2 = Counting(), Counting()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        c1 = tracking._VelCache(shared=SharedFieldCache(shm, "tagDL"))
        v1 = c1.get(3, r1, ds, mesh, cfg.mode)
        c2 = tracking._VelCache(shared=SharedFieldCache(shm, "tagDL"))
        v2 = c2.get(3, r2, ds, mesh, cfg.mode)
    assert (r1.calls, r2.calls) == (1, 0)
    assert not any("shared_fields" in str(w.message) for w in rec)
    local = tracking._VelCache().get(3, reader, ds, mesh, cfg.mode)
    for v in (v1, v2, local):
        assert v.uv_vertical == "layer_linear"
        assert is_checked(v, mesh)
    for f in ("u", "v", "w", "eta"):
        assert_bitwise(getattr(v2, f), getattr(local, f), f)
    assert c2.get(3, r2, ds, mesh, cfg.mode) is v2       # the stored object
    ds.close()


def test_record_check_refuses_nan_in_prefix_fesom2(tmp_path):
    """A NaN above the bed (node 1, layer 0) in record 2 refuses the run;
    the message names the node, the point and ``time index 2`` — with and
    without prefetch windows."""
    def u_vals(year, k):
        out = np.full((3, 4), _u_layer(year, k))
        if year == 1958 and k == 0:
            out[2, 1] = np.nan
        return out
    for prefetch in (1, 3):
        cfg = _fesom2_cfg(tmp_path / f"p{prefetch}", u_values=u_vals,
                          prefetch=prefetch)
        with pytest.raises(ValueError,
                           match=r"node 1, point 0 .*time index 2"):
            _run(cfg)


def test_record_check_accepts_fill_below_last_fesom2(tmp_path, monkeypatch):
    """Node 1's bed at 10 m leaves it one usable point (5 m): NaN (1958)
    and a raw 1e30 fill (1959) in its second layer run, finite, and bit for
    bit like finite values there — the slot is never read."""
    monkeypatch.setattr(synth, "ZBAR_N_BOTTOM_NEG",
                        np.array([-30.0, -10.0, -30.0, -30.0]))

    def u_vals(fill):
        def f(year, k):
            out = np.full((3, 4), _u_layer(year, k))
            if k == 1:
                out[:, 1] = fill if fill is not None else (
                    np.nan if year == 1958 else 1e30)
            return out
        return f
    seeds = [(0.55, 0.15, 7.0), (0.6, 0.25, 12.0), (0.5, 0.3, 18.0)]
    outs = []
    for name, fill in (("fill", None), ("finite", 0.25)):
        cfg = _fesom2_cfg(tmp_path / name, u_values=u_vals(fill),
                          seeds=seeds, n_steps=5, integrator="rk2")
        _run(cfg)
        outs.append(load_trajectories(cfg.data.output_path))
    t = outs[0]
    assert np.isfinite(t["lon"]).all() and np.isfinite(t["vel_u"]).all()
    assert (t["status"] == ParticleStatus.ALIVE).all()
    for k in ("lon", "lat", "z", "vel_u", "vel_v"):
        assert_bitwise(outs[0][k], outs[1][k], k)


def test_run_single_realization_and_track_particles_option(tmp_path):
    """The option through the functional entry points (mode argument):
    ``track_particles`` and ``run_single_realization`` agree bit for bit
    and follow the hand sums; the option off gives the layer rule."""
    cfg = _fesom2_cfg(tmp_path)
    reader, mesh, ds = _fesom2_env(cfg)
    lon0, lat0, z0 = (np.array(c, dtype=np.float64) for c in zip(*SEEDS))
    tp = tracking.track_particles(
        mesh, ds, reader, lon0, lat0, z0, start_time_idx=0, n_steps=5,
        dt=DT, integrator="euler", mode=_fesom2_mode(),
        uv_interp="barycentric", w_interp="barycentric").to_arrays()
    rs = _rsr(mesh, ds, reader, _fesom2_mode(), SEEDS, n_steps=5)
    for k in ("lon", "lat", "z", "vel_u"):
        assert_bitwise(rs[k], tp[k], k)
    np.testing.assert_allclose(tp["lon"].T, _hand_euler(SEEDS, range(5),
                                                        _u_lin),
                               rtol=0, atol=1e-11)
    off = _rsr(mesh, ds, reader, _fesom2_mode(uv_vertical="layer"), SEEDS,
               n_steps=5)
    np.testing.assert_allclose(off["lon"].T, _hand_euler(SEEDS, range(5),
                                                         _u_step),
                               rtol=0, atol=1e-11)
    ds.close()


def _restart_env(tmp_path, output_freq=1):
    """The FESOM2 fixture opened for direct calls, RK2 with the next record,
    and a runner writing one output (checkpoint every 2 steps)."""
    cfg = _fesom2_cfg(tmp_path)
    reader, mesh, ds = _fesom2_env(cfg)
    file_times, file_dt = extract_time_info(ds)
    numerics = NumericsConfig(dt=DT, integrator="rk2",
                              uv_interp="barycentric",
                              w_interp="barycentric",
                              output_freq=output_freq)

    def run(name, mode, state, start, n, start_step_i, writer=None,
            ext=".nc"):
        path = str(tmp_path / f"{name}{ext}")
        w = writer or TrajectoryWriter(path, state.n_particles).open()
        try:
            tracking.run_single_realization(
                mesh, ds, reader, state, mode, numerics,
                start_time_idx=start, n_steps=n, writer=w,
                checkpoint_freq=2, file_times=file_times, file_dt=file_dt,
                start_step_i=start_step_i)
        finally:
            if writer is None:
                w.close()
        return path
    return mesh, ds, run


@pytest.mark.parametrize("option", ["linear", "layer"])
def test_restart_continuous_equals_resumed_option_on(tmp_path, option):
    """A run cut at a checkpoint and resumed under the same rule (the
    validator passes) equals the continuous run bit for bit: record times,
    every recorded array, the final restart step and record index.  The
    checkpoint carries the rule: ``"linear"`` with the option, nothing
    (``None``, no attribute) without it."""
    mesh, ds, run = _restart_env(tmp_path)
    mode = _fesom2_mode(uv_vertical=option)
    init = _state(mesh, SEEDS)
    whole = run("whole", mode, init, 0, 4, 0)
    a = run("a", mode, init, 0, 2, 0)
    r = load_restart(a)
    assert r["uv_vertical"] == ("linear" if option == "linear" else None)
    tracking._validate_restart_uv_vertical(r["uv_vertical"], mode)
    n = r["step_i"] + 1
    b = run("b", mode, r["state"], r["file_idx"] + 1, 2, n)
    tw, ta, tb = (load_trajectories(p) for p in (whole, a, b))
    assert_bitwise(np.concatenate([ta["time"], tb["time"]]), tw["time"],
                   "time")
    for k in ("lon", "lat", "z", "status", "vel_u", "vel_v", "vel_w"):
        assert_bitwise(np.concatenate([ta[k], tb[k]], axis=1), tw[k], k)
    rw, rb = load_restart(whole), load_restart(b)
    assert (rb["step_i"], rb["file_idx"]) == (rw["step_i"], rw["file_idx"])
    assert rb["uv_vertical"] == rw["uv_vertical"] == r["uv_vertical"]
    for k in ("lon", "lat", "z"):
        assert_bitwise(getattr(rb["state"], k), getattr(rw["state"], k), k)
    attrs = _global_attrs(whole)
    if option == "linear":
        assert attrs["uv_vertical"] == "linear"
    else:
        assert not any("uv_vertical" in k for k in attrs)
    ds.close()


@pytest.mark.parametrize("first,second", [("linear", "layer"),
                                          ("layer", "linear")])
def test_restart_with_the_other_rule_refused(tmp_path, first, second):
    """A checkpoint written under one ``mode.uv_vertical`` does not resume
    under the other: the validator names both rules before anything of the
    new segment is written; the same rule resumes."""
    mesh, ds, run = _restart_env(tmp_path)
    a = run("a", _fesom2_mode(uv_vertical=first), _state(mesh, SEEDS), 0, 2,
            0)
    r = load_restart(a)
    with pytest.raises(ValueError,
                       match=rf"written with mode.uv_vertical '{first}', "
                             rf"this run has '{second}'"):
        tracking._validate_restart_uv_vertical(
            r["uv_vertical"], _fesom2_mode(uv_vertical=second))
    tracking._validate_restart_uv_vertical(r["uv_vertical"],
                                           _fesom2_mode(uv_vertical=first))
    ds.close()


@pytest.mark.parametrize("option", ["linear", "layer"])
def test_zarr_checkpoint_carries_the_rule(tmp_path, option):
    """The Zarr backend: the checkpoint's rule comes back from
    ``load_restart`` like the NetCDF one (``None`` without the option)."""
    pytest.importorskip("zarr")
    mesh, ds, run = _restart_env(tmp_path)
    path = run("z", _fesom2_mode(uv_vertical=option), _state(mesh, SEEDS),
               0, 2, 0, ext=".zarr")
    assert load_restart(path)["uv_vertical"] == (
        "linear" if option == "linear" else None)
    ds.close()


@pytest.mark.parametrize("first,second", [("linear", "layer"),
                                          ("layer", "linear")])
def test_one_writer_keeps_one_rule(tmp_path, first, second):
    """Two segments written through ONE writer must share the rule: the
    second segment with the other rule is refused before it writes a
    record (the file stays one rule); with the same rule both segments
    go into the file."""
    mesh, ds, run = _restart_env(tmp_path)
    init = _state(mesh, SEEDS)
    # the state after the first segment, from a separate output (the chain
    # file stays open while it is written)
    r = load_restart(run("a", _fesom2_mode(uv_vertical=first), init, 0, 2,
                         0))
    path = str(tmp_path / "chain.nc")
    w = TrajectoryWriter(path, init.n_particles).open()
    try:
        run("seg1", _fesom2_mode(uv_vertical=first), init, 0, 2, 0, writer=w)
        n_before = w.n_obs
        assert n_before == 2
        with pytest.raises(ValueError,
                           match=rf"mode.uv_vertical '{first}', this "
                                 rf"segment has '{second}'"):
            run("seg2", _fesom2_mode(uv_vertical=second), r["state"],
                r["file_idx"] + 1, 2, r["step_i"] + 1, writer=w)
        assert w.n_obs == n_before
        run("seg2", _fesom2_mode(uv_vertical=first), r["state"],
            r["file_idx"] + 1, 2, r["step_i"] + 1, writer=w)
        assert w.n_obs == 4
    finally:
        w.close()
    attrs = _global_attrs(path)
    assert attrs.get("uv_vertical") == (
        "linear" if first == "linear" else None)
    ds.close()


@pytest.mark.parametrize("kind", sorted(WRITER_KINDS))
@pytest.mark.parametrize("first,second", [("layer", "linear"),
                                          ("linear", "layer")])
def test_writer_bound_by_a_segment_without_records(tmp_path, kind, first,
                                                   second):
    """A segment that wrote a checkpoint but no record (output every 3
    steps, checkpoint every 2, two steps) binds the writer to its rule all
    the same: the next segment with the other rule is refused before it
    writes anything, on every writer kind."""
    mesh, ds, run = _restart_env(tmp_path, output_freq=3)
    init = _state(mesh, SEEDS)
    r = load_restart(run("a", _fesom2_mode(uv_vertical=first), init, 0, 2,
                         0))
    w = _writer(tmp_path, "chain", kind, init.n_particles).open()
    try:
        run("seg1", _fesom2_mode(uv_vertical=first), init, 0, 2, 0, writer=w)
        assert w.n_obs == 0
        with pytest.raises(ValueError,
                           match=rf"mode.uv_vertical '{first}', this "
                                 rf"segment has '{second}'"):
            run("seg2", _fesom2_mode(uv_vertical=second), r["state"],
                r["file_idx"] + 1, 2, r["step_i"] + 1, writer=w)
        assert w.n_obs == 0
        # the same rule continues
        run("seg2", _fesom2_mode(uv_vertical=first), r["state"],
            r["file_idx"] + 1, 2, r["step_i"] + 1, writer=w)
    finally:
        w.close()
    assert ("uv_vertical" in _global_attrs(w.path)) == (first == "linear")
    ds.close()


@pytest.mark.parametrize("kind", sorted(WRITER_KINDS))
@pytest.mark.parametrize("first,second", [("linear", "layer"),
                                          ("layer", "linear")])
def test_reopened_writer_starts_a_new_file(tmp_path, kind, first, second):
    """``open()`` creates a new file: a writer object reused after
    ``close()`` forgets the rule (and the attributes) of the run before, so
    a run with the other rule is accepted, and the new file carries the
    ``uv_vertical`` attribute only when that run has the option."""
    mesh, ds, run = _restart_env(tmp_path)
    init = _state(mesh, SEEDS)
    w = _writer(tmp_path, "reused", kind, init.n_particles)
    w.open()
    try:
        run("r1", _fesom2_mode(uv_vertical=first), init, 0, 2, 0, writer=w)
    finally:
        w.close()
    w.open()
    try:
        run("r2", _fesom2_mode(uv_vertical=second), init, 0, 2, 0, writer=w)
        assert w.n_obs == 2
    finally:
        w.close()
    attrs = _global_attrs(w.path)
    assert attrs.get("uv_vertical") == (
        "linear" if second == "linear" else None)
    ds.close()


def test_standalone_restart_file_carries_the_rule(tmp_path):
    """``write_restart_file(uv_vertical=...)`` stores ``"linear"`` as
    ``restart_uv_vertical`` and ``load_restart`` returns it; ``"layer"``
    and ``None`` store nothing (the files are attribute-identical to a call
    without the argument) and come back as ``None``."""
    from kelp.writers import write_restart_file
    st = make_state(np.array([0.1, 0.2]), np.array([0.3, 0.4]),
                    np.array([5.0, 6.0]))
    paths = {}
    for name, kw in (("lin", {"uv_vertical": "linear"}),
                     ("lay", {"uv_vertical": "layer"}),
                     ("none", {"uv_vertical": None}), ("plain", {})):
        paths[name] = str(tmp_path / f"{name}.nc")
        write_restart_file(paths[name], st, 3, 4, **kw)
    assert load_restart(paths["lin"])["uv_vertical"] == "linear"
    assert _global_attrs(paths["lin"])["restart_uv_vertical"] == "linear"
    plain = _global_attrs(paths["plain"])
    for name in ("lay", "none", "plain"):
        assert load_restart(paths[name])["uv_vertical"] is None
        assert _global_attrs(paths[name]) == plain
    with pytest.raises(ValueError, match="'linear', this run has 'layer'"):
        tracking._validate_restart_uv_vertical(
            load_restart(paths["lin"])["uv_vertical"],
            _fesom2_mode(uv_vertical="layer"))


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------

class TestRefusalsAtEachSite:
    """Each unsupported combination is refused where its facts are known,
    before any output file exists."""

    def test_config_2d_and_value(self):
        with pytest.raises(ValueError, match="Fdrift3D"):
            ModeConfig(Fdrift2D=True, Fdrift3D=False, uv_vertical="linear")
        with pytest.raises(ValueError, match="uv_vertical"):
            ModeConfig(Fdrift2D=False, Fdrift3D=True, uv_vertical="level")
        assert ModeConfig().uv_vertical == "layer"

    def test_uv_interp_after_load_mesh(self, tmp_path):
        cfg = _fesom2_cfg(tmp_path, uv_interp="leastsq")
        cfg.data.ncinput = str(tmp_path / "missing")   # never opened
        with pytest.raises(ValueError, match="barycentric"):
            _run(cfg)
        assert not Path(cfg.data.output_path).exists()

    def test_sigma_mesh_after_load_mesh(self, tmp_path):
        # a FESOM-C file of the w-contract fixture (sigma coordinates)
        from tests.test_fesomc_w_contract import SIG_U, _grid, _write
        n_t = 3
        lon, lat, _tris, _nv = _grid()
        k = SIG_U.size
        path = _write(tmp_path / "fesomc.nc", np.arange(n_t) * 3600.0,
                      np.zeros((n_t, lon.size)),
                      np.zeros((n_t, k, lon.size)))
        seeds = tmp_path / "s.dat"
        seeds.write_text(f"{lon.mean()} {lat.mean()} 2.0\n")
        cfg = AppConfig(
            mode=ModeConfig(Fdrift2D=False, Fdrift3D=True,
                            uv_vertical="linear"),
            numerics=NumericsConfig(dt=3600.0, uv_interp="barycentric"),
            mesh=MeshConfig(type="fesomc", meshpath=str(path)),
            data=DataConfig(ncinput=str(tmp_path / "missing.nc"),
                            inicloud=str(seeds),
                            output_path=str(tmp_path / "o.nc"),
                            time_averaging="instant"),
            time=TimeConfig(start_time_idx=0, n_steps=2))
        with pytest.raises(ValueError, match="z-level"):
            _run(cfg)
        assert not Path(cfg.data.output_path).exists()

    def test_element_layout_at_the_pre_writer_probe(self, tmp_path):
        data = tmp_path / "data"
        data.mkdir()
        for var in ("u", "v"):
            synth.write_multiyear(data, var, YEARS, node_layout=False,
                                  vertical="nz1", continuous_time=True)
        synth.write_multiyear(data, "w", YEARS, vertical="nz",
                              continuous_time=True,
                              values=lambda y, k: 0.0)
        synth.write_multiyear(data, "ssh", YEARS, vertical=None,
                              continuous_time=True)
        cfg = _fesom2_cfg(tmp_path, data=data, var_u="u", var_v="v")
        probes = []
        orig = get_reader("fesom2").probe_uv_layout

        class Probing(get_reader("fesom2")):
            @staticmethod
            def probe_uv_layout(ds, mode_cfg, mesh):
                probes.append(orig(ds, mode_cfg, mesh))
                return probes[-1]
        import kelp.tracking as trk
        old = trk.get_reader
        trk.get_reader = lambda name: Probing
        try:
            with pytest.raises(ValueError, match="node layout"):
                _run(cfg)
        finally:
            trk.get_reader = old
        assert probes == ["element"]
        assert not Path(cfg.data.output_path).exists()

    def test_step0_backstop(self):
        tracking._validate_uv_compat("element", "barycentric")
        tracking._validate_uv_compat("node", "barycentric",
                                     uv_vertical_option="linear")
        with pytest.raises(ValueError, match="node layout"):
            tracking._validate_uv_compat("element", "barycentric",
                                         uv_vertical_option="linear")

    def test_direct_call_refused_before_any_read(self, tmp_path):
        cfg = _fesom2_cfg(tmp_path)
        reader, mesh, ds = _fesom2_env(cfg)

        class Never:
            def read_velocities(self, *a, **k):
                raise AssertionError("read before the refusal")
        numerics = NumericsConfig(dt=DT, integrator="euler",
                                  uv_interp="leastsq")
        with pytest.raises(ValueError, match="barycentric"):
            tracking.run_single_realization(
                mesh, ds, Never(), _state(mesh, SEEDS), _fesom2_mode(),
                numerics, start_time_idx=0, n_steps=2)
        ds.close()

    def test_mark_depth_linear(self):
        n, k = 5, 4
        rng = np.random.default_rng(3)
        layer = VelocityField(u=rng.random((k - 1, n)),
                              v=rng.random((k - 1, n)), eta=np.zeros(n),
                              w=rng.random((k, n)), uv_layout="node")
        level = VelocityField(u=rng.random((k, n)), v=rng.random((k, n)),
                              eta=np.zeros(n), w=rng.random((k, n)),
                              uv_layout="node", uv_vertical="level")
        flat = VelocityField(u=rng.random(n), v=rng.random(n),
                             eta=np.zeros(n), uv_layout="node")
        elem = VelocityField(u=rng.random((k - 1, 3)),
                             v=rng.random((k - 1, 3)), eta=np.zeros(n),
                             w=rng.random((k, n)))
        mark = tracking._mark_depth_linear
        assert mark(layer, "layer", None, record="r") is layer
        assert layer._dl_token is None
        with pytest.raises(ValueError, match="level"):
            mark(level, "linear", None, record="r")
        with pytest.raises(ValueError, match="3-D"):
            mark(flat, "linear", None, record="r")
        with pytest.raises(ValueError, match="node layout"):
            mark(elem, "linear", None, record="r")
        with pytest.raises(ValueError, match="uv_vertical"):
            mark(layer, "cubic", None, record="r")
        ll = dataclasses.replace(layer, uv_vertical="layer_linear")
        with pytest.raises(ValueError, match="layer_linear"):
            mark(ll, "layer", None, record="r")


def test_mark_depth_linear_semantics():
    """layer → a shallow copy marked "layer_linear" (same arrays) and
    checked; a marked field comes back as itself; a level field comes back
    as itself, checked, with the option off (the option on refuses it:
    ``TestRefusalsAtEachSite``); NaN in a usable point is refused with the
    record named."""
    mesh = _zmesh()
    n = mesh.n_nodes
    u = np.tile(np.array([[1.0], [2.0], [3.0]]), (1, n))
    layer = VelocityField(u=u, v=u * 0.5, eta=np.zeros(n),
                          w=np.zeros((4, n)), uv_layout="node")
    mark = tracking._mark_depth_linear
    out = mark(layer, "linear", mesh, record="time index 7")
    assert out is not layer and out.uv_vertical == "layer_linear"
    assert out.u is layer.u and out.v is layer.v and out.w is layer.w
    assert layer.uv_vertical == "layer" and layer._dl_token is None
    assert is_checked(out, mesh)
    assert mark(out, "linear", mesh, record="x") is out
    level = VelocityField(u=np.zeros((4, n)), v=np.zeros((4, n)),
                          eta=np.zeros(n), w=np.zeros((4, n)),
                          uv_layout="node", uv_vertical="level")
    got = mark(level, "layer", mesh, record="r")
    assert got is level and is_checked(level, mesh)
    bad_u = u.copy()
    bad_u[0, 2] = np.nan
    bad = VelocityField(u=bad_u, v=u, eta=np.zeros(n), w=np.zeros((4, n)),
                        uv_layout="node")
    with pytest.raises(ValueError, match=r"node 2, point 0 .*time index 7"):
        mark(bad, "linear", mesh, record="time index 7")
    # a lazy blend: both parents marked and checked, nothing blended
    from kelp.core.io import LazyLerpField
    other = dataclasses.replace(layer, u=u + 1.0)
    lazy = mark(LazyLerpField(layer, other, 0.25), "linear", mesh,
                record="r")
    assert isinstance(lazy, LazyLerpField)
    assert lazy.uv_vertical == "layer_linear"
    assert lazy.vel_a.u is layer.u and lazy.vel_b.u is other.u
    assert is_checked(lazy, mesh) and "u" not in lazy.__dict__


def _zmesh():
    """A 4-node z-level mesh (no search structures: the record check and
    the max_levels cut need none) with levels 0, 10, 30, 70 m, bed 70 m."""
    from kelp.core.mesh import Mesh
    n = 4
    return Mesh(
        name="z", nodes_lon=np.arange(n, dtype=np.float64),
        nodes_lat=np.zeros(n), depth=np.full(n, 70.0),
        elem_nodes=np.array([[0, 1, 2]], dtype=np.int32),
        elem_center_lon=np.array([1.0]), elem_center_lat=np.array([0.0]),
        node_to_elems=np.zeros((n, 1), dtype=np.int32),
        node_to_elems_count=np.ones(n, dtype=np.int32),
        sigma=np.array([0.0, -10.0, -30.0, -70.0]))


def test_level_reader_with_option_refused_before_writer(tmp_path,
                                                        monkeypatch):
    """A reader whose data sit on levels (``native_uv_vertical =
    "level"``) is always depth-linear: the option on such a reader is
    refused right after load_mesh, before any output exists."""
    base = get_reader("fesom2")
    level_reader = type("LevelReader", (base,),
                        {"native_uv_vertical": "level"})
    opened = []
    real_open = base.open_dataset

    def spy_open(*a, **k):
        opened.append(1)
        return real_open(*a, **k)
    level_reader.open_dataset = staticmethod(spy_open)
    monkeypatch.setattr(tracking, "get_reader", lambda name: level_reader)
    cfg = _fesom2_cfg(tmp_path)
    with pytest.raises(ValueError, match="level"):
        _run(cfg)
    assert opened == []
    assert not Path(cfg.data.output_path).exists()
    # the same reader with the option off passes this site
    cfg_off = _fesom2_cfg(tmp_path, uv_vertical="layer", out="off.nc")
    _run(cfg_off)
    assert opened == [1] and Path(cfg_off.data.output_path).exists()


# ---------------------------------------------------------------------------
# HEALPix
# ---------------------------------------------------------------------------

healpy = pytest.importorskip("healpy")


def test_healpix_u_w_prefix_mismatch_option_on(tmp_path):
    """u finite on 2 layers, w on all 4 interfaces, ``deptho`` (123 m)
    deeper than both: the declaration caps the table at the u prefix
    (``uv_data_last == 1``) and a particle at 40 m moves with layer 1's u,
    never with the zero the reader writes into layer 2."""
    from tests.test_healpix_reader import _pix_lonlat, _wet
    lon, lat = _pix_lonlat(8, "nest")
    store = _hp_store(tmp_path / "3D_mis_ocean.nc",
                      n_fin=np.where(_wet(lat), 2, 0), deptho=True,
                      w_full=True)
    mode = _hp_mode()
    from kelp.readers.healpix import HealpixReader
    mesh = HealpixReader.load_mesh(_hp_mesh_cfg(store), mode_cfg=mode)
    decl = mesh.metadata["uv_data_last"]
    assert decl.dtype == np.int32 and decl.shape == (mesh.n_nodes,)
    np.testing.assert_array_equal(decl, 1)
    np.testing.assert_allclose(mesh.depth, 123.0)
    assert depth_points(mesh, "layer_linear").last.max() == 1
    seeds = [(10.0, 5.0, 40.0), (-40.0, -20.0, 12.5)]
    cfg = _hp_cfg(tmp_path, store, seeds)
    _run(cfg)
    traj = load_trajectories(cfg.data.output_path)
    np.testing.assert_allclose(
        traj["lon"], _hp_hand(seeds, 2, lambda z: _hp_u(z, (5.0, 20.0))),
        rtol=0, atol=1e-9)
    # without the declaration the reader refuses the raw NaN slot before
    # it zeroes it — the zero never becomes data
    del mesh.metadata["uv_data_last"]
    ds = HealpixReader.open_dataset(str(store), mode_cfg=mode)
    with pytest.raises(ValueError, match=r"point 2"):
        HealpixReader.read_velocities(ds, 0, mesh, mode_cfg=mode)
    ds.close()


#: zarr 3 warns that consolidated metadata is outside its specification
#: when the extract tool writes the sector store; only that case writes one
_ZARR_CONSOLIDATED = pytest.mark.filterwarnings(
    "ignore:Consolidated metadata is currently not part in the Zarr format 3"
    ":UserWarning")


@pytest.mark.parametrize("kind", ["nest", "ring", pytest.param(
    "extract", marks=_ZARR_CONSOLIDATED)])
def test_healpix_prefix_in_mesh_order(tmp_path, kind):
    """Prefixes of 1, 2 or 3 layers that vary from cell to cell: the
    declaration of every mesh node is the prefix of its OWN column (the
    store cell found from the node's coordinates), on a NEST store, a RING
    store and a regional extract."""
    from kelp.readers.healpix import HealpixReader
    from tests.test_healpix_reader import _make_uv_store, _pix_lonlat, _wet
    nside = 8
    order = "ring" if kind == "ring" else "nest"
    lon, lat = _pix_lonlat(nside, order)
    cell = np.arange(lon.size)
    n_fin = np.where(_wet(lat), 1 + (cell * 7 + lon.astype(int)) % 3, 0)
    store = _make_uv_store(tmp_path / "3D_pref_ocean.nc", nside=nside,
                           order=order, with_eta=True, with_w=True,
                           n_finite_levels=n_fin)
    mesh_cfg = _hp_mesh_cfg(store)
    if kind == "extract":
        pytest.importorskip("zarr")
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
        from extract_healpix_region import extract_region
        out = tmp_path / "sector.zarr"
        from tests.test_healpix_reader import LEVELS_M
        extract_region({
            "source": {"path": str(store), "levels": list(LEVELS_M)},
            "region": {"bbox": [-30.0, 30.0, -30.0, 30.0],
                       "halo_layers": 1},
            "variables": ["avg_uoe", "avg_von", "avg_wo"],
            "output": {"path": str(out), "time_chunk": 2},
        })
        mesh_cfg = _hp_mesh_cfg(out, levels=None)
    mesh = HealpixReader.load_mesh(mesh_cfg, mode_cfg=_hp_mode())
    decl = mesh.metadata["uv_data_last"]
    assert decl.dtype == np.int32 and decl.shape == (mesh.n_nodes,)
    own = _store_cell_of_nodes(mesh, nside, order)
    np.testing.assert_array_equal(decl, n_fin[own] - 1)
    assert len(set(decl.tolist())) == 3          # the test can tell orders


def test_healpix_later_record_nan_option_on_and_off(tmp_path):
    """A NaN inside a column's prefix in record 2, read in a later prefetch
    window: with the option the reader refuses it before zeroing (node and
    point named); without it the run is the run of a store with 0 there
    (and differs from the run on the store without the NaN, so the record
    and the node are used)."""
    from kelp.readers.healpix import HealpixReader
    from tests.test_healpix_reader import _pix_lonlat, _wet
    lon, lat = _pix_lonlat(8, "nest")
    cell = int(np.flatnonzero(_wet(lat) & (lat < -10.0) & (lon > 10.0)
                              & (lon < 170.0))[0])
    stores = {"nan": _hp_store(tmp_path / "3D_nan_ocean.nc",
                               nan_at=(2, 2, cell, np.nan)),
              "zero": _hp_store(tmp_path / "3D_zero_ocean.nc",
                                nan_at=(2, 2, cell, 0.0)),
              "orig": _hp_store(tmp_path / "3D_orig_ocean.nc")}
    mesh = HealpixReader.load_mesh(_hp_mesh_cfg(stores["nan"]),
                                   mode_cfg=_hp_mode())
    node = int(np.flatnonzero(_store_cell_of_nodes(mesh, 8, "nest")
                              == cell)[0])
    seeds = [(float(lon[cell]) + 0.3, float(lat[cell]) + 0.2, 45.0),
             (10.0, 5.0, 30.0)]
    kw = dict(prefetch=2, n_steps=2, integrator="rk2")
    cfg = _hp_cfg(tmp_path, stores["nan"], seeds, **kw)
    with pytest.raises(ValueError,
                       match=rf"u = nan is non-finite at node {node}, "
                             r"point 2"):
        _run(cfg)
    outs = {}
    for name, store in stores.items():
        c = _hp_cfg(tmp_path, store, seeds, uv_vertical="layer",
                    out=f"{name}.nc", **kw)
        _run(c)
        outs[name] = load_trajectories(c.data.output_path)
    for k in ("lon", "lat", "z", "vel_u", "vel_v", "vel_w"):
        assert_bitwise(outs["nan"][k], outs["zero"][k], k)
    assert not np.array_equal(outs["orig"]["lon"][0], outs["zero"]["lon"][0])


def test_healpix_prefix_read_only_for_3d_loads(tmp_path, monkeypatch):
    """The u prefix (one extra pass over the first record) is read and
    declared only for a 3-D load (a mode_cfg with Fdrift3D); a 2-D load
    and a load without mode_cfg neither read it nor declare it."""
    from kelp.readers import healpix as hp_mod
    from kelp.readers.healpix import HealpixReader
    store = _hp_store(tmp_path / "3D_load_ocean.nc")
    calls = []
    real = hp_mod._finite_prefix_per_cell

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)
    monkeypatch.setattr(hp_mod, "_finite_prefix_per_cell", spy)
    two_d = ModeConfig(Fdrift2D=True, Fdrift3D=False)
    for mode in (two_d, None):
        mesh = HealpixReader.load_mesh(_hp_mesh_cfg(store), mode_cfg=mode)
        assert "uv_data_last" not in mesh.metadata
    assert calls == []
    mesh = HealpixReader.load_mesh(_hp_mesh_cfg(store), mode_cfg=_hp_mode())
    assert calls == [1] and "uv_data_last" in mesh.metadata


def test_healpix_rotated_nan_in_v_alone_named_v(tmp_path):
    """Under ``mesh.rotation`` the raw record is checked BEFORE the u/v
    rotation mix: a NaN in v alone, inside a column's prefix, is refused
    as ``v`` (after the mix both components would be NaN and the refusal
    would name u)."""
    from kelp.core.rotation import FESOM2_ANGLES, euler_matrix, rotate_lonlat
    from kelp.readers.healpix import HealpixReader
    from tests.test_healpix_reader import _pix_lonlat
    nside = 8
    raw = _hp_store(tmp_path / "3D_rot_raw.nc", nside=nside)
    lon, lat = _pix_lonlat(nside, "nest")
    _, rlat = rotate_lonlat(lon, lat, euler_matrix(*FESOM2_ANGLES))
    ds = xr.open_dataset(raw).load()
    ds.close()
    # both rotated caps dry (as a real ocean mask keeps them)
    dry = np.abs(rlat) >= 75.0
    for name in ("avg_uoe", "avg_von", "avg_wo"):
        a = ds[name].values.copy()
        a[..., dry] = np.nan
        ds[name] = (ds[name].dims, a)
    a = ds["avg_ssh"].values.copy() if "avg_ssh" in ds else None
    if a is not None:
        a[..., dry] = np.nan
        ds["avg_ssh"] = (ds["avg_ssh"].dims, a)
    cell = int(np.flatnonzero(np.isfinite(ds["avg_von"].values[0, 2])
                              & (lat < -10.0) & (lat > -40.0))[0])
    v = ds["avg_von"].values.copy()
    # v alone, its deepest wet layer (a hole above finite data would be
    # refused earlier as a corrupt column); u keeps the layer
    v[0, 2, cell] = np.nan
    ds["avg_von"] = (ds["avg_von"].dims, v)
    store = tmp_path / "3D_rot_ocean.nc"
    ds.to_netcdf(store)
    mesh_cfg = dataclasses.replace(_hp_mesh_cfg(store), rotation="fesom2")
    mode = _hp_mode()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")        # near-pole notes, if any
        mesh = HealpixReader.load_mesh(mesh_cfg, mode_cfg=mode)
    assert mesh.rotation_angles is not None
    node = int(np.flatnonzero(_store_cell_of_nodes_geo(mesh, nside)
                              == cell)[0])
    ds2 = HealpixReader.open_dataset(str(store), mode_cfg=mode)
    with pytest.raises(ValueError,
                       match=rf"v = nan is non-finite at node {node}, "
                             r"point 2"):
        HealpixReader.read_velocities(ds2, 0, mesh, mode_cfg=mode)
    # the same record without the planted NaN reads (the mask is fine)
    HealpixReader.read_velocities(ds2, 1, mesh, mode_cfg=mode)
    ds2.close()


def _store_cell_of_nodes_geo(mesh, nside):
    """The NEST store cell of every node of a ROTATED mesh (its stored
    coordinates are rotated: map them back to geographic first)."""
    import healpy
    from kelp.core.rotation import euler_matrix, unrotate_lonlat
    glon, glat = unrotate_lonlat(mesh.nodes_lon, mesh.nodes_lat,
                                 euler_matrix(*mesh.rotation_angles))
    return healpy.ang2pix(nside, glon, glat, nest=True, lonlat=True)


def test_healpix_reused_mesh_off_on_off(tmp_path):
    """One mesh loaded with the option off carries the declaration; a
    depth-linear run on it uses it (layer 1's u at 40 m, no zero-filled
    slot); option-off runs before and after are bit-identical to each other
    and to a run on a freshly loaded mesh."""
    from kelp.readers.healpix import HealpixReader
    from tests.test_healpix_reader import _pix_lonlat, _wet
    lon, lat = _pix_lonlat(8, "nest")
    store = _hp_store(tmp_path / "3D_reuse_ocean.nc",
                      n_fin=np.where(_wet(lat), 2, 0), deptho=True)
    off, on = _hp_mode(uv_vertical="layer"), _hp_mode()
    mesh = HealpixReader.load_mesh(_hp_mesh_cfg(store), mode_cfg=off)
    assert "uv_data_last" in mesh.metadata
    ds = HealpixReader.open_dataset(str(store), mode_cfg=off)
    seeds = [(10.0, 5.0, 40.0), (-40.0, -20.0, 12.5)]
    kw = dict(n_steps=2, n_time_total=3)
    r1 = _rsr(mesh, ds, HealpixReader, off, seeds, **kw)
    r2 = _rsr(mesh, ds, HealpixReader, on, seeds, **kw)
    r3 = _rsr(mesh, ds, HealpixReader, off, seeds, **kw)
    fresh = HealpixReader.load_mesh(_hp_mesh_cfg(store), mode_cfg=off)
    r4 = _rsr(fresh, ds, HealpixReader, off, seeds, **kw)
    for k in ("lon", "lat", "z", "vel_u"):
        assert_bitwise(r3[k], r1[k], k)
        assert_bitwise(r4[k], r1[k], k)
    np.testing.assert_allclose(
        r2["lon"].T, _hp_hand(seeds, 2, lambda z: _hp_u(z, (5.0, 20.0))),
        rtol=0, atol=1e-9)
    assert not np.allclose(r2["lon"], r1["lon"], atol=1e-6)
    ds.close()


# ---------------------------------------------------------------------------
# max_levels
# ---------------------------------------------------------------------------

def test_max_levels_cut_rebuilds_table(tmp_path):
    """max_levels 3 keeps levels 0, 10, 30 m (layer centres 5, 20 m): a
    particle seeded at 45 m, below the last kept level, moves with the
    constant extension u(20 m); the table built before the cut is replaced
    by one of the cut levels, in run_tracking and on a direct entry."""
    from kelp.readers.healpix import HealpixReader
    store = _hp_store(tmp_path / "3D_cut_ocean.nc")
    seeds = [(10.0, 5.0, 45.0), (-40.0, -20.0, 12.5)]
    u_cut = lambda z: _hp_u(z, (5.0, 20.0))            # noqa: E731
    cfg = _hp_cfg(tmp_path, store, seeds, max_levels=3)
    _run(cfg)
    traj = load_trajectories(cfg.data.output_path)
    np.testing.assert_allclose(traj["lon"], _hp_hand(seeds, 2, u_cut),
                               rtol=0, atol=1e-9)
    # direct entry on an uncut mesh whose full table was built first
    mesh = HealpixReader.load_mesh(_hp_mesh_cfg(store), mode_cfg=_hp_mode())
    before = depth_points(mesh, "layer_linear")
    np.testing.assert_array_equal(before.p, HP_CENTRES)
    ds = HealpixReader.open_dataset(str(store), mode_cfg=_hp_mode())
    r = _rsr(mesh, ds, HealpixReader, _hp_mode(max_levels=3), seeds,
             n_steps=2, n_time_total=3)
    after = depth_points(mesh, "layer_linear")
    assert after is not before
    np.testing.assert_array_equal(after.p, [5.0, 20.0])
    np.testing.assert_array_equal(np.abs(mesh.sigma), [0.0, 10.0, 30.0])
    assert_bitwise(r["lon"].T, traj["lon"], "lon")
    ds.close()


def _write_deep_fesom2(work):
    """A one-file FESOM2 dataset on levels 0, 10, 30, 70, 150 m (bed
    150 m): u = A + B·z at the layer centres 5, 20, 50, 110 m, v = w = 0;
    the mesh diag with the same levels."""
    import tests._fesom2_synth as s
    zbar = np.array([0.0, -10.0, -30.0, -70.0, -150.0])
    centres = 0.5 * (np.abs(zbar[:-1]) + np.abs(zbar[1:]))
    old = (s.ZBAR_NEG, s.ZBAR_N_BOTTOM_NEG)
    s.ZBAR_NEG, s.ZBAR_N_BOTTOM_NEG = zbar, np.full(4, -150.0)
    try:
        mesh_path = s.write_meshdiag(work / "deep.mesh.diag.nc", "ugrid")
    finally:
        s.ZBAR_NEG, s.ZBAR_N_BOTTOM_NEG = old
    nt = 3
    u = np.broadcast_to((A + B * centres)[None, :, None], (nt, 4, 4))
    ds = xr.Dataset(
        {"unod": (("time", "nz1", "nod2"), u.astype(np.float32)),
         "vnod": (("time", "nz1", "nod2"), np.zeros((nt, 4, 4), np.float32)),
         "w": (("time", "nz", "nod2"), np.zeros((nt, 5, 4), np.float32)),
         "ssh": (("time", "nod2"), np.zeros((nt, 4), np.float32))},
        coords={"time": ("time", np.arange(nt) * DT,
                         {"units": "seconds since 2000-01-01",
                          "calendar": "standard"})})
    data = work / "deep.nc"
    ds.to_netcdf(data)
    return mesh_path, data, centres


def test_direct_entry_max_levels(tmp_path):
    """Independently loaded FESOM2 and HEALPix meshes, option on, direct
    run_single_realization calls with max_levels 3, then 4, then None on
    the same mesh: the levels, the depth-point table and the arrays the
    reader returns agree each time, and a particle below the third level
    moves with the u of each call's rule."""
    from kelp.readers.healpix import HealpixReader
    # FESOM2, levels 0..150 m: a particle at 80 m
    mesh_path, data, centres = _write_deep_fesom2(tmp_path)
    reader = get_reader("fesom2")
    mesh = reader.load_mesh(MeshConfig(type="fesom2", meshpath=str(mesh_path)))
    seeds = [(0.2, 0.5, 80.0)]
    expect = {3: A + B * 20.0, 4: A + B * 50.0, None: A + B * 80.0}
    for ml, n_lev in ((3, 3), (4, 4), (None, 5)):
        mode = _fesom2_mode(max_levels=ml)
        ds = reader.open_dataset(str(data), mode_cfg=mode)
        r = _rsr(mesh, ds, reader, mode, seeds, n_steps=2)
        assert mesh.sigma.size == n_lev
        dp = depth_points(mesh, "layer_linear")
        np.testing.assert_array_equal(dp.p, centres[:n_lev - 1])
        vel = reader.read_velocities(ds, 0, mesh, mode)
        assert vel.u.shape[0] == n_lev - 1 and vel.w.shape[0] == n_lev
        x = 0.2 + np.arange(1, 3) * _dlon(expect[ml], 0.5)
        np.testing.assert_allclose(r["lon"][:, 0], x, rtol=0, atol=1e-11)
        ds.close()
    # HEALPix, levels 0..70 m: a particle at 40 m
    store = _hp_store(tmp_path / "3D_direct_ocean.nc")
    hmesh = HealpixReader.load_mesh(_hp_mesh_cfg(store), mode_cfg=_hp_mode())
    hp_seeds = [(10.0, 5.0, 40.0)]
    for ml, n_lev in ((3, 3), (4, 4), (None, 4)):
        mode = _hp_mode(max_levels=ml)
        ds = HealpixReader.open_dataset(str(store), mode_cfg=mode)
        r = _rsr(hmesh, ds, HealpixReader, mode, hp_seeds, n_steps=2,
                 n_time_total=3)
        assert hmesh.sigma.size == n_lev
        np.testing.assert_array_equal(
            depth_points(hmesh, "layer_linear").p, HP_CENTRES[:n_lev - 1])
        vel = HealpixReader.read_velocities(ds, 0, hmesh, mode_cfg=mode)
        assert vel.u.shape[0] == n_lev - 1 and vel.w.shape[0] == n_lev
        u = _hp_u(40.0, HP_CENTRES[:n_lev - 1])
        np.testing.assert_allclose(
            r["lon"][:, 0], 10.0 + np.arange(1, 3) * _dlon(u, 5.0),
            rtol=0, atol=1e-9)
        ds.close()


def test_apply_max_levels_helper():
    """The cut keeps the uncut table once and cuts from it; the same limit
    again changes nothing (same object); None restores the full table; a
    2-level limit in 3-D is refused as before."""
    mesh = _zmesh()
    full = mesh.sigma
    tracking._apply_max_levels(mesh, ModeConfig(Fdrift3D=True,
                                                Fdrift2D=False,
                                                max_levels=3))
    cut = mesh.sigma
    np.testing.assert_array_equal(cut, full[:3])
    assert mesh.metadata["sigma_full"] is full
    tracking._apply_max_levels(mesh, ModeConfig(Fdrift3D=True,
                                                Fdrift2D=False,
                                                max_levels=3))
    assert mesh.sigma is cut
    tracking._apply_max_levels(mesh, ModeConfig(Fdrift3D=True,
                                                Fdrift2D=False,
                                                max_levels=99))
    np.testing.assert_array_equal(mesh.sigma, full)
    tracking._apply_max_levels(mesh, ModeConfig(Fdrift3D=True,
                                                Fdrift2D=False,
                                                max_levels=2))
    tracking._apply_max_levels(mesh, ModeConfig(Fdrift3D=True,
                                                Fdrift2D=False))
    np.testing.assert_array_equal(mesh.sigma, full)
    with pytest.raises(ValueError, match="max_levels must be >= 2"):
        tracking._apply_max_levels(mesh, ModeConfig(Fdrift3D=True,
                                                    Fdrift2D=False,
                                                    max_levels=1))
    fresh = _zmesh()
    tracking._apply_max_levels(fresh, ModeConfig(Fdrift3D=True,
                                                 Fdrift2D=False))
    assert "sigma_full" not in fresh.metadata


# ---------------------------------------------------------------------------
# the reader hook prepare(mesh, ds)
# ---------------------------------------------------------------------------

def test_prepare_called_per_dataset(tmp_path, monkeypatch):
    """``prepare`` runs once per run_tracking (two realizations), right
    after open_dataset; on direct entries again for a second dataset on the
    same mesh, never twice in a row for the same dataset; the dataset is
    named by a uuid token, never an object id, and the mesh remembers the
    dataset and the level table it is prepared for."""
    base = get_reader("fesom2")
    calls, events = [], []
    out_files = [tmp_path / "traj_r0.nc", tmp_path / "traj_r1.nc"]

    class Preparing(base):
        @staticmethod
        def prepare(mesh, ds):
            calls.append((mesh, ds))
            events.append(("prepare", any(f.exists() for f in out_files)))

        @staticmethod
        def probe_uv_layout(ds, mode_cfg, mesh):
            events.append(("probe", None))
            return base.probe_uv_layout(ds, mode_cfg, mesh)
    monkeypatch.setattr(tracking, "get_reader", lambda name: Preparing)
    cfg = _fesom2_cfg(tmp_path, uv_vertical="layer", num_runs=2)
    _run(cfg)
    assert len(calls) == 1
    # right after open_dataset: before the pre-writer probe, no output yet
    assert events[:2] == [("prepare", False), ("probe", None)]
    m0, d0 = calls[0]
    tok = d0.attrs["kelp_dataset_token"]
    assert re.fullmatch(r"[0-9a-f]{32}", tok)
    assert m0.metadata["kelp_prepared_dataset"] == tok
    assert m0.metadata["kelp_prepared_sigma"] is m0.sigma
    # direct entries: a second dataset on the same mesh is prepared too
    calls.clear()
    mesh = Preparing.load_mesh(cfg.mesh)
    ds1 = Preparing.open_dataset(cfg.data.ncinput, mode_cfg=cfg.mode)
    ds2 = Preparing.open_dataset(cfg.data.ncinput, mode_cfg=cfg.mode)
    _rsr(mesh, ds1, Preparing, cfg.mode, SEEDS, n_steps=1)
    _rsr(mesh, ds1, Preparing, cfg.mode, SEEDS, n_steps=1)
    assert [c[1] is ds1 for c in calls] == [True]
    _rsr(mesh, ds2, Preparing, cfg.mode, SEEDS, n_steps=1)
    assert len(calls) == 2 and calls[1][1] is ds2
    assert ds1.attrs["kelp_dataset_token"] != ds2.attrs["kelp_dataset_token"]
    assert (mesh.metadata["kelp_prepared_dataset"]
            == ds2.attrs["kelp_dataset_token"])
    # another mesh, the same dataset: prepared for that mesh
    other = Preparing.load_mesh(cfg.mesh)
    _rsr(other, ds1, Preparing, cfg.mode, SEEDS, n_steps=1)
    assert len(calls) == 3 and calls[2][0] is other
    # a time window of ds1 is the same dataset (it inherits the token)
    _rsr(other, ds1.isel(time=slice(0, 4)), Preparing, cfg.mode, SEEDS,
         n_steps=1)
    assert len(calls) == 3
    ds1.close()
    ds2.close()


def _declaring_reader(placement):
    """A reader whose ``prepare`` declares the dataset's own u/v data prefix
    (``ds.attrs["decl"]``, a new array each call) and records its calls
    (dataset name, kept levels).  ``"layer_linear"``: the FESOM2 reader (the
    option marks its fields); ``"level"``: u = A + B·z on the levels of
    ``mesh.sigma`` (``native_uv_vertical = "level"``), w = v = 0."""
    base = get_reader("fesom2")
    calls = []

    class Declaring(base):
        @staticmethod
        def prepare(mesh, ds):
            calls.append((ds.attrs["name"], int(mesh.sigma.size)))
            mesh.metadata["uv_data_last"] = np.array(ds.attrs["decl"],
                                                     dtype=np.int32)
    if placement == "level":
        def read_velocities(ds, timestep, mesh, mode_cfg=None, **kw):
            z = np.abs(np.asarray(mesh.sigma, dtype=np.float64))
            n = mesh.n_nodes
            u = np.ascontiguousarray(
                np.repeat((A + B * z)[:, None], n, axis=1))
            return VelocityField(u=u, v=np.zeros_like(u), eta=np.zeros(n),
                                 w=np.zeros((z.size, n)), uv_layout="node",
                                 uv_vertical="level")
        Declaring.native_uv_vertical = "level"
        Declaring.read_velocities = staticmethod(read_velocities)
    return Declaring, calls


#: per placement: option, particle depth, the full (A) and the shallow (B)
#: declaration, and the u each gives at that depth
_SWITCH = {
    "layer_linear": ("linear", 12.5, [1, 1, 1, 1], [0, 0, 0, 0],
                     A + B * 12.5, A + B * 5.0),
    "level": ("layer", 25.0, [2, 2, 2, 2], [1, 1, 1, 1],
              A + B * 25.0, A + B * 10.0),
}


def _switch_env(tmp_path, placement):
    option, z, decl_a, decl_b, u_a, u_b = _SWITCH[placement]
    cfg = _fesom2_cfg(tmp_path)
    reader, calls = _declaring_reader(placement)
    mesh = reader.load_mesh(cfg.mesh)
    dss = {}
    for name, decl in (("A", decl_a), ("B", decl_b)):
        dss[name] = reader.open_dataset(cfg.data.ncinput, mode_cfg=cfg.mode)
        dss[name].attrs.update(name=name, decl=decl)
    seeds = [(0.15, 0.4, z)]

    def run(name, **mode_kw):
        mode = _fesom2_mode(uv_vertical=option, **mode_kw)
        return _rsr(mesh, dss[name], reader, mode, seeds, n_steps=1)
    return mesh, dss, run, calls, {"A": u_a, "B": u_b}


@pytest.mark.parametrize("placement", sorted(_SWITCH))
@pytest.mark.parametrize("order", ["ABA", "BAB"])
def test_switching_datasets_prepares_again(tmp_path, placement, order):
    """Two datasets with different data prefixes on ONE mesh, used A → B → A
    (and B → A → B): every switch runs ``prepare`` again, so each run reads
    with its own dataset's declaration and the third run equals the first
    bit for bit."""
    mesh, dss, run, calls, u = _switch_env(tmp_path, placement)
    out = [run(name) for name in order]
    for name, r in zip(order, out):        # record 0: the 1958 file
        assert r["vel_u"][0, 0] == pytest.approx(u[name], rel=0, abs=1e-15)
    for k in ("lon", "lat", "z", "vel_u"):
        assert_bitwise(out[2][k], out[0][k], k)
    assert not np.array_equal(out[1]["vel_u"], out[0]["vel_u"])
    assert [c[0] for c in calls] == list(order)
    for ds in dss.values():
        ds.close()


@pytest.mark.parametrize("placement", sorted(_SWITCH))
def test_prepare_follows_the_kept_levels(tmp_path, placement):
    """The same dataset twice: ``prepare`` once.  A direct call with another
    ``max_levels`` re-cuts ``mesh.sigma`` and runs ``prepare`` again (it
    sees the cut table); the same limit again does not; None restores the
    table and prepares again.  The depth-point table follows the levels."""
    mesh, dss, run, calls, u = _switch_env(tmp_path, placement)
    run("A")
    run("A")
    assert calls == [("A", 3)]
    run("A", max_levels=2)
    assert calls[-1] == ("A", 2) and len(calls) == 2
    run("A", max_levels=2)
    assert len(calls) == 2
    run("A")
    assert calls[-1] == ("A", 3) and len(calls) == 3
    dp = depth_points(mesh, placement)
    full = np.abs(mesh.sigma)
    np.testing.assert_array_equal(
        dp.p, full if placement == "level" else 0.5 * (full[:-1] + full[1:]))
    for ds in dss.values():
        ds.close()


def test_concat_refuses_shards_with_different_option(tmp_path):
    """The output attribute takes part in the sharding concat's equality
    check: shards written with and without the option do not merge."""
    pytest.importorskip("zarr")
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
    from concat_trajectories import concat_trajectories
    shards = []
    for k, meta in enumerate(({"uv_vertical": "linear"}, {})):
        p = str(tmp_path / f"shard{k}.zarr")
        st = make_state(np.zeros(2), np.zeros(2), np.zeros(2))
        with TrajectoryWriter(p, 2, metadata={"particle_offset": 2 * k,
                                              "random_draws": 0,
                                              **meta}) as w:
            w.set_time_units("seconds since 2001-01-01 00:00:00",
                             "standard")
            w.write_release(np.zeros(2), np.zeros(2), np.zeros(2),
                            np.zeros(2))
            w.append(st, 3600.0)
        shards.append(p)
    with pytest.raises(ValueError, match="uv_vertical"):
        concat_trajectories(shards, str(tmp_path / "m.zarr"))
