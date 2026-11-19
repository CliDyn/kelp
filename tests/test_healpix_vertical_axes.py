"""HEALPix reader: the vertical axis of each variable.

The reader finds the vertical dim of every variable by one helper
(``readers.healpix._vertical_dim``): a dim whose lower-cased name is one of
``_LEVEL_DIMS`` or starts with ``level``.  So a store with w on its own
``level_w`` axis (the layer variables on ``level`` with K-1 entries, w on
``level_w`` with K) reads as the twin layout (layers on ``lev``, w on
``level``), and a layer axis named ``level_<anything>`` is found at every
place that needs it.

The fixture (``_vstore``, ``_run``, ``_same``, ``K``, ``M``, ``LEV``,
``SEEDS``) is shared with the other vertical-axis tests: nside 8, K = 6
interfaces, three daily records, land north of 60 N, finite-prefix columns
of 2 ... K-1 layers whose values vary with depth.

New names of the reader are imported inside the tests, so this file can be
collected on a source without them.
"""
import numpy as np
import pytest

healpy = pytest.importorskip("healpy")
xr = pytest.importorskip("xarray")

from kelp.core.config import (AppConfig, DataConfig, MeshConfig,  # noqa: E402
                              ModeConfig, NumericsConfig, TimeConfig)
from kelp.readers.healpix import HealpixReader  # noqa: E402

K, M, NSIDE = 6, 4, 8
LEV = [0.0, 10.0, 30.0, 70.0, 150.0, 300.0]
NPIX = 12 * NSIDE * NSIDE
TIMES = np.array(["2020-01-02", "2020-01-03", "2020-01-04"],
                 dtype="datetime64[ns]")

MODE_3D = ModeConfig(Fdrift2D=False, Fdrift3D=True)
MODE_3D_LINEAR = ModeConfig(Fdrift2D=False, Fdrift3D=True,
                            uv_vertical="linear")


def _cells():
    """NEST lon (in [-180, 180)), lat and layer count of every pixel."""
    cell = np.arange(NPIX)
    lon, lat = healpy.pix2ang(NSIDE, cell, nest=True, lonlat=True)
    lon = (lon + 180.0) % 360.0 - 180.0
    nfin = 2 + (cell * 7) % (K - 2)          # 2 ... K-1 layers per sea column
    return lon, lat, nfin


def _seed_cell(n_layers):
    """The sea column with ``n_layers`` layers nearest to (0, 0), inside
    |lat| < 25 and |lon| <= 30."""
    lon, lat, nfin = _cells()
    ok = np.flatnonzero((nfin == n_layers) & (np.abs(lat) < 25.0)
                        & (np.abs(lon) <= 30.0))
    pick = ok[np.argmin(np.hypot(lon[ok], lat[ok]))]
    return int(pick), float(lon[pick]), float(lat[pick])


CELL_DEEP = _seed_cell(K - 1)                # nest 293: (-5.625, 0.0)
CELL_SHALLOW = _seed_cell(2)                 # nest 304: (0.0, 4.78...)

SEEDS = [
    (CELL_DEEP[1], CELL_DEEP[2], 5.0),
    (CELL_DEEP[1], CELL_DEEP[2], 100.0),
    (CELL_DEEP[1], CELL_DEEP[2], 200.0),
    (CELL_SHALLOW[1], CELL_SHALLOW[2], 12.0),
    (10.0, 5.0, 40.0),
    (-20.0, -10.0, 65.0),
]


def _vstore(path, *, layer_dim="level", w_dim="level_w", pad=False,
            embed_table=False, with_eta=True, edit=None):
    """Write the fixture.  The layer variables sit on (time, layer_dim,
    value) with K-1 entries, or — pad=True — K entries whose K-th plane is
    all-NaN (layer_dim may then equal w_dim); avg_wo on (time, w_dim, value)
    with K entries.  embed_table adds level_interfaces_m (level_interface,)
    = LEV.  with_eta=False leaves avg_zos out.  edit(ds) changes the
    in-memory dataset before the write (one defect per test).  A path ending
    in .zarr is written as Zarr, else as NetCDF.  Returns the path."""
    lon, lat, nfin = _cells()
    sea = lat < 60.0
    nfin = np.where(sea, nfin, 0)
    nt = TIMES.size
    n_lay = K if pad else K - 1
    k_lay = np.arange(n_lay)[:, None]
    alive = k_lay < nfin[None, :]            # (n_lay, NPIX); the pad plane: never

    def layer(values):
        plane = np.where(alive, values, np.nan)
        return (("time", layer_dim, "value"),
                np.repeat(plane[None], nt, axis=0))

    k_ifc = np.arange(K)[:, None]
    w = np.where((k_ifc <= nfin[None, :]) & sea[None, :],
                 -2e-5 * (1.0 + 0.1 * k_ifc), np.nan)
    ds = xr.Dataset(
        {
            "avg_uoe": layer(0.05 * (k_lay + 1) + 0.001 * lat[None, :]),
            "avg_von": layer(0.01 * (k_lay + 1) + 0.0 * lat[None, :]),
            "avg_thetao": layer(273.0 + k_lay + 0.01 * lat[None, :]),
            "avg_so": layer(35.0 - 0.1 * k_lay + 0.0 * lat[None, :]),
            "avg_wo": (("time", w_dim, "value"),
                       np.repeat(w[None], nt, axis=0)),
        },
        coords={"time": TIMES},
    )
    if with_eta:
        ds["avg_zos"] = (("time", "value"),
                         np.repeat(np.where(sea, 0.1, np.nan)[None], nt,
                                   axis=0))
    if embed_table:
        ds["level_interfaces_m"] = (("level_interface",),
                                    np.asarray(LEV, dtype=np.float64))
    ds["crs"] = ((), np.int32(0),
                 {"grid_mapping_name": "healpix", "healpix_nside": NSIDE,
                  "healpix_order": "nest"})
    if edit is not None:
        edited = edit(ds)
        if edited is not None:
            ds = edited
    if str(path).endswith(".zarr"):
        ds.to_zarr(path, mode="w")
    else:
        ds.to_netcdf(path)
    return path


_SECTIONS = (("mode", ModeConfig), ("numerics", NumericsConfig),
             ("time", TimeConfig), ("data", DataConfig))


def _run(tmp_path, store, tag, *, levels=LEV, sample=("avg_thetao", "avg_so"),
         seeds=None, **over):
    """run_tracking: 3-D, uv_interp barycentric, integrator rk2, two steps
    from record 0, time_averaging instant, output tmp_path/f"traj_{tag}.nc";
    `over` sets mode / numerics / time / data keys (uv_vertical,
    max_levels, Fdrift3D, sigma_layer, prefetch_size, beaching, n_steps,
    start_time_idx, num_runs, time_stamp, time_window, time_averaging).
    Returns load_trajectories of the file the run wrote: traj_{tag}.nc, or
    traj_{tag}_r0.nc when num_runs > 1 (run_tracking then writes _r0, _r1,
    ... and NO unsuffixed file).
    `levels` is mesh.levels.  Pass levels=None for EVERY store that embeds
    its table (every extract, E, EP, U, T): a mesh.levels that differs from
    an embedded table is refused at load (healpix.py:1173-1185)."""
    from kelp.tracking import run_tracking
    from kelp.writers import load_trajectories

    seeds_path = tmp_path / f"seeds_{tag}.txt"
    seeds_path.write_text("".join(
        " ".join(repr(float(c)) for c in s) + "\n"
        for s in (SEEDS if seeds is None else seeds)))
    out = tmp_path / f"traj_{tag}.nc"
    kw = {
        "mode": dict(Fdrift2D=False, Fdrift3D=True,
                     sample_vars=list(sample or ())),
        "numerics": dict(uv_interp="barycentric", integrator="rk2"),
        "time": dict(start_time_idx=0, n_steps=2),
        "data": dict(ncinput=str(store), inicloud=str(seeds_path),
                     output_path=str(out), time_averaging="instant"),
    }
    if (("time_stamp" in over or "time_window" in over)
            and "time_averaging" not in over):
        kw["data"]["time_averaging"] = None   # refused next to the stamp keys
    for key, value in over.items():
        owners = [sec for sec, cls in _SECTIONS
                  if key in cls.__dataclass_fields__]
        if len(owners) != 1:
            raise TypeError(f"_run: {key!r} is not a key of exactly one of "
                            f"the mode/numerics/time/data sections")
        kw[owners[0]][key] = value
    cfg = AppConfig(
        mode=ModeConfig(**kw["mode"]),
        numerics=NumericsConfig(**kw["numerics"]),
        mesh=MeshConfig(type="healpix", meshpath=str(store), levels=levels),
        data=DataConfig(**kw["data"]),
        time=TimeConfig(**kw["time"]),
    )
    run_tracking(cfg)
    if cfg.time.num_runs > 1:
        assert not out.exists()
        out = out.with_stem(f"{out.stem}_r0")
    return load_trajectories(str(out))


def _same(a, b):
    """First assert set(a) == set(b): the two outputs hold the same
    variables (vel_w only in 3-D, every sampled scalar).  Then
    np.array_equal(a[k], b[k], equal_nan=True) for EVERY key."""
    assert set(a) == set(b), (sorted(set(a) ^ set(b)))
    for k in a:
        assert np.array_equal(a[k], b[k], equal_nan=True), k


def _mesh(store, mode_cfg, levels=LEV):
    return HealpixReader.load_mesh(
        MeshConfig(type="healpix", meshpath=str(store), levels=levels),
        mode_cfg=mode_cfg)


# ---------------------------------------------------------------------------
# the helper
# ---------------------------------------------------------------------------

def test_vertical_dim_helper():
    from kelp.readers.healpix import _vertical_dim

    def da(*dims):
        return xr.DataArray(np.zeros((2,) * len(dims)), dims=dims, name="x")

    assert _vertical_dim(da("level", "value")) == "level"
    for d in ("lev", "depth", "level_w", "Level_Mid"):
        assert _vertical_dim(da(d, "cell")) == d
    assert _vertical_dim(da("value")) is None
    assert _vertical_dim(da("time", "cell")) is None
    with pytest.raises(ValueError, match="more than one vertical dim") as ei:
        _vertical_dim(da("level", "depth", "cell"))
    assert "'x'" in str(ei.value)            # da.name when no name is given
    with pytest.raises(ValueError, match="'avg_wo' has more than one"):
        _vertical_dim(da("level", "depth", "cell"), "avg_wo")


# ---------------------------------------------------------------------------
# configured runs (spec test 1)
# ---------------------------------------------------------------------------

def test_w_on_its_own_axis_equals_twin_layout(tmp_path):
    a_store = _vstore(tmp_path / "A.nc", layer_dim="level", w_dim="level_w")
    b_store = _vstore(tmp_path / "B.nc", layer_dim="lev", w_dim="level")
    a = _run(tmp_path, a_store, "A")
    b = _run(tmp_path, b_store, "B")
    _same(a, b)
    # not vacuous: every particle moved, w and the depth-varying T arrived
    assert "vel_w" in a and np.isfinite(a["vel_w"]).any()
    assert (a["lon"][:, -1] != a["lon"][:, 0]).all()
    assert np.isfinite(a["avg_thetao"]).all()
    assert np.ptp(a["avg_thetao"][:3, 0]) > 1.0   # 5 / 100 / 200 m differ


@pytest.mark.parametrize("leg", ["i_default", "ii_linear", "iii_2d",
                                 "iv_load_mesh"])
def test_level_named_layer_axis_at_every_call_site(tmp_path, leg):
    """The layer family on ``level_mid`` (not in ``_LEVEL_DIMS``), w on
    ``level_w``, no depth variable: every call site of the helper is on the
    path of one leg — the wet-mask level pick (every leg), the 3-D read
    (i, ii), the scalar read (i), the u data prefix (ii, iv), the
    bathymetry from the u column (iv), the 2-D level pick (iii)."""
    c_store = _vstore(tmp_path / "C.nc", layer_dim="level_mid",
                      w_dim="level_w")
    b_store = _vstore(tmp_path / "B.nc", layer_dim="lev", w_dim="level")
    if leg == "i_default":
        _same(_run(tmp_path, c_store, "C"), _run(tmp_path, b_store, "B"))
    elif leg == "ii_linear":
        _same(_run(tmp_path, c_store, "C", uv_vertical="linear"),
              _run(tmp_path, b_store, "B", uv_vertical="linear"))
    elif leg == "iii_2d":
        seeds = list(dict.fromkeys((lo, la) for lo, la, _ in SEEDS))
        two_d = dict(seeds=seeds, sample=("avg_thetao",), Fdrift2D=True,
                     Fdrift3D=False, sigma_layer=1)
        _same(_run(tmp_path, c_store, "C", **two_d),
              _run(tmp_path, b_store, "B", **two_d))
    else:
        mc = _mesh(c_store, MODE_3D_LINEAR)
        mb = _mesh(b_store, MODE_3D_LINEAR)
        assert (mb.depth > 0).any()
        np.testing.assert_array_equal(mc.depth, mb.depth)
        uc = mc.metadata.get("uv_data_last")
        ub = mb.metadata.get("uv_data_last")
        assert ub is not None
        assert uc is not None, "no uv_data_last on the level_mid store"
        np.testing.assert_array_equal(uc, ub)


# ---------------------------------------------------------------------------
# refusals at the read
# ---------------------------------------------------------------------------

def _read_w(store):
    mesh = _mesh(store, MODE_3D)
    ds = HealpixReader.open_dataset(str(store), mode_cfg=MODE_3D)
    return HealpixReader.read_velocities(ds, 0, mesh, mode_cfg=MODE_3D)


def test_two_vertical_dims_refused_at_read(tmp_path):
    def two_dims(ds):
        w = ds["avg_wo"].expand_dims(depth=2, axis=2)
        ds["avg_wo"] = w.transpose("time", "level_w", "depth", "value")

    store = _vstore(tmp_path / "W2.nc", edit=two_dims)
    with pytest.raises(ValueError, match="more than one vertical dim"):
        _read_w(store)


def test_missing_level_dim_names_the_rule(tmp_path):
    def no_dim(ds):
        ds["avg_wo"] = ds["avg_wo"].isel(level_w=0)

    store = _vstore(tmp_path / "W0.nc", edit=no_dim)
    with pytest.raises(ValueError, match="no level dim") as ei:
        _read_w(store)
    assert "starting with 'level'" in str(ei.value)
