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
import warnings

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")
xr = pytest.importorskip("xarray")

from kelp.core.config import (AppConfig, DataConfig, MeshConfig,  # noqa: E402
                              ModeConfig, NumericsConfig, TimeConfig)
import kelp.readers.healpix as hx  # noqa: E402
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


# ---------------------------------------------------------------------------
# the rule at every 3-D read (spec tests 2-6)
# ---------------------------------------------------------------------------
#
# P is the padded layout: the layer variables on K entries whose K-th plane is
# empty, all on one ``level`` axis with w.  B is the twin layout (layers on
# ``lev`` with K-1 entries, w on ``level``).  A defect variable is put on its
# OWN dim ``level_x``, so that one defect does not cause a second one.

#: three sea columns the dual mesh tracks (the seed columns of 5, 2 and 3
#: layers near (0, 0))
TRACKED3 = (CELL_DEEP[0], CELL_SHALLOW[0], _seed_cell(3)[0])
OLD_TOOL = "extract_healpix_region v1.0"
NEW_TOOL = "extract_healpix_region v1.1"
FRAG_W = "must sit on exactly the K interfaces"
FRAG_L = "a layer variable needs K-1 ="
FRAG_P = "last level carries data at"
FRAG_OLD = "was written by extract_healpix_region v1.0"


def _p_store(path, **kw):
    return _vstore(path, layer_dim="level", w_dim="level", pad=True, **kw)


def _b_store(path, **kw):
    return _vstore(path, layer_dim="lev", w_dim="level", **kw)


def _put(name, n, *, value=None, records=slice(None), cells=TRACKED3,
         dim="level_x"):
    """An ``edit``: ``name`` on ``dim`` with ``n`` entries — its first
    entries, NaN planes appended — and, with ``value``, that value at
    ``cells`` of the last entry (index n - 1) at ``records``."""
    def edit(ds):
        a = ds[name].values
        nt, n0, npix = a.shape
        b = np.full((nt, n, npix), np.nan)
        b[:, :min(n, n0)] = a[:, :min(n, n0)]
        if value is not None:
            b[records, n - 1, list(cells)] = value
        ds[name] = (("time", dim, "value"), b)
    return edit


def _with_dry3d(n, value=None):
    """An ``edit``: a 3-D dry mask ``dry3d`` (0.0 where T is defined) on
    its own dim with ``n`` entries, ``value`` at TRACKED3 of the last."""
    def edit(ds):
        ds["dry3d"] = xr.zeros_like(ds["avg_thetao"]).where(
            np.isfinite(ds["avg_thetao"]))
        _put("dry3d", n, value=value)(ds)
    return edit


def _as_extract(src_path, out_path, *, tool, cut=None, last_nan=False,
                deptho=False):
    """A regional extract of a ``_vstore`` file, written as Zarr: the cells
    with |lat| < 45 (sorted NEST ids, ``healpix_index`` on the cell dim),
    the interface table embedded, ``attrs["tool"] = tool`` (None: no
    attribute).  ``cut=M`` slices EVERY vertical axis and the table to M
    entries (the cut of the old extractor); ``last_nan`` sets the last
    level of the four layer variables to NaN; ``deptho`` adds a per-cell
    ``deptho`` of 300.0 at sea."""
    with xr.open_dataset(src_path) as src:
        ds = src.load()
    _, lat, _ = _cells()
    keep = np.flatnonzero(np.abs(lat) < 45.0)
    ds = ds.isel(value=keep)
    ds["healpix_index"] = (("value",), keep.astype(np.int64))
    ds = ds.drop_vars("level_interfaces_m", errors="ignore")
    table = np.asarray(LEV, dtype=np.float64)
    if cut is not None:
        vdims = [d for d in ds.dims
                 if str(d) == "lev" or str(d).startswith("level")]
        ds = ds.isel({d: slice(0, cut) for d in vdims})
        table = table[:cut]
    if last_nan:
        for nm in ("avg_uoe", "avg_von", "avg_thetao", "avg_so"):
            a = ds[nm].values.copy()
            a[:, -1, :] = np.nan
            ds[nm] = (ds[nm].dims, a)
    if deptho:
        ds["deptho"] = (("value",), np.where(lat[keep] < 60.0, 300.0, np.nan))
    ds["level_interfaces_m"] = (("level_interface",), table)
    if tool is None:
        ds.attrs.pop("tool", None)
    else:
        ds.attrs["tool"] = tool
    ds.to_zarr(out_path, mode="w")
    return out_path


def _open(store):
    return HealpixReader.open_dataset(str(store), mode_cfg=MODE_3D)


def _cut(mesh, max_levels):
    from kelp.readers import _apply_max_levels
    _apply_max_levels(mesh, ModeConfig(Fdrift2D=False, Fdrift3D=True,
                                       max_levels=max_levels))
    return mesh


def _old_extract_warnings(rec):
    return [w for w in rec if FRAG_OLD in str(w.message)]


def test_padded_equals_layer_mids(tmp_path):
    p = _p_store(tmp_path / "P.nc")
    b = _b_store(tmp_path / "B.nc")
    _same(_run(tmp_path, p, "P"), _run(tmp_path, b, "B"))


_DIRECT_LEGS = {
    # leg: (edit, what is read, fragments of the refusal; None = valid)
    "w_k_plus_1": (_put("avg_wo", K + 1), "vel",
                   (FRAG_W, "'avg_wo' has 7 entries", "'level_x'",
                    "K = 6 interfaces")),
    "w_k_minus_1": (_put("avg_wo", K - 1), "vel",
                    (FRAG_W, "'avg_wo' has 5 entries", "K = 6 interfaces")),
    "u_k_plus_1": (_put("avg_uoe", K + 1), "vel",
                   (FRAG_L, "'avg_uoe' has 7 entries", "K-1 = 5 entries")),
    "u_k_minus_2": (_put("avg_uoe", K - 2), "vel",
                    (FRAG_L, "'avg_uoe' has 4 entries", "K-1 = 5 entries")),
    "v_k_plus_1": (_put("avg_von", K + 1), "vel",
                   (FRAG_L, "'avg_von' has 7 entries")),
    "u_last_one": (_put("avg_uoe", K, value=1.0), "vel",
                   (FRAG_P, "'avg_uoe' has K = 6 entries on 'level_x'",
                    "(time index 0 of the dataset read (time 2020-01-02")),
    "u_last_zero": (_put("avg_uoe", K, value=0.0), "vel",
                    (FRAG_P, "'avg_uoe' has K = 6 entries")),
    "thetao_last": (_put("avg_thetao", K, value=1.0), "scalar",
                    (FRAG_P, "'avg_thetao' has K = 6 entries")),
    "thetao_k_plus_1": (_put("avg_thetao", K + 1), "scalar",
                        (FRAG_L, "'avg_thetao' has 7 entries")),
    "valid_b_and_p": (None, None, None),
}


@pytest.mark.parametrize("leg", list(_DIRECT_LEGS))
@pytest.mark.parametrize("cut", [False, True], ids=["uncut", "cut"])
def test_direct_read_rules(tmp_path, cut, leg):
    """Direct read_velocities / read_scalar_fields: the size rule against
    the FULL table and the empty K-th plane, on a mesh uncut and cut with
    ``_apply_max_levels`` (spec test 4)."""
    b = _b_store(tmp_path / "B.nc")
    mesh = _mesh(b, MODE_3D)
    assert set(TRACKED3) <= set(mesh._healpix_data_idx.tolist())
    if cut:
        _cut(mesh, M)
    k = int(mesh.sigma.size)
    assert k == (M if cut else K)
    n = mesh.n_nodes
    edit, what, frags = _DIRECT_LEGS[leg]
    if edit is None:
        p = _p_store(tmp_path / "P.nc")
        got = {}
        for tag, store in (("B", b), ("P", p)):
            ds = _open(store)
            vf = HealpixReader.read_velocities(ds, 0, mesh, mode_cfg=MODE_3D)
            sc = HealpixReader.read_scalar_fields(
                ds, 0, mesh, ["avg_thetao", "avg_so"], MODE_3D)
            assert vf.u.shape == vf.v.shape == (k - 1, n)
            assert vf.w.shape == (k, n)
            assert sc["avg_thetao"].shape == sc["avg_so"].shape == (k - 1, n)
            got[tag] = (vf.u, vf.v, vf.w, sc["avg_thetao"], sc["avg_so"])
        for x, y in zip(got["B"], got["P"]):
            np.testing.assert_array_equal(x, y)
        return
    ds = _open(_b_store(tmp_path / f"{leg}.nc", edit=edit))
    with pytest.raises(ValueError, match=frags[0]) as ei:
        if what == "vel":
            HealpixReader.read_velocities(ds, 0, mesh, mode_cfg=MODE_3D)
        else:
            HealpixReader.read_scalar_fields(ds, 0, mesh, ["avg_thetao"],
                                             MODE_3D)
    msg = str(ei.value)
    for frag in frags[1:]:
        assert frag in msg, (frag, msg)
    if frags[0] == FRAG_P:
        assert f"at 3 of the {n} tracked node columns" in msg, msg
    assert "(mesh.levels)" in msg, msg
    cut_text = "(mode.max_levels keeps 4 of them; the check uses all 6)"
    assert (cut_text in msg) == cut, msg


def test_table_one_short_reports_w(tmp_path):
    """A table one interface short breaks w AND the layer variables; w is
    checked first, so the refusal names the table, not u."""
    p = _p_store(tmp_path / "P.nc")
    mesh = _mesh(p, MODE_3D, levels=LEV[:5])
    with pytest.raises(ValueError, match=FRAG_W) as ei:
        HealpixReader.read_velocities(_open(p), 0, mesh, mode_cfg=MODE_3D)
    assert "'avg_wo' has 6 entries" in str(ei.value)
    assert "K = 5 interfaces" in str(ei.value)


def test_hand_sliced_table_is_refused(tmp_path):
    """mesh.sigma sliced by hand presents the slice as the full table."""
    p = _p_store(tmp_path / "P.nc")
    mesh = _mesh(p, MODE_3D)
    mesh.sigma = mesh.sigma[:M]
    with pytest.raises(ValueError, match=FRAG_W) as ei:
        HealpixReader.read_velocities(_open(p), 0, mesh, mode_cfg=MODE_3D)
    assert "K = 4 interfaces" in str(ei.value)


def test_full_table_under_max_levels(tmp_path):
    """mode.max_levels below K on a valid store runs (spec test 5)."""
    p = _p_store(tmp_path / "P.nc")
    b = _b_store(tmp_path / "B.nc")
    a = _run(tmp_path, p, "P", max_levels=M)
    c = _run(tmp_path, b, "B", max_levels=M)
    _same(a, c)
    assert np.isfinite(a["lon"]).all()              # both ran to the end


def test_stray_value_on_untracked_cell(tmp_path):
    """A value in the K-th plane at a land cell the mesh does not track
    refuses nothing (spec test 6)."""
    _, lat, _ = _cells()
    land = int(np.flatnonzero(lat >= 60.0)[0])
    p = _p_store(tmp_path / "P.nc")
    s = _p_store(tmp_path / "S.nc",
                 edit=_put("avg_uoe", K, value=1.0, cells=(land,),
                           dim="level"))
    _same(_run(tmp_path, s, "S"), _run(tmp_path, p, "P"))


_PROBE_LEGS = {
    "thetao_last": (_put("avg_thetao", K, value=1.0), ("avg_thetao",),
                    None, FRAG_P),
    "thetao_k_plus_1": (_put("avg_thetao", K + 1), ("avg_thetao",),
                        None, FRAG_L),
    "dry_last": (_with_dry3d(K, value=1.0), (), "dry3d", FRAG_P),
    "dry_k_plus_1": (_with_dry3d(K + 1), (), "dry3d", FRAG_L),
}


@pytest.mark.parametrize("leg", list(_PROBE_LEGS))
def test_scalar_refused_in_the_probe_before_output(tmp_path, leg):
    """A sampled scalar or the 3-D dry mask that breaks the rule is refused
    in the start probe, before the writer exists (spec test 3)."""
    edit, sample, dry, frag = _PROBE_LEGS[leg]
    store = _b_store(tmp_path / f"{leg}.nc", edit=edit)
    over = {}
    if dry:
        over["beaching"] = dict(enabled=True, criteria=["depth"],
                                depth_min=0.3, dry_mask_var=dry,
                                use_eta=False)
    with pytest.raises(ValueError, match=frag):
        _run(tmp_path, store, leg, sample=sample, **over)
    assert not list(tmp_path.glob(f"traj_{leg}*"))


def test_scalar_plane_filled_at_a_later_record(tmp_path):
    """The K-th plane of a sampled scalar empty at record 0 and filled at
    record 1: a run from record 1 is refused in its probe (spec test 4)."""
    store = _p_store(tmp_path / "L.nc",
                     edit=_put("avg_thetao", K, value=1.0, records=1,
                               dim="level"))
    with pytest.raises(ValueError, match=FRAG_P) as ei:
        _run(tmp_path, store, "L", sample=("avg_thetao",), start_time_idx=1,
             n_steps=1)
    assert "2020-01-03" in str(ei.value)
    assert not list(tmp_path.glob("traj_L*"))


@pytest.mark.parametrize("leg", ["a_one_step", "b_reaches_record_2"])
@pytest.mark.parametrize("var", ["avg_uoe", "avg_thetao"])
def test_later_record_with_prefetch(tmp_path, var, leg):
    """The K-th plane filled at record 2 only; prefetch windows of 3 from
    record 0, so the start probe (record 0) is clean.  (a) One step never
    uses record 2: the failing look-ahead read is neither cached nor
    raised.  (b) Two steps reach record 2: refused there, naming its time
    stamp, after the output file exists (spec test 4)."""
    store = _p_store(tmp_path / "L.nc",
                     edit=_put(var, K, value=1.0, records=2, dim="level"))
    common = dict(sample=("avg_thetao",), prefetch_size=3)
    if leg == "a_one_step":
        out = _run(tmp_path, store, "L", n_steps=1, **common)
        assert np.isfinite(out["lon"]).all()
        return
    with pytest.raises(ValueError, match=FRAG_P) as ei:
        _run(tmp_path, store, "L", n_steps=2, **common)
    assert f"{var!r} has K = 6 entries" in str(ei.value)
    assert "2020-01-04" in str(ei.value)
    assert (tmp_path / "traj_L.nc").exists()


@pytest.mark.parametrize("leg", ["v10_cut", "v11_cut", "no_tool_cut",
                                 "v10_uncut"])
def test_old_extract_reads(tmp_path, leg):
    """An extract of the old tool (attribute ``tool`` exactly
    "extract_healpix_region v1.0") cut with region.max_levels keeps one
    layer too many; it is read as the K-1 layers the table bounds, the
    scalar probe warns once (W-OLD).  The same extract with another tool
    attribute, or none, is refused (M-P).  An UNCUT old extract (layer
    variables on K-1 entries) reads without a warning."""
    hx._WARNED.clear()
    p = _p_store(tmp_path / "P.nc")
    names = ["avg_thetao", "avg_so"]
    if leg == "v10_uncut":
        b = _b_store(tmp_path / "B.nc")
        u_store = _as_extract(b, tmp_path / "U.zarr", tool=OLD_TOOL)
        mesh = _mesh(u_store, MODE_3D, levels=None)
        ds = _open(u_store)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            vf = HealpixReader.read_velocities(ds, 0, mesh, mode_cfg=MODE_3D)
            sc = HealpixReader.read_scalar_fields(ds, 0, mesh, names,
                                                  MODE_3D)
        n = mesh.n_nodes
        assert vf.u.shape == vf.v.shape == (K - 1, n)
        assert vf.w.shape == (K, n)
        assert sc["avg_thetao"].shape == sc["avg_so"].shape == (K - 1, n)
        assert not _old_extract_warnings(rec)
        return
    if leg in ("v11_cut", "no_tool_cut"):
        tool = NEW_TOOL if leg == "v11_cut" else None
        e = _as_extract(p, tmp_path / "E.zarr", tool=tool, cut=M)
        mesh = _mesh(e, MODE_3D, levels=None)
        with pytest.raises(ValueError, match=FRAG_P) as ei:
            HealpixReader.read_velocities(_open(e), 0, mesh, mode_cfg=MODE_3D)
        assert "'avg_uoe' has K = 4 entries" in str(ei.value)
        assert "(the store's level_interfaces_m)" in str(ei.value)
        return
    e = _as_extract(p, tmp_path / "E.zarr", tool=OLD_TOOL, cut=M)
    ep = _as_extract(p, tmp_path / "EP.zarr", tool=NEW_TOOL, cut=M,
                     last_nan=True)
    mesh_e = _mesh(e, MODE_3D, levels=None)
    mesh_p = _mesh(ep, MODE_3D, levels=None)
    assert mesh_e.sigma.size == M
    n = mesh_e.n_nodes
    ds_e, ds_p = _open(e), _open(ep)
    vf_e = HealpixReader.read_velocities(ds_e, 0, mesh_e, mode_cfg=MODE_3D)
    vf_p = HealpixReader.read_velocities(ds_p, 0, mesh_p, mode_cfg=MODE_3D)
    assert vf_e.u.shape == vf_e.v.shape == (M - 1, n)
    np.testing.assert_array_equal(vf_e.u, vf_p.u)
    np.testing.assert_array_equal(vf_e.v, vf_p.v)
    ref = HealpixReader.read_scalar_fields(ds_p, 0, mesh_p, names, MODE_3D)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        got = HealpixReader.read_scalar_fields(ds_e, 0, mesh_e, names,
                                               MODE_3D)
    for nm in names:
        np.testing.assert_array_equal(got[nm], ref[nm])
    old = _old_extract_warnings(rec)
    assert len(old) == 1, [str(w.message) for w in rec]
    assert issubclass(old[0].category, RuntimeWarning)
    msg = str(old[0].message)
    for frag in ("'avg_thetao'", "'avg_so'", "K = 4 interfaces",
                 "the K-1 = 3 layers the table bounds",
                 "the layer below 70 m is not used",
                 ", and the bed is placed no deeper than 70 m",
                 "region.max_levels: 5 or more"):
        assert frag in msg, (frag, msg)
    assert "of which mode.max_levels keeps" not in msg
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        HealpixReader.read_scalar_fields(ds_e, 0, mesh_e, names, MODE_3D)
    assert not _old_extract_warnings(rec)
    # with a depth variable the bed is the store's: no bed clause
    ed = _as_extract(p, tmp_path / "ED.zarr", tool=OLD_TOOL, cut=M,
                     deptho=True)
    mesh_d = _mesh(ed, MODE_3D, levels=None)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        HealpixReader.read_scalar_fields(_open(ed), 0, mesh_d, names,
                                         MODE_3D)
    old = _old_extract_warnings(rec)
    assert len(old) == 1, [str(w.message) for w in rec]
    assert "the bed is placed" not in str(old[0].message)
