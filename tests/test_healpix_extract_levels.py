"""HEALPix extractor: the vertical axes against the interface table.

``utils/extract_healpix_region.py`` checks the level axes of the source
against the FULL interface table (the store's ``level_interfaces_m``, else
``source.levels``) before it cuts anything; ``region.max_levels: M`` then
keeps M interfaces for w and M-1 entries for every layer variable, renames
w's vertical dim when it shared one with a layer variable, carries the
source's bed as ``deptho`` (computed from the uncut u and the full table),
and writes the (cut) table.  ``max_levels: 1`` stays the table-less surface
cut for 2-D runs.  New extracts carry ``tool = extract_healpix_region v1.1``.

Sources (the fixture of ``test_healpix_vertical_axes``, nside 8, K = 6,
M = 4): A = the layer variables on ``level`` (K-1), w on its own
``level_w`` axis (K); P = a padded store, everything on one ``level`` axis
(K entries, the K-th layer plane empty); T = layers on ``lev`` (K-1), w on
``level`` (K), the table embedded.  Every run on an extract passes
``levels=None`` (an extract embeds its table, or none).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

healpy = pytest.importorskip("healpy")
xr = pytest.importorskip("xarray")
pytest.importorskip("zarr")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_healpix_region import extract_region  # noqa: E402

from test_healpix_reader import _make_eta_store  # noqa: E402
from test_healpix_vertical_axes import (FRAG_P, K, LEV, M, NPIX,  # noqa: E402
                                        NSIDE, SEEDS, TRACKED3, _cells,
                                        _p_store, _put, _run, _same,
                                        _vstore)

ALL5 = ["avg_uoe", "avg_von", "avg_wo", "avg_thetao", "avg_so"]
LAYERS4 = ["avg_uoe", "avg_von", "avg_thetao", "avg_so"]
TOOL_V11 = "extract_healpix_region v1.1"
SURFACE_NOTE = "a surface extract for 2-D runs"
FRAG_XP = "region.max_levels would drop that level"
FRAG_XW = "must sit on exactly the K interfaces"


def _xcfg(src, out, *, levels=None, max_levels=None, variables=ALL5,
          bbox=(-30, 30, -30, 30), halo=2, var_u=None, var_w=None,
          time=None):
    """The extractor config dict (``variables=None``: the default list)."""
    cfg = {
        "source": {"path": str(src)},
        "region": {"bbox": [float(b) for b in bbox], "halo_layers": halo},
        "output": {"path": str(out), "time_chunk": 2},
        "metadata": {"subset_name": "t", "description": "test"},
    }
    if levels is not None:
        cfg["source"]["levels"] = [float(v) for v in levels]
    if var_u is not None:
        cfg["source"]["var_u"] = var_u
    if var_w is not None:
        cfg["source"]["var_w"] = var_w
    if max_levels is not None:
        cfg["region"]["max_levels"] = max_levels
    if variables is not None:
        cfg["variables"] = list(variables)
    if time is not None:
        cfg["time"] = dict(time)
    return cfg


def _src(tmp_path, kind, tag="src", **kw):
    """Source A, P or T (module docstring) as NetCDF in tmp_path."""
    path = tmp_path / f"{kind}_{tag}.nc"
    if kind == "A":
        return _vstore(path, **kw)
    if kind == "P":
        return _p_store(path, **kw)
    if kind == "T":
        return _vstore(path, layer_dim="lev", w_dim="level",
                       embed_table=True, **kw)
    raise ValueError(kind)


def _extract(tmp_path, src, tag, *, dry_run=False, **kw):
    """Run the extractor into tmp_path/f"{tag}.zarr"; returns that path."""
    out = tmp_path / f"{tag}.zarr"
    extract_region(_xcfg(src, out, **kw), dry_run=dry_run)
    return out


def _bed(idx):
    """The bed the reader computes on the whole fixture store at the NEST
    cells ``idx``: LEV[min(nfin, K-1)], 0 where u has no finite level."""
    _, lat, nfin = _cells()
    nf = np.where(lat < 60.0, nfin, 0)[np.asarray(idx)]
    return np.where(nf > 0, np.asarray(LEV)[np.minimum(nf, K - 1)], 0.0)


def _rename(old, new):
    def edit(ds):
        return ds.rename({old: new})
    return edit


def _sizes(ds, name):
    return dict(ds[name].sizes)


# ---------------------------------------------------------------------------
# 8a, 8b, 8g: the cut with a table
# ---------------------------------------------------------------------------

def test_cut_w_on_its_own_axis(tmp_path):
    """A (w on its own ``level_w`` axis), ``source.levels`` + max_levels M:
    w M, the layer variables M-1, the table M, the source's bed."""
    a = _src(tmp_path, "A")
    out = _extract(tmp_path, a, "ext", levels=LEV, max_levels=M)
    ds = xr.open_zarr(out)
    assert _sizes(ds, "avg_wo")["level_w"] == M
    for nm in LAYERS4:
        assert _sizes(ds, nm)["level"] == M - 1, nm
    np.testing.assert_array_equal(ds["level_interfaces_m"].values, LEV[:M])
    assert "deptho" in ds.variables
    assert ds["deptho"].dims == ("value",)
    idx = ds["healpix_index"].values
    np.testing.assert_array_equal(ds["deptho"].values, _bed(idx))
    assert ds["deptho"].attrs["units"] == "m"
    assert ds["deptho"].attrs["positive"] == "down"
    assert "deptho" in list(ds.attrs["variables"])
    assert ds.attrs["tool"] == TOOL_V11


def _level_coord(ds):
    return ds.assign_coords(level=np.arange(ds.sizes["level"]))


@pytest.mark.parametrize("coord", [False, True],
                         ids=["no_coordinate", "index_coordinate"])
def test_cut_shared_level(tmp_path, coord):
    """P (w and the layers on one ``level`` axis): w's dim is renamed to
    ``level_w`` (and its index coordinate with it), so the layers keep M-1
    entries and are not padded back."""
    p = _src(tmp_path, "P", edit=_level_coord if coord else None)
    out = _extract(tmp_path, p, "ext", levels=LEV, max_levels=M)
    ds = xr.open_zarr(out)
    assert ds["avg_wo"].dims == ("time", "level_w", "value")
    assert ds.sizes["level_w"] == M
    for nm in LAYERS4:
        assert ds[nm].dims == ("time", "level", "value"), nm
        assert ds.sizes["level"] == M - 1
    assert np.isfinite(ds["avg_uoe"].isel(level=-1).values).any()
    if coord:
        assert "level_w" in ds.coords
        assert ds["level_w"].values.size == M
        np.testing.assert_array_equal(ds["level_w"].values, np.arange(M))
        np.testing.assert_array_equal(ds["level"].values, np.arange(M - 1))
    assert "deptho" in ds.variables


def test_cut_shared_level_with_depth_coordinate(tmp_path):
    """P with a non-dimension coordinate ``depth(level)``: renamed with w
    (``depth_w*`` on ``level_w``), no MergeError, lengths right after the
    reopen."""
    def depth_coord(ds):
        return ds.assign_coords(depth=("level", np.asarray(LEV)))

    p = _src(tmp_path, "P", edit=depth_coord)
    out = _extract(tmp_path, p, "ext", levels=LEV, max_levels=M)
    ds = xr.open_zarr(out)
    assert ds.sizes["level"] == M - 1
    assert ds.sizes["level_w"] == M
    assert ds["depth"].dims == ("level",)
    np.testing.assert_array_equal(ds["depth"].values, LEV[:M - 1])
    dw = [c for c in ds.variables if str(c).startswith("depth_w")]
    assert len(dw) == 1, dw
    assert ds[dw[0]].dims == ("level_w",)
    np.testing.assert_array_equal(ds[dw[0]].values, LEV[:M])


@pytest.mark.parametrize("coord", [False, True],
                         ids=["no_coordinate", "index_coordinate"])
def test_rename_picks_an_unused_name(tmp_path, coord):
    """A source variable already called ``level_w`` (and listed): w's dim
    takes ``level_w_1``, the variable ``level_w`` is carried untouched."""
    def edit(ds):
        if coord:
            ds = _level_coord(ds)
        ds["level_w"] = (("time", "value"), np.full((ds.sizes["time"], NPIX),
                                                    7.0))
        return ds

    p = _src(tmp_path, "P", edit=edit)
    out = _extract(tmp_path, p, "ext", levels=LEV, max_levels=M,
                   variables=ALL5 + ["level_w"])
    ds = xr.open_zarr(out)
    assert ds["avg_wo"].dims == ("time", "level_w_1", "value")
    assert ds.sizes["level_w_1"] == M
    assert "level_w" in ds.data_vars
    assert ds["level_w"].dims == ("time", "value")
    assert (ds["level_w"].values == 7.0).all()


# ---------------------------------------------------------------------------
# 8c: a run on the cut extract equals the run on the source with max_levels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("uv_vertical", ["layer", "linear"])
@pytest.mark.parametrize("kind", ["A", "P"])
def test_run_on_cut_extract_equals_source_with_max_levels(tmp_path, kind,
                                                          uv_vertical):
    """The default seeds (two lie below 70 m in a 5-layer column): the run
    on the extract equals, bit for bit, the run on the source with
    ``mode.max_levels: M`` — same layers, same w, same bed."""
    src = _src(tmp_path, kind)
    out = _extract(tmp_path, src, "ext", levels=LEV, max_levels=M)
    on_ext = _run(tmp_path, out, "ext", levels=None, uv_vertical=uv_vertical)
    on_src = _run(tmp_path, src, "src", levels=LEV, max_levels=M,
                  uv_vertical=uv_vertical)
    _same(on_ext, on_src)
    # not vacuous: the two deep seeds of the 5-layer column stay below the
    # deepest kept interface (70 m), which the bed of the source allows
    assert SEEDS[1][2] > LEV[M - 1] and SEEDS[2][2] > LEV[M - 1]
    assert (on_ext["z"][1:3, -1] > LEV[M - 1]).all(), on_ext["z"][1:3]


# ---------------------------------------------------------------------------
# 8d: refusals before the cut (nothing written)
# ---------------------------------------------------------------------------

_REFUSALS = {
    # leg: (source, source kwargs, extractor kwargs, exception, fragments)
    "w_k_minus_1": (
        "A", dict(edit=_put("avg_wo", K - 1, dim="level_w")),
        dict(levels=LEV), ValueError,
        (FRAG_XW, "'avg_wo' has 5 entries on 'level_w'",
         "(source.levels) has K = 6 interfaces")),
    "thetao_k_plus_1": (
        "A", dict(edit=_put("avg_thetao", K + 1)), dict(levels=LEV),
        ValueError,
        ("the layer variable 'avg_thetao' has 7 entries on 'level_x'",
         "a layer variable needs 5 entries")),
    "thetao_k_minus_2": (
        "A", dict(edit=_put("avg_thetao", K - 2)), dict(levels=LEV),
        ValueError,
        ("the layer variable 'avg_thetao' has 4 entries on 'level_x'",
         "a layer variable needs 5 entries")),
    "p_table_one_short": (
        "P", {}, dict(levels=LEV[:5], max_levels=3), ValueError,
        (FRAG_XW, "'avg_wo' has 6 entries on 'level'",
         "(source.levels) has K = 5 interfaces")),
    "u_live_tail": (
        "A", dict(edit=_put("avg_uoe", K, value=1.0)),
        dict(levels=LEV, max_levels=M), ValueError,
        (FRAG_XP, "'avg_uoe' has 6 entries on 'level_x'",
         "in source record 0 (time 2020-01-02")),
    # any non-NaN value in the K-th plane is data: 0.0 and +-inf too
    "u_zero_tail": (
        "A", dict(edit=_put("avg_uoe", K, value=0.0)),
        dict(levels=LEV, max_levels=M), ValueError,
        (FRAG_XP, "'avg_uoe' has 6 entries on 'level_x'",
         "in source record 0 (time 2020-01-02")),
    "u_posinf_tail": (
        "A", dict(edit=_put("avg_uoe", K, value=np.inf)),
        dict(levels=LEV, max_levels=M), ValueError,
        (FRAG_XP, "'avg_uoe' has 6 entries on 'level_x'",
         "in source record 0 (time 2020-01-02")),
    "u_neginf_tail": (
        "A", dict(edit=_put("avg_uoe", K, value=-np.inf)),
        dict(levels=LEV, max_levels=M), ValueError,
        (FRAG_XP, "'avg_uoe' has 6 entries on 'level_x'",
         "in source record 0 (time 2020-01-02")),
    "max_levels_0_table": (
        "A", {}, dict(levels=LEV, max_levels=0), ValueError,
        ("region.max_levels: 0 is refused",)),
    "max_levels_0_no_table": (
        "A", {}, dict(max_levels=0), ValueError,
        ("region.max_levels: 0 is refused",)),
    "var_w_absent": (
        "A", {}, dict(levels=LEV, var_w="nope", variables=None), KeyError,
        ("'nope' not found",)),
    "var_w_absent_listed": (
        "A", {}, dict(levels=LEV, var_w="nope",
                      variables=["avg_uoe", "avg_von"]), KeyError,
        ("'nope' not found",)),
    "var_w_not_listed": (
        "A", {}, dict(levels=LEV, var_w="avg_wo",
                      variables=["avg_uoe", "avg_von"]), ValueError,
        ("source.var_w 'avg_wo' is not in variables:",)),
    "t_conflicting_levels": (
        "T", {}, dict(levels=LEV[:5]), ValueError,
        ("conflicts with the interface table", "(level_interfaces_m, 6")),
}


@pytest.mark.parametrize("leg", list(_REFUSALS))
def test_refusals_before_the_cut(tmp_path, leg):
    kind, skw, xkw, exc, frags = _REFUSALS[leg]
    src = _src(tmp_path, kind, **skw)
    out = tmp_path / "ext.zarr"
    with pytest.raises(exc) as ei:
        extract_region(_xcfg(src, out, **xkw))
    for frag in frags:
        assert frag in str(ei.value), (frag, str(ei.value))
    assert not out.exists()


@pytest.mark.parametrize("leg", ["my_w", "avg_zos_as_w"])
def test_w_under_a_non_alias_name(tmp_path, leg):
    """``source.var_w`` names w outside the aliases: it is the interface
    family (M entries, on its own ``level_w``); a selected w without a
    vertical dim is refused."""
    if leg == "my_w":
        a = _src(tmp_path, "A", edit=_rename("avg_wo", "my_w"))
        out = _extract(tmp_path, a, "ext", levels=LEV, max_levels=M,
                       var_w="my_w",
                       variables=["avg_uoe", "avg_von", "my_w",
                                  "avg_thetao", "avg_so"])
        ds = xr.open_zarr(out)
        assert ds["my_w"].dims == ("time", "level_w", "value")
        assert ds.sizes["level_w"] == M
        for nm in LAYERS4:
            assert _sizes(ds, nm)["level"] == M - 1, nm
        return
    a = _src(tmp_path, "A", tag="zos")
    out = tmp_path / "ext_zos.zarr"
    with pytest.raises(ValueError, match="has no vertical dim") as ei:
        extract_region(_xcfg(a, out, levels=LEV, var_w="avg_zos",
                             variables=ALL5 + ["avg_zos"]))
    assert "the vertical velocity 'avg_zos'" in str(ei.value)
    assert not out.exists()


@pytest.mark.parametrize("max_levels", [K, 99])
def test_cut_at_or_beyond_the_table(tmp_path, max_levels):
    """max_levels >= K is still a cut: the padding plane of the layer
    variables goes (K-1 entries), w keeps K on ``level_w``; no bed (the
    table is not shortened)."""
    p = _src(tmp_path, "P")
    out = _extract(tmp_path, p, "ext", levels=LEV, max_levels=max_levels)
    ds = xr.open_zarr(out)
    assert ds["avg_wo"].dims == ("time", "level_w", "value")
    assert ds.sizes["level_w"] == K
    for nm in LAYERS4:
        assert _sizes(ds, nm)["level"] == K - 1, nm
    assert np.isfinite(ds["avg_uoe"].isel(level=-1).values).any()
    np.testing.assert_array_equal(ds["level_interfaces_m"].values, LEV)
    assert "deptho" not in ds.variables


def test_bed_uses_source_var_u(tmp_path):
    """u outside the aliases (``source.var_u``) still gives the bed."""
    a = _src(tmp_path, "A", edit=_rename("avg_uoe", "my_u"))
    out = _extract(tmp_path, a, "ext", levels=LEV, max_levels=M,
                   var_u="my_u",
                   variables=["my_u", "avg_von", "avg_wo", "avg_thetao",
                              "avg_so"])
    ds = xr.open_zarr(out)
    assert "deptho" in ds.variables
    np.testing.assert_array_equal(ds["deptho"].values,
                                  _bed(ds["healpix_index"].values))


def test_lengths_checked_in_the_written_store(tmp_path, monkeypatch):
    """The lengths are checked again on the written store, reopened: a
    reopen that shows u one entry short on its level dim is a RuntimeError
    naming the variable, the dim, the found and the expected length; the
    written store stays on disk.  (u alone on ``level_x``, K-1 entries, so
    one variable can be short.)"""
    a = _src(tmp_path, "A", edit=_put("avg_uoe", K - 1))
    out = tmp_path / "ext.zarr"
    open_zarr = xr.open_zarr
    reopened = []

    def short_reopen(store, *args, **kwargs):
        reopened.append(str(store))
        return open_zarr(store, *args, **kwargs).isel(
            level_x=slice(0, M - 2))

    monkeypatch.setattr(xr, "open_zarr", short_reopen)
    with pytest.raises(RuntimeError) as ei:
        extract_region(_xcfg(a, out, levels=LEV, max_levels=M))
    msg = str(ei.value)
    assert "in the written store" in msg, msg
    assert "'avg_uoe' has 2 entries on 'level_x', expected 3" in msg, msg
    assert reopened == [str(out)]
    assert out.exists()
    with open_zarr(out) as ds:
        assert ds["avg_uoe"].dims == ("time", "level_x", "value")
        assert ds.sizes["level_x"] == M - 1


def test_two_store_eta_with_a_table(tmp_path):
    """The eta from the 2-D sibling store (not in the main store) goes
    through the same checks as every other extracted variable."""
    a = _vstore(tmp_path / "3D_daily_x_ocean.nc", with_eta=False)
    _make_eta_store(tmp_path / "2D_daily_x_ocean.nc", nside=NSIDE)
    out = _extract(tmp_path, a, "ext", levels=LEV, max_levels=M)
    ds = xr.open_zarr(out)
    assert "avg_zos" in ds.variables
    assert ds.attrs["eta_source"].endswith("2D_daily_x_ocean.nc")
    assert ds.sizes["level_w"] == M
    for nm in LAYERS4:
        assert _sizes(ds, nm)["level"] == M - 1, nm
    assert ds.sizes["level_interface"] == M
    assert "deptho" in ds.variables

    out2 = _extract(tmp_path, a, "ext_nocut", levels=LEV)
    ds2 = xr.open_zarr(out2)
    assert "avg_zos" in ds2.variables
    np.testing.assert_array_equal(ds2["level_interfaces_m"].values, LEV)


# ---------------------------------------------------------------------------
# max_levels: 1 = the table-less surface cut (S4 (b))
# ---------------------------------------------------------------------------

def _with_deptho(ds):
    _, lat, _ = _cells()
    ds["deptho"] = (("value",), np.where(lat < 60.0, 123.0, np.nan))


_SURFACE = {
    # leg: (source, source kwargs, source.levels, 2-D run check)
    "t_embedded": ("T", {}, None, True),
    "a_levels": ("A", {}, LEV, True),
    "t_conflicting_levels": ("T", {}, LEV[:5], False),
    "a_levels_source_deptho": ("A", dict(edit=_with_deptho), LEV, False),
}


@pytest.mark.parametrize("leg", list(_SURFACE))
def test_surface_extract_with_a_table(tmp_path, capsys, leg):
    """``max_levels: 1`` takes the table-less surface cut whatever table
    the source has or the config gives: every vertical axis 1 entry, no
    table, a printed note, no synthesised bed (the source's own depth
    variable is carried)."""
    kind, skw, levels, run2d = _SURFACE[leg]
    src = _src(tmp_path, kind, **skw)
    capsys.readouterr()
    out = _extract(tmp_path, src, "ext", levels=levels, max_levels=1)
    printed = capsys.readouterr().out
    ds = xr.open_zarr(out)
    want_dims = {"lev", "level"} if kind == "T" else {"level", "level_w"}
    vdims = {str(d) for d in ds.dims
             if str(d) == "lev" or str(d).startswith("level")}
    assert want_dims <= vdims, vdims
    for d in vdims:
        assert ds.sizes[d] == 1, (d, ds.sizes[d])
    assert "level_interfaces_m" not in ds.variables
    assert SURFACE_NOTE in printed, printed
    if leg == "a_levels_source_deptho":
        assert "deptho" in ds.variables
        with xr.open_dataset(src) as s:
            want = s["deptho"].values[ds["healpix_index"].values]
        np.testing.assert_array_equal(ds["deptho"].values, want)
    else:
        assert "deptho" not in ds.variables
    if run2d:
        seeds = list(dict.fromkeys((lo, la) for lo, la, _ in SEEDS))
        two_d = dict(levels=None, seeds=seeds, Fdrift2D=True,
                     Fdrift3D=False, sigma_layer=0)
        _same(_run(tmp_path, out, "ext", **two_d),
              _run(tmp_path, src, "src", **two_d))


# ---------------------------------------------------------------------------
# the tail check: every exported record, the exported cells only, ring order
# ---------------------------------------------------------------------------

def _thetao_tail_at_record_2(ds):
    a = ds["avg_thetao"].values.copy()
    assert np.isnan(a[:, K - 1]).all()
    a[2, K - 1, list(TRACKED3)] = 1.0
    ds["avg_thetao"] = (ds["avg_thetao"].dims, a)


@pytest.mark.parametrize("leg", ["start_0", "start_2", "stop_2"])
def test_tail_check_reads_every_exported_record(tmp_path, leg):
    """The K-th plane of ``avg_thetao`` is empty at records 0 and 1 and
    carries 1.0 at record 2 inside the sector: refused while record 2 is
    exported (named by its index on the SOURCE's time axis), written when
    the time selection excludes it."""
    p = _src(tmp_path, "P", edit=_thetao_tail_at_record_2)
    time = {"start_0": {"start": 0}, "start_2": {"start": 2},
            "stop_2": {"stop": 2}}[leg]
    out = tmp_path / "ext.zarr"
    cfg = _xcfg(p, out, levels=LEV, max_levels=M, time=time)
    if leg == "stop_2":
        extract_region(cfg)
        ds = xr.open_zarr(out)
        assert ds.sizes["time"] == 2
        for nm in LAYERS4:
            assert _sizes(ds, nm)["level"] == M - 1, nm
        assert ds["avg_wo"].dims == ("time", "level_w", "value")
        assert ds.sizes["level_w"] == M
        assert ds.sizes["level_interface"] == M
        return
    with pytest.raises(ValueError, match=FRAG_XP) as ei:
        extract_region(cfg)
    msg = str(ei.value)
    assert "'avg_thetao' has 6 entries on 'level'" in msg
    assert "source record 2" in msg and "2020-01-04" in msg, msg
    assert not out.exists()


def test_tail_outside_the_sector_is_accepted(tmp_path):
    """Data in the K-th plane at a sea cell outside bbox + halo is not
    exported, so the cut drops nothing of the extract."""
    c_out = int(healpy.ang2pix(NSIDE, 150.0, -45.0, nest=True, lonlat=True))

    def edit(ds):
        a = ds["avg_uoe"].values.copy()
        a[:, K - 1, c_out] = 1.0
        ds["avg_uoe"] = (ds["avg_uoe"].dims, a)

    p = _src(tmp_path, "P", edit=edit)
    out = _extract(tmp_path, p, "ext", levels=LEV, max_levels=M)
    ds = xr.open_zarr(out)
    assert c_out not in set(ds["healpix_index"].values.tolist())
    assert ds.sizes["level"] == M - 1
    assert ds.sizes["level_w"] == M


def _as_ring(extra=None):
    """An ``edit``: the store written in RING order (``healpix_order:
    ring``), after ``extra`` (on the NEST-ordered dataset) when given."""
    def edit(ds):
        if extra is not None:
            ds = extra(ds) or ds
        ds = ds.isel(value=healpy.ring2nest(NSIDE, np.arange(NPIX)))
        ds["crs"] = ds["crs"].assign_attrs(healpix_order="ring")
        return ds
    return edit


def test_ring_source_cut_bed_and_tail(tmp_path):
    """A ring-ordered source: the bed and the tail check address the
    source by RING position and the extract is written in NEST order."""
    p = _src(tmp_path, "P", tag="ring", edit=_as_ring())
    out = _extract(tmp_path, p, "ext", levels=LEV, max_levels=M)
    ds = xr.open_zarr(out)
    idx = ds["healpix_index"].values
    assert ds.attrs["source_order"] == "ring"
    np.testing.assert_array_equal(ds["deptho"].values, _bed(idx))

    # (b) one in-sector cell c whose RING position is not itself one of
    # the exported NEST ids: only a ring-position gather sees its tail
    in_sector = set(idx.tolist())
    cands = [int(c) for c in idx
             if int(healpy.nest2ring(NSIDE, int(c))) not in in_sector]
    c = 0 if 0 in cands else cands[0]
    assert int(healpy.nest2ring(NSIDE, c)) not in in_sector

    def tail_at_c(ds):
        a = ds["avg_uoe"].values.copy()
        a[:, K - 1, c] = 1.0          # NEST id c, before the ring reorder
        ds["avg_uoe"] = (ds["avg_uoe"].dims, a)

    p2 = _src(tmp_path, "P", tag="ring_tail", edit=_as_ring(tail_at_c))
    out2 = tmp_path / "ext_tail.zarr"
    with pytest.raises(ValueError, match=FRAG_XP) as ei:
        extract_region(_xcfg(p2, out2, levels=LEV, max_levels=M))
    assert "'avg_uoe'" in str(ei.value)
    assert "at 1 of the" in str(ei.value)
    assert not out2.exists()


# ---------------------------------------------------------------------------
# 8e: unchanged behaviour and the new defaults
# ---------------------------------------------------------------------------

_DEFAULT_LEGS = ["a_surface_no_table", "default_list", "t_embedded_no_cut",
                 "t_cut", "var_w_default_list", "p_table_no_cut",
                 "p_cut_no_table"]


@pytest.mark.parametrize("leg", _DEFAULT_LEGS)
def test_unchanged_and_new_defaults(tmp_path, leg):
    if leg == "a_surface_no_table":
        out = _extract(tmp_path, _src(tmp_path, "A"), "ext", max_levels=1,
                       variables=["avg_uoe", "avg_von"])
        ds = xr.open_zarr(out)
        assert ds.sizes["level"] == 1
        assert "level_interfaces_m" not in ds.variables
        assert "deptho" not in ds.variables
    elif leg == "default_list":
        out = _extract(tmp_path, _src(tmp_path, "A"), "ext", variables=None)
        ds = xr.open_zarr(out)
        assert list(ds.attrs["variables"]) == ["avg_uoe", "avg_von",
                                               "avg_zos"]
    elif leg == "t_embedded_no_cut":
        out = _extract(tmp_path, _src(tmp_path, "T"), "ext")
        ds = xr.open_zarr(out)
        np.testing.assert_array_equal(ds["level_interfaces_m"].values, LEV)
        assert ds.sizes["lev"] == K - 1 and ds.sizes["level"] == K
    elif leg == "t_cut":
        out = _extract(tmp_path, _src(tmp_path, "T"), "ext", max_levels=M)
        ds = xr.open_zarr(out)
        assert ds.sizes["lev"] == M - 1
        assert ds.sizes["level"] == M
        np.testing.assert_array_equal(ds["level_interfaces_m"].values,
                                      LEV[:M])
        assert ds["avg_wo"].dims == ("time", "level", "value")
        assert not any(str(d).startswith("level_w") for d in ds.dims)
    elif leg == "var_w_default_list":
        out = _extract(tmp_path, _src(tmp_path, "A"), "ext", variables=None,
                       var_w="avg_wo")
        ds = xr.open_zarr(out)
        assert "avg_wo" in ds.data_vars
        assert list(ds.attrs["variables"]) == ["avg_uoe", "avg_von",
                                               "avg_wo", "avg_zos"]
    elif leg == "p_table_no_cut":
        out = _extract(tmp_path, _src(tmp_path, "P"), "ext", levels=LEV)
        ds = xr.open_zarr(out)
        assert ds["avg_wo"].dims == ("time", "level", "value")
        assert ds.sizes["level"] == K
        assert not any(str(d).startswith("level_w") for d in ds.dims)
        assert "deptho" not in ds.variables
        np.testing.assert_array_equal(ds["level_interfaces_m"].values, LEV)
    else:   # p_cut_no_table
        out = _extract(tmp_path, _src(tmp_path, "P"), "ext", max_levels=M)
        ds = xr.open_zarr(out)
        assert ds["avg_wo"].dims == ("time", "level", "value")
        assert ds.sizes["level"] == M
        assert "level_interfaces_m" not in ds.variables
        assert "deptho" not in ds.variables
        # a v1.1 extract gets the full rule: the M-th layer carries data
        with pytest.raises(ValueError, match=FRAG_P):
            _run(tmp_path, out, "ext", levels=LEV[:M])
    assert ds.attrs["tool"] == TOOL_V11


# ---------------------------------------------------------------------------
# --dry-run (S5)
# ---------------------------------------------------------------------------

def _recording_open(reads):
    """An ``_open_healpix_store`` for the extractor module: the store read
    into memory, every variable on the cell dim a dask array that appends
    its name to ``reads`` whenever a block of it is computed (a data
    read); dims, sizes and attributes stay readable without a read."""
    from kelp.readers.healpix import _open_healpix_store

    def opener(path):
        with _open_healpix_store(path) as ds:
            ds = ds.load()
        for nm in list(ds.data_vars):
            if "value" not in ds[nm].dims:
                continue

            def logged(block, nm=nm):
                reads.append(nm)
                return block

            arr = ds[nm].chunk().data
            ds[nm] = ds[nm].copy(data=arr.map_blocks(
                logged, dtype=arr.dtype, meta=np.array((), arr.dtype)))
        return ds
    return opener


def test_dry_run_checks_metadata_only(tmp_path, capsys, monkeypatch):
    """A dry run makes every metadata check (X-L refuses) but reads no
    plane: the last-level check is announced as not run, the bed of the
    cut (``keep < K``, no depth variable in the source) is not computed,
    no data variable is read, nothing is written.  The same config
    without ``--dry-run`` reads u (the recorder sees reads) and is
    refused by the last-level check."""
    import extract_healpix_region as xhr

    reads = []
    monkeypatch.setattr(xhr, "_open_healpix_store", _recording_open(reads))
    a = _src(tmp_path, "A", tag="xl", edit=_put("avg_thetao", K + 1))
    out = tmp_path / "ext_xl.zarr"
    with pytest.raises(ValueError, match="a layer variable needs 5 entries"):
        extract_region(_xcfg(a, out, levels=LEV), dry_run=True)
    assert not out.exists()

    a2 = _src(tmp_path, "A", tag="xp", edit=_put("avg_uoe", K, value=1.0))
    out2 = tmp_path / "ext_xp.zarr"
    capsys.readouterr()
    extract_region(_xcfg(a2, out2, levels=LEV, max_levels=M), dry_run=True)
    printed = capsys.readouterr().out
    assert "dry run — the last-level check of" in printed, printed
    assert "avg_uoe" in printed.split("last-level check of", 1)[1]
    assert reads == [], reads
    assert not out2.exists()

    with pytest.raises(ValueError, match=FRAG_XP):
        extract_region(_xcfg(a2, out2, levels=LEV, max_levels=M))
    assert "avg_uoe" in reads, reads
    assert not out2.exists()
