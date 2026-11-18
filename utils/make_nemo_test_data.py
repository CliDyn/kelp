#!/usr/bin/env python
"""Build the ``KELP_TEST_NEMO_DATA`` directory of the data-gated NEMO tests.

``tests/test_nemo_cgrid_data.py`` and ``tests/test_nemo_cgrid_lagrangian.py``
skip unless ``KELP_TEST_NEMO_DATA`` points at a directory of NEMO crops.  This
script downloads the public sources and writes the crops in the layout the tests
expect::

    <out>/orca05_{arctic,southern,equatorial}_{mesh,grid_T,grid_U,grid_V,grid_W}.nc
    <out>/orca05_equatorial_split/orca05_1m_2011{0101_20110228,0301_20110331}_grid_{T,U,V,W}.nc
    <out>/eorca025_{arctic,southern,equatorial}_{domain_cfg,T,U,V,W}.nc
    <out>/eorca025_equatorial_split/eorca025_1m_1976{0101_19760229,0301_19760331}_grid_{T,U,V,W}.nc
    <out>/nemo_orca05/{mesh_mask.nc,mesh_mask_3d.nc,ORCA05_1step3d_grid_{T,U,V,W}.nc}
    <out>/esgf_orca2/data/{thetao,so,uo,vo}_t0.nc, {areacello,deptho,sftof,volcello}_fx_*.nc

(the tests also look for ``nemo_orca05/`` and ``esgf_orca2/data/`` next to or
one/two levels above ``KELP_TEST_NEMO_DATA``; they are written below ``--out``
here, which the tests find directly).

Sources
-------
orca05
    ORCA05 / FOCI1.14 test data, Zenodo record 3755721
    (DOI 10.5281/zenodo.3755721), licence **CC-BY-4.0** (J. Harlass and W. Rath,
    2020; "no way suitable for scientific use").  Needs the 2011 files
    ``mesh_mask.nc`` + ``..._20110101_20111231_grid_{T,U,V,W}.nc`` = 3.8 GB,
    downloaded into ``<out>/_sources/orca05/`` (resumable; ``--drop-sources``
    removes them afterwards).  Plain HTTPS, no login.
eorca025
    NOC SE-NEMO eORCA025 JRA55 v1, zarr v3 store
    ``https://noc-msm-o.s3-ext.jc.rl.ac.uk/npd-eorca025-jra55v1`` (domain_cfg,
    U1m_4d, V1m_4d, W1m_4d, T1m_4d).  The licence of this public data is
    **unstated**: use for local testing only, do NOT redistribute the crops.
    Only the crops are read over the network (a few hundred MB).  Needs ``zarr>=3``,
    ``fsspec`` and ``aiohttp`` (``pip install -e '.[zarr]' fsspec aiohttp``).
orca2
    IPSL-CM5A-LR (ORCA2, CMIP5) from ESGF/CEDA: first record of ``uo vo thetao so``
    (historical, Omon, r1i1p1, 2000-2005 file) through OPeNDAP (needs a netCDF4
    library built with DAP support, as the conda-forge/PyPI wheels are) and the
    fixed fields ``areacello deptho sftof volcello`` (1pctCO2, fx, r0i0p0) over
    HTTPS.  CMIP5 terms of use
    (https://pcmdi.llnl.gov/mips/cmip5/terms-of-use.html): unrestricted use,
    acknowledge the providers, pass the terms on when redistributing.  ``wo`` is not fetched (not served).

If a download fails (CEDA and NOC endpoints move; ESGF sometimes asks for a login)
the script prints the URL and the file name to place under the target directory
and carries on with the other sources.

Usage
-----
::

    python utils/make_nemo_test_data.py all                # everything
    python utils/make_nemo_test_data.py orca05 --dry-run   # list what would be fetched
    python utils/make_nemo_test_data.py all --verify       # + consistency checks
    python utils/make_nemo_test_data.py verify             # checks only
    export KELP_TEST_NEMO_DATA=$PWD/.tmp/data/nemo_crops

Existing output files are never overwritten (idempotent; delete a file to redo it).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np
import xarray as xr

# ----------------------------------------------------------------------------------
# sources and licences
# ----------------------------------------------------------------------------------
ZENODO_RECORD = "https://zenodo.org/api/records/3755721/files"
ZENODO_PREFIX = "FOCI1.14-TEST.2020.04.1_1m_20110101_20111231_grid_"
ORCA05_SOURCES = {          # name -> (file name in the record, size in bytes)
    "mesh_mask.nc": ("mesh_mask.nc", 52325827),
    "2011_grid_T.nc": (ZENODO_PREFIX + "T.nc", 648387081),
    "2011_grid_U.nc": (ZENODO_PREFIX + "U.nc", 727753005),
    "2011_grid_V.nc": (ZENODO_PREFIX + "V.nc", 728999242),
    "2011_grid_W.nc": (ZENODO_PREFIX + "W.nc", 1588032163),
}
ORCA05_LICENCE = "CC-BY-4.0 (Zenodo 10.5281/zenodo.3755721, FOCI1.14 test data, Harlass & Rath 2020)"
EORCA025_URL = "https://noc-msm-o.s3-ext.jc.rl.ac.uk/npd-eorca025-jra55v1/"
EORCA025_LICENCE = "licence of the NOC public data is unstated; use for local testing only, do not redistribute"
ORCA2_LICENCE = "CMIP5 terms of use, https://pcmdi.llnl.gov/mips/cmip5/terms-of-use.html"
CEDA = "https://esgf.ceda.ac.uk/thredds"
IPSL = "esg_dataroot/cmip5/output1/IPSL/IPSL-CM5A-LR"
ORCA2_OPENDAP = f"{CEDA}/dodsC/{IPSL}/historical/mon/ocean/Omon/r1i1p1/v20111119"
ORCA2_FX_URL = f"{CEDA}/fileServer/{IPSL}/1pctCO2/fx/ocean/fx/r0i0p0/latest"
ORCA2_VARS = ["uo", "vo", "thetao", "so"]
ORCA2_FX = ["areacello", "deptho", "sftof", "volcello"]

METRICS = ["glamt", "glamu", "glamv", "glamf", "gphit", "gphiu", "gphiv", "gphif", "e1t", "e1u", "e1v", "e1f",
           "e2t", "e2u", "e2v", "e2f", "e3t_0", "e3u_0", "e3v_0", "e3w_0", "e3t_1d", "e3w_1d", "gdept_1d",
           "gdepw_1d", "gdept_0", "gdepw_0", "tmask", "umask", "vmask", "fmask", "mbathy", "bottom_level",
           "top_level"]
# variables of the 2-D ``nemo_orca05/mesh_mask.nc`` (first level of the 3-D mesh_mask)
MESH_MASK_2D = ["nav_lon", "nav_lat", "nav_lev", "time_counter", "tmask", "umask", "vmask", "fmask", "glamt",
                "glamu", "glamv", "glamf", "gphit", "gphiu", "gphiv", "gphif", "e1t", "e1u", "e1v", "e1f", "e2t",
                "e2u", "e2v", "e2f", "ff", "mbathy"]


# ----------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------
def _write(ds: xr.Dataset, path: Path, attrs: dict | None = None) -> None:
    """Write ``ds`` to NetCDF (zlib level 4 on 2-D+ numeric variables) unless it exists.

    Parameters
    ----------
    ds : xarray.Dataset
        Loaded dataset.
    path : pathlib.Path
        Target file; skipped (with a message) if present, parent created.
    attrs : dict, optional
        Global attributes added to a copy of ``ds``.
    """
    if path.exists():
        print(f"  skip (exists) {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = ds.copy()
    ds.attrs.update(attrs or {})
    enc = {k: dict(zlib=True, complevel=4) for k, v in ds.data_vars.items() if v.dtype.kind in "fiu" and v.ndim >= 2}
    tmp = path.with_name(path.name + ".part")
    ds.to_netcdf(tmp, encoding=enc)
    tmp.rename(path)
    print(f"  wrote {path.name} ({path.stat().st_size / 1e6:.1f} MB)", flush=True)


def _download(url: str, dest: Path, size: int | None = None) -> bool:
    """Download ``url`` to ``dest`` (resumable via ``dest.part``); False and a manual hint on failure.

    Parameters
    ----------
    url : str
        HTTPS source.
    dest : pathlib.Path
        Final file; skipped when present (its size is not checked).
    size : int, optional
        Expected size in bytes (checked after the download).

    Returns
    -------
    bool
        True when ``dest`` exists afterwards.
    """
    if dest.exists():
        print(f"  skip (exists) {dest.name}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have and size is not None and have == size:
        part.rename(dest)
        return True
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    try:
        print(f"  download {url}", flush=True)
        with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have and r.status == 206 else "wb") as f:
            shutil.copyfileobj(r, f, 1 << 22)
    except Exception as exc:                                  # noqa: BLE001 - report, never crash a multi-source run
        print(f"  FAILED ({exc!r}).\n  Manual: download {url}\n  and save it as {dest}", file=sys.stderr)
        return False
    if size is not None and part.stat().st_size != size:
        print(f"  size mismatch for {dest.name}: {part.stat().st_size} != {size}; delete {part} and retry",
              file=sys.stderr)
        return False
    part.rename(dest)
    return True


def _sel(ds, j0, j1, i0, i1):
    return ds.isel(y=slice(j0, j1), x=slice(i0, i1))


# ----------------------------------------------------------------------------------
# ORCA05
# ----------------------------------------------------------------------------------
def orca05_urls() -> list[tuple[str, str]]:
    """Return ``[(url, local name)]`` of the ORCA05 source files."""
    return [(f"{ZENODO_RECORD}/{n}/content", loc) for loc, (n, _) in ORCA05_SOURCES.items()]


def make_orca05(out: Path, drop_sources: bool = False) -> bool:
    """Download the ORCA05 sources and write the crops + the ``nemo_orca05/`` sibling files.

    Crops: ``arctic`` (rows 440:511), ``southern`` (97:152), both full width, and an
    ``equatorial`` box (-160..-120 deg, |lat| <= 10) of the global 511 x 722 grid; first
    three monthly records (Jan-Mar 2011); the equatorial box also split in two files.

    Parameters
    ----------
    out : pathlib.Path
        Output directory (``KELP_TEST_NEMO_DATA``).
    drop_sources : bool
        Remove the downloaded 3.8 GB sources afterwards.

    Returns
    -------
    bool
        False when a source could not be downloaded.
    """
    src = out / "_sources" / "orca05"
    ok = all(_download(f"{ZENODO_RECORD}/{n}/content", src / loc, sz) for loc, (n, sz) in ORCA05_SOURCES.items())
    if not ok:
        print("orca05: sources incomplete, nothing written", file=sys.stderr)
        return False
    sib = out / "nemo_orca05"
    sib.mkdir(parents=True, exist_ok=True)
    if not (sib / "mesh_mask_3d.nc").exists():
        shutil.copyfile(src / "mesh_mask.nc", sib / "mesh_mask_3d.nc")
    m = xr.open_dataset(sib / "mesh_mask_3d.nc")
    if not (sib / "mesh_mask.nc").exists():
        _write(m[[k for k in MESH_MASK_2D if k in m.variables]].isel(z=[0]).load(), sib / "mesh_mask.nc")
    # first record of the full 3-D files: the Lagrangian benchmark window
    for g in "TUVW":
        d = xr.open_dataset(src / f"2011_grid_{g}.nc", decode_times=False)
        d = d.isel(time_counter=[0])
        if g == "W":                                   # drop the (unused) vertical-mixing diagnostics
            d = d.drop_vars([k for k in ("votkeavt", "votkeevd", "votkeavm", "votkeevm", "voddmavs", "soleahtw",
                                         "soleaeiw") if k in d])
        _write(d.load(), sib / f"ORCA05_1step3d_grid_{g}.nc")
    lat, lon = m.gphit.values[0], m.glamt.values[0]
    r = 260                                  # a row near the equator (5.5 N)
    # -160..-120 deg on that row (i = 255..334); i > 214 only guards against a match west of it
    ii = np.where((lon[r] >= -160) & (lon[r] <= -120) & (np.arange(722) > 214))[0]
    i0, i1 = int(ii.min()), int(ii.max()) + 1
    c = (i0 + i1) // 2
    jj = np.where(np.abs(lat[:, c]) <= 10)[0]
    crops = {"arctic": (440, 511, 0, 722), "southern": (97, 152, 0, 722),
             "equatorial": (int(jj.min()), int(jj.max()) + 1, i0, i1)}
    print("orca05 crops (j0, j1, i0, i1):", crops)
    for name, (j0, j1, i0, i1) in crops.items():
        at = dict(
            crop_source="ORCA05 FOCI1.14-TEST.2020.04.1, Zenodo 10.5281/zenodo.3755721 (mesh_mask_3d.nc + 2011_grid_{T,U,V,W}.nc records 0-2 = Jan-Mar 2011), "
                        "full grid 511x722",
            crop_j0=j0, crop_j1=j1, crop_i0=i0, crop_i1=i1, crop_name=name,
            crop_topology_note="ORCA05 F-pivot north fold, E-W periodic with halo columns 0/721 (NEMO jperio=6); "
                               "topology attrs absent in source, added by the crop script",
            NFtype="F", Iperio=1, Jperio=0, NFold=1)
        me = _sel(m, j0, j1, i0, i1)[[k for k in METRICS if k in m]]
        _write(me, out / f"orca05_{name}_mesh.nc", at)
        for g in "TUVW":
            f = None
            targets = [(out / f"orca05_{name}_grid_{g}.nc", slice(0, 3))]
            if name == "equatorial":
                sp = out / "orca05_equatorial_split"
                targets += [(sp / f"orca05_1m_20110101_20110228_grid_{g}.nc", slice(0, 2)),
                            (sp / f"orca05_1m_20110301_20110331_grid_{g}.nc", slice(2, 3))]
            if all(t.exists() for t, _ in targets):
                for t, _ in targets:
                    print(f"  skip (exists) {t.name}")
                continue
            d = xr.open_dataset(src / f"2011_grid_{g}.nc", decode_times=False)
            f = _sel(d, j0, j1, i0, i1).isel(time_counter=slice(0, 3)).load()
            for t, sl in targets:
                _write(f.isel(time_counter=sl), t, at)
    if drop_sources:
        shutil.rmtree(src)
    return True


# ----------------------------------------------------------------------------------
# eORCA025
# ----------------------------------------------------------------------------------
def _op(path: str) -> xr.Dataset:
    return xr.open_zarr(EORCA025_URL + path, consolidated=True, zarr_format=3)


def make_eorca025(out: Path, nj: int = 70, ns: int = 28, only: list[str] | None = None) -> bool:
    """Crop the NOC eORCA025 store (arctic / southern / equatorial; first 3 monthly records).

    Parameters
    ----------
    out : pathlib.Path
        Output directory.
    nj, ns : int
        Number of rows of the arctic crop (counted down from row 1206) and of the
        southern one (counted down from the northern edge of the -60..-45 deg band).
    only : list of str, optional
        Crop names to make (default all three).

    Returns
    -------
    bool
        False when the store could not be opened.
    """
    try:
        dc = _op("domain_cfg")
    except Exception as exc:                                  # noqa: BLE001
        print(f"eorca025: cannot open {EORCA025_URL}domain_cfg ({exc!r}).\n  Needs zarr>=3, fsspec and aiohttp and "
              f"network access to the NOC S3 endpoint; no manual download alternative.", file=sys.stderr)
        return False
    gt, gl = dc.gphit.values, dc.glamt.values
    sj = np.where((np.median(gt, axis=1) >= -60) & (np.median(gt, axis=1) <= -45))[0]
    r = 680                                  # a row near the equator
    ii = np.where((gl[r] >= -160) & (gl[r] <= -120))[0]
    i0, i1 = int(ii.min()), int(ii.max()) + 1
    c = (i0 + i1) // 2
    jj = np.where(np.abs(gt[:, c]) <= 10)[0]
    # 1206 = the number of rows: the arctic crop ends at the north fold
    crops = {"arctic": (1206 - nj, 1206, 0, 1440), "southern": (int(sj.max()) + 1 - ns, int(sj.max()) + 1, 0, 1440),
             "equatorial": (int(jj.min()), int(jj.max()) + 1, i0, i1)}
    print("eorca025 crops (j0, j1, i0, i1):", crops, flush=True)
    kinds = [("U1m_4d", ["uo", "e3u"]), ("V1m_4d", ["vo", "e3v"]), ("W1m_4d", ["wo", "e3w"]),
             ("T1m_4d", ["thetao_con", "so_abs", "e3t"])]
    for name in only or list(crops):
        j0, j1, i0, i1 = crops[name]
        at = dict(
            crop_source="NOC eORCA025 JRA55 v1 (npd-eorca025-jra55v1: domain_cfg, U1m_4d, V1m_4d, W1m_4d, T1m_4d), "
                        "full grid 1206x1440, first 3 monthly records time_counter=0..2 (1976-01..03)",
            crop_j0=j0, crop_j1=j1, crop_i0=i0, crop_i1=i1, crop_name=name,
            source_url=EORCA025_URL, source_licence_note=EORCA025_LICENCE)
        target = out / f"eorca025_{name}_domain_cfg.nc"
        if not target.exists():
            _write(_sel(dc, j0, j1, i0, i1)[[k for k in METRICS if k in dc]].load(), target, at)
        for p, vs in kinds:
            files = [(out / f"eorca025_{name}_{p[0]}.nc", slice(0, 3))]
            if name == "equatorial":
                sp = out / "eorca025_equatorial_split"
                files += [(sp / f"eorca025_1m_19760101_19760229_grid_{p[0]}.nc", slice(0, 2)),
                          (sp / f"eorca025_1m_19760301_19760331_grid_{p[0]}.nc", slice(2, 3))]
            if all(t.exists() for t, _ in files):
                for t, _ in files:
                    print(f"  skip (exists) {t.name}")
                continue
            s = _op(p)[vs + ["nav_lat", "nav_lon"]].isel(time_counter=slice(0, 3), y=slice(j0, j1),
                                                         x=slice(i0, i1)).load()
            for t, sl in files:
                _write(s.isel(time_counter=sl), t, at)
    return True


# ----------------------------------------------------------------------------------
# ORCA2 (IPSL-CM5A-LR, CMIP5)
# ----------------------------------------------------------------------------------
def make_orca2(out: Path) -> bool:
    """Fetch the IPSL-CM5A-LR ORCA2 records (``*_t0.nc``) and fixed fields into ``<out>/esgf_orca2/data``.

    Parameters
    ----------
    out : pathlib.Path
        Output directory (``KELP_TEST_NEMO_DATA``).

    Returns
    -------
    bool
        False when a record or a fixed field could not be fetched.
    """
    d = out / "esgf_orca2" / "data"
    d.mkdir(parents=True, exist_ok=True)
    ok = True
    for v in ORCA2_VARS:
        t = d / f"{v}_t0.nc"
        if t.exists():
            print(f"  skip (exists) {t.name}")
            continue
        url = f"{ORCA2_OPENDAP}/{v}/{v}_Omon_IPSL-CM5A-LR_historical_r1i1p1_200001-200512.nc"
        try:
            with xr.open_dataset(url, decode_times=False) as ds:
                _write(ds.isel(time=[0]).load(), t)
        except Exception as exc:                              # noqa: BLE001
            print(f"  FAILED {v} ({str(exc)[:120]}).\n  Manual: fetch time index 0 of {url}\n"
                  f"  (e.g. ncks -d time,0 <file> {t}) and save it as {t}", file=sys.stderr)
            ok = False
    for v in ORCA2_FX:
        name = f"{v}_fx_IPSL-CM5A-LR_1pctCO2_r0i0p0.nc"
        ok = _download(f"{ORCA2_FX_URL}/{v}/{name}", d / name) and ok
    return ok


# ----------------------------------------------------------------------------------
# verification
# ----------------------------------------------------------------------------------
def verify(out: Path) -> int:
    """Check the written crops (shapes, mask vs NaN, halo and north-fold identities).

    Prints one line per crop.  Files that are absent are reported and counted as
    failures.

    Parameters
    ----------
    out : pathlib.Path
        Directory written by this script (``KELP_TEST_NEMO_DATA``).

    Returns
    -------
    int
        Number of failed checks.
    """
    wl = lambda a: (a + 180) % 360 - 180               # noqa: E731
    bad = 0

    def chk(label, val, tol=1e-6):
        nonlocal bad
        ok = bool(val <= tol)
        bad += not ok
        print(f"   {'ok  ' if ok else 'FAIL'} {label} = {val:.3g}")

    for name in ("arctic", "southern", "equatorial"):
        try:
            m = xr.open_dataset(out / f"eorca025_{name}_domain_cfg.nc")
            U, V, T = (xr.open_dataset(out / f"eorca025_{name}_{k}.nc") for k in "UVT")
        except FileNotFoundError as exc:
            print(f"E025 {name}: missing {exc.filename}")
            bad += 1
            continue
        g, l, tm = (m[k].values.squeeze() for k in ("gphit", "glamt", "tmask"))
        print(f"E025 {name}", m.tmask.shape, U.uo.shape, "lat", round(g.min(), 1), round(g.max(), 1), "lon",
              round(l.min(), 1), round(l.max(), 1), "times", [str(x)[:10] for x in U.time_counter.values])
        chk("mask-vs-NaN mismatch", ((~np.isnan(T.thetao_con.values[0])) != (tm > 0)).sum())
        chk("NFtype==T and Iperio==1 (0 = ok)", float(m.attrs.get("NFtype") != "T" or int(m.attrs.get("Iperio", 0)) != 1))
        if name == "arctic":
            ny, nx = g.shape
            I = np.arange(nx)
            t = T.thetao_con.values[0, 0]
            v, uo = V.vo.values[0, 0], U.uo.values[0, 0]
            chk("T fold", np.nanmax(np.abs(t[ny - 1] - t[ny - 1, -I % nx])))
            chk("V fold sign (v[ny-1,i] + v[ny-2,-i])", np.nanmax(np.abs(v[ny - 1] + v[ny - 2, -I % nx])))
            chk("U fold sign (u[ny-1,i] + u[ny-1,-1-i])", np.nanmax(np.abs(uo[ny - 1] + uo[ny - 1, (-1 - I) % nx])), 1e30)
            gf, lf = m.gphif.values.squeeze(), m.glamf.values.squeeze()
            chk("F geometry fold", max(np.abs(wl(lf[ny - 1] - lf[ny - 2, (-1 - I) % nx])).max(),
                                       np.abs(gf[ny - 1] - gf[ny - 2, (-1 - I) % nx]).max()))
            chk("e1v fold", np.abs(m.e1v.values.squeeze()[ny - 1] - m.e1v.values.squeeze()[ny - 2, -I % nx]).max())
        if name == "southern":
            chk("E-W periodic dlon step wrap - typical step", abs(np.abs(wl(l[:, 0] - l[:, -1])).max()
                                                                   - np.abs(wl(l[:, 1] - l[:, 0])).max()), 1e-6)
    for name in ("arctic", "southern", "equatorial"):
        try:
            m = xr.open_dataset(out / f"orca05_{name}_mesh.nc")
            T, V, W = (xr.open_dataset(out / f"orca05_{name}_grid_{k}.nc", decode_times=False) for k in "TVW")
        except FileNotFoundError as exc:
            print(f"O05 {name}: missing {exc.filename}")
            bad += 1
            continue
        g, l = m.gphit.values[0], m.glamt.values[0]
        print(f"O05 {name}", m.tmask.shape, T.votemper.shape, W.vovecrtz.shape, "lat", round(g.min(), 1),
              round(g.max(), 1), "lon", round(l.min(), 1), round(l.max(), 1), "nrec", T.sizes["time_counter"])
        chk("records == 3", abs(T.sizes["time_counter"] - 3))
        if name in ("arctic", "southern"):
            t = T.votemper.values[0, 0]
            chk("halo T[:,0]==T[:,-2]", np.nanmax(np.abs(t[:, 0] - t[:, -2])))
            chk("halo T[:,-1]==T[:,1]", np.nanmax(np.abs(t[:, -1] - t[:, 1])))
        if name == "arctic":
            ny, nx = g.shape
            I = np.arange(1, nx - 1)
            t = T.votemper.values[0, 0]
            chk("F-pivot T[ny-1,i]==T[ny-2,nx-1-i]", np.nanmax(np.abs(t[ny - 1, I] - t[ny - 2, nx - 1 - I])))
            chk("F-pivot geometry", max(np.abs(g[ny - 1, I] - g[ny - 2, nx - 1 - I]).max(),
                                        np.abs(wl(l[ny - 1, I] - l[ny - 2, nx - 1 - I])).max()))
    near = [out, out.parent, out.parent.parent]        # the tests look in these three places
    has = lambda *parts: any(b.joinpath(*parts).exists() for b in near)   # noqa: E731
    for f in ("thetao_t0.nc", "uo_t0.nc", "vo_t0.nc", "areacello_fx_IPSL-CM5A-LR_1pctCO2_r0i0p0.nc"):
        if not has("esgf_orca2", "data", f):
            print(f"ORCA2: missing esgf_orca2/data/{f}")
            bad += 1
    for f in ("mesh_mask.nc", "mesh_mask_3d.nc", "ORCA05_1step3d_grid_T.nc"):
        if not has("nemo_orca05", f):
            print(f"ORCA05: missing nemo_orca05/{f}")
            bad += 1
    print("verify:", "all checks passed" if not bad else f"{bad} check(s) FAILED")
    return bad


# ----------------------------------------------------------------------------------
def main(argv=None) -> int:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("source", choices=["orca05", "eorca025", "orca2", "all", "verify"], help="what to build")
    ap.add_argument("--out", type=Path, default=Path(".tmp/data/nemo_crops"),
                    help="output directory = KELP_TEST_NEMO_DATA (default ./.tmp/data/nemo_crops)")
    ap.add_argument("--verify", action="store_true", help="run the consistency checks afterwards")
    ap.add_argument("--dry-run", action="store_true", help="only list the source URLs")
    ap.add_argument("--drop-sources", action="store_true", help="delete the downloaded ORCA05 sources afterwards")
    ap.add_argument("--eorca025-rows", type=int, nargs=2, default=[70, 28], metavar=("ARCTIC", "SOUTHERN"),
                    help="row counts of the eORCA025 arctic / southern crops (default 70 28)")
    a = ap.parse_args(argv)
    out = a.out.resolve()
    todo = ["orca05", "eorca025", "orca2"] if a.source == "all" else [a.source]
    if a.dry_run:
        if "orca05" in todo:
            for (url, loc), (_, sz) in zip(orca05_urls(), ORCA05_SOURCES.values()):
                print(f"{url}  ({sz / 1e6:.0f} MB) -> {out}/_sources/orca05/{loc}   [{ORCA05_LICENCE}]")
        if "eorca025" in todo:
            print(f"{EORCA025_URL}{{domain_cfg,T1m_4d,U1m_4d,V1m_4d,W1m_4d}} (zarr v3, cropped remotely)   "
                  f"[{EORCA025_LICENCE}]")
        if "orca2" in todo:
            for v in ORCA2_VARS:
                print(f"{ORCA2_OPENDAP}/{v}/{v}_Omon_IPSL-CM5A-LR_historical_r1i1p1_200001-200512.nc (OPeNDAP, time 0)")
            for v in ORCA2_FX:
                print(f"{ORCA2_FX_URL}/{v}/{v}_fx_IPSL-CM5A-LR_1pctCO2_r0i0p0.nc")
            print(f"   [{ORCA2_LICENCE}]")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    ok = True
    if "orca05" in todo:
        ok = make_orca05(out, a.drop_sources) and ok
    if "eorca025" in todo:
        ok = make_eorca025(out, *a.eorca025_rows) and ok
    if "orca2" in todo:
        ok = make_orca2(out) and ok
    bad = verify(out) if (a.verify or a.source == "verify") else 0
    print(f"\nexport KELP_TEST_NEMO_DATA={out}")
    return 1 if (bad or not ok) else 0


if __name__ == "__main__":
    sys.exit(main())
