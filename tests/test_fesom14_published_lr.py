"""FESOM 1.4 reader on published output: CMIP6 AWI-CM-1-1-LR, hist-1950.

The data are the monthly means of the CMIP6 experiment ``hist-1950`` of
AWI-CM-1-1-LR (member ``r1i1p1f2``, licence CC BY-SA 4.0), in the directory
layout of the CMIP6 archive: ``Omon/<var>/gn/<version>/*.nc`` for ``uo``,
``vo``, ``wo``, ``thetao``, ``so``, ``zos`` (one file per variable and decade,
each with its own epoch) and ``Ofx/deptho/gn/<version>/*.nc``.  Set
``KELP_TEST_FESOM14_DATA`` to the member directory (the one holding ``Omon``
and ``Ofx``); without it the module is skipped.  The tests only read the
files; everything they write goes to pytest's temporary directory.

What is pinned, and how each pin can fail:

- **Mesh recovery** (``test_lr_mesh_recovery``): 126,859 nodes and 244,660
  triangles recovered from the cell polygons, checked here independently of
  the recovery (edges, nodes, duplicates, every triangle centroid a vertex of
  its three nodes' polygons within 1 cm, the Euler characteristic), the
  level table, ``deptho`` equal to the depth of the last wet level EXACTLY at
  every node, and geographic node coordinates (land and ocean points fall
  where they must; the mesh reaches the North Pole region).
- **The files** (``test_lr_levelwise_files``, ``test_lr_zos_complete``): every
  3-D file is ``(time, depth, ncells)`` on the same 46 levels with the same
  wet mask; ``zos`` has a value at every node in every record.
- **The absolute time axis** (``test_lr_time_axis_absolute``): every resolved
  instant equals the centre of its ``time_bnds`` decoded here with
  ``netCDF4``/``cftime`` from each file's own epoch; a run from 1951-01-16
  12:00 with dt = 1 day releases at that instant and records at that instant
  + k days.  The velocity each particle sees in its first step is computed
  here by hand from the files: probes sit exactly on mesh nodes (barycentric
  weight 1 at the node) and a quarter of a layer below a level, so the
  expected value is the vertical lerp between the two levels of the records
  that bracket the start — record 0 for the run started on it, records 1 and
  2 (time weight 14/29.5) for the 3-month run started on 1951-03-01 00:00.
  dt (1 day, 6 h) is far below the record spacing and the second start lies
  between records, so a shift of the axis by any amount moves the time
  weight and the velocities.
- **3-D runs** (``test_lr_3d_run_three_months``, ``test_lr_pchip_run``): 2,000
  particles, RK2 barycentric, 90 days from 1951-03-01, lerp and PCHIP in
  time, sampling ``thetao``: no NaN, every particle ALIVE or BEACHED, none
  below the bed of its nearest node or above the surface.  The PCHIP run's
  first-step velocities at the probes equal PCHIP in time at each level
  (``scipy.interpolate.PchipInterpolator`` on records 0-3), then the lerp in
  depth — and differ from the reversed order (lerp in depth, then PCHIP),
  the probes being chosen where the two orders differ.
- **w sign** (``test_lr_w_sign_equator``): ``wo`` is upward positive in the
  files (equatorial Pacific upwelling at 50 m); the tracker's depth rate is
  its negative, and particles there rise.
- **Level-linear vs layer-constant** (``test_lr_level_linear_vs_half_sum``):
  the same run with u/v averaged into layers (half sums of neighbouring
  levels, read as layer-constant data) separates from the level-linear run.
- **The two file forms** (``test_lr_form_equivalence``): the published fields
  and the recovered mesh written here as an ASCII FESOM 1.4 mesh
  (``nod3d.out`` level-major, ``aux3d.out`` with -999 for dry slots) and as
  raw-writer files over the 3-D nodes (``seconds since`` the start of each
  job segment, stamps at the window ends, ``output_schedule``) give
  bit-identical trajectories.

Measured values (the pins below hold a margin) are quoted in the docstrings
with the commit they were measured on.
"""
from __future__ import annotations

import contextlib
import glob
import io
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest

_DATA = os.environ.get("KELP_TEST_FESOM14_DATA", "")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not _DATA,
        reason="KELP_TEST_FESOM14_DATA is not set (the member directory of "
               "the published CMIP6 AWI-CM-1-1-LR hist-1950 output, holding "
               "Omon/ and Ofx/)"),
]

netCDF4 = pytest.importorskip("netCDF4")
cftime = pytest.importorskip("cftime")
scipy_interpolate = pytest.importorskip("scipy.interpolate")
scipy_stats = pytest.importorskip("scipy.stats")

from kelp import tracking  # noqa: E402
from kelp.core.config import (AppConfig, BeachingConfig, DataConfig,  # noqa: E402
                              MeshConfig, ModeConfig, NumericsConfig,
                              TimeConfig)
from kelp.core.io import VelocityField  # noqa: E402
from kelp.core.mesh import (Mesh, find_containing_elements,  # noqa: E402
                            find_nearest_nodes)
from kelp.core.particles import ParticleStatus  # noqa: E402
from kelp.readers.fesom14 import Fesom14Reader  # noqa: E402
from kelp.writers import load_trajectories  # noqa: E402

from tests._bitwise import assert_bitwise  # noqa: E402

#: the variables on the levels, and the 2-D one
VARS_3D = ("uo", "vo", "wo", "thetao", "so")
VAR_2D = "zos"
#: decades used (each variable has one file per decade)
N_DECADES = 2

N_NODES = 126_859
N_TRIANGLES = 244_660
#: the 46 levels of the published files (m, positive down)
LEVELS = np.array(
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 115, 135, 160, 190, 230,
     280, 340, 410, 490, 580, 680, 790, 910, 1040, 1180, 1330, 1500, 1700,
     1920, 2150, 2400, 2650, 2900, 3150, 3400, 3650, 3900, 4150, 4400, 4650,
     4900, 5150, 5400, 5650, 5900], dtype=np.float64)

CAL = "proleptic_gregorian"
DAY = 86400.0
#: the first resolved instant: the centre of January 1951 (bounds 0..31 d)
T0 = cftime.datetime(1951, 1, 16, 12, calendar=CAL)
#: the start of the 3-month runs: between record 1 (1951-02-15 00:00) and
#: record 2 (1951-03-16 12:00), so PCHIP has records 0..3 from the first step
T_RUN = cftime.datetime(1951, 3, 1, 0, calendar=CAL)

#: records read here by hand (0..5: January..June 1951)
N_REC = 6

#: the 3-month runs: dt (a record after every step), days, particles
DT_RUN = 6 * 3600.0
STEPS_PER_DAY = 4
N_DAYS = 90
N_PARTICLES = 2000
#: the form-equivalence runs: days
N_DAYS_FORMS = 20
#: the half-sum run: days
N_DAYS_HALF = 30

#: tolerance of a first-step velocity against its hand value (m/s); the
#: weights at a node are exact up to rounding (measured: at most 1.1e-16)
VEL_TOL = 1e-12
#: a PCHIP probe needs time-first and depth-first values this far apart
#: (m/s; the 24 probes chosen: at least 2.8e-3)
ORDER_GAP = 1e-4


# ---------------------------------------------------------------------------
# The published files
# ---------------------------------------------------------------------------

def _files(var: str, table: str = "Omon") -> List[Path]:
    """The files of one variable (latest version directory), in name order."""
    versions = sorted(glob.glob(os.path.join(_DATA, table, var, "gn", "*")))
    if not versions:
        raise FileNotFoundError(
            f"KELP_TEST_FESOM14_DATA holds no {table}/{var}/gn/<version> "
            "directory; it must be the member directory of the published "
            "AWI-CM-1-1-LR hist-1950 output")
    return sorted(Path(f) for f in glob.glob(os.path.join(versions[-1], "*.nc")))


def _quiet(fn: Callable, *args, **kw):
    """Call ``fn`` with its standard output captured."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kw)


def _bounds_centres(path: Path) -> list:
    """The centre of every record's ``time_bnds``, decoded here from the
    file's own units and calendar."""
    with netCDF4.Dataset(str(path)) as nc:
        tv = nc["time"]
        bname = tv.getncattr("bounds")
        b = np.asarray(nc[bname][:], dtype=np.float64)
        units, calendar = tv.getncattr("units"), tv.getncattr("calendar")
    lo = cftime.num2date(b[:, 0], units, calendar,
                         only_use_cftime_datetimes=True)
    hi = cftime.num2date(b[:, 1], units, calendar,
                         only_use_cftime_datetimes=True)
    return [a + (z - a) / 2 for a, z in zip(lo, hi)]


def _read_records(path: Path, var: str, n: int) -> np.ndarray:
    """Records ``0..n-1`` of ``var`` as stored (float32, fill as NaN)."""
    with netCDF4.Dataset(str(path)) as nc:
        v = nc[var]
        v.set_auto_mask(False)
        a = np.asarray(v[:n])
    a = a.astype(np.float32)
    a[~(np.abs(a) < 1e29)] = np.nan
    return a


@dataclass
class _LR:
    """The published LR files, the recovered mesh and records 0..5."""

    farm: Path                 # symlinks to the files the runs read
    mesh: Mesh
    levels: np.ndarray         # (K,)
    lon: np.ndarray            # geographic, [-180, 180)
    lat: np.ndarray
    bed: np.ndarray            # (N,) depth of the last wet level
    wet: np.ndarray            # (K, N)
    interior: np.ndarray       # (N,) bool: on no boundary edge
    centres: list              # resolved instants of records 0..N_REC-1
    rec: Dict[str, np.ndarray]  # uo/vo/wo/thetao: (N_REC, K, N) float32
    deptho: np.ndarray         # (N,) as published (float32)


def _boundary_nodes(tri: np.ndarray, n_nodes: int) -> np.ndarray:
    """Nodes on an edge used by one triangle only."""
    e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]],
                                tri[:, [2, 0]]]), axis=1).astype(np.int64)
    key = e[:, 0] * n_nodes + e[:, 1]
    uniq, count = np.unique(key, return_counts=True)
    edge = uniq[count == 1]
    out = np.zeros(n_nodes, dtype=bool)
    out[edge // n_nodes] = True
    out[edge % n_nodes] = True
    return out


@pytest.fixture(scope="module")
def lr(tmp_path_factory) -> _LR:
    """The farm of published files, the mesh recovered by the reader and
    records 0..5 of u, v, w and T read here with ``netCDF4``."""
    farm = tmp_path_factory.mktemp("lr_farm")
    for var in VARS_3D + (VAR_2D,):
        for f in _files(var)[:N_DECADES]:
            (farm / f.name).symlink_to(f)
    deptho = _files("deptho", "Ofx")[0]
    (farm / deptho.name).symlink_to(deptho)

    mesh = _quiet(Fesom14Reader.load_mesh,
                  MeshConfig(type="fesom14", meshpath=str(farm)))
    first = {v: _files(v)[0] for v in VARS_3D}
    rec = {v: _read_records(first[v], v, N_REC)
           for v in ("uo", "vo", "wo", "thetao")}
    with netCDF4.Dataset(str(deptho)) as nc:
        nc.set_auto_mask(False)
        d = np.asarray(nc["deptho"][:])
    wet = np.asarray(mesh.metadata["fesom14_wet"])
    return _LR(
        farm=farm, mesh=mesh, levels=np.asarray(mesh.metadata["fesom14_levels"]),
        lon=np.asarray(mesh.metadata["fesom14_lon"]),
        lat=np.asarray(mesh.metadata["fesom14_lat"]),
        bed=np.asarray(mesh.depth), wet=wet,
        interior=~_boundary_nodes(np.asarray(mesh.elem_nodes), mesh.n_nodes),
        centres=_bounds_centres(first["uo"])[:N_REC], rec=rec, deptho=d)


# ---------------------------------------------------------------------------
# Hand values
# ---------------------------------------------------------------------------

def _secs(t: cftime.datetime, origin: cftime.datetime = T0) -> float:
    """Seconds from ``origin`` to the cftime instant ``t``."""
    return (t - origin).total_seconds()


def _bracket(levels: np.ndarray, z: np.ndarray
             ) -> Tuple[np.ndarray, np.ndarray]:
    """Level ``k`` and weight ``a`` of depth ``z`` between levels k, k+1."""
    k = np.searchsorted(levels, z, side="right") - 1
    a = (z - levels[k]) / (levels[k + 1] - levels[k])
    return k, a


def _at_depth(col: np.ndarray, nodes: np.ndarray, k: np.ndarray,
              a: np.ndarray) -> np.ndarray:
    """``col`` (K, N) lerped in depth at the probes."""
    f0 = col[k, nodes].astype(np.float64)
    f1 = col[k + 1, nodes].astype(np.float64)
    return (1.0 - a) * f0 + a * f1


def _time_weight(lr: _LR, t) -> Tuple[int, float]:
    """Record ``i`` and weight ``w`` of the instant ``t`` between the
    resolved instants of records i and i+1."""
    secs = np.array([_secs(c) for c in lr.centres])
    q = _secs(t)
    i = int(np.searchsorted(secs, q, side="right") - 1)
    return i, (q - secs[i]) / (secs[i + 1] - secs[i])


def _lerp_hand(lr: _LR, var: str, nodes, z, t) -> np.ndarray:
    """``var`` at the probes and the instant ``t``: lerp in time between the
    bracketing records, lerp in depth (in either order: both are linear)."""
    i, w = _time_weight(lr, t)
    k, a = _bracket(lr.levels, z)
    return ((1.0 - w) * _at_depth(lr.rec[var][i], nodes, k, a)
            + w * _at_depth(lr.rec[var][i + 1], nodes, k, a))


def _level_hand(lr: _LR, var: str, nodes, k, t) -> np.ndarray:
    """``var`` at level ``k`` of the nodes and the instant ``t`` (lerp in
    time)."""
    i, w = _time_weight(lr, t)
    return ((1.0 - w) * lr.rec[var][i][k, nodes].astype(np.float64)
            + w * lr.rec[var][i + 1][k, nodes].astype(np.float64))


def _pchip(values: np.ndarray, secs: np.ndarray, q: float) -> np.ndarray:
    """Monotone cubic (scipy's PCHIP) through four values per probe, at
    ``q``, which lies between the second and the third instant."""
    f = scipy_interpolate.PchipInterpolator(secs, values, axis=0)
    return np.asarray(f(q), dtype=np.float64)


def _pchip_hand(lr: _LR, var: str, nodes, z, t, *, time_first: bool = True
                ) -> np.ndarray:
    """``var`` at the probes and the instant ``t`` (between records 1 and
    2): PCHIP in time over records 0..3, then lerp in depth (the reader's
    rule) — or with ``time_first=False`` the reversed order."""
    secs = np.array([_secs(c) for c in lr.centres[:4]])
    q = _secs(t)
    assert secs[1] < q < secs[2]
    k, a = _bracket(lr.levels, z)
    snaps = lr.rec[var][:4]
    if time_first:
        up = _pchip(np.stack([s[k, nodes] for s in snaps]).astype(np.float64),
                    secs, q)
        lo = _pchip(np.stack([s[k + 1, nodes] for s in snaps]).astype(np.float64),
                    secs, q)
        return (1.0 - a) * up + a * lo
    return _pchip(np.stack([_at_depth(s, nodes, k, a) for s in snaps]), secs, q)


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

@dataclass
class _Seeds:
    """Release points: the cloud, then the probes (each kind a slice)."""

    lon: np.ndarray
    lat: np.ndarray
    z: np.ndarray
    node: np.ndarray            # the node a probe sits on; -1 in the cloud
    kinds: Dict[str, slice]

    def write(self, path: Path) -> Path:
        """Write the seed file (``lon lat z`` per line) and return its path."""
        with open(path, "w") as f:
            for lo, la, z in zip(self.lon, self.lat, self.z):
                # repr of a Python float round-trips exactly
                f.write(f"{float(lo)!r} {float(la)!r} {float(z)!r}\n")
        return path


def _quarter_depth(levels: np.ndarray, k: np.ndarray) -> np.ndarray:
    """A quarter of the layer below level ``k``."""
    return levels[k] + 0.25 * (levels[k + 1] - levels[k])


def _node_probes(lr: _LR, cand: np.ndarray, layers: Sequence[int],
                 score: Optional[np.ndarray], n: int
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """``n`` probes on the candidate nodes, a quarter layer below the levels
    ``layers`` (cycled), ranked by ``score`` (descending) when given."""
    k = np.asarray(layers)[np.arange(cand.size) % len(layers)]
    if score is not None:
        order = np.argsort(-score, kind="stable")[:n]
        cand, k = cand[order], k[order]
    else:
        cand, k = cand[:n], k[:n]
    return cand, _quarter_depth(lr.levels, k)


@pytest.fixture(scope="module")
def seeds(lr: _LR) -> Dict[str, _Seeds]:
    """Seeds of the run started on record 0 ("axis") and of the 3-month
    runs ("months")."""
    rng = np.random.default_rng(1951)
    levels, bed = lr.levels, lr.bed
    deep = lr.interior & (bed >= 500.0) & (np.abs(lr.lat) <= 60.0)
    layers = (0, 2, 5, 11, 16)

    # time probes: nodes where u or v changes between records 0 and 1
    cand = rng.permutation(np.flatnonzero(deep))[:4000]
    kk = np.asarray(layers)[np.arange(cand.size) % len(layers)]
    du = (np.abs(lr.rec["uo"][1][kk, cand] - lr.rec["uo"][0][kk, cand])
          + np.abs(lr.rec["vo"][1][kk, cand] - lr.rec["vo"][0][kk, cand]))
    t_nodes, t_z = _node_probes(lr, cand[du > 1e-3], layers, None, 24)
    assert t_nodes.size == 24

    # equatorial Pacific upwelling, and the North Pacific subtropical gyre
    eq = np.flatnonzero(lr.interior & (np.abs(lr.lat) <= 1.5)
                        & (lr.lon >= -170.0) & (lr.lon <= -100.0)
                        & (bed >= 1000.0))
    eq = eq[np.argsort(lr.lon[eq])][::max(1, eq.size // 24)][:24]
    gy = np.flatnonzero(lr.interior & (lr.lat >= 20.0) & (lr.lat <= 30.0)
                        & ((lr.lon >= 160.0) | (lr.lon <= -150.0))
                        & (bed >= 3000.0))
    gy = gy[np.argsort(lr.lon[gy])][::max(1, gy.size // 24)][:24]
    z50 = float(levels[5])                       # 50 m, a level

    axis = _assemble_seeds(lr, [], {"time": (t_nodes, t_z),
                                    "equator": (eq, np.full(eq.size, z50)),
                                    "gyre": (gy, np.full(gy.size, z50))})

    # PCHIP probes: nodes where PCHIP-then-lerp and lerp-then-PCHIP differ
    cand = rng.permutation(np.flatnonzero(deep))[:6000]
    kk = np.asarray(layers)[np.arange(cand.size) % len(layers)]
    zz = _quarter_depth(levels, kk)
    gap = np.zeros(cand.size)
    for var in ("uo", "vo"):
        gap = np.maximum(gap, np.abs(
            _pchip_hand(lr, var, cand, zz, T_RUN)
            - _pchip_hand(lr, var, cand, zz, T_RUN, time_first=False)))
    p_nodes, p_z = _node_probes(lr, cand, layers, gap, 24)
    assert (np.sort(gap)[::-1][:24] >= ORDER_GAP).all()
    n_cloud = N_PARTICLES - p_nodes.size

    # the cloud: element centres of deep enough elements, random depths
    tri = np.asarray(lr.mesh.elem_nodes)
    shallow = bed[tri].min(axis=1)
    lat_c = np.asarray(lr.mesh.elem_center_lat)
    ok = np.flatnonzero((shallow >= 50.0) & (np.abs(lat_c) <= 80.0))
    elems = rng.choice(ok, n_cloud, replace=False)
    zc = rng.uniform(0.0, 1.0, n_cloud) ** 2 * np.minimum(
        0.95 * shallow[elems], 3000.0)
    cloud = (np.asarray(lr.mesh.elem_center_lon)[elems], lat_c[elems], zc)
    months = _assemble_seeds(lr, [cloud], {"probes": (p_nodes, p_z)})
    return {"axis": axis, "months": months}


def _assemble_seeds(lr: _LR, clouds: Sequence[Tuple[np.ndarray, ...]],
                    probes: Dict[str, Tuple[np.ndarray, np.ndarray]]
                    ) -> _Seeds:
    """The clouds ``(lon, lat, z)`` first, then each kind of probe
    ``(nodes, z)`` placed exactly on its nodes."""
    lon, lat, z, node, kinds = [], [], [], [], {}
    n = 0
    for clo, cla, cz in clouds:
        lon.append(clo); lat.append(cla); z.append(cz)
        node.append(np.full(clo.size, -1))
        n += clo.size
    for name, (nodes, zp) in probes.items():
        lon.append(lr.lon[nodes]); lat.append(lr.lat[nodes]); z.append(zp)
        node.append(nodes)
        kinds[name] = slice(n, n + nodes.size)
        n += nodes.size
    return _Seeds(lon=np.concatenate(lon), lat=np.concatenate(lat),
                  z=np.concatenate(z), node=np.concatenate(node), kinds=kinds)


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

@dataclass
class _Run:
    """One run's output: the arrays, the record dates, the global attributes."""

    tr: Dict[str, np.ndarray]
    dates: list
    release: list
    attrs: Dict[str, object]


def _iso(t: cftime.datetime) -> str:
    """``YYYY-MM-DDThh:mm``, the form of ``time.start_date``."""
    return t.strftime("%Y-%m-%dT%H:%M")


def _cfg(out: Path, seeds_file: Path, *, data: Path, meshpath: Path,
         start: cftime.datetime, n_steps: int, dt: float, output_freq: int,
         pchip: bool = False, sample: bool = True) -> AppConfig:
    """A 3-D RK2 barycentric run, lerp (or PCHIP) in time, sampling T (unless
    ``sample`` is false), beaching on failure (a particle that would leave
    the mesh stops there)."""
    num = dict(dt=dt, integrator="rk2", uv_interp="barycentric",
               w_interp="barycentric", sample_interp="barycentric",
               temporal_interp=True, output_freq=output_freq)
    if pchip:
        num["temporal_reconstruction"] = "pchip"
    num["beaching"] = BeachingConfig(enabled=True, criteria=["on_failure"])
    return AppConfig(
        mode=ModeConfig(Fdrift2D=False, Fdrift3D=True,
                        sample_vars=["thetao"] if sample else []),
        numerics=NumericsConfig(**num),
        mesh=MeshConfig(type="fesom14", meshpath=str(meshpath)),
        data=DataConfig(ncinput=str(data), inicloud=str(seeds_file),
                        output_path=str(out)),
        time=TimeConfig(start_date=_iso(start), n_steps=n_steps),
    )


def _run(cfg: AppConfig) -> _Run:
    """``run_tracking`` with its summary captured; the output read back,
    its dates decoded here from the file's own units and calendar."""
    _quiet(tracking.run_tracking, cfg)
    path = cfg.data.output_path
    with netCDF4.Dataset(path) as nc:
        tv, rv = nc["time"], nc["release_time"]
        dates = list(cftime.num2date(tv[:], tv.units, tv.calendar,
                                     only_use_cftime_datetimes=True))
        release = list(cftime.num2date(rv[:], rv.units, rv.calendar,
                                       only_use_cftime_datetimes=True))
        attrs = {k: nc.getncattr(k) for k in nc.ncattrs()}
    return _Run(tr=load_trajectories(path), dates=dates, release=release,
                attrs=attrs)


def _months_cfg(lr: _LR, seeds: Dict[str, _Seeds], d: Path, name: str, *,
                days: int, data: Optional[Path] = None,
                meshpath: Optional[Path] = None, pchip: bool = False,
                sample: bool = True) -> AppConfig:
    """A run of the "months" seeds from ``T_RUN``: dt = 6 h, a record after
    every step."""
    sf = seeds["months"].write(d / f"seeds_{name}.dat")
    return _cfg(d / f"{name}.nc", sf, data=data or lr.farm,
                meshpath=meshpath or lr.farm, start=T_RUN,
                n_steps=int(days * DAY / DT_RUN), dt=DT_RUN,
                output_freq=1, pchip=pchip, sample=sample)


@pytest.fixture(scope="module")
def axis_run(lr, seeds, tmp_path_factory) -> _Run:
    """Forward from the first instant, dt = 1 day, a record every step."""
    d = tmp_path_factory.mktemp("axis")
    sf = seeds["axis"].write(d / "seeds.dat")
    return _run(_cfg(d / "axis.nc", sf, data=lr.farm, meshpath=lr.farm,
                     start=T0, n_steps=10, dt=DAY, output_freq=1))


@pytest.fixture(scope="module")
def lerp_run(lr, seeds, tmp_path_factory) -> _Run:
    d = tmp_path_factory.mktemp("lerp")
    return _run(_months_cfg(lr, seeds, d, "lerp", days=N_DAYS))


@pytest.fixture(scope="module")
def pchip_run(lr, seeds, tmp_path_factory) -> _Run:
    """The lerp run's twin with PCHIP in time.  It samples no field: with a
    sampled field, the record-time sampling of a PCHIP run re-reads the two
    outer records of the stencil at every step (the run cache keeps only
    the bracketing pair there), which makes this run ~15x slower (measured
    on LR: 120 steps 127 s against 8 s); sampling under PCHIP is covered by
    the form-equivalence runs."""
    d = tmp_path_factory.mktemp("pchip")
    return _run(_months_cfg(lr, seeds, d, "pchip", days=N_DAYS, pchip=True,
                            sample=False))


# ---------------------------------------------------------------------------
# Checks shared by the runs
# ---------------------------------------------------------------------------

def _first_step_velocity(run: _Run, sl: slice) -> Dict[str, np.ndarray]:
    """The velocities of the first recorded step (its start position and
    time) of the particles in ``sl``; w positive down."""
    return {c: run.tr[f"vel_{c}"][sl, 0] for c in ("u", "v", "w")}


def _assert_hand(got: np.ndarray, want: np.ndarray, what: str) -> None:
    """Every probe within ``VEL_TOL`` of its hand value."""
    print(f"measured: {what}: largest difference from the hand value "
          f"{float(np.max(np.abs(got - want))):.3e} m/s")
    bad = ~(np.abs(got - want) <= VEL_TOL)          # NaN counts as off
    assert not bad.any(), (
        f"{what}: {int(bad.sum())}/{bad.size} probes off their hand value; "
        f"largest difference {float(np.nanmax(np.abs(got - want))):.3e} m/s, "
        f"first: got {got[bad][0]!r}, expected {want[bad][0]!r}")


def _assert_sane_run(lr: _LR, run: _Run) -> None:
    """No NaN; every particle ALIVE or BEACHED; none above the surface or
    below the bed of its nearest node (the tracker's end-of-step clamp)."""
    tr = run.tr
    st = tr["status"]
    alive = st == ParticleStatus.ALIVE
    ok = alive | (st == ParticleStatus.BEACHED)
    assert ok.all(), (
        f"{int((~ok).sum())} particle records neither ALIVE nor BEACHED "
        f"(statuses {np.unique(st[~ok]).tolist()})")
    for name in ("lon", "lat", "z"):
        assert np.isfinite(tr[name]).all(), f"NaN in {name}"
    for name in ("vel_u", "vel_v", "vel_w", "thetao"):
        if name in tr:
            assert np.isfinite(tr[name][alive]).all(), \
                f"NaN in {name} of an ALIVE particle"
    z = tr["z"].ravel()
    nn = find_nearest_nodes(lr.mesh, tr["lon"].ravel(), tr["lat"].ravel())
    assert (z >= 0.0).all(), f"above the surface: min z {z.min()}"
    below = z - lr.bed[nn]
    assert (below <= 1e-9).all(), (
        f"{int((below > 1e-9).sum())} particle records below the bed of "
        f"their nearest node (up to {below.max():.3g} m)")


def _rec(day: int) -> int:
    """The record of a 3-month run written ``day`` days after its start."""
    return day * STEPS_PER_DAY - 1


def _sep_km(a: _Run, b: _Run, rec: int) -> np.ndarray:
    """Great-circle distance (km) between two runs' particles at record
    ``rec``, for the particles ALIVE in both."""
    both = ((a.tr["status"][:, rec] == ParticleStatus.ALIVE)
            & (b.tr["status"][:, rec] == ParticleStatus.ALIVE))
    return _gc_km(a.tr["lon"][both, rec], a.tr["lat"][both, rec],
                  b.tr["lon"][both, rec], b.tr["lat"][both, rec])


def _disp_km(run: _Run, rec: int) -> np.ndarray:
    """Great-circle distance (km) from the release point at record ``rec``,
    for the particles ALIVE then."""
    alive = run.tr["status"][:, rec] == ParticleStatus.ALIVE
    return _gc_km(run.tr["release_lon"][alive], run.tr["release_lat"][alive],
                  run.tr["lon"][alive, rec], run.tr["lat"][alive, rec])


def _gc_km(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Great-circle distance (km, haversine, R = 6371 km)."""
    lo1, la1, lo2, la2 = (np.deg2rad(np.asarray(a, dtype=np.float64))
                          for a in (lon1, lat1, lon2, lat2))
    h = (np.sin((la2 - la1) / 2) ** 2
         + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.minimum(h, 1.0)))


# ---------------------------------------------------------------------------
# The mesh and the files
# ---------------------------------------------------------------------------

def test_lr_mesh_recovery(lr):
    """The mesh recovered from the cell polygons, checked independently.

    The counts (126,859 nodes, 244,660 triangles); the levels; every edge
    in one or two triangles, every node in a triangle, no duplicate
    triangle; every triangle's spherical centroid a vertex of the polygons
    of its three nodes within 1 cm; V - E + F = 2 - (boundary loops) as the
    recovery reports them; no pinch triangle on LR.  ``deptho`` equals the
    depth of the last wet level EXACTLY at every node (the reader allows
    0.01 m), the mask is a prefix of each column, every column has at least
    three wet levels.  Geographic coordinates: the nodes reach 89.9 N and
    every longitude, ocean points lie in a triangle and land points in none.
    """
    mesh = lr.mesh
    tri = np.asarray(mesh.elem_nodes, dtype=np.int64)
    assert mesh.n_nodes == N_NODES
    assert tri.shape == (N_TRIANGLES, 3)
    assert_bitwise(lr.levels, LEVELS, "levels")
    assert_bitwise(np.asarray(mesh.sigma), LEVELS, "mesh.sigma")
    rep = mesh.metadata["fesom14_polygon_report"]
    assert rep["n_pinch_triangles"] == 0
    assert rep["n_components"] == 1

    # edges, nodes, duplicates
    e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]],
                                tri[:, [2, 0]]]), axis=1)
    _, count = np.unique(e[:, 0] * N_NODES + e[:, 1], return_counts=True)
    assert set(np.unique(count).tolist()) <= {1, 2}
    assert np.array_equal(np.unique(tri), np.arange(N_NODES))
    st = np.sort(tri, axis=1)
    assert np.unique(st[:, 0] * N_NODES ** 2 + st[:, 1] * N_NODES + st[:, 2]).size \
        == N_TRIANGLES
    n_edges = count.size
    euler = N_NODES - n_edges + N_TRIANGLES
    assert euler == rep["euler_characteristic"] == 2 - rep["n_boundary_loops"]

    # centroids: a vertex of each of the three polygons, within 1 cm
    with netCDF4.Dataset(str(_files("uo")[0])) as nc:
        lon_b = np.asarray(nc["lon_bnds"][:], dtype=np.float64)
        lat_b = np.asarray(nc["lat_bnds"][:], dtype=np.float64)

    def xyz(lon, lat):
        lo, la = np.deg2rad(lon), np.deg2rad(lat)
        return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo),
                         np.sin(la)], axis=-1)

    p = xyz(lr.lon, lr.lat)[tri].sum(axis=1)
    c = p / np.linalg.norm(p, axis=1)[:, None]
    worst = 0.0
    for j in range(3):
        v = xyz(lon_b[tri[:, j]], lat_b[tri[:, j]])        # (E, nv, 3)
        d = np.linalg.norm(v - c[:, None, :], axis=2).min(axis=1)
        worst = max(worst, float(d.max()) * 6_367_500.0)
    assert worst <= 0.01, f"a centroid is {worst:.4f} m from its polygon vertex"

    # the mask and deptho
    wet = lr.wet
    ncol = wet.sum(axis=0)
    assert np.array_equal(wet, np.arange(LEVELS.size)[:, None] < ncol[None, :])
    assert ncol.min() >= 3
    assert_bitwise(lr.bed, LEVELS[ncol - 1], "mesh.depth")
    assert lr.deptho.shape == (N_NODES,)
    assert np.array_equal(lr.deptho.astype(np.float64), lr.bed), (
        "deptho differs from the depth of the last wet level at "
        f"{int((lr.deptho.astype(np.float64) != lr.bed).sum())} nodes")

    # geographic: the Arctic is there; ocean in triangles, land in none
    assert lr.lat.max() > 89.9
    assert np.unique(np.floor(lr.lon)).size == 360
    ocean = np.array([(-150.0, 0.0), (-30.0, 30.0), (80.0, -20.0),
                      (0.0, -60.0), (0.0, 88.0), (160.0, 40.0)])
    land = np.array([(20.0, 5.0), (90.0, 45.0), (-100.0, 45.0),
                     (-60.0, -5.0), (135.0, -25.0), (0.0, -85.0),
                     (-40.0, 75.0)])
    for pts, inside in ((ocean, True), (land, False)):
        nn = find_nearest_nodes(mesh, pts[:, 0], pts[:, 1])
        _, found = find_containing_elements(mesh, pts[:, 0], pts[:, 1], nn)
        assert (found == inside).all(), (pts, found)


def test_lr_levelwise_files(lr):
    """Every 3-D file is ``(time, depth, ncells)``: 46 levels, 126,859 nodes,
    the same level values; the same wet mask in u, v, w, T and S (first and
    last record of the first decade); fill below the bed, no gap in a
    column."""
    for var in VARS_3D:
        for f in _files(var):
            with netCDF4.Dataset(str(f)) as nc:
                assert nc[var].dimensions == ("time", "depth", "ncells"), f.name
                assert nc.dimensions["depth"].size == LEVELS.size
                assert nc.dimensions["ncells"].size == N_NODES
                assert_bitwise(np.asarray(nc["depth"][:], dtype=np.float64),
                               LEVELS, f"depth of {f.name}")
    for var in VARS_3D:
        f = _files(var)[0]
        with netCDF4.Dataset(str(f)) as nc:
            v = nc[var]
            v.set_auto_mask(False)
            for i in (0, nc.dimensions["time"].size - 1):
                a = np.asarray(v[i])
                assert np.array_equal(np.abs(a) < 1e29, lr.wet), (var, i)
                assert np.isfinite(a).all()


def test_lr_zos_complete():
    """``zos`` has a value at every node in every record of every file."""
    n = 0
    for f in _files(VAR_2D):
        with netCDF4.Dataset(str(f)) as nc:
            v = nc[VAR_2D]
            v.set_auto_mask(False)
            assert v.dimensions == ("time", "ncells")
            a = np.asarray(v[:])
        bad = ~(np.isfinite(a) & (np.abs(a) < 1e29))
        assert not bad.any(), (f.name, np.argwhere(bad)[:3].tolist())
        n += a.shape[0]
    assert n > 0


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------

def test_lr_time_axis_absolute(lr, seeds, request):
    """The resolved instants are the centres of the files' time bounds.

    In this order (the runs are started only after the reader's axis has
    passed, so a shifted axis fails on its own assertion):

    1. The reader's axis over two decades (two files per variable, epochs
       ``days since 1951-1-1`` and ``days since 1961-1-1``) equals the
       centres of ``time_bnds`` decoded here from each file's own units and
       calendar; record 0 is 1951-01-16 12:00, record 1 1951-02-15 00:00,
       record 120 1961-01-16 12:00.
    2. The 3-month lerp run starts on 1951-03-01 00:00, between records 1
       and 2: its first-step velocities at the probes (on nodes, a quarter
       layer below a level) equal the lerp between those records with weight
       14/29.5, lerped in depth — any shift of the axis seen by the engine
       changes that weight.
    3. A run from 1951-01-16 12:00 with dt = 1 day and a record every step
       writes 10 records, record k (1-based) at that instant + k days,
       releases every particle at that instant, and its first-step
       velocities at the time probes equal record 0 lerped in depth.
    """
    ds = _quiet(Fesom14Reader.open_dataset, str(lr.farm),
                mode_cfg=ModeConfig(Fdrift3D=True))
    try:
        got = list(ds["time"].values)
        assert ds.attrs["fesom14_time_source"] == "bounds"
    finally:
        ds.close()
    want = []
    for f in _files("uo")[:N_DECADES]:
        want += _bounds_centres(f)
    assert len(got) == len(want) == 120 * N_DECADES
    assert got == want
    assert got[0] == T0
    assert got[1] == cftime.datetime(1951, 2, 15, calendar=CAL)
    assert got[120] == cftime.datetime(1961, 1, 16, 12, calendar=CAL)
    assert lr.centres[0] == T0

    lerp_run = request.getfixturevalue("lerp_run")
    s = seeds["months"]
    sl = s.kinds["probes"]
    nodes, z = s.node[sl], s.z[sl]
    assert lerp_run.release[0] == T_RUN
    assert lerp_run.dates[0] == T_RUN + timedelta(seconds=DT_RUN)
    vel = _first_step_velocity(lerp_run, sl)
    for comp, var, sign in (("u", "uo", 1.0), ("v", "vo", 1.0),
                            ("w", "wo", -1.0)):
        _assert_hand(vel[comp], sign * _lerp_hand(lr, var, nodes, z, T_RUN),
                     f"3-month run, 1951-03-01, {comp}")

    run = request.getfixturevalue("axis_run")
    assert len(run.dates) == 10
    assert run.dates == [T0 + timedelta(days=k) for k in range(1, 11)]
    assert run.release == [T0] * len(run.release)
    assert run.attrs["time_averaging"] == "resolved_by_reader"
    s = seeds["axis"]
    sl = s.kinds["time"]
    nodes, z = s.node[sl], s.z[sl]
    vel = _first_step_velocity(run, sl)
    for comp, var, sign in (("u", "uo", 1.0), ("v", "vo", 1.0),
                            ("w", "wo", -1.0)):
        _assert_hand(vel[comp], sign * _lerp_hand(lr, var, nodes, z, T0),
                     f"axis run, record 0, {comp}")


def test_lr_w_sign_equator(lr, seeds, axis_run):
    """``wo`` is positive up: equatorial Pacific upwelling at 50 m, downward
    flow in the North Pacific subtropical gyre (record 0, read here).  The
    tracker's depth rate (positive down) is its negative at the probes, and
    the equatorial particles rise over the 10 days of the run."""
    s = seeds["axis"]
    tr = axis_run.tr
    for kind, sign in (("equator", 1.0), ("gyre", -1.0)):
        sl = s.kinds[kind]
        nodes = s.node[sl]
        assert nodes.size >= 12, kind
        wo = lr.rec["wo"][0][5, nodes].astype(np.float64)    # 50 m
        assert sign * np.median(wo) > 0.0, (kind, np.median(wo))
        _assert_hand(tr["vel_w"][sl, 0], -wo, f"{kind}: w at 50 m")
    sl = s.kinds["equator"]
    alive = tr["status"][sl, -1] == ParticleStatus.ALIVE
    rose = tr["z"][sl, -1][alive] < s.z[sl][alive]
    print(f"measured: equatorial probes risen after 10 days: {rose.mean():.3f}")
    assert rose.mean() >= 0.75, rose.mean()


# ---------------------------------------------------------------------------
# 3-D runs
# ---------------------------------------------------------------------------

def test_lr_3d_run_three_months(lr, seeds, lerp_run):
    """2,000 particles, 90 days from 1951-03-01, RK2 barycentric, lerp in
    time, beaching on failure, ``thetao`` sampled: no NaN, all ALIVE or
    BEACHED, none below the bed of its nearest node."""
    run = lerp_run
    assert run.tr["lon"].shape == (N_PARTICLES, N_DAYS * STEPS_PER_DAY)
    assert run.dates[-1] == T_RUN + timedelta(days=N_DAYS)
    _assert_sane_run(lr, run)
    alive = (run.tr["status"][:, -1] == ParticleStatus.ALIVE).mean()
    print(f"measured: ALIVE after {N_DAYS} days: {alive:.4f}")
    assert alive >= 0.9, alive


#: median separation (km) of the PCHIP run from the lerp run after 30 days:
#: measured 1.705 km (reader at d1d7eb5); pinned at half
PCHIP_SEP_KM_MIN = 0.85


def test_lr_pchip_run(lr, seeds, lerp_run, pchip_run):
    """The 3-month run with ``temporal_reconstruction: pchip``.

    The same checks as the lerp run.  First-step velocities at the PCHIP
    probes equal PCHIP in time at each of the two levels (records 0-3),
    then the lerp in depth, and differ from the reversed order by at least
    ``ORDER_GAP`` for u or v.  The median separation from the lerp run after
    30 days stays above ``PCHIP_SEP_KM_MIN`` (measured value in the
    constant's comment).
    """
    run = pchip_run
    _assert_sane_run(lr, run)
    s = seeds["months"]
    sl = s.kinds["probes"]
    nodes, z = s.node[sl], s.z[sl]
    vel = _first_step_velocity(run, sl)
    gap = np.zeros(nodes.size)
    for comp, var, sign in (("u", "uo", 1.0), ("v", "vo", 1.0),
                            ("w", "wo", -1.0)):
        want = sign * _pchip_hand(lr, var, nodes, z, T_RUN)
        _assert_hand(vel[comp], want, f"PCHIP run, {comp}")
        if comp != "w":
            gap = np.maximum(gap, np.abs(
                vel[comp] - sign * _pchip_hand(lr, var, nodes, z, T_RUN,
                                               time_first=False)))
    assert (gap >= ORDER_GAP).all(), np.sort(gap)[:3]

    sep = np.median(_sep_km(run, lerp_run, _rec(30)))
    print(f"measured: PCHIP vs lerp: median separation after 30 days "
          f"{sep:.4f} km; smallest depth-first gap {gap.min():.3e} m/s")
    assert sep >= PCHIP_SEP_KM_MIN, sep


# ---------------------------------------------------------------------------
# Level-linear vs layer-constant
# ---------------------------------------------------------------------------

def _half_sum_reads(monkeypatch) -> None:
    """Make the reader hand the engine u/v averaged into the layers between
    neighbouring levels — layer-constant data, the FESOM2 placement — with
    w unchanged on the levels."""
    level_read = Fesom14Reader.read_velocities

    def read(*args, **kw):
        vel = level_read(*args, **kw)
        if vel.uv_vertical != "level":
            return vel
        u = np.ascontiguousarray(0.5 * (vel.u[:-1] + vel.u[1:]))
        v = np.ascontiguousarray(0.5 * (vel.v[:-1] + vel.v[1:]))
        return VelocityField(u=u, v=v, eta=vel.eta, w=vel.w,
                             uv_layout="node", scalars=vel.scalars,
                             uv_vertical="layer")

    monkeypatch.setattr(Fesom14Reader, "read_velocities", staticmethod(read))


@pytest.fixture(scope="module")
def half_sum_run(lr, seeds, tmp_path_factory) -> _Run:
    d = tmp_path_factory.mktemp("half_sum")
    with pytest.MonkeyPatch.context() as mp:
        _half_sum_reads(mp)
        return _run(_months_cfg(lr, seeds, d, "half_sum", days=N_DAYS_HALF))


#: the half-sum run against the level-linear run after 30 days, measured with
#: the reader at d1d7eb5: median separation 1.654 km (pinned at half), 0.027
#: of the median displacement of 61.0 km (pinned below 0.1: small); rank
#: correlation of the separation with the release-point driver 0.771
#: (pinned at half); median separation in the strongest driver quartile
#: 8.93 km against 0.29 km in the weakest, a ratio of 30.9 (pinned at 10)
HALF_SUM_SEP_KM_MIN = 0.8
HALF_SUM_SEP_FRAC_MAX = 0.1
HALF_SUM_RANK_MIN = 0.385
HALF_SUM_QUARTILE_RATIO_MIN = 10.0


def _shear_driver(lr: _LR, run: _Run, idx: np.ndarray) -> np.ndarray:
    """What separates the two readings at a release point: the level-linear
    value minus the layer half sum is ``(a - 1/2) (f[k+1] - f[k])`` at depth
    weight ``a`` between levels k and k+1, so its size is ``|a - 1/2|``
    times the velocity difference of the two levels (m/s), here at the
    release's nearest node and the run's start."""
    lon = run.tr["release_lon"][idx]
    lat = run.tr["release_lat"][idx]
    z = run.tr["release_z"][idx]
    nn = find_nearest_nodes(lr.mesh, lon, lat).astype(np.int64)
    k = np.minimum(np.searchsorted(lr.levels, z, side="right") - 1,
                   lr.wet.sum(axis=0)[nn] - 2)
    a = (z - lr.levels[k]) / (lr.levels[k + 1] - lr.levels[k])
    dvel = np.hypot(
        _level_hand(lr, "uo", nn, k + 1, T_RUN)
        - _level_hand(lr, "uo", nn, k, T_RUN),
        _level_hand(lr, "vo", nn, k + 1, T_RUN)
        - _level_hand(lr, "vo", nn, k, T_RUN))
    return np.abs(np.clip(a, 0.0, 1.0) - 0.5) * dvel


def test_lr_level_linear_vs_half_sum(lr, seeds, lerp_run, half_sum_run):
    """Level-linear against layer-constant reading of the same files.

    The same 30-day run with u/v averaged into the layers between
    neighbouring levels and read as layer-constant data: the first-step
    velocities differ at (almost) every particle; the median separation from the
    level-linear run after 30 days is above ``HALF_SUM_SEP_KM_MIN`` and small
    against the displacement (below ``HALF_SUM_SEP_FRAC_MAX`` of it); and it
    grows with the shear: the separation ranks with the release-point driver
    ``|a - 1/2| |f[k+1] - f[k]|`` (rank correlation above
    ``HALF_SUM_RANK_MIN``), the strongest quarter of the drivers separating
    at least ``HALF_SUM_QUARTILE_RATIO_MIN`` times farther than the weakest.
    Measured values with the constants.
    """
    run = half_sum_run
    _assert_sane_run(lr, run)
    alive = ((run.tr["status"][:, 0] == ParticleStatus.ALIVE)
             & (lerp_run.tr["status"][:, 0] == ParticleStatus.ALIVE))
    du = np.abs(run.tr["vel_u"][alive, 0] - lerp_run.tr["vel_u"][alive, 0])
    differ = float((du > 0.0).mean())
    rec = _rec(N_DAYS_HALF)
    med = float(np.median(_sep_km(run, lerp_run, rec)))
    disp = float(np.median(_disp_km(lerp_run, rec)))

    cloud = np.flatnonzero(seeds["months"].node < 0)
    both = cloud[(run.tr["status"][cloud, rec] == ParticleStatus.ALIVE)
                 & (lerp_run.tr["status"][cloud, rec] == ParticleStatus.ALIVE)]
    sep = _gc_km(run.tr["lon"][both, rec], run.tr["lat"][both, rec],
                 lerp_run.tr["lon"][both, rec], lerp_run.tr["lat"][both, rec])
    drive = _shear_driver(lr, lerp_run, both)
    assert np.isfinite(drive).all()
    rho = float(scipy_stats.spearmanr(drive, sep).statistic)
    q = np.quantile(drive, [0.25, 0.75])
    weak = float(np.median(sep[drive <= q[0]]))
    strong = float(np.median(sep[drive >= q[1]]))
    print(f"measured: half sum vs level-linear: first-step u differs at "
          f"{differ:.4f} of the particles; median separation after "
          f"{N_DAYS_HALF} days {med:.4f} km, median displacement {disp:.2f} km "
          f"(ratio {med / disp:.4f}); rank correlation with the driver "
          f"{rho:.4f}; weakest/strongest driver quartile {weak:.4f}/"
          f"{strong:.4f} km")
    assert differ >= 0.99, differ
    assert med >= HALF_SUM_SEP_KM_MIN, med
    assert med <= HALF_SUM_SEP_FRAC_MAX * disp, (med, disp)
    assert rho >= HALF_SUM_RANK_MIN, rho
    assert strong >= HALF_SUM_QUARTILE_RATIO_MIN * weak, (weak, strong)


# ---------------------------------------------------------------------------
# The two file forms
# ---------------------------------------------------------------------------

#: the raw writer's monthly schedule
MONTHLY = "unit: m first: 1 rate: 1"
#: job segments of the raw files: (start, the day offsets of the month ends
#: from it, the records they hold)
SEGMENTS = (("1951-01-01", (31.0, 59.0, 90.0), (0, 1, 2)),
            ("1951-04-01", (30.0, 61.0, 91.0), (3, 4, 5)))


def _write_ascii_mesh(d: Path, lon: np.ndarray, lat: np.ndarray,
                      tri: np.ndarray, wet: np.ndarray,
                      levels: np.ndarray) -> Path:
    """A FESOM 1.4 ASCII mesh: ``nod2d.out``, ``elem2d.out``, ``nod3d.out``
    (3-D nodes level-major: each level's wet nodes in 2-D order) and
    ``aux3d.out`` (``max_num_layers`` = K, the (N, K) table of 1-based 3-D
    nodes with -999 in dry slots, the 2-D node of every 3-D node, a stub for
    the 3-D element block)."""
    d.mkdir(parents=True, exist_ok=True)
    n2d, k_lev = lon.size, levels.size
    coast = _boundary_nodes(tri, n2d).astype(np.int64)
    with open(d / "nod2d.out", "w") as f:
        f.write(f"{n2d}\n")
        np.savetxt(f, np.column_stack([np.arange(1, n2d + 1), lon, lat, coast]),
                   fmt="%d %.17g %.17g %d")
    with open(d / "elem2d.out", "w") as f:
        f.write(f"{tri.shape[0]}\n")
        np.savetxt(f, tri + 1, fmt="%d")
    kk, nn = np.nonzero(wet)                   # level-major, 2-D order inside
    n3d = kk.size
    z = np.where(levels[kk] > 0.0, -levels[kk], 0.0)
    with open(d / "nod3d.out", "w") as f:
        f.write(f"{n3d}\n")
        np.savetxt(f, np.column_stack([np.arange(1, n3d + 1), lon[nn], lat[nn],
                                       z, np.zeros(n3d)]),
                   fmt="%d %.17g %.17g %.17g %d")
    table = np.full((k_lev, n2d), -999, dtype=np.int64)
    table[wet] = np.arange(1, n3d + 1)
    with open(d / "aux3d.out", "w") as f:
        f.write(f"{k_lev}\n")
        np.savetxt(f, table.T, fmt="%d")
        np.savetxt(f, nn + 1, fmt="%d")
        f.write("1\n1\n")
    return d


def _write_raw(path: Path, var: str, epoch: str, ends: Sequence[float],
               values: np.ndarray, dim: str) -> Path:
    """A raw-writer file: ``var(time, dim)`` float32, stamps at the window
    ends in seconds since the job start ``epoch``, ``output_schedule``."""
    with netCDF4.Dataset(str(path), "w") as nc:
        nc.createDimension("time", None)
        nc.createDimension(dim, values.shape[1])
        t = nc.createVariable("time", "f8", ("time",))
        t.setncattr("long_name", "time")
        t.setncattr("units", f"seconds since {epoch} 0:0:0")
        t.setncattr("calendar", "standard")
        t[:] = np.asarray(ends, dtype=np.float64) * DAY
        v = nc.createVariable(var, "f4", ("time", dim))
        v[:] = values
        nc.setncattr("output_schedule", MONTHLY)
    return path


@dataclass
class _Forms:
    """The converted files and the two runs of the form comparison."""

    meshdir: Path
    raw: Path
    level: _Run
    nodes3d: _Run


@pytest.fixture(scope="module")
def forms(lr, seeds, tmp_path_factory) -> _Forms:
    """The published fields written as an ASCII mesh and raw-writer files
    over the 3-D nodes, and the same 20-day PCHIP run on both forms."""
    d = tmp_path_factory.mktemp("forms")
    meshdir = _write_ascii_mesh(d / "mesh", lr.lon, lr.lat,
                                np.asarray(lr.mesh.elem_nodes, dtype=np.int64),
                                lr.wet, lr.levels)
    raw = d / "raw"
    raw.mkdir()
    zos = _read_records(_files(VAR_2D)[0], VAR_2D, N_REC)
    for epoch, ends, recs in SEGMENTS:
        tag = epoch.replace("-", "")
        for var in ("uo", "vo", "wo", "thetao"):
            # level-major: boolean indexing of (K, N) walks level by level
            vals = np.stack([lr.rec[var][i][lr.wet] for i in recs])
            assert np.isfinite(vals).all()
            _write_raw(raw / f"{var}_lr_{tag}.nc", var, epoch, ends, vals,
                       "nodes_3d")
        _write_raw(raw / f"{VAR_2D}_lr_{tag}.nc", VAR_2D, epoch, ends,
                   zos[list(recs)], "nodes_2d")
    level = _run(_months_cfg(lr, seeds, d, "levelwise", days=N_DAYS_FORMS,
                             pchip=True))
    nodes3d = _run(_months_cfg(lr, seeds, d, "nodes3d", days=N_DAYS_FORMS,
                               data=raw, meshpath=meshdir, pchip=True))
    return _Forms(meshdir=meshdir, raw=raw, level=level, nodes3d=nodes3d)


def _stamp(t: cftime.datetime) -> str:
    """The date and time of an instant, whatever its calendar."""
    return t.strftime("%Y-%m-%dT%H:%M:%S")


def test_lr_form_equivalence(lr, forms):
    """The published fields written as an ASCII FESOM 1.4 mesh and raw-writer
    vectors over the 3-D nodes (two job segments, each ``seconds since`` its
    start, stamps at the month ends, ``output_schedule`` monthly, calendar
    ``standard``, no lon/lat, ``wo`` without a ``standard_name``).  The
    ASCII mesh gives the mesh recovered from the polygons bit for bit; the
    raw records resolve, from their schedule, to the centres of the
    published time bounds; and the 20-day PCHIP run gives bit-identical
    positions, statuses, velocities, sampled temperatures and record times.
    """
    m = _quiet(Fesom14Reader.load_mesh,
               MeshConfig(type="fesom14", meshpath=str(forms.meshdir)))
    assert m.metadata["fesom14_source"] == "ascii"
    for name in ("nodes_lon", "nodes_lat", "elem_nodes", "depth", "sigma",
                 "elem_center_lon", "elem_center_lat"):
        assert_bitwise(getattr(m, name), getattr(lr.mesh, name), name)
    assert_bitwise(m.metadata["uv_data_last"], lr.mesh.metadata["uv_data_last"],
                   "uv_data_last")

    ds = _quiet(Fesom14Reader.open_dataset, str(forms.raw),
                mode_cfg=ModeConfig(Fdrift3D=True, sample_vars=["thetao"]))
    try:
        assert ds.attrs["fesom14_form"] == "nodes3d"
        assert ds.attrs["fesom14_time_source"] == "schedule"
        got = [_stamp(t) for t in ds["time"].values]
    finally:
        ds.close()
    assert got == [_stamp(t) for t in lr.centres]

    level, nodes3d = forms.level, forms.nodes3d
    # the calendars differ (standard / proleptic_gregorian), the dates not
    assert [_stamp(t) for t in nodes3d.dates] == [_stamp(t) for t in level.dates]
    assert len(level.dates) == N_DAYS_FORMS * STEPS_PER_DAY
    assert sorted(nodes3d.tr) == sorted(level.tr)
    for name in sorted(level.tr):
        assert_bitwise(nodes3d.tr[name], level.tr[name], name)
    assert (level.tr["status"] == ParticleStatus.ALIVE).mean() > 0.9
