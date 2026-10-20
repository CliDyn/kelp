"""mode.uv_vertical: linear on FESOM2 element-layout u/v through the runs.

The option reaches element-layout ``u``/``v`` (``uv_interp`` ``leastsq``,
``edge_leastsq`` or ``barycentric``) through ``run_tracking`` and the
direct entry points: the velocity cache marks each element record
``"layer_linear"`` (prefetch windows and the shared snapshot cache
included), the output says ``uv_vertical = "linear"``, restarts resume bit
for bit and refuse the other rule, a ``max_levels`` cut keeps every element
wet down to its bed (reading its last kept point), and the diffusion
compositions and multi-realization runs carry it.  Refused before the
writer: a mesh without the element bottoms (the mesh diagnostic has
neither ``nlevels`` nor ``zbar_e_bottom``; an ASCII mesh directory) and a
mesh with ice-shelf cavities (``ulevels > 1``).

Fixture (all synthetic, ``tests/_fesom2_synth``): a regular 9x9-node grid,
0.25 deg spacing from (4E, 4N), every cell split by its SW-NE diagonal (128
triangles; every interior node has six triangles, point-symmetric about
it), equal element areas from the mesh diagnostic, levels
``[0, 10, 30, 60, 100]`` m (layer centres ``[5, 20, 45, 80]``), element
bottoms from ``nlevels``, node beds the deepest of their elements; eight
daily records of float64 element ``u``/``v`` that vary from element to
element and in time,

``u_e(k, r) = a + gx*lon_e + gy*lat_e + b*p_k + c*r``

(``lon_e``, ``lat_e`` the element centres the reader computes, ``p_k`` the
layer centre, ``r`` the record), zero below each element's bed, ``w = 0``.
The three interpolators reproduce a field linear in the centres exactly
where every stencil member is wet (the two plane fits; the element
barycentric vertex means, by the symmetric rings and equal areas, on
triangles whose vertices are interior), so a particle away from the mesh
edge moves by the hand stage combination of
``u(lon, lat, z, t) = a + gx*lon + gy*lat + b*clip(z, 5, p_last) + c*t``.
"""
from __future__ import annotations

import contextlib
import dataclasses
import io
import warnings
from pathlib import Path

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
netCDF4 = pytest.importorskip("netCDF4")

from kelp import tracking  # noqa: E402
from kelp.core.config import (AppConfig, DataConfig,  # noqa: E402
                              DiffusionConfig, MeshConfig, ModeConfig,
                              NumericsConfig, TimeConfig)
from kelp.core.geometry import (DEG2RAD, METERS_PER_DEG_LAT,  # noqa: E402
                                wrap_lon)
from kelp.core.mesh import find_nearest_nodes  # noqa: E402
from kelp.core.particles import ParticleStatus, make_state  # noqa: E402
from kelp.numerics.vertical_uv import is_checked  # noqa: E402
from kelp.readers import extract_time_info, get_reader  # noqa: E402
from kelp.writers import (TrajectoryWriter, load_restart,  # noqa: E402
                          load_trajectories)
from tests import _fesom2_synth as synth  # noqa: E402
from tests._bitwise import assert_bitwise  # noqa: E402
from tests.test_integrators_depth_linear import HAND  # noqa: E402

# ---------------------------------------------------------------------------
# The fixture
# ---------------------------------------------------------------------------

LEVELS = (0.0, 10.0, 30.0, 60.0, 100.0)
K = len(LEVELS)
#: layer centres (the data points of a "layer_linear" field)
P_PTS = (5.0, 20.0, 45.0, 80.0)
NX = NY = 9
GRID0, STEP = 4.0, 0.25
N_REC = 8
DT = 86400.0
#: equal element areas (m^2) in the mesh diagnostic
AREA = 7.0e8

#: (a, gx, gy, b, c) of u and v: m/s, m/s per degree, m/s per metre, m/s
#: per record
COEF_U = (0.04, 0.02, -0.01, 2.0e-4, 0.002)
COEF_V = (-0.02, 0.01, 0.005, -1.0e-4, -0.001)

#: away from the mesh edge for the whole run; depths above the first
#: centre, between centres and below the last one
SEEDS = [(4.62, 4.71, 3.0), (5.13, 5.28, 12.5), (4.87, 4.84, 33.0),
         (4.71, 5.11, 70.0), (5.04, 4.63, 90.0)]
N_STEPS = 5

INTERPS = ["leastsq", "edge_leastsq", "barycentric"]
INTEGRATORS = ["euler", "rk2", "rk4"]
#: stage times (fractions of dt) of each integrator's stages
STAGE_C = {"euler": (0.0,), "rk2": (0.0, 0.5), "rk4": (0.0, 0.5, 0.5, 1.0)}

_FIELDS = ("lon", "lat", "z", "status", "vel_u", "vel_v", "vel_w")


def _col(coef, lon, lat, z, t):
    a, gx, gy, b, c = coef
    return a + gx * lon + gy * lat + b * z + c * t


def _hand_uv(lon, lat, z, t, z_hi=P_PTS[-1]):
    """The hand rule at a point: the depth clipped to the first centre and
    the last usable one (``z_hi``)."""
    zc = min(max(z, P_PTS[0]), z_hi)
    return _col(COEF_U, lon, lat, zc, t), _col(COEF_V, lon, lat, zc, t)


def _hand_run(integrator, seeds, n_steps, *, dt=DT, t0=0.0,
              z_hi=P_PTS[-1]):
    """Positions after each step, ``(particle, step)``: the stage
    combination of ``integrator`` (w = 0) with the hand rule at each
    stage's time (records ``t0 + (s + c) * dt / DT``)."""
    lon_out, lat_out = [], []
    for lo, la, z in seeds:
        lon, lat, row_lon, row_lat = lo, la, [], []
        for s in range(n_steps):
            t_s = t0 + s * dt / DT

            def uv(stage, x, y, zz, t_s=t_s):
                t = t_s + STAGE_C[integrator][stage] * dt / DT
                return _hand_uv(x, y, zz, t, z_hi)
            dlon, dlat, _ = HAND[integrator](lon, lat, z, dt, 1.0, 0.0, uv)
            lon = float(wrap_lon(lon + dlon))
            lat = lat + dlat
            row_lon.append(lon)
            row_lat.append(lat)
        lon_out.append(row_lon)
        lat_out.append(row_lat)
    return np.array(lon_out), np.array(lat_out)


def _assert_metres(lon, lat, h_lon, h_lat, tol=1e-9):
    """Every position within ``tol`` metres of the hand position."""
    dx = (np.abs(lon - h_lon) * np.cos(h_lat * DEG2RAD)
          * METERS_PER_DEG_LAT)
    dy = np.abs(lat - h_lat) * METERS_PER_DEG_LAT
    assert np.all(dx <= tol) and np.all(dy <= tol), (dx.max(), dy.max())


def _node_beds(tri, elem_bed, n_nodes):
    """A node's bed is the deepest of its elements (FESOM2)."""
    bed = np.zeros(n_nodes)
    for e, row in enumerate(np.asarray(tri) - 1):
        bed[row] = np.maximum(bed[row], elem_bed[e])
    return bed


def _write_mesh(path, *, elem_bed=None, bottoms="nlevels", ulevels=None):
    """The grid's mesh diagnostic; ``bottoms`` ``"nlevels"``,
    ``"zbar_e_bottom"`` or None (neither variable)."""
    lon, lat, tri = synth.grid_geometry(NX, NY, GRID0, GRID0, STEP)
    n_el = tri.shape[0]
    bed = (np.full(n_el, LEVELS[-1]) if elem_bed is None
           else np.asarray(elem_bed, dtype=np.float64))
    kw = {}
    if bottoms == "nlevels":
        kw["nlevels"] = np.searchsorted(LEVELS, bed) + 1
    elif bottoms == "zbar_e_bottom":
        kw["zbar_e_bottom"] = bed
    synth.write_meshdiag(
        path, "ugrid", geometry=(lon, lat, tri), zbar=-np.asarray(LEVELS),
        zbar_n_bottom=-_node_beds(tri, bed, lon.size),
        elem_area=np.full(n_el, AREA), ulevels=ulevels, **kw)
    return path


def _load_mesh(path):
    with contextlib.redirect_stdout(io.StringIO()):
        return get_reader("fesom2").load_mesh(
            MeshConfig(type="fesom2", meshpath=str(path)))


def _write_data(data, mesh, elem_bed):
    """u, v on the elements (zero below each element's bed), w and ssh
    zero on the nodes; float64, ``N_REC`` daily records."""
    lon_c, lat_c = mesh.elem_center_lon, mesh.elem_center_lat
    n_el, n_nd = mesh.n_elements, mesh.n_nodes
    for var, coef in (("u", COEF_U), ("v", COEF_V)):
        def values(year, k, coef=coef):
            wet = LEVELS[k + 1] <= elem_bed
            return np.stack([np.where(wet, _col(coef, lon_c, lat_c,
                                                P_PTS[k], r), 0.0)
                             for r in range(N_REC)])
        synth.write_multiyear(data, var, {1958: N_REC}, node_layout=False,
                              vertical="nz1", values=values,
                              continuous_time=True, n_space=n_el,
                              n_levels=K, dtype=np.float64)
    zero = lambda y, k: 0.0                              # noqa: E731
    synth.write_multiyear(data, "w", {1958: N_REC}, vertical="nz",
                          values=zero, continuous_time=True, n_space=n_nd,
                          n_levels=K, dtype=np.float64)
    synth.write_multiyear(data, "ssh", {1958: N_REC}, vertical=None,
                          values=zero, continuous_time=True, n_space=n_nd,
                          dtype=np.float64)


def _env(work, *, elem_bed=None, bottoms="nlevels", ulevels=None):
    """Mesh diagnostic and data under ``work``: ``(mesh_path, data_dir)``."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    lon, lat, tri = synth.grid_geometry(NX, NY, GRID0, GRID0, STEP)
    bed = (np.full(tri.shape[0], LEVELS[-1]) if elem_bed is None
           else np.asarray(elem_bed, dtype=np.float64))
    mesh_path = _write_mesh(work / "fesom.mesh.diag.nc", elem_bed=bed,
                            bottoms=bottoms, ulevels=ulevels)
    data = work / "data"
    data.mkdir()
    _write_data(data, _load_mesh(mesh_path), bed)
    return mesh_path, data


def _mode(**kw):
    base = dict(Fdrift2D=False, Fdrift3D=True, var_u="u", var_v="v",
                var_w="w", var_eta="ssh", uv_vertical="linear")
    base.update(kw)
    return ModeConfig(**base)


def _cfg(env, *, integrator="euler", uv_interp="leastsq", seeds=SEEDS,
         n_steps=N_STEPS, start=0, out="traj.nc", num_runs=1, dt=DT,
         mesh_path=None, **kw):
    """A run_tracking config on ``env``; ``kw`` goes to NumericsConfig
    when it names one of its fields, else to the mode."""
    mesh_file, data = env
    work = Path(data).parent
    seeds_path = work / f"seeds_{out}.dat"
    seeds_path.write_text("".join(f"{lo} {la} {z}\n" for lo, la, z in seeds))
    num_names = {f.name for f in dataclasses.fields(NumericsConfig)}
    num_kw = {k: v for k, v in kw.items() if k in num_names}
    mode_kw = {k: v for k, v in kw.items() if k not in num_names}
    num_kw.setdefault("w_interp", "barycentric")
    return AppConfig(
        mode=_mode(**mode_kw),
        numerics=NumericsConfig(dt=dt, integrator=integrator,
                                uv_interp=uv_interp, **num_kw),
        mesh=MeshConfig(type="fesom2",
                        meshpath=str(mesh_path or mesh_file)),
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
    with netCDF4.Dataset(str(path)) as nc:
        return {k: nc.getncattr(k) for k in nc.ncattrs()}


def _assert_same_run(path, ref, fields=_FIELDS):
    a, b = load_trajectories(str(path)), load_trajectories(str(ref))
    for k in fields:
        assert_bitwise(a[k], b[k], k)


def _state(mesh, seeds=SEEDS):
    lon, lat, z = (np.array(c, dtype=np.float64) for c in zip(*seeds))
    st = make_state(lon, lat, z)
    return st.replace(nearest_node=find_nearest_nodes(mesh, st.lon, st.lat))


def _open(env):
    mesh = _load_mesh(env[0])
    reader = get_reader("fesom2")
    ds = reader.open_dataset(str(env[1]), mode_cfg=_mode())
    return reader, mesh, ds


# ---------------------------------------------------------------------------
# The option on element data: the hand rule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("integrator", INTEGRATORS)
@pytest.mark.parametrize("uv_interp", INTERPS)
def test_run_tracking_linear_elements(tmp_path, uv_interp, integrator):
    """Every interpolator x euler/rk2/rk4 (discrete records, the RK stages
    blend the next record): every position equals the hand stage
    combination to 1e-9 m; the output says ``uv_vertical = "linear"``; the
    layer rule (option off) moves the seeds between centres elsewhere."""
    env = _env(tmp_path)
    cfg = _cfg(env, integrator=integrator, uv_interp=uv_interp)
    _run(cfg)
    t = load_trajectories(cfg.data.output_path)
    assert (t["status"] == ParticleStatus.ALIVE).all()
    h_lon, h_lat = _hand_run(integrator, SEEDS, N_STEPS)
    _assert_metres(t["lon"], t["lat"], h_lon, h_lat)
    np.testing.assert_array_equal(
        t["z"], np.repeat([[s[2]] for s in SEEDS], N_STEPS, axis=1))
    assert _global_attrs(cfg.data.output_path)["uv_vertical"] == "linear"
    off = _cfg(env, integrator=integrator, uv_interp=uv_interp,
               uv_vertical="layer", out="off.nc")
    _run(off)
    t_off = load_trajectories(off.data.output_path)
    # seeds 1-3 sit between centres: the layer value is 1.5e-3 .. 5e-3 m/s
    # off, hundreds of metres a day
    gap = (np.abs(t_off["lon"] - t["lon"]) * METERS_PER_DEG_LAT
           * np.cos(t["lat"] * DEG2RAD))
    assert np.all(gap[1:4] > 50.0)
    assert "uv_vertical" not in _global_attrs(off.data.output_path)


@pytest.mark.parametrize("mode", ["lerp", "pchip"])
@pytest.mark.parametrize("uv_interp", INTERPS)
def test_temporal_interp_linear_elements(tmp_path, uv_interp, mode):
    """Temporal interpolation, dt half a record, rk4: the stages blend the
    records at their own times (linearly, or by PCHIP, which reproduces a
    field linear in time); every position equals the hand rule to 1e-9 m."""
    env = _env(tmp_path)
    recon = "pchip" if mode == "pchip" else "linear"
    cfg = _cfg(env, integrator="rk4", uv_interp=uv_interp, dt=DT / 2,
               n_steps=8, temporal_interp=True,
               temporal_reconstruction=recon)
    _run(cfg)
    t = load_trajectories(cfg.data.output_path)
    assert (t["status"] == ParticleStatus.ALIVE).all()
    h_lon, h_lat = _hand_run("rk4", SEEDS, 8, dt=DT / 2)
    _assert_metres(t["lon"], t["lat"], h_lon, h_lat)


# ---------------------------------------------------------------------------
# Refusals before the writer
# ---------------------------------------------------------------------------

_NO_BOTTOMS = "the mesh diagnostic has neither nlevels nor zbar_e_bottom"
_FIX = ("use the run's full fesom.mesh.diag.nc, or re-extract the region "
        "with utils/extract_fesom_region.py, which keeps them")
_CAVITIES = ("ice-shelf cavities are not supported by mode.uv_vertical: "
             "linear on element data yet")


class _CountingReader:
    """The FESOM2 reader with its reads counted."""

    def __init__(self):
        self.base = get_reader("fesom2")
        self.reads = 0

    def probe_uv_layout(self, ds, mode_cfg, mesh):
        return self.base.probe_uv_layout(ds, mode_cfg, mesh)

    def read_velocities(self, *a, **k):
        self.reads += 1
        return self.base.read_velocities(*a, **k)


def test_missing_element_bottoms_refused_before_writer(tmp_path):
    """A mesh diagnostic with neither ``nlevels`` nor ``zbar_e_bottom``:
    the option on element data is refused before any output file exists,
    naming the mesh file and the fix; direct calls are refused before any
    read and before anything is written to the caller's writer; the layer
    rule runs on the same mesh."""
    env = _env(tmp_path, bottoms=None)
    cfg = _cfg(env)
    with pytest.raises(ValueError) as err:
        _run(cfg)
    msg = str(err.value)
    assert str(env[0]) in msg and _NO_BOTTOMS in msg and _FIX in msg
    assert not Path(cfg.data.output_path).exists()
    # direct call with a writer: nothing read, nothing written
    _, mesh, ds = _open(env)
    reader = _CountingReader()
    w = TrajectoryWriter(str(tmp_path / "direct.nc"), len(SEEDS)).open()
    try:
        with pytest.raises(ValueError, match=_NO_BOTTOMS):
            tracking.run_single_realization(
                mesh, ds, reader, _state(mesh), _mode(),
                NumericsConfig(dt=DT, uv_interp="leastsq",
                               w_interp="barycentric"),
                start_time_idx=0, n_steps=2, writer=w)
        assert reader.reads == 0 and w.n_obs == 0
        assert w.global_attr("kelp_version") is None
    finally:
        w.close()
        ds.close()
    off = _cfg(env, uv_vertical="layer", out="off.nc")
    _run(off)
    assert Path(off.data.output_path).exists()


def test_zbar_e_bottom_alone_is_enough(tmp_path):
    """The element bottoms from ``zbar_e_bottom`` (no ``nlevels``): the
    same run as with ``nlevels``, bit for bit."""
    a = _cfg(_env(tmp_path / "a"), integrator="rk2")
    b = _cfg(_env(tmp_path / "b", bottoms="zbar_e_bottom"), integrator="rk2")
    _run(a)
    _run(b)
    _assert_same_run(b.data.output_path, a.data.output_path)


def test_ascii_mesh_refused(tmp_path):
    """An ASCII mesh directory declares no element bottoms: the option on
    element data is refused before the writer, naming the directory."""
    env = _env(tmp_path / "nc")
    lon, lat, tri = synth.grid_geometry(NX, NY, GRID0, GRID0, STEP)
    mdir = tmp_path / "ascii_mesh"
    mdir.mkdir()
    (mdir / "nod2d.out").write_text(f"{lon.size}\n" + "".join(
        f"{i + 1} {x:.17g} {y:.17g} 0\n"
        for i, (x, y) in enumerate(zip(lon, lat))))
    (mdir / "elem2d.out").write_text(f"{tri.shape[0]}\n" + "".join(
        f"{a} {b} {c}\n" for a, b, c in tri))
    (mdir / "aux3d.out").write_text("".join(f"{z:g}\n" for z in LEVELS))
    (mdir / "depth.out").write_text(f"{LEVELS[-1]:g}\n" * lon.size)
    cfg = _cfg(env, mesh_path=mdir, out="ascii.nc")
    with pytest.raises(ValueError) as err:
        _run(cfg)
    msg = str(err.value)
    assert str(mdir) in msg and _NO_BOTTOMS in msg and _FIX in msg
    assert not Path(cfg.data.output_path).exists()


def test_cavities_refused(tmp_path):
    """``ulevels > 1`` anywhere (an ice-shelf cavity): refused before the
    writer, naming the mesh file; ``ulevels`` all 1 runs, bit for bit the
    run without the variable."""
    n_el = 2 * (NX - 1) * (NY - 1)
    ul = np.ones(n_el, dtype=np.int32)
    ul[37] = 2
    env = _env(tmp_path / "cav", ulevels=ul)
    cfg = _cfg(env)
    with pytest.raises(ValueError) as err:
        _run(cfg)
    msg = str(err.value)
    assert _CAVITIES in msg and str(env[0]) in msg
    assert not Path(cfg.data.output_path).exists()
    flat = _cfg(_env(tmp_path / "flat", ulevels=np.ones(n_el)))
    ref = _cfg(_env(tmp_path / "ref"))
    _run(flat)
    _run(ref)
    _assert_same_run(flat.data.output_path, ref.data.output_path)


# ---------------------------------------------------------------------------
# Caches, element search, restarts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("uv_interp", INTERPS)
def test_prefetch_and_shared_cache(tmp_path, monkeypatch, uv_interp):
    """Prefetch windows of 3 and the shared snapshot cache each give the
    plain run bit for bit; a second shared cache attaches the published
    element record without reading it, marked ``"layer_linear"`` and
    checked, bit for bit the local read."""
    from kelp.core.shared_fields import SharedFieldCache
    env = _env(tmp_path)
    base = _cfg(env, integrator="rk2", uv_interp=uv_interp, out="base.nc")
    _run(base)
    loads = []
    orig = tracking._VelCache._load_window

    def counting(self, start, *a, **k):
        loads.append(start)
        return orig(self, start, *a, **k)
    monkeypatch.setattr(tracking._VelCache, "_load_window", counting)
    pre = _cfg(env, integrator="rk2", uv_interp=uv_interp, prefetch_size=3,
               out="pre.nc")
    _run(pre)
    assert len(loads) >= 2
    shm = tmp_path / "shm"
    shm.mkdir()
    shared = _cfg(env, integrator="rk2", uv_interp=uv_interp,
                  shared_fields_dir=str(shm),
                  shared_fields_generation="elements", out="shared.nc")
    attached = []
    orig_get = tracking._VelCache._shared_get

    def shared_get(self, file_idx, *a, **k):
        attached.append(file_idx)
        return orig_get(self, file_idx, *a, **k)
    monkeypatch.setattr(tracking._VelCache, "_shared_get", shared_get)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _run(shared)
    assert not any("shared_fields" in str(w.message) for w in rec)
    assert sorted(set(attached)) == list(range(N_STEPS + 1))
    for cfg in (pre, shared):
        _assert_same_run(cfg.data.output_path, base.data.output_path)
    # the attached record
    reader, mesh, ds = _open(env)
    r2 = _CountingReader()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        c1 = tracking._VelCache(shared=SharedFieldCache(shm / "c", "tagE"))
        c1.get(3, reader, ds, mesh, _mode())
        c2 = tracking._VelCache(shared=SharedFieldCache(shm / "c", "tagE"))
        v2 = c2.get(3, r2, ds, mesh, _mode())
    assert r2.reads == 0
    assert not any("shared_fields" in str(w.message) for w in rec)
    local = tracking._VelCache().get(3, reader, ds, mesh, _mode())
    for v in (v2, local):
        assert v.uv_layout == "element" and v.uv_vertical == "layer_linear"
        assert is_checked(v, mesh)
    for f in ("u", "v", "w", "eta"):
        assert_bitwise(getattr(v2, f), getattr(local, f), f)
    ds.close()


def _wall_beds():
    """Element beds with walls: every fifth element 30 m (dry below 30 m),
    the rest 100 m."""
    n_el = 2 * (NX - 1) * (NY - 1)
    return np.where(np.arange(n_el) % 5 == 0, 30.0, 100.0)


#: depths below the 30 m elements' bed (walls) and above it
WALL_SEEDS = [(4.62, 4.71, 40.0), (5.13, 5.28, 55.0), (4.87, 4.84, 12.5),
              (4.71, 5.11, 70.0), (5.04, 4.63, 33.0)]


@pytest.mark.parametrize("bc", ["freeslip", "noslip"])
@pytest.mark.parametrize("integrator", INTEGRATORS)
def test_walk_equals_local_run(tmp_path, integrator, bc):
    """``elem_search: walk`` with ``edge_leastsq`` on element data with
    walls at depth: bit for bit the run with the per-step search."""
    env = _env(tmp_path, elem_bed=_wall_beds())
    runs = {}
    for search in ("local", "walk"):
        cfg = _cfg(env, integrator=integrator, uv_interp="edge_leastsq",
                   edge_leastsq_bc=bc, elem_search=search,
                   seeds=WALL_SEEDS, out=f"{search}.nc")
        _run(cfg)
        runs[search] = cfg.data.output_path
    t = load_trajectories(runs["walk"])
    assert np.isfinite(t["lon"]).all()
    assert _global_attrs(runs["walk"])["elem_search"] == "walk"
    _assert_same_run(runs["walk"], runs["local"])


@pytest.mark.parametrize("uv_interp", ["leastsq", "edge_leastsq"])
def test_restart_with_option(tmp_path, uv_interp):
    """A run cut at a checkpoint and resumed under the option equals the
    continuous run bit for bit (positions, velocities, record times, the
    final restart); the checkpoint carries ``"linear"`` and resuming it
    without the option is refused."""
    env = _env(tmp_path, elem_bed=_wall_beds())
    reader, mesh, ds = _open(env)
    file_times, file_dt = extract_time_info(ds)
    numerics = NumericsConfig(dt=DT, integrator="rk2", uv_interp=uv_interp,
                              w_interp="barycentric")
    mode = _mode()

    def run(name, state, start, n, start_step_i):
        path = str(tmp_path / f"{name}.nc")
        w = TrajectoryWriter(path, state.n_particles).open()
        try:
            tracking.run_single_realization(
                mesh, ds, reader, state, mode, numerics,
                start_time_idx=start, n_steps=n, writer=w,
                checkpoint_freq=2, file_times=file_times, file_dt=file_dt,
                start_step_i=start_step_i)
        finally:
            w.close()
        return path
    init = _state(mesh, WALL_SEEDS)
    whole = run("whole", init, 0, 4, 0)
    a = run("a", init, 0, 2, 0)
    r = load_restart(a)
    assert r["uv_vertical"] == "linear"
    tracking._validate_restart_uv_vertical(r["uv_vertical"], mode)
    b = run("b", r["state"], r["file_idx"] + 1, 2, r["step_i"] + 1)
    tw, ta, tb = (load_trajectories(p) for p in (whole, a, b))
    assert_bitwise(np.concatenate([ta["time"], tb["time"]]), tw["time"],
                   "time")
    for k in _FIELDS:
        assert_bitwise(np.concatenate([ta[k], tb[k]], axis=1), tw[k], k)
    rw, rb = load_restart(whole), load_restart(b)
    assert (rb["step_i"], rb["file_idx"]) == (rw["step_i"], rw["file_idx"])
    for k in ("lon", "lat", "z"):
        assert_bitwise(getattr(rb["state"], k), getattr(rw["state"], k), k)
    with pytest.raises(ValueError,
                       match="written with mode.uv_vertical 'linear', this "
                             "run has 'layer'"):
        tracking._validate_restart_uv_vertical(
            r["uv_vertical"], _mode(uv_vertical="layer"))
    ds.close()


# ---------------------------------------------------------------------------
# max_levels, diffusion, realizations
# ---------------------------------------------------------------------------

def _cut_beds():
    """Element beds 60 or 100 m, every seventh element 30 m (a wall for the
    particles below 30 m)."""
    n_el = 2 * (NX - 1) * (NY - 1)
    e = np.arange(n_el)
    return np.where(e % 7 == 3, 30.0, np.where(e % 2 == 0, 60.0, 100.0))


#: every particle below the 30 m cut and above the beds 60 / 100 m
CUT_SEEDS = [(4.62, 4.71, 40.0), (5.13, 5.28, 50.0), (4.87, 4.84, 35.0),
             (4.71, 5.11, 45.0), (5.04, 4.63, 57.0)]


@pytest.mark.parametrize("integrator", ["euler", "rk2"])
def test_max_levels_cut_blend(tmp_path, integrator):
    """``max_levels: 3`` keeps the levels 0, 10, 30 m (points 5 and 20 m):
    the elements stay wet down to their beds (60 or 100 m), so particles
    below the cut read each wet ring member at its last kept point (20 m)
    and fit those values — the hand rule at 20 m to 1e-9 m, not one
    member's value; the 30 m elements are walls, left out of the fit.
    Uncut, the same particles read every member at their own depth."""
    env = _env(tmp_path, elem_bed=_cut_beds())
    cut = _cfg(env, integrator=integrator, seeds=CUT_SEEDS, max_levels=3,
               out="cut.nc")
    _run(cut)
    t = load_trajectories(cut.data.output_path)
    assert (t["status"] == ParticleStatus.ALIVE).all()
    h_lon, h_lat = _hand_run(integrator, CUT_SEEDS, N_STEPS, z_hi=P_PTS[1])
    _assert_metres(t["lon"], t["lat"], h_lon, h_lat)
    full = _cfg(env, integrator=integrator, seeds=CUT_SEEDS, out="full.nc")
    _run(full)
    t_full = load_trajectories(full.data.output_path)
    # uncut, a particle at or above 45 m (the last centre of the 60 m
    # elements) reads every wet member at its own depth
    up = [i for i, s in enumerate(CUT_SEEDS) if s[2] <= P_PTS[2]]
    h_lon, h_lat = _hand_run(integrator, [CUT_SEEDS[i] for i in up],
                             N_STEPS)
    _assert_metres(t_full["lon"][up], t_full["lat"][up], h_lon, h_lat)
    gap = (np.abs(t_full["lon"] - t["lon"]) * METERS_PER_DEG_LAT
           * np.cos(t["lat"] * DEG2RAD))
    assert np.all(gap > 50.0)


@pytest.mark.parametrize("uv_interp", INTERPS)
def test_split_uniform_and_az_with_option(tmp_path, uv_interp):
    """Split uniform Kh and the Az kick on element data with the option:
    the same seed gives the same run bit for bit, another seed another run;
    the depths move (the kick), every particle stays alive."""
    env = _env(tmp_path, elem_bed=_wall_beds())

    def diff(seed):
        return DiffusionConfig(enabled=True, scheme="uniform", kh_zonal=5.0,
                               kh_meridional=5.0, seed=seed,
                               composition="split", az_enabled=True,
                               az_const=1.0e-4)
    paths = []
    for name, seed in (("a", 11), ("b", 11), ("c", 12)):
        cfg = _cfg(env, integrator="rk2", uv_interp=uv_interp,
                   seeds=WALL_SEEDS, diffusion=diff(seed), out=f"{name}.nc")
        _run(cfg)
        paths.append(cfg.data.output_path)
    _assert_same_run(paths[1], paths[0])
    ta, tc = load_trajectories(paths[0]), load_trajectories(paths[2])
    assert (ta["status"] == ParticleStatus.ALIVE).all()
    assert not np.array_equal(ta["lon"], tc["lon"])
    assert np.all(np.ptp(ta["z"], axis=1) > 0.0)
    assert _global_attrs(paths[0])["uv_vertical"] == "linear"


@pytest.mark.parametrize("uv_interp", INTERPS)
def test_multi_realization(tmp_path, uv_interp):
    """Two realizations (``num_runs: 2``): each output is the single run,
    bit for bit, and says ``uv_vertical = "linear"``."""
    env = _env(tmp_path, elem_bed=_wall_beds())
    single = _cfg(env, integrator="rk2", uv_interp=uv_interp,
                  seeds=WALL_SEEDS, out="single.nc")
    _run(single)
    multi = _cfg(env, integrator="rk2", uv_interp=uv_interp,
                 seeds=WALL_SEEDS, num_runs=2, out="multi.nc")
    _run(multi)
    for r in range(2):
        path = Path(env[1]).parent / f"multi_r{r}.nc"
        _assert_same_run(path, single.data.output_path)
        assert _global_attrs(path)["uv_vertical"] == "linear"


def test_field_without_uv_layout_is_element_layout(tmp_path):
    """A field-like object without ``uv_layout`` counts as element layout,
    the default of ``VelocityField``: the record check reads the element
    table, and a node-shaped one is refused naming the elements."""
    from types import SimpleNamespace
    from kelp.numerics.vertical_uv import (check_depth_linear_record,
                                           depth_points)
    mesh = _load_mesh(_write_mesh(tmp_path / "m.nc"))
    n_el, n_nd = mesh.n_elements, mesh.n_nodes
    elem = SimpleNamespace(u=np.zeros((K - 1, n_el)),
                           v=np.zeros((K - 1, n_el)),
                           uv_vertical="layer_linear", _dl_token=None)
    check_depth_linear_record(elem, mesh)
    assert elem._dl_token is depth_points(mesh, "layer_linear", "element")
    node = SimpleNamespace(u=np.zeros((K - 1, n_nd)),
                           v=np.zeros((K - 1, n_nd)),
                           uv_vertical="layer_linear", _dl_token=None)
    with pytest.raises(ValueError,
                       match=rf"u has {n_nd} elements, the mesh has {n_el}"):
        check_depth_linear_record(node, mesh)
