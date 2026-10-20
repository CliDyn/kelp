"""Depth-linear horizontal velocity against analytic answers, through the
public run path.

``mode.uv_vertical: linear`` reads node-layout u/v (``uv_interp:
barycentric``) and element-layout u/v (``leastsq``, ``edge_leastsq``,
element ``barycentric``) linearly in depth between the layer centres; a
stencil member dry at the particle depth is a wall, like the coast.  This
module checks the rule against answers known in closed form, on synthetic
FESOM2 files read by the FESOM2 reader and run by ``run_tracking`` (the
profile tests call the public interpolators), and on FESOM-C sigma files
(below):

- ``test_depth_constant_same_as_layer`` — u, v independent of depth (per
  column constants, varying from column to column and in time; NaN below
  each column's last usable point, so a read there would show): the linear
  and the layer run agree at every record, bit for bit where every member
  reads a stored value (above the first centre, at a centre, below the
  last one; a particle exactly at its members' bed; one between a partial
  bed and the level above it), within a few ulp elsewhere.
- ``test_linear_in_depth_exact`` — ``u = u0 + gy (lat - lat_c) + b z``,
  ``v = 0``: the linear run moves by the closed form ``x = u T`` (with a
  uniform ``w`` crossing layers: ``x = u(z0) T + b W T^2 / 2``, exact for
  RK2 and RK4), the layer run is off by ``b (c_k - z) T``.
- ``test_profile_reconstruction_order`` — ``u = G(x) cos(pi z / H)`` on 10,
  20 and 40 layers, read at 1601 depths: the maximum error falls by 4 per
  halving in linear mode (second order), by 2 in layer mode (first order).
- ``test_trajectory_convergence`` — the same field, ``w = 0``, 50 particles
  at equally spaced depths, RK4 against a reference solution: the RMS end
  error falls by 4 per halving in linear mode and by 2 in layer mode; the
  time-step error is below 1 % of the error of either mode.
- ``test_trench_wall_keeps_speed`` — a uniform flow along a straight trench
  below the bed of the shelf beside it: linear mode keeps the speed (the
  shelf is a wall), layer mode is slower (it reads the shelf's zeros).

On FESOM-C sigma meshes the same rule is read in the sigma fraction
``s = z / H_p`` between the layer centres ``1 - (sigma_k + sigma_{k+1}) /
2``, every column wet to the bed; synthetic FESOM-C files read by the
FESOM-C reader and run by ``run_tracking``:

- ``test_sigma_depth_constant_same_as_layer`` — per-column constants: the
  linear run equals the layer run, bit for bit beyond the outer centres,
  within a few ulp between them.
- ``test_sigma_profile_reconstruction_order`` and
  ``test_sigma_trajectory_convergence`` — ``u = G(x) cos(pi s)`` on 10, 20
  and 40 sigma layers, the two forms above (a zero dia-sigma W keeps each
  particle's ``s`` while its depth follows the sloping bed and the moving
  surface): linear mode second order, layer mode first order.

Fixtures (all synthetic, ``tests/_fesom2_synth``): regular grids of
:func:`tests._fesom2_synth.grid_geometry` (0.25 deg spacing from 4E 4N,
every cell split by its SW-NE diagonal), equal element areas, element beds
from ``zbar_e_bottom`` (partial beds included), node beds the deepest of
their elements unless set otherwise, float64 daily records, ``w`` zero
unless stated.  The FESOM-C files (:func:`_write_fesomc`) use the same
grids, uniform sigma, a bed sloping in lon and lat and a surface rising and
tilted, element ``u``/``v`` and the same columns on the nodes
(``unode``/``vnode``).

Mutation pass (commit 204a8b4 with this module, 2026-10-20, every row;
the depth_constant cells of M2, M6 and M8 again on b03afac, the same day,
after the field's constants moved into one binade: the same cells red;
every row again on 575bcfb, the same day, with the no-slip trench case:
the same cells red, plus that case for M10 and the old sources).
Each defect went into a full copy of the checkout, one at a time, with a
fresh Numba cache, and this module ran against it; the unmutated copy
passed 45 of 45, the sources before the element work (28b20c9) failed 38
(element data refused; node layout: a shallow vertex read at its own
bed).  Every row again on 60d4a42 with the sigma tests, the same day,
together with ``tests/test_integrators_depth_linear_sigma.py`` (the sigma
step by hand: "step" below), plus the sigma mutants M13 .. M18 (M5 on the
stage-2 coordinate the sigma work introduced): the unmutated copy passed
368 cells of the two modules (4 skipped), the sources before the sigma
work (4013cb4) failed 323 — every sigma cell, 20 here and 303 of the step
(sigma meshes refused; the step's spies find no ``_sigma_frac_for_uv``) —
and passed the 45 z-level cells.  M1 .. M12 turned the same cells of this
module red as before, plus the sigma cells named below; the others leave
every sigma cell green (a sigma column is wet at every ``s``: no walls, no
declaration, no bed).  Red tests (parametrised cells) and the mechanism,
from the assertion messages:

=====  ===========================  =======================================
id     red tests                    mutant -> mechanism
=====  ===========================  =======================================
M1     linear_in_depth_exact 18,    ``uv_at`` weights swapped -> linear
       profile 4, trajectory 4,     mode 6.4 % off the closed form (1.8 %
       sigma_profile 4,             with W), first order in both tests
       sigma_trajectory 4,          (node RK2 is not red: its fused twin
       step 212                     blends in line, not through ``uv_at``);
                                    in ``s`` first order too (ratios 1.99
                                    .. 2.09), the step off its hand stages
M2     depth_constant 12,           ``uv_wet``: ``<=`` -> ``<`` -> the
       profile 4                    particle at its members' bed loses them
                                    all: 0.014 .. 0.040 deg off the layer
                                    run; the profile's bed point reads a
                                    fallback (linear ratios 0.93 .. 1.17)
M3     trench[leastsq]              a dry ring element kept in the
                                    ``leastsq`` fit -> the shelf's last
                                    value pulls: 18 % slow
M4     trench[leastsq]              the ``leastsq`` too-few fallback
                                    returns zero -> the particle at the
                                    trench end stops (``vel_u`` 0)
M5     linear_in_depth_exact        RK2 stage 2 at the entry depth -> 1.5 %
       [element-*-rk2-w] 3,         off ``u(z0) T + B W T^2 / 2`` (node
       step 92 (rk2)                RK2 runs its fused twin); on a sigma
                                    mesh the entry depth in metres: stage
                                    2 reads the bottom centre
M6     depth_constant[element] 9    the element table without the
                                    declaration -> the record check meets
                                    the NaN below an element's last point
M7     linear_in_depth_exact        the fused Euler twin reads the layer
       [element-leastsq-euler-w0]   row -> the layer answer, 3.6 % off
M8     depth_constant 12            ``bed = lower[last]`` -> the particle
                                    between a partial bed's last level and
                                    the bed loses every member: 0.008 ..
                                    0.025 deg off the layer run
M9     linear_in_depth_exact 20,    ``uv_bracket`` reads the nearest
       profile 4, trajectory 4,     centre (the layer path's values) ->
       sigma_profile 4,             1.1 .. 3.6 % off, ratios 1.95 .. 2.00
       sigma_trajectory 4,          (in ``s``: the layer mode's errors to
       step 212                     the digit)
M10    trench[edge_leastsq] 2       a dry edge neighbour kept as a member
       (free and no slip)           (no ghost) -> 23 % slow; no slip:
                                    0.56 U off the hand fit
M11    trench[leastsq],             a dry element kept in the wet-member
       trench[barycentric-element]  mean -> element barycentric 50 % slow,
                                    the ``leastsq`` trench end 44 %
M12    trench[barycentric-node]     every vertex counted wet -> the shelf
                                    nodes' last value: 21 % slow
M13    step 68 (rk2; stage_s 8)     RK2 stage 2 reads its depth in the
                                    ENTRY column, ``z_mid / H_1`` -> the
                                    stage-2 ``s`` 1.3 % off the hand stage
                                    ``s``, the step 3.5e-4 off (this
                                    module runs RK4 only: green)
M14    sigma_trajectory 4,          ``_sigma_frac_for_uv`` returns metres
       step 261                     -> every read clipped to the bottom
                                    centre: 35.6 km RMS at every
                                    resolution (ratios 0.99), the step 68 %
                                    off; the profile (the interpolators
                                    given ``s``) and the depth-constant
                                    field (``s`` cannot show) green
M15    sigma_depth_constant 9,      the table's points at the upper
       sigma_profile 4,             interfaces ``1 - sigma_k``, not the
       sigma_trajectory 4,          centres -> first order (ratios 1.99 ..
       step 233                     2.00), the step 7.5 % off; the particle
                                    at ``s = 0.02`` now between two points:
                                    1 ulp off the layer run
M16    step 76 (rk2; stage_s 8)     RK2 stage 2 reads the ENTRY ``s`` ->
                                    10 % off the hand stage ``s``, the step
                                    4 % off (this module: green, RK4 only)
M17    sigma_trajectory 4,          RK4 stages 2 .. 4 read the entry depth
       step 76 (rk4; stage_s 8)     in their own column, ``z_0 / H_k`` ->
                                    the stage ``s`` up to 21 % off, the
                                    step 4 % off; the trajectory's
                                    time-step error 64 % of its error
                                    (88.9 of 139.9 m on 10 layers)
M18    step 76 (euler)              Euler hands the depth in metres -> the
                                    bottom centre for every particle, 76 %
                                    off (this module: green, its Euler
                                    cells hold ``s``-independent data)
=====  ===========================  =======================================
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
netCDF4 = pytest.importorskip("netCDF4")

from kelp.core.config import (AppConfig, DataConfig, MeshConfig,  # noqa: E402
                              ModeConfig, NumericsConfig, TimeConfig)
from kelp.core.geometry import DEG2RAD, METERS_PER_DEG_LAT  # noqa: E402
from kelp.core.io import VelocityField  # noqa: E402
from kelp.core.mesh import find_nearest_nodes  # noqa: E402
from kelp.core.particles import ParticleStatus, make_state  # noqa: E402
from kelp.numerics import interpolate as ip  # noqa: E402
from kelp.numerics.vertical import find_layers  # noqa: E402
from kelp.numerics.vertical_sigma import find_layers_sigma  # noqa: E402
from kelp.readers import get_reader  # noqa: E402
from kelp.writers import load_trajectories  # noqa: E402
from tests import _fesom2_synth as synth  # noqa: E402
from tests._bitwise import assert_bitwise  # noqa: E402
from tests.test_integrators_depth_linear_sigma import (  # noqa: E402
    _centres as _sigma_centres)
from tests.test_uv_vertical_elements_run import (  # noqa: E402
    _global_attrs, _load_mesh, _node_beds, _run)

M_DEG = METERS_PER_DEG_LAT
GRID0, STEP = 4.0, 0.25
DT = 86400.0
#: equal element areas (m^2) in the mesh diagnostic
AREA = 7.0e8

#: (layout, uv_interp) of every depth-linear path
CASES = [("node", "barycentric"), ("element", "leastsq"),
         ("element", "edge_leastsq"), ("element", "barycentric")]
CASE_IDS = [f"{lay}-{ui}" for lay, ui in CASES]
INTEGRATORS = ["euler", "rk2", "rk4"]

_INTERP_FN = {"leastsq": ip.interp_leastsq,
              "edge_leastsq": ip.interp_edge_leastsq,
              "barycentric": ip.interp_barycentric}


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _centres(levels):
    z = np.asarray(levels, dtype=np.float64)
    return 0.5 * (z[:-1] + z[1:])


def _hand_last(levels, bed):
    """The last usable layer of a column with bed ``bed``, by the
    geometric rule (the deepest layer whose lower level is at or above the
    bed), computed here and not by the engine, so a table that loses the
    rule cannot hide it."""
    lower = np.asarray(levels, dtype=np.float64)[1:]
    return np.searchsorted(lower, np.asarray(bed, dtype=np.float64),
                           side="right") - 1


def _write_mesh(path, nx, ny, levels, elem_bed, node_bed=None):
    """A grid mesh diagnostic: element beds from ``zbar_e_bottom``, node
    beds ``node_bed`` (default the deepest of each node's elements)."""
    lon, lat, tri = synth.grid_geometry(nx, ny, GRID0, GRID0, STEP)
    n_el = tri.shape[0]
    bed = np.array(np.broadcast_to(np.asarray(elem_bed, dtype=np.float64),
                                   (n_el,)))
    nbed = (_node_beds(tri, bed, lon.size) if node_bed is None
            else np.asarray(node_bed, dtype=np.float64))
    synth.write_meshdiag(
        path, "ugrid", geometry=(lon, lat, tri),
        zbar=-np.asarray(levels, dtype=np.float64), zbar_n_bottom=-nbed,
        zbar_e_bottom=bed, elem_area=np.full(n_el, AREA))
    return Path(path)


def _elem_centres(nx, ny):
    """The element centres (vertex means) of the grid, before any file is
    written."""
    lon, lat, tri = synth.grid_geometry(nx, ny, GRID0, GRID0, STEP)
    return lon[tri - 1].mean(axis=1), lat[tri - 1].mean(axis=1)


def _cols(mesh, layout):
    """Column coordinates: element centres or nodes."""
    if layout == "node":
        return mesh.nodes_lon, mesh.nodes_lat
    return mesh.elem_center_lon, mesh.elem_center_lat


def _col_beds(mesh, layout, elem_bed):
    if layout == "node":
        return np.asarray(mesh.depth, dtype=np.float64)
    return np.array(np.broadcast_to(np.asarray(elem_bed, dtype=np.float64),
                                    (mesh.n_elements,)))


def _write_data(data, mesh, layout, u, v, *, w=0.0):
    """u, v ``(records, layers, columns)`` on ``layout``; ``w`` (file
    convention, positive up) uniform on the node levels; ssh zero."""
    data = Path(data)
    data.mkdir(parents=True, exist_ok=True)
    n_rec, n_lay, n_col = u.shape
    for var, arr in (("u", u), ("v", v)):
        synth.write_multiyear(
            data, var, {1958: n_rec}, node_layout=(layout == "node"),
            vertical="nz1", values=lambda year, k, arr=arr: arr[:, k, :],
            continuous_time=True, n_space=n_col, n_levels=n_lay + 1,
            dtype=np.float64)
    synth.write_multiyear(data, "w", {1958: n_rec}, vertical="nz",
                          values=lambda year, k: float(w),
                          continuous_time=True, n_space=mesh.n_nodes,
                          n_levels=n_lay + 1, dtype=np.float64)
    synth.write_multiyear(data, "ssh", {1958: n_rec}, vertical=None,
                          values=lambda year, k: 0.0, continuous_time=True,
                          n_space=mesh.n_nodes, dtype=np.float64)
    return data


def _cfg(mesh_path, data, seeds, *, out, uv_interp, integrator="rk4",
         uv_vertical="linear", n_steps=4, dt=DT, **num):
    """A run_tracking config; ``num`` goes to NumericsConfig (``w_interp``
    defaults to ``"nearest"``, which lets Euler with ``leastsq`` take the
    fused kernel)."""
    work = Path(data).parent
    seeds_path = work / f"seeds_{out}.dat"
    seeds_path.write_text("".join(f"{lo:.17g} {la:.17g} {z:.17g}\n"
                                  for lo, la, z in seeds))
    num.setdefault("w_interp", "nearest")
    return AppConfig(
        mode=ModeConfig(Fdrift2D=False, Fdrift3D=True, var_u="u", var_v="v",
                        var_w="w", var_eta="ssh", uv_vertical=uv_vertical),
        numerics=NumericsConfig(dt=dt, integrator=integrator,
                                uv_interp=uv_interp, **num),
        mesh=MeshConfig(type="fesom2", meshpath=str(mesh_path)),
        data=DataConfig(ncinput=str(data), inicloud=str(seeds_path),
                        output_path=str(work / out),
                        time_averaging="instant"),
        time=TimeConfig(start_time_idx=0, n_steps=n_steps),
    )


def _traj(cfg):
    """Run ``cfg``; its trajectories, every particle ALIVE throughout."""
    _run(cfg)
    t = load_trajectories(cfg.data.output_path)
    assert (t["status"] == ParticleStatus.ALIVE).all(), t["status"]
    return t


def _east_north(lon, lat, lon0, lat0):
    """Displacement (m) east and north of ``(lon0, lat0)``, on the metric
    of the integrators at the end latitude."""
    return ((lon - lon0) * np.cos(lat * DEG2RAD) * M_DEG,
            (lat - lat0) * M_DEG)


def _wrap_quantum(lon):
    """One rounding of the per-step longitude wrap
    ``((lon + 180) % 360) - 180``: the spacing of ``lon + 180``
    (about 3.1e-9 m at 5E)."""
    return np.spacing(np.abs(lon) + 180.0)


# ---------------------------------------------------------------------------
# A depth-constant field: the layer answer
# ---------------------------------------------------------------------------

LEVELS_C = (0.0, 10.0, 30.0, 60.0, 100.0, 150.0)
#: 17 x 17 nodes (4E..8E, 4N..8N), four quadrants of element beds split at
#: 6E and 6N: SW 150 m, SE 100 m, NW 60 m (a level), NE 50 m (partial: the
#: layer 30..60 m is not usable, the column is wet down to 50 m)
NC = 17
N_STEPS_C = 4

#: (lon, lat, z, kind) — "data": every member reads a stored value (above
#: the first centre, at a centre, below the last centre; at its members'
#: bed; between a partial bed's last centre and the bed): bit for bit the
#: layer run; "blend": between two centres, within a few ulp.  About four
#: cells from the quadrant borders (the runs move them one cell at most),
#: so every stencil member of every stage lies in the particle's quadrant.
CONST_SEEDS = [
    (4.83, 4.91, 3.0, "data"), (5.08, 4.78, 20.0, "data"),
    (4.92, 5.12, 33.0, "blend"), (5.17, 5.06, 140.0, "data"),
    (6.88, 4.86, 12.5, "blend"), (7.09, 5.07, 70.0, "blend"),
    (4.97, 6.88, 60.0, "data"),     # z == the bed of every member
    (6.93, 6.90, 40.0, "data"),     # partial bed 50 m, last centre 20 m
]

#: ulp of max|c| allowed between two velocities of the same position: the
#: plane fits 8, element barycentric 4, node barycentric 2 (``max|c|`` of
#: the stencil's members; every constant lies in one binade, so its ulp is
#: that of any stencil)
VEL_ULP = {"leastsq": 8, "edge_leastsq": 8, "barycentric-element": 4,
           "barycentric-node": 2}
#: the binade of every constant of the depth-constant field (m/s)
C_BINADE = (2.0 ** -5, 2.0 ** -4)


def _quadrant_beds(lo, la):
    """Element beds by the quadrant of the element centre."""
    return np.where(la < 6.0, np.where(lo < 6.0, 150.0, 100.0),
                    np.where(lo < 6.0, 60.0, 50.0))


def _const_cols(n, n_rec):
    """Per-column constants ``(records, columns)`` of the depth-constant
    fields: varying from column to column and in time, within
    ``C_BINADE`` (0.032 .. 0.062 m/s over six records)."""
    j = np.arange(n, dtype=np.float64)
    r = np.arange(n_rec, dtype=np.float64)[:, None]
    cu = 0.047 + 0.01 * np.sin(1.7 * j + 0.3)
    cv = 0.047 + 0.01 * np.cos(1.3 * j)
    du = 0.001 * np.cos(0.9 * j)
    dv = -0.001 * np.sin(0.7 * j)
    return cu + r * du, cv + r * dv


def _const_field(mesh, layout, bed, n_rec):
    """Per-column constants (:func:`_const_cols`), the same at every usable
    point of a column, NaN below it."""
    n = mesh.n_nodes if layout == "node" else mesh.n_elements
    cu, cv = _const_cols(n, n_rec)
    last = _hand_last(LEVELS_C, bed)
    n_lay = len(LEVELS_C) - 1
    usable = np.arange(n_lay)[:, None] <= last[None, :]
    u = np.where(usable[None], cu[:, None, :], np.nan)
    v = np.where(usable[None], cv[:, None, :], np.nan)
    return u, v


@pytest.mark.parametrize("integrator", INTEGRATORS)
@pytest.mark.parametrize("layout,uv_interp", CASES, ids=CASE_IDS)
def test_depth_constant_same_as_layer(tmp_path, layout, uv_interp,
                                      integrator):
    """u, v independent of depth, deep columns (every member wet at every
    stage depth; the particle at its members' bed is wet), NaN below each
    column's last usable point: the linear run equals the layer run — bit
    for bit for the particles whose members all read stored values, and
    elsewhere every position within 4 ulp (longitude: within one rounding
    of the per-step wrap, which is coarser); every step starts at the same
    position in both runs, and every velocity is within the interpolator's
    ulp bound of the stencil's ``max|c|`` (the constants lie in one binade,
    so that ulp is the field's).

    Measured (2026-10-20, b03afac with these constants): every position of
    every run bit for bit the layer run's; up to 4 of the 64 velocities of
    a run differ, by at most 3 ulp of ``max|c|`` (``leastsq``), 2
    (``edge_leastsq``), 2 (element ``barycentric``); node ``barycentric``
    none."""
    elem_bed = _quadrant_beds(*_elem_centres(NC, NC))
    mesh_path = _write_mesh(tmp_path / "fesom.mesh.diag.nc", NC, NC,
                            LEVELS_C, elem_bed)
    mesh = _load_mesh(mesh_path)
    assert np.array_equal(mesh.metadata["elem_bed"], elem_bed)
    bed = _col_beds(mesh, layout, elem_bed)
    u, v = _const_field(mesh, layout, bed, N_STEPS_C + 2)
    data = _write_data(tmp_path / "data", mesh, layout, u, v)
    seeds = [s[:3] for s in CONST_SEEDS]
    runs = {}
    for mode in ("linear", "layer"):
        cfg = _cfg(mesh_path, data, seeds, out=f"{mode}.nc",
                   uv_interp=uv_interp, integrator=integrator,
                   uv_vertical=mode, n_steps=N_STEPS_C)
        runs[mode] = _traj(cfg)
    lin, lay = runs["linear"], runs["layer"]
    assert _global_attrs(Path(data).parent / "linear.nc")[
        "uv_vertical"] == "linear"
    exact = np.array([s[3] == "data" for s in CONST_SEEDS])
    for k in ("lon", "lat", "z", "vel_u", "vel_v"):
        assert_bitwise(lin[k][exact], lay[k][exact], k)
    for k in ("lat", "z"):
        assert np.all(np.abs(lin[k] - lay[k])
                      <= 4 * np.spacing(np.abs(lay[k]))), k
    lon_tol = np.maximum(4 * np.spacing(np.abs(lay["lon"])),
                         _wrap_quantum(lay["lon"]))
    assert np.all(np.abs(lin["lon"] - lay["lon"]) <= lon_tol), (
        np.abs(lin["lon"] - lay["lon"]) / _wrap_quantum(lay["lon"]))
    # velocities at the step starts, where both runs stand at the same place
    # (record 0: the release) — at every step: the velocities differ by far
    # less than a position ulp moves in a step
    lon0 = np.array([s[0] for s in seeds])[:, None]
    lat0 = np.array([s[1] for s in seeds])[:, None]
    start_l = np.concatenate([lon0, lin["lon"][:, :-1]], axis=1)
    start_c = np.concatenate([lon0, lay["lon"][:, :-1]], axis=1)
    same = ((start_l == start_c)
            & (np.concatenate([lat0, lin["lat"][:, :-1]], axis=1)
               == np.concatenate([lat0, lay["lat"][:, :-1]], axis=1)))
    assert same.all(), np.argwhere(~same)
    key = (uv_interp if uv_interp != "barycentric"
           else f"barycentric-{layout}")
    # every constant in one binade: the ulp of any stencil's max|c|
    ulp = np.spacing(C_BINADE[0])
    for c in (u, v):
        mag = np.abs(c[np.isfinite(c)])
        assert C_BINADE[0] <= mag.min() and mag.max() < C_BINADE[1]
        assert np.spacing(mag.min()) == np.spacing(mag.max()) == ulp
    tol = VEL_ULP[key] * ulp
    for k in ("vel_u", "vel_v"):
        gap = np.abs(lin[k] - lay[k])
        assert np.all(gap <= tol), (k, gap.max() / ulp)


# ---------------------------------------------------------------------------
# A field linear in depth: exact
# ---------------------------------------------------------------------------

LEVELS_L = (0.0, 10.0, 30.0, 60.0, 100.0, 150.0)
P_L = _centres(LEVELS_L)                     # 5, 20, 45, 80, 125
#: 13 x 13 nodes (4E..7E, 4N..7N), every bed 150 m
NL = 13
LON_C = LAT_C = 5.5
#: u = U0 + GY (lat - LAT_C) + B z (m/s, m/s per degree, m/s per metre)
U0, GY, B = 0.06, 0.02, 2.0e-4
N_STEPS_L = 4
#: depths off the centres and the levels, between the first and the last
#: centre (no clipping)
LIN_SEEDS = [(5.10, 5.20, 8.0), (5.25, 5.80, 14.0), (5.45, 5.35, 26.0),
             (5.60, 5.65, 37.0), (5.75, 5.25, 52.0), (5.90, 5.75, 66.0),
             (5.20, 5.50, 91.0), (5.35, 5.40, 113.0)]
#: uniform W (positive down): the particles sink 34.56 m in 4 days and
#: stay between the centres 5 and 125 m
W_DOWN = 1.0e-4
LIN_W_SEEDS = [s for s in LIN_SEEDS if s[2] + 4 * DT * W_DOWN < P_L[-1]]


def _layer_centre(z):
    lev = np.asarray(LEVELS_L)
    k = np.searchsorted(lev, z, side="right") - 1
    return P_L[k]


LIN_CELLS = ([(lay, ui, integ, 0.0) for lay, ui in CASES
              for integ in INTEGRATORS]
             + [(lay, ui, integ, W_DOWN) for lay, ui in CASES
                for integ in ("rk2", "rk4")])


@pytest.mark.parametrize(
    "layout,uv_interp,integrator,w_down", LIN_CELLS,
    ids=[f"{lay}-{ui}-{integ}-{'w' if w else 'w0'}"
         for lay, ui, integ, w in LIN_CELLS])
def test_linear_in_depth_exact(tmp_path, layout, uv_interp, integrator,
                               w_down):
    """``u = U0 + GY (lat - LAT_C) + B z`` (every column, every point),
    ``v = 0``: the latitude stays fixed and the linear run moves east by
    the closed form ``x = u(lat0, z) T`` to 1e-9 relative; the layer run
    by ``u(lat0, c_k) T``, its error ``B (c_k - z) T`` to 1e-9 relative
    (``c_k`` the centre of the particle's layer).  With a uniform W the
    particles cross layers and ``x = u(lat0, z0) T + B W T^2 / 2``, which
    RK2 (stage 2 at the midpoint depth) and RK4 integrate exactly.

    Measured (2026-10-20, 204a8b4): linear mode within 2.3e-13 of the
    closed form (6.1e-14 with W), the layer error within 7.4e-12 of
    ``B (c_k - z) T`` (207 .. 968 m), in every cell."""
    mesh_path = _write_mesh(tmp_path / "fesom.mesh.diag.nc", NL, NL,
                            LEVELS_L, LEVELS_L[-1])
    mesh = _load_mesh(mesh_path)
    _, y = _cols(mesh, layout)
    col = U0 + GY * (y - LAT_C)
    n_rec = N_STEPS_L + 2
    u = np.broadcast_to(col[None, None, :] + B * P_L[None, :, None],
                        (n_rec, P_L.size, col.size)).copy()
    data = _write_data(tmp_path / "data", mesh, layout, u, np.zeros_like(u),
                       w=-w_down)
    seeds = LIN_W_SEEDS if w_down else LIN_SEEDS
    lon0, lat0, z0 = (np.array(c) for c in zip(*seeds))
    T = N_STEPS_L * DT
    t = _traj(_cfg(mesh_path, data, seeds, out="lin.nc",
                   uv_interp=uv_interp, integrator=integrator,
                   n_steps=N_STEPS_L))
    assert_bitwise(t["lat"][:, -1], lat0, "lat")
    np.testing.assert_allclose(t["z"][:, -1], z0 + T * w_down, rtol=1e-12)
    x, _ = _east_north(t["lon"][:, -1], t["lat"][:, -1], lon0, lat0)
    u_start = U0 + GY * (lat0 - LAT_C) + B * z0
    x_exact = u_start * T + 0.5 * B * w_down * T * T
    assert np.all(np.abs(x - x_exact) <= 1e-9 * np.abs(x_exact)), (
        np.abs(x / x_exact - 1.0).max())
    if w_down:
        return
    t_l = _traj(_cfg(mesh_path, data, seeds, out="layer.nc",
                     uv_interp=uv_interp, integrator=integrator,
                     uv_vertical="layer", n_steps=N_STEPS_L))
    x_l, _ = _east_north(t_l["lon"][:, -1], t_l["lat"][:, -1], lon0, lat0)
    err = B * (_layer_centre(z0) - z0) * T
    assert np.all(np.abs(err) > 50.0)                   # metres: visible
    assert np.all(np.abs((x_l - x_exact) - err) <= 1e-9 * np.abs(err)), (
        np.abs((x_l - x_exact) / err - 1.0).max())


# ---------------------------------------------------------------------------
# A smooth profile: the order of the reconstruction
# ---------------------------------------------------------------------------

H_PROF = 400.0
N_LAYERS = (10, 20, 40)
#: horizontal factors G_u = AU + GXU (lon - LON_C) + GYU (lat - LAT_C), G_v
#: likewise (m/s, m/s per degree), on a 9 x 9 grid (4E..6E) for the profile
AU, GXU, GYU = 0.08, 0.01, 0.03
AV, GXV, GYV = 0.03, -0.02, 0.005
LON_P, LAT_P = 4.97, 5.06
LON_CP = LAT_CP = 5.0


def _g(lon, lat):
    return (AU + GXU * (lon - LON_CP) + GYU * (lat - LAT_CP),
            AV + GXV * (lon - LON_CP) + GYV * (lat - LAT_CP))


def _profile(z):
    return np.cos(np.pi * np.asarray(z) / H_PROF)


def _profile_field(mesh, layout, levels, placement):
    x, y = _cols(mesh, layout)
    gu, gv = _g(x, y)
    f = _profile(_centres(levels))[:, None]
    return VelocityField(
        u=f * gu[None, :], v=f * gv[None, :], eta=np.zeros(mesh.n_nodes),
        w=np.zeros((len(levels), mesh.n_nodes)), uv_layout=layout,
        uv_vertical=placement)


@pytest.mark.parametrize("layout,uv_interp", CASES, ids=CASE_IDS)
def test_profile_reconstruction_order(tmp_path, layout, uv_interp):
    """``u = G(x) cos(pi z / H)`` sampled at the layer centres of 10, 20
    and 40 equal layers, read through the public interpolator at one point
    at 1601 depths from the surface to the bed: the maximum error of linear
    mode falls by >= 3.5 per halving of the layers (second order), of the
    layer mode by 1.6 .. 2.4 (first order); finer is closer in both.

    Measured (2026-10-20, 204a8b4; the same for the four interpolators, which
    are exact in the horizontal here): linear 1.003e-3, 2.512e-4, 6.283e-5
    m/s (ratios 3.994, 3.998), layer 1.275e-2, 6.394e-3, 3.200e-3 m/s
    (ratios 1.994, 1.998)."""
    z = np.linspace(0.0, H_PROF, 1601)
    gu, gv = _g(LON_P, LAT_P)
    errs = {"layer_linear": [], "layer": []}
    for n in N_LAYERS:
        levels = np.linspace(0.0, H_PROF, n + 1)
        mesh = _load_mesh(_write_mesh(tmp_path / f"n{n}.nc", 9, 9, levels,
                                      H_PROF))
        st = make_state(np.full(z.size, LON_P), np.full(z.size, LAT_P),
                        z.copy())
        nn = find_nearest_nodes(mesh, st.lon, st.lat)
        for placement in errs:
            vel = _profile_field(mesh, layout, levels, placement)
            layer = (find_layers(mesh.z_levels, st.z, nn, mesh.depth,
                                 st.alive_mask)
                     if placement == "layer" else 0)
            U, V, ok = _INTERP_FN[uv_interp](st, mesh, vel, nn, 1.0, layer)
            assert ok.all()
            f = _profile(z)
            errs[placement].append(max(np.abs(U - gu * f).max(),
                                       np.abs(V - gv * f).max()))
    lin, lay = np.array(errs["layer_linear"]), np.array(errs["layer"])
    r_lin, r_lay = lin[:-1] / lin[1:], lay[:-1] / lay[1:]
    assert np.all(r_lin >= 3.5), (lin, r_lin)
    assert np.all((r_lay >= 1.6) & (r_lay <= 2.4)), (lay, r_lay)


# ---------------------------------------------------------------------------
# Trajectories: the order of the run
# ---------------------------------------------------------------------------

#: 13 x 13 nodes (4E..7E): the particles start at the centre (5.5E, 5.5N)
#: and stay inside the interior triangles, where every interpolator
#: reproduces a field linear in the horizontal exactly
TRAJ_START = (5.5, 5.5)
N_DEPTHS = 50
N_STEPS_T = 4


def _traj_g(lon, lat):
    """The horizontal factor of the trajectory field (m/s): a shear in
    each component, so the paths curve."""
    return (0.08 + 0.04 * (lat - TRAJ_START[1]),
            0.03 - 0.03 * (lon - TRAJ_START[0]))


def _reference(f, T, n_sub=2000):
    """End positions of ``d lon/dt = f G_u / (cos(lat) M)``, ``d lat/dt =
    f G_v / M`` from the start, for each depth factor ``f``: classical
    RK4 with ``n_sub`` sub-steps (its own error far below the ones
    measured)."""
    lon = np.full(f.size, TRAJ_START[0])
    lat = np.full(f.size, TRAJ_START[1])
    h = T / n_sub

    def rhs(x, y):
        gu, gv = _traj_g(x, y)
        return (f * gu / (np.cos(y * DEG2RAD) * M_DEG), f * gv / M_DEG)
    for _ in range(n_sub):
        k1 = rhs(lon, lat)
        k2 = rhs(lon + 0.5 * h * k1[0], lat + 0.5 * h * k1[1])
        k3 = rhs(lon + 0.5 * h * k2[0], lat + 0.5 * h * k2[1])
        k4 = rhs(lon + h * k3[0], lat + h * k3[1])
        lon = lon + h / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        lat = lat + h / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return lon, lat


def _rms_gap(lon_a, lat_a, lon_b, lat_b):
    dx, dy = _east_north(lon_a, lat_a, lon_b, lat_b)
    return float(np.sqrt(np.mean(dx * dx + dy * dy)))


@pytest.mark.parametrize("layout,uv_interp", CASES, ids=CASE_IDS)
def test_trajectory_convergence(tmp_path, layout, uv_interp):
    """``u = G(x) cos(pi z / H)``, ``w = 0``, 50 particles at equally
    spaced depths (the column without its ends), RK4 over four days on 10,
    20 and 40 layers: the RMS distance of the end positions from the
    reference solution falls by >= 3.5 per halving of the layers in linear
    mode and by 1.6 .. 2.4 in layer mode; the time-step error (``dt``
    against ``dt / 2``) is below 1 % of either mode's error.

    Measured (2026-10-20, 204a8b4; the same for the four interpolators):
    linear 182.5, 45.66, 11.42 m (ratios 3.997, 3.998), layer 1860, 955.3,
    476.7 m (ratios 1.947, 2.004); the time-step error at most 5.2e-6 of
    the linear error, 1.3e-7 of the layer error; displacements up to 29 km;
    the reference's own error (2000 against 4000 sub-steps) 3e-9 m."""
    z = np.linspace(0.0, H_PROF, N_DEPTHS + 2)[1:-1]
    seeds = [(TRAJ_START[0], TRAJ_START[1], float(zi)) for zi in z]
    T = N_STEPS_T * DT
    ref_lon, ref_lat = _reference(_profile(z), T)
    errs = {"linear": [], "layer": []}
    for n in N_LAYERS:
        levels = np.linspace(0.0, H_PROF, n + 1)
        work = tmp_path / f"n{n}"
        work.mkdir()
        mesh_path = _write_mesh(work / "fesom.mesh.diag.nc", NL, NL, levels,
                                H_PROF)
        mesh = _load_mesh(mesh_path)
        x, y = _cols(mesh, layout)
        gu, gv = _traj_g(x, y)
        f = _profile(_centres(levels))[None, :, None]
        n_rec = N_STEPS_T + 2
        u = np.broadcast_to(f * gu[None, None, :], (n_rec, n, gu.size))
        v = np.broadcast_to(f * gv[None, None, :], (n_rec, n, gv.size))
        data = _write_data(work / "data", mesh, layout, u.copy(), v.copy())
        for mode in errs:
            ends = []
            for k, dt in enumerate((DT, DT / 2)):
                # dt / 2 sub-steps the daily records, which needs
                # temporal_interp (discrete mode refuses a dt other than
                # the record step); both runs then take the time-aware
                # RK4, on this steady field the classical one
                t = _traj(_cfg(mesh_path, data, seeds, out=f"{mode}{k}.nc",
                               uv_interp=uv_interp, uv_vertical=mode,
                               n_steps=N_STEPS_T * (k + 1), dt=dt,
                               temporal_interp=True))
                ends.append((t["lon"][:, -1], t["lat"][:, -1]))
            err = _rms_gap(*ends[0], ref_lon, ref_lat)
            err_t = _rms_gap(*ends[0], *ends[1])
            assert err_t < 0.01 * err, (mode, n, err_t, err)
            errs[mode].append(err)
    lin, lay = np.array(errs["linear"]), np.array(errs["layer"])
    r_lin, r_lay = lin[:-1] / lin[1:], lay[:-1] / lay[1:]
    assert np.all(r_lin >= 3.5), (lin, r_lin)
    assert np.all((r_lay >= 1.6) & (r_lay <= 2.4)), (lay, r_lay)


# ---------------------------------------------------------------------------
# A trench: the walls keep the speed
# ---------------------------------------------------------------------------

LEVELS_T = (0.0, 10.0, 30.0, 60.0, 100.0)
#: 17 x 9 nodes (4E..8E, 4N..6N); the shelf bed 30 m, the trench 100 m
NX_T, NY_T = 17, 9
SHELF, DEEP = 30.0, 100.0
#: element layout: the trench is the row of cells between 5.00N and
#: 5.25N, from 4.5E to 7.25E (cells 2..12 of row 4, both triangles of each:
#: the only wall edge of each trench triangle runs east-west)
TRENCH_ROW, TRENCH_I0, TRENCH_I1 = 4, 2, 12
#: node layout: the line of deep nodes along 5.00N from 4.5E to 7.0E
LINE_ROW = 4
#: the flow: U along the trench (east) at every depth, the shelf's own
#: speed in its water (layers 0 and 1), zeros below the shelf bed as
#: FESOM2 writes them
U_T, U_SHELF = 0.05, 0.02
N_STEPS_T2 = 4
#: below the shelf bed, away from the trench ends during the run (the
#: east end 1.2 deg ahead); latitude offsets 0.2 .. 0.35 cell from the
#: nearest line of nodes
TRENCH_SEEDS = {
    "element": [(5.00, 5.05, 35.0), (5.30, 5.075, 50.0),
                (5.60, 5.175, 72.0), (5.90, 5.20, 95.0)],
    "node": [(5.00, 5.05, 35.0), (5.30, 5.0875, 50.0),
             (5.60, 4.95, 72.0), (5.90, 4.9125, 95.0)],
}
#: at the east end for the whole run: the nearest node at the trench's end
#: column (fewer than three wet ring elements: their area mean) for
#: leastsq, the containing triangle's wet vertices for element
#: barycentric; the node line's last deep node for node barycentric
TRENCH_END_SEEDS = {
    "leastsq": [(7.15, 5.075, 50.0)],
    "barycentric-element": [(7.15, 5.075, 50.0)],
    "barycentric-node": [(6.90, 5.05, 50.0)],
}
#: no-slip: the element seeds with the first off the node column at 5.00E
#: (the fit of ``edge_leastsq`` jumps across an element edge, so a
#: particle on one has two answers)
TRENCH_NOSLIP_SEEDS = [(5.02, 5.05, 35.0)] + TRENCH_SEEDS["element"][1:]
TRENCH_CASES = [("element", "leastsq", "freeslip"),
                ("element", "edge_leastsq", "freeslip"),
                ("element", "edge_leastsq", "noslip"),
                ("element", "barycentric", "freeslip"),
                ("node", "barycentric", "freeslip")]
TRENCH_IDS = ["leastsq", "edge_leastsq-freeslip", "edge_leastsq-noslip",
              "barycentric-element", "barycentric-node"]


def _trench_beds():
    """Element beds: the trench cells deep, the rest shelf (two triangles
    per cell, cells row by row, as :func:`synth.grid_geometry` orders
    them)."""
    n_cell_x = NX_T - 1
    cell = np.arange(2 * n_cell_x * (NY_T - 1)) // 2
    ci, cj = cell % n_cell_x, cell // n_cell_x
    deep = (cj == TRENCH_ROW) & (ci >= TRENCH_I0) & (ci <= TRENCH_I1)
    return np.where(deep, DEEP, SHELF)


def _line_beds():
    """Node beds: the line of deep nodes, the rest shelf."""
    i = np.arange(NX_T * NY_T) % NX_T
    j = np.arange(NX_T * NY_T) // NX_T
    deep = (j == LINE_ROW) & (i >= TRENCH_I0) & (i <= TRENCH_I1)
    return np.where(deep, DEEP, SHELF)


def _noslip_hand(mesh, lon, lat, z):
    """The ``edge_leastsq`` no-slip value of the trench flow at ``(lon,
    lat)``, depth ``z`` below the shelf bed: the plane fitted by least
    squares through the containing element's centre (``U_T``), the centre
    of each edge neighbour wet at ``z`` (``U_T``, a trench element) and,
    for each dry one (the shelf), the ghost — the own centre mirrored
    across the shared edge — at ``-U_T``, read at the particle.  Metres
    east and north of the particle, as the fit's; the element centres are
    the reader's; the edge neighbours are found here from the vertices."""
    tri = np.asarray(mesh.elem_nodes)[:, :3]
    bed = mesh.metadata["elem_bed"]
    x, y = mesh.nodes_lon[tri], mesh.nodes_lat[tri]
    det = ((x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0])
           - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0]))
    l1 = ((lon - x[:, 0]) * (y[:, 2] - y[:, 0])
          - (x[:, 2] - x[:, 0]) * (lat - y[:, 0])) / det
    l2 = ((x[:, 1] - x[:, 0]) * (lat - y[:, 0])
          - (lon - x[:, 0]) * (y[:, 1] - y[:, 0])) / det
    inside = np.flatnonzero(np.minimum(np.minimum(l1, l2), 1.0 - l1 - l2)
                            > 1e-6)
    assert inside.size == 1, (lon, lat, inside)    # never on an edge
    e = int(inside[0])
    assert bed[e] == DEEP

    def metric(lo, la):
        return ((lo - lon) * np.cos(lat * DEG2RAD) * M_DEG,
                (la - lat) * M_DEG)
    cx, cy = metric(mesh.elem_center_lon[e], mesh.elem_center_lat[e])
    pts = [(cx, cy, U_T)]
    for k in range(3):
        a, b = tri[e, k], tri[e, (k + 1) % 3]
        nb = [f for f in np.flatnonzero((tri == a).any(axis=1)
                                        & (tri == b).any(axis=1)) if f != e]
        assert len(nb) == 1
        if z <= bed[nb[0]]:
            assert bed[nb[0]] == DEEP
            pts.append(metric(mesh.elem_center_lon[nb[0]],
                              mesh.elem_center_lat[nb[0]]) + (U_T,))
            continue
        ax, ay = metric(mesh.nodes_lon[a], mesh.nodes_lat[a])
        bx, by = metric(mesh.nodes_lon[b], mesh.nodes_lat[b])
        tx, ty = bx - ax, by - ay
        s = ((cx - ax) * tx + (cy - ay) * ty) / (tx * tx + ty * ty)
        pts.append((2.0 * (ax + s * tx) - cx, 2.0 * (ay + s * ty) - cy,
                    -U_T))
    pts = np.array(pts)
    scale = STEP * M_DEG                     # a well-scaled design matrix
    design = np.column_stack([np.ones(len(pts)), pts[:, 0] / scale,
                              pts[:, 1] / scale])
    return np.linalg.lstsq(design, pts[:, 2], rcond=None)[0][0]


@pytest.mark.parametrize("layout,uv_interp,bc", TRENCH_CASES,
                         ids=TRENCH_IDS)
def test_trench_wall_keeps_speed(tmp_path, layout, uv_interp, bc):
    """A uniform eastward flow ``U`` along a straight trench, the shelf
    beside it flowing at its own speed in its own water: particles below
    the shelf bed keep ``U`` in linear mode to 1e-12 relative (the shelf
    is a wall at their depth: ``leastsq`` and element ``barycentric``
    leave its elements out, ``edge_leastsq`` free-slip mirrors the own
    value across each wall edge, which runs along the flow, node
    ``barycentric`` leaves the shelf nodes out); in layer mode they read
    the shelf's zeros below its bed and are slower.  At the east end of
    the trench (``leastsq``: fewer than three wet ring elements, their
    area mean; ``barycentric``: the wet vertices) the speed is ``U`` too.
    (``edge_leastsq``'s end wall, across the flow, brakes it as a coast
    does.)  ``edge_leastsq_bc: noslip`` brakes at every wall by design: the
    recorded ``vel_u`` is the hand value of the no-slip fit
    (:func:`_noslip_hand`: the own value negated across the shelf edge) at
    each step's start to 1e-12 of ``U``, far from ``U`` at every start,
    and every particle covers well under ``U T``; no layer run (the layer
    fit reads the shelf's zeros, the linear one the negated ghost).  The
    plane is not bounded by ``U``: in a trench triangle whose wall edge is
    the far one it extrapolates above ``U`` near the opposite side.

    Measured (2026-10-20, 204a8b4): linear ``vel_u`` within 1.6e-15 of
    ``U``; the layer run's distance over the linear one 0.70 .. 0.80
    (``leastsq``), 0.77 .. 0.83 (``edge_leastsq``), 0.50 (element
    ``barycentric``), 0.65 .. 0.80 (node ``barycentric``).  No slip
    (2026-10-20, 4a5fc2a): ``vel_u`` within 1.1e-14 of ``U`` from the hand
    fit, 0.27 .. 1.42 ``U`` (above ``U`` at two release points in a
    triangle walled to the north, 2 of 16 step starts); the distances
    0.28 .. 0.63 ``U T``."""
    path = tmp_path / "fesom.mesh.diag.nc"
    if layout == "node":
        mesh_path = _write_mesh(path, NX_T, NY_T, LEVELS_T, SHELF,
                                node_bed=_line_beds())
    else:
        mesh_path = _write_mesh(path, NX_T, NY_T, LEVELS_T, _trench_beds())
    mesh = _load_mesh(mesh_path)
    bed = (np.asarray(mesh.depth) if layout == "node"
           else mesh.metadata["elem_bed"])
    deep = bed == DEEP
    assert deep.sum() == (11 if layout == "node" else 22)
    n_lay = len(LEVELS_T) - 1
    last = _hand_last(LEVELS_T, bed)
    usable = np.arange(n_lay)[:, None] <= last[None, :]
    col = np.where(usable, U_SHELF, 0.0)
    col[:, deep] = U_T
    n_rec = N_STEPS_T2 + 2
    u = np.broadcast_to(col[None], (n_rec,) + col.shape).copy()
    data = _write_data(tmp_path / "data", mesh, layout, u, np.zeros_like(u))
    key = uv_interp if uv_interp != "barycentric" else f"barycentric-{layout}"
    main = TRENCH_NOSLIP_SEEDS if bc == "noslip" else TRENCH_SEEDS[layout]
    seeds = main + TRENCH_END_SEEDS.get(key, [])
    lon0, lat0, z0 = (np.array(c) for c in zip(*seeds))
    runs = {}
    for mode in ("linear", "layer") if bc == "freeslip" else ("linear",):
        runs[mode] = _traj(_cfg(mesh_path, data, seeds, out=f"{mode}.nc",
                                uv_interp=uv_interp, uv_vertical=mode,
                                n_steps=N_STEPS_T2, edge_leastsq_bc=bc))
    t = runs["linear"]
    lon = np.concatenate([lon0[:, None], t["lon"]], axis=1)
    lat = np.concatenate([lat0[:, None], t["lat"]], axis=1)
    assert_bitwise(t["lat"], np.repeat(lat0[:, None], N_STEPS_T2, axis=1),
                   "lat")
    assert_bitwise(t["z"], np.repeat(z0[:, None], N_STEPS_T2, axis=1), "z")
    # every stage-1 velocity: U, or the no-slip fit at the step's start;
    # v zero
    if bc == "freeslip":
        u_hand = np.full(t["vel_u"].shape, U_T)
    else:
        u_hand = np.array([[_noslip_hand(mesh, lon[i, k], lat0[i], z0[i])
                            for k in range(N_STEPS_T2)]
                           for i in range(len(seeds))])
        # the test can tell the two conditions at every step start
        assert np.all(np.abs(u_hand - U_T) > 0.1 * U_T), u_hand / U_T
    assert np.all(np.abs(t["vel_u"] - u_hand) <= 1e-12 * U_T), (
        np.abs(t["vel_u"] - u_hand).max() / U_T)
    assert np.all(np.abs(t["vel_v"]) <= 1e-12 * U_T)
    dx = (np.diff(lon, axis=1) * np.cos(lat[:, 1:] * DEG2RAD) * M_DEG)
    if bc == "noslip":
        # braked over the run: every particle well short of U T
        assert np.all((dx > 0.0).all(axis=1)
                      & (dx.sum(axis=1) < 0.8 * U_T * N_STEPS_T2 * DT)), (
            dx.sum(axis=1) / (U_T * N_STEPS_T2 * DT))
        return
    # free slip: every step moves U dt east
    quantum = (_wrap_quantum(lon[:, 1:]) * np.cos(lat[:, 1:] * DEG2RAD)
               * M_DEG)
    assert np.all(np.abs(dx - U_T * DT) <= 1e-12 * U_T * DT + quantum), (
        np.abs(dx / (U_T * DT) - 1.0).max())
    # layer mode: slower than U, every main particle
    t_l = runs["layer"]
    n_main = len(main)
    x_lin, _ = _east_north(t["lon"][:n_main, -1], t["lat"][:n_main, -1],
                           lon0[:n_main], lat0[:n_main])
    x_lay, _ = _east_north(t_l["lon"][:n_main, -1], t_l["lat"][:n_main, -1],
                           lon0[:n_main], lat0[:n_main])
    assert np.all(x_lay < 0.9 * x_lin), x_lay / x_lin


# ---------------------------------------------------------------------------
# FESOM-C sigma meshes: the same rule in the sigma fraction s = z / H_p
# ---------------------------------------------------------------------------

# The dia-sigma W of every sigma file is zero: a particle keeps its sigma
# fraction ``s``, and its depth ``z = s H_p`` follows the sloping bed and
# the moving surface (``_sigma_bed``, ``_sigma_eta``).

#: the time axis of the FESOM-C files (records ``DT`` apart from 0)
SIGMA_TIME_UNITS = "seconds since 2000-01-01 00:00:00"


def _sigma_bed(lon, lat):
    """Node beds of the sigma files (m): sloping in lon and lat, 90 .. 510 m
    over 4E..7E, 4N..7N."""
    return 300.0 + 80.0 * (np.asarray(lon) - 5.5) + 60.0 * (np.asarray(lat)
                                                            - 5.5)


def _sigma_eta(t, lon, lat):
    """The surface (m): rising 0.8 m a day, tilted (linear in time, so a
    linear blend of the records is the surface at every time)."""
    return 0.8 * (t / DT) * (1.0 + 0.2 * (np.asarray(lat) - 5.5)) + 0.0 * lon


def _sigma_col(lon, lat, t):
    """The column height ``H = bed + eta`` (no column here is near
    ``Dmin``)."""
    return _sigma_bed(lon, lat) + _sigma_eta(t, lon, lat)


def _write_fesomc(path, nx, ny, sig, uv, *, n_rec):
    """A FESOM-C file in the layout of ``tests/test_fesomc_w_contract`` on
    the grid of :func:`tests._fesom2_synth.grid_geometry` (``nx`` x ``ny``
    nodes from 4E 4N, 0.25 deg; the triangles closed by their first vertex
    in ``nv``): sigma interfaces ``sig`` (1 at the surface, 0 at the bed),
    node beds :func:`_sigma_bed`, equal element areas, ``n_rec`` records
    ``DT`` apart with the surface :func:`_sigma_eta` and a zero dia-sigma
    ``w``.  ``uv(x, y) -> (u, v)``, each ``(n_rec, K-1, x.size)``, gives the
    columns at the element centres (``u``/``v``, ``(time, nele,
    sigmam1)``) and at the nodes (``unode``/``vnode``)."""
    lon, lat, tri = synth.grid_geometry(nx, ny, GRID0, GRID0, STEP)
    lon_e, lat_e = _elem_centres(nx, ny)
    sig = np.asarray(sig, dtype=np.float64)
    t = np.arange(n_rec, dtype=np.float64) * DT
    ue, ve = uv(lon_e, lat_e)
    un, vn = uv(lon, lat)

    def lay(a):
        return np.ascontiguousarray(np.transpose(a, (0, 2, 1)),
                                    dtype=np.float64)
    ds = xr.Dataset(
        {"lon": (["node"], lon), "lat": (["node"], lat),
         "depth": (["node"], _sigma_bed(lon, lat)),
         "nv": (["nele", "four"],
                np.column_stack([tri, tri[:, :1]]).astype(np.int32)),
         "lon_elem": (["nele"], lon_e), "lat_elem": (["nele"], lat_e),
         "elem_area": (["nele"], np.full(tri.shape[0], AREA)),
         "sigma_lev": (["sigma"], sig),
         "u": (["time", "nele", "sigmam1"], lay(ue)),
         "v": (["time", "nele", "sigmam1"], lay(ve)),
         "unode": (["time", "node", "sigmam1"], lay(un)),
         "vnode": (["time", "node", "sigmam1"], lay(vn)),
         "w": (["time", "node", "sigma"], np.zeros((n_rec, lon.size,
                                                    sig.size))),
         "eta": (["time", "node"],
                 np.array([_sigma_eta(ti, lon, lat) for ti in t]))},
        coords={"time": ("time", t, {"units": SIGMA_TIME_UNITS})})
    ds.to_netcdf(path)
    return Path(path)


def _load_fesomc(path):
    with contextlib.redirect_stdout(io.StringIO()):
        mesh = get_reader("fesomc").load_mesh(
            MeshConfig(type="fesomc", meshpath=str(path)))
    assert mesh.vertical_coord == "sigma"
    return mesh


def _cfg_fesomc(path, seeds, *, out, layout, uv_interp, integrator="rk4",
                uv_vertical="linear", n_steps=4, dt=DT, **num):
    """A run_tracking config on the FESOM-C file ``path`` (element u/v
    ``u``/``v`` or node u/v ``unode``/``vnode`` by ``layout``); ``num`` goes
    to NumericsConfig (``w_interp`` defaults to ``"barycentric"``, which
    gives the column height of a plane bed and surface exactly)."""
    work = Path(path).parent
    seeds_path = work / f"seeds_{out}.dat"
    seeds_path.write_text("".join(f"{lo:.17g} {la:.17g} {z:.17g}\n"
                                  for lo, la, z in seeds))
    num.setdefault("w_interp", "barycentric")
    var_u, var_v = ("unode", "vnode") if layout == "node" else ("u", "v")
    return AppConfig(
        mode=ModeConfig(Fdrift2D=False, Fdrift3D=True, var_u=var_u,
                        var_v=var_v, var_w="w", var_eta="eta",
                        uv_vertical=uv_vertical),
        numerics=NumericsConfig(dt=dt, integrator=integrator,
                                uv_interp=uv_interp, **num),
        mesh=MeshConfig(type="fesomc", meshpath=str(path)),
        data=DataConfig(ncinput=str(path), inicloud=str(seeds_path),
                        output_path=str(work / out),
                        time_averaging="instant"),
        time=TimeConfig(start_time_idx=0, n_steps=n_steps),
    )


def _sigma_seeds(lon, lat, s):
    """``(lon, lat, z)`` at the sigma fractions ``s`` of the columns at
    the start (t = 0)."""
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    z = np.asarray(s, float) * _sigma_col(lon, lat, 0.0)
    return [(float(a), float(b), float(c)) for a, b, c in zip(lon, lat, z)]


def _sigma_end_s(t, n_steps, dt):
    """The sigma fraction of every particle after each step, ``z / H`` in
    the column at its position and time."""
    times = dt * np.arange(1, n_steps + 1, dtype=np.float64)[None, :]
    return t["z"] / _sigma_col(t["lon"], t["lat"], times)


# ---------------------------------------------------------------------------
# Sigma: a depth-constant field gives the layer answer
# ---------------------------------------------------------------------------

#: uniform sigma, 10 layers: centres 0.05 .. 0.95
SIG_C = np.linspace(1.0, 0.0, 11)
#: (lon, lat, s, kind) on the 13 x 13 grid (4E..7E): "data" = beyond the
#: outer centres (``s`` kept along the path, W = 0): every member reads a
#: stored value, bit for bit the layer run; "blend" = between two centres,
#: within a few ulp
CONST_SIGMA_SEEDS = [
    (5.10, 5.20, 0.0, "data"),        # at the surface
    (5.30, 5.70, 0.02, "data"),
    (5.60, 5.35, 0.33, "blend"),
    (5.85, 5.60, 0.5, "blend"),
    (5.40, 5.90, 0.61, "blend"),
    (5.70, 5.10, 0.97, "data"),
    (5.20, 5.45, 0.995, "data"),
]


@pytest.mark.parametrize("integrator", INTEGRATORS)
@pytest.mark.parametrize("layout,uv_interp", CASES, ids=CASE_IDS)
def test_sigma_depth_constant_same_as_layer(tmp_path, layout, uv_interp,
                                            integrator):
    """FESOM-C: u, v independent of ``s`` (per-column constants, varying
    from column to column and in time), a sloping bed and a moving surface:
    the linear run equals the layer run — bit for bit for the particles
    beyond the outer centres (every member reads a stored value at every
    stage), elsewhere every position within 4 ulp (longitude: within one
    rounding of the per-step wrap); every step starts at the same position
    in both runs, and every velocity is within the interpolator's ulp bound
    of ``max|c|`` (the constants lie in one binade, so that ulp is the
    field's).

    Measured (2026-10-20, on 60d4a42): every position of every run bit for
    bit the layer run's; up to 2 of the 56 velocities of a run differ, by
    at most 4 ulp of ``max|c|`` (``leastsq``, RK2; 2 with Euler and RK4),
    2 (element ``barycentric``), none (``edge_leastsq``, node
    ``barycentric``); ``s`` kept to 3.7e-15, the depths at ``s > 0``
    moving 0.12 m or more per step."""
    n_rec = N_STEPS_C + 2
    n_lay = SIG_C.size - 1

    def uv(x, y):
        cu, cv = _const_cols(x.size, n_rec)
        return (np.repeat(cu[:, None, :], n_lay, axis=1),
                np.repeat(cv[:, None, :], n_lay, axis=1))
    path = _write_fesomc(tmp_path / "fesomc.nc", NL, NL, SIG_C, uv,
                         n_rec=n_rec)
    s0 = np.array([s[2] for s in CONST_SIGMA_SEEDS])
    seeds = _sigma_seeds([s[0] for s in CONST_SIGMA_SEEDS],
                         [s[1] for s in CONST_SIGMA_SEEDS], s0)
    runs = {}
    for mode in ("linear", "layer"):
        runs[mode] = _traj(_cfg_fesomc(
            path, seeds, out=f"{mode}.nc", layout=layout,
            uv_interp=uv_interp, integrator=integrator, uv_vertical=mode,
            n_steps=N_STEPS_C))
    lin, lay = runs["linear"], runs["layer"]
    assert _global_attrs(Path(path).parent / "linear.nc")[
        "uv_vertical"] == "linear"
    # W = 0: every particle keeps its s, so the data particles stay beyond
    # the outer centres, the others between them; the depths follow the
    # sloping bed and the moving surface
    s_end = _sigma_end_s(lin, N_STEPS_C, DT)
    assert np.all(np.abs(s_end - s0[:, None]) <= 1e-12)
    assert np.all(np.abs(np.diff(lin["z"], axis=1))[s0 > 0] > 0.05)
    cen = _sigma_centres(SIG_C)
    exact = np.array([s[3] == "data" for s in CONST_SIGMA_SEEDS])
    assert np.all((s0[exact] < cen[0]) | (s0[exact] > cen[-1]))
    assert np.all((s0[~exact] > cen[0]) & (s0[~exact] < cen[-1]))
    for k in ("lon", "lat", "z", "vel_u", "vel_v"):
        assert_bitwise(lin[k][exact], lay[k][exact], k)
    for k in ("lat", "z"):
        assert np.all(np.abs(lin[k] - lay[k])
                      <= 4 * np.spacing(np.abs(lay[k]))), k
    lon_tol = np.maximum(4 * np.spacing(np.abs(lay["lon"])),
                         _wrap_quantum(lay["lon"]))
    assert np.all(np.abs(lin["lon"] - lay["lon"]) <= lon_tol), (
        np.abs(lin["lon"] - lay["lon"]) / _wrap_quantum(lay["lon"]))
    # velocities at the step starts, where both runs stand at the same place
    lon0 = np.array([s[0] for s in seeds])[:, None]
    lat0 = np.array([s[1] for s in seeds])[:, None]
    same = ((np.concatenate([lon0, lin["lon"][:, :-1]], axis=1)
             == np.concatenate([lon0, lay["lon"][:, :-1]], axis=1))
            & (np.concatenate([lat0, lin["lat"][:, :-1]], axis=1)
               == np.concatenate([lat0, lay["lat"][:, :-1]], axis=1)))
    assert same.all(), np.argwhere(~same)
    key = (uv_interp if uv_interp != "barycentric"
           else f"barycentric-{layout}")
    # every constant in one binade (the elements' columns include the
    # nodes': the same sequence, longer)
    ulp = np.spacing(C_BINADE[0])
    for c in _const_cols(2 * (NL - 1) ** 2, n_rec):
        mag = np.abs(c)
        assert C_BINADE[0] <= mag.min() and mag.max() < C_BINADE[1]
    tol = VEL_ULP[key] * ulp
    for k in ("vel_u", "vel_v"):
        gap = np.abs(lin[k] - lay[k])
        assert np.all(gap <= tol), (k, gap.max() / ulp)


# ---------------------------------------------------------------------------
# Sigma: the order of the reconstruction and of the run
# ---------------------------------------------------------------------------

def _profile_s(s):
    """The smooth profile in ``s``: ``cos(pi s)`` (flat at the surface and
    at the bed, so reading the outer centres there is second order)."""
    return np.cos(np.pi * np.asarray(s))


def _sigma_profile_uv(sig, g, n_rec):
    """``uv(x, y)`` of :func:`_write_fesomc`: ``g(x, y) cos(pi c_k)`` at the
    layer centres ``c_k`` of ``sig``."""
    f = _profile_s(_sigma_centres(sig))[None, :, None]

    def uv(x, y):
        gu, gv = g(x, y)
        shape = (n_rec, f.shape[1], np.size(x))
        return (np.broadcast_to(f * gu[None, None, :], shape).copy(),
                np.broadcast_to(f * gv[None, None, :], shape).copy())
    return uv


@pytest.mark.parametrize("layout,uv_interp", CASES, ids=CASE_IDS)
def test_sigma_profile_reconstruction_order(tmp_path, layout, uv_interp):
    """FESOM-C: ``u = G(x) cos(pi s)`` sampled at the layer centres of 10,
    20 and 40 equal sigma layers (``c_k = 1 - (sigma_k + sigma_{k+1}) / 2``),
    read through the public interpolator at one point at 1601 sigma
    fractions from the surface to the bed (linear mode: the ``s`` the
    integrators hand it; layer mode: the layer of ``z = s H`` in the
    column): the maximum error of linear mode falls by >= 3.5 per halving
    of the layers (second order), of the layer mode by 1.6 .. 2.4 (first
    order).

    Measured (2026-10-20, on 60d4a42; the same for the four interpolators,
    and the numbers of :func:`test_profile_reconstruction_order` with ``s``
    for ``z / H``): linear 1.003e-3, 2.512e-4, 6.283e-5 m/s (ratios 3.994,
    3.998), layer 1.275e-2, 6.394e-3, 3.200e-3 m/s (ratios 1.994,
    1.998)."""
    s = np.linspace(0.0, 1.0, 1601)
    gu, gv = _g(LON_P, LAT_P)
    errs = {"layer_linear": [], "layer": []}
    for n in N_LAYERS:
        sig = np.linspace(1.0, 0.0, n + 1)
        mesh = _load_fesomc(_write_fesomc(
            tmp_path / f"n{n}.nc", 9, 9, sig,
            _sigma_profile_uv(sig, _g, 1), n_rec=1))
        x, y = _cols(mesh, layout)
        cu, cv = _g(x, y)
        f = _profile_s(_sigma_centres(sig))[:, None]
        lon, lat = np.full(s.size, LON_P), np.full(s.size, LAT_P)
        nn = find_nearest_nodes(mesh, lon, lat)
        h = np.asarray(mesh.depth)[nn]
        for placement in errs:
            vel = VelocityField(
                u=f * cu[None, :], v=f * cv[None, :],
                eta=np.zeros(mesh.n_nodes),
                w=np.zeros((sig.size, mesh.n_nodes)), uv_layout=layout,
                uv_vertical=placement)
            if placement == "layer":
                st = make_state(lon, lat, s * h)
                layer = find_layers_sigma(mesh.sigma, st.z, nn, mesh.depth,
                                          vel.eta, st.alive_mask,
                                          H_particle=h)
            else:
                st = make_state(lon, lat, s.copy())
                layer = 0
            U, V, ok = _INTERP_FN[uv_interp](st, mesh, vel, nn, 1.0, layer)
            assert ok.all()
            p = _profile_s(s)
            errs[placement].append(max(np.abs(U - gu * p).max(),
                                       np.abs(V - gv * p).max()))
    lin, lay = np.array(errs["layer_linear"]), np.array(errs["layer"])
    r_lin, r_lay = lin[:-1] / lin[1:], lay[:-1] / lay[1:]
    assert np.all(r_lin >= 3.5), (lin, r_lin)
    assert np.all((r_lay >= 1.6) & (r_lay <= 2.4)), (lay, r_lay)


@pytest.mark.parametrize("layout,uv_interp", CASES, ids=CASE_IDS)
def test_sigma_trajectory_convergence(tmp_path, layout, uv_interp):
    """FESOM-C: ``u = G(x) cos(pi s)``, a zero dia-sigma W (every particle
    keeps its ``s`` while its depth follows the sloping bed and the moving
    surface), 50 particles at equally spaced ``s`` (the column without its
    ends), RK4 over four days on 10, 20 and 40 sigma layers: the RMS
    distance of the end positions from the reference solution of
    :func:`test_trajectory_convergence` (the same ``G``, depth factor
    ``cos(pi s)``) falls by >= 3.5 per halving of the layers in linear mode
    and by 1.6 .. 2.4 in layer mode; the time-step error (``dt`` against
    ``dt / 2``) is below 1 % of either mode's error.

    Measured (2026-10-20, on 60d4a42; the same for the four interpolators,
    and the numbers of the z-level test): linear 182.5, 45.66, 11.42 m
    (ratios 3.997, 3.998), layer 1860, 955.3, 476.7 m (ratios 1.947,
    2.004); the time-step error at most 5.2e-6 of the linear error, 1.2e-7
    of the layer error; ``s`` kept to 4.4e-16 while the depths moved up to
    22 m; displacements up to 30 km."""
    s = np.linspace(0.0, 1.0, N_DEPTHS + 2)[1:-1]
    seeds = _sigma_seeds(np.full(s.size, TRAJ_START[0]),
                         np.full(s.size, TRAJ_START[1]), s)
    z0 = np.array([q[2] for q in seeds])
    T = N_STEPS_T * DT
    ref_lon, ref_lat = _reference(_profile_s(s), T)
    errs = {"linear": [], "layer": []}
    for n in N_LAYERS:
        sig = np.linspace(1.0, 0.0, n + 1)
        work = tmp_path / f"n{n}"
        work.mkdir()
        path = _write_fesomc(work / "fesomc.nc", NL, NL, sig,
                             _sigma_profile_uv(sig, _traj_g, N_STEPS_T + 2),
                             n_rec=N_STEPS_T + 2)
        for mode in errs:
            ends = []
            for k, dt in enumerate((DT, DT / 2)):
                n_steps = N_STEPS_T * (k + 1)
                t = _traj(_cfg_fesomc(path, seeds, out=f"{mode}{k}.nc",
                                      layout=layout, uv_interp=uv_interp,
                                      uv_vertical=mode, n_steps=n_steps,
                                      dt=dt, temporal_interp=True))
                # W = 0: s kept to rounding, z moved with the column
                s_end = _sigma_end_s(t, n_steps, dt)
                assert np.all(np.abs(s_end - s[:, None]) <= 1e-12)
                assert np.abs(t["z"][:, -1] - z0).max() > 10.0
                ends.append((t["lon"][:, -1], t["lat"][:, -1]))
            err = _rms_gap(*ends[0], ref_lon, ref_lat)
            err_t = _rms_gap(*ends[0], *ends[1])
            assert err_t < 0.01 * err, (mode, n, err_t, err)
            errs[mode].append(err)
    lin, lay = np.array(errs["linear"]), np.array(errs["layer"])
    r_lin, r_lay = lin[:-1] / lin[1:], lay[:-1] / lay[1:]
    assert np.all(r_lin >= 3.5), (lin, r_lin)
    assert np.all((r_lay >= 1.6) & (r_lay <= 2.4)), (lay, r_lay)
