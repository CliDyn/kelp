"""FESOM 1.4 reader (``mesh.type: fesom14``): the mesh.

:meth:`Fesom14Reader.load_mesh` builds a KELP :class:`~kelp.core.mesh.Mesh`
from one of two sources, chosen by what ``mesh.meshpath`` is:

- **an ASCII mesh directory** (it holds ``nod3d.out``): ``nod2d.out``,
  ``elem2d.out``, ``nod3d.out`` and ``aux3d.out`` are read and cross-checked
  (:mod:`._ascii_mesh`); the static wet mask comes from the 3-D node index
  table.
- **level-wise data** (a NetCDF file, or a directory of ``*.nc`` files)
  whose files carry the node coordinates (``lon``, ``lat``) and the node cell
  polygons (``lon_bnds``, ``lat_bnds``): the triangles are recovered from the
  polygons (:mod:`._polygons`), the levels come from the level coordinate of
  the u variable and the static wet mask from its first record.

Either way the mesh is a z-level mesh on the full level table:
``mesh.sigma`` holds the level depths (m, positive down, ``sigma[0] == 0``),
``mesh.depth`` the bed of each column, i.e. the depth of its last wet level,
taken exactly from the level table. The reader stores u/v on the levels
(``native_uv_vertical = "level"``) and declares the last wet level of each
column in ``mesh.metadata["uv_data_last"]``.

Node longitudes are brought to [-180, 180); element centres are circular
means of their nodes (as in the FESOM2 reader), so a mesh across the
dateline, written with longitudes 0..360 or not, needs nothing special.

Rotation: an ASCII mesh may be stored in the rotated frame of the model run;
``mesh.input_rotation`` then gives the Euler angles (alpha, beta, gamma, in
degrees, as in the run's namelist) of that frame and the coordinates are
rotated back to geographic ones on load, with the convention of
:func:`kelp.core.rotation.euler_matrix` (``v_rot = R @ v_geo``). The angles
are never guessed: a global ASCII mesh without a node north of 85 N, loaded
without ``mesh.input_rotation``, draws a warning (a geographic FESOM mesh
reaches the Arctic; a mesh stored in a rotated frame has its poles on land).
``mesh.rotation`` (the frame KELP tracks in) works as for the FESOM2 reader,
whose helpers are used.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

from ...core.mesh import Mesh, build_node_to_elems, build_search
from ...core.rotation import euler_matrix, unrotate_lonlat
from .. import register
from ..fesom2 import (
    _apply_mesh_rotation,
    _compute_elem_centres,
    _resolve_rotation_angles,
    _warn_near_pole_fesom2,
)
from ._ascii_mesh import MESH_FILES, NOD2D, NOD3D, read_ascii_mesh
from ._layout import TIME_DIM, detect_form, is_data, wet_mask_levelwise
from ._polygons import recover_triangles

__all__ = ["Fesom14Reader", "R_EARTH_FESOM14", "DEFAULT_VAR_U", "DEPTHO"]

#: Earth radius of FESOM 1.4 (m), the same value as FESOM2's
R_EARTH_FESOM14 = 6_367_500.0

#: the variable whose first record gives the static wet mask of a mesh built
#: from cell polygons, unless ``mode.var_u`` names another
DEFAULT_VAR_U = "uo"

#: the sea-floor depth variable checked against that mask when present
DEPTHO = "deptho"

#: a first level within this distance of 0 (m) is the surface
_SURFACE_TOL = 1e-6

#: largest difference (m) between ``deptho`` and the depth of the last wet level
_DEPTHO_TOL = 0.01

#: largest difference (deg) between the node coordinates of two files
_COORD_TOL = 1e-6

#: a mesh whose nodes cover more longitude than this (deg) is global
_GLOBAL_SPAN = 350.0

#: a global geographic FESOM mesh has nodes north of this latitude (deg)
_ARCTIC_LAT = 85.0

_KINDS = (
    "mesh.meshpath must be a FESOM 1.4 ASCII mesh directory (holding "
    f"{', '.join(MESH_FILES)}) or level-wise data with cell polygons: a NetCDF "
    "file, or a directory of *.nc files, whose u variable has the dimensions "
    "(time, depth, ncells) and whose files carry lon, lat, lon_bnds and lat_bnds"
)


@dataclass
class _Parts:
    """What a mesh source yields before the mesh is assembled."""

    source: str                     # "ascii" | "polygons"
    lon: np.ndarray                 # geographic, [-180, 180)
    lat: np.ndarray
    tri: np.ndarray                 # (E, 3) int32
    levels: np.ndarray              # (K,) float64, positive down, [0] == 0
    wet: np.ndarray                 # (K, N) bool, a prefix of each column
    metadata: Dict[str, object] = field(default_factory=dict)


@register("fesom14")
class Fesom14Reader:
    """Reader for FESOM 1.4 meshes and output.

    u/v are stored on the levels: ``native_uv_vertical = "level"``.
    """

    native_uv_vertical = "level"

    @staticmethod
    def load_mesh(mesh_cfg, mode_cfg=None) -> Mesh:
        """Build the mesh from an ASCII mesh directory or from cell polygons.

        Parameters
        ----------
        mesh_cfg : MeshConfig
            ``meshpath`` (see below), optional ``input_rotation`` (the Euler
            angles in degrees of the frame an ASCII mesh is stored in),
            ``rotation`` (the frame KELP tracks in: ``None``, ``"fesom2"`` or
            three angles) and ``r_earth`` (default 6 367 500 m, FESOM 1.4's
            value).
        mode_cfg : ModeConfig, optional
            ``var_u`` names the u variable whose first record gives the wet
            mask of a mesh built from cell polygons (default ``"uo"``).

        Returns
        -------
        Mesh
            A z-level mesh: ``sigma`` = the levels (m, positive down),
            ``depth`` = the depth of each column's last wet level (exact),
            node longitudes in [-180, 180), element centres by circular mean,
            in the tracking frame when ``mesh.rotation`` is set (with
            ``_vec_rot_factors`` as the FESOM2 reader sets them). Its
            ``metadata`` holds:

            ``fesom14_source``
                ``"ascii"`` or ``"polygons"``.
            ``fesom14_levels``
                The full level table, float64 ``(K,)`` (``mesh.sigma`` may
                later be cut by ``max_levels``).
            ``fesom14_wet``
                The static wet mask, bool ``(K, N)``, C-contiguous.
            ``fesom14_lon``, ``fesom14_lat``
                The geographic node coordinates (longitudes in
                [-180, 180)), whatever the tracking frame.
            ``uv_data_last``
                int32 ``(N,)``, the index of each column's last wet level
                (a new array; never written into).
            ``fesom14_idx``
                ASCII only: the ``(N, K)`` 3-D node index table, -1 = dry.
            ``fesom14_polygon_report``
                Polygons only: the report of the triangle recovery
                (``n_pinch_triangles``, ``n_full_slot_polygons``, ...).
            ``fesom14_mask_file``, ``fesom14_deptho_file``
                Polygons only: the file whose first record gave the wet mask,
                and the file whose ``deptho`` was checked against it (None
                when there is none).

            The array entries other than ``uv_data_last`` are read-only.

        Raises
        ------
        FileNotFoundError
            ``mesh.meshpath`` does not exist, or an ASCII mesh directory
            lacks one of its four files.
        ValueError
            ``mesh.meshpath`` is neither kind of source; a mesh file fails
            its checks; ``mesh.input_rotation`` is given for a mesh built
            from cell polygons. The checks of a mesh from cell polygons, in
            order: the u variable is level-wise; its level coordinate is
            finite, starts at the surface (0 m) and increases (positive
            down); the file carries ``lon``, ``lat`` and the polygons; the
            first record is data from the surface down to each column's bed
            and fill below; ``deptho`` (when present) agrees with that mask;
            the polygons give a valid triangulation. Every message names the
            file, variable or node and how to fix it.

        Notes
        -----
        **Mesh from cell polygons.** ``mesh.meshpath`` is either one file or a
        directory; in a directory every ``*.nc`` file is looked at in name
        order and the first one holding the u variable gives the mesh and
        the mask (the mask of a FESOM 1.4 run is static, the same in every
        record; the records read later are compared against it). ``deptho`` is
        taken from the first file that holds it (in a single file: that
        file); when it carries ``lon``/``lat`` they must be the mesh's nodes
        within 1e-6 deg (any longitude convention). Its values must equal
        the depth of each column's last level with data within 0.01 m.
        This is the only guard against level-wise data that writes 0
        instead of a fill value below the bed: such a record has data at
        every level, so its mask is wet everywhere. **Without** ``deptho``
        the first record's mask is used as it is, and that variant cannot
        be detected (an ASCII mesh, whose index table gives the bed, can).
        """
        meshpath = str(getattr(mesh_cfg, "meshpath", "") or "").strip()
        if not meshpath:
            # Path("") is the working directory: never scan it by accident
            raise ValueError(f"fesom14: mesh.meshpath is not set; {_KINDS}")
        path = Path(meshpath).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"fesom14: mesh.meshpath {path} does not exist; {_KINDS}")
        input_angles = _input_rotation(mesh_cfg)
        tracking_angles = _resolve_rotation_angles(mesh_cfg)
        r_earth = getattr(mesh_cfg, "r_earth", None) or R_EARTH_FESOM14

        if path.is_dir() and (path / NOD3D).is_file():
            parts = _from_ascii(path, input_angles)
        else:
            if input_angles is not None:
                raise ValueError(
                    f"fesom14: mesh.input_rotation is set, but {path} is not an "
                    f"ASCII mesh directory (no {NOD3D}): the angles apply only to "
                    "an ASCII mesh stored in a rotated frame. A mesh from cell "
                    "polygons takes the coordinates of the data files, which "
                    "are geographic; remove mesh.input_rotation")
            var_u = (getattr(mode_cfg, "var_u", None) if mode_cfg is not None
                     else None) or DEFAULT_VAR_U
            parts = _from_polygons(path, var_u)
        return _assemble(parts, path, tracking_angles, float(r_earth))


# ---------------------------------------------------------------------------
# Common pieces
# ---------------------------------------------------------------------------


def _input_rotation(mesh_cfg) -> Optional[Tuple[float, float, float]]:
    """``mesh.input_rotation`` as three finite floats, or None."""
    value = getattr(mesh_cfg, "input_rotation", None)
    if value is None:
        return None
    try:
        if isinstance(value, (str, bytes)):
            raise TypeError("a string")
        angles = tuple(float(a) for a in value)
    except (TypeError, ValueError):
        angles = ()
    if len(angles) != 3 or not np.isfinite(angles).all():
        raise ValueError(
            "fesom14: mesh.input_rotation must be the three Euler angles "
            f"(alpha, beta, gamma; finite degrees), got {value!r}")
    return angles


def _normalise_lon(lon: np.ndarray) -> np.ndarray:
    """A copy of ``lon`` with every value in [-180, 180).

    Values already in range are kept bit for bit; values in [180, 360] lose
    exactly 360 (an exact subtraction); others are wrapped by modulo.
    """
    out = np.array(lon, dtype=np.float64)
    high = (out >= 180.0) & (out <= 360.0)
    out[high] -= 360.0
    other = (out < -180.0) | (out >= 180.0)
    if other.any():
        out[other] = (out[other] + 180.0) % 360.0 - 180.0
        out[out >= 180.0] -= 360.0       # rounding of the modulo
    return out


def _lon_coverage(lon: np.ndarray) -> float:
    """Degrees of longitude covered by the nodes: 360 minus the widest gap
    between neighbouring node longitudes around the circle."""
    u = np.unique(_normalise_lon(lon))
    if u.size < 2:
        return 0.0
    widest = max(float(np.diff(u).max()), float(u[0] + 360.0 - u[-1]))
    return 360.0 - widest


def _lon_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|a - b| in degrees of longitude, across the dateline."""
    d = np.abs(np.asarray(a, dtype=np.float64)
               - np.asarray(b, dtype=np.float64)) % 360.0
    return np.minimum(d, 360.0 - d)


def _frozen(a: np.ndarray) -> np.ndarray:
    """A read-only copy of ``a``."""
    a = np.array(a, copy=True)
    a.setflags(write=False)
    return a


def _assemble(parts: _Parts, path: Path, tracking_angles, r_earth: float) -> Mesh:
    """The Mesh from a source's parts (shared by both sources)."""
    lon_geo, lat_geo = parts.lon, parts.lat
    n_nodes = lon_geo.size
    wet = np.ascontiguousarray(parts.wet, dtype=bool)
    # the index of each column's last wet level (wet is a prefix, never empty)
    last = (wet.sum(axis=0) - 1).astype(np.int32)
    depth = parts.levels[last].astype(np.float64)

    tri = np.ascontiguousarray(parts.tri, dtype=np.int32)
    lon_c, lat_c = _compute_elem_centres(lon_geo, lat_geo, tri)
    _warn_near_pole_fesom2(lat_geo, tracking_angles)
    lon, lat = lon_geo.copy(), lat_geo.copy()
    vec_rot_factors = None
    if tracking_angles is not None:
        lon, lat, lon_c, lat_c, vec_rot_factors = _apply_mesh_rotation(
            lon, lat, lon_c, lat_c, tracking_angles)
    n2e, n2e_count = build_node_to_elems(n_nodes, tri)

    metadata = {
        "meshpath": str(path),
        "fesom14_source": parts.source,
        "fesom14_levels": _frozen(parts.levels),
        "fesom14_wet": _frozen(wet),
        "fesom14_lon": _frozen(lon_geo),
        "fesom14_lat": _frozen(lat_geo),
        "uv_data_last": last,
    }
    metadata.update(parts.metadata)
    mesh = Mesh(
        name="fesom14",
        nodes_lon=np.ascontiguousarray(lon, dtype=np.float64),
        nodes_lat=np.ascontiguousarray(lat, dtype=np.float64),
        depth=depth,
        elem_nodes=tri,
        elem_center_lon=lon_c,
        elem_center_lat=lat_c,
        node_to_elems=n2e,
        node_to_elems_count=n2e_count,
        sigma=parts.levels.copy(),
        vertical_coord="zlevel",
        r_earth=r_earth,
        rotation_angles=tracking_angles,
        metadata=metadata,
    )
    mesh._vec_rot_factors = vec_rot_factors
    source = "ASCII files" if parts.source == "ascii" else "cell polygons"
    print(f"  FESOM 1.4 mesh from {source}: {n_nodes:,} nodes, {tri.shape[0]:,} "
          f"triangles, {parts.levels.size} levels (to {parts.levels[-1]:g} m)")
    report = parts.metadata.get("fesom14_polygon_report")
    if report:
        print(f"    polygons filling every slot: {report['n_full_slot_polygons']}, "
              f"pinch triangles: {report['n_pinch_triangles']}")
    build_search(mesh)
    return mesh


# ---------------------------------------------------------------------------
# ASCII mesh
# ---------------------------------------------------------------------------


def _from_ascii(meshdir: Path, input_angles) -> _Parts:
    """The parts of a mesh read from an ASCII mesh directory."""
    am = read_ascii_mesh(meshdir)
    if input_angles is not None:
        lon, lat = unrotate_lonlat(am.lon, am.lat, euler_matrix(*input_angles))
        lon = _normalise_lon(lon)
    else:
        lon, lat = _normalise_lon(am.lon), np.array(am.lat, dtype=np.float64)
        _warn_if_rotated(lon, lat, meshdir)
    return _Parts(
        source="ascii",
        lon=lon,
        lat=lat,
        tri=am.tri,
        levels=np.array(am.levels, dtype=np.float64),
        wet=np.ascontiguousarray((am.idx >= 0).T),
        metadata={"fesom14_idx": _frozen(am.idx)},
    )


def _warn_if_rotated(lon: np.ndarray, lat: np.ndarray, meshdir: Path) -> None:
    """Warn when a global mesh has no node north of 85 N."""
    if _lon_coverage(lon) > _GLOBAL_SPAN and not (lat > _ARCTIC_LAT).any():
        warnings.warn(
            f"fesom14: the ASCII mesh in {meshdir} covers all longitudes but has "
            f"no node north of {_ARCTIC_LAT:g} N (northernmost "
            f"{float(lat.max()):.2f} N). A geographic FESOM mesh reaches the "
            "Arctic; this mesh is probably stored in the rotated frame of the "
            "model run, and tracking it as geographic would put every particle "
            "in the wrong place. Set mesh.input_rotation to the Euler angles "
            "(alphaEuler, betaEuler, gammaEuler) of the run's namelist; if the "
            "mesh really is geographic, ignore this warning.",
            UserWarning, stacklevel=4)


# ---------------------------------------------------------------------------
# Mesh from cell polygons
# ---------------------------------------------------------------------------


def _from_polygons(path: Path, var_u: str) -> _Parts:
    """The parts of a mesh recovered from the cell polygons of level-wise data."""
    u_file, d_file = _polygon_sources(path, var_u)
    with _open(u_file) as ds:
        da = ds[var_u]
        try:
            form = detect_form(da)
        except ValueError as exc:
            raise ValueError(f"fesom14: {u_file}: {_strip(exc)} {_KINDS}") from exc
        if form != "levelwise":
            raise ValueError(
                f"fesom14: variable {var_u!r} in {u_file} is one vector over the "
                "3-D nodes (dimensions (time, nodes_3d)), which carries no "
                "levels and no cell polygons: a mesh for such files must come "
                "from the FESOM 1.4 ASCII mesh. Set mesh.meshpath to the ASCII "
                f"mesh directory (holding {', '.join(MESH_FILES)})")
        zdim, hdim = da.dims[1], da.dims[2]
        levels = _levels(ds, zdim, var_u, u_file)
        lon, lat, lon_b, lat_b = _node_polygons(ds, hdim, u_file)
        if ds.sizes[TIME_DIM] == 0:
            raise ValueError(
                f"fesom14: variable {var_u!r} in {u_file} has no record; the "
                "static wet mask is read from its first record")
        first = np.asarray(da.isel({TIME_DIM: 0}).values)

    try:
        wet = wet_mask_levelwise(first)
    except ValueError as exc:
        raise ValueError(
            f"fesom14: variable {var_u!r}, record 0 of {u_file}: {_strip(exc)} "
            "The static wet mask of a mesh from cell polygons is read from this "
            "record; check the file, or build the mesh from the FESOM 1.4 ASCII "
            "mesh") from exc
    last = wet.sum(axis=0) - 1
    if d_file is not None:
        _check_deptho(d_file, lon, lat, levels, last, var_u, u_file)

    try:
        tri, report = recover_triangles(lon, lat, lon_b, lat_b)
    except ValueError as exc:
        raise ValueError(f"fesom14: {u_file}: {exc}") from exc
    return _Parts(
        source="polygons",
        lon=_normalise_lon(lon),
        lat=lat,
        tri=tri,
        levels=levels,
        wet=wet,
        metadata={
            "fesom14_polygon_report": dict(report),
            "fesom14_mask_file": str(u_file),
            "fesom14_deptho_file": None if d_file is None else str(d_file),
        },
    )


def _strip(exc: Exception) -> str:
    """The message of a layout error without its ``fesom14:`` prefix."""
    msg = str(exc)
    return msg[len("fesom14: "):] if msg.startswith("fesom14: ") else msg


def _open(path: Path) -> xr.Dataset:
    """A data file, opened lazily with its numbers undecoded in time."""
    try:
        return xr.open_dataset(path, decode_times=False)
    except Exception as exc:     # xarray raises several types for "not readable"
        raise ValueError(
            f"fesom14: {path} cannot be opened as a NetCDF file ({exc}); "
            f"{_KINDS}") from exc


def _variable_names(path: Path) -> List[str]:
    """The variable names of a NetCDF file (its header only)."""
    import netCDF4

    try:
        with netCDF4.Dataset(str(path), "r") as nc:
            return list(nc.variables)
    except Exception as exc:     # netCDF4 raises OSError and others
        raise ValueError(
            f"fesom14: {path} cannot be opened as a NetCDF file ({exc}); "
            "every *.nc file of a mesh.meshpath directory must be readable "
            "(move other files out of it)") from exc


def _polygon_sources(path: Path, var_u: str) -> Tuple[Path, Optional[Path]]:
    """The file holding the u variable and the file holding ``deptho``."""
    if path.is_file():
        names = _open_names(path)
        if var_u not in names:
            raise ValueError(
                f"fesom14: {path} holds no variable {var_u!r} (it holds "
                f"{', '.join(sorted(names)) or 'no variable'}); the mesh from cell "
                f"polygons and its wet mask are read from the u variable "
                f"{var_u!r} (mode.var_u). Point mesh.meshpath at a file of "
                f"{var_u!r}, or set mode.var_u")
        return path, (path if DEPTHO in names else None)

    files = sorted(p for p in path.iterdir()
                   if p.suffix == ".nc" and p.is_file() and not p.name.startswith("."))
    if not files:
        hint = ""
        if (path / NOD2D).is_file():
            hint = (f" It holds {NOD2D} but not {NOD3D}: a FESOM2 mesh "
                    "directory is read with mesh.type: fesom2.")
        raise ValueError(
            f"fesom14: mesh.meshpath {path} holds neither {NOD3D} (an ASCII "
            f"mesh) nor any *.nc file (level-wise data).{hint} {_KINDS}")
    u_file = d_file = None
    for f in files:
        names = _variable_names(f)
        if u_file is None and var_u in names:
            u_file = f
        if d_file is None and DEPTHO in names:
            d_file = f
        if u_file is not None and d_file is not None:
            break
    if u_file is None:
        raise ValueError(
            f"fesom14: no *.nc file in {path} holds the u variable {var_u!r}; the "
            "mesh from cell polygons and its wet mask are read from it. Add its "
            "files to the directory, or set mode.var_u to the name the files use")
    return u_file, d_file


def _open_names(path: Path) -> List[str]:
    """The variable names of a file given as mesh.meshpath."""
    with _open(path) as ds:
        return list(ds.variables)


def _levels(ds: xr.Dataset, zdim: str, var_u: str, path: Path) -> np.ndarray:
    """The level depths (m, positive down) from the level coordinate."""
    where = f"the level coordinate {zdim!r} of {var_u!r} in {path}"
    if zdim not in ds.variables:
        raise ValueError(
            f"fesom14: {where} is missing: the dimension {zdim!r} has no "
            "coordinate variable, and the reader needs the level depths (m, "
            "positive down, 0 at the surface)")
    z = np.asarray(ds[zdim].values, dtype=np.float64)
    if z.ndim != 1 or z.size == 0:
        raise ValueError(f"fesom14: {where} must be 1-D and not empty, got shape "
                         f"{z.shape}")
    bad = np.flatnonzero(~np.isfinite(z))
    if bad.size:
        k = int(bad[0])
        raise ValueError(
            f"fesom14: {where}: level {k} is {float(z[k])!r}; every level needs "
            "a finite depth in metres")
    if abs(z[0]) > _SURFACE_TOL:
        raise ValueError(
            f"fesom14: {where}: level 0 is at {float(z[0]):g} m, but FESOM 1.4 "
            "levels start at the surface (0 m); the files hold data on the "
            "levels of the model, the first of them at the surface")
    up = np.flatnonzero(np.diff(z) <= 0.0)
    if up.size:
        k = int(up[0]) + 1
        raise ValueError(
            f"fesom14: {where}: level {k} ({float(z[k]):g} m) is not deeper "
            f"than level {k - 1} ({float(z[k - 1]):g} m); the level depths must "
            "increase, positive down (a level axis stored positive up, with "
            "negative depths, must be negated)")
    z = z.copy()
    z[0] = 0.0                           # the surface (also no -0.0)
    return z


def _node_polygons(ds: xr.Dataset, hdim: str, path: Path):
    """Node coordinates and cell polygons of a level-wise file."""
    n_nodes = ds.sizes[hdim]
    out = []
    for name in ("lon", "lat"):
        if name not in ds.variables:
            raise ValueError(
                f"fesom14: {path} has no variable {name!r}: a mesh from cell "
                "polygons needs the node coordinates lon and lat and the cell "
                "polygons lon_bnds and lat_bnds (data files without them need "
                "the FESOM 1.4 ASCII mesh as mesh.meshpath)")
        coord = np.asarray(ds[name].values, dtype=np.float64)
        if coord.shape != (n_nodes,):
            raise ValueError(
                f"fesom14: {name!r} in {path} has shape {coord.shape}, but the "
                f"horizontal dimension {hdim!r} has {n_nodes} nodes")
        out.append(coord)
    for name in ("lon", "lat"):
        bname = str(ds[name].attrs.get("bounds", f"{name}_bnds"))
        if bname not in ds.variables:
            raise ValueError(
                f"fesom14: {path} has no cell polygons: the bounds variable "
                f"{bname!r} of {name!r} is missing. A mesh from data files needs "
                "lon_bnds and lat_bnds (the node cell polygons); without them, "
                "set mesh.meshpath to the FESOM 1.4 ASCII mesh directory")
        bnds = np.asarray(ds[bname].values, dtype=np.float64)
        if bnds.ndim != 2 or bnds.shape[0] != n_nodes:
            raise ValueError(
                f"fesom14: the cell polygons {bname!r} in {path} have shape "
                f"{bnds.shape}, expected ({n_nodes}, nv) for the {n_nodes} nodes "
                f"of {hdim!r}")
        out.append(bnds)
    return out[0], out[1], out[2], out[3]


def _check_deptho(d_file: Path, lon: np.ndarray, lat: np.ndarray,
                  levels: np.ndarray, last: np.ndarray, var_u: str,
                  u_file: Path) -> None:
    """``deptho`` must equal the depth of each column's last wet level."""
    n_nodes = lon.size
    with _open(d_file) as ds:
        values = np.asarray(ds[DEPTHO].values, dtype=np.float64)
        coords = None
        if "lon" in ds.variables and "lat" in ds.variables:
            coords = (np.asarray(ds["lon"].values, dtype=np.float64),
                      np.asarray(ds["lat"].values, dtype=np.float64))
    if values.shape != (n_nodes,):
        raise ValueError(
            f"fesom14: {DEPTHO} in {d_file} has shape {values.shape}, but the "
            f"mesh from {u_file} has {n_nodes} nodes; {DEPTHO} must be one value "
            "per node of the same mesh. Move a file of another mesh out of "
            "mesh.meshpath")
    if (coords is not None and coords[0].shape == (n_nodes,)
            and coords[1].shape == (n_nodes,)):
        off = (_lon_diff(coords[0], lon) > _COORD_TOL) | (
            np.abs(coords[1] - lat) > _COORD_TOL)
        if off.any():
            n = int(np.argmax(off))
            raise ValueError(
                f"fesom14: the node coordinates of {DEPTHO} in {d_file} differ "
                f"from those of {var_u!r} in {u_file} at node {n}: "
                f"({float(coords[0][n]):.7f}, {float(coords[1][n]):.7f}) against "
                f"({float(lon[n]):.7f}, {float(lat[n]):.7f}) ({int(off.sum())} "
                f"node(s) differ by more than {_COORD_TOL:g} deg). The files "
                "must list the same nodes in the same order; move a file of "
                "another mesh out of mesh.meshpath")
    missing = ~is_data(values)
    if missing.any():
        n = int(np.argmax(missing))
        raise ValueError(
            f"fesom14: {DEPTHO} in {d_file} has no value at node {n} "
            f"({float(values[n])!r}; {int(missing.sum())} node(s)); every node "
            f"of a FESOM 1.4 mesh is ocean and needs its sea-floor depth")
    depth = levels[last]
    off = np.abs(values - depth) > _DEPTHO_TOL
    if off.any():
        n = int(np.argmax(off))
        raise ValueError(
            f"fesom14: {DEPTHO} in {d_file} disagrees with the wet mask of "
            f"{var_u!r} (record 0 of {u_file}) at node {n}: {DEPTHO} = "
            f"{float(values[n]):g} m, but the last level with data is level "
            f"{int(last[n])} at {float(depth[n]):g} m ({int(off.sum())} node(s) "
            f"differ by more than {_DEPTHO_TOL:g} m). Data written as 0 (not as "
            "a fill value) below the bed look wet at every level, and their bed "
            "cannot be read from the data; build the mesh from the FESOM 1.4 "
            f"ASCII mesh instead. If {DEPTHO} belongs to another mesh or run, "
            "move its file out of mesh.meshpath")
