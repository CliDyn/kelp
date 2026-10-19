"""FESOM 1.4 reader (``mesh.type: fesom14``): the mesh and the data files.

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

:meth:`Fesom14Reader.open_dataset` opens the variables the run consumes —
level-wise ``(time, level, node)`` or one vector over the 3-D nodes ``(time,
nodes_3d)`` — grouping the files by the variables they hold, resolves what
time each record of each variable represents (:mod:`._timeaxis`) and requires
every variable to have the records of u. It checks only what the files say
about themselves; the checks against the mesh belong to ``prepare``.

:meth:`Fesom14Reader.prepare` checks a dataset against the mesh — the sizes,
the full level axis, the node coordinates when the data carry them, and the
wet mask of record 0 of every variable — and changes nothing. The engine
runs it at every entry of a run (after ``open_dataset``, then at each
realization).

:meth:`Fesom14Reader.read_velocities` and
:meth:`Fesom14Reader.read_scalar_fields` read one record. Every record of
every variable is first compared with the mesh's static wet mask (data in
exactly the wet slots), on all levels, before anything is changed; only then
are values below the bed set to 0 (u/v/w) or to the last value above the bed
(sampled scalars), the levels cut to the kept ones (``mesh.sigma``, which
``mode.max_levels`` may have cut) and u/v rotated into the tracking frame.
3-D tracking gets u/v/w on the levels, ``(K, N)``, ``uv_vertical="level"``,
w positive down; 2-D tracking gets u/v ``(N,)`` from one level
(``mode.sigma_layer``, level 0 by default) or the depth mean to the bed
(``mode.sigma_mean``). The sea-surface height is optional (0 without it).
"""
from __future__ import annotations

import glob
import os
import warnings
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import cftime
import numpy as np
import xarray as xr

from ...core.io import VelocityField
from ...core.mesh import Mesh, build_node_to_elems, build_search
from ...core.rotation import euler_matrix, unrotate_lonlat
from .. import register, resolve_w_sign
from ..fesom2 import (
    _apply_mesh_rotation,
    _compute_elem_centres,
    _resolve_rotation_angles,
    _warn_near_pole_fesom2,
)
from ._ascii_mesh import MESH_FILES, NOD2D, NOD3D, read_ascii_mesh
from ._layout import (DEPTH_DIMS, HORIZONTAL_DIMS, MISSING_TEXT, TIME_DIM,
                      check_record, detect_form, first_node_level,
                      gather_nodes3d, is_data, wet_mask_levelwise)
from ._polygons import recover_triangles
from ._timeaxis import (FileAxis, ResolvedAxis, TimeOptions,
                        _canonical_calendar, first_difference,
                        parse_output_schedule, resolve_axis)

if TYPE_CHECKING:
    from ...core.config import DiffusionConfig, MeshConfig, ModeConfig

__all__ = ["Fesom14Reader", "R_EARTH_FESOM14", "DEFAULT_VAR_U", "DEFAULT_VAR_V",
           "DEFAULT_VAR_W", "DEFAULT_VAR_ETA", "DEPTHO"]

#: Euler angles (alpha, beta, gamma) in degrees
Angles = Tuple[float, float, float]

#: Earth radius of FESOM 1.4 (m), the same value as FESOM2's
R_EARTH_FESOM14 = 6_367_500.0

#: the variable whose first record gives the static wet mask of a mesh built
#: from cell polygons, unless ``mode.var_u`` names another
DEFAULT_VAR_U = "uo"

#: default names of the other variables ``open_dataset`` consumes
DEFAULT_VAR_V = "vo"
DEFAULT_VAR_W = "wo"
#: the sea-surface height, optional (``mode.var_eta`` names another)
DEFAULT_VAR_ETA = "zos"

#: the sea-floor depth variable checked against that mask when present
DEPTHO = "deptho"

#: a first level within this distance of 0 (m) is the surface
_SURFACE_TOL = 1e-6

#: largest difference (m) between ``deptho`` and the depth of the last wet level
_DEPTHO_TOL = 0.01

#: largest difference (deg) between the node coordinates of two files
_COORD_TOL = 1e-6

#: largest difference (m) between the level coordinates of two data files
_LEVEL_TOL = 1e-3

#: a file holding one of these is a restart file
_RESTART_VARS = ("wpot", "iter")

#: fields of the older branch's mean output, written together over the 3-D nodes
_OLDER_BRANCH_FIELDS = ("u", "v", "w", "temp", "salt")

#: characters that make a data path a glob pattern
_GLOB_CHARS = "*?["

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
    def load_mesh(mesh_cfg: "MeshConfig",
                  mode_cfg: Optional["ModeConfig"] = None) -> Mesh:
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
                (a new array the reader never writes into or replaces; the
                engine puts a read-only copy in its place at its first use).
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

    @staticmethod
    def open_dataset(path: str, v_path: Optional[str] = None,
                     w_path: Optional[str] = None,
                     mode_cfg: Optional["ModeConfig"] = None, *,
                     time_stamp: Optional[str] = None,
                     time_window: Optional[str] = None,
                     time_offset: Optional[str] = None) -> xr.Dataset:
        """Open the variables a run consumes, on one exactly resolved time axis.

        Only checks intrinsic to the files happen here; every check against
        the mesh (sizes, levels, node coordinates, wet masks) is made by the
        reader's ``prepare(mesh, ds)`` hook.

        Parameters
        ----------
        path : str
            ``data.ncinput``: a NetCDF file, a directory (its ``*.nc``
            files) or a glob pattern.
        v_path, w_path : str, optional
            More files, directories or patterns (``data.ncinput_v``,
            ``data.ncinput_w``); their files join those of ``path``.
        mode_cfg : ModeConfig, optional
            ``var_u``/``var_v``/``var_w`` (defaults ``uo``/``vo``/``wo``),
            ``var_eta`` (default ``zos``), ``sample_vars`` and ``Fdrift3D``.
        time_stamp, time_window, time_offset : str, optional
            ``data.time_stamp``, ``data.time_window``, ``data.time_offset``:
            what each record's stamp means (see
            :mod:`kelp.readers.fesom14._timeaxis`); validated before any file
            is read.

        Returns
        -------
        xarray.Dataset
            The CONSUMED variables only, chosen before any time resolution
            or alignment: u and v; w for 3-D tracking (``mode.Fdrift3D``;
            without a mode config, when present); the sea-surface height
            when present; every ``mode.sample_vars`` entry. Other variables
            and files under ``path`` are ignored. Each variable keeps the
            dimensions of its files: ``(time, <depth>, <horizontal>)``
            (level-wise) or ``(time, nodes_3d)`` (one vector over the 3-D
            nodes) for 3-D fields, ``(time, <horizontal>)`` for 2-D ones;
            its records are read lazily, one record per chunk, and missing
            values (NaN after decoding, or a raw fill) are left as they are.
            Coordinates: ``time``, the resolved instants (cftime in the
            calendar of the u files; the window centre of a mean, the stamp
            of a snapshot); the level coordinate of the level-wise form; the
            node coordinates ``lon``/``lat`` of the u files when they carry
            them. Attributes: ``fesom14_form`` (``"levelwise"`` or
            ``"nodes3d"``), ``fesom14_has_eta`` (``1`` or ``0``) and
            ``fesom14_time_source`` (``"options"``, ``"bounds"`` or
            ``"schedule"``: what resolved the u axis). ``close()`` closes
            every file. Without a sea-surface height, one line of the run
            summary says so (eta is then 0).

        Raises
        ------
        FileNotFoundError
            A data path does not exist, or a pattern matches no file.
        ValueError
            Invalid or incomplete time options; an unreadable file; a
            FESOM 1.4 restart file (it holds ``wpot`` or ``iter``) or mean
            output of the older branch (``<runid>.<year>.oce.mean.nc``) that
            holds a consumed variable, or that is the only candidate for a
            missing one; a required variable that no file holds (also a
            ``mode.var_eta`` set but absent); a layout that is not one of the
            two FESOM 1.4 forms; files or variables of different layouts,
            dimension sizes, levels (beyond 1e-3 m) or node coordinates
            (beyond 1e-6 deg, either longitude convention); any refusal of
            the time resolution (no source of time, a stamp off its schedule
            or convention, a gap, an overlap); variables whose resolved
            records (instants and windows) differ — compared by value, so
            output schedules written differently but giving the same events
            are accepted, and different schedules are named when the records
            differ. Messages name the files, variables and records (by their
            instant) and the fix.

        Notes
        -----
        Files are grouped per variable by the variables they hold, never by
        their names. The time axis of every variable is resolved on its own
        from all its files (each with its own epoch and calendar), then
        every consumed variable must have exactly the records of u: the
        same instants and windows (within 1 ms), whatever source of time
        (bounds, schedule, options) resolved them.
        """
        opts = TimeOptions(stamp=time_stamp, window=time_window,
                           offset=time_offset)
        opts.validate()                  # before any file is read
        sources = [str(p) for p in (path, v_path, w_path) if p]
        heads = [_header(f) for f in _data_files(sources)]
        roles = _consumed_roles(mode_cfg)
        holders = _holders(roles, heads, sources)
        eta_name = next(r.name for r in roles if r.eta)
        roles = [r for r in roles if holders[r.name]]
        opened: Dict[str, xr.Dataset] = {}
        try:
            ds = _open_consumed(roles, holders, opts, opened)
        except BaseException:
            for f in opened.values():
                f.close()
            raise
        if not ds.attrs["fesom14_has_eta"]:
            print(f"  FESOM 1.4 data: no sea-surface height ({eta_name!r} is in "
                  "no file under data.ncinput): eta = 0 at every node")
        return ds

    @staticmethod
    def prepare(mesh: Mesh, ds: xr.Dataset) -> None:
        """Check a dataset against the mesh; change nothing.

        The engine calls this hook at every entry: ``run_tracking`` right
        after :meth:`open_dataset`, then each realization after the
        ``max_levels`` cut, and ``kelp info`` once. It only checks: neither
        the mesh (``sigma``, ``depth``, ``metadata``, ``uv_data_last``
        included) nor the dataset is changed, so a repeated call is
        harmless; each reads one record per variable. The static wet mask
        of the mesh is never re-declared from the data, only compared with
        it; the declaration ``uv_data_last`` stays the one :meth:`load_mesh`
        made (the engine puts a read-only copy with the same values in its
        place at its first use).

        Parameters
        ----------
        mesh : Mesh
            A mesh from :meth:`load_mesh`. The facts compared come from
            ``mesh.metadata``: the FULL level table ``fesom14_levels``
            (``mesh.sigma`` may already be cut by ``max_levels``), the node
            coordinates ``fesom14_lon``/``fesom14_lat``, the 3-D node index
            table ``fesom14_idx`` (ASCII) or the static wet mask
            ``fesom14_wet`` (cell polygons).
        ds : xarray.Dataset
            A dataset from :meth:`open_dataset`, or a time window of one.
            Every variable with a leading ``time`` dimension is checked.

        Raises
        ------
        ValueError
            In this order: the mesh is not from this reader; the dataset
            has no variable with a ``time`` dimension, or a variable's
            dimensions are not a FESOM 1.4 layout; a ``(time, nodes_3d)``
            vector on a mesh from cell polygons (it has no 3-D node table);
            a horizontal size other than the mesh's 2-D nodes (level-wise
            and 2-D fields) or 3-D nodes (``(time, nodes_3d)`` vectors); a
            level axis other than the mesh's full level table (in length, or
            by more than 1e-3 m at a level), or a level dimension without a
            coordinate; node coordinates ``lon``/``lat`` in the dataset that
            differ from the mesh's by more than 1e-6 deg (either longitude
            convention); a dataset without records; record 0 of a variable
            whose wet mask is not the mesh's: a level-wise record with a
            value in a dry slot or none in a wet one (the ASCII mesh's index
            table, or the mask a mesh from cell polygons was built with), a
            2-D field or a ``(time, nodes_3d)`` vector without a value at
            some node. The messages name the variable, its first file, the
            record (by its instant), the node and level, and the fix.

        Notes
        -----
        Only record 0 of each variable is read, lazily, one record per
        variable: the mask of a FESOM 1.4 run is static, and the reads
        compare every record with it. Memory: one variable at a time, at
        the peak about 11 bytes per slot of a 3-D variable (its float32
        record of ``K`` levels by ``N`` nodes, a copy of the magnitudes and
        three masks), ~65 MB for 46 levels and 127k nodes. A dataset whose
        nodes are ALL in another order than the mesh's (a common
        permutation) can have masks that agree with the mesh's (columns of
        equal length); only its node coordinates show it, so data without
        ``lon``/``lat`` cannot be checked for it.
        """
        facts = _mesh_facts(mesh)
        fields = _fields(ds)
        _check_sizes(fields, facts)
        _check_level_axes(ds, fields, facts)
        _check_mesh_coords(ds, fields, facts)
        _check_first_records(ds, fields, facts)

    @staticmethod
    def read_velocities(ds: xr.Dataset, timestep: int, mesh: Mesh,
                        mode_cfg: Optional["ModeConfig"] = None,
                        diffusion: Optional["DiffusionConfig"] = None,
                        sample_vars: Optional[Sequence[str]] = None,
                        velocity_float32: bool = False) -> VelocityField:
        """Read one record of u, v (and w in 3-D) and the sea-surface height.

        Every variable's record is first compared with the mesh's static wet
        mask, on every level: data in each wet slot, none in a dry one. Only
        then are the values changed: 0 below the bed, cut to the kept levels,
        u/v rotated into the tracking frame.

        Parameters
        ----------
        ds : xarray.Dataset
            A dataset from :meth:`open_dataset`, or a time window of one
            (a prefetch window counts its records from its own start).
        timestep : int
            The record, an index into ``ds.time``.
        mesh : Mesh
            The mesh from :meth:`load_mesh`; ``mesh.sigma`` holds the kept
            levels (``mode.max_levels`` cuts it before the run).
        mode_cfg : ModeConfig, optional
            ``Fdrift3D`` (3-D tracking; without a mode, 2-D), ``var_u``,
            ``var_v``, ``var_w``, ``var_eta`` (defaults ``uo``, ``vo``,
            ``wo``, ``zos``), ``w_positive``, and for 2-D tracking
            ``sigma_layer`` or ``sigma_mean``.
        diffusion : DiffusionConfig, optional
            Not read: FESOM 1.4 output carries no diffusivity fields.
        sample_vars : sequence of str, optional
            Variables read with :meth:`read_scalar_fields` into
            ``VelocityField.scalars``.
        velocity_float32 : bool
            Return u/v/w in float32 (eta and scalars stay float64).

        Returns
        -------
        VelocityField
            Node layout. **3-D**: u, v, w of shape ``(K, N)`` on the levels,
            ``K = mesh.sigma.size``, C-contiguous, ``uv_vertical="level"``,
            0 below each column's bed; w positive down (the files' w,
            positive up unless its ``positive`` attribute, ``standard_name``
            or ``mode.w_positive`` says otherwise, is negated). **2-D**: u, v
            of shape ``(N,)``, ``uv_vertical="layer"``, no w: the level
            ``mode.sigma_layer`` (a 0-based index of a kept level; 0 below
            the bed), else with ``mode.sigma_mean`` the depth mean from the
            surface to each node's bed (trapezoids between the kept wet
            levels, the last kept level held constant down to the bed when
            ``mode.max_levels`` cut the column; a column wet at the surface
            only gives its surface value), else level 0. With
            ``mesh.rotation``, u/v are in the tracking frame. ``eta``: the
            sea-surface height, float64 ``(N,)``; 0 everywhere when the
            files have none.

        Raises
        ------
        ValueError
            A record with no value in a wet slot (NaN, an infinity or a fill
            value) or a value in a dry slot, on any level, also below the
            kept levels; a vector over the 3-D nodes of another length than
            the mesh's 3-D nodes; a sea-surface height that is not one value
            per 2-D node, or lacks one at a node; ``mode.sigma_layer`` that
            is not a kept level; a variable missing from ``ds``; a mesh not
            built by this reader. The messages name the variable, the record
            (by its instant), the node and level, and the fix.
        """
        st = _statics(mesh)
        record = _instant(ds, timestep)
        is_3d = _is_3d(mode_cfg)
        names = _uvw_names(mode_cfg)
        vd = np.float32 if velocity_float32 else np.float64
        u_full = _record_kn(ds, names[0], "mode.var_u", timestep, st, record)
        v_full = _record_kn(ds, names[1], "mode.var_v", timestep, st, record)
        if is_3d:
            w_full = _record_kn(ds, names[2], "mode.var_w", timestep, st,
                                record)
            u = _kept(u_full, st, vd)
            v = _kept(v_full, st, vd)
            w = _kept(w_full, st, vd)
            if resolve_w_sign(ds[names[2]], mode_cfg, "up") < 0.0:
                # wet slots only: the zeros below the bed stay +0
                np.negative(w, out=w, where=st.wet[:st.k_kept])
            placement = "level"
        else:
            u = _reduce_2d(_zeroed(u_full, st), st, mesh, mode_cfg)
            v = _reduce_2d(_zeroed(v_full, st), st, mesh, mode_cfg)
            w = None
            placement = "layer"
        if getattr(mesh, "rotation_angles", None) is not None:
            # geographic east/north -> the tracking frame: the tangent map
            # (c, s) of each node, broadcast over the levels
            c, s = mesh._vec_rot_factors["node"]
            u, v = c * u - s * v, s * u + c * v
        u = np.ascontiguousarray(u, dtype=vd)
        v = np.ascontiguousarray(v, dtype=vd)
        eta = _read_eta(ds, timestep, mesh, mode_cfg, st, record)
        scalars = None
        if sample_vars:
            scalars = Fesom14Reader.read_scalar_fields(
                ds, timestep, mesh, list(sample_vars), mode_cfg)
        return VelocityField(u=u, v=v, eta=eta, w=w, uv_layout="node",
                             scalars=scalars, uv_vertical=placement)

    @staticmethod
    def read_scalar_fields(ds: xr.Dataset, timestep: int, mesh: Mesh,
                           varnames: Sequence[str],
                           mode_cfg: Optional["ModeConfig"] = None
                           ) -> Dict[str, np.ndarray]:
        """Read one record of variables sampled along the trajectories.

        Parameters
        ----------
        ds : xarray.Dataset
            A dataset from :meth:`open_dataset` (opened with these names in
            ``mode.sample_vars``), or a time window of one.
        timestep : int
            The record, an index into ``ds.time``.
        mesh : Mesh
            The mesh from :meth:`load_mesh` (``mesh.sigma`` = the kept
            levels).
        varnames : sequence of str
            The variables to read.
        mode_cfg : ModeConfig, optional
            ``Fdrift3D`` (without a mode, 2-D), ``sigma_layer``,
            ``sigma_mean``.

        Returns
        -------
        dict of str to numpy.ndarray
            float64, C-contiguous. A 3-D variable (level-wise, or a vector
            over the 3-D nodes): in 3-D tracking ``(K, N)`` on the kept
            levels, every slot below a column's bed holding the column's
            last value above it (forward-filled, so the sampler never meets
            a fill value at a shallow neighbour); in 2-D tracking ``(N,)``,
            reduced as u/v are (``mode.sigma_layer``, the depth mean with
            ``mode.sigma_mean``, else level 0) from the filled column. A 2-D
            variable ``(time, <horizontal>)``: ``(N,)``.

        Raises
        ------
        ValueError
            A variable missing from ``ds``, or not a FESOM 1.4 layout; a
            record with no value in a wet slot or a value in a dry slot (3-D
            variables, checked on every level before the fill and the cut),
            or without a value at some node (2-D variables); a vector over
            the 3-D nodes of another length than the mesh's 3-D nodes;
            ``mode.sigma_layer`` that is not a kept level. The messages
            name the variable, the record (by its instant), the node and
            level, and the fix.
        """
        st = _statics(mesh)
        record = _instant(ds, timestep)
        is_3d = _is_3d(mode_cfg)
        levelwise = _is_levelwise(ds)
        out: Dict[str, np.ndarray] = {}
        for name in varnames:
            if name not in ds.data_vars:
                raise ValueError(
                    f"fesom14: the sampled variable {name!r} (mode.sample_vars) "
                    "is not in the dataset; open_dataset opens the variables of "
                    "mode.sample_vars, so open the data with the same mode, and "
                    f"check that a file under data.ncinput holds {name!r}")
            layout = _classify(tuple(ds[name].dims), levelwise)
            if layout == "2d":
                out[name] = _record_2d(ds, name, "mode.sample_vars", timestep,
                                       st, record)
                continue
            full = _record_kn(ds, name, "mode.sample_vars", timestep, st,
                              record)
            filled = _filled(full, st)
            if is_3d:
                out[name] = np.ascontiguousarray(
                    filled[:st.k_kept] if st.k_kept < filled.shape[0]
                    else filled)
            else:
                out[name] = _reduce_2d(filled, st, mesh, mode_cfg)
        return out

    @staticmethod
    def probe_uv_layout(ds: xr.Dataset, mode_cfg: Optional["ModeConfig"] = None,
                        mesh: Optional[Mesh] = None) -> str:
        """The layout of u/v: always ``"node"`` (FESOM 1.4 velocities are
        nodal).

        Run before the output file is created. With a mesh, it also refuses
        from metadata alone (no record is read) what every read would
        refuse: a sea-surface height that is not one value per 2-D node, and
        in 2-D tracking a ``mode.sigma_layer`` that is not a kept level.

        Parameters
        ----------
        ds : xarray.Dataset
            A dataset from :meth:`open_dataset`.
        mode_cfg : ModeConfig, optional
            The run's mode.
        mesh : Mesh, optional
            The mesh from :meth:`load_mesh`, with the kept levels.

        Returns
        -------
        str
            ``"node"``.

        Raises
        ------
        ValueError
            See above (only with a mesh).
        """
        if mesh is not None:
            st = _statics(mesh)
            _eta_layout(ds, mode_cfg, st, _is_levelwise(ds))
            if not _is_3d(mode_cfg):
                _layer_index(mode_cfg, st)
        return "node"


# ---------------------------------------------------------------------------
# Common pieces
# ---------------------------------------------------------------------------


def _input_rotation(mesh_cfg: "MeshConfig") -> Optional[Angles]:
    """``mesh.input_rotation`` as three finite floats, or None."""
    # MeshConfig.__post_init__ has validated it already; this re-check only
    # serves duck-typed configs (any object with the attribute)
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


def _assemble(parts: _Parts, path: Path, tracking_angles: Optional[Angles],
              r_earth: float) -> Mesh:
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


def _from_ascii(meshdir: Path, input_angles: Optional[Angles]) -> _Parts:
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
        if values[n] > depth[n]:
            why = (f"{DEPTHO} is deeper: the data stop above the bed it gives, "
                   "so values are missing where it says there is ocean (a "
                   "record with missing values at depth, or a "
                   f"{DEPTHO} of another mesh or run). Check record 0 of "
                   f"{var_u!r}")
        else:
            why = ("Data written as 0 (not as a fill value) below the bed look "
                   "wet at every level, and their bed cannot be read from the "
                   "data; build the mesh from the FESOM 1.4 ASCII mesh instead")
        raise ValueError(
            f"fesom14: {DEPTHO} in {d_file} disagrees with the wet mask of "
            f"{var_u!r} (record 0 of {u_file}) at node {n}: {DEPTHO} = "
            f"{float(values[n]):g} m, but the last level with data is level "
            f"{int(last[n])} at {float(depth[n]):g} m ({int(off.sum())} node(s) "
            f"differ by more than {_DEPTHO_TOL:g} m). {why}. If {DEPTHO} "
            "belongs to another mesh or run, move its file out of "
            "mesh.meshpath")


# ---------------------------------------------------------------------------
# open_dataset: files, headers, consumed variables
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Head:
    """What the header of one data file says."""

    path: str
    dims: Dict[str, Tuple[str, ...]]    # variable -> its dimensions
    schedule: Optional[str]             # global attribute output_schedule


@dataclass(frozen=True)
class _Role:
    """One variable the run consumes."""

    name: str        # the variable's name in the files
    key: str         # the config key that names it (for messages)
    kind: str        # "u" | "v" | "w" | "eta" | "sample"
    required: bool
    eta: bool = False    # it is the sea-surface height (also when sampled)


def _data_files(sources: Sequence[str]) -> List[str]:
    """The data files of every source, each once, in the order found."""
    files: List[str] = []
    seen = set()
    for src in sources:
        for f in _expand(src):
            key = os.path.realpath(f)
            if key not in seen:
                seen.add(key)
                files.append(f)
    return files


def _expand(src: str) -> List[str]:
    """A file, the ``*.nc`` files of a directory, or the files of a glob."""
    text = os.path.expanduser(src.strip())
    if any(c in text for c in _GLOB_CHARS) and not os.path.exists(text):
        found = sorted(f for f in glob.glob(text) if os.path.isfile(f))
        if not found:
            raise FileNotFoundError(
                f"fesom14: the data pattern {src} matches no file; data.ncinput "
                "is a NetCDF file, a directory of *.nc files or a glob pattern")
        return found
    p = Path(text)
    if p.is_dir():
        found = sorted(str(f) for f in p.iterdir()
                       if f.suffix == ".nc" and f.is_file()
                       and not f.name.startswith("."))
        if not found:
            raise ValueError(
                f"fesom14: the data directory {p} holds no *.nc file; "
                "data.ncinput is a NetCDF file, a directory of *.nc files or a "
                "glob pattern")
        return found
    if p.is_file():
        return [str(p)]
    raise FileNotFoundError(
        f"fesom14: the data path {src} does not exist; data.ncinput is a NetCDF "
        "file, a directory of *.nc files or a glob pattern")


def _header(path: str) -> _Head:
    """Variables, their dimensions and ``output_schedule`` of one file."""
    import netCDF4

    try:
        with netCDF4.Dataset(path, "r") as nc:
            dims = {name: tuple(var.dimensions)
                    for name, var in nc.variables.items()}
            schedule = (nc.getncattr("output_schedule")
                        if "output_schedule" in nc.ncattrs() else None)
    except Exception as exc:     # netCDF4 raises OSError and others
        raise ValueError(
            f"fesom14: {path} cannot be opened as a NetCDF file ({exc}); every "
            "file under data.ncinput must be readable NetCDF (move other files "
            "out of the directory, or select the data files with a glob "
            "pattern)") from exc
    return _Head(path=path, dims=dims,
                 schedule=None if schedule is None else str(schedule))


def _consumed_roles(mode_cfg: Optional["ModeConfig"]) -> List[_Role]:
    """The variables the run consumes, u first, each name once."""
    def cfg(attr: str):
        return getattr(mode_cfg, attr, None) if mode_cfg is not None else None

    roles = [_Role(str(cfg("var_u") or DEFAULT_VAR_U), "mode.var_u", "u", True),
             _Role(str(cfg("var_v") or DEFAULT_VAR_V), "mode.var_v", "v", True)]
    if mode_cfg is None or bool(getattr(mode_cfg, "Fdrift3D", False)):
        # 3-D tracking needs w; a direct call without a mode takes it if present
        roles.append(_Role(str(cfg("var_w") or DEFAULT_VAR_W), "mode.var_w",
                           "w", mode_cfg is not None))
    eta = cfg("var_eta")
    roles.append(_Role(str(eta or DEFAULT_VAR_ETA), "mode.var_eta", "eta",
                       bool(eta), eta=True))
    for name in cfg("sample_vars") or ():
        roles.append(_Role(str(name), "mode.sample_vars", "sample", True))
    out: List[_Role] = []
    for r in roles:
        same = [i for i, o in enumerate(out) if o.name == r.name]
        if not same:
            out.append(r)
            continue
        first = out[same[0]]
        if r.required and not first.required:
            # the key that made it required speaks in the messages (e.g. zos
            # in mode.sample_vars); it stays the sea-surface height
            out[same[0]] = replace(r, eta=first.eta or r.eta)
        elif r.eta and not first.eta:
            out[same[0]] = replace(first, eta=True)
    return out


def _file_kind(head: _Head) -> Optional[str]:
    """``"restart"``, ``"older"`` (the older branch's mean output) or None."""
    if any(name in head.dims for name in _RESTART_VARS):
        return "restart"
    over_3d = [n for n in _OLDER_BRANCH_FIELDS
               if len(head.dims.get(n, ())) == 2 and head.dims[n][-1] == "nodes_3d"]
    if len(over_3d) >= 2 and head.schedule is None:
        return "older"
    return None


def _kind_error(head: _Head, kind: str, why: str) -> ValueError:
    """The refusal of a file KELP does not read."""
    if kind == "restart":
        marks = " and ".join(repr(n) for n in _RESTART_VARS if n in head.dims)
        return ValueError(
            f"fesom14: {head.path} is a FESOM 1.4 restart file (it holds "
            f"{marks}){why}. A restart file is a snapshot of the model state "
            "for continuing a run, with u and v in the rotated model frame; it "
            "is not mean output, and KELP does not read it as velocity input. "
            "Point data.ncinput at the mean output (<var>_<runid>_<YYYYMMDD>.nc, "
            "one variable per file) or at level-wise files")
    fields = ", ".join(n for n in (*_OLDER_BRANCH_FIELDS, "ssh") if n in head.dims)
    return ValueError(
        f"fesom14: {head.path} is mean output of the older branch of FESOM 1.4 "
        f"(<runid>.<year>.oce.mean.nc: {fields} together over the 3-D nodes, "
        f"no output_schedule attribute){why}. KELP does not read this format: "
        "its writer is not part of FESOM 1.4, so neither the frame of its u and "
        "v (geographic or the rotated model frame) nor what time each record "
        "represents can be checked. Point data.ncinput at the output of the "
        "current writer (<var>_<runid>_<YYYYMMDD>.nc, with output_schedule) or "
        "at level-wise files with time bounds")


_ROLE_WHAT = {"u": "the eastward velocity", "v": "the northward velocity",
              "w": "the vertical velocity of 3-D tracking",
              "eta": "the sea-surface height",
              "sample": "a variable to sample along the trajectories"}


def _holders(roles: List[_Role], heads: List[_Head],
             sources: List[str]) -> Dict[str, List[_Head]]:
    """The files holding each consumed variable; refusals for files KELP
    does not read and for required variables no file holds."""
    kinds = {h.path: _file_kind(h) for h in heads}
    holders = {r.name: [h for h in heads if r.name in h.dims] for r in roles}
    for r in roles:
        for h in holders[r.name]:
            if kinds[h.path]:
                raise _kind_error(h, kinds[h.path],
                                  f", and {r.key} selects its variable {r.name!r}")
    # a file KELP does not read that holds a consumed variable was refused
    # above, so such files never hold a missing variable: they are only noted
    foreign = [h for h in heads if kinds[h.path]]
    for r in roles:
        if holders[r.name] or not r.required:
            continue
        found = sorted({n for h in heads for n in h.dims})
        shown = ", ".join(found[:40]) + (", ..." if len(found) > 40 else "")
        note = ""
        if foreign:
            more = (f" ({len(foreign) - 1} more such file(s) are not read either)"
                    if len(foreign) > 1 else "")
            note = (" Not read: "
                    + _strip(_kind_error(foreign[0], kinds[foreign[0].path], ""))
                    + more)
        what = "the sea-surface height" if r.eta else _ROLE_WHAT[r.kind]
        raise ValueError(
            f"fesom14: no file under {', '.join(sources)} holds the variable "
            f"{r.name!r}, {what} ({r.key}); the files hold "
            f"{shown or 'no variable'}. Add its files to data.ncinput, or set "
            f"{r.key} to the name the files use.{note}")
    return holders


# ---------------------------------------------------------------------------
# open_dataset: layouts, levels, node coordinates
# ---------------------------------------------------------------------------


def _open_lazy(path: str) -> xr.Dataset:
    """A data file opened lazily: numbers undecoded in time, one record per
    chunk."""
    try:
        return xr.open_dataset(path, decode_times=False, decode_timedelta=False,
                               chunks={TIME_DIM: 1})
    except Exception as exc:     # xarray raises several types for "not readable"
        raise ValueError(f"fesom14: {path} cannot be opened as a NetCDF file "
                         f"({exc})") from exc


def _layout(role: _Role, path: str, da: xr.DataArray) -> str:
    """``"levelwise"``, ``"nodes3d"`` or ``"2d"`` (``(time, <horizontal>)``)."""
    dims = tuple(da.dims)
    is_2d = len(dims) == 2 and dims[0] == TIME_DIM and dims[1] in HORIZONTAL_DIMS
    if role.kind == "eta":
        if not is_2d:
            raise ValueError(
                f"fesom14: variable {role.name!r} in {path} has dimensions "
                f"{dims}; the sea-surface height must be (time, <horizontal>) "
                f"with <horizontal> one of {HORIZONTAL_DIMS}. {_role_hint(role)}")
        return "2d"
    if role.kind == "sample" and dims != (TIME_DIM, "nodes_3d") and is_2d:
        return "2d"
    try:
        return detect_form(da)
    except ValueError as exc:
        extra = (" A sampled variable is 3-D like the velocities or 2-D "
                 f"(time, <horizontal>). {_role_hint(role)}."
                 if role.kind == "sample" else "")
        raise ValueError(f"fesom14: {path} ({role.key}): {_strip(exc)}{extra}") from exc


def _describe_layout(da: xr.DataArray) -> str:
    return "(" + ", ".join(f"{d}={n}" for d, n in zip(da.dims, da.shape)) + ")"


def _check_layouts(roles: List[_Role],
                   files: Dict[str, List[Tuple[_Head, xr.Dataset]]]) -> str:
    """Every variable has one layout across its files; every 3-D variable the
    layout of u; shared dimension names have one size. Returns the form."""
    u = roles[0].name
    first: Dict[str, Tuple[str, xr.DataArray, str]] = {}   # name -> (path, da, layout)
    for r in roles:
        for h, ds in files[r.name]:
            da = ds[r.name]
            lay = _layout(r, h.path, da)
            if r.name not in first:
                first[r.name] = (h.path, da, lay)
                continue
            p0, da0, _ = first[r.name]
            if da.dims != da0.dims or da.shape[1:] != da0.shape[1:]:
                raise ValueError(
                    f"fesom14: variable {r.name!r} has the layout "
                    f"{_describe_layout(da0)} in {p0} but {_describe_layout(da)} "
                    f"in {h.path} (record counts aside); all files of one "
                    "variable must have the same layout: leave out the files of "
                    "another run or mesh")
    pu, dau, form = first[u]
    for r in roles[1:]:
        p, da, lay = first[r.name]
        if lay == "2d":
            continue
        if da.dims != dau.dims or da.shape[1:] != dau.shape[1:]:
            raise ValueError(
                f"fesom14: variable {r.name!r} ({p}) has the layout "
                f"{_describe_layout(da)}, but {u!r} ({pu}) has "
                f"{_describe_layout(dau)} (record counts aside); every 3-D "
                "variable must have the layout of the velocities: leave out the "
                "files of another form, run or mesh")
    sizes: Dict[str, Tuple[int, str, str]] = {}
    for r in roles:
        p, da, lay = first[r.name]
        for d, n in zip(da.dims[1:], da.shape[1:]):
            if d not in sizes:
                sizes[d] = (n, r.name, p)
            elif sizes[d][0] != n:
                n0, name0, p0 = sizes[d]
                raise ValueError(
                    f"fesom14: the dimension {d!r} has {n0} entries for {name0!r} "
                    f"({p0}) but {n} for {r.name!r} ({p}); variables that share a "
                    "dimension name must share its size: leave out the files of "
                    "another mesh or layout")
        if form == "levelwise" and lay == "2d" and da.shape[-1] != dau.shape[-1]:
            raise ValueError(
                f"fesom14: variable {r.name!r} ({p}) has {da.shape[-1]} nodes "
                f"({da.dims[-1]!r}), but the level-wise {u!r} ({pu}) has "
                f"{dau.shape[-1]} ({dau.dims[-1]!r}); a 2-D field holds one value "
                f"per node of the same mesh. "
                + _role_hint(r, "Leave out the files of another mesh"))
    return form


def _check_levels(roles: List[_Role],
                  files: Dict[str, List[Tuple[_Head, xr.Dataset]]]
                  ) -> Optional[Tuple[str, np.ndarray, dict]]:
    """Every file of every level-wise 3-D variable is on the levels of the
    first u file. Returns (dimension, levels, attributes) or None."""
    ref = None
    for r in roles:
        for h, ds in files[r.name]:
            da = ds[r.name]
            if da.ndim != 3:
                continue
            zdim = da.dims[1]
            if zdim not in ds.variables:
                raise ValueError(
                    f"fesom14: the level dimension {zdim!r} of {r.name!r} in "
                    f"{h.path} has no coordinate variable; the reader needs the "
                    "level depths (m, positive down) of every file")
            z = np.asarray(ds[zdim].values, dtype=np.float64)
            if ref is None:
                ref = (r.name, h.path, z, dict(ds[zdim].attrs))
                continue
            name0, p0, z0, _ = ref
            off = np.flatnonzero(~(np.abs(z - z0) <= _LEVEL_TOL))
            if off.size:
                k = int(off[0])
                raise ValueError(
                    f"fesom14: the levels of {r.name!r} in {h.path} differ from "
                    f"those of {name0!r} in {p0} at level {k}: {float(z[k]):g} m "
                    f"against {float(z0[k]):g} m ({off.size} level(s) differ by "
                    f"more than {_LEVEL_TOL:g} m); every 3-D variable and file "
                    "must be on the same levels: leave out the files of another "
                    "run or mesh")
    if ref is None:
        return None
    zdim = files[roles[0].name][0][1][roles[0].name].dims[1]
    return zdim, ref[2], ref[3]


def _check_node_coords(roles: List[_Role],
                       files: Dict[str, List[Tuple[_Head, xr.Dataset]]]
                       ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Node coordinates ``lon``/``lat`` (where a file carries them over the
    variable's horizontal dimension) agree between every two files of the
    same number of nodes within 1e-6 deg, either longitude convention.
    Returns the coordinates by number of nodes."""
    refs: Dict[int, Tuple[np.ndarray, np.ndarray, str, str]] = {}
    for r in roles:
        for h, ds in files[r.name]:
            hdim = ds[r.name].dims[-1]
            if not all(c in ds.variables and ds[c].dims == (hdim,)
                       for c in ("lon", "lat")):
                continue
            lon = np.asarray(ds["lon"].values, dtype=np.float64)
            lat = np.asarray(ds["lat"].values, dtype=np.float64)
            if lon.size not in refs:
                refs[lon.size] = (lon, lat, r.name, h.path)
                continue
            lon0, lat0, name0, p0 = refs[lon.size]
            off = ~((_lon_diff(lon, lon0) <= _COORD_TOL)
                    & (np.abs(lat - lat0) <= _COORD_TOL))
            if off.any():
                n = int(np.argmax(off))
                raise ValueError(
                    f"fesom14: the node coordinates of {r.name!r} in {h.path} "
                    f"differ from those of {name0!r} in {p0} at node {n}: "
                    f"({float(lon[n]):.7f}, {float(lat[n]):.7f}) against "
                    f"({float(lon0[n]):.7f}, {float(lat0[n]):.7f}) "
                    f"({int(off.sum())} node(s) differ by more than "
                    f"{_COORD_TOL:g} deg, the longitude convention aside). Every "
                    "variable must list the same nodes in the same order: leave "
                    "out the files of another mesh or run")
    return {n: (v[0], v[1]) for n, v in refs.items()}


# ---------------------------------------------------------------------------
# open_dataset: time
# ---------------------------------------------------------------------------


def _role_hint(role: _Role, velocities: str = "") -> str:
    """How to get rid of a disagreeing variable; ``velocities`` is the fix
    when it is a velocity."""
    if role.kind == "eta":
        return (f"{role.name!r} is optional: leave the {role.name!r} files out of "
                "data.ncinput (the sea-surface height is then 0), or give it the "
                "records of the velocities")
    if role.kind == "sample" and role.eta:
        return (f"Remove {role.name!r} from mode.sample_vars and leave its files "
                "out of data.ncinput (the sea-surface height is then 0), or give "
                "it the records of the velocities")
    if role.kind == "sample":
        return (f"Remove {role.name!r} from mode.sample_vars, or give it the "
                "records of the velocities")
    return velocities


def _file_axis(head: _Head, ds: xr.Dataset, name: str) -> FileAxis:
    """The undecoded time axis of one file of one variable."""
    who = f"{head.path} (variable {name!r})"
    if TIME_DIM not in ds.variables:
        raise ValueError(
            f"fesom14: {who}: the dimension {TIME_DIM!r} has no coordinate "
            "variable, so its records carry no time stamps; repair the file or "
            "leave it out")
    tv = ds[TIME_DIM]
    bounds = None
    bname = tv.attrs.get("bounds")
    if bname:
        if bname not in ds.variables:
            raise ValueError(
                f"fesom14: {who}: the time axis names its bounds {bname!r}, "
                "which the file does not hold; repair the file, or set "
                "data.time_stamp and data.time_window")
        bounds = np.asarray(ds[bname].values)
    return FileAxis(values=np.asarray(tv.values),
                    units=str(tv.attrs.get("units", "")),
                    calendar=str(tv.attrs.get("calendar", "")),
                    bounds=bounds, schedule=head.schedule, path=head.path)


def _first_stamp(axis: FileAxis) -> Tuple[int, ...]:
    """The fields of a file's first decoded stamp (the order resolution
    uses: files of one variable are in one calendar by then)."""
    v = float(np.ma.getdata(np.asarray(axis.values)).ravel()[0])
    t = cftime.num2date(v, axis.units, calendar=_canonical_calendar(axis.calendar),
                        only_use_cftime_datetimes=True)
    return (t.year, t.month, t.day, t.hour, t.minute, t.second, t.microsecond)


def _resolve(role: _Role, files: List[Tuple[_Head, xr.Dataset]],
             opts: TimeOptions) -> Tuple[ResolvedAxis, List[Tuple[_Head, xr.Dataset]]]:
    """The resolved axis of one variable and its files with records, in
    time order."""
    axes = [_file_axis(h, ds, role.name) for h, ds in files]
    try:
        ax = resolve_axis(axes, opts, variable=role.name)
    except ValueError as exc:
        hint = _role_hint(role)
        hint = f" ({hint})" if hint else ""
        raise ValueError(f"fesom14: {exc}{hint}") from exc
    keep = [i for i, a in enumerate(axes) if np.size(a.values)]
    keep.sort(key=lambda i: _first_stamp(axes[i]))
    ordered = [files[i] for i in keep]
    n_records = sum(ds[role.name].shape[0] for _, ds in ordered)
    if n_records != len(ax.instants):     # resolve_axis gives one per record
        raise RuntimeError(f"fesom14: {role.name!r}: {n_records} records but "
                           f"{len(ax.instants)} resolved instants")
    return ax, ordered


def _schedules(files: List[Tuple[_Head, xr.Dataset]]) -> Dict[tuple, str]:
    """The output schedules of a variable's files: parsed -> as written."""
    out: Dict[tuple, str] = {}
    for h, _ in files:
        if h.schedule is not None:
            out.setdefault(parse_output_schedule(h.schedule), h.schedule)
    return out


def _sub(ax: ResolvedAxis, a: int, b: int) -> ResolvedAxis:
    return ResolvedAxis(instants=ax.instants[a:b],
                        windows=None if ax.windows is None else ax.windows[a:b],
                        source=ax.source)


def _start_in(big: ResolvedAxis, small: ResolvedAxis) -> Optional[int]:
    """Where ``small`` starts in ``big`` when it is a contiguous run of it."""
    n = len(small.instants)
    if n == 0 or n > len(big.instants):
        return None
    head = _sub(small, 0, 1)
    for a in range(len(big.instants) - n + 1):
        if first_difference(_sub(big, a, a + 1), head) is None:
            return a if first_difference(_sub(big, a, a + n), small) is None else None
    return None


def _span(ax: ResolvedAxis, a: int, b: int) -> str:
    """Records ``a .. b-1`` of an axis as an interval."""
    if ax.windows is not None:
        s, e = ax.windows[a, 0], ax.windows[b - 1, 1]
    else:
        s, e = ax.instants[a], ax.instants[b - 1]
    n = b - a
    return f"{s.isoformat()} to {e.isoformat()} ({n} record{'s' if n != 1 else ''})"


def _record_text(ax: ResolvedAxis, i: int) -> str:
    if i >= len(ax.instants):
        return "no record"
    s = ax.instants[i].isoformat()
    if ax.windows is not None:
        return (f"{s} (window {ax.windows[i, 0].isoformat()} – "
                f"{ax.windows[i, 1].isoformat()})")
    return f"{s} (a snapshot)"


def _misaligned(ref: _Role, rax: ResolvedAxis, role: _Role,
                ax: ResolvedAxis, schedules: str = "") -> ValueError:
    """The refusal of a variable whose records differ from those of u;
    ``schedules`` names their output schedules when those differ."""
    rule = ("every variable KELP reads must have the same records (the same "
            "instants and windows)")
    if schedules:
        rule = f"{schedules}; {rule}"
    hint = _role_hint(role, (
        "u, v and w must come from output written on one schedule" if schedules
        else "Add the missing files, or leave out the files that only some "
             "variables have"))
    for big, bax, small, sax in ((ref.name, rax, role.name, ax),
                                 (role.name, ax, ref.name, rax)):
        a = _start_in(bax, sax)
        if a is None:
            continue
        end = a + len(sax.instants)
        gaps = ([_span(bax, 0, a)] if a > 0 else []) + (
            [_span(bax, end, len(bax.instants))] if end < len(bax.instants) else [])
        return ValueError(
            f"fesom14: variable {small!r} has no record from "
            f"{', nor from '.join(gaps)}, where {big!r} has records; {rule}. {hint}")
    i = first_difference(rax, ax)
    if not schedules:
        hint = _role_hint(role, "Check the time axes (stamps and bounds) of "
                                "the files of both variables")
    # ResolvedAxis.calendar is cftime's canonical name ("gregorian" comes back
    # as "standard", "365_day" as "noleap"), so this set test is exact
    cals = {rax.calendar, ax.calendar}
    calendars = ""
    if len(cals) > 1 and not cals <= {"standard", "proleptic_gregorian"}:
        calendars = (f" ({ref.name!r} is in the {rax.calendar} calendar, "
                     f"{role.name!r} in the {ax.calendar} calendar: dates of "
                     "different calendars are different instants)")
    return ValueError(
        f"fesom14: variables {ref.name!r} and {role.name!r} do not have the same "
        f"records: the first that differs is {_record_text(rax, i)} in "
        f"{ref.name!r} against {_record_text(ax, i)} in {role.name!r}"
        f"{calendars}; {rule}. {hint}")


def _align(roles: List[_Role], axes: Dict[str, ResolvedAxis],
           files: Dict[str, List[Tuple[_Head, xr.Dataset]]]) -> None:
    """Every consumed variable has exactly the records of u, compared by
    value (instants and windows). Output schedules written differently but
    giving the same events (``first: 0`` and ``first: 1``) are the same; when
    the records differ, schedules that differ are named in the refusal."""
    ref = roles[0]
    rax = axes[ref.name]
    for r in roles[1:]:
        ax = axes[r.name]
        if first_difference(rax, ax) is None:
            continue
        schedules = ""
        if rax.source == "schedule" and ax.source == "schedule":
            s_ref, s_r = _schedules(files[ref.name]), _schedules(files[r.name])
            if set(s_ref) != set(s_r):
                def show(s, name):
                    return (", ".join(repr(t) for t in s.values()) + " in "
                            + files[name][0][0].path)
                schedules = (f"they are written on different output schedules: "
                             f"{ref.name!r} on {show(s_ref, ref.name)} / "
                             f"{r.name!r} on {show(s_r, r.name)}")
        raise _misaligned(ref, rax, r, ax, schedules)


# ---------------------------------------------------------------------------
# open_dataset: the dataset
# ---------------------------------------------------------------------------


def _close_all(datasets: List[xr.Dataset]) -> None:
    for ds in datasets:
        ds.close()


def _open_consumed(roles: List[_Role], holders: Dict[str, List[_Head]],
                   opts: TimeOptions, opened: Dict[str, xr.Dataset]) -> xr.Dataset:
    """Check, resolve, align and assemble the consumed variables (the files
    opened go into ``opened``)."""
    files: Dict[str, List[Tuple[_Head, xr.Dataset]]] = {}
    for r in roles:
        files[r.name] = []
        for h in holders[r.name]:
            if h.path not in opened:
                opened[h.path] = _open_lazy(h.path)
            files[r.name].append((h, opened[h.path]))
    form = _check_layouts(roles, files)
    levels = _check_levels(roles, files)
    coords_by_size = _check_node_coords(roles, files)

    axes: Dict[str, ResolvedAxis] = {}
    for r in roles:
        axes[r.name], files[r.name] = _resolve(r, files[r.name], opts)
    _align(roles, axes, files)

    u = roles[0].name
    data_vars = {}
    for r in roles:
        parts = [ds[r.name].drop_vars(list(ds[r.name].coords))
                 for _, ds in files[r.name]]
        data_vars[r.name] = (parts[0] if len(parts) == 1 else xr.concat(
            parts, dim=TIME_DIM, coords="minimal", compat="override",
            join="override", combine_attrs="override"))
    coords = {TIME_DIM: (TIME_DIM, axes[u].instants)}
    if levels is not None:
        zdim, z, zattrs = levels
        coords[zdim] = (zdim, z, zattrs)
    hdim = data_vars[u].dims[-1]
    node_coords = coords_by_size.get(data_vars[u].sizes[hdim])
    if node_coords is not None:
        coords["lon"] = (hdim, node_coords[0])
        coords["lat"] = (hdim, node_coords[1])
    ds = xr.Dataset(data_vars, coords=coords, attrs={
        "fesom14_form": form,
        "fesom14_has_eta": int(any(r.eta for r in roles)),
        "fesom14_time_source": axes[u].source,
    })
    ds.set_close(partial(_close_all, list(opened.values())))
    return ds


# ---------------------------------------------------------------------------
# prepare: the dataset against the mesh
# ---------------------------------------------------------------------------


_SAME_RUN = ("The data must be output of a run on this mesh: point data.ncinput "
             "at that output, or mesh.meshpath at the mesh of these data")


@dataclass(frozen=True)
class _MeshFacts:
    """What ``prepare`` compares a dataset with, from ``mesh.metadata``."""

    levels: np.ndarray              # (K,) the full level table
    lon: np.ndarray                 # (N,) geographic, [-180, 180)
    lat: np.ndarray
    wet: np.ndarray                 # (K, N) the static wet mask
    mask_text: str                  # where the mask comes from (messages)
    idx: Optional[np.ndarray]       # (N, K) ASCII only, -1 = dry
    n3d: int                        # number of 3-D nodes (0 without a table)


@dataclass(frozen=True)
class _Field:
    """One time-dependent variable of a dataset, as ``prepare`` sees it."""

    name: str
    layout: str                     # "levelwise" | "nodes3d" | "2d"
    hdim: str
    size: int                       # entries along hdim
    zdim: Optional[str]             # level-wise only
    where: str                      # " (first file ...)" or ""


def _mesh_facts(mesh: Mesh) -> _MeshFacts:
    """The facts of a mesh from :meth:`Fesom14Reader.load_mesh`."""
    md = getattr(mesh, "metadata", None) or {}
    keys = ["fesom14_source", "fesom14_levels", "fesom14_wet", "fesom14_lon",
            "fesom14_lat"]
    if md.get("fesom14_source") == "ascii":
        keys.append("fesom14_idx")
    missing = [k for k in keys if k not in md]
    if missing:
        raise ValueError(
            f"fesom14: the mesh has no {missing[0]!r} in mesh.metadata; the "
            "data are checked against a mesh built by the fesom14 reader's "
            "load_mesh (mesh.type: fesom14). Build the mesh with that reader")
    if md["fesom14_source"] == "ascii":
        idx = np.asarray(md["fesom14_idx"])
        wet = (idx >= 0).T
        return _MeshFacts(
            levels=np.asarray(md["fesom14_levels"], np.float64),
            lon=np.asarray(md["fesom14_lon"], np.float64),
            lat=np.asarray(md["fesom14_lat"], np.float64), wet=wet,
            mask_text="the mesh's 3-D node index table (the ASCII mesh)",
            idx=idx, n3d=int(np.count_nonzero(wet)))
    mask_file = md.get("fesom14_mask_file")
    return _MeshFacts(
        levels=np.asarray(md["fesom14_levels"], np.float64),
        lon=np.asarray(md["fesom14_lon"], np.float64),
        lat=np.asarray(md["fesom14_lat"], np.float64),
        wet=np.asarray(md["fesom14_wet"], dtype=bool),
        mask_text=("the mesh's static wet mask (record 0 of the u variable in "
                   f"{mask_file}, the file the mesh was built from)"),
        idx=None, n3d=0)


def _where(da: xr.DataArray) -> str:
    """`` (first file <path>)`` when the variable's encoding names it."""
    src = da.encoding.get("source")
    return f" (first file {src})" if src else ""


def _is_levelwise(ds: xr.Dataset) -> bool:
    """True when a variable of the dataset is level-wise ``(time, <depth>,
    <horizontal>)``."""
    return any(ds[n].ndim == 3 and tuple(ds[n].dims[:1]) == (TIME_DIM,)
               for n in ds.data_vars)


def _classify(dims: Tuple[str, ...], levelwise: bool) -> Optional[str]:
    """The layout of a variable from its dimensions: ``"levelwise"``,
    ``"nodes3d"``, ``"2d"``, or None for none of them.

    A ``(time, nodes_3d)`` variable is a vector over the 3-D nodes, unless
    the dataset is level-wise (``levelwise``): there it is a 2-D field over a
    horizontal dimension named ``nodes_3d`` (the level-wise writer's name).
    """
    if not dims or dims[0] != TIME_DIM:
        return None
    if len(dims) == 3 and dims[1] in DEPTH_DIMS and dims[2] in HORIZONTAL_DIMS:
        return "levelwise"
    if dims == (TIME_DIM, "nodes_3d") and not levelwise:
        return "nodes3d"
    if len(dims) == 2 and dims[1] in HORIZONTAL_DIMS:
        return "2d"
    return None


def _fields(ds: xr.Dataset) -> List[_Field]:
    """The variables with a leading ``time`` dimension and their layouts
    (:func:`_classify`)."""
    names = [n for n in ds.data_vars if tuple(ds[n].dims[:1]) == (TIME_DIM,)]
    if not names:
        raise ValueError(
            f"fesom14: the dataset holds no variable with a {TIME_DIM!r} "
            "dimension; the data are checked against the mesh as opened by "
            "the fesom14 reader's open_dataset")
    levelwise = _is_levelwise(ds)
    out: List[_Field] = []
    for name in names:
        da = ds[name]
        dims = tuple(da.dims)
        where = _where(da)
        layout = _classify(dims, levelwise)
        if layout is not None:
            out.append(_Field(name, layout, dims[-1], da.sizes[dims[-1]],
                              dims[1] if layout == "levelwise" else None,
                              where))
        else:
            raise ValueError(
                f"fesom14: variable {name!r}{where} has dimensions {dims}, which "
                f"is not a FESOM 1.4 layout: ('{TIME_DIM}', <depth>, "
                f"<horizontal>) level-wise, ('{TIME_DIM}', 'nodes_3d') over the "
                f"3-D nodes, or ('{TIME_DIM}', <horizontal>) for a 2-D field, "
                f"with <depth> one of {DEPTH_DIMS} and <horizontal> one of "
                f"{HORIZONTAL_DIMS}. Open the data with the fesom14 reader's "
                "open_dataset")
    return out


def _check_sizes(fields: List[_Field], facts: _MeshFacts) -> None:
    """Vectors over the 3-D nodes need the ASCII mesh's table and have one
    entry per 3-D node; other fields have one value per 2-D node."""
    for f in fields:
        if f.layout == "nodes3d" and facts.idx is None:
            raise ValueError(
                f"fesom14: variable {f.name!r}{f.where} is one vector over the "
                "3-D nodes (dimensions (time, nodes_3d)), but the mesh was built "
                "from cell polygons, which give no 3-D node table to place its "
                "entries on the levels. Set mesh.meshpath to the FESOM 1.4 ASCII "
                f"mesh directory of the run (holding {', '.join(MESH_FILES)})")
    n_nodes = facts.lon.size
    for f in fields:
        if f.layout == "nodes3d":
            if f.size != facts.n3d:
                also = (f" ({f.size} is the number of 2-D nodes: a field of one "
                        "value per 2-D node is not a vector over the 3-D nodes)"
                        if f.size == n_nodes else "")
                raise ValueError(
                    f"fesom14: variable {f.name!r}{f.where} is one vector over "
                    f"the 3-D nodes with {f.size} entries, but the ASCII mesh has "
                    f"{facts.n3d} 3-D nodes{also}. {_SAME_RUN}")
        elif f.size != n_nodes:
            raise ValueError(
                f"fesom14: variable {f.name!r}{f.where} has {f.size} nodes along "
                f"{f.hdim!r}, but the mesh has {n_nodes}. {_SAME_RUN}")


def _level_span(z: np.ndarray) -> str:
    return f"{float(z[0]):g} to {float(z[-1]):g} m" if z.size else "none"


def _check_level_axes(ds: xr.Dataset, fields: List[_Field],
                      facts: _MeshFacts) -> None:
    """The level axis of every level-wise variable is the mesh's FULL level
    table, within ``_LEVEL_TOL``."""
    lev = facts.levels
    seen = set()
    for f in fields:
        if f.layout != "levelwise" or f.zdim in seen:
            continue
        seen.add(f.zdim)
        if f.zdim not in ds.variables:
            raise ValueError(
                f"fesom14: the level dimension {f.zdim!r} of variable "
                f"{f.name!r}{f.where} has no coordinate in the dataset; the "
                "level depths of the data (m, positive down) are compared with "
                "the mesh's. Open the data with the fesom14 reader's "
                "open_dataset")
        z = np.asarray(ds[f.zdim].values, dtype=np.float64).ravel()
        if z.shape != lev.shape:
            raise ValueError(
                f"fesom14: variable {f.name!r}{f.where} has {z.size} levels along "
                f"{f.zdim!r} ({_level_span(z)}), but the mesh has {lev.size} "
                f"({_level_span(lev)}); the data must hold every level of the "
                "mesh's table, whatever mode.max_levels keeps. " + _SAME_RUN)
        off = np.flatnonzero(~(np.abs(z - lev) <= _LEVEL_TOL))
        if off.size:
            k = int(off[0])
            raise ValueError(
                f"fesom14: the levels of variable {f.name!r}{f.where} differ "
                f"from the mesh's at level {k}: {float(z[k]):g} m against "
                f"{float(lev[k]):g} m ({off.size} level(s) differ by more than "
                f"{_LEVEL_TOL:g} m). " + _SAME_RUN)


def _owners(idx: np.ndarray, n3d: int) -> Tuple[np.ndarray, np.ndarray]:
    """The 2-D node and the level of every 3-D node of an index table."""
    nodes, levels = np.nonzero(idx >= 0)
    node_of = np.empty(n3d, dtype=np.int64)
    level_of = np.empty(n3d, dtype=np.int64)
    node_of[idx[nodes, levels]] = nodes
    level_of[idx[nodes, levels]] = levels
    return node_of, level_of


def _check_mesh_coords(ds: xr.Dataset, fields: List[_Field],
                       facts: _MeshFacts) -> None:
    """Node coordinates ``lon``/``lat`` over a horizontal dimension of the
    data equal the mesh's within ``_COORD_TOL`` (either longitude
    convention); over ``nodes_3d`` (vectors over the 3-D nodes) each entry
    has the coordinates of its 2-D node."""
    seen = set()
    for f in fields:
        if f.hdim in seen:
            continue
        seen.add(f.hdim)
        if not all(c in ds.variables and tuple(ds[c].dims) == (f.hdim,)
                   for c in ("lon", "lat")):
            continue
        lon = np.asarray(ds["lon"].values, dtype=np.float64)
        lat = np.asarray(ds["lat"].values, dtype=np.float64)
        node_of = level_of = None
        if f.layout == "nodes3d":
            node_of, level_of = _owners(facts.idx, facts.n3d)
            ref_lon, ref_lat = facts.lon[node_of], facts.lat[node_of]
        else:
            ref_lon, ref_lat = facts.lon, facts.lat
        off = ~((_lon_diff(lon, ref_lon) <= _COORD_TOL)
                & (np.abs(lat - ref_lat) <= _COORD_TOL))
        if not off.any():
            continue
        i = int(np.argmax(off))
        at = (f"3-D node {i} (node {int(node_of[i])}, level {int(level_of[i])})"
              if node_of is not None else f"node {i}")
        raise ValueError(
            f"fesom14: the node coordinates of the data (lon, lat over "
            f"{f.hdim!r}; variable {f.name!r}{f.where}) differ from the mesh's "
            f"at {at}: ({float(lon[i]):.7f}, {float(lat[i]):.7f}) against "
            f"({float(ref_lon[i]):.7f}, {float(ref_lat[i]):.7f}) "
            f"({int(off.sum())} node(s) differ by more than {_COORD_TOL:g} deg, "
            "the longitude convention aside). The data must list the mesh's "
            "nodes in the mesh's order; data of this mesh with the nodes in "
            "another order pass every other check. " + _SAME_RUN)


def _record_label(ds: xr.Dataset) -> str:
    """Record 0 of a dataset, named by its instant."""
    if TIME_DIM not in ds.variables:
        return "0 (the first of the dataset)"
    t = ds[TIME_DIM].values[0]
    text = t.isoformat() if hasattr(t, "isoformat") else str(t)
    return f"{text} (the first of the dataset)"


def _check_first_records(ds: xr.Dataset, fields: List[_Field],
                         facts: _MeshFacts) -> None:
    """Record 0 of every variable has the mesh's static wet mask: one record
    per variable is read."""
    if ds.sizes.get(TIME_DIM, 0) == 0:
        raise ValueError(
            f"fesom14: the dataset has no record (its {TIME_DIM!r} dimension is "
            "empty); record 0 of every variable is read to compare its wet mask "
            "with the mesh's. Give the run data with records")
    rec = _record_label(ds)
    for f in fields:
        first = np.asarray(ds[f.name].isel({TIME_DIM: 0}).values)
        data = is_data(first)
        head = f"fesom14: variable {f.name!r}, record {rec}{f.where}"
        if f.layout == "levelwise":
            _check_mask_kn(head, first, data, facts)
            continue
        missing = ~data
        if not missing.any():
            continue
        i = int(np.argmax(missing))
        fix = ("Check the file; leave the data of another mesh or run out of "
               "data.ncinput")
        if f.layout == "nodes3d":
            node_of, level_of = _owners(facts.idx, facts.n3d)
            raise ValueError(
                f"{head}: no value ({MISSING_TEXT}) at 3-D node {i} (node "
                f"{int(node_of[i])}, level {int(level_of[i])}); "
                f"{int(missing.sum())} 3-D node(s) without one. Every 3-D node "
                "of the mesh is wet, so a vector over the 3-D nodes holds a value "
                f"in every entry. {fix}")
        raise ValueError(
            f"{head}: no value ({MISSING_TEXT}) at node {i}; "
            f"{int(missing.sum())} node(s) without one. A 2-D field holds a "
            "value at every node of the mesh, which is wet at the surface "
            f"everywhere. {fix}")


def _check_mask_kn(head: str, first: np.ndarray, data: np.ndarray,
                   facts: _MeshFacts) -> None:
    """The wet mask of a level-wise record equals the mesh's."""
    wet = facts.wet
    if np.array_equal(data, wet):
        return
    missing = wet & ~data
    extra = data & ~wet
    n, k = first_node_level(missing | extra)
    mesh_state = "wet" if wet[k, n] else "dry (below the bed)"
    data_state = (f"the value {float(first[k, n]):g}" if data[k, n]
                  else f"no value ({MISSING_TEXT})")
    if extra.any() and np.all(first[extra] == 0):
        why = ("The data are written as 0 below the bed (not as a fill value "
               "such as NaN or 1e30): such a record looks wet at every level, "
               "and a missing value cannot be told from the bed; FESOM 1.4 "
               "level-wise output holds a fill value below the bed. Check how "
               "the files were written. " + _SAME_RUN)
    else:
        why = ("Every variable and record holds data in exactly the wet slots "
               "of the mesh's static mask. " + _SAME_RUN)
    raise ValueError(
        f"{head}: its wet mask differs from {facts.mask_text} at node {n}, "
        f"level {k} (0-based): the mesh: {mesh_state}, the data: {data_state} "
        f"({int(missing.sum())} wet slot(s) without data, {int(extra.sum())} "
        f"dry slot(s) with data). {why}")


# ---------------------------------------------------------------------------
# Reads: velocities, sampled scalars, the sea-surface height
# ---------------------------------------------------------------------------


_RECORD_FIX = ("Replace the file holding this record (it is damaged, or of "
               "another run or mesh), or keep the run's time window clear of "
               "this record")


@dataclass(frozen=True)
class _Statics:
    """What the reads take from the mesh."""

    levels: np.ndarray              # (K_full,) the full level table
    wet: np.ndarray                 # (K_full, N) the static wet mask
    idx: Optional[np.ndarray]       # (N, K_full) ASCII only, -1 = dry
    k_kept: int                     # mesh.sigma.size (max_levels may cut it)
    n_nodes: int


def _statics(mesh: Mesh) -> _Statics:
    """The static facts of a mesh from :meth:`Fesom14Reader.load_mesh`."""
    md = getattr(mesh, "metadata", None) or {}
    missing = [k for k in ("fesom14_levels", "fesom14_wet") if k not in md]
    if missing:
        raise ValueError(
            f"fesom14: the mesh has no {missing[0]!r} in mesh.metadata; the "
            "data are read on a mesh built by the fesom14 reader's load_mesh "
            "(mesh.type: fesom14). Build the mesh with that reader")
    levels = np.asarray(md["fesom14_levels"], dtype=np.float64)
    wet = np.asarray(md["fesom14_wet"], dtype=bool)
    k_kept = int(np.asarray(mesh.sigma).size)
    if not 1 <= k_kept <= levels.size:
        raise ValueError(
            f"fesom14: mesh.sigma holds {k_kept} levels, but the mesh's level "
            f"table has {levels.size}; mesh.sigma keeps the first 1 .. "
            f"{levels.size} of them (mode.max_levels cuts it)")
    return _Statics(levels=levels, wet=wet, idx=md.get("fesom14_idx"),
                    k_kept=k_kept, n_nodes=int(wet.shape[1]))


def _last_wet(st: _Statics) -> np.ndarray:
    """The index of each column's last wet level, ``(N,)`` (the mask is a
    non-empty prefix of every column)."""
    return st.wet.sum(axis=0) - 1


def _instant(ds: xr.Dataset, timestep: int) -> str:
    """A record of ``ds``, named by its instant."""
    try:
        return str(ds[TIME_DIM].values[timestep])
    except (KeyError, IndexError):
        return f"time index {timestep}"


def _is_3d(mode_cfg: Optional["ModeConfig"]) -> bool:
    return mode_cfg is not None and bool(getattr(mode_cfg, "Fdrift3D", False))


def _uvw_names(mode_cfg: Optional["ModeConfig"]) -> Tuple[str, str, str]:
    """The names of u, v and w (``mode.var_u/var_v/var_w`` or the defaults,
    as :meth:`Fesom14Reader.open_dataset` takes them)."""
    def name(attr: str, default: str) -> str:
        value = getattr(mode_cfg, attr, None) if mode_cfg is not None else None
        return str(value or default)

    return (name("var_u", DEFAULT_VAR_U), name("var_v", DEFAULT_VAR_V),
            name("var_w", DEFAULT_VAR_W))


def _record_kn(ds: xr.Dataset, name: str, key: str, timestep: int,
               st: _Statics, record: str) -> np.ndarray:
    """One record of a 3-D variable as ``(K_full, N)`` levels x nodes, on
    every level, checked against the static wet mask and otherwise as read
    (NaN or fill values below the bed)."""
    if name not in ds.data_vars:
        raise ValueError(
            f"fesom14: the dataset has no variable {name!r} ({key}); open the "
            "data with the fesom14 reader's open_dataset and the run's mode, "
            f"or set {key} to the name the files use")
    da = ds[name]
    dims = tuple(da.dims)
    layout = _classify(dims, _is_levelwise(ds))
    if layout == "levelwise":
        arr = np.asarray(da.isel({TIME_DIM: timestep}).values)
    elif layout == "nodes3d":
        if st.idx is None:
            raise ValueError(
                f"fesom14: variable {name!r} is one vector over the 3-D nodes "
                "(dimensions (time, nodes_3d)), but the mesh was built from "
                "cell polygons, which give no 3-D node table to place its "
                "entries on the levels. Set mesh.meshpath to the FESOM 1.4 "
                f"ASCII mesh directory of the run (holding {', '.join(MESH_FILES)})")
        vec = np.asarray(da.isel({TIME_DIM: timestep}).values)
        n3d = int(np.count_nonzero(st.wet))
        if vec.shape != (n3d,):
            raise ValueError(
                f"fesom14: variable {name!r}, record {record}: a vector over the "
                f"3-D nodes with {vec.size} entries, but the ASCII mesh has "
                f"{n3d} 3-D nodes. {_SAME_RUN}")
        arr = gather_nodes3d(vec, st.idx)
    else:
        raise ValueError(
            f"fesom14: variable {name!r} ({key}) has dimensions {dims}, but a "
            f"3-D variable is level-wise ('{TIME_DIM}', <depth>, <horizontal>) "
            f"or one vector over the 3-D nodes ('{TIME_DIM}', 'nodes_3d'), with "
            f"<depth> one of {DEPTH_DIMS} and <horizontal> one of "
            f"{HORIZONTAL_DIMS}. Set {key} to a 3-D variable of the files")
    try:
        check_record(arr, st.wet, name=name, record=record)
    except ValueError as exc:
        raise ValueError(f"{exc} {_RECORD_FIX}") from exc
    return arr


def _record_2d(ds: xr.Dataset, name: str, key: str, timestep: int,
               st: _Statics, record: str) -> np.ndarray:
    """One record of a 2-D field, float64 ``(N,)`` (a new array), with a
    value at every node."""
    da = ds[name]
    if da.ndim != 2 or da.shape[1] != st.n_nodes:
        raise ValueError(
            f"fesom14: variable {name!r} ({key}) has the layout "
            f"{_describe_layout(da)}, but a 2-D field holds one value per 2-D "
            f"node: the mesh has {st.n_nodes}. {_SAME_RUN}")
    vals = np.array(da.isel({TIME_DIM: timestep}).values, dtype=np.float64)
    missing = ~is_data(vals)
    if missing.any():
        n = int(np.argmax(missing))
        raise ValueError(
            f"fesom14: variable {name!r}, record {record}: no value "
            f"({MISSING_TEXT}) at node {n} ({int(missing.sum())} node(s) "
            "without one); a 2-D field holds a value at every node of the "
            f"mesh, which is wet at the surface everywhere. {_RECORD_FIX}")
    return vals


def _kept(full: np.ndarray, st: _Statics, dtype) -> np.ndarray:
    """The kept levels of a checked record, 0 below the bed: a new
    C-contiguous ``(K, N)`` array of ``dtype``."""
    k = st.k_kept
    out = np.zeros((k, full.shape[1]), dtype=dtype)
    np.copyto(out, full[:k], where=st.wet[:k])
    return out


def _zeroed(full: np.ndarray, st: _Statics) -> np.ndarray:
    """A checked record on every level, float64, 0 below the bed."""
    return np.where(st.wet, np.asarray(full, dtype=np.float64), 0.0)


def _filled(full: np.ndarray, st: _Statics) -> np.ndarray:
    """A checked record on every level, float64, each slot below a column's
    bed holding the column's last value above it."""
    rows = np.minimum(np.arange(full.shape[0])[:, None], _last_wet(st)[None, :])
    return np.take_along_axis(np.asarray(full, dtype=np.float64), rows, axis=0)


def _layer_index(mode_cfg: Optional["ModeConfig"], st: _Statics) -> Optional[int]:
    """``mode.sigma_layer`` as the index of a kept level, or None."""
    layer = getattr(mode_cfg, "sigma_layer", None) if mode_cfg is not None else None
    if layer is None:
        return None
    k = st.k_kept
    if (isinstance(layer, (bool, np.bool_))
            or not isinstance(layer, (int, np.integer)) or not 0 <= layer < k):
        cut = (f"; mode.max_levels keeps {k} of the mesh's {st.levels.size} "
               "levels" if k < st.levels.size else "")
        raise ValueError(
            f"fesom14: mode.sigma_layer = {layer!r} is not a kept level: 2-D "
            "tracking on 3-D data reads the level of that 0-based index, one of "
            f"0 .. {k - 1} ({st.levels[0]:g} .. {st.levels[k - 1]:g} m{cut}). "
            "Set mode.sigma_layer to one of them, or mode.sigma_mean: true for "
            "the depth mean")
    return int(layer)


def _reduce_2d(full: np.ndarray, st: _Statics, mesh: Mesh,
               mode_cfg: Optional["ModeConfig"]) -> np.ndarray:
    """``(N,)`` float64 from a checked record on every level, zeroed or
    filled below the bed: the level ``mode.sigma_layer``, the depth mean
    (``mode.sigma_mean``) or level 0."""
    k = _layer_index(mode_cfg, st)
    if k is not None:
        return np.array(full[k], dtype=np.float64)
    if mode_cfg is not None and getattr(mode_cfg, "sigma_mean", False):
        return _depth_mean(full, st, mesh)
    return np.array(full[0], dtype=np.float64)


def _depth_mean(full: np.ndarray, st: _Statics, mesh: Mesh) -> np.ndarray:
    """The mean of each column from the surface to its bed ``mesh.depth``.

    Trapezoids between the kept levels down to ``L``, the column's last kept
    wet level, then the value at ``L`` held constant from that level to the
    bed (a column cut by ``max_levels`` above its bed), divided by the bed
    depth; a column wet at the surface only (bed at 0 m) gives its surface
    value.
    """
    k = st.k_kept
    p = st.levels[:k]
    last = np.minimum(_last_wet(st), k - 1)
    a = np.asarray(full, dtype=np.float64)
    acc = np.zeros(a.shape[1])
    for j in range(k - 1):
        seg = 0.5 * (a[j] + a[j + 1]) * (p[j + 1] - p[j])
        acc += np.where(last >= j + 1, seg, 0.0)
    depth = np.asarray(mesh.depth, dtype=np.float64)
    acc += a[last, np.arange(a.shape[1])] * (depth - p[last])
    out = a[0].copy()
    deep = depth > 0.0
    out[deep] = acc[deep] / depth[deep]
    return out


def _eta_layout(ds: xr.Dataset, mode_cfg: Optional["ModeConfig"],
                st: _Statics, levelwise: bool) -> Optional[str]:
    """The name of the sea-surface height when the dataset holds it, once it
    is known to be one value per 2-D node (from metadata only)."""
    name = str((getattr(mode_cfg, "var_eta", None) if mode_cfg is not None
                else None) or DEFAULT_VAR_ETA)
    if name not in ds.data_vars:
        return None
    da = ds[name]
    layout = _classify(tuple(da.dims), levelwise)
    if layout == "2d" and da.shape[1] == st.n_nodes:
        return name
    if layout == "nodes3d":
        what = (f"one vector over the 3-D nodes ({da.shape[1]} entries along "
                "'nodes_3d')")
    else:
        what = f"of the layout {_describe_layout(da)}"
    raise ValueError(
        f"fesom14: the sea-surface height {name!r} is {what}, but it holds one "
        f"value per 2-D node: the mesh has {st.n_nodes} 2-D nodes. Leave the "
        f"{name!r} files out of data.ncinput (the sea-surface height is then "
        "0), or give it as a 2-D field over the 2-D nodes")


def _read_eta(ds: xr.Dataset, timestep: int, mesh: Mesh,
              mode_cfg: Optional["ModeConfig"], st: _Statics,
              record: str) -> np.ndarray:
    """The sea-surface height of a record, float64 ``(N,)``; zeros without
    one."""
    name = _eta_layout(ds, mode_cfg, st, _is_levelwise(ds))
    if name is None:
        return np.zeros(st.n_nodes, dtype=np.float64)
    return _record_2d(ds, name, "mode.var_eta", timestep, st, record)
