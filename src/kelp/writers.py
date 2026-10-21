"""Trajectory output writers with streaming and restart support.

Supports two backends (auto-detected from file extension):

* ``.nc``   → NetCDF4  (single file, zlib compression)
* ``.zarr`` → Zarr v3  (directory store, chunked arrays)

Both backends write the same schema — trajectory output **format 2**
(release 1.0; described in ``docs/io_and_time.md`` §5) — through the same
public API, with identical attributes.

Format 2 schema::

    Global attributes: Conventions = "CF-1.8", featureType = "trajectory",
        kelp_output_format = 2 (int32), history, plus the run metadata.

    Dimensions:   trajectory (P, fixed), time (records; NetCDF unlimited,
                  Zarr resizable)

    Coordinate variables:
        trajectory(trajectory)   int32    particle ID, cf_role = "trajectory_id"
                                          (global IDs: particle_offset + i)
        time(time)               float64  time at the end of the recorded step,
                                          seconds; units "seconds since <epoch>"
                                          + calendar (standard_name/axis set)
                                          when the forcing has an epoch, else
                                          units "seconds" + time_origin

    Per record, dims (trajectory, time):
        lon     float64  degrees_east,  standard_name longitude, fill NaN
        lat     float64  degrees_north, standard_name latitude,  fill NaN
        z       float64  m, standard_name depth, positive down,   fill NaN
        status  int32    ParticleStatus flags, coordinates "lat lon z"
      lazily created on first use, then written at EVERY record (the fill
      when a record does not carry it):
        vel_u, vel_v  float64  m s-1, fill NaN (created when a record has u,
                               or up front by ``declare_velocity`` — every
                               run_single_realization output has them)
        vel_w         float64  m s-1, depth rate positive down, fill NaN
                               (created when a record has w, or by
                               ``declare_velocity(w=True)`` in a 3-D run:
                               no vel_w in 2-D)
        failure_reason int8    FailureReason flags, 0 (none) when absent,
                               coordinates "lat lon z"
        <sampled>     float64  fill NaN, coordinates "lat lon z", comment

    Per particle, dims (trajectory,), written by ``write_release``:
        release_lon, release_lat  float64  geographic release position
        release_z                 float64  release depth [m, positive down]
        release_time              float64  instant the particle starts moving
                                           (units/calendar of ``time``)

    Restart block (created by the FIRST checkpoint, overwritten by later
    ones; a file without checkpoints has no ``restart_*`` variables):
        restart_valid                int32 scalar (0 = none/interrupted,
                                     1 = valid; created and written 0 FIRST)
        restart_lon/lat/z(trajectory)          float64
        restart_status(trajectory)             int32
        restart_nearest_node(trajectory)       int32
        restart_step_i, restart_file_idx       int32 scalars
      optional, when the state carries them:
        restart_failure_reason, restart_release_step, restart_elem,
        restart_wet_streak, restart_release_lon/lat/z

Known CF deviation: CF wants an unlimited dimension outermost; ``time`` is
the inner dimension so each track is contiguous per particle.  Files without
``kelp_output_format`` are format 1 (record dimension ``obs``, no epoch);
:func:`load_trajectories` and :func:`load_restart` read both.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import os

import numpy as np

from .core.particles import ParticleState, ParticleStatus


#: Trajectory output format written by this module (global attribute
#: ``kelp_output_format``).  Files without the attribute are format 1.
OUTPUT_FORMAT = 2

# Variable names that the trajectory schema owns; sampled fields may not
# reuse them (nor the ``restart_`` checkpoint prefix).  Guarded in
# ``_ensure_sampled_vars`` so a colliding dataset variable fails loudly
# rather than silently clobbering schema data.
_RESERVED_VAR_NAMES = frozenset({
    "lon", "lat", "z", "time", "status", "trajectory", "obs",
    "_scalar",  # zarr restart-scalar dim name (defensive)
    "vel_u", "vel_v", "vel_w", "failure_reason",
    "release_lon", "release_lat", "release_z", "release_time",
})

#: per-particle release variables (format 2), dims ``(trajectory,)``
_RELEASE_NAMES = ("release_lon", "release_lat", "release_z", "release_time")

# ---------------------------------------------------------------------------
# Format-2 attribute tables (spec §4, §6, §8) — shared by both backends
# ---------------------------------------------------------------------------

_POS_ATTRS = {
    "lon": {"standard_name": "longitude", "long_name": "longitude",
            "units": "degrees_east"},
    "lat": {"standard_name": "latitude", "long_name": "latitude",
            "units": "degrees_north"},
    "z": {"standard_name": "depth", "long_name": "depth below the surface",
          "units": "m", "positive": "down"},
}
_VEL_COMMENT = ("flow velocity (the same sign in forward and backward runs) "
                "that the integrator evaluated for the step ending at this "
                "record, at the step's start position and time (rk45: at its "
                "end); NaN unless the particle was ALIVE through the step")
_VEL_ATTRS = {
    "vel_u": {"long_name": "eastward flow velocity at the particle "
                           "(geographic)",
              "units": "m s-1", "comment": _VEL_COMMENT},
    "vel_v": {"long_name": "northward flow velocity at the particle "
                           "(geographic)",
              "units": "m s-1", "comment": _VEL_COMMENT},
    "vel_w": {"long_name": "vertical flow velocity at the particle, positive "
                           "down (sigma meshes: the dia-sigma flux)",
              "units": "m s-1", "comment": _VEL_COMMENT},
}
_REL_ATTRS = {
    "release_lon": {"long_name": "release longitude", "units": "degrees_east"},
    "release_lat": {"long_name": "release latitude", "units": "degrees_north"},
    "release_z": {"long_name": "release depth, positive down", "units": "m"},
    "release_time": {"long_name": "instant the particle starts moving"},
}
_COORDS = "lat lon z"
_SAMPLED_COMMENT = ("forcing field at the particle position and record time "
                    "(discrete mode: the forcing record at the end of the "
                    "step; interpolated mode: blended to the record time; "
                    "at the end of the forcing, both modes: the "
                    "start-of-step field)")
_STATUS_ATTRS = {
    "long_name": "particle status",
    "flag_meanings": "alive beached outside not_released",
    "coordinates": _COORDS,
}
_STATUS_FLAGS = (int(ParticleStatus.ALIVE), int(ParticleStatus.BEACHED),
                 int(ParticleStatus.OUTSIDE), int(ParticleStatus.NOT_RELEASED))


def _failure_flag_values():
    from .core.particles import FailureReason
    return (int(FailureReason.NONE), int(FailureReason.INTERPOLATION_FAILED),
            int(FailureReason.OUT_OF_BOUNDS), int(FailureReason.SURFACE_HIT),
            int(FailureReason.BOTTOM_HIT))


_FAILURE_ATTRS = {
    "long_name": "particle failure reason",
    "flag_meanings": ("none interpolation_failed out_of_bounds "
                      "surface_hit bottom_hit"),
    "coordinates": _COORDS,
}


def _time_attrs(units, calendar=None, origin_note=None, cycle_period=None):
    """Attributes of ``time`` (spec §4); CF time attributes only with an epoch."""
    a = {"long_name": "time at the end of the recorded step", "units": units}
    if " since " in units:
        a.update(standard_name="time", axis="T")
        if calendar:
            a["calendar"] = calendar
    if origin_note:
        a["time_origin"] = origin_note
    if cycle_period is not None:
        a["forcing_cycle_period"] = float(cycle_period)
        a["comment"] = ("on_exhausted=cycle: time is the run clock; the "
                        "forcing repeats with forcing_cycle_period seconds")
    return a


def _release_time_attrs(units, calendar=None, origin_note=None):
    """Attributes of ``release_time``: the units, calendar and (no epoch)
    origin note of ``time``."""
    a = dict(_REL_ATTRS["release_time"])
    a["units"] = units
    if calendar and " since " in units:
        a["calendar"] = calendar
    if origin_note:
        a["time_origin"] = origin_note
    return a


def _sampled_attrs(user_attrs):
    """Attributes of a sampled-field variable: the caller's (``units``,
    ``long_name``, …; a caller's own ``comment`` wins) plus ``coordinates``."""
    a = {k: v for k, v in (user_attrs or {}).items() if v is not None}
    a.setdefault("comment", _SAMPLED_COMMENT)
    a["coordinates"] = _COORDS
    return a


def _check_length(name, arr, n):
    """``ValueError`` (message names ``length``) unless ``arr`` is ``(n,)``."""
    a = np.asarray(arr)
    if a.ndim != 1 or a.shape[0] != n:
        got = a.shape[0] if a.ndim == 1 else f"shape {a.shape}"
        raise ValueError(
            f"{name}: length {got} differs from the particle count {n} — "
            "slice a state with ParticleState.subset, never field by field")
    return a


def _check_record(state, sampled, n):
    """Every per-particle array of a record is ``(n,)`` — checked BEFORE any
    buffering or writing (zarr truncates a longer assignment silently; review
    2026-10-13)."""
    for name in ("lon", "lat", "z", "status"):
        _check_length(f"append: {name}", getattr(state, name), n)
    for name in ("u", "v", "w", "failure_reason"):
        a = getattr(state, name, None)
        if a is not None:
            _check_length(f"append: {name}", a, n)
    for name, vals in (sampled or {}).items():
        _check_length(f"append: sampled {name}", vals, n)


def _check_checkpoint(state, wet_streak, release_step, n):
    """Every per-particle array of a checkpoint is ``(n,)`` — checked BEFORE
    the validity flag is cleared, so a bad call leaves the previous
    checkpoint valid (review 2026-10-13)."""
    for name in ("lon", "lat", "z", "status", "nearest_node"):
        _check_length(f"write_checkpoint: {name}", getattr(state, name), n)
    _check_release_triplet(state, "write_checkpoint")
    for name in ("failure_reason", "elem", "release_lon", "release_lat",
                 "release_z"):
        a = getattr(state, name, None)
        if a is not None:
            _check_length(f"write_checkpoint: {name}", a, n)
    for name, a in (("release_step", release_step),
                    ("wet_streak", wet_streak)):
        if a is not None:
            _check_length(f"write_checkpoint: {name}", a, n)


def _check_release_triplet(state, where):
    """``release_lon/lat/z`` are set together or not at all — a checkpoint
    stores and lists them as one unit (round 2 review)."""
    have = [getattr(state, k, None) is not None
            for k in ("release_lon", "release_lat", "release_z")]
    if any(have) and not all(have):
        raise ValueError(
            f"{where}: release_lon, release_lat and release_z must be set "
            "together (or all be None)")


_INT32 = np.iinfo(np.int32)


def _check_trajectory_ids(ids):
    """Integral values inside int32 (the ``trajectory`` variable's type)."""
    a = np.asarray(ids)
    if a.size and not np.issubdtype(a.dtype, np.integer):
        if not np.all(np.isfinite(a)) or np.any(np.mod(a, 1) != 0):
            raise ValueError("trajectory IDs must be integers")
    if a.size and (a.min() < _INT32.min or a.max() > _INT32.max):
        raise ValueError(
            f"trajectory IDs {a.min()}..{a.max()} exceed int32 (the type of "
            "the trajectory variable): particle_offset + particle count must "
            f"stay <= {_INT32.max}")
    return a.astype(np.int32)


#: optional per-particle restart arrays; a checkpoint lists the ones it wrote
#: in the ``restart_optional`` attribute so a rewrite never leaves a stale one
_RESTART_OPTIONAL = ("restart_failure_reason", "restart_release_step",
                     "restart_elem", "restart_release_lon",
                     "restart_release_lat", "restart_release_z",
                     "restart_wet_streak")
_RESTART_ROTATION = ("restart_rotation_alpha", "restart_rotation_beta",
                     "restart_rotation_gamma")


def _restart_optional_present(state, release_step, wet_streak):
    """Names of the optional restart arrays this checkpoint writes."""
    names = []
    if state.failure_reason is not None:
        names.append("restart_failure_reason")
    if release_step is not None:
        names.append("restart_release_step")
    if state.elem is not None:
        names.append("restart_elem")
    if getattr(state, "release_lon", None) is not None:
        names += ["restart_release_lon", "restart_release_lat",
                  "restart_release_z"]
    if wet_streak is not None:
        names.append("restart_wet_streak")
    return names


# ---------------------------------------------------------------------------
# Backend: NetCDF4
# ---------------------------------------------------------------------------

class _NetCDF4Backend:
    """NetCDF4 streaming backend."""

    def __init__(self, path: str, n_particles: int, metadata: Optional[dict],
                 sample_var_attrs: Optional[dict] = None,
                 flush_every: int = 0):
        self._path = path
        self._np = n_particles
        self._meta = metadata
        self._ds = None
        self._is_open = False      # guards append/checkpoint against misuse
        self._obs_idx = 0
        self._has_uv = False   # vel_u/vel_v created?
        self._has_w = False    # vel_w created? (independent: none in 2-D)
        self._sample_attrs = dict(sample_var_attrs or {})
        self._sampled_names = []  # sampled-field variables created so far
        # (units, calendar, origin note) of `time`
        self._time_units = ("seconds", None, None)
        # Buffered/close-reopen mode (>0): accumulate this many record points
        # in RAM, then open the file in append mode, write the batch, and
        # close it again — so the on-disk file holds no HDF5 lock between
        # flushes and another process can read it mid-run.  0 = the original
        # behaviour: keep the file open and ``sync()`` after every append
        # (byte-identical output either way).
        self._flush_every = int(flush_every or 0)
        self._buf = []  # list of buffered record tuples (see append())
        self._opened_here = False  # did _begin_write open the ds itself?

    def open(self):
        import netCDF4 as nc4
        ds = nc4.Dataset(self._path, "w", format="NETCDF4")

        ds.createDimension("trajectory", self._np)
        ds.createDimension("time", None)

        traj_var = ds.createVariable("trajectory", "i4", ("trajectory",))
        traj_var.cf_role = "trajectory_id"
        traj_var.long_name = "particle ID"
        traj_var[:] = np.arange(self._np, dtype=np.int32)

        time_var = ds.createVariable("time", "f8", ("time",))
        for k, val in _time_attrs("seconds").items():
            time_var.setncattr(k, val)

        for name in ("lon", "lat", "z"):
            v = ds.createVariable(name, "f8", ("trajectory", "time"),
                                  zlib=True, complevel=4, fill_value=np.nan)
            for k, val in _POS_ATTRS[name].items():
                v.setncattr(k, val)

        sv = ds.createVariable("status", "i4", ("trajectory", "time"),
                               zlib=True, complevel=4)
        sv.long_name = _STATUS_ATTRS["long_name"]
        sv.flag_values = np.array(_STATUS_FLAGS, dtype=np.int32)
        sv.flag_meanings = _STATUS_ATTRS["flag_meanings"]
        sv.coordinates = _STATUS_ATTRS["coordinates"]

        ds.Conventions = "CF-1.8"
        ds.featureType = "trajectory"
        ds.setncattr("kelp_output_format", np.int32(OUTPUT_FORMAT))
        ds.history = "Created by kelp TrajectoryWriter"
        if self._meta:
            for k, v in self._meta.items():
                ds.setncattr(str(k), v)

        ds.sync()
        # Reset per-file mutable state so a writer reused across open()s starts
        # the new file cleanly (fresh lazy-var / buffer state).
        self._obs_idx = 0
        self._has_uv = False
        self._has_w = False
        self._sampled_names = []
        self._time_units = ("seconds", None, None)
        self._buf = []
        self._opened_here = False
        self._is_open = True
        if self._flush_every > 0:
            # Buffered mode: the schema now exists on disk; release the file
            # so it is readable until the first batch flush.
            ds.close()
            self._ds = None
        else:
            self._ds = ds

    def _create_vel_var(self, name):
        """One lazy velocity variable, NaN-filled: records written before it
        existed read NaN (HDF5 returns the fill for unwritten columns)."""
        v = self._ds.createVariable(name, "f8", ("trajectory", "time"),
                                    zlib=True, complevel=4, fill_value=np.nan)
        for k, val in _VEL_ATTRS[name].items():
            v.setncattr(k, val)

    def _ensure_uv_vars(self):
        """Create ``vel_u``/``vel_v`` on first use."""
        if self._has_uv:
            return
        for name in ("vel_u", "vel_v"):
            if name not in self._ds.variables:
                self._create_vel_var(name)
        self._has_uv = True

    def _ensure_w_var(self):
        """Create ``vel_w`` on first use — only when a record carries ``w``
        (no ``vel_w`` in 2-D)."""
        if self._has_w:
            return
        if "vel_w" not in self._ds.variables:
            self._create_vel_var("vel_w")
        self._has_w = True

    def _ensure_failure_reason_var(self):
        """Create the optional ``failure_reason`` variable on first use.

        Lazy creation: only allocated if the user runs the integrator
        with ``state.failure_reason`` set.  TODO 2.6.  No ``_FillValue``
        attribute: a fill of 0 would mask every NONE code in xarray.
        """
        if "failure_reason" in self._ds.variables:
            return
        v = self._ds.createVariable(
            "failure_reason", "i1", ("trajectory", "time"),
            zlib=True, complevel=4,
        )
        v.long_name = _FAILURE_ATTRS["long_name"]
        v.flag_values = np.array(_failure_flag_values(), dtype=np.int8)
        v.flag_meanings = _FAILURE_ATTRS["flag_meanings"]
        v.coordinates = _FAILURE_ATTRS["coordinates"]
        # created after records exist (a direct writer user; tracking
        # allocates it at the start): the earlier records are NONE (0), not
        # the NetCDF byte fill (review 2026-10-13)
        zero = np.zeros(self._np, dtype=np.int8)
        for k in range(self._obs_idx):
            v[:, k] = zero

    def _ensure_sampled_vars(self, names):
        """Create sampled-field variables on first use (lazy, after the
        fixed schema — keeps no-sampling files byte-identical)."""
        for name in names:
            if name in self._sampled_names:
                continue
            if name in _RESERVED_VAR_NAMES or name.startswith("restart_"):
                raise ValueError(
                    f"sampled variable name '{name}' is reserved in "
                    "trajectory output — rename the dataset variable")
            v = self._ds.createVariable(
                name, "f8", ("trajectory", "time"),
                zlib=True, complevel=4, fill_value=np.nan)
            for attr, val in _sampled_attrs(
                    self._sample_attrs.get(name)).items():
                v.setncattr(attr, val)
            self._sampled_names.append(name)

    def _write_obs(self, time_value, lon, lat, z, status,
                   u, v, w, failure_reason, sampled):
        """Write one record at ``self._obs_idx`` to the open ``self._ds`` and
        advance the record counter.  Shared by the streaming and buffered
        paths; assumes the file is open for writing (caller syncs).

        Every lazy variable that exists is written at EVERY record — the
        record's value, else its fill (NaN; ``failure_reason`` 0) — so no
        record is left to the HDF5 default fill."""
        i = self._obs_idx
        ds = self._ds
        ds.variables["time"][i] = time_value
        ds.variables["lon"][:, i] = lon
        ds.variables["lat"][:, i] = lat
        ds.variables["z"][:, i] = z
        ds.variables["status"][:, i] = status

        if u is not None or v is not None:
            self._ensure_uv_vars()
        if w is not None:
            self._ensure_w_var()
        if failure_reason is not None:
            self._ensure_failure_reason_var()
        if sampled:
            self._ensure_sampled_vars(sampled.keys())

        nan = None
        if self._has_uv or self._has_w or self._sampled_names:
            nan = np.full(self._np, np.nan)
        if self._has_uv:
            ds.variables["vel_u"][:, i] = nan if u is None else u
            ds.variables["vel_v"][:, i] = nan if v is None else v
        if self._has_w:
            ds.variables["vel_w"][:, i] = nan if w is None else w
        if "failure_reason" in ds.variables:
            ds.variables["failure_reason"][:, i] = (
                np.zeros(self._np, dtype=np.int8) if failure_reason is None
                else failure_reason)
        for name in self._sampled_names:
            vals = sampled.get(name) if sampled else None
            ds.variables[name][:, i] = nan if vals is None else vals

        self._obs_idx += 1

    def _begin_write(self):
        """Ensure the file is open for a write op.  In buffered mode it is
        normally closed, so open it in append mode and remember to close it
        again in :meth:`_end_write`; in streaming mode the ds is already open
        and left open."""
        if self._ds is None:
            import netCDF4 as nc4
            self._ds = nc4.Dataset(self._path, "a")
            self._opened_here = True
        else:
            self._opened_here = False

    def _end_write(self):
        """Flush, and close only if :meth:`_begin_write` opened the ds."""
        self._ds.sync()
        if self._opened_here:
            self._ds.close()
            self._ds = None
            self._opened_here = False

    def _flush(self):
        """Write all buffered records in one open/append/close cycle."""
        if not self._buf:
            return
        self._begin_write()
        try:
            for (tv, lon, lat, z, status, u, v, w, fr, samp) in self._buf:
                self._write_obs(tv, lon, lat, z, status, u, v, w, fr, samp)
        finally:
            # Always release the file (and reset _opened_here) even if a write
            # raises, so the writer can never get stuck holding the lock with a
            # stale "streaming" view of self._ds on the next _begin_write.
            self._end_write()
        self._buf.clear()

    def append(self, state: ParticleState, time_value: float,
               sampled: Optional[dict] = None):
        if not self._is_open:
            # In buffered mode append() no longer touches the file on each call,
            # so an append before open() / after close() would silently buffer
            # then be dropped — fail loudly instead.
            raise RuntimeError(
                "TrajectoryWriter is not open — call open() or use the "
                "context manager before append().")
        if self._flush_every > 0:
            # Buffer COPIES — the integrator reuses/rebinds state arrays each
            # step, so references would be unsafe across the flush window.
            self._buf.append((
                float(time_value),
                state.lon.copy(), state.lat.copy(), state.z.copy(),
                state.status.copy(),
                None if state.u is None else state.u.copy(),
                None if state.v is None else state.v.copy(),
                None if state.w is None else state.w.copy(),
                None if state.failure_reason is None
                else state.failure_reason.copy(),
                None if not sampled
                else {k: np.asarray(v).copy() for k, v in sampled.items()},
            ))
            if len(self._buf) >= self._flush_every:
                self._flush()
            return

        # Streaming mode (file held open, sync per append) — unchanged.
        self._write_obs(time_value, state.lon, state.lat, state.z,
                        state.status, state.u, state.v, state.w,
                        state.failure_reason, sampled)
        self._ds.sync()

    def write_checkpoint(self, state: ParticleState, step_i: int, file_idx: int,
                         wet_streak=None, rotation_angles=None, release_step=None):
        # Buffered mode: persist any buffered observations first so the restart
        # obs index is consistent with the trajectory data on disk, then open
        # the file for the checkpoint write.
        if self._flush_every > 0:
            self._flush()
        self._begin_write()
        try:
            self._write_checkpoint_payload(state, step_i, file_idx, wet_streak,
                                           rotation_angles, release_step)
        finally:
            self._end_write()

    _RESTART_F8 = ("restart_lon", "restart_lat", "restart_z")
    _RESTART_I4 = ("restart_status", "restart_nearest_node")

    def _ensure_restart_block(self):
        """Create the restart block at the first checkpoint: the validity
        flag FIRST, written 0 and synced, so an interrupted first
        checkpoint can never read as valid (spec §7)."""
        ds = self._ds
        if "restart_valid" in ds.variables:
            return
        ds.createVariable("restart_valid", "i4")
        ds.variables["restart_valid"][0] = 0
        ds.sync()
        for name in self._RESTART_F8:
            ds.createVariable(name, "f8", ("trajectory",))
        for name in self._RESTART_I4:
            ds.createVariable(name, "i4", ("trajectory",))
        ds.createVariable("restart_step_i", "i4")
        ds.createVariable("restart_file_idx", "i4")

    def _write_checkpoint_payload(self, state, step_i, file_idx, wet_streak,
                                  rotation_angles, release_step):
        self._ensure_restart_block()
        # Invalidate FIRST and validate LAST: an interrupted rewrite (an
        # exception, a killed process) must leave restart_valid = 0, never
        # a mixture of two steps under a valid flag (review 2026-09-22).
        self._ds.variables["restart_valid"][0] = 0
        self._ds.sync()
        self._ds.variables["restart_lon"][:] = state.lon
        self._ds.variables["restart_lat"][:] = state.lat
        self._ds.variables["restart_z"][:] = state.z
        self._ds.variables["restart_status"][:] = state.status
        self._ds.variables["restart_nearest_node"][:] = state.nearest_node
        self._ds.variables["restart_step_i"][0] = step_i
        self._ds.variables["restart_file_idx"][0] = file_idx
        if state.failure_reason is not None:
            if "restart_failure_reason" not in self._ds.variables:
                self._ds.createVariable(
                    "restart_failure_reason", "i1", ("trajectory",),
                )
            self._ds.variables["restart_failure_reason"][:] = state.failure_reason
        if release_step is not None:
            if "restart_release_step" not in self._ds.variables:
                self._ds.createVariable(
                    "restart_release_step", "i4", ("trajectory",),
                )
            self._ds.variables["restart_release_step"][:] = release_step
        if state.elem is not None:
            if "restart_elem" not in self._ds.variables:
                self._ds.createVariable(
                    "restart_elem", "i4", ("trajectory",),
                )
            self._ds.variables["restart_elem"][:] = state.elem
        if getattr(state, "release_lon", None) is not None:
            for name, arr in (("restart_release_lon", state.release_lon),
                              ("restart_release_lat", state.release_lat),
                              ("restart_release_z", state.release_z)):
                if name not in self._ds.variables:
                    self._ds.createVariable(name, "f8", ("trajectory",))
                self._ds.variables[name][:] = arr
        if wet_streak is not None:
            if "restart_wet_streak" not in self._ds.variables:
                self._ds.createVariable(
                    "restart_wet_streak", "i4", ("trajectory",),
                )
            self._ds.variables["restart_wet_streak"][:] = wet_streak
        # Rotated-frame: store the mesh angles the (rotated) state lives in
        # as scalar global attrs, mirroring how _meta is written at open();
        # a geographic checkpoint removes angles a previous one left.
        if rotation_angles is not None:
            a, b, g = (float(x) for x in rotation_angles)
            self._ds.setncattr("restart_rotation_alpha", a)
            self._ds.setncattr("restart_rotation_beta", b)
            self._ds.setncattr("restart_rotation_gamma", g)
        else:
            for k in _RESTART_ROTATION:
                if k in self._ds.ncattrs():
                    self._ds.delncattr(k)
        self._ds.setncattr("restart_optional", " ".join(
            _restart_optional_present(state, release_step, wet_streak)))
        self._ds.variables["restart_valid"][0] = 1
        self._ds.sync()

    def set_global_attr(self, name: str, value) -> None:
        """Write a single global attribute (mirrors the open()-time meta loop)."""
        # Buffered mode: persist any pending obs first so the on-disk file is
        # consistent with n_obs, matching write_checkpoint's flush-first
        # contract (a no-op when nothing is buffered / in streaming mode).
        if self._flush_every > 0:
            self._flush()
        self._begin_write()
        self._ds.setncattr(str(name), value)
        self._end_write()

    @staticmethod
    def _replace_attrs(var, attrs):
        """Replace every attribute of ``var`` (``_FillValue`` kept) by ``attrs``."""
        for k in list(var.ncattrs()):
            if k != "_FillValue":
                var.delncattr(k)
        for k, val in attrs.items():
            var.setncattr(k, val)

    def set_time_units(self, units, calendar=None, origin_note=None,
                       cycle_period=None):
        """(Re)write the attributes of ``time`` (and of ``release_time``
        when it exists, so the two always share units and calendar)."""
        if self._flush_every > 0:
            self._flush()
        self._begin_write()
        try:
            self._replace_attrs(self._ds.variables["time"],
                                _time_attrs(units, calendar, origin_note,
                                            cycle_period))
            if "release_time" in self._ds.variables:
                self._replace_attrs(
                    self._ds.variables["release_time"],
                    _release_time_attrs(units, calendar, origin_note))
        finally:
            self._end_write()
        self._time_units = (units, calendar, origin_note)

    def write_release(self, lon, lat, z, time):
        """Write the per-particle ``release_*`` variables (created once)."""
        if self._flush_every > 0:
            self._flush()
        self._begin_write()
        try:
            for name, arr in (("release_lon", lon), ("release_lat", lat),
                              ("release_z", z), ("release_time", time)):
                if name not in self._ds.variables:
                    var = self._ds.createVariable(name, "f8", ("trajectory",),
                                                  fill_value=np.nan)
                    for k, val in _REL_ATTRS[name].items():
                        var.setncattr(k, val)
                self._ds.variables[name][:] = np.asarray(arr, dtype=np.float64)
            self._replace_attrs(self._ds.variables["release_time"],
                                _release_time_attrs(*self._time_units))
        finally:
            self._end_write()

    def set_trajectory_ids(self, ids):
        """Overwrite the ``trajectory`` coordinate values."""
        if self._flush_every > 0:
            self._flush()
        self._begin_write()
        try:
            self._ds.variables["trajectory"][:] = np.asarray(ids, dtype=np.int32)
        finally:
            self._end_write()

    def declare_velocity(self, w):
        """Create ``vel_u``/``vel_v`` (and ``vel_w`` when ``w``) now."""
        if self._flush_every > 0:
            self._flush()
        self._begin_write()
        try:
            self._ensure_uv_vars()
            if w:
                self._ensure_w_var()
        finally:
            self._end_write()

    def close(self):
        self._is_open = False
        if self._flush_every > 0:
            # Flush the tail batch; _end_write closes the file.  Nothing stays
            # open between flushes in buffered mode.
            self._flush()
            return
        if self._ds is not None:
            self._ds.close()
            self._ds = None

    @property
    def n_obs(self) -> int:
        # Buffered records not yet flushed still count as written observations.
        return self._obs_idx + len(self._buf)


# ---------------------------------------------------------------------------
# Backend: Zarr v3
# ---------------------------------------------------------------------------

class _ZarrBackend:
    """Zarr v3 streaming backend.

    Writes are **block-buffered**: obs columns accumulate in RAM and are
    flushed one whole obs-chunk at a time (``arr[:, start:end] = block``), so
    each chunk is compressed and written exactly ONCE.  The previous
    per-step ``arr[:, i] = ...`` rewrote (read-modify-recompress) the entire
    ``(n_particles, OBS_CHUNK)`` chunk on every single timestep — with many
    particles that chunk is hundreds of MB, so a per-step run recompressed it
    OBS_CHUNK times per chunk and was several times slower than NetCDF (whose
    HDF5 chunk cache absorbs the repeated same-chunk writes).  The obs-chunk
    width is also chosen adaptively from the trajectory-BLOCK size (== the
    particle count while the axis is a single block; TODO 5.4 splits it
    into 500k blocks above ``TRAJ_CHUNK_MAX``) to keep each chunk near
    ``TARGET_CHUNK_BYTES`` instead of a fixed 256 (260 MB at 127k nodes).
    """

    OBS_CHUNK_MAX = 256          # upper bound on the obs-chunk width
    TARGET_CHUNK_BYTES = 4_000_000  # aim for ~4 MB per chunk
    #: TODO 5.4 — above this particle count the trajectory axis is split
    #: into blocks of this size.  Below it the axis stays a single block
    #: (the historical layout, byte-identical).  Rationale: at large N
    #: the obs width bottoms out at 1 and a chunk is the WHOLE particle
    #: column (~63 MB at 7.9M particles) — reading a particle subset
    #: then costs the same as reading everything; with blocks, a subset
    #: touches only its blocks and each chunk stays near the target.
    TRAJ_CHUNK_MAX = 500_000

    def __init__(self, path: str, n_particles: int, metadata: Optional[dict],
                 sample_var_attrs: Optional[dict] = None):
        self._path = path
        self._np = n_particles
        self._meta = metadata
        self._store = None
        self._is_open = False      # guards append/checkpoint against misuse
        self._obs_idx = 0          # total appended (flushed + buffered)
        self._has_uv = False       # vel_u/vel_v created?
        self._has_w = False        # vel_w created? (independent: none in 2-D)
        # (units, calendar, origin note) of `time`
        self._time_units = ("seconds", None, None)
        self._sample_attrs = dict(sample_var_attrs or {})
        self._sampled_names = []  # sampled-field arrays created so far
        # Trajectory-axis blocks (TODO 5.4): single block up to
        # TRAJ_CHUNK_MAX (historical layout), fixed-size blocks above.
        self._traj_chunk = (self._np if self._np <= self.TRAJ_CHUNK_MAX
                            else self.TRAJ_CHUNK_MAX)
        # Adaptive obs-chunk: keep each (traj_chunk x obs_chunk) f8 chunk
        # near TARGET_CHUNK_BYTES (4 MB), clamped to [1, OBS_CHUNK_MAX].
        # Derived from the BLOCK size, not the full particle count — for
        # an unsplit axis the two are the same (byte-identical layout).
        per_col = max(1, self._traj_chunk * 8)
        self._obs_chunk = int(max(1, min(self.OBS_CHUNK_MAX,
                                         self.TARGET_CHUNK_BYTES // per_col)))
        self._obs_buf = []   # buffered records (copies), flushed a chunk at a time
        self._flushed = 0    # obs already written to disk

    def open(self):
        import zarr

        store = zarr.open_group(self._path, mode="w")

        traj_arr = store.create_array("trajectory", shape=(self._np,), dtype="i4",
                                      dimension_names=("trajectory",))
        traj_arr.attrs.update({"cf_role": "trajectory_id",
                               "long_name": "particle ID"})
        traj_arr[:] = np.arange(self._np, dtype=np.int32)

        # Later resizes go through fresh ``store["time"]`` handles, which
        # re-read the metadata, so ``set_time_units`` attrs survive flushes.
        tarr = store.create_array("time", shape=(0,), chunks=(self._obs_chunk,),
                                  dtype="f8", dimension_names=("time",))
        tarr.attrs.update(_time_attrs("seconds"))

        for name in ("lon", "lat", "z"):
            arr = store.create_array(
                name, shape=(self._np, 0),
                chunks=(self._traj_chunk, self._obs_chunk), dtype="f8",
                dimension_names=("trajectory", "time"), fill_value=np.nan)
            arr.attrs.update(_POS_ATTRS[name])

        sv = store.create_array("status", shape=(self._np, 0),
                                chunks=(self._traj_chunk, self._obs_chunk),
                                dtype="i4", dimension_names=("trajectory", "time"))
        # the NetCDF backend's CF flags (set on the live handle)
        sv.attrs.update({"long_name": _STATUS_ATTRS["long_name"],
                         "flag_values": list(_STATUS_FLAGS),
                         "flag_meanings": _STATUS_ATTRS["flag_meanings"],
                         "coordinates": _STATUS_ATTRS["coordinates"]})

        # Metadata as attrs (the restart block is created by the first
        # checkpoint, see _ensure_restart_block)
        store.attrs["Conventions"] = "CF-1.8"
        store.attrs["featureType"] = "trajectory"
        store.attrs["kelp_output_format"] = int(OUTPUT_FORMAT)
        store.attrs["history"] = "Created by kelp TrajectoryWriter"
        if self._meta:
            for k, v in self._meta.items():
                store.attrs[str(k)] = v

        self._store = store
        # Reset ALL per-file mutable state so a writer reused across open()s
        # starts a fresh file cleanly (not at stale columns / with stale lazy
        # vars).
        self._obs_idx = 0
        self._flushed = 0
        self._obs_buf = []
        self._has_uv = False
        self._has_w = False
        self._time_units = ("seconds", None, None)
        self._sampled_names = []
        self._is_open = True

    def _create_rec_array(self, name, dtype, fill_value, attrs):
        """A lazily created ``(trajectory, time)`` array, attrs on the live
        handle; resized to the on-disk record count so the records written
        before it existed read its fill value."""
        kw = {} if fill_value is None else {"fill_value": fill_value}
        arr = self._store.create_array(
            name, shape=(self._np, 0),
            chunks=(self._traj_chunk, self._obs_chunk), dtype=dtype,
            dimension_names=("trajectory", "time"), **kw)
        arr.attrs.update(attrs)
        if self._flushed > 0:
            arr.resize((self._np, self._flushed))
        return arr

    def _ensure_uv_vars(self):
        """Create ``vel_u``/``vel_v`` on first use (NaN fill)."""
        if self._has_uv:
            return
        for name in ("vel_u", "vel_v"):
            if name not in self._store:
                self._create_rec_array(name, "f8", np.nan, _VEL_ATTRS[name])
        self._has_uv = True

    def _ensure_w_var(self):
        """Create ``vel_w`` on first use — only when a record carries ``w``
        (no ``vel_w`` in 2-D)."""
        if self._has_w:
            return
        if "vel_w" not in self._store:
            self._create_rec_array("vel_w", "f8", np.nan, _VEL_ATTRS["vel_w"])
        self._has_w = True

    def _ensure_failure_reason_var(self):
        """Lazily create ``failure_reason`` array (TODO 2.6 + review #2);
        the int8 default fill 0 = NONE."""
        if "failure_reason" in self._store:
            return
        attrs = dict(_FAILURE_ATTRS)
        attrs["flag_values"] = list(_failure_flag_values())
        self._create_rec_array("failure_reason", "i1", None, attrs)

    def _ensure_sampled_vars(self, names):
        """Create sampled-field arrays on first use (lazy, after the fixed
        schema).  A var created late (first sampled at ``obs_idx > 0``) is
        resized to the current obs length so the already-written columns
        back-fill with the array ``fill_value`` (NaN) — keeping it
        obs-aligned with ``lon``/``lat``/etc."""
        for name in names:
            if name in self._sampled_names:
                continue
            if name in _RESERVED_VAR_NAMES or name.startswith("restart_"):
                raise ValueError(
                    f"sampled variable name '{name}' is reserved in "
                    "trajectory output — rename the dataset variable")
            # Attrs must go on the live create_array handle: a later resize()
            # on that handle rewrites zarr.json from its metadata, dropping
            # attrs set via transient store[name] handles.  The resize to
            # _flushed back-fills the already-on-disk columns with NaN.
            self._create_rec_array(name, "f8", np.nan, _sampled_attrs(
                self._sample_attrs.get(name)))
            self._sampled_names.append(name)

    def append(self, state: ParticleState, time_value: float,
               sampled: Optional[dict] = None):
        if not self._is_open:
            # Buffered writes no longer touch the store on each append, so an
            # append before open() / after close() would silently buffer then
            # be dropped — fail loudly instead (mirrors the old per-column
            # writer, which raised on the closed store).
            raise RuntimeError(
                "TrajectoryWriter is not open — call open() or use the "
                "context manager before append().")
        # Buffer COPIES — the integrator reuses/rebinds state arrays each step.
        # Records flush a whole obs-chunk at a time (see _flush_block), so each
        # chunk is compressed exactly once.
        self._obs_buf.append((
            float(time_value),
            state.lon.copy(), state.lat.copy(), state.z.copy(),
            state.status.copy(),
            None if state.u is None else state.u.copy(),
            None if state.v is None else state.v.copy(),
            None if state.w is None else state.w.copy(),
            None if state.failure_reason is None else state.failure_reason.copy(),
            None if not sampled
            else {k: np.asarray(v).copy() for k, v in sampled.items()},
        ))
        self._obs_idx += 1
        if len(self._obs_buf) >= self._obs_chunk:
            self._flush_block()

    def _flush_block(self):
        """Write all buffered records as one contiguous obs-block per variable.

        Each variable is grown once and assigned ``[:, start:end]`` in a single
        call, so the underlying chunk(s) are compressed once instead of being
        re-read/recompressed per timestep.
        """
        buf = self._obs_buf
        if not buf:
            return
        n = len(buf)
        start = self._flushed
        end = start + n

        tarr = self._store["time"]
        tarr.resize((end,))
        tarr[start:end] = np.array([r[0] for r in buf], dtype="f8")

        for col, name in ((1, "lon"), (2, "lat"), (3, "z"), (4, "status")):
            arr = self._store[name]
            arr.resize((self._np, end))
            arr[:, start:end] = np.stack([r[col] for r in buf], axis=1)

        # Lazy variables: created when ANY record in the block carries them
        # (vel_u/vel_v with u, vel_w only with w — none in 2-D), then written
        # in EVERY block; records lacking a value contribute the fill (NaN;
        # failure_reason 0).
        if any(r[5] is not None or r[6] is not None for r in buf):
            self._ensure_uv_vars()
        if any(r[7] is not None for r in buf):
            self._ensure_w_var()
        if any(r[8] is not None for r in buf):
            self._ensure_failure_reason_var()
        keys = []
        for r in buf:
            if r[9]:
                keys.extend(k for k in r[9] if k not in keys)
        if keys:
            self._ensure_sampled_vars(keys)

        nanfill = np.full(self._np, np.nan)

        def _put(name, column):
            arr = self._store[name]
            arr.resize((self._np, end))
            arr[:, start:end] = np.stack(column, axis=1)

        vel_cols = []
        if self._has_uv:
            vel_cols += [(5, "vel_u"), (6, "vel_v")]
        if self._has_w:
            vel_cols.append((7, "vel_w"))
        for col, name in vel_cols:
            _put(name, [(r[col] if r[col] is not None else nanfill)
                        for r in buf])

        if "failure_reason" in self._store:
            z8 = np.zeros(self._np, dtype=np.int8)
            _put("failure_reason",
                 [(r[8] if r[8] is not None else z8) for r in buf])

        for k in self._sampled_names:
            _put(k, [(r[9][k] if (r[9] and k in r[9]) else nanfill)
                     for r in buf])

        self._flushed = end
        self._obs_buf = []

    def write_checkpoint(self, state: ParticleState, step_i: int, file_idx: int,
                         wet_streak=None, rotation_angles=None, release_step=None):
        # Persist buffered observations first so the restart obs index is
        # consistent with the trajectory data on disk.
        self._flush_block()
        self._ensure_restart_block()
        # Invalidate FIRST and validate LAST (see the NetCDF writer).
        self._store["restart_valid"][0] = 0
        self._store["restart_lon"][:] = state.lon
        self._store["restart_lat"][:] = state.lat
        self._store["restart_z"][:] = state.z
        self._store["restart_status"][:] = state.status
        self._store["restart_nearest_node"][:] = state.nearest_node
        self._store["restart_step_i"][0] = step_i
        self._store["restart_file_idx"][0] = file_idx
        if state.failure_reason is not None:
            if "restart_failure_reason" not in self._store:
                self._store.create_array(
                    "restart_failure_reason", shape=(self._np,),
                    dtype="i1", dimension_names=("trajectory",),
                )
            self._store["restart_failure_reason"][:] = state.failure_reason
        if release_step is not None:
            if "restart_release_step" not in self._store:
                self._store.create_array(
                    "restart_release_step", shape=(self._np,),
                    dtype="i4", dimension_names=("trajectory",),
                )
            self._store["restart_release_step"][:] = release_step
        if state.elem is not None:
            if "restart_elem" not in self._store:
                self._store.create_array(
                    "restart_elem", shape=(self._np,),
                    dtype="i4", dimension_names=("trajectory",),
                )
            self._store["restart_elem"][:] = state.elem
        if getattr(state, "release_lon", None) is not None:
            for name, arr in (("restart_release_lon", state.release_lon),
                              ("restart_release_lat", state.release_lat),
                              ("restart_release_z", state.release_z)):
                if name not in self._store:
                    self._store.create_array(
                        name, shape=(self._np,), dtype="f8",
                        dimension_names=("trajectory",))
                self._store[name][:] = arr
        if wet_streak is not None:
            if "restart_wet_streak" not in self._store:
                self._store.create_array(
                    "restart_wet_streak", shape=(self._np,),
                    dtype="i4", dimension_names=("trajectory",),
                )
            self._store["restart_wet_streak"][:] = wet_streak
        # Rotated-frame: store the mesh angles the (rotated) state lives in
        # as group attrs, mirroring how _meta is written at open(); a
        # geographic checkpoint removes angles a previous one left.
        if rotation_angles is not None:
            a, b, g = (float(x) for x in rotation_angles)
            self._store.attrs["restart_rotation_alpha"] = a
            self._store.attrs["restart_rotation_beta"] = b
            self._store.attrs["restart_rotation_gamma"] = g
        else:
            for k in _RESTART_ROTATION:
                if k in self._store.attrs:
                    del self._store.attrs[k]
        self._store.attrs["restart_optional"] = " ".join(
            _restart_optional_present(state, release_step, wet_streak))
        self._store["restart_valid"][0] = 1

    def set_global_attr(self, name: str, value) -> None:
        """Write a single group attribute (mirrors the open()-time meta loop)."""
        # Flush pending obs first so the on-disk file is consistent with n_obs
        # (mirrors write_checkpoint's flush-first contract).
        self._flush_block()
        self._store.attrs[str(name)] = value

    def _ensure_restart_block(self):
        """Create the restart block at the first checkpoint, the validity
        flag FIRST and written 0 (spec §7; zarr writes are immediate)."""
        s = self._store
        if "restart_valid" in s:
            return
        flag = s.create_array("restart_valid", shape=(1,), dtype="i4",
                              dimension_names=("_scalar",))
        flag[0] = 0
        for name in ("restart_lon", "restart_lat", "restart_z"):
            s.create_array(name, shape=(self._np,), dtype="f8",
                           dimension_names=("trajectory",))
        for name in ("restart_status", "restart_nearest_node"):
            s.create_array(name, shape=(self._np,), dtype="i4",
                           dimension_names=("trajectory",))
        for name in ("restart_step_i", "restart_file_idx"):
            s.create_array(name, shape=(1,), dtype="i4",
                           dimension_names=("_scalar",))

    def set_time_units(self, units, calendar=None, origin_note=None,
                       cycle_period=None):
        """(Re)write the attributes of ``time`` (and of ``release_time`` when
        it exists).  Fresh handles: every later ``_flush_block`` resize
        re-reads the metadata, so the attributes survive (pinned)."""
        self._flush_block()
        self._store["time"].attrs.put(
            _time_attrs(units, calendar, origin_note, cycle_period))
        if "release_time" in self._store:
            self._store["release_time"].attrs.put(
                _release_time_attrs(units, calendar, origin_note))
        self._time_units = (units, calendar, origin_note)

    def write_release(self, lon, lat, z, time):
        """Write the per-particle ``release_*`` arrays (created once)."""
        for name, vals in (("release_lon", lon), ("release_lat", lat),
                           ("release_z", z), ("release_time", time)):
            if name in self._store:
                arr = self._store[name]
            else:
                arr = self._store.create_array(
                    name, shape=(self._np,), dtype="f8", fill_value=np.nan,
                    dimension_names=("trajectory",))
                arr.attrs.update(_REL_ATTRS[name])
            arr[:] = np.asarray(vals, dtype=np.float64)
        self._store["release_time"].attrs.put(
            _release_time_attrs(*self._time_units))

    def set_trajectory_ids(self, ids):
        """Overwrite the ``trajectory`` coordinate values."""
        self._store["trajectory"][:] = np.asarray(ids, dtype=np.int32)

    def declare_velocity(self, w):
        """Create ``vel_u``/``vel_v`` (and ``vel_w`` when ``w``) now."""
        self._ensure_uv_vars()
        if w:
            self._ensure_w_var()

    def _pad_lazy_vars(self):
        """Pad any lazily-created (trajectory, time) array that is shorter
        than the record count, so every record variable shares one width.

        Zarr arrays have independent shapes (unlike NetCDF's shared unlimited
        ``time`` dim).  ``_flush_block`` writes every existing lazy variable
        in every block, so this is a safety net: the resize back-fills with
        the array's fill value (NaN for velocities and sampled fields, 0 for
        failure_reason)."""
        n = self._flushed
        names = []
        if self._has_uv:
            names += ["vel_u", "vel_v"]
        if self._has_w:
            names.append("vel_w")
        if "failure_reason" in self._store:
            names.append("failure_reason")
        names += list(self._sampled_names)
        for nm in names:
            arr = self._store[nm]
            if arr.shape[1] < n:
                arr.resize((self._np, n))

    def close(self):
        # Flush the final (possibly partial) obs-block, pad ragged lazy vars,
        # then release.
        if self._store is not None:
            self._flush_block()
            self._pad_lazy_vars()
        self._store = None
        self._is_open = False

    @property
    def n_obs(self) -> int:
        # _obs_idx counts appended records (flushed + still buffered).
        return self._obs_idx


# ---------------------------------------------------------------------------
# Public API: TrajectoryWriter
# ---------------------------------------------------------------------------

class TrajectoryWriter:
    """Streaming trajectory writer with restart checkpoints.

    Backend is auto-detected from the file extension:
    ``.nc`` → NetCDF4, ``.zarr`` → Zarr.

    Parameters
    ----------
    path : str
        Output file path (``.nc`` or ``.zarr``).
    n_particles : int
        Number of particles (fixed for the run).
    metadata : dict or None
        Optional key-value pairs stored as global attributes.
    sample_var_attrs : dict or None
        Optional ``{var_name: {attr: value, ...}}`` map of CF attributes
        (e.g. ``units``, ``long_name``) applied to lazily-created
        sampled-field variables when they first appear via
        ``append(..., sampled=...)``.
    flush_every : int
        NetCDF backend only.  ``0`` (default) keeps the file open for the
        whole run and ``sync()``\\s after every append (original behaviour).
        ``N > 0`` buffers ``N`` record points in memory, then opens the file
        in append mode, writes the batch, and closes it — so the file holds
        no HDF5 write lock between flushes and another process can open it
        read-only mid-run.  Output is byte-identical to streaming mode; only
        the on-disk flush cadence differs.  Ignored by the Zarr backend
        (already lock-free).

    Examples
    --------
    >>> with TrajectoryWriter("out.zarr", n_particles=100) as w:
    ...     for step in range(n_steps):
    ...         state = step_fn(state, vel, mesh, dt, time_dir)
    ...         w.append(state, time_value=step * dt)
    ...         if step % 50 == 0:
    ...             w.write_checkpoint(state, step, file_idx)
    """

    def __init__(self, path: str, n_particles: int,
                 metadata: Optional[Dict[str, object]] = None,
                 sample_var_attrs: Optional[Dict[str, dict]] = None,
                 flush_every: int = 0):
        self.path = path
        self.n_particles = n_particles
        self.metadata = metadata
        ext = Path(path).suffix.lower()
        if ext == ".zarr" or path.endswith(".zarr"):
            # Zarr is already lock-free and readable mid-run (chunks are
            # separate files), so ``flush_every`` is a NetCDF-only concern and
            # is intentionally ignored here.
            self._backend = _ZarrBackend(path, n_particles, metadata,
                                         sample_var_attrs)
        else:
            self._backend = _NetCDF4Backend(path, n_particles, metadata,
                                            sample_var_attrs,
                                            flush_every=flush_every)

    def open(self) -> TrajectoryWriter:
        """Create the file and define arrays/variables.

        Returns self for chaining.
        """
        self._backend.open()
        return self

    def append(self, state: ParticleState, time_value: float,
               sampled: Optional[Dict[str, np.ndarray]] = None) -> None:
        """Write one observation (snapshot) for all particles.

        Parameters
        ----------
        state : ParticleState
            Current particle state.  ``u``/``v`` and ``w`` create the
            ``vel_u/vel_v`` and ``vel_w`` variables independently on first
            use (no ``vel_w`` without ``w``); a record without them writes
            NaN once they exist.
        time_value : float
            Seconds, in the units set by :meth:`set_time_units` (default
            ``"seconds"`` without an epoch).
        sampled : dict or None
            Optional ``{var_name: (P,) array}`` of sampled scalar values.
            Variables are created lazily on first use (after the fixed
            schema), so a run without sampling produces a byte-identical
            file; a record that lacks an existing one writes NaN.  Reserved
            names (schema vars, ``release_*``, ``restart_`` prefix) raise
            ``ValueError``.  Every per-particle array must be
            ``(n_particles,)`` (``ValueError`` before anything is written).
        """
        _check_record(state, sampled, self.n_particles)
        if not np.isfinite(float(time_value)):
            raise ValueError(
                f"append: time_value must be finite, got {time_value!r} — "
                "under CF time units a NaN decodes to the epoch")
        self._backend.append(state, time_value, sampled)

    def set_time_units(self, units: str, calendar: Optional[str] = None,
                       origin_note: Optional[str] = None,
                       cycle_period: Optional[float] = None) -> None:
        """Set the attributes of the ``time`` coordinate (spec §4).

        ``units`` ``"seconds since YYYY-MM-DD hh:mm:ss"`` (with ``calendar``)
        adds ``standard_name = "time"`` and ``axis = "T"``; bare
        ``"seconds"`` (a forcing without a calendar epoch) carries
        ``origin_note`` as ``time_origin`` instead.  ``cycle_period``
        (seconds, ``on_exhausted: cycle``) adds ``forcing_cycle_period`` and a
        ``comment``.  ``release_time``, when written, gets the same units
        and calendar.  Call before the first :meth:`append`; a writer used
        without it writes ``units = "seconds"``.
        """
        self._backend.set_time_units(str(units), calendar, origin_note,
                                     cycle_period)

    def write_release(self, lon, lat, z, time) -> None:
        """Write the per-particle release variables (spec §5).

        ``lon``/``lat`` geographic degrees, ``z`` metres positive down,
        ``time`` seconds in the units of ``time`` (the instant the particle
        starts moving); each a ``(n_particles,)`` array, else ``ValueError``.
        Rewriting overwrites the values.
        """
        arrs = [_check_length(f"write_release: {name}", a, self.n_particles)
                for name, a in zip(_RELEASE_NAMES, (lon, lat, z, time))]
        if not np.all(np.isfinite(np.asarray(arrs[3], dtype=np.float64))):
            raise ValueError(
                "write_release: release_time must be finite — under CF time "
                "units a NaN decodes to the epoch, not to a missing value")
        self._backend.write_release(*arrs)

    def declare_velocity(self, w: bool = False) -> None:
        """Create ``vel_u``/``vel_v`` (and ``vel_w`` when ``w``) now,
        NaN-filled, instead of at the first record that carries a velocity.

        ``run_single_realization`` calls it (``w`` = a 3-D run), so every
        file of a run — each chain segment, each shard — has the same
        variable set: a segment resumed when every particle is already dead
        records no velocity and, created lazily, would lack the variables.
        Records without a velocity read NaN either way.
        """
        self._backend.declare_velocity(bool(w))

    def set_trajectory_ids(self, ids) -> None:
        """Overwrite the ``trajectory`` coordinate (default ``0..P-1``), e.g.
        with the global IDs ``particle_offset + arange(P)`` of a shard.
        ``ids`` must be ``(n_particles,)``, else ``ValueError``."""
        ids = _check_length("set_trajectory_ids: ids", ids, self.n_particles)
        self._backend.set_trajectory_ids(_check_trajectory_ids(ids))

    def write_checkpoint(self, state: ParticleState, step_i: int,
                         file_idx: int, wet_streak=None,
                         rotation_angles=None, release_step=None) -> None:
        """Save restart checkpoint (overwrites previous).

        ``wet_streak`` (optional, int32 ``(P,)``) is the beaching
        consecutive-wet-step counter; persisted lazily as
        ``restart_wet_streak`` only when supplied (mirrors
        ``restart_failure_reason``).  ``None`` → no beaching array written.

        ``release_step`` (optional, int32 ``(P,)``) is the delayed-release
        per-particle activation step (TODO 2.5); persisted lazily as
        ``restart_release_step`` only when supplied, so resuming a staggered
        run restores the schedule (and the ``NOT_RELEASED`` statuses ride in
        ``restart_status``).  ``None`` → no array written.

        ``rotation_angles`` (optional, ``(alpha, beta, gamma)`` degrees) is
        the mesh coordinate frame the checkpointed (rotated) state lives
        in; stored as the scalar attributes ``restart_rotation_alpha/beta/
        gamma`` only when supplied.  ``None`` → no attribute written
        (geographic run; ``load_restart`` returns ``rotation_angles=None``).
        Resume validation: :func:`kelp.tracking._validate_restart_frame`.

        The restart block is created by the first call (the validity flag
        first, written 0); ``state.release_lon/lat/z`` (mesh frame), when
        set, are stored as ``restart_release_lon/lat/z`` like ``elem``.
        """
        # Auto-persist the delayed-release schedule from the state (mirrors how
        # failure_reason is read from the state in the backend): a checkpoint
        # must never reload NOT_RELEASED particles with release_step=None, or
        # they would never activate.  An explicit kwarg overrides.
        if release_step is None:
            release_step = state.release_step
        # lengths checked before the validity flag is cleared: a bad call
        # leaves the previous checkpoint valid
        _check_checkpoint(state, wet_streak, release_step, self.n_particles)
        self._backend.write_checkpoint(
            state, step_i, file_idx, wet_streak, rotation_angles, release_step)

    def set_global_attr(self, name: str, value) -> None:
        """Write a single global attribute on the open file.

        Mirrors how ``metadata`` is written at :meth:`open`; usable after
        ``open()`` for provenance a caller could not supply at construction
        (e.g. ``rotation_angles`` for a direct ``run_single_realization`` /
        ``track_particles`` caller).  Setting an existing key overwrites it.
        """
        self._backend.set_global_attr(name, value)

    def close(self) -> None:
        """Flush and close."""
        self._backend.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def n_obs(self) -> int:
        """Number of observations written so far."""
        return self._backend.n_obs


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def _detect_format(path: str) -> str:
    """Return 'zarr' or 'netcdf4' based on path."""
    if Path(path).suffix.lower() == ".zarr" or path.endswith(".zarr"):
        return "zarr"
    return "netcdf4"


def write_restart_file(path: str, state, step_i: int, file_idx: int, *,
                       wet_streak=None, rotation_angles=None,
                       release_step=None, particle_offset=None,
                       uv_vertical=None) -> None:
    """Write a STANDALONE restart checkpoint (NetCDF) — readable by
    :func:`load_restart`.

    Unlike ``TrajectoryWriter.write_checkpoint`` (which embeds the ``restart_*``
    variables inside a trajectory output and overwrites them each call), this
    writes a self-contained file holding the full state for a bit-identical
    resume.  Use it to drop one numbered restart file per checkpoint into a
    dedicated restart directory (kept, never overwritten).

    Parameters mirror ``write_checkpoint``: ``state`` (ParticleState), the
    global ``step_i`` and ``file_idx`` bookkeeping, plus the optional beaching
    ``wet_streak``, the rotated-frame ``rotation_angles`` (alpha, beta, gamma),
    and the delayed-release ``release_step`` (falls back to ``state.release_step``).
    ``particle_offset`` (a shard's first global particle index, the key of its
    random draws) is stored as the attribute ``restart_particle_offset``.
    ``uv_vertical`` (the run's ``mode.uv_vertical``) is stored as the
    attribute ``restart_uv_vertical`` when it is ``"linear"``; ``None`` and
    ``"layer"`` store nothing (a file without it means ``"layer"``).
    ``state.release_lon/lat/z``, when set, are stored as
    ``restart_release_lon/lat/z`` (read from the state like ``elem``).

    Raises
    ------
    ValueError
        When a per-particle array's length differs from ``len(state.lon)``
        (e.g. a state sliced field by field — use ``ParticleState.subset``);
        nothing is written and an existing ``path`` is kept.
    """
    import netCDF4
    P = int(np.asarray(state.lon).shape[0])
    _check_release_triplet(state, "write_restart_file")
    # Written to a temporary sibling and renamed into place only after a
    # clean close: an interrupted write never replaces (or destroys) the
    # previous restart at ``path`` (review 2026-09-22).
    tmp = f"{path}.tmp{os.getpid()}"
    ds = netCDF4.Dataset(tmp, "w")
    ok = False

    def _put(name, dtype, val):
        arr = _check_length(f"write_restart_file: {name}", val, P)
        ds.createVariable(name, dtype, ("trajectory",))[:] = arr

    try:
        ds.createDimension("trajectory", P)
        for name, dtype, val in (
            ("restart_lon", "f8", state.lon),
            ("restart_lat", "f8", state.lat),
            ("restart_z", "f8", state.z),
            ("restart_status", "i4", state.status),
            ("restart_nearest_node", "i4", state.nearest_node),
        ):
            _put(name, dtype, val)
        for name in ("restart_step_i", "restart_file_idx", "restart_valid"):
            ds.createVariable(name, "i4")
        ds.variables["restart_step_i"][0] = int(step_i)
        ds.variables["restart_file_idx"][0] = int(file_idx)
        ds.variables["restart_valid"][0] = 0
        if state.failure_reason is not None:
            _put("restart_failure_reason", "i1", state.failure_reason)
        rs = release_step if release_step is not None else getattr(
            state, "release_step", None)
        if rs is not None:
            _put("restart_release_step", "i4", rs)
        el = getattr(state, "elem", None)
        if el is not None:
            _put("restart_elem", "i4", el)
        for name, attr in (("restart_release_lon", "release_lon"),
                           ("restart_release_lat", "release_lat"),
                           ("restart_release_z", "release_z")):
            arr = getattr(state, attr, None)
            if arr is not None:
                _put(name, "f8", arr)
        if wet_streak is not None:
            _put("restart_wet_streak", "i4", wet_streak)
        if rotation_angles is not None:
            a, b, g = (float(x) for x in rotation_angles)
            ds.setncattr("restart_rotation_alpha", a)
            ds.setncattr("restart_rotation_beta", b)
            ds.setncattr("restart_rotation_gamma", g)
        if particle_offset is not None:
            ds.setncattr("restart_particle_offset", int(particle_offset))
        if uv_vertical is not None and str(uv_vertical) != "layer":
            ds.setncattr("restart_uv_vertical", str(uv_vertical))
        ds.setncattr("restart_optional", " ".join(
            _restart_optional_present(state, rs, wet_streak)))
        ds.variables["restart_valid"][0] = 1
        ok = True
    finally:
        try:
            ds.close()
        except Exception:
            ok = False
            raise
        finally:
            if ok:
                os.replace(tmp, path)
            else:
                try:
                    os.remove(tmp)
                except OSError:
                    pass


def load_restart(path: str) -> dict:
    """Load restart state from a trajectory file (NetCDF or Zarr).

    Parameters
    ----------
    path : str
        Path to the trajectory file.

    Returns
    -------
    dict
        Keys: ``state`` (ParticleState), ``step_i`` (int), ``file_idx`` (int),
        ``wet_streak`` (int32 ``(P,)`` or ``None`` when not stored — the
        beaching consecutive-wet-step counter), ``rotation_angles``
        (``(alpha, beta, gamma)`` degrees or ``None`` — the coordinate
        frame the checkpointed state lives in; validate on resume with
        :func:`kelp.tracking._validate_restart_frame`), ``release_step``
        (int32 ``(P,)`` or ``None`` — the delayed-release activation steps;
        also attached to ``state.release_step`` so a resumed run continues
        the staggered schedule), ``particle_offset`` (int or ``None`` — the
        shard offset the random draws were keyed with: the standalone file's
        ``restart_particle_offset`` or the trajectory file's
        ``particle_offset`` attribute; ``None`` when neither is stored; pass it
        on as ``particle_offset=``, where ``None`` means 0), ``uv_vertical``
        (str or ``None`` — the ``mode.uv_vertical`` the checkpoint was written
        with: the standalone file's ``restart_uv_vertical`` or the trajectory
        file's ``uv_vertical`` attribute; ``None`` when neither is stored,
        which means ``"layer"``; validate on resume with
        :func:`kelp.tracking._validate_restart_uv_vertical`).
        The carried containing element (``restart_elem``, edge-walk
        tracking) is attached to ``state.elem`` when stored, and the release
        positions (``restart_release_lon/lat/z``, mesh frame) to
        ``state.release_lon/lat/z`` (``None`` for a restart written before
        output format 2).

    Raises
    ------
    ValueError
        If no valid checkpoint exists in the file: no ``restart_valid``
        variable (a format-2 file without checkpoints), or a flag that is
        not 1 — 0 (an interrupted checkpoint) or never written (masked).
    """
    fmt = _detect_format(path)

    failure_reason = None
    wet_streak = None
    rotation_angles = None
    release_step = None
    elem = None
    particle_offset = None
    uv_vertical = None
    release = {}
    if fmt == "zarr":
        import zarr
        store = zarr.open_group(path, mode="r")
        # an unwritten zarr i4 reads its fill value 0 (= invalid)
        if "restart_valid" not in store or store["restart_valid"][0] != 1:
            raise ValueError(f"No valid restart checkpoint in {path}")
        listed = (set(str(store.attrs["restart_optional"]).split())
                  if "restart_optional" in store.attrs else None)

        def _has(name):
            return name in store and (listed is None or name in listed)
        lon = np.array(store["restart_lon"][:], dtype=np.float64)
        lat = np.array(store["restart_lat"][:], dtype=np.float64)
        z = np.array(store["restart_z"][:], dtype=np.float64)
        status = np.array(store["restart_status"][:], dtype=np.int32)
        nearest_node = np.array(store["restart_nearest_node"][:], dtype=np.int32)
        step_i = int(store["restart_step_i"][0])
        file_idx = int(store["restart_file_idx"][0])
        if _has("restart_failure_reason"):
            failure_reason = np.array(
                store["restart_failure_reason"][:], dtype=np.int8,
            )
        if _has("restart_wet_streak"):
            wet_streak = np.array(
                store["restart_wet_streak"][:], dtype=np.int32,
            )
        if _has("restart_release_step"):
            release_step = np.array(
                store["restart_release_step"][:], dtype=np.int32,
            )
        if _has("restart_elem"):
            elem = np.array(store["restart_elem"][:], dtype=np.int32)
        for k in ("lon", "lat", "z"):
            if _has(f"restart_release_{k}"):
                release[k] = np.array(store[f"restart_release_{k}"][:],
                                      dtype=np.float64)
        if "restart_rotation_alpha" in store.attrs:
            rotation_angles = (
                float(store.attrs["restart_rotation_alpha"]),
                float(store.attrs["restart_rotation_beta"]),
                float(store.attrs["restart_rotation_gamma"]),
            )
        for key in ("restart_particle_offset", "particle_offset"):
            if key in store.attrs:
                particle_offset = int(store.attrs[key])
                break
        for key in ("restart_uv_vertical", "uv_vertical"):
            if key in store.attrs:
                uv_vertical = str(store.attrs[key])
                break
    else:
        import netCDF4 as nc4
        with nc4.Dataset(path, "r") as ds:
            # A created-but-never-written flag reads MASKED, and
            # ``masked != 1`` is masked (falsy) — fill it with 0 first.
            flag = (np.ma.filled(ds.variables["restart_valid"][:], 0).ravel()
                    if "restart_valid" in ds.variables else np.zeros(1))
            if flag.size < 1 or int(flag[0]) != 1:
                raise ValueError(f"No valid restart checkpoint in {path}")
            listed = (set(str(ds.getncattr("restart_optional")).split())
                      if "restart_optional" in ds.ncattrs() else None)

            def _has(name):
                return (name in ds.variables
                        and (listed is None or name in listed))
            lon = np.array(ds.variables["restart_lon"][:], dtype=np.float64)
            lat = np.array(ds.variables["restart_lat"][:], dtype=np.float64)
            z = np.array(ds.variables["restart_z"][:], dtype=np.float64)
            status = np.array(ds.variables["restart_status"][:], dtype=np.int32)
            nearest_node = np.array(ds.variables["restart_nearest_node"][:], dtype=np.int32)
            step_i = int(ds.variables["restart_step_i"][0])
            file_idx = int(ds.variables["restart_file_idx"][0])
            if _has("restart_failure_reason"):
                failure_reason = np.array(
                    ds.variables["restart_failure_reason"][:], dtype=np.int8,
                )
            if _has("restart_wet_streak"):
                wet_streak = np.array(
                    ds.variables["restart_wet_streak"][:], dtype=np.int32,
                )
            if _has("restart_release_step"):
                release_step = np.array(
                    ds.variables["restart_release_step"][:], dtype=np.int32,
                )
            if _has("restart_elem"):
                elem = np.array(
                    ds.variables["restart_elem"][:], dtype=np.int32,
                )
            for k in ("lon", "lat", "z"):
                if _has(f"restart_release_{k}"):
                    release[k] = np.array(
                        ds.variables[f"restart_release_{k}"][:],
                        dtype=np.float64)
            if "restart_rotation_alpha" in ds.ncattrs():
                rotation_angles = (
                    float(ds.getncattr("restart_rotation_alpha")),
                    float(ds.getncattr("restart_rotation_beta")),
                    float(ds.getncattr("restart_rotation_gamma")),
                )
            for key in ("restart_particle_offset", "particle_offset"):
                if key in ds.ncattrs():
                    particle_offset = int(ds.getncattr(key))
                    break
            for key in ("restart_uv_vertical", "uv_vertical"):
                if key in ds.ncattrs():
                    uv_vertical = str(ds.getncattr(key))
                    break

    state = ParticleState(
        lon=lon, lat=lat, z=z,
        status=status, nearest_node=nearest_node,
        failure_reason=failure_reason,
        release_step=release_step,
        elem=elem,
        release_lon=release.get("lon"),
        release_lat=release.get("lat"),
        release_z=release.get("z"),
    )
    return {"state": state, "step_i": step_i, "file_idx": file_idx,
            "wet_streak": wet_streak, "rotation_angles": rotation_angles,
            "release_step": release_step, "particle_offset": particle_offset,
            "uv_vertical": uv_vertical}


def load_trajectories(path: str) -> dict:
    """Read a trajectory file (NetCDF or Zarr) into numpy arrays.

    Parameters
    ----------
    path : str
        Path to the trajectory file.

    Returns
    -------
    dict
        Keys: ``lon``, ``lat``, ``z``, ``status`` (shape ``(P, T)``),
        ``time`` (shape ``(T,)``, raw seconds — or, in a format-1 file,
        whatever that file stored), ``trajectory`` (shape ``(P,)``).
        Any extra ``vel_*`` / ``failure_reason`` diagnostics, any
        sampled-field record variables and the format-2 ``release_lon``,
        ``release_lat``, ``release_z``, ``release_time`` (shape ``(P,)``)
        present in the file are returned under their own names.  Reads both
        the format-2 layout ``(trajectory, time)`` and the format-1 layout
        ``(trajectory, obs)``; NetCDF fill values of float variables come
        back as NaN.
    """
    fmt = _detect_format(path)

    vel_keys = ("vel_u", "vel_v", "vel_w")

    if fmt == "zarr":
        import zarr
        store = zarr.open_group(path, mode="r")
        n_particles = store["lon"].shape[0]
        result = {
            "lon": np.array(store["lon"][:]),
            "lat": np.array(store["lat"][:]),
            "z": np.array(store["z"][:]),
            "status": np.array(store["status"][:]),
            "time": np.array(store["time"][:]),
            "trajectory": np.array(store["trajectory"][:]),
        }
        for k in vel_keys:
            if k in store:
                result[k] = np.array(store[k][:])
        if "failure_reason" in store:
            result["failure_reason"] = np.array(store["failure_reason"][:])
        for k in _RELEASE_NAMES:
            if k in store:
                result[k] = np.array(store[k][:], dtype=np.float64)
        # Generic discovery: any remaining (trajectory, record) array
        # (sampled scalars), in either layout: match on the per-particle
        # 2-D shape; schema/restart names are already in result or skipped
        # by prefix.
        for name in store.array_keys():
            if name in result or name.startswith("restart_"):
                continue
            a = store[name]
            if a.ndim == 2 and a.shape[0] == n_particles:
                result[name] = np.array(a[:])
        return result
    else:
        import netCDF4 as nc4

        def _float(var):
            # netCDF4 returns a masked array for fill values (NaN-filled
            # variables, or unwritten cells of a format-1 file); collapse the
            # mask to plain NaN so callers see a float array.
            return np.ma.filled(var[:], np.nan).astype(np.float64)

        rec_dims = {("trajectory", "obs"), ("trajectory", "time")}
        with nc4.Dataset(path, "r") as ds:
            v = ds.variables
            result = {
                "lon": _float(v["lon"]),
                "lat": _float(v["lat"]),
                "z": _float(v["z"]),
                "status": np.array(v["status"][:]),
                "time": np.array(v["time"][:]),
                "trajectory": np.array(v["trajectory"][:]),
            }
            for k in vel_keys:
                if k in v:
                    result[k] = _float(v[k])
            if "failure_reason" in v:
                result["failure_reason"] = np.array(v["failure_reason"][:])
            for k in _RELEASE_NAMES:
                if k in v:
                    result[k] = _float(v[k])
            # Generic discovery: any remaining record variable (sampled
            # scalars), either layout.
            for name, var in v.items():
                if name in result or name.startswith("restart_"):
                    continue
                if tuple(getattr(var, "dimensions", ())) in rec_dims:
                    result[name] = _float(var)
            return result


__all__ = [
    "OUTPUT_FORMAT",
    "TrajectoryWriter",
    "load_restart",
    "load_trajectories",
    "write_restart_file",
]
