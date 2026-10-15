"""Particle state container and status management.

Pure NumPy, no JAX dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Status codes
# ---------------------------------------------------------------------------

class ParticleStatus(IntEnum):
    """Coarse particle status codes.

    These four codes are the public API for particle health (BEACHED is
    the only reversible one; NOT_RELEASED flips to ALIVE at release).  When a particle dies, its ``status`` is
    set to ``OUTSIDE``; for finer-grained reasons (interpolation
    failure vs. domain exit vs. surface/bottom hit) see
    ``FailureReason`` and the optional ``ParticleState.failure_reason``
    array.
    """
    ALIVE = 1         # actively being tracked
    BEACHED = 2       # stuck on coast / shallow water
    OUTSIDE = 3       # left the domain
    NOT_RELEASED = 4  # delayed release: frozen until its release step (TODO 2.5)


class FailureReason(IntEnum):
    """Per-particle failure reason (TODO 2.6).

    Set by the integrator alongside ``ParticleStatus.OUTSIDE`` to
    explain *why* a particle died.  All values fit in ``int8``.

    Sticky semantics: once a particle's reason is set, the integrator
    leaves it alone on subsequent steps.  Reading
    ``state.failure_reason`` at the end of a run gives a per-particle
    cause-of-death.

    * ``NONE`` (0) — particle has not died, or the integrator path was
      not asked to track failure reasons.  Default for ALIVE particles.
    * ``INTERPOLATION_FAILED`` (1) — velocity LSQ stencil was singular
      or under-determined at the particle position.  Particle reverted
      to its pre-step location and marked OUTSIDE.
    * ``OUT_OF_BOUNDS`` (2) — post-displacement position failed the
      containing-element check.  Particle reverted and marked OUTSIDE.
    * ``SURFACE_HIT`` (3), ``BOTTOM_HIT`` (4) — *reserved* for a future
      diagnostic-events field.  Currently unused (3D vertical clamps
      do not kill particles, so they don't fit the death-reason model
      this enum currently tracks).

    Backward-compatibility: ``ParticleState.failure_reason`` is
    optional (``None`` = not tracked).  Existing code that ignores it
    keeps working unchanged.

    Coverage: ALL integrator paths populate the field since
    2026-08-16 — the unfused / Python-side paths via
    ``_update_failure_reason`` and every fused Numba kernel (2D
    Euler/RK2/RK4 incl. the tinterp and PCHIP twins, and 3D Euler)
    via a parallel int8 ``fr_out`` output merged sticky by
    ``_merge_fused_failure``.  Opt-in as always: ``None`` stays
    ``None`` on every path.
    """
    NONE = 0
    INTERPOLATION_FAILED = 1
    OUT_OF_BOUNDS = 2
    SURFACE_HIT = 3   # reserved
    BOTTOM_HIT = 4    # reserved


def status_to_string(code: int) -> str:
    """Convert a status code to a human-readable string."""
    _map = {
        ParticleStatus.ALIVE: "alive",
        ParticleStatus.BEACHED: "beached",
        ParticleStatus.OUTSIDE: "outside",
        ParticleStatus.NOT_RELEASED: "not_released",
    }
    return _map.get(code, f"unknown({code})")


def failure_reason_to_string(code: int) -> str:
    """Convert a ``FailureReason`` code to a human-readable string."""
    _map = {
        FailureReason.NONE: "none",
        FailureReason.INTERPOLATION_FAILED: "interpolation_failed",
        FailureReason.OUT_OF_BOUNDS: "out_of_bounds",
        FailureReason.SURFACE_HIT: "surface_hit",
        FailureReason.BOTTOM_HIT: "bottom_hit",
    }
    return _map.get(code, f"unknown({code})")


def count_by_status(status: np.ndarray) -> dict:
    """Count particles in each status category.

    Parameters
    ----------
    status : np.ndarray
        Array of ParticleStatus codes.

    Returns
    -------
    dict
        Counts keyed by status name.
    """
    counts = {
        "alive": int(np.sum(status == ParticleStatus.ALIVE)),
        "beached": int(np.sum(status == ParticleStatus.BEACHED)),
        "outside": int(np.sum(status == ParticleStatus.OUTSIDE)),
        "total": int(status.size),
    }
    # Only surface the delayed-release category when present, so the dict is
    # byte-identical for runs that never use it.
    n_dormant = int(np.sum(status == ParticleStatus.NOT_RELEASED))
    if n_dormant:
        counts["not_released"] = n_dormant
    return counts


def count_failure_reasons(failure_reason: Optional[np.ndarray]) -> dict:
    """Count particles by failure-reason category.

    Returns an empty dict when ``failure_reason`` is ``None`` (i.e. the
    integrator was not asked to track per-failure diagnostics).
    """
    if failure_reason is None:
        return {}
    return {
        "none": int(np.sum(failure_reason == FailureReason.NONE)),
        "interpolation_failed": int(
            np.sum(failure_reason == FailureReason.INTERPOLATION_FAILED)
        ),
        "out_of_bounds": int(
            np.sum(failure_reason == FailureReason.OUT_OF_BOUNDS)
        ),
        "surface_hit": int(np.sum(failure_reason == FailureReason.SURFACE_HIT)),
        "bottom_hit": int(np.sum(failure_reason == FailureReason.BOTTOM_HIT)),
        "total": int(failure_reason.size),
    }


# ---------------------------------------------------------------------------
# Particle state
# ---------------------------------------------------------------------------

@dataclass
class ParticleState:
    """State of all particles at a single point in time.

    Attributes
    ----------
    lon : np.ndarray, shape (P,)
        Longitude in degrees.
    lat : np.ndarray, shape (P,)
        Latitude in degrees.
    z : np.ndarray, shape (P,)
        Depth in meters (positive down).  Zero for 2D tracking.
    status : np.ndarray, shape (P,)
        ParticleStatus codes (int32).
    nearest_node : np.ndarray, shape (P,)
        Cached nearest mesh node index.  -1 if unknown.
    u : np.ndarray or None, shape (P,)
        Interpolated u-velocity (m/s) at particle position.  None if not tracked.
    v : np.ndarray or None, shape (P,)
        Interpolated v-velocity (m/s) at particle position.  None if not tracked.
    w : np.ndarray or None, shape (P,)
        Interpolated w-velocity (m/s) at particle position.  None if not tracked.
    failure_reason : np.ndarray or None, shape (P,)
        Per-particle ``FailureReason`` code (int8).  ``None`` (default)
        means the integrator is not asked to track failure reasons —
        existing behaviour.  When set to a zero-array on input, EVERY
        integrator path (unfused and fused) populates it with the
        specific reason for any ``status = OUTSIDE`` transition.
        ``SURFACE_HIT`` / ``BOTTOM_HIT`` remain RESERVED codes — 3-D
        clamps are transient and deliberately not recorded.  A beaching
        remap / resuspension clears the code (reversible, not a death).
    release_step : np.ndarray or None, shape (P,)
        Delayed release (TODO 2.5): the global step index at which each
        particle activates (``NOT_RELEASED → ALIVE``).  ``None`` (default)
        means immediate release for all — byte-identical to no feature.
        Frozen particles (``status == NOT_RELEASED``) are excluded from
        ``alive_mask`` (not advected) until ``global_step_i >= release_step``.
    elem : np.ndarray or None, shape (P,)
        Carried containing-element index (int32) for edge-walk element
        tracking (``numerics.elem_search: walk``): the element each
        particle was last validated inside, -1 if unknown/outside.
        ``None`` (default) = tracking disabled — byte-identical to no
        feature.  Rides through ``replace``/``snapshot`` and is
        checkpointed as ``restart_elem``.
    release_lon, release_lat, release_z : np.ndarray or None, shape (P,)
        Release position of each particle in the mesh frame (like ``lon``/
        ``lat``), set by ``run_single_realization`` when a run starts at
        global step 0.  Ride through ``replace``/``snapshot``/``subset``
        and are checkpointed as ``restart_release_*``, so every segment of a
        resumed chain writes the same ``release_*`` output variables
        (output format 2).  ``None`` = not known (a restart written before
        format 2).
    """
    lon: np.ndarray
    lat: np.ndarray
    z: np.ndarray
    status: np.ndarray
    nearest_node: np.ndarray
    u: Optional[np.ndarray] = None
    v: Optional[np.ndarray] = None
    w: Optional[np.ndarray] = None
    failure_reason: Optional[np.ndarray] = None
    release_step: Optional[np.ndarray] = None
    elem: Optional[np.ndarray] = None
    release_lon: Optional[np.ndarray] = None
    release_lat: Optional[np.ndarray] = None
    release_z: Optional[np.ndarray] = None

    #: every optional per-particle array (``replace``/``snapshot``/``subset``)
    _OPTIONAL_ARRAYS = ("u", "v", "w", "failure_reason", "release_step",
                        "elem", "release_lon", "release_lat", "release_z")

    @property
    def n_particles(self) -> int:
        return self.lon.shape[0]

    @property
    def alive_mask(self) -> np.ndarray:
        """Boolean mask: True for active particles."""
        return self.status == ParticleStatus.ALIVE

    def snapshot(self) -> dict:
        """Return a copy of the current state as a dictionary of arrays."""
        d = {
            "lon": self.lon.copy(),
            "lat": self.lat.copy(),
            "z": self.z.copy(),
            "status": self.status.copy(),
        }
        for name in self._OPTIONAL_ARRAYS:
            a = getattr(self, name)
            if a is not None:
                d[name] = a.copy()
        return d

    def subset(self, index) -> "ParticleState":
        """A new state holding only the particles ``index`` selects (an index
        array, a boolean mask or a slice), every per-particle array sliced
        the same way — required and optional, ``None`` staying ``None``.
        Use it instead of slicing fields one by one: a hand-written field
        list silently misses fields added later (the release arrays of
        output format 2, review 2026-10-13)."""
        kw = {name: np.asarray(getattr(self, name))[index].copy()
              for name in ("lon", "lat", "z", "status", "nearest_node")}
        for name in self._OPTIONAL_ARRAYS:
            a = getattr(self, name)
            kw[name] = None if a is None else np.asarray(a)[index].copy()
        return ParticleState(**kw)

    def save(self, path: str) -> None:
        """Save particle positions to a whitespace-separated file.

        Format matches ``load`` / ``load_particles``: ``lon lat [depth]``.
        Depth column is written only when any depth is non-zero.

        Parameters
        ----------
        path : str
            Output file path.
        """
        has_depth = np.any(self.z != 0.0)
        with open(path, "w") as f:
            f.write("# lon lat" + (" depth\n" if has_depth else "\n"))
            for i in range(self.n_particles):
                line = f"{self.lon[i]:.10g} {self.lat[i]:.10g}"
                if has_depth:
                    line += f" {self.z[i]:.10g}"
                f.write(line + "\n")

    @classmethod
    def load(cls, path: str) -> ParticleState:
        """Load particle initial positions from a whitespace-separated file.

        Expected format: each row is ``lon lat [depth]``.
        Lines starting with ``#`` are ignored.

        Parameters
        ----------
        path : str
            Path to the initial positions file.

        Returns
        -------
        ParticleState
        """
        data = np.loadtxt(path, comments="#", dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        lon = data[:, 0]
        lat = data[:, 1]
        z = data[:, 2] if data.shape[1] >= 3 else None
        return make_state(lon, lat, z)

    @classmethod
    def from_grid(cls, mesh, nx: int = 5, ny: int = 5,
                  margin: float = 0.0) -> ParticleState:
        """Generate particles on a regular lon/lat grid, keeping only in-mesh points.

        Parameters
        ----------
        mesh : Mesh
            The mesh to seed particles in.
        nx, ny : int
            Number of grid points in longitude and latitude directions.
        margin : float
            Fraction of the mesh extent to shrink inward (0.0 = full extent,
            0.05 = 5% margin on each side).

        Returns
        -------
        ParticleState
            Only particles that lie inside a mesh element.
        """
        from .particles_gen import generate_grid
        return generate_grid(mesh, nx=nx, ny=ny, margin=margin)

    @classmethod
    def from_around_point(cls, mesh, lon0: float, lat0: float,
                          n: int = 20, radius_km: float = 5.0,
                          seed: Optional[int] = None) -> ParticleState:
        """Generate random particles around a point, keeping only in-mesh points.

        Parameters
        ----------
        mesh : Mesh
            The mesh to seed particles in.
        lon0, lat0 : float
            Centre point in degrees.
        n : int
            Number of candidate particles to generate.
        radius_km : float
            Radius of the circle in kilometres.
        seed : int or None
            Random seed for reproducibility.

        Returns
        -------
        ParticleState
            Only particles that lie inside a mesh element.
        """
        from .particles_gen import generate_around_point
        return generate_around_point(mesh, lon0, lat0, n=n,
                                     radius_km=radius_km, seed=seed)

    def replace(self, **kwargs) -> ParticleState:
        """Return a new ParticleState with specified fields replaced.

        Parameters
        ----------
        **kwargs
            Fields to replace.  Any field not specified is copied from self.
        """
        fields = {
            "lon": self.lon,
            "lat": self.lat,
            "z": self.z,
            "status": self.status,
            "nearest_node": self.nearest_node,
            "u": self.u,
            "v": self.v,
            "w": self.w,
            "failure_reason": self.failure_reason,
            "release_step": self.release_step,
            "elem": self.elem,
            "release_lon": self.release_lon,
            "release_lat": self.release_lat,
            "release_z": self.release_z,
        }
        fields.update(kwargs)
        return ParticleState(**fields)


def make_state(lon: np.ndarray, lat: np.ndarray,
               z: Optional[np.ndarray] = None) -> ParticleState:
    """Create a ParticleState from coordinate arrays.

    Parameters
    ----------
    lon, lat : array_like
        Particle positions in degrees.
    z : array_like or None
        Depths in meters.  If None, set to zero (2D).

    Returns
    -------
    ParticleState
    """
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    P = lon.shape[0]
    if z is None:
        z = np.zeros(P, dtype=np.float64)
    else:
        z = np.asarray(z, dtype=np.float64)
    status = np.full(P, ParticleStatus.ALIVE, dtype=np.int32)
    nearest_node = np.full(P, -1, dtype=np.int32)
    return ParticleState(lon=lon, lat=lat, z=z,
                         status=status, nearest_node=nearest_node)


def load_particles(path: str) -> ParticleState:
    """Load particle initial positions from a whitespace-separated file.

    Thin wrapper around ``ParticleState.load`` kept for backward compatibility.

    Parameters
    ----------
    path : str
        Path to the initial positions file.

    Returns
    -------
    ParticleState
    """
    return ParticleState.load(path)


__all__ = [
    "ParticleStatus",
    "FailureReason",
    "status_to_string",
    "failure_reason_to_string",
    "count_by_status",
    "count_failure_reasons",
    "ParticleState",
    "make_state",
    "load_particles",
]
