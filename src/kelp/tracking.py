"""Main tracking simulation loop.

Wires together: reader → mesh → particles → integrator → output.
Mirrors the Fortran ``drift-fesom-3d.f90`` realization/time loop.

Supports two velocity modes:

* **Discrete** (default, ``temporal_interp=False``):
  one file time index per tracking step — original behaviour.
* **Temporal interpolation** (``temporal_interp=True``):
  velocity is linearly interpolated between two adjacent file
  snapshots.  ``dt`` can be smaller than the file time step.
"""
from __future__ import annotations

import time as _time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .core.config import (
    AppConfig, ModeConfig, NumericsConfig, DiffusionConfig, TimeConfig,
    resolve_time_params,
)
from .numerics.diffusion import DiffusionStepArgs
from .numerics.integrators import get_integrator
from .core.geometry import DEG2RAD
from .core.io import LazyLerpField, VelocityField
from .core.mesh import Mesh, find_containing_elements, find_nearest_nodes
from .core.particles import ParticleState, make_state, ParticleStatus, count_by_status
from .core.release import (activate_released, mask_dormant_for_record,
                           mask_inactive_velocity)
from .readers import (get_reader, extract_time_info, time_axis_epoch,
                      NO_EPOCH_NOTE)


# ---------------------------------------------------------------------------
# Trajectory storage (in-memory, kept for backward compat + small runs)
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryRecord:
    """Collected trajectory snapshots for one realization.

    Attributes
    ----------
    lons : list of np.ndarray
        Longitude at each output step, shape (P,) each.
    lats : list of np.ndarray
    zs : list of np.ndarray
    times : list of float
        Time of each output snapshot, in seconds on the forcing time axis
        (the end of the recorded step; output format 2 — both modes).
    us, vs, ws : list of np.ndarray
        Velocity diagnostics, one ``(P,)`` array per snapshot once any
        snapshot carried them (NaN where a snapshot did not); ``ws`` stays
        empty in 2-D.
    sampled : dict of str -> list of np.ndarray
        Per-step sampled scalar values, keyed by variable name; each
        snapshot is a ``(P,)`` array.  Empty unless ``append`` is called
        with ``sampled=``.
    release_lon, release_lat, release_z, release_time : np.ndarray or None
        Per-particle release position (geographic) and instant, set by
        :meth:`set_release`.

    Notes
    -----
    All record points of a run must pass the same sampled keys — the
    per-key lists are obs-aligned only then (the streaming writer path
    NaN-backfills late-appearing keys instead).
    """
    lons: List[np.ndarray] = field(default_factory=list)
    lats: List[np.ndarray] = field(default_factory=list)
    zs: List[np.ndarray] = field(default_factory=list)
    statuses: List[np.ndarray] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    us: List[np.ndarray] = field(default_factory=list)
    vs: List[np.ndarray] = field(default_factory=list)
    ws: List[np.ndarray] = field(default_factory=list)
    sampled: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    release_lon: Optional[np.ndarray] = None
    release_lat: Optional[np.ndarray] = None
    release_z: Optional[np.ndarray] = None
    release_time: Optional[np.ndarray] = None

    def append(self, state: ParticleState, t: float,
               sampled: Optional[Dict[str, np.ndarray]] = None) -> None:
        """Append one snapshot; ``sampled`` (optional) is an obs-aligned
        ``{var_name: (P,) array}`` dict of per-particle sampled scalars."""
        n_prev = len(self.times)
        nan = np.full(state.lon.shape[0], np.nan)
        self.lons.append(state.lon.copy())
        self.lats.append(state.lat.copy())
        self.zs.append(state.z.copy())
        self.statuses.append(state.status.copy())
        self.times.append(t)
        # Velocity lists stay aligned with the positions: the first snapshot
        # that carries a component back-fills NaN for the earlier ones, a
        # later snapshot without it appends NaN; no component is invented
        # (2-D has no w).
        if state.u is not None or self.us:
            if not self.us and n_prev:
                self.us.extend(nan.copy() for _ in range(n_prev))
                self.vs.extend(nan.copy() for _ in range(n_prev))
            self.us.append(nan.copy() if state.u is None else state.u.copy())
            self.vs.append(nan.copy() if state.v is None else state.v.copy())
        if state.w is not None or self.ws:
            if not self.ws and n_prev:
                self.ws.extend(nan.copy() for _ in range(n_prev))
            self.ws.append(nan.copy() if state.w is None else state.w.copy())
        if sampled:
            for k, v in sampled.items():
                self.sampled.setdefault(k, []).append(
                    np.asarray(v, dtype=np.float64).copy())

    def set_release(self, lon, lat, z, time) -> None:
        """Store the per-particle release position and instant (copies)."""
        self.release_lon = np.array(lon, dtype=np.float64, copy=True)
        self.release_lat = np.array(lat, dtype=np.float64, copy=True)
        self.release_z = np.array(z, dtype=np.float64, copy=True)
        self.release_time = np.array(time, dtype=np.float64, copy=True)

    @property
    def n_steps(self) -> int:
        return len(self.times)

    def to_arrays(self):
        """Stack snapshots into (T, P) arrays; ``release_*`` are ``(P,)``."""
        result = {
            "lon": np.array(self.lons),      # (T, P)
            "lat": np.array(self.lats),
            "z": np.array(self.zs),
            "status": np.array(self.statuses),  # (T, P)
            "time": np.array(self.times, dtype=np.float64),
        }
        if self.us:
            result["vel_u"] = np.array(self.us)
            result["vel_v"] = np.array(self.vs)
        if self.ws:
            result["vel_w"] = np.array(self.ws)
        for k, lst in self.sampled.items():
            result[k] = np.array(lst)      # (T, P)
        if self.release_lon is not None:
            result["release_lon"] = self.release_lon.copy()
            result["release_lat"] = self.release_lat.copy()
            result["release_z"] = self.release_z.copy()
            result["release_time"] = self.release_time.copy()
        return result


# ---------------------------------------------------------------------------
# Scalar sampling at record points
# ---------------------------------------------------------------------------

def _sample_record_fields(state, mesh, vel, sample_interp, Dmin=0.5):
    """Sample ``vel.scalars`` at post-step particle positions (record points).

    Returns an obs-aligned ``{var_name: (P,) array}`` dict, or ``None`` when
    the velocity field carries no scalars (dormant path).  Uses the fresh
    post-step ``state.nearest_node`` and post-step positions; dead particles
    yield ``NaN`` (the sampler's contract).  Sigma column geometry pulls eta
    from the *same* field the scalars came from (``vel.eta``).  Sampling is
    direction-free — no ``time_dir`` enters here.  ``Dmin`` is the run's
    minimum sigma-column thickness (``numerics.Dmin``), the same floor the
    step uses.
    """
    if not getattr(vel, "scalars", None):
        return None
    from .numerics.sampling import sample_field
    alive = state.status == ParticleStatus.ALIVE
    out = {}
    for name, arr in vel.scalars.items():
        out[name] = sample_field(
            arr, mesh, state.lon, state.lat,
            zs=state.z if arr.ndim == 2 else None,
            nearest=state.nearest_node, alive=alive,
            eta=vel.eta, method=sample_interp, Dmin=Dmin)
    return out


# ---------------------------------------------------------------------------
# Beaching post-step hook (TODO 2.2)
# ---------------------------------------------------------------------------

def _apply_beaching_hook(state, prev_status, mesh, vel, run_cfg, wet_streak,
                         real_id, step_i, dt):
    """Apply the beaching criteria + resuspension to a post-step state.

    Per-run constants ride ``run_cfg`` (a ``BeachRunConfig``, built once in
    ``run_single_realization``'s startup gate); only per-step variability
    (``state``, ``vel``, ``wet_streak``, ``real_id``, ``step_i``, ``dt``)
    is passed positionally — the ``DiffusionStepArgs`` house pattern.

    Order (spec Architecture §3):
      1. ``on_failure`` remap: a particle newly killed by the integrator
         this step (ALIVE→OUTSIDE, position already reverted = frozen anchor)
         becomes BEACHED when its anchor sits in coastal CONTEXT
         (``H_raw < zone_depth`` OR the anchor's containing element is
         coastal when topology is built).  Deep, non-coastal exits (open
         boundary) stay OUTSIDE.
      2. ``evaluate_beaching`` on the (possibly remapped) state: depth /
         coastal_element / stochastic criteria + resuspension; the kernel
         returns boolean decisions which are applied here.

    Same-step flicker (accepted v1 behaviour, spec §Interaction rules): a
    particle remapped OUTSIDE→BEACHED in step 1 enters step 2 as BEACHED and
    — with a wet anchor and ``min_wet_steps=1`` — may release immediately.

    Returns ``(state, wet_streak)`` (immutable update via ``.replace``).
    """
    from .numerics.beaching import (
        BeachingStepArgs, evaluate_beaching, _RESUSP_CODES,
    )

    beach_cfg = run_cfg.cfg
    crit = run_cfg.crit
    elem_coastal = run_cfg.elem_coastal
    need_topo = run_cfg.need_topo
    beach_seed = run_cfg.seed
    beach_zone = run_cfg.zone

    nn = state.nearest_node
    use_eta = beach_cfg.use_eta
    P = state.lon.shape[0]

    # Masks that drive both the containment subset and the two phases
    # below.  Computed up front so the containing-element lookup can run
    # on only the particles that consume it (perf gate — see below).
    alive_mask = state.status == ParticleStatus.ALIVE
    beached_mask = state.status == ParticleStatus.BEACHED
    # Newly killed by the integrator this step (ALIVE->OUTSIDE, position
    # already reverted to a frozen in-domain anchor) — the on_failure
    # remap consumes these even though they are OUTSIDE, not alive.
    newly_failed_mask = (state.status == ParticleStatus.OUTSIDE) & (
        prev_status == ParticleStatus.ALIVE)

    # Defensive anchor refresh: a particle that fails on its FIRST step (or
    # on the unfused 3D-RK / sigma and diffusion combined/split-kick failure
    # paths) can still carry the fresh-state nearest_node == -1 — then h_f
    # would be inf and the containment subset would skip it, so the context
    # gate could never remap it (silent on_failure degradation).  Positions
    # are already reverted by the integrator, so the refreshed nearest_node
    # IS the pre-step anchor.  Recompute it for those rows before both the
    # containment lookup and the H_raw read below see ``nn``.
    stale = newly_failed_mask & (nn < 0)
    if stale.any():
        from .core.mesh import find_nearest_nodes
        nn_fix = nn.copy()
        nn_fix[stale] = find_nearest_nodes(
            mesh, state.lon[stale], state.lat[stale])
        state = state.replace(nearest_node=nn_fix)
        nn = state.nearest_node

    # Containing element of each particle (anchor for dead ones — the
    # integrator keeps the pre-step nearest_node on failure, so this is
    # the in-domain anchor element).  Computed once when topology is built,
    # and ONLY for the particles that actually use it: ALIVE (criteria +
    # zone-OR), BEACHED (resuspension), and newly-failed (on_failure remap).
    # Skipping the dead-and-settled majority avoids the measured ~8.8x
    # containment waste at high mortality (reviewer perf gate).  Non-subset
    # rows stay -1 so the downstream ``inside`` guards treat them as
    # out-of-domain.
    containing = None
    if need_topo:
        containing = np.full(P, -1, dtype=np.int32)
        sub = alive_mask | beached_mask | newly_failed_mask
        if sub.any():
            idx = np.nonzero(sub)[0]
            ce, _ = find_containing_elements(
                mesh, state.lon[idx], state.lat[idx], nn[idx])
            containing[idx] = ce

    # --- 1. on_failure context-gated remap (OUTSIDE -> BEACHED) ---
    if "on_failure" in crit:
        newly = newly_failed_mask
        if newly.any():
            # kept in sync with beaching.evaluate_beaching's h_raw (raw,
            # unclamped): mesh.depth[nn] (+ vel.eta[nn] when use_eta).
            h_f = np.full(P, np.inf)
            valid = nn >= 0
            h_f[valid] = mesh.depth[nn[valid]]
            if use_eta:
                h_f[valid] = h_f[valid] + vel.eta[nn[valid]]
            context = h_f < beach_zone
            if containing is not None and elem_coastal is not None:
                inside = containing >= 0
                cc = np.zeros(P, dtype=np.bool_)
                cc[inside] = elem_coastal[containing[inside]]
                context = context | cc
            remap = newly & context
            if remap.any():
                new_status = state.status.copy()
                new_status[remap] = ParticleStatus.BEACHED
                # BEACHED via remap is reversible, not a death: clear the
                # freshly stamped code so failure_reason keeps its
                # cause-of-death semantics (a later REAL death re-stamps).
                if state.failure_reason is not None:
                    fr = state.failure_reason.copy()
                    fr[remap] = 0
                    state = state.replace(failure_reason=fr)
                state = state.replace(status=new_status)

    # --- 2. criteria + resuspension via the numerics kernel ---
    dry_mask = None
    if beach_cfg.dry_mask_var:
        scalars = getattr(vel, "scalars", None) or {}
        dm = scalars.get(beach_cfg.dry_mask_var)
        if dm is not None:
            dry_mask = dm[0] if dm.ndim > 1 else dm   # surface layer if 3D

    args = BeachingStepArgs(
        use_depth="depth" in crit,
        use_coastal="coastal_element" in crit,
        use_stochastic="stochastic" in crit,
        depth_min=beach_cfg.depth_min,
        use_eta=use_eta,
        zone_depth=beach_zone,
        t_beach=beach_cfg.t_beach or 0.0,
        resuspension=_RESUSP_CODES[beach_cfg.resuspension],
        rewet_hysteresis=beach_cfg.rewet_hysteresis,
        min_wet_steps=beach_cfg.min_wet_steps,
        t_resuspend=beach_cfg.t_resuspend or 0.0,
        seed=beach_seed,
        real_id=real_id,
        step_i=step_i,
        dt=dt,
        pid_offset=int(getattr(run_cfg, "pid_offset", 0)),
    )
    to_beach, to_release, wet_streak = evaluate_beaching(
        state, mesh, vel, args, dry_mask=dry_mask,
        elem_is_coastal=elem_coastal, containing_elems=containing,
        wet_streak=wet_streak)

    if to_beach.any() or to_release.any():
        new_status = state.status.copy()
        new_status[to_beach] = ParticleStatus.BEACHED
        new_status[to_release] = ParticleStatus.ALIVE
        if state.failure_reason is not None and to_release.any():
            # resuspended particles live again — drop any stale code
            fr = state.failure_reason.copy()
            fr[to_release] = 0
            state = state.replace(failure_reason=fr)
        state = state.replace(status=new_status)

    return state, wet_streak


# ---------------------------------------------------------------------------
# Velocity cache for temporal interpolation
# ---------------------------------------------------------------------------

class _VelCache:
    """Cache for adjacent velocity fields to avoid re-reading.

    With ``prefetch_size > 1`` (TODO 5.2) a cache miss bulk-loads a window of
    that many consecutive snapshots in the travel direction in ONE
    ``isel(time=slice).load()`` — fewer file seeks / dask computes on big meshes
    where I/O dominates.  ``prefetch_size == 1`` (default) reads one timestep at
    a time, byte-identical to the historical behaviour.
    """

    def __init__(self, velocity_float32: bool = False,
                 prefetch_size: int = 1, direction: int = 1,
                 n_time: Optional[int] = None, cycle: bool = False,
                 shared=None):
        self._cache = {}   # file_idx -> VelocityField
        # Node-shared snapshot cache (core.shared_fields.SharedFieldCache):
        # the first co-resident process reads+publishes each snapshot, the
        # rest attach the same bytes read-only. Fail-OPEN: any shared-layer
        # error downgrades to plain local reads with one warning.
        self._shared = shared
        if self._shared is not None and prefetch_size > 1:
            warnings.warn(
                "shared_fields: prefetch_size is forced to 1 — the shared "
                "writer already amortizes reads across processes.",
                RuntimeWarning)
            prefetch_size = 1
        # Keep u/v/w in float32 (native FESOM2 dtype) to halve snapshot memory
        # (TODO 5.3). Default False -> read upcasts to float64 (byte-identical).
        self._velocity_float32 = bool(velocity_float32)
        self._prefetch = max(1, int(prefetch_size))
        # Disable prefetch under cyclic forcing: the window is clamped at the
        # seam (no wrap), so ahead-snapshots get loaded then evicted before the
        # wrap reuses them -> a net I/O *regression* on short-cycle runs. Fall
        # back to per-timestep reads (correct + never worse).
        if cycle:
            self._prefetch = 1
        self._direction = 1 if direction >= 0 else -1
        self._n_time = n_time
        self._n_window_loads = 0   # diagnostics / test hook

    def _kw(self, diffusion, sample_vars):
        # Only forward optional kwargs when set: keeps the dormant path
        # byte-identical and compatible with readers predating a kwarg.
        kw = {"diffusion": diffusion}
        if sample_vars:
            kw["sample_vars"] = sample_vars
        if self._velocity_float32:
            kw["velocity_float32"] = True
        return kw

    def get(self, file_idx: int, reader, ds, mesh, mode,
            diffusion=None, sample_vars=None) -> VelocityField:
        if file_idx not in self._cache and self._shared is not None:
            kw = self._kw(diffusion, sample_vars)
            try:
                self._cache[file_idx] = self._shared_get(
                    file_idx, reader, ds, mesh, mode, kw)
            except Exception as exc:
                warnings.warn(
                    f"shared_fields disabled for this run: {exc!r} — "
                    "falling back to local reads.", RuntimeWarning)
                self._shared = None
        if file_idx not in self._cache:
            kw = self._kw(diffusion, sample_vars)
            if self._prefetch > 1:
                self._load_window(file_idx, reader, ds, mesh, mode, kw)
            else:
                self._cache[file_idx] = reader.read_velocities(
                    ds, file_idx, mesh, mode, **kw)
        return self._cache[file_idx]

    def _shared_get(self, file_idx, reader, ds, mesh, mode,
                    kw) -> VelocityField:
        """Attach the published snapshot, or win the election and build it."""
        key = f"t{file_idx:08d}"
        (arrays, meta), published = self._shared.get_or_publish(
            key, lambda: _flatten_vel(reader.read_velocities(
                ds, file_idx, mesh, mode, **kw)))
        if published:
            # only the writer collects — one gc per snapshot advance;
            # a straggler that still needs a collected key just rebuilds
            keep = {f"t{i:08d}"
                    for i in range(file_idx - _SHARED_GC_WINDOW,
                                   file_idx + _SHARED_GC_WINDOW + 1)
                    if i >= 0}
            self._shared.gc(keep=keep)
        return _assemble_vel(arrays, meta)

    def _load_window(self, start, reader, ds, mesh, mode, kw):
        """Bulk-load a window of prefetch_size consecutive snapshots (one read).

        The window runs in the travel direction from ``start``; indices outside
        ``[0, n_time)`` are dropped (no cycle-seam wrap — a wrapped index gets
        its own small window on its own miss).  Reading from the in-memory
        window gives values identical to reading the lazy ``ds`` at the global
        index, so trajectories are bit-identical to ``prefetch_size == 1``.
        """
        n_time = self._n_time if self._n_time is not None else ds.sizes["time"]
        idxs = [start + self._direction * k for k in range(self._prefetch)]
        idxs = [i for i in idxs if 0 <= i < n_time]
        if not idxs:
            # ``start`` itself out of range (caller bug — every production caller
            # passes a clamped index): read it directly so the error, if any,
            # comes from the reader, not a 0-length window slice.
            self._cache[start] = reader.read_velocities(
                ds, start, mesh, mode, **kw)
            return
        lo, hi = min(idxs), max(idxs) + 1
        window = ds.isel(time=slice(lo, hi)).load()   # one contiguous read
        self._n_window_loads += 1
        for i in idxs:
            if i not in self._cache:
                try:
                    self._cache[i] = reader.read_velocities(
                        window, i - lo, mesh, mode, **kw)
                except Exception:
                    # A look-ahead record the run may never use (e.g. past
                    # the end of a partly covering W file) must not fail the
                    # requested one: it is read, and refused, on demand.
                    if i == start:
                        raise

    def evict_except(self, keep: set):
        """Remove cached entries not in *keep*.

        With prefetch on, also retain the still-ahead window so the loop's
        per-step ``evict_except({file_idx, next_idx})`` does not drop the
        bulk-loaded block before it is consumed (memory stays ~prefetch_size).
        """
        keep = set(keep)
        if self._prefetch > 1 and keep:
            front = max(keep) if self._direction > 0 else min(keep)
            keep |= {front + self._direction * k
                     for k in range(self._prefetch)}
        for k in list(self._cache):
            if k not in keep:
                del self._cache[k]


# --- node-shared snapshot cache glue (core.shared_fields) ------------------
#: gc window: snapshots kept around the writer's current index (±);
#: a straggler process outside it simply rebuilds via get_or_publish
_SHARED_GC_WINDOW = 3

#: VelocityField array attributes that ride the shared store
_SHARED_VEL_FIELDS = ("u", "v", "eta", "w", "kh_zonal", "kh_meridional",
                      "az")


def _flatten_vel(vel: VelocityField):
    """VelocityField -> (arrays dict, JSON-able meta) for publishing."""
    arrays = {}
    for f in _SHARED_VEL_FIELDS:
        a = getattr(vel, f)
        if a is not None:
            arrays[f] = a
    meta = {"uv_layout": vel.uv_layout, "kh_layout": vel.kh_layout,
            "scalar_names": sorted(vel.scalars) if vel.scalars else []}
    for nm in meta["scalar_names"]:
        arrays["scalar__" + nm] = vel.scalars[nm]
    return arrays, meta


def _assemble_vel(arrays, meta) -> VelocityField:
    """Inverse of `_flatten_vel` over mmap-attached arrays."""
    scal = {nm: arrays["scalar__" + nm]
            for nm in meta.get("scalar_names", [])}
    return VelocityField(
        u=arrays["u"], v=arrays["v"], eta=arrays.get("eta"),
        w=arrays.get("w"), kh_zonal=arrays.get("kh_zonal"),
        kh_meridional=arrays.get("kh_meridional"),
        kh_layout=meta.get("kh_layout"),
        uv_layout=meta.get("uv_layout", "element"),
        scalars=scal or None, az=arrays.get("az"))


def _kelp_version() -> str:
    """Engine version for output provenance: the package version plus the
    commit when the source is a git checkout — read from ``.git`` directly,
    so it does not depend on a ``git`` binary on the compute node (shards
    of one run must stamp the same string)."""
    from pathlib import Path
    from . import __version__
    try:
        git = Path(__file__).resolve().parents[2] / ".git"
        if git.is_file():                      # worktree: "gitdir: <path>"
            txt = git.read_text().strip()
            if txt.startswith("gitdir:"):
                g = Path(txt.split(":", 1)[1].strip())
                git = g if g.is_absolute() else (git.parent / g)
        head = (git / "HEAD").read_text().strip()
        if head.startswith("ref:"):
            ref = head.split(":", 1)[1].strip()
            common = git
            if (git / "commondir").exists():
                common = (git / (git / "commondir").read_text().strip())
            sha = None
            for base in (git, common):
                if (base / ref).exists():
                    sha = (base / ref).read_text().strip()
                    break
            if sha is None and (common / "packed-refs").exists():
                for line in (common / "packed-refs").read_text().splitlines():
                    if line.endswith(" " + ref):
                        sha = line.split()[0]
                        break
        else:
            sha = head
        if sha:
            return f"{__version__}+g{sha[:9]}"
    except Exception:
        pass
    return str(__version__)


def _shared_fields_generation(numerics) -> Optional[str]:
    """Run generation of the shared snapshot cache, or None.

    The explicit ``numerics.shared_fields_generation``, else
    ``$KELP_SHARED_FIELDS_GENERATION``, else the SLURM array job id (array
    tasks = shards share it) or job id, plus the srun step id when set (the
    tasks of one ``srun`` share it; two sequential steps do not).  Two runs
    started one after the other in the same batch step without srun share
    the SLURM value — give each its own explicit generation.
    """
    import os as _os
    explicit = (getattr(numerics, "shared_fields_generation", None)
                or _os.environ.get("KELP_SHARED_FIELDS_GENERATION"))
    if explicit:
        return str(explicit)
    job = (_os.environ.get("SLURM_ARRAY_JOB_ID")
           or _os.environ.get("SLURM_JOB_ID"))
    if not job:
        return None
    step = _os.environ.get("SLURM_STEP_ID")
    return f"slurm:{job}" + (f".{step}" if step else "")


def _shared_fields_tag(config) -> str:
    """Shared-cache namespace tag for a run configuration.

    Particle shards of one run differ ONLY in seed file and output path
    — both excluded here — so they land in one namespace and share
    snapshots; anything that changes what ``read_velocities`` returns
    (data stores, variable names, mode, dtype, rotation, diffusion
    fields) changes the tag and gets its own namespace.
    """
    import hashlib
    import json
    from dataclasses import asdict
    ident = {
        "mesh": asdict(config.mesh),
        "mode": asdict(config.mode),
        "data": {k: getattr(config.data, k, None)
                 for k in ("ncinput", "ncinput_v", "ncinput_w",
                           "ncinput_eta")},
        "velocity_float32": getattr(config.numerics,
                                    "velocity_float32", False),
        "diffusion": None,
    }
    diff = getattr(config.numerics, "diffusion", None)
    if diff is not None and getattr(diff, "enabled", False):
        ident["diffusion"] = {
            k: getattr(diff, k, None)
            for k in ("scheme", "var_kh_zonal", "var_kh_meridional",
                      "var_az", "az_enabled")}
    # beaching's dry mask rides the scalar payload (run_tracking appends
    # it to the read set) — it must namespace the cache too (review
    # review 2026-08: a shared snapshot without it would silently
    # degrade beaching to dry_mask=None)
    beach = getattr(config.numerics, "beaching", None)
    if beach is not None and getattr(beach, "enabled", False):
        ident["beaching_dry_mask_var"] = getattr(beach,
                                                 "dry_mask_var", None)
    # Identity of what is READ, not just where: a store rewritten at the
    # same path, a changed file set in a data directory, or a new engine
    # must get a fresh namespace (review 2026-09-22, F08).  Stat-based —
    # size + mtime of files and of a directory's top-level entries — so it
    # is cheap on multi-TB archives; shards of one run see the same stats.
    # A rewrite that preserves every stat needs a new shared_fields_dir.
    ident["data_stat"] = {
        k: _path_fingerprint(v) for k, v in (
            ("mesh", getattr(config.mesh, "meshpath", None)),
            *((k, getattr(config.data, k, None))
              for k in ("ncinput", "ncinput_v", "ncinput_w", "ncinput_eta")))
        if v}
    ident["engine"] = _kelp_version()
    # The run GENERATION: stats cannot see an in-place rewrite of a nested
    # Zarr chunk (review 2026-09-23), so a namespace is also private to one
    # run.  Without one there is no namespace (the caller falls back to
    # local reads with a warning — review round 2: a local run had
    # generation None and shared stale snapshots with the next one).
    gen = _shared_fields_generation(config.numerics)
    if gen is None:
        raise ValueError(
            "the shared snapshot cache needs a run generation that the "
            "run's shards share and no other run uses: set "
            "numerics.shared_fields_generation (or the environment variable "
            "KELP_SHARED_FIELDS_GENERATION); under SLURM the job id and srun "
            "step are used automatically")
    ident["generation"] = gen
    payload = json.dumps(ident, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _path_fingerprint(path) -> list:
    """(name, size, mtime_ns) of a file, or of each top-level entry of a
    directory (a Zarr store, a multi-year directory); 'missing' otherwise."""
    import os
    try:
        st = os.stat(path)
    except OSError:
        return ["missing"]
    if not os.path.isdir(path):
        return [[os.path.basename(str(path)), st.st_size, st.st_mtime_ns]]
    out = []            # the directory's own mtime churns while it is written
    try:
        with os.scandir(path) as it:
            for e in sorted(it, key=lambda e: e.name):
                try:
                    es = e.stat()
                    out.append([e.name, es.st_size, es.st_mtime_ns])
                except OSError:
                    out.append([e.name, "unreadable"])
    except OSError:
        pass
    return out


# ---------------------------------------------------------------------------
# Time-averaging midpoint shift
# ---------------------------------------------------------------------------

# --- u/v interpolation x layout compatibility (spec 2026-06-06) ---
def _validate_uv_compat(uv_layout: str, uv_interp: str) -> None:
    """leastsq / edge_leastsq are element-centre math and cannot consume
    node-based velocities — fail with the actionable fix."""
    if uv_layout == "node" and uv_interp in ("leastsq", "edge_leastsq"):
        raise ValueError(
            "node-based velocities (uv_layout='node', e.g. FESOM2 "
            "unod/vnod) require numerics.uv_interp: 'barycentric' — "
            f"{uv_interp!r} interpolates element-centre values.")


# --- rotated-frame boundary (spec 2026-06-06: Seeds / Output / Checkpoints) ---
def _validate_periodic_windows(mesh, periodic_lon, periodic_lat,
                               warn=True):
    """Sanity-check periodic windows against the mesh (fail fast on the
    configs whose failure mode would otherwise be silent).

    Raises
    ------
    ValueError
        * window and mesh extent do not overlap (every fold would land
          off-mesh — silent mass death);
        * with ``periodic_lon``: the mesh straddles the ±180 branch cut
          (the integrators normalize positions to [-180, 180) after
          every displacement, so a position crossing the cut RELOCATES
          into the mesh's far side and the fold teleports it — round-2
          review measured a 340° jump; dateline channels are
          unsupported with periodic lon windows).

    Warns (``warn=True``)
    ---------------------
    RuntimeWarning per window side the mesh does not extend past —
    particles crossing that side die OUTSIDE before the fold can catch
    them (the ghost-band contract).

    Longitude comparisons are BRANCH-AWARE: mesh node lons are unwrapped
    around the window centre first, so a mesh stored in the 0..360
    branch (e.g. nodes 349..356 with window [-10, -5]) validates
    correctly (review round 2: raw min/max false-rejected it).
    """
    for _axis, _win, _vals in (
        ("lon", periodic_lon, mesh.nodes_lon),
        ("lat", periodic_lat, mesh.nodes_lat),
    ):
        if _win is None:
            continue
        lo, hi = float(_win[0]), float(_win[1])
        if _axis == "lon":
            # Window-branch view of the mesh (unwrap around the window
            # centre — 0..360-branch meshes validate correctly).
            c = 0.5 * (lo + hi)
            m = c + ((np.asarray(_vals) - c + 180.0) % 360.0) - 180.0
            vmin, vmax = float(np.min(m)), float(np.max(m))
            # SYMMETRIC dateline rule (round-3 review: the previous
            # wrap-span heuristic caught +180 but not -180): the whole
            # mesh — channel AND ghost band — must sit strictly inside
            # (-180, 180) in the window's branch.  The integrators
            # normalize positions to that branch after every
            # displacement, so a mesh reaching the cut either gets
            # crossings RELOCATED into its far side (teleported fold) or
            # has its ghost band made unreachable (silent mass death at
            # the seam).  Global meshes wrap natively at ±180 and need
            # no periodic window — also rejected here.
            if vmin <= -180.0 or vmax >= 180.0:
                raise ValueError(
                    "numerics.periodic_lon: the mesh (in the window's "
                    "branch) touches or crosses the ±180 cut — tracking "
                    "normalizes positions to [-180, 180) after every "
                    "displacement, which would teleport folds or make "
                    "the ghost band unreachable. The channel INCLUDING "
                    "its ghost band must lie strictly inside (-180, "
                    "180); global meshes wrap natively and need no "
                    "periodic window."
                )
        else:
            vmin = float(np.min(_vals))
            vmax = float(np.max(_vals))
        if vmax <= lo or vmin >= hi:
            raise ValueError(
                f"numerics.periodic_{_axis} window [{lo}, {hi}] does "
                f"not overlap the mesh extent [{vmin}, {vmax}] — "
                f"every fold would land off-mesh."
            )
        if not warn:
            continue
        if vmin >= lo:
            warnings.warn(
                f"mesh does not extend past the WEST/SOUTH side of "
                f"the periodic_{_axis} window [{lo}, {hi}] (mesh "
                f"{_axis} min = {vmin}): particles crossing that "
                f"side die OUTSIDE before the fold — add a ghost "
                f"band there if that side is reachable.",
                RuntimeWarning,
            )
        if vmax <= hi:
            warnings.warn(
                f"mesh does not extend past the EAST/NORTH side of "
                f"the periodic_{_axis} window [{lo}, {hi}] (mesh "
                f"{_axis} max = {vmax}): particles crossing that "
                f"side die OUTSIDE before the fold — add a ghost "
                f"band there if that side is reachable.",
                RuntimeWarning,
            )


def _apply_periodic_wrap(state, mesh, periodic_lon, periodic_lat,
                         include_dormant=False):
    """Fold ALIVE particles back into the periodic window(s).

    The ghost-band contract: the mesh extends beyond the window by at
    least one step's displacement, particles track INTO the ghost band,
    and this post-step fold maps them back — so containment/search
    machinery needs no seam awareness.  In-window particles are bitwise
    untouched (only out-of-window ones are rewritten); dead/beached/
    not-released particles keep their positions (death records stay
    where they died).  Wrapped particles get a FRESH nearest-node
    anchor at the folded position (the pre-fold anchor points at the
    ghost band, and beaching/sampling read it this same step) and a
    reset carried element under edge-walk tracking (the next walk
    re-seeds via its global fallback).  ``include_dormant=True`` (the
    INIT fold only) also folds NOT_RELEASED seeds, so a delayed-release
    particle placed in the ghost band activates at the folded position
    instead of dying where an immediate seed would survive.  Returns the
    SAME object when both windows are None or nothing crossed —
    byte-identity.
    """
    if periodic_lon is None and periodic_lat is None:
        return state
    lon = state.lon
    lat = state.lat
    alive = state.alive_mask
    if include_dormant:
        alive = alive | (state.status == ParticleStatus.NOT_RELEASED)
    moved = np.zeros(state.n_particles, dtype=bool)
    new_lon, new_lat = lon, lat
    if periodic_lon is not None:
        x0, x1 = float(periodic_lon[0]), float(periodic_lon[1])
        need = alive & ((lon < x0) | (lon >= x1))
        if need.any():
            folded = x0 + np.mod(lon - x0, x1 - x0)
            # float mod can round a tiny negative onto the full period —
            # keep the fold strictly inside [x0, x1)
            folded = np.where(folded >= x1, x0, folded)
            new_lon = np.where(need, folded, lon)
            moved |= need
    if periodic_lat is not None:
        y0, y1 = float(periodic_lat[0]), float(periodic_lat[1])
        need = alive & ((lat < y0) | (lat >= y1))
        if need.any():
            folded = y0 + np.mod(lat - y0, y1 - y0)
            folded = np.where(folded >= y1, y0, folded)
            new_lat = np.where(need, folded, lat)
            moved |= need
    if not moved.any():
        return state
    idx = np.where(moved)[0]
    nn = state.nearest_node.copy()
    nn[idx] = find_nearest_nodes(mesh, new_lon[idx], new_lat[idx])
    kw = dict(lon=new_lon, lat=new_lat, nearest_node=nn)
    if state.elem is not None:
        elem = state.elem.copy()
        elem[idx] = -1
        kw["elem"] = elem
    return state.replace(**kw)


def _maybe_rotate_seeds(state, mesh):
    """Rotate USER/geographic seed positions into the mesh frame.

    No-op (same object) when the mesh is geographic.  Applied ONLY at
    the two user entry points (run_tracking seed load, track_particles)
    — run_single_realization treats its init_state as already in-frame
    (restart states arrive rotated).
    """
    angles = getattr(mesh, "rotation_angles", None)
    if angles is None:
        return state
    from .core.rotation import euler_matrix, rotate_lonlat
    R = euler_matrix(*angles)
    lon_r, lat_r = rotate_lonlat(state.lon, state.lat, R)
    return state.replace(lon=lon_r, lat=lat_r)


def _geo_view_for_record(state, mesh, vel_base=None):
    """Geographic COPY of the state for output (positions + vel vectors).

    The internal state stays rotated; this view feeds record/writer only.
    Returns the SAME object when the mesh is geographic (byte-identity).
    ``vel_base`` = ``(lon, lat)`` in the MESH frame where ``u``/``v`` were
    evaluated (the step's start position); the geographic components of a
    tangent vector depend on its base point, so un-rotating at the record
    (end) position turned the vector by the frame's change of direction
    along the step (~0.4 % at 80 °N over a day, ~180° across the pole).
    ``None`` = the record position (rk45 evaluates at the end).
    """
    angles = getattr(mesh, "rotation_angles", None)
    if angles is None:
        return state
    from .core.rotation import (euler_matrix, unrotate_lonlat,
                                 unrotate_vectors)
    R = euler_matrix(*angles)
    lon_g, lat_g = unrotate_lonlat(state.lon, state.lat, R)
    kw = dict(lon=lon_g, lat=lat_g)
    if state.u is not None:
        if vel_base is None:
            blon, blat = lon_g, lat_g
        else:
            blon, blat = unrotate_lonlat(vel_base[0], vel_base[1], R)
        u_g, v_g = unrotate_vectors(state.u, state.v, blon, blat, R)
        kw.update(u=u_g, v=v_g)
    return state.replace(**kw)


def _record_clock_origin(*, use_tinterp, sim_time, start_time_idx, axis,
                         n_time_total, on_exhausted, dt, direction,
                         start_step_i):
    """``(T_g0, on_axis)`` for the record labels of output format 2.

    ``T_g0`` is the time of the START of global step 0, so the record written
    after global step ``g`` is labelled ``T_g0 + (g + 1)·dt·direction`` in
    both modes.  Interpolated mode: the run's first ``sim_time`` minus the
    steps already done.  Discrete mode: the time of the record where global
    step 0 started, ``start_time_idx − start_step_i·direction`` on ``axis``
    (the UNWRAPPED index a resume passes; a single record is that record) —
    anchoring on the resumed segment's own start record instead made the
    labels depend on the chain cuts wherever the record spacing differs from
    ``dt`` within ``_check_discrete_dt``'s tolerance (review 2026-10-13);
    ``sim_time_start`` does not decide the label (the reads follow the start
    index).  Without an axis, records are taken ``dt`` apart from record 0.
    ``on_axis`` says whether the clock is measured on the forcing axis —
    only then may the output carry the forcing's calendar epoch.
    """
    if use_tinterp:
        return float(sim_time) - start_step_i * dt * direction, True
    idx0 = int(start_time_idx) - int(start_step_i) * int(direction)
    if axis is not None and len(axis) >= n_time_total and n_time_total > 0:
        if n_time_total == 1:
            # one record repeated with period dt (the advertised
            # forcing_cycle_period): record idx0 of the unwrapped axis
            return float(axis[0]) + float(idx0) * float(dt), True
        return _index_to_sim_time(idx0, axis, n_time_total,
                                  on_exhausted), True
    return float(idx0) * float(dt), False


def _release_arrays(state, mesh, t_g0, dt, direction):
    """Geographic release positions and release instants of every particle.

    The instant is the start of the particle's release step ``r``
    (``release_step``; activation happens before the integrator of global
    step ``r``): ``T_g0 + max(r, 0)·dt·direction`` — always defined.  The
    positions come from ``state.release_*`` (mesh frame, un-rotated here);
    NaN when the state does not know them (a restart written before output
    format 2).
    """
    n = state.n_particles
    r = (np.zeros(n) if state.release_step is None
         else np.maximum(np.asarray(state.release_step, dtype=np.float64),
                         0.0))
    t = t_g0 + r * dt * direction
    if state.release_lon is None:
        nan = np.full(n, np.nan)
        return nan, nan.copy(), nan.copy(), t
    lon, lat = state.release_lon, state.release_lat
    angles = getattr(mesh, "rotation_angles", None)
    if angles is not None:
        from .core.rotation import euler_matrix, unrotate_lonlat
        lon, lat = unrotate_lonlat(lon, lat, euler_matrix(*angles))
    return (np.array(lon, dtype=np.float64, copy=True),
            np.array(lat, dtype=np.float64, copy=True),
            np.array(state.release_z, dtype=np.float64, copy=True), t)


def _validate_restart_frame(restart_angles, mesh):
    """Reject resuming a checkpoint in a different coordinate frame.

    Caller-driven resume recipe::

        r = load_restart(path)
        _validate_restart_frame(r["rotation_angles"], mesh)
        run_single_realization(..., init_state=r["state"],
                               start_step_i=r["step_i"] + 1, ...)
    """
    mesh_angles = getattr(mesh, "rotation_angles", None)
    ra = tuple(restart_angles) if restart_angles is not None else None
    ma = tuple(mesh_angles) if mesh_angles is not None else None
    if ra != ma:
        raise ValueError(
            f"checkpoint frame {ra} does not match mesh frame {ma}; "
            "resume with the same mesh.rotation as the original run")


def _resolve_composition(diff_cfg, integrator_name: str) -> str:
    """Resolve ``DiffusionConfig.composition`` against the integrator.

    Implements the composition rules of docs/diffusion.md §5:

    * uniform + any integrator → ``"split"``
    * em/m1   + Euler          → ``"combined"``
    * em/m1   + RK2 / RK4      → ``"split_rk"`` opt-in; any
      other composition value rejected.

    The user may override ``composition`` only when the override matches
    the auto-resolved value (uniform + any, em/m1 + Euler) or when it
    is the ``"split_rk"`` opt-in (em/m1 + RK).  Any other
    override is rejected with a clear NotImplementedError.

    Returns the resolved composition string.  Callers should record
    *this* value in writer metadata, not ``diff_cfg.composition``.
    """
    if not getattr(diff_cfg, "enabled", False):
        return diff_cfg.composition

    user_set = diff_cfg.composition != "split"

    if diff_cfg.scheme == "uniform":
        if user_set and diff_cfg.composition != "split":
            raise NotImplementedError(
                f"diffusion.composition={diff_cfg.composition!r} on "
                "scheme='uniform' is not supported; uniform always uses "
                "'split'."
            )
        return "split"

    # scheme is "em" or "m1"
    if integrator_name == "euler":
        if user_set and diff_cfg.composition != "combined":
            raise NotImplementedError(
                f"diffusion.composition={diff_cfg.composition!r} on "
                f"scheme={diff_cfg.scheme!r} + Euler is not "
                "supported; em/m1 + Euler must use 'combined'."
            )
        return "combined"

    # em/m1 + rk45: the adaptive path implements no variable-Kh kick at
    # all (v1) — reject before the split_rk opt-in can accept it.
    if integrator_name == "rk45":
        raise NotImplementedError(
            f"diffusion.scheme={diff_cfg.scheme!r} is not supported with "
            "integrator 'rk45' (v1); use scheme='uniform' or rk2/rk4 with "
            "composition='split_rk'."
        )

    # em/m1 + RK2/RK4: only "split_rk" is allowed, as an explicit opt-in.
    if diff_cfg.composition == "split_rk":
        return "split_rk"
    raise NotImplementedError(
        f"diffusion.scheme={diff_cfg.scheme!r} on integrator "
        f"{integrator_name!r} requires composition='split_rk' "
        "(an explicit opt-in).  Variable-Kh schemes compose with Euler via "
        "'combined' or with RK2/RK4 via 'split_rk'."
    )


def _compute_default_dres_m(mesh) -> float:
    """Mesh-derived default for ``DiffusionConfig.dres_m``.

    The central-difference gradient samples Kh at ``±dres_m`` around the
    particle.  A reasonable default is the median element edge length —
    short enough to resolve gradients on the local stencil, long enough
    to avoid land-mask noise — clipped to ``[100 m, 50 km]`` to handle
    both fine FESOM-C and coarse FESOM2 global meshes.

    The computation walks each element's edges in metres using the
    centre-element's latitude for the longitude metric — exact enough
    for a heuristic default; per-element local metrics would make the
    formula prettier but would not change the clipped result.
    """
    elem_nodes = mesh.elem_nodes
    nodes_lon = mesh.nodes_lon
    nodes_lat = mesh.nodes_lat
    if elem_nodes.shape[0] == 0:
        return 1000.0  # degenerate mesh fallback

    # Sample at most 5000 elements to keep the cost trivial on global meshes.
    n_e = elem_nodes.shape[0]
    if n_e > 5000:
        rng = np.random.default_rng(0)  # deterministic
        idx = rng.choice(n_e, 5000, replace=False)
    else:
        idx = np.arange(n_e)

    lengths = []
    SLM = 111120.0
    for e in idx:
        verts = elem_nodes[e]
        verts = verts[verts >= 0]
        if verts.size < 2:
            continue
        # Triangle: 3 edges; quad: 4 edges (with last-to-first wrap).
        for k in range(verts.size):
            a = int(verts[k])
            b = int(verts[(k + 1) % verts.size])
            if a == b:
                continue
            lat_mid = 0.5 * (nodes_lat[a] + nodes_lat[b])
            z0 = max(np.cos(lat_mid * DEG2RAD) * SLM, 1.0)
            # Dateline-safe lon delta — without this, an edge crossing
            # ±180° would measure as ~360° and dominate the median,
            # forcing the default toward the 50 km clip on global meshes.
            dlon_deg = ((nodes_lon[a] - nodes_lon[b]) + 180.0) % 360.0 - 180.0
            dx = dlon_deg * z0
            dy = (nodes_lat[a] - nodes_lat[b]) * SLM
            lengths.append(np.hypot(dx, dy))
    if not lengths:
        return 1000.0
    median = float(np.median(lengths))
    return float(np.clip(median, 100.0, 50000.0))


def _shift_to_midpoints(file_times: np.ndarray) -> np.ndarray:
    """Shift end-of-window timestamps to midpoints of averaging windows.

    For time-averaged output (e.g. monthly means), file timestamps mark
    the *end* of each averaging window.  The effective velocity time is
    the *midpoint* of the window ``[t_{i-1}, t_i]``.

    Parameters
    ----------
    file_times : np.ndarray
        Seconds-since-first-record, shape ``(N,)``.

    Returns
    -------
    np.ndarray
        Shifted times, same shape.  ``shifted[i] = (file_times[i-1] +
        file_times[i]) / 2`` for ``i > 0``; ``shifted[0]`` is
        extrapolated using the first interval.
    """
    shifted = np.empty_like(file_times)
    if len(file_times) < 2:
        return file_times.copy()
    # i > 0: midpoint of [t_{i-1}, t_i]
    shifted[1:] = 0.5 * (file_times[:-1] + file_times[1:])
    # i == 0: extrapolate back by half the first interval
    shifted[0] = file_times[0] - 0.5 * (file_times[1] - file_times[0])
    return shifted


# ---------------------------------------------------------------------------
# start_date resolution
# ---------------------------------------------------------------------------

def _iso_components(text: str):
    """``(Y, M, D, h, m, s, us)`` of an ISO date string, not validated
    against any calendar (the caller builds the date in the axis calendar)."""
    import re
    m = re.fullmatch(
        r"\s*(-?\d{1,6})-(\d{1,2})-(\d{1,2})"
        r"(?:[T ](\d{1,2})(?::(\d{1,2})(?::(\d{1,2})(?:\.(\d{1,6}))?)?)?)?\s*",
        str(text))
    if m is None:
        raise ValueError(
            f"start_date '{text}' is not an ISO date "
            "(YYYY-MM-DD[THH[:MM[:SS[.ffffff]]]]).")
    y, mo, d, hh, mi, ss, frac = m.groups()
    us = int((frac or "0").ljust(6, "0"))
    return (int(y), int(mo), int(d), int(hh or 0), int(mi or 0),
            int(ss or 0), us)


def _resolve_start_date(ds, start_date_str: str, file_times: np.ndarray,
                        temporal_interp: bool):
    """Resolve a start_date string to a file index and optional sim_time.

    Parameters
    ----------
    ds : xr.Dataset
        Open dataset with a ``time`` dimension.
    start_date_str : str
        ISO-format datetime, e.g. ``"2020-01-15T00:00"``.
    file_times : np.ndarray
        Seconds-since-first-timestep array from ``extract_time_info``.
    temporal_interp : bool
        Whether temporal interpolation is enabled.

    Returns
    -------
    (start_idx, sim_time_start)
        ``start_idx`` — file time index to start reading from.
        ``sim_time_start`` — float seconds offset into ``file_times`` to
        begin the simulation.  Equals ``file_times[start_idx]`` when the
        requested time matches a file snapshot exactly, or a between-snapshot
        value when temporal interpolation is used.  With temporal
        interpolation it is the requested instant itself (seconds since the
        first raw stamp), exact stamp or not; in discrete mode (exact stamps
        only) it is the record's time, which under ``time_averaging: mean``
        is its window centre.

    Raises
    ------
    ValueError
        If ``start_date`` does not match any file time and
        ``temporal_interp`` is False, or if the requested time is outside
        the file time range, or if the dataset has no usable time coordinate.
    """
    import pandas as pd

    if file_times is None or "time" not in ds.dims:
        raise ValueError(
            "start_date requires a dataset with an interpretable time "
            "coordinate (extract_time_info returned None)."
        )

    file_datetimes = ds["time"].values
    tattrs = ds["time"].attrs
    if (np.issubdtype(file_datetimes.dtype, np.number)
            and " since " in str(tattrs.get("units", "")).lower()):
        # an undecoded numeric CF axis: decode it with its own units and
        # calendar (review round 2)
        import xarray as _xr
        file_datetimes = np.asarray(_xr.coding.times.decode_cf_datetime(
            np.asarray(file_datetimes), str(tattrs["units"]),
            calendar=tattrs.get("calendar", "standard")))
    # Seconds of every RAW stamp and of the target, relative to the first raw
    # stamp, in the axis' own calendar (cftime noleap / 360_day / ...: the
    # datetime64 path compared across calendars and raised a TypeError —
    # review 2026-09-23).
    if np.issubdtype(file_datetimes.dtype, np.datetime64):
        target = np.datetime64(pd.Timestamp(start_date_str))
        raw_sec = (file_datetimes - file_datetimes[0]) / np.timedelta64(1, "s")
        target_sec = (target - file_datetimes[0]) / np.timedelta64(1, "s")
    elif file_datetimes.dtype == object:
        t0 = file_datetimes[0]
        cal = getattr(t0, "calendar", None)
        # parse the ISO components WITHOUT Gregorian validation: 2001-02-30
        # is a valid 360_day date (review round 2)
        comps = _iso_components(start_date_str)
        try:
            if cal is not None:
                import cftime
                target = cftime.datetime(*comps, calendar=cal)
            else:
                import datetime as _dt
                target = _dt.datetime(*comps)
        except ValueError as exc:
            raise ValueError(
                f"start_date '{start_date_str}' does not exist in the "
                f"dataset's '{cal}' calendar: {exc}") from exc
        raw_sec = np.array([(v - t0).total_seconds() for v in file_datetimes])
        target_sec = (target - t0).total_seconds()
    else:
        raise ValueError(
            "start_date needs a time axis of dates (datetime64, cftime or a "
            f"numeric CF axis '<unit> since <epoch>'); got dtype "
            f"{file_datetimes.dtype} with units "
            f"{tattrs.get('units', None)!r}.")

    # Check exact match.  Discrete mode reads whole records: an exact stamp
    # starts its record (at the record's time; under time_averaging: mean the
    # window centre).  With temporal interpolation the run starts at the
    # requested INSTANT, exact stamp or not (below): until 2026-10-04 an
    # exact stamp started half a window before the date and a date one second
    # later on time — a 12 h jump on daily means (audit F-038, review).
    exact = np.where(np.asarray(raw_sec, dtype=np.float64) == float(target_sec))[0]
    if len(exact) > 0 and not temporal_interp:
        idx = int(exact[0])
        return idx, file_times[idx]

    # No exact match — need temporal interpolation
    if not temporal_interp:
        raise ValueError(
            f"start_date '{start_date_str}' does not match any file timestep. "
            f"Either use an exact file time or set temporal_interp: true in "
            f"the config so the simulation can start between file snapshots."
        )

    # Offset in seconds from the first raw stamp: the physical instant (the
    # run clock's origin).  Under time_averaging: mean the records sit at
    # their window centres (file_times shifted) and the velocity at this
    # instant is interpolated between them.
    dt_offset = float(target_sec)
    if len(exact) > 0 and dt_offset > file_times[-1]:
        # The last stamp of time-averaged records lies past the last window
        # centre, where nothing is left to interpolate towards: keep the
        # record start it always had (review 2026-10-04, round 2).
        idx = int(exact[0])
        warnings.warn(
            f"start_date '{start_date_str}' is the last stamp of time-"
            "averaged records: the run starts at that record's window "
            "centre, half a record earlier (there is no later record to "
            "interpolate towards)", RuntimeWarning, stacklevel=2)
        return idx, file_times[idx]
    if dt_offset < 0 or dt_offset > file_times[-1]:
        raise ValueError(
            f"start_date '{start_date_str}' is outside the file time range "
            f"({file_datetimes[0]} .. {file_datetimes[-1]}); with "
            "temporal interpolation a start must lie between the first "
            "stamp and the last record's time (under time_averaging: mean, "
            "the last window centre) or be the last stamp itself."
        )

    # start_idx = index of the snapshot just before the target
    idx = int(np.searchsorted(file_times, dt_offset, side="right")) - 1
    idx = max(0, min(idx, len(file_times) - 1))

    return idx, float(dt_offset)


# ---------------------------------------------------------------------------
# Single-realization tracking
# ---------------------------------------------------------------------------

def _check_discrete_dt(dt, temporal_interp, file_dt, file_times, n_records=None):
    """Refuse a step unlike the record spacing in discrete mode.

    Discrete mode (``temporal_interp`` off) reads the next record every step
    while the particles move by ``dt``; a ``dt`` unlike the record spacing
    runs the forcing at another speed than the particles.  The spacing is
    that of ``file_times`` whenever two or more record times are known (a
    configured ``numerics.file_dt`` describes a file WITHOUT a usable axis
    and cannot override a decoded one — it let calendar months pass, audit
    2026-10-04), else ``file_dt``; ``max(0.1 s, 1e-3 spacing)`` of stamp
    jitter is allowed.  Irregular record times (calendar months, a gap) fit no
    single ``dt`` and are refused -- interpolated mode handles them.  Fewer
    than two records, or no time axis at all: nothing to compare.
    """
    if temporal_interp or dt is None:
        return
    n = len(file_times) if file_times is not None else n_records
    if n is not None and n < 2:
        return
    dt = float(dt)
    d = (np.abs(np.diff(np.asarray(file_times, dtype=np.float64)))
         if file_times is not None and len(file_times) >= 2 else None)
    if d is not None and d.max() > 0.0:
        # the axis decides unless it has no positive spacing at all (all
        # stamps equal: a repeated-snapshot file, which the docs send to
        # numerics.file_dt); one repeated stamp in a moving axis is
        # irregular and refused below
        if not np.allclose(d, d[0], rtol=1e-3, atol=0.0):
            raise ValueError(
                f"the record times are irregular ({d.min():g}..{d.max():g} s "
                "between records), which no single numerics.dt matches in "
                "discrete mode (temporal_interp: false), where every step "
                "reads the next record while the particles move by dt; set "
                "temporal_interp: true")
        file_dt = float(d[0])
    elif file_dt is None:
        return
    file_dt = float(file_dt)
    if abs(dt - file_dt) <= max(0.1, 1e-3 * file_dt):
        return
    raise ValueError(
        f"numerics.dt = {dt:g} s differs from the file step {file_dt:g} s in "
        "discrete mode (temporal_interp: false), where every step reads the "
        "next record while the particles move by dt; set dt equal to the "
        "file step, or set temporal_interp: true to sub-step between records")


def run_single_realization(
    mesh: Mesh,
    ds,
    reader,
    init_state: ParticleState,
    mode: ModeConfig,
    numerics: NumericsConfig,
    start_time_idx: int,
    n_steps: Optional[int] = None,
    n_time_total: Optional[int] = None,
    on_exhausted: str = "stop",
    verbose: bool = False,
    shared_cache=None,
    *,
    end_time_idx: Optional[int] = None,
    writer=None,
    checkpoint_freq: int = 0,
    temporal_interp: bool = False,
    file_times: Optional[np.ndarray] = None,
    file_dt: Optional[float] = None,
    sim_time_start: Optional[float] = None,
    realization_id: int = 0,
    start_step_i: int = 0,
    sample_vars: Optional[List[str]] = None,
    sample_interp: str = "leastsq",
    output_sample_vars: Optional[List[str]] = None,
    start_wet_streak: Optional[np.ndarray] = None,
    particle_offset: int = 0,
) -> TrajectoryRecord:
    """Run one realization of particle tracking.

    Parameters
    ----------
    mesh : Mesh
    ds : xr.Dataset (or similar)
        Open velocity dataset.
    reader : reader class
        Must have ``read_velocities(ds, timestep, mesh, mode)``.
    init_state : ParticleState
        Initial particle positions (will be copied).
    mode : ModeConfig
    numerics : NumericsConfig
    start_time_idx : int
        First file time index.
    n_steps : int or None
        Number of tracking steps.  Takes priority over ``end_time_idx``.
    n_time_total : int or None
        Total number of time steps in the dataset (for bounds checking
        and cycling).  If None, defaults to ``ds.sizes['time']``.
    on_exhausted : str
        ``"stop"`` (default) or ``"cycle"``.
    verbose : bool
        Print progress.
    end_time_idx : int or None
        Legacy parameter.  If ``n_steps`` is not given, computes
        ``n_steps = end_time_idx - start_time_idx``.
    writer : TrajectoryWriter or None
        If given, trajectory snapshots are streamed to this writer
        (must already be opened) and the returned in-memory
        TrajectoryRecord stays empty (records go to one or the other).
    checkpoint_freq : int
        If > 0 and writer is given, write a restart checkpoint every
        this many steps.
    temporal_interp : bool
        If True, linearly interpolate velocity fields between adjacent
        file time steps.  Requires ``file_times``.
    file_times : np.ndarray or None
        Seconds-since-start for each file time index (from
        ``extract_time_info``).  Required when ``temporal_interp=True``.
    file_dt : float or None
        Optional bookkeeping; nominal file time step in seconds.  Not
        required — time brackets are derived from ``file_times``.
    sim_time_start : float or None
        Override the initial simulation time (seconds, same reference as
        ``file_times``).  Used by ``start_date`` to begin between file
        snapshots.  If None, defaults to ``file_times[start_time_idx]``.
    realization_id : int
        Index of the current realization in a multi-run loop.  Used as
        the second key in the diffusion RNG tuple so that distinct
        realizations get independent stochastic streams.  Default 0.
    start_step_i : int
        Global step offset for the diffusion RNG and checkpoint
        bookkeeping.  Set to ``load_restart(...)["step_i"] + 1`` when
        resuming from a writer checkpoint so the diffusion stream
        continues from the right point.  The local loop counter inside
        this function still runs ``0..n_steps - 1``; the diffusion
        kick keys on ``start_step_i + step_i`` and the checkpoint
        records the same global value.  Default 0 (fresh run).
    sample_vars : list of str or None
        Node-based dataset variable names to sample at each record point.
        ``None`` or empty → dormant path (no reads, no sampling, no new
        output variables — byte-identical to no-sampling).
    sample_interp : str
        Horizontal interpolation method for scalar sampling
        (``"nearest" | "leastsq" | "barycentric" | "idw"``).  Default
        ``"leastsq"``.
    output_sample_vars : list of str or None
        Subset of ``sample_vars`` that should actually appear in the
        trajectory output.  ``sample_vars`` may include variables read
        only for internal use (e.g. the beaching ``dry_mask_var``); those
        not in ``output_sample_vars`` are dropped before writing.  ``None``
        → equal to ``sample_vars`` (everything read is written).
    particle_offset : int
        Global index of ``init_state``'s particle 0 in the full (unsharded)
        release.  Every random draw (diffusion kicks, the vertical Az kick,
        stochastic beaching and resuspension) is keyed by ``particle_offset
        + i``, so a shard of a release draws exactly the numbers of the
        unsharded run.  Default 0.
    start_wet_streak : np.ndarray or None
        Initial beaching consecutive-wet-step counter (int32 ``(P,)``).
        Used by ``load_restart`` to resume the resuspension state bit-for-bit.
        ``None`` → all-zeros.

    Returns
    -------
    TrajectoryRecord
    """
    from .core.release import check_particle_offset
    # None (a restart written without an offset) means an unsharded run
    particle_offset = check_particle_offset(
        0 if particle_offset is None else particle_offset)
    # -- resolve n_steps (backward compat) --
    if n_steps is None:
        if end_time_idx is not None:
            # one rule with config.resolve_time_params: end - start steps
            # (this path added one more step until 2026-10-04)
            n_steps = end_time_idx - start_time_idx
            # Legacy: backward tracking started from end_time_idx
            if mode.time_dir < 0:
                start_time_idx = end_time_idx
        else:
            raise ValueError("Must provide either n_steps or end_time_idx")

    if n_time_total is None:
        n_time_total = ds.sizes["time"]

    step_fn = get_integrator(numerics.integrator)
    dt = numerics.dt
    time_dir = mode.time_dir
    output_freq = numerics.output_freq
    direction = 1 if time_dir >= 0 else -1
    # A resume computes ``start0 + n * time_dir`` with the config's FLOAT
    # time_dir; a float index reached xarray's isel and the checkpoint.
    if int(start_time_idx) != start_time_idx:
        raise ValueError(
            f"start_time_idx must be a whole record index, got "
            f"{start_time_idx!r}")
    start_time_idx = int(start_time_idx)
    # Euler has a single stage, so vel_next/dt_to_next don't affect it —
    # suppress the dt > dt_to_next warning for Euler to avoid false positives.
    _is_multistage = numerics.integrator in ("rk2", "rk4", "rk45")

    # Diffusion gate.  The composition rules spell out the
    # composition matrix; ``_resolve_composition`` enforces it.
    diff_cfg: DiffusionConfig = numerics.diffusion
    integrator_name = numerics.integrator
    composition_resolved = _resolve_composition(diff_cfg, integrator_name)
    # Uniform-scheme SCALARS carry the same diagonal-tensor semantics as
    # the em/m1 fields: under a rotated frame an unequal pair is applied
    # along the ROTATED axes (the ellipse turns with the frame).  The
    # field case warns in the readers; the scalar case warns here.
    if (diff_cfg.enabled and diff_cfg.scheme == "uniform"
            and getattr(mesh, "rotation_angles", None) is not None
            and diff_cfg.kh_zonal != diff_cfg.kh_meridional):
        warnings.warn(
            "ANISOTROPIC uniform Kh (kh_zonal != kh_meridional) under "
            "mesh.rotation: the pair is applied along the ROTATED axes, "
            "not geographic east/north. Use equal values under rotation; "
            "see docs/coordinate_frame.md §3.",
            RuntimeWarning,
        )

    # Horizontal u/v interpolation method ("leastsq" default /
    # "barycentric" / "edge_leastsq").  A step-0 guard below pins it
    # against the actual uv_layout the reader returns.
    uv_interp_resolved = numerics.uv_interp
    # Boundary-ghost value for edge_leastsq (inert for other methods).
    edge_lsq_bc_resolved = getattr(numerics, "edge_leastsq_bc", "freeslip")

    # Mesh-derived default for ``dres_m``:
    # median element edge length, clipped to [100 m, 50 km], computed
    # once per realization.  ``DiffusionConfig.dres_m`` overrides if
    # the user sets it explicitly.  Only needed for em/m1 schemes.
    dres_m_resolved = 0.0
    if diff_cfg.enabled and composition_resolved in ("combined", "split_rk"):
        if diff_cfg.dres_m is not None:
            dres_m_resolved = float(diff_cfg.dres_m)
        else:
            dres_m_resolved = _compute_default_dres_m(mesh)

    # -- Beaching gate (TODO 2.2).  Dormant (None / enabled=False) →
    # byte-identical no-beaching path: no topology build, no hook, no
    # status remap.  Resolve all per-run constants once here.
    beach_cfg = getattr(numerics, "beaching", None)
    beach_enabled = beach_cfg is not None and beach_cfg.enabled
    beach_run_cfg = None
    if beach_enabled:
        from .numerics.beaching import BeachRunConfig
        # Entry-point fail-fasts (spec 9/9b).  ``run_tracking`` performs the
        # same checks against the dataset and wires the internal read list,
        # but a direct ``run_single_realization`` / ``track_particles`` caller
        # bypasses that path — so re-assert them here rather than silently
        # substituting zeros (use_eta) or skipping the dry-mask read.
        eta_names = ["ssh", "eta", "zos"]
        cfg_eta = getattr(mode, "var_eta", None)
        if cfg_eta:
            eta_names.append(cfg_eta)
        if beach_cfg.use_eta and not any(k in ds for k in eta_names):
            raise ValueError(
                "beaching.use_eta=true requires an ssh/eta/zos (or the "
                "configured mode.var_eta) variable in the dataset (the "
                "FESOM2 reader otherwise substitutes zeros, hiding all "
                "eta-driven drying).")
        if beach_cfg.dry_mask_var and (
                sample_vars is None
                or beach_cfg.dry_mask_var not in sample_vars):
            raise ValueError(
                "beaching.dry_mask_var requires the run_tracking wiring (the "
                "variable must be in the internal sample list).")
        beach_crit = frozenset(beach_cfg.criteria)
        # Boundary topology + containing-element lookup are needed when
        # EITHER the deterministic coastal criterion OR the stochastic
        # criterion is active (the stochastic zone-OR keys on topology
        # data availability — Task-3 contract).
        need_topo = bool({"coastal_element", "stochastic"} & beach_crit)
        elem_coastal = None
        if need_topo:
            from .core.mesh import build_boundary_info
            build_boundary_info(mesh)
            elem_coastal = mesh._elem_is_coastal
        # seed falls back to the diffusion seed when unset (spec).
        beach_seed = (beach_cfg.seed if beach_cfg.seed is not None
                      else numerics.diffusion.seed)
        beach_zone = (beach_cfg.zone_depth if beach_cfg.zone_depth is not None
                      else 2.0 * beach_cfg.depth_min)
        # Bundle all per-run constants once (house pattern; the hook then
        # carries only per-step variability).
        beach_run_cfg = BeachRunConfig(
            cfg=beach_cfg, crit=beach_crit, need_topo=need_topo,
            elem_coastal=elem_coastal, seed=beach_seed, zone=beach_zone,
            pid_offset=int(particle_offset),
        )
    wet_streak = (np.asarray(start_wet_streak, dtype=np.int32).copy()
                  if start_wet_streak is not None
                  else np.zeros(init_state.n_particles, dtype=np.int32))
    if output_sample_vars is None:
        output_sample_vars = sample_vars

    # Temporal interpolation setup — file_dt is not required, file_times suffice.
    # Requested interpolation WITHOUT a physical axis is an error: the old
    # silent fallback advanced one snapshot per step instead (review
    # 2026-09-22).
    if temporal_interp and file_times is None and file_dt is None:
        # a direct caller passed no record times: read them from the dataset
        # (run_tracking always passes them, midpoint-shifted when averaged)
        file_times, file_dt = extract_time_info(ds)
    if temporal_interp and file_times is None and file_dt:
        # the caller declared the snapshot interval (as run_tracking does
        # with numerics.file_dt): synthesise the axis (review 2026-09-23)
        file_times = np.arange(n_time_total, dtype=np.float64) * float(file_dt)
    if temporal_interp and file_times is None:
        raise ValueError(
            "temporal_interp is on but the dataset has no usable time axis "
            "(no decodable 'time' coordinate with physical units); set "
            "numerics.file_dt to declare the snapshot interval, or turn "
            "temporal_interp off")
    if file_times is not None and len(file_times) > n_time_total:
        # the run reads records 0..n_time_total-1 only: ONE active axis for
        # the brackets, the exhaustion checks, the cycle period and the
        # record clock (PR review 2026-10-14)
        file_times = np.asarray(file_times)[:n_time_total]
    # direct callers (track_particles, chain drivers) often pass no record
    # times: read the axis from the dataset for this check only (passing it
    # on would switch on the discrete path's vel_next).
    _chk_times, _chk_dt = file_times, file_dt
    if not temporal_interp and file_times is None:
        # the dataset's own axis decides when it has one (a direct caller's
        # file_dt let monthly records pass; review 2026-10-04)
        _ax_times, _ax_dt = extract_time_info(ds)
        if _ax_times is not None:
            _chk_times, _chk_dt = _ax_times, (_ax_dt if file_dt is None else file_dt)
    _check_discrete_dt(numerics.dt, temporal_interp, _chk_dt, _chk_times,
                       n_records=n_time_total)
    use_tinterp = temporal_interp and file_times is not None
    _sigma3d = (getattr(mesh, "vertical_coord", "zlevel") == "sigma"
                and bool(getattr(mode, "Fdrift3D", False)))
    # PCHIP temporal reconstruction (4-snapshot) — only in tinterp mode.
    recon_enabled = use_tinterp and numerics.temporal_reconstruction == "pchip"
    vel_cache = _VelCache(
        velocity_float32=getattr(numerics, "velocity_float32", False),
        prefetch_size=getattr(numerics, "prefetch_size", 1),
        direction=direction, n_time=n_time_total,
        cycle=(on_exhausted == "cycle"),
        shared=shared_cache)
    if use_tinterp:
        # simulation time in seconds, relative to file_times[0]
        sim_time = sim_time_start if sim_time_start is not None else file_times[start_time_idx]

    # Copy initial state
    state = init_state.replace(
        lon=init_state.lon.copy(),
        lat=init_state.lat.copy(),
        z=init_state.z.copy(),
        status=init_state.status.copy(),
        nearest_node=init_state.nearest_node.copy(),
    )
    if (mesh.metadata or {}).get("exact_nearest", False):
        # a checkpoint written before the guard carries approximate anchors;
        # re-anchor once so the first unfused step starts exact too (the
        # fused kernels re-anchor in-kernel anyway) — idempotent otherwise
        alive0 = np.flatnonzero(state.alive_mask)
        if alive0.size:
            nn0 = state.nearest_node.copy()
            nn0[alive0] = find_nearest_nodes(mesh, state.lon[alive0], state.lat[alive0])
            state = state.replace(nearest_node=nn0)

    # -- Periodic boundary windows (idealized channel domains) --
    # Resolved once; None/None keeps every wrap call a cheap no-op.
    # Seeds placed in the ghost band (delayed-release ones included) fold
    # in before the first step.
    periodic_lon = getattr(numerics, "periodic_lon", None)
    periodic_lat = getattr(numerics, "periodic_lat", None)
    if periodic_lon is not None or periodic_lat is not None:
        # Windows are MESH-FRAME coordinates: combining them with a
        # rotated mesh would fold rotated longitudes while the user
        # reasons in geographic ones — silent nonsense; fail fast (the
        # same posture every rotation-incompatible subsystem takes).
        if getattr(mesh, "rotation_angles", None) is not None:
            raise ValueError(
                "numerics.periodic_lon/periodic_lat apply to MESH-FRAME "
                "coordinates and cannot be combined with mesh.rotation "
                "(the fold would act on rotated longitudes)."
            )
        # Ghost-band / branch sanity (branch-aware; raises on the
        # silent-failure configs, warns per missing ghost side).
        _validate_periodic_windows(mesh, periodic_lon, periodic_lat,
                                   warn=True)
    state = _apply_periodic_wrap(state, mesh, periodic_lon, periodic_lat,
                                 include_dormant=True)

    # -- Edge-walk element tracking (numerics.elem_search: walk) --
    # Config is authoritative: "walk" seeds the carried elements with ONE
    # global search at initialization (a restart state that already
    # carries them is resumed warm); "local" strips any carried elems so
    # a walk-run checkpoint resumed under local behaves byte-identically
    # to a pure local run.  The integrators key on state.elem is not None.
    elem_search = getattr(numerics, "elem_search", "local")
    if elem_search == "walk":
        if state.elem is None:
            from .core.mesh import find_nearest_nodes as _fnn
            nn0 = _fnn(mesh, state.lon, state.lat)
            el0, _ = find_containing_elements(mesh, state.lon, state.lat,
                                              nn0)
            state = state.replace(elem=el0.astype(np.int32))
        else:
            state = state.replace(elem=state.elem.copy())
    elif state.elem is not None:
        state = state.replace(elem=None)

    # -- Release positions (output format 2) --
    # A run that starts at global step 0 releases its particles here: the
    # seeds after the periodic fold, NOT_RELEASED ones included.  They ride
    # the state and every checkpoint, so each segment of a resumed chain
    # writes the same release_* variables.
    if state.release_lon is None:
        if start_step_i == 0:
            state = state.replace(release_lon=state.lon.copy(),
                                  release_lat=state.lat.copy(),
                                  release_z=state.z.copy())
        else:
            warnings.warn(
                "resumed run without release positions (a restart written "
                "before output format 2): release_lon/lat/z are NaN in this "
                "output", RuntimeWarning, stacklevel=2)

    # Rotated-frame provenance, even for caller-supplied writers: run_tracking
    # also sets this via construction metadata, so a doubled set is harmless
    # (overwrites with the same value).  Output positions are geographic
    # regardless; the attr documents the run's frame.
    angles = getattr(mesh, "rotation_angles", None)
    if writer is not None and angles is not None:
        writer.set_global_attr("rotation_angles", list(angles))
    # Provenance: stamped here too, so outputs written through this function
    # with a caller's writer (resumable-chain drivers, track_particles) carry
    # the engine and the W contract (review 2026-09-23); run_tracking's
    # construction metadata sets the same values.
    if writer is not None:
        writer.set_global_attr("kelp_version", _kelp_version())
        # the shard's place in the full release (random-draw keys; the
        # sharding concat checks it against the shard order)
        writer.set_global_attr("particle_offset", int(particle_offset))
        # whether this run draws random numbers at all (diffusion, stochastic
        # stranding or resuspension): the concat tool refuses shards whose
        # offsets do not follow the merge order when they do (or say nothing)
        writer.set_global_attr("random_draws", int(_draws_random_numbers(numerics)))
        writer.set_global_attr("temporal_interp", int(bool(temporal_interp)))
        writer.set_global_attr("temporal_reconstruction", str(
            getattr(numerics, "temporal_reconstruction", "linear")))
        if getattr(mode, "Fdrift3D", False):
            writer.set_global_attr("w_interp", str(
                getattr(numerics, "w_interp", "leastsq")))
            writer.set_global_attr("w_positive", str(
                getattr(mode, "w_positive", "auto")))
            writer.set_global_attr("w_contract", "reader-converted-2026-09-23")
    # Same pattern for periodic windows: caller-supplied writers get the
    # attrs too (run_tracking also sets them via construction metadata —
    # a doubled set overwrites with the same value).
    if writer is not None:
        if periodic_lon is not None:
            writer.set_global_attr("periodic_lon", list(periodic_lon))
        if periodic_lat is not None:
            writer.set_global_attr("periodic_lat", list(periodic_lat))

    # -- Record clock, epoch and release output (output format 2) --
    # One label formula for both modes, T_g0 + (g + 1)·dt·direction, with
    # T_g0 the time of the start of global step 0: labels do not depend on
    # how a run is cut into chain segments.  The epoch is stamped only when
    # the clock is measured on the forcing axis.
    _axis = file_times if file_times is not None else _chk_times
    if _axis is not None and len(_axis) > n_time_total:
        # the run uses records 0..n_time_total-1 only: the cycle period of
        # the reads, so of the clock too (round 2 review)
        _axis = _axis[:n_time_total]
    t_g0, on_axis = _record_clock_origin(
        use_tinterp=use_tinterp, sim_time=(sim_time if use_tinterp else None),
        start_time_idx=start_time_idx, axis=_axis, n_time_total=n_time_total,
        on_exhausted=on_exhausted, dt=dt, direction=direction,
        start_step_i=start_step_i)
    release = _release_arrays(state, mesh, t_g0, dt, direction)
    if writer is not None:
        epoch = time_axis_epoch(ds) if on_axis else None
        cycle_period = None
        if on_exhausted == "cycle":
            cycle_period = (
                float(_axis[-1] - _axis[0] + (_axis[-1] - _axis[-2]))
                if on_axis and _axis is not None and len(_axis) > 1
                else float(n_time_total) * float(dt))
        if epoch is not None:
            writer.set_time_units(epoch[0], epoch[1],
                                  cycle_period=cycle_period)
        else:
            writer.set_time_units("seconds", origin_note=NO_EPOCH_NOTE,
                                  cycle_period=cycle_period)
        if int(particle_offset):
            writer.set_trajectory_ids(int(particle_offset)
                                      + np.arange(state.n_particles))
        writer.write_release(*release)
        # every segment and shard gets the same variables, even one whose
        # particles are all dead and so never record a velocity
        writer.declare_velocity(w=bool(getattr(mode, "Fdrift3D", False)))

    record = TrajectoryRecord()
    record.set_release(*release)
    vel = None
    _step_times = [] if verbose else None
    _phase_times = [] if verbose else None  # (io, step, beach, rec) per step, 6.6
    _downgrade_count = 0  # how many steps had to drop time-aware RK to steady
    _recon_edge_warned = False  # one-time warn when PCHIP falls back to linear

    # For human-readable time display in verbose mode
    _ds_times = None
    _t0_epoch = None  # epoch seconds of file_times[0], for sim_time → datetime
    if verbose and "time" in ds.dims:
        _ds_times = ds["time"].values
        if np.issubdtype(_ds_times.dtype, np.datetime64):
            _t0_epoch = _ds_times[0]

    # Last fully-executed (global step, file_idx) for the final checkpoint;
    # stays None if the loop breaks before any step runs (see end of loop).
    last_global_step_i = None
    last_file_idx = start_time_idx

    for step_i in range(n_steps):
        _t_step_start = _time.perf_counter() if verbose else 0
        # Global step index — drives the diffusion RNG and checkpoint
        # bookkeeping so a checkpoint→resume run reproduces the
        # continuous run bit-for-bit (the counter-RNG keying rule).
        global_step_i = start_step_i + step_i

        # -- Delayed release (TODO 2.5): activate particles whose release step
        #    has arrived (NOT_RELEASED -> ALIVE) BEFORE the step, so they advect
        #    starting this step.  No-op / same object when release_step is None,
        #    so the default path is byte-identical.
        state = activate_released(state, global_step_i)

        # Defaults: time-aware extras passed to integrator kwargs.  None
        # means steady mode (integrator ignores).
        vel_next = None
        dt_to_next = None
        recon = None  # PCHIP 4-snapshot bundle (None => linear stage build)

        if use_tinterp:
            # --- Temporal interpolation path ---
            # direction makes the payload hand RK stages the snapshot the
            # step is moving TOWARD (backward: the previous one), and an
            # on-snapshot backward query bracket downward.
            bracket = _resolve_time_bracket(
                sim_time, file_times, n_time_total, on_exhausted,
                direction=direction,
            )
            if bracket is None:
                if verbose:
                    print(f"  sim_time={sim_time:.1f}s out of file range, stopping.")
                break
            # PCHIP 4-snapshot reconstruction (multistage integrators only —
            # Euler ignores ``recon`` and would just waste the reads).  Resolve
            # the stencil FIRST (pure index math) so the payload keeps its
            # flanking snapshots cached — a sliding window of one read per step
            # instead of evicting + re-reading the two flanks every step.
            stencil = (_resolve_recon_stencil(bracket, n_time_total)
                       if (recon_enabled and _is_multistage) else None)
            vel, vel_next, dt_to_next, file_idx = _build_vel_payload(
                bracket, vel_cache, reader, ds, mesh, mode,
                diffusion=diff_cfg, sample_vars=sample_vars,
                direction=direction,
                keep_extra=(set(stencil[0]) if stencil is not None else None),
            )
            if stencil is not None:
                idxs4, q_time = stencil
                snaps4 = [vel_cache.get(i, reader, ds, mesh, mode,
                                        diffusion=diff_cfg,
                                        sample_vars=sample_vars)
                          for i in idxs4]
                times4 = [float(file_times[i]) for i in idxs4]
                recon = (snaps4, times4, q_time, direction)
            elif recon_enabled and _is_multistage and not _recon_edge_warned:
                warnings.warn(
                    "temporal_reconstruction='pchip': fewer than 4 "
                    "snapshots at the range edge; this step uses linear "
                    "time interpolation. Further occurrences in this "
                    "realization suppressed.",
                    RuntimeWarning, stacklevel=2,
                )
                _recon_edge_warned = True
        else:
            # --- Discrete path ---
            file_idx = start_time_idx + step_i * direction
            # the checkpoint keeps the UNWRAPPED index: a cycle-mode resume
            # from ``file_idx + 1`` then labels like the uncut run (the
            # wrapped one shifted the record clock by whole forcing periods;
            # review 2026-10-13)
            ckpt_file_idx = file_idx

            if on_exhausted == "cycle":
                file_idx = file_idx % n_time_total
            else:  # "stop"
                if file_idx < 0 or file_idx >= n_time_total:
                    if verbose:
                        print(f"  Time index {file_idx} out of range "
                              f"[0, {n_time_total}), stopping.")
                    break

            # Time-aware for discrete path: velocity samples at adjacent
            # snapshots; RK stages interpolate in time across dt.
            # Requires file_times to know inter-snapshot spacing.
            next_idx_raw = file_idx + direction
            have_next = False
            if file_times is not None:
                if on_exhausted == "cycle":
                    next_idx = next_idx_raw % n_time_total
                    have_next = True
                elif 0 <= next_idx_raw < n_time_total:
                    next_idx = next_idx_raw
                    have_next = True

            if have_next:
                vel = vel_cache.get(file_idx, reader, ds, mesh, mode,
                                    diffusion=diff_cfg, sample_vars=sample_vars)
                # vel_next is read with sample_vars too: the cache stores one
                # field per index, and this entry is reused as ``vel`` on the
                # next step (where its scalars ARE sampled), so it must carry
                # them now or the next record would see a scalar-less field.
                vel_next = vel_cache.get(next_idx, reader, ds, mesh, mode,
                                         diffusion=diff_cfg,
                                         sample_vars=sample_vars)
                vel_cache.evict_except({file_idx, next_idx})
                # dt_to_next in seconds between file[file_idx] → file[next_idx]
                ta = file_times[file_idx]
                tb = file_times[next_idx]
                if on_exhausted == "cycle" and direction > 0 and tb <= ta:
                    # Forward wrap (last → 0): use the trailing interval
                    last_dt = (file_times[-1] - file_times[-2]
                               if n_time_total > 1 else 1.0)
                    tb = ta + last_dt
                elif direction < 0 and tb >= ta:
                    # Backward step: dt_to_next is the gap back to the
                    # previous snapshot.  For irregular spacing this is
                    # the trailing interval (consistent with forward-cycle
                    # wrap above), not the leading one.
                    last_dt = (file_times[-1] - file_times[-2]
                               if n_time_total > 1 else 1.0)
                    tb = ta - last_dt
                dt_to_next = abs(tb - ta)
            else:
                # No next snapshot — use steady mode for this step
                vel = vel_cache.get(file_idx, reader, ds, mesh, mode,
                                    diffusion=diff_cfg, sample_vars=sample_vars)
                vel_cache.evict_except({file_idx})

        # uv_interp x uv_layout backstop (both tinterp & discrete paths have
        # set `vel` by now).  Two cases reach this step-0 guard instead of
        # run_tracking's pre-writer probe: (1) readers that don't implement
        # probe_uv_layout, and (2) direct run_single_realization callers that
        # never pass through run_tracking at all (cheap: one comparison).
        if step_i == 0:
            _validate_uv_compat(getattr(vel, "uv_layout", "element"),
                                uv_interp_resolved)

        # Downgrade to steady if tracking dt exceeds the bracket
        # (start-between-snapshots, end-of-range, irregular spacing).
        # RK stages lose their intra-step time order here — warn once.
        if dt_to_next is not None and dt > dt_to_next + 1e-9:
            if _downgrade_count == 0 and _is_multistage:
                warnings.warn(
                    f"step {step_i}: dt={dt:.3g}s exceeds dt_to_next="
                    f"{dt_to_next:.3g}s; RK stages fall back to steady "
                    "velocity for this step. Further occurrences in this "
                    "realization suppressed; see summary at end.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            _downgrade_count += 1
            vel_next = None
            dt_to_next = None
            recon = None  # dt exceeds the bracket → PCHIP stencil invalid too

        # Sigma meshes integrate s = z / H and place the particle at z = s H:
        # every step gets the surface at its start and at its end from ONE
        # function of time (PCHIP where a stencil exists and PCHIP is on,
        # else the linear payload blend), so the end of a step and the start
        # of the next are bit-identical — s is continuous across steps even
        # where consecutive steps interpolate differently (a steady-fallback
        # step between PCHIP steps), and a resumed run reproduces the
        # continuous one (review of PR #102 / fix/sigma-step).
        sigma_eta_start = sigma_eta_end = None
        if _sigma3d:
            if use_tinterp:
                _pch = recon_enabled and _is_multistage
                _ea = (on_exhausted, direction, vel_cache, reader, ds, mesh,
                       mode, diff_cfg, sample_vars)
                sigma_eta_start = _eta_at_time(
                    sim_time, file_times, n_time_total, *_ea, pchip=_pch)
                sigma_eta_end = _eta_at_time(
                    sim_time + dt * direction, file_times, n_time_total, *_ea,
                    pchip=_pch)
            else:
                # discrete mode: the step starts at this record and ends at
                # the next one
                nxt = file_idx + direction
                if on_exhausted == "cycle":
                    nxt %= n_time_total
                if 0 <= nxt < n_time_total:
                    sigma_eta_end = vel_cache.get(
                        nxt, reader, ds, mesh, mode, diffusion=diff_cfg,
                        sample_vars=sample_vars).eta

        # Diffusion args for this step (None when disabled).
        diff_args = None
        if diff_cfg.enabled:
            diff_args = DiffusionStepArgs(
                scheme=diff_cfg.scheme,
                kh_x=diff_cfg.kh_zonal,
                kh_y=diff_cfg.kh_meridional,
                seed=diff_cfg.seed,
                real_id=realization_id,
                step_i=global_step_i,
                composition=composition_resolved,
                dres_m=dres_m_resolved,
                kh_eval_position=diff_cfg.kh_eval_position,
                az_enabled=getattr(diff_cfg, "az_enabled", False),
                az_const=getattr(diff_cfg, "az_const", 0.0),
                az_requires_field=bool(getattr(diff_cfg, "var_az", None)),
                pid_offset=int(particle_offset),
            )

        # Snapshot of pre-step status for the on_failure remap (the
        # integrator returns a NEW state via .replace, so no copy needed:
        # ``prev_status`` keeps referencing the old array).
        prev_status = state.status
        # Rotated frame: the start position, where the integrators evaluate
        # the recorded u/v — the base point for their un-rotation at record
        # time (copies: cheap, and immune to any in-place update further
        # down the step).  rk45 evaluates them at the END of the step,
        # before a diffusion kick: it reports that point in ``rk45_diag``.
        vel_base = None
        rk45_diag = None
        if getattr(mesh, "rotation_angles", None) is not None:
            if numerics.integrator == "rk45":
                rk45_diag = {}
            else:
                vel_base = (state.lon.copy(), state.lat.copy())

        # Phase timing (6.6, verbose only): everything before this point
        # is payload build / prep ("io"); kernel-internal nn/interp/move
        # phases are not separable for the fused kernels, so the
        # breakdown is at the orchestration level.
        _t_io_end = _time.perf_counter() if verbose else 0.0

        # Step particles
        _sig_kw = {}
        if sigma_eta_start is not None:
            _sig_kw["sigma_eta_start"] = sigma_eta_start
        if sigma_eta_end is not None:
            _sig_kw["sigma_eta_end"] = sigma_eta_end
        state = step_fn(
            state, vel, mesh, dt, time_dir, layer=0, **_sig_kw,
            Dmin=numerics.Dmin, z0b_min=numerics.z0b_min,
            w_interp=numerics.w_interp,
            uv_interp=uv_interp_resolved,
            edge_leastsq_bc=edge_lsq_bc_resolved,
            vel_next=vel_next, dt_to_next=dt_to_next,
            recon=recon,
            diffusion=diff_args,
            rk45_tol_m=getattr(numerics, "rk45_tol_m", 1.0),
            rk45_h_min=getattr(numerics, "rk45_h_min", None),
            **({"rk45_diag": rk45_diag} if rk45_diag is not None else {}),
        )
        if rk45_diag:
            vel_base = (rk45_diag["lon_end"], rk45_diag["lat_end"])

        # Periodic fold BEFORE beaching/record: those consumers must see
        # in-window positions.  No-op (same object) when off.
        state = _apply_periodic_wrap(state, mesh, periodic_lon,
                                     periodic_lat)
        _t_step_end = _time.perf_counter() if verbose else 0.0

        # -- Beaching post-step hook (TODO 2.2) --
        # Runs after the integrator step, BEFORE sim_time advance / record.
        # Dormant unless beaching is enabled.
        if beach_enabled:
            state, wet_streak = _apply_beaching_hook(
                state, prev_status, mesh, vel, beach_run_cfg, wet_streak,
                realization_id, global_step_i, dt,
            )
        _t_beach_end = _time.perf_counter() if verbose else 0.0

        # Advance simulation time (temporal interp path)
        if use_tinterp:
            sim_time += dt * direction

        # Output check — on the GLOBAL step count, so a run resumed in
        # chunks (start_step_i > 0) records exactly the steps the
        # uninterrupted run records.  A per-call counter restarted the phase
        # at every chunk: 91-step chunks with output_freq=5 wrote a 6-step
        # gap at each boundary (review 2026-09-22, F04).
        if (global_step_i + 1) % max(int(output_freq), 1) == 0:
            # output format 2: the same label in both modes (seconds on
            # the forcing axis at the end of the recorded step)
            time_val = t_g0 + (global_step_i + 1) * dt * direction

            # Scalar sampling at the record point (TODO 2.4).  Dormant
            # unless sample_vars is set.
            sampled = None
            if sample_vars:
                if use_tinterp:
                    # Re-lerp scalars (and eta) to the post-step sim_time —
                    # the exact recorded timestamp.  Cache hits make this
                    # cheap.  At end-of-range the bracket may not resolve;
                    # fall back to the in-hand (start-of-step) field.
                    # direction only matters here for WHICH snapshots the
                    # cache ends up holding (vel_next is discarded): the
                    # re-lerped field value at sim_time is direction-free.
                    rec_bracket = _resolve_time_bracket(
                        sim_time, file_times, n_time_total, on_exhausted,
                        direction=direction,
                    )
                    if rec_bracket is not None:
                        rec_vel, _vn, _dtn, _fi = _build_vel_payload(
                            rec_bracket, vel_cache, reader, ds, mesh, mode,
                            diffusion=diff_cfg, sample_vars=sample_vars,
                            direction=direction,
                        )
                    else:
                        rec_vel = vel
                    sampled = _sample_record_fields(
                        state, mesh, rec_vel, sample_interp,
                        Dmin=numerics.Dmin)
                else:
                    # Discrete mode: the field at the record's instant is the
                    # NEXT forcing record (the end of the step; normally a
                    # cache hit, it is the next step's start field); without
                    # one (last step of a stop run) the start-of-step field.
                    rec_vel = vel
                    nxt = file_idx + direction
                    if on_exhausted == "cycle":
                        nxt %= n_time_total
                    if 0 <= nxt < n_time_total:
                        rec_vel = vel_cache.get(nxt, reader, ds, mesh, mode,
                                                diffusion=diff_cfg,
                                                sample_vars=sample_vars)
                    sampled = _sample_record_fields(
                        state, mesh, rec_vel, sample_interp,
                        Dmin=numerics.Dmin)

            # Drop internal-only sampled vars (e.g. the beaching dry_mask_var
            # read for the hook but not user-requested) before output.
            if sampled and output_sample_vars is not None:
                _keep = set(output_sample_vars)
                sampled = {k: v for k, v in sampled.items() if k in _keep}
                if not sampled:
                    sampled = None

            # Rotated-frame OUTPUT: positions and vel_u/vel_v diagnostics
            # un-rotated to geographic on a COPY (the internal state stays
            # rotated).  No-op / same object on a geographic mesh, so the
            # default path is byte-identical.  The sampled dict is left as
            # computed from the ROTATED state — scalars are frame-invariant
            # and sampling lives on the rotated mesh.
            rec_state = _geo_view_for_record(state, mesh, vel_base)
            # Delayed release: frozen particles are written as NaN positions
            # (+ vel/sampled) so the output trajectory "starts" at release; the
            # status column still records NOT_RELEASED.  Same object when none
            # are dormant (byte-identical default path).
            rec_state, sampled = mask_dormant_for_record(rec_state, sampled)
            # A velocity is defined only where the particle was ALIVE through
            # the whole step (output format 2): NaN in the record of a death
            # or stranding, after it, and for a particle resuspended this step.
            rec_state = mask_inactive_velocity(rec_state, prev_status)
            # The integrators carry time_dir·velocity (the tracking
            # velocity); the output holds the FLOW velocity in both
            # directions (output format 2): flip it back in a backward run.
            if direction < 0 and (rec_state.u is not None
                                  or rec_state.w is not None):
                rec_state = rec_state.replace(
                    u=None if rec_state.u is None else -rec_state.u,
                    v=None if rec_state.v is None else -rec_state.v,
                    w=None if rec_state.w is None else -rec_state.w)
            if writer is not None:
                writer.append(rec_state, float(time_val), sampled=sampled)
            else:
                record.append(rec_state, time_val, sampled=sampled)

        # Checkpoint — store the GLOBAL step index so that resuming
        # callers can pass ``step_i + 1`` as the next ``start_step_i``.
        if (checkpoint_freq > 0 and writer is not None
                and (global_step_i + 1) % checkpoint_freq == 0):
            writer.write_checkpoint(
                state, global_step_i,
                file_idx if use_tinterp else ckpt_file_idx,
                wet_streak=wet_streak if beach_enabled else None,
                rotation_angles=getattr(mesh, "rotation_angles", None),
                release_step=state.release_step)

        if verbose:
            _t_now = _time.perf_counter()
            step_dt = _t_now - _t_step_start
            _step_times.append(step_dt)
            # Orchestration-level phase breakdown (TODO 6.6): io = payload
            # build + prep, step = the integrator kernel(s), beach = the
            # beaching hook (0 when dormant), rec = sampling + record /
            # writer append (incl. checkpoint when it fires this step).
            _ph_io = _t_io_end - _t_step_start
            _ph_step = _t_step_end - _t_io_end
            _ph_beach = _t_beach_end - _t_step_end
            _ph_rec = _t_now - _t_beach_end
            _phase_times.append((_ph_io, _ph_step, _ph_beach, _ph_rec))
            counts = count_by_status(state.status)
            # Build time label: step number + current datetime or file index
            step_label = f"{step_i + 1:4d}/{n_steps}"
            if use_tinterp and _t0_epoch is not None:
                cur_dt = _t0_epoch + np.timedelta64(int(sim_time), "s")
                time_label = str(cur_dt)[:16]  # "2018-04-25T03:00"
            elif _ds_times is not None and 0 <= file_idx < len(_ds_times):
                time_label = str(_ds_times[file_idx])[:16]
            else:
                time_label = f"file_idx={file_idx}"
            # Beaching count only when enabled (keeps the dormant line
            # byte-identical to the no-beaching path).
            beach_label = f"  beached={counts['beached']}" if beach_enabled else ""
            nr_label = (f"  not_released={counts['not_released']}"
                        if "not_released" in counts else "")
            bt = f" beach={_ph_beach*1e3:.1f}" if beach_enabled else ""
            phase_label = (f"  (io={_ph_io*1e3:.1f} step={_ph_step*1e3:.1f}"
                           f"{bt} rec={_ph_rec*1e3:.1f})")
            print(f"  step {step_label}  {time_label}  alive={counts['alive']}/{counts['total']}{beach_label}{nr_label}  {step_dt*1000:.1f}ms{phase_label}")

        # Track the last FULLY-EXECUTED step for the final checkpoint.  An early
        # break (exhausted forcing) exits BEFORE the step runs, so the formula
        # ``start_step_i + n_steps - 1`` would over-stamp the step and a resume
        # would jump past pending steps — wrong global step for delayed-release
        # activation, the diffusion RNG, and wet_streak.
        last_global_step_i = global_step_i
        last_file_idx = file_idx if use_tinterp else ckpt_file_idx

    # Time-aware downgrade summary
    if _downgrade_count > 0 and verbose and _is_multistage:
        print(f"  [time-aware] {_downgrade_count}/{n_steps} step(s) "
              f"fell back to steady RK (dt > dt_to_next)")

    # Timing summary
    if verbose and _step_times:
        arr = np.array(_step_times) * 1000  # ms
        n = len(arr)
        q1 = arr[:n // 4].mean() if n >= 4 else arr[0]
        q4 = arr[-n // 4:].mean() if n >= 4 else arr[-1]
        print(f"\n  Timing: mean={arr.mean():.1f}ms  min={arr.min():.1f}ms  max={arr.max():.1f}ms  "
              f"first25%={q1:.1f}ms  last25%={q4:.1f}ms  total={arr.sum() / 1000:.1f}s")
        if _phase_times:
            ph = np.array(_phase_times) * 1000
            print(f"  Phases (mean ms): io={ph[:, 0].mean():.1f}  "
                  f"step={ph[:, 1].mean():.1f}  beach={ph[:, 2].mean():.1f}  "
                  f"rec={ph[:, 3].mean():.1f}")

    # Final checkpoint — stamped with the LAST FULLY-EXECUTED step (not the
    # nominal end), so a resume after an early break (exhausted forcing) picks
    # up at the right global step.  Skipped if no step ran.
    if (writer is not None and checkpoint_freq > 0
            and last_global_step_i is not None):
        writer.write_checkpoint(
            state, last_global_step_i, last_file_idx,
            wet_streak=wet_streak if beach_enabled else None,
            rotation_angles=getattr(mesh, "rotation_angles", None),
            release_step=state.release_step)

    return record


def _index_to_sim_time(
    idx: int,
    file_times: np.ndarray,
    n_time_total: int,
    on_exhausted: str,
) -> float:
    """Return the sim-time corresponding to file index ``idx``.

    For ``on_exhausted == "cycle"`` indices outside ``[0, n_time_total)``
    wrap with full-period offset, where the period equals
    ``file_times[-1] - file_times[0] + trailing_interval``.  For ``"stop"``
    out-of-range indices are clamped to the valid range.
    """
    if n_time_total <= 0:
        return 0.0
    if on_exhausted == "cycle":
        # Synthetic 1s period for single-snapshot cycling, matching
        # _resolve_time_bracket's treatment of the same degenerate case.
        if n_time_total == 1:
            period = 1.0
        else:
            period = (file_times[-1] - file_times[0]
                      + (file_times[-1] - file_times[-2]))
        n_full, rem = divmod(int(idx), n_time_total)
        return n_full * period + float(file_times[rem])
    # stop: clamp
    clipped = max(0, min(int(idx), n_time_total - 1))
    return float(file_times[clipped])


def _resolve_time_bracket(
    sim_time: float,
    file_times: np.ndarray,
    n_time_total: int,
    on_exhausted: str,
    direction: int = 1,
):
    """Resolve (idx_a, idx_b, ta, tb, wrapped_time) for sim_time.

    Returns None if sim_time is out of range in stop mode.
    Pure index math — no I/O, no caching, no field allocation.

    ``direction`` only matters when ``sim_time`` sits exactly ON a snapshot:
    a forward step moves into ``[t_k, t_{k+1}]`` (today's behaviour, the
    default), a backward step moves into ``[t_{k-1}, t_k]`` — so for
    ``direction < 0`` the bracket shifts down one interval and the position
    lands on its TOP (alpha = 1, the field at the snapshot either way).
    At the last snapshot in stop mode this un-collapses the bracket that a
    forward query degenerates to; at the first snapshot it degenerates
    instead (nothing earlier), mirroring forward behaviour at the end.
    Strictly inside a bracket the tuple is direction-independent (the
    direction-dependent ``vel_next``/``dt_to_next`` live in
    ``_build_vel_payload``).
    """
    t_start = file_times[0]
    t_end = file_times[-1]

    if on_exhausted == "cycle":
        last_dt = file_times[-1] - file_times[-2] if n_time_total > 1 else 1.0
        total_span = t_end - t_start + last_dt
        sim_time_wrapped = t_start + ((sim_time - t_start) % total_span)
    else:
        sim_time_wrapped = sim_time
        if sim_time_wrapped < t_start - 1e-6 or sim_time_wrapped > t_end + 1e-6:
            return None

    idx_b = int(np.searchsorted(file_times, sim_time_wrapped, side="right"))
    idx_a = idx_b - 1

    if on_exhausted == "cycle":
        idx_a_mod = idx_a % n_time_total
        idx_b_mod = idx_b % n_time_total
    else:
        idx_a_mod = max(0, min(idx_a, n_time_total - 1))
        idx_b_mod = max(0, min(idx_b, n_time_total - 1))

    ta = file_times[idx_a_mod]
    tb = file_times[idx_b_mod]
    if on_exhausted == "cycle" and tb <= ta:
        last_dt = file_times[-1] - file_times[-2] if n_time_total > 1 else 1.0
        tb = ta + last_dt

    # Backward on-snapshot shift (tolerance matches the range check above).
    # Covers both shapes the forward resolution produces at a snapshot:
    # ta == sim_time (mid-range / clamped top in stop mode, alpha = 0) and
    # the fully collapsed stop-mode top (idx_a == idx_b == n-1).
    if direction < 0 and sim_time_wrapped <= ta + 1e-6:
        idx_b_new = idx_a_mod
        if on_exhausted == "cycle":
            idx_a_new = (idx_a_mod - 1) % n_time_total
        else:
            idx_a_new = idx_a_mod - 1
        if idx_a_new < 0 or idx_a_new == idx_b_new:
            # Nothing before the first snapshot (stop) / single-snapshot
            # cycle: degenerate bracket → steady fallback in the payload.
            return idx_b_new, idx_b_new, ta, ta, sim_time_wrapped
        ta_new = file_times[idx_a_new]
        tb_new = file_times[idx_b_new]
        if tb_new <= ta_new:
            # Cycle wrap (standing on file_times[0]): the bracket is the
            # trailing interval, the position its top end — express the
            # wrapped time in that bracket's coordinate so alpha = 1.
            last_dt = (file_times[-1] - file_times[-2]
                       if n_time_total > 1 else 1.0)
            tb_new = ta_new + last_dt
        return idx_a_new, idx_b_new, ta_new, tb_new, tb_new

    return idx_a_mod, idx_b_mod, ta, tb, sim_time_wrapped


def _resolve_recon_stencil(bracket, n_time_total: int):
    """4-snapshot PCHIP stencil from a 2-snapshot bracket, or None at edges.

    Given a bracket ``(idx_a, idx_b, ta, tb, q)`` whose central interval
    ``[ta, tb]`` brackets the query ``q``, return
    ``((idx_a-1, idx_a, idx_b, idx_b+1), q)`` when all four indices are valid,
    strictly consecutive (no cycle wrap), and interior to the file range —
    i.e. ``file_times`` is monotone increasing across the stencil and
    ``ta <= q <= tb`` is the central interval (``times[1] <= q <= times[2]`` in
    :func:`reconstruct_pchip` terms).  Returns ``None`` (→ linear fallback) at
    the first/last interval of the range, on an exact snapshot (degenerate
    bracket), or across a cycle seam.
    """
    idx_a, idx_b, _ta, _tb, q = bracket
    if idx_b != idx_a + 1:          # degenerate snapshot or cycle-wrapped seam
        return None
    i0, i3 = idx_a - 1, idx_b + 1
    if i0 < 0 or i3 >= n_time_total:  # not enough snapshots on one side
        return None
    return (i0, idx_a, idx_b, i3), q


def _eta_at_time(t, file_times, n_time_total, on_exhausted, direction,
                 vel_cache, reader, ds, mesh, mode, diffusion, sample_vars,
                 pchip=False):
    """Surface elevation at time ``t`` as a step starting there would see it
    — the PCHIP stage at c = 0 where ``pchip`` and a 4-snapshot stencil
    exist (bit-identical to ``reconstruct_pchip``), else the linear payload
    blend (``_build_vel_payload``) — or None outside the data."""
    bracket = _resolve_time_bracket(t, file_times, n_time_total, on_exhausted,
                                    direction=direction)
    if bracket is None:
        return None
    idx_a, idx_b, ta, tb, tw = bracket
    kw = dict(diffusion=diffusion, sample_vars=sample_vars)
    interval = tb - ta
    alpha = (tw - ta) / interval if interval > 0 else 0.0
    if idx_a == idx_b or abs(alpha) < 1e-12:
        # on a snapshot: the snapshot itself (PCHIP at s = 0 and the linear
        # blend both return it exactly) — without loading the next one early
        return vel_cache.get(idx_a, reader, ds, mesh, mode, **kw).eta
    if abs(alpha - 1.0) < 1e-12:
        return vel_cache.get(idx_b, reader, ds, mesh, mode, **kw).eta
    if pchip:
        stencil = _resolve_recon_stencil(bracket, n_time_total)
        if stencil is not None:
            from .numerics.temporal_reconstruct import _pchip_blend4
            idxs4, q = stencil
            e = [vel_cache.get(i, reader, ds, mesh, mode, **kw).eta
                 for i in idxs4]
            t4 = [float(file_times[i]) for i in idxs4]
            return _pchip_blend4(e[0], e[1], e[2], e[3], *t4, q)
    ea = vel_cache.get(idx_a, reader, ds, mesh, mode, **kw).eta
    eb = vel_cache.get(idx_b, reader, ds, mesh, mode, **kw).eta
    from .core.io import _blend
    return _blend(ea, eb, alpha)


def _build_vel_payload(
    bracket,
    vel_cache: _VelCache,
    reader, ds, mesh, mode,
    diffusion=None,
    sample_vars=None,
    direction: int = 1,
    keep_extra=None,
):
    """Build time-aware velocity payload from a bracket tuple.

    Returns (vel_step, vel_next, dt_to_next, file_idx) where:
        vel_step   — velocity at sim_time (lerped from snapshots);
                     direction-free (the field at one instant)
        vel_next   — raw snapshot the step is moving TOWARD: at t=tb for
                     forward, at t=ta for backward (None if degenerate)
        dt_to_next — positive seconds from sim_time to that snapshot:
                     tb - sim_time (forward) / sim_time - ta (backward)
        file_idx   — idx_a (used for checkpointing / verbose)

    When on exact snapshot (idx_a == idx_b): vel_next=None, dt_to_next=None.
    Caller can pass these to integrator kwargs directly — RK stages blend
    vel → vel_next at c·dt/dt_to_next, which samples t + c·dt forward and
    t - c·dt backward (resolve the bracket with the SAME ``direction`` so
    an on-snapshot backward query brackets downward instead of degrading).
    """
    idx_a, idx_b, ta, tb, sim_time_wrapped = bracket
    # ``keep_extra`` (the PCHIP 4-snapshot stencil indices) is retained in the
    # cache so the recon block does not re-read the flanking snapshots each
    # step — keeping a sliding window of one new read per step.
    extra = set(keep_extra) if keep_extra else set()

    vel_a = vel_cache.get(idx_a, reader, ds, mesh, mode, diffusion=diffusion,
                          sample_vars=sample_vars)
    if idx_a == idx_b:
        vel_cache.evict_except({idx_a} | extra)
        return vel_a, None, None, idx_a

    vel_b = vel_cache.get(idx_b, reader, ds, mesh, mode, diffusion=diffusion,
                          sample_vars=sample_vars)
    vel_cache.evict_except({idx_a, idx_b} | extra)

    interval = tb - ta
    alpha = (sim_time_wrapped - ta) / interval if interval > 0 else 0.0

    if abs(alpha) < 1e-12:
        vel_step = vel_a
    elif abs(alpha - 1.0) < 1e-12:
        vel_step = vel_b
    else:
        # Lazy view: nothing is blended until a consumer actually reads
        # a field.  The fused 2D tinterp dispatch reads the RAW pair and
        # never materializes u/v; beaching/sampling/unfused paths blend
        # only the fields they touch (same _blend -> bit-identical).
        vel_step = LazyLerpField(vel_a, vel_b, alpha)

    if direction >= 0:
        dt_to_next = tb - sim_time_wrapped
        return vel_step, vel_b, dt_to_next, idx_a

    # Backward: the next snapshot along the motion is the one at ta.
    dt_to_prev = sim_time_wrapped - ta
    if dt_to_prev <= 1e-6:
        # Defensive: an unshifted bracket whose position sits on (or within
        # the 1e-6 s snap band above) its lower snapshot has no usable room
        # behind it — degrade to steady rather than hand the integrator a
        # non-positive or sub-microsecond dt_to_next.  Unreachable from the
        # run loop (it resolves brackets with the same ``direction``, which
        # shifts these down); guards direct payload callers.
        return vel_step, None, None, idx_a
    return vel_step, vel_a, dt_to_prev, idx_a


def _interp_velocity_at_time(
    sim_time: float,
    file_times: np.ndarray,
    n_time_total: int,
    on_exhausted: str,
    direction: int,
    vel_cache: _VelCache,
    reader, ds, mesh, mode,
):
    """Legacy 2-tuple API: returns (vel_step, file_idx) or (None, -1).

    New code should use _resolve_time_bracket + _build_vel_payload to get
    vel_next and dt_to_next as well.  ``direction`` is forwarded so an
    on-snapshot backward query brackets downward like the main loop
    (review follow-up: the accepted-but-ignored parameter was an API trap).
    """
    bracket = _resolve_time_bracket(sim_time, file_times, n_time_total,
                                    on_exhausted, direction=direction)
    if bracket is None:
        return None, -1
    vel_step, _vel_next, _dt_to_next, file_idx = _build_vel_payload(
        bracket, vel_cache, reader, ds, mesh, mode, direction=direction,
    )
    return vel_step, file_idx


# ---------------------------------------------------------------------------
# Full tracking with multiple realizations
# ---------------------------------------------------------------------------

def _draws_random_numbers(numerics) -> bool:
    """True when a run keys random draws by the particle index: diffusion
    (horizontal or vertical) or stochastic stranding / resuspension."""
    diff = getattr(numerics, "diffusion", None)
    if diff is not None and getattr(diff, "enabled", False):
        return True
    beach = getattr(numerics, "beaching", None)
    if beach is not None and getattr(beach, "enabled", False):
        if "stochastic" in (getattr(beach, "criteria", None) or ()):
            return True
        if getattr(beach, "resuspension", "never") in ("stochastic",
                                                      "auto_or_stochastic"):
            return True
    return False


def _resolve_particle_offset(config) -> int:
    """The run's ``particle_offset``: ``data.particle_offset`` or the
    ``# kelp particle_offset N`` header that ``utils/split_seeds.py`` writes
    into each shard file (they must agree when both are given), else 0.

    A pulsed release (``time.n_pulses > 1``) replicates the seed SET, so a
    shard of it cannot be keyed consistently with the unsharded run (and the
    sharding concat refuses such stores) — refused here.
    """
    from .core.release import read_particle_offset
    header = read_particle_offset(config.data.inicloud)
    cfg_off = config.data.particle_offset
    if cfg_off is not None and header is not None and int(cfg_off) != header:
        raise ValueError(
            f"data.particle_offset = {cfg_off} disagrees with the seed file's "
            f"'# kelp particle_offset {header}' header "
            f"({config.data.inicloud}); drop one of them")
    offset = int(cfg_off if cfg_off is not None else (header or 0))
    if offset > 0 and config.time.n_pulses > 1:
        raise ValueError(
            "a pulsed release (time.n_pulses > 1) cannot be sharded: the "
            "release replicates the seed set, so a shard's particles are not "
            "a block of the full release (data.particle_offset / the shard "
            "header). Run the pulsed release unsharded.")
    return offset


def run_tracking(
    config: AppConfig,
    verbose: bool = False,
    output_path: Optional[str] = None,
    checkpoint_freq: Optional[int] = None,
    temporal_interp: Optional[bool] = None,
) -> List[TrajectoryRecord]:
    """Run the full tracking simulation from an AppConfig.

    Parameters
    ----------
    config : AppConfig
        Complete configuration.
    verbose : bool
        Print progress.
    output_path : str or None
        If given, stream trajectories to this NetCDF file.
        Overrides ``config.data.output_path``.
        A suffix ``_rN`` is added for multi-realization runs.
    checkpoint_freq : int or None
        If > 0 and output is enabled, write restart checkpoints
        every this many steps.  Overrides ``config.numerics.checkpoint_freq``.
    temporal_interp : bool or None
        If True, linearly interpolate velocity fields between adjacent
        file time steps.  Overrides ``config.numerics.temporal_interp``.

    Returns
    -------
    list of TrajectoryRecord
        One record per realization.
    """
    # Resolve parameters: kwargs override config values
    if output_path is None and config.data.output_path:
        output_path = config.data.output_path
    if checkpoint_freq is None:
        checkpoint_freq = config.numerics.checkpoint_freq
    if temporal_interp is None:
        temporal_interp = config.numerics.temporal_interp

    # PCHIP temporal reconstruction is inert without temporal interpolation
    # (it selects HOW snapshots are blended in time); warn rather than error.
    if config.numerics.temporal_reconstruction == "pchip" and not temporal_interp:
        warnings.warn(
            "numerics.temporal_reconstruction='pchip' has no effect when "
            "temporal_interp is off (discrete mode reads one snapshot per "
            "step); set temporal_interp: true to use it.",
            RuntimeWarning, stacklevel=2,
        )

    reader = get_reader(config.mesh.type)

    # Load mesh
    if verbose:
        print(f"Loading mesh ({config.mesh.type}) from {config.mesh.meshpath}")
    # Readers whose mesh is derived FROM the data (healpix) need the mode
    # config at mesh build (var_u/var_v overrides + sigma_layer pick for
    # the wet mask); the classic readers keep the plain signature.
    import inspect as _inspect
    if "mode_cfg" in _inspect.signature(reader.load_mesh).parameters:
        mesh = reader.load_mesh(config.mesh, mode_cfg=config.mode)
    else:
        mesh = reader.load_mesh(config.mesh)

    # Exact nearest-node anchor guard: ALWAYS set from the config (False clears
    # a stale flag on a reused mesh object) — BEFORE the cell rule, whose mode
    # depends on it.
    from .core.mesh import (set_exact_nearest, set_grid_hash_cell,
                            apply_grid_hash_cell_rule, describe_grid_hash)
    set_exact_nearest(mesh, bool(getattr(config.mesh, "exact_nearest", False)))
    if verbose and config.mesh.exact_nearest:
        print("    nearest node: exact anchor guard ON")
    # The grid-hash cell (reader-agnostic, one site): an explicit
    # grid_hash_cell_size is pinned in mesh.metadata so ANY later rebuild
    # keeps it; otherwise mesh.grid_hash_cell_rule ("occupancy": measure and
    # choose, "legacy": keep the reader's auto cell) — the resolved cell and
    # the rule go into the output metadata.
    if getattr(config.mesh, "grid_hash_cell_size", None) is not None:
        set_grid_hash_cell(mesh, config.mesh.grid_hash_cell_size)
        cell_rule_used = "explicit"
        if verbose:
            print(f"    grid hash: explicit cell "
                  f"{config.mesh.grid_hash_cell_size:g} deg")
    else:
        cell_rule_used = apply_grid_hash_cell_rule(
            mesh, getattr(config.mesh, "grid_hash_cell_rule", "occupancy"))["rule"]
        if verbose:
            print("    " + describe_grid_hash(mesh))

    # Slice vertical interfaces to match max_levels if configured.
    # For 3D staggered fields, max_levels is interpreted as the number of
    # interfaces to keep from the surface downward.
    max_levels = getattr(config.mode, "max_levels", None)
    if max_levels is not None and mesh.sigma.size > 0:
        if config.mode.Fdrift3D and max_levels < 2:
            raise ValueError(
                f"max_levels must be >= 2 in 3D mode, got {max_levels}"
            )
        keep = min(int(max_levels), mesh.sigma.size)
        mesh.sigma = mesh.sigma[:keep]
        if verbose:
            if getattr(mesh, "vertical_coord", "zlevel") == "sigma":
                print(f"Vertical interfaces limited to {keep} (sigma mesh)")
            else:
                print(f"Depth levels limited to {keep} "
                      f"(max depth {mesh.sigma[-1]:.1f}m)")

    # Open velocity dataset
    if verbose:
        print(f"Opening velocity dataset: {config.data.ncinput}")
    v_path = config.data.ncinput_v or None
    w_path = config.data.ncinput_w or None
    # eta_path is passed only when set: only readers that support a separate
    # ssh store (healpix) accept the kwarg — a set key with another reader
    # fails loudly with a TypeError naming it.
    _extra = {}
    if getattr(config.data, "ncinput_eta", ""):
        _extra["eta_path"] = config.data.ncinput_eta
    # var_az family for the multi-year exact layout (Av.fesom.<year>.nc):
    # pass diffusion only on the one reader/combination that consumes it,
    # so other readers' open_dataset signatures stay untouched.
    _d0 = config.numerics.diffusion
    if (config.mesh.type == "fesom2" and _d0.enabled
            and getattr(_d0, "az_enabled", False)
            and getattr(_d0, "var_az", None)):
        _extra["diffusion"] = _d0
    ds = reader.open_dataset(config.data.ncinput, v_path=v_path, w_path=w_path,
                             mode_cfg=config.mode, **_extra)

    # Load initial particle seeds (delayed-release aware: an optional 4th
    # column is the release offset in seconds).  The release STATE is built
    # below, once dt is resolved (release_step = round(release_sec / dt)), and
    # rotated into the mesh frame there — see "Build the initial particle
    # state".
    from .core.release import read_seeds_with_release
    _seed_lon, _seed_lat, _seed_z, _seed_release_sec = read_seeds_with_release(
        config.data.inicloud)
    particle_offset = _resolve_particle_offset(config)
    if verbose:
        print(f"Loaded {_seed_lon.shape[0]} seed location(s) "
              f"from {config.data.inicloud}")

    n_time_total = ds.sizes["time"]

    # -- Time info: auto-detect dt and file_dt, validate --
    file_times, file_dt_detected = extract_time_info(ds)

    # -- time_averaging: warn if not set, apply midpoint shift --
    time_avg = config.data.time_averaging
    if time_avg is None:
        mesh_type = config.mesh.type.lower()
        if "fesom2" in mesh_type:
            suggestion = "mean"
            hint = "FESOM2 typically writes time-averaged (monthly/daily mean) output"
        elif "healpix" in mesh_type:
            suggestion = "mean"
            hint = ("healpix products are typically daily means stamped at "
                    "the END of the averaging window (ng5 convention)")
        else:
            suggestion = "instant"
            hint = "FESOM-C typically writes instantaneous snapshots"
        warnings.warn(
            f"data.time_averaging is not set. {hint} — "
            f"assuming '{suggestion}'. Please set data.time_averaging "
            f"explicitly to 'mean' or 'instant' in your config.",
            UserWarning,
            stacklevel=2,
        )
        time_avg = suggestion
    elif time_avg not in ("mean", "instant"):
        raise ValueError(
            f"data.time_averaging must be 'mean' or 'instant', got '{time_avg}'"
        )

    if time_avg == "mean" and file_times is not None and len(file_times) > 1:
        raw_times = file_times.copy()
        file_times = _shift_to_midpoints(file_times)
        shift_s = float(raw_times[1] - file_times[1])
        if verbose:
            print(f"  Time averaging:    mean → shifted to window midpoints "
                  f"(~{shift_s / 86400:.1f} d back)")
    elif verbose and time_avg == "instant":
        print(f"  Time averaging:    instant (no shift)")

    # file_dt: config overrides auto-detected value
    if config.numerics.file_dt is not None:
        file_dt = config.numerics.file_dt
        if verbose:
            print(f"Using configured file_dt = {file_dt:.1f} s")
        # Rebuild file_times from file_dt if auto-detect failed
        if file_times is None:
            file_times = np.arange(n_time_total, dtype=np.float64) * file_dt
    else:
        file_dt = file_dt_detected
    if temporal_interp and file_times is None:
        # fail before any output exists (run_single_realization backstops)
        raise ValueError(
            "temporal_interp: true needs a physical time axis, but the "
            "dataset's 'time' coordinate could not be decoded (missing, an "
            "index 0..N-1 without units, or unparseable); set "
            "numerics.file_dt to declare the snapshot interval, or set "
            "temporal_interp: false")

    # dt: auto-detect from file_dt if not set
    if config.numerics.dt is None:
        auto_dt = file_dt
        if auto_dt is None and file_times is not None and len(file_times) > 1:
            auto_dt = float(np.median(np.diff(file_times)))
        if auto_dt is not None:
            config.numerics.dt = auto_dt
            if verbose:
                print(f"Auto-detected dt = {auto_dt:.1f} s from file time coordinate")
        else:
            raise ValueError(
                "Cannot auto-detect dt from file time coordinate. "
                "Please set numerics.dt (and optionally numerics.file_dt) in the config."
            )

    # -- Summary --
    dt = config.numerics.dt

    # Discrete mode reads one record per step while the particles move by
    # dt: refuse a dt unlike the record spacing before any output exists
    # (run_single_realization repeats the check for direct callers).
    _check_discrete_dt(dt, temporal_interp, file_dt, file_times,
                       n_records=n_time_total)

    def _fmt_dt(s):
        if s >= 86400 and s % 86400 == 0:
            return f"{s:.0f} s ({s / 86400:.0f} d)"
        if s >= 3600:
            return f"{s:.0f} s ({s / 3600:.1f} h)"
        if s >= 60:
            return f"{s:.0f} s ({s / 60:.0f} min)"
        return f"{s:.0f} s"

    # Compute display file_dt: use detected value, or median from file_times
    display_file_dt = file_dt
    irregular_time = False
    if display_file_dt is None and file_times is not None and len(file_times) > 1:
        diffs = np.diff(file_times)
        display_file_dt = float(np.median(diffs))
        irregular_time = not np.allclose(diffs, diffs[0], rtol=1e-3)

    if display_file_dt:
        label = f"  File time step:    {_fmt_dt(display_file_dt)}"
        if irregular_time:
            label += " (irregular, median)"
        print(label)
    if file_times is not None:
        total_span = file_times[-1] - file_times[0]
        print(f"  File time span:    {_fmt_dt(total_span)}, {n_time_total} steps")
    if temporal_interp:
        print(f"  Tracking dt:       {_fmt_dt(dt)} (temporal interpolation between file steps)")
    else:
        if irregular_time and display_file_dt and abs(dt - display_file_dt) > 0.1:
            print(f"  Tracking dt:       {_fmt_dt(dt)}  NOTE: the file step is irregular; "
                  f"each step reads the next record and moves the particles by dt")
        else:
            print(f"  Tracking dt:       {_fmt_dt(dt)}")

    # -- Resolve start_date if provided --
    sim_time_start = None
    if config.time.start_date is not None:
        start_idx, sim_time_start = _resolve_start_date(
            ds, config.time.start_date, file_times, temporal_interp,
        )
        # duration_days → n_steps
        if config.time.duration_days is not None:
            n_steps = int(config.time.duration_days * 86400.0 / dt)
        else:
            # n_steps / end_time_idx must still be set
            _, n_steps = resolve_time_params(config.time)

        # Print start_date info
        is_exact = (sim_time_start == file_times[start_idx])
        if is_exact:
            msg = f"  start_date:        {config.time.start_date} (exact match, file_idx={start_idx})"
        else:
            # Show bracketing snapshots
            t_before = ds["time"].values[start_idx]
            t_after = ds["time"].values[start_idx + 1] if start_idx + 1 < len(file_times) else "end"
            msg = (f"  start_date:        {config.time.start_date} "
                   f"(between [{start_idx}] {t_before} and [{start_idx+1}] {t_after})")
        print(msg)
        if config.time.duration_days is not None:
            print(f"  duration_days:     {config.time.duration_days} → {n_steps} steps")
    else:
        start_idx, n_steps = resolve_time_params(config.time)

    on_exhausted = config.time.on_exhausted

    # Legacy backward compat: if using end_time_idx with backward tracking,
    # start from end_time_idx (old behavior started the loop there)
    if (config.time.start_date is None
            and config.time.n_steps is None and config.time.end_time_idx is not None
            and config.mode.time_dir < 0):
        start_idx = config.time.end_time_idx

    direction_str = "forward" if config.mode.time_dir >= 0 else "backward"
    print(f"  Tracking steps:    {n_steps} ({direction_str}), start_idx={start_idx}")
    print(f"  Integrator:        {config.numerics.integrator}")
    if temporal_interp:
        print(f"  Temporal interp:   on")

    # -- Build the initial particle state (delayed/staggered release, TODO 2.5).
    # n_pulses=1 + no release column => trivial schedule => plain seed load
    # (release_step=None), byte-identical to the no-feature path.  Rotation into
    # the mesh frame happens here, on the (possibly replicated) state — one of
    # exactly two user/geographic entry points (run_single_realization treats
    # its init_state as already in-frame).
    from .core.release import build_release_state
    init_state = build_release_state(
        _seed_lon, _seed_lat, _seed_z, _seed_release_sec,
        n_pulses=config.time.n_pulses, pulse_interval=config.time.pulse_interval,
        dt=dt, start_step_i=0)
    init_state = _maybe_rotate_seeds(init_state, mesh)
    # Opt-in failure_reason tracking (mode.track_failure_reason): allocate
    # the int8 array on the seed state — every integrator path stamps it,
    # the writer lazily creates the output variable, checkpoints carry it.
    if getattr(config.mode, "track_failure_reason", False):
        init_state = init_state.replace(
            failure_reason=np.zeros(init_state.n_particles, dtype=np.int8))
    if init_state.release_step is not None:
        n_total = init_state.n_particles
        n_frozen = int(np.sum(init_state.status == ParticleStatus.NOT_RELEASED))
        print(f"  Delayed release:   {n_total} particles, "
              f"{config.time.n_pulses} pulse(s), {n_frozen} frozen at start")
        if int(init_state.release_step.max()) >= n_steps:
            n_never = int(np.sum(init_state.release_step >= n_steps))
            warnings.warn(
                f"delayed release: {n_never} particle(s) have a release step "
                f">= n_steps ({n_steps}); they never activate in this run "
                f"(all-NaN output).", RuntimeWarning, stacklevel=2)

    # Review finding #5: surface the diffusion configuration in
    # the run summary so users can sanity-check it without parsing the
    # output NetCDF metadata.
    if config.numerics.diffusion.enabled:
        d = config.numerics.diffusion
        composition_resolved = _resolve_composition(
            d, config.numerics.integrator,
        )
        line = (f"  Diffusion:         scheme={d.scheme}, "
                f"composition={composition_resolved}, seed={d.seed}")
        if d.scheme == "uniform":
            line += f", Kh=({d.kh_zonal}, {d.kh_meridional}) m^2/s"
        else:
            dres_str = (f"{d.dres_m:.0f} m" if d.dres_m is not None
                        else "auto (mesh-derived)")
            line += f", dres_m={dres_str}"
            if composition_resolved == "split_rk":
                line += f", kh_eval_position={d.kh_eval_position}"
        if (getattr(d, "az_enabled", False)
                and (bool(getattr(d, "var_az", None))
                     or getattr(d, "az_const", 0.0) > 0.0)):
            az_src = (f"var {d.var_az!r}" if getattr(d, "var_az", None)
                      else f"const {d.az_const} m^2/s")
            line += f", Az={az_src}"
        print(line)

    # -- Scalar sampling setup (TODO 2.4) --
    # Fail fast BEFORE the realization loop: probe-read the requested
    # variables at the resolved start index so a missing or element-based
    # variable surfaces here, not deep inside the first step.  Also build
    # the {name: {units, long_name}} attrs map from the dataset variable
    # attributes (only keys that are present), wired into the writer.
    # ``user_sample_vars`` is what the user asked to WRITE; ``read_vars`` is
    # what we actually READ each step (adds the beaching dry_mask_var when
    # set, which must not leak to output — see below).
    user_sample_vars = list(config.mode.sample_vars or [])
    sample_interp = config.numerics.sample_interp
    sample_var_attrs = {}
    if user_sample_vars:
        # Probe at the start index (clamped into range for cycle/legacy).
        probe_idx = max(0, min(int(start_idx), n_time_total - 1))
        reader.read_scalar_fields(ds, probe_idx, mesh, user_sample_vars,
                                  config.mode)
        for name in user_sample_vars:
            attrs = {}
            if name in ds.variables:
                src = ds[name].attrs
                for key in ("units", "long_name"):
                    if key in src:
                        attrs[key] = str(src[key])
            if attrs:
                sample_var_attrs[name] = attrs
        print(f"  Sampling:          {', '.join(user_sample_vars)} "
              f"(method={sample_interp})")

    # -- uv_interp x uv_layout fail-fast (pre-writer; spec 2026-06-06) --
    # The probe is metadata-only (dims/sizes).  Readers without it keep
    # the step-0 in-loop guard as backstop.
    probe_fn = getattr(reader, "probe_uv_layout", None)
    if probe_fn is not None:
        _validate_uv_compat(probe_fn(ds, config.mode, mesh),
                            config.numerics.uv_interp)
    if config.numerics.uv_interp != "leastsq":
        print(f"  uv_interp:         {config.numerics.uv_interp}")
    if (getattr(mesh, "metadata", None)
            and mesh.metadata.get("healpix_geometry") == "pixel"
            and config.numerics.diffusion.enabled
            and config.numerics.diffusion.scheme in ("em", "m1")):
        # Pre-writer twin of the reader's fail-fast: without it the
        # NotImplementedError fires on the first velocity read, AFTER
        # the output file was created (2026-08 review round 3).
        raise ValueError(
            "healpix pixel geometry v1 does not support spatial Kh "
            "fields (em/m1 diffusion) — use the dual geometry or "
            "uniform Kh.")
    if config.numerics.uv_interp == "edge_leastsq":
        # Pre-warm the lazy edge-adjacency table so step 0 doesn't pay
        # the one-time build inside the tracking loop.
        from .core.mesh import build_elem_neighbors
        build_elem_neighbors(mesh)
        print(f"  edge_leastsq_bc:   {config.numerics.edge_leastsq_bc}")

    if config.numerics.elem_search != "local":
        print(f"  elem_search:       {config.numerics.elem_search}")
        # Pre-warm the edge-adjacency table the walk crosses on (lazy,
        # idempotent — a no-op when edge_leastsq already built it).
        from .core.mesh import build_elem_neighbors
        build_elem_neighbors(mesh)
        # Honest scope note: the walk lives in the unfused paths; the
        # fused leastsq kernels (2D euler/rk2/rk4, 3D euler) keep their
        # grid-hash search and neither consume nor update the carried
        # elements.
        if config.numerics.uv_interp == "leastsq":
            warnings.warn(
                "numerics.elem_search='walk' accelerates only the UNFUSED "
                "integrator paths; with uv_interp='leastsq' the fused "
                "Numba kernels keep the grid-hash search and the carried "
                "elements are neither used nor updated there (they still "
                "serve the unfused 3D RK stages and domain checks; on a "
                "fully-fused run the checkpointed restart_elem stays at "
                "its seed-time value).  Use uv_interp: barycentric or "
                "edge_leastsq to benefit fully.",
                RuntimeWarning, stacklevel=2)

    # -- combined-diffusion x uv_interp fail-fast (pre-writer) --
    # em/m1 + Euler auto-resolves to composition='combined', whose drift
    # kernel is element-centre leastsq math; without this guard the
    # conflict only surfaced inside step 0, AFTER the writer had created
    # the output file (review 2026-07-02; the run_single_realization /
    # euler_step check remains as the backstop for direct callers).
    _diff_pre = config.numerics.diffusion
    if (_diff_pre.enabled and _diff_pre.scheme in ("em", "m1")
            and config.numerics.uv_interp != "leastsq"
            and _resolve_composition(
                _diff_pre, config.numerics.integrator) == "combined"):
        raise ValueError(
            "diffusion scheme 'em'/'m1' with integrator 'euler' resolves "
            "to composition='combined', whose drift kernel is element-"
            "centre leastsq math — incompatible with numerics.uv_interp="
            f"{config.numerics.uv_interp!r}.  Use uv_interp: leastsq, or "
            "integrator rk2/rk4 with diffusion.composition: split_rk.")

    # -- periodic-window fail-fasts (pre-writer) --
    # The windows act on MESH-FRAME coordinates; a rotated mesh would
    # fold rotated longitudes while the user reasons geographically.
    # The raising sanity checks (window/mesh non-overlap, dateline
    # straddle) also run here so a typo'd config cannot leave an empty
    # output file behind; run_single_realization repeats both (with the
    # ghost-side warnings) as the direct-caller backstop.
    if (getattr(config.numerics, "periodic_lon", None) is not None
            or getattr(config.numerics, "periodic_lat", None) is not None):
        if getattr(mesh, "rotation_angles", None) is not None:
            raise ValueError(
                "numerics.periodic_lon/periodic_lat apply to MESH-FRAME "
                "coordinates and cannot be combined with mesh.rotation "
                "(the fold would act on rotated longitudes)."
            )
        _validate_periodic_windows(
            mesh, getattr(config.numerics, "periodic_lon", None),
            getattr(config.numerics, "periodic_lat", None), warn=False)

    # -- rk45 fail-fasts (pre-writer, TODO 2.3) --
    # The adaptive integrator is v1-narrow: 2-D + leastsq + linear time
    # interpolation only.  Reject unsupported combinations BEFORE the
    # writer creates the output file (node-layout + leastsq is already
    # rejected by the uv-compat probe above).
    if config.numerics.integrator == "rk45":
        if config.mode.Fdrift3D:
            raise NotImplementedError(
                "integrator 'rk45' is 2-D only (v1); use rk2/rk4 for 3-D "
                "runs")
        if config.numerics.uv_interp != "leastsq":
            raise NotImplementedError(
                "integrator 'rk45' supports uv_interp='leastsq' only "
                f"(v1), got {config.numerics.uv_interp!r} — node-layout "
                "runs need rk2/rk4 + barycentric")
        if getattr(config.numerics, "temporal_reconstruction",
                   "linear") == "pchip":
            raise ValueError(
                "integrator 'rk45' does not support "
                "temporal_reconstruction='pchip' (v1) — per-particle "
                "substep times cannot use the 4-snapshot bundle yet")
        if getattr(config.numerics, "elem_search", "local") == "walk":
            raise ValueError(
                "elem_search='walk' is not supported with integrator "
                "'rk45' (v1): carried elements would be neither used nor "
                "updated")

    # -- Az vertical diffusion fail-fasts (pre-writer, TODO 2.2b) --
    # Gated on an ACTUAL source (var_az set or az_const > 0): az_enabled
    # with a zero source is documented byte-identical/dormant, so it must
    # neither raise here nor add metadata below (review 2026-08-15).
    # Guard order matters for error clarity: an unsupported var_az reader
    # is reported before the 2-D complaint (a fesomc config typically has
    # both problems at once; the reader restriction is the deeper one).
    _az_active = (
        _diff_pre.enabled and getattr(_diff_pre, "az_enabled", False)
        and (bool(getattr(_diff_pre, "var_az", None))
             or getattr(_diff_pre, "az_const", 0.0) > 0.0))
    if _az_active:
        if getattr(_diff_pre, "var_az", None) and config.mesh.type != "fesom2":
            raise ValueError(
                "diffusion.var_az is supported on the fesom2 reader only "
                f"(v1); mesh.type={config.mesh.type!r} runs must use the "
                "constant az_const source")
        if not config.mode.Fdrift3D:
            raise ValueError(
                "diffusion.az_enabled requires 3-D tracking "
                "(mode.Fdrift3D: true): the vertical Az kick is "
                "meaningless in 2-D")
        if _diff_pre.var_az:
            if _diff_pre.var_az not in ds:
                raise ValueError(
                    f"diffusion.var_az={_diff_pre.var_az!r} not found in "
                    f"the dataset (variables: {sorted(ds.data_vars)})")
            # Metadata-only layout probe (no data load), so a wrong-layout
            # Az fails BEFORE the writer creates the output file — the
            # same promise the sample_vars probe makes.
            from .readers.fesom2 import _az_var_is_mid_layer
            if _az_var_is_mid_layer(ds, _diff_pre.var_az):
                raise ValueError(
                    f"diffusion.var_az={_diff_pre.var_az!r} lives on the "
                    "mid-layers — Az must be interface-based (dim nz, "
                    "same layout as w)")

    # -- Beaching setup (TODO 2.2) --
    # Fail fast at startup (spec): use_eta with no ssh/eta variable would
    # silently substitute zeros (FESOM2 reader) and hide all drying; a
    # missing dry_mask_var surfaces here, not deep in the first step.  The
    # dry-mask field is read internally but excluded from output unless the
    # user also lists it in mode.sample_vars.
    read_vars = list(user_sample_vars)
    beach_cfg = config.numerics.beaching
    if beach_cfg is not None and beach_cfg.enabled:
        eta_names = ["ssh", "eta", "zos"]
        cfg_eta = getattr(config.mode, "var_eta", None)
        if cfg_eta:
            eta_names.append(cfg_eta)
        if beach_cfg.use_eta and not any(k in ds for k in eta_names):
            raise ValueError(
                "beaching.use_eta=true requires an ssh/eta/zos (or the "
                "configured mode.var_eta) variable in the dataset (the "
                "FESOM2 reader otherwise substitutes zeros, hiding all "
                "eta-driven drying).")
        if beach_cfg.dry_mask_var:
            probe_idx = max(0, min(int(start_idx), n_time_total - 1))
            reader.read_scalar_fields(
                ds, probe_idx, mesh, [beach_cfg.dry_mask_var], config.mode)
            if beach_cfg.dry_mask_var not in read_vars:
                read_vars.append(beach_cfg.dry_mask_var)
        crit_csv = ",".join(beach_cfg.criteria)
        print(f"  Beaching:          criteria=[{crit_csv}], "
              f"depth_min={beach_cfg.depth_min}, use_eta={beach_cfg.use_eta}, "
              f"resuspension={beach_cfg.resuspension}")

    records = []

    for ire in range(config.time.num_runs):
        if verbose:
            print(f"Realization {ire + 1}/{config.time.num_runs}")

        # Open writer if output_path given
        writer = None
        if output_path is not None:
            from .writers import TrajectoryWriter
            if config.time.num_runs > 1:
                from pathlib import Path
                p = Path(output_path)
                out_file = str(p.with_stem(f"{p.stem}_r{ire}"))
            else:
                out_file = output_path
            meta = {
                "integrator": config.numerics.integrator,
                "dt": config.numerics.dt,
                "mesh_type": config.mesh.type,
                "time_dir": config.mode.time_dir,
                "start_time_idx": start_idx,
                "n_steps": n_steps,
                # dimensionality stamp: the sharding concat tool needs it
                # to pick the writer-parity backfill for vel_w on an
                # all-dormant shard (2-D: w=None -> writer zeros; 3-D:
                # dormant w is NaN-masked)
                "mode_3d": int(bool(getattr(config.mode, "Fdrift3D",
                                            False))),
                # trajectory-defining settings, ALWAYS stamped so that the
                # sharding concat refuses shards that differ in them
                # (review 2026-09-22, F07) and an output names its engine
                "temporal_interp": int(bool(temporal_interp)),
                "temporal_reconstruction": str(
                    config.numerics.temporal_reconstruction),
                "time_averaging": str(time_avg),
                "kelp_version": _kelp_version(),
            }
            if getattr(config.mode, "Fdrift3D", False):
                meta["w_interp"] = str(config.numerics.w_interp)
                # the file-side W convention and the 2026-09-23 reader
                # contract (readers convert producer W to the internal
                # positive-down depth rate); an output without it was
                # written by an engine that used FESOM/HEALPix W unconverted
                meta["w_positive"] = str(config.mode.w_positive)
                meta["w_contract"] = "reader-converted-2026-09-23"
            if config.time.n_pulses > 1:
                # pulsed release replicates the SEED SET — stamped so the
                # sharding concat tool can refuse such stores (per-shard
                # replication reorders particles; 2026-08 review).
                meta["n_pulses"] = int(config.time.n_pulses)
            if config.numerics.diffusion.enabled:
                d = config.numerics.diffusion
                # Record the *resolved* composition so metadata reflects
                # what actually ran (review finding #6).  E.g. em+Euler
                # with the user leaving composition at its "split"
                # default still runs as "combined".
                composition_meta = _resolve_composition(
                    d, config.numerics.integrator,
                )
                meta.update({
                    "diffusion_enabled": 1,
                    "diffusion_scheme": d.scheme,
                    "diffusion_kh_zonal": d.kh_zonal,
                    "diffusion_kh_meridional": d.kh_meridional,
                    "diffusion_seed": d.seed,
                    "diffusion_realization_id": ire,
                    "diffusion_composition": composition_meta,
                })
                # Az attrs ONLY when the vertical kick has an actual
                # source (dormant / zero-source files carry no new
                # attrs — byte-identity guarantee).
                if (getattr(d, "az_enabled", False)
                        and (bool(getattr(d, "var_az", None))
                             or getattr(d, "az_const", 0.0) > 0.0)):
                    meta["diffusion_az_enabled"] = 1
                    meta["diffusion_az_const"] = d.az_const
                    if getattr(d, "var_az", None):
                        meta["diffusion_var_az"] = d.var_az
            # Beaching metadata ONLY when enabled (dormant files carry no
            # new attrs — the byte-identity test compares global attrs too).
            if beach_cfg is not None and beach_cfg.enabled:
                b = beach_cfg
                meta.update({
                    "beaching_enabled": 1,
                    "beaching_criteria": ",".join(b.criteria),
                    "beaching_depth_min": b.depth_min,
                    "beaching_use_eta": int(b.use_eta),
                    "beaching_resuspension": b.resuspension,
                    "beaching_seed": (b.seed if b.seed is not None
                                      else config.numerics.diffusion.seed),
                    "beaching_realization_id": ire,
                })
            if config.numerics.integrator == "rk45":
                meta["rk45_tol_m"] = float(
                    getattr(config.numerics, "rk45_tol_m", 1.0))
                if getattr(config.numerics, "rk45_h_min", None) is not None:
                    meta["rk45_h_min"] = float(config.numerics.rk45_h_min)
            # Periodic windows ONLY when set (dormant files carry no new
            # attrs — byte-identity).
            for _pn in ("periodic_lon", "periodic_lat"):
                _pw = getattr(config.numerics, _pn, None)
                if _pw is not None:
                    meta[_pn] = [float(_pw[0]), float(_pw[1])]
            if user_sample_vars:
                meta["sample_interp"] = sample_interp
            if config.numerics.uv_interp != "leastsq":
                meta["uv_interp"] = config.numerics.uv_interp
            # always stamped (0/1) since the default flipped: an absent attribute
            # means a file written before the guard existed
            meta["exact_nearest"] = 1 if getattr(config.mesh, "exact_nearest", False) else 0
            # the resolved grid-hash cell (degrees) and how it was chosen
            # (occupancy / legacy / explicit): pin grid_hash_cell_size to it
            # to repeat this run exactly
            meta["grid_hash_cell"] = float(mesh._gh_cell_size)
            meta["grid_hash_cell_rule"] = cell_rule_used
            if config.numerics.uv_interp == "edge_leastsq":
                meta["edge_leastsq_bc"] = config.numerics.edge_leastsq_bc
            if config.numerics.elem_search != "local":
                meta["elem_search"] = config.numerics.elem_search
            if getattr(config.numerics, "velocity_float32", False):
                meta["velocity_float32"] = "true"
            if getattr(config.numerics, "prefetch_size", 1) > 1:
                meta["prefetch_size"] = int(config.numerics.prefetch_size)
            # Rotated-frame provenance: output positions are geographic
            # regardless — this attribute documents the run's frame, only
            # present when rotation is on (geographic runs stay byte-
            # identical, the byte-identity test compares global attrs too).
            angles = getattr(mesh, "rotation_angles", None)
            if angles is not None:
                meta["rotation_angles"] = list(angles)
            writer = TrajectoryWriter(
                out_file, init_state.n_particles, metadata=meta,
                sample_var_attrs=sample_var_attrs,
                flush_every=getattr(config.numerics, "output_flush_every", 0),
            )
            writer.open()

        shared_cache = None
        shared_dir = getattr(config.numerics, "shared_fields_dir", None)
        if shared_dir:
            try:
                from .core.shared_fields import SharedFieldCache
                shared_cache = SharedFieldCache(
                    shared_dir, _shared_fields_tag(config))
            except Exception as exc:
                warnings.warn(
                    f"shared_fields disabled: {exc!r} — local reads.",
                    RuntimeWarning)
        try:
            rec = run_single_realization(
                mesh=mesh,
                ds=ds,
                reader=reader,
                init_state=init_state,
                mode=config.mode,
                numerics=config.numerics,
                start_time_idx=start_idx,
                n_steps=n_steps,
                n_time_total=n_time_total,
                on_exhausted=on_exhausted,
                verbose=verbose,
                shared_cache=shared_cache,
                writer=writer,
                checkpoint_freq=checkpoint_freq,
                temporal_interp=temporal_interp,
                file_times=file_times,
                file_dt=file_dt,
                sim_time_start=sim_time_start,
                realization_id=ire,
                sample_vars=read_vars,
                sample_interp=sample_interp,
                output_sample_vars=user_sample_vars,
                particle_offset=particle_offset,
            )
        finally:
            if writer is not None:
                writer.close()
            if shared_cache is not None:
                shared_cache.close()

        records.append(rec)

        # Shift start for next realization (n_steps stays fixed).  When
        # start_date is used with temporal_interp, also shift sim_time_start
        # so later realizations don't replay the first one.  For cycling
        # runs the shift must account for wrap (start_idx can exceed
        # n_time_total); for clamped runs we clip to the valid range.
        prev_idx = start_idx
        start_idx += config.time.run_interval
        if (
            sim_time_start is not None
            and file_times is not None
            and n_time_total > 0
        ):
            sim_time_start += (
                _index_to_sim_time(start_idx, file_times, n_time_total,
                                   on_exhausted)
                - _index_to_sim_time(prev_idx, file_times, n_time_total,
                                     on_exhausted)
            )

    ds.close()
    return records


# ---------------------------------------------------------------------------
# Functional interface (no AppConfig needed)
# ---------------------------------------------------------------------------

def track_particles(
    mesh: Mesh,
    ds,
    reader,
    lon0: np.ndarray,
    lat0: np.ndarray,
    z0: Optional[np.ndarray] = None,
    start_time_idx: int = 0,
    n_steps: Optional[int] = None,
    on_exhausted: str = "stop",
    dt: Optional[float] = 900.0,
    time_dir: float = 1.0,
    integrator: str = "euler",
    output_freq: int = 1,
    mode: Optional[ModeConfig] = None,
    *,
    end_time_idx: Optional[int] = None,
    writer=None,
    checkpoint_freq: int = 0,
    temporal_interp: bool = False,
    file_times: Optional[np.ndarray] = None,
    file_dt: Optional[float] = None,
    w_interp: str = "leastsq",
    uv_interp: str = "leastsq",
    edge_leastsq_bc: str = "freeslip",
    elem_search: str = "local",
    Dmin: float = 0.5,
    z0b_min: float = 0.1,
    rk45_tol_m: float = 1.0,
    rk45_h_min: Optional[float] = None,
    periodic_lon: Optional[Tuple[float, float]] = None,
    periodic_lat: Optional[Tuple[float, float]] = None,
    diffusion: Optional[DiffusionConfig] = None,
    realization_id: int = 0,
    start_step_i: int = 0,
    exact_nearest: bool = True,
    grid_hash_cell_rule: str = "occupancy",
    sample_vars: Optional[List[str]] = None,
    sample_interp: str = "leastsq",
    particle_offset: int = 0,
) -> TrajectoryRecord:
    """Simple functional interface for tracking.

    ``exact_nearest`` (default ``True``, the same default as the config)
    sets the mesh's exact nearest-node anchor guard
    (``core.mesh.set_exact_nearest``) before the run — a mesh from
    ``load_mesh`` carries no flag by itself; pass ``False`` together with
    ``grid_hash_cell_rule="legacy"`` for the pre-guard anchors bit for bit.
    ``grid_hash_cell_rule`` (``"occupancy"``, the config default, or
    ``"legacy"``) is applied right after it through
    ``core.mesh.apply_grid_hash_cell_rule``: an explicitly pinned mesh
    (``set_grid_hash_cell``) is left alone, the same rule and mode on a
    reused mesh reuse the earlier decision, a different one decides afresh.

    Parameters
    ----------
    mesh : Mesh
    ds : xr.Dataset
    reader : reader class
    lon0, lat0 : np.ndarray
        Initial particle positions.
    z0 : np.ndarray or None
    start_time_idx : int
        First file time index.
    n_steps : int or None
        Number of tracking steps.  If None and ``end_time_idx`` is given,
        computes ``n_steps = end_time_idx - start_time_idx``.
        If both are None, defaults to 10.
    on_exhausted : str
        ``"stop"`` or ``"cycle"``.
    dt : float
        Time step (seconds).
    time_dir : float
        Forward (+1) or backward (-1).
    integrator : str
        "euler", "rk2", "rk4", or "rk45" (adaptive; tune with
        ``rk45_tol_m`` / ``rk45_h_min``).
    output_freq : int
        Output every output_freq steps.
    mode : ModeConfig or None
        Defaults to 2D forward.
    end_time_idx : int or None
        Legacy parameter.
    writer : TrajectoryWriter or None
        Optional streaming writer.
    checkpoint_freq : int
        Checkpoint interval (steps).
    temporal_interp : bool
        Enable temporal velocity interpolation.
    file_times : np.ndarray or None
        File time coordinate in seconds (for temporal_interp).
    file_dt : float or None
        File time step in seconds (for temporal_interp).
    w_interp : str
        Horizontal interpolation method for the vertical velocity W
        (``"nearest" | "leastsq" | "barycentric" | "idw"``); 3D only.
    uv_interp : str
        Horizontal u/v interpolation method: ``"leastsq"`` (element-centre
        least-squares, the Fortran-parity default), ``"barycentric"``
        (vertex blend) or ``"edge_leastsq"`` (containing element + its
        edge-sharing neighbors, FESOM-style).  Node-based velocity fields
        require ``"barycentric"``.
    edge_leastsq_bc : str
        Boundary ghost value for ``"edge_leastsq"``: ``"freeslip"``
        (default) or ``"noslip"``.  Ignored by the other methods.
    elem_search : str
        Containing-element search strategy: ``"local"`` (per-step
        nearest-node scan, default) or ``"walk"`` (search once at release,
        then cross edges incrementally; identical verdicts, unfused paths
        only).
    Dmin : float
        Minimum water depth placeholder for vertical-coordinate backends.
    z0b_min : float
        Bottom roughness length placeholder for vertical-coordinate backends.
    periodic_lon, periodic_lat : (lo, hi) or None
        Periodic fold windows for idealized channel domains (see
        ``numerics.periodic_lon`` in docs/configuration.md — the
        ghost-band mesh contract applies).
    sample_vars : list of str or None
        Node-based dataset variable names to sample at record points
        (forwarded to ``run_single_realization``).  ``None``/empty →
        dormant (no sampling).
    sample_interp : str
        Horizontal interpolation method for scalar sampling.

    Notes
    -----
    Beaching rides on ``NumericsConfig.beaching`` and is resolved inside
    ``run_single_realization``.  The two startup guards still apply here:
    ``use_eta=true`` without an ssh/eta/zos variable, and ``dry_mask_var``
    set without the variable in ``sample_vars``, both raise ``ValueError``
    (safe-by-raise — not silently ignored).  ``dry_mask_var`` in particular
    needs the ``run_tracking`` wiring to build the internal read list, so
    drive a dry-mask run through ``run_tracking`` rather than this
    functional shortcut.

    Returns
    -------
    TrajectoryRecord
    """
    # Resolve n_steps
    _legacy_end = False
    if n_steps is None:
        if end_time_idx is not None:
            # end - start, as config.resolve_time_params (was + 1 here)
            n_steps = end_time_idx - start_time_idx
            _legacy_end = True
        else:
            n_steps = 10

    if hasattr(mesh, "metadata"):          # a stub mesh in unit tests carries none
        from .core.mesh import set_exact_nearest, apply_grid_hash_cell_rule
        set_exact_nearest(mesh, bool(exact_nearest))
        if getattr(mesh, "_gh_offsets", None) is not None:
            apply_grid_hash_cell_rule(mesh, grid_hash_cell_rule)

    init_state = make_state(lon0, lat0, z0)
    # USER/geographic seeds enter the rotated frame here (no-op on a
    # geographic mesh) — the second of the two user/geographic entry
    # points (see _maybe_rotate_seeds).
    init_state = _maybe_rotate_seeds(init_state, mesh)
    if mode is None:
        mode = ModeConfig(Fdrift2D=True, Fdrift3D=False, time_dir=time_dir)

    # Legacy: backward tracking started from end_time_idx
    actual_start = start_time_idx
    if _legacy_end and mode.time_dir < 0:
        actual_start = end_time_idx

    numerics_kwargs = dict(
        dt=dt, integrator=integrator, output_freq=output_freq,
        w_interp=w_interp, uv_interp=uv_interp,
        edge_leastsq_bc=edge_leastsq_bc, elem_search=elem_search,
        Dmin=Dmin, z0b_min=z0b_min,
        rk45_tol_m=rk45_tol_m, rk45_h_min=rk45_h_min,
        periodic_lon=periodic_lon, periodic_lat=periodic_lat,
    )
    if diffusion is not None:
        numerics_kwargs["diffusion"] = diffusion
    numerics = NumericsConfig(**numerics_kwargs)
    return run_single_realization(
        mesh, ds, reader, init_state, mode, numerics,
        start_time_idx=actual_start,
        n_steps=n_steps,
        on_exhausted=on_exhausted,
        writer=writer,
        checkpoint_freq=checkpoint_freq,
        temporal_interp=temporal_interp,
        file_times=file_times,
        file_dt=file_dt,
        realization_id=realization_id,
        start_step_i=start_step_i,
        sample_vars=sample_vars,
        sample_interp=sample_interp,
        particle_offset=particle_offset,
    )


__all__ = [
    "TrajectoryRecord",
    "run_single_realization",
    "run_tracking",
    "track_particles",
]
