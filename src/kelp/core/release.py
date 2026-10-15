"""Delayed / staggered particle release (TODO 2.5).

Release tracers from the same location(s) at different times within a single
run / output file.  All particles are pre-allocated and the not-yet-released
ones are FROZEN (``ParticleStatus.NOT_RELEASED`` → excluded from ``alive_mask``
→ not advected) until their release step, instead of growing the pool mid-run
(which would break the fixed-``trajectory``-dimension streaming writer).

Release time is an OFFSET in seconds from the run start, converted to a step
index ``release_step = round(release_sec / dt)``; activation is
``global_step_i >= release_step``.  Because the run advances ``step_i`` as
0,1,2,… regardless of direction, the mechanism is direction-agnostic (a
backward run decrements ``sim_time`` but increments ``step_i``).
"""
from __future__ import annotations

import re
import warnings
from typing import Optional, Tuple

import numpy as np

from .particles import ParticleState, ParticleStatus, make_state


def build_release_state(lon, lat, z=None, release_sec=None, *,
                        n_pulses: int = 1, pulse_interval: float = 0.0,
                        dt: float, start_step_i: int = 0) -> ParticleState:
    """Build a :class:`ParticleState` with per-particle delayed-release steps.

    Seeds ``(lon, lat, z, release_sec)`` — ``release_sec`` an offset in seconds
    from the run start (default 0) — are replicated ``n_pulses`` times; pulse
    ``p`` (0-based) adds ``p * pulse_interval`` seconds to each seed's base
    release.  ``release_step = round(total_sec / dt)``.  Particles whose
    ``release_step <= start_step_i`` start ``ALIVE``; the rest start
    ``NOT_RELEASED``.  Total particles ``= n_pulses * len(lon)``.

    When the schedule is trivial (``n_pulses == 1`` and all ``release_sec ==
    0``) a plain :func:`make_state` is returned (``release_step=None``, all
    ``ALIVE``) so the no-feature path is byte-identical.

    Note
    ----
    A delayed particle, *after* it activates, follows the same trajectory as a
    normally-seeded particle ONLY in a deterministic flow.  With stochastic
    diffusion the counter-RNG keys on the GLOBAL step index, so a particle
    released at step ``K`` draws its kicks at steps ``K, K+1, …`` (not ``0,
    1, …``) — its stochastic path differs from a step-0 particle's, by design
    (and is itself reproducible / restart-safe).
    """
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    S = lon.shape[0]
    z = np.zeros(S, dtype=np.float64) if z is None else np.asarray(z, dtype=np.float64)
    release_sec = (np.zeros(S, dtype=np.float64) if release_sec is None
                   else np.asarray(release_sec, dtype=np.float64))
    if release_sec.shape[0] != S:
        raise ValueError(
            f"release_sec length {release_sec.shape[0]} != number of seeds {S}")
    if not np.all(np.isfinite(release_sec)):
        # A non-finite release would otherwise slip the trivial all-ALIVE path
        # (NaN > 0 is False) or cast to a bogus int32 step (inf -> INT32_MIN ->
        # immediate activation).
        raise ValueError("release_sec must be finite")
    if np.any(release_sec < 0.0):
        raise ValueError("release_sec must be >= 0 (offset from run start)")

    n_pulses = int(n_pulses)
    if n_pulses < 1:
        raise ValueError(f"n_pulses must be >= 1, got {n_pulses}")
    pulse_interval = float(pulse_interval)
    if not np.isfinite(pulse_interval):
        raise ValueError("pulse_interval must be finite")
    if n_pulses > 1 and not (pulse_interval > 0.0):
        raise ValueError(
            "pulse_interval must be > 0 when n_pulses > 1 (else every pulse "
            "would release at the same time)")
    if dt is None or not (float(dt) > 0.0):
        raise ValueError("build_release_state needs a positive dt")
    dt = float(dt)

    # Trivial schedule -> identical to a normal seed load (byte-identity).
    if n_pulses == 1 and not np.any(release_sec > 0.0):
        return make_state(lon, lat, z)

    lon_a = np.tile(lon, n_pulses)
    lat_a = np.tile(lat, n_pulses)
    z_a = np.tile(z, n_pulses)
    pulse_offset = (np.repeat(np.arange(n_pulses, dtype=np.float64), S)
                    * pulse_interval)
    total_sec = np.tile(release_sec, n_pulses) + pulse_offset

    steps_float = total_sec / dt
    # Guard the int32 cast: a release offset beyond ~2.1e9 steps (absurd, but a
    # huge release_sec / tiny dt could reach it) would silently wrap negative.
    if np.any(np.abs(steps_float) >= np.iinfo(np.int32).max):
        raise ValueError(
            "release time / dt exceeds the int32 step range; reduce release_sec "
            "or increase dt")
    release_step = np.rint(steps_float).astype(np.int32)
    if np.any(np.abs(steps_float - release_step) > 1e-6):
        warnings.warn(
            "delayed release: some release times are not a whole multiple of "
            "dt; rounded to the nearest step.", RuntimeWarning, stacklevel=2)

    st = make_state(lon_a, lat_a, z_a)
    status = st.status.copy()
    status[release_step > int(start_step_i)] = ParticleStatus.NOT_RELEASED
    return st.replace(status=status, release_step=release_step)


def activate_released(state: ParticleState, global_step_i: int) -> ParticleState:
    """Flip ``NOT_RELEASED → ALIVE`` where ``global_step_i >= release_step``.

    Returns the SAME object when nothing activates (or ``release_step`` is
    ``None``), so the steady path is allocation-free / byte-identical.
    ``nearest_node`` is left as-is — the next integrator step fills it for the
    newly-alive particles via its alive-only nearest-node search.
    """
    rs = state.release_step
    if rs is None:
        return state
    to_activate = ((state.status == ParticleStatus.NOT_RELEASED)
                   & (int(global_step_i) >= rs))
    if not np.any(to_activate):
        return state
    status = state.status.copy()
    status[to_activate] = ParticleStatus.ALIVE
    return state.replace(status=status)


def mask_dormant_for_record(
    state: ParticleState, sampled: Optional[dict] = None,
) -> Tuple[ParticleState, Optional[dict]]:
    """NaN-mask dormant (``NOT_RELEASED``) rows for OUTPUT only.

    Returns ``(rec_state, sampled)`` — a COPY of ``state`` with ``lon/lat/z``
    (and ``u/v/w`` when present) set to NaN where ``status == NOT_RELEASED``,
    plus a copy of the ``sampled`` dict with the same rows NaN'd.  Returns the
    SAME objects when there are no dormant particles (byte-identical default
    path).  The original state's seed coords are left intact (needed at
    activation) — the NaN is output-only.  ``status`` is left untouched so the
    output still records ``NOT_RELEASED`` for frozen particles.
    """
    dormant = state.status == ParticleStatus.NOT_RELEASED
    if not np.any(dormant):
        return state, sampled

    def _nan(a):
        if a is None:
            return None
        b = a.astype(np.float64, copy=True)
        b[dormant] = np.nan
        return b

    rec = state.replace(
        lon=_nan(state.lon), lat=_nan(state.lat), z=_nan(state.z),
        u=_nan(state.u), v=_nan(state.v), w=_nan(state.w),
    )
    out_sampled = sampled
    if sampled:
        out_sampled = {}
        for k, v in sampled.items():
            vv = np.asarray(v, dtype=np.float64).copy()
            vv[dormant] = np.nan
            out_sampled[k] = vv
    return rec, out_sampled


def mask_inactive_velocity(state: ParticleState,
                           prev_status) -> ParticleState:
    """NaN-mask ``u/v/w`` for OUTPUT where the particle was not ALIVE through
    the whole step.

    ``prev_status`` is the status after the release activation and before
    the step; ``state.status`` is the status after the step and the
    beaching hook.  A velocity is defined only where both are ``ALIVE``: the
    record in which a particle dies or strands, every later record of a
    dead, stranded or dormant particle, and a particle resuspended during
    this step (the kernels skipped it and wrote 0) all get NaN (output
    format 2).  Returns the SAME object when nothing is masked; the input
    arrays are never modified.
    """
    if state.u is None and state.w is None:
        return state
    bad = ((state.status != ParticleStatus.ALIVE)
           | (np.asarray(prev_status) != ParticleStatus.ALIVE))
    if not np.any(bad):
        return state

    def _nan(a):
        if a is None:
            return None
        b = np.array(a, dtype=np.float64, copy=True)
        b[bad] = np.nan
        return b

    return state.replace(u=_nan(state.u), v=_nan(state.v), w=_nan(state.w))


#: header that ``utils/split_seeds.py`` writes into every shard seed file
PARTICLE_OFFSET_TAG = "kelp particle_offset"


#: largest offset the uint64 random keys accept without wrapping (signed range)
PARTICLE_OFFSET_MAX = 2 ** 63 - 1


def check_particle_offset(value) -> int:
    """``value`` as an int in ``[0, 2**63)`` (an integer type or an integer
    string), else ``ValueError`` — the one rule for ``data.particle_offset``,
    the seed header and the ``particle_offset=`` keyword of the entry points."""
    import numbers
    bad = ValueError(f"particle_offset must be an integer in [0, 2**63), got {value!r}")
    if isinstance(value, (bool, np.bool_)):
        raise bad
    if isinstance(value, str):
        try:
            value = int(value.strip())
        except ValueError:
            raise bad from None
    if not isinstance(value, numbers.Integral):
        raise bad
    value = int(value)
    if not 0 <= value <= PARTICLE_OFFSET_MAX:
        raise bad
    return value


def _header_tokens(line: str):
    """The tokens of a header line candidate, or None.

    A comment that mentions ``particle_offset`` or ``particle-offset`` (any
    case) is a header candidate and must read exactly
    ``# kelp particle_offset N`` with N plain ASCII digits (anything after a
    further ``#`` is a comment); a near miss is refused rather than read as 0.
    ``kelp particle offset ...`` (spaces) is a candidate too; other prose such
    as "particle offset from the coast" is not.
    """
    body = line.strip().lstrip("#").split("#", 1)[0]
    if not (re.search(r"particle[_-]offset", body, re.IGNORECASE)
            or re.match(r"\s*kelp\s+particle\s+offset", body, re.IGNORECASE)):
        return None
    return body.split()


def read_particle_offset(path) -> Optional[int]:
    """The ``# kelp particle_offset N`` header of a seed file, or ``None``.

    ``utils/split_seeds.py`` writes it as the first line of every shard: the
    global index of the shard's first particle in the full release, so the
    shard's random draws are keyed like the unsharded run's.  Only the comment
    block before the first data row is searched.  Any comment there that
    mentions ``particle_offset`` must read exactly ``# kelp particle_offset N``
    (N plain digits, below 2**63; ``kelp`` in any case): a near miss such as
    ``particle_offset: 5`` or prose about it is refused, never read as 0. Two
    headers are refused.
    """
    found = None
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if not s.startswith("#"):
                break                       # the header precedes the data
            body = _header_tokens(s)
            if body is None:
                continue
            if (len(body) != 3 or body[0].lower() != "kelp"
                    or body[1] != "particle_offset"
                    or not (body[2].isascii() and body[2].isdigit())):
                raise ValueError(
                    f"seed file {path!r}: malformed header {s!r} (expected "
                    f"'# {PARTICLE_OFFSET_TAG} N')")
            try:
                off = check_particle_offset(body[2])
            except ValueError:
                raise ValueError(
                    f"seed file {path!r}: malformed header {s!r}") from None
            if found is not None:
                raise ValueError(
                    f"seed file {path!r}: two '{PARTICLE_OFFSET_TAG}' headers")
            found = off
    return found


def read_seeds_with_release(path: str):
    """Read a seed file ``lon lat [depth] [release_sec]`` (``#`` comments).

    Returns ``(lon, lat, z, release_sec)`` as float64 arrays; ``z`` is zeros and
    ``release_sec`` is zeros when their columns are absent.  This is the
    delayed-release-aware companion to :meth:`ParticleState.load` (which reads
    only ``lon lat [depth]``).
    """
    data = np.loadtxt(path, comments="#", dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(
            f"seed file {path!r} needs at least lon, lat columns")
    lon = data[:, 0]
    lat = data[:, 1]
    n = lon.shape[0]
    z = data[:, 2] if data.shape[1] >= 3 else np.zeros(n, dtype=np.float64)
    release_sec = (data[:, 3] if data.shape[1] >= 4
                   else np.zeros(n, dtype=np.float64))
    return lon, lat, z, release_sec


__all__ = [
    "build_release_state",
    "activate_released",
    "mask_dormant_for_record",
    "read_seeds_with_release",
    "read_particle_offset",
    "check_particle_offset",
    "PARTICLE_OFFSET_TAG",
]
