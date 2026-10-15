#!/usr/bin/env python3
"""Concatenate sharded kelp trajectory outputs along the particle axis.

The inverse of ``utils/split_seeds.py``: shard Zarr outputs, given in
SHARD ORDER, are merged into one store whose particle order equals the
original (unsharded) seed order.  For a deterministic ``n_pulses=1`` run
the merged store is **data-identical** to the unsharded run's output —
every variable's VALUES match bit for bit (pinned by
tests/test_sharding.py).  Container geometry (chunk layout, creation
order) may legitimately differ from a store the streaming writer
produced; the merged store opens in xarray with the same dimension names.

Safety rules (2026-08 review):
- Zarr only, both sides (``.nc`` raises with a clear message);
- the OUTPUT must not be one of the inputs (``mode="w"`` would destroy
  the shard before it is read) and an existing non-empty output needs
  ``force=True`` / ``--force``;
- shard paths carrying trailing integers must arrive in ascending 0..S-1
  order (a shell glob puts ``shard10`` before ``shard2`` — silently
  wrong particle order otherwise); duplicates always raise; pass
  ``check_order=False`` / ``--unordered`` only when your naming carries
  no order;
- output format 2 (``kelp_output_format = 2``, record dimension ``time``,
  ``time`` with CF ``units``/``calendar``) and format 1 (dimension ``obs``)
  both merge; shards must agree on the format, the record dimension name
  and the ``time`` units/calendar (a mixture is refused); merged float
  arrays keep the shards' fill value (NaN in format 2) and the merged
  ``trajectory`` IDs are the first shard's ``particle_offset`` +
  ``0..n-1`` (global IDs);
- the ``time`` axes and the VARIABLE SETS of all shards must be
  identical, with ONE exception: ``vel_*`` may be absent from a shard
  whose particles stayed ``NOT_RELEASED`` for the whole run (the
  integrators early-exit with no alive particles) — those slices are
  NaN-backfilled, matching what the unsharded run records for dormant
  particles; any other set difference means the shard configs differed;
- run-defining attrs (integrator, dt, uv_interp, time_dir, mesh_type,
  start_time_idx, n_steps, diffusion_*/beaching_*) must agree across
  shards — except ``diffusion_seed``/``beaching_seed`` under
  ``independent_shards``, where shards drawing random numbers must instead
  NOT share seeds over overlapping particle indices;
- stores stamped ``n_pulses > 1`` are refused (per-shard seed-set
  replication reorders particles relative to the unsharded run) unless
  ``independent_shards`` (then merged shard-major, with a warning);
- shards that draw random numbers must carry ``particle_offset`` values that
  follow the merge order (``utils/split_seeds.py`` headers), unless
  ``independent_shards``;
- every shard variable must have exactly ``(count, n_obs)`` /
  ``(count,)`` shape (a truncated store — e.g. a killed job — raises
  instead of being broadcast);
- per-shard ``restart_*`` checkpoint arrays are NOT merged (resume each
  shard separately, then merge); ``restart_*`` group attrs are dropped;
- multi-realization runs write ``<name>_rN.zarr`` per realization —
  merge each realization separately;
- writes stream in obs blocks (no whole-shard-variable materialization),
  destination chunked with the writer's formula (particle axis blocked
  at 500k; obs chunk from the ~4 MB target).

Usage:
    python3 concat_trajectories.py out.zarr shard0.zarr shard1.zarr ... \
        [--force] [--unordered] [--independent-shards]
"""
from __future__ import annotations

import argparse
import re
import warnings
import sys
from pathlib import Path

import numpy as np
import zarr

#: writer parity (writers.py TrajectoryWriter constants)
TRAJ_CHUNK_MAX = 500_000
OBS_CHUNK_MAX = 256
TARGET_CHUNK_BYTES = 4_000_000

#: provenance/bookkeeping attrs excluded from the cross-shard equality
#: check — EVERYTHING else must match (blocklist, not allowlist: the
#: writer keeps growing run-defining attrs; review 2026-08 round 3)
_ATTR_IGNORE = ("history", "Conventions", "sharded_from", "particle_offset")


def _run_attrs(a):
    return {k: v for k, v in a.items()
            if k not in _ATTR_IGNORE and not k.startswith("restart_")}


def _stochastic(attrs):
    """True unless the shard says it draws no random numbers.  Every engine
    that stamps ``particle_offset`` also stamps ``random_draws`` (0/1, from
    ``run_single_realization``, so chain drivers carry it too); a store
    without it falls back to the ``run_tracking`` metadata, and a store with
    neither counts as stochastic (unknown is refused, not waved through)."""
    if "random_draws" in attrs:
        return bool(int(attrs["random_draws"]))
    if int(attrs.get("diffusion_enabled", 0) or 0):
        return True
    if int(attrs.get("beaching_enabled", 0) or 0):
        crit = str(attrs.get("beaching_criteria", ""))
        resusp = str(attrs.get("beaching_resuspension", ""))
        return "stochastic" in crit.split(",") or "stochastic" in resusp
    return "diffusion_enabled" not in attrs and "beaching_enabled" not in attrs


def _says_random(attrs):
    """True only when the metadata POSITIVELY records a random feature (no
    benefit of the doubt, unlike ``_stochastic``)."""
    if "random_draws" in attrs:
        return bool(int(attrs["random_draws"]))
    if int(attrs.get("diffusion_enabled", 0) or 0):
        return True
    return bool(int(attrs.get("beaching_enabled", 0) or 0)) and (
        "stochastic" in str(attrs.get("beaching_criteria", "")).split(",")
        or "stochastic" in str(attrs.get("beaching_resuspension", "")))


def _used_seeds(attrs):
    """``[diffusion_seed, beaching_seed]`` with None for a feature that draws
    no random numbers in this shard."""
    diff = (attrs.get("diffusion_seed")
            if int(attrs.get("diffusion_enabled", 0) or 0) else None)
    beach_random = (int(attrs.get("beaching_enabled", 0) or 0) and (
        "stochastic" in str(attrs.get("beaching_criteria", "")).split(",")
        or "stochastic" in str(attrs.get("beaching_resuspension", ""))))
    beach = attrs.get("beaching_seed") if beach_random else None
    return [diff, beach]


def _seeds_known(attrs):
    """True when the store records the seed of every random feature it uses
    (``run_tracking`` metadata); a driver store stamps ``random_draws`` only."""
    if int(attrs.get("diffusion_enabled", 0) or 0) and "diffusion_seed" not in attrs:
        return False
    beach_random = (int(attrs.get("beaching_enabled", 0) or 0) and (
        "stochastic" in str(attrs.get("beaching_criteria", "")).split(",")
        or "stochastic" in str(attrs.get("beaching_resuspension", ""))))
    if beach_random and "beaching_seed" not in attrs:
        return False
    return ("diffusion_enabled" in attrs) or ("beaching_enabled" in attrs)


def _require_zarr_dir(p):
    p = Path(p)
    if p.suffix == ".nc" or (p.exists() and p.is_file()):
        raise ValueError(
            f"{p}: only Zarr stores are supported (NetCDF concatenation "
            "is not implemented — write shard outputs to .zarr)")
    return p


def _check_order(paths):
    """Trailing-integer order check: catches glob misordering
    (``shard10`` before ``shard2``), reversals and duplicates."""
    resolved = [p.resolve() for p in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("duplicate shard paths in the input list")
    nums = []
    for p in paths:
        m = re.search(r"(\d+)(?=\D*$)", p.name)
        if m is None:
            return                      # no numeric naming — nothing to check
        nums.append(int(m.group(1)))
    if nums != sorted(nums):
        raise ValueError(
            f"shard paths are not in ascending numeric order: {nums} — a "
            "shell glob orders 'shard10' before 'shard2'; pass shards "
            "explicitly in order (zero-padded names avoid this) or use "
            "--unordered/check_order=False if your naming is not ordinal")


#: run attributes allowed to differ between shards merged with
#: ``independent_shards=True`` (each such shard draws its own noise)
_SEED_ATTRS = ("diffusion_seed", "beaching_seed")


def concat_trajectories(shard_paths, out_path, *, force=False,
                        check_order=True, independent_shards=False):
    """Merge shard Zarr stores (in shard order) into ``out_path``.

    ``independent_shards=True`` merges shards that are NOT contiguous blocks of
    one release keyed like it — e.g. the seed-block shards of a pulsed run
    split through its restart, each with its own seed: the offset-order and
    pulsed-release refusals become warnings, the seed attributes may differ,
    and the merged store is stamped ``independent_shards = 1`` (its particle
    order and noise are the shards', not an unsharded run's).
    """
    if len(shard_paths) < 1:
        raise ValueError("need at least one shard store")
    shard_paths = [_require_zarr_dir(p) for p in shard_paths]
    out_path = Path(out_path)
    if out_path.suffix == ".nc":
        raise ValueError(
            f"{out_path}: only Zarr output is supported (use .zarr)")
    if out_path.resolve() in {p.resolve() for p in shard_paths}:
        raise ValueError(
            f"{out_path}: output aliases an input shard — writing would "
            "destroy the shard before it is read")
    if out_path.exists() and any(out_path.iterdir()) and not force:
        raise ValueError(
            f"{out_path}: exists and is not empty — pass force=True / "
            "--force to overwrite")
    if check_order:
        _check_order(shard_paths)
    groups = [zarr.open_group(str(p), mode="r") for p in shard_paths]

    time0 = np.asarray(groups[0]["time"])
    for p, g in zip(shard_paths[1:], groups[1:]):
        t = np.asarray(g["time"])
        if t.shape != time0.shape or not np.array_equal(t, time0):
            raise ValueError(
                f"time axis of shard {p} differs from the first shard — "
                "shards must come from identical configurations")

    def _record_dim(g):
        d = getattr(g["time"].metadata, "dimension_names", None)
        return tuple(d) if d else ("obs",)       # pre-2026-08 stores: format 1

    def _time_attrs(g):
        return {k: g["time"].attrs.get(k) for k in ("units", "calendar")}

    dim0, tattr0 = _record_dim(groups[0]), _time_attrs(groups[0])
    for p, g in zip(shard_paths[1:], groups[1:]):
        if _record_dim(g) != dim0:
            raise ValueError(
                f"shard {p}: record dimension {_record_dim(g)} differs from "
                f"the first shard's {dim0} — the shards were written in "
                "different output formats")
        if _time_attrs(g) != tattr0:
            raise ValueError(
                f"shard {p}: time units/calendar {_time_attrs(g)} differ from "
                f"the first shard's {tattr0} — the shards count time from "
                "different epochs")
    rec_dim = dim0[0]

    attrs0 = dict(groups[0].attrs)
    for p, g in zip(shard_paths, groups):
        a = dict(g.attrs)
        if int(a.get("n_pulses", 1)) > 1 and independent_shards:
            warnings.warn(
                f"shard {p}: n_pulses={a['n_pulses']} merged as an independent "
                "shard (shard-major particle order)", RuntimeWarning,
                stacklevel=2)
        elif int(a.get("n_pulses", 1)) > 1:
            raise ValueError(
                f"shard {p} was written with n_pulses="
                f"{a['n_pulses']} — pulsed-release runs must not be "
                "sharded (per-shard seed-set replication reorders "
                "particles)")
        ra, r0 = _run_attrs(a), _run_attrs(attrs0)
        if independent_shards:
            for k in _SEED_ATTRS:
                ra.pop(k, None)
                r0.pop(k, None)
        for k in set(ra) | set(r0):
            if ra.get(k) != r0.get(k):
                raise ValueError(
                    f"shard {p}: attr {k!r} = {ra.get(k)!r} differs from "
                    f"the first shard's {r0.get(k)!r} — shards must "
                    "come from identical configurations")

    per_names = [{n for n in g.array_keys()
                  if not n.startswith("restart_")} for g in groups]
    union = set().union(*per_names)
    # vel_* may legitimately be missing ONLY from a shard whose particles
    # were dormant (NOT_RELEASED=4) for the ENTIRE run: the integrators
    # early-exit with no alive particles, so u/v/w never appear and the
    # writer never creates the diagnostics (review 2026-08 round 3).
    # The unsharded twin records NaN for those rows -> NaN backfill is
    # data-identical.  Format 2 also: a shard none of whose particles is
    # ever ALIVE (=1) — e.g. resumed when all were dead — written before
    # the run declared its velocities up front (2026-10-14); format 2
    # records NaN for every particle that is not alive.  Any other set
    # difference means the shard configs differed and is an error.
    backfill_nan = [set() for _ in groups]
    for i, (p, g, names) in enumerate(zip(shard_paths, groups,
                                          per_names)):
        missing = union - names
        vel_missing = {m for m in missing if m.startswith("vel_")}
        hard = missing - vel_missing
        if vel_missing:
            st = np.asarray(g["status"])
            fmt2 = int(g.attrs.get("kelp_output_format", 1)) >= 2
            if (st == 4).all() or (fmt2 and not (st == 1).any()):
                backfill_nan[i] = vel_missing
            else:
                hard |= vel_missing
        if hard:
            raise ValueError(
                f"variable set of shard {p} differs from the first "
                f"shard: {sorted(hard)} — the writer creates every "
                "variable a config will ever have, so a set difference "
                "means the shard configs differed")
    names0 = union

    counts = [int(g["lon"].shape[0]) for g in groups]
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(int)
    # Each shard keyed its random draws by its place in the full release
    # (particle_offset, from utils/split_seeds.py): the merge order must be
    # that place, or particles would be paired with another particle's noise
    # (and the merged store would not equal the unsharded run).
    stamped = [g.attrs.get("particle_offset") for g in groups]
    if (len(groups) > 1 and all(o is None for o in stamped)
            and not independent_shards
            and any(_says_random(dict(g.attrs)) for g in groups)):
        warnings.warn(
            "the shards carry no particle_offset (written before 2026-10-04): "
            "their random draws were keyed per shard, so the merge is "
            "statistically valid but does not equal an unsharded run",
            RuntimeWarning, stacklevel=2)
    if any(o is not None for o in stamped):
        if any(o is None for o in stamped):
            raise ValueError(
                "some shards carry a particle_offset and some do not — they "
                "were not written by the same engine version")
        base = int(stamped[0])
        expect = [base + int(o) for o in offsets]
        got = [int(o) for o in stamped]
        if got != expect:
            if len(set(got)) == 1:
                why = (f"every shard was keyed from {got[0]}: the seed files "
                       "carried no '# kelp particle_offset' header (split by "
                       "hand or before 2026-10-04), or a driver did not pass "
                       "particle_offset=")
            else:
                why = ("the shards are not in the order utils/split_seeds.py "
                       "wrote them")
            msg = (f"shard particle_offset values {got} do not follow the "
                   f"merge order (expected {expect}) — {why}")
            if independent_shards:
                warnings.warn(msg + "; merged as independent shards on request",
                              RuntimeWarning, stacklevel=2)
            elif any(_stochastic(dict(g.attrs)) for g in groups):
                raise ValueError(
                    msg + "; the shards draw random numbers (or do not say "
                    "whether they do), so the merge would not equal the "
                    "unsharded run. Re-split with utils/split_seeds.py or set "
                    "data.particle_offset; shards that are independent by "
                    "design (e.g. a pulsed run split through its restart, one "
                    "seed per shard) merge with independent_shards=True / "
                    "--independent-shards")
            else:
                warnings.warn(msg + "; harmless here (the shards draw no random "
                              "numbers)", RuntimeWarning, stacklevel=2)
    shard_seeds = [[g.attrs.get(k) for k in _SEED_ATTRS] for g in groups]
    # the seeds that actually key random draws in each shard (a beaching seed
    # of a deterministic beaching setup keys nothing)
    used_seeds = [_used_seeds(dict(g.attrs)) for g in groups]
    if independent_shards and any(_stochastic(dict(g.attrs)) for g in groups):
        # independent shards must draw DIFFERENT noise: two shards with the
        # same seeds whose particle indices overlap would repeat each other's
        # kicks particle by particle
        base = [int(o) if o is not None else 0 for o in stamped]
        span = [(b, b + c) for b, c in zip(base, counts)]
        known = [_seeds_known(dict(g.attrs)) for g in groups]
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                if not (known[a] and known[b]
                        and span[a][0] < span[b][1] and span[b][0] < span[a][1]):
                    continue
                # each random feature has its own stream: ONE shared seed
                # repeats that stream, whatever the other seed is
                for name, sa, sb in zip(_SEED_ATTRS, used_seeds[a], used_seeds[b]):
                    if sa is not None and sa == sb:
                        raise ValueError(
                            f"shards {shard_paths[a]} and {shard_paths[b]} use the "
                            f"same {name} {sa} for overlapping particle indices: "
                            "their random draws repeat particle by particle — "
                            "give each independent shard its own seeds")
        if not all(known):
            warnings.warn(
                "independent shards: some shard stores do not record their "
                "seeds (diffusion_seed / beaching_seed), so distinct noise per "
                "shard cannot be checked — make sure each shard had its own seed",
                RuntimeWarning, stacklevel=2)
    n_total = int(sum(counts))
    n_obs = int(time0.shape[0])
    traj_chunk = min(n_total, TRAJ_CHUNK_MAX) or 1
    obs_chunk = int(min(max(TARGET_CHUNK_BYTES // (traj_chunk * 8), 1),
                        OBS_CHUNK_MAX, max(n_obs, 1)))

    out = zarr.open_group(str(out_path), mode="w")
    out.attrs.update({k: v for k, v in attrs0.items()
                      if not k.startswith("restart_")})
    out.attrs["sharded_from"] = [str(p) for p in shard_paths]
    if independent_shards:
        # shard-major order and per-shard noise: the scalar offset and seeds
        # of the first shard would mislead; record every shard's instead
        for k in ("particle_offset",) + _SEED_ATTRS:
            if k in out.attrs:
                del out.attrs[k]
        out.attrs["independent_shards"] = 1
        out.attrs["shard_counts"] = [int(c) for c in counts]
        out.attrs["shard_particle_offsets"] = [None if o is None else int(o)
                                               for o in stamped]
        out.attrs["shard_seeds"] = shard_seeds
    elif stamped[0] is not None:
        out.attrs["particle_offset"] = int(stamped[0])

    for name in sorted(names0):
        first_i = next(i for i, names in enumerate(per_names)
                       if name in names)
        a0 = groups[first_i][name]
        dims = getattr(a0.metadata, "dimension_names", None)
        if dims is None:
            # stores from older writers lack dimension_names — infer the
            # writer's convention so the merged store still opens in
            # xarray (review 2026-08 round 3)
            if name == "time":
                dims = (rec_dim,)
            elif a0.ndim == 2:
                dims = ("trajectory", rec_dim)
            else:
                dims = ("trajectory",)
        dim_kw = {"dimension_names": dims, "fill_value": a0.fill_value}
        if name == "time":
            arr = out.create_array("time", shape=time0.shape,
                                   dtype=a0.dtype, **dim_kw)
            arr[:] = time0
        elif name == "trajectory":
            arr = out.create_array("trajectory", shape=(n_total,),
                                   dtype=a0.dtype, **dim_kw)
            # global IDs: the first shard's place in the release (0 for an
            # unsharded numbering, and for independent shards)
            base = (int(stamped[0]) if stamped[0] is not None
                    and not independent_shards else 0)
            if base + n_total - 1 > np.iinfo(a0.dtype).max:
                raise ValueError(
                    f"trajectory IDs {base}..{base + n_total - 1} exceed "
                    f"{a0.dtype} (the type of the trajectory variable)")
            arr[:] = base + np.arange(n_total, dtype=np.int64)
        elif a0.ndim == 2:
            arr = out.create_array(
                name, shape=(n_total, n_obs), dtype=a0.dtype,
                chunks=(traj_chunk, obs_chunk), **dim_kw)
            for i, (p, g, off, cnt) in enumerate(zip(
                    shard_paths, groups, offsets, counts)):
                if name in backfill_nan[i]:
                    # writer parity: dormant u/v are NaN-masked; a format-1
                    # 2-D writer substituted 0 for the missing w (format 2
                    # has no vel_w in 2-D at all) — the mode_3d stamp decides
                    if (name == "vel_w"
                            and "kelp_output_format" not in attrs0
                            and not int(attrs0.get("mode_3d", 1))):
                        arr[off:off + cnt, :] = 0.0
                    else:
                        arr[off:off + cnt, :] = np.nan
                    continue
                src = g[name]
                if src.shape != (cnt, n_obs):
                    raise ValueError(
                        f"shard {p}: variable {name!r} has shape "
                        f"{src.shape}, expected ({cnt}, {n_obs}) — "
                        "truncated or damaged store")
                for o0 in range(0, n_obs, obs_chunk):
                    o1 = min(o0 + obs_chunk, n_obs)
                    arr[off:off + cnt, o0:o1] = src[:, o0:o1]
        elif a0.ndim == 1:
            arr = out.create_array(name, shape=(n_total,),
                                   dtype=a0.dtype, **dim_kw)
            for p, g, off, cnt in zip(shard_paths, groups, offsets,
                                      counts):
                src = g[name]          # 1-D vars are never lazy-missing
                if src.shape != (cnt,):
                    raise ValueError(
                        f"shard {p}: variable {name!r} has shape "
                        f"{src.shape}, expected ({cnt},)")
                arr[off:off + cnt] = np.asarray(src)
        else:
            raise ValueError(
                f"variable {name!r} has unexpected shape {a0.shape} — "
                "not a per-particle or time variable")
        arr.attrs.update(dict(a0.attrs))
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("output")
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--unordered", action="store_true",
                    help="skip the trailing-integer order check")
    ap.add_argument("--independent-shards", action="store_true",
                    help="merge shards that are not blocks of one release keyed "
                         "like it (e.g. a pulsed run split through its restart, "
                         "one seed per shard)")
    args = ap.parse_args(argv)
    out = concat_trajectories(args.shards, args.output, force=args.force,
                              check_order=not args.unordered,
                              independent_shards=args.independent_shards)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
