#!/usr/bin/env python3
"""Split a kelp seed file into contiguous shards for parallel runs.

Particles are independent, so a run over N seeds can be executed as S
independent runs over contiguous shards and the outputs concatenated
(``utils/concat_trajectories.py``) — the linear-scaling lever for
memory-bandwidth-bound tracking: several processes (per NUMA domain, per
node) each get their OWN memory bus instead of queueing on one.

The split is deterministic and ORDER-PRESERVING: shard k holds data rows
``[offsets[k], offsets[k] + counts[k])`` of the input, DATA rows copied
verbatim (no float reformatting; comment/blank lines are dropped and
line endings normalized to LF — the runtime parser ignores them anyway)
after one header line ``# kelp particle_offset <first + offsets[k]>`` (``first``
= the input's own header offset when it is itself a shard, else 0),
so concatenating shard outputs in shard order reproduces the unsharded
particle order exactly.  A ``shards_meta.npz`` sidecar records
``n_total`` (rows of THIS input) / ``first_offset`` / ``offsets`` (global) /
``counts``.

Multi-realization runs (``time.num_runs > 1``) write one store per
realization (``<name>_rN.zarr``) — merge each realization separately.

Do NOT shard configs with ``time.n_pulses > 1``: pulsed release
replicates the SEED SET, so per-shard replication changes the particle
ordering relative to the unsharded run.

Determinism contract: a run sharded this way is data-identical to the
unsharded run (every variable's values match bit for bit; container chunk
layout may differ), stochastic runs included.  Every shard file starts with
a ``# kelp particle_offset N`` header (N = the global index of its first
particle); the engine keys every random draw (diffusion kicks, the vertical
Az kick, stochastic beaching/resuspension) by ``N + local index``, so with
ONE common ``diffusion.seed`` (and ``beaching.seed``) each particle draws the
noise it draws in the unsharded run.  Keep the seeds equal across shards
(the concatenator refuses shards whose configurations differ).

Usage:
    python3 split_seeds.py seeds.dat --shards 4 [--outdir DIR] [--force]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np


def _read_particle_offset(path):
    """The seed file's ``# kelp particle_offset N`` header (or None) —
    ``kelp.core.release.read_particle_offset`` when the package is
    importable, else the same rule inline (the utilities run standalone)."""
    try:
        from kelp.core.release import read_particle_offset
    except ImportError:          # pragma: no cover - standalone fallback
        read_particle_offset = None
    if read_particle_offset is not None:
        return read_particle_offset(path)
    found = None
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if not s.startswith("#"):
                break
            text = s.lstrip("#").split("#", 1)[0]
            if not (re.search(r"particle[_-]offset", text, re.IGNORECASE)
                    or re.match(r"\s*kelp\s+particle\s+offset", text, re.IGNORECASE)):
                continue
            body = text.split()
            if (len(body) != 3 or body[0].lower() != "kelp"
                    or body[1] != "particle_offset"
                    or not (body[2].isascii() and body[2].isdigit())
                    or int(body[2]) > 2 ** 63 - 1 or found is not None):
                raise ValueError(f"{path}: malformed or repeated header {s!r}")
            found = int(body[2])
    return found


#: the names this tool writes (the only files its cleanup may remove)
_SHARD_NAME = re.compile(r"seeds_shard\d{3,}\.dat")


def split_seeds(seed_path, n_shards: int, outdir=None, overwrite=False):
    """Split ``seed_path`` into ``n_shards`` contiguous shard files.

    Returns the list of shard paths (``seeds_shard<k>.dat``) in order.  Existing
    shard files or ``shards_meta.npz`` in ``outdir`` are refused unless
    ``overwrite`` (a re-split into the input's own directory would otherwise
    overwrite its sibling shards, and the merge could not notice); writing over
    the input itself is always refused.
    """
    seed_path = Path(seed_path)
    n_shards = int(n_shards)
    # re-splitting a shard (e.g. to rebalance a slow one) keeps its place in
    # the full release: its own header offsets every new shard
    first = _read_particle_offset(seed_path) or 0
    if n_shards < 1:
        raise ValueError(f"shards must be >= 1, got {n_shards}")
    # DATA rows only: the runtime parser (np.loadtxt) ignores comment
    # lines and blanks, so counting them as particles would desync the
    # shard row count from the run's particle count (review 2026-08).
    # Comment/blank lines are DROPPED from the shards; data rows are
    # copied verbatim (modulo line-ending normalization to LF).
    lines = [ln.rstrip("\r") for ln in seed_path.read_text().splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    n = len(lines)
    if n_shards > n:
        raise ValueError(
            f"more shards ({n_shards}) than seed rows ({n})")
    outdir = Path(outdir) if outdir is not None else seed_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    targets = [outdir / f"seeds_shard{k:03d}.dat" for k in range(n_shards)]
    src = seed_path.resolve()

    def _is_input(p):          # the same FILE: symlinks, hard links, case-folding
        try:
            return os.path.samefile(p, seed_path)
        except OSError:
            return False

    # files this tool wrote: exact generated names only, never the input
    stale = sorted(p for p in outdir.iterdir()
                   if _SHARD_NAME.fullmatch(p.name) and not _is_input(p))
    in_outdir = os.path.samefile(src.parent, outdir)
    if in_outdir and any(_SHARD_NAME.fullmatch(n.lower())
                         for n in (seed_path.name, src.name)):
        raise ValueError(
            f"{seed_path} is a shard file in the output directory: a re-split "
            "there would overwrite its sibling shards — pass a new --outdir")
    if any(_is_input(t) for t in targets if t.exists() or t.is_symlink()):
        raise ValueError(
            f"{seed_path}: a shard would overwrite the input — pass --outdir")
    existing = [t for t in targets + [outdir / "shards_meta.npz"] if t.exists()]
    if (existing or stale) and not overwrite:
        found = existing or stale
        raise ValueError(
            f"{outdir} already holds shard files ({', '.join(p.name for p in found[:3])}"
            f"{', ...' if len(found) > 3 else ''}) — pass a new --outdir, or "
            "--force to replace them")
    for p in stale:
        if p not in targets:
            p.unlink()          # a smaller re-split must not leave old shards behind

    base = n // n_shards
    extra = n % n_shards
    counts = [base + (1 if k < extra else 0) for k in range(n_shards)]
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(int)

    files = []
    for k in range(n_shards):
        p = outdir / f"seeds_shard{k:03d}.dat"
        chunk = lines[offsets[k]:offsets[k] + counts[k]]
        # the shard's place in the full release: keys its random draws
        header = f"# kelp particle_offset {first + int(offsets[k])}"
        if p.exists() or p.is_symlink():
            p.unlink()          # never write THROUGH a link into another file
        p.write_text(header + "\n" + "\n".join(chunk) + "\n")
        files.append(p)
    meta_path = outdir / "shards_meta.npz"
    if meta_path.exists() or meta_path.is_symlink():
        meta_path.unlink()      # never write THROUGH a link into another file
    np.savez(meta_path,
             n_total=n, first_offset=first,
             offsets=np.asarray(offsets, dtype=np.int64) + first,
             counts=np.asarray(counts, dtype=np.int64),
             source=str(seed_path))
    warnings.warn(
        "sharded runs reproduce the unsharded run bit for bit when every "
        "shard uses the SAME diffusion/beaching seed (each shard file's "
        "'# kelp particle_offset' header keys its random draws; a driver "
        "calling run_single_realization passes particle_offset= itself); a "
        "pulsed release (n_pulses > 1) cannot be sharded this way.",
        stacklevel=2)
    return files


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("seed_file")
    ap.add_argument("--shards", type=int, required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing shard files in --outdir")
    args = ap.parse_args(argv)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        files = split_seeds(args.seed_file, args.shards, args.outdir,
                            overwrite=args.force)
    for w in rec:
        print(f"WARNING: {w.message}", file=sys.stderr)
    for f in files:
        print(f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
