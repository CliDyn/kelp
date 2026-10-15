#!/usr/bin/env python3
"""Spatially sort a kelp seed file for memory locality.

The tracking loop's per-particle reads are scattered gathers from
gigabyte velocity arrays; when particles adjacent in the state arrays
are also adjacent on the sphere, consecutive loop iterations reuse the
same cache lines and the effective memory bandwidth multiplies
(measured: +17 % on the fused 3-D step from horizontal sorting alone at
1M particles).  This tool reorders the SEED FILE once, before the run —
it changes the (arbitrary) particle numbering, nothing else.  It
composes with ``utils/split_seeds.py``: sort FIRST, then shard — every
shard is then spatially compact too.

Sort key: HEALPix NEST cell index of (lon, lat) at ``--nside`` (NEST
neighbours are spatially close), with the depth column as the secondary
key when present (3-D runs read layer rows — grouping by depth extends
the locality to the vertical).  The sort is STABLE and deterministic;
data rows are permuted verbatim (comment/blank lines dropped, like the
splitter).

Note: initial sortedness decays as particles disperse over months —
an in-run periodic re-sort is a staged engine feature; this tool is the
cheap first-order win.

Usage:
    python3 sort_seeds.py seeds.dat [--nside 64] [-o sorted.dat]
"""
from __future__ import annotations

import argparse
import re
import sys
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


def sort_seeds(seed_path, nside: int = 64, out_path=None):
    """Sort ``seed_path`` rows by (NEST cell, depth); returns the output
    path (``<stem>_sorted<suffix>`` next to the input by default)."""
    import healpy as hp
    seed_path = Path(seed_path)
    if _read_particle_offset(seed_path) is not None:
        raise ValueError(
            f"{seed_path} is a shard (it carries a '# kelp particle_offset' "
            "header): sorting it would change which particle each random "
            "draw belongs to — sort the full seed file, then split it")
    lines = [ln.rstrip("\r") for ln in seed_path.read_text().splitlines()
             if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"{seed_path}: no data rows")
    # keys parsed from the numeric part only: a trailing inline comment
    # ("lon lat depth  # note") is valid for np.loadtxt and must not
    # break the sort (review 2026-08); the ORIGINAL line is re-emitted.
    def _fields(ln):
        return ln.split("#", 1)[0].split()
    width = min(len(_fields(ln)) for ln in lines)
    cols = np.array([_fields(ln)[:width] for ln in lines],
                    dtype=np.float64)
    lon, lat = cols[:, 0], cols[:, 1]
    pix = hp.ang2pix(int(nside), lon, lat, nest=True, lonlat=True)
    if cols.shape[1] >= 3:
        order = np.lexsort((cols[:, 2], pix))    # depth secondary key
    else:
        order = np.argsort(pix, kind="stable")
    if out_path is None:
        out_path = seed_path.with_name(
            seed_path.stem + "_sorted" + seed_path.suffix)
    out_path = Path(out_path)
    if out_path.resolve() == seed_path.resolve():
        raise ValueError(
            f"{out_path}: output would overwrite the input seed file — "
            "write to a new path")
    out_path.write_text("\n".join(lines[i] for i in order) + "\n")
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("seed_file")
    ap.add_argument("--nside", type=int, default=64,
                    help="HEALPix NEST resolution of the sort key "
                         "(≈ the mesh scale; default 64 ≈ 1°)")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args(argv)
    print(sort_seeds(args.seed_file, nside=args.nside,
                     out_path=args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
