#!/usr/bin/env python
"""Keep docs/doc_map.yaml honest and surface documentation drift.

Pure-core functions (testable without git) + a thin CLI.  The map is the
canonical subsystem -> {doc, code anchors, verified-at commit} table; this
tool validates it, checks file:symbol anchors against the source, reports
which docs have code changed under them since they were verified, and can
re-stamp the verified-at commit (--bless).
"""
from __future__ import annotations

import argparse
import functools
import pathlib
import re
import subprocess
import sys

import yaml


def load_map(path) -> dict:
    """Parse doc_map.yaml into {subsystem: {doc, code, verified_at}}."""
    data = yaml.safe_load(pathlib.Path(path).read_text()) or {}
    return data


def _split_anchor(entry: str):
    """'path.py:Symbol' -> ('path.py', 'Symbol'); 'path.py' -> (path, None)."""
    if ":" in entry:
        path, sym = entry.split(":", 1)
        return path, sym
    return entry, None


def validate_structure(doc_map: dict, root: pathlib.Path) -> list[str]:
    """Return a list of problems: missing doc files or missing code paths."""
    problems = []
    for name, entry in doc_map.items():
        doc = entry.get("doc")
        if not doc or not (root / doc).exists():
            problems.append(f"[{name}] doc not found: {doc!r}")
        # Every entry must carry a verified_at: without it find_drift cannot
        # key the git query, so drift goes silently undetected (this is how a
        # dropped stamp once slipped through unnoticed).
        if not entry.get("verified_at"):
            problems.append(f"[{name}] missing verified_at")
        for code in entry.get("code", []):
            path, _sym = _split_anchor(code)
            if not (root / path).exists():
                problems.append(f"[{name}] code path not found: {path!r}")
    return problems


def check_anchors(doc_map: dict, root: pathlib.Path) -> list[str]:
    """Return problems for file:symbol anchors whose symbol is not defined."""
    problems = []
    for name, entry in doc_map.items():
        for code in entry.get("code", []):
            path, sym = _split_anchor(code)
            if sym is None:
                continue
            f = root / path
            if not f.exists():
                continue  # missing path already reported by validate_structure
            text = f.read_text()
            # match `def sym(` / `async def sym(` / `class sym(` / `class sym:`
            pat = rf"^[ \t]*(?:async[ \t]+)?(?:def|class)[ \t]+{re.escape(sym)}\b"
            if not re.search(pat, text, re.MULTILINE):
                problems.append(
                    f"[{name}] anchor symbol not found: {sym!r} in {path}")
    return problems


class Drift:
    """One (subsystem, code-path) that changed since verified_at."""
    def __init__(self, subsystem, path, commits):
        self.subsystem = subsystem
        self.path = path
        self.commits = commits

    def __repr__(self):
        return f"Drift({self.subsystem!r}, {self.path!r}, n={len(self.commits)})"


def commits_since(path: str, ref: str, root: pathlib.Path = None) -> list[str]:
    """Git commits touching `path` after `ref` (oneline), newest first."""
    cmd = ["git", "log", "--oneline", f"{ref}..HEAD", "--", path]
    out = subprocess.run(cmd, cwd=str(root) if root else None,
                         capture_output=True, text=True)
    if out.returncode != 0:
        return []
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


_git_commits_since = commits_since


def ref_resolves(ref: str, root=None) -> bool:
    """True when `ref` names a commit in this repository.

    A stamp from another history (a fresh-history clone, a squashed export)
    makes ``git log ref..HEAD`` fail, and `commits_since` then reports no
    drift — silently verifying nothing.  `run_report` uses this to say so.
    """
    out = subprocess.run(["git", "rev-parse", "--verify", "--quiet",
                          f"{ref}^{{commit}}"],
                         cwd=str(root) if root else None,
                         capture_output=True, text=True)
    return out.returncode == 0


def find_drift(doc_map: dict, commits_since=None, root=None) -> list:
    """Drift reports for every code path changed since its subsystem's
    verified_at.  The git query runs in ``root`` (the current directory when
    None); ``commits_since(path, ref)`` is injectable for testing."""
    if commits_since is None:
        commits_since = functools.partial(_git_commits_since, root=root)
    reports = []
    for name, entry in doc_map.items():
        ref = entry.get("verified_at")
        for code in entry.get("code", []):
            path, _sym = _split_anchor(code)
            changed = commits_since(path, ref)
            if changed:
                reports.append(Drift(name, path, changed))
    return reports


def bless(doc_map: dict, subsystem: str, head: str) -> dict:
    """Return a copy of doc_map with subsystem's verified_at set to head."""
    if subsystem not in doc_map:
        raise KeyError(subsystem)
    import copy
    out = copy.deepcopy(doc_map)
    out[subsystem]["verified_at"] = head
    return out


def run_report(map_path, root, commits_since=None) -> int:
    """Validate + anchor-check + drift-report.  Returns nonzero if any
    structural/anchor problem OR drift is found.  The drift query runs in
    ``root`` (it ran in the current directory and reported 'in sync' from
    anywhere outside the repository root — audit 2026-10-04)."""
    root = pathlib.Path(root)
    doc_map = load_map(map_path)
    problems = validate_structure(doc_map, root) + check_anchors(doc_map, root)
    for p in problems:
        print(f"PROBLEM {p}")
    stamps = {e.get("verified_at") for e in doc_map.values() if e.get("verified_at")}
    # "HEAD" resolves but ``git log HEAD..HEAD`` is always empty, so a HEAD
    # stamp can never report drift either.
    unverifiable = sorted(r for r in stamps
                          if r == "HEAD" or not ref_resolves(r, root))
    if unverifiable:
        print(f"NOTE   {len(unverifiable)} verified_at stamp(s) cannot be checked "
              f"here ({', '.join(unverifiable)}): a stamp from another history, or "
              "the literal 'HEAD' — drift is NOT verified for the subsystems "
              "carrying them.")
    all_unverifiable = bool(stamps) and len(unverifiable) == len(stamps)
    drift = find_drift(doc_map, commits_since=commits_since, root=root)
    for d in drift:
        print(f"DRIFT  [{d.subsystem}] {d.path} changed in "
              f"{len(d.commits)} commit(s) since verified_at:")
        for c in d.commits:
            print(f"         {c}")
    if not problems and not drift:
        if all_unverifiable:
            print(f"OK  structure + anchors valid; drift NOT checked "
                  f"({len(stamps)}/{len(stamps)} stamps unresolvable here)")
        else:
            print("OK  all docs in sync, anchors resolve, structure valid")
        return 0
    return 1


def run_bless(map_path, subsystem, root=None) -> int:
    """Stamp the repository's HEAD into one subsystem's verified_at,
    rewrite map."""
    map_path = pathlib.Path(map_path)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          cwd=str(root) if root else None,
                          capture_output=True, text=True,
                          check=True).stdout.strip()
    doc_map = load_map(map_path)
    out = bless(doc_map, subsystem, head)
    map_path.write_text(yaml.safe_dump(out, sort_keys=True,
                                       default_flow_style=False))
    print(f"blessed [{subsystem}] -> verified_at={head}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="doc_map sync checker")
    here = pathlib.Path(__file__).resolve().parent
    ap.add_argument("--map", default=str(here / "doc_map.yaml"))
    ap.add_argument("--root", default=str(here.parent),
                    help="repo root (default: parent of docs/)")
    ap.add_argument("--bless", metavar="SUBSYSTEM",
                    help="stamp current HEAD into this subsystem's verified_at")
    args = ap.parse_args(argv)
    if args.bless:
        return run_bless(args.map, args.bless, root=args.root)
    return run_report(args.map, args.root)


if __name__ == "__main__":
    sys.exit(main())
