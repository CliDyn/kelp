"""Node-shared snapshot cache: publish once, attach many.

Co-resident kelp processes (particle shards of one run) each used to
read and hold their OWN copy of every velocity snapshot — S full-field
replicas per node and S store reads whose memory spikes align at every
model-day boundary. This module lets the first process publish the fully
processed arrays (compacted, rotated, cast) to a shared directory with
an atomic rename, while every other process attaches the SAME bytes as
read-only memory maps: one physical field copy per node page cache and
one store read per publish scope. Point the root at a node-local tmpfs
for pure-RAM sharing, or at a directory on the shared filesystem — then
ONE writer serves the whole multi-node run and each node still shares
the mapped pages through its page cache.

Engine-agnostic by design: ``dict[name -> ndarray]`` plus a JSON-able
``meta`` dict go in, the same come out — assembling ``VelocityField``
objects stays in the tracking layer.

On-disk layout: ``<root>/<tag>/<key>/{<name>.npy, manifest.json}``.
``tag`` namespaces a run configuration (different configs never share
bytes); ``key`` is typically the snapshot's time index. Writers build in
``<key>.tmp.<pid>`` and ``os.rename`` it into place — atomic within one
filesystem, so ``attach`` can never observe a half-written snapshot.
Writer election is a blocking ``flock`` on ``<root>/<tag>/.lock``:
losers wake up, re-check, and attach what the winner published. Stale
``.tmp`` dirs from crashed writers are swept by the next lock holder.
``gc(keep=...)`` drops published keys outside the caller's window —
safe even while other processes still map them (POSIX keeps unlinked
pages alive until the last unmap); a process that still needs a
collected key simply rebuilds it through ``get_or_publish``.
Every user registers a marker ``<root>/<tag>/.users/<host>.<pid>.<rand>``;
``close()`` (also run at interpreter exit) removes it.  A namespace belongs
to one run generation, so once all its users have ended nothing reads it
again: a NEW cache removes such namespaces of earlier runs under a
root-level lock (``_sweep_stale_namespaces``).  A marker counts as ended when
it is gone or names a dead process on THIS host; markers from other hosts
count as alive (their namespace is kept).  Nobody wipes its own namespace —
wiping it at the last close raced with a sibling that had not joined yet
(review 2026-09-23).  Liveness is judged by (hostname, pid): processes in
separate PID namespaces on one host (containers), or containers with equal
hostnames on nodes sharing the root, can look ended and lose their snapshots;
so can every node of a run when the filesystem's ``flock`` is node-local
(Lustre ``localflock``).  All of these fail OPEN (the tracking layer falls
back to local reads), never to wrong data.
"""
from __future__ import annotations

import atexit
import json
import os
import shutil
import socket
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import numpy as np

try:                                    # POSIX-only writer election
    import fcntl
except ImportError:                     # pragma: no cover - non-POSIX
    fcntl = None

_MANIFEST = "manifest.json"

#: (arrays, meta) as produced by a builder / returned by attach
Snapshot = Tuple[Dict[str, np.ndarray], Dict[str, Any]]


class SharedFieldCache:
    """Publish-once / attach-many array snapshots under ``root/tag``."""

    def __init__(self, root_dir, tag: str):
        if fcntl is None:               # pragma: no cover - non-POSIX
            raise OSError(
                "shared_fields needs POSIX file locks (fcntl).")
        base = Path(root_dir)
        base.mkdir(parents=True, exist_ok=True)
        self.root = base / str(tag)
        self._marker = None
        self._pid = os.getpid()
        # join the namespace and sweep ended ones under one root lock, so a
        # namespace being created is never seen without its creator's marker
        with open(base / ".root.lock", "w") as rl:
            fcntl.flock(rl, fcntl.LOCK_EX)
            try:
                users = self.root / ".users"
                users.mkdir(parents=True, exist_ok=True)
                self._marker = users / (
                    f"{socket.gethostname()}.{os.getpid()}."
                    f"{uuid.uuid4().hex[:8]}")
                self._marker.touch()
                doomed = self._sweep_stale_namespaces(base)
            finally:
                fcntl.flock(rl, fcntl.LOCK_UN)
        # delete outside the lock: siblings starting now need not wait
        for tomb in doomed:
            shutil.rmtree(tomb, ignore_errors=True)
        atexit.register(self.close)

    def close(self) -> None:
        """Leave the namespace (its snapshots stay for siblings that have
        not joined yet; the next run removes them)."""
        if os.getpid() != getattr(self, "_pid", os.getpid()):
            return        # a forked child: the marker is the parent's
        marker, self._marker = getattr(self, "_marker", None), None
        if marker is None:
            return
        try:
            marker.unlink()
        except OSError:
            pass

    @staticmethod
    def _marker_alive(marker: Path) -> bool:
        host, _, rest = marker.name.rpartition(".")[0].rpartition(".")
        try:
            pid = int(rest)
        except ValueError:
            return True                 # not ours to judge
        if host != socket.gethostname():
            return True                 # another node: assume alive
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except OSError:
            return True
        return True

    def _sweep_stale_namespaces(self, base: Path) -> list:
        """Rename the namespaces of earlier runs whose users have all ended
        to tombstones (root lock held) and return them for deletion after
        the lock is released.  Namespaces without a ``.users`` directory
        (written before 2026-09-23) are left alone."""
        doomed = []
        for d in base.iterdir():
            if (not d.is_dir() or d.name.startswith(".")
                    or d == self.root):
                continue
            users = d / ".users"
            if not users.is_dir():
                continue
            try:
                if any(self._marker_alive(m) for m in users.iterdir()):
                    continue
            except OSError:
                continue
            tomb = base / f".gone.{d.name}.{os.getpid()}"
            try:
                os.rename(d, tomb)
            except OSError:
                continue
            doomed.append(tomb)
        doomed.extend(p for p in base.glob(".gone.*") if p not in doomed)
        return doomed

    # -- read side ---------------------------------------------------
    def attach(self, key: str) -> Optional[Snapshot]:
        """Map a published snapshot read-only, or None if not published.

        The manifest only exists inside a fully renamed snapshot dir, so
        a successful read here implies every array file is complete.
        """
        d = self.root / key
        try:
            man = json.loads((d / _MANIFEST).read_text())
        except (OSError, ValueError):
            return None
        arrays: Dict[str, np.ndarray] = {}
        try:
            for name in man["arrays"]:
                arrays[name] = np.load(d / f"{name}.npy", mmap_mode="r")
        except OSError:                 # collected under our feet
            return None
        return arrays, man.get("meta", {})

    # -- write side --------------------------------------------------
    def get_or_publish(self, key: str,
                       builder: Callable[[], Snapshot]
                       ) -> Tuple[Snapshot, bool]:
        """Attach ``key``, or win the writer lock, build and publish.

        Returns ``(snapshot, was_published)``; the winner also gets the
        MAPPED view (not its private build), so every process consumes
        the shared bytes.
        """
        got = self.attach(key)
        if got is not None:
            return got, False
        with open(self.root / ".lock", "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                got = self.attach(key)      # a sibling won the race
                if got is not None:
                    return got, False
                self._sweep_tmp()
                target = self.root / key
                if target.exists():
                    # attach failed on an EXISTING dir: a gc killed
                    # mid-delete left a broken snapshot — quarantine and
                    # drop it, or the publish rename below would fail
                    # (review 2026-08)
                    self._destroy(target)
                arrays, meta = builder()
                tmp = self.root / f"{key}.tmp.{os.getpid()}"
                shutil.rmtree(tmp, ignore_errors=True)
                tmp.mkdir()
                for name, arr in arrays.items():
                    if "/" in name or name == _MANIFEST:
                        raise ValueError(
                            f"shared_fields: array name {name!r} is not "
                            "filename-safe")
                    # C-order on disk: consumers (parallel time blends)
                    # gate on C-contiguity
                    np.save(tmp / f"{name}.npy",
                            np.ascontiguousarray(arr))
                (tmp / _MANIFEST).write_text(json.dumps(
                    {"arrays": sorted(arrays), "meta": meta}))
                os.rename(tmp, d := self.root / key)
                assert d.exists()
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)
        att = self.attach(key)
        if att is None:                 # pragma: no cover - fs failure
            raise OSError(
                f"shared_fields: snapshot {key!r} vanished right after "
                "publish — is the shared directory being cleaned "
                "externally?")
        return att, True

    # -- maintenance -------------------------------------------------
    def gc(self, keep: Iterable[str]) -> None:
        """Remove published snapshots whose key is not in ``keep``."""
        keep = set(keep)
        with open(self.root / ".lock", "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                for d in self.root.iterdir():
                    if (d.is_dir() and d.name not in keep
                            and ".tmp." not in d.name
                            and not d.name.startswith(".")):
                        self._destroy(d)
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    def _destroy(self, d: Path) -> None:
        """Remove a snapshot dir atomically-for-readers (lock held).

        Rename to a hidden tombstone FIRST (atomic — attach can never
        see a half-deleted snapshot), then delete the tombstone; a kill
        mid-delete leaves only a tombstone, swept by the next writer.
        """
        tomb = d.parent / f".gone.{d.name}.{os.getpid()}"
        try:
            os.rename(d, tomb)
        except OSError:
            tomb = d                    # already tomb-named / raced away
        shutil.rmtree(tomb, ignore_errors=True)

    def _sweep_tmp(self) -> None:
        """Drop half-built and half-deleted leftovers (lock held)."""
        for p in self.root.glob("*.tmp.*"):
            shutil.rmtree(p, ignore_errors=True)
        for p in self.root.glob(".gone.*"):
            shutil.rmtree(p, ignore_errors=True)
