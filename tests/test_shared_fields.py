"""Node-shared velocity-field cache (`core/shared_fields.py`).

One process per shared directory publishes the fully processed snapshot
arrays (atomic rename); every other process attaches the SAME bytes via
read-only memory maps, so co-resident processes stop replicating the
fields and stop reading the store S times. The module is engine-
agnostic: dict[str, ndarray] + a small JSON-able meta dict in, the same
out — VelocityField assembly stays in the tracking layer.
"""
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kelp.core.shared_fields import SharedFieldCache  # noqa: E402


def _arrays():
    rng = np.random.default_rng(3)
    return {"u": rng.standard_normal((4, 9)),
            "v": rng.standard_normal((4, 9)).astype(np.float32),
            "eta": rng.standard_normal(9)}


class TestRoundtrip:
    def test_publish_then_attach_bitwise(self, tmp_path):
        c = SharedFieldCache(tmp_path, tag="cfgA")
        src = _arrays()
        meta = {"uv_layout": "node", "n": 3}
        got, was_published = c.get_or_publish(
            "t000005", lambda: (src, meta))
        assert was_published
        c2 = SharedFieldCache(tmp_path, tag="cfgA")
        att, was_published2 = c2.get_or_publish(
            "t000005", lambda: (_ for _ in ()).throw(AssertionError(
                "builder must not run on attach")))
        assert not was_published2
        for name, a in src.items():
            b = att[0][name]
            assert b.dtype == a.dtype and b.shape == a.shape
            np.testing.assert_array_equal(a, b)
            assert not b.flags.writeable          # read-only mapping
        assert att[1] == meta

    def test_different_tag_is_a_different_namespace(self, tmp_path):
        a = SharedFieldCache(tmp_path, tag="cfgA")
        b = SharedFieldCache(tmp_path, tag="cfgB")
        a.get_or_publish("t1", lambda: ({"x": np.ones(3)}, {}))
        ran = []
        b.get_or_publish("t1", lambda: (ran.append(1),
                                        ({"x": np.zeros(3)}, {}))[1])
        assert ran                                 # cfgB built its own

    def test_partial_tmp_dir_is_ignored(self, tmp_path):
        c = SharedFieldCache(tmp_path, tag="cfgA")
        junk = c.root / "t7.tmp.12345"
        junk.mkdir(parents=True)
        (junk / "u.npy").write_bytes(b"garbage")
        got, was_published = c.get_or_publish(
            "t7", lambda: ({"u": np.arange(3.0)}, {}))
        assert was_published
        np.testing.assert_array_equal(got[0]["u"], np.arange(3.0))

    def test_gc_keeps_only_requested_keys(self, tmp_path):
        c = SharedFieldCache(tmp_path, tag="cfgA")
        for k in ("t1", "t2", "t3"):
            c.get_or_publish(k, lambda: ({"x": np.ones(2)}, {}))
        c.gc(keep=("t2", "t3"))
        assert c.attach("t1") is None
        assert c.attach("t2") is not None and c.attach("t3") is not None

    def test_attach_absent_returns_none(self, tmp_path):
        assert SharedFieldCache(tmp_path, tag="z").attach("t9") is None


def _race_worker(root, counter_dir, q):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    import numpy as np  # noqa: F811
    from kelp.core.shared_fields import SharedFieldCache

    def build():
        # side-effect file marks one builder execution
        n = len(list(Path(counter_dir).glob("ran*")))
        (Path(counter_dir) / f"ran{n}_{mp.current_process().pid}").touch()
        return {"u": np.full(5, 7.0)}, {"who": "worker"}

    c = SharedFieldCache(root, tag="race")
    got, _ = c.get_or_publish("t0", build)
    q.put(float(got[0]["u"][0]))


class TestConcurrency:
    def test_two_processes_one_builder(self, tmp_path):
        counter = tmp_path / "counter"
        counter.mkdir()
        q = mp.Queue()
        ps = [mp.Process(target=_race_worker,
                         args=(str(tmp_path), str(counter), q))
              for _ in range(2)]
        for p in ps:
            p.start()
        vals = [q.get(timeout=60) for _ in ps]
        for p in ps:
            p.join(timeout=60)
        assert vals == [7.0, 7.0]
        assert len(list(counter.glob("ran*"))) == 1   # exactly one build


class TestLifecycleRobustness:
    def test_broken_leftover_key_is_rebuilt(self, tmp_path):
        # simulate a gc killed mid-delete: key dir exists but manifest gone
        c = SharedFieldCache(tmp_path, tag="cfgA")
        c.get_or_publish("t5", lambda: ({"x": np.ones(3)}, {}))
        (c.root / "t5" / "manifest.json").unlink()
        got, published = c.get_or_publish(
            "t5", lambda: ({"x": np.full(3, 2.0)}, {}))
        assert published                        # rebuilt, not crashed
        np.testing.assert_array_equal(got[0]["x"], np.full(3, 2.0))

    def test_tombstones_are_swept(self, tmp_path):
        c = SharedFieldCache(tmp_path, tag="cfgA")
        (c.root / ".gone.t9.111").mkdir(parents=True)
        c.get_or_publish("t1", lambda: ({"x": np.ones(2)}, {}))
        assert not list(c.root.glob(".gone.*"))


class TestStaleNamespaces:
    """Review 2026-09-23: with one namespace per run generation every chain
    leg left its snapshots behind.  A first fix let the LAST process to close
    a namespace wipe it — which raced with a sibling that had not joined yet
    (Python 3.13 forked children run atexit; the sibling rebuilt, and
    test_two_processes_one_builder saw two builds).  Now a NEW cache removes
    the namespaces of earlier runs whose processes have all ended; nobody
    ever wipes its own namespace."""

    def test_ended_run_is_swept_by_the_next(self, tmp_path):
        old = SharedFieldCache(tmp_path, tag="gen1")
        old.get_or_publish("t1", lambda: ({"x": np.ones(3)}, {}))
        old.close()
        assert (tmp_path / "gen1").exists()           # nothing wiped at close
        SharedFieldCache(tmp_path, tag="gen2")
        assert not (tmp_path / "gen1").exists()

    def test_live_run_is_kept(self, tmp_path):
        live = SharedFieldCache(tmp_path, tag="gen1")
        live.get_or_publish("t1", lambda: ({"x": np.ones(3)}, {}))
        SharedFieldCache(tmp_path, tag="gen2")
        assert live.attach("t1") is not None

    def test_dead_process_on_this_host_counts_as_ended(self, tmp_path):
        import os
        import socket
        p = mp.Process(target=int)
        p.start()
        p.join()
        ns = tmp_path / "gen1"
        (ns / ".users").mkdir(parents=True)
        (ns / ".users" / f"{socket.gethostname()}.{p.pid}.dead0000").touch()
        (ns / "t1").mkdir()
        SharedFieldCache(tmp_path, tag="gen2")
        assert not ns.exists()
        assert os.getpid() != p.pid

    def test_marker_from_another_host_is_kept(self, tmp_path):
        ns = tmp_path / "gen1"
        (ns / ".users").mkdir(parents=True)
        (ns / ".users" / "some-other-node.12345.abcd0000").touch()
        (ns / "t1").mkdir()
        SharedFieldCache(tmp_path, tag="gen2")
        assert ns.exists()

    def test_own_namespace_never_swept_and_close_idempotent(self, tmp_path):
        a = SharedFieldCache(tmp_path, tag="gen1")
        a.get_or_publish("t1", lambda: ({"x": np.ones(3)}, {}))
        a.close()
        a.close()
        b = SharedFieldCache(tmp_path, tag="gen1")   # a late joiner
        assert b.attach("t1") is not None

    def test_forked_child_leaves_the_parents_marker(self, tmp_path):
        """A raw os.fork child that exits normally runs the inherited atexit
        close(); it must not remove the parent's marker (review round 2).
        Run in a separate interpreter so the fork does not copy pytest."""
        import subprocess
        import sys
        script = tmp_path / "fork_probe.py"
        script.write_text(
            "import os, sys, numpy as np\n"
            f"sys.path.insert(0, {str(Path(__file__).resolve().parent.parent / 'src')!r})\n"
            "from kelp.core.shared_fields import SharedFieldCache\n"
            "root = sys.argv[1]\n"
            "c = SharedFieldCache(root, tag='gen1')\n"
            "c.get_or_publish('t1', lambda: ({'x': np.ones(3)}, {}))\n"
            "pid = os.fork()\n"
            "if pid == 0:\n"
            "    sys.exit(0)\n"
            "os.waitpid(pid, 0)\n"
            "users = os.listdir(os.path.join(root, 'gen1', '.users'))\n"
            "SharedFieldCache(root, tag='gen2')\n"
            "print(len(users), c.attach('t1') is not None)\n")
        out = subprocess.run([sys.executable, str(script), str(tmp_path)],
                             capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, out.stderr
        assert out.stdout.split() == ["1", "True"], out.stdout
