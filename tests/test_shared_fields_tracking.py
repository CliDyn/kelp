"""Tracking-layer wiring of the node-shared snapshot cache.

`_VelCache(shared=...)` must: read the store ONCE per shared directory
(the second process attaches, its reader is never called), reproduce the
VelocityField bit for bit (incl. w=None, scalars, kh pair), force
prefetch back to 1 (the shared writer already amortizes I/O), and fail
OPEN — any shared-layer breakage falls back to plain local reads with a
one-time warning, never a dead run.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kelp.core.io import VelocityField  # noqa: E402
from kelp.core.shared_fields import SharedFieldCache  # noqa: E402
from kelp.tracking import _VelCache  # noqa: E402


class _CountingReader:
    def __init__(self, with_w=True, with_scalars=True, with_kh=False):
        self.calls = 0
        self.with_w = with_w
        self.with_scalars = with_scalars
        self.with_kh = with_kh

    def read_velocities(self, ds, t, mesh, mode, **kw):
        self.calls += 1
        rng = np.random.default_rng(100 + t)
        n, k = 11, 4
        kh = {}
        if self.with_kh:
            kh = dict(kh_zonal=rng.standard_normal(n),
                      kh_meridional=rng.standard_normal(n),
                      kh_layout="node")
        return VelocityField(
            u=rng.standard_normal((k - 1, n)),
            v=rng.standard_normal((k - 1, n)),
            eta=rng.standard_normal(n),
            w=rng.standard_normal((k, n)) if self.with_w else None,
            uv_layout="node",
            scalars=({"temp": rng.standard_normal((k - 1, n))}
                     if self.with_scalars else None),
            **kh)


def _assert_vel_equal(a, b):
    for f in ("u", "v", "eta", "w", "kh_zonal", "kh_meridional"):
        x, y = getattr(a, f), getattr(b, f)
        if x is None:
            assert y is None
        else:
            assert y.dtype == x.dtype
            np.testing.assert_array_equal(x, y)
    assert a.uv_layout == b.uv_layout
    assert a.kh_layout == b.kh_layout
    if a.scalars is None:
        assert b.scalars is None
    else:
        assert set(a.scalars) == set(b.scalars)
        for nm in a.scalars:
            np.testing.assert_array_equal(a.scalars[nm], b.scalars[nm])


class TestSharedVelCache:
    @pytest.mark.parametrize("with_w,with_scalars,with_kh",
                             [(True, True, False), (False, False, False),
                              (True, False, True)])
    def test_second_process_attaches_bitwise(self, tmp_path, with_w,
                                             with_scalars, with_kh):
        mk = lambda: _CountingReader(with_w, with_scalars, with_kh)
        r1, r2 = mk(), mk()
        c1 = _VelCache(shared=SharedFieldCache(tmp_path, "tagX"))
        v1 = c1.get(3, r1, None, None, None)
        assert r1.calls == 1
        c2 = _VelCache(shared=SharedFieldCache(tmp_path, "tagX"))
        v2 = c2.get(3, r2, None, None, None)
        assert r2.calls == 0                    # attached, never read
        _assert_vel_equal(v1, v2)

    def test_shared_failure_falls_open(self, tmp_path):
        shared = SharedFieldCache(tmp_path, "tagY")
        r = _CountingReader()
        c = _VelCache(shared=shared)
        # brick the shared root AFTER init: attach and lock both fail
        import os
        import shutil
        shutil.rmtree(shared.root)
        (tmp_path / "tagY").write_text("not a directory")
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            vel = c.get(0, r, None, None, None)
        assert r.calls == 1                     # local read happened
        assert vel.u.shape == (3, 11)
        assert any("shared" in str(w.message).lower() for w in rec)
        # and the cache stays functional (shared disabled, no re-raise)
        c.get(1, r, None, None, None)
        assert r.calls == 2

    def test_prefetch_forced_to_one(self, tmp_path):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            c = _VelCache(prefetch_size=4,
                          shared=SharedFieldCache(tmp_path, "tagZ"))
        assert c._prefetch == 1
        assert any("prefetch" in str(w.message).lower() for w in rec)

    def test_writer_gcs_old_snapshots(self, tmp_path):
        shared = SharedFieldCache(tmp_path, "tagG")
        r = _CountingReader()
        c = _VelCache(shared=shared)
        for t in range(9):
            c.get(t, r, None, None, None)
            c.evict_except({t})
        # far-behind snapshots must be gone from the shared dir
        assert shared.attach("t00000000") is None
        assert shared.attach("t00000008") is not None


class TestSharedFieldsTag:
    @pytest.fixture(autouse=True)
    def _one_generation(self, monkeypatch):
        # every tag in these tests belongs to one run unless a test says
        # otherwise (a generation is required since review round 2)
        monkeypatch.setenv("KELP_SHARED_FIELDS_GENERATION", "test-run")

    def _cfg(self, tmp_path, name, inicloud="a.dat", out="o.zarr",
             var_u=None):
        from kelp.core.config import load_config
        extra = f"\n  var_u: {var_u}" if var_u else ""
        (tmp_path / name).write_text(f"""
mode:
  Fdrift2D: true{extra}
numerics:
  dt: 3600.0
mesh:
  type: fesomc
  meshpath: nowhere/mesh.nc
data:
  ncinput: nowhere/data.nc
  inicloud: {inicloud}
  output_path: {out}
time:
  start_time_idx: 0
  n_steps: 2
""")
        return load_config(tmp_path / name)

    def test_shards_share_tag_but_var_change_does_not(self, tmp_path):
        from kelp.tracking import _shared_fields_tag
        c1 = self._cfg(tmp_path, "c1.yaml", inicloud="shard0.dat",
                       out="s0.zarr")
        c2 = self._cfg(tmp_path, "c2.yaml", inicloud="shard1.dat",
                       out="s1.zarr")
        c3 = self._cfg(tmp_path, "c3.yaml", var_u="unod")
        t1, t2, t3 = map(_shared_fields_tag, (c1, c2, c3))
        assert t1 == t2                     # shards: same namespace
        assert t1 != t3                     # different read → different ns
    def test_rewritten_data_changes_the_tag(self, tmp_path):
        """Review 2026-09-22 (F08): the namespace hashed paths and config
        only, so a store rewritten at the same path was served from the old
        cache (u=1 read after the source became u=2)."""
        import os
        import time
        from kelp.tracking import _shared_fields_tag
        data = tmp_path / "data.nc"
        data.write_bytes(b"u=1")
        cfg = self._cfg(tmp_path, "d.yaml")
        cfg.data.ncinput = str(data)
        t1 = _shared_fields_tag(cfg)
        assert _shared_fields_tag(cfg) == t1          # stable
        data.write_bytes(b"u=22")                     # new content + size
        os.utime(data, ns=(time.time_ns(), time.time_ns() + 10**9))
        assert _shared_fields_tag(cfg) != t1

    def test_rewritten_store_directory_changes_the_tag(self, tmp_path):
        import os
        import shutil
        import time
        from kelp.tracking import _shared_fields_tag
        store = tmp_path / "u.zarr"
        (store / "u").mkdir(parents=True)
        (store / "u" / "c0").write_bytes(b"1")
        cfg = self._cfg(tmp_path, "z.yaml")
        cfg.data.ncinput = str(store)
        t1 = _shared_fields_tag(cfg)
        shutil.rmtree(store)
        (store / "u").mkdir(parents=True)
        (store / "u" / "c0").write_bytes(b"22")
        later = time.time_ns() + 10**9
        for p in (store / "u", store):
            os.utime(p, ns=(later, later))
        assert _shared_fields_tag(cfg) != t1

    def test_generation_separates_runs(self, tmp_path, monkeypatch):
        """Review 2026-09-23 (blocker): an in-place rewrite of a nested
        Zarr chunk changes no stat the fingerprint sees, so the namespace
        also carries the run generation: shards of one job share it, two
        jobs never do."""
        from kelp.tracking import _shared_fields_tag
        cfg = self._cfg(tmp_path, "g.yaml")
        for k in ("KELP_SHARED_FIELDS_GENERATION", "SLURM_ARRAY_JOB_ID",
                  "SLURM_JOB_ID", "SLURM_STEP_ID"):
            monkeypatch.delenv(k, raising=False)
        # review round 2: no identifiable run -> no namespace (the
        # run_tracking caller turns this into local reads + a warning)
        with pytest.raises(ValueError, match="shared_fields_generation"):
            _shared_fields_tag(cfg)
        monkeypatch.setenv("SLURM_JOB_ID", "1001")
        t1 = _shared_fields_tag(cfg)
        assert _shared_fields_tag(cfg) == t1           # shards of one job
        monkeypatch.setenv("SLURM_STEP_ID", "0")       # srun step 0 of 1001
        t_s0 = _shared_fields_tag(cfg)
        monkeypatch.setenv("SLURM_STEP_ID", "1")       # the next srun step
        assert _shared_fields_tag(cfg) != t_s0
        monkeypatch.delenv("SLURM_STEP_ID")
        monkeypatch.setenv("SLURM_JOB_ID", "1002")
        assert _shared_fields_tag(cfg) != t1           # the next job
        monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "77")  # array tasks share
        monkeypatch.setenv("SLURM_JOB_ID", "1003")
        t_a = _shared_fields_tag(cfg)
        monkeypatch.setenv("SLURM_JOB_ID", "1004")
        assert _shared_fields_tag(cfg) == t_a
        cfg.numerics.shared_fields_generation = "campaign-v2"   # explicit wins
        t_e = _shared_fields_tag(cfg)
        monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "78")
        assert _shared_fields_tag(cfg) == t_e

    def test_beaching_dry_mask_changes_tag(self, tmp_path):
        from kelp.tracking import _shared_fields_tag
        base = self._cfg(tmp_path, "b1.yaml")
        import dataclasses
        beach = dataclasses.replace(
            base.numerics.beaching, enabled=True,
            criteria=["depth"], dry_mask_var="wetdry")
        num = dataclasses.replace(base.numerics, beaching=beach)
        withmask = dataclasses.replace(base, numerics=num)
        assert _shared_fields_tag(base) != _shared_fields_tag(withmask)


class TestEndToEndByteIdentity:
    def test_run_with_shared_equals_run_without(self, tmp_path):
        # full run_tracking twice on the CLI suite's minimal FESOM-C
        # fixture: shared cache ON vs OFF must produce identical
        # trajectories AND identical metadata (no execution-detail leak)
        from test_cli import (_make_fesomc_minimal, _write_config,
                              _write_particles)
        import xarray as xr
        from kelp.core.config import load_config
        from kelp.tracking import run_tracking

        nc = _make_fesomc_minimal(tmp_path)
        seeds = tmp_path / "seeds.dat"
        _write_particles(seeds)
        outs = {}
        for mode in ("off", "on"):
            out = tmp_path / f"traj_{mode}.nc"
            cfg_p = _write_config(tmp_path / f"cfg_{mode}.yaml", nc,
                                  seeds, out)
            if mode == "on":
                txt = cfg_p.read_text().replace(
                    "numerics:",
                    "numerics:\n  shared_fields_dir: "
                    f"{tmp_path / 'shm'}\n"
                    "  shared_fields_generation: e2e-1")
                cfg_p.write_text(txt)
            run_tracking(load_config(cfg_p), output_path=str(out))
            outs[mode] = out
        with xr.open_dataset(outs["off"]) as a, \
                xr.open_dataset(outs["on"]) as b:
            assert set(a.data_vars) == set(b.data_vars)
            for v in a.data_vars:
                np.testing.assert_array_equal(a[v].values, b[v].values)
            ka = {k: v for k, v in a.attrs.items() if k != "history"}
            kb = {k: v for k, v in b.attrs.items() if k != "history"}
            assert ka == kb
        # and the shared dir was actually used (one published namespace)
        assert any((tmp_path / "shm").iterdir())


def test_no_generation_falls_back_to_local_reads(tmp_path, monkeypatch):
    """Without a generation the run must not attach to (or publish) a shared
    namespace: local reads and one warning naming the setting."""
    from test_cli import _make_fesomc_minimal, _write_config, _write_particles
    from kelp.core.config import load_config
    from kelp.tracking import run_tracking
    for k in ("KELP_SHARED_FIELDS_GENERATION", "SLURM_ARRAY_JOB_ID",
              "SLURM_JOB_ID", "SLURM_STEP_ID"):
        monkeypatch.delenv(k, raising=False)
    nc = _make_fesomc_minimal(tmp_path)
    seeds = tmp_path / "seeds.dat"
    _write_particles(seeds)
    out = tmp_path / "traj.nc"
    cfg_p = _write_config(tmp_path / "cfg.yaml", nc, seeds, out)
    cfg_p.write_text(cfg_p.read_text().replace(
        "numerics:", f"numerics:\n  shared_fields_dir: {tmp_path / 'shm'}"))
    with pytest.warns(RuntimeWarning, match="shared_fields_generation"):
        run_tracking(load_config(cfg_p), output_path=str(out))
    assert not (tmp_path / "shm").exists() or not any(
        (tmp_path / "shm").iterdir())
