"""Phase 5a — real-store contract tests on the CORE2→HEALPix twin
(docs/healpix.md §5; the twin stores are described by the AGENTS.md
shipped next to them).

Gated on KELP_TEST_DATA (skip elsewhere).  FAST by design —
mesh cached per module, runs are a few daily steps on the smallest
(z6) store.  The statistical native-twin validation lives in the @slow
suite (test_healpix_native_twin.py); these tests pin the CONTRACT:
the reader loads the combined twin stores, tracks 2-D and 3-D through
run_tracking machinery, samples T/S, and handles the element-sourced
monthly stores.
"""
from pathlib import Path
from types import SimpleNamespace

import os

import numpy as np
import pytest

# Set KELP_TEST_DATA to the directory of the CORE2 HEALPix twin stores; unset -> skipped.
_DATA = os.environ.get("KELP_TEST_DATA", "")
_Z6 = f"{_DATA}/lagdrift_core2_2005_z6.nc"  # kept on purpose: a real name on disk / in the sibling clone, NOT renamed with the package
_Z7 = f"{_DATA}/lagdrift_core2_2005_z7.nc"
_ELEM_Z6 = f"{_DATA}/lagdrift_core2_2005_elem_monthly_z6.nc"

pytestmark = pytest.mark.skipif(
    not (_DATA and Path(_Z6).exists()),
    reason="CORE2 healpix twin stores not available (set KELP_TEST_DATA)")

from kelp.core.particles import ParticleStatus, make_state
from kelp.readers import get_reader
from kelp.tracking import run_single_realization
from kelp.core.config import NumericsConfig


def _mode(three_d=False, sample_vars=None):
    return SimpleNamespace(
        Fdrift2D=not three_d, Fdrift3D=three_d, time_dir=1.0,
        sigma_layer=None, var_u=None, var_v=None, var_w=None,
        var_eta=None, sample_vars=list(sample_vars or []),
        max_levels=None)


def _mesh_cfg(path):
    return SimpleNamespace(meshpath=path, healpix_zoom=None,
                           crop_bbox=None, levels=None, rotation=None,
                           r_earth=None)


@pytest.fixture(scope="module")
def z6():
    reader = get_reader("healpix")
    mesh = reader.load_mesh(_mesh_cfg(_Z6), _mode())
    ds = reader.open_dataset(_Z6)
    yield reader, mesh, ds
    ds.close()


class TestZ6MeshContract:
    def test_counts_match_agents_md(self, z6):
        _, mesh, _ = z6
        assert mesh.n_nodes == 34988          # wet pixels, nside 64
        assert mesh.n_elements == 33561       # corner dual cells (exact)
        assert mesh.sigma.size == 48          # level_interfaces_m embedded
        assert mesh.sigma[0] == 0.0
        assert mesh.sigma[-1] == 6250.0
        assert mesh.vertical_coord == "zlevel"

    def test_geometry_sane(self, z6):
        _, mesh, _ = z6
        assert np.isfinite(mesh.nodes_lon).all()
        assert np.isfinite(mesh.nodes_lat).all()
        # ocean pixels reach the Arctic but stop at the Antarctic coast
        assert mesh.nodes_lat.max() > 85.0
        assert mesh.nodes_lat.min() < -70.0
        assert 0.0 < mesh.depth.min()
        assert mesh.depth.max() <= 6250.0

    def test_z7_mesh_counts(self):
        if not Path(_Z7).exists():
            pytest.skip("z7 store absent")
        reader = get_reader("healpix")
        mesh7 = reader.load_mesh(_mesh_cfg(_Z7), _mode())
        assert mesh7.n_nodes == 140325        # AGENTS.md pin


class TestZ6Velocities:
    def test_2d_read(self, z6):
        reader, mesh, ds = z6
        vel = reader.read_velocities(ds, 0, mesh, _mode())
        assert vel.uv_layout == "node"
        assert vel.u.shape == (mesh.n_nodes,)
        assert np.isfinite(vel.u).all()       # wet-node NaNs are zeroed
        assert np.abs(vel.u).max() < 3.0      # m/s, physical
        assert np.abs(vel.eta).max() < 3.0    # m

    def test_3d_read(self, z6):
        reader, mesh, ds = z6
        vel = reader.read_velocities(ds, 0, mesh, _mode(three_d=True))
        K = mesh.sigma.size
        assert vel.u.shape == (K - 1, mesh.n_nodes)   # 47 mid-layers
        assert vel.w.shape == (K, mesh.n_nodes)       # 48 interfaces
        # surface interface w is numerically zero (remap preserves the
        # FESOM convention — the Phase-4 structural check)
        assert np.abs(vel.w[0]).max() < 1e-12
        assert np.abs(vel.w).max() < 0.1              # m/s
        # ...and the interior is genuinely NON-zero (review: without a
        # lower bound a silently-zeroed wo passed; measured max 2.5e-3)
        assert np.abs(vel.w[1:]).max() > 1e-6

    def test_scalars_physical(self, z6):
        reader, mesh, ds = z6
        sc = reader.read_scalar_fields(ds, 0, mesh, ["temp", "salt"],
                                       _mode(three_d=True))
        t, s = sc["temp"], sc["salt"]
        assert t.shape == (mesh.sigma.size - 1, mesh.n_nodes)
        # CELSIUS (unlike ng5's Kelvin avg_thetao) and psu
        assert -3.0 < np.nanmin(t) and np.nanmax(t) < 40.0
        assert 0.0 < np.nanmin(s) and np.nanmax(s) < 42.0


class TestZ6Tracking:
    def _seeds(self):
        # North Atlantic patch (CORE2 well-resolved; away from poles/coast)
        rng = np.random.default_rng(42)
        return make_state(rng.uniform(-35.0, -25.0, 8),
                          rng.uniform(40.0, 48.0, 8))

    def test_2d_five_daily_steps(self, z6):
        reader, mesh, ds = z6
        num = NumericsConfig(dt=86400.0, integrator="rk2", output_freq=1,
                             uv_interp="barycentric")
        rec = run_single_realization(
            mesh, ds, reader, self._seeds(), _mode(), num, 0,
            n_steps=5, n_time_total=365)
        arrs = rec.to_arrays()
        assert np.all(arrs["status"][-1] == ParticleStatus.ALIVE)
        # moved, but no further than ~2 m/s * 5 d
        d0 = np.hypot((arrs["lon"][-1] - arrs["lon"][0])
                      * np.cos(np.deg2rad(arrs["lat"][0])),
                      arrs["lat"][-1] - arrs["lat"][0]) * 111.12  # km
        assert np.all(d0 > 0.1)
        assert np.all(d0 < 900.0)

    def test_3d_five_daily_steps(self, z6):
        reader, mesh, ds = z6
        rng = np.random.default_rng(7)
        st = make_state(rng.uniform(-35.0, -25.0, 6),
                        rng.uniform(40.0, 48.0, 6),
                        np.full(6, 100.0))
        num = NumericsConfig(dt=86400.0, integrator="rk2", output_freq=1,
                             uv_interp="barycentric",
                             w_interp="barycentric")
        rec = run_single_realization(
            mesh, ds, reader, st, _mode(three_d=True), num, 0,
            n_steps=5, n_time_total=365)
        arrs = rec.to_arrays()
        assert np.all(arrs["status"][-1] == ParticleStatus.ALIVE)
        z = arrs["z"][-1]
        assert np.all((z > 0.0) & (z < 6250.0))
        # vertical drift over 5 days stays modest at |w| <~ mm/s
        # (measured excursion ~2.2 m) — and vertical motion genuinely
        # HAPPENS (review: z == 100.0 for every particle passed before)
        assert np.all(np.abs(z - 100.0) < 50.0)
        assert np.any(np.abs(z - 100.0) > 1e-3)

    def test_sampling_along_2d_run(self, z6):
        reader, mesh, ds = z6
        num = NumericsConfig(dt=86400.0, integrator="euler",
                             output_freq=1, uv_interp="barycentric")
        rec = run_single_realization(
            mesh, ds, reader, self._seeds(), _mode(sample_vars=["temp",
                                                                "salt"]),
            num, 0, n_steps=3, n_time_total=365,
            sample_vars=["temp", "salt"])
        arrs = rec.to_arrays()
        assert "temp" in arrs and "salt" in arrs
        t = arrs["temp"]
        assert np.isfinite(t).all()
        assert (t > -3.0).all() and (t < 40.0).all()


class TestNodeVsElementSource:
    def test_january_surface_fields_agree(self):
        """The element-sourced monthly remap and the node-sourced daily
        remap describe the SAME ocean: January surface u/v agree
        (measured 2026-08-16: corr 0.996/0.987, rms ~1.1 cm/s, bias
        ~1e-4 — the remap operator commutes with time averaging, so the
        only differences are the FESOM-internal element-vs-node velocity
        representations)."""
        if not Path(_ELEM_Z6).exists():
            pytest.skip("elem-monthly z6 store absent")
        import xarray as xr
        with xr.open_dataset(_Z6) as dn, xr.open_dataset(_ELEM_Z6) as de:
            for var, cmin in (("uo", 0.97), ("vo", 0.95)):
                n = dn[var].isel(lev=0).isel(time=slice(0, 31)) \
                    .mean("time").values
                e = de[var].isel(lev=0).isel(time=0).values
                both = np.isfinite(n) & np.isfinite(e)
                assert both.sum() > 30000
                r = np.corrcoef(n[both], e[both])[0, 1]
                assert r > cmin, (var, r)
                diff = e[both] - n[both]
                assert abs(np.mean(diff)) < 5e-3, var
                assert np.sqrt(np.mean(diff ** 2)) < 0.03, var


class TestElementMonthlyStore:
    def test_opens_and_tracks_2d(self):
        if not Path(_ELEM_Z6).exists():
            pytest.skip("elem-monthly z6 store absent")
        reader = get_reader("healpix")
        # the missing-ssh warning is part of the documented contract —
        # pin the WARNING itself, not just its zero-eta consequence
        with pytest.warns(RuntimeWarning, match="eta = 0"):
            mesh = reader.load_mesh(_mesh_cfg(_ELEM_Z6), _mode())
            ds = reader.open_dataset(_ELEM_Z6)
        try:
            vel = reader.read_velocities(ds, 0, mesh, _mode())
            assert vel.uv_layout == "node"
            assert np.abs(vel.u).max() < 3.0
            # no ssh in these stores: eta must be all zeros (the
            # documented warn-then-zero behaviour)
            assert np.all(vel.eta == 0.0)
            rng = np.random.default_rng(3)
            st = make_state(rng.uniform(-35.0, -25.0, 6),
                            rng.uniform(40.0, 48.0, 6))
            num = NumericsConfig(dt=86400.0, integrator="euler",
                                 output_freq=1, uv_interp="barycentric")
            # Monthly records: discrete mode would read one month per daily
            # step, which the engine refuses since 2026-09-24 — sub-step
            # between the records instead.
            rec = run_single_realization(
                mesh, ds, reader, st, _mode(), num, 0,
                n_steps=2, n_time_total=12, temporal_interp=True)
            arrs = rec.to_arrays()
            assert np.all(arrs["status"][-1] == ParticleStatus.ALIVE)
        finally:
            ds.close()
