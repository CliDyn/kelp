"""Optional end-to-end smoke test for FESOM-C sigma 3D tracking on real data.

Reads ``work/medi/input/medi.nc`` if present, seeds a handful of particles
inside the mesh, runs a few steps with each integrator + ``w_interp="nearest"``,
and checks that particles stay alive and within the local water column.

This test is skipped when the file is missing — which is the case on a
clean checkout.  The file may be overwritten or removed at any time; the
test is purely a runtime sanity check on the real dataset, not a regression
anchor.

Run explicitly::

    source .venv/bin/activate
    pytest tests/test_fesomc_sigma_medi.py -v -s
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kelp.numerics.integrators import euler_step, rk2_step, rk4_step
from kelp.core.particles import make_state, ParticleStatus
from kelp.readers import get_reader


MEDI_PATH = Path(__file__).resolve().parents[1] / "work" / "medi" / "input" / "medi.nc"


pytestmark = pytest.mark.skipif(
    not MEDI_PATH.exists(),
    reason=f"Optional dataset not present: {MEDI_PATH}",
)


def _seed_particles_inside(mesh, n=5, seed=0):
    """Pick ``n`` random lon/lat points near mesh centroids and mid-depth."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(mesh.n_elements, size=n, replace=False)
    lon = mesh.elem_center_lon[idx].astype(np.float64).copy()
    lat = mesh.elem_center_lat[idx].astype(np.float64).copy()
    # Start at half the bottom depth of a nearby node
    from kelp.core.mesh import find_nearest_nodes
    nn = find_nearest_nodes(mesh, lon, lat)
    z = 0.5 * mesh.depth[nn]
    z = np.clip(z, 1.0, None)
    return make_state(lon=lon, lat=lat, z=z)


def _load_mesh_and_dataset():
    reader = get_reader("fesomc")
    mesh = reader.load_mesh(SimpleNamespace(meshpath=str(MEDI_PATH)))
    ds = reader.open_dataset(str(MEDI_PATH))
    return reader, mesh, ds


@pytest.fixture(scope="module")
def medi_setup():
    reader, mesh, ds = _load_mesh_and_dataset()
    yield reader, mesh, ds
    ds.close()


@pytest.fixture
def medi_vel(medi_setup):
    reader, mesh, ds = medi_setup
    mode = SimpleNamespace(
        Fdrift2D=False, Fdrift3D=True, Vpuls=False,
        var_u=None, var_v=None, var_w=None, var_eta=None,
        sigma_layer=None, sigma_mean=False, max_levels=None,
    )
    return reader.read_velocities(ds, 0, mesh, mode)


class TestMediSigmaTracking:
    """One step per integrator on the real medi.nc grid."""

    def test_mesh_has_sigma_contract(self, medi_setup):
        _, mesh, _ = medi_setup
        assert mesh.vertical_coord == "sigma"
        assert mesh.sigma.shape == (11,)
        assert mesh.sigma[0] == pytest.approx(1.0)
        assert mesh.sigma[-1] == pytest.approx(0.0)

    def test_velocity_field_shapes(self, medi_setup, medi_vel):
        _, mesh, _ = medi_setup
        K = mesh.sigma.shape[0]
        assert medi_vel.u.shape == (K - 1, mesh.n_elements)
        assert medi_vel.v.shape == (K - 1, mesh.n_elements)
        assert medi_vel.w.shape == (K, mesh.n_nodes)

    @pytest.mark.parametrize("integrator", [euler_step, rk2_step, rk4_step])
    def test_one_step_keeps_particles_alive_and_bounded(
        self, medi_setup, medi_vel, integrator,
    ):
        """Each integrator moves particles through one step; survivors
        must stay within their local water column.

        Some seeded particles may land near coastal boundaries and go
        OUTSIDE the mesh — that is a 2D horizontal concern unrelated to
        sigma. The test only asserts that (a) at least one particle stays
        alive, and (b) every alive particle has a physically valid depth.
        """
        _, mesh, _ = medi_setup
        state = _seed_particles_inside(
            mesh, n=20, seed=abs(hash(integrator.__name__)),
        )
        from kelp.core.mesh import find_nearest_nodes
        nn_in = find_nearest_nodes(mesh, state.lon, state.lat)
        state = state.replace(nearest_node=nn_in)

        new = integrator(
            state, medi_vel, mesh, dt=600.0, time_dir=1.0,
            w_interp="nearest",
        )

        alive = new.status == ParticleStatus.ALIVE
        assert alive.any(), f"{integrator.__name__}: no particles survived"

        nn_out = find_nearest_nodes(mesh, new.lon[alive], new.lat[alive])
        bottom = mesh.depth[nn_out]
        assert np.all(new.z[alive] >= 0.0), (
            f"{integrator.__name__}: alive particle above surface: "
            f"z={new.z[alive]}"
        )
        assert np.all(new.z[alive] <= bottom + 1e-6), (
            f"{integrator.__name__}: alive particle below bottom: "
            f"z={new.z[alive]}, bottom={bottom}"
        )
        # At least one alive particle must have actually moved (u/v != 0)
        moved = np.abs(new.lon[alive] - state.lon[alive]).max() > 0
        moved |= np.abs(new.lat[alive] - state.lat[alive]).max() > 0
        assert moved, f"{integrator.__name__}: no horizontal motion observed"
