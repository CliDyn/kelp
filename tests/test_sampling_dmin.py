"""Scalar sampling on sigma meshes honours ``numerics.Dmin``.

The sampler floored every sigma column at a hard-coded 0.5 m while the
tracking step floors it at ``numerics.Dmin``: on a column shallower than
0.5 m with a smaller configured ``Dmin`` the sampled mid-layer depths sat
on a different column than the particle.  ``sample_field`` now takes
``Dmin`` and the record sampler passes the run's value.
"""
import numpy as np
import pytest

from kelp.core.io import VelocityField
from kelp.core.mesh import Mesh, build_node_to_elems, build_search
from kelp.core.particles import make_state
from kelp.numerics.sampling import sample_field


def _shallow_sigma_mesh(depth=0.3):
    lon = np.array([0.0, 1.0, 0.0, 1.0])
    lat = np.array([0.0, 0.0, 1.0, 1.0])
    en = np.array([[0, 1, 3, 2]], np.int32)
    n2e, cnt = build_node_to_elems(lon.size, en)
    m = Mesh(name="shallow", nodes_lon=lon, nodes_lat=lat,
             depth=np.full(lon.size, depth), elem_nodes=en,
             elem_center_lon=np.array([0.5]), elem_center_lat=np.array([0.5]),
             node_to_elems=n2e, node_to_elems_count=cnt,
             sigma=np.array([1.0, 0.5, 0.0]), vertical_coord="sigma")
    build_search(m)
    return m


def test_mid_layer_sample_uses_the_given_dmin():
    # Column 0.3 m, Dmin 0.1 m: mid-layers at 0.075 m and 0.225 m, so a
    # particle at 0.15 m sits halfway between layer values 1 and 3 -> 2.
    # With the column floored at 0.5 m the midpoints would be 0.125 and
    # 0.375 m and the sample 1.2.
    m = _shallow_sigma_mesh(0.3)
    vals = np.vstack([np.full(4, 1.0), np.full(4, 3.0)])
    got = sample_field(vals, m, np.array([0.4]), np.array([0.6]),
                       np.array([0.15]), eta=np.zeros(4), method="nearest",
                       Dmin=0.1)
    assert got[0] == pytest.approx(2.0, abs=1e-12)


def test_interface_sample_uses_the_given_dmin():
    # Interface field (K, N) with values 0, 10, 20 at sigma 1, 0.5, 0:
    # column 0.3 m -> interfaces at 0, 0.15, 0.3 m; a particle at 0.15 m
    # reads 10.  Floored at 0.5 m it would read 6.
    m = _shallow_sigma_mesh(0.3)
    vals = np.vstack([np.zeros(4), np.full(4, 10.0), np.full(4, 20.0)])
    got = sample_field(vals, m, np.array([0.4]), np.array([0.6]),
                       np.array([0.15]), eta=np.zeros(4), method="nearest",
                       Dmin=0.1)
    assert got[0] == pytest.approx(10.0, abs=1e-12)


def test_default_dmin_is_the_config_default():
    m = _shallow_sigma_mesh(0.3)
    vals = np.vstack([np.full(4, 1.0), np.full(4, 3.0)])
    got = sample_field(vals, m, np.array([0.4]), np.array([0.6]),
                       np.array([0.15]), eta=np.zeros(4), method="nearest")
    assert got[0] == pytest.approx(1.2, abs=1e-12)


def test_record_sampler_passes_the_run_dmin():
    from kelp.tracking import _sample_record_fields
    m = _shallow_sigma_mesh(0.3)
    vals = np.vstack([np.full(4, 1.0), np.full(4, 3.0)])
    vel = VelocityField(u=np.zeros((2, 1)), v=np.zeros((2, 1)),
                        eta=np.zeros(4), w=np.zeros((3, 4)),
                        scalars={"temp": vals})
    st = make_state(np.array([0.4]), np.array([0.6]), np.array([0.15]))
    st = st.replace(nearest_node=np.array([2], dtype=np.int64))
    out = _sample_record_fields(st, m, vel, "nearest", Dmin=0.1)
    assert out["temp"][0] == pytest.approx(2.0, abs=1e-12)


def test_run_samples_on_the_particle_column_with_the_config_dmin(tmp_path):
    """End to end: a 0.3 m column, numerics.Dmin = 0.1 m, the particle parked
    at 0.15 m (no flow): the recorded temperature is on the 0.3 m column
    (mid-layers 0.075 / 0.225 m, layer values 0 / 10 -> 5), not on a column
    floored at 0.5 m (-> 1)."""
    from types import SimpleNamespace
    from kelp.core.config import ModeConfig, NumericsConfig
    from kelp.readers import get_reader
    from kelp.tracking import run_single_realization
    from tests.test_sampling import _make_fesomc_sigma_e2e
    nc = str(tmp_path / "shallow.nc")
    _make_fesomc_sigma_e2e(nc, eta_val=0.0, depth_val=0.3)
    reader = get_reader("fesomc")
    mesh = reader.load_mesh(SimpleNamespace(meshpath=nc))
    ds = reader.open_dataset(nc)
    rec = run_single_realization(
        mesh, ds, reader,
        make_state(np.array([5.0]), np.array([5.0]), np.array([0.15])),
        ModeConfig(Fdrift2D=False, Fdrift3D=True),
        NumericsConfig(dt=3600.0, integrator="euler", sample_interp="nearest",
                       Dmin=0.1),
        start_time_idx=0, n_steps=1, sample_vars=["temp"])
    ds.close()
    arrs = rec.to_arrays()
    assert arrs["z"][0, 0] == pytest.approx(0.15, abs=1e-12)
    assert arrs["temp"][0, 0] == pytest.approx(5.0, abs=1e-9)


def test_interpolated_run_samples_with_the_config_dmin(tmp_path):
    """Same as above in interpolated mode, which samples through the other
    record-sampler call (the payload re-lerped to the record time)."""
    from types import SimpleNamespace
    from kelp.core.config import ModeConfig, NumericsConfig
    from kelp.readers import get_reader
    from kelp.tracking import run_single_realization
    from tests.test_sampling import _make_fesomc_sigma_e2e
    nc = str(tmp_path / "shallow.nc")
    _make_fesomc_sigma_e2e(nc, eta_val=0.0, depth_val=0.3)
    reader = get_reader("fesomc")
    mesh = reader.load_mesh(SimpleNamespace(meshpath=nc))
    ds = reader.open_dataset(nc)
    rec = run_single_realization(
        mesh, ds, reader,
        make_state(np.array([5.0]), np.array([5.0]), np.array([0.15])),
        ModeConfig(Fdrift2D=False, Fdrift3D=True),
        NumericsConfig(dt=1800.0, integrator="euler", sample_interp="nearest",
                       Dmin=0.1),
        start_time_idx=0, n_steps=1, n_time_total=3, sample_vars=["temp"],
        temporal_interp=True, file_times=np.array([0.0, 3600.0, 7200.0]))
    ds.close()
    arrs = rec.to_arrays()
    assert arrs["z"][0, 0] == pytest.approx(0.15, abs=1e-12)
    assert arrs["temp"][0, 0] == pytest.approx(5.0, abs=1e-9)
