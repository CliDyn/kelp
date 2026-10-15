"""
Tests for the visualization module using synthetic data.
"""
import pytest
import numpy as np
from io import BytesIO

from kelp.core.mesh import Mesh
from kelp.visualization.core import TrajectoryDataset
from kelp.visualization.plot_2d import plot_mesh, plot_trajectories
from kelp.visualization.plot_3d import plot_mesh_3d, plot_trajectories_3d
from kelp.visualization.stats import compute_presence_probability
from kelp.visualization.animation import TrajectoryAnimator
from kelp.core.particles import ParticleStatus


@pytest.fixture
def mock_mesh():
    """Create a simple 3x3 node grid with 4 quads."""
    x = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=float)
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=float)
    depth = np.full_like(x, 100.0)
    
    # 4 elements: 2 quads, 2 triangles (FESOM-C style with -1 and repeating node)
    elem_nodes = np.array([
        [0, 1, 4, 3],    # quad
        [1, 2, 5, -1],   # triangle padded with -1
        [3, 4, 7, 3],    # triangle repeating the last node
        [4, 5, 8, 7]     # quad
    ], dtype=np.int32)
    
    return Mesh(
        name="mock",
        nodes_lon=x,
        nodes_lat=y,
        depth=depth,
        elem_nodes=elem_nodes,
        elem_center_lon=np.array([0.5, 1.5, 0.5, 1.5]),
        elem_center_lat=np.array([0.5, 0.5, 1.5, 1.5]),
        node_to_elems=np.zeros((9, 4), dtype=np.int32),  # Mocked
        node_to_elems_count=np.ones((9,), dtype=np.int32),
        sigma=np.array([])
    )


@pytest.fixture
def mock_trajectory_dataset(mock_mesh):
    """Create a mock TrajectoryDataset of 2 particles passing over 5 timesteps."""
    n_particles = 2
    n_steps = 5
    
    lon = np.array([
        [0.0, 0.5, 1.0, 1.5, 2.0],
        [2.0, 1.5, 1.0, 0.5, 0.0]
    ])
    lat = np.array([
        [0.0, 0.5, 1.0, 1.5, 2.0],
        [2.0, 1.5, 1.0, 0.5, 0.0]
    ])
    z = np.full((n_particles, n_steps), 5.0)
    # Create status array using ParticleStatus
    status = np.full((n_particles, n_steps), ParticleStatus.ALIVE, dtype=np.int32)
    # Give one particle a different final status
    status[0, -1] = ParticleStatus.BEACHED
    time_arr = np.linspace(0, 4000, n_steps)
    
    data = {
        "lon": lon,
        "lat": lat,
        "z": z,
        "status": status,
        "time": time_arr
    }
    
    return TrajectoryDataset(mock_mesh, data)


def test_trajectory_dataset(mock_trajectory_dataset):
    ds = mock_trajectory_dataset
    assert ds.n_particles == 2
    assert ds.n_steps == 5
    
    # Test step extraction
    step_vec = ds.get_step(2)
    assert np.all(step_vec["lon"] == np.array([1.0, 1.0]))
    
    # Test bounding box
    bbox = ds.bounding_box
    assert len(bbox) == 4
    
    # Filter status testing
    beached_ds = ds.filter_status(ParticleStatus.BEACHED)
    assert beached_ds.n_particles == 1
    
    alive_ds = ds.filter_status(ParticleStatus.ALIVE)
    assert alive_ds.n_particles == 1
    
    # Test array transposition from (T, P)
    data_tp = {
        "lon": ds.lon.T.copy(),
        "lat": ds.lat.T.copy(),
        "z": ds.z.T.copy(),
        "status": ds.status.T.copy(),
        "time": ds.time.copy()
    }
    ds_tp = TrajectoryDataset(ds.mesh, data_tp)
    assert ds_tp.lon.shape == (2, 5)  # Automatically transposed to (P, T)

    # Test shape exceptions
    with pytest.raises(ValueError):
        TrajectoryDataset(ds.mesh, {"lon": np.zeros((2, 5)), "lat": np.zeros((2, 4))})
    with pytest.raises(ValueError):
        TrajectoryDataset(ds.mesh, {"lon": np.zeros((2, 5)), "lat": np.zeros((2, 5)), "z": np.zeros((2, 6))})


def test_plot_2d_execution(mock_trajectory_dataset):
    """Ensure plotting functions run without errors and can save to buffer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    ds = mock_trajectory_dataset
    mesh = ds.mesh
    
    fig, ax = plt.subplots()
    ax = plot_mesh(mesh, ax=ax, projection=None)  # Cartopy is skipped for raw ax
    ax = plot_trajectories(ds, ax=ax, projection=None)
    
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    assert len(buf.read()) > 0
    
    # Test color_by functionality and time coloring
    fig2, ax2 = plt.subplots()
    ax2 = plot_trajectories(ds, ax=ax2, projection=None, color_by="depth")
    # Test safety against missing time mapping
    ds_notime = TrajectoryDataset(ds.mesh, {"lon": ds.lon, "lat": ds.lat})
    ax2 = plot_trajectories(ds_notime, ax=ax2, projection=None, color_by="time")
    
    buf2 = BytesIO()
    fig2.savefig(buf2, format="png")
    buf2.seek(0)
    assert len(buf2.read()) > 0
    
    plt.close("all")

def test_plot_3d_execution(mock_trajectory_dataset):
    """Ensure 3D plotting functions run without errors."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    ds = mock_trajectory_dataset
    mesh = ds.mesh
    
    
    # Render with show=False to prevent blocking
    # This also covers the Matplotlib fallback logic if pyvista isn't tested
    plotter = plot_mesh_3d(mesh, show=False)
    plotter = plot_trajectories_3d(ds, plotter=plotter, show=False)
    
    # Test 0 particles edge case
    ds_empty = TrajectoryDataset(ds.mesh, {"lon": np.empty((0, 5)), "lat": np.empty((0, 5))})
    plot_trajectories_3d(ds_empty, plotter=plotter, show=False)

    assert plotter is not None
    plt.close("all")


def _rotated_mock_mesh():
    """A mock mesh whose stored coords are ROTATED (simulating a
    reader-rotated mesh): the geographic coords are a 3x3 patch near the
    pole, rotated into the FESOM2 computational frame, with
    ``rotation_angles`` set so the display path knows to un-rotate."""
    from kelp.core.rotation import FESOM2_ANGLES, euler_matrix, rotate_lonlat

    # geographic patch (the "truth" the user expects to see plotted)
    gx = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]) + 5.0
    gy = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]) + 80.0
    R = euler_matrix(*FESOM2_ANGLES)
    rx, ry = rotate_lonlat(gx, gy, R)          # what the reader would store

    elem_nodes = np.array([
        [0, 1, 4, 3],
        [1, 2, 5, -1],
        [3, 4, 7, 3],
        [4, 5, 8, 7],
    ], dtype=np.int32)
    mesh = Mesh(
        name="rot",
        nodes_lon=rx,
        nodes_lat=ry,
        depth=np.full(9, 100.0),
        elem_nodes=elem_nodes,
        elem_center_lon=rx[:4].copy(),
        elem_center_lat=ry[:4].copy(),
        node_to_elems=np.zeros((9, 4), dtype=np.int32),
        node_to_elems_count=np.ones((9,), dtype=np.int32),
        sigma=np.array([]),
        rotation_angles=FESOM2_ANGLES,
    )
    return mesh, gx, gy, rx, ry


def test_plot_mesh_unrotates_rotated_mesh():
    """plot_mesh on a rotated mesh must draw the GEOGRAPHIC coordinates
    (trajectory output is geographic), not the stored rotated ones."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mesh, gx, gy, rx, ry = _rotated_mock_mesh()
    # the rotated frame is well separated from the geographic patch, so a
    # failure to un-rotate is unambiguous.
    assert abs(rx.min() - gx.min()) > 1.0 or abs(ry.min() - gy.min()) > 1.0

    fig, ax = plt.subplots()
    plot_mesh(mesh, ax=ax, projection=None)

    # Gather every vertex actually drawn (PolyCollection paths).
    verts = np.vstack([
        p.vertices for coll in ax.collections for p in coll.get_paths()
    ])
    drawn_lon = verts[:, 0]
    drawn_lat = verts[:, 1]

    # plotted extent matches the GEOGRAPHIC patch, not the rotated one.
    np.testing.assert_allclose(drawn_lon.min(), gx.min(), atol=1e-6)
    np.testing.assert_allclose(drawn_lon.max(), gx.max(), atol=1e-6)
    np.testing.assert_allclose(drawn_lat.min(), gy.min(), atol=1e-6)
    np.testing.assert_allclose(drawn_lat.max(), gy.max(), atol=1e-6)
    plt.close("all")

def test_plot_2d_edge_cases(mock_mesh):
    """Test 0 particles and 1 timestep logic."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    # 0 particles
    ds_empty = TrajectoryDataset(mock_mesh, {"lon": np.empty((0, 5)), "lat": np.empty((0, 5))})
    fig, ax = plt.subplots()
    plot_trajectories(ds_empty, ax=ax)
    
    # 1 timestep
    ds_1step = TrajectoryDataset(mock_mesh, {
        "lon": np.zeros((2, 1)), 
        "lat": np.zeros((2, 1)),
        "z": np.zeros((2, 1)),
        "time": np.zeros((1,))
    })
    fig2, ax2 = plt.subplots()
    plot_trajectories(ds_1step, ax=ax2, color_by="time")
    
    plt.close("all")



def test_compute_presence_probability(mock_trajectory_dataset):
    from kelp.visualization.stats import compute_presence_probability, compute_status_ratios
    ds = mock_trajectory_dataset
    
    lon_bins = np.linspace(0, 5, 10)
    lat_bins = np.linspace(0, 5, 10)
    
    prob = compute_presence_probability(ds, lon_bins, lat_bins)
    
    # Output should be (N-1, M-1)
    assert prob.shape == (9, 9)
    # Since particles move in [0, 5], there should be some non-zero probability
    assert np.sum(prob) > 0
    
def test_compute_status_ratios(mock_trajectory_dataset):
    from kelp.visualization.stats import compute_status_ratios
    ratios = compute_status_ratios(mock_trajectory_dataset)
    assert ratios[ParticleStatus.ALIVE] == 0.5
    assert ratios[ParticleStatus.BEACHED] == 0.5


def test_trajectory_animator(mock_trajectory_dataset):
    import matplotlib
    matplotlib.use("Agg")
    
    animator = TrajectoryAnimator(mock_trajectory_dataset, projection=None)
    animator.animate()
    assert animator.anim is not None
    # Cannot easily mock the ffmpeg save output without file writing,
    # so we just test the generation of the internal FuncAnimation object.
