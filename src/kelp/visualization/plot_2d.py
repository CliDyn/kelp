"""
2D Visualization routines using Matplotlib and Cartopy for kelp.

Publication-quality map plots and per-particle diagnostic panels.
"""
from typing import Optional, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.tri as mtri
    from matplotlib.axes import Axes
    from matplotlib.collections import LineCollection, PolyCollection
except ImportError:
    plt, cm, mcolors, mtri, Axes, LineCollection, PolyCollection = (
        None, None, None, None, None, None, None,
    )

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:
    ccrs, cfeature = None, None

from kelp.core.mesh import Mesh
from kelp.core.geometry import METERS_PER_DEG_LAT, DEG2RAD
from .core import TrajectoryDataset, display_coords

# Publication-quality defaults
_PUB_RCPARAMS = {
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.linewidth': 0.8,
}


def _setup_map_axes(
    ax: Optional['Axes'],
    bbox: Tuple[float, float, float, float],
    projection: Optional['ccrs.Projection'] = None,
    figsize: Tuple[float, float] = (12, 9),
    coastline_resolution: str = '50m',
) -> Tuple['Axes', Optional['ccrs.Projection']]:
    """Create or configure map axes with publication-quality styling.

    Parameters
    ----------
    ax : Axes or None
        Existing axes, or None to create new figure.
    bbox : tuple
        (lon_min, lon_max, lat_min, lat_max) extent.
    projection : cartopy projection, optional
        Map projection.  Defaults to Mercator if cartopy available.
    figsize : tuple
        Figure size if creating new axes.
    coastline_resolution : str
        Cartopy coastline resolution ('110m', '50m', '10m').

    Returns
    -------
    ax : Axes
    transform : CRS or None
        PlateCarree transform for plotting data, or None.
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting.")

    transform = None

    if ax is None:
        plt.rcParams.update(_PUB_RCPARAMS)
        if ccrs is not None:
            if projection is None:
                projection = ccrs.Mercator()
            fig, ax = plt.subplots(
                figsize=figsize,
                subplot_kw=dict(projection=projection),
                constrained_layout=True,
            )
        else:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    if ccrs is not None and hasattr(ax, 'coastlines'):
        transform = ccrs.PlateCarree()
        ax.set_extent(bbox, crs=transform)
        ax.coastlines(resolution=coastline_resolution, linewidth=0.8,
                      color='#333333')
        ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', edgecolor='none')
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray',
                          alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
    else:
        ax.set_xlim(bbox[0], bbox[1])
        ax.set_ylim(bbox[2], bbox[3])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    return ax, transform


def plot_mesh(
    mesh: Mesh,
    ax: Optional['Axes'] = None,
    color: str = 'grey',
    edgecolor: str = 'black',
    linewidth: float = 0.1,
    alpha: float = 0.5,
    projection: Optional['ccrs.Projection'] = None,
) -> 'Axes':
    """Plot the unstructured FESOM mesh on a 2D map.

    Parameters
    ----------
    mesh : Mesh
        The unstructured mesh to plot.
    ax : matplotlib.axes.Axes, optional
        Target axes.  If None, a new figure is created.
    color : str
        Face color for elements.
    edgecolor : str
        Edge color for elements.
    linewidth : float
        Width of element edges.
    alpha : float
        Transparency.
    projection : cartopy.crs.Projection, optional
        Map projection (default: Mercator if cartopy available).

    Returns
    -------
    matplotlib.axes.Axes
    """
    if plt is None:
        raise ImportError("matplotlib is required to plot meshes.")

    # Tracking may run in a rotated frame, but trajectory output (and hence
    # the map overlay) is geographic — un-rotate the mesh for display.
    nodes_lon, nodes_lat = display_coords(mesh)

    minlon, maxlon = float(np.min(nodes_lon)), float(np.max(nodes_lon))
    minlat, maxlat = float(np.min(nodes_lat)), float(np.max(nodes_lat))
    bbox = (minlon, maxlon, minlat, maxlat)

    ax, transform = _setup_map_axes(ax, bbox, projection)

    # Build PolyCollection from elem_nodes
    verts = []
    for face in mesh.elem_nodes:
        face_valid = face[face != -1]
        # FESOM-C: triangles encoded as quad with repeated first node
        if len(face_valid) == 4 and face_valid[3] == face_valid[0]:
            face_valid = face_valid[:3]
        if len(face_valid) >= 3:
            poly = np.column_stack((
                nodes_lon[face_valid],
                nodes_lat[face_valid],
            ))
            verts.append(poly)

    pc = PolyCollection(
        verts, facecolors=color, edgecolors=edgecolor,
        linewidths=linewidth, alpha=alpha, zorder=1,
    )
    if transform is not None:
        pc.set_transform(transform)
    ax.add_collection(pc)

    return ax


def _face_vertex_lists(mesh: Mesh):
    """Per-element compacted vertex index lists, with the element index.

    Same compaction rule as plot_mesh (FESOM-C ``-1`` pad dropped,
    quad-encoded triangle ``[a,b,c,a]`` truncated) — but ALSO skips
    degenerate rows with fewer than 3 compacted vertices (all ``-1``,
    collapsed elements from ``_sanitize_connectivity``), returning
    ``[(elem_id, verts), ...]`` so scalar arrays stay aligned with the
    surviving faces.
    """
    faces = []
    for k, face in enumerate(mesh.elem_nodes):
        face_valid = face[face != -1]
        if len(face_valid) == 4 and face_valid[3] == face_valid[0]:
            face_valid = face_valid[:3]
        if len(face_valid) >= 3:
            faces.append((k, face_valid))
    return faces


def plot_field(
    mesh: Mesh,
    values: np.ndarray,
    layout: str = 'auto',
    ax: Optional['Axes'] = None,
    cmap: str = 'viridis',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    norm: Optional['mcolors.Normalize'] = None,
    shading: str = 'auto',
    add_colorbar: bool = True,
    cbar_label: Optional[str] = None,
    alpha: float = 1.0,
    edgecolor: str = 'none',
    linewidth: float = 0.0,
    projection: Optional['ccrs.Projection'] = None,
) -> 'Axes':
    """Plot a scalar field on the mesh as color-coded faces or nodes.

    ``values`` may live on NODES (``(N,)``: temperature, ssh, any
    node-based scalar) or on ELEMENTS (``(E,)``: element velocity
    magnitude, presence density from
    :func:`~kelp.visualization.stats.compute_presence_density_mesh`).
    Node scalars on an all-triangle mesh render smoothly
    (``tripcolor`` with Gouraud shading); node scalars on a mesh with
    quads fall back to flat faces colored by the vertex mean; element
    scalars are always flat faces.  Masked entries and NaNs are not
    drawn (a masked NODE value masks the faces touching it in Gouraud
    mode).

    Display frame: a COMPACT mesh (longitudes fit one contiguous 350°
    branch) is unwrapped around its own branch, so a dateline-straddling
    regional mesh draws contiguously (bbox stays tight); on a global
    mesh, faces whose displayed longitude extent exceeds 180° — the seam
    column and the immediate polar ring — are masked from display (a raw
    polygon there would smear across the whole map).  A rotated-frame
    mesh draws at geographic positions (``display_coords``), matching
    the trajectory overlay.

    Parameters
    ----------
    mesh : Mesh
        The mesh whose faces/nodes carry ``values``.
    values : np.ndarray
        Scalar per node ``(N,)`` or per element ``(E,)``; 1-D only.
        Masked arrays keep their mask.
    layout : str
        ``'auto'`` (resolve by length; ambiguous when ``N == E`` —
        then pass explicitly), ``'node'`` or ``'element'``.
    ax : Axes, optional
        Existing axes; None creates a new map figure.
    cmap, vmin, vmax : colormap controls.
    norm : matplotlib.colors.Normalize, optional
        Custom normalization (e.g. ``LogNorm`` for per-area particle
        densities spanning decades).  Mutually exclusive with
        ``vmin``/``vmax``.
    shading : str
        ``'auto'`` (Gouraud where possible, else flat), ``'gouraud'``
        (node values on an all-triangle mesh only — raises otherwise)
        or ``'flat'``.
    add_colorbar : bool
        Attach a colorbar (labelled with ``cbar_label``).
    alpha, edgecolor, linewidth : face styling (flat modes).
    projection : cartopy projection, optional

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ValueError
        Non-1-D ``values``; a size not matching the layout; unknown
        ``layout``/``shading``; ``shading='gouraud'`` off the
        node+all-triangle case; ``norm`` combined with ``vmin``/``vmax``.
    ImportError
        matplotlib is not installed.
    """
    if plt is None:
        raise ImportError("matplotlib is required to plot fields.")

    values = np.ma.masked_invalid(np.ma.asarray(values, dtype=float))
    if values.ndim != 1:
        raise ValueError(
            f"plot_field: values must be 1-D, got shape {values.shape}"
        )
    n_nodes = mesh.n_nodes
    n_elems = mesh.n_elements
    if layout == 'auto':
        if values.size == n_nodes and values.size == n_elems:
            raise ValueError(
                "plot_field: mesh has n_nodes == n_elements "
                f"({n_nodes}); pass layout='node' or layout='element'"
            )
        if values.size == n_nodes:
            layout = 'node'
        elif values.size == n_elems:
            layout = 'element'
        else:
            raise ValueError(
                f"plot_field: values has {values.size} entries, expected "
                f"{n_nodes} (nodes) or {n_elems} (elements)"
            )
    elif layout not in ('node', 'element'):
        raise ValueError(
            f"plot_field: layout must be 'auto'|'node'|'element', "
            f"got {layout!r}"
        )
    expected = n_nodes if layout == 'node' else n_elems
    if values.size != expected:
        raise ValueError(
            f"plot_field: values has {values.size} entries for "
            f"layout={layout!r}, expected {expected}"
        )
    if shading not in ('auto', 'gouraud', 'flat'):
        raise ValueError(
            f"plot_field: shading must be 'auto'|'gouraud'|'flat', "
            f"got {shading!r}"
        )
    if norm is not None and (vmin is not None or vmax is not None):
        raise ValueError("plot_field: pass either norm or vmin/vmax, "
                         "not both")

    nodes_lon, nodes_lat = display_coords(mesh)

    # Display branch: a COMPACT mesh unwraps around its own contiguous
    # branch (ref = first node), so dateline-straddling regional meshes
    # draw contiguously and get a tight bbox.  A global mesh (no
    # contiguous branch) keeps raw [-180, 180) coordinates and masks the
    # over-wide faces below.
    ref = float(nodes_lon[0]) if n_nodes else 0.0
    dlon = ((nodes_lon - ref + 180.0) % 360.0) - 180.0
    if n_nodes and (dlon.max() - dlon.min()) <= 350.0:
        lon_disp = ref + dlon
    else:
        lon_disp = nodes_lon

    faces = _face_vertex_lists(mesh)
    all_tri = all(len(f) == 3 for _, f in faces)
    want_gouraud = (layout == 'node' and shading in ('auto', 'gouraud'))
    if shading == 'gouraud' and (layout != 'node' or not all_tri):
        raise ValueError(
            "plot_field: shading='gouraud' requires node values on an "
            "all-triangle mesh"
        )
    use_gouraud = want_gouraud and all_tri

    # Vectorized over-wide-face mask (O(E) numpy; the padded -1 slots are
    # mapped to a real vertex so they cannot affect the span).
    en = mesh.elem_nodes
    vmask = en >= 0
    safe = np.where(vmask, en, 0)
    lf = lon_disp[safe]
    lf_min = np.where(vmask, lf, np.inf).min(axis=1)
    lf_max = np.where(vmask, lf, -np.inf).max(axis=1)
    with np.errstate(invalid='ignore'):
        wide = (lf_max - lf_min) > 180.0

    minlon = float(np.min(lon_disp)) if n_nodes else 0.0
    maxlon = float(np.max(lon_disp)) if n_nodes else 1.0
    minlat = float(np.min(nodes_lat)) if n_nodes else 0.0
    maxlat = float(np.max(nodes_lat)) if n_nodes else 1.0
    bbox = (minlon, maxlon, minlat, maxlat)
    ax, transform = _setup_map_axes(ax, bbox, projection)
    tkw = {'transform': transform} if transform is not None else {}

    if use_gouraud:
        tri_idx = np.array([f for _, f in faces], dtype=np.int64)
        triang = mtri.Triangulation(lon_disp, nodes_lat, tri_idx)
        tri_wide = np.array([bool(wide[k]) for k, _ in faces])
        if tri_wide.any():
            triang.set_mask(tri_wide)
        mappable = ax.tripcolor(
            triang, values, shading='gouraud', cmap=cmap,
            vmin=vmin, vmax=vmax, norm=norm, alpha=alpha, **tkw,
        )
    else:
        verts = []
        face_vals = []
        for k, f in faces:
            if wide[k]:
                continue
            verts.append(np.column_stack((lon_disp[f], nodes_lat[f])))
            if layout == 'element':
                face_vals.append(values[k])
            else:
                fv = values[f]
                # ANY masked/NaN vertex masks the whole face — the same
                # semantics matplotlib's gouraud tripcolor applies to
                # triangles touching a masked node.
                if np.ma.is_masked(fv):
                    face_vals.append(np.nan)
                else:
                    face_vals.append(float(fv.mean()))
        pc = PolyCollection(
            verts, cmap=cmap, alpha=alpha,
            edgecolors=edgecolor, linewidths=linewidth, zorder=1,
        )
        pc.set_array(np.ma.masked_invalid(np.ma.asarray(face_vals)))
        if norm is not None:
            pc.set_norm(norm)
        else:
            pc.set_clim(vmin, vmax)
        if transform is not None:
            pc.set_transform(transform)
        ax.add_collection(pc)
        ax.autoscale_view()
        mappable = pc

    if add_colorbar:
        cb = ax.figure.colorbar(mappable, ax=ax, shrink=0.85, pad=0.02)
        if cbar_label:
            cb.set_label(cbar_label)

    return ax




def plot_mesh_nodes(
    mesh: Mesh,
    ax: 'Axes',
    bbox: Optional[Tuple[float, float, float, float]] = None,
    markersize: float = 0.5,
    color: str = 'grey',
    alpha: float = 0.3,
) -> None:
    """Plot mesh nodes as dots (fast, no polygons).

    Parameters
    ----------
    mesh : Mesh
    ax : Axes
    bbox : tuple, optional
        (lon_min, lon_max, lat_min, lat_max) to filter nodes.
    markersize : float
    color : str
    alpha : float
    """
    # Display in the geographic frame (un-rotated for a rotated mesh) to
    # match the trajectory overlay.
    nodes_lon, nodes_lat = display_coords(mesh)
    if bbox is not None:
        mask = (
            (nodes_lon >= bbox[0]) & (nodes_lon <= bbox[1]) &
            (nodes_lat >= bbox[2]) & (nodes_lat <= bbox[3])
        )
        lons, lats = nodes_lon[mask], nodes_lat[mask]
    else:
        lons, lats = nodes_lon, nodes_lat

    transform = None
    if ccrs is not None and hasattr(ax, 'coastlines'):
        transform = ccrs.PlateCarree()

    ax.plot(lons, lats, '.', color=color, markersize=markersize,
            alpha=alpha, zorder=1,
            **(dict(transform=transform) if transform else {}))


def plot_trajectories(
    ds: TrajectoryDataset,
    ax: Optional['Axes'] = None,
    step_modulo: int = 1,
    color_by: Optional[str] = None,
    linewidth: float = 0.6,
    alpha: float = 0.8,
    cmap: str = "plasma",
    projection: Optional['ccrs.Projection'] = None,
    show_endpoints: bool = True,
    show_mesh_nodes: bool = False,
    title: Optional[str] = None,
    margin: Optional[float] = None,
    figsize: Tuple[float, float] = (12, 9),
    coastline_resolution: str = '50m',
) -> 'Axes':
    """Plot particle trajectories on a publication-quality map.

    Uses LineCollection for time/depth coloring (smooth lines, not scatter).

    Parameters
    ----------
    ds : TrajectoryDataset
        Trajectory data (P, T).
    ax : Axes, optional
        Existing axes.  If None, creates new figure.
    step_modulo : int
        Plot every N-th step.
    color_by : str, optional
        ``"time"`` or ``"depth"`` for color-mapped lines.
        None for solid blue lines.
    linewidth : float
    alpha : float
    cmap : str
        Colormap name for color_by mode.
    projection : cartopy projection, optional
    show_endpoints : bool
        Draw start (green circle) and end (red square) markers.
    show_mesh_nodes : bool
        Overlay mesh nodes as grey dots.
    title : str, optional
        Plot title.
    margin : float, optional
        Margin around trajectory bbox in degrees.  Auto if None.
    figsize : tuple
    coastline_resolution : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    if plt is None:
        raise ImportError("matplotlib is required to plot trajectories.")
    if ds.n_particles == 0:
        return ax

    # Compute bbox
    min_x, max_x, min_y, max_y = ds.bounding_box
    span = max(max_x - min_x, max_y - min_y)
    if margin is None:
        margin = max(span * 0.3, 0.5)
    bbox = (min_x - margin, max_x + margin, min_y - margin, max_y + margin)

    ax, transform = _setup_map_axes(
        ax, bbox, projection, figsize, coastline_resolution,
    )
    plot_kw = dict(transform=transform) if transform else {}

    # Mesh nodes
    if show_mesh_nodes and ds.mesh is not None:
        plot_mesh_nodes(ds.mesh, ax, bbox=bbox)

    # Build color data
    if color_by == "time":
        if ds.time is not None:
            c_arr = ds.time[::step_modulo]
            c_label = "Time"
        else:
            c_arr = np.arange(0, ds.n_steps, step_modulo, dtype=float)
            c_label = "Step"
    elif color_by == "depth":
        c_label = "Depth [m]"
    else:
        c_arr = None
        c_label = None

    # Plot trajectories
    if color_by is not None:
        # LineCollection for smooth color-mapped lines
        colormap = plt.get_cmap(cmap)

        for p in range(ds.n_particles):
            lon_s = ds.lon[p, ::step_modulo]
            lat_s = ds.lat[p, ::step_modulo]

            if color_by == "depth" and ds.z is not None:
                c_seg = ds.z[p, ::step_modulo]
            elif c_arr is not None:
                c_seg = c_arr
            else:
                c_seg = np.arange(len(lon_s), dtype=float)

            points = np.column_stack([lon_s, lat_s]).reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            norm = mcolors.Normalize(
                vmin=float(np.nanmin(c_seg)),
                vmax=float(np.nanmax(c_seg)),
            )
            # Use global norm for consistency across particles
            if color_by == "time" and ds.time is not None:
                norm = mcolors.Normalize(
                    vmin=float(np.nanmin(ds.time)),
                    vmax=float(np.nanmax(ds.time)),
                )
            elif color_by == "depth" and ds.z is not None:
                norm = mcolors.Normalize(
                    vmin=float(np.nanmin(ds.z)),
                    vmax=float(np.nanmax(ds.z)),
                )

            lc = LineCollection(
                segments, cmap=colormap, norm=norm,
                linewidths=linewidth, alpha=alpha, zorder=3,
            )
            lc.set_array(c_seg[:-1])
            if transform is not None:
                lc.set_transform(transform)
            ax.add_collection(lc)

        # Colorbar
        sm = cm.ScalarMappable(norm=norm, cmap=colormap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal',
                            pad=0.06, shrink=0.6)
        cbar.set_label(c_label, fontsize=11)
        cbar.ax.tick_params(labelsize=9)
    else:
        # Simple solid lines
        for p in range(ds.n_particles):
            ax.plot(
                ds.lon[p, ::step_modulo], ds.lat[p, ::step_modulo],
                '-', color='#2c3e50', linewidth=linewidth, alpha=alpha,
                zorder=3, **plot_kw,
            )

    # Start / end markers
    if show_endpoints:
        ax.plot(ds.lon[:, 0], ds.lat[:, 0], 'o', color='#2ecc71',
                markersize=5, markeredgecolor='white', markeredgewidth=0.5,
                label='start', zorder=5, **plot_kw)
        ax.plot(ds.lon[:, -1], ds.lat[:, -1], 's', color='#e74c3c',
                markersize=5, markeredgecolor='white', markeredgewidth=0.5,
                label='end', zorder=5, **plot_kw)
        ax.legend(loc='upper left', framealpha=0.9, fontsize=9)

    if title:
        ax.set_title(title, fontweight='bold', pad=12)

    return ax


def plot_diagnostics(
    ds: TrajectoryDataset,
    dt: float = 3600.0,
    max_particles: int = 20,
    figsize_per_row: float = 2.5,
) -> 'plt.Figure':
    """Plot per-particle diagnostic panels: lon(t), lat(t), speed(t).

    Parameters
    ----------
    ds : TrajectoryDataset
        Trajectory data.
    dt : float
        Time step in seconds (for speed computation and time axis).
    max_particles : int
        Maximum number of particles to show.
    figsize_per_row : float
        Height per particle row in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if plt is None:
        raise ImportError("matplotlib is required for diagnostics.")

    plt.rcParams.update(_PUB_RCPARAMS)

    n_show = min(ds.n_particles, max_particles)
    if n_show == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No particles', ha='center', va='center')
        return fig

    # Time axis in hours
    time_hrs = np.arange(ds.n_steps) * dt / 3600.0

    # Compute speed from displacements
    dlon = np.diff(ds.lon, axis=1)
    dlat = np.diff(ds.lat, axis=1)
    cos_lat = np.cos(ds.lat[:, :-1] * DEG2RAD)
    dx = dlon * cos_lat * METERS_PER_DEG_LAT
    dy = dlat * METERS_PER_DEG_LAT
    speed = np.sqrt(dx**2 + dy**2) / dt

    fig, axes = plt.subplots(
        n_show, 3,
        figsize=(16, figsize_per_row * n_show),
        constrained_layout=True,
        squeeze=False,
    )

    for p in range(n_show):
        # Longitude
        axes[p, 0].plot(time_hrs, ds.lon[p, :], '-', linewidth=0.8,
                        color='#2c3e50')
        axes[p, 0].set_ylabel(f'P{p} lon')
        axes[p, 0].grid(True, alpha=0.3)
        if p == 0:
            axes[p, 0].set_title('Longitude [deg]', fontweight='bold')
        if p == n_show - 1:
            axes[p, 0].set_xlabel('Time [hours]')

        # Latitude
        axes[p, 1].plot(time_hrs, ds.lat[p, :], '-', linewidth=0.8,
                        color='#8e44ad')
        axes[p, 1].set_ylabel(f'P{p} lat')
        axes[p, 1].grid(True, alpha=0.3)
        if p == 0:
            axes[p, 1].set_title('Latitude [deg]', fontweight='bold')
        if p == n_show - 1:
            axes[p, 1].set_xlabel('Time [hours]')

        # Speed
        axes[p, 2].plot(time_hrs[:-1], speed[p, :], '-', linewidth=0.8,
                        color='#e67e22')
        axes[p, 2].set_ylabel(f'P{p} [m/s]')
        axes[p, 2].grid(True, alpha=0.3)
        if p == 0:
            axes[p, 2].set_title('Speed [m/s]', fontweight='bold')
        if p == n_show - 1:
            axes[p, 2].set_xlabel('Time [hours]')

    fig.suptitle(
        f'Per-particle diagnostics ({n_show} of {ds.n_particles})',
        fontsize=14, fontweight='bold',
    )

    return fig
