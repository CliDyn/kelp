"""
3D Visualization routines for unstructured mesh and tracks.
Will gracefully degrade to matplotlib 3D if pyvista is not available.
"""
from typing import Optional, Any
import numpy as np
from kelp.core.mesh import Mesh
from .core import TrajectoryDataset, display_coords

try:
    import pyvista as pv
    HAVE_PYVISTA = True
except ImportError:
    HAVE_PYVISTA = False


def build_pyvista_mesh(mesh: Mesh) -> Optional[Any]:
    """
    Construct a PyVista UnstructuredGrid from the FESOM mesh.
    
    Parameters
    ----------
    mesh : Mesh
        The unstructured mesh.
        
    Returns
    -------
    pyvista.UnstructuredGrid or None
        The pyvista object. None if pyvista is not installed.
    """
    if not HAVE_PYVISTA:
        return None
        
    # Display in the geographic frame (un-rotated for a rotated mesh) so the
    # bathymetry sits under the geographic trajectory output.
    nodes_lon, nodes_lat = display_coords(mesh)

    # PyVista points
    points = np.zeros((nodes_lon.shape[0], 3))
    points[:, 0] = nodes_lon
    points[:, 1] = nodes_lat
    # Use negative depth for Z axis so sea surface is Z=0 and bottom is -depth
    points[:, 2] = -mesh.depth

    cells = []
    cell_types = []
    
    # 5 = triangle, 9 = quad
    import vtk
    
    for face in mesh.elem_nodes:
        valid = face[face != -1]
        
        if len(valid) == 4 and valid[3] == valid[0]:
            valid = valid[:3]
            
        n_verts = len(valid)
        
        cells.append(n_verts)
        cells.extend(valid)
        
        if n_verts == 3:
            cell_types.append(vtk.VTK_TRIANGLE)
        elif n_verts == 4:
            cell_types.append(vtk.VTK_QUAD)
        else:
            cell_types.append(vtk.VTK_POLYGON)
            
    ugrid = pv.UnstructuredGrid(cells, np.array(cell_types), points)
    return ugrid


def plot_mesh_3d(mesh: Mesh, show: bool = True) -> Any:
    """
    Plot the 3D unstructured bathymetry using PyVista.
    
    Parameters
    ----------
    mesh : Mesh
        The unstructured mesh.
    show : bool
        Whether to blocking show the plot.
        
    Returns
    -------
    Plotter object from PyVista or Matplotlib depending on availability.
    """
    if HAVE_PYVISTA:
        ugrid = build_pyvista_mesh(mesh)
        
        plotter = pv.Plotter()
        plotter.add_mesh(ugrid, show_edges=True, scalars=ugrid.points[:, 2], cmap="ocean")
        plotter.set_scale(zscale=100.0) # Exaggerate depth for visibility
        if show:
            plotter.show()
        return plotter
    else:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Build strict triangles from elem_nodes to avoid artificial holes or interpolations
        triangles = []
        for face in mesh.elem_nodes:
            valid = face[face != -1]
            if len(valid) == 4 and valid[3] == valid[0]:
                valid = valid[:3]
                
            if len(valid) == 3:
                triangles.append(valid)
            elif len(valid) == 4:
                triangles.append([valid[0], valid[1], valid[2]])
                triangles.append([valid[0], valid[2], valid[3]])
                
        nodes_lon, nodes_lat = display_coords(mesh)
        if len(triangles) > 0:
            ax.plot_trisurf(
                nodes_lon,
                nodes_lat,
                triangles,
                -mesh.depth,
                cmap='ocean',
                edgecolor='none',
                alpha=0.7
            )
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Depth (m)')
        if show:
            plt.show()
        return ax


def plot_trajectories_3d(ds: TrajectoryDataset, plotter: Any = None, show: bool = True) -> Any:
    """
    Draw 3D lines. Requires a PyVista plotter or matplotlib 3d Axes.
    """
    if HAVE_PYVISTA and isinstance(plotter, pv.Plotter):
        for p in range(ds.n_particles):
            lon = ds.lon[p, :]
            lat = ds.lat[p, :]
            # Depth parameter is natively positive down, convert to Z
            z = -ds.z[p, :] if ds.z is not None else np.zeros_like(lon)
            
            pts = np.column_stack([lon, lat, z])
            # PyVista lines require padding the number of points
            lines = np.hstack([[len(pts)], np.arange(len(pts))])
            poly = pv.PolyData(pts, lines=lines)
            plotter.add_mesh(poly, color="red", line_width=2)
            
        if show:
            plotter.show()
        return plotter
        
    else:
        import matplotlib.pyplot as plt
        if plotter is None:
            fig = plt.figure(figsize=(10, 8))
            plotter = fig.add_subplot(111, projection='3d')
            
        for p in range(ds.n_particles):
            z = -ds.z[p, :] if ds.z is not None else np.zeros_like(ds.lon[p, :])
            plotter.plot(ds.lon[p, :], ds.lat[p, :], z, color='red', alpha=0.5, linewidth=1.0)
            
        plotter.set_xlabel('Longitude')
        plotter.set_ylabel('Latitude')
        plotter.set_zlabel('Depth (m)')
        if show:
            plt.show()
        return plotter
