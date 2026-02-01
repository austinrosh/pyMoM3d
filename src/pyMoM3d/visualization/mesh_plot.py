"""Mesh visualization utilities."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Optional

from ..mesh.mesh_data import Mesh
from ..mesh.rwg_basis import RWGBasis


def plot_mesh_3d(
    mesh: Mesh,
    ax: Optional[plt.Axes] = None,
    show_edges: bool = True,
    show_normals: bool = False,
    normal_scale: float = 0.1,
    color: str = 'lightblue',
    alpha: float = 0.7,
    edge_color: str = 'black',
    edge_width: float = 0.5
) -> plt.Axes:
    """
    Plot a 3D triangular mesh.
    
    Parameters
    ----------
    mesh : Mesh
        Mesh object to visualize
    ax : matplotlib.axes.Axes, optional
        Existing 3D axes. If None, creates new figure and axes.
    show_edges : bool, default True
        Whether to draw triangle edges
    show_normals : bool, default False
        Whether to draw triangle normal vectors
    normal_scale : float, default 0.1
        Scale factor for normal vectors
    color : str, default 'lightblue'
        Face color for triangles
    alpha : float, default 0.7
        Transparency of triangle faces
    edge_color : str, default 'black'
        Color for triangle edges
    edge_width : float, default 0.5
        Width of triangle edges
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        3D axes object
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Collect triangle vertices
    triangles_3d = []
    for tri in mesh.triangles:
        triangle_verts = mesh.vertices[tri]
        triangles_3d.append(triangle_verts)
    
    # Create Poly3DCollection
    collection = Poly3DCollection(
        triangles_3d,
        facecolors=color,
        edgecolors=edge_color if show_edges else 'none',
        linewidths=edge_width if show_edges else 0,
        alpha=alpha
    )
    ax.add_collection3d(collection)
    
    # Draw normal vectors if requested
    if show_normals:
        for i, tri in enumerate(mesh.triangles):
            # Triangle center
            center = np.mean(mesh.vertices[tri], axis=0)
            normal = mesh.triangle_normals[i]
            
            # Draw arrow from center along normal
            end = center + normal_scale * normal
            ax.plot(
                [center[0], end[0]],
                [center[1], end[1]],
                [center[2], end[2]],
                'r-', linewidth=2
            )
    
    # Set axis limits
    vertices = mesh.vertices
    ax.set_xlim([vertices[:, 0].min(), vertices[:, 0].max()])
    ax.set_ylim([vertices[:, 1].min(), vertices[:, 1].max()])
    ax.set_zlim([vertices[:, 2].min(), vertices[:, 2].max()])
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Equal aspect ratio
    max_range = np.array([
        vertices[:, 0].max() - vertices[:, 0].min(),
        vertices[:, 1].max() - vertices[:, 1].min(),
        vertices[:, 2].max() - vertices[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_title(f'Mesh: {mesh.get_num_vertices()} vertices, '
                 f'{mesh.get_num_triangles()} triangles, '
                 f'{mesh.get_num_edges()} edges')
    
    return ax


def plot_mesh(
    mesh: Mesh,
    projection: str = 'xy',
    ax: Optional[plt.Axes] = None,
    show_edges: bool = True,
    color: str = 'lightblue',
    alpha: float = 0.7,
    edge_color: str = 'black',
    edge_width: float = 0.5
) -> plt.Axes:
    """
    Plot a 2D projection of the mesh.
    
    Parameters
    ----------
    mesh : Mesh
        Mesh object to visualize
    projection : str, default 'xy'
        Projection plane: 'xy', 'xz', or 'yz'
    ax : matplotlib.axes.Axes, optional
        Existing 2D axes. If None, creates new figure and axes.
    show_edges : bool, default True
        Whether to draw triangle edges
    color : str, default 'lightblue'
        Face color for triangles
    alpha : float, default 0.7
        Transparency of triangle faces
    edge_color : str, default 'black'
        Color for triangle edges
    edge_width : float, default 0.5
        Width of triangle edges
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        2D axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Map projection to axis indices
    proj_map = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
    if projection not in proj_map:
        raise ValueError(f"projection must be one of {list(proj_map.keys())}")
    
    idx1, idx2 = proj_map[projection]
    
    # Plot triangles
    for tri in mesh.triangles:
        triangle_verts = mesh.vertices[tri]
        x = triangle_verts[:, idx1]
        y = triangle_verts[:, idx2]
        
        # Close the triangle
        x_closed = np.append(x, x[0])
        y_closed = np.append(y, y[0])
        
        ax.fill(x, y, color=color, alpha=alpha, edgecolor=edge_color if show_edges else 'none',
                linewidth=edge_width if show_edges else 0)

    # Set labels
    labels = ['X', 'Y', 'Z']
    ax.set_xlabel(labels[idx1])
    ax.set_ylabel(labels[idx2])
    ax.set_title(f'Mesh projection ({projection}): {mesh.get_num_vertices()} vertices, '
                 f'{mesh.get_num_triangles()} triangles')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    return ax


def compute_triangle_current_density(
    I: np.ndarray,
    basis: RWGBasis,
    mesh: Mesh,
) -> np.ndarray:
    """Compute surface current density magnitude |J| at each triangle centroid.

    Evaluates J(r) = sum_n I_n * f_n(r) at the centroid of each triangle,
    where f_n is the RWG basis function. Returns |J| per triangle.

    Parameters
    ----------
    I : ndarray, shape (N,), complex128
        RWG basis function coefficients.
    basis : RWGBasis
        RWG basis function data.
    mesh : Mesh
        Surface mesh.

    Returns
    -------
    J_mag : ndarray, shape (num_triangles,), float64
        Surface current density magnitude at each triangle centroid.
    """
    num_tri = mesh.get_num_triangles()
    J = np.zeros((num_tri, 3), dtype=np.complex128)

    centroids = np.mean(mesh.vertices[mesh.triangles], axis=1)  # (num_tri, 3)

    for n in range(basis.num_basis):
        l_n = basis.edge_length[n]

        # T+ contribution: f_n = (l_n / 2A+) * (r - r_free+)
        t_p = basis.t_plus[n]
        A_p = basis.area_plus[n]
        r_free_p = mesh.vertices[basis.free_vertex_plus[n]]
        rho_p = centroids[t_p] - r_free_p
        J[t_p] += I[n] * (l_n / (2.0 * A_p)) * rho_p

        # T- contribution: f_n = (l_n / 2A-) * (r_free- - r)
        t_m = basis.t_minus[n]
        A_m = basis.area_minus[n]
        r_free_m = mesh.vertices[basis.free_vertex_minus[n]]
        rho_m = r_free_m - centroids[t_m]
        J[t_m] += I[n] * (l_n / (2.0 * A_m)) * rho_m

    return np.linalg.norm(J, axis=1).real


def plot_surface_current(
    I: np.ndarray,
    basis: RWGBasis,
    mesh: Mesh,
    ax: Optional[plt.Axes] = None,
    cmap: str = 'hot',
    log_scale: bool = False,
    show_edges: bool = True,
    edge_color: str = 'gray',
    edge_width: float = 0.3,
    alpha: float = 1.0,
    title: Optional[str] = None,
    clim: Optional[tuple] = None,
) -> tuple:
    """Plot surface current density |J| as a heatmap on the 3D mesh.

    Parameters
    ----------
    I : ndarray, shape (N,), complex128
        RWG basis function coefficients from the MoM solve.
    basis : RWGBasis
        RWG basis function data.
    mesh : Mesh
        Surface mesh.
    ax : matplotlib 3D axes, optional
        If None, creates a new figure.
    cmap : str, default 'hot'
        Matplotlib colormap name.
    log_scale : bool, default False
        If True, plot 10*log10(|J|) in dB scale.
    show_edges : bool, default True
        Whether to draw triangle edges.
    edge_color : str, default 'gray'
        Color of triangle edges.
    edge_width : float, default 0.3
        Width of triangle edges.
    alpha : float, default 1.0
        Face transparency.
    title : str, optional
        Plot title. Auto-generated if None.
    clim : tuple of (vmin, vmax), optional
        Color limits. Auto-scaled if None.

    Returns
    -------
    ax : matplotlib 3D axes
    mappable : ScalarMappable
        For creating a colorbar via plt.colorbar(mappable).
    """
    J_mag = compute_triangle_current_density(I, basis, mesh)

    if log_scale:
        values = 10.0 * np.log10(np.maximum(J_mag, 1e-30))
        label = '|J| (dB A/m)'
    else:
        values = J_mag
        label = '|J| (A/m)'

    if clim is not None:
        vmin, vmax = clim
    else:
        vmin, vmax = values.min(), values.max()

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.get_cmap(cmap)

    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

    triangles_3d = []
    face_colors = []
    for i, tri in enumerate(mesh.triangles):
        triangles_3d.append(mesh.vertices[tri])
        face_colors.append(colormap(norm(values[i])))

    collection = Poly3DCollection(
        triangles_3d,
        facecolors=face_colors,
        edgecolors=edge_color if show_edges else 'none',
        linewidths=edge_width if show_edges else 0,
        alpha=alpha,
    )
    ax.add_collection3d(collection)

    # Equal aspect ratio
    vertices = mesh.vertices
    max_range = np.array([
        vertices[:, 0].max() - vertices[:, 0].min(),
        vertices[:, 1].max() - vertices[:, 1].min(),
        vertices[:, 2].max() - vertices[:, 2].min(),
    ]).max() / 2.0
    mid = (vertices.max(axis=0) + vertices.min(axis=0)) * 0.5
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    if title is None:
        title = f'Surface current density, {basis.num_basis} basis functions'
    ax.set_title(title)

    # ScalarMappable for colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])

    # Add colorbar to the figure
    cbar = ax.figure.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label(label)

    return ax, sm
