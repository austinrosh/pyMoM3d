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
    
    # Set labels with LaTeX formatting
    ax.set_xlabel(r'$x$ (m)')
    ax.set_ylabel(r'$y$ (m)')
    ax.set_zlabel(r'$z$ (m)')

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

    n_v = mesh.get_num_vertices()
    n_t = mesh.get_num_triangles()
    n_e = mesh.get_num_edges()
    ax.set_title(rf'Mesh: $N_v = {n_v}$, $N_t = {n_t}$, $N_e = {n_e}$')
    
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

    # Set labels with LaTeX formatting
    labels = [r'$x$ (m)', r'$y$ (m)', r'$z$ (m)']
    ax.set_xlabel(labels[idx1])
    ax.set_ylabel(labels[idx2])
    n_v = mesh.get_num_vertices()
    n_t = mesh.get_num_triangles()
    ax.set_title(rf'Mesh projection ({projection}): $N_v = {n_v}$, $N_t = {n_t}$')
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


def compute_triangle_current_vectors(
    I: np.ndarray,
    basis: RWGBasis,
    mesh: Mesh,
    component: str = 'real',
) -> tuple:
    """Compute surface current density vectors J(r) at each triangle centroid.

    Evaluates J(r) = sum_n I_n * f_n(r) at the centroid of each triangle,
    where f_n is the RWG basis function. Returns J as 3D vectors.

    Parameters
    ----------
    I : ndarray, shape (N,), complex128
        RWG basis function coefficients.
    basis : RWGBasis
        RWG basis function data.
    mesh : Mesh
        Surface mesh.
    component : str, default 'real'
        Which part of the complex current to return: 'real' or 'imag'.

    Returns
    -------
    J_vectors : ndarray, shape (num_triangles, 3), float64
        Surface current density vectors at each triangle centroid.
    J_mag : ndarray, shape (num_triangles,), float64
        Surface current density magnitude at each triangle centroid.
    centroids : ndarray, shape (num_triangles, 3), float64
        Centroid positions of each triangle.
    """
    if component not in ('real', 'imag'):
        raise ValueError("component must be 'real' or 'imag'")

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

    J_mag = np.linalg.norm(J, axis=1).real

    if component == 'real':
        J_vectors = J.real
    else:
        J_vectors = J.imag

    return J_vectors, J_mag, centroids


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
        label = r'$|\mathbf{J}|$ (dB A/m)'
    else:
        values = J_mag
        label = r'$|\mathbf{J}|$ (A/m)'

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

    ax.set_xlabel(r'$x$ (m)')
    ax.set_ylabel(r'$y$ (m)')
    ax.set_zlabel(r'$z$ (m)')

    if title is None:
        title = rf'Surface current density $|\mathbf{{J}}|$, $N = {basis.num_basis}$'
    ax.set_title(title)

    # ScalarMappable for colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])

    # Add colorbar to the figure
    cbar = ax.figure.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label(label)

    return ax, sm


def _subsample_indices(
    J_mag: np.ndarray,
    max_arrows: Optional[int],
    method: str = 'magnitude',
) -> np.ndarray:
    """Select indices for subsampling arrows.

    Parameters
    ----------
    J_mag : ndarray, shape (N,)
        Current density magnitudes.
    max_arrows : int or None
        Maximum number of arrows. If None or >= len(J_mag), return all indices.
    method : str, default 'magnitude'
        Subsampling method: 'magnitude' keeps highest |J|, 'uniform' picks random.

    Returns
    -------
    indices : ndarray
        Selected triangle indices.
    """
    num_tri = len(J_mag)
    if max_arrows is None or num_tri <= max_arrows:
        return np.arange(num_tri)

    if method == 'magnitude':
        return np.argsort(J_mag)[-max_arrows:]
    else:  # 'uniform'
        return np.random.choice(num_tri, max_arrows, replace=False)


def plot_surface_current_vectors(
    I: np.ndarray,
    basis: RWGBasis,
    mesh: Mesh,
    ax: Optional[plt.Axes] = None,
    component: str = 'real',
    scale: float = 1.0,
    normalize: bool = False,
    subsample: Optional[int] = None,
    subsample_method: str = 'magnitude',
    color_by_magnitude: bool = True,
    cmap: str = 'viridis',
    arrow_color: str = 'black',
    arrow_width: float = 1.5,
    show_mesh: bool = True,
    mesh_alpha: float = 0.3,
    mesh_color: str = 'lightgray',
    title: Optional[str] = None,
    clim: Optional[tuple] = None,
) -> tuple:
    """Plot surface current density as 3D vector arrows on the mesh.

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
    component : str, default 'real'
        Which part of complex J to show: 'real' or 'imag'.
    scale : float, default 1.0
        Arrow length multiplier (auto-scaled to ~5% of mesh size).
    normalize : bool, default False
        If True, all arrows have same length (direction only).
    subsample : int, optional
        Maximum number of arrows. If None, plots all triangles.
    subsample_method : str, default 'magnitude'
        'magnitude' keeps highest |J|, 'uniform' picks random triangles.
    color_by_magnitude : bool, default True
        If True, color arrows by |J| using colormap. Otherwise use arrow_color.
    cmap : str, default 'viridis'
        Matplotlib colormap name (used if color_by_magnitude=True).
    arrow_color : str, default 'black'
        Uniform arrow color (used if color_by_magnitude=False).
    arrow_width : float, default 1.5
        Line width for arrows.
    show_mesh : bool, default True
        Whether to render underlying mesh surface.
    mesh_alpha : float, default 0.3
        Transparency of mesh surface.
    mesh_color : str, default 'lightgray'
        Color of mesh surface.
    title : str, optional
        Plot title. Auto-generated if None.
    clim : tuple of (vmin, vmax), optional
        Color limits for magnitude coloring. Auto-scaled if None.

    Returns
    -------
    ax : matplotlib 3D axes
    mappable : ScalarMappable or None
        For creating a colorbar (None if color_by_magnitude=False).
    """
    # Compute current vectors
    J_vectors, J_mag, centroids = compute_triangle_current_vectors(
        I, basis, mesh, component=component
    )

    # Subsample if requested
    indices = _subsample_indices(J_mag, subsample, subsample_method)
    J_vectors = J_vectors[indices]
    J_mag_sub = J_mag[indices]
    centroids = centroids[indices]

    # Compute auto-scale based on mesh size
    vertices = mesh.vertices
    mesh_size = np.array([
        vertices[:, 0].max() - vertices[:, 0].min(),
        vertices[:, 1].max() - vertices[:, 1].min(),
        vertices[:, 2].max() - vertices[:, 2].min(),
    ]).max()

    J_mag_max = J_mag_sub.max() if J_mag_sub.max() > 0 else 1.0
    arrow_scale = 0.05 * mesh_size * scale / J_mag_max

    # Normalize vectors if requested
    if normalize:
        norms = np.linalg.norm(J_vectors, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        J_vectors = J_vectors / norms
        arrow_scale = 0.05 * mesh_size * scale

    # Create figure if needed
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

    # Draw mesh surface if requested
    if show_mesh:
        triangles_3d = [mesh.vertices[tri] for tri in mesh.triangles]
        collection = Poly3DCollection(
            triangles_3d,
            facecolors=mesh_color,
            edgecolors='gray',
            linewidths=0.3,
            alpha=mesh_alpha,
        )
        ax.add_collection3d(collection)

    # Prepare arrow colors
    sm = None
    if color_by_magnitude:
        if clim is not None:
            vmin, vmax = clim
        else:
            vmin, vmax = J_mag_sub.min(), J_mag_sub.max()

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        colormap = plt.get_cmap(cmap)
        colors = colormap(norm(J_mag_sub))

        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
    else:
        colors = arrow_color

    # Draw quiver arrows
    ax.quiver(
        centroids[:, 0], centroids[:, 1], centroids[:, 2],
        J_vectors[:, 0] * arrow_scale,
        J_vectors[:, 1] * arrow_scale,
        J_vectors[:, 2] * arrow_scale,
        colors=colors,
        linewidth=arrow_width,
        arrow_length_ratio=0.3,
        normalize=False,
    )

    # Set axis limits with equal aspect ratio
    max_range = np.array([
        vertices[:, 0].max() - vertices[:, 0].min(),
        vertices[:, 1].max() - vertices[:, 1].min(),
        vertices[:, 2].max() - vertices[:, 2].min(),
    ]).max() / 2.0
    mid = (vertices.max(axis=0) + vertices.min(axis=0)) * 0.5
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.set_xlabel(r'$x$ (m)')
    ax.set_ylabel(r'$y$ (m)')
    ax.set_zlabel(r'$z$ (m)')

    if title is None:
        if component == 'real':
            component_label = r'$\mathrm{Re}(\mathbf{J})$'
        else:
            component_label = r'$\mathrm{Im}(\mathbf{J})$'
        title = rf'Surface current vectors {component_label}, $N = {basis.num_basis}$'
    ax.set_title(title)

    return ax, sm
