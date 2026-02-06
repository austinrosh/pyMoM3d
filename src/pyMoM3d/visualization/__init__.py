"""Visualization module for plotting and visualization utilities."""

from .mesh_plot import (
    plot_mesh,
    plot_mesh_3d,
    plot_surface_current,
    compute_triangle_current_density,
    plot_surface_current_vectors,
    compute_triangle_current_vectors,
)

from .plot_style import (
    configure_latex_style,
    restore_default_style,
    latex_style,
    format_frequency_label,
    format_angle_label,
    format_rcs_label,
    format_impedance_label,
    format_current_label,
    format_eigenvalue_label,
    format_modal_significance_label,
    format_directivity_label,
    format_coordinate_label,
    format_rcs_title,
    format_current_title,
    format_cma_title,
    format_resonance_annotation,
    format_plane_wave_annotation,
)

__all__ = [
    # Mesh plotting
    'plot_mesh',
    'plot_mesh_3d',
    'plot_surface_current',
    'compute_triangle_current_density',
    'plot_surface_current_vectors',
    'compute_triangle_current_vectors',
    # Plot style configuration
    'configure_latex_style',
    'restore_default_style',
    'latex_style',
    # Label formatters
    'format_frequency_label',
    'format_angle_label',
    'format_rcs_label',
    'format_impedance_label',
    'format_current_label',
    'format_eigenvalue_label',
    'format_modal_significance_label',
    'format_directivity_label',
    'format_coordinate_label',
    # Title formatters
    'format_rcs_title',
    'format_current_title',
    'format_cma_title',
    # Annotation helpers
    'format_resonance_annotation',
    'format_plane_wave_annotation',
]
