"""Visualization module for plotting and visualization utilities."""

from .mesh_plot import (
    plot_mesh,
    plot_mesh_3d,
    plot_surface_current,
    compute_triangle_current_density,
    plot_surface_current_vectors,
    compute_triangle_current_vectors,
    plot_array_layout,
    plot_structure_with_ports,
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
    format_directivity_label,
    format_coordinate_label,
    format_rcs_title,
    format_current_title,
    format_resonance_annotation,
    format_plane_wave_annotation,
)

from .peec_plot import plot_peec_currents

from .inductor_plots import (
    plot_inductor_characterization,
    plot_model_fit,
    plot_z_vs_y_comparison,
    format_inductance_label,
    format_quality_factor_label,
    format_resistance_label,
)

__all__ = [
    # Mesh plotting
    'plot_mesh',
    'plot_mesh_3d',
    'plot_surface_current',
    'compute_triangle_current_density',
    'plot_surface_current_vectors',
    'compute_triangle_current_vectors',
    'plot_array_layout',
    'plot_structure_with_ports',
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
    'format_directivity_label',
    'format_coordinate_label',
    # Inductor label formatters
    'format_inductance_label',
    'format_quality_factor_label',
    'format_resistance_label',
    # Inductor plots
    'plot_inductor_characterization',
    'plot_model_fit',
    'plot_z_vs_y_comparison',
    # Title formatters
    'format_rcs_title',
    'format_current_title',
    # Annotation helpers
    'format_resonance_annotation',
    'format_plane_wave_annotation',
    # PEEC visualization
    'plot_peec_currents',
]
