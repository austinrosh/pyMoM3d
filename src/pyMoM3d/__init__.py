"""pyMoM3d: 3D Method of Moments solver with RWG basis functions."""

from .geometry import RectangularPlate, Sphere, Cylinder, Cube, Pyramid
from .mesh import (
    Mesh,
    RWGBasis,
    create_mesh_from_vertices,
    create_rectangular_mesh,
    PythonMesher,
    create_mesh_from_trimesh,
    GmshMesher,
    compute_rwg_connectivity,
)
from .visualization import (
    plot_mesh,
    plot_mesh_3d,
    plot_surface_current,
    compute_triangle_current_density,
    plot_surface_current_vectors,
    compute_triangle_current_vectors,
    # Plot style
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
from .utils.constants import c0, mu0, eps0, eta0
from .utils.reporter import TerminalReporter, SilentReporter, RecordingReporter
from .mom import (
    fill_impedance_matrix,
    PlaneWaveExcitation,
    DeltaGapExcitation,
    solve_direct,
    solve_gmres,
)
from .mom.excitation import StripDeltaGapExcitation, find_feed_edges
from .fields import compute_far_field, compute_rcs
from .simulation import Simulation, SimulationConfig, SimulationResult, load_stl
from .analysis.cma import (
    CMAResult,
    compute_characteristic_modes,
    compute_modal_significance,
    compute_characteristic_angle,
    solve_cma,
    track_modes_across_frequency,
    verify_orthogonality,
    verify_eigenvalue_reality,
    compute_modal_excitation_coefficient,
    expand_current_in_modes,
    cma_frequency_sweep,
)

__version__ = '0.2.0'

__all__ = [
    # Geometry
    'RectangularPlate', 'Sphere', 'Cylinder', 'Cube', 'Pyramid',
    # Mesh
    'Mesh', 'RWGBasis',
    'create_mesh_from_vertices', 'create_rectangular_mesh',
    'PythonMesher', 'create_mesh_from_trimesh', 'GmshMesher', 'compute_rwg_connectivity',
    # Visualization
    'plot_mesh', 'plot_mesh_3d', 'plot_surface_current', 'compute_triangle_current_density',
    'plot_surface_current_vectors', 'compute_triangle_current_vectors',
    # Plot style
    'configure_latex_style', 'restore_default_style', 'latex_style',
    'format_frequency_label', 'format_angle_label', 'format_rcs_label',
    'format_impedance_label', 'format_current_label', 'format_eigenvalue_label',
    'format_modal_significance_label', 'format_directivity_label', 'format_coordinate_label',
    'format_rcs_title', 'format_current_title', 'format_cma_title',
    'format_resonance_annotation', 'format_plane_wave_annotation',
    # Constants
    'c0', 'mu0', 'eps0', 'eta0',
    # MoM
    'fill_impedance_matrix',
    'PlaneWaveExcitation', 'DeltaGapExcitation', 'StripDeltaGapExcitation', 'find_feed_edges',
    'solve_direct', 'solve_gmres',
    # Fields
    'compute_far_field', 'compute_rcs',
    # Simulation
    'Simulation', 'SimulationConfig', 'SimulationResult', 'load_stl',
    # Reporter
    'TerminalReporter', 'SilentReporter', 'RecordingReporter',
    # CMA
    'CMAResult',
    'compute_characteristic_modes',
    'compute_modal_significance',
    'compute_characteristic_angle',
    'solve_cma',
    'track_modes_across_frequency',
    'verify_orthogonality',
    'verify_eigenvalue_reality',
    'compute_modal_excitation_coefficient',
    'expand_current_in_modes',
    'cma_frequency_sweep',
]
