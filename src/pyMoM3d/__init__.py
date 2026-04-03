"""pyMoM3d: 3D Method of Moments solver with RWG basis functions."""

from .geometry import RectangularPlate, Sphere, Cylinder, Cube, Pyramid
from .mesh import (
    Mesh,
    RWGBasis,
    create_mesh_from_vertices,
    create_rectangular_mesh,
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
    plot_array_layout,
    # Plot style
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
    # Inductor plots
    plot_inductor_characterization,
    plot_model_fit,
    plot_z_vs_y_comparison,
    format_inductance_label,
    format_quality_factor_label,
    format_resistance_label,
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
from .mom.assembly import fill_matrix
from .mom.operators import EFIEOperator, MFIEOperator, CFIEOperator
from .mom.excitation import (
    StripDeltaGapExcitation, find_feed_edges,
    MultiPortExcitation, find_feed_edges_near_center,
)
from .fields import compute_far_field, compute_rcs
from .arrays import (
    LinearDipoleArray,
    compute_array_factor,
    combine_meshes,
    uniform_excitation,
    progressive_phase_excitation,
    arbitrary_excitation,
    scan_angle_to_phase_shift,
)
from .simulation import Simulation, SimulationConfig, SimulationResult, load_stl
from .network import Port, NetworkResult, NetworkExtractor
from .network.port import GroundVia
from .medium import Layer, LayerStack
from .analysis.transmission_line import (
    microstrip_z0_hammerstad, stripline_z0_cohn,
    s_to_abcd, extract_propagation_constant, extract_z0_from_s,
)
from .analysis.inductor import (
    wheeler_inductance, quality_factor, inductance_from_z,
    inductance_from_y, quality_factor_y, resistance_from_y,
    self_resonant_frequency,
)
from .analysis.inductor_model import PiModelParams, fit_pi_model
from .analysis.inductor_characterization import (
    InductorCharacterization, CharacterizationResult,
)
from .greens.layered import LayeredGreensFunction
from .greens.free_space_gf import FreeSpaceGreensFunction
from .mom.operators.efie_layered import MultilayerEFIEOperator
from .mom.surface_impedance import ConductorProperties
from .peec import PEECExtractor, Trace, TraceNetwork, PEECPort
__version__ = '0.2.0'

__all__ = [
    # Geometry
    'RectangularPlate', 'Sphere', 'Cylinder', 'Cube', 'Pyramid',
    # Mesh
    'Mesh', 'RWGBasis',
    'create_mesh_from_vertices', 'create_rectangular_mesh',
    'GmshMesher', 'compute_rwg_connectivity',
    # Visualization
    'plot_mesh', 'plot_mesh_3d', 'plot_surface_current', 'compute_triangle_current_density',
    'plot_surface_current_vectors', 'compute_triangle_current_vectors',
    'plot_array_layout',
    # Plot style
    'configure_latex_style', 'restore_default_style', 'latex_style',
    'format_frequency_label', 'format_angle_label', 'format_rcs_label',
    'format_impedance_label', 'format_current_label',
    'format_directivity_label', 'format_coordinate_label',
    'format_rcs_title', 'format_current_title',
    'format_resonance_annotation', 'format_plane_wave_annotation',
    # Constants
    'c0', 'mu0', 'eps0', 'eta0',
    # MoM
    'fill_impedance_matrix', 'fill_matrix',
    'EFIEOperator', 'MFIEOperator', 'CFIEOperator',
    'PlaneWaveExcitation', 'DeltaGapExcitation', 'StripDeltaGapExcitation', 'find_feed_edges',
    'MultiPortExcitation', 'find_feed_edges_near_center',
    'solve_direct', 'solve_gmres',
    # Fields
    'compute_far_field', 'compute_rcs',
    # Arrays
    'LinearDipoleArray', 'compute_array_factor', 'combine_meshes',
    'uniform_excitation', 'progressive_phase_excitation',
    'arbitrary_excitation', 'scan_angle_to_phase_shift',
    # Simulation
    'Simulation', 'SimulationConfig', 'SimulationResult', 'load_stl',
    # Network extraction
    'Port', 'GroundVia', 'NetworkResult', 'NetworkExtractor',
    # Multilayer
    'Layer', 'LayerStack', 'LayeredGreensFunction',
    'FreeSpaceGreensFunction', 'MultilayerEFIEOperator',
    # Surface impedance
    'ConductorProperties',
    # Analysis — transmission line
    'microstrip_z0_hammerstad', 'stripline_z0_cohn',
    's_to_abcd', 'extract_propagation_constant', 'extract_z0_from_s',
    # Analysis — inductor (Z-based)
    'wheeler_inductance', 'quality_factor', 'inductance_from_z',
    # Analysis — inductor (Y-based, EDA standard)
    'inductance_from_y', 'quality_factor_y', 'resistance_from_y',
    'self_resonant_frequency',
    # Analysis — inductor model & characterization
    'PiModelParams', 'fit_pi_model',
    'InductorCharacterization', 'CharacterizationResult',
    # Inductor plots
    'plot_inductor_characterization', 'plot_model_fit', 'plot_z_vs_y_comparison',
    'format_inductance_label', 'format_quality_factor_label', 'format_resistance_label',
    # PEEC extraction
    'PEECExtractor', 'Trace', 'TraceNetwork', 'PEECPort',
    # Reporter
    'TerminalReporter', 'SilentReporter', 'RecordingReporter',
]
