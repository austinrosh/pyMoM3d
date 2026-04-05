"""2D electrostatic cross-section solver for transmission line parameters.

Provides a finite-difference Laplace solver that computes per-unit-length
capacitance, inductance, characteristic impedance, and effective permittivity
for arbitrary planar transmission line cross-sections.

Example
-------
>>> from pyMoM3d.cross_section import microstrip_cross_section, extract_tl_params
>>> xs = microstrip_cross_section(W=3.06e-3, h=1.6e-3, eps_r=4.4)
>>> tl = extract_tl_params(xs)
>>> print(f"Z0 = {tl.Z0:.1f} Ohm, eps_eff = {tl.eps_eff:.3f}")
"""

from .geometry import CrossSection, Conductor, DielectricRegion
from .grid import NonUniformGrid, build_grid, build_grid_for_cross_section
from .solver import LaplaceSolver2D, LaplaceSolution
from .extraction import (
    CrossSectionResult, extract_tl_params, extract_multiconductor_params,
    compute_reference_impedance,
)
from .presets import (
    microstrip_cross_section,
    stripline_cross_section,
    cpw_cross_section,
    coupled_microstrip_cross_section,
    from_layer_stack,
)

__all__ = [
    # Geometry
    'CrossSection', 'Conductor', 'DielectricRegion',
    # Grid
    'NonUniformGrid', 'build_grid', 'build_grid_for_cross_section',
    # Solver
    'LaplaceSolver2D', 'LaplaceSolution',
    # Extraction
    'CrossSectionResult', 'extract_tl_params', 'extract_multiconductor_params',
    'compute_reference_impedance',
    # Presets
    'microstrip_cross_section', 'stripline_cross_section',
    'cpw_cross_section', 'coupled_microstrip_cross_section',
    'from_layer_stack',
]
