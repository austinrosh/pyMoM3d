"""Green's functions, quadrature rules, and singularity handling."""

from .quadrature import triangle_quad_rule, integrate_over_triangle
from .free_space import scalar_green
from .singularity import integrate_green_singular, integrate_rho_green_singular

__all__ = [
    'triangle_quad_rule',
    'integrate_over_triangle',
    'scalar_green',
    'integrate_green_singular',
    'integrate_rho_green_singular',
]
