"""Antenna array support for pyMoM3d."""

from .linear_array import (
    LinearDipoleArray,
    compute_array_factor,
    combine_meshes,
    combine_array_meshes,
    uniform_excitation,
    progressive_phase_excitation,
    arbitrary_excitation,
    scan_angle_to_phase_shift,
)

__all__ = [
    'LinearDipoleArray',
    'compute_array_factor',
    'combine_meshes',
    'combine_array_meshes',
    'uniform_excitation',
    'progressive_phase_excitation',
    'arbitrary_excitation',
    'scan_angle_to_phase_shift',
]
