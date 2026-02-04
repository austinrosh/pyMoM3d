"""Analysis tools: Mie series, convergence, impedance, pattern, and CMA."""

from .mie_series import mie_rcs_pec_sphere, mie_monostatic_rcs_pec_sphere
from .convergence import mesh_convergence_study
from .impedance_analysis import compute_s11, impedance_vs_frequency, s11_vs_frequency
from .pattern_analysis import compute_directivity, compute_beamwidth_3dB
from .cma import (
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

__all__ = [
    'mie_rcs_pec_sphere',
    'mie_monostatic_rcs_pec_sphere',
    'mesh_convergence_study',
    'compute_s11',
    'impedance_vs_frequency',
    's11_vs_frequency',
    'compute_directivity',
    'compute_beamwidth_3dB',
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
