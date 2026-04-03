"""Analysis tools: Mie series, convergence, impedance, pattern, transmission line, inductor."""

from .mie_series import mie_rcs_pec_sphere, mie_monostatic_rcs_pec_sphere
from .convergence import mesh_convergence_study
from .impedance_analysis import compute_s11, impedance_vs_frequency, s11_vs_frequency
from .pattern_analysis import compute_directivity, compute_beamwidth_3dB
from .transmission_line import (
    microstrip_z0_hammerstad,
    stripline_z0_cohn,
    s_to_abcd,
    extract_propagation_constant,
    extract_z0_from_s,
)
from .inductor import (
    wheeler_inductance, quality_factor, inductance_from_z,
    inductance_from_y, quality_factor_y, resistance_from_y,
    self_resonant_frequency,
    differential_inductance_from_y, differential_quality_factor_y,
)
from .inductor_model import PiModelParams, fit_pi_model, estimate_initial_params
from .inductor_characterization import (
    InductorCharacterization, CharacterizationResult, InductorParams,
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
    'microstrip_z0_hammerstad',
    'stripline_z0_cohn',
    's_to_abcd',
    'extract_propagation_constant',
    'extract_z0_from_s',
    # Inductor — Z-based
    'wheeler_inductance',
    'quality_factor',
    'inductance_from_z',
    # Inductor — Y-based (EDA standard)
    'inductance_from_y',
    'quality_factor_y',
    'resistance_from_y',
    'self_resonant_frequency',
    'differential_inductance_from_y',
    'differential_quality_factor_y',
    # Inductor — broadband model
    'PiModelParams',
    'fit_pi_model',
    'estimate_initial_params',
    # Inductor — characterization orchestrator
    'InductorCharacterization',
    'CharacterizationResult',
    'InductorParams',
]
