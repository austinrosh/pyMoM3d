"""Far-field and RCS computation."""

from .far_field import compute_far_field
from .rcs import compute_rcs, compute_monostatic_rcs

__all__ = ['compute_far_field', 'compute_rcs', 'compute_monostatic_rcs']
