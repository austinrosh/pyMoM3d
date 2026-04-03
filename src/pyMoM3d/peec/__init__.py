"""PEEC (Partial Element Equivalent Circuit) extraction module.

Provides accurate inductance, resistance, and capacitance extraction
for planar conductor traces by decomposing them into filamentary
segments with constrained current flow direction.

This approach solves the fundamental limitation of surface-MoM (EFIE)
for electrically small structures: by constraining current to flow
along the trace direction, the inductive response is correctly captured
without scalar potential dominance artifacts.
"""

from .trace import TraceSegment, Trace, TraceNetwork, PEECPort
from .extractor import PEECExtractor

__all__ = [
    'TraceSegment',
    'Trace',
    'TraceNetwork',
    'PEECPort',
    'PEECExtractor',
]
