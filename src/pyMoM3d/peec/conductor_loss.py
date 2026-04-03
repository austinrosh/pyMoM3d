"""Frequency-dependent conductor resistance for PEEC segments.

For a trace segment of length l, width w, and surface impedance Z_s(f):

    R_segment(f) = Z_s(f) * l / w

This reuses ConductorProperties.surface_impedance() which implements
the full coth(gamma*t) expression including skin effect.

At DC:   R = l / (sigma * w * t)  (bulk resistance)
At HF:   R ~ l/w * sqrt(pi*f*mu/sigma)  (skin-effect limited)
"""

from __future__ import annotations

from typing import List

import numpy as np

from .trace import TraceSegment


def segment_resistance(seg: TraceSegment, freq: float) -> complex:
    """Compute frequency-dependent resistance of a single segment.

    Parameters
    ----------
    seg : TraceSegment
    freq : float
        Frequency (Hz).

    Returns
    -------
    R : complex
        Segment resistance (Ohm).  Complex due to skin-effect phase.
    """
    Z_s = seg.conductor.surface_impedance(freq)
    return Z_s * seg.length / seg.width


def resistance_vector(
    segments: List[TraceSegment],
    freq: float,
) -> np.ndarray:
    """Compute resistance for all segments at one frequency.

    Parameters
    ----------
    segments : list of TraceSegment, length M
    freq : float
        Frequency (Hz).

    Returns
    -------
    R : ndarray, shape (M,), complex128
    """
    return np.array([segment_resistance(s, freq) for s in segments],
                    dtype=np.complex128)
