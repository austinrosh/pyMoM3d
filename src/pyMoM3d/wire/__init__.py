"""Wire basis functions and hybrid wire-surface MoM assembly.

Provides 1D rooftop (piecewise-linear) basis functions on wire segments,
thin-wire EFIE kernels, wire-surface coupling integrals, and block-structured
hybrid assembly for combined wire + surface problems (e.g. probe feeds).
"""

from .wire_basis import WireSegment, WireMesh, WireBasis, compute_wire_connectivity
from .kernels import (
    fill_wire_wire, fill_wire_surface, wire_quad_rule,
    fill_wire_wire_static, fill_wire_surface_static,
)
from .hybrid import (
    HybridBasis, HybridBasisAdapter, fill_hybrid_matrix,
)
from .probe_port import create_probe_port

__all__ = [
    'WireSegment', 'WireMesh', 'WireBasis', 'compute_wire_connectivity',
    'fill_wire_wire', 'fill_wire_surface', 'wire_quad_rule',
    'fill_wire_wire_static', 'fill_wire_surface_static',
    'HybridBasis', 'HybridBasisAdapter', 'fill_hybrid_matrix',
    'create_probe_port',
]
