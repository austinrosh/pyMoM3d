"""Probe feed port construction for wire-surface hybrid MoM.

Creates Port objects for vertical probe feeds, where the excitation voltage
is applied at the wire base (ground plane connection).
"""

from __future__ import annotations

from ..network.port import Port
from .wire_basis import WireBasis, WireMesh


def create_probe_port(
    wire_basis: WireBasis,
    n_surface: int,
    name: str = 'probe',
    V_ref: complex = 1.0,
) -> Port:
    """Create a Port for a vertical probe feed.

    The port voltage is applied at the base of the wire (the first wire
    basis function, located at the ground plane end).  The basis index
    is offset by ``n_surface`` to reference the correct position in the
    hybrid impedance matrix.

    Parameters
    ----------
    wire_basis : WireBasis
        Wire basis functions for this probe.
    n_surface : int
        Number of surface (RWG) basis functions.  Wire indices in the
        global matrix start at this offset.
    name : str
        Port label.
    V_ref : complex
        Reference excitation voltage (V).

    Returns
    -------
    Port
    """
    if wire_basis.num_basis < 1:
        raise ValueError("Wire must have at least 1 basis function (2 segments)")

    # First wire basis function is at the bottom of the wire
    base_idx = n_surface + 0
    return Port(
        name=name,
        feed_basis_indices=[base_idx],
        feed_signs=[+1],
        V_ref=V_ref,
    )
