"""PEEC-based network parameter extraction for planar traces.

Top-level orchestrator that:
1. Takes a TraceNetwork (geometry + ports)
2. Computes the partial inductance matrix Lp (once, frequency-independent)
3. Optionally computes partial capacitance matrix Cp (once)
4. Builds the circuit topology
5. Solves the MNA circuit at each frequency
6. Returns NetworkResult objects compatible with the existing analysis pipeline
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

from ..network.network_result import NetworkResult
from .trace import TraceNetwork
from .partial_inductance import partial_inductance_matrix
from .circuit import PEECCircuit


class PEECExtractor:
    """PEEC-based network parameter extraction.

    Parameters
    ----------
    network : TraceNetwork
        Trace geometry with port definitions.
    include_capacitance : bool
        Include inter-segment partial capacitances.  Default False.
    oxide_thickness : float, optional
        Oxide thickness to ground plane (m).  If provided, adds shunt
        capacitance C_ox = eps_0 * eps_ox * A_seg / h_ox at each segment.
        This models the dominant parasitic for on-chip inductors.
    oxide_eps_r : float
        Relative permittivity of oxide (default 3.9 for SiO2).
    substrate_conductivity : float, optional
        Substrate conductivity (S/m).  If provided, adds shunt substrate
        loss G = sigma_sub * A_seg / h_ox at each segment.
    use_ribbon : bool
        Use finite-width ribbon Neumann formula for off-diagonal
        inductance terms.  Default False (filamentary).
    n_width_points : int
        Gauss points across width for ribbon formula.
    Z0 : float
        Reference impedance for S-parameters (Ohm).

    Examples
    --------
    >>> from pyMoM3d import ConductorProperties
    >>> from pyMoM3d.peec import PEECExtractor, Trace, TraceNetwork, PEECPort
    >>> copper = ConductorProperties(sigma=5.8e7, thickness=2e-6)
    >>> trace = Trace.rectangular_spiral(
    ...     n_turns=2.5, d_out=2e-3, w_trace=100e-6,
    ...     s_space=100e-6, thickness=2e-6, conductor=copper)
    >>> port = PEECPort('P1', positive_segment_idx=0)
    >>> network = TraceNetwork([trace], [port])
    >>> # Free-space (no substrate):
    >>> ext = PEECExtractor(network)
    >>> # On-chip with 3um SiO2 over lossy silicon:
    >>> ext = PEECExtractor(network, oxide_thickness=3e-6,
    ...                     substrate_conductivity=10.0)
    """

    def __init__(
        self,
        network: TraceNetwork,
        include_capacitance: bool = False,
        oxide_thickness: Optional[float] = None,
        oxide_eps_r: float = 3.9,
        substrate_conductivity: Optional[float] = None,
        use_ribbon: bool = False,
        n_width_points: int = 3,
        Z0: float = 50.0,
    ):
        self.network = network
        self.Z0 = Z0

        # Get flat segment list
        self.segments = network.all_segments
        M = len(self.segments)
        if M == 0:
            raise ValueError("TraceNetwork has no segments")

        # Build circuit topology
        self.connectivity, self.num_nodes = network.build_connectivity()

        # Compute partial inductance matrix (frequency-independent)
        self.Lp = partial_inductance_matrix(
            self.segments,
            use_ribbon=use_ribbon,
            n_width_points=n_width_points,
        )

        # Compute partial capacitance matrix (frequency-independent)
        self.Cp = None
        if include_capacitance:
            from .partial_capacitance import partial_capacitance_matrix
            self.Cp = partial_capacitance_matrix(self.segments)

        # Compute shunt capacitance and conductance to ground
        C_shunt = None
        G_shunt = None
        if oxide_thickness is not None:
            from ..utils.constants import eps0
            C_shunt = np.zeros(M, dtype=np.float64)
            for i, seg in enumerate(self.segments):
                area = seg.length * seg.width
                C_shunt[i] = eps0 * oxide_eps_r * area / oxide_thickness

        if substrate_conductivity is not None and oxide_thickness is not None:
            G_shunt = np.zeros(M, dtype=np.float64)
            for i, seg in enumerate(self.segments):
                area = seg.length * seg.width
                # Substrate loss: G = sigma * area / thickness
                # This is a simplified model; actual substrate loss involves
                # the silicon substrate thickness and geometry
                G_shunt[i] = substrate_conductivity * area / oxide_thickness

        # Build circuit
        self.circuit = PEECCircuit(
            Lp=self.Lp,
            connectivity=self.connectivity,
            num_nodes=self.num_nodes,
            ports=network.ports,
            Cp=self.Cp,
            C_shunt=C_shunt,
            G_shunt=G_shunt,
        )

    def extract(
        self,
        frequencies: Union[float, List[float], np.ndarray],
    ) -> List[NetworkResult]:
        """Extract network parameters at specified frequencies.

        Parameters
        ----------
        frequencies : float or array-like
            Frequencies (Hz).

        Returns
        -------
        results : list of NetworkResult
            One per frequency.  Compatible with InductorCharacterization.
        """
        frequencies = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
        return self.circuit.solve_sweep(frequencies, self.segments, Z0=self.Z0)

    @property
    def total_inductance_estimate(self) -> float:
        """Quick estimate of total series inductance (H).

        Sums all elements of the partial inductance matrix.  This equals
        the total inductance when current flows uniformly through all
        segments in series (which is exact for a single trace at DC).
        """
        return float(np.sum(self.Lp))

    @property
    def dc_resistance(self) -> float:
        """Total DC resistance of all segments in series (Ohm)."""
        return sum(
            s.length / (s.conductor.sigma * s.width * s.thickness)
            for s in self.segments
        )
