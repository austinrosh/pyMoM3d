"""RLCG network assembly and MNA circuit solve for PEEC.

Assembles and solves the Modified Nodal Analysis (MNA) system:

    [Y_node    A^T ] [V  ]   [I_port]
    [A      -Z_br  ] [I_br] = [0     ]

where:
- Y_node: nodal admittance matrix (from partial capacitances)
- A: incidence matrix (branch-node topology)
- Z_br = R + j*omega*Lp: branch impedance matrix
- V: node voltages
- I_br: branch currents
- I_port: port current excitations

For inductance extraction (no capacitance), Y_node = 0 and the system
simplifies to solving the branch impedance equation with KCL constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..network.network_result import NetworkResult
from .trace import PEECPort, TraceSegment
from .conductor_loss import resistance_vector


@dataclass
class PEECCircuit:
    """Assembled RLCG network from partial elements.

    Parameters
    ----------
    Lp : ndarray, shape (M, M)
        Partial inductance matrix (H).
    connectivity : ndarray, shape (M, 2), int
        Node indices [start_node, end_node] for each segment.
    num_nodes : int
        Total number of unique circuit nodes.
    ports : list of PEECPort
        Port definitions.
    Cp : ndarray, shape (M, M), optional
        Partial capacitance matrix (F).  If None, capacitance is ignored.
    C_shunt : ndarray, shape (M,), optional
        Per-segment shunt capacitance to ground (F).  Models oxide/substrate
        capacitance for on-chip inductors.  Distributed equally to the
        two nodes of each segment.
    G_shunt : ndarray, shape (M,), optional
        Per-segment shunt conductance to ground (S).  Models substrate loss.
    """

    Lp: np.ndarray
    connectivity: np.ndarray
    num_nodes: int
    ports: List[PEECPort]
    Cp: Optional[np.ndarray] = None
    C_shunt: Optional[np.ndarray] = None
    G_shunt: Optional[np.ndarray] = None

    def _build_incidence_matrix(self) -> np.ndarray:
        """Build the branch-node incidence matrix A.

        A[m, n] = +1 if branch m starts at node n
        A[m, n] = -1 if branch m ends at node n

        Convention: current flows from start_node to end_node.

        Returns
        -------
        A : ndarray, shape (M, num_nodes)
        """
        M = len(self.connectivity)
        A = np.zeros((M, self.num_nodes), dtype=np.float64)
        for m in range(M):
            n_start, n_end = self.connectivity[m]
            A[m, n_start] = +1.0
            A[m, n_end] = -1.0
        return A

    def solve_impedance(
        self,
        freq: float,
        segments: List[TraceSegment],
    ) -> np.ndarray:
        """Solve the MNA system at one frequency for port Z-matrix.

        For each port, we inject 1A at the port nodes and solve for
        all node voltages.  The port voltage gives the Z-matrix column.

        The MNA system is:

            [Y_node    A^T ] [V  ]   [I_s ]
            [A      -Z_br  ] [I_br] = [V_s ]

        where Z_br = diag(R) + j*omega*Lp is the branch impedance matrix.

        For ground-return ports (negative_segment_idx == -1), we add a
        ground node and ground the reference via a short circuit.

        Parameters
        ----------
        freq : float
            Frequency (Hz).
        segments : list of TraceSegment

        Returns
        -------
        Z_port : ndarray, shape (P, P), complex128
            Port impedance matrix.
        """
        omega = 2.0 * np.pi * freq
        M = len(segments)
        N = self.num_nodes
        P = len(self.ports)

        # Branch impedance matrix: Z_br = diag(R) + j*omega*Lp
        R = resistance_vector(segments, freq)
        Z_br = np.diag(R) + 1j * omega * self.Lp  # (M, M)

        # Incidence matrix
        A = self._build_incidence_matrix()  # (M, N)

        # Build nodal admittance from capacitance if present
        Y_node = np.zeros((N, N), dtype=np.complex128)

        if self.Cp is not None and omega > 0:
            # Y_cap = j*omega * C  (nodal capacitive admittance)
            # Transform from branch to nodal: Y_node = A^T * Y_cap_br * A
            # where Y_cap_br = j*omega * Cp
            Y_cap_br = 1j * omega * self.Cp
            Y_node += A.T @ Y_cap_br @ A

        # Add shunt capacitance to ground (oxide/substrate)
        if self.C_shunt is not None and omega > 0:
            for m in range(M):
                c_half = 0.5 * self.C_shunt[m]  # split between 2 nodes
                y_shunt = 1j * omega * c_half
                n_start, n_end = self.connectivity[m]
                Y_node[n_start, n_start] += y_shunt
                Y_node[n_end, n_end] += y_shunt

        # Add shunt conductance to ground (substrate loss)
        if self.G_shunt is not None:
            for m in range(M):
                g_half = 0.5 * self.G_shunt[m]
                n_start, n_end = self.connectivity[m]
                Y_node[n_start, n_start] += g_half
                Y_node[n_end, n_end] += g_half

        # MNA system:
        # [Y_node   A^T ] [V   ]   [I_ext]
        # [A      -Z_br ] [I_br] = [0    ]
        #
        # Size: (N + M) x (N + M)

        S = N + M
        MNA = np.zeros((S, S), dtype=np.complex128)
        MNA[:N, :N] = Y_node
        MNA[:N, N:] = A.T
        MNA[N:, :N] = A
        MNA[N:, N:] = -Z_br

        # We need to set a voltage reference (ground).
        # Pick a ground node. For single-port ground-return,
        # ground the first node of the first trace.
        # We'll stamp the ground constraint by replacing one row
        # in the nodal block with V_gnd = 0.
        ground_node = self._find_ground_node()

        # Replace ground node equation with V_ground = 0
        MNA[ground_node, :] = 0.0
        MNA[ground_node, ground_node] = 1.0

        # Solve for each port excitation
        Z_port = np.zeros((P, P), dtype=np.complex128)

        for p_exc in range(P):
            port = self.ports[p_exc]
            rhs = np.zeros(S, dtype=np.complex128)

            # Inject current at port nodes
            pos_seg = port.positive_segment_idx
            pos_start_node = self.connectivity[pos_seg, 0]

            if port.negative_segment_idx >= 0:
                neg_seg = port.negative_segment_idx
                neg_end_node = self.connectivity[neg_seg, 1]
                # Current in at positive, out at negative
                rhs[pos_start_node] += 1.0
                rhs[neg_end_node] -= 1.0
            else:
                # Ground return: current in at positive start node
                rhs[pos_start_node] += 1.0

            # Ground node constraint
            rhs[ground_node] = 0.0

            # Solve
            x = np.linalg.solve(MNA, rhs)

            # Extract node voltages
            V_nodes = x[:N]

            # Compute port voltages for all ports
            for p_meas in range(P):
                port_m = self.ports[p_meas]
                pos_seg_m = port_m.positive_segment_idx
                v_pos = V_nodes[self.connectivity[pos_seg_m, 0]]

                if port_m.negative_segment_idx >= 0:
                    neg_seg_m = port_m.negative_segment_idx
                    v_neg = V_nodes[self.connectivity[neg_seg_m, 1]]
                    Z_port[p_meas, p_exc] = v_pos - v_neg
                else:
                    # Ground return: V_port = V_pos - V_ground = V_pos
                    Z_port[p_meas, p_exc] = v_pos

        return Z_port

    def _find_ground_node(self) -> int:
        """Find the ground reference node.

        For ground-return ports, uses the last node of the last trace.
        For differential ports, uses an arbitrary node not connected to ports.

        Returns
        -------
        node_idx : int
        """
        # Use the end node of the last segment as ground
        # This is the "far end" of the last trace
        last_seg_idx = len(self.connectivity) - 1
        return int(self.connectivity[last_seg_idx, 1])

    def solve_sweep(
        self,
        frequencies: np.ndarray,
        segments: List[TraceSegment],
        Z0: float = 50.0,
    ) -> List[NetworkResult]:
        """Frequency sweep returning NetworkResult objects.

        Lp is frequency-independent (computed once during PEECExtractor init).
        R(f) is recomputed at each frequency.

        Parameters
        ----------
        frequencies : ndarray
            Frequencies to solve at (Hz).
        segments : list of TraceSegment
        Z0 : float
            Reference impedance for S-parameters.

        Returns
        -------
        results : list of NetworkResult
            One per frequency. Compatible with InductorCharacterization.
        """
        frequencies = np.atleast_1d(frequencies)
        port_names = [p.name for p in self.ports]

        results = []
        for freq in frequencies:
            Z_port = self.solve_impedance(freq, segments)
            results.append(NetworkResult(
                frequency=float(freq),
                Z_matrix=Z_port,
                port_names=port_names,
                Z0=Z0,
            ))

        return results
