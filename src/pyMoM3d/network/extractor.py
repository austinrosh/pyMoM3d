"""NetworkExtractor: multi-port network parameter extraction.

Assembles the system impedance matrix Z_sys once per frequency, then solves
P right-hand sides simultaneously (one per port) using a single LAPACK LU
factorisation via ``np.linalg.solve(Z_sys, V_all)``.  The resulting P×P
Z-matrix is extracted from the current solutions.

Performance
-----------
``np.linalg.solve(Z_sys, V_all)`` with V_all of shape (N, P) performs one
LU factorisation and P back-substitutions via LAPACK ``zgesv`` — no custom
C++ kernels are needed for the linear-algebra step.  For P > ~20 ports or
N > ~10k, use ``use_lu_cache=True`` to pre-factor Z_sys with
``scipy.linalg.lu_factor`` and reuse it across port solves.

Physical notes
--------------
See ``port.py`` for a full discussion of port reference planes, return paths,
and the delta-gap model approximations.  In brief:

* All extracted Z/Y/S parameters are conditional on the port geometry and
  reference plane.
* For closed PEC shells, use ``formulation='CFIE'`` in SimulationConfig to
  avoid spurious interior resonances in the Z-matrix.
* For multi-conductor structures, both conductors must be meshed — absent
  ground planes produce non-physical (non-passive) results.
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

from .port import Port
from .network_result import NetworkResult


class NetworkExtractor:
    """Multi-port network parameter extractor.

    Wraps a fully-initialised :class:`~pyMoM3d.simulation.Simulation` object
    (mesh + RWG basis already computed) and a list of
    :class:`~pyMoM3d.network.Port` objects.  Assembles the system impedance
    matrix once per frequency and solves P right-hand sides simultaneously.

    Parameters
    ----------
    simulation : Simulation
        Fully initialised simulation (mesh and RWG basis computed, operator
        formulation set via ``simulation.config.formulation``).
    ports : list of Port
        Port definitions.  Each port maps one or more RWG basis indices to a
        circuit terminal.  Order determines the row/column ordering of the
        extracted Z-matrix.
    Z0 : float, optional
        Reference impedance for S-parameter normalisation (Ω).  Equal, real
        value applied to all ports.  Default 50 Ω.
    store_currents : bool, optional
        If True, ``NetworkResult.I_solutions`` (shape N×P) is populated with
        all P current-coefficient vectors.  Useful for field visualisation,
        coupling debugging, and symmetry verification.  Default False.
    use_lu_cache : bool, optional
        If True, use ``scipy.linalg.lu_factor`` / ``lu_solve`` instead of
        ``np.linalg.solve`` for the P-RHS solve.  Saves memory for large P
        but requires scipy.  Default False (``np.linalg.solve`` is used,
        which calls the same LAPACK routine and handles the (N, P) system
        in one call).

    Examples
    --------
    Single-port self-impedance::

        sim = Simulation(config, mesh=mesh)
        port = Port.from_x_plane(sim.mesh, sim.basis, x_coord=0.0)
        extractor = NetworkExtractor(sim, [port])
        [result] = extractor.extract()
        Z_in = result.Z_matrix[0, 0]

    Two-port mutual impedance sweep::

        extractor = NetworkExtractor(sim, [port1, port2])
        results = extractor.extract(frequencies)
        Z12 = [r.Z_matrix[0, 1] for r in results]
        S11 = [r.S_matrix[0, 0] for r in results]
    """

    def __init__(
        self,
        simulation,
        ports: List[Port],
        Z0: float = 50.0,
        store_currents: bool = False,
        use_lu_cache: bool = False,
    ):
        self.sim = simulation
        self.ports = list(ports)
        self.Z0 = float(Z0)
        self.store_currents = store_currents
        self.use_lu_cache = use_lu_cache
        self._validate_ports()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_ports(self) -> None:
        N = self.sim.basis.num_basis
        for port in self.ports:
            for idx in port.feed_basis_indices:
                if not (0 <= idx < N):
                    raise ValueError(
                        f"Port '{port.name}': feed basis index {idx} out of "
                        f"range [0, {N})"
                    )
            for idx in port.return_basis_indices:
                if not (0 <= idx < N):
                    raise ValueError(
                        f"Port '{port.name}': return basis index {idx} out of "
                        f"range [0, {N})"
                    )

    # ------------------------------------------------------------------
    # Main extraction entry point
    # ------------------------------------------------------------------

    def extract(
        self,
        frequencies: Union[None, float, List[float]] = None,
    ) -> List[NetworkResult]:
        """Extract network parameters at one or more frequencies.

        One ``fill_matrix`` call and one (N, P) solve per frequency.

        Parameters
        ----------
        frequencies : float or list of float, optional
            Frequencies (Hz) at which to extract parameters.  If None,
            uses ``simulation.config.frequency``.

        Returns
        -------
        list of NetworkResult
            One entry per frequency, in the same order as ``frequencies``.
        """
        from ..utils.constants import c0, eta0
        from ..mom.assembly import fill_matrix

        if frequencies is None:
            frequencies = [self.sim.config.frequency]
        elif np.isscalar(frequencies):
            frequencies = [float(frequencies)]
        else:
            frequencies = list(frequencies)

        results = []
        for freq in frequencies:
            if self.sim.config.layer_stack is not None:
                from ..greens.layered import LayeredGreensFunction
                _gf = LayeredGreensFunction(self.sim.config.layer_stack, freq)
                k   = complex(_gf.wavenumber)
                eta = complex(_gf.wave_impedance)
            else:
                k   = 2.0 * np.pi * freq / c0
                eta = eta0

            # 1. Assemble Z_sys — one fill per frequency
            op = self._make_operator(freq)
            Z_sys = fill_matrix(
                op,
                self.sim.basis,
                self.sim.mesh,
                k,
                eta,
                quad_order=self.sim.config.quad_order,
                near_threshold=self.sim.config.near_threshold,
                backend=self.sim.config.backend,
            )

            # 2. Build (N, P) RHS matrix — one column per port
            V_all = np.column_stack([
                p.build_excitation_vector(self.sim.basis) for p in self.ports
            ])

            # 3. Batched solve: one LU, P back-substitutions via LAPACK
            I_all = self._solve(Z_sys, V_all)    # (N, P)

            # 4. Z-matrix extraction
            P = len(self.ports)
            Z_mat = np.zeros((P, P), dtype=np.complex128)
            for q, port_q in enumerate(self.ports):
                for p in range(P):
                    I_term = port_q.terminal_current(I_all[:, p], self.sim.basis)
                    V_ref  = self.ports[p].V_ref
                    if abs(I_term) > 1e-30:
                        Z_mat[q, p] = V_ref / I_term
                    else:
                        Z_mat[q, p] = np.inf + 0j

            cond = float(np.linalg.cond(Z_sys))
            results.append(NetworkResult(
                frequency=freq,
                Z_matrix=Z_mat,
                port_names=[p.name for p in self.ports],
                Z0=self.Z0,
                condition_number=cond,
                I_solutions=I_all.copy() if self.store_currents else None,
            ))

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_operator(self, frequency: float = None):
        """Return the appropriate impedance operator from the Simulation config."""
        from ..mom.operators import EFIEOperator, MFIEOperator, CFIEOperator
        f = self.sim.config.formulation
        if f == 'MFIE':
            return MFIEOperator()
        if f == 'CFIE':
            return CFIEOperator(alpha=self.sim.config.cfie_alpha)
        if self.sim.config.layer_stack is not None and frequency is not None:
            from ..greens.layered import LayeredGreensFunction
            from ..mom.operators.efie_layered import MultilayerEFIEOperator
            gf = LayeredGreensFunction(self.sim.config.layer_stack, frequency)
            return MultilayerEFIEOperator(gf)
        return EFIEOperator()

    def _solve(self, Z_sys: np.ndarray, V_all: np.ndarray) -> np.ndarray:
        """Solve Z_sys @ I_all = V_all for shape (N, P).

        Uses scipy LU cache when ``use_lu_cache=True``, otherwise
        ``np.linalg.solve`` (one LAPACK call, handles multiple RHS natively).
        """
        if self.use_lu_cache:
            import scipy.linalg
            lu, piv = scipy.linalg.lu_factor(Z_sys)
            return scipy.linalg.lu_solve((lu, piv), V_all)
        return np.linalg.solve(Z_sys, V_all)
