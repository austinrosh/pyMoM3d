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

from .port import Port, GroundVia
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
    ground_vias : list of GroundVia, optional
        Via-to-ground connections.  Each via adds a low-impedance path from
        specified basis functions to the Green's function ground plane.
        Applied to Z_sys after assembly, before the solve.
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
    low_freq_stabilization : str, optional
        Low-frequency stabilization strategy.  One of:

        - ``'auto'`` (default): estimate kD per frequency; use A-EFIE when
          kD < 0.5, standard EFIE otherwise.
        - ``'aefie'``: always use the Augmented EFIE.
        - ``'loop_star'``: loop-star basis (retained for backward compat).
        - ``'none'``: standard EFIE direct solve, no stabilization.

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

    Inductor extraction with ground via and loop-star::

        via = GroundVia('inner', basis_indices=via_edges)
        extractor = NetworkExtractor(sim, [port], ground_vias=[via],
                                     low_freq_stabilization='loop_star')
        results = extractor.extract(frequencies)
    """

    def __init__(
        self,
        simulation,
        ports: List[Port],
        ground_vias: Optional[List[GroundVia]] = None,
        Z0: float = 50.0,
        store_currents: bool = False,
        use_lu_cache: bool = False,
        low_freq_stabilization: str = 'none',
        conductor=None,
        hybrid_basis=None,
    ):
        self.sim = simulation
        self.ports = list(ports)
        self.ground_vias = list(ground_vias) if ground_vias else []
        self.Z0 = float(Z0)
        self.store_currents = store_currents
        self.use_lu_cache = use_lu_cache
        self.conductor = conductor
        self.hybrid_basis = hybrid_basis
        self._validate_ports()

        # Precompute Gram matrix if conductor properties are specified
        # (mesh-dependent, reused across frequencies)
        self._gram_matrix = None
        if self.conductor is not None:
            from ..mom.surface_impedance import build_gram_matrix
            self._gram_matrix = build_gram_matrix(
                self.sim.basis, self.sim.mesh,
                quad_order=self.sim.config.quad_order,
            )

        self.low_freq_stabilization = low_freq_stabilization

        _valid = ('auto', 'aefie', 'loop_star', 'none')
        if self.low_freq_stabilization not in _valid:
            raise ValueError(
                f"low_freq_stabilization must be one of {_valid}, "
                f"got '{self.low_freq_stabilization}'"
            )

        # Precompute loop-star basis (mesh-dependent, reused across freqs)
        self._ls_P = None
        self._ls_n_loops = None
        if self.low_freq_stabilization == 'loop_star':
            from ..mom.loop_star import build_loop_star_basis
            self._ls_P, self._ls_n_loops = build_loop_star_basis(
                self.sim.basis, self.sim.mesh
            )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_ports(self) -> None:
        if self.hybrid_basis is not None:
            N = self.hybrid_basis.num_total
        else:
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
                _gf = LayeredGreensFunction(
                    self.sim.config.layer_stack, freq,
                    source_layer_name=self.sim.config.source_layer_name,
                    backend=getattr(self.sim.config, 'gf_backend', 'auto'),
                )
                k   = complex(_gf.wavenumber)
                eta = complex(_gf.wave_impedance)
            else:
                k   = 2.0 * np.pi * freq / c0
                eta = eta0

            # Resolve stabilization strategy for this frequency
            stab = self._resolve_stabilization(k)

            # Determine which basis object to use for port operations
            if self.hybrid_basis is not None:
                from ..wire.hybrid import HybridBasisAdapter, fill_hybrid_matrix
                _basis_for_ports = HybridBasisAdapter(self.hybrid_basis)
            else:
                _basis_for_ports = self.sim.basis

            # 2. Build (N, P) RHS matrix — one column per port
            V_all = np.column_stack([
                p.build_excitation_vector(_basis_for_ports) for p in self.ports
            ])

            # 3. Solve
            if stab == 'aefie':
                I_all = self._solve_aefie(freq, k, eta, V_all)
            elif stab == 'loop_star':
                I_all = self._solve_loop_star(freq, k, eta, V_all)
            else:
                # Standard EFIE solve
                op = self._make_operator(freq)

                if self.hybrid_basis is not None:
                    Z_sys = fill_hybrid_matrix(
                        op,
                        self.sim.basis,
                        self.sim.mesh,
                        self.hybrid_basis.wire_basis,
                        self.hybrid_basis.wire_mesh,
                        k, eta,
                        quad_order=self.sim.config.quad_order,
                        near_threshold=self.sim.config.near_threshold,
                        backend=self.sim.config.backend,
                        junctions=self.hybrid_basis.junctions or None,
                    )
                else:
                    Z_sys = fill_matrix(
                        op,
                        self.sim.basis,
                        self.sim.mesh,
                        k, eta,
                        quad_order=self.sim.config.quad_order,
                        near_threshold=self.sim.config.near_threshold,
                        backend=self.sim.config.backend,
                    )

                for via in self.ground_vias:
                    via.apply_to_matrix(Z_sys, self.sim.basis)

                # Apply surface impedance BC if conductor is specified
                if self.conductor is not None:
                    from ..mom.surface_impedance import apply_surface_impedance
                    apply_surface_impedance(
                        Z_sys, self.sim.basis, self.sim.mesh,
                        self.conductor, freq,
                        gram_matrix=self._gram_matrix,
                    )

                I_all = self._solve(Z_sys, V_all)    # (N, P)
                try:
                    self._last_cond = float(np.linalg.cond(Z_sys))
                except np.linalg.LinAlgError:
                    self._last_cond = float('inf')

            # 4. Z-matrix extraction
            #
            # MoM delta-gap excitation shorts non-excited ports (V_q = 0),
            # so the terminal currents give the Y-matrix directly:
            #   Y[q, p] = I_term[q, p] / V_ref_p
            # The Z-matrix is obtained by matrix inversion: Z = inv(Y).
            #
            # For a single port this reduces to Z = V_ref / I_term (scalar).
            P = len(self.ports)
            Y_mat = np.zeros((P, P), dtype=np.complex128)
            for q, port_q in enumerate(self.ports):
                for p in range(P):
                    I_term = port_q.terminal_current(I_all[:, p], _basis_for_ports)
                    V_ref  = self.ports[p].V_ref
                    Y_mat[q, p] = I_term / V_ref

            if P == 1:
                # Single port: Z = 1/Y (avoid unnecessary matrix ops)
                if abs(Y_mat[0, 0]) > 1e-30:
                    Z_mat = np.array([[1.0 / Y_mat[0, 0]]], dtype=np.complex128)
                else:
                    Z_mat = np.array([[np.inf + 0j]], dtype=np.complex128)
            else:
                try:
                    Z_mat = np.linalg.inv(Y_mat)
                except np.linalg.LinAlgError:
                    Z_mat = np.full((P, P), np.inf + 0j)

            try:
                cond = float(self._last_cond)
            except (AttributeError, TypeError):
                cond = float('inf')
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
            gf = LayeredGreensFunction(
                self.sim.config.layer_stack, frequency,
                source_layer_name=self.sim.config.source_layer_name,
                backend=getattr(self.sim.config, 'gf_backend', 'auto'),
            )
            return MultilayerEFIEOperator(gf)
        return EFIEOperator()

    def _resolve_stabilization(self, k: float) -> str:
        """Decide which stabilization to use for this frequency.

        Returns ``'aefie'``, ``'loop_star'``, or ``'none'``.
        """
        if self.low_freq_stabilization == 'none':
            return 'none'
        if self.low_freq_stabilization in ('aefie', 'loop_star'):
            return self.low_freq_stabilization
        # 'auto': estimate kD and pick
        from ..mom.aefie import estimate_kD
        kD = estimate_kD(self.sim.mesh, abs(k))
        return 'aefie' if kD < 0.5 else 'none'

    def _solve_aefie(
        self,
        freq: float,
        k: float,
        eta: float,
        V_all: np.ndarray,
    ) -> np.ndarray:
        """A-EFIE solve path: assemble Z_A, G_s, D, then augmented solve.

        Automatically detects multilayer configurations and uses the
        appropriate operator and Green's function backend.
        """
        from ..mom.assembly import fill_matrix
        from ..mom.aefie import (
            build_divergence_matrix,
            fill_scalar_green_matrix,
            solve_aefie,
        )

        # Determine if multilayer
        is_multilayer = (self.sim.config.layer_stack is not None
                         and freq is not None)

        greens_fn = None
        if is_multilayer:
            from ..greens.layered import LayeredGreensFunction
            from ..mom.operators.efie_layered import MultilayerEFIEOperator
            greens_fn = LayeredGreensFunction(
                self.sim.config.layer_stack, freq,
                source_layer_name=self.sim.config.source_layer_name,
                backend=getattr(self.sim.config, 'gf_backend', 'auto'),
            )
            op_vp = MultilayerEFIEOperator(greens_fn, a_only=True)
            # Use the layered GF's k and eta
            k = complex(greens_fn.wavenumber)
            eta = complex(greens_fn.wave_impedance)
        else:
            from ..mom.operators.vector_potential import VectorPotentialOperator
            op_vp = VectorPotentialOperator()

        # Z_A: vector-potential-only matrix (N x N)
        Z_A = fill_matrix(
            op_vp,
            self.sim.basis,
            self.sim.mesh,
            k, eta,
            quad_order=self.sim.config.quad_order,
            near_threshold=self.sim.config.near_threshold,
            backend=self.sim.config.backend,
        )

        # D: sparse divergence matrix (T x N)
        D = build_divergence_matrix(self.sim.basis, self.sim.mesh)

        # G_s: triangle-to-triangle scalar Green's function (T x T)
        # For multilayer, adds smooth correction from layered GF
        G_s = fill_scalar_green_matrix(
            self.sim.mesh,
            k,
            quad_order=self.sim.config.quad_order,
            near_threshold=self.sim.config.near_threshold,
            backend=self.sim.config.backend,
            greens_fn=greens_fn,
        )

        # Solve augmented system
        I_all = solve_aefie(Z_A, G_s, D, V_all, k, eta)

        # Condition number of the augmented system
        try:
            D_dense = D.toarray()
            N = Z_A.shape[0]
            T = G_s.shape[0]
            Z_aug = np.zeros((N + T, N + T), dtype=np.complex128)
            Z_aug[:N, :N] = Z_A
            Z_aug[:N, N:] = D_dense.T @ G_s
            Z_aug[N:, :N] = -D_dense
            np.fill_diagonal(Z_aug[N:, N:], 1j * k / eta)
            self._last_cond = float(np.linalg.cond(Z_aug))
        except Exception:
            self._last_cond = float('inf')

        return I_all

    def _solve_loop_star(
        self,
        freq: float,
        k: float,
        eta: float,
        V_all: np.ndarray,
    ) -> np.ndarray:
        """Loop-star solve path."""
        from ..mom.assembly import fill_matrix
        from ..mom.loop_star import solve_loop_star_hybrid

        op = self._make_operator(freq)
        Z_sys = fill_matrix(
            op,
            self.sim.basis,
            self.sim.mesh,
            k, eta,
            quad_order=self.sim.config.quad_order,
            near_threshold=self.sim.config.near_threshold,
            backend=self.sim.config.backend,
        )
        for via in self.ground_vias:
            via.apply_to_matrix(Z_sys, self.sim.basis)

        # Build Z_A (vector-potential only) — use multilayer operator when
        # a layer stack is present so k can be complex.
        is_multilayer = (self.sim.config.layer_stack is not None
                         and self.sim.config.source_layer_name is not None
                         and freq is not None)
        if is_multilayer:
            from ..greens.layered import LayeredGreensFunction
            from ..mom.operators.efie_layered import MultilayerEFIEOperator
            greens_fn = LayeredGreensFunction(
                self.sim.config.layer_stack, freq,
                source_layer_name=self.sim.config.source_layer_name,
                backend=getattr(self.sim.config, 'gf_backend', 'auto'),
            )
            op_vp = MultilayerEFIEOperator(greens_fn, a_only=True)
        else:
            from ..mom.operators.vector_potential import VectorPotentialOperator
            op_vp = VectorPotentialOperator()

        Z_A = fill_matrix(
            op_vp,
            self.sim.basis,
            self.sim.mesh,
            k, eta,
            quad_order=self.sim.config.quad_order,
            near_threshold=self.sim.config.near_threshold,
            backend=self.sim.config.backend,
        )

        I_all = solve_loop_star_hybrid(
            Z_sys, V_all, self._ls_P, self._ls_n_loops,
            Z_A=Z_A,
        )

        try:
            self._last_cond = float(np.linalg.cond(Z_sys))
        except np.linalg.LinAlgError:
            self._last_cond = float('inf')

        return I_all

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
