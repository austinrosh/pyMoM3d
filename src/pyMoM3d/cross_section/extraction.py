"""Transmission line parameter extraction from 2D cross-section solutions.

Implements the two-solve method:
  1. Solve with actual dielectrics → C_diel
  2. Solve with ε_r = 1 everywhere → C_vac
  3. L = 1/(c₀²·C_vac)
  4. Z₀ = 1/(c₀·√(C_diel·C_vac))
  5. ε_eff = C_diel / C_vac

For multi-conductor lines, N independent solves are performed per
dielectric condition, yielding full C and L matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..utils.constants import c0, mu0, eps0
from .grid import NonUniformGrid, build_grid_for_cross_section
from .geometry import CrossSection, Conductor, DielectricRegion
from .solver import LaplaceSolver2D


@dataclass
class CrossSectionResult:
    """Transmission line parameters from 2D cross-section analysis.

    Parameters
    ----------
    Z0 : float
        Characteristic impedance (Ohm).
    eps_eff : float
        Effective relative permittivity.
    v_phase : float
        Phase velocity (m/s).
    C_pul : float
        Capacitance per unit length (F/m).
    L_pul : float
        Inductance per unit length (H/m).
    C_matrix : ndarray or None
        Multi-conductor capacitance matrix (F/m).
    L_matrix : ndarray or None
        Multi-conductor inductance matrix (H/m).
    C_diel : float
        Capacitance with actual dielectrics (F/m).
    C_vac : float
        Capacitance with ε_r = 1 (F/m).
    grid_shape : tuple of int
        Grid dimensions used.
    """

    Z0: float
    eps_eff: float
    v_phase: float
    C_pul: float
    L_pul: float
    C_matrix: Optional[np.ndarray] = None
    L_matrix: Optional[np.ndarray] = None
    C_diel: float = 0.0
    C_vac: float = 0.0
    grid_shape: tuple = (0, 0)

    def gamma(self, freq: float) -> complex:
        """Propagation constant at given frequency (lossless TEM).

        Parameters
        ----------
        freq : float
            Frequency (Hz).

        Returns
        -------
        gamma : complex
            j·β = j·ω·√(L·C)
        """
        omega = 2.0 * np.pi * freq
        return 1j * omega * np.sqrt(self.L_pul * self.C_pul)

    def beta(self, freq: float) -> float:
        """Phase constant (rad/m)."""
        omega = 2.0 * np.pi * freq
        return omega * np.sqrt(self.L_pul * self.C_pul)


def extract_tl_params(
    cross_section: CrossSection,
    grid: Optional[NonUniformGrid] = None,
    signal_conductor: Optional[str] = None,
    **solver_kwargs,
) -> CrossSectionResult:
    """Extract TL parameters via the two-solve method.

    Parameters
    ----------
    cross_section : CrossSection
        Geometry with at least one signal conductor (voltage != 0).
    grid : NonUniformGrid, optional
        Computational grid.  If None, auto-built from geometry.
    signal_conductor : str, optional
        Name of the signal conductor.  If None, uses the first signal conductor.
    **solver_kwargs
        Passed to LaplaceSolver2D (margin_factor, base_cells).

    Returns
    -------
    CrossSectionResult
    """
    signals = cross_section.signal_conductors()
    if not signals:
        raise ValueError("CrossSection has no signal conductors (voltage != 0)")

    if signal_conductor is not None:
        sig = next((c for c in signals if c.name == signal_conductor), None)
        if sig is None:
            raise ValueError(f"No signal conductor named '{signal_conductor}'")
    else:
        sig = signals[0]

    solver = LaplaceSolver2D(cross_section, grid=grid, **solver_kwargs)

    # Solve 1: actual dielectrics
    sol_diel = solver.solve()
    Q_diel = solver.integrate_charge(sol_diel, sig)
    C_diel = abs(Q_diel / sig.voltage)

    # Solve 2: vacuum (eps_r = 1 everywhere)
    cs_vac = cross_section.vacuum_copy()
    eps_vac = cs_vac.eps_r_map(solver.grid)
    sol_vac = solver.solve(eps_r_map=eps_vac)
    Q_vac = solver.integrate_charge(sol_vac, sig)
    C_vac = abs(Q_vac / sig.voltage)

    if C_vac <= 0 or C_diel <= 0:
        raise RuntimeError(
            f"Non-positive capacitance: C_diel={C_diel:.4e}, C_vac={C_vac:.4e}. "
            f"Check geometry and grid resolution."
        )

    # Extract TL parameters
    L_pul = 1.0 / (c0**2 * C_vac)
    eps_eff = C_diel / C_vac
    Z0 = 1.0 / (c0 * np.sqrt(C_diel * C_vac))
    v_phase = c0 / np.sqrt(eps_eff)

    return CrossSectionResult(
        Z0=float(Z0),
        eps_eff=float(eps_eff),
        v_phase=float(v_phase),
        C_pul=float(C_diel),
        L_pul=float(L_pul),
        C_diel=float(C_diel),
        C_vac=float(C_vac),
        grid_shape=solver.grid.shape,
    )


def extract_multiconductor_params(
    cross_section: CrossSection,
    grid: Optional[NonUniformGrid] = None,
    **solver_kwargs,
) -> CrossSectionResult:
    """Multi-conductor extraction returning full C and L matrices.

    For N signal conductors, performs N independent solves per dielectric
    condition (actual + vacuum).  In solve k, conductor k is set to 1V
    and all other conductors to 0V.

    Parameters
    ----------
    cross_section : CrossSection
    grid : NonUniformGrid, optional
    **solver_kwargs
        Passed to LaplaceSolver2D.

    Returns
    -------
    CrossSectionResult
        With C_matrix and L_matrix populated.
    """
    import copy

    signals = cross_section.signal_conductors()
    N = len(signals)
    if N == 0:
        raise ValueError("No signal conductors")

    solver = LaplaceSolver2D(cross_section, grid=grid, **solver_kwargs)
    grid_used = solver.grid

    C_diel_mat = np.zeros((N, N), dtype=np.float64)
    C_vac_mat = np.zeros((N, N), dtype=np.float64)

    for k in range(N):
        # Set conductor k to 1V, all others to 0V
        cs_k = copy.deepcopy(cross_section)
        for i, sig in enumerate(cs_k.signal_conductors()):
            sig.voltage = 1.0 if i == k else 0.0

        solver_k = LaplaceSolver2D(cs_k, grid=grid_used)

        # Solve with dielectric
        sol = solver_k.solve()
        for i, sig in enumerate(signals):
            Q = solver_k.integrate_charge(sol, sig)
            C_diel_mat[i, k] = Q  # Already includes eps0

        # Solve vacuum
        cs_vac = cs_k.vacuum_copy()
        eps_vac = cs_vac.eps_r_map(grid_used)
        sol_vac = solver_k.solve(eps_r_map=eps_vac)
        for i, sig in enumerate(signals):
            Q = solver_k.integrate_charge(sol_vac, sig)
            C_vac_mat[i, k] = Q

    # L = mu0*eps0 * C_vac^{-1}  (only valid for TEM/quasi-TEM)
    L_mat = mu0 * eps0 * np.linalg.inv(C_vac_mat)

    # Scalar results from first signal conductor
    C_diel_1 = C_diel_mat[0, 0]
    C_vac_1 = C_vac_mat[0, 0]
    L_1 = L_mat[0, 0]
    eps_eff = C_diel_1 / C_vac_1 if C_vac_1 > 0 else 1.0
    Z0 = np.sqrt(L_1 / C_diel_1) if C_diel_1 > 0 else float('inf')
    v_phase = 1.0 / np.sqrt(L_1 * C_diel_1) if C_diel_1 > 0 and L_1 > 0 else c0

    return CrossSectionResult(
        Z0=float(Z0),
        eps_eff=float(eps_eff),
        v_phase=float(v_phase),
        C_pul=float(C_diel_1),
        L_pul=float(L_1),
        C_matrix=C_diel_mat,
        L_matrix=L_mat,
        C_diel=float(C_diel_1),
        C_vac=float(C_vac_1),
        grid_shape=grid_used.shape,
    )


def compute_reference_impedance(
    layer_stack,
    strip_width: float,
    strip_z: Optional[float] = None,
    source_layer_name: Optional[str] = None,
    **solver_kwargs,
) -> CrossSectionResult:
    """Compute TL reference impedance from a LayerStack and strip geometry.

    Auto-detects microstrip vs stripline from the layer stack topology
    and builds the appropriate 2D cross-section for the two-solve extraction.

    This is the bridge between the 3D MoM simulation setup (LayerStack,
    strip geometry) and the 2D cross-section solver that provides the
    authoritative Z₀ and ε_eff for S-parameter normalization.

    Parameters
    ----------
    layer_stack : LayerStack
        Stratified medium definition (same object used by the 3D solver).
    strip_width : float
        Width of the signal conductor (m).
    strip_z : float, optional
        z-coordinate of the strip conductor.  If None, inferred from
        ``source_layer_name`` (uses the top of that layer).
    source_layer_name : str, optional
        Name of the dielectric layer containing the strip.  Used to
        determine strip_z if not provided explicitly.
    **solver_kwargs
        Passed to ``extract_tl_params`` (e.g., ``base_cells``,
        ``margin_factor``).

    Returns
    -------
    CrossSectionResult
        Contains Z0, eps_eff, v_phase, C_pul, L_pul, gamma(f), beta(f).

    Examples
    --------
    >>> from pyMoM3d import Layer, LayerStack
    >>> from pyMoM3d.cross_section import compute_reference_impedance
    >>> stack = LayerStack([
    ...     Layer('pec', z_bot=-np.inf, z_top=0, eps_r=1.0, is_pec=True),
    ...     Layer('FR4', z_bot=0, z_top=1.6e-3, eps_r=4.4),
    ...     Layer('air', z_bot=1.6e-3, z_top=np.inf, eps_r=1.0),
    ... ])
    >>> tl = compute_reference_impedance(stack, strip_width=3.06e-3,
    ...                                   source_layer_name='FR4')
    >>> print(f"Z0 = {tl.Z0:.1f} Ohm")
    """
    # Determine strip z-coordinate
    if strip_z is None:
        if source_layer_name is not None:
            source_layer = layer_stack.get_layer(source_layer_name)
            strip_z = source_layer.z_top
        else:
            raise ValueError(
                "Either strip_z or source_layer_name must be provided"
            )

    # Classify layers relative to the strip
    pec_layers = [L for L in layer_stack.layers if L.is_pec]
    finite_layers = [
        L for L in layer_stack.layers
        if not L.is_pec
        and np.isfinite(L.z_bot) and np.isfinite(L.z_top)
    ]

    # Find PEC ground planes
    pec_below = [
        L for L in pec_layers
        if np.isfinite(L.z_top) and L.z_top <= strip_z + 1e-15
    ]
    pec_above = [
        L for L in pec_layers
        if np.isfinite(L.z_bot) and L.z_bot >= strip_z - 1e-15
    ]

    # Ground extent: make it wide enough to act as infinite
    ground_extent = 100.0 * strip_width

    conductors = []
    dielectrics = []

    # Add ground planes as zero-thickness conductors at their interface z
    for L in pec_below:
        z_gnd = L.z_top
        conductors.append(Conductor(
            name=f'ground_{L.name}',
            x_min=-ground_extent, x_max=ground_extent,
            y_min=z_gnd, y_max=z_gnd,
            voltage=0.0,
        ))
    for L in pec_above:
        z_gnd = L.z_bot
        conductors.append(Conductor(
            name=f'ground_{L.name}',
            x_min=-ground_extent, x_max=ground_extent,
            y_min=z_gnd, y_max=z_gnd,
            voltage=0.0,
        ))

    # Signal strip (zero-thickness at strip_z)
    conductors.append(Conductor(
        name='signal',
        x_min=-strip_width / 2.0, x_max=strip_width / 2.0,
        y_min=strip_z, y_max=strip_z,
        voltage=1.0,
    ))

    # Add dielectric regions from finite layers with eps_r != 1
    for L in finite_layers:
        eps_r = float(np.real(L.eps_r))
        if abs(eps_r - 1.0) > 1e-10:
            dielectrics.append(DielectricRegion(
                name=L.name,
                x_min=-ground_extent, x_max=ground_extent,
                y_min=L.z_bot, y_max=L.z_top,
                eps_r=eps_r,
            ))

    if not conductors:
        raise ValueError(
            "No PEC ground planes found in layer stack. "
            "compute_reference_impedance requires at least one PEC layer."
        )

    cs = CrossSection(conductors=conductors, dielectric_regions=dielectrics)
    return extract_tl_params(cs, **solver_kwargs)
