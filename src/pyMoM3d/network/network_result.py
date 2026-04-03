"""NetworkResult: container for multi-port network parameters.

Stores the raw impedance matrix (Z) extracted at the RWG edge midpoints and
provides Y and S matrices as on-demand properties.  De-embedding helpers
allow post-processing to shift the reference plane or subtract lumped gap
parasitics.

Raw vs. de-embedded parameters
-------------------------------
``Z_matrix`` (and S/Y derived from it) are the **raw** values defined at the
physical location of the port gap — the RWG edge midpoints.  They include:

* The two-terminal device response
* Fringing inductance / capacitance of the gap itself
* Any transmission-line offset between the gap and the device plane

``deembed_phase`` shifts the reference plane by a user-supplied electrical
length per port (corrects for transmission-line offsets).

``correct_port_parasitics`` (Phase 2, not yet implemented) subtracts
approximate lumped-element gap parasitics when the gap geometry is known.

**Never compare raw parameters from simulations with different gap sizes or
port locations — the reference planes differ.**

S-parameter convention
----------------------
Power-wave S-parameters with equal, real Z₀ for all ports:

    S = (Z − Z₀ I)(Z + Z₀ I)⁻¹

Per-port reference impedances are not supported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class NetworkResult:
    """Multi-port network parameters at a single frequency.

    Parameters
    ----------
    frequency : float
        Frequency (Hz).
    Z_matrix : ndarray, shape (P, P), complex128
        Raw impedance matrix at the RWG edge midpoints (open-circuit
        definition, V_ref = 1 V per port).
    port_names : list of str
        Port labels, length P.
    Z0 : float, optional
        Reference impedance for S-parameter normalisation (Ω).  Same value
        used for all ports.  Default 50 Ω.
    condition_number : float, optional
        Condition number of the system impedance matrix Z_sys used during
        extraction.  Useful for diagnosing near-resonance behaviour.
    I_solutions : ndarray, shape (N, P), optional
        RWG current coefficient vectors for each of the P port excitations.
        Stored only when ``NetworkExtractor(store_currents=True)``.  Enables:
        - Field / current-density visualisation per port excitation
        - Coupling behaviour debugging
        - Verification of current distribution symmetry
        - Post-hoc far-field or near-field computation
    """

    frequency: float
    Z_matrix: np.ndarray
    port_names: List[str]
    Z0: float = 50.0
    condition_number: Optional[float] = None
    I_solutions: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Derived network parameters (computed on demand)
    # ------------------------------------------------------------------

    @property
    def Y_matrix(self) -> np.ndarray:
        """Admittance matrix Y = Z⁻¹, shape (P, P)."""
        return np.linalg.inv(self.Z_matrix)

    @property
    def S_matrix(self) -> np.ndarray:
        """Power-wave S-matrix S = (Z − Z₀ I)(Z + Z₀ I)⁻¹, shape (P, P).

        Assumes equal, real reference impedance Z₀ at all ports.
        S is ill-conditioned (not a numerical error) when any eigenvalue of
        Z ≈ −Z₀ (physically: a purely reactive load matching −Z₀).
        """
        P = len(self.Z_matrix)
        I_id = np.eye(P, dtype=np.complex128)
        Z_plus  = self.Z_matrix + self.Z0 * I_id
        Z_minus = self.Z_matrix - self.Z0 * I_id
        return np.linalg.solve(Z_plus, Z_minus)

    # ------------------------------------------------------------------
    # De-embedding helpers
    # ------------------------------------------------------------------

    def deembed_phase(self, delta_theta: list) -> 'NetworkResult':
        """Shift reference plane by electrical lengths Δθ_p at each port.

        Corrects for a transmission-line section of electrical length
        Δθ_p = k · Δl_p between the raw port plane and the desired reference
        plane at port p.

        The de-embedded S-matrix is

            S_deemb = Φ · S_raw · Φ,    Φ = diag(exp(−j Δθ_p))

        and the corresponding Z-matrix is recovered via

            Z_deemb = Z₀ (I + S_deemb)(I − S_deemb)⁻¹

        Parameters
        ----------
        delta_theta : list of float, length P
            Electrical lengths to de-embed (radians).  Positive values move
            the reference plane toward the device (away from the port gap).

        Returns
        -------
        NetworkResult
            New result with de-embedded Z_matrix.  I_solutions is not
            transferred (they correspond to the original, raw reference plane).
        """
        delta_theta = np.asarray(delta_theta, dtype=np.float64)
        if len(delta_theta) != len(self.Z_matrix):
            raise ValueError(
                f"deembed_phase: delta_theta length {len(delta_theta)} "
                f"!= number of ports {len(self.Z_matrix)}"
            )
        phi = np.diag(np.exp(-1j * delta_theta))
        S_deemb = phi @ self.S_matrix @ phi
        P = len(S_deemb)
        I_id = np.eye(P, dtype=np.complex128)
        Z_deemb = self.Z0 * np.linalg.solve(I_id - S_deemb, I_id + S_deemb)
        return NetworkResult(
            frequency=self.frequency,
            Z_matrix=Z_deemb,
            port_names=list(self.port_names),
            Z0=self.Z0,
            condition_number=self.condition_number,
        )

    def correct_port_parasitics(
        self,
        series_Z: list,
        shunt_Y: list,
    ) -> 'NetworkResult':
        """Subtract lumped gap parasitics via open-short de-embedding.

        Removes series impedance and shunt admittance at each port,
        following the standard open-short calibration procedure used in
        on-wafer measurements.

        De-embedding order (outside-in):

        1. Remove shunt Y:  ``Y_temp[p, p] -= shunt_Y[p]``
        2. Remove series Z: ``Z_temp[p, p] -= series_Z[p]``

        Approximate analytical estimates for the gap parasitics:

        - Series inductance: ``Z_L = jω μ₀ d / (2π)``  (d = gap width)
        - Shunt capacitance: ``Y_C = jω ε₀ w``  (w = edge width)

        For accurate corrections, calibration against a known reference
        structure (open/short stub) is recommended.

        Parameters
        ----------
        series_Z : list of complex, length P
            Series impedance to subtract at each port (Ω).
        shunt_Y : list of complex, length P
            Shunt admittance to subtract at each port (S).

        Returns
        -------
        NetworkResult
            New result with de-embedded Z_matrix.  ``I_solutions`` is not
            transferred (they correspond to the original reference plane).
        """
        P = len(self.Z_matrix)
        series_Z = np.asarray(series_Z, dtype=np.complex128)
        shunt_Y = np.asarray(shunt_Y, dtype=np.complex128)
        if len(series_Z) != P or len(shunt_Y) != P:
            raise ValueError(
                f"series_Z and shunt_Y must have length {P} "
                f"(got {len(series_Z)}, {len(shunt_Y)})"
            )

        # Step 1: remove shunt admittance in Y-domain
        Y_temp = self.Y_matrix.copy()
        for p in range(P):
            Y_temp[p, p] -= shunt_Y[p]

        # Step 2: remove series impedance in Z-domain
        Z_temp = np.linalg.inv(Y_temp)
        for p in range(P):
            Z_temp[p, p] -= series_Z[p]

        return NetworkResult(
            frequency=self.frequency,
            Z_matrix=Z_temp,
            port_names=list(self.port_names),
            Z0=self.Z0,
            condition_number=self.condition_number,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save to a .npz file.

        Parameters
        ----------
        path : str
            Output path (the .npz extension is added automatically if absent).
        """
        kw = dict(
            Z_matrix=self.Z_matrix,
            frequency=np.array(self.frequency),
            port_names=np.array(self.port_names),
            Z0=np.array(self.Z0),
            condition_number=np.array(
                self.condition_number if self.condition_number is not None else np.nan
            ),
        )
        if self.I_solutions is not None:
            kw['I_solutions'] = self.I_solutions
        np.savez(path, **kw)

    @classmethod
    def load(cls, path: str) -> 'NetworkResult':
        """Load from a .npz file saved with :meth:`save`.

        Parameters
        ----------
        path : str

        Returns
        -------
        NetworkResult
        """
        d = np.load(path, allow_pickle=True)
        cond = float(d['condition_number'])
        return cls(
            frequency=float(d['frequency']),
            Z_matrix=d['Z_matrix'],
            port_names=list(d['port_names']),
            Z0=float(d['Z0']),
            condition_number=cond if np.isfinite(cond) else None,
            I_solutions=d['I_solutions'] if 'I_solutions' in d else None,
        )
