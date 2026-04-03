"""Layer and LayerStack: stratified medium definitions for multilayer MoM.

Each Layer represents a homogeneous dielectric/lossy slab bounded by two
horizontal (z = const) planes.  LayerStack orders layers from bottom to top
and provides helpers for BEOL stackup import.

Convention
----------
- z increases upward
- Layers must be contiguous and non-overlapping
- The bottom-most and top-most layers may extend to ±inf
- conductivity is additive to Im(eps_r): eps_eff = eps_r - j*sigma/(omega*eps0)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Layer:
    """A single homogeneous slab in a stratified medium.

    Parameters
    ----------
    name : str
        Human-readable label, e.g. 'Si_substrate', 'ILD_1', 'M2'.
    z_bot : float
        Bottom interface z-coordinate (m).  Use -np.inf for the lowest layer.
    z_top : float
        Top interface z-coordinate (m).  Use np.inf for the topmost layer.
    eps_r : complex
        Relative permittivity.  Imaginary part encodes dielectric loss directly.
    mu_r : complex
        Relative permeability.  Typically 1.0 for non-magnetic materials.
    conductivity : float
        Ohmic conductivity (S/m).  Added to Im(eps_r) via eps_eff formula.
        Use this for metals and semiconductors instead of a large Im(eps_r).
    is_pec : bool
        If True, this layer acts as a perfect electric conductor.  Only
        meaningful for half-space layers (z_bot = -inf or z_top = inf).
        When set, the Strata backend passes ``pec_bot=True`` or
        ``pec_top=True`` to model an image-theory ground plane.
    """

    name: str
    z_bot: float
    z_top: float
    eps_r: complex = 1.0 + 0.0j
    mu_r: complex = 1.0 + 0.0j
    conductivity: float = 0.0  # S/m
    is_pec: bool = False

    def eps_r_eff(self, omega: float) -> complex:
        """Complex effective permittivity including ohmic loss.

        eps_eff = eps_r - j * sigma / (omega * eps0)

        Parameters
        ----------
        omega : float
            Angular frequency (rad/s).
        """
        from ..utils.constants import eps0
        if omega == 0.0:
            return complex(self.eps_r)
        return complex(self.eps_r) - 1j * self.conductivity / (omega * eps0)

    def wavenumber(self, omega: float) -> complex:
        """Wavenumber k = omega * sqrt(eps_eff * mu_r) / c0."""
        from ..utils.constants import c0
        eps_eff = self.eps_r_eff(omega)
        return (omega / c0) * np.sqrt(complex(eps_eff) * complex(self.mu_r))

    def wave_impedance(self, omega: float) -> complex:
        """Intrinsic wave impedance eta = eta0 * sqrt(mu_r / eps_eff)."""
        from ..utils.constants import eta0
        eps_eff = self.eps_r_eff(omega)
        return eta0 * np.sqrt(complex(self.mu_r) / complex(eps_eff))

    def resistivity(self) -> float:
        """Resistivity (Ω·m) = 1/conductivity.  Returns inf if conductivity == 0."""
        return 1.0 / self.conductivity if self.conductivity > 0 else math.inf

    def contains_z(self, z: float, tol: float = 1e-12) -> bool:
        """Return True if z lies within [z_bot, z_top] (inclusive with tol)."""
        return (self.z_bot - tol) <= z <= (self.z_top + tol)


@dataclass
class LayerStack:
    """Ordered stack of homogeneous layers representing a stratified medium.

    Layers must be listed bottom-to-top (ascending z_bot).  Contiguity
    (z_top[i] == z_bot[i+1]) is checked at construction time.

    Parameters
    ----------
    layers : list of Layer
        Ordered (bottom-to-top) list of layer definitions.

    Examples
    --------
    Simple air-over-silicon half-space::

        stack = LayerStack([
            Layer('silicon', z_bot=-np.inf, z_top=0.0, eps_r=11.7, conductivity=10.0),
            Layer('air',     z_bot=0.0,     z_top=np.inf),
        ])

    BEOL snippet (180 nm CMOS, not to scale)::

        stack = LayerStack.from_yaml('process/tsmc180.yaml')
    """

    layers: List[Layer]

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if len(self.layers) == 0:
            raise ValueError("LayerStack must contain at least one layer")
        for i, layer in enumerate(self.layers):
            if layer.z_top <= layer.z_bot:
                raise ValueError(
                    f"Layer '{layer.name}': z_top ({layer.z_top}) must be > z_bot ({layer.z_bot})"
                )
        # Check contiguity (allow small floating-point gaps)
        for i in range(len(self.layers) - 1):
            gap = abs(self.layers[i + 1].z_bot - self.layers[i].z_top)
            if gap > 1e-12:
                raise ValueError(
                    f"Gap between layer '{self.layers[i].name}' (z_top={self.layers[i].z_top}) "
                    f"and layer '{self.layers[i+1].name}' (z_bot={self.layers[i+1].z_bot}): "
                    f"gap = {gap:.3e} m (must be < 1e-12)"
                )

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def layer_at_z(self, z: float) -> Layer:
        """Return the layer containing z.

        Parameters
        ----------
        z : float
            z-coordinate (m).

        Returns
        -------
        Layer
        """
        for layer in self.layers:
            if layer.contains_z(z):
                return layer
        raise ValueError(
            f"z={z} m is outside the LayerStack extent "
            f"[{self.layers[0].z_bot}, {self.layers[-1].z_top}]"
        )

    def layer_index_at_z(self, z: float) -> int:
        """Return the index (0-based, bottom-to-top) of the layer containing z.

        Parameters
        ----------
        z : float
            z-coordinate (m).

        Returns
        -------
        int
        """
        for i, layer in enumerate(self.layers):
            if layer.contains_z(z):
                return i
        raise ValueError(
            f"z={z} m is outside the LayerStack extent "
            f"[{self.layers[0].z_bot}, {self.layers[-1].z_top}]"
        )

    def z_of_layer(self, name: str) -> float:
        """Return the z_top of the named layer (mesh placement surface)."""
        for layer in self.layers:
            if layer.name == name:
                return layer.z_top
        raise KeyError(f"No layer named '{name}'")

    def get_layer(self, name: str) -> Layer:
        """Return the layer with the given name."""
        for layer in self.layers:
            if layer.name == name:
                return layer
        raise KeyError(f"No layer named '{name}'")

    @property
    def z_interfaces(self) -> np.ndarray:
        """Interior interface z-coordinates (excluding ±inf boundaries).

        Returns array of finite z-values where adjacent layers meet.
        Used by empymod ``depth`` parameter.
        """
        interfaces = []
        for layer in self.layers[:-1]:
            z = layer.z_top
            if math.isfinite(z):
                interfaces.append(z)
        return np.array(interfaces, dtype=np.float64)

    @property
    def resistivities(self) -> np.ndarray:
        """Layer resistivities (Ω·m), one per layer, for empymod ``res`` parameter."""
        return np.array([l.resistivity() for l in self.layers], dtype=np.float64)

    # ------------------------------------------------------------------
    # Factory: YAML
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> 'LayerStack':
        """Load a LayerStack from a YAML stackup file.

        YAML format::

            stackup:
              - name: Si_substrate
                z_bot: -.inf
                z_top: 0.0
                eps_r: 11.7
                conductivity: 10.0
              - name: ILD_0
                z_bot: 0.0
                z_top: 5.0e-7
                eps_r: 4.1

        Parameters
        ----------
        path : str
            Path to YAML file.
        """
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "PyYAML is required for LayerStack.from_yaml(). "
                "Install with: pip install pyyaml"
            ) from e

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        entries = data.get('stackup', data.get('layers', []))
        layers = []
        for entry in entries:
            layers.append(Layer(
                name=str(entry['name']),
                z_bot=float(entry['z_bot']),
                z_top=float(entry['z_top']),
                eps_r=complex(entry.get('eps_r', 1.0)),
                mu_r=complex(entry.get('mu_r', 1.0)),
                conductivity=float(entry.get('conductivity', 0.0)),
            ))
        return cls(layers=layers)
