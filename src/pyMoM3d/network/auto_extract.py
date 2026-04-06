"""Automatic source selection: QS probe ↔ half-RWG gap dispatch.

Provides :class:`DualBandExtractor` which automatically selects the
quasi-static solver (grounded probe feeds) for low frequencies and the
full-wave solver (half-RWG gap ports) for high frequencies, based on
the electrical size kD at each frequency.

This mirrors Keysight Momentum's grounded/floating source switching,
adapted to pyMoM3d's solver architecture:

- **kD < threshold** → QS probe (conductive return via G_φ)
- **kD ≥ threshold** → FW half-RWG (displacement current return)

Both solvers return :class:`NetworkResult` objects, so the output is a
unified frequency-ordered list regardless of which solver produced each
result.

Usage
-----
::

    # Set up both solvers independently
    qs = QuasiStaticSolver(sim_qs, qs_ports, probe_feeds=True)
    fw = NetworkExtractor(sim_fw, fw_ports)

    # Wrap in DualBandExtractor
    dual = DualBandExtractor(qs, fw, max_dimension=0.02)

    # Extract across full band — dispatch is automatic
    results = dual.extract([0.5e9, 1e9, 5e9, 10e9, 15e9])
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

from ..utils.constants import c0
from .network_result import NetworkResult


def compute_kD(freq: float, max_dimension: float) -> float:
    """Compute the electrical size kD at a given frequency.

    Parameters
    ----------
    freq : float
        Frequency (Hz).
    max_dimension : float
        Maximum physical dimension of the structure (m).

    Returns
    -------
    kD : float
        Electrical size (dimensionless).
    """
    k = 2.0 * np.pi * freq / c0
    return k * max_dimension


def partition_frequencies(
    frequencies: list,
    max_dimension: float,
    kD_threshold: float = 0.5,
) -> tuple:
    """Partition frequencies into QS and FW bands based on electrical size.

    Parameters
    ----------
    frequencies : list of float
        Frequencies to partition (Hz).
    max_dimension : float
        Maximum physical dimension of the structure (m).
    kD_threshold : float
        Electrical size crossover.  Frequencies with kD < threshold
        are assigned to QS; kD ≥ threshold to FW.

    Returns
    -------
    qs_freqs : list of float
        Frequencies for the quasi-static solver.
    fw_freqs : list of float
        Frequencies for the full-wave solver.
    qs_indices : list of int
        Original indices of QS frequencies in the input list.
    fw_indices : list of int
        Original indices of FW frequencies in the input list.
    """
    qs_freqs, fw_freqs = [], []
    qs_indices, fw_indices = [], []

    for i, f in enumerate(frequencies):
        kD = compute_kD(f, max_dimension)
        if kD < kD_threshold:
            qs_freqs.append(f)
            qs_indices.append(i)
        else:
            fw_freqs.append(f)
            fw_indices.append(i)

    return qs_freqs, fw_freqs, qs_indices, fw_indices


def mesh_max_dimension(mesh) -> float:
    """Compute the maximum dimension of a mesh from its bounding box diagonal.

    Parameters
    ----------
    mesh : Mesh
        Triangular surface mesh.

    Returns
    -------
    D : float
        Bounding box diagonal length (m).
    """
    verts = mesh.vertices
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    return float(np.linalg.norm(bbox_max - bbox_min))


class DualBandExtractor:
    """Automatic QS/FW source selection for wideband extraction.

    Wraps a quasi-static solver (with probe feeds) and a full-wave
    extractor (with half-RWG ports).  At each frequency, the electrical
    size kD determines which solver is used:

    - kD < ``kD_threshold``: quasi-static (grounded probe, conductive return)
    - kD ≥ ``kD_threshold``: full-wave (half-RWG gap, displacement return)

    Both solvers must extract the same number of ports in the same order.

    Parameters
    ----------
    qs_solver : QuasiStaticSolver
        Quasi-static solver with probe feeds configured.
    fw_extractor : NetworkExtractor
        Full-wave extractor with half-RWG gap ports.
    max_dimension : float
        Maximum physical dimension of the structure (m).  Used to compute
        kD = 2πf/c₀ × D.  Can be computed from the mesh bounding box
        via :func:`mesh_max_dimension`.
    kD_threshold : float
        Electrical size crossover (default 0.5).
    """

    def __init__(
        self,
        qs_solver,
        fw_extractor,
        max_dimension: float,
        kD_threshold: float = 0.5,
    ):
        self.qs = qs_solver
        self.fw = fw_extractor
        self.max_dim = float(max_dimension)
        self.kD_threshold = float(kD_threshold)

    def extract(
        self,
        frequencies: Union[float, List[float]],
    ) -> List[NetworkResult]:
        """Extract network parameters with automatic solver dispatch.

        Parameters
        ----------
        frequencies : float or list of float
            Frequencies (Hz).

        Returns
        -------
        list of NetworkResult
            One entry per frequency, in the same order as ``frequencies``.
            Use :meth:`solver_at` to determine which solver produced a
            given frequency's result.
        """
        if np.isscalar(frequencies):
            frequencies = [float(frequencies)]
        else:
            frequencies = list(frequencies)

        qs_freqs, fw_freqs, qs_idx, fw_idx = partition_frequencies(
            frequencies, self.max_dim, self.kD_threshold
        )

        # Allocate output array
        results = [None] * len(frequencies)

        # QS extraction
        if qs_freqs:
            qs_results = self.qs.extract(qs_freqs)
            for i, result in zip(qs_idx, qs_results):
                results[i] = result

        # FW extraction
        if fw_freqs:
            fw_results = self.fw.extract(fw_freqs)
            for i, result in zip(fw_idx, fw_results):
                results[i] = result

        return results

    def crossover_frequency(self) -> float:
        """Compute the crossover frequency where kD = kD_threshold.

        Returns
        -------
        f_crossover : float
            Crossover frequency (Hz).
        """
        return self.kD_threshold * c0 / (2.0 * np.pi * self.max_dim)

    def solver_at(self, freq: float) -> str:
        """Return which solver would be used at a given frequency.

        Parameters
        ----------
        freq : float
            Frequency (Hz).

        Returns
        -------
        solver : str
            ``'qs'`` or ``'fw'``.
        """
        kD = compute_kD(freq, self.max_dim)
        return 'qs' if kD < self.kD_threshold else 'fw'
