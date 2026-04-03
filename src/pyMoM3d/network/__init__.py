"""Multi-port network extraction for pyMoM3d.

Public API
----------
Port
    First-class port object mapping RWG basis indices to a circuit terminal.
NetworkResult
    Container for extracted Z/Y/S matrices at one frequency.
NetworkExtractor
    Orchestrates multi-port extraction: assembles Z_sys once per frequency,
    solves P RHS simultaneously, returns list[NetworkResult].
"""

from .port import Port, GroundVia
from .network_result import NetworkResult
from .extractor import NetworkExtractor

__all__ = ['Port', 'GroundVia', 'NetworkResult', 'NetworkExtractor']
