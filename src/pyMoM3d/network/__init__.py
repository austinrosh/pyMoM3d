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
from .tl_extraction import extract_tl_from_y11, extract_tl_from_extractor, Y11ExtractionResult
from .feedline_calibration import FeedlineCalibration
from .soc_deembedding import SOCDeembedding
from .tl_deembedding import TLDeembedding, tl_abcd
from .port_calibration import PortCalibration, abcd_sqrt, solve_symmetric_fixture
from .auto_extract import (
    DualBandExtractor, partition_frequencies, compute_kD, mesh_max_dimension,
)

__all__ = [
    'Port', 'GroundVia', 'NetworkResult', 'NetworkExtractor',
    'extract_tl_from_y11', 'extract_tl_from_extractor', 'Y11ExtractionResult',
    'FeedlineCalibration', 'SOCDeembedding',
    'TLDeembedding', 'tl_abcd',
    'PortCalibration', 'abcd_sqrt', 'solve_symmetric_fixture',
    'DualBandExtractor', 'partition_frequencies', 'compute_kD', 'mesh_max_dimension',
]
