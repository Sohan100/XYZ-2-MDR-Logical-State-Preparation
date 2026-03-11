"""
xyz2_mdr package exports
------------------------
Public package surface for the XYZ^2 MDR simulation toolkit.

This module re-exports the constants, circuit builders, sweep utilities,
table generators, and plotting helpers that are intended for external use.
Importing from `xyz2_mdr` therefore provides a stable, concise entrypoint
without requiring callers to know the internal module layout.
"""

from .constants import (
    DEFAULT_DISTANCES,
    DEFAULT_NUM_REPLICATES,
    DEFAULT_P_SPAM,
    DEFAULT_ROUNDS,
    DEFAULT_SHOTS,
    NOISE_MODEL_DISPLAY_NAMES,
    NOISE_MODEL_PARAM_NAMES,
    default_probabilities,
)
from .analysis_reporting import (
    NotebookFinalRoundAnalysis,
    NotebookThresholdAnalysis,
)
from .mdr_circuit import MDRCircuit
from .mdr_noise_sweep import MdrNoiseSweep
from .mdr_simulation import MDRSimulation
from .mdr_table import MDRTable
from .plotters import MDRSimulationPlotter, MdrNoiseSweepPlotter
from .robust_toggle_generator import RobustToggleGenerator
from .xyz2_logical_generator import XYZ2LogicalGenerator
from .xyz2_stabilizer_generator import XYZ2StabilizerGenerator

__all__ = [
    "DEFAULT_DISTANCES",
    "DEFAULT_NUM_REPLICATES",
    "DEFAULT_P_SPAM",
    "DEFAULT_ROUNDS",
    "DEFAULT_SHOTS",
    "NOISE_MODEL_DISPLAY_NAMES",
    "NOISE_MODEL_PARAM_NAMES",
    "default_probabilities",
    "NotebookFinalRoundAnalysis",
    "NotebookThresholdAnalysis",
    "MDRCircuit",
    "MdrNoiseSweep",
    "MDRSimulation",
    "MDRSimulationPlotter",
    "MDRTable",
    "MdrNoiseSweepPlotter",
    "RobustToggleGenerator",
    "XYZ2LogicalGenerator",
    "XYZ2StabilizerGenerator",
]
