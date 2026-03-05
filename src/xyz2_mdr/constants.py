from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np


DEFAULT_DISTANCES: List[int] = [3, 5, 7, 9, 11]
DEFAULT_SHOTS: int = 3000
DEFAULT_NUM_REPLICATES: int = 30
DEFAULT_ROUNDS: List[int] = [1, 5, 10]
DEFAULT_P_SPAM: float = 1.339e-3
DEFAULT_DATA_DIR: Path = Path("data")
DEFAULT_TABLES_DIR: Path = DEFAULT_DATA_DIR / "tables"
DEFAULT_RESULTS_DIR: Path = DEFAULT_DATA_DIR / "simulation_results"
DEFAULT_PLOTS_DIR: Path = DEFAULT_DATA_DIR / "plots"

NOISE_MODEL_PARAM_NAMES: Dict[str, List[str]] = {
    "unbiased": [
        "g1_x",
        "g1_y",
        "g1_z",
        "IX",
        "IY",
        "IZ",
        "XI",
        "XX",
        "XY",
        "XZ",
        "YI",
        "YX",
        "YY",
        "YZ",
        "ZI",
        "ZX",
        "ZY",
        "ZZ",
    ],
    "z_type": ["g1_z", "IZ", "ZI", "ZZ", "ZX", "ZY", "XZ", "YZ"],
    "pure_z": ["g1_z", "IZ", "ZI", "ZZ"],
}

NOISE_MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "unbiased": "Unbiased Depolarizing Noise",
    "z_type": "Z Type Noise",
    "pure_z": "Pure Z Noise",
}


def default_probabilities() -> List[float]:
    """
    Return the default probability sweep for local and Slurm experiments.

    Returns:
        List[float]: Log-spaced values from `1e-5` to `1e-1`, plus three
        higher-probability points (`0.2`, `0.5`, and `1.0`).
    """
    return [*np.logspace(-5, -1, 26), 0.2, 0.5, 1.0]
