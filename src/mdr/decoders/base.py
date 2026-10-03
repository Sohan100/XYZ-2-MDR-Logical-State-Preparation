"""
Shared decoder interfaces for MDR state-preparation workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np


@dataclass(frozen=True)
class DecodeResult:
    """
    Batch decoder output.

    For the Logical-X threshold workflow, ``observable_flips`` is sufficient:
    each value is a length-``shots`` binary vector saying whether to flip the
    sampled observable eigenvalue. Full Pauli-frame masks are optional because
    the production path never physically applies the correction.
    """

    frame_x: np.ndarray | None
    frame_z: np.ndarray | None
    observable_flips: Mapping[str, np.ndarray]
    decoder_known: np.ndarray
    log_likelihood_gap: np.ndarray
    truncation_mass_lost: np.ndarray


class StatePrepDecoder(Protocol):
    """
    Protocol for state-preparation decoders consumed by ``MDRSimulation``.
    """

    def decode_batch(
        self,
        *,
        syndrome_rounds: np.ndarray,
        observable_labels: list[str],
    ) -> DecodeResult:
        """
        Decode a batch of raw active-check syndrome histories.
        """
