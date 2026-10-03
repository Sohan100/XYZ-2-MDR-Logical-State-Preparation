"""
Decoder factory helpers.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..mdr_circuit import MDRCircuit
from .base import StatePrepDecoder
from .dem import detector_fault_model_from_circuit
from .mps_mld import MpsMldDecoder


SUPPORTED_DECODER_MODES = [
    "toggle_frame",
    "mps_mld",
    "mps_mld_exact_small",
]
DEFAULT_MPS_MAX_BOND_DIMENSION = 4096


def build_decoder(
    *,
    decoder_mode: str,
    mdr: MDRCircuit,
    rounds: int,
    observable_label: str,
    observable_pauli: str,
    decoder_config: Mapping[str, Any] | None = None,
) -> StatePrepDecoder:
    """
    Build the requested decoder for one logical observable and round count.
    """
    if decoder_mode == "toggle_frame":
        raise ValueError("toggle_frame is handled directly by MDRSimulation.")
    if decoder_mode not in SUPPORTED_DECODER_MODES:
        raise ValueError(
            "decoder_mode must be one of: "
            + ", ".join(SUPPORTED_DECODER_MODES)
        )

    config = dict(decoder_config or {})
    max_bond_dimension = config.get("max_bond_dimension")
    if decoder_mode == "mps_mld" and max_bond_dimension is None:
        max_bond_dimension = DEFAULT_MPS_MAX_BOND_DIMENSION
    if decoder_mode == "mps_mld_exact_small":
        max_bond_dimension = None

    circuit = mdr.build_detector_annotated_state_prep(
        rounds=rounds,
        final_observable_label=observable_label,
        final_observable_pauli=observable_pauli,
    )
    model = detector_fault_model_from_circuit(circuit)
    return MpsMldDecoder(
        model=model,
        num_checks=len(mdr.stabilizers),
        rounds=rounds,
        max_bond_dimension=max_bond_dimension,
    )
