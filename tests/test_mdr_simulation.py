"""

test_mdr_simulation.py
----------------------------------------------------------------------------
Pytest coverage for mdr simulation behavior and regression checks.
"""

from __future__ import annotations

import numpy as np
import stim

from mdr.decoders.base import DecodeResult
from mdr.mdr_circuit import MDRCircuit
from mdr.mdr_simulation import MDRSimulation


def test_pauli_frame_matches_physical_recovery_for_single_round() -> None:
    """
    One-round Pauli-frame recovery should match physical recovery exactly.

    This toy model uses one measured `Z0` stabilizer with toggle `X0`. The
    initial state is `|1>` on qubit 0, so the stabilizer syndrome should
    trigger a correction. Both physical recovery and Pauli-frame recovery
    should therefore report final `Z0 = +1`.

    Returns:
    None
    """
    psi = stim.Circuit("X 0")
    physical = MDRSimulation(
        mdr=MDRCircuit(
            stabilizers=["Z0"],
            toggles=["X0"],
            ancillas=1,
            psi_circuit=psi,
            recovery_mode="each_round",
            correction_mode="physical",
        ),
        stabilizer_pauli_strings=["Z0"],
        logical_pauli_strings={},
        shots_per_measurement=64,
        total_mdr_rounds=1,
        num_replicates=1,
    )
    pauli_frame = MDRSimulation(
        mdr=MDRCircuit(
            stabilizers=["Z0"],
            toggles=["X0"],
            ancillas=1,
            psi_circuit=psi,
            recovery_mode="each_round",
            correction_mode="pauli_frame",
        ),
        stabilizer_pauli_strings=["Z0"],
        logical_pauli_strings={},
        shots_per_measurement=64,
        total_mdr_rounds=1,
        num_replicates=1,
    )

    assert physical._stats_stabilizers["Z0"]["centers"][1] == 1.0
    assert pauli_frame._stats_stabilizers["Z0"]["centers"][1] == 1.0


def test_pauli_frame_corrects_later_round_syndromes_in_each_round_mode() -> (
    None
):
    """
    Each-round Pauli-frame recovery must reinterpret later syndrome rounds.

    Starting from `|1>`, the first `Z0` syndrome triggers an `X0` recovery. In
    physical mode the second syndrome is measured on the corrected state, so no
    second correction is applied. Pauli-frame mode must reproduce this by
    interpreting the second raw syndrome in the current frame before deciding
    whether another toggle fires.

    Returns:
    None
    """
    psi = stim.Circuit("X 0")
    physical = MDRSimulation(
        mdr=MDRCircuit(
            stabilizers=["Z0"],
            toggles=["X0"],
            ancillas=1,
            psi_circuit=psi,
            recovery_mode="each_round",
            correction_mode="physical",
        ),
        stabilizer_pauli_strings=["Z0"],
        logical_pauli_strings={},
        shots_per_measurement=64,
        total_mdr_rounds=2,
        num_replicates=1,
    )
    pauli_frame = MDRSimulation(
        mdr=MDRCircuit(
            stabilizers=["Z0"],
            toggles=["X0"],
            ancillas=1,
            psi_circuit=psi,
            recovery_mode="each_round",
            correction_mode="pauli_frame",
        ),
        stabilizer_pauli_strings=["Z0"],
        logical_pauli_strings={},
        shots_per_measurement=64,
        total_mdr_rounds=2,
        num_replicates=1,
    )

    assert physical._stats_stabilizers["Z0"]["centers"][2] == 1.0
    assert pauli_frame._stats_stabilizers["Z0"]["centers"][2] == 1.0


def test_pauli_frame_matches_physical_recovery_for_final_round_mode() -> None:
    """
    Final-round Pauli-frame recovery should match deferred physical recovery.

    This case checks the simpler deferred-recovery policy where only the last
    syndrome block is converted into a correction. Both implementations should
    report the same final stabilizer expectation.

    Returns:
    None
    """
    psi = stim.Circuit("X 0")
    physical = MDRSimulation(
        mdr=MDRCircuit(
            stabilizers=["Z0"],
            toggles=["X0"],
            ancillas=1,
            psi_circuit=psi,
            recovery_mode="final_round",
            correction_mode="physical",
        ),
        stabilizer_pauli_strings=["Z0"],
        logical_pauli_strings={},
        shots_per_measurement=64,
        total_mdr_rounds=2,
        num_replicates=1,
    )
    pauli_frame = MDRSimulation(
        mdr=MDRCircuit(
            stabilizers=["Z0"],
            toggles=["X0"],
            ancillas=1,
            psi_circuit=psi,
            recovery_mode="final_round",
            correction_mode="pauli_frame",
        ),
        stabilizer_pauli_strings=["Z0"],
        logical_pauli_strings={},
        shots_per_measurement=64,
        total_mdr_rounds=2,
        num_replicates=1,
    )

    assert physical._stats_stabilizers["Z0"]["centers"][2] == 1.0
    assert pauli_frame._stats_stabilizers["Z0"]["centers"][2] == 1.0


def test_mps_decoder_called_in_pauli_frame_mode(monkeypatch) -> None:
    """
    Non-toggle decoder modes should delegate logical correction to a decoder.
    """
    calls: list[tuple[int, int, int]] = []

    class FakeDecoder:
        def decode_batch(self, *, syndrome_rounds, observable_labels):
            calls.append(syndrome_rounds.shape)
            shots = syndrome_rounds.shape[0]
            return DecodeResult(
                frame_x=None,
                frame_z=None,
                observable_flips={
                    label: np.zeros(shots, dtype=np.uint8)
                    for label in observable_labels
                },
                decoder_known=np.ones(shots, dtype=bool),
                log_likelihood_gap=np.ones(shots, dtype=float),
                truncation_mass_lost=np.zeros(shots, dtype=float),
            )

    def fake_build_decoder(**kwargs):
        return FakeDecoder()

    monkeypatch.setattr(
        "mdr.mdr_simulation.build_decoder",
        fake_build_decoder,
    )

    sim = MDRSimulation(
        mdr=MDRCircuit(
            stabilizers=["Z0"],
            toggles=["X0"],
            ancillas=1,
            psi_circuit=stim.Circuit("H 0"),
            recovery_mode="final_round",
            correction_mode="pauli_frame",
            num_qubits=1,
        ),
        stabilizer_pauli_strings=[],
        logical_pauli_strings={"Logical X": "X0"},
        shots_per_measurement=8,
        total_mdr_rounds=1,
        num_replicates=1,
        decoder_mode="mps_mld",
    )

    assert calls.count((8, 0, 1)) == 2
    assert calls.count((8, 1, 1)) == 2
    assert sim.decoder_diagnostic_summary(
        label="Logical X",
        round_count=1,
    )["decoder_unknown_fraction"] == 0.0
