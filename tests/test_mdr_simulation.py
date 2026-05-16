"""

test_mdr_simulation.py
----------------------------------------------------------------------------
Pytest coverage for mdr simulation behavior and regression checks.
"""

from __future__ import annotations

import stim

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
