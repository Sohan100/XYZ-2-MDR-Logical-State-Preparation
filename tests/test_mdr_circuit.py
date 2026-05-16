"""

test_mdr_circuit.py
----------------------------------------------------------------------------
Pytest coverage for mdr circuit behavior and regression checks.
"""

from __future__ import annotations

import pytest
import stim

from mdr.mdr_circuit import MDRCircuit
from mdr.mdr_table import MDRTable


def test_mdr_circuit_build_smoke(d3_table: MDRTable) -> None:
    """
    Smoke-test that a full MDR circuit can be constructed for distance 3.

    The test builds stabilizers, logicals, and toggles from the table helper,
    then verifies that `MDRCircuit.build()` returns a non-empty Stim circuit.

    Returns:
    None
    """
    stabs = d3_table.get_stabilizers()
    logs = d3_table.get_logicals_dict()
    stab_toggles, log_x_toggle = d3_table.get_toggles()

    logical_x = logs["Logical X"]
    psi = stim.Circuit()
    psi.append_operation("H", [int(term[1:]) for term in logical_x.split()])

    circ = MDRCircuit(
        stabilizers=stabs + [logical_x],
        toggles=stab_toggles + [log_x_toggle],
        ancillas=1,
        p_spam=1.339e-3,
        psi_circuit=psi,
    ).build(include_psi=True)

    assert circ.num_qubits > 0
    assert len(circ) > 0


def test_mdr_circuit_can_build_without_recovery(d3_table: MDRTable) -> None:
    """
    Verify syndrome-only and recovery-only subcircuits can be built.

    Returns:
    None
    """
    stabs = d3_table.get_stabilizers()
    logs = d3_table.get_logicals_dict()
    stab_toggles, log_x_toggle = d3_table.get_toggles()
    logical_x = logs["Logical X"]

    circ = MDRCircuit(
        stabilizers=stabs + [logical_x],
        toggles=stab_toggles + [log_x_toggle],
        ancillas=1,
        recovery_mode="final_round",
    )

    syndrome_only = circ.build(include_psi=False, include_recovery=False)
    recovery_only = circ.build_recovery_only()

    assert "rec[" not in str(syndrome_only)
    assert "rec[" in str(recovery_only)


def test_invalid_correction_mode_raises_value_error() -> None:
    """
    Invalid correction-mode strings should be rejected at construction time.

    Returns:
    None
    """
    try:
        MDRCircuit(
            stabilizers=["Z0"],
            toggles=["X0"],
            correction_mode="bad_mode",
        )
    except ValueError as exc:
        assert "correction_mode" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid correction_mode")


def test_explicit_data_qubit_count_allows_pruned_checks(
    d3_table: MDRTable,
) -> None:
    """
    Pruned MDR variants still reserve all data qubits before ancillas.
    """
    stabs = d3_table.get_stabilizers()
    stab_toggles, _ = d3_table.get_toggles()
    active_stabs = stabs[9:]
    active_toggles = stab_toggles[9:]

    circ = MDRCircuit(
        stabilizers=active_stabs,
        toggles=active_toggles,
        ancillas=len(active_stabs),
        num_qubits=d3_table.n_qubits,
    ).build()

    assert circ.num_qubits == d3_table.n_qubits + len(active_stabs)
    assert f"R {d3_table.n_qubits}" in str(circ)


def test_explicit_data_qubit_count_must_cover_all_terms() -> None:
    """
    Explicit data-qubit counts should reject undersized registers.
    """
    with pytest.raises(ValueError, match="num_qubits"):
        MDRCircuit(
            stabilizers=["Z17"],
            toggles=["X17"],
            num_qubits=17,
        )


def test_physical_feedforward_correction_has_gate_noise() -> None:
    """
    Physical recovery gates should receive the configured one-qubit noise.
    """
    circ = MDRCircuit(
        stabilizers=["Z0"],
        toggles=["X0"],
        ancillas=1,
        g1_z=0.2,
        correction_mode="physical",
        num_qubits=1,
    ).build()
    text = str(circ)

    assert "CX rec[-1] 0" in text
    assert "PAULI_CHANNEL_1" in text
    assert "0.2" in text
