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
from mdr.preparation import PREP_MODE_LINK_LOGICAL_PLUS
from mdr.workflows import build_code_inputs


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


def test_si1000_syndrome_round_uses_rate_assignments() -> None:
    """
    SI1000 expands one swept p into reset, gate, idle, and measurement rates.
    """
    circuit = MDRCircuit(
        stabilizers=["Z0"],
        toggles=["X0"],
        ancillas=1,
        num_qubits=1,
        noise_profile=MDRCircuit.SI1000_PROFILE,
        si1000_p=0.03,
    ).build(include_psi=False, include_recovery=False)
    text = str(circuit)

    assert "PAULI_CHANNEL_1(0.06, 0, 0) 1" in text
    assert "PAULI_CHANNEL_1(0.001, 0.001, 0.001) 1" in text
    assert "PAULI_CHANNEL_2(0.002" in text
    assert "PAULI_CHANNEL_1(0.01, 0.01, 0.01) 1" in text
    assert "PAULI_CHANNEL_1(0.15, 0, 0) 1\nM 1" in text


def test_si1000_initial_preparation_gets_reset_and_gate_noise() -> None:
    """
    The implicit data reset and explicit prep gates use SI1000 prep noise.
    """
    mdr = MDRCircuit(
        stabilizers=["Z0"],
        toggles=["X0"],
        num_qubits=1,
        psi_circuit=stim.Circuit("H 0"),
        noise_profile=MDRCircuit.SI1000_PROFILE,
        si1000_p=0.03,
    )
    text = str(mdr.psi())

    assert "PAULI_CHANNEL_1(0.06, 0, 0) 0" in text
    assert "H 0" in text
    assert "PAULI_CHANNEL_1(0.001, 0.001, 0.001) 0" in text


def test_si1000_terminal_measurement_is_errorless() -> None:
    """
    Final data readout is left ideal for the SI1000 terminal measurement.
    """
    circuit = MDRCircuit(
        stabilizers=["Z0"],
        toggles=["X0"],
        num_qubits=1,
        noise_profile=MDRCircuit.SI1000_PROFILE,
        si1000_p=0.03,
    ).build_detector_annotated_state_prep(
        rounds=0,
        final_observable_label="Logical X",
        final_observable_pauli="X0",
        include_psi=False,
    )
    text = str(circuit)

    assert "MX 0" in text
    assert "PAULI_CHANNEL_1" not in text


def test_si1000_probability_rejects_invalid_measurement_rate() -> None:
    """
    The SI1000 MERR(5p) channel requires p <= 0.2.
    """
    with pytest.raises(ValueError, match="si1000_p"):
        MDRCircuit(
            stabilizers=["Z0"],
            toggles=["X0"],
            noise_profile=MDRCircuit.SI1000_PROFILE,
            si1000_p=0.21,
        )


def test_state_prep_detector_circuit_has_no_first_round_detectors(
    tmp_path,
) -> None:
    """
    Link-logical-plus active checks should use only temporal detectors.
    """
    code_inputs = build_code_inputs(
        distance=3,
        table_csv=tmp_path / "mdr_table_xyz2_d3.csv",
        code_family="xyz2",
        prep_mode=PREP_MODE_LINK_LOGICAL_PLUS,
    )
    mdr = MDRCircuit(
        # type: ignore[arg-type]
        stabilizers=code_inputs["code_stabilizers"],
        # type: ignore[arg-type]
        toggles=code_inputs["combined_toggles"],
        ancillas=len(code_inputs["active_stabilizers"]),  # type: ignore[arg-type]
        num_qubits=int(code_inputs["num_qubits"]),
        psi_circuit=code_inputs["psi_circuit"],
        correction_mode="pauli_frame",
        recovery_mode="final_round",
    )
    rounds = 2
    active_checks = len(code_inputs["code_stabilizers"])  # type: ignore[arg-type]

    circuit = mdr.build_detector_annotated_state_prep(
        rounds=rounds,
        final_observable_label="Logical X",
        final_observable_pauli=str(code_inputs["logical_x"]),
    )

    assert circuit.num_detectors == (rounds - 1) * active_checks


def test_state_prep_detector_circuit_has_logical_x_observable(
    tmp_path,
) -> None:
    """
    The annotated state-prep circuit should expose one Logical-X observable.
    """
    code_inputs = build_code_inputs(
        distance=3,
        table_csv=tmp_path / "mdr_table_xyz2_d3.csv",
        code_family="xyz2",
        prep_mode=PREP_MODE_LINK_LOGICAL_PLUS,
    )
    mdr = MDRCircuit(
        # type: ignore[arg-type]
        stabilizers=code_inputs["code_stabilizers"],
        # type: ignore[arg-type]
        toggles=code_inputs["combined_toggles"],
        ancillas=len(code_inputs["active_stabilizers"]),  # type: ignore[arg-type]
        num_qubits=int(code_inputs["num_qubits"]),
        psi_circuit=code_inputs["psi_circuit"],
        correction_mode="pauli_frame",
        recovery_mode="final_round",
    )

    circuit = mdr.build_detector_annotated_state_prep(
        rounds=2,
        final_observable_label="Logical X",
        final_observable_pauli=str(code_inputs["logical_x"]),
    )

    assert circuit.num_observables == 1
