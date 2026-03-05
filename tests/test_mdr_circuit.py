from __future__ import annotations

import stim

from xyz2_mdr.mdr_circuit import MDRCircuit
from xyz2_mdr.mdr_table import MDRTable


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
