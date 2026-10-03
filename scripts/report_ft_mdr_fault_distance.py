"""
report_ft_mdr_fault_distance.py
----------------------------------------------------------------------------
Print the circuit-level fault distance of the FT MDR protocol, and of the
|+>^n start for comparison, using Stim's undetectable-logical-error search.

    python scripts/report_ft_mdr_fault_distance.py --distances 3 5 7
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mdr.ft import CircuitNoise, FTMDRCircuit  # noqa: E402
from mdr.mdr_circuit import MDRCircuit  # noqa: E402
from mdr.preparation import build_preparation_plan  # noqa: E402
from xyz2 import XYZ2LogicalGenerator, XYZ2StabilizerGenerator  # noqa: E402


def legacy_link_logical_plus(d: int, rounds: int, p: float = 1e-3):
    """
    Build the repository's detector-annotated link_logical_plus circuit.
    """
    stabs = XYZ2StabilizerGenerator(d).generate_stabilizers()
    lx = XYZ2LogicalGenerator(d).generate_logicals()["Logical X"]
    plan = build_preparation_plan(code_family="xyz2", num_qubits=2 * d * d,
                                  stabilizers=stabs, logical_x=lx,
                                  prep_mode="link_logical_plus")
    active = [stabs[i] for i in plan.active_stabilizer_indices]
    mdr = MDRCircuit(stabilizers=active, toggles=["I"] * len(active),
                     ancillas=1, p_spam=p, g1_x=p / 3, g1_y=p / 3,
                     g1_z=p / 3, gate_noise_2q=[p / 15] * 15,
                     psi_circuit=plan.psi_circuit, num_qubits=2 * d * d,
                     correction_mode="pauli_frame")
    return mdr.build_detector_annotated_state_prep(
        rounds=rounds, final_observable_label="Logical X",
        final_observable_pauli=lx)


def fault_distance(circuit, max_set: int = 6, explain: bool = False) -> int:
    errors = circuit.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=max_set,
        dont_explore_edges_with_degree_above=9999,
        dont_explore_edges_increasing_symptom_degree=False,
        canonicalize_circuit_errors=True,
    )
    if explain:
        for err in errors:
            for loc in err.circuit_error_locations:
                print("     ", loc.flipped_pauli_product, "at tick",
                      loc.tick_offset)
    return len(errors)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--explain", action="store_true")
    args = ap.parse_args()
    noise = CircuitNoise.uniform(1e-3)
    print("d  rounds  init   final  fault_distance")
    for d in args.distances:
        for rounds in sorted({1, d}):
            for final in ("frame", "ideal"):
                c = FTMDRCircuit(d, rounds, noise, final=final).build()
                fd = fault_distance(c, explain=args.explain)
                print(f"{d}  {rounds:6d}  frame  {final:5s}  {fd}")
        c = FTMDRCircuit(d, d, noise, init="plus", detectors="all").build()
        print(f"{d}  {d:6d}  plus   frame  "
              f"{fault_distance(c, 3, explain=args.explain)}")
        c = legacy_link_logical_plus(d, d)
        print(f"{d}  {d:6d}  legacy link_logical_plus (repo MDRCircuit)  "
              f"{fault_distance(c, 3, explain=args.explain)}")


if __name__ == "__main__":
    main()
