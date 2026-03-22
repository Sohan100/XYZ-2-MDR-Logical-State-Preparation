"""
tableau_analysis.py
-------------------
Helpers for comparing generated MDR toggles against Stim tableau bases.

This module is used for analysis and validation rather than simulation. It
converts between the project's sparse Pauli-string representation and Stim's
dense tableau objects, then assembles a dataframe that compares the generated
toggle basis against Stim's destabilizer basis for the same stabilizer set.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd
import stim

from mdr.robust_toggle_generator import RobustToggleGenerator

from .logical_generator import XYZ2LogicalGenerator
from .stabilizer_generator import XYZ2StabilizerGenerator


def sparse_pauli_to_stim(spec: str, num_qubits: int) -> stim.PauliString:
    """
    Convert a sparse project-format Pauli string into a dense Stim object.

    Args:
        spec: Sparse Pauli string such as `"X1 Z4 Y9"`.
        num_qubits: Total number of physical qubits in the code instance.

    Returns:
        stim.PauliString: Dense Pauli operator with explicit identity support
        on qubits not mentioned in `spec`.
    """
    chars = ["_"] * num_qubits
    if spec.strip() and spec.strip() != "I":
        for token in spec.split():
            chars[int(token[1:])] = token[0]
    return stim.PauliString("+" + "".join(chars))


def stim_pauli_to_sparse(pauli: stim.PauliString) -> str:
    """
    Convert a dense Stim Pauli string into the project's sparse format.

    Args:
        pauli: Dense Stim Pauli string.

    Returns:
        str: Sparse Pauli-string representation used elsewhere in the project.
    """
    dense = str(pauli)
    if dense and dense[0] in "+-":
        dense = dense[1:]
    tokens = [f"{gate}{idx}" for idx, gate in enumerate(dense) if gate != "_"]
    return " ".join(tokens) if tokens else "I"


def sparse_pauli_weight(spec: str) -> int:
    """
    Count the support weight of a sparse Pauli string.

    Args:
        spec: Sparse Pauli string in project format.

    Returns:
        int: Number of non-identity Pauli terms in the operator support.
    """
    if not spec or spec == "I":
        return 0
    return len(spec.split())


def build_destabilizer_report(
    distance: int,
    random_seed: int | None = 0,
) -> pd.DataFrame:
    """
    Build a comparison table between generated toggles and Stim destabilizers.

    The returned dataframe records one row per stabilizer plus the Logical-X
    constraint, including the generated toggle, its support weight, Stim's
    corresponding destabilizer, and the weight difference between the two.

    Args:
        distance: Odd code distance `d` with `d >= 3`.
        random_seed: Optional seed passed through to the toggle generator.

    Returns:
        pd.DataFrame: Comparison table used by reports and notebook analyses.
    """
    num_qubits = 2 * distance * distance
    stabilizers = XYZ2StabilizerGenerator(distance).generate_stabilizers()
    logicals = XYZ2LogicalGenerator(distance).generate_logicals()
    logical_x = logicals["Logical X"]
    constraints = stabilizers + [logical_x]

    generator = RobustToggleGenerator(
        stabilizers,
        logical_x,
        num_qubits,
        random_seed=random_seed,
    )
    stab_toggles, logical_toggle = generator.generate_toggles()
    generated_toggles = stab_toggles + [logical_toggle]
    optimization_statuses = list(generator.last_optimization_statuses)

    tableau = stim.Tableau.from_stabilizers(
        [sparse_pauli_to_stim(spec, num_qubits) for spec in constraints]
    )

    rows: List[Dict[str, object]] = []
    for idx, constraint in enumerate(constraints):
        category = "Logical" if idx == len(stabilizers) else "Stabilizer"
        label = "Logical X" if category == "Logical" else f"S_{idx}"
        generated_toggle = generated_toggles[idx]
        stim_destabilizer = stim_pauli_to_sparse(tableau.x_output(idx))
        rows.append(
            {
                "distance": distance,
                "index": idx,
                "category": category,
                "label": label,
                "constraint": constraint,
                "generated_toggle": generated_toggle,
                "generated_weight": sparse_pauli_weight(generated_toggle),
                "optimization_status": optimization_statuses[idx],
                "stim_destabilizer": stim_destabilizer,
                "stim_weight": sparse_pauli_weight(stim_destabilizer),
                "weight_delta": (
                    sparse_pauli_weight(stim_destabilizer)
                    - sparse_pauli_weight(generated_toggle)
                ),
            }
        )

    return pd.DataFrame(rows)
