"""

preparation.py
----------------------------------------------------------------------------
Initial-state and active-check planning for MDR protocol variants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

import stim

try:
    from ortools.sat.python import cp_model
except ImportError:  # pragma: no cover - optional dependency fallback
    cp_model = None


PREP_MODE_FULL_MDR = "full_mdr"
PREP_MODE_LINK_LOGICAL_PLUS = "link_logical_plus"
PREP_MODE_INDEPENDENT_HIGH_WEIGHT = "independent_high_weight"
PREP_MODES = [
    PREP_MODE_FULL_MDR,
    PREP_MODE_LINK_LOGICAL_PLUS,
    PREP_MODE_INDEPENDENT_HIGH_WEIGHT,
]


@dataclass(frozen=True)
class PreparationPlan:
    """
    Complete MDR preparation plan for one protocol variant.

    Attributes:
    mode: Name of the selected preparation mode.
    psi_circuit: Stim circuit preparing the guaranteed +1 eigenspaces.
    active_stabilizer_indices: Stabilizer rows still measured by MDR.
    prepared_stabilizer_indices: Stabilizer rows prepared directly at time
        zero and therefore omitted from MDR syndrome extraction.
    include_logical_x: Whether Logical X is included as an MDR check.
    description: Human-readable explanation suitable for metadata.
    """

    mode: str
    psi_circuit: stim.Circuit
    active_stabilizer_indices: List[int]
    prepared_stabilizer_indices: List[int]
    include_logical_x: bool
    description: str


def pauli_terms(pauli_spec: str) -> List[Tuple[str, int]]:
    """
    Parse a sparse Pauli string into ordered ``(pauli, qubit)`` terms.
    """
    terms: List[Tuple[str, int]] = []
    for token in pauli_spec.split():
        if token and token != "I":
            terms.append((token[0].upper(), int(token[1:])))
    return terms


def pauli_support(pauli_spec: str) -> Set[int]:
    """
    Return the qubit support of one sparse Pauli string.
    """
    return {qubit for _, qubit in pauli_terms(pauli_spec)}


def is_xyz2_link_stabilizer(pauli_spec: str) -> bool:
    """
    Identify the vertical XX link checks in the XYZ2 stabilizer ordering.
    """
    terms = pauli_terms(pauli_spec)
    return len(terms) == 2 and all(pauli == "X" for pauli, _ in terms)


def plus_state_circuit(num_qubits: int) -> stim.Circuit:
    """
    Prepare every data qubit in the +1 eigenstate of X.
    """
    circuit = stim.Circuit()
    if num_qubits > 0:
        circuit.append_operation("H", list(range(num_qubits)))
    return circuit


def logical_x_plus_circuit(logical_x: str) -> stim.Circuit:
    """
    Prepare the historical MDR Logical-X support in the X-basis.
    """
    circuit = stim.Circuit()
    targets = [qubit for _, qubit in pauli_terms(logical_x)]
    if targets:
        circuit.append_operation("H", targets)
    return circuit


def product_plus_eigenstate_circuit(
    stabilizers: Sequence[str],
) -> stim.Circuit:
    """
    Prepare disjoint stabilizers as products of single-qubit +1 eigenstates.

    Each selected stabilizer must be qubit-disjoint from every other selected
    stabilizer. For an X, Y, or Z term, the touched qubit is placed in the
    corresponding one-qubit +1 Pauli eigenstate. Since selected stabilizers do
    not share qubits, those local assignments jointly prepare every selected
    product stabilizer in its +1 eigenspace.
    """
    basis_by_qubit: Dict[int, str] = {}
    for stabilizer in stabilizers:
        for pauli, qubit in pauli_terms(stabilizer):
            existing = basis_by_qubit.get(qubit)
            if existing is not None:
                raise ValueError(
                    "Prepared stabilizers must not share qubits. "
                    f"Qubit {qubit} appears in both {existing} and {pauli}."
                )
            basis_by_qubit[qubit] = pauli

    x_targets = sorted(
        qubit for qubit, pauli in basis_by_qubit.items() if pauli == "X"
    )
    y_targets = sorted(
        qubit for qubit, pauli in basis_by_qubit.items() if pauli == "Y"
    )
    circuit = stim.Circuit()
    if x_targets:
        circuit.append_operation("H", x_targets)
    if y_targets:
        circuit.append_operation("H", y_targets)
        circuit.append_operation("S", y_targets)
    return circuit


def _select_independent_high_weight_exact(
    stabilizers: Sequence[str],
    supports: Sequence[Set[int]],
    weights: Sequence[int],
) -> List[int] | None:
    """
    Solve the high-weight disjoint-stabilizer packing exactly with CP-SAT.
    """
    if cp_model is None:
        return None

    model = cp_model.CpModel()
    selected = [
        model.NewBoolVar(f"prepared_stabilizer_{idx}")
        for idx in range(len(stabilizers))
    ]
    all_qubits = sorted(set().union(*supports) if supports else set())
    for qubit in all_qubits:
        sharing = [
            selected[idx]
            for idx, support in enumerate(supports)
            if qubit in support
        ]
        if sharing:
            model.Add(sum(sharing) <= 1)

    max_weight = max(weights, default=0)
    high_weight_terms = [
        selected[idx] for idx, weight in enumerate(weights)
        if weight == max_weight
    ]
    high_weight_count = sum(high_weight_terms)
    total_weight = sum(
        int(weight) * selected[idx] for idx, weight in enumerate(weights)
    )
    total_count = sum(selected)
    weight_scale = len(stabilizers) + 1
    high_scale = weight_scale * (sum(weights) + 1)
    model.Maximize(
        high_scale * high_weight_count
        + weight_scale * total_weight
        + total_count
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    return [
        idx for idx, var in enumerate(selected)
        if int(solver.Value(var)) == 1
    ]


def select_independent_high_weight_stabilizers(
    stabilizers: Sequence[str],
) -> List[int]:
    """
    Select disjoint prepared stabilizers, prioritizing highest-weight checks.

    The optimization is lexicographic through a scaled objective: maximize the
    number of maximum-weight stabilizers first, then the total number of
    prepared qubits, then the number of prepared stabilizers. For XYZ2 this
    means the bulk weight-6 plaquettes are favored before boundary or link
    checks are used to fill unused qubits.
    """
    supports = [pauli_support(stabilizer) for stabilizer in stabilizers]
    weights = [len(support) for support in supports]
    exact = _select_independent_high_weight_exact(
        stabilizers=stabilizers,
        supports=supports,
        weights=weights,
    )
    if exact is not None:
        return sorted(exact)

    used_qubits: Set[int] = set()
    selected: List[int] = []
    max_weight = max(weights, default=0)
    order = sorted(
        range(len(stabilizers)),
        key=lambda idx: (
            weights[idx] != max_weight,
            -weights[idx],
            idx,
        ),
    )
    for idx in order:
        support = supports[idx]
        if used_qubits.isdisjoint(support):
            selected.append(idx)
            used_qubits.update(support)
    return sorted(selected)


def build_preparation_plan(
    *,
    code_family: str,
    num_qubits: int,
    stabilizers: Sequence[str],
    logical_x: str,
    prep_mode: str,
) -> PreparationPlan:
    """
    Build a deterministic preparation plan for one MDR protocol mode.
    """
    if prep_mode not in PREP_MODES:
        raise ValueError(
            "prep_mode must be one of: " + ", ".join(PREP_MODES)
        )
    if code_family != "xyz2" and prep_mode != PREP_MODE_FULL_MDR:
        raise ValueError(
            "Only prep_mode='full_mdr' is currently supported for "
            f"code_family='{code_family}'."
        )

    all_indices = list(range(len(stabilizers)))
    if prep_mode == PREP_MODE_FULL_MDR:
        return PreparationPlan(
            mode=prep_mode,
            psi_circuit=logical_x_plus_circuit(logical_x),
            active_stabilizer_indices=all_indices,
            prepared_stabilizer_indices=[],
            include_logical_x=True,
            description=(
                "Historical MDR path: prepare Logical X support in the "
                "X basis, then measure all stabilizers and Logical X."
            ),
        )

    if prep_mode == PREP_MODE_LINK_LOGICAL_PLUS:
        prepared = [
            idx for idx, stabilizer in enumerate(stabilizers)
            if is_xyz2_link_stabilizer(stabilizer)
        ]
        active = [idx for idx in all_indices if idx not in set(prepared)]
        return PreparationPlan(
            mode=prep_mode,
            psi_circuit=plus_state_circuit(num_qubits),
            active_stabilizer_indices=active,
            prepared_stabilizer_indices=prepared,
            include_logical_x=False,
            description=(
                "All data qubits are prepared in |+>, so the XX links and "
                "Logical X start in their +1 eigenspaces. MDR measures only "
                "the remaining stabilizers."
            ),
        )

    prepared = select_independent_high_weight_stabilizers(stabilizers)
    active = [idx for idx in all_indices if idx not in set(prepared)]
    prepared_ops = [stabilizers[idx] for idx in prepared]
    return PreparationPlan(
        mode=prep_mode,
        psi_circuit=product_plus_eigenstate_circuit(prepared_ops),
        active_stabilizer_indices=active,
        prepared_stabilizer_indices=prepared,
        include_logical_x=True,
        description=(
            "Prepare an optimized disjoint set of high-weight stabilizers "
            "directly, then MDR-measure the remaining stabilizers plus "
            "Logical X."
        ),
    )
