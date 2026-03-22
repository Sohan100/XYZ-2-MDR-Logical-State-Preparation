from __future__ import annotations

import stim

from surface_code.logical_generator import (
    SurfaceCodeLogicalGenerator,
)
from surface_code.stabilizer_generator import (
    SurfaceCodeStabilizerGenerator,
)


def _sparse_to_stim(spec: str, num_qubits: int) -> stim.PauliString:
    chars = ["_"] * num_qubits
    for token in spec.split():
        chars[int(token[1:])] = token[0]
    return stim.PauliString("+" + "".join(chars))


def test_surface_code_stabilizer_counts_and_commutation() -> None:
    """
    The unrotated surface code should yield commuting X/Z checks.
    """
    generator = SurfaceCodeStabilizerGenerator(distance=3)
    stabilizers = generator.generate_stabilizers()

    assert generator.n == 13
    assert len(stabilizers) == 12
    assert {len(spec.split()) for spec in stabilizers} == {3, 4}

    paulis = [_sparse_to_stim(spec, generator.n) for spec in stabilizers]
    for i in range(len(paulis)):
        for j in range(i + 1, len(paulis)):
            assert paulis[i].commutes(paulis[j])


def test_surface_code_logicals_match_distance_and_commutation() -> None:
    """
    Boundary logical chains should commute with stabilizers and anti-commute.
    """
    distance = 5
    stabilizer_generator = SurfaceCodeStabilizerGenerator(distance=distance)
    logical_generator = SurfaceCodeLogicalGenerator(distance=distance)

    stabilizers = stabilizer_generator.generate_stabilizers()
    logicals = logical_generator.generate_logicals()
    logical_x = _sparse_to_stim(logicals["Logical X"], logical_generator.n)
    logical_z = _sparse_to_stim(logicals["Logical Z"], logical_generator.n)

    assert len(logicals["Logical X"].split()) == distance
    assert len(logicals["Logical Z"].split()) == distance
    assert not logical_x.commutes(logical_z)

    for stabilizer in stabilizers:
        stab = _sparse_to_stim(stabilizer, logical_generator.n)
        assert logical_x.commutes(stab)
        assert logical_z.commutes(stab)

    tableau = stim.Tableau.from_stabilizers(
        [
            _sparse_to_stim(spec, logical_generator.n)
            for spec in stabilizers + [logicals["Logical X"]]
        ]
    )
    assert len(tableau) == logical_generator.n
