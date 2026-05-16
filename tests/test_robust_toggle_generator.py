"""

test_robust_toggle_generator.py
----------------------------------------------------------------------------
Pytest coverage for robust toggle generator behavior and regression checks.
"""

from __future__ import annotations

import pytest

from mdr.robust_toggle_generator import (
    RobustToggleGenerator,
    cp_model,
)
from xyz2.logical_generator import XYZ2LogicalGenerator
from xyz2.stabilizer_generator import XYZ2StabilizerGenerator


@pytest.mark.parametrize("distance", [3, 5])
def test_toggles_commutation_structure(distance: int) -> None:
    """
    Check toggle commutation structure against stabilizer constraints.

    For each generated toggle, the test verifies anti-commutation with its
    paired constraint and commutation with all others.

    Returns:
    None
    """
    n = 2 * distance * distance
    stabs = XYZ2StabilizerGenerator(distance).generate_stabilizers()
    logical_x = XYZ2LogicalGenerator(distance).generate_logicals()["Logical X"]

    tg = RobustToggleGenerator(stabs, logical_x, n, random_seed=0)
    stab_toggles, logical_toggle = tg.generate_toggles()

    constraints = stabs + [logical_x]
    toggles = stab_toggles + [logical_toggle]
    assert len(constraints) == len(toggles)

    for i, toggle in enumerate(toggles):
        v_toggle = tg._str_to_vec_standard(toggle)
        for j, constraint in enumerate(constraints):
            v_constraint = tg._str_to_vec_standard(constraint)
            commutes = tg._symp_product(v_toggle, v_constraint) == 0
            expected_commutes = i != j
            assert commutes == expected_commutes, (
                f"toggle idx={i} vs constraint idx={j} had "
                f"commutes={commutes}, expected {expected_commutes}"
            )

    # sanity: no all-identity output for d=3 constraints
    assert not any(toggle == "I" for toggle in toggles)
    assert all(len(toggle.split()) >= 1 for toggle in toggles)


@pytest.mark.parametrize("distance", [3, 5])
def test_toggles_commute_pairwise(distance: int) -> None:
    """
    The synthesized toggle family should commute internally.

    Returns:
    None
    """
    n = 2 * distance * distance
    stabs = XYZ2StabilizerGenerator(distance).generate_stabilizers()
    logical_x = XYZ2LogicalGenerator(distance).generate_logicals()["Logical X"]

    tg = RobustToggleGenerator(stabs, logical_x, n, random_seed=0)
    stab_toggles, logical_toggle = tg.generate_toggles()
    toggles = stab_toggles + [logical_toggle]
    toggle_vecs = [tg._str_to_vec_standard(toggle) for toggle in toggles]

    for i in range(len(toggle_vecs)):
        for j in range(i + 1, len(toggle_vecs)):
            assert tg._symp_product(toggle_vecs[i], toggle_vecs[j]) == 0


@pytest.mark.skipif(
    cp_model is None,
    reason="exact toggle-weight optimizer is unavailable in this environment",
)
def test_toggle_weight_reduction_regression_d5() -> None:
    """
    The commuting exact post-pass should stay well below tableau weights.

    Returns:
    None
    """
    d = 5
    n = 2 * d * d
    stabs = XYZ2StabilizerGenerator(d).generate_stabilizers()
    logical_x = XYZ2LogicalGenerator(d).generate_logicals()["Logical X"]

    tg = RobustToggleGenerator(stabs, logical_x, n, random_seed=0)
    stab_toggles, logical_toggle = tg.generate_toggles()
    weights = [
        len(toggle.split()) for toggle in stab_toggles + [logical_toggle]
    ]

    assert sum(weights) <= 139
    assert max(weights) <= 7
    assert set(tg.last_optimization_statuses) == {"exact_optimal"}
