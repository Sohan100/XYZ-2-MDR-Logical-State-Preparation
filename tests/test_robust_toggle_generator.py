from __future__ import annotations

from xyz2_mdr.robust_toggle_generator import RobustToggleGenerator
from xyz2_mdr.xyz2_logical_generator import XYZ2LogicalGenerator
from xyz2_mdr.xyz2_stabilizer_generator import XYZ2StabilizerGenerator


def test_toggles_commutation_structure_d3() -> None:
    """
    Check toggle commutation structure against stabilizer constraints.

    For each generated toggle, the test verifies anti-commutation with its
    paired constraint and commutation with all others.

    Returns:
        None
    """
    d = 3
    n = 2 * d * d
    stabs = XYZ2StabilizerGenerator(d).generate_stabilizers()
    logical_x = XYZ2LogicalGenerator(d).generate_logicals()["Logical X"]

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
