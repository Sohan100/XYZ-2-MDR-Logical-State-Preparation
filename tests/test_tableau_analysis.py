from __future__ import annotations

from mdr.robust_toggle_generator import RobustToggleGenerator
from xyz2.logical_generator import XYZ2LogicalGenerator
from xyz2.stabilizer_generator import XYZ2StabilizerGenerator
from xyz2.tableau_analysis import build_destabilizer_report


def test_build_destabilizer_report_has_expected_shape_d3() -> None:
    """
    The report should include one row for each stabilizer plus Logical X.

    Returns:
        None
    """
    report = build_destabilizer_report(distance=3)

    assert len(report) == 18
    assert list(report.columns) == [
        "distance",
        "index",
        "category",
        "label",
        "constraint",
        "generated_toggle",
        "generated_weight",
        "optimization_status",
        "stim_destabilizer",
        "stim_weight",
        "weight_delta",
    ]
    assert report["label"].iloc[-1] == "Logical X"
    assert report["category"].iloc[-1] == "Logical"
    assert report["optimization_status"].isin(
        ["heuristic", "exact_optimal", "exact_feasible", "exact_fallback"]
    ).all()


def test_stim_destabilizers_form_dual_basis_d3() -> None:
    """
    Stim's extracted destabilizers should anti-commute with only one row.

    Returns:
        None
    """
    distance = 3
    num_qubits = 2 * distance * distance
    stabilizers = XYZ2StabilizerGenerator(distance).generate_stabilizers()
    logical_x = XYZ2LogicalGenerator(distance).generate_logicals()["Logical X"]
    constraints = stabilizers + [logical_x]
    generator = RobustToggleGenerator(stabilizers, logical_x, num_qubits)
    report = build_destabilizer_report(distance=distance)

    for idx, stim_destabilizer in enumerate(report["stim_destabilizer"]):
        destab_vec = generator._str_to_vec_standard(stim_destabilizer)
        for jdx, constraint in enumerate(constraints):
            constraint_vec = generator._str_to_vec_standard(constraint)
            commutes = generator._symp_product(destab_vec, constraint_vec) == 0
            assert commutes == (idx != jdx)
