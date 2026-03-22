from __future__ import annotations

import threshold_experiments
from mdr import MDRCircuit
from surface_code import (
    SurfaceCodeLogicalGenerator,
    SurfaceCodeStabilizerGenerator,
)
from xyz2 import (
    XYZ2LogicalGenerator,
    XYZ2StabilizerGenerator,
    build_destabilizer_report,
)


def test_canonical_packages_expose_core_classes() -> None:
    """
    Canonical package names should expose the project's public helpers.
    """
    assert MDRCircuit.__name__ == "MDRCircuit"
    assert XYZ2LogicalGenerator.__name__ == "XYZ2LogicalGenerator"
    assert SurfaceCodeLogicalGenerator.__name__ == "SurfaceCodeLogicalGenerator"
    assert callable(build_destabilizer_report)


def test_threshold_experiments_namespace_is_not_a_wrapper_surface() -> None:
    """
    Threshold experiments should stay a namespace package, not a mirror export.
    """
    assert threshold_experiments.__all__ == [
        "decoders",
        "experiments",
        "noise_models",
    ]


def test_code_family_generators_smoke() -> None:
    """
    Canonical family packages should generate the expected logical operators.
    """
    xyz2_logicals = XYZ2LogicalGenerator(3).generate_logicals()
    surface_logicals = SurfaceCodeLogicalGenerator(3).generate_logicals()

    assert xyz2_logicals["Logical X"] == "X9 X10 X11"
    assert surface_logicals["Logical X"] == "X0 X5 X10"
    assert len(XYZ2StabilizerGenerator(3).generate_stabilizers()) > 0
    assert len(SurfaceCodeStabilizerGenerator(3).generate_stabilizers()) > 0
