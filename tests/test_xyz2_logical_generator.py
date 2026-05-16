"""

test_xyz2_logical_generator.py
----------------------------------------------------------------------------
Pytest coverage for xyz2 logical generator behavior and regression checks.
"""

from __future__ import annotations

from xyz2.logical_generator import XYZ2LogicalGenerator


def test_d3_logicals_match_expected() -> None:
    """
    Validate expected distance-3 logical-operator strings.

    The expected values match the project’s reference operator definitions.

    Returns:
    None
    """
    logicals = XYZ2LogicalGenerator(distance=3).generate_logicals()
    assert logicals["Logical X"] == "X9 X10 X11"
    assert logicals["Logical Y"] == "X4 Z7 Y10 X13"
    assert logicals["Logical Z"] == "X4 Z7 X9 Z10 X11 X13"
