from __future__ import annotations

from collections import Counter

import pytest

from xyz2_mdr.xyz2_stabilizer_generator import XYZ2StabilizerGenerator


def test_d5_stabilizer_weight_counts() -> None:
    """
    Check distance-5 stabilizer count distribution by operator weight.

    The expected counts follow the XYZ^2 lattice construction rules.

    Returns:
        None
    """
    stabs = XYZ2StabilizerGenerator(distance=5).generate_stabilizers()
    weight_counts = Counter(len(s.split()) for s in stabs)
    assert weight_counts[2] == 25
    assert weight_counts[6] == 16
    assert weight_counts[3] == 8
    assert len(stabs) == 49


@pytest.mark.parametrize("invalid_distance", [0, 2, 4], ids=["d0", "d2", "d4"])
def test_distance_validation(invalid_distance: int) -> None:
    """
    Ensure invalid code distances raise `ValueError`.

    The generator only supports odd distances greater than or equal to 3.

    Returns:
        None
    """
    with pytest.raises(ValueError, match="odd and >= 3"):
        XYZ2StabilizerGenerator(distance=invalid_distance)
