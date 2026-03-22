"""
logical_generator.py
--------------------
Utility for constructing logical operators for the unrotated surface code.

This module derives a canonical single-logical-qubit operator basis for the
unrotated planar surface-code patch used in this repository. The returned
operators follow the same sparse Pauli-string format used everywhere else in
the project so they can be passed directly into MDR-table generation and
simulation workflows.
"""

from __future__ import annotations

from typing import Dict, List

from .stabilizer_generator import SurfaceCodeStabilizerGenerator


class SurfaceCodeLogicalGenerator:
    """
    Generate logical Pauli operators for one unrotated planar surface-code patch.

    The generator is intentionally lightweight: it stores only the code
    distance, the number of data qubits, and the shared geometry helper used
    to map boundary paths into row-major physical-qubit indices.

    Attributes
    ----------
    d : int
        Odd code distance for the unrotated surface-code patch.
    n : int
        Total number of data qubits in the patch, equal to
        `d^2 + (d - 1)^2`.
    _geom : SurfaceCodeStabilizerGenerator
        Geometry helper reused to convert lattice coordinates into row-major
        physical-qubit indices.

    Methods
    -------
    __init__(distance)
        Validate the requested distance and initialize the shared geometry
        helper for the patch.
    generate_logicals()
        Return the canonical Logical X, Y, and Z operators for the patch.
    _get_logical_x()
        Construct the left-boundary X logical string.
    _get_logical_z()
        Construct the top-boundary Z logical string.
    _multiply_paulis(a_str, b_str)
        Multiply two sparse Pauli strings while discarding any global phase.
    """

    def __init__(self, distance: int) -> None:
        """
        Initialize the logical-operator generator.

        Args:
            distance: Odd code distance `d` with `d >= 3`.

        Raises:
            ValueError: If `distance` is even or smaller than 3.
        """
        if distance < 3 or distance % 2 == 0:
            raise ValueError("Distance d must be odd and >= 3")
        self.d = distance
        self.n = distance * distance + (distance - 1) * (distance - 1)
        self._geom = SurfaceCodeStabilizerGenerator(distance)

    def generate_logicals(self) -> Dict[str, str]:
        """
        Return Logical X, Y, and Z for the code patch.

        Logical X and Logical Z are built directly from opposite boundaries of
        the rotated patch, while Logical Y is obtained from their phase-agnostic
        Pauli product.

        Returns:
            Dict[str, str]: Mapping with keys `"Logical X"`, `"Logical Y"`,
            and `"Logical Z"`.
        """
        logical_x = self._get_logical_x()
        logical_z = self._get_logical_z()
        logical_y = self._multiply_paulis(logical_x, logical_z)
        return {
            "Logical X": logical_x,
            "Logical Y": logical_y,
            "Logical Z": logical_z,
        }

    def _get_logical_x(self) -> str:
        """
        Construct the left-boundary X string.

        Returns:
            str: Weight-`d` sparse Pauli string for Logical X.
        """
        qubits = [
            self._geom.coord_to_index(0, y)
            for y in range(0, 2 * self.d - 1, 2)
        ]
        return " ".join(f"X{q}" for q in qubits)

    def _get_logical_z(self) -> str:
        """
        Construct the top-boundary Z string.

        Returns:
            str: Weight-`d` sparse Pauli string for Logical Z.
        """
        qubits = [
            self._geom.coord_to_index(x, 0)
            for x in range(0, 2 * self.d - 1, 2)
        ]
        return " ".join(f"Z{q}" for q in qubits)

    def _multiply_paulis(self, a_str: str, b_str: str) -> str:
        """
        Multiply two sparse Pauli strings while discarding global phase.

        Args:
            a_str: First sparse Pauli string.
            b_str: Second sparse Pauli string.

        Returns:
            str: Sparse Pauli-string product with any global phase removed.
        """
        x_res = [0] * self.n
        z_res = [0] * self.n

        def apply_string(spec: str) -> None:
            for token in spec.split():
                pauli = token[0]
                qubit = int(token[1:])
                if pauli in {"X", "Y"}:
                    x_res[qubit] ^= 1
                if pauli in {"Z", "Y"}:
                    z_res[qubit] ^= 1

        apply_string(a_str)
        apply_string(b_str)

        tokens: List[str] = []
        for qubit in range(self.n):
            x_bit = x_res[qubit]
            z_bit = z_res[qubit]
            if x_bit and z_bit:
                tokens.append(f"Y{qubit}")
            elif x_bit:
                tokens.append(f"X{qubit}")
            elif z_bit:
                tokens.append(f"Z{qubit}")
        return " ".join(tokens)
