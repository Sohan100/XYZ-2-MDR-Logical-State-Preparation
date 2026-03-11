"""
xyz2_logical_generator.py
────────────────────────────────────────────────────────────────────────────
Utility for constructing logical Pauli operators for the XYZ^2 Hex Code.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .xyz2_stabilizer_generator import XYZ2StabilizerGenerator


class XYZ2LogicalGenerator:
    """
    Generator that produces Logical X, Y, and Z operators for the XYZ^2 code.
    Constructs the sparse Pauli representatives of Logical X, Logical Y, and
    Logical Z using the XYZ^2 lattice geometry.

    Attributes
    ----------
    d : int
        Code distance.
    n : int
        Total number of physical qubits, equal to `2 * d^2`.
    _geom : XYZ2StabilizerGenerator
        Geometry helper used for coordinate-to-qubit mapping.

    Methods
    -------
    __init__(...)
        Initialize the logical-operator generator for a given code distance.
    generate_logicals()
        Return a mapping containing Logical X, Logical Y, and Logical Z.
    _get_logical_x()
        Construct the Logical X operator string.
    _get_logical_y()
        Construct the Logical Y operator string.
    _multiply_paulis(a_str, b_str)
        Multiply two sparse Pauli strings while discarding global phase.
    """

    # ─────────────────────────────────────────────────────────────────────
    # construction
    # ─────────────────────────────────────────────────────────────────────
    def __init__(self, distance: int) -> None:
        """
        Initialise the logical operator generator.

        The constructor validates `distance` and creates an internal
        geometry helper used to map lattice coordinates to qubit indices.

        Args:
            distance: Code distance `d`. Must be an odd integer >= 3.

        Raises:
            ValueError: If `distance` is even or less than 3.
        """
        if distance < 3 or distance % 2 == 0:
            raise ValueError("Distance d must be odd and >= 3")
        self.d = distance
        self.n = 2 * distance * distance
        self._geom = XYZ2StabilizerGenerator(distance)

    # ─────────────────────────────────────────────────────────────────────
    # public api
    # ─────────────────────────────────────────────────────────────────────
    def generate_logicals(self) -> Dict[str, str]:
        """
        Compute Logical X, Y, and Z operators.

        Returns:
            Dict[str, str]: Mapping with keys `"Logical X"`, `"Logical Y"`,
            and `"Logical Z"`, where each value is a sparse Pauli string.
        """
        logical_x = self._get_logical_x()
        logical_y = self._get_logical_y()
        logical_z = self._multiply_paulis(logical_x, logical_y)
        return {
            "Logical X": logical_x,
            "Logical Y": logical_y,
            "Logical Z": logical_z,
        }

    # ─────────────────────────────────────────────────────────────────────
    # operator construction
    # ─────────────────────────────────────────────────────────────────────
    def _get_logical_x(self) -> str:
        """
        Construct Logical X along the anti-diagonal lower-vertex strip.

        The support is chosen from lattice coordinates satisfying
        `j = d - 1 - i`, taking the lower qubit at each coordinate pair.

        Returns:
            str: Sparse Pauli string for Logical X.
        """
        qubits: List[int] = []
        for i in range(self.d):
            j = self.d - 1 - i
            _, lo = self._geom._coord_to_verts(i, j)
            qubits.append(lo)
        qubits.sort()
        return " ".join(f"X{q}" for q in qubits)

    def _get_logical_y(self) -> str:
        """
        Construct Logical Y using the center-block strategy.

        The support is built from three pieces:
        1. A central two-qubit block with `Z` on the upper vertex and `Y` on
           the lower vertex.
        2. A lower-diagonal cluster of `X` operators on lower vertices with
           `i + j = d - 2`.
        3. An upper-diagonal cluster of `X` operators on upper vertices with
           `i + j = d`.

        Returns:
            str: Sparse Pauli string for Logical Y.
        """
        center = (self.d - 1) // 2
        u_cc, l_cc = self._geom._coord_to_verts(center, center)
        toks: List[Tuple[int, str]] = [(u_cc, "Z"), (l_cc, "Y")]

        radius = (self.d - 1) // 2

        # Left X cluster: lower vertices on diagonal i + j = d - 2.
        for t in range(radius):
            i = (self.d - 2) - t
            j = t
            _, lo = self._geom._coord_to_verts(i, j)
            toks.append((lo, "X"))

        # Right X cluster: upper vertices on diagonal i + j = d.
        for t in range(radius):
            i = center - t
            j = self.d - i
            up, _ = self._geom._coord_to_verts(i, j)
            toks.append((up, "X"))

        toks.sort(key=lambda item: item[0])
        return " ".join(f"{pauli}{q}" for q, pauli in toks)

    # ─────────────────────────────────────────────────────────────────────
    # pauli algebra helpers
    # ─────────────────────────────────────────────────────────────────────
    def _multiply_paulis(self, a_str: str, b_str: str) -> str:
        """
        Multiply two sparse Pauli strings while discarding global phase.

        This helper accumulates the X and Z support bits implied by each input
        string and then reconstructs the resulting sparse Pauli operator. It
        is used to derive Logical Z from Logical X and Logical Y.

        Args:
            a_str: First Pauli string (e.g., "X0 Z1").
            b_str: Second Pauli string (e.g., "Y0 X2").

        Returns:
            str: Resulting sparse Pauli string (for example
            `"Z0 Z1 X2"`).
        """
        x_res = [0] * self.n
        z_res = [0] * self.n

        def apply_string(spec: str) -> None:
            for tok in spec.split():
                pauli, q_idx = tok[0], int(tok[1:])
                if pauli in ("X", "Y"):
                    x_res[q_idx] ^= 1
                if pauli in ("Z", "Y"):
                    z_res[q_idx] ^= 1

        apply_string(a_str)
        apply_string(b_str)

        out_toks: List[str] = []
        for q in range(self.n):
            x_bit = x_res[q]
            z_bit = z_res[q]
            if x_bit and z_bit:
                out_toks.append(f"Y{q}")
            elif x_bit:
                out_toks.append(f"X{q}")
            elif z_bit:
                out_toks.append(f"Z{q}")
        return " ".join(out_toks)

