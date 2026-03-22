"""
stabilizer_generator.py
-----------------------
Utility for generating stabilizer sets for the XYZ^2 Hex Code.

The generator encodes the doubled-layer diagonal indexing used by this
project's honeycomb patch and emits the full stabilizer set in the sparse
Pauli-string format. The resulting ordered list contains the weight-2 XX link
checks, the weight-6 plaquette checks, and the four boundary families needed
to keep the finite patch local while preserving a single logical qubit.
"""

from __future__ import annotations

from typing import List, Tuple


class XYZ2StabilizerGenerator:
    """
    Build the full stabilizer set for one XYZ^2 hex-code patch.

    The class precomputes the layer offsets needed to map logical lattice
    coordinates onto the repository's flattened physical-qubit indexing. That
    geometry is then reused to emit all stabilizers in a deterministic order
    suitable for tests, MDR-table generation, and downstream simulations.

    Attributes
    ----------
    d : int
        Odd code distance used to define the finite honeycomb patch.
    n : int
        Total number of physical qubits in the XYZ^2 code, equal to `2d^2`.
    _layer_offsets : List[int]
        Prefix offsets used to flatten the doubled diagonal layers of the
        honeycomb layout into a single qubit-index space.

    Methods
    -------
    __init__(distance)
        Validate the distance and precompute the diagonal-layer offsets that
        define the lattice indexing.
    generate_stabilizers()
        Emit the complete ordered stabilizer list for the finite patch.
    _coord_to_verts(i, j)
        Map one logical lattice coordinate onto the corresponding upper and
        lower physical vertices.
    """

    def __init__(self, distance: int) -> None:
        """
        Initialize the generator and pre-compute lattice geometry.

        Args:
            distance: Odd code distance `d` with `d >= 3`.

        Raises:
            ValueError: If `distance` is even or smaller than 3.
        """
        if distance < 3 or distance % 2 == 0:
            raise ValueError("Distance d must be odd and >= 3")

        self.d = distance
        self.n = 2 * distance * distance

        diag_counts = [
            self.d - abs((self.d - 1) - ell)
            for ell in range(2 * self.d - 1)
        ]

        self._layer_offsets: List[int] = []
        cur = 0
        for count in diag_counts:
            self._layer_offsets.append(cur)
            cur += count
            self._layer_offsets.append(cur)
            cur += count

    def generate_stabilizers(self) -> List[str]:
        """
        Compute the full ordered stabilizer list for distance `d`.

        The output order is:
        1. all vertical XX link checks,
        2. all bulk six-body plaquette checks,
        3. all four boundary families.

        Returns:
            List[str]: One sparse Pauli string per stabilizer generator.
        """
        stabs: List[str] = []

        for i in range(self.d):
            for j in range(self.d):
                up, lo = self._coord_to_verts(i, j)
                stabs.append(f"X{up} X{lo}")

        for i in range(self.d - 1):
            for j in range(self.d - 1):
                _, tl_l = self._coord_to_verts(i, j)
                tr_u, tr_l = self._coord_to_verts(i, j + 1)
                bl_u, bl_l = self._coord_to_verts(i + 1, j)
                br_u, _ = self._coord_to_verts(i + 1, j + 1)
                terms = [
                    f"X{tl_l}",
                    f"Z{bl_u}",
                    f"Y{tr_u}",
                    f"Y{bl_l}",
                    f"Z{tr_l}",
                    f"X{br_u}",
                ]
                stabs.append(" ".join(terms))

        for i in range(0, self.d - 1, 2):
            a_u, a_l = self._coord_to_verts(i, 0)
            b_u, _ = self._coord_to_verts(i + 1, 0)
            stabs.append(f"Y{a_u} Z{a_l} X{b_u}")

        for j in range(0, self.d - 1, 2):
            _, a_l = self._coord_to_verts(self.d - 1, j)
            b_u, b_l = self._coord_to_verts(self.d - 1, j + 1)
            stabs.append(f"X{a_l} Y{b_u} Z{b_l}")

        for j in range(1, self.d - 1, 2):
            a_u, a_l = self._coord_to_verts(0, j)
            b_u, _ = self._coord_to_verts(0, j + 1)
            stabs.append(f"Z{a_u} Y{a_l} X{b_u}")

        for i in range(1, self.d - 1, 2):
            _, a_l = self._coord_to_verts(i, self.d - 1)
            b_u, b_l = self._coord_to_verts(i + 1, self.d - 1)
            stabs.append(f"X{a_l} Z{b_u} Y{b_l}")

        return stabs

    def _coord_to_verts(self, i: int, j: int) -> Tuple[int, int]:
        """
        Map one logical lattice coordinate to its doubled physical vertices.

        Args:
            i: Logical row index inside the triangular patch coordinates.
            j: Logical column index inside the triangular patch coordinates.

        Returns:
            Tuple[int, int]: `(upper_vertex, lower_vertex)` indices for the
            doubled qubit at logical coordinate `(i, j)`.
        """
        ell = i + j
        max_i = min(self.d - 1, ell)
        pos = max_i - i
        up = self._layer_offsets[2 * ell] + pos
        lo = self._layer_offsets[2 * ell + 1] + pos
        return up, lo
