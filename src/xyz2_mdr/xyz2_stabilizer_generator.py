"""
xyz2_stabilizer_generator.py
────────────────────────────────────────────────────────────────────────────
Utility for generating stabilizer sets for the XYZ^2 Hex Code.
"""

from __future__ import annotations

from typing import List, Tuple


class XYZ2StabilizerGenerator:
    """
    Generator that produces stabilizer strings for the XYZ^2 Hex Code.
    Builds the full stabilizer set of the XYZ^2 hex code from the lattice
    geometry at a chosen odd code distance.

    Attributes
    ----------
    d : int
        Code distance.
    n : int
        Total number of physical qubits, equal to `2 * d^2`.
    _layer_offsets : List[int]
        Precomputed offsets used to map lattice coordinates to upper and
        lower qubit indices.

    Methods
    -------
    __init__(...)
        Initialize the stabilizer generator and precompute lattice geometry.
    generate_stabilizers()
        Return the full ordered list of stabilizer generators.
    _coord_to_verts(i, j)
        Map a logical lattice coordinate to the corresponding upper and lower
        physical qubit indices.
    """

    # ─────────────────────────────────────────────────────────────────────
    # construction
    # ─────────────────────────────────────────────────────────────────────
    def __init__(self, distance: int) -> None:
        """
        Initialise the generator and pre-compute lattice geometry.
        
        Args:
            distance: Code distance `d`. Must be an odd integer >= 3.
        
        Raises:
            ValueError: If d is even or less than 3.
        """
        if distance < 3 or distance % 2 == 0:
            raise ValueError("Distance d must be odd and >= 3")

        self.d = distance
        self.n = 2 * distance * distance

        # diag_counts is the number of links in each diagonal slice.
        diag_counts = [
            self.d - abs((self.d - 1) - ell)
            for ell in range(2 * self.d - 1)
        ]

        self._layer_offsets: List[int] = []
        cur = 0
        for count in diag_counts:
            self._layer_offsets.append(cur)  # upper layer
            cur += count
            self._layer_offsets.append(cur)  # lower layer
            cur += count

    # ─────────────────────────────────────────────────────────────────────
    # public api
    # ─────────────────────────────────────────────────────────────────────
    def generate_stabilizers(self) -> List[str]:
        """
        Compute the full list of stabilizer generators for distance d.

        The output ordering is deterministic and grouped by geometric type:
        vertical `XX` links first, then bulk plaquettes, followed by the four
        boundary families.

        Returns:
            List[str]: Sparse Pauli strings such as
            `["X0 X1", "Y0 Z1 X2", ...]`.
        """
        stabs: List[str] = []

        # (1) Vertical XX links.
        for i in range(self.d):
            for j in range(self.d):
                up, lo = self._coord_to_verts(i, j)
                stabs.append(f"X{up} X{lo}")

        # (2) Bulk XYZXYZ plaquettes (weight 6).
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

        # (3) Boundary half-hexes (weight 3).
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

    # ─────────────────────────────────────────────────────────────────────
    # geometry helpers
    # ─────────────────────────────────────────────────────────────────────
    def _coord_to_verts(self, i: int, j: int) -> Tuple[int, int]:
        """
        Map logical 2D coordinate (i,j) to physical qubit indices.

        The lattice is indexed by diagonal slices `ell = i + j`, and each
        slice has an upper and lower layer. This helper converts a lattice
        cell coordinate into the pair `(upper_vertex, lower_vertex)` used by
        the stabilizer and logical-construction routines.

        Args:
            i: Row index in the range `0 .. d - 1`.
            j: Column index in the range `0 .. d - 1`.

        Returns:
            Tuple[int, int]: `(upper_qubit_index, lower_qubit_index)`.
        """
        ell = i + j
        max_i = min(self.d - 1, ell)
        # Order i decreasing implies left-to-right in each diagonal strip.
        pos = max_i - i
        up = self._layer_offsets[2 * ell] + pos
        lo = self._layer_offsets[2 * ell + 1] + pos
        return up, lo

