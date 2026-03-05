"""
xyz2_stabilizer_generator.py
────────────────────────────────────────────────────────────────────────────
Utility for generating stabilizer sets for the XYZ^2 Hex Code.
"""
from typing import List, Tuple, Dict
import math

class XYZ2StabilizerGenerator:
    """
    Generator that produces stabilizer strings for the XYZ^2 Hex Code.

    The XYZ^2 code is defined on a honeycomb lattice with lattice 
    parameter `d`. The code typically employs:
        - Weight-2 XX links on vertical edges.
        - Weight-6 XYZXYZ plaquettes in the bulk.
        - Weight-3 boundary checks.

    Attributes
    ----------
    d : int
        Code distance (must be odd and >= 3).
    n : int
        Total number of qubits (2 * d^2).
    _layer_offsets : list[int]
        Precomputed offsets for mapping (i,j) coordinates to qubit indices.

    Methods
    -------
    generate_stabilizers()
        Return a list of sparse Pauli strings representing the stabilizers.
    _coord_to_verts(i, j)
        Map 2D lattice coordinates (i, j) to the pair of qubit indices (up, lo).

    Example
    -------
    >>> gen = XYZ2StabilizerGenerator(distance=3)
    >>> gen.generate_stabilizers()
    ['X0 X1', 'Y0 Z1 X2', ...]
    """

    # ─────────────────────────────────────────────────────────────────────
    # construction
    # ─────────────────────────────────────────────────────────────────────
    def __init__(
        self, 
        distance: int
    ) -> None:
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
        
        # Precompute layer offsets for coordinate mapping
        # diag_counts represents the number of vertical links in each diagonal slice
        diag_counts = [
            self.d - abs((self.d - 1) - ell) 
            for ell in range(2 * self.d - 1)
        ]

        self._layer_offsets = []
        cur = 0
        for c in diag_counts:
            self._layer_offsets.append(cur)   # upper layer for this diagonal
            cur += c
            self._layer_offsets.append(cur)   # lower layer for this diagonal
            cur += c

    # ─────────────────────────────────────────────────────────────────────
    # public api
    # ─────────────────────────────────────────────────────────────────────
    def generate_stabilizers(
        self
    ) -> List[str]:
        """
        Compute the full list of stabilizer generators for distance d.

        Returns
        -------
        List[str]
            List of Pauli strings (e.g., ["X0 X1", "Y0 Z1 X2", ...]).
        """
        stabs: List[str] = []

        # (1) Vertical XX links
        # Defined for every coordinate pair (i, j) in the lattice
        for i in range(self.d):
            for j in range(self.d):
                up, lo = self._coord_to_verts(i, j)
                stabs.append(f"X{up} X{lo}")

        # (2) Bulk XYZXYZ hex plaquettes (weight 6)
        # Defined for the bulk squares
        for i in range(self.d - 1):
            for j in range(self.d - 1):
                # Get vertices for the four corners of the plaquette
                # Top-Left, Top-Right, Bottom-Left, Bottom-Right
                _, tl_l = self._coord_to_verts(i, j)
                tr_u, tr_l = self._coord_to_verts(i, j + 1)
                bl_u, bl_l = self._coord_to_verts(i + 1, j)
                br_u, _ = self._coord_to_verts(i + 1, j + 1)

                # Order: TL_low -> BL_up -> TR_up -> BL_low -> TR_low -> BR_up
                terms = [
                    f"X{tl_l}", f"Z{bl_u}", f"Y{tr_u}",
                    f"Y{bl_l}", f"Z{tr_l}", f"X{br_u}"
                ]
                stabs.append(" ".join(terms))

        # (3) Boundary half-hexes (weight 3)
        
        # Left side boundaries
        for i in range(0, self.d - 1, 2):
            a_u, a_l = self._coord_to_verts(i, 0)
            b_u, _   = self._coord_to_verts(i + 1, 0)
            stabs.append(f"Y{a_u} Z{a_l} X{b_u}")

        # Bottom side boundaries
        for j in range(0, self.d - 1, 2):
            _, a_l = self._coord_to_verts(self.d - 1, j)
            b_u, b_l = self._coord_to_verts(self.d - 1, j + 1)
            stabs.append(f"X{a_l} Y{b_u} Z{b_l}")

        # Top side boundaries
        for j in range(1, self.d - 1, 2):
            a_u, a_l = self._coord_to_verts(0, j)
            b_u, _   = self._coord_to_verts(0, j + 1)
            stabs.append(f"Z{a_u} Y{a_l} X{b_u}")

        # Right side boundaries
        for i in range(1, self.d - 1, 2):
            _, a_l   = self._coord_to_verts(i, self.d - 1)
            b_u, b_l = self._coord_to_verts(i + 1, self.d - 1)
            stabs.append(f"X{a_l} Z{b_u} Y{b_l}")

        return stabs

    # ─────────────────────────────────────────────────────────────────────
    # geometry helpers
    # ─────────────────────────────────────────────────────────────────────
    def _coord_to_verts(self, i: int, j: int) -> Tuple[int, int]:
        """
        Map logical 2D coordinate (i,j) to physical qubit indices.

        The lattice is indexed by diagonals ell = i + j.
        
        Args:
            i: Row index (0 to d-1)
            j: Column index (0 to d-1)

        Returns:
            Tuple[int, int]: (upper_qubit_index, lower_qubit_index)
        """
        ell = i + j
        max_i = min(self.d - 1, ell)
        # Order i decreasing implies left-to-right within a diagonal strip
        pos = max_i - i  
        
        up = self._layer_offsets[2 * ell] + pos
        lo = self._layer_offsets[2 * ell + 1] + pos
        return up, lo