"""
xyz2_logical_generator.py
────────────────────────────────────────────────────────────────────────────
Utility for constructing logical Pauli operators for the XYZ^2 Hex Code.
"""
from typing import Dict, List, Tuple
import math

class XYZ2LogicalGenerator:
    """
    Generator that produces Logical X, Y, and Z operators for the XYZ^2 code.

    The logical operators are constructed based on the code distance `d`:
    - Logical X: A diagonal strip of X operators.
    - Logical Y: A specific "butterfly" pattern centered at ((d-1)/2, (d-1)/2).
    - Logical Z: Calculated as the product Logical X * Logical Y.

    Attributes
    ----------
    d : int
        Code distance.
    n : int
        Total number of qubits (2 * d^2).
    _stabilizer_gen : XYZ2StabilizerGenerator
        Internal instance for coordinate mapping logic.

    Methods
    -------
    generate_logicals()
        Return a dictionary containing 'Logical X', 'Logical Y', and 'Logical Z'.
    _get_logical_x()
        Construct the Logical X string.
    _get_logical_y()
        Construct the Logical Y string.
    _multiply_paulis(op_a, op_b)
        Compute the product of two Pauli strings (ignoring global phase).

    Example
    -------
    >>> log_gen = XYZ2LogicalGenerator(distance=3)
    >>> log_gen.generate_logicals()
    {'Logical X': 'X4 X5', 'Logical Y': 'X4 Z7 Y10 X13', 'Logical Z': ...}
    """

    # ─────────────────────────────────────────────────────────────────────
    # construction
    # ─────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        distance: int
    ) -> None:
        """
        Initialise the logical operator generator.

        Args:
            distance: Code distance `d`. Must be an odd integer >= 3.

        Returns:
            None
        """
        self.d = distance
        self.n = 2 * distance * distance
        # reuse geometry logic from stabilizer generator
        self._geom = XYZ2StabilizerGenerator(distance)

    # ─────────────────────────────────────────────────────────────────────
    # public api
    # ─────────────────────────────────────────────────────────────────────
    def generate_logicals(
        self
    ) -> Dict[str, str]:
        """
        Compute Logical X, Y, and Z operators.

        Returns
        -------
        Dict[str, str]
            Dictionary keys: "Logical X", "Logical Y", "Logical Z".
            Values are sparse Pauli strings.
        """
        lx = self._get_logical_x()
        ly = self._get_logical_y()
        lz = self._multiply_paulis(lx, ly)
        
        return {
            "Logical X": lx,
            "Logical Y": ly,
            "Logical Z": lz
        }

    # ─────────────────────────────────────────────────────────────────────
    # operator construction
    # ─────────────────────────────────────────────────────────────────────
    def _get_logical_x(self) -> str:
        """
        Construct Logical X along the main diagonal.
        Corresponds to 'lower' vertices where j = d - 1 - i.

        Returns:
            str: Sparse Pauli string for Logical X.
        """
        qs: List[int] = []
        for i in range(self.d):
            j = self.d - 1 - i
            # We target the lower qubit on this diagonal
            _, lo = self._geom._coord_to_verts(i, j)
            qs.append(lo)
        
        qs.sort()
        return " ".join([f"X{q}" for q in qs])

    def _get_logical_y(self) -> str:
        """
        Construct Logical Y using the center-block strategy.
        
        Structure:
        1. Center block (c,c): Z on upper vertex, Y on lower vertex.
        2. Lower diagonal (i+j = d-2): X on lower vertices.
        3. Upper diagonal (i+j = d): X on upper vertices.
        """
        c = (self.d - 1) // 2
        
        # Center block
        u_cc, l_cc = self._geom._coord_to_verts(c, c)
        
        toks: List[Tuple[int, str]] = []
        toks.append((u_cc, "Z"))
        toks.append((l_cc, "Y"))

        r = (self.d - 1) // 2

        # Left X cluster: lower vertices on diagonal i + j = d - 2
        for t in range(r):
            i = (self.d - 2) - t
            j = t
            _, lo = self._geom._coord_to_verts(i, j)
            toks.append((lo, "X"))

        # Right X cluster: upper vertices on diagonal i + j = d
        for t in range(r):
            i = c - t
            j = self.d - i
            up, _ = self._geom._coord_to_verts(i, j)
            toks.append((up, "X"))

        # Sort by qubit index for canonical string format
        toks.sort(key=lambda x: x[0])
        return " ".join([f"{p}{q}" for q, p in toks])

    # ─────────────────────────────────────────────────────────────────────
    # pauli algebra helpers
    # ─────────────────────────────────────────────────────────────────────
    def _multiply_paulis(self, a_str: str, b_str: str) -> str:
        """
        Symbolic multiplication of two Pauli strings A * B.
        
        Used to derive Logical Z = Logical X * Logical Y.
        Ignores global phase.
        
        Args:
            a_str: First Pauli string (e.g., "X0 Z1").
            b_str: Second Pauli string (e.g., "Y0 X2").
            
        Returns:
            Resulting Pauli string (e.g., "Z0 Z1 X2").
        """
        # 1. Parse into dense arrays
        x_res = [0] * self.n
        z_res = [0] * self.n
        
        def apply_string(s):
            for tok in s.split():
                if not tok: continue
                p, q_idx = tok[0], int(tok[1:])
                if p in ("X", "Y"): x_res[q_idx] ^= 1
                if p in ("Z", "Y"): z_res[q_idx] ^= 1

        apply_string(a_str)
        apply_string(b_str)

        # 2. Convert back to string
        out_toks = []
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
