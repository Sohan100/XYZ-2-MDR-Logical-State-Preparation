"""

logical_generator.py
----------------------------------------------------------------------------
logical_generator.py.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .stabilizer_generator import XYZ2StabilizerGenerator


class XYZ2LogicalGenerator:
    """
    Generate Logical X, Y, and Z operators for one XYZ^2 code instance.

    The class owns only geometric metadata and does not cache a logical table.
    Each call to :meth:`generate_logicals` rebuilds the sparse logical strings
    from the underlying lattice coordinates so the output always reflects the
    chosen distance and the same coordinate conventions used by the stabilizer
    generator.

    Attributes
    ----------
    d : int Odd code distance associated with this logical-operator generator.
    n : int Total number of physical qubits in the `[[2d^2, 1, d]]` XYZ^2 code.
    _geom : XYZ2StabilizerGenerator Shared geometry helper used to map logical
    lattice coordinates onto flattened physical-qubit indices.

    Methods
    -------
    __init__(distance) Validate the requested code distance and initialize the
    geometry helper used by the logical construction routines.
    generate_logicals() Build the canonical dictionary containing Logical X,
    Logical Y, and Logical Z in sparse Pauli-string format. _get_logical_x()
    Construct the weight-`d` anti-diagonal Logical X operator. _get_logical_y()
    Construct the mixed-Pauli Logical Y operator using the center-block pattern
    used in the project notes. _multiply_paulis(a_str, b_str) Multiply two
    sparse Pauli strings while discarding any global phase.
    """

    def __init__(self, distance: int) -> None:
        """
        Initialize the logical-operator generator for one code distance.

        Args:
        distance: Odd code distance `d` with `d >= 3`.

        Raises:
        ValueError: If `distance` is even or smaller than 3.
        """
        if distance < 3 or distance % 2 == 0:
            raise ValueError("Distance d must be odd and >= 3")
        self.d = distance
        self.n = 2 * distance * distance
        self._geom = XYZ2StabilizerGenerator(distance)

    def generate_logicals(self) -> Dict[str, str]:
        """
        Build the canonical logical-operator dictionary.

        Logical X is constructed first from the anti-diagonal lower-vertex
        strip, Logical Y is built from the center-block pattern, and Logical Z
        is computed as the phase-agnostic Pauli product of X and Y.

        Returns:
        Dict[str, str]: Mapping with keys `"Logical X"`, `"Logical Y"`, and
        `"Logical Z"`.
        """
        logical_x = self._get_logical_x()
        logical_y = self._get_logical_y()
        logical_z = self._multiply_paulis(logical_x, logical_y)
        return {
            "Logical X": logical_x,
            "Logical Y": logical_y,
            "Logical Z": logical_z,
        }

    def _get_logical_x(self) -> str:
        """
        Construct Logical X along the anti-diagonal lower-vertex strip.

        The support is chosen by walking one site per layer along the
        anti-diagonal and selecting the lower vertex of each doubled-qubit
        coordinate pair. The resulting operator has weight `d`.

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
        Construct Logical Y using the project center-block strategy.

        This helper reproduces the mixed-Pauli logical used elsewhere in the
        repository. It starts from the central doubled qubit, places the
        required `Z`/`Y` pair there, then extends the operator with the two
        flanking `X` arms whose endpoints depend on the code radius.

        Returns:
        str: Sparse Pauli string for Logical Y.
        """
        center = (self.d - 1) // 2
        u_cc, l_cc = self._geom._coord_to_verts(center, center)
        toks: List[Tuple[int, str]] = [(u_cc, "Z"), (l_cc, "Y")]

        radius = (self.d - 1) // 2

        for t in range(radius):
            i = (self.d - 2) - t
            j = t
            _, lo = self._geom._coord_to_verts(i, j)
            toks.append((lo, "X"))

        for t in range(radius):
            i = center - t
            j = self.d - i
            up, _ = self._geom._coord_to_verts(i, j)
            toks.append((up, "X"))

        toks.sort(key=lambda item: item[0])
        return " ".join(f"{pauli}{q}" for q, pauli in toks)

    def _multiply_paulis(self, a_str: str, b_str: str) -> str:
        """
        Multiply two sparse Pauli strings while discarding global phase.

        The multiplication is performed by tracking the X and Z support bits on
        every qubit, then converting the combined support back into the sparse
        Pauli format expected by the rest of the project.

        Args:
        a_str: First sparse Pauli string. b_str: Second sparse Pauli string.

        Returns:
        str: Product operator in sparse format, with any global phase removed.
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
