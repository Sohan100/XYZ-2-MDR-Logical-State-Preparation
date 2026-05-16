"""

stabilizer_generator.py
----------------------------------------------------------------------------
stabilizer_generator.py.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


class SurfaceCodeStabilizerGenerator:
    """
    Generate stabilizers for one unrotated planar surface-code patch.

    The generator precomputes the full data-qubit coordinate map for the open
    patch and emits the stabilizers in deterministic row order. X checks are
    returned first, followed by Z checks, matching the ordering used by the MDR
    table builder and the surface-code logical generator.

    Attributes
    ----------
    d : int Odd code distance for the planar patch. n : int Number of data
    qubits in the patch, equal to `d^2 + (d - 1)^2`. data_coordinates :
    List[Tuple[int, int]] Row-major list of all even-parity lattice coordinates
    that host data qubits. _coord_to_index_map : Dict[Tuple[int, int], int]
    Mapping from lattice coordinates onto flattened physical-qubit indices.

    Methods
    -------
    __init__(distance) Validate the distance and initialize the unrotated patch
    geometry. generate_stabilizers() Return the full ordered stabilizer list
    for the patch. coord_to_index(x, y) Convert a data-qubit lattice coordinate
    into a physical qubit index. _iter_check_coordinates() Enumerate all
    stabilizer-center coordinates in row order. _neighbor_indices(x, y) Return
    the adjacent data-qubit indices for one stabilizer center.
    """

    def __init__(self, distance: int) -> None:
        """
        Initialize surface-code geometry for a chosen odd distance.

        Args:
        distance: Odd code distance `d` with `d >= 3`.

        Raises:
        ValueError: If `distance` is even or smaller than 3.
        """
        if distance < 3 or distance % 2 == 0:
            raise ValueError("Distance d must be odd and >= 3")

        self.d = distance
        self.n = distance * distance + (distance - 1) * (distance - 1)
        self.data_coordinates: List[Tuple[int, int]] = [
            (x, y)
            for y in range(2 * self.d - 1)
            for x in range(2 * self.d - 1)
            if (x + y) % 2 == 0
        ]
        self._coord_to_index_map: Dict[Tuple[int, int], int] = {
            coord: idx for idx, coord in enumerate(self.data_coordinates)
        }

    def generate_stabilizers(self) -> List[str]:
        """
        Return the full ordered stabilizer list for the surface-code patch.

        Returns:
        List[str]: Sparse Pauli strings for all X-type checks followed by all
        Z-type checks.
        """
        stabilizers: List[str] = []

        for check_x, check_y in self._iter_check_coordinates():
            if check_x % 2 != 1:
                continue
            qubits = self._neighbor_indices(check_x, check_y)
            stabilizers.append(" ".join(f"X{q}" for q in qubits))

        for check_x, check_y in self._iter_check_coordinates():
            if check_x % 2 != 0:
                continue
            qubits = self._neighbor_indices(check_x, check_y)
            stabilizers.append(" ".join(f"Z{q}" for q in qubits))

        return stabilizers

    def coord_to_index(self, x: int, y: int) -> int:
        """
        Map a data-qubit lattice coordinate onto the corresponding index.

        Args:
        x: Lattice x coordinate of a data-qubit site. y: Lattice y coordinate
        of a data-qubit site.

        Returns:
        int: Flattened physical-qubit index.
        """
        return self._coord_to_index_map[(x, y)]

    def _iter_check_coordinates(self) -> List[Tuple[int, int]]:
        """
        Enumerate all odd-parity stabilizer-center coordinates in row order.

        Returns:
        List[Tuple[int, int]]: Stabilizer-center coordinates.
        """
        return [
            (x, y)
            for y in range(2 * self.d - 1)
            for x in range(2 * self.d - 1)
            if (x + y) % 2 == 1
        ]

    def _neighbor_indices(self, x: int, y: int) -> List[int]:
        """
        Return the adjacent data-qubit indices for one stabilizer center.

        Args:
        x: Stabilizer-center x coordinate. y: Stabilizer-center y coordinate.

        Returns:
        List[int]: Sorted data-qubit indices adjacent to that center.
        """
        qubits: List[int] = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            coord = (x + dx, y + dy)
            if coord in self._coord_to_index_map:
                qubits.append(self._coord_to_index_map[coord])
        qubits.sort()
        return qubits
