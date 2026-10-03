"""

xyz2_geometry.py
----------------------------------------------------------------------------
Lattice metadata for the XYZ^2 checks used by the fault-tolerant MDR path.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from xyz2.logical_generator import XYZ2LogicalGenerator
from xyz2.stabilizer_generator import XYZ2StabilizerGenerator


class XYZ2Geometry:
    """
    Attach block and plaquette coordinates to every XYZ^2 stabilizer.

    The stabilizer strings and their order come from
    `XYZ2StabilizerGenerator.generate_stabilizers()`, so indices agree with
    the rest of the repository. Each data qubit belongs to one doubled block
    `(i, j)` with an upper and a lower vertex joined by an XX link. Every
    non-link check is the (possibly truncated) hexagon of a virtual cell
    `(i, j)` whose six positions are

    - `TL`  = lower vertex of block `(i, j)`        (Pauli X)
    - `TRu` = upper vertex of block `(i, j + 1)`    (Pauli Y)
    - `TRl` = lower vertex of block `(i, j + 1)`    (Pauli Z)
    - `BR`  = upper vertex of block `(i + 1, j + 1)` (Pauli X)
    - `BLl` = lower vertex of block `(i + 1, j)`    (Pauli Y)
    - `BLu` = upper vertex of block `(i + 1, j)`    (Pauli Z)

    Boundary checks are cells with one coordinate outside `[0, d - 1]`.
    Cells with `i + j` even are called class B, cells with `i + j` odd are
    class A.

    Attributes
    ----------
    d : int Code distance. n : int Number of data qubits, `2 d^2`.
    stabilizers : List[str] Ordered stabilizer strings from the repository
    generator. logicals : Dict[str, str] Logical X, Y, Z strings. checks :
    List[Dict] One record per stabilizer with keys `kind` (`link` or `hex`),
    `terms` (list of `(pauli, qubit)`), `block` (links) or `cell` and
    `positions` (hexagons and boundary checks).
    """

    POSITIONS = ("TL", "TRu", "TRl", "BR", "BLl", "BLu")

    def __init__(self, distance: int) -> None:
        self._gen = XYZ2StabilizerGenerator(distance)
        self.d = distance
        self.n = self._gen.n
        self.stabilizers: List[str] = self._gen.generate_stabilizers()
        self.logicals: Dict[str, str] = XYZ2LogicalGenerator(
            distance
        ).generate_logicals()
        self.checks: List[Dict] = [self._describe(s) for s in self.stabilizers]

    def verts(self, i: int, j: int) -> Tuple[int, int]:
        """
        Return `(upper, lower)` qubit indices of block `(i, j)`.
        """
        return self._gen._coord_to_verts(i, j)

    def block_parity(self, qubit: int) -> int:
        """
        Return `(i + j) % 2` for the block that contains `qubit`.
        """
        return self._block_of[qubit][0]

    def cell_positions(self, i: int, j: int) -> Dict[str, int]:
        """
        Return the in-patch positions of virtual cell `(i, j)`.
        """
        d = self.d

        def inside(a: int, b: int) -> bool:
            return 0 <= a < d and 0 <= b < d

        out: Dict[str, int] = {}
        if inside(i, j):
            out["TL"] = self.verts(i, j)[1]
        if inside(i, j + 1):
            out["TRu"], out["TRl"] = self.verts(i, j + 1)
        if inside(i + 1, j + 1):
            out["BR"] = self.verts(i + 1, j + 1)[0]
        if inside(i + 1, j):
            out["BLu"], out["BLl"] = self.verts(i + 1, j)
        return out

    def stabilizer_matrix(self) -> np.ndarray:
        """
        Return the binary symplectic matrix `[x | z]` of the stabilizers.

        The matrix is computed once and returned read-only.
        """
        cached = getattr(self, "_stab_cache", None)
        if cached is not None:
            return cached
        rows = []
        for ch in self.checks:
            x = np.zeros(self.n, dtype=np.uint8)
            z = np.zeros(self.n, dtype=np.uint8)
            for pauli, q in ch["terms"]:
                if pauli in "XY":
                    x[q] ^= 1
                if pauli in "ZY":
                    z[q] ^= 1
            rows.append(np.concatenate([x, z]))
        mat = np.array(rows, dtype=np.uint8)
        mat.setflags(write=False)
        self._stab_cache = mat
        return mat

    @property
    def _block_of(self) -> Dict[int, Tuple[int, Tuple[int, int]]]:
        cache = getattr(self, "_block_cache", None)
        if cache is None:
            cache = {}
            for i in range(self.d):
                for j in range(self.d):
                    up, lo = self.verts(i, j)
                    cache[up] = ((i + j) % 2, (i, j))
                    cache[lo] = ((i + j) % 2, (i, j))
            self._block_cache = cache
        return cache

    def _describe(self, spec: str) -> Dict:
        terms = [(tok[0], int(tok[1:])) for tok in spec.split()]
        if len(terms) == 2 and all(p == "X" for p, _ in terms):
            _, (i, j) = self._block_of[terms[0][1]]
            return {"kind": "link", "terms": terms, "spec": spec,
                    "block": (i, j)}
        support = {q for _, q in terms}
        for cell in self._candidate_cells(terms):
            pos = self.cell_positions(*cell)
            if set(pos.values()) == support:
                qpos = {q: p for p, q in pos.items()}
                return {"kind": "hex", "terms": terms, "spec": spec,
                        "cell": cell,
                        "positions": {q: qpos[q] for _, q in terms}}
        raise ValueError(f"Could not place check {spec} on a cell.")

    def _candidate_cells(self, terms) -> List[Tuple[int, int]]:
        cells = set()
        for _, q in terms:
            _, (i, j) = self._block_of[q]
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    cells.add((i + di, j + dj))
        return sorted(cells)
