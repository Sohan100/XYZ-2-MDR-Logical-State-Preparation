"""

frame_basis.py
----------------------------------------------------------------------------
Product frame used to initialise and read out the XYZ^2 logical plus state.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .xyz2_geometry import XYZ2Geometry

_PAULI_BITS = {"X": (1, 0), "Y": (1, 1), "Z": (0, 1)}


def _bits(row: np.ndarray) -> int:
    """Binary row as a Python integer, bit c set when row[c] is 1."""
    v = 0
    for c in np.flatnonzero(row):
        v |= 1 << int(c)
    return v


def _insert(pivots: Dict[int, int], v: int) -> bool:
    """Reduce v against the basis {leading bit: row}; add it and return True if it is independent."""
    while v:
        lead = v.bit_length() - 1
        row = pivots.get(lead)
        if row is None:
            pivots[lead] = v
            return True
        v ^= row
    return False


class XYZ2FrameBasis:
    """
    Single-qubit product frame that makes the XYZ^2 plus state preparable.

    Blocks with `i + j` even are prepared and read out in the X basis on both
    vertices. Blocks with `i + j` odd use Z on the upper vertex and Y on the
    lower vertex (`odd_rep="ZY"`). In this frame Logical X and a subgroup
    `S_0` of the stabilizer group with `d^2` independent generators are
    products of frame Paulis. They are therefore deterministic right after
    initialisation and can be reconstructed from a destructive frame readout.
    The all-X frame (`|+>^n`) only fixes the XX links, which cannot tell the
    two vertices of a block apart, so it has fault distance 1 or 2.

    Attributes
    ----------
    geometry : XYZ2Geometry Lattice metadata. basis : Dict[int, str] Frame
    Pauli for every data qubit. s0_rows : np.ndarray Binary matrix of shape
    `(d^2, m)`. Row `k` lists the generators whose product is the `k`-th
    frame-deterministic stabilizer.
    """

    def __init__(self, geometry: XYZ2Geometry, odd_rep: str = "ZY") -> None:
        if odd_rep not in {"ZY", "YZ"}:
            raise ValueError("odd_rep must be 'ZY' or 'YZ'.")
        self.geometry = geometry
        self.odd_rep = odd_rep
        self.basis: Dict[int, str] = {}
        for i in range(geometry.d):
            for j in range(geometry.d):
                up, lo = geometry.verts(i, j)
                if (i + j) % 2 == 0:
                    self.basis[up], self.basis[lo] = "X", "X"
                else:
                    self.basis[up], self.basis[lo] = odd_rep[0], odd_rep[1]
        self.s0_rows = self.deterministic_rows(self.basis)

    def deterministic_rows(self, basis: Dict[int, str]) -> np.ndarray:
        """
        Return generator combinations whose product matches `basis`.

        A product stabilizer is deterministic on the product state defined by
        `basis` exactly when it commutes with every single-qubit frame Pauli,
        which is a linear condition over GF(2).
        """
        geo = self.geometry
        n = geo.n
        h = geo.stabilizer_matrix()
        a = np.zeros((n, h.shape[0]), dtype=np.uint8)
        for q in range(n):
            bx, bz = _PAULI_BITS[basis[q]]
            a[q] = (h[:, q] * bz + h[:, n + q] * bx) % 2
        return self._nullspace(a)

    def gauge_generators(self) -> List[int]:
        """
        Return generators that complete `s0_rows` to a basis of all checks.

        These are the checks whose first outcome is random on the frame
        state (odd-block links, class-A hexagons, left and right boundary
        checks). Links are tried first so that every odd link becomes its own
        gauge detector, which the lower-level (link) decoder reads directly.
        """
        cached = getattr(self, "_gauge_cache", None)
        if cached is not None:
            return list(cached)
        geo = self.geometry
        m = len(geo.checks)
        order = [ci for ci, ch in enumerate(geo.checks) if ch["kind"] == "link"]
        order += [ci for ci, ch in enumerate(geo.checks) if ch["kind"] != "link"]
        # Incremental GF(2) elimination on bit sets: a candidate raises the rank
        # exactly when it does not reduce to zero against the current basis.
        pivots: Dict[int, int] = {}
        rank = sum(_insert(pivots, _bits(r)) for r in self.s0_rows)
        chosen: List[int] = []
        for ci in order:
            if rank == m:
                break
            if _insert(pivots, 1 << ci):
                chosen.append(ci)
                rank += 1
        self._gauge_cache = tuple(chosen)
        return chosen

    @staticmethod
    def _rank(a: np.ndarray) -> int:
        pivots: Dict[int, int] = {}
        return sum(_insert(pivots, _bits(r)) for r in np.asarray(a) % 2)

    def product_spec(self, row: np.ndarray) -> str:
        """
        Return the sparse Pauli string of the product of selected generators.
        """
        n = self.geometry.n
        v = (row @ self.geometry.stabilizer_matrix()) % 2
        toks: List[str] = []
        for q in range(n):
            x, z = v[q], v[n + q]
            if x and z:
                toks.append(f"Y{q}")
            elif x:
                toks.append(f"X{q}")
            elif z:
                toks.append(f"Z{q}")
        return " ".join(toks)

    def logical_x_in_frame(self) -> bool:
        """
        Check that Logical X is a product of frame Paulis.
        """
        terms = self.geometry.logicals["Logical X"].split()
        return all(self.basis[int(t[1:])] == t[0] for t in terms)

    @staticmethod
    def _nullspace(a: np.ndarray) -> np.ndarray:
        a = a.copy() % 2
        rows, cols = a.shape
        pivots: List[int] = []
        r = 0
        for c in range(cols):
            piv = next((i for i in range(r, rows) if a[i, c]), None)
            if piv is None:
                continue
            a[[r, piv]] = a[[piv, r]]
            for i in range(rows):
                if i != r and a[i, c]:
                    a[i] ^= a[r]
            pivots.append(c)
            r += 1
            if r == rows:
                break
        free = [c for c in range(cols) if c not in pivots]
        basis = []
        for f in free:
            v = np.zeros(cols, dtype=np.uint8)
            v[f] = 1
            for i, pc in enumerate(pivots):
                if a[i, f]:
                    v[pc] = 1
            basis.append(v)
        return np.array(basis, dtype=np.uint8).reshape(len(basis), cols)
