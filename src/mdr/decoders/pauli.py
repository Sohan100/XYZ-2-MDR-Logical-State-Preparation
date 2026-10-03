"""
Sparse Pauli-string helpers used by Pauli-frame decoders.
"""

from __future__ import annotations

import numpy as np


def pauli_string_to_masks(spec: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a sparse Pauli string into binary X/Z support masks.
    """
    x = np.zeros(n, dtype=np.uint8)
    z = np.zeros(n, dtype=np.uint8)
    for token in spec.split():
        if not token or token == "I":
            continue
        pauli = token[0].upper()
        qubit = int(token[1:])
        if pauli not in {"X", "Y", "Z"}:
            raise ValueError(f"Invalid Pauli letter: {pauli}")
        if pauli in {"X", "Y"}:
            x[qubit] ^= 1
        if pauli in {"Z", "Y"}:
            z[qubit] ^= 1
    return x, z


def symp(
    x1: np.ndarray,
    z1: np.ndarray,
    x2: np.ndarray,
    z2: np.ndarray,
) -> int:
    """
    Return the binary symplectic inner product of two Pauli masks.
    """
    return (int(x1 @ z2) + int(z1 @ x2)) & 1


def frame_anticommutation(
    frame_x: np.ndarray,
    frame_z: np.ndarray,
    op_x: np.ndarray,
    op_z: np.ndarray,
) -> np.ndarray:
    """
    Compute per-shot anticommutation parity between frames and an operator.
    """
    return np.mod((frame_x @ op_z) + (frame_z @ op_x), 2).astype(np.uint8)
