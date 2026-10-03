"""
fast_osd.py
----------------------------------------------------------------------------
BP followed by the combination-sweep OSD of ldpc, with the candidates scored all at once.

ldpc's BpOsdDecoder with ``osd_method="osd_cs"`` tries every weight-one string on the
k information bits and every weight-two string on the first ``order`` of them. It stores
the k candidate strings (k^2 bytes) and solves the triangular systems once per candidate,
so a circuit with S = (r + 1) d^2 has memory and time that grow as the square of the
number of fault mechanisms (about 80 GB and minutes per decode at d = 21 with r = 21).

This class returns the same decoding with less work. With A = P L U (ldpc's
PluDecomposition of the check matrix with columns sorted by the BP soft output), flipping
an information column c changes the pivot bits by R_c = U_piv^{-1} U[:, c]. One
back-substitution with all k right-hand sides packed into 64-bit words gives R, and the
weight of every candidate follows from R and the OSD-0 solution. Memory is m k / 8 bytes
for R (m detectors) and the time is dominated by the back-substitution.

The candidate weight and the tie rule are ldpc's: sum of log(1/p_j) over the support,
candidates in the order weight-one then weight-two, and a candidate replaces the best one
only if its weight is strictly smaller. BP that converges returns its own decoding, as in
BpOsdDecoder. Columns with equal soft output may be ordered differently from ldpc (which
uses qsort), so rare shots can differ, with the same statistics.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

try:
    import numba

    _njit = numba.njit(cache=True, nogil=True)
except ImportError:  # pragma: no cover - numba is in the decoders extra
    def _njit(f):
        return f


@_njit
def _back_substitute(indptr, indices, piv_row_of_col, init_words, m):
    """Rows of R = U_piv^{-1} U_np, from the bottom row up (bit-packed rows, in place)."""
    nw = init_words.shape[1]
    for i in range(m - 1, -1, -1):
        for e in range(indptr[i], indptr[i + 1]):
            j = piv_row_of_col[indices[e]]
            if j > i:
                for w in range(nw):
                    init_words[i, w] ^= init_words[j, w]
    return init_words


@_njit
def _pack_rows(rows, cpos, m, nw):
    """Bit-packed matrix with ones at (rows[e], cpos[e]) (XOR for repeated entries)."""
    R = np.zeros((m, nw), dtype=np.uint64)
    for e in range(rows.shape[0]):
        R[rows[e], cpos[e] >> 6] ^= np.uint64(1) << np.uint64(cpos[e] & 63)
    return R


@_njit
def _weights_from_rows(R, v, k):
    """W[c] = sum_s R[s, c] v[s] for the bit-packed rows of R."""
    out = np.zeros(k)
    m, nw = R.shape
    for s in range(m):
        vs = v[s]
        if vs == 0.0:
            continue
        for w in range(nw):
            word = R[s, w]
            while word != np.uint64(0):
                low = word & (~word + np.uint64(1))
                b = np.int64(np.log2(np.float64(low)))
                c = w * 64 + b
                if c < k:
                    out[c] += vs
                word ^= low
    return out


class FastBpOsd:
    """
    BP + OSD-CS with ldpc's candidates and weights.

    Parameters
    ----------
    H : scipy sparse matrix (uint8), checks by fault mechanisms.
    channel_probs : prior probability of every mechanism.
    bp_iters : iterations of product-sum BP.
    order : the weight-two strings use the first `order` information bits.
    """

    def __init__(self, H, channel_probs, bp_iters: int = 30, order: int = 10) -> None:
        from ldpc import BpDecoder

        self.H = sp.csc_matrix(H, dtype=np.uint8)
        self.m, self.n = self.H.shape
        self.order = int(order)
        self.bp = BpDecoder(self.H, error_channel=list(channel_probs), max_iter=bp_iters,
                            bp_method="product_sum", input_vector_type="syndrome")
        self.p = np.asarray(channel_probs, dtype=float)

    def update_channel_probs(self, probs) -> None:
        self.p = np.asarray(probs, dtype=float)
        self.bp.update_channel_probs(list(self.p))

    def decode(self, syndrome) -> np.ndarray:
        from ldpc.mod2 import PluDecomposition

        syndrome = np.asarray(syndrome, dtype=np.uint8)
        x_bp = np.asarray(self.bp.decode(syndrome), dtype=np.uint8)
        if self.bp.converge:
            return x_bp
        llr = np.asarray(self.bp.log_prob_ratios, dtype=float)
        order = np.argsort(llr, kind="stable")          # most likely faults first
        Hp = self.H[:, order]
        plu = PluDecomposition(Hp)
        x0 = np.asarray(plu.lu_solve(syndrome), dtype=np.uint8)
        piv = np.asarray(plu.pivots, dtype=np.int64)
        rank = len(piv)
        w = np.log(1.0 / np.clip(self.p[order], 1e-300, 1.0))
        W0 = float(w[x0 == 1].sum())
        best = x0
        if self.order > 0 and rank < self.n:
            U = sp.csr_matrix(plu.U)
            nonpiv = np.setdiff1d(np.arange(self.n), piv)        # in column order, as in ldpc
            k = len(nonpiv)
            piv_row_of_col = np.full(self.n, -1, dtype=np.int64)
            piv_row_of_col[piv] = np.arange(rank)
            # bit-packed U[:, nonpiv] rows
            col_pos = np.full(self.n, -1, dtype=np.int64)
            col_pos[nonpiv] = np.arange(k)
            nw = (k + 63) // 64
            Uc = U.tocoo()
            keep = (col_pos[Uc.col] >= 0) & (Uc.row < rank)
            R = _pack_rows(Uc.row[keep].astype(np.int64), col_pos[Uc.col[keep]], rank, nw)
            R = _back_substitute(U.indptr.astype(np.int64), U.indices.astype(np.int64), piv_row_of_col, R, rank)
            # weight change of every weight-one candidate
            wp = w[piv]
            v = wp * (1.0 - 2.0 * x0[piv])
            W1 = W0 + w[nonpiv] * (1.0 - 2.0 * x0[nonpiv]) + _weights_from_rows(R, v, k)
            cand = [(W1, None)]
            # weight-two strings on the first `order` information bits
            lam = min(self.order, k)
            cols = np.array([[(R[:, c // 64] >> np.uint64(c % 64)) & np.uint64(1)] for c in range(lam)],
                            dtype=np.uint8).reshape(lam, rank) if lam else np.zeros((0, rank), np.uint8)
            pairs = [(a, b) for a in range(lam) for b in range(a + 1, lam)]
            W2 = np.array([W0 + w[nonpiv[a]] * (1 - 2.0 * x0[nonpiv[a]]) + w[nonpiv[b]] * (1 - 2.0 * x0[nonpiv[b]])
                           + float(np.dot(cols[a] ^ cols[b], v)) for a, b in pairs])
            # ldpc: weight-one candidates first, then weight-two; strict improvement only
            allw = np.concatenate([W1, W2])
            j = int(np.argmin(allw)) if allw.size else -1
            if j >= 0 and allw[j] < W0:
                x = x0.copy()
                if j < k:
                    rbits = ((R[:, j // 64] >> np.uint64(j % 64)) & np.uint64(1)).astype(np.uint8)
                    x[piv] ^= rbits
                    x[nonpiv[j]] ^= 1
                else:
                    a, b = pairs[j - k]
                    x[piv] ^= cols[a] ^ cols[b]
                    x[nonpiv[a]] ^= 1
                    x[nonpiv[b]] ^= 1
                best = x
        out = np.zeros(self.n, dtype=np.uint8)
        out[order] = best
        return out
