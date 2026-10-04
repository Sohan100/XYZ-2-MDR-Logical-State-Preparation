"""
tn_decoder.py
----------------------------------------------------------------------------
Maximum-likelihood decoding of a detector error model by a tensor-network sweep.

The probability of logical class l given the detection events sigma is

    Z_l = sum_{x : H x = sigma, L x = l}  prod_j p_j^{x_j} (1 - p_j)^{1 - x_j},

a sum over every fault configuration that explains the syndrome. The decoder returns the
class with the larger Z_l, which is the optimal (maximum-likelihood) decision, and the
free-energy difference ln Z_1 - ln Z_0.

The sum is contracted by a sweep. Fault mechanisms are processed in the order of their
position along one axis of the lattice. The partial sum is a function of the parities of
the open detectors (touched by processed mechanisms, not yet by all of their mechanisms)
and of the logical parity, stored as a matrix product state (MPS) whose sites are the
open detectors ordered across the sweep (by position, then time) plus one logical site.
A mechanism j acts as (1 - p_j) I + p_j prod_{i in j} X_i, a sum of two MPS that doubles
the bond dimension on its span; a QR sweep and an SVD sweep bring it back to at most chi.
A detector is opened as a site in the state |0> before its first mechanism and closed
after its last one by projecting onto the observed parity.

With chi large enough the contraction is exact. The bond dimension it needs grows
exponentially with the thickness of the cross-section of the sweep, which is about
(r + 1) rows of detectors. A single round (r = 1) needs a small chi up to d = 21, while
r = d is exact only for small d. `discarded` reports the truncated weight of each shot,
and decoding with 2 chi tells whether chi is large enough.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg


# --------------------------------------------------------------------------- geometry
def detector_positions(ft, circuit) -> np.ndarray:
    """(x, y, t) of every detector of an FTMDRCircuit with detectors="combined".

    Detector coordinates are (k, round, phase): k indexes the rows of S_0 for the frame
    detectors and the checks for the gauge detectors (see TwoLevelDecoder). The position
    is the mean block coordinate (column j, row i) of the data qubits in the support.
    """
    from .two_level_decoder import GAUGE_PHASES

    geo = ft.geometry
    qpos: Dict[int, Tuple[float, float]] = {}
    for i in range(geo.d):
        for j in range(geo.d):
            up, lo = geo.verts(i, j)
            qpos[up] = (float(j), float(i) - 0.2)
            qpos[lo] = (float(j), float(i) + 0.2)
    check_q = [np.array([q for _, q in ch["terms"]]) for ch in geo.checks]

    def support_xy(qs):
        if len(qs) == 0:
            return (0.0, 0.0)
        a = np.array([qpos[int(q)] for q in qs])
        return tuple(a.mean(axis=0))

    def rows_xy(rows):
        out = []
        for row in rows:
            qs = set()
            for c in np.flatnonzero(row):
                qs ^= set(int(q) for q in check_q[int(c)])
            out.append(support_xy(sorted(qs)))
        return out

    s0_xy = rows_xy(ft.frame.s0_rows)
    init_rows = (ft.frame.s0_rows if getattr(ft, "init", "frame") == "frame"
                 else ft.frame.deterministic_rows(ft.init_basis))
    init_xy = rows_xy(init_rows)
    check_xy = [support_xy(q) for q in check_q]
    coords = circuit.get_detector_coordinates()
    out = np.zeros((circuit.num_detectors, 3))
    n_init = 0
    for i in range(circuit.num_detectors):
        k, rnd, ph = int(coords[i][0]), float(coords[i][1]), int(coords[i][2])
        if rnd == 0 and ph == 0:
            # the detectors of the first round all have coordinates (0, 0, 0), one per row
            # of the initial deterministic group, in order
            x, y = init_xy[n_init]
            n_init += 1
        elif ph in GAUGE_PHASES:
            x, y = check_xy[k]
        else:
            x, y = s0_xy[k]
        out[i] = (x, y, rnd + 0.1 * ph)
    return out


# --------------------------------------------------------------------------- schedule
class SweepSchedule:
    """Order of mechanisms and the open/close events of the detectors."""

    def __init__(self, mech_dets: Sequence[Sequence[int]], mech_obs: Sequence[bool],
                 pos: np.ndarray, axis: Optional[int] = None) -> None:
        nd = pos.shape[0]
        nm = len(mech_dets)
        self.nd, self.nm = nd, nm
        # sweep along the axis on which the logical mechanisms spread the most, so that the
        # logical site stays local in the transverse order
        lx = [pos[list(ds), :2].mean(axis=0) for ds, o in zip(mech_dets, mech_obs) if o and len(ds)]
        if axis is None:
            if lx:
                spread = np.ptp(np.array(lx), axis=0)
                axis = int(np.argmax(spread))
            else:
                axis = 0
        self.axis = axis
        trans = 1 - axis
        # transverse key of a detector: position across the sweep, then along it, then time,
        # so that neighbouring detectors are neighbouring sites of the MPS
        span_t = float(np.ptp(pos[:, 2])) + 1.0
        span_a = float(np.ptp(pos[:, axis])) + 1.0
        dkey = (pos[:, trans] * span_a + (pos[:, axis] - pos[:, axis].min())) * span_t + (pos[:, 2] - pos[:, 2].min())
        self.logical_key = ((float(np.mean([p[trans] for p in lx])) + 0.25) * span_a * span_t) if lx else -1.0
        # mechanisms column by column (by their smallest coordinate along the sweep), snaking
        # across the column so that the orthogonality centre of the MPS moves little
        mx = np.array([np.floor(pos[list(ds), axis].min() + 1e-9) if len(ds) else 0.0 for ds in mech_dets])
        mt = np.array([dkey[list(ds)].mean() if len(ds) else 0.0 for ds in mech_dets])
        cols = np.unique(mx)
        rank_of = {c: i for i, c in enumerate(cols)}
        snake = np.array([mt[j] if rank_of[mx[j]] % 2 == 0 else -mt[j] for j in range(nm)])
        order = np.lexsort((snake, mx))
        self.order = order
        first = np.full(nd, -1, dtype=np.int64)
        last = np.full(nd, -1, dtype=np.int64)
        for step, j in enumerate(order):
            for i in mech_dets[j]:
                if first[i] < 0:
                    first[i] = step
                last[i] = step
        self.opens: List[List[int]] = [[] for _ in range(nm)]
        self.closes: List[List[int]] = [[] for _ in range(nm)]
        for i in range(nd):
            if first[i] >= 0:
                self.opens[first[i]].append(i)
                self.closes[last[i]].append(i)
        for lst in self.opens:
            lst.sort(key=lambda i: dkey[i])
        self.unused = np.flatnonzero(first < 0)
        self.dkey = dkey
        # largest number of open detectors (the length of the MPS)
        width = 0
        cur = 0
        for step in range(nm):
            cur += len(self.opens[step])
            width = max(width, cur)
            cur -= len(self.closes[step])
        self.max_open = width


# --------------------------------------------------------------------------- MPS sweep
class _Mps:
    """MPS with a moving orthogonality centre, sites inserted and removed on the fly."""

    def __init__(self, chi: int, cutoff: float) -> None:
        self.A: List[np.ndarray] = []
        self.keys: List[float] = []
        self.ids: List[int] = []
        self.c = 0
        self.log = 0.0
        self.chi = chi
        self.cutoff = cutoff
        self.discarded = 0.0

    def pos_of(self, did: int) -> int:
        return self.ids.index(did)

    def insert(self, key: float, did: int) -> None:
        import bisect
        p = bisect.bisect(self.keys, key)
        if not self.A:
            D = 1
        elif p == 0:
            D = self.A[0].shape[0]
        elif p == len(self.A):
            D = self.A[-1].shape[2]
        else:
            D = self.A[p].shape[0]
        T = np.zeros((D, 2, D))
        T[:, 0, :] = np.eye(D)
        self.A.insert(p, T)
        self.keys.insert(p, key)
        self.ids.insert(p, did)
        if p <= self.c and len(self.A) > 1:
            self.c += 1

    def move(self, to: int) -> None:
        A = self.A
        while self.c < to:
            t = A[self.c]
            Dl, _, Dr = t.shape
            q, r = np.linalg.qr(t.reshape(Dl * 2, Dr))
            A[self.c] = q.reshape(Dl, 2, q.shape[1])
            A[self.c + 1] = np.tensordot(r, A[self.c + 1], axes=(1, 0))
            self.c += 1
        while self.c > to:
            t = A[self.c]
            Dl, _, Dr = t.shape
            q, r = np.linalg.qr(t.reshape(Dl, 2 * Dr).T)
            A[self.c] = q.T.reshape(q.shape[1], 2, Dr)
            A[self.c - 1] = np.tensordot(A[self.c - 1], r.T, axes=(2, 0))
            self.c -= 1

    def normalize(self) -> None:
        t = self.A[self.c]
        nrm = float(np.linalg.norm(t))
        if nrm == 0.0:
            raise FloatingPointError("the syndrome has zero probability under the model")
        self.A[self.c] = t / nrm
        self.log += math.log(nrm)

    def apply(self, sites: Sequence[int], alpha: float, beta: float) -> None:
        """psi <- alpha psi + beta X_sites psi (sites sorted)."""
        A = self.A
        a, b = sites[0], sites[-1]
        self.move(a)
        if a == b:
            A[a] = alpha * A[a] + beta * A[a][:, ::-1, :]
            self.normalize()
            return
        flip = set(sites)
        for k in range(a, b + 1):
            t = A[k]
            u = t[:, ::-1, :] if k in flip else t
            Dl, _, Dr = t.shape
            if k == a:
                B = np.concatenate([alpha * t, beta * u], axis=2)
            elif k == b:
                B = np.concatenate([t, u], axis=0)
            else:
                B = np.zeros((2 * Dl, 2, 2 * Dr))
                B[:Dl, :, :Dr] = t
                B[Dl:, :, Dr:] = u
            A[k] = B
        # QR sweep to the right end of the span, then SVD sweep back with truncation
        for k in range(a, b):
            t = A[k]
            Dl, _, Dr = t.shape
            q, r = np.linalg.qr(t.reshape(Dl * 2, Dr))
            A[k] = q.reshape(Dl, 2, q.shape[1])
            A[k + 1] = np.tensordot(r, A[k + 1], axes=(1, 0))
        for k in range(b, a, -1):
            t = A[k]
            Dl, _, Dr = t.shape
            # LAPACK's gesvd: numpy's default gesdd fails to converge on some of these matrices,
            # whose entries span ~24 orders of magnitude (pure Z noise, wide networks)
            u, s, vh = scipy.linalg.svd(t.reshape(Dl, 2 * Dr), full_matrices=False,
                                        lapack_driver="gesvd", check_finite=False)
            tot = float(np.dot(s, s))
            keep = int(np.sum(s > self.cutoff * s[0])) if s.size else 0
            keep = max(1, min(self.chi, keep))
            if keep < s.size and tot > 0:
                self.discarded += float(np.dot(s[keep:], s[keep:])) / tot
            A[k] = vh[:keep].reshape(keep, 2, Dr)
            A[k - 1] = np.tensordot(A[k - 1], u[:, :keep] * s[:keep], axes=(2, 0))
        self.c = a
        self.normalize()

    def close(self, p: int, bit: int) -> None:
        A = self.A
        self.move(p)
        M = A[p][:, bit, :]
        if p > 0:
            A[p - 1] = np.tensordot(A[p - 1], M, axes=(2, 0))
            new_c = p - 1
        else:
            A[p + 1] = np.tensordot(M, A[p + 1], axes=(1, 0))
            new_c = p
        del A[p]
        del self.keys[p]
        del self.ids[p]
        self.c = new_c
        self.normalize()


class TensorNetworkDecoder:
    """
    Maximum-likelihood decoder for a detector error model with one observable.

    Parameters
    ----------
    mech_dets : detectors of every fault mechanism.
    mech_obs : whether every mechanism flips the logical observable.
    priors : probability of every mechanism.
    pos : (x, y, t) of every detector (`detector_positions`).
    chi : largest bond dimension of the sweep MPS.
    cutoff : relative singular-value cutoff.
    """

    LOGICAL = -1

    def __init__(self, mech_dets, mech_obs, priors, pos, chi: int = 32, cutoff: float = 1e-14,
                 axis: Optional[int] = None) -> None:
        self.mech_dets = [np.asarray(ds, dtype=np.int64) for ds in mech_dets]
        self.mech_obs = np.asarray(mech_obs, dtype=bool)
        self.p = np.asarray(priors, dtype=float)
        self.pos = np.asarray(pos, dtype=float)
        self.chi = int(chi)
        self.cutoff = float(cutoff)
        self.sched = SweepSchedule(self.mech_dets, self.mech_obs, self.pos, axis)
        self.last_discarded = 0.0

    def log_z(self, det_row: np.ndarray, chi: Optional[int] = None) -> Tuple[float, float, float]:
        """(ln Z_0, ln Z_1, discarded weight) for one shot."""
        det_row = np.asarray(det_row, dtype=np.uint8)
        if np.any(det_row[self.sched.unused]):
            raise ValueError("a detector that no mechanism touches has fired")
        sch = self.sched
        mps = _Mps(chi or self.chi, self.cutoff)
        mps.insert(sch.logical_key, self.LOGICAL)
        for step, j in enumerate(sch.order):
            for i in sch.opens[step]:
                mps.insert(sch.dkey[i], int(i))
            ids = mps.ids
            sites = sorted(ids.index(int(i)) for i in self.mech_dets[j])
            if self.mech_obs[j]:
                sites = sorted(sites + [ids.index(self.LOGICAL)])
            pj = self.p[j]
            mps.apply(sites, 1.0 - pj, pj)
            for i in sch.closes[step]:
                mps.close(mps.ids.index(int(i)), int(det_row[i]))
        assert len(mps.A) == 1 and mps.ids[0] == self.LOGICAL
        v = mps.A[0].reshape(2)
        with np.errstate(divide="ignore"):
            l0 = mps.log + math.log(abs(v[0])) if v[0] != 0 else -math.inf
            l1 = mps.log + math.log(abs(v[1])) if v[1] != 0 else -math.inf
        self.last_discarded = mps.discarded
        return l0, l1, mps.discarded

    def decode_shot(self, det_row) -> Tuple[int, float, float]:
        """(predicted logical flip, ln Z_1 - ln Z_0, discarded weight)."""
        l0, l1, disc = self.log_z(det_row)
        return int(l1 > l0), l1 - l0, disc

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        dets = np.asarray(dets, dtype=np.uint8)
        out = np.zeros((dets.shape[0], 1), dtype=bool)
        for s in range(dets.shape[0]):
            out[s, 0] = bool(self.decode_shot(dets[s])[0])
        return out
