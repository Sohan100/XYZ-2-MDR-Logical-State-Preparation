"""

sequential_decoder.py
----------------------------------------------------------------------------
The sequential decoder of Srivastava et al. (arXiv:2505.03691) for
circuit-level noise, on the frame-initialised XYZ^2 MDR circuits.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import itertools

import numpy as np
import scipy.sparse as sp
import stim

from .ft_mdr_circuit import REP_PHASE

# PyMatching and BP take probabilities in (0, 1/2); mechanisms that cannot occur in a shot get
# BP_FLOOR in BP (an exact zero gives infinite log-likelihoods) and are left out of the matching graph.
BP_FLOOR = 1e-14
STATIC, DATA, TIME, OTHER = 0, 1, 2, 3


def _or(ps: Sequence[float]) -> float:
    """Probability that an odd number of independent mechanisms fire (the DEM's merge rule)."""
    q = 0.0
    for p in ps:
        q = q * (1 - p) + p * (1 - q)
    return q


def _weights(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 0.5 - 1e-9)
    return np.log((1 - p) / p)


def pauli_signatures(circuit: stim.Circuit, inserts: List[Tuple[int, int, str]],
                     chunk: int = 2048) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Detectors and observables flipped by single Paulis inserted into the noiseless circuit.

    Each insert is (position, qubit, pauli): the Pauli acts just before the top-level instruction
    `position` (len(circuit) for the very end). Noise channels are skipped and measurements lose
    their flip probabilities, so the circuit is the noiseless copy. Returns, per insert, the flipped
    detector ids and observable ids.
    """
    nq, nd, no = circuit.num_qubits, circuit.num_detectors, circuit.num_observables
    ops = []
    for inst in circuit:
        if isinstance(inst, stim.CircuitRepeatBlock):
            raise ValueError("pauli_signatures needs a circuit without REPEAT blocks.")
        gd = stim.gate_data(inst.name)
        if gd.produces_measurements:
            if inst.gate_args_copy():
                inst = stim.CircuitInstruction(inst.name, inst.targets_copy())
        elif gd.is_noisy_gate:
            inst = None
        ops.append(inst)
    dets_out: List[np.ndarray] = [None] * len(inserts)
    obs_out: List[np.ndarray] = [None] * len(inserts)
    for a in range(0, len(inserts), chunk):
        part = inserts[a:a + chunk]
        by_pos: Dict[int, List[Tuple[int, int, str]]] = {}
        for i, (pos, q, pauli) in enumerate(part):
            by_pos.setdefault(pos, []).append((i, q, pauli))
        sim = stim.FlipSimulator(batch_size=len(part), num_qubits=nq,
                                 disable_stabilizer_randomization=True)

        def apply(pos):
            for pauli in "XYZ":
                sel = [(i, q) for i, q, pp in by_pos.get(pos, ()) if pp == pauli]
                if sel:
                    mask = np.zeros((nq, len(part)), dtype=bool)
                    for i, q in sel:
                        mask[q, i] = True
                    sim.broadcast_pauli_errors(pauli=pauli, mask=mask)

        for k, inst in enumerate(ops):
            apply(k)
            if inst is not None:
                sim.do(inst)
        apply(len(ops))
        df = sim.get_detector_flips()
        of = sim.get_observable_flips()
        for i in range(len(part)):
            dets_out[a + i] = np.flatnonzero(df[:, i])
            obs_out[a + i] = np.flatnonzero(of[:, i])
    return dets_out, obs_out


class SequentialDecoder:
    """
    Sequential decoder of Srivastava et al. on a `TwoLevelDecoder` setup.

    1. Every link is decoded alone along time. Its detectors (the link's S_0
       detectors for an even link, its gauge detectors for an odd link, and
       the comparisons of repeated links) form a chain. The faults whose link
       detectors are exactly two consecutive detectors of a chain form its
       time edges (probability q), those with exactly one link detector its
       data edges (probability p). ``link_rule="literal"`` pairs the events of
       a chain along time (minimum weight with time-edge weights, a boundary
       one step beyond each end of the chain) and removes a pair (t1, t2) as
       measurement errors when prod q > p(t1) p(t2); an event matched to a time
       end is a data error. ``link_rule="fast"`` matches all chains in one
       graph of time edges and a boundary (data) edge at every detector.
    2. Every data error (link, t) gets Z on the top qubit of the link at the
       start of round t; its signature and observable flip are XORed into the
       shot, so no link event is left.
    3. The upper level gets the conditional priors of Table I at circuit
       level: a corrected (link, t) brings in its data-edge faults with their
       residual signatures and probabilities p_j / sum p, a removed pair the
       faults of its time edges with p_j / sum p, faults without link
       detectors keep their priors, any other fault with link detectors
       enters (with its residual) only when all its link detectors are
       corrected. With ``table="full"`` an uncorrected link round also brings
       in every pair of its data-edge faults (no link event, combined
       signature, probability p_j p_k): the second-order entries of the s = 0
       column of Table I (ZZ, YY, ZY, YZ on the two qubits of the link).
       ``table="first"`` leaves them out. ``pair_top=K`` keeps only the K
       most probable pair signatures of every link round (under phen a link
       round has at most five, so K >= 5 changes nothing there).
    4. ``upper="match"`` (sri_m): matching on the non-link S_0 detectors.
       ``upper="bm"`` (sri_bm): BP on every non-link detector, posteriors
       pushed onto the S_0 edges, then matching; ``bp_iters <= 0`` runs d
       iterations, as their belief-matching does.

    Attributes of interest: `stats` (fault classes and their probability),
    `counters` (per-shot events: corrections without data-edge faults,
    matching retries).
    """

    def __init__(self, tl, upper: str = "match", link_rule: str = "literal",
                 bp_iters: int = 10, bp_method: str = "product_sum", table: str = "full",
                 pair_top: Optional[int] = None) -> None:
        if upper not in ("match", "bm"):
            raise ValueError("upper must be 'match' or 'bm'.")
        if link_rule not in ("literal", "fast"):
            raise ValueError("link_rule must be 'literal' or 'fast'.")
        if table not in ("full", "first"):
            raise ValueError("table must be 'full' or 'first'.")
        if bp_iters <= 0:
            # their belief-matching: "d rounds of belief propagation" (Srivastava et al., Fig. 9)
            bp_iters = tl.ft.geometry.d
        self.bp_iters = bp_iters
        self.tl = tl
        self.upper = upper
        self.link_rule = link_rule
        self.table = table
        self.pair_top = pair_top
        self.circuit = tl.circuit
        self.nd = self.circuit.num_detectors
        self.no = self.circuit.num_observables
        self.counters = {"shots": 0, "corrections": 0, "empty_data_edge": 0,
                         "removed_pairs": 0, "match_retry": 0}
        self._setup_chains()
        self._setup_faults()
        self._setup_corrections()
        self._setup_upper()
        self._setup_link_rule()
        if upper == "bm":
            self._setup_bp(bp_iters, bp_method)

    # ------------------------------------------------------------------ setup
    def _setup_chains(self) -> None:
        """
        The link detectors and their chains. `TwoLevelDecoder.link_cols` is not used: the circuit gives
        every round-0 detector the coordinates (0, 0, 0), so the S_0 row of a round-0 detector is its
        position among the round-0 detectors (they are declared in the order of `frame.s0_rows`).
        """
        ft = self.tl.ft
        geo = ft.geometry
        coords = self.circuit.get_detector_coordinates()
        s0_rows = ft.frame.s0_rows
        link_rows = {k for k, row in enumerate(s0_rows)
                     if row.sum() == 1 and geo.checks[int(np.flatnonzero(row)[0])]["kind"] == "link"}
        found, n0 = [], 0
        for x in range(self.nd):
            c = coords[x]
            k, rnd, ph = int(c[0]), int(c[1]), int(c[2])
            if ph in (0, 1, 2):
                if rnd == 0 and ph == 0:
                    k, n0 = n0, n0 + 1
                if k in link_rows:
                    found.append((x, int(np.flatnonzero(s0_rows[k])[0]), rnd, 0))
            elif geo.checks[k]["kind"] == "link":
                found.append((x, k, rnd, int(c[3]) if ph == REP_PHASE else 0))
        if n0 != len(s0_rows):
            raise RuntimeError("round-0 detectors do not match the S_0 rows of the frame")
        self.link_cols = np.array([x for x, _, _, _ in found], dtype=np.int64)
        nl = len(self.link_cols)
        chains: Dict[int, List[Tuple[int, int, int]]] = {}
        self.lk_round = np.zeros(nl, dtype=np.int64)
        self.lk_sub = np.zeros(nl, dtype=np.int64)
        for i, (x, ci, rnd, sub) in enumerate(found):
            chains.setdefault(ci, []).append((rnd, sub, i))
            self.lk_round[i], self.lk_sub[i] = rnd, sub
        self.chain_ci = sorted(chains)
        self.chains = [np.array([i for _, _, i in sorted(chains[ci])], dtype=np.int64)
                       for ci in self.chain_ci]
        self.lk_chain = np.zeros(nl, dtype=np.int64)
        self.lk_pos = np.zeros(nl, dtype=np.int64)
        for c, ch in enumerate(self.chains):
            self.lk_chain[ch] = c
            self.lk_pos[ch] = np.arange(len(ch))
        self.link_of = np.full(self.nd, -1, dtype=np.int64)
        self.link_of[self.link_cols] = np.arange(nl)
        self.is_link = self.link_of >= 0

    def _setup_faults(self) -> None:
        keys, pri = self.tl.mech_keys, self.tl.priors
        nm = len(keys)
        self.kind = np.zeros(nm, dtype=np.int8)
        data: Dict[int, List[int]] = {}
        time: Dict[int, List[int]] = {}           # keyed by the earlier link detector of the edge
        self.other_links: Dict[int, Tuple[int, ...]] = {}
        self.other_of: Dict[int, List[int]] = {}
        for j, (dets, _) in enumerate(keys):
            L = [int(self.link_of[x]) for x in dets if self.link_of[x] >= 0]
            if not L:
                continue
            if len(L) == 1:
                self.kind[j] = DATA
                data.setdefault(L[0], []).append(j)
                continue
            if len(L) == 2:
                a, b = sorted(L, key=lambda i: self.lk_pos[i])
                if self.lk_chain[a] == self.lk_chain[b] and self.lk_pos[b] == self.lk_pos[a] + 1:
                    self.kind[j] = TIME
                    time.setdefault(a, []).append(j)
                    continue
            self.kind[j] = OTHER
            self.other_links[j] = tuple(L)
            for i in L:
                self.other_of.setdefault(i, []).append(j)
        nl = len(self.link_cols)
        self.data_mechs = [np.array(data.get(i, []), dtype=np.int64) for i in range(nl)]
        self.time_mechs = [np.array(time.get(i, []), dtype=np.int64) for i in range(nl)]
        self.p_data = np.array([_or(pri[m]) for m in self.data_mechs])
        self.q_time = np.array([_or(pri[m]) for m in self.time_mechs])
        self.data_cond = [pri[m] / pri[m].sum() if len(m) else pri[m] for m in self.data_mechs]
        self.time_cond = [pri[m] / pri[m].sum() if len(m) else pri[m] for m in self.time_mechs]
        tot = {k: float(pri[self.kind == k].sum()) for k in (STATIC, DATA, TIME, OTHER)}
        linked = tot[DATA] + tot[TIME] + tot[OTHER]
        self.stats = {"mechanisms": nm, "static": int((self.kind == STATIC).sum()),
                      "data": int((self.kind == DATA).sum()), "time": int((self.kind == TIME).sum()),
                      "other": int((self.kind == OTHER).sum()),
                      "prob_static": tot[STATIC], "prob_data": tot[DATA], "prob_time": tot[TIME],
                      "prob_other": tot[OTHER],
                      "other_fraction_of_linked": tot[OTHER] / linked if linked else 0.0,
                      "links_without_data_edge": int(sum(len(m) == 0 for m in self.data_mechs))}

    def _setup_corrections(self) -> None:
        """Z on the top qubit of the link at the start of the round of every link detector."""
        ft = self.tl.ft
        geo = ft.geometry
        nl = len(self.link_cols)
        top = np.array([geo.verts(*geo.checks[self.chain_ci[c]]["block"])[0]
                        for c in self.lk_chain], dtype=np.int64)
        inserts, owner = [], []
        for i in range(nl):
            if self.lk_sub[i] == 0:
                inserts.append((ft.round_start[self.lk_round[i]], int(top[i]), "Z"))
                owner.append(i)
            else:
                # comparison of repetitions j - 1 and j of a link within one round: every gap between
                # two instructions of the round on the top qubit is a candidate, the earliest valid one wins
                r = int(self.lk_round[i])
                lo, hi = ft.round_start[r], ft.round_start[r + 1]
                for k in range(lo, hi):
                    inst = self.circuit[k]
                    if any(t.qubit_value == top[i] for t in inst.targets_copy()
                           if t.qubit_value is not None):
                        inserts.append((k + 1, int(top[i]), "Z"))
                        owner.append(i)
        dets, obs = pauli_signatures(self.circuit, inserts)
        self.corr_dets: List[Optional[np.ndarray]] = [None] * nl
        self.corr_obs: List[Optional[np.ndarray]] = [None] * nl
        for (pos, _, _), i, dd, oo in zip(inserts, owner, dets, obs):
            if self.corr_dets[i] is not None:
                continue
            link_part = dd[self.is_link[dd]]
            if len(link_part) == 1 and link_part[0] == self.link_cols[i]:
                self.corr_dets[i], self.corr_obs[i] = dd, oo
            elif self.lk_sub[i] == 0:
                raise RuntimeError(f"Z on the top qubit of link {self.chain_ci[self.lk_chain[i]]} at the "
                                   f"start of round {self.lk_round[i]} flips the link detectors "
                                   f"{list(link_part)}, not only {self.link_cols[i]}.")
        missing = [i for i in range(nl) if self.corr_dets[i] is None]
        if missing:
            raise RuntimeError(f"no single Z flips only the link detectors {missing[:5]} ...")
        self.corr_obs_mask = np.zeros((nl, self.no), dtype=bool)
        for i in range(nl):
            self.corr_obs_mask[i, self.corr_obs[i]] = True

    def _residual(self, j: int) -> Optional[Tuple[frozenset, frozenset, Tuple[int, ...]]]:
        """(non-link detectors, observables, corrections) of mechanism j at the upper level."""
        dets, obs = self.tl.mech_keys[j]
        D = frozenset(x for x in dets if not self.is_link[x])
        O = frozenset(obs)
        k = self.kind[j]
        if k in (STATIC, TIME):
            return D, O, ()
        corr = (int(self.link_of[[x for x in dets if self.is_link[x]][0]]),) if k == DATA \
            else self.other_links[j]
        for i in corr:
            cd = self.corr_dets[i]
            D = D.symmetric_difference(int(x) for x in cd if not self.is_link[x])
            O = O.symmetric_difference(int(x) for x in self.corr_obs[i])
        return D, O, corr

    def upper_signature(self, col: int) -> Tuple[frozenset, frozenset]:
        """(non-link detectors, observables) of an upper-level column (a mechanism or a pair)."""
        if col < len(self.tl.mech_keys):
            return self._residual(col)[:2]
        U, O = self.pair_info[col - len(self.tl.mech_keys)][:2]
        return U, O

    def _s0_split(self, dets: Sequence[int], obs: Sequence[int]):
        """Stim's graphlike split of an S_0 signature (combined detector ids, all S_0 detectors kept)."""
        idx = tuple(sorted(self.s0_index[x] for x in dets if x in self.s0_index))
        key = (idx, tuple(sorted(obs)))
        if len(idx) <= 2:
            return [(set(idx), set(key[1]))] if (idx or key[1]) else []
        eids = self.tl.s0_decomp.get(key)
        if eids is None:
            return None
        return [(set(self.tl.edge_keys[e][0]), set(self.tl.edge_keys[e][1])) for e in eids]

    def _split(self, faults: Sequence[int], corr: Tuple[int, ...]):
        """
        Graphlike split of the upper-level signature of some faults (one mechanism, or the two of a
        pair) and their corrections: the XOR of the splits of each, components joined through the link
        detectors they share.
        """
        comps = []
        for j in faults:
            dets, obs = self.tl.mech_keys[j]
            more = self._s0_split(dets, obs)
            if more is None:
                return None
            comps = comps + more
        for i in corr:
            more = self._s0_split(self.corr_dets[i], self.corr_obs[i])
            if more is None:
                return None
            comps = comps + more
        link_s0 = self.link_s0_idx
        while True:
            hit = None
            for a in range(len(comps)):
                for x in comps[a][0] & link_s0:
                    b = next((b for b in range(a + 1, len(comps)) if x in comps[b][0]), None)
                    if b is not None:
                        hit = (a, b)
                        break
                if hit:
                    break
            if hit is None:
                break
            a, b = hit
            comps[a] = (comps[a][0] ^ comps[b][0], comps[a][1] ^ comps[b][1])
            del comps[b]
        out_d, out_o = [], []
        for d_, o_ in comps:
            if d_ & link_s0:
                return None
            if not d_:
                if o_:
                    if not out_o and len(comps) == 1:
                        return None
                    out_o.append(o_)      # folded into a component with detectors below
                continue
            out_d.append([d_, set(o_)])
        if not out_d:
            return None
        for o_ in out_o:
            out_d[0][1] ^= o_
        res = []
        for d_, o_ in out_d:
            nodes = tuple(sorted(self.node_of_s0[x] for x in d_))
            if len(nodes) > 2:
                return None
            res.append((nodes, tuple(sorted(o_))))
        return res

    def _pairs(self) -> None:
        """Second-order s = 0 entries: pairs of data-edge faults of one link round, merged by signature."""
        nm = len(self.tl.mech_keys)
        pri = self.tl.priors
        own = {}
        self.pair_info: List[Tuple[frozenset, frozenset, Tuple[int, int], float, int]] = []
        self.pair_cols: List[np.ndarray] = []
        for i, ms in enumerate(self.data_mechs):
            cols = []
            if self.table == "full" and len(ms) >= 2:
                for j in ms:
                    if j not in own:
                        dets, obs = self.tl.mech_keys[j]
                        own[j] = (frozenset(x for x in dets if not self.is_link[x]), frozenset(obs))
                agg: Dict[Tuple[frozenset, frozenset], list] = {}
                for a_, b_ in itertools.combinations(ms, 2):
                    U, O = own[a_][0] ^ own[b_][0], own[a_][1] ^ own[b_][1]
                    if not U and not O:
                        continue
                    pab = pri[a_] * pri[b_]
                    e = agg.setdefault((U, O), [0.0, (int(a_), int(b_))])
                    e[0] = e[0] * (1 - pab) + pab * (1 - e[0])
                items = list(agg.items())
                if self.pair_top is not None:
                    # the most probable pair signatures of the link round (all of them under phen)
                    items = sorted(items, key=lambda kv: -kv[1][0])[:self.pair_top]
                for (U, O), (pq, ab) in items:
                    cols.append(nm + len(self.pair_info))
                    self.pair_info.append((U, O, ab, pq, i))
            self.pair_cols.append(np.array(cols, dtype=np.int64))

    def _setup_upper(self) -> None:
        tl = self.tl
        nm = len(tl.mech_keys)
        self._pairs()
        ncol = self.ncol = nm + len(self.pair_info)
        s0_cols = np.asarray(tl.s0_cols, dtype=np.int64)
        self.s0_index = {int(c): i for i, c in enumerate(s0_cols)}
        self.link_s0_idx = {self.s0_index[int(x)] for x in self.link_cols if int(x) in self.s0_index}
        self.up_cols = np.array([c for c in s0_cols if not self.is_link[c]], dtype=np.int64)
        node_of = {int(c): k for k, c in enumerate(self.up_cols)}
        self.node_of_s0 = {self.s0_index[c]: k for c, k in node_of.items()}
        self.bp_rows = np.flatnonzero(~self.is_link)
        bp_of = {int(c): k for k, c in enumerate(self.bp_rows)}
        edge_id: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], int] = {}
        a_rows, a_cols = [], []
        h_rows, h_cols = [], []
        self.bp_mechs = []
        self.never = np.zeros(ncol, dtype=bool)
        self.split_failed: List[int] = []
        self.undetectable: List[int] = []
        for j in range(ncol):
            if j < nm:
                if self.kind[j] == OTHER and any(self.corr_dets[i] is None for i in self.other_links[j]):
                    self.never[j] = True
                    continue
                D, O, corr = self._residual(j)
                faults = (j,)
            else:
                D, O, faults, _, _ = self.pair_info[j - nm]
                corr = ()
            if D:
                col = len(self.bp_mechs)
                self.bp_mechs.append(j)
                for x in D:
                    h_rows.append(bp_of[x])
                    h_cols.append(col)
            S = tuple(sorted(node_of[x] for x in D if x in node_of))
            if not S:
                if O and not D:
                    self.undetectable.append(j)
                continue
            if len(S) <= 2:
                comps = [(S, tuple(sorted(O)))]
            else:
                comps = self._split(faults, corr)
                if comps is None:
                    self.split_failed.append(j)
                    continue
            for key in comps:
                e = edge_id.setdefault(key, len(edge_id))
                a_rows.append(e)
                a_cols.append(j)
        self.edge_keys = list(edge_id)
        ne = len(self.edge_keys)
        self.A = sp.csr_matrix((np.ones(len(a_rows)), (a_rows, a_cols)), shape=(ne, ncol))
        er, ec, orow, ocol = [], [], [], []
        for e, (nodes, obs) in enumerate(self.edge_keys):
            er += list(nodes)
            ec += [e] * len(nodes)
            orow += list(obs)
            ocol += [e] * len(obs)
        self.E = sp.csc_matrix((np.ones(len(er), dtype=np.uint8), (er, ec)), shape=(len(self.up_cols), ne))
        self.F = sp.csc_matrix((np.ones(len(orow), dtype=np.uint8), (orow, ocol)), shape=(self.no, ne))
        self.bp_mechs = np.array(self.bp_mechs, dtype=np.int64)
        self.H_bp = sp.csc_matrix((np.ones(len(h_rows), dtype=np.uint8), (h_rows, h_cols)),
                                  shape=(len(self.bp_rows), len(self.bp_mechs)))
        # pairs are in by default and leave when their link round is corrected
        self.q_static = np.concatenate([np.where(self.kind == STATIC, tl.priors, 0.0),
                                        np.array([x[3] for x in self.pair_info])])
        # the edge sums at the default priors; a shot changes only the columns of its link decisions
        self.A_csc = self.A.tocsc()
        self.logkeep0 = self.A @ np.log1p(-np.clip(self.q_static, 0.0, 1 - 1e-12))
        self.nact0 = self.A @ (self.q_static > 0).astype(float)
        pri = np.concatenate([tl.priors, self.q_static[nm:]])
        self.stats.update(split_failed=len(self.split_failed),
                          prob_split_failed=float(pri[self.split_failed].sum()),
                          undetectable=len(self.undetectable),
                          prob_undetectable=float(pri[self.undetectable].sum()),
                          pair_columns=len(self.pair_info), prob_pairs=float(pri[nm:].sum()),
                          upper_edges=ne, upper_nodes=len(self.up_cols), bp_rows=len(self.bp_rows))

    def _setup_link_rule(self) -> None:
        nl = len(self.link_cols)
        self.w_time = _weights(self.q_time)         # time edge after each link detector (last: unused)
        # per chain: prefix sums of the time-edge weights and the weights of the steps beyond its ends
        self.chain_W, self.chain_ends = [], []
        for ch in self.chains:
            w = self.w_time[ch[:-1]]
            self.chain_W.append(np.concatenate([[0.0], np.cumsum(w)]))
            self.chain_ends.append((float(w[0]), float(w[-1])) if len(ch) > 1 else (0.0, 0.0))
        with np.errstate(divide="ignore"):
            self.logq = np.log(self.q_time)
            self.logp = np.log(self.p_data)
        if self.link_rule == "fast":
            import pymatching

            m = pymatching.Matching()
            for c, ch in enumerate(self.chains):
                for a, b in zip(ch[:-1], ch[1:]):
                    m.add_edge(int(a), int(b), weight=float(self.w_time[a]),
                               error_probability=float(self.q_time[a]))
                for a in ch:
                    m.add_boundary_edge(int(a), weight=float(_weights(self.p_data[a:a + 1])[0]),
                                        error_probability=float(self.p_data[a]))
            self._link_match = m

    def _setup_bp(self, iters: int, method: str) -> None:
        from ldpc import BpDecoder

        q = np.clip(self.q_static[self.bp_mechs], BP_FLOOR, 1 - BP_FLOOR)
        self._bp = BpDecoder(self.H_bp, error_channel=list(q), max_iter=iters, bp_method=method,
                             input_vector_type="syndrome")

    # ------------------------------------------------------------------ link level
    def link_decisions(self, link_row: np.ndarray, rule: Optional[str] = None
                       ) -> Tuple[List[int], List[int]]:
        """
        (corrected link detectors, time edges of removed pairs) for the link events of one shot.
        Both are indices into `link_cols`; a time edge is named by its earlier detector.
        """
        rule = rule or self.link_rule
        ev = np.flatnonzero(link_row)
        if not ev.size:
            return [], []
        if rule == "fast":
            corr, time = [], []
            for a, b in self._link_match.decode_to_edges_array(np.asarray(link_row, dtype=np.uint8)):
                a, b = int(a), int(b)
                if b < 0 or a < 0:
                    corr.append(max(a, b))
                else:
                    time.append(a if self.lk_pos[a] < self.lk_pos[b] else b)
            return sorted(corr), sorted(time)
        corr, time = [], []
        by_chain: Dict[int, List[int]] = {}
        for i in ev:
            by_chain.setdefault(int(self.lk_chain[i]), []).append(int(i))
        for c, items in by_chain.items():
            ch = self.chains[c]
            pos = sorted(int(self.lk_pos[i]) for i in items)
            pairs, ends = self._pair_chain(c, pos)
            for t in ends:
                corr.append(int(ch[t]))
            for t1, t2 in pairs:
                seg = ch[t1:t2]
                if self.logq[seg].sum() > self.logp[ch[t1]] + self.logp[ch[t2]]:
                    time += [int(x) for x in seg]
                else:
                    corr += [int(ch[t1]), int(ch[t2])]
        return sorted(corr), sorted(time)

    def _pair_chain(self, c: int, pos: List[int]):
        """Minimum-weight pairing of events along chain c, with a boundary one step beyond each end."""
        W = self.chain_W[c]
        n = len(W)
        wst, wen = self.chain_ends[c]
        k = len(pos)

        def pair_cost(s, t):
            return sum(W[pos[i + 1]] - W[pos[i]] for i in range(s, t, 2))

        start = W[pos[0]] + wst
        end = W[n - 1] - W[pos[-1]] + wen
        if k % 2 == 0:
            opts = [(pair_cost(0, k), 0, k, [])]
            if k >= 2:
                opts.append((start + end + pair_cost(1, k - 1), 1, k - 1, [pos[0], pos[-1]]))
        else:
            opts = [(start + pair_cost(1, k), 1, k, [pos[0]]),
                    (end + pair_cost(0, k - 1), 0, k - 1, [pos[-1]])]
        _, s, t, ends = min(opts, key=lambda o: o[0])
        pairs = [(pos[i], pos[i + 1]) for i in range(s, t, 2)]
        return pairs, ends

    def changes(self, corr: Sequence[int], time: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        """The upper-level columns whose priors differ from `q_static` in a shot, and their values."""
        cols, vals = [], []
        for i in corr:
            cols += [self.data_mechs[i], self.pair_cols[i]]
            vals += [self.data_cond[i], np.zeros(len(self.pair_cols[i]))]
        for i in time:
            cols.append(self.time_mechs[i])
            vals.append(self.time_cond[i])
        if self.other_of:
            cs, seen = set(corr), set()
            for i in corr:
                for j in self.other_of.get(i, ()):
                    if j not in seen and not self.never[j] and all(x in cs for x in self.other_links[j]):
                        seen.add(j)
                        cols.append(np.array([j]))
                        vals.append(self.tl.priors[[j]])
        if not cols:
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        return np.concatenate(cols).astype(np.int64), np.concatenate(vals)

    def priors(self, corr: Sequence[int], time: Sequence[int]) -> np.ndarray:
        """Per-shot priors of every upper-level column (zero: the column is left out)."""
        q = self.q_static.copy()
        cols, vals = self.changes(corr, time)
        q[cols] = vals
        return q

    # ------------------------------------------------------------------ upper level
    def _match(self, p_e: np.ndarray, active: np.ndarray, syn: np.ndarray) -> np.ndarray:
        import pymatching

        try:
            m = pymatching.Matching.from_check_matrix(
                self.E[:, active], weights=_weights(p_e[active]), faults_matrix=self.F[:, active],
                use_virtual_boundary_node=True)
            return m.decode(syn)
        except ValueError:
            # a detection event that no edge of this shot can explain: fall back to every edge
            self.counters["match_retry"] += 1
            m = pymatching.Matching.from_check_matrix(
                self.E, weights=_weights(np.maximum(p_e, 1e-12)), faults_matrix=self.F,
                use_virtual_boundary_node=True)
            return m.decode(syn)

    def decode_shot(self, row: np.ndarray) -> np.ndarray:
        row = np.array(row, dtype=np.uint8)
        corr, time = self.link_decisions(row[self.link_cols])
        pred = np.zeros(self.no, dtype=bool)
        for i in corr:
            row[self.corr_dets[i]] ^= 1
            pred ^= self.corr_obs_mask[i]
            if not len(self.data_mechs[i]):
                self.counters["empty_data_edge"] += 1
        for i in time:
            ch = self.chains[self.lk_chain[i]]
            row[self.link_cols[i]] ^= 1
            row[self.link_cols[ch[self.lk_pos[i] + 1]]] ^= 1
        if row[self.link_cols].any():
            raise RuntimeError("link events left after the link decisions")
        self.counters["shots"] += 1
        self.counters["corrections"] += len(corr)
        self.counters["removed_pairs"] += len(time)
        cols, vals = self.changes(corr, time)
        old = self.q_static[cols]
        sub = self.A_csc[:, cols]
        active = (self.nact0 + sub @ ((vals > 0).astype(float) - (old > 0))) > 0.5
        if self.upper == "bm":
            q = self.q_static.copy()
            q[cols] = vals
            syn = row[self.bp_rows]
            if syn.any():
                self._bp.update_channel_probs(np.clip(q[self.bp_mechs], BP_FLOOR, 1 - BP_FLOOR))
                self._bp.decode(syn)
                llr = np.asarray(self._bp.log_prob_ratios)
                q[self.bp_mechs] = np.where(q[self.bp_mechs] > 0, 1.0 / (1.0 + np.exp(llr)), 0.0)
            p_e = -np.expm1(self.A @ np.log1p(-np.clip(q, 0.0, 1 - 1e-12)))
        else:
            # only the changed columns move the edge sums away from their defaults
            dl = np.log1p(-np.clip(vals, 0.0, 1 - 1e-12)) - np.log1p(-np.clip(old, 0.0, 1 - 1e-12))
            p_e = -np.expm1(self.logkeep0 + sub @ dl)
        syn = row[self.up_cols]
        if not syn.any():
            return pred
        return pred ^ np.asarray(self._match(p_e, active, syn), dtype=bool)

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        dets = np.asarray(dets, dtype=np.uint8)
        out = np.zeros((dets.shape[0], self.no), dtype=bool)
        for s in range(dets.shape[0]):
            out[s] = self.decode_shot(dets[s])
        return out
