"""

two_level_decoder.py
----------------------------------------------------------------------------
Lower-level inference passed as priors (or erasures) to an upper-level
matching decoder, for circuit-level noise.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import stim

from .ft_mdr_circuit import FTMDRCircuit
from .s0_matching_decoder import LogicalErrorEstimate

S0_PHASES = {0, 1, 2}
GAUGE_PHASES = {3, 4}


class TwoLevelDecoder:
    """
    Two-level decoder for frame-initialised XYZ^2 MDR circuits.

    The upper level is the graph of the frame-deterministic ($S_0$)
    detectors, which decides the logical outcome and alone has the full
    fault distance. The lower level is formed by the gauge detectors (odd
    links, class-A hexagons, left and right boundary checks), whose first
    outcomes are random and which are compared round to round from round 2.
    Modes, with the names used in the paper:

    - ``mwpm`` (MWPM): matching on the $S_0$ graph with static weights.
    - ``seq_soft`` (sequential BP): BP on the link detectors only, posteriors
      pushed onto the $S_0$ edges, then matching. Circuit-level analogue of
      the sequential decoder of Srivastava et al. (2025).
    - ``seq_erasure``: as ``seq_soft`` but every $S_0$ edge whose posterior
      exceeds ``erasure_threshold`` is set to $p=1/2$.
    - ``seq_match`` (erasure passing): matching on the link detectors; the
      $S_0$ edges of every fault with a matched link signature become
      erasures ($p=1/2$).
    - ``bp_full`` (belief-matching): BP on the full detector error model,
      posteriors pushed onto the $S_0$ edges, then matching.
    - ``corr_split`` (hierarchical correlated matching, HCM): each fault is
      split into its upper part and its lower part, and PyMatching's
      two-pass correlated matching reweights each level with the edges
      matched on the other. ``lower="gauge"`` (default, HCM) uses every gauge
      detector, ``lower="links"`` only the odd links ("HCM with links").
    - ``tesseract``: Tesseract on the full detector error model (reference).
    - ``cfe``: coset free-energy decoder (two-class BP-OSD, local descent
      with degeneracy moves and a free-energy comparison of the two logical
      classes, see `coset_decoder.py`). ``osd_method="osd0"`` gives the
      variant without the combination sweep (CFE-0), whose memory and time
      grow much more slowly with the size of the circuit.

    The circuit must be built with ``detectors="combined"``.
    """

    MODES = ("mwpm", "seq_soft", "seq_erasure", "seq_match", "bp_full",
             "corr_split", "bp_corr", "tesseract", "cfe", "tnml", "cfe_tn")

    def __init__(self, ft: FTMDRCircuit, mode: str = "bp_full",
                 bp_iters: int = 30, erasure_threshold: float = 0.2,
                 bp_method: str = "product_sum", lower: str = "gauge",
                 kappa: float = 0.5, osd_order: int = 10,
                 cfe_guided: bool = True, cfe_gate: Optional[float] = None,
                 osd_method: str = "osd_cs", osd_impl: str = "fast",
                 chi: int = 32, chi_max: int = 256, tn_tol: float = 0.05,
                 tn_max_open: int = 64) -> None:
        if ft.detectors != "combined":
            raise ValueError("TwoLevelDecoder needs detectors='combined'.")
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        if lower not in ("gauge", "links"):
            raise ValueError("lower must be 'gauge' or 'links'.")
        self.ft = ft
        self.mode = mode
        self.lower = lower
        self.bp_iters = bp_iters
        self.kappa = kappa
        self.osd_order = osd_order
        self.osd_method = osd_method
        self.osd_impl = osd_impl
        self.chi, self.chi_max, self.tn_tol, self.tn_max_open = chi, chi_max, tn_tol, tn_max_open
        self.tn_used = 0          # shots decided by the converged tensor network (cfe_tn)
        self.cfe_guided = cfe_guided
        self.cfe_gate = cfe_gate
        self.erasure_threshold = erasure_threshold
        self.circuit = ft.build()
        self._setup_columns()
        self._setup_full_dem()
        self._setup_upper_graph()
        self._setup_lower_level(bp_method)

    # ------------------------------------------------------------------ setup
    def _setup_columns(self) -> None:
        coords = self.circuit.get_detector_coordinates()
        nd = self.circuit.num_detectors
        phase = np.array([int(coords[i][2]) for i in range(nd)])
        self.s0_cols = np.flatnonzero(np.isin(phase, list(S0_PHASES)))
        self.gauge_cols = np.flatnonzero(np.isin(phase, list(GAUGE_PHASES)))
        geo = self.ft.geometry
        link_rows = {k for k, row in enumerate(self.ft.frame.s0_rows)
                     if row.sum() == 1
                     and geo.checks[int(np.flatnonzero(row)[0])]["kind"] == "link"}
        is_link = np.zeros(nd, dtype=bool)
        for i in range(nd):
            k, ph = int(coords[i][0]), int(coords[i][2])
            if ph in S0_PHASES and k in link_rows:
                is_link[i] = True
            if ph in GAUGE_PHASES and geo.checks[k]["kind"] == "link":
                is_link[i] = True
        self.link_cols = np.flatnonzero(is_link)
        s0_ft = FTMDRCircuit(
            self.ft.geometry.d, self.ft.rounds, self.ft.noise,
            schedule=self.ft.schedule, init=self.ft.init, final=self.ft.final,
            detectors="s0", odd_rep=self.ft.frame.odd_rep,
            share_ancillas=len(self.ft.shared_pairs),
        )
        self.s0_circuit = s0_ft.build()
        s0_coords = self.s0_circuit.get_detector_coordinates()
        assert len(s0_coords) == len(self.s0_cols)
        for a, b in zip(self.s0_cols, range(len(s0_coords))):
            assert list(coords[int(a)]) == list(s0_coords[b])

    def _setup_full_dem(self) -> None:
        dem = self.circuit.detector_error_model(
            decompose_errors=False, approximate_disjoint_errors=True)
        self.full_dem = dem
        cols: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], int] = {}
        priors: List[float] = []
        for inst in dem.flattened():
            if inst.type != "error":
                continue
            p = inst.args_copy()[0]
            dets, obs = set(), set()
            for t in inst.targets_copy():
                if t.is_relative_detector_id():
                    dets ^= {t.val}
                elif t.is_logical_observable_id():
                    obs ^= {t.val}
            key = (tuple(sorted(dets)), tuple(sorted(obs)))
            if key not in cols:
                cols[key] = len(priors)
                priors.append(0.0)
            j = cols[key]
            priors[j] = priors[j] * (1 - p) + p * (1 - priors[j])
        self.mech_keys = list(cols.keys())
        self.priors = np.array(priors)
        nd = self.circuit.num_detectors
        rows, cs = [], []
        for (dets, _), j in cols.items():
            for dd in dets:
                rows.append(dd)
                cs.append(j)
        self.H_full = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.uint8), (rows, cs)),
            shape=(nd, len(priors)))

    def _setup_upper_graph(self) -> None:
        dem = self.s0_circuit.detector_error_model(
            decompose_errors=True, approximate_disjoint_errors=True)
        edge_ids: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], int] = {}
        decomp: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], List[int]] = {}
        edge_prior: Dict[int, float] = {}
        for inst in dem.flattened():
            if inst.type != "error":
                continue
            p = inst.args_copy()[0]
            comps: List[Tuple[set, set]] = [(set(), set())]
            for t in inst.targets_copy():
                if t.is_separator():
                    comps.append((set(), set()))
                elif t.is_relative_detector_id():
                    comps[-1][0].symmetric_difference_update({t.val})
                elif t.is_logical_observable_id():
                    comps[-1][1].symmetric_difference_update({t.val})
            all_d, all_o = set(), set()
            eids = []
            for dset, oset in comps:
                all_d ^= dset
                all_o ^= oset
                ek = (tuple(sorted(dset)), tuple(sorted(oset)))
                if not ek[0]:
                    continue
                if ek not in edge_ids:
                    edge_ids[ek] = len(edge_ids)
                eids.append(edge_ids[ek])
                e = edge_ids[ek]
                edge_prior[e] = edge_prior.get(e, 0.0) * (1 - p) + p * (
                    1 - edge_prior.get(e, 0.0))
            key = (tuple(sorted(all_d)), tuple(sorted(all_o)))
            if key not in decomp or len(eids) < len(decomp[key]):
                decomp[key] = eids
        self.edge_keys = list(edge_ids.keys())
        self.num_edges = len(self.edge_keys)
        s0_index = {int(c): i for i, c in enumerate(self.s0_cols)}
        rows, cs = [], []
        for j, (dets, obs) in enumerate(self.mech_keys):
            s0d = tuple(sorted(s0_index[d] for d in dets if d in s0_index))
            if not s0d:
                continue
            key = (s0d, obs)
            if len(s0d) <= 2:
                if key not in edge_ids:
                    edge_ids[key] = len(edge_ids)
                    self.edge_keys.append(key)
                eids = [edge_ids[key]]
            else:
                eids = decomp.get(key)
                if eids is None:
                    continue
            for e in eids:
                rows.append(e)
                cs.append(j)
        self.num_edges = len(self.edge_keys)
        self.A = sp.csr_matrix((np.ones(len(rows)), (rows, cs)),
                               shape=(self.num_edges, len(self.mech_keys)))
        n_s0 = len(self.s0_cols)
        er, ec = [], []
        orow, ocol = [], []
        for e, (dets, obs) in enumerate(self.edge_keys):
            for dd in dets:
                er.append(dd)
                ec.append(e)
            for o in obs:
                orow.append(o)
                ocol.append(e)
        self.E = sp.csc_matrix((np.ones(len(er), dtype=np.uint8), (er, ec)),
                               shape=(n_s0, self.num_edges))
        self.F = sp.csc_matrix((np.ones(len(orow), dtype=np.uint8),
                                (orow, ocol)),
                               shape=(self.circuit.num_observables,
                                      self.num_edges))
        self.static_p = self._edge_probs(self.priors)

    def _setup_lower_level(self, bp_method: str) -> None:
        self._bp = None
        self._tess = None
        if self.mode in ("seq_soft", "seq_erasure", "bp_full", "bp_corr"):
            from ldpc import BpDecoder

            rows = self.link_cols if self.mode not in ("bp_full", "bp_corr") \
                else np.arange(self.circuit.num_detectors)
            self._bp_rows = rows
            H = self.H_full[rows, :]
            self._bp = BpDecoder(H.tocsc(), error_channel=list(self.priors),
                                 max_iter=self.bp_iters, bp_method=bp_method,
                                 input_vector_type="syndrome")
        if self.mode in ("tnml", "cfe_tn"):
            from .tn_decoder import TensorNetworkDecoder, detector_positions

            pos = detector_positions(self.ft, self.circuit)
            self._tn = TensorNetworkDecoder([k[0] for k in self.mech_keys],
                                            [0 in k[1] for k in self.mech_keys],
                                            self.priors, pos, chi=self.chi)
            self.tn_width = self._tn.sched.max_open
        if self.mode in ("cfe", "cfe_tn"):
            from .coset_decoder import CosetFreeEnergyDecoder

            guide = None
            if self.cfe_guided:
                import pymatching

                self.split_dem = self._build_split_dem()
                self._corr = pymatching.Matching.from_detector_error_model(
                    self.split_dem, enable_correlations=True)
                edge_mechs: Dict[Tuple[int, ...], List[int]] = {}
                for j, txt in self._split_terms:
                    for comp in txt.split("^"):
                        key = tuple(sorted(int(t[1:]) for t in comp.split()
                                           if t.startswith("D")))
                        edge_mechs.setdefault(key, []).append(j)

                def guide(det_row, _em=edge_mechs):
                    # raise the priors of the faults on the HCM solution
                    edges = self._corr.decode_to_edges_array(
                        np.asarray(det_row, dtype=np.uint8),
                        enable_correlations=True)
                    q = self.priors.copy()
                    for a, b in edges:
                        key = tuple(sorted(int(x) for x in (a, b) if x >= 0))
                        for j in _em.get(key, ()):
                            q[j] = max(q[j], 0.3)
                    return np.clip(q, 1e-12, 0.49)

            self._cfe = CosetFreeEnergyDecoder(
                self.H_full, self.mech_keys, self.priors, kappa=self.kappa,
                osd_order=self.osd_order, bp_iters=self.bp_iters, guide=guide,
                gate=self.cfe_gate, osd_method=self.osd_method, osd_impl=self.osd_impl)
        if self.mode == "tesseract":
            import tesseract_decoder.tesseract as tesseract

            cfg = tesseract.TesseractConfig(dem=self.full_dem, det_beam=20,
                                            beam_climbing=True)
            self._tess = cfg.compile_decoder()
        if self.mode == "mwpm":
            import pymatching

            self._static = pymatching.Matching.from_check_matrix(
                self.E, weights=self._weights(self.static_p),
                faults_matrix=self.F, use_virtual_boundary_node=True)
        if self.mode in ("corr_split", "bp_corr"):
            import pymatching

            self.split_dem = self._build_split_dem()
            self._corr = pymatching.Matching.from_detector_error_model(
                self.split_dem, enable_correlations=True)
        if self.mode == "seq_match":
            self._setup_link_matching()

    def _build_split_dem(self) -> stim.DetectorErrorModel:
        """
        Decompose every fault into its $S_0$ part and its gauge part.
        """
        s0set = set(int(c) for c in self.s0_cols)
        keep = set(range(self.circuit.num_detectors))
        if self.lower == "links":
            keep = s0set | set(int(c) for c in self.link_cols)
        self._lower_mask = np.array(
            [i in keep for i in range(self.circuit.num_detectors)])
        s0_rev = {i: int(c) for i, c in enumerate(self.s0_cols)}
        s0_index = {int(c): i for i, c in enumerate(self.s0_cols)}
        dem_s0 = self.s0_circuit.detector_error_model(
            decompose_errors=True, approximate_disjoint_errors=True)
        decomp: Dict = {}
        for inst in dem_s0.flattened():
            if inst.type != "error":
                continue
            comps = [[set(), set()]]
            for t in inst.targets_copy():
                if t.is_separator():
                    comps.append([set(), set()])
                elif t.is_relative_detector_id():
                    comps[-1][0] ^= {t.val}
                elif t.is_logical_observable_id():
                    comps[-1][1] ^= {t.val}
            tot_d, tot_o = set(), set()
            for d_, o_ in comps:
                tot_d ^= d_
                tot_o ^= o_
            key = (tuple(sorted(tot_d)), tuple(sorted(tot_o)))
            cl = [(tuple(sorted(s0_rev[x] for x in d_)), tuple(sorted(o_)))
                  for d_, o_ in comps]
            if key not in decomp or len(cl) < len(decomp[key]):
                decomp[key] = cl
        gauge_edges = set()
        for dets, _ in self.mech_keys:
            g = tuple(sorted(x for x in dets if x not in s0set and x in keep))
            if 0 < len(g) <= 2:
                gauge_edges.add(g)
        lines = []
        self._split_terms = []
        self.split_dropped = 0
        for j, (dets, obs) in enumerate(self.mech_keys):
            S = tuple(sorted(x for x in dets if x in s0set))
            G = tuple(sorted(x for x in dets if x not in s0set and x in keep))
            comps = []
            if len(S) <= 2:
                if S:
                    comps.append((S, obs))
                elif obs:
                    self.split_dropped += 1
                    continue
            else:
                key = (tuple(sorted(s0_index[x] for x in S)), obs)
                if key not in decomp:
                    self.split_dropped += 1
                    continue
                comps += decomp[key]
            if 0 < len(G) <= 2:
                comps.append((G, ()))
            elif len(G) > 2:
                rest, parts, ok = list(G), [], True
                while rest:
                    a = rest.pop(0)
                    b = next((x for x in rest
                              if tuple(sorted((a, x))) in gauge_edges), None)
                    if b is None:
                        if (a,) in gauge_edges:
                            parts.append((a,))
                        else:
                            ok = False
                            break
                    else:
                        rest.remove(b)
                        parts.append(tuple(sorted((a, b))))
                if ok:
                    comps += [(pp, ()) for pp in parts]
            if not comps:
                continue
            txt = " ^ ".join(
                " ".join([f"D{x}" for x in d_] + [f"L{o}" for o in o_])
                for d_, o_ in comps)
            lines.append(f"error({float(self.priors[j])!r}) {txt}")
            self._split_terms.append((j, txt))
        self._split_tail = [f"detector D{i}" for i in range(self.circuit.num_detectors)]
        lines += self._split_tail
        return stim.DetectorErrorModel("\n".join(lines))

    def _setup_link_matching(self) -> None:
        """
        Lower-level matching graph on the link detectors.
        """
        import pymatching

        link = [int(x) for x in self.link_cols]
        pos = {c: i for i, c in enumerate(link)}
        sig_of: Dict[Tuple[int, ...], List[int]] = {}
        for j, (dets, _) in enumerate(self.mech_keys):
            sig = tuple(sorted(pos[x] for x in dets if x in pos))
            if 0 < len(sig) <= 2:
                sig_of.setdefault(sig, []).append(j)
        self._sig_list = list(sig_of.keys())
        self._sig_mechs = [sig_of[k] for k in self._sig_list]
        rows, cols, probs = [], [], []
        for e, sig in enumerate(self._sig_list):
            q = 0.0
            for j in sig_of[sig]:
                q = q * (1 - self.priors[j]) + self.priors[j] * (1 - q)
            probs.append(q)
            for x in sig:
                rows.append(x)
                cols.append(e)
        H = sp.csc_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)),
                          shape=(len(link), len(self._sig_list)))
        self._link_H = H
        self._link_match = pymatching.Matching.from_check_matrix(
            H, weights=self._weights(np.array(probs)),
            use_virtual_boundary_node=True)
        self._sig_index = {k: i for i, k in enumerate(self._sig_list)}
        # edge -> mechanisms -> upper edges
        mech_to_edges = self.A.T.tocsr()
        self._sig_upper = []
        for mechs in self._sig_mechs:
            ups = set()
            for j in mechs:
                ups.update(mech_to_edges.indices[mech_to_edges.indptr[j]:
                                                 mech_to_edges.indptr[j + 1]])
            self._sig_upper.append(np.array(sorted(ups), dtype=np.int64))

    # ------------------------------------------------------------------ core
    def _edge_probs(self, q: np.ndarray) -> np.ndarray:
        q = np.clip(q, 0.0, 1 - 1e-12)
        log_keep = self.A @ np.log1p(-q)
        return -np.expm1(log_keep)

    @staticmethod
    def _weights(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-12, 0.5 - 1e-9)
        return np.log((1 - p) / p)

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        """
        Predict observable flips for detection events of the combined circuit.
        """
        dets = np.asarray(dets, dtype=np.uint8)
        s0 = dets[:, self.s0_cols]
        if self.mode == "mwpm":
            return np.asarray(self._static.decode_batch(s0), dtype=bool)
        if self.mode == "tesseract":
            return np.asarray(self._tess.decode_batch(dets.astype(bool)),
                              dtype=bool)
        if self.mode == "cfe":
            return self._cfe.decode_batch(dets)
        if self.mode in ("tnml", "cfe_tn"):
            return self._decode_tn(dets)
        if self.mode == "corr_split":
            d2 = dets if self.lower == "gauge" else dets * self._lower_mask
            return np.asarray(self._corr.decode_batch(
                d2, enable_correlations=True), dtype=bool)
        if self.mode == "bp_corr":
            return self._decode_bp_corr(dets)
        import pymatching

        if self.mode == "seq_match":
            out = np.zeros((dets.shape[0], self.circuit.num_observables),
                           dtype=bool)
            link = dets[:, self.link_cols]
            for i in range(dets.shape[0]):
                p_e = self.static_p
                if link[i].any():
                    pairs = self._link_match.decode_to_edges_array(link[i])
                    hit = []
                    for a, b in pairs:
                        sig = tuple(sorted(x for x in (int(a), int(b))
                                           if x >= 0))
                        k = self._sig_index.get(sig)
                        if k is not None:
                            hit.append(self._sig_upper[k])
                    if hit:
                        p_e = self.static_p.copy()
                        p_e[np.concatenate(hit)] = 0.5
                m = pymatching.Matching.from_check_matrix(
                    self.E, weights=self._weights(p_e), faults_matrix=self.F,
                    use_virtual_boundary_node=True)
                out[i] = m.decode(s0[i])
            return out

        out = np.zeros((dets.shape[0], self.circuit.num_observables),
                       dtype=bool)
        for i in range(dets.shape[0]):
            syn = dets[i, self._bp_rows]
            if not syn.any() and self.mode != "bp_full":
                p_e = self._edge_probs(self._posterior_quiet())
            else:
                self._bp.decode(syn)
                llr = np.asarray(self._bp.log_prob_ratios)
                q = 1.0 / (1.0 + np.exp(llr))
                p_e = self._edge_probs(q)
            if self.mode == "seq_erasure":
                p_e = np.where(p_e > self.erasure_threshold, 0.5, p_e)
            m = pymatching.Matching.from_check_matrix(
                self.E, weights=self._weights(p_e), faults_matrix=self.F,
                use_virtual_boundary_node=True)
            out[i] = m.decode(s0[i])
        return out

    def _split_dem_with(self, q: np.ndarray) -> stim.DetectorErrorModel:
        """
        The split detector error model with mechanism probabilities `q`.
        """
        q = np.clip(q, 1e-9, 0.5 - 1e-6)
        lines = [f"error({q[j]:.9g}) {txt}" for j, txt in self._split_terms]
        return stim.DetectorErrorModel("\n".join(lines + self._split_tail))

    def _decode_bp_corr(self, dets: np.ndarray) -> np.ndarray:
        """
        Belief-seeded hierarchical correlated matching.

        BP on the full detector error model gives a posterior for every fault.
        The posteriors replace the priors of the split (two-level) model, which
        is then decoded with two-pass correlated matching.
        """
        import pymatching

        out = np.zeros((dets.shape[0], self.circuit.num_observables),
                       dtype=bool)
        for i in range(dets.shape[0]):
            syn = dets[i]
            if not syn.any():
                continue
            self._bp.decode(syn)
            llr = np.asarray(self._bp.log_prob_ratios)
            q = 1.0 / (1.0 + np.exp(llr))
            m = pymatching.Matching.from_detector_error_model(
                self._split_dem_with(q), enable_correlations=True)
            out[i] = m.decode(syn, enable_correlations=True)
        return out

    def _posterior_quiet(self) -> np.ndarray:
        cached = getattr(self, "_quiet_q", None)
        if cached is None:
            syn = np.zeros(len(self._bp_rows), dtype=np.uint8)
            self._bp.decode(syn)
            llr = np.asarray(self._bp.log_prob_ratios)
            cached = 1.0 / (1.0 + np.exp(llr))
            self._quiet_q = cached
        return cached

    def tn_shot(self, det_row: np.ndarray):
        """
        Tensor-network decision of one shot with a converged bond dimension.

        Starts at `chi` and doubles until the truncated weight is below 1e-9 or the
        free-energy difference changes by less than `tn_tol` between chi/2 and chi,
        up to `chi_max`. Returns (decision, ln Z_1 - ln Z_0, converged).
        """
        chi = self.chi
        prev = None
        while True:
            l0, l1, disc = self._tn.log_z(det_row, chi=chi)
            df = l1 - l0
            if disc < 1e-9 or (prev is not None and abs(df - prev) < self.tn_tol
                               and (df > 0) == (prev > 0)):
                return int(df > 0), df, True
            if chi >= self.chi_max:
                return int(df > 0), df, False
            prev = df
            chi *= 2

    def _decode_tn(self, dets: np.ndarray) -> np.ndarray:
        """
        ``tnml``: maximum likelihood by the tensor network (approximate when the bond
        dimension does not converge, counted in `tn_unconverged`).
        ``cfe_tn``: the converged tensor network where its frontier has at most
        `tn_max_open` detectors, the CFE decoder otherwise; `tn_used` counts the shots
        decided by the tensor network.
        """
        dets = np.asarray(dets, dtype=np.uint8)
        out = np.zeros((dets.shape[0], 1), dtype=bool)
        use_tn = self.mode == "tnml" or self.tn_width <= self.tn_max_open
        for s in range(dets.shape[0]):
            row = dets[s]
            if use_tn:
                bit, _, ok = self.tn_shot(row)
                if ok or self.mode == "tnml":
                    out[s, 0] = bool(bit)
                    if ok:
                        self.tn_used += 1
                    else:
                        self.tn_unconverged = getattr(self, "tn_unconverged", 0) + 1
                    continue
            out[s, 0] = bool(self._cfe.decode_shot(row)[0])
        return out

    def estimate(self, max_shots: int = 100_000, max_errors: int = 500,
                 batch: int = 2000, seed: Optional[int] = None,
                 time_limit: float = 600.0) -> LogicalErrorEstimate:
        """
        Monte-Carlo logical error rate of the combined circuit.
        """
        sampler = self.circuit.compile_detector_sampler(seed=seed)
        shots = errors = 0
        start = time.time()
        while (shots < max_shots and errors < max_errors
               and time.time() - start < time_limit):
            size = min(batch, max_shots - shots)
            dets, obs = sampler.sample(size, separate_observables=True)
            pred = self.decode_batch(dets)
            errors += int(np.sum(np.any(pred != obs, axis=1)))
            shots += size
        return LogicalErrorEstimate(shots, errors, time.time() - start)
