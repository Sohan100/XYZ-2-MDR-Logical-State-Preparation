"""

competitor_decoders.py
----------------------------------------------------------------------------
Decoders built from the detector error model of a circuit (for the competitor
circuits of `competitor_circuits.py`), with the interface of `TwoLevelDecoder`.

    dec = competitor_decoder(circuit, "bposd")
    pred = dec.decode_batch(dets)          # (shots, observables) bool
    est = dec.estimate(max_shots=10**5, max_errors=1000)

Names mirror our decoders (`scripts/run_decoder_threshold_sweep.py`,
`DECODERS`) where they apply:

- ``mwpm``: PyMatching on Stim's decomposed (graphlike) model.
- ``corr``: PyMatching's two-pass correlated matching
  (``enable_correlations=True``) on the same decomposed model. Our ``corr_*``
  decoders use the same PyMatching call on a split model; a surface code has
  no gauge level, so the plain decomposed model is the counterpart. The
  reweighting pass decodes a few single faults of d = 3 surface-code
  circuits wrongly (faults that Stim splits into three pieces, two of them
  boundary edges; Stim's own generated surface-code circuits show the same);
  every single fault is corrected from d = 5 on.
- ``bm``: belief-matching (Higgott et al., PRX 13, 031007 (2023)) with the
  settings of our ``bm`` (`TwoLevelDecoder` mode ``bp_full`` with
  ``bp_method="product_sum"`` and 10 iterations): product-sum BP of ldpc on
  the full, undecomposed model (faults with equal symptoms merged), every
  posterior pushed onto the matching edges of its graphlike decomposition as
  p_e = 1 - prod(1 - q_j), then matching on weights log((1 - p_e) / p_e).
- ``bm_pkg``: the ``beliefmatching`` package itself (BeliefMatching on the
  decomposed model). The package is optional and not installed in the
  project environment; asking for it without the package raises ImportError.
- ``bposd``: ldpc's BpOsdDecoder on the full model, product-sum BP with 30
  iterations (as our CFE) and OSD-CS of order 10. ``impl="fast"`` uses
  `fast_osd.FastBpOsd` instead (the same decoding with memory that grows
  slowly with the circuit, as our CFE uses at large d).
- ``tesseract``: Tesseract on the full model with ``det_beam=20`` and
  ``beam_climbing=True``, exactly as `TwoLevelDecoder` mode ``tesseract``.

All models are built with ``approximate_disjoint_errors=True``, as in
`TwoLevelDecoder` (PAULI_CHANNEL_1/2 and the EM3 correlated errors need it).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import stim
from scipy.special import expit

from .s0_matching_decoder import LogicalErrorEstimate

# Default options of every decoder name; keyword arguments of
# `competitor_decoder` override them.
DECODERS: Dict[str, dict] = {
    "mwpm": dict(),
    "corr": dict(),
    "bm": dict(bp_method="product_sum", bp_iters=10),
    "bm_pkg": dict(bp_iters=20),
    "bposd": dict(bp_method="product_sum", bp_iters=30, osd_method="osd_cs",
                  osd_order=10, impl="ldpc"),
    "tesseract": dict(det_beam=20, beam_climbing=True),
}

Key = Tuple[Tuple[int, ...], Tuple[int, ...]]


def merged_mechanisms(dem: stim.DetectorErrorModel):
    """
    Merge the error instructions of `dem` by symptom (detectors, observables).

    Returns `(keys, priors, decomp)`: the symptom of every merged fault, its
    probability (independent faults combined as p = p1 (1 - p2) + p2 (1 - p1),
    as in `TwoLevelDecoder`), and its shortest graphlike decomposition in
    `dem` (one component per symptom when `dem` is not decomposed).
    """
    cols: Dict[Key, int] = {}
    priors: List[float] = []
    decomp: List[List[Key]] = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        comps: List[List[set]] = [[set(), set()]]
        for t in inst.targets_copy():
            if t.is_separator():
                comps.append([set(), set()])
            elif t.is_relative_detector_id():
                comps[-1][0] ^= {t.val}
            elif t.is_logical_observable_id():
                comps[-1][1] ^= {t.val}
        tot_d: set = set()
        tot_o: set = set()
        for dd, oo in comps:
            tot_d ^= dd
            tot_o ^= oo
        key = (tuple(sorted(tot_d)), tuple(sorted(tot_o)))
        parts = [(tuple(sorted(dd)), tuple(sorted(oo))) for dd, oo in comps]
        j = cols.get(key)
        if j is None:
            j = cols[key] = len(priors)
            priors.append(0.0)
            decomp.append(parts)
        elif len(parts) < len(decomp[j]):
            decomp[j] = parts
        priors[j] = priors[j] * (1 - p) + p * (1 - priors[j])
    return list(cols), np.array(priors), decomp


def _incidence(rows: List[List[int]], n_rows: int) -> sp.csr_matrix:
    """Binary matrix with a one at (r, j) for every r in rows[j] (XOR of repeats)."""
    rr, cc = [], []
    for j, rs in enumerate(rows):
        for r in rs:
            rr.append(r)
            cc.append(j)
    m = sp.csr_matrix((np.ones(len(rr), dtype=np.uint8), (rr, cc)),
                      shape=(n_rows, len(rows)))
    m.data %= 2
    m.eliminate_zeros()
    return m


def _weights(p: np.ndarray) -> np.ndarray:
    """Matching weights log((1 - p) / p), with p clipped as in `TwoLevelDecoder`."""
    p = np.clip(p, 1e-12, 0.5 - 1e-9)
    return np.log((1 - p) / p)


class CompetitorDecoder:
    """
    Decoder of any detector-annotated circuit, by name (see `DECODERS`).

    `decode_batch(dets)` takes detection events of shape (shots, detectors)
    and returns predicted observable flips of shape (shots, observables) as
    bool, like `TwoLevelDecoder.decode_batch`; `estimate` samples the
    circuit and counts logical errors like `TwoLevelDecoder.estimate`.

    Attributes
    ----------
    circuit : stim.Circuit Decoded circuit. name : str Decoder name. opts :
    dict Options in use (defaults of `DECODERS[name]` and overrides).
    """

    def __init__(self, circuit: stim.Circuit, name: str = "mwpm", **opts) -> None:
        if name not in DECODERS:
            raise ValueError(f"decoder must be one of {tuple(DECODERS)}")
        unknown = set(opts) - set(DECODERS[name])
        if unknown:
            raise ValueError(f"unknown options for {name}: {sorted(unknown)}")
        self.circuit = circuit
        self.name = name
        self.opts = {**DECODERS[name], **opts}
        self.num_detectors = circuit.num_detectors
        self.num_observables = circuit.num_observables
        getattr(self, "_setup_" + name)()

    # ------------------------------------------------------------------ setup
    def _decomposed_dem(self) -> stim.DetectorErrorModel:
        return self.circuit.detector_error_model(
            decompose_errors=True, approximate_disjoint_errors=True)

    def _full_dem(self) -> stim.DetectorErrorModel:
        return self.circuit.detector_error_model(
            decompose_errors=False, approximate_disjoint_errors=True)

    def _setup_mwpm(self) -> None:
        import pymatching

        self.dem = self._decomposed_dem()
        self._matching = pymatching.Matching.from_detector_error_model(self.dem)

    def _setup_corr(self) -> None:
        import pymatching

        self.dem = self._decomposed_dem()
        self._matching = pymatching.Matching.from_detector_error_model(
            self.dem, enable_correlations=True)

    def _setup_bm(self) -> None:
        from ldpc import BpDecoder

        self.dem = self._decomposed_dem()
        keys, priors, decomp = merged_mechanisms(self.dem)
        self.mech_keys, self.priors = keys, priors
        edge_ids: Dict[Key, int] = {}
        edges_of: List[List[int]] = []
        for parts in decomp:
            eids = []
            for dd, oo in parts:
                if not dd:
                    continue  # an observable flip without detectors cannot be matched
                eids.append(edge_ids.setdefault((dd, oo), len(edge_ids)))
            edges_of.append(eids)
        self.edge_keys = list(edge_ids)
        nd, no = self.num_detectors, self.num_observables
        self.H = _incidence([list(k[0]) for k in keys], nd)
        # A: edges x faults (fault j lies on the edges of its decomposition)
        self.A = _incidence(edges_of, len(self.edge_keys)).astype(float)
        self.E = _incidence([list(k[0]) for k in self.edge_keys], nd).tocsc()
        self.F = _incidence([list(k[1]) for k in self.edge_keys], no).tocsc()
        self.static_p = self._edge_probs(priors)
        self._bp = BpDecoder(self.H.tocsc(), error_channel=list(priors),
                             max_iter=self.opts["bp_iters"],
                             bp_method=self.opts["bp_method"],
                             input_vector_type="syndrome")

    def _setup_bm_pkg(self) -> None:
        try:
            from beliefmatching import BeliefMatching
        except ImportError as exc:
            raise ImportError(
                "the beliefmatching package is not installed in this environment; "
                "'bm' is belief-matching with the settings of our own 'bm' decoder") from exc
        self.dem = self._decomposed_dem()
        self._bmp = BeliefMatching(self.dem, max_bp_iters=self.opts["bp_iters"])

    def _setup_bposd(self) -> None:
        self.dem = self._full_dem()
        keys, priors, _ = merged_mechanisms(self.dem)
        self.mech_keys, self.priors = keys, priors
        self.H = _incidence([list(k[0]) for k in keys], self.num_detectors).tocsc()
        self.L = _incidence([list(k[1]) for k in keys], self.num_observables).tocsr()
        o = self.opts
        if o["impl"] == "fast":
            if o["osd_method"] != "osd_cs" or o["bp_method"] != "product_sum":
                raise ValueError("impl='fast' implements product-sum BP with OSD-CS only.")
            from .fast_osd import FastBpOsd

            self._osd = FastBpOsd(self.H, priors, bp_iters=o["bp_iters"], order=o["osd_order"])
        elif o["impl"] == "ldpc":
            from ldpc import BpOsdDecoder

            self._osd = BpOsdDecoder(self.H, error_channel=list(priors),
                                     max_iter=o["bp_iters"], bp_method=o["bp_method"],
                                     osd_method=o["osd_method"],
                                     osd_order=o["osd_order"] if o["osd_method"] != "osd0" else 0,
                                     input_vector_type="syndrome")
        else:
            raise ValueError("impl must be 'ldpc' or 'fast'.")

    def _setup_tesseract(self) -> None:
        import tesseract_decoder.tesseract as tesseract

        self.dem = self._full_dem()
        cfg = tesseract.TesseractConfig(dem=self.dem, det_beam=self.opts["det_beam"],
                                        beam_climbing=self.opts["beam_climbing"])
        self._tess = cfg.compile_decoder()

    # ------------------------------------------------------------------ core
    def _edge_probs(self, q: np.ndarray) -> np.ndarray:
        """p_e = 1 - prod over the faults j on edge e of (1 - q_j)."""
        q = np.clip(q, 0.0, 1 - 1e-12)
        return -np.expm1(self.A @ np.log1p(-q))

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        """
        Predict observable flips, shape (shots, observables), for detection events.
        """
        dets = np.asarray(dets, dtype=np.uint8)
        if dets.ndim == 1:
            dets = dets[None, :]
        if self.name in ("mwpm", "corr"):
            pred = self._matching.decode_batch(dets, enable_correlations=self.name == "corr")
            return np.asarray(pred, dtype=bool).reshape(len(dets), self.num_observables)
        if self.name == "tesseract":
            return np.asarray(self._tess.decode_batch(dets.astype(bool)), dtype=bool)
        if self.name == "bm_pkg":
            return np.asarray(self._bmp.decode_batch(dets), dtype=bool)
        out = np.zeros((dets.shape[0], self.num_observables), dtype=bool)
        for i in np.flatnonzero(dets.any(axis=1)):  # a quiet shot predicts no flip
            out[i] = self._decode_shot(dets[i])
        return out

    def _decode_shot(self, syn: np.ndarray) -> np.ndarray:
        if self.name == "bposd":
            e = np.asarray(self._osd.decode(syn), dtype=np.int64)
            return (self.L @ e) % 2 == 1
        import pymatching

        self._bp.decode(syn)
        q = expit(-np.asarray(self._bp.log_prob_ratios, dtype=float))
        m = pymatching.Matching.from_check_matrix(
            self.E, weights=_weights(self._edge_probs(q)), faults_matrix=self.F,
            use_virtual_boundary_node=True)
        return np.asarray(m.decode(syn), dtype=bool)

    def estimate(self, max_shots: int = 100_000, max_errors: int = 500,
                 batch: int = 2000, seed: Optional[int] = None,
                 time_limit: float = 600.0) -> LogicalErrorEstimate:
        """
        Monte-Carlo logical error rate (a shot fails if any observable is wrong).
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


def competitor_decoder(circuit: stim.Circuit, name: str = "mwpm", **opts) -> CompetitorDecoder:
    """
    Build the decoder `name` (see `DECODERS`) from the detector error model of `circuit`.
    """
    return CompetitorDecoder(circuit, name, **opts)
