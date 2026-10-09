"""
phen_paper.py
----------------------------------------------------------------------------
Conventions of Srivastava et al., arXiv:2505.03691 (phenomenological noise),
added on top of the repository's XYZ^2 circuits without editing src/mdr/ft.

- `PaperPhenCircuit`: the phenomenological XYZ^2 memory of the paper. The code
  starts in a code state with every check known (a noiseless round 0 after the
  frame preparation), then `noisy_rounds` rounds of data noise + noisy
  measurement of every check, then one noiseless round of every check and of
  the logical operator (`final="ideal"`). Every link therefore has detectors at
  both time boundaries, as in the paper's time-direction link decoding.
- `SequentialHardDecoder`: the paper's sequential decoder. Every XX link is
  decoded alone in the time direction (1D matching: a pair of events at
  separation r is two data flips if p_f^2 > q^r, else r measurement errors).
  Each round, the hard link decisions give conditional probabilities of every
  single-qubit data fault (the [[2,1,1]] code-capacity step); these replace the
  priors of the upper (S_0) matching graph of `TwoLevelDecoder`, which is then
  matched once over all rounds.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import stim

import mdr.ft.two_level_decoder as tld
from mdr.ft import FTMDRCircuit, TwoLevelDecoder
from mdr.ft.ft_mdr_circuit import _RESET


class PaperPhenCircuit(FTMDRCircuit):
    """
    `FTMDRCircuit` under phenomenological noise whose round 0 is noiseless
    (perfect initial syndrome, code-state start). Build it with
    `rounds = noisy_rounds + 1` and `final="ideal"` (see `paper_circuit`).
    """

    def _build_phen(self) -> stim.Circuit:
        geo, nz = self.geometry, self.noise
        n = geo.n
        data = list(range(n))
        c = stim.Circuit()
        for q in data:
            c.append("QUBIT_COORDS", [q], [q, 0])
        for b in "XYZ":
            qs = [q for q in data if self.init_basis[q] == b]
            if qs:
                c.append(_RESET[b], qs)
        c.append("TICK")
        init_rows = self.frame.s0_rows
        xyz = tuple(nz.data_xyz)
        rec: Dict[Tuple[int, int], int] = {}
        reps: Dict[Tuple[int, int], List[int]] = {}
        count = 0
        for r in range(self.rounds):
            if r > 0 and any(xyz):
                c.append("PAULI_CHANNEL_1", data, list(xyz))
            c.append("TICK")
            for ci, ch in enumerate(geo.checks):
                flip = nz.p_meas if r > 0 else 0.0
                if flip:
                    c.append("MPP", self._mpp(ch["spec"]), flip)
                else:
                    c.append("MPP", self._mpp(ch["spec"]))
                rec[(r, ci)] = count
                if ci in self.links:
                    reps[(r, ci)] = [count]
                count += 1
            self._round_detectors(c, r, rec, reps, count, init_rows)
            c.append("TICK")
        return self._final_readout(c, rec, reps, count, readout_noise=False)


def paper_circuit(d: int, noise, logical: str, schedule) -> PaperPhenCircuit:
    return PaperPhenCircuit(d, d + 1, noise, final="ideal", detectors="combined",
                            schedule=schedule, logical=logical)


def two_level(ft: FTMDRCircuit, **opts) -> TwoLevelDecoder:
    """TwoLevelDecoder whose internal S_0 circuit is built with the class of `ft`."""
    saved = tld.FTMDRCircuit
    tld.FTMDRCircuit = type(ft)
    try:
        return TwoLevelDecoder(ft, **opts)
    finally:
        tld.FTMDRCircuit = saved


class SequentialHardDecoder:
    """
    Paper-style sequential decoder on a `PaperPhenCircuit` (see module doc).
    """

    def __init__(self, ft: PaperPhenCircuit) -> None:
        if not isinstance(ft, PaperPhenCircuit):
            raise TypeError("needs a PaperPhenCircuit")
        self.ft = ft
        self.base = two_level(ft, mode="mwpm")
        b = self.base
        circ = b.circuit
        px, py, pz = ft.noise.data_xyz
        self.q = ft.noise.p_meas
        self.a = py + pz                                # one qubit anticommutes with XX
        self.pf = 2 * self.a * (1 - self.a)             # link flipped by one round of data noise
        self.T = ft.rounds - 1                          # noisy rounds 1..T, final detectors at T + 1
        geo = ft.geometry
        self.links = list(ft.links)
        link_of_q = {}
        for li, ci in enumerate(self.links):
            for _, q in geo.checks[ci]["terms"]:
                link_of_q[q] = li
        # detector -> (link index, time) for every link detector
        coords = circ.get_detector_coordinates()
        rows = ft.frame.s0_rows
        self.det_link = {}
        for i, cc in coords.items():
            k, t, ph = int(cc[0]), int(cc[1]), int(cc[2])
            ci = None
            if ph in (0, 1, 2):
                nz = np.flatnonzero(rows[k])
                if len(nz) == 1 and geo.checks[int(nz[0])]["kind"] == "link":
                    ci = int(nz[0])
            elif ph in (3, 4) and geo.checks[k]["kind"] == "link":
                ci = k
            if ci is not None and t >= 1:
                self.det_link[i] = (self.links.index(ci), t)
        self.det_ids = np.array(sorted(self.det_link), dtype=np.int64)
        # identify every merged mechanism of the full DEM by its circuit faults
        m = len(geo.checks)
        key_index = {k: j for j, k in enumerate(b.mech_keys)}
        # per mechanism: list of components (kind, link, t, prior-type)
        self.comp: List[List[tuple]] = [[] for _ in b.mech_keys]
        expl = circ.explain_detector_error_model_errors(reduce_to_one_representative_error=False)
        noise_ticks = sorted({loc.tick_offset for e in expl for loc in e.circuit_error_locations
                              if loc.flipped_measurement is None})
        round_of_tick = {t: r + 1 for r, t in enumerate(noise_ticks)}
        pxyz = {"X": px, "Y": py, "Z": pz}
        for e in expl:
            dets, obs = set(), set()
            for t in e.dem_error_terms:
                if t.dem_target.is_relative_detector_id():
                    dets.add(t.dem_target.val)
                elif t.dem_target.is_logical_observable_id():
                    obs.add(t.dem_target.val)
            j = key_index.get((tuple(sorted(dets)), tuple(sorted(obs))))
            if j is None:
                continue
            for loc in e.circuit_error_locations:
                if loc.flipped_measurement is not None and not loc.flipped_pauli_product:
                    idx = loc.flipped_measurement.record_index
                    r, ci = divmod(idx, m)
                    if ci in self.links:
                        self.comp[j].append(("meas_link", self.links.index(ci), r, self.q))
                    else:
                        self.comp[j].append(("meas", -1, r, self.q))
                else:
                    pp = loc.flipped_pauli_product
                    if len(pp) != 1:
                        raise RuntimeError("unexpected multi-qubit data fault")
                    g = pp[0].gate_target
                    pa = "X" if g.is_x_target else ("Y" if g.is_y_target else "Z")
                    r = round_of_tick[loc.tick_offset]
                    kind = "data_x" if pa == "X" else "data_flip"
                    self.comp[j].append((kind, link_of_q[g.value], r, pxyz[pa]))
        missing = [j for j, cl in enumerate(self.comp) if not cl]
        if missing:
            raise RuntimeError(f"{len(missing)} mechanisms not identified")
        self.w_d = -np.log(self.pf)
        self.w_q = -np.log(self.q)

    def _time_decode(self, events: List[int]) -> Tuple[set, set]:
        """1D matching of one link's events (times 1..T+1); returns (data flip times, meas error times)."""
        T = self.T
        ev = sorted(events)
        k = len(ev)
        inf = float("inf")
        f = [0.0] + [inf] * k
        choice = [None] * (k + 1)
        for i in range(1, k + 1):
            t = ev[i - 1]
            # a data flip at min(t, T), joined to t by measurement errors (only the event at T + 1)
            cd = f[i - 1] + self.w_d + max(0, t - T) * self.w_q
            if cd < f[i]:
                f[i], choice[i] = cd, "d"
            if i >= 2:
                c2 = f[i - 2] + (t - ev[i - 2]) * self.w_q
                if c2 < f[i]:
                    f[i], choice[i] = c2, "m"
        data, meas = set(), set()
        i = k
        while i > 0:
            if choice[i] == "d":
                data.add(min(ev[i - 1], T))
                meas |= set(range(T, ev[i - 1]))
                i -= 1
            else:
                for t in range(ev[i - 2], ev[i - 1]):
                    meas.add(t)
                i -= 2
        return data, meas

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        import pymatching

        b = self.base
        a = self.a
        out = np.zeros((dets.shape[0], b.circuit.num_observables), dtype=bool)
        for s in range(dets.shape[0]):
            row = dets[s]
            ev: Dict[int, List[int]] = {}
            for i in self.det_ids[row[self.det_ids] != 0]:
                li, t = self.det_link[int(i)]
                ev.setdefault(li, []).append(t)
            flags, mflags = set(), set()
            for li, ts in ev.items():
                dd, mm = self._time_decode(ts)
                flags |= {(li, t) for t in dd}
                mflags |= {(li, t) for t in mm}
            q = np.empty(len(self.comp))
            for j, cl in enumerate(self.comp):
                keep = 1.0
                for kind, li, t, pe in cl:
                    if kind == "meas":
                        x = pe
                    elif kind == "meas_link":
                        x = 0.5 if (li, t) in mflags else 1e-9
                    elif kind == "data_flip":
                        x = pe / (2 * a) if (li, t) in flags else pe * a / (1 - 2 * a + 2 * a * a)
                    else:  # X fault: commutes with the link
                        x = pe / (2 * (1 - a)) if (li, t) in flags else pe * (1 - a) / (1 - 2 * a * (1 - a))
                    keep *= 1 - x
                q[j] = 1 - keep
            p_e = b._edge_probs(q)
            mt = pymatching.Matching.from_check_matrix(
                b.E, weights=b._weights(p_e), faults_matrix=b.F, use_virtual_boundary_node=True)
            out[s] = mt.decode(row[b.s0_cols])
        return out
