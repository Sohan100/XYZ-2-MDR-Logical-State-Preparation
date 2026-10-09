"""
split_hex.py
----------------------------------------------------------------------------
Depth-3 hybrid extraction of XYZ^2 with every hexagon split over two
ancillas joined in a cat state (a compilation ExtractionSchedule cannot
express).

Under hyb the links are pair measurements, so a data qubit has only three
gates per round (its three plaquettes), while one ancilla per hexagon needs
six layers: every data qubit idles in three of the six gate layers. Here a
plaquette of weight w > 3 uses two ancillas h1, h2: h1 is reset in |+>, h2
in |0>, a CX h1 -> h2 makes the cat state (+1 eigenstate of X1 X2), each
ancilla applies the controlled Paulis of its half of the plaquette, and both
are read out in the X basis; the product of the two outcomes is the
plaquette. Plaquettes of weight <= 3 keep one ancilla. A round is

    R (ancilla resets; link pair measurements, every data qubit busy)
    C (cat CX gates; data qubits idle)
    3 gate layers (bulk data qubits and ancillas busy in every layer)
    M (ancilla readout; data qubits idle)

so a bulk data qubit idles in 2 steps per round instead of 4 (s1525 under
hyb). The schedule is translation invariant: `tau[cls][pos]` in {0, 1, 2},
every layer used by exactly two positions of a hexagon, one of each half;
`half[cls]` is the set of positions on h1. Validity is the rule of
ExtractionSchedule (no qubit twice per layer, the interleaving parity rule
between overlapping plaquettes), checked on the patch; the cat makes the
pair of ancillas act as one X-type ancilla, so the rule is unchanged, and
Stim's detector check confirms it (a wrong schedule gives non-deterministic
detectors).

Noise (the rates of the model, as in FTMDRCircuit): reset flips p_prep (Z
for h1, X for h2), DEPOLARIZE2 p2 after each cat CX, gate noise and idling
as in FTMDRCircuit, Z flip p_meas before every MX, link pair measurements
with the EM3 channel p_mpp. Data idle during R only where no link covers
them (none in the patch), during C and M always.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from pathlib import Path
import sys
from typing import Dict, FrozenSet, List, Tuple

import stim

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from mdr.ft import FTMDRCircuit  # noqa: E402
from mdr.ft.extraction_schedule import ExtractionSchedule  # noqa: E402
from mdr.ft.ft_mdr_circuit import _CTRL, _RESET, _FLIP, _append_em3_mpp  # noqa: E402
from mdr.ft.xyz2_geometry import XYZ2Geometry  # noqa: E402

POS = XYZ2Geometry.POSITIONS


@dataclass(frozen=True)
class SplitSchedule(ExtractionSchedule):
    """
    `sigma` holds the layers tau (0..2); `half[cls]` the positions on h1.
    `link` is unused (links are pair measurements in the reset step).
    """

    half: Dict[str, FrozenSet[str]] = None

    def validate(self, geometry):
        sm = self.slot_map(geometry)
        keep = [ci for ci, ch in enumerate(geometry.checks) if ch["kind"] != "link"]
        per_qubit: Dict[int, List[int]] = {}
        for ci in keep:
            for _, q in geometry.checks[ci]["terms"]:
                per_qubit.setdefault(q, []).append(sm[(ci, q)])
        for q, ts in per_qubit.items():
            if len(ts) != len(set(ts)):
                return False, f"data qubit {q} used twice in one layer"
        paulis = {ci: {q: p for p, q in geometry.checks[ci]["terms"]} for ci in keep}
        for ci in keep:
            for anc in split_ancillas(geometry, self, ci):
                ts = [sm[(ci, q)] for q in anc]
                if len(ts) != len(set(ts)):
                    return False, f"ancilla of check {ci} used twice in one layer"
        for x, a in enumerate(keep):
            for b in keep[x + 1:]:
                anti = [q for q in set(paulis[a]) & set(paulis[b]) if paulis[a][q] != paulis[b][q]]
                if sum(1 for q in anti if sm[(a, q)] < sm[(b, q)]) % 2:
                    return False, f"checks {a} and {b} violate interleaving"
        return True, "ok"


def split_ancillas(geometry, sched: SplitSchedule, ci: int) -> List[List[int]]:
    """Data qubits of each ancilla of plaquette ci (one list if weight <= 3)."""
    ch = geometry.checks[ci]
    i, j = ch["cell"]
    cls = "B" if (i + j) % 2 == 0 else "A"
    qs = [q for _, q in ch["terms"]]
    if len(qs) <= 3:
        return [qs]
    h1 = [q for q in qs if ch["positions"][q] in sched.half[cls]]
    h2 = [q for q in qs if ch["positions"][q] not in sched.half[cls]]
    return [h for h in (h1, h2) if h]


def enumerate_split(check_d: int = 5) -> List[SplitSchedule]:
    """
    Every translation-invariant depth-3 split schedule valid on the patch of
    distance `check_d` (tau pairs times the four h1/h2 splits per class,
    up to exchanging h1 and h2).
    """
    geo = XYZ2Geometry(check_d)
    taus = []
    for t in itertools.product(range(3), repeat=6):
        if all(t.count(k) == 2 for k in range(3)):
            taus.append(dict(zip(POS, t)))
    out = []
    for ta in taus:
        for tb in taus:
            # the halves take one position per layer by construction, so
            # validity does not depend on them
            s = SplitSchedule(3, {"A": ta, "B": tb}, {0: (0, 1), 1: (0, 1)},
                              {"A": _halves(ta)[0], "B": _halves(tb)[0]})
            if not s.validate(geo)[0]:
                continue
            for ha in _halves(ta):
                for hb in _halves(tb):
                    out.append(SplitSchedule(3, {"A": ta, "B": tb}, {0: (0, 1), 1: (0, 1)},
                                             {"A": ha, "B": hb}))
    return out


def _halves(tau) -> List[FrozenSet[str]]:
    by = {k: [p for p in POS if tau[p] == k] for k in range(3)}
    out = []
    for pick in itertools.product((0, 1), repeat=2):
        h1 = {by[0][0], by[1][pick[0]], by[2][pick[1]]}
        out.append(frozenset(h1))
    return out


class SplitHexCircuit(FTMDRCircuit):
    """
    FTMDRCircuit (hybrid noise, link_reps = 1) with the split-hexagon round.
    Check outcomes are lists of measurement records (two for a split
    plaquette); `_detector` flattens them, so the detector and observable
    code of FTMDRCircuit is reused unchanged.
    """

    def __init__(self, d, rounds, noise, schedule: SplitSchedule, logical="X",
                 detectors="combined"):
        super().__init__(d, rounds, noise, schedule=schedule, final="frame",
                         detectors=detectors, logical=logical)
        if self.native != "hybrid" or self.link_reps != 1 or not self.link_noise:
            raise ValueError("SplitHexCircuit needs hyb noise with link_reps = 1")
        geo = self.geometry
        n = geo.n
        nxt = n
        self.anc_of: Dict[int, List[Tuple[int, List[int]]]] = {}
        for ci, ch in enumerate(geo.checks):
            if ch["kind"] == "link":
                continue
            self.anc_of[ci] = []
            for qs in split_ancillas(geo, schedule, ci):
                self.anc_of[ci].append((nxt, qs))
                nxt += 1
        self.ancillas = [a for ci in self.anc_of for a, _ in self.anc_of[ci]]
        self.flip_of = {ci: nxt + k for k, ci in enumerate(self.links)}
        self._args = (d, rounds, noise, schedule, logical)

    def s0_copy(self):
        d, r, nz, s, lg = self._args
        return SplitHexCircuit(d, r, nz, s, lg, detectors="s0")

    def physical_qubits(self) -> int:
        return self.geometry.n + len(self.ancillas)

    def _detector(self, c, idx, count, coords):
        flat: List[int] = []
        for x in idx:
            flat.extend(x if isinstance(x, list) else [x])
        acc = set()
        for x in flat:
            acc ^= {x}
        c.append("DETECTOR", [stim.target_rec(i - count) for i in sorted(acc)],
                 list(coords))

    def build(self) -> stim.Circuit:
        geo, nz = self.geometry, self.noise
        n, m = geo.n, len(geo.checks)
        data = list(range(n))
        all_q = data + list(self.ancillas)
        h1s = [lst[0][0] for lst in self.anc_of.values()]
        h2s = [lst[1][0] for lst in self.anc_of.values() if len(lst) > 1]
        cats = [(lst[0][0], lst[1][0]) for lst in self.anc_of.values() if len(lst) > 1]
        pauli = {ci: {q: p for p, q in geo.checks[ci]["terms"]} for ci in range(m)}
        layers: List[List[Tuple[int, int, str]]] = [[] for _ in range(3)]
        for ci, lst in self.anc_of.items():
            for a, qs in lst:
                for q in qs:
                    layers[self.slots[(ci, q)]].append((a, q, pauli[ci][q]))
        c = stim.Circuit()
        for q in data:
            c.append("QUBIT_COORDS", [q], [q, 0])
        for k, a in enumerate(self.ancillas):
            c.append("QUBIT_COORDS", [a], [k, 1])
        for b in "XYZ":
            qs = [q for q in data if self.init_basis[q] == b]
            if qs:
                c.append(_RESET[b], qs)
                if nz.p_prep:
                    c.append(_FLIP[b], qs, nz.p_prep)
        c.append("TICK")
        rec: Dict[Tuple[int, int], object] = {}
        count = 0
        init_rows = self.frame.s0_rows
        order = [ci for ci in self.anc_of]
        for r in range(self.rounds):
            # R: ancilla resets, link pair measurements
            c.append("RX", h1s)
            if h2s:
                c.append("R", h2s)
            if nz.p_prep:
                c.append("Z_ERROR", h1s, nz.p_prep)
                if h2s:
                    c.append("X_ERROR", h2s, nz.p_prep)
            used = set()
            for ci in self.links:
                (pu, u), (pl, lo) = geo.checks[ci]["terms"]
                _append_em3_mpp(c, u, pu, lo, pl, self.flip_of[ci], nz.p_mpp)
                rec[(r, ci)] = count
                count += 1
                used.update((u, lo))
            self._idle_mr(c, [q for q in data if q not in used])
            c.append("TICK")
            # C: cat gates
            if cats:
                tg = [x for pair in cats for x in pair]
                c.append("CX", tg)
                if nz.p2:
                    c.append("DEPOLARIZE2", tg, nz.p2)
                busy = set(tg)
                self._idle(c, [q for q in all_q if q not in busy])
            c.append("TICK")
            for layer in layers:
                pairs: List[int] = []
                for gate in ("CX", "CY", "CZ"):
                    tg = []
                    for a, q, p in sorted(layer):
                        if _CTRL[p] == gate:
                            tg += [a, q]
                    if tg:
                        c.append(gate, tg)
                        pairs += tg
                if pairs and nz.p2:
                    c.append("DEPOLARIZE2", pairs, nz.p2)
                busy = set(pairs)
                self._idle(c, [q for q in all_q if q not in busy])
                c.append("TICK")
            # M: readout of every ancilla
            meas = [a for ci in order for a, _ in self.anc_of[ci]]
            if nz.p_meas:
                c.append("Z_ERROR", meas, nz.p_meas)
            c.append("MX", meas)
            k = 0
            for ci in order:
                rec[(r, ci)] = [count + k + j for j in range(len(self.anc_of[ci]))]
                k += len(self.anc_of[ci])
            count += k
            self._idle_mr(c, data)
            self._round_detectors(c, r, rec, {}, count, init_rows)
            c.append("TICK")
        return self._final_readout(c, rec, {}, count)


class PipelinedSplitHexCircuit(SplitHexCircuit):
    """
    SplitHexCircuit with two ancilla banks that alternate between rounds.

    The reset and cat CX of the bank of round r + 1 run during gate layers 1
    and 2 of round r (ancilla-only operations, no data qubit involved), and
    the links are measured by their pair measurements in the readout step M,
    after every plaquette gate on their block (a valid order). A round is
    then four steps, L0 L1 L2 M, and a bulk data qubit is busy in all four
    (three plaquette gates and one pair measurement): no data idling in the
    bulk. Round 0 is preceded by one reset step and one cat step for bank 0
    (data idle there, like the reset step of FTMDRCircuit). Ancillas get idle
    noise whenever they hold a live state and have no operation (the fresh
    cat of the next bank during M). Qubit count: 2 x (ancillas of
    SplitHexCircuit).
    """

    def __init__(self, d, rounds, noise, schedule, logical="X", detectors="combined"):
        super().__init__(d, rounds, noise, schedule, logical, detectors)
        off = len(self.ancillas)
        base = min(self.ancillas)
        # bank 1 ancillas follow the flip qubits
        top = max(list(self.flip_of.values()) + self.ancillas) + 1
        self.bank_shift = top - base
        self.ancillas_all = self.ancillas + [a + self.bank_shift for a in self.ancillas]
        self._off = off

    def s0_copy(self):
        d, r, nz, s, lg = self._args
        return PipelinedSplitHexCircuit(d, r, nz, s, lg, detectors="s0")

    def physical_qubits(self) -> int:
        return self.geometry.n + len(self.ancillas_all)

    def build(self) -> stim.Circuit:
        geo, nz = self.geometry, self.noise
        n, m = geo.n, len(geo.checks)
        data = list(range(n))
        sh = self.bank_shift

        def bank(x, b):
            return x + sh * b

        h1s = [lst[0][0] for lst in self.anc_of.values()]
        h2s = [lst[1][0] for lst in self.anc_of.values() if len(lst) > 1]
        cats = [(lst[0][0], lst[1][0]) for lst in self.anc_of.values() if len(lst) > 1]
        pauli = {ci: {q: p for p, q in geo.checks[ci]["terms"]} for ci in range(m)}
        layers: List[List[Tuple[int, int, str]]] = [[] for _ in range(3)]
        for ci, lst in self.anc_of.items():
            for a, qs in lst:
                for q in qs:
                    layers[self.slots[(ci, q)]].append((a, q, pauli[ci][q]))
        c = stim.Circuit()
        for q in data:
            c.append("QUBIT_COORDS", [q], [q, 0])
        for k, a in enumerate(self.ancillas_all):
            c.append("QUBIT_COORDS", [a], [k, 1])
        for b in "XYZ":
            qs = [q for q in data if self.init_basis[q] == b]
            if qs:
                c.append(_RESET[b], qs)
                if nz.p_prep:
                    c.append(_FLIP[b], qs, nz.p_prep)
        c.append("TICK")

        def reset(b):
            c.append("RX", [bank(a, b) for a in h1s])
            if h2s:
                c.append("R", [bank(a, b) for a in h2s])
            if nz.p_prep:
                c.append("Z_ERROR", [bank(a, b) for a in h1s], nz.p_prep)
                if h2s:
                    c.append("X_ERROR", [bank(a, b) for a in h2s], nz.p_prep)
            return {bank(a, b) for a in h1s + h2s}

        def cat(b):
            tg = [bank(x, b) for pair in cats for x in pair]
            if tg:
                c.append("CX", tg)
                if nz.p2:
                    c.append("DEPOLARIZE2", tg, nz.p2)
            return set(tg)

        # bank 0 before round 0: reset step and cat step, data idle
        live = reset(0)
        self._idle_mr(c, data)
        c.append("TICK")
        busy = cat(0)
        self._idle(c, data + [a for a in live if a not in busy])
        c.append("TICK")
        rec: Dict[Tuple[int, int], object] = {}
        count = 0
        order = list(self.anc_of)
        for r in range(self.rounds):
            b = r % 2
            nb = 1 - b
            more = r < self.rounds - 1
            for t, layer in enumerate(layers):
                pairs: List[int] = []
                for gate in ("CX", "CY", "CZ"):
                    tg = []
                    for a, q, p in sorted(layer):
                        if _CTRL[p] == gate:
                            tg += [bank(a, b), q]
                    if tg:
                        c.append(gate, tg)
                        pairs += tg
                if pairs and nz.p2:
                    c.append("DEPOLARIZE2", pairs, nz.p2)
                busy = set(pairs)
                if more and t == 1:
                    live |= reset(nb)
                    busy |= {bank(a, nb) for a in h1s + h2s}
                if more and t == 2:
                    busy |= cat(nb)
                self._idle(c, [q for q in data + sorted(live) if q not in busy])
                c.append("TICK")
            # M: readout of bank b, link pair measurements, next bank idles
            meas = [bank(a, b) for ci in order for a, _ in self.anc_of[ci]]
            if nz.p_meas:
                c.append("Z_ERROR", meas, nz.p_meas)
            c.append("MX", meas)
            live -= set(meas)
            k = 0
            for ci in order:
                rec[(r, ci)] = [count + k + j for j in range(len(self.anc_of[ci]))]
                k += len(self.anc_of[ci])
            count += k
            used = set()
            for ci in self.links:
                (pu, u), (pl, lo) = geo.checks[ci]["terms"]
                _append_em3_mpp(c, u, pu, lo, pl, self.flip_of[ci], nz.p_mpp)
                rec[(r, ci)] = count
                count += 1
                used.update((u, lo))
            self._idle_mr(c, [q for q in data if q not in used])
            self._idle(c, sorted(live))
            self._round_detectors(c, r, rec, {}, count, self.frame.s0_rows)
            c.append("TICK")
        return self._final_readout(c, rec, {}, count)
