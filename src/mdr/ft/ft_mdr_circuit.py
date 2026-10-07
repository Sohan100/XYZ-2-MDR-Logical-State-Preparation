"""

ft_mdr_circuit.py
----------------------------------------------------------------------------
Fault-tolerant Measure-Decode-Recover preparation of the XYZ^2 plus state.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import stim

from .circuit_noise import CircuitNoise
from .extraction_schedule import DEPTH6_SCHEDULE, ExtractionSchedule
from .frame_basis import XYZ2FrameBasis
from .xyz2_geometry import XYZ2Geometry

_CTRL = {"X": "CX", "Y": "CY", "Z": "CZ"}
_RESET = {"X": "RX", "Y": "RY", "Z": "R"}
_MEAS = {"X": "MX", "Y": "MY", "Z": "M"}
_FLIP = {"X": "Z_ERROR", "Y": "X_ERROR", "Z": "X_ERROR"}
_PAULI2 = [a + b for a in "IXYZ" for b in "IXYZ"][1:]
# a controlled Pauli is the native RZZ gate conjugated by a basis change on
# the target, which maps a target Pauli of the RZZ frame to the gate frame
_TARGET_MAP = {"CZ": {"I": "I", "X": "X", "Y": "Y", "Z": "Z"},
               "CX": {"I": "I", "X": "Z", "Y": "Y", "Z": "X"},
               "CY": {"I": "I", "X": "Z", "Y": "X", "Z": "Y"}}
# the 31 non-trivial elements of {I,X,Y,Z}^2 x {no flip, flip} of the EM3 channel
_EM3_COMBOS = [(a, b, f) for a in "IXYZ" for b in "IXYZ" for f in (0, 1)][1:]
NATIVES = ("gates", "pairs", "hybrid", "phen")
# detector phase of the comparisons of repeated link measurements (lower level)
REP_PHASE = 5


def _to_gate_frame(probs, gate: str) -> List[float]:
    """
    Map PAULI_CHANNEL_2 probabilities from the RZZ frame to a controlled Pauli.
    """
    out = dict.fromkeys(_PAULI2, 0.0)
    for pq, pr in zip(_PAULI2, probs):
        out[pq[0] + _TARGET_MAP[gate][pq[1]]] += pr
    return [out[k] for k in _PAULI2]


def _append_em3_mpp(c: stim.Circuit, q1: int, p1: str, q2: int, p2: str,
                    f: int, p: float) -> None:
    """
    Pair measurement P1 P2 with the EM3 error of Gidney et al.

    With probability p the MPP is followed by an element of
    {I,X,Y,Z}^2 x {flip, no flip} chosen uniformly. The flip stays correlated
    with the Pauli error through the auxiliary qubit f, reset in |0> and
    included in the MPP as Z_f (f is a bookkeeping device, not a physical
    qubit, and gets no other noise).
    """
    c.append("R", [f])
    if p:
        acc = 0.0
        for j, (e1, e2, ff) in enumerate(_EM3_COMBOS):
            tg = []
            if e1 != "I":
                tg.append(stim.target_pauli(q1, e1))
            if e2 != "I":
                tg.append(stim.target_pauli(q2, e2))
            if ff:
                tg.append(stim.target_x(f))
            c.append("E" if j == 0 else "ELSE_CORRELATED_ERROR", tg,
                     (p / 32.0) / (1.0 - acc))
            acc += p / 32.0
    c.append("MPP", [stim.target_pauli(q1, p1), stim.target_combiner(),
                     stim.target_pauli(q2, p2), stim.target_combiner(),
                     stim.target_z(f)])


def link_rep_slots(schedule: ExtractionSchedule, reps: int
                   ) -> Dict[int, List[Tuple[int, int]]]:
    """
    Gate layers (upper, lower) of every repetition of an XX link, per parity.

    Repetition 0 is the link of `schedule`. Every further repetition uses its
    own ancilla and two more CX gates, placed in the first layers (earliest
    last layer, then earliest sum) where both vertices of the block are free
    and where the order of the two gates agrees with every plaquette that
    anticommutes with XX on the block (both before or both after its two
    gates on the block), so the plaquettes are still measured exactly. Layers
    beyond `schedule.depth` are added when no such pair exists inside it.
    The bulk slots of a parity also serve the boundary blocks, which have
    fewer plaquettes.
    """
    a, b = schedule.sigma["A"], schedule.sigma["B"]
    # (upper slots, lower slots, anticommuting (upper, lower) pairs) per parity
    bulk = {
        0: ({b["BR"], a["TRu"], a["BLu"]}, {b["TL"], a["TRl"], a["BLl"]},
            [(a["TRu"], a["TRl"]), (a["BLu"], a["BLl"])]),
        1: ({a["BR"], b["TRu"], b["BLu"]}, {a["TL"], b["TRl"], b["BLl"]},
            [(b["TRu"], b["TRl"]), (b["BLu"], b["BLl"])]),
    }
    out: Dict[int, List[Tuple[int, int]]] = {}
    for par, (up, lo, anti) in bulk.items():
        up, lo = set(up), set(lo)
        first = tuple(schedule.link[par])
        up.add(first[0])
        lo.add(first[1])
        slots = [first]
        horizon = schedule.depth + 2 * reps
        for _ in range(1, reps):
            best = None
            for tu in range(horizon):
                for tl in range(horizon):
                    if tu == tl or tu in up or tl in lo:
                        continue
                    if any((tu < x) != (tl < y) for x, y in anti):
                        continue
                    key = (max(tu, tl), tu + tl, tu)
                    if best is None or key < best[0]:
                        best = (key, (tu, tl))
            tu, tl = best[1]
            up.add(tu)
            lo.add(tl)
            slots.append((tu, tl))
        out[par] = slots
    return out


def hybrid_link_steps(schedule: ExtractionSchedule, reps: int
                      ) -> Dict[int, List[object]]:
    """
    Steps of the pair measurements of an XX link in the hybrid extraction.

    A step is "R" (the step in which the plaquette ancillas are reset, before
    the first gate layer), a gate layer t, or "M" (the step in which they are
    measured, after the last layer). Repetition 0 is in "R", so the link is
    measured before every plaquette gate on its block, as the controlled-Pauli
    link of `schedule` is. Repetition 1 goes into the gate layer in which both
    vertices of the block are free of plaquette gates and which does not fall
    between the two gates of an anticommuting plaquette (for DEPTH6_SCHEDULE:
    layer 5 for even blocks, layer 3 for odd blocks), or into "M" when the
    schedule has no such layer; the last repetition of three goes into "M".
    None of them adds a step to the round.
    """
    if reps > 3:
        raise NotImplementedError(
            "the hybrid extraction measures a link at most three times per "
            "round (reset step, a free gate layer, measurement step).")
    a, b = schedule.sigma["A"], schedule.sigma["B"]
    bulk = {
        0: ({b["BR"], a["TRu"], a["BLu"]}, {b["TL"], a["TRl"], a["BLl"]},
            [(a["TRu"], a["TRl"]), (a["BLu"], a["BLl"])]),
        1: ({a["BR"], b["TRu"], b["BLu"]}, {a["TL"], b["TRl"], b["BLl"]},
            [(b["TRu"], b["TRl"]), (b["BLu"], b["BLl"])]),
    }
    out: Dict[int, List[object]] = {}
    for par, (up, lo, anti) in bulk.items():
        free = [t for t in range(schedule.depth) if t not in up and t not in lo
                and all((t < x) == (t < y) for x, y in anti)]
        if reps == 3 and not free:
            raise NotImplementedError(
                "no gate layer of this schedule leaves both vertices of a "
                "block free for a third pair measurement.")
        if free:
            mid = min(free, key=lambda t: (abs(2 * t + 1 - schedule.depth), t))
            out[par] = ["R", mid, "M"][:reps]
        else:
            out[par] = ["R", "M"][:reps]
    return out


class FTMDRCircuit:
    """
    Stim circuit for frame-initialised, spacetime-decoded MDR.

    Protocol:

    1. Prepare every data qubit in its frame eigenstate (`XYZ2FrameBasis`).
       Logical X and `d^2` stabilizer products start deterministic.
    2. Run `rounds` rounds of interleaved extraction of all `2 d^2 - 1`
       checks, one ancilla per check, `schedule.depth` entangling layers.
    3. Decode the detector history (all rounds at once) with minimum-weight
       matching on the frame-deterministic detectors.
    4. Recover by updating the Pauli frame: the decoder output fixes the
       Logical X sign and the measured random checks fix the toggles.

    The circuit ends either with a destructive readout in the frame basis
    (`final="frame"`, possible on hardware) or with one noiseless round of
    stabilizer and Logical X measurements (`final="ideal"`, which scores the
    prepared state after ideal error correction). `init="plus"` reproduces
    the old `|+>^n` start for comparison. `share_ancillas=k` lets k link
    ancillas, which finish after layer 1, be measured, reset, and reused for
    k boundary checks that start at layer 2 or later (d = 5 then fits on the
    98-qubit Helios with 95 qubits). `detectors="s0"` declares only the
    frame-deterministic stabilizer products (graph-like detector model for
    matching), `detectors="all"` declares every generator comparison.

    `logical="Y"` runs the same extraction as a memory of the conjugate
    logical operator: the frame of `XYZ2FrameBasis(logical="Y")` is prepared
    and read out, and the observable is a frame representative of Logical Y.
    With DEPTH6_SCHEDULE that memory has fault distance only (d + 1) / 2;
    use BOTH_BASES_SCHEDULE (distance >= d in both bases) for a fair memory.

    Extraction variants (`noise.native`, see `CircuitNoise`):

    - "gates": one ancilla per check, `schedule.depth` layers of controlled
      Paulis between one reset step and one measurement step per round.
    - "pairs" (EM3): `_build_pairs`.
    - "hybrid": every XX link is one pair measurement with the EM3 channel
      `p_mpp`, exactly as in `_build_pairs`; plaquettes and boundary checks
      use ancillas and the gate layers of `schedule` with the gate noise of
      the model. The link measurements take no step of their own: they run
      in the step that resets the plaquette ancillas (`hybrid_link_steps`),
      where the data qubits would otherwise idle. A qubit not used in a step
      idles with the noise of the model, as with gates.
    - "phen": phenomenological noise (`_build_phen`).

    `noise.link_reps = k` measures every link k times per round:

    - gates: repetition j > 0 uses its own ancilla and two CX gates in the
      layers of `link_rep_slots` (DEPTH6_SCHEDULE: even blocks (4, 5) and
      (5, 6), odd blocks (2, 3) and (3, 4); k = 2 adds no layer, k = 3 adds
      layer 6). All link ancillas are reset and measured with the others.
      The outcomes of a round are ordered by the layers of their gates.
    - hybrid (k <= 3): reset step, free gate layer (or the measurement
      step when the schedule has none), measurement step; no step is added.
    - phen: k noisy pair measurements.

    The frame-deterministic detectors and the gauge detectors of the other
    checks use repetition 0 (the link of the schedule) exactly as for k = 1.
    The gauge detector of a random link compares its first outcome of the
    round with the last of the round before, and every round adds the
    comparisons of consecutive repetitions of every link (detector phase
    REP_PHASE, coordinates (check, round, 5, j)), which belong to the lower
    (gauge) level. Matching on the S_0 graph alone (mode "mwpm") does not
    see them.

    `noise.link_noise = False` makes every operation of the link checks
    noiseless (ancilla reset, gates, idling, measurement, and the pair
    measurements of hybrid and pairs; the outcome flip under phen). Data
    qubits in a noiseless link gate are busy and do not idle. Diagnostic only.

    Attributes
    ----------
    geometry : XYZ2Geometry Lattice metadata. frame : XYZ2FrameBasis Product
    frame and deterministic rows. schedule : ExtractionSchedule Layer
    assignment. rounds : int Number of extraction rounds. noise :
    CircuitNoise Noise model. ancillas : List[int] Ancilla qubits (with the
    ancillas of repeated links). ancilla_of : Dict[int, int] Ancilla of every
    check measured with gates. depth : int Gate layers per round. logical :
    str "X" or "Y".
    """

    def __init__(
        self,
        distance: int,
        rounds: int,
        noise: CircuitNoise,
        schedule: ExtractionSchedule = DEPTH6_SCHEDULE,
        init: str = "frame",
        final: str = "frame",
        detectors: str = "s0",
        odd_rep: str = "ZY",
        share_ancillas: int = 0,
        logical: str = "X",
    ) -> None:
        if rounds < 1:
            raise ValueError("rounds must be at least 1.")
        if init not in {"frame", "plus"}:
            raise ValueError("init must be 'frame' or 'plus'.")
        if final not in {"frame", "ideal"}:
            raise ValueError("final must be 'frame' or 'ideal'.")
        if detectors not in {"s0", "all", "combined"}:
            raise ValueError("detectors must be 's0', 'all' or 'combined'.")
        if init == "plus" and detectors == "s0":
            raise ValueError("init='plus' needs detectors='all'.")
        if logical not in {"X", "Y"}:
            raise ValueError("logical must be 'X' or 'Y'.")
        if logical != "X" and init == "plus":
            raise ValueError("init='plus' is defined for logical='X' only.")
        self.native = getattr(noise, "native", "gates")
        self.link_reps = int(getattr(noise, "link_reps", 1))
        self.link_noise = bool(getattr(noise, "link_noise", True))
        if self.native not in NATIVES:
            raise ValueError(f"noise.native must be one of {NATIVES}.")
        if self.link_reps < 1:
            raise ValueError("noise.link_reps must be at least 1.")
        if self.native == "pairs" and self.link_reps > 1:
            raise NotImplementedError(
                "link_reps > 1 is not implemented for the pair-measurement "
                "extraction (native='pairs'): every data qubit is busy in all "
                "four steps of its round, so a repetition needs a new step.")
        if share_ancillas and (self.native != "gates" or self.link_reps > 1
                               or not self.link_noise):
            raise ValueError("share_ancillas needs native='gates', link_reps=1 "
                             "and link_noise=True.")
        self.geometry = XYZ2Geometry(distance)
        self.logical = logical
        self.frame = XYZ2FrameBasis(self.geometry, odd_rep=odd_rep,
                                    logical=logical)
        ok, msg = schedule.validate(self.geometry)
        if not ok:
            raise ValueError(f"Invalid schedule: {msg}")
        self.schedule = schedule
        self.rounds = rounds
        self.noise = noise
        self.init = init
        self.final = final
        self.detectors = detectors
        n = self.geometry.n
        m = len(self.geometry.checks)
        self.links = [ci for ci, ch in enumerate(self.geometry.checks)
                      if ch["kind"] == "link"]
        self.slots = schedule.slot_map(self.geometry)
        self.shared_pairs = self._pick_shared_pairs(share_ancillas)
        late = {b for _, b in self.shared_pairs}
        self.ancilla_of: Dict[int, int] = {}
        nxt = n
        if self.native != "phen":
            for ci in range(m):
                if ci in late or (self.native == "hybrid" and ci in self.links):
                    continue
                self.ancilla_of[ci] = nxt
                nxt += 1
        for a, b in self.shared_pairs:
            self.ancilla_of[b] = self.ancilla_of[a]
        self.depth = schedule.depth
        # ancilla of repetition j >= 1 of every link (gates)
        self.rep_ancilla: Dict[Tuple[int, int], int] = {}
        self.rep_slots: Dict[int, List[Tuple[int, int]]] = {}
        self.link_steps: Dict[int, List[object]] = {}
        self.rep_order: Dict[int, List[int]] = {}
        if self.native == "gates" and self.link_reps > 1:
            self.rep_slots = link_rep_slots(schedule, self.link_reps)
            self.depth = max(schedule.depth, 1 + max(
                max(s) for v in self.rep_slots.values() for s in v))
            # time order of the repetitions (repetition 0 is the link of the schedule)
            self.rep_order = {par: sorted(range(len(v)), key=lambda j, v=v: (max(v[j]), min(v[j])))
                              for par, v in self.rep_slots.items()}
            for j in range(1, self.link_reps):
                for ci in self.links:
                    self.rep_ancilla[(ci, j)] = nxt
                    nxt += 1
        self.ancillas = sorted(set(self.ancilla_of.values())) + [
            self.rep_ancilla[(ci, j)] for j in range(1, self.link_reps)
            for ci in self.links if (ci, j) in self.rep_ancilla]
        # flip qubits of the EM3 channel of the hybrid link measurements
        self.flip_of: Dict[int, int] = {}
        if self.native == "hybrid":
            self.link_steps = hybrid_link_steps(schedule, self.link_reps)
            for ci in self.links:
                self.flip_of[ci] = nxt
                nxt += 1
        # ancillas whose operations are noiseless (link_noise=False)
        self.quiet = set()
        if not self.link_noise:
            self.quiet = {self.ancilla_of[ci] for ci in self.links
                          if ci in self.ancilla_of}
            self.quiet |= set(self.rep_ancilla.values())
        self.init_basis = (
            {q: "X" for q in range(n)} if init == "plus"
            else dict(self.frame.basis)
        )
        self.gauge = self.frame.gauge_generators()

    def physical_qubits(self) -> int:
        """
        Data qubits and ancillas (without the EM3 flip qubits, which are a
        device of the noise model).
        """
        if self.native == "pairs":
            plaq = sum(len(ch["terms"]) for ch in self.geometry.checks
                       if ch["kind"] != "link")
            return self.geometry.n + 2 * plaq
        return self.geometry.n + len(self.ancillas)

    def _parity(self, ci: int) -> int:
        i, j = self.geometry.checks[ci]["block"]
        return (i + j) % 2

    def build(self) -> stim.Circuit:
        """
        Return the noisy, detector-annotated Stim circuit.
        """
        if self.native == "pairs":
            return self._build_pairs()
        if self.native == "phen":
            return self._build_phen()
        return self._build_gates()

    def _build_gates(self) -> stim.Circuit:
        """
        Extraction with ancillas and controlled Paulis ("gates"), and the
        hybrid extraction whose links are pair measurements ("hybrid").
        """
        geo, nz = self.geometry, self.noise
        n, m = geo.n, len(geo.checks)
        hybrid = self.native == "hybrid"
        data = list(range(n))
        all_q = data + list(self.ancillas)
        quiet = self.quiet
        noisy_anc = [a for a in self.ancillas if a not in quiet]
        c = stim.Circuit()
        for q in data:
            c.append("QUBIT_COORDS", [q], [q, 0])
        for k, a in enumerate(self.ancillas):
            c.append("QUBIT_COORDS", [a], [k, 1])
        for k, ci in enumerate(self.links):
            if ci in self.flip_of:
                c.append("QUBIT_COORDS", [self.flip_of[ci]], [k, 0, 3])
        for b in "XYZ":
            qs = [q for q in data if self.init_basis[q] == b]
            if qs:
                c.append(_RESET[b], qs)
                if nz.p_prep:
                    c.append(_FLIP[b], qs, nz.p_prep)
        c.append("TICK")

        layers: List[List[Tuple[int, int, str]]] = [
            [] for _ in range(self.depth)
        ]
        for (ci, q), t in self.slots.items():
            if hybrid and ci in self.flip_of:
                continue
            pauli = dict((qq, p) for p, qq in geo.checks[ci]["terms"])[q]
            layers[t].append((self.ancilla_of[ci], q, pauli))
        for ci in self.links:
            (_, u), (_, lo) = geo.checks[ci]["terms"]
            for j, (tu, tl) in enumerate(self.rep_slots.get(self._parity(ci), [])[1:], 1):
                layers[tu].append((self.rep_ancilla[(ci, j)], u, "X"))
                layers[tl].append((self.rep_ancilla[(ci, j)], lo, "X"))
        # pair measurements of the hybrid links: step -> [(link, repetition)]
        mpp_at: Dict[object, List[Tuple[int, int]]] = {}
        for ci in self.flip_of:
            for j, st in enumerate(self.link_steps[self._parity(ci)]):
                mpp_at.setdefault(st, []).append((ci, j))
        mid_layer = max(
            (max(t for (cj, _), t in self.slots.items() if cj == a)
             for a, _ in self.shared_pairs), default=None)
        early = [a for a, _ in self.shared_pairs]
        end_checks = [ci for ci in range(m) if ci not in set(early)
                      and ci in self.ancilla_of]
        reps_meas = [(ci, j) for j in range(1, self.link_reps)
                     for ci in self.links if (ci, j) in self.rep_ancilla]
        init_rows = (
            self.frame.s0_rows if self.init == "frame"
            else self.frame.deterministic_rows(self.init_basis)
        )
        rec: Dict[Tuple[int, int], int] = {}
        reps: Dict[Tuple[int, int], List[int]] = {}
        count = 0

        def link_mpps(step, r) -> set:
            """Pair measurements of the hybrid links in this step."""
            nonlocal count
            used = set()
            for ci, j in mpp_at.get(step, ()):
                (pu, u), (pl, lo) = geo.checks[ci]["terms"]
                _append_em3_mpp(c, u, pu, lo, pl, self.flip_of[ci],
                                nz.p_mpp if self.link_noise else 0.0)
                reps.setdefault((r, ci), []).append(count)
                if j == 0:
                    rec[(r, ci)] = count
                count += 1
                used.update((u, lo))
            return used

        for r in range(self.rounds):
            if self.ancillas:
                c.append("RX", self.ancillas)
            if nz.p_prep and noisy_anc:
                c.append("Z_ERROR", noisy_anc, nz.p_prep)
            used = link_mpps("R", r)
            if not nz.xtalk_per_mcmr:
                self._xtalk(c, data)
            self._idle_mr(c, [q for q in data if q not in used])
            if r == 0:
                self._memory(c, [q for q in all_q if q not in quiet])
            c.append("TICK")
            for t, layer in enumerate(layers):
                pairs: List[int] = []
                noisy: List[int] = []
                by_type: Dict[str, List[int]] = {}
                for gate in ("CX", "CY", "CZ"):
                    targets = []
                    for a, q, p in sorted(layer):
                        if _CTRL[p] == gate:
                            targets += [a, q]
                    if targets:
                        c.append(gate, targets)
                        pairs += targets
                        loud = [x for k in range(0, len(targets), 2)
                                if targets[k] not in quiet
                                for x in targets[k:k + 2]]
                        noisy += loud
                        if loud:
                            by_type[gate] = loud
                if noisy and nz.p2_paulis is not None and nz.p2_rzz_frame:
                    for gate, targets in by_type.items():
                        c.append("PAULI_CHANNEL_2", targets,
                                 _to_gate_frame(nz.p2_paulis, gate))
                elif noisy and nz.p2_paulis is not None:
                    c.append("PAULI_CHANNEL_2", noisy, list(nz.p2_paulis))
                elif noisy and nz.p2:
                    c.append("DEPOLARIZE2", noisy, nz.p2)
                if noisy and nz.dress_1q and nz.p1:
                    c.append("DEPOLARIZE1", noisy, nz.p1)
                busy = set(pairs) | link_mpps(t, r)
                self._idle(c, [q for q in all_q if q not in busy and q not in quiet])
                self._memory(c, [q for q in all_q if q not in quiet])
                c.append("TICK")
                if t == mid_layer:
                    shared = [self.ancilla_of[a] for a in early]
                    if nz.p_meas:
                        c.append("Z_ERROR", shared, nz.p_meas)
                    c.append("MX", shared)
                    for k, a in enumerate(early):
                        rec[(r, a)] = count + k
                    count += len(early)
                    c.append("RX", shared)
                    if nz.p_prep:
                        c.append("Z_ERROR", shared, nz.p_prep)
                    self._xtalk(c, data, len(shared))
                    c.append("TICK")
            end_anc = [self.ancilla_of[ci] for ci in end_checks]
            meas = end_anc + [self.rep_ancilla[key] for key in reps_meas]
            loud = [a for a in meas if a not in quiet]
            if nz.p_meas and loud:
                c.append("Z_ERROR", loud, nz.p_meas)
            if meas:
                c.append("MX", meas)
            for k, ci in enumerate(end_checks):
                rec[(r, ci)] = count + k
                if ci in self.links:
                    reps[(r, ci)] = [count + k]
            count += len(end_checks)
            for k, (ci, j) in enumerate(reps_meas):
                reps[(r, ci)].append(count + k)
            count += len(reps_meas)
            if reps_meas:
                # repetitions in the order of their gate layers
                for ci in self.links:
                    out = reps[(r, ci)]
                    reps[(r, ci)] = [out[j] for j in self.rep_order[self._parity(ci)]]
            used = link_mpps("M", r)
            self._xtalk(c, data, len(loud))
            self._idle_mr(c, [q for q in data if q not in used])
            self._memory(c, data)
            self._round_detectors(c, r, rec, reps, count, init_rows)
            c.append("TICK")
        return self._final_readout(c, rec, reps, count)

    def _build_phen(self) -> stim.Circuit:
        """
        Phenomenological noise (as in Srivastava et al., arXiv:2505.03691).

        The data qubits are prepared in the frame and read out in the frame
        without noise. Every round starts with PAULI_CHANNEL_1(`data_xyz`) on
        every data qubit, followed by a noiseless measurement of every check
        (one MPP per check, `link_reps` per link) whose outcome is flipped
        with probability `p_meas`. The detectors are those of the circuit
        with gates.
        """
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
        init_rows = (
            self.frame.s0_rows if self.init == "frame"
            else self.frame.deterministic_rows(self.init_basis)
        )
        xyz = tuple(nz.data_xyz) if nz.data_xyz is not None else (0.0, 0.0, 0.0)
        rec: Dict[Tuple[int, int], int] = {}
        reps: Dict[Tuple[int, int], List[int]] = {}
        count = 0
        for r in range(self.rounds):
            if any(xyz):
                c.append("PAULI_CHANNEL_1", data, list(xyz))
            c.append("TICK")
            for ci, ch in enumerate(geo.checks):
                link = ci in self.links
                flip = nz.p_meas if (self.link_noise or not link) else 0.0
                for j in range(self.link_reps if link else 1):
                    if flip:
                        c.append("MPP", self._mpp(ch["spec"]), flip)
                    else:
                        c.append("MPP", self._mpp(ch["spec"]))
                    if j == 0:
                        rec[(r, ci)] = count
                    if link:
                        reps.setdefault((r, ci), []).append(count)
                    count += 1
            self._round_detectors(c, r, rec, reps, count, init_rows)
            c.append("TICK")
        return self._final_readout(c, rec, reps, count, readout_noise=False)

    # ------------------------------------------------------------ detectors
    def _prev(self, rec, reps, r: int, ci: int) -> int:
        """Outcome of check ci in round r that the next round compares with:
        the last repetition of a repeated link, the only outcome otherwise."""
        if self.link_reps > 1 and ci in self.links:
            return reps[(r, ci)][-1]
        return rec[(r, ci)]

    def _first(self, rec, reps, r: int, ci: int) -> int:
        """First outcome of check ci in round r (the first repetition of a link)."""
        if self.link_reps > 1 and ci in self.links:
            return reps[(r, ci)][0]
        return rec[(r, ci)]

    def _round_detectors(self, c: stim.Circuit, r: int, rec, reps, count: int,
                         init_rows) -> None:
        """
        Detectors of round r: the frame-deterministic rows (and the gauge
        comparisons with `detectors="combined"`, or every check with
        `detectors="all"`), then the comparisons of consecutive repetitions
        of every link (not with `detectors="s0"`).
        """
        m = len(self.geometry.checks)
        if r == 0:
            for row in init_rows:
                self._detector(c, [rec[(0, ci)] for ci in np.flatnonzero(row)],
                               count, (0, 0, 0))
        elif self.detectors == "combined":
            for k, row in enumerate(self.frame.s0_rows):
                idx = [rec[(rr, ci)] for rr in (r, r - 1)
                       for ci in np.flatnonzero(row)]
                self._detector(c, idx, count, (k, r, 0))
            for k, ci in enumerate(self.gauge):
                self._detector(c, [self._first(rec, reps, r, ci),
                                   self._prev(rec, reps, r - 1, ci)], count, (ci, r, 3))
        elif self.detectors == "all":
            for ci in range(m):
                self._detector(c, [self._first(rec, reps, r, ci),
                                   self._prev(rec, reps, r - 1, ci)], count, (ci, r, 0))
        else:
            for k, row in enumerate(self.frame.s0_rows):
                idx = [rec[(rr, ci)] for rr in (r, r - 1)
                       for ci in np.flatnonzero(row)]
                self._detector(c, idx, count, (k, r, 0))
        if self.link_reps > 1 and self.detectors != "s0":
            for ci in self.links:
                out = reps[(r, ci)]
                for j in range(1, len(out)):
                    c.append("DETECTOR", [stim.target_rec(out[j - 1] - count),
                                          stim.target_rec(out[j] - count)],
                             [ci, r, REP_PHASE, j])

    def _final_readout(self, c: stim.Circuit, rec, reps, count: int,
                       readout_noise: bool = True) -> stim.Circuit:
        """
        Final noiseless round (`final="ideal"`) or destructive frame readout
        (`final="frame"`), its detectors and the observable.
        """
        geo, nz = self.geometry, self.noise
        n, m = geo.n, len(geo.checks)
        data = list(range(n))
        last = self.rounds - 1
        if self.final == "ideal":
            for ch in geo.checks:
                c.append("MPP", self._mpp(ch["spec"]))
            c.append("MPP", self._mpp(self.frame.logical_spec))
            base = count
            count += m + 1
            rows = (np.eye(m, dtype=np.uint8) if self.detectors == "all"
                    else self.frame.s0_rows)
            for k, row in enumerate(rows):
                idx = [base + ci for ci in np.flatnonzero(row)]
                if self.detectors == "all":
                    idx += [self._prev(rec, reps, last, ci) for ci in np.flatnonzero(row)]
                else:
                    idx += [rec[(last, ci)] for ci in np.flatnonzero(row)]
                self._detector(c, idx, count, (k, self.rounds, 1))
            if self.detectors == "combined":
                for ci in self.gauge:
                    self._detector(c, [base + ci, self._prev(rec, reps, last, ci)],
                                   count, (ci, self.rounds, 4))
            c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)
            return c
        position: Dict[int, int] = {}
        for b in "XYZ":
            qs = [q for q in data if self.frame.basis[q] == b]
            if qs:
                if nz.p_meas and readout_noise:
                    c.append(_FLIP[b], qs, nz.p_meas)
                c.append(_MEAS[b], qs)
                for k, q in enumerate(qs):
                    position[q] = count + k
                count += len(qs)
        for k, row in enumerate(self.frame.s0_rows):
            spec = self.frame.product_spec(row)
            idx = [position[int(t[1:])] for t in spec.split()]
            idx += [rec[(last, ci)] for ci in np.flatnonzero(row)]
            self._detector(c, idx, count, (k, self.rounds, 2))
        lx = self.frame.logical_spec.split()
        c.append("OBSERVABLE_INCLUDE",
                 [stim.target_rec(position[int(t[1:])] - count) for t in lx], 0)
        return c

    # ------------------------------------------------------------ pairs
    def plaquette_colors(self) -> Dict[int, int]:
        """
        Three-colouring of the plaquettes (hexagons and boundary checks).

        Plaquettes that share a qubit get different colours. In the pair
        extraction each data qubit couples to its plaquettes in the order of
        their colours, so that two overlapping plaquettes are coupled in the
        same order on all the qubits they share.
        """
        geo = self.geometry
        plaq = [ci for ci, ch in enumerate(geo.checks) if ch["kind"] != "link"]
        qs = {ci: {q for _, q in geo.checks[ci]["terms"]} for ci in plaq}
        nb = {ci: [cj for cj in plaq if cj != ci and qs[ci] & qs[cj]] for ci in plaq}
        color: Dict[int, int] = {}

        def solve(order, k=0):
            if k == len(order):
                return True
            ci = order[k]
            for col in range(3):
                if all(color.get(cj) != col for cj in nb[ci]):
                    color[ci] = col
                    if solve(order, k + 1):
                        return True
                    del color[ci]
            return False

        order, seen = [], set()
        for start in plaq:
            if start in seen:
                continue
            queue = [start]
            seen.add(start)
            while queue:
                ci = queue.pop(0)
                order.append(ci)
                for cj in sorted(nb[ci]):
                    if cj not in seen:
                        seen.add(cj)
                        queue.append(cj)
        if not solve(order):
            raise RuntimeError("plaquettes are not three-colourable")
        return color

    def cat_tree(self, ci: int):
        """
        Tree of Z_a Z_a' measurements that joins the ancillas of plaquette ci.

        The ancillas are grouped by block. One ancilla of a group of size one
        is the centre, every other group hangs from the centre, and a group of
        two qubits is a path centre - x - y. A wrong outcome on a tree edge
        acts like the plaquette operator on the qubits below that edge, so it
        is equivalent to an error on one qubit or on the two qubits of one
        block. Returns (edges, layer, parent): edges as pairs of ancilla
        indices (parent, child), the layer (0, 1, 2) of each edge, and the
        parent of every ancilla (-1 for the centre).
        """
        geo = self.geometry
        block = {}
        for ch in geo.checks:
            if ch["kind"] == "link":
                for _, q in ch["terms"]:
                    block[q] = ch["block"]
        qs = [q for _, q in geo.checks[ci]["terms"]]
        groups: List[List[int]] = []
        where: Dict[Tuple[int, int], int] = {}
        for k, q in enumerate(qs):
            key = block[q]
            if key not in where:
                where[key] = len(groups)
                groups.append([])
            groups[where[key]].append(k)
        singles = [g for g in groups if len(g) == 1]
        cg = singles[0] if singles else groups[0]
        center = cg[0]
        edges: List[Tuple[int, int]] = []
        for g in groups:
            if g is cg:
                for k in g[1:]:
                    edges.append((center, k))
                continue
            edges.append((center, g[0]))
            for a, b in zip(g, g[1:]):
                edges.append((a, b))
        # layers: edges at the centre in turn, the others in the first free layer
        layer: List[int] = []
        busy: Dict[int, set] = {}
        cen = 0
        for a, b in edges:
            if a == center:
                t = cen
                cen += 1
            else:
                t = next(t for t in range(3) if t not in busy.get(a, set()) | busy.get(b, set()))
            layer.append(t)
            busy.setdefault(a, set()).add(t)
            busy.setdefault(b, set()).add(t)
        if max(layer) > 2:
            raise RuntimeError("cat tree needs more than three layers")
        parent = [-1] * len(qs)
        for a, b in edges:
            parent[b] = a
        return edges, layer, parent

    def _build_pairs(self) -> stim.Circuit:
        """
        Extraction with native two-qubit Pauli product measurements (EM3).

        A link X_u X_l is measured directly by one MPP. A plaquette with w
        qubits uses w ancillas, which are reset in |+> and joined into a cat
        state by w - 1 MPPs Z_a Z_a' along the tree of `cat_tree`. Each
        ancilla is then measured jointly with one data qubit by an MPP X_a P_q
        and read out in the Z basis in the next step. The product of the w
        coupling outcomes is the plaquette, because X^w = +1 on the cat state.
        The coupling MPPs do not commute with the other checks qubit by qubit,
        so a check C that overlaps a plaquette A changes sign by the Z readouts
        of the ancillas of A on the qubits where C and A differ, times the
        cat parity between those ancillas. These signs enter the detectors and
        the observable. Plaquettes couple in the order of their colour, so two
        overlapping plaquettes are coupled in the same order on every qubit
        they share. Two banks of ancillas alternate between rounds, and a
        round has four steps: the links, then the three colours.

        EM3 noise (Gidney et al. 2021): with probability p an MPP is followed
        by an element of {I,X,Y,Z}^2 x {flip, no flip} chosen uniformly. A
        flip that has to stay correlated with the Pauli error acts through an
        auxiliary qubit in |0> that is included in the MPP as Z_f. For a
        coupling the same distribution is a uniform two-qubit Pauli before
        the MPP, because the ancilla is read out right after it.
        """
        geo, nz = self.geometry, self.noise
        n, m = geo.n, len(geo.checks)
        p = nz.p_mpp
        data = list(range(n))
        checks = geo.checks
        plaq = [ci for ci, ch in enumerate(checks) if ch["kind"] != "link"]
        links = [ci for ci, ch in enumerate(checks) if ch["kind"] == "link"]
        color = self.plaquette_colors()
        pauli = {ci: {q: pp for pp, q in checks[ci]["terms"]} for ci in range(m)}
        terms = {ci: [q for _, q in checks[ci]["terms"]] for ci in plaq}
        trees = {ci: self.cat_tree(ci) for ci in plaq}
        nxt = n
        cat: List[Dict[int, List[int]]] = [{}, {}]
        for b in (0, 1):
            for ci in plaq:
                w = len(terms[ci])
                cat[b][ci] = list(range(nxt, nxt + w))
                nxt += w
        per_layer = [sum(1 for ci in plaq for t in trees[ci][1] if t == L) for L in range(3)]
        n_pool = len(links) + max(per_layer)
        pool = list(range(nxt, nxt + n_pool))

        c = stim.Circuit()
        for q in data:
            c.append("QUBIT_COORDS", [q], [q, 0])
        for b in (0, 1):
            for ci in plaq:
                for k, a in enumerate(cat[b][ci]):
                    c.append("QUBIT_COORDS", [a], [ci, k, b + 1])
        for k, f in enumerate(pool):
            c.append("QUBIT_COORDS", [f], [k, 0, 3])

        count = 0
        rec_m: Dict[Tuple[int, int], List[int]] = {}
        rec_z: Dict[Tuple[int, int, int], int] = {}
        rec_c: Dict[Tuple[int, int, int], int] = {}
        live = set(data)

        def noisy_mpp(q1, p1, q2, p2, f, prob=p):
            """MPP P1 P2 with the correlated EM3 error, using the flip qubit f."""
            nonlocal count
            _append_em3_mpp(c, q1, p1, q2, p2, f, prob)
            count += 1
            return count - 1

        def end_step(used):
            idle_q = sorted(q for q in live if q not in used)
            if idle_q and nz.p_idle:
                c.append("DEPOLARIZE1", idle_q, nz.p_idle)
            c.append("TICK")

        def reset_bank(b):
            qs = [q for ci in plaq for q in cat[b][ci]]
            c.append("RX", qs)
            if nz.p_prep:
                c.append("Z_ERROR", qs, nz.p_prep)
            live.update(qs)
            return qs

        def cat_layer(b, rr, L, fi0=0):
            used = []
            fi = fi0
            for ci in plaq:
                edges, layer, _ = trees[ci]
                for e, ((k1, k2), t) in enumerate(zip(edges, layer)):
                    if t != L:
                        continue
                    a1, a2 = cat[b][ci][k1], cat[b][ci][k2]
                    rec_c[(rr, ci, e)] = noisy_mpp(a1, "Z", a2, "Z", pool[fi])
                    fi += 1
                    used += [a1, a2]
            return used

        def readout(items):
            """Z readout of coupled ancillas; items are (round, plaquette, k, ancilla)."""
            nonlocal count
            if not items:
                return []
            qs = [a for _, _, _, a in items]
            if nz.p_meas:
                c.append("X_ERROR", qs, nz.p_meas)
            c.append("M", qs)
            for j, (rr, ci, k, _) in enumerate(items):
                rec_z[(rr, ci, k)] = count + j
            count += len(qs)
            live.difference_update(qs)
            return qs

        # preparation of the data and of the first cat states
        for bb in "XYZ":
            qs = [q for q in data if self.init_basis[q] == bb]
            if qs:
                c.append(_RESET[bb], qs)
                if nz.p_prep:
                    c.append(_FLIP[bb], qs, nz.p_prep)
        used = data + reset_bank(0)
        end_step(set(used))
        end_step(set(cat_layer(0, 0, 0)))
        end_step(set(cat_layer(0, 0, 1)))

        pending: List[Tuple[int, int, int, int]] = []
        for r in range(self.rounds):
            b = r % 2
            more = r < self.rounds - 1
            # step 0: links, last layer of this round's cats, readout of the
            # last colour of the previous round
            used = []
            for fi, ci in enumerate(links):
                (pu, u), (pl, l) = checks[ci]["terms"]
                rec_m[(r, ci)] = [noisy_mpp(u, pu, l, pl, pool[fi],
                                            p if self.link_noise else 0.0)]
                used += [u, l]
            used += cat_layer(b, r, 2, fi0=len(links))
            used += readout(pending)
            pending = []
            end_step(set(used))
            # steps 1-3: couplings by colour, readout of the previous colour,
            # cat preparation of the other bank
            for col in range(3):
                used = []
                layer = [(ci, k, q) for ci in plaq if color[ci] == col
                         for k, q in enumerate(terms[ci])]
                for ci, k, q in layer:
                    a = cat[b][ci][k]
                    if p:
                        c.append("DEPOLARIZE2", [a, q], 15.0 * p / 16.0)
                    c.append("MPP", [stim.target_x(a), stim.target_combiner(),
                                     stim.target_pauli(q, pauli[ci][q])])
                    rec_m.setdefault((r, ci), []).append(count)
                    count += 1
                    used += [a, q]
                used += readout(pending)
                pending = [(r, ci, k, cat[b][ci][k]) for ci, k, q in layer]
                if more:
                    if col == 0:
                        used += reset_bank(1 - b)
                    else:
                        used += cat_layer(1 - b, r + 1, col - 1)
                end_step(set(used))
        end_step(set(readout(pending)))

        # ---------------------------------------------------------- signs
        def step_of(ci):
            return 0 if checks[ci]["kind"] == "link" else color[ci] + 1

        def corr(rr, a_ci, op):
            """Records by which plaquette a_ci in round rr flips the sign of the Pauli `op`."""
            Q = {k for k, q in enumerate(terms[a_ci]) if q in op and op[q] != pauli[a_ci][q]}
            if not Q:
                return []
            if len(Q) % 2:
                raise RuntimeError("operator anticommutes with a plaquette")
            edges, _, parent = trees[a_ci]
            below = {k: int(k in Q) for k in range(len(terms[a_ci]))}
            # count Q-nodes below every node (children before parents)
            depth = {}
            for k in range(len(parent)):
                dd, x = 0, k
                while parent[x] >= 0:
                    x = parent[x]
                    dd += 1
                depth[k] = dd
            for k in sorted(below, key=lambda k: -depth[k]):
                if parent[k] >= 0:
                    below[parent[k]] += below[k]
            out = [rec_z[(rr, a_ci, k)] for k in sorted(Q)]
            for e, (k1, k2) in enumerate(edges):
                if below[k2] % 2:
                    out.append(rec_c[(rr, a_ci, e)])
            return out

        def between(ci, r0, r1):
            s = step_of(ci)
            out = []
            for a_ci in plaq:
                if a_ci == ci:
                    continue
                sa = step_of(a_ci)
                for rr in range(r0, r1 + 1):
                    if (rr == r0 and sa <= s) or (rr == r1 and sa >= s):
                        continue
                    out.append((rr, a_ci))
            return out

        def xor(lists):
            acc = set()
            for lst in lists:
                for x in lst:
                    acc ^= {x}
            return sorted(acc)

        dets = []
        init_rows = (self.frame.s0_rows if self.init == "frame"
                     else self.frame.deterministic_rows(self.init_basis))
        for row in init_rows:
            idx = []
            for ci in np.flatnonzero(row):
                s = step_of(ci)
                idx.append(rec_m[(0, ci)])
                idx += [corr(0, a, pauli[ci]) for a in plaq if a != ci and step_of(a) < s]
            dets.append((xor(idx), (0, 0, 0)))
        for r in range(1, self.rounds):
            per_check = {}
            for ci in range(m):
                lst = [rec_m[(r, ci)], rec_m[(r - 1, ci)]]
                lst += [corr(rr, a, pauli[ci]) for rr, a in between(ci, r - 1, r)]
                per_check[ci] = xor(lst)
            if self.detectors in ("combined", "s0"):
                for k, row in enumerate(self.frame.s0_rows):
                    dets.append((xor([per_check[ci] for ci in np.flatnonzero(row)]), (k, r, 0)))
                if self.detectors == "combined":
                    for ci in self.gauge:
                        dets.append((per_check[ci], (ci, r, 3)))
            else:
                for ci in range(m):
                    dets.append((per_check[ci], (ci, r, 0)))
        last = self.rounds - 1

        def end_value(ci):
            s = step_of(ci)
            return xor([rec_m[(last, ci)]] + [corr(last, a, pauli[ci]) for a in plaq
                                              if a != ci and step_of(a) > s])

        if self.final == "ideal":
            for ch in checks:
                c.append("MPP", self._mpp(ch["spec"]))
            c.append("MPP", self._mpp(self.frame.logical_spec))
            base = count
            count += m + 1
            rows = (np.eye(m, dtype=np.uint8) if self.detectors == "all" else self.frame.s0_rows)
            for k, row in enumerate(rows):
                dets.append((xor([[base + ci] + end_value(ci) for ci in np.flatnonzero(row)]),
                             (k, self.rounds, 1)))
            if self.detectors == "combined":
                for ci in self.gauge:
                    dets.append((xor([[base + ci], end_value(ci)]), (ci, self.rounds, 4)))
            lx = {int(t[1:]): t[0] for t in self.frame.logical_spec.split()}
            obs = [[base + m]]
        else:
            position: Dict[int, int] = {}
            for bb in "XYZ":
                qs = [q for q in data if self.frame.basis[q] == bb]
                if qs:
                    if nz.p_meas:
                        c.append(_FLIP[bb], qs, nz.p_meas)
                    c.append(_MEAS[bb], qs)
                    for k, q in enumerate(qs):
                        position[q] = count + k
                    count += len(qs)
            for k, row in enumerate(self.frame.s0_rows):
                spec = self.frame.product_spec(row)
                idx = [[position[int(t[1:])] for t in spec.split()]]
                idx += [end_value(ci) for ci in np.flatnonzero(row)]
                dets.append((xor(idx), (k, self.rounds, 2)))
            lx = {int(t[1:]): t[0] for t in self.frame.logical_spec.split()}
            obs = [[position[q] for q in lx]]
        # every coupled plaquette can flip the logical operator
        for rr in range(self.rounds):
            for a in plaq:
                obs.append(corr(rr, a, lx))
        obs = xor(obs)
        for idx, coords in dets:
            self._detector(c, idx, count, coords)
        c.append("OBSERVABLE_INCLUDE", [stim.target_rec(i - count) for i in obs], 0)
        return c

    def _pick_shared_pairs(self, k: int) -> List[Tuple[int, int]]:
        """
        Pair `k` link checks with boundary checks whose layers start later.
        """
        if k <= 0:
            return []
        checks = self.geometry.checks
        window = {}
        for (ci, _), t in self.slots.items():
            lo, hi = window.get(ci, (t, t))
            window[ci] = (min(lo, t), max(hi, t))
        links = [ci for ci, ch in enumerate(checks) if ch["kind"] == "link"]
        link_end = max(window[ci][1] for ci in links)
        late = [ci for ci, ch in enumerate(checks)
                if ch["kind"] == "hex" and len(ch["terms"]) == 3
                and window[ci][0] > link_end]
        if k > len(late):
            raise ValueError(
                f"share_ancillas={k} but only {len(late)} boundary checks "
                f"start after layer {link_end}.")
        return list(zip(links[:k], late[:k]))

    def _detector(self, c: stim.Circuit, idx: Sequence[int], count: int,
                  coords: Tuple[int, int, int]) -> None:
        c.append("DETECTOR", [stim.target_rec(i - count) for i in idx],
                 list(coords))

    def _idle(self, c: stim.Circuit, qs: Sequence[int]) -> None:
        if not qs:
            return
        if self.noise.idle_xyz is not None:
            c.append("PAULI_CHANNEL_1", list(qs), list(self.noise.idle_xyz))
        elif self.noise.p_idle:
            c.append("DEPOLARIZE1", list(qs), self.noise.p_idle)

    def _idle_mr(self, c: stim.Circuit, qs: Sequence[int]) -> None:
        """
        Idle noise on data qubits while the ancillas are reset or measured.
        """
        if self.noise.p_idle_mr is None:
            self._idle(c, qs)
        elif qs and self.noise.p_idle_mr:
            c.append("DEPOLARIZE1", list(qs), self.noise.p_idle_mr)

    def _memory(self, c: stim.Circuit, qs: Sequence[int]) -> None:
        if qs and self.noise.p_mem_z:
            c.append("Z_ERROR", list(qs), self.noise.p_mem_z)

    def _xtalk(self, c: stim.Circuit, qs: Sequence[int], k: int = 1) -> None:
        """
        Crosstalk on the data qubits from measuring and resetting ancillas.

        With `xtalk_per_mcmr` every one of the `k` measured ancillas applies
        depolarizing noise `p_xtalk` to each data qubit, which composes to a
        single depolarizing channel with 3/4 [1 - (1 - 4 p_xtalk / 3)^k].
        Otherwise the noise is applied once per measurement or reset step.
        """
        p = self.noise.p_xtalk
        if not qs or not p:
            return
        if self.noise.xtalk_per_mcmr:
            p = 0.75 * (1.0 - (1.0 - 4.0 * p / 3.0) ** k)
        c.append("DEPOLARIZE1", list(qs), p)

    @staticmethod
    def _mpp(spec: str) -> List[stim.GateTarget]:
        out: List[stim.GateTarget] = []
        make = {"X": stim.target_x, "Y": stim.target_y, "Z": stim.target_z}
        for k, tok in enumerate(spec.split()):
            if k:
                out.append(stim.target_combiner())
            out.append(make[tok[0]](int(tok[1:])))
        return out
