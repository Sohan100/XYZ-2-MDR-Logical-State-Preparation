"""

competitor_circuits.py
----------------------------------------------------------------------------
Memory experiments of competitor codes under the circuit-level noise
conventions of `FTMDRCircuit`, for threshold comparisons with XYZ^2.

    from mdr.ft.competitor_circuits import competitor_circuit
    circuit = competitor_circuit("xzzx", d=5, rounds=5,
                                 noise=CircuitNoise.biased(5e-3, 100), basis="X")

Codes (`CODES`):

- ``css``: the rotated surface code (XXXX and ZZZZ plaquettes);
- ``xzzx``: the rotated XZZX surface code (Bonilla Ataides et al., Nat.
  Commun. 12, 2172 (2021)), every plaquette X Z Z X;
- ``xy``: the rotated XY surface code with XXXX and YYYY plaquettes (Tuckett,
  Bartlett and Flammia, PRL 120, 050505 (2018), on the rotated layout).

XZZX and XY are the CSS code conjugated by single-qubit Cliffords on the data
qubits (a Hadamard on every data qubit with x + y odd, and Z -> Y, Y -> Z on
every data qubit, respectively). Every element of the circuits (plaquette
supports, gate schedule, logical operators, preparation and readout bases) is
built for the CSS code and mapped through that Clifford, so under noise that
is invariant under single-qubit Cliffords (sd6, si1000, em3) the three codes
have the same detector error model; they differ only under biased noise
(biased10, biased100, purez and the Z dephasing of the trapped-ion models).

Floquet code (`FLOQUET_CODES`):

- ``honeycomb``: the periodic Hastings-Haah honeycomb code in the layout of
  Gidney, Newman, Fowler and Broughton (Quantum 5, 605 (2021)), see
  `HoneycombCircuit`.
"""

from __future__ import annotations

import collections
import math
from typing import Dict, List, Sequence, Tuple

import stim

from .circuit_noise import CircuitNoise
from .ft_mdr_circuit import (_CTRL, _FLIP, _MEAS, _RESET, FTMDRCircuit,
                             _to_gate_frame)

CODES = ("css", "xzzx", "xy")
FLOQUET_CODES = ("honeycomb",)
BASES = ("X", "Z")

# Corner order of the hook-safe schedule for each CSS type of plaquette. The
# corners of the plaquette of cell (x, y) are NW = (x, y), NE = (x + 1, y),
# SW = (x, y + 1), SE = (x + 1, y + 1); corner k of the order is coupled in
# gate layer k. X plaquettes go in a "Z" shape, so the hook left after two
# gates is a horizontal X pair; Z plaquettes go in an "N" shape, so their
# hook is a vertical Z pair. Logical X is a vertical X string and logical Z a
# horizontal Z string, so each hook is perpendicular to the logical operator
# that it could extend (Tomita and Svore, PRA 90, 062320 (2014)).
SCHEDULE = {"X": ("NW", "NE", "SW", "SE"), "Z": ("NW", "SW", "NE", "SE")}
# Cat-state star for the pair-measurement (EM3) extraction: centre first, then
# the leaves in the order of their tree layers. A Pauli fault on the centre
# between layers 1 and 2 leaves the recorded cat parities of the first two
# edges wrong, which acts on the data like the plaquette operator on the first
# two leaves. Those two leaves are a horizontal pair for X plaquettes and a
# vertical pair for Z plaquettes, the same hook geometry as `SCHEDULE`.
CAT_STAR = {"X": ("NW", "SW", "SE", "NE"), "Z": ("NW", "NE", "SE", "SW")}
_CORNERS = {"NW": (0, 0), "NE": (1, 0), "SW": (0, 1), "SE": (1, 1)}


def code_pauli(code: str, x: int, y: int, css_pauli: str) -> str:
    """
    Pauli of `code` on data qubit (x, y) that is the image of `css_pauli`.

    ``xzzx`` applies a Hadamard (X <-> Z) on data qubits with x + y odd, which
    maps both XXXX and ZZZZ plaquettes to X(NW) Z(NE) Z(SW) X(SE). ``xy``
    applies Z -> Y, Y -> Z on every data qubit (the Clifford SQRT_X_DAG up to
    signs), which maps ZZZZ to YYYY and leaves X unchanged.
    """
    if code == "css":
        return css_pauli
    if code == "xzzx":
        if (x + y) % 2:
            return {"X": "Z", "Y": "Y", "Z": "X"}[css_pauli]
        return css_pauli
    if code == "xy":
        return {"X": "X", "Y": "Z", "Z": "Y"}[css_pauli]
    raise ValueError(f"code must be one of {CODES}.")


class RotatedSurfaceLayout:
    """
    Rotated distance-d patch of the CSS, XZZX or XY surface code.

    Data qubit q = y d + x sits at (x, y), 0 <= x, y < d. Plaquette cell
    (x, y) has the corners NW = (x, y), NE = (x + 1, y), SW = (x, y + 1) and
    SE = (x + 1, y + 1) and the CSS type X when x + y is even, Z otherwise.
    The d^2 - 1 checks are the (d - 1)^2 weight-4 cells inside the patch,
    the weight-2 X cells along the top (y = -1) and bottom (y = d - 1) edges
    and the weight-2 Z cells along the left (x = -1) and right (x = d - 1)
    edges. The CSS logical X is X on the column x = 0 (it joins the two X
    boundaries) and the CSS logical Z is Z on the row y = 0.

    Attributes
    ----------
    code : str Code name. d : int Distance. n : int Number of data qubits.
    coords : List[Tuple[int, int]] Position of every data qubit. checks :
    List[Dict] One record per check with keys `css_type`, `cell`, `center`,
    `kind` (`bulk` or `boundary`), `terms` (list of (pauli, qubit) in gate
    order), `layer` (qubit -> gate layer), `cat` (qubits in cat-star order)
    and `color` (four-colouring of the cells, (x mod 2) + 2 (y mod 2)).
    """

    def __init__(self, code: str, d: int) -> None:
        if code not in CODES:
            raise ValueError(f"code must be one of {CODES}.")
        if d < 2:
            raise ValueError("d must be at least 2.")
        self.code = code
        self.d = d
        self.n = d * d
        self.coords: List[Tuple[int, int]] = [(q % d, q // d) for q in range(self.n)]
        index = {xy: q for q, xy in enumerate(self.coords)}
        self.checks: List[Dict] = []
        for y in range(-1, d):
            for x in range(-1, d):
                t = "X" if (x + y) % 2 == 0 else "Z"
                inside = {c: (x + dx, y + dy) for c, (dx, dy) in _CORNERS.items()
                          if (x + dx, y + dy) in index}
                if len(inside) == 4:
                    kind = "bulk"
                elif len(inside) == 2 and ((y in (-1, d - 1) and t == "X")
                                           or (x in (-1, d - 1) and t == "Z")):
                    kind = "boundary"
                else:
                    continue
                terms, layer = [], {}
                for k, c in enumerate(SCHEDULE[t]):
                    if c in inside:
                        q = index[inside[c]]
                        terms.append((code_pauli(code, *inside[c], t), q))
                        layer[q] = k
                cat = [index[inside[c]] for c in CAT_STAR[t] if c in inside]
                self.checks.append({
                    "css_type": t, "cell": (x, y), "center": (x + 0.5, y + 0.5),
                    "kind": kind, "terms": terms, "layer": layer, "cat": cat,
                    "color": (x % 2) + 2 * (y % 2),
                })
        if len(self.checks) != d * d - 1:
            raise RuntimeError("wrong number of checks")

    def frame(self, basis: str) -> Dict[int, str]:
        """
        Preparation and readout Pauli of every data qubit for a memory in `basis`.

        `basis` names the CSS logical ("X" or "Z"); the data qubits are
        prepared in the image of the CSS product state |+>^n or |0>^n.
        """
        if basis not in BASES:
            raise ValueError(f"basis must be one of {BASES}.")
        return {q: code_pauli(self.code, x, y, basis)
                for q, (x, y) in enumerate(self.coords)}

    def logical(self, basis: str) -> List[Tuple[str, int]]:
        """
        Logical operator of `basis` as a list of (pauli, qubit).
        """
        if basis not in BASES:
            raise ValueError(f"basis must be one of {BASES}.")
        d = self.d
        pts = [(0, y) for y in range(d)] if basis == "X" else [(x, 0) for x in range(d)]
        return [(code_pauli(self.code, x, y, basis), y * d + x) for x, y in pts]


class CompetitorCircuit:
    """
    Memory experiment of a competitor code with the noise of `FTMDRCircuit`.

    Protocol: prepare every data qubit in the product state of the logical
    basis (the frame of `RotatedSurfaceLayout.frame`), run `rounds` rounds
    of extraction of all checks, read every data qubit out in the same frame
    and compare. Detectors are the checks that are deterministic on the
    product state in round 0, every check against its previous round from
    round 1 on, and the deterministic checks against the data readout. The
    observable is the logical operator of `basis`, read from the data.

    Noise conventions (identical to `FTMDRCircuit.build`; the idle, memory,
    crosstalk and detector helpers are FTMDRCircuit's own methods):

    - Native gates (`noise.native == "gates"`): one ancilla per check, reset
      in |+> (RX, then Z_ERROR p_prep), coupled to each data qubit q of its
      support by the controlled Pauli of the check on q (CX, CY or CZ, ancilla
      control, data target), read out in the X basis (Z_ERROR p_meas, then
      MX). No Hadamard or other single-qubit gate is used, as in FTMDRCircuit.
    - After the gates of a layer: PAULI_CHANNEL_2(p2_paulis) on (ancilla,
      data) when `p2_paulis` is set (mapped from the RZZ frame to the frame of
      each controlled Pauli when `p2_rzz_frame`), DEPOLARIZE2(p2) otherwise.
      The Z-biased channels therefore act after each controlled Pauli in the
      computational frame, with the ancilla (control) first: the controlled
      Paulis are bias-preserving and undressed under the biased models, as in
      FTMDRCircuit. When `dress_1q` (trapped-ion models), DEPOLARIZE1(p1) on
      both qubits of every gate (the single-qubit basis changes that turn the
      native ZZ interaction into the controlled Pauli), for CX, CY and CZ
      alike.
    - Every qubit (data or ancilla) without a gate in a layer gets
      PAULI_CHANNEL_1(idle_xyz) if set, else DEPOLARIZE1(p_idle). Every qubit
      gets Z_ERROR(p_mem_z) in every gate layer.
    - Ancilla reset step: data qubits get the resonator idle (DEPOLARIZE1
      p_idle_mr, or the layer idle when p_idle_mr is None) and, unless
      `xtalk_per_mcmr`, DEPOLARIZE1(p_xtalk); all qubits get Z_ERROR(p_mem_z)
      in the reset step of round 0 only.
    - Ancilla measurement step: data qubits get the crosstalk of the k
      measured ancillas (3/4 [1 - (1 - 4 p_xtalk / 3)^k] with
      `xtalk_per_mcmr`, else p_xtalk), the resonator idle and Z_ERROR(p_mem_z).
    - Data preparation: R / RX / RY followed by the flip to the orthogonal
      state with p_prep (X_ERROR for Z and Y, Z_ERROR for X); final readout:
      the same flip with p_meas, then M / MX / MY. No idle noise is added
      to these two steps, as in FTMDRCircuit.

    Gate schedule: four layers, corner k of `SCHEDULE[css_type]` in layer
    k, so every data qubit is used once per layer and every pair of
    overlapping X and Z (or image) plaquettes is interleaved consistently.
    The hooks are perpendicular to the logical operators (fault distance d
    under depolarizing noise). For XZZX and XY the schedule is the image of
    the CSS one, so their hooks are the images of harmless CSS hooks. Under
    Z-biased noise the dominant ancilla faults are Z errors on the control of
    the controlled Paulis, which commute with the gates and do not spread, so
    the dominant faults have no hooks at all; the rare X and Y ancilla faults
    spread like in the CSS code.

    Pair measurements (`noise.native == "pairs"`, EM3): the construction of
    `FTMDRCircuit._build_pairs` applied to these codes. A check of weight w
    uses w ancillas reset in |+> (RX, Z_ERROR p_prep), joined into a cat
    state by w - 1 pair measurements Z_a Z_a' along a star (`cat_tree`),
    then coupled to the data by MPP X_a P_q and read out in the Z basis
    (X_ERROR p_meas, M) in the next step. Each MPP that is not followed by
    an immediate readout (cat edges, and direct pair measurements of
    weight-2 checks with `weight2="direct"`) carries the EM3 error: with
    probability p_mpp an element of {I,X,Y,Z}^2 x {flip, no flip}, the flip
    acting through an auxiliary qubit in |0> included in the MPP as Z_f.
    Couplings get DEPOLARIZE2(15 p_mpp / 16) before the MPP (the same
    distribution, because the ancilla is read out right after). Every live
    qubit that is not used in a step gets DEPOLARIZE1(p_idle). The checks
    are coupled in the order of a proper colouring of the cells (four
    colours: the four plaquettes around a bulk vertex overlap pairwise), and
    the sign changes of overlapping checks by the couplings are tracked as in
    FTMDRCircuit. Two banks of ancillas alternate between rounds; a round has
    one step for the readout of the last colour of the previous round (and
    any direct pair measurements) and one step per colour, during which the
    other bank is reset and its cat states are prepared.

    Weight-2 boundary checks are measured with a two-ancilla cat by default
    (`weight2="cat"`). FTMDRCircuit measures its weight-2 XX links directly
    with one MPP, where the correlated two-qubit error of an EM3 fault is a
    single-qubit error of the outer code; on a surface-code boundary the same
    correlated error lies along the boundary (for example Z_u Z_l along the
    top row, which carries a logical Z), so `weight2="direct"` lowers the
    fault distance to (d + 1) / 2. A round has 1 + 4 steps here (four
    colours) against 1 + 3 in FTMDRCircuit.

    The circuit fault distance is d for every code and basis under sd6,
    purez (also with its non-Z components removed) and em3
    (tests/test_competitor_circuits.py). Choice of basis: Z errors commute
    with the CSS logical Z, so under Z-biased noise the CSS Z-basis memory
    only sees the preparation and readout flips and looks far better than
    the X-basis memory; compare codes in the basis that the dominant errors
    flip (X for CSS) or in both.

    Attributes
    ----------
    code : str Code name. d : int Distance. rounds : int Extraction rounds.
    noise : CircuitNoise Noise model. basis : str Logical basis ("X" or "Z",
    named after the CSS logical). layout : RotatedSurfaceLayout Patch.
    frame : Dict[int, str] Preparation and readout Pauli of every data qubit.
    logical : List[Tuple[str, int]] Observable. det_checks : List[int] Checks
    deterministic on the initial product state. ancilla_of : Dict[int, int]
    Ancilla of every check (native gates). layers : List[List[Tuple[int, int,
    str]]] (ancilla, data qubit, Pauli) of every gate layer.
    """

    # The noise and detector helpers of FTMDRCircuit, shared verbatim: they
    # only read `self.noise`, so both circuits apply the same channels.
    _idle = FTMDRCircuit._idle
    _idle_mr = FTMDRCircuit._idle_mr
    _memory = FTMDRCircuit._memory
    _xtalk = FTMDRCircuit._xtalk
    _detector = FTMDRCircuit._detector

    def __init__(self, code: str, distance: int, rounds: int, noise: CircuitNoise,
                 basis: str = "Z", weight2: str = "cat") -> None:
        if rounds < 1:
            raise ValueError("rounds must be at least 1.")
        if weight2 not in ("cat", "direct"):
            raise ValueError("weight2 must be 'cat' or 'direct'.")
        self.code = code
        self.d = distance
        self.rounds = rounds
        self.noise = noise
        self.basis = basis
        self.weight2 = weight2
        self.layout = RotatedSurfaceLayout(code, distance)
        self.n = self.layout.n
        self.checks = self.layout.checks
        self.frame = self.layout.frame(basis)
        self.logical = self.layout.logical(basis)
        if any(self.frame[q] != p for p, q in self.logical):
            raise RuntimeError("logical operator is not a product of frame Paulis")
        self.det_checks = [ci for ci, ch in enumerate(self.checks)
                           if all(self.frame[q] == p for p, q in ch["terms"])]
        self.ancilla_of = {ci: self.n + ci for ci in range(len(self.checks))}
        depth = 1 + max(t for ch in self.checks for t in ch["layer"].values())
        self.layers: List[List[Tuple[int, int, str]]] = [[] for _ in range(depth)]
        for ci, ch in enumerate(self.checks):
            for p, q in ch["terms"]:
                self.layers[ch["layer"][q]].append((self.ancilla_of[ci], q, p))

    def build(self) -> stim.Circuit:
        """
        Return the noisy, detector-annotated Stim circuit.
        """
        if getattr(self.noise, "native", "gates") == "pairs":
            return self._build_pairs()
        return self._build_gates()

    # ------------------------------------------------------------ gates
    def _build_gates(self) -> stim.Circuit:
        nz = self.noise
        n, m = self.n, len(self.checks)
        data = list(range(n))
        anc = [self.ancilla_of[ci] for ci in range(m)]
        all_q = data + anc
        center = [ch["center"] for ch in self.checks]
        c = stim.Circuit()
        for q in data:
            c.append("QUBIT_COORDS", [q], list(self.layout.coords[q]))
        for ci, a in enumerate(anc):
            c.append("QUBIT_COORDS", [a], list(center[ci]))
        for b in "XYZ":
            qs = [q for q in data if self.frame[q] == b]
            if qs:
                c.append(_RESET[b], qs)
                if nz.p_prep:
                    c.append(_FLIP[b], qs, nz.p_prep)
        c.append("TICK")
        rec: Dict[Tuple[int, int], int] = {}
        count = 0
        for r in range(self.rounds):
            c.append("RX", anc)
            if nz.p_prep:
                c.append("Z_ERROR", anc, nz.p_prep)
            if not nz.xtalk_per_mcmr:
                self._xtalk(c, data)
            self._idle_mr(c, data)
            if r == 0:
                self._memory(c, all_q)
            c.append("TICK")
            for layer in self.layers:
                pairs: List[int] = []
                by_type: Dict[str, List[int]] = {}
                for gate in ("CX", "CY", "CZ"):
                    targets = []
                    for a, q, p in sorted(layer):
                        if _CTRL[p] == gate:
                            targets += [a, q]
                    if targets:
                        c.append(gate, targets)
                        pairs += targets
                        by_type[gate] = targets
                if pairs and nz.p2_paulis is not None and nz.p2_rzz_frame:
                    for gate, targets in by_type.items():
                        c.append("PAULI_CHANNEL_2", targets,
                                 _to_gate_frame(nz.p2_paulis, gate))
                elif pairs and nz.p2_paulis is not None:
                    c.append("PAULI_CHANNEL_2", pairs, list(nz.p2_paulis))
                elif pairs and nz.p2:
                    c.append("DEPOLARIZE2", pairs, nz.p2)
                if pairs and nz.dress_1q and nz.p1:
                    c.append("DEPOLARIZE1", pairs, nz.p1)
                busy = set(pairs)
                self._idle(c, [q for q in all_q if q not in busy])
                self._memory(c, all_q)
                c.append("TICK")
            if nz.p_meas:
                c.append("Z_ERROR", anc, nz.p_meas)
            c.append("MX", anc)
            for ci in range(m):
                rec[(r, ci)] = count + ci
            count += m
            self._xtalk(c, data, len(anc))
            self._idle_mr(c, data)
            self._memory(c, data)
            if r == 0:
                for ci in self.det_checks:
                    self._detector(c, [rec[(0, ci)]], count, (*center[ci], 0))
            else:
                for ci in range(m):
                    self._detector(c, [rec[(r, ci)], rec[(r - 1, ci)]], count,
                                   (*center[ci], r))
            c.append("TICK")
        last = self.rounds - 1
        position = self._read_data(c, data, count)
        count += n
        for ci in self.det_checks:
            idx = [position[q] for _, q in self.checks[ci]["terms"]] + [rec[(last, ci)]]
            self._detector(c, idx, count, (*center[ci], self.rounds))
        c.append("OBSERVABLE_INCLUDE",
                 [stim.target_rec(position[q] - count) for _, q in self.logical], 0)
        return c

    def _read_data(self, c: stim.Circuit, data: Sequence[int], count: int) -> Dict[int, int]:
        """
        Destructive data readout in the frame; returns the record index of every qubit.
        """
        nz = self.noise
        position: Dict[int, int] = {}
        for b in "XYZ":
            qs = [q for q in data if self.frame[q] == b]
            if qs:
                if nz.p_meas:
                    c.append(_FLIP[b], qs, nz.p_meas)
                c.append(_MEAS[b], qs)
                for k, q in enumerate(qs):
                    position[q] = count + k
                count += len(qs)
        return position

    # ------------------------------------------------------------ pairs
    def plaquette_colors(self) -> Dict[int, int]:
        """
        Colour of every check measured with a cat state.

        The cells are coloured (x mod 2) + 2 (y mod 2), so two cells that
        share a data qubit have different colours. Colours that no cat check
        uses are dropped and the rest renumbered in order.
        """
        cats = self._cat_checks()
        used = sorted({self.checks[ci]["color"] for ci in cats})
        rank = {col: k for k, col in enumerate(used)}
        return {ci: rank[self.checks[ci]["color"]] for ci in cats}

    def cat_tree(self, ci: int):
        """
        Star of Z_a Z_a' measurements that joins the ancillas of check ci.

        Ancilla k couples to the k-th qubit of `checks[ci]["cat"]`; ancilla 0
        is the centre and edge k - 1 (layer k - 1) joins it to leaf k. This is
        the tree that `FTMDRCircuit.cat_tree` builds when no two qubits of the
        check share a block. A wrong outcome on one edge acts like the check
        operator on its leaf, and the leaf order of `CAT_STAR` keeps the only
        two-qubit equivalent (wrong parities on the first two edges) away from
        the direction of the logical operators. Returns (edges, layer,
        parent) in the format of `FTMDRCircuit.cat_tree`.
        """
        w = len(self.checks[ci]["cat"])
        edges = [(0, k) for k in range(1, w)]
        layer = list(range(w - 1))
        parent = [-1] + [0] * (w - 1)
        if layer and max(layer) > 2:
            raise RuntimeError("cat tree needs more than three layers")
        return edges, layer, parent

    def _cat_checks(self) -> List[int]:
        m = len(self.checks)
        if self.weight2 == "direct":
            return [ci for ci in range(m) if len(self.checks[ci]["terms"]) > 2]
        return list(range(m))

    def _build_pairs(self) -> stim.Circuit:
        """
        Extraction with native two-qubit Pauli product measurements (EM3).

        Mirrors `FTMDRCircuit._build_pairs` step for step; see the class
        docstring. The detectors and the observable include the sign changes
        of each check by the couplings of the cat checks that overlap it.
        """
        nz = self.noise
        p = nz.p_mpp
        n, m = self.n, len(self.checks)
        data = list(range(n))
        checks = self.checks
        plaq = self._cat_checks()
        plaq_set = set(plaq)
        direct = [ci for ci in range(m) if ci not in plaq_set]
        pauli = {ci: {q: pp for pp, q in checks[ci]["terms"]} for ci in range(m)}
        # cat ancilla k of check ci couples to terms[ci][k]
        terms = {ci: list(checks[ci]["cat"]) for ci in plaq}
        seen: set = set()
        for ci in direct:
            if set(pauli[ci]) & seen:
                raise RuntimeError("directly measured checks must not overlap")
            seen |= set(pauli[ci])
        color = self.plaquette_colors()
        ncol = 1 + max(color.values(), default=-1)
        trees = {ci: self.cat_tree(ci) for ci in plaq}
        n_layers = 1 + max((max(trees[ci][1], default=-1) for ci in plaq), default=-1)
        # cat layers prepared during the colour steps of the previous round, and in step 0
        early = [L for L in range(n_layers) if L + 1 < ncol]
        late = [L for L in range(n_layers) if L + 1 >= ncol]
        if len(late) > 1:
            raise RuntimeError("the cat states need more layers than the colour steps leave")
        nxt = n
        cat: List[Dict[int, List[int]]] = [{}, {}]
        for b in (0, 1):
            for ci in plaq:
                w = len(terms[ci])
                cat[b][ci] = list(range(nxt, nxt + w))
                nxt += w
        per_layer = [sum(1 for ci in plaq for t in trees[ci][1] if t == L)
                     for L in range(n_layers)]
        n_pool = len(direct) + max(per_layer, default=0)
        pool = list(range(nxt, nxt + n_pool))

        c = stim.Circuit()
        for q in data:
            c.append("QUBIT_COORDS", [q], list(self.layout.coords[q]))
        for b in (0, 1):
            for ci in plaq:
                cx, cy = checks[ci]["center"]
                for k, a in enumerate(cat[b][ci]):
                    qx, qy = self.layout.coords[terms[ci][k]]
                    c.append("QUBIT_COORDS", [a], [(cx + qx) / 2, (cy + qy) / 2, b + 1])
        for k, f in enumerate(pool):
            c.append("QUBIT_COORDS", [f], [k, -2, 3])

        count = 0
        rec_m: Dict[Tuple[int, int], List[int]] = {}
        rec_z: Dict[Tuple[int, int, int], int] = {}
        rec_c: Dict[Tuple[int, int, int], int] = {}
        combos = [(a, b2, f) for a in "IXYZ" for b2 in "IXYZ" for f in (0, 1)][1:]
        live = set(data)

        def noisy_mpp(q1, p1, q2, p2, f):
            """MPP P1 P2 with the correlated EM3 error, using the flip qubit f."""
            nonlocal count
            c.append("R", [f])
            if p:
                acc = 0.0
                for j, (e1, e2, ff) in enumerate(combos):
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
            """Z readout of coupled ancillas; items are (round, check, k, ancilla)."""
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
            qs = [q for q in data if self.frame[q] == bb]
            if qs:
                c.append(_RESET[bb], qs)
                if nz.p_prep:
                    c.append(_FLIP[bb], qs, nz.p_prep)
        used = data + reset_bank(0)
        end_step(set(used))
        for L in early:
            end_step(set(cat_layer(0, 0, L)))

        pending: List[Tuple[int, int, int, int]] = []
        for r in range(self.rounds):
            b = r % 2
            more = r < self.rounds - 1
            # step 0: direct pair measurements, the cat layers left for this
            # step, readout of the last colour of the previous round
            used = []
            for fi, ci in enumerate(direct):
                (pu, u), (pl, l) = checks[ci]["terms"]
                rec_m[(r, ci)] = [noisy_mpp(u, pu, l, pl, pool[fi])]
                used += [u, l]
            for L in late:
                used += cat_layer(b, r, L, fi0=len(direct))
            used += readout(pending)
            pending = []
            end_step(set(used))
            # one step per colour: couplings, readout of the previous colour,
            # reset and cat preparation of the other bank
            for col in range(ncol):
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
                    elif col - 1 in early:
                        used += cat_layer(1 - b, r + 1, col - 1)
                end_step(set(used))
        end_step(set(readout(pending)))

        # ---------------------------------------------------------- signs
        step = {ci: (color[ci] + 1 if ci in plaq_set else 0) for ci in range(m)}
        support = {ci: set(pauli[ci]) for ci in range(m)}
        touching = {ci: [a for a in plaq if a != ci and support[a] & support[ci]]
                    for ci in range(m)}
        depth = {}
        for ci in plaq:
            parent = trees[ci][2]
            dd = []
            for k in range(len(parent)):
                h, x = 0, k
                while parent[x] >= 0:
                    x = parent[x]
                    h += 1
                dd.append(h)
            depth[ci] = dd

        def corr(rr, a_ci, op):
            """Records by which cat check a_ci in round rr flips the sign of the Pauli `op`."""
            Q = {k for k, q in enumerate(terms[a_ci]) if q in op and op[q] != pauli[a_ci][q]}
            if not Q:
                return []
            if len(Q) % 2:
                raise RuntimeError("operator anticommutes with a check")
            edges, _, parent = trees[a_ci]
            below = {k: int(k in Q) for k in range(len(terms[a_ci]))}
            for k in sorted(below, key=lambda k: -depth[a_ci][k]):
                if parent[k] >= 0:
                    below[parent[k]] += below[k]
            out = [rec_z[(rr, a_ci, k)] for k in sorted(Q)]
            for e, (_, k2) in enumerate(edges):
                if below[k2] % 2:
                    out.append(rec_c[(rr, a_ci, e)])
            return out

        def between(ci, r0, r1):
            s = step[ci]
            out = []
            for a_ci in touching[ci]:
                sa = step[a_ci]
                for rr in range(r0, r1 + 1):
                    if (rr == r0 and sa <= s) or (rr == r1 and sa >= s):
                        continue
                    out.append((rr, a_ci))
            return out

        def xor(lists):
            acc: set = set()
            for lst in lists:
                for x in lst:
                    acc ^= {x}
            return sorted(acc)

        center = [ch["center"] for ch in checks]
        dets = []
        for ci in self.det_checks:
            s = step[ci]
            idx = [rec_m[(0, ci)]]
            idx += [corr(0, a, pauli[ci]) for a in touching[ci] if step[a] < s]
            dets.append((xor(idx), (*center[ci], 0)))
        for r in range(1, self.rounds):
            for ci in range(m):
                lst = [rec_m[(r, ci)], rec_m[(r - 1, ci)]]
                lst += [corr(rr, a, pauli[ci]) for rr, a in between(ci, r - 1, r)]
                dets.append((xor(lst), (*center[ci], r)))
        last = self.rounds - 1

        def end_value(ci):
            s = step[ci]
            return xor([rec_m[(last, ci)]] + [corr(last, a, pauli[ci])
                                              for a in touching[ci] if step[a] > s])

        position = self._read_data(c, data, count)
        count += n
        for ci in self.det_checks:
            idx = [[position[q] for q in pauli[ci]], end_value(ci)]
            dets.append((xor(idx), (*center[ci], self.rounds)))
        lx = {q: pp for pp, q in self.logical}
        obs = [[position[q] for q in lx]]
        # every coupled check that overlaps the logical operator can flip it
        for rr in range(self.rounds):
            for a in plaq:
                if support[a] & set(lx):
                    obs.append(corr(rr, a, lx))
        obs = xor(obs)
        for idx, coords in dets:
            self._detector(c, idx, count, coords)
        c.append("OBSERVABLE_INCLUDE", [stim.target_rec(i - count) for i in obs], 0)
        return c


# ---------------------------------------------------------------- honeycomb
# Layout of the periodic honeycomb code of Gidney, Newman, Fowler and Broughton,
# "A Fault-Tolerant Honeycomb Memory", Quantum 5, 605 (2021), ported from their
# reference implementation (github.com/Strilanc/honeycomb_threshold, Apache-2.0,
# src/honeycomb_layout.py and src/honeycomb_circuit.py). Coordinates are complex
# numbers x + i y; hexagon centres have even x and data qubits odd x.
_HC_EDGE_TYPES = ((2 - 3j, 1 - 1j), (2 + 3j, 1 + 1j), (4 + 0j, 1 + 0j))  # (hex to hex, hex to qubit)
_HC_FIRST = ((1 - 1j, 1 + 0j), (1 + 1j, -1 + 1j), (-1 + 0j, -1 - 1j))
_HC_SECOND = ((-1 - 1j, 1 - 1j), (1 + 0j, 1 + 1j), (-1 + 1j, -1 + 0j))
_HC_OBS_H = ("XXXX", "X__X", "Z__Z", "_ZZ_", "_YY_", "YYYY")
_HC_OBS_V = ("_ZZ", "_YY", "YY_", "XX_", "X_X", "Z_Z")

Edge = Tuple[complex, complex, complex]  # (left, right, centre)


def _sorted_complex(xs) -> List[complex]:
    return sorted(xs, key=lambda v: (v.real, v.imag))


class HoneycombLayout:
    """
    Periodic honeycomb patch with `d` data-qubit columns.

    The data qubits sit on the vertices of a honeycomb lattice on a torus of
    `d` x `h` data qubits, h = 6 ceil(d / 4) (the smallest multiple of 6 with
    2 h / 3 >= d), so the distance against single-qubit errors, min(d, 2 h / 3)
    in Gidney et al., is d. The hexagons are coloured 0, 1, 2; an edge of colour
    c joins two hexagons of colour c and is the check P P with P = "XYZ"[c]
    (Gidney et al.'s variant: the Pauli type of an edge is its colour, so the
    plaquette of a colour-c hexagon is ("XYZ"[c])^6). Sub-round k measures
    every edge of colour k mod 3; the edges of one colour cover every qubit
    once.

    Attributes
    ----------
    d : int Width (code distance). height : int Height. n : int Number of data
    qubits. coords : List[complex] Data qubit positions (index = position in
    the list). hexes : List[List[complex]] Hexagon centres by colour. edges :
    List[List[Edge]] Edges by colour, each (left, right, centre).
    """

    def __init__(self, d: int) -> None:
        if d < 4 or d % 2:
            raise ValueError("the honeycomb code needs an even d >= 4.")
        self.d = d
        self.height = 6 * math.ceil(d / 4)
        self.tile_width = d // 2
        self.tile_height = self.height // 6
        self.coord_width = 4.0 * self.tile_width
        self.coord_height = 6.0 * self.tile_height
        colour: Dict[complex, int] = {}
        for row in range(3 * self.tile_height):
            for col in range(2 * self.tile_width):
                colour[self.wrap(row * 2j + 2 * col - 1j * (col % 2))] = (-row - col % 2) % 3
        self.hexes = [_sorted_complex(h for h, c in colour.items() if c == r) for r in range(3)]
        self.edges: List[List[Edge]] = []
        for r in range(3):
            es = [self._edge(h + dq, h + dh - dq) for h in self.hexes[r] for dh, dq in _HC_EDGE_TYPES]
            self.edges.append(sorted(es, key=lambda e: (e[2].real, e[2].imag)))
        self.coords = _sorted_complex({q for es in self.edges for e in es for q in e[:2]})
        self.index = {q: i for i, q in enumerate(self.coords)}
        self.n = len(self.coords)
        if self.n != d * self.height:
            raise RuntimeError("wrong number of data qubits")

    def wrap(self, c: complex) -> complex:
        return complex(c.real % self.coord_width, c.imag % self.coord_height)

    @staticmethod
    def _parity(q: complex) -> bool:
        return (q.real // 2 + q.imag) % 2 != 0

    def _edge(self, a: complex, b: complex) -> Edge:
        center = self.wrap((a + b) / 2)
        a, b = self.wrap(a), self.wrap(b)
        if (self._parity(a), a.real, a.imag) > (self._parity(b), b.real, b.imag):
            a, b = b, a
        return (a, b, center)

    def first_edges(self, h: complex) -> List[Edge]:
        """Edges of colour c + 1 around the hexagon h of colour c."""
        return [self._edge(h + a, h + b) for a, b in _HC_FIRST]

    def second_edges(self, h: complex) -> List[Edge]:
        """Edges of colour c + 2 around the hexagon h of colour c."""
        return [self._edge(h + a, h + b) for a, b in _HC_SECOND]

    def qubits_around(self, h: complex) -> List[complex]:
        return _sorted_complex(self.wrap(h + s * dq) for _, dq in _HC_EDGE_TYPES for s in (-1, 1))

    def observable(self, obs: str, sub_round: int) -> Tuple[str, List[complex]]:
        """
        Basis and qubits of the logical observable `obs` ("H" or "V") just before `sub_round`.
        """
        if obs == "H":
            pattern = _HC_OBS_H[sub_round % 6] * self.tile_width
            qs = sorted((q for q in self.coords if q.imag in (0, 1)),
                        key=lambda q: (q.real, (1 + q.imag + q.real // 2) % 2))
        elif obs == "V":
            pattern = _HC_OBS_V[sub_round % 6] * (2 * self.tile_height)
            qs = _sorted_complex(q for q in self.coords if q.real == 1)
        else:
            raise ValueError("obs must be 'H' or 'V'.")
        basis, = set(pattern) - {"_"}
        return basis, [q for ch, q in zip(pattern, qs) if ch != "_"]

    def observable_edges(self, obs: str) -> set:
        """Edges whose outcomes are multiplied into the observable `obs`."""
        all_edges = [e for es in self.edges for e in es]
        if obs == "H":
            return {e for e in all_edges if e[0].imag in (0, 1) and e[1].imag in (0, 1)}
        return {e for e in all_edges if e[0].real == e[1].real == 1}


class _Prev:
    """The record of `key` `offset` entries before the last one (port of Gidney's Prev)."""

    def __init__(self, key, offset: int = 1) -> None:
        self.key = key
        self.offset = offset


class _Tracker:
    """
    Measurement record bookkeeping (port of Gidney et al.'s MeasurementTracker).

    Every key has a history of record sets; `None` marks an unknown value
    (an obstacle), an empty set a value known to be +1.
    """

    def __init__(self) -> None:
        self.history = collections.defaultdict(list)
        self.t = 0

    def measure(self, *keys) -> None:
        for key in keys:
            self.history[key].append(frozenset([self.t]))
            self.t += 1

    def dummies(self, *keys, obstacle: bool = False) -> None:
        for key in keys:
            self.history[key].append(None if obstacle else frozenset())

    def group(self, *keys, key) -> None:
        self.history[key].append(self.times(*keys))

    def times(self, *keys):
        out = frozenset()
        for k in keys:
            back = 1
            if isinstance(k, _Prev):
                back += k.offset
                k = k.key
            v = self.history[k][-back]
            if v is None:
                return None
            out ^= v
        return out

    def targets(self, *keys):
        ts = self.times(*keys)
        if ts is None:
            return None
        return [stim.target_rec(t - self.t) for t in sorted(ts)]


class HoneycombCircuit:
    """
    Memory experiment of the periodic honeycomb (Floquet) code with the noise of `FTMDRCircuit`.

    Construction of Gidney et al. (2021): every data qubit is prepared in the
    basis of the protected observable ("H" for `basis="X"`, "V" for
    `basis="Z"`), `3 rounds` sub-rounds measure the edges of colours 0, 1, 2,
    0, ... (pairs XX, YY, ZZ), and the data qubits are read out in the basis
    of the observable at the end. Each detector compares two consecutive
    values of a plaquette, each value being the product of the edges of the
    other two colours around it measured in two consecutive sub-rounds; the
    edges along the observable path are multiplied into the observable as
    they are measured. A round (three sub-rounds) infers every plaquette once,
    as a surface-code round does.

    Noise conventions (as `CompetitorCircuit` and `FTMDRCircuit`):

    - Pair measurements (`noise.native == "pairs"`, EM3): every edge is a
      weight-2 check and is measured directly by one MPP P_u P_v with the
      EM3 error of FTMDRCircuit's direct link measurement (with probability
      p_mpp a uniform element of {I,X,Y,Z}^2 x {flip, no flip}, the flip
      through an auxiliary qubit in |0> included in the MPP). Every data
      qubit takes part in one MPP per sub-round, so there is no idle step.
    - Native gates: one ancilla per edge of the sub-round, reset in |+>
      (RX, Z_ERROR p_prep), controlled P on the left then the right qubit of
      its edge (ancilla control, two gate layers), read out in the X basis
      (Z_ERROR p_meas, MX), with the reset, gate-layer, idle, dressing,
      memory and crosstalk channels of `CompetitorCircuit`. Each sub-round is
      reset step, two gate layers and measurement step, like a round of
      FTMDRCircuit (no pipelining of consecutive sub-rounds).
    - Data preparation R / RX / RY with the flip p_prep, readout with the flip
      p_meas, as in FTMDRCircuit.

    The detector count and circuit fault distance equal those of the reference
    circuits of Gidney et al. at every tested size (d = 4, 6, 8). The fault
    distance is below the code distance d: about d / 2 under EM3, where a
    pair-measurement fault acts like a two-qubit error (d = 4, 6, 8: 2, 4, 4
    for "H" and 2, 3, 4 for "V"), and 3, 6, 6 ("H") and 4, 6, 8 ("V") with
    native gates. Compare it with other codes at equal qubit count or logical
    error rate rather than at equal d.

    Attributes
    ----------
    d : int Distance (width). rounds : int Rounds of three sub-rounds.
    noise : CircuitNoise Noise model. basis : str "X" (observable "H") or "Z"
    (observable "V"). layout : HoneycombLayout Lattice.
    """

    _idle = FTMDRCircuit._idle
    _idle_mr = FTMDRCircuit._idle_mr
    _memory = FTMDRCircuit._memory
    _xtalk = FTMDRCircuit._xtalk

    def __init__(self, distance: int, rounds: int, noise: CircuitNoise, basis: str = "Z") -> None:
        if rounds < 1:
            raise ValueError("rounds must be at least 1.")
        if basis not in BASES:
            raise ValueError(f"basis must be one of {BASES}.")
        self.d = distance
        self.rounds = rounds
        self.noise = noise
        self.basis = basis
        self.obs = "H" if basis == "X" else "V"
        self.layout = HoneycombLayout(distance)
        self.n = self.layout.n

    def build(self) -> stim.Circuit:
        """
        Return the noisy, detector-annotated Stim circuit.
        """
        lay, nz = self.layout, self.noise
        pairs = getattr(nz, "native", "gates") == "pairs"
        n = self.n
        data = list(range(n))
        # one ancilla (native gates) or flip qubit (pairs) per edge of a sub-round
        extra = list(range(n, n + n // 2))
        mt = _Tracker()
        c = stim.Circuit()
        for q in data:
            c.append("QUBIT_COORDS", [q], [lay.coords[q].real, lay.coords[q].imag])
        for k, a in enumerate(extra):
            c.append("QUBIT_COORDS", [a], [k, -2])

        def detector(*keys, coords):
            tg = mt.targets(*keys)
            if tg is not None:
                c.append("DETECTOR", tg, coords)

        init_basis = lay.observable(self.obs, 0)[0]
        c.append(_RESET[init_basis], data)
        if nz.p_prep:
            c.append(_FLIP[init_basis], data, nz.p_prep)
        c.append("TICK")
        edge_init = "XYZ".index(init_basis)
        half_init = (edge_init - 1) % 3
        for r in range(3):
            # the edges and plaquettes of the preparation basis start known; the
            # plaquettes of the previous colour start with their first half known
            mt.dummies(*[("e", e) for e in lay.edges[r]], obstacle=r != edge_init)
            mt.dummies(*[("h", h) for h in lay.hexes[r]], obstacle=r != edge_init)
            mt.dummies(*[("half", h) for h in lay.hexes[r]], obstacle=r != half_init)
        obs_edges = lay.observable_edges(self.obs)
        n_sub = 3 * self.rounds
        for k in range(n_sub):
            col = k % 3
            p = "XYZ"[col]
            edges = lay.edges[col]
            if pairs:
                self._pair_sub_round(c, edges, p, extra)
            else:
                self._gate_sub_round(c, edges, p, data, extra, first=k == 0)
            mt.measure(*[("m", e) for e in edges])
            for e in edges:
                mt.group(_Prev(("m", e), 0), key=("e", e))
            tg = mt.targets(*[("e", e) for e in edges if e in obs_edges])
            if tg:
                c.append("OBSERVABLE_INCLUDE", tg, 0)
            if k == 0 and init_basis == "X":
                for e in edges:
                    detector(("e", e), coords=[e[2].real, e[2].imag, k])
            for h in lay.hexes[(k - 1) % 3]:
                mt.group(*[("e", e) for e in lay.first_edges(h)], key=("half", h))
            for h in lay.hexes[(k - 2) % 3]:
                mt.group(*[("e", e) for e in lay.second_edges(h)], ("half", h), key=("h", h))
                detector(("h", h), _Prev(("h", h)), coords=[h.real, h.imag, k])
            c.append("TICK")
        # readout of the data in the basis of the observable
        obs_basis, obs_qubits = lay.observable(self.obs, n_sub)
        if nz.p_meas:
            c.append(_FLIP[obs_basis], data, nz.p_meas)
        c.append(_MEAS[obs_basis], data)
        mt.measure(*[("q", q) for q in lay.coords])
        last_basis = "XYZ"[(n_sub - 1) % 3]
        if last_basis == obs_basis:
            for e in lay.edges["XYZ".index(obs_basis)]:
                mt.group(("q", e[0]), ("q", e[1]), key=("e", e))
                detector(("e", e), _Prev(("e", e)), coords=[e[2].real, e[2].imag, n_sub])
        else:
            other, = set("XYZ") - {last_basis, obs_basis}
            for h in lay.hexes["XYZ".index(other)]:
                mt.group(*[("q", q) for q in lay.qubits_around(h)], ("half", h), key=("h", h))
                detector(("h", h), _Prev(("h", h)), coords=[h.real, h.imag, n_sub])
        for h in lay.hexes["XYZ".index(obs_basis)]:
            mt.group(*[("q", q) for q in lay.qubits_around(h)], key=("h", h))
            detector(("h", h), _Prev(("h", h)), coords=[h.real, h.imag, n_sub])
        c.append("OBSERVABLE_INCLUDE", mt.targets(*[("q", q) for q in obs_qubits]), 0)
        return c

    def _pair_sub_round(self, c: stim.Circuit, edges: Sequence[Edge], p: str,
                        flips: Sequence[int]) -> None:
        """One EM3 step: a noisy MPP P_u P_v per edge (FTMDRCircuit's direct link measurement)."""
        pm = self.noise.p_mpp
        idx = self.layout.index
        combos = [(a, b2, f) for a in "IXYZ" for b2 in "IXYZ" for f in (0, 1)][1:]
        for e, f in zip(edges, flips):
            q1, q2 = idx[e[0]], idx[e[1]]
            c.append("R", [f])
            if pm:
                acc = 0.0
                for j, (e1, e2, ff) in enumerate(combos):
                    tg = []
                    if e1 != "I":
                        tg.append(stim.target_pauli(q1, e1))
                    if e2 != "I":
                        tg.append(stim.target_pauli(q2, e2))
                    if ff:
                        tg.append(stim.target_x(f))
                    c.append("E" if j == 0 else "ELSE_CORRELATED_ERROR", tg,
                             (pm / 32.0) / (1.0 - acc))
                    acc += pm / 32.0
            c.append("MPP", [stim.target_pauli(q1, p), stim.target_combiner(),
                             stim.target_pauli(q2, p), stim.target_combiner(), stim.target_z(f)])
        # every data qubit is in one pair measurement: no idle qubits in this step

    def _gate_sub_round(self, c: stim.Circuit, edges: Sequence[Edge], p: str,
                        data: Sequence[int], anc: Sequence[int], first: bool) -> None:
        """Reset step, two gate layers and measurement step for one colour of edges."""
        nz = self.noise
        idx = self.layout.index
        anc = list(anc[:len(edges)])
        all_q = list(data) + anc
        c.append("RX", anc)
        if nz.p_prep:
            c.append("Z_ERROR", anc, nz.p_prep)
        if not nz.xtalk_per_mcmr:
            self._xtalk(c, data)
        self._idle_mr(c, data)
        if first:
            self._memory(c, all_q)
        c.append("TICK")
        gate = _CTRL[p]
        for side in (0, 1):
            targets = [t for a, e in zip(anc, edges) for t in (a, idx[e[side]])]
            c.append(gate, targets)
            if nz.p2_paulis is not None and nz.p2_rzz_frame:
                c.append("PAULI_CHANNEL_2", targets, _to_gate_frame(nz.p2_paulis, gate))
            elif nz.p2_paulis is not None:
                c.append("PAULI_CHANNEL_2", targets, list(nz.p2_paulis))
            elif nz.p2:
                c.append("DEPOLARIZE2", targets, nz.p2)
            if nz.dress_1q and nz.p1:
                c.append("DEPOLARIZE1", targets, nz.p1)
            busy = set(targets)
            self._idle(c, [q for q in all_q if q not in busy])
            self._memory(c, all_q)
            c.append("TICK")
        if nz.p_meas:
            c.append("Z_ERROR", anc, nz.p_meas)
        c.append("MX", anc)
        self._xtalk(c, data, len(anc))
        self._idle_mr(c, data)
        self._memory(c, data)


def competitor_circuit(code: str, d: int, rounds: int, noise: CircuitNoise,
                       basis: str = "Z", **kwargs) -> stim.Circuit:
    """
    Memory-experiment circuit of `code` with the noise of FTMDRCircuit.

    `code` is one of `CODES` (rotated surface codes, `CompetitorCircuit`) or
    `FLOQUET_CODES` (``honeycomb``, `HoneycombCircuit`, even d >= 4, `rounds`
    rounds of three sub-rounds). `basis` names the logical that is prepared
    and read out ("X" or "Z"): the CSS logical, mapped through the code's
    Clifford for ``xzzx`` and ``xy``, and the observable "H" ("X") or "V"
    ("Z") for ``honeycomb``. Extra keyword arguments go to
    `CompetitorCircuit` (`weight2` for the pair-measurement circuits).
    """
    if code in FLOQUET_CODES:
        return HoneycombCircuit(d, rounds, noise, basis=basis, **kwargs).build()
    return CompetitorCircuit(code, d, rounds, noise, basis=basis, **kwargs).build()
