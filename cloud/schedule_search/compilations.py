"""
compilations.py
----------------------------------------------------------------------------
XYZ^2 compilations used by the schedule search, and the decoder glue.

- "s<idx>": FTMDRCircuit with `ExtractionSchedule.enumerate_depth(6)[idx]`
  (s1525 is BOTH_BASES_SCHEDULE, s910 is DEPTH6_SCHEDULE).
- names in CUSTOM: compilations that ExtractionSchedule cannot express,
  built here by subclassing FTMDRCircuit (src/mdr/ft is not edited).

Noise names are those of scripts/run_decoder_threshold_sweep.py plus the
diagnostics below (never hardware claims):

- "<base>_noidle": the base model with every idle location removed
  (p_idle = 0, which also removes the data idling of the reset and
  measurement steps). It bounds from above what any shorter or pipelined
  compilation with the same gates can gain.
- "<base>_nolayeridle": idle removed in the gate layers only (below).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mdr.ft import FTMDRCircuit  # noqa: E402
from mdr.ft.extraction_schedule import ExtractionSchedule  # noqa: E402
from mdr.ft.two_level_decoder import TwoLevelDecoder  # noqa: E402
from run_decoder_threshold_sweep import DECODERS, NOISE as BASE_NOISE  # noqa: E402

SCHEDULES = ExtractionSchedule.enumerate_depth(6)

NOISE = dict(BASE_NOISE)
for _b in ("sd6", "hyb", "sd6_il"):
    NOISE[f"{_b}_noidle"] = (lambda b: lambda v: replace(
        BASE_NOISE[b](v), p_idle=0.0, name=f"{b}_noidle"))(_b)

# idle removed in the gate layers only; data qubits still idle (at p) while
# the ancillas are reset and measured, as in every SD6 circuit of the other
# codes. This bounds what a deeper-pipelined or shorter hexagon schedule can
# gain against XZZX, whose data qubits never idle in its four gate layers.
for _b in ("sd6", "hyb"):
    NOISE[f"{_b}_nolayeridle"] = (lambda b: lambda v: replace(
        BASE_NOISE[b](v), p_idle=0.0, p_idle_mr=v, name=f"{b}_nolayeridle"))(_b)

CUSTOM = {}


def build_ft(comp: str, d: int, rounds: int, noise, logical: str) -> FTMDRCircuit:
    if comp.startswith("s"):
        sched = SCHEDULES[int(comp[1:])]
        return FTMDRCircuit(d, rounds, noise, final="frame", detectors="combined",
                            schedule=sched, logical=logical)
    return CUSTOM[comp](d, rounds, noise, logical)


class _Decoder(TwoLevelDecoder):
    """
    TwoLevelDecoder whose S_0 circuit is built by the compilation itself
    (`ft.s0_copy()`) when it is a subclass of FTMDRCircuit; the base class
    always rebuilds a plain FTMDRCircuit.
    """

    def _setup_columns(self) -> None:
        if not hasattr(self.ft, "s0_copy"):
            return super()._setup_columns()
        coords = self.circuit.get_detector_coordinates()
        nd = self.circuit.num_detectors
        from mdr.ft.two_level_decoder import GAUGE_PHASES, S0_PHASES
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
        self.s0_circuit = self.ft.s0_copy().build()
        s0_coords = self.s0_circuit.get_detector_coordinates()
        assert len(s0_coords) == len(self.s0_cols)
        for a, b in zip(self.s0_cols, range(len(s0_coords))):
            assert list(coords[int(a)]) == list(s0_coords[b])


def xyz2_decoder(ft: FTMDRCircuit, name: str) -> TwoLevelDecoder:
    return _Decoder(ft, **DECODERS[name])


# ---------------------------------------------------------------- hybrid
class HexOnlySchedule(ExtractionSchedule):
    """
    Schedule whose links are not gates (hybrid extraction: every link is a
    pair measurement outside the gate layers). Only the hexagons and
    boundary checks must pass the clash and interleaving rules, so sigma
    pairs that `enumerate_depth` drops for want of free link layers are
    allowed. `link` is a placeholder and never used.
    """

    def validate(self, geometry):
        sm = self.slot_map(geometry)
        keep = {ci for ci, ch in enumerate(geometry.checks) if ch["kind"] != "link"}
        per_qubit = {}
        for (ci, q), t in sm.items():
            if ci in keep:
                per_qubit.setdefault(q, []).append(t)
        for q, ts in per_qubit.items():
            if len(ts) != len(set(ts)):
                return False, f"data qubit {q} used twice in one layer"
        paulis = {ci: {q: p for p, q in geometry.checks[ci]["terms"]} for ci in keep}
        idx = sorted(keep)
        for x, a in enumerate(idx):
            ts = [sm[(a, q)] for q in paulis[a]]
            if len(ts) != len(set(ts)):
                return False, f"ancilla of check {a} used twice in one layer"
            for b in idx[x + 1:]:
                anti = [q for q in set(paulis[a]) & set(paulis[b]) if paulis[a][q] != paulis[b][q]]
                if sum(1 for q in anti if sm[(a, q)] < sm[(b, q)]) % 2:
                    return False, f"checks {a} and {b} violate interleaving"
        return True, "ok"


def hex_only_sigma_pairs():
    """
    Every (sigma_A, sigma_B) of depth 6 that passes the bulk hexagon rules
    (the rules of `enumerate_depth` without the link layers), in a fixed order.
    """
    import itertools
    pos = ("TL", "TRu", "TRl", "BR", "BLl", "BLu")
    perms = [p for p in itertools.permutations(range(6), 6) if (p[1] < p[5]) == (p[2] < p[4])]
    out = []
    for sa in perms:
        a = dict(zip(pos, sa))
        for sb in perms:
            b = dict(zip(pos, sb))
            ok = True
            for c, cp in ((a, b), (b, a)):
                if (c["TL"] < cp["BLl"]) != (c["TRu"] < cp["BR"]):
                    ok = False
                if (c["TRl"] < cp["TL"]) != (c["BR"] < cp["BLu"]):
                    ok = False
            if not ok:
                continue
            sets = ({b["BR"], a["TRu"], a["BLu"]}, {b["TL"], a["TRl"], a["BLl"]},
                    {a["BR"], b["TRu"], b["BLu"]}, {a["TL"], b["TRl"], b["BLl"]})
            if min(map(len, sets)) < 3:
                continue
            out.append(HexOnlySchedule(6, {"A": a, "B": b}, {0: (0, 1), 1: (0, 1)}))
    return out


HEX_PAIRS = None


def hex_pair(i: int) -> HexOnlySchedule:
    global HEX_PAIRS
    if HEX_PAIRS is None:
        HEX_PAIRS = hex_only_sigma_pairs()
    return HEX_PAIRS[i]


def link_step_options(schedule: ExtractionSchedule):
    """
    Steps where a hybrid link pair measurement can go, per block parity: the
    reset step "R", the measurement step "M", and every gate layer in which
    both vertices of the block are free and which does not fall between the
    two gates of a plaquette that anticommutes with XX on the block.
    """
    a, b = schedule.sigma["A"], schedule.sigma["B"]
    bulk = {
        0: ({b["BR"], a["TRu"], a["BLu"]}, {b["TL"], a["TRl"], a["BLl"]},
            [(a["TRu"], a["TRl"]), (a["BLu"], a["BLl"])]),
        1: ({a["BR"], b["TRu"], b["BLu"]}, {a["TL"], b["TRl"], b["BLl"]},
            [(b["TRu"], b["TRl"]), (b["BLu"], b["BLl"])]),
    }
    out = {}
    for par, (up, lo, anti) in bulk.items():
        free = [t for t in range(schedule.depth) if t not in up and t not in lo
                and all((t < x) == (t < y) for x, y in anti)]
        out[par] = ["R"] + free + ["M"]
    return out


class HybridLinkStep(FTMDRCircuit):
    """
    Hybrid extraction (noise.native == "hybrid") with a chosen step for the
    single pair measurement of the links of each block parity: `steps =
    (even, odd)`, each "R", "M" or a free gate layer (`link_step_options`).
    FTMDRCircuit always uses "R".
    """

    def __init__(self, d, rounds, noise, schedule, logical="X", steps=("R", "R"),
                 detectors="combined"):
        super().__init__(d, rounds, noise, schedule=schedule, final="frame",
                         detectors=detectors, logical=logical)
        if self.native != "hybrid" or self.link_reps != 1:
            raise ValueError("HybridLinkStep needs hybrid noise with link_reps = 1")
        opts = link_step_options(schedule)
        for par, st in enumerate(steps):
            if st not in opts[par]:
                raise ValueError(f"step {st} is not allowed for parity {par}: {opts[par]}")
        self.link_steps = {0: [steps[0]], 1: [steps[1]]}
        self._args = (d, rounds, noise, schedule, logical, tuple(steps))

    def s0_copy(self):
        d, r, nz, s, lg, st = self._args
        return HybridLinkStep(d, r, nz, s, lg, st, detectors="s0")


def _parse_step(x):
    return x if x in ("R", "M") else int(x)


def _custom(comp, d, rounds, noise, logical):
    # "h<i>_<even>_<odd>": hex-only sigma pair i, link steps per parity;
    # "s<i>_<even>_<odd>": enumerate_depth(6)[i] with the link steps.
    head, e, o = comp.split("_")
    sched = hex_pair(int(head[1:])) if head[0] == "h" else SCHEDULES[int(head[1:])]
    if noise.native != "hybrid":
        if head[0] == "h":
            raise ValueError("hex-only schedules need the hybrid extraction")
        return FTMDRCircuit(d, rounds, noise, final="frame", detectors="combined",
                            schedule=sched, logical=logical)
    return HybridLinkStep(d, rounds, noise, sched, logical,
                          (_parse_step(e), _parse_step(o)))


_build_ft_base = build_ft


def build_ft(comp: str, d: int, rounds: int, noise, logical: str) -> FTMDRCircuit:  # noqa: F811
    if "_" in comp:
        return _custom(comp, d, rounds, noise, logical)
    return _build_ft_base(comp, d, rounds, noise, logical)


# ---------------------------------------------------------------- split hexagons
_SPLIT = None


def split_schedule(i: int):
    """enumerate_split(5)[i] (split_hex.py), cached in data/split_schedules.pkl."""
    global _SPLIT
    if _SPLIT is None:
        import pickle
        f = HERE / "data" / "split_schedules.pkl"
        from split_hex import enumerate_split  # noqa: F401  (unpickling needs the class)
        if f.exists():
            _SPLIT = pickle.loads(f.read_bytes())
        else:
            _SPLIT = enumerate_split(5)
    return _SPLIT[i]


_build_ft_prev = build_ft


def build_ft(comp: str, d: int, rounds: int, noise, logical: str) -> FTMDRCircuit:  # noqa: F811
    # "x<i>": depth-3 split-hexagon hybrid compilation enumerate_split(5)[i];
    # "y<i>": the same with two pipelined ancilla banks (no bulk data idling)
    if comp.startswith("y"):
        from split_hex import PipelinedSplitHexCircuit
        return PipelinedSplitHexCircuit(d, rounds, noise, split_schedule(int(comp[1:])), logical)
    if comp.startswith("x"):
        from split_hex import SplitHexCircuit
        return SplitHexCircuit(d, rounds, noise, split_schedule(int(comp[1:])), logical)
    return _build_ft_prev(comp, d, rounds, noise, logical)
