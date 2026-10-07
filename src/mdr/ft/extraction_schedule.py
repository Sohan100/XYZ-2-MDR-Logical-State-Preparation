"""

extraction_schedule.py
----------------------------------------------------------------------------
Translation-invariant interleaved syndrome-extraction schedules.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Dict, List, Tuple

from .xyz2_geometry import XYZ2Geometry


@dataclass(frozen=True)
class ExtractionSchedule:
    """
    Gate-layer assignment for one round of XYZ^2 syndrome extraction.

    Every check uses its own ancilla prepared in `|+>`, applies one
    controlled-Pauli (ancilla control, data target) per support qubit, and is
    read out in the X basis. `sigma[cls][pos]` is the layer in which a class
    `cls` hexagon touches position `pos`. Boundary checks inherit the layers
    of their virtual cell. `link[parity]` gives the layers of the upper and
    lower CNOTs of an XX link on a block of that parity.

    A schedule is valid when no qubit is used twice in one layer and every
    pair of overlapping checks satisfies the interleaving rule: on the shared
    qubits where their Paulis anticommute, check `a` must act before check `b`
    an even number of times. Valid schedules measure the intended operators.
    Hook errors then depend on the order inside each hexagon, which is what
    the fault-distance search selects for.

    Attributes
    ----------
    depth : int Number of entangling layers per round. sigma : Dict[str,
    Dict[str, int]] Layer of each hexagon position, per class (`A`, `B`).
    link : Dict[int, Tuple[int, int]] `(upper, lower)` layers of the link
    CNOTs, per block parity.
    """

    depth: int
    sigma: Dict[str, Dict[str, int]]
    link: Dict[int, Tuple[int, int]]

    def slot_map(self, geometry: XYZ2Geometry) -> Dict[Tuple[int, int], int]:
        """
        Return `(check index, qubit) -> layer` for one round.
        """
        out: Dict[Tuple[int, int], int] = {}
        for ci, ch in enumerate(geometry.checks):
            if ch["kind"] == "link":
                i, j = ch["block"]
                t_up, t_lo = self.link[(i + j) % 2]
                (_, q_up), (_, q_lo) = ch["terms"]
                out[(ci, q_up)] = t_up
                out[(ci, q_lo)] = t_lo
                continue
            i, j = ch["cell"]
            cls = "B" if (i + j) % 2 == 0 else "A"
            for _, q in ch["terms"]:
                out[(ci, q)] = self.sigma[cls][ch["positions"][q]]
        return out

    def validate(self, geometry: XYZ2Geometry) -> Tuple[bool, str]:
        """
        Check layer clashes and the interleaving rule on a concrete patch.
        """
        sm = self.slot_map(geometry)
        per_qubit: Dict[int, List[int]] = {}
        for (ci, q), t in sm.items():
            per_qubit.setdefault(q, []).append(t)
        for q, ts in per_qubit.items():
            if len(ts) != len(set(ts)):
                return False, f"data qubit {q} used twice in one layer"
        paulis = []
        for ci, ch in enumerate(geometry.checks):
            ts = [sm[(ci, q)] for _, q in ch["terms"]]
            if len(ts) != len(set(ts)):
                return False, f"ancilla of check {ci} used twice in one layer"
            paulis.append({q: p for p, q in ch["terms"]})
        for a in range(len(paulis)):
            for b in range(a + 1, len(paulis)):
                shared = set(paulis[a]) & set(paulis[b])
                anti = [q for q in shared if paulis[a][q] != paulis[b][q]]
                before = sum(1 for q in anti if sm[(a, q)] < sm[(b, q)])
                if before % 2:
                    return False, f"checks {a} and {b} violate interleaving"
        return True, "ok"

    @staticmethod
    def enumerate_depth(depth: int = 6) -> List["ExtractionSchedule"]:
        """
        List every translation-invariant schedule of the given depth that
        passes the local clash and interleaving rules (bulk rules only).
        """
        pos = XYZ2Geometry.POSITIONS
        slots = range(depth)
        perms = [
            p for p in itertools.permutations(slots, 6)
            if (p[1] < p[5]) == (p[2] < p[4])
        ]
        found: List[ExtractionSchedule] = []
        links = [(u, l) for u in slots for l in slots if u != l]
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
                even_up = {b["BR"], a["TRu"], a["BLu"]}
                even_lo = {b["TL"], a["TRl"], a["BLl"]}
                odd_up = {a["BR"], b["TRu"], b["BLu"]}
                odd_lo = {a["TL"], b["TRl"], b["BLl"]}
                if min(map(len, (even_up, even_lo, odd_up, odd_lo))) < 3:
                    continue
                l0 = [
                    (u, l) for u, l in links
                    if u not in even_up and l not in even_lo
                    and (u < a["TRu"]) == (l < a["TRl"])
                    and (u < a["BLu"]) == (l < a["BLl"])
                ]
                l1 = [
                    (u, l) for u, l in links
                    if u not in odd_up and l not in odd_lo
                    and (u < b["TRu"]) == (l < b["TRl"])
                    and (u < b["BLu"]) == (l < b["BLl"])
                ]
                for x in l0:
                    for y in l1:
                        found.append(ExtractionSchedule(
                            depth, {"A": a, "B": b}, {0: x, 1: y}
                        ))
        return found


# Depth-6 schedule selected by an exhaustive fault-distance scan over all
# 8928 valid translation-invariant depth-6 schedules (d = 3), then confirmed
# at d = 5 and d = 7. It reaches circuit-level fault distance d + 1 for the
# frame-initialised plus-state protocol.
DEPTH6_SCHEDULE = ExtractionSchedule(
    depth=6,
    sigma={
        "A": {"TL": 0, "BLu": 1, "TRu": 2, "BLl": 3, "TRl": 4, "BR": 5},
        "B": {"TL": 0, "TRu": 1, "TRl": 2, "BR": 3, "BLu": 4, "BLl": 5},
    },
    link={0: (0, 1), 1: (0, 1)},
)


# Depth-6 schedule for the memory of both conjugate logical operators. With
# DEPTH6_SCHEDULE the memory of Logical Y (FTMDRCircuit(logical="Y")) has
# circuit-level fault distance only (d + 1) / 2 (2 at d = 3, 3 at d = 5, SD6,
# exact CP-SAT), because its hooks line up with the Logical Y string. This
# schedule was selected among the 8928 valid depth-6 schedules: 164 of them
# keep fault distance d + 1 for Logical X and reach d for Logical Y at d = 3
# (CP-SAT, SD6, r = 3, S_0 detectors; none reaches d + 1 for Logical Y), 26
# of those were confirmed at d = 5 (Y >= 5, X >= 6), and among them this one
# has the lowest logical error rate in the weaker basis under SD6 and MWPM at
# p = 4e-3 (d = 7 and 9, r = d: X 3.3%/3.4%, Y 3.7%/3.3%; DEPTH6_SCHEDULE: X
# 2.6%/2.5%, Y 6.4%/6.9%). It is index 1525 of `enumerate_depth(6)`.
BOTH_BASES_SCHEDULE = ExtractionSchedule(
    depth=6,
    sigma={
        "A": {"TL": 0, "BLu": 1, "BLl": 2, "TRu": 3, "TRl": 4, "BR": 5},
        "B": {"TL": 0, "TRu": 1, "TRl": 2, "BLu": 3, "BR": 4, "BLl": 5},
    },
    link={0: (2, 3), 1: (0, 1)},
)
