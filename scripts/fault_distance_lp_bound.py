"""
fault_distance_lp_bound.py
----------------------------------------------------------------------------
Lower bound on the circuit-level fault distance from a linear program.

Stim decomposes every fault mechanism of the S0 detector error model into at
most two edges of a matching graph. Give each edge a weight w_e >= 0 such that
the edges of every mechanism weigh at most one together. The edges of an
undetectable logical error F, counted modulo two, form an even subgraph that
flips the observable, so it contains a cycle that flips the observable and
weighs at most |F|. The minimum weight of such a cycle is therefore a lower
bound on the fault distance. The script maximizes this bound over w with
cutting planes and certifies the result after rescaling w.

    python scripts/fault_distance_lp_bound.py --distance 7 --rounds 7
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mdr.ft import CircuitNoise, FTMDRCircuit  # noqa: E402


def matching_graph(circuit):
    """Edges (u, v, obs) of the decomposed DEM and the edge sets of all mechanisms.

    The boundary is vertex `num_detectors`.
    """
    dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    nd = dem.num_detectors
    edges: dict[tuple[int, int, int], int] = {}
    mechs: set[frozenset[int]] = set()
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        comps: list[list] = [[]]
        for t in inst.targets_copy():
            if t.is_separator():
                comps.append([])
            else:
                comps[-1].append(t)
        eset: set[int] = set()
        for comp in comps:
            dets = sorted(t.val for t in comp if t.is_relative_detector_id())
            obs = sum(1 for t in comp if t.is_logical_observable_id()) % 2
            if not dets:
                if obs:
                    raise ValueError("a single fault flips the observable without detection")
                continue
            if len(dets) > 2:
                raise ValueError("decomposition left a component with more than two detectors")
            u, v = (dets[0], nd) if len(dets) == 1 else (dets[0], dets[1])
            eset ^= {edges.setdefault((u, v, obs), len(edges))}
        if eset:
            mechs.add(frozenset(eset))
    return nd, list(edges), [sorted(m) for m in mechs]


def min_odd_cycle(nd, elist, w, eps=1e-9):
    """Minimum weight of a closed walk that flips the observable, and its edges mod 2."""
    n = nd + 1
    rows, cols, vals = [], [], []
    for (u, v, o), we in zip(elist, w):
        for p in (0, 1):
            a, b = 2 * u + p, 2 * v + (p ^ o)
            rows += [a, b]
            cols += [b, a]
            vals += [we + eps, we + eps]
    g = coo_matrix((vals, (rows, cols)), shape=(2 * n, 2 * n)).tocsr()
    dist, pred = dijkstra(g, directed=True, indices=np.arange(n) * 2, return_predecessors=True)
    odd = dist[np.arange(n), np.arange(n) * 2 + 1]
    v = int(np.argmin(odd))
    path = [2 * v + 1]
    while path[-1] != 2 * v:
        path.append(int(pred[v, path[-1]]))
    index = {e: k for k, e in enumerate(elist)}
    used: set[int] = set()
    for a, b in zip(path, path[1:]):
        x, y = sorted((a // 2, b // 2))
        used ^= {index[(x, y, (a % 2) ^ (b % 2))]}
    return float(odd[v]), sorted(used)


def lp_bound(nd, elist, mechs, max_iter=5000, time_limit=3600.0):
    """Maximize the odd-cycle bound over admissible weights with cutting planes."""
    E = len(elist)
    A_m = np.zeros((len(mechs), E + 1))
    for i, m in enumerate(mechs):
        A_m[i, m] = 1.0
    cost = np.zeros(E + 1)
    cost[E] = -1.0
    cuts = []
    w = np.full(E, 0.5)
    best, best_w = 0.0, w.copy()
    t0 = time.time()
    for _ in range(max_iter):
        c, cyc = min_odd_cycle(nd, elist, w)
        if c > best:
            best, best_w = c, w.copy()
        row = np.zeros(E + 1)
        row[cyc] = -1.0
        row[E] = 1.0
        cuts.append(row)
        res = linprog(cost, A_ub=np.vstack([A_m, np.array(cuts)]),
                      b_ub=np.concatenate([np.ones(len(mechs)), np.zeros(len(cuts))]),
                      bounds=[(0, 1)] * E + [(0, None)], method="highs")
        w, t = res.x[:E], res.x[E]
        if t <= best + 1e-7 or time.time() - t0 > time_limit:
            break
    return best_w


def certify(nd, elist, mechs, w):
    """Rescale w so that every mechanism weighs at most one, then recompute the bound."""
    w = np.clip(np.asarray(w, float), 0.0, None)
    w = w / max(1.0, max(float(np.sum(w[m])) for m in mechs))
    c, _ = min_odd_cycle(nd, elist, w, eps=1e-12)
    return c - 1e-12 * len(elist)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--distance", type=int, required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--final", default="frame", choices=["frame", "ideal"])
    ap.add_argument("--time-limit", type=float, default=3600.0)
    args = ap.parse_args()
    circuit = FTMDRCircuit(args.distance, args.rounds, CircuitNoise.uniform(1e-3),
                           final=args.final, detectors="s0").build()
    nd, elist, mechs = matching_graph(circuit)
    w = lp_bound(nd, elist, mechs, time_limit=args.time_limit)
    bound = certify(nd, elist, mechs, w)
    print(f"d={args.distance} r={args.rounds} final={args.final}: edges={len(elist)} "
          f"mechanisms={len(mechs)} certified bound {bound:.9f}, "
          f"fault distance >= {int(np.ceil(bound - 1e-6))}")


if __name__ == "__main__":
    main()
