"""
exact_fault_distance.py
----------------------------------------------------------------------------
Exact circuit-level fault distance of the FT MDR circuits with CP-SAT.

A set of fault mechanisms of the detector error model, encoded as a binary
vector x, is an undetectable logical error when H x = 0 and L x = 1 (mod 2),
where H holds the detector sets and L the observable flips. The script asks
whether such an x with at most k faults exists. An INFEASIBLE answer for
k = d together with a solution for k = d + 1 proves fault distance d + 1.

    python scripts/exact_fault_distance.py --distance 5 --rounds 5 --max-faults 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mdr.ft import CircuitNoise, FTMDRCircuit  # noqa: E402


def dem_columns(circuit):
    """
    Distinct (detectors, observables) signatures of the fault mechanisms.
    """
    dem = circuit.detector_error_model(decompose_errors=False,
                                       approximate_disjoint_errors=True)
    cols = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        dets, obs = set(), set()
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets ^= {t.val}
            elif t.is_logical_observable_id():
                obs ^= {t.val}
        cols[(tuple(sorted(dets)), tuple(sorted(obs)))] = 1
    return list(cols), dem.num_detectors


def exists_logical_error(circuit, k, time_limit=3600.0, workers=1):
    """
    CP-SAT status of: an undetectable logical error with at most k faults.
    """
    from ortools.sat.python import cp_model

    cols, nd = dem_columns(circuit)
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x{j}") for j in range(len(cols))]
    by_det = [[] for _ in range(nd)]
    obs = []
    for j, (dets, o) in enumerate(cols):
        for dd in dets:
            by_det[dd].append(x[j])
        if 0 in o:
            obs.append(x[j])
    for vs in by_det:
        if vs:
            model.AddBoolXOr(vs + [model.NewConstant(1)])
    model.AddBoolXOr(obs)
    model.Add(sum(x) <= k)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = workers
    start = time.time()
    status = solver.Solve(model)
    return solver.StatusName(status), len(cols), time.time() - start


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", type=int, required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--final", default="frame", choices=["frame", "ideal"])
    ap.add_argument("--init", default="frame", choices=["frame", "plus"])
    ap.add_argument("--max-faults", type=int, required=True)
    ap.add_argument("--share-ancillas", type=int, default=0)
    ap.add_argument("--time-limit", type=float, default=3600.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--noise", default="sd6", choices=["sd6", "em3"],
                    help="em3 uses the pair-measurement extraction")
    args = ap.parse_args()
    detectors = "s0" if args.init == "frame" else "all"
    noise = CircuitNoise.em3(1e-3) if args.noise == "em3" else CircuitNoise.uniform(1e-3)
    circuit = FTMDRCircuit(args.distance, args.rounds, noise,
                           final=args.final, init=args.init,
                           detectors=detectors,
                           share_ancillas=args.share_ancillas).build()
    status, ncols, sec = exists_logical_error(circuit, args.max_faults,
                                              args.time_limit, args.workers)
    print(f"{args.noise} d={args.distance} r={args.rounds} final={args.final} "
          f"init={args.init}: undetectable logical error with <= "
          f"{args.max_faults} faults: {status} ({ncols} mechanisms, "
          f"{sec:.0f} s)")


if __name__ == "__main__":
    main()
