"""Threshold of FT-MDR as a function of the number of extraction rounds.

For every odd distance and every round count r (fixed values and r = d) the
script samples the SD6 circuit at a list of physical error rates and decodes
with PyMatching on the frame-deterministic (S_0) detectors. Results are
appended to a sinter CSV, so an interrupted run resumes where it stopped.

The paper uses two passes:

    # coarse pass, 2e-3 <= p <= 0.1
    python scripts/run_rounds_threshold_sweep.py --out coarse.csv --pass coarse \
        --distances 3 5 7 9 11 13 15 17 19
    # refinement around the crossing points
    python scripts/run_rounds_threshold_sweep.py --out fine.csv --pass fine \
        --distances 3 5 7 9 11 13 15 17 19 --max-shots 60000 --max-errors 1500

`paper/analysis/rounds_thresholds.py` pools the CSV files, finds the crossing
points and fits the finite-size scaling form.
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import sinter

from mdr.ft import CircuitNoise, FTMDRCircuit

ROUNDS = ["1", "2", "3", "4", "5", "6", "8", "10", "d"]
COARSE_P = [0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.01, 0.012, 0.015, 0.02,
            0.025, 0.03, 0.04, 0.05, 0.07, 0.1]
# windows that contain the crossing points of every pair of distances
FINE_WINDOWS = {"1": (0.012, 0.024), "2": (0.008, 0.016), "3": (0.007, 0.0135),
                "4": (0.0055, 0.0105), "5": (0.005, 0.0095), "6": (0.0048, 0.009),
                "8": (0.0042, 0.008), "10": (0.004, 0.0075), "d": (0.0034, 0.0058)}


def p_values(key: str, which: str) -> list:
    if which == "coarse":
        return COARSE_P
    lo, hi = FINE_WINDOWS[key]
    return [float(round(p, 6)) for p in np.geomspace(lo, hi, 8)]


def tasks_for(d: int, rounds: list, which: str):
    # building the frame basis is the expensive part, so it is done once per d
    base = FTMDRCircuit(d, 1, CircuitNoise.uniform(1e-3))
    for key in rounds:
        r = d if key == "d" else int(key)
        for p in p_values(key, which):
            ft = copy.copy(base)
            ft.rounds = r
            ft.noise = CircuitNoise.uniform(p)
            yield sinter.Task(
                circuit=ft.build(),
                json_metadata={"d": d, "r": r, "rmode": "d" if key == "d" else "fixed", "p": p,
                               "noise": "sd6", "pass": which},
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="sinter CSV file (appended, resumable)")
    ap.add_argument("--pass", dest="which", choices=["coarse", "fine"], default="coarse")
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7, 9, 11, 13, 15, 17, 19])
    ap.add_argument("--rounds", nargs="+", default=ROUNDS, help="round counts, 'd' for r = d")
    ap.add_argument("--max-shots", type=int, default=20_000)
    ap.add_argument("--max-errors", type=int, default=200)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    for d in args.distances:
        if d % 2 == 0 or d < 3:
            raise SystemExit("the XYZ^2 code needs odd d >= 3")
        t0 = time.time()
        tasks = list(tasks_for(d, args.rounds, args.which))
        print(f"d={d}: {len(tasks)} tasks built in {time.time() - t0:.0f}s", flush=True)
        sinter.collect(num_workers=args.workers, tasks=tasks, decoders=["pymatching"],
                       max_shots=args.max_shots, max_errors=args.max_errors,
                       save_resume_filepath=args.out, print_progress=False)
        print(f"d={d}: done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
