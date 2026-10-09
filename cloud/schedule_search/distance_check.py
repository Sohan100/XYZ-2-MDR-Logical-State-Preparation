"""
distance_check.py
----------------------------------------------------------------------------
Fault distance (Stim undetectable-logical-error search, an upper bound) of
named compilations (compilations.build_ft) in both logical bases.

    python cloud/schedule_search/distance_check.py --d 7 --rounds 3 --k 3 \
        --jobs s1525_5_R:hyb s7402:sd6
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compilations import NOISE, build_ft  # noqa: E402
from distance_scan import fault_distance  # noqa: E402

A = {}


def run(job):
    comp, noise, lg = job
    t = time.time()
    c = build_ft(comp, A["d"], A["rounds"], NOISE[noise](1e-3), lg).build()
    return comp, noise, lg, fault_distance(c, A["k"]), round(time.time() - t, 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--jobs", nargs="+", required=True, help="comp:noise")
    a = ap.parse_args()
    A.update(d=a.d, rounds=a.rounds, k=a.k)
    jobs = [(j.split(":")[0], j.split(":")[1], lg) for j in a.jobs for lg in "XY"]
    with Pool(os.cpu_count()) as pool:
        for row in pool.imap(run, jobs):
            print(f"d={a.d} r={a.rounds} k={a.k}", *row, flush=True)
