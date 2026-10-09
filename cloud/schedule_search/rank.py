"""
rank.py
----------------------------------------------------------------------------
Fast ranking of compilations by their MWPM logical error rate on the S_0
graph (the decision of TwoLevelDecoder mode "mwpm", built here directly from
the detectors="s0" circuit to skip the lower-level setup).

    python cloud/schedule_search/rank.py --comps-file pass_d5.txt --prefix s \
        --noises sd6 --d 5 --p 5e-3 --shots 5000 --out data/rank_sd6_d5.csv
"""

from __future__ import annotations

import argparse
import csv
from multiprocessing import Pool
import os
from pathlib import Path
import sys

import numpy as np
import pymatching

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from compilations import NOISE, build_ft  # noqa: E402

OPT = {}


def s0_circuit(comp, d, rounds, noise, logical):
    ft = build_ft(comp, d, rounds, noise, logical)
    if hasattr(ft, "s0_copy"):
        return ft.s0_copy().build()
    from mdr.ft import FTMDRCircuit
    return FTMDRCircuit(d, rounds, noise, schedule=ft.schedule, final="frame",
                        detectors="s0", logical=logical).build()


def run(job):
    comp, noise = job
    d, p, shots = OPT["d"], OPT["p"], OPT["shots"]
    row = {"comp": comp, "noise": noise}
    for lg in "XY":
        c = s0_circuit(comp, d, OPT["rounds"] or d, NOISE[noise](p), lg)
        m = pymatching.Matching.from_detector_error_model(
            c.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True))
        dets, obs = c.compile_detector_sampler(seed=OPT["seed"]).sample(
            shots, separate_observables=True)
        row[f"p{lg}"] = float(np.mean(np.any(m.decode_batch(dets) != obs, axis=1)))
    row["worst"] = max(row["pX"], row["pY"])
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comps", nargs="*", default=[])
    ap.add_argument("--comps-file", type=Path)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--noises", nargs="+", required=True)
    ap.add_argument("--d", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=0)
    ap.add_argument("--p", type=float, default=5e-3)
    ap.add_argument("--shots", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    OPT.update(d=a.d, rounds=a.rounds, p=a.p, shots=a.shots, seed=a.seed)
    comps = list(a.comps)
    if a.comps_file:
        comps += [a.prefix + x for x in a.comps_file.read_text().split()]
    jobs = [(c, n) for n in a.noises for c in comps]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as fh, Pool(os.cpu_count()) as pool:
        w = csv.DictWriter(fh, fieldnames=["comp", "noise", "pX", "pY", "worst"])
        w.writeheader()
        for k, row in enumerate(pool.imap_unordered(run, jobs, chunksize=2)):
            w.writerow(row)
            if k % 100 == 0:
                fh.flush()
                print(k, len(jobs), row, flush=True)


if __name__ == "__main__":
    main()
