"""
distance_scan.py
----------------------------------------------------------------------------
Circuit-level fault distance of every valid depth-6 ExtractionSchedule in
both logical bases under sd6 and hyb.

The distance is the size of the smallest undetectable logical error that
Stim's `search_for_undetectable_logical_errors` finds (an upper bound on the
true fault distance; exact for the small detection-set budgets used here in
the cases we cross-checked with CP-SAT). The memory starts and ends in the
frame basis, so there are no time-like logical errors and a few rounds are
enough to expose every hook.

    python cloud/schedule_search/distance_scan.py --d 3 --rounds 3 --k 4 \
        --out cloud/schedule_search/data/dist_d3.csv
    python cloud/schedule_search/distance_scan.py --d 5 --rounds 3 --k 3 \
        --only cloud/schedule_search/data/pass_d3.txt --out .../dist_d5.csv
    python cloud/schedule_search/distance_scan.py --family hex --noises hyb \
        --d 3 --out cloud/schedule_search/data/hex_d3.csv
"""

from __future__ import annotations

import argparse
import csv
from multiprocessing import Pool
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mdr.ft import CircuitNoise, FTMDRCircuit  # noqa: E402
from mdr.ft.extraction_schedule import ExtractionSchedule  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compilations import hex_only_sigma_pairs  # noqa: E402

NOISES = {"sd6": CircuitNoise.uniform(1e-3), "hyb": CircuitNoise.hybrid(1e-3)}
SCHEDULES = ExtractionSchedule.enumerate_depth(6)
# hex-only sigma pairs (hybrid extraction, links are pair measurements)
HEX = hex_only_sigma_pairs()
ARGS = {}


def fault_distance(circuit, k: int) -> int:
    errs = circuit.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=k,
        dont_explore_edges_with_degree_above=9999,
        dont_explore_edges_increasing_symptom_degree=False,
        canonicalize_circuit_errors=False,
    )
    return len(errs)


def score(idx: int):
    d, r, k = ARGS["d"], ARGS["rounds"], ARGS["k"]
    s = (HEX if ARGS["family"] == "hex" else SCHEDULES)[idx]
    out = {"idx": idx}
    for nm in ARGS["noises"]:
        for lg in "XY":
            try:
                c = FTMDRCircuit(d, r, NOISES[nm], final="frame",
                                 detectors="combined", schedule=s,
                                 logical=lg).build()
            except ValueError:
                out[f"{nm}_{lg}"] = -1  # fails patch validation
                continue
            try:
                out[f"{nm}_{lg}"] = fault_distance(c, k)
            except ValueError:
                out[f"{nm}_{lg}"] = 99  # nothing found within the budget
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--noises", nargs="+", default=["sd6", "hyb"])
    ap.add_argument("--family", default="enum", choices=["enum", "hex"],
                    help="enum: enumerate_depth(6); hex: hex_only_sigma_pairs (hyb only)")
    ap.add_argument("--only", type=Path, default=None,
                    help="file with one schedule index per line")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ARGS.update(family=a.family, d=a.d, rounds=a.rounds, k=a.k, noises=a.noises)
    idxs = (range(len(HEX if a.family == "hex" else SCHEDULES)) if a.only is None
            else [int(x) for x in a.only.read_text().split()])
    done = set()
    if a.out.exists():
        with a.out.open() as fh:
            done = {int(row["idx"]) for row in csv.DictReader(fh)}
    todo = [i for i in idxs if i not in done]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["idx"] + [f"{nm}_{lg}" for nm in a.noises for lg in "XY"]
    new = not a.out.exists()
    with a.out.open("a", newline="") as fh, Pool(a.workers) as pool:
        w = csv.DictWriter(fh, fieldnames=cols)
        if new:
            w.writeheader()
        for n, row in enumerate(pool.imap_unordered(score, todo, chunksize=4)):
            w.writerow(row)
            if n % 200 == 0:
                fh.flush()
                print(n, len(todo), row, flush=True)


if __name__ == "__main__":
    main()
