"""
campaign_fair.py
----------------------------------------------------------------------------
The fair XYZ^2 rerun of docs/fair_comparison.md: every noise model and extraction variant, with the
both-bases schedule, in both logical bases (Logical X and Logical Y memories; the threshold that counts
is the weaker basis), for every decoder, with both extraction schedules (each code uses its best compilation).

    python scripts/campaign_fair.py --out data/campaign/tasks10.jsonl \
        --centers docs/data/campaign/thresholds.csv docs/data/campaign/points.csv \
        --pilot /pscratch/sd/s/sohan100/ftmdr_tools/competitors/pilot_centers_xyz2_variants.csv

Noise labels are "model@both-X" and "model@both-Y" (campaign.split_noise). Grids have 14 points over a factor
4 for every decoder.
- A noise model of the campaign is centred on its measured thresholds.
- An extraction variant is centred on its base model's thresholds, scaled by the ratio of their MWPM pilot
  crossings.
- The phenomenological models are centred on their own pilots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign as C  # noqa: E402

VARIANT_BASE = {"hyb": "sd6", "hyb_lr2": "sd6", "sd6_lr2": "sd6", "sd6_lr3": "sd6", "sd6_il": "sd6",
                "si1000_lr2": "si1000", "phen": None, "phen_b10": None}
VARIANTS = list(VARIANT_BASE)
# beam search and OSD grow fast with the circuit (the 4x campaign spent half its time on Tesseract at d > 11)
DMAX = {"tesseract": 11}
PILOT_BASE = {"sd6": 0.418e-2, "si1000": 0.361e-2}    # MWPM 7/9, r = d, same pilot method (agent report)


def read_pilot(path: str) -> dict:
    """{(noise, rounds, decoder): crossing} from the variant pilot, the largest pair of each series."""
    out, pair = {}, {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                x = float(r["crossing_p"])
            except (TypeError, ValueError):
                continue
            if math.isnan(x):
                continue
            key = (r["noise"], r["rounds"], r["decoder"])
            dp = int(r["d_pair"].split("/")[1])
            if dp >= pair.get(key, 0):
                out[key], pair[key] = x, dp
    return out


def centers(meas: dict, pilot: dict, noises, decoders, rounds) -> dict:
    """{(label noise, decoder, rounds): centre} for every series of the rerun."""
    out = {}
    for noise in noises:
        base = VARIANT_BASE.get(noise, noise)
        for dec in decoders:
            for r in rounds:
                if base is not None:                                  # measured base, scaled for a variant
                    c = C.center_from(meas, base, dec, r)
                    if noise in VARIANT_BASE:                         # same ratio for every r and decoder
                        c *= pilot.get((noise, "d", "mwpm"), PILOT_BASE[base]) / PILOT_BASE[base]
                else:                                                 # phenomenological: own pilot
                    p1, pd = pilot.get((noise, "1", "mwpm")), pilot.get((noise, "d", "mwpm"))
                    if r == "d" or p1 is None:
                        c = pd
                    elif r == "1":
                        c = p1
                    else:
                        t = min(1.0, math.log(int(r)) / math.log(C.R_EFF_D))
                        c = math.exp((1 - t) * math.log(p1) + t * math.log(pd))
                    c *= C._G_DEFAULT.get(dec, 1.3) if dec != "mwpm" else 1.0
                out[(noise, dec, r)] = c
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--centers", nargs=2, required=True, metavar=("THRESHOLDS_CSV", "POINTS_CSV"))
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--noises", nargs="+", default=C.NOISES + VARIANTS)
    ap.add_argument("--decoders", nargs="+", default=C.DECODERS)
    ap.add_argument("--rounds", nargs="+", default=["d", "1", "2", "3", "5", "10"])
    ap.add_argument("--bases", nargs="+", default=["X", "Y"])
    ap.add_argument("--schedules", nargs="+", default=["both", "depth6"],
                    help="each code uses its best compilation: both schedules in both bases (the campaign's depth6 "
                         "Logical-X memory of the campaign's noise models is already measured, as label 'noise')")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--rep-hours", type=float, default=2.0)
    args = ap.parse_args()
    meas = C.measured_centers(*args.centers)
    cen = centers(meas, read_pilot(args.pilot), args.noises, args.decoders, args.rounds)
    tasks = []
    for noise in args.noises:
        for sched, basis in [(s_, b) for s_ in args.schedules for b in args.bases]:
            if (sched, basis) == ("depth6", "X") and noise in C.NOISES:
                continue
            label = f"{noise}@{sched}-{basis}"
            lab_c = {(label, dec, r): c for (n, dec, r), c in cen.items() if n == noise}
            tasks += C.make_tasks([label], args.decoders, args.rounds, C.DISTANCES, scale=args.scale,
                                  rep_hours=args.rep_hours, centers=lab_c, wide=True,
                                  dmax={**C.DMAX, **DMAX})
    tasks.sort(key=lambda t: (-t["budget"], t["id"]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for t in tasks:
            fh.write(json.dumps(t) + "\n")
    est = C.estimate(tasks)
    print(f"{len(tasks)} tasks -> {args.out}")
    print("upper bound (core-hours): " + ", ".join(f"{k} {v:,.0f}" for k, v in sorted(est.items(), key=lambda kv: -kv[1]))
          + f"; total {sum(est.values()):,.0f} = {sum(est.values()) / 128:,.0f} node-hours")


if __name__ == "__main__":
    main()
