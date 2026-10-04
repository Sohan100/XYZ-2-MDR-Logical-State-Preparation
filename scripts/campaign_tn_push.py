"""
campaign_tn_push.py
----------------------------------------------------------------------------
Tensor-network stage of the campaign: CFE + TN (cfe_tn) and pure tensor-network maximum likelihood (tnml)
at every case where the network converges, with bond dimension up to 256, in reasonable time. Every point
gets the 4x budget of the main matrix or enough time for about TARGET_SHOTS shots, whichever is larger.

    python scripts/campaign_tn_push.py --out data/campaign/tasks7.jsonl \
        --centers docs/data/campaign/thresholds.csv docs/data/campaign/points.csv

Seconds per shot (SD6 at the CFE crossing, bond dimension up to 256; pilots of 2026-10-03/04). Raising the
bond dimension to 1024 gave no shot within two hours at r = 2, d = 11; r = 3, d = 5 and 7; r = 4, d = 3
and 5; r = 5 to 8, d = 3; r = d, d = 5. r = 5 and 6 at d = 3 alone would cost ~65,000 core-hours and give
no threshold (no second distance), so they are left out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign as C  # noqa: E402

CASES = {"1": list(range(3, 22, 2)), "2": [3, 5, 7, 9], "3": [3, 5], "4": [3], "d": [3]}
SHOT_S = {("2", 7): 126.0, ("2", 9): 294.0, ("3", 3): 61.0, ("3", 5): 1264.0, ("4", 3): 150.0, ("d", 3): 33.0}
TARGET_SHOTS = 400
BASE_SCALE = 4.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--centers", nargs=2, required=True, metavar=("THRESHOLDS_CSV", "POINTS_CSV"))
    ap.add_argument("--rep-hours", type=float, default=2.0)
    args = ap.parse_args()
    meas = C.measured_centers(*args.centers)
    tasks = []
    for r, ds in CASES.items():
        for d in ds:
            t_shot = SHOT_S.get((r, d), 0.0)
            scale = max(BASE_SCALE, TARGET_SHOTS * t_shot / C.budget("cfe_tn", d, r))
            tasks += C.make_tasks(C.NOISES, ["cfe_tn", "tnml"], [r], [d], scale=scale, rep_hours=args.rep_hours,
                                  centers=meas, wide=True)
    tasks.sort(key=lambda t: (-t["budget"], t["id"]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for t in tasks:
            fh.write(json.dumps(t) + "\n")
    est = C.estimate(tasks)
    print(f"{len(tasks)} tasks ({len({t['id'].rsplit('|', 1)[0] for t in tasks})} points) -> {args.out}")
    print("upper bound of the cost (core-hours): " + ", ".join(f"{k} {v:,.0f}" for k, v in est.items())
          + f"; total {sum(est.values()):,.0f} = {sum(est.values()) / 128:,.0f} node-hours at 128 cores")
    print(f"longest task {max(t['budget'] for t in tasks) / 3600:.1f} h")


if __name__ == "__main__":
    main()
