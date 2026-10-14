"""
campaign_gap.py
----------------------------------------------------------------------------
Gap diagnostic: how much of the threshold gap between XYZ^2 and the XZZX code under SD6 is left once the
link checks are noiseless and the idle errors are gone. New series, r = d, the campaign's distances and grids:

- XYZ^2 both-bases memory (Logical X and Y) under sd6_noidle and sd6_il_noidle, decoders mwpm, corr_gauge, bm;
- the CSS and XZZX surface codes (bases X and Z) under sd6_noidle, decoders mwpm, corr, bm.

XYZ^2 under sd6 and sd6_il and the surface codes under sd6 come from the fair rerun (tasks10, tasks9).

    python scripts/campaign_gap.py pilot --out data/campaign/gap/pilot_tasks.jsonl
    python scripts/campaign.py run data/campaign/gap/pilot_tasks.jsonl data/campaign/gap/points_pilot.csv
    python scripts/campaign_hw.py pilot-fit "data/campaign/gap/points_pilot*.csv" --out data/campaign/gap/pilot_centers.csv
    python scripts/campaign_gap.py tasks --pilot data/campaign/gap/pilot_centers.csv --out data/campaign/gap/tasks_gap.jsonl

The MWPM grids are centred on the crossings of d = 7 and 9 of the earlier side study under sd6_noidle (XYZ^2
Logical X 0.68%, Logical Y 0.73%, surface codes 0.91%) and on the pilot crossing under sd6_il_noidle (started
near 0.8%), and scaled by the decoder gains of the first run (campaign._G_DEFAULT for XYZ^2,
campaign_competitors.GAIN for the surface codes). 14 values over a factor 4.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import campaign as C  # noqa: E402
import campaign_competitors as CC  # noqa: E402
import campaign_hw as H  # noqa: E402

XYZ2_CENTERS = {("sd6_noidle", "X"): 0.68e-2, ("sd6_noidle", "Y"): 0.73e-2}
SURFACE_CENTER = {"sd6_noidle": 0.91e-2}
PILOT_START = 0.8e-2
XYZ2_DECODERS = ["mwpm", "corr_gauge", "bm"]
COMP_DECODERS = ("mwpm", "corr", "bm")


def pilot_tasks(noise: str = "sd6_il_noidle", start: float = PILOT_START) -> list:
    """MWPM pilot of XYZ^2 (both bases), r = d, d = 5, 7, 9, 10 values over a factor 2.6 around `start`."""
    values = [float(f"{start * C.STEP ** k:.4g}") for k in C.KRANGE["narrow"]]
    return [H._task(f"{noise}@both-{L}", "mwpm", d, p) for L in ("X", "Y") for d in H.PILOT_D_DEFAULT
            for p in values]


def make_tasks(pilot: dict, scale: float = 1.0, rep_hours: float = 2.0) -> list:
    tasks = []
    centers = dict(XYZ2_CENTERS)
    for L in ("X", "Y"):
        x = pilot.get(("xyz2", L, "sd6_il_noidle"), (None,))[0]
        if x is None:
            x = pilot.get(("xyz2", "Y" if L == "X" else "X", "sd6_il_noidle"), (None,))[0]
        if x is None:
            raise SystemExit("no pilot crossing for XYZ^2 under sd6_il_noidle")
        centers[("sd6_il_noidle", L)] = x
    for (noise, L), x in sorted(centers.items()):
        label = f"{noise}@both-{L}"
        cen = {(label, dec, "d"): x * (C._G_DEFAULT[dec] if dec != "mwpm" else 1.0) for dec in XYZ2_DECODERS}
        tasks += C.make_tasks([label], XYZ2_DECODERS, ["d"], C.DISTANCES, scale=scale, rep_hours=rep_hours,
                              centers=cen, wide=True)
    for noise, x in SURFACE_CENTER.items():
        comp = {(code, b, noise, "d"): (x, None, None) for code in ("css", "xzzx") for b in ("X", "Z")}
        tasks += CC.make_tasks([noise], ["d"], comp, scale=scale, rep_hours=rep_hours, codes=("css", "xzzx"),
                               decoders=COMP_DECODERS)
    tasks.sort(key=lambda t: (-t["budget"], t["id"]))
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pilot", help="MWPM pilot of XYZ^2 under sd6_il_noidle")
    a.add_argument("--out", required=True)
    a.add_argument("--start", type=float, default=PILOT_START)
    t = sub.add_parser("tasks", help="task list of the gap diagnostic and its cost")
    t.add_argument("--pilot", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--scale", type=float, default=1.0, help="multiply every time budget")
    t.add_argument("--rep-hours", type=float, default=2.0)
    args = ap.parse_args()
    if args.cmd == "pilot":
        tasks = pilot_tasks(start=args.start)
        H._write(tasks, args.out)
        print(f"{len(tasks)} pilot tasks -> {args.out}")
        return
    tasks = make_tasks(H.read_pilot(args.pilot), args.scale, args.rep_hours)
    H._write(tasks, args.out)
    est = C.estimate(tasks)
    tot = sum(est.values())
    print(f"{len(tasks)} tasks ({len({t['id'].rsplit('|', 1)[0] for t in tasks})} points) -> {args.out}")
    print("upper bound of the cost (core-hours): " + ", ".join(f"{k} {v:,.0f}" for k, v in sorted(est.items()))
          + f"; total {tot:,.0f} = {tot / 128:,.1f} node-hours at 128 cores")
    print(f"longest task {max(t['budget'] for t in tasks) / 3600:.2f} h, "
          f"largest memory estimate {max(t['mem'] for t in tasks):.2f} GB")


if __name__ == "__main__":
    main()
