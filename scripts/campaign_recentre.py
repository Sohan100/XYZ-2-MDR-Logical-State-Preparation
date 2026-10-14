"""
campaign_recentre.py
----------------------------------------------------------------------------
Second pass of a threshold run: every series whose fitted threshold lies within 25% of either end of its grid
(p_th < 1.25 p_min or p_th > p_max / 1.25) gets a new grid of 14 values over a factor 4 centred on the fit,
with the budgets of the first pass (campaign.make_tasks for XYZ^2, campaign_competitors.make_tasks for the
other codes). Values that the first pass already has keep their task ids, so they are not run twice.

    python scripts/campaign_recentre.py --thresholds docs/data/campaign/hw/thresholds.csv \
        --tasks data/campaign/hw/tasks_hw.jsonl --out data/campaign/hw/tasks_hw_recentre.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import campaign as C  # noqa: E402
import campaign_competitors as CC  # noqa: E402

EDGE = 1.25


def grids(task_file: str) -> dict:
    """{(noise, decoder, rounds): (p_min, p_max, scale)} of the series of a task file (scale: budget factor)."""
    out = {}
    for line in open(task_file):
        if not line.strip():
            continue
        t = json.loads(line)
        key = (t["noise"], t["decoder"], str(t["rounds"]))
        lo, hi = out.get(key, (t["value"], t["value"]))
        out[key] = (min(lo, t["value"]), max(hi, t["value"]))
    return out


def near_edge(th: pd.DataFrame, g: dict) -> list:
    rows = []
    for r in th.itertuples():
        key = (r.noise, r.decoder, str(r.rounds))
        if key not in g or not pd.notna(r.pth):
            continue
        lo, hi = g[key]
        if r.pth < EDGE * lo or r.pth > hi / EDGE:
            rows.append(dict(noise=r.noise, decoder=r.decoder, rounds=str(r.rounds), pth=float(r.pth), lo=lo, hi=hi))
    return rows


def new_tasks(rows: list, scale: float, rep_hours: float) -> list:
    tasks = []
    for r in rows:
        comp = C.competitor(r["decoder"])
        if comp is None:
            tasks += C.make_tasks([r["noise"]], [r["decoder"]], [r["rounds"]], C.DISTANCES, scale=scale,
                                  rep_hours=rep_hours, centers={(r["noise"], r["decoder"], r["rounds"]): r["pth"]},
                                  wide=True, dmax={**C.DMAX, "tesseract": 11})
        else:
            code, basis, dec = comp
            pilot = {(code, basis, r["noise"], r["rounds"]): (r["pth"] / CC.GAIN[dec], None, None)}
            tasks += [t for t in CC.make_tasks([r["noise"]], [r["rounds"]], pilot, scale=scale, rep_hours=rep_hours,
                                               codes=(code,), decoders=(dec,))
                      if t["decoder"] == r["decoder"]]
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thresholds", required=True, help="fits of the first pass (noise, decoder, rounds, pth)")
    ap.add_argument("--tasks", nargs="+", required=True, help="task files of the first pass")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--rep-hours", type=float, default=2.0)
    args = ap.parse_args()
    th = pd.read_csv(args.thresholds)
    th["rounds"] = th.rounds.astype(str)
    g = {}
    for f in args.tasks:
        g.update(grids(f))
    rows = near_edge(th, g)
    done = set()
    for f in args.tasks:
        done |= {json.loads(line)["id"] for line in open(f) if line.strip()}
    tasks = [t for t in new_tasks(rows, args.scale, args.rep_hours) if t["id"] not in done]
    tasks.sort(key=lambda t: (-t["budget"], t["id"]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for t in tasks:
            fh.write(json.dumps(t) + "\n")
    for r in rows:
        print(f"{r['noise']:22s} {r['decoder']:20s} r={r['rounds']}: p_th {100 * r['pth']:.3f}% in grid "
              f"[{100 * r['lo']:.3f}, {100 * r['hi']:.3f}]%")
    est = sum(C.estimate(tasks).values())
    print(f"{len(rows)} series near an edge -> {len(tasks)} new tasks, upper bound {est:,.0f} core-hours -> {args.out}")


if __name__ == "__main__":
    main()
