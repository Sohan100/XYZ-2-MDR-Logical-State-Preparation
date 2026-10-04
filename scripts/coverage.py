"""
coverage.py
----------------------------------------------------------------------------
Coverage of the threshold campaign: does every (noise, decoder, rounds, d) instance have data, and does
every (noise, decoder, rounds) series have a threshold?

    python scripts/coverage.py --tasks "data/campaign/tasks*.jsonl" --points "data/campaign/points_*.csv" \
        --thresholds docs/data/campaign/thresholds.csv --merged docs/data/campaign/points.csv \
        --out docs/data/campaign/coverage

The full matrix is every noise model, decoder, number of rounds (1 to 21 and r = d) and distance (3 to 21)
of scripts/campaign.py. An instance is "planned" if a task file has a task for it; the tensor-network
decoders are planned only where their network can be contracted (see campaign.py), the rest of the matrix
for them is reported as not computable. Writes <out>.csv (one row per instance) and <out>.md (summary).
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import campaign as C


def instances_from_tasks(paths) -> set:
    out = set()
    for f in paths:
        for line in open(f):
            if line.strip():
                t = json.loads(line)
                out.add((t["noise"], t["decoder"], t["rounds"], int(t["d"])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default="data/campaign/tasks*.jsonl")
    ap.add_argument("--points", default="data/campaign/points_*.csv")
    ap.add_argument("--thresholds", default="docs/data/campaign/thresholds.csv")
    ap.add_argument("--merged", default="docs/data/campaign/points.csv")
    ap.add_argument("--out", default="docs/data/campaign/coverage")
    ap.add_argument("--minerr", type=int, default=3, help="errors for a point to count as measured")
    args = ap.parse_args()

    planned = instances_from_tasks(sorted(glob.glob(args.tasks)))
    # counts per point (replicas summed), from the raw rows, whose task ids keep r = d apart from r = 7 at d = 7
    point = defaultdict(lambda: [0, 0])
    for tid, (shots, errors, _, _) in C.progress(sorted(glob.glob(args.points))).items():
        noise, dec, r, d, p, _ = tid.split("|")
        key = (noise, dec, r[1:], int(d[1:]), p[1:])
        point[key][0] += shots
        point[key][1] += errors
    inst = defaultdict(lambda: [0, 0, 0, 0])          # points with shots, points with >= minerr errors, shots, errors
    for (noise, dec, rk, d, _), (sh, er) in point.items():
        a = inst[(noise, dec, rk, d)]
        a[0] += sh > 0
        a[1] += er >= args.minerr
        a[2] += sh
        a[3] += er

    rows = []
    for noise in C.NOISES:
        for dec in C.DECODERS:
            for rk in C.ROUNDS:
                for d in C.DISTANCES:
                    k = (noise, dec, rk, d)
                    a = inst.get(k, [0, 0, 0, 0])
                    rows.append(dict(noise=noise, decoder=dec, rounds=rk, d=d, planned=k in planned,
                                     points=a[0], measured_points=a[1], shots=a[2], errors=a[3]))
    df = pd.DataFrame(rows)
    df["status"] = np.where(df.points > 0, np.where(df.measured_points >= 3, "measured", "sparse"),
                            np.where(df.planned, "missing", "not planned"))
    df.to_csv(f"{args.out}.csv", index=False)

    # thresholds of every series
    th = pd.read_csv(args.thresholds) if Path(args.thresholds).exists() else pd.DataFrame()
    if not th.empty:
        th["rounds"] = th.rounds.astype(str)
        pts = pd.read_csv(args.merged)
        pts["rk"] = np.where(pts.rounds == pts.d, "d", pts.rounds.astype(str))
        rng = pts.groupby(["noise", "decoder", "rk"]).value.agg(["min", "max"]).reset_index().rename(columns={"rk": "rounds"})
        th = th.merge(rng, on=["noise", "decoder", "rounds"], how="left")
        th["edge"] = th.pth.notna() & ((th.pth <= 1.08 * th["min"]) | (th.pth >= 0.92 * th["max"]))
    ser = df.groupby(["noise", "decoder", "rounds"]).agg(planned=("planned", "any"), distances=("measured_points", lambda x: int((x >= 3).sum()))).reset_index()
    if not th.empty:
        ser = ser.merge(th[["noise", "decoder", "rounds", "method", "pth", "flag", "edge"]], on=["noise", "decoder", "rounds"], how="left")
    else:
        ser["method"], ser["pth"], ser["flag"], ser["edge"] = "none", np.nan, "", False
    ser["flag"] = ser.flag.fillna("")
    ser["result"] = np.select(
        [~ser.planned, ser.pth.isna(), ser.flag == "drift", ser.edge.fillna(False).astype(bool), ser.method == "pair"],
        ["not planned", "no threshold", "drift (no threshold)", "at grid edge", "threshold (2 largest d)"],
        "threshold (finite-size fit)")

    lines = ["# Coverage of the threshold campaign", "",
             f"Full matrix: {len(C.NOISES)} noise models x {len(C.DECODERS)} decoders x {len(C.ROUNDS)} numbers of rounds "
             f"(1 to 21 and r = d) x {len(C.DISTANCES)} distances = {len(df)} instances. An instance is measured when at "
             f"least 3 of its points have {args.minerr} or more logical errors; the tensor-network decoders are planned only "
             "where their network can be contracted.", "",
             "## Instances (noise, decoder, rounds, d)", "",
             "| decoder | planned | measured | sparse | missing | not planned |", "|---|---|---|---|---|---|"]
    for dec in C.DECODERS:
        x = df[df.decoder == dec]
        c = x.status.value_counts()
        lines.append(f"| {dec} | {int(x.planned.sum())} | {c.get('measured', 0)} | {c.get('sparse', 0)} | "
                     f"{c.get('missing', 0)} | {c.get('not planned', 0)} |")
    c = df.status.value_counts()
    lines.append(f"| **all** | {int(df.planned.sum())} | {c.get('measured', 0)} | {c.get('sparse', 0)} | "
                 f"{c.get('missing', 0)} | {c.get('not planned', 0)} |")
    lines += ["", "## Series (noise, decoder, rounds): thresholds", "",
              "| decoder | planned | finite-size fit | 2 largest d | at grid edge | drift | no threshold |", "|---|---|---|---|---|---|---|"]
    for dec in C.DECODERS:
        x = ser[ser.decoder == dec]
        c = x.result.value_counts()
        lines.append(f"| {dec} | {int(x.planned.sum())} | {c.get('threshold (finite-size fit)', 0)} | "
                     f"{c.get('threshold (2 largest d)', 0)} | {c.get('at grid edge', 0)} | {c.get('drift (no threshold)', 0)} | "
                     f"{c.get('no threshold', 0)} |")
    miss = df[df.status == "missing"]
    lines += ["", f"## Planned instances without data: {len(miss)}", ""]
    for (noise, dec), x in miss.groupby(["noise", "decoder"]):
        lines.append(f"- {noise} / {dec}: " + ", ".join(f"r={r} d={d}" for r, d in zip(x.rounds, x.d)))
    nt = ser[ser.planned & ser.result.isin(["no threshold", "at grid edge"])]
    lines += ["", f"## Planned series without a bracketed threshold: {len(nt)}", ""]
    for (noise, dec), x in nt.groupby(["noise", "decoder"]):
        lines.append(f"- {noise} / {dec}: " + ", ".join(f"r={r} ({res})" for r, res in zip(x.rounds, x.result)))
    Path(f"{args.out}.md").write_text("\n".join(lines) + "\n")
    ser.to_csv(f"{args.out}_series.csv", index=False)
    print(f"{len(df)} instances: {int(df.planned.sum())} planned, {(df.status == 'measured').sum()} measured, "
          f"{len(miss)} planned without data; {int(ser.planned.sum())} planned series -> {args.out}.md")


if __name__ == "__main__":
    main()
