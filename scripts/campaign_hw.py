"""
campaign_hw.py
----------------------------------------------------------------------------
The comparison with Hetenyi and Wootton (arXiv:2306.17786, Phys. Rev. A 109, 032433, Table I). Their table gives
the intercepts of a plane fitted to the threshold surface of a spin-qubit model, that is, the threshold with
only one kind of error switched on. We measure the same corners in our own circuits with the one-parameter
models of run_decoder_threshold_sweep.py: hw_g (gates only), hw_g1q (gates with noisy single-qubit basis
changes), hw_t (idling only, eta_T = 20) and hw_rr (readout only). Codes: XYZ^2 (both-bases schedule, Logical X
and Logical Y memories), the rotated CSS and XZZX surface codes and the honeycomb Floquet code (X and Z bases).

    python scripts/campaign_hw.py pilot --out data/campaign/hw/pilot_tasks.jsonl
    python scripts/campaign.py run data/campaign/hw/pilot_tasks.jsonl data/campaign/hw/points_pilot.csv
    python scripts/campaign_hw.py pilot-fit "data/campaign/hw/points_pilot*.csv" --out data/campaign/hw/pilot_centers.csv
    python scripts/campaign_hw.py pilot --refine data/campaign/hw/pilot_centers.csv \
        --out data/campaign/hw/pilot_tasks2.jsonl      # finer grids where the first scan was coarse
    python scripts/campaign_hw.py tasks --pilot data/campaign/hw/pilot_centers.csv --out data/campaign/hw/tasks_hw.jsonl

Pilot: MWPM, r = d, d = 5, 7 and 9 (4 and 8 for the honeycomb), 10 values of p per series, 300 logical errors or
2 x 10^4 shots per point. The gate models start from the expected thresholds (GATE_START); the idling and
readout models scan 0.5% to 30% and are then refined around the crossing. `check` writes the tasks that
reproduce the earlier spot check (CSS surface code under SD6 without idle errors, crossing of d = 7 and 9 near
0.91(5)%).

Tasks: XYZ^2 through campaign.make_tasks (labels "hw_g@both-X", "hw_g@both-Y", ...; decoders mwpm, corr_gauge
and bm), the other codes through campaign_competitors.make_tasks (decoders mwpm, corr and bm), r = d, every
distance of the campaign, 14 values of p over a factor 4 centred on the pilot crossing (of the largest pair of
distances) times the gain of each decoder over MWPM.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "paper" / "analysis"))
import campaign as C  # noqa: E402
import campaign_competitors as CC  # noqa: E402
from run_decoder_threshold_sweep import HW_NOISES  # noqa: E402

CODES = ("xyz2", "css", "xzzx", "honeycomb")
BASES = {"xyz2": ("X", "Y"), "css": ("X", "Z"), "xzzx": ("X", "Z"), "honeycomb": ("X", "Z")}
PILOT_D = {"honeycomb": [4, 8]}
PILOT_D_DEFAULT = [5, 7, 9]
XYZ2_DECODERS = ["mwpm", "corr_gauge", "bm"]
COMP_DECODERS = ("mwpm", "corr", "bm")
COMP_CODES = ("css", "xzzx", "honeycomb")
# expected thresholds of the gate models (from the spot checks), the centres of the first pilot grids
GATE_START = {"hw_g": {"css": 1.0e-2, "xzzx": 1.0e-2, "xyz2": 0.75e-2, "honeycomb": 0.3e-2},
              "hw_g1q": {"css": 0.4e-2, "xzzx": 0.4e-2, "xyz2": 0.3e-2, "honeycomb": 0.12e-2}}
SCAN = (5e-3, 0.30)                     # first scan of the idling and readout models
PILOT_TARGET, PILOT_SHOTS, PILOT_BUDGET = 300, 20_000, 3600.0


def series(code: str, basis: str, noise: str, dec: str = "mwpm") -> tuple:
    """(noise label, decoder name) of a series in the task and point files."""
    if code == "xyz2":
        return f"{noise}@both-{basis}", dec
    return noise, f"{code}-{basis}:{dec}"


def parse(noise: str, decoder: str) -> tuple:
    """(code, basis, model, decoder) of a series."""
    if ":" in decoder:
        code, basis, dec = C.competitor(decoder)
        return code, basis, noise, dec
    base, _, logical = C.split_noise(noise)
    return "xyz2", logical, base, decoder


def _task(noise: str, decoder: str, d: int, p: float, target: int = PILOT_TARGET, max_shots: int = PILOT_SHOTS,
          budget: float = PILOT_BUDGET, tag: str = "pilot") -> dict:
    return dict(id=f"{noise}|{decoder}|rd|d{d}|p{p:.4g}|{tag}", noise=noise, decoder=decoder, rounds="d", d=d,
                value=p, target=target, max_shots=max_shots, budget=budget,
                mem=C.memory_for(noise, C.base_decoder(decoder), d, "d"))


def pilot_tasks(noises, codes, refine: dict | None = None) -> list:
    """MWPM pilot tasks. With `refine` (next_values of the earlier scans), only the series it lists, at its
    values."""
    out = []
    for noise in noises:
        for code in codes:
            for basis in BASES[code]:
                if refine is not None:
                    values = refine.get((code, basis, noise))
                    if not values:
                        continue
                elif noise in GATE_START:
                    c = GATE_START[noise][code]
                    values = [float(f"{c * C.STEP ** k:.4g}") for k in C.KRANGE["narrow"]]
                else:
                    values = [float(f"{x:.4g}") for x in np.geomspace(*SCAN, 10)]
                label, dec = series(code, basis, noise)
                for d in PILOT_D.get(code, PILOT_D_DEFAULT):
                    for p in values:
                        out.append(_task(label, dec, d, p, tag="pilot"))
    return out


def next_values(path: str) -> dict:
    """{(code, basis, model): values of p} for a further scan: 10 values over a factor 2.6 around a crossing
    found only between coarse grid values, or 10 values over a factor 5 beyond the range of a series without a
    crossing. Series with a fitted crossing need nothing more."""
    rows = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["decoder"] != "mwpm":
                continue
            key = (r["code"], r["basis"], r["noise"])
            d2 = int(r["d_pair"].split("/")[1])
            if d2 >= rows.get(key, ({}, 0))[1]:
                rows[key] = (r, d2)
    out = {}
    for key, (r, _) in rows.items():
        lo, hi = float(r["lo"]), float(r["hi"])
        if r["how"] == "grid":
            c = float(r["crossing_p"])
            out[key] = [float(f"{c * C.STEP ** k:.4g}") for k in C.KRANGE["narrow"]]
        elif not r["crossing_p"] and "above" in r["note"]:
            out[key] = [float(f"{x:.4g}") for x in np.geomspace(hi, min(5 * hi, 0.44), 10)]
        elif not r["crossing_p"] and "below" in r["note"]:
            out[key] = [float(f"{x:.4g}") for x in np.geomspace(lo / 5, lo, 10)]
    return out


def check_tasks() -> list:
    """The earlier spot check through the campaign worker: CSS surface code, SD6 without idle errors, MWPM,
    r = d, d = 5, 7 and 9 around 0.91%."""
    out = []
    for basis in ("X", "Z"):
        for d in (5, 7, 9):
            for p in np.linspace(0.70e-2, 1.15e-2, 10):
                out.append(_task("sd6_noidle", f"css-{basis}:mwpm", d, float(f"{p:.4g}"), target=3000,
                                 max_shots=2_000_000, budget=3600.0, tag="check"))
    return out


# --------------------------------------------------------------------------- pilot fit
def load_points(patterns) -> "pd.DataFrame":
    """Counts of every point (summed over tasks and partial rows) from raw campaign files."""
    import pandas as pd

    files = sorted({f for pat in patterns for f in glob.glob(pat)})
    rows = C.read_raw(files)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for k in ("value", "seconds"):
        df[k] = df[k].astype(float)
    for k in ("d", "rounds", "shots", "errors"):
        df[k] = df[k].astype(int)
    df["value"] = df.value.round(10)
    df = df.groupby(C.KEYS, as_index=False).agg(shots=("shots", "sum"), errors=("errors", "sum"),
                                                seconds=("seconds", "sum"))
    df = df[df.shots > 0]
    df["p_L"] = df.errors / df.shots
    return df


SATURATED = 0.4     # points with p_L at or above this carry no information on a crossing (p_L -> 1/2)


def crossings_of(g) -> list:
    """Crossings of consecutive distances: the fitted crossing of campaign_report (lines of ln p_L in a +-15%
    window, bootstrap error), or, where the grid is too coarse for that window, the log-linear interpolation
    between the two grid values where the curves cross (no error). Saturated points are left out."""
    import campaign_report as R

    g = g[g.p_L < SATURATED]
    fitted = {(c["d1"], c["d2"]): c for c in R.pair_crossings(g)}
    out = []
    ds = sorted(int(x) for x in g.d.unique())
    for d1, d2 in zip(ds, ds[1:]):
        if (d1, d2) in fitted:
            c = fitted[(d1, d2)]
            out.append(dict(d1=d1, d2=d2, p=c["p"], err=c["err"], how="fit"))
            continue
        x = R.first_guess(g, d1, d2)
        if np.isfinite(x):
            out.append(dict(d1=d1, d2=d2, p=x, err=np.nan, how="grid"))
    return out


def pilot_fit(df) -> list:
    """Rows of the pilot CSV (the format campaign_competitors.pilot_centers reads, plus err and how)."""
    rows = []
    for (noise, decoder), g in df.groupby(["noise", "decoder"]):
        code, basis, model, dec = parse(noise, decoder)
        lo, hi = float(g.value.min()), float(g.value.max())
        cr = crossings_of(g)
        base = dict(code=code, basis=basis, noise=model, rounds="d", decoder=dec, lo=lo, hi=hi)
        if not cr:
            # the side of the range: where both distances have errors at the lowest p, is the larger worse?
            ds = sorted(int(x) for x in g.d.unique())
            both = g[g.errors > 0].pivot_table(index="value", columns="d", values="p_L").dropna()
            worse = (both[ds[-1]] > both[ds[0]]) if ds[0] in both and ds[-1] in both else None
            side = "?" if worse is None or worse.empty else ("below" if bool(worse.iloc[0]) else "above")
            rows.append(dict(base, d_pair=f"{ds[0]}/{ds[-1]}", crossing_p="", err="", how="",
                             note=f"no crossing in [{lo:.4g}, {hi:.4g}]: threshold {side} the range"))
        for c in cr:
            rows.append(dict(base, d_pair=f"{c['d1']}/{c['d2']}", crossing_p=f"{c['p']:.5g}",
                             err="" if not np.isfinite(c["err"]) else f"{c['err']:.3g}", how=c["how"], note=""))
    return rows


PILOT_COLS = ["code", "basis", "noise", "rounds", "decoder", "d_pair", "crossing_p", "err", "lo", "hi", "how", "note"]


def read_pilot(path: str) -> dict:
    """{(code, basis, model): (crossing, lo, hi)} from the pilot CSV: the crossing of the largest pair of
    distances (7/9, or 4/8 for the honeycomb); a series without one gets (None, lo, hi)."""
    out, pair = {}, {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["decoder"] != "mwpm" or r["rounds"] != "d":
                continue
            key = (r["code"], r["basis"], r["noise"])
            lo, hi = float(r["lo"]), float(r["hi"])
            if not r["crossing_p"]:
                out.setdefault(key, (None, lo, hi))
                continue
            d2 = int(r["d_pair"].split("/")[1])
            if d2 >= pair.get(key, 0):
                out[key], pair[key] = (float(r["crossing_p"]), lo, hi), d2
    return out


# --------------------------------------------------------------------------- campaign tasks
def make_tasks(pilot: dict, noises=HW_NOISES, scale: float = 1.0, rep_hours: float = 2.0, skip=()) -> list:
    """Every task of the comparison. `skip` lists (code, model) pairs left out."""
    tasks = []
    for noise in noises:
        for basis in BASES["xyz2"]:
            x = pilot.get(("xyz2", basis, noise), (None,))[0]
            if x is None:                       # no crossing in this basis: centre on the other basis
                x = pilot.get(("xyz2", "Y" if basis == "X" else "X", noise), (None,))[0]
            if x is None:
                raise SystemExit(f"no pilot crossing for XYZ^2 under {noise}")
            label = f"{noise}@both-{basis}"
            cen = {(label, dec, "d"): x * (C._G_DEFAULT[dec] if dec != "mwpm" else 1.0) for dec in XYZ2_DECODERS}
            tasks += C.make_tasks([label], XYZ2_DECODERS, ["d"], C.DISTANCES, scale=scale, rep_hours=rep_hours,
                                  centers=cen, wide=True)
    comp = {}
    for (code, basis, model), v in pilot.items():
        if code != "xyz2":
            comp[(code, basis, model, "d")] = v
    for noise in noises:
        codes = [c for c in COMP_CODES if (c, noise) not in set(skip)]
        for code in codes:
            if all(comp.get((code, b, noise, "d"), (None,))[0] is None for b in BASES[code]):
                raise SystemExit(f"no pilot crossing for {code} under {noise}")
        tasks += CC.make_tasks([noise], ["d"], comp, scale=scale, rep_hours=rep_hours, codes=tuple(codes),
                               decoders=COMP_DECODERS)
    tasks.sort(key=lambda t: (-t["budget"], t["id"]))
    return tasks


def _write(tasks: list, out: str) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for t in tasks:
            fh.write(json.dumps(t) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pilot", help="MWPM pilot tasks")
    a.add_argument("--out", required=True)
    a.add_argument("--noises", nargs="+", default=HW_NOISES)
    a.add_argument("--codes", nargs="+", default=list(CODES))
    a.add_argument("--skip", nargs="*", default=[], metavar="CODE:MODEL", help="series left out, e.g. honeycomb:hw_rr")
    a.add_argument("--refine", default=None, help="pilot CSV of the earlier scans: refine coarse crossings and "
                                                  "scan beyond the range of series without one")
    b = sub.add_parser("check", help="tasks of the spot check (CSS, SD6 without idling)")
    b.add_argument("--out", required=True)
    f = sub.add_parser("pilot-fit", help="crossings of the pilot")
    f.add_argument("points", nargs="+")
    f.add_argument("--out", required=True)
    t = sub.add_parser("tasks", help="task list of the comparison and its cost")
    t.add_argument("--pilot", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--noises", nargs="+", default=HW_NOISES)
    t.add_argument("--scale", type=float, default=1.0, help="multiply every time budget")
    t.add_argument("--rep-hours", type=float, default=2.0)
    t.add_argument("--skip", nargs="*", default=[], metavar="CODE:MODEL")
    args = ap.parse_args()

    if args.cmd == "pilot":
        skip = {tuple(x.split(":")) for x in args.skip}
        ref = next_values(args.refine) if args.refine else None
        tasks = [t for t in pilot_tasks(args.noises, args.codes, ref)
                 if (parse(t["noise"], t["decoder"])[0], parse(t["noise"], t["decoder"])[2]) not in skip]
        _write(tasks, args.out)
        print(f"{len(tasks)} pilot tasks -> {args.out}")
    elif args.cmd == "check":
        tasks = check_tasks()
        _write(tasks, args.out)
        print(f"{len(tasks)} check tasks -> {args.out}")
    elif args.cmd == "pilot-fit":
        rows = pilot_fit(load_points(args.points))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=PILOT_COLS)
            w.writeheader()
            w.writerows(rows)
        for r in rows:
            print(f"{r['noise']:7s} {r['code']:9s} {r['basis']} {r['d_pair']:5s} {r['crossing_p'] or '--':>9s} "
                  f"{r['err'] or '':>9s} {r['how']:4s} {r['note']}")
        print(f"{len(rows)} rows -> {args.out}")
    else:
        tasks = make_tasks(read_pilot(args.pilot), args.noises, args.scale, args.rep_hours,
                           skip={tuple(x.split(":")) for x in args.skip})
        _write(tasks, args.out)
        est = C.estimate(tasks)
        by = {}
        for k, v in est.items():
            code = "xyz2" if ":" not in k else k.split("-")[0]
            name = k.split(":")[-1]
            by[(code, name)] = by.get((code, name), 0.0) + v
        tot = sum(est.values())
        npts = len({t["id"].rsplit("|", 1)[0] for t in tasks})
        print(f"{len(tasks)} tasks ({npts} points) -> {args.out}")
        print("upper bound of the cost (core-hours): "
              + ", ".join(f"{c} {n} {v:,.0f}" for (c, n), v in sorted(by.items()))
              + f"; total {tot:,.0f} = {tot / 128:,.1f} node-hours at 128 cores")
        print(f"longest task {max(t['budget'] for t in tasks) / 3600:.2f} h, "
              f"largest memory estimate {max(t['mem'] for t in tasks):.2f} GB")


if __name__ == "__main__":
    main()
