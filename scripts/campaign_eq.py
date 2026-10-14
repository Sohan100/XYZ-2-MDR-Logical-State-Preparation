"""
campaign_eq.py
----------------------------------------------------------------------------
Equal-qubit comparison below threshold: the logical error per round of XYZ^2 and the rotated surface codes at
p = 1e-3, 2e-3 and 3e-3, r = 3d rounds, compared at equal numbers of qubits.

Series (noise model, code, basis, decoders):
- XYZ^2 both-bases memory, Logical X and Y ("sd6@both-X", ...): sd6, biased100, purez; mwpm, corr_gauge, bm;
- CSS surface code, bases X and Z: sd6, biased100, purez; mwpm, corr (under SD6 the CSS, XZZX and XY circuits
  have the same detector error model, so CSS stands for all three);
- XZZX surface code, bases X and Z: biased100, purez; mwpm, corr, bm.
Distances: XYZ^2 d = 3 to 11, the surface codes 3 to 15. The equal-qubit pairs (qubits from the Stim circuits)
are XYZ^2 d = 5 / surface 7 (99 / 97), 9 / 13 (323 / 337) and 11 / 15 (483 / 449). At the pair points the
memory also runs with r = d, so that the boundary term drops out of the error per round.

Stopping (campaign.py run / pool-run): 100 logical errors (300 at the pair points) or 1e9 shots, and a time
budget of at most 4 core-hours per point for mwpm and corr and 8 for bm (twice that at the pair points), split
into replicas of at most --rep-hours as campaign.make_tasks does. bm runs where 30 logical errors fit in its
budget. A series stops at the first distance whose expected count at the cap is below 10 logical errors.

    python scripts/campaign_eq.py calib --out data/campaign/eq/calib_tasks.jsonl
    python scripts/campaign.py run data/campaign/eq/calib_tasks.jsonl data/campaign/eq/points_calib.csv
    python scripts/campaign_eq.py tasks --calib "data/campaign/eq/points_calib*.csv" --out data/campaign/eq/tasks_eq.jsonl

`tasks` predicts each point from the calibration (ln epsilon linear in d per series, p and decoder; seconds per
shot as a power of the circuit size) and sets the budget of each point to its cap or to SAFETY times the
predicted time to its target, whichever is smaller. A point that ends on that budget short of its target is
topped up to its cap by `topup`.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import campaign as C  # noqa: E402
import campaign_hw as H  # noqa: E402

NOISES = ["sd6", "biased100", "purez"]
PS = [1e-3, 2e-3, 3e-3]
XYZ2_D = [3, 5, 7, 9, 11]
SURFACE_D = [3, 5, 7, 9, 11, 13, 15]
PAIRS = [(5, 7), (9, 13), (11, 15)]           # (XYZ^2 d, surface-code d) with about equal qubit counts
PAIR_D = {"xyz2": {a for a, _ in PAIRS}, "css": {b for _, b in PAIRS}, "xzzx": {b for _, b in PAIRS}}
DECODERS = {"xyz2": ["mwpm", "corr_gauge", "bm"], "css": ["mwpm", "corr"], "xzzx": ["mwpm", "corr", "bm"]}
CODE_NOISES = {"xyz2": NOISES, "css": NOISES, "xzzx": ["biased100", "purez"]}
BASES = {"xyz2": ("X", "Y"), "css": ("X", "Z"), "xzzx": ("X", "Z")}
CAP_H = {"mwpm": 4.0, "corr": 4.0, "corr_gauge": 4.0, "bm": 8.0}
TARGET, PAIR_TARGET, MAX_SHOTS = 100, 300, 1_000_000_000
BM_MIN_ERRORS = 30              # bm only where this many logical errors fit in its budget
SKIP_BELOW = 10                 # a series stops after the first distance with fewer errors than this at the cap
SAFETY = 4.0                    # budget = min(cap, SAFETY x the predicted time to the target)
MIN_BUDGET = 120.0
CALIB_BUDGET = 120.0
CALIB_D = {"xyz2": [3, 5, 7], "css": [3, 5, 7, 9], "xzzx": [3, 5, 7, 9]}


def distances(code: str) -> list:
    return XYZ2_D if code == "xyz2" else SURFACE_D


def series_names(code: str, basis: str, noise: str, dec: str) -> tuple:
    """(noise label, decoder name) of a series in the task and point files."""
    if code == "xyz2":
        return f"{noise}@both-{basis}", dec
    return noise, f"{code}-{basis}:{dec}"


def every_series():
    for code in ("xyz2", "css", "xzzx"):
        for noise in CODE_NOISES[code]:
            for basis in BASES[code]:
                for dec in DECODERS[code]:
                    yield code, basis, noise, dec


def task(code, basis, noise, dec, d, p, r, target, max_shots, budget, rep=0, tag=None) -> dict:
    label, name = series_names(code, basis, noise, dec)
    return dict(id=f"{label}|{name}|r{r}|d{d}|p{p:.4g}|{tag if tag is not None else rep}", noise=label,
                decoder=name, rounds=r, d=d, value=p, target=target, max_shots=max_shots, budget=round(budget, 1),
                mem=C.memory_for(label, C.base_decoder(name), d, r))


def calib_tasks() -> list:
    out = []
    for code, basis, noise, dec in every_series():
        for d in CALIB_D[code]:
            for p in PS:
                out.append(task(code, basis, noise, dec, d, p, "3d", TARGET, MAX_SHOTS, CALIB_BUDGET, tag="calib"))
    return out


# --------------------------------------------------------------------------- prediction
def circuit_size(d: int, r: str) -> float:
    return (C.n_rounds(r, d) + 1) * d * d


def calib_table(patterns) -> dict:
    """{(code, basis, noise, dec, p): {d: (shots, errors, seconds)}} from the calibration points."""
    df = H.load_points(patterns)
    out = {}
    for row in df.itertuples():
        if row.rounds != 3 * row.d:
            continue
        if ":" in row.decoder:
            code, basis, dec = C.competitor(row.decoder)
            noise = row.noise
        else:
            noise, _, basis = C.split_noise(row.noise)
            code, dec = "xyz2", row.decoder
        out.setdefault((code, basis, noise, dec, round(row.value, 10)), {})[int(row.d)] = (
            int(row.shots), int(row.errors), float(row.seconds))
    return out


def per_shot_rate(shots: int, errors: int, r: int) -> float:
    """Logical error per shot (zero errors: 1/shots, a deliberately pessimistic stand-in for planning)."""
    return max(errors, 1) / shots


# wall seconds per CPU second of a worker (sampling, batching and flushing around the timed decoding)
TIMING_OVERHEAD = 1.5


def timing_table(path: str) -> dict:
    """{(code, dec, d, p): CPU seconds per shot of sampling and decoding, r = 3d} (scripts/eq_timing.py)."""
    import csv as _csv

    out = {}
    with open(path, newline="") as fh:
        for r in _csv.DictReader(fh):
            out[(r["code"], r["dec"], int(r["d"]), round(float(r["p"]), 10))] = \
                float(r["decode_s"]) + float(r["sample_s"])
    return out


def predict(cal: dict, code: str, basis: str, noise: str, dec: str, p: float, timing: dict | None = None):
    """Functions d -> predicted p_L per shot of the r = 3d memory and d -> seconds per shot."""
    pts = cal.get((code, basis, noise, dec, round(p, 10)), {})
    good = [(d, s, e, t) for d, (s, e, t) in sorted(pts.items()) if s > 0]
    if not good:
        raise SystemExit(f"no calibration for {code} {basis} {noise} {dec} {p}")
    if timing is not None:
        # measured CPU time per shot at r = 3d; r = d scales with the number of rounds + 1
        tab = {d: v for (c, dd, d, pp), v in timing.items() if c == code and dd == dec and pp == round(p, 10)}

        def t_shot(d, r="3d"):
            if d in tab:
                base = tab[d]
            else:                       # nearest measured distance, scaled with the circuit size
                d0 = min(tab, key=lambda x: abs(x - d))
                base = tab[d0] * circuit_size(d, "3d") / circuit_size(d0, "3d")
            return float(TIMING_OVERHEAD * base * circuit_size(d, r) / circuit_size(d, "3d"))
    else:
        # seconds per shot against the circuit size, from the calibration's wall times
        sizes = [circuit_size(d, "3d") for d, *_ in good]
        secs = [max(t, 1e-3) / s for _, s, _, t in good]
        if len(good) >= 2:
            a, b = np.polyfit(np.log(sizes), np.log(secs), 1)
            a = max(a, 1.0)             # at least linear in the circuit size
        else:
            a, b = 1.0, math.log(secs[0]) - math.log(sizes[0])

        def t_shot(d, r="3d"):
            return float(math.exp(b) * circuit_size(d, r) ** a)

    # error per round, fitted where there are errors; the slope is kept no steeper than the measured one
    eps = [(d, (1 - (1 - 2 * min(e, 0.49 * s) / s) ** (1 / (3 * d))) / 2) for d, s, e, _ in good if e >= 3]
    if len(eps) >= 2:
        k, c = np.polyfit([d for d, _ in eps], np.log([x for _, x in eps]), 1)
        k = min(k, -0.05)
    elif len(eps) == 1:
        k, c = -0.5, math.log(eps[0][1]) + 0.5 * eps[0][0]
    else:                               # no errors anywhere: the per-shot bound of the largest point
        d0, s0, _, _ = good[-1]
        k, c = -0.5, math.log(1.0 / (s0 * 3 * d0)) + 0.5 * d0

    def p_shot(d, r="3d"):
        e = min(math.exp(c + k * d), 0.49)
        n = C.n_rounds(r, d)
        return max(float(-0.5 * math.expm1(n * math.log1p(-2 * e))), 1e-300)

    return p_shot, t_shot


def make_tasks(cal: dict, rep_hours: float = 2.0, safety: float = SAFETY, timing: dict | None = None) -> tuple:
    """(tasks, report rows). Every series runs its distances in order and stops after the first one whose
    expected count at the cap is below SKIP_BELOW. bm runs only where BM_MIN_ERRORS fit in its cap."""
    tasks, rows = [], []
    for code, basis, noise, dec in every_series():
        for p in PS:
            p_shot, t_shot = predict(cal, code, basis, noise, dec, p, timing)
            for r in ("3d", "d"):
                for d in distances(code):
                    pair = d in PAIR_D[code]
                    if r == "d" and not pair:
                        continue
                    cap = 3600.0 * CAP_H[dec] * (2 if pair else 1)
                    target = PAIR_TARGET if pair else TARGET
                    pl, ts = p_shot(d, r), t_shot(d, r)
                    shots_cap = min(MAX_SHOTS, cap / ts)
                    expect = pl * shots_cap
                    if dec == "bm" and expect < BM_MIN_ERRORS:
                        rows.append(dict(code=code, basis=basis, noise=noise, dec=dec, p=p, r=r, d=d, pair=pair,
                                         p_shot=pl, t_shot=ts, expect=expect, budget=0.0, need=0.0,
                                         note="bm: < 30 at cap"))
                        continue
                    need = target / pl * ts
                    budget = max(MIN_BUDGET, min(cap, safety * need))
                    nrep = max(1, math.ceil(budget / (3600.0 * rep_hours)))
                    for i in range(nrep):
                        tasks.append(task(code, basis, noise, dec, d, p, r, math.ceil(target / nrep),
                                          math.ceil(MAX_SHOTS / nrep), budget / nrep, rep=i))
                    rows.append(dict(code=code, basis=basis, noise=noise, dec=dec, p=p, r=r, d=d, pair=pair,
                                     p_shot=pl, t_shot=ts, expect=expect, budget=budget, need=need,
                                     note="last: < 10 at cap" if expect < SKIP_BELOW else ""))
                    if r == "3d" and expect < SKIP_BELOW:
                        break               # larger distances of this series are skipped (upper bound)
    tasks.sort(key=lambda t: (-t["budget"], t["id"]))
    return tasks, rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("calib", help="short calibration runs at small d")
    a.add_argument("--out", required=True)
    t = sub.add_parser("tasks", help="task list with predicted budgets, and its cost")
    t.add_argument("--calib", nargs="+", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--rep-hours", type=float, default=2.0)
    t.add_argument("--safety", type=float, default=SAFETY)
    t.add_argument("--timing", default=None, help="CPU seconds per shot (scripts/eq_timing.py)")
    args = ap.parse_args()
    if args.cmd == "calib":
        tasks = calib_tasks()
        H._write(tasks, args.out)
        print(f"{len(tasks)} calibration tasks -> {args.out}")
        return
    tasks, rows = make_tasks(calib_table(args.calib), args.rep_hours, args.safety,
                             timing_table(args.timing) if args.timing else None)
    H._write(tasks, args.out)
    import pandas as pd

    rep = pd.DataFrame(rows)
    rep.to_csv(Path(args.out).with_suffix(".plan.csv"), index=False)
    est = C.estimate(tasks)
    tot = sum(est.values())
    by = {}
    for k, v in est.items():
        name = k.split(":")[-1]
        code = "xyz2" if ":" not in k else k.split("-")[0]
        by[(code, name)] = by.get((code, name), 0.0) + v
    print(f"{len(tasks)} tasks ({len({t['id'].rsplit('|', 1)[0] for t in tasks})} points) -> {args.out}")
    print("upper bound of the cost (core-hours): " + ", ".join(f"{c} {n} {v:,.0f}" for (c, n), v in sorted(by.items()))
          + f"; total {tot:,.0f} = {tot / 128:,.1f} node-hours at 128 cores")
    exp_h = sum(min(rr["budget"], rr["need"]) for rr in rows) / 3600
    print(f"expected (predicted time to target, capped): about {exp_h:,.0f} core-hours = {exp_h / 128:,.1f} node-hours")
    print(f"skipped bm points: {int((rep.note == 'bm: < 30 at cap').sum())}; series ending early "
          f"(< 10 errors expected at the cap): {int((rep.note == 'last: < 10 at cap').sum())}")
    print(f"longest task {max(t['budget'] for t in tasks) / 3600:.2f} h, largest memory {max(t['mem'] for t in tasks):.2f} GB")


if __name__ == "__main__":
    main()
