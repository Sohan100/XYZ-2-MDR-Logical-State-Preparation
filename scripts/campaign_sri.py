"""
campaign_sri.py
----------------------------------------------------------------------------
Task lists of the study of the sequential decoder of Srivastava et al. (arXiv:2505.03691) at circuit
level (docs/srivastava_exact_decoder.md): the decoders sri_m (their sequential matching) and sri_bm
(their sequential belief-matching), r = d unless stated.

Parts (--part):
- circuit: the 8 circuit-level noise models, each as the preparation (depth-6 schedule, Logical X, the
  bare label) and the both-bases memories in X and Y ("@both-X", "@both-Y"); grids centred on the measured
  thresholds of seq_match (sri_m) and bm (sri_bm) in docs/data/campaign/thresholds.csv;
- phen: their phenomenological models (phen, phen_b10, both-bases memories), and the same with a noisy
  final readout (phen_nr, phen_b10_nr), which bounds the effect of our noiseless time boundary;
- codecap: their code capacity (codecap, codecap_b10: two noiseless rounds around one layer of data noise,
  r = 2), for sri_m, sri_bm, mwpm and bm, d = 5 to 21;
- in phen and codecap also sri_m1, their matching with Table I to first order only (a diagnostic);
- eq: the equal-qubit comparison below threshold (scripts/campaign_eq.py): XYZ^2 at d = 5 (99 qubits),
  eta = 100 and pure Z, p = 1e-3, 2e-3, 3e-3, r = 3d, both bases.

--decoders replaces sri_m and sri_bm in the circuit and eq parts (for example with their fast link rule,
sri_m_fast and sri_bm_fast, which take the centres and caps of the decoders they vary).

--dset pilot (d = 3 to 9) or main (d = 11 to 21) or all. Budgets, targets and memory are those of
scripts/campaign.py (sri_m as seq_match, sri_bm as bm); eq points stop at 300 errors with the caps of
campaign_eq.py (sri_m as corr_gauge, sri_bm as bm, doubled at the equal-qubit pair).

    python scripts/campaign_sri.py tasks --part phen codecap --dset all --out data/campaign/sri/tasks_valid.jsonl
    python scripts/campaign_sri.py estimate data/campaign/sri/tasks_valid.jsonl --timing docs/data/campaign/sri/timing.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import campaign as C  # noqa: E402
import campaign_eq as EQ  # noqa: E402

CIRCUIT_NOISES = ["sd6", "si1000", "biased10", "biased100", "purez", "em3", "helios_p_noxt", "h2_p_noxt"]
SERIES = ["", "@both-X", "@both-Y"]           # preparation, both-bases memories
SRI = ["sri_m", "sri_bm"]
PILOT_D = [3, 5, 7, 9]
MAIN_D = [11, 13, 15, 17, 19, 21]
# code capacity: their thresholds are 18% to 29%; every decoder gets the wide grid around these centres
CODECAP_CENTER = {"codecap": 0.185, "codecap_b10": 0.21}
CODECAP_D = [5, 7, 9, 11, 13, 15, 17, 19, 21]
CODECAP_DECODERS = ["sri_m", "sri_bm", "mwpm", "bm", "sri_m1"]
# Table I to first order only (the circuit form without the s = 0 pairs), in their models: a diagnostic
FIRST_ORDER = ["sri_m1"]
# the noisy-readout variants take the centres of the model they extend
PHEN_NOISES = {"phen": "phen", "phen_b10": "phen_b10", "phen_nr": "phen", "phen_b10_nr": "phen_b10"}
EQ_NOISES = ["biased100", "purez"]
EQ_CAP_H = {"sri_m": EQ.CAP_H["corr_gauge"], "sri_bm": EQ.CAP_H["bm"]}


def base_of(dec: str) -> str:
    """The decoder a variant varies (sri_m_fast -> sri_m)."""
    return "sri_bm" if dec.startswith("sri_bm") else "sri_m"


def dists(dset: str) -> list:
    return {"pilot": PILOT_D, "main": MAIN_D, "all": PILOT_D + MAIN_D}[dset]


def measured() -> dict:
    """Measured centres of every series of the campaign (campaign.measured_centers)."""
    data = ROOT / "docs" / "data" / "campaign"
    points = data / "points.csv"
    return C.measured_centers(str(data / "thresholds.csv"), str(points if points.exists() else data / "points.csv.gz"))


def series_tasks(label: str, decoders, distances, rounds="d", centers=None, center=None, wide=False,
                 scale=1.0, rep_hours=4.0) -> list:
    """campaign.make_tasks for one noise label, centred on `centers` (measured) or on `center`."""
    if center is None:
        return C.make_tasks([label], decoders, [rounds], distances, scale=scale, rep_hours=rep_hours,
                            centers=centers, wide=wide)
    out = []
    for dec in decoders:
        values = C.grid(label, dec, rounds, c=center, wide=wide)
        for d in distances:
            b = C.budget(dec, d, rounds) * scale
            for p in values:
                bp = max(C.BUDGET[dec][2], b * C.weight(label, dec, rounds, p, center))
                nrep = max(1, math.ceil(bp / (3600.0 * rep_hours)))
                for i in range(nrep):
                    out.append(dict(id=f"{label}|{dec}|r{rounds}|d{d}|p{p:.4g}|{i}", noise=label, decoder=dec,
                                    rounds=rounds, d=d, value=p, target=math.ceil(C.TARGET[dec] / nrep),
                                    max_shots=math.ceil(C.MAX_SHOTS / nrep), budget=round(bp / nrep, 1),
                                    mem=C.memory_for(label, dec, d, rounds)))
    return out


def circuit_tasks(dset: str, rep_hours: float, decoders=SRI) -> list:
    """Each series centred on the measured r = d threshold of its proxy (seq_match, bm) for the same noise
    label, or, where that series has none (si1000@both-Y bm drifts), of the preparation series."""
    meas = measured()
    out = []
    for noise in CIRCUIT_NOISES:
        for s in SERIES:
            for dec in decoders:
                proxy = C.PROXY[dec]
                c = meas.get((noise + s, proxy, "d"), meas.get((noise, proxy, "d")))
                out += series_tasks(noise + s, [dec], dists(dset), center=c, rep_hours=rep_hours)
    return out


def phen_tasks(dset: str, rep_hours: float) -> list:
    meas = measured()
    out = []
    for noise, base in PHEN_NOISES.items():
        for s in ("@both-X", "@both-Y"):
            for dec in SRI + (FIRST_ORDER if noise == base else []):
                proxy = C.PROXY[dec]
                c = meas.get((base + s, proxy, "d"))
                out += series_tasks(noise + s, [dec], dists(dset), center=c, rep_hours=rep_hours)
    return out


def codecap_tasks(dset: str, rep_hours: float) -> list:
    ds = [d for d in CODECAP_D if d in dists(dset)]
    out = []
    for noise, c in CODECAP_CENTER.items():
        for s in ("@both-X", "@both-Y"):
            out += series_tasks(noise + s, CODECAP_DECODERS, ds, rounds="2", center=c, wide=True,
                                rep_hours=rep_hours)
    return out


def variant_tasks(dset: str, rep_hours: float) -> list:
    """Validation diagnostics: their matching and belief-matching with the fast link rule (phen), their
    belief-matching with d iterations of BP (phen and code capacity), centred like the decoders they vary."""
    meas = measured()
    out = []
    for noise in ("phen", "phen_b10"):
        for s in ("@both-X", "@both-Y"):
            for dec in ("sri_m_fast", "sri_bm_dit", "sri_bm_fast"):
                c = meas.get((noise + s, C.PROXY[dec], "d"))
                out += series_tasks(noise + s, [dec], dists(dset), center=c, rep_hours=rep_hours)
    ds = [d for d in CODECAP_D if d in dists(dset)]
    for noise, c in CODECAP_CENTER.items():
        for s in ("@both-X", "@both-Y"):
            out += series_tasks(noise + s, ["sri_bm_dit"], ds, rounds="2", center=c, wide=True, rep_hours=rep_hours)
    return out


def eq_tasks(rep_hours: float = 2.0, decoders=SRI) -> list:
    """XYZ^2 d = 5 at r = 3d with sri_m and sri_bm, the equal-qubit pair of the XZZX code at d = 7."""
    out = []
    for noise in EQ_NOISES:
        for basis in ("X", "Y"):
            for dec in decoders:
                for p in EQ.PS:
                    cap = 3600.0 * EQ_CAP_H[base_of(dec)] * 2          # doubled at the equal-qubit pairs
                    nrep = max(1, math.ceil(cap / (3600.0 * rep_hours)))
                    for i in range(nrep):
                        t = EQ.task("xyz2", basis, noise, dec, 5, p, "3d", math.ceil(EQ.PAIR_TARGET / nrep),
                                    math.ceil(EQ.MAX_SHOTS / nrep), cap / nrep, rep=i)
                        t["mem"] = C.memory_for(t["noise"], dec, 5, "3d")
                        out.append(t)
    return out


def read_tasks(path) -> list:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def estimate(tasks: list, timing_csv: str | None = None) -> dict:
    """
    Cost per part in core-hours: the upper bound (every task uses its budget) and the expected cost, which
    scales the core-hours that the proxy series (seq_match for sri_m, bm for sri_bm, same noise label, r = d)
    took in the campaign by the measured ratio of the decode times per shot (`timing_csv`: decoder, d, ms).
    """
    import pandas as pd

    th = pd.read_csv(ROOT / "docs" / "data" / "campaign" / "thresholds.csv")
    th = th[th.rounds == "d"].set_index(["noise", "decoder"]).core_hours
    decs = sorted({x["decoder"] for x in tasks if x["decoder"].startswith("sri")} | set(SRI))
    ratio = {dec: 1.0 for dec in decs}
    if timing_csv:
        t = pd.read_csv(timing_csv)
        if "config" in t:
            # the campaign's configuration: the proxies as run, sri_* with the 16 likeliest pairs per link round
            t = t[t.config.isin(["campaign", "top 16 pairs"])]
        for dec in decs:
            proxy = C.PROXY[dec]
            own = dec if (t.decoder == dec).any() else base_of(dec)    # a variant without timings: its base
            r = []
            for d in sorted(set(t.d)):
                a = t[(t.decoder == own) & (t.d == d)].ms.mean()
                b = t[(t.decoder == proxy) & (t.d == d)].ms.mean()
                if np.isfinite(a) and np.isfinite(b):
                    r.append(a / b)
            if r:
                ratio[dec] = float(max(r))
    by_series = {}
    for x in tasks:
        k = (x["noise"], x["decoder"], x["rounds"])
        by_series.setdefault(k, [0.0, 0])
        by_series[k][0] += x["budget"] / 3600.0
        by_series[k][1] += 1
    upper = sum(v[0] for v in by_series.values())
    expected, unknown = 0.0, 0.0
    for (noise, dec, r), (b, _) in by_series.items():
        proxy = C.PROXY.get(dec, dec)
        base = noise.replace("_nr", "")
        if r == "d" and (base, proxy) in th.index:
            expected += min(b, float(th[(base, proxy)]) * ratio.get(dec, 1.0))
        else:
            unknown += b
    return dict(tasks=len(tasks), series=len(by_series), upper_core_hours=upper,
                expected_core_hours=expected + unknown, of_which_at_upper_bound=unknown, ratios=ratio)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("tasks")
    a.add_argument("--part", nargs="+", required=True, choices=["circuit", "phen", "codecap", "eq", "variants"])
    a.add_argument("--dset", default="all", choices=["pilot", "main", "all"])
    a.add_argument("--rep-hours", type=float, default=4.0)
    a.add_argument("--out", required=True)
    a.add_argument("--timing", default=None)
    a.add_argument("--decoders", nargs="+", default=SRI, help="circuit and eq parts")
    e = sub.add_parser("estimate")
    e.add_argument("tasks", nargs="+")
    e.add_argument("--timing", default=None)
    args = ap.parse_args()
    if args.cmd == "tasks":
        tasks = []
        for part in args.part:
            if part == "circuit":
                tasks += circuit_tasks(args.dset, args.rep_hours, args.decoders)
            elif part == "phen":
                tasks += phen_tasks(args.dset, args.rep_hours)
            elif part == "codecap":
                tasks += codecap_tasks(args.dset, args.rep_hours)
            elif part == "variants":
                tasks += variant_tasks(args.dset, args.rep_hours)
            else:
                tasks += eq_tasks(decoders=args.decoders)
        tasks.sort(key=lambda t: (-t["budget"], t["id"]))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            for t in tasks:
                fh.write(json.dumps(t) + "\n")
        print(json.dumps(estimate(tasks, args.timing), indent=1))
    else:
        tasks = [t for f in args.tasks for t in read_tasks(f)]
        print(json.dumps(estimate(tasks, args.timing), indent=1))


if __name__ == "__main__":
    main()
