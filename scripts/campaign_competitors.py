"""
campaign_competitors.py
----------------------------------------------------------------------------
Competitor codes under the noise models of the campaign, to compare their thresholds with XYZ^2's:

- the rotated CSS, XZZX and XY surface codes, as memories in the X and Z basis;
- the periodic honeycomb Floquet code (Hastings-Haah, Gidney et al.), with the H and V observables
  (bases X and Z).

The circuits come from src/mdr/ft/competitor_circuits.py, which uses the same noise conventions as
FTMDRCircuit. They are decoded by src/mdr/ft/competitor_decoders.py. Every series is named
"code-basis:decoder" (see campaign.competitor) and uses the budget, target and memory of the XYZ^2
decoder of the same kind (campaign.COMPETITOR_DECODERS).

    python scripts/campaign_competitors.py --out data/campaign/tasks9.jsonl \
        --centers /pscratch/sd/s/sohan100/ftmdr_tools/competitors/pilot_centers.csv

Grids have 14 points over a factor 4, centred on the series' pilot crossing (MWPM, d = 5/7, or 4/8 for
the honeycomb, whose d = 4, 8, ..., 20 patches share one shape). For the other decoders the centre is multiplied by the gain our own decoders show over
MWPM (campaign._G_DEFAULT). A series without a pilot crossing is centred at twice the top of the
scanned range. A crosstalk model without a crossing takes the fixed range of our own runs
(campaign.XT_RANGE).
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

BASES = ("X", "Z")
CODES = ("css", "xzzx", "xy", "honeycomb")
DECODERS = ("mwpm", "corr", "bm", "tesseract", "bposd")
# beam search and OSD grow fast with the circuit; the honeycomb is decoded by matching and belief-matching
DMAX = {"tesseract": 11, "bposd": 11}
HONEYCOMB_DECODERS = ("mwpm", "corr", "bm")
SURFACE_D = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
# the honeycomb patch is d x 6 ceil(d / 4): d = 4, 8, ..., 20 keep the same shape (d x 1.5 d), which the
# fit of a threshold across d needs (d = 6 has the height of d = 8)
HONEYCOMB_D = [4, 8, 12, 16, 20]
# pilot rows to centre on: the largest pair for the surface codes, the pair of equal shapes for the honeycomb
PILOT_PAIR = {"css": "5/7", "xzzx": "5/7", "xy": "5/7", "honeycomb": "4/8"}
PILOT_BASIS = {"H": "X", "V": "Z", "X": "X", "Z": "Z"}
GAIN = {"mwpm": 1.0, "corr": 1.18, "bm": 1.3, "tesseract": 1.3, "bposd": 1.3}


def pilot_centers(path: str) -> dict:
    """{(code, basis, noise, rounds): (crossing or None, lo, hi)} from the pilot CSV (MWPM rows; corr for xy
    when MWPM has no crossing)."""
    def num(v):
        try:
            x = float(v)
        except (TypeError, ValueError):
            return None
        return None if math.isnan(x) else x

    rows = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("d_pair") and r["d_pair"] != PILOT_PAIR.get(r["code"], r["d_pair"]):
                continue
            key = (r["code"], PILOT_BASIS[r["basis"]], r["noise"], r["rounds"])
            rows.setdefault(key, {})[r["decoder"]] = (num(r.get("crossing_p")), num(r.get("lo")), num(r.get("hi")))
    out = {}
    for key, by in rows.items():
        found = [by[x] for x in ("mwpm", "corr") if x in by and by[x][0] is not None]
        out[key] = found[0] if found else by.get("mwpm", next(iter(by.values())))
    return out


def pilot_center(pilot: dict, code: str, basis: str, noise: str, r: str):
    """(crossing, lo, hi) of a series; for r other than 1 and d, the crossing interpolated in log r between
    r = 1 and r = d (placed at campaign.R_EFF_D rounds), as campaign.center_from does for our code."""
    if r in ("1", "d") or (code, basis, noise, "1") not in pilot or (code, basis, noise, "d") not in pilot:
        return pilot.get((code, basis, noise, r), (None, None, None))
    x1, xd = pilot[(code, basis, noise, "1")][0], pilot[(code, basis, noise, "d")][0]
    if x1 is None or xd is None:
        return (None, None, None)
    t = min(1.0, math.log(int(r)) / math.log(C.R_EFF_D))
    return (math.exp((1 - t) * math.log(x1) + t * math.log(xd)), None, None)


def series_grid(noise: str, code: str, basis: str, r: str, dec: str, pilot: dict):
    """(p values, centre) of a series."""
    x, lo, hi = pilot_center(pilot, code, basis, noise, r)
    if x is None:                                   # e.g. no crossing below 25% (phen_b10, r = 1): the other basis
        x = pilot_center(pilot, code, "Z" if basis == "X" else "X", noise, r)[0]
    if x is None and noise in C.XT_RANGE:
        return C.grid(noise, "mwpm", r), C.center(noise, "mwpm", r)
    if x is None:
        x = 2.0 * (hi if hi else C.center(noise, "mwpm", r))
    c = x * GAIN[dec]
    # every rate of a noise model must stay a probability (phenomenological r = 1 crossings reach 20%)
    return [v for v in (float(f"{c * C.STEP ** k:.4g}") for k in C.KRANGE["wide"]) if v < 0.45], c


def make_tasks(noises, rounds, pilot: dict, scale: float = 1.0, rep_hours: float = 2.0, codes=CODES,
               decoders=DECODERS, dmax=None) -> list:
    caps = dict(DMAX if dmax is None else dmax)
    out = []
    for code in codes:
        ds = HONEYCOMB_D if code == "honeycomb" else SURFACE_D
        decs = [x for x in decoders if code != "honeycomb" or x in HONEYCOMB_DECODERS]
        for basis in BASES:
            for dec in decs:
                name = f"{code}-{basis}:{dec}"
                base = C.base_decoder(name)
                for noise in noises:
                    for r in rounds:
                        values, c = series_grid(noise, code, basis, r, dec, pilot)
                        for d in ds:
                            if d > caps.get(dec, 99):
                                continue
                            b = C.budget(base, d, r) * scale
                            for p in values:
                                bp = max(C.BUDGET[base][2], b * C.weight(noise, base, r, p, c))
                                nrep = max(1, math.ceil(bp / (3600.0 * rep_hours)))
                                for i in range(nrep):
                                    out.append(dict(
                                        id=f"{noise}|{name}|r{r}|d{d}|p{p:.4g}|{i}", noise=noise, decoder=name,
                                        rounds=r, d=d, value=p, target=math.ceil(C.TARGET[base] / nrep),
                                        max_shots=math.ceil(C.MAX_SHOTS / nrep), budget=round(bp / nrep, 1),
                                        mem=C.memory(base, d, r),
                                        **({"mem_run": C.memory_run(base, d, r)} if base in C.RUN_FRACTION else {})))
    out.sort(key=lambda t: (-t["budget"], t["id"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--centers", required=True, help="pilot crossings (code, basis, noise, rounds, decoder, "
                                                     "d_pair, crossing_p, lo, hi, note)")
    ap.add_argument("--noises", nargs="+", default=C.NOISES)
    ap.add_argument("--rounds", nargs="+", default=["d", "1"])
    ap.add_argument("--codes", nargs="+", default=list(CODES))
    ap.add_argument("--decoders", nargs="+", default=list(DECODERS))
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every time budget")
    ap.add_argument("--rep-hours", type=float, default=2.0)
    args = ap.parse_args()
    tasks = make_tasks(args.noises, args.rounds, pilot_centers(args.centers), args.scale, args.rep_hours,
                       args.codes, args.decoders)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        for t in tasks:
            fh.write(json.dumps(t) + "\n")
    est = C.estimate(tasks)
    by = {}
    for k, v in est.items():
        by[k.split(":")[1]] = by.get(k.split(":")[1], 0.0) + v
    print(f"{len(tasks)} tasks ({len({t['id'].rsplit('|', 1)[0] for t in tasks})} points) -> {args.out}")
    print("upper bound of the cost (core-hours): " + ", ".join(f"{k} {v:,.0f}" for k, v in sorted(by.items()))
          + f"; total {sum(est.values()):,.0f} = {sum(est.values()) / 128:,.0f} node-hours at 128 cores")


if __name__ == "__main__":
    main()
