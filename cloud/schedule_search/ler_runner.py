"""
ler_runner.py
----------------------------------------------------------------------------
Monte-Carlo logical error rates of XYZ^2 compilations and of the XZZX
reference, in parallel over points. One CSV row per point; points already in
the output file are skipped.

A point is (code, comp, noise, p, d, logical, decoder):

- code "xyz2": comp is "s<idx>" (schedule `enumerate_depth(6)[idx]`) or a
  name in `compilations.CUSTOM`; decoder in run_decoder_threshold_sweep.DECODERS.
- code "xzzx": comp is "-", logical is the CSS basis "X" or "Z", decoder in
  competitor_decoders.DECODERS.

    python cloud/schedule_search/ler_runner.py --code xyz2 --comps s1525 \
        --noises sd6 --ps 5e-3 6e-3 --ds 5 7 --logicals X Y --decoders mwpm \
        --out cloud/schedule_search/data/ler.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
from multiprocessing import Pool
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))

from compilations import NOISE, build_ft, xyz2_decoder  # noqa: E402
from mdr.ft.competitor_circuits import competitor_circuit  # noqa: E402
from mdr.ft.competitor_decoders import competitor_decoder  # noqa: E402

COLS = ["code", "comp", "noise", "p", "d", "rounds", "logical", "decoder",
        "shots", "errors", "p_L", "stderr", "seconds"]
OPT = {}


def run(pt):
    code, comp, noise, p, d, lg, dec = pt
    t0 = time.time()
    if code == "xzzx":
        c = competitor_circuit("xzzx", d, d, NOISE[noise](p), basis=lg)
        D = competitor_decoder(c, dec)
        slow = dec in ("bm", "bposd", "tesseract")
    else:
        ft = build_ft(comp, d, d, NOISE[noise](p), lg)
        D = xyz2_decoder(ft, dec)
        slow = dec not in ("mwpm", "corr_links", "corr_gauge")
    est = D.estimate(max_shots=OPT["max_shots"], max_errors=OPT["max_errors"],
                     batch=500 if slow else 10000, time_limit=OPT["time_limit"])
    return [code, comp, noise, p, d, d, lg, dec, est.shots, est.errors,
            est.rate, est.stderr, round(time.time() - t0, 1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="xyz2", choices=["xyz2", "xzzx"])
    ap.add_argument("--comps", nargs="+", default=["-"])
    ap.add_argument("--noises", nargs="+", required=True)
    ap.add_argument("--ps", type=float, nargs="+", required=True)
    ap.add_argument("--ds", type=int, nargs="+", required=True)
    ap.add_argument("--logicals", nargs="+", default=["X", "Y"])
    ap.add_argument("--decoders", nargs="+", default=["mwpm"])
    ap.add_argument("--max-shots", type=int, default=200_000)
    ap.add_argument("--max-errors", type=int, default=400)
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    OPT.update(max_shots=a.max_shots, max_errors=a.max_errors, time_limit=a.time_limit)
    done = set()
    if a.out.exists():
        with a.out.open() as fh:
            for r in csv.DictReader(fh):
                done.add((r["code"], r["comp"], r["noise"], float(r["p"]), int(r["d"]),
                          r["logical"], r["decoder"]))
    pts = [pt for pt in itertools.product([a.code], a.comps, a.noises, a.ps, a.ds,
                                          a.logicals, a.decoders) if pt not in done]
    # largest first, so the long points do not end up alone at the end
    pts.sort(key=lambda pt: (-pt[4], pt[6] == "mwpm"))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    new = not a.out.exists()
    with a.out.open("a", newline="") as fh, Pool(a.workers, maxtasksperchild=1) as pool:
        w = csv.writer(fh)
        if new:
            w.writerow(COLS)
        for row in pool.imap_unordered(run, pts):
            w.writerow(row)
            fh.flush()
            print(" ".join(str(x) for x in row), flush=True)


if __name__ == "__main__":
    main()
