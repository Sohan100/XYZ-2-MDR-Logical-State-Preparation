"""
run_decoder_threshold_sweep.py
----------------------------------------------------------------------------
Threshold sweeps of the FT MDR protocol for several circuit-level noise
models and decoders. Appends one CSV row per point and skips points that are
already in the file, so interrupted runs can resume.

    python scripts/run_decoder_threshold_sweep.py --noise sd6 \
        --decoders mwpm corr_links --distances 3 5 7 9 \
        --values 3e-3 4e-3 5e-3 6e-3 --out data/ft_mdr/thresholds.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mdr.ft import CircuitNoise, FTMDRCircuit  # noqa: E402
from mdr.ft.two_level_decoder import TwoLevelDecoder  # noqa: E402

NOISE = {
    "sd6": lambda v: CircuitNoise.uniform(v),
    "si1000": lambda v: CircuitNoise.si1000(v),
    "biased10": lambda v: CircuitNoise.biased(v, 10),
    "biased100": lambda v: CircuitNoise.biased(v, 100),
    "purez": lambda v: CircuitNoise.biased(v, 1e9),
    # the same Quantinuum models with v = p/p0 (p0 is the device point), as in docs/data/helios/helios_hcm*.csv
    "helios": lambda v: CircuitNoise.quantinuum("helios", scale=v),
    "h2": lambda v: CircuitNoise.quantinuum("h2", scale=v),
    "helios_cb": lambda v: CircuitNoise.quantinuum("helios", scale=v, two_qubit="cb"),
    # one-parameter Quantinuum models: v is the two-qubit depolarizing probability p
    "helios_p": lambda v: CircuitNoise.trapped_ion(v, "helios"),
    "h2_p": lambda v: CircuitNoise.trapped_ion(v, "h2"),
    "helios_p_noxt": lambda v: CircuitNoise.trapped_ion(v, "helios", crosstalk=False),
    "h2_p_noxt": lambda v: CircuitNoise.trapped_ion(v, "h2", crosstalk=False),
    # entangling-measurement model of Gidney et al. with pair-measurement extraction
    "em3": lambda v: CircuitNoise.em3(v),
}
DECODERS = {
    "mwpm": dict(mode="mwpm"),
    "corr_links": dict(mode="corr_split", lower="links"),
    "corr_gauge": dict(mode="corr_split", lower="gauge"),
    "seq_soft": dict(mode="seq_soft"),
    "seq_match": dict(mode="seq_match"),
    "bp_full": dict(mode="bp_full"),
    "bm": dict(mode="bp_full", bp_method="product_sum", bp_iters=10),
    "bp_corr": dict(mode="bp_corr", bp_method="product_sum", bp_iters=10),
    "tesseract": dict(mode="tesseract"),
    "cfe": dict(mode="cfe", kappa=0.5, osd_order=10, bp_iters=30),
    # CFE without the combination sweep (weaker than CFE under bias; kept for comparison)
    "cfe0": dict(mode="cfe", kappa=0.5, osd_order=0, bp_iters=30, osd_method="osd0"),
    # maximum likelihood by a tensor-network sweep (exact once the bond dimension converges)
    "tnml": dict(mode="tnml", chi=32, chi_max=256),
    # CFE whose decision is replaced by the converged tensor network where the network is
    # narrow enough (r = 1 up to d = 21, r = 2 up to d ~ 9, r = d at d = 3)
    "cfe_tn": dict(mode="cfe_tn", kappa=0.5, osd_order=10, bp_iters=30, chi=32, chi_max=256,
                   tn_max_open=60),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", required=True, choices=sorted(NOISE))
    ap.add_argument("--decoders", nargs="+", required=True,
                    choices=sorted(DECODERS))
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--values", type=float, nargs="+", required=True)
    ap.add_argument("--rounds", default="d")
    ap.add_argument("--final", default="frame")
    ap.add_argument("--max-shots", type=int, default=100_000)
    ap.add_argument("--max-errors", type=int, default=1000)
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    done = set()
    if args.out.exists():
        with args.out.open() as fh:
            for row in csv.DictReader(fh):
                done.add((row["noise"], float(row["value"]), int(row["d"]),
                          int(row["rounds"]), row["decoder"], row["final"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    new = not args.out.exists()
    with args.out.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["noise", "value", "d", "rounds", "final", "decoder",
                        "shots", "errors", "p_L", "stderr", "seconds"])
        for value in args.values:
            for d in args.distances:
                rounds = d if args.rounds == "d" else int(args.rounds)
                ft = FTMDRCircuit(d, rounds, NOISE[args.noise](value),
                                  final=args.final, detectors="combined")
                for name in args.decoders:
                    key = (args.noise, value, d, rounds, name, args.final)
                    if key in done:
                        continue
                    t0 = time.time()
                    dec = TwoLevelDecoder(ft, **DECODERS[name])
                    slow = name in ("seq_soft", "seq_match", "bp_full", "bm", "tnml", "cfe_tn",
                                    "tesseract", "cfe")
                    batch = args.batch or (500 if slow else 20000)
                    est = dec.estimate(max_shots=args.max_shots,
                                       max_errors=args.max_errors,
                                       batch=batch,
                                       time_limit=args.time_limit)
                    w.writerow([args.noise, value, d, rounds, args.final,
                                name, est.shots, est.errors, est.rate,
                                est.stderr, round(time.time() - t0, 1)])
                    fh.flush()
                    print(f"{args.noise} v={value:g} d={d} {name}: "
                          f"pL={est.rate:.3e} ({est.errors}/{est.shots}) "
                          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
