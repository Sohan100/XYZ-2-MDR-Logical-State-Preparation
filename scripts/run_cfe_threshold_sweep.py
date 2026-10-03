"""
run_cfe_threshold_sweep.py
----------------------------------------------------------------------------
Threshold sweep of the coset free-energy (CFE) decoder with paired MWPM, HCM
and belief-matching decoding of the same shots (r = d rounds).

For every shot and every CFE candidate the script stores the weight w and the
local entropy S, so the decision can be recomputed for any kappa without
decoding again. One .npz file is written per (noise, d, p):

    python scripts/run_cfe_threshold_sweep.py --noise sd6 --distance 5 \
        --shots 2000 --seed 102 --values 0.005 0.0055 0.006 0.0065 0.007 \
        --out docs/data/cfe

`paper/analysis/cfe_plots.py` reads the files, fits the thresholds and draws
the comparison figure.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mdr.ft import CircuitNoise, FTMDRCircuit  # noqa: E402
from mdr.ft.two_level_decoder import TwoLevelDecoder  # noqa: E402

NOISE = {
    "sd6": lambda v: CircuitNoise.uniform(v),
    "si1000": lambda v: CircuitNoise.si1000(v),
    "b10": lambda v: CircuitNoise.biased(v, 10),
    "b100": lambda v: CircuitNoise.biased(v, 100),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[3])
    ap.add_argument("--noise", required=True, choices=sorted(NOISE))
    ap.add_argument("--distance", type=int, required=True)
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--values", type=float, nargs="+", required=True)
    ap.add_argument("--kappa", type=float, default=0.5, help="only used for the printed summary")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "data" / "cfe")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    d = args.distance
    for p in args.values:
        fn = args.out / f"{args.noise}_d{d}_p{p}_s{args.seed}.npz"
        if fn.exists():
            continue
        t0 = time.time()
        ft = FTMDRCircuit(d, d, NOISE[args.noise](p), detectors="combined")
        cfe = TwoLevelDecoder(ft, mode="cfe", kappa=args.kappa, cfe_guided=True)
        others = {"mwpm": TwoLevelDecoder(ft, mode="mwpm"),
                  "hcm": TwoLevelDecoder(ft, mode="corr_split", lower="gauge"),
                  "bm": TwoLevelDecoder(ft, mode="bp_full", bp_method="product_sum", bp_iters=10)}
        dets, obs = cfe.circuit.compile_detector_sampler(seed=args.seed).sample(
            args.shots, separate_observables=True)
        preds = {k: v.decode_batch(dets)[:, 0] for k, v in others.items()}
        t1 = time.time()
        C = cfe._cfe
        W = np.full((args.shots, 2, 2), np.inf)
        S = np.zeros((args.shots, 2, 2))
        for i in range(args.shots):
            if not dets[i].any():
                W[i, 0, :] = 0.0
                continue
            for ell, cands in enumerate(C.class_candidates(dets[i])):
                for k, mask in enumerate(cands):
                    C.descend(mask)
                    W[i, ell, k], S[i, ell, k] = C.free_energy(mask)
        t2 = time.time()
        np.savez(fn, W=W, S=S, obs=obs[:, 0], **{f"pred_{k}": v for k, v in preds.items()},
                 t_cfe=(t2 - t1) / args.shots, t_setup=t1 - t0)
        F = (W - args.kappa * S).min(axis=2)
        e_cfe = int(np.sum((F[:, 1] < F[:, 0]) != obs[:, 0]))
        msg = " ".join(f"{k}={int(np.sum(v != obs[:, 0]))}" for k, v in preds.items())
        print(f"{args.noise} d={d} p={p}: cfe={e_cfe} {msg} /{args.shots}  "
              f"{1e3 * (t2 - t1) / args.shots:.0f} ms/shot", flush=True)


if __name__ == "__main__":
    main()
