"""
eq_timing.py
----------------------------------------------------------------------------
CPU seconds per shot of every decoder of the equal-qubit runs (scripts/campaign_eq.py), r = 3d, for planning
the budgets: sampling and decoding of one large batch, timed with the process CPU clock (decoder construction
excluded), so that a loaded login node does not inflate the numbers.

    python scripts/eq_timing.py --out data/campaign/eq/timing.csv --workers 12
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import campaign_eq as E  # noqa: E402

CASES = [("xyz2", "X", "sd6"), ("css", "X", "sd6"), ("xzzx", "X", "biased100")]


def one(job):
    code, basis, noise, dec, d, p = job
    import numpy as np

    from run_decoder_threshold_sweep import DECODERS, NOISE, SCHEDULES

    t_build = time.process_time()
    if code == "xyz2":
        from mdr.ft import FTMDRCircuit
        from mdr.ft.two_level_decoder import TwoLevelDecoder

        ft = FTMDRCircuit(d, 3 * d, NOISE[noise](p), final="frame", detectors="combined",
                          schedule=SCHEDULES["both"], logical=basis)
        decoder = TwoLevelDecoder(ft, **DECODERS[dec])
        circuit = decoder.circuit
    else:
        from mdr.ft.competitor_circuits import competitor_circuit
        from mdr.ft.competitor_decoders import competitor_decoder

        circuit = competitor_circuit(code, d, 3 * d, NOISE[noise](p), basis)
        decoder = competitor_decoder(circuit, dec)
    build = time.process_time() - t_build
    sampler = circuit.compile_detector_sampler()
    n, spent = 64, 0.0
    shots = 0
    while spent < 6.0 and shots < 2_000_000:
        dets, obs = sampler.sample(n, separate_observables=True)
        t0 = time.process_time()
        decoder.decode_batch(dets)
        spent += time.process_time() - t0
        shots += n
        n = min(4 * n, 200_000)
    t0 = time.process_time()
    sampler.sample(min(shots, 100_000), separate_observables=True)
    samp = (time.process_time() - t0) / min(shots, 100_000)
    return dict(code=code, basis=basis, noise=noise, dec=dec, d=d, p=p, shots=shots,
                decode_s=spent / shots, sample_s=samp, build_s=build)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    jobs = [(code, basis, noise, dec, d, p) for code, basis, noise in CASES for dec in E.DECODERS[code]
            for d in E.distances(code) for p in E.PS]
    jobs.sort(key=lambda j: -j[4])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers) as pool, open(args.out, "w", newline="") as fh:
        w = None
        for row in pool.imap_unordered(one, jobs):
            if w is None:
                w = csv.DictWriter(fh, fieldnames=list(row))
                w.writeheader()
            w.writerow(row)
            fh.flush()
            print(row, flush=True)


if __name__ == "__main__":
    main()
