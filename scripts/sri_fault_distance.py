"""
sri_fault_distance.py
----------------------------------------------------------------------------
Fewest faults that make a decoder fail: every single fault and every pair (and triple, if asked) of
fault mechanisms of the detector error model, decoded as one shot each. A set whose syndrome is empty
but which flips the observable is an undetectable logical error (the circuit's fault distance), not a
decoder failure, and is counted apart.

    python scripts/sri_fault_distance.py --noise sd6 --p 3e-3 --d 3 --size 3 \
        --decoders mwpm corr_gauge sri_m sri_bm --workers 64 --out docs/data/campaign/sri/fault_distance.csv
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import itertools  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mdr.ft import FTMDRCircuit  # noqa: E402
from mdr.ft.two_level_decoder import TwoLevelDecoder  # noqa: E402
from run_decoder_threshold_sweep import DECODERS, NOISE, SCHEDULES  # noqa: E402

CHUNK = 4000
_G = {}


def _chunk(args):
    """Failures and undetectable logicals among the fault sets of one chunk of combinations."""
    combos = args
    dec, M, O = _G["dec"], _G["M"], _G["O"]
    syn = np.bitwise_xor.reduce(M[combos], axis=1)
    obs = np.bitwise_xor.reduce(O[combos], axis=1)
    quiet = ~syn.any(axis=1)
    undet = int((quiet & obs.any(axis=1)).sum())
    live = ~quiet
    fails = []
    if live.any():
        pred = np.asarray(dec.decode_batch(syn[live]), dtype=np.uint8)
        bad = np.flatnonzero((pred != obs[live]).any(axis=1))
        fails = [tuple(int(x) for x in combos[live][b]) for b in bad]
    return len(combos), undet, fails


def run(noise, p, d, rounds, schedule, logical, decoder, size, workers, log=print):
    ft = FTMDRCircuit(d, rounds, NOISE[noise](p), final="frame", detectors="combined",
                      schedule=SCHEDULES[schedule], logical=logical)
    dec = TwoLevelDecoder(ft, **DECODERS[decoder])
    nd, no = dec.circuit.num_detectors, dec.circuit.num_observables
    nm = len(dec.mech_keys)
    M = np.zeros((nm, nd), dtype=np.uint8)
    O = np.zeros((nm, no), dtype=np.uint8)
    for j, (dets, obs) in enumerate(dec.mech_keys):
        M[j, list(dets)] = 1
        O[j, list(obs)] = 1
    _G.update(dec=dec, M=M, O=O)
    out = []
    for k in range(1, size + 1):
        t0 = time.time()
        it = itertools.combinations(range(nm), k)

        def chunks():
            while True:
                block = list(itertools.islice(it, CHUNK))
                if not block:
                    return
                yield np.array(block, dtype=np.int64)

        tot = undet = 0
        fails = []
        if workers > 1:
            with mp.get_context("fork").Pool(workers) as pool:
                for n, u, f in pool.imap_unordered(_chunk, chunks(), chunksize=1):
                    tot, undet = tot + n, undet + u
                    fails += f
        else:
            for c in chunks():
                n, u, f = _chunk(c)
                tot, undet = tot + n, undet + u
                fails += f
        row = dict(noise=noise, p=p, d=d, rounds=rounds, schedule=schedule, logical=logical, decoder=decoder,
                   size=k, mechanisms=nm, sets=tot, undetectable=undet, failures=len(fails),
                   failure_prob=float(sum(np.prod(dec.priors[list(f)]) for f in fails)),
                   example=";".join(str(dec.mech_keys[j]) for j in fails[0]) if fails else "",
                   seconds=round(time.time() - t0, 1))
        log(f"{noise} d={d} {decoder} size {k}: {len(fails)} failures, {undet} undetectable, "
            f"{tot} sets, {row['seconds']} s")
        out.append(row)
        if fails or undet:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", default="sd6")
    ap.add_argument("--p", type=float, default=3e-3)
    ap.add_argument("--d", type=int, nargs="+", default=[3, 5])
    ap.add_argument("--rounds", default="d")
    ap.add_argument("--schedule", default="depth6")
    ap.add_argument("--logical", default="X")
    ap.add_argument("--decoders", nargs="+", default=["mwpm", "corr_gauge", "sri_m", "sri_bm"])
    ap.add_argument("--size", type=int, nargs="+", default=[3, 2], help="largest set size, per distance")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sizes = dict(zip(args.d, args.size + [args.size[-1]] * len(args.d)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for d in args.d:
        r = d if args.rounds == "d" else int(args.rounds)
        for name in args.decoders:
            rows = run(args.noise, args.p, d, r, args.schedule, args.logical, name, sizes[d], args.workers)
            new = not args.out.exists()
            with args.out.open("a", newline="") as fh:           # every decoder as soon as it is done
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                if new:
                    w.writeheader()
                w.writerows(rows)


if __name__ == "__main__":
    main()
