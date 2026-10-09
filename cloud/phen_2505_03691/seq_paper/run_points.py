"""
run_points.py
----------------------------------------------------------------------------
Phenomenological threshold points for the comparison with Srivastava et al.,
arXiv:2505.03691. One CSV row per (code, variant, noise, basis, decoder, d, p);
rows already in the CSV are skipped, so runs resume.

Codes and variants:
  xyz2  ours   FTMDRCircuit(d, d, phen, final="frame") (frame start, noiseless frame readout)
  xyz2  paper  PaperPhenCircuit: code-state start, d noisy rounds, noiseless final round of all checks
  xzzx  ours   CompetitorCircuit("xzzx", d, d, phen) (product-state start, noiseless readout)

Decoders: xyz2 -> TwoLevelDecoder modes from scripts/run_decoder_threshold_sweep.DECODERS
(mwpm, bm, seq_soft, bp_full, ...) and seq_hard (paper's sequential decoder, variant paper
only); xzzx -> competitor_decoders (mwpm, bm).

    python cloud/phen_2505_03691/seq_paper/run_points.py --code xyz2 --variant ours --noise phen \
        --bases X Y --decoders mwpm bm --distances 3 5 7 --values 0.03 0.035 0.04 \
        --shots 10000 --out cloud/phen_2505_03691/seq_paper/data/points.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import multiprocessing as mp
import os
import sys
import time
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT / "scripts"), str(ROOT / "src")]

import numpy as np  # noqa: E402

from mdr.ft import CircuitNoise, FTMDRCircuit  # noqa: E402
from mdr.ft.competitor_circuits import CompetitorCircuit  # noqa: E402
from mdr.ft.competitor_decoders import competitor_decoder  # noqa: E402
from phen_paper import SequentialHardDecoder, paper_circuit, two_level  # noqa: E402
from run_decoder_threshold_sweep import DECODERS, SCHEDULES  # noqa: E402

ETA = {"phen": 0.5, "phen_b10": 10.0}
FIELDS = ["code", "variant", "noise", "basis", "decoder", "d", "rounds", "p",
          "shots", "errors", "p_L", "seconds"]


def build(code, variant, noise, basis, decoder, d, p):
    nz = CircuitNoise.phenomenological(p, ETA[noise])
    if code == "xzzx":
        circ = CompetitorCircuit("xzzx", d, d, nz, basis=basis).build()
        return circ, competitor_decoder(circ, decoder)
    # under phen every check is one noiseless MPP, so the gate schedule plays no role
    sched = SCHEDULES["both"]
    if variant == "paper":
        ft = paper_circuit(d, nz, basis, sched)
        dec = SequentialHardDecoder(ft) if decoder == "seq_hard" else two_level(ft, **DECODERS[decoder])
    else:
        ft = FTMDRCircuit(d, d, nz, final="frame", detectors="combined", schedule=sched,
                          logical=basis)
        dec = two_level(ft, **DECODERS[decoder])
    return ft.build(), dec


def run(task):
    code, variant, noise, basis, decoder, d, p, shots, seed = task
    t0 = time.time()
    circ, dec = build(code, variant, noise, basis, decoder, d, p)
    sampler = circ.compile_detector_sampler(seed=seed)
    errors = done = 0
    batch = 20000 if decoder == "mwpm" else 500
    while done < shots:
        n = min(batch, shots - done)
        dets, obs = sampler.sample(n, separate_observables=True)
        pred = dec.decode_batch(dets)
        errors += int(np.sum(np.any(pred != obs, axis=1)))
        done += n
    return dict(code=code, variant=variant, noise=noise, basis=basis, decoder=decoder, d=d,
                rounds=d, p=p, shots=done, errors=errors, p_L=errors / done,
                seconds=round(time.time() - t0, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True, choices=["xyz2", "xzzx"])
    ap.add_argument("--variant", default="ours", choices=["ours", "paper"])
    ap.add_argument("--noise", required=True, choices=sorted(ETA))
    ap.add_argument("--bases", nargs="+", required=True)
    ap.add_argument("--decoders", nargs="+", required=True)
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--values", type=float, nargs="+", required=True)
    ap.add_argument("--shots", type=int, default=10000)
    ap.add_argument("--mwpm-shots", type=int, default=50000)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    done = set()
    if a.out.exists():
        with a.out.open() as fh:
            for r in csv.DictReader(fh):
                done.add((r["code"], r["variant"], r["noise"], r["basis"], r["decoder"],
                          int(r["d"]), float(r["p"])))
    tasks = []
    for basis, dec, d, p in itertools.product(a.bases, a.decoders, a.distances, a.values):
        key = (a.code, a.variant, a.noise, basis, dec, d, p)
        if key in done:
            continue
        shots = a.mwpm_shots if dec == "mwpm" else a.shots
        seed = zlib.crc32(repr((key, shots)).encode())
        tasks.append((a.code, a.variant, a.noise, basis, dec, d, p, shots, seed))
    # longest first
    tasks.sort(key=lambda t: -(t[5] ** 3) * (1 if t[4] == "mwpm" else 30))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    new = not a.out.exists()
    with a.out.open("a", newline="") as fh, mp.Pool(a.workers) as pool:
        w = csv.DictWriter(fh, FIELDS)
        if new:
            w.writeheader()
        for row in pool.imap_unordered(run, tasks):
            w.writerow(row)
            fh.flush()
            print(f"{row['code']}/{row['variant']} {row['noise']} {row['basis']} {row['decoder']} "
                  f"d={row['d']} p={row['p']:g}: {row['p_L']:.4f} ({row['errors']}/{row['shots']}) "
                  f"{row['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
