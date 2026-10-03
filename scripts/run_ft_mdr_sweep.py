"""
run_ft_mdr_sweep.py
----------------------------------------------------------------------------
Sweep the fault-tolerant MDR protocol over distance and noise strength.

Examples
--------
Uniform circuit-level depolarizing noise, one extraction round:

    python scripts/run_ft_mdr_sweep.py --noise uniform --rounds 1 \
        --distances 3 5 7 --values 1e-3 2e-3 4e-3 6e-3 8e-3

One-parameter Helios model (CircuitNoise.trapped_ion) with two-qubit error
rate p from half to twice the device point p0 = 1e-3, d rounds:

    python scripts/run_ft_mdr_sweep.py --noise helios --rounds d \
        --distances 3 5 7 --values 5e-4 1e-3 1.5e-3 2e-3
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mdr.ft import CircuitNoise, FTMDRCircuit, S0MatchingDecoder  # noqa: E402
from mdr.ft.circuit_noise import QUANTINUUM_P2  # noqa: E402


def make_noise(kind: str, value: float, memory_scale: float) -> CircuitNoise:
    if kind == "uniform":
        return CircuitNoise.uniform(value)
    if kind == "si1000":
        return CircuitNoise.si1000(value)
    # value is the two-qubit error rate p; every other rate keeps its data-sheet ratio to p
    return CircuitNoise.quantinuum(kind, scale=value / QUANTINUUM_P2[kind], memory_scale=memory_scale)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--noise", default="uniform",
                    choices=["uniform", "si1000", "helios", "h2"])
    ap.add_argument("--values", type=float, nargs="+", required=True,
                    help="p for uniform/si1000, the two-qubit error rate p for helios/h2 "
                         "(device points 1e-3 and 1.875e-3)")
    ap.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--rounds", default="1", help="an integer or 'd'")
    ap.add_argument("--final", default="frame", choices=["frame", "ideal"])
    ap.add_argument("--init", default="frame", choices=["frame", "plus"])
    ap.add_argument("--decoder", default="pymatching",
                    choices=["pymatching", "bposd", "tesseract"])
    ap.add_argument("--memory-scale", type=float, default=1.0)
    ap.add_argument("--max-shots", type=int, default=2_000_000)
    ap.add_argument("--max-errors", type=int, default=1000)
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data" / "ft_mdr" / "ft_mdr_results.csv")
    args = ap.parse_args()

    detectors = "all" if args.init == "plus" else "s0"
    if args.init == "plus" and args.decoder == "pymatching":
        print("note: init=plus has a hypergraph DEM; prefer --decoder tesseract")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    new = not args.out.exists()
    with args.out.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if new:
            writer.writerow(["noise", "value", "memory_scale", "d", "rounds",
                             "init", "final", "detectors", "decoder", "shots",
                             "errors", "p_L", "stderr", "seconds"])
        for value in args.values:
            for d in args.distances:
                rounds = d if args.rounds == "d" else int(args.rounds)
                circuit = FTMDRCircuit(
                    d, rounds, make_noise(args.noise, value, args.memory_scale),
                    init=args.init, final=args.final, detectors=detectors,
                ).build()
                est = S0MatchingDecoder(circuit, args.decoder).estimate(
                    max_shots=args.max_shots, max_errors=args.max_errors,
                    time_limit=args.time_limit,
                )
                writer.writerow([args.noise, value, args.memory_scale, d, rounds,
                                 args.init, args.final, detectors, args.decoder,
                                 est.shots, est.errors, est.rate, est.stderr,
                                 round(est.seconds, 1)])
                fh.flush()
                print(f"{args.noise} value={value:g} d={d} r={rounds} "
                      f"p_L={est.rate:.3e} ({est.errors}/{est.shots})")


if __name__ == "__main__":
    main()
