"""
export_ft_mdr_qasm.py
----------------------------------------------------------------------------
Export the FT MDR circuit for hardware runs (Quantinuum H2 / Helios).

Writes, for each requested distance and round count:

- `ft_mdr_d{d}_r{r}.qasm`: noiseless OpenQASM 2.0 (resets, mid-circuit
  measurements, CX/CY/CZ). Compile it to the native ZZ gate set with pytket
  or Quantinuum Nexus. Keep the classical register order: the decoder reads
  the measurement record in this order.
- `ft_mdr_d{d}_r{r}_{noise}.stim`: the same circuit with detector and
  observable annotations and the chosen noise model. Use it to build the
  matching graph and to decode hardware shots with
  `S0MatchingDecoder(circuit).decode_measurements(bits)`.

    python scripts/export_ft_mdr_qasm.py --distances 3 5 --rounds 1 3 \
        --share-ancillas 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mdr.ft import CircuitNoise, FTMDRCircuit  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--distances", type=int, nargs="+", default=[3])
    ap.add_argument("--rounds", type=int, nargs="+", default=[1])
    ap.add_argument("--noise", default="helios", choices=["helios", "h2"])
    ap.add_argument("--share-ancillas", type=int, default=0,
                    help="reuse k link ancillas for late boundary checks "
                         "(d=5 with k=4 needs 95 qubits, fits Helios)")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "data" / "ft_mdr" / "hardware")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for d in args.distances:
        for r in args.rounds:
            ft = FTMDRCircuit(d, r, CircuitNoise.quantinuum(args.noise),
                              share_ancillas=args.share_ancillas if d > 3 else 0)
            noisy = ft.build()
            stem = args.out_dir / f"ft_mdr_d{d}_r{r}"
            qasm = noisy.without_noise().to_qasm(open_qasm_version=2,
                                                 skip_dets_and_obs=True)
            stem.with_suffix(".qasm").write_text(qasm)
            Path(f"{stem}_{args.noise}.stim").write_text(str(noisy))
            meta = {
                "distance": d,
                "rounds": r,
                "num_qubits": noisy.num_qubits,
                "data_qubits": list(range(ft.geometry.n)),
                "ancilla_of_check": {str(k): v for k, v in ft.ancilla_of.items()},
                "shared_ancilla_pairs": ft.shared_pairs,
                "checks": [ch["spec"] for ch in ft.geometry.checks],
                "frame_basis": {str(q): b for q, b in ft.frame.basis.items()},
                "logical_x": ft.geometry.logicals["Logical X"],
                "num_measurements": noisy.num_measurements,
                "num_detectors": noisy.num_detectors,
                "two_qubit_gates": sum(
                    len(inst.targets_copy()) // 2 for inst in noisy.flattened()
                    if inst.name in {"CX", "CY", "CZ"}
                ),
            }
            Path(f"{stem}.json").write_text(json.dumps(meta, indent=2))
            print(f"d={d} r={r}: {noisy.num_qubits} qubits, "
                  f"{meta['two_qubit_gates']} two-qubit gates, "
                  f"{noisy.num_measurements} measurements -> {stem}.qasm")


if __name__ == "__main__":
    main()
