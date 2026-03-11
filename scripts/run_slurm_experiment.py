from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _ensure_src_on_path() -> None:
    """
    Add the repository `src/` directory to `sys.path` if needed.

    This allows running the script directly via `python scripts/...` without
    requiring prior package installation.

    Returns:
        None
    """
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_ensure_src_on_path()

from xyz2_mdr.mdr_noise_sweep import MdrNoiseSweep  # noqa: E402
from xyz2_mdr.workflows import (  # noqa: E402
    build_code_inputs,
    noise_param_names,
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for one Slurm probability-index task.

    Returns:
        argparse.Namespace: Parsed argument values containing run name, index,
        root directory, and overwrite policy.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run one probability index for a prepared Slurm XYZ2 MDR run."
        )
    )
    parser.add_argument("run_name", type=str)
    parser.add_argument("probability_index", type=int)
    parser.add_argument("--root-dir", type=Path,
                        default=Path("XYZ2-experiment-data-slurm"))
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing partial output file.")
    return parser.parse_args()


def main() -> None:
    """
    Execute one indexed sweep point and write its partial CSV output.

    This entry point loads run metadata, resolves one probability value by
    index, runs the corresponding sweep slice, and stores the partial result.

    Returns:
        None
    """
    args = parse_args()
    run_dir = args.root_dir / args.run_name
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing run config: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    probs = list(config["probabilities"])
    idx = args.probability_index
    if idx < 0 or idx >= len(probs):
        raise IndexError(
            f"probability_index {idx} out of range [0, {len(probs) - 1}]")

    p_val = float(probs[idx])
    shots = int(config["shots"])
    shots_override_file = run_dir / "shots.txt"
    if shots_override_file.exists():
        text = shots_override_file.read_text(encoding="utf-8").strip()
        if text:
            shots = int(text)

    table_csv = run_dir / \
        config.get("table_csv", f"mdr_table_d{config['distance']}.csv")
    code_inputs = build_code_inputs(distance=int(
        config["distance"]), table_csv=table_csv)
    param_names = noise_param_names(str(config["noise_model"]))

    out_csv = run_dir / "partials" / f"result_idx{idx:03d}.csv"
    if out_csv.exists() and not args.force:
        print(f"Partial result already exists, skipping: {out_csv}")
        return

    MdrNoiseSweep(
        # type: ignore[arg-type]
        code_stabilizers=code_inputs["code_stabilizers"],
        toggles=code_inputs["combined_toggles"],  # type: ignore[arg-type]
        # type: ignore[arg-type]
        measure_stabilizers=code_inputs["stabilizers"],
        # type: ignore[arg-type]
        logical_operators=code_inputs["logical_operators"],
        ancillas=1,
        psi_circuit=code_inputs["psi_circuit"],
        p_spam=float(config["p_spam"]),
        recovery_mode=str(config.get("recovery_mode", "each_round")),
        param_names=param_names,
        param_values=[p_val],
        round_list=[int(x) for x in config["rounds"]],
        shots=shots,
        num_replicates=int(config["num_replicates"]),
        save_data_filename=out_csv,
    )

    print(f"Completed probability idx={idx}, p={p_val}")
    print(f"Saved: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
