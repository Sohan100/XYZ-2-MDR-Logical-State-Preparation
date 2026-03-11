from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

from xyz2_mdr.constants import (  # noqa: E402
    DEFAULT_NUM_REPLICATES,
    DEFAULT_P_SPAM,
    DEFAULT_ROUNDS,
    DEFAULT_SHOTS,
    NOISE_MODEL_PARAM_NAMES,
    default_probabilities,
)
from xyz2_mdr.workflows import (  # noqa: E402
    build_code_inputs,
    noise_param_names,
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for Slurm run-folder preparation.

    Returns:
        argparse.Namespace: Parsed argument values defining run metadata,
        probabilities, and output locations.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Create a Slurm run folder and metadata for XYZ2 MDR sweeps."
        )
    )
    parser.add_argument("--distance", type=int, required=True)
    parser.add_argument(
        "--noise-model",
        choices=sorted(NOISE_MODEL_PARAM_NAMES),
        required=True,
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--root-dir", type=Path,
                        default=Path("XYZ2-experiment-data-slurm"))
    parser.add_argument("--code-name", type=str, default="xyz2_mdr")
    parser.add_argument("--probabilities", type=float, nargs="+", default=None)
    parser.add_argument("--rounds", type=int, nargs="+",
                        default=DEFAULT_ROUNDS)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--num-replicates", type=int,
                        default=DEFAULT_NUM_REPLICATES)
    parser.add_argument("--p-spam", type=float, default=DEFAULT_P_SPAM)
    parser.add_argument(
        "--recovery-mode",
        choices=["each_round", "final_round"],
        default="each_round",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """
    Create a Slurm run directory, table CSV, and JSON metadata.

    This includes run configuration, working-folder pointers, and code-name
    metadata used by the batch launch script.

    Returns:
        None
    """
    args = parse_args()
    if args.probabilities is not None:
        probabilities = args.probabilities
    else:
        probabilities = default_probabilities()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    run_name = (
        args.run_name
        or f"Run-{timestamp}-d{args.distance}-{args.noise_model}"
    )
    run_dir = args.root_dir / run_name

    if run_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Run directory already exists: {run_dir}. "
            "Use --overwrite to reuse it."
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "partials").mkdir(parents=True, exist_ok=True)

    table_csv = run_dir / f"mdr_table_d{args.distance}.csv"
    build_code_inputs(distance=args.distance, table_csv=table_csv)

    config = {
        "code_name": args.code_name,
        "distance": args.distance,
        "noise_model": args.noise_model,
        "param_names": noise_param_names(args.noise_model),
        "probabilities": list(probabilities),
        "rounds": list(args.rounds),
        "shots": args.shots,
        "num_replicates": args.num_replicates,
        "p_spam": args.p_spam,
        "recovery_mode": args.recovery_mode,
        "table_csv": table_csv.name,
        "created_at_utc": timestamp,
    }

    config_path = run_dir / "run_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    (run_dir /
     "code_name.txt").write_text(f"{args.code_name}\n", encoding="utf-8")

    working_file = args.root_dir / f"working-folder-{args.code_name}.txt"
    args.root_dir.mkdir(parents=True, exist_ok=True)
    working_file.write_text(f"{run_name}\n", encoding="utf-8")
    (args.root_dir /
     "working-folder.txt").write_text(f"{run_name}\n", encoding="utf-8")

    print(f"Run name: {run_name}")
    print(f"Run dir: {run_dir.resolve()}")
    print(f"Config: {config_path.resolve()}")
    print(f"Probabilities: {len(probabilities)} values")


if __name__ == "__main__":
    main()
