from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import pandas as pd


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for merging Slurm partial CSV outputs.

    Returns:
        argparse.Namespace: Parsed argument values defining run location and
        copy behavior for merged results.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Merge per-index Slurm partial CSVs into a final sweep CSV."
        )
    )
    parser.add_argument("run_name", type=str)
    parser.add_argument("--root-dir", type=Path,
                        default=Path("XYZ2-experiment-data-slurm"))
    parser.add_argument(
        "--copy-to",
        type=Path,
        default=Path("data/simulation_results"),
        help="Directory to copy merged CSV into.",
    )
    parser.add_argument("--no-copy", action="store_true",
                        help="Do not copy merged CSV into --copy-to.")
    return parser.parse_args()


def main() -> None:
    """
    Merge partial run CSVs and optionally copy the merged result.

    The merged frame is deduplicated and sorted by swept parameters, round,
    and operator before writing.

    Returns:
        None
    """
    args = parse_args()
    run_dir = args.root_dir / args.run_name
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing run config: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    partial_dir = run_dir / "partials"
    partial_files = sorted(partial_dir.glob("result_idx*.csv"))
    if not partial_files:
        raise FileNotFoundError(
            f"No partial result files found in {partial_dir}")

    frames = [pd.read_csv(pf) for pf in partial_files]
    merged = pd.concat(frames, ignore_index=True).drop_duplicates()
    sort_cols = [*config["param_names"], "round", "operator"]
    merged = merged.sort_values(by=sort_cols).reset_index(drop=True)

    merged_name = f"results_{config['noise_model']}_d{config['distance']}.csv"
    merged_path = run_dir / merged_name
    merged.to_csv(merged_path, index=False)
    print(
        f"Merged {len(partial_files)} partial files into "
        f"{merged_path.resolve()}"
    )

    if not args.no_copy:
        args.copy_to.mkdir(parents=True, exist_ok=True)
        copy_target = args.copy_to / merged_name
        shutil.copy2(merged_path, copy_target)
        print(f"Copied merged file to {copy_target.resolve()}")


if __name__ == "__main__":
    main()
