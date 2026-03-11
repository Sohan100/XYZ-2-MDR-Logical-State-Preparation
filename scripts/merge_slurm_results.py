from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import pandas as pd


def _ensure_src_on_path() -> None:
    """
    Add the repository `src/` directory to `sys.path` if needed.

    Returns:
        None
    """
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_ensure_src_on_path()

from xyz2_mdr.workflows import (  # noqa: E402
    build_simulation_spec,
    simulation_results_path,
    simulation_spec_path,
)


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
    parser.add_argument(
        "--tables-copy-to",
        type=Path,
        default=Path("data/tables"),
        help="Directory to copy MDR table CSV into.",
    )
    parser.add_argument("--no-copy", action="store_true",
                        help="Do not copy merged CSV into --copy-to.")
    parser.add_argument(
        "--no-table-copy",
        action="store_true",
        help="Do not copy MDR table CSV into --tables-copy-to.",
    )
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
        spec = build_simulation_spec(
            distance=int(config["distance"]),
            noise_model=str(config["noise_model"]),
            probabilities=[float(p) for p in config["probabilities"]],
            rounds=[int(r) for r in config["rounds"]],
            shots=int(config["shots"]),
            num_replicates=int(config["num_replicates"]),
            p_spam=float(config["p_spam"]),
            recovery_mode=str(config.get("recovery_mode", "each_round")),
        )
        copy_target = simulation_results_path(args.copy_to, spec)
        copy_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(merged_path, copy_target)
        spec_path = simulation_spec_path(copy_target)
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        print(f"Copied merged file to {copy_target.resolve()}")
        print(f"Wrote simulation spec to {spec_path.resolve()}")

    if not args.no_table_copy:
        table_name = str(config.get("table_csv", f"mdr_table_d{config['distance']}.csv"))
        table_src = run_dir / table_name
        if not table_src.exists():
            raise FileNotFoundError(f"Missing table CSV: {table_src}")
        args.tables_copy_to.mkdir(parents=True, exist_ok=True)
        table_dst = args.tables_copy_to / f"mdr_table_d{int(config['distance'])}.csv"
        shutil.copy2(table_src, table_dst)
        print(f"Copied table CSV to {table_dst.resolve()}")


if __name__ == "__main__":
    main()
