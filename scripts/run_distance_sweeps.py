from __future__ import annotations

import argparse
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
    DEFAULT_RESULTS_DIR,
    DEFAULT_TABLES_DIR,
    DEFAULT_DISTANCES,
    DEFAULT_NUM_REPLICATES,
    DEFAULT_P_SPAM,
    DEFAULT_ROUNDS,
    DEFAULT_SHOTS,
    NOISE_MODEL_DISPLAY_NAMES,
    NOISE_MODEL_PARAM_NAMES,
    default_probabilities,
)
from xyz2_mdr.workflows import run_noise_sweep_with_cache  # noqa: E402


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for local multi-distance sweeps.

    Returns:
        argparse.Namespace: Parsed argument values controlling distances,
        noise models, probabilities, shot count, output locations, and cache
        behavior.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run XYZ2 MDR sweeps for selected distances and noise models."
        )
    )
    parser.add_argument("--distances", type=int, nargs="+",
                        default=DEFAULT_DISTANCES)
    parser.add_argument("--noise-models", nargs="+", choices=sorted(
        NOISE_MODEL_PARAM_NAMES), default=sorted(NOISE_MODEL_PARAM_NAMES))
    parser.add_argument("--probabilities", type=float, nargs="+",
                        default=None, help="Optional custom probability list.")
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
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR)
    parser.add_argument("--output-dir", type=Path,
                        default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def main() -> None:
    """
    Execute requested sweeps with specification-based cache handling.

    For each `(distance, noise_model)` pair, this entry point either loads an
    exact cached simulation result or runs a new sweep and saves it.

    Returns:
        None
    """
    args = parse_args()
    if args.probabilities is not None:
        probabilities = args.probabilities
    else:
        probabilities = default_probabilities()
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for distance in args.distances:
        print(f"\n=== Distance d={distance} ===")
        table_csv = args.tables_dir / f"mdr_table_d{distance}.csv"
        for noise_model in args.noise_models:
            display_name = NOISE_MODEL_DISPLAY_NAMES[noise_model]
            print(f"-> Running {display_name}")
            _, out_csv, loaded = run_noise_sweep_with_cache(
                distance=distance,
                noise_model=noise_model,
                probabilities=probabilities,
                rounds=args.rounds,
                shots=args.shots,
                num_replicates=args.num_replicates,
                p_spam=args.p_spam,
                recovery_mode=args.recovery_mode,
                table_csv=table_csv,
                results_dir=args.output_dir,
                force_rerun=args.force_rerun,
            )
            if loaded:
                print(f"   cache hit | loaded: {out_csv}")
            else:
                print(f"   ran simulation | saved: {out_csv}")

    print("\nDone. All requested sweeps finished.")


if __name__ == "__main__":
    main()
