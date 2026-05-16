"""

report_tableau_destabilizers.py
----------------------------------------------------------------------------
Command-line utilities for report tableau destabilizers.
"""

from __future__ import annotations

import argparse
from pathlib import Path
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

from mdr.constants import DEFAULT_DISTANCES  # noqa: E402
from xyz2.tableau_analysis import build_destabilizer_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for tableau destabilizer reporting.

    Returns:
    argparse.Namespace: Parsed distances, output path, and random seed.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compare generated MDR toggles against Stim tableau "
            "destabilizers for selected code distances."
        )
    )
    parser.add_argument(
        "--distances",
        type=int,
        nargs="+",
        default=DEFAULT_DISTANCES,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path for the full row-by-row report.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Seed forwarded to the robust toggle generator.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Build and optionally save a destabilizer comparison report.

    Returns:
    None
    """
    args = parse_args()
    reports = [
        build_destabilizer_report(distance=d, random_seed=args.random_seed)
        for d in args.distances
    ]
    report = pd.concat(reports, ignore_index=True)

    summary = report.groupby("distance", as_index=False).agg(
        generated_total_weight=("generated_weight", "sum"),
        generated_avg_weight=("generated_weight", "mean"),
        generated_max_weight=("generated_weight", "max"),
        stim_total_weight=("stim_weight", "sum"),
        stim_avg_weight=("stim_weight", "mean"),
        stim_max_weight=("stim_weight", "max"),
    )
    summary["total_weight_delta"] = (
        summary["stim_total_weight"] - summary["generated_total_weight"]
    )

    print(summary.to_string(index=False))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.output, index=False)
        print(f"\nSaved full report to: {args.output}")


if __name__ == "__main__":
    main()
