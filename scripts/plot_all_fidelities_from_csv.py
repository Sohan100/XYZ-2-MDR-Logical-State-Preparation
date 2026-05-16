"""

plot_all_fidelities_from_csv.py
----------------------------------------------------------------------------
Regenerate all-operator MDR fidelity plots from saved sweep CSV files.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _ensure_src_on_path() -> None:
    """
    Add the repository `src/` directory to `sys.path` if needed.
    """
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_ensure_src_on_path()

from mdr.constants import (  # noqa: E402
    CODE_FAMILY_DISPLAY_NAMES,
    DEFAULT_DISTANCES,
    DEFAULT_PLOTS_DIR,
    DEFAULT_RESULTS_DIR,
    DEFAULT_ROUNDS,
    NOISE_MODEL_DISPLAY_NAMES,
    NOISE_MODEL_PARAM_NAMES,
)
from mdr.mdr_noise_sweep import MdrNoiseSweep  # noqa: E402
from mdr.preparation import (  # noqa: E402
    PREP_MODE_FULL_MDR,
    PREP_MODES,
)
from mdr.workflows import resolve_family_search_dirs  # noqa: E402


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for all-operator fidelity plotting.
    """
    parser = argparse.ArgumentParser(
        description="Reload MDR sweep CSVs and plot all operator fidelities."
    )
    parser.add_argument(
        "--code-family",
        choices=sorted(CODE_FAMILY_DISPLAY_NAMES),
        default="xyz2",
    )
    parser.add_argument(
        "--prep-mode",
        choices=PREP_MODES,
        default=PREP_MODE_FULL_MDR,
    )
    parser.add_argument(
        "--noise-models",
        nargs="+",
        choices=sorted(NOISE_MODEL_PARAM_NAMES),
        default=sorted(NOISE_MODEL_PARAM_NAMES),
    )
    parser.add_argument(
        "--distances",
        type=int,
        nargs="+",
        default=DEFAULT_DISTANCES,
    )
    parser.add_argument(
        "--rounds",
        type=int,
        nargs="+",
        default=DEFAULT_ROUNDS,
    )
    parser.add_argument("--p-spam", type=float, default=None)
    parser.add_argument(
        "--recovery-mode",
        choices=["each_round", "final_round"],
        default=None,
    )
    parser.add_argument(
        "--correction-mode",
        choices=["physical", "pauli_frame"],
        default=None,
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    return parser.parse_args()


def _close(a: float, b: float, tol: float = 1e-15) -> bool:
    """
    Return True when two floating-point values are approximately equal.
    """
    return abs(float(a) - float(b)) <= tol


def _resolve_result_csv(
    *,
    input_dir: Path,
    code_family: str,
    noise_model: str,
    distance: int,
    prep_mode: str,
    p_spam: float | None,
    recovery_mode: str | None,
    correction_mode: str | None,
) -> Path | None:
    """
    Resolve the newest spec-matched sweep CSV for one configuration.
    """
    matches: List[Path] = []
    for search_dir in resolve_family_search_dirs(input_dir, code_family):
        patterns = [
            f"results_{code_family}_{noise_model}_d{distance}_*.spec.json",
        ]
        if code_family == "xyz2":
            patterns.append(f"results_{noise_model}_d{distance}_*.spec.json")

        for pattern in patterns:
            for spec_path in sorted(search_dir.glob(pattern)):
                spec = json.loads(spec_path.read_text(encoding="utf-8"))
                if str(spec.get("code_family", "xyz2")) != code_family:
                    continue
                if str(spec.get("prep_mode", PREP_MODE_FULL_MDR)) != prep_mode:
                    continue
                if p_spam is not None and not _close(
                    float(spec.get("p_spam", -1.0)),
                    p_spam,
                ):
                    continue
                if recovery_mode is not None and (
                    str(spec.get("recovery_mode", "each_round"))
                    != recovery_mode
                ):
                    continue
                if correction_mode is not None and (
                    str(spec.get("correction_mode", "physical"))
                    != correction_mode
                ):
                    continue

                csv_path = spec_path.with_suffix("").with_suffix(".csv")
                if csv_path.exists():
                    matches.append(csv_path)

    if not matches:
        return None
    matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0]


def _operator_labels(sweep: MdrNoiseSweep, category: str) -> List[str]:
    """
    Return the saved operator labels for one category.
    """
    if category == "stabilizer":
        return list(sweep.measure_stabilizers)
    if category == "logical":
        return list(sweep.logical_operators)
    raise ValueError("category must be 'stabilizer' or 'logical'.")


def _plot_fidelity(
    *,
    sweep: MdrNoiseSweep,
    title: str,
    category: str,
    rounds: List[int],
    save_path: Path,
) -> None:
    """
    Plot |<O>| versus p for every saved operator in one category.
    """
    labels = _operator_labels(sweep, category)
    if not labels:
        return

    combos = sorted(
        sweep.param_combos,
        key=lambda combo: tuple(float(x) for x in combo),
    )
    p_vals = np.asarray([float(combo[0]) for combo in combos], dtype=float)
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = [
        "o",
        "s",
        "^",
        "v",
        "<",
        ">",
        "D",
        "P",
        "X",
        "*",
        "h",
        "H",
        "+",
        "x",
        ".",
    ]
    style_map = {
        label: (
            colours[idx % len(colours)],
            markers[(idx // len(colours)) % len(markers)],
        )
        for idx, label in enumerate(labels)
    }

    cols = min(3, len(rounds))
    rows = math.ceil(len(rounds) / cols)
    fig, grid = plt.subplots(
        rows,
        cols,
        figsize=(6 * cols, 4.5 * rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    axes = list(grid.flatten())

    for idx, round_idx in enumerate(rounds):
        ax = axes[idx]
        ax.set_title(f"Round {round_idx}")
        ax.grid(True, which="both", alpha=0.3)
        for label in labels:
            means = np.asarray(
                [sweep.results[combo][round_idx][label] for combo in combos],
                dtype=float,
            )
            stds = np.asarray(
                [
                    sweep.results_std[combo][round_idx][label]
                    for combo in combos
                ],
                dtype=float,
            )
            color, marker = style_map[label]
            ax.errorbar(
                p_vals,
                means,
                yerr=stds,
                fmt=f"-{marker}",
                color=color,
                capsize=3,
                linewidth=1,
                markersize=3,
                label=label,
            )
        ax.set_xscale("log")
        ax.set_xlabel("p")
        ax.set_ylim(-0.02, 1.02)
        ylabel = "|<S>|" if category == "stabilizer" else "|<L>|"
        ax.set_ylabel(ylabel)

    for extra_ax in axes[len(rounds) :]:
        extra_ax.set_visible(False)

    handles, labels_for_legend = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_for_legend,
        loc="center left",
        bbox_to_anchor=(0.92, 0.5),
        fontsize="x-small",
    )
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 0.9, 0.96])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(fig)


def _suffix(args: argparse.Namespace) -> str:
    """
    Build a filename suffix describing the selected simulation slice.
    """
    parts = [args.prep_mode]
    if args.p_spam is not None:
        parts.append(f"pspam_{args.p_spam:.3e}".replace("+", ""))
    if args.recovery_mode is not None:
        parts.append(args.recovery_mode)
    if args.correction_mode is not None:
        parts.append(args.correction_mode)
    return "_".join(parts)


def main() -> None:
    """
    Generate all requested all-operator fidelity plots.
    """
    args = parse_args()
    output_dir = (
        args.output_dir
        / args.code_family
        / "fidelities"
        / args.prep_mode
    )
    suffix = _suffix(args)

    for noise_model in args.noise_models:
        for distance in args.distances:
            csv_path = _resolve_result_csv(
                input_dir=args.input_dir,
                code_family=args.code_family,
                noise_model=noise_model,
                distance=distance,
                prep_mode=args.prep_mode,
                p_spam=args.p_spam,
                recovery_mode=args.recovery_mode,
                correction_mode=args.correction_mode,
            )
            if csv_path is None:
                print(
                    "Warning: missing "
                    f"{args.code_family} {noise_model} d={distance} "
                    f"prep_mode={args.prep_mode}"
                )
                continue

            sweep = MdrNoiseSweep(load_data_filename=csv_path)
            for category in ("stabilizer", "logical"):
                title = (
                    f"{CODE_FAMILY_DISPLAY_NAMES[args.code_family]} "
                    f"{NOISE_MODEL_DISPLAY_NAMES[noise_model]} "
                    f"d={distance} {category} fidelities"
                )
                save_path = output_dir / (
                    f"fidelity_{args.code_family}_{noise_model}_"
                    f"d{distance}_{category}_{suffix}.pdf"
                )
                _plot_fidelity(
                    sweep=sweep,
                    title=title,
                    category=category,
                    rounds=args.rounds,
                    save_path=save_path,
                )
                print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
