"""
Create round-scan fidelity panels from MDR sweep CSV files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    DEFAULT_PLOTS_DIR,
    DEFAULT_RESULTS_DIR,
    NOISE_MODEL_DISPLAY_NAMES,
    NOISE_MODEL_PARAM_NAMES,
)
from mdr.decoders.factory import SUPPORTED_DECODER_MODES  # noqa: E402
from mdr.preparation import PREP_MODE_LINK_LOGICAL_PLUS, PREP_MODES  # noqa: E402
from mdr.workflows import resolve_family_search_dirs  # noqa: E402

RESERVED_COLUMNS = {
    "round",
    "operator",
    "mean",
    "std",
    "mean_signed",
    "std_signed",
    "decoder_mode",
    "decoder_unknown_fraction",
    "decoder_mean_gap",
    "decoder_truncation_mass",
}


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Plot stabilizer and Logical-X fidelity versus MDR round for "
            "saved MDR sweep CSVs."
        )
    )
    parser.add_argument("--code-family", default="xyz2")
    parser.add_argument("--distances", type=int, nargs="+", default=[3, 5])
    parser.add_argument(
        "--noise-models",
        nargs="+",
        choices=sorted(NOISE_MODEL_PARAM_NAMES),
        default=["unbiased", "z_type", "pure_z"],
    )
    parser.add_argument("--rounds", type=int, nargs="+", default=list(range(2, 11)))
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    parser.add_argument(
        "--prep-mode",
        choices=PREP_MODES,
        default=PREP_MODE_LINK_LOGICAL_PLUS,
    )
    parser.add_argument("--p-spam", type=float, default=None)
    parser.add_argument(
        "--recovery-mode",
        choices=["each_round", "final_round"],
        default="final_round",
    )
    parser.add_argument(
        "--correction-mode",
        choices=["physical", "pauli_frame"],
        default="pauli_frame",
    )
    parser.add_argument(
        "--decoder-mode",
        choices=SUPPORTED_DECODER_MODES,
        default="mps_mld",
    )
    parser.add_argument("--decoder-max-bond-dimension", type=int, default=None)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["png", "pdf"],
        default=["png", "pdf"],
    )
    return parser.parse_args()


def _close(a: float, b: float, tol: float = 1e-15) -> bool:
    """
    Return True when two floating-point values are approximately equal.
    """
    return abs(float(a) - float(b)) <= tol


def _spec_matches(
    spec: dict,
    *,
    code_family: str,
    distance: int,
    noise_model: str,
    prep_mode: str,
    p_spam: float | None,
    recovery_mode: str | None,
    correction_mode: str | None,
    decoder_mode: str | None,
    decoder_max_bond_dimension: int | None,
) -> bool:
    """
    Check whether a result sidecar matches the requested plot slice.
    """
    if str(spec.get("code_family", "xyz2")) != code_family:
        return False
    if int(spec.get("distance", -1)) != int(distance):
        return False
    if str(spec.get("noise_model", "")) != noise_model:
        return False
    if str(spec.get("prep_mode", "")) != prep_mode:
        return False
    if recovery_mode is not None and str(spec.get("recovery_mode", "")) != recovery_mode:
        return False
    if correction_mode is not None and str(spec.get("correction_mode", "")) != correction_mode:
        return False
    if decoder_mode is not None and str(spec.get("decoder_mode", "toggle_frame")) != decoder_mode:
        return False
    if decoder_max_bond_dimension is not None:
        config = dict(spec.get("decoder_config", {}))
        if int(config.get("max_bond_dimension", -1)) != decoder_max_bond_dimension:
            return False
    if p_spam is not None and not _close(float(spec.get("p_spam", -1.0)), p_spam):
        return False
    return True


def resolve_result_csv(
    *,
    input_dir: Path,
    code_family: str,
    distance: int,
    noise_model: str,
    prep_mode: str,
    p_spam: float | None,
    recovery_mode: str | None,
    correction_mode: str | None,
    decoder_mode: str | None,
    decoder_max_bond_dimension: int | None,
) -> Path | None:
    """
    Resolve the newest spec-matched CSV for a distance and noise model.
    """
    matches: list[Path] = []
    for search_dir in resolve_family_search_dirs(input_dir, code_family):
        pattern = f"results_{code_family}_{noise_model}_d{distance}_*.spec.json"
        for spec_path in search_dir.glob(pattern):
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            if not _spec_matches(
                spec,
                code_family=code_family,
                distance=distance,
                noise_model=noise_model,
                prep_mode=prep_mode,
                p_spam=p_spam,
                recovery_mode=recovery_mode,
                correction_mode=correction_mode,
                decoder_mode=decoder_mode,
                decoder_max_bond_dimension=decoder_max_bond_dimension,
            ):
                continue
            csv_path = spec_path.with_suffix("").with_suffix(".csv")
            if csv_path.exists():
                matches.append(csv_path)
    if not matches:
        return None
    matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0]


def _parameter_columns(df: pd.DataFrame) -> list[str]:
    """
    Return swept physical-noise parameter columns.
    """
    return [col for col in df.columns if col not in RESERVED_COLUMNS]


def _p_label(p_value: float) -> str:
    """
    Format a physical-noise point for plot legends.
    """
    if p_value == 0:
        return "p=0"
    return f"p={p_value:.1e}"


def _category_series(
    df: pd.DataFrame,
    *,
    rounds: Iterable[int],
    category: str,
    p_value: float,
    parameter_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract one category curve versus round for one physical-noise point.
    """
    subset = df[np.isclose(df[parameter_column].astype(float), p_value)]
    if category == "stabilizer":
        subset = subset[~subset["operator"].astype(str).str.startswith("Logical")]
    elif category == "logical_x":
        subset = subset[subset["operator"].astype(str) == "Logical X"]
    else:
        raise ValueError("category must be 'stabilizer' or 'logical_x'.")

    xs: list[int] = []
    ys: list[float] = []
    yerrs: list[float] = []
    for round_idx in rounds:
        round_rows = subset[subset["round"].astype(int) == int(round_idx)]
        if round_rows.empty:
            continue
        means = round_rows["mean"].astype(float).to_numpy()
        stds = round_rows["std"].astype(float).to_numpy()
        xs.append(int(round_idx))
        ys.append(float(np.mean(means)))
        yerrs.append(float(np.sqrt(np.sum(stds**2)) / max(len(stds), 1)))
    return (
        np.asarray(xs, dtype=int),
        np.asarray(ys, dtype=float),
        np.asarray(yerrs, dtype=float),
    )


def plot_distance(
    *,
    distance: int,
    frames: Dict[str, pd.DataFrame],
    rounds: list[int],
    output_dir: Path,
    code_family: str,
    prep_mode: str,
    recovery_mode: str,
    correction_mode: str,
    decoder_mode: str,
    decoder_max_bond_dimension: int | None,
    formats: list[str],
) -> list[Path]:
    """
    Plot one distance as a two-row, three-noise-model panel.
    """
    if not frames:
        raise ValueError("No data frames supplied.")

    noise_order = [model for model in ["unbiased", "z_type", "pure_z"] if model in frames]
    noise_order.extend(model for model in frames if model not in noise_order)
    fig, axes = plt.subplots(
        2,
        len(noise_order),
        figsize=(5.5 * len(noise_order), 8.0),
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    all_p_values = sorted(
        {
            float(value)
            for df in frames.values()
            for value in df[_parameter_columns(df)[0]].unique()
        }
    )
    style_map = {
        p_value: colors[index % len(colors)]
        for index, p_value in enumerate(all_p_values)
    }

    for col, noise_model in enumerate(noise_order):
        df = frames[noise_model]
        p_col = _parameter_columns(df)[0]
        p_values = sorted(float(value) for value in df[p_col].unique())
        axes[0, col].set_title(NOISE_MODEL_DISPLAY_NAMES[noise_model])
        for p_value in p_values:
            color = style_map[p_value]
            label = _p_label(p_value)
            for row, category in enumerate(["stabilizer", "logical_x"]):
                xs, ys, yerrs = _category_series(
                    df,
                    rounds=rounds,
                    category=category,
                    p_value=p_value,
                    parameter_column=p_col,
                )
                axes[row, col].errorbar(
                    xs,
                    ys,
                    yerr=yerrs,
                    marker="o",
                    linewidth=1.6,
                    markersize=3.5,
                    capsize=2,
                    color=color,
                    label=label,
                )
                axes[row, col].grid(True, alpha=0.3)
                axes[row, col].set_xticks(rounds)
                axes[row, col].set_ylim(-0.02, 1.02)

    axes[0, 0].set_ylabel("Avg stabilizer fidelity")
    axes[1, 0].set_ylabel("Logical X fidelity")
    for ax in axes[1, :]:
        ax.set_xlabel("MDR round")

    chi_label = (
        f", chi={decoder_max_bond_dimension}"
        if decoder_max_bond_dimension is not None
        else ""
    )
    fig.suptitle(
        (
            f"{code_family} d={distance}: round-scan fidelities "
            f"({prep_mode}, {decoder_mode}{chi_label}, "
            f"{recovery_mode}/{correction_mode})"
        ),
        y=0.995,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(len(labels), 6),
        bbox_to_anchor=(0.5, 0.955),
        fontsize="small",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])

    saved: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"round_fidelity_{code_family}_d{distance}_{prep_mode}_"
        f"{decoder_mode}_{recovery_mode}_{correction_mode}"
    )
    if decoder_max_bond_dimension is not None:
        stem += f"_chi{decoder_max_bond_dimension}"
    for fmt in formats:
        out_path = output_dir / f"{stem}.{fmt}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        saved.append(out_path)
    plt.close(fig)
    return saved


def main() -> None:
    """
    Load saved sweep CSVs and render requested round-scan plots.
    """
    args = parse_args()
    output_dir = args.output_dir / args.code_family / "round_fidelity"
    all_saved: list[Path] = []
    for distance in args.distances:
        frames: dict[str, pd.DataFrame] = {}
        for noise_model in args.noise_models:
            csv_path = resolve_result_csv(
                input_dir=args.input_dir,
                code_family=args.code_family,
                distance=distance,
                noise_model=noise_model,
                prep_mode=args.prep_mode,
                p_spam=args.p_spam,
                recovery_mode=args.recovery_mode,
                correction_mode=args.correction_mode,
                decoder_mode=args.decoder_mode,
                decoder_max_bond_dimension=args.decoder_max_bond_dimension,
            )
            if csv_path is None:
                print(
                    "Warning: missing "
                    f"{args.code_family} d={distance} {noise_model}"
                )
                continue
            frames[noise_model] = pd.read_csv(csv_path)
            print(f"Loaded {csv_path}")
        if not frames:
            print(f"Skipping d={distance}: no matching CSVs.")
            continue
        all_saved.extend(
            plot_distance(
                distance=distance,
                frames=frames,
                rounds=args.rounds,
                output_dir=output_dir,
                code_family=args.code_family,
                prep_mode=args.prep_mode,
                recovery_mode=args.recovery_mode,
                correction_mode=args.correction_mode,
                decoder_mode=args.decoder_mode,
                decoder_max_bond_dimension=args.decoder_max_bond_dimension,
                formats=args.formats,
            )
        )

    for path in all_saved:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
