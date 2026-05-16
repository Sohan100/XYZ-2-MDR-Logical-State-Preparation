"""

plot_thresholds_from_csv.py
----------------------------------------------------------------------------
Command-line helpers for plotting thresholds from csv outputs.
"""

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
from mdr.plotters import MdrNoiseSweepPlotter  # noqa: E402
from mdr.preparation import (  # noqa: E402
    PREP_MODE_FULL_MDR,
    PREP_MODES,
)
from mdr.workflows import (  # noqa: E402
    code_family_subdir,
    resolve_family_search_dirs,
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for threshold-plot regeneration.

    Returns:
    argparse.Namespace: Parsed argument values controlling distances, input CSV
    directory, and output plot directory.
    """
    parser = argparse.ArgumentParser(
        description="Reload sweep CSVs and regenerate threshold PDFs."
    )
    parser.add_argument(
        "--distances", type=int, nargs="+", default=DEFAULT_DISTANCES
    )
    parser.add_argument("--rounds", type=int, nargs="+", default=DEFAULT_ROUNDS)
    parser.add_argument(
        "--code-family",
        choices=sorted(CODE_FAMILY_DISPLAY_NAMES),
        default="xyz2",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    parser.add_argument(
        "--p-spam",
        type=float,
        default=None,
        help=(
            "Optional SPAM value filter. If set, load only spec-matched files "
            "for this p_spam."
        ),
    )
    parser.add_argument(
        "--prep-mode",
        choices=PREP_MODES,
        default=PREP_MODE_FULL_MDR,
        help="MDR preparation variant to load from spec sidecars.",
    )
    parser.add_argument(
        "--recovery-mode",
        choices=["each_round", "final_round"],
        default=None,
        help="Optional recovery timing filter for spec-matched files.",
    )
    parser.add_argument(
        "--correction-mode",
        choices=["physical", "pauli_frame"],
        default=None,
        help="Optional correction implementation filter.",
    )
    parser.add_argument(
        "--metric",
        choices=["observable_loss", "state_prep_error"],
        default="observable_loss",
        help=(
            "Plot the Logical-X error rate `1 - |<X_L>|`. The "
            "`state_prep_error` option is retained as a compatibility alias, "
            "and `observable_loss` uses the same Logical-X convention here."
        ),
    )
    parser.add_argument(
        "--allow-legacy-approx",
        action="store_true",
        help=(
            "Retained for backwards compatibility. The restored `1 - |<X_L>|` "
            "metric no longer requires signed logical expectations."
        ),
    )
    parser.add_argument(
        "--combine-noise-models",
        action="store_true",
        help=(
            "Also create one side-by-side Logical-X threshold figure with one "
            "panel per noise model."
        ),
    )
    parser.add_argument(
        "--x-min",
        type=float,
        default=None,
        help="Optional lower x-limit for zoomed threshold plots.",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=None,
        help="Optional upper x-limit for zoomed threshold plots.",
    )
    return parser.parse_args()


def _close(a: float, b: float, tol: float = 1e-15) -> bool:
    """
    Return True when two floating-point values are approximately equal.
    """
    return abs(float(a) - float(b)) <= tol


def _resolve_result_csv(
    input_dir: Path,
    noise_model: str,
    distance: int,
    code_family: str = "xyz2",
    p_spam: float | None = None,
    prep_mode: str = PREP_MODE_FULL_MDR,
    recovery_mode: str | None = None,
    correction_mode: str | None = None,
) -> Path | None:
    """
    Resolve a saved result CSV using legacy and spec-based naming.
    """
    matches: list[Path] = []
    for search_dir in resolve_family_search_dirs(input_dir, code_family):
        legacy_candidates = [
            search_dir
            / f"results_{code_family}_{noise_model}_d{distance}.csv",
        ]
        if code_family == "xyz2":
            legacy_candidates.append(
                search_dir / f"results_{noise_model}_d{distance}.csv"
            )
        for legacy in legacy_candidates:
            if (
                legacy.exists()
                and prep_mode == PREP_MODE_FULL_MDR
                and (p_spam is None or _close(p_spam, 0.0))
            ):
                return legacy

        pattern_candidates = [
            f"results_{code_family}_{noise_model}_d{distance}_*.spec.json",
        ]
        if code_family == "xyz2":
            pattern_candidates.append(
                f"results_{noise_model}_d{distance}_*.spec.json"
            )
        for pattern in pattern_candidates:
            for spec_path in sorted(search_dir.glob(pattern)):
                spec = json.loads(spec_path.read_text(encoding="utf-8"))
                if str(spec.get("code_family", "xyz2")) != code_family:
                    continue
                if (
                    str(spec.get("prep_mode", PREP_MODE_FULL_MDR))
                    != prep_mode
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
                if p_spam is not None:
                    val = float(spec.get("p_spam", -1.0))
                    if not _close(val, p_spam):
                        continue
                csv_path = spec_path.with_suffix("").with_suffix(".csv")
                if csv_path.exists():
                    matches.append(csv_path)

    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def _load_sweeps(
    input_dir: Path,
    distances: list[int],
    noise_model: str,
    code_family: str = "xyz2",
    p_spam: float | None = None,
    prep_mode: str = PREP_MODE_FULL_MDR,
    recovery_mode: str | None = None,
    correction_mode: str | None = None,
    include_noise_model_in_label: bool = True,
) -> dict[str, MdrNoiseSweep]:
    """
    Load sweep CSV files for one noise model across requested distances.

    Args:
    input_dir: Directory containing saved sweep CSV files. distances: Code
    distances to include. noise_model: Noise model key (`pure_z`, `z_type`, or
    `unbiased`).

    Returns:
    dict[str, MdrNoiseSweep]: Mapping from display label to loaded sweep
    object.
    """
    sweeps: dict[str, MdrNoiseSweep] = {}
    display = NOISE_MODEL_DISPLAY_NAMES[noise_model]
    for d in distances:
        csv_path = _resolve_result_csv(
            input_dir,
            noise_model,
            d,
            code_family=code_family,
            p_spam=p_spam,
            prep_mode=prep_mode,
            recovery_mode=recovery_mode,
            correction_mode=correction_mode,
        )
        if csv_path is not None:
            label = (
                f"{display} (d={d})"
                if include_noise_model_in_label
                else f"d={d}"
            )
            sweeps[label] = MdrNoiseSweep(load_data_filename=csv_path)
        else:
            msg = f"Warning: missing {noise_model} d={d}"
            if p_spam is not None:
                msg += f", p_spam={p_spam:g}"
            if prep_mode != PREP_MODE_FULL_MDR:
                msg += f", prep_mode={prep_mode}"
            if recovery_mode is not None:
                msg += f", recovery_mode={recovery_mode}"
            if correction_mode is not None:
                msg += f", correction_mode={correction_mode}"
            print(msg)
    return sweeps


def _format_limit_tag(value: float) -> str:
    """
    Format a numeric axis limit into a compact filename-safe tag.
    """
    return f"{value:.0e}".replace("+", "")


def main() -> None:
    """
    Generate threshold PDF plots from saved simulation CSV files.

    For each noise model, the script loads available distance sweeps and emits
    logical-X error-rate plots.

    Returns:
    None
    """
    args = parse_args()
    if (args.x_min is None) != (args.x_max is None):
        raise ValueError("Use both --x-min and --x-max together.")
    x_limits = (
        (float(args.x_min), float(args.x_max))
        if args.x_min is not None
        else None
    )
    if x_limits is not None and x_limits[0] >= x_limits[1]:
        raise ValueError("--x-min must be smaller than --x-max.")

    if args.output_dir.name == "thresholds":
        output_dir = args.output_dir
    elif args.output_dir.name == args.code_family:
        output_dir = args.output_dir / "thresholds"
    else:
        output_dir = (
            code_family_subdir(
                args.output_dir,
                args.code_family,
            )
            / "thresholds"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix_core = (
        f"pspam_{args.p_spam:.3e}".replace("+", "")
        if args.p_spam is not None
        else "noise"
    )
    suffix_parts = [args.metric]
    if args.prep_mode != PREP_MODE_FULL_MDR:
        suffix_parts.append(args.prep_mode)
    if args.recovery_mode is not None:
        suffix_parts.append(args.recovery_mode)
    if args.correction_mode is not None:
        suffix_parts.append(args.correction_mode)
    suffix_parts.append(suffix_core)
    suffix = "_".join(suffix_parts)
    combined_panels: dict[str, dict[str, MdrNoiseSweep]] = {}

    for noise_model in NOISE_MODEL_PARAM_NAMES:
        sweeps = _load_sweeps(
            args.input_dir,
            args.distances,
            noise_model,
            code_family=args.code_family,
            p_spam=args.p_spam,
            prep_mode=args.prep_mode,
            recovery_mode=args.recovery_mode,
            correction_mode=args.correction_mode,
        )
        if not sweeps:
            print(f"Skipping {noise_model}: no CSV files found.")
            continue

        filename = f"threshold_{args.code_family}_{noise_model}_{suffix}.pdf"
        out_pdf = output_dir / filename
        if args.metric == "observable_loss":
            MdrNoiseSweepPlotter.plot_error_multi(
                sweeps=sweeps,
                category="logical",
                rounds=args.rounds,
                subset=["Logical X"],
                overlay=False,
                log_x=True,
                save_path=out_pdf,
            )
        else:
            MdrNoiseSweepPlotter.plot_state_prep_error_multi(
                sweeps=sweeps,
                rounds=args.rounds,
                logical_label="Logical X",
                overlay=False,
                log_x=True,
                save_path=out_pdf,
                allow_legacy_approx=args.allow_legacy_approx,
            )
        print(f"Saved {out_pdf}")
        if args.combine_noise_models:
            combined_panels[NOISE_MODEL_DISPLAY_NAMES[noise_model]] = {
                label.rsplit("(", maxsplit=1)[-1].rstrip(")"): sweep
                for label, sweep in sweeps.items()
            }

    if args.combine_noise_models and combined_panels:
        for round_idx in args.rounds:
            combined_name = (
                f"threshold_{args.code_family}_noise_model_comparison_"
                f"r{round_idx}_{suffix}"
            )
            if x_limits is not None:
                combined_name += (
                    f"_x_{_format_limit_tag(x_limits[0])}_to_"
                    f"{_format_limit_tag(x_limits[1])}"
                )
            combined_pdf = output_dir / f"{combined_name}.pdf"
            MdrNoiseSweepPlotter.plot_logical_x_error_panels(
                panels=combined_panels,
                round_idx=round_idx,
                log_x=True,
                save_path=combined_pdf,
                x_limits=x_limits,
                allow_legacy_approx=args.allow_legacy_approx,
            )
            print(f"Saved {combined_pdf}")


if __name__ == "__main__":
    main()
