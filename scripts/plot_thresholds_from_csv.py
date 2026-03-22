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
    NOISE_MODEL_DISPLAY_NAMES,
    NOISE_MODEL_PARAM_NAMES,
)
from mdr.mdr_noise_sweep import MdrNoiseSweep  # noqa: E402
from mdr.plotters import MdrNoiseSweepPlotter  # noqa: E402
from mdr.workflows import (  # noqa: E402
    code_family_subdir,
    resolve_family_search_dirs,
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for threshold-plot regeneration.

    Returns:
        argparse.Namespace: Parsed argument values controlling distances, input
        CSV directory, and output plot directory.
    """
    parser = argparse.ArgumentParser(
        description="Reload sweep CSVs and regenerate threshold PDFs.")
    parser.add_argument("--distances", type=int, nargs="+",
                        default=DEFAULT_DISTANCES)
    parser.add_argument(
        "--code-family",
        choices=sorted(CODE_FAMILY_DISPLAY_NAMES),
        default="xyz2",
    )
    parser.add_argument("--input-dir", type=Path,
                        default=DEFAULT_RESULTS_DIR)
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
        "--metric",
        choices=["observable_loss", "state_prep_error"],
        default="observable_loss",
        help=(
            "Plot the original Logical-X error rate `1 - |<X_L>|`. The "
            "`state_prep_error` option is retained as a compatibility alias."
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
    return parser.parse_args()


def _close(a: float, b: float, tol: float = 1e-15) -> bool:
    """Return True when two floating-point values are approximately equal."""
    return abs(float(a) - float(b)) <= tol


def _resolve_result_csv(
    input_dir: Path,
    noise_model: str,
    distance: int,
    code_family: str = "xyz2",
    p_spam: float | None = None,
) -> Path | None:
    """Resolve a saved result CSV using legacy and spec-based naming."""
    matches: list[Path] = []
    for search_dir in resolve_family_search_dirs(input_dir, code_family):
        legacy_candidates = [
            search_dir / f"results_{code_family}_{noise_model}_d{distance}.csv",
        ]
        if code_family == "xyz2":
            legacy_candidates.append(
                search_dir / f"results_{noise_model}_d{distance}.csv"
            )
        for legacy in legacy_candidates:
            if legacy.exists() and (p_spam is None or _close(p_spam, 0.0)):
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
) -> dict[str, MdrNoiseSweep]:
    """
    Load sweep CSV files for one noise model across requested distances.

    Args:
        input_dir: Directory containing saved sweep CSV files.
        distances: Code distances to include.
        noise_model: Noise model key (`pure_z`, `z_type`, or `unbiased`).

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
        )
        if csv_path is not None:
            sweeps[f"{display} (d={d})"] = MdrNoiseSweep(
                load_data_filename=csv_path)
        else:
            msg = f"Warning: missing {noise_model} d={d}"
            if p_spam is not None:
                msg += f", p_spam={p_spam:g}"
            print(msg)
    return sweeps


def main() -> None:
    """
    Generate threshold PDF plots from saved simulation CSV files.

    For each noise model, the script loads available distance sweeps and
    emits logical-X error-rate plots.

    Returns:
        None
    """
    args = parse_args()
    if args.output_dir.name == "thresholds":
        output_dir = args.output_dir
    elif args.output_dir.name == args.code_family:
        output_dir = args.output_dir / "thresholds"
    else:
        output_dir = code_family_subdir(
            args.output_dir,
            args.code_family,
        ) / "thresholds"
    output_dir.mkdir(parents=True, exist_ok=True)

    for noise_model in sorted(NOISE_MODEL_PARAM_NAMES):
        sweeps = _load_sweeps(
            args.input_dir,
            args.distances,
            noise_model,
            code_family=args.code_family,
            p_spam=args.p_spam,
        )
        if not sweeps:
            print(f"Skipping {noise_model}: no CSV files found.")
            continue

        suffix_core = (
            f"pspam_{args.p_spam:.3e}".replace("+", "")
            if args.p_spam is not None
            else "noise"
        )
        suffix = f"{args.metric}_{suffix_core}"
        filename = f"threshold_{args.code_family}_{noise_model}_{suffix}.pdf"
        out_pdf = output_dir / filename
        if args.metric == "observable_loss":
            MdrNoiseSweepPlotter.plot_error_multi(
                sweeps=sweeps,
                category="logical",
                rounds=[1],
                subset=["Logical X"],
                overlay=False,
                log_x=True,
                save_path=out_pdf,
            )
        else:
            MdrNoiseSweepPlotter.plot_state_prep_error_multi(
                sweeps=sweeps,
                rounds=[1],
                logical_label="Logical X",
                overlay=False,
                log_x=True,
                save_path=out_pdf,
                allow_legacy_approx=args.allow_legacy_approx,
            )
        print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
