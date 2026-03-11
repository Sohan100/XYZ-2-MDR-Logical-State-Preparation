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

from xyz2_mdr.constants import (  # noqa: E402
    DEFAULT_DISTANCES,
    DEFAULT_PLOTS_DIR,
    DEFAULT_RESULTS_DIR,
    NOISE_MODEL_DISPLAY_NAMES,
    NOISE_MODEL_PARAM_NAMES,
)
from xyz2_mdr.mdr_noise_sweep import MdrNoiseSweep  # noqa: E402
from xyz2_mdr.plotters import MdrNoiseSweepPlotter  # noqa: E402


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
        default="state_prep_error",
        help=(
            "Plot either the legacy observable-loss metric `1 - |<X_L>|` or "
            "the corrected logical state-preparation error `(1 - <X_L>) / 2`."
        ),
    )
    parser.add_argument(
        "--allow-legacy-approx",
        action="store_true",
        help=(
            "Allow old CSVs without signed logical expectations to approximate "
            "state-preparation error as `(1 - |<X_L>|) / 2`."
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
    p_spam: float | None = None,
) -> Path | None:
    """Resolve a saved result CSV using legacy and spec-based naming."""
    legacy = input_dir / f"results_{noise_model}_d{distance}.csv"
    if legacy.exists() and (p_spam is None or _close(p_spam, 0.0)):
        return legacy

    spec_files = sorted(
        input_dir.glob(f"results_{noise_model}_d{distance}_*.spec.json")
    )
    matches: list[Path] = []
    for spec_path in spec_files:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
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
        csv_path = _resolve_result_csv(input_dir, noise_model, d, p_spam=p_spam)
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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for noise_model in sorted(NOISE_MODEL_PARAM_NAMES):
        sweeps = _load_sweeps(
            args.input_dir, args.distances, noise_model, p_spam=args.p_spam
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
        out_pdf = args.output_dir / f"threshold_{noise_model}_{suffix}.pdf"
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
