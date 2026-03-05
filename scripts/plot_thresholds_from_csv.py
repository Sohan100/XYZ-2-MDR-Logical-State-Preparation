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
    DEFAULT_DISTANCES,
    DEFAULT_PLOTS_DIR,
    DEFAULT_RESULTS_DIR,
    NOISE_MODEL_DISPLAY_NAMES,
    NOISE_MODEL_PARAM_NAMES,
)
from xyz2_mdr.mdr_noise_sweep import MdrNoiseSweep  # noqa: E402


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
    return parser.parse_args()


def _load_sweeps(
    input_dir: Path,
    distances: list[int],
    noise_model: str,
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
        csv_path = input_dir / f"results_{noise_model}_d{d}.csv"
        if not csv_path.exists():
            candidates = sorted(
                input_dir.glob(f"results_{noise_model}_d{d}_spec-*.csv"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                csv_path = candidates[0]
        if csv_path.exists():
            sweeps[f"{display} (d={d})"] = MdrNoiseSweep(
                load_data_filename=csv_path)
        else:
            print(f"Warning: missing {csv_path}")
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
        sweeps = _load_sweeps(args.input_dir, args.distances, noise_model)
        if not sweeps:
            print(f"Skipping {noise_model}: no CSV files found.")
            continue

        out_pdf = args.output_dir / f"threshold_{noise_model}_noise.pdf"
        MdrNoiseSweep.plot_error_multi(
            sweeps=sweeps,
            category="logical",
            rounds=[1],
            subset=["Logical X"],
            overlay=False,
            log_x=True,
            save_path=out_pdf,
        )
        print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
