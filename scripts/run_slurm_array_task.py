"""
run_slurm_array_task.py
----------------------------------------------------------------------------
Run one Slurm array point for an MDR sweep.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Iterable, Sequence


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
    DEFAULT_DISTANCES,
    DEFAULT_NUM_REPLICATES,
    DEFAULT_P_SPAM,
    DEFAULT_ROUNDS,
    DEFAULT_SHOTS,
    NOISE_MODEL_PARAM_NAMES,
    SUPPORTED_CODE_FAMILIES,
    default_probabilities,
)
from mdr.preparation import (  # noqa: E402
    PREP_MODE_FULL_MDR,
    PREP_MODES,
)
from mdr.workflows import code_family_subdir  # noqa: E402


def _default_task_index() -> int | None:
    """
    Return the Slurm array task id when this script is running in an array.
    """
    raw = os.environ.get("SLURM_ARRAY_TASK_ID")
    return None if raw is None else int(raw)


def _default_run_tag() -> str:
    """
    Return a stable tag shared by every task in one Slurm array submission.
    """
    return (
        os.environ.get("RUN_TAG")
        or os.environ.get("SLURM_ARRAY_JOB_ID")
        or os.environ.get("SLURM_JOB_ID")
        or time.strftime("%Y%m%d-%H%M%SZ", time.gmtime())
    )


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for one Slurm array task.
    """
    parser = argparse.ArgumentParser(
        description="Run one MDR Slurm-array sweep point."
    )
    parser.add_argument("--task-index", type=int, default=_default_task_index())
    parser.add_argument("--run-tag", type=str, default=_default_run_tag())
    parser.add_argument(
        "--code-family",
        choices=SUPPORTED_CODE_FAMILIES,
        required=True,
    )
    parser.add_argument("--code-name", type=str, required=True)
    parser.add_argument("--run-suffix", type=str, required=True)
    parser.add_argument(
        "--root-dir", type=Path, default=Path("XYZ2-experiment-data-slurm")
    )
    parser.add_argument(
        "--scripts-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--results-copy-dir",
        type=Path,
        default=Path("data/simulation_results"),
    )
    parser.add_argument(
        "--tables-copy-dir",
        type=Path,
        default=Path("data/tables"),
    )
    parser.add_argument(
        "--plots-output-dir",
        type=Path,
        default=Path("data/plots"),
    )
    parser.add_argument(
        "--distances", type=int, nargs="+", default=DEFAULT_DISTANCES
    )
    parser.add_argument(
        "--noise-models",
        nargs="+",
        choices=sorted(NOISE_MODEL_PARAM_NAMES),
        default=list(NOISE_MODEL_PARAM_NAMES),
    )
    parser.add_argument(
        "--probabilities",
        type=float,
        nargs="+",
        default=default_probabilities(),
    )
    parser.add_argument("--rounds", type=int, nargs="+", default=DEFAULT_ROUNDS)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument(
        "--num-replicates", type=int, default=DEFAULT_NUM_REPLICATES
    )
    parser.add_argument("--p-spam", type=float, default=DEFAULT_P_SPAM)
    parser.add_argument(
        "--recovery-mode",
        choices=["each_round", "final_round"],
        default="each_round",
    )
    parser.add_argument(
        "--correction-mode",
        choices=["physical", "pauli_frame"],
        default="physical",
    )
    parser.add_argument(
        "--prep-mode",
        choices=PREP_MODES,
        default=PREP_MODE_FULL_MDR,
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plot-logical-x", action="store_true")
    parser.add_argument("--plot-all-fidelities", action="store_true")
    return parser.parse_args()


def task_count(
    *,
    distances: Sequence[int],
    noise_models: Sequence[str],
    probabilities: Sequence[float],
) -> int:
    """
    Return the number of array tasks required by a sweep grid.
    """
    return len(distances) * len(noise_models) * len(probabilities)


def task_coordinates(
    task_index: int,
    *,
    distances: Sequence[int],
    noise_models: Sequence[str],
    probabilities: Sequence[float],
) -> tuple[str, int, int, float]:
    """
    Map one flat Slurm array index to a noise model, distance, and p index.
    """
    num_distances = len(distances)
    num_probabilities = len(probabilities)
    total = task_count(
        distances=distances,
        noise_models=noise_models,
        probabilities=probabilities,
    )
    if task_index < 0 or task_index >= total:
        raise IndexError(
            f"task_index {task_index} is outside the sweep range [0, "
            f"{total - 1}]"
        )

    tasks_per_noise = num_distances * num_probabilities
    noise_idx = task_index // tasks_per_noise
    remainder = task_index % tasks_per_noise
    distance_idx = remainder // num_probabilities
    probability_idx = remainder % num_probabilities
    return (
        noise_models[noise_idx],
        distances[distance_idx],
        probability_idx,
        probabilities[probability_idx],
    )


def run_name_for(
    *,
    run_tag: str,
    distance: int,
    noise_model: str,
    run_suffix: str,
) -> str:
    """
    Build the deterministic run name shared by all p tasks for one sweep.
    """
    suffix = run_suffix.strip("-")
    return f"Run-{run_tag}-d{distance}-{noise_model}-{suffix}"


def acquire_lock(lock_dir: Path, *, timeout_s: float = 1800.0) -> None:
    """
    Acquire a simple directory lock.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            lock_dir.mkdir(parents=True)
            return
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for lock: {lock_dir}")
            time.sleep(2.0)


def release_lock(lock_dir: Path) -> None:
    """
    Release a simple directory lock.
    """
    try:
        lock_dir.rmdir()
    except FileNotFoundError:
        return


def _run(command: Sequence[str]) -> None:
    """
    Run a subprocess command and fail loudly if it fails.
    """
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def setup_run_if_needed(
    *,
    args: argparse.Namespace,
    run_name: str,
    run_dir: Path,
) -> None:
    """
    Create the run config once for a shared distance/noise sweep.
    """
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        return

    locks_dir = run_dir.parent / ".locks"
    setup_lock = locks_dir / f"{run_name}.setup.lock"
    acquire_lock(setup_lock)
    try:
        if config_path.exists():
            return
        command = [
            sys.executable,
            str(args.scripts_dir / "setup_slurm_run.py"),
            "--distance",
            str(args.distance),
            "--code-family",
            args.code_family,
            "--code-name",
            args.code_name,
            "--noise-model",
            args.noise_model,
            "--run-name",
            run_name,
            "--root-dir",
            str(args.root_dir),
            "--shots",
            str(args.shots),
            "--num-replicates",
            str(args.num_replicates),
            "--p-spam",
            str(args.p_spam),
            "--recovery-mode",
            args.recovery_mode,
            "--correction-mode",
            args.correction_mode,
            "--prep-mode",
            args.prep_mode,
            "--rounds",
            *[str(round_value) for round_value in args.rounds],
            "--probabilities",
            *[str(probability) for probability in args.probabilities],
            "--overwrite",
        ]
        _run(command)
    finally:
        release_lock(setup_lock)


def expected_partial_files(run_dir: Path, probabilities: Sequence[float]) -> list[Path]:
    """
    Return the expected partial result CSVs for one run.
    """
    return [
        run_dir / "partials" / f"result_idx{idx:03d}.csv"
        for idx in range(len(probabilities))
    ]


def all_exist(paths: Iterable[Path]) -> bool:
    """
    Return True when every path exists.
    """
    return all(path.exists() for path in paths)


def run_probability_task(
    *,
    args: argparse.Namespace,
    run_name: str,
    probability_idx: int,
) -> None:
    """
    Run one probability-index task.
    """
    command = [
        sys.executable,
        str(args.scripts_dir / "run_slurm_experiment.py"),
        run_name,
        str(probability_idx),
        "--root-dir",
        str(args.root_dir),
        "--code-family",
        args.code_family,
    ]
    if args.force:
        command.append("--force")
    _run(command)


def merge_if_complete(
    *,
    args: argparse.Namespace,
    run_name: str,
    run_dir: Path,
) -> None:
    """
    Merge a run once all probability partials are present.
    """
    if not all_exist(expected_partial_files(run_dir, args.probabilities)):
        print(f"Run {run_name} is not complete yet; merge skipped.", flush=True)
        return

    merged_marker = run_dir / ".merged"
    merged_csv = run_dir / (
        f"results_{args.code_family}_{args.noise_model}_d{args.distance}.csv"
    )
    if merged_marker.exists() and merged_csv.exists():
        print(f"Run {run_name} is already merged.", flush=True)
        return

    locks_dir = run_dir.parent / ".locks"
    merge_lock = locks_dir / f"{run_name}.merge.lock"
    acquire_lock(merge_lock)
    try:
        if merged_marker.exists() and merged_csv.exists():
            print(f"Run {run_name} is already merged.", flush=True)
            return
        if not all_exist(expected_partial_files(run_dir, args.probabilities)):
            print(
                f"Run {run_name} became incomplete before merge; skipped.",
                flush=True,
            )
            return
        command = [
            sys.executable,
            str(args.scripts_dir / "merge_slurm_results.py"),
            run_name,
            "--root-dir",
            str(args.root_dir),
            "--code-family",
            args.code_family,
            "--copy-to",
            str(args.results_copy_dir),
            "--tables-copy-to",
            str(args.tables_copy_dir),
        ]
        _run(command)
        merged_marker.write_text(
            time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()),
            encoding="utf-8",
        )
    finally:
        release_lock(merge_lock)


def all_run_names(args: argparse.Namespace) -> list[str]:
    """
    Return the run names for every distance/noise pair in this array sweep.
    """
    names: list[str] = []
    for noise_model in args.noise_models:
        for distance in args.distances:
            names.append(
                run_name_for(
                    run_tag=args.run_tag,
                    distance=distance,
                    noise_model=noise_model,
                    run_suffix=args.run_suffix,
                )
            )
    return names


def plots_if_complete(args: argparse.Namespace) -> None:
    """
    Optionally create final plots once every distance/noise run is merged.
    """
    if not args.plot_logical_x and not args.plot_all_fidelities:
        return

    family_root = code_family_subdir(args.root_dir, args.code_family)
    run_dirs = [family_root / run_name for run_name in all_run_names(args)]
    if not all((run_dir / ".merged").exists() for run_dir in run_dirs):
        print("Not all runs are merged yet; plot step skipped.", flush=True)
        return

    locks_dir = family_root / ".locks"
    plots_lock = locks_dir / f"Run-{args.run_tag}-{args.run_suffix}.plots.lock"
    plots_marker = family_root / f".plots-{args.run_tag}-{args.run_suffix}"
    if plots_marker.exists():
        print("Plots already completed for this array sweep.", flush=True)
        return

    acquire_lock(plots_lock)
    try:
        if plots_marker.exists():
            print("Plots already completed for this array sweep.", flush=True)
            return
        if args.plot_logical_x:
            _run(
                [
                    sys.executable,
                    str(args.scripts_dir / "plot_thresholds_from_csv.py"),
                    "--code-family",
                    args.code_family,
                    "--prep-mode",
                    args.prep_mode,
                    "--p-spam",
                    str(args.p_spam),
                    "--metric",
                    "observable_loss",
                    "--combine-noise-models",
                    "--recovery-mode",
                    args.recovery_mode,
                    "--correction-mode",
                    args.correction_mode,
                    "--distances",
                    *[str(distance) for distance in args.distances],
                    "--rounds",
                    *[str(round_value) for round_value in args.rounds],
                    "--input-dir",
                    str(args.results_copy_dir),
                    "--output-dir",
                    str(args.plots_output_dir),
                ]
            )
        if args.plot_all_fidelities:
            _run(
                [
                    sys.executable,
                    str(args.scripts_dir / "plot_all_fidelities_from_csv.py"),
                    "--code-family",
                    args.code_family,
                    "--prep-mode",
                    args.prep_mode,
                    "--p-spam",
                    str(args.p_spam),
                    "--recovery-mode",
                    args.recovery_mode,
                    "--correction-mode",
                    args.correction_mode,
                    "--distances",
                    *[str(distance) for distance in args.distances],
                    "--noise-models",
                    *args.noise_models,
                    "--rounds",
                    *[str(round_value) for round_value in args.rounds],
                    "--input-dir",
                    str(args.results_copy_dir),
                    "--output-dir",
                    str(args.plots_output_dir),
                ]
            )
        plots_marker.write_text(
            time.strftime("%Y-%m-%dT%H:%M:%SZ\n", time.gmtime()),
            encoding="utf-8",
        )
    finally:
        release_lock(plots_lock)


def main() -> None:
    """
    Run one Slurm array point and opportunistically merge/plot.
    """
    args = parse_args()
    if args.task_index is None:
        raise ValueError(
            "Missing --task-index and SLURM_ARRAY_TASK_ID is not set."
        )

    total = task_count(
        distances=args.distances,
        noise_models=args.noise_models,
        probabilities=args.probabilities,
    )
    if args.task_index >= total:
        print(
            f"Task index {args.task_index} is outside this sweep's {total} "
            "tasks; exiting.",
            flush=True,
        )
        return

    noise_model, distance, probability_idx, probability = task_coordinates(
        args.task_index,
        distances=args.distances,
        noise_models=args.noise_models,
        probabilities=args.probabilities,
    )
    args.noise_model = noise_model
    args.distance = distance

    run_name = run_name_for(
        run_tag=args.run_tag,
        distance=distance,
        noise_model=noise_model,
        run_suffix=args.run_suffix,
    )
    family_root = code_family_subdir(args.root_dir, args.code_family)
    run_dir = family_root / run_name

    print(f"Task index: {args.task_index} / {total - 1}", flush=True)
    print(f"Run tag: {args.run_tag}", flush=True)
    print(f"Run name: {run_name}", flush=True)
    print(f"Code family: {args.code_family}", flush=True)
    print(f"Noise model: {noise_model}", flush=True)
    print(f"Distance: {distance}", flush=True)
    print(f"Probability index: {probability_idx}", flush=True)
    print(f"Probability: {probability}", flush=True)

    setup_run_if_needed(args=args, run_name=run_name, run_dir=run_dir)
    (run_dir / "shots.txt").write_text(f"{args.shots}\n", encoding="utf-8")
    run_probability_task(
        args=args,
        run_name=run_name,
        probability_idx=probability_idx,
    )
    merge_if_complete(args=args, run_name=run_name, run_dir=run_dir)
    plots_if_complete(args)


if __name__ == "__main__":
    main()
