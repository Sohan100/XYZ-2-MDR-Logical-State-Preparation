"""
Tests for Slurm array task indexing helpers.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_slurm_array_task.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_slurm_array_task",
    SCRIPT_PATH,
)
assert SPEC is not None
assert SPEC.loader is not None
slurm_array_task = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(slurm_array_task)

run_name_for = slurm_array_task.run_name_for
task_coordinates = slurm_array_task.task_coordinates
task_count = slurm_array_task.task_count


def test_task_count_covers_full_grid() -> None:
    """
    The array size should be the product of noise, distance, and p grids.
    """
    assert (
        task_count(
            distances=[3, 5],
            noise_models=["z_type", "pure_z", "unbiased"],
            probabilities=[0.1, 0.2, 0.3],
        )
        == 18
    )


def test_task_coordinates_use_noise_distance_probability_order() -> None:
    """
    Flat array ids should advance p first, then distance, then noise.
    """
    distances = [3, 5]
    noise_models = ["z_type", "pure_z"]
    probabilities = [0.1, 0.2, 0.3]

    assert task_coordinates(
        0,
        distances=distances,
        noise_models=noise_models,
        probabilities=probabilities,
    ) == ("z_type", 3, 0, 0.1)
    assert task_coordinates(
        3,
        distances=distances,
        noise_models=noise_models,
        probabilities=probabilities,
    ) == ("z_type", 5, 0, 0.1)
    assert task_coordinates(
        6,
        distances=distances,
        noise_models=noise_models,
        probabilities=probabilities,
    ) == ("pure_z", 3, 0, 0.1)


def test_run_name_is_shared_across_probability_tasks() -> None:
    """
    Probability index should not appear in the shared run folder name.
    """
    assert (
        run_name_for(
            run_tag="123",
            distance=5,
            noise_model="pure_z",
            run_suffix="with-spam",
        )
        == "Run-123-d5-pure_z-with-spam"
    )
