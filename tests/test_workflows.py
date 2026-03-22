from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mdr.workflows import (
    build_code_inputs,
    build_simulation_spec,
    canonical_table_path,
    ensure_table_csv,
    run_noise_sweep_with_cache,
    simulation_results_path,
    simulation_spec_hash,
)


@pytest.mark.parametrize(
    (
        "p_spam_a",
        "p_spam_b",
        "shots_a",
        "shots_b",
        "mode_a",
        "mode_b",
        "correction_a",
        "correction_b",
    ),
    [
        (
            0.0,
            1e-3,
            100,
            100,
            "each_round",
            "each_round",
            "physical",
            "physical",
        ),
        (
            0.0,
            0.0,
            100,
            200,
            "each_round",
            "each_round",
            "physical",
            "physical",
        ),
        (
            0.0,
            0.0,
            100,
            100,
            "each_round",
            "final_round",
            "physical",
            "physical",
        ),
        (
            0.0,
            0.0,
            100,
            100,
            "each_round",
            "each_round",
            "physical",
            "pauli_frame",
        ),
    ],
    ids=[
        "different_pspam",
        "different_shots",
        "different_recovery_mode",
        "different_correction_mode",
    ],
)
def test_simulation_spec_hash_changes_with_parameters(
    p_spam_a: float,
    p_spam_b: float,
    shots_a: int,
    shots_b: int,
    mode_a: str,
    mode_b: str,
    correction_a: str,
    correction_b: str,
) -> None:
    """
    Ensure specification hash changes when one simulation parameter changes.

    This guards the cache-key contract used for result reuse.

    Returns:
        None
    """
    spec_a = build_simulation_spec(
        distance=3,
        noise_model="pure_z",
        probabilities=[1e-5, 1e-4],
        rounds=[1, 5],
        shots=shots_a,
        num_replicates=3,
        p_spam=p_spam_a,
        recovery_mode=mode_a,
        correction_mode=correction_a,
    )
    spec_b = build_simulation_spec(
        distance=3,
        noise_model="pure_z",
        probabilities=[1e-5, 1e-4],
        rounds=[1, 5],
        shots=shots_b,
        num_replicates=3,
        p_spam=p_spam_b,
        recovery_mode=mode_b,
        correction_mode=correction_b,
    )
    assert simulation_spec_hash(spec_a) != simulation_spec_hash(spec_b)


def test_run_noise_sweep_with_cache_loads_existing(
    tmp_path: Path,
) -> None:
    """
    Ensure cache-hit flow loads an existing CSV instead of recomputing.

    Args:
        tmp_path: Per-test temporary directory provided by pytest.

    Returns:
        None
    """
    spec = build_simulation_spec(
        distance=3,
        noise_model="pure_z",
        probabilities=[1e-5],
        rounds=[1],
        shots=100,
        num_replicates=3,
        p_spam=1.339e-3,
    )
    csv_path = simulation_results_path(tmp_path, spec)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "g1_z": 1e-5,
                "IZ": 1e-5,
                "ZI": 1e-5,
                "ZZ": 1e-5,
                "round": 1,
                "operator": "Logical X",
                "mean": 0.9,
                "std": 0.01,
            }
        ]
    ).to_csv(csv_path, index=False)

    sweep, out_path, loaded = run_noise_sweep_with_cache(
        distance=3,
        noise_model="pure_z",
        probabilities=[1e-5],
        rounds=[1],
        shots=100,
        num_replicates=3,
        p_spam=1.339e-3,
        table_csv=tmp_path / "mdr_table_xyz2_d3.csv",
        results_dir=tmp_path,
        force_rerun=False,
    )

    assert loaded is True
    assert out_path == csv_path
    assert out_path.parent.name == "xyz2"
    assert "Logical X" in sweep.logical_operators


def test_surface_workflow_uses_distinct_spec_and_filename(tmp_path: Path) -> None:
    """
    Surface-code caches should not collide with the legacy XYZ2 naming path.
    """
    xyz2_spec = build_simulation_spec(
        distance=3,
        noise_model="pure_z",
        probabilities=[1e-5],
        rounds=[1],
        shots=100,
        num_replicates=3,
        p_spam=0.0,
    )
    surface_spec = build_simulation_spec(
        distance=3,
        noise_model="pure_z",
        probabilities=[1e-5],
        rounds=[1],
        shots=100,
        num_replicates=3,
        p_spam=0.0,
        code_family="surface",
    )

    xyz2_path = simulation_results_path(tmp_path, xyz2_spec)
    surface_path = simulation_results_path(tmp_path, surface_spec)
    assert simulation_spec_hash(xyz2_spec) != simulation_spec_hash(surface_spec)
    assert xyz2_path.parent.name == "xyz2"
    assert xyz2_path.name.startswith("results_xyz2_pure_z_d3_")
    assert surface_path.parent.name == "surface"
    assert surface_path.name.startswith("results_surface_pure_z_d3_")


def test_build_code_inputs_surface_smoke(tmp_path: Path) -> None:
    """
    Surface-code inputs should expose the full MDR structures.
    """
    table_csv = tmp_path / "mdr_table_surface_d3.csv"
    code_inputs = build_code_inputs(
        distance=3,
        table_csv=table_csv,
        code_family="surface",
    )

    assert table_csv.exists()
    assert code_inputs["logical_x"] == "X0 X5 X10"
    assert len(code_inputs["stabilizers"]) == 12
    assert len(code_inputs["combined_toggles"]) == 13


def test_ensure_table_csv_reuses_canonical_table(tmp_path: Path) -> None:
    """
    Slurm-prepared run folders should reuse persisted canonical tables.

    Args:
        tmp_path: Per-test temporary directory provided by pytest.

    Returns:
        None
    """
    tables_dir = tmp_path / "data" / "tables"
    canonical_csv = canonical_table_path(
        distance=3,
        code_family="surface",
        tables_dir=tables_dir,
    )
    build_code_inputs(
        distance=3,
        table_csv=canonical_csv,
        code_family="surface",
    )

    run_table_csv = (
        tmp_path / "slurm" / "surface" / "Run-test" / canonical_csv.name
    )
    ensure_table_csv(
        distance=3,
        target_table_csv=run_table_csv,
        code_family="surface",
        canonical_tables_dir=tables_dir,
    )

    assert canonical_csv.exists()
    assert run_table_csv.exists()
    assert run_table_csv.read_text(encoding="utf-8") == canonical_csv.read_text(
        encoding="utf-8"
    )
