"""

test_workflows.py
----------------------------------------------------------------------------
Pytest coverage for workflows behavior and regression checks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mdr.preparation import (
    PREP_MODE_FULL_MDR,
    PREP_MODE_INDEPENDENT_HIGH_WEIGHT,
    PREP_MODE_LINK_LOGICAL_PLUS,
    pauli_support,
)
from mdr.workflows import (
    build_code_inputs,
    build_simulation_spec,
    canonical_table_path,
    default_ancilla_count,
    ensure_table_csv,
    run_noise_sweep_with_cache,
    simulation_results_path,
    simulation_spec_hash,
    simulation_spec_path,
)


def _max_disjoint_high_weight_count(stabilizers: list[str]) -> int:
    """
    Brute-force the maximum number of max-weight disjoint stabilizers.
    """
    supports = [pauli_support(stabilizer) for stabilizer in stabilizers]
    max_weight = max(len(support) for support in supports)
    best = 0
    for mask in range(1 << len(stabilizers)):
        used: set[int] = set()
        count = 0
        valid = True
        for idx, support in enumerate(supports):
            if not (mask & (1 << idx)):
                continue
            if not used.isdisjoint(support):
                valid = False
                break
            used.update(support)
            if len(support) == max_weight:
                count += 1
        if valid:
            best = max(best, count)
    return best


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


def test_link_logical_plus_prunes_prepared_links(tmp_path: Path) -> None:
    """
    The all-|+> variant omits XX links and Logical X from MDR checks.
    """
    table_csv = tmp_path / "mdr_table_xyz2_d3.csv"
    code_inputs = build_code_inputs(
        distance=3,
        table_csv=table_csv,
        code_family="xyz2",
        prep_mode=PREP_MODE_LINK_LOGICAL_PLUS,
    )

    assert len(code_inputs["stabilizers"]) == 17
    assert len(code_inputs["prepared_stabilizers"]) == 9
    assert len(code_inputs["active_stabilizers"]) == 8
    assert len(code_inputs["combined_toggles"]) == 8
    assert len(code_inputs["code_stabilizers"]) == 8
    assert code_inputs["include_logical_x_check"] is False
    assert code_inputs["logical_x"] not in code_inputs["code_stabilizers"]
    assert default_ancilla_count(
        distance=3,
        table_csv=table_csv,
        code_family="xyz2",
        prep_mode=PREP_MODE_LINK_LOGICAL_PLUS,
    ) == 8
    assert str(code_inputs["psi_circuit"]).strip() == (
        "H 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17"
    )


def test_independent_high_weight_mode_selects_disjoint_checks(
    tmp_path: Path,
) -> None:
    """
    The optimized product-state variant prepares disjoint high-weight checks.
    """
    table_csv = tmp_path / "mdr_table_xyz2_d3.csv"
    code_inputs = build_code_inputs(
        distance=3,
        table_csv=table_csv,
        code_family="xyz2",
        prep_mode=PREP_MODE_INDEPENDENT_HIGH_WEIGHT,
    )
    prepared = code_inputs["prepared_stabilizers"]
    supports = [pauli_support(stabilizer) for stabilizer in prepared]
    selected_high_weight = sum(
        len(support) == 6 for support in supports
    )
    max_possible_high_weight = _max_disjoint_high_weight_count(
        code_inputs["stabilizers"]
    )

    for idx, left in enumerate(supports):
        for right in supports[idx + 1 :]:
            assert left.isdisjoint(right)
    assert selected_high_weight == max_possible_high_weight
    assert selected_high_weight > 0
    assert code_inputs["include_logical_x_check"] is True
    assert len(code_inputs["code_stabilizers"]) == (
        len(code_inputs["active_stabilizers"]) + 1
    )
    assert sorted(
        [
            *code_inputs["prepared_stabilizer_indices"],
            *code_inputs["active_stabilizer_indices"],
        ]
    ) == list(range(len(code_inputs["stabilizers"])))


def test_simulation_spec_hash_changes_with_prep_mode() -> None:
    """
    Preparation mode participates in the cache key and filename.
    """
    base_kwargs = dict(
        distance=3,
        noise_model="pure_z",
        probabilities=[1e-5],
        rounds=[1],
        shots=100,
        num_replicates=3,
        p_spam=0.0,
    )
    full_spec = build_simulation_spec(
        **base_kwargs,
        prep_mode=PREP_MODE_FULL_MDR,
        ancillas=17,
    )
    link_spec = build_simulation_spec(
        **base_kwargs,
        prep_mode=PREP_MODE_LINK_LOGICAL_PLUS,
        ancillas=8,
    )

    assert simulation_spec_hash(full_spec) != simulation_spec_hash(link_spec)
    assert "_full_mdr_anc17_" in simulation_results_path(
        Path("results"),
        full_spec,
    ).name
    assert "_link_logical_plus_anc8_" in simulation_results_path(
        Path("results"),
        link_spec,
    ).name


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
    table_csv = tmp_path / "mdr_table_xyz2_d3.csv"
    resolved_ancillas = default_ancilla_count(
        distance=3,
        table_csv=table_csv,
        code_family="xyz2",
    )
    spec = build_simulation_spec(
        distance=3,
        noise_model="pure_z",
        probabilities=[1e-5],
        rounds=[1],
        shots=100,
        num_replicates=3,
        p_spam=1.339e-3,
        ancillas=resolved_ancillas,
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
        table_csv=table_csv,
        results_dir=tmp_path,
        force_rerun=False,
    )

    assert loaded is True
    assert out_path == csv_path
    assert out_path.parent.name == "xyz2"
    assert "Logical X" in sweep.logical_operators


def test_surface_workflow_uses_distinct_spec_and_filename(
    tmp_path: Path,
) -> None:
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
    assert simulation_spec_hash(xyz2_spec) != simulation_spec_hash(
        surface_spec
    )
    assert xyz2_path.parent.name == "xyz2"
    assert xyz2_path.name.startswith("results_xyz2_pure_z_d3_")
    assert surface_path.parent.name == "surface"
    assert surface_path.name.startswith("results_surface_pure_z_d3_")


def test_simulation_spec_hash_changes_with_ancilla_count() -> None:
    """
    Ancilla count should participate in the MDR cache specification.
    """
    spec_one = build_simulation_spec(
        distance=3,
        noise_model="pure_z",
        probabilities=[1e-5],
        rounds=[1],
        shots=100,
        num_replicates=3,
        p_spam=0.0,
        ancillas=1,
    )
    spec_many = build_simulation_spec(
        distance=3,
        noise_model="pure_z",
        probabilities=[1e-5],
        rounds=[1],
        shots=100,
        num_replicates=3,
        p_spam=0.0,
        ancillas=12,
    )

    assert simulation_spec_hash(spec_one) != simulation_spec_hash(spec_many)


def test_run_noise_sweep_with_cache_defaults_to_parallel_ancillas(
    tmp_path: Path,
) -> None:
    """
    High-level MDR workflows should default to one ancilla per stabilizer and
    persist that choice in the cached simulation spec.
    """
    table_csv = tmp_path / "mdr_table_surface_d3.csv"
    expected_ancillas = default_ancilla_count(
        distance=3,
        table_csv=table_csv,
        code_family="surface",
    )

    _, out_path, _ = run_noise_sweep_with_cache(
        distance=3,
        noise_model="pure_z",
        probabilities=[1e-5],
        rounds=[1],
        shots=16,
        num_replicates=1,
        p_spam=0.0,
        table_csv=table_csv,
        results_dir=tmp_path,
        force_rerun=True,
        code_family="surface",
    )

    spec = json.loads(
        simulation_spec_path(out_path).read_text(encoding="utf-8")
    )
    assert spec["ancillas"] == expected_ancillas
    assert expected_ancillas == 12


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
    assert run_table_csv.read_text(
        encoding="utf-8"
    ) == canonical_csv.read_text(encoding="utf-8")
