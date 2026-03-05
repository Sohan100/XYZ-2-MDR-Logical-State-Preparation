from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from xyz2_mdr.workflows import (
    build_simulation_spec,
    run_noise_sweep_with_cache,
    simulation_results_path,
    simulation_spec_hash,
)


@pytest.mark.parametrize(
    ("p_spam_a", "p_spam_b", "shots_a", "shots_b"),
    [
        (0.0, 1e-3, 100, 100),
        (0.0, 0.0, 100, 200),
    ],
    ids=["different_pspam", "different_shots"],
)
def test_simulation_spec_hash_changes_with_parameters(
    p_spam_a: float,
    p_spam_b: float,
    shots_a: int,
    shots_b: int,
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
    )
    spec_b = build_simulation_spec(
        distance=3,
        noise_model="pure_z",
        probabilities=[1e-5, 1e-4],
        rounds=[1, 5],
        shots=shots_b,
        num_replicates=3,
        p_spam=p_spam_b,
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
        table_csv=tmp_path / "mdr_table_d3.csv",
        results_dir=tmp_path,
        force_rerun=False,
    )

    assert loaded is True
    assert out_path == csv_path
    assert "Logical X" in sweep.logical_operators
