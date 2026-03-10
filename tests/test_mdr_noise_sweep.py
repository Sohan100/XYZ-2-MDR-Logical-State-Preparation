from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from xyz2_mdr.mdr_noise_sweep import MdrNoiseSweep


def test_noise_sweep_save_and_load(
    tmp_path: Path,
    d3_code_inputs: dict[str, object],
) -> None:
    """
    Verify sweep outputs can be saved and loaded for plotting workflows.

    Args:
        tmp_path: Per-test temporary directory provided by pytest.
        d3_code_inputs: Prebuilt distance-3 code inputs fixture.

    Returns:
        None
    """
    out_csv = tmp_path / "results_pure_z_d3.csv"

    sweep = MdrNoiseSweep(
        # type: ignore[arg-type]
        code_stabilizers=d3_code_inputs["code_stabilizers"],
        toggles=d3_code_inputs["combined_toggles"],  # type: ignore[arg-type]
        # type: ignore[arg-type]
        measure_stabilizers=d3_code_inputs["stabilizers"],
        # type: ignore[arg-type]
        logical_operators=d3_code_inputs["logical_operators"],
        ancillas=1,
        psi_circuit=d3_code_inputs["psi_circuit"],
        p_spam=1.339e-3,
        param_names=["g1_z", "IZ", "ZI", "ZZ"],
        param_values=[1e-5, 1e-4],
        round_list=[1],
        shots=100,
        num_replicates=3,
        save_data_filename=out_csv,
    )
    assert sweep.param_combos
    assert out_csv.exists()
    saved = pd.read_csv(out_csv)
    assert "mean_signed" in saved.columns
    assert "std_signed" in saved.columns

    loaded = MdrNoiseSweep(load_data_filename=out_csv)
    assert len(loaded.param_combos) == 2
    assert "Logical X" in loaded.logical_operators
    assert loaded.has_exact_signed_results is True


def test_state_prep_error_legacy_approx_load(
    tmp_path: Path,
) -> None:
    """
    Legacy CSVs without signed columns can be loaded for approximate plotting.

    Returns:
        None
    """
    out_csv = tmp_path / "legacy_results.csv"
    pd.DataFrame(
        [
            {
                "g1_z": 1e-5,
                "IZ": 1e-5,
                "ZI": 1e-5,
                "ZZ": 1e-5,
                "round": 1,
                "operator": "Logical X",
                "mean": 0.8,
                "std": 0.1,
            }
        ]
    ).to_csv(out_csv, index=False)

    loaded = MdrNoiseSweep(load_data_filename=out_csv)
    _, y_vals, y_errs = loaded._metric_series_for_operator(
        round_idx=1,
        operator="Logical X",
        metric="state_prep_error",
        allow_legacy_approx=True,
    )

    assert loaded.has_exact_signed_results is False
    assert y_vals.tolist() == pytest.approx([0.1])
    assert y_errs.tolist() == pytest.approx([0.05])
