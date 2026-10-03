"""

test_mdr_noise_sweep.py
----------------------------------------------------------------------------
Pytest coverage for mdr noise sweep behavior and regression checks.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mdr.mdr_noise_sweep import MdrNoiseSweep


def test_noise_sweep_save_and_load(
    tmp_path: Path,
    d3_code_inputs: dict[str, object],
) -> None:
    """
    Verify sweep outputs can be saved and loaded for plotting workflows.

    Args:
    tmp_path: Per-test temporary directory provided by pytest. d3_code_inputs:
    Prebuilt distance-3 code inputs fixture.

    Returns:
    None
    """
    out_csv = tmp_path / "results_xyz2_pure_z_d3.csv"

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
    assert "decoder_mode" in saved.columns
    assert "decoder_unknown_fraction" in saved.columns

    loaded = MdrNoiseSweep(load_data_filename=out_csv)
    assert len(loaded.param_combos) == 2
    assert "Logical X" in loaded.logical_operators
    assert loaded.has_exact_signed_results is True


def test_state_prep_error_uses_original_logical_x_error_rate(
    tmp_path: Path,
) -> None:
    """

    Logical-X sweep error should use the restored `1 - |<X>|` metric.

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
    )

    assert loaded.has_exact_signed_results is False
    assert y_vals.tolist() == pytest.approx([0.2])
    assert y_errs.tolist() == pytest.approx([0.1])


def test_si1000_sweep_uses_single_p_parameter(tmp_path: Path) -> None:
    """
    SI1000 sweeps should save one physical p column.
    """
    out_csv = tmp_path / "results_si1000.csv"

    sweep = MdrNoiseSweep(
        code_stabilizers=["Z0"],
        toggles=["X0"],
        measure_stabilizers=["Z0"],
        logical_operators={"Logical X": "X0"},
        ancillas=1,
        num_qubits=1,
        param_names=["p"],
        param_values=[0.03],
        round_list=[1],
        shots=8,
        num_replicates=1,
        save_data_filename=out_csv,
    )
    saved = pd.read_csv(out_csv)

    assert sweep.param_names == ["p"]
    assert "p" in saved.columns
    assert saved["p"].tolist() == pytest.approx([0.03, 0.03])


def test_si1000_sweep_rejects_too_large_p() -> None:
    """
    p values above 0.2 would make MERR(5p) invalid.
    """
    with pytest.raises(ValueError, match="MERR"):
        MdrNoiseSweep(
            code_stabilizers=["Z0"],
            toggles=["X0"],
            measure_stabilizers=["Z0"],
            logical_operators={"Logical X": "X0"},
            ancillas=1,
            num_qubits=1,
            param_names=["p"],
            param_values=[0.21],
            round_list=[1],
            shots=1,
            num_replicates=1,
        )
