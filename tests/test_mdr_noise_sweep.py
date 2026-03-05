from __future__ import annotations

from pathlib import Path

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

    loaded = MdrNoiseSweep(load_data_filename=out_csv)
    assert len(loaded.param_combos) == 2
    assert "Logical X" in loaded.logical_operators
