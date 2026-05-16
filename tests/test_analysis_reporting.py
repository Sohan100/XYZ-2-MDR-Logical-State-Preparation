"""

test_analysis_reporting.py
----------------------------------------------------------------------------
Pytest coverage for analysis reporting behavior and regression checks.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

from mdr.analysis_reporting import (
    NotebookFinalRoundAnalysis,
    NotebookThresholdAnalysis,
)

matplotlib.use("Agg")


def test_notebook_threshold_analysis_loads_and_plots(tmp_path: Path) -> None:
    """
    Test that notebook threshold analysis loads and plots.

    Args:
    tmp_path: Tmp path.
    """
    results_dir = tmp_path / "results"
    plots_dir = tmp_path / "plots"
    family_results_dir = results_dir / "xyz2"
    family_results_dir.mkdir(parents=True)

    csv_path = (
        family_results_dir
        / (
            "results_xyz2_pure_z_d3_pspam0.000e00_"
            "shots10_reps2_spec-testcase.csv"
        )
    )
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
                "mean_signed": 0.8,
                "std_signed": 0.1,
            }
        ]
    ).to_csv(csv_path, index=False)
    (
        family_results_dir
        / (
            "results_xyz2_pure_z_d3_pspam0.000e00_"
            "shots10_reps2_spec-testcase.spec.json"
        )
    ).write_text(
        (
            "{\n"
            '  "p_spam": 0.0,\n'
            '  "recovery_mode": "final_round",\n'
            '  "correction_mode": "pauli_frame"\n'
            "}"
        ),
        encoding="utf-8",
    )

    analysis = NotebookThresholdAnalysis(
        results_dir=results_dir,
        plots_dir=plots_dir,
        distances=[3],
        noise_models={"pure_z": "Pure Z"},
    )
    sweeps_by_model, records_df = analysis.load_sweeps_for_p_spam(
        0.0,
        recovery_mode="final_round",
        correction_mode="pauli_frame",
    )

    assert not records_df.empty
    assert "Pure Z (d=3)" in sweeps_by_model["pure_z"]
    assert str(csv_path) in records_df["csv_path"].tolist()
    assert set(records_df["correction_mode"]) == {"pauli_frame"}

    saved_paths = analysis.plot_threshold_suite(
        sweeps_by_model=sweeps_by_model,
        output_label="no_spam",
    )
    assert len(saved_paths) == 1
    assert saved_paths[0].exists()
    assert saved_paths[0].name == "threshold_xyz2_pure_z_no_spam.pdf"


def test_notebook_threshold_analysis_loads_mode_comparison(
    tmp_path: Path,
) -> None:
    """
    Comparison loading should return physical and Pauli-frame sweeps together.

    Args:
    tmp_path: Per-test temporary directory provided by pytest.

    Returns:
    None
    """
    results_dir = tmp_path / "results"
    plots_dir = tmp_path / "plots"
    family_results_dir = results_dir / "xyz2"
    family_results_dir.mkdir(parents=True)

    for tag, correction_mode in (
        ("physical", "physical"),
        ("pauli", "pauli_frame"),
    ):
        csv_path = (
            family_results_dir
            / (
                "results_xyz2_pure_z_d3_pspam0.000e00_"
                f"shots10_reps2_spec-{tag}.csv"
            )
        )
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
                    "mean_signed": 0.8,
                    "std_signed": 0.1,
                }
            ]
        ).to_csv(csv_path, index=False)
        csv_path.with_suffix(".spec.json").write_text(
            (
                "{\n"
                '  "p_spam": 0.0,\n'
                '  "recovery_mode": "each_round",\n'
                f'  "correction_mode": "{correction_mode}"\n'
                "}"
            ),
            encoding="utf-8",
        )

    analysis = NotebookThresholdAnalysis(
        results_dir=results_dir,
        plots_dir=plots_dir,
        distances=[3],
        noise_models={"pure_z": "Pure Z"},
    )
    sweeps_by_model, records_df = analysis.load_correction_mode_comparison(
        distance=3,
        p_spam=0.0,
        recovery_mode="each_round",
    )

    assert set(records_df["correction_mode"]) == {"physical", "pauli_frame"}
    assert set(sweeps_by_model["pure_z"]) == {
        "Physical (d=3)",
        "Pauli Frame (d=3)",
    }


def test_notebook_final_round_analysis_summarizes_and_plots(
    tmp_path: Path,
) -> None:
    """
    Test that notebook final round analysis summarizes and plots.

    Args:
    tmp_path: Tmp path.
    """
    dataset_csv = tmp_path / "final_round.csv"
    plots_dir = tmp_path / "plots"
    rows = []
    for display_name in ("Pure Z Noise", "Unbiased Depolarizing Noise"):
        for round_idx in (0, 1):
            for replicate_idx in range(2):
                rows.append(
                    {
                        "noise_model": "pure_z",
                        "display_name": display_name,
                        "category": "stabilizer",
                        "operator": "S0",
                        "round": round_idx,
                        "replicate_idx": replicate_idx,
                        "fidelity": 0.9,
                        "p_spam": 0.0,
                        "recovery_mode": "final_round",
                        "shots": 10,
                        "num_replicates": 2,
                        "distance": 3,
                    }
                )
                for logical_label, fidelity in (
                    ("Logical X", 0.8),
                    ("Logical Y", 0.7),
                    ("Logical Z", 0.6),
                ):
                    rows.append(
                        {
                            "noise_model": "pure_z",
                            "display_name": display_name,
                            "category": "logical",
                            "operator": logical_label,
                            "round": round_idx,
                            "replicate_idx": replicate_idx,
                            "fidelity": fidelity,
                            "p_spam": 0.0,
                            "recovery_mode": "final_round",
                            "shots": 10,
                            "num_replicates": 2,
                            "distance": 3,
                        }
                    )
    pd.DataFrame(rows).to_csv(dataset_csv, index=False)

    analysis = NotebookFinalRoundAnalysis(
        dataset_csv=dataset_csv,
        plots_dir=plots_dir,
    )
    dataset = analysis.load_dataset()
    summary = analysis.build_round_summary_table()

    assert not dataset.empty
    assert set(summary.columns) == {
        "display_name",
        "round",
        "avg_stabilizer_fidelity",
        "logical_x_fidelity",
        "logical_y_fidelity",
        "logical_z_fidelity",
    }

    stabilizer_path = analysis.plot_fidelity(
        category="stabilizer",
        save_path=plots_dir / "stabilizer.pdf",
        figsize=(8, 4),
    )
    logical_path = analysis.plot_fidelity(
        category="logical",
        save_path=plots_dir / "logical.pdf",
        figsize=(8, 4),
    )
    combined_path = analysis.plot_combined_fidelity(
        save_path=plots_dir / "combined.pdf",
        figsize=(8, 6),
    )

    assert stabilizer_path.exists()
    assert logical_path.exists()
    assert combined_path.exists()
