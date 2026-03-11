from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

from xyz2_mdr.analysis_reporting import (
    NotebookFinalRoundAnalysis,
    NotebookThresholdAnalysis,
)


matplotlib.use("Agg")


def test_notebook_threshold_analysis_loads_and_plots(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    plots_dir = tmp_path / "plots"
    results_dir.mkdir()

    csv_path = results_dir / "results_pure_z_d3.csv"
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
    ).to_csv(csv_path, index=False)

    analysis = NotebookThresholdAnalysis(
        results_dir=results_dir,
        plots_dir=plots_dir,
        distances=[3],
        noise_models={"pure_z": "Pure Z"},
    )
    sweeps_by_model, records_df = analysis.load_sweeps_for_p_spam(0.0)

    assert not records_df.empty
    assert "Pure Z (d=3)" in sweeps_by_model["pure_z"]

    saved_paths = analysis.plot_threshold_suite(
        sweeps_by_model=sweeps_by_model,
        output_label="no_spam",
    )
    assert len(saved_paths) == 1
    assert saved_paths[0].exists()


def test_notebook_final_round_analysis_summarizes_and_plots(
    tmp_path: Path,
) -> None:
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
