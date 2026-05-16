"""

test_plotters.py
----------------------------------------------------------------------------
Pytest coverage for plotters behavior and regression checks.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

from mdr.mdr_circuit import MDRCircuit
from mdr.mdr_noise_sweep import MdrNoiseSweep
from mdr.mdr_simulation import MDRSimulation
from mdr.plotters import MDRSimulationPlotter, MdrNoiseSweepPlotter

matplotlib.use("Agg")


def test_simulation_classes_do_not_expose_plotting_methods() -> None:
    """
    Test that simulation classes do not expose plotting methods.
    """
    assert not hasattr(MDRSimulation, "plot_multi_fidelity")
    assert not hasattr(MdrNoiseSweep, "plot_error_multi")
    assert not hasattr(MdrNoiseSweep, "plot_state_prep_error_multi")


def test_mdr_simulation_plotter_writes_pdf(
    tmp_path: Path,
    d3_code_inputs: dict[str, object],
) -> None:
    """
    Test that MDR simulation plotter writes pdf.

    Args:
    tmp_path: Tmp path. d3_code_inputs: D3 code inputs.
    """
    sim = MDRSimulation(
        mdr=MDRCircuit(
            # type: ignore[arg-type]
            stabilizers=d3_code_inputs["code_stabilizers"],
            # type: ignore[arg-type]
            toggles=d3_code_inputs["combined_toggles"],
            ancillas=1,
            psi_circuit=d3_code_inputs["psi_circuit"],
            recovery_mode="final_round",
        ),
        # type: ignore[arg-type]
        stabilizer_pauli_strings=d3_code_inputs["stabilizers"],
        # type: ignore[arg-type]
        logical_pauli_strings=d3_code_inputs["logical_operators"],
        shots_per_measurement=20,
        total_mdr_rounds=2,
        num_replicates=2,
    )
    out_pdf = tmp_path / "sim_plot.pdf"
    MDRSimulationPlotter.plot_multi_fidelity(
        {"test": sim},
        category="logical",
        show_violin=False,
        show_replicates=False,
        save_path=out_pdf,
    )
    assert out_pdf.exists()


def test_noise_sweep_plotter_writes_pdf(tmp_path: Path) -> None:
    """
    Test that noise sweep plotter writes pdf.

    Args:
    tmp_path: Tmp path.
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
    out_pdf = tmp_path / "sweep_plot.pdf"
    MdrNoiseSweepPlotter.plot_state_prep_error_multi(
        sweeps={"test": loaded},
        rounds=[1],
        save_path=out_pdf,
        allow_legacy_approx=True,
    )
    assert out_pdf.exists()


def test_noise_sweep_plotter_writes_combined_panel_pdf(
    tmp_path: Path,
) -> None:
    """
    Test that the combined Logical-X threshold panel plot writes a pdf.

    Args:
    tmp_path: Tmp path.
    """
    out_csv = tmp_path / "combined_results.csv"
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
                "std": 0.02,
            },
            {
                "g1_z": 1e-4,
                "IZ": 1e-4,
                "ZI": 1e-4,
                "ZZ": 1e-4,
                "round": 1,
                "operator": "Logical X",
                "mean": 0.7,
                "std": 0.03,
            },
        ]
    ).to_csv(out_csv, index=False)

    loaded = MdrNoiseSweep(load_data_filename=out_csv)
    out_pdf = tmp_path / "combined_panel_plot.pdf"
    MdrNoiseSweepPlotter.plot_logical_x_error_panels(
        panels={
            "Pure Z Noise": {"d=3": loaded, "d=5": loaded},
            "Z Type Noise": {"d=3": loaded, "d=5": loaded},
            "Unbiased Depolarizing Noise": {
                "d=3": loaded,
                "d=5": loaded,
            },
        },
        round_idx=1,
        log_x=True,
        save_path=out_pdf,
        x_limits=(1e-5, 1e-4),
        allow_legacy_approx=True,
    )
    assert out_pdf.exists()
