from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

from xyz2_mdr.mdr_circuit import MDRCircuit
from xyz2_mdr.mdr_noise_sweep import MdrNoiseSweep
from xyz2_mdr.mdr_simulation import MDRSimulation
from xyz2_mdr.plotters import MDRSimulationPlotter, MdrNoiseSweepPlotter


matplotlib.use("Agg")


def test_simulation_classes_do_not_expose_plotting_methods() -> None:
    assert not hasattr(MDRSimulation, "plot_multi_fidelity")
    assert not hasattr(MdrNoiseSweep, "plot_error_multi")
    assert not hasattr(MdrNoiseSweep, "plot_state_prep_error_multi")


def test_mdr_simulation_plotter_writes_pdf(
    tmp_path: Path,
    d3_code_inputs: dict[str, object],
) -> None:
    sim = MDRSimulation(
        mdr=MDRCircuit(
            # type: ignore[arg-type]
            stabilizers=d3_code_inputs["code_stabilizers"],
            toggles=d3_code_inputs["combined_toggles"],  # type: ignore[arg-type]
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
