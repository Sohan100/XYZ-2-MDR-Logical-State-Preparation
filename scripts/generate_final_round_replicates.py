"""
generate_final_round_replicates.py
----------------------------------
Build a replicate-level final-round-recovery dataset for notebook plots.

This script runs one `MDRSimulation` per named noise model at a chosen
reference physical-noise point, stores every replicate fidelity for every
operator and MDR round, and writes the resulting long-form CSV into the
family-specific `data/analysis/` directory used by `analysis_plots.ipynb`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List

import pandas as pd


def _ensure_src_on_path() -> None:
    """
    Add the repository `src/` directory to `sys.path` if needed.

    This allows the script to be executed directly from the repository root
    without requiring a package installation step first.

    Returns:
        None
    """
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_ensure_src_on_path()

from mdr.constants import (  # noqa: E402
    DEFAULT_P_SPAM,
    NOISE_MODEL_DISPLAY_NAMES,
    SUPPORTED_CODE_FAMILIES,
)
from mdr.mdr_circuit import MDRCircuit  # noqa: E402
from mdr.mdr_noise_sweep import MdrNoiseSweep  # noqa: E402
from mdr.mdr_simulation import MDRSimulation  # noqa: E402
from mdr.workflows import (  # noqa: E402
    build_code_inputs,
    canonical_table_path,
    noise_param_names,
)

REFERENCE_NOISE_POINTS: Dict[str, float] = {
    "pure_z": 0.00173780082875,
    "z_type": 0.0036307805477,
    "unbiased": 0.00758577575029,
}


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for final-round dataset generation.

    Returns:
        argparse.Namespace: Parsed arguments defining the code family,
        distance, reference noise points, and CSV output path.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate a replicate-level final-round MDR dataset for notebook "
            "round-by-round fidelity plots."
        )
    )
    parser.add_argument(
        "--code-family",
        choices=SUPPORTED_CODE_FAMILIES,
        default="surface",
    )
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument(
        "--noise-models",
        nargs="+",
        default=["z_type", "pure_z", "unbiased"],
        choices=sorted(REFERENCE_NOISE_POINTS),
    )
    parser.add_argument(
        "--noise-point",
        action="append",
        default=[],
        metavar="MODEL=VALUE",
        help=(
            "Override the reference physical-noise point for one noise model. "
            "May be provided multiple times, for example "
            "--noise-point pure_z=0.0017."
        ),
    )
    parser.add_argument("--shots", type=int, default=2000)
    parser.add_argument("--num-replicates", type=int, default=30)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--p-spam", type=float, default=DEFAULT_P_SPAM)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Destination CSV path. Defaults to "
            "data/analysis/<code_family>/d<distance>_<code_family>_"
            "final_round_fidelity_replicates.csv."
        ),
    )
    return parser.parse_args()


def parse_noise_points(
    overrides: List[str],
    selected_models: List[str],
) -> Dict[str, float]:
    """
    Build the per-noise-model probability map used by the round scan.

    Args:
        overrides: CLI overrides supplied as `MODEL=VALUE` strings.
        selected_models: Noise models that will be simulated.

    Returns:
        Dict[str, float]: Reference probability for each requested noise
        model.

    Raises:
        ValueError: If an override is malformed or names an unknown model.
    """
    points = {
        model: float(REFERENCE_NOISE_POINTS[model]) for model in selected_models
    }
    for item in overrides:
        if "=" not in item:
            raise ValueError(
                f"Invalid --noise-point value '{item}'. Expected MODEL=VALUE."
            )
        model, value = item.split("=", 1)
        model = model.strip()
        if model not in REFERENCE_NOISE_POINTS:
            raise ValueError(
                f"Unknown noise model '{model}'. Valid values: "
                + ", ".join(sorted(REFERENCE_NOISE_POINTS))
            )
        points[model] = float(value)
    return points


def build_noise_kwargs(
    param_names: List[str],
    probability: float,
) -> Dict[str, Any]:
    """
    Construct MDR circuit noise arguments for one named noise model point.

    This mirrors the project sweep convention where all active single-qubit
    channels in the model share one budget and all active two-qubit channels
    share another budget. The requested `probability` therefore means the
    model-level point from the notebook sweep grid, not the already-split
    per-channel probability written into the circuit.

    Args:
        param_names: Ordered active parameter names for one noise model.
        probability: Model-level probability assigned to each active
            parameter before one-qubit/two-qubit splitting.

    Returns:
        Dict[str, Any]: Keyword arguments ready to merge into `MDRCircuit`.
    """
    kwargs: Dict[str, Any] = {
        "p_x": 0.0,
        "p_y": 0.0,
        "p_z": 0.0,
        "g1_x": 0.0,
        "g1_y": 0.0,
        "g1_z": 0.0,
        "gate_noise_2q": [0.0] * 15,
    }
    one_q_params = [
        name for name in param_names if name in MdrNoiseSweep.single_params
    ]
    two_q_params = [
        name for name in param_names if name not in MdrNoiseSweep.single_params
    ]

    for name in param_names:
        if name in MdrNoiseSweep.single_params:
            kwargs[name] = probability / max(len(one_q_params), 1)
            continue
        index = MdrNoiseSweep.two_qubit_index[name]
        kwargs["gate_noise_2q"][index] = probability / max(len(two_q_params), 1)

    sum_2q = float(sum(kwargs["gate_noise_2q"]))
    if sum_2q > 1.0 - 1e-9:
        scale = (1.0 / sum_2q) * 0.999
        kwargs["gate_noise_2q"] = [p * scale for p in kwargs["gate_noise_2q"]]

    sum_1q = float(kwargs["g1_x"] + kwargs["g1_y"] + kwargs["g1_z"])
    if sum_1q > 1.0 - 1e-9:
        scale = (1.0 / sum_1q) * 0.999
        kwargs["g1_x"] *= scale
        kwargs["g1_y"] *= scale
        kwargs["g1_z"] *= scale

    return kwargs


def rows_for_noise_model(
    *,
    code_family: str,
    distance: int,
    noise_model: str,
    probability: float,
    p_spam: float,
    shots: int,
    num_replicates: int,
    max_rounds: int,
) -> List[Dict[str, object]]:
    """
    Run one final-round simulation and return notebook-compatible rows.

    Args:
        code_family: Code family to simulate.
        distance: Code distance used to load operators and the MDR table.
        noise_model: Named noise model key such as `pure_z`.
        probability: Reference model-level physical-noise probability.
        p_spam: SPAM error probability.
        shots: Shot count used for each replicate estimate.
        num_replicates: Number of replicate estimates per round.
        max_rounds: Largest MDR round index included in the dataset.

    Returns:
        List[Dict[str, object]]: Long-form replicate rows matching the schema
        expected by `NotebookFinalRoundAnalysis`.
    """
    table_csv = canonical_table_path(
        distance=distance,
        code_family=code_family,
    )
    code_inputs = build_code_inputs(
        distance=distance,
        table_csv=table_csv,
        code_family=code_family,
    )
    param_names = noise_param_names(noise_model)
    noise_kwargs = build_noise_kwargs(param_names, probability)

    sim = MDRSimulation(
        mdr=MDRCircuit(
            stabilizers=code_inputs["code_stabilizers"],  # type: ignore[arg-type]
            toggles=code_inputs["combined_toggles"],  # type: ignore[arg-type]
            ancillas=1,
            p_spam=p_spam,
            psi_circuit=code_inputs["psi_circuit"],
            recovery_mode="final_round",
            correction_mode="physical",
            **noise_kwargs,
        ),
        stabilizer_pauli_strings=code_inputs["stabilizers"],  # type: ignore[arg-type]
        logical_pauli_strings=code_inputs["logical_operators"],  # type: ignore[arg-type]
        shots_per_measurement=shots,
        total_mdr_rounds=max_rounds,
        num_replicates=num_replicates,
    )

    rows: List[Dict[str, object]] = []
    display_name = NOISE_MODEL_DISPLAY_NAMES[noise_model]

    for operator, dist_map in sim._replicate_means_stabilizers.items():
        for round_idx, values in sorted(dist_map.items()):
            for replicate_idx, fidelity in enumerate(values):
                rows.append(
                    {
                        "noise_model": noise_model,
                        "display_name": display_name,
                        "category": "stabilizer",
                        "operator": operator,
                        "round": int(round_idx),
                        "replicate_idx": int(replicate_idx),
                        "fidelity": float(fidelity),
                        "p_spam": float(p_spam),
                        "recovery_mode": "final_round",
                        "shots": int(shots),
                        "num_replicates": int(num_replicates),
                        "distance": int(distance),
                        "signed_fidelity": None,
                    }
                )

    for operator, dist_map in sim._replicate_means_logicals.items():
        signed_map = sim._replicate_means_logicals_signed[operator]
        for round_idx, values in sorted(dist_map.items()):
            signed_values = signed_map[round_idx]
            for replicate_idx, (fidelity, signed_fidelity) in enumerate(
                zip(values, signed_values)
            ):
                rows.append(
                    {
                        "noise_model": noise_model,
                        "display_name": display_name,
                        "category": "logical",
                        "operator": operator,
                        "round": int(round_idx),
                        "replicate_idx": int(replicate_idx),
                        "fidelity": float(fidelity),
                        "p_spam": float(p_spam),
                        "recovery_mode": "final_round",
                        "shots": int(shots),
                        "num_replicates": int(num_replicates),
                        "distance": int(distance),
                        "signed_fidelity": float(signed_fidelity),
                    }
                )

    return rows


def default_output_csv(code_family: str, distance: int) -> Path:
    """
    Return the canonical analysis CSV path for one final-round dataset.

    Args:
        code_family: Code family owning the dataset.
        distance: Code distance encoded in the filename.

    Returns:
        Path: Family-scoped analysis CSV path under `data/analysis/`.
    """
    return (
        Path("data")
        / "analysis"
        / code_family
        / f"d{distance}_{code_family}_final_round_fidelity_replicates.csv"
    )


def main() -> None:
    """
    Generate the requested final-round replicate dataset and save it.

    Returns:
        None
    """
    args = parse_args()
    noise_points = parse_noise_points(args.noise_point, args.noise_models)
    output_csv = args.output_csv or default_output_csv(
        code_family=args.code_family,
        distance=args.distance,
    )

    all_rows: List[Dict[str, object]] = []
    for noise_model in args.noise_models:
        probability = noise_points[noise_model]
        print(
            "Running "
            f"{args.code_family} d={args.distance} {noise_model} "
            f"at p={probability:.12g}"
        )
        all_rows.extend(
            rows_for_noise_model(
                code_family=args.code_family,
                distance=args.distance,
                noise_model=noise_model,
                probability=probability,
                p_spam=args.p_spam,
                shots=args.shots,
                num_replicates=args.num_replicates,
                max_rounds=args.max_rounds,
            )
        )

    df = pd.DataFrame(all_rows)
    df = df[
        [
            "noise_model",
            "display_name",
            "category",
            "operator",
            "round",
            "replicate_idx",
            "fidelity",
            "p_spam",
            "recovery_mode",
            "shots",
            "num_replicates",
            "distance",
            "signed_fidelity",
        ]
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, float_format="%.12g")
    print(f"Saved {len(df)} rows to {output_csv.resolve()}")


if __name__ == "__main__":
    main()
