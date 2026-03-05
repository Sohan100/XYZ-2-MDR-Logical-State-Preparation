from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import stim

from .constants import DEFAULT_RESULTS_DIR, NOISE_MODEL_PARAM_NAMES
from .mdr_noise_sweep import MdrNoiseSweep
from .mdr_table import MDRTable

SIM_SPEC_VERSION = 1


def noise_param_names(noise_model: str) -> List[str]:
    """
    Resolve the parameter names associated with a named noise model.

    Args:
        noise_model: Key in `NOISE_MODEL_PARAM_NAMES`.

    Returns:
        List[str]: Ordered parameter names for the selected model.

    Raises:
        ValueError: If `noise_model` is unknown.
    """
    if noise_model not in NOISE_MODEL_PARAM_NAMES:
        valid = ", ".join(sorted(NOISE_MODEL_PARAM_NAMES))
        raise ValueError(
            f"Unknown noise model '{noise_model}'. Valid values: {valid}")
    return NOISE_MODEL_PARAM_NAMES[noise_model]


def build_code_inputs(
    distance: int,
    table_csv: str | Path | None = None,
) -> Dict[str, object]:
    """
    Build all code-level inputs required to run MDR sweeps.

    This helper loads an existing table when available, otherwise generates
    one, then derives stabilizer/logical structures and an initial `psi`
    circuit for Logical-X preparation.

    Args:
        distance: Code distance used when generating a new table.
        table_csv: Optional path to an existing or target table CSV.

    Returns:
        Dict[str, object]: Dictionary containing stabilizers, logicals,
        toggles, combined code operators, `psi_circuit`, and logical X.
    """
    table_path = Path(table_csv) if table_csv is not None else None
    if table_path is not None and table_path.exists():
        table = MDRTable.from_csv(table_path)
    elif table_path is not None:
        table = MDRTable(distance=distance, save_filename=table_path)
    else:
        table = MDRTable(distance=distance)

    stabs = table.get_stabilizers()
    logs = table.get_logicals_dict()
    stab_toggles, log_x_toggle = table.get_toggles()

    logical_x = logs["Logical X"]
    logical_ops = {
        "Logical X": logical_x,
        "Logical Y": logs["Logical Y"],
        "Logical Z": logs["Logical Z"],
    }
    combined_toggles = stab_toggles + [log_x_toggle]
    code_stabilizers = stabs + [logical_x]

    psi = stim.Circuit()
    lx_qubits = [int(term[1:]) for term in logical_x.split()]
    psi.append_operation("H", lx_qubits)

    return {
        "stabilizers": stabs,
        "logical_operators": logical_ops,
        "combined_toggles": combined_toggles,
        "code_stabilizers": code_stabilizers,
        "psi_circuit": psi,
        "logical_x": logical_x,
    }


def run_noise_sweep(
    distance: int,
    noise_model: str,
    probabilities: List[float],
    rounds: List[int],
    shots: int,
    num_replicates: int,
    p_spam: float,
    table_csv: str | Path | None = None,
    save_csv: str | Path | None = None,
) -> MdrNoiseSweep:
    """
    Create and execute a configured `MdrNoiseSweep`.

    Args:
        distance: Code distance.
        noise_model: Noise model key used to determine swept parameters.
        probabilities: Shared probability list for swept parameters.
        rounds: MDR rounds to record.
        shots: Shot count per expectation estimate.
        num_replicates: Replicate count per round/combo.
        p_spam: SPAM noise probability.
        table_csv: Optional path to load/create the MDR table.
        save_csv: Optional path for saving flattened sweep results.

    Returns:
        MdrNoiseSweep: Executed sweep object containing results and plotting
        utilities.
    """
    code_inputs = build_code_inputs(distance=distance, table_csv=table_csv)
    param_names = noise_param_names(noise_model)
    return MdrNoiseSweep(
        # type: ignore[arg-type]
        code_stabilizers=code_inputs["code_stabilizers"],
        toggles=code_inputs["combined_toggles"],  # type: ignore[arg-type]
        # type: ignore[arg-type]
        measure_stabilizers=code_inputs["stabilizers"],
        # type: ignore[arg-type]
        logical_operators=code_inputs["logical_operators"],
        ancillas=1,
        psi_circuit=code_inputs["psi_circuit"],
        p_spam=p_spam,
        param_names=param_names,
        param_values=probabilities,
        round_list=rounds,
        shots=shots,
        num_replicates=num_replicates,
        save_data_filename=save_csv,
    )


def build_simulation_spec(
    distance: int,
    noise_model: str,
    probabilities: List[float],
    rounds: List[int],
    shots: int,
    num_replicates: int,
    p_spam: float,
) -> Dict[str, Any]:
    """
    Build a canonical simulation specification used for caching.

    Args:
        distance: Code distance.
        noise_model: Noise model key.
        probabilities: Probability sweep values.
        rounds: MDR rounds included in the simulation.
        shots: Shot count per expectation estimate.
        num_replicates: Replicate count per round.
        p_spam: SPAM noise probability.

    Returns:
        Dict[str, Any]: Canonical simulation specification.
    """
    return {
        "spec_version": SIM_SPEC_VERSION,
        "distance": int(distance),
        "noise_model": str(noise_model),
        "param_names": noise_param_names(str(noise_model)),
        "probabilities": [float(p) for p in probabilities],
        "rounds": [int(r) for r in rounds],
        "shots": int(shots),
        "num_replicates": int(num_replicates),
        "p_spam": float(p_spam),
        "split_2q": True,
    }


def simulation_spec_hash(spec: Dict[str, Any]) -> str:
    """
    Compute a short, deterministic hash for a simulation specification.

    Args:
        spec: Canonical simulation specification dictionary.

    Returns:
        str: 12-character hash digest.
    """
    canonical = json.dumps(
        spec,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest[:12]


def simulation_results_path(
    results_dir: str | Path,
    spec: Dict[str, Any],
) -> Path:
    """
    Return the canonical CSV output path for a simulation spec.

    Args:
        results_dir: Root directory for simulation CSV files.
        spec: Canonical simulation specification.

    Returns:
        Path: Deterministic CSV path for this spec.
    """
    spec_hash = simulation_spec_hash(spec)
    noise_model = str(spec["noise_model"])
    distance = int(spec["distance"])
    shots = int(spec["shots"])
    reps = int(spec["num_replicates"])
    p_spam = float(spec["p_spam"])
    p_spam_tag = f"{p_spam:.3e}".replace("+", "")
    filename = (
        f"results_{noise_model}_d{distance}_"
        f"pspam{p_spam_tag}_shots{shots}_reps{reps}_spec-{spec_hash}.csv"
    )
    return Path(results_dir) / filename


def simulation_spec_path(csv_path: str | Path) -> Path:
    """
    Return the JSON sidecar path storing the simulation specification.

    Args:
        csv_path: Path to simulation CSV output.

    Returns:
        Path: Sidecar JSON specification path.
    """
    path = Path(csv_path)
    return path.with_suffix(".spec.json")


def run_noise_sweep_with_cache(
    distance: int,
    noise_model: str,
    probabilities: List[float],
    rounds: List[int],
    shots: int,
    num_replicates: int,
    p_spam: float,
    table_csv: str | Path | None = None,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    force_rerun: bool = False,
) -> Tuple[MdrNoiseSweep, Path, bool]:
    """
    Run or load a simulation based on an exact parameter specification.

    If a CSV for the exact same specification already exists and
    `force_rerun=False`, the simulation is loaded from disk instead of being
    recomputed.

    Args:
        distance: Code distance.
        noise_model: Noise model key.
        probabilities: Probability sweep values.
        rounds: MDR rounds included in the simulation.
        shots: Shot count per expectation estimate.
        num_replicates: Replicate count per round.
        p_spam: SPAM noise probability.
        table_csv: Optional table path for loading/generation.
        results_dir: Directory where simulation CSV files are stored.
        force_rerun: If True, recompute even when cached output exists.

    Returns:
        Tuple[MdrNoiseSweep, Path, bool]:
        `(sweep, csv_path, loaded_from_cache)`.
    """
    spec = build_simulation_spec(
        distance=distance,
        noise_model=noise_model,
        probabilities=probabilities,
        rounds=rounds,
        shots=shots,
        num_replicates=num_replicates,
        p_spam=p_spam,
    )
    csv_path = simulation_results_path(results_dir=results_dir, spec=spec)
    spec_path = simulation_spec_path(csv_path)

    if csv_path.exists() and not force_rerun:
        if not spec_path.exists():
            spec_path.parent.mkdir(parents=True, exist_ok=True)
            spec_path.write_text(
                json.dumps(spec, indent=2),
                encoding="utf-8",
            )
        print(f"Simulation already exists: {csv_path}")
        print("Loading cached simulation results.")
        sweep = MdrNoiseSweep(load_data_filename=csv_path)
        return sweep, csv_path, True

    sweep = run_noise_sweep(
        distance=distance,
        noise_model=noise_model,
        probabilities=probabilities,
        rounds=rounds,
        shots=shots,
        num_replicates=num_replicates,
        p_spam=p_spam,
        table_csv=table_csv,
        save_csv=csv_path,
    )
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return sweep, csv_path, False


def load_table_components(table_csv: str | Path) -> pd.DataFrame:
    """
    Load a saved MDR table CSV into a dataframe.

    Args:
        table_csv: Path to table CSV.

    Returns:
        pd.DataFrame: Loaded table contents.
    """
    return pd.read_csv(table_csv)
