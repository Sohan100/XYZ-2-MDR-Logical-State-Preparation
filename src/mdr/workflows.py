"""
workflows.py
------------
High-level workflow helpers for running, caching, and reloading MDR sweeps.

This module sits above the low-level circuit and sweep classes. It assembles
code inputs from MDR tables, maps named noise models to their parameter sets,
builds canonical simulation specifications, and manages cache file naming so
local scripts and Slurm jobs can share the same result layout.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import stim

from .constants import (
    DEFAULT_RESULTS_DIR,
    NOISE_MODEL_PARAM_NAMES,
    SUPPORTED_CODE_FAMILIES,
)
from .mdr_noise_sweep import MdrNoiseSweep
from .mdr_table import MDRTable

SIM_SPEC_VERSION = 2


def code_family_subdir(
    base_dir: str | Path,
    code_family: str,
) -> Path:
    """
    Return the canonical family-specific subdirectory under a base path.

    If the provided path already ends in the requested family name, it is
    returned unchanged. This keeps CLI overrides ergonomic while still
    enforcing family-separated storage by default.
    """
    path = Path(base_dir)
    if path.name == code_family:
        return path
    return path / code_family


def resolve_family_search_dirs(
    base_dir: str | Path,
    code_family: str,
) -> List[Path]:
    """
    Return candidate directories to search for family-scoped artifacts.

    The family-specific directory is preferred, with nearby legacy flat
    locations retained as backwards-compatible fallbacks for older layouts.
    """
    base_path = Path(base_dir)
    family_dir = code_family_subdir(base_path, code_family)
    candidates = [family_dir]
    if family_dir == base_path:
        candidates.append(base_path.parent)
    else:
        candidates.append(base_path)

    unique_candidates: List[Path] = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


def resolve_slurm_run_dir(
    root_dir: str | Path,
    run_name: str,
    code_family: str | None = None,
) -> Path:
    """
    Resolve a Slurm run directory under the family-aware layout.

    The resolver first checks the explicit `code_family` path when provided,
    then the historical flat layout, and finally scans all supported family
    subdirectories for a unique match.
    """
    root = Path(root_dir)
    if code_family is not None:
        candidate = code_family_subdir(root, code_family) / run_name
        if candidate.exists():
            return candidate

    direct = root / run_name
    if direct.exists():
        return direct

    matches = [
        code_family_subdir(root, family) / run_name
        for family in SUPPORTED_CODE_FAMILIES
        if (code_family_subdir(root, family) / run_name).exists()
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileExistsError(
            f"Run name '{run_name}' exists under multiple code-family folders: "
            + ", ".join(str(path) for path in matches)
        )

    searched = [direct]
    if code_family is not None:
        searched.insert(0, code_family_subdir(root, code_family) / run_name)
    else:
        searched.extend(
            code_family_subdir(root, family) / run_name
            for family in SUPPORTED_CODE_FAMILIES
        )
    raise FileNotFoundError(
        "Could not locate Slurm run directory. Searched: "
        + ", ".join(str(path) for path in searched)
    )


def noise_param_names(noise_model: str) -> List[str]:
    """
    Resolve the parameter names associated with a named noise model.

    This helper provides the single source of truth for mapping a user-facing
    noise model name such as `"pure_z"` to the ordered parameter list expected
    by `MdrNoiseSweep`. The returned order is also embedded into the cached
    simulation specification, so it must remain deterministic.

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
    code_family: str = "xyz2",
) -> Dict[str, object]:
    """
    Build the full set of operator and preparation inputs for MDR sweeps.

    This helper loads an existing table when available, otherwise generates
    one, then derives stabilizer/logical structures and an initial `psi`
    circuit for Logical-X preparation. The returned dictionary is shaped to be
    consumed directly by the sweep and simulation helpers in this package.

    Args:
        distance: Code distance used when generating a new table.
        table_csv: Optional path to an existing or target table CSV.
        code_family: Code family to load or generate.

    Returns:
        Dict[str, object]: Dictionary containing stabilizers, logicals,
        toggles, combined code operators, `psi_circuit`, and logical X.

    Raises:
        KeyError: If the generated table does not provide the expected logical
            operators required by current MDR workflows.
    """
    if code_family not in SUPPORTED_CODE_FAMILIES:
        raise ValueError(
            "code_family must be one of: "
            + ", ".join(SUPPORTED_CODE_FAMILIES)
        )

    table_path = Path(table_csv) if table_csv is not None else None
    if table_path is not None and table_path.exists():
        table = MDRTable.from_csv(table_path)
        if table.code_family != code_family:
            raise ValueError(
                f"Requested code_family='{code_family}' but table "
                f"{table_path} stores '{table.code_family}'."
            )
    elif table_path is not None:
        table = MDRTable(
            distance=distance,
            save_filename=table_path,
            code_family=code_family,
        )
    else:
        table = MDRTable(distance=distance, code_family=code_family)

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
        "code_family": code_family,
        "psi_circuit": psi,
        "logical_x": logical_x,
    }


def default_table_filename(
    distance: int,
    code_family: str = "xyz2",
) -> str:
    """
    Return the canonical MDR table filename for one code family and distance.
    """
    return f"mdr_table_{code_family}_d{distance}.csv"


def run_noise_sweep(
    distance: int,
    noise_model: str,
    probabilities: List[float],
    rounds: List[int],
    shots: int,
    num_replicates: int,
    p_spam: float,
    recovery_mode: str = "each_round",
    correction_mode: str = "physical",
    table_csv: str | Path | None = None,
    save_csv: str | Path | None = None,
    code_family: str = "xyz2",
) -> MdrNoiseSweep:
    """
    Create and execute a configured `MdrNoiseSweep`.

    This wrapper combines table generation/loading, logical-state
    preparation, named-noise-model expansion, and sweep construction into a
    single call used by scripts, notebooks, and cache-aware workflow helpers.

    Args:
        distance: Code distance.
        noise_model: Noise model key used to determine swept parameters.
        probabilities: Shared probability list for swept parameters.
        rounds: MDR rounds to record.
        shots: Shot count per expectation estimate.
        num_replicates: Replicate count per round/combo.
        p_spam: SPAM noise probability.
        recovery_mode: Whether recovery toggles are applied after each round
            or only after the final round.
        correction_mode: Whether recovery is implemented with physical Pauli
            gates or by a tracked Pauli frame.
        table_csv: Optional path to load/create the MDR table.
        save_csv: Optional path for saving flattened sweep results.
        code_family: Code family whose MDR operators should be used.

    Returns:
        MdrNoiseSweep: Executed sweep object containing in-memory summary
        statistics and CSV-serializable results.
    """
    code_inputs = build_code_inputs(
        distance=distance,
        table_csv=table_csv,
        code_family=code_family,
    )
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
        recovery_mode=recovery_mode,
        correction_mode=correction_mode,
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
    recovery_mode: str = "each_round",
    correction_mode: str = "physical",
    code_family: str = "xyz2",
) -> Dict[str, Any]:
    """
    Build a canonical simulation specification used for caching.

    The resulting dictionary is normalized to plain Python scalar types so it
    can be serialized deterministically and hashed into stable result
    filenames. Any workflow change that should invalidate old cache files must
    be reflected here, typically by updating `SIM_SPEC_VERSION`.

    Args:
        distance: Code distance.
        noise_model: Noise model key.
        probabilities: Probability sweep values.
        rounds: MDR rounds included in the simulation.
        shots: Shot count per expectation estimate.
        num_replicates: Replicate count per round.
        p_spam: SPAM noise probability.
        recovery_mode: Whether recovery toggles are applied after each round
            or only after the final round.
        correction_mode: Whether recovery is implemented with physical Pauli
            gates or by a tracked Pauli frame.
        code_family: Code family simulated by the workflow.

    Returns:
        Dict[str, Any]: Canonical simulation specification.
    """
    spec = {
        "spec_version": SIM_SPEC_VERSION,
        "code_family": str(code_family),
        "distance": int(distance),
        "noise_model": str(noise_model),
        "param_names": noise_param_names(str(noise_model)),
        "probabilities": [float(p) for p in probabilities],
        "rounds": [int(r) for r in rounds],
        "shots": int(shots),
        "num_replicates": int(num_replicates),
        "p_spam": float(p_spam),
        "split_2q": True,
        "recovery_mode": str(recovery_mode),
        "correction_mode": str(correction_mode),
    }
    return spec


def simulation_spec_hash(spec: Dict[str, Any]) -> str:
    """
    Compute a short, deterministic hash for a simulation specification.

    The hash is used only for filenames and sidecar naming, not for
    cryptographic purposes. The canonical JSON encoding ensures equal specs
    always map to the same digest regardless of dictionary insertion order.

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

    The filename embeds human-readable summary fields such as distance and
    SPAM rate, plus a short spec hash to distinguish otherwise similar runs
    without requiring very long filenames.

    Args:
        results_dir: Root directory for simulation CSV files.
        spec: Canonical simulation specification.

    Returns:
        Path: Deterministic CSV path for this spec.
    """
    spec_hash = simulation_spec_hash(spec)
    code_family = str(spec.get("code_family", "xyz2"))
    noise_model = str(spec["noise_model"])
    distance = int(spec["distance"])
    shots = int(spec["shots"])
    reps = int(spec["num_replicates"])
    p_spam = float(spec["p_spam"])
    p_spam_tag = f"{p_spam:.3e}".replace("+", "")
    filename = (
        f"results_{code_family}_{noise_model}_d{distance}_"
        f"pspam{p_spam_tag}_shots{shots}_reps{reps}_spec-{spec_hash}.csv"
    )
    return code_family_subdir(results_dir, code_family) / filename


def simulation_spec_path(csv_path: str | Path) -> Path:
    """
    Return the JSON sidecar path storing the simulation specification.

    The sidecar is written alongside the CSV so cached data can always be
    traced back to the exact workflow inputs used to produce it.

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
    recovery_mode: str = "each_round",
    correction_mode: str = "physical",
    table_csv: str | Path | None = None,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    force_rerun: bool = False,
    code_family: str = "xyz2",
) -> Tuple[MdrNoiseSweep, Path, bool]:
    """
    Run or load a simulation based on an exact parameter specification.

    If a CSV for the exact same specification already exists and
    `force_rerun=False`, the simulation is loaded from disk instead of being
    recomputed. This keeps local iteration fast while preserving a fully
    reproducible mapping from simulation inputs to output filenames.

    Args:
        distance: Code distance.
        noise_model: Noise model key.
        probabilities: Probability sweep values.
        rounds: MDR rounds included in the simulation.
        shots: Shot count per expectation estimate.
        num_replicates: Replicate count per round.
        p_spam: SPAM noise probability.
        recovery_mode: Whether recovery toggles are applied after each round
            or only after the final round.
        correction_mode: Whether recovery is implemented with physical Pauli
            gates or by a tracked Pauli frame.
        table_csv: Optional table path for loading/generation.
        results_dir: Directory where simulation CSV files are stored.
        force_rerun: If True, recompute even when cached output exists.
        code_family: Code family whose MDR workflow should be used.

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
        recovery_mode=recovery_mode,
        correction_mode=correction_mode,
        code_family=code_family,
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
        recovery_mode=recovery_mode,
        correction_mode=correction_mode,
        table_csv=table_csv,
        save_csv=csv_path,
        code_family=code_family,
    )
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return sweep, csv_path, False


def load_table_components(table_csv: str | Path) -> pd.DataFrame:
    """
    Load a saved MDR table CSV into a dataframe.

    This is a thin convenience wrapper used by scripts and notebooks that
    want direct dataframe access without instantiating `MDRTable`.

    Args:
        table_csv: Path to table CSV.

    Returns:
        pd.DataFrame: Loaded table contents.
    """
    return pd.read_csv(table_csv)
