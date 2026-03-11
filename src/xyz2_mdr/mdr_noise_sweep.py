"""
mdr_noise_sweep.py
------------------
Parameter-sweep execution and CSV persistence for MDR simulations.

This module coordinates repeated `MDRSimulation` runs over one or more noise
parameters, aggregates the resulting observable statistics, and exposes the
in-memory structures used by plotting helpers and cache-aware workflows.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .constants import DEFAULT_RESULTS_DIR
from .mdr_circuit import MDRCircuit
from .mdr_simulation import MDRSimulation


class MdrNoiseSweep:
    """
    Sweep noise parameters and record average observables and error rates.
    Supports saving simulation results to CSV and reloading them for later
    analysis.

    Attributes
    ----------
    two_qubit_index : Dict[str, int]
        Maps two-qubit noise keys such as `ZZ` to indices into the 15-entry
        `gate_noise_2q` list used by the MDR circuit engine.
    single_params : set[str]
        Valid single-qubit noise parameter names such as `g1_z` and `p_x`.
    code_stabilizers : List[str]
        Code operators passed into the MDR circuit builder for syndrome
        extraction and recovery logic.
    toggles : List[str]
        Recovery toggle strings aligned with `code_stabilizers`.
    measure_stabilizers : List[str]
        Stabilizer observables reported in the stored results.
    logical_operators : Dict[str, str]
        Mapping from logical labels to sparse Pauli strings.
    ancillas : int
        Number of ancilla qubits used during syndrome extraction.
    psi_circuit : Any
        Optional Stim circuit preparing the initial logical state.
    p_spam : float
        SPAM error probability applied during state preparation and
        measurement.
    recovery_mode : str
        Whether recovery toggles are applied after each round or only after
        the final round.
    round_list : List[int]
        MDR rounds whose results are stored in this sweep object.
    shots : int
        Shots used for each replicate expectation estimate.
    num_replicates : int
        Number of independent repeats per parameter combination.
    split_2q : bool
        Whether one-qubit and two-qubit probability budgets are split across
        active parameters of the same type.
    sync : bool
        True when a single shared list of values is used for all parameters.
    param_names : List[str]
        Names of the noise parameters being varied in this sweep.
    param_values_list : List[float]
        Shared list of parameter values when `sync=True`.
    param_values_map : Dict[str, List[float]]
        Per-parameter lists of values when `sync=False`.
    param_combos : List[Tuple[float, ...]]
        Explicit parameter tuples simulated or loaded from CSV.
    results : Dict[Tuple[float, ...], Dict[int, Dict[str, float]]]
        Mean `|<O>|` values for each parameter tuple, round, and operator.
    results_std : Dict[Tuple[float, ...], Dict[int, Dict[str, float]]]
        Sample standard deviations corresponding to `results`.
    results_signed : Dict[Tuple[float, ...], Dict[int, Dict[str, float]]]
        Signed means for each parameter tuple, round, and operator. Legacy
        CSVs without signed columns populate these values from `results`.
    results_std_signed : Dict[Tuple[float, ...], Dict[int, Dict[str, float]]]
        Signed sample standard deviations for each parameter tuple, round,
        and operator.
    has_exact_signed_results : bool
        Whether the signed columns were produced exactly by simulation or
        loaded exactly from a new-format CSV.

    Methods
    -------
    __init__(...)
        Configure sweep settings, build parameter combinations, and either
        run the simulations or load results from disk.
    _perform_sweep()
        Run `MDRSimulation` for each parameter tuple while applying the
        project-specific one-qubit and two-qubit splitting rules.
    save_results(filename)
        Export flattened simulation results to a CSV file.
    load_results(filename)
        Import results from a CSV file and reconstruct the in-memory sweep
        state.
    _metric_series_for_operator(round_idx, operator, metric, allow_legacy_approx)
        Extract aligned x/y/error arrays for one plotted metric series.
    """

    two_qubit_index: Dict[str, int] = {
        "IX": 0,
        "IY": 1,
        "IZ": 2,
        "XI": 3,
        "XX": 4,
        "XY": 5,
        "XZ": 6,
        "YI": 7,
        "YX": 8,
        "YY": 9,
        "YZ": 10,
        "ZI": 11,
        "ZX": 12,
        "ZY": 13,
        "ZZ": 14,
    }
    single_params = {"p_x", "p_y", "p_z", "g1_x", "g1_y", "g1_z"}

    def __init__(
        self,
        code_stabilizers: Optional[List[str]] = None,
        toggles: Optional[List[str]] = None,
        measure_stabilizers: Optional[List[str]] = None,
        logical_operators: Optional[Dict[str, str]] = None,
        ancillas: int = 1,
        psi_circuit: Any = None,
        p_spam: float = 0.0,
        recovery_mode: str = "each_round",
        param_names: Union[str, List[str]] = (),
        param_values: Union[List[float], Dict[str, List[float]]] = (),
        round_list: List[int] = [1],
        shots: int = 1000,
        num_replicates: int = 30,
        split_2q: bool = True,
        save_data_filename: Optional[str | Path] = None,
        load_data_filename: Optional[str | Path] = None,
    ) -> None:
        """
        Configure, run, or load a complete MDR noise-parameter sweep.

        The constructor supports two modes:
        1. Simulation mode, where circuit inputs and sweep values are
           provided and all parameter combinations are executed immediately.
        2. Load mode, where an existing CSV is parsed and the in-memory result
           structures are reconstructed without rerunning Stim.

        Args:
            code_stabilizers: Pauli strings defining the code checks used by
                the MDR circuit.
            toggles: Recovery toggle strings aligned with the code checks.
            measure_stabilizers: Stabilizer observables reported in outputs.
            logical_operators: Logical observables reported in outputs.
            ancillas: Number of ancilla qubits used during syndrome
                extraction.
            psi_circuit: Optional Stim circuit preparing the initial state.
            p_spam: SPAM error probability.
            recovery_mode: Whether recovery toggles are applied
                `each_round` or only on the `final_round`.
            param_names: Name or list of names of the noise parameters to
                sweep.
            param_values: Shared list (synchronous sweep) or per-parameter
                mapping (asynchronous sweep) of values.
            round_list: MDR round indices to retain in the stored results.
            shots: Shots per replicate expectation estimate.
            num_replicates: Independent repeats per parameter combination.
            split_2q: If True, split a shared probability budget across all
                active one-qubit or two-qubit channels of the same type.
            save_data_filename: Optional CSV path written after simulation.
            load_data_filename: Optional CSV path to load instead of running.

        Raises:
            ValueError: If required simulation inputs are missing in
                simulation mode, `param_names` is empty, or asynchronous sweep
                values are missing keys for requested parameters.
        """
        if load_data_filename is not None:
            self.load_results(load_data_filename)
            return

        if (
            code_stabilizers is None
            or toggles is None
            or measure_stabilizers is None
            or logical_operators is None
        ):
            raise ValueError(
                "Simulation mode requires code_stabilizers, toggles, "
                "measure_stabilizers, and logical_operators."
            )

        self.code_stabilizers = code_stabilizers
        self.toggles = toggles
        self.measure_stabilizers = measure_stabilizers
        self.logical_operators = logical_operators
        self.ancillas = ancillas
        self.psi_circuit = psi_circuit
        self.p_spam = p_spam
        self.recovery_mode = recovery_mode
        self.round_list = round_list
        self.shots = shots
        self.num_replicates = num_replicates
        self.split_2q = split_2q

        self.param_names = [param_names] if isinstance(
            param_names, str
        ) else list(param_names)
        if not self.param_names:
            raise ValueError("param_names cannot be empty in simulation mode.")

        if isinstance(param_values, list):
            self.sync = True
            self.param_values_list = list(param_values)
            self.param_values_map = {
                name: list(param_values) for name in self.param_names
            }
        else:
            self.sync = False
            self.param_values_map = dict(param_values)
            missing = set(self.param_names) - set(self.param_values_map)
            if missing:
                raise ValueError(f"Missing param_values for keys: {missing}")
            first_name = self.param_names[0]
            self.param_values_list = list(self.param_values_map[first_name])

        if self.sync:
            self.param_combos = [
                tuple([p] * len(self.param_names))
                for p in self.param_values_list
            ]
        else:
            value_lists = [self.param_values_map[n] for n in self.param_names]
            self.param_combos = list(product(*value_lists))

        (
            self.results,
            self.results_std,
            self.results_signed,
            self.results_std_signed,
        ) = self._perform_sweep()
        self.has_exact_signed_results = True
        if save_data_filename is not None:
            self.save_results(save_data_filename)

    def _perform_sweep(
        self,
    ) -> Tuple[
        Dict[Tuple[float, ...], Dict[int, Dict[str, float]]],
        Dict[Tuple[float, ...], Dict[int, Dict[str, float]]],
        Dict[Tuple[float, ...], Dict[int, Dict[str, float]]],
        Dict[Tuple[float, ...], Dict[int, Dict[str, float]]],
    ]:
        """
        Execute the configured sweep over all parameter combinations.

        For each parameter tuple, this method constructs an `MDRCircuit`,
        applies the project's one-qubit and two-qubit probability-splitting
        conventions, enforces a small safety margin when probabilities sum
        close to one, then runs a corresponding `MDRSimulation`. The output
        includes both absolute-value and signed summaries so later analysis
        can distinguish legacy fidelity-style metrics from exact logical
        state-preparation error metrics.

        Returns:
            Tuple[
                Dict[Tuple[float, ...], Dict[int, Dict[str, float]]],
                Dict[Tuple[float, ...], Dict[int, Dict[str, float]]],
                Dict[Tuple[float, ...], Dict[int, Dict[str, float]]],
                Dict[Tuple[float, ...], Dict[int, Dict[str, float]]],
            ]:
            `(all_means, all_stds, all_signed_means, all_signed_stds)` where
            each top-level key is a parameter tuple and each nested mapping is
            `round_index -> {operator_label: summary_value}`.
        """
        all_means: Dict[Tuple[float, ...], Dict[int, Dict[str, float]]] = {}
        all_stds: Dict[Tuple[float, ...], Dict[int, Dict[str, float]]] = {}
        all_signed_means: Dict[
            Tuple[float, ...], Dict[int, Dict[str, float]]
        ] = {}
        all_signed_stds: Dict[
            Tuple[float, ...], Dict[int, Dict[str, float]]
        ] = {}

        one_q_params = [n for n in self.param_names if n in self.single_params]
        two_q_params = [
            n for n in self.param_names if n not in self.single_params
        ]
        num_1q = len(one_q_params)
        num_2q = len(two_q_params)

        for combo in self.param_combos:
            kwargs: Dict[str, Any] = {
                "stabilizers": self.code_stabilizers,
                "toggles": self.toggles,
                "ancillas": self.ancillas,
                "p_spam": self.p_spam,
                "recovery_mode": self.recovery_mode,
                "p_x": 0.0,
                "p_y": 0.0,
                "p_z": 0.0,
                "g1_x": 0.0,
                "g1_y": 0.0,
                "g1_z": 0.0,
                "gate_noise_2q": [0.0] * 15,
                "psi_circuit": self.psi_circuit,
            }

            for name, val in zip(self.param_names, combo):
                if name in self.single_params:
                    if self.split_2q and num_1q > 0:
                        kwargs[name] = val / num_1q
                    else:
                        kwargs[name] = val
                else:
                    idx = self.two_qubit_index[name]
                    if self.split_2q and num_2q > 0:
                        kwargs["gate_noise_2q"][idx] = val / num_2q
                    else:
                        kwargs["gate_noise_2q"][idx] = val

            sum_2q = float(sum(kwargs["gate_noise_2q"]))
            if sum_2q > 1.0 - 1e-9:
                scale = (1.0 / sum_2q) * 0.999
                kwargs["gate_noise_2q"] = [
                    p * scale for p in kwargs["gate_noise_2q"]
                ]

            sum_1q = float(kwargs["g1_x"] + kwargs["g1_y"] + kwargs["g1_z"])
            if sum_1q > 1.0 - 1e-9:
                scale = (1.0 / sum_1q) * 0.999
                kwargs["g1_x"] *= scale
                kwargs["g1_y"] *= scale
                kwargs["g1_z"] *= scale

            sim = MDRSimulation(
                mdr=MDRCircuit(**kwargs),
                stabilizer_pauli_strings=self.measure_stabilizers,
                logical_pauli_strings=self.logical_operators,
                shots_per_measurement=self.shots,
                total_mdr_rounds=max(self.round_list),
                num_replicates=self.num_replicates,
            )

            mean_dict = {round_idx: {} for round_idx in self.round_list}
            std_dict = {round_idx: {} for round_idx in self.round_list}
            signed_mean_dict = {round_idx: {} for round_idx in self.round_list}
            signed_std_dict = {round_idx: {} for round_idx in self.round_list}

            for label in self.measure_stabilizers:
                stats = sim._stats_stabilizers[label]
                for r, ctr, sd in zip(
                    stats["rounds"],
                    stats["centers"],
                    stats["stds"],
                ):
                    if r in mean_dict:
                        mean_dict[r][label] = ctr
                        std_dict[r][label] = sd
                        signed_mean_dict[r][label] = ctr
                        signed_std_dict[r][label] = sd

            for label in self.logical_operators:
                stats = sim._stats_logicals[label]
                signed_stats = sim._stats_logicals_signed[label]
                for r, ctr, sd in zip(
                    stats["rounds"],
                    stats["centers"],
                    stats["stds"],
                ):
                    if r in mean_dict:
                        mean_dict[r][label] = ctr
                        std_dict[r][label] = sd
                for r, ctr, sd in zip(
                    signed_stats["rounds"],
                    signed_stats["centers"],
                    signed_stats["stds"],
                ):
                    if r in signed_mean_dict:
                        signed_mean_dict[r][label] = ctr
                        signed_std_dict[r][label] = sd

            all_means[combo] = mean_dict
            all_stds[combo] = std_dict
            all_signed_means[combo] = signed_mean_dict
            all_signed_stds[combo] = signed_std_dict

        return all_means, all_stds, all_signed_means, all_signed_stds

    def save_results(self, filename: str | Path) -> None:
        """
        Flatten in-memory sweep results and write them to a CSV file.

        Each output row corresponds to one `(parameter_combo, round,
        operator)` triple and contains both the legacy absolute-value metrics
        (`mean`, `std`) and the signed logical columns
        (`mean_signed`, `std_signed`).

        Args:
            filename: Destination CSV path. If a bare filename is supplied, it
                is written under `data/simulation_results/`.
        """
        rows: List[Dict[str, Any]] = []
        for combo, round_dict in self.results.items():
            for round_idx, ops_dict in round_dict.items():
                for label, mean_val in ops_dict.items():
                    row: Dict[str, Any] = {}
                    for pname, pval in zip(self.param_names, combo):
                        row[pname] = pval
                    row["round"] = round_idx
                    row["operator"] = label
                    row["mean"] = mean_val
                    row["std"] = self.results_std[combo][round_idx][label]
                    row["mean_signed"] = self.results_signed[combo][round_idx][
                        label
                    ]
                    row["std_signed"] = self.results_std_signed[combo][
                        round_idx
                    ][label]
                    rows.append(row)

        df = pd.DataFrame(rows)
        cols = self.param_names + [
            "round",
            "operator",
            "mean",
            "std",
            "mean_signed",
            "std_signed",
        ]
        df = df[cols]

        out_path = Path(filename)
        if (not out_path.is_absolute()) and out_path.parent == Path("."):
            out_path = DEFAULT_RESULTS_DIR / out_path
        df = df.sort_values(
            by=[*self.param_names, "round", "operator"],
        ).reset_index(drop=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False, float_format="%.12g")
        print(f"Data saved to: {out_path.resolve()}")

    def load_results(self, filename: str | Path) -> None:
        """
        Load sweep results from CSV and reconstruct the analysis state.

        After this method completes, the instance behaves like a sweep that
        had been run in memory: parameter combinations, operator labels, round
        lists, and summary dictionaries are all restored. If the CSV predates
        the signed-output update, the signed dictionaries are populated from
        the absolute-value columns and `has_exact_signed_results` is set to
        False.

        Args:
            filename: Path to the CSV file to load. If the exact path does not
                exist, fallback lookups are attempted under
                `data/simulation_results/` and the legacy
                `simulation_results/` directory.

        Raises:
            FileNotFoundError: If the CSV cannot be found in any supported
                location.
        """
        in_path = Path(filename)
        if not in_path.exists():
            default_candidates = [
                DEFAULT_RESULTS_DIR / in_path,
                Path("simulation_results") / in_path,
            ]
            for candidate in default_candidates:
                if candidate.exists():
                    in_path = candidate
                    break
            else:
                raise FileNotFoundError(f"File not found: {filename}")

        df = pd.read_csv(in_path)
        reserved = {
            "round",
            "operator",
            "mean",
            "std",
            "mean_signed",
            "std_signed",
        }
        self.param_names = [col for col in df.columns if col not in reserved]
        self.results = {}
        self.results_std = {}
        self.results_signed = {}
        self.results_std_signed = {}
        self.has_exact_signed_results = {
            "mean_signed",
            "std_signed",
        }.issubset(df.columns)

        combo_df = df[self.param_names].drop_duplicates()
        self.param_combos = [
            tuple(row.tolist()) for _, row in combo_df.iterrows()
        ]

        if len(self.param_names) == 1:
            self.sync = True
        else:
            first = self.param_names[0]
            self.sync = all(
                (combo_df[first] == combo_df[name]).all()
                for name in self.param_names[1:]
            )

        if self.sync:
            self.param_values_list = sorted(
                {float(combo[0]) for combo in self.param_combos}
            )
        else:
            self.param_values_list = []

        self.param_values_map = {
            name: sorted(df[name].unique().tolist())
            for name in self.param_names
        }

        for combo in self.param_combos:
            self.results[combo] = {}
            self.results_std[combo] = {}
            self.results_signed[combo] = {}
            self.results_std_signed[combo] = {}
            mask = np.ones(len(df), dtype=bool)
            for name, val in zip(self.param_names, combo):
                mask &= df[name] == val
            subset = df[mask]
            for _, row in subset.iterrows():
                round_idx = int(row["round"])
                op = str(row["operator"])
                mn = float(row["mean"])
                sd = float(row["std"])
                mn_signed = float(row["mean_signed"]) if (
                    self.has_exact_signed_results
                ) else mn
                sd_signed = float(row["std_signed"]) if (
                    self.has_exact_signed_results
                ) else sd
                self.results[combo].setdefault(round_idx, {})[op] = mn
                self.results_std[combo].setdefault(round_idx, {})[op] = sd
                self.results_signed[combo].setdefault(
                    round_idx, {}
                )[op] = mn_signed
                self.results_std_signed[combo].setdefault(
                    round_idx, {}
                )[op] = sd_signed

        unique_ops = sorted(df["operator"].unique().tolist())
        logical_labels = [op for op in unique_ops if op.startswith("Logical")]
        stab_labels = [op for op in unique_ops if not op.startswith("Logical")]
        self.measure_stabilizers = stab_labels
        self.logical_operators = {label: label for label in logical_labels}
        self.round_list = sorted(int(x) for x in df["round"].unique().tolist())
        print(
            "Loaded "
            f"{len(self.param_combos)} parameter combinations from "
            f"{in_path.resolve()}"
        )

    def _metric_series_for_operator(
        self,
        round_idx: int,
        operator: str,
        metric: str,
        allow_legacy_approx: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract a single operator metric series for plotting.

        This helper converts the nested in-memory dictionaries into aligned
        NumPy arrays ordered by increasing parameter tuple. It supports both
        the legacy observable-loss metric and the signed logical
        state-preparation error metric used by the updated plotting path.

        Args:
            round_idx: MDR round to plot.
            operator: Operator label to extract.
            metric: One of `observable_loss` or `state_prep_error`.
            allow_legacy_approx: If True, old CSVs without signed means may
                approximate state-preparation error as `(1 - |<X_L>|)/2`.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
            `(p_vals, y_vals, y_errs)` arrays ready for plotting.

        Raises:
            ValueError: If the metric name is unknown, the operator is invalid
                for state-preparation error, or exact signed results are
                required but unavailable.
        """
        combos = sorted(
            self.param_combos,
            key=lambda combo: tuple(float(x) for x in combo),
        )
        p_vals = np.array([float(combo[0]) for combo in combos], dtype=float)

        if metric == "observable_loss":
            means = np.array(
                [self.results[c][round_idx][operator] for c in combos],
                dtype=float,
            )
            stds = np.array(
                [self.results_std[c][round_idx][operator] for c in combos],
                dtype=float,
            )
            return p_vals, 1 - means, stds

        if metric != "state_prep_error":
            raise ValueError(f"Unknown metric '{metric}'.")
        if operator != "Logical X":
            raise ValueError(
                "state_prep_error is only defined for the target logical "
                "operator 'Logical X'."
            )
        if not self.has_exact_signed_results and not allow_legacy_approx:
            raise ValueError(
                "Loaded CSV lacks signed logical expectations. Exact "
                "state-preparation error requires rerunning the simulation "
                "with the updated code, or setting allow_legacy_approx=True."
            )

        means = np.array(
            [self.results_signed[c][round_idx][operator] for c in combos],
            dtype=float,
        )
        stds = np.array(
            [self.results_std_signed[c][round_idx][operator] for c in combos],
            dtype=float,
        )
        return p_vals, 0.5 * (1 - means), 0.5 * stds
