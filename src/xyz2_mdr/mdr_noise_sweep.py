"""
mdr_noise_sweep.py
────────────────────────────────────────────────────────────────────────────
Class for managing parameter sweeps, executing MDR simulations, and 
handling data persistence/visualisation.
"""

from __future__ import annotations

from itertools import product
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd

from .constants import DEFAULT_RESULTS_DIR
from .mdr_circuit import MDRCircuit
from .mdr_simulation import MDRSimulation


class MdrNoiseSweep:
    """
    Sweep noise parameters and record average observables and error rates.
    Supports saving simulation results to CSV and reloading them for plotting.
    
    Attributes
    ----------
    two_qubit_index : Dict[str, int]
        Maps two-qubit noise keys (e.g. 'ZZ') to indices into the 15-entry
        gate_noise_2q list used by the MDR circuit engine.
    single_params : set[str]
        Valid single-qubit noise parameter names (e.g. 'g1_z', 'p_x').
    sync : bool
        True if using a single shared list of values for all parameters (simultaneous sweep).
    param_names : list[str]
        Names of noise parameters being varied in this sweep.
    param_values_list : list[float]
        Shared list of noise values when sync=True.
    param_values_map : Dict[str, list[float]]
        Per-parameter lists of noise values when sync=False (asynchronous sweep).
    param_combos : list[tuple[float,...]]
        Cartesian-product of parameter values to sweep.
    results : Dict[tuple, Dict[int, Dict[str, float]]]
        Mean ⟨|O|⟩ values for each combo, round, and operator.
        Structure: results[param_combo][round_index][operator_label] = mean_val
    results_std : Dict[tuple, Dict[int, Dict[str, float]]]
        Sample standard deviations of ⟨|O|⟩ for each combo, round, and operator.
    results_signed : Dict[tuple, Dict[int, Dict[str, float]]]
        Signed means for each combo, round, and operator. For legacy CSVs
        without signed columns, these values are populated from `results`.
    results_std_signed : Dict[tuple, Dict[int, Dict[str, float]]]
        Signed sample standard deviations for each combo, round, and operator.
    
    Methods
    -------
    __init__(...)
        Configure sweep settings, build parameter combinations, and run (or load).
    _perform_sweep()
        Internal: run MDRSimulation per combo, applying 1q/2q splitting logic.
    save_results(filename)
        Export flattened simulation results to a CSV file.
    load_results(filename)
        Import results from a CSV file and reconstruct class state for plotting.
    plot_expectations_vs_param(...)
        Plot mean ⟨|O|⟩ vs noise parameter p.
    plot_error_rates_vs_param(...)
        Plot error rate = 1 - ⟨|O|⟩ vs noise p.
    plot_multi(...)
        Static: overlay or grid of multiple sweeps of ⟨|O|⟩ vs p.
    plot_error_multi(...)
        Static: overlay or grid of multiple sweeps of error rates with bars.
    table_round_delta(...)
        Print table comparing fidelity differences between rounds.
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

    # ─────────────────────────────────────────────────────────────────────
    # construction
    # ─────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        code_stabilizers: Optional[List[str]] = None,
        toggles: Optional[List[str]] = None,
        measure_stabilizers: Optional[List[str]] = None,
        logical_operators: Optional[Dict[str, str]] = None,
        ancillas: int = 1,
        psi_circuit: Any = None,
        p_spam: float = 0.0,
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
        Configure and execute the full noise-parameter sweep.
        
        Can operate in two modes:
        1. **Simulation Mode**: Runs the sweep and optionally saves data.
        2. **Load Mode**: Loads existing data from a CSV for plotting
           (skips execution).
        
        Args
        ----
        code_stabilizers : List[str]
            Pauli strings defining the code stabilizers.
        toggles : List[str]
            Pauli strings for recovery toggles.
        measure_stabilizers : List[str]
            Pauli strings to measure each round.
        logical_operators : Dict[str, str]
            Mapping from logical operator labels to Pauli strings.
        ancillas : int
            Number of ancilla qubits.
        psi_circuit : Any
            stim.Circuit preparing the initial state.
        p_spam : float
            SPAM error probability.
        param_names : str or List[str]
            Name(s) of noise parameters to sweep.
        param_values : List[float] or Dict[str, List[float]]
            Shared list (sync) or per-parameter lists (async) of values.
        round_list : List[int]
            MDR round indices to record.
        shots : int, optional
            Shots per Monte Carlo measurement. Default is 1000.
        num_replicates : int, optional
            Independent repeats per combo. Default is 30.
        split_2q : bool, default True
            If True, divides the input probability `p` evenly among all 
            active parameters of the same type (1-qubit or 2-qubit). 
            This prevents probabilities summing > 1 in Stim.
        save_data_filename : Optional[str]
            If provided, saves simulation results to this CSV path after
            running.
        load_data_filename : Optional[str]
            If provided, SKIPS simulation and loads data from this CSV. 
            Useful for plotting previously run jobs.
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

    # ─────────────────────────────────────────────────────────────────────
    # simulation
    # ─────────────────────────────────────────────────────────────────────
    def _perform_sweep(
        self,
    ) -> Tuple[
        Dict[Tuple[float, ...], Dict[int, Dict[str, float]]],
        Dict[Tuple[float, ...], Dict[int, Dict[str, float]]],
        Dict[Tuple[float, ...], Dict[int, Dict[str, float]]],
        Dict[Tuple[float, ...], Dict[int, Dict[str, float]]],
    ]:
        """
        Internal: run MDRSimulation for each combo and gather both means and
        stddevs.
        
        Implements the Probability Splitting Logic:
        - 1-qubit params get the full value `p` (or p/N if N params active).
        - 2-qubit params get `p / num_2q_params` if split_2q is True.
        
        Updates:
        - Includes strict safety check to ensure probabilities sum <= 1.0.
        - Uses a 0.999 safety factor if sum approaches 1 to counteract 
          string formatting rounding up in Stim (e.g. {:.6g}).
        
        Returns
        -------
        all_means : Dict[tuple, Dict[int, Dict[str, float]]]
            combo → round → {label: mean ⟨|O|⟩}
        all_stds : Dict[tuple, Dict[int, Dict[str, float]]]
            combo → round → {label: sample stddev}
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

    # ─────────────────────────────────────────────────────────────────────
    # io helpers
    # ─────────────────────────────────────────────────────────────────────
    def save_results(self, filename: str | Path) -> None:
        """
        Flatten the results dictionary and save to CSV.
        
        The CSV structure will be:
        [param_1, param_2, ..., round, operator, mean, std,
         mean_signed, std_signed]
        
        Args
        ----
        filename : str
            Path to save the CSV file. A bare filename is written under
            `data/simulation_results/`.
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
        Load results from CSV and reconstruct attributes needed for plotting.
        
        This allows a new MdrNoiseSweep object to behave as if it had run
        the simulation, enabling the usage of plotting methods.
        
        Args
        ----
        filename : str
            Path to the CSV file to load. If not found directly, this method
            also searches `data/simulation_results/` and the legacy
            `simulation_results/`.
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
        Return x/y data for a named plotting metric.

        Args:
            round_idx: MDR round to plot.
            operator: Operator label to extract.
            metric: One of `observable_loss` or `state_prep_error`.
            allow_legacy_approx: If True, old CSVs without signed means may
                approximate state-preparation error as `(1 - |<X_L>|)/2`.
        """
        combos = self._sorted_combos_for_plot()
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

    # ─────────────────────────────────────────────────────────────────────
    # plotting
    # ─────────────────────────────────────────────────────────────────────
    def _sorted_combos_for_plot(self) -> List[Tuple[float, ...]]:
        """
        Return parameter combinations sorted numerically for plotting.

        Sorting is done lexicographically on float-cast values so that plotted
        curves follow a deterministic left-to-right parameter progression.

        Returns:
            List[Tuple[float, ...]]: Sorted tuples of swept parameter values.
        """
        return sorted(
            self.param_combos,
            key=lambda combo: tuple(float(x) for x in combo),
        )

    @staticmethod
    def plot_error_multi(
        sweeps: Dict[str, "MdrNoiseSweep"],
        category: str,
        rounds: List[int],
        subset: Optional[List[str]] = None,
        overlay: bool = False,
        log_x: bool = False,
        figsize: Tuple[int, int] = (15, 6),
        save_path: Optional[str | Path] = None,
    ) -> None:
        """
        Plot error rate = 1 − ⟨|O|⟩ for multiple sweeps,
        connecting means with solid lines and ±1σ error bars.
        
        Args
        ----
        sweeps : Dict[str, MdrNoiseSweep]
            Map legend→sweep instance (requires .results & .results_std).
        category : {'stabilizer','logical'}
            Which operators to plot.
        rounds : List[int]
            MDR round indices to include.
        subset : Optional[List[str]]
            If provided, restrict to these operator labels.
        overlay : bool
            If True, overlay all rounds on one axis.
        log_x : bool
            If True, set x-axis to logarithmic scale.
        figsize : Tuple[int,int]
            Figure size in inches.
        save_path : Optional[str]
            Path to save the figure (dpi=300).
        
        Raises
        ------
        ValueError
            If sweeps empty or category/subset invalid.
        
        Returns
        -------
        None
        """
        if not sweeps:
            raise ValueError("No sweeps provided")

        first = next(iter(sweeps.values()))
        if category == "stabilizer":
            labels = list(first.measure_stabilizers)
        elif category == "logical":
            labels = list(first.logical_operators.keys())
        else:
            raise ValueError("category must be 'stabilizer' or 'logical'")

        if subset:
            missing = set(subset) - set(labels)
            if missing:
                raise ValueError(f"Unknown labels: {missing}")
            labels = subset

        colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        markers = [
            "o",
            "s",
            "^",
            "v",
            "<",
            ">",
            "8",
            "p",
            "P",
            "*",
            "h",
            "H",
            "+",
            "x",
            "X",
            "D",
            "d",
            "|",
            "_",
            ".",
        ]
        style_keys = [(model, op) for model in sweeps for op in labels]
        style_map = {
            key: (
                colours[idx % len(colours)],
                markers[(idx // len(colours)) % len(markers)],
            )
            for idx, key in enumerate(style_keys)
        }

        if overlay:
            fig, ax0 = plt.subplots(figsize=figsize)
            axes = [ax0]
        else:
            n = len(rounds)
            cols = min(3, n)
            rows = math.ceil(n / cols)
            fig, grid = plt.subplots(
                rows,
                cols,
                figsize=figsize,
                squeeze=False,
            )
            axes = list(grid.flatten())

        plot_axes = [axes[0]] if overlay else axes[: len(rounds)]
        if not overlay and len(axes) > len(rounds):
            for extra_ax in axes[len(rounds) :]:
                extra_ax.set_visible(False)

        for idx, ax in enumerate(plot_axes):
            if not overlay:
                round_idx = rounds[idx]
                ax.set_title(f"Round {round_idx}")
            ax.grid(True)

            for model, sweep in sweeps.items():
                combos = sweep._sorted_combos_for_plot()
                p_vals = np.array([float(combo[0]) for combo in combos])

                if overlay:
                    for op in labels:
                        for round_idx in rounds:
                            means = np.array(
                                [
                                    sweep.results[c][round_idx][op]
                                    for c in combos
                                ],
                                dtype=float,
                            )
                            stds = np.array(
                                [
                                    sweep.results_std[c][round_idx][op]
                                    for c in combos
                                ],
                                dtype=float,
                            )
                            color, marker = style_map[(model, op)]
                            ax.errorbar(
                                p_vals,
                                1 - means,
                                yerr=stds,
                                fmt=f"-{marker}",
                                color=color,
                                capsize=4,
                                label=f"{model}: {op} (r={round_idx})",
                            )
                else:
                    round_idx = rounds[idx]
                    for op in labels:
                        means = np.array(
                            [sweep.results[c][round_idx][op] for c in combos],
                            dtype=float,
                        )
                        stds = np.array(
                            [
                                sweep.results_std[c][round_idx][op]
                                for c in combos
                            ],
                            dtype=float,
                        )
                        color, marker = style_map[(model, op)]
                        ax.errorbar(
                            p_vals,
                            1 - means,
                            yerr=stds,
                            fmt=f"-{marker}",
                            color=color,
                            capsize=4,
                            label=f"{model}: {op}",
                        )

            if log_x:
                ax.set_xscale("log")
            ax.set_xlabel("p", fontsize=14)
            ax.set_ylabel("Error rate (1 - |<O>|)", fontsize=14)

        plt.tight_layout(rect=[0, 0, 0.75, 1])
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bboxes = [ax.get_tightbbox(renderer) for ax in plot_axes]
        union_bbox = Bbox.union(bboxes)
        bb = union_bbox.transformed(fig.transFigure.inverted())
        lx = bb.x1 + 0.01
        ly = bb.y0 + bb.height / 2

        handles, lbls = plot_axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            lbls,
            loc="center left",
            bbox_to_anchor=(lx, ly),
            fontsize="small",
        )

        if save_path is not None:
            out_path = Path(save_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
        if "agg" not in plt.get_backend().lower():
            plt.show()
        plt.close(fig)

    @staticmethod
    def plot_state_prep_error_multi(
        sweeps: Dict[str, "MdrNoiseSweep"],
        rounds: List[int],
        logical_label: str = "Logical X",
        overlay: bool = False,
        log_x: bool = False,
        figsize: Tuple[int, int] = (15, 6),
        save_path: Optional[str | Path] = None,
        allow_legacy_approx: bool = False,
    ) -> None:
        """
        Plot logical state-preparation error `(1 - <X_L>) / 2`.

        For legacy CSVs saved before signed expectations were preserved, this
        metric is only approximate because those files contain `|<X_L>|`
        instead of `<X_L>`.
        """
        if not sweeps:
            raise ValueError("No sweeps provided")

        if logical_label != "Logical X":
            raise ValueError(
                "state-preparation error is currently supported only for "
                "'Logical X'."
            )

        colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        markers = ["o", "s", "^", "v", "<", ">", "D", "P", "X", "*"]
        style_map = {
            model: (
                colours[idx % len(colours)],
                markers[idx % len(markers)],
            )
            for idx, model in enumerate(sweeps)
        }

        if overlay:
            fig, ax0 = plt.subplots(figsize=figsize)
            axes = [ax0]
        else:
            n = len(rounds)
            cols = min(3, n)
            rows = math.ceil(n / cols)
            fig, grid = plt.subplots(
                rows,
                cols,
                figsize=figsize,
                squeeze=False,
            )
            axes = list(grid.flatten())

        plot_axes = [axes[0]] if overlay else axes[: len(rounds)]
        if not overlay and len(axes) > len(rounds):
            for extra_ax in axes[len(rounds) :]:
                extra_ax.set_visible(False)

        legacy_models = [
            model
            for model, sweep in sweeps.items()
            if not sweep.has_exact_signed_results
        ]
        if legacy_models and allow_legacy_approx:
            print(
                "Warning: using legacy approximate state-preparation error "
                "for "
                + ", ".join(legacy_models)
                + "."
            )

        for idx, ax in enumerate(plot_axes):
            if not overlay:
                round_idx = rounds[idx]
                ax.set_title(f"Round {round_idx}")
            ax.grid(True)

            for model, sweep in sweeps.items():
                color, marker = style_map[model]

                if overlay:
                    for round_idx in rounds:
                        p_vals, y_vals, y_errs = sweep._metric_series_for_operator(
                            round_idx=round_idx,
                            operator=logical_label,
                            metric="state_prep_error",
                            allow_legacy_approx=allow_legacy_approx,
                        )
                        ax.errorbar(
                            p_vals,
                            y_vals,
                            yerr=y_errs,
                            fmt=f"-{marker}",
                            color=color,
                            capsize=4,
                            label=f"{model} (r={round_idx})",
                        )
                else:
                    round_idx = rounds[idx]
                    p_vals, y_vals, y_errs = sweep._metric_series_for_operator(
                        round_idx=round_idx,
                        operator=logical_label,
                        metric="state_prep_error",
                        allow_legacy_approx=allow_legacy_approx,
                    )
                    ax.errorbar(
                        p_vals,
                        y_vals,
                        yerr=y_errs,
                        fmt=f"-{marker}",
                        color=color,
                        capsize=4,
                        label=model,
                    )

            if log_x:
                ax.set_xscale("log")
            ax.set_xlabel("p", fontsize=14)
            ylabel = "State-prep error (1 - <X_L>) / 2"
            if legacy_models and allow_legacy_approx:
                ylabel += " [legacy approx]"
            ax.set_ylabel(ylabel, fontsize=14)

        plt.tight_layout(rect=[0, 0, 0.75, 1])
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bboxes = [ax.get_tightbbox(renderer) for ax in plot_axes]
        union_bbox = Bbox.union(bboxes)
        bb = union_bbox.transformed(fig.transFigure.inverted())
        lx = bb.x1 + 0.01
        ly = bb.y0 + bb.height / 2

        handles, lbls = plot_axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            lbls,
            loc="center left",
            bbox_to_anchor=(lx, ly),
            fontsize="small",
        )

        if save_path is not None:
            out_path = Path(save_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
        if "agg" not in plt.get_backend().lower():
            plt.show()
        plt.close(fig)
