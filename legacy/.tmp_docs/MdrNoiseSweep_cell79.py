"""
mdr_noise_sweep.py
────────────────────────────────────────────────────────────────────────────
Class for managing parameter sweeps, executing MDR simulations, and 
handling data persistence/visualisation.
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from matplotlib.transforms import Bbox
import math
import pandas as pd
import os
from typing import List, Dict, Tuple, Any, Union, Optional

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
        'IX':0, 'IY':1, 'IZ':2, 'XI':3, 'XX':4, 'XY':5, 'XZ':6,
        'YI':7, 'YX':8, 'YY':9, 'YZ':10, 'ZI':11, 'ZX':12, 'ZY':13, 'ZZ':14
    }
    single_params = {'p_x','p_y','p_z','g1_x','g1_y','g1_z'}

    # ─────────────────────────────────────────────────────────────────────
    # construction & io
    # ─────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        # Simulation Arguments (Optional if loading)
        code_stabilizers: Optional[List[str]] = None,
        toggles: Optional[List[str]] = None,
        measure_stabilizers: Optional[List[str]] = None,
        logical_operators: Optional[Dict[str, str]] = None,
        ancillas: int = 1,
        psi_circuit: Any = None,
        p_spam: float = 0.0,
        param_names: Union[str, List[str]] = [],
        param_values: Union[List[float], Dict[str, List[float]]] = [],
        round_list: List[int] = [1],
        shots: int = 1000,
        num_replicates: int = 30,
        split_2q: bool = True,
        # IO Arguments
        save_data_filename: Optional[str] = None,
        load_data_filename: Optional[str] = None
    ) -> None:
        """
        Configure and execute the full noise-parameter sweep.

        Can operate in two modes:
        1. **Simulation Mode**: Runs the sweep and optionally saves data.
        2. **Load Mode**: Loads existing data from a CSV for plotting (skips execution).

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
            If provided, saves the simulation results to this CSV path after running.
        load_data_filename : Optional[str]
            If provided, SKIPS simulation and loads data from this CSV. 
            Useful for plotting previously run jobs.
        """
        # MODE 1: LOAD EXISTING DATA
        if load_data_filename:
            self.load_results(load_data_filename)
            return

        # MODE 2: RUN NEW SIMULATION
        self.code_stabilizers    = code_stabilizers
        self.toggles             = toggles
        self.measure_stabilizers = measure_stabilizers
        self.logical_operators   = logical_operators
        self.ancillas            = ancillas
        self.psi_circuit         = psi_circuit
        self.p_spam              = p_spam
        self.round_list          = round_list
        self.shots               = shots
        self.num_replicates      = num_replicates
        self.split_2q            = split_2q

        if isinstance(param_names, str):
            self.param_names = [param_names]
        else:
            self.param_names = param_names

        # Handle Sync vs Async parameter sweeps
        if isinstance(param_values, list):
            self.sync              = True
            self.param_values_list = param_values[:]
            self.param_values_map  = {k: param_values[:] for k in self.param_names}
        else:
            self.sync              = False
            self.param_values_map  = param_values
            missing = set(self.param_names) - set(param_values)
            if missing:
                raise ValueError(f"Missing param_values for keys: {missing}")

        # Generate all parameter combinations to simulate
        if self.sync:
            self.param_combos = [tuple([p]*len(self.param_names)) for p in self.param_values_list]
        else:
            lists = [self.param_values_map[k] for k in self.param_names]
            self.param_combos = list(product(*lists))

        # Execute Simulation
        self.results, self.results_std = self._perform_sweep()

        # Optional: Save Data
        if save_data_filename:
            self.save_results(save_data_filename)


    def _perform_sweep(
        self
    ) -> Tuple[
        Dict[Tuple[float,...], Dict[int, Dict[str, float]]],
        Dict[Tuple[float,...], Dict[int, Dict[str, float]]]
    ]:
        """
        Internal: run MDRSimulation for each combo and gather both means and stddevs.
        
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
        all_means: Dict[Tuple[float,...], Dict[int, Dict[str,float]]] = {}
        all_stds:  Dict[Tuple[float,...], Dict[int, Dict[str,float]]] = {}

        # 1. Count Active Parameters by Type for Splitting Logic
        one_q_params_in_sweep = [n for n in self.param_names if n in self.single_params]
        two_q_params_in_sweep = [n for n in self.param_names if n not in self.single_params]
        
        num_1q = len(one_q_params_in_sweep)
        num_2q = len(two_q_params_in_sweep)

        for combo in self.param_combos:
            kwargs = {
                'stabilizers':   self.code_stabilizers,
                'toggles':       self.toggles,
                'ancillas':      self.ancillas,
                'p_spam':        self.p_spam,
                'p_x':0.0, 'p_y':0.0, 'p_z':0.0,
                'g1_x':0.0,'g1_y':0.0,'g1_z':0.0,
                'gate_noise_2q':[0.0]*15,
                'psi_circuit':   self.psi_circuit
            }
            
            for name, val in zip(self.param_names, combo):
                if name in self.single_params:
                    # FIX: Split 1Q noise if multiple components exist (e.g. Unbiased X/Y/Z)
                    # This prevents p_x=0.5, p_y=0.5, p_z=0.5 -> Sum=1.5 (Error)
                    if self.split_2q and num_1q > 0:
                        kwargs[name] = val / num_1q
                    else:
                        kwargs[name] = val
                else:
                    # Case B: 2-qubit parameter. Apply split value based on number of active 2Q params.
                    idx = self.two_qubit_index[name]
                    if self.split_2q and num_2q > 0:
                        kwargs['gate_noise_2q'][idx] = val / num_2q
                    else:
                        kwargs['gate_noise_2q'][idx] = val

            # --- STRICT SAFETY NORMALIZATION ---
            # Explicitly force sum <= 1.0 to handle floating point drift 
            # and Stim's string conversion rounding (e.g. 0.0666667 * 15 > 1).
            # We use a safety factor of 0.999 (0.1% margin) to effectively "round down"
            # the parameters enough to satisfy Stim's strict checks.
            
            # 1. Normalize 2-Qubit Noise Vector
            sum_2q = sum(kwargs['gate_noise_2q'])
            if sum_2q > 1.0 - 1e-9: # Check if close to or exceeding 1
                # Scale down by factor + margin
                scale_factor = (1.0 / sum_2q) * 0.999
                kwargs['gate_noise_2q'] = [p * scale_factor for p in kwargs['gate_noise_2q']]

            # 2. Normalize 1-Qubit Noise Vector
            sum_1q = kwargs['g1_x'] + kwargs['g1_y'] + kwargs['g1_z']
            if sum_1q > 1.0 - 1e-9:
                scale_factor = (1.0 / sum_1q) * 0.999
                kwargs['g1_x'] *= scale_factor
                kwargs['g1_y'] *= scale_factor
                kwargs['g1_z'] *= scale_factor

            mdr_inst = MDRCircuit(**kwargs)
            sim = MDRSimulation(
                mdr=mdr_inst,
                stabilizer_pauli_strings=self.measure_stabilizers,
                logical_pauli_strings   =self.logical_operators,
                shots_per_measurement   =self.shots,
                total_mdr_rounds        =max(self.round_list),
                num_replicates          =self.num_replicates
            )

            mean_dict = {r: {} for r in self.round_list}
            std_dict  = {r: {} for r in self.round_list}

            # Collect Stabilizer Stats
            for label in self.measure_stabilizers:
                stats = sim._stats_stabilizers[label]
                for r, ctr, sd in zip(stats['rounds'], stats['centers'], stats['stds']):
                    if r in mean_dict:
                        mean_dict[r][label] = ctr
                        std_dict [r][label] = sd

            # Collect Logical Stats
            for label in self.logical_operators:
                stats = sim._stats_logicals[label]
                for r, ctr, sd in zip(stats['rounds'], stats['centers'], stats['stds']):
                    if r in mean_dict:
                        mean_dict[r][label] = ctr
                        std_dict [r][label] = sd

            all_means[combo] = mean_dict
            all_stds [combo] = std_dict

        return all_means, all_stds

    # ─────────────────────────────────────────────────────────────────────
    # IO Methods
    # ─────────────────────────────────────────────────────────────────────
    def save_results(self, filename: str) -> None:
        """
        Flatten the results dictionary and save to CSV.
        
        The CSV structure will be:
        [param_1, param_2, ..., round, operator, mean, std]

        Args
        ----
        filename : str
            Path to save the CSV file.
        """
        rows = []
        for combo, round_dict in self.results.items():
            for r, ops_dict in round_dict.items():
                for label, mean_val in ops_dict.items():
                    std_val = self.results_std[combo][r][label]
                    
                    row = {}
                    # Add parameter columns (dynamic based on sweep config)
                    for pname, pval in zip(self.param_names, combo):
                        row[pname] = pval
                    
                    row["round"] = r
                    row["operator"] = label
                    row["mean"] = mean_val
                    row["std"] = std_val
                    rows.append(row)
        
        df = pd.DataFrame(rows)
        # Reorder columns to put params first for readability
        cols = self.param_names + ["round", "operator", "mean", "std"]
        df = df[cols]
        
        # Check if dir exists, if not create? (Optional, but good practice)
        dirpath = "simulation_results"
        os.makedirs(dirpath, exist_ok=True)
        filepath = os.path.join(dirpath, filename)
        df.to_csv(filepath, index=False)
        print(f"Data saved to: {os.path.abspath(filepath)}")


    def load_results(self, filename: str) -> None:
        """
        Load results from CSV and reconstruct attributes needed for plotting.
        
        This allows a new MdrNoiseSweep object to behave as if it had run
        the simulation, enabling the usage of plotting methods.

        Args
        ----
        filename : str
            Path to the CSV file to load.
        """
        filepath = f"simulation_results/{filename}"
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        df = pd.read_csv(filepath)
        
        # 1. Infer Param Names (columns that are not result metadata)
        reserved = {"round", "operator", "mean", "std"}
        self.param_names = [c for c in df.columns if c not in reserved]
        
        # 2. Reconstruct Results Dictionaries
        self.results = {}
        self.results_std = {}
        
        # Group by parameters to rebuild the 'combo' structure
        # df[self.param_names] gives a dataframe of just params.
        unique_combos_df = df[self.param_names].drop_duplicates()
        
        self.param_combos = []
        self.param_values_list = [] # Approximation for sync plotting
        
        # We need to rebuild: self.results[combo][round][label] = mean
        for _, row_combo in unique_combos_df.iterrows():
            # Convert series to tuple
            combo = tuple(row_combo.tolist())
            self.param_combos.append(combo)
            self.param_values_list.append(combo[0]) # Assuming sync for plot_multi x-axis
            
            # Filter main df for this combo
            mask = (df[self.param_names] == row_combo).all(axis=1)
            subset = df[mask]
            
            self.results[combo] = {}
            self.results_std[combo] = {}
            
            for _, r_row in subset.iterrows():
                rr = int(r_row["round"])
                lbl = r_row["operator"]
                mn = r_row["mean"]
                sd = r_row["std"]
                
                if rr not in self.results[combo]:
                    self.results[combo][rr] = {}
                    self.results_std[combo][rr] = {}
                
                self.results[combo][rr][lbl] = mn
                self.results_std[combo][rr][lbl] = sd

        # 3. Infer Other Attributes for Plotting Lists
        unique_ops = df["operator"].unique()
        self.measure_stabilizers = sorted([op for op in unique_ops if op.startswith("S_")])
        
        # Reconstruct logicals as a dict (values are placeholders, keys needed for plotting)
        log_ops = sorted([op for op in unique_ops if op.startswith("Logical")])
        self.logical_operators = {op: "Unknown_Pauli" for op in log_ops}
        
        # 4. Set Sync/Async (heuristic)
        # For plotting, if we have >1 param name but they change together in the file, 
        # we treat it as sync.
        self.sync = True 
        self.param_values_map = {p: unique_combos_df[p].tolist() for p in self.param_names}
        
        print(f"Loaded {len(self.param_combos)} parameter combinations from {filepath}")

    # ─────────────────────────────────────────────────────────────────────
    # visualization
    # ─────────────────────────────────────────────────────────────────────
    def plot_expectations_vs_param(
        self,
        category: str,
        rounds: List[int],
        subset: Optional[List[str]] = None,
        overlay: bool = False,
        cols: int = 2,
        figsize: Tuple[int,int] = (12,8),
        save_path: Optional[str] = None,
        log_x: bool = False
    ) -> None:
        """
        Plot the average observable ⟨|O|⟩ versus noise parameter p.

        Args
        ----
        category : {'stabilizer', 'logical'}
            Which operators to plot.
        rounds : List[int]
            MDR round indices to include.
        subset : Optional[List[str]]
            If provided, restrict to these operator labels.
        overlay : bool
            If True, overlay all rounds on a single axis.
        cols : int
            Number of columns for grid when overlay=False.
        figsize : Tuple[int,int]
            Figure size in inches.
        save_path : Optional[str]
            Path to save the figure (dpi=300).
        log_x : bool
            If True, set x-axis to logarithmic scale.

        Raises
        ------
        ValueError
            If category invalid or subset contains unknown labels.

        Returns
        -------
        None
        """
        if category == 'stabilizer':
            labels = list(self.measure_stabilizers)
        elif category == 'logical':
            labels = list(self.logical_operators.keys())
        else:
            raise ValueError("category must be 'stabilizer' or 'logical'")

        if subset:
            missing = set(subset) - set(labels)
            if missing:
                raise ValueError(f"Unknown labels: {missing}")
            labels = subset

        p_vals = (self.param_values_list if self.sync
                  else self.param_values_map[self.param_names[0]])
        combos = [(p,)*len(self.param_names) for p in p_vals]
        series = [(lbl, r) for lbl in labels for r in rounds]

        colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
        markers = ['o','s','^','v','<','>','8','p','P','*','h','H',
                   '+','x','X','D','d','|','_','.']
        ncol, nmark = len(colours), len(markers)

        if overlay:
            fig, ax = plt.subplots(figsize=figsize)
            axes = [ax]
        else:
            rows = math.ceil(len(rounds)/cols)
            fig, grid = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
            axes = grid.flatten()

        for i, (ax_idx, ax) in enumerate(zip(range(len(axes)), axes)):
            if not overlay and ax_idx < len(rounds):
                rr = rounds[ax_idx]
                ax.set_title(f"Round {rr}")
            ax.grid(True)

            for idx, (lbl, r) in enumerate(series):
                if overlay or r == rounds[ax_idx]:
                    col = colours[idx % ncol]
                    mark = markers[(idx//ncol) % nmark]
                    ys = [self.results[c][r][lbl] for c in combos]
                    ax.scatter(p_vals, ys, color=col, marker=mark, s=40,
                               label=f"{lbl} (r={r})")

            if log_x:
                ax.set_xscale('log')
            ax.set_xlabel('p')
            ax.set_ylabel('⟨|O|⟩')
            ax.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize='small')

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.tight_layout()
        plt.show()


    def plot_error_rates_vs_param(
        self,
        category: str,
        rounds: List[int],
        subset: Optional[List[str]] = None,
        overlay: bool = False,
        cols: int = 2,
        figsize: Tuple[int,int] = (12,8),
        save_path: Optional[str] = None,
        log_x: bool = False
    ) -> None:
        """
        Plot error rate = 1 − ⟨|O|⟩ versus noise parameter p.

        Args
        ----
        category : {'stabilizer', 'logical'}
            Which operators to plot.
        rounds : List[int]
            MDR round indices to include.
        subset : Optional[List[str]]
            If provided, restrict to these operator labels.
        overlay : bool
            If True, overlay all rounds on a single axis.
        cols : int
            Number of columns for grid when overlay=False.
        figsize : Tuple[int,int]
            Figure size in inches.
        save_path : Optional[str]
            Path to save the figure (dpi=300).
        log_x : bool
            If True, set x-axis to logarithmic scale.

        Raises
        ------
        ValueError
            If category invalid or subset contains unknown labels.

        Returns
        -------
        None
        """
        if category == 'stabilizer':
            labels = list(self.measure_stabilizers)
        elif category == 'logical':
            labels = list(self.logical_operators.keys())
        else:
            raise ValueError("category must be 'stabilizer' or 'logical'")

        if subset:
            missing = set(subset) - set(labels)
            if missing:
                raise ValueError(f"Unknown labels: {missing}")
            labels = subset

        p_vals = (self.param_values_list if self.sync
                  else self.param_values_map[self.param_names[0]])
        combos = [(p,)*len(self.param_names) for p in p_vals]
        series = [(lbl, r) for lbl in labels for r in rounds]

        colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
        markers = ['o','s','^','v','<','>','8','p','P','*','h','H',
                   '+','x','X','D','d','|','_','.']
        ncol, nmark = len(colours), len(markers)

        if overlay:
            fig, ax = plt.subplots(figsize=figsize)
            axes = [ax]
        else:
            rows = math.ceil(len(rounds)/cols)
            fig, grid = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
            axes = grid.flatten()

        for i, (ax_idx, ax) in enumerate(zip(range(len(axes)), axes)):
            if not overlay and ax_idx < len(rounds):
                rr = rounds[ax_idx]
                ax.set_title(f"Round {rr}")
            ax.grid(True)

            for idx, (lbl, r) in enumerate(series):
                if overlay or r == rounds[ax_idx]:
                    col = colours[idx % ncol]
                    mark = markers[(idx//ncol) % nmark]
                    errs = [1 - self.results[c][r][lbl] for c in combos]
                    ax.scatter(p_vals, errs, color=col, marker=mark, s=40,
                               label=f"{lbl} (r={r})")

            # dotted y=x reference
            ax.plot(p_vals, p_vals, linestyle=':', color='gray')

            if log_x:
                ax.set_xscale('log')
            ax.set_xlabel('p', fontsize=16)
            ax.set_ylabel('Error (1 - ⟨|O|⟩)', fontsize=16)
            ax.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize='small')

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_multi(
        sweeps: Dict[str, 'MdrNoiseSweep'],
        category: str,
        rounds: List[int],
        subset: Optional[List[str]] = None,
        overlay: bool = False,
        log_x: bool = False,
        fit: bool = False,
        figsize: Tuple[int,int] = (15,6),
        save_path: Optional[str] = None
    ) -> None:
        """
        Overlay ⟨|O|⟩ vs p for multiple sweeps or grid by round,
        with ±1σ errorbars and optional straight-line fits.

        Args:
            sweeps: Dict mapping legend → MdrNoiseSweep instance.
            category: 'stabilizer' or 'logical'.
            rounds: List of MDR rounds to plot.
            subset: If provided, restrict to these operator labels.
            overlay: If True, overlay all rounds on one axis.
            log_x: If True, set x-axis to logarithmic scale.
            fit: If True, overplot solid linear fits (excluding p=0).
            figsize: Figure size (width, height) in inches.
            save_path: If given, save figure (dpi=300) before showing.

        Raises:
            ValueError: If sweeps is empty or category/subset invalid.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.transforms import Bbox
        import math

        if not sweeps:
            raise ValueError("No sweeps provided")

        # Validate category & subset
        first = next(iter(sweeps.values()))
        if category == 'stabilizer':
            labels = list(first.measure_stabilizers)
        elif category == 'logical':
            labels = list(first.logical_operators.keys())
        else:
            raise ValueError("category must be 'stabilizer' or 'logical'")
        if subset:
            missing = set(subset) - set(labels)
            if missing:
                raise ValueError(f"Unknown labels: {missing}")
            labels = subset

        # Prepare colours and markers
        colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
        markers = ['o','s','^','v','<','>','8','p','P','*','h','H',
                   '+','x','X','D','d','|','_','.']

        # Build style_map keys
        if overlay:
            style_keys = [(m, lbl, r)
                          for m in sweeps for lbl in labels for r in rounds]
        else:
            style_keys = [(m, lbl)
                          for m in sweeps for lbl in labels]
        style_map = {
            key: (
                colours[i % len(colours)],
                markers[(i // len(colours)) % len(markers)]
            )
            for i, key in enumerate(style_keys)
        }

        # Create figure and axes
        if overlay:
            fig, ax0 = plt.subplots(figsize=figsize)
            axes = [ax0]
        else:
            n = len(rounds)
            cols = min(3, n)
            rows = math.ceil(n/cols)
            fig, grid = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
            axes = grid.flatten()

        # Plot data
        for idx, ax in enumerate(axes if not overlay else [axes[0]]):
            if not overlay:
                rr = rounds[idx]
                ax.set_title(f"Round {rr}", fontsize=16)
            ax.grid(True)

            for model, sweep in sweeps.items():
                # Extract p-values & combos
                if sweep.sync:
                    p_vals = sweep.param_values_list
                    combos = [(p,)*len(sweep.param_names) for p in p_vals]
                else:
                    pn     = sweep.param_names[0]
                    p_vals = sweep.param_values_map[pn]
                    combos = [(p,) for p in p_vals]

                # Choose overlay vs grid logic
                if overlay:
                    for lbl in labels:
                        for rr in rounds:
                            means = np.array([sweep.results[c][rr][lbl]
                                              for c in combos])
                            stds  = np.array([sweep.results_std[c][rr][lbl]
                                              for c in combos])
                            col, mark = style_map[(model, lbl, rr)]
                            # Plot errorbars
                            ax.errorbar(
                                p_vals, means,
                                yerr=stds,
                                fmt=mark,
                                color=col,
                                capsize=4,
                                elinewidth=1.5,
                                markeredgewidth=1.5,
                                zorder=3,
                                label=f"{model}: {lbl} (r={rr})"
                            )
                            # Plot fit if requested
                            if fit:
                                mask = np.array(p_vals) > 0
                                if mask.sum() >= 2:
                                    x_fit = np.array(p_vals)[mask]
                                    y_fit = means[mask]
                                    try:
                                        A, B = np.polyfit(x_fit, y_fit, 1)
                                        x_line = np.linspace(
                                            x_fit.min(), x_fit.max(), 200)
                                        y_line = A * x_line + B
                                        ax.plot(
                                            x_line, y_line,
                                            '-', color=col, linewidth=1.5,
                                            zorder=4
                                        )
                                    except np.linalg.LinAlgError:
                                        pass
                else:
                    rr = rounds[idx]
                    for lbl in labels:
                        means = np.array([sweep.results[c][rr][lbl]
                                          for c in combos])
                        stds  = np.array([sweep.results_std[c][rr][lbl]
                                          for c in combos])
                        col, mark = style_map[(model, lbl)]
                        ax.errorbar(
                            p_vals, means,
                            yerr=stds,
                            fmt=mark,
                            color=col,
                            capsize=4,
                            elinewidth=1.5,
                            markeredgewidth=1.5,
                            zorder=3,
                            label=f"{model}: {lbl}"
                        )
                        if fit:
                            mask = np.array(p_vals) > 0
                            if mask.sum() >= 2:
                                x_fit = np.array(p_vals)[mask]
                                y_fit = means[mask]
                                try:
                                    A, B = np.polyfit(x_fit, y_fit, 1)
                                    x_line = np.linspace(
                                        x_fit.min(), x_fit.max(), 200)
                                    y_line = A * x_line + B
                                    ax.plot(
                                        x_line, y_line,
                                        '-', color=col, linewidth=1.5,
                                        zorder=4
                                    )
                                except np.linalg.LinAlgError:
                                    pass

            if log_x:
                ax.set_xscale('log')
            ax.set_xlabel('p', fontsize=16)
            ax.set_ylabel('⟨|O|⟩', fontsize=16)

        # Dynamic legend placement
        plt.tight_layout(rect=[0,0,0.75,1])
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bboxes = [ax.get_tightbbox(renderer) for ax in axes]
        union_bbox = Bbox.union(bboxes)
        bb = union_bbox.transformed(fig.transFigure.inverted())
        pad = 0.01
        lx = bb.x1 + pad
        ly = bb.y0 + bb.height/2
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc='center left',
            bbox_to_anchor=(lx, ly),
            fontsize=16
        )

        # Save and show
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


    @staticmethod
    def plot_error_multi(
        sweeps: Dict[str, 'MdrNoiseSweep'],
        category: str,
        rounds: List[int],
        subset: Optional[List[str]] = None,
        overlay: bool = False,
        log_x: bool = False,
        figsize: Tuple[int,int] = (15,6),
        save_path: Optional[str] = None
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
        if category == 'stabilizer':
            labels = list(first.measure_stabilizers)
        elif category == 'logical':
            labels = list(first.logical_operators.keys())
        else:
            raise ValueError("category must be 'stabilizer' or 'logical'")
        if subset:
            missing = set(subset) - set(labels)
            if missing:
                raise ValueError(f"Unknown labels: {missing}")
            labels = subset

        colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
        markers = ['o','s','^','v','<','>','8','p','P','*','h','H',
                   '+','x','X','D','d','|','_','.']
        style_keys = [(m,op) for m in sweeps for op in labels]
        style_map  = {
            (m,op): (colours[i % len(colours)],
                     markers[(i // len(colours)) % len(markers)])
            for i,(m,op) in enumerate(style_keys)
        }

        if overlay:
            fig, ax0 = plt.subplots(figsize=figsize)
            axes = [ax0]
        else:
            n = len(rounds)
            cols = min(3, n)
            rows = math.ceil(n/cols)
            fig, grid = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
            axes = grid.flatten()

        for idx, ax in enumerate(axes if not overlay else [axes[0]]):
            if not overlay:
                r = rounds[idx]
                ax.set_title(f"Round {r}")
            ax.grid(True)

            for model, sweep in sweeps.items():
                if sweep.sync:
                    p_vals = np.array(sweep.param_values_list)
                    combos = [(p,)*len(sweep.param_names) for p in p_vals]
                else:
                    pn     = sweep.param_names[0]
                    p_vals = np.array(sweep.param_values_map[pn])
                    combos = [(p,) for p in p_vals]

                if overlay:
                    for op in labels:
                        for r in rounds:
                            means = np.array([sweep.results[c][r][op] for c in combos])
                            stds  = np.array([sweep.results_std[c][r][op] for c in combos])
                            col, mark = style_map[(model,op)]
                            ax.errorbar(
                                p_vals, 1-means,
                                yerr=stds,
                                fmt=f'-{mark}',
                                color=col,
                                capsize=4,
                                label=f"{model}: {op} (r={r})"
                            )
                else:
                    r = rounds[idx]
                    for op in labels:
                        means = np.array([sweep.results[c][r][op] for c in combos])
                        stds  = np.array([sweep.results_std[c][r][op] for c in combos])
                        col, mark = style_map[(model,op)]
                        ax.errorbar(
                            p_vals, 1-means,
                            yerr=stds,
                            fmt=f'-{mark}',
                            color=col,
                            capsize=4,
                            label=f"{model}: {op}"
                        )

            if log_x:
                ax.set_xscale('log')
            ax.set_xlabel('p', fontsize=16)
            ax.set_ylabel('Error rate (1 - |⟨O⟩|)', fontsize=16)

        plt.tight_layout(rect=[0,0,0.75,1])
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bboxes = [ax.get_tightbbox(renderer) for ax in axes]
        union_bbox = Bbox.union(bboxes)
        bb = union_bbox.transformed(fig.transFigure.inverted())
        pad = 0.01
        lx = bb.x1 + pad
        ly = bb.y0 + bb.height/2

        handles, lbls = axes[0].get_legend_handles_labels()
        fig.legend(handles, lbls,
                   loc='center left',
                   bbox_to_anchor=(lx, ly),
                   fontsize='small')

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def table_round_delta(
        self,
        rounds: Tuple[int, int],
        *,
        category: str = 'logical',
        subset: Optional[List[str]] = None,
        precision: int = 4,
        return_df: bool = False
    ):
        """
        Compare replicate‑mean fidelity between two MDR rounds (r2 − r1) and
        print a table of differences for every swept‑parameter point.

        Args
        ----
        rounds : Tuple[int, int]
            (r1, r2) MDR round indices to compare (both must be in
            ``self.round_list`` and must differ).
        category : {'stabilizer', 'logical'}, default 'logical'
            Which operator set to analyse.
        subset : list[str] | None, default None
            Restrict to these operator labels if provided.
        precision : int, default 4
            Number of decimal places when printing the table.
        return_df : bool, default False
            If True, also return the :class:`pandas.DataFrame`.

        Returns
        -------
        pandas.DataFrame | None
            A tidy table whose columns are
            * ``param``   – the swept noise parameter (or tuple of parameters)
            * ``Δ<label>`` – change μ(r₂) − μ(r₁) for each operator
            * ``Δavg``    – arithmetic mean of all Δ columns.
            Returned only when *return_df* is True.
        """
        r1, r2 = rounds
        if r1 == r2:
            raise ValueError("rounds must be two *different* indices")
        if r1 not in self.round_list or r2 not in self.round_list:
            raise ValueError("rounds must be in self.round_list")

        if category == 'stabilizer':
            labels = list(self.measure_stabilizers)
        elif category == 'logical':
            labels = list(self.logical_operators.keys())
        else:
            raise ValueError("category must be 'stabilizer' or 'logical'")

        if subset:
            missing = set(subset) - set(labels)
            if missing:
                raise ValueError(f"Unknown labels: {missing}")
            labels = subset

        rows = []
        for combo, res in self.results.items():
            row: Dict[str, Any] = {}
            # If the sweep is synchronous, combo is like (p,) – show single val
            row['param'] = combo[0] if self.sync else combo
            for lbl in labels:
                mu1 = res[r1][lbl]
                mu2 = res[r2][lbl]
                row[f'Δ{lbl}'] = mu2 - mu1
            row['Δavg'] = float(np.mean([row[f'Δ{lbl}'] for lbl in labels]))
            rows.append(row)

        df = pd.DataFrame(rows)
        pd.set_option('display.precision', precision)
        print(df.to_string(index=False))

        if return_df:
            return df