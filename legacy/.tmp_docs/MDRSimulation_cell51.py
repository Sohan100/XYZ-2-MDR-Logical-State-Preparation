"""
mdr_simulation.py
────────────────────────────────────────────────────────────────────────
Encapsulates measurement-based decoding & recovery (MDR) utilities in the
MDRSimulation class, with precomputed violin-distribution overlays of
absolute mean-estimates and detailed uncertainty reporting.
"""

import stim                                 # Stim: quantum circuit simulator
import numpy as np                         # NumPy: numerical computing
import matplotlib.pyplot as plt            # Matplotlib: plotting library
from typing import Callable, List, Tuple, Dict, Mapping, Union, Optional

class MDRSimulation:
    """
    Tools for simulating Measurement-based Decoding & Recovery (MDR).

    This class precomputes, for each specified stabilizer and logical
    operator, the distribution of absolute mean-estimates obtained by
    repeating an N-shot experiment (`shots_per_measurement`) `num_replicates`
    times per MDR round (0…`total_mdr_rounds`), and caches both the full
    replicate lists (for violin plots) and summary statistics (mean ±
    sample-stddev). It also computes an overall average expectation per
    operator (mean of the per-round means).

    Once initialized, `plot_group()` and `results()` render or print the
    precomputed data instantly. For on-demand, single-run curves,
    `plot_expectation()` and `plot_many_expectations()` sample fresh circuits
    and can overlay violins of the replicate-mean distributions if desired.

    Attributes:
        prepare_circuit_function (Callable[[], stim.Circuit]):
            Factory returning a fresh stim.Circuit to prepare the initial state.
        mdr_circuit (stim.Circuit):
            stim.Circuit template implementing one round of MDR.
        stabilizer_pauli_strings (List[str]):
            Sparse Pauli-string specifications for stabilizer measurements.
        logical_pauli_strings (Dict[str, str]):
            Mapping from logical operator labels to sparse Pauli specs.
        shots_per_measurement (int):
            Number of Monte-Carlo samples per expectation estimate.
        total_mdr_rounds (int):
            Maximum number of MDR rounds to simulate (inclusive of round 0).
        num_replicates (int):
            Number of independent N-shot repeats per round for violin
            distributions.
        last_results_storage (Dict[str,Dict[str,Tuple[List[int],List[float]]]]):
            Backward-compatible single-run mean-value curves.
        _replicate_means_stabilizers (Dict[str,Dict[int,List[float]]]):
            Raw lists of absolute replicate-mean estimates per stabilizer &
            round.
        _replicate_means_logicals (Dict[str,Dict[int,List[float]]]):
            Raw lists of absolute replicate-mean estimates per logical & round.
        _stats_stabilizers (Dict[str,Dict[str,List[float]]]):
            Summary statistics for each stabilizer:
            `'rounds'`, `'centers'` (means), and `'stds'` (sample-stddevs).
        _stats_logicals (Dict[str,Dict[str,List[float]]]):
            Summary statistics for each logical operator.
        _avg_stabilizers (Dict[str, float]):
            Overall average expectation per stabilizer (mean of `'centers'`).
        _avg_logicals (Dict[str, float]):
            Overall average expectation per logical operator.

    Methods:
        spec_to_measurement_ops(pauli_spec: str) -> List[Tuple[str,int]]:
            Convert a sparse Pauli string (e.g. "X0 Z1") into a list of
            stim measurement operations [(gate, qubit_index), …].

        compute_parity_expectation(circuit, measurement_ops) -> float:
            Append the given measurement operations to `circuit`, sample
            `shots_per_measurement` bit-strings, compute the parity product
            of the measured bits, map to ±1 eigenvalues, and return the
            absolute mean |⟨O⟩|.

        calculate_replicated_means_vs_rounds(pauli_spec: str)
            -> Dict[int, List[float]]:
            For each MDR round r, build the circuit with r rounds, then run
            `num_replicates` independent N-shot experiments. Return a mapping
            {r → [|mean₁|, |mean₂|, …]} of all replicate absolute-mean values.

        calculate_expectation_vs_rounds(pauli_spec: str)
            -> (List[int], List[float]):
            For each MDR round r, build the circuit and perform a single
            N-shot experiment. Return two lists: the round indices and
            the corresponding |⟨O⟩| estimates.

        calculate_many_expectations(spec_map: Mapping[str,str])
            -> Dict[str, (List[int],List[float])]:
            Apply `calculate_expectation_vs_rounds` to each Pauli spec in
            `spec_map`, returning a dict label → (rounds, means).

        calculate_group_data() -> Dict[str,Dict[str,(List[int],List[float])]]:
            Return the precomputed single-run mean-value curves stored in
            `last_results_storage` (for backward compatibility).

        plot_data(data_map, title, x_label, y_label, **kwargs) -> None:
            Generic renderer for precomputed violin distributions:
              • Plots only markers (no connecting lines) for the summary means.
              • Overlays violins of the replicate-mean distributions, with
                mean & extrema lines, all colored to match each marker.
              • Optionally scatters every replicate point inside its violin.
              • Handles saving to disk if `save_path` is provided.

        plot_expectation(pauli_specification=None, save_path=None, **kw) -> None:
            Single-operator plotting: if no spec is given, plots all
            stabilizers; if a string is given, plots that one; if a list
            is given, plots each. Delegates to `plot_data` (optionally adding
            violins) but uses fresh single-run sampling, not the precomputed
            cache.

        plot_many_expectations(pauli_specs, save_path=None, **kw) -> None:
            Multi-operator plotting: takes either a list of Pauli strings or a
            mapping label→spec, performs fresh single-run sampling for each,
            then delegates to `plot_data`.

        plot_group(save_path_stabilizers=None, save_path_logicals=None, **kw)
            -> None:
            Render two figures—one for all stabilizers, one for all logicals—
            using the precomputed violin distributions and summary means.
            Saves each figure if a corresponding `save_path_*` is provided.

        results() -> None:
            Print tabular, precomputed statistics for each operator:
              Round | Mean | StdDev
            followed by the overall average expectation for that operator.
    """

    def __init__(
        self,
        mdr: MDRCircuit,
        stabilizer_pauli_strings: List[str],
        logical_pauli_strings: Dict[str, str],
        shots_per_measurement: int = 1000,
        total_mdr_rounds: int = 10,
        num_replicates: int = 30
    ) -> None:
        """
        Initialize MDRSimulation and precompute all distributions, stats, and
        averages.

        Args:
            mdr (MDRCircuit):
                MDRCircuit instance containing the circuit preparation
            stabilizer_pauli_strings (List[str]):
                List of sparse stabilizer Pauli specifications.
            logical_pauli_strings (Dict[str, str]):
                Dictionary mapping logical labels to Pauli specifications.
            shots_per_measurement (int, optional):
                Number of shots per expectation estimate. Default is 1000.
            total_mdr_rounds (int, optional):
                Number of MDR rounds (inclusive of 0). Default is 10.
            num_replicates (int, optional):
                Number of repeats per round for violin distributions. Default 30.

        Returns:
            None
        """
        # Store MDR Class
        self.mdr = mdr
        # Store the circuit preparation function
        self.prepare_circuit_function = mdr.psi
        # Store the MDR circuit template
        self.mdr_circuit = mdr.build(include_psi=False)
        # Store SPAM noise parameter
        self.p_spam = mdr.p_spam
        # Store the list of stabilizer Pauli strings
        self.stabilizer_pauli_strings = stabilizer_pauli_strings
        # Store the dictionary of logical Pauli strings
        self.logical_pauli_strings = logical_pauli_strings
        # Store the number of shots per measurement
        self.shots_per_measurement = shots_per_measurement
        # Store the total number of MDR rounds
        self.total_mdr_rounds = total_mdr_rounds
        # Store the number of replicates per round
        self.num_replicates = num_replicates

        # Precompute replicate-means for stabilizers
        self._replicate_means_stabilizers: \
            Dict[str, Dict[int, List[float]]] ={}
        for spec in self.stabilizer_pauli_strings:
            # For each stabilizer, compute replicate means vs rounds
            self._replicate_means_stabilizers[spec] = \
                self.calculate_replicated_means_vs_rounds(spec)

        # Precompute replicate-means for logicals
        self._replicate_means_logicals: Dict[str, Dict[int, List[float]]] = {}
        for label, spec in self.logical_pauli_strings.items():
            # For each logical, compute replicate means vs rounds
            self._replicate_means_logicals[label] = \
                self.calculate_replicated_means_vs_rounds(spec)

        # Summarize stats and compute overall averages for stabilizers
        self._stats_stabilizers: Dict[str, Dict[str, List[float]]] = {}
        self._avg_stabilizers: Dict[str, float] = {}
        for spec, dist_map in self._replicate_means_stabilizers.items():
            # Sort rounds for consistent plotting
            rounds = sorted(dist_map)
            # Compute mean for each round
            centers = [float(np.mean(dist_map[r])) for r in rounds]
            # Compute stddev for each round
            stds    = [float(np.std(dist_map[r], ddof=1)) for r in rounds]
            # Store stats for this stabilizer
            self._stats_stabilizers[spec] = {
                'rounds': rounds, 'centers': centers, 'stds': stds
            }
            # Store overall average for this stabilizer
            self._avg_stabilizers[spec] = float(np.mean(centers))

        # Summarize stats and compute overall averages for logicals
        self._stats_logicals: Dict[str, Dict[str, List[float]]] = {}
        self._avg_logicals: Dict[str, float] = {}
        for label, dist_map in self._replicate_means_logicals.items():
            # Sort rounds for consistent plotting
            rounds = sorted(dist_map)
            # Compute mean for each round
            centers = [float(np.mean(dist_map[r])) for r in rounds]
            # Compute stddev for each round
            stds    = [float(np.std(dist_map[r], ddof=1)) for r in rounds]
            # Store stats for this logical
            self._stats_logicals[label] = {
                'rounds': rounds, 'centers': centers, 'stds': stds
            }
            # Store overall average for this logical
            self._avg_logicals[label] = float(np.mean(centers))

        # Backward-compatible single-run curves
        self.last_results_storage = {
            'stabilizers': {
                spec: (
                    self._stats_stabilizers[spec]['rounds'],
                    self._stats_stabilizers[spec]['centers']
                )
                for spec in self.stabilizer_pauli_strings
            },
            'logicals': {
                label: (
                    self._stats_logicals[label]['rounds'],
                    self._stats_logicals[label]['centers']
                )
                for label in self.logical_pauli_strings
            }
        }

    @staticmethod
    def spec_to_measurement_ops(pauli_specification: str) -> List[Tuple[str, int]]:
        """
        Convert a sparse Pauli string (e.g. "X0 Z1") into Stim measurement ops.

        Args:
            pauli_specification (str): Sparse Pauli string, e.g., "X0 Z1".

        Returns:
            List[Tuple[str, int]]: List of tuples (gate, qubit index), e.g.,
            [("MX", 0), ("MZ", 1)].
        """
        ops: List[Tuple[str, int]] = []
        for term in pauli_specification.split():
            # Extract Pauli letter and qubit index
            p = term[0].upper()
            q = int(term[1:])
            # Map Pauli letter to Stim measurement gate
            gate_map = {'X': 'MX', 'Y': 'MY', 'Z': 'MZ'}
            if p not in gate_map:
                # Raise error if invalid Pauli letter
                raise ValueError(f"Invalid Pauli letter: {p}")
            # Append (gate, qubit) tuple to ops list
            ops.append((gate_map[p], q))
        return ops

    def compute_parity_expectation(
        self,
        circuit: stim.Circuit,
        measurement_ops: List[Tuple[str, int]]
    ) -> float:
        """
        Append measurement_ops to the circuit, sample, and compute |⟨O⟩|.

        Args:
            circuit (stim.Circuit): Circuit to which measurement ops are added.
            measurement_ops (List[Tuple[str, int]]): List of (gate, qubit) to
            measure.

        Returns:
            float: Absolute mean of the ±1 parity outcomes.
        """
        for gate, qubit in measurement_ops:
            #add SPAM noise if applicable
            if self.p_spam > 0:
                circuit += \
                    stim.Circuit(f"PAULI_CHANNEL_1({self.p_spam},0,0) {qubit}")
            # Append each measurement operation to the circuit
            circuit.append_operation(gate, qubit)
        #display(SVG(str(circuit.diagram('timeline-svg'))))

        # Compile the circuit into a sampler
        sampler = circuit.compile_sampler()
        # Sample bit-strings from the circuit
        samples = sampler.sample(shots=self.shots_per_measurement,
                                 bit_packed=False)
        # Select columns corresponding to the measured qubits
        cols = np.arange(-len(measurement_ops), 0)
        # Compute parity for each shot
        parity = np.sum(samples[:, cols], axis=1) % 2
        # Map parity to ±1 eigenvalues
        eigen = 1 - 2 * parity
        # Return the absolute mean of eigenvalues
        return abs(np.mean(eigen))

    def calculate_replicated_means_vs_rounds(
        self,
        pauli_specification: str
    ) -> Dict[int, List[float]]:
        """
        For each round r, repeat the N-shot experiment `num_replicates` times
        and collect the absolute mean eigenvalue of each replicate.

        Args:
            pauli_specification (str): Sparse Pauli string to measure.

        Returns:
            Dict[int, List[float]]: Mapping from round index to list of replicate
            means.
        """
        # Convert Pauli string to measurement operations
        measurement_ops  = self.spec_to_measurement_ops(pauli_specification)
        # Initialize dictionary to store replicate means per round
        dist_map: Dict[int, List[float]] = {}
        for r in range(self.total_mdr_rounds + 1):
            # Prepare the base circuit for this round
            base = self.prepare_circuit_function()
            if r > 0:
                # Add r rounds of MDR circuit if r > 0
                base += self.mdr_circuit * r
            replicate_means: List[float] = []

            for _ in range(self.num_replicates):
                # Copy the base circuit for each replicate
                circ = base.copy()
                absolute_eigen = \
                    self.compute_parity_expectation(circ, measurement_ops)
                # Append absolute mean to replicate_means
                replicate_means.append(absolute_eigen)
            # Store replicate means for this round
            dist_map[r] = replicate_means
        return dist_map

    def calculate_expectation_vs_rounds(
        self,
        pauli_specification: str
    ) -> Tuple[List[int], List[float]]:
        """
        Single-run N-shot estimate of |⟨O⟩| vs round (no replicates).

        Args:
            pauli_specification (str): Sparse Pauli string to measure.

        Returns:
            Tuple[List[int], List[float]]: (List of rounds, List of mean values
            per round)
        """
        # Convert Pauli string to measurement operations
        ops = self.spec_to_measurement_ops(pauli_specification)
        # Initialize lists to store rounds and means
        rounds, means = [], []
        for r in range(self.total_mdr_rounds + 1):
            # Prepare the circuit for this round
            circ = self.prepare_circuit_function()
            if r > 0:
                # Add r rounds of MDR circuit if r > 0
                circ += self.mdr_circuit * r
            # Compute the parity expectation
            m = self.compute_parity_expectation(circ, ops)
            # Append round index and mean value
            rounds.append(r)
            means.append(m)
        return rounds, means

    def calculate_many_expectations(
        self,
        spec_map: Mapping[str, str]
    ) -> Dict[str, Tuple[List[int], List[float]]]:
        """
        Compute single-run curves for multiple Pauli specs.

        Args:
            spec_map (Mapping[str, str]): Mapping from label to Pauli spec.

        Returns:
            Dict[str, Tuple[List[int], List[float]]]: Mapping label → (rounds,
            means).
        """
        # Initialize output dictionary
        out: Dict[str, Tuple[List[int], List[float]]] = {}
        for label, spec in spec_map.items():
            # Compute expectation vs rounds for each Pauli spec
            out[label] = self.calculate_expectation_vs_rounds(spec)
        return out

    def calculate_group_data(self) -> Dict[str, Dict[str, Tuple[List[int],
        List[float]]]]:
        """
        Return precomputed single-run mean-value data (backward-compat).

        Returns:
            Dict[str, Dict[str, Tuple[List[int], List[float]]]]:
                Dictionary with keys 'stabilizers' and 'logicals', each mapping
                to a dictionary of label → (rounds, means).
        """
        # Return the last results storage dictionary
        return self.last_results_storage


    def plot_data(
        self,
        data_map: Dict[str, Tuple[List[int], List[float]]],
        title: str,
        x_label: str,
        y_label: str,
        *,
        legend_location: str = 'upper left',
        bbox_to_anchor: Tuple[float, float] = (1.02, 1),
        save_path: str = None,
        show_violin: bool = True,
        violin_kwargs: Dict = None,
        **legend_kwargs
    ) -> None:
        """
        Generic plotting of expectation-vs-rounds curves with unique Tab10
        colors and block-wise marker changes every 10 curves.

        This function visualizes the evolution of expectation values (e.g.,
        |⟨O⟩|) across MDR rounds for multiple operators. Each operator is
        assigned a distinct color from the Tab10 palette, and markers are
        cycled every 10 curves for clarity. If precomputed replicate
        distributions are available, violin plots are overlaid to show the
        spread of replicate means, with mean and extrema lines colored to
        match the corresponding marker. Individual replicate points can also
        be scattered within each violin for additional detail. The function
        supports saving the resulting figure to disk and customizing legend
        placement and violin plot appearance.

        Args:
            data_map (Dict[str, Tuple[List[int], List[float]]]):
            Mapping from operator label to (round indices, mean values).
            title (str): Title for the plot.
            x_label (str): Label for the x-axis.
            y_label (str): Label for the y-axis.
            legend_location (str, optional): Location of the legend. Default is
            'upper left'.
            bbox_to_anchor (Tuple[float, float], optional): Anchor point for
            the legend. Default is (1.02, 1).
            save_path (str, optional): If provided, saves the figure to this
            path.
            show_violin (bool, optional): Whether to overlay violin plots for
            replicate distributions. Default is True.
            violin_kwargs (Dict, optional): Additional keyword arguments for
            the violin plot.
            **legend_kwargs: Additional keyword arguments for the legend.

        Returns:
            None
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Tab10 has 10 distinct colours
        colours = plt.rcParams['axes.prop_cycle'].by_key()['color'][::2]
        # extra markers for blocks beyond the first 10
        extra_markers = ['s','^','D','v','p','*','h','+','x','1']
        block_size = len(colours)  # 10

        for idx, label in enumerate(data_map):
            # pick the Tab10 colour cyclically
            colour = colours[idx % block_size]

            # pick marker by block:
            #   curves 0–9    → 'o'
            #   curves 10–19  → extra_markers[0]
            #   curves 20–29  → extra_markers[1], etc.
            if idx < block_size:
                marker = 'o'
            else:
                block = (idx // block_size) - 1
                marker = extra_markers[block % len(extra_markers)]

            # detect whether we have precomputed stats+distributions
            if hasattr(self, '_stats_stabilizers') and label in self._stats_stabilizers:
                dist_map = self._replicate_means_stabilizers[label]
                stats    = self._stats_stabilizers[label]
                rounds   = stats['rounds']
                centers  = stats['centers']
                dists    = [dist_map[r] for r in rounds]
            elif hasattr(self, '_stats_logicals') and label in self._stats_logicals:
                dist_map = self._replicate_means_logicals[label]
                stats    = self._stats_logicals[label]
                rounds   = stats['rounds']
                centers  = stats['centers']
                dists    = [dist_map[r] for r in rounds]
            else:
                # fresh single-run data: just (rounds, centers)
                rounds, centers = data_map[label]
                dists = None

            # plot the mean-points
            ax.plot(
                rounds,
                centers,
                marker=marker,
                linestyle='None',
                color=colour,
                label=label
            )

            # optionally overlay violins & replicate scatter
            if show_violin and dists is not None:
                vp = ax.violinplot(
                    dists,
                    positions=rounds,
                    showmeans=True,
                    showextrema=True,
                    showmedians=False,
                    **(violin_kwargs or {})
                )
                # colour the violin bodies
                for body in vp['bodies']:
                    body.set_facecolor(colour)
                    body.set_edgecolor(colour)
                    body.set_alpha(0.3)
                # colour the mean/extrema lines
                for key in ('cmeans','cmaxes','cmins','cbars'):
                    lines = vp[key]
                    if isinstance(lines, list):
                        for ln in lines:
                            ln.set_color(colour)
                            ln.set_linewidth(1.5)
                    else:
                        lines.set_color(colour)
                        lines.set_linewidth(2 if key == 'cmeans' else 1.5)
                # scatter all replicates
                for r, vals in zip(rounds, dists):
                    ax.scatter([r]*len(vals), vals,
                               color=colour, alpha=0.05, s=4)

        # finalize
        ax.set_title(title, fontsize=18)
        ax.set_xlabel(x_label, fontsize=16)
        ax.set_ylabel(y_label, fontsize=16)
        ax.legend(loc=legend_location,
                  bbox_to_anchor=bbox_to_anchor,
                  fontsize='small',
                  **legend_kwargs)
        ax.grid(True)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        
    def plot_expectation(
        self,
        pauli_specification: Union[str, List[str]] = None,
        save_path: str = None,
        **plot_kwargs
    ) -> None:
        """
        Bespoke single-operator plotting: computes a fresh single-run curve,
        then delegates to plot_data.

        Args:
            pauli_specification (Union[str, List[str]], optional):
                Pauli specification(s) to plot. If None, uses all stabilizers.
            save_path (str, optional): If provided, saves the figure to this path.
            **plot_kwargs: Additional keyword arguments for plot_data.

        Returns:
            None
        """
        if pauli_specification is None:
            # Use all stabilizers if no specification is given
            specs = self.stabilizer_pauli_strings
        elif isinstance(pauli_specification, str):
            # Wrap single string in a list
            specs = [pauli_specification]
        else:
            # Use the provided list of specifications
            specs = pauli_specification

        # Create a mapping from spec to itself for labeling
        spec_map = {s: s for s in specs}
        # Compute data for each specification
        data = self.calculate_many_expectations(spec_map)

        # Plot the data using plot_data
        self.plot_data(data,
                       title='|⟨O⟩| vs MDR rounds',
                       x_label='MDR rounds',
                       y_label='|⟨O⟩|',
                       save_path=save_path,
                       **plot_kwargs)

    def plot_many_expectations(
        self,
        pauli_specifications: Union[List[str], Mapping[str, str]],
        save_path: str = None,
        **plot_kwargs
    ) -> None:
        """
        Bespoke multi-operator plotting: computes fresh single-run curves,
        then delegates to plot_data.

        Args:
            pauli_specifications (Union[List[str], Mapping[str, str]]):
                List or mapping of Pauli specifications to plot.
            save_path (str, optional): If provided, saves the figure to this path.
            **plot_kwargs: Additional keyword arguments for plot_data.

        Returns:
            None
        """
        if isinstance(pauli_specifications, Mapping):
            # Use values from mapping if provided
            specs = list(pauli_specifications.values())
        else:
            # Use the provided list of specifications
            specs = pauli_specifications

        # Create a mapping from spec to itself for labeling
        spec_map = {s: s for s in specs}
        # Compute data for each specification
        data = self.calculate_many_expectations(spec_map)

        # Plot the data using plot_data
        self.plot_data(data,
                       title='|⟨O⟩| vs MDR rounds',
                       x_label='MDR rounds',
                       y_label='|⟨O⟩|',
                       save_path=save_path,
                       **plot_kwargs)

    def plot_group(
        self,
        save_path_stabilizers: str = None,
        save_path_logicals: str = None,
        **plot_kwargs
    ) -> None:
        """
        Render precomputed grouped violins & markers (no lines) for all
        stabilizers and logicals. Saves figures if paths are provided.

        Args:
            save_path_stabilizers (str, optional): Path to save stabilizer plot.
            save_path_logicals (str, optional): Path to save logicals plot.
            **plot_kwargs: Additional keyword arguments for plot_data.

        Returns:
            None
        """
        # Plot all stabilizers using precomputed data
        self.plot_data(self.last_results_storage['stabilizers'],
                       title='|⟨S⟩| vs MDR rounds (stabilizers)',
                       x_label='MDR rounds',
                       y_label='|⟨S⟩|',
                       save_path=save_path_stabilizers,
                       **plot_kwargs)
        # Plot all logicals using precomputed data
        self.plot_data(self.last_results_storage['logicals'],
                       title='|⟨L⟩| vs MDR rounds (logicals)',
                       x_label='MDR rounds',
                       y_label='|⟨L⟩|',
                       save_path=save_path_logicals,
                       **plot_kwargs)

    @staticmethod
    def plot_multi_fidelity(
        sims: Dict[str, 'MDRSimulation'],
        category: str,
        operators: Optional[Union[str, List[str]]] = None,
        *,
        show_violin: bool = True,
        show_replicates: bool = False,
        figsize: Tuple[int,int] = (16, 8),
        save_path: Optional[str] = None
    ) -> None:
        """
        Compare ⟨|O|⟩ vs MDR round across multiple MDRSimulation instances.
        Plots only isolated markers with ±1σ error‐bars, plus optional
        violin + replicate overlays.  Supports 'stabilizer', 'logical', or 'both'.

        Args:
            sims:            Mapping label → MDRSimulation (one column per).
            category:        'stabilizer', 'logical', or 'both'.
            operators:       None (all), a single str, or a list of str.
            show_violin:     If True, overlay violin plots.
            show_replicates: If True, scatter individual replicate points.
            figsize:         Figure size (width, height) in inches.
            save_path:       If provided, save figure (dpi=300).

        Raises:
            ValueError: If `sims` is empty, or `category` invalid,
                        or any `operators` not found.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import math

        if not sims:
            raise ValueError("No simulations provided")

        # determine rows
        if category == 'both':
            cats = ['stabilizer', 'logical']
        elif category in ('stabilizer', 'logical'):
            cats = [category]
        else:
            raise ValueError("category must be 'stabilizer', 'logical' or 'both'")

        # collect which ops go in each row
        ops_dict: Dict[str, List[str]] = {}
        for cat in cats:
            first = next(iter(sims.values()))
            stats_attr = '_stats_stabilizers' if cat=='stabilizer' else '_stats_logicals'
            all_ops = list(getattr(first, stats_attr).keys())

            # select subset
            if operators is None:
                ops = all_ops
            elif isinstance(operators, str):
                ops = [operators]
            else:
                ops = operators[:]
            missing = set(ops) - set(all_ops)
            if missing:
                raise ValueError(f"Unknown operators for {cat}: {missing}")
            ops_dict[cat] = ops

        # build one global style map so no repeats across rows
        colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
        markers = ['o','s','^','v','<','>','8','p','P','*','h','H',
                   '+','x','X','D','d','|','_','.']
        global_ops = sum((ops_dict[cat] for cat in cats), [])
        style_map: Dict[str, Tuple[str,str]] = {}
        for i, op in enumerate(global_ops):
            c = colours[i % len(colours)]
            m = markers[(i // len(colours)) % len(markers)]
            style_map[op] = (c, m)

        # set up figure
        nrows = len(cats)
        ncols = len(sims)
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=figsize,
            squeeze=False,
            sharex='col'
        )

        # collect legend handles
        legend_entries: Dict[str, plt.Line2D] = {}

        # plot each cell
        for row, cat in enumerate(cats):
            stats_attr = '_stats_stabilizers' if cat=='stabilizer' else '_stats_logicals'
            dist_attr  = '_replicate_means_stabilizers' if cat=='stabilizer' else '_replicate_means_logicals'
            ylabel     = '|⟨S⟩|' if cat=='stabilizer' else '|⟨L⟩|'
            ops = ops_dict[cat]

            for col, (label, sim) in enumerate(sims.items()):
                ax = axes[row][col]
                if row == 0:
                    ax.set_title(label, fontsize=18)
                ax.grid(True)

                stats_map = getattr(sim, stats_attr)
                dist_map  = getattr(sim, dist_attr)

                for op in ops:
                    rounds  = stats_map[op]['rounds']
                    centers = stats_map[op]['centers']
                    stds    = stats_map[op]['stds']
                    c, m    = style_map[op]

                    eb = ax.errorbar(
                        rounds, centers,
                        yerr=stds,
                        fmt=m,             # marker only
                        color=c,
                        capsize=3,
                        linestyle='None',
                        label=op,
                        zorder=3
                    )
                    line = eb.lines[0]  # matplotlib>=3.7

                    # record for shared legend
                    if op not in legend_entries:
                        legend_entries[op] = line

                    if show_violin:
                        data = [dist_map[op][r] for r in rounds]
                        vp = ax.violinplot(
                            data,
                            positions=rounds,
                            widths=0.6,
                            showmeans=False,
                            showextrema=False
                        )
                        for body in vp['bodies']:
                            body.set_facecolor(c)
                            body.set_edgecolor(c)
                            body.set_alpha(0.2)

                    if show_replicates:
                        for r in rounds:
                            ax.scatter(
                                [r]*len(dist_map[op][r]),
                                dist_map[op][r],
                                color=c,
                                alpha=0.05,
                                s=4,
                                zorder=2
                            )

                if col == 0:
                    ax.set_ylabel(ylabel, fontsize=16)
                if row == nrows - 1:
                    ax.set_xlabel("MDR rounds", fontsize=16)

        # single legend beside the grid
        handles = list(legend_entries.values())
        labels  = list(legend_entries.keys())
        fig.legend(
            handles, labels,
            loc='center left',
            bbox_to_anchor=(0.93, 0.5),
            fontsize=16,
        )

        # make room for legend
        fig.tight_layout(rect=[0, 0, 0.935, 1.0])

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


    
    def results(self) -> None:
        """
        Print precomputed tabular statistics (round | mean | stddev)
        and overall average expectation for each operator.

        Args:
            None

        Returns:
            None
        """
        for category, (stats_map, avg_map) in (
            ('stabilizers', (self._stats_stabilizers, self._avg_stabilizers)),
            ('logicals',    (self._stats_logicals,    self._avg_logicals))
        ):
            # Print category header
            print(f"=== {category.upper()} ===")
            for label, stats in stats_map.items():
                # Print label header
                print(f"-- {label} --")
                print(f"{'Round':>5} | {'Mean':>7} | {'StdDev':>7}")
                print("-----+---------+--------")
                for r, m, s in zip(stats['rounds'], stats['centers'],
                                   stats['stds']):
                    # Print round, mean, and stddev for each round
                    print(f"{r:5d} | {m:7.4f} | {s:7.4f}")
                # Print overall average for this operator
                print(f"Avg across rounds: {avg_map[label]:.4f}\n")

    
    def tau_table(
        self,
        category: str,
        *,
        ignore_round0: bool = True,
        return_dict: bool = False,
        precision: int = 4,
        label: Optional[str] = None
    ):
        """
        Compute and display per-round across-operator dispersion τ_r and mean μ_r.

        Definitions (for the chosen category):
            For each round r let the set of replicate-mean fidelities across
            operators be { m_{k,r} }_{k=1}^K, where m_{k,r} is the plotted mean
            for operator k at round r (already stored in stats['centers']).

            μ_r  = (1/K) Σ_k m_{k,r}
            τ_r  = sqrt( Σ_k (m_{k,r} - μ_r)^2 / (K-1) )  (sample std across operators)

        We report:
            * Per-round μ_r and τ_r.
            * Average τ across rounds (excluding r=0 if requested).
            * Average μ across rounds (excluding r=0 if requested).
            * Both also including round 0 for reference.

        Args:
            category (str):
                'stabilizer' or 'logical'.
            ignore_round0 (bool, optional):
                Exclude round 0 from the averages (default True).
            return_dict (bool, optional):
                If True, return a dict of computed quantities.
            precision (int, optional):
                Number of decimal places in printed table.
            label (str, optional):
                Optional label (e.g. noise model name) for header.

        Returns:
            Optional[Dict[str, Any]]:
                {
                  'rounds': [...],
                  'tau': [...],
                  'mu': [...],
                  'avg_tau_excluding_round0': float,
                  'avg_tau_including_round0': float,
                  'avg_mu_excluding_round0': float,
                  'avg_mu_including_round0': float
                } if return_dict=True, else None.

        Raises:
            ValueError: If category invalid.
        """
        import numpy as np

        if category not in ('stabilizer', 'logical'):
            raise ValueError("category must be 'stabilizer' or 'logical'")

        stats_map = self._stats_stabilizers if category == 'stabilizer' else self._stats_logicals

        first_stats = next(iter(stats_map.values()))
        rounds = first_stats['rounds']

        tau_list = []
        mu_list = []
        for idx_r, r in enumerate(rounds):
            vals_r = [op_stats['centers'][idx_r] for op_stats in stats_map.values()]
            arr = np.asarray(vals_r, dtype=float)
            mu_r = float(arr.mean())
            mu_list.append(mu_r)
            tau_r = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
            tau_list.append(tau_r)

        # Slicing for averages
        if ignore_round0 and rounds and rounds[0] == 0:
            tau_ex = tau_list[1:]
            mu_ex  = mu_list[1:]
        else:
            tau_ex = tau_list
            mu_ex  = mu_list

        avg_tau_ex = float(np.mean(tau_ex)) if tau_ex else 0.0
        avg_tau_in = float(np.mean(tau_list)) if tau_list else 0.0
        avg_mu_ex  = float(np.mean(mu_ex)) if mu_ex else 0.0
        avg_mu_in  = float(np.mean(mu_list)) if mu_list else 0.0

        hdr_label = label if label is not None else "Simulation"
        print(f"\n=== τ_r & μ_r ({category}s) for {hdr_label} ===")
        print(f"{'Round':>5} | {'μ_r':>8} | {'τ_r':>8}")
        print("------+----------+----------")
        fmt = f"{{:5d}} | {{:{8}.{precision}f}} | {{:{8}.{precision}f}}"
        for r, mu_r, tau_r in zip(rounds, mu_list, tau_list):
            print(fmt.format(r, mu_r, tau_r))
        excl_txt = " (excl. round 0)" if ignore_round0 else ""
        print("-" * 36)
        print(f"Average τ{excl_txt}: {avg_tau_ex:.{precision}f}")
        print(f"Average τ (incl. round 0): {avg_tau_in:.{precision}f}")
        print(f"Average μ{excl_txt}: {avg_mu_ex:.{precision}f}")
        print(f"Average μ (incl. round 0): {avg_mu_in:.{precision}f}\n")

        if return_dict:
            return {
                'rounds': rounds,
                'tau': tau_list,
                'mu': mu_list,
                'avg_tau_excluding_round0': avg_tau_ex,
                'avg_tau_including_round0': avg_tau_in,
                'avg_mu_excluding_round0': avg_mu_ex,
                'avg_mu_including_round0': avg_mu_in
            }

    @staticmethod
    def compare_tau(
        sims: Dict[str, 'MDRSimulation'],
        category: str,
        *,
        ignore_round0: bool = True,
        precision: int = 4,
        return_table: bool = False
    ):
        """
        Compare average across-operator dispersion τ and average mean μ
        among multiple simulations.

        For each simulation and the chosen category:
           * Compute τ_r and μ_r per round.
           * Report averages excluding (optionally) round 0 and including it.

        Args:
            sims (Dict[str, MDRSimulation]):
                Mapping simulation label → instance.
            category (str):
                'stabilizer' or 'logical'.
            ignore_round0 (bool, optional):
                Exclude round 0 from averages (default True).
            precision (int, optional):
                Printing precision.
            return_table (bool, optional):
                If True, return list of row dicts.

        Returns:
            Optional[List[Dict[str, float]]]:
                Each row has: simulation, avg_tau, avg_tau_including_round0,
                avg_mu, avg_mu_including_round0 (if return_table=True).

        Raises:
            ValueError: If sims empty or category invalid.
        """
        import numpy as np

        if not sims:
            raise ValueError("No simulations provided")
        if category not in ('stabilizer', 'logical'):
            raise ValueError("category must be 'stabilizer' or 'logical'")

        rows = []
        for label, sim in sims.items():
            stats_map = sim._stats_stabilizers if category == 'stabilizer' else sim._stats_logicals
            first_stats = next(iter(stats_map.values()))
            rounds = first_stats['rounds']

            # Build per-round operator arrays
            per_round_vals = []
            for idx_r, r in enumerate(rounds):
                arr = np.array([op_stats['centers'][idx_r] for op_stats in stats_map.values()], dtype=float)
                per_round_vals.append(arr)

            tau_list = [float(arr.std(ddof=1)) if arr.size > 1 else 0.0 for arr in per_round_vals]
            mu_list  = [float(arr.mean()) for arr in per_round_vals]

            if ignore_round0 and rounds and rounds[0] == 0:
                tau_ex = tau_list[1:]
                mu_ex  = mu_list[1:]
            else:
                tau_ex = tau_list
                mu_ex  = mu_list

            avg_tau_ex = float(np.mean(tau_ex)) if tau_ex else 0.0
            avg_tau_in = float(np.mean(tau_list)) if tau_list else 0.0
            avg_mu_ex  = float(np.mean(mu_ex)) if mu_ex else 0.0
            avg_mu_in  = float(np.mean(mu_list)) if mu_list else 0.0

            rows.append({
                'simulation': label,
                'avg_tau': avg_tau_ex,
                'avg_tau_including_round0': avg_tau_in,
                'avg_mu': avg_mu_ex,
                'avg_mu_including_round0': avg_mu_in
            })

        excl = " (excl. round 0)" if ignore_round0 else ""
        print(f"\n=== Average τ and μ{excl} comparison ({category}s) ===")
        print(f"{'Simulation':<30} | {'avg τ':>10} | {'avg τ (incl r0)':>16} | {'avg μ':>10} | {'avg μ (incl r0)':>16}")
        print("-" * 94)
        for row in rows:
            print(f"{row['simulation']:<30} | "
                  f"{row['avg_tau']:{10}.{precision}f} | "
                  f"{row['avg_tau_including_round0']:{16}.{precision}f} | "
                  f"{row['avg_mu']:{10}.{precision}f} | "
                  f"{row['avg_mu_including_round0']:{16}.{precision}f}")
        print()

        if return_table:
            return rows
