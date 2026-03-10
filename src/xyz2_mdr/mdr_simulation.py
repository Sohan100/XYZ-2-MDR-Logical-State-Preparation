"""
mdr_simulation.py
────────────────────────────────────────────────────────────────────────
Encapsulates measurement-based decoding & recovery (MDR) utilities in the
MDRSimulation class, with precomputed violin-distribution overlays and
detailed uncertainty reporting.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import stim

from .mdr_circuit import MDRCircuit


class MDRSimulation:
    """
    Tools for simulating Measurement-based Decoding & Recovery (MDR).
    
    This class precomputes, for each specified stabilizer and logical
    operator, the distribution of mean-estimates obtained by
    repeating an N-shot experiment (`shots_per_measurement`) `num_replicates`
    times per MDR round (0…`total_mdr_rounds`), and caches both the full
    replicate lists (for violin plots) and summary statistics (mean ±
    sample-stddev). Stabilizers and the legacy logical-fidelity view retain
    absolute values, while logical operators also keep signed expectations
    for state-preparation error analysis.
    
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

    # ─────────────────────────────────────────────────────────────────────
    # construction
    # ─────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        mdr: MDRCircuit,
        stabilizer_pauli_strings: List[str],
        logical_pauli_strings: Dict[str, str],
        shots_per_measurement: int = 1000,
        total_mdr_rounds: int = 10,
        num_replicates: int = 30,
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
                Number of repeats per round for violin distributions.
                Default 30.
        
        Returns:
            None
        """
        self.mdr = mdr
        self.prepare_circuit_function = mdr.psi
        self.mdr_circuit = mdr.build(include_psi=False)
        self.p_spam = mdr.p_spam
        self.stabilizer_pauli_strings = stabilizer_pauli_strings
        self.logical_pauli_strings = logical_pauli_strings
        self.shots_per_measurement = shots_per_measurement
        self.total_mdr_rounds = total_mdr_rounds
        self.num_replicates = num_replicates

        self._replicate_means_stabilizers: Dict[str, Dict[int, List[float]]]
        self._replicate_means_stabilizers = {}
        for spec in stabilizer_pauli_strings:
            self._replicate_means_stabilizers[spec] = (
                self.calculate_replicated_means_vs_rounds(spec)
            )

        self._replicate_means_logicals: Dict[str, Dict[int, List[float]]]
        self._replicate_means_logicals = {}
        self._replicate_means_logicals_signed: Dict[str, Dict[int, List[float]]]
        self._replicate_means_logicals_signed = {}
        for label, spec in logical_pauli_strings.items():
            self._replicate_means_logicals[label] = (
                self.calculate_replicated_means_vs_rounds(spec)
            )
            self._replicate_means_logicals_signed[label] = (
                self.calculate_replicated_means_vs_rounds(
                    spec,
                    absolute_value=False,
                )
            )

        self._stats_stabilizers: Dict[str, Dict[str, List[float]]] = {}
        self._avg_stabilizers: Dict[str, float] = {}
        for spec, dist_map in self._replicate_means_stabilizers.items():
            stats = self._summarize_distribution_map(dist_map)
            self._stats_stabilizers[spec] = stats
            self._avg_stabilizers[spec] = float(np.mean(stats["centers"]))

        self._stats_logicals: Dict[str, Dict[str, List[float]]] = {}
        self._avg_logicals: Dict[str, float] = {}
        self._stats_logicals_signed: Dict[str, Dict[str, List[float]]] = {}
        self._avg_logicals_signed: Dict[str, float] = {}
        for label, dist_map in self._replicate_means_logicals.items():
            stats = self._summarize_distribution_map(dist_map)
            self._stats_logicals[label] = stats
            self._avg_logicals[label] = float(np.mean(stats["centers"]))
            signed_stats = self._summarize_distribution_map(
                self._replicate_means_logicals_signed[label]
            )
            self._stats_logicals_signed[label] = signed_stats
            self._avg_logicals_signed[label] = float(
                np.mean(signed_stats["centers"])
            )

    # ─────────────────────────────────────────────────────────────────────
    # measurement helpers
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def spec_to_measurement_ops(
        pauli_specification: str,
    ) -> List[Tuple[str, int]]:
        """
        Convert a sparse Pauli string (e.g. "X0 Z1") into Stim measurement ops.
        
        Args:
            pauli_specification (str): Sparse Pauli string, e.g., "X0 Z1".
        
        Returns:
            List[Tuple[str, int]]: List of tuples (gate, qubit index), e.g.,
            [("MX", 0), ("MZ", 1)].
        """
        ops: List[Tuple[str, int]] = []
        gate_map = {"X": "MX", "Y": "MY", "Z": "MZ"}
        for term in pauli_specification.split():
            pauli = term[0].upper()
            if pauli not in gate_map:
                raise ValueError(f"Invalid Pauli letter: {pauli}")
            ops.append((gate_map[pauli], int(term[1:])))
        return ops

    def compute_parity_expectation(
        self,
        circuit: stim.Circuit,
        measurement_ops: List[Tuple[str, int]],
        absolute_value: bool = True,
    ) -> float:
        """
        Append measurement_ops to the circuit, sample, and compute |⟨O⟩|.
        
        Args:
            circuit (stim.Circuit): Circuit to which measurement ops are added.
            measurement_ops (List[Tuple[str, int]]): List of (gate, qubit) to
            measure.
        
        Args:
            absolute_value (bool): If True, return `|<O>|`; otherwise return
            the signed expectation `<O>`.

        Returns:
            float: Mean of the ±1 parity outcomes, optionally absolute-valued.
        """
        for gate, qubit in measurement_ops:
            if self.p_spam > 0:
                circuit += stim.Circuit(
                    f"PAULI_CHANNEL_1({self.p_spam},0,0) {qubit}"
                )
            circuit.append_operation(gate, qubit)

        sampler = circuit.compile_sampler()
        samples = sampler.sample(
            shots=self.shots_per_measurement,
            bit_packed=False,
        )
        cols = np.arange(-len(measurement_ops), 0)
        parity = np.sum(samples[:, cols], axis=1) % 2
        eigen = 1 - 2 * parity
        mean_val = float(np.mean(eigen))
        return float(abs(mean_val)) if absolute_value else mean_val

    # ─────────────────────────────────────────────────────────────────────
    # public api
    # ─────────────────────────────────────────────────────────────────────
    def calculate_replicated_means_vs_rounds(
        self,
        pauli_specification: str,
        absolute_value: bool = True,
    ) -> Dict[int, List[float]]:
        """
        For each round r, repeat the N-shot experiment `num_replicates` times
        and collect the absolute mean eigenvalue of each replicate.
        
        Args:
            pauli_specification (str): Sparse Pauli string to measure.
            absolute_value (bool): If True, collect `|<O>|`; otherwise collect
                signed `<O>`.
        
        Returns:
            Dict[int, List[float]]: Mapping from round index to list of
            replicate means.
        """
        measurement_ops = self.spec_to_measurement_ops(pauli_specification)
        dist_map: Dict[int, List[float]] = {}
        for round_idx in range(self.total_mdr_rounds + 1):
            base = self.prepare_circuit_function()
            if round_idx > 0:
                base += self.mdr_circuit * round_idx

            replicate_means: List[float] = []
            for _ in range(self.num_replicates):
                mean_val = self.compute_parity_expectation(
                    base.copy(),
                    measurement_ops,
                    absolute_value=absolute_value,
                )
                replicate_means.append(mean_val)
            dist_map[round_idx] = replicate_means
        return dist_map

    @staticmethod
    def _summarize_distribution_map(
        dist_map: Dict[int, List[float]],
    ) -> Dict[str, List[float]]:
        """
        Convert replicate distributions into per-round summary statistics.

        Args:
            dist_map: Mapping `round_index -> replicate_mean_values`.

        Returns:
            Dict[str, List[float]]: Dictionary with keys:
            `rounds` (sorted round indices), `centers` (mean values), and
            `stds` (sample standard deviations with `ddof=1`, or `0.0` when a
            round has a single replicate).
        """
        rounds = sorted(dist_map)
        centers: List[float] = []
        stds: List[float] = []
        for round_idx in rounds:
            vals = np.asarray(dist_map[round_idx], dtype=float)
            centers.append(float(np.mean(vals)))
            if len(vals) > 1:
                stds.append(float(np.std(vals, ddof=1)))
            else:
                stds.append(0.0)
        return {"rounds": rounds, "centers": centers, "stds": stds}

