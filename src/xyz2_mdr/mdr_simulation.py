"""
mdr_simulation.py
-----------------
Core MDR simulation object responsible for computing and caching per-round
observable statistics.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import stim

from .mdr_circuit import MDRCircuit


class MDRSimulation:
    """
    Compute and cache Measurement-based Decoding & Recovery (MDR) statistics.
    Supports repeated-round evaluation of stabilizer and logical observables
    while caching replicate-level summary statistics for downstream analysis.

    Attributes
    ----------
    mdr : MDRCircuit
        Stored MDR circuit builder supplying the syndrome and recovery
        templates for this simulation instance.
    prepare_circuit_function : Callable[[], stim.Circuit]
        Callable returning the initial state-preparation circuit.
    recovery_mode : str
        Recovery timing policy inherited from the MDR circuit
        (`each_round` or `final_round`).
    mdr_circuit : stim.Circuit
        One complete MDR round including recovery when recovery is applied
        after each round.
    syndrome_round_circuit : stim.Circuit
        One syndrome-extraction round without appended recovery toggles.
    recovery_circuit : stim.Circuit
        Standalone recovery circuit used when recovery is deferred until the
        final round.
    stabilizer_pauli_strings : List[str]
        Stabilizer observables measured by the simulation.
    logical_pauli_strings : Dict[str, str]
        Mapping from logical labels to sparse Pauli strings.
    shots_per_measurement : int
        Number of Stim samples used for each replicate estimate.
    total_mdr_rounds : int
        Maximum MDR round index simulated and cached.
    num_replicates : int
        Number of independent replicate estimates recorded per round.
    _replicate_means_stabilizers : Dict[str, Dict[int, List[float]]]
        Replicate distributions for stabilizer observables by round.
    _replicate_means_logicals : Dict[str, Dict[int, List[float]]]
        Absolute-value replicate distributions for logical observables.
    _replicate_means_logicals_signed : Dict[str, Dict[int, List[float]]]
        Signed replicate distributions for logical observables.
    _stats_stabilizers : Dict[str, Dict[str, List[float]]]
        Per-round summary statistics for stabilizer observables.
    _stats_logicals : Dict[str, Dict[str, List[float]]]
        Per-round summary statistics for absolute-value logical observables.
    _stats_logicals_signed : Dict[str, Dict[str, List[float]]]
        Per-round summary statistics for signed logical observables.
    _avg_stabilizers : Dict[str, float]
        Average stabilizer fidelity across cached rounds for each stabilizer.
    _avg_logicals : Dict[str, float]
        Average absolute-value logical fidelity across cached rounds.
    _avg_logicals_signed : Dict[str, float]
        Average signed logical expectation across cached rounds.

    Methods
    -------
    __init__(...)
        Initialize the simulation object and precompute all cached
        distributions and summary statistics.
    spec_to_measurement_ops(pauli_specification)
        Convert a sparse Pauli string into the corresponding Stim
        single-qubit measurement operations.
    compute_parity_expectation(circuit, measurement_ops, absolute_value)
        Sample a measured operator parity and return the signed or
        absolute-value expectation.
    calculate_replicated_means_vs_rounds(pauli_specification, absolute_value)
        Compute replicate means for one observable across all MDR rounds.
    _summarize_distribution_map(dist_map)
        Reduce replicate distributions into per-round centers and standard
        deviations.
    """

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
        Initialize the simulation and precompute all cached round statistics.

        The constructor expands the supplied MDR circuit object into the
        circuit fragments needed for both recovery modes, then immediately
        evaluates every requested stabilizer and logical observable for every
        round from `0` through `total_mdr_rounds`. The resulting replicate
        distributions and summary statistics are cached on the instance so
        downstream plotting and sweep code can reuse them without re-running
        Stim.

        Args:
            mdr: Configured MDR circuit builder.
            stabilizer_pauli_strings: Stabilizer observables to measure.
            logical_pauli_strings: Logical observables to measure.
            shots_per_measurement: Shot count for each replicate estimate.
            total_mdr_rounds: Largest MDR round index to evaluate.
            num_replicates: Number of replicate estimates per round.
        """
        self.mdr = mdr
        self.prepare_circuit_function = mdr.psi
        self.recovery_mode = mdr.recovery_mode
        self.mdr_circuit = mdr.build(include_psi=False, include_recovery=True)
        self.syndrome_round_circuit = mdr.build(
            include_psi=False,
            include_recovery=False,
        )
        self.recovery_circuit = stim.Circuit()
        if self.recovery_mode == "final_round":
            self.recovery_circuit = mdr.build_recovery_only()
        self.p_spam = mdr.p_spam
        self.stabilizer_pauli_strings = stabilizer_pauli_strings
        self.logical_pauli_strings = logical_pauli_strings
        self.shots_per_measurement = shots_per_measurement
        self.total_mdr_rounds = total_mdr_rounds
        self.num_replicates = num_replicates

        self._replicate_means_stabilizers: Dict[str, Dict[int, List[float]]] = {}
        for spec in stabilizer_pauli_strings:
            self._replicate_means_stabilizers[spec] = (
                self.calculate_replicated_means_vs_rounds(spec)
            )

        self._replicate_means_logicals: Dict[str, Dict[int, List[float]]] = {}
        self._replicate_means_logicals_signed: Dict[
            str, Dict[int, List[float]]
        ] = {}
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

    @staticmethod
    def spec_to_measurement_ops(
        pauli_specification: str,
    ) -> List[Tuple[str, int]]:
        """
        Convert a sparse Pauli string into Stim measurement instructions.

        Each token such as `X7` or `Z12` is mapped to the corresponding
        single-qubit Stim measurement gate (`MX`, `MY`, or `MZ`) plus the
        target qubit index. The returned sequence preserves token order so the
        parity convention used later remains deterministic.

        Args:
            pauli_specification: Sparse Pauli string such as `"X0 Z1 Y4"`.

        Returns:
            List[Tuple[str, int]]: Ordered `(measurement_gate, qubit)` pairs.

        Raises:
            ValueError: If a token starts with an unsupported Pauli letter.
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
        Measure an operator parity and return its expectation value.

        This method appends the requested basis measurements to a circuit
        copy, optionally inserts X-type SPAM noise before each measurement,
        samples the circuit, and converts the measured parity bits into
        `+1/-1` eigenvalues. The returned mean is either signed or absolute
        depending on `absolute_value`.

        Args:
            circuit: Circuit ending in the state to be measured.
            measurement_ops: Ordered measurement operations produced by
                :meth:`spec_to_measurement_ops`.
            absolute_value: If True, return `|<O>|`; otherwise return the
                signed expectation `<O>`.

        Returns:
            float: Mean parity eigenvalue for the requested observable.
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

    def calculate_replicated_means_vs_rounds(
        self,
        pauli_specification: str,
        absolute_value: bool = True,
    ) -> Dict[int, List[float]]:
        """
        Evaluate one observable across all MDR round counts.

        For every round index from `0` through `total_mdr_rounds`, this method
        assembles the appropriate circuit according to the configured recovery
        policy, repeats the observable estimate `num_replicates` times, and
        stores the resulting replicate means. These replicate distributions are
        the raw statistical data used later to compute the cached centers and
        standard deviations.

        Args:
            pauli_specification: Sparse Pauli string defining the observable.
            absolute_value: If True, record `|<O>|`; otherwise record signed
                `<O>`.

        Returns:
            Dict[int, List[float]]: Mapping `round_index -> replicate_means`.
        """
        measurement_ops = self.spec_to_measurement_ops(pauli_specification)
        dist_map: Dict[int, List[float]] = {}
        for round_idx in range(self.total_mdr_rounds + 1):
            base = self.prepare_circuit_function()
            if round_idx > 0:
                if self.recovery_mode == "each_round":
                    base += self.mdr_circuit * round_idx
                else:
                    base += self.syndrome_round_circuit * round_idx
                    base += self.recovery_circuit

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
        Reduce replicate distributions to per-round summary statistics.

        Args:
            dist_map: Mapping from round index to the replicate values
                collected for that round.

        Returns:
            Dict[str, List[float]]: Dictionary with aligned `rounds`,
            `centers`, and `stds` lists suitable for plotting and CSV export.
            The standard deviation is computed with `ddof=1` when at least two
            replicate values are available, otherwise `0.0` is reported.
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
