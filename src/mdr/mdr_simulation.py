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
    correction_mode : str
        Recovery implementation policy inherited from the MDR circuit
        (`physical` or `pauli_frame`).
    mdr_circuit : stim.Circuit
        One complete MDR round template used by this simulation. In physical
        recovery mode this includes recovery when recovery is applied after
        each round. In Pauli-frame mode it contains only syndrome extraction.
    syndrome_round_circuit : stim.Circuit
        One syndrome-extraction round without appended recovery toggles.
    recovery_circuit : stim.Circuit
        Standalone recovery circuit used when recovery is deferred until the
        final round in physical-recovery mode.
    num_syndrome_bits_per_round : int
        Number of recorded syndrome bits produced by one MDR round.
    _toggle_x_masks : np.ndarray
        Binary X-support masks for each recovery toggle.
    _toggle_z_masks : np.ndarray
        Binary Z-support masks for each recovery toggle.
    _stabilizer_x_masks : np.ndarray
        Binary X-support masks for each syndrome operator measured per round.
    _stabilizer_z_masks : np.ndarray
        Binary Z-support masks for each syndrome operator measured per round.
    _operator_mask_cache : Dict[str, Tuple[np.ndarray, np.ndarray]]
        Cache from sparse Pauli strings to symplectic X/Z masks.
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
    _pauli_spec_to_mask(pauli_specification)
        Convert a sparse Pauli string into symplectic X/Z support masks.
    _frame_anticommutation(frame_x, frame_z, operator_x, operator_z)
        Compute per-shot anti-commutation parities between frames and an
        operator.
    _accumulate_pauli_frame(syndrome_bits, round_count)
        Reconstruct the net Pauli frame implied by measured syndrome history.
    compute_parity_expectation(circuit, pauli_specification, measurement_ops,
        round_count, absolute_value)
        Sample a measured operator parity and return the signed or
        absolute-value expectation, including Pauli-frame correction when
        requested.
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
        self.correction_mode = mdr.correction_mode
        self.syndrome_round_circuit = mdr.build(
            include_psi=False,
            include_recovery=False,
        )
        self.mdr_circuit = self.syndrome_round_circuit
        self.recovery_circuit = stim.Circuit()
        if self.correction_mode == "physical":
            self.mdr_circuit = mdr.build(
                include_psi=False,
                include_recovery=True,
            )
        if (
            self.correction_mode == "physical"
            and self.recovery_mode == "final_round"
        ):
            self.recovery_circuit = mdr.build_recovery_only()
        self.p_spam = mdr.p_spam
        self.num_syndrome_bits_per_round = len(mdr.stabilizers)
        self._toggle_x_masks, self._toggle_z_masks = self._build_mask_table(
            mdr.toggles,
            mdr.num_qubits,
        )
        self._stabilizer_x_masks, self._stabilizer_z_masks = (
            self._build_mask_table(
                mdr.stabilizers,
                mdr.num_qubits,
            )
        )
        self._operator_mask_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
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

    @staticmethod
    def _build_mask_table(
        pauli_specifications: List[str],
        num_qubits: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a list of sparse Pauli strings into aligned X/Z mask tables.

        Args:
            pauli_specifications: Sparse Pauli strings to encode.
            num_qubits: Number of data qubits represented by the masks.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
            `(x_masks, z_masks)` with shape
            `(len(pauli_specifications), num_qubits)` and `uint8` entries.
        """
        if not pauli_specifications:
            empty = np.zeros((0, num_qubits), dtype=np.uint8)
            return empty.copy(), empty

        x_masks: List[np.ndarray] = []
        z_masks: List[np.ndarray] = []
        for spec in pauli_specifications:
            x_mask = np.zeros(num_qubits, dtype=np.uint8)
            z_mask = np.zeros(num_qubits, dtype=np.uint8)
            for term in spec.split():
                pauli = term[0].upper()
                qubit = int(term[1:])
                if pauli in {"X", "Y"}:
                    x_mask[qubit] ^= 1
                if pauli in {"Z", "Y"}:
                    z_mask[qubit] ^= 1
            x_masks.append(x_mask)
            z_masks.append(z_mask)

        return np.asarray(x_masks, dtype=np.uint8), np.asarray(
            z_masks,
            dtype=np.uint8,
        )

    def _pauli_spec_to_mask(
        self,
        pauli_specification: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert one sparse Pauli string into cached X/Z support masks.

        Args:
            pauli_specification: Sparse Pauli string such as `"X0 Z3 Y8"`.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
            `(x_mask, z_mask)` vectors over the data-qubit register.
        """
        cached = self._operator_mask_cache.get(pauli_specification)
        if cached is not None:
            return cached

        x_masks, z_masks = self._build_mask_table(
            [pauli_specification],
            self.mdr.num_qubits,
        )
        masks = (x_masks[0], z_masks[0])
        self._operator_mask_cache[pauli_specification] = masks
        return masks

    @staticmethod
    def _frame_anticommutation(
        frame_x: np.ndarray,
        frame_z: np.ndarray,
        operator_x: np.ndarray,
        operator_z: np.ndarray,
    ) -> np.ndarray:
        """
        Compute anti-commutation parity between frames and one Pauli operator.

        Args:
            frame_x: Per-shot X support for the accumulated frame.
            frame_z: Per-shot Z support for the accumulated frame.
            operator_x: X support mask for the operator of interest.
            operator_z: Z support mask for the operator of interest.

        Returns:
            np.ndarray: Length-`shots` vector with `1` where the frame
            anti-commutes with the operator and `0` otherwise.
        """
        parity = (frame_x @ operator_z) + (frame_z @ operator_x)
        return np.mod(parity, 2).astype(np.uint8)

    def _accumulate_pauli_frame(
        self,
        syndrome_bits: np.ndarray,
        round_count: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct the net Pauli frame implied by measured syndrome history.

        In `physical` mode the circuit itself already applies recovery and no
        frame reconstruction is needed. In `pauli_frame` mode this method
        emulates the same logical recovery policy without applying any data
        gates:
        - `final_round`: use only the last raw syndrome block, matching the
          physical deferred-recovery circuit.
        - `each_round`: interpret each round's syndrome in the current frame,
          then update the frame with the round's inferred toggle.

        Args:
            syndrome_bits: Array of shape `(shots, round_count * n_stabilizers)`
                containing the raw syndrome measurements preceding the final
                observable measurement.
            round_count: Number of MDR rounds executed before final readout.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
            `(frame_x, frame_z)` arrays of shape `(shots, num_qubits)`.
        """
        shots = syndrome_bits.shape[0]
        frame_x = np.zeros((shots, self.mdr.num_qubits), dtype=np.uint8)
        frame_z = np.zeros((shots, self.mdr.num_qubits), dtype=np.uint8)

        if round_count == 0 or syndrome_bits.size == 0:
            return frame_x, frame_z

        syndrome_rounds = syndrome_bits.reshape(
            shots,
            round_count,
            self.num_syndrome_bits_per_round,
        ).astype(np.uint8)

        if self.recovery_mode == "final_round":
            toggle_bits = syndrome_rounds[:, -1, :]
            frame_x ^= np.mod(toggle_bits @ self._toggle_x_masks, 2).astype(
                np.uint8
            )
            frame_z ^= np.mod(toggle_bits @ self._toggle_z_masks, 2).astype(
                np.uint8
            )
            return frame_x, frame_z

        stabilizer_z_t = self._stabilizer_z_masks.T
        stabilizer_x_t = self._stabilizer_x_masks.T
        for round_idx in range(round_count):
            raw_bits = syndrome_rounds[:, round_idx, :]
            frame_syndrome = np.mod(
                (frame_x @ stabilizer_z_t) + (frame_z @ stabilizer_x_t),
                2,
            ).astype(np.uint8)
            corrected_bits = np.bitwise_xor(raw_bits, frame_syndrome)
            frame_x ^= np.mod(
                corrected_bits @ self._toggle_x_masks,
                2,
            ).astype(np.uint8)
            frame_z ^= np.mod(
                corrected_bits @ self._toggle_z_masks,
                2,
            ).astype(np.uint8)

        return frame_x, frame_z

    def compute_parity_expectation(
        self,
        circuit: stim.Circuit,
        pauli_specification: str,
        measurement_ops: List[Tuple[str, int]],
        round_count: int = 0,
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
            pauli_specification: Sparse Pauli string defining the measured
                operator.
            measurement_ops: Ordered measurement operations produced by
                :meth:`spec_to_measurement_ops`.
            round_count: Number of MDR rounds represented by `circuit`.
            absolute_value: If True, return `|<O>|`; otherwise return the
                signed expectation `<O>`.

        Returns:
            float: Mean parity eigenvalue for the requested observable.
        """
        pre_measurement_count = circuit.num_measurements
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
        if self.correction_mode == "pauli_frame" and round_count > 0:
            syndrome_count = round_count * self.num_syndrome_bits_per_round
            syndrome_start = pre_measurement_count - syndrome_count
            syndrome_end = pre_measurement_count
            syndrome_bits = samples[:, syndrome_start:syndrome_end]
            frame_x, frame_z = self._accumulate_pauli_frame(
                syndrome_bits=syndrome_bits,
                round_count=round_count,
            )
            operator_x, operator_z = self._pauli_spec_to_mask(
                pauli_specification
            )
            anti = self._frame_anticommutation(
                frame_x=frame_x,
                frame_z=frame_z,
                operator_x=operator_x,
                operator_z=operator_z,
            )
            frame_sign = 1 - 2 * anti.astype(np.int8)
            eigen = eigen * frame_sign
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
                    pauli_specification=pauli_specification,
                    measurement_ops=measurement_ops,
                    round_count=round_idx,
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
