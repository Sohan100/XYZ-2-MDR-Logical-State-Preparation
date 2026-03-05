import stim
from typing import List

class MDRCircuit:
    """
    Builds a Measurement-based Decoding & Recovery (MDR) circuit with
    configurable noise models and defaults.

    Args:
        stabilizers (List[str] | None): List of stabilizer generator strings.
            If None, a default set is used.
        toggles (List[str] | None): List of recovery toggle strings.
            If None, a default set is used.
        ancillas (int): Number of ancilla qubits to use (min 1).
        p_spam (float): Probability of SPAM noise.
        p_x (float): Probability of X error during idling.
        p_y (float): Probability of Y error during idling.
        p_z (float): Probability of Z error during idling.
        g1_x (float): Prob. of X error after each 1-qubit gate.
        g1_y (float): Prob. of Y error after each 1-qubit gate.
        g1_z (float): Prob. of Z error after each 1-qubit gate.
        gate_noise_2q (List[float] | None): 15 two-qubit error probabilities.
        psi_circuit (stim.Circuit | None): initial state preparation circuit.

    Methods
    -------
    __init__(...): Initialize MDRCircuit with defaults and noise parameters.
    _insert_pauli_channel(...): Add a 1-qubit biased depolarising channel.
    _insert_pauli_channel_2(...): Add a 2-qubit depolarising channel.
    add_idle_noise(...): Apply idling noise to specified idle qubits.
    psi(): Prepare and return the initial state preparation circuit.
    _gate(...): Append a quantum gate and its associated noise to the circuit.
    _spam_gate(...): Append a gate with SPAM noise (for R and M gates).
    build(...): Construct and return the full MDR protocol circuit, optionally
        including the |+> state preparation stage.
    """
    def __init__(
        self,
        stabilizers: List[str] | None = None,
        toggles: List[str] | None = None,
        ancillas: int = 1,
        p_spam: float = 0,
        p_x: float = 0,
        p_y: float = 0,
        p_z: float = 0,
        g1_x: float = 0,
        g1_y: float = 0,
        g1_z: float = 0,
        gate_noise_2q: List[float] | None = None,
        psi_circuit: stim.Circuit | None = None
    ) -> None:
        """
        Initialize the MDRCircuit object with user-specified or default
        stabilizers, toggles, ancilla count, and noise parameters. Validates
        input for two-qubit noise and ancilla count. Sets up all circuit
        parameters for later use in circuit construction.

        Args:
            stabilizers: List of stabilizer generator strings. If None, uses
                a default set for the code.
            toggles: List of recovery toggle strings. If None, uses a default
                set for the code.
            ancillas: Number of ancilla qubits to use for syndrome extraction.
                Must be at least 1.
            p_spam: Probability of SPAM (state preparation and measurement)
                noise.
            p_x: Probability of X error during idling.
            p_y: Probability of Y error during idling.
            p_z: Probability of Z error during idling.
            g1_x: Probability of X error after each 1-qubit gate.
            g1_y: Probability of Y error after each 1-qubit gate.
            g1_z: Probability of Z error after each 1-qubit gate.
            gate_noise_2q: List of 15 probabilities for two-qubit gate errors.
                If None, all set to 0.
            psi_circuit: stim.Circuit for initial state preparation. If None,
                no preparation is included.
        Raises:
            ValueError: If gate_noise_2q is not length 15 or ancillas < 1.
        """
        if gate_noise_2q is None:
            gate_noise_2q = [0.0] * 15
        elif len(gate_noise_2q) != 15:
            raise ValueError("gate_noise_2q must have 15 floats.")
        if ancillas < 1:
            raise ValueError("ancillas must be at least 1.")
        
        self.stabilizers = stabilizers
        self.num_qubits = len(self.stabilizers) + 1
        self.toggles = toggles
        self.ancillas = ancillas
        self.p_spam = p_spam
        self.p_x = p_x
        self.p_y = p_y
        self.p_z = p_z
        self.g1_x = g1_x
        self.g1_y = g1_y
        self.g1_z = g1_z
        self.gate_noise_2q = gate_noise_2q
        self.psi_circuit = psi_circuit or None

    def _insert_pauli_channel(
        self,
        circ: stim.Circuit,
        tgts: List[int],
        p_x: float,
        p_y: float,
        p_z: float
    ):
        """
        Append a 1-qubit biased depolarising channel to the circuit. This
        channel applies X, Y, and Z errors to the specified qubits with
        probabilities p_x, p_y, and p_z, respectively. If all probabilities
        are zero or tgts is empty, no operation is added.

        Args:
            circ: stim.Circuit to append the noise operation to.
            tgts: List of qubit indices to apply the noise channel to.
            p_x: Probability of X error for each target qubit.
            p_y: Probability of Y error for each target qubit.
            p_z: Probability of Z error for each target qubit.
        """
        if not tgts or (p_x == p_y == p_z == 0):
            return
        circ += stim.Circuit(
            f"PAULI_CHANNEL_1({p_x},{p_y},{p_z}) "
            f"{' '.join(str(q) for q in tgts)}"
        )

    def _insert_pauli_channel_2(
        self,
        circ: stim.Circuit,
        q_a: int,
        q_d: int,
        noise: List[float]
    ):
        """
        Append a 2-qubit depolarising channel to the circuit. This channel
        applies all possible two-qubit Pauli errors (except II) with the
        specified probabilities. If all probabilities are zero, no operation
        is added.

        Args:
            circ: stim.Circuit to append the noise operation to.
            q_a: Index of the first qubit in the two-qubit channel.
            q_d: Index of the second qubit in the two-qubit channel.
            noise: List of 15 probabilities for each nontrivial two-qubit
                Pauli error (order: IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY,
                YZ, ZI, ZX, ZY, ZZ).
        """
        if not any(noise):
            return
        circ += stim.Circuit(
            f"PAULI_CHANNEL_2({','.join(f'{p:.6g}' for p in noise)})"
            f" {q_a} {q_d}"
        )

    def add_idle_noise(
        self,
        circ: stim.Circuit,
        idle_qs: List[int],
        p_x: float = 1e-4,
        p_y: float = 1e-4,
        p_z: float = 8e-4
    ):
        """
        Apply biased idling noise to a list of idle qubits. This method
        inserts a 1-qubit Pauli channel with the specified error rates for
        X, Y, and Z errors. Used to model decoherence on qubits not involved
        in active gates during a circuit tick.

        Args:
            circ: stim.Circuit to append the idle noise operation to.
            idle_qs: List of qubit indices considered idle at this tick.
            p_x: Probability of X error for each idle qubit.
            p_y: Probability of Y error for each idle qubit.
            p_z: Probability of Z error for each idle qubit.
        """
        self._insert_pauli_channel(circ, idle_qs, p_x, p_y, p_z)

    def psi(self) -> stim.Circuit:
        """
        Prepare and return the initial state preparation circuit for the MDR
        protocol. If a psi_circuit was provided at initialization, returns a
        copy of that circuit. Otherwise, returns an empty circuit.

        Returns:
            stim.Circuit: Circuit that prepares the initial state for the
                protocol, as specified by psi_circuit.
        """
        circ = stim.Circuit(str(self.psi_circuit))
        return circ

    def _gate(
        self,
        circ: stim.Circuit,
        name: str,
        tgts: List[int]
    ):
        """
        Append a quantum gate and its associated noise to the circuit. For
        1-qubit gates, applies a 1-qubit Pauli channel after the gate. For
        2-qubit gates, applies a 2-qubit Pauli channel. For gates with more
        than 2 targets, applies 1-qubit noise to each target. If all noise
        rates are zero, only the gate is appended.

        Args:
            circ: stim.Circuit to modify by appending the gate and noise.
            name: Name of the quantum gate (e.g., 'H', 'CX', 'CY', 'CZ').
            tgts: List of qubit indices the gate acts on.
        """
        circ.append_operation(name, tgts)

        if (self.g1_x == self.g1_y == self.g1_z == 0
            and not any(self.gate_noise_2q)):
            return

        real = [q for q in tgts if isinstance(q, int)]
        if not real:
            return

        if len(real) > 2:
            for q in real:
                self._insert_pauli_channel(
                    circ, [q], self.g1_x, self.g1_y, self.g1_z
                )
            return

        if len(real) == 1:
            self._insert_pauli_channel(
                circ, real, self.g1_x, self.g1_y, self.g1_z
            )
        else:
            q1, q2 = real
            self._insert_pauli_channel_2(
                circ, q1, q2, self.gate_noise_2q
            )

    def _spam_gate(self, circ: stim.Circuit, name: str, tgts: List[int]):
        """
        Append a gate to the circuit with SPAM noise. For 'R' (reset), the
        noise is applied after the reset. For 'M' (measurement), the noise
        is applied before the measurement. If p_spam is zero, only the gate
        is appended. Used to model state preparation and measurement errors.

        Args:
            circ: stim.Circuit to modify by appending the gate and noise.
            name: Name of the gate ('R' for reset, 'M' for measurement).
            tgts: List of qubit indices the gate acts on.
        """
        if self.p_spam == 0:
            circ.append_operation(name, tgts)
            return
        
        if name == 'R':
            circ.append_operation(name, tgts)
            self._insert_pauli_channel(
                circ, tgts, self.p_spam, 0, 0
            )

        if name == 'M':
            self._insert_pauli_channel(
                circ, tgts, self.p_spam, 0, 0
            )
            circ.append_operation(name, tgts)
    
    def build(
        self,
        include_psi: bool = True,
    ) -> stim.Circuit:
        """
        Construct and return the full MDR protocol circuit. This includes
        syndrome extraction using ancilla qubits, application of idle noise,
        and recovery toggles based on measurement results. Optionally
        prepends the initial state preparation circuit if include_psi is
        True.

        Args:
            include_psi (bool): If True, prepends the |+> state preparation
                circuit (psi_circuit) to the protocol. If False, starts with
                an empty circuit.

        Returns:
            stim.Circuit: The complete MDR protocol circuit, including
                syndrome extraction, idle noise, and recovery toggles.
        """
        if include_psi:
            circ = self.psi()
        else:
            circ = stim.Circuit()

        anc_ids = list(range(self.num_qubits, self.num_qubits + self.ancillas))
        total = self.num_qubits + self.ancillas
        all_qs = set(range(total))

        # Syndrome extraction
        for start in range(0, len(self.stabilizers), self.ancillas):
            stabs = self.stabilizers[start:start + self.ancillas]
            ancs = anc_ids[:len(stabs)]

            # prepare ancillas
            self._spam_gate(circ, 'R', ancs)
            self._gate(circ, 'H', ancs)

            # measure each stabilizer
            active = set(ancs)
            for anc, stab in zip(ancs, stabs):
                for term in stab.split():
                    p, d = term[0], int(term[1:])
                    active.add(d)
                    g = {'X': 'CX', 'Y': 'CY', 'Z': 'CZ'}[p]
                    self._gate(circ, g, [anc, d])

            # finalize measurement
            self._gate(circ, 'H', ancs)
            # SPAM noise before measurement
            self._spam_gate(circ, 'M', ancs)
            active.update(ancs)

            # idle noise and tick
            idle = [q for q in all_qs if q not in active]
            self.add_idle_noise(circ, idle, self.p_x, self.p_y, self.p_z)
            circ.append_operation('TICK')

        # Recovery toggles
        for idx, ops in enumerate(self.toggles):
            rec_target = stim.target_rec(-(len(self.stabilizers) - idx))
            active = set()
            for op in ops.split():
                p, d = op[0], int(op[1:])
                active.add(d)
                g = {'X': 'CX', 'Y': 'CY', 'Z': 'CZ'}[p]
                self._gate(circ, g, [rec_target, d])

            # idle noise and tick
            idle = [q for q in all_qs if q not in active]
            self.add_idle_noise(circ, idle, self.p_x, self.p_y, self.p_z)
            circ.append_operation('TICK')

        return circ