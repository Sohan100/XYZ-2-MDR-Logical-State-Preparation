"""

circuit_noise.py
----------------------------------------------------------------------------
Circuit-level Pauli noise models for the fault-tolerant MDR circuits.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class CircuitNoise:
    """
    Circuit-level Pauli noise applied by `FTMDRCircuit`.

    Attributes
    ----------
    p1 : float Single-qubit depolarizing probability (used for the basis
    changes that dress each native ZZ gate when `dress_1q` is set). p2 :
    float Two-qubit depolarizing probability after each entangling gate.
    p_prep : float Probability that a reset prepares the orthogonal state.
    p_meas : float Measurement bit-flip probability. p_idle : float
    Depolarizing probability on qubits that idle during an entangling layer.
    p_mem_z : float Dephasing probability per qubit per layer, applied to all
    qubits (trapped-ion memory/transport error). p_xtalk : float
    Depolarizing probability on every data qubit during each ancilla
    measure or reset step (mid-circuit measurement crosstalk). dress_1q :
    bool Whether `p1` is added on both qubits of every entangling gate.
    p_idle_mr : float or None Depolarizing probability on data qubits that
    idle while the ancillas are reset or measured. `None` means `p_idle`.

    Extraction variants (for the fair comparison with other codes, see
    docs/fair_comparison.md); every variant keeps circuit-level noise on
    every operation it uses:

    native : str "gates" (one ancilla per check, controlled Paulis),
    "pairs" (every check from noisy two-qubit Pauli product measurements,
    EM3), "hybrid" (every weight-2 check is one noisy pair measurement with
    the EM3 channel `p_mpp`; every other check uses an ancilla and gates with
    the gate noise of this model), or "phen" (phenomenological: noiseless
    extraction, `data_xyz` Pauli noise on every data qubit once per round, and
    every check outcome flipped with `p_meas`). link_reps : int Each weight-2
    link check is measured this many times per round (with the noise of its
    extraction every time); 1 is the standard circuit. link_noise : bool
    False makes every operation of the link checks noiseless (a diagnostic,
    not a hardware model). data_xyz : tuple (px, py, pz) of the
    phenomenological data noise.
    """

    p1: float = 0.0
    p2: float = 0.0
    p_prep: float = 0.0
    p_meas: float = 0.0
    p_idle: float = 0.0
    p_mem_z: float = 0.0
    p_xtalk: float = 0.0
    dress_1q: bool = False
    p2_paulis: tuple | None = None
    idle_xyz: tuple | None = None
    p_idle_mr: float | None = None
    xtalk_per_mcmr: bool = False
    p2_rzz_frame: bool = False
    p_mpp: float = 0.0
    native: str = "gates"
    name: str = ""
    link_reps: int = 1
    link_noise: bool = True
    data_xyz: tuple | None = None

    @staticmethod
    def hybrid(p: float) -> "CircuitNoise":
        """
        SD6 gates and native pair measurements on the same hardware: weight-2
        checks are single pair measurements with the EM3 channel at `p`
        (a random two-qubit Pauli and/or a result flip), every other check is
        measured with an ancilla and controlled Paulis under SD6 noise at `p`.
        A code without weight-2 checks runs exactly as under SD6.
        """
        return replace(CircuitNoise.uniform(p), p_mpp=p, native="hybrid", name="hyb")

    @staticmethod
    def phenomenological(p: float, eta: float = 0.5) -> "CircuitNoise":
        """
        Phenomenological noise (Srivastava et al., arXiv:2505.03691): before
        every round each data qubit suffers Pauli noise of total rate `p`
        (Z-biased with eta = p_z / (p_x + p_y); 0.5 is depolarizing) and every
        check outcome is flipped with probability `p`; the extraction is
        otherwise noiseless.
        """
        pz = p * eta / (eta + 1.0)
        px = py = (p - pz) / 2.0
        return CircuitNoise(p_meas=p, native="phen", data_xyz=(px, py, pz),
                            name="phen" if eta == 0.5 else f"phen_b{eta:g}")

    @staticmethod
    def uniform(p: float) -> "CircuitNoise":
        """
        Standard circuit-level depolarizing model with one rate `p` (SD6).
        """
        return CircuitNoise(p1=p, p2=p, p_prep=p, p_meas=p, p_idle=p,
                            name="sd6")

    @staticmethod
    def si1000(p: float) -> "CircuitNoise":
        """
        SI1000 rate assignment (Gidney et al.) for one parameter `p`.

        Two-qubit gates `p`, single-qubit gates and idling `p/10`, reset
        `2p`, measurement `5p`, and `2p` on data qubits that idle while the
        ancillas are reset or measured (resonator idle).
        """
        return CircuitNoise(p1=p / 10, p2=p, p_prep=2 * p, p_meas=5 * p,
                            p_idle=p / 10, p_idle_mr=2 * p, name="si1000")

    @staticmethod
    def em3(p: float) -> "CircuitNoise":
        """
        EM3 model of Gidney et al. for hardware with native pair measurements.

        Every two-qubit Pauli product measurement (MPP) is followed, with
        probability `p`, by an error drawn uniformly from
        {I, X, Y, Z}^2 x {flip, no flip}. Preparations and single-qubit
        measurements are flipped with probability `p`, and every qubit that
        is not used in a step suffers depolarizing noise `p`. With this model
        `FTMDRCircuit` measures the checks with pair measurements
        (`native="pairs"`) instead of controlled Pauli gates.
        """
        return CircuitNoise(p1=p, p2=0.0, p_prep=p, p_meas=p, p_idle=p,
                            p_mpp=p, native="pairs", name="em3")

    @staticmethod
    def biased(p: float, eta: float) -> "CircuitNoise":
        r"""
        Z-biased circuit noise with total rate `p` at every location.

        Single-qubit locations use $p_z = p\eta/(\eta+1)$ and
        $p_x = p_y = p/(2(\eta+1))$, so $\eta = p_z/(p_x+p_y)$. Two-qubit
        gates put $p\eta/(\eta+1)$ on $IZ$, $ZI$, $ZZ$ and spread
        $p/(\eta+1)$ over the other twelve Paulis. Reset and measurement
        flips stay at `p`. `eta = 0.5` is depolarizing.
        """
        hi = p * eta / (eta + 1.0)
        lo = p / (eta + 1.0)
        z_type = {2, 11, 14}  # IZ, ZI, ZZ in Stim's PAULI_CHANNEL_2 order
        p2 = tuple(hi / 3 if k in z_type else lo / 12 for k in range(15))
        idle = (lo / 2, lo / 2, hi)
        return CircuitNoise(p1=p, p2=p, p_prep=p, p_meas=p, p_idle=p,
                            p2_paulis=p2, idle_xyz=idle,
                            name=f"biased_eta{eta:g}")

    @staticmethod
    def quantinuum(machine: str = "helios", scale: float = 1.0,
                   memory_scale: float = 1.0,
                   two_qubit: str = "depolarizing") -> "CircuitNoise":
        """
        Data-sheet noise for Quantinuum trapped-ion machines.

        The data sheets quote average infidelities r from randomized
        benchmarking. A depolarizing channel on n qubits with total Pauli
        error probability p has r = p d / (d + 1) with d = 2^n, so
        `p1 = 3/2 r1` and `p2 = 5/4 r2`. The memory error per qubit at an
        average depth-1 circuit is dephasing, and a Z flip with probability p
        has r = 2p/3, so `p_mem_z = 3/2 r_mem`, applied to every qubit in
        every gate layer and to the data qubits in every measurement step. The
        SPAM error is a classical outcome error and is split evenly between
        preparation and measurement. The mid-circuit measurement crosstalk
        error r_xt is the infidelity of every spectator qubit per measured and
        reset qubit, applied as depolarizing noise with `p_xt = 3/2 r_xt` once
        for every ancilla that is measured (`xtalk_per_mcmr`).

        - helios (data sheet v1.2, June 2026, typical): r1 3e-5, r2 8e-4,
          SPAM 5e-4, memory 6e-4, crosstalk 5e-5.
        - h2 (data sheet v2.00, October 2024, typical): r1 3e-5, r2 1.5e-3,
          SPAM 1.5e-3, memory 5e-4, crosstalk 1e-5.

        `two_qubit="cb"` (Helios only) replaces the depolarizing two-qubit
        channel by the Pauli error rates of the native RZZ(pi/2) gate measured
        by cycle benchmarking (Ransford et al., Table A4), mapped to the frame
        of each controlled Pauli. `scale` multiplies every rate.
        """
        if machine == "helios":
            r1, r2, spam, mem, xt = 3e-5, 8e-4, 5e-4, 6e-4, 5e-5
        elif machine == "h2":
            r1, r2, spam, mem, xt = 3e-5, 1.5e-3, 1.5e-3, 5e-4, 1e-5
        else:
            raise ValueError("machine must be 'helios' or 'h2'.")
        s = float(scale)
        p2_paulis = None
        if two_qubit == "cb":
            if machine != "helios":
                raise ValueError("cycle-benchmarking rates exist for Helios only.")
            p2_paulis = tuple(s * v for v in HELIOS_RZZ_CB)
        elif two_qubit != "depolarizing":
            raise ValueError("two_qubit must be 'depolarizing' or 'cb'.")
        return CircuitNoise(
            p1=1.5 * r1 * s,
            p2=1.25 * r2 * s,
            p_prep=0.5 * spam * s,
            p_meas=0.5 * spam * s,
            p_idle=0.0,
            p_mem_z=1.5 * mem * s * memory_scale,
            p_xtalk=1.5 * xt * s,
            dress_1q=True,
            p2_paulis=p2_paulis,
            p2_rzz_frame=p2_paulis is not None,
            xtalk_per_mcmr=True,
            name=machine if two_qubit == "depolarizing" else machine + "_cb",
        )

    @staticmethod
    def trapped_ion(p: float, machine: str = "helios", crosstalk: bool = True,
                    two_qubit: str = "depolarizing") -> "CircuitNoise":
        """
        One-parameter Quantinuum model in the style of SI1000.

        `p` is the two-qubit depolarizing probability. Every other rate keeps
        the ratio to `p` that it has on the data sheet of `machine`
        (`quantinuum`), so the machine itself sits at `p = QUANTINUUM_P2[machine]`,
        1.0e-3 for Helios and 1.875e-3 for H2. In units of `p`:

        - helios: p1 0.045, preparation and measurement flips 0.25 each,
          memory dephasing 0.9, crosstalk 0.075 per measured ancilla;
        - h2: p1 0.024, preparation and measurement flips 0.4 each, memory
          dephasing 0.4, crosstalk 0.008 per measured ancilla.

        `crosstalk=False` removes the measurement crosstalk.
        """
        if machine not in QUANTINUUM_P2:
            raise ValueError("machine must be 'helios' or 'h2'.")
        noise = CircuitNoise.quantinuum(machine, scale=p / QUANTINUUM_P2[machine],
                                        two_qubit=two_qubit)
        if not crosstalk:
            noise = replace(noise, p_xtalk=0.0, name=noise.name + "_noxt")
        return noise


# two-qubit depolarizing probability p2 = 5/4 r2 at the typical data-sheet rates
QUANTINUUM_P2 = {"helios": 1.25 * 8e-4, "h2": 1.25 * 1.5e-3}


# Pauli error rates of the Helios RZZ(pi/2) gate from cycle benchmarking
# (Ransford et al., arXiv:2511.05465, Table A4), in Stim's PAULI_CHANNEL_2
# order IX IY IZ XI XX XY XZ YI YX YY YZ ZI ZX ZY ZZ, with the first Pauli on
# the control (ancilla) and the second on the target (data qubit), in the
# frame of the RZZ gate. FTMDRCircuit maps the target Pauli to the frame of
# each controlled Pauli.
HELIOS_RZZ_CB = (
    4.5e-5, 4.5e-5, 19e-5,          # IX IY IZ
    5.8e-5, 0.06e-5, 0.06e-5, 5.8e-5,  # XI XX XY XZ
    5.8e-5, 0.06e-5, 0.06e-5, 5.8e-5,  # YI YX YY YZ
    19e-5, 4.5e-5, 4.5e-5, 5.9e-5,  # ZI ZX ZY ZZ
)
