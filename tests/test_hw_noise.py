"""Tests for the one-parameter models of the comparison with Hetenyi and Wootton (arXiv:2306.17786):
hw_g (gates only), hw_g1q (gates with noisy single-qubit basis changes), hw_t (idling only, eta_T = 20) and
hw_rr (readout only) in scripts/run_decoder_threshold_sweep.py. Each model must put errors where it says and
nowhere else, in XYZ^2 and in the competitor circuits."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import stim

from mdr.ft import BOTH_BASES_SCHEDULE, DEPTH6_SCHEDULE, FTMDRCircuit
from mdr.ft.competitor_circuits import competitor_circuit
from mdr.ft.competitor_decoders import competitor_decoder
from mdr.ft.two_level_decoder import TwoLevelDecoder

ROOT = Path(__file__).resolve().parents[1]
HAS_LDPC = importlib.util.find_spec("ldpc") is not None


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SWEEP = _load_script("run_decoder_threshold_sweep")
NOISE = SWEEP.NOISE
HW = SWEEP.HW_NOISES
P = 1e-2
NOISE_OPS = {"DEPOLARIZE1", "DEPOLARIZE2", "PAULI_CHANNEL_1", "PAULI_CHANNEL_2", "X_ERROR", "Y_ERROR",
             "Z_ERROR", "E", "ELSE_CORRELATED_ERROR", "HERALDED_ERASE", "HERALDED_PAULI_CHANNEL_1"}
TWO_QUBIT_GATES = {"CX", "CY", "CZ"}
MEASUREMENTS = {"M", "MX", "MY"}


def _circuits(noise):
    """(label, circuit) of every code the comparison runs, small enough for a test."""
    out = []
    for sched_name, sched in (("depth6", DEPTH6_SCHEDULE), ("both", BOTH_BASES_SCHEDULE)):
        for logical in ("X", "Y"):
            if (sched_name, logical) == ("depth6", "Y"):
                continue                      # the depth-6 schedule has distance (d + 1) / 2 in Logical Y
            ft = FTMDRCircuit(3, 2, noise, schedule=sched, final="frame", detectors="combined", logical=logical)
            out.append((f"xyz2-{sched_name}-{logical}", ft.build()))
    for code in ("css", "xzzx"):
        for basis in ("X", "Z"):
            out.append((f"{code}-{basis}", competitor_circuit(code, 3, 2, noise, basis)))
    for basis in ("X", "Z"):
        out.append((f"honeycomb-{basis}", competitor_circuit("honeycomb", 4, 2, noise, basis)))
    return out


def _channels(c: stim.Circuit) -> set:
    return {(inst.name, tuple(round(a, 15) for a in inst.gate_args_copy()))
            for inst in c.flattened() if inst.name in NOISE_OPS}


def _qubits(inst) -> list:
    return [t.value for t in inst.targets_copy() if t.is_qubit_target]


def test_hw_names_are_registered():
    assert HW == ["hw_g", "hw_g1q", "hw_t", "hw_rr"]
    for name in HW:
        assert NOISE[name](P).name == name
    comp = _load_script("campaign")
    assert comp.split_noise("hw_t@both-Y") == ("hw_t", "both", "Y")


def test_hw_t_has_their_bias_and_total_rate():
    px, py, pz = NOISE["hw_t"](P).idle_xyz
    assert px == py
    assert px + py + pz == pytest.approx(P)
    assert pz / (px + py) == pytest.approx(20.0)        # eta_T = p_T2 / p_T1, p_T1 = p_X + p_Y
    assert NOISE["hw_t"](P).p_idle_mr is None           # the same channel while the ancillas are reset and read


@pytest.mark.parametrize("name", HW)
def test_hw_models_have_one_kind_of_error(name):
    nz = NOISE[name](P)
    rates = dict(p_prep=nz.p_prep, p_meas=nz.p_meas, p_idle=nz.p_idle, p_mem_z=nz.p_mem_z, p_xtalk=nz.p_xtalk,
                 p_mpp=nz.p_mpp, p2=nz.p2)
    on = {k for k, v in rates.items() if v}
    assert on == {"hw_g": {"p2"}, "hw_g1q": {"p2"}, "hw_t": set(), "hw_rr": {"p_meas"}}[name]
    assert nz.native == "gates" and nz.link_reps == 1 and nz.link_noise
    assert nz.p2_paulis is None and nz.data_xyz is None
    assert nz.dress_1q == (name == "hw_g1q")
    assert (nz.idle_xyz is not None) == (name == "hw_t")


EXPECTED = {
    "hw_g": {("DEPOLARIZE2", (P,))},
    "hw_g1q": {("DEPOLARIZE2", (P,)), ("DEPOLARIZE1", (P,))},
    "hw_t": {("PAULI_CHANNEL_1", tuple(round(x, 15) for x in (P / 42, P / 42, 20 * P / 21)))},
}


@pytest.mark.parametrize("name", HW)
def test_only_the_intended_channels_appear(name):
    for label, c in _circuits(NOISE[name](P)):
        ch = _channels(c)
        if name == "hw_rr":
            assert ch and ch <= {("X_ERROR", (P,)), ("Z_ERROR", (P,))}, label
        else:
            assert ch == EXPECTED[name], (label, ch)


@pytest.mark.parametrize("name", ["hw_g", "hw_g1q"])
def test_gate_noise_sits_on_the_gates(name):
    """Every gate channel acts on exactly the qubits of the entangling gates just before it."""
    for label, c in _circuits(NOISE[name](P)):
        gate_q: list = []
        for inst in c.flattened():
            if inst.name in TWO_QUBIT_GATES:
                gate_q += _qubits(inst)
            elif inst.name in ("DEPOLARIZE2", "DEPOLARIZE1"):
                assert gate_q and sorted(_qubits(inst)) == sorted(gate_q), (label, inst)
            elif inst.name == "TICK":
                gate_q = []
            else:
                assert inst.name not in NOISE_OPS, (label, inst)


def test_readout_noise_sits_on_the_measurements():
    """hw_rr: every flip is followed by a measurement of the same qubits, and every measurement has a flip."""
    for label, c in _circuits(NOISE["hw_rr"](P)):
        ops = [inst for inst in c.flattened() if inst.name not in ("DETECTOR", "OBSERVABLE_INCLUDE",
                                                                    "QUBIT_COORDS", "SHIFT_COORDS")]
        flipped = set()
        for a, b in zip(ops, ops[1:]):
            if a.name in ("X_ERROR", "Z_ERROR"):
                assert b.name in MEASUREMENTS and set(_qubits(a)) <= set(_qubits(b)), (label, a, b)
                assert a.name == ("Z_ERROR" if b.name == "MX" else "X_ERROR"), (label, a, b)
                flipped |= set(_qubits(a))
        measured = {q for inst in ops if inst.name in MEASUREMENTS for q in _qubits(inst)}
        assert measured and measured == flipped, label


def test_idle_noise_reaches_data_during_ancilla_reset_and_readout():
    """hw_t: the data qubits get the idle channel while the ancillas are reset (p_idle_mr = None)."""
    nz = NOISE["hw_t"](P)
    ft = FTMDRCircuit(3, 2, nz, schedule=BOTH_BASES_SCHEDULE, detectors="combined", logical="Y")
    data = set(range(ft.geometry.n))
    moment: list = []
    seen = 0
    for inst in ft.build().flattened():
        if inst.name == "TICK":
            names = [i.name for i in moment]
            if "RX" in names and set(_qubits(moment[names.index("RX")])) == set(ft.ancillas):
                idle = {q for i in moment if i.name == "PAULI_CHANNEL_1" for q in _qubits(i)}
                assert data <= idle
                seen += 1
            moment = []
        else:
            moment.append(inst)
    assert seen == ft.rounds


# well below each model's threshold (about 0.5%, 0.3%, 4% and 15% for XYZ^2 in the spin-qubit study)
LOW = {"hw_g": 2e-3, "hw_g1q": 1e-3, "hw_t": 1e-2, "hw_rr": 1e-2}


@pytest.mark.parametrize("name", HW)
@pytest.mark.parametrize("logical", ["X", "Y"])
def test_xyz2_decoders_run(name, logical):
    """The decoders of the comparison (mwpm, corr_gauge, bm) build and decode under every model."""
    ft = FTMDRCircuit(3, 3, NOISE[name](LOW[name]), schedule=BOTH_BASES_SCHEDULE, final="frame",
                      detectors="combined", logical=logical)
    dets, obs = ft.build().compile_detector_sampler(seed=5).sample(400, separate_observables=True)
    assert dets.any()
    for dec in ("mwpm", "corr_gauge") + (("bm",) if HAS_LDPC else ()):
        out = TwoLevelDecoder(ft, **SWEEP.DECODERS[dec]).decode_batch(dets)
        assert out.shape == obs.shape
        assert np.mean(np.any(out != obs, axis=1)) < 0.25, dec


@pytest.mark.parametrize("name", HW)
@pytest.mark.parametrize("code,d", [("css", 3), ("xzzx", 3), ("honeycomb", 4)])
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_competitor_decoders_run(name, code, d, basis):
    # the honeycomb's thresholds are lower (about 0.4%, 1.2% and 1.0% for gates, idling and readout)
    c = competitor_circuit(code, d, 3, NOISE[name](LOW[name] / (5 if code == "honeycomb" else 1)), basis)
    dets, obs = c.compile_detector_sampler(seed=5).sample(400, separate_observables=True)
    assert dets.any()
    for dec in ("mwpm", "corr") + (("bm",) if HAS_LDPC else ()):
        out = competitor_decoder(c, dec).decode_batch(dets)
        assert out.shape == obs.shape
        assert np.mean(np.any(out != obs, axis=1)) < 0.25, dec


@pytest.mark.parametrize("name", ["hw_g", "hw_g1q", "hw_rr"])
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_css_and_xzzx_have_the_same_error_model(name, basis):
    """These models are invariant under single-qubit Cliffords, so XZZX is CSS in another frame: the same
    mechanisms with the same rates and the same graphlike distance. Stim splits a few faults into different
    graphlike pieces in the two frames (the decomposed models differ by a handful of components)."""
    for d in (3, 5):
        circ = {code: competitor_circuit(code, d, d, NOISE[name](P), basis) for code in ("css", "xzzx")}
        full = {k: c.detector_error_model(approximate_disjoint_errors=True) for k, c in circ.items()}
        assert full["xzzx"].approx_equals(full["css"], atol=1e-12)
        assert full["xzzx"].num_errors == full["css"].num_errors
        dist = {k: len(c.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
                       .shortest_graphlike_error()) for k, c in circ.items()}
        assert dist["css"] == dist["xzzx"] == d


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_honeycomb_readout_faults_are_hyperedges(basis):
    """Under readout errors alone a flipped edge outcome of the honeycomb code sets off three or four
    detectors, and no other fault gives Stim the graphlike pieces to split it into (the matching decoders
    then use competitor_decoders.hinted_decomposition)."""
    c = competitor_circuit("honeycomb", 4, 3, NOISE["hw_rr"](P), basis)
    dem = c.detector_error_model(approximate_disjoint_errors=True)
    sizes = {sum(t.is_relative_detector_id() for t in inst.targets_copy())
             for inst in dem.flattened() if inst.type == "error"}
    assert max(sizes) >= 3
    with pytest.raises(ValueError, match="decompose"):
        c.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_hinted_decomposition_keeps_the_faults(basis):
    """The fallback of the matching decoders: every fault of the circuit, with its probability, split
    into graphlike pieces."""
    from mdr.ft.competitor_decoders import DECOMPOSITION_HINT, _symptoms, hinted_decomposition

    c = competitor_circuit("honeycomb", 8, 4, NOISE["hw_rr"](P), basis)
    real = _symptoms(c.detector_error_model(approximate_disjoint_errors=True))
    dem = hinted_decomposition(c)
    got = _symptoms(dem)
    assert got.keys() == real.keys()
    assert max(abs(got[k] - real[k]) for k in real) < 100 * DECOMPOSITION_HINT
    for inst in dem.flattened():
        if inst.type == "error":
            comps = [[]]
            for t in inst.targets_copy():
                if t.is_separator():
                    comps.append([])
                elif t.is_relative_detector_id():
                    comps[-1].append(t.val)
            assert all(len(x) <= 2 for x in comps)
    assert len(dem.shortest_graphlike_error()) > 0


@pytest.mark.parametrize("code,d,name", [("honeycomb", 4, "hw_g"), ("css", 3, "hw_rr"), ("honeycomb", 4, "sd6")])
def test_fallback_only_where_stim_fails(code, d, name):
    """Models that Stim decomposes keep Stim's own decomposition."""
    c = competitor_circuit(code, d, 3, NOISE[name](P), "X")
    dec = competitor_decoder(c, "mwpm")
    assert dec.dem == c.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)


@pytest.mark.parametrize("d", [5, 7])
def test_xyz2_both_bases_graphlike_distance(d):
    """The both-bases schedule under gate noise: d + 1 for Logical X and d for Logical Y."""
    for logical, want in (("X", d + 1), ("Y", d)):
        ft = FTMDRCircuit(d, d, NOISE["hw_g"](P), schedule=BOTH_BASES_SCHEDULE, final="frame",
                          detectors="combined", logical=logical)
        dec = TwoLevelDecoder(ft, **SWEEP.DECODERS["mwpm"])
        dem = dec.s0_circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
        assert len(dem.shortest_graphlike_error()) == want, logical


def test_hw_tr_idles_only_during_reset_and_readout():
    """The diagnostic hw_tr: the biased channel on every data qubit at the reset and the readout of the
    ancillas, nowhere else."""
    p = P
    nz = NOISE["hw_tr"](p)
    assert nz.idle_mr_xyz[2] / (nz.idle_mr_xyz[0] + nz.idle_mr_xyz[1]) == pytest.approx(20.0)
    assert 2 * sum(nz.idle_mr_xyz) == pytest.approx(p)
    ft = FTMDRCircuit(3, 2, nz, schedule=BOTH_BASES_SCHEDULE, detectors="combined", logical="X")
    data = set(range(ft.geometry.n))
    hits = 0
    for inst in ft.build().flattened():
        if inst.name in NOISE_OPS:
            assert inst.name == "PAULI_CHANNEL_1" and set(_qubits(inst)) == data, inst
            hits += 1
    assert hits == 2 * ft.rounds
    for label, c in _circuits(nz):
        assert _channels(c) == {("PAULI_CHANNEL_1", tuple(round(x, 15) for x in nz.idle_mr_xyz))}, label


@pytest.mark.parametrize("name", SWEEP.GAP_NOISES)
def test_gap_models_have_no_idle_errors(name):
    """sd6_noidle and sd6_il_noidle: SD6 with every idle error removed (and noiseless links for the second)."""
    nz = NOISE[name](P)
    assert nz.p_idle == 0.0 and nz.p_idle_mr == 0.0 and nz.idle_xyz is None
    assert (nz.p1, nz.p2, nz.p_prep, nz.p_meas) == (P, P, P, P)
    assert nz.link_noise == (name == "sd6_noidle")
    for logical in ("X", "Y"):
        c = FTMDRCircuit(3, 2, nz, schedule=BOTH_BASES_SCHEDULE, detectors="combined", logical=logical).build()
        ch = _channels(c)
        assert ("DEPOLARIZE2", (P,)) in ch
        assert ch <= {("DEPOLARIZE2", (P,)), ("X_ERROR", (P,)), ("Z_ERROR", (P,))}, ch


def test_three_d_rounds():
    comp = _load_script("campaign")
    assert comp.n_rounds("3d", 5) == 15 and comp.n_rounds("d", 5) == 5 and comp.n_rounds("4", 5) == 4
