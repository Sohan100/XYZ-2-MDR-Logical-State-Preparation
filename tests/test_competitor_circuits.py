"""Tests for the competitor-code circuits and decoders (src/mdr/ft/competitor_*.py)."""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import stim

import mdr.ft.competitor_circuits as cc
from mdr.ft import CircuitNoise, FTMDRCircuit
from mdr.ft.competitor_circuits import (CODES, CompetitorCircuit, HoneycombCircuit,
                                        competitor_circuit)
from mdr.ft.competitor_decoders import DECODERS, competitor_decoder, merged_mechanisms

ROOT = Path(__file__).resolve().parents[1]
HAS_LDPC = importlib.util.find_spec("ldpc") is not None
HAS_TESSERACT = importlib.util.find_spec("tesseract_decoder") is not None
HAS_BELIEFMATCHING = importlib.util.find_spec("beliefmatching") is not None


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# the named noise models of the threshold sweeps
NOISE = _load_script("run_decoder_threshold_sweep").NOISE
# models whose parameter scales a device point instead of being a probability
SCALED = {"helios", "h2", "helios_cb"}


def _noise(name: str, p: float = 2e-3) -> CircuitNoise:
    return NOISE[name](1.0 if name in SCALED else p)


def _is_link_variant(name: str) -> bool:
    """XYZ^2's link variants (repeated or noiseless links), which these codes reject."""
    noise = _noise(name)
    return noise.link_reps != 1 or not noise.link_noise


LINK_VARIANTS = sorted(name for name in NOISE if _is_link_variant(name))
BUILDING = sorted(name for name in NOISE if not _is_link_variant(name))
PHEN = sorted(name for name in NOISE if _noise(name).native == "phen")


def _fault_distance(circuit: stim.Circuit, max_set: int = 6) -> int:
    return len(circuit.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=max_set,
        dont_explore_edges_with_degree_above=9999,
        dont_explore_edges_increasing_symptom_degree=False,
    ))


def _infinite_bias(p: float) -> CircuitNoise:
    """purez with the non-Z gate and idle components set to exactly zero."""
    z_type = {2, 11, 14}  # IZ, ZI, ZZ
    return dataclasses.replace(
        NOISE["purez"](p), p2_paulis=tuple(p / 3 if k in z_type else 0.0 for k in range(15)),
        idle_xyz=(0.0, 0.0, p))


def _single_faults(circuit: stim.Circuit, min_p: float = 0.0):
    dem = circuit.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
    keys, priors, _ = merged_mechanisms(dem)
    keys = [k for k, p in zip(keys, priors) if p >= min_p]
    syn = np.zeros((len(keys), circuit.num_detectors), dtype=np.uint8)
    flips = np.zeros((len(keys), circuit.num_observables), dtype=bool)
    for j, (dets, obs) in enumerate(keys):
        syn[j, list(dets)] = 1
        flips[j, list(obs)] = True
    return syn, flips


# noiseless versions of every extraction (noise model and compilation)
NOISELESS = {
    "gates": (CircuitNoise(), {}),
    "pairs": (CircuitNoise.em3(0.0), {}),
    "pairs_direct": (CircuitNoise.em3(0.0), {"weight2": "direct"}),
    "hybrid": (CircuitNoise.hybrid(0.0), {}),
    "hybrid_direct": (CircuitNoise.hybrid(0.0), {"weight2": "direct"}),
    "phen": (CircuitNoise.phenomenological(0.0), {}),
}


# ---------------------------------------------------------------- circuits
@pytest.mark.parametrize("code", CODES)
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("native", sorted(NOISELESS))
@pytest.mark.parametrize("d,rounds", [(3, 1), (3, 3), (5, 2)])
def test_noiseless_circuit_is_deterministic(code, basis, native, d, rounds):
    noise, kw = NOISELESS[native]
    c = competitor_circuit(code, d, rounds, noise, basis, **kw)
    dets, obs = c.compile_detector_sampler(seed=1).sample(200, separate_observables=True)
    assert c.num_observables == 1
    assert c.num_detectors == (d * d - 1) // 2 * 2 + (rounds - 1) * (d * d - 1)
    assert not dets.any()
    assert not obs.any()


@pytest.mark.parametrize("code", CODES)
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("noise", ["sd6", "purez", "em3", "hyb", "phen", "phen_b10"])
@pytest.mark.parametrize("d", [3, 5])
def test_fault_distance_is_d(code, basis, noise, d):
    c = competitor_circuit(code, d, d, _noise(noise, 1e-3), basis)
    assert len(c.shortest_graphlike_error()) == d
    assert _fault_distance(c) == d


@pytest.mark.parametrize("code", CODES)
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("d", [3, 5])
def test_dominant_faults_of_purez_keep_distance_d(code, basis, d):
    # only the Z-type gate and idle faults and the preparation and readout flips
    c = competitor_circuit(code, d, d, _infinite_bias(1e-3), basis)
    assert _fault_distance(c) == d


@pytest.mark.parametrize("code", CODES)
@pytest.mark.parametrize("noise", ["sd6", "purez", "em3"])
def test_exact_fault_distance_d3(code, noise):
    pytest.importorskip("ortools")
    efd = _load_script("exact_fault_distance")
    c = competitor_circuit(code, 3, 3, _noise(noise, 1e-3), "X")
    assert efd.exists_logical_error(c, 2, time_limit=120)[0] == "INFEASIBLE"
    assert efd.exists_logical_error(c, 3, time_limit=120)[0] in ("OPTIMAL", "FEASIBLE")


def test_hook_unsafe_orders_lose_distance(monkeypatch):
    # the distance tests are sensitive to the order: swapping the two shapes
    # puts every hook along a logical operator
    monkeypatch.setitem(cc.SCHEDULE, "X", ("NW", "SW", "NE", "SE"))
    monkeypatch.setitem(cc.SCHEDULE, "Z", ("NW", "NE", "SW", "SE"))
    c = competitor_circuit("css", 5, 5, CircuitNoise.uniform(1e-3), "X")
    assert _fault_distance(c, 4) == 3
    monkeypatch.undo()
    monkeypatch.setitem(cc.CAT_STAR, "X", ("NW", "NE", "SE", "SW"))
    monkeypatch.setitem(cc.CAT_STAR, "Z", ("NW", "SW", "SE", "NE"))
    c = competitor_circuit("css", 5, 5, CircuitNoise.em3(1e-3), "X")
    assert _fault_distance(c, 4) == 3


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_direct_weight2_pair_measurements_halve_the_distance(basis):
    c = CompetitorCircuit("css", 5, 5, CircuitNoise.em3(1e-3), basis=basis,
                          weight2="direct").build()
    assert _fault_distance(c, 4) == 3


# ---------------------------------------------------------------- hybrid hardware
@pytest.mark.parametrize("code", CODES)
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("d,rounds", [(3, 3), (5, 2)])
def test_hybrid_default_is_the_sd6_circuit(code, basis, d, rounds):
    # the best compilation of a surface code on gates + pair measurements is all gates
    hyb = competitor_circuit(code, d, rounds, _noise("hyb", 1e-3), basis)
    sd6 = competitor_circuit(code, d, rounds, _noise("sd6", 1e-3), basis)
    assert str(hyb) == str(sd6)
    assert hyb.detector_error_model(approximate_disjoint_errors=True) == \
        sd6.detector_error_model(approximate_disjoint_errors=True)
    # all pairs on the same hardware is the em3 circuit
    cat = competitor_circuit(code, d, rounds, _noise("hyb", 1e-3), basis, weight2="cat")
    assert str(cat) == str(competitor_circuit(code, d, rounds, _noise("em3", 1e-3), basis))


@pytest.mark.parametrize("code", CODES)
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("d", [3, 5])
def test_hybrid_direct_boundary_pair_measurements_lose_distance(code, basis, d):
    # one pair measurement per weight-2 boundary check: its correlated two-qubit
    # error lies along the boundary, so the fault distance drops to (d + 1) / 2
    c = competitor_circuit(code, d, d, _noise("hyb", 1e-3), basis, weight2="direct")
    assert len(c.shortest_graphlike_error()) == (d + 1) // 2
    assert _fault_distance(c, 4) == (d + 1) // 2
    ops = {inst.name for inst in c.flattened()}
    assert "MPP" in ops and {"CX", "CY", "CZ"} & ops  # pairs on the boundary, gates in the bulk


@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("d,rounds", [(4, 2), (8, 2)])
def test_honeycomb_hybrid_is_its_pair_circuit(basis, d, rounds):
    hyb = competitor_circuit("honeycomb", d, rounds, _noise("hyb", 1e-3), basis)
    em3 = competitor_circuit("honeycomb", d, rounds, _noise("em3", 1e-3), basis)
    assert str(hyb) == str(em3)
    gates = competitor_circuit("honeycomb", d, rounds, _noise("hyb", 1e-3), basis, weight2="gates")
    assert str(gates) == str(competitor_circuit("honeycomb", d, rounds, _noise("sd6", 1e-3), basis))


def test_compilations_are_checked():
    p = 1e-3
    with pytest.raises(ValueError, match="compilation"):
        competitor_circuit("css", 3, 3, CircuitNoise.uniform(p), weight2="direct")
    with pytest.raises(ValueError, match="compilation"):
        competitor_circuit("css", 3, 3, CircuitNoise.em3(p), weight2="gates")
    with pytest.raises(ValueError, match="compilation"):
        competitor_circuit("xzzx", 3, 3, CircuitNoise.phenomenological(p), weight2="gates")
    with pytest.raises(ValueError, match="compilation"):
        competitor_circuit("honeycomb", 4, 2, CircuitNoise.em3(p), weight2="cat")
    with pytest.raises(ValueError, match="compilation"):
        competitor_circuit("honeycomb", 4, 2, CircuitNoise.uniform(p), weight2="direct")
    with pytest.raises(ValueError, match="native"):
        competitor_circuit("css", 3, 3, dataclasses.replace(CircuitNoise.uniform(p), native="magic"))


# ---------------------------------------------------------------- link variants of XYZ^2
@pytest.mark.parametrize("name", LINK_VARIANTS)
@pytest.mark.parametrize("code", CODES + ("honeycomb",))
def test_link_variants_are_rejected(name, code):
    d = 4 if code == "honeycomb" else 3
    with pytest.raises(ValueError, match="no XX link checks"):
        competitor_circuit(code, d, 2, _noise(name), "X")


def test_link_options_raise_on_their_own():
    for noise in (dataclasses.replace(CircuitNoise.uniform(1e-3), link_reps=2),
                  dataclasses.replace(CircuitNoise.em3(1e-3), link_noise=False),
                  dataclasses.replace(CircuitNoise.phenomenological(1e-2), link_reps=3)):
        with pytest.raises(ValueError, match="no XX link checks"):
            CompetitorCircuit("css", 3, 3, noise)
        with pytest.raises(ValueError, match="no XX link checks"):
            HoneycombCircuit(4, 2, noise)


# ---------------------------------------------------------------- phenomenological noise
def _phen_census(c: stim.Circuit):
    """(noise instructions, MPP flip probabilities and product counts, other measurements)."""
    noise_ops, mpps, meas = [], [], []
    for inst in c.flattened():
        if inst.name in _NOISE_OPS:
            noise_ops.append((inst.name, tuple(inst.gate_args_copy()), len(inst.targets_copy())))
        elif inst.name == "MPP":
            n_prod = sum(1 for t in inst.targets_copy() if not t.is_combiner) - \
                sum(1 for t in inst.targets_copy() if t.is_combiner)
            mpps.append((tuple(inst.gate_args_copy()), n_prod))
        elif inst.name in ("M", "MX", "MY", "MR", "MRX", "MRY"):
            meas.append((inst.name, tuple(inst.gate_args_copy())))
    return noise_ops, mpps, meas


@pytest.mark.parametrize("name", PHEN)
@pytest.mark.parametrize("code", CODES + ("honeycomb",))
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_phenomenological_noise_is_data_noise_and_outcome_flips(name, code, basis):
    p, rounds = 0.02, 3
    noise = _noise(name, p)
    d = 4 if code == "honeycomb" else 3
    c = competitor_circuit(code, d, rounds, noise, basis)
    n = d * d if code != "honeycomb" else d * 6
    assert c.num_qubits == n  # no ancillas or auxiliary qubits
    noise_ops, mpps, meas = _phen_census(c)
    steps = 3 * rounds if code == "honeycomb" else rounds  # a honeycomb round has three sub-rounds
    checks = n // 2 if code == "honeycomb" else d * d - 1
    # one layer of data noise on every data qubit before every step, nothing else
    assert noise_ops == [("PAULI_CHANNEL_1", tuple(noise.data_xyz), n)] * steps
    assert sum(noise.data_xyz) == pytest.approx(p)
    # every check (edge) outcome flips with p_meas, once per step
    assert mpps == [((p,), checks)] * steps
    # noiseless preparation and readout of the data
    assert all(not args for _, args in meas)
    assert len(c.shortest_graphlike_error()) >= 3


@pytest.mark.parametrize("code", CODES)
def test_phenomenological_threshold_lies_between_two_and_five_percent(code):
    # MWPM under phen (data depolarizing p, outcome flips p): d = 5 beats d = 3 below the
    # threshold (about 3.5 % in this convention, 2.9 % when data and outcomes flip equally
    # often) and loses above it
    def rate(d, p):
        c = competitor_circuit(code, d, d, _noise("phen", p), "X")
        return competitor_decoder(c, "mwpm").estimate(max_shots=20000, max_errors=10**9,
                                                      batch=10000, seed=5).rate
    assert rate(5, 0.02) < rate(3, 0.02)
    assert rate(5, 0.05) > rate(3, 0.05)


@pytest.mark.parametrize("noise", ["sd6", "si1000", "em3", "phen"])
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_codes_are_clifford_images_of_css(noise, basis):
    dems = {code: competitor_circuit(code, 3, 3, _noise(noise), basis)
            .detector_error_model(approximate_disjoint_errors=True) for code in CODES}
    for code in CODES:
        assert dems[code].approx_equals(dems["css"], atol=1e-12)


@pytest.mark.parametrize("noise", ["purez", "biased10", "helios_p", "phen_b10"])
def test_biased_noise_distinguishes_the_codes(noise):
    dems = {code: competitor_circuit(code, 3, 3, _noise(noise), "X")
            .detector_error_model(approximate_disjoint_errors=True) for code in CODES}
    assert not dems["xzzx"].approx_equals(dems["css"], atol=1e-12)
    assert not dems["xy"].approx_equals(dems["css"], atol=1e-12)


@pytest.mark.parametrize("name", BUILDING)
@pytest.mark.parametrize("code", CODES)
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_every_noise_model_builds(name, code, basis):
    c = competitor_circuit(code, 3, 2, _noise(name), basis)
    dem = c.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    assert dem.num_detectors == c.num_detectors > 0
    assert dem.num_errors > 0


_NOISE_OPS = {"DEPOLARIZE1", "DEPOLARIZE2", "PAULI_CHANNEL_1", "PAULI_CHANNEL_2", "X_ERROR",
              "Y_ERROR", "Z_ERROR", "E", "ELSE_CORRELATED_ERROR"}


def _channels(c: stim.Circuit, xtalk: float):
    out = set()
    for inst in c.flattened():
        if inst.name in _NOISE_OPS:
            args = tuple(inst.gate_args_copy())
            if xtalk and inst.name == "DEPOLARIZE1" and args == (xtalk,):
                args = ("crosstalk",)
            out.add((inst.name, args))
    return out


@pytest.mark.parametrize("name", sorted(set(BUILDING) - set(PHEN)))
@pytest.mark.parametrize("code", CODES)
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_noise_channels_are_those_of_ftmdr(name, code, basis):
    """Every channel of the competitor circuit, with its rates, also appears in FTMDRCircuit.

    The phenomenological models are checked by
    test_phenomenological_noise_is_data_noise_and_outcome_flips."""
    noise = _noise(name)
    d = 3

    def xtalk(k):
        if not noise.p_xtalk:
            return 0.0
        return 0.75 * (1 - (1 - 4 * noise.p_xtalk / 3) ** k) if noise.xtalk_per_mcmr else noise.p_xtalk

    ours = _channels(FTMDRCircuit(d, 2, noise, detectors="combined").build(), xtalk(2 * d * d - 1))
    theirs = _channels(competitor_circuit(code, d, 2, noise, basis), xtalk(d * d - 1))
    assert theirs <= ours
    # what FTMDRCircuit has in addition: readout flips of Y and Z data qubits, gate
    # frames of controlled Paulis that the code does not use, and under hybrid the
    # pair-measurement errors of its links (the surface codes use gates only there)
    extra = {(n, a) for n, a in ours - theirs
             if not (n == "X_ERROR" and a in ((noise.p_prep,), (noise.p_meas,)))
             and not (n == "PAULI_CHANNEL_2" and noise.p2_rzz_frame)
             and not (noise.native == "hybrid" and n in ("E", "ELSE_CORRELATED_ERROR"))}
    assert not extra


def test_bad_arguments_are_rejected():
    with pytest.raises(ValueError):
        competitor_circuit("toric", 3, 3, CircuitNoise.uniform(1e-3))
    with pytest.raises(ValueError):
        competitor_circuit("css", 3, 3, CircuitNoise.uniform(1e-3), basis="Y")
    with pytest.raises(ValueError):
        competitor_circuit("css", 3, 0, CircuitNoise.uniform(1e-3))
    with pytest.raises(ValueError):
        competitor_circuit("honeycomb", 5, 3, CircuitNoise.uniform(1e-3))


# ---------------------------------------------------------------- honeycomb
# Detector count and graphlike fault distance of the reference circuits of
# Gidney et al. (github.com/Strilanc/honeycomb_threshold, styles SD6 and EM3,
# observables H and V, d x 6 ceil(d / 4) data qubits, 3 d sub-rounds), which
# the port reproduces. The honeycomb code's circuit distance is below its code
# distance d: EM3 pair-measurement faults act like two-qubit errors.
HONEYCOMB_REFERENCE = {
    4: (56, {("X", "sd6"): 3, ("Z", "sd6"): 4, ("X", "em3"): 2, ("Z", "em3"): 2}),
    6: (240, {("X", "sd6"): 6, ("Z", "sd6"): 6, ("X", "em3"): 4, ("Z", "em3"): 3}),
    8: (416, {("X", "sd6"): 6, ("Z", "sd6"): 8, ("X", "em3"): 4, ("Z", "em3"): 4}),
}


@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("native", ["gates", "pairs"])
@pytest.mark.parametrize("d,rounds", [(4, 1), (4, 2), (6, 3), (8, 2)])
def test_honeycomb_noiseless_circuit_is_deterministic(basis, native, d, rounds):
    noise = CircuitNoise() if native == "gates" else CircuitNoise.em3(0.0)
    c = competitor_circuit("honeycomb", d, rounds, noise, basis)
    dets, obs = c.compile_detector_sampler(seed=1).sample(100, separate_observables=True)
    assert c.num_observables == 1
    assert not dets.any()
    assert not obs.any()


@pytest.mark.parametrize("d", sorted(HONEYCOMB_REFERENCE))
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("noise", ["sd6", "em3", "purez"])
def test_honeycomb_matches_the_reference_circuits(d, basis, noise):
    c = competitor_circuit("honeycomb", d, d, _noise(noise, 1e-3), basis)
    n_det, dist = HONEYCOMB_REFERENCE[d]
    assert c.num_detectors == n_det
    # purez keeps every Pauli component (with a tiny rate), so its distance is that of sd6
    assert len(c.shortest_graphlike_error()) == dist[(basis, "em3" if noise == "em3" else "sd6")]


@pytest.mark.parametrize("noise", PHEN)
@pytest.mark.parametrize("basis,d,dist", [("X", 4, 3), ("Z", 4, 4), ("X", 6, 6), ("Z", 6, 6)])
def test_honeycomb_phenomenological_fault_distance(noise, basis, d, dist):
    # the distances of the honeycomb's gate circuits (HONEYCOMB_REFERENCE, sd6): a data error
    # before a sub-round or a flipped edge outcome is a single fault, as with native gates
    c = competitor_circuit("honeycomb", d, d, _noise(noise, 1e-3), basis)
    assert _fault_distance(c, 4) == dist


@pytest.mark.parametrize("name", BUILDING)
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_honeycomb_builds_with_every_noise_model(name, basis):
    noise = _noise(name)
    c = competitor_circuit("honeycomb", 4, 2, noise, basis)
    dem = c.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    assert dem.num_errors > 0
    if noise.native == "phen":  # see test_phenomenological_noise_is_data_noise_and_outcome_flips
        return
    if noise.native == "hybrid":  # the pair circuit: the channels of em3 at the same p
        noise = _noise("em3")

    def xtalk(k):
        if not noise.p_xtalk:
            return 0.0
        return 0.75 * (1 - (1 - 4 * noise.p_xtalk / 3) ** k) if noise.xtalk_per_mcmr else noise.p_xtalk

    # every channel, with its rate, also appears in FTMDRCircuit
    ours = _channels(FTMDRCircuit(3, 2, noise, detectors="combined").build(), xtalk(17))
    theirs = _channels(c, xtalk(12))  # 12 edges (ancillas) per sub-round at d = 4
    assert theirs <= ours


@pytest.mark.parametrize("name", ["mwpm", "corr", "bm", "bposd", "tesseract"])
@pytest.mark.parametrize("noise", ["sd6", "em3", "phen"])
def test_decoders_correct_every_single_fault_of_the_honeycomb_code(name, noise):
    if not _available(name):
        pytest.skip(f"{name} needs an optional package")
    c = competitor_circuit("honeycomb", 8, 2, _noise(noise), "Z")
    syn, flips = _single_faults(c)
    pred = competitor_decoder(c, name).decode_batch(syn)
    assert not np.any(pred != flips)
    assert not competitor_decoder(c, name).decode_batch(np.zeros_like(syn[:2])).any()


# ---------------------------------------------------------------- decoders
def _available(name: str) -> bool:
    if name in ("bm", "bposd"):
        return HAS_LDPC
    if name == "tesseract":
        return HAS_TESSERACT
    if name == "bm_pkg":
        return HAS_BELIEFMATCHING
    return True


@pytest.mark.parametrize("name", sorted(DECODERS))
@pytest.mark.parametrize("noise", ["sd6", "em3"])
def test_decoders_decode_a_trivial_syndrome(name, noise):
    if not _available(name):
        pytest.skip(f"{name} needs an optional package")
    c = competitor_circuit("xzzx", 3, 3, _noise(noise), "Z")
    dec = competitor_decoder(c, name)
    pred = dec.decode_batch(np.zeros((4, c.num_detectors), dtype=np.uint8))
    assert pred.shape == (4, 1) and pred.dtype == bool
    assert not pred.any()


@pytest.mark.parametrize("name", ["mwpm", "corr", "bm", "bposd", "tesseract", "bm_pkg"])
@pytest.mark.parametrize("noise", ["sd6", "em3", "phen", "phen_b10"])
def test_decoders_correct_every_single_fault(name, noise):
    if not _available(name):
        pytest.skip(f"{name} needs an optional package")
    # d = 5: PyMatching's correlated pass misreads a few single faults of d = 3
    # surface-code circuits (Stim's generated ones as well)
    c = competitor_circuit("css", 5, 2, _noise(noise), "X")
    syn, flips = _single_faults(c)
    pred = competitor_decoder(c, name).decode_batch(syn)
    assert not np.any(pred != flips)


def test_bm_pkg_says_when_the_package_is_missing():
    if HAS_BELIEFMATCHING:
        pytest.skip("beliefmatching is installed")
    c = competitor_circuit("css", 3, 2, CircuitNoise.uniform(1e-3), "X")
    with pytest.raises(ImportError, match="beliefmatching"):
        competitor_decoder(c, "bm_pkg")


def test_decoder_options_are_checked():
    c = competitor_circuit("css", 3, 2, CircuitNoise.uniform(1e-3), "X")
    with pytest.raises(ValueError):
        competitor_decoder(c, "unionfind")
    with pytest.raises(ValueError):
        competitor_decoder(c, "mwpm", osd_order=3)


@pytest.mark.parametrize("code", ["css", "xzzx"])
def test_matching_suppresses_errors_below_threshold(code):
    rates = []
    for d in (3, 5):
        c = competitor_circuit(code, d, d, CircuitNoise.uniform(2e-3), "X")
        est = competitor_decoder(c, "mwpm").estimate(max_shots=40000, max_errors=10**9,
                                                     batch=20000, seed=11)
        rates.append(est.rate)
    assert 0 < rates[1] < rates[0]
