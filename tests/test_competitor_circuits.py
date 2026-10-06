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
from mdr.ft.competitor_circuits import CODES, CompetitorCircuit, competitor_circuit
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


# ---------------------------------------------------------------- circuits
@pytest.mark.parametrize("code", CODES)
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("native", ["gates", "pairs"])
@pytest.mark.parametrize("d,rounds", [(3, 1), (3, 3), (5, 2)])
def test_noiseless_circuit_is_deterministic(code, basis, native, d, rounds):
    noise = CircuitNoise() if native == "gates" else CircuitNoise.em3(0.0)
    c = competitor_circuit(code, d, rounds, noise, basis)
    dets, obs = c.compile_detector_sampler(seed=1).sample(200, separate_observables=True)
    assert c.num_observables == 1
    assert c.num_detectors == (d * d - 1) // 2 * 2 + (rounds - 1) * (d * d - 1)
    assert not dets.any()
    assert not obs.any()


@pytest.mark.parametrize("code", CODES)
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("noise", ["sd6", "purez", "em3"])
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


@pytest.mark.parametrize("noise", ["sd6", "si1000", "em3"])
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_codes_are_clifford_images_of_css(noise, basis):
    dems = {code: competitor_circuit(code, 3, 3, _noise(noise), basis)
            .detector_error_model(approximate_disjoint_errors=True) for code in CODES}
    for code in CODES:
        assert dems[code].approx_equals(dems["css"], atol=1e-12)


@pytest.mark.parametrize("noise", ["purez", "biased10", "helios_p"])
def test_biased_noise_distinguishes_the_codes(noise):
    dems = {code: competitor_circuit(code, 3, 3, _noise(noise), "X")
            .detector_error_model(approximate_disjoint_errors=True) for code in CODES}
    assert not dems["xzzx"].approx_equals(dems["css"], atol=1e-12)
    assert not dems["xy"].approx_equals(dems["css"], atol=1e-12)


@pytest.mark.parametrize("name", sorted(NOISE))
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


@pytest.mark.parametrize("name", sorted(NOISE))
@pytest.mark.parametrize("code", CODES)
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_noise_channels_are_those_of_ftmdr(name, code, basis):
    """Every channel of the competitor circuit, with its rates, also appears in FTMDRCircuit."""
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
    # frames of controlled Paulis that the code does not use
    extra = {(n, a) for n, a in ours - theirs
             if not (n == "X_ERROR" and a in ((noise.p_prep,), (noise.p_meas,)))
             and not (n == "PAULI_CHANNEL_2" and noise.p2_rzz_frame)}
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


@pytest.mark.parametrize("name", sorted(NOISE))
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_honeycomb_builds_with_every_noise_model(name, basis):
    noise = _noise(name)
    c = competitor_circuit("honeycomb", 4, 2, noise, basis)
    dem = c.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    assert dem.num_errors > 0

    def xtalk(k):
        if not noise.p_xtalk:
            return 0.0
        return 0.75 * (1 - (1 - 4 * noise.p_xtalk / 3) ** k) if noise.xtalk_per_mcmr else noise.p_xtalk

    # every channel, with its rate, also appears in FTMDRCircuit
    ours = _channels(FTMDRCircuit(3, 2, noise, detectors="combined").build(), xtalk(17))
    theirs = _channels(c, xtalk(12))  # 12 edges (ancillas) per sub-round at d = 4
    assert theirs <= ours


@pytest.mark.parametrize("name", ["mwpm", "corr", "bm", "bposd", "tesseract"])
@pytest.mark.parametrize("noise", ["sd6", "em3"])
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
@pytest.mark.parametrize("noise", ["sd6", "em3"])
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
