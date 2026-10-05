"""Tests for the noise models and the two-level decoders of the FT MDR path."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from mdr.ft import CircuitNoise, FTMDRCircuit
from mdr.ft.two_level_decoder import TwoLevelDecoder

HAS_LDPC = importlib.util.find_spec("ldpc") is not None


@pytest.mark.parametrize("eta", [0.5, 10.0, 100.0])
def test_biased_noise_keeps_total_rate(eta):
    p = 3e-3
    nz = CircuitNoise.biased(p, eta)
    assert sum(nz.p2_paulis) == pytest.approx(p)
    assert sum(nz.idle_xyz) == pytest.approx(p)
    px, py, pz = nz.idle_xyz
    assert pz / (px + py) == pytest.approx(eta)


def test_si1000_adds_resonator_idle_on_data():
    p = 1e-3
    text = str(FTMDRCircuit(3, 2, CircuitNoise.si1000(p)).build())
    # one reset step and one measurement step per round
    assert text.count(f"DEPOLARIZE1({2 * p:g})") == 4
    assert text.count(f"DEPOLARIZE1({p / 10:g})") == 12


@pytest.mark.parametrize("d,rounds", [(3, 1), (3, 3), (5, 2)])
def test_combined_detectors_extend_s0_detectors(d, rounds):
    noise = CircuitNoise.uniform(1e-3)
    s0 = FTMDRCircuit(d, rounds, noise, detectors="s0").build()
    comb = FTMDRCircuit(d, rounds, noise, detectors="combined").build()
    n_gauge = (d * d - 1) * (rounds - 1)
    assert comb.num_detectors == s0.num_detectors + n_gauge
    assert s0.num_detectors == d * d * (rounds + 1)
    # the combined circuit is still deterministic without noise
    clean = FTMDRCircuit(d, rounds, CircuitNoise(), detectors="combined").build()
    dets, obs = clean.compile_detector_sampler(seed=1).sample(50, separate_observables=True)
    assert not dets.any() and not obs.any()


@pytest.mark.parametrize("mode,kw", [
    ("mwpm", {}),
    ("corr_split", {"lower": "links"}),
    ("corr_split", {"lower": "gauge"}),
    ("seq_match", {}),
])
def test_matching_based_modes_decode_trivial_syndrome(mode, kw):
    ft = FTMDRCircuit(3, 3, CircuitNoise.uniform(2e-3), detectors="combined")
    dec = TwoLevelDecoder(ft, mode=mode, **kw)
    zeros = np.zeros((4, ft.build().num_detectors), dtype=np.uint8)
    assert not dec.decode_batch(zeros).any()


@pytest.mark.skipif(not HAS_LDPC, reason="ldpc not installed")
@pytest.mark.parametrize("mode", ["seq_soft", "bp_full"])
def test_bp_modes_decode_trivial_syndrome(mode):
    ft = FTMDRCircuit(3, 2, CircuitNoise.uniform(2e-3), detectors="combined")
    dec = TwoLevelDecoder(ft, mode=mode, bp_iters=10)
    zeros = np.zeros((2, ft.build().num_detectors), dtype=np.uint8)
    assert not dec.decode_batch(zeros).any()


def test_bp_corr_edge_weights_keep_their_sign():
    """A bp_corr shot at d = 21 (sd6, p = 0.0026) that PyMatching refused: 17 faults on one edge, three of
    which BP was sure of. The merged weight came out -1.1e-16 instead of +6e-18, and the correlated
    reweighting of that edge then raised. The same faults in the same order, each with its own partner edge."""
    import pymatching
    import stim
    from mdr.ft.two_level_decoder import SPLIT_EDGE_FLOOR

    q = [2.85267329e-07, 2.73626235e-06, 1.42579968e-06, 6.86657655e-06, 0.000619036821, 0.00131830763,
         0.0200238345, 0.6, 2.94819001e-05, 5.15077965e-05, 0.0005896192, 0.00123583736, 0.152530713,
         0.00122778285, 0.6, 0.228791136, 0.6]
    corr = [k != 4 for k in range(len(q))]
    terms = [f"D{2 * k + 2} D{2 * k + 3} ^ D0 D1" if c else "D0 D1" for k, c in enumerate(corr)]
    terms += [f"D{2 * k + 2} D{2 * k + 3}" for k, c in enumerate(corr) if c]
    q = np.array(q + [0.3] * sum(corr))
    dec = TwoLevelDecoder.__new__(TwoLevelDecoder)
    dec._split_terms = list(enumerate(terms))
    dec._split_tail = [f"detector D{i}" for i in range(2 * len(corr) + 2)]
    syn = np.zeros(2 * len(corr) + 2, dtype=np.uint8)
    syn[:2] = 1
    plain = stim.DetectorErrorModel("\n".join(
        [f"error({x:.9g}) {t}" for x, t in zip(np.clip(q, 1e-9, 0.5 - 1e-6), terms)] + dec._split_tail))
    with pytest.raises(ValueError, match="change the sign"):
        pymatching.Matching.from_detector_error_model(plain, enable_correlations=True).decode(
            syn, enable_correlations=True)
    dem = dec._split_dem_with(q)
    probs = np.array([inst.args_copy()[0] for inst in dem if inst.type == "error"])
    assert np.exp(dec._split_inc @ np.log1p(-2.0 * probs)).min() >= 0.99 * SPLIT_EDGE_FLOOR
    pymatching.Matching.from_detector_error_model(dem, enable_correlations=True).decode(syn, enable_correlations=True)
    # posteriors away from 0.5 give the model as it was
    q2 = np.full(len(terms), 1e-3)
    assert str(dec._split_dem_with(q2)) == str(stim.DetectorErrorModel("\n".join(
        [f"error({x:.9g}) {t}" for x, t in zip(q2, terms)] + dec._split_tail)))


def test_hierarchical_matching_beats_s0_matching():
    ft = FTMDRCircuit(5, 5, CircuitNoise.biased(5e-3, 100), detectors="combined")
    circuit = ft.build()
    dets, obs = circuit.compile_detector_sampler(seed=3).sample(
        4000, separate_observables=True)
    errs = {}
    for name, kw in [("mwpm", dict(mode="mwpm")),
                     ("hcm", dict(mode="corr_split", lower="gauge"))]:
        pred = TwoLevelDecoder(ft, **kw).decode_batch(dets)
        errs[name] = int(np.any(pred != obs, axis=1).sum())
    assert errs["hcm"] < 0.6 * errs["mwpm"]


def test_split_dem_keeps_almost_every_fault():
    ft = FTMDRCircuit(3, 3, CircuitNoise.uniform(1e-3), detectors="combined")
    dec = TwoLevelDecoder(ft, mode="corr_split", lower="gauge")
    assert dec.split_dropped <= 0.02 * len(dec.mech_keys)
    assert dec.split_dem.num_detectors == ft.build().num_detectors


@pytest.mark.parametrize("rounds", [1, 3])
def test_exact_fault_distance_d3(rounds):
    import importlib.util as iu
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "exact_fault_distance.py"
    spec = iu.spec_from_file_location("exact_fault_distance", path)
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    circuit = FTMDRCircuit(3, rounds, CircuitNoise.uniform(1e-3)).build()
    assert mod.exists_logical_error(circuit, 3, time_limit=60)[0] == "INFEASIBLE"
    assert mod.exists_logical_error(circuit, 4, time_limit=60)[0] in ("OPTIMAL", "FEASIBLE")
