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
