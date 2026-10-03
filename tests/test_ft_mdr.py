"""Tests for the fault-tolerant MDR path (src/mdr/ft)."""

from __future__ import annotations

import numpy as np
import pytest
import stim

from mdr.ft import (
    DEPTH6_SCHEDULE,
    CircuitNoise,
    ExtractionSchedule,
    FTMDRCircuit,
    S0MatchingDecoder,
    XYZ2FrameBasis,
    XYZ2Geometry,
)
from xyz2.stabilizer_generator import XYZ2StabilizerGenerator


def _fault_distance(circuit: stim.Circuit, max_set: int = 6) -> int:
    return len(circuit.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=max_set,
        dont_explore_edges_with_degree_above=9999,
        dont_explore_edges_increasing_symptom_degree=False,
    ))


@pytest.mark.parametrize("d", [3, 5, 7])
def test_geometry_matches_repository_generator(d):
    geo = XYZ2Geometry(d)
    assert geo.stabilizers == XYZ2StabilizerGenerator(d).generate_stabilizers()
    kinds = [ch["kind"] for ch in geo.checks]
    assert kinds.count("link") == d * d
    assert kinds.count("hex") == (d - 1) ** 2 + 2 * (d - 1)


@pytest.mark.parametrize("d", [3, 5, 7])
def test_frame_fixes_logical_x_and_d_squared_stabilizers(d):
    frame = XYZ2FrameBasis(XYZ2Geometry(d))
    assert frame.logical_x_in_frame()
    assert frame.s0_rows.shape[0] == d * d
    all_x = {q: "X" for q in range(2 * d * d)}
    assert frame.deterministic_rows(all_x).shape[0] == d * d


@pytest.mark.parametrize("d", [3, 5, 7])
def test_depth6_schedule_is_valid(d):
    ok, msg = DEPTH6_SCHEDULE.validate(XYZ2Geometry(d))
    assert ok, msg


def test_schedule_enumeration_contains_selected_schedule():
    found = ExtractionSchedule.enumerate_depth(6)
    assert len(found) == 8928
    key = (DEPTH6_SCHEDULE.sigma, DEPTH6_SCHEDULE.link)
    assert any((s.sigma, s.link) == key for s in found)


@pytest.mark.parametrize("final", ["frame", "ideal"])
@pytest.mark.parametrize("rounds", [1, 3])
def test_noiseless_circuit_has_no_logical_errors(final, rounds):
    c = FTMDRCircuit(3, rounds, CircuitNoise(), final=final).build()
    dets, obs = c.compile_detector_sampler(seed=1).sample(
        200, separate_observables=True)
    assert not dets.any()
    assert not obs.any()


@pytest.mark.parametrize("d,rounds", [(3, 1), (3, 3), (5, 1), (5, 5)])
@pytest.mark.parametrize("final", ["frame", "ideal"])
def test_fault_distance_is_d_plus_one(d, rounds, final):
    c = FTMDRCircuit(d, rounds, CircuitNoise.uniform(1e-3), final=final).build()
    assert _fault_distance(c) == d + 1


def test_plus_start_is_not_fault_tolerant():
    c = FTMDRCircuit(3, 3, CircuitNoise.uniform(1e-3), init="plus",
                     detectors="all").build()
    assert _fault_distance(c, 4) <= 2


def test_matching_decoder_suppresses_errors_with_distance():
    rates = []
    for d in (3, 5):
        c = FTMDRCircuit(d, 1, CircuitNoise.uniform(3e-3)).build()
        est = S0MatchingDecoder(c).estimate(max_shots=100_000,
                                            max_errors=10**9, seed=7)
        rates.append(est.rate)
    assert rates[1] < 0.5 * rates[0]


def test_quantinuum_presets():
    helios = CircuitNoise.quantinuum("helios")
    # average infidelities converted to Pauli probabilities
    assert helios.p1 == pytest.approx(1.5 * 3e-5)
    assert helios.p2 == pytest.approx(1.25 * 8e-4)
    assert helios.p_mem_z == pytest.approx(1.5 * 6e-4)
    assert helios.p_xtalk == pytest.approx(1.5 * 5e-5)
    assert helios.p_prep == helios.p_meas == pytest.approx(2.5e-4)
    assert helios.xtalk_per_mcmr
    h2 = CircuitNoise.quantinuum("h2", scale=2.0)
    assert h2.p2 == pytest.approx(2 * 1.25 * 1.5e-3)
    with pytest.raises(ValueError):
        CircuitNoise.quantinuum("ibm")


def test_crosstalk_composes_over_measured_ancillas():
    nz = CircuitNoise.quantinuum("helios")
    c = FTMDRCircuit(3, 1, nz).build()
    k = c.num_qubits - 18  # one crosstalk event per measured ancilla
    p = 0.75 * (1 - (1 - 4 * nz.p_xtalk / 3) ** k)
    assert f"DEPOLARIZE1({p:.10g})"[:22] in str(c)


def test_ancilla_sharing_fits_d5_on_98_qubits_and_keeps_distance():
    ft = FTMDRCircuit(5, 1, CircuitNoise.uniform(1e-3), share_ancillas=4)
    c = ft.build()
    assert c.num_qubits == 95
    assert _fault_distance(c) == 6
    clean = FTMDRCircuit(5, 2, CircuitNoise(), share_ancillas=4).build()
    dets, obs = clean.compile_detector_sampler(seed=2).sample(
        100, separate_observables=True)
    assert not dets.any() and not obs.any()


def test_decode_measurements_matches_detector_sampling():
    c = FTMDRCircuit(3, 1, CircuitNoise.quantinuum("helios", scale=3.0)).build()
    dec = S0MatchingDecoder(c)
    meas = c.compile_sampler(seed=11).sample(50_000)
    fails = dec.decode_measurements(meas)
    est = dec.estimate(max_shots=50_000, max_errors=10**9, seed=11)
    assert abs(fails.mean() - est.rate) < 5 * est.stderr + 1e-4


def _mechanism_masks(circuit: stim.Circuit):
    dem = circuit.detector_error_model(decompose_errors=False,
                                       approximate_disjoint_errors=True)
    nd = dem.num_detectors
    masks = set()
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        m = 0
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                m ^= 1 << t.val
            elif t.is_logical_observable_id():
                m ^= 1 << (nd + t.val)
        if m:
            masks.add(m)
    return sorted(masks), 1 << nd


@pytest.mark.parametrize("rounds", [1, 3])
def test_no_logical_error_from_three_or_fewer_faults_d3(rounds):
    masks, obs = _mechanism_masks(
        FTMDRCircuit(3, rounds, CircuitNoise.uniform(1e-3)).build())
    singles = set(masks)
    assert obs not in singles
    assert all((a ^ obs) not in singles for a in masks)
    for i, a in enumerate(masks):
        for b in masks[i + 1:]:
            assert (a ^ b ^ obs) not in singles
