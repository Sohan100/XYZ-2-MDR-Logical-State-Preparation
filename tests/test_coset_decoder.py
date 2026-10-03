"""Tests for the coset free-energy decoder."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from mdr.ft import CircuitNoise, FTMDRCircuit
from mdr.ft.coset_decoder import degeneracy_moves

HAS_LDPC = importlib.util.find_spec("ldpc") is not None


def test_moves_of_a_square_with_a_hyperedge():
    a, b, c, e, h = ((0, 1), ()), ((1, 2), ()), ((0, 3), ()), ((3, 2), ()), ((0, 1, 2, 3), ())
    G = degeneracy_moves([a, b, c, e, h])
    moves = {tuple(int(x) for x in g if x >= 0) for g in G}
    assert (0, 1, 2, 3) in moves      # two paths around the square
    assert (1, 2, 4) in moves         # hyperedge = (1,2) + (0,3)
    assert (0, 3, 4) in moves         # hyperedge = (0,1) + (3,2)


def test_moves_preserve_syndrome_and_observable():
    ft = FTMDRCircuit(3, 2, CircuitNoise.uniform(1e-3), detectors="combined")
    dem = ft.build().detector_error_model(approximate_disjoint_errors=True)
    keys = []
    for inst in dem.flattened():
        if inst.type == "error":
            d = tuple(sorted(t.val for t in inst.targets_copy() if t.is_relative_detector_id()))
            o = tuple(sorted(t.val for t in inst.targets_copy() if t.is_logical_observable_id()))
            keys.append((d, o))
    keys = list(dict.fromkeys(keys))
    G = degeneracy_moves(keys)
    assert len(G) > 0
    for g in G[:: max(1, len(G) // 500)]:
        dets, obs = set(), set()
        for j in g:
            if j >= 0:
                dets ^= set(keys[j][0])
                obs ^= set(keys[j][1])
        assert not dets and not obs


@pytest.mark.skipif(not HAS_LDPC, reason="needs ldpc")
def test_cfe_descent_keeps_class_and_lowers_weight():
    from mdr.ft.two_level_decoder import TwoLevelDecoder

    ft = FTMDRCircuit(3, 3, CircuitNoise.uniform(4e-3), detectors="combined")
    dec = TwoLevelDecoder(ft, mode="cfe", cfe_guided=False)
    C = dec._cfe
    H = dec.H_full.tocsr()
    dets, _ = dec.circuit.compile_detector_sampler(seed=5).sample(40, separate_observables=True)
    for i in range(40):
        if not dets[i].any():
            continue
        for ell, cands in enumerate(C.class_candidates(dets[i])):
            for mask in cands:
                w0 = C.free_energy(mask)[0]
                C.descend(mask)
                x = mask[:-1].astype(np.uint8)
                assert np.array_equal(H @ x % 2, dets[i].astype(np.uint8))
                assert int(C.L.astype(np.uint8) @ x % 2) == ell
                assert C.free_energy(mask)[0] <= w0 + 1e-9


@pytest.mark.skipif(not HAS_LDPC, reason="needs ldpc")
def test_cfe_beats_matching_at_d3():
    from mdr.ft.two_level_decoder import TwoLevelDecoder

    ft = FTMDRCircuit(3, 3, CircuitNoise.biased(5e-3, 100), detectors="combined")
    cfe = TwoLevelDecoder(ft, mode="cfe")
    mwpm = TwoLevelDecoder(ft, mode="mwpm")
    dets, obs = cfe.circuit.compile_detector_sampler(seed=9).sample(1500, separate_observables=True)
    e_cfe = int(np.sum(cfe.decode_batch(dets) != obs))
    e_mwpm = int(np.sum(mwpm.decode_batch(dets) != obs))
    assert e_cfe < e_mwpm


@pytest.mark.skipif(not HAS_LDPC, reason="needs ldpc")
def test_cfe_gate_that_always_opens_matches_full_decoder():
    from mdr.ft.two_level_decoder import TwoLevelDecoder

    ft = FTMDRCircuit(3, 3, CircuitNoise.uniform(6e-3), detectors="combined")
    full = TwoLevelDecoder(ft, mode="cfe")
    gated = TwoLevelDecoder(ft, mode="cfe", cfe_gate=float("inf"))
    dets, _ = full.circuit.compile_detector_sampler(seed=4).sample(200, separate_observables=True)
    assert np.array_equal(full.decode_batch(dets), gated.decode_batch(dets))
