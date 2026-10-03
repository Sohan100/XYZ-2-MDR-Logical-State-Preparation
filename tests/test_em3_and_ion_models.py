"""Tests for the EM3 pair-measurement extraction and the one-parameter Quantinuum models."""

from __future__ import annotations

import numpy as np
import pytest

from mdr.ft import CircuitNoise, FTMDRCircuit
from mdr.ft.circuit_noise import QUANTINUUM_P2
from mdr.ft.two_level_decoder import TwoLevelDecoder


@pytest.mark.parametrize("machine", ["helios", "h2"])
def test_trapped_ion_model_reproduces_the_device_point(machine):
    p0 = QUANTINUUM_P2[machine]
    assert CircuitNoise.trapped_ion(p0, machine) == CircuitNoise.quantinuum(machine, 1.0)
    a, b = CircuitNoise.trapped_ion(2 * p0, machine), CircuitNoise.trapped_ion(p0, machine)
    for field in ("p1", "p2", "p_prep", "p_meas", "p_mem_z", "p_xtalk"):
        assert getattr(a, field) == pytest.approx(2 * getattr(b, field))
    assert CircuitNoise.trapped_ion(p0, machine, crosstalk=False).p_xtalk == 0.0


def test_helios_ratios():
    n = CircuitNoise.trapped_ion(1e-3, "helios")
    assert n.p2 == pytest.approx(1e-3)
    assert n.p1 / n.p2 == pytest.approx(0.045)
    assert n.p_mem_z / n.p2 == pytest.approx(0.9)
    assert n.p_xtalk / n.p2 == pytest.approx(0.075)
    assert n.p_prep / n.p2 == pytest.approx(0.25)


@pytest.mark.parametrize("d,rounds", [(3, 1), (3, 3), (5, 2)])
@pytest.mark.parametrize("final", ["frame", "ideal"])
def test_em3_noiseless_circuit_is_deterministic(d, rounds, final):
    c = FTMDRCircuit(d, rounds, CircuitNoise.em3(0.0), final=final,
                     detectors="combined").build()
    dets, obs = c.compile_detector_sampler(seed=3).sample(200, separate_observables=True)
    assert not dets.any()
    assert not obs.any()


def test_em3_plaquettes_are_three_coloured_and_trees_are_block_aligned():
    ft = FTMDRCircuit(5, 1, CircuitNoise.em3(0.0))
    color = ft.plaquette_colors()
    checks = ft.geometry.checks
    for ci in color:
        for cj in color:
            if ci < cj:
                qi = {q for _, q in checks[ci]["terms"]}
                qj = {q for _, q in checks[cj]["terms"]}
                if qi & qj:
                    assert color[ci] != color[cj]
    block = {q: ch["block"] for ch in checks if ch["kind"] == "link" for _, q in ch["terms"]}
    for ci in color:
        edges, layer, parent = ft.cat_tree(ci)
        qs = [q for _, q in checks[ci]["terms"]]
        assert len(edges) == len(qs) - 1 and max(layer) <= 2
        # every subtree below an edge is one qubit or the two qubits of one block
        for _, child in edges:
            below = [k for k in range(len(qs)) if _is_below(parent, k, child)]
            assert len(below) == 1 or (len(below) == 2 and block[qs[below[0]]] == block[qs[below[1]]])


def _is_below(parent, k, node):
    while k >= 0:
        if k == node:
            return True
        k = parent[k]
    return False


def test_em3_fault_distance_d3_is_three():
    pytest.importorskip("ortools")
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "efd", Path(__file__).resolve().parents[1] / "scripts" / "exact_fault_distance.py")
    efd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(efd)
    c = FTMDRCircuit(3, 3, CircuitNoise.em3(1e-3)).build()
    assert efd.exists_logical_error(c, 2, time_limit=120)[0] == "INFEASIBLE"
    assert efd.exists_logical_error(c, 3, time_limit=120)[0] in ("OPTIMAL", "FEASIBLE")


def test_em3_decoders_run_and_suppress_errors():
    rates = []
    for d in (3, 5):
        ft = FTMDRCircuit(d, d, CircuitNoise.em3(1e-3), detectors="combined")
        dec = TwoLevelDecoder(ft, mode="corr_split", lower="gauge")
        est = dec.estimate(max_shots=20000, max_errors=10**9, batch=10000, seed=5)
        rates.append(est.rate)
    assert rates[1] < rates[0]
