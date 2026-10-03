"""Tests of the tensor-network decoder and of the fast OSD-CS used by CFE."""
import math

import numpy as np
import pytest
import scipy.sparse as sp

from mdr.ft import CircuitNoise, FTMDRCircuit
from mdr.ft.two_level_decoder import TwoLevelDecoder

pytest.importorskip("ldpc")


def _small(d=3, r=1, p=1e-2):
    ft = FTMDRCircuit(d, r, CircuitNoise.uniform(p), final="frame", detectors="combined")
    return ft, TwoLevelDecoder(ft, mode="mwpm")


def _dense_log_z(tn, mech_dets, mech_obs, priors, row):
    """Exact coset probabilities with a dense table over the open detectors (small codes)."""
    sch = tn.sched
    state = {((), 0): 1.0}
    opened = []
    for step, j in enumerate(sch.order):
        opened += [int(i) for i in sch.opens[step]]
        idx = [opened.index(int(i)) for i in mech_dets[j]]
        new = {}
        for (bits, l), v in state.items():
            b = list(bits) + [0] * (len(opened) - len(bits))
            k0 = (tuple(b), l)
            new[k0] = new.get(k0, 0.0) + v * (1 - priors[j])
            for q in idx:
                b[q] ^= 1
            k1 = (tuple(b), l ^ int(mech_obs[j]))
            new[k1] = new.get(k1, 0.0) + v * priors[j]
        state = new
        for i in sch.closes[step]:
            q = opened.index(int(i))
            st = {}
            for (bits, l), v in state.items():
                b = list(bits) + [0] * (len(opened) - len(bits))
                if b[q] != int(row[i]):
                    continue
                b.pop(q)
                st[(tuple(b), l)] = st.get((tuple(b), l), 0.0) + v
            opened.pop(q)
            state = st
    z = [sum(v for (_, l), v in state.items() if l == c) for c in (0, 1)]
    return [math.log(x) if x > 0 else -math.inf for x in z]


def test_tensor_network_is_exact_on_small_code():
    from mdr.ft.tn_decoder import TensorNetworkDecoder, detector_positions

    ft, dec = _small()
    pos = detector_positions(ft, dec.circuit)
    md = [k[0] for k in dec.mech_keys]
    mo = [0 in k[1] for k in dec.mech_keys]
    tn = TensorNetworkDecoder(md, mo, dec.priors, pos, chi=64)
    dets, _ = dec.circuit.compile_detector_sampler(seed=3).sample(5, separate_observables=True)
    for row in dets:
        l0, l1, disc = tn.log_z(row)
        e0, e1 = _dense_log_z(tn, md, mo, dec.priors, row)
        assert disc < 1e-12
        assert abs((l1 - l0) - (e1 - e0)) < 1e-8
        assert abs(l0 - e0) < 1e-8


def test_fast_osd_matches_ldpc():
    from ldpc import BpOsdDecoder

    from mdr.ft.fast_osd import FastBpOsd

    ft, dec = _small(3, 3, 6e-3)
    L = np.zeros((1, len(dec.mech_keys)), dtype=np.uint8)
    for j, (_, o) in enumerate(dec.mech_keys):
        L[0, j] = 1 if 0 in o else 0
    H = sp.vstack([sp.csr_matrix(dec.H_full), sp.csr_matrix(L)]).tocsc().astype(np.uint8)
    ref = BpOsdDecoder(H, error_channel=list(dec.priors), max_iter=30, bp_method="product_sum",
                       osd_method="osd_cs", osd_order=10)
    fast = FastBpOsd(H, dec.priors, bp_iters=30, order=10)
    dets, _ = dec.circuit.compile_detector_sampler(seed=4).sample(40, separate_observables=True)
    for row in dets:
        for ell in (0, 1):
            syn = np.concatenate([row.astype(np.uint8), [ell]])
            assert np.array_equal(np.asarray(ref.decode(syn)), fast.decode(syn))


def test_cfe_tn_uses_the_tensor_network():
    ft = FTMDRCircuit(3, 1, CircuitNoise.uniform(1e-2), final="frame", detectors="combined")
    dec = TwoLevelDecoder(ft, mode="cfe_tn", chi=16, chi_max=64)
    dets, obs = dec.circuit.compile_detector_sampler(seed=5).sample(50, separate_observables=True)
    pred = dec.decode_batch(dets)
    assert dec.tn_used == 50
    assert int(np.sum(np.any(pred != obs, axis=1))) < 10
