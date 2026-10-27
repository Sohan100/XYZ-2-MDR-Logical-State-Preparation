"""Tests of the circuit-level sequential decoder of Srivastava et al. (sri_m, sri_bm)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from mdr.ft import CircuitNoise, FTMDRCircuit
from mdr.ft.sequential_decoder import DATA, OTHER, STATIC, TIME, pauli_signatures
from mdr.ft.two_level_decoder import TwoLevelDecoder

HAS_LDPC = importlib.util.find_spec("ldpc") is not None


def _decoder(noise, d, mode="sri_m", **kw):
    ft = FTMDRCircuit(d, d, noise, detectors="combined")
    return TwoLevelDecoder(ft, mode=mode, bp_iters=10, bp_method="product_sum", **kw)


# --------------------------------------------------------------------------- reference implementation
def table_one(px, py, pz):
    """Table I of Srivastava et al.: {(sigma, s): P_sigma,s} for one link."""
    pi = 1.0 - px - py - pz
    p0 = (pi + px) ** 2 + (py + pz) ** 2
    p1 = 2 * (pi + px) * (py + pz)
    return {("I", 0): (pi ** 2 + px ** 2) / p0, ("X", 0): (pz ** 2 + py ** 2) / p0,
            ("Y", 0): 2 * py * pz / p0, ("Z", 0): 2 * px * pi / p0,
            ("I", 1): (pz * pi + px * py) / p1, ("X", 1): (pz * pi + px * py) / p1,
            ("Y", 1): (py * pi + px * pz) / p1, ("Z", 1): (py * pi + px * pz) / p1}


class Reference:
    """
    Their Sec. III A 2 written directly for the phenomenological circuit: link syndromes per link and
    round, pairs removed when q^r > p^2 (q = p, p the physical rate), Z on the top qubit of every other
    event, and the upper-level model of Table I (one entry per link, round and YZZY Pauli).
    """

    def __init__(self, dec):
        ft, circ = dec.ft, dec.circuit
        self.ft, self.circ = ft, circ
        geo = ft.geometry
        self.p = ft.noise.p_meas
        coords = circ.get_detector_coordinates()
        rows = ft.frame.s0_rows
        self.link_det = {}                      # (link check, round) -> detector
        n0 = 0
        for x in range(circ.num_detectors):
            k, t, ph = (int(v) for v in coords[x][:3])
            if ph in (0, 2):
                if t == 0 and ph == 0:
                    k, n0 = n0, n0 + 1
                if rows[k].sum() == 1 and geo.checks[int(np.flatnonzero(rows[k])[0])]["kind"] == "link":
                    self.link_det[(int(np.flatnonzero(rows[k])[0]), t)] = x
            elif ph == 3 and geo.checks[k]["kind"] == "link":
                self.link_det[(k, t)] = x
        self.links = sorted({ci for ci, _ in self.link_det})
        self.top = {ci: geo.verts(*geo.checks[ci]["block"])[0] for ci in self.links}
        self.bot = {ci: geo.verts(*geo.checks[ci]["block"])[1] for ci in self.links}

    def decisions(self, row):
        """{(link, round) corrected}, {(link, round) of the time edges of removed pairs}."""
        p, w = self.p, np.log((1 - self.p) / self.p)
        corr, time = set(), set()
        for ci in self.links:
            ts = sorted(t for (c, t) in self.link_det if c == ci)
            ev = [t for t in ts if row[self.link_det[(ci, t)]]]
            if not ev:
                continue
            first, last = ts[0], ts[-1]
            pos = {t: i for i, t in enumerate(ts)}

            def cost(a, b):
                return w * (pos[b] - pos[a])
            start = w * (pos[ev[0]] - pos[first] + 1)
            end = w * (pos[last] - pos[ev[-1]] + 1)
            k = len(ev)

            def pc(s, e):
                return sum(cost(ev[i], ev[i + 1]) for i in range(s, e, 2))
            if k % 2 == 0:
                opts = [(pc(0, k), 0, k, [])]
                if k >= 2:
                    opts.append((start + end + pc(1, k - 1), 1, k - 1, [ev[0], ev[-1]]))
            else:
                opts = [(start + pc(1, k), 1, k, [ev[0]]), (end + pc(0, k - 1), 0, k - 1, [ev[-1]])]
            _, s, e, ends = min(opts, key=lambda o: o[0])
            corr.update((ci, t) for t in ends)
            for i in range(s, e, 2):
                a, b = ev[i], ev[i + 1]
                r = pos[b] - pos[a]
                # no data noise between the last round and the noiseless frame readout
                pa, pb = (0.0 if t == self.ft.rounds else p for t in (a, b))
                if p ** r > pa * pb:
                    time.update((ci, ts[pos[a] + j]) for j in range(r))
                else:
                    corr.update([(ci, a), (ci, b)])
        return corr, time

    def model(self, corrected, px, py, pz):
        """{(link, round, sigma): probability} of the YZZY data errors, Table I column by correction."""
        tab = table_one(px, py, pz)
        out = {}
        for ci in self.links:
            for t in range(self.ft.rounds):
                s = int((ci, t) in corrected)
                for sig in "XYZ":
                    out[(ci, t, sig)] = tab[(sig, s)]
        return out

    def signatures(self, nonlink):
        """(non-link detectors, observables) of every YZZY Pauli at every link and round."""
        rep = {"X": ("Z", "Z"), "Y": ("Z", "Y"), "Z": ("X", None)}
        ins, keys = [], []
        for ci in self.links:
            for t in range(self.ft.rounds):
                for sig, (a, b) in rep.items():
                    keys.append((ci, t, sig))
                    ins.append([(self.ft.round_start[t], self.top[ci], a)]
                               + ([(self.ft.round_start[t], self.bot[ci], b)] if b else []))
        flat = [x for group in ins for x in group]
        dd, oo = pauli_signatures(self.circ, flat)
        out, k = {}, 0
        for key, group in zip(keys, ins):
            D, O = set(), set()
            for _ in group:
                D ^= {int(x) for x in dd[k] if nonlink[x]}
                O ^= {int(x) for x in oo[k]}
                k += 1
            out[key] = (frozenset(D), frozenset(O))
        return out


@pytest.fixture(scope="module", params=[(3, 0.5), (5, 0.5), (3, 10.0), (5, 10.0)])
def phen_case(request):
    d, eta = request.param
    noise = CircuitNoise.phenomenological(0.02, eta)
    dec = _decoder(noise, d)
    return dec, Reference(dec), noise


def test_link_decisions_match_their_rule(phen_case):
    dec, ref, noise = phen_case
    sri = dec._sri
    dets = dec.circuit.compile_detector_sampler(seed=7).sample(400)
    lk = {int(x): (sri.chain_ci[sri.lk_chain[i]], int(sri.lk_round[i])) for i, x in enumerate(sri.link_cols)}
    assert set(lk.values()) == set(ref.link_det)
    for row in dets.astype(np.uint8):
        corr, time = sri.link_decisions(row[sri.link_cols])
        c_ref, t_ref = ref.decisions(row)
        assert {lk[int(sri.link_cols[i])] for i in corr} == c_ref
        assert {lk[int(sri.link_cols[i])] for i in time} == t_ref


def test_upper_priors_reproduce_table_one(phen_case):
    """
    A corrected link round brings in its data-edge faults with the probabilities of Table I (column
    s = 1, by the YZZY Pauli of the residual, to first order in p); an uncorrected one brings in none.
    The rounds next to a time end are left out: there the circuit adds a link outcome flip to the
    data edge (an odd link's first outcome is random, its last is not compared with a readout).
    """
    dec, ref, noise = phen_case
    sri = dec._sri
    px, py, pz = noise.data_xyz
    p = noise.p_meas
    tab = table_one(px, py, pz)
    sigs = ref.signatures(~sri.is_link)
    lk = {int(x): (sri.chain_ci[sri.lk_chain[i]], int(sri.lk_round[i])) for i, x in enumerate(sri.link_cols)}
    first = {ci: min(t for c, t in ref.link_det if c == ci) for ci in ref.links}
    last = {ci: max(t for c, t in ref.link_det if c == ci) for ci in ref.links}
    checked = 0
    dets = dec.circuit.compile_detector_sampler(seed=11).sample(100)
    for row in dets.astype(np.uint8):
        corr, time = sri.link_decisions(row[sri.link_cols])
        q = sri.priors(corr, time)
        cs = set(corr)
        for i, x in enumerate(sri.link_cols):
            ci, t = lk[int(x)]
            mechs = sri.data_mechs[i]
            if i not in cs:
                assert not q[mechs].any()
                continue
            if not first[ci] < t < last[ci]:
                continue
            by_class = dict.fromkeys("IXYZ", 0.0)
            for j in mechs:
                D, O, _ = sri._residual(j)
                cls = "I" if not D and not O else next(
                    (sig for sig in "XYZ" if sigs[(ci, t, sig)] == (D, O)), None)
                assert cls is not None, (ci, t, sorted(D))
                by_class[cls] += q[j]
            for cls, got in by_class.items():
                assert abs(got - tab[(cls, 1)]) <= 4 * p * tab[(cls, 1)] + 4 * p * p, (ci, t, cls, got)
            checked += 1
    assert checked > 0 or dec.ft.geometry.d == 3


def test_uncorrected_rounds_carry_the_second_order_column(phen_case):
    """
    An uncorrected link round brings in the pairs of its data-edge faults: the X and Y entries of the
    s = 0 column of Table I, (p_z^2 + p_y^2) / p_0 and 2 p_y p_z / p_0, to a relative O(p).
    """
    dec, ref, noise = phen_case
    sri = dec._sri
    px, py, pz = noise.data_xyz
    p = noise.p_meas
    tab = table_one(px, py, pz)
    sigs = ref.signatures(~sri.is_link)
    nm = len(dec.mech_keys)
    checked = 0
    for i, x in enumerate(sri.link_cols):
        ci, t = sri.chain_ci[sri.lk_chain[i]], int(sri.lk_round[i])
        if not (0 < sri.lk_pos[i] < len(sri.chains[sri.lk_chain[i]]) - 1):
            continue
        by_class = dict.fromkeys("XYZ", 0.0)
        for col in sri.pair_cols[i]:
            U, O = sri.upper_signature(int(col))
            cls = next((sig for sig in "XYZ" if sigs[(ci, t, sig)] == (U, O)), None)
            assert cls is not None, (ci, t)
            by_class[cls] += sri.q_static[col]
        for cls in "XY":
            assert by_class[cls] == pytest.approx(tab[(cls, 0)], rel=4 * p), (ci, t, cls)
        # both pairs dropped once the round is corrected
        assert not sri.priors([i], [])[sri.pair_cols[i]].any()
        checked += 1
    assert checked or dec.ft.geometry.d == 3
    assert sri.ncol == nm + len(sri.pair_info)


def test_data_edges_at_the_time_ends(phen_case):
    """
    The data edge of a link round holds the Y and Z errors of its two qubits; at an odd link's first
    and last detector it also holds the outcome flip (its first outcome is random, its last is not
    compared with the readout), and the readout comparison of an even link has none.
    """
    dec, ref, noise = phen_case
    sri = dec._sri
    p = noise.p_meas
    px, py, pz = noise.data_xyz
    a = py + pz
    data = 2 * a * (1 - a)                       # an odd number of the two qubits flips the link
    for i in range(len(sri.link_cols)):
        ch = sri.chains[sri.lk_chain[i]]
        gauge = sri.chain_ci[sri.lk_chain[i]] in dec.ft.gauge
        end = sri.lk_pos[i] in (0, len(ch) - 1)
        if gauge and end:
            want = data * (1 - p) + p * (1 - data)
        elif not gauge and sri.lk_round[i] == dec.ft.rounds:
            want = 0.0
        else:
            want = data
        assert sri.p_data[i] == pytest.approx(want, rel=1e-9, abs=1e-15)


@pytest.mark.parametrize("name,noise", [
    ("sd6", CircuitNoise.uniform(3e-3)),
    ("phen", CircuitNoise.phenomenological(0.02)),
])
@pytest.mark.parametrize("d", [3, 5])
@pytest.mark.parametrize("mode", ["sri_m", pytest.param(
    "sri_bm", marks=pytest.mark.skipif(not HAS_LDPC, reason="ldpc missing"))])
def test_every_single_fault_is_corrected(name, noise, d, mode):
    dec = _decoder(noise, d, mode)
    nd = dec.circuit.num_detectors
    bad = []
    for j, (dets, obs) in enumerate(dec.mech_keys):
        row = np.zeros(nd, dtype=np.uint8)
        row[list(dets)] = 1
        want = np.zeros(dec.circuit.num_observables, dtype=bool)
        want[list(obs)] = True
        if (dec._sri.decode_shot(row) != want).any():
            bad.append(j)
    assert not bad, f"{len(bad)} single faults fail, e.g. {[dec.mech_keys[j] for j in bad[:3]]}"


def test_fault_classes_and_corrections():
    dec = _decoder(CircuitNoise.uniform(3e-3), 5)
    sri = dec._sri
    # every correction flips exactly its own link detector among the link detectors
    for i, dd in enumerate(sri.corr_dets):
        assert list(dd[sri.is_link[dd]]) == [sri.link_cols[i]]
    # every link detector of SD6 has data-edge faults, and the classes cover every fault
    assert all(len(m) for m in sri.data_mechs)
    assert sri.stats["mechanisms"] == sum(sri.stats[k] for k in ("static", "data", "time", "other"))
    assert set(np.unique(sri.kind)) <= {STATIC, DATA, TIME, OTHER}
    assert sri.stats["other_fraction_of_linked"] < 0.05


def test_code_capacity_corrects_every_link_event():
    noise = CircuitNoise.code_capacity(0.1)
    ft = FTMDRCircuit(5, 2, noise, detectors="combined")
    dec = TwoLevelDecoder(ft, mode="sri_m")
    sri = dec._sri
    assert not sri.q_time.any()                        # no measurement errors
    dets = dec.circuit.compile_detector_sampler(seed=3).sample(200).astype(np.uint8)
    for row in dets:
        corr, time = sri.link_decisions(row[sri.link_cols])
        assert not time
        assert sorted(corr) == sorted(np.flatnonzero(row[sri.link_cols]).tolist())


@pytest.mark.parametrize("mode", ["sri_m", pytest.param(
    "sri_bm", marks=pytest.mark.skipif(not HAS_LDPC, reason="ldpc missing"))])
def test_trivial_syndrome(mode):
    dec = _decoder(CircuitNoise.uniform(2e-3), 3, mode)
    zeros = np.zeros((3, dec.circuit.num_detectors), dtype=np.uint8)
    assert not dec.decode_batch(zeros).any()
