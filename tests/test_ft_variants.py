"""Tests for the extraction variants of the fair comparison (docs/fair_comparison.md):
hybrid links, repeated links, noiseless links, phenomenological noise, and the
memory of the conjugate logical operator (logical="Y")."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from mdr.ft import CircuitNoise, DEPTH6_SCHEDULE, FTMDRCircuit
from mdr.ft.ft_mdr_circuit import REP_PHASE, hybrid_link_steps, link_rep_slots
from mdr.ft.two_level_decoder import TwoLevelDecoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_decoder_threshold_sweep import DECODERS, NOISE  # noqa: E402

HAS_LDPC = importlib.util.find_spec("ldpc") is not None
HAS_TESSERACT = importlib.util.find_spec("tesseract_decoder") is not None
# The commit before the variants were implemented: every circuit and every decoder
# output of the models that existed then must stay exactly as it was.
REFERENCE_COMMIT = "33fa1d0"
OLD_NOISES = ["sd6", "si1000", "biased10", "biased100", "purez", "helios", "h2", "helios_cb",
              "helios_p", "h2_p", "helios_p_noxt", "h2_p_noxt", "em3"]
NEW_NOISES = ["hyb", "hyb_lr2", "sd6_lr2", "sd6_lr3", "si1000_lr2", "sd6_il", "phen", "phen_b10"]


def _value(name: str) -> float:
    if name in ("helios", "h2", "helios_cb"):
        return 1.0
    return 3e-2 if name.startswith("phen") else 3e-3


@pytest.fixture(scope="module")
def ref():
    """The src/mdr/ft package of REFERENCE_COMMIT, imported as `ftref`."""
    if shutil.which("git") is None:
        pytest.skip("git not available")
    out = ROOT / ".pytest_tmp" / f"ftref_{REFERENCE_COMMIT}"
    pkg = out / "src" / "mdr" / "ft"
    if not (pkg / "__init__.py").exists():
        out.mkdir(parents=True, exist_ok=True)
        try:
            archive = subprocess.run(["git", "-C", str(ROOT), "archive", REFERENCE_COMMIT, "src/mdr/ft"],
                                     check=True, capture_output=True).stdout
            subprocess.run(["tar", "-x", "-C", str(out)], input=archive, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip(f"commit {REFERENCE_COMMIT} not available")
    name = f"ftref_{REFERENCE_COMMIT}"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, pkg / "__init__.py",
                                                      submodule_search_locations=[str(pkg)])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    mod = sys.modules[name]
    importlib.import_module(name + ".two_level_decoder")
    return mod


# ---------------------------------------------------------------- regression
@pytest.mark.parametrize("name", OLD_NOISES)
@pytest.mark.parametrize("d", [3, 5])
def test_existing_circuits_are_unchanged(ref, name, d):
    for rounds in sorted({1, 3, d}):
        for detectors in ("s0", "combined"):
            kw = dict(final="frame", detectors=detectors)
            new = FTMDRCircuit(d, rounds, NOISE[name](_value(name)), **kw).build()
            old = ref.FTMDRCircuit(d, rounds, NOISE[name](_value(name)), **kw).build()
            assert str(new) == str(old), (name, d, rounds, detectors)


@pytest.mark.parametrize("name", ["sd6", "si1000", "helios_cb", "em3"])
@pytest.mark.parametrize("kw", [dict(final="ideal", detectors="combined"), dict(final="ideal", detectors="all"),
                                dict(detectors="all"), dict(init="plus", detectors="all"),
                                dict(odd_rep="YZ", detectors="combined")])
def test_existing_circuit_options_are_unchanged(ref, name, kw):
    for d, rounds in ((3, 3), (5, 2)):
        new = FTMDRCircuit(d, rounds, NOISE[name](_value(name)), **kw).build()
        old = ref.FTMDRCircuit(d, rounds, NOISE[name](_value(name)), **kw).build()
        assert str(new) == str(old)


def test_shared_ancillas_are_unchanged(ref):
    for d, k in ((3, 1), (5, 2)):
        nz = CircuitNoise.uniform(2e-3)
        new = FTMDRCircuit(d, 3, nz, share_ancillas=k, detectors="combined").build()
        old = ref.FTMDRCircuit(d, 3, nz, share_ancillas=k, detectors="combined").build()
        assert str(new) == str(old)


def _modes():
    out = [("mwpm", DECODERS["mwpm"]), ("corr_links", DECODERS["corr_links"]),
           ("corr_gauge", DECODERS["corr_gauge"]), ("seq_match", DECODERS["seq_match"])]
    if HAS_LDPC:
        out += [(k, DECODERS[k]) for k in ("seq_soft", "seq_erasure", "bp_full", "bm", "bp_corr")]
        out += [("cfe0", DECODERS["cfe0"]), ("cfe", dict(DECODERS["cfe"], osd_order=4))]
        out += [("tnml", dict(mode="tnml", chi=8, chi_max=16)),
                ("cfe_tn", dict(DECODERS["cfe_tn"], chi=8, chi_max=16))]
    if HAS_TESSERACT:
        out += [("tesseract", DECODERS["tesseract"])]
    return out


@pytest.mark.parametrize("mode,kw", _modes(), ids=[m for m, _ in _modes()])
@pytest.mark.parametrize("name", ["sd6", "em3", "biased10"])
def test_decoders_are_unchanged_on_existing_circuits(ref, mode, kw, name):
    d, rounds = 3, (1 if mode in ("tnml", "cfe_tn") else 3)
    p = 6e-3 if name != "em3" else 4e-3
    ft = FTMDRCircuit(d, rounds, NOISE[name](p), detectors="combined")
    ft_old = ref.FTMDRCircuit(d, rounds, NOISE[name](p), detectors="combined")
    shots = 40 if mode in ("cfe", "cfe_tn", "tnml") else 200
    dets, _ = ft.build().compile_detector_sampler(seed=11).sample(shots, separate_observables=True)
    new = TwoLevelDecoder(ft, **kw).decode_batch(dets)
    old = sys.modules[ref.__name__ + ".two_level_decoder"].TwoLevelDecoder(ft_old, **kw).decode_batch(dets)
    assert dets.any(axis=1).sum() > shots // 4
    np.testing.assert_array_equal(new, old)


# ---------------------------------------------------------------- new variants
def _noiseless(name: str) -> CircuitNoise:
    nz = NOISE[name](1e-3)
    return replace(nz, p1=0.0, p2=0.0, p_prep=0.0, p_meas=0.0, p_idle=0.0, p_mpp=0.0,
                   p_idle_mr=None if nz.p_idle_mr is None else 0.0,
                   data_xyz=None if nz.data_xyz is None else (0.0, 0.0, 0.0))


@pytest.mark.parametrize("name", NEW_NOISES + ["sd6", "em3"])
@pytest.mark.parametrize("logical", ["X", "Y"])
@pytest.mark.parametrize("d", [3, 5])
def test_noiseless_variants_are_deterministic(name, logical, d):
    for rounds in sorted({1, 3, d}):
        for detectors, final in (("s0", "frame"), ("combined", "frame"), ("all", "frame"),
                                 ("combined", "ideal")):
            c = FTMDRCircuit(d, rounds, _noiseless(name), final=final, detectors=detectors,
                             logical=logical).build()
            dets, obs = c.compile_detector_sampler(seed=2).sample(64, separate_observables=True)
            assert not dets.any() and not obs.any(), (name, logical, d, rounds, detectors, final)


def test_repeated_link_slots_of_depth6():
    assert link_rep_slots(DEPTH6_SCHEDULE, 2) == {0: [(0, 1), (4, 5)], 1: [(0, 1), (2, 3)]}
    assert link_rep_slots(DEPTH6_SCHEDULE, 3) == {0: [(0, 1), (4, 5), (5, 6)],
                                                  1: [(0, 1), (2, 3), (3, 4)]}
    assert FTMDRCircuit(3, 1, NOISE["sd6_lr2"](1e-3)).depth == 6
    assert FTMDRCircuit(3, 1, NOISE["sd6_lr3"](1e-3)).depth == 7
    assert hybrid_link_steps(DEPTH6_SCHEDULE, 2) == {0: ["R", 5], 1: ["R", 3]}


@pytest.mark.parametrize("name,reps", [("sd6_lr2", 2), ("sd6_lr3", 3), ("hyb_lr2", 2)])
def test_repeated_links_add_lower_level_detectors(name, reps):
    d, rounds = 3, 3
    base = FTMDRCircuit(d, rounds, NOISE["sd6"](1e-3), detectors="combined").build()
    ft = FTMDRCircuit(d, rounds, NOISE[name](1e-3), detectors="combined")
    c = ft.build()
    n_links = d * d
    assert c.num_detectors == base.num_detectors + (reps - 1) * n_links * rounds
    rep_dets = [k for k, v in c.get_detector_coordinates().items() if int(v[2]) == REP_PHASE]
    assert len(rep_dets) == (reps - 1) * n_links * rounds
    dec = TwoLevelDecoder(ft, mode="mwpm")
    assert set(rep_dets) <= set(int(x) for x in dec.gauge_cols)
    assert set(rep_dets) <= set(int(x) for x in dec.link_cols)
    # the upper level (S_0 detectors) is the one of the circuit without repetitions
    assert len(dec.s0_cols) == d * d * (rounds + 1)


def test_hybrid_link_measurement_is_the_pair_measurement_of_em3():
    p = 2e-3
    hyb = str(FTMDRCircuit(3, 1, CircuitNoise.hybrid(p)).build())
    em3 = str(FTMDRCircuit(3, 1, CircuitNoise.em3(p)).build())

    def first_link_block(text):
        lines = text.splitlines()
        k = next(i for i, ln in enumerate(lines) if ln.startswith("MPP") and ln.count("X") == 2)
        start = max(i for i in range(k) if lines[i].startswith("R "))
        return [ln.split(" ", 1)[0] for ln in lines[start:k + 1]]

    blk = first_link_block(hyb)
    assert blk == first_link_block(em3)
    assert blk[0] == "R" and blk[-1] == "MPP" and len(blk) == 33       # R, 31 error branches, MPP
    # no link ancilla: plaquette ancillas only, and no CNOT on the links
    ft = FTMDRCircuit(3, 1, CircuitNoise.hybrid(p))
    assert len(ft.ancillas) == 8 and ft.physical_qubits() == 18 + 8


def test_hybrid_rounds_have_the_steps_of_sd6():
    for name in ("hyb", "hyb_lr2"):
        a = FTMDRCircuit(5, 5, NOISE[name](1e-3)).build()
        b = FTMDRCircuit(5, 5, NOISE["sd6"](1e-3)).build()
        assert a.num_ticks == b.num_ticks


def test_noiseless_links_have_no_noise_on_link_operations():
    nz = NOISE["sd6_il"](1e-3)
    ft = FTMDRCircuit(3, 2, nz)
    link_anc = {ft.ancilla_of[ci] for ci in ft.links}
    c = ft.build()
    for inst in c.flattened():
        if inst.name in ("DEPOLARIZE1", "DEPOLARIZE2", "Z_ERROR", "X_ERROR"):
            qs = {t.value for t in inst.targets_copy()}
            assert not qs & link_anc, inst


def test_phenomenological_circuit():
    p, d, rounds = 0.03, 3, 3
    nz = CircuitNoise.phenomenological(p, 10)
    c = FTMDRCircuit(d, rounds, nz, detectors="combined").build()
    text = str(c)
    px, py, pz = nz.data_xyz
    assert pz / (px + py) == pytest.approx(10) and px + py + pz == pytest.approx(p)
    assert text.count("PAULI_CHANNEL_1") == rounds
    m = 2 * d * d - 1

    def noisy_products(circ):
        return sum(len(inst.target_groups()) for inst in circ.flattened()
                   if inst.name == "MPP" and inst.gate_args_copy() == [p])

    assert noisy_products(c) == m * rounds
    assert c.num_measurements == m * rounds + 2 * d * d
    # no other noise: frame preparation and readout are perfect
    assert "DEPOLARIZE" not in text and "_ERROR" not in text
    # links without noise
    c_il = FTMDRCircuit(d, rounds, replace(nz, link_noise=False)).build()
    assert noisy_products(c_il) == (m - d * d) * rounds


@pytest.mark.parametrize("kw,err", [
    (dict(noise=replace(CircuitNoise.em3(1e-3), link_reps=2)), NotImplementedError),
    (dict(noise=replace(CircuitNoise.hybrid(1e-3), link_reps=4)), NotImplementedError),
    (dict(noise=replace(CircuitNoise.uniform(1e-3), link_reps=0)), ValueError),
    (dict(noise=replace(CircuitNoise.uniform(1e-3), native="cats")), ValueError),
    (dict(noise=CircuitNoise.hybrid(1e-3), share_ancillas=1), ValueError),
    (dict(noise=replace(CircuitNoise.uniform(1e-3), link_reps=2), share_ancillas=1), ValueError),
    (dict(noise=CircuitNoise.uniform(1e-3), logical="Z"), ValueError),
    (dict(noise=CircuitNoise.uniform(1e-3), logical="Y", init="plus", detectors="all"), ValueError),
])
def test_unsupported_combinations_raise(kw, err):
    with pytest.raises(err):
        FTMDRCircuit(3, 2, **kw).build()


# ---------------------------------------------------------------- second logical basis
@pytest.mark.parametrize("d", [3, 5, 7])
def test_conjugate_frame_fixes_logical_y(d):
    from mdr.ft import XYZ2FrameBasis, XYZ2Geometry

    geo = XYZ2Geometry(d)
    fy = XYZ2FrameBasis(geo, logical="Y")
    fx = XYZ2FrameBasis(geo)
    assert fy.logical_in_frame() and fx.logical_in_frame()
    assert fx.logical_spec == geo.logicals["Logical X"]
    assert fy.s0_rows.shape[0] == d * d - 1
    assert len(fy.logical_spec.split()) == d + 1

    def vec(spec):
        v = {}
        for t in spec.split():
            v[int(t[1:])] = t[0]
        return v

    # the representative is Logical Y times a stabilizer: it anticommutes with Logical X
    lx, ly = vec(geo.logicals["Logical X"]), vec(fy.logical_spec)
    anti = sum(1 for q in set(lx) & set(ly) if lx[q] != ly[q])
    assert anti % 2 == 1
    # the frames swap the parities of the blocks
    for i in range(d):
        for j in range(d):
            up, lo = geo.verts(i, j)
            assert (fy.basis[up] == "X") == ((i + j) % 2 == 1) == (fx.basis[up] != "X")


def test_depth6_schedule_halves_the_distance_of_the_conjugate_basis():
    """DEPTH6_SCHEDULE: fault distance d + 1 for Logical X but 2 for Logical Y at d = 3, which is why
    BOTH_BASES_SCHEDULE exists (distance >= d in both bases)."""
    pytest.importorskip("ortools")
    efd = _efd()
    c = FTMDRCircuit(3, 3, CircuitNoise.uniform(1e-3), logical="Y").build()
    assert efd.exists_logical_error(c, 2, time_limit=300)[0] in ("OPTIMAL", "FEASIBLE")


# ---------------------------------------------------------------- fault distance
def _efd():
    spec = importlib.util.spec_from_file_location("efd", ROOT / "scripts" / "exact_fault_distance.py")
    efd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(efd)
    return efd


@pytest.mark.parametrize("name", NEW_NOISES)
@pytest.mark.parametrize("logical", ["X", "Y"])
def test_fault_distance_d3(name, logical):
    """Fault distance d = 3 of every variant in both bases, upper level alone (S_0
    detectors, what matching sees) and the whole circuit. logical="Y" uses
    BOTH_BASES_SCHEDULE: DEPTH6_SCHEDULE has distance (d + 1) / 2 in that basis."""
    pytest.importorskip("ortools")
    from mdr.ft import BOTH_BASES_SCHEDULE

    efd = _efd()
    sch = BOTH_BASES_SCHEDULE if logical == "Y" else DEPTH6_SCHEDULE
    for rounds in (1, 3):
        for detectors in ("s0", "combined"):
            c = FTMDRCircuit(3, rounds, NOISE[name](_value(name)), schedule=sch, detectors=detectors,
                             logical=logical).build()
            assert efd.exists_logical_error(c, 2, time_limit=300)[0] == "INFEASIBLE", (rounds, detectors)


# ---------------------------------------------------------------- decoders
@pytest.mark.parametrize("name", NEW_NOISES)
@pytest.mark.parametrize("logical", ["X", "Y"])
def test_every_decoder_decodes_a_trivial_syndrome(name, logical):
    for mode, kw in _modes():
        rounds = 1 if mode in ("tnml", "cfe_tn") else 2
        ft = FTMDRCircuit(3, rounds, NOISE[name](_value(name)), detectors="combined", logical=logical)
        dec = TwoLevelDecoder(ft, **kw)
        zeros = np.zeros((2, dec.circuit.num_detectors), dtype=np.uint8)
        assert not dec.decode_batch(zeros).any(), mode
