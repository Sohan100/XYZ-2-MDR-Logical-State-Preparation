"""
sweep.py
----------------------------------------------------------------------------
Phenomenological threshold sweeps of XYZ^2 (FT MDR circuits) and XZZX for the
check of arXiv:2505.03691 (Srivastava et al., "Sequential decoding of the
XYZ^2 hexagonal stabilizer code"). Runs a grid of points on all cores and
appends one CSV row per point (resumable: points already in the file are
skipped).

Noise models (all with q = p on every check outcome, data noise of total rate p
with eta = p_z / (p_x + p_y) and p_x = p_y, noiseless extraction and a
noiseless final data readout):

- ``phen``, ``phen_b10``: `CircuitNoise.phenomenological(p, eta)` of the repo,
  rounds = d noisy rounds starting from the product (frame) state.
- ``phen_cs``, ``phen_b10_cs``: the same model with one extra *noiseless* round
  in front, so the d noisy rounds start from a code state (a memory
  experiment, as in the paper). Implemented here by stripping the noise of the
  first round of the circuits built with rounds = d + 1; no file under
  src/mdr/ft is changed.

    python cloud/phen_2505_03691/sweep.py --grid main --out cloud/phen_2505_03691/data/results.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import stim  # noqa: E402

from mdr.ft import BOTH_BASES_SCHEDULE, CircuitNoise, FTMDRCircuit  # noqa: E402
from mdr.ft import competitor_circuits as cc  # noqa: E402
from mdr.ft.competitor_decoders import competitor_decoder  # noqa: E402
from mdr.ft.two_level_decoder import TwoLevelDecoder  # noqa: E402
from run_decoder_threshold_sweep import DECODERS  # noqa: E402

ETA = {"phen": 0.5, "phen_b10": 10.0, "phen_cs": 0.5, "phen_b10_cs": 10.0}


def noise(model: str, p: float) -> CircuitNoise:
    nz = CircuitNoise.phenomenological(p, ETA[model])
    if model.endswith("_cs"):
        from dataclasses import replace
        nz = replace(nz, name=nz.name + "_cs")
    return nz


def strip_first_round(c: stim.Circuit) -> stim.Circuit:
    """Remove the data noise and the outcome flips of the first phenomenological round."""
    out = stim.Circuit()
    seen = 0
    for inst in c:
        if isinstance(inst, stim.CircuitRepeatBlock):
            raise ValueError("REPEAT blocks are not expected in phen circuits")
        if inst.name == "PAULI_CHANNEL_1":
            seen += 1
            if seen == 1:
                continue
        if inst.name == "MPP" and seen <= 1:
            out.append("MPP", inst.targets_copy())
            continue
        out.append(inst)
    assert seen >= 2, "codespace start needs rounds >= 2"
    return out


# Codespace start: wrap the builders so every circuit built for a *_cs noise
# (also the S_0 circuit that TwoLevelDecoder builds internally) loses the
# noise of its first round.
_ft_build = FTMDRCircuit.build
_cc_build = cc.CompetitorCircuit.build


def _ft_build_cs(self):
    c = _ft_build(self)
    return strip_first_round(c) if self.noise.name.endswith("_cs") else c


def _cc_build_cs(self):
    c = _cc_build(self)
    return strip_first_round(c) if self.noise.name.endswith("_cs") else c


FTMDRCircuit.build = _ft_build_cs
cc.CompetitorCircuit.build = _cc_build_cs


def run_point(job):
    code, model, basis, dec, d, p, shots, max_errors = job
    rounds = d + 1 if model.endswith("_cs") else d
    t0 = time.time()
    nz = noise(model, p)
    if code == "xyz2":
        ft = FTMDRCircuit(d, rounds, nz, final="frame", detectors="combined",
                          schedule=BOTH_BASES_SCHEDULE, logical=basis)
        decoder = TwoLevelDecoder(ft, **DECODERS[dec])
        batch = 2000 if dec == "mwpm" else 500
    else:
        circ = cc.competitor_circuit(code, d, rounds, nz, basis=basis)
        decoder = competitor_decoder(circ, dec)
        batch = 5000 if dec == "mwpm" else 500
    est = decoder.estimate(max_shots=shots, max_errors=max_errors, batch=batch,
                           time_limit=3600)
    return [code, model, basis, dec, d, d, p, est.shots, est.errors,
            round(time.time() - t0, 1)]


def grid(name: str):
    """Jobs (code, model, basis, decoder, d, p, shots, max_errors)."""
    pts = lambda a, b, n: [round(a + (b - a) * k / (n - 1), 5) for k in range(n)]  # noqa: E731
    jobs = []
    if name == "coarse":
        for model, ps in (("phen", pts(0.026, 0.050, 7)), ("phen_b10", pts(0.040, 0.070, 7))):
            for (code, basis), d, p in itertools.product(
                    [("xyz2", "X"), ("xyz2", "Y"), ("xzzx", "X"), ("xzzx", "Z")], (3, 5, 7), ps):
                jobs.append((code, model, basis, "mwpm", d, p, 4000, 10**9))
                if d <= 5:
                    jobs.append((code, model, basis, "bm", d, p, 2000, 10**9))
        return jobs
    if name == "main":
        spec = {
            # model: (XYZ^2 window, XZZX window)
            "phen": (pts(0.032, 0.044, 5), pts(0.032, 0.044, 5)),
            "phen_b10": (pts(0.046, 0.062, 5), pts(0.046, 0.062, 5)),
        }
        for model, (wx, wz) in spec.items():
            for basis in "XY":
                for p in wx:
                    for d in (3, 5, 7, 9):
                        jobs.append(("xyz2", model, basis, "mwpm", d, p, 40000, 4000))
                    for d in (3, 5, 7):
                        jobs.append(("xyz2", model, basis, "bm", d, p, 12000 if d == 7 else 20000, 2000))
                        jobs.append(("xyz2", model, basis, "bp_full", d, p, 8000 if d == 7 else 12000, 2000))
            for basis in "XZ":
                for p in wz:
                    for d in (3, 5, 7, 9):
                        jobs.append(("xzzx", model, basis, "mwpm", d, p, 40000, 4000))
                    for d in (3, 5, 7):
                        jobs.append(("xzzx", model, basis, "bm", d, p, 20000, 2000))
        return jobs
    if name == "ext":
        # windows moved after the main grid: XYZ^2 MWPM under bias crosses near 3 %
        for basis in "XY":
            for p in pts(0.022, 0.042, 6):
                for d in (3, 5, 7, 9):
                    jobs.append(("xyz2", "phen_b10", basis, "mwpm", d, p, 40000, 4000))
            for d in (3, 5, 7, 9):
                for p in (0.024, 0.028):
                    jobs.append(("xyz2", "phen", basis, "mwpm", d, p, 40000, 4000))
            # sequential-like decoders (erasure passing, HCM with links)
            for dec in ("seq_match", "corr_links"):
                for model, ps in (("phen", pts(0.028, 0.044, 5)), ("phen_b10", pts(0.030, 0.050, 5))):
                    for p in ps:
                        for d in (3, 5, 7):
                            jobs.append(("xyz2", model, basis, dec, d, p, 20000, 2000))
        for dec in ("bm", "bp_full"):
            for p in (0.047, 0.050):
                for d in (3, 5, 7):
                    jobs.append(("xyz2", "phen", "Y", dec, d, p, 12000 if d < 7 else 8000, 2000))
        for p in (0.047,):
            for basis in "XZ":
                for d in (3, 5, 7):
                    jobs.append(("xzzx", "phen", basis, "bm", d, p, 20000, 2000))
        # d = 9 for belief-matching in the weaker basis near the crossing
        for model, basis, ps in (("phen", "X", (0.036, 0.039, 0.042)), ("phen_b10", "X", (0.052, 0.055, 0.058))):
            for p in ps:
                jobs.append(("xyz2", model, basis, "bm", 7, p, 12000, 2000))
                jobs.append(("xyz2", model, basis, "bm", 9, p, 8000, 2000))
        for model, basis, ps in (("phen", "X", (0.041, 0.044, 0.047)), ("phen_b10", "Z", (0.046, 0.050, 0.054)),
                                 ("phen_b10", "X", (0.054, 0.058, 0.062))):
            for p in ps:
                jobs.append(("xzzx", model, basis, "bm", 7, p, 20000, 2000))
                jobs.append(("xzzx", model, basis, "bm", 9, p, 10000, 2000))
        return jobs
    if name == "ext2":
        # finite-size drift of XZZX under bias: d = 9 BM above 5.4 %, d = 11 MWPM
        for p in (0.058, 0.062):
            jobs.append(("xzzx", "phen_b10", "Z", "bm", 9, p, 10000, 2000))
        for basis in "XZ":
            for p in (0.050, 0.054, 0.058, 0.062):
                jobs.append(("xzzx", "phen_b10", basis, "mwpm", 11, p, 30000, 3000))
            for p in (0.035, 0.038, 0.041, 0.044):
                jobs.append(("xzzx", "phen", basis, "mwpm", 11, p, 30000, 3000))
        for basis in "XY":
            for p in (0.026, 0.030, 0.034):
                jobs.append(("xyz2", "phen_b10", basis, "mwpm", 11, p, 30000, 3000))
            for p in (0.028, 0.032, 0.035):
                jobs.append(("xyz2", "phen", basis, "mwpm", 11, p, 30000, 3000))
        return jobs
    if name == "cs":
        # memory from a code state (paper convention)
        for model, wx, wm in (("phen_cs", pts(0.032, 0.044, 4), pts(0.026, 0.038, 4)),
                              ("phen_b10_cs", pts(0.046, 0.062, 4), pts(0.022, 0.038, 4))):
            for (code, basis) in (("xyz2", "X"), ("xyz2", "Y"), ("xzzx", "X"), ("xzzx", "Z")):
                for p in wx:
                    for d in (3, 5, 7):
                        if code == "xzzx" or model == "phen_cs":
                            jobs.append((code, model, basis, "mwpm", d, p, 30000, 3000))
                        jobs.append((code, model, basis, "bm", d, p, 8000 if d == 7 else 12000, 1500))
                if code == "xyz2":
                    for p in (wm if model == "phen_b10_cs" else wm[:2]):
                        for d in (3, 5, 7):
                            jobs.append((code, model, basis, "mwpm", d, p, 30000, 3000))
        return jobs
    raise ValueError(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--only", nargs="*", default=None, help="filter: code:model:decoder")
    args = ap.parse_args()
    header = ["code", "noise", "basis", "decoder", "d", "rounds", "p", "shots", "errors", "seconds"]
    done = set()
    if args.out.exists():
        with args.out.open() as fh:
            for r in csv.DictReader(fh):
                done.add((r["code"], r["noise"], r["basis"], r["decoder"], int(r["d"]), float(r["p"])))
    jobs = [j for j in grid(args.grid) if j[:6] not in done]
    if args.only:
        keep = {tuple(s.split(":")) for s in args.only}
        jobs = [j for j in jobs if (j[0], j[1], j[3]) in keep]
    # expensive first, so the pool ends evenly
    cost = lambda j: (j[3] != "mwpm") * 10 * j[4] ** 3 * j[6] / 1e4 + j[4] ** 3 * j[6] / 1e5  # noqa: E731
    jobs.sort(key=cost, reverse=True)
    print(f"{len(jobs)} jobs", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    new = not args.out.exists()
    with args.out.open("a", newline="") as fh, mp.Pool(args.workers) as pool:
        w = csv.writer(fh)
        if new:
            w.writerow(header)
        for row in pool.imap_unordered(run_point, jobs):
            w.writerow(row)
            fh.flush()
            print(" ".join(map(str, row)), flush=True)


if __name__ == "__main__":
    main()
