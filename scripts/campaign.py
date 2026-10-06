"""
campaign.py
----------------------------------------------------------------------------
Threshold campaign over every decoder, number of rounds, distance and noise model.

The unit of work is a task: one point (noise model, decoder, number of rounds r,
distance d, physical error rate p), or one replica of a point when the point is
expensive. A task stops at a target number of logical errors, a maximum number
of shots or a time budget, whichever comes first. While it runs, a task appends
its counts to a CSV every few minutes, so a task that is cut off at the end of a
Slurm job keeps its work and continues where it stopped when the job runs again.
`merge` sums the counts of all tasks and replicas of a point.

    python scripts/campaign.py tasks --out data/campaign/tasks.jsonl
    python scripts/campaign.py run data/campaign/tasks.jsonl data/campaign/points_0.csv \
        --chunk 0 --nchunks 1 --workers 8
    python scripts/campaign.py status data/campaign/tasks.jsonl "data/campaign/points_*.csv"
    python scripts/campaign.py merge "data/campaign/points_*.csv" --out docs/data/campaign/points.csv

On Perlmutter, slurm/ft_mdr/submit_campaign.sh submits everything; see
docs/nersc_campaign.md. A pool runs the unfinished tasks of several task files
on any number of nodes, each node taking units of tasks as it frees up
(slurm/ft_mdr/run_pool.sh runs it on 128-node jobs):

    python scripts/campaign.py pool data/campaign/pool1.jsonl --tasks data/campaign/tasks4.jsonl ...
    python scripts/campaign.py pool-run data/campaign/pool1.jsonl
    python scripts/campaign.py pool-status data/campaign/pool1.jsonl

Grids. The values of p of a series (noise, decoder, r, d) are a geometric
lattice around the expected threshold, c * 4^(k/13). Matching-type decoders use
k = -7..6 (14 points over a factor 4), the slow decoders k = -5..4 (10 points
over a factor 2.6). The expected thresholds come from earlier runs (r = d,
distances up to 19) and from the rounds study of matching.

Budgets. The time budget of a point grows with the size of the circuit,
S = (r + 1) d^2, as B = B21 * (S / S21)^alpha with S21 = 22 * 21^2, and is
halved (quartered) for points more than 20% (35%) below the expected
threshold, where many shots give few errors. Points with a budget above
--rep-hours are split into replicas with independent random seeds.
"""

from __future__ import annotations

import os

# one thread per process: the parallelism is over tasks
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import queue  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

NOISES = ["sd6", "si1000", "biased10", "biased100", "purez", "em3",
          "helios_p", "h2_p", "helios_p_noxt", "h2_p_noxt"]
DECODERS = ["mwpm", "corr_links", "corr_gauge", "seq_soft", "seq_erasure", "seq_match", "bm", "bp_full", "bp_corr",
            "tesseract", "cfe", "cfe0", "cfe_tn", "tnml"]
ROUNDS = [str(r) for r in range(1, 22)] + ["d"]
DISTANCES = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
# Tesseract's beam search grows too fast with the circuit beyond d = 9.
DMAX = {"tesseract": 9}
# CFE runs ldpc's OSD-CS decoding through src/mdr/ft/fast_osd.py (same decoding, memory and time
# that grow slowly), so it reaches d = 21 for every r. cfe_tn replaces the CFE decision by the
# converged tensor-network maximum-likelihood decision where the sweep frontier has at most
# TN_MAX_OPEN detectors; the frontier holds about (3.5 r - 1) d detectors, so cfe_tn tasks are made
# only where that estimate is below the limit (r = 1 up to d = 21, r = 2 up to d = 9, r = 3 at d = 5,
# r <= 6 and r = d at d = 3). The decoder checks the real width and falls back to CFE above it.
TN_MAX_OPEN = 60

# Threshold of matching for r = d at large d, and the ratio of the threshold for r rounds to that for
# r = d (SD6, matching). Used only to centre the grids.
P_MWPM = {"sd6": 4.35e-3, "si1000": 3.67e-3, "biased10": 4.49e-3, "biased100": 4.57e-3,
          "purez": 4.53e-3, "em3": 2.3e-3, "helios_p_noxt": 2.8e-3, "h2_p_noxt": 4.4e-3}
F_ROUNDS = {"1": 4.14, "2": 2.80, "3": 2.23, "4": 1.91, "5": 1.72, "6": 1.63, "8": 1.47, "10": 1.40, "d": 1.0}


def f_rounds(r: str) -> float:
    """F_ROUNDS, and for the other numbers of rounds the fit 1.055 + 3.45 / r of its values for r >= 6."""
    return F_ROUNDS[r] if r in F_ROUNDS else 1.055 + 3.45 / int(r)

# threshold of each decoder relative to matching (measured for r = d; guesses for seq_* and the ion models)
_G = {
    "corr_links": dict(sd6=1.17, si1000=1.19, biased10=1.26, biased100=1.29, purez=1.29, em3=1.2),
    "corr_gauge": dict(sd6=1.18, si1000=1.20, biased10=1.39, biased100=1.45, purez=1.49, em3=1.28),
    "seq_soft": {}, "seq_match": {},
    "bm": dict(sd6=1.28, si1000=1.25, biased10=1.76, biased100=1.89, purez=1.85, em3=1.3),
}
_G_DEFAULT = {"mwpm": 1.0, "corr_links": 1.18, "corr_gauge": 1.2, "seq_soft": 1.05, "seq_match": 0.9, "bm": 1.3}
_G["tesseract"] = _G["cfe"] = _G["cfe0"] = _G["cfe_tn"] = _G["tnml"] = _G["bp_full"] = _G["bp_corr"] = _G["bm"]
_G_DEFAULT["tesseract"] = _G_DEFAULT["cfe"] = _G_DEFAULT["cfe0"] = _G_DEFAULT["cfe_tn"] = 1.3
_G_DEFAULT["tnml"] = _G_DEFAULT["bp_full"] = _G_DEFAULT["bp_corr"] = 1.3
_G["seq_erasure"], _G_DEFAULT["seq_erasure"] = {}, 1.05
# The crosstalk models have no threshold: the crossings of consecutive distances drift to small p.
XT_RANGE = {"helios_p": (6e-5, 3e-3), "h2_p": (5e-4, 6e-3)}

STEP = 4.0 ** (1 / 13)
KRANGE = {"wide": range(-7, 7), "narrow": range(-5, 5)}
SLOW = {"seq_soft", "seq_erasure", "bm", "bp_full", "bp_corr", "tesseract", "cfe", "cfe0", "cfe_tn", "tnml"}
TN = {"cfe_tn", "tnml"}           # decoders with the tensor network: tasks only where it can be contracted

TARGET = {"mwpm": 500, "corr_links": 400, "corr_gauge": 400, "seq_soft": 250, "seq_match": 300,
          "bm": 250, "tesseract": 150, "cfe": 200, "cfe0": 150, "cfe_tn": 200,
          "seq_erasure": 250, "bp_full": 250, "bp_corr": 250, "tnml": 200}
MAX_SHOTS = 2_000_000
S21 = 22 * 21 * 21
S9 = 10 * 9 * 9
# budget in seconds at S21 (S9 for Tesseract), exponent, smallest budget
BUDGET = {"mwpm": (600, 1.15, 30), "corr_links": (1200, 1.15, 60), "corr_gauge": (1800, 1.15, 60),
          "seq_match": (1800, 1.1, 60), "seq_soft": (5400, 1.1, 120), "bm": (5400, 1.1, 120),
          "tesseract": (7200, 2.0, 120), "cfe": (60_000, 1.3, 300), "cfe0": (60_000, 1.4, 300),
          "cfe_tn": (150_000, 1.0, 600), "seq_erasure": (5400, 1.1, 120), "bp_full": (5400, 1.1, 120),
          "bp_corr": (5400, 1.1, 120), "tnml": (150_000, 1.0, 600)}
# memory in GB: base + slope * S / S21 (measured on d = 9 to 21 circuits, with margin); CFE: see memory()
MEMORY = {"mwpm": (0.4, 1.0), "corr_links": (0.4, 1.2), "corr_gauge": (0.4, 1.2), "seq_match": (0.4, 1.2),
          "seq_soft": (0.4, 1.8), "bm": (0.4, 2.0), "tesseract": (0.3, 0.162),
          "seq_erasure": (0.4, 1.8), "bp_full": (0.4, 2.0), "bp_corr": (0.4, 2.0)}
# Peak memory measured on Perlmutter CPU nodes (docs/data/campaign/memory_probe.csv: the largest task of every
# decoder and noise model at the lowest and highest p of its grid, and CFE-0 from S = 810 to 9702). A batch
# holds at most BATCH_BITS detector bits, and the fast decoders keep up to 6.2 bytes per sampled bit while
# they decode it (BATCH_GB adds 8 bytes per bit to their estimate). The peak of every CFE variant is the
# decoder construction (degeneracy moves, BP and ldpc's OSD-0, or the fast OSD-CS): up to 2.9e-5 GB per
# fault mechanism, 9.5 GB at d = 21, r = 21 (CFE0_GB_PER_MECH = 3.8e-5 with a margin of 1.3). Tesseract
# (MEMORY) was measured from d = 9 to 21 at the highest p: 1.44 GB at d = 21, r = 21, about linear in S.
BATCH_BITS = 25_000_000
BATCH_GB = 8 * BATCH_BITS / 1e9
FAST = {"mwpm", "corr_links", "corr_gauge", "seq_match", "bp_corr"}
CFE0_GB_PER_MECH = 3.8e-5

# Other codes under the same noise models (src/mdr/ft/competitor_circuits.py): a series "code-basis:decoder",
# e.g. "xzzx-X:mwpm", is the memory of that code in that basis decoded by that decoder
# (src/mdr/ft/competitor_decoders.py). Budgets, targets and memory are those of the XYZ^2 decoder of the
# same kind; the competitor circuits are at most as large as ours at the same d and r.
COMPETITOR_CODES = ("css", "xzzx", "xy", "honeycomb")
COMPETITOR_DECODERS = {"mwpm": "mwpm", "corr": "corr_links", "bm": "bm", "bposd": "cfe0", "tesseract": "tesseract"}


def competitor(decoder: str):
    """(code, basis, decoder) of a competitor series "code-basis:decoder", None for an XYZ^2 decoder."""
    if ":" not in decoder:
        return None
    cb, name = decoder.split(":", 1)
    code, basis = cb.split("-", 1)
    if code not in COMPETITOR_CODES or name not in COMPETITOR_DECODERS or basis not in ("X", "Z"):
        raise ValueError(f"unknown competitor series {decoder!r}")
    return code, basis, name


def base_decoder(decoder: str) -> str:
    """The XYZ^2 decoder whose budget, target and memory a series uses."""
    comp = competitor(decoder)
    return decoder if comp is None else COMPETITOR_DECODERS[comp[2]]


COLS = ["noise", "value", "d", "rounds", "final", "decoder", "shots", "errors", "p_L", "stderr", "seconds"]
RAW = COLS + ["task", "note"]
KEYS = ["noise", "value", "d", "rounds", "final", "decoder"]


# --------------------------------------------------------------------------- tasks
def n_rounds(r: str, d: int) -> int:
    return d if r == "d" else int(r)


def center(noise: str, decoder: str, r: str) -> float:
    if noise in XT_RANGE:
        lo, hi = XT_RANGE[noise]
        return math.sqrt(lo * hi)
    base = noise.replace("_noxt", "") if noise.endswith("_noxt") else noise
    g = _G.get(decoder, {}).get(base, _G_DEFAULT.get(decoder, 1.0)) if decoder != "mwpm" else 1.0
    return P_MWPM[noise] * f_rounds(r) * g


def grid(noise: str, decoder: str, r: str, c: float | None = None, wide: bool = False) -> list:
    """p values of a series: geometric around the centre `c` (default: the expected threshold).
    `wide` gives every decoder the 14 points over a factor 4 of the matching decoders."""
    if noise in XT_RANGE:
        lo, hi = XT_RANGE[noise]
        lo, hi = lo * f_rounds(r) ** 0.5, hi * f_rounds(r)
        n = 14 if (decoder not in SLOW or wide) else 10
        return [float(f"{x:.4g}") for x in np.geomspace(lo, hi, n)]
    c = center(noise, decoder, r) if c is None else c
    ks = KRANGE["narrow" if (decoder in SLOW and not wide) else "wide"]
    return [float(f"{c * STEP ** k:.4g}") for k in ks]


# decoders without measured thresholds take the centres of a close relative
PROXY = {"bp_full": "bm", "bp_corr": "bm", "seq_erasure": "seq_soft", "tnml": "cfe_tn"}
R_EFF_D = 18          # the r = d threshold, set by d ~ 15 to 21, stands for about 18 rounds


def measured_centers(thresholds_csv: str, points_csv: str) -> dict:
    """{(noise, decoder, rounds): centre} from the thresholds of an earlier analysis (drift left out).
    A threshold within 8% of the edge of the p values it was fitted on is moved out by 20%."""
    import pandas as pd

    th = pd.read_csv(thresholds_csv)
    th = th[th.pth.notna() & (th.flag.fillna("") != "drift")]
    pts = pd.read_csv(points_csv)
    pts["rk"] = np.where(pts.rounds == pts.d, "d", pts.rounds.astype(str))
    rng = pts.groupby(["noise", "decoder", "rk"]).value.agg(["min", "max"])
    out = {}
    for row in th.itertuples():
        key = (row.noise, row.decoder, str(row.rounds))
        c = float(row.pth)
        if (key[0], key[1], key[2]) in rng.index:
            lo, hi = rng.loc[(key[0], key[1], key[2])]
            c = c * 0.8 if c <= 1.08 * lo else (c * 1.25 if c >= 0.92 * hi else c)
        out[key] = c
    return out


def center_from(meas: dict, noise: str, decoder: str, r: str) -> float:
    """Centre of a series from measured thresholds: the measured value, else interpolated in log r between
    the measured numbers of rounds (towards the r = d value, placed at R_EFF_D rounds), else `center`."""
    if noise in XT_RANGE:
        return center(noise, decoder, r)
    for dec in (decoder, PROXY.get(decoder)):
        if dec is None:
            continue
        have = {int(k[2]): v for k, v in meas.items() if k[0] == noise and k[1] == dec and k[2] != "d"}
        cd = meas.get((noise, dec, "d"))
        if r == "d":
            if cd is not None:
                return cd
            continue
        if r in {str(x) for x in have}:
            return have[int(r)]
        pts = sorted(have.items()) + ([(R_EFF_D, cd)] if cd is not None and (not have or max(have) < R_EFF_D) else [])
        if not pts:
            continue
        x = int(r)
        if x <= pts[0][0]:
            return pts[0][1]
        if x >= pts[-1][0]:
            return pts[-1][1]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= x <= x1:
                t = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
                return float(math.exp((1 - t) * math.log(y0) + t * math.log(y1)))
    return center(noise, decoder, r)


def budget(decoder: str, d: int, r: str) -> float:
    b21, alpha, bmin = BUDGET[decoder]
    s = (n_rounds(r, d) + 1) * d * d
    ref = S9 if decoder == "tesseract" else S21
    return max(bmin, b21 * (s / ref) ** alpha)


def memory(decoder: str, d: int, r: str) -> float:
    s = (n_rounds(r, d) + 1) * d * d
    if decoder in ("cfe", "cfe0", "cfe_tn", "tnml"):
        n = 33.0 * s                       # fault mechanisms of the detector error model
        # decoder construction, measured (see CFE0_GB_PER_MECH); cfe_tn adds the tensor network
        return round(0.6 + CFE0_GB_PER_MECH * n + (0.2 if decoder in TN else 0.0), 2)
    base, slope = MEMORY[decoder]
    ref = S9 if decoder == "tesseract" else S21
    return round(base + slope * s / ref + (BATCH_GB if decoder in FAST else 0.0), 2)


# Memory of a CFE task once its decoder is built, as a fraction of `memory`: at most 0.30 measured while
# decoding at the highest p of the grid, d = 17 and 21 (docs/data/campaign/memory_run_probe.csv), so 0.45
# keeps a margin of 1.3. The runner reserves `memory` until the decoder is built and `mem_run` after.
RUN_FRACTION = {"cfe": 0.45, "cfe0": 0.45}


def memory_run(decoder: str, d: int, r: str) -> float:
    return round(RUN_FRACTION.get(decoder, 1.0) * memory(decoder, d, r), 2)


def weight(noise: str, decoder: str, r: str, p: float, c: float | None = None) -> float:
    """Budget factor: points well below the expected threshold (centre `c`) get less time."""
    if noise in XT_RANGE:
        return 1.0
    x = p / (center(noise, decoder, r) if c is None else c)
    return 1.0 if x >= 0.8 else (0.5 if x >= 0.65 else 0.25)


def make_tasks(noises, decoders, rounds, distances, scale=1.0, cfe_scale=1.0, rep_hours=4.0,
               tn_max_open=TN_MAX_OPEN, exclude=(), tn_dmax=None, centers=None, wide=False, dmax=None) -> list:
    """`tn_dmax` ({rounds: largest d}, e.g. {"1": 21, "2": 5}) limits the tensor-network tasks further.
    `centers` (from measured_centers) centres every series on its measured threshold; `wide` gives every
    decoder the wide grid; `dmax` ({decoder: largest d}) replaces DMAX."""
    caps = dict(DMAX if dmax is None else dmax)
    out = []
    for noise in noises:
        for dec in decoders:
            for r in rounds:
                c = center_from(centers, noise, dec, r) if centers is not None else center(noise, dec, r)
                values = grid(noise, dec, r, c=c, wide=wide)
                for d in distances:
                    if d > caps.get(dec, 99):
                        continue
                    if dec in TN and (3.5 * n_rounds(r, d) - 1) * d > tn_max_open:
                        continue
                    if dec in TN and tn_dmax is not None and d > tn_dmax.get(r, 0):
                        continue
                    b = budget(dec, d, r) * scale * (cfe_scale if dec.startswith("cfe") else 1.0)
                    for p in values:
                        bp = max(BUDGET[dec][2], b * weight(noise, dec, r, p, c))
                        nrep = max(1, math.ceil(bp / (3600.0 * rep_hours)))
                        for i in range(nrep):
                            if f"{noise}|{dec}|r{r}|d{d}|p{p:.4g}|{i}" in exclude:
                                continue
                            out.append(dict(
                                id=f"{noise}|{dec}|r{r}|d{d}|p{p:.4g}|{i}", noise=noise, decoder=dec, rounds=r, d=d,
                                value=p, target=math.ceil(TARGET[dec] / nrep), max_shots=math.ceil(MAX_SHOTS / nrep),
                                budget=round(bp / nrep, 1), mem=memory(dec, d, r),
                                **({"mem_run": memory_run(dec, d, r)} if dec in RUN_FRACTION else {})))
    # long tasks first, so that no chunk ends with one long straggler
    out.sort(key=lambda t: (-t["budget"], t["id"]))
    return out


def estimate(tasks: list) -> dict:
    """Upper bound of the cost in core-hours (every task uses its whole budget) by decoder."""
    by = {}
    for t in tasks:
        by[t["decoder"]] = by.get(t["decoder"], 0.0) + t["budget"] / 3600.0
    return by


# --------------------------------------------------------------------------- progress
def read_raw(paths) -> list:
    rows = []
    for f in sorted(set(paths)):
        if os.path.exists(f) and os.path.getsize(f) > 0:
            with open(f, newline="") as fh:
                rows += list(csv.DictReader(fh))
    return rows


def progress(paths) -> dict:
    """Counts so far per task: {task id: [shots, errors, seconds, note]}, note = the failure or "done"."""
    prog = {}
    for r in read_raw(paths):
        tid = r.get("task")
        if not tid:
            continue
        try:
            shots, errors, secs = int(r["shots"]), int(r["errors"]), float(r["seconds"])
        except (TypeError, ValueError):   # a line that another job is still writing
            continue
        s = prog.setdefault(tid, [0, 0, 0.0, ""])
        s[0] += shots
        s[1] += errors
        s[2] += secs
        note = r.get("note") or ""
        if note.startswith("failed"):
            s[3] = note
        elif note == "done" and not s[3]:
            # the worker reached its error target, shot cap or budget (the seconds of its rows are
            # rounded, so their sum can fall a fraction of a second short of the budget)
            s[3] = "done"
    return prog


def finished(t: dict, s) -> bool:
    if s is None:
        return False
    if s[3].startswith("failed") or s[1] >= t["target"] or s[0] >= t["max_shots"] or s[2] >= t["budget"]:
        return True
    # "done" with the seconds a little short of the budget (each row rounds them) is the budget reached;
    # a task listed again with a larger budget (a later stage) goes on
    return s[3] == "done" and s[2] >= 0.99 * t["budget"] - 1.0


# --------------------------------------------------------------------------- worker
FLUSH = 300.0       # seconds between partial rows
BATCH = 8.0         # target seconds per sampled batch


def _row(t, shots, errors, seconds, note=""):
    rounds = n_rounds(t["rounds"], t["d"])
    pl = errors / shots if shots else 0.0
    se = math.sqrt(max(pl * (1 - pl), 1e-12) / shots) if shots else 0.0
    return [t["noise"], t["value"], t["d"], rounds, "frame", t["decoder"], shots, errors, pl, se,
            round(seconds, 1), t["id"], note]


def _work(t: dict, prev, q, ev=None) -> None:
    from mdr.ft import FTMDRCircuit
    from mdr.ft.two_level_decoder import TwoLevelDecoder
    from run_decoder_threshold_sweep import DECODERS as DEC, NOISE

    s_tot, e_tot, sec0 = (prev[0], prev[1], prev[2]) if prev else (0, 0, 0.0)
    t0 = last = time.time()
    try:
        noise = NOISE[t["noise"]](t["value"])
        comp = competitor(t["decoder"])
        if comp is not None:                        # another code: see COMPETITOR_DECODERS
            from mdr.ft.competitor_circuits import competitor_circuit
            from mdr.ft.competitor_decoders import competitor_decoder

            code, basis, name = comp
            circuit = competitor_circuit(code, t["d"], n_rounds(t["rounds"], t["d"]), noise, basis=basis)
            dec = competitor_decoder(circuit, name)
        else:
            ft = FTMDRCircuit(t["d"], n_rounds(t["rounds"], t["d"]), noise, final="frame", detectors="combined")
            dec = TwoLevelDecoder(ft, **DEC[t["decoder"]])
            circuit = dec.circuit
        sampler = circuit.compile_detector_sampler()
        # the memory of a batch must not grow with the decoding speed (see BATCH_BITS)
        max_size = int(np.clip(BATCH_BITS // max(circuit.num_detectors, 1), 1, 200_000))
        if ev is not None:
            ev.put(os.getpid())                     # the decoder is built: its construction peak is over
    except Exception as exc:  # noqa: BLE001  (e.g. a decoder package missing on this machine)
        q.put(_row(t, 0, 0, time.time() - t0, f"failed: {type(exc).__name__}: {exc}"[:300]))
        return
    shots = errors = 0
    size = 1
    try:
        while (e_tot < t["target"] and s_tot < t["max_shots"]
               and sec0 + time.time() - t0 < t["budget"]):
            n = int(min(size, t["max_shots"] - s_tot))
            tb = time.time()
            dets, obs = sampler.sample(n, separate_observables=True)
            k = int(np.sum(np.any(dec.decode_batch(dets) != obs, axis=1)))
            dt = max(time.time() - tb, 1e-3)
            shots += n
            errors += k
            s_tot += n
            e_tot += k
            size = int(np.clip(min(4 * n, n * BATCH / dt), 1, max_size))
            if time.time() - last >= FLUSH:
                q.put(_row(t, shots, errors, time.time() - last))
                shots = errors = 0
                last = time.time()
    except Exception as exc:  # noqa: BLE001
        q.put(_row(t, shots, errors, time.time() - last, f"failed: {type(exc).__name__}: {exc}"[:300]))
        return
    q.put(_row(t, shots, errors, time.time() - last, "done"))


def _writer(q, out: str) -> None:
    new = not os.path.exists(out) or os.path.getsize(out) == 0
    with open(out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(RAW)
            fh.flush()
        while True:
            row = q.get()
            if row is None:
                break
            w.writerow(row)
            fh.flush()


def node_memory_gb() -> float:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return 16.0


def run(tasks: list, out: str, workers: int, mem_gb: float, log=print) -> None:
    """Run tasks in fresh processes, at most `workers` at a time and within `mem_gb` of estimated memory."""
    # import the heavy packages once, so that every forked worker starts warm
    import mdr.ft  # noqa: F401
    import mdr.ft.two_level_decoder  # noqa: F401
    import run_decoder_threshold_sweep  # noqa: F401

    # progress of every chunk, so that a task keeps its counts if the number of chunks changes
    prog = progress(glob.glob(os.path.join(os.path.dirname(out) or ".", "points_*.csv")) + [out])
    todo = [t for t in tasks if not finished(t, prog.get(t["id"]))]
    prog = {t["id"]: prog[t["id"]] for t in todo if t["id"] in prog}
    log(f"{len(tasks)} tasks in this chunk, {len(todo)} to run, {workers} workers, {mem_gb:.0f} GB")
    # the forked workers share the parent's pages; keep the cyclic collector from touching (and so
    # copying) them in every worker
    import gc
    gc.collect()
    gc.freeze()
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    ev = ctx.Queue()            # pids of workers whose decoder is built (see mem_run)
    wt = threading.Thread(target=_writer, args=(q, out), daemon=True)
    wt.start()
    running = {}
    used = 0.0
    done = 0
    while todo or running:
        # a task reserves its construction peak `mem` until its decoder is built, then `mem_run`
        while True:
            try:
                pid = ev.get_nowait()
            except queue.Empty:
                break
            if pid in running:
                p, t, m = running[pid]
                m2 = min(m, t.get("mem_run", m))
                used -= m - m2
                running[pid] = (p, t, m2)
        for pid in list(running):
            p, t, m = running[pid]
            if not p.is_alive():
                p.join()
                used -= m
                del running[pid]
                done += 1
                if done % 50 == 0 or not todo:
                    log(f"{time.strftime('%H:%M:%S')} finished {done}, running {len(running)}, left {len(todo)}")
        i = 0
        while len(running) < workers and i < len(todo):
            t = todo[i]
            if used + t["mem"] <= mem_gb or not running:
                p = ctx.Process(target=_work, args=(t, prog.get(t["id"]), q, ev))
                p.start()
                running[p.pid] = (p, t, t["mem"])
                used += t["mem"]
                todo.pop(i)
            else:
                i += 1
        time.sleep(0.5)
    q.put(None)
    wt.join()


# --------------------------------------------------------------------------- pool
# A pool holds every unfinished task of some task files, with its counts so far ("prev"), in units of
# about a hundred tasks. A node claims a unit by creating a file next to the pool and refreshes the
# claim while it works, so any number of jobs of any size can work through one pool at once, and the
# unit of a node that stopped is taken over by another node POOL_STALE seconds later. The counts of a
# unit go to its own file, points_<pool>_u<unit>.csv, so whoever takes a unit over finds where its
# tasks stand in that one file.
POOL_STALE = 900.0      # seconds without a refresh after which a claim is free again
POOL_BEAT = 60.0        # seconds between refreshes of the claims a node holds
POOL_SCAN = 60.0        # seconds between looks for a free unit when none was found
POOL_WINDOW = 512       # a node takes its unit from this many of the first free units, at random


def pool_unit_out(pool: str, k: int) -> str:
    name = os.path.splitext(os.path.basename(pool))[0]
    return os.path.join(os.path.dirname(pool) or ".", f"points_{name}_u{k}.csv")


def pool_dirs(pool: str) -> tuple:
    stem = os.path.splitext(pool)[0]
    return stem + ".idx", stem + ".d/claims", stem + ".d/done"


def build_pool(task_files, points, out: str, unit_size: int = 128, seed: int = 0, first=()) -> tuple:
    """Write every unfinished task of `task_files`, with its counts in `points`, to the pool `out`; the tasks
    whose number of rounds is in `first` (e.g. "d") come before all others."""
    index, claims, done_dir = pool_dirs(out)
    if os.path.exists(os.path.dirname(claims)):
        raise SystemExit(f"{os.path.dirname(claims)} exists: claims and counts refer to a pool by its name, "
                         "so a new pool needs a new name")
    prog = progress(points)
    # a later stage can repeat a task with a larger budget (e.g. 4x): the largest one counts
    best = {}
    for f in task_files:
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    t = json.loads(line)
                    if t["id"] not in best or t["budget"] > best[t["id"]]["budget"]:
                        best[t["id"]] = t
    todo = []
    for t in best.values():
        s = prog.get(t["id"])
        if finished(t, s):
            continue
        if s:
            t["prev"] = s[:3]
        todo.append(t)
    del best
    # the rounds asked for first, then longest remaining budget first, in steps of ten minutes and at
    # random within a step: the pool ends with short tasks, and every unit mixes decoders, noise models
    # and sizes (and so memory)
    tie = np.random.default_rng(seed).random(len(todo))
    left = [round((t["budget"] - t.get("prev", [0, 0, 0.0])[2]) / 600.0) for t in todo]
    tier = [0 if t["rounds"] in set(first) else 1 for t in todo]
    order = sorted(range(len(todo)), key=lambda i: (tier[i], -left[i], tie[i]))
    offsets, pos = [], 0
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out + ".tmp", "w") as fh:
        for k, a in enumerate(range(0, len(order), unit_size)):
            line = json.dumps({"unit": k, "tasks": [todo[i] for i in order[a:a + unit_size]]}) + "\n"
            offsets.append(pos)
            fh.write(line)
            pos += len(line.encode())
    with open(index, "w") as fh:
        json.dump(offsets, fh)
    os.makedirs(claims)
    os.makedirs(done_dir)
    os.replace(out + ".tmp", out)
    return len(todo), len(offsets)


def _pool_writer(q, unit_of: dict, me: str) -> None:
    """Append the rows of every unit to its file; ("close", pool, unit, done) ends a unit, after its last row."""
    files = {}
    while True:
        item = q.get()
        if item is None:
            break
        if isinstance(item, tuple):
            _, pool, k, mark = item
            fh = files.pop((pool, k), None)
            if fh is not None:
                fh.close()
            _, claims, done_dir = pool_dirs(pool)
            path = os.path.join(claims, str(k))
            try:
                if mark and open(path).read().strip() == me:
                    os.rename(path, os.path.join(done_dir, str(k)))
            except OSError:
                pass
            continue
        key = unit_of.get(item[11])
        if key is None:                             # cannot happen, but keep the row
            key = (next(iter(unit_of.values()), ("orphans", "x"))[0], "x")
        if key not in files:
            out = pool_unit_out(*key)
            size = os.path.getsize(out) if os.path.exists(out) else 0
            fh = files[key] = open(out, "a", newline="")
            if size == 0:
                csv.writer(fh).writerow(RAW)
            else:
                with open(out, "rb") as rb:
                    rb.seek(size - 1)
                    if rb.read(1) != b"\n":     # a line cut off when an earlier node stopped
                        fh.write("\n")
        csv.writer(files[key]).writerow(item)
        files[key].flush()
    for fh in files.values():
        fh.close()


def pool_order(pool: str, order_file: str | None) -> list:
    """The pools to work on, first to last: those listed in `order_file` (one per line, relative to the
    file's folder; "#" starts a comment), then `pool` if the file does not list it."""
    pools = []
    if order_file:
        try:
            with open(order_file) as fh:
                for line in fh:
                    x = line.split("#", 1)[0].strip()
                    if x:
                        pools.append(os.path.normpath(os.path.join(os.path.dirname(order_file), x)))
        except OSError:
            pass
    if os.path.normpath(pool) not in pools:
        pools.append(os.path.normpath(pool))
    return pools


class _PoolView:
    """The units of one pool as one node sees them."""

    def __init__(self, path: str):
        self.path = path
        index, self.claims, self.done = pool_dirs(path)
        with open(index) as fh:
            self.offsets = json.load(fh)
        self.known_done = set()
        self.seen = {}              # unit -> time before which its claim by another node cannot be stale


def pool_run(pool: str, workers: int, mem_gb: float, log=print, max_seconds: float = 0.0,
             stale: float = POOL_STALE, beat: float = POOL_BEAT, scan: float = POOL_SCAN,
             order_file: str | None = None, refresh: float = 600.0) -> int:
    """Work through a pool on this node, like `run` on one chunk: claim a unit whenever every task claimed
    so far has started, and stop when no unit is free and no other node of this Slurm job holds one (its
    nodes are held until the last one ends, so they wait for units of nodes that stop). With `order_file`
    (read again every `refresh` seconds), the node takes its units from the pools listed there first, in
    that order, then from `pool`. Returns the number of units finished here."""
    import random
    import socket

    import mdr.ft  # noqa: F401
    import mdr.ft.two_level_decoder  # noqa: F401
    import run_decoder_threshold_sweep  # noqa: F401

    wait_until = time.time() + 3 * 3600         # a job submitted ahead of its pool waits for it
    while not any(os.path.exists(x) for x in pool_order(pool, order_file)):
        if time.time() > wait_until:
            raise SystemExit(f"{pool} does not exist")
        log(f"{time.strftime('%H:%M:%S')} waiting for {pool}")
        time.sleep(60)
    job = os.environ.get("SLURM_JOB_ID", "local")
    me = f"{job}.{os.environ.get('SLURM_RESTART_COUNT', '0')}.{socket.gethostname()}.{os.getpid()}"
    rng = random.Random(me)
    views = {}                      # pool path -> _PoolView
    order = []

    def refresh_order():
        nonlocal order
        order = []
        for x in pool_order(pool, order_file):
            if x not in views and os.path.exists(x):
                try:
                    views[x] = _PoolView(x)
                    log(f"{time.strftime('%H:%M:%S')} pool {x}: {len(views[x].offsets)} units")
                except (OSError, ValueError) as exc:
                    log(f"{time.strftime('%H:%M:%S')} pool {x}: cannot read it ({exc})")
                    continue
            if x in views:
                order.append(views[x])

    def claim_path(v, k):
        return os.path.join(v.claims, str(k))

    def owner(v, k):
        try:
            with open(claim_path(v, k)) as fh:
                return fh.read().strip()
        except OSError:
            return None

    def try_claim(v, k) -> bool:
        path = claim_path(v, k)
        for _ in range(2):
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                try:
                    mtime = os.stat(path).st_mtime
                except FileNotFoundError:
                    continue
                if time.time() - mtime < stale:
                    v.seen[k] = mtime + stale
                    return False
                # its node stopped: move the claim aside (one taker succeeds), check it, claim anew
                aside = f"{path}.stale.{me}"
                try:
                    os.rename(path, aside)
                except FileNotFoundError:
                    return False
                try:
                    if time.time() - os.stat(aside).st_mtime < stale:    # a fresh claim slipped in
                        try:
                            os.link(aside, path)
                        except FileExistsError:
                            pass
                        return False
                finally:
                    os.unlink(aside)
                log(f"{time.strftime('%H:%M:%S')} unit {k}: taking over the claim of a node that stopped")
                continue
            with os.fdopen(fd, "w") as fh:
                fh.write(me + "\n")
            if os.path.exists(os.path.join(v.done, str(k))):     # finished just before
                os.unlink(path)
                v.known_done.add(k)
                return False
            return True
        return False

    def next_unit():
        now = time.time()
        for v in order:
            free = [k for k in range(len(v.offsets)) if k not in v.known_done and v.seen.get(k, 0.0) <= now]
            if not free:
                continue
            start = rng.randrange(min(len(free), POOL_WINDOW))
            for k in free[start:] + free[:start]:
                if os.path.exists(os.path.join(v.done, str(k))):
                    v.known_done.add(k)
                elif try_claim(v, k):
                    return v, k
        return None

    def load_unit(v, k):
        with open(v.path, "rb") as fh:
            fh.seek(v.offsets[k])
            rec = json.loads(fh.readline())
        assert rec["unit"] == k, (rec["unit"], k)
        prog = progress([pool_unit_out(v.path, k)])
        out = []
        for t in rec["tasks"]:
            p0 = t.pop("prev", None) or [0, 0, 0.0]
            s = prog.get(t["id"]) or [0, 0, 0.0, ""]
            tot = [p0[0] + s[0], p0[1] + s[1], p0[2] + s[2], s[3]]
            if finished(t, tot):
                continue
            if max_seconds:
                t["budget"] = min(t["budget"], max_seconds)
            out.append((t, tot))
        return out

    def job_busy() -> bool:
        if job == "local":
            return False
        for v in views.values():
            try:
                names = os.listdir(v.claims)
            except OSError:
                continue
            for name in names:
                if not name.isdigit() or (v.path, int(name)) in left:
                    continue
                path = os.path.join(v.claims, name)
                try:
                    if time.time() - os.stat(path).st_mtime >= stale:
                        continue
                    with open(path) as fh:
                        if fh.read().split(".", 1)[0] == job:
                            return True
                except OSError:
                    continue
        return False

    import gc
    gc.collect()
    gc.freeze()
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    ev = ctx.Queue()
    unit_of = {}                    # task id -> (pool, unit)
    wt = threading.Thread(target=_pool_writer, args=(q, unit_of, me), daemon=True)
    wt.start()
    refresh_order()
    log(f"{me}: pools {[v.path for v in order]}, {workers} workers, {mem_gb:.0f} GB")
    pending = []                    # (task, counts so far, (pool, unit))
    running = {}                    # pid -> (process, task, memory reserved, (pool, unit))
    left = {}                       # (pool, unit) -> its tasks claimed here that have not ended
    lost = set()
    used = 0.0
    n_done = 0
    next_beat = time.time() + beat
    next_report = time.time() + 1800
    next_refresh = time.time() + refresh
    next_scan = 0.0
    idle_check = False

    def end_task(key):
        nonlocal n_done
        left[key] -= 1
        if left[key] == 0:
            del left[key]
            q.put(("close", key[0], key[1], key not in lost))
            if key not in lost:
                n_done += 1
                log(f"{time.strftime('%H:%M:%S')} unit {key[1]} of {key[0]} finished ({n_done} here), "
                    f"running {len(running)}")
            lost.discard(key)

    while True:
        now = time.time()
        while True:
            try:
                pid = ev.get_nowait()
            except queue.Empty:
                break
            if pid in running:
                p, t, m, key = running[pid]
                m2 = min(m, t.get("mem_run", m))
                used -= m - m2
                running[pid] = (p, t, m2, key)
        for pid in list(running):
            p, t, m, key = running[pid]
            if p.is_alive():
                continue
            p.join()
            used -= m
            del running[pid]
            if p.exitcode not in (0, -15):           # -15: the end of the job (SIGTERM)
                q.put(_row(t, 0, 0, 0.0, f"failed: worker exit code {p.exitcode} (-9: out of memory?)"))
            end_task(key)
        if now >= next_report:
            log(f"{time.strftime('%H:%M:%S')} running {len(running)}, waiting {len(pending)}, "
                f"{used:.0f} of {mem_gb:.0f} GB reserved, units held {len(left)}, finished {n_done}")
            next_report = now + 1800
        if now >= next_refresh:
            refresh_order()
            next_refresh = now + refresh
        if now >= next_beat:
            for key in list(left):
                if key in lost:
                    continue
                v = views[key[0]]
                if owner(v, key[1]) != me:
                    lost.add(key)
                    drop = [x for x in pending if x[2] == key]
                    pending = [x for x in pending if x[2] != key]
                    log(f"{time.strftime('%H:%M:%S')} lost the claim of unit {key[1]} of {key[0]}; "
                        f"leaving {len(drop)} tasks")
                    for _ in drop:
                        end_task(key)
                else:
                    try:
                        os.utime(claim_path(v, key[1]))
                    except OSError:
                        pass
            next_beat = now + beat
        if not pending and len(running) < workers and now >= next_scan:
            try:
                got = next_unit()
            except OSError as exc:                  # e.g. a file system hiccup: look again later
                log(f"{time.strftime('%H:%M:%S')} looking for a unit: {exc}")
                got = None
            if got is None:
                next_scan = now + scan
                idle_check = True
                refresh_order()                     # a pool may have been added
            else:
                v, k = got
                key = (v.path, k)
                try:
                    tasks = load_unit(v, k)
                except Exception as exc:  # noqa: BLE001  (a unit this node cannot read: leave it to others)
                    log(f"{time.strftime('%H:%M:%S')} unit {k} of {v.path}: cannot load it "
                        f"({type(exc).__name__}: {exc})")
                    v.known_done.add(k)
                    try:
                        os.unlink(claim_path(v, k))
                    except OSError:
                        pass
                    continue
                log(f"{time.strftime('%H:%M:%S')} unit {k} of {v.path}: {len(tasks)} tasks to run")
                if not tasks:
                    q.put(("close", v.path, k, True))
                    n_done += 1
                else:
                    left[key] = len(tasks)
                    for t, tot in tasks:
                        unit_of[t["id"]] = key
                        pending.append((t, tot, key))
        i = 0
        while len(running) < workers and i < len(pending):
            t, tot, key = pending[i]
            if used + t["mem"] <= mem_gb or not running:
                p = ctx.Process(target=_work, args=(t, tot, q, ev))
                p.start()
                running[p.pid] = (p, t, t["mem"], key)
                used += t["mem"]
                pending.pop(i)
            else:
                i += 1
        if not pending and not running and idle_check:
            idle_check = False
            if not job_busy():
                break
        time.sleep(0.5)
    q.put(None)
    wt.join()
    log(f"{time.strftime('%H:%M:%S')} no unit left: {n_done} units finished here")
    return n_done


def pool_status(pool: str, stale: float = POOL_STALE) -> dict:
    index, claims, done_dir = pool_dirs(pool)
    with open(index) as fh:
        n = len(json.load(fh))
    done = sum(1 for x in os.listdir(done_dir) if x.isdigit())
    live, old, jobs = 0, 0, {}
    now = time.time()
    for name in os.listdir(claims):
        if not name.isdigit():
            continue
        path = os.path.join(claims, name)
        try:
            age = now - os.stat(path).st_mtime
            with open(path) as fh:
                o = fh.read().strip()
        except OSError:
            continue
        if age < stale:
            live += 1
            jobs[o.split(".", 1)[0]] = jobs.get(o.split(".", 1)[0], 0) + 1
        else:
            old += 1
    return {"units": n, "done": done, "claimed": live, "stale": old, "free": n - done - live - old, "jobs": jobs}


# --------------------------------------------------------------------------- merge
def merge(paths, out: str) -> int:
    import pandas as pd

    parts = [pd.read_csv(f) for f in sorted(set(paths)) if os.path.getsize(f) > 0]
    df = pd.concat(parts, ignore_index=True)
    for c in ("shots", "errors", "seconds", "value", "d", "rounds"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["shots", "errors", "value", "d", "rounds"])
    df = df[df.shots > 0]
    df["d"] = df.d.astype(int)
    df["rounds"] = df.rounds.astype(int)
    df["value"] = df["value"].astype(float)
    df = df.groupby(KEYS, as_index=False).agg(shots=("shots", "sum"), errors=("errors", "sum"),
                                              seconds=("seconds", "sum"))
    df["p_L"] = df.errors / df.shots
    df["stderr"] = np.sqrt(np.maximum(df.p_L * (1 - df.p_L), 1e-12) / df.shots)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(KEYS)[COLS].to_csv(out, index=False)
    return len(df)


def _expand(patterns) -> list:
    return sorted(set(sum((glob.glob(x) for x in patterns), [])))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("tasks", help="write the task list and print the cost estimate")
    a.add_argument("--out", required=True)
    a.add_argument("--noises", nargs="+", default=NOISES)
    a.add_argument("--decoders", nargs="+", default=DECODERS)
    a.add_argument("--rounds", nargs="+", default=ROUNDS)
    a.add_argument("--distances", nargs="+", type=int, default=DISTANCES)
    a.add_argument("--scale", type=float, default=1.0, help="multiply every time budget")
    a.add_argument("--cfe-scale", type=float, default=1.0, help="multiply the CFE time budgets")
    a.add_argument("--rep-hours", type=float, default=4.0, help="longest task; longer points become replicas")
    a.add_argument("--tn-max-open", type=int, default=TN_MAX_OPEN,
                   help="make cfe_tn tasks where the tensor-network frontier is at most this wide")
    a.add_argument("--exclude", nargs="*", default=[],
                   help="task files whose task ids are left out (e.g. the first stage)")
    a.add_argument("--centers", nargs=2, default=None, metavar=("THRESHOLDS_CSV", "POINTS_CSV"),
                   help="centre every series on the thresholds of an earlier analysis (thresholds.csv and the "
                        "points.csv they were fitted on)")
    a.add_argument("--wide", action="store_true", help="14 points over a factor 4 for every decoder")
    a.add_argument("--dmax", nargs="+", default=None, metavar="DEC=D",
                   help="largest d per decoder, replacing the defaults (e.g. tesseract=21)")
    a.add_argument("--tn-dmax", nargs="+", default=None, metavar="R=D",
                   help="largest d of the cfe_tn tasks for each number of rounds, e.g. 1=21 2=5 "
                        "(rounds not listed get none); default: every case within --tn-max-open")
    b = sub.add_parser("run", help="run one chunk of the task list")
    b.add_argument("tasks")
    b.add_argument("out")
    b.add_argument("--chunk", type=int, default=0)
    b.add_argument("--nchunks", type=int, default=1)
    b.add_argument("--workers", type=int, default=0, help="default: number of physical cores")
    b.add_argument("--mem-gb", type=float, default=0.0, help="default: 85%% of the node memory")
    b.add_argument("--limit", type=int, default=0, help="run only the first N tasks of the chunk (tests)")
    b.add_argument("--max-seconds", type=float, default=0.0, help="cap every task's time budget (tests)")
    c = sub.add_parser("merge", help="sum all tasks of every point into one CSV")
    c.add_argument("files", nargs="+")
    c.add_argument("--out", required=True)
    s = sub.add_parser("status", help="progress of the campaign")
    s.add_argument("tasks")
    s.add_argument("files", nargs="+")
    po = sub.add_parser("pool", help="write the unfinished tasks of task files to a new work pool")
    po.add_argument("out", help="the pool, e.g. data/campaign/pool1.jsonl (a new name for every pool)")
    po.add_argument("--tasks", nargs="+", required=True, help="task files")
    po.add_argument("--points", nargs="+", default=["data/campaign/points_*.csv"], help="counts so far")
    po.add_argument("--unit-size", type=int, default=128)
    po.add_argument("--first", nargs="*", default=[], metavar="ROUNDS",
                    help="numbers of rounds whose tasks go first (e.g. d: every r = d threshold before the rest)")
    pr = sub.add_parser("pool-run", help="work through a pool on this node (one per node; run_pool.sh)")
    pr.add_argument("pool")
    pr.add_argument("--workers", type=int, default=0, help="default: number of physical cores")
    pr.add_argument("--mem-gb", type=float, default=0.0, help="default: 85%% of the node memory")
    pr.add_argument("--max-seconds", type=float, default=0.0, help="cap every task's time budget (tests)")
    pr.add_argument("--order", default=None,
                    help="file listing pools to work on before this one, read again every 10 minutes "
                         "(default: pool_order.txt next to the pool; an empty string turns it off)")
    ps = sub.add_parser("pool-status", help="units of a pool finished, claimed (by job) and free")
    ps.add_argument("pool")
    args = ap.parse_args()

    if args.cmd == "tasks":
        excl = set()
        for f in args.exclude:
            excl |= {json.loads(line)["id"] for line in open(f) if line.strip()}
        tasks = make_tasks(args.noises, args.decoders, args.rounds, args.distances,
                           args.scale, args.cfe_scale, args.rep_hours, args.tn_max_open, excl,
                           None if args.tn_dmax is None else {k: int(v) for k, v in (x.split("=") for x in args.tn_dmax)},
                           None if args.centers is None else measured_centers(*args.centers), args.wide,
                           None if args.dmax is None else
                           {**DMAX, **{k: int(v) for k, v in (x.split("=") for x in args.dmax)}})
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            for t in tasks:
                fh.write(json.dumps(t) + "\n")
        est = estimate(tasks)
        npts = len({t["id"].rsplit("|", 1)[0] for t in tasks})
        print(f"{len(tasks)} tasks ({npts} points) -> {args.out}")
        print("upper bound of the cost (core-hours): "
              + ", ".join(f"{k} {v:,.0f}" for k, v in sorted(est.items(), key=lambda kv: -kv[1]))
              + f"; total {sum(est.values()):,.0f} = {sum(est.values()) / 128:,.0f} node-hours at 128 cores")
        print(f"longest task {max(t['budget'] for t in tasks) / 3600:.1f} h")
    elif args.cmd == "run":
        tasks = [json.loads(line) for line in open(args.tasks) if line.strip()]
        mine = tasks[args.chunk::args.nchunks]
        del tasks                                   # only this chunk stays in memory
        if args.limit:
            mine = mine[:args.limit]
        if args.max_seconds:
            for t in mine:
                t["budget"] = min(t["budget"], args.max_seconds)
        workers = args.workers or max(1, (os.cpu_count() or 2) // 2)
        mem = args.mem_gb or 0.85 * node_memory_gb()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        run(mine, args.out, workers, mem, log=lambda m: print(m, flush=True))
    elif args.cmd == "merge":
        n = merge(_expand(args.files), args.out)
        print(f"{n} points -> {args.out}")
    elif args.cmd == "pool":
        n, units = build_pool(_expand(args.tasks), _expand(args.points), args.out, args.unit_size, first=args.first)
        print(f"{n} unfinished tasks in {units} units -> {args.out}")
    elif args.cmd == "pool-run":
        workers = args.workers or max(1, (os.cpu_count() or 2) // 2)
        mem = args.mem_gb or 0.85 * node_memory_gb()
        order = args.order if args.order is not None else os.path.join(os.path.dirname(args.pool), "pool_order.txt")
        pool_run(args.pool, workers, mem, log=lambda m: print(m, flush=True), max_seconds=args.max_seconds,
                 order_file=order or None)
    elif args.cmd == "pool-status":
        st = pool_status(args.pool)
        print(" ".join(f"{k} {v}" for k, v in st.items() if k != "jobs")
              + "; claimed by job: " + (", ".join(f"{j} {n}" for j, n in sorted(st["jobs"].items())) or "none"))
    else:
        tasks = [json.loads(line) for line in open(args.tasks) if line.strip()]
        prog = progress(_expand(args.files))
        tot = {}
        for t in tasks:
            s = prog.get(t["id"])
            k = t["decoder"]
            a_ = tot.setdefault(k, [0, 0, 0, 0, 0.0, 0.0])
            a_[0] += 1
            if finished(t, s):
                a_[1] += 1
            elif s:
                a_[2] += 1
            if s and s[3].startswith("failed"):
                a_[3] += 1
            a_[4] += (s[2] if s else 0.0) / 3600
            a_[5] += t["budget"] / 3600
        print(f"{'decoder':12s} {'tasks':>7s} {'done':>7s} {'partial':>7s} {'failed':>6s} {'core-h used':>12s} {'of at most':>11s}")
        for k, v in tot.items():
            print(f"{k:12s} {v[0]:7d} {v[1]:7d} {v[2]:7d} {v[3]:6d} {v[4]:12.1f} {v[5]:11.1f}")
        fails = sorted({prog[t["id"]][3] for t in tasks
                        if t["id"] in prog and prog[t["id"]][3].startswith("failed")})
        for f in fails[:10]:
            print("  ", f)


if __name__ == "__main__":
    main()
