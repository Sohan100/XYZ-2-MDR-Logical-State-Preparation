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
docs/nersc_campaign.md.

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
DECODERS = ["mwpm", "corr_links", "corr_gauge", "seq_soft", "seq_match", "bm", "tesseract", "cfe", "cfe_tn"]
ROUNDS = ["1", "2", "3", "4", "5", "6", "8", "10", "d"]
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
# threshold of each decoder relative to matching (measured for r = d; guesses for seq_* and the ion models)
_G = {
    "corr_links": dict(sd6=1.17, si1000=1.19, biased10=1.26, biased100=1.29, purez=1.29, em3=1.2),
    "corr_gauge": dict(sd6=1.18, si1000=1.20, biased10=1.39, biased100=1.45, purez=1.49, em3=1.28),
    "seq_soft": {}, "seq_match": {},
    "bm": dict(sd6=1.28, si1000=1.25, biased10=1.76, biased100=1.89, purez=1.85, em3=1.3),
}
_G_DEFAULT = {"mwpm": 1.0, "corr_links": 1.18, "corr_gauge": 1.2, "seq_soft": 1.05, "seq_match": 0.9, "bm": 1.3}
_G["tesseract"] = _G["cfe"] = _G["cfe0"] = _G["cfe_tn"] = _G["bm"]
_G_DEFAULT["tesseract"] = _G_DEFAULT["cfe"] = _G_DEFAULT["cfe0"] = _G_DEFAULT["cfe_tn"] = 1.3
# The crosstalk models have no threshold: the crossings of consecutive distances drift to small p.
XT_RANGE = {"helios_p": (6e-5, 3e-3), "h2_p": (5e-4, 6e-3)}

STEP = 4.0 ** (1 / 13)
KRANGE = {"wide": range(-7, 7), "narrow": range(-5, 5)}
SLOW = {"seq_soft", "bm", "tesseract", "cfe", "cfe0", "cfe_tn"}

TARGET = {"mwpm": 500, "corr_links": 400, "corr_gauge": 400, "seq_soft": 250, "seq_match": 300,
          "bm": 250, "tesseract": 150, "cfe": 200, "cfe0": 150, "cfe_tn": 200}
MAX_SHOTS = 2_000_000
S21 = 22 * 21 * 21
S9 = 10 * 9 * 9
# budget in seconds at S21 (S9 for Tesseract), exponent, smallest budget
BUDGET = {"mwpm": (600, 1.15, 30), "corr_links": (1200, 1.15, 60), "corr_gauge": (1800, 1.15, 60),
          "seq_match": (1800, 1.1, 60), "seq_soft": (5400, 1.1, 120), "bm": (5400, 1.1, 120),
          "tesseract": (7200, 2.0, 120), "cfe": (60_000, 1.3, 300), "cfe0": (60_000, 1.4, 300),
          "cfe_tn": (150_000, 1.0, 600)}
# memory in GB: base + slope * S / S21 (measured on d = 9 to 21 circuits, with margin); CFE: see memory()
MEMORY = {"mwpm": (0.4, 1.0), "corr_links": (0.4, 1.2), "corr_gauge": (0.4, 1.2), "seq_match": (0.4, 1.2),
          "seq_soft": (0.4, 1.8), "bm": (0.4, 2.0), "tesseract": (0.5, 6.0)}

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
    return P_MWPM[noise] * F_ROUNDS[r] * g


def grid(noise: str, decoder: str, r: str) -> list:
    if noise in XT_RANGE:
        lo, hi = XT_RANGE[noise]
        lo, hi = lo * F_ROUNDS[r] ** 0.5, hi * F_ROUNDS[r]
        n = 14 if decoder not in SLOW else 10
        return [float(f"{x:.4g}") for x in np.geomspace(lo, hi, n)]
    c = center(noise, decoder, r)
    ks = KRANGE["narrow" if decoder in SLOW else "wide"]
    return [float(f"{c * STEP ** k:.4g}") for k in ks]


def budget(decoder: str, d: int, r: str) -> float:
    b21, alpha, bmin = BUDGET[decoder]
    s = (n_rounds(r, d) + 1) * d * d
    ref = S9 if decoder == "tesseract" else S21
    return max(bmin, b21 * (s / ref) ** alpha)


def memory(decoder: str, d: int, r: str) -> float:
    s = (n_rounds(r, d) + 1) * d * d
    if decoder in ("cfe", "cfe0", "cfe_tn"):
        n = 33.0 * s                       # fault mechanisms of the detector error model
        # degeneracy moves and BP (1.2e-5 n) and the bit-packed OSD-CS matrix (m n / 8, m ~ 1.8 S)
        return round(0.8 + 1.2e-5 * n + (2.0 * 1.8 * s * n / 8 / 1e9 if decoder != "cfe0" else 0.0), 2)
    base, slope = MEMORY[decoder]
    ref = S9 if decoder == "tesseract" else S21
    return round(base + slope * s / ref, 2)


def weight(noise: str, decoder: str, r: str, p: float) -> float:
    """Budget factor: points well below the expected threshold get less time."""
    if noise in XT_RANGE:
        return 1.0
    x = p / center(noise, decoder, r)
    return 1.0 if x >= 0.8 else (0.5 if x >= 0.65 else 0.25)


def make_tasks(noises, decoders, rounds, distances, scale=1.0, cfe_scale=1.0, rep_hours=4.0,
               tn_max_open=TN_MAX_OPEN, exclude=()) -> list:
    out = []
    for noise in noises:
        for dec in decoders:
            for r in rounds:
                values = grid(noise, dec, r)
                for d in distances:
                    if d > DMAX.get(dec, 99):
                        continue
                    if dec == "cfe_tn" and (3.5 * n_rounds(r, d) - 1) * d > tn_max_open:
                        continue
                    b = budget(dec, d, r) * scale * (cfe_scale if dec.startswith("cfe") else 1.0)
                    for p in values:
                        bp = max(BUDGET[dec][2], b * weight(noise, dec, r, p))
                        nrep = max(1, math.ceil(bp / (3600.0 * rep_hours)))
                        for i in range(nrep):
                            if f"{noise}|{dec}|r{r}|d{d}|p{p:.4g}|{i}" in exclude:
                                continue
                            out.append(dict(
                                id=f"{noise}|{dec}|r{r}|d{d}|p{p:.4g}|{i}", noise=noise, decoder=dec, rounds=r, d=d,
                                value=p, target=math.ceil(TARGET[dec] / nrep), max_shots=math.ceil(MAX_SHOTS / nrep),
                                budget=round(bp / nrep, 1), mem=memory(dec, d, r)))
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
    """Counts so far per task: {task id: [shots, errors, seconds, note]}."""
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
        if (r.get("note") or "").startswith("failed"):
            s[3] = r["note"]
    return prog


def finished(t: dict, s) -> bool:
    if s is None:
        return False
    return bool(s[3]) or s[1] >= t["target"] or s[0] >= t["max_shots"] or s[2] >= t["budget"]


# --------------------------------------------------------------------------- worker
FLUSH = 300.0       # seconds between partial rows
BATCH = 8.0         # target seconds per sampled batch


def _row(t, shots, errors, seconds, note=""):
    rounds = n_rounds(t["rounds"], t["d"])
    pl = errors / shots if shots else 0.0
    se = math.sqrt(max(pl * (1 - pl), 1e-12) / shots) if shots else 0.0
    return [t["noise"], t["value"], t["d"], rounds, "frame", t["decoder"], shots, errors, pl, se,
            round(seconds, 1), t["id"], note]


def _work(t: dict, prev, q) -> None:
    from mdr.ft import FTMDRCircuit
    from mdr.ft.two_level_decoder import TwoLevelDecoder
    from run_decoder_threshold_sweep import DECODERS as DEC, NOISE

    s_tot, e_tot, sec0 = (prev[0], prev[1], prev[2]) if prev else (0, 0, 0.0)
    t0 = last = time.time()
    try:
        ft = FTMDRCircuit(t["d"], n_rounds(t["rounds"], t["d"]), NOISE[t["noise"]](t["value"]),
                          final="frame", detectors="combined")
        dec = TwoLevelDecoder(ft, **DEC[t["decoder"]])
        sampler = dec.circuit.compile_detector_sampler()
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
            size = int(np.clip(min(4 * n, n * BATCH / dt), 1, 200_000))
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
    log(f"{len(tasks)} tasks in this chunk, {len(todo)} to run, {workers} workers, {mem_gb:.0f} GB")
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    wt = threading.Thread(target=_writer, args=(q, out), daemon=True)
    wt.start()
    running = {}
    used = 0.0
    done = 0
    while todo or running:
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
                p = ctx.Process(target=_work, args=(t, prog.get(t["id"]), q))
                p.start()
                running[p.pid] = (p, t, t["mem"])
                used += t["mem"]
                todo.pop(i)
            else:
                i += 1
        time.sleep(0.5)
    q.put(None)
    wt.join()


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
    args = ap.parse_args()

    if args.cmd == "tasks":
        excl = set()
        for f in args.exclude:
            excl |= {json.loads(line)["id"] for line in open(f) if line.strip()}
        tasks = make_tasks(args.noises, args.decoders, args.rounds, args.distances,
                           args.scale, args.cfe_scale, args.rep_hours, args.tn_max_open, excl)
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
            if s and s[3]:
                a_[3] += 1
            a_[4] += (s[2] if s else 0.0) / 3600
            a_[5] += t["budget"] / 3600
        print(f"{'decoder':12s} {'tasks':>7s} {'done':>7s} {'partial':>7s} {'failed':>6s} {'core-h used':>12s} {'of at most':>11s}")
        for k, v in tot.items():
            print(f"{k:12s} {v[0]:7d} {v[1]:7d} {v[2]:7d} {v[3]:6d} {v[4]:12.1f} {v[5]:11.1f}")
        fails = sorted({s[3] for s in prog.values() if s[3]})
        for f in fails[:10]:
            print("  ", f)


if __name__ == "__main__":
    main()
