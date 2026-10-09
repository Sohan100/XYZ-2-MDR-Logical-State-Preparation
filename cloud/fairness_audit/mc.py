"""Monte-Carlo sweeps of the audit (tasks 2a and 2b), all cores, results appended to a jsonl."""
import argparse
import json
import multiprocessing as mp
import os
import zlib

from common import run_task


def tasks_2a():
    for kind in ("css", "stim"):
        for d in (3, 5, 7):
            for p in (0.004, 0.006, 0.008):
                for basis in ("X", "Z"):
                    yield dict(study="2a", kind=kind, d=d, rounds=d, p=p, basis=basis,
                               decoder="mwpm", max_errors=4000, max_shots=4_000_000,
                               max_seconds=300)


P2B = (0.006, 0.0065, 0.007, 0.0075, 0.008, 0.0085, 0.009)


def tasks_2b():
    for d in (5, 7, 9, 11):
        for p in P2B:
            for basis in ("X", "Z"):
                for kind, dec in (("css", "mwpm"), ("xzzx", "mwpm"), ("stim", "mwpm"),
                                  ("css", "corr"), ("xzzx", "corr")):
                    yield dict(study="2b", kind=kind, d=d, rounds=d, p=p, basis=basis,
                               decoder=dec, max_errors=3000, max_shots=2_000_000,
                               max_seconds=240, batch=1000)


def tasks_abl():
    for d in (5, 7, 9):
        for p in P2B:
            for kind in ("css_mr", "css_mr_noidle"):
                yield dict(study="abl", kind=kind, d=d, rounds=d, p=p, basis="X",
                           decoder="mwpm", max_errors=3000, max_shots=2_000_000,
                           max_seconds=240, batch=1000)


def work(t):
    r = run_task(t)
    r["study"] = t["study"]
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("study", choices=["2a", "2b", "abl"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    out = a.out or f"results_{a.study}.jsonl"
    done = set()
    if os.path.exists(out):
        for line in open(out):
            r = json.loads(line)
            done.add((r["kind"], r["d"], r["p"], r["basis"], r["decoder"]))
    ts = [t for t in ({"2a": tasks_2a, "2b": tasks_2b, "abl": tasks_abl}[a.study]())
          if (t["kind"], t["d"], t["p"], t["basis"], t["decoder"]) not in done]
    for i, t in enumerate(ts):
        t["seed"] = zlib.crc32(repr((t["kind"], t["d"], t["p"], t["basis"], t["decoder"])).encode())
    # big distances first so the pool drains evenly
    ts.sort(key=lambda t: -t["d"])
    with mp.Pool(a.procs) as pool, open(out, "a") as f:
        for r in pool.imap_unordered(work, ts):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(r, flush=True)
