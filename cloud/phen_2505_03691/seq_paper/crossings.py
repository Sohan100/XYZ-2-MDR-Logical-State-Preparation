"""
crossings.py
----------------------------------------------------------------------------
Pairwise crossings p*(d1, d2) of the logical error curves in points CSVs of
run_points.py, with parametric-bootstrap error bars.

For every series (code, variant, noise, basis, decoder) and every pair of
consecutive distances, logit(p_L) of each distance is fitted linearly in p
(weighted by the binomial variance) on the window of up to `--window` (6) points
around the first sign change of p_L(d2) - p_L(d1); the crossing is the root
of the difference of the two lines. The error bar is the standard deviation
(half the 16-84 % range) of the crossing over `--boot` binomial resamplings of every point. A pair
without a sign change in the scanned range is reported as a bound.

    python cloud/phen_2505_03691/seq_paper/crossings.py cloud/phen_2505_03691/seq_paper/data/points.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np

KEY = ("code", "variant", "noise", "basis", "decoder")


def load(paths):
    series = defaultdict(lambda: defaultdict(dict))
    for path in paths:
        with open(path) as fh:
            for r in csv.DictReader(fh):
                k = tuple(r[x] for x in KEY)
                d, p = int(r["d"]), float(r["p"])
                n, e = int(r["shots"]), int(r["errors"])
                old = series[k][d].get(p)
                if old:                       # merge repeated points
                    n, e = n + old[0], e + old[1]
                series[k][d][p] = (n, e)
    return series


def _fit(ps, n, e):
    f = np.clip((e + 0.5) / (n + 1.0), 1e-6, 1 - 1e-6)
    y = np.log(f / (1 - f))
    w = n * f * (1 - f)                     # 1 / var(logit)
    A = np.vstack([np.ones_like(ps), ps]).T
    W = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(A * W[:, None], y * W, rcond=None)
    return coef


def crossing(pts1, pts2, window=4, boot=400, rng=None):
    ps = np.array(sorted(set(pts1) & set(pts2)))
    if len(ps) < 2:
        return None
    r1 = np.array([pts1[p][1] / pts1[p][0] for p in ps])
    r2 = np.array([pts2[p][1] / pts2[p][0] for p in ps])
    diff = r2 - r1
    sc = np.flatnonzero((diff[:-1] < 0) & (diff[1:] >= 0))
    if len(sc) == 0:
        if np.all(diff < 0):
            return ("bound", ">", ps[-1])
        if np.all(diff >= 0):
            return ("bound", "<", ps[0])
        sc = np.flatnonzero(np.sign(diff[:-1]) != np.sign(diff[1:]))
    i = sc[0]
    lo = max(0, i + 1 - window // 2)
    hi = min(len(ps), lo + window)
    lo = max(0, hi - window)
    sel = ps[lo:hi]
    n1 = np.array([pts1[p][0] for p in sel]); e1 = np.array([pts1[p][1] for p in sel])
    n2 = np.array([pts2[p][0] for p in sel]); e2 = np.array([pts2[p][1] for p in sel])

    def root(e1_, e2_):
        a1, b1 = _fit(sel, n1, e1_)
        a2, b2 = _fit(sel, n2, e2_)
        if b2 == b1:
            return np.nan
        return (a1 - a2) / (b2 - b1)

    x0 = root(e1, e2)
    rng = rng or np.random.default_rng(0)
    xs = [root(rng.binomial(n1, e1 / n1), rng.binomial(n2, e2 / n2)) for _ in range(boot)]
    xs = np.array([x for x in xs if np.isfinite(x)])
    lo16, hi84 = np.percentile(xs, [16, 84])          # robust: nearly parallel lines give outliers
    return ("cross", x0, float((hi84 - lo16) / 2), (sel[0], sel[-1]))


def fmt(c):
    if c is None:
        return "n/a"
    if c[0] == "bound":
        return f"{c[1]} {100 * c[2]:.2f}"
    return f"{100 * c[1]:.2f} ± {100 * c[2]:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--boot", type=int, default=400)
    a = ap.parse_args()
    series = load(a.csv)
    print("| code | variant | noise | basis | decoder | p*(3,5) % | p*(5,7) % |")
    print("|---|---|---|---|---|---|---|")
    for k in sorted(series):
        s = series[k]
        cells = []
        for d1, d2 in ((3, 5), (5, 7)):
            if d1 in s and d2 in s:
                cells.append(fmt(crossing(s[d1], s[d2], a.window, a.boot)))
            else:
                cells.append("n/a")
        print("| " + " | ".join(k) + " | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
