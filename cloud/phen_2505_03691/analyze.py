"""
analyze.py
----------------------------------------------------------------------------
Crossings of the sweeps of sweep.py, with parametric-bootstrap error bars.

For every series (code, noise, basis, decoder) and every pair of consecutive
distances, log p_L is fitted linearly in log p for each distance (weighted
least squares over at most four common p values around the sign change of
p_L(d2) - p_L(d1)) and the crossing of the two fits is reported. Error bars:
half the 16-84 % range of 400 parametric-bootstrap replicas (errors ~
Binomial(shots, p_L)); wider than 1 % is reported as unresolved, and no sign
change as a bound. A finite-size-scaling fit
p_L = A + B x + C x^2, x = (p - p_th) d^(1/nu), over all distances of the
series, using the points within 20 % of the median pairwise crossing, is
reported too (bootstrap error).

    python cloud/phen_2505_03691/analyze.py cloud/phen_2505_03691/data/main.csv [...] > table.md
"""

from __future__ import annotations

import collections
import csv
import sys

import numpy as np
from scipy.optimize import curve_fit

RNG = np.random.default_rng(1)
NBOOT = 400


def load(paths):
    series = collections.defaultdict(lambda: collections.defaultdict(list))
    for path in paths:
        with open(path) as fh:
            for r in csv.DictReader(fh):
                key = (r["code"], r["noise"], r["basis"], r["decoder"])
                series[key][int(r["d"])].append((float(r["p"]), int(r["shots"]), int(r["errors"])))
    return series


def fit_line(p, k, n):
    rate = (k + 0.5) / (n + 1.0)
    y = np.log(rate)
    var = (1 - rate) / (rate * (n + 1.0))
    w = 1 / np.maximum(var, 1e-12)
    x = np.log(p)
    A = np.vstack([np.ones_like(x), x]).T
    W = np.diag(w)
    return np.linalg.solve(A.T @ W @ A, A.T @ W @ y)


def crossing(pa, ka, na, pb, kb, nb):
    a0, a1 = fit_line(pa, ka, na)
    b0, b1 = fit_line(pb, kb, nb)
    if abs(a1 - b1) < 1e-9:
        return np.nan
    return float(np.exp((b0 - a0) / (a1 - b1)))


def pair_crossing(da, db):
    """Crossing of two distances from the points nearest to the sign change of
    p_L(d_b) - p_L(d_a) (two on each side at most). Returns (x, err, note)."""
    A = {p: (n, k) for p, n, k in da}
    B = {p: (n, k) for p, n, k in db}
    ps = sorted(set(A) & set(B))
    if len(ps) < 2:
        return np.nan, np.nan, "–"
    diff = np.array([B[p][1] / B[p][0] - A[p][1] / A[p][0] for p in ps])
    ch = [i for i in range(len(ps) - 1) if diff[i] < 0 <= diff[i + 1]]
    if not ch:
        ch = [i for i in range(len(ps) - 1) if np.sign(diff[i]) != np.sign(diff[i + 1])]
    if not ch:
        side = "below" if diff[0] > 0 else "above"
        bound = ps[0] if side == "below" else ps[-1]
        return np.nan, np.nan, f"{'<' if side == 'below' else '>'} {100 * bound:.2f}"
    i = ch[0]
    sel = ps[max(0, i - 1): i + 3]
    pa = np.array(sel); na = np.array([A[p][0] for p in sel], float); ka = np.array([A[p][1] for p in sel], float)
    nb = np.array([B[p][0] for p in sel], float); kb = np.array([B[p][1] for p in sel], float)
    x0 = crossing(pa, ka, na, pa, kb, nb)
    boots = []
    for _ in range(NBOOT):
        ka_ = RNG.binomial(na.astype(int), np.clip(ka / na, 0, 1))
        kb_ = RNG.binomial(nb.astype(int), np.clip(kb / nb, 0, 1))
        boots.append(crossing(pa, ka_, na, pa, kb_, nb))
    boots = np.array(boots)
    boots = boots[np.isfinite(boots)]
    err = (np.percentile(boots, 84) - np.percentile(boots, 16)) / 2 if len(boots) else np.nan
    note = ""
    if len(set(np.sign(diff))) > 1 and len(ch) > 1:
        note = "multiple sign changes"
    return x0, err, note


def fss_model(X, pth, nu, A, B, C):
    p, d = X
    x = (p - pth) * d ** (1 / nu)
    return A + B * x + C * x ** 2


def fss(data, p0):
    P, D, R, S = [], [], [], []
    for d, pts in data.items():
        for p, n, k in pts:
            r = (k + 0.5) / (n + 1)
            P.append(p); D.append(d); R.append(r); S.append(np.sqrt(r * (1 - r) / n))
    P, D, R, S = map(np.array, (P, D, R, S))
    N = np.array([n for d, pts in data.items() for p, n, k in pts], float)

    def one(R):
        try:
            popt, _ = curve_fit(fss_model, (P, D), R, p0=[p0, 1.0, np.median(R), 1.0, 0.0],
                                sigma=np.maximum(S, 1e-6), maxfev=20000)
            return popt[0], popt[1]
        except Exception:
            return np.nan, np.nan
    pth, nu = one(R)
    boots = [one(RNG.binomial(N.astype(int), R) / N)[0] for _ in range(100)]
    boots = np.array([b for b in boots if np.isfinite(b)])
    return pth, (np.std(boots) if len(boots) else np.nan), nu


def fmt(x, e, note):
    if not np.isfinite(x):
        return note
    if not np.isfinite(e) or e > 0.01:
        return f"~{100 * x:.1f} (unresolved)"
    s = f"{100 * x:.2f} ± {100 * e:.2f}"
    return s + (f" ({note})" if note else "")


def main(paths):
    series = load(paths)
    print("| code | noise | basis | decoder | " + " | ".join(
        ["3/5", "5/7", "7/9", "9/11"]) + " | FSS fit (all d) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for key in sorted(series):
        data = series[key]
        ds = sorted(data)
        cells = []
        xs = []
        for a, b in (("3", "5"), ("5", "7"), ("7", "9"), ("9", "11")):
            a, b = int(a), int(b)
            if a in data and b in data:
                x, e, note = pair_crossing(data[a], data[b])
                cells.append(fmt(x, e, note))
                if np.isfinite(x):
                    xs.append(x)
            else:
                cells.append("–")
        p0 = float(np.median(xs)) if xs else np.mean([p for p, _, _ in data[ds[0]]])
        near = {d: [t for t in pts if abs(t[0] - p0) <= 0.2 * p0] for d, pts in data.items()}
        pth, e, nu = fss({d: v for d, v in near.items() if len(v) >= 2}, p0)
        f = f"{100 * pth:.2f} ± {100 * e:.2f} (ν={nu:.2f})" if np.isfinite(pth) else "fail"
        print(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | " + " | ".join(cells) + f" | {f} |")
    print()
    print("Raw logical error rates p_L (errors/shots) per distance:\n")
    for key in sorted(series):
        data = series[key]
        ps = sorted({p for d in data for p, _, _ in data[d]})
        print(f"**{' / '.join(key)}**\n")
        print("| p (%) | " + " | ".join(f"d={d}" for d in sorted(data)) + " |")
        print("|---|" + "---|" * len(data))
        for p in ps:
            row = []
            for d in sorted(data):
                m = [(n, k) for pp, n, k in data[d] if pp == p]
                row.append(f"{m[0][1] / m[0][0]:.4f} ({m[0][1]}/{m[0][0]})" if m else "–")
            print(f"| {100 * p:.2f} | " + " | ".join(row) + " |")
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
