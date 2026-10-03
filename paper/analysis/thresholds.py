"""Finite-size scaling threshold fits for r = d rounds.

Usage: python paper/analysis/thresholds.py [extra_sweep.csv ...]

Reads docs/data/thresholds/sweeps_pooled.csv (and any extra sweep files),
fits every noise model and decoder and writes docs/data/thresholds/fits.csv.
"""
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from common import NOISE_ORDER, THR, load_sweeps


def load(extra=()):
    df = load_sweeps(extra)
    df = df[df.rounds == df.d]
    keys = ["noise", "value", "d", "rounds", "decoder"]
    df = df.groupby(keys, as_index=False).agg(shots=("shots", "sum"), errors=("errors", "sum"))
    df["p_L"] = df.errors / df.shots
    return df


def crude_crossing(sub, ds):
    """Average crossing point of consecutive distances (log-linear interpolation)."""
    xs = []
    for d1, d2 in zip(ds, ds[1:]):
        a = sub[sub.d == d1].set_index("value").p_L
        b = sub[sub.d == d2].set_index("value").p_L
        v = sorted(set(a.index) & set(b.index))
        diff = [np.log(b[x]) - np.log(a[x]) for x in v]
        for i in range(len(v) - 1):
            if diff[i] <= 0 < diff[i + 1] or (diff[i] < 0 <= diff[i + 1]):
                t = -diff[i] / (diff[i + 1] - diff[i])
                xs.append(v[i] + t * (v[i + 1] - v[i]))
                break
    return float(np.mean(xs)) if xs else np.nan, xs


def pair_crossing(sub, d1, d2, nboot=300, seed=2):
    """Crossing of the d1 and d2 curves by linear interpolation of log p_L, with bootstrap error."""
    a = sub[sub.d == d1].set_index("value")
    b = sub[sub.d == d2].set_index("value")
    v = sorted(set(a.index) & set(b.index))
    if len(v) < 2:
        return np.nan, np.nan
    pa = np.array([a.p_L[x] for x in v]); na = np.array([a.shots[x] for x in v])
    pb = np.array([b.p_L[x] for x in v]); nb = np.array([b.shots[x] for x in v])

    def cross(qa, qb):
        diff = np.log(np.maximum(qb, 1e-9)) - np.log(np.maximum(qa, 1e-9))
        for i in range(len(v) - 1):
            if diff[i] <= 0 < diff[i + 1] or (diff[i] < 0 <= diff[i + 1]):
                t = -diff[i] / (diff[i + 1] - diff[i])
                return v[i] + t * (v[i + 1] - v[i])
        return np.nan

    c0 = cross(pa, pb)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(nboot):
        qa = rng.binomial(na.astype(int), pa) / na
        qb = rng.binomial(nb.astype(int), pb) / nb
        bs.append(cross(qa, qb))
    bs = np.array([x for x in bs if np.isfinite(x)])
    return c0, (float(np.std(bs)) if len(bs) else np.nan)


def fss_fit(sub, ds, p0, window=0.35, nboot=40, seed=1):
    s = sub[sub.d.isin(ds) & (abs(sub.value - p0) <= window * p0)].copy()
    s = s[s.errors >= 20]
    p = s.value.values
    d = s.d.values.astype(float)
    y = s.p_L.values
    sig = np.sqrt(np.maximum(y * (1 - y), 1e-12) / s.shots.values)

    def resid(th, yy):
        pth, nu, A, B, C = th
        x = (p - pth) * d ** (1 / nu)
        return (A + B * x + C * x ** 2 - yy) / sig

    th0 = [p0, 1.3, np.median(y), np.median(y) / (0.1 * p0), 0.0]
    r = least_squares(resid, th0, args=(y,), x_scale="jac", max_nfev=20000)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(nboot):
        yb = y + rng.normal(0, sig)
        rb = least_squares(resid, r.x, args=(yb,), x_scale="jac", max_nfev=20000)
        boots.append(rb.x[0])
    chi2 = float(np.sum(r.fun ** 2)) / max(1, len(y) - 5)
    return r.x[0], float(np.std(boots)), r.x[1], chi2, len(y)


if __name__ == "__main__":
    df = load(sys.argv[1:])
    rows = []
    for noise in NOISE_ORDER:
        for dec in ["mwpm", "corr_links", "corr_gauge"]:
            sub = df[(df.noise == noise) & (df.decoder == dec)]
            ds = sorted(sub.d.unique())
            if len(ds) < 3:
                continue
            c, xs = crude_crossing(sub, ds)
            if not np.isfinite(c):
                print(noise, dec, "no crossing yet")
                continue
            big = [d for d in ds if d >= 5] if len(ds) >= 4 else ds
            pth, err, nu, chi2, n = fss_fit(sub, big, c)
            pth_all, err_all, *_ = fss_fit(sub, ds, c)
            last_pairs = xs[-2:] if len(xs) >= 2 else xs
            spread = [pth, pth_all] + list(last_pairs)
            unc = max(err, 0.5 * (max(spread) - min(spread)))
            if len(ds) == 3:
                # only d = 3, 5, 7: quote the crossing of the two largest distances
                c57, e57 = pair_crossing(sub, ds[1], ds[2])
                c35, e35 = pair_crossing(sub, ds[0], ds[1])
                if np.isfinite(c57):
                    pth, err = c57, e57
                    unc = max(e57, 0.5 * abs(c57 - c35)) if np.isfinite(c35) else e57
                    big = ds[1:]
            rows.append((noise, dec, c, pth, err, unc, nu, chi2, n, pth_all, ",".join(str(d) for d in big)))
            print(f"{noise:10s} {dec:11s} crude={c:.4g} [{', '.join(f'{x:.3g}' for x in xs)}]  "
                  f"fss({big})={pth:.4g}+-{err:.2g} nu={nu:.2f} chi2r={chi2:.2f} n={n}  fss(all)={pth_all:.4g}  unc={unc:.2g}")
    # belief-matching: crossing of d = 5 and d = 7 from bm_thresholds.py
    bm = THR / "bm_crossings.csv"
    if bm.exists():
        for _, b in pd.read_csv(bm).iterrows():
            rows.append((b.noise, "bm", b.p57, b.p57, b.err57, b.err57, np.nan, np.nan, np.nan, b.p35, "5,7"))
    out = pd.DataFrame(rows, columns=["noise", "decoder", "crude", "pth", "err", "unc", "nu", "chi2r", "n", "pth_all", "dists"])
    out.to_csv(THR / "fits.csv", index=False)
    print("wrote", THR / "fits.csv")
