"""
crossings.py
----------------------------------------------------------------------------
Pairwise crossing points of p_L(p) curves of consecutive distances from
ler_runner.py output (log p_L linear in log p between grid points), with a
bootstrap error from the binomial error bars. Small-d crossings are only a
rough indication of the threshold (the campaign rules ask for d >= 11 fits).

    python cloud/schedule_search/crossings.py data/ler_*.csv
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd


def crossing(ps, a, b):
    """First p where log b - log a changes sign (b: larger distance)."""
    f = np.log(np.maximum(b, 1e-9)) - np.log(np.maximum(a, 1e-9))
    lp = np.log(ps)
    for k in range(len(ps) - 1):
        if f[k] < 0 <= f[k + 1]:
            t = -f[k] / (f[k + 1] - f[k])
            return float(np.exp(lp[k] + t * (lp[k + 1] - lp[k])))
    if np.all(f >= 0):
        return -1.0   # above threshold over the whole grid (crossing below)
    if np.all(f < 0):
        return np.inf  # below threshold over the whole grid (crossing above)
    return np.nan


def table(df: pd.DataFrame, nboot: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for key, g in df.groupby(["code", "comp", "noise", "logical", "decoder"]):
        ds = sorted(g.d.unique())
        for d1, d2 in zip(ds, ds[1:]):
            a = g[g.d == d1].set_index("p")
            b = g[g.d == d2].set_index("p")
            ps = sorted(set(a.index) & set(b.index))
            if len(ps) < 2:
                continue
            a, b = a.loc[ps], b.loc[ps]
            x = crossing(np.array(ps), a.p_L.values, b.p_L.values)
            boots = []
            for _ in range(nboot):
                ra = rng.binomial(a.shots.values.astype(int), a.p_L.values) / a.shots.values
                rb = rng.binomial(b.shots.values.astype(int), b.p_L.values) / b.shots.values
                boots.append(crossing(np.array(ps), ra, rb))
            boots = np.array([v for v in boots if np.isfinite(v) and v > 0])
            err = float(np.std(boots)) if len(boots) > 10 else np.nan
            out.append(dict(zip(["code", "comp", "noise", "logical", "decoder"], key),
                            d_pair=f"{d1}/{d2}", p_cross=x, err=err))
    return pd.DataFrame(out)


if __name__ == "__main__":
    df = pd.concat([pd.read_csv(f) for f in sys.argv[1:]])
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(table(df).to_string(index=False))
