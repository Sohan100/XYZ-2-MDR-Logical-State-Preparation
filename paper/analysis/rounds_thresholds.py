"""Threshold versus number of rounds from the pooled sweep data.

usage: python paper/analysis/rounds_thresholds.py [SINTER.csv ...]

Reads docs/data/rounds_threshold/sweeps_pooled.csv, pools it with any sinter
CSV files written by scripts/run_rounds_threshold_sweep.py, and writes the
fits to docs/data/rounds_threshold/thresholds.csv.

For every number of rounds r we fit the finite-size scaling form
    p_L = A + B x + C x^2,   x = (p - p_th) d^(1/nu),
to the distances d >= 11 (up to 21) on a window of +-15% around the crossing.
The statistical error comes from a parametric bootstrap, and the systematic
error is half the spread of the estimates obtained with two window widths (15%
and 25%) and three sets of distances (d >= 9, 11 and 13). The drift of the
crossing with distance is measured by fits to triples (d, d+2, d+4). For
r = d the fits are repeated with the logical error rate per round (key
"d_round"), which removes the extra rounds of the larger distance.
"""
import sys

import numpy as np
import pandas as pd
import sinter
from scipy.optimize import least_squares

from common import DATA, read

ROUNDS_DIR = DATA / "rounds_threshold"


def load(paths=()):
    """Pooled sweep data, one row per (d, r, p)."""
    rows = []
    for s in (sinter.read_stats_from_csv_files(*paths) if paths else []):
        m = s.json_metadata
        rows.append(dict(d=m["d"], r=m["r"], rmode=m["rmode"], p=round(m["p"], 6), shots=s.shots,
                         errors=s.errors, seconds=s.seconds))
    parts = [read(ROUNDS_DIR / "sweeps_pooled.csv"), pd.DataFrame(rows)]
    df = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    df["p"] = df.p.round(6)
    df = df.groupby(["d", "r", "rmode", "p"], as_index=False).agg(
        shots=("shots", "sum"), errors=("errors", "sum"), seconds=("seconds", "sum"))
    df["pL"] = df.errors / df.shots
    df["key"] = np.where(df.rmode == "d", "d", df.r.astype(str))
    return df

RNG = np.random.default_rng(2026)
KEYS = ["1", "2", "3", "4", "5", "6", "8", "10", "d"]
CENTER0 = {"1": 0.018, "2": 0.0115, "3": 0.0097, "4": 0.0076, "5": 0.0069, "6": 0.0066, "8": 0.0058,
           "10": 0.0055, "d": 0.0044}


def fss(g, dset, center, half, boot=0, plo=0.005, phi=0.45, per_round=False):
    """Fit the scaling form near the crossing.

    With per_round=True the fit uses the logical error rate per round,
    eps = [1 - (1 - 2 p_L)^(1/(r+1))] / 2, with propagated binomial errors.
    """
    s = g[g.d.isin(dset) & (np.abs(g.p / center - 1) <= half) & (g.pL > plo) & (g.pL < phi)]
    if len(s) < 7 or s.d.nunique() < 2:
        return None
    d = s.d.values.astype(float)
    p = s.p.values
    n = s.shots.values
    pl = s.pL.values
    k = 1.0 / (s.r.values + 1.0)

    def transform(plv):
        if not per_round:
            return plv
        return 0.5 * (1 - np.clip(1 - 2 * plv, 1e-12, None) ** k)

    y0 = transform(pl)
    if per_round:
        base = np.clip(1 - 2 * pl, 1e-12, None)
        sig = np.sqrt(np.clip(pl * (1 - pl), 1e-12, None) / n) * k * base ** (k - 1)
    else:
        sig = None

    def resid(th, y):
        pth, nu, A, B, C = th
        x = (p - pth) * d ** (1 / nu)
        m = A + B * x + C * x * x
        sg = sig if sig is not None else np.sqrt(np.clip(m * (1 - m), 1e-6, None) / n)
        return (m - y) / sg

    bounds = ([center * (1 - half), 0.3, 0, -1e7, -1e10], [center * (1 + half), 6.0, 1, 1e7, 1e10])
    best = None
    for nu0 in (0.8, 1.2, 1.8, 2.5):
        f = least_squares(resid, [center, nu0, float(np.median(y0)), 20.0 if not per_round else 1.0, 0.0],
                          args=(y0,), bounds=bounds)
        if best is None or f.cost < best.cost:
            best = f
    out = dict(pth=float(best.x[0]), nu=float(best.x[1]), chi2=float(2 * best.cost / max(len(y0) - 5, 1)),
               npts=len(y0), stat=np.nan, nu_err=np.nan)
    if boot:
        bs = []
        for _ in range(boot):
            yb = transform(RNG.binomial(n, np.clip(pl, 0, 1)) / n)
            fb = least_squares(resid, best.x, args=(yb,), bounds=bounds)
            bs.append(fb.x[:2])
        bs = np.array(bs)
        out["stat"] = float(np.std(bs[:, 0]))
        out["nu_err"] = float(np.std(bs[:, 1]))
    return out


def analyse(df):
    rows = []
    jobs = [(k, k, False) for k in KEYS] + [("d_round", "d", True)]
    for name, key, per_round in jobs:
        g = df[df.key == key]
        if g.empty:
            continue
        c0 = 0.0050 if per_round else CENTER0[key]
        dmax = int(g.d.max())
        big = list(range(11, dmax + 1, 2))
        wide = fss(g, big, c0, 0.35, per_round=per_round)
        c = wide["pth"] if wide else c0
        main = fss(g, big, c, 0.15, boot=300, per_round=per_round)
        variants = []
        for half in (0.15, 0.25):
            for ds in (list(range(9, dmax + 1, 2)), big, list(range(13, dmax + 1, 2))):
                f = fss(g, ds, c, half, per_round=per_round)
                if f:
                    variants.append(f["pth"])
        syst = 0.5 * (max(variants) - min(variants))
        err = float(np.hypot(main["stat"], syst))
        rows.append(dict(key=name, kind="fss", d1=11, d2=dmax, p=main["pth"], err=err, stat=main["stat"], syst=syst,
                         nu=main["nu"], nu_err=main["nu_err"], chi2=main["chi2"], npts=main["npts"]))
        print(f"r={name}: pth={100*main['pth']:.3f}% stat={100*main['stat']:.3f} syst={100*syst:.3f} "
              f"nu={main['nu']:.2f}({main['nu_err']:.2f}) chi2={main['chi2']:.2f} n={main['npts']}", flush=True)
        for d1 in range(3, dmax - 3, 2):
            f = fss(g, [d1, d1 + 2, d1 + 4], c, 0.3, boot=60, per_round=per_round)
            if f:
                rows.append(dict(key=name, kind="triple", d1=d1, d2=d1 + 4, p=f["pth"], err=f["stat"], nu=f["nu"],
                                 chi2=f["chi2"], npts=f["npts"]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = analyse(load(sys.argv[1:]))
    out.to_csv(ROUNDS_DIR / "thresholds.csv", index=False)
    print(out[out.kind == "triple"].pivot(index="d1", columns="key", values="p").round(5).to_string())
