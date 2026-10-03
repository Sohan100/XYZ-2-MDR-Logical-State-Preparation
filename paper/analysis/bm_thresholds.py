"""Belief-matching thresholds from the crossing of d = 5 and d = 7, and the appendix figure.

Data: docs/data/thresholds/bm_pooled.csv, the belief-matching runs of
scripts/run_decoder_threshold_sweep.py (first sweep and dense points around the crossings)
pooled with the belief-matching decisions stored with the CFE sweeps (same model,
independent shots). Writes docs/data/thresholds/bm_crossings.csv and fig_bm.pdf. Every distance is fitted by a straight line in ln p_L over a window of
+-15% around the crossing, and the threshold is where the lines of d = 5 and d = 7 meet.
The figure draws exactly these lines, so the vertical line passes through their
intersection.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import make_plots as M  # noqa: E402
import crossings as X  # noqa: E402
from common import THR, read  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

NOISES = [("sd6", "SD6", 6.0e-3), ("si1000", "SI1000", 4.5e-3), ("biased10", r"$\eta=10$", 8.0e-3),
          ("biased100", r"$\eta=100$", 8.6e-3), ("purez", r"Pure $Z$", 8.8e-3)]
REL = 0.15
OUT_CSV = THR / "bm_crossings.csv"


def ratio_guess(g, d1, d2, guess, rel=0.3):
    """Zero of a weighted line through ln(p_L(d2)/p_L(d1)) over a wide window."""
    a = g[g.d == d1].set_index("value")
    b = g[g.d == d2].set_index("value")
    ps = sorted(set(a.index) & set(b.index))
    ps = [p for p in ps if abs(p - guess) <= rel * guess]
    if len(ps) < 3:
        return guess
    qa = a.loc[ps].errors / a.loc[ps].shots
    qb = b.loc[ps].errors / b.loc[ps].shots
    y = np.log(qb.values / qa.values)
    var = (1 - qa.values) / a.loc[ps].errors.values + (1 - qb.values) / b.loc[ps].errors.values
    A = np.vstack([np.ones(len(ps)), np.array(ps) - guess]).T
    w = 1 / var
    c0, c1 = np.linalg.solve(A.T @ (A * w[:, None]), A.T @ (w * y))
    return guess - c0 / c1 if c1 > 0 else guess


def analyse(a):
    rows = []
    for noise, _, guess in NOISES:
        g = a[a.noise == noise]
        out = dict(noise=noise)
        for d1, d2 in ((5, 7), (3, 5)):
            p0 = ratio_guess(g, d1, d2, guess)
            w = (p0 * (1 - REL), p0 * (1 + REL))
            x, e = X.crossing(g, d1, d2, w, deg=1, boot=600)
            out[f"p{d1}{d2}"], out[f"err{d1}{d2}"], out[f"lo{d1}{d2}"], out[f"hi{d1}{d2}"] = x, e, w[0], w[1]
        rows.append(out)
    return pd.DataFrame(rows)


def figure(a, res):
    fig, axes = plt.subplots(2, len(NOISES), figsize=(M.FULL, 3.3), sharex="col",
                             gridspec_kw=dict(height_ratios=[1.35, 1.0]))
    for col, (noise, title, _) in enumerate(NOISES):
        ax, bx = axes[0, col], axes[1, col]
        g = a[a.noise == noise]
        r = res[res.noise == noise].iloc[0]
        pth, err = r.p57, r.err57
        lo, hi = 0.6 * pth, 1.4 * pth
        w = (r.lo57, r.hi57)
        cen = 0.5 * (w[0] + w[1])
        fits = {}
        for d in (3, 5, 7):
            s = g[(g.d == d) & (g.value >= lo) & (g.value <= hi) & (g.errors > 0)].sort_values("value")
            y = s.errors / s.shots
            e = np.sqrt(y * (1 - y) / s.shots)
            ax.errorbar(s.value, y, yerr=e, ls="none", marker=M.MARK[d], ms=2.6, color=M.RAMP[d],
                        mec=M.RAMP[d], mfc=M.RAMP[d], elinewidth=0.6, capsize=0, label=f"$d={d}$")
            f = X.curve(g, d, w, deg=1, center=cen)
            used = g[(g.d == d) & (g.value >= w[0]) & (g.value <= w[1]) & (g.errors > 0)].value
            if f is not None and len(used):
                xx = np.linspace(used.min(), used.max(), 50)
                ax.plot(xx, f(xx), color=M.RAMP[d], lw=1.0)
                fits[d] = (f, used.min(), used.max())
        # ratios of consecutive distances, which cross one at the crossing points
        for (d1, d2), mk, fill in (((3, 5), "o", False), ((5, 7), "^", True)):
            A = g[g.d == d1].set_index("value")
            B = g[g.d == d2].set_index("value")
            ps = sorted(p for p in set(A.index) & set(B.index) if lo <= p <= hi)
            qa = (A.loc[ps].errors / A.loc[ps].shots).values
            qb = (B.loc[ps].errors / B.loc[ps].shots).values
            rr = qb / qa
            er = rr * np.sqrt((1 - qa) / A.loc[ps].errors.values + (1 - qb) / B.loc[ps].errors.values)
            c = M.RAMP[d2]
            bx.errorbar(ps, rr, yerr=er, ls="none", marker=mk, ms=2.6, color=c, mec=c,
                        mfc=c if fill else "white", elinewidth=0.6, capsize=0,
                        label=rf"$p_L^{{({d2})}}/p_L^{{({d1})}}$")
            if d1 in fits and d2 in fits:
                x0 = max(fits[d1][1], fits[d2][1])
                x1 = min(fits[d1][2], fits[d2][2])
                xx = np.linspace(x0, x1, 50)
                bx.plot(xx, fits[d2][0](xx) / fits[d1][0](xx), color=c, lw=1.0)
        bx.axhline(1.0, color=M.INK2, lw=0.6)
        for axx in (ax, bx):
            axx.axvspan(pth - err, pth + err, color=M.DEC_COL["bm"], alpha=0.18, lw=0)
            axx.axvline(pth, color=M.DEC_COL["bm"], lw=0.8, ls=":")
            axx.set_xlim(lo, hi)
        ax.set_yscale("log")
        bx.set_yscale("log")
        bx.set_ylim(0.45, 2.2)
        bx.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.5, 1, 2]))
        bx.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        bx.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        bx.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        bx.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v*1e3:g}"))
        bx.set_xlabel(r"$p\;(\times10^{-3})$")
        ax.set_title(title + rf", $p_{{\mathrm{{th}}}}={100*pth:.2f}\%$", color=M.INK)
        if col:
            ax.tick_params(labelleft=False)
            bx.tick_params(labelleft=False)
    for axx in axes[0]:
        axx.set_ylim(8e-3, 0.3)
    axes[0, 0].set_ylabel(r"$p_L$")
    axes[1, 0].set_ylabel("ratio")
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[1, 0].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=5, frameon=False, fontsize=7,
               handlelength=1.2, columnspacing=1.6, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(w_pad=0.3, h_pad=0.4, rect=(0, 0.06, 1, 1))
    fig.savefig(f"{M.OUT}/fig_bm.pdf")
    plt.close(fig)


if __name__ == "__main__":
    a = read(THR / "bm_pooled.csv")
    res = analyse(a)
    pd.set_option("display.width", 200)
    print((res.set_index("noise") * 1e3).round(3).to_string())
    res.to_csv(OUT_CSV, index=False)
    figure(a, res)
