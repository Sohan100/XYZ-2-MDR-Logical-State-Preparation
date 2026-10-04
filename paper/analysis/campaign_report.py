"""Thresholds and figures of the full campaign (every noise model, decoder, number of rounds and distance).

usage: python paper/analysis/campaign_report.py [--points docs/data/campaign/points.csv] [--no-pool]
                                                [--out paper/figures/campaign] [--workers N]

Reads the merged campaign data (scripts/campaign.py merge), pools it with the earlier sweeps of
docs/data/thresholds and docs/data/rounds_threshold (same circuits, same noise models) unless
--no-pool is given, and writes

  docs/data/campaign/thresholds.csv    one row per (noise, decoder, rounds)
  docs/data/campaign/crossings.csv     crossing of every pair of consecutive distances
  docs/data/campaign/thresholds.md     the tables in markdown
  paper/figures/campaign/tab_*.tex     LaTeX tables (caption below the tabular)
  paper/figures/campaign/*.pdf         figures, see FIGURES below

Threshold estimate. For every (noise, decoder, r) the crossings of consecutive distances (d, d + 2)
come from weighted line fits of ln p_L against p near the crossing (paper/analysis/crossings.py),
with bootstrap errors. With at least three distances d >= 11 the threshold is a finite-size-scaling
fit p_L = A + B x + C x^2, x = (p - p_th) d^(1/nu), to the largest distances (at most six) in a window
of +-20% around the last crossings. Its error adds the bootstrap error and half the spread of the
fits with windows of 15% and 25% and with the largest four, five and six distances. Otherwise the
threshold is the crossing of the two largest distances. When the crossings fall by more than 25%
from the smallest to the largest distances and keep falling, the series has no threshold (as with
measurement crosstalk) and the table reports the last crossing with the flag "drift".

FIGURES
  thr_<noise>_<decoder>.pdf   p_L against p for every r (3 x 3 panels) and every distance
  rounds_<noise>.pdf          threshold against the number of rounds for every decoder
  crossings_<noise>.pdf       crossings of consecutive distances against distance (r = 1 and r = d)
  fig_thr_<decoder>.pdf       paper style, r = d, one panel per noise model, d = 3 to 21 (r = 1 for a
                              decoder without any r = d threshold, i.e. CFE + TN)
  summary.pdf                 thresholds of every decoder and noise model for r = 1 and r = d
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))

import crossings as X  # noqa: E402

DATA = ROOT / "docs" / "data"
KEYS = ["noise", "value", "d", "rounds", "final", "decoder"]
NOISES = ["sd6", "si1000", "biased10", "biased100", "purez", "em3",
          "helios_p_noxt", "h2_p_noxt", "helios_p", "h2_p"]
NOISE_LAB = {"sd6": "SD6", "si1000": "SI1000", "biased10": r"Biased, $\eta=10$", "biased100": r"Biased, $\eta=100$",
             "purez": r"Pure $Z$", "em3": "EM3", "helios_p": "Helios", "h2_p": "H2",
             "helios_p_noxt": "Helios, no crosstalk", "h2_p_noxt": "H2, no crosstalk"}
DECODERS = ["mwpm", "corr_links", "corr_gauge", "seq_match", "seq_soft", "seq_erasure", "bm", "bp_full", "bp_corr",
            "tesseract", "cfe", "cfe0", "cfe_tn", "tnml"]
DEC_LAB = {"mwpm": "MWPM", "corr_links": "HCM, links", "corr_gauge": "HCM", "seq_match": "Erasure passing",
           "seq_soft": "Sequential BP", "bm": "Belief-matching", "tesseract": "Tesseract", "cfe": "CFE",
           "cfe0": "CFE-0", "cfe_tn": "CFE + TN", "seq_erasure": "Sequential BP, erasures",
           "bp_full": "Belief-matching, 30 it.", "bp_corr": "BP + HCM", "tnml": "TN ML"}
DEC_COL = {"mwpm": "#F97316", "corr_links": "#C026D3", "corr_gauge": "#E11D48", "seq_match": "#EAB308",
           "seq_soft": "#EC4899", "bm": "#7C3AED", "tesseract": "#1C1917", "cfe": "#4C1D95", "cfe0": "#9F1239",
           "cfe_tn": "#B45309", "seq_erasure": "#DB2777", "bp_full": "#6366F1", "bp_corr": "#0E7490",
           "tnml": "#78350F"}
DEC_MARK = {"mwpm": "o", "corr_links": "^", "corr_gauge": "s", "seq_match": "x", "seq_soft": "v", "bm": "D",
            "tesseract": "*", "cfe": "P", "cfe0": "p", "cfe_tn": "h", "seq_erasure": "<", "bp_full": "d",
            "bp_corr": ">", "tnml": "H"}
RKEYS = [str(r) for r in range(1, 22)] + ["d"]
DISTS = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
DCOL = dict(zip(DISTS, ["#FBBF24", "#F59E0B", "#F97316", "#EF4444", "#E11D48", "#EC4899", "#D946EF",
                        "#9333EA", "#6D28D9", "#3B0764"]))
DMARK = dict(zip(DISTS, ["o", "s", "^", "D", "v", "o", "s", "^", "D", "v"]))
INK, INK2, GRID, GUIDE = "#1C1917", "#57534E", "#E7E5E4", "#7C3AED"
FULL = 7.0
RNG = np.random.default_rng(2026)


# --------------------------------------------------------------------------- data
def _read(path):
    path = Path(path)
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 80 else pd.DataFrame()


def load(points, pool=True):
    parts = [_read(points)]
    if pool:
        parts.append(_read(DATA / "thresholds" / "sweeps_pooled.csv"))
        rt = _read(DATA / "rounds_threshold" / "sweeps_pooled.csv")
        if not rt.empty:
            rt = pd.DataFrame(dict(noise="sd6", value=rt.p, d=rt.d, rounds=np.where(rt.rmode == "d", rt.d, rt.r),
                                   final="frame", decoder="mwpm", shots=rt.shots, errors=rt.errors,
                                   seconds=rt.get("seconds", 0.0)))
            parts.append(rt)
    df = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    if "seconds" not in df:
        df["seconds"] = 0.0
    df["value"] = df.value.astype(float).round(10)
    df = df.groupby(KEYS, as_index=False).agg(shots=("shots", "sum"), errors=("errors", "sum"),
                                              seconds=("seconds", "sum"))
    df = df[df.shots > 0]
    df["p_L"] = df.errors / df.shots
    df["rkey"] = np.where(df.rounds == df.d, "d", df.rounds.astype(str))
    # with r = d the point belongs to the r = d series; a fixed r equal to d is the same circuit
    extra = df[(df.rkey == "d") & df.rounds.astype(str).isin(RKEYS[:-1])].copy()
    extra["rkey"] = extra.rounds.astype(str)
    return pd.concat([df, extra], ignore_index=True)


# --------------------------------------------------------------------------- fits
def fss(g, dset, center, half, boot=0, plo=0.003, phi=0.45):
    """Finite-size-scaling fit p_L = A + B x + C x^2, x = (p - p_th) d^(1/nu)."""
    s = g[g.d.isin(dset) & (np.abs(g.value / center - 1) <= half) & (g.p_L > plo) & (g.p_L < phi)]
    if len(s) < 7 or s.d.nunique() < 3:
        return None
    d = s.d.values.astype(float)
    p = s.value.values
    n = s.shots.values.astype(float)
    y0 = s.p_L.values

    def resid(th, y):
        pth, nu, A, B, C = th
        x = (p - pth) * d ** (1 / nu)
        m = A + B * x + C * x * x
        return (m - y) / np.sqrt(np.clip(m * (1 - m), 1e-6, None) / n)

    bounds = ([center * (1 - half), 0.3, 0, -1e7, -1e10], [center * (1 + half), 6.0, 1, 1e7, 1e10])
    best = None
    for nu0 in (0.8, 1.2, 1.8):
        try:
            f = least_squares(resid, [center, nu0, float(np.median(y0)), 20.0, 0.0], args=(y0,), bounds=bounds)
        except ValueError:
            continue
        if best is None or f.cost < best.cost:
            best = f
    if best is None:
        return None
    out = dict(pth=float(best.x[0]), nu=float(best.x[1]), chi2=float(2 * best.cost / max(len(y0) - 5, 1)),
               npts=len(y0), stat=np.nan)
    if boot:
        bs = []
        for _ in range(boot):
            yb = RNG.binomial(n.astype(int), np.clip(y0, 0, 1)) / n
            try:
                bs.append(least_squares(resid, best.x, args=(yb,), bounds=bounds).x[0])
            except ValueError:
                pass
        out["stat"] = float(np.std(bs)) if len(bs) > 10 else np.nan
    return out


def first_guess(g, d1, d2):
    """p where ln p_L of d2 overtakes that of d1, from the values both distances share."""
    a = g[g.d == d1].set_index("value").p_L
    b = g[g.d == d2].set_index("value").p_L
    common = sorted(set(a.index) & set(b.index))
    common = [v for v in common if a[v] > 0 and b[v] > 0]
    if len(common) < 2:
        return np.nan
    diff = np.array([math.log(b[v]) - math.log(a[v]) for v in common])
    for i in range(len(common) - 1):
        if diff[i] < 0 <= diff[i + 1]:
            t = -diff[i] / (diff[i + 1] - diff[i])
            return float(np.exp((1 - t) * np.log(common[i]) + t * np.log(common[i + 1])))
    return np.nan


def pair_crossings(g):
    out = []
    ds = sorted(int(x) for x in g.d.unique())
    for d1, d2 in zip(ds, ds[1:]):
        x0 = first_guess(g, d1, d2)
        if not np.isfinite(x0):
            continue
        try:
            c, e, w = X.refine(g, d1, d2, x0, rel=0.15, deg=1, boot=200)
        except Exception:  # noqa: BLE001
            continue
        if np.isfinite(c) and 0.5 * x0 < c < 2 * x0:
            out.append(dict(d1=d1, d2=d2, p=c, err=e))
    return out


def analyse(job):
    noise, dec, rkey, g = job
    cr = pair_crossings(g)
    row = dict(noise=noise, decoder=dec, rounds=rkey, method="none", pth=np.nan, err=np.nan, stat=np.nan,
               syst=np.nan, nu=np.nan, chi2=np.nan, npts=0, dmin=int(g.d.min()), dmax=int(g.d.max()),
               cross_first=np.nan, cross_last=np.nan, cross_last_err=np.nan, drift=np.nan, flag="",
               shots=int(g.shots.sum()), core_hours=float(g.seconds.sum()) / 3600)
    if not cr:
        return row, cr
    row.update(cross_first=cr[0]["p"], cross_last=cr[-1]["p"], cross_last_err=cr[-1]["err"])
    if len(cr) >= 3:
        row["drift"] = cr[-1]["p"] / cr[0]["p"]
        tail = [c["p"] for c in cr[-3:]]
        if row["drift"] < 0.75 and tail[2] < tail[1] < tail[0]:
            row["flag"] = "drift"
    big = sorted(d for d in g.d.unique() if d >= 11)
    c0 = float(np.median([c["p"] for c in cr[-3:]]))
    if len(big) >= 3 and row["flag"] != "drift":
        sets = [big[-k:] for k in (6, 5, 4) if len(big) >= min(k, 3)]
        main = fss(g, sets[0], c0, 0.2, boot=150)
        if main is not None:
            variants = [f["pth"] for half in (0.15, 0.25) for ds in sets
                        for f in [fss(g, ds, main["pth"], half)] if f is not None]
            syst = 0.5 * (max(variants) - min(variants)) if variants else 0.0
            stat = main["stat"] if np.isfinite(main["stat"]) else 0.0
            row.update(method="fss", pth=main["pth"], stat=stat, syst=syst, err=float(np.hypot(stat, syst)),
                       nu=main["nu"], chi2=main["chi2"], npts=main["npts"], dmin=int(min(sets[0])))
            return row, cr
    last = cr[-1]
    row.update(method="pair", pth=last["p"], err=last["err"], stat=last["err"], syst=0.0,
               dmin=last["d1"], dmax=last["d2"])
    return row, cr


def thresholds(df, workers=0):
    jobs = []
    for (noise, dec, rkey), g in df.groupby(["noise", "decoder", "rkey"]):
        if g.d.nunique() >= 2:
            jobs.append((noise, dec, rkey, g[["d", "value", "shots", "errors", "p_L", "seconds"]].copy()))
    workers = workers or max(1, min(len(jobs), (os.cpu_count() or 2) // 2))
    if workers > 1:
        with Pool(workers) as pool:
            res = pool.map(analyse, jobs, chunksize=1)
    else:
        res = [analyse(j) for j in jobs]
    th = pd.DataFrame([r for r, _ in res])
    cr = pd.DataFrame([dict(noise=j[0], decoder=j[1], rounds=j[2], **c) for j, (_, cs) in zip(jobs, res) for c in cs])
    return th, cr


# --------------------------------------------------------------------------- figures
def setup_style():
    import matplotlib
    matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    tex = shutil.which("latex") is not None and shutil.which("dvipng") is not None
    plt.rcParams.update({
        "text.usetex": tex, "font.family": "serif", "mathtext.fontset": "cm",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2,
        "ytick.color": INK2, "xtick.direction": "in", "ytick.direction": "in", "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "axes.unicode_minus": False,
    })
    if tex:
        plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amssymb}"
    return plt


def pct(x):
    return r"\%" if PLT.rcParams["text.usetex"] else "%"


def _stderr(s):
    p = s.p_L.values
    return np.sqrt(np.maximum(p * (1 - p), 1e-12) / s.shots.values)


def curves(ax, g, pth=None, err=None, lo=None, hi=None, minerr=3, ms=2.4, logx=True):
    import matplotlib
    if pth is not None and np.isfinite(pth):
        if err is not None and np.isfinite(err):
            ax.axvspan(pth - err, pth + err, color=GUIDE, alpha=0.16, lw=0, zorder=0)
        ax.axvline(pth, color="#5B21B6", lw=0.7, ls=":", zorder=0)
    w = g[g.errors >= minerr]
    if lo is not None:
        w = w[(w.value >= lo) & (w.value <= hi)]
    for d in DISTS:
        s = w[w.d == d].sort_values("value")
        if s.empty:
            continue
        ax.errorbar(s.value, s.p_L, yerr=_stderr(s), color=DCOL[d], marker=DMARK[d], ms=ms, lw=0.85,
                    elinewidth=0.6, capsize=0, zorder=2 + d / 100)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0,) if logx else (1.0, 2.0, 5.0)))
    ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    if logx:
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{1e3 * v:g}"))


def dist_legend(fig, y=-0.01):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=DCOL[d], marker=DMARK[d], ms=3, lw=0.9, label=f"${d}$") for d in DISTS]
    fig.legend(handles=handles, loc="lower center", ncol=10, frameon=False, bbox_to_anchor=(0.53, y),
               title=r"distance $d$", title_fontsize=7, fontsize=7, handlelength=1.4, columnspacing=1.0)


def fig_noise_decoder(df, th, noise, dec, out):
    plt = PLT
    sub = df[(df.noise == noise) & (df.decoder == dec)]
    if sub.empty:
        return
    # one panel per number of rounds with points to draw, five per row
    rks = [rk for rk in RKEYS if ((sub.rkey == rk) & (sub.errors >= 3)).any()]
    if not rks:
        return
    ncol = min(5, len(rks))
    nrow = math.ceil(len(rks) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(FULL, 1.9 * nrow + 0.6), squeeze=False)
    for ax in axes.flat[len(rks):]:
        ax.set_visible(False)
    for ax, rk in zip(axes.flat, rks):
        g = sub[sub.rkey == rk]
        t = th[(th.noise == noise) & (th.decoder == dec) & (th.rounds == rk)]
        pth = float(t.pth.iloc[0]) if not t.empty else np.nan
        err = float(t.err.iloc[0]) if not t.empty else np.nan
        curves(ax, g, pth, err)
        lab = r"$r=d$" if rk == "d" else rf"$r={rk}$"
        ax.set_title(lab, loc="left", color=INK)
        if np.isfinite(pth):
            flag = t.flag.iloc[0]
            txt = (rf"$p_c^{{\,{int(t.dmax.iloc[0])}}}={100 * pth:.3f}${pct(0)}" if flag == "drift"
                   else rf"$p_{{\mathrm{{th}}}}={100 * pth:.3f}${pct(0)}")
            ax.text(0.04, 0.95, txt, transform=ax.transAxes, fontsize=6.8, color=INK, va="top")
    for ax in axes[-1]:
        ax.set_xlabel(r"$p\;(\times10^{-3})$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$p_L$")
    fig.suptitle(f"{NOISE_LAB[noise]}, {DEC_LAB.get(dec, dec)}", x=0.02, ha="left", fontsize=9, color=INK)
    dist_legend(fig)
    fig.tight_layout(w_pad=0.6, h_pad=0.8, rect=(0, 0.05, 1, 0.97))
    fig.savefig(out / f"thr_{noise}_{dec}.pdf")
    plt.close(fig)


def fig_rounds(th, noise, out):
    plt = PLT
    t = th[(th.noise == noise) & th.pth.notna()]
    if t.empty:
        return
    xs = {rk: i for i, rk in enumerate(RKEYS)}
    fig, ax = plt.subplots(figsize=(FULL, 3.2))
    for dec in DECODERS:
        s = t[t.decoder == dec]
        if s.empty:
            continue
        s = s.assign(x=s.rounds.map(xs)).sort_values("x")
        mfc = ["white" if f == "drift" else DEC_COL[dec] for f in s.flag]
        ax.errorbar(s.x, 100 * s.pth, yerr=100 * s.err.fillna(0), color=DEC_COL[dec], lw=0.9, ms=3.5,
                    marker=DEC_MARK[dec], elinewidth=0.6, capsize=0, label=DEC_LAB[dec], mfc=DEC_COL[dec])
        for x, y, c in zip(s.x, 100 * s.pth, mfc):
            if c == "white":
                ax.plot([x], [y], marker=DEC_MARK[dec], ms=3.5, mfc="white", mec=DEC_COL[dec], ls="none")
    ax.set_xticks(range(len(RKEYS)))
    ax.set_xticklabels([rk if rk != "d" else "$d$" for rk in RKEYS], fontsize=6)
    ax.set_yscale("log")
    import matplotlib
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel(r"rounds $r$")
    ax.set_ylabel(f"threshold ({pct(0)})")
    ax.set_title(NOISE_LAB[noise], loc="left", color=INK)
    ax.legend(fontsize=5.5, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / f"rounds_{noise}.pdf")
    plt.close(fig)


def fig_crossings(cr, noise, out):
    plt = PLT
    c = cr[cr.noise == noise]
    if c.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.5), sharey=False)
    for ax, rk in zip(axes, ["1", "d"]):
        s0 = c[c.rounds == rk]
        for dec in DECODERS:
            s = s0[s0.decoder == dec].sort_values("d1")
            if s.empty:
                continue
            ax.errorbar(s.d1 + 1, 100 * s.p, yerr=100 * s.err.fillna(0), color=DEC_COL[dec], marker=DEC_MARK[dec],
                        ms=3, lw=0.9, elinewidth=0.6, capsize=0, label=DEC_LAB[dec])
        ax.set_xlabel(r"middle distance $d+1$")
        ax.set_ylabel(f"crossing of $d$ and $d+2$ ({pct(0)})")
        ax.set_title((r"$r=1$" if rk == "1" else r"$r=d$") + f", {NOISE_LAB[noise]}", loc="left", color=INK)
        ax.set_xticks([4, 8, 12, 16, 20])
    axes[1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out / f"crossings_{noise}.pdf")
    plt.close(fig)


PAPER_NOISES = ["sd6", "si1000", "biased10", "biased100", "purez", "em3"]
HW_NOISES = ["helios_p_noxt", "h2_p_noxt", "helios_p", "h2_p"]


def fig_paper(df, th, dec, noises, name, out, lo=0.7, hi=1.4, rkey="d"):
    plt = PLT
    ncol = 3
    nrow = math.ceil(len(noises) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(FULL, 2.15 * nrow + 0.4), squeeze=False)
    any_panel = False
    for ax, noise in zip(axes.flat, noises):
        g = df[(df.noise == noise) & (df.decoder == dec) & (df.rkey == rkey)]
        t = th[(th.noise == noise) & (th.decoder == dec) & (th.rounds == rkey)]
        if g.empty or t.empty or not np.isfinite(t.pth.iloc[0]):
            ax.set_visible(False)
            continue
        any_panel = True
        pth, err = float(t.pth.iloc[0]), float(t.err.iloc[0])
        curves(ax, g, pth, err, lo=lo * pth, hi=hi * pth, logx=False)
        ax.set_xlim(lo * pth * 0.98, hi * pth * 1.02)
        ax.set_title(NOISE_LAB[noise] + ("" if rkey == "d" else rf", $r={rkey}$"), loc="left", color=INK)
        txt = (rf"$p_c={100 * pth:.3f}${pct(0)}" if t.flag.iloc[0] == "drift"
               else rf"$p_{{\mathrm{{th}}}}={100 * pth:.3f}${pct(0)}")
        ax.text(0.04, 0.95, txt, transform=ax.transAxes, fontsize=6.8, color=INK, va="top")
    for ax in axes.flat[len(noises):]:
        ax.set_visible(False)
    if not any_panel:
        plt.close(fig)
        return
    for ax in axes[-1]:
        ax.set_xlabel(r"$p\;(\times10^{-3})$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"logical error rate $p_L$")
    for ax, s in zip([a for a in axes.flat if a.get_visible()], "abcdefghij"):
        ax.text(-0.02, 1.02, f"({s})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")
    dist_legend(fig)
    fig.tight_layout(w_pad=0.6, h_pad=0.8, rect=(0, 0.07, 1, 1))
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def fig_summary(th, out):
    plt = PLT
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 3.2))
    for ax, rk in zip(axes, ["1", "d"]):
        t = th[(th.rounds == rk)]
        noises = [n for n in NOISES if n in set(t.noise)]
        decs = [d for d in DECODERS if d in set(t.decoder)]
        M = np.full((len(noises), len(decs)), np.nan)
        for i, n in enumerate(noises):
            for j, d in enumerate(decs):
                s = t[(t.noise == n) & (t.decoder == d)]
                if not s.empty and s.flag.iloc[0] != "drift":
                    M[i, j] = 100 * s.pth.iloc[0]
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("warm", ["#FEF3C7", "#F97316", "#E11D48", "#6D28D9"])
        im = ax.imshow(M, cmap=cmap, aspect="auto")
        for i in range(len(noises)):
            for j in range(len(decs)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                            color="white" if M[i, j] > np.nanpercentile(M, 60) else INK)
        ax.set_xticks(range(len(decs)))
        ax.set_xticklabels([DEC_LAB[d] for d in decs], rotation=60, ha="right", fontsize=6)
        ax.set_yticks(range(len(noises)))
        ax.set_yticklabels([NOISE_LAB[n] for n in noises], fontsize=6)
        ax.grid(False)
        ax.set_title((r"$r=1$" if rk == "1" else r"$r=d$") + f", threshold ({pct(0)})", loc="left", color=INK)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out / "summary.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- tables
def fmt(p, e):
    if not np.isfinite(p):
        return "--"
    if not np.isfinite(e) or e <= 0:
        return f"{100 * p:.3f}"
    digits = min(max(0, -int(math.floor(math.log10(100 * e)))), 4)
    err = int(round(100 * e * 10 ** digits))
    if err >= 10 and digits > 0:
        digits -= 1
        err = int(round(100 * e * 10 ** digits))
    return f"{100 * p:.{digits}f}({err})"


def tables(th, out_tex: Path, out_md: Path):
    lines_md = ["# Thresholds of the full campaign", "",
                "Thresholds in percent of p with their uncertainty in the last digits. "
                "`drift`: the crossings of consecutive distances keep falling (no threshold), "
                "the value is the crossing of the two largest distances. "
                "Method: finite-size scaling on the largest distances (fss) or crossing of the two largest (pair).", ""]
    for rk in RKEYS:
        t = th[th.rounds == rk]
        if t.empty:
            continue
        decs = [d for d in DECODERS if d in set(t.decoder)]
        noises = [n for n in NOISES if n in set(t.noise)]
        lines_md.append(f"## r = {rk}")
        lines_md.append("")
        lines_md.append("| noise | " + " | ".join(DEC_LAB[d] for d in decs) + " |")
        lines_md.append("|---|" + "---|" * len(decs))
        for n in noises:
            cells = []
            for d in decs:
                s = t[(t.noise == n) & (t.decoder == d)]
                if s.empty:
                    cells.append("")
                    continue
                r = s.iloc[0]
                c = fmt(r.pth, r.err)
                if r.flag == "drift":
                    c += " drift"
                elif r.method == "pair":
                    c += f" ({int(r.dmin)}/{int(r.dmax)})"
                cells.append(c)
            lines_md.append(f"| {n} | " + " | ".join(cells) + " |")
        lines_md.append("")
    out_md.write_text("\n".join(lines_md))

    for rk, name, what in [("d", "tab_campaign_rd", r"$r=d$"), ("1", "tab_campaign_r1", r"$r=1$")]:
        t = th[th.rounds == rk]
        if t.empty:
            continue
        decs = [d for d in DECODERS if d in set(t.decoder)]
        noises = [n for n in NOISES if n in set(t.noise)]
        rows = []
        for n in noises:
            cells = []
            for d in decs:
                s = t[(t.noise == n) & (t.decoder == d)]
                if s.empty:
                    cells.append("")
                    continue
                r = s.iloc[0]
                c = fmt(r.pth, r.err)
                if r.flag == "drift":
                    c = r"\textit{" + c + "}"
                elif r.method == "pair":
                    c += r"$^\dagger$"
                cells.append(c)
            rows.append(NOISE_LAB[n] + " & " + " & ".join(cells) + r" \\")
        tex = "\n".join([
            r"\begin{table*}[t]",
            r"  \begin{ruledtabular}",
            r"  \begin{tabular}{l" + "c" * len(decs) + "}",
            "    Noise & " + " & ".join(DEC_LAB[d] for d in decs) + r" \\",
            r"    \hline",
            *["    " + x for x in rows],
            r"  \end{tabular}",
            r"  \end{ruledtabular}",
            r"  \caption{Thresholds (in \%) of every decoder with " + what + r" rounds, from finite-size-scaling fits "
            r"to the largest distances (up to $d=21$). A dagger marks the crossing of the two largest distances "
            r"when fewer than three distances $d\ge 11$ were simulated, and italics the crossing of the two largest "
            r"distances when the crossings keep falling with $d$ (no threshold).}",
            r"  \label{tab:" + name.replace("tab_", "") + "}",
            r"\end{table*}", ""])
        (out_tex / f"{name}.tex").write_text(tex)


PLT = None


def main():
    global PLT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", default=str(DATA / "campaign" / "points.csv"))
    ap.add_argument("--no-pool", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "paper" / "figures" / "campaign"))
    ap.add_argument("--data-out", default=str(DATA / "campaign"))
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dout = Path(args.data_out)
    dout.mkdir(parents=True, exist_ok=True)

    df = load(args.points, pool=not args.no_pool)
    print(f"{len(df)} points, {df.shots.sum():.3g} shots", flush=True)
    th, cr = thresholds(df, args.workers)
    th = th.sort_values(["noise", "decoder", "rounds"])
    th.to_csv(dout / "thresholds.csv", index=False)
    cr.to_csv(dout / "crossings.csv", index=False)
    tables(th, out, dout / "thresholds.md")
    print(f"{len(th)} thresholds -> {dout / 'thresholds.csv'}", flush=True)
    if args.no_figures:
        return
    PLT = setup_style()
    for noise in NOISES:
        for dec in DECODERS:
            fig_noise_decoder(df, th, noise, dec, out)
        fig_rounds(th, noise, out)
        fig_crossings(cr, noise, out)
    for dec in DECODERS:
        # paper style at r = d; a decoder without any r = d threshold (CFE + TN, whose network converges
        # at r = d only for d = 3) is shown at r = 1 instead, and its panels say so
        rk = "d" if th[(th.decoder == dec) & (th.rounds == "d")].pth.notna().any() else "1"
        fig_paper(df, th, dec, PAPER_NOISES, f"fig_thr_{dec}", out, rkey=rk)
        fig_paper(df, th, dec, HW_NOISES, f"fig_thr_hw_{dec}", out, lo=0.6, hi=1.6, rkey=rk)
    fig_summary(th, out)
    print(f"figures -> {out}", flush=True)


if __name__ == "__main__":
    main()
