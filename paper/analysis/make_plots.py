"""Result figures of the paper (matplotlib with LaTeX text).

Usage: python paper/analysis/make_plots.py [scaling thr dec bm]

The Quantinuum figures come from helios_plots.py and the CFE comparison from
cfe_plots.py.

Reads docs/data and writes PDF figures to paper/build/figures. Needs a LaTeX
installation with the cm-super fonts for text.usetex.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from common import DATA, THR, load_fits, load_sweeps, out_dir, read  # noqa: E402

OUT = out_dir("figures")

INK, INK2, GUIDE, GRID = "#1C1917", "#57534E", "#7C3AED", "#E7E5E4"
RAMP = {3: "#F97316", 5: "#E11D48", 7: "#C026D3", 9: "#6D28D9"}
MARK = {3: "o", 5: "s", 7: "^", 9: "D"}
DEC_COL = {"mwpm": "#F97316", "seq_soft": "#EC4899", "seq_match": "#EAB308", "bm": "#7C3AED",
           "bp_full": "#7C3AED", "corr_links": "#C026D3", "corr_gauge": "#E11D48", "tesseract": "#1C1917",
           "bp_corr": "#9F1239", "ens": "#4C1D95", "cfe": "#4C1D95"}
DEC_LAB = {"mwpm": "MWPM", "seq_soft": "Sequential BP", "seq_match": "Erasure passing",
           "bm": "Belief-matching", "bp_full": "Belief-matching", "corr_links": "HCM, links",
           "corr_gauge": "HCM", "tesseract": "Tesseract", "cfe": "CFE"}
DEC_MARK = {"mwpm": "o", "seq_soft": "v", "seq_match": "x", "bm": "D", "bp_full": "D",
            "corr_links": "^", "corr_gauge": "s", "tesseract": "*", "cfe": "P"}

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "xtick.direction": "in", "ytick.direction": "in",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "legend.frameon": False, "legend.handlelength": 1.6, "legend.borderaxespad": 0.3,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
FULL, COL = 7.0, 3.375


def stderr(df):
    p = df.p_L.values
    return np.sqrt(np.maximum(p * (1 - p), 1e-15) / df.shots.values)


def series(ax, df, d, col=None, ls="-", filled=True, label=None, xcol="value", minerr=5, ms=3.2, lw=1.0, marker=None):
    s = df[(df.d == d) & (df.errors >= minerr)].sort_values(xcol)
    if s.empty:
        return
    c = col or RAMP[d]
    ax.errorbar(s[xcol].values, s.p_L.values, yerr=stderr(s), color=c, ls=ls, lw=lw,
                marker=marker or MARK[d], ms=ms, mfc=c if filled else "white", mec=c, mew=0.8,
                elinewidth=0.6, capsize=0, label=label)


def panel_label(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")


def place_text(ax, s, candidates=None, **kw):
    """Put s in the first corner of ax where it touches no curve, legend or other text."""
    try:
        import overlap_check as oc
    except ImportError:
        x, y, ha, va = (candidates or [(0.96, 0.05, "right", "bottom")])[0]
        return ax.text(x, y, s, transform=ax.transAxes, ha=ha, va=va, **kw)
    return oc.place(ax, s, candidates or oc.CORNERS, **kw)


def guide_label(ax, s, xs, lift=1.3, color=None, below=False):
    """Write s along the diagonal p_L = p, just above (or below) it, at the first x where it is clear."""
    try:
        import overlap_check as oc
    except ImportError:
        oc = None
    fig = ax.figure
    fig.draw_without_rendering()
    a = ax.transData.transform([[1e-3, 1e-3], [1e-2, 1e-2]])
    ang = np.degrees(np.arctan2(a[1, 1] - a[0, 1], a[1, 0] - a[0, 0]))
    kw = dict(color=color or GUIDE, fontsize=6.5, rotation=ang, rotation_mode="anchor", ha="left",
              va="top" if below else "bottom")
    for x in xs:
        t = ax.text(x, x / lift if below else x * lift, s, **kw)
        if oc is None or oc.is_clear(ax, t):
            return t
        t.remove()
    return ax.text(xs[0], xs[0] / lift if below else xs[0] * lift, s, **kw)


def all_threshold_data():
    return load_sweeps()


# --------------------------------------------------------------------------- sub-threshold scaling
def fig_scaling(df):
    r1 = df[(df.noise == "sd6") & (df.rounds == 1) & (df.decoder == "mwpm")]
    if r1.empty:
        r1 = read(DATA / "res_uniform_r1.csv").rename(columns={"p": "value"})
    rd = df[(df.noise == "sd6") & (df.rounds == df.d)]
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.95), sharey=True)
    titles = [r"$r=1$", r"$r=d$"]
    for ax, t in zip(axes, titles):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.plot([5e-4, 1.6e-2], [5e-4, 1.6e-2], color=GUIDE, lw=0.9, ls=":", zorder=0)
        ax.set_title(t, loc="left", color=INK)
        ax.set_xlabel(r"$p$")
        ax.set_xlim(8e-4, 1.5e-2)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    for d in (3, 5, 7, 9):
        series(axes[0], r1[r1.value <= 1.5e-2], d, label=f"$d={d}$", ms=2.8)
        series(axes[1], rd[(rd.decoder == "mwpm") & (rd.value <= 1.5e-2)], d, ls="--", filled=False, ms=2.8)
        series(axes[1], rd[(rd.decoder == "corr_gauge") & (rd.value <= 1.5e-2)], d, ms=2.8)
    axes[0].set_ylabel(r"logical error rate $p_L$")
    axes[0].set_ylim(5e-7, 0.6)
    axes[0].legend(loc="lower right", fontsize=6, handlelength=1.3, labelspacing=0.25)
    h = [Line2D([], [], color=INK, ls="--", marker="o", mfc="white", ms=2.8, lw=0.9, label="MWPM"),
         Line2D([], [], color=INK, ls="-", marker="o", ms=2.8, lw=0.9, label="HCM")]
    axes[1].legend(handles=h, loc="lower right", fontsize=6, handlelength=1.8)
    for ax, s in zip(axes, "ab"):
        panel_label(ax, f"({s})")
    fig.tight_layout(w_pad=0.4)
    guide_label(axes[1], r"$p_L=p$", [5.5e-3, 6.5e-3, 4.5e-3, 7.5e-3], below=True)
    fig.savefig(f"{OUT}/fig_scaling.pdf")
    plt.close(fig)




NOISES = [("sd6", "SD6"), ("si1000", "SI1000"), ("biased10", r"$\eta=10$"),
          ("biased100", r"$\eta=100$"), ("purez", r"Pure $Z$"), ("em3", "EM3")]


def fig_thresholds(df, th):
    fig, axes = plt.subplots(2, len(NOISES), figsize=(FULL, 3.15), sharey="row")
    labels = []
    for k, (noise, title) in enumerate(NOISES):
        for row, dec in enumerate(["mwpm", "corr_gauge"]):
            ax = axes[row, k]
            sub = df[(df.noise == noise) & (df.decoder == dec) & (df.rounds == df.d)]
            t = th.get((noise, dec))
            if t is not None:
                lo, hi = t[0] - t[1], t[0] + t[1]
                ax.axvspan(lo, hi, color=DEC_COL[dec], alpha=0.22, lw=0)
                ax.axvline(t[0], color=DEC_COL[dec], lw=0.6, ls=":")
                x0 = t[0]
                win = sub[(sub.value >= 0.6 * x0) & (sub.value <= 1.45 * x0)]
            else:
                win = sub
            for d in (3, 5, 7, 9):
                series(ax, win, d, ms=2.6, lw=0.9)
            ax.set_yscale("log")
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v*1e3:g}"))
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
            if row == 0:
                ax.set_title(title, color=INK)
            if row == 1:
                ax.set_xlabel(r"$p\;(\times10^{-3})$")
            if t is not None:
                # two short lines, so the label fits between the threshold line and the right edge
                txt = rf"$p_{{\mathrm{{th}}}}$" "\n" rf"${100*t[0]:.2f}\%$"
                labels.append((ax, txt))
        axes[0, 0].set_ylabel(r"MWPM" "\n" r"$p_L$")
        axes[1, 0].set_ylabel(r"HCM" "\n" r"$p_L$")
    handles = [Line2D([], [], color=RAMP[d], marker=MARK[d], ms=3, lw=0.9, label=f"$d={d}$") for d in (3, 5, 7, 9)]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(w_pad=0.3, h_pad=0.6, rect=(0, 0.05, 1, 1))
    for ax, txt in labels:
        place_text(ax, txt, fontsize=6.5, color=INK, multialignment="right", linespacing=1.1)
    fig.savefig(f"{OUT}/fig_thresholds.pdf")
    plt.close(fig)



def fig_decoders(df):
    fig = plt.figure(figsize=(FULL, 3.9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.8], hspace=0.42, wspace=0.62)
    decs = ["mwpm", "seq_match", "seq_soft", "corr_links", "corr_gauge", "bm", "tesseract", "cfe"]
    axes = {}
    for row, d in enumerate((3, 5)):
        for col, (noise, title) in enumerate([("sd6", "SD6"), ("biased100", r"Biased, $\eta=100$")]):
            ax = fig.add_subplot(gs[row, col], sharey=axes.get((row, 0)))
            axes[(row, col)] = ax
            for dec in decs:
                names = ["bm", "bp_full"] if dec == "bm" else [dec]
                sub = df[(df.noise == noise) & (df.decoder.isin(names)) & (df.d == d) & (df.rounds == d)]
                sub = sub[(sub.value >= 1.9e-3) & (sub.value <= 6.5e-3)]
                sub = sub.sort_values("shots").drop_duplicates("value", keep="last")
                series(ax, sub, d, col=DEC_COL[dec], marker=DEC_MARK[dec], label=DEC_LAB[dec],
                       ms=3.3 if dec != "tesseract" else 5.0, lw=0.9,
                       ls="--" if dec in ("mwpm", "tesseract") else "-", minerr=8)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_title(rf"{title}, $d={d}$, $r={d}$", loc="left", color=INK)
            ax.set_xticks([2e-3, 3e-3, 4e-3, 5e-3, 6e-3])
            ax.set_xticklabels([r"$2$", r"$3$", r"$4$", r"$5$", r"$6$"])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            if row == 1:
                ax.set_xlabel(r"$p\;(\times10^{-3})$")
            if col == 0:
                ax.set_ylabel(r"logical error rate $p_L$")
            else:
                plt.setp(ax.get_yticklabels(), visible=False)
            panel_label(ax, "(" + "abde"[2 * row + col] + ")")
    # timing
    ax = fig.add_subplot(gs[0, 2])
    tm = read(THR / "timing.csv")
    if not tm.empty:
        order = [d for d in decs if d in set(tm.decoder)]
        y = np.arange(len(order))
        vals = [float(tm[tm.decoder == d].us_per_shot.iloc[0]) for d in order]
        ax.barh(y, vals, left=1.0, color=[DEC_COL[d] for d in order], height=0.62)
        for yy, v, d in zip(y, vals, order):
            lab = f"{v:.0f}" + r"$\,\mu$s" if v < 1000 else (f"{v/1000:.1f} ms" if v < 1e6 else f"{v/1e6:.1f} s")
            # name and time to the right of each bar, so nothing reaches into the next panel
            ax.text(v * 1.5, yy, f"{DEC_LAB[d]}, {lab}", va="center", ha="left", fontsize=6.2, color=INK)
        ax.set_yticks([])
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlim(1, 3e10)
        ax.set_xticks([1e1, 1e3, 1e5, 1e7, 1e9])
        ax.grid(False)
        ax.set_xlabel(r"time per shot ($\mu$s)")
        ax.grid(axis="y", visible=False)
        ax.set_title(r"SD6, $d=5$, $p=4\times10^{-3}$", loc="left", color=INK)
    panel_label(ax, "(c)")
    # legend panel
    lax = fig.add_subplot(gs[1, 2])
    lax.axis("off")
    h, l = axes[(1, 0)].get_legend_handles_labels()
    order = [DEC_LAB[d] for d in decs]
    hl = sorted(zip(h, l), key=lambda t: order.index(t[1]) if t[1] in order else 99)
    lax.legend([x[0] for x in hl], [x[1] for x in hl], loc="center", fontsize=7, labelspacing=0.45, handlelength=2.2)
    fig.savefig(f"{OUT}/fig_decoders.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------- Quantinuum
# --------------------------------------------------------------------------- rounds
# --------------------------------------------------------------------------- belief-matching thresholds
if __name__ == "__main__":
    df = all_threshold_data()
    th = load_fits()
    # fig_bm is drawn by bm_thresholds.py
    which = sys.argv[1:] or ["scaling", "thr", "dec"]
    if "scaling" in which:
        fig_scaling(df)
    if "thr" in which:
        fig_thresholds(df, th)
    if "dec" in which:
        fig_decoders(df)
    if "bm" in which:
        import runpy
        runpy.run_path(os.path.join(os.path.dirname(__file__), "bm_thresholds.py"), run_name="__main__")
    print("figures in", OUT)
