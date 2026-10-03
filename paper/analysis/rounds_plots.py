"""Figures and table for the threshold as a function of the number of rounds.

usage: python paper/analysis/rounds_plots.py

Needs docs/data/rounds_threshold/thresholds.csv from rounds_thresholds.py.
Writes paper/build/figures/fig_rounds_curves.pdf (zoomed on each threshold),
fig_rounds_full.pdf (full sweep), fig_rounds_pth.pdf and
paper/build/tables/tab_rounds.tex.
"""
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import make_plots as M  # noqa: E402  (rcParams and palette)
from common import out_dir  # noqa: E402
from rounds_thresholds import ROUNDS_DIR, load  # noqa: E402
from latex_util import caption_below  # noqa: E402

OUT = str(out_dir("figures"))
TABLES = out_dir("tables")
DISTS = [3, 5, 7, 9, 11, 13, 15, 17, 19]
DCOL = dict(zip(DISTS, ["#FBBF24", "#F59E0B", "#F97316", "#EF4444", "#E11D48", "#EC4899", "#D946EF",
                        "#9333EA", "#5B21B6"]))
KEYS = ["1", "2", "3", "4", "5", "6", "8", "10", "d"]
RCOL = LinearSegmentedColormap.from_list("rr", ["#F97316", "#E11D48", "#C026D3", "#6D28D9", "#4C1D95"])


def data():
    return load()


def thresholds():
    t = pd.read_csv(ROUNDS_DIR / "thresholds.csv")
    t["key"] = t["key"].astype(str)
    return t


def fig_curves(df, th, zoom=True, name="fig_rounds_curves"):
    """Logical error rate versus p for every round count.

    With zoom=True each panel shows a window of p around its threshold, on a
    linear p axis, so the crossing is visible. With zoom=False the panels show
    the full sweep up to p = 0.1 (appendix figure).
    """
    fig, axes = plt.subplots(3, 3, figsize=(M.FULL, 5.6), sharex=not zoom, sharey=not zoom)
    for ax, key in zip(axes.flat, KEYS):
        g = df[df.key == key]
        f = th[(th.key == key) & (th.kind == "fss")]
        pth = err_th = None
        if not f.empty:
            pth, err_th = float(f.p.iloc[0]), float(f.err.iloc[0])
            ax.axvspan(pth - err_th, pth + err_th, color="#7C3AED", alpha=0.18, lw=0, zorder=0)
            ax.axvline(pth, color="#5B21B6", lw=0.7, ls=":", zorder=0)
            val = f"{100*pth:.2f}" if pth >= 0.01 else f"{100*pth:.3f}"
            ax.text(0.96, 0.06, rf"$p_{{\mathrm{{th}}}}={val}\%$", transform=ax.transAxes, fontsize=7.5,
                    color=M.INK, ha="right", va="bottom")
        lo, hi = (0.6 * pth, 1.4 * pth) if (zoom and pth) else (1.8e-3, 1.1e-1)
        ys = []
        for d in DISTS:
            s = g[(g.d == d) & (g.errors >= 3) & (g.p >= lo * 0.999) & (g.p <= hi * 1.001)].sort_values("p")
            if s.empty:
                continue
            err = np.sqrt(np.maximum(s.pL * (1 - s.pL), 1e-12) / s.shots)
            ax.errorbar(s.p, s.pL, yerr=err, color=DCOL[d], lw=0.8, marker="o", ms=1.8 if zoom else 1.6, mew=0,
                        elinewidth=0.5, capsize=0, label=f"$d={d}$")
            ys += list(s.pL)
        ax.set_yscale("log")
        if zoom and pth:
            ax.set_xlim(lo, hi)
            if ys:
                ax.set_ylim(max(min(ys) * 0.6, 1e-5), min(max(ys) * 1.6, 0.7))
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
            dec = 1 if pth >= 0.01 else 2
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _, k=dec: f"{100*v:.{k}f}"))
            ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
            ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        else:
            ax.set_xscale("log")
            ax.set_xlim(lo, hi)
            ax.set_ylim(1e-4, 0.8)
            ax.axhline(0.5, color=M.INK2, lw=0.5, ls="--", zorder=0)
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        title = r"$r=d$" if key == "d" else rf"$r={key}$"
        ax.set_title(title, loc="left", color=M.INK)
    for ax in axes[-1]:
        ax.set_xlabel(r"physical error rate $p$ (\%)" if zoom else r"physical error rate $p$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$p_L$")
    handles = [Line2D([], [], color=DCOL[d], marker="o", ms=2.5, lw=0.9, label=f"$d={d}$") for d in DISTS]
    fig.legend(handles=handles, loc="lower center", ncol=9, frameon=False, bbox_to_anchor=(0.5, -0.005),
               columnspacing=1.0, handlelength=1.4)
    fig.tight_layout(w_pad=0.3, h_pad=0.5, rect=(0, 0.035, 1, 1))
    fig.savefig(f"{OUT}/{name}.pdf")
    plt.close(fig)


def fig_pth(th):
    fig, axes = plt.subplots(1, 2, figsize=(M.COL, 2.1), gridspec_kw=dict(width_ratios=[1.0, 1.0]))
    ax = axes[0]
    fixed = [k for k in KEYS if k != "d"]
    xs, ys, es = [], [], []
    for k in fixed:
        f = th[(th.key == k) & (th.kind == "fss")]
        if f.empty:
            continue
        xs.append(int(k)); ys.append(100 * float(f.p.iloc[0])); es.append(100 * float(f.err.iloc[0]))
    shot = th[(th.key == "d") & (th.kind == "fss")]
    rnd = th[(th.key == "d_round") & (th.kind == "fss")]
    if not shot.empty and not rnd.empty:
        y0, y1 = 100 * float(shot.p.iloc[0]), 100 * float(rnd.p.iloc[0])
        ax.axhspan(y0, y1, color="#7C3AED", alpha=0.18, lw=0)
        ax.axhline(y0, color="#5B21B6", lw=0.8, ls=":")
        ax.axhline(y1, color="#5B21B6", lw=0.8, ls="--")
        # a small legend in the empty space above the curve names the two lines
        from matplotlib.lines import Line2D
        hd = [Line2D([], [], color="#5B21B6", lw=0.8, ls="--", label=r"$r{=}d$, round"),
              Line2D([], [], color="#5B21B6", lw=0.8, ls=":", label=r"$r{=}d$, shot")]
        ax.legend(handles=hd, loc="upper right", fontsize=5.8, handlelength=1.4,
                  borderaxespad=0.3, handletextpad=0.4)
    ax.errorbar(xs, ys, yerr=es, color="#E11D48", marker="o", ms=3.2, lw=1.0, elinewidth=0.7, capsize=0, zorder=3)
    ax.set_xlabel(r"rounds $r$")
    ax.set_ylabel(r"threshold $p_{\mathrm{th}}$ (\%)")
    ax.set_xticks([1, 2, 4, 6, 8, 10])
    ax.set_xlim(0.4, 10.7)
    ax.set_ylim(0, 2.45)
    M.panel_label(ax, "(a)")
    ax = axes[1]
    show = [("1", r"$r=1$"), ("2", r"$r=2$"), ("4", r"$r=4$"), ("8", r"$r=8$"), ("d_round", r"$r=d$, round"),
            ("d", r"$r=d$, shot")]
    cols = {"1": "#F97316", "2": "#E11D48", "4": "#D946EF", "8": "#7C3AED", "d_round": "#4C1D95", "d": "#4C1D95"}
    for k, lab in show:
        g = th[(th.key == k) & (th.kind == "triple")].dropna(subset=["p"])
        if g.empty:
            continue
        dm = g.d1 + 2
        ax.errorbar(dm, 100 * g.p, yerr=100 * g.err.fillna(0), color=cols[k], marker="o", ms=2.4, lw=0.9,
                    ls="--" if k == "d_round" else "-", mfc="white" if k == "d_round" else cols[k],
                    elinewidth=0.5, capsize=0, label=lab)
    ax.set_xlabel(r"middle distance $d$")
    ax.set_ylabel(r"crossing $p_c$ (\%)")
    ax.set_yscale("log")
    ax.set_yticks([0.4, 0.6, 1, 1.5, 2])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(0.35, 2.3)
    ax.set_xticks([5, 9, 13, 17])
    M.panel_label(ax, "(b)")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=6, handlelength=1.6, labelspacing=0.3)
    fig.tight_layout(w_pad=0.5)
    fig.savefig(f"{OUT}/fig_rounds_pth.pdf")
    plt.close(fig)


def table(th):
    lines = [r"\begin{table}[t]",
             r"  \caption{Threshold of MWPM as a function of the number of rounds $r$ under SD6 noise, from the fit of \autoref{eq:fss} to $d=11$ to $19$. The uncertainty combines the bootstrap error with half the spread of fits with other windows and sets of distances. For $r=d$ the fit uses either the logical error rate per shot, $\pL$, or the logical error rate per round, $\epsilon_L$ of \autoref{eq:per_round}. The last column gives the exponent $\nu$ with its bootstrap error.}",
             r"  \label{tab:rounds}",
             r"  \begin{ruledtabular}",
             r"  \begin{tabular}{lcc}",
             r"    $r$ & $p_{\mathrm{th}}$ (\%) & $\nu$ \\",
             r"    \hline"]
    labels = {"d": r"$d$, per shot", "d_round": r"$d$, per round"}
    for k in KEYS + ["d_round"]:
        f = th[(th.key == k) & (th.kind == "fss")]
        if f.empty:
            continue
        pth, err = 100 * float(f.p.iloc[0]), 100 * float(f.err.iloc[0])
        nu, nue = float(f.nu.iloc[0]), float(f.nu_err.iloc[0])
        dig = 3
        e = max(1, int(round(err * 10 ** dig)))
        ne = max(1, int(round(nue * 10)))
        lab = labels.get(k, k)
        lines.append(f"    {lab} & ${pth:.{dig}f}({e})$ & ${nu:.1f}({ne})$ \\\\")
    lines += [r"  \end{tabular}", r"  \end{ruledtabular}", r"\end{table}"]
    (TABLES / "tab_rounds.tex").write_text(caption_below("\n".join(lines)) + "\n")


if __name__ == "__main__":
    df = data()
    th = thresholds()
    fig_curves(df, th, zoom=True, name="fig_rounds_curves")
    fig_curves(df, th, zoom=False, name="fig_rounds_full")
    fig_pth(th)
    table(th)
    print("done")
