"""Figures and tables of the Quantinuum section.

The Helios and H2 models are one-parameter families: p is the two-qubit depolarizing
probability and every other rate keeps its data-sheet ratio to p
(CircuitNoise.trapped_ion). The machines sit at p0 = 1.0e-3 (Helios) and 1.875e-3 (H2).
Older runs were labelled by the scale lambda = p / p0 and are converted here.
"""
import os
import sys

import numpy as np
import pandas as pd
import sinter

sys.path.insert(0, os.path.dirname(__file__))
import make_plots as M  # noqa: E402
import crossings as X  # noqa: E402
from common import DATA, out_dir  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from latex_util import caption_below  # noqa: E402

HD = os.environ.get("HELIOS_DATA", str(DATA / "helios"))
OUT = M.OUT
TAB = [str(out_dir("tables"))]
P0 = {"helios": 1.0e-3, "h2": 1.875e-3, "helios_cb": 1.0e-3}
MACH_COL = {"helios": "#F97316", "h2": "#7C3AED"}
SPAM = {"helios": 5e-4, "h2": 1.5e-3}


def load_mwpm():
    rows = []
    for s in sinter.read_stats_from_csv_files(f"{HD}/helios_mwpm_sinter.csv"):
        m = s.json_metadata
        rows.append(dict(noise=m["machine"], value=float(m["lam"]) * P0[m["machine"]], d=int(m["d"]),
                         rounds=int(m["r"]), decoder="mwpm", shots=s.shots, errors=s.errors))
    return pd.DataFrame(rows)


def load_hcm():
    parts = [pd.read_csv(f"{HD}/{f}") for f in ("helios_hcm.csv", "helios_hcm_rounds.csv")]
    df = pd.concat(parts, ignore_index=True)
    df["value"] = df.value * df.noise.map(P0)
    df["decoder"] = "hcm"
    return df[["noise", "value", "d", "rounds", "decoder", "shots", "errors"]]


def load_p():
    df = pd.read_csv(f"{HD}/ion_p_sweeps.csv")
    df["noise"] = df.noise.str.replace("_p", "", regex=False)
    df["decoder"] = df.decoder.replace({"corr_gauge": "hcm"})
    return df[["noise", "value", "d", "rounds", "decoder", "shots", "errors"]]


def load_all():
    df = pd.concat([load_mwpm(), load_hcm(), load_p()], ignore_index=True)
    df["value"] = df.value.round(9)
    df = df.groupby(["noise", "value", "d", "rounds", "decoder"], as_index=False)[["shots", "errors"]].sum()
    df["p_L"] = df.errors / df.shots
    return df


def err(df):
    p = np.maximum(df.p_L.values, 1e-12)
    return np.sqrt(p * (1 - p) / df.shots.values)


def curve(ax, s, d, ls, filled, label=None, minerr=5, x="value"):
    s = s[(s.d == d) & (s.errors >= minerr)].sort_values(x)
    if s.empty:
        return
    c = M.RAMP[d]
    ax.errorbar(s[x], s.p_L, yerr=err(s), color=c, ls=ls, marker=M.MARK[d], ms=3.0,
                mfc=c if filled else "white", mec=c, lw=0.9, elinewidth=0.6, capsize=0, label=label)


def pair_crossings(df, noise, decoder):
    """Crossing points of consecutive distances for r = d."""
    g = df[(df.noise == noise) & (df.decoder == decoder) & (df.rounds == df.d)]
    out = {}
    for d1, d2 in ((3, 5), (5, 7), (7, 9)):
        a = g[g.d == d1].set_index("value").p_L
        b = g[g.d == d2].set_index("value").p_L
        v = sorted(set(a.index) & set(b.index))
        guess = np.nan
        for i in range(len(v) - 1):
            if a[v[i]] >= b[v[i]] and a[v[i + 1]] < b[v[i + 1]]:
                guess = 0.5 * (v[i] + v[i + 1])
                break
        if not np.isfinite(guess):
            out[(d1, d2)] = (np.nan, np.nan)
            continue
        w = X.window_around(g, d1, d2, guess, rel=0.15, min_points=3)
        out[(d1, d2)] = X.crossing(g, d1, d2, w, deg=1, boot=300)
    return out


def fig_quantinuum(df):
    fig, axes = plt.subplots(2, 3, figsize=(M.FULL, 4.25))
    for row, mach in enumerate(("helios", "h2")):
        name = "Helios" if mach == "helios" else "H2"
        p0 = P0[mach]
        # one round
        ax = axes[row, 0]
        s = df[(df.noise == mach) & (df.rounds == 1) & (df.decoder == "mwpm")]
        for d in (3, 5, 7):
            curve(ax, s, d, "-", True, label=f"$d={d}$")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(rf"{name}, $r=1$", loc="left", color=M.INK)
        # r = d around the crossings
        bx = axes[row, 1]
        s = df[(df.noise == mach) & (df.rounds == df.d) & (df.decoder == "mwpm")]
        h = df[(df.noise == mach) & (df.rounds == df.d) & (df.decoder == "hcm")]
        lo, hi = (0.6e-3, 1.6e-3) if mach == "helios" else (1.2e-3, 4.4e-3)
        for d in (3, 5, 7, 9):
            curve(bx, s[(s.value >= lo) & (s.value <= hi)], d, "--", False)
            curve(bx, h[(h.value >= lo) & (h.value <= hi)], d, "-", True, label=f"$d={d}$")
        bx.set_xscale("log"); bx.set_yscale("log")
        bx.set_xlim(lo * 0.95, hi * 1.05)
        bx.set_title(rf"{name}, $r=d$", loc="left", color=M.INK)
        for axx in (ax, bx):
            axx.axvline(p0, color=M.GUIDE, lw=0.9, ls=":")
            axx.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v*1e3:g}"))
            axx.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            axx.set_xlabel(r"two-qubit error $p\;(\times10^{-3})$")
        ticks_a = [0.25e-3, 0.5e-3, 1e-3, 2e-3, 3e-3] if mach == "helios" else [0.5e-3, 1e-3, 2e-3, 4e-3]
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks_a))
        ticks_b = [0.6e-3, 0.8e-3, 1e-3, 1.2e-3, 1.5e-3] if mach == "helios" else [1.5e-3, 2e-3, 3e-3, 4e-3]
        bx.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks_b))
        ax.set_ylabel(r"logical error rate $p_L$")
    ax = axes[0, 2]
    s = df[(df.noise == "helios") & np.isclose(df.value, P0["helios"]) & (df.decoder == "mwpm")]
    h = df[(df.noise == "helios") & np.isclose(df.value, P0["helios"]) & (df.decoder == "hcm")]
    for d in (3, 5, 7):
        for q, ls, filled in ((s, "--", False), (h, "-", True)):
            q = q[(q.d == d) & (q.errors >= 5) & (q.rounds <= 7)].sort_values("rounds")
            if q.empty:
                continue
            ax.errorbar(q.rounds, q.p_L, yerr=err(q), color=M.RAMP[d], ls=ls, marker=M.MARK[d], ms=3.0,
                        mfc=M.RAMP[d] if filled else "white", mec=M.RAMP[d], lw=0.9, elinewidth=0.6,
                        capsize=0, label=f"$d={d}$" if filled else None)
    ax.set_yscale("log")
    ax.set_xlabel(r"rounds $r$")
    ax.set_xticks(range(1, 8))
    ax.set_title(r"Helios, $p=p_0$", loc="left", color=M.INK)
    ax.axhline(SPAM["helios"], color=M.INK2, lw=0.6, ls=":")
    ax.set_ylim(2.5e-6, 1.5e-2)
    hd = [Line2D([], [], color=M.INK, ls="--", marker="o", mfc="white", ms=3, lw=0.9, label="MWPM"),
          Line2D([], [], color=M.INK, ls="-", marker="o", ms=3, lw=0.9, label="HCM"),
          Line2D([], [], color=M.INK2, ls=":", lw=0.6, label="physical SPAM")]
    hdd, _ = ax.get_legend_handles_labels()
    ax.legend(handles=hdd + hd, loc="lower right", ncol=2, fontsize=6.0, columnspacing=0.8,
              handlelength=1.5, borderaxespad=0.2)
    # crossing points of consecutive distances
    cx = axes[1, 2]
    xs = np.arange(3)
    res = {}
    for mach in ("helios", "h2"):
        for tag, ls, filled in (("", "-", True), ("_noxt", "--", False)):
            cr = pair_crossings(df, mach + tag, "mwpm")
            res[mach + tag] = cr
            y = np.array([cr[k][0] for k in ((3, 5), (5, 7), (7, 9))])
            e = np.array([cr[k][1] for k in ((3, 5), (5, 7), (7, 9))])
            ok = np.isfinite(y)
            c = MACH_COL[mach]
            cx.errorbar(xs[ok], 1e3 * y[ok], yerr=1e3 * np.nan_to_num(e[ok]), color=c, ls=ls, marker="o",
                        ms=3.2, mfc=c if filled else "white", mec=c, lw=0.9, elinewidth=0.6, capsize=0)
        cx.axhline(1e3 * P0[mach], color=MACH_COL[mach], lw=0.6, ls=":")
    cx.set_xticks(xs)
    cx.set_xticklabels([r"$3/5$", r"$5/7$", r"$7/9$"])
    cx.set_xlim(-0.3, 2.3)
    cx.set_xlabel(r"distances $d/d'$")
    cx.set_ylabel(r"crossing $p_c\;(\times10^{-3})$")
    cx.set_title(r"Crossing points, MWPM", loc="left", color=M.INK)
    hc = [Line2D([], [], color=MACH_COL["helios"], marker="o", ms=3, lw=0.9, label="Helios"),
          Line2D([], [], color=MACH_COL["h2"], marker="o", ms=3, lw=0.9, label="H2"),
          Line2D([], [], color=M.INK, ls="-", marker="o", ms=3, lw=0.9, label="crosstalk"),
          Line2D([], [], color=M.INK, ls="--", marker="o", mfc="white", ms=3, lw=0.9, label="no crosstalk")]
    cx.set_ylim(0.55, 6.4)
    cx.legend(handles=hc, loc="upper center", fontsize=5.8, ncol=2, columnspacing=1.0, handlelength=1.6,
              borderaxespad=0.25)
    axes[0, 0].legend(loc="lower right", fontsize=6)
    axes[0, 1].legend(loc="lower right", fontsize=6)
    for ax, t in zip(axes.flat, "abcdef"):
        M.panel_label(ax, f"({t})")
    fig.tight_layout(w_pad=0.6, h_pad=0.8)
    fig.savefig(f"{OUT}/fig_quantinuum.pdf")
    plt.close(fig)
    return res


def fmt(x, digits=2):
    if not np.isfinite(x) or x <= 0:
        return "--"
    e = int(np.floor(np.log10(x)))
    m = x / 10 ** e
    if round(m, digits - 1) >= 10:
        m /= 10
        e += 1
    return rf"${m:.{digits-1}f}\times10^{{{e}}}$"


def per_round_fits(df):
    """Linear fit p_L = a + b (r - 1) for r = 1..7 at the device point, weighted by binomial errors."""
    out = {}
    for d in (3, 5, 7):
        q = df[(df.d == d) & (df.rounds <= 7) & (df.errors >= 5)].sort_values("rounds")
        if len(q) < 3:
            continue
        x = q.rounds.values - 1.0
        y = q.p_L.values
        w = 1.0 / err(q) ** 2
        A = np.vstack([np.ones_like(x), x]).T
        cov = np.linalg.inv(A.T @ (A * w[:, None]))
        a, b = cov @ (A.T @ (w * y))
        out[d] = (a, b, np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1]))
    return out


def tables(df):
    def val(noise, dec, d, r):
        q = df[(df.noise == noise) & (df.decoder == dec) & (df.d == d) & (df.rounds == r)
               & np.isclose(df.value, P0[noise])]
        if q.empty or q.errors.iloc[0] < 3:
            return np.nan
        return float(q.p_L.iloc[0])

    lines = [r"\begin{table*}[t]",
             r"  \caption{Logical error rate of $\plusL$ under the Quantinuum models at the device point $p=p_0$. Helios, CB uses the two-qubit Pauli error rates measured by cycle benchmarking~\cite{ransford2026} instead of depolarizing gate noise. The physical SPAM error is $5\times10^{-4}$ for Helios and $1.5\times10^{-3}$ for H2. Helios holds the protocol up to $d=5$ and H2 up to $d=3$.}",
             r"  \label{tab:quantinuum}",
             r"  \setlength{\tabcolsep}{4pt}",
             r"  \begin{ruledtabular}",
             r"  \begin{tabular}{llcccccc}",
             r"    & & \multicolumn{3}{c}{$r=1$} & \multicolumn{3}{c}{$r=d$} \\",
             r"    \cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             r"    Model & Decoder & $d=3$ & $d=5$ & $d=7$ & $d=3$ & $d=5$ & $d=7$ \\",
             r"    \hline"]
    for noise, name in (("helios", "Helios"), ("helios_cb", "Helios, CB"), ("h2", "H2")):
        for dec, dname in (("hcm", "HCM"), ("mwpm", "MWPM")):
            cells = []
            for r in ("1", "d"):
                for d in (3, 5, 7):
                    cells.append(fmt(val(noise, dec, d, 1 if r == "1" else d)))
            lead = name if dec == "hcm" else ""
            lines.append(f"    {lead} & {dname} & " + " & ".join(cells) + r" \\")
    lines += [r"  \end{tabular}", r"  \end{ruledtabular}", r"\end{table*}", ""]
    txt = "\n".join(lines)
    for t in TAB:
        with open(f"{t}/tab_quantinuum.tex", "w") as fh:
            fh.write(caption_below(txt))
    return txt


def budget_table():
    path = f"{HD}/helios_budget.csv"
    if not os.path.exists(path):
        return None
    b = pd.read_csv(path, header=None, names=["d", "r", "name", "shots", "errors", "rate", "stderr"])
    b = b.groupby(["d", "r", "name"], as_index=False)[["shots", "errors"]].sum()
    b["rate"] = b.errors / b.shots
    names = [("full", "Full model"), ("no_gates", "No gate errors"), ("no_memory", "No memory errors"),
             ("no_crosstalk", "No crosstalk"), ("no_spam", "No SPAM errors"),
             ("xtalk_per_step", "Crosstalk per step")]
    cols = [(3, 1), (5, 1), (3, 3), (5, 5), (7, 7)]
    lines = [r"\begin{table*}[t]",
             r"  \caption{Error budget of the Helios model at the device point $p=p_0$, decoded with HCM. Each row removes one component of the model. The last row applies the crosstalk once per reset step and once per measurement step instead of once per measured ancilla.}",
             r"  \label{tab:budget}",
             r"  \begin{ruledtabular}",
             r"  \begin{tabular}{lccccc}",
             r"    & \multicolumn{2}{c}{$r=1$} & \multicolumn{3}{c}{$r=d$} \\",
             r"    \cmidrule(lr){2-3}\cmidrule(lr){4-6}",
             r"    & $d=3$ & $d=5$ & $d=3$ & $d=5$ & $d=7$ \\",
             r"    \hline"]
    for key, lab in names:
        cells = []
        for d, rr in cols:
            q = b[(b.d == d) & (b.r == rr) & (b.name == key)]
            cells.append(fmt(float(q.rate.iloc[0])) if len(q) and q.errors.iloc[0] >= 10 else "--")
        lines.append(f"    {lab} & " + " & ".join(cells) + r" \\")
    lines += [r"  \end{tabular}", r"  \end{ruledtabular}", r"\end{table*}", ""]
    txt = "\n".join(lines)
    for t in TAB:
        with open(f"{t}/tab_budget.tex", "w") as fh:
            fh.write(caption_below(txt))
    return b


if __name__ == "__main__":
    df = load_all()
    res = fig_quantinuum(df)
    for k, v in res.items():
        print(k, {kk: (round(1e3 * x, 3), round(1e3 * e, 3)) for kk, (x, e) in v.items()})
    print(tables(df))
    for name in ("mwpm", "hcm"):
        q = df[(df.noise == "helios") & np.isclose(df.value, P0["helios"]) & (df.decoder == name)]
        for d, (a, b, ea, eb) in per_round_fits(q).items():
            print(f"{name} d={d}: intercept {a:.3e}({ea:.1e}) per round {b:.3e}({eb:.1e})")
    budget_table()
