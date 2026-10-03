"""CFE decoder against belief-matching on the same shots, d = 3, 5, 7, r = d.

Reads the sweep files written by scripts/run_cfe_threshold_sweep.py (one .npz per noise,
distance and p, with the weight W and entropy S of every candidate and the decisions of
MWPM, HCM and belief-matching on the same shots). Draws fig_cfe_thr and writes
tab_cfe_thr with the ratio of failures of CFE and belief-matching per distance.
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import make_plots as M  # noqa: E402
from common import DATA, out_dir  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from latex_util import caption_below  # noqa: E402

CFE = os.environ.get("CFE_DATA", str(DATA / "cfe"))
TAB = [str(out_dir("tables"))]
KAPPA = 0.5
RNG = np.random.default_rng(7)
NOISES = (("sd6", "SD6"), ("b100", r"Biased, $\eta=100$"))


def load():
    shots = {}
    rows = []
    for f in sorted(glob.glob(f"{CFE}/*.npz")):
        m = re.match(r"(\w+?)_d(\d+)_p([\d.]+)_s(\d+)\.npz", os.path.basename(f))
        noise, d, p = m.group(1), int(m.group(2)), float(m.group(3))
        z = np.load(f)
        obs = z["obs"].astype(bool)
        F = (z["W"] - KAPPA * z["S"]).min(axis=2)
        W0 = z["W"].min(axis=2)
        fail = {"cfe": (F[:, 1] < F[:, 0]) != obs, "tc": (W0[:, 1] < W0[:, 0]) != obs}
        for k in ("mwpm", "hcm", "bm"):
            fail[k] = z[f"pred_{k}"].astype(bool) != obs
        shots[(noise, d, p)] = fail
        for k, v in fail.items():
            rows.append(dict(noise=noise, d=d, p=p, decoder=k, shots=len(obs), errors=int(v.sum())))
        rows.append(dict(noise=noise, d=d, p=p, decoder="time", shots=len(obs), errors=float(z["t_cfe"])))
    return pd.DataFrame(rows), shots


def ratios(shots, boot=2000):
    """Failures of CFE over failures of belief-matching on the same shots, pooled over p."""
    out = {}
    for noise, _ in NOISES:
        for d in (3, 5, 7):
            keys = [k for k in shots if k[0] == noise and k[1] == d]
            if not keys:
                continue
            c = np.concatenate([shots[k]["cfe"] for k in keys])
            b = np.concatenate([shots[k]["bm"] for k in keys])
            r0 = c.sum() / b.sum()
            idx = RNG.integers(0, len(c), size=(boot, len(c)))
            rb = c[idx].sum(axis=1) / np.maximum(b[idx].sum(axis=1), 1)
            out[(noise, d)] = (r0, float(np.std(rb)), int(c.sum()), int(b.sum()), len(c),
                               int((c & ~b).sum()), int((b & ~c).sum()))
    return out


def figure(df, rat):
    fig, axes = plt.subplots(1, 3, figsize=(M.FULL, 2.35), gridspec_kw=dict(width_ratios=[1, 1, 0.8]))
    for ax, (noise, title) in zip(axes, NOISES):
        s = df[df.noise == noise]
        for d in (3, 5, 7):
            for dec, ls, filled in (("bm", "--", False), ("cfe", "-", True)):
                q = s[(s.decoder == dec) & (s.d == d)].sort_values("p")
                if q.empty:
                    continue
                y = q.errors / q.shots
                e = np.sqrt(np.clip(y * (1 - y), 1e-12, None) / q.shots)
                ax.errorbar(q.p * 1e3, y, yerr=e, color=M.RAMP[d], ls=ls, marker=M.MARK[d], ms=3.0,
                            mfc=M.RAMP[d] if filled else "white", mec=M.RAMP[d], lw=0.9, elinewidth=0.6,
                            capsize=0, label=f"$d={d}$" if dec == "cfe" else None)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel(r"$p\;(\times10^{-3})$")
        ax.set_title(title, loc="left", color=M.INK)
    axes[0].set_ylabel(r"logical error rate $p_L$")
    axes[0].legend(loc="lower right", fontsize=6.5)
    h = [Line2D([], [], color=M.INK, ls="--", marker="o", mfc="white", ms=3, lw=0.9, label="BM"),
         Line2D([], [], color=M.INK, ls="-", marker="o", ms=3, lw=0.9, label="CFE")]
    axes[1].legend(handles=h, loc="lower right", fontsize=6.5)
    ax = axes[2]
    for noise, title in NOISES:
        ds = [d for d in (3, 5, 7) if (noise, d) in rat]
        y = [rat[(noise, d)][0] for d in ds]
        e = [rat[(noise, d)][1] for d in ds]
        c = M.DEC_COL["mwpm"] if noise == "sd6" else M.DEC_COL["bm"]
        ax.errorbar(ds, y, yerr=e, color=c, marker="o", ms=3.4, lw=1.0, elinewidth=0.7, capsize=0,
                    label="SD6" if noise == "sd6" else r"$\eta=100$")
    ax.axhline(1.0, color=M.INK2, lw=0.6, ls=":")
    ax.set_xticks([3, 5, 7])
    ax.set_xlim(2.5, 7.5)
    ax.set_ylim(0.55, 1.35)
    ax.set_xlabel(r"distance $d$")
    ax.set_ylabel(r"failures CFE / BM")
    ax.set_title("Same shots", loc="left", color=M.INK)
    ax.legend(loc="lower left", fontsize=6.5)
    for a, t in zip(axes, "abc"):
        M.panel_label(a, f"({t})")
    fig.tight_layout(w_pad=0.9)
    fig.savefig(f"{M.OUT}/fig_cfe_thr.pdf")
    plt.close(fig)


def table(rat):
    lines = [r"\begin{table}[t]",
             r"  \caption{Failures of the CFE decoder and of belief-matching on the same shots with $r=d$, summed over the values of $p$ in \autoref{fig:cfe_thr}. The last column is their ratio with its bootstrap error.}",
             r"  \label{tab:cfe_thr}",
             r"  \begin{ruledtabular}",
             r"  \begin{tabular}{llrrrc}",
             r"    Noise & $d$ & Shots & CFE & BM & CFE/BM \\",
             r"    \hline"]
    for noise, name in (("sd6", "SD6"), ("b100", r"$\eta=100$")):
        for k, d in enumerate((3, 5, 7)):
            if (noise, d) not in rat:
                continue
            r0, er, nc, nb, n, _, _ = rat[(noise, d)]
            lead = name if k == 0 else ""
            lines.append(f"    {lead} & {d} & {n} & {nc} & {nb} & ${r0:.2f}({max(int(round(100 * er)), 1)})$ \\\\")
    lines += [r"  \end{tabular}", r"  \end{ruledtabular}", r"\end{table}", ""]
    txt = "\n".join(lines)
    for t in TAB:
        with open(f"{t}/tab_cfe_thr.tex", "w") as fh:
            fh.write(caption_below(txt))
    return txt


if __name__ == "__main__":
    df, shots = load()
    dd = df[df.decoder != "time"]
    print(dd.pivot_table(index=["noise", "d", "p"], columns="decoder", values="errors").to_string())
    print(df[df.decoder == "time"].groupby(["noise", "d"]).errors.mean())
    rat = ratios(shots)
    for k, v in rat.items():
        print(k, "ratio %.3f(%.3f) cfe %d bm %d n %d cfe-only %d bm-only %d" % v)
    figure(dd, rat)
    print(table(rat))
    dd.to_csv(DATA / "cfe" / "cfe_summary.csv", index=False)
