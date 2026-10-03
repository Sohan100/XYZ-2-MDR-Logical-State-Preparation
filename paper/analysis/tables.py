"""LaTeX and Markdown tables of the paper.

Usage: python paper/analysis/tables.py

Reads docs/data and writes tab_*.tex and md_*.md to paper/build/tables.
"""
import pandas as pd

from common import DATA, NOISE_ORDER, THR, out_dir, read
from latex_util import caption_below  # noqa: E402

TH = read(THR / "fits.csv")
NOISE_NAME = {"sd6": "SD6", "si1000": "SI1000", "biased10": r"Biased, $\eta=10$",
              "biased100": r"Biased, $\eta=100$", "purez": r"Pure $Z$", "em3": "EM3"}
DECS = [("mwpm", "MWPM"), ("corr_links", "HCM, links"), ("corr_gauge", "HCM"), ("bm", "BM")]


def fmt_th(noise, dec):
    r = TH[(TH.noise == noise) & (TH.decoder == dec)]
    if r.empty:
        return ""
    r = r.iloc[0]
    val, unc = 100 * r.pth, 100 * r.unc
    digits = max(1, int(round(100 * unc)))  # in units of 0.01 %
    return f"${val:.2f}({digits})$"


def threshold_table():
    lines = [r"\begin{table}[t]",
             r"  \caption{Thresholds for $r=d$ rounds in percent of the physical error rate $p$. The number in parentheses is the uncertainty in the last digit. MWPM and HCM come from finite-size scaling fits with $d=5$, $7$ and $9$. BM is belief-matching, for which we quote the crossing point of $d=5$ and $d=7$ (\autoref{fig:bm}). Under EM3 the plaquettes are measured with pair measurements (\autoref{sec:em3}).}",
             r"  \label{tab:thresholds}",
             r"  \setlength{\tabcolsep}{3pt}",
             r"  \begin{ruledtabular}",
             r"  \begin{tabular}{l" + "c" * len(DECS) + "}",
             "    Noise model & " + " & ".join(n for _, n in DECS) + r" \\",
             r"    \hline"]
    for noise in NOISE_ORDER:
        lines.append(f"    {NOISE_NAME[noise]} & " + " & ".join(fmt_th(noise, d) for d, _ in DECS) + r" \\")
    lines += [r"  \end{tabular}", r"  \end{ruledtabular}", r"\end{table}"]
    return "\n".join(lines)


def fit_table():
    fits = TH[TH.decoder != "bm"]
    lines = [r"\begin{table}[t]",
             r"  \caption{Finite-size scaling fits of \autoref{eq:fss} with $d=5$, $7$ and $9$. $\nu$ is the fitted exponent and $\chi^2_r$ the reduced chi-square. Each fit uses between "
             + f"{int(fits.n.min())} and {int(fits.n.max())}" + r" data points.}",
             r"  \label{tab:fits}",
             r"  \setlength{\tabcolsep}{4pt}",
             r"  \begin{ruledtabular}",
             r"  \begin{tabular}{lcccccc}",
             r"    & \multicolumn{2}{c}{MWPM} & \multicolumn{2}{c}{HCM, links} & \multicolumn{2}{c}{HCM} \\",
             r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r"    Noise model & $\nu$ & $\chi^2_r$ & $\nu$ & $\chi^2_r$ & $\nu$ & $\chi^2_r$ \\",
             r"    \hline"]
    for noise in NOISE_ORDER:
        cells = []
        for dec in ["mwpm", "corr_links", "corr_gauge"]:
            r = fits[(fits.noise == noise) & (fits.decoder == dec)]
            if r.empty:
                cells += ["", ""]
                continue
            r = r.iloc[0]
            assert r.dists == "5,7,9", (noise, dec, r.dists)
            cells += [f"${r.nu:.2f}$", f"${r.chi2r:.1f}$"]
        lines.append(f"    {NOISE_NAME[noise]} & " + " & ".join(cells) + r" \\")
    lines += [r"  \end{tabular}", r"  \end{ruledtabular}", r"\end{table}"]
    return "\n".join(lines)


def fd_table():
    fd = read(DATA / "fault_distance.csv")
    lines = [r"\begin{table}[t]",
             r"  \caption{Circuit-level fault distance. For each instance CP-SAT proved that no undetectable logical error with fewer faults exists and found one with the listed number of faults. For $d=7$ with seven rounds the lower bound comes from the linear program of the text and the upper bound from the search of Stim. Mechanisms counts the distinct fault mechanisms of the detector error model. The $\ket{+}^{\otimes n}$ start uses all generator detectors.}",
             r"  \label{tab:fdapp}",
             r"  \begin{ruledtabular}",
             r"  \begin{tabular}{cccccc}",
             r"    $d$ & $r$ & Start & Readout & Mechanisms & Fault distance \\",
             r"    \hline"]
    for r in fd.itertuples():
        if pd.isna(r.lower_bound) or pd.isna(r.upper_bound):
            continue
        start = "frame" if r.init == "frame" else r"$\ket{+}^{\otimes n}$"
        lo, hi = int(r.lower_bound), int(r.upper_bound)
        val = f"{lo}" if lo == hi else f"{lo} or {hi}"
        lines.append(f"    {r.d} & {r.rounds} & {start} & {r.final} & {r.mechanisms} & {val} \\\\")
    lines += [r"  \end{tabular}", r"  \end{ruledtabular}", r"\end{table}"]
    return "\n".join(lines)


def md_table():
    names = {"sd6": "SD6", "si1000": "SI1000", "biased10": "biased, eta=10", "biased100": "biased, eta=100",
             "purez": "pure Z", "em3": "EM3"}
    out = ["| noise | MWPM | HCM, links | HCM | belief-matching |", "|---|---|---|---|---|"]
    for noise in NOISE_ORDER:
        cells = []
        for dec, _ in DECS:
            r = TH[(TH.noise == noise) & (TH.decoder == dec)]
            if r.empty:
                cells.append("")
                continue
            r = r.iloc[0]
            cells.append(f"{100 * r.pth:.2f}% ± {100 * max(r.unc, 1e-4):.2f}%")
        out.append(f"| {names[noise]} | " + " | ".join(cells) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    out = out_dir("tables")
    # tab_quantinuum is written by helios_plots.py
    tabs = {"tab_thresholds": threshold_table(), "tab_fits": fit_table(), "tab_fd": fd_table()}
    for name, tex in tabs.items():
        (out / f"{name}.tex").write_text(caption_below(tex) + "\n")
    (out / "md_thresholds.md").write_text(md_table() + "\n")
    print("tables in", out)
