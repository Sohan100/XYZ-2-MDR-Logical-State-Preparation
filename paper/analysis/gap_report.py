"""
gap_report.py
----------------------------------------------------------------------------
Gap diagnostic (scripts/campaign_gap.py): how much of the threshold gap between XYZ^2 and the XZZX code under SD6
each variant of the noise model closes.

    python paper/analysis/gap_report.py --points docs/data/campaign/gap/points.csv \
        --campaign docs/data/campaign/thresholds.csv --out docs/data/campaign/gap

Values follow the paper and competitor_report: the both-bases schedule of XYZ^2 (Logical X and Y memories), the
best of MWPM, correlated matching and belief-matching in each basis, then the weaker basis; MWPM alone in its
weaker basis as well; r = d; finite-size fits over d = 11 to 21 that do not drift (campaign_report rules).
- sd6, sd6_il, sd6_lr2, sd6_lr3 and hyb of XYZ^2 and the surface codes under sd6 come from the campaign's
  thresholds.csv (the fair rerun);
- sd6_noidle and sd6_il_noidle of XYZ^2 and the surface codes under sd6_noidle are fitted here.
The fraction of the gap that a variant closes is (T_variant - T_sd6) / (T_ref - T_sd6), with T_ref the XZZX
threshold under sd6 or under sd6_noidle; its error propagates the errors of the three thresholds (first order,
independent).

Writes gap.csv (every value), fractions.csv, gap.md and series.csv (the new fits).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign_report as R  # noqa: E402
import competitor_report as CR  # noqa: E402

XYZ2_VARIANTS = ["sd6", "sd6_il", "sd6_lr2", "sd6_lr3", "hyb", "sd6_noidle", "sd6_il_noidle"]
VARIANT_LAB = {"sd6": "SD6", "sd6_il": "noiseless links", "sd6_lr2": "links twice", "sd6_lr3": "links three times",
               "hyb": "hybrid", "sd6_noidle": "no idle errors", "sd6_il_noidle": "noiseless links, no idle errors"}
XYZ2_DECODERS = ["mwpm", "corr_gauge", "bm"]
SURFACE_DECODERS = ["mwpm", "corr", "bm"]


def usable(th: pd.DataFrame) -> pd.DataFrame:
    th = th.copy()
    th["rounds"] = th.rounds.astype(str)
    th = th[th.rounds == "d"]
    return CR.usable(th, CR.MIN_DMAX)


def xyz2_value(th: pd.DataFrame, model: str) -> dict:
    """Best decoder per basis then the weaker basis, and MWPM alone, of the both-bases schedule."""
    best, mw = {}, {}
    for L in ("X", "Y"):
        g = th[(th.noise == f"{model}@both-{L}") & th.decoder.isin(XYZ2_DECODERS)]
        if len(g):
            best[L] = g.sort_values("pth").iloc[-1]
        m = g[g.decoder == "mwpm"]
        if len(m):
            mw[L] = m.iloc[0]
    out = dict(code="xyz2", model=model)
    if len(best) == 2:
        L, r = min(best.items(), key=lambda kv: kv[1].pth)
        out.update(pth=float(r.pth), err=float(r.err), basis=L, decoder=r.decoder,
                   other=float(max(v.pth for v in best.values())))
    if len(mw) == 2:
        L, r = min(mw.items(), key=lambda kv: kv[1].pth)
        out.update(mwpm=float(r.pth), mwpm_err=float(r.err), mwpm_basis=L)
    return out


def surface_value(th: pd.DataFrame, code: str, model: str) -> dict:
    best, mw = {}, {}
    for b in ("X", "Z"):
        g = th[(th.noise == model) & th.decoder.isin([f"{code}-{b}:{d}" for d in SURFACE_DECODERS])]
        if len(g):
            best[b] = g.sort_values("pth").iloc[-1]
        m = g[g.decoder == f"{code}-{b}:mwpm"]
        if len(m):
            mw[b] = m.iloc[0]
    out = dict(code=code, model=model)
    if len(best) == 2:
        b, r = min(best.items(), key=lambda kv: kv[1].pth)
        out.update(pth=float(r.pth), err=float(r.err), basis=b, decoder=r.decoder.split(":")[1])
    if len(mw) == 2:
        b, r = min(mw.items(), key=lambda kv: kv[1].pth)
        out.update(mwpm=float(r.pth), mwpm_err=float(r.err), mwpm_basis=b)
    return out


def fraction(tv, ev, t0, e0, tr, er):
    """(T_v - T_0) / (T_ref - T_0) and its first-order error."""
    D = tr - t0
    f = (tv - t0) / D
    s = math.sqrt((ev / D) ** 2 + (e0 * (tr - tv) / D ** 2) ** 2 + (er * (tv - t0) / D ** 2) ** 2)
    return f, s


def ratio(a, ea, b, eb):
    q = a / b
    return q, q * math.hypot(ea / a, eb / b)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", required=True, help="merged points of the gap runs")
    ap.add_argument("--campaign", required=True, help="thresholds.csv of the campaign (fair rerun)")
    ap.add_argument("--competitors", default=None, help="competitors.csv of the campaign (surface codes)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = R.load(args.points, pool=False)
    df = df[df.rkey == "d"]
    new, cr = R.thresholds(df, workers=args.workers)
    new["rounds"] = new.rounds.astype(str)
    new.to_csv(out / "series.csv", index=False)
    cr.to_csv(out / "crossings.csv", index=False)
    old = pd.read_csv(args.campaign)
    comp = pd.read_csv(args.competitors or str(Path(args.campaign).with_name("competitors.csv")))
    th = usable(pd.concat([old, comp, new], ignore_index=True))
    rows = [xyz2_value(th, m) for m in XYZ2_VARIANTS]
    rows += [surface_value(th, c, m) for c in ("xzzx", "css") for m in ("sd6", "sd6_noidle")]
    vals = pd.DataFrame(rows)
    vals.to_csv(out / "gap.csv", index=False)
    get = {(r.code, r.model): r for r in vals.itertuples()}
    x0 = get[("xyz2", "sd6")]
    frs = []
    for refm in ("sd6", "sd6_noidle"):
        ref = get.get(("xzzx", refm))
        if ref is None or not np.isfinite(getattr(ref, "pth", np.nan)):
            continue
        for m in XYZ2_VARIANTS[1:]:
            v = get[("xyz2", m)]
            row = dict(variant=m, reference=f"xzzx {refm}")
            if np.isfinite(getattr(v, "pth", np.nan)):
                f, s = fraction(v.pth, v.err, x0.pth, x0.err, ref.pth, ref.err)
                q, sq = ratio(v.pth, v.err, ref.pth, ref.err)
                row.update(fraction=f, fraction_err=s, ratio=q, ratio_err=sq)
            if np.isfinite(getattr(v, "mwpm", np.nan)) and np.isfinite(getattr(ref, "mwpm", np.nan)):
                f, s = fraction(v.mwpm, v.mwpm_err, x0.mwpm, x0.mwpm_err, ref.mwpm, ref.mwpm_err)
                q, sq = ratio(v.mwpm, v.mwpm_err, ref.mwpm, ref.mwpm_err)
                row.update(fraction_mwpm=f, fraction_mwpm_err=s, ratio_mwpm=q, ratio_mwpm_err=sq)
            frs.append(row)
    frs = pd.DataFrame(frs)
    frs.to_csv(out / "fractions.csv", index=False)
    f = R.fmt
    L = ["# Gap diagnostic: XYZ^2 against XZZX under SD6 and its variants", "",
         "Memory thresholds in %, r = d, d = 11 to 21: the best of MWPM, correlated matching and belief-matching "
         "in each basis, then the weaker basis (XYZ^2: both-bases schedule), and MWPM alone in its weaker basis.",
         "", "| code | noise model | best decoder | basis, decoder | MWPM alone |", "|---|---|---|---|---|"]
    for r in vals.itertuples():
        lab = VARIANT_LAB.get(r.model, r.model) if r.code == "xyz2" else r.model
        L.append(f"| {CR.LABEL.get(r.code, r.code)} | {lab} | {f(getattr(r, 'pth', np.nan), getattr(r, 'err', np.nan))} | "
                 f"{getattr(r, 'basis', '')}, {getattr(r, 'decoder', '')} | "
                 f"{f(getattr(r, 'mwpm', np.nan), getattr(r, 'mwpm_err', np.nan))} |")
    L += ["", "Fraction of the gap closed, (T_variant - T_SD6) / (T_XZZX - T_SD6), with propagated errors:", "",
          "| variant | reference | best decoder | MWPM alone | XYZ^2 / reference (best) |", "|---|---|---|---|---|"]
    for r in frs.itertuples():
        a = f"{r.fraction:.2f}({max(1, round(100 * r.fraction_err))})" if pd.notna(getattr(r, "fraction", np.nan)) else "--"
        b = (f"{r.fraction_mwpm:.2f}({max(1, round(100 * r.fraction_mwpm_err))})"
             if pd.notna(getattr(r, "fraction_mwpm", np.nan)) else "--")
        c = f"{r.ratio:.2f}({max(1, round(100 * r.ratio_err))})" if pd.notna(getattr(r, "ratio", np.nan)) else "--"
        L.append(f"| {VARIANT_LAB[r.variant]} | {r.reference} | {a} | {b} | {c} |")
    (out / "gap.md").write_text("\n".join(L) + "\n")
    (out / "fig_gap_pgf.tex").write_text(pgf_figure(vals))
    print("\n".join(L))


PGF_ORDER = ["sd6", "sd6_il", "sd6_lr2", "sd6_lr3", "hyb", "sd6_noidle", "sd6_il_noidle"]
PGF_LAB = {"sd6": "SD6", "sd6_il": "noiseless links", "sd6_lr2": "links twice", "sd6_lr3": "links three times",
           "hyb": "hybrid", "sd6_noidle": "no idle errors", "sd6_il_noidle": "noiseless links, no idle errors"}


def pgf_figure(vals: pd.DataFrame) -> str:
    """The figure fig:gap as inline pgfplots: XYZ^2 memory thresholds of every variant (best of three decoders,
    weaker basis) with their errors, a dashed line for XZZX under SD6 and a dotted one for XZZX without idle
    errors."""
    get = {(r.code, r.model): r for r in vals.itertuples()}
    rows = [m for m in PGF_ORDER if ("xyz2", m) in get and np.isfinite(getattr(get[("xyz2", m)], "pth", np.nan))]
    coords = " ".join(f"({100 * get[('xyz2', m)].pth:.3f},{k}) +- ({100 * get[('xyz2', m)].err:.3f},0)"
                      for k, m in enumerate(rows))
    n = len(rows)
    xz, xn = get.get(("xzzx", "sd6")), get.get(("xzzx", "sd6_noidle"))
    lines = [r"\begin{figure}[t]", r"  \centering", r"  \begin{tikzpicture}",
             r"    \begin{axis}[width=0.72\columnwidth, height=5.4cm, xbar, bar width=7pt, y dir=reverse,",
             rf"      xmin=0.40, xmax=1.00, ymin=-0.6, ymax={n - 0.4}, xlabel={{memory threshold (\%)}},",
             rf"      ytick={{0,...,{n - 1}}}, yticklabels={{" + ",".join("{" + PGF_LAB[m] + "}" for m in rows) + "},",
             r"      yticklabel style={font=\footnotesize, text=ink}, xticklabel style={font=\footnotesize, text=ink},",
             r"      label style={font=\footnotesize, text=ink}, axis line style={stone}, tick style={stone},",
             r"      xmajorgrids, grid style={grid}, clip=false]",
             r"      \addplot[fill=xyzcode, draw=xyzcode, error bars/.cd, x dir=both, x explicit,",
             r"        error bar style={ink}] coordinates {" + coords + "};"]
    if xz is not None and np.isfinite(getattr(xz, "pth", np.nan)):
        lines.append(rf"      \draw[xzzxcode, dashed, thick] (axis cs:{100 * xz.pth:.3f},{n - 0.4}) -- "
                     rf"(axis cs:{100 * xz.pth:.3f},-0.6) node[above, font=\footnotesize, text=xzzxcode] {{XZZX}};")
    if xn is not None and np.isfinite(getattr(xn, "pth", np.nan)):
        lines.append(rf"      \draw[xzzxcode, dotted, thick] (axis cs:{100 * xn.pth:.3f},{n - 0.4}) -- "
                     rf"(axis cs:{100 * xn.pth:.3f},-0.6) node[above, font=\footnotesize, text=xzzxcode, align=center] "
                     r"{XZZX,\\no idle};")
    lines += [r"    \end{axis}", r"  \end{tikzpicture}", "  CAPTION", r"  \label{fig:gap}", r"\end{figure}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
