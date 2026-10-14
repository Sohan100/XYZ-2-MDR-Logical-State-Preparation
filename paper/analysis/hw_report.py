"""
hw_report.py
----------------------------------------------------------------------------
Thresholds at the corners of the threshold surface of Hetenyi and Wootton (arXiv:2306.17786, Table I): XYZ^2
(both-bases schedule, Logical X and Y), the rotated CSS and XZZX surface codes and the honeycomb Floquet code
under hw_g (gates only), hw_g1q (gates with noisy single-qubit basis changes), hw_t (idling only, eta_T = 20)
and hw_rr (readout only), r = d (scripts/campaign_hw.py).

    python paper/analysis/hw_report.py --points docs/data/campaign/hw/points.csv --out docs/data/campaign/hw

Fits follow campaign_report and competitor_report: a finite-size-scaling fit over the largest distances
(d = 11 to 21, the honeycomb 12, 16 and 20) in a +-20% window around the last crossings, its error from the
bootstrap and the spread over windows and distance sets, the drift flag, and only finite-size fits that reach
d >= 11 count. A code's value is its weaker basis (each basis with its best decoder); MWPM alone is its weaker
basis under MWPM. The crossing of d = 7 and 9 (honeycomb: 4 and 8) is MWPM's.

Writes, in --out: thresholds.csv (every series), crossings.csv, summary.csv and thresholds.md (the
comparison), and fig_<model>.pdf and .png (p_L against p for every series).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign_report as R  # noqa: E402
import competitor_report as CR  # noqa: E402

MODELS = ["hw_g", "hw_g1q", "hw_t", "hw_rr"]
MODEL_LAB = {"hw_g": "gates only", "hw_g1q": "gates, noisy 1q basis changes", "hw_t": "idling only",
             "hw_rr": "readout only"}
CODES = ["css", "xzzx", "xyz2", "honeycomb"]
CODE_LAB = {"css": "rotated surface (CSS)", "xzzx": "XZZX surface", "xyz2": "XYZ^2", "honeycomb": "honeycomb Floquet"}
BASES = {"xyz2": ("X", "Y"), "css": ("X", "Z"), "xzzx": ("X", "Z"), "honeycomb": ("X", "Z")}
DECODERS = {"xyz2": ["mwpm", "corr_gauge", "bm"], "css": ["mwpm", "corr", "bm"], "xzzx": ["mwpm", "corr", "bm"],
            "honeycomb": ["mwpm", "corr", "bm"]}
DEC_LAB = {"mwpm": "MWPM", "corr": "corr", "corr_gauge": "corr", "bm": "BM"}
SMALL_PAIR = {"honeycomb": (4, 8)}
# Table I of Hetenyi and Wootton: intercepts (p_G, p_T, p_RR) in %, eta_T = 20, eta_G = 1; no gate value for hw_g1q
THEIRS = {"css": {"hw_g": 0.82, "hw_t": 3.94, "hw_rr": 14.5}, "xzzx": {"hw_g": 0.37, "hw_t": 15.1, "hw_rr": 12.9},
          "xyz2": {"hw_g": 0.465, "hw_t": 4.2, "hw_rr": 16.5}, "honeycomb": {"hw_g": 0.43, "hw_t": 1.16, "hw_rr": 0.99}}


def split(noise: str, decoder: str):
    """(model, code, basis, decoder) of a series."""
    if ":" in decoder:
        code, basis, dec = CR.split(decoder)
        return noise, code, basis, dec
    base, _, logical = CR.xyz2_label(noise)
    return base, "xyz2", logical, decoder


def label(series) -> pd.DataFrame:
    parts = [split(n, d) for n, d in zip(series.noise, series.decoder)]
    out = series.copy()
    out["model"], out["code"], out["basis"], out["dec"] = zip(*parts) if parts else ([], [], [], [])
    return out


def small_crossing(g: pd.DataFrame, d1: int, d2: int, guess: float):
    """Crossing of d1 and d2 by campaign_report's estimate (crossings.refine, +-15% window, bootstrap error),
    started from `guess`; (nan, nan) if it does not converge within a factor 2 of the guess."""
    import crossings as X

    if not np.isfinite(guess):
        return np.nan, np.nan
    try:
        c, e, _ = X.refine(g, d1, d2, guess, rel=0.15, deg=1, boot=200)
    except Exception:  # noqa: BLE001
        return np.nan, np.nan
    return (c, e) if np.isfinite(c) and 0.5 * guess < c < 2 * guess else (np.nan, np.nan)


def series_table(th: pd.DataFrame, cr: pd.DataFrame, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every series with its fit, whether it counts, and MWPM's small-d crossing. The crossing is that of
    campaign_report's pair crossings; where its first guess (the first sign change of ln p_L(d2) - ln p_L(d1)
    from small p) lands on noise at small p and the estimate fails, the same estimate is started from the
    series' threshold instead (small_how = "from pth")."""
    th = label(th[th.noise.isin(MODELS) | th.noise.str.contains("@")])
    th = th[th.model.isin(MODELS)].copy()
    ok = CR.usable(th, CR.MIN_DMAX)
    th["counts"] = th.index.isin(ok.index)
    small = []
    for r in th.itertuples():
        d1, d2 = SMALL_PAIR.get(r.code, (7, 9))
        c = cr[(cr.noise == r.noise) & (cr.decoder == r.decoder) & (cr.rounds == r.rounds) & (cr.d1 == d1)
               & (cr.d2 == d2)] if len(cr) else cr
        if len(c):
            small.append((f"{d1}/{d2}", float(c.p.iloc[0]), float(c.err.iloc[0]), "pair"))
            continue
        x, e = np.nan, np.nan
        if df is not None:
            g = df[(df.noise == r.noise) & (df.decoder == r.decoder) & (df.rkey == r.rounds)]
            x, e = small_crossing(g, d1, d2, r.pth)
        small.append((f"{d1}/{d2}", x, e, "from pth" if np.isfinite(x) else ""))
    th["small_pair"], th["cross_small"], th["cross_small_err"], th["small_how"] = \
        zip(*small) if small else ([], [], [], [])
    order = {(m, c, b, d): i for i, (m, c, b, d) in enumerate(
        (m, c, b, d) for m in MODELS for c in CODES for b in BASES[c] for d in DECODERS[c])}
    th["k"] = [order.get((r.model, r.code, r.basis, r.dec), 1e9) for r in th.itertuples()]
    cols = ["model", "code", "basis", "dec", "noise", "decoder", "rounds", "pth", "err", "stat", "syst", "method",
            "flag", "counts", "nu", "chi2", "npts", "dmin", "dmax", "cross_first", "cross_last", "cross_last_err",
            "drift", "small_pair", "cross_small", "cross_small_err", "small_how", "shots", "core_hours"]
    return th.sort_values("k")[cols].reset_index(drop=True)


def summary(st: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, code): the weaker basis with its best decoder, MWPM alone, the small-d crossing."""
    rows = []
    for m in MODELS:
        for code in CODES:
            g = st[(st.model == m) & (st.code == code)]
            row = dict(model=m, code=code, theirs=THEIRS[code].get(m, np.nan))
            ok = g[g.counts]
            best = {b: ok[ok.basis == b].sort_values("pth").iloc[-1] for b in BASES[code] if (ok.basis == b).any()}
            if len(best) == len(BASES[code]):
                weak = min(best.values(), key=lambda r: r.pth)
                strong = max(best.values(), key=lambda r: r.pth)
                row.update(pth=weak.pth, err=weak.err, basis=weak.basis, decoder=weak.dec, other_basis=strong.pth)
            mw = {b: ok[(ok.basis == b) & (ok.dec == "mwpm")] for b in BASES[code]}
            if all(len(v) for v in mw.values()):
                r = min((v.iloc[0] for v in mw.values()), key=lambda r: r.pth)
                row.update(mwpm=r.pth, mwpm_err=r.err, mwpm_basis=r.basis)
            sm = g[(g.dec == "mwpm") & g.cross_small.notna()]
            if len(sm):
                r = sm.sort_values("cross_small").iloc[0]
                missing = [b for b in BASES[code] if b not in set(sm.basis)]
                row.update(cross_small=r.cross_small, cross_small_err=r.cross_small_err, small_pair=r.small_pair,
                           small_basis=r.basis + (f"; no {', '.join(missing)} crossing" if missing else ""))
            rows.append(row)
    return pd.DataFrame(rows)


def markdown(st: pd.DataFrame, sm: pd.DataFrame) -> str:
    f = R.fmt
    lines = ["# Corners of the Hetenyi-Wootton threshold surface in our circuits", "",
             "Thresholds in % of p, r = d, uncertainty in the last digits (campaign_report rules: finite-size fit "
             "over d = 11 to 21, honeycomb 12 to 20; `drift`, pair-only and d < 11 fits do not count). Theirs: "
             "Table I of arXiv:2306.17786 (d = 11 to 17, matching, intercepts of a fitted plane). Ours, MWPM: "
             "the weaker basis under MWPM. Best: each basis with its best decoder (MWPM, correlated matching, "
             "belief-matching), then the weaker basis. Crossing: MWPM, d = 7/9 (honeycomb 4/8), weaker basis.", ""]
    for m in MODELS:
        lines += [f"## {m}: {MODEL_LAB[m]}", "", "| code | theirs | ours, MWPM | ours, best | basis, decoder | "
                  "small-d crossing |", "|---|---|---|---|---|---|"]
        for r in sm[sm.model == m].itertuples():
            theirs = f"{r.theirs:g}" if np.isfinite(r.theirs) else "--"
            mw = f(getattr(r, "mwpm", np.nan), getattr(r, "mwpm_err", np.nan)) if hasattr(r, "mwpm") else "--"
            best = f(r.pth, r.err) if hasattr(r, "pth") and np.isfinite(getattr(r, "pth", np.nan)) else "--"
            bd = f"{r.basis}, {DEC_LAB.get(r.decoder, r.decoder)}" if isinstance(getattr(r, "basis", None), str) else ""
            cs = (f(r.cross_small, r.cross_small_err) + f" ({r.small_pair}, {r.small_basis})"
                  if hasattr(r, "cross_small") and np.isfinite(getattr(r, "cross_small", np.nan)) else "--")
            lines.append(f"| {CODE_LAB[r.code]} | {theirs} | {mw} | {best} | {bd} | {cs} |")
        lines.append("")
    lines += ["## Every series", "", "| model | code | basis | decoder | threshold | method | d | counts | "
              "small-d crossing |", "|---|---|---|---|---|---|---|---|---|"]
    for r in st.itertuples():
        th = f(r.pth, r.err) + (" drift" if r.flag == "drift" else "")
        cs = f(r.cross_small, r.cross_small_err) + f" ({r.small_pair})" if np.isfinite(r.cross_small) else "--"
        lines.append(f"| {r.model} | {r.code} | {r.basis} | {DEC_LAB.get(r.dec, r.dec)} | {th} | {r.method} | "
                     f"{int(r.dmin)}-{int(r.dmax)} | {'yes' if r.counts else 'no'} | {cs} |")
    return "\n".join(lines) + "\n"


HONEYCOMB_DISTS = [4, 8, 12, 16, 20]


def figures(df: pd.DataFrame, st: pd.DataFrame, out: Path) -> None:
    plt = R.setup_style()
    R.PLT = plt
    # campaign_report draws d = 3, 5, ..., 21; the honeycomb runs at d = 4, 8, ..., 20
    R.DISTS = sorted(set(R.DISTS) | set(HONEYCOMB_DISTS))
    for d, c, m in zip(HONEYCOMB_DISTS, ["#A3E635", "#22C55E", "#14B8A6", "#0EA5E9", "#1E3A8A"],
                       ["<", ">", "p", "h", "X"]):
        R.DCOL.setdefault(d, c)
        R.DMARK.setdefault(d, m)
    df = label(df)
    for m in MODELS:
        rows = [(c, b) for c in CODES for b in BASES[c]]
        fig, axes = plt.subplots(len(rows), 3, figsize=(R.FULL, 1.55 * len(rows) + 0.5), squeeze=False)
        for i, (code, basis) in enumerate(rows):
            for j, dec in enumerate(DECODERS[code]):
                ax = axes[i, j]
                g = df[(df.model == m) & (df.code == code) & (df.basis == basis) & (df.dec == dec) & (df.rkey == "d")]
                t = st[(st.model == m) & (st.code == code) & (st.basis == basis) & (st.dec == dec)]
                if g.empty:
                    ax.set_visible(False)
                    continue
                pth = float(t.pth.iloc[0]) if len(t) else np.nan
                err = float(t.err.iloc[0]) if len(t) else np.nan
                R.curves(ax, g, pth, err)
                ax.set_title(f"{CODE_LAB[code]}, {basis}, {DEC_LAB[dec]}", loc="left", fontsize=6.5, color=R.INK)
                if np.isfinite(pth):
                    ok = bool(t.counts.iloc[0])
                    ax.text(0.04, 0.95, f"{R.fmt(pth, err)}%" + ("" if ok else " (n.c.)"), transform=ax.transAxes,
                            fontsize=6, color=R.INK, va="top")
                if i == len(rows) - 1:
                    ax.set_xlabel(r"$p\;(\times10^{-3})$")
                if j == 0:
                    ax.set_ylabel(r"$p_L$")
        fig.suptitle(f"{m}: {MODEL_LAB[m]}, r = d", x=0.02, ha="left", fontsize=9, color=R.INK)
        R.dist_legend(fig)
        fig.tight_layout(w_pad=0.5, h_pad=0.6, rect=(0, 0.045, 1, 0.985))
        fig.savefig(out / f"fig_{m}.pdf")
        fig.savefig(out / f"fig_{m}.png", dpi=110)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", required=True, help="merged points (campaign.py merge)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = R.load(args.points, pool=False)
    df = df[df.noise.isin(MODELS) | df.noise.str.contains("@")]
    df = df[df.rkey == "d"]
    th, cr = R.thresholds(df, workers=args.workers)
    th["rounds"] = th.rounds.astype(str)
    if len(cr):
        cr["rounds"] = cr.rounds.astype(str)
    st = series_table(th, cr, df)
    sm = summary(st)
    st.to_csv(out / "thresholds.csv", index=False)
    cr.to_csv(out / "crossings.csv", index=False)
    sm.to_csv(out / "summary.csv", index=False)
    (out / "thresholds.md").write_text(markdown(st, sm))
    if not args.no_figures:
        figures(df, st, out)
    print(f"{len(st)} series ({int(st.counts.sum())} count) -> {out / 'thresholds.csv'}; summary -> {out / 'thresholds.md'}")


if __name__ == "__main__":
    main()
