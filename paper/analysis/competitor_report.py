"""
competitor_report.py
----------------------------------------------------------------------------
Thresholds of XYZ^2 against competitor codes under the same circuit-level noise models
(scripts/campaign_competitors.py): the rotated CSS, XZZX and XY surface codes and the honeycomb Floquet code.

    python paper/analysis/competitor_report.py --points docs/data/campaign/points.csv \
        --xyz2 docs/data/campaign/thresholds.csv --out docs/data/campaign/competitors

A competitor series is "code-basis:decoder", and its threshold is fitted like XYZ^2's (campaign_report). A
code's threshold for a noise model and number of rounds is its best decoder's in its weaker basis, since a
memory must protect both. The XYZ^2 value is the best of its own decoders, for its Logical-X memory.
Thresholds flagged "drift" are left out. The same comparison is also made with MWPM alone.

Writes:
- <out>.csv: every competitor series;
- <out>_table.csv and <out>.md: the comparison;
- <out>.pdf: thresholds per noise model at r = d and r = 1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import campaign_report as R  # noqa: E402

CODES = ["css", "xzzx", "xy", "honeycomb"]
LABEL = {"xyz2": "XYZ$^2$", "css": "CSS surface", "xzzx": "XZZX surface", "xy": "XY surface",
         "honeycomb": "Honeycomb Floquet"}
NOISES = ["sd6", "si1000", "biased10", "biased100", "purez", "em3", "helios_p_noxt", "h2_p_noxt", "helios_p", "h2_p"]


def split(series: str):
    cb, dec = series.split(":", 1)
    code, basis = cb.split("-", 1)
    return code, basis, dec


MIN_DMAX = 11      # a threshold counts when its series reaches this distance (small-d crossings still move)


def usable(th: pd.DataFrame, min_dmax: int = MIN_DMAX) -> pd.DataFrame:
    return th[th.pth.notna() & (th.flag.fillna("") != "drift") & (th.dmax >= min_dmax)]


def code_values(th: pd.DataFrame) -> pd.DataFrame:
    """One row per (noise, rounds, code): best decoder per basis, the weaker basis, and the same for MWPM."""
    th = usable(th).copy()
    parts = th.decoder.map(split)
    th["code"], th["basis"], th["dec"] = [p[0] for p in parts], [p[1] for p in parts], [p[2] for p in parts]
    rows = []
    for (noise, rk, code), g in th.groupby(["noise", "rounds", "code"]):
        best = {b: g[g.basis == b].sort_values("pth").iloc[-1] for b in ("X", "Z") if (g.basis == b).any()}
        mw = {b: g[(g.basis == b) & (g.dec == "mwpm")] for b in ("X", "Z")}
        row = dict(noise=noise, rounds=str(rk), code=code)
        if len(best) == 2:
            weak = min(best.values(), key=lambda r: r.pth)
            row.update(pth=weak.pth, err=weak.err, basis=weak.basis, decoder=weak.dec,
                       other=max(best.values(), key=lambda r: r.pth).pth)
        if all(len(v) for v in mw.values()):
            row["mwpm"] = min(float(v.pth.iloc[0]) for v in mw.values())
        rows.append(row)
    return pd.DataFrame(rows)


def xyz2_values(th: pd.DataFrame) -> pd.DataFrame:
    th = usable(th)
    th = th[~th.decoder.str.contains(":")]
    rows = []
    for (noise, rk), g in th.groupby(["noise", "rounds"]):
        b = g.sort_values("pth").iloc[-1]
        m = g[g.decoder == "mwpm"]
        rows.append(dict(noise=noise, rounds=str(rk), code="xyz2", pth=b.pth, err=b.err, decoder=b.decoder,
                         basis="L_X", mwpm=float(m.pth.iloc[0]) if len(m) else np.nan))
    return pd.DataFrame(rows)


def table(vals: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (noise, rk), g in vals.groupby(["noise", "rounds"]):
        row = dict(noise=noise, rounds=rk)
        for code in ["xyz2"] + CODES:
            r = g[g.code == code]
            row[code] = float(r.pth.iloc[0]) if len(r) and "pth" in r and pd.notna(r.pth.iloc[0]) else np.nan
            row[code + "_dec"] = (f"{r.decoder.iloc[0]}" + (f", {r.basis.iloc[0]}" if code != "xyz2" else "")
                                  if len(r) and pd.notna(row[code]) else "")
            row[code + "_mwpm"] = float(r.mwpm.iloc[0]) if len(r) and "mwpm" in r and pd.notna(r.mwpm.iloc[0]) \
                else np.nan
        comp = {c: row[c] for c in CODES if np.isfinite(row[c])}
        if comp and np.isfinite(row["xyz2"]):
            top = max(comp, key=comp.get)
            row.update(best_other=top, ratio=row["xyz2"] / comp[top],
                       winner="xyz2" if row["xyz2"] >= comp[top] else top)
        out.append(row)
    t = pd.DataFrame(out)
    order = {n: i for i, n in enumerate(NOISES)}
    rorder = {r: i for i, r in enumerate(["d", "1", "2", "3", "5", "10"])}
    return t.sort_values(by=["rounds", "noise"], key=lambda s: s.map(rorder if s.name == "rounds" else order))


def markdown(t: pd.DataFrame) -> str:
    def f(x):
        return "--" if not np.isfinite(x) else f"{100 * x:.2f}"

    lines = ["# XYZ^2 against competitor codes", "",
             "Thresholds in % of p. A competitor's value is its best decoder in its weaker basis; XYZ^2's is its "
             "best decoder (Logical-X memory). Ratio: XYZ^2 over the best competitor. Thresholds that drift, and "
             f"series that stop below d = {MIN_DMAX} (honeycomb: d = 12), are left out.", ""]
    for rk, g in t.groupby("rounds", sort=False):
        if not g[CODES].notna().any().any():
            continue
        lines += [f"## r = {rk}", "", "| noise | XYZ^2 | CSS | XZZX | XY | Honeycomb | ratio | best |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in g.itertuples():
            cells = [f"{f(getattr(r, c))} ({getattr(r, c + '_dec')})" if np.isfinite(getattr(r, c)) else "--"
                     for c in ["xyz2"] + CODES]
            ratio = f"{r.ratio:.2f}" if hasattr(r, "ratio") and np.isfinite(getattr(r, "ratio", np.nan)) else "--"
            lines.append(f"| {r.noise} | " + " | ".join(cells) + f" | {ratio} | {getattr(r, 'winner', '') or ''} |")
        lines += ["", f"MWPM only, r = {rk}:", "", "| noise | XYZ^2 | CSS | XZZX | XY | Honeycomb |",
                  "|---|---|---|---|---|---|"]
        for r in g.itertuples():
            lines.append(f"| {r.noise} | " + " | ".join(f(getattr(r, c + "_mwpm")) for c in ["xyz2"] + CODES) + " |")
        lines.append("")
    return "\n".join(lines)


def figure(t: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R.setup_style()
    marks = {"xyz2": ("o", "#d62728"), "css": ("s", "#1f77b4"), "xzzx": ("D", "#2ca02c"), "xy": ("^", "#9467bd"),
             "honeycomb": ("v", "#7f7f7f")}
    rks = [rk for rk in ("d", "1") if (t.rounds == rk).any()]
    fig, axes = plt.subplots(1, len(rks), figsize=(3.4 * len(rks), 2.8), squeeze=False)
    for ax, rk in zip(axes[0], rks):
        g = t[t.rounds == rk].reset_index(drop=True)
        for k, code in enumerate(["xyz2"] + CODES):
            y = 100 * g[code].values
            ax.plot(np.arange(len(g)) + (k - 2) * 0.1, y, marks[code][0], color=marks[code][1], ms=4,
                    label=LABEL[code], ls="none")
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels(g.noise, rotation=60, ha="right", fontsize=7)
        ax.set_yscale("log")
        ax.set_ylabel("threshold (%)")
        ax.set_title(f"r = {rk}", fontsize=9)
    axes[0][0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", nargs="+", required=True, help="merged points with the competitor series")
    ap.add_argument("--xyz2", required=True, help="thresholds.csv of XYZ^2 (campaign_report)")
    ap.add_argument("--out", required=True, help="output stem")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    df = pd.concat([R.load(p, pool=False) for p in args.points], ignore_index=True)
    df = df[df.decoder.str.contains(":")]
    th, cr = R.thresholds(df, workers=args.workers)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    th.to_csv(out.with_suffix(".csv"), index=False)
    cr.to_csv(out.parent / (out.name + "_crossings.csv"), index=False)
    x = pd.read_csv(args.xyz2)
    x["rounds"] = x.rounds.astype(str)
    th["rounds"] = th.rounds.astype(str)
    vals = pd.concat([xyz2_values(x), code_values(th)], ignore_index=True)
    t = table(vals)
    t.to_csv(out.parent / (out.name + "_table.csv"), index=False)
    (out.parent / (out.name + ".md")).write_text(markdown(t))
    figure(t, out.with_suffix(".pdf"))
    print(f"{len(th)} competitor series -> {out.with_suffix('.csv')}; table -> {out.name}.md, {out.name}_table.csv")


if __name__ == "__main__":
    main()
