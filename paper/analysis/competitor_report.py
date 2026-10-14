"""
competitor_report.py
----------------------------------------------------------------------------
Thresholds of XYZ^2 against competitor codes under the same circuit-level noise models
(scripts/campaign_competitors.py): the rotated CSS, XZZX and XY surface codes and the honeycomb Floquet code.

    python paper/analysis/competitor_report.py --points docs/data/campaign/points.csv \
        --xyz2 docs/data/campaign/thresholds.csv --out docs/data/campaign/competitors

A competitor series is "code-basis:decoder", and its threshold is fitted like XYZ^2's (campaign_report). A
code's threshold for a noise model and number of rounds is its best decoder's in its weaker basis, since a
memory must protect both. XYZ^2 series are "noise@schedule-L" (L = X or Y; a bare noise name is the
depth-6 Logical-X memory): its value is, per schedule, the weaker of its two logical bases (each with its best
decoder), and then the best schedule (docs/fair_comparison.md). Under a variant that changes the hardware,
each competitor gets its best compilation on that hardware (COMPETITOR_HARDWARE). Thresholds flagged "drift"
are left out. The same comparison is also made with MWPM alone.

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
NOISES = ["sd6", "si1000", "biased10", "biased100", "purez", "em3", "helios_p_noxt", "h2_p_noxt", "helios_p", "h2_p",
          "hyb", "hyb_lr2", "sd6_lr2", "sd6_lr3", "si1000_lr2", "phen", "phen_b10", "sd6_il",
          "hw_g", "hw_g1q", "hw_t", "hw_rr"]
# the competitors' noise models for a variant that changes only what XYZ^2 can do (docs/fair_comparison.md §2):
# hybrid hardware has gates and pair measurements, so a competitor takes its best of all gates (sd6) and all
# pairs (em3); repeated links do not exist in the surface codes, which keep their best on the base hardware;
# sd6_il is a diagnostic, shown against sd6
COMPETITOR_HARDWARE = {"hyb": ["sd6", "em3"], "hyb_lr2": ["sd6", "em3"], "sd6_lr2": ["sd6"], "sd6_lr3": ["sd6"],
                       "si1000_lr2": ["si1000"], "sd6_il": ["sd6"]}


def xyz2_label(noise: str):
    """(noise model, schedule, logical basis) of an XYZ^2 series; a bare name is depth-6 Logical X."""
    if "@" not in noise:
        return noise, "depth6", "X"
    base, rest = noise.split("@", 1)
    sched, logical = rest.rsplit("-", 1)
    return base, sched, logical


def split(series: str):
    cb, dec = series.split(":", 1)
    code, basis = cb.split("-", 1)
    return code, basis, dec


MIN_DMAX = 11      # a threshold counts when its series reaches this distance (small-d crossings still move)


def usable(th: pd.DataFrame, min_dmax: int = MIN_DMAX) -> pd.DataFrame:
    # a finite-size fit over the large distances; a crossing of the two largest distances alone is
    # left out (Tesseract's and the crosstalk models' pair crossings jump between distance pairs)
    return th[th.pth.notna() & (th.flag.fillna("") != "drift") & (th.dmax >= min_dmax) & (th.method == "fss")]


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
    """One row per (noise model, rounds): per schedule the weaker logical basis (each with its best decoder),
    then the best schedule. A model with only one basis measured keeps that value, marked "only"."""
    th = usable(th)
    th = th[~th.decoder.str.contains(":")].copy()
    if th.empty:
        return pd.DataFrame(columns=["noise", "rounds", "code", "pth", "err", "decoder", "basis", "mwpm", "lx"])
    lab = th.noise.map(xyz2_label)
    th["base"], th["sched"], th["logical"] = [x[0] for x in lab], [x[1] for x in lab], [x[2] for x in lab]
    rows = []
    for (base, rk), g in th.groupby(["base", "rounds"]):
        both, single, mw = [], [], []
        for sched, gs in g.groupby("sched"):
            best = {L: gs[gs.logical == L].sort_values("pth").iloc[-1] for L in ("X", "Y") if (gs.logical == L).any()}
            if len(best) == 2:
                L, r = min(best.items(), key=lambda kv: kv[1].pth)
                both.append((float(r.pth), float(r.err), sched, L, r.decoder))
            else:
                single += [(float(r.pth), float(r.err), sched, L, r.decoder) for L, r in best.items()]
            m = {L: gs[(gs.logical == L) & (gs.decoder == "mwpm")] for L in ("X", "Y")}
            if all(len(v) for v in m.values()):
                mw.append(min(float(v.pth.iloc[0]) for v in m.values()))
        lx = g[(g.sched == "depth6") & (g.logical == "X")]
        row = dict(noise=base, rounds=str(rk), code="xyz2", mwpm=max(mw) if mw else np.nan,
                   lx=float(lx.pth.max()) if len(lx) else np.nan)
        if both:
            p, e, s, L, d = max(both, key=lambda x: x[0])
            row.update(pth=p, err=e, decoder=d, basis=f"{s} L_{L}")
        elif single:
            p, e, s, L, d = max(single, key=lambda x: x[0])
            row.update(pth=p, err=e, decoder=d, basis=f"{s} L_{L} only")
        rows.append(row)
    return pd.DataFrame(rows)


def competitor_for(comp: pd.DataFrame, noise: str, rk: str) -> dict:
    """code -> its row under `noise` at `rounds` rk: the best over the noise models of its hardware."""
    srcs = [noise] + COMPETITOR_HARDWARE.get(noise, [])
    g = comp[(comp.rounds == rk) & comp.noise.isin(srcs)]
    out = {}
    for code, gc in g.groupby("code"):
        gc = gc[gc.pth.notna()] if "pth" in gc else gc.iloc[:0]
        if len(gc):
            r = gc.sort_values("pth").iloc[-1].copy()
            if r.noise != noise:
                r["basis"] = f"{r.basis}, as {r.noise}"
            r["mwpm"] = gc.mwpm.max() if "mwpm" in gc else np.nan
            out[code] = r
    return out


def table(xyz: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    out = []
    keys = sorted(set(zip(xyz.noise, xyz.rounds)) | set(zip(comp.noise, comp.rounds)))
    for noise, rk in keys:
        row = dict(noise=noise, rounds=rk)
        x = xyz[(xyz.noise == noise) & (xyz.rounds == rk)]
        rs = {"xyz2": x.iloc[0] if len(x) else None, **competitor_for(comp, noise, rk)}
        for code in ["xyz2"] + CODES:
            r = rs.get(code)
            ok = r is not None and "pth" in r and pd.notna(r.get("pth"))
            row[code] = float(r.pth) if ok else np.nan
            row[code + "_dec"] = f"{r.decoder}, {r.basis}" if ok else ""
            row[code + "_mwpm"] = float(r.mwpm) if r is not None and pd.notna(r.get("mwpm", np.nan)) else np.nan
        row["xyz2_lx"] = float(x.lx.iloc[0]) if len(x) and pd.notna(x.lx.iloc[0]) else np.nan
        others = {c: row[c] for c in CODES if np.isfinite(row[c])}
        if others and np.isfinite(row["xyz2"]):
            top = max(others, key=others.get)
            row.update(best_other=top, ratio=row["xyz2"] / others[top],
                       winner="xyz2" if row["xyz2"] >= others[top] else top)
        out.append(row)
    t = pd.DataFrame(out)
    order = {n: i for i, n in enumerate(NOISES)}
    rorder = {r: i for i, r in enumerate(["d", "1", "2", "3", "5", "10"])}
    return t.sort_values(by=["rounds", "noise"], key=lambda s: s.map(rorder if s.name == "rounds" else order))


def markdown(t: pd.DataFrame) -> str:
    def f(x):
        return "--" if not np.isfinite(x) else f"{100 * x:.2f}"

    lines = ["# XYZ^2 against competitor codes", "",
             "Thresholds in % of p. A competitor's value is its best decoder in its weaker basis; under a variant "
             "that changes the hardware, its best compilation on that hardware (\"as sd6\": all gates, \"as em3\": "
             "all pair measurements). XYZ^2's value is, per extraction schedule, its weaker logical basis (L_X or "
             "L_Y, each with its best decoder), then its best schedule; \"only\" marks a model with one basis "
             "measured. Ratio: XYZ^2 over the best competitor. Only finite-size fits over d >= 11 count: "
             f"thresholds that drift, crossings of two distances only, and series that stop below d = {MIN_DMAX} "
             "(honeycomb: d = 12) are left out. sd6_il (noiseless link checks) is a diagnostic, not hardware.", ""]
    for rk, g in t.groupby("rounds", sort=False):
        if not g[CODES].notna().any().any():
            continue
        lines += [f"## r = {rk}", "", "| noise | XYZ^2 | CSS | XZZX | XY | Honeycomb | ratio | best |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in g.itertuples():
            cells = [f"{f(getattr(r, c))} ({getattr(r, c + '_dec')})" if np.isfinite(getattr(r, c)) else "--"
                     for c in ["xyz2"] + CODES]
            ratio = f"{r.ratio:.2f}" if hasattr(r, "ratio") and np.isfinite(getattr(r, "ratio", np.nan)) else "--"
            lines.append(f"| {r.noise} | " + " | ".join(cells) + f" | {ratio} | {r.winner if isinstance(getattr(r, 'winner', None), str) else ''} |")
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
    t = table(xyz2_values(x), code_values(th))
    t.to_csv(out.parent / (out.name + "_table.csv"), index=False)
    (out.parent / (out.name + ".md")).write_text(markdown(t))
    figure(t, out.with_suffix(".pdf"))
    print(f"{len(th)} competitor series -> {out.with_suffix('.csv')}; table -> {out.name}.md, {out.name}_table.csv")


if __name__ == "__main__":
    main()
