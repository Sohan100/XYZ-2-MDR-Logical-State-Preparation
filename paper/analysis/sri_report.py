"""Thresholds and comparisons of the sequential decoder of Srivastava et al. at circuit level.

usage: python paper/analysis/sri_report.py [--merge] [--workers N]

--merge sums the raw counts of data/campaign/sri/points_*.csv into docs/data/campaign/sri/points.csv first
(scripts/campaign.py merge). The fits follow paper/analysis/campaign_report.py: a finite-size-scaling fit over
the largest distances d >= 11 (at most six) in a window of +-20% around the last crossings, its error the
bootstrap error plus half the spread over windows of 15% and 25% and over the largest four to six distances,
and the drift flag. The comparisons follow docs/fair_comparison.md and docs/xyz2_usefulness.md: every code in
its weaker basis, competitors with the best of MWPM, correlated MWPM and belief-matching, the XYZ^2 memory in
its both-bases schedule, and a win or a loss only at three standard errors (stat and syst).

Writes to docs/data/campaign/sri/: thresholds_sri.csv (columns of thresholds.csv), crossings_sri.csv,
validation.csv, their_model.csv, circuit_decoders.csv, codes.csv, eq.csv, report.md; and the figures
docs/figures/sri/pl_<noise>.{pdf,png}, thr_bias.{pdf,png}, ratios.{pdf,png}.
"""
from __future__ import annotations

import argparse
import glob
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))

import campaign_report as R  # noqa: E402
import competitor_report as CR  # noqa: E402
import eq_report as EQR  # noqa: E402

DATA = ROOT / "docs" / "data" / "campaign"
OUT = DATA / "sri"
FIG = ROOT / "docs" / "figures" / "sri"
RAW = ROOT / "data" / "campaign" / "sri"

NOISES = ["sd6", "si1000", "biased10", "biased100", "purez", "em3", "helios_p_noxt", "h2_p_noxt"]
SERIES = {"": "preparation", "@both-X": "memory X", "@both-Y": "memory Y"}
XYZ2_DEC = ["sri_m", "sri_bm", "sri_m_fast", "sri_bm_fast", "mwpm", "corr_gauge", "bm", "seq_soft", "seq_match"]
OURS_SYM = ["mwpm", "corr_gauge", "bm"]          # the symmetric rules of docs/xyz2_usefulness.md
SRI = ["sri_m", "sri_bm"]                        # their decoder, literal link rule
SRI_FAST = ["sri_m_fast", "sri_bm_fast"]         # their decoder, link rule as one matching per link
# the versions of their decoder compared at circuit level: (name, decoders)
SRI_SETS = (("their decoder", SRI), ("their decoder, fast links", SRI_FAST))
CODES = ["xzzx", "xy", "css", "honeycomb"]
CODE_LAB = {"xyz2": "XYZ²", "xzzx": "XZZX", "xy": "XY", "css": "CSS", "honeycomb": "honeycomb"}
DEC_LAB = {"sri_m": "their matching", "sri_bm": "their belief-matching", "mwpm": "MWPM",
           "corr_gauge": "correlated matching", "bm": "belief-matching", "seq_soft": "sequential BP",
           "seq_match": "erasure passing", "sri_m1": "their matching, Table I to first order",
           "sri_bm1": "their belief-matching, Table I to first order", "sri_m_fast": "their matching, fast links",
           "sri_bm_fast": "their belief-matching, fast links", "sri_bm_dit": "their belief-matching, d BP iterations"}
# their published thresholds (arXiv:2505.03691v1, Figs. 8 to 11): (value, error)
THEIRS = {("codecap", "sri_m"): (0.1834, 0.0008), ("codecap", "sri_bm"): (0.1852, 0.0008),
          ("codecap", "mps"): (0.1892, 0.0005), ("codecap_b10", "sri_m"): (0.1836, 0.0011),
          ("codecap_b10", "sri_bm"): (0.2408, 0.0005), ("codecap_b10", "mps"): (0.2861, 0.0005),
          ("phen", "sri_m"): (0.0342, 0.0001), ("phen", "sri_bm"): (0.0343, 0.0001),
          ("phen_b10", "sri_m"): (0.0349, 0.0002), ("phen_b10", "sri_bm"): (0.0430, 0.0002)}
STOP = {("phen", "sri_m"): (0.032, 0.039), ("phen_b10", "sri_bm"): (0.039, 0.048),
        ("phen", "sri_m_fast"): (0.032, 0.039), ("phen_b10", "sri_bm_fast"): (0.039, 0.048),
        ("phen_b10", "sri_bm_dit"): (0.039, 0.048)}
# variants of their decoder: fast link rule, d iterations of BP, Table I to first order
VARIANT_OF = {"sri_m_fast": "sri_m", "sri_bm_fast": "sri_bm", "sri_bm_dit": "sri_bm", "sri_m1": "sri_m",
              "sri_bm1": "sri_bm"}
ETA = {"sd6": 0.5, "biased10": 10, "biased100": 100, "purez": np.inf}


# --------------------------------------------------------------------------- values
def value(th, noise, dec, rounds="d"):
    """(pth, err, method, flag) of a series, or None."""
    g = th[(th.noise == noise) & (th.decoder == dec) & (th.rounds.astype(str) == str(rounds))]
    if g.empty or not np.isfinite(g.pth.iloc[0]):
        return None
    r = g.iloc[0]
    return float(r.pth), float(r.err), str(r.method), str(r.flag) if isinstance(r.flag, str) else ""


def counts(v):
    """A threshold that counts (docs/fair_comparison.md §3.1): a finite-size fit, no drift."""
    return v is not None and v[2] == "fss" and v[3] != "drift"


def weaker(th, base, decs, rounds="d"):
    """Both-bases memory: per basis the best of `decs`, then the weaker basis. (pth, err, basis, decoder)."""
    best = {}
    for L in ("X", "Y"):
        vals = [(v, dec) for dec in decs for v in [value(th, f"{base}@both-{L}", dec, rounds)] if counts(v)]
        if vals:
            v, dec = max(vals, key=lambda x: x[0][0])
            best[L] = (v[0], v[1], L, dec)
    if len(best) < 2:
        return None
    return min(best.values(), key=lambda x: x[0])


def fmt(v, digits=None):
    """0.507(7)% style; None or nan as an em-free dash."""
    if v is None:
        return "-"
    p, e = v[0], v[1]
    if not np.isfinite(p):
        return "-"
    p, e = 100 * p, 100 * e
    if not np.isfinite(e) or e <= 0:
        return f"{p:.3f}%"
    k = max(0, -int(math.floor(math.log10(e))) + (1 if e < 2 * 10 ** math.floor(math.log10(e)) else 0)) \
        if digits is None else digits
    ue = round(e * 10 ** k)
    return f"{p:.{k}f}({ue})%"


def ratio(a, b):
    """a / b with its error, and z = (a - b) / sqrt(ea^2 + eb^2)."""
    r = a[0] / b[0]
    er = r * math.hypot(a[1] / a[0], b[1] / b[0])
    z = (a[0] - b[0]) / math.hypot(a[1], b[1])
    return r, er, z


def label(z_list):
    """win, tie or loss against a set of competitors (z of XYZ^2 minus each competitor)."""
    if any(z <= -3 for z in z_list):
        return "loss"
    if all(z >= 3 for z in z_list):
        return "win"
    return "tie"


# --------------------------------------------------------------------------- tables
def _val_row(th, base, dec, rounds, model, readout):
    vals = {L: value(th, f"{base}@both-{L}", dec, rounds) for L in ("X", "Y")}
    row = dict(model=model, decoder=dec, readout=readout)
    for L, v in vals.items():
        row[f"pth_{L}"], row[f"pth_{L}_err"] = (v[0], v[1]) if v else (np.nan, np.nan)
        row[f"method_{L}"] = (v[2] + (" " + v[3] if v[3] else "")) if v else ""
    ok = [v for v in vals.values() if v is not None]
    if len(ok) == 2:
        m = min(ok, key=lambda v: v[0])
        row["pth_min"], row["pth_min_err"] = m[0], m[1]
    theirs = THEIRS.get((model, VARIANT_OF.get(dec, dec)))
    row["theirs"], row["theirs_err"] = theirs if theirs else (np.nan, np.nan)
    if theirs and np.isfinite(row.get("pth_min", np.nan)):
        row["ratio_min"] = row["pth_min"] / theirs[0]
    if (model, dec) in STOP and readout == "noiseless" and np.isfinite(row.get("pth_min", np.nan)):
        lo, hi = STOP[(model, dec)]
        row["stop_window"] = f"{100 * lo:.1f} to {100 * hi:.1f}%"
        row["inside"] = bool(lo <= row["pth_min"] <= hi)
    return row


def validation(th_sri, th_main):
    """Their numbers against ours: code capacity (r = 2) and phenomenological noise (r = d), both bases."""
    rows = []
    for model in ("codecap", "codecap_b10", "phen", "phen_b10"):
        cc = model.startswith("codecap")
        rounds = "2" if cc else "d"
        decs = ["sri_m", "sri_bm", "sri_bm_dit", "sri_m1", "mwpm", "bm"] if cc else \
            ["sri_m", "sri_m_fast", "sri_m1", "sri_bm", "sri_bm_fast", "sri_bm_dit"] + XYZ2_DEC[2:]
        for dec in decs:
            src = th_sri if (cc or dec in SRI or dec in VARIANT_OF) else th_main
            rows.append(_val_row(src, model, dec, rounds, model, "noiseless"))
        if not cc:
            for dec in SRI:
                rows.append(_val_row(th_sri, model + "_nr", dec, "d", model, "noisy"))
    return pd.DataFrame(rows)


FIT_RANGES = (("theirs", None), ("d 5-15", (5, 15)), ("d 11-21", (11, 21)))
THEIR_DMAX = {"phen": 15, "phen_b10": 15, "codecap": 25, "codecap_b10": 25}   # legends of their Figs. 8 to 11


def fit_range(df):
    """The same finite-size fit over their distances (d = 3 to 15 for phen, every d at code capacity) and over
    larger ones, both link rules, each basis: how far the distances alone move the threshold."""
    rows = []
    for model, dmax in THEIR_DMAX.items():
        cc = model.startswith("codecap")
        rounds = "2" if cc else "d"
        for dec in (SRI if cc else SRI + SRI_FAST):
            ref = THEIRS[(model, VARIANT_OF.get(dec, dec))][0]
            for L in ("X", "Y"):
                g = df[(df.noise == f"{model}@both-{L}") & (df.decoder == dec) & (df.rkey == rounds)]
                if g.empty:
                    continue
                have = sorted(int(d) for d in g.d.unique())
                row = dict(model=model, decoder=dec, basis=L, theirs=ref)
                for name, rng in FIT_RANGES:
                    ds = [d for d in have if d <= dmax] if rng is None else [d for d in have if rng[0] <= d <= rng[1]]
                    f = None
                    for half in (0.2, 0.3):
                        f = R.fss(g, ds, ref, half, boot=150)
                        if f is not None:
                            break
                    row[f"d_{name}"] = f"{min(ds)}-{max(ds)}" if ds else ""
                    row[f"pth_{name}"], row[f"pth_{name}_err"] = (f["pth"], f["stat"]) if f else (np.nan, np.nan)
                rows.append(row)
    return pd.DataFrame(rows)


def competitors():
    comp = pd.read_csv(DATA / "competitors.csv")
    comp["rounds"] = comp.rounds.astype(str)
    return CR.code_values(comp)


def their_model(th_sri, th_main, cv):
    """XYZ^2 with their decoder against the other codes in their phenomenological model (r = d)."""
    rows = []
    for model in ("phen", "phen_b10"):
        cands = [(name, weaker(th_sri, model, decs)) for name, decs in SRI_SETS]
        cands.append(("mwpm, corr_gauge, bm", weaker(th_main, model, OURS_SYM)))
        for name, x in cands:
            row = dict(model=model, xyz2_decoders=name)
            if x is None:
                rows.append(row)
                continue
            row.update(xyz2=x[0], xyz2_err=x[1], xyz2_basis=x[2], xyz2_dec=x[3])
            zs = []
            for code in CODES:
                c = cv[(cv.noise == model) & (cv.rounds == "d") & (cv.code == code)]
                if c.empty or not np.isfinite(c.pth.iloc[0]):
                    continue
                cval = (float(c.pth.iloc[0]), float(c.err.iloc[0]))
                r, er, z = ratio((x[0], x[1]), cval)
                row[code], row[code + "_err"] = cval
                row[f"ratio_{code}"], row[f"ratio_{code}_err"], row[f"z_{code}"] = r, er, z
                zs.append(z)
            row["label"] = label(zs) if zs else ""
            rows.append(row)
    return pd.DataFrame(rows)


def circuit_decoders(th_sri, th_main):
    rows = []
    for noise in NOISES:
        for s, sname in SERIES.items():
            lab = noise + s
            row = dict(noise=noise, series=sname)
            vals = {}
            for dec in XYZ2_DEC:
                v = value(th_sri if dec.startswith("sri") else th_main, lab, dec)
                vals[dec] = v
                row[dec], row[dec + "_err"] = (v[0], v[1]) if v else (np.nan, np.nan)
                row[dec + "_flag"] = (v[3] if v[2] == "fss" else f"pair {v[3]}".strip()) if v else ""
            for a, b in (("sri_m", "seq_match"), ("sri_m", "mwpm"), ("sri_bm", "bm"), ("sri_bm", "corr_gauge"),
                         ("sri_bm", "seq_soft"), ("sri_bm", "sri_m"), ("sri_m_fast", "seq_match"),
                         ("sri_bm_fast", "bm"), ("sri_m_fast", "sri_m"), ("sri_bm_fast", "sri_bm")):
                if counts(vals[a]) and counts(vals[b]):
                    r, er, z = ratio(vals[a][:2], vals[b][:2])
                    row[f"{a}/{b}"], row[f"{a}/{b}_err"], row[f"z_{a}/{b}"] = r, er, z
            sri_all = SRI + SRI_FAST
            best_sri = max(((vals[d], d) for d in sri_all if counts(vals[d])), key=lambda x: x[0][0], default=None)
            best_ours = max(((vals[d], d) for d in XYZ2_DEC if d not in sri_all and counts(vals[d])),
                            key=lambda x: x[0][0], default=None)
            if best_sri and best_ours:
                r, er, z = ratio(best_sri[0][:2], best_ours[0][:2])
                row.update(best_sri=best_sri[1], best_ours=best_ours[1],
                           **{"bestsri/bestours": r, "bestsri/bestours_err": er, "z_bestsri/bestours": z})
            rows.append(row)
    return pd.DataFrame(rows)


def codes(th_sri, th_main, cv):
    """The XYZ^2 memory against every other code: their decoder, the symmetric decoders, and both."""
    rows = []
    for noise in NOISES + ["phen", "phen_b10"]:
        both = pd.concat([th_sri, th_main])
        variants = {name: weaker(th_sri, noise, decs) for name, decs in SRI_SETS}
        variants.update({"symmetric": weaker(th_main, noise, OURS_SYM),
                         "symmetric + theirs": weaker(both, noise, OURS_SYM + SRI + SRI_FAST)})
        comp = {}
        for code in CODES:
            c = cv[(cv.noise == noise) & (cv.rounds == "d") & (cv.code == code)]
            if not c.empty and np.isfinite(c.pth.iloc[0]):
                comp[code] = (float(c.pth.iloc[0]), float(c.err.iloc[0]), str(c.decoder.iloc[0]),
                              str(c.basis.iloc[0]))
        for name, x in variants.items():
            row = dict(noise=noise, xyz2_decoders=name)
            if x is not None:
                row.update(xyz2=x[0], xyz2_err=x[1], xyz2_basis=x[2], xyz2_dec=x[3])
                zs = []
                for code, c in comp.items():
                    r, er, z = ratio(x[:2], c[:2])
                    row[code], row[code + "_err"] = c[0], c[1]
                    row[f"ratio_{code}"], row[f"ratio_{code}_err"], row[f"z_{code}"] = r, er, z
                    zs.append(z)
                row["label"] = label(zs) if zs else ""
                ranked = sorted([("xyz2", x[0])] + [(k, v[0]) for k, v in comp.items()], key=lambda kv: -kv[1])
                row["rank"] = 1 + [k for k, _ in ranked].index("xyz2")
                row["order"] = " > ".join(CODE_LAB[k] for k, _ in ranked)
            rows.append(row)
    return pd.DataFrame(rows)


def eq_table(points: Path):
    """Error per round at the 99 / 97 qubit pair: XYZ^2 (d = 5) with sri_m and sri_bm against XYZ^2
    belief-matching and the XZZX code with correlated matching (docs/data/campaign/eq/points_eps.csv)."""
    if not points.exists():
        return pd.DataFrame()
    df = pd.read_csv(points)
    df = df[(df.rounds == 3 * df.d) & df.decoder.isin(SRI + SRI_FAST)].copy()
    if df.empty:
        return pd.DataFrame()
    old = pd.read_csv(DATA / "eq" / "points_eps.csv")
    old = old[old.rkey == "3d"]
    rows = []

    def eps_row(k, n, r):
        lo, hi = EQR.wilson(int(k), int(n))
        return float(EQR.eps_of(k / n, r)), float(EQR.eps_of(lo, r)), float(EQR.eps_of(hi, r))

    for model in ("biased100", "purez"):
        for p in (1e-3, 2e-3, 3e-3):
            refs = {}
            for code, dec, d in (("xyz2", "bm", 5), ("xyz2", "corr_gauge", 5), ("xzzx", "corr", 7)):
                g = old[(old.code == code) & (old.model == model) & (old.dec == dec) & (old.d == d)
                        & np.isclose(old.p, p)]
                if len(g) == 2:
                    w = g.sort_values("eps").iloc[-1]          # weaker basis: larger epsilon
                    refs[(code, dec)] = (float(w.eps), float(w.eps_lo), float(w.eps_hi), w.basis)
            for dec in SRI + SRI_FAST:
                g = df[(df.decoder == dec) & df.noise.str.startswith(model + "@") & np.isclose(df.value, p)]
                per = {}
                for _, r in g.iterrows():
                    L = r.noise.rsplit("-", 1)[1]
                    per[L] = eps_row(r.errors, r.shots, r.rounds) + (int(r.errors), int(r.shots))
                if len(per) < 2:
                    continue
                L, e = max(per.items(), key=lambda kv: kv[1][0])
                row = dict(model=model, p=p, decoder=dec, basis=L, eps=e[0], eps_lo=e[1], eps_hi=e[2],
                           errors=e[3], shots=e[4])
                for (code, rdec), ref in refs.items():
                    key = f"{code}_{rdec}"
                    row[key] = ref[0]
                    row[f"ratio_{key}"] = e[0] / ref[0]
                    row[f"ratio_{key}_lo"] = e[1] / ref[2]
                    row[f"ratio_{key}_hi"] = e[2] / ref[1]
                rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- figures
def fig_pl(df, th_sri, noise, out):
    plt = R.PLT
    labels = [noise + s for s in SERIES] if noise in NOISES else [noise + "@both-X", noise + "@both-Y"]
    rounds = "2" if noise.startswith("codecap") else "d"
    if rounds == "2":
        decs = ["sri_m", "sri_bm", "mwpm", "bm"]
    elif noise in NOISES:
        decs = SRI + SRI_FAST
    else:
        decs = SRI
    fig, axes = plt.subplots(len(labels), len(decs), figsize=(R.FULL, 1.9 * len(labels) + 0.6), squeeze=False)
    for i, lab in enumerate(labels):
        for j, dec in enumerate(decs):
            ax = axes[i, j]
            g = df[(df.noise == lab) & (df.decoder == dec) & (df.rkey == rounds)]
            v = value(th_sri, lab, dec, rounds)
            if not g.empty:
                R.curves(ax, g, pth=v[0] if v else None, err=v[1] if v else None)
            name = SERIES.get(lab[len(noise):], lab)
            ax.set_title(f"{name}, {DEC_LAB[dec]}" + (f"\n$p_{{th}}$ = {fmt(v)}" if v else ""), loc="left",
                         fontsize=6.2)
            if i == len(labels) - 1:
                ax.set_xlabel(r"$p$ ($10^{-3}$)")
            if j == 0:
                ax.set_ylabel(r"$p_L$")
    fig.tight_layout()
    R.dist_legend(fig, y=-0.04)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"pl_{noise}.{ext}", dpi=200)
    plt.close(fig)


def fig_bias(th_sri, th_main, out):
    plt = R.PLT
    noises = ["sd6", "biased10", "biased100", "purez"]
    x = np.arange(len(noises))
    fig, axes = plt.subplots(1, 3, figsize=(R.FULL, 2.4), sharey=True)
    for ax, (s, sname) in zip(axes, SERIES.items()):
        for dec in XYZ2_DEC:
            theirs = dec.startswith("sri")
            src = th_sri if theirs else th_main
            ys = [value(src, n + s, dec) for n in noises]
            yv = np.array([v[0] if counts(v) else np.nan for v in ys])
            ye = np.array([v[1] if counts(v) else np.nan for v in ys])
            col = {"sri_m": "#0F766E", "sri_bm": "#0369A1", "sri_m_fast": "#0F766E",
                   "sri_bm_fast": "#0369A1"}.get(dec, R.DEC_COL.get(dec, "#444"))
            ls = "--" if dec in SRI_FAST else "-" if theirs else ":"
            ax.errorbar(x, 100 * yv, yerr=100 * ye, color=col, marker=R.DEC_MARK.get(dec, "o"), ms=3, lw=1.1 if
                        theirs else 0.8, ls=ls, label=DEC_LAB[dec], capsize=0)
        ax.set_xticks(x, [r"$\frac{1}{2}$", "10", "100", r"$\infty$"])
        ax.set_xlabel(r"bias $\eta$")
        ax.set_title(sname, loc="left")
    axes[0].set_ylabel(r"threshold (%)")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"thr_bias.{ext}", dpi=200)
    plt.close(fig)


SHORT = {"sd6": "SD6", "si1000": "SI1000", "biased10": "η=10", "biased100": "η=100", "purez": "pure Z",
         "em3": "EM3", "helios_p_noxt": "Helios", "h2_p_noxt": "H2", "phen": "phen", "phen_b10": "phen, η=10"}


def fig_ratios(codes_t, out):
    plt = R.PLT
    sets = (("their decoder", "#0F766E", "o"), ("their decoder, fast links", "#0369A1", "^"),
            ("symmetric", "#7C3AED", "s"))
    t = codes_t[codes_t.xyz2_decoders.isin([s[0] for s in sets])]
    noises = [n for n in NOISES + ["phen", "phen_b10"] if (t.noise == n).any()]
    fig, axes = plt.subplots(1, len(CODES), figsize=(R.FULL, 2.6), sharey=True)
    for ax, code in zip(axes, CODES):
        for k, (name, col, mk) in enumerate(sets):
            g = t[t.xyz2_decoders == name].set_index("noise").reindex(noises)
            y = g.get(f"ratio_{code}", pd.Series(np.nan, index=noises)).values.astype(float)
            e = g.get(f"ratio_{code}_err", pd.Series(np.nan, index=noises)).values.astype(float)
            ax.errorbar(np.arange(len(noises)) + (k - 1) * 0.22, y, yerr=e, color=col, marker=mk, ms=3, lw=0,
                        elinewidth=0.8, label=name)
        ax.axhline(1.0, color="#57534E", lw=0.6, ls="--")
        ax.set_xticks(np.arange(len(noises)), [SHORT.get(n, n) for n in noises], rotation=90, fontsize=6)
        ax.set_title(f"XYZ² / {CODE_LAB[code]}", loc="left")
    axes[0].set_ylabel("threshold ratio (weaker basis)")
    axes[-1].legend(fontsize=6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"ratios.{ext}", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- markdown
def md_table(df, cols, heads):
    lines = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(c(r)) if callable(c) else str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def pv(r, k):
    v = r.get(k, np.nan)
    e = r.get(k + "_err", np.nan)
    return fmt((v, e)) if np.isfinite(v) else "-"


def rv(r, k, z=None):
    v = r.get(k, np.nan)
    if not np.isfinite(v):
        return "-"
    e = r.get(k + "_err", np.nan)
    s = f"{v:.3f}({round(1000 * e)})" if np.isfinite(e) else f"{v:.3f}"
    if z is not None and np.isfinite(r.get(z, np.nan)):
        s += f", z {r[z]:+.1f}"
    return s


def report_md(val, fr, their, circ, cod, eq, fd) -> str:
    """The tables of docs/srivastava_exact_decoder.md."""
    out = ["## Validation: their numbers and ours", "",
           "Ours in both bases (X, Y) and their minimum; code capacity r = 2, phen r = d; fits over d = 11 to 21.",
           ""]
    lines = ["| model | decoder | readout | theirs | ours X | ours Y | ours min | ours / theirs |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in val.iterrows():
        lines.append(f"| {r.model} | {r.decoder} | {r.readout} | "
                     f"{fmt((r.theirs, r.theirs_err)) if np.isfinite(r.theirs) else '-'} | {pv(r, 'pth_X')} | "
                     f"{pv(r, 'pth_Y')} | {pv(r, 'pth_min')} | "
                     f"{r.ratio_min:.2f} |" if np.isfinite(r.get('ratio_min', np.nan)) else
                     f"| {r.model} | {r.decoder} | {r.readout} | "
                     f"{fmt((r.theirs, r.theirs_err)) if np.isfinite(r.theirs) else '-'} | {pv(r, 'pth_X')} | "
                     f"{pv(r, 'pth_Y')} | {pv(r, 'pth_min')} | - |")
    out += lines
    if fr is not None and not fr.empty:
        out += ["", "## Validation: the distances of the fit", "",
                "The same finite-size fit (statistical errors) over their distances and over larger ones.", ""]
        lines = ["| model | decoder | basis | theirs | " + " | ".join(
                 f"ours, {n}" for n, _ in FIT_RANGES) + " |", "|" + "---|" * (4 + len(FIT_RANGES))]
        for _, r in fr.iterrows():
            cells = [pv(r, f"pth_{n}") + (f" (d {r[f'd_{n}']})" if n == "theirs" else "") for n, _ in FIT_RANGES]
            lines.append(f"| {r.model} | {r.decoder} | {r.basis} | {100 * r.theirs:.2f}% | " + " | ".join(cells) + " |")
        out += lines
    out += ["", "## Their comparison in their model (phen, r = d)", ""]
    lines = ["| model | XYZ² decoders | XYZ² | XZZX | XY | CSS | honeycomb | label |", "|---|---|---|---|---|---|---|---|"]
    for _, r in their.iterrows():
        cells = [rv(r, f"ratio_{c}", f"z_{c}") for c in CODES]
        lines.append(f"| {r.model} | {r.xyz2_decoders} | {fmt((r.get('xyz2', np.nan), r.get('xyz2_err', np.nan)))}"
                     f" {r.get('xyz2_dec', '')} {r.get('xyz2_basis', '')} | " + " | ".join(cells) + f" | {r.get('label', '')} |")
    out += lines
    out += ["", "## Circuit level: XYZ² decoders (r = d)", ""]
    pairs = ["sri_m/seq_match", "sri_bm/bm", "sri_m_fast/seq_match", "sri_bm_fast/bm", "sri_m_fast/sri_m",
             "sri_bm_fast/sri_bm"]
    lines = ["| noise | series | " + " | ".join(XYZ2_DEC) + " | " + " | ".join(pairs)
             + " | best sri / best other |", "|" + "---|" * (len(XYZ2_DEC) + len(pairs) + 3)]
    for _, r in circ.iterrows():
        cells = [pv(r, d) + (f" {r[d + '_flag']}" if isinstance(r.get(d + "_flag"), str) and r[d + "_flag"] else "")
                 for d in XYZ2_DEC]
        lines.append(f"| {r.noise} | {r.series} | " + " | ".join(cells) + " | "
                     + " | ".join(rv(r, k, "z_" + k) for k in pairs) + " | "
                     f"{rv(r, 'bestsri/bestours', 'z_bestsri/bestours')} ({r.get('best_sri', '')} against "
                     f"{r.get('best_ours', '')}) |")
    out += lines
    out += ["", "## Circuit level: codes (memory, weaker basis, r = d)", ""]
    lines = ["| noise | XYZ² decoders | XYZ² | / XZZX | / XY | / CSS | / honeycomb | label | rank |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in cod.iterrows():
        cells = [rv(r, f"ratio_{c}", f"z_{c}") for c in CODES]
        lines.append(f"| {r.noise} | {r.xyz2_decoders} | {fmt((r.get('xyz2', np.nan), r.get('xyz2_err', np.nan)))} "
                     f"({r.get('xyz2_dec', '')}, {r.get('xyz2_basis', '')}) | " + " | ".join(cells)
                     + f" | {r.get('label', '')} | {r.get('rank', '')} |")
    out += lines
    if not eq.empty:
        out += ["", "## Below threshold: 99 / 97 qubits, r = 3d, weaker basis", ""]
        lines = ["| model | p | decoder | eps per round [68%] | / XYZ² bm | / XYZ² corr_gauge | / XZZX corr |",
                 "|---|---|---|---|---|---|---|"]
        for _, r in eq.iterrows():
            def rr(k):
                if f"ratio_{k}" not in r or not np.isfinite(r[f"ratio_{k}"]):
                    return "-"
                return f"{r[f'ratio_{k}']:.2f} [{r[f'ratio_{k}_lo']:.2f}, {r[f'ratio_{k}_hi']:.2f}]"
            lines.append(f"| {r.model} | {r.p:g} | {r.decoder} | {r.eps:.3g} [{r.eps_lo:.3g}, {r.eps_hi:.3g}] "
                         f"({int(r.errors)} errors) | {rr('xyz2_bm')} | {rr('xyz2_corr_gauge')} | {rr('xzzx_corr')} |")
        out += lines
    if fd is not None and not fd.empty:
        out += ["", "## Fault distance (SD6, preparation, r = d)", ""]
        lines = ["| d | decoder | sets of size k | failing sets | undetectable |", "|---|---|---|---|---|"]
        for _, r in fd.iterrows():
            lines.append(f"| {r.d} | {r.decoder} | {r['size']} ({int(r.sets)}) | {int(r.failures)} | {int(r.undetectable)} |")
        out += lines
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    points = OUT / "points.csv"
    if args.merge:
        import campaign as C
        n = C.merge(glob.glob(str(RAW / "points_*.csv")), str(points))
        print(f"merged {n} points")
    df = R.load(points, pool=False)
    eqmask = df.rounds == 3 * df.d
    th_sri, cr = R.thresholds(df[~eqmask], workers=args.workers)
    th_sri = th_sri.sort_values(["noise", "decoder", "rounds"])
    th_sri.to_csv(OUT / "thresholds_sri.csv", index=False)
    cr.to_csv(OUT / "crossings_sri.csv", index=False)
    th_main = pd.read_csv(DATA / "thresholds.csv")
    th_main["rounds"] = th_main.rounds.astype(str)
    th_sri["rounds"] = th_sri.rounds.astype(str)
    cv = competitors()
    val = validation(th_sri, th_main)
    fr = fit_range(df[~eqmask])
    their = their_model(th_sri, th_main, cv)
    circ = circuit_decoders(th_sri, th_main)
    cod = codes(th_sri, th_main, cv)
    eq = eq_table(points)
    for name, t in (("validation", val), ("fit_range", fr), ("their_model", their), ("circuit_decoders", circ),
                    ("codes", cod), ("eq", eq)):
        t.to_csv(OUT / f"{name}.csv", index=False)
    if not args.no_figures:
        R.PLT = R.setup_style()
        for noise in NOISES + ["phen", "phen_b10", "phen_nr", "phen_b10_nr", "codecap", "codecap_b10"]:
            if (df.noise.str.startswith(noise) & df.rkey.isin(["d", "2"]) & ~eqmask).any():
                fig_pl(df, th_sri, noise, FIG)
        fig_bias(th_sri, th_main, FIG)
        fig_ratios(cod, FIG)
    # markdown of the thresholds, as thresholds.md
    t = th_sri.copy()
    lines = ["| noise | decoder | rounds | threshold | method | d | flag | core-hours |", "|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        lines.append(f"| {r.noise} | {r.decoder} | {r.rounds} | {fmt((r.pth, r.err))} | {r.method} | "
                     f"{r.dmin}-{r.dmax} | {r.flag if isinstance(r.flag, str) else ''} | {r.core_hours:.1f} |")
    (OUT / "thresholds_sri.md").write_text("\n".join(lines) + "\n")
    fdp = OUT / "fault_distance.csv"
    fd = pd.read_csv(fdp) if fdp.exists() else None
    (OUT / "report.md").write_text(report_md(val, fr, their, circ, cod, eq, fd))
    print(f"{len(th_sri)} series, {th_sri.core_hours.sum():.0f} core-hours")


if __name__ == "__main__":
    main()
