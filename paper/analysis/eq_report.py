"""
eq_report.py
----------------------------------------------------------------------------
Equal-qubit comparison below threshold (scripts/campaign_eq.py): XYZ^2 against the CSS and XZZX surface codes at
p = 1e-3, 2e-3, 3e-3 under sd6, biased100 and purez.

    python paper/analysis/eq_report.py --points docs/data/campaign/eq/points.csv --out docs/data/campaign/eq

Per point: the logical error per shot p_L with its 68% Wilson interval, and the error per round epsilon from
p_L = (1 - (1 - 2 epsilon)^r) / 2 (r = 3d). A point without logical errors is a 95% upper bound p_L < 3/N.

- Decoder. For every code and noise model one decoder is used at every point: among the decoders measured at
  every point of the series, the one whose fits give the lowest epsilon at the largest measured distance
  (mean of ln epsilon over p and basis). MWPM alone is reported as well.
- Weaker basis. For every code, noise model and p, the basis whose fit gives the larger epsilon at the largest
  measured distance.
- Lambda. Weighted fit of ln epsilon against d over d >= 5 and points with at least 30 logical errors;
  Lambda = epsilon(d) / epsilon(d + 2) = exp(-2 slope), with its fit error (scaled by sqrt(chi2/dof) when
  that exceeds 1) and its change when the smallest remaining distance is dropped.
- Equal qubits. Directly at the pairs XYZ^2 d = 5, 9, 11 against surface d = 7, 13, 15 (99/97, 323/337,
  483/449 qubits), and from the fits at the qubit counts of XYZ^2, with d(n) = sqrt((n + 1) / 4) for XYZ^2 and
  sqrt((n + 1) / 2) for the surface codes (the counts of the Stim circuits).
- Qubits for epsilon = 1e-6 and 1e-9 per round from the fits, marked as extrapolations beyond the largest
  measured distance; 1e-9 only within two decades of the last measured point.
- Time boundary. At the pair points with r = d and r = 3d, epsilon from the two lengths:
  (1 - 2 p_L(3d)) / (1 - 2 p_L(d)) = (1 - 2 epsilon)^(2d).

Writes points_eps.csv, lambda.csv, decoders.csv, pairs.csv, interp.csv, targets.csv, boundary.csv, eq.md,
fig_eq.png and fig_eq_pgf.tex (the pgfplots panels of the paper figure).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import campaign as C  # noqa: E402

MODELS = ["sd6", "biased100", "purez"]
MODEL_LAB = {"sd6": "SD6", "biased100": r"$\eta=100$", "purez": r"Pure $Z$"}
CODES = ["xyz2", "css", "xzzx"]
CODE_LAB = {"xyz2": "XYZ^2", "css": "CSS", "xzzx": "XZZX"}
QUBIT_C = {"xyz2": 4, "css": 2, "xzzx": 2}               # n = c d^2 - 1 (Stim circuits)
PAIRS = [(5, 7), (9, 13), (11, 15)]
PS = [1e-3, 2e-3, 3e-3]
Z68 = 0.9944578832097535                                 # two-sided 68.27%
MIN_FIT_ERRORS = 30
TARGETS = [1e-6, 1e-9]


def qubits(code: str, d: float) -> float:
    return QUBIT_C[code] * d * d - 1


def d_of(code: str, n: float) -> float:
    return math.sqrt((n + 1) / QUBIT_C[code])


def wilson(k: int, n: int, z: float = Z68):
    if n <= 0:
        return np.nan, np.nan
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(c - h, 0.0), min(c + h, 1.0)


def eps_of(pl, r):
    pl = np.minimum(np.asarray(pl, dtype=float), 0.4999999)
    return 0.5 * (1 - (1 - 2 * pl) ** (1.0 / np.asarray(r, dtype=float)))


def parse(noise: str, decoder: str):
    if ":" in decoder:
        code, basis, dec = C.competitor(decoder)
        return code, basis, noise, dec
    model, _, basis = C.split_noise(noise)
    return "xyz2", basis, model, decoder


def dec_family(dec: str) -> str:
    """corr_gauge (XYZ^2) and corr (surface codes) are both 'corr'."""
    return "corr" if dec.startswith("corr") else dec


def load(points: str) -> pd.DataFrame:
    df = pd.read_csv(points)
    df = df[df.shots > 0].copy()
    parts = [parse(n, d) for n, d in zip(df.noise, df.decoder)]
    df["code"], df["basis"], df["model"], df["dec"] = zip(*parts)
    df["fam"] = df.dec.map(dec_family)
    df["rkey"] = np.where(df.rounds == 3 * df.d, "3d", np.where(df.rounds == df.d, "d", "other"))
    df = df[df.rkey != "other"]
    df["p"] = df.value.round(10)
    df["n_qubits"] = [qubits(c, d) for c, d in zip(df.code, df.d)]
    lo, hi = zip(*[wilson(int(k), int(n)) for k, n in zip(df.errors, df.shots)])
    df["pL"] = df.errors / df.shots
    df["pL_lo"], df["pL_hi"] = lo, hi
    zero = df.errors == 0
    df.loc[zero, "pL_hi"] = 3.0 / df.loc[zero, "shots"]      # 95% upper bound
    df.loc[zero, "pL_lo"] = 0.0
    df["eps"] = eps_of(df.pL, df.rounds)
    df["eps_lo"] = eps_of(df.pL_lo, df.rounds)
    df["eps_hi"] = eps_of(df.pL_hi, df.rounds)
    df.loc[zero, "eps"] = np.nan
    df["upper"] = zero
    return df.sort_values(["code", "model", "basis", "fam", "p", "rkey", "d"]).reset_index(drop=True)


# --------------------------------------------------------------------------- fits
def wfit(g: pd.DataFrame):
    """Weighted fit ln eps = a + b d. Returns dict(a, b, cov, chi2, dof, ds) or None."""
    g = g[(g.errors >= MIN_FIT_ERRORS) & (g.d >= 5)]
    if g.d.nunique() < 2:
        return None
    x = g.d.values.astype(float)
    y = np.log(g.eps.values)
    s = 0.5 * (np.log(g.eps_hi.values) - np.log(np.maximum(g.eps_lo.values, 1e-300)))
    w = 1.0 / np.maximum(s, 1e-6) ** 2
    A = np.vstack([np.ones_like(x), x]).T
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    a, b = cov @ (A.T @ (w * y))
    res = y - (a + b * x)
    dof = len(x) - 2
    chi2 = float(np.sum(w * res ** 2))
    if dof > 0 and chi2 / dof > 1:
        cov = cov * (chi2 / dof)
    return dict(a=float(a), b=float(b), cov=cov, chi2=chi2, dof=dof, ds=sorted(int(v) for v in x),
                dmax=int(x.max()), eps_last=float(np.exp(y[np.argmax(x)])))


def lam(fit):
    if fit is None:
        return np.nan, np.nan
    L = math.exp(-2 * fit["b"])
    return L, 2 * L * math.sqrt(fit["cov"][1, 1])


def fit_eps(fit, d):
    """(epsilon, lo, hi) of a fit at distance d (68%)."""
    v = np.array([1.0, d])
    m = fit["a"] + fit["b"] * d
    s = math.sqrt(float(v @ fit["cov"] @ v))
    return math.exp(m), math.exp(m - s), math.exp(m + s)


def series_fits(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (code, model, basis, fam, p), g in df[df.rkey == "3d"].groupby(["code", "model", "basis", "fam", "p"]):
        f = wfit(g)
        L, sL = lam(f)
        row = dict(code=code, model=model, basis=basis, dec=fam, p=p, Lambda=L, Lambda_err=sL)
        if f is not None:
            row.update(a=f["a"], b=f["b"], chi2=f["chi2"], dof=f["dof"], ds=" ".join(map(str, f["ds"])),
                       dmax=f["dmax"])
            g2 = g[(g.errors >= MIN_FIT_ERRORS) & (g.d >= 5)]
            if g2.d.nunique() >= 3:
                f2 = wfit(g2[g2.d > g2.d.min()])
                L2, _ = lam(f2)
                row["Lambda_drop"] = L2
                row["Lambda_change"] = L2 - L
        rows.append(row)
    return pd.DataFrame(rows)


SKIP_BELOW = 10          # a series ends after the first distance with fewer errors than this at its cap
CODE_D = {"xyz2": [3, 5, 7, 9, 11], "css": [3, 5, 7, 9, 11, 13, 15], "xzzx": [3, 5, 7, 9, 11, 13, 15]}


def complete(g: pd.DataFrame, fam: str, code: str, ref_last: dict) -> bool:
    """Whether decoder `fam` covers its series: for every basis and p of MWPM, every distance from the smallest
    up to its last one is measured (r = 3d), and the series reaches MWPM's last distance or ended there by the
    skip rule (fewer than SKIP_BELOW errors at the cap, an upper bound beyond)."""
    gf = g[g.fam == fam]
    for (basis, p), last_mw in ref_last.items():
        s = gf[(gf.basis == basis) & (gf.p == p)].sort_values("d")
        if s.empty:
            return False
        last = int(s.d.max())
        want = [d for d in CODE_D[code] if d <= last]
        if sorted(int(x) for x in s.d.unique()) != want:
            return False
        if last < last_mw and int(s[s.d == last].errors.iloc[0]) >= SKIP_BELOW:
            return False
    return True


def choose_decoders(df: pd.DataFrame, fits: pd.DataFrame) -> pd.DataFrame:
    """One decoder per code and noise model (see the module docstring): among the decoders that cover the
    series (`complete`), the one whose fits give the lowest epsilon at the largest distance that all of them
    measured, mean of ln epsilon over p and basis."""
    rows = []
    for (code, model), g in df[df.rkey == "3d"].groupby(["code", "model"]):
        mw = g[g.fam == "mwpm"]
        ref_last = {(b, p): int(s.d.max()) for (b, p), s in mw.groupby(["basis", "p"])}
        cover = {fam: complete(g, fam, code, ref_last) for fam in sorted(set(g.fam))}
        cands = [f for f, ok in cover.items() if ok]
        fs = fits[(fits.code == code) & (fits.model == model) & fits.a.notna()] if "a" in fits else fits.iloc[:0]
        score = {fam: [] for fam in cands}
        for (basis, p), gp in fs.groupby(["basis", "p"]):
            if not set(cands) <= set(gp.dec):
                continue
            meas = [set(g[(g.fam == fam) & (g.basis == basis) & (g.p == p)].d) for fam in cands]
            d_eval = max(set.intersection(*meas)) if meas and set.intersection(*meas) else int(gp.dmax.min())
            for fam in cands:
                r = gp[gp.dec == fam].iloc[0]
                score[fam].append(r.a + r.b * d_eval)
        score = {k: float(np.mean(v)) for k, v in score.items() if v}
        best = min(score, key=score.get) if score else "mwpm"
        rows.append(dict(code=code, model=model, decoder=best, candidates=" ".join(sorted(cands)),
                         not_complete=" ".join(sorted(f for f, ok in cover.items() if not ok)),
                         **{f"score_{k}": v for k, v in score.items()}))
    return pd.DataFrame(rows)


def point_or_bound(pts: pd.DataFrame, d: int) -> dict | None:
    """The r = 3d point at distance d, or, when the series ended earlier by the skip rule, the upper bound of
    its last point (epsilon falls with d below threshold)."""
    a = pts[(pts.d == d) & (pts.rkey == "3d")]
    if len(a):
        r = a.iloc[0]
        return dict(eps=r.eps, lo=r.eps_lo, hi=r.eps_hi, err=int(r.errors), upper=bool(r.upper), note="")
    s = pts[(pts.rkey == "3d") & (pts.d < d)].sort_values("d")
    if len(s) and int(s.errors.iloc[-1]) < SKIP_BELOW:
        r = s.iloc[-1]
        return dict(eps=np.nan, lo=0.0, hi=r.eps_hi, err=int(r.errors), upper=True, note=f"bound from d = {int(r.d)}")
    return None


def weaker_basis(fits: pd.DataFrame, df: pd.DataFrame, code: str, model: str, fam: str, p: float) -> str | None:
    fs = fits[(fits.code == code) & (fits.model == model) & (fits.dec == fam) & (fits.p == p)]
    fs = fs[fs.a.notna()] if "a" in fs else fs.iloc[:0]
    if len(fs) == 2:
        dmax = int(fs.dmax.min())
        return max(fs.itertuples(), key=lambda r: r.a + r.b * dmax).basis
    g = df[(df.code == code) & (df.model == model) & (df.fam == fam) & (df.p == p) & (df.rkey == "3d")]
    if g.empty:
        return None
    dcom = max(set(g[g.basis == g.basis.iloc[0]].d) & set(g[g.basis != g.basis.iloc[0]].d) or set(g.d))
    gg = g[g.d == dcom]
    return gg.sort_values("eps_hi").iloc[-1].basis


# --------------------------------------------------------------------------- comparisons
def comparisons(df, fits, choice):
    pair_rows, interp_rows, target_rows, lam_rows, bnd_rows = [], [], [], [], []
    for model in MODELS:
        codes = [c for c in CODES if not (c == "xzzx" and model == "sd6")]
        for mode in ("chosen", "mwpm"):
            for p in PS:
                sel = {}
                for code in codes:
                    ch = choice[(choice.code == code) & (choice.model == model)]
                    fam = "mwpm" if mode == "mwpm" or ch.empty else ch.decoder.iloc[0]
                    b = weaker_basis(fits, df, code, model, fam, p)
                    if b is None:
                        continue
                    pts = df[(df.code == code) & (df.model == model) & (df.fam == fam) & (df.p == p) & (df.basis == b)]
                    f = fits[(fits.code == code) & (fits.model == model) & (fits.dec == fam) & (fits.p == p)
                             & (fits.basis == b)]
                    fit = None
                    if len(f) and "a" in f and pd.notna(f.a.iloc[0]):
                        g3 = pts[pts.rkey == "3d"]
                        fit = wfit(g3)
                    sel[code] = (fam, b, pts, fit)
                    if mode == "chosen" or fam == "mwpm":
                        L, sL = lam(fit)
                        fr = f.iloc[0] if len(f) else None
                        lam_rows.append(dict(model=model, p=p, code=code, mode=mode, decoder=fam, basis=b, Lambda=L,
                                             Lambda_err=sL, Lambda_change=(fr.get("Lambda_change", np.nan)
                                                                           if fr is not None else np.nan),
                                             ds=(fr.get("ds", "") if fr is not None else ""),
                                             dmax_measured=int(pts[pts.rkey == "3d"].d.max()) if len(pts) else 0))
                # direct pairs
                if "xyz2" in sel:
                    fx, bx, px, fitx = sel["xyz2"]
                    for dx, ds in PAIRS:
                        a = point_or_bound(px, dx)
                        for code in [c for c in codes if c != "xyz2" and c in sel]:
                            fs_, bs, ps_, fits_ = sel[code]
                            s = point_or_bound(ps_, ds)
                            row = dict(model=model, p=p, mode=mode, other=code, d_xyz2=dx, d_other=ds,
                                       n_xyz2=qubits("xyz2", dx), n_other=qubits(code, ds),
                                       dec_xyz2=fx, basis_xyz2=bx, dec_other=fs_, basis_other=bs)
                            if a:
                                row.update(eps_xyz2=a["eps"], lo_xyz2=a["lo"], hi_xyz2=a["hi"], err_xyz2=a["err"],
                                           upper_xyz2=a["upper"], note_xyz2=a["note"])
                            if s:
                                row.update(eps_other=s["eps"], lo_other=s["lo"], hi_other=s["hi"], err_other=s["err"],
                                           upper_other=s["upper"], note_other=s["note"])
                            if a and s and not a["upper"] and not s["upper"]:
                                row.update(ratio=row["eps_xyz2"] / row["eps_other"],
                                           ratio_lo=row["lo_xyz2"] / row["hi_other"],
                                           ratio_hi=row["hi_xyz2"] / max(row["lo_other"], 1e-300))
                            pair_rows.append(row)
                    # interpolation at the qubit counts of XYZ^2
                    for code in [c for c in codes if c != "xyz2" and c in sel]:
                        fits_ = sel[code][3]
                        if fitx is None or fits_ is None:
                            continue
                        for dx in sorted(set(px[px.rkey == "3d"].d)):
                            n = qubits("xyz2", dx)
                            dso = d_of(code, n)
                            ex, exl, exh = fit_eps(fitx, dx)
                            eo, eol, eoh = fit_eps(fits_, dso)
                            inside = min(fits_["ds"]) <= dso <= fits_["dmax"]
                            interp_rows.append(dict(model=model, p=p, mode=mode, other=code, n=n, d_xyz2=dx,
                                                    d_other=round(dso, 3), eps_xyz2_fit=ex, eps_other_fit=eo,
                                                    ratio=ex / eo, ratio_lo=exl / eoh, ratio_hi=exh / eol,
                                                    other_interpolated=inside))
                # qubits for the target error rates
                for code, (fam, b, pts, fit) in sel.items():
                    if fit is None or fit["b"] >= 0:
                        continue
                    for tgt in TARGETS:
                        dstar = (math.log(tgt) - fit["a"]) / fit["b"]
                        n = qubits(code, dstar)
                        extrap = dstar > fit["dmax"]
                        within = tgt >= fit["eps_last"] / 100
                        target_rows.append(dict(model=model, p=p, mode=mode, code=code, decoder=fam, basis=b,
                                                target=tgt, d=dstar, n=n, extrapolated=extrap,
                                                quoted=bool(tgt > 1e-9 or within), eps_last=fit["eps_last"],
                                                d_last=fit["dmax"]))
                # time boundary at the pair points
                for code, (fam, b, pts, fit) in sel.items():
                    for d in sorted(set(pts[pts.rkey == "d"].d)):
                        a3 = pts[(pts.d == d) & (pts.rkey == "3d")]
                        a1 = pts[(pts.d == d) & (pts.rkey == "d")]
                        if a3.empty or a1.empty:
                            continue
                        r3, r1 = a3.iloc[0], a1.iloc[0]
                        if r3.upper or r1.upper or r3.errors < MIN_FIT_ERRORS or r1.errors < MIN_FIT_ERRORS:
                            bnd_rows.append(dict(model=model, p=p, mode=mode, code=code, decoder=fam, basis=b, d=d,
                                                 eps_3d=r3.eps, eps_two=np.nan, change=np.nan,
                                                 note="fewer than 30 errors"))
                            continue
                        q = (1 - 2 * r3.pL) / (1 - 2 * r1.pL)
                        e2 = 0.5 * (1 - q ** (1 / (2 * d)))
                        ql = (1 - 2 * r3.pL_hi) / (1 - 2 * r1.pL_lo)
                        qh = (1 - 2 * r3.pL_lo) / (1 - 2 * r1.pL_hi)
                        e2_hi = 0.5 * (1 - ql ** (1 / (2 * d))) if ql > 0 else np.nan
                        e2_lo = 0.5 * (1 - min(qh, 1.0) ** (1 / (2 * d)))
                        bnd_rows.append(dict(model=model, p=p, mode=mode, code=code, decoder=fam, basis=b, d=d,
                                             eps_3d=r3.eps, eps_3d_lo=r3.eps_lo, eps_3d_hi=r3.eps_hi,
                                             eps_two=e2, eps_two_lo=e2_lo, eps_two_hi=e2_hi,
                                             change=e2 / r3.eps - 1, note=""))
    return (pd.DataFrame(pair_rows), pd.DataFrame(interp_rows), pd.DataFrame(target_rows),
            pd.DataFrame(lam_rows), pd.DataFrame(bnd_rows))


# --------------------------------------------------------------------------- output
def sci(x, digits=2):
    if x is None or not np.isfinite(x):
        return "--"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    return f"{m:.{digits - 1}f}e{e}"


def _mode(t: pd.DataFrame, mode: str) -> pd.DataFrame:
    return t[t["mode"] == mode] if len(t) and "mode" in t else t.iloc[:0]


def markdown(df, fits, choice, pairs, interp, targets, lams, bnd) -> str:
    L = ["# Equal-qubit comparison below threshold", "",
         "Error per round epsilon (r = 3d), 68% Wilson intervals; a point without logical errors is a 95% upper "
         "bound (3/N per shot). One decoder per code and noise model (decoders.csv), and MWPM alone. Weaker basis "
         "per code, noise model and p. Lambda = epsilon(d)/epsilon(d+2) from d >= 5 and points with >= 30 errors.",
         "", "## Decoders", "", "| code | noise | decoder | measured at every point |", "|---|---|---|---|"]
    for r in choice.itertuples():
        L.append(f"| {CODE_LAB[r.code]} | {r.model} | {r.decoder} | {r.candidates} |")
    for mode in ("chosen", "mwpm"):
        L += ["", f"## Equal-qubit pairs ({'chosen decoder' if mode == 'chosen' else 'MWPM alone'})", "",
              "| noise | p | pair (qubits) | XYZ^2 eps | other | other eps | ratio XYZ^2/other |", "|---|---|---|---|---|---|---|"]
        for r in _mode(pairs, mode).itertuples():
            ex = (f"< {sci(r.hi_xyz2)}" if getattr(r, "upper_xyz2", False) else
                  f"{sci(r.eps_xyz2)} [{sci(r.lo_xyz2)}, {sci(r.hi_xyz2)}]") if pd.notna(getattr(r, "hi_xyz2", np.nan)) else "--"
            eo = (f"< {sci(r.hi_other)}" if getattr(r, "upper_other", False) else
                  f"{sci(r.eps_other)} [{sci(r.lo_other)}, {sci(r.hi_other)}]") if pd.notna(getattr(r, "hi_other", np.nan)) else "--"
            ra = (f"{r.ratio:.2f} [{r.ratio_lo:.2f}, {r.ratio_hi:.2f}]" if pd.notna(getattr(r, "ratio", np.nan)) else "--")
            L.append(f"| {r.model} | {r.p:g} | {r.d_xyz2}/{r.d_other} ({int(r.n_xyz2)}/{int(r.n_other)}) | {ex} | "
                     f"{CODE_LAB[r.other]} ({r.dec_other}, {r.basis_other}) | {eo} | {ra} |")
        L += ["", f"## Lambda ({'chosen decoder' if mode == 'chosen' else 'MWPM alone'})", "",
              "| noise | p | code | decoder, basis | Lambda | change without smallest d | distances |", "|---|---|---|---|---|---|---|"]
        for r in _mode(lams, mode).itertuples():
            lv = f"{r.Lambda:.2f}({max(1, round(100 * r.Lambda_err)):d})" if np.isfinite(r.Lambda) else "--"
            ch = f"{r.Lambda_change:+.2f}" if pd.notna(r.Lambda_change) else "--"
            L.append(f"| {r.model} | {r.p:g} | {CODE_LAB[r.code]} | {r.decoder}, {r.basis} | {lv} | {ch} | {r.ds} |")
        L += ["", f"## Qubits per logical qubit for epsilon per round ({'chosen decoder' if mode == 'chosen' else 'MWPM alone'})", "",
              "| noise | p | code | 1e-6 | 1e-9 |", "|---|---|---|---|---|"]
        tg = _mode(targets, mode)
        for (model, p, code), g in (tg.groupby(["model", "p", "code"], sort=False) if len(tg) else []):
            cells = []
            for tgt in TARGETS:
                r = g[g.target == tgt]
                if r.empty:
                    cells.append("--")
                    continue
                r = r.iloc[0]
                if not r.quoted:
                    cells.append("beyond two decades")
                else:
                    cells.append(f"{r.n:,.0f} (d = {r.d:.1f})" + (" extrapolated" if r.extrapolated else ""))
            L.append(f"| {model} | {p:g} | {CODE_LAB[code]} | " + " | ".join(cells) + " |")
    L += ["", "## Time boundary at the pair points (chosen decoder)", "",
          "| noise | p | code | d | eps (r = 3d) | eps (r = d and 3d) | change |", "|---|---|---|---|---|---|---|"]
    for r in _mode(bnd, "chosen").itertuples():
        L.append(f"| {r.model} | {r.p:g} | {CODE_LAB[r.code]} | {r.d} | {sci(r.eps_3d)} | {sci(r.eps_two)} | "
                 + (f"{100 * r.change:+.0f}%" if pd.notna(r.change) else r.note) + " |")
    return "\n".join(L) + "\n"


COLORS = {"xyz2": "xyzcode", "css": "csscode", "xzzx": "xzzxcode"}
MARKS = {"xyz2": "*", "css": "square*", "xzzx": "diamond*"}


def pgf(df, choice, fits, p_fig: dict, caption: str = "CAPTION") -> str:
    """The figure* of the paper: three panels (SD6, eta = 100, pure Z) of the error per round against the number
    of qubits, chosen decoder and weaker basis of every code, 68% Wilson intervals, open triangles for 95%
    upper bounds (no logical error)."""
    lo = df[(df.rkey == "3d") & ~df.upper].eps_lo.min()
    hi = df[(df.rkey == "3d")].eps_hi.max()
    ymin = 10 ** math.floor(math.log10(max(lo, 1e-12)))
    ymax = 10 ** math.ceil(math.log10(min(hi, 0.5)))
    head = [r"\begin{figure*}[t]", r"  \centering", r"  \begin{tikzpicture}",
            r"    \begin{groupplot}[group style={group size=3 by 1, horizontal sep=0.55cm, y descriptions at=edge left},",
            r"      width=0.35\textwidth, height=5.2cm, ymode=log, xmin=0, xmax=520, ymin=%g, ymax=%g," % (ymin, ymax),
            r"      ytick={%s}, minor ytick={%s}," % (
                ",".join(f"1e{k}" for k in range(round(math.log10(ymin)), round(math.log10(ymax)) + 1, 2)),
                ",".join(f"1e{k}" for k in range(round(math.log10(ymin)), round(math.log10(ymax)) + 1))),
            r"      xlabel={qubits $n$}, ylabel={error per round $\epsilon$}, ymajorgrids, grid style={grid},",
            r"      axis line style={stone}, tick style={stone}, tick label style={font=\footnotesize, text=ink},",
            r"      label style={font=\footnotesize, text=ink}, title style={font=\footnotesize, text=ink},",
            r"      legend style={font=\footnotesize, draw=none, fill=none, at={(0.97,0.97)}, anchor=north east},",
            r"      error bars/error bar style={line width=0.5pt}, every axis plot/.append style={line width=0.8pt}]"]
    body = []
    for model in MODELS:
        p = p_fig[model]
        codes = [c for c in CODES if not (c == "xzzx" and model == "sd6")]
        body.append(rf"    \nextgroupplot[title={{{MODEL_LAB[model]}, $p={p * 1e3:g}\times10^{{-3}}$}}]")
        for code in codes:
            ch = choice[(choice.code == code) & (choice.model == model)]
            fam = ch.decoder.iloc[0] if len(ch) else "mwpm"
            b = weaker_basis(fits, df, code, model, fam, p)
            g = df[(df.code == code) & (df.model == model) & (df.fam == fam) & (df.p == p) & (df.basis == b)
                   & (df.rkey == "3d")].sort_values("d")
            meas = g[~g.upper]
            if len(meas):
                ys = " ".join(f"({int(r.n_qubits)},{r.eps:.4g}) -= (0,{r.eps - r.eps_lo:.3g}) += (0,{r.eps_hi - r.eps:.3g})"
                              for r in meas.itertuples())
                body.append(rf"      \addplot[{COLORS[code]}, mark={MARKS[code]}, mark size=1.8pt, error bars/.cd, "
                            rf"y dir=both, y explicit] coordinates {{{ys}}};")
                body.append(rf"      \addlegendentry{{{CODE_TEX[code]}}}")
            up = g[g.upper]
            if len(up):
                coords = " ".join(f"({int(r.n_qubits)},{r.eps_hi:.4g})" for r in up.itertuples())
                body.append(rf"      \addplot[{COLORS[code]}, only marks, mark=triangle, mark size=2pt, "
                            rf"mark options={{rotate=180}}, forget plot] coordinates {{{coords}}};")
    tail = [r"    \end{groupplot}", r"  \end{tikzpicture}", "  " + caption, r"  \label{fig:equal_qubit}",
            r"\end{figure*}", ""]
    return "\n".join(head + body + tail)


CODE_TEX = {"xyz2": r"\XYZ{}", "css": "CSS", "xzzx": "XZZX"}


def figure_png(df, choice, fits, p_fig, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = {"xyz2": "#E11D48", "css": "#A8A29E", "xzzx": "#7C3AED"}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, model in zip(axes, MODELS):
        p = p_fig[model]
        for code in [c for c in CODES if not (c == "xzzx" and model == "sd6")]:
            ch = choice[(choice.code == code) & (choice.model == model)]
            fam = ch.decoder.iloc[0] if len(ch) else "mwpm"
            b = weaker_basis(fits, df, code, model, fam, p)
            g = df[(df.code == code) & (df.model == model) & (df.fam == fam) & (df.p == p) & (df.basis == b)
                   & (df.rkey == "3d")].sort_values("d")
            m = g[~g.upper]
            ax.errorbar(m.n_qubits, m.eps, yerr=[m.eps - m.eps_lo, m.eps_hi - m.eps], color=col[code], marker="o",
                        ms=3, label=f"{CODE_LAB[code]} ({fam}, {b})")
            u = g[g.upper]
            ax.plot(u.n_qubits, u.eps_hi, "v", color=col[code], mfc="none")
        ax.set_yscale("log")
        ax.set_xlabel("qubits")
        ax.set_title(f"{model}, p = {p:g}", fontsize=9)
        ax.legend(fontsize=6)
    axes[0].set_ylabel("error per round")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load(args.points)
    fits = series_fits(df)
    choice = choose_decoders(df, fits)
    pairs, interp, targets, lams, bnd = comparisons(df, fits, choice)
    # the panel rate: 1e-3 unless another rate covers more distances with at least 30 errors
    p_fig = {}
    for model in MODELS:
        g = df[(df.model == model) & (df.rkey == "3d") & (df.errors >= MIN_FIT_ERRORS)]
        cover = {p: g[g.p == p].groupby("code").d.nunique().sum() for p in PS}
        p_fig[model] = 1e-3 if cover.get(1e-3, 0) >= max(cover.values()) else max(cover, key=cover.get)
    df.to_csv(out / "points_eps.csv", index=False)
    fits.to_csv(out / "fits.csv", index=False)
    choice.to_csv(out / "decoders.csv", index=False)
    pairs.to_csv(out / "pairs.csv", index=False)
    interp.to_csv(out / "interp.csv", index=False)
    targets.to_csv(out / "targets.csv", index=False)
    lams.to_csv(out / "lambda.csv", index=False)
    bnd.to_csv(out / "boundary.csv", index=False)
    (out / "eq.md").write_text(markdown(df, fits, choice, pairs, interp, targets, lams, bnd)
                               + "\nPanel rates: " + ", ".join(f"{k} {v:g}" for k, v in p_fig.items()) + "\n")
    (out / "fig_eq_pgf.tex").write_text(pgf(df, choice, fits, p_fig) + "\n")
    figure_png(df, choice, fits, p_fig, out / "fig_eq.png")
    print(f"{len(df)} points; decoders: " + ", ".join(f"{r.code}/{r.model} {r.decoder}" for r in choice.itertuples())
          + f"; panel rates {p_fig}")


if __name__ == "__main__":
    main()
