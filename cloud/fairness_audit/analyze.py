"""Tables and threshold estimates from results_*.jsonl (prints markdown)."""
import collections
import json
import os
import sys

import numpy as np
from scipy.optimize import curve_fit

rng = np.random.default_rng(1)


def load(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def sig(r):
    return max(1e-12, (r["hi"] - r["lo"]) / 2)


def crossing(ra, rb, nboot=400):
    """p where the LER curves of two distances cross (log-LER linear interpolation), with bootstrap sigma."""
    ps = sorted(set(r["p"] for r in ra) & set(r["p"] for r in rb))
    A = {r["p"]: r for r in ra}
    B = {r["p"]: r for r in rb}

    def one(la, lb):
        diff = np.log(lb) - np.log(la)
        for i in range(len(ps) - 1):
            if diff[i] < 0 <= diff[i + 1]:
                t = -diff[i] / (diff[i + 1] - diff[i])
                return ps[i] + t * (ps[i + 1] - ps[i])
        return np.nan

    la = np.array([A[p]["ler"] for p in ps])
    lb = np.array([B[p]["ler"] for p in ps])
    est = one(la, lb)
    sa = np.array([sig(A[p]) for p in ps])
    sb = np.array([sig(B[p]) for p in ps])
    boots = [one(np.clip(la + rng.normal(0, sa), 1e-9, 1), np.clip(lb + rng.normal(0, sb), 1e-9, 1))
             for _ in range(nboot)]
    boots = np.array(boots)
    ok = boots[~np.isnan(boots)]
    return est, (ok.std() if len(ok) > 10 else np.nan), 1 - len(ok) / nboot


def fss(rows, nboot=200):
    """Fit LER = A + B x + C x^2, x = (p - pc) d^(1/nu); returns pc, sigma(pc), nu."""
    p = np.array([r["p"] for r in rows])
    d = np.array([r["d"] for r in rows], float)
    y = np.array([r["ler"] for r in rows])
    s = np.array([sig(r) for r in rows])

    def f(X, pc, nu, a, b, c):
        pp, dd = X
        x = (pp - pc) * dd ** (1 / nu)
        return a + b * x + c * x * x

    p0 = [np.median(p), 1.5, np.median(y), 10.0, 0.0]
    try:
        popt, _ = curve_fit(f, (p, d), y, p0=p0, sigma=s, maxfev=20000)
    except Exception:
        return np.nan, np.nan, np.nan, np.nan
    chi2 = float((((f((p, d), *popt) - y) / s) ** 2).sum() / max(1, len(y) - 5))
    pcs = []
    for _ in range(nboot):
        yb = y + rng.normal(0, s)
        try:
            pb, _ = curve_fit(f, (p, d), yb, p0=popt, sigma=s, maxfev=20000)
            pcs.append(pb[0])
        except Exception:
            pass
    return popt[0], float(np.std(pcs)), popt[1], chi2


def pct(x, e=None):
    if x is None or np.isnan(x):
        return "—"
    return f"{100 * x:.3f}" + ("" if e is None or np.isnan(e) else f" ± {100 * e:.3f}")


def main():
    out = []
    a = load("results_2a.jsonl")
    if a:
        out.append("### 2a. CSS rotated surface code, sd6, MWPM: our circuit vs `stim.Circuit.generated`\n")
        out.append("| d | p | basis | ours LER | stim LER | ours / stim |")
        out.append("|---|---|---|---|---|---|")
        idx = {(r["kind"], r["d"], r["p"], r["basis"]): r for r in a}
        for d in (3, 5, 7):
            for p in (0.004, 0.006, 0.008):
                for b in ("X", "Z"):
                    o, s = idx.get(("css", d, p, b)), idx.get(("stim", d, p, b))
                    if not (o and s):
                        continue
                    ratio = o["ler"] / s["ler"]
                    rerr = ratio * np.hypot(sig(o) / o["ler"], sig(s) / s["ler"])
                    out.append(f"| {d} | {p:.3f} | {b} | {o['ler']:.4f} ± {sig(o):.4f} | "
                               f"{s['ler']:.4f} ± {sig(s):.4f} | {ratio:.2f} ± {rerr:.2f} |")
        out.append("")
    rows = load("results_2b.jsonl") + load("results_abl.jsonl")
    if rows:
        series = collections.defaultdict(list)
        for r in rows:
            series[(r["kind"], r["decoder"], r["basis"])].append(r)
        out.append("### 2b. Threshold estimates (sd6, r = d), % \n")
        out.append("Pairwise crossings use log-LER interpolation on the p grid 0.60–0.90 %; "
                   "± is a 1σ bootstrap over the binomial errors. FSS: quadratic finite-size "
                   "fit over every d of the series.\n")
        ds_all = sorted(set(r["d"] for r in rows))
        pairs = list(zip(ds_all[:-1], ds_all[1:]))
        out.append("| circuit | decoder | basis | " + " | ".join(f"cross d={a}/{b}" for a, b in pairs)
                   + " | FSS p_c (d≥7) | ν | χ²/dof |")
        out.append("|---|---|---|" + "---|" * len(pairs) + "---|---|---|")
        for key in sorted(series):
            S = series[key]
            byd = collections.defaultdict(list)
            for r in S:
                byd[r["d"]].append(r)
            cells = []
            for a_, b_ in pairs:
                if a_ in byd and b_ in byd:
                    est, e, miss = crossing(byd[a_], byd[b_])
                    cells.append(pct(est, e) + (f" ({miss:.0%} no cross)" if miss > 0.05 else ""))
                else:
                    cells.append("—")
            pc, epc, nu, chi2 = fss([r for r in S if r["d"] >= 7])
            out.append(f"| {key[0]} | {key[1]} | {key[2]} | " + " | ".join(cells)
                       + f" | {pct(pc, epc)} | {nu:.2f} | {chi2:.1f} |")
        out.append("")
        out.append("### 2b. Raw logical error rates (LER per r = d memory)\n")
        out.append("| circuit | decoder | basis | d | " + " | ".join(f"p={100*p:.2f}%" for p in sorted(set(r['p'] for r in rows))) + " |")
        ps = sorted(set(r["p"] for r in rows))
        out.append("|---|---|---|---|" + "---|" * len(ps))
        for key in sorted(series):
            byd = collections.defaultdict(dict)
            for r in series[key]:
                byd[r["d"]][r["p"]] = r
            for d in sorted(byd):
                out.append(f"| {key[0]} | {key[1]} | {key[2]} | {d} | " + " | ".join(
                    (f"{byd[d][p]['ler']:.4f}±{sig(byd[d][p]):.4f}" if p in byd[d] else "—") for p in ps) + " |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
