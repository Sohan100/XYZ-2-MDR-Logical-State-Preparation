"""
make_tables.py
----------------------------------------------------------------------------
Markdown tables of the schedule search (distance distributions, p_L in %
with one-sigma binomial errors, pairwise crossings) for results.md.

    python cloud/schedule_search/make_tables.py > cloud/schedule_search/tables.md
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from crossings import table as crossing_table

D = Path(__file__).resolve().parent / "data"


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.values) + " |")
    return "\n".join(out)


def main() -> None:
    d3 = pd.read_csv(D / "dist_d3.csv")
    d5 = pd.read_csv(D / "dist_d5.csv")
    print("## Fault distance of the 8928 depth-6 schedules\n")
    print("d = 3 (r = 3, detection-set budget 4), all 8928:\n")
    print(md(d3.groupby(["sd6_X", "sd6_Y", "hyb_X", "hyb_Y"]).size()
             .reset_index(name="schedules")))
    print("\nd = 5 (r = 3, budget 3), the 1690 that keep distance 3 in all four columns at d = 3:\n")
    print(md(d5.groupby(["sd6_X", "sd6_Y", "hyb_X", "hyb_Y"]).size()
             .reset_index(name="schedules")))
    h3 = pd.read_csv(D / "hex_d3.csv")
    h5 = pd.read_csv(D / "hex_d5.csv")
    print("\nHex-only sigma pairs (hyb, links are pair measurements) not in "
          "`enumerate_depth(6)`: 1128. d = 3:\n")
    print(md(h3.groupby(["hyb_X", "hyb_Y"]).size().reset_index(name="pairs")))
    print("\nd = 5 for the 155 with distance 3 at d = 3:\n")
    print(md(h5.groupby(["hyb_X", "hyb_Y"]).size().reset_index(name="pairs")))

    df = pd.concat([pd.read_csv(D / f) for f in
                    ("ler_mwpm.csv", "ler_diag.csv", "ler_xzzx.csv", "ler_bp.csv")])
    df["cell"] = [f"{100 * a:.2f} ± {100 * b:.2f}" for a, b in zip(df.p_L, df.stderr)]
    print("\n## Logical error rates (%, r = d, one-sigma binomial errors)\n")
    for key, g in df.groupby(["code", "comp", "noise", "decoder"], sort=False):
        t = g.pivot_table(index=["logical", "d"], columns="p", values="cell",
                          aggfunc="first").reset_index()
        t.columns = [c if isinstance(c, str) else f"p={100 * c:.1f}%" for c in t.columns]
        print(f"\n**{key[0]} {key[1]} / {key[2]} / {key[3]}**\n")
        print(md(t))
    ct = crossing_table(df)
    ct["p_cross (%)"] = [("< grid" if x == -1 else "> grid" if x == float("inf")
                          else f"{100 * x:.2f} ± {100 * e:.2f}")
                         for x, e in zip(ct.p_cross, ct.err)]
    print("\n## Pairwise crossings\n")
    print(md(ct[["code", "comp", "noise", "logical", "decoder", "d_pair", "p_cross (%)"]]))


if __name__ == "__main__":
    main()
