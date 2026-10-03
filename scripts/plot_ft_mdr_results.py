"""
plot_ft_mdr_results.py
----------------------------------------------------------------------------
Plot logical error versus noise strength from `run_ft_mdr_sweep.py` output.

One panel per (noise, rounds, init, final, decoder) group, one line per
distance. Points with fewer than `--min-errors` logical errors are dropped.

    python scripts/plot_ft_mdr_results.py \
        --csv data/ft_mdr/ft_mdr_results.csv --out data/plots/ft_mdr.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAMP = ["#F97316", "#E11D48", "#C026D3", "#6D28D9", "#4C1D95"]  # the distance ramp of the paper
MARKERS = ["o", "s", "^", "D", "v"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path,
                    default=ROOT / "data" / "ft_mdr" / "ft_mdr_results.csv")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "data" / "plots" / "ft_mdr" / "ft_mdr.png")
    ap.add_argument("--min-errors", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["rounds_label"] = df["rounds"].astype(str)
    keys = ["noise", "init", "final", "decoder"]
    df["round_mode"] = np.where(df["rounds"] == df["d"], "r=d",
                                "r=" + df["rounds"].astype(str))
    groups = list(df.groupby(keys + ["round_mode"]))
    fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 3.6),
                             sharey=True, squeeze=False)
    for ax, (key, sub) in zip(axes[0], groups):
        for k, d in enumerate(sorted(sub["d"].unique())):
            s = sub[(sub["d"] == d) & (sub["errors"] >= args.min_errors)]
            s = s.sort_values("value")
            if s.empty:
                continue
            err = np.sqrt(s["errors"]) / s["shots"]
            ax.errorbar(s["value"], s["p_L"], yerr=err,
                        color=RAMP[k % len(RAMP)],
                        marker=MARKERS[k % len(MARKERS)], ms=4.5, lw=1.6,
                        label=f"d = {d}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, color="#E7E5E4", lw=0.6)
        noise, init, final, decoder, mode = key
        ax.set_title(f"{noise}, {mode}, init={init}, final={final}",
                     fontsize=9, loc="left")
        ax.set_xlabel("p (two-qubit error rate for helios/h2)")
        ax.legend(fontsize=8, frameon=False)
    axes[0][0].set_ylabel("logical error rate of |+_L>")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
