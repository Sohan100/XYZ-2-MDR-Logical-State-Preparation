"""PNG previews of the main results for docs/figures.

Usage: python paper/analysis/docs_figures.py

fig_old_vs_ft.png compares the old lookup-table protocol with FT-MDR under
uniform circuit noise. fig_quantinuum.png and fig_rounds.png are PNG copies of
fig_quantinuum and fig_rounds_pth of the paper, saved while helios_plots.py
and rounds_plots.py run.
"""
import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import make_plots as M  # noqa: E402  (rcParams and palette)
from common import DATA, ROOT, load_fits, load_sweeps, read  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

DOCS = ROOT / "docs" / "figures"
DPI = 220


def fig_old_vs_ft():
    old = read(DATA / "res_old.csv")
    old = old[old.protocol == "old_full_mdr_lookup"].rename(columns={"p": "value"})
    df = load_sweeps()
    sd6 = df[(df.noise == "sd6") & (df.decoder == "mwpm") & (df.final == "frame")]
    panels = [(old, r"Lookup table, $r=3$"),
              (sd6[sd6.rounds == sd6.d], r"FT-MDR, $r=d$"),
              (sd6[sd6.rounds == 1], r"FT-MDR, $r=1$")]
    th = load_fits().get(("sd6", "mwpm"))
    fig, axes = plt.subplots(1, 3, figsize=(M.FULL, 2.25), sharey=True)
    for k, (ax, (s, title)) in enumerate(zip(axes, panels)):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.plot([2e-4, 2e-2], [2e-4, 2e-2], color=M.GUIDE, lw=0.8, ls=":", zorder=0)
        for d in (3, 5, 7, 9):
            M.series(ax, s[(s.value >= 2e-4) & (s.value <= 1.25e-2)], d, label=f"$d={d}$", ms=2.6, lw=0.9)
        ax.set_title(title, loc="left", color=M.INK)
        ax.set_xlabel(r"physical error rate $p$")
        ax.set_xlim(2e-4, 1.5e-2)
        M.panel_label(ax, f"({'abc'[k]})")
    if th is not None:
        axes[1].axvline(th[0], color=M.DEC_COL["mwpm"], lw=0.7, ls="--")
        M.place_text(axes[1], rf"$p_{{\mathrm{{th}}}}={100 * th[0]:.2f}\%$", fontsize=6.5, color=M.INK)
    axes[0].set_ylabel(r"logical error rate $p_L$")
    axes[0].set_ylim(1e-7, 0.6)
    h = [Line2D([], [], color=M.RAMP[d], marker=M.MARK[d], ms=2.6, lw=0.9, label=f"$d={d}$") for d in (3, 5, 7, 9)]
    axes[0].legend(handles=h, loc="lower right", fontsize=6.5)
    fig.tight_layout(w_pad=0.6)
    M.guide_label(axes[0], r"$p_L=p$", [1.5e-3, 3e-3, 6e-4], below=True)
    fig.savefig(DOCS / "fig_old_vs_ft.png", dpi=DPI)
    plt.close(fig)


def png_copies(script, names):
    """Run a plotting script and save PNG copies of the figures named in `names`."""
    orig = Figure.savefig

    def savefig(self, fname, *args, **kwargs):
        orig(self, fname, *args, **kwargs)
        stem = Path(str(fname)).stem
        if stem in names:
            orig(self, DOCS / names[stem], dpi=DPI)

    Figure.savefig = savefig
    try:
        runpy.run_path(os.path.join(os.path.dirname(__file__), script), run_name="__main__")
    finally:
        Figure.savefig = orig


if __name__ == "__main__":
    DOCS.mkdir(parents=True, exist_ok=True)
    fig_old_vs_ft()
    png_copies("helios_plots.py", {"fig_quantinuum": "fig_quantinuum.png"})
    png_copies("rounds_plots.py", {"fig_rounds_pth": "fig_rounds.png"})
    print("figures in", DOCS)
