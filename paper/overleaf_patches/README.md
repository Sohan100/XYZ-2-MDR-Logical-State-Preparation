# Patches for the Overleaf project (Results.tex and the appendix)

The Overleaf connector could not run in the session that wrote these (claudeleaf needs `npx`, which is not on Perlmutter's PATH), so they are LaTeX snippets to merge by hand into the live files. Each file starts with comments that say exactly where it goes. All four compile together in a revtex4-2 test document with the paper's macros (`\XYZ`, `\Xb`, `\Yb`) and colours (`xyzcode` E11D48, `csscode` A8A29E, `xzzxcode` 7C3AED, `ink` 292524, `stone` 57534E, `grid` E7E5E4), with pgfplots and the groupplots library.

| file | where | what |
|---|---|---|
| `ranking.tex` | sec:ranking, replaces the comment line starting `% Equal-qubit comparison` | paragraph on the equal-qubit comparison below threshold, Eq. `eq:eps_per_round`, and the three-panel `figure*` `fig:equal_qubit` (SD6, η = 100, pure Z at p = 10⁻³) |
| `hw_relation.tex` | sec:hw_relation | replaces the first two paragraphs; adds Table `tab:hw_corners` right after Table IV (`tab:hw`) |
| `gap.tex` | sec:gap | replaces the four paragraphs and `fig:gap` (adds the bars "no idle errors" and "noiseless links, no idle errors", a dotted line for XZZX without idle errors, new caption) |
| `app_fits.tex` | app:fits | two sentences on the sampling of the sub-threshold runs |

Sources: `docs/hetenyi_wootton_check.md`, `docs/gap_diagnostic.md`, `docs/equal_qubit_comparison.md` and the data in `docs/data/campaign/hw/`, `gap/` and `eq/`.

The live abstract, introduction and conclusion could not be read from here. Check their claims against these results, in particular any sentence that says:
- XYZ² ranks above the honeycomb code with gates (at the pure gate corner the two tie);
- XYZ² has an advantage below threshold or at equal qubit count (it beats the CSS code under biased noise only, and never the XZZX code);
- part of the gap to XZZX is unexplained (noisy links and idling account for all of it within errors).

`preview/preview.pdf` typesets the four patches with the paper's preamble, macros and colours (built from `preview/preview.tex` with `latexmk -pdf`). It is not the whole paper: the live Overleaf files could not be read here, so references into the rest of the paper point to placeholders, and `preview.bib` adds the two entries the patches cite (`hetenyi2024spin`, `criger2016fourtwotwo`) in case the live bibliography lacks them.
