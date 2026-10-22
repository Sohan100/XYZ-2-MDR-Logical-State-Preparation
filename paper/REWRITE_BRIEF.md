# Brief for the paper rewrite: new results on the XYZ² code (Perlmutter, Oct 14 to 22, 2026)

You are rewriting the paper "Fault-tolerant state preparation and hierarchical decoding for the $XYZ^2$ hexagonal stabilizer code" (Ghosh, Kim, Kairys) to make it cohesive, human and direct. While that was going on, another Claude session ran new simulations on NERSC Perlmutter to answer referee-style questions about how XYZ² compares with other codes. This brief gives you everything that session found.

Treat it as input, not as a plan. You decide where each result goes, what to merge or cut, and how to say it. Two things are fixed:
- **The numbers.** Use them as given, with their uncertainties, and do not invent new ones. Every number has a source file in the repository below.
- **The verdict.** The paper must not claim an XYZ² advantage that these data do not support. The data are negative, and the author wants the negative result stated plainly.

## Where everything is

- **Repository:** https://github.com/Sohan100/XYZ-2-MDR-Logical-State-Preparation (branch `main`). `docs/README.md` indexes every document.
- **Ready-made LaTeX** for four places in the paper, in the paper's macros and colours, in `paper/overleaf_patches/`:
  - `hw_relation.tex`: two paragraphs plus a new table for the subsection on earlier XYZ² results;
  - `ranking.tex`: the equal-qubit paragraph, an equation and a three-panel figure;
  - `gap.tex`: four paragraphs and a new gap figure;
  - `app_fits.tex`: two appendix sentences.

  `paper/overleaf_patches/README.md` says where each one goes, and `paper/overleaf_patches/preview/preview.pdf` shows them typeset. The full text of all four is at the end of this brief. Rewrite them freely; they were written before your rewrite started.
- **Result documents:**
  - `docs/hetenyi_wootton_check.md`
  - `docs/gap_diagnostic.md`
  - `docs/equal_qubit_comparison.md`
  - `docs/xyz2_usefulness.md` (the final verdict)
  - `docs/noncss_comparison.md`
  - `docs/fair_comparison.md` (the comparison rules)
- **Data:** `docs/data/campaign/hw/`, `gap/` and `eq/`, plus `docs/data/campaign/thresholds.csv` and `competitors_table.csv`.
- **Overleaf:** the project id is `6843ad9c5df2ccc67d51ec05`. The author's rule for Overleaf access: work only through the Overleaf connector (claudeleaf and the Overleaf git bridge) and pass that project id on every call. Never use a browser, git, curl or a script against Overleaf, and never open `overleaf-mcp-server/.env`.
- **Labels the patches assume exist in the live paper:**
  - Table IV is `tab:hw`.
  - The sections are `sec:hw_relation`, `sec:ranking`, `sec:gap` and `sec:cssmap`, and the appendix is `app:fits`.
  - The citations are `hetenyi2024spin` and `criger2016fourtwotwo`. Add them to the bibliography if they are missing:
    - Hetényi and Wootton, PRA 109, 032433 (2024), arXiv:2306.17786;
    - Criger and Terhal, QIC 16, 1261 (2016), arXiv:1604.04062.

## The bottom line

XYZ² has no established use on any hardware model we simulated, either as a memory or for single-round MDR state preparation.
- **Thresholds:** XYZ² never beats the XZZX surface code under circuit-level noise.
- **Equal qubit count:** below threshold it loses to the CSS code under depolarizing noise, and to XZZX under biased noise.
- **Why:** its threshold gap to XZZX comes from idling and noisy link checks.
- **Only lead:** one idealized spin-qubit corner that no real device matches.

What still stands is the paper's actual contribution:
- the fault-tolerant MDR protocol, with fault distance d + 1;
- the decoders: hierarchical correlated matching, which raises the threshold by 22% under SD6 and 53% at η = 100, and coset free-energy decoding;
- the threshold-versus-rounds results;
- the Helios projection: one round at d = 5 gives about 8×10⁻⁵, below the physical SPAM error.

The code comparison is a clear negative result and should read as one.

## 1. The corners of Hetényi and Wootton's spin-qubit model

**Their model.** Hetényi and Wootton (arXiv:2306.17786) fit a plane to the threshold surface of a spin-qubit model with three error rates: a gate error p_G, an idle error p_T with bias η_T = p_Z/(p_X+p_Y) = 20, and a combined reset-and-readout error p_RR. The intercepts of that plane (their Table I) are thresholds with only one kind of error switched on.

**What we did.** We measured the same corners in our own circuits:
- one error parameter p per corner;
- r = d rounds;
- finite-size fits over d = 11 to 21 (12 to 20 for the honeycomb code);
- the weaker logical basis.

Thresholds in %:

| code | gates: theirs / ours MWPM / ours best decoder | gates + noisy 1q (MWPM) | idling: theirs / MWPM / best | readout: theirs / MWPM / best |
|---|---|---|---|---|
| Surface (CSS) | 0.82 / 1.15(1) / 1.40(5) | 0.433(3) | 3.94 / 3.6(4) / 3.8(4) | 14.5 / 9.8(4) / 10.5(6) |
| XZZX | 0.37 / 1.15(2) / 1.39(1) | 0.438(4) | 15.1 / 9.8(3) / 10(2) | 12.9 / 9.8(3) / 10.3(4) |
| XYZ² | 0.465 / 0.89(2) / 1.10(4) | 0.326(8) | 4.2 / 0.88(1) / 1.330(6) | 16.5 / 10.6(3) / 10.6(5) |
| Honeycomb Floquet | 0.43 / 0.88(1) / 1.081(8) | 0.293(2) | 1.16 / 0.392(3) / 0.51(1) | 0.99 / 1.007(7) / 1.72(3) |

"Best decoder" means the best of matching, correlated matching and belief-matching in each basis, then the weaker basis.

**Gate corner.**
- Our thresholds are higher than theirs for every code.
- Both studies rank the surface code above XYZ². The XYZ²/CSS ratio is 0.770(17) for us and 0.57 for them.

**Their low XZZX value comes from noisy single-qubit gates.**
- Our XZZX circuit uses native controlled Paulis. At the gate corner it has the same detector error model as CSS, so both reach 1.15(2)%.
- Their XZZX circuit adds single-qubit basis changes on the data qubits, each as noisy as a two-qubit gate.
- Adding a depolarizing error p on both qubits of every two-qubit gate drops our XZZX to 0.438(4)%. Relative to our CSS without those errors that is 0.380(6), against 0.45 for their XZZX/CSS ratio.
- When every code uses the same gates, XZZX ranks above XYZ², with and without single-qubit errors.

**Honeycomb.**
- At the gate corner the honeycomb code ties XYZ²: 0.88(1)% against 0.89(2)% with matching, and 1.081(8)% against 1.10(4)% with the best decoders. Hetényi and Wootton put XYZ² slightly above, 0.465% against 0.43%.
- With noisy single-qubit gates XYZ² is above the honeycomb, 0.326(8)% against 0.293(2)%.

**Readout corner.**
- The honeycomb code is lowest in both studies: 1.007(7)% for us, about 10% for the other three codes.
- With matching, XYZ² (10.6(3)%) is above CSS (9.8(4)%). With the best decoders they tie.

**Idle corner: the ranking flips.**
- XYZ² is below CSS in our circuits (0.88(1)% against 3.6(4)%) and above it in theirs (4.2% against 3.94%).
- The cause is where idle errors act. Ours act on every qubit that waits in a gate layer, and XYZ²'s link ancillas wait in 4 of the 6 layers of a round. Theirs act only on data qubits while ancillas are read out and reset.
- With their placement (our diagnostic model `hw_tr`), the d = 7/9 crossings are 10.5(6)% for XYZ² and 6.6(2)% for CSS. That restores their order.

**Other differences in their circuit:**
- their XYZ² uses ZZ links, measured in a second readout window;
- ancilla pairs;
- SWAP gates for the plaquettes;
- a readout error that includes two-qubit depolarizing noise.

All our `hw_*` models use XX links.

**Combined rates.** Their combination rule, 1/p = Σ 1/p_i, applied to our corners gives XYZ² 0.909(26)% against XZZX 1.150(13)%. This is an inference, not a measurement.

**Rankings that differ from their study:**
- The honeycomb ties XYZ² at the gate corner.
- XYZ² falls below CSS at the idle corner, explained by idle placement.
- XYZ² and CSS tie at the readout corner with the best decoders.

## 2. Where the threshold gap to XZZX comes from

All values are memory thresholds in %, under SD6, at r = d, over d = 11 to 21, with the best decoder in the weaker basis.

| model | XYZ² (best) | XYZ² (MWPM) | XZZX (best) |
|---|---|---|---|
| SD6 | 0.487(7) | 0.400(6) | 0.813(20) |
| noiseless link checks (diagnostic) | 0.626(8) | 0.556(6) | |
| every link measured twice per round | 0.558(21) | 0.404(5) | |
| every link measured three times per round | 0.526(8) | 0.360(4) | |
| hybrid: links as EM3 pair measurements, plaquettes with gates | 0.575(6) | 0.470(5) | |
| no idle errors | 0.854(6) | 0.691(6) | 1.103(9) |
| noiseless links and no idle errors | 1.12(3) | 0.99(1) | |

The hybrid circuit has fault distance d (checked at d = 3), because a faulty link measurement puts a two-qubit error on one block.

Share of the gap each change closes, (T_variant − T_SD6)/(T_reference − T_SD6):
- **Against XZZX without idle errors:**
  - idle errors: 0.60(1);
  - link noise: 0.23(2);
  - both together: 1.02(5), so the gap closes completely within errors.
- **Against XZZX under SD6:** noiseless links close 0.43(4). This replaces "about two fifths" in the current text.
- **Matching alone gives the same picture:** 0.691(6)% and 0.99(1)% for the two idle-free XYZ² models, against 0.93(2)% for XZZX without idle errors.
- **The two effects are super-additive**, at about 3.5σ.

What this means:
- **Weight-6 checks:** at the threshold level nothing is left for the weight-six hexagons. The concern in arXiv:2112.06036, that noise on the weight-6 checks drives the gap, is not supported at threshold level.
- **Where the cost shows instead:** below threshold. XYZ² needs about twice the qubits for the same distance.
- **Schedules:** no schedule or compilation we tried closes the gap under SD6.
- **Idling counts:**
  - XYZ² has 8 ticks per round against 6.
  - At 99/97 qubits it has 254 gate-layer idle sites per round against 52 for XZZX. Of these, 130 are on data qubits, 100 on link ancillas that wait through 4 of 6 gate layers, and 24 on plaquette ancillas.
- **Context:** Criger and Terhal found that the threshold of the [[4,2,2]]-concatenated toric code depends on the schedule of its weight-eight checks. Their shortest schedule reached about 0.41%, against about 0.6% for the toric code.

## 3. Equal qubit count below threshold

**Method.**
- **Runs:** memories with r = 3d rounds at p = 10⁻³, 2×10⁻³ and 3×10⁻³, under three noise models: SD6, Z-biased with η = 100, and pure Z.
- **Error per round:** ε from p_L = ½[1 − (1 − 2ε)^r], numbered as equation `eq:eps_per_round` in the patch.
- **Decoder and basis:** one decoder per code and noise model, correlated matching (it gives the lowest error at the largest distances), and the weaker logical basis.
- **Qubit pairs:**

  | XYZ² | surface code |
  |---|---|
  | d = 5, 99 qubits | d = 7, 97 qubits |
  | d = 9, 323 qubits | d = 13, 337 qubits |
  | d = 11, 483 qubits | d = 15, 449 qubits |

- **Stopping rule:** each point stops at 100 logical errors (300 at the pairs) or 10⁹ shots. A point with no errors gives the 95% bound 3/N.
- **Intervals:** 68% Wilson.

Ratio of XYZ²'s error per round to the competitor's, at 99/97, 323/337 and 483/449 qubits:

| noise, competitor | p = 10⁻³ | p = 2×10⁻³ | p = 3×10⁻³ |
|---|---|---|---|
| SD6, against CSS | 11.1 / 693 / – | 8.5 / 118 / 332 | 6.5 / 76 / 129 |
| η = 100, against CSS | 0.21 / 0.05 / 0.12 | 0.25 / 0.08 / 0.03 | 0.29 / 0.11 / 0.05 |
| η = 100, against XZZX | 10.4 / – / – | 10.5 / 154 / – | 10.7 / 170 / 309 |
| pure Z, against XZZX | 17.3 / – / – | 16.1 / 154 / 82 | 17.4 / 367 / 850 |

A dash means the competitor saw no logical error, so only a bound exists. Under SD6, CSS stands for XZZX and XY, which share its detector error model.

**SD6.**
- XYZ²'s error per round is 11(1) times CSS's at 99 qubits.
- At about 330 qubits it is about 700 times, within a factor of two, because the CSS point rests on 2 errors. Its 3σ bound is still at least 107×.
- Using each code's weaker basis at every point gives 7.5 to 470×.

**Λ = ε(d)/ε(d+2):**

| noise | p | XYZ² | CSS |
|---|---|---|---|
| SD6 | 2×10⁻³ | 4.04(20) | 6.56(14) |
| SD6 | 10⁻³ | 11.0(4) | 13.8(12) |
| SD6 | 3×10⁻³ | 2.34(9) | 3.98(5) |

Each step in d costs XYZ² twice as many qubits.

**Qubits for an error of 10⁻⁶ per round at p = 10⁻³:**
- SD6: XYZ² 244, CSS 120.
- η = 100: XYZ² 134, CSS 192, XZZX 82.

**Biased noise.**
- **Against CSS:** XYZ² beats CSS at equal qubits under biased noise, by 3 to 5× at 99 qubits and 8 to 60× at the larger pairs. With matching alone it falls behind CSS under bias too, so that gain needs a decoder that reads the gauge level.
- **Against XZZX:** XZZX beats XYZ² by 10 to 17× at 99 qubits and 80 to 850× at the larger pairs. With belief-matching for XYZ² against correlated matching for XZZX it is still 3.4 to 6.9× at 99/97 and 28 to 51× at 323/337.

**Time boundaries.** Taking ε from runs with r = d and r = 3d removes the time boundaries. It changes ε by −46% to +17% and changes none of these comparisons.

**Conclusion.** Equal qubit counts do not reverse the threshold ranking.

## 4. When is XYZ² useful? Consensus of a multi-agent review

**How it was reached.**
- 22 agents each examined the data through one lens.
- A synthesis followed, then eight review rounds of 10 adversarial reviewers each.
- Disputed claims were revised by one agent and checked by two verifiers.
- Final agreement: 35 of 39 items (38 claims plus the overall answer) at ≥80% of reviewers. The four below 80% are disputes about details, such as whether six or seven pilot leads survive the strictest error model. No reviewer disputes the verdict.

The full document is `docs/xyz2_usefulness.md`.

**Short answer.**
- **No threshold win.** Of 107 head-to-head (model, round count) comparisons, XYZ² is nominally below the best competitor in 103. With stat plus syst errors that is 0 wins, 22 ties and 85 losses. The four nominal exceptions are ties or fit artifacts in non-hardware or crosstalk models.
- **Gate hardware:** SD6 and SI1000 give 0.60 and 0.83 of the best surface code.
- **Bias-preserving Z bias:** at r = d, 0.64, 0.51 and 0.42 of XZZX for η = 10, η = 100 and pure Z. XZZX leads by at least 6.9σ at every r.
- **Trapped ions without crosstalk:** about 0.76 of the best surface code. With measurement crosstalk no code has a threshold.
- **Pair-measurement hardware (EM3):**
  - XYZ² beats this repository's cat-state compilations of the surface codes by 1.29 to 1.36×. The honeycomb code is 6.1× higher there.
  - The published 3aux pair-measurement surface code (arXiv:2310.12981, not simulated) is better than XYZ².
  - So "XYZ² beats every surface code on pair-measurement hardware" is not a valid claim.
- **Gates plus pair measurements (hybrid):** 0.30 to 0.61 of the honeycomb.
- **Relative wins:** these hold only over codes that are themselves beaten on that hardware.
  - Over the honeycomb under gate models, 1.6 to 4.5×, except a tie with gate errors alone.
  - Over the square CSS code under bias.
- **r = 1 MDR preparation**, the protocol of the paper:
  - Over 13 hardware-motivated models the threshold in the MDR basis gives 10 losses, 3 ties and no win. Under SD6 it is 1.960(73)% against XZZX 2.417(47)%. Under bias XYZ² stays at about 1.97 to 1.99% while XZZX reaches 3.2 to 3.5%.
  - On every tested footprint a surface-code patch that fits prepares the state better. On 35 SI1000 qubits a 6×3 CSS patch is about 18× better. At the Helios and H2 device points CSS is 2.8× and 4.7× better.
- **The one established lead is idealized.**
  - The corner is `hw_tr`: perfect syndromes, η = 20 dephasing on data qubits only during ancilla readout and reset, and no gate-layer idling.
  - There, at p_T = 4% with 99/97 qubits, r = 6d and belief-matching, XYZ²'s error per round is 0.384(39) of XZZX's.
  - It reverses for r = 1 preparation, where XYZ² is 18.5(7)× worse.
  - Device-like variants remove it. Gate-layer dephasing of 2% of p_T gives 1.02(7). The surface code's QEC-optimal readout time gives 1.72(16)× worse.
- **Phenomenological noise with η = 10 (not hardware)** is a threshold tie: 5.539(54)% against 5.537(55)%. It needs BP-class decoding, and no below-threshold lead is established there.

**Literature.**
- arXiv:2112.06036 places XYZ²'s advantage at code capacity with η = 10 and near-ML decoding, against XZZX in its d frame: about 1.7 to 2.7×. Its threshold is level with XZZX there. That advantage does not survive circuit-level noise in our data.
- Hetényi and Wootton's combined ordering, XYZ² above XZZX, is not reproduced once XZZX uses native gates.

**Labelling error in the existing docs.** The campaign's "biased" noise models are bias-preserving. Bias with non-bias-preserving gates is untested.

**Untested** (a use is not excluded only here):
- gate-layer dephasing between 0 and 2% of p_T on a spin device;
- each code at its own optimal readout time;
- η = 10 at p_T = 4 to 6%;
- Hetényi–Wootton's two-window 4d²:3d² layout;
- a real-time BP-class decoder;
- strong Z bias below p = 10⁻³;
- erasure and leakage;
- neutral atoms;
- lattice surgery.

## 5. What this means for the paper

**Statements the data no longer support.** Check the abstract, introduction, results and conclusion for any of these:
1. **XYZ² ranks above the honeycomb code with gates.** At the pure gate corner they tie. XYZ² is ahead only in the gate models that also have idling and readout noise, and there the surface codes are better than both.
2. **XYZ² has an advantage below threshold, or at equal qubit count.** It beats the CSS code under bias, only with a gauge-reading decoder, and never XZZX.
3. **Part of the gap to XZZX is unexplained, or comes from the weight-six checks.** Idling plus link noise account for all of it within errors.
4. **XYZ² beats every surface code on pair-measurement hardware.** That holds only against this repository's cat-state compilations.
5. **XYZ²'s bias tolerance is an advantage over the surface codes.** Under bias-preserving noise XZZX has a 1.5 to 2.4× higher threshold.
6. **XZZX does poorly on spin qubits.** That comes from noisy single-qubit basis changes, not from the code.

**Suggested framing (your call).**
- The paper's contribution is the MDR protocol, its decoders and the hardware projection.
- The code comparison is an honest negative result. It belongs in one place and should be said plainly:
  - XYZ² loses;
  - it loses because of idling and link noise and the doubled qubit count;
  - its only lead is an idealized corner.
- The equal-qubit figure and the gap figure carry that argument. The Hetényi–Wootton table explains why an earlier study ranked XYZ² higher.

## 6. Conventions the new text follows

The author set these conventions for the new paragraphs. Keep or reconcile them with your rewrite.
- **Voice:** "we" sentences and plain declaratives.
- **Punctuation:** no em dashes and no semicolons.
- **Ranges:** written "X to Y".
- **Uncertainties:** in the form 0.91(5)\%.
- **Source layout:** one line per paragraph in the LaTeX source.
- **Macros:** `\XYZ{}`, `\Xb`, `\Yb`.
- **Figure colours:** pgfplots, using xyzcode E11D48, csscode A8A29E, xzzxcode 7C3AED, ink 292524, stone 57534E and grid E7E5E4.

## 7. The four patches, verbatim

Each file starts with comments saying exactly what it replaces.

### `paper/overleaf_patches/hw_relation.tex`

```latex
% Patch for Results.tex, subsection "Relation to earlier \XYZ{} results" (\label{sec:hw_relation}).
% 1. The two paragraphs below replace the first two paragraphs of the subsection, which begin
%    "The circuits, noise models and decoders of our study ..." and "We tested this explanation in our own circuits ...".
% 2. The table below goes right after Table IV (\label{tab:hw}).
% Data: docs/data/campaign/hw/summary.csv, thresholds.csv and diag_hw_tr_centers.csv; docs/hetenyi_wootton_check.md.

% ---------------------------------------------------------------- paragraph 1 (replaces the first paragraph)
The circuits, noise models and decoders of our study and of that of Het\'enyi and Wootton differ (Table~\ref{tab:hw}), so we compare rankings rather than values. They fit a plane to the threshold surface of a spin-qubit model with three error rates, a gate error, an idle error with bias $\eta_T=p_Z/(p_X+p_Y)=20$ and a combined reset and readout error~\cite{hetenyi2024spin}. The intercepts of this plane are thresholds with only one kind of error switched on, so at the gate corner idle, reset and readout errors are all off. We measured the same corners in our own circuits with one error parameter $p$ each, matching, $r=d$ rounds and finite-size fits over $d=11$ to $21$ (Table~\ref{tab:hw_corners}). At the gate corner our thresholds are higher than theirs for every code, and in both studies the rotated surface code ranks above the \XYZ{} code. The ratio of the two thresholds is 0.770(17) in our circuits and 0.57 in theirs. The two studies rank the XZZX code differently. Our XZZX circuit uses native controlled Paulis and has the same detector error model as our CSS circuit at the gate corner, so the two codes reach the same threshold, 1.15(2)\%. Their XZZX circuit adds single-qubit basis changes on the data qubits, each as noisy as a two-qubit gate, and reaches 0.37\%. The honeycomb Floquet code reaches 0.88(1)\% at the gate corner, the same as the 0.89(2)\% of the \XYZ{} code, and the two codes also agree with the best of matching, correlated matching and belief-matching, 1.081(8)\% against 1.10(4)\%. Het\'enyi and Wootton place the \XYZ{} code slightly above the honeycomb code, with 0.465\% against 0.43\%.

% ---------------------------------------------------------------- paragraph 2 (replaces the second paragraph)
We then added a depolarizing error of probability $p$ on both qubits of every two-qubit gate, which models single-qubit gates as noisy as two-qubit gates. The XZZX threshold falls to 0.438(4)\%, near the 0.37\% of Het\'enyi and Wootton. Relative to our CSS threshold without these errors it falls to 0.380(6), against 0.45 for the ratio of their XZZX and CSS values. The noisy single-qubit gates therefore account for the low XZZX value of the spin-qubit study. When every code uses the same gates, the XZZX code ranks above the \XYZ{} code with and without single-qubit errors, as in the rest of our comparison. With single-qubit errors the \XYZ{} code also ranks above the honeycomb code, 0.326(8)\% against 0.293(2)\%. At the readout corner the honeycomb code has the lowest threshold in both studies, 1.007(7)\% in ours against about 10\% for the other three codes. With matching the \XYZ{} code reaches 10.6(3)\% there, against 9.8(4)\% for the CSS code, but with the best of our three decoders the \XYZ{} code and the surface codes agree within their errors. At the idle corner the \XYZ{} code ranks below the CSS code in our circuits, 0.88(1)\% against 3.6(4)\%, while it ranks above it in theirs, 4.2\% against 3.94\%. Our idle error acts on every qubit that waits in a gate layer, and the link ancillas of the \XYZ{} code wait in four of the six layers of a round. Their idle error acts only on the data qubits while the ancillas are read out and reset. With the idle error placed that way, the crossing of $d=7$ and $9$ is 10.5(6)\% for the \XYZ{} code and 6.6(2)\% for the CSS code, which restores their order.

% ---------------------------------------------------------------- table (insert right after Table IV, \label{tab:hw})
\begin{table}[t]
  \setlength{\tabcolsep}{1.6pt}
  \footnotesize
  \begin{ruledtabular}
  \begin{tabular}{lccccccc}
    & \multicolumn{3}{c}{Gates} & \multicolumn{2}{c}{Idling} & \multicolumn{2}{c}{Readout} \\
    Code & HW & Ours & Ours, 1q & HW & Ours & HW & Ours \\
    \hline
    Surface (CSS) & 0.82 & 1.15(1) & 0.433(3) & 3.94 & 3.6(4) & 14.5 & 9.8(4) \\
    XZZX & 0.37 & 1.15(2) & 0.438(4) & 15.1 & 9.8(3) & 12.9 & 9.8(3) \\
    \XYZ{} & 0.465 & 0.89(2) & 0.326(8) & 4.2 & 0.88(1) & 16.5 & 10.6(3) \\
    Honeycomb & 0.43 & 0.88(1) & 0.293(2) & 1.16 & 0.392(3) & 0.99 & 1.007(7) \\
  \end{tabular}
  \end{ruledtabular}
  \caption{Thresholds in percent at the corners of the threshold surface of Het\'enyi and Wootton, where only one kind of error is switched on. HW: the intercepts of their fitted plane~\cite{hetenyi2024spin}, with $\eta_T=20$, $d=11$ to $17$ and matching. Ours: matching, $r=d$ rounds, finite-size fits over $d=11$ to $21$ ($d=12$ to $20$ for the honeycomb code), weaker basis. Gates: a two-qubit depolarizing error of probability $p$ after every two-qubit gate. Ours, 1q: in addition a depolarizing error of probability $p$ on both qubits of every two-qubit gate. Idling: a Pauli error of total probability $p$ with $p_Z/(p_X+p_Y)=20$ on every qubit that waits in a gate layer and on the data qubits while the ancillas are reset and read out. Readout: every measurement outcome flipped with probability $p$.}
  \label{tab:hw_corners}
\end{table}
```

### `paper/overleaf_patches/ranking.tex`

```latex
% Patch for Results.tex, subsection "Ranking" (\label{sec:ranking}).
% Replace the comment line that starts "% Equal-qubit comparison" with the paragraph (and its equation) and the figure* below.
% Data: docs/data/campaign/eq/ (pairs.csv, lambda.csv, targets.csv, boundary.csv, eq.md), docs/equal_qubit_comparison.md.

% ---------------------------------------------------------------- paragraph
We also compare the codes below threshold at equal numbers of qubits, where the original proposals place the advantage of the \XYZ{} code. We run memories with $r=3d$ rounds at $p=10^{-3}$, $2\times10^{-3}$ and $3\times10^{-3}$ and convert the logical error rate $p_L$ of each run into an error per round $\epsilon$ through
\begin{equation}
p_L=\tfrac{1}{2}\left[1-(1-2\epsilon)^{r}\right].
\label{eq:eps_per_round}
\end{equation}
For each code and noise model we use one decoder at every point, correlated matching, which gives the lowest error at the largest distances, and the weaker logical basis. The \XYZ{} code at $d=5$, 9 and 11 has 99, 323 and 483 qubits, close to the 97, 337 and 449 qubits of the surface codes at $d=7$, 13 and 15 (Fig.~\ref{fig:equal_qubit}). Under SD6 at $p=10^{-3}$ the error per round of the \XYZ{} code is 11(1) times that of the CSS code at 99 qubits and about 700 times, within a factor of two, at about 330 qubits. At $p=2\times10^{-3}$ and $3\times10^{-3}$ the factor grows from 9 and 7 at 99 qubits to 330 and 130 at about 470 qubits. The suppression factor $\Lambda=\epsilon(d)/\epsilon(d+2)$ at $p=2\times10^{-3}$ is 4.0(2) for the \XYZ{} code against 6.6(1) for the CSS code, and each step in $d$ costs the \XYZ{} code twice as many qubits. Under biased noise the \XYZ{} code beats the CSS code at equal qubits, by a factor of 3 to 5 at 99 qubits and 8 to 60 at the larger pairs for $\eta=100$ and pure $Z$ noise. The XZZX code beats the \XYZ{} code by a factor of 10 to 17 at 99 qubits and 80 to 850 at the larger pairs. With matching alone the \XYZ{} code also falls behind the CSS code under biased noise, so its gain over the CSS code relies on a decoder that reads the gauge level. At $p=10^{-3}$ an error of $10^{-6}$ per round takes 244 qubits for the \XYZ{} code against 120 for the CSS code under SD6, and 134 against 192 for the CSS code and 82 for the XZZX code at $\eta=100$. Taking $\epsilon$ from runs with $r=d$ and $r=3d$, which removes the contribution of the time boundaries, changes it by $-46\%$ to $+17\%$ and changes none of these comparisons. Equal qubit counts therefore do not reverse the ranking of the thresholds. The \XYZ{} code gains on the CSS code under biased noise, and it stays behind the XZZX code below threshold as well as at threshold.

% ---------------------------------------------------------------- figure* (place near the paragraph)
\begin{figure*}[t]
  \centering
  \begin{tikzpicture}
    \begin{groupplot}[group style={group size=3 by 1, horizontal sep=0.55cm, y descriptions at=edge left},
      width=0.35\textwidth, height=5.2cm, ymode=log, xmin=0, xmax=520, ymin=1e-11, ymax=0.1,
      ytick={1e-11,1e-9,1e-7,1e-5,1e-3,1e-1}, minor ytick={1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1},
      xlabel={qubits $n$}, ylabel={error per round $\epsilon$}, ymajorgrids, grid style={grid},
      axis line style={stone}, tick style={stone}, tick label style={font=\footnotesize, text=ink},
      label style={font=\footnotesize, text=ink}, title style={font=\footnotesize, text=ink},
      legend style={font=\footnotesize, draw=none, fill=none, at={(0.97,0.97)}, anchor=north east},
      error bars/error bar style={line width=0.5pt}, every axis plot/.append style={line width=0.8pt}]
    \nextgroupplot[title={SD6, $p=1\times10^{-3}$}]
      \addplot[xyzcode, mark=*, mark size=1.8pt, error bars/.cd, y dir=both, y explicit] coordinates {(35,0.0003417) -= (0,2.02e-05) += (0,2.14e-05) (99,2.949e-05) -= (0,1.62e-06) += (0,1.71e-06) (195,2.715e-06) -= (0,2.56e-07) += (0,2.82e-07) (323,2.487e-07) -= (0,1.39e-08) += (0,1.47e-08) (483,2.09e-08) -= (0,2.74e-09) += (0,3.15e-09)};
      \addlegendentry{\XYZ{}}
      \addplot[csscode, mark=square*, mark size=1.8pt, error bars/.cd, y dir=both, y explicit] coordinates {(17,0.0006185) -= (0,5.36e-05) += (0,5.86e-05) (49,3.921e-05) -= (0,3.24e-06) += (0,3.53e-06) (97,2.658e-06) -= (0,1.48e-07) += (0,1.56e-07) (161,2.495e-07) -= (0,2.36e-08) += (0,2.61e-08) (241,1.219e-08) -= (0,1.53e-09) += (0,1.75e-09) (337,3.587e-10) -= (0,1.79e-10) += (0,3.56e-10)};
      \addlegendentry{CSS}
    \nextgroupplot[title={$\eta=100$, $p=1\times10^{-3}$}]
      \addplot[xyzcode, mark=*, mark size=1.8pt, error bars/.cd, y dir=both, y explicit] coordinates {(35,0.0002408) -= (0,1.68e-05) += (0,1.81e-05) (99,4.074e-06) -= (0,2.27e-07) += (0,2.4e-07) (195,1.299e-07) -= (0,1.23e-08) += (0,1.36e-08) (323,1.667e-09) -= (0,5.2e-10) += (0,7.55e-10) (483,4.09e-10) -= (0,2.52e-10) += (0,6.56e-10)};
      \addlegendentry{\XYZ{}}
      \addplot[csscode, mark=square*, mark size=1.8pt, error bars/.cd, y dir=both, y explicit] coordinates {(17,0.001072) -= (0,7.15e-05) += (0,7.66e-05) (49,0.000143) -= (0,1e-05) += (0,1.08e-05) (97,1.98e-05) -= (0,1.09e-06) += (0,1.15e-06) (161,2.287e-06) -= (0,2.16e-07) += (0,2.39e-07) (241,3.165e-07) -= (0,2.99e-08) += (0,3.31e-08) (337,3.509e-08) -= (0,2.26e-09) += (0,2.41e-09) (449,3.317e-09) -= (0,7.28e-10) += (0,9.33e-10)};
      \addlegendentry{CSS}
      \addplot[xzzxcode, mark=diamond*, mark size=1.8pt, error bars/.cd, y dir=both, y explicit] coordinates {(17,0.0002791) -= (0,1.82e-05) += (0,1.94e-05) (49,1.068e-05) -= (0,9.97e-07) += (0,1.1e-06) (97,3.921e-07) -= (0,2.19e-08) += (0,2.32e-08) (161,1.684e-08) -= (0,1.59e-09) += (0,1.76e-09) (241,5.413e-10) -= (0,2.34e-10) += (0,4.13e-10)};
      \addlegendentry{XZZX}
    \nextgroupplot[title={Pure $Z$, $p=1\times10^{-3}$}]
      \addplot[xyzcode, mark=*, mark size=1.8pt, error bars/.cd, y dir=both, y explicit] coordinates {(35,0.0002178) -= (0,1.6e-05) += (0,1.72e-05) (99,4.088e-06) -= (0,2.28e-07) += (0,2.42e-07) (195,9.843e-08) -= (0,9.31e-09) += (0,1.03e-08) (323,2.405e-09) -= (0,6.76e-10) += (0,9.4e-10)};
      \addlegendentry{\XYZ{}}
      \addplot[csscode, mark=square*, mark size=1.8pt, error bars/.cd, y dir=both, y explicit] coordinates {(17,0.0009895) -= (0,6.86e-05) += (0,7.37e-05) (49,0.0001384) -= (0,9.86e-06) += (0,1.06e-05) (97,1.782e-05) -= (0,9.86e-07) += (0,1.04e-06) (161,2.32e-06) -= (0,2.2e-07) += (0,2.42e-07) (241,3.369e-07) -= (0,3.19e-08) += (0,3.52e-08) (337,4.041e-08) -= (0,2.47e-09) += (0,2.63e-09) (449,8.255e-09) -= (0,1.28e-09) += (0,1.51e-09)};
      \addlegendentry{CSS}
      \addplot[xzzxcode, mark=diamond*, mark size=1.8pt, error bars/.cd, y dir=both, y explicit] coordinates {(17,0.0002331) -= (0,1.65e-05) += (0,1.78e-05) (49,7.832e-06) -= (0,7.41e-07) += (0,8.19e-07) (97,2.369e-07) -= (0,1.32e-08) += (0,1.4e-08) (161,7.948e-09) -= (0,8.51e-10) += (0,9.53e-10) (241,3.181e-10) -= (0,1.58e-10) += (0,3.16e-10)};
      \addlegendentry{XZZX}
    \end{groupplot}
  \end{tikzpicture}
  \caption{Error per round $\epsilon$ of Eq.~\eqref{eq:eps_per_round} against the number of qubits $n$ at $p=10^{-3}$ and $r=3d$, with correlated matching and the weaker logical basis of each code, under SD6 (left), $Z$-biased noise with $\eta=100$ (middle) and pure $Z$ noise (right). Error bars are 68\% Wilson intervals. Open triangles are 95\% upper bounds where a run saw no logical error. Under SD6 the CSS circuit stands for the XZZX and XY circuits, which share its detector error model.}
  \label{fig:equal_qubit}
\end{figure*}
```

### `paper/overleaf_patches/gap.tex`

```latex
% Patch for Results.tex, subsection "Where the gap comes from" (\label{sec:gap}).
% Replaces the four paragraphs of the subsection and the figure fig:gap (between the first and the second paragraph).
% The figure is a full replacement in inline pgfplots with the paper's colours; merge its styling with the live
% figure if that differs. Data: docs/data/campaign/gap/gap.csv and fractions.csv, docs/gap_diagnostic.md.

% ---------------------------------------------------------------- paragraph 1
We locate the gap to the XZZX code under SD6 with six variants of the noise model (Fig.~\ref{fig:gap}). In the noiseless-link model, which we use only as a diagnostic, the link checks are noiseless. In the repeated-link models every link is measured twice or three times per round. In the hybrid model the links are EM3 pair measurements and the plaquettes keep SD6 gates. In the two idle-free models we remove every idle error, in the gate layers and while the ancillas are reset and read out, once with noisy and once with noiseless link checks.

% ---------------------------------------------------------------- figure (replaces fig:gap)
\begin{figure}[t]
  \centering
  \begin{tikzpicture}
    \begin{axis}[width=0.64\columnwidth, height=5.4cm, xbar, bar width=7pt, y dir=reverse,
      xmin=0.40, xmax=1.20, ymin=-0.6, ymax=6.6, xlabel={memory threshold (\%)},
      ytick={0,...,6}, yticklabels={{SD6},{noiseless links},{links twice},{links three times},{hybrid},{no idle errors},{noiseless links, no idle errors}},
      yticklabel style={font=\footnotesize, text=ink}, xticklabel style={font=\footnotesize, text=ink},
      label style={font=\footnotesize, text=ink}, axis line style={stone}, tick style={stone},
      xmajorgrids, grid style={grid}, clip=false]
      \addplot[fill=xyzcode, draw=xyzcode, error bars/.cd, x dir=both, x explicit,
        error bar style={ink}] coordinates {(0.487,0) +- (0.007,0) (0.626,1) +- (0.008,0) (0.558,2) +- (0.021,0) (0.526,3) +- (0.008,0) (0.575,4) +- (0.006,0) (0.854,5) +- (0.006,0) (1.117,6) +- (0.031,0)};
      \draw[xzzxcode, dashed, thick] (axis cs:0.813,6.6) -- (axis cs:0.813,-0.6) node[above, font=\footnotesize, text=xzzxcode] {XZZX};
      \draw[xzzxcode, dotted, thick] (axis cs:1.103,6.6) -- (axis cs:1.103,-0.6) node[above, font=\footnotesize, text=xzzxcode, align=center] {XZZX,\\no idle};
    \end{axis}
  \end{tikzpicture}
  \caption{Memory thresholds of the \XYZ{} code under SD6 and six variants of it: the best of matching, correlated matching and belief-matching in the weaker logical basis, $r=d$ rounds, finite-size fits over $d=11$ to $21$. The noiseless-link variants are diagnostics, not hardware. In the idle-free variants no qubit has an idle error, in the gate layers or while the ancillas are reset and read out. The dashed line marks the XZZX code under SD6 and the dotted line the XZZX code without idle errors.}
  \label{fig:gap}
\end{figure}

% ---------------------------------------------------------------- paragraph 2
The first diagnostic is noise on the link checks. With noiseless link checks the \XYZ{} memory threshold rises from 0.487(7)\% to 0.626(8)\%, against 0.813(20)\% for the XZZX code, which closes 0.43(4) of the gap. Measuring every link twice per round gives 0.558(21)\%, and measuring it three times gives 0.526(8)\%, less than twice. Measuring the links as pair measurements raises the memory threshold to 0.575(6)\%, still below the 0.81\% of the surface codes. This hybrid circuit has fault distance $d$, which we checked at $d=3$, because a faulty link measurement puts a two-qubit error on one block.

% ---------------------------------------------------------------- paragraph 3
The second diagnostic is idling. Each bulk data qubit takes part in one link and three hexagons, so it is busy in four of the six gate layers and idles in the other two. Without idle errors the \XYZ{} memory threshold rises from 0.487(7)\% to 0.854(6)\%, above the 0.813(20)\% of the XZZX code under SD6. The XZZX code also gains from removing its idle errors, to 1.103(9)\%. Measured against this reference, the idle errors account for 0.60(1) of the gap and the noise on the link checks for 0.23(2). With noiseless link checks and without idle errors the \XYZ{} threshold is 1.12(3)\%, which closes 1.02(5) of the gap. Matching alone gives the same picture, with 0.691(6)\% and 0.99(1)\% for the two idle-free \XYZ{} models against 0.93(2)\% for the XZZX code without idle errors. No schedule or compilation that we tried closes the gap under SD6.

% ---------------------------------------------------------------- paragraph 4
The map of Sec.~\ref{sec:cssmap} shows the structural cost. Under depolarizing noise the \XYZ{} code is a rotated surface code concatenated with two-qubit repetition codes. It uses $4d^2-1$ qubits with gates, against $2d^2-1$ for the surface codes at the same distance, and every hexagon has weight six. Criger and Terhal found that the circuit-level threshold of the $[[4,2,2]]$-concatenated toric code depends on the schedule of its weight-eight checks, and their shortest schedule reached about 0.41\%, against about 0.6\% for the toric code~\cite{criger2016fourtwotwo}. In our circuits the combined diagnostic leaves none of the threshold gap, within its error, for the weight-six hexagons and the doubled qubit count. Their cost appears below threshold, where the \XYZ{} code needs more qubits than the surface codes for the same error per round under SD6 (Sec.~\ref{sec:ranking}).
```

### `paper/overleaf_patches/app_fits.tex`

```latex
% Patch for the appendix that describes the campaign (\label{app:fits}): add these two sentences at the end of the
% paragraph that describes the sampling of the campaign (or as a short paragraph of their own after it).
% Source: scripts/campaign_eq.py, docs/equal_qubit_comparison.md.

The sub-threshold runs of Sec.~\ref{sec:ranking} stop each point at 100 logical errors, 300 at the equal-qubit pairs, or at $10^9$ shots, within at most 4 core-hours for matching and correlated matching and 8 for belief-matching, twice that at the pairs, split into replicas of at most two hours. We plan these budgets from short calibration runs and measured decoding times, end a series after the first distance with fewer than 10 errors at its cap, and treat a point without errors in $N$ shots as the 95\% upper bound $3/N$ on its logical error rate per shot.
```

