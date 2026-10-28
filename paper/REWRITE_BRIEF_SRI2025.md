# Addendum to the rewrite brief: the sequential decoder of Srivastava et al. at circuit level

You are rewriting the paper "Fault-tolerant state preparation and hierarchical decoding for the $XYZ^2$ hexagonal stabilizer code" (Ghosh, Kim, Kairys), working from `paper/REWRITE_BRIEF.md` (October 22, 2026). Since then, another Claude session on NERSC Perlmutter finished one more study. It implemented the sequential decoder of Srivastava, Xiao, Frisk Kockum, Criger and Granath (arXiv:2505.03691, cite key `Sri2025`) at circuit level on our MDR circuits. It then compared that decoder with ours, and the XYZ² code with the other codes. This addendum gives you what it found. You decide whether each result goes in the paper, where, and how to say it.

The same two things are fixed as in the main brief:
- **The numbers.** Use them as given, with their uncertainties, and do not invent new ones. Every number has a source file in the repository.
- **The verdict.** The paper must not claim an XYZ² advantage that the data do not support.

## Where everything is

- **Repository:** https://github.com/Sohan100/XYZ-2-MDR-Logical-State-Preparation (branch `main`, commits `78acf4d` and `f56b0d5`).
- **Full write-up:** `docs/srivastava_exact_decoder.md`. It covers the algorithm, the choices their paper leaves open, the tests, the validation, the circuit-level tables, equal qubits, fault distance and the proposed text.
- **Data:** `docs/data/campaign/sri/`:
  - `validation.csv`, `fit_range.csv` and `their_model.csv`;
  - `circuit_decoders.csv` and `codes.csv`;
  - `eq.csv` and `fault_distance.csv`;
  - `report.md`, every table in one file.
- **Figures:** `docs/figures/sri/`:
  - `thr_bias`: thresholds against bias, their decoder and ours;
  - `ratios`: threshold ratios against the other codes;
  - `pl_<noise>`: the logical error rates.
- **Nothing in the main brief changes.** None of this touches `docs/data/campaign/thresholds.csv`, `competitors.csv` or any number in `REWRITE_BRIEF.md`.

## Names

**Their decoders:** sequential matching (`sri_m`) and sequential belief-matching (`sri_bm`).

**Ours, as the paper names them:**
- MWPM;
- erasure passing (`seq_match`);
- sequential BP (`seq_soft`);
- hierarchical correlated matching, HCM (`corr_gauge`);
- belief-matching, BM (`bm`).

**Their decoder, step by step:**
1. It decodes each [[2,1,1]] link alone along time.
2. It puts $Z$ on the top qubit of a link for every event it calls a data error.
3. It decodes the YZZY surface code with the conditional probabilities of their Table I.

**Two forms of the link step:**
- **Literal rule.** The link step as their paper writes it: pair the link events along time, then remove a pair as two outcome flips when the flips are more likely than two data errors.
- **Fast rule.** One matching graph per link, with a boundary edge at every detector.

Their paper does not say which form they ran. We ran both.

## The bottom line

**The verdict stands.** Their decoder does not change the paper's verdict.

**Our decoders do at least as well everywhere.** We tested three settings:
- their own phenomenological model;
- circuit level, under all eight of our noise models;
- below threshold at equal qubit count.

In all three, XYZ² does no better with their decoder than with ours. Our belief-matching beats their decoder under every noise model except EM3, where the two tie.

**It answers the obvious referee question:** "Did you try the decoder that was built for XYZ²?"

## 1. Validation against their numbers

**Code capacity.**
- **Matching** is reproduced: 18.1(4)% against their 18.34(8)%.
- **Belief-matching** is 4% to 8% low at every fit range: 17.8% against 18.52%, and 22.5% against 24.08% at $\eta=10$. Their paper does not specify its BP step closely enough to locate the difference.

**Phenomenological noise, $r=d$.** The result depends on the distances of the fit. Their fits use $d=3$ to 15, ours $d=11$ to 21. Each cell gives the $\bar X$ / $\bar Y$ memory, in %:

| model | decoder | theirs | literal, $d$ 3 to 15 | literal, $d$ 11 to 21 | fast, $d$ 3 to 15 | fast, $d$ 11 to 21 |
|---|---|---|---|---|---|---|
| depolarizing | matching | 3.42(1) | 3.30 / 3.52 | 3.18 / 3.21 | 3.39 / 3.64 | 3.44 / 3.48 |
| depolarizing | belief-matching | 3.43(1) | 3.15 / 3.41 | 3.10 / 3.12 | 3.23 / 3.54 | 3.41 / 3.43 |
| $\eta=10$ | matching | 3.49(2) | 3.56 / 3.68 | 3.35 / 3.38 | 3.70 / 3.85 | 3.64 / 3.66 |
| $\eta=10$ | belief-matching | 4.30(2) | 3.93 / 4.24 | 3.68 / 3.78 | 4.15 / 4.40 | 4.06 / 4.15 |

Statistical errors are 0.02 to 0.05 points.

- **The literal rule depends on the distances.** Over their distances it reproduces their numbers. Over $d=11$ to 21 its threshold falls to 3.12(13)% with matching, and to 3.64(5)% with belief-matching at $\eta=10$.
- **The fast rule is stable.** It reproduces their numbers at both ranges.
- **Their Table I needs its second-order entries.** To first order, Table I gives no threshold (about 10% at code capacity and 2% under phenomenological noise, with no clean crossing). The second-order entries cover two errors on one link that leave no link event. This is worth a clause in the appendix.

**How to say it:** "Over $d=11$ to 21 the threshold of the literal rule falls below their values." Do not write that their numbers are wrong or that we could not reproduce them. We reproduced them over the distances they used.

## 2. Their comparison redone in their own model

Phenomenological noise, both-bases memory, weaker basis, other codes from `competitors.csv`:
- **Against XZZX:** XYZ² with their decoder reaches 0.66 to 0.76 of the XZZX threshold, under either link rule.
- **Against XY:** it reaches 0.68 to 0.76 of the XY threshold.
- **Margin:** it is below both by at least 7 standard errors.

**Their claim does not hold.** They wrote that XYZ² is "competitive with other codes tailored to biased noise". In their own model, with their own decoder, it is not. As `REWRITE_BRIEF.md` §4 already says, only full belief-matching ties XZZX there, at $\eta=10$.

## 3. Circuit level: their decoder against ours

**Setup:**
- the eight noise models of the main campaign;
- the preparation and the both-bases memories, $r=d$, $d=3$ to 21;
- fits over $d=11$ to 21;
- both link rules, keeping the 16 likeliest pair signatures per link round.

Preparation thresholds in %. The last column is the best of their four variants over the best of our decoders, for the preparation, $\bar X$ and $\bar Y$:

| noise | their matching, literal / fast | their BM, literal / fast | MWPM | erasure passing | sequential BP | HCM | BM | best theirs / best ours |
|---|---|---|---|---|---|---|---|---|
| SD6 | 0.431 / 0.442 | 0.435 / 0.458 | 0.433 | 0.366 | 0.507 | 0.508 | 0.542 | 0.84, 0.91, 0.84 |
| SI1000 | 0.372 / 0.387 | 0.373 / 0.389 | 0.365 | 0.318 | 0.436 | 0.437 | 0.462 | 0.84, 0.88, 0.90 |
| $\eta=10$ | 0.492 / 0.525 | 0.548 / 0.58 | 0.445 | 0.364 | 0.579 | 0.625 | 0.746 | 0.78, 0.77, 0.76 |
| $\eta=100$ | 0.529 / 0.549 | 0.69 / 0.65 | 0.452 | 0.363 | 0.606 | 0.656 | 0.831 | 0.83, 0.77, 0.77 |
| pure $Z$ | 0.533 / 0.554 | 0.60 / 0.667 | 0.449 | 0.365 | 0.608 | 0.661 | 0.845 | 0.79, 0.80, 0.82 |
| EM3 | 0.312 / 0.316 | 0.309 / 0.313 | 0.249 | 0.270 | 0.327 | 0.314 | 0.300 | 0.97, 0.96, 0.98 |
| Helios | 0.315 / 0.323 | 0.355 / 0.351 | 0.294 | 0.248 | 0.360 | 0.376 | 0.429 | 0.83, 0.85, 0.85 |
| H2 | 0.473 / 0.487 | 0.491 / 0.517 | 0.458 | 0.390 | 0.544 | 0.559 | 0.621 | 0.83, 0.90, 0.85 |

The uncertainties are in `circuit_decoders.csv`. Over all 24 series:

- **Their decoder loses to ours under seven models.** The best of their four variants reaches 0.76 to 0.91 of our best decoder, at 3.2 to 10.7 standard errors. Our best is belief-matching, except in one series where it is HCM.
- **Under EM3 it ties.** It reaches 0.96 to 0.98 of our sequential BP, within 2.2 standard errors.
- **Their belief-matching against ours:** 0.71 to 0.88, and 1.01 to 1.03 under EM3.
- **Their matching is a good matching decoder:**
  - it is 13% to 49% above erasure passing, our other sequential matching decoder;
  - it ties MWPM under SD6 and SI1000;
  - it is 3% to 19% above MWPM under bias and the trapped-ion models, and 25% above it under EM3.
- **The link rule barely matters at circuit level.** Against the literal rule, the fast rule changes matching by −1% to +7% and belief-matching by −7% to +11%, mostly within 2 standard errors.
- **One gap in the data.** The literal rule has no clean fit in the $\bar Y$ memory under SI1000, as our belief-matching has none there either.

## 4. Circuit level: the XYZ² code against the other codes

These are the rules of `docs/xyz2_usefulness.md`. In each basis we take the best decoder of the set, then the weaker basis. The other codes are the best of MWPM, correlated MWPM and BM, in their weaker basis.

**Adding their decoder to our set moves no XYZ² threshold.**

**With their decoder alone:**
- **Against XZZX:** XYZ² reaches 0.31 to 0.74 of the XZZX threshold under every model but EM3.
- **Ranks:**
  - fourth of five under SD6, SI1000, $\eta=10$, Helios and H2, above the honeycomb code only;
  - third or fourth under $\eta=100$ and pure $Z$;
  - second under EM3, 26% to 29% above XZZX, XY and CSS. The honeycomb code is 6 times higher there.
- **Against CSS under bias:**
  - with our decoders, XYZ² leads CSS by 14% to 35%;
  - with theirs, it is 13% to 18% below CSS at $\eta=10$ and ties it at $\eta=100$ and pure $Z$.

So XYZ²'s threshold win over CSS under bias depends on a decoder like ours that reads the gauge level.

## 5. Below threshold at equal qubit count

**Setup:**
- 99 qubits ($d=5$) against 97 (XZZX, $d=7$);
- $r=3d$ rounds;
- $\eta=100$ and pure $Z$ noise;
- $p=10^{-3}$, $2\times10^{-3}$ and $3\times10^{-3}$;
- the weaker basis.

**Results.** XYZ² with their decoder has, per round:
- 11 to 40 times the error of the XZZX code with HCM;
- 2 to 8 times the error of XYZ² with our BM;
- the same error as XYZ² with our HCM, within 0.8 to 1.2, when their decoder is belief-matching.

**Their argument does not survive at equal qubits.** They argued that the $2d^2$ pure-$Z$ distance suppresses failures under $Z$ bias.

## 6. Fault distance

**Setup:** SD6, preparation, $r=d$. Every single fault and every pair of faults is decoded as one shot. The MDR circuit has fault distance $d+1$, so at $d=5$ a decoder that keeps it corrects every pair.

**Failing pairs at $d=5$**, out of 5,757,921:
- MWPM: none;
- HCM: 3;
- their matching: 1,708;
- their belief-matching: 1,097.

**The cause is the hard link decisions.** 1,676 of the 1,708 pairs contain a fault that flips the detectors of more than one link, a hook through a hexagon ancilla. BP on the upper level cannot recover what those decisions discard.

## 7. What this means for the paper

### Where it could go

All of these are options, and your call. The labels come from the draft the earlier session was given, so check them against your rewrite.
- **Main text, where the decoders under bias are discussed (`sec:decoders_bias`):** one sentence placing their decoder below belief-matching.
- **Decoder appendix (`app:decoders`):** one paragraph covering their algorithm, the validation, the caveat about fit distances, the circuit-level ranking and the fault distance.
- **Comparison appendix (`app:comparison_thresholds`):** the result in their own model, at circuit level and below threshold.
- **Subsection on earlier XYZ² results (`sec:hw_relation`):** Sri2025 is the third earlier study with a positive claim for XYZ², after arXiv:2112.06036 and Hetényi and Wootton. One sentence there may fit better than in an appendix.
- **Figures:** none is needed. If you want one, `docs/figures/sri/thr_bias.png` is the candidate, redrawn in the paper's pgfplots colours.

### Statements to check for in the paper

1. **A dedicated or sequential decoder could close the gap to XZZX.** The published one does not.
2. **Sri2025 supports XYZ²'s competitiveness under bias.** Their claim does not hold in their own model.
3. **Hard decisions on the link level are harmless.** They lower the fault distance.
4. **Erasure passing or sequential BP is "sequential decoding" in the sense of Sri2025.** Keep them apart. Theirs decides each link along time and passes Table I priors to the YZZY level.

### Phrasings to avoid

- **"Their thresholds are wrong" or "we could not reproduce them."** We reproduced them over their distances.
- **"Their decoder has no threshold."** It has one. Only Table I to first order has none.
- **"Their decoder fails our validation."** Say that the fit over larger distances gives lower values.

## 8. Proposed text

This text is from `docs/srivastava_exact_decoder.md`, written before your rewrite. Rewrite it freely, merge it, or drop it.

**sec:decoders_bias.** We also implemented the sequential decoder of \cite{Sri2025} at circuit level (App.~\ref{app:decoders}), and under seven of our eight noise models its threshold lies between 0.76 and 0.91 of that of belief-matching on the full \XYZ{} detector error model, while under EM3 it ties our best decoder.

**app:decoders.** We also implemented the sequential decoder of \cite{Sri2025} at circuit level. Each link is decoded alone along time: a pair of link events is removed as outcome flips when the flips are more likely than two data errors, and every other event gets $Z$ on the top qubit of its link. The upper level then receives the conditional probabilities of Table I of \cite{Sri2025}, computed from the detector error model, including the second-order entries of uncorrected links, without which the decoder has no threshold. Fitted over their distances, $d=3$ to 15, this decoder reproduces their phenomenological thresholds, 3.30(2)% and 3.52(2)% in the two logical bases against their 3.42(1)% with matching, and their code-capacity threshold, 18.0(1)% and 18.7(2)% against 18.34(8)%. Over $d=11$ to 21 its phenomenological threshold falls to 3.12(13)%. Deciding the links with one matching graph over the time chain of each link removes this drift, giving 3.44(3)% at $d=11$ to 21, and changes the circuit-level thresholds by at most 11%. At circuit level the threshold of their sequential matching is 13% to 49% above that of our erasure passing, and that of their belief-matching is 0.71 to 0.88 of that of belief-matching on the full detector error model, except under EM3, where the two agree. Its hard link decisions cost fault distance: at $d=5$ two faults defeat it under SD6, where matching needs three.

**app:comparison_thresholds.** In the phenomenological model of \cite{Sri2025} the \XYZ{} code with their sequential decoder reaches 0.66 to 0.76 of the threshold of the XZZX code, below it by more than seven standard errors. At circuit level their decoder raises none of our \XYZ{} thresholds. With it the \XYZ{} code reaches 0.31 to 0.74 of the threshold of the XZZX code under the seven noise models other than EM3, and its advantage over the CSS code under bias becomes a tie or a loss. Below threshold their decoder does not rescue the \XYZ{} code either: at 99 qubits and $\eta=100$ or pure $Z$ noise its error per round is 11 to 40 times that of the XZZX code at 97 qubits.

In LaTeX, write the percent signs as `\%`.

## 9. Bibliography

Add `Sri2025` if it is missing:
- **Authors:** Srivastava, Xiao, Frisk Kockum, Criger and Granath.
- **Title:** "Sequential decoding of the XYZ² hexagonal stabilizer code".
- **Reference:** arXiv:2505.03691 (2025).

Take the authors' full names, and any journal version, from the arXiv page.

## 10. Conventions

These are the conventions of `REWRITE_BRIEF.md` §6:
- "we" sentences;
- no em dashes and no semicolons;
- ranges written "X to Y";
- uncertainties in the form 0.91(5)\%;
- the macros `\XYZ{}`, `\Xb` and `\Yb`;
- `\cite{Sri2025}`.

The Overleaf access rules of the main brief apply unchanged.
