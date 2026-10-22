# XYZ² against the best non-CSS codes

**Question.** Does the XYZ² code beat the best non-CSS codes in circuit-level threshold, and under which noise models? Where it does not, why?

**Status (2026-10-12).** Done, including the fair rerun (§4): every hardware variant the user asked for (hybrid links and plaquettes, repeated links, the phenomenological model of arXiv:2505.03691, ideal links as a diagnostic), both XYZ² logical bases and both extraction schedules, d up to 21, rules in `docs/fair_comparison.md`. The first comparison (§2–3, 2026-10-07) counted XYZ²'s Logical-X memory only; §4 supersedes it. Full tables: `docs/data/campaign/competitors.md`.

## 1. What the literature reports

Circuit-level thresholds, in % of the physical error rate. "Ours" are XYZ² thresholds from the 1× campaign at r = d, with the best decoder in parentheses. Sources with arXiv ids are in `$SCRATCH/ftmdr_tools/noncss/lit_*.md`, and the downloaded papers are in `papers/` next to them.

| noise class | best non-CSS code in the literature | ours (XYZ²) |
|---|---|---|
| SD6 (uniform depolarizing) | XZZX 0.66 (MWPM, every location at p; arXiv:2505.17718). By Clifford equivalence with the CSS surface code: 0.82 (MWPM) and 0.94 (belief-matching) (arXiv:2203.04948, lighter SPAM). Honeycomb Floquet 0.2–0.35 | 0.43 (MWPM), 0.54–0.56 (BM, Tesseract) |
| SI1000 | honeycomb Floquet 0.1–0.15, the only non-CSS SI1000 number. CSS surface 0.3–0.5 | 0.37 (MWPM) to 0.45 (BM) |
| biased, gates not bias-preserving | XZZX 0.85–0.93 (η = 10–1000, MWPM; arXiv:2505.17718); 0.73–0.80 with H on the data qubits | — (not run: our biased models keep every CX, CY and CZ bias-preserving) |
| biased, bias-preserving gates | XZZX ≈2.0 (η = 100) to ≈2.4 (η = 1000) (arXiv:2606.17709, read from a figure); Kerr-cat XZZX, CX infidelity up to ≈6.5 | 0.75 / 0.83 / 0.86 (BM; η = 10 / 100 / ∞) |
| EM3 (native pair measurements) | honeycomb Floquet 1.5–2.0 (arXiv:2108.10457, 2202.11845). Pair-measurement surface codes 0.24–0.66 | 0.25–0.33 |
| trapped ion | no non-CSS threshold under Quantinuum-like noise | Helios / H2, no crosstalk: 0.43 / 0.63 |
| spin qubits (gate-error corner) | XZZX 0.37, CSS 0.82, honeycomb Floquet 0.43; XYZ² itself 0.465 (arXiv:2306.17786, Table I, MWPM, intercepts of a fitted plane) | 0.89 (MWPM), 1.10 (BM) (`hw_g`, `docs/data/campaign/hw/thresholds.md`) |
| erasure | XZZX 4–13 with gate-only noise; 2.85 with SPAM | — |

No single-parameter circuit-level threshold of XYZ² under SD6, SI1000, EM3 or biased noise has been published. The campaign's values are the first. The only earlier circuit-level XYZ² study is Hetényi and Wootton's spin-qubit model. They print isotropic thresholds only for the surface code (0.65%) and XZZX (0.35%). Their XYZ² plane intercepts (Table I, ηT = 20: p_G 0.465, p_T 4.2, p_RR 16.5) combined with their Eq. 10, 1/p_ic = 1/p_G + 1/p_T + 1/p_RR, give ≈0.41% with matching. This value is derived here and is not in the paper. It is not comparable with our sd6 MWPM 0.433(4)% (r = d, `docs/data/campaign/thresholds.csv`). Their circuit uses ancilla pairs, SWAP gates for the plaquettes, ZZ links measured in a second readout round, idling only during readout and reset, and a readout error that includes two-qubit depolarizing noise. The two numbers are close by coincidence. Our runs at their three corners are in `docs/hetenyi_wootton_check.md`. Nothing in the literature corresponds to our r = 1 values.

## 2. The head-to-head, under our noise models

Literature numbers use different noise conventions: SPAM rates, idle noise, bias-preserving gates, ideal first and last rounds, and how rounds are counted. They are therefore only indicative. The comparison that decides the question runs the competitors through the same code as XYZ²:

- **Codes.** Built by `src/mdr/ft/competitor_circuits.py`:
  - the rotated CSS, XZZX and XY surface codes, as X- and Z-basis memories;
  - the periodic honeycomb Floquet code, with H and V observables.
- **Noise.** Every noise channel comes from `FTMDRCircuit`'s own methods, so idle, reset, measurement, resonator-idle, memory-dephasing, crosstalk and biased two-qubit channels are identical. The competitors use the same native controlled-Pauli gates. Under em3 they use the same pair-measurement construction.
- **Decoders and experiments.**
  - Decoders: MWPM, correlated MWPM, belief-matching and BP-OSD-CS (order 10), all on the circuit's detector error model, plus Tesseract.
  - Memories run for r = d, 1, 2, 3, 5 and 10 rounds.
  - Distances: d = 3 to 21 (honeycomb 4 to 20, keeping one patch shape), and up to d = 11 for Tesseract and BP-OSD.
- **Scoring.**
  - A code's threshold is that of its best decoder in its weaker basis, since a memory must protect both.
  - XYZ²'s is that of its best decoder, for its Logical-X memory.
  - Thresholds must come from series reaching d ≥ 11, and must not drift.

### Result: thresholds (%), best decoder of each code

Only finite-size fits over d ≥ 11 count, for every code. Fits that drift, and crossings of the two largest distances alone, are left out. Competitors count in their weaker basis.

| noise | r = d: XYZ² | CSS | XZZX | XY | honeycomb | r = 1: XYZ² | XZZX |
|---|---|---|---|---|---|---|---|
| sd6 | 0.54 | 0.81 | **0.81** | 0.81 | 0.22 | 1.96 | **2.42** |
| si1000 | 0.47 | **0.52** | 0.51 | 0.51 | 0.12 | 1.06 | 1.06 (CSS 1.08) |
| biased10 | 0.76 | 0.64 | **1.17** | 1.02 | 0.23 | 1.97 | **3.18** |
| biased100 | 0.84 | 0.62 | **1.65** | 1.05 | 0.23 | 1.97 | **3.45** |
| purez | 0.86 | 0.62 | **2.06** | 1.05 | 0.24 | 1.99 | **3.48** |
| em3 | 0.33 | 0.24 | 0.25 | 0.24 | **1.98** | 1.07 | 0.81 (honeycomb **3.50**) |
| helios_p_noxt | 0.43 | 0.40 | **0.54** | 0.53 | 0.23 | 1.54 | **2.37** |
| h2_p_noxt | 0.62 | 0.64 | 0.75 | **0.77** | 0.36 | 2.24 | **3.03** |
| helios_p | 0.20 | 0.28 | **0.33** | -- | -- | 0.33 | **0.59** |
| h2_p | 0.31 † | -- | **0.41** | -- | 0.17 | 1.16 | **1.39** |

† Weaker basis (L_X of the `both` schedule, bm): 0.312 ± 0.002%, with ν at its lower bound (0.3). The L_X-only value of the `depth6` schedule, 0.44 ± 0.06% (bp_corr, stat + syst), is within 1σ of XZZX's 0.41 ± 0.02%. Under crosstalk 58 of the 160 h2_p finite-size fits sit at a ν bound, so the h2_p thresholds are rough.

With MWPM for every code, at r = d under sd6: XYZ² 0.43%; CSS, XZZX and XY 0.69–0.70%.

## 3. Verdict

**XYZ² does not beat the best non-CSS code. Under circuit-level noise with native gates, the XZZX surface code has the higher threshold in 9 of our 10 noise models, at r = d and at r = 1.**

- Under unbiased noise (sd6, si1000, trapped ions without crosstalk), XYZ² reaches 0.67–0.90× XZZX's threshold.
- Under bias, XZZX turns its bias-preserving gates into 1.5–2.4× XYZ²'s threshold (biased10 to purez).
- The two-level decoders already tried (sequential BP, erasure passing, hierarchical correlated matching, BP+HCM) do not change this. Their best value under sd6, 0.54%, is below what XZZX gets with plain matching (0.70%).

**Where XYZ² does win**

1. **Against the honeycomb Floquet code under every gate-based model of the campaign table.** At r = d, in its weaker basis, XYZ² is 1.6–3.7× higher (sd6 0.51 vs 0.22, si1000 0.43 vs 0.12, purez 0.86 vs 0.24; smallest margin h2_p_noxt, 0.58 vs 0.36). The honeycomb is the other non-CSS code built from weight-2 checks. Under gate errors alone (`hw_g`, `docs/hetenyi_wootton_check.md`) the two tie: 1.097(37)% against 1.081(8)%.
2. **Against this repository's cat-state compilations of the CSS, XZZX and XY surface codes under em3 (native pair measurements), at every round count** (1.29–1.36× the best of the three). At r = d it reaches 0.33% against 0.24–0.25%, and at r = 1 1.07% against 0.79–0.82%. Its weight-2 links are native pair measurements and its plaquettes compile to few of them. This is not a win against the best known pair-measurement surface code: the published 3aux code (arXiv:2310.12981, not simulated here) reaches 0.66% under the harsher noise model of Chao et al., where the honeycomb reaches 1.3%. Relative to the honeycomb, 3aux reaches 0.33–0.51 and XYZ² 0.165 (a cross-model comparison). The honeycomb is still 6× higher under em3.
3. **Against the CSS surface code**, with XYZ² in its weaker basis (§4; errors stat ⊕ syst): a tie under Helios without crosstalk (0.41 ± 0.04 vs 0.40 ± 0.01), but below CSS under si1000 (0.43 ± 0.01 vs 0.52 ± 0.01, 8.2σ) and H2 without crosstalk (0.58 ± 0.02 vs 0.64 ± 0.01, 2.3σ). Under biased noise it beats CSS (0.75–0.86 vs 0.62–0.64), but not XZZX.

**Why XYZ² loses to XZZX** (decoder-independent; see the near-ML references in the campaign):

- **It costs twice as many qubits and checks for the same distance.** [[2d², 1, d]] has 2d² − 1 checks, so with one ancilla per check that is about 4d² qubits, against 2d² for a rotated surface code. More fault locations per logical path lower the threshold, and at equal qubit count XYZ²'s distance is √2 smaller, so it is further behind there than in threshold.
- **The weight-6 plaquettes** cost threshold through idling, not through faults in their measurement. With noiseless links and no idle errors (`sd6_il_noidle`), XYZ² still measures its weight-6 plaquettes with noisy gates and reaches 1.12(3)% against 1.103(9)% for XZZX without idle errors, a ratio of 1.01(3) (`docs/gap_diagnostic.md`, `docs/data/campaign/gap/gap.csv`, `fractions.csv`). The six gate layers leave more qubits waiting: at equal qubits (99 vs 97) XYZ² has 254 gate-layer idle sites per round against 52 for XZZX, 130 of them on data qubits and 100 on link ancillas that wait through 4 of the 6 layers (`docs/data/campaign/usefulness_pilots/claim_C25.txt`, `c37/cnt_idle2.py`). Removing all idle errors closes 0.60(1) of the gap to XZZX without idle errors.
- **Its two-level decoding pays off under phenomenological noise with bias, not at circuit level.** That model (arXiv:2505.03691) measures every check with error q = p, links included. Under phen_b10 (η = 10) XYZ² ties XZZX at r = d, 5.54(5)% vs 5.54(5)%, and is above CSS (3.11(5)%). Without bias (phen) it is 4σ below XZZX, 4.16(6)% vs 4.51(7)% (stat plus syst, `thresholds.csv`). The paper's own 4.30% at η = 10 is 0.78× XZZX, because our belief-matching is stronger than theirs (§4). At circuit level link noise is the smaller part of the gap to XZZX: noiseless link checks close 0.23(2) of it, removing idle errors 0.60(1) (`docs/gap_diagnostic.md`).
- **Bias.** With bias-preserving gates XZZX reduces to repetition codes that tolerate measurement errors. XYZ²'s bias structure is coupled through its noisy links.

**When to use XYZ²:**
- on hardware with native two-qubit Pauli measurements, where it beats every surface code;
- in place of the honeycomb Floquet code on gate-based hardware.

**When not to use it:** for gate-based memories with native controlled-Pauli gates, where XZZX (or another surface code) is better at every round count and noise model we simulated, h2_p included (0.31% vs 0.41% at r = d). The one nominal exception, h2_p at r = 5 (XYZ² 0.82 ± 0.02% L_X only against XZZX 0.58%), is a Tesseract fit with ν at its upper bound (6.0) and χ² = 14.8 over 7 points.


## 4. Fair rerun (2026-10-12)

The campaign was repeated under `docs/fair_comparison.md`. Every code uses the same noise primitives. A hardware capability given to XYZ² is given to every competitor, and each code then takes its best compilation. XYZ² now counts with its **weaker** logical basis (L_X or L_Y, each with its best decoder), for its best extraction schedule (`depth6` or `both`). Thresholds are finite-size fits over d ≥ 11 without drift.

Independent checks (cloud runs, branches `cloud/*`):
- **Fairness audit (`cloud/fairness-audit`).** The competitor circuits are fair. They are slightly *harsher* than Stim's reference circuit, because data qubits idle twice per round. XYZ² is built the same way. XZZX's 0.81% under sd6 comes from correlated decoding; with MWPM it gives 0.69–0.71%, consistent with the literature's 0.66%. Test suite: 920 passed, 0 failed.
- **Schedule search (`cloud/schedule-search`).** Of 8,928 XYZ² extraction schedules, 1,690 keep full distance in both bases. No compilation closes the gap to XZZX. A pipelined hybrid circuit lowers p_L about 3× at fixed p but barely moves the threshold.
- **arXiv:2505.03691 check (`cloud/phen-2505-03691`).** The paper's matching thresholds are 3.42 ± 0.01% at η = 0.5 and 3.49 ± 0.02% at η = 10. Our sequential matching decoder gives 3.64% at η = 0.5 (3.636 ± 0.049% to 3.644 ± 0.039% over both schedules and both bases) and 3.58–3.68% at η = 10 (errors 0.03–0.08%). These are finite-size fits over d = 11–21 at r = d (`docs/data/campaign/thresholds.csv`, `seq_match`; errors are stat plus syst). Ours are 6% above the paper at η = 0.5 (4–6σ) and 3–6% above it at η = 10 (1.5–4σ). The gap is smaller than the finite-size drift: from the first to the last pair of consecutive distances, the crossing moves by 6–24% at η = 0.5 (crossings 3.06–4.14%) and by 4–12% at η = 10 (crossings 3.36–3.89%). Our belief-matching runs BP on the full detector error model, so it is stronger than the paper's (5.4–5.5% vs 4.30% at η = 10).

### Thresholds at r = d (%), each code's best decoder in its weaker basis

| noise | XYZ² | CSS | XZZX | XY | honeycomb | XYZ² / best |
|---|---|---|---|---|---|---|
| sd6 | 0.51 | 0.81 | 0.81 | 0.81 | 0.22 | 0.63 |
| si1000 | 0.43 | 0.52 | 0.51 | 0.51 | 0.12 | 0.83 |
| biased10 | 0.75 | 0.64 | 1.17 | 1.02 | 0.23 | 0.64 |
| biased100 | 0.84 | 0.62 | 1.65 | 1.05 | 0.23 | 0.51 |
| purez | 0.86 | 0.62 | 2.06 | 1.05 | 0.24 | 0.42 |
| em3 (pair measurements) | 0.33 | 0.24 | 0.25 | 0.24 | **1.98** | 0.16 |
| helios_p_noxt | 0.41 | 0.40 | 0.54 | 0.53 | 0.23 | 0.76 |
| h2_p_noxt | 0.58 | 0.64 | 0.75 | 0.77 | 0.36 | 0.76 |
| h2_p (crosstalk) | 0.31 | — | 0.41 | — | 0.17 | 0.77 |
| hyb (links as pair measurements) | 0.59 | 0.81 ¹ | 0.81 ¹ | 0.81 ¹ | **1.98** ² | 0.30 |
| hyb_lr2 | 0.62 | 0.81 ¹ | 0.81 ¹ | 0.81 ¹ | **1.98** ² | 0.31 |
| sd6_lr2 / sd6_lr3 (repeated links) | 0.57 / 0.53 | 0.81 | 0.81 | 0.81 | 0.22 | 0.70 / 0.65 |
| si1000_lr2 | 0.44 | 0.52 | 0.51 | 0.51 | 0.12 | 0.85 |
| phen (η = 0.5) | 4.16 | 4.57 | 4.51 | 4.55 | 0.74 | 0.91 |
| **phen_b10 (η = 10)** | **5.54** | 3.11 | **5.54 ± 0.05** | 5.37 | 0.77 | **1.00 (tie)** |
| sd6_il (diagnostic: noiseless links) | 0.64 | 0.81 | 0.81 | 0.81 | 0.22 | 0.79 |

¹ all-gate compilation (= sd6). ² native pair measurements (= em3). helios_p is left out of the table: with crosstalk, XYZ²'s weaker basis falls to ≈0.05%.

### Verdict

**Under fair circuit-level noise, XYZ² does not beat XZZX.** The only exception is the phenomenological model with bias η = 10, where the two **tie** (5.54% each, within error). XYZ² never wins outright against the best competitor.

- **Gate-based hardware** (sd6, si1000, ion models): XYZ² reaches 0.63–0.85× the best surface code. Requiring both logical bases lowers XYZ² a little more than the Logical-X-only numbers of §3 suggested; the h2_p "win" of §3 disappears (0.31 vs 0.41).
- **Hybrid hardware** (links as native pair measurements): XYZ² improves from 0.51% to 0.59–0.62%, but the same hardware lets XZZX keep 0.81% with gates, and lets the honeycomb reach 1.98% with pair measurements.
- **Repeated links** raise XYZ² by at most 0.06 percentage points (sd6 0.51 → 0.57) and never close the gap.
- **Ideal links** (diagnostic) recover only part of the gap (0.64% vs 0.81%). Noisy links lower XYZ²'s threshold by about 20%, but idling is the larger part of the gap (`docs/gap_diagnostic.md`). Measured against XZZX without idle errors, idle errors account for 0.60(1) of the gap and link noise for 0.23(2); against XZZX under SD6, link noise accounts for 0.43(4). With noiseless links and no idle errors, XYZ² ties XZZX without idle errors (1.12(3)% vs 1.103(9)%, ratio 1.01(3)). Nothing is left at the threshold level for the weight-6 plaquettes and the doubled qubit count. Their cost shows up below threshold, as the qubits needed per distance (`docs/equal_qubit_comparison.md`).
- **Phenomenological noise** (the setting of arXiv:2505.03691): at η = 0.5 XYZ² is 0.91× the surface codes, and at η = 10 it ties XZZX. These thresholds do not test the claim of arXiv:2112.06036. At code capacity that paper already puts the two thresholds level ("about 28 % for both" at η = 10). Its advantage is a lower failure rate below threshold at equal qubit count (Fig. 7a inset: Z bias, near-ML decoding). The equal-qubit pilots test that claim (docs/data/campaign/usefulness_pilots/: memory at r = 3d, weaker basis of each code, XYZ² BM against XZZX corr, statistical errors only). At η = 10 and p = 1.5–3%, XYZ²'s error per round is 0.70–1.06× XZZX's. Some points are more than 3σ below 1 with Poisson errors, but none stays at 3σ once run-to-run scatter is included, so they count as ties. At η = 0.5 and p = 2.5% the ratio is 1.36(8), 1.83(12) and 2.36(18) at 35/31, 99/97 and 195/199 qubits. That is a loss at the two larger sizes (8.9σ and 11.0σ with Poisson errors).

**Reasons to use XYZ²** (unchanged from §3): it beats every surface code on pair-measurement hardware (em3: 0.33 vs 0.25), but the honeycomb Floquet code is better there. It also beats the honeycomb on all gate-based hardware, and it beats the CSS surface code under bias.

**Footprint.** XYZ² uses about 4d² qubits against 2d² for a rotated surface code (3d² with hybrid links), so at equal qubit count it is further behind than these thresholds show.
