# XYZ² against the best non-CSS codes

**Question.** Does the XYZ² code beat the best non-CSS codes in circuit-level threshold, and under which noise models? Where it does not, why?

**Status (2026-10-07).** Done. The competitors ran on Perlmutter (`data/campaign/tasks9.jsonl`, 259,720 tasks at 4× budgets, d up to 21, r = d, 1, 2, 3, 5 and 10) next to the full XYZ² matrix. The full tables, every round count and the MWPM-only comparison are in `docs/data/campaign/competitors.md`.

## 1. What the literature reports

Circuit-level thresholds, in % of the physical error rate. "Ours" are XYZ² thresholds from the 1× campaign at r = d, with the best decoder in parentheses. Sources with arXiv ids are in `$SCRATCH/ftmdr_tools/noncss/lit_*.md`, and the downloaded papers are in `papers/` next to them.

| noise class | best non-CSS code in the literature | ours (XYZ²) |
|---|---|---|
| SD6 (uniform depolarizing) | XZZX 0.66 (MWPM, every location at p; arXiv:2505.17718). By Clifford equivalence with the CSS surface code: 0.82 (MWPM) and 0.94 (belief-matching) (arXiv:2203.04948, lighter SPAM). Honeycomb Floquet 0.2–0.35 | 0.43 (MWPM), 0.54–0.56 (BM, Tesseract) |
| SI1000 | honeycomb Floquet 0.1–0.15, the only non-CSS SI1000 number. CSS surface 0.3–0.5 | 0.37 (MWPM) to 0.45 (BM) |
| biased, gates not bias-preserving | XZZX 0.85–0.93 (η = 10–1000, MWPM; arXiv:2505.17718); 0.73–0.80 with H on the data qubits | 0.75 / 0.83 / 0.86 (BM; η = 10 / 100 / ∞) |
| biased, bias-preserving gates | XZZX ≈2.0 (η = 100) to ≈2.4 (η = 1000) (arXiv:2606.17709, read from a figure); Kerr-cat XZZX, CX infidelity up to ≈6.5 | — |
| EM3 (native pair measurements) | honeycomb Floquet 1.5–2.0 (arXiv:2108.10457, 2202.11845). Pair-measurement surface codes 0.24–0.66 | 0.25–0.33 |
| trapped ion | no non-CSS threshold under Quantinuum-like noise. Spin-qubit model (arXiv:2306.17786), gate-error corner: XYZ² 0.465, XZZX 0.37, CSS 0.82 | Helios / H2, no crosstalk: 0.43 / 0.63 |
| erasure | XZZX 4–13 with gate-only noise; 2.85 with SPAM | — |

No single-parameter circuit-level threshold of XYZ² under SD6, SI1000, EM3 or biased noise has been published. The campaign's values are the first. The only earlier circuit-level XYZ² study is Hetényi and Wootton's spin-qubit model; its isotropic value, ≈0.41% with matching, agrees with our 0.43% for MWPM. Nothing in the literature corresponds to our r = 1 values.

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
| h2_p | **0.44** | -- | 0.41 | -- | 0.17 | 1.16 | **1.39** |

With MWPM for every code, at r = d under sd6: XYZ² 0.43%; CSS, XZZX and XY 0.69–0.70%.

## 3. Verdict

**XYZ² does not beat the best non-CSS code. Under circuit-level noise with native gates, the XZZX surface code has the higher threshold in 9 of our 10 noise models, at r = d and at r = 1.**

- Under unbiased noise (sd6, si1000, trapped ions without crosstalk), XYZ² reaches 0.67–0.90× XZZX's threshold.
- Under bias, XZZX turns its bias-preserving gates into 1.5–2.4× XYZ²'s threshold (biased10 to purez).
- The two-level decoders already tried (sequential BP, erasure passing, hierarchical correlated matching, BP+HCM) do not change this. Their best value under sd6, 0.54%, is below what XZZX gets with plain matching (0.70%).

**Where XYZ² does win**

1. **Against the honeycomb Floquet code under every gate-based model.** XYZ² is 2–4× higher (sd6 0.54 vs 0.22, si1000 0.47 vs 0.12, purez 0.86 vs 0.24). The honeycomb is the other non-CSS code built from weight-2 checks.
2. **Against every surface code under em3 (native pair measurements), at every round count.** At r = d it reaches 0.33% against 0.24–0.25%, and at r = 1 1.07% against 0.79–0.82%. Its weight-2 links are native pair measurements and its plaquettes compile to few of them. The honeycomb is still 6× higher under em3.
3. **Trapped-ion noise with measurement crosstalk (h2_p).** XYZ² 0.44% against XZZX 0.41%, a small margin. These crosstalk models drift with d in several series, so this is tentative.
4. **XYZ² matches the CSS surface code** under si1000 (0.47 vs 0.52), Helios (0.43 vs 0.40) and H2 (0.62 vs 0.64) without crosstalk, and under biased noise it beats CSS (0.76–0.86 vs 0.62–0.64). It beats CSS but not XZZX.

**Why XYZ² loses to XZZX** (decoder-independent; see the near-ML references in the campaign):

- **It costs twice as many qubits and checks for the same distance.** [[2d², 1, d]] has 2d² − 1 checks, so with one ancilla per check that is about 4d² qubits, against 2d² for a rotated surface code. More fault locations per logical path lower the threshold, and at equal qubit count XYZ²'s distance is √2 smaller, so it is further behind there than in threshold.
- **The weight-6 plaquettes** need six sequential gates, so ancilla faults spread to several data qubits.
- **Its two-level advantage only holds when the link measurements are perfect.** It shows at code capacity and in the phenomenological model (3.4–4.3% vs about 3%). At circuit level, noisy link measurements make the lower-level flags unreliable.
- **Bias.** With bias-preserving gates XZZX reduces to repetition codes that tolerate measurement errors. XYZ²'s bias structure is coupled through its noisy links.

**When to use XYZ²:**
- on hardware with native two-qubit Pauli measurements, where it beats every surface code;
- in place of the honeycomb Floquet code on gate-based hardware.

**When not to use it:** for gate-based memories with native controlled-Pauli gates, where XZZX (or plain CSS) is better at every round count and noise model we simulated except h2_p.
