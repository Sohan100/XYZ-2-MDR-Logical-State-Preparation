# Gap diagnostic: how much of the gap to XZZX is left without link noise and idling

**Question (referee).** Under SD6 the XYZ² memory threshold is 0.487(7)% (best of matching, correlated matching and belief-matching, weaker basis) against 0.813(20)% for the XZZX code. How much of this gap is left once the link checks are noiseless and the idle errors are gone?

**Status (2026-10-14).** Done. 3,504 tasks plus 16 reruns (a stale runner on old code had failed them). Every new series has a finite-size fit over d = 11 to 21 without drift, and no fitted threshold lies within 25% of an end of its grid, so no second pass was needed.

## 1. Models

Both are registered in `scripts/run_decoder_threshold_sweep.py` and the report lists, like the `hw_` models:

- `sd6_noidle`: `replace(CircuitNoise.uniform(p), p_idle=0.0, p_idle_mr=0.0)`, which is SD6 without any idle error (no error on qubits that wait in a gate layer, and none on the data qubits while the ancillas are reset or measured);
- `sd6_il_noidle`: `sd6_il` (noiseless link checks, a diagnostic and not hardware) with the same idle rates set to zero.

## 2. Runs

`scripts/campaign_gap.py`:

- XYZ² both-bases memory (Logical X̄ and Ȳ) under both models; CSS and XZZX surface codes (bases X and Z) under `sd6_noidle`.
- Decoders: MWPM, correlated matching (`corr_gauge` for XYZ²) and belief-matching. r = d, d = 3 to 21, 14 values of p over a factor 4, 2x campaign budgets.
- MWPM grids are centred on the earlier crossings of d = 7 and 9 under `sd6_noidle`: 0.68% for X̄, 0.73% for Ȳ and 0.91% for the surface codes. Under `sd6_il_noidle` they are centred on a pilot (MWPM, d = 5, 7, 9, started at 0.8%): 0.97(6)% for X̄ and 1.06(4)% for Ȳ (d = 7/9). The other decoders' grids are scaled by the gains of the first run.
- 3,504 tasks with an upper bound of 1,088 core-hours (8.5 node-hours).
- After the first pass, every series whose fitted threshold lay within 25% of either end of its grid was rerun with a grid centred on the fit (`scripts/campaign_recentre.py`).

XYZ² under SD6, `sd6_il`, the repeated-link and hybrid variants, and the surface codes under SD6, come from the fair rerun (`docs/data/campaign/thresholds.csv`, `competitors.csv`). The fits follow `campaign_report` (finite-size fit over d = 11 to 21, window and error rules, drift flag), and `paper/analysis/gap_report.py` computes the values.

## 3. Results

Memory thresholds in %, r = d, d = 11 to 21. Best: the best of MWPM, correlated matching and belief-matching in each basis, then the weaker basis. XYZ² uses the both-bases schedule.

| code | noise model | best decoder | weaker basis, decoder | MWPM alone |
|---|---|---|---|---|
| XYZ² | SD6 | 0.487(7) | X̄, BM | 0.400(6) |
| XYZ² | noiseless links (`sd6_il`) | 0.626(8) | X̄, corr | 0.556(6) |
| XYZ² | links twice | 0.558(21) | X̄, BM | 0.404(5) |
| XYZ² | links three times | 0.526(8) | X̄, corr | 0.360(4) |
| XYZ² | hybrid | 0.575(6) | X̄, corr | 0.470(5) |
| XYZ² | no idle errors (`sd6_noidle`) | 0.854(6) | X̄, corr | 0.691(6) |
| XYZ² | noiseless links, no idle errors (`sd6_il_noidle`) | 1.12(3) | X̄, corr | 0.99(1) |
| XZZX | SD6 | 0.813(20) | Z, BM | 0.697(9) |
| XZZX | no idle errors | 1.103(9) | Z, corr | 0.93(2) |
| CSS | SD6 | 0.81(1) | X, corr | 0.69(1) |
| CSS | no idle errors | 1.10(1) | X, corr | 0.93(2) |

Fraction of the gap that each variant closes, (T_variant − T_SD6)/(T_ref − T_SD6), with first-order propagated errors:

| variant | vs XZZX under SD6, best | MWPM alone | vs XZZX without idle errors, best | MWPM alone |
|---|---|---|---|---|
| noiseless links | 0.43(4) | 0.53(3) | 0.23(2) | 0.29(2) |
| links twice | 0.22(7) | 0.01(3) | 0.12(4) | 0.01(1) |
| links three times | 0.12(3) | −0.13(3) | 0.06(2) | −0.07(1) |
| hybrid | 0.27(3) | 0.24(2) | 0.14(1) | 0.13(1) |
| no idle errors | 1.13(7) | 0.98(4) | 0.60(1) | 0.55(2) |
| noiseless links, no idle errors | 1.93(15) | 1.98(7) | 1.02(5) | 1.10(4) |

Ratios XYZ²/XZZX (best decoders): 0.77(2) with noiseless links, 1.05(3) without idle errors against XZZX under SD6, 0.774(8) without idle errors against XZZX without idle errors, and 1.01(3) with both changes against XZZX without idle errors.

What this says:

- **Idle errors are the larger part of the gap.** Removing them raises XYZ² from 0.487% to 0.854%, more than the XZZX code gains (0.813% to 1.103%). Measured against XZZX without idle errors, which is the like-for-like reference, idling accounts for 0.60(1) of the gap. Against XZZX under SD6 the same change closes 1.13(7) of the gap, because only XYZ² loses its idle errors in that comparison.
- **Link noise is the smaller part:** 0.23(2) of the like-for-like gap, or 0.43(4) against XZZX under SD6.
- **Both together close the whole gap:** 1.02(5) with the best decoders and 1.10(4) with MWPM alone. With noiseless links and no idle errors, XYZ² reaches 1.12(3)% against 1.103(9)% for XZZX without idle errors. Within the errors nothing is left at the threshold level for the weight-six hexagons and the doubled qubit count of the map. Their cost shows up below threshold, as the qubits needed per distance (see `docs/equal_qubit_comparison.md`).
- The noiseless-link models are diagnostics, not hardware: every operation of the link checks is noiseless.
- The surface codes without idle errors keep their reset, measurement and gate errors. CSS and XZZX agree there, as they share the detector error model (1.10(1)% and 1.103(9)%).

The MWPM crossings of the earlier side study (d = 7/9: 0.68% X̄, 0.73% Ȳ and 0.91% for the surface codes under `sd6_noidle`) agree with the full-distance MWPM thresholds: 0.691(6)%, 0.73(1)% and 0.93(2)%.

## 4. Files

- `docs/data/campaign/gap/`: `points.csv` (every point of these runs), `series.csv` (every new fit), `gap.csv` (every value of the comparison), `fractions.csv`, `gap.md`, `fig_gap_pgf.tex`.
- Raw counts in `data/campaign/gap/` on `$SCRATCH`, mirrored to `/global/cfs/cdirs/m4980/ftmdr_campaign/gap/`.
