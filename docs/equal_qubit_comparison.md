# Equal-qubit comparison below threshold

**Question (referee).** Does the XYZ² code do better below threshold at equal qubit count, which is where the original papers place its advantage?

**Status (2026-10-15).** Stages 1 and 2 are done: 2,189 + 104 tasks, no failed task except three SIGBUS crashes on a node that rebooted, all rerun. A stage 3 (one belief-matching series, XYZ² X̄ under η = 100 at p = 3 × 10⁻³, d = 9) is running and changes no comparison below. Fits and tables: `docs/data/campaign/eq/`. Paper text: `paper/overleaf_patches/ranking.tex`.

## 1. Series

`scripts/campaign_eq.py`:

- **Codes.** The XYZ² both-bases memory (X̄ and Ȳ, labels `sd6@both-X` and so on), the CSS surface code (bases X and Z) and the XZZX surface code (bases X and Z). XZZX runs under `biased100` and `purez` only, because under SD6 the CSS, XZZX and XY circuits have the same detector error model, so CSS stands for all three.
- **Noise.** `sd6`, `biased100` and `purez` at p = 10⁻³, 2 × 10⁻³ and 3 × 10⁻³.
- **Rounds and distances.** r = 3d (`campaign.n_rounds("3d", d)`). XYZ² runs at d = 3 to 11 and the surface codes at d = 3 to 15.
- **Equal-qubit pairs.** Qubit counts come from the Stim circuits: 4d² − 1 for XYZ², 2d² − 1 for the surface codes. The pairs are XYZ² d = 5 (99 qubits) with surface d = 7 (97), d = 9 (323) with d = 13 (337), and d = 11 (483) with d = 15 (449). These points also run with r = d.
- **Decoders.** MWPM and correlated matching (`corr_gauge` for XYZ²) at every point. Belief-matching for XYZ² and XZZX wherever 30 logical errors fit in its budget.

## 2. Sampling

- **Stopping.** A point stops at 100 logical errors, or 300 at the equal-qubit pairs, or at 10⁹ shots. Its time budget is at most 4 core-hours for MWPM and correlated matching and 8 for belief-matching, twice that at the pairs. The budget is split into replicas of at most two hours, as `campaign.make_tasks` does.
- **Planning.** Short calibration runs (d ≤ 7 or 9, two minutes per point) give ln ε against d per series, rate and decoder. `scripts/eq_timing.py` measured the CPU seconds per shot of every decoder and distance on a large batch, with decoder construction excluded. The wall times of the calibration on login nodes at load 300 to 1,000 overstated the cost by orders of magnitude.
- **Budgets.** Each point's budget is its cap, or four times its predicted time to target if that is smaller. A series stops after the first distance where fewer than 10 errors are expected at the cap, and its larger distances are reported as upper bounds. Points that used their predicted budget short of the target were topped up to the cap in a second stage (`campaign_eq.py stage2`), and series whose last distance still had at least 10 errors got their next distance.
- **Cost.** 2,189 tasks in the first stage, with an upper bound of 3,265 core-hours (25.5 node-hours) and about 19.8 node-hours expected.

## 3. Analysis

`paper/analysis/eq_report.py`:

- **Error per round.** ε comes from p_L = ½[1 − (1 − 2ε)^r]. p_L carries a 68% Wilson interval, and a point with zero errors is a 95% upper bound of 3/N per shot.
- **Decoder choice.** For each code and noise model, one decoder is chosen from the fits at the largest measured distances and used at every point. The candidates are the decoders measured at every point of the series. MWPM alone is reported as well.
- **Weaker basis.** For each code, noise model and p, the weaker basis is the one with the larger fitted ε at the largest measured distance.
- **Λ.** Λ = ε(d)/ε(d + 2) comes from a weighted fit of ln ε against d. The fit leaves out d = 3 and uses only points with at least 30 errors. Λ is given with its fit error and with its change when the smallest remaining distance is dropped.
- **Equal qubits.** The codes are compared directly at the three pairs, and by interpolation of the fits at the qubit counts of XYZ².
- **Qubits for a target.** The qubit count for ε = 10⁻⁶ and 10⁻⁹ per round comes from the fits, marked as an extrapolation beyond the largest measured distance. A count for 10⁻⁹ is quoted only within two decades of the last measured point.
- **Time boundary.** At the pair points, ε is also taken from the two lengths r = d and r = 3d, from (1 − 2p_L(3d))/(1 − 2p_L(d)) = (1 − 2ε)^(2d), which removes the boundary term.

## 4. Results

Decoder for every code and noise model: **correlated matching** (lowest fitted ε at the largest measured distances among the decoders that cover every point; belief-matching was skipped at the larger distances to save cost and so cannot be used at every point). MWPM alone is in `docs/data/campaign/eq/eq.md`.

### Equal-qubit pairs: ratio of the error per round, XYZ² / other code (68% interval)

| noise | p | 99 / 97 qubits | 323 / 337 qubits | 483 / 449 qubits |
|---|---|---|---|---|
| SD6, vs CSS | 10⁻³ | 11.1 [9.9, 12.4] | 690 [330, 1,460] | > 30 (CSS: no error, < 7 × 10⁻¹⁰) |
| SD6, vs CSS | 2 × 10⁻³ | 8.5 [7.7, 9.5] | 118 [106, 133] | 332 [262, 422] |
| SD6, vs CSS | 3 × 10⁻³ | 6.5 [6.0, 7.2] | 76 [68, 85] | 129 [115, 145] |
| η = 100, vs CSS | 10⁻³ | 0.21 [0.18, 0.23] | 0.048 [0.031, 0.074] | 0.12 [0.04, 0.41] |
| η = 100, vs CSS | 2 × 10⁻³ | 0.25 [0.23, 0.28] | 0.075 [0.067, 0.084] | 0.029 [0.023, 0.035] |
| η = 100, vs CSS | 3 × 10⁻³ | 0.29 [0.26, 0.31] | 0.11 [0.10, 0.12] | 0.052 [0.047, 0.059] |
| η = 100, vs XZZX | 10⁻³ | 10.4 [9.3, 11.7] | XZZX no error (< 9.5 × 10⁻¹⁰) | XZZX no error |
| η = 100, vs XZZX | 2 × 10⁻³ | 10.5 [9.4, 11.8] | 154 [108, 220] | XZZX no error |
| η = 100, vs XZZX | 3 × 10⁻³ | 10.7 [9.6, 11.8] | 170 [148, 195] | 309 [219, 435] |
| pure Z, vs CSS | 10⁻³ | 0.23 [0.21, 0.26] | 0.060 [0.040, 0.088] | XYZ² no error (< 3.3 × 10⁻⁹) |
| pure Z, vs CSS | 2 × 10⁻³ | 0.21 [0.19, 0.23] | 0.076 [0.067, 0.085] | 0.017 [0.014, 0.022] |
| pure Z, vs CSS | 3 × 10⁻³ | 0.25 [0.23, 0.27] | 0.11 [0.10, 0.12] | 0.046 [0.041, 0.051] |
| pure Z, vs XZZX | 10⁻³ | 17.3 [15.4, 19.4] | XZZX no error | both no error |
| pure Z, vs XZZX | 2 × 10⁻³ | 16.1 [14.4, 18.0] | 154 [108, 220] | 82 [27, 255] |
| pure Z, vs XZZX | 3 × 10⁻³ | 17.4 [15.7, 19.3] | 367 [309, 436] | 850 [455, 1,590] |

With MWPM alone, XYZ² is worse than CSS under biased noise as well (ratio 1.7 to 9.3), and far behind XZZX (60 to 7 × 10⁴). Its gain over CSS under bias relies on the gauge-aware correlated decoder.

### Λ = ε(d)/ε(d + 2), correlated matching, weaker basis (d ≥ 5, points with ≥ 30 errors; change when the smallest distance is dropped)

| noise | p | XYZ² | CSS | XZZX |
|---|---|---|---|---|
| SD6 | 10⁻³ | 11.0(4), +0.3 | 13.8(12), −0.2 | — |
| SD6 | 2 × 10⁻³ | 4.0(2), −0.2 | 6.6(1), −0.0 | — |
| SD6 | 3 × 10⁻³ | 2.34(9), −0.1 | 3.98(5), +0.0 | — |
| η = 100 | 10⁻³ | 31(4) (d = 5, 7 only) | 8.0(2), +0.2 | 25(2), −1.9 |
| η = 100 | 2 × 10⁻³ | 12.3(4), −0.9 | 3.70(6), −0.0 | 13.6(6), −0.3 |
| η = 100 | 3 × 10⁻³ | 5.5(2), −0.0 | 2.37(5), −0.1 | 7.8(2), +0.1 |
| pure Z | 10⁻³ | 42(5) (d = 5, 7 only) | 7.5(2), −0.1 | 32(2), −1.7 |
| pure Z | 2 × 10⁻³ | 12.2(5), −0.6 | 3.67(5), −0.0 | 14.7(7), −0.7 |
| pure Z | 3 × 10⁻³ | 5.7(1), +0.1 | 2.41(3), +0.0 | 9.1(3), −0.5 |

Each step from d to d + 2 costs XYZ² 16(d + 1) more qubits against 8(d + 1) for the surface codes.

### Qubits for ε = 10⁻⁶ and 10⁻⁹ per round (correlated matching; "extrap." beyond the largest measured distance; 10⁻⁹ only within two decades of the last measured point)

| noise | p | XYZ² 10⁻⁶ | CSS 10⁻⁶ | XZZX 10⁻⁶ | XYZ² 10⁻⁹ | CSS 10⁻⁹ | XZZX 10⁻⁹ |
|---|---|---|---|---|---|---|---|
| SD6 | 10⁻³ | 244 | 120 | — | 738 extrap. | 339 extrap. | — |
| SD6 | 2 × 10⁻³ | 730 extrap. | 245 | — | not quoted | 679 extrap. | — |
| SD6 | 3 × 10⁻³ | 2,029 extrap. | 455 extrap. | — | not quoted | not quoted | — |
| η = 100 | 10⁻³ | 134 | 192 | 82 | not quoted | 540 extrap. | 229 extrap. |
| η = 100 | 2 × 10⁻³ | 279 | 488 extrap. | 142 | 769 extrap. | not quoted | 378 extrap. |
| η = 100 | 3 × 10⁻³ | 566 extrap. | 1,124 extrap. | 214 | not quoted | not quoted | 582 extrap. |
| pure Z | 10⁻³ | 132 | 194 | 75 | 357 extrap. | 560 extrap. | 206 extrap. |
| pure Z | 2 × 10⁻³ | 272 | 492 extrap. | 127 | 758 extrap. | not quoted | 344 extrap. |
| pure Z | 3 × 10⁻³ | 547 extrap. | 1,096 extrap. | 189 | not quoted | not quoted | 511 extrap. |

### Time boundary

At the pair points, ε from the two lengths r = d and r = 3d differs from ε at r = 3d alone by −46% to +17% (median +1%, 54 points). The largest changes are for XYZ² under biased noise at d = 5 (−40% to −46%), where the boundary adds errors at r = 3d. No comparison above changes sign or crosses its interval.

### Verdict

- **SD6.** XYZ² is worse than the surface codes at every equal-qubit pair: 7 to 11 times at about 100 qubits, rising to 80 to 700 times at 320 to 480 qubits. It needs about twice the qubits for 10⁻⁶ per round at p = 10⁻³, and more at higher p.
- **Biased noise (η = 100, pure Z).** With correlated matching, XYZ² beats the CSS surface code at equal qubits by 3 to 60 times and needs 30% to 50% fewer qubits for 10⁻⁶. The XZZX code beats XYZ² by 10 to 850 times and needs 40% to 60% fewer qubits than XYZ².
- The original proposals' sub-threshold advantage therefore holds against the CSS code under bias only, not against the XZZX code, which is the natural competitor for biased noise.

## 5. Files

- `docs/data/campaign/eq/`: `points.csv` (raw merged counts), `points_eps.csv` (ε and intervals per point), `fits.csv`, `decoders.csv`, `lambda.csv`, `pairs.csv`, `interp.csv`, `targets.csv`, `boundary.csv`, `eq.md`, `fig_eq.png`, `fig_eq_pgf.tex`, `timing.csv`, `calibration_points.csv`.
- Raw counts in `data/campaign/eq/` on `$SCRATCH`, mirrored to `/global/cfs/cdirs/m4980/ftmdr_campaign/eq/`.
