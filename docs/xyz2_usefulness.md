# When is the XYZ^2 code useful? Consensus

The XYZ^2 hexagonal stabilizer code is a YZZY surface code concatenated with two-qubit phase-flip codes on the links of a honeycomb lattice. With one ancilla per check it uses 4d^2-1 qubits. A rotated surface code of the same distance uses 2d^2-1. This document states where XYZ^2 is useful as a memory and for fault-tolerant logical state preparation by measurement, decoding and recovery (MDR). The comparators are the rotated CSS, XZZX and XY surface codes and the honeycomb Floquet code.

## 1. Short answer

**XYZ^2 has no established use on any modelled hardware, either as a memory or for single-round (r = 1) MDR state preparation. The data are negative.**

- **No threshold win.** XYZ^2's threshold is nominally below the best competitor's in 103 of the 107 head-to-head (model, round count) rows. With stat plus syst errors the 107 rows give 0 wins, 22 ties and 85 losses (D/competitors_table.csv; C24, C33). The four nominal exceptions are ties or fit artifacts in non-hardware or crosstalk models (C27).
- **Memory at equal qubits loses on every gate model that was run.**
  - Under SD6, XYZ^2's logical error per round is 6.5-693x the CSS code's (C4). Using the weaker basis at each point it is 7.5-470x (M7).
  - Under strong Z bias it is 10-17x XZZX's at 99/97 qubits and 154-367x at 323/337 qubits (C6).
  - Under biased10 (pilot) it is 3.35(28)x XZZX's (C32).
- **r = 1 preparation, the protocol the paper studies, loses.**
  - Over 13 hardware-motivated models, the threshold in the MDR basis gives 10 losses, 3 ties and no win (C10).
  - On every tested footprint and device, a surface-code patch that fits prepares the state better.
  - On 35 SI1000 qubits a 6x3 CSS patch is 17.8-18.8x better (C11). The upper end of that range is disputed upward to 19.0-19.5x (M10, D4).
- **The one established lead is a memory result in an idealized corner.**
  - The corner is hw_tr: perfect syndromes, eta = 20 dephasing on data qubits only during ancilla readout and reset, and no gate-layer idling.
  - With BM decoding, p_T 4%, 99/97 qubits and r = 6d, XYZ^2's error per round is 0.384(39) of square XZZX's (Poisson, stat only).
  - It is measured against XZZX only.
  - The margin is narrow. delta* is 0.43 if each run has its own offset and 0.30 if the two XYZ^2 runs share one (C33, C35; see also M8).
  - hw_tr is therefore competitive, not useful.
- **phen_b10 (not hardware) is a threshold tie.** At r = d XYZ^2 has 5.539(54)% against XZZX's 5.537(55)% (stat plus syst). No below-threshold lead is established there (C17, C31).
- **Both idealized edges reverse for r = 1 preparation.** In L_X, XYZ^2 is 18.5(7)x worse in hw_tr and 8.4(6)x worse in phen_b10 (stat only). Both losses hold (C9, C31, C35).
- **No device-like spin variant establishes a lead.**
  - The maximum-fidelity proxy m1 gives 0.70(4) and fails the replicate-scatter test.
  - With 2% of p_T on every gate-layer idle site the ratio is 1.02(7).
  - At the rotated surface code's QEC-optimal readout (run E), XYZ^2 is 1.72(16)x worse, a loss for run offsets up to 0.22 (C37, C38).
- **A use is not excluded only where nothing was simulated.** The untested conditions are (section 6):
  - gate-layer dephasing between 0 and 2% of p_T on a spin device;
  - each code at its own optimal readout time;
  - eta = 10 at p_T 4-6%;
  - a harsher preparation boundary at eta = 20;
  - Hetenyi-Wootton's two-window 4d^2:3d^2 layout;
  - a real-time BP-class decoder;
  - strong Z bias below p = 1e-3.

## 2. Conventions

- **Paths.**
  - Paths are relative to the repository root /pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation.
  - D/ is docs/data/campaign/. P/ is docs/data/campaign/usefulness/pilots/.
  - Papers are the .txt files named by arXiv number in /pscratch/sd/s/sohan100/ftmdr_tools/noncss/papers/.
- **Labels in this document.**
  - C1-C38 are the claims in the agreement record (section 9).
  - D1-D4 are open disputes (section 7). M1-M10 are material omissions (section 8).
  - A D or M label next to a number means that number is the subject of that item.
- **Status words.** Measured means read from campaign or pilot output. Inference means derived from measured data or the literature under a stated assumption. Untested means not simulated.
- **Ratios.** Threshold ratios are XYZ^2/competitor, so below 1 means XYZ^2 is worse. Equal-qubit error ratios are XYZ^2/competitor, so above 1 means XYZ^2 is worse.
- **Labels.** A loss means XYZ^2 is at least 3 sigma below some competitor. A tie means XYZ^2 is within 3 sigma of every competitor. A win means XYZ^2 is at least 3 sigma above the best competitor.
- **Threshold significance.**
  - Threshold z is (XYZ^2 minus competitor)/sigma, so it is negative when XYZ^2 is lower.
  - It uses stat plus syst errors (the err column of D/thresholds.csv) unless marked stat only.
  - Pair crossings, taken from two distances, carry stat errors only and do not count as thresholds (docs/fair_comparison.md).
- **Threshold rule sets.**
  - The table rule is the selection in D/competitors_table.csv: each code's weaker basis, its best usable finite-size fit with dmax >= 11, and XYZ^2's better schedule.
  - The symmetric rules restrict XYZ^2 to its both-bases schedule with mwpm, corr_gauge or bm. Competitors use mwpm, corr or bm (P/fairrev/sym.py).
- **Equal-qubit campaign.**
  - It covers memory at r = 3d and p = 1e-3, 2e-3 and 3e-3 under SD6, biased100 and purez (D/eq/, docs/equal_qubit_comparison.md).
  - Point brackets are 68% Wilson intervals, stat only.
  - Ratio brackets combine the opposite endpoints of the two intervals (D/eq/pairs.csv).
- **Pilot significance.** Pilots are not campaign standard, and all pilot errors are stat only. A pilot ratio R carries its Poisson error. Its significance is z = |ln R|/sigma, in the same form for leads and losses, under three error models:
  - (a) Poisson.
  - (b) Poisson errors scaled by f. For a pool, f = max(1, sqrt(chi2/dof)) from its own replicates, read as Student t when f > 1 is estimated. For a single run on either side, f = 1.7.
  - (c) An extra run-level log-rate variance delta^2 for each XYZ^2 BM run. Competitor runs stay Poisson.

  This document quotes z under (a) and (b), and delta*. delta* is the largest delta at which (c) still gives z = 3.
- **Pilot rule.**
  - A pilot lead or loss holds when z >= 3 under (a) and (b) and delta* >= 0.37.
  - A point that passes (a) and (b) with a smaller delta* is delta-dependent and is not counted.
  - A point that fails (a) or (b) is a tie.
  - Pilots without XYZ^2 BM runs are outside (c) and are labelled on (a) and (b) alone (M6).
- **The delta range.** delta lies in about 0.065-0.37 (C33).
  - 0.065 is the estimate over all 42 replicate sets of P/statrev/homog.py (output P/r6om/homog.out). It is a lower estimate.
  - Over the XYZ^2 BM sets alone, the claims quote 0.074 over 35 sets. Over all 37 such sets it is 0.068 (M3).
  - 0.37 is a point estimate from one flagged set with one degree of freedom (P/cons/x3_X_a.txt and x3_X_b.txt), selected after inspection (M8).
  - The rule discounts leads and losses alike, so which pilot results count depends on delta.

## 3. Regimes and verdicts

### 3a. Hardware-motivated and idealized regimes

| Regime | Verdict | Key numbers (errors used) | Source | Claim |
|---|---|---|---|---|
| SD6 and SI1000 gate noise, links repeated; memory threshold at r = d | Not useful | SD6 symmetric 0.487(7)%, 0.60 of XZZX; table entry 0.5116(78)% vs XZZX-Z 0.8128(200)%, z -14.0. SI1000 0.4308(85)% vs CSS-Z 0.5200(69)%, z -8.2. Repeated links 0.65-0.85. Stat plus syst | D/thresholds.csv, D/competitors_table.csv, P/fairrev/sym.py | C1 |
| 11 gate models without crosstalk, r = 1, 2, 3, 5, 10, d | Not useful | XYZ^2 nominally lower in 66 of 66 rows; highest ratio 0.98 (si1000, si1000_lr2 at r = 1). Table rule: 12 ties, 54 losses (stat plus syst). Strongest loss biased100 r = d, z -40.2 | D/competitors_table.csv, D/thresholds.csv, paper/analysis/competitor_report.py | C2 |
| SI1000 below r = d (si1000, si1000_lr2) | Not useful (ties, no win) | si1000 r = 2: 0.808(33)% vs CSS-Z 0.899(28)%, z -2.1, a tie under both rule sets. si1000_lr2 r = 1-10: ties (0.89-0.98). r = d: loss, z -8.2. Stat plus syst | D/thresholds.csv | C3 |
| Z-biased bias-preserving gates (biased10, biased100, purez), all r | Not useful | r = d vs XZZX: 0.64, 0.51, 0.42 (z -23.9, -40.2, -18.2). XZZX leads by at least 6.9 sigma at every r. Stat plus syst | D/thresholds.csv, D/competitors_table.csv | C5 |
| Square untailored CSS only, under bias | Not useful (relative win over CSS only) | r = d: XYZ^2 1.16-1.38x CSS with BP-class decoding (6.7-17 sigma, stat plus syst); MWPM on both 0.73-0.76x. Equal qubits: 3.5-58x better with corr, 1.7-9.3x worse with MWPM. XZZX dominates XYZ^2 on the same gates | D/thresholds.csv, D/eq/pairs.csv, D/eq/targets.csv | C7 |
| Trapped ions without crosstalk (helios_p_noxt, h2_p_noxt) | Not useful | r = d 0.76 of the best surface code (symmetric 0.757 and 0.748). Helios: XYZ^2 BM 0.407(3)% vs XZZX-Z 0.537(6)%. Device points sit at 0.25 and 0.32 of XYZ^2's r = d thresholds | D/thresholds.csv, docs/data/helios/helios_budget.csv | C12 |
| Trapped ions with crosstalk (helios_p, h2_p) | Not useful | No code has a threshold; listed values are fit artifacts. Matched d = 9/11 pair crossings: XYZ^2 0.47-0.65 of the best surface code (stat only) | D/thresholds.csv, D/crossings.csv | C13 |
| Native pair measurements (em3) | Not useful | XYZ^2 1.29-1.36x the cat-state compilations (r = d 0.3254(45)% vs XZZX 0.2483(23)%, z +15.2). The honeycomb, native there, has 1.976(21)%, 6.1x XYZ^2 | D/thresholds.csv, D/competitors_table.csv | C14 |
| Gates plus pair measurements (hyb, hyb_lr2) | Not useful | Against the honeycomb, the rules' comparator, without nu-pinned fits: 0.30-0.53 (hyb) and 0.31-0.61 (hyb_lr2). Every entry is a loss with abs(z) >= 3.8 (stat plus syst) | D/competitors_table.csv, D/thresholds.csv | C15 |
| Gate hardware against the honeycomb only | Not useful (relative win over the honeycomb only) | XYZ^2 1.60-4.46x the honeycomb's threshold; hw_g tie 1.02x (0.4 sigma), hw_g1q 1.08x (3.5 sigma). The honeycomb is never the best gate code | D/competitors_table.csv, D/hw/summary.csv | C16 |
| Spin corners with one channel: hw_g, hw_g1q, hw_t (r = d) | Not useful | hw_g 1.097(37)% vs CSS 1.400(49)% (z -5.0) and XZZX 1.389(14)% (z -7.4). hw_g1q: XYZ^2 third. hw_t: 0.35 of CSS, 0.13 of XZZX. Stat plus syst | D/hw/summary.csv, D/hw/thresholds.csv | C19 |
| Spin readout corner hw_rr | Not useful | Threshold tie: 10.63(46)% vs CSS 10.50(56)% and XZZX 10.32(44)% (z +0.2, +0.5). Equal qubits, per-run p_L (pilot, stat only): 2.10(2)x worse with corr_gauge (z 75, 44) and 2.09(1)x with MWPM (z 104, 61), losses that hold; BM 1.98(7)x, delta* 0.225, delta-dependent | D/hw/summary.csv, P/rv/ | C20 |
| Combined spin noise at equal rates (inference) | Not useful | Hetenyi-Wootton combination rule on the repo's corners: XYZ^2 0.909(26)% vs XZZX 1.150(13)%, 0.79, 8.5 sigma below. Without XYZ^2's hw_tr term: 0.995(30)%, 4.7 sigma below. hw_g and hw_rr errors are stat plus syst, hw_tr errors are fit errors. Rests on the hw_g loss and the planar rule | D/hw/summary.csv, D/hw/diag_hw_tr_centers.csv, 2306.17786.txt | C21 |
| phen (eta 0.5; not hardware) | Not useful | r = d: 4.156(57)% vs XZZX 4.510(66)%, 0.92, z -4.1; r = 1-10 ties. Equal qubits (pilot, stat only): 98/100 data qubits 2.36(18)x, z 11.0, 8.7, delta* 0.39, a loss that holds (M2). 50/49 1.83(12)x and 18/16 1.36(8)x are delta-dependent | D/thresholds.csv, P/bq/phen.log | C18 |
| phen_b10 threshold (eta 10; not hardware) | Competitive (r = d only) | r = d tie 5.539(54)% vs XZZX 5.537(55)%, z +0.03; symmetric 0.984, z -1.3. Needs BP-class decoding (MWPM 0.57, corr_gauge 0.78 of XZZX). XZZX higher at r = 1-3 (>= 3.4 sigma) and at r = 5 under symmetric rules (z -4.6) | D/thresholds.csv, P/fairrev/sym.py | C17 |
| phen_b10 at equal qubits, memory at r = 3d (pilots; not hardware) | Competitive (no lead established) | Stat only. 50/49 data qubits: 2% L_Y 0.81(4), z 4.1, 4.1, delta* 0.13; 2.5% L_X 0.78(5), delta* 0.069; 3% L_X 0.81(4), delta* 0.078. 18/16 vs XZZX d = 4: 0.70(5), delta* 0.097. All delta-dependent | P/bq/, P/ul/, P/stat/, P/ovc/ | C31 |
| SD6 at equal qubits, 99-483 qubits | Not useful | XYZ^2/CSS per round 6.5-693x with corr on both, fixed bases (7.5-470x with each code's weaker basis per point, M7). 99/97 at 1e-3: 11.09 [9.90, 12.43]. The 693x pair rests on 2 CSS errors; its one-sided 3-sigma Poisson bound is 106.6x | D/eq/pairs.csv, D/eq/points_eps.csv | C4 |
| Strong Z bias at equal qubits (biased100, purez), p = 1e-3 to 3e-3 | Not useful | 68% Wilson brackets, stat only. 99/97, corr: biased100 10.39 [9.26, 11.65] to 10.65x, purez 16.09-17.39x. 323/337: 154-367x at 2e-3 and 3e-3. XYZ^2 BM vs XZZX corr: 3.4-6.9x at 99/97; 28x [20, 39] and 51x [35, 73] at 323/337. Bulk only: 6.2-14.9x at 99/97 | D/eq/pairs.csv, D/eq/points_eps.csv, D/eq/boundary.csv, D/eq/targets.csv | C6 |
| biased10 at equal qubits (pilot), 99/97, p = 3e-3 | Not useful | XYZ^2 BM L_Y vs XZZX corr X: 3.35(28)x, z 14.7, 14.7 (stat only), delta* 1.3, a loss that holds. corr_gauge 6.4-6.7x | P/ul/, P/stat/, P/bq/b10.log, P/ovc/resb10.jsonl, P/hwr/ | C32 |
| Equal distance under strong bias (about 2x the qubits) | Not useful | 1e-3, corr: XYZ^2 lower by 2.6-6.1x (biased100) and 1.9-3.3x (purez) at d = 5-9. At 2e-3, purez 1.00-1.55, not lower. Qubits x d favours XZZX by 1.48-1.63x | D/eq/points_eps.csv, D/eq/targets.csv | C8 |
| hw_tr memory (idealized spin corner) | Competitive | r = 6d, p_T 4%, 99/97: 0.384(39), z 9.5, 9.4 (stat only), delta* 0.43. The one pilot lead that holds, against XZZX only. r = 3d points 0.61-0.68: ties or delta-dependent. eta 100: 1.14(19). corr_gauge 2.27(3)x worse | P/cons/, P/litrev/, P/cg_Y.txt | C35 |
| Device-like hw_tr variants (gate, readout, gate-layer dephasing) | Not useful | Stat only. m1 (f = 0) 0.70(4), z 5.7, 2.0, fails (b). Run A 0.75(6), tie. f = 0.02: 1.02(7). f = 0.05: 1.27(12). Run E 1.72(16), z 5.7, 5.7, delta* 0.217, delta-dependent loss | P/hwrev_q7/, P/cons/, P/litrev/ | C37 |
| Hetenyi-Wootton example spin device, mapped to the tested mixtures | Not useful | At each code's best tested point (E): XZZX 6.60e-4 per round vs XYZ^2 1.13e-3, 1.72(16)x. Their threshold planes (inference): p/p_th 1.11 (XYZ^2) vs 0.62 (XZZX) at the maximum-fidelity readout, 0.79 vs 0.62 at the QEC-optimal readout | P/hwrev_q7/, P/c38.py, 2306.17786.txt | C38 |
| r = 1 MDR-basis threshold (depth6 L_X), 13 hardware-motivated models | Not useful | 10 losses, 3 ties (SI1000; si1000_lr2; sd6_lr3 through one TN ML fit), 0 wins. SD6 1.960(73)% vs XZZX-Z 2.417(47)%, z -5.3. Under bias XYZ^2 stays at 1.97-1.99% while XZZX reaches 3.2-3.5% and CSS-Z 4.9-5.9% | D/thresholds.csv, D/competitors_table.csv | C10 |
| r = 1 at fixed footprint or device (SI1000, SD6, ions) | Not useful | Pilots, stat only. SI1000, 35 qubits: XYZ^2 L_X 1.139(33)e-3 vs CSS 6x3 6.16(16)e-5 to 6.39(16)e-5, 17.8-18.8x (M10, D4). SD6, 35 qubits: 57x vs CSS 6x3. SI1000 99/97: 2.5-2.7x vs CSS d = 7. Helios 2.80(12)x, H2 4.70(16)x vs CSS-X | P/bq/si.log, P/ovc/resr1.jsonl, P/c11r7/, P/c36r6/, P/r7rev6/, P/rev5/, P/bq/ion.log | C11 |
| r = 1 in the idealized edges (hw_tr, phen_b10) | Not useful | Stat only. hw_tr p_T 4%, 99/97: 18.5(7)x, z 76, 45. phen_b10 p = 3%, 50/49: 8.4(6)x, z 32, 19, delta* 0.71. Both hold | P/r1_xyz2.out, P/r1_xzzx.out, P/r1_phen_xyz2.out, P/r1_phen_xzzx.out | C9, C31, C35 |
| Platform map | Not useful | No modelled platform gives an established use. Ions lose at both device points (2.8-37.3x Helios, 4.7-26.5x H2). SI1000 has no remaining 35-48 qubit window. Spin variants as above. em3 favours the honeycomb by 6.1x. Neutral atoms were not modelled | P/bq/ion.log, P/c11r7/, P/c36r6/, P/hwrev_q7/, D/thresholds.csv | C36 |

### 3b. Cross-cutting findings

| Topic | Verdict | Key numbers (errors used) | Source | Claim |
|---|---|---|---|---|
| Mechanism: the MDR time boundary | Not useful (mechanism) | XYZ^2's r = 1 threshold in the MDR basis does not grow with bias: 1.960(73)% (sd6) to 1.989(75)% (purez). In phen_b10 its r = 1 fits reach at most 15.6%, against 25.2-28.6% for XZZX. In hw_tr the L_X boundary term is 76(20)% of the r = 3d L_X error, 1.3 sigma above one half (stat only) | D/thresholds.csv, P/cons/, P/litrev/, src/mdr/ft/extraction_schedule.py | C9 |
| Gap diagnostic (SD6 variants; not hardware) | Not useful (diagnostic) | With noiseless links and no idling XYZ^2 ties XZZX, 1.013(30). Idling is 0.60(1) of the gap. Link noise is 0.23(2) against XZZX without idling or 0.43(4) against SD6 XZZX. The two are super-additive at about 3.5 sigma. Repeated links recover 0.12-0.22 | D/gap/gap.csv, D/gap/fractions.csv, docs/gap_diagnostic.md | C22 |
| Decoder dependence | Not useful | No decoder gives a win on any hardware-motivated model. At r = d, best/MWPM is 1.16-1.90 for XYZ^2 and 1.01-1.18 for XZZX. Competitor BP-OSD and Tesseract exist only as pair crossings; admitting them lowers every XYZ^2 ratio they change | D/competitors_table.csv, D/thresholds.csv, D/eq/timing.csv | C23 |
| Fairness and selection effects | Not useful | The rules favour XYZ^2 in the median. Its entry is the maximum over 16-27 fits (median 21), against at most 12 for competitors, and exceeds the symmetric value by a median 2.5% (up to 12.7%). Symmetric rules: 82 losses and 15 ties of 97 rows | P/fairrev/sym.py, P/fairrev/pool.py, D/thresholds.csv | C24 |
| Hardware cost at equal distance | Not useful | About 2x the qubits, 2.08-2.25x the two-qubit gates per round, 8 vs 6 ticks, coupler degree 6 vs 4, three gate types vs two. At 99/97, 254 vs 52 gate-layer idle sites per round (4.9x) | P/qrev/cnt3.py, P/c37/cnt_idle2.py, 2306.17786.txt | C25 |
| Labelling of the biased models; non-bias-preserving gates | Untested | The campaign's biased models are bias-preserving, although three documents list them as not. Bias with CX compiled through H (2505.17718) is untested. XYZ^2's 0.75/0.83/0.86% lie between XZZX's two non-bias-preserving compilations (0.73-0.80% and 0.85-0.93%) | src/mdr/ft/circuit_noise.py, docs/noncss_comparison.md line 15, lit_hardware.md | C26 |
| Table entries that are artifacts or non-wins | Not useful | The four 'winner xyz2' rows: phen_b10 r = d (1.0005, z 0.0), phen r = 1 (1.08, z 1.0), phen_b10 r = 10 (1.096, z 1.2), h2_p r = 5 (1.41; Tesseract fit with nu pinned at 6.0, chi2 14.8). Also biased100 r = 10 vs XY (decoder mismatch) and hyb_lr2 r = 1 (nu-pinned fit) | D/competitors_table.csv, D/thresholds.csv | C27 |
| Noise lacking the link Pauli | Untested | (a) Z dephasing in the XX-link frame was tested in hw_tr; the nominal lead needs BM and f = 0. (b) Amplitude damping with ZZ links is untested | P/cons/, P/hwrev_q7/ | C28 |
| Long memories under extreme Z bias | Untested | Boundary-free ratio at 99/97 favours XZZX by 6.2-9.0x (biased100) and 10.6-14.9x (purez). XYZ^2 BM Lambda (d = 7 to 9, 3e-3) is 7.5 and 9.4 against a break-even of 18.4-22.8. One sentence is disputed (D1) | D/eq/boundary.csv, D/eq/points_eps.csv, D/eq/lambda.csv | C29 |
| Regimes and competitors not simulated | Untested | See section 6 | D/eq/pairs.csv, src/mdr/ft/circuit_noise.py | C30 |
| Statistical standard | Not useful (scope statement) | Threshold labels are 0 wins, 22 ties, 85 losses with either competitor error file. Replicates exceed Poisson scatter in 4 of about 17 sets; the homogeneity test over 42 sets gives chi2 80.8 on 69 (p 0.16). One qubit-matched pilot lead holds. Several items are disputed (D3) | D/competitors.csv, D/thresholds.csv, P/statrev/homog.py, P/r6om/ | C33 |
| Literature cross-check | Not useful | Hetenyi-Wootton's corners, combined here, give XYZ^2 0.41% above XZZX's 0.35%; this is not reproduced. 2112.06036's equal-qubit advantage is code capacity at eta = 10 with near-ML decoding against d-frame XZZX (about 1.7-2.7x), and near parity against the d^2 frame. 2505.03691 places the advantage close to threshold | 2306.17786.txt, 2112.06036.txt, 2505.03691.txt | C34 |

## 4. Where XYZ^2 is useful or competitive

### 4.1 Useful

Nowhere established. No threshold comparison establishes an XYZ^2 lead over the best competitor run. Neither does any qubit-matched pilot on a gate-level or device-like model (C33).

### 4.2 Competitive: two idealized memory models

Both models are diagnostics, not device models. Both edges need BM or BP-class decoding, and both reverse for r = 1 preparation.

**hw_tr (C35, C9, C33).**
- The model has noiseless gates, readout and reset (perfect syndromes). Data qubits get eta = 20 Z-biased dephasing only while ancillas are reset and read out. There is no gate-layer idling. Links are XX, and the memory protects both bases.
- XYZ^2 BM is compared with square XZZX corr in its distance-d frame, at equal total qubits.
- Where both decoders were run, XZZX BM matches XZZX corr (0.99(6) at p_T 6%).

| Point (99/97 qubits unless stated) | R (Poisson) | z (a), (b), stat only | delta* | Label | Files |
|---|---|---|---|---|---|
| p_T 4%, r = 6d, worse basis vs worse basis | 0.384(39) | 9.5, 9.4 | 0.43 (0.30 if the two XYZ^2 runs share an offset) | Lead holds | P/cons/b_x_Y_a.txt, b_x_Y_b.txt, b_z_X.txt |
| p_T 4%, r = 6d, XYZ^2 L_X vs XZZX X | 0.362(38) | 9.8, 9.7 | 0.46 (0.32 shared) | Lead holds | P/cons/b_x_X_a.txt, b_x_X_b.txt, b_z_X.txt |
| p_T 4%, r = 6d, better basis vs better basis | 0.564(59) | 5.5, 5.45 | 0.227 | Delta-dependent | P/cons/b_x_X_*.txt, b_z_Z.txt |
| p_T 6%, r = 3d | 0.613(42) | 7.2, 4.2 | 0.148 | Delta-dependent | P/litrev/hwtr_xyz2_Y.txt, hwtr_xzzx_X.txt |
| p_T 4%, r = 3d | 0.608(77) | 3.9, 2.3 | | Tie | P/litrev/hwtr4_xyz2_X.txt, hwtr4_xzzx_X.txt |
| p_T 3%, r = 3d | 0.679(80) | 3.3, 3.3 | 0.075 | Delta-dependent | P/cons/x3_Y_a.txt, x3_Y_b.txt, z3_X.txt |
| p_T 6%, 195 qubits, vs XZZX interpolated | 0.538(100) | 3.3, 3.3 | 0.125 | Delta-dependent | P/litrev/hwtr_xyz2_Y_d7.txt, _d7b.txt, hwtr_xzzx_X_d9.txt, _d11.txt |
| p_T 3%, eta = 10 | 0.807(82) | 2.1 | | Tie | P/cons/e10_x_Y.txt, e10b_x_Y.txt, e10_z_X.txt |
| p_T 3%, eta = 100 | 1.14(19) | 0.8 | | Tie | P/cons/e100_x_Y.txt, e100_z_X.txt |
| p_T 4%, XYZ^2 corr_gauge, worse basis | 2.27(3) | | outside (c) | Loss | P/cg_Y.txt, P/litrev/hwtr4_xzzx_X.txt |
| p_T 4%, r = 1 preparation, L_X | 18.5(7) | 76, 45 | outside (c) | Loss holds | P/r1_xyz2.out, P/r1_xzzx.out |

Conditions and limits of the hw_tr edge:
- **Memory at r = 6d only.** The established point is a memory result at r = 6d. Every r = 3d point is a tie or delta-dependent.
- **Narrow margin.** delta* 0.43 is 15% above the top of the delta range, and that top rests on one degree of freedom (M8). Model (c) gives each of the two XYZ^2 runs per basis its own offset. With one shared offset, delta* is 0.30 (L_Y) and 0.32 (L_X), and the lead becomes delta-dependent (C33, C35).
- **r = 6d against r = 3d.** The worse-basis ratio is nominally lower at r = 6d by 1.59x. That is 2.8 sigma under (a) and 1.9 under (b), so the change is not established. L_Y is flat in r, so a plausible cause is that L_X's time-boundary term is spread over more rounds (inference; C9; D1).
- **BM is required.** XYZ^2 corr_gauge is 2.27(3)x worse than XZZX corr in its worse basis (P/cg_Y.txt). XYZ^2 MWPM is about 15x worse (runs whose outputs were not kept). Eleven of the repository's fourteen decoders were not run under hw_tr.
- **One usable competitor.**
  - Only XZZX in its d frame is decoded well enough to count.
  - XY with BM is 5.2-5.5x worse than XYZ^2, and Y-biased XZZX with BM is 4.9(7)x worse. Both are also worse than d-frame XZZX, which points to decoder failure rather than the frame (inference).
  - CSS-Z, the honeycomb, rectangular patches and ML-decoded d^2-frame patches were not run at equal qubits (C34, C35).
- **No gate-layer idling.** The model leaves out exactly the noise where XYZ^2 has 254 idle sites per round against 52 for XZZX (C25).
- **Bias window not established.** At p_T 3%, eta = 10, 20 and 100 give 0.807(82), 0.679(80) and 1.14(19). The eta = 10 and eta = 100 values differ from eta = 20 by 1.1 and 2.5 sigma (Poisson).
- **Preparation boundary.** A harsher boundary was tested only at eta = 10, with phen, noiseless readout and a full noise layer before the first round. There XYZ^2's L_Y becomes:
  - 1.44(9) of XZZX with BM (z 6.0, 4.7; delta* 0.21), a delta-dependent loss;
  - 1.91(11) with corr_gauge (z 11.3, 6.6), a loss that holds (C9).
- **Size trend unresolved.** Lambda is 6.5(13) against a break-even of 5.5-5.8.
- **Decoder speed.** The repository's Python BM takes 1.1-1.7 ms per round at d = 5 on SD6 graphs (D/eq/timing.csv). That is roughly two orders of magnitude longer than a 9-13 us spin cycle. Real-time BP-class decoding is untested.

**phen_b10 (C17, C31).** Phenomenological noise with Z bias eta = 10 and readout flips at q = p on all checks (the model of 2505.03691). Not hardware.
- Threshold:
  - At r = d it is a tie, 5.539(54)% against XZZX's 5.537(55)% (stat plus syst), and it needs BP-class decoding.
  - XZZX is higher at r = 1-3 by at least 3.4 sigma, and at r = 5 under the symmetric rules (z -4.6).
  - r = 10 is unresolved.
- Equal qubits, memory at r = 3d, XYZ^2 BM in its weaker basis against square XZZX corr (pilots, stat only):

| Point | R (Poisson) | z (a), (b) | delta* | Label | Files |
|---|---|---|---|---|---|
| 50/49 data qubits, 2%, L_Y (8 runs) | 0.81(4) | 4.1, 4.1 | 0.13 | Delta-dependent | P/ul/b10*_xyz2_Y.txt, P/stat/, P/bq/p02b.log, P/ovc/res20.jsonl |
| 50/49, 2%, L_X (10 runs, f 1.53) | 0.78(4) | 5.3, 2.7 | | Tie | same sources, X basis |
| 50/49, 2.5%, L_X | 0.78(5) | 3.7, 3.7 | 0.069 | Delta-dependent | P/stat/run_0.025_*, P/ovc/res25.jsonl |
| 50/49, 3%, L_X | 0.81(4) | 4.8, 4.7 | 0.078 | Delta-dependent | P/bq/p03.log, P/ul/c_xyz2_X_d5_p0.03_bm_5.txt |
| 18/16, 3%, vs XZZX d = 4 | 0.70(5) | 5.3, 3.1 | 0.097 | Delta-dependent | P/bq/p03.log |
| 18/16, 3%, vs XZZX interpolated to 18 data qubits | 0.78(5) | 4.0, 2.4 | | Tie | P/bq/p03.log |
| 98/100, 2-3% | 0.88-0.94 | below 3 | | Ties | P/bq/, P/ul/, P/ovc/ |
| 50/49, 1.5%, L_X | 1.06(12) | | | Tie | P/stat/run_0.015_*, run15_* |
| Noiseless readout (q = 0), 18/16, 3% | 0.62(4) | 7.5, 4.4 | 0.144 | Delta-dependent lead (M1) | P/bq/cc4.log |
| q = 0, 50/49, 3%, L_Y | 1.44(9) | 6.0, 4.7 | 0.213 | Delta-dependent loss | P/bq/cc4.log, P/cons/cc_phen0_Y.txt |
| q = 0, 50/49, 3%, L_X | 1.22(8) | 3.1, 1.6 | | Tie | P/bq/cc4.log, P/cons/cc_phen0_X.txt |
| r = 1 preparation, 50/49, 3%, L_X | 8.4(6) | 32, 19 | 0.71 | Loss holds | P/r1_phen_xyz2.out, P/r1_phen_xzzx.out |

- **No lead is established.** At delta 0.065, the four leads would hold: 50/49 at 2%, 2.5% and 3%, and 18/16. At delta 0.37 none does.
- **Matching-class decoders lose.** At 2%, 50/49, L_X, XYZ^2 corr_gauge is 1.62(3)x XZZX corr, and XYZ^2 MWPM is 7.62(8)x XZZX MWPM.
- **Repeat r = 1 runs.** The repeat runs whose outputs were not kept give 8.4-10.7x against XZZX L_X and 12.1-13.5x against XZZX L_Z (C10). The range quoted in C31 is disputed (D2).
- **d^2 frame.** Against the d^2 frame (run as the XY code), XYZ^2 leads. But XY with BM and with corr is itself worse than d-frame XZZX, which points to decoder limits (inference). ML-class decoding of the d^2 frame is untested.

### 4.3 Ties, not leads

- Hardware threshold rows that tie under both the table rule and the symmetric rules: si1000 at r = 2, and si1000_lr2 at r = 1, 2, 3, 5 and 10 (C3, C24).
- Hardware threshold rows that tie under the table rule only: si1000 at r = 1 and 5, h2_p_noxt at r = 3 (z -2.98), h2_p at r = 1, and the h2_p r = 5 fit artifact (C2, C24, C27).
- r = 1 thresholds in the MDR basis:
  - SI1000: the CFE-TN fit is within 0.38 sigma of every CSS, XZZX and XY basis. With matched decoders, corr_gauge 0.995(14)% is 0.91-0.94 of each competitor fit, with a largest pull of 2.6 sigma (stat plus syst). With stat errors only it is 4.2 sigma below XZZX-X corr (C3, C10).
  - si1000_lr2: bp_corr 1.074(52)% is within 0.35 sigma of every CSS, XZZX and XY basis (C10).
  - sd6_lr3: a tie only through its TN ML fit, 1.975(151)%. With matched decoders BM loses at 11.5 sigma (C10).
- The hw_rr threshold (C20).
- phen_b10 at r = d, with BP-class decoding (C17).
- SD6 at 35 qubits against square D = 4 patches: 1.04-1.07x, at most 2.1 sigma under (a). A 6x3 CSS rectangle on the same 35 qubits is 57x better (C11).

### 4.4 Relative wins over codes that are themselves beaten

- **Square untailored CSS under bias (C7).**
  - At r = d XYZ^2's threshold is 1.16-1.38x CSS's with BP-class decoding, and 1.19/1.33/1.44x with belief-matching on both codes.
  - With correlated matching on both codes, C7 reports a win only under biased100 (1.066, 4.2 sigma). With the both schedule that comparison is 1.065, z 2.6, a tie (D4).
  - With MWPM on both codes XYZ^2 is lower at every r.
  - XZZX dominates XYZ^2 on the same gates, and rectangular or tailored CSS was not run.
- **The honeycomb on gate hardware (C16).** XYZ^2's threshold is 1.60-4.46x the honeycomb's. The repository's SD6 honeycomb (0.218(3)%) is at the bottom of the published 0.2-0.357% range, so the margin is inflated.
- **em3 cat-state compilations (C14).** XYZ^2 is 1.29-1.36x these. The honeycomb, native on that hardware, is 6.1x XYZ^2.

## 5. Where XYZ^2 is not useful

**Memory thresholds on every hardware-motivated model.**
- Gate models at r = d (C1, C5, C12, C24):
  - SD6 0.60 (symmetric rules) and SI1000 0.83;
  - biased10/biased100/purez 0.62/0.49/0.41;
  - Helios and H2 without crosstalk 0.76 and 0.75;
  - with MWPM for every code, SD6 0.57-0.58 and SI1000 0.77 (C1).
- Over 11 gate models without crosstalk at r = 1, 2, 3, 5, 10 and d, XYZ^2 is nominally lower in all 66 rows (C2). At r = 4, 6-9 and 11-21 only XYZ^2 has fits, so the verdict there is an inference.
- Ions with crosstalk give no threshold for any code (C13). em3 and hyb favour the honeycomb (C14, C15).
- Spin corners: hw_g and hw_t are losses, and hw_rr is a tie (C19, C20).
- hw_tr thresholds (C21):
  - MWPM d = 7/9 pair crossings put XZZX-Z at 18.95(160)% and XYZ^2-X at 10.49(55)%.
  - The gap is 5.0 sigma stat only, and 1.7 sigma when each code's shift between distance pairs counts as syst.
  - Pair crossings do not count as thresholds, and the best-decoder ranking is untested.
- Combined spin noise at equal rates is a loss by inference (C21).
- Hetenyi-Wootton's own threshold planes put XYZ^2 above both CSS and XZZX only near the readout corner, by at most 1.14x. That is about 0.9 sigma if the plane deviations are read as errors (inference; C21).

**Memory at equal qubits on every gate model run.**
- **SD6 (C4).**
  - At equal qubits XYZ^2 has 1/sqrt2 of the distance, so to win it needs Lambda_XYZ2 > Lambda_CSS^sqrt2.
  - It reaches 0.27-0.33 of that in its X basis and 0.38-0.40 in its Y basis (inference).
  - At 35 qubits (d = 3) it is 3.3-5.0x worse than CSS interpolated to 35 qubits.
- **Strong Z bias (C6).**
  - XYZ^2 needs 1.6-2.9x XZZX's qubits to reach 1e-6 per round; the upper values are extrapolated (D/eq/targets.csv).
  - One same-shot test showed a 2.0-2.5x gain from CFE and Tesseract. A gain of that size would leave XYZ^2 about 1.4-3.5x worse at 99/97 (inference).
- **biased10 (C32).** XYZ^2 is 3.35(28)x worse at 99/97 and p = 3e-3. The original paper's equal-qubit advantage at eta = 10 was found at code capacity. It does not carry over to gate-based extraction at this size and rate.
- **hw_rr (C20).**
  - XYZ^2 is 2.10(2)x (corr_gauge) and 2.09(1)x (MWPM) worse per run.
  - In Hetenyi-Wootton's spin layout, XYZ^2 d = 5 (100 qubits) has 1.156(6)x the error of XZZX d = 5 (75 qubits) with MWPM.

**r = 1 preparation.**
- **Threshold in the MDR basis.** 10 losses, 3 ties, 0 wins (C10). The ion crosstalk models and the spin corners hw_g, hw_g1q, hw_t and hw_rr have no usable r = 1 fits.
- **Fixed footprint, 35 SI1000 qubits (C11).**
  - A 6x3 CSS patch prepares |+> with 6.16(16)e-5 to 6.39(16)e-5 in three archived runs (P/c11r7/rect.txt, P/c36r6/rect_css_6x3_X.txt, P/r7rev6/rect_6x3_X.txt). The shortest logical error Stim's search finds for it has 6 faults.
  - XYZ^2 d = 3 has 1.139(33)e-3 (corr_gauge), 1.1698(21)e-3 (MWPM) and 1.170(58)e-3 (BM).
  - With BM, z is 52 under (a) and 31 under (b), and delta* is 0.97 (stat only).
  - In the other basis, square CSS D = 4 on 31 qubits beats XYZ^2's best L_Y by 1.53-1.62x: at least 12.5 sigma under (a) and 7.3 under (b) (C33, C36). The upper end of that range is disputed (D4).
- **Square patches alone.** Against square patches only, XYZ^2 d = 3 would lead below 49 qubits (1.43-1.66x with MWPM or corr_gauge). The rectangles on the same footprint remove this window (C11, C36).
- **Larger SI1000 footprints.**
  - At 99/97, XYZ^2 corr_gauge is 2.5-2.7x worse than square CSS d = 7, at least 22.6 sigma under (a).
  - XYZ^2 BM is 2.1-2.3x worse there, but with delta* 0.19-0.22, so the square patch alone does not establish that loss.
  - A CSS 8x6 patch on 95 qubits is 7.1x below XYZ^2 BM (delta* 0.63) and 8.3x below corr_gauge.
  - At 195/161, XYZ^2 is 1.74x worse than CSS D = 9. Footprints of 323 qubits and more were not run.
- **Other models at fixed footprint.**
  - SD6 at 99/97: XYZ^2 is 7.7-8.1x worse.
  - Helios: 2.8-2.9x worse than CSS-X d = 7. Scaling XYZ^2's memory dephasing by 6/8, as a gate-count timing proxy, still leaves it 1.91(7)x worse (P/c11r7/memscale.txt).
  - H2: 4.7-4.8x worse than CSS-X d = 5.

**Device-like spin variants (C37, C38).**
- All runs are at 99/97 qubits (the repository's 2:1 convention), r = 3d and eta = 20. XYZ^2 d = 5 BM is compared with XZZX d = 7 corr.
- f is the fraction of p_T applied on every gate-layer idle site.

| Variant | Noise | R (Poisson) | z (a), (b) | delta* | Label | Files |
|---|---|---|---|---|---|---|
| m1, maximum-fidelity proxy | p_T 4%, gates 0.1%, readout 1%, f = 0 | 0.70(4) | 5.7, 2.0 (t 3.44 on 3 dof) | | Tie (fails (b)) | P/cons/m1_*, P/litrev/m1_xyz2_X.txt, P/hwrev_q7/D0_xyz2_X.txt |
| m1 at 195 qubits | as m1 | 0.81(17) | 1.0 | | Tie | P/cons/m1d7_*, m1d9_z_X.txt, m1d11_z_X.txt |
| A | gates 0.3%, otherwise m1 | 0.75(6) | 3.75, 2.2 | | Tie | P/hwrev_q7/A_* |
| f = 0.02 | m1 plus 2% of p_T on gate-layer idle sites | 1.02(7) | 0.3 | | Tie, no edge | P/hwrev_q7/C_*, D2_xyz2_X.txt |
| f = 0.05 | m1 plus 5% | 1.27(12) | 2.65, 1.6 | | Tie, nominally worse | P/hwrev_q7/B_* |
| E, QEC-optimal readout of the rotated surface code (5.8 us) | p_T 1.5%, readout 2.7%, gates 0.1%, f = 0 | 1.72(16) | 5.7, 5.7 | 0.217 | Delta-dependent loss | P/hwrev_q7/E_*, E2_xyz2_Y.txt |

- **m1 lead lost with gate-layer dephasing.** From m1 to f = 0.02 and f = 0.05, ln R shifts by 3.9 and 5.4 sigma (Poisson), but by 2.3 and 2.97 under (c) at delta 0.15. The loss of the m1 lead is therefore a trend, not an established change.
- **Best tested points.** Both codes do best at E, where XZZX's 6.60e-4 per round is 1.72x below XYZ^2's 1.13e-3. XZZX at E is also 2.03(13)x below XYZ^2 at m1 (z 11.3 under (a), 2.7 under (b)), which is not established.
- **The device's f is not pinned down (inference).** Neither the repository nor Hetenyi-Wootton give gate durations.
  - If each idle phase dephases on its own, as in their Eq. D1, a 100 ns idle gives f below 1e-3.
  - If gate-layer dephasing adds coherently to the readout window, a 100 ns layer gives f of 0.015 at the maximum-fidelity readout and 0.022 at the QEC-optimal readout. f reaches 0.02 at layers of about 130 ns and 90 ns.
  - Echoed idles bring f toward 0.
  - Neither tested end of this range gives XYZ^2 an established lead.
- **Hetenyi-Wootton's layout.** It uses 4d^2 qubits for XYZ^2 and 3d^2 for XZZX, with two readout windows per cycle for XYZ^2. A crude two-window estimate at bare hw_tr gives XYZ^2 d = 5 (100 qubits) 0.96(7)x XZZX d = 5 (75 qubits) per cycle. That is a tie in which XZZX uses 0.75x the qubits (inference; C37). M9 raises how to normalize it per unit time and per qubit.
- **Not modelled in these mixtures:** reset error, single-qubit gate error, readout-pair two-qubit error, quasi-static dephasing and the second readout window.

**Ions (C12, C36).**
- The ion models apply pure-Z memory dephasing in every gate layer. That is hw_t's placement, where XYZ^2 loses.
- Ion memory error is about 5-6e-3 per data qubit per round. The hw_tr edge was measured at 3-6% per round, and it is absent at eta = 100.
- The ion models therefore offer no route to the hw_tr edge (inference).

**The rules favour XYZ^2, and it still loses (C23, C24).**
- In 86-87 of 107 rows, XYZ^2's entry uses a decoder that no competitor has at fss.
- Under the symmetric rules, six table-rule ties become losses: si1000 r = 1 and 5, h2_p_noxt r = 3, phen_b10 r = 5, and sd6_il r = 5 and 10. One loss becomes a tie: si1000 r = 3.
- Per-row labels near abs(z) = 3 are not robust; the ranking is.

### 5.1 Where the cost comes from

- **Threshold gap (C22, C34).**
  - With noiseless links and no idling, XYZ^2 ties XZZX at 1.013(30) (D/gap/fractions.csv). At threshold level this does not support the concern in 2112.06036 that noise on the weight-6 checks drives the gap.
  - Idling accounts for 0.60(1) of the gap and link noise for 0.23(2) to 0.43(4). The two are super-additive.
- **Idling (C22, C25).**
  - XYZ^2 has 8 ticks per round against 6.
  - At 99/97 qubits it has 254 gate-layer idle sites per round against 52: 130 on data qubits, 100 on link ancillas that wait through 4 of 6 gate layers, and 24 on plaquette ancillas.
  - A d = 7/9 MWPM schedule-search diagnostic removed only gate-layer idling. That raises XYZ^2's crossing from 0.38(2)% to 0.57(4)%, about 0.70(16) of the gain from removing all idling (cloud/schedule_search/results.md on branch origin/cloud/schedule-search).
  - That the idling cost comes mainly from the deeper gate schedule is an inference.
- **Hardware (C25).** At equal distance XYZ^2 needs about twice the qubits, 2.08-2.25x the two-qubit gates per round, coupler degree 6 against 4, and three gate types against two. SI1000 does not charge for degree-6 couplings, which favours XYZ^2 (C11).
- **Below threshold (C4).** At equal qubits XYZ^2 has 1/sqrt2 of the distance (inference).
- **Time boundary (C9).** MDR preparation sits on XYZ^2's weak side. Its r = 1 threshold in the MDR basis does not grow with bias. In phen_b10 its r = 1 threshold reaches at most 15.6%, against 25.2-28.6% for XZZX.

## 6. Untested, and what would test it

| Untested condition | Evidence so far | What would test it | Claims |
|---|---|---|---|
| Gate-layer dephasing 0 < f < 0.02 on a spin device; the device's gate-layer duration and echo scheme | m1 (f = 0) 0.70(4), fails (b); f = 0.02 1.02(7); device f from below 1e-3 to above 0.02 (inference) | m1-type runs at f = 0.005 and 0.01 with replicates and recorded seeds; measured gate-layer durations and echo scheme | C37, C38 |
| Each code at its own QEC-optimal readout time; the exact maximum-fidelity point (p_T 3.3%, p_RR 1.6%) | E (surface-code optimum, 5.8 us) 1.72(16)x worse; m1 has 1.2x the device's p_T | Runs at about 7.3 us (XYZ^2) and 7.9 us (XZZX) from Eq. 14 with the repository's corners (5.7 and 7.5 us with their Table I), with reset error included | C30, C37, C38 |
| eta = 10 at p_T 4-6%; eta between 10 and 20; eta for realistic data T1 | eta = 10 run only at p_T 3%: 0.807(82), a tie. eta = 100: 1.14(19). T1 = 10 ms gives eta of about 70-100 (inference) | hw_tr at eta 10 and 15, p_T 4-6%, 99/97, with more shots | C34, C35, C37 |
| A harsher preparation boundary at eta = 20 | At eta = 10 (phen q = 0): BM L_Y 1.44(9), delta-dependent loss; corr_gauge 1.91(11), loss that holds | hw_tr or phen with a full noise layer before the first round at eta = 20 | C9, C35 |
| Hetenyi-Wootton's 4d^2:3d^2 layout with a true two-window cycle and ZZ links | Crude bare-hw_tr estimate 0.96(7)x per cycle at 100/75 qubits (M9) | Build the two-window circuit; run XZZX d = 6 (108 qubits) | C25, C30, C37 |
| Just-in-time or Z-basis link ancillas | 100 of XYZ^2's 254 gate-layer idle sites are waiting link ancillas | Schedule variant with late link-ancilla preparation | C25, C30 |
| Real-time BP-class decoding | Repository Python BM 1.1-1.7 ms per round at d = 5 on SD6 graphs | Time an optimized BM or BP implementation | C23, C35, C37, C38 |
| The other eleven decoders under hw_tr | Only BM, corr_gauge and MWPM were run | bp_full, bp_corr, CFE, Tesseract and the TN decoders at the r = 6d and r = 3d points | C35 |
| Competitors in hw_tr at equal qubits | Only d-frame XZZX counts; the d^2-frame proxies show decoder failure (inference) | CSS-Z, honeycomb, XY, rectangular, co-prime and ML or tensor-network d^2-frame patches | C30, C34, C35 |
| hw_tr size trend and run offsets | Lambda 6.5(13) against 5.5-5.8; a shared offset gives delta* 0.30 | d = 7 and 9 at r = 6d; replicate batches with recorded seeds | C33, C35 |
| r = 1 preparation in hw_tr at campaign standard | One archived pair, 18.5(7)x | Other p_T, eta, d = 3 and the device-like mixtures | C9, C30, C35 |
| Strong Z bias below p = 1e-3 | Lambda slope trend 3.9 sigma stat only, 2.5 with the Lambda_change systematic (M4), measured with corr only (M5); the 99/97 gap is flat at 10-17x | XZZX d = 13-15 and XYZ^2 d >= 9 with BM in both bases at p <= 1e-3; CFE or Tesseract at r = 3d | C6, C9, C30, C33 |
| biased10 at campaign standard | One pooled pilot point, 3.35(28)x | BM for both codes, p = 1e-3 to 3e-3, d = 5-11 | C32 |
| r = 1 at other fixed footprints | SI1000 tested to 322 qubits; SD6 at 35 and 99 qubits; ions at device points | XYZ^2 d >= 9 under SI1000; BM at d = 7; other models at 35-48 qubits; an SD6 layout serving both bases at 35 qubits; a 10x5 patch on 99 qubits; routing cost on degree-4 or heavy-hex chips | C11 |
| Noise types not simulated | | Non-bias-preserving gates (2505.17718); X- and Y-biased circuit noise; erasure and leakage; amplitude damping with ZZ links; quasi-static dephasing; intra-link correlated ZZ dephasing | C26, C28, C30 |
| Missing campaign runs | | Equal-qubit runs under hyb, the ions, em3 and SI1000 memory; competitor BP-OSD and Tesseract finite-size fits; competitor runs at r = 4, 6-9 and 11-21; r = 1 thresholds for the spin corners; rectangular or tailored CSS under gate-level bias; repetition cat codes; 3aux under em3 | C2, C7, C10, C14, C23, C30 |
| Platforms not modelled | | Neutral atoms with erasure and atom loss; shuttling spin architectures; a per-gate timing model for ions; a threshold study of the heavy-hex compilation of 2109.13308 | C30, C36 |
| Logical operations | | Lattice surgery and injection, which touch the weak time boundary | C30 |

## 7. Open disputes

These are the items below 80% agreement in the last round: C29, C31, C33 and OVERALL.

### D1. C29, long memories under extreme Z bias (agreement 0.7)

- **Current text.** Long memories are untested and weakened by measured data. BM loses 3.4-6.9x at 99/97 and 28-51x at 323/337, and the boundary-free ratio favours XZZX by 6.2-9.0x and 10.6-14.9x. In hw_tr "the worse-basis ratio improves from 0.61 to 0.38 only because L_X's time-boundary term is spread over more rounds (C9)".
- **Objections (reviewers 3, 4, 5; major).**
  - The sentence states as fact a change that is not established, and a cause that C9 marks as inference.
  - The change is ln(0.608/0.3835)/sqrt(1/62 + 1/11782 + 1/99 + 1/14848) = 2.8 sigma under (a) and 1.9 under (b), stat only (P/litrev/hwtr4_xyz2_X.txt, P/litrev/hwtr4_xzzx_X.txt, P/cons/b_x_Y_a.txt, b_x_Y_b.txt, b_z_X.txt).
  - L_X's own drop is 2.9 sigma. Its boundary share, 76(20)%, is only 1.3 sigma above one half.
- **Proposed correction.** "The worse-basis ratio is nominally lower at r = 6d (0.61(8) to 0.38(4), a factor of 1.59; z 2.8 under (a) and 1.9 under (b), stat only; not established). L_Y is flat in r, so a plausible cause is that L_X's time-boundary term is spread over more rounds (inference; C9)."
- **Effect on the verdict.** None. C29 stays untested, and the correction matches C9, which has full agreement. Reviewer 5 checked the other C29 numbers and found them correct: BM Lambda 7.47 and 9.38 against 18.4-22.8, and boundary ratios 6.2-9.0 and 10.6-14.9.

### D2. C31, phen_b10 at equal qubits (agreement 0.6)

- **Current text.** phen_b10 equal-qubit memory is competitive. The nominal leads at 18/16 against XZZX d = 4 and at 50/49 for p = 2% (L_Y), 2.5% and 3% are delta-dependent (delta* 0.069-0.133), so no lead is established. At r = 1 XYZ^2 is 8.4(6)x worse in the archived run, and repeat runs give "9.8-11.2x against XZZX L_X in C10's runs ... up to 14.1x against XZZX L_Z (C10)".
- **Objections (reviewers 1, 3, 4, 5; major).**
  - The repeat-run range uses three values: XYZ^2 L_X 1.696e-3, XZZX X 1.512e-4 and XZZX Z 1.264e-4.
  - C10 does not use these values, and they appear in no run output file under P/. They are recorded only in P/claim_C10.txt line 94.
  - C10's ranges are 8.4-10.7x against XZZX X and 12.1-13.5x against XZZX Z.
  - C31's evidence says that C10's evidence lists XYZ^2 L_X 1.605-1.696e-3, XZZX X 1.512-1.643e-4 and Z 1.200-1.264e-4. That does not match C10.
- **Proposed corrections.**
  - Reviewers 4 and 5: "the recorded unarchived runs give 8.4-10.7x against XZZX L_X and 12.1-13.5x against XZZX L_Z (C10), and 9.9(5)x in L_X and 15.1(7)x in L_Y in the run recorded here". Reviewer 5 also gives 9.2-10.7x against XZZX L_X for the repeat runs alone. Both delete the evidence sentence about C10's evidence.
  - Reviewer 3: quote C10's set (same-run L_X pairs 9.8(5) and 9.9(5); all pairings 9.2-10.7 against X; 12.1-13.5 against Z; L_Y 14.3-15.1 against X). Alternatively, have C10 admit the runs recorded with counts in P/claim_C10.txt line 94.
  - Reviewer 1: either quote C10's ranges, or cite P/claim_C10.txt line 94 directly and have C10 admit the same runs.
- **Effect on the verdict.** None.
  - Every repeat value is a loss, and the archived 8.4(6)x loss holds (z 32, 19, delta* 0.71).
  - The memory verdict (competitive, no lead established) and every label are unchanged.
  - Reviewer 1 recomputed the other C31 numbers and found them correct.

### D3. C33, statistical standard (agreement 0.7)

- **Current text.**
  - Threshold labels are 0 wins, 22 ties and 85 losses with stat plus syst errors.
  - Pilot labels need 3 sigma under (a), (b) and (c) at delta 0.37, the top of a 0.065-0.37 range.
  - Among qubit-matched pilots, one lead holds (hw_tr, r = 6d, worse basis, 0.384(39)).
  - At delta 0.065-0.074, six more leads and five more losses would hold.
  - "One trend is significant": the strong-Z-bias Lambda slope, at 3.9 sigma.
- **Objections (reviewers 2, 3, 5; major).**
  1. **Lead count (reviewers 2, 3; M1).** The count of leads that would hold at delta 0.065-0.074 omits the phen_b10 q = 0 18/16 lead (0.62(4), z 7.4, 4.4, delta* 0.144; P/bq/cc4.log). It does count the q = 0 50/49 L_Y loss. The count is seven, not six.
  2. **Shared offsets (reviewer 2; M2).** The shared-offset check is applied only to the r = 6d lead. Both XYZ^2 runs of the phen 98/100 loss are in one batch log. With a shared offset its delta* is 0.275 and z at 0.37 is 2.27, which makes it delta-dependent.
  3. **Notation (reviewer 2).** Writing 6.347(16)e-5, 6.227(16)e-5, 6.278(16)e-5, 6.388(16)e-5 and 7.248(19)e-6 understates the errors tenfold. The files give +-1.6e-6 and +-1.9e-7, so the values should read 6.35(16)e-5 and so on. This document uses that notation.
  4. **Lambda trend (reviewers 2, 3, 5; M4).** "One trend is significant" rests on stat errors only. With lambda.csv's Lambda_change as a relative systematic, the 3e-3 to 2e-3 comparisons are 2.54 sigma (biased100) and 2.48 sigma (purez). The four comparisons together span 1.8-2.6 sigma, consistent with C9's "not established". Proposed: "One trend reaches 3.9 sigma with stat errors only; with Lambda_change added as a systematic it is 1.8-2.6 sigma (C9), so it is not established at stat plus syst."
  5. **Poisson bound (reviewers 2, 5).** "At least 107x" should be 106.6x, as in C4. This document uses 106.6x.
  6. **Lower delta estimate (reviewer 2; M3).** P/r6om/mom.py selects XYZ^2 BM sets by the substring 'xyz2'. It therefore misses the sets keyed 'biased10@both-X' and 'biased10@both-Y' in P/r6om/homog.out. Over all 37 XYZ^2 BM sets, chi2 is 74.4 on 64, W is 2226 and delta is 0.068, not 0.074 over 35.
  7. **Competitor scatter (reviewer 3).** The estimate from five XZZX counts (chi2 14.0 on 4, delta 0.044) includes one count (1268) that is not among the runs C10 uses. The four-count result (chi2 8.76 on 3, delta 0.039) is the one the claims support. Either value changes no label.
  8. **Exemption from (c) (reviewer 3; M6).** C33 calls offsets of non-BM XYZ^2 runs untested. That understates the evidence for exempting them:
     - hw_rr corr_gauge and MWPM runs differ by 0.002 in ln (0.33 sigma);
     - SI1000 MWPM runs from two sources differ by 0.94 sigma;
     - the biased10 corr_gauge sources agree within 0.4-1.2 sigma.
     The labels that depend on the exemption should be named.
  9. **Slope values (reviewer 5).** The 2e-3 to 1e-3 slopes 1.36(17) and 0.90(12) compute to 1.35 and 0.89, as in C9.
  10. **Delta range (reviewer 5; M8).** The top of the range is a one-degree-of-freedom point estimate, not an upper limit. The range should be described as running from the all-set estimate to the largest flagged-set estimate, not as a confidence interval.
- **Effect on the verdict.** None.
  - Items 1, 3, 5, 6, 7, 9 and 10 change no label at delta 0.37.
  - Item 2 would make one listed "loss that holds" (phen 98/100) delta-dependent under shared offsets. The same assumption makes the r = 6d lead delta-dependent. phen remains not useful, because its r = d threshold is also a loss.
  - Item 4 changes the headline "one trend is significant" to "not established at stat plus syst". No verdict rests on that trend.
  - Item 8 changes no label under the current rule.

### D4. OVERALL, the summary answer (agreement 7 of 9)

- **Current text.**
  - XYZ^2 has no established use on any modelled hardware, as a memory or for r = 1 MDR preparation.
  - Its one established lead is an idealized hw_tr memory point (0.384(39) of XZZX at r = 6d with BM), measured against XZZX only.
  - It is competitive, not useful, in hw_tr and phen_b10.
  - Both edges become losses for r = 1 preparation (18.5(7)x and 8.4(6)x).
- **Objections, reviewer 2 (major).** The answer is right, with these fixes:
  - "six more weaker-basis qubit-matched leads would hold" should be seven, adding the phen_b10 q = 0 18/16 lead (M1);
  - "phen ... Only 98/100 holds (delta* 0.39)" needs the same shared-offset caveat as the r = 6d lead; under it delta* is 0.275 (M2);
  - "0.074 over the 35 XYZ^2 BM sets" should be 0.068 over 37 sets (M3);
  - the conclusion's "XYZ^2's Lambda grows faster than XZZX's" should add "(3.9 sigma stat only; 2.5 sigma with the Lambda_change systematic, not established; C9)" (M4);
  - "wins only under biased100 (1.066, 4.2 sigma)", with correlated matching on both codes, is schedule-selected. The both schedule gives biased100@both-Y corr_gauge 0.6551(132)% against CSS corr 0.6150(78)%, which is 1.065, z 2.6, a tie (D/thresholds.csv);
  - "the biased100 ratio rises with d (0.73 to 1.10)" at equal distance is only about 2 sigma, stat only.
- **Objections, reviewer 3 (major).** The answer does not change, with these fixes:
  - there are seven such leads, not six. The 'Not lower' list names the q = 0 50/49 losses but not the q = 0 18/16 lead (M1);
  - add "(about 2.5 sigma with the Lambda_change systematic, C9)" to the 3.9-sigma Lambda trend (M4);
  - against CSS D = 4's weaker basis, the L_Y loss at 35 qubits is 1.53-1.59x, not 1.53-1.62x;
  - "17.8-18.8x" does not cover the listed inputs. They reach 1.201e-3/6.164e-5 = 19.5x (P/bq/si.log MWPM against P/r7rev6/rect_6x3_X.txt);
  - the convention "Pilot z values are quoted as (a), (b), (c) at delta 0.15/0.25" does not match labels that use delta*. This document quotes (a), (b) and delta*;
  - the SD6 headline 6.5-693x uses fixed bases. Using the weaker basis of each code at each point it is 7.5-470x (M7).
- **Effect on the verdict.** None.
  - Both objecting reviewers state that the answer is correct: no established use, and one idealized hw_tr lead.
  - The corrections change counts and ranges.
  - Under the both schedule, the biased100 correlated-matching win over CSS becomes a tie. C7 already rates the relative win over CSS not useful.

## 8. Material omissions raised in the last round

Each item is referenced at the numbers it bears on in sections 1-6. None changes a verdict.

| # | Omission | Source | Claims | Effect on the verdict |
|---|---|---|---|---|
| M1 | The phen_b10 q = 0 lead at 18/16, p = 3%, weaker basis against weaker basis (XYZ^2 BM L_Y 1.958e-3, 426 errors; XZZX d = 4 corr L_X 3.141e-3, 596 errors) is 0.623(40), z 7.45 and 4.38, delta* 0.144. It is missing from the count of leads that hold at delta 0.065-0.074, which becomes seven | P/bq/cc4.log | C33, C31 | None; makes the lead and loss bookkeeping symmetric |
| M2 | Both XYZ^2 runs of the phen 98/100 loss are in one batch log (t = 104 s and 106 s). With one shared offset, delta* is 0.275 and z at delta 0.37 is 2.27. The loss becomes delta-dependent under the shared-offset check applied to the r = 6d lead | P/bq/phen.log | C33, C18 | None; phen remains not useful (threshold loss) |
| M3 | The lower delta estimate over all 37 XYZ^2 BM replicate sets is 0.068 (chi2 74.4 on 64, W 2226), not 0.074 over 35. P/r6om/mom.py misses the sets keyed biased10@both-X and biased10@both-Y. At 0.068 the phen_b10 2.5% L_X lead (delta* 0.069) sits on the boundary | P/r6om/homog.out, P/r6om/mom.py | C33, C31 | None; no label at delta 0.37 changes |
| M4 | The strong-Z-bias Lambda slope difference is 3.94 and 3.85 sigma stat only, and 2.54 and 2.48 sigma with Lambda_change as a systematic. Over all four comparisons it is 1.76-2.63 sigma, so it is not established at stat plus syst | D/eq/lambda.csv | C6, C9, C30, C33 | None; the trend is not a basis of any verdict |
| M5 | The faster Lambda growth is measured with corr (corr_gauge) only. With BM in its worse basis, XYZ^2's d = 5 to 7 Lambda is 9.0(1.1) at 3e-3 and 11.3(3.1) at 2e-3, a slope of about 0.6(7) against 1.35-1.6 for XZZX corr. The trend should be labelled corr-only | D/eq/points_eps.csv, D/eq/decoders.csv, D/thresholds.csv | C6, C9, C23, C30, C33 | None; qualifies the open question at p < 1e-3 |
| M6 | Model (c) applies only to XYZ^2 BM runs. Applied to every XYZ^2 run at delta 0.37, it makes these delta-dependent: the hw_rr corr_gauge and MWPM losses (delta* about 0.25), the SI1000 35-qubit L_Y loss (about 0.14), SI1000 99/97 corr_gauge (0.30-0.33), Helios (0.34), phen_b10 q = 0 corr_gauge L_Y (0.21) and the square-patch corr_gauge lead (0.13). H2, SD6 99/97, SI1000 8x6, the 35-qubit rectangles, hw_tr r = 1, biased10 and phen 98/100 survive. Agreement across decoders and sources supports the exemption | P/rv/, P/bq/si.log, P/ovc/resr1.jsonl, P/bq/ion.log | C33, C11, C20, C36 | None; names the labels that rest on the exemption |
| M7 | Using each code's weaker basis at each point, the SD6 equal-qubit range is 7.5-470x, not the fixed-basis 6.5-693x. At the 693x pair CSS's weaker basis is X (3 errors, 5.29e-10), and at 99/97 XYZ^2's weaker basis is Y | D/eq/points_eps.csv, D/eq/pairs.csv | C4, C33 | None; every pair remains a large loss |
| M8 | The top of the delta range, 0.374, is a one-degree-of-freedom point estimate with no useful upper limit. Pooling the four flagged sets gives about 0.20 (chi2 46.6 on 17, W 747). The range is a set of estimates, not a confidence interval, so the r = 6d lead is established by the convention of testing at 0.37 | P/r6om/, P/cons/x3_X_a.txt, P/cons/x3_X_b.txt | C33, C35 | None; makes the one established lead conditional on that convention |
| M9 | In Hetenyi-Wootton's layout XYZ^2 reads plaquettes and links in two readout rounds per cycle. The per-cycle estimate 0.96(7)x is therefore 0.479(33)x per unit time (z 10.8, 6.4, delta* 0.24): a delta-dependent lead with 1.33x the qubits. Against XZZX interpolated to 100 qubits it is 1.85-1.99x worse per cycle and 0.92-0.99x per unit time. XZZX d = 6 was not run | 2306.17786.txt lines 1082-1084, 1884-1888, 2042; P/cons/z3_X.txt; P/litrev/hwtr_xyz2_Y.txt | C37, C25, C38 | None; no established lead appears under either normalization |
| M10 | Including the archived run P/r7rev6/rect_6x3_X.txt (CSS 6x3 X 6.164e-5, 1519 errors), the SI1000 35-qubit loss range is 17.8-19.0x | P/r7rev6/rect_6x3_X.txt | C11 | None; the upper end moves within the per-point errors |

## 9. Agreement record

Fraction of reviewers agreeing with each claim in the last round. There were 10 reviewers, and 9 of them responded on OVERALL.

| Claim | Agreement | Claim | Agreement | Claim | Agreement |
|---|---|---|---|---|---|
| C1 | 1.0 | C14 | 1.0 | C27 | 1.0 |
| C2 | 1.0 | C15 | 1.0 | C28 | 1.0 |
| C3 | 1.0 | C16 | 1.0 | C29 | 0.7 |
| C4 | 1.0 | C17 | 1.0 | C30 | 1.0 |
| C5 | 1.0 | C18 | 1.0 | C31 | 0.6 |
| C6 | 1.0 | C19 | 1.0 | C32 | 1.0 |
| C7 | 1.0 | C20 | 1.0 | C33 | 0.7 |
| C8 | 1.0 | C21 | 1.0 | C34 | 1.0 |
| C9 | 1.0 | C22 | 1.0 | C35 | 0.8 |
| C10 | 1.0 | C23 | 1.0 | C36 | 1.0 |
| C11 | 1.0 | C24 | 1.0 | C37 | 1.0 |
| C12 | 1.0 | C25 | 1.0 | C38 | 1.0 |
| C13 | 1.0 | C26 | 1.0 | OVERALL | 0.78 (7/9) |

Claims at or above 80% agreement, by round:

| Round | Claims reviewed | Claims at >= 80% | OVERALL |
|---|---|---|---|
| 1 | 30 | 17 | not reviewed |
| 2 | 34 | 27 | not reviewed |
| 3 | 36 | 30 | not reviewed |
| 4 | 37 | 22 | not reviewed |
| 5 | 38 | 36 | 0.625 |
| 6 | 38 | 36 | 0.125 |
| 7 | 38 | 34 | 1.0 |
| 8 | 38 | 35 | 0.78 |

Twelve claims were at or above 80% in every round in which they were reviewed. C1, C4, C5, C12, C13, C18, C22, C26, C27, C28 and C30 were at or above 80% in all eight rounds, and C38 in rounds 5-8. OVERALL reached 80% in one of its four rounds.