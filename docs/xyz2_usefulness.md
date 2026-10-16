# When is the XYZ² code useful? A multi-agent consensus

**Question (user, 2026-10-15).** In which instance (noise model, hardware, metric, application) is the XYZ² code useful compared with the rotated CSS, XZZX and XY surface codes and the honeycomb Floquet code, and where is it not? A negative result is important information.

**How this was produced.** A workflow (`wf_806a106f-964`) gave 22 agents one lens each (noise families, below threshold, state preparation, hardware platforms, decoders, overhead, literature, statistics, fairness, a steelman and a devil's advocate). Each read the campaign data, and some ran small pilots. A synthesizer merged their claims, and 10 adversarial reviewers checked every claim against the files in each round. The synthesizer then revised. Reviewers were told to disagree unless a claim was backed by the files.

- Rounds 1 to 3 completed: 17/31, 27/36 and 30/37 claims at ≥ 80% reviewer agreement.
- Round 4 (the cap) and the final write-up agent did not complete: the Claude session was cut off repeatedly.
- This document is the round-3 revision, with its claims, verdicts and the round-3 agreement record.
- The full draft, with every claim's evidence, is in `docs/data/campaign/usefulness_consensus_draft_r3.json`.
- The agents' pilot scripts and outputs are in `docs/data/campaign/usefulness_pilots/`. They are small, short runs, not campaign-standard, and the claims that rest on them are marked as pilots.

## Short answer

No simulated device model favours XYZ^2. That negative result is the answer for every hardware platform modelled here. The only significant advantage is in an idealized diagnostic, and every device-like change tried removes it, so it does not count as a use case.

**Rules used here.**
- A win or loss needs |z| >= 3, and this revision applies that everywhere.
- Pilots are labelled as pilots. Their scripts and outputs are archived in `docs/data/campaign/usefulness_pilots/`.
- I checked the disputed numbers myself, from the repository summaries and from those pilot files over ssh.

**Coverage.**
- 18 campaign models: 12 gate models (helios_p and h2_p have crosstalk and no threshold), em3, hyb and hyb_lr2, the diagnostic sd6_il, and the non-hardware phen and phen_b10.
- The 4 Hetenyi-Wootton spin corners, plus the hw_tr diagnostic.
- The gap diagnostics.
- Head-to-head thresholds at r = 1, 2, 3, 5, 10 and d.
- Equal-qubit runs at r = 3d. Stage 3 is merged at f2ee14b, and five BM series are still incomplete.

**Device models: XYZ^2 is never ahead.**
- Threshold at r = d (gate and hybrid models, no crosstalk): 0.42-0.85 of the best surface code. Over r = 1-10: 0.46-0.98. 103 of 107 head-to-head rows are nominally lower, and none is significantly higher.
- Symmetric rules (both schedule, matched decoders): SD6 0.60, SI1000 0.83, bias 0.62/0.49/0.41, ions 0.75-0.76.
- Spin corners: hw_g 0.78 of the surface codes, hw_g1q third of four, hw_t 0.13 of XZZX. hw_rr ties in threshold but is 2.8-2.9x worse at 99/97 qubits below threshold (C20).
- em3: 1.29-1.36x the cat-state surface compilations, but only 0.16-0.32 of the honeycomb.
- Equal qubits below threshold:
  - SD6: 6.5-693x worse than CSS (6.5-332x without the 693x pair, which rests on 2 CSS errors).
  - Strong bias, corr against corr: 10-17x worse than XZZX at 99/97, 154-367x at 323/337, and 82-850x at 483/449. The 82x and 850x pairs rest on 1 and 3 XZZX errors.
  - XYZ^2 on BM: 3.4-6.9x worse, then 28x and 51x at 323/337.
  - Circuit-level eta = 10 pilot: 3.3x worse (BM).
- Single-round (r = 1) MDR preparation: below every competitor's best basis, with two exceptions.
  - SI1000 is a tie: 0.92 of CSS-Z at 2.3 sigma with matched decoders.
  - hyb_lr2 ties only with decoders that only XYZ^2 gets; under symmetric rules it is a 3.1 sigma loss.
  - At fixed footprint under SI1000, XYZ^2 needs 1.0x, 1.24x and 1.33x a CSS patch's qubits for the same error. Its 1.5x lead over the 31-qubit square patch at 35 qubits comes from square-patch granularity.

**The only measured edge (C35): an idealized diagnostic.** hw_tr has noiseless gates, readout and reset. Its only noise is dephasing at eta = 20 on the data qubits during ancilla readout and reset. The links are XX, the memory covers both bases, and the comparison is at equal qubits against square XZZX.
- XYZ^2's error per round is 0.61-0.68 of XZZX's at 99/97 qubits, 0.54(10) at 195, and 0.38(4) at r = 6d. It matches XZZX's error with 0.81-0.82 of XZZX's qubits.
- With 0.1% gates and 1% readout it is 0.70(7) at 99/97 and a tie at 195.
- At eta = 10 it is a tie, and at eta = 100 it is level.
- It is gone with 2% gate-layer dephasing (1.02(8)).
- It becomes a loss at Hetenyi-Wootton's QEC-optimal readout time (1.72(16)) and with a full-strength first noise layer (1.44(9)).
- Their two-readout-window spin layout makes it about 6x worse (inference) (C37).
- Untested: CSS and the honeycomb at equal qubits; rectangular, co-prime or d^2-frame XZZX with ML decoding (2112.06036 Fig. 7c predicts parity at code capacity); reset and single-qubit gate noise.
- This reproduces the 2112.06036 code-capacity regime with MDR boundaries. It is not a device instance.

**A second edge in a non-hardware model (C31, phen_b10).**
- 0.70(5) at 18/16 data qubits and 0.81(4) at 50/49 (p = 3%).
- It fades to 0.91(8) at 98/100 and is gone at p = 1.5% (0.99(16)).
- It needs BP or BM decoding, and with noiseless readout it reverses (1.44(9)).

**Everything else is a tie, or a relative win over a code that is itself beaten on that hardware.**
- phen: r = 1, 2, 3, 5 and 10, and r = d against CSS.
- phen_b10: r = d against XZZX.
- si1000 at r = 1-10, si1000_lr2, sd6_il at r = 2, 5 and 10, h2_p_noxt at r = 3, hyb_lr2 at r = 1-2.
- hw_g against the honeycomb.
- The honeycomb on gate hardware; square CSS under bias; em3's cat-state compilations.

**Conclusion (recommendation, inference).** XYZ^2 is not a useful memory or state-preparation code for any modelled hardware. The paper's contribution is the fault-tolerant MDR protocol and its decoders. The circuit-level losses bear out the risks 2112.06036 itself flagged: measurement noise on weight-6 checks, and practical value conditional on a circuit-level analysis.

## Every claim, with its verdict and the round-3 agreement

Verdicts: `useful` is a measured win over the named comparator only; `competitive` is a tie or a contested edge; `not_useful`; `untested`. Agreement is the share of the 10 round-3 reviewers who agreed with the claim as worded in round 3. Claims below 0.8 are contested.

| id | regime | verdict | round-3 agreement |
|---|---|---|---|
| C1 | Generic depolarizing gate noise (SD6, platform-neutral) and SI1000 (superconducting-inspired); links repeated (sd6_lr2/lr3, si1000_lr2). Memory threshold at r = d, weaker basis, best decoder of each code, finite-size fits d = 11-21 | not_useful | 10/10 |
| C2 | Gate models without crosstalk (sd6, si1000, biased10/100, purez, helios_p_noxt, h2_p_noxt, lr and il variants). Threshold at r = 1, 2, 3, 5, 10 and d; head-to-head data exist only at these r. Fits at r = 4, 6-9 and 11-21 exist for XYZ^2 only | not_useful | 10/10 |
| C3 | SI1000-type noise at short round counts (si1000 r = 1; si1000_lr2 r = 1-3) and hyb_lr2 r = 1-2 against the gate surface codes; threshold | not_useful | 10/10 |
| C4 | Equal qubit count below threshold, SD6, p = 1e-3 to 3e-3, r = 3d. CSS stands for XZZX and XY, which have isomorphic detector error models under SD6. Measured 35/17-qubit points included | not_useful | 9/10 |
| C5 | Z-biased circuit noise (biased10, biased100, purez): bias-preserving Z-type channels after every CX/CY/CZ, measurement and reset flips at p. Threshold at all r, against XZZX and XY | not_useful | 9/10 |
| C6 | Equal qubit count below threshold under strong Z bias (biased100, purez), p = 1e-3 to 3e-3, r = 3d, vs XZZX; measured d = 3 points included | not_useful | 10/10 |
| C7 | Strong Z bias (biased10/100, purez), XYZ^2 vs the square, untailored rotated CSS code only (threshold and equal-qubit data) | not_useful | 10/10 |
| C8 | Strong Z bias (biased100, purez), equal code distance (about twice the qubits), p = 1e-3 to 3e-3 | not_useful | 10/10 |
| C9 | Mechanism: MDR preparation/readout time boundaries vs the bulk of a memory; pure-Z and biased gate noise; perfect-syndrome idle noise (hw_tr) vs phenomenological q = 0 | not_useful | 5/10 |
| C10 | Single-round MDR state preparation (r = 1), threshold, the basis MDR protects (depth6 L_X, fault distance d+1), against each competitor's weaker basis and its best single basis | not_useful | 9/10 |
| C11 | r = 1 preparation at fixed footprint or fixed device: SI1000 and SD6 at p = 1e-3, and the trapped-ion device points with crosstalk (Helios 98 qubits, H2 56). MWPM/corr/BM pilots, not campaign data | not_useful | 7/10 |
| C12 | Trapped-ion one-parameter models without crosstalk (helios_p_noxt, h2_p_noxt), r = d and r = 1 | not_useful | 10/10 |
| C13 | Trapped-ion models with mid-circuit-measurement crosstalk (helios_p, h2_p) | not_useful | 10/10 |
| C14 | Native pair-measurement hardware (em3), threshold at all r | not_useful | 10/10 |
| C15 | Hybrid hardware: gates plus native pair measurements at the same rate (hyb, hyb_lr2); one reading of the resource count in 2112.06036 | not_useful | 8/10 |
| C16 | Gate-based hardware, XYZ^2 vs the honeycomb Floquet code only | not_useful | 9/10 |
| C17 | Phenomenological noise with bias eta = 10 (phen_b10, model of arXiv:2505.03691, not hardware), threshold at all r | competitive | 10/10 |
| C18 | Phenomenological depolarizing noise (phen, eta = 0.5; not hardware), threshold at all r and equal qubits | not_useful | 10/10 |
| C19 | Hetenyi-Wootton spin-qubit corners: gates (hw_g, hw_g1q) and eta_T = 20 idling in every gate layer as well as readout (hw_t); r = d, d = 11-21; both schedule, matched decoders | not_useful | 8/10 |
| C20 | Spin-qubit readout-only corner (hw_rr: outcome flips only): threshold at r = d, and an equal-qubit probe at p = 4% (0.38 of threshold), r = 3d, 99/97 qubits | not_useful | 7/10 |
| C21 | Spin-qubit idling only during ancilla readout and reset (hw_tr diagnostic): threshold ranking from MWPM pair crossings, d <= 9; and Hetenyi-Wootton's fitted threshold plane | not_useful | 6/10 |
| C22 | Gap diagnostic (SD6 variants, not hardware): noiseless links and/or no idle errors; links repeated; links as pair measurements; schedule search | not_useful | 9/10 |
| C23 | Decoder dependence: MWPM, HCM, BM/BP, sequential two-level, CFE, Tesseract, TN ML; all models, including the pilot regimes of C31 and C35 | not_useful | 10/10 |
| C24 | Fairness and selection effects of the comparison rules | not_useful | 9/10 |
| C25 | Hardware cost at equal distance, and qubit overhead by layout | not_useful | 9/10 |
| C26 | Labelling of the campaign's biased noise models, and biased noise with gates that do not preserve bias | untested | 10/10 |
| C27 | Reported XYZ^2 'wins' or values in the campaign tables that are fit or coverage artifacts | not_useful | 9/10 |
| C28 | Noise that lacks the link Pauli: (a) Z-biased dephasing in the XX-link frame (tested; C35, C37); (b) amplitude-damping idle noise (p_X = p_Y >> p_Z) with ZZ links (untested) | untested | 10/10 |
| C29 | Long memories (r >> d) under circuit-level extreme Z bias (purez, biased100), full detector-error-model decoders, equal qubits | untested | 10/10 |
| C30 | Regimes and competitors not simulated | untested | 9/10 |
| C31 | Phenomenological noise with Z bias eta = 10 (phen_b10; readout flips at q = p on weight-6 and weight-2 checks alike; not hardware). Equal qubit count below threshold, r = 3d, p = 1.5-3%. Pooled pilots; pilot qubit counts are data qubits | competitive | 3/10 |
| C32 | Circuit-level Z-biased gate noise at eta = 10 (biased10, the bias of the original paper's equal-qubit claim and the realistic cat-qubit level), equal qubits 99/97, p = 3e-3, r = 3d; pooled pilots | not_useful | 10/10 |
| C33 | Statistical standard of the threshold comparisons and of the pilots (all models) | not_useful | 8/10 |
| C34 | Literature cross-check of the campaign's competitors and of the published case for XYZ^2 | not_useful | 9/10 |
| C35 | Idealized diagnostic hw_tr: - noiseless gates, readout and reset (perfect syndromes); - Z-biased Pauli dephasing (eta = 20) on data qubits only while ancillas are reset and read out (p_T per round); - XX links; both-basis memory; MDR time boundaries; - equal total qubits, r = 3d plus one r = 6d point; - XYZ^2 on BM against square XZZX on corr. Pilots, not campaign standard | competitive | 0/10 |
| C36 | Platform map: which device classes the campaign models represent, and whether they can host XYZ^2's circuit natively | not_useful | 8/10 |
| C37 | Device-like variants of the hw_tr corner (spin-like hardware): gate-layer dephasing, readout duration, two readout windows per cycle, larger bias, harsher initial boundary. 99/97 qubits, r = 3d, pilots | not_useful | added after round 3 |

## Claims in full

### C1: Generic depolarizing gate noise (SD6, platform-neutral) and SI1000 (superconducting-inspired); links repeated (sd6_lr2/lr3, si1000_lr2). Mem

**Verdict:** not_useful (confidence 0.95, round-3 agreement 10/10).

Measured. XYZ^2 is far below the rotated surface codes.
- SD6: 0.63 of their threshold (14.0 sigma vs XZZX, 21.3 sigma vs CSS).
- SI1000: 0.83 (8.2 sigma vs CSS).
- Same decoder for every code: MWPM 0.57-0.58 (SD6) and 0.77 (SI1000); BM 0.85 (SI1000).
- Matched rules (both schedule; mwpm/corr_gauge/bm against mwpm/corr/bm): SD6 0.487(7)%, which is 0.60 of XZZX.
- Repeating the links does not close the gap (0.65-0.85).
- With idling removed for both codes (sd6_noidle, a diagnostic) XYZ^2 is still at 0.774(8). The loss is not only an idle-convention effect.

Under SD6 the CSS, XZZX and XY detector error models are isomorphic. Pooled they give 0.809(9)% against the 0.813 maximum, so taking the competitor maximum adds under 1%. For SI1000 the isomorphism is an inference from equal measured thresholds.

Hardware reading (inference):
- SD6 is a generic model, not a superconducting one. Only SI1000 is superconducting-inspired.
- This repo's XYZ^2 circuit needs degree-6 couplers. On square-grid or heavy-hex chips it would need routing, which was not modelled, so these ratios are optimistic there.
- Alternatively it could use a 2-body compilation such as Wootton's heavy-hex hexagonal matching code (2109.13308). That code was run only on small devices and has no threshold study.
- 2112.06036 itself warned that the weight-6 checks could make XYZ^2 more susceptible to measurement noise.

**Evidence.** docs/data/campaign/competitors_table.csv and thresholds.csv, r = d:
- SD6: sd6@both-X bp_corr 0.512(8)% vs xzzx-Z bm 0.813(20), css-X corr 0.809(12), xy 0.807(21). z 14.0 and 21.3.
- SD6 MWPM: 0.400 vs 0.685-0.697.
- SI1000: si1000@both-X bm 0.431(8) vs css-Z bm 0.520(7), z 8.2. MWPM 0.340 vs 0.440-0.441; BM 0.431 vs 0.510.
- Link repeats: sd6_lr2 0.70, sd6_lr3 0.65, si1000_lr2 0.85.
gap/gap.csv: sd6_noidle 0.854(6) vs XZZX 1.103(9). gap/fractions.csv: 0.774(8).
Error calibration: see C33.
Literature:
- Hetenyi-Wootton Table I with their rule 1/p = sum 1/p_i gives XYZ^2 0.408, rotated surface 0.648, XZZX 0.351 (recomputed). Their XZZX ranking is the opposite of ours, so the 0.63 agreement is a cross-model coincidence.
- 2109.13308.txt lines 1-14, 40-110.
- 2112.06036.txt lines 169-178.
Coupler degree from Stim builds: XYZ^2 maximum 6, XZZX and CSS maximum 4.

### C2: Gate models without crosstalk (sd6, si1000, biased10/100, purez, helios_p_noxt, h2_p_noxt, lr and il variants). Threshold at r = 1, 2, 3, 5,

**Verdict:** not_useful (confidence 0.9, round-3 agreement 10/10).

Measured. XYZ^2 is nominally below the best competitor at every round count. The highest ratio is 0.98 (si1000 and si1000_lr2 at r = 1).

Under |z| >= 3 several of these are ties, not losses:
- si1000 at r = 1, 2, 5 and 10;
- si1000_lr2 at r = 1-10;
- h2_p_noxt at r = 3 (z -2.98);
- sd6_il at r = 2, 5 and 10. sd6_il at r = 1 is a loss by a hair (z -3.0).
No round count shows a significant XYZ^2 lead.

Every tie has the same sign. With stat-only errors they are 3.5-7.7 sigma losses. Only the fit-window syst term, which is not calibrated across codes (C33), turns them into ties.

The trend with r depends on the model:
- highest at r = 1 for SD6, SI1000 and strong bias;
- rising toward r = d for Helios;
- 0.71-0.76 for h2_p_noxt;
- U-shaped for biased10.
These trends are 1.5-2.6 sigma. Only the ranking is robust.

Fixed-r fits have larger nu: median 1.67-1.76 (5-95% range 1.27-2.43), against 1.05 at r = d. They are finite-r thresholds, not pseudo-thresholds.

The 0.98 at si1000 r = 1 is a depth6 L_Y TN-ML fit (C3). Under symmetric rules the ratios at r = 1/2/3/5/10/d are:
- sd6: 0.72/0.67/0.62/0.64/0.61/0.60;
- si1000: 0.89/0.90/0.87/0.85/0.85/0.83.
These are one reviewer's recomputation.

**Evidence.** competitors_table.csv, ratio at r = 1/2/3/5/10/d:
- sd6: 0.73/0.71/0.66/0.66/0.64/0.63
- si1000: 0.98/0.90/0.88/0.89/0.85/0.83
- biased10: 0.62/0.56/0.55/0.58/0.60/0.64
- biased100: 0.57/0.51/0.49/0.50/0.47/0.51
- purez: 0.56/0.50/0.50/0.46/0.46/0.42
- helios_p_noxt: 0.64/0.63/0.68/0.69/0.73/0.76
- h2_p_noxt: 0.71/0.76/0.73/0.73/0.76/0.76
z values recomputed by reviewers from thresholds.csv (z_tot / z_stat):
- si1000 r = 1/2/5/10: -0.2, -2.1 / -3.5, -2.0 / -4.5, -2.6 / -4.5.
- h2_p_noxt r = 3: 1.147(127) vs 1.565(59), -2.98 / -7.65.
- sd6_il r = 1: 2.132(82) vs 2.417(47), -3.0. r = 2/5/10: -1.4, -2.8, -2.0.
- si1000_lr2 r = 1-10: -0.2, -0.9, -0.8, -1.8, -1.4.
Symmetric ratios: reviewer script fairrev/sym.py (login13 scratchpad).

### C3: SI1000-type noise at short round counts (si1000 r = 1; si1000_lr2 r = 1-3) and hyb_lr2 r = 1-2 against the gate surface codes; threshold

**Verdict:** not_useful (confidence 0.85, round-3 agreement 10/10).

Measured. The nominal threshold ties need advantages only XYZ^2 gets:
- the TN ML and CFE-TN decoders;
- the best of 24-28 schedule-decoder candidates;
- the depth6 L_Y memory, whose fault distance is only (d+1)/2. The hyb rule refuses that kind of schedule to the surface codes.

Symmetric rules (both schedule; mwpm/corr_gauge/bm against mwpm/corr/bm), si1000 at r = 1, XYZ^2 0.959(16):
- 0.89 of CSS (2.0 sigma), a tie;
- 0.91 of XZZX (3.4 sigma), a loss;
- 0.906 of the pooled isomorphic competitors, 1.058(20) (3.9 sigma), a loss.
In the MDR basis (depth6 L_X) with matched decoders it is 0.92 of CSS-Z (2.3 sigma), a tie.

hyb_lr2:
- r = 1: ties the gate surface codes only with depth6-X TN ML (2.299(206) vs 2.417(45), 0.6 sigma). Under symmetric rules it is 2.067(103), 0.855 at z -3.1, a loss.
- r = 2: a tie with CSS (z -2.0).

si1000_lr2 at r = 1-3: ties under either rule set (symmetric 0.90-0.94, z -1.4 to -2.0).
si1000 at r = 3: a 3.3 sigma loss.

Below threshold the depth6 L_Y basis is weak. At d = 3, r = 1, p = 1e-3 it errs 3x more often than L_X and 2x more than a 31-qubit CSS patch.

A threshold tie at 1.1-1.5x the qubits is not an advantage (C10, C11).

**Evidence.** thresholds.csv, si1000 r = 1:
- depth6-Y tnml 1.059(91) (nu 2.41, drift 1.18).
- depth6-X: cfe_tn 1.061(46) (drift 1.57), bp_corr 1.053(57), corr_gauge 0.995(14), mwpm 0.969(16).
- both-X: mwpm 0.959(16), bm 0.958(31), corr_gauge 0.954(38), cfe_tn 0.996(70).
- Competitors: css-X bm 1.078(58), css-Z bm 1.083(35), xzzx-Z bm 1.055(23), xy-X bm 1.059(54).
Other rows:
- si1000 r = 3: 0.711(22) vs xy-X 0.812(21), z -3.3.
- hyb_lr2@both-Y r = 1: bm 2.067(103), mwpm 1.883(44). @both-X corr_gauge 2.074(41). @depth6-X tnml 2.299(206). Gate XZZX-Z corr 2.417(45).
- hyb_lr2 r = 2: 1.507(109) vs css-X bm 1.749(54).
Rerun (login13 scratchpad ovc/resr1.jsonl), SI1000, p = 1e-3, r = 1, d = 3: depth6-Y MWPM 3.45e-3, L_X 1.170e-3 (313,454 errors), CSS D = 4 1.72-1.81e-3.
Code: src/mdr/ft/extraction_schedule.py lines 159-163; competitor_circuits.py lines 41-45.

### C4: Equal qubit count below threshold, SD6, p = 1e-3 to 3e-3, r = 3d. CSS stands for XZZX and XY, which have isomorphic detector error models un

**Verdict:** not_useful (confidence 0.92, round-3 agreement 9/10).

Measured, far outside the error bars.

At equal qubit count XYZ^2's error per round is 6.5-693 times the CSS code's:
- The 693x pair rests on 2 CSS errors. Without it the range is 6.5-332x.
- The 1e-3 pair at 483/449 is only a bound.
- At 99/97 the range is 7.5-15.2x if each point takes XYZ^2's worse basis. pairs.csv fixes the X basis, but Y is worse at d = 5.
- The gap grows with size.
- Per tick (8 vs 6 ticks) the ratios shrink by at most 1.33x.
- The pairs are not exactly equal: CSS has 4% more qubits at 323/337, and XYZ^2 has 7.6% more at 483/449.

Smallest size, measured (d = 3, no extrapolation):
- XYZ^2 at 35 qubits has 0.81, 1.00 and 1.55 times the error of a 17-qubit CSS patch at p = 1, 2 and 3e-3.
- Against CSS log-interpolated between its measured 17- and 49-qubit points it is about 4-5x worse.
- interp.csv's 2.2-3.1x understates this, because it extrapolates the X basis.

Inference:
- At equal qubits XYZ^2 has 1/sqrt2 of the distance, so to win it needs Lambda_XYZ2 > Lambda_CSS^sqrt2. It reaches 0.27-0.33 of that.
- The exponent fraction ln Lambda_XYZ2 / (sqrt2 ln Lambda_CSS) rises as p falls, from 0.435 to 0.645(43). The error is inflated for chi2/dof up to 3.7. Under depolarizing noise the fraction is expected to approach 1/sqrt2, below the 1 needed to break even.

Every pair is corr against corr, so the decoders are matched. CSS was never run with BM, and under SD6 XYZ^2's BM is not better than its corr (measured to d = 11).

**Evidence.** data/campaign/eq/interim_report/pairs.csv (corr, 68% Wilson), XYZ^2/CSS at p = 1e-3 / 2e-3 / 3e-3:
- 99/97: 11.09 [9.90, 12.43], 8.54, 6.53.
- 323/337: 693 [329, 1463] (2 CSS errors), 118, 76.
- 483/449: 1e-3 is a bound from d = 13; 332 [262, 422] (30 errors); 129 [115, 145].
- MWPM: 5.9-170.
Worse basis at 99 qubits (points_eps.csv): XYZ^2 corr_gauge Y 4.47e-5, 4.72e-4, 2.03e-3 vs CSS d = 7 corr X 2.94e-6, 4.95e-5, 2.69e-4. Ratios 15.2, 9.5, 7.5.
d = 3 points, points_eps.csv rounds 3d:
- XYZ^2 corr_gauge Y 5.88e-4 / 2.75e-3 / 7.15e-3.
- CSS corr d = 3 X 7.27e-4 / 2.75e-3 / 4.62e-3.
- CSS d = 5 (49 qubits) 3.92e-5 / 3.35e-4 / 1.08e-3.
lambda.csv: 11.00(39) vs 13.84(123); 4.04(20) vs 6.56(14); 2.34(9) vs 3.98(5). Break-even 41.1, 14.3, 7.05.
fits.csv: xyz2 sd6 corr 2e-3 chi2 7.45 on 2 dof; css sd6 Z corr 1e-3 chi2 7.12 on 2 dof.
targets.csv: 244 vs 120, 730 vs 245, 2029 vs 455 (730, 2029 and 455 extrapolated).
decoders.csv: CSS candidates 'corr mwpm'.

### C5: Z-biased circuit noise (biased10, biased100, purez): bias-preserving Z-type channels after every CX/CY/CZ, measurement and reset flips at p.

**Verdict:** not_useful (confidence 0.93, round-3 agreement 9/10).

Measured.

XZZX beats XYZ^2 at every r (|z| >= 6.9), and the gap widens with bias:
- 0.64, 0.51 and 0.42 at eta = 10, 100 and infinity (z -23.9, -40.2 and -18.2 at r = d);
- XZZX with plain MWPM already beats XYZ^2 with its best decoder.

XY, in its weaker basis with BM (1.017 / 1.052 / 1.045):
- beats XYZ^2 at r = d (z 8.6-14.6), at r = 5 (3.3-5.7) and for biased10 at r = 2 (3.5);
- at r = 1 and 3, and for biased10 and purez at r = 10, its lead is 0.9-2.8 sigma, a tie.

The biased100 r = 10 entry is an artifact. XY's BM and corr fits there are drift-flagged, so the table falls back to XY MWPM 0.843(16). That puts XYZ^2 (1.029(45)) nominally 3.9 sigma above XY. It is not a lead over the best competitor.

XY was decoded only with mwpm, corr and bm. Its known advantage under bias needs tailored or ML decoders, so XY's ranking is decoder-limited. This does not touch the XZZX verdict.

The code-capacity similarity of thresholds in 2112.06036 (about 28% at eta = 10, near-ML decoding) does not carry over to circuit-level gate noise. Even XYZ^2's stronger basis is below XZZX.

Hardware note (inference from the literature notes):
- Kerr-cat CX bias maps to eta of about 3-59 in this repo's units.
- Dissipative cats have bias of order tens.
- So cat hardware spans roughly biased10 to near biased100, and purez is a limit, not a device.
- XYZ^2 loses at all three.
- 2112.06036 names Kerr-cat concatenation as the biased-noise route. Cat hardware actually deploys repetition codes, which were not run.

**Evidence.** competitors_table.csv and thresholds.csv, r = d:
- XYZ^2 0.748(12) / 0.836(5) / 0.857(7); both schedule 0.740 / 0.833 / 0.853.
- XZZX 1.175(13) / 1.655(20) / 2.061(65); XZZX MWPM 1.102 / 1.615 / 2.046.
z vs XY (biased10 / biased100 / purez), recomputed by reviewers:
- r = 1: 1.7 / 1.1 / 0.9
- r = 2: 3.5 / 2.3 / 2.2
- r = 3: 2.3 / 2.1 / 2.5
- r = 5: 5.7 / 3.4 / 3.3
- r = 10: 2.5 / -3.9 (artifact) / 2.8
- r = d: 14.6 / 8.6 / 11.1
biased100 r = 10: xy-X mwpm fss 0.843(16); xy-X bm only as a drift-flagged pair crossing, 1.192.
XYZ^2 stronger-basis maxima: 0.772 / 0.868 / 0.897.
2112.06036.txt lines 34-37, 1095-1100 and 1318-1323.
lit_hardware.md table A and notation paragraph. papers/2409.13025.txt (dissipative cats).

### C6: Equal qubit count below threshold under strong Z bias (biased100, purez), p = 1e-3 to 3e-3, r = 3d, vs XZZX; measured d = 3 points included

**Verdict:** not_useful (confidence 0.88, round-3 agreement 10/10).

Measured. XZZX beats XYZ^2 at every resolved pair, rate and decoder. The gap grows with size under every decoder.

Correlated matching on both sides:
- 10-17x at 99/97 qubits;
- 154-367x at 323/337;
- 82-850x at 483/449, where the 82x and 850x pairs rest on 1 and 3 XZZX errors;
- XYZ^2 needs 1.6-2.9x XZZX's qubits for 1e-6 per round.

XYZ^2 on BM against XZZX on corr, XZZX's lower-error decoder at the overlapping points:
- 3.4-6.9x at 99/97;
- about 8-17x at 195 (XZZX interpolated);
- 28x and 51x at 323/337 (p = 3e-3). These are XYZ^2's X basis only; X was the weaker basis at d = 7, and no Y-basis BM point exists at d = 9.
- BM against BM where XZZX BM exists: 2.8-4.6x at 99/97.
- At 35 qubits (measured d = 3, BM): about 2-3x worse than XZZX log-interpolated to 35 qubits.
- With BM, the qubits for 1e-6 per round at 1e-3 are about 1.4-1.55x XZZX's (inference; the d = 7 BM points have 0-1 errors).

At p = 1e-3, XZZX d = 11 (241 qubits) beats XYZ^2 d = 9 (323 qubits) by 4.5x and 6.2x, worse basis against worse basis. That rests on 2-3 XZZX errors and uses corr only.

Lambda at 1e-3, corr:
- From the well-measured d = 5 to 9 points, XYZ^2's Lambda is nominally above XZZX's: about 38 and 41 against 25 and 31.
- The d = 11 endpoint (1 error each) gives 21.5 against 27.
- Either way the exponent fraction ln Lambda_XYZ2 / (sqrt2 ln Lambda_XZZX) is 0.66-0.80, below 1. It rises from 0.56-0.58 at 3e-3. The 1e-3 values come from fits with no spare degrees of freedom.

Inference: extrapolating ln Lambda linearly in ln p from 2e-3 to 1e-3, the exponent gap closes only near p of about 5e-9 (biased100) and 4e-6 (purez). The 10-17x gap at 99/97 would remain even then. So p < 1e-3 is untested, but no practical crossover is expected.

CFE, the strongest XYZ^2 decoder, gains only 11-28% over BM.

**Evidence.** pairs.csv, corr XYZ^2/XZZX:
- 99/97: 10.39-17.39.
- 323/337: 154 [108, 220] to 367 [309, 436].
- 483/449: 309 [219, 435], 82.2 [26.5, 255.2] (1 error), 850 [455, 1586] (3 errors).
points_eps.csv:
- BM X at d = 9, 3e-3: biased100 1.81e-6 (16 errors) vs XZZX d = 13 corr X 6.41e-8 = 28 [20, 39]; purez 1.51e-6 vs 2.97e-8 = 51 [35, 73].
- 1e-3 corr worse basis: XYZ^2 d = 5 Y 4.07e-6 and 4.09e-6; d = 9 2.76e-9 (11 errors) and 2.40e-9 (9). XZZX d = 5 1.07e-5 and 7.83e-6; d = 9 1.68e-8 and 7.95e-9.
- XZZX d = 11 worse basis Z: 6.08e-10 (3 errors) and 3.89e-10 (2).
- d = 3, biased100 1e-3: XYZ^2 BM Y 1.19e-4 vs XZZX d = 3 X 2.66e-4, d = 5 X 1.07e-5 (log-interpolated at 35 qubits: 4.4e-5, so 2.7x worse). Same comparison for purez 1e-3: 3.1x; biased100 3e-3: 2.2x.
- BM vs BM at 99/97: 2.8, 3.1 (biased100); 3.8, 4.6 (purez).
targets.csv: 134/279/566 vs 82/142/214 (biased100); 132/272/547 vs 75/127/189 (purez).
lambda.csv: 31.4(36), 41.5(48) vs 25.2(18), 31.5(24); fits.csv dof 0 at 1e-3.
paper/manuscript/Results.tex line 239 (CFE vs BM).

### C7: Strong Z bias (biased10/100, purez), XYZ^2 vs the square, untailored rotated CSS code only (threshold and equal-qubit data)

**Verdict:** not_useful (confidence 0.85, round-3 agreement 10/10).

Measured. This is a relative win over CSS only.

Threshold:
- r = d: XYZ^2 beats CSS by 1.16-1.38x (6.7-17 sigma).
- r = 5 and 10: 0.0-2.9 sigma, ties under the rule.
- r = 3: biased10 has CSS nominally ahead (0.9 sigma); the other two have XYZ^2 nominally ahead.
- r = 1-2: CSS nominally ahead by at most 1.5 sigma.

Equal qubits:
- With matched corr decoding XYZ^2 is 3.5-58x better and needs 0.50-0.70 of CSS's qubits. Only the 1e-3 qubit fractions (0.68-0.70) are interpolated; the others use extrapolated targets.
- With MWPM it is 1.7-9.3x worse, so the gain relies on the gauge-aware decoder.

Not a use case:
- XZZX runs on a subset of the same gate types and dominates XYZ^2 (C5, C6).
- Rectangular or tailored CSS patches, the usual biased-noise baseline that fair_comparison.md section 2 requires ('each code uses its best compilation'), were not run. Inference: they would likely remove this relative win too.

**Evidence.** thresholds.csv, weaker basis:
- r = d: 0.748(12) / 0.836(5) / 0.857(7) vs CSS 0.645(10) / 0.615(25) / 0.620(12); z 6.7 / 8.7 / 16.6.
- r = 5: z 0.0 / 2.0 / 2.3. r = 10: z 0.8 / 1.8 / 2.9. r = 3: z -0.9 / 0.4 / 0.8.
pairs.csv, XYZ^2/CSS: corr 0.017-0.287; MWPM 1.67-9.27.
eq.md: qubits-per-logical table with 'extrapolated' flags. targets.csv ratios 0.50-0.70.
docs/fair_comparison.md line 13. lit_hardware.md lines 128 and 133.

### C8: Strong Z bias (biased100, purez), equal code distance (about twice the qubits), p = 1e-3 to 3e-3

**Verdict:** not_useful (confidence 0.75, round-3 agreement 10/10).

Measured at points and from fits; preliminary.

At equal distance, XYZ^2's worse-basis error under strong bias is below XZZX's only at about 1e-3 with corr, and at all tested p with BM. This is the below-threshold trace of XYZ^2's large pure-Z distance (2d^2 vs d, 2112.06036).

p = 1e-3, corr, measured worse-basis points. XYZ^2 is lower by:
- biased100: 2.6x, 3.0x and 6.1x at d = 5, 7 and 9. The d = 9 point rests on 11 errors (+-30%).
- purez: 1.9x, 2.4x and 3.3x.
- biased100 d = 11: 0.67 against XZZX's worse basis, unresolved (1 vs 3 errors).

p = 2e-3, corr:
- biased100: the ratio rises with d, 0.73, 0.79, 0.88 and 1.10 at d = 5, 7, 9, 11.
- purez: XYZ^2 is already worse, 1.00-1.55.
- XZZX's Lambda exceeds XYZ^2's at 2e-3 in both models.

p = 3e-3:
- corr reverses: 1.50-6.9x worse.
- BM: 0.41-0.73 at d = 7 and 0.44-0.67 at d = 9 against XZZX corr.

At p = 1e-3, d = 5, BM gives 0.18-0.21. MWPM is 6-950x worse.

The benefit is small. At 1e-3 it buys at most about 10% fewer rounds per logical operation (d 5.8 vs 6.2-6.4 for 1e-6 per round) at 1.6-1.8x the qubits. Qubits x d favours XZZX by 1.48-1.65x with corr, and by about 1.2-1.3x with BM (inference). The 2x qubit cost removes the advantage (C6).

**Evidence.** docs/data/campaign/eq/points_eps.csv, corr, worse basis.
1e-3, d = 5/7/9:
- XYZ^2 biased100 4.07e-6 / 1.30e-7 / 2.76e-9; purez 4.09e-6 / 9.84e-8 / 2.40e-9.
- XZZX 1.07e-5 / 3.92e-7 / 1.68e-8 and 7.83e-6 / 2.37e-7 / 7.95e-9.
- d = 11 biased100: 4.09e-10 (1 error) vs XZZX Z 6.08e-10 (3).
2e-3:
- biased100 6.87e-5 / 5.18e-6 / 4.41e-7 / 4.10e-8 vs 9.47e-5 / 6.55e-6 / 5.01e-7 / 3.72e-8.
- purez XYZ^2 Y 5.93e-5 / 4.29e-6 / 4.10e-7 / 2.82e-8 vs XZZX X 5.91e-5 / 3.68e-6 / 2.65e-7 / 1.87e-8.
3e-3:
- corr biased100 1.50 / 2.16 / 2.65 / 4.11; purez 1.65 / 2.96 / 4.87 / 6.94.
- BM d = 7: 1.35e-5 and 1.41e-5 vs XZZX corr 3.30e-5 and 1.92e-5.
BM at 1e-3, d = 5: biased100 Y 1.90e-6, purez Y 1.63e-6 vs XZZX X 1.07e-5 and 7.83e-6.
lambda.csv. targets.csv: d 5.8 vs 6.4 and 5.8 vs 6.2; qubits 134/82 and 132/75.

### C9: Mechanism: MDR preparation/readout time boundaries vs the bulk of a memory; pure-Z and biased gate noise; perfect-syndrome idle noise (hw_tr

**Verdict:** not_useful (confidence 0.7, round-3 agreement 5/10).

Partly measured, partly inference, one part unverified.

1. Unverified (outputs not saved): CP-SAT bulk fault distances. Pure-Z upper bounds 18 (d = 3) and 50 (d = 5), lower bounds 5-7. At biased100, d = 3: L_X 4 and L_Y 3, against XZZX 3/3.

2. Measured from the code: the preparation boundary caps the fault distance.
- depth6 L_X: d+1.
- depth6 L_Y: (d+1)/2 under SD6.
- An even fault distance d+1 needs as many faults to fail as d, so it gains a prefactor, not an exponent (inference).

3. Measured: XYZ^2's r = 1 MDR-basis threshold is flat in bias (1.96-1.99%), while XZZX's and CSS-Z's rise strongly. In phen_b10, even TN ML reaches only about 15% at r = 1, against 25-28% for XZZX. So the boundary penalty exists without gates.

4. Measured, hw_tr pilot (eta 20, p_T 4%, 99/97 qubits).
- Only XYZ^2's L_X improves per round from r = 3d to r = 6d (1.86e-4 to 1.15e-4).
- L_Y is flat (1.22e-4 to 1.21e-4).
- The implied X bulk, 0.43(33)e-4, is only 1.3 sigma from zero.
- XZZX is flat.
- So the r = 3d figure contains a large L_X time-boundary term. A both-basis bulk advantage is shown only through L_Y's 0.38-0.40 ratio.

5. Measured (pilot) at eta 10, p = 3%: moving noise from the end of the memory to its start.
- phen q = 0 has a full-strength noise layer before round 1 and noiseless last syndromes. hw_tr has half layers at both ends. The bulk per round is the same.
- XYZ^2's error doubles (2.05x). XZZX's rises 14.5%.
- The 0.81(8) tie becomes a 1.44(9) loss, pooled over four XYZ^2 replicates.
- The two models use different builders (MPP checks vs ancilla circuits), so attributing this to the boundary is an inference.
- At 18/16 data qubits (d = 3 vs 4) the q = 0 ratio is still 0.62.

6. Measured: XYZ^2's Lambda grows faster than XZZX's as p falls. Slopes d ln Lambda / d ln(1/p) from 2e-3 to 1e-3 are 1.36 and 1.77 against 0.89 and 1.10. This is not bias-specific: under SD6 XYZ^2's slope also exceeds CSS's (1.35 vs 1.23, 1.45 vs 1.08), because XYZ^2 sits closer to its lower threshold. Only the absolute Lambda at 1e-3 is bias-specific (C6).

7. Measured: under strong-bias gate noise, removing the time boundary changes XYZ^2's 99-qubit error by only 20-46%. Its 6-15x deficit there is bulk-dominated (C29).

Conclusion: the MDR time boundary is a measured weak point of XYZ^2. The bulk helps only with perfect syndromes (hw_tr), and there only clearly in L_Y. MDR preparation, the paper's own protocol, sits on the weak side.

**Evidence.** src/mdr/ft/extraction_schedule.py lines 159-163. src/mdr/ft/ft_mdr_circuit.py lines 551-590 (_build_phen: full PAULI_CHANNEL_1 before every round, noiseless final readout).
thresholds.csv r = 1:
- depth6-X 1.960(73) / 1.966(107) / 1.974(110) / 1.989(75) (sd6 / biased10 / biased100 / purez).
- XZZX weaker 2.417 / 3.176 / 3.447 / 3.482. CSS-Z 2.441 / 4.921 / 5.869 / 5.636.
- phen_b10: XYZ^2 tnml 15.2-15.5 vs xzzx-X bm 28.4.
lambda.csv:
- sd6 XYZ^2 2.34 / 4.04 / 11.00, CSS 3.98 / 6.56 / 13.84.
- biased100 5.45 / 12.26 / 31.36 vs 7.84 / 13.56 / 25.20.
- purez 5.67 / 12.20 / 41.54 vs 9.12 / 14.71 / 31.53.
boundary.csv: XYZ^2 corr d = 5 change -0.20 to -0.46.
Pilots, login13 scratchpad:
- litrev/hwtr4_xyz2_X/Y (r = 15): 62/22,344, 44/24,063.
- cons/b_x_X_a/b (r = 30): 46/13,542, 47/13,601. cons/b_x_Y_a/b: 52/13,687, 47/13,611.
- cons/b_z_X: 3.163e-4; litrev/hwtr4_xzzx_X: 3.050e-4.
- Phen q = 0, Y: cons/cc_phen0_Y 112/34,800 plus bq/cc4.log 110, 99, 95 per 34,800, giving 2.00e-4 (416 errors, chi2 1.98 on 3 dof).
- XZZX: bq/cc4.log corr X 1.383e-4 (759 errors); bq/cc5.log XZZX bm X 1.236e-4 (53).
- hw_tr eta 10: cons/e10*_x_Y 99/67,777 shots (9.76e-5) vs cons/e10_z_X 1.208e-4 (7370).
- bq/cc4.log d = 3/4: XYZ^2 Y 1.958e-3 vs XZZX corr X 3.141e-3.
CP-SAT scripts (milp_*.py, bulk_d3.log) are not on disk.

### C10: Single-round MDR state preparation (r = 1), threshold, the basis MDR protects (depth6 L_X, fault distance d+1), against each competitor's we

**Verdict:** not_useful (confidence 0.88, round-3 agreement 9/10).

Measured. In the MDR basis, XYZ^2's r = 1 threshold is below the best competitor on every hardware model except two ties.

Against the competitors' weaker basis:
- SD6: 0.81 (5.3 sigma).
- Ion models without crosstalk: 0.65 and 0.74.
- Bias: flat at 1.96-1.99%, while XZZX reaches 3.2-3.5%.

Preparing one state lets each competitor pick its best basis, and the ratios fall:
- SD6: 0.79 (3.9 sigma) against XZZX-X.
- Ions: 0.40 (Helios) and 0.58 (H2) against CSS-Z |0>.
- Bias: CSS-Z reaches 4.9-5.9%.

These entries use TN ML or CFE-TN, which no competitor gets. With matched decoders the SD6 value falls to 1.82%, 0.73-0.75 of XZZX. So the ratios above are XYZ^2-favourable upper values.

Exceptions:
- SI1000: a tie. Nominally 0.98 with XYZ^2-only decoders; 0.92 of CSS-Z at 2.3 sigma with matched decoders (C3).
- em3: 1.22x the best surface-code basis (4.1 sigma), but the honeycomb is 4.75x higher.

The extra fault distance d+1 buys nothing per qubit.
- Square rotated patches of distance 4 and 6 need 31 and 71 qubits, against 35 and 99.
- A rectangular patch puts distance 6 in the prepared basis on exactly 35 qubits (3x6), and distance 10 on 99 (5x10). This is untested (inference).

2112.06036 uses odd d and notes that even d behaves differently. L_X is the stronger depth6 memory basis at r = d only under SD6.

**Evidence.** thresholds.csv r = 1:
- depth6-X: sd6 cfe_tn 1.960(73) (matched decoders: bm 1.822(47), mwpm 1.809(19)); helios_p_noxt tnml 1.539(101); h2_p_noxt tnml 2.236(155); em3 cfe_tn 1.066(41); bias 1.966 / 1.974 / 1.989.
- XZZX: sd6 X bm 2.489(116), Z corr 2.417(45); helios_p_noxt Z mwpm 2.371; h2_p_noxt 3.031.
- CSS-Z: helios_p_noxt mwpm 3.836(60); h2_p_noxt bm 3.841(91); bias 4.921 / 5.869 / 5.636.
- SI1000: depth6-X corr_gauge 0.995(14) vs css-Z bm 1.083(35).
- em3: css-Z 0.853(31), xzzx-Z 0.838(22), xy-Z 0.875(23), honeycomb-X bm 5.068(338).
Rotated dx x dz patch: 2*dx*dz - 1 qubits.
2112.06036.txt lines 307-309.

### C11: r = 1 preparation at fixed footprint or fixed device: SI1000 and SD6 at p = 1e-3, and the trapped-ion device points with crosstalk (Helios 9

**Verdict:** not_useful (confidence 0.75, round-3 agreement 7/10).

Measured in pilots. Reruns agree, but the outputs are not in the repository.

At equal footprint, or with the best patch that fits the device, a surface code prepares the state better, except for one granularity effect at the smallest size.

SI1000, r = 1, footprint scan:
- XYZ^2 needs 1.0x, 1.24x and 1.33x the qubits that a CSS patch needs for the same error, at 35, 99 and 195 qubits. For example, CSS reaches XYZ^2's 99-qubit error (1.18e-4) with about 80 qubits.
- At 99/97 directly, XYZ^2 d = 5 is 2.1-2.7x worse than CSS d = 7. With BM it is 1.01e-4, still about 2.2x worse.
- At 35 qubits the XYZ^2 MDR basis (1.17e-3) beats the 31-qubit square CSS D = 4 patch (1.72-1.81e-3) by 1.47-1.55x, with high statistics. That lead is square-patch granularity. CSS log-interpolated to 35 qubits is level (about 1.2e-3).
- A 35-qubit 3x6 rectangular patch was not tested and would likely win (inference).
- L_Y is 2x worse than CSS D = 4.
- These errors are below the physical SPAM error (SI1000: reset 2p, measurement 5p).

SD6, r = 1, 99/97: XYZ^2 is 8.1x worse than CSS-X and 7.7x worse than XZZX-X. Against CSS-Z, a different state, it is 9.6x worse.

Helios, |+> preparation:
- The rerun used the standard 99-qubit XYZ^2 circuit, which does not fit the 98-qubit device. It matches the manuscript's 95-qubit ancilla-reuse value (8.4e-5).
- XYZ^2 is 2.8-2.9x worse than CSS-X d = 7, 8.7-9.5x worse than XZZX d = 7, and 36-37x worse than CSS-Z |0>.
- The reuse variant's mid-round measure-and-reset step has crosstalk but no memory dephasing or data idling, which slightly favours XYZ^2.

H2 (56 qubits), with each patch that fits: XYZ^2 d = 3 (35 qubits) is 4.7-4.8x worse than CSS-X d = 5 (49 qubits), about 10x worse than XZZX d = 5, and 26x worse than CSS-Z.

The manuscript's own use case, a d = 5 preparation six times below SPAM on Helios, is not specific to XYZ^2. A surface code that fits does better.

Inference on ion timing. The ion models charge memory per gate layer: 7 memory steps per round for XYZ^2 against 5 for CSS and XZZX, at the same two-qubit gate count (170 vs 168). A gate-count timing model could lower XYZ^2's error by up to about 2-2.7x at d = 5. So the 2.9x margin against CSS-X is not robust; the about 9x margin against XZZX probably is.

**Evidence.** login13 scratchpad:
- bq/si.log (SI1000 r = 1, p = 1e-3):
  - XYZ^2 corr_gauge d = 3 / 5 / 7: 1.139e-3, 1.184e-4 (1242 errors), 1.125e-5; BM d = 5 1.009e-4 (38).
  - CSS corr D = 4 1.715-1.794e-3; D = 5 3.09-3.26e-4; D = 6 (71 qubits) 1.97-2.02e-4; D = 7 4.40-4.73e-5; D = 8 (127) 2.38-2.44e-5; D = 9 6.47e-6.
- ovc/resr1.jsonl: XYZ^2 1.170e-3 (313,454 errors) vs CSS D = 4 Z 1.719e-3 (556,163), X 1.811e-3.
- rev5/sd6_*.txt (SD6 r = 1): XYZ^2 corr_gauge 2.082e-5 (1048) vs CSS X 2.563e-6 (258), Z 2.167e-6, XZZX X 2.720e-6 (251).
- bq/ion.log:
  - helios_p xyz2 d = 5 nq = 99: mwpm 8.742e-5, corr_gauge 8.331e-5. CSS d = 7 X 2.980e-5, Z 2.344e-6. XZZX d = 7 9.16-9.57e-6.
  - h2_p xyz2 d = 3 nq = 35: 9.31-9.56e-4. CSS d = 5 nq = 49: X 1.983e-4, Z 3.606e-5. XZZX d = 5 9.5e-5.
Repository:
- paper/manuscript/Hardware.tex lines 20, 72 ('seven memory steps') and 101.
- src/mdr/ft/ft_mdr_circuit.py lines 510-522 (reuse mid-layer).
- src/mdr/ft/circuit_noise.py lines 119-120.
The original pilot scripts (pl.py, si.py, dev.py, hwreal_ion.py) are absent.

### C12: Trapped-ion one-parameter models without crosstalk (helios_p_noxt, h2_p_noxt), r = d and r = 1

**Verdict:** not_useful (confidence 0.88, round-3 agreement 10/10).

Measured. XYZ^2 reaches 0.76 of the best surface code on both machines (0.757 and 0.748 under symmetric rules).
- Helios: it ties CSS (0.1 sigma), because Z dephasing weakens CSS's X basis (CSS-X 0.403 against CSS-Z 0.782).
- H2: nominally below CSS (2.3 sigma with the table fit, 4.0 sigma with BM).
- r = 1: 0.64-0.71 against weaker bases, and 0.40-0.58 against CSS-Z |0> (C10).

Memory dephasing matters most for XYZ^2. At the Helios device point, removing it lowers XYZ^2's preparation error 8.7x (d = 3) and 9.8x (d = 5), far more than removing gates, SPAM or crosstalk. Memory alone produces only 27% (d = 3) and 6% (d = 5) of the full error, so the effect is an interaction of memory with the other channels. No competitor budget exists.

The device points (p = 1.0e-3 and 1.875e-3) are 0.25 and 0.32 of XYZ^2's r = d thresholds and 0.065 and 0.084 of its r = 1 thresholds. The fixed-size comparison (C11) decides at those points.

These models charge memory dephasing per gate layer. An ion model with memory error concentrated in measurement and cooling windows would resemble the hw_tr corner on a natively all-to-all platform. That model is untested; it is the most plausible device route to C35 (inference).

**Evidence.** thresholds.csv:
- helios_p_noxt r = d: both-X bp_corr 0.408(35), bm 0.407(3). Competitors: xzzx-Z corr 0.537(6), xzzx-X bm 0.541(7), xy weaker 0.532, css-X 0.403(5), css-Z 0.782(16).
- h2_p_noxt: bp_corr 0.581(25), bm 0.574(14); xy-X bm 0.766(10); xzzx 0.754(8); css-X 0.641(9).
competitors_table.csv r = 1: 0.637, 0.706.
docs/data/helios/helios_budget.csv, r = 1:
- d = 3: full 8.26e-4, no_memory 9.49e-5, only_memory 2.23e-4.
- d = 5: full 9.09e-5, no_memory 9.25e-6, only_memory 5.75e-6.
src/mdr/ft/circuit_noise.py lines 225-245.

### C13: Trapped-ion models with mid-circuit-measurement crosstalk (helios_p, h2_p)

**Verdict:** not_useful (confidence 0.85, round-3 agreement 10/10).

Measured. No code has a threshold under these models. The listed thresholds are fit artifacts, on both sides; they include XYZ^2's 0.052% (seq_match, nu 0.30, chi2 490).
- The helios_p r = d XYZ^2 entry is a nominal loss (ratio 0.158).
- The only nominal crosstalk-model 'win' is h2_p at r = 5, a Tesseract fit with nu pinned (C27).

Matched distance-pair crossings put XYZ^2 at 0.47-0.65 of the surface codes. The range depends slightly on whether competitor Tesseract pair crossings count. Pair crossings are diagnostics under rule 3.1 and are underestimated, so read them as +-15%.

At equal d, XYZ^2 measures twice the ancillas and receives twice the crosstalk; per-ancilla scaling matches the vendor emulator. At fixed device size the crosstalk is equal, and XYZ^2 still loses (C11).

**Evidence.** thresholds.csv, fss fits with nu pinned at 0.30 or 6.0: helios_p 87-88 of 109, h2_p 58-60 of 160, purez 12 of 565.
crossings.csv, (9,11) pairs:
- Helios: 0.075(4) vs 0.140-0.158.
- H2: 0.340-0.367 vs XZZX 0.559 and CSS 0.524.
- Helios r = 1 (7,9): 0.267 vs 0.497-0.523.
competitors_table.csv: helios_p r = d ratio 0.158; h2_p r = 5 ratio 1.413 (tesseract).
scripts/campaign.py line 108. docs/fair_comparison.md line 25. Hardware.tex lines 51, 70, 72.

### C14: Native pair-measurement hardware (em3), threshold at all r

**Verdict:** not_useful (confidence 0.85, round-3 agreement 10/10).

Measured. XYZ^2 beats this repo's cat-state compilations of CSS, XZZX and XY by 1.29-1.36x at every r:
- 14.3 sigma vs XZZX and 5.8 sigma vs XY at r = d;
- 1.13x with MWPM;
- 1.27-1.30x under matched decoders and the both schedule.

It is not a use case:
1. Measured. On the same hardware the honeycomb Floquet code is 6.1x higher at r = d, and 3.1x (weaker basis) to 4.75x (best basis) higher at r = 1.
2. The cat-state circuits are not the best pair-measurement surface codes. The published 3aux code reaches 0.66% under the noise model of Chao et al., and 0.61% or 0.43% (torus) for its hook-preventing variant. That model is harsher than em3: the honeycomb is 1.3% there against 1.98% here. Relative to the honeycomb, 3aux reaches 0.33-0.51 and XYZ^2 0.165. This is a cross-model inference; 3aux was not re-simulated.
3. At d = 5, XYZ^2's em3 circuit uses 290 physical qubits (355 with flip auxiliaries). CSS uses 185 (209). 3aux uses 4d^2-4d+1 = 81, so XYZ^2 needs 3.6-4.4x.
4. No native QND pair-measurement device exists. Spin parity readout is destructive.

Doc fixes needed in noncss_comparison.md:
- 'beats every surface code' (lines 70, 82, 130) should say 'cat-state compilations';
- line 130's 'beats the honeycomb on all gate-based hardware' is contradicted by the hw_g tie (C16).

**Evidence.** thresholds.csv em3:
- r = d: XYZ^2 0.325(5) vs xzzx-X 0.248(2), xy 0.244(13), css 0.243(5), honeycomb 1.976(21).
- Matched decoders and both schedule: em3@both-X corr_gauge 0.316(5).
- r = 1: XYZ^2 1.118 (table), depth6-X 1.066(41); honeycomb-Z 3.497, honeycomb-X 5.068(338).
competitors_table.csv cat-state ratios at r = 1/2/3/5/10/d: 1.36, 1.31, 1.36, 1.35, 1.29, 1.31.
src/mdr/ft/competitor_circuits.py lines 39-40 and 345-357.
papers/2310.12981.txt: lines 13, 1761 (0.66%), 1047-1049 (N = 4d^2-4d+1), 2370-2393 (0.61 / 0.43), Table 1 (honeycomb 1.3%).
2306.17786.txt lines 1423-1425 (destructive parity measurement).
docs/noncss_comparison.md lines 70, 82, 130.

### C15: Hybrid hardware: gates plus native pair measurements at the same rate (hyb, hyb_lr2); one reading of the resource count in 2112.06036

**Verdict:** not_useful (confidence 0.85, round-3 agreement 8/10).

Measured. 2112.06036 is a code-capacity study and proposes no hardware. It counts only plaquette ancillas (d^2-1) and says the weight-2 links 'may be possible to measure without ancilla qubits', citing joint-parity measurements. hyb is one reading of that resource count; Kerr-cat concatenation is the other route it names (C5).

XYZ^2 loses under hyb:
- r = d: 0.587%, against 0.813% for XZZX with gates and 1.976% for the honeycomb with pair measurements.
- r = 1-10: hyb loses to the gate surface codes at |z| 4.9-10.2.
- hyb_lr2 at r = 1 ties the gate surface codes only with TN ML (2.30 vs 2.42%, 0.6 sigma). Under symmetric rules it is a 3.1 sigma loss (0.855).
- hyb_lr2 at r = 2 ties CSS (z -2.0).
- The honeycomb (3.50% at r = 1) dominates.

Schedule search (outside the d >= 11 rule; crossings at d = 5/7 and 7/9):
- a link-step placement crosses at 0.51% (MWPM);
- split hexagons;
- pipelined split hexagons, which lower p_L about 3x at fixed p and cross at 0.63-0.68% (bp_full, d = 5/7).
XZZX corr crosses at 0.80-0.83% at the same distances.

The 'XZZX pipelined at about 1.0%' reference is the sd6_noidle proxy, not a built circuit. It removes all idling, including reset and readout idling, so it is an upper bound. A like-for-like pipelined ratio is therefore 0.63-0.85, still a loss.

Qubits: XYZ^2 uses 3d^2-1 physical qubits against 2d^2-1 (the Stim circuit declares 4d^2-1 with flip auxiliaries). Pipelined: 130/266/450 at d = 5/7/9. No equal-qubit data exist under hyb, and no current platform offers both gates and pair measurements.

**Evidence.** thresholds.csv:
- hyb r = d 0.587(9); hyb_lr2 0.619.
- hyb_lr2@depth6-X r = 1 tnml 2.299(206); @both-Y bm 2.067(103). Gate xzzx-Z corr 2.417(45). Honeycomb 3.497.
- hyb_lr2 r = 2: 1.507(109) vs css-X bm 1.749(54).
gap/fractions.csv: hyb 0.272(28).
git show origin/cloud/schedule-search:cloud/schedule_search/results.md:
- line 26 (*_noidle removes reset and readout idling);
- section 3 table (s1525_5_R 0.51; y20/y12 bp_full 0.63/0.68; XZZX corr 0.80/0.83; 'XZZX pipelined (data never idle, sd6_noidle)' 1.00-1.07);
- section 5.
2112.06036.txt lines 176-178 and 748-751 (ancilla count) and 1318-1323 (Kerr-cat).

### C16: Gate-based hardware, XYZ^2 vs the honeycomb Floquet code only

**Verdict:** not_useful (confidence 0.85, round-3 agreement 9/10).

Measured. This is a relative win over the honeycomb only.
- Gate models without crosstalk, all r: XYZ^2's threshold is 1.60x (h2_p_noxt, r = d) to 4.46x (si1000_lr2, r = 10) the honeycomb's.
- Spin corners: hw_g is a tie (1.02x, 0.4 sigma); hw_g1q a 1.08x lead (3.5 sigma).
- Large margins come from idling and readout: hw_t 2.6x, hw_rr 6.2x. hw_tr is about 8x, but that is an MWPM pair-crossing pilot ratio, not a threshold.
- Our SD6 honeycomb (0.218%) is about 0.6 of the best published ancilla-free circuits (0.35-0.357%). Against those the SD6 margin is about 1.4-1.5x.

It is not a use case, because the honeycomb is never the best code on gate hardware.

The draft's reason (honeycomb hardware cannot host XYZ^2 without SWAPs) is withdrawn. Wootton (2109.13308) measures a related hexagonal matching code (XYZXYZ plaquettes, ZZ links) on heavy-hex hardware with 2-body measurements only. The measured reason is that the repo's pair-measurement analogue (em3) loses to the honeycomb by 6x (C14).

Hetenyi-Wootton's Floquet results use toric boundaries with no fault-tolerant initialisation or readout, so their XYZ^2/honeycomb ratios are not like-for-like either.

An equal-qubit comparison is untested. Their spin layouts use 4d^2 qubits for XYZ^2 against 6d^2 for the Floquet codes.

**Evidence.** competitors_table.csv, XYZ^2/honeycomb: minimum 0.581/0.363 = 1.60, maximum 4.46. Crosstalk models 1.23-1.79 (not thresholds).
hw/summary.csv: hw_g 1.097(37) vs 1.081(8); hw_g1q 0.402(7) vs 0.372(5); hw_t 1.330 vs 0.507; hw_rr 10.63 vs 1.72.
hw/diag_hw_tr_centers.csv: 10.49 vs 1.24-1.36 (MWPM pairs).
2109.13308.txt lines 1-14, 40-110.
2306.17786.txt: lines 1072-1080, 1190-1197 (toric, no fault-tolerant init or readout), 1263-1268, 1826, 1885.
lit_color_floquet_xyz2.md lines 40-41 and 74-76.

### C17: Phenomenological noise with bias eta = 10 (phen_b10, model of arXiv:2505.03691, not hardware), threshold at all r

**Verdict:** competitive (confidence 0.82, round-3 agreement 10/10).

Measured.

r = d: XYZ^2 ties XZZX (5.539(54) vs 5.537(55)) and XY (1.8 sigma lead, a tie).
- Under symmetric rules it is 5.450(40), 0.984 of XZZX (1.3 sigma).
- The two XYZ^2 schedules are statistical replicates in this model (_build_phen ignores the schedule). Their mean is 1.00 of XZZX.
- XYZ^2's X-basis fits span 5.40-5.54 against XZZX-Z 5.48-5.54.
- The XZZX-Z crossings drift upward, so the tie rests on d = 19/21.
- The tie needs BP-based decoding: MWPM reaches 0.57 of XZZX and corr_gauge 0.78.

Other round counts:
- r = 1-3: XZZX is higher (z >= 3.4).
- r = 5: probably below. 2.6-2.7 sigma with the table fit; 3.5-4.8 sigma with the depth6-X BM fit against XZZX-Z BM or corr.
- r = 10: unresolved. XZZX's X basis has only drift-flagged pair crossings; symmetric XYZ^2 vs XY is 0.6 sigma.
- XYZ^2 beats XY only at r = 1 (6 sigma), and beats CSS (1.78x at r = d).

The best published XYZ^2 decoder for this model (BM, 4.30%, 2505.03691) reaches 0.78 of our XZZX.

Cross-model inference: Hetenyi-Wootton's Table I with their rule gives a phenomenological limit (p_T = p_RR) of XYZ^2 3.35% vs XZZX 6.96% (ratio 0.48), with ZZ links at eta_T = 20. So the phen_b10 tie depends on the XX-link orientation.

Below threshold at equal qubits: see C31.

**Evidence.** thresholds.csv phen_b10:
- r = d: both-X bp_full 5.539(54), bm 5.450(40), mwpm 3.135(39), corr_gauge 4.336(69); depth6-X bp_corr 5.500(71). xzzx-Z corr 5.537(55) (crossings 4.32 to 5.53, drift 1.28); xzzx-X corr 5.715(132); xy-Z bm 5.371(77); css-X bm 3.105(52).
- r = 1: 15.465(797) vs xzzx-Z 25.77(82), XY 10.49(23).
- r = 3: 8.508(641) vs 12.477(959).
- r = 5: both-X bp_full 7.525(630), depth6-X bm 7.525(321); xzzx-Z bm 9.617(496), corr 9.320(187).
- r = 10: depth6-X bp_corr 6.844(332); both-X bm 6.484(180); xzzx-Z corr 7.656(143); XY 6.245(379).
src/mdr/ft/ft_mdr_circuit.py lines 551-590.
2505.03691.txt lines 26, 872, 916.
2306.17786.txt lines 570-575 and Table I (1/(1/4.2+1/16.5) = 3.35; 1/(1/15.1+1/12.9) = 6.96).

### C18: Phenomenological depolarizing noise (phen, eta = 0.5; not hardware), threshold at all r and equal qubits

**Verdict:** not_useful (confidence 0.82, round-3 agreement 10/10).

Measured.

Threshold:
- r = d: 0.92 of XZZX (4.1 sigma), a loss. Against the table's nominal best, CSS, it is 0.91 at 2.8 sigma, a tie under the rule (7.2 sigma with stat-only errors).
- r = 1, 2, 3 and 10: ties (|z| <= 1.4). r = 5: 1.9 sigma, a tie.
- The r = 1 'win' (1.08, 1.05 sigma) is the maximum over replicate schedules: CFE gives 16.95 (both-X) and 15.92 (depth6-X) on identical circuits. It disappears with BM on both sides (15.74 vs 15.72). Symmetric rules give 1.009 (0.3 sigma).

Equal qubits (pilots, p = 2.5%, r = 3d): XYZ^2 is 1.36x, 1.83x and 2.36x worse at 18/16, 50/49 and 98/100 data qubits (35/31, 99/97 and 195/199 total-equivalent). The XZZX partners at 16 and 100 data qubits have even distance (d = 4 and 10), which handicaps XZZX, so these losses are conservative. The gap grows with size.

This is consistent with 2112.06036's low-p statement (failure rate tracks distance). With measurement errors XYZ^2 already loses threshold, as that paper anticipated.

**Evidence.** thresholds.csv phen:
- r = d: depth6-X bp_corr 4.156(57) vs xzzx-X bm 4.510(66), css-Z bm 4.570(135).
- r = 1: both-X cfe 16.955(1116), depth6-X cfe 15.915(525), mwpm 15.854(281), bm 15.737(43). xzzx-Z bm 15.719(38).
- r = 5: 6.456(135) vs css-X 6.963(237).
Pilots (login13 scratchpad):
- bq/phen.log: XYZ^2 d = 3 Y 8.616e-3 vs XZZX d = 4 corr Z 6.323e-3; d = 5 Y 2.555e-3 vs d = 7 X 1.400e-3; d = 7 Y 8.00e-4 and 8.74e-4 (114 and 129 errors) vs d = 10 X 3.55e-4.
- stat/runph_*.
2112.06036.txt lines 172-175 and 1063-1083.

### C19: Hetenyi-Wootton spin-qubit corners: gates (hw_g, hw_g1q) and eta_T = 20 idling in every gate layer as well as readout (hw_t); r = d, d = 11-

**Verdict:** not_useful (confidence 0.85, round-3 agreement 8/10).

Measured. These are the fairest comparisons in the repository: both schedule only, and matched decoders.
- hw_g: XYZ^2 ties the honeycomb for third. It is 0.78 of CSS (4.9 sigma) and 0.79 of XZZX (7.4 sigma).
- hw_g1q: XYZ^2 is third.
- hw_t: XYZ^2 reaches 0.35 of CSS and 0.13 of XZZX. That value is a corr_gauge Y fit with chi2 4.77, whose error is not inflated for misfit (read about +-0.014).

hw_t charges idling in every gate layer, which is harsher than Hetenyi-Wootton's placement. hw_tr puts idling where they do, during readout and reset (C21, C35).

Their ordering XYZ^2 > XZZX at the gate corner (0.465 vs 0.37) is not reproduced.
- hw_g1q cannot test it: it puts the same single-qubit error on every code, so CSS and XZZX keep identical detector error models.
- Their XZZX carries extra data-qubit basis changes. They state this themselves (not our inference): XZZX's 'local basis transformations necessitate extra single-qubit gates'.

Absolute corner values differ from theirs by factors of 0.32-3.1 across codes and corners (3.75 with best decoders). The 1.4-1.9x applies only to the MWPM gate corner for CSS and XYZ^2.

The surface codes' hw_g crossings drift up more than XYZ^2's, so the hw_g gap may be understated.

**Evidence.** docs/data/campaign/hw/summary.csv:
- hw_g: 1.097(37) / CSS 1.400(49) / XZZX 1.389(14) / honeycomb 1.081(8). Theirs: 0.465 / 0.82 / 0.37 / 0.43.
- hw_g1q: 0.402(7) / 0.519(13) / 0.514(10) / 0.372(5).
- hw_t: 1.330(6) / 3.82(40) / 10.0(17) / 0.507(11).
- XZZX hw_g MWPM 1.151 vs their 0.37.
hw/thresholds.csv: CSS X bm drift 1.77; XYZ^2 1.26.
2306.17786.txt: Table I (lines 400-456), lines 1011-1016 (XZZX single-qubit gates), 1076-1080 (ZZ links).
docs/hetenyi_wootton_check.md sections 2-6 and lines 95-98, 121.

### C20: Spin-qubit readout-only corner (hw_rr: outcome flips only): threshold at r = d, and an equal-qubit probe at p = 4% (0.38 of threshold), r = 

**Verdict:** not_useful (confidence 0.8, round-3 agreement 7/10).

Threshold, measured: a tie.
- XYZ^2's 10.63(46)% comes from its X basis, against CSS 10.50(56) and XZZX 10.32(44).
- The only fss fit in XYZ^2's Y basis (MWPM, 10.85(22)) lies above its X basis (10.60(28)). The lower Y pair crossings (10.27, 9.95) are drift-flagged diagnostics.
- So XYZ^2 is nominally highest with every decoder class, but within errors. MWPM: 8% higher at 1.65 and 1.96 sigma.

Below threshold at equal qubits (a new single-point probe with the repo's circuits): the tie does not become an advantage.
- XYZ^2's weaker basis is 2.79(10)x worse than XZZX with BM against corr.
- With MWPM on both sides it is 2.95x worse.
- CSS gives the same as XZZX (MWPM 3.61-3.66e-4).

Withdrawn from the draft:
- the break-even readout-to-gate ratios (68, 169), which no file backs;
- 'real devices sit at a ratio of about 1'. Both errors below 1% does not fix their ratio; the repo's calibrated models put readout to gate at about 0.25-5.

hw_rr omits Hetenyi-Wootton's two-qubit error on the readout pair and their reset error, so their exact corner is untested. Their own readout-corner lead (16.5 vs 14.5) is inside their 8-12% plane-fit deviation. Qubit cost in their layout: 1.33x.

**Evidence.** hw/summary.csv hw_rr: XYZ^2 10.634(460) (X corr_gauge, 'no Y crossing'); CSS 10.504(563); XZZX 10.319(437); honeycomb 1.72(3). MWPM 10.60(28) vs 9.78(41) and 9.85(26).
hw/thresholds.csv: XYZ^2 Y mwpm fss 10.853(219); Y corr_gauge pair 10.27(54), drift; Y bm pair 9.95(82).
Probe outputs (login13 scratchpad rv/, script rv/eqprobe.py):
- XYZ^2 d = 5: BM Y 1.016e-3 (887/59,048), BM X 6.69e-4 (574), corr_gauge Y 1.077e-3 (33,044), MWPM Y 1.075e-3 (70,997), MWPM X 6.99e-4.
- XZZX d = 7: corr X 3.638e-4 (14,883); MWPM X 3.647e-4, Z 3.636e-4.
- CSS d = 7: MWPM X 3.657e-4, Z 3.610e-4.
src/mdr/ft/circuit_noise.py: si1000 p_meas = 5p; quantinuum helios spam 5e-4 vs r2 8e-4; h2 1.5e-3 vs 1.5e-3.
2306.17786.txt lines 34-36 and Table I (deviation column).

### C21: Spin-qubit idling only during ancilla readout and reset (hw_tr diagnostic): threshold ranking from MWPM pair crossings, d <= 9; and Hetenyi-

**Verdict:** not_useful (confidence 0.75, round-3 agreement 6/10).

Measured (pilot pair crossings, which rule 3.1 treats as diagnostics). By threshold XYZ^2 is second to XZZX:
- XZZX Z: 18.95(160)%.
- XYZ^2 X: 10.49(55)%.
- CSS X: 6.64(21)%.
- Honeycomb: 1.24-1.36% (a single 4/8 pair).
The d-trends are not significant: XYZ^2 Y rises +36% at 2.1 sigma, and the XYZ^2/CSS ratio falls 1.76 to 1.58 at 1.1 sigma.

Correction to the repository. diag_hw_tr_centers.csv reports no crossing below 44% for XZZX-X, but its own points (points_diag.csv) show the XZZX-X curves for d = 5/7/9 meeting at p of about 0.19-0.30. That entry is a fit failure. XZZX-X is in fact XZZX's weaker basis below threshold (C35). Only CSS-Z is genuinely immune: d = 9 is lowest at every p up to 44%. This also needs fixing in hetenyi_wootton_check.md lines 68 and 112.

The XYZ^2 MWPM crossing (10.5%) is far below the published code-capacity thresholds at eta = 10: 24.1% (BM) and 28.6% (ML) for XYZ^2, and about 28% for XZZX. hw_tr has perfect syndromes, so the threshold ranking is decoder-limited.

Hetenyi-Wootton's plane (ZZ links), recomputed independently by three reviewers (no saved script): XYZ^2 leads both CSS and XZZX in 0.19-0.21% of the uniform simplex, by at most 1.13-1.14x, at the readout corner.

Their hypothetical scenario (pG ~ 0.1%, a few percent idling during slow readout) names XZZX as the usable code. Their XYZ^2 differs from ours in two confounded ways:
- ZZ links;
- links read in a separate measurement round, so each data qubit idles through two readout windows per cycle.
Which factor drives their 4.2% XYZ^2 idling threshold is not separated (inference).

Threshold is not the deciding metric here. Below threshold see C35 and C37.

**Evidence.** hw/diag_hw_tr_centers.csv, crossings 5/7 then 7/9:
- XYZ^2 X 9.58, 10.49(55); XYZ^2 Y 10.27, 13.99(166).
- CSS X 5.45, 6.64(21).
- XZZX Z 14.26, 18.95(160).
- Honeycomb 4/8 1.24-1.36.
hw/points_diag.csv, XZZX-X MWPM at r = d:
- p = 0.1903: d = 5 0.2205 (301/1365), d = 7 0.2108 (1151/5461), d = 9 0.1990 (1087/5461).
- p = 0.30: 0.408, 0.407, 0.415 (1365 shots each).
- p = 0.313: 0.429, 0.414, 0.462.
hw/points_diag.csv, CSS-Z: d = 9 lowest at every p, e.g. 33/20,000 at 0.44 vs 161 for d = 5.
2505.03691.txt lines 22-25. lit_surface_family.md line 125.
2306.17786.txt lines 1083-1104, 1222-1243, 1323-1333, 1885-1887, 2042, Table I.

### C22: Gap diagnostic (SD6 variants, not hardware): noiseless links and/or no idle errors; links repeated; links as pair measurements; schedule sea

**Verdict:** not_useful (confidence 0.88, round-3 agreement 9/10).

Measured. XYZ^2 reaches XZZX's threshold only when the link checks are noiseless and idle errors are zero; then it ties (1.01(3)).

Each change alone leaves it at 0.77:
- noiseless links, against XZZX under SD6: 0.770(21);
- no idling, against XZZX without idling: 0.774(8).

The effects are super-additive (3.4 sigma). As fractions of the gap:
- idling 0.60(1);
- link noise 0.23(2) against XZZX without idling, or 0.43(4) against XZZX under SD6;
- both together 1.02(5).
XYZ^2 without idling against XZZX with idling gives 1.050(27). That is a cross-model comparison, not a use case.

Physical substitutes recover little: hybrid 0.27(3), links twice 0.22(7), links three times 0.12(3).

Schedule search:
- the best of 1690 full-distance depth-6 schedules is within statistics of the default;
- removing gate-layer idling alone raises XYZ^2's MWPM crossing from 0.38% to 0.57% (sd6, a d = 7/9 diagnostic crossing).
The repo's like-for-like idle-free comparison is 0.774(8). The 'XZZX without idling crosses at about 1.0%' reference is the sd6_noidle proxy, not a built pipelined circuit.

sd6_noidle is the repo's only idle-free gate-noise model, similar in spirit to a neutral-atom error budget. XYZ^2 still loses there.

Open: a depth-4 pipelined sd6 circuit with gate links, and y20 at d >= 11.

**Evidence.** docs/data/campaign/gap/gap.csv:
- XYZ^2: 0.487(7), sd6_il 0.626(8), sd6_noidle 0.854(6), sd6_il_noidle 1.117(31).
- XZZX: 0.813(20) and 1.103(9).
gap/fractions.csv:
- 0.427(37) / 0.770(21); vs XZZX noidle 0.226(15).
- 0.596(14) / 0.774(8).
- 1.022(53) / 1.013(30).
- hyb 0.272(28); lr2 0.218(68); lr3 0.122(32); sd6_noidle vs XZZX sd6 1.050(27).
gap_diagnostic.md lines 59-60.
schedule-search results.md: line 26, section 3 table, section 5.

### C23: Decoder dependence: MWPM, HCM, BM/BP, sequential two-level, CFE, Tesseract, TN ML; all models, including the pilot regimes of C31 and C35

**Verdict:** not_useful (confidence 0.85, round-3 agreement 10/10).

Measured. No decoder flips any verdict against the best competitor on a hardware model.

Decoder gains at r = d:
- XYZ^2's best/MWPM is 1.16-1.90, against 1.01-1.17 for XZZX. XY gains as much under bias (1.71-1.82).
- So decoders that fit a real-time budget make XYZ^2 look worse.
- For r = 1 preparation, decoding need not be real-time. There TN ML or CFE-TN gains only 1.08x over MWPM in the MDR basis, and XYZ^2 still loses.

XYZ^2's best decoder against every competitor on MWPM:
- loses under SD6 (0.73), bias (0.42-0.68) and ion noise;
- ties under SI1000 (0.98) and phen_b10 (1.01);
- leads under phen (1.09; non-hardware).

Asymmetry in decoder coverage:
- fair_comparison.md section 3.2 promises the surface codes BP-OSD and Tesseract. Those competitor series exist only as pair crossings up to d = 11 (427 and 429 pair rows, 0 fss), so the fss rule removes them.
- XYZ^2 has 86 Tesseract fss fits up to d = 21, plus bp_corr, bp_full, seq_*, CFE and TN ML.
- XYZ^2's head-to-head entry uses an XYZ^2-only decoder in 65 of 107 rows.

At equal qubits decoders matter more:
- against CSS under bias: 3.5-58x better (corr) to 1.7-9.3x worse (MWPM);
- against XZZX: 10-850x worse (corr), and 61-72,000x worse (MWPM);
- BM narrows the XZZX gap to 3.4-6.9x at 99/97 qubits, but not at 323/337 (28-51x);
- CFE gains only 11-28% over BM.

The only regimes where XYZ^2 leads (C31, C35) rest entirely on BM/BP for XYZ^2.
- In phen_b10 at p = 2% and 50/49 data qubits, XYZ^2 with corr_gauge is 1.6x worse than XZZX corr, and with MWPM 9.6x worse.
- No matching-class hw_tr equal-qubit run exists.
- BM costs about 0.2-0.5 s per shot at d = 11-15, against 1e-4 to 6e-4 s for corr (eq/timing.csv, as reported by a reviewer).

**Evidence.** competitors_table.csv r = d:
- XYZ^2 best/MWPM: sd6_il 1.16, hyb 1.25, sd6 1.28, purez 1.90.
- XZZX 1.01-1.17. XY under bias 1.71-1.77; phen_b10 1.82.
- biased10: 0.748/1.102 = 0.68.
thresholds.csv: competitor rows with decoder bposd have 427 pair and 0 fss; tesseract 429 pair and 0 fss (dmax 5-11). XYZ^2 tesseract fss 86.
competitors_table.csv xyz2_dec counts: bp_corr 31, bp_full 21, bm 18, tnml 12, seq_soft 9, seq_match 3, seq_erasure 3, cfe_tn 2, corr_gauge 2, cfe0 2, cfe 2, corr_links 1, tesseract 1.
sd6 r = 1: tnml 1.960(190), cfe_tn 1.960(73), mwpm 1.809(19).
Pilots (login13 scratchpad stat/): run_0.02_xyz2_X_corr_gauge_m 3.03e-4 (4495 errors), run_0.02_xyz2_X_mwpm_m 1.80e-3 (81,809) vs XZZX corr X 1.875e-4.
pairs.csv; points_eps.csv. Results.tex line 239. docs/fair_comparison.md line 28. docs/ft_mdr_decoders.md line 43.

### C24: Fairness and selection effects of the comparison rules

**Verdict:** not_useful (confidence 0.85, round-3 agreement 9/10).

Measured. Overall the rules favour XYZ^2, and it still loses.

Favouring XYZ^2:
- Its entry is the maximum over about 19-27 usable schedule-decoder candidates of its weaker basis, with no nu or chi2 filter. The competitor maximum is over about 12 usable fits (24 under hyb).
- An XYZ^2-only decoder is used in 65 of 107 head-to-head rows. The competitors' promised BP-OSD and Tesseract never reach fss (C23). This is the largest asymmetry.
- XYZ^2 may use depth6 L_Y, whose fault distance is (d+1)/2. The hyb rule refuses that to the surface codes.
- Its BOTH_BASES_SCHEDULE was selected on SD6 MWPM error and is then compared under SD6: mild in-sample selection.
- Native degree-6 connectivity is assumed with no routing cost (C36).
- In phen models its two schedules are replicates.

Favouring the competitors:
- In isomorphic models the table takes the maximum of 3 replicates (CSS/XZZX/XY). This is at most about 2%; for si1000 at r = 1, the maximum 1.078(58) against the pool 1.058(20).
- Under per-ancilla crosstalk XYZ^2 gets twice the crosstalk at equal d.
- Against XYZ^2 in equal-qubit runs: the campaign pairs use corr for XYZ^2 although its BM is 2-6x better under bias. The ratios cited elsewhere give BM separately.

Not asymmetries:
- The SD6 idle convention is rule 1 ('a code whose rounds take more steps pays for them'). It is the largest real cost, about 60% of the gap.
- The competitor circuits being harsher than Stim's reference (1.7-2.5x p_L) matters only for comparisons with the literature; XYZ^2 has the same idle steps.

Symmetric rules (both schedule; mwpm/corr_gauge/bm vs mwpm/corr/bm; one reviewer's recomputation):
- r = d: sd6 0.60, si1000 0.83, biased10/100/purez 0.62/0.49/0.41, helios/h2_noxt 0.76/0.75, phen 0.89, phen_b10 0.98, em3 1.27-1.30x the cat-state codes.
- r = 1: si1000 0.89, phen 1.01, hyb_lr2 0.855 (z -3.1).
- XYZ^2 drops 0-4.9% at r = d (sd6 4.9%, em3 2.8%) and up to 10-11% at r = 1, 3 and 10.

Only with gate errors absent (hw_rr) or links noiseless and no idling (sd6_il_noidle) does XYZ^2 reach parity in threshold. The one setting where it leads below threshold (hw_tr, C35) has noiseless gates, readout and reset. That breaks rule 1 ('nothing is noiseless, except in the diagnostic'), so by the repo's own rules it is a diagnostic, not a hardware claim.

**Evidence.** paper/analysis/competitor_report.py lines 68-127. scripts/run_decoder_threshold_sweep.py lines 69-100. extraction_schedule.py lines 163-167. competitor_circuits.py lines 36-45.
docs/fair_comparison.md sections 1, 3.2, 3.4 (lines 8, 13, 28, 31).
thresholds.csv:
- si1000 r = 1: 1.059 to 0.959(16).
- phen r = 1: 16.96 to 15.85(28).
- em3 r = d: 0.325 to 0.316 (corr_gauge), 0.304 (BM only).
- hyb_lr2@both-Y r = 1: bm 2.067(103).
git show origin/cloud/fairness-audit:cloud/fairness_audit/README.md section 2a (lines 67, 73-74). Schedule-search results.md section 1.
Symmetric table: fairrev/sym.py (login13 scratchpad).

### C25: Hardware cost at equal distance, and qubit overhead by layout

**Verdict:** not_useful (confidence 0.85, round-3 agreement 9/10).

Measured from the repo's circuits (one ancilla per check). XYZ^2 needs more hardware on every axis:
- about twice the qubits;
- 2.08-2.25x the two-qubit gates per round;
- 8 ticks per round against 6;
- coupler degree 6 against 4;
- three gate types against two.
At equal qubits (99/97) the two-qubit gate count is the same (170 vs 168), but XYZ^2 has 2.4x the gate-layer idle sites per round (354 vs 150). Any gate-layer idling therefore charges it more (C37).

The XZZX and CSS d = 3 coupling graphs are not subgraphs of XYZ^2's d = 3 or d = 5 graphs, so each code needs its own coupler layout. The 'same hardware' premise is exact for all-to-all or reconfigurable platforms (ions, neutral atoms). For triangular lattices it is unverified for this repo's 4d^2-1 circuit; 2112.06036's triangular proposal has no link ancillas (3d^2-1, i.e. hyb).

The overhead depends on the layout:
- 2x with one ancilla per check;
- 1.5x hybrid;
- about 1.6x under em3;
- 1.83x pipelined (266 vs 145 at d = 7);
- 1.33x in Hetenyi-Wootton's spin layout (4d^2 vs 3d^2), but at a time cost. There the links reuse the plaquette ancilla pairs and are read in a separate measurement round, so data idle through two readout windows per cycle. A one-window spin layout with a readout pair per check needs about 6d^2 vs 3d^2, i.e. 2x again (inference).
- 0.67x against Floquet codes in their layout (4d^2 vs 6d^2).

A truncated XYZ^2 patch (2112.06036) could lower the count; untested.

**Evidence.** Stim circuit counts by several analyses, d = 3/5/7:
- qubits 35/99/195 vs 17/49/97;
- two-qubit gates per round 54/170/350 vs 24/80/168;
- ticks 8 vs 6;
- maximum degree 6 vs 4;
- gates CX/CY/CZ vs CX/CZ.
Idle sites per round at 99/97: 354 vs 150 (one reviewer's Stim build).
subgraph_is_monomorphic False in all four cases (networkx; reviewer checks).
Hardware.tex line 20.
2306.17786.txt lines 1083-1091, 1103-1104, 1263-1268, 1885-1887, 2042 ('every qubit is subject to two idling phase during measurements').
2112.06036.txt lines 300-302 and 744-754.
schedule-search results.md (pipelined counts).

### C26: Labelling of the campaign's biased noise models, and biased noise with gates that do not preserve bias

**Verdict:** untested (confidence 0.75, round-3 agreement 10/10).

Measured from the code. The biased models put Z-type channels (IZ, ZI, ZZ) after every native CX, CY and CZ, so they preserve bias; measurement and reset flips stay at p. Three files wrongly list XYZ^2's 0.75/0.83/0.86% under 'gates not bias-preserving'.

The device form of bias (Z-dephasing memory with depolarizing gates) was simulated, and the placement decides the outcome:
- dephasing per gate layer (ion models, hw_t): XYZ^2 loses;
- dephasing only during readout and reset with perfect syndromes (hw_tr): XYZ^2 leads in pilots (C35);
- small gate-layer fractions remove that lead (C37).

Untested: bias at every location with a CX compiled through H (2505.17718).

Literature comparison (cross-model): XYZ^2's values lie between XZZX's two non-bias-preserving compilations, above the CZ-only one (0.73-0.80%) and below the HBD one (0.85-0.93%). That model could produce anything from a loss to a tie.

Gate set: CY = (I x S) CX (I x S^dag) is bias-preserving in the cat-qubit set {CX, Z(theta), ZZ(theta)}. XZZX's CZ also needs compiling there. Treating CY and CZ as native favours both codes alike, apart from the omitted noise of the Z(pi/2) rotations.

Source of the bias-preserving XZZX literature values: lit_hardware.md line 112 gives about 1.3 / 2.0 / 2.4% at eta = 10 / 100 / 1000, all read from a figure. noncss_comparison.md line 16 omits eta = 10.

**Evidence.** src/mdr/ft/circuit_noise.py lines 139-156.
Wrong labels: docs/noncss_comparison.md line 15; ftmdr_tools/noncss/lit_hardware.md line 114; lit_surface_family.md line 163.
lit_hardware.md lines 112-114: 2505.17718 HBD 0.85/0.92/0.93 and CZ-only 0.73/0.79/0.80.
papers/1905.00450.txt and 1904.09474.txt line 394 (bias-preserving gate set).
Our XZZX 1.175/1.655/2.061.

### C27: Reported XYZ^2 'wins' or values in the campaign tables that are fit or coverage artifacts

**Verdict:** not_useful (confidence 0.88, round-3 agreement 9/10).

Measured. These entries should not be cited:
- h2_p r = 5 (ratio 1.41): Tesseract with nu pinned at 6.0 and chi2 14.8. The BM alternative is also pinned (nu 0.30).
- helios_p r = d 0.052%: seq_match, nu 0.30, chi2 490 (a nominal loss; C13).
- phen_b10 r = 10 (1.10 vs XY, 'winner xyz2'): unresolved (C17). The both-X bp_corr 7.41 fit there has nu pinned at 6.0.
- biased100 r = 10: XY falls back to MWPM (0.843) because its BM and corr values are drift-flagged. That creates a nominal 3.9 sigma XYZ^2 lead over XY (C5).
- hw_g1q L_Y BM 0.644%: nu 0.31, chi2 9.5; the raw curves cross at about 0.41%.
- sd6 both-Y cfe0 0.615 (nu 0.30, chi2 13.7): chosen as that basis's maximum, hidden only because L_X is weaker.
- XYZ^2 pair-crossing outliers: Tesseract at d = 9/11; purez cfe0 1.371 and tesseract 1.415.
- hw_tr diag_hw_tr_centers.csv 'no crossing' for XZZX-X (C21).

Weaker-basis entries at r = d that use the depth6 schedule's L_Y (fault distance (d+1)/2): biased10, em3, hyb_lr2 and sd6_il; helios_p uses both L_Y. biased100, purez and phen use depth6 L_X. The both-schedule values are 0.001-0.07 percentage points lower; no conclusion changes.

**Evidence.** competitors_table.csv r = d xyz2_dec:
- biased10 'bp_corr, depth6 L_Y'; biased100 'bp_full, depth6 L_X'; purez 'bp_full, depth6 L_X';
- em3 'seq_soft, depth6 L_Y'; hyb_lr2 'corr_gauge, depth6 L_Y'; phen 'bp_corr, depth6 L_X'; sd6_il 'corr_gauge, depth6 L_Y'; helios_p 'seq_match, both L_Y'.
thresholds.csv:
- h2_p r = 5: tesseract 0.8167(170), nu 5.999, chi2 14.77; bm 0.4204, nu 0.300.
- helios_p r = d: seq_match, nu 0.300, chi2 489.96.
- sd6@both-Y cfe0 0.6153, nu 0.300, chi2 13.71.
- phen_b10@both-X r = 10 bp_corr 7.414, nu 6.000.
- biased100 r = 10: xy-X bm pair 1.192 (drift); mwpm fss 0.843(16).
hw/thresholds.csv: hw_g1q xyz2 Y bm 0.644, nu 0.311, chi2 9.54.
competitors_table.csv: winner = xyz2 in exactly four rows.

### C28: Noise that lacks the link Pauli: (a) Z-biased dephasing in the XX-link frame (tested; C35, C37); (b) amplitude-damping idle noise (p_X = p_Y

**Verdict:** untested (confidence 0.35, round-3 agreement 10/10).

Case (a), measured.
- With perfect syndrome extraction (noiseless gates and readout) and eta = 20 data dephasing, XYZ^2 beats square XZZX at equal qubits (C35).
- With 0.1% gates and 1% readout the edge is significant at one size only.
- Gate-layer dephasing of 2% of p_T removes it (C37).
- With gate-level noise at every location (purez, biased100) XYZ^2 loses by 3.4-51x with BM (C6).
So the bulk structure helps only when syndrome extraction is essentially perfect.

Case (b), untested.
- The inner [[2,1,1]] link detects both non-link Paulis.
- A phenomenological probe reported bulk distance 6 at d = 3, against 3-4 for CSS, XZZX and XY. Per sqrt(data qubits) (18 vs 9) that is 1.4x more distance against the distance-3 bases and 1.06x against the distance-4 ones. The script is not on disk.
- Literature source: 2505.03691 proposes -ZZ links (code space |01>, |10>) as a dual-rail encoding tailored to relaxation.

Inference about platforms: T1-limited idling occurs mainly on superconducting qubits, which lack degree-6 couplers. Ions, atoms and spins have dephasing-dominated idling.

**Evidence.** C35 and C37 pilot outputs.
pairs.csv / points_eps.csv for purez and biased100.
2505.03691.txt lines 946-953.
yzprobe.py and its outputs are not on disk.

### C29: Long memories (r >> d) under circuit-level extreme Z bias (purez, biased100), full detector-error-model decoders, equal qubits

**Verdict:** untested (confidence 0.35, round-3 agreement 10/10).

Untested as a long-memory statement, and weakened by measured data.
- Full-DEM data (XYZ^2 BM) lose by 3.4-6.9x at 99/97 qubits and by 28-51x at 323/337 (C6).
- Removing the time boundary does not help. The boundary-free (two-round difference) error at 99/97 favours XZZX by 6.2-9.0x (biased100) and 10.6-14.9x (purez). Removing the boundary changes XYZ^2's error by only 20-46%, so the deficit is bulk-dominated.
- XYZ^2's measured BM Lambda from d = 7 to 9 at 3e-3 is 7.5 (biased100) and 9.3 (purez). It rests on 16 d = 9 errors per model (+-25%), against XZZX corr 7.84 and 9.12. Break-even needs 18.4-22.8.
- CP-SAT and MILP weights are upper bounds with weak lower bounds, and were not saved.
- XY gives d^2 pure-Z protection on half the qubits and has a higher purez threshold.

Contrast: with perfect syndromes (hw_tr) the bulk ratio at r = 6d (0.38) is better than at r = 3d (C35). Circuit-level gate noise removes that. MDR preparation would not benefit (C9).

**Evidence.** docs/data/campaign/eq/points_eps.csv, XYZ^2 BM X at 3e-3: biased100 d = 7 1.353e-5 (98 errors), d = 9 1.812e-6 (16); purez 1.410e-5 (100), 1.505e-6 (16).
boundary.csv eps_two:
- biased100: XYZ^2 2.44e-6 / 5.20e-5 / 2.80e-4 vs XZZX 3.92e-7 / 6.48e-6 / 3.11e-5;
- purez: 2.21e-6 / 3.77e-5 / 2.57e-4 vs 2.08e-7 / 3.22e-6 / 1.72e-5;
- XYZ^2 change -0.20 to -0.46.
lambda.csv: XZZX 7.84(17), 9.12(29).
thresholds.csv purez r = d: XY 1.045 vs XYZ^2 0.857.

### C30: Regimes and competitors not simulated

**Verdict:** untested (confidence 0.5, round-3 agreement 9/10).

Untested; the expected directions are inference.

Correction to the draft: code capacity is not literature-only. hw_tr has no gate, measurement, reset or gate-layer idle noise, so every syndrome is perfect and the bulk is exact repeated code capacity. C35 is therefore measured code capacity with MDR time boundaries. What remains untested there is ML or tensor-network decoding.

Noise:
- Strong bias at p < 1e-3 (C6; the extrapolation suggests no practical crossover).
- X- and Y-biased circuit noise. 2112.06036 already reports XYZ^2 worse than XZZX under X bias. Under Y bias XYZ^2 is unchanged while XZZX improves and scales with data-qubit count, i.e. no equal-qubit XYZ^2 advantage.
- Non-bias-preserving gates (C26).
- Erasure.
- Leakage (about twice the gates and measurements per round).
- Intra-link correlated ZZ dephasing.
- Time-correlated (quasi-static or 1/f) dephasing.
- Amplitude damping with ZZ links (C28).
- Hybrid hardware with unequal gate and pair-measurement rates.

hw_tr-type models (C35, C37):
- gate-layer dephasing fractions between 0 and 0.05 at more sizes;
- readout error at matched p with XZZX comparators;
- p_T < 3%; eta between 20 and 100;
- r = 6d at any setting other than eta 20, p_T 4%;
- reset, single-qubit gate and readout-pair two-qubit errors;
- each code at its own optimal readout time;
- matching-class XYZ^2 decoders, and XYZ^2 bp_corr and CFE (only BM was used).

Hardware:
- Neutral atoms (with ions, the only natively degree-6 class).
- An ion model with memory error in measurement and cooling windows.
- Routing on degree-4 and degree-3 chips, and the heavy-hex 2-body compilation of 2109.13308.
- A realistic spin compilation: SWAPs, ZZ vs XX links, Hetenyi-Wootton's two-readout-round layout, spin-blockade final readout.
- Shuttling-based spin architectures.
- helios_cb runs for the competitors; a per-gate timing model for ions.

Competitors:
- CSS, honeycomb and XY at equal qubits in hw_tr at p_T 3-4% and in the mixture.
- Rectangular, co-prime (e.g. 6x7 = 83 qubits) and d^2-frame XZZX or XY with ML or tensor-network decoding in hw_tr. The literature predicts parity with a best-framed square code at code capacity (2112.06036 Fig. 7c).
- Rectangular single-basis patches for r = 1 preparation (3x6 = 35, 5x10 = 99).
- Rectangular or tailored XZZX and CSS under gate-level bias.
- Clifford-deformed and XYZ colour codes; repetition cat codes.
- 3aux under em3.
- 6.6.6 colour and bivariate-bicycle codes at degree 6.
- Competitor BP-OSD and Tesseract at fss (promised in fair_comparison.md).
- A built pipelined XZZX circuit; the surface codes' direct hyb compilation, measured rather than refused.
- Competitor runs at r = 4, 6-9 and 11-21.

Missing equal-qubit runs: hyb, ions, em3, SI1000 memory, and hw_tr at campaign standard.

Other: the truncated XYZ^2 patch; a depth-4 pipelined sd6 circuit; twist-defect logic without weight-5 checks; logical operations (lattice surgery, injection), which all touch the weak time boundary.

**Evidence.** pairs.csv covers sd6, biased100 and purez only.
thresholds.csv has no X-biased, erasure or non-bias-preserving rows; competitor bposd/tesseract have pair rows only.
src/mdr/ft/circuit_noise.py lines 55-75 (all rates default to 0). scripts/run_decoder_threshold_sweep.py line 71 (hw_tr = CircuitNoise(idle_mr_xyz=...)). ft_mdr_circuit.py lines 467-476 and 543-546.
docs/data/helios/: helios_cb is XYZ^2-only.
2112.06036.txt lines 1093-1119.
2306.17786.txt lines 1068-1072, 1083-1104, 1885-1887.
2109.13308.txt.
lit_hardware.md lines 28-32, 106 and 123-133. lit_surface_family.md lines 125-126.
schedule-search results.md sections 2 and 5.

### C31: Phenomenological noise with Z bias eta = 10 (phen_b10; readout flips at q = p on weight-6 and weight-2 checks alike; not hardware). Equal qu

**Verdict:** competitive (confidence 0.75, round-3 agreement 3/10).

Measured in pooled pilots, not campaign standard. XYZ^2 BM in its weaker basis against XZZX corr, XZZX's lower-error decoder here. The draft's size labels were shifted one step; these are corrected.

At 18/16 data qubits (35/31 total-equivalent), p = 3%: 0.70(5). The XZZX partner (d = 4) has even distance.

At 50/49 data qubits (99/97):
- 0.81(4) at p = 3%;
- 0.73 at 2.5%;
- 0.80 at 2%. The X-basis replicates are overdispersed (chi2 16.4 on 6 dof), so with the scatter this is 0.80(7), z about 2.8;
- 0.99(16) at 1.5%: the edge is gone.

At 98/100 data qubits (195/199): 0.91(8) at 3% against a directly measured XZZX d = 10, and 0.83(12) at 2.5%. Both are ties. At 2% it is unresolved (8 errors per basis).

So the edge is significant at the two smallest sizes at p = 3%. It fades with size (Lambda 3.6 for d 5 to 7 is below XZZX's Lambda^sqrt2 of about 4.1) and with falling p.

It also depends on decoder and readout:
- With corr_gauge XYZ^2 is 1.6x worse; with MWPM 9.6x worse (p = 2%, 50/49).
- With noiseless readout (q = 0) at p = 3% it is still 0.62 at 18/16, but becomes a 1.44(9)x loss at 50/49 (C9).
- At eta = 100: 0.98(9). At q = p/10, p = 5%: 0.92(5).
- So more readout noise helped XYZ^2 here relative to XZZX. That runs against the hw_tr result at p_T 1.5% / readout 2.7% (C37). The cause is not isolated.

XY and other competitors:
- XY is not the best competitor here, and it was decoded without ML.
- Rectangular and tailored XZZX were not run.

This is a second idealized, non-hardware instance of a small-size XYZ^2 edge, alongside C35.

**Evidence.** Pilots, node-local login13 scratchpad docs/data/campaign/usefulness_pilots.
18/16, p = 3% (bq/p03.log): XYZ^2 d = 3 Y 3.823e-3 (408), X 3.609e-3 (515) vs XZZX d = 4 corr X 5.441e-3 (503), BM X 5.614e-3.
50/49, p = 3%:
- XYZ^2 X: bq/p03.log 8.392e-4 (254) and ul/c_xyz2_X_d5_p0.03 8.881e-4 (260); Y 7.858e-4 (238).
- XZZX corr X: ul/e_phenb10_xzzx_X_d7 1.067e-3 (25,589); bq/p03.log 1.021e-3 (688).
50/49, p = 2%:
- XYZ^2 X pooled 341/151,256 shots, about 1.50e-4 (stat/run_p02_*, stat/run_0.02_*, ul/b10*, ul/c_*, bq/p02b.log, ovc/res20.jsonl).
- XZZX corr X 1.81-1.88e-4 (bq/p02.log, p02b.log, stat, ul).
50/49, p = 1.5% (stat/run_0.015_*): XYZ^2 BM X 22/23,917 and 15/24,557, Y 18/24,006 (about 5.1e-5) vs XZZX corr X 5.156e-5 (2718).
98/100, p = 3%: XYZ^2 X 188 errors over 39,480 shots, about 2.27e-4 (bq/p03.log, p03b.log, ul/c_*, ul/e_*) vs XZZX d = 10 corr X 2.490e-4 (486), Z 1.878e-4.
98/100, p = 2% (bq/p02.log): XYZ^2 X 8 and Y 8 errors in 25,024 shots vs XZZX d = 10 corr X 1.943e-5, Z 1.684e-5.
Decoders (stat/run_0.02_*_m.log): corr_gauge 3.03e-4, MWPM 1.80e-3.
q = 0: bq/cc4.log, cons/cc_phen0_*. eta 100: ul/e_phenb100_*. q = 0.1p: ul/f_phenb10q0.1_*.
Replicate dispersion: rev_disp.py.
thresholds.csv phen_b10 r = d: 5.539(54) vs 5.537(55).

### C32: Circuit-level Z-biased gate noise at eta = 10 (biased10, the bias of the original paper's equal-qubit claim and the realistic cat-qubit leve

**Verdict:** not_useful (confidence 0.8, round-3 agreement 10/10).

Measured in pooled pilots by several analyses (not campaign standard). The replicates are homogeneous.
- With XYZ^2 on BM, its weaker basis is 3.2-3.3x worse than XZZX d = 7 on corr (+-8%). Per tick (8 vs 6) about 2.5x.
- With correlated matching it is 6.3-6.9x worse.

With the r = d threshold ratio of 0.64 (C5) and the biased100/purez results (C6), the original paper's code-capacity equal-qubit advantage at eta = 10 does not carry over to gate-based extraction at this size and rate. Lower p and larger sizes are untested at eta = 10.

**Evidence.** XYZ^2 BM pooled: Y 159 errors in 41,264 shots, X 133 in 38,306 (2.58e-4 and 2.32e-4 per round). Sources (login13 scratchpad): ul/g_biased10_*, stat/runb10_*, bq/b10.log, ovc/resb10.jsonl, plus 9/3600 and 7/3900 runs.
XZZX d = 7 corr X at r = 21: 8.02e-5 (2541 errors), 8.05e-5 (2890), 7.75e-5 (2437), 7.78e-5 (400). bq/b10.log's XZZX rows use r = 15 and are not used.
corr_gauge: hwr/b10_xyz2_X/Y.log 5.317e-4 / 5.231e-4 (437 / 430 errors).
competitors_table.csv biased10 r = d: 0.748 vs 1.175.

### C33: Statistical standard of the threshold comparisons and of the pilots (all models)

**Verdict:** not_useful (confidence 0.82, round-3 agreement 8/10).

Measured (methodology).

1. Calibration.
- On identical SD6 detector error models the stat-only pull rms is 0.74-1.20, depending on the basis pairing, and 0.43-0.72 with stat+syst. The fit-window syst term is common to identical models, so this checks stat errors only.
- Pair-crossing comparisons have rms pulls up to 1.7. Mixed fss/pair comparisons reach about 2; deflate those z by about 2x.
- High-statistics fits are not inflated for misfit: XYZ^2 fits with chi2 > 4 have about 6x smaller relative stat errors than fits with chi2 < 1.5.

2. Selection.
- XYZ^2 entries are maxima over 19-27 usable candidates of the weaker basis. Competitor entries are maxima over about 12 fits, and in sd6, si1000, phen, hw_g and hw_rr over 3 isomorphic replicates too.
- The net upward bias for XYZ^2 is therefore well below 1-1.7 sigma in isomorphic models, and largest under bias, where XZZX alone sets the competitor value.

3. Multiplicity.
- competitors_table.csv has 261 rows; 107 are XYZ^2-vs-best comparisons. That implies about 2.4 one-sided false 2-sigma excursions (about 9 over 408 pairwise comparisons).
- 103 of 107 rows have XYZ^2 nominally lower. The 4 'xyz2' winners: phen_b10 r = d (z 0.0), phen r = 1 (1.0), h2_p r = 5 (2.3, an artifact), phen_b10 r = 10 (1.2, vs XY).
- The biased100 r = 10 nominal 3.9 sigma lead over XY is an MWPM-fallback artifact.
- The per-row ties are uniformly same-sign. With stat-only errors they are 3.5-7.7 sigma losses; they are ties only because of the uncalibrated syst term.

4. Pilots.
- XYZ^2 X-basis BM replicates are overdispersed in 3 of 17 replicate sets (p 0.012-0.021): phen_b10 p = 2%, the m1 mixture at 99 qubits, and hw_tr p = 3%. All Y-basis sets are consistent.
- Pilot ratios should use the replicate scatter; this revision does where it matters (C31, C35).
- Pilot seeds are time-based and not recorded.

Rule applied in every claim of this revision: only |z| >= 3 counts as a win or loss.
- Under it, XYZ^2 has no threshold win over the best competitor in any model.
- In the C35 diagnostic, only eta = 20 with noiseless gates and readout is significant at two sizes. eta = 10 (2.4 sigma) and the 195-qubit mixture point (1.1 sigma) are ties.
- Equal-qubit pair ratios resting on 1-3 errors (the 82x, 693x and 850x pairs) are not 3-sigma results.

**Evidence.** competitors_table.csv: 261 rows; 107 with a ratio; winners xzzx 53, css 20, honeycomb 18, xy 12, xyz2 4.
Pulls: reviewer statrev/pull.py on thresholds.csv SD6 css/xzzx/xy fss pairs.
Tie z values: statrev/z3.py. chi2 vs error: statrev/chi.py (5558 XYZ^2 fss fits).
Replicate chi2 (login13 scratchpad rev_disp.py, statrev/homog.py):
- phen_b10 p = 2% X: 16.4 on 6 dof.
- m1 99 X: 8.3 on 3 dof (80/3401, 69/3187, 45/3201, 64/3269).
- hw_tr p = 3% X: 40/33,432 vs 22/33,653.
- phen_b10 d = 7 p = 3% Y: p = 0.002.
pairs.csv.

### C34: Literature cross-check of the campaign's competitors and of the published case for XYZ^2

**Verdict:** not_useful (confidence 0.82, round-3 agreement 9/10).

Literature, consistent with the campaign at circuit level, with these qualifications.

1. Hetenyi-Wootton.
- Their combined Table I ranks XYZ^2 (0.41%) above XZZX (0.35%). Their readout corner puts XYZ^2 first (16.5 vs 14.5, within their 8-12% fit deviation).
- The first is not reproduced here (C19). The rotated (0.65%) and 3-CX (0.53%) surface codes still beat XYZ^2 in their table.
- Their table implies a phenomenological limit (p_T = p_RR) of XYZ^2/XZZX = 3.35/6.96 = 0.48 with ZZ links.
- They suggest XZZX-type and Floquet codes for spin qubits, and their slow-readout scenario names XZZX.

2. 2112.06036.
- Its equal-data-qubit advantage is code capacity at eta = 10 with near-ML (EWD) decoding, and only against XZZX in its distance-d frame.
- In the d^2 frame (Fig. 7c, Y bias) the failure rates scale with data-qubit count, i.e. parity at equal data qubits.
- The XY code's tensor-network threshold at eta = 10 (28.1%) also equals XYZ^2's (about 28%).
- So the literature predicts no ML-level advantage over a best-framed square surface code even at code capacity.
- The paper attributes its advantage to the pure-Z distance 2d^2, which should grow with bias. Our hw_tr edge instead vanishes at eta = 100 (C35).
- Its stated risks are borne out: weight-6 checks more susceptible to measurement noise; practical value conditional on a decoder and a circuit-level analysis.

3. 2505.03691 makes no XYZ^2-vs-XZZX claim for phenomenological noise.

Competitor baselines:
- XZZX MWPM SD6 0.69-0.71 vs 0.66 (2505.17718).
- Honeycomb em3 1.98 vs 1.5-2.0.
- Our SD6 honeycomb 0.218 is at the bottom of the published 0.2-0.35 range, so the relative wins over the honeycomb (C16) are inflated.

Decoders: 'XYZ^2 is not under-decoded' holds against published circuit and phenomenological decoders. Sequential matching reproduces 2505.03691 (3.42%, 3.49%), and our BP decoders exceed every published XYZ^2 decoder there. But in the code-capacity-like hw_tr corner our XYZ^2 MWPM crossing (10.5%) is under half the published BM and ML values (24.1%, 28.6%).

The 0.400 vs 0.41 MWPM match with Hetenyi-Wootton is a cross-model coincidence; their XZZX differs by 2x.

**Evidence.** 2306.17786.txt:
- Table I (lines 400-456): corners XYZ^2 0.465/4.2/16.5, surface 0.82/3.94/14.5, XZZX 0.37/15.1/12.9, 3-CX 0.65/4.1/8.7.
- Combined 0.408/0.648/0.351/0.527.
- Lines 570-575, 1222-1243, 1323-1333.
2112.06036.txt lines 26-37, 155-178, 1063-1119, 1291-1323.
2505.03691.txt lines 22-27, 335-341, 872, 913-917.
lit_surface_family.md line 125 (XY TN 28.1% at eta = 10, 1812.08186).
competitors_table.csv: honeycomb sd6 0.218, em3 1.976.
hw/diag_hw_tr_centers.csv.
docs/noncss_comparison.md line 95. lit_color_floquet_xyz2.md.

### C35: Idealized diagnostic hw_tr:

**Verdict:** competitive (confidence 0.6, round-3 agreement 0/10).

Measured in pilots. This is the only instance where XYZ^2 has a significantly lower error per round than the best simulated competitor at more than one size. The model is a diagnostic under the repo's own rules (fair_comparison.md rule 1 and section 3.4), and device-like changes remove the edge (C37). So this is an advantage in an idealized model, not a use case.

Weaker basis, XYZ^2 BM against XZZX corr (XZZX BM is equal within errors):
- 99/97 qubits, r = 3d: 0.61(4) at p_T = 6%, 0.61(8) at 4%, 0.68(8) at 3% (z 4.0-9.2).
- 195 qubits, p_T 6%: 0.54(10) against XZZX log-interpolated between d = 9 (161 qubits) and d = 11 (241). XZZX d = 11, with 24% more qubits, ties (1.06(20)).
- In qubits: XYZ^2 reaches XZZX's error with 0.81 (at 99) and 0.82 (at 195) of XZZX's qubits.
- r = 6d (99/97, 4%): 0.38(4). Only L_X improves with rounds (1.86e-4 to 1.15e-4). L_Y is flat (1.22e-4 to 1.21e-4) and was already 0.40(6) at r = 3d, so the r = 3d figure is set by an L_X time-boundary term.
- Size trend: XYZ^2's Lambda (d 5 to 7, 6%) is 6.5(13), against the break-even Lambda_XZZX^sqrt2 of about 5.5, so the edge grows nominally.
- With 0.1% gates and 1% readout (m1; still no reset, single-qubit gate or gate-layer idle noise): 0.70(7) at 99/97 (z about 4 after scaling for replicate scatter) and 0.81(17) at 195, a tie. Lambda 2.95 is below the break-even 3.35, so the edge is projected to erode (inference).

Conditions, under |z| >= 3:
- Bias: eta = 20 only. eta = 10 gives 0.81(8), a tie; eta = 100 gives 1.14(19).
- Competitors: at p_T = 3-4% and in m1 only square XZZX was run. XY (BM, 6%) is 5.2-5.5x worse than XYZ^2 and 3.2-3.4x worse than XZZX. CSS and the honeycomb were never run at equal qubits.
- Single-basis preparation: XYZ^2's better basis against XZZX's better basis (Z) is 0.90 (6%), 0.61 (4%), 0.91 (3%), 1.02 (eta 10) and 0.70 (m1). The draft's 'XZZX-X has no crossing below 44%' was a fit failure (C21). CSS-Z (|0>) is immune to the Z part of the channel (no crossing up to 44%; 0 errors at d = 9 up to p_T 12%), so for preparing |0> a CSS patch would win (inference; no equal-qubit run).

Mechanism (inference):
- hw_tr's bulk is exact repeated code capacity, the setting of 2112.06036. There the XYZ^2 edge holds against XZZX in its distance-d frame and vanishes at equal data qubits against the d^2 frame (Fig. 7c).
- Here the d^2-frame XZZX is 4.9x worse than XYZ^2 with BM, so with available decoders the d-frame square XZZX is the strongest competitor.
- Untested: whether an ML or tensor-network decoded best-framed, rectangular or co-prime surface patch removes the edge. Per-round code-capacity decoding applies directly because syndromes are perfect.
- The edge does not grow with bias, as the 2d^2 pure-Z distance would predict.
- Hetenyi-Wootton's scenario uses ZZ links and two readout windows per cycle; this result needs XX links and one window.

Look-elsewhere: about 12 pilot conditions were explored:
- hw_tr at eta 10/20/100 and p_T 3/4/6%, and r = 6d;
- the m1, m2 and m3 mixtures (m2 and m3 without comparators);
- phen_b10 at several p, with q = 0.1p and q = 0, and phen_b100;
- biased10.
The noiseless eta = 20 points survive (z 4-10). Poisson errors may be optimistic: 2 of 10 replicate pairs differ by about 2.3 sigma.

hw_tr charges noise per readout window. If gate time matters, XYZ^2's 8 vs 6 ticks lengthen its rounds, which is not modelled.

**Evidence.** Outputs on docs/data/campaign/usefulness_pilots/ (archived from login13 node-local /tmp). Scripts: cons/hwtr.py, hwtr_eta.py, hwtr_r.py, litrev/mix.py, litlens/hwtr_ybias.py.
litrev/, p_T = 6%, r = 3d:
- XYZ^2 d = 5 BM: Y 6.678e-4 (218/21,968), X 5.943e-4 (210).
- XZZX d = 7 corr: X 1.089e-3 (29,621), Z 6.635e-4. BM X 1.078e-3 (238).
- XZZX corr X d = 5 3.676e-3, d = 9 3.150e-4 (4486), d = 11 9.715e-5 (999).
- XYZ^2 d = 7 (195 qubits): Y 12/5410 and 17/8059 (about 1.03e-4), X 7/5397 and 19/8365.
- XY d = 7 BM: X 3.505e-3 (992), Z 3.692e-3.
p_T 4%: litrev/hwtr4_xyz2_X 1.855e-4 (62), Y 1.221e-4 (44) vs hwtr4_xzzx_X 3.050e-4 (11,782), Z 2.016e-4.
p_T 3% (cons/x3_*, z3_*): XYZ^2 X 40/33,432 and 22/33,653; Y 36/33,978 and 37/34,130 (7.15e-5) vs XZZX X 1.054e-4 (6166), Z 6.775e-5.
r = 6d (cons/b_*): XYZ^2 Y 99/27,298 (1.21e-4), X 93/27,143 (1.15e-4) vs XZZX X 3.163e-4 (14,848), Z 2.033e-4.
Bias (cons/e10*, e100*):
- eta 10: XYZ^2 Y 99/67,777 (9.76e-5), X 8.95e-5 vs XZZX X 1.208e-4 (7370), Z 8.743e-5.
- eta 100: XYZ^2 Y 5.14e-5 (36), X 4.19e-5 vs XZZX X 4.502e-5 (2922), Z 1.740e-5.
m1 (pT 4%, pG 0.1%, pR 1%, g1q = False):
- 99/97: XYZ^2 X 80/3401, 69/3187, 45/3201, 64/3269 (hwrev_q7/D0), about 1.34e-3; Y 61/3496, 52/3151, 43/3139 vs XZZX corr X 1.916e-3 (34,217), Z 1.452e-3.
- 195: XYZ^2 X 23/2429 (4.55e-4), Y 15/2420 vs XZZX d = 9 X 8.031e-4, d = 11 X 3.458e-4 (interpolated 5.6e-4).
d^2-frame XZZX (litlens/y_xzzx_*, hw_tr with Y-biased idling, p 4%): BM X 9.03e-4 (306), Z 9.18e-4; corr 2.58e-3.
hw/points_diag.csv (CSS-Z, XZZX-X).
scripts/run_decoder_threshold_sweep.py line 71; circuit_noise.py lines 55-75; ft_mdr_circuit.py lines 467-476, 543-546.
2112.06036.txt lines 155-159, 1095-1119. 2306.17786.txt lines 1083-1104, 1222-1243, 1885-1887, 2042.

### C36: Platform map: which device classes the campaign models represent, and whether they can host XYZ^2's circuit natively

**Verdict:** not_useful (confidence 0.75, round-3 agreement 8/10).

Inference from the model definitions, combined with the measured results.

Ions:
- Only the ion models (helios, h2) are calibrated to real devices that natively host XYZ^2's circuit (all-to-all). XYZ^2 loses on all of them (C11-C13).
- These models charge memory dephasing per gate layer, worth 8.7-9.8x of XYZ^2's error budget.
- A placement with memory error mainly in measurement and cooling windows is untested (C12).

Superconducting:
- SD6 is generic and platform-neutral; only SI1000 is superconducting-inspired.
- On degree-4 or heavy-hex chips this repo's degree-6 circuit would need routing, which was not modelled and so favours XYZ^2 (C1-C4).
- A heavy-hex 2-body compilation of a related hexagonal matching code exists (2109.13308). It has no threshold study, and the pair-measurement analogue here (em3) loses to the honeycomb.

Spin arrays (the hw corners):
- XYZ^2 loses wherever idling is charged in every gate layer (hw_t), and in the readout-only corner below threshold (hw_rr, C20).
- It leads only in the perfect-syndrome hw_tr diagnostic at eta = 20 with XX links (C35). That lead needs gate-layer dephasing of 2% of p_T or less, a long (not QEC-optimal) readout time, and one readout window per cycle (C37).
- Spin-blockade readout uses ancilla pairs, so XYZ^2 needs either two readout windows per cycle (Hetenyi-Wootton's 4d^2 layout, which uses ZZ links) or about 6d^2 qubits for one window. Its final logical readout also needs data qubits in readout pairs.
- Pauli spin blockade gives Z parity, so XX links would cost extra single-qubit gates (inference).
- Shuttling-based spin architectures put dephasing on every two-qubit gate, the hw_t-like placement (2601.10461, as reported by a reviewer).

Other:
- em3 and hyb have no existing device (QND pair measurements).
- Cat-qubit hardware deploys repetition codes, which were not run; the biased models cover their bias range and XYZ^2 loses there (C5).
- Neutral atoms, the other natively degree-6 class, were not modelled. sd6_noidle is a partial idle-free proxy, and XYZ^2 is at 0.774(8) there.
- phen, phen_b10, sd6_il, hw_tr and the gap variants are not hardware.

**Evidence.** src/mdr/ft/circuit_noise.py (model definitions; QUANTINUUM_P2 line 245).
scripts/run_decoder_threshold_sweep.py lines 56-71 (hw models; hw_tr marked diagnostic).
docs/fair_comparison.md section 3.4. docs/hetenyi_wootton_check.md section 2 (lines 35, 40).
gap/fractions.csv sd6_noidle 0.774(8). helios_budget.csv.
2109.13308.txt lines 1-14, 40-110.
2306.17786.txt lines 1076-1104, 1263-1268, 1423-1425, 1885-1887.
lit_hardware.md sections A, D and E.
Coupler degree from Stim builds.

### C37: Device-like variants of the hw_tr corner (spin-like hardware): gate-layer dephasing, readout duration, two readout windows per cycle, larger

**Verdict:** not_useful (confidence 0.65, round-3 agreement added after round 3).

Measured in pilots, except where marked. Each device-like change tried removes or reverses the C35 edge.

Gate-layer dephasing. eta = 20 dephasing at a fraction f of p_T was added to every gate layer of the 0.1% gate / 1% readout / 4% idle mixture:
- f = 0: 0.70(7).
- f = 0.02: 1.02(8), level.
- f = 0.05: 1.27(12), a nominal loss (2.3 sigma).
- XZZX's error rises 3% and 9%; XYZ^2's rises 51% and 99%. At equal qubits XYZ^2 has 2.4x the gate-layer idle sites per round (C25).
- hw_t (f = 1) is a large loss (threshold 0.13 of XZZX).

Readout duration.
- Hetenyi-Wootton's example device has maximum readout fidelity at 9.8 us but a QEC-optimal readout time of 5.8 us.
- At the QEC-optimal point (readout 2.7%, idle 1.5%, computed by one reviewer from their Eqs. 5, 6 and 12) XYZ^2 is 1.72(16)x worse.
- XZZX at that point (6.6e-4) also beats XYZ^2 at the maximum-fidelity point (1.34e-3).
- p_T is also lower there (1.5%), and the phen_b10 edge shrinks at low p (C31), so readout and p are confounded.

Readout windows (inference). Hetenyi-Wootton's 4d^2 XYZ^2 spin layout reads the links in a second round, so data idle through two windows per cycle. Modelled crudely as XYZ^2 at 2 p_T (6%) against XZZX at p_T (3%), XYZ^2 is 6.3x worse. A one-window layout needs about 6d^2 qubits against 3d^2.

Bias.
- eta = 100: 1.14(19), level. eta = 10: 0.81(8), a tie.
- Their Eq. 5 gives eta of about 9 at T1 = 1 ms and about 100 at T1 = 10 ms for their example (one reviewer's computation), so the window needs T1 of about 1-2 ms at those T2 and readout times.

Initial boundary. A full-strength first noise layer (phen q = 0) gives a 1.44(9) loss (C9).

What does not remove the edge: 0.3% gates give 0.75(6). Gate fidelity is not the binding constraint; gate-layer dephasing, readout duration and layout are.

Not modelled anywhere: reset or preparation error, single-qubit gate error (all runs g1q = False), the two-qubit error on the readout pair, SWAPs, basis changes for Z-parity readout of XX links, and quasi-static dephasing. Inference: each charges XYZ^2 more than XZZX (more gates, the boundary weakness of C9).

**Evidence.** login13 scratchpad hwrev_q7/ (script mixg.py; XYZ^2 d = 5 BM both schedule vs XZZX d = 7 corr; 99/97 qubits; r = 3d; eta 20):
- A, pG 0.3%: XYZ^2 X 3.850e-3 (174/3179), Y 3.144e-3 vs XZZX X 5.116e-3 (57,973).
- D0, f = 0: X 1.330e-3 (64/3269).
- C and D2, f = 0.02: X 108/3237 and 83/3238 (2.30e-3 and 1.75e-3); Y 67/3199 and 74/3210 vs XZZX X 1.982e-3 (32,049).
- B, f = 0.05: X 2.633e-3 (122), Y 2.664e-3 (123) vs XZZX X 2.096e-3 (33,367), Z 1.588e-3.
- E and E2 (pT 1.5%, pR 2.7%): XYZ^2 Y 56/3379 and 56/3308 (1.12e-3 and 1.15e-3), X 8.58e-4 (43) vs XZZX X 6.603e-4 (11,601), Z 5.765e-4.
Other runs:
- cons/m1_z_X 1.916e-3.
- cons/e100_*, cons/e10*_*.
- litrev/hwtr_xyz2_Y 6.678e-4 (p_T 6%) vs cons/z3_X 1.054e-4 (p_T 3%).
hw/summary.csv hw_t.
2306.17786.txt: Eq. 6 (lines 481-490); line 693 (tau_min 1.5 us, T1R 1 ms, tau_res 3 us, T2 50 us); lines 900-905 (9.8 vs 5.8 us); lines 1083-1104, 1885-1887; line 2042 ('two idling phase during measurements').
Idle-site counts: C25.

## Open questions (what would settle the contested and untested items)

- Does the hw_tr edge (C35) survive campaign-standard tests that also cover device realism? This needs:
  - f (gate-layer dephasing as a fraction of p_T) from 0 to 0.05, readout-to-idle ratios from 0.1 to 2, p_T 1-6%, eta 10-100;
  - d up to 9-11, with r = 3d and r >= 6d;
  - reset error and single-qubit gate error;
  - BM, bp_corr and CFE for XYZ^2, plus a matching-class decoder;
  - CSS, honeycomb and XY at equal qubits;
  - rectangular, co-prime (e.g. 6x7 = 83) and d^2-frame XZZX/XY, decoded with per-round ML or tensor-network decoders. Syndromes are perfect, so code-capacity decoding applies. The literature (2112.06036 Fig. 7c; XY TN 28.1% at eta = 10) predicts parity with a best-framed square code.
- A realistic spin compilation, which decides whether any device regime exists:
  - Hetenyi-Wootton's 4d^2 layout with two readout windows per cycle versus a one-window ~6d^2 layout;
  - ZZ versus XX links, including the basis changes XX links need under Z-parity readout;
  - SWAPs, readout-pair two-qubit error and spin-blockade final readout;
  - each code at its own QEC-optimal readout time (their Eq. 14), with T1-dependent eta;
  - quasi-static dephasing.
- Why the readout dependence is not monotonic. In phen_b10, noiseless readout turns XYZ^2's 0.81 win into a 1.44x loss at 50/49 data qubits. In hw_tr, readout 2.7% with idle 1.5% gives a 1.72x loss. Same-builder scans are needed: phen q = 0 versus hw_tr at equal p, with long memories (r >= 6d), to separate the time boundary from the builder, the decoder and the p dependence.
- Strong bias at p < 1e-3. An extrapolation of ln Lambda versus ln p puts the exponent crossover near p of about 5e-9 (biased100) and 4e-6 (purez), and the 10-17x gap at 99/97 would remain even then. Equal-qubit runs at p = 5e-4 to 1e-3, with d = 13-15 XZZX points and XYZ^2 BM at d >= 7, would confirm that no practical crossover exists.
- Rectangular single-basis surface patches at XYZ^2's exact footprint (3x6 = 35 qubits, 5x10 = 99) for r = 1 preparation under SI1000 and the ion models. They decide the last small-size preparation case (C11).
- Circuit-level biased10 equal-qubit runs at campaign standard (BM for XYZ^2 and XZZX, p = 1e-3 to 3e-3, d = 5-11). They directly test the original paper's eta = 10 claim at circuit level; only pooled single-point pilots exist (C32).
- Coverage gaps:
  - Equal-qubit runs under hyb, the ion models, em3 (against the honeycomb), SI1000 memory and hw_rr at campaign standard.
  - Competitor runs under helios_cb.
  - A gate-count timing model for ions, and an ion model with memory error in measurement and cooling windows (the only natively all-to-all route to a C35-like regime).
  - A neutral-atom model.
  - Routing costs on degree-4 and degree-3 chips, and a threshold study of the heavy-hex 2-body compilation of 2109.13308.
- Fairness gaps: competitor BP-OSD and Tesseract at fss (promised in fair_comparison.md section 3.2); a built pipelined XZZX circuit instead of the sd6_noidle bound; the surface codes' direct hyb compilation, measured rather than refused; a symmetric-rules threshold table as the headline; a worse-basis-per-point version of pairs.csv; BM for CSS in the equal-qubit stage.
- Biased noise with gates that do not preserve bias, for all codes (the 2505.17718 model); X- and Y-biased circuit noise; amplitude-damping idle noise with ZZ links (C28).
- Should the pilot scripts and outputs be committed (e.g. data/campaign/pilots/)? The pilots cited here were archived from node-local /tmp on login13 to docs/data/campaign/usefulness_pilots/{bq, cons, hwrev_q7, litlens, litrev, ovc, rev5, rv, stat, ul, hwr, fair, fairrev, statrev}. They will not persist, and their seeds were not recorded. Earlier scripts are already missing: milp_dist.py, milp_s0.py, bulk_d3.log, eqphen.py, pl.py, si.py, dev.py, hwreal_ion.py, plane.py, emb.py, yzprobe.py, wbulk.py, milp_cost.py.
- Documentation fixes the user may want:
  - Reclassify XYZ^2's biased numbers as bias-preserving: noncss_comparison.md line 15, lit_hardware.md line 114, lit_surface_family.md line 163.
  - noncss_comparison.md lines 70, 82 and 130: restrict 'beats every surface code' to the cat-state compilations, and drop 'beats the honeycomb on all gate-based hardware' (hw_g is a tie).
  - Fix the XZZX-X 'no crossing below 44%' entry in docs/data/campaign/hw/diag_hw_tr_centers.csv and in hetenyi_wootton_check.md lines 68 and 112. The file's own points cross near 19-30%.
  - In hetenyi_wootton_check.md: note that hw_tr is perfect-syndrome code capacity, that its equal-qubit pilot favours XYZ^2 only at eta = 20 with f = 0, and that their XYZ^2 uses ZZ links and two readout rounds per cycle.
  - Flag or drop the crosstalk-model thresholds and wins, the h2_p r = 5 entries, the biased100 r = 10 XY entry, the phen_b10 r = 10 'winner = xyz2' entry, and the sd6 both-Y cfe0 fit.
  - Quote both-schedule values where depth6 L_Y was used (biased10, em3, hyb_lr2, sd6_il).
  - Update equal_qubit_comparison.md: stage 3 is merged, and the measured BM d = 9 points give 28-51x.
  - Add a symmetric-rules threshold table as the headline.

## Agreement record

| round | claims at ≥ 80% | claims |
|---|---|---|
| 1 | 17 | 31 |
| 2 | 27 | 36 |
| 3 | 30 | 37 |
