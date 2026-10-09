# Syndrome-extraction search for XYZ² against XZZX (cloud run, 2026-10-09)

Base commit e2db03a. No file under `src/mdr/ft` was changed. All code is in this directory, the raw CSVs are in `data/`, and the full tables (every p_L with its error bar, every crossing) are in `tables.md`, produced by `make_tables.py`.

The machine had 2 cores and 7 GB, so the Monte Carlo is small: d = 5, 7, 9 with MWPM and d = 5, 7 with bp_full, and 100-300 errors per point (bp_full: 30-150). Crossings of two small distances are only an indication. The campaign rules ask for d ≥ 11 fits, and nothing here meets that rule.

## 1. Valid schedules and their distance (task 1)

- `ExtractionSchedule.enumerate_depth(6)` returns 8928 schedules. All 8928 also pass `validate` on the d = 3 patch, and the 1690 we built at d = 5 pass there too. `BOTH_BASES_SCHEDULE` is index 1525 and `DEPTH6_SCHEDULE` is index 910. The 8928 schedules contain 1536 distinct hexagon orders (sigma pairs).
- We measured fault distance with Stim's `search_for_undetectable_logical_errors` (r = 3, frame init and readout, `detectors="combined"`). The search gives an upper bound. At d = 3 it reproduces the documented CP-SAT count of 164 schedules with sd6 distance (X, Y) = (4, 3).

d = 3, all 8928 (sd6 X, sd6 Y): (2,2) 1982, (2,3) 1986, (3,2) 1904, (3,3) 1930, (4,2) 962, (4,3) 164. Under hyb no schedule exceeds X = 3, because a pair-measurement fault on a link costs one extra unit. 1690 schedules keep distance ≥ 3 in all four columns (sd6 X/Y, hyb X/Y).

d = 5, for those 1690: all 1690 keep full distance. 1526 have (sd6 X, sd6 Y, hyb X, hyb Y) = (5,5,5,5) and 164 have (6,5,5,5). These 1690 contain 220 distinct sigma pairs.

d = 7 (budget 3) was checked only for the finalists: s1525 under hyb (7, 7), s1525_5_R (7, 7), s7400_0_M (7, 7), s7402 under sd6 (8, 7), y20 (7, 7).

## 2. Compilations the module cannot express (task 2)

All of these are built by subclassing `FTMDRCircuit` (`compilations.py`, `split_hex.py`). `compilations._Decoder` is a `TwoLevelDecoder` subclass that builds the S₀ circuit from the compilation itself, because the base class always rebuilds a plain `FTMDRCircuit`.

1. **Hex-only sigma pairs (hyb).** When links are pair measurements, the link gate layers no longer constrain the schedule. 2664 sigma pairs satisfy the hexagon rules, which is 1128 more than `enumerate_depth` can express. 155 of the new pairs keep distance 3 at d = 3, and 150 of those keep distance 5 at d = 5. The best of them (h2261) is no better than the best enumerated sigma.
2. **Link pair-measurement step (hyb).** `FTMDRCircuit` always measures the links in the reset step R. `HybridLinkStep` puts the links of each block parity in R, in M, or in any gate layer where both vertices of the block are free. With the `BOTH_BASES_SCHEDULE` sigma, moving the even-block links into layer 5 (`s1525_5_R`) lowers p_L at p = 0.5% from 6.6% to 5.3% at d = 5 and from 7.1% to 5.5% at d = 7 (MWPM, weaker basis).
3. **Split hexagons, depth 3 (`SplitHexCircuit`, hyb).** Each weight-6 plaquette is split over two ancillas in a cat state (|+>|0> and a CX), with three controlled Paulis each. The gate part of a round then takes 3 layers instead of 6. 32 translation-invariant schedules exist (2 layer assignments times 16 half splits), and 16 of them keep full distance at d = 3 and d = 5. This compilation reduces data idling to 2 steps per round, but it adds a cat gate, a second reset and a second readout per hexagon. At d = 5 it lowers p_L by 20%. The MWPM crossing does not move (x12: 0.43% X, 0.50% Y).
4. **Split hexagons with two pipelined ancilla banks (`PipelinedSplitHexCircuit`, hyb).** The reset and cat gate of the next round's bank run during gate layers 1 and 2 of the current round. The links are measured by their pair measurements in the readout step. A round is then 4 steps (3 gate layers and M), and every bulk data qubit is busy in all of them. This is the only compilation that clearly lowers p_L. Qubits: 130 / 266 / 450 at d = 5 / 7 / 9, against 74 / 146 / 242 for the standard hyb circuit.
5. **Diagnostics (not hardware claims).** `*_noidle` sets p_idle = 0 everywhere, which also removes the data idling during reset and readout. `*_nolayeridle` removes idling only inside the gate layers and keeps the reset and readout idling that every SD6 circuit pays. These two bound what a shorter or pipelined compilation with the same gates can gain.
6. **Not built:** flag qubits and other ancilla bases. The distance is already full in both bases, so flags would only add gates and locations, while the losses we measured come from idling and entropy, not from hooks. With depolarizing noise, ancilla-in-|0> with data-controlled gates has the same hook structure as the ancilla-controlled form. A depth-4 pipelined split-hexagon circuit for sd6 (links as gates) is the natural next step, but we did not build it.

## 3. Top candidates (exact assignments)

Position order (TL, TRu, TRl, BR, BLl, BLu). The link is (upper, lower) per block parity (0 = even i + j).

| name | compilation | sigma A | sigma B | links |
|---|---|---|---|---|
| s1525 | BOTH_BASES_SCHEDULE (reference) | 0 3 4 5 2 1 | 0 1 2 4 5 3 | gates: even (2,3), odd (0,1). hyb: both parities in R |
| s7402 | best sd6 depth-6 (time reverse of the s1525 sigma) | 5 2 1 0 3 4 | 5 4 3 1 0 2 | gates: even (3,2), odd (5,4) |
| s1525_5_R | best depth-6 hyb | as s1525 | as s1525 | pair measurement: even blocks in gate layer 5, odd blocks in R |
| s7400_0_M | runner-up depth-6 hyb (mirror of s1525_5_R) | 5 2 1 0 3 4 | 5 4 3 1 0 2 | even blocks in layer 0, odd in M |
| y20 | pipelined split hexagons, hyb | tau 2 2 1 0 0 1, h1 = {TRu, TRl, BR} | tau 2 1 0 0 1 2, h1 = {TL, TRu, TRl} | pair measurement in M, both parities |
| y12 | pipelined split hexagons, hyb (time reverse of y20) | tau 0 0 1 2 2 1, h1 = {TL, BLl, BLu} | tau 0 1 2 2 1 0, h1 = {TL, TRu, TRl} | pair measurement in M |

Ranking used MWPM on the S₀ graph at d = 5, p = 0.5%: 220 sigma representatives under sd6 and hyb at 5000 shots, then the link variants of the top 25 sigmas (sd6) and the link-step variants (hyb), then the top entries at 40k shots and at d = 7. Under sd6 the spread among the top schedules is within 5% of s1525 (d = 7, p = 0.5%: s7402 8.3%, s1525 8.5%). Under hyb the link step is what matters.

## 4. Logical error rates (task 3)

p_L in % (r = d, ± one sigma). Weaker basis in bold. The full grid is in `tables.md`.

**p = 0.5%, MWPM**

| code / compilation | noise | d = 5 | d = 7 | d = 9 |
|---|---|---|---|---|
| XYZ² s1525 | sd6 | X 6.11 ± 0.24, **Y 7.28 ± 0.26** | X 7.41 ± 0.26, **Y 8.23 ± 0.27** | **X 9.74 ± 0.30**, Y 8.91 ± 0.28 |
| XYZ² s7402 | sd6 | X 6.07 ± 0.24, **Y 7.92 ± 0.27** | X 7.89 ± 0.27, **Y 8.11 ± 0.27** | **X 9.31 ± 0.29**, Y 9.06 ± 0.29 |
| XYZ² s1525 | hyb | X 4.48 ± 0.21, **Y 6.58 ± 0.25** | X 4.52 ± 0.21, **Y 7.09 ± 0.26** | X 4.95 ± 0.22, **Y 6.63 ± 0.25** |
| XYZ² s1525_5_R | hyb | X 4.98 ± 0.22, **Y 5.29 ± 0.22** | X 5.12 ± 0.22, **Y 5.54 ± 0.23** | **X 5.06 ± 0.22**, Y 4.64 ± 0.21 |
| XYZ² x12 (split, one bank) | hyb | X 4.40 ± 0.21, **Y 4.41 ± 0.21** | **X 5.05 ± 0.22**, Y 4.79 ± 0.21 | **X 5.88 ± 0.24**, Y 4.79 ± 0.21 |
| XYZ² y20 (split, pipelined) | hyb | X 2.47 ± 0.11, **Y 2.57 ± 0.11** | **X 2.18 ± 0.10**, Y 1.99 ± 0.10 | **X 2.41 ± 0.11**, Y 1.78 ± 0.09 |
| XZZX | sd6 (= hyb) | **X 3.31 ± 0.18**, Z 3.08 ± 0.17 | **X 2.40 ± 0.11**, Z 2.27 ± 0.11 | **X 1.91 ± 0.10**, Z 1.86 ± 0.10 |
| XZZX corr | sd6 (= hyb) | X 2.26 ± 0.11, **Z 2.44 ± 0.11** | **X 1.50 ± 0.09**, Z 1.39 ± 0.07 | X 0.83 ± 0.05, **Z 0.94 ± 0.05** |

**bp_full (XYZ²) against corr (XZZX), p = 0.5 / 0.6 / 0.7%**

| code / compilation | noise | d | X | Y (XZZX: Z) |
|---|---|---|---|---|
| XYZ² s1525 | hyb | 5 | 3.26 ± 0.30 / 5.40 ± 0.51 / 8.27 ± 0.71 | 3.57 ± 0.34 / 6.93 ± 0.66 / 10.80 ± 0.98 |
| | | 7 | 3.35 ± 0.40 / 5.35 ± 0.50 / 10.00 ± 0.95 | 2.80 ± 0.37 / 6.73 ± 0.65 / 10.40 ± 0.97 |
| XYZ² s1525_5_R | hyb | 5 | 3.37 ± 0.33 / 4.80 ± 0.43 / 10.00 ± 0.77 | 4.04 ± 0.39 / 5.15 ± 0.49 / 9.40 ± 0.75 |
| | | 7 | 2.90 ± 0.53 / 6.60 ± 0.79 / 10.60 ± 0.97 | 2.90 ± 0.53 / 5.40 ± 0.71 / 8.93 ± 0.74 |
| XYZ² y20 | hyb | 5 | - / 2.60 ± 0.25 / 3.53 ± 0.34 | - / 2.83 ± 0.26 / 3.97 ± 0.36 |
| | | 7 | - / 2.30 ± 0.34 / 4.70 ± 0.47 | - / 2.25 ± 0.33 / 3.70 ± 0.42 |
| XYZ² y12 | hyb | 5 | - / 2.14 ± 0.20 / 4.32 ± 0.41 | - / 3.14 ± 0.29 / 4.48 ± 0.41 |
| | | 7 | - / 1.85 ± 0.30 / 3.30 ± 0.40 | - / 1.55 ± 0.28 / 5.05 ± 0.49 |
| XYZ² s1525 | sd6 | 7 | 4.00 ± 0.51 / 8.70 ± 0.89 / 16.5 ± 1.2 | 4.20 ± 0.52 / 8.70 ± 0.89 / 13.6 ± 1.1 |
| XZZX corr | sd6 | 5 | 2.26 ± 0.11 / 4.00 ± 0.20 / 5.77 ± 0.23 | 2.44 ± 0.11 / 3.86 ± 0.19 / 5.48 ± 0.23 |
| | | 7 | 1.50 ± 0.09 / 3.36 ± 0.18 / 5.03 ± 0.22 | 1.39 ± 0.07 / 3.04 ± 0.17 / 4.75 ± 0.21 |
| | | 9 | 0.83 ± 0.05 / 2.14 ± 0.10 / 4.36 ± 0.20 | 0.94 ± 0.05 / 2.12 ± 0.10 / 4.50 ± 0.21 |

The sd6 bp_full runs for d = 5 and for s7402 were stopped to make room for the pipelined runs, so sd6 has no bp_full crossing here.

**Threshold estimates (weaker basis, crossing of d = 7 and 9 for MWPM and of d = 5 and 7 for bp_full)**

| | sd6 | hyb |
|---|---|---|
| XYZ² s1525, MWPM | 0.38 ± 0.02 (X) | 0.47 ± 0.02 (X) |
| XYZ² best depth-6, MWPM | s7402: 0.39 ± 0.02 (X) | s1525_5_R: 0.51 ± 0.02 (X) |
| XYZ² y20 / y12 pipelined, MWPM | not built | 0.45 ± 0.02 (X) / 0.51 ± 0.02 (Y) |
| XYZ² s1525, bp_full (5/7) | - | 0.60 ± 0.05 (X), Y above 0.7 |
| XYZ² s1525_5_R, bp_full (5/7) | - | 0.53 ± 0.02 (X) |
| XYZ² y20 / y12, bp_full (5/7) | - | 0.63 ± 0.02 (X) / 0.68 ± 0.01 (Y, X above 0.7) |
| XZZX, MWPM | 0.64 ± 0.05 (X) | same circuit |
| XZZX, corr | 7/9: 0.80 ± 0.04 (X), 0.91 (Z). 5/7: 0.83 (X), 0.80 (Z) | same circuit |
| XZZX pipelined (data never idle, `sd6_noidle`), corr | 7/9: 1.00 ± 0.06 (Z). 5/7: 1.07 (X and Z) | same circuit |
| *diag* XYZ² s1525 `*_nolayeridle`, MWPM | 0.57 ± 0.04 | 0.66 ± 0.02 (X) |
| *diag* XYZ² s1525 `*_noidle`, MWPM | 0.65 ± 0.02 | above 0.7 |

## 5. Verdict

Under the fair rule, no compilation we found closes the gap to XZZX.

1. **Schedule choice alone (tasks 1 and 2.1) does almost nothing.** Under sd6 the best of 1690 full-distance depth-6 schedules is within statistics of `BOTH_BASES_SCHEDULE`, with MWPM crossings of 0.38-0.39% against 0.64% for XZZX. Under hyb the link pair-measurement step moves the MWPM crossing from 0.47% to about 0.51% and lowers p_L by about 20% at d = 7. Its bp_full crossing at d = 5/7 (0.53%) is no better than s1525's. That is real but small next to a factor of 1.6.
2. **Idling inside the gate layers is the largest loss we could locate.** A weight-6 hexagon with one ancilla needs 6 layers, while a data qubit has only 4 gates (sd6) or 3 gates (hyb) per round. Removing that idling alone (`nolayeridle`) raises the XYZ² MWPM crossing from 0.38% to 0.57% (sd6) and from 0.47% to 0.66% (hyb). The hyb value is on par with XZZX MWPM (0.64%).
3. **The pipelined split-hexagon circuit (y20, y12) removes most of that idling in hardware terms.** It cuts p_L by a factor of about 3 against s1525 under hyb at d = 7 and 9 (p = 0.5%, MWPM: 2.2% against 7.1% at d = 7, 2.4% against 6.6% at d = 9). At p = 0.6%, d = 7, bp_full gives 1.6-2.3% against 3.0-3.4% for XZZX with correlated matching. At d = 5 it matches XZZX-corr. But its p_L grows slowly with d near 0.5-0.6%, so its crossings barely move: MWPM 0.45-0.51%, and bp_full 0.63-0.68% at d = 5/7 against 0.60% for s1525 and 0.80% for XZZX-corr at the same distances. XZZX pulls ahead again at d = 9 (2.4% against 0.9% at p = 0.5%).
4. **Fairness cancels the gain.** The pipelined circuit uses a second ancilla bank. The same hardware lets XZZX pipeline its own reset and readout, and then its data qubits never idle. That XZZX (`sd6_noidle`) crosses at about 1.0% with correlated matching, so the ratio stays near 0.6-0.7 for every compilation we tried. The pipelined XYZ² also uses 1.8 times the qubits of the standard XYZ² circuit and of a pipelined XZZX at the same d (266 against 146 and 145 at d = 7).

The remaining gap is intrinsic to the weight-6 plaquettes (more faults per check and more entropy per syndrome bit), not to the gate order. The only lever that moved p_L by a large factor (removing idling) is equally available to XZZX. One open lead is a depth-4 pipelined split-hexagon circuit for sd6 with the links as gates. The other is to check whether y20's flat d-scaling persists at d ≥ 11 with bp_full or correlated decoding. If it does not, the pipelined circuit could beat the standard (non-pipelined) XZZX circuit at low p. That is still not a fair win unless the hardware forbids a second ancilla bank for XZZX.

## Reproduce

```
python cloud/schedule_search/distance_scan.py --d 3 --rounds 3 --k 4 --out cloud/schedule_search/data/dist_d3.csv
python cloud/schedule_search/distance_scan.py --d 5 --rounds 3 --k 3 --only cloud/schedule_search/data/pass_d3.txt --out cloud/schedule_search/data/dist_d5.csv
python cloud/schedule_search/distance_scan.py --family hex --noises hyb --d 3 --only cloud/schedule_search/data/hex_new.txt --out cloud/schedule_search/data/hex_d3.csv
python cloud/schedule_search/rank.py --comps-file cloud/schedule_search/data/pass_d5_sigma_reps.txt --prefix s --noises sd6 hyb --d 5 --p 5e-3 --shots 5000 --seed 1 --out cloud/schedule_search/data/rank1_enum_d5.csv
python cloud/schedule_search/distance_check.py --d 7 --jobs y20:hyb s1525_5_R:hyb
python cloud/schedule_search/ler_runner.py --comps y20 s1525_5_R --noises hyb --ps 4e-3 5e-3 6e-3 --ds 5 7 9 --decoders mwpm --out cloud/schedule_search/data/ler_mwpm.csv
python cloud/schedule_search/ler_runner.py --code xzzx --noises sd6 --ps 5e-3 6e-3 7e-3 --ds 5 7 9 --logicals X Z --decoders mwpm corr --out cloud/schedule_search/data/ler_xzzx.csv
python cloud/schedule_search/crossings.py cloud/schedule_search/data/ler_*.csv
python cloud/schedule_search/make_tables.py > cloud/schedule_search/tables.md
```

Compilation names: `s<i>` is `enumerate_depth(6)[i]`. `s<i>_<even>_<odd>` and `h<i>_<even>_<odd>` add a hyb link step per parity (R, M or a gate layer), with `h` for `compilations.hex_only_sigma_pairs()[i]`. `x<i>` and `y<i>` are `split_hex.enumerate_split(5)[i]` with one or two ancilla banks. Noise names are those of `run_decoder_threshold_sweep.NOISE` plus `*_noidle` and `*_nolayeridle`.
