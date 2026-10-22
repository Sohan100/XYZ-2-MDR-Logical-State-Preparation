# XYZ^2 against competitor codes

Thresholds in % of p. A competitor's value is its best decoder in its weaker basis; under a variant that changes the hardware, its best compilation on that hardware ("as sd6": all gates, "as em3": all pair measurements). XYZ^2's value is, per extraction schedule, its weaker logical basis (L_X or L_Y, each with its best decoder), then its best schedule; "only" marks a model with one basis measured. Ratio: XYZ^2 over the best competitor. Only finite-size fits over d >= 11 count: thresholds that drift, crossings of two distances only, and series that stop below d = 11 (honeycomb: d = 12) are left out. sd6_il (noiseless link checks) is a diagnostic, not hardware. "best" is the largest central value, with no significance test: a row is an XYZ^2 win only when XYZ^2 is at least 3 sigma (stat plus syst) above every competitor, and no row here passes that test. Two "xyz2" rows are fit or coverage artifacts, and competitors_table.csv has the same winner = xyz2 rows. h2_p, r = 5: the 0.82 comes from one Tesseract fit (L_X only, d = 11-21, 7 points) with nu = 6.00, which is the upper bound of the fit, and chi2 = 14.8. The other XYZ^2 finite-size fits at r = 5 give 0.42 (bm) and 0.38 (seq_soft), and XYZ^2 gives 0.31 at r = 10 and at r = d. XY has no fit and the honeycomb code has none over d >= 12. phen_b10, r = 10: XZZX has no fit (its X basis has only drifting two-distance crossings) and CSS has none in Z. Against XY, XYZ^2 has 6.84 +- 0.33 and XY has 6.25 +- 0.38, which is 1.19 sigma apart (stat plus syst), so the two are not resolved. phen_b10, r = d (ratio 1.00): XYZ^2 has 5.54 +- 0.05 and XZZX has 5.54 +- 0.05, which is a tie.

## r = d

| noise | XYZ^2 | CSS | XZZX | XY | Honeycomb | ratio | best |
|---|---|---|---|---|---|---|---|
| sd6 | 0.51 (bp_corr, both L_X) | 0.81 (corr, X) | 0.81 (bm, Z) | 0.81 (bm, X) | 0.22 (bm, Z) | 0.63 | xzzx |
| si1000 | 0.43 (bm, both L_X) | 0.52 (bm, Z) | 0.51 (bm, Z) | 0.51 (bm, Z) | 0.12 (bm, Z) | 0.83 | css |
| biased10 | 0.75 (bp_corr, depth6 L_Y) | 0.64 (corr, X) | 1.17 (corr, Z) | 1.02 (bm, Z) | 0.23 (bm, X) | 0.64 | xzzx |
| biased100 | 0.84 (bp_full, depth6 L_X) | 0.62 (bm, X) | 1.65 (corr, Z) | 1.05 (bm, Z) | 0.23 (bm, X) | 0.51 | xzzx |
| purez | 0.86 (bp_full, depth6 L_X) | 0.62 (corr, X) | 2.06 (corr, Z) | 1.05 (bm, Z) | 0.24 (bm, X) | 0.42 | xzzx |
| em3 | 0.33 (seq_soft, depth6 L_Y) | 0.24 (corr, X) | 0.25 (corr, X) | 0.24 (corr, X) | 1.98 (corr, X) | 0.16 | honeycomb |
| helios_p_noxt | 0.41 (bp_corr, both L_X) | 0.40 (corr, X) | 0.54 (corr, Z) | 0.53 (bm, Z) | 0.23 (bm, X) | 0.76 | xzzx |
| h2_p_noxt | 0.58 (bp_corr, both L_X) | 0.64 (corr, X) | 0.75 (corr, Z) | 0.77 (bm, X) | 0.36 (bm, X) | 0.76 | xy |
| helios_p | 0.05 (seq_match, both L_Y) | 0.28 (corr, X) | 0.33 (bm, X) | -- | -- | 0.16 | xzzx |
| h2_p | 0.31 (bm, both L_X) | -- | 0.41 (bm, X) | -- | 0.17 (bm, X) | 0.77 | xzzx |
| hyb | 0.59 (bp_corr, both L_X) | 0.81 (corr, X, as sd6) | 0.81 (bm, Z, as sd6) | 0.81 (bm, X, as sd6) | 1.98 (corr, X, as em3) | 0.30 | honeycomb |
| hyb_lr2 | 0.62 (corr_gauge, depth6 L_Y) | 0.81 (corr, X, as sd6) | 0.81 (bm, Z, as sd6) | 0.81 (bm, X, as sd6) | 1.98 (corr, X, as em3) | 0.31 | honeycomb |
| sd6_lr2 | 0.57 (seq_soft, both L_X) | 0.81 (corr, X, as sd6) | 0.81 (bm, Z, as sd6) | 0.81 (bm, X, as sd6) | 0.22 (bm, Z, as sd6) | 0.70 | xzzx |
| sd6_lr3 | 0.53 (corr_links, both L_X) | 0.81 (corr, X, as sd6) | 0.81 (bm, Z, as sd6) | 0.81 (bm, X, as sd6) | 0.22 (bm, Z, as sd6) | 0.65 | xzzx |
| si1000_lr2 | 0.44 (bp_corr, both L_X) | 0.52 (bm, Z, as si1000) | 0.51 (bm, Z, as si1000) | 0.51 (bm, Z, as si1000) | 0.12 (bm, Z, as si1000) | 0.85 | css |
| phen | 4.16 (bp_corr, depth6 L_X) | 4.57 (bm, Z) | 4.51 (bm, X) | 4.55 (bm, Z) | 0.74 (bm, X) | 0.91 | css |
| phen_b10 | 5.54 (bp_full, both L_X) | 3.11 (bm, X) | 5.54 (corr, Z) | 5.37 (bm, Z) | 0.77 (bm, X) | 1.00 | xyz2 |
| sd6_il | 0.64 (corr_gauge, depth6 L_Y) | 0.81 (corr, X, as sd6) | 0.81 (bm, Z, as sd6) | 0.81 (bm, X, as sd6) | 0.22 (bm, Z, as sd6) | 0.79 | xzzx |

MWPM only, r = d:

| noise | XYZ^2 | CSS | XZZX | XY | Honeycomb |
|---|---|---|---|---|---|
| sd6 | 0.40 | 0.69 | 0.70 | 0.69 | 0.17 |
| si1000 | 0.34 | 0.44 | 0.44 | 0.44 | 0.08 |
| biased10 | 0.44 | 0.61 | 1.10 | 0.60 | 0.16 |
| biased100 | 0.45 | 0.59 | 1.61 | 0.60 | 0.16 |
| purez | 0.45 | 0.60 | 2.05 | 0.59 | 0.16 |
| em3 | 0.25 | 0.22 | 0.22 | 0.22 | 1.60 |
| helios_p_noxt | 0.27 | 0.37 | 0.49 | 0.37 | 0.16 |
| h2_p_noxt | 0.42 | 0.56 | 0.67 | 0.56 | 0.25 |
| helios_p | -- | -- | -- | -- | -- |
| h2_p | -- | -- | -- | -- | -- |
| hyb | 0.47 | 0.69 | 0.70 | 0.69 | 1.60 |
| hyb_lr2 | 0.47 | 0.69 | 0.70 | 0.69 | 1.60 |
| sd6_lr2 | 0.40 | 0.69 | 0.70 | 0.69 | 0.17 |
| sd6_lr3 | 0.36 | 0.69 | 0.70 | 0.69 | 0.17 |
| si1000_lr2 | 0.32 | 0.44 | 0.44 | 0.44 | 0.08 |
| phen | 3.25 | 3.79 | 3.81 | 3.76 | 0.50 |
| phen_b10 | 3.13 | 2.99 | 5.50 | 2.95 | 0.51 |
| sd6_il | 0.56 | 0.69 | 0.70 | 0.69 | 0.17 |

## r = 1

| noise | XYZ^2 | CSS | XZZX | XY | Honeycomb | ratio | best |
|---|---|---|---|---|---|---|---|
| sd6 | 1.76 (bp_full, both L_X) | 2.41 (bm, X) | 2.42 (corr, Z) | 2.38 (bm, Z) | 0.83 (mwpm, Z) | 0.73 | xzzx |
| si1000 | 1.06 (tnml, depth6 L_Y) | 1.08 (bm, X) | 1.06 (bm, Z) | 1.06 (bm, X) | 0.40 (bm, Z) | 0.98 | css |
| biased10 | 1.97 (cfe_tn, depth6 L_X) | 2.19 (bm, X) | 3.18 (mwpm, Z) | 2.16 (mwpm, X) | 0.73 (corr, Z) | 0.62 | xzzx |
| biased100 | 1.97 (tnml, depth6 L_X) | 2.14 (bm, X) | 3.45 (mwpm, Z) | 2.12 (bm, X) | 0.72 (corr, Z) | 0.57 | xzzx |
| purez | 1.96 (tnml, depth6 L_Y) | 2.25 (bm, X) | 3.48 (corr, Z) | 2.14 (corr, Z) | 0.71 (mwpm, Z) | 0.56 | xzzx |
| em3 | 1.12 (tnml, both L_X) | 0.82 (bm, X) | 0.81 (corr, X) | 0.79 (bm, X) | 3.50 (mwpm, Z) | 0.32 | honeycomb |
| helios_p_noxt | 1.51 (tnml, both L_X) | 1.61 (mwpm, X) | 2.37 (mwpm, Z) | 1.61 (mwpm, X) | 0.73 (corr, Z) | 0.64 | xzzx |
| h2_p_noxt | 2.14 (bp_corr, both L_X) | 2.44 (mwpm, X) | 3.03 (bm, X) | 2.40 (mwpm, X) | 1.22 (corr, Z) | 0.71 | xzzx |
| helios_p | 0.33 (cfe, depth6 L_X only) | 0.53 (mwpm, Z) | 0.59 (corr, X) | 0.56 (corr, Z) | -- | 0.56 | xzzx |
| h2_p | 1.16 (cfe_tn, depth6 L_X only) | 1.24 (corr, X) | 1.39 (corr, Z) | 1.27 (bm, Z) | 0.89 (mwpm, Z) | 0.83 | xzzx |
| hyb | 1.86 (tnml, both L_Y) | 2.41 (bm, X, as sd6) | 2.42 (corr, Z, as sd6) | 2.38 (bm, Z, as sd6) | 3.50 (mwpm, Z, as em3) | 0.53 | honeycomb |
| hyb_lr2 | 2.30 (tnml, depth6 L_X) | 2.41 (bm, X, as sd6) | 2.42 (corr, Z, as sd6) | 2.38 (bm, Z, as sd6) | 3.50 (mwpm, Z, as em3) | 0.66 | honeycomb |
| sd6_lr2 | 1.95 (tnml, depth6 L_X) | 2.41 (bm, X, as sd6) | 2.42 (corr, Z, as sd6) | 2.38 (bm, Z, as sd6) | 0.83 (mwpm, Z, as sd6) | 0.81 | xzzx |
| sd6_lr3 | 1.81 (tnml, both L_X) | 2.41 (bm, X, as sd6) | 2.42 (corr, Z, as sd6) | 2.38 (bm, Z, as sd6) | 0.83 (mwpm, Z, as sd6) | 0.75 | xzzx |
| si1000_lr2 | 1.05 (tnml, depth6 L_Y) | 1.08 (bm, X, as si1000) | 1.06 (bm, Z, as si1000) | 1.06 (bm, X, as si1000) | 0.40 (bm, Z, as si1000) | 0.98 | css |
| phen | 16.95 (cfe, both L_X) | 15.68 (bm, X) | 15.72 (bm, Z) | 15.45 (bm, Z) | 2.98 (corr, Z) | 1.08 | xyz2 |
| phen_b10 | 15.46 (tnml, both L_Y) | -- | 25.77 (mwpm, Z) | 10.49 (corr, Z) | 3.00 (bm, Z) | 0.60 | xzzx |
| sd6_il | 2.13 (tnml, both L_X) | 2.41 (bm, X, as sd6) | 2.42 (corr, Z, as sd6) | 2.38 (bm, Z, as sd6) | 0.83 (mwpm, Z, as sd6) | 0.88 | xzzx |

MWPM only, r = 1:

| noise | XYZ^2 | CSS | XZZX | XY | Honeycomb |
|---|---|---|---|---|---|
| sd6 | 1.72 | 2.37 | 2.34 | 2.34 | 0.83 |
| si1000 | 0.96 | 0.99 | 1.03 | 1.01 | 0.38 |
| biased10 | 1.84 | 2.15 | 3.18 | 2.10 | 0.73 |
| biased100 | 1.87 | 2.11 | 3.44 | 2.06 | 0.70 |
| purez | 1.88 | 2.09 | 3.37 | 2.05 | 0.71 |
| em3 | 1.00 | 0.78 | 0.78 | 0.78 | 3.50 |
| helios_p_noxt | 1.42 | 1.61 | 2.37 | 1.61 | 0.71 |
| h2_p_noxt | 2.03 | 2.44 | 2.95 | 2.29 | 1.21 |
| helios_p | -- | 0.53 | -- | 0.36 | -- |
| h2_p | -- | -- | -- | -- | 0.89 |
| hyb | 1.83 | 2.37 | 2.34 | 2.34 | 3.50 |
| hyb_lr2 | 1.88 | 2.37 | 2.34 | 2.34 | 3.50 |
| sd6_lr2 | 1.75 | 2.37 | 2.34 | 2.34 | 0.83 |
| sd6_lr3 | 1.57 | 2.37 | 2.34 | 2.34 | 0.83 |
| si1000_lr2 | 0.93 | 0.99 | 1.03 | 1.01 | 0.38 |
| phen | 15.85 | 14.51 | 14.94 | 14.95 | 2.84 |
| phen_b10 | 14.41 | -- | 25.77 | 10.44 | 2.98 |
| sd6_il | 2.05 | 2.37 | 2.34 | 2.34 | 0.83 |

## r = 2

| noise | XYZ^2 | CSS | XZZX | XY | Honeycomb | ratio | best |
|---|---|---|---|---|---|---|---|
| sd6 | 1.23 (bp_corr, both L_X) | 1.75 (bm, X) | 1.74 (bm, Z) | 1.75 (corr, Z) | 0.53 (mwpm, X) | 0.71 | css |
| si1000 | 0.81 (bm, both L_X) | 0.90 (bm, Z) | 0.88 (bm, Z) | 0.89 (bm, X) | 0.28 (bm, X) | 0.90 | css |
| biased10 | 1.41 (bp_full, depth6 L_X) | 1.50 (bm, X) | 2.49 (mwpm, X) | 1.76 (bm, Z) | 0.51 (mwpm, Z) | 0.56 | xzzx |
| biased100 | 1.46 (bp_full, depth6 L_Y) | 1.52 (bm, X) | 2.87 (mwpm, Z) | 1.69 (bm, X) | 0.49 (corr, Z) | 0.51 | xzzx |
| purez | 1.45 (bp_full, both L_X) | 1.50 (bm, X) | 2.88 (mwpm, Z) | 1.69 (bm, X) | 0.50 (corr, Z) | 0.50 | xzzx |
| em3 | 0.74 (seq_soft, both L_X) | 0.55 (bm, X) | 0.54 (corr, X) | 0.56 (bm, X) | 3.39 (bm, X) | 0.22 | honeycomb |
| helios_p_noxt | 0.99 (bp_full, both L_Y) | 1.04 (bm, X) | 1.57 (bm, Z) | 1.12 (bm, Z) | 0.53 (corr, Z) | 0.63 | xzzx |
| h2_p_noxt | 1.43 (bp_corr, both L_X) | 1.58 (bm, X) | 1.89 (bm, Z) | 1.61 (bm, X) | 0.88 (corr, Z) | 0.76 | xzzx |
| helios_p | -- | 0.78 (corr, X) | 0.67 (mwpm, Z) | 0.55 (corr, Z) | -- | -- |  |
| h2_p | 0.83 (cfe0, depth6 L_X only) | 1.63 (bm, X) | 0.83 (corr, X) | 0.83 (corr, X) | 0.57 (corr, Z) | 0.51 | css |
| hyb | 1.36 (bp_corr, both L_Y) | 1.75 (bm, X, as sd6) | 1.74 (bm, Z, as sd6) | 1.75 (corr, Z, as sd6) | 3.39 (bm, X, as em3) | 0.40 | honeycomb |
| hyb_lr2 | 1.51 (bp_corr, depth6 L_Y) | 1.75 (bm, X, as sd6) | 1.74 (bm, Z, as sd6) | 1.75 (corr, Z, as sd6) | 3.39 (bm, X, as em3) | 0.44 | honeycomb |
| sd6_lr2 | 1.32 (seq_erasure, both L_X) | 1.75 (bm, X, as sd6) | 1.74 (bm, Z, as sd6) | 1.75 (corr, Z, as sd6) | 0.53 (mwpm, X, as sd6) | 0.76 | css |
| sd6_lr3 | 1.24 (bm, depth6 L_Y) | 1.75 (bm, X, as sd6) | 1.74 (bm, Z, as sd6) | 1.75 (corr, Z, as sd6) | 0.53 (mwpm, X, as sd6) | 0.71 | css |
| si1000_lr2 | 0.85 (bm, both L_X) | 0.90 (bm, Z, as si1000) | 0.88 (bm, Z, as si1000) | 0.89 (bm, X, as si1000) | 0.28 (bm, X, as si1000) | 0.94 | css |
| phen | 9.72 (bp_corr, depth6 L_X) | 9.87 (bm, X) | 10.02 (bm, Z) | 9.79 (bm, X) | 1.86 (corr, X) | 0.97 | xzzx |
| phen_b10 | 9.82 (bm, depth6 L_X) | -- | 15.41 (corr, Z) | 8.10 (bm, Z) | 1.70 (corr, Z) | 0.64 | xzzx |
| sd6_il | 1.51 (bp_corr, depth6 L_Y) | 1.75 (bm, X, as sd6) | 1.74 (bm, Z, as sd6) | 1.75 (corr, Z, as sd6) | 0.53 (mwpm, X, as sd6) | 0.87 | css |

MWPM only, r = 2:

| noise | XYZ^2 | CSS | XZZX | XY | Honeycomb |
|---|---|---|---|---|---|
| sd6 | 1.11 | 1.66 | 1.66 | 1.70 | 0.51 |
| si1000 | 0.75 | 0.87 | 0.86 | 0.88 | 0.27 |
| biased10 | 1.23 | 1.46 | 2.46 | 1.48 | 0.51 |
| biased100 | 1.25 | 1.46 | 2.87 | 1.46 | 0.47 |
| purez | 1.24 | 1.46 | 2.88 | 1.44 | 0.50 |
| em3 | 0.67 | 0.53 | 0.54 | 0.52 | 3.19 |
| helios_p_noxt | 0.87 | 1.01 | 1.44 | 1.00 | 0.51 |
| h2_p_noxt | 1.27 | 1.52 | 1.83 | 1.52 | 0.81 |
| helios_p | -- | -- | 0.67 | -- | -- |
| h2_p | -- | -- | 0.77 | 0.69 | 0.51 |
| hyb | 1.24 | 1.66 | 1.66 | 1.70 | 3.19 |
| hyb_lr2 | 1.24 | 1.66 | 1.66 | 1.70 | 3.19 |
| sd6_lr2 | 1.14 | 1.66 | 1.66 | 1.70 | 0.51 |
| sd6_lr3 | 1.00 | 1.66 | 1.66 | 1.70 | 0.51 |
| si1000_lr2 | 0.71 | 0.87 | 0.86 | 0.88 | 0.27 |
| phen | 9.11 | 9.24 | 9.19 | 9.09 | 1.72 |
| phen_b10 | 8.47 | -- | -- | 6.64 | 1.69 |
| sd6_il | 1.38 | 1.66 | 1.66 | 1.70 | 0.51 |

## r = 3

| noise | XYZ^2 | CSS | XZZX | XY | Honeycomb | ratio | best |
|---|---|---|---|---|---|---|---|
| sd6 | 1.03 (bp_corr, both L_X) | 1.45 (corr, Z) | 1.51 (bm, X) | 1.54 (bm, Z) | 0.41 (bm, Z) | 0.66 | xy |
| si1000 | 0.71 (bp_corr, both L_X) | 0.81 (bm, Z) | 0.80 (bm, X) | 0.81 (bm, X) | 0.21 (bm, Z) | 0.88 | xy |
| biased10 | 1.23 (bm, both L_X) | 1.31 (bm, X) | 2.26 (mwpm, Z) | 1.35 (corr, X) | 0.41 (bm, Z) | 0.55 | xzzx |
| biased100 | 1.30 (bp_full, both L_X) | 1.27 (bm, X) | 2.67 (corr, Z) | 1.63 (bm, Z) | 0.39 (bm, Z) | 0.49 | xzzx |
| purez | 1.32 (bp_full, depth6 L_X) | 1.24 (bm, X) | 2.64 (mwpm, Z) | 1.59 (bm, Z) | 0.43 (bm, Z) | 0.50 | xzzx |
| em3 | 0.63 (seq_soft, both L_Y) | 0.46 (bm, X) | 0.45 (bm, X) | 0.46 (corr, X) | 2.86 (bm, Z) | 0.22 | honeycomb |
| helios_p_noxt | 0.79 (bp_corr, depth6 L_Y) | 0.83 (corr, X) | 1.18 (bm, X) | 0.96 (bm, Z) | 0.39 (bm, Z) | 0.68 | xzzx |
| h2_p_noxt | 1.15 (bp_corr, both L_X) | 1.30 (bm, X) | 1.57 (bm, Z) | 1.38 (bm, X) | 0.67 (bm, Z) | 0.73 | xzzx |
| helios_p | 0.31 (cfe0, depth6 L_X only) | 0.31 (mwpm, X) | 0.56 (mwpm, Z) | 0.28 (mwpm, X) | -- | 0.55 | xzzx |
| h2_p | 0.61 (bp_corr, depth6 L_X only) | 0.73 (bm, X) | 0.70 (mwpm, X) | 1.14 (corr, X) | -- | 0.54 | xy |
| hyb | 1.16 (bp_corr, depth6 L_Y) | 1.45 (corr, Z, as sd6) | 1.51 (bm, X, as sd6) | 1.54 (bm, Z, as sd6) | 2.86 (bm, Z, as em3) | 0.41 | honeycomb |
| hyb_lr2 | 1.28 (seq_erasure, both L_Y) | 1.45 (corr, Z, as sd6) | 1.51 (bm, X, as sd6) | 1.54 (bm, Z, as sd6) | 2.86 (bm, Z, as em3) | 0.45 | honeycomb |
| sd6_lr2 | 1.18 (seq_soft, both L_X) | 1.45 (corr, Z, as sd6) | 1.51 (bm, X, as sd6) | 1.54 (bm, Z, as sd6) | 0.41 (bm, Z, as sd6) | 0.77 | xy |
| sd6_lr3 | 1.04 (bm, depth6 L_Y) | 1.45 (corr, Z, as sd6) | 1.51 (bm, X, as sd6) | 1.54 (bm, Z, as sd6) | 0.41 (bm, Z, as sd6) | 0.67 | xy |
| si1000_lr2 | 0.77 (bp_full, both L_X) | 0.81 (bm, Z, as si1000) | 0.80 (bm, X, as si1000) | 0.81 (bm, X, as si1000) | 0.21 (bm, Z, as si1000) | 0.95 | xy |
| phen | 7.86 (bp_full, both L_X) | 8.33 (bm, Z) | 8.54 (bm, X) | 8.00 (bm, Z) | 1.38 (bm, Z) | 0.92 | xzzx |
| phen_b10 | 8.51 (bp_full, both L_X) | -- | 12.48 (bm, Z) | 7.05 (bm, Z) | 1.44 (bm, Z) | 0.68 | xzzx |
| sd6_il | 1.27 (bp_corr, both L_X) | 1.45 (corr, Z, as sd6) | 1.51 (bm, X, as sd6) | 1.54 (bm, Z, as sd6) | 0.41 (bm, Z, as sd6) | 0.83 | xy |

MWPM only, r = 3:

| noise | XYZ^2 | CSS | XZZX | XY | Honeycomb |
|---|---|---|---|---|---|
| sd6 | 0.83 | 1.37 | -- | 1.41 | 0.39 |
| si1000 | 0.64 | 0.77 | 0.76 | 0.76 | 0.20 |
| biased10 | 0.99 | 1.23 | 2.14 | 1.23 | 0.36 |
| biased100 | 1.00 | 1.20 | 2.60 | 1.21 | 0.36 |
| purez | 1.01 | 1.21 | 2.64 | 1.21 | 0.36 |
| em3 | 0.54 | 0.44 | 0.44 | 0.44 | 2.68 |
| helios_p_noxt | 0.66 | 0.80 | 1.15 | 0.80 | 0.38 |
| h2_p_noxt | 0.98 | 1.22 | 1.47 | 1.20 | 0.60 |
| helios_p | -- | -- | 0.42 | 0.28 | -- |
| h2_p | -- | -- | -- | -- | -- |
| hyb | 0.99 | 1.37 | 0.44 | 1.41 | 2.68 |
| hyb_lr2 | 1.02 | 1.37 | 0.44 | 1.41 | 2.68 |
| sd6_lr2 | 0.90 | 1.37 | -- | 1.41 | 0.39 |
| sd6_lr3 | 0.79 | 1.37 | -- | 1.41 | 0.39 |
| si1000_lr2 | 0.62 | 0.77 | 0.76 | 0.76 | 0.20 |
| phen | 6.99 | 7.33 | 7.45 | 7.47 | 1.25 |
| phen_b10 | 6.56 | -- | 12.00 | 5.43 | 1.25 |
| sd6_il | 1.16 | 1.37 | -- | 1.41 | 0.39 |

## r = 5

| noise | XYZ^2 | CSS | XZZX | XY | Honeycomb | ratio | best |
|---|---|---|---|---|---|---|---|
| sd6 | 0.86 (bp_corr, both L_X) | 1.25 (corr, Z) | 1.30 (bm, X) | 1.21 (corr, X) | 0.34 (bm, X) | 0.66 | xzzx |
| si1000 | 0.65 (bp_corr, both L_X) | 0.73 (bm, X) | 0.70 (corr, X) | 0.72 (bm, Z) | 0.18 (bm, X) | 0.89 | css |
| biased10 | 1.08 (bp_corr, both L_X) | 1.08 (bm, X) | 1.87 (mwpm, Z) | 1.42 (bm, Z) | 0.35 (bm, Z) | 0.58 | xzzx |
| biased100 | 1.18 (bp_full, both L_X) | 1.05 (corr, X) | 2.36 (mwpm, Z) | 1.47 (bm, X) | 0.35 (bm, Z) | 0.50 | xzzx |
| purez | 1.17 (bp_full, both L_Y) | 1.02 (corr, X) | 2.55 (corr, Z) | 1.48 (bm, Z) | 0.35 (bm, X) | 0.46 | xzzx |
| em3 | 0.52 (bp_corr, depth6 L_X) | 0.38 (corr, X) | 0.39 (bm, X) | 0.38 (bm, X) | 2.51 (corr, Z) | 0.21 | honeycomb |
| helios_p_noxt | 0.65 (bp_full, depth6 L_Y) | 0.67 (corr, X) | 0.94 (bm, Z) | 0.79 (bm, Z) | 0.34 (bm, Z) | 0.69 | xzzx |
| h2_p_noxt | 0.94 (bp_corr, both L_X) | 1.03 (corr, X) | 1.28 (bm, Z) | 1.22 (bm, Z) | 0.55 (bm, X) | 0.73 | xzzx |
| helios_p | 0.05 (seq_match, depth6 L_Y) | 0.44 (bm, X) | 0.42 (mwpm, Z) | 0.22 (corr, Z) | -- | 0.12 | css |
| h2_p | 0.82 (tesseract, depth6 L_X only) | 0.54 (bm, X) | 0.58 (bm, X) | -- | -- | 1.41 | xyz2 |
| hyb | 0.96 (seq_soft, both L_Y) | 1.25 (corr, Z, as sd6) | 1.30 (bm, X, as sd6) | 1.21 (corr, X, as sd6) | 2.51 (corr, Z, as em3) | 0.38 | honeycomb |
| hyb_lr2 | 1.03 (seq_soft, depth6 L_Y) | 1.25 (corr, Z, as sd6) | 1.30 (bm, X, as sd6) | 1.21 (corr, X, as sd6) | 2.51 (corr, Z, as em3) | 0.41 | honeycomb |
| sd6_lr2 | 0.94 (bp_corr, both L_X) | 1.25 (corr, Z, as sd6) | 1.30 (bm, X, as sd6) | 1.21 (corr, X, as sd6) | 0.34 (bm, X, as sd6) | 0.73 | xzzx |
| sd6_lr3 | 0.87 (bm, both L_X) | 1.25 (corr, Z, as sd6) | 1.30 (bm, X, as sd6) | 1.21 (corr, X, as sd6) | 0.34 (bm, X, as sd6) | 0.67 | xzzx |
| si1000_lr2 | 0.67 (bm, both L_X) | 0.73 (bm, X, as si1000) | 0.70 (corr, X, as si1000) | 0.72 (bm, Z, as si1000) | 0.18 (bm, X, as si1000) | 0.92 | css |
| phen | 6.46 (bp_corr, depth6 L_X) | 6.96 (bm, X) | 6.87 (bm, X) | 6.76 (bm, Z) | 1.05 (bm, Z) | 0.93 | css |
| phen_b10 | 7.52 (bp_full, both L_X) | -- | 9.62 (bm, Z) | 6.77 (bm, Z) | 1.13 (bm, Z) | 0.78 | xzzx |
| sd6_il | 1.05 (bp_full, both L_X) | 1.25 (corr, Z, as sd6) | 1.30 (bm, X, as sd6) | 1.21 (corr, X, as sd6) | 0.34 (bm, X, as sd6) | 0.81 | xzzx |

MWPM only, r = 5:

| noise | XYZ^2 | CSS | XZZX | XY | Honeycomb |
|---|---|---|---|---|---|
| sd6 | 0.69 | 1.15 | -- | 1.15 | 0.30 |
| si1000 | 0.54 | 0.67 | 0.66 | 0.67 | 0.15 |
| biased10 | 0.79 | 1.01 | 1.87 | 1.01 | 0.29 |
| biased100 | 0.79 | 1.01 | 2.36 | 0.99 | 0.28 |
| purez | 0.80 | 1.01 | 2.45 | 1.00 | 0.28 |
| em3 | 0.43 | 0.36 | 0.36 | 0.36 | 2.28 |
| helios_p_noxt | 0.50 | 0.64 | 0.90 | 0.64 | 0.29 |
| h2_p_noxt | 0.76 | 0.97 | 1.19 | 0.98 | 0.46 |
| helios_p | -- | -- | 0.42 | -- | -- |
| h2_p | -- | -- | -- | -- | -- |
| hyb | 0.75 | 1.15 | 0.36 | 1.15 | 2.28 |
| hyb_lr2 | 0.81 | 1.15 | 0.36 | 1.15 | 2.28 |
| sd6_lr2 | 0.70 | 1.15 | -- | 1.15 | 0.30 |
| sd6_lr3 | 0.63 | 1.15 | -- | 1.15 | 0.30 |
| si1000_lr2 | 0.52 | 0.67 | 0.66 | 0.67 | 0.15 |
| phen | 5.50 | 6.06 | 6.01 | -- | 0.93 |
| phen_b10 | 5.22 | -- | 9.24 | 4.58 | 0.92 |
| sd6_il | 0.94 | 1.15 | -- | 1.15 | 0.30 |

## r = 10

| noise | XYZ^2 | CSS | XZZX | XY | Honeycomb | ratio | best |
|---|---|---|---|---|---|---|---|
| sd6 | 0.70 (bp_corr, depth6 L_Y) | 1.10 (corr, Z) | 1.05 (corr, Z) | 1.09 (bm, Z) | 0.29 (bm, Z) | 0.64 | css |
| si1000 | 0.57 (bm, both L_X) | 0.66 (bm, Z) | 0.66 (bm, Z) | 0.67 (bm, Z) | 0.13 (corr, X) | 0.85 | xy |
| biased10 | 0.98 (bp_full, both L_Y) | 0.91 (corr, X) | 1.62 (corr, Z) | 1.37 (bm, X) | 0.28 (bm, X) | 0.60 | xzzx |
| biased100 | 1.03 (bm, both L_X) | 0.91 (corr, X) | 2.18 (mwpm, Z) | 0.84 (mwpm, X) | 0.30 (bm, Z) | 0.47 | xzzx |
| purez | 1.12 (bm, depth6 L_X) | 0.87 (corr, X) | 2.41 (corr, Z) | 1.30 (bm, X) | 0.28 (bm, X) | 0.46 | xzzx |
| em3 | 0.44 (seq_soft, depth6 L_X) | 0.34 (bm, X) | 0.33 (bm, X) | 0.33 (corr, X) | 2.47 (corr, X) | 0.18 | honeycomb |
| helios_p_noxt | 0.55 (bp_full, both L_X) | 0.56 (corr, X) | 0.75 (bm, Z) | 0.69 (bm, Z) | 0.24 (mwpm, X) | 0.73 | xzzx |
| h2_p_noxt | 0.82 (bp_corr, both L_X) | 0.91 (corr, X) | 1.08 (bm, Z) | 1.03 (bm, X) | 0.47 (bm, X) | 0.76 | xzzx |
| helios_p | 0.05 (seq_match, both L_Y) | -- | 0.29 (mwpm, Z) | 0.15 (corr, Z) | -- | 0.17 | xzzx |
| h2_p | 0.31 (seq_erasure, depth6 L_Y) | -- | -- | 0.49 (bm, X) | 0.25 (mwpm, X) | 0.64 | xy |
| hyb | 0.78 (bm, both L_X) | 1.10 (corr, Z, as sd6) | 1.05 (corr, Z, as sd6) | 1.09 (bm, Z, as sd6) | 2.47 (corr, X, as em3) | 0.31 | honeycomb |
| hyb_lr2 | 0.87 (bp_full, depth6 L_Y) | 1.10 (corr, Z, as sd6) | 1.05 (corr, Z, as sd6) | 1.09 (bm, Z, as sd6) | 2.47 (corr, X, as em3) | 0.35 | honeycomb |
| sd6_lr2 | 0.80 (bm, depth6 L_Y) | 1.10 (corr, Z, as sd6) | 1.05 (corr, Z, as sd6) | 1.09 (bm, Z, as sd6) | 0.29 (bm, Z, as sd6) | 0.73 | css |
| sd6_lr3 | 0.72 (bm, depth6 L_Y) | 1.10 (corr, Z, as sd6) | 1.05 (corr, Z, as sd6) | 1.09 (bm, Z, as sd6) | 0.29 (bm, Z, as sd6) | 0.65 | css |
| si1000_lr2 | 0.59 (bm, both L_X) | 0.66 (bm, Z, as si1000) | 0.66 (bm, Z, as si1000) | 0.67 (bm, Z, as si1000) | 0.13 (corr, X, as si1000) | 0.89 | xy |
| phen | 5.48 (bm, both L_X) | 5.95 (bm, X) | 6.07 (bm, Z) | 6.18 (bm, X) | 0.95 (bm, X) | 0.89 | xy |
| phen_b10 | 6.84 (bp_corr, depth6 L_X) | -- | -- | 6.25 (bm, Z) | 0.96 (bm, Z) | 1.10 | xyz2 |
| sd6_il | 0.90 (seq_soft, depth6 L_Y) | 1.10 (corr, Z, as sd6) | 1.05 (corr, Z, as sd6) | 1.09 (bm, Z, as sd6) | 0.29 (bm, Z, as sd6) | 0.82 | css |

MWPM only, r = 10:

| noise | XYZ^2 | CSS | XZZX | XY | Honeycomb |
|---|---|---|---|---|---|
| sd6 | 0.56 | 0.98 | 0.95 | -- | 0.23 |
| si1000 | 0.46 | 0.59 | 0.59 | 0.59 | 0.12 |
| biased10 | 0.63 | 0.85 | 1.57 | 0.87 | 0.23 |
| biased100 | 0.63 | 0.87 | 2.18 | 0.84 | 0.22 |
| purez | 0.64 | 0.85 | 2.36 | 0.84 | 0.23 |
| em3 | 0.35 | 0.30 | 0.30 | 0.30 | -- |
| helios_p_noxt | 0.39 | 0.52 | 0.71 | 0.52 | -- |
| h2_p_noxt | 0.59 | 0.79 | 0.95 | 0.80 | -- |
| helios_p | -- | -- | 0.16 | 0.12 | -- |
| h2_p | -- | -- | -- | -- | 0.19 |
| hyb | 0.66 | 0.98 | 0.95 | 0.30 | 0.23 |
| hyb_lr2 | 0.66 | 0.98 | 0.95 | 0.30 | 0.23 |
| sd6_lr2 | 0.57 | 0.98 | 0.95 | -- | 0.23 |
| sd6_lr3 | 0.48 | 0.98 | 0.95 | -- | 0.23 |
| si1000_lr2 | 0.43 | 0.59 | 0.59 | 0.59 | 0.12 |
| phen | 4.45 | 5.10 | 5.12 | 5.11 | 0.68 |
| phen_b10 | 4.24 | -- | -- | 3.87 | 0.71 |
| sd6_il | 0.72 | 0.98 | 0.95 | -- | 0.23 |
