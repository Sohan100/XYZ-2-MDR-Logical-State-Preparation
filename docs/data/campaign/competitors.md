# XYZ^2 against competitor codes

Thresholds in % of p. A competitor's value is its best decoder in its weaker basis; XYZ^2's is its best decoder (Logical-X memory). Ratio: XYZ^2 over the best competitor. Only finite-size fits over d >= 11 count: thresholds that drift, crossings of two distances only, and series that stop below d = 11 (honeycomb: d = 12) are left out.

## r = d

| noise | XYZ^2 | CSS | XZZX | XY | Honeycomb | ratio | best |
|---|---|---|---|---|---|---|---|
| sd6 | 0.54 (bp_corr) | 0.81 (corr, X) | 0.81 (bm, Z) | 0.81 (bm, X) | 0.22 (bm, Z) | 0.67 | xzzx |
| si1000 | 0.47 (bp_corr) | 0.52 (bm, Z) | 0.51 (bm, Z) | 0.51 (bm, Z) | 0.12 (bm, Z) | 0.90 | css |
| biased10 | 0.76 (bp_corr) | 0.64 (corr, X) | 1.17 (corr, Z) | 1.02 (bm, Z) | 0.23 (bm, X) | 0.64 | xzzx |
| biased100 | 0.84 (bp_full) | 0.62 (bm, X) | 1.65 (corr, Z) | 1.05 (bm, Z) | 0.23 (bm, X) | 0.51 | xzzx |
| purez | 0.86 (bp_full) | 0.62 (corr, X) | 2.06 (corr, Z) | 1.05 (bm, Z) | 0.24 (bm, X) | 0.42 | xzzx |
| em3 | 0.33 (seq_soft) | 0.24 (corr, X) | 0.25 (corr, X) | 0.24 (corr, X) | 1.98 (corr, X) | 0.17 | honeycomb |
| helios_p_noxt | 0.43 (bp_corr) | 0.40 (corr, X) | 0.54 (corr, Z) | 0.53 (bm, Z) | 0.23 (bm, X) | 0.81 | xzzx |
| h2_p_noxt | 0.62 (bm) | 0.64 (corr, X) | 0.75 (corr, Z) | 0.77 (bm, X) | 0.36 (bm, X) | 0.81 | xy |
| helios_p | 0.20 (bm) | 0.28 (corr, X) | 0.33 (bm, X) | -- | -- | 0.60 | xzzx |
| h2_p | 0.44 (bp_corr) | -- | 0.41 (bm, X) | -- | 0.17 (bm, X) | 1.08 | xyz2 |

MWPM only, r = d:

| noise | XYZ^2 | CSS | XZZX | XY | Honeycomb |
|---|---|---|---|---|---|
| sd6 | 0.43 | 0.69 | 0.70 | 0.69 | 0.17 |
| si1000 | 0.37 | 0.44 | 0.44 | 0.44 | 0.08 |
| biased10 | 0.44 | 0.61 | 1.10 | 0.60 | 0.16 |
| biased100 | 0.45 | 0.59 | 1.61 | 0.60 | 0.16 |
| purez | 0.45 | 0.60 | 2.05 | 0.59 | 0.16 |
| em3 | 0.25 | 0.22 | 0.22 | 0.22 | 1.60 |
| helios_p_noxt | 0.29 | 0.37 | 0.49 | 0.37 | 0.16 |
| h2_p_noxt | 0.46 | 0.56 | 0.67 | 0.56 | 0.25 |
| helios_p | -- | -- | -- | -- | -- |
| h2_p | -- | -- | -- | -- | -- |

## r = 1

| noise | XYZ^2 | CSS | XZZX | XY | Honeycomb | ratio | best |
|---|---|---|---|---|---|---|---|
| sd6 | 1.96 (cfe_tn) | 2.41 (bm, X) | 2.42 (corr, Z) | 2.38 (bm, Z) | 0.83 (mwpm, Z) | 0.81 | xzzx |
| si1000 | 1.06 (cfe_tn) | 1.08 (bm, X) | 1.06 (bm, Z) | 1.06 (bm, X) | 0.40 (bm, Z) | 0.98 | css |
| biased10 | 1.97 (cfe_tn) | 2.19 (bm, X) | 3.18 (mwpm, Z) | 2.16 (mwpm, X) | 0.73 (corr, Z) | 0.62 | xzzx |
| biased100 | 1.97 (tnml) | 2.14 (bm, X) | 3.45 (mwpm, Z) | 2.12 (bm, X) | 0.72 (corr, Z) | 0.57 | xzzx |
| purez | 1.99 (bm) | 2.25 (bm, X) | 3.48 (corr, Z) | 2.14 (corr, Z) | 0.71 (mwpm, Z) | 0.57 | xzzx |
| em3 | 1.07 (cfe_tn) | 0.82 (bm, X) | 0.81 (corr, X) | 0.79 (bm, X) | 3.50 (mwpm, Z) | 0.30 | honeycomb |
| helios_p_noxt | 1.54 (tnml) | 1.61 (mwpm, X) | 2.37 (mwpm, Z) | 1.61 (mwpm, X) | 0.73 (corr, Z) | 0.65 | xzzx |
| h2_p_noxt | 2.24 (tnml) | 2.44 (mwpm, X) | 3.03 (bm, X) | 2.40 (mwpm, X) | 1.22 (corr, Z) | 0.74 | xzzx |
| helios_p | 0.33 (cfe) | 0.53 (mwpm, Z) | 0.59 (corr, X) | 0.56 (corr, Z) | -- | 0.56 | xzzx |
| h2_p | 1.16 (cfe_tn) | 1.24 (corr, X) | 1.39 (corr, Z) | 1.27 (bm, Z) | 0.89 (mwpm, Z) | 0.83 | xzzx |

MWPM only, r = 1:

| noise | XYZ^2 | CSS | XZZX | XY | Honeycomb |
|---|---|---|---|---|---|
| sd6 | 1.81 | 2.37 | 2.34 | 2.34 | 0.83 |
| si1000 | 0.97 | 0.99 | 1.03 | 1.01 | 0.38 |
| biased10 | 1.83 | 2.15 | 3.18 | 2.10 | 0.73 |
| biased100 | 1.87 | 2.11 | 3.44 | 2.06 | 0.70 |
| purez | 1.88 | 2.09 | 3.37 | 2.05 | 0.71 |
| em3 | 0.98 | 0.78 | 0.78 | 0.78 | 3.50 |
| helios_p_noxt | 1.45 | 1.61 | 2.37 | 1.61 | 0.71 |
| h2_p_noxt | 2.14 | 2.44 | 2.95 | 2.29 | 1.21 |
| helios_p | -- | 0.53 | -- | 0.36 | -- |
| h2_p | -- | -- | -- | -- | 0.89 |

## r = 2

| noise | XYZ^2 | CSS | XZZX | XY | Honeycomb | ratio | best |
|---|---|---|---|---|---|---|---|
| sd6 | 1.30 (seq_erasure) | 1.75 (bm, X) | 1.74 (bm, Z) | 1.75 (corr, Z) | 0.53 (mwpm, X) | 0.74 | css |
| si1000 | 0.85 (bm) | 0.90 (bm, Z) | 0.88 (bm, Z) | 0.89 (bm, X) | 0.28 (bm, X) | 0.94 | css |
| biased10 | 1.41 (bp_full) | 1.50 (bm, X) | 2.49 (mwpm, X) | 1.76 (bm, Z) | 0.51 (mwpm, Z) | 0.56 | xzzx |
| biased100 | 1.50 (bp_corr) | 1.52 (bm, X) | 2.87 (mwpm, Z) | 1.69 (bm, X) | 0.49 (corr, Z) | 0.52 | xzzx |
| purez | 1.46 (tesseract) | 1.50 (bm, X) | 2.88 (mwpm, Z) | 1.69 (bm, X) | 0.50 (corr, Z) | 0.51 | xzzx |
| em3 | 0.73 (bm) | 0.55 (bm, X) | 0.54 (corr, X) | 0.56 (bm, X) | 3.39 (bm, X) | 0.22 | honeycomb |
| helios_p_noxt | 1.00 (tesseract) | 1.04 (bm, X) | 1.57 (bm, Z) | 1.12 (bm, Z) | 0.53 (corr, Z) | 0.64 | xzzx |
| h2_p_noxt | 1.47 (bp_corr) | 1.58 (bm, X) | 1.89 (bm, Z) | 1.61 (bm, X) | 0.88 (corr, Z) | 0.78 | xzzx |
| helios_p | -- | 0.78 (corr, X) | 0.67 (mwpm, Z) | 0.55 (corr, Z) | -- | -- | nan |
| h2_p | 0.83 (cfe0) | 1.63 (bm, X) | 0.83 (corr, X) | 0.83 (corr, X) | 0.57 (corr, Z) | 0.51 | css |

MWPM only, r = 2:

| noise | XYZ^2 | CSS | XZZX | XY | Honeycomb |
|---|---|---|---|---|---|
| sd6 | 1.19 | 1.66 | 1.66 | 1.70 | 0.51 |
| si1000 | 0.79 | 0.87 | 0.86 | 0.88 | 0.27 |
| biased10 | 1.23 | 1.46 | 2.46 | 1.48 | 0.51 |
| biased100 | 1.26 | 1.46 | 2.87 | 1.46 | 0.47 |
| purez | 1.25 | 1.46 | 2.88 | 1.44 | 0.50 |
| em3 | 0.66 | 0.53 | 0.54 | 0.52 | 3.19 |
| helios_p_noxt | 0.89 | 1.01 | 1.44 | 1.00 | 0.51 |
| h2_p_noxt | 1.33 | 1.52 | 1.83 | 1.52 | 0.81 |
| helios_p | -- | -- | 0.67 | -- | -- |
| h2_p | -- | -- | 0.77 | 0.69 | 0.51 |

## r = 3

| noise | XYZ^2 | CSS | XZZX | XY | Honeycomb | ratio | best |
|---|---|---|---|---|---|---|---|
| sd6 | 1.09 (bp_full) | 1.45 (corr, Z) | 1.51 (bm, X) | 1.54 (bm, Z) | 0.41 (bm, Z) | 0.71 | xy |
| si1000 | 0.76 (bm) | 0.81 (bm, Z) | 0.80 (bm, X) | 0.81 (bm, X) | 0.21 (bm, Z) | 0.94 | xy |
| biased10 | 1.24 (bm) | 1.31 (bm, X) | 2.26 (mwpm, Z) | 1.35 (corr, X) | 0.41 (bm, Z) | 0.55 | xzzx |
| biased100 | 1.28 (bp_corr) | 1.27 (bm, X) | 2.67 (corr, Z) | 1.63 (bm, Z) | 0.39 (bm, Z) | 0.48 | xzzx |
| purez | 1.32 (bp_full) | 1.24 (bm, X) | 2.64 (mwpm, Z) | 1.59 (bm, Z) | 0.43 (bm, Z) | 0.50 | xzzx |
| em3 | 0.62 (bm) | 0.46 (bm, X) | 0.45 (bm, X) | 0.46 (corr, X) | 2.86 (bm, Z) | 0.22 | honeycomb |
| helios_p_noxt | 0.83 (bp_full) | 0.83 (corr, X) | 1.18 (bm, X) | 0.96 (bm, Z) | 0.39 (bm, Z) | 0.70 | xzzx |
| h2_p_noxt | 1.64 (tesseract) | 1.30 (bm, X) | 1.57 (bm, Z) | 1.38 (bm, X) | 0.67 (bm, Z) | 1.05 | xyz2 |
| helios_p | 0.31 (cfe0) | 0.31 (mwpm, X) | 0.56 (mwpm, Z) | 0.28 (mwpm, X) | -- | 0.55 | xzzx |
| h2_p | 0.61 (bp_corr) | 0.73 (bm, X) | 0.70 (mwpm, X) | 1.14 (corr, X) | -- | 0.54 | xy |

MWPM only, r = 3:

| noise | XYZ^2 | CSS | XZZX | XY | Honeycomb |
|---|---|---|---|---|---|
| sd6 | 0.95 | 1.37 | -- | 1.41 | 0.39 |
| si1000 | 0.67 | 0.77 | 0.76 | 0.76 | 0.20 |
| biased10 | 0.98 | 1.23 | 2.14 | 1.23 | 0.36 |
| biased100 | 0.99 | 1.20 | 2.60 | 1.21 | 0.36 |
| purez | 1.00 | 1.21 | 2.64 | 1.21 | 0.36 |
| em3 | 0.54 | 0.44 | 0.44 | 0.44 | 2.68 |
| helios_p_noxt | 0.68 | 0.80 | 1.15 | 0.80 | 0.38 |
| h2_p_noxt | 1.04 | 1.22 | 1.47 | 1.20 | 0.60 |
| helios_p | -- | -- | 0.42 | 0.28 | -- |
| h2_p | -- | -- | -- | -- | -- |

## r = 5

| noise | XYZ^2 | CSS | XZZX | XY | Honeycomb | ratio | best |
|---|---|---|---|---|---|---|---|
| sd6 | 0.90 (bp_corr) | 1.25 (corr, Z) | 1.30 (bm, X) | 1.21 (corr, X) | 0.34 (bm, X) | 0.69 | xzzx |
| si1000 | 0.68 (bp_corr) | 0.73 (bm, X) | 0.70 (corr, X) | 0.72 (bm, Z) | 0.18 (bm, X) | 0.93 | css |
| biased10 | 1.12 (bm) | 1.08 (bm, X) | 1.87 (mwpm, Z) | 1.42 (bm, Z) | 0.35 (bm, Z) | 0.60 | xzzx |
| biased100 | 1.15 (bp_corr) | 1.05 (corr, X) | 2.36 (mwpm, Z) | 1.47 (bm, X) | 0.35 (bm, Z) | 0.49 | xzzx |
| purez | 1.17 (bp_full) | 1.02 (corr, X) | 2.55 (corr, Z) | 1.48 (bm, Z) | 0.35 (bm, X) | 0.46 | xzzx |
| em3 | 0.52 (bp_corr) | 0.38 (corr, X) | 0.39 (bm, X) | 0.38 (bm, X) | 2.51 (corr, Z) | 0.21 | honeycomb |
| helios_p_noxt | 0.67 (bp_full) | 0.67 (corr, X) | 0.94 (bm, Z) | 0.79 (bm, Z) | 0.34 (bm, Z) | 0.72 | xzzx |
| h2_p_noxt | 1.01 (bp_full) | 1.03 (corr, X) | 1.28 (bm, Z) | 1.22 (bm, Z) | 0.55 (bm, X) | 0.78 | xzzx |
| helios_p | 0.26 (seq_soft) | 0.44 (bm, X) | 0.42 (mwpm, Z) | 0.22 (corr, Z) | -- | 0.58 | css |
| h2_p | 0.82 (tesseract) | 0.54 (bm, X) | 0.58 (bm, X) | -- | -- | 1.41 | xyz2 |

MWPM only, r = 5:

| noise | XYZ^2 | CSS | XZZX | XY | Honeycomb |
|---|---|---|---|---|---|
| sd6 | 0.76 | 1.15 | -- | 1.15 | 0.30 |
| si1000 | 0.58 | 0.67 | 0.66 | 0.67 | 0.15 |
| biased10 | 0.79 | 1.01 | 1.87 | 1.01 | 0.29 |
| biased100 | 0.79 | 1.01 | 2.36 | 0.99 | 0.28 |
| purez | 0.80 | 1.01 | 2.45 | 1.00 | 0.28 |
| em3 | 0.43 | 0.36 | 0.36 | 0.36 | 2.28 |
| helios_p_noxt | 0.53 | 0.64 | 0.90 | 0.64 | 0.29 |
| h2_p_noxt | 0.81 | 0.97 | 1.19 | 0.98 | 0.46 |
| helios_p | -- | -- | 0.42 | -- | -- |
| h2_p | -- | -- | -- | -- | -- |

## r = 10

| noise | XYZ^2 | CSS | XZZX | XY | Honeycomb | ratio | best |
|---|---|---|---|---|---|---|---|
| sd6 | 0.75 (bp_corr) | 1.10 (corr, Z) | 1.05 (corr, Z) | 1.09 (bm, Z) | 0.29 (bm, Z) | 0.68 | css |
| si1000 | 0.60 (bm) | 0.66 (bm, Z) | 0.66 (bm, Z) | 0.67 (bm, Z) | 0.13 (corr, X) | 0.90 | xy |
| biased10 | 1.00 (bp_corr) | 0.91 (corr, X) | 1.62 (corr, Z) | 1.37 (bm, X) | 0.28 (bm, X) | 0.62 | xzzx |
| biased100 | 1.04 (bm) | 0.91 (corr, X) | 2.18 (mwpm, Z) | 0.84 (mwpm, X) | 0.30 (bm, Z) | 0.48 | xzzx |
| purez | 1.12 (bm) | 0.87 (corr, X) | 2.41 (corr, Z) | 1.30 (bm, X) | 0.28 (bm, X) | 0.46 | xzzx |
| em3 | 0.44 (seq_soft) | 0.34 (bm, X) | 0.33 (bm, X) | 0.33 (corr, X) | 2.47 (corr, X) | 0.18 | honeycomb |
| helios_p_noxt | 0.58 (bp_full) | 0.56 (corr, X) | 0.75 (bm, Z) | 0.69 (bm, Z) | 0.24 (mwpm, X) | 0.77 | xzzx |
| h2_p_noxt | 0.85 (bp_corr) | 0.91 (corr, X) | 1.08 (bm, Z) | 1.03 (bm, X) | 0.47 (bm, X) | 0.79 | xzzx |
| helios_p | -- | -- | 0.29 (mwpm, Z) | 0.15 (corr, Z) | -- | -- | nan |
| h2_p | 0.41 (seq_soft) | -- | -- | 0.49 (bm, X) | 0.25 (mwpm, X) | 0.83 | xy |

MWPM only, r = 10:

| noise | XYZ^2 | CSS | XZZX | XY | Honeycomb |
|---|---|---|---|---|---|
| sd6 | 0.61 | 0.98 | 0.95 | -- | 0.23 |
| si1000 | 0.49 | 0.59 | 0.59 | 0.59 | 0.12 |
| biased10 | 0.63 | 0.85 | 1.57 | 0.87 | 0.23 |
| biased100 | 0.65 | 0.87 | 2.18 | 0.84 | 0.22 |
| purez | 0.64 | 0.85 | 2.36 | 0.84 | 0.23 |
| em3 | 0.35 | 0.30 | 0.30 | 0.30 | -- |
| helios_p_noxt | 0.42 | 0.52 | 0.71 | 0.52 | -- |
| h2_p_noxt | 0.66 | 0.79 | 0.95 | 0.80 | -- |
| helios_p | -- | -- | 0.16 | 0.12 | -- |
| h2_p | -- | -- | -- | -- | 0.19 |
