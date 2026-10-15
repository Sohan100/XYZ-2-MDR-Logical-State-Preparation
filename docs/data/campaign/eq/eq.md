# Equal-qubit comparison below threshold

Error per round epsilon (r = 3d), 68% Wilson intervals; a point without logical errors is a 95% upper bound (3/N per shot). One decoder per code and noise model (decoders.csv), and MWPM alone. Weaker basis per code, noise model and p. Lambda = epsilon(d)/epsilon(d+2) from d >= 5 and points with >= 30 errors.

## Decoders

| code | noise | decoder | measured at every point |
|---|---|---|---|
| CSS | biased100 | corr | corr mwpm |
| CSS | purez | corr | corr mwpm |
| CSS | sd6 | corr | corr mwpm |
| XYZ^2 | biased100 | corr | corr mwpm |
| XYZ^2 | purez | corr | corr mwpm |
| XYZ^2 | sd6 | corr | corr mwpm |
| XZZX | biased100 | corr | corr mwpm |
| XZZX | purez | corr | corr mwpm |

## Equal-qubit pairs (chosen decoder)

| noise | p | pair (qubits) | XYZ^2 eps | other | other eps | ratio XYZ^2/other |
|---|---|---|---|---|---|---|
| sd6 | 0.001 | 5/7 (99/97) | 2.9e-5 [2.8e-5, 3.1e-5] | CSS (corr, Z) | 2.7e-6 [2.5e-6, 2.8e-6] | 11.09 [9.90, 12.43] |
| sd6 | 0.001 | 9/13 (323/337) | 2.5e-7 [2.3e-7, 2.6e-7] | CSS (corr, Z) | 3.6e-10 [1.8e-10, 7.1e-10] | 693.43 [328.59, 1463.39] |
| sd6 | 0.001 | 11/15 (483/449) | 2.1e-8 [1.8e-8, 2.4e-8] | CSS (corr, Z) | < 7.1e-10 | -- |
| sd6 | 0.002 | 5/7 (99/97) | 3.9e-4 [3.7e-4, 4.1e-4] | CSS (corr, Z) | 4.6e-5 [4.3e-5, 4.8e-5] | 8.54 [7.65, 9.54] |
| sd6 | 0.002 | 9/13 (323/337) | 2.1e-5 [1.9e-5, 2.2e-5] | CSS (corr, Z) | 1.7e-7 [1.6e-7, 1.8e-7] | 118.34 [105.53, 132.70] |
| sd6 | 0.002 | 11/15 (483/449) | 6.2e-6 [5.9e-6, 6.6e-6] | CSS (corr, Z) | 1.9e-8 [1.6e-8, 2.3e-8] | 332.13 [261.59, 421.68] |
| sd6 | 0.003 | 5/7 (99/97) | 1.8e-3 [1.7e-3, 1.8e-3] | CSS (corr, Z) | 2.7e-4 [2.6e-4, 2.8e-4] | 6.53 [5.95, 7.16] |
| sd6 | 0.003 | 9/13 (323/337) | 3.3e-4 [3.1e-4, 3.5e-4] | CSS (corr, Z) | 4.3e-6 [4.1e-6, 4.6e-6] | 76.11 [67.86, 85.35] |
| sd6 | 0.003 | 11/15 (483/449) | 1.3e-4 [1.3e-4, 1.4e-4] | CSS (corr, Z) | 1.0e-6 [9.7e-7, 1.1e-6] | 129.29 [115.35, 144.91] |
| biased100 | 0.001 | 5/7 (99/97) | 4.1e-6 [3.8e-6, 4.3e-6] | CSS (corr, X) | 2.0e-5 [1.9e-5, 2.1e-5] | 0.21 [0.18, 0.23] |
| biased100 | 0.001 | 5/7 (99/97) | 4.1e-6 [3.8e-6, 4.3e-6] | XZZX (corr, X) | 3.9e-7 [3.7e-7, 4.2e-7] | 10.39 [9.26, 11.65] |
| biased100 | 0.001 | 9/13 (323/337) | 1.7e-9 [1.1e-9, 2.4e-9] | CSS (corr, X) | 3.5e-8 [3.3e-8, 3.8e-8] | 0.05 [0.03, 0.07] |
| biased100 | 0.001 | 9/13 (323/337) | 1.7e-9 [1.1e-9, 2.4e-9] | XZZX (corr, X) | < 9.5e-10 | -- |
| biased100 | 0.001 | 11/15 (483/449) | 4.1e-10 [1.6e-10, 1.1e-9] | CSS (corr, X) | 3.3e-9 [2.6e-9, 4.3e-9] | 0.12 [0.04, 0.41] |
| biased100 | 0.001 | 11/15 (483/449) | 4.1e-10 [1.6e-10, 1.1e-9] | XZZX (corr, X) | < 9.5e-10 | -- |
| biased100 | 0.002 | 5/7 (99/97) | 6.9e-5 [6.5e-5, 7.3e-5] | CSS (corr, X) | 2.7e-4 [2.6e-4, 2.9e-4] | 0.25 [0.23, 0.28] |
| biased100 | 0.002 | 5/7 (99/97) | 6.9e-5 [6.5e-5, 7.3e-5] | XZZX (corr, X) | 6.5e-6 [6.2e-6, 6.9e-6] | 10.49 [9.36, 11.76] |
| biased100 | 0.002 | 9/13 (323/337) | 4.4e-7 [4.2e-7, 4.7e-7] | CSS (corr, X) | 5.9e-6 [5.5e-6, 6.2e-6] | 0.08 [0.07, 0.08] |
| biased100 | 0.002 | 9/13 (323/337) | 4.4e-7 [4.2e-7, 4.7e-7] | XZZX (corr, X) | 2.9e-9 [2.1e-9, 3.9e-9] | 154.05 [107.90, 219.96] |
| biased100 | 0.002 | 11/15 (483/449) | 4.1e-8 [3.5e-8, 4.7e-8] | CSS (corr, X) | 1.4e-6 [1.4e-6, 1.5e-6] | 0.03 [0.02, 0.04] |
| biased100 | 0.002 | 11/15 (483/449) | 4.1e-8 [3.5e-8, 4.7e-8] | XZZX (corr, X) | < 1.1e-9 | -- |
| biased100 | 0.003 | 5/7 (99/97) | 3.5e-4 [3.4e-4, 3.7e-4] | CSS (corr, X) | 1.2e-3 [1.2e-3, 1.3e-3] | 0.29 [0.26, 0.31] |
| biased100 | 0.003 | 5/7 (99/97) | 3.5e-4 [3.4e-4, 3.7e-4] | XZZX (corr, X) | 3.3e-5 [3.1e-5, 3.5e-5] | 10.65 [9.61, 11.81] |
| biased100 | 0.003 | 9/13 (323/337) | 1.1e-5 [1.0e-5, 1.2e-5] | CSS (corr, X) | 10.0e-5 [9.4e-5, 1.1e-4] | 0.11 [0.10, 0.12] |
| biased100 | 0.003 | 9/13 (323/337) | 1.1e-5 [1.0e-5, 1.2e-5] | XZZX (corr, X) | 6.4e-8 [5.9e-8, 6.9e-8] | 169.96 [148.25, 194.85] |
| biased100 | 0.003 | 11/15 (483/449) | 2.3e-6 [2.1e-6, 2.4e-6] | CSS (corr, X) | 4.3e-5 [4.1e-5, 4.6e-5] | 0.05 [0.05, 0.06] |
| biased100 | 0.003 | 11/15 (483/449) | 2.3e-6 [2.1e-6, 2.4e-6] | XZZX (corr, X) | 7.3e-9 [5.5e-9, 9.7e-9] | 308.80 [219.03, 435.38] |
| purez | 0.001 | 5/7 (99/97) | 4.1e-6 [3.9e-6, 4.3e-6] | CSS (corr, X) | 1.8e-5 [1.7e-5, 1.9e-5] | 0.23 [0.20, 0.26] |
| purez | 0.001 | 5/7 (99/97) | 4.1e-6 [3.9e-6, 4.3e-6] | XZZX (corr, X) | 2.4e-7 [2.2e-7, 2.5e-7] | 17.26 [15.39, 19.36] |
| purez | 0.001 | 9/13 (323/337) | 2.4e-9 [1.7e-9, 3.3e-9] | CSS (corr, X) | 4.0e-8 [3.8e-8, 4.3e-8] | 0.06 [0.04, 0.09] |
| purez | 0.001 | 9/13 (323/337) | 2.4e-9 [1.7e-9, 3.3e-9] | XZZX (corr, X) | < 6.3e-10 | -- |
| purez | 0.001 | 11/15 (483/449) | < 3.3e-9 | CSS (corr, X) | 8.3e-9 [7.0e-9, 9.8e-9] | -- |
| purez | 0.001 | 11/15 (483/449) | < 3.3e-9 | XZZX (corr, X) | < 6.3e-10 | -- |
| purez | 0.002 | 5/7 (99/97) | 5.9e-5 [5.6e-5, 6.3e-5] | CSS (corr, X) | 2.9e-4 [2.7e-4, 3.0e-4] | 0.21 [0.19, 0.23] |
| purez | 0.002 | 5/7 (99/97) | 5.9e-5 [5.6e-5, 6.3e-5] | XZZX (corr, X) | 3.7e-6 [3.5e-6, 3.9e-6] | 16.09 [14.37, 18.01] |
| purez | 0.002 | 9/13 (323/337) | 4.1e-7 [3.9e-7, 4.3e-7] | CSS (corr, X) | 5.4e-6 [5.1e-6, 5.8e-6] | 0.08 [0.07, 0.08] |
| purez | 0.002 | 9/13 (323/337) | 4.1e-7 [3.9e-7, 4.3e-7] | XZZX (corr, X) | 2.7e-9 [2.0e-9, 3.6e-9] | 154.31 [108.07, 220.32] |
| purez | 0.002 | 11/15 (483/449) | 2.8e-8 [2.4e-8, 3.4e-8] | CSS (corr, X) | 1.6e-6 [1.5e-6, 1.7e-6] | 0.02 [0.01, 0.02] |
| purez | 0.002 | 11/15 (483/449) | 2.8e-8 [2.4e-8, 3.4e-8] | XZZX (corr, X) | 3.4e-10 [1.3e-10, 8.9e-10] | 82.20 [26.47, 255.22] |
| purez | 0.003 | 5/7 (99/97) | 3.3e-4 [3.2e-4, 3.5e-4] | CSS (corr, X) | 1.3e-3 [1.3e-3, 1.4e-3] | 0.25 [0.23, 0.27] |
| purez | 0.003 | 5/7 (99/97) | 3.3e-4 [3.2e-4, 3.5e-4] | XZZX (corr, X) | 1.9e-5 [1.8e-5, 2.0e-5] | 17.39 [15.68, 19.29] |
| purez | 0.003 | 9/13 (323/337) | 1.1e-5 [1.0e-5, 1.2e-5] | CSS (corr, X) | 10.0e-5 [9.4e-5, 1.1e-4] | 0.11 [0.10, 0.12] |
| purez | 0.003 | 9/13 (323/337) | 1.1e-5 [1.0e-5, 1.2e-5] | XZZX (corr, X) | 3.0e-8 [2.7e-8, 3.3e-8] | 367.23 [309.14, 436.23] |
| purez | 0.003 | 11/15 (483/449) | 1.8e-6 [1.7e-6, 1.9e-6] | CSS (corr, X) | 3.9e-5 [3.7e-5, 4.1e-5] | 0.05 [0.04, 0.05] |
| purez | 0.003 | 11/15 (483/449) | 1.8e-6 [1.7e-6, 1.9e-6] | XZZX (corr, X) | 2.1e-9 [1.2e-9, 3.7e-9] | 849.87 [455.42, 1585.96] |

## Lambda (chosen decoder)

| noise | p | code | decoder, basis | Lambda | change without smallest d | distances |
|---|---|---|---|---|---|---|
| sd6 | 0.001 | XYZ^2 | corr, X | 11.00(39) | +0.28 | 5 7 9 11 |
| sd6 | 0.001 | CSS | corr, Z | 13.84(123) | -0.20 | 5 7 9 11 |
| sd6 | 0.002 | XYZ^2 | corr, X | 4.04(20) | -0.17 | 5 7 9 11 |
| sd6 | 0.002 | CSS | corr, Z | 6.56(14) | -0.03 | 5 7 9 11 13 15 |
| sd6 | 0.003 | XYZ^2 | corr, X | 2.34(9) | -0.14 | 5 7 9 11 |
| sd6 | 0.003 | CSS | corr, Z | 3.98(5) | +0.02 | 5 7 9 11 13 15 |
| biased100 | 0.001 | XYZ^2 | corr, Y | 31.36(360) | -- | 5 7 |
| biased100 | 0.001 | CSS | corr, X | 8.04(18) | +0.18 | 5 7 9 11 13 |
| biased100 | 0.001 | XZZX | corr, X | 25.20(176) | -1.92 | 5 7 9 |
| biased100 | 0.002 | XYZ^2 | corr, Y | 12.26(44) | -0.90 | 5 7 9 11 |
| biased100 | 0.002 | CSS | corr, X | 3.70(6) | -0.02 | 5 7 9 11 13 15 |
| biased100 | 0.002 | XZZX | corr, X | 13.56(56) | -0.32 | 5 7 9 11 |
| biased100 | 0.003 | XYZ^2 | corr, Y | 5.45(17) | -0.04 | 5 7 9 11 |
| biased100 | 0.003 | CSS | corr, X | 2.37(5) | -0.05 | 5 7 9 11 13 15 |
| biased100 | 0.003 | XZZX | corr, X | 7.84(17) | +0.13 | 5 7 9 11 13 |
| purez | 0.001 | XYZ^2 | corr, Y | 41.54(477) | -- | 5 7 |
| purez | 0.001 | CSS | corr, X | 7.46(19) | -0.07 | 5 7 9 11 13 15 |
| purez | 0.001 | XZZX | corr, X | 31.53(236) | -1.72 | 5 7 9 |
| purez | 0.002 | XYZ^2 | corr, Y | 12.20(49) | -0.63 | 5 7 9 11 |
| purez | 0.002 | CSS | corr, X | 3.67(5) | -0.01 | 5 7 9 11 13 15 |
| purez | 0.002 | XZZX | corr, X | 14.71(72) | -0.71 | 5 7 9 11 |
| purez | 0.003 | XYZ^2 | corr, Y | 5.67(13) | +0.06 | 5 7 9 11 |
| purez | 0.003 | CSS | corr, X | 2.41(3) | +0.00 | 5 7 9 11 13 15 |
| purez | 0.003 | XZZX | corr, X | 9.12(29) | -0.48 | 5 7 9 11 13 |

## Qubits per logical qubit for epsilon per round (chosen decoder)

| noise | p | code | 1e-6 | 1e-9 |
|---|---|---|---|---|
| sd6 | 0.001 | XYZ^2 | 244 (d = 7.8) | 738 (d = 13.6) extrapolated |
| sd6 | 0.001 | CSS | 120 (d = 7.8) | 339 (d = 13.0) extrapolated |
| sd6 | 0.002 | XYZ^2 | 730 (d = 13.5) extrapolated | beyond two decades |
| sd6 | 0.002 | CSS | 245 (d = 11.1) | 679 (d = 18.4) extrapolated |
| sd6 | 0.003 | XYZ^2 | 2,029 (d = 22.5) extrapolated | beyond two decades |
| sd6 | 0.003 | CSS | 455 (d = 15.1) extrapolated | beyond two decades |
| biased100 | 0.001 | XYZ^2 | 134 (d = 5.8) | beyond two decades |
| biased100 | 0.001 | CSS | 192 (d = 9.8) | 540 (d = 16.4) extrapolated |
| biased100 | 0.001 | XZZX | 82 (d = 6.4) | 229 (d = 10.7) extrapolated |
| biased100 | 0.002 | XYZ^2 | 279 (d = 8.4) | 769 (d = 13.9) extrapolated |
| biased100 | 0.002 | CSS | 488 (d = 15.6) extrapolated | beyond two decades |
| biased100 | 0.002 | XZZX | 142 (d = 8.5) | 378 (d = 13.8) extrapolated |
| biased100 | 0.003 | XYZ^2 | 566 (d = 11.9) extrapolated | beyond two decades |
| biased100 | 0.003 | CSS | 1,124 (d = 23.7) extrapolated | beyond two decades |
| biased100 | 0.003 | XZZX | 214 (d = 10.4) | 582 (d = 17.1) extrapolated |
| purez | 0.001 | XYZ^2 | 132 (d = 5.8) | 357 (d = 9.5) extrapolated |
| purez | 0.001 | CSS | 194 (d = 9.9) | 560 (d = 16.7) extrapolated |
| purez | 0.001 | XZZX | 75 (d = 6.2) | 206 (d = 10.2) extrapolated |
| purez | 0.002 | XYZ^2 | 272 (d = 8.3) | 758 (d = 13.8) extrapolated |
| purez | 0.002 | CSS | 492 (d = 15.7) extrapolated | beyond two decades |
| purez | 0.002 | XZZX | 127 (d = 8.0) | 344 (d = 13.1) extrapolated |
| purez | 0.003 | XYZ^2 | 547 (d = 11.7) extrapolated | beyond two decades |
| purez | 0.003 | CSS | 1,096 (d = 23.4) extrapolated | beyond two decades |
| purez | 0.003 | XZZX | 189 (d = 9.8) | 511 (d = 16.0) extrapolated |

## Equal-qubit pairs (MWPM alone)

| noise | p | pair (qubits) | XYZ^2 eps | other | other eps | ratio XYZ^2/other |
|---|---|---|---|---|---|---|
| sd6 | 0.001 | 5/7 (99/97) | 7.7e-5 [7.3e-5, 8.2e-5] | CSS (mwpm, X) | 8.8e-6 [8.3e-6, 9.3e-6] | 8.79 [7.84, 9.86] |
| sd6 | 0.001 | 9/13 (323/337) | 1.6e-6 [1.5e-6, 1.7e-6] | CSS (mwpm, X) | 1.1e-8 [1.0e-8, 1.2e-8] | 143.32 [125.32, 163.91] |
| sd6 | 0.001 | 11/15 (483/449) | 2.0e-7 [1.9e-7, 2.1e-7] | CSS (mwpm, X) | 1.2e-9 [8.8e-10, 1.6e-9] | 169.53 [118.74, 242.06] |
| sd6 | 0.002 | 5/7 (99/97) | 7.9e-4 [7.6e-4, 8.2e-4] | CSS (mwpm, X) | 1.3e-4 [1.3e-4, 1.4e-4] | 5.88 [5.36, 6.45] |
| sd6 | 0.002 | 9/13 (323/337) | 9.5e-5 [8.9e-5, 1.0e-4] | CSS (mwpm, X) | 1.6e-6 [1.5e-6, 1.7e-6] | 59.86 [53.40, 67.10] |
| sd6 | 0.002 | 11/15 (483/449) | 3.4e-5 [3.2e-5, 3.6e-5] | CSS (mwpm, X) | 3.8e-7 [3.6e-7, 4.0e-7] | 90.13 [80.36, 101.08] |
| sd6 | 0.003 | 5/7 (99/97) | 4.3e-3 [4.0e-3, 4.5e-3] | CSS (mwpm, X) | 6.1e-4 [5.8e-4, 6.3e-4] | 7.05 [6.39, 7.77] |
| sd6 | 0.003 | 9/13 (323/337) | 1.2e-3 [1.1e-3, 1.2e-3] | CSS (mwpm, X) | 3.1e-5 [2.9e-5, 3.3e-5] | 37.55 [33.57, 42.00] |
| sd6 | 0.003 | 11/15 (483/449) | 6.4e-4 [6.1e-4, 6.8e-4] | CSS (mwpm, X) | 1.0e-5 [9.6e-6, 1.1e-5] | 63.30 [56.66, 70.71] |
| biased100 | 0.001 | 5/7 (99/97) | 5.7e-5 [5.4e-5, 6.1e-5] | CSS (mwpm, X) | 2.3e-5 [2.1e-5, 2.4e-5] | 2.53 [2.26, 2.83] |
| biased100 | 0.001 | 5/7 (99/97) | 5.7e-5 [5.4e-5, 6.1e-5] | XZZX (mwpm, X) | 5.8e-7 [5.5e-7, 6.2e-7] | 98.02 [87.47, 109.85] |
| biased100 | 0.001 | 9/13 (323/337) | 5.4e-7 [5.1e-7, 5.7e-7] | CSS (mwpm, X) | 6.0e-8 [5.7e-8, 6.3e-8] | 9.01 [8.03, 10.10] |
| biased100 | 0.001 | 9/13 (323/337) | 5.4e-7 [5.1e-7, 5.7e-7] | XZZX (mwpm, X) | < 1.9e-10 | -- |
| biased100 | 0.001 | 11/15 (483/449) | 5.6e-8 [5.3e-8, 5.9e-8] | CSS (mwpm, X) | 7.6e-9 [6.7e-9, 8.6e-9] | 7.32 [6.11, 8.78] |
| biased100 | 0.001 | 11/15 (483/449) | 5.6e-8 [5.3e-8, 5.9e-8] | XZZX (mwpm, X) | < 1.9e-10 | -- |
| biased100 | 0.002 | 5/7 (99/97) | 6.5e-4 [6.2e-4, 6.8e-4] | CSS (mwpm, X) | 3.2e-4 [3.0e-4, 3.4e-4] | 2.03 [1.84, 2.25] |
| biased100 | 0.002 | 5/7 (99/97) | 6.5e-4 [6.2e-4, 6.8e-4] | XZZX (mwpm, X) | 8.4e-6 [7.9e-6, 8.9e-6] | 77.13 [69.76, 85.28] |
| biased100 | 0.002 | 9/13 (323/337) | 4.9e-5 [4.6e-5, 5.1e-5] | CSS (mwpm, X) | 7.9e-6 [7.4e-6, 8.3e-6] | 6.17 [5.50, 6.91] |
| biased100 | 0.002 | 9/13 (323/337) | 4.9e-5 [4.6e-5, 5.1e-5] | XZZX (mwpm, X) | 5.3e-9 [4.5e-9, 6.2e-9] | 9130.72 [7343.42, 11353.02] |
| biased100 | 0.002 | 11/15 (483/449) | 1.2e-5 [1.2e-5, 1.3e-5] | CSS (mwpm, X) | 2.1e-6 [2.0e-6, 2.2e-6] | 5.97 [5.32, 6.70] |
| biased100 | 0.002 | 11/15 (483/449) | 1.2e-5 [1.2e-5, 1.3e-5] | XZZX (mwpm, X) | 8.8e-10 [5.9e-10, 1.3e-9] | 14183.72 [8949.69, 22478.76] |
| biased100 | 0.003 | 5/7 (99/97) | 2.5e-3 [2.5e-3, 2.6e-3] | CSS (mwpm, X) | 1.5e-3 [1.4e-3, 1.5e-3] | 1.75 [1.62, 1.89] |
| biased100 | 0.003 | 5/7 (99/97) | 2.5e-3 [2.5e-3, 2.6e-3] | XZZX (mwpm, X) | 4.2e-5 [3.9e-5, 4.4e-5] | 61.01 [55.59, 66.95] |
| biased100 | 0.003 | 9/13 (323/337) | 6.7e-4 [6.3e-4, 7.1e-4] | CSS (mwpm, X) | 1.4e-4 [1.3e-4, 1.4e-4] | 4.94 [4.40, 5.54] |
| biased100 | 0.003 | 9/13 (323/337) | 6.7e-4 [6.3e-4, 7.1e-4] | XZZX (mwpm, X) | 1.3e-7 [1.2e-7, 1.3e-7] | 5327.02 [4747.34, 5977.49] |
| biased100 | 0.003 | 11/15 (483/449) | 3.2e-4 [3.0e-4, 3.4e-4] | CSS (mwpm, X) | 6.1e-5 [5.8e-5, 6.5e-5] | 5.20 [4.64, 5.83] |
| biased100 | 0.003 | 11/15 (483/449) | 3.2e-4 [3.0e-4, 3.4e-4] | XZZX (mwpm, X) | 1.7e-8 [1.5e-8, 1.9e-8] | 18877.02 [15989.04, 22286.62] |
| purez | 0.001 | 5/7 (99/97) | 5.0e-5 [4.8e-5, 5.3e-5] | CSS (mwpm, X) | 2.4e-5 [2.3e-5, 2.6e-5] | 2.07 [1.85, 2.31] |
| purez | 0.001 | 5/7 (99/97) | 5.0e-5 [4.8e-5, 5.3e-5] | XZZX (mwpm, X) | 3.0e-7 [2.9e-7, 3.2e-7] | 165.98 [148.31, 185.76] |
| purez | 0.001 | 9/13 (323/337) | 5.4e-7 [5.1e-7, 5.7e-7] | CSS (mwpm, X) | 5.8e-8 [5.5e-8, 6.2e-8] | 9.27 [8.27, 10.40] |
| purez | 0.001 | 9/13 (323/337) | 5.4e-7 [5.1e-7, 5.7e-7] | XZZX (mwpm, X) | < 8.8e-10 | -- |
| purez | 0.001 | 11/15 (483/449) | 5.6e-8 [5.3e-8, 5.9e-8] | CSS (mwpm, X) | 8.0e-9 [7.1e-9, 9.0e-9] | 7.01 [5.91, 8.32] |
| purez | 0.001 | 11/15 (483/449) | 5.6e-8 [5.3e-8, 5.9e-8] | XZZX (mwpm, X) | < 8.8e-10 | -- |
| purez | 0.002 | 5/7 (99/97) | 5.9e-4 [5.6e-4, 6.1e-4] | CSS (mwpm, X) | 3.4e-4 [3.2e-4, 3.6e-4] | 1.74 [1.57, 1.92] |
| purez | 0.002 | 5/7 (99/97) | 5.9e-4 [5.6e-4, 6.1e-4] | XZZX (mwpm, X) | 4.9e-6 [4.6e-6, 5.1e-6] | 120.41 [108.72, 133.35] |
| purez | 0.002 | 9/13 (323/337) | 4.8e-5 [4.5e-5, 5.0e-5] | CSS (mwpm, X) | 7.4e-6 [7.0e-6, 7.8e-6] | 6.44 [5.75, 7.22] |
| purez | 0.002 | 9/13 (323/337) | 4.8e-5 [4.5e-5, 5.0e-5] | XZZX (mwpm, X) | 2.0e-9 [1.6e-9, 2.6e-9] | 23257.13 [17502.79, 30903.29] |
| purez | 0.002 | 11/15 (483/449) | 1.1e-5 [1.1e-5, 1.2e-5] | CSS (mwpm, X) | 2.0e-6 [1.9e-6, 2.2e-6] | 5.45 [4.86, 6.11] |
| purez | 0.002 | 11/15 (483/449) | 1.1e-5 [1.1e-5, 1.2e-5] | XZZX (mwpm, X) | 3.2e-10 [1.6e-10, 6.3e-10] | 35079.93 [16624.14, 74024.93] |
| purez | 0.003 | 5/7 (99/97) | 2.5e-3 [2.4e-3, 2.6e-3] | CSS (mwpm, X) | 1.5e-3 [1.5e-3, 1.6e-3] | 1.67 [1.55, 1.80] |
| purez | 0.003 | 5/7 (99/97) | 2.5e-3 [2.4e-3, 2.6e-3] | XZZX (mwpm, X) | 2.6e-5 [2.4e-5, 2.7e-5] | 98.51 [89.78, 108.08] |
| purez | 0.003 | 9/13 (323/337) | 6.4e-4 [6.1e-4, 6.8e-4] | CSS (mwpm, X) | 1.3e-4 [1.2e-4, 1.3e-4] | 5.13 [4.60, 5.71] |
| purez | 0.003 | 9/13 (323/337) | 6.4e-4 [6.1e-4, 6.8e-4] | XZZX (mwpm, X) | 3.8e-8 [3.5e-8, 4.0e-8] | 17151.85 [15341.89, 19175.32] |
| purez | 0.003 | 11/15 (483/449) | 3.1e-4 [2.9e-4, 3.3e-4] | CSS (mwpm, X) | 6.4e-5 [6.0e-5, 6.7e-5] | 4.91 [4.38, 5.50] |
| purez | 0.003 | 11/15 (483/449) | 3.1e-4 [2.9e-4, 3.3e-4] | XZZX (mwpm, X) | 4.3e-9 [3.5e-9, 5.4e-9] | 72108.40 [54821.85, 94845.73] |

## Lambda (MWPM alone)

| noise | p | code | decoder, basis | Lambda | change without smallest d | distances |
|---|---|---|---|---|---|---|
| sd6 | 0.001 | XYZ^2 | mwpm, X | 7.27(24) | +0.64 | 5 7 9 11 |
| sd6 | 0.001 | CSS | mwpm, X | 9.25(24) | +0.05 | 5 7 9 11 13 |
| sd6 | 0.002 | XYZ^2 | mwpm, X | 2.85(6) | -0.03 | 5 7 9 11 |
| sd6 | 0.002 | CSS | mwpm, X | 4.32(6) | +0.05 | 5 7 9 11 13 15 |
| sd6 | 0.003 | XYZ^2 | mwpm, Y | 1.88(5) | -0.04 | 5 7 9 11 |
| sd6 | 0.003 | CSS | mwpm, X | 2.73(4) | +0.03 | 5 7 9 11 13 15 |
| biased100 | 0.001 | XYZ^2 | mwpm, X | 10.10(26) | -0.37 | 5 7 9 11 |
| biased100 | 0.001 | CSS | mwpm, X | 7.08(14) | +0.20 | 5 7 9 11 13 15 |
| biased100 | 0.001 | XZZX | mwpm, X | 23.35(164) | +0.48 | 5 7 9 |
| biased100 | 0.002 | XYZ^2 | mwpm, X | 3.70(9) | -0.13 | 5 7 9 11 |
| biased100 | 0.002 | CSS | mwpm, X | 3.47(5) | +0.03 | 5 7 9 11 13 15 |
| biased100 | 0.002 | XZZX | mwpm, X | 11.20(38) | -0.06 | 5 7 9 11 13 |
| biased100 | 0.003 | XYZ^2 | mwpm, X | 1.99(4) | +0.02 | 5 7 9 11 |
| biased100 | 0.003 | CSS | mwpm, X | 2.24(4) | -0.03 | 5 7 9 11 13 15 |
| biased100 | 0.003 | XZZX | mwpm, X | 7.06(14) | -0.06 | 5 7 9 11 13 15 |
| purez | 0.001 | XYZ^2 | mwpm, X | 9.66(25) | +0.17 | 5 7 9 11 |
| purez | 0.001 | CSS | mwpm, X | 7.26(15) | +0.22 | 5 7 9 11 13 15 |
| purez | 0.001 | XZZX | mwpm, X | 27.52(192) | +0.69 | 5 7 9 |
| purez | 0.002 | XYZ^2 | mwpm, X | 3.69(13) | +0.04 | 5 7 9 11 |
| purez | 0.002 | CSS | mwpm, X | 3.59(5) | +0.01 | 5 7 9 11 13 15 |
| purez | 0.002 | XZZX | mwpm, X | 13.09(52) | -0.45 | 5 7 9 11 |
| purez | 0.003 | XYZ^2 | mwpm, X | 2.00(4) | +0.02 | 5 7 9 11 |
| purez | 0.003 | CSS | mwpm, X | 2.20(3) | +0.02 | 5 7 9 11 13 15 |
| purez | 0.003 | XZZX | mwpm, X | 8.73(17) | +0.01 | 5 7 9 11 13 |

## Qubits per logical qubit for epsilon per round (MWPM alone)

| noise | p | code | 1e-6 | 1e-9 |
|---|---|---|---|---|
| sd6 | 0.001 | XYZ^2 | 354 (d = 9.4) | beyond two decades |
| sd6 | 0.001 | CSS | 158 (d = 8.9) | 457 (d = 15.1) extrapolated |
| sd6 | 0.002 | XYZ^2 | 1,257 (d = 17.7) extrapolated | beyond two decades |
| sd6 | 0.002 | CSS | 373 (d = 13.7) | beyond two decades |
| sd6 | 0.003 | XYZ^2 | 3,934 (d = 31.4) extrapolated | beyond two decades |
| sd6 | 0.003 | CSS | 776 (d = 19.7) extrapolated | beyond two decades |
| biased100 | 0.001 | XYZ^2 | 287 (d = 8.5) | 835 (d = 14.5) extrapolated |
| biased100 | 0.001 | CSS | 204 (d = 10.1) | 590 (d = 17.2) extrapolated |
| biased100 | 0.001 | XZZX | 88 (d = 6.7) | 243 (d = 11.0) extrapolated |
| biased100 | 0.002 | XYZ^2 | 884 (d = 14.9) extrapolated | beyond two decades |
| biased100 | 0.002 | CSS | 528 (d = 16.3) extrapolated | beyond two decades |
| biased100 | 0.002 | XZZX | 153 (d = 8.8) | 419 (d = 14.5) extrapolated |
| biased100 | 0.003 | XYZ^2 | 3,099 (d = 27.8) extrapolated | beyond two decades |
| biased100 | 0.003 | CSS | 1,261 (d = 25.1) extrapolated | beyond two decades |
| biased100 | 0.003 | XZZX | 234 (d = 10.8) | 641 (d = 17.9) extrapolated |
| purez | 0.001 | XYZ^2 | 285 (d = 8.5) | 846 (d = 14.6) extrapolated |
| purez | 0.001 | CSS | 205 (d = 10.1) | 585 (d = 17.1) extrapolated |
| purez | 0.001 | XZZX | 78 (d = 6.3) | 217 (d = 10.4) extrapolated |
| purez | 0.002 | XYZ^2 | 872 (d = 14.8) extrapolated | beyond two decades |
| purez | 0.002 | CSS | 520 (d = 16.1) extrapolated | beyond two decades |
| purez | 0.002 | XZZX | 135 (d = 8.3) | 371 (d = 13.6) extrapolated |
| purez | 0.003 | XYZ^2 | 3,049 (d = 27.6) extrapolated | beyond two decades |
| purez | 0.003 | CSS | 1,291 (d = 25.4) extrapolated | beyond two decades |
| purez | 0.003 | XZZX | 198 (d = 10.0) | 534 (d = 16.4) extrapolated |

## Time boundary at the pair points (chosen decoder)

| noise | p | code | d | eps (r = 3d) | eps (r = d and 3d) | change |
|---|---|---|---|---|---|---|
| sd6 | 0.001 | XYZ^2 | 5 | 2.9e-5 | 2.8e-5 | -3% |
| sd6 | 0.001 | XYZ^2 | 9 | 2.5e-7 | 2.6e-7 | +4% |
| sd6 | 0.001 | XYZ^2 | 11 | 2.1e-8 | 2.1e-8 | +1% |
| sd6 | 0.001 | CSS | 7 | 2.7e-6 | 2.3e-6 | -14% |
| sd6 | 0.001 | CSS | 13 | 3.6e-10 | -- | fewer than 30 errors |
| sd6 | 0.002 | XYZ^2 | 5 | 3.9e-4 | 4.1e-4 | +4% |
| sd6 | 0.002 | XYZ^2 | 9 | 2.1e-5 | 2.1e-5 | +4% |
| sd6 | 0.002 | XYZ^2 | 11 | 6.2e-6 | 7.3e-6 | +17% |
| sd6 | 0.002 | CSS | 7 | 4.6e-5 | 4.3e-5 | -6% |
| sd6 | 0.002 | CSS | 13 | 1.7e-7 | 1.9e-7 | +7% |
| sd6 | 0.002 | CSS | 15 | 1.9e-8 | 1.5e-8 | -18% |
| sd6 | 0.003 | XYZ^2 | 5 | 1.8e-3 | 1.9e-3 | +6% |
| sd6 | 0.003 | XYZ^2 | 9 | 3.3e-4 | 3.7e-4 | +11% |
| sd6 | 0.003 | XYZ^2 | 11 | 1.3e-4 | 1.5e-4 | +10% |
| sd6 | 0.003 | CSS | 7 | 2.7e-4 | 2.5e-4 | -6% |
| sd6 | 0.003 | CSS | 13 | 4.3e-6 | 4.5e-6 | +2% |
| sd6 | 0.003 | CSS | 15 | 1.0e-6 | 1.0e-6 | -1% |
| biased100 | 0.001 | XYZ^2 | 5 | 4.1e-6 | 2.4e-6 | -40% |
| biased100 | 0.001 | XYZ^2 | 9 | 1.7e-9 | -- | fewer than 30 errors |
| biased100 | 0.001 | XYZ^2 | 11 | 4.1e-10 | -- | fewer than 30 errors |
| biased100 | 0.001 | CSS | 7 | 2.0e-5 | 2.2e-5 | +12% |
| biased100 | 0.001 | CSS | 13 | 3.5e-8 | 3.7e-8 | +4% |
| biased100 | 0.001 | CSS | 15 | 3.3e-9 | -- | fewer than 30 errors |
| biased100 | 0.001 | XZZX | 7 | 3.9e-7 | 3.9e-7 | -0% |
| biased100 | 0.002 | XYZ^2 | 5 | 6.9e-5 | 5.2e-5 | -24% |
| biased100 | 0.002 | XYZ^2 | 9 | 4.4e-7 | 3.4e-7 | -23% |
| biased100 | 0.002 | XYZ^2 | 11 | 4.1e-8 | 3.5e-8 | -14% |
| biased100 | 0.002 | CSS | 7 | 2.7e-4 | 2.9e-4 | +5% |
| biased100 | 0.002 | CSS | 13 | 5.9e-6 | 6.6e-6 | +12% |
| biased100 | 0.002 | CSS | 15 | 1.4e-6 | 1.5e-6 | +7% |
| biased100 | 0.002 | XZZX | 7 | 6.5e-6 | 6.5e-6 | -1% |
| biased100 | 0.002 | XZZX | 13 | 2.9e-9 | -- | fewer than 30 errors |
| biased100 | 0.002 | XZZX | 15 | -- | -- | fewer than 30 errors |
| biased100 | 0.003 | XYZ^2 | 5 | 3.5e-4 | 2.8e-4 | -20% |
| biased100 | 0.003 | XYZ^2 | 9 | 1.1e-5 | 9.5e-6 | -13% |
| biased100 | 0.003 | XYZ^2 | 11 | 2.3e-6 | 2.3e-6 | +1% |
| biased100 | 0.003 | CSS | 7 | 1.2e-3 | 1.3e-3 | +3% |
| biased100 | 0.003 | CSS | 13 | 10.0e-5 | 1.1e-4 | +12% |
| biased100 | 0.003 | CSS | 15 | 4.3e-5 | 4.9e-5 | +15% |
| biased100 | 0.003 | XZZX | 7 | 3.3e-5 | 3.1e-5 | -6% |
| biased100 | 0.003 | XZZX | 13 | 6.4e-8 | 6.7e-8 | +5% |
| biased100 | 0.003 | XZZX | 15 | 7.3e-9 | -- | fewer than 30 errors |
| purez | 0.001 | XYZ^2 | 5 | 4.1e-6 | 2.2e-6 | -46% |
| purez | 0.001 | XYZ^2 | 9 | 2.4e-9 | -- | fewer than 30 errors |
| purez | 0.001 | CSS | 7 | 1.8e-5 | 1.8e-5 | +1% |
| purez | 0.001 | CSS | 13 | 4.0e-8 | 4.3e-8 | +6% |
| purez | 0.001 | CSS | 15 | 8.3e-9 | -- | fewer than 30 errors |
| purez | 0.001 | XZZX | 7 | 2.4e-7 | 2.1e-7 | -12% |
| purez | 0.002 | XYZ^2 | 5 | 5.9e-5 | 3.8e-5 | -36% |
| purez | 0.002 | XYZ^2 | 9 | 4.1e-7 | 3.6e-7 | -13% |
| purez | 0.002 | XYZ^2 | 11 | 2.8e-8 | 1.9e-8 | -34% |
| purez | 0.002 | CSS | 7 | 2.9e-4 | 3.1e-4 | +8% |
| purez | 0.002 | CSS | 13 | 5.4e-6 | 5.5e-6 | +1% |
| purez | 0.002 | CSS | 15 | 1.6e-6 | 1.8e-6 | +8% |
| purez | 0.002 | XZZX | 7 | 3.7e-6 | 3.2e-6 | -13% |
| purez | 0.002 | XZZX | 13 | 2.7e-9 | -- | fewer than 30 errors |
| purez | 0.002 | XZZX | 15 | 3.4e-10 | -- | fewer than 30 errors |
| purez | 0.003 | XYZ^2 | 5 | 3.3e-4 | 2.6e-4 | -23% |
| purez | 0.003 | XYZ^2 | 9 | 1.1e-5 | 1.0e-5 | -4% |
| purez | 0.003 | XYZ^2 | 11 | 1.8e-6 | 1.6e-6 | -10% |
| purez | 0.003 | CSS | 7 | 1.3e-3 | 1.4e-3 | +5% |
| purez | 0.003 | CSS | 13 | 10.0e-5 | 1.1e-4 | +6% |
| purez | 0.003 | CSS | 15 | 3.9e-5 | 4.2e-5 | +7% |
| purez | 0.003 | XZZX | 7 | 1.9e-5 | 1.7e-5 | -10% |
| purez | 0.003 | XZZX | 13 | 3.0e-8 | 3.1e-8 | +3% |
| purez | 0.003 | XZZX | 15 | 2.1e-9 | -- | fewer than 30 errors |

Panel rates: sd6 0.001, biased100 0.001, purez 0.001
