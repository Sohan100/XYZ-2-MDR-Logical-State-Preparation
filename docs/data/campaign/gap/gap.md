# Gap diagnostic: XYZ^2 against XZZX under SD6 and its variants

Memory thresholds in %, r = d, d = 11 to 21: the best of MWPM, correlated matching and belief-matching in each basis, then the weaker basis (XYZ^2: both-bases schedule), and MWPM alone in its weaker basis.

| code | noise model | best decoder | basis, decoder | MWPM alone |
|---|---|---|---|---|
| XYZ$^2$ | SD6 | 0.487(7) | X, bm | 0.400(6) |
| XYZ$^2$ | noiseless links | 0.626(8) | X, corr_gauge | 0.556(6) |
| XYZ$^2$ | links twice | 0.56(2) | X, bm | 0.404(5) |
| XYZ$^2$ | links three times | 0.526(8) | X, corr_gauge | 0.360(4) |
| XYZ$^2$ | hybrid | 0.575(6) | X, corr_gauge | 0.470(5) |
| XYZ$^2$ | no idle errors | 0.854(6) | X, corr_gauge | 0.691(6) |
| XYZ$^2$ | noiseless links, no idle errors | 1.12(3) | X, corr_gauge | 0.99(1) |
| XZZX surface | sd6 | 0.81(2) | Z, bm | 0.697(9) |
| XZZX surface | sd6_noidle | 1.103(9) | Z, corr | 0.93(2) |
| CSS surface | sd6 | 0.81(1) | X, corr | 0.69(1) |
| CSS surface | sd6_noidle | 1.10(1) | X, corr | 0.93(2) |

Fraction of the gap closed, (T_variant - T_SD6) / (T_XZZX - T_SD6), with propagated errors:

| variant | reference | best decoder | MWPM alone | XYZ^2 / reference (best) |
|---|---|---|---|---|
| noiseless links | xzzx sd6 | 0.43(4) | 0.53(3) | 0.77(2) |
| links twice | xzzx sd6 | 0.22(7) | 0.01(3) | 0.69(3) |
| links three times | xzzx sd6 | 0.12(3) | -0.13(3) | 0.65(2) |
| hybrid | xzzx sd6 | 0.27(3) | 0.24(2) | 0.71(2) |
| no idle errors | xzzx sd6 | 1.13(7) | 0.98(4) | 1.05(3) |
| noiseless links, no idle errors | xzzx sd6 | 1.93(15) | 1.98(7) | 1.37(5) |
| noiseless links | xzzx sd6_noidle | 0.23(2) | 0.29(2) | 0.57(1) |
| links twice | xzzx sd6_noidle | 0.12(4) | 0.01(1) | 0.51(2) |
| links three times | xzzx sd6_noidle | 0.06(2) | -0.07(1) | 0.48(1) |
| hybrid | xzzx sd6_noidle | 0.14(1) | 0.13(1) | 0.52(1) |
| no idle errors | xzzx sd6_noidle | 0.60(1) | 0.55(2) | 0.77(1) |
| noiseless links, no idle errors | xzzx sd6_noidle | 1.02(5) | 1.10(4) | 1.01(3) |
