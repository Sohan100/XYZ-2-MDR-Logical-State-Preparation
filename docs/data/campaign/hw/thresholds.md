# Corners of the Hetenyi-Wootton threshold surface in our circuits

Thresholds in % of p, r = d, uncertainty in the last digits (campaign_report rules: finite-size fit over d = 11 to 21, honeycomb 12 to 20; `drift`, pair-only and d < 11 fits do not count). Theirs: Table I of arXiv:2306.17786 (d = 11 to 17, matching, intercepts of a fitted plane). Ours, MWPM: the weaker basis under MWPM. Best: each basis with its best decoder (MWPM, correlated matching, belief-matching), then the weaker basis. Crossing: MWPM, d = 7/9 (honeycomb 4/8), weaker basis.

## hw_g: gates only

| code | theirs | ours, MWPM | ours, best | basis, decoder | small-d crossing |
|---|---|---|---|---|---|
| rotated surface (CSS) | 0.82 | 1.15(1) | 1.40(5) | X, BM | 1.05(8) (7/9, X) |
| XZZX surface | 0.37 | 1.15(2) | 1.39(1) | Z, corr | 1.06(3) (7/9, Z) |
| XYZ^2 | 0.465 | 0.89(2) | 1.10(4) | X, BM | 0.86(3) (7/9, X) |
| honeycomb Floquet | 0.43 | 0.88(1) | 1.081(8) | X, corr | 0.97(1) (4/8, X) |

## hw_g1q: gates, noisy 1q basis changes

| code | theirs | ours, MWPM | ours, best | basis, decoder | small-d crossing |
|---|---|---|---|---|---|
| rotated surface (CSS) | -- | 0.433(3) | 0.52(1) | Z, BM | 0.41(1) (7/9, X) |
| XZZX surface | -- | 0.438(4) | 0.51(1) | X, BM | 0.40(1) (7/9, Z) |
| XYZ^2 | -- | 0.326(8) | 0.402(7) | X, BM | 0.315(7) (7/9, X) |
| honeycomb Floquet | -- | 0.293(2) | 0.372(5) | Z, BM | 0.316(3) (4/8, X) |

## hw_t: idling only

| code | theirs | ours, MWPM | ours, best | basis, decoder | small-d crossing |
|---|---|---|---|---|---|
| rotated surface (CSS) | 3.94 | 3.6(4) | 3.8(4) | X, BM | 3.1(2) (7/9, X) |
| XZZX surface | 15.1 | 9.8(3) | 10(2) | Z, BM | 7.7(4) (7/9, Z) |
| XYZ^2 | 4.2 | 0.88(1) | 1.330(6) | Y, corr | 0.88(2) (7/9, X) |
| honeycomb Floquet | 1.16 | 0.392(3) | 0.51(1) | Z, BM | 0.416(6) (4/8, X) |

## hw_rr: readout only

| code | theirs | ours, MWPM | ours, best | basis, decoder | small-d crossing |
|---|---|---|---|---|---|
| rotated surface (CSS) | 14.5 | 9.8(4) | 10.5(6) | Z, BM | 10.2(5) (7/9, Z) |
| XZZX surface | 12.9 | 9.8(3) | 10.3(4) | X, BM | 9.8(5) (7/9, Z) |
| XYZ^2 | 16.5 | 10.6(3) | 10.6(5) | X, corr | 11.2(4) (7/9, X; no Y crossing) |
| honeycomb Floquet | 0.99 | 1.007(7) | 1.72(3) | Z, BM | 1.19(1) (4/8, Z) |

## Every series

| model | code | basis | decoder | threshold | method | d | counts | small-d crossing |
|---|---|---|---|---|---|---|---|---|
| hw_g | css | X | MWPM | 1.16(2) | fss | 11-21 | yes | 1.05(8) (7/9) |
| hw_g | css | X | corr | 1.40(3) | fss | 11-21 | yes | 1.37(4) (7/9) |
| hw_g | css | X | BM | 1.40(5) | fss | 11-21 | yes | 1.3(1) (7/9) |
| hw_g | css | Z | MWPM | 1.15(1) | fss | 11-21 | yes | 1.09(3) (7/9) |
| hw_g | css | Z | corr | 1.42(2) | fss | 11-21 | yes | 1.32(3) (7/9) |
| hw_g | css | Z | BM | 1.41(2) | fss | 11-21 | yes | 1.3(1) (7/9) |
| hw_g | xzzx | X | MWPM | 1.15(2) | fss | 11-21 | yes | 1.14(3) (7/9) |
| hw_g | xzzx | X | corr | 1.39(2) | fss | 11-21 | yes | 1.35(2) (7/9) |
| hw_g | xzzx | X | BM | 1.39(2) | fss | 11-21 | yes | 1.24(9) (7/9) |
| hw_g | xzzx | Z | MWPM | 1.15(4) | fss | 11-21 | yes | 1.06(3) (7/9) |
| hw_g | xzzx | Z | corr | 1.39(1) | fss | 11-21 | yes | 1.32(4) (7/9) |
| hw_g | xzzx | Z | BM | 1.35(4) | fss | 11-21 | yes | 1.38(9) (7/9) |
| hw_g | xyz2 | X | MWPM | 0.89(2) | fss | 11-21 | yes | 0.86(3) (7/9) |
| hw_g | xyz2 | X | corr | 1.043(8) | fss | 11-21 | yes | 1.00(3) (7/9) |
| hw_g | xyz2 | X | BM | 1.10(4) | fss | 11-21 | yes | 1.09(7) (7/9) |
| hw_g | xyz2 | Y | MWPM | 0.94(1) | fss | 11-21 | yes | 0.92(3) (7/9) |
| hw_g | xyz2 | Y | corr | 1.09(4) | fss | 11-21 | yes | 1.13(6) (7/9) |
| hw_g | xyz2 | Y | BM | 1.12(1) | fss | 11-21 | yes | 1.12(6) (7/9) |
| hw_g | honeycomb | X | MWPM | 0.88(1) | fss | 12-20 | yes | 0.97(1) (4/8) |
| hw_g | honeycomb | X | corr | 1.081(8) | fss | 12-20 | yes | 1.15(1) (4/8) |
| hw_g | honeycomb | X | BM | 1.08(1) | fss | 12-20 | yes | 1.19(2) (4/8) |
| hw_g | honeycomb | Z | MWPM | 0.901(8) | fss | 12-20 | yes | 1.00(1) (4/8) |
| hw_g | honeycomb | Z | corr | 1.077(7) | fss | 12-20 | yes | 1.16(1) (4/8) |
| hw_g | honeycomb | Z | BM | 1.12(2) | fss | 12-20 | yes | 1.18(2) (4/8) |
| hw_g1q | css | X | MWPM | 0.433(3) | fss | 11-21 | yes | 0.41(1) (7/9) |
| hw_g1q | css | X | corr | 0.520(6) | fss | 11-21 | yes | 0.47(1) (7/9) |
| hw_g1q | css | X | BM | 0.52(2) | fss | 11-21 | yes | 0.48(5) (7/9) |
| hw_g1q | css | Z | MWPM | 0.442(6) | fss | 11-21 | yes | 0.43(2) (7/9) |
| hw_g1q | css | Z | corr | 0.518(9) | fss | 11-21 | yes | 0.505(9) (7/9) |
| hw_g1q | css | Z | BM | 0.52(1) | fss | 11-21 | yes | 0.50(4) (7/9) |
| hw_g1q | xzzx | X | MWPM | 0.438(4) | fss | 11-21 | yes | 0.43(1) (7/9) |
| hw_g1q | xzzx | X | corr | 0.511(6) | fss | 11-21 | yes | 0.51(2) (7/9) |
| hw_g1q | xzzx | X | BM | 0.51(1) | fss | 11-21 | yes | 0.52(2) (7/9) |
| hw_g1q | xzzx | Z | MWPM | 0.44(1) | fss | 11-21 | yes | 0.40(1) (7/9) |
| hw_g1q | xzzx | Z | corr | 0.52(2) | fss | 11-21 | yes | 0.50(2) (7/9) |
| hw_g1q | xzzx | Z | BM | 0.51(2) | fss | 11-21 | yes | 0.50(4) (7/9) |
| hw_g1q | xyz2 | X | MWPM | 0.326(8) | fss | 11-21 | yes | 0.315(7) (7/9) |
| hw_g1q | xyz2 | X | corr | 0.388(3) | fss | 11-21 | yes | 0.37(1) (7/9) |
| hw_g1q | xyz2 | X | BM | 0.402(7) | fss | 11-21 | yes | 0.39(4) (7/9) |
| hw_g1q | xyz2 | Y | MWPM | 0.339(3) | fss | 11-21 | yes | 0.350(7) (7/9) |
| hw_g1q | xyz2 | Y | corr | 0.403(3) | fss | 11-21 | yes | 0.402(9) (7/9) |
| hw_g1q | xyz2 | Y | BM | 0.64(3) | fss | 11-21 | yes | -- |
| hw_g1q | honeycomb | X | MWPM | 0.293(2) | fss | 12-20 | yes | 0.316(3) (4/8) |
| hw_g1q | honeycomb | X | corr | 0.362(4) | fss | 12-20 | yes | 0.384(5) (4/8) |
| hw_g1q | honeycomb | X | BM | 0.377(5) | fss | 12-20 | yes | 0.40(1) (4/8) |
| hw_g1q | honeycomb | Z | MWPM | 0.293(2) | fss | 12-20 | yes | 0.332(4) (4/8) |
| hw_g1q | honeycomb | Z | corr | 0.365(5) | fss | 12-20 | yes | 0.388(3) (4/8) |
| hw_g1q | honeycomb | Z | BM | 0.372(5) | fss | 12-20 | yes | 0.397(8) (4/8) |
| hw_t | css | X | MWPM | 3.6(4) | fss | 11-21 | yes | 3.1(2) (7/9) |
| hw_t | css | X | corr | 5.0(5) | pair | 19-21 | no | 3.3(2) (7/9) |
| hw_t | css | X | BM | 3.8(4) | fss | 11-21 | yes | 3.1(1) (7/9) |
| hw_t | css | Z | MWPM | 24.7(9) | fss | 11-21 | yes | 32(2) (7/9) |
| hw_t | css | Z | corr | 24.9(4) | fss | 11-21 | yes | 30(1) (7/9) |
| hw_t | css | Z | BM | 23.0(6) | fss | 11-21 | yes | 30(2) (7/9) |
| hw_t | xzzx | X | MWPM | 11(1) | fss | 11-21 | yes | 8.3(4) (7/9) |
| hw_t | xzzx | X | corr | 9.8(5) | fss | 11-21 | yes | 10.4(4) (7/9) |
| hw_t | xzzx | X | BM | 10(1) | fss | 11-21 | yes | 9.7(9) (7/9) |
| hw_t | xzzx | Z | MWPM | 9.8(3) | fss | 11-21 | yes | 7.7(4) (7/9) |
| hw_t | xzzx | Z | corr | 13(2) | pair | 19-21 | no | 7.8(3) (7/9) |
| hw_t | xzzx | Z | BM | 10(2) | fss | 11-21 | yes | 7.6(5) (7/9) |
| hw_t | xyz2 | X | MWPM | 0.88(1) | fss | 11-21 | yes | 0.88(2) (7/9) |
| hw_t | xyz2 | X | corr | 1.34(1) | fss | 11-21 | yes | 1.43(3) (7/9) |
| hw_t | xyz2 | X | BM | 1.67(3) | fss | 11-21 | yes | 1.88(5) (7/9) |
| hw_t | xyz2 | Y | MWPM | 0.93(1) | fss | 11-21 | yes | 0.99(3) (7/9) |
| hw_t | xyz2 | Y | corr | 1.330(6) | fss | 11-21 | yes | 1.41(4) (7/9) |
| hw_t | xyz2 | Y | BM | 1.6(2) drift | pair | 19-21 | no | 1.95(9) (7/9) |
| hw_t | honeycomb | X | MWPM | 0.395(5) | fss | 12-20 | yes | 0.416(6) (4/8) |
| hw_t | honeycomb | X | corr | 0.471(4) | fss | 12-20 | yes | 0.487(6) (4/8) |
| hw_t | honeycomb | X | BM | 0.514(7) | fss | 12-20 | yes | 0.58(2) (4/8) |
| hw_t | honeycomb | Z | MWPM | 0.392(3) | fss | 12-20 | yes | 0.439(6) (4/8) |
| hw_t | honeycomb | Z | corr | 0.472(4) | fss | 12-20 | yes | 0.55(1) (4/8) |
| hw_t | honeycomb | Z | BM | 0.51(1) | fss | 12-20 | yes | 0.59(1) (4/8) |
| hw_rr | css | X | MWPM | 10.0(2) | fss | 11-21 | yes | 10(1) (7/9) |
| hw_rr | css | X | corr | 10.0(3) | fss | 11-21 | yes | 10.4(4) (7/9) |
| hw_rr | css | X | BM | 10.5(7) | fss | 11-21 | yes | 11(1) (7/9) |
| hw_rr | css | Z | MWPM | 9.8(4) | fss | 11-21 | yes | 10.2(5) (7/9) |
| hw_rr | css | Z | corr | 9.9(4) | fss | 11-21 | yes | 10.2(8) (7/9) |
| hw_rr | css | Z | BM | 10.5(6) | fss | 11-21 | yes | 9.2(5) (7/9) |
| hw_rr | xzzx | X | MWPM | 9.8(3) | fss | 11-21 | yes | 10.7(9) (7/9) |
| hw_rr | xzzx | X | corr | 9.9(2) | fss | 11-21 | yes | 9.9(6) (7/9) |
| hw_rr | xzzx | X | BM | 10.3(4) | fss | 11-21 | yes | -- |
| hw_rr | xzzx | Z | MWPM | 9.9(4) | fss | 11-21 | yes | 9.8(5) (7/9) |
| hw_rr | xzzx | Z | corr | 10.0(2) | fss | 11-21 | yes | 10.6(9) (7/9) |
| hw_rr | xzzx | Z | BM | 10.4(8) | fss | 11-21 | yes | 10(1) (7/9) |
| hw_rr | xyz2 | X | MWPM | 10.6(3) | fss | 11-21 | yes | 11.2(4) (7/9) |
| hw_rr | xyz2 | X | corr | 10.6(5) | fss | 11-21 | yes | 10.4(5) (7/9) |
| hw_rr | xyz2 | X | BM | 10(1) | fss | 11-21 | yes | 10.1(8) (7/9) |
| hw_rr | xyz2 | Y | MWPM | 10.9(2) | fss | 11-21 | yes | -- |
| hw_rr | xyz2 | Y | corr | 10.3(5) drift | pair | 19-21 | no | 11(1) (7/9) |
| hw_rr | xyz2 | Y | BM | 9.9(8) drift | pair | 19-21 | no | 13(2) (7/9) |
| hw_rr | honeycomb | X | MWPM | 1.03(1) | fss | 12-20 | yes | 1.20(2) (4/8) |
| hw_rr | honeycomb | X | corr | 1.14(2) | fss | 12-20 | yes | 1.17(2) (4/8) |
| hw_rr | honeycomb | X | BM | 1.75(3) | fss | 12-20 | yes | 2.33(5) (4/8) |
| hw_rr | honeycomb | Z | MWPM | 1.007(7) | fss | 12-20 | yes | 1.19(1) (4/8) |
| hw_rr | honeycomb | Z | corr | 1.14(2) | fss | 12-20 | yes | 1.22(1) (4/8) |
| hw_rr | honeycomb | Z | BM | 1.72(3) | fss | 12-20 | yes | 2.28(4) (4/8) |
