# Coverage of the threshold campaign

Full matrix: 10 noise models x 14 decoders x 22 numbers of rounds (1 to 21 and r = d) x 10 distances = 30800 instances. An instance is measured when at least 3 of its points have 3 or more logical errors; the tensor-network decoders are planned only where their network can be contracted.

## Instances (noise, decoder, rounds, d)

| decoder | planned | measured | sparse | missing | not planned |
|---|---|---|---|---|---|
| mwpm | 2200 | 2200 | 0 | 0 | 0 |
| corr_links | 2200 | 2200 | 0 | 0 | 0 |
| corr_gauge | 2200 | 2200 | 0 | 0 | 0 |
| seq_soft | 2200 | 2200 | 0 | 0 | 0 |
| seq_erasure | 2200 | 2200 | 0 | 0 | 0 |
| seq_match | 2200 | 2200 | 0 | 0 | 0 |
| bm | 2200 | 2200 | 0 | 0 | 0 |
| bp_full | 2200 | 2200 | 0 | 0 | 0 |
| bp_corr | 2200 | 2200 | 0 | 0 | 0 |
| tesseract | 2200 | 2200 | 0 | 0 | 0 |
| cfe | 2200 | 2200 | 0 | 0 | 0 |
| cfe0 | 2200 | 2200 | 0 | 0 | 0 |
| cfe_tn | 180 | 178 | 2 | 0 | 2020 |
| tnml | 180 | 176 | 4 | 0 | 2020 |
| **all** | 26760 | 26754 | 6 | 0 | 4040 |

## Series (noise, decoder, rounds): thresholds

| decoder | planned | finite-size fit | 2 largest d | at grid edge | drift | no threshold |
|---|---|---|---|---|---|---|
| mwpm | 220 | 176 | 15 | 2 | 26 | 1 |
| corr_links | 220 | 177 | 14 | 2 | 26 | 1 |
| corr_gauge | 220 | 178 | 13 | 1 | 27 | 1 |
| seq_soft | 220 | 182 | 10 | 0 | 28 | 0 |
| seq_erasure | 220 | 177 | 14 | 0 | 28 | 1 |
| seq_match | 220 | 182 | 21 | 0 | 17 | 0 |
| bm | 220 | 181 | 15 | 1 | 22 | 1 |
| bp_full | 220 | 175 | 15 | 4 | 21 | 5 |
| bp_corr | 220 | 179 | 14 | 1 | 20 | 6 |
| tesseract | 220 | 81 | 86 | 21 | 29 | 3 |
| cfe | 220 | 164 | 16 | 2 | 34 | 4 |
| cfe0 | 220 | 165 | 18 | 5 | 31 | 1 |
| cfe_tn | 50 | 9 | 11 | 0 | 0 | 30 |
| tnml | 50 | 8 | 11 | 1 | 1 | 29 |

## Planned instances without data: 0


## Planned series without a bracketed threshold: 123

- biased10 / cfe_tn: r=3 (no threshold), r=4 (no threshold), r=d (no threshold)
- biased10 / tesseract: r=12 (at grid edge), r=13 (at grid edge), r=14 (at grid edge)
- biased10 / tnml: r=4 (no threshold), r=d (no threshold)
- biased100 / cfe_tn: r=4 (no threshold), r=d (no threshold)
- biased100 / tesseract: r=2 (at grid edge)
- biased100 / tnml: r=3 (no threshold), r=4 (no threshold), r=d (no threshold)
- em3 / cfe0: r=17 (at grid edge), r=18 (at grid edge), r=19 (at grid edge), r=20 (at grid edge), r=21 (at grid edge)
- em3 / cfe_tn: r=3 (no threshold), r=4 (no threshold), r=d (no threshold)
- em3 / tnml: r=3 (no threshold), r=4 (no threshold), r=d (no threshold)
- h2_p / cfe_tn: r=2 (no threshold), r=3 (no threshold), r=4 (no threshold), r=d (no threshold)
- h2_p / corr_gauge: r=2 (at grid edge)
- h2_p / corr_links: r=16 (at grid edge)
- h2_p / mwpm: r=4 (at grid edge), r=8 (at grid edge)
- h2_p / tesseract: r=12 (at grid edge), r=14 (at grid edge), r=15 (at grid edge), r=19 (at grid edge), r=6 (at grid edge)
- h2_p / tnml: r=2 (no threshold), r=3 (no threshold), r=4 (no threshold), r=d (no threshold)
- h2_p_noxt / cfe_tn: r=3 (no threshold), r=4 (no threshold), r=d (no threshold)
- h2_p_noxt / tesseract: r=11 (at grid edge)
- h2_p_noxt / tnml: r=3 (no threshold), r=4 (no threshold), r=d (no threshold)
- helios_p / bm: r=16 (no threshold)
- helios_p / bp_corr: r=13 (no threshold), r=14 (no threshold), r=16 (no threshold), r=2 (no threshold), r=3 (no threshold), r=9 (no threshold)
- helios_p / bp_full: r=15 (no threshold), r=5 (no threshold), r=6 (no threshold)
- helios_p / cfe: r=15 (no threshold), r=2 (at grid edge), r=20 (no threshold), r=21 (no threshold), r=5 (at grid edge), r=9 (no threshold)
- helios_p / cfe0: r=18 (no threshold)
- helios_p / cfe_tn: r=2 (no threshold), r=3 (no threshold), r=4 (no threshold), r=d (no threshold)
- helios_p / corr_gauge: r=6 (no threshold)
- helios_p / corr_links: r=1 (no threshold)
- helios_p / mwpm: r=2 (no threshold)
- helios_p / seq_erasure: r=3 (no threshold)
- helios_p / tesseract: r=13 (no threshold), r=5 (no threshold), r=7 (no threshold)
- helios_p / tnml: r=1 (no threshold), r=2 (no threshold), r=3 (no threshold), r=4 (no threshold), r=d (no threshold)
- helios_p_noxt / cfe_tn: r=4 (no threshold), r=d (no threshold)
- helios_p_noxt / tnml: r=4 (no threshold), r=d (no threshold)
- purez / bm: r=1 (at grid edge)
- purez / bp_corr: r=1 (at grid edge)
- purez / bp_full: r=1 (at grid edge)
- purez / cfe_tn: r=3 (no threshold), r=4 (no threshold), r=d (no threshold)
- purez / corr_links: r=21 (at grid edge)
- purez / tesseract: r=1 (at grid edge), r=11 (at grid edge), r=2 (at grid edge), r=20 (at grid edge), r=9 (at grid edge)
- purez / tnml: r=4 (no threshold), r=d (no threshold)
- sd6 / bp_full: r=1 (no threshold), r=2 (no threshold), r=3 (at grid edge), r=4 (at grid edge), r=5 (at grid edge)
- sd6 / cfe_tn: r=4 (no threshold), r=d (no threshold)
- sd6 / tesseract: r=12 (at grid edge), r=20 (at grid edge)
- sd6 / tnml: r=2 (at grid edge), r=4 (no threshold), r=d (no threshold)
- si1000 / cfe_tn: r=2 (no threshold), r=3 (no threshold), r=4 (no threshold), r=d (no threshold)
- si1000 / tesseract: r=11 (at grid edge), r=13 (at grid edge), r=18 (at grid edge), r=2 (at grid edge)
- si1000 / tnml: r=2 (no threshold), r=4 (no threshold), r=d (no threshold)
