#!/bin/sh
# Correlated matching for both codes (XZZX: PyMatching correlations; XYZ^2: HCM, corr_gauge), and the
# check that mwpm sees the same statistics under the paper's boundary conventions.
P=cloud/phen_2505_03691/seq_paper; O=$P/data/points.csv
R="python $P/run_points.py --shots 20000 --mwpm-shots 50000 --out $O"
$R --code xzzx --noise phen --bases X Z --decoders corr --values 0.035 0.0375 0.04 0.0425 0.045 0.0475
$R --code xzzx --noise phen_b10 --bases X Z --decoders corr --values 0.045 0.0475 0.05 0.0525 0.055 0.0575 0.06 0.0625
$R --code xyz2 --noise phen --bases X Y --decoders corr_gauge --values 0.0325 0.035 0.0375 0.04 0.0425 0.045
$R --code xyz2 --noise phen_b10 --bases X Y --decoders corr_gauge --values 0.04 0.0425 0.045 0.0475 0.05 0.0525 0.055
$R --code xyz2 --variant paper --noise phen --bases X Y --decoders mwpm --values 0.0275 0.03 0.0325 0.035 0.0375 0.04
