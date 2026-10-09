#!/bin/sh
# Extensions of run_all.sh where a crossing fell outside the first window.
P=cloud/phen_2505_03691/seq_paper; O=$P/data/points.csv
R="python $P/run_points.py --shots 10000 --mwpm-shots 50000 --out $O"
$R --code xyz2 --noise phen --bases Y --decoders bm --values 0.0375 0.04
$R --code xyz2 --noise phen --bases X --decoders bm --values 0.0325
$R --code xyz2 --noise phen --bases Y --decoders seq_soft --values 0.045 0.0475
$R --code xyz2 --noise phen_b10 --bases X --decoders seq_soft --values 0.0325 0.035
$R --code xyz2 --variant paper --noise phen --bases X Y --decoders bm --values 0.0325 0.035 0.0375
$R --code xyz2 --variant paper --noise phen --bases X --decoders seq_hard --values 0.02 0.0225 0.025
$R --code xyz2 --variant paper --noise phen_b10 --bases Y --decoders bm --values 0.06 0.0625
$R --code xyz2 --variant paper --noise phen_b10 --bases X Y --decoders seq_hard --values 0.04 0.0425
$R --code xzzx --noise phen_b10 --bases X --decoders bm --values 0.0625 0.065 0.0675 0.07
$R --code xzzx --noise phen_b10 --bases Z --decoders bm --values 0.04 0.0425
$R --code xzzx --noise phen_b10 --bases Z --decoders mwpm --values 0.04 0.0425
$R --code xzzx --noise phen_b10 --bases X --decoders mwpm --values 0.0675 0.07
$R --code xzzx --noise phen --bases Z --decoders mwpm --values 0.0275 0.03
