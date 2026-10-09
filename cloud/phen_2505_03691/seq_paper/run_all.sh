#!/bin/sh
# Refined scans around the pilot crossings (data/pilot.csv); 10^4 shots per point for BP/sequential
# decoders, 5x10^4 for mwpm. Run from the repository root with the venv active.
P=cloud/phen_2505_03691/seq_paper; O=$P/data/points.csv
R="python $P/run_points.py --shots 10000 --mwpm-shots 50000 --out $O"
# XYZ^2, our circuit (frame start, frame readout)
$R --code xyz2 --noise phen --bases X Y --decoders mwpm --values 0.0275 0.03 0.0325 0.035 0.0375 0.04
$R --code xyz2 --noise phen --bases X --decoders bm --values 0.035 0.0375 0.04 0.0425 0.045
$R --code xyz2 --noise phen --bases Y --decoders bm --values 0.0425 0.045 0.0475 0.05 0.0525
$R --code xyz2 --noise phen --bases X Y --decoders seq_soft --values 0.0325 0.035 0.0375 0.04 0.0425
$R --code xyz2 --noise phen_b10 --bases X Y --decoders mwpm --values 0.025 0.0275 0.03 0.0325 0.035 0.0375
$R --code xyz2 --noise phen_b10 --bases X --decoders bm --values 0.045 0.0475 0.05 0.0525 0.055 0.0575
$R --code xyz2 --noise phen_b10 --bases Y --decoders bm --values 0.05 0.0525 0.055 0.0575 0.06 0.0625
$R --code xyz2 --noise phen_b10 --bases X Y --decoders seq_soft --values 0.0375 0.04 0.0425 0.045 0.0475 0.05
# XZZX
$R --code xzzx --noise phen --bases X Z --decoders mwpm --values 0.0325 0.035 0.0375 0.04 0.0425 0.045
$R --code xzzx --noise phen --bases X Z --decoders bm --values 0.04 0.0425 0.045 0.0475 0.05 0.0525
$R --code xzzx --noise phen_b10 --bases X Z --decoders mwpm --values 0.045 0.0475 0.05 0.0525 0.055 0.0575 0.06 0.0625 0.065
$R --code xzzx --noise phen_b10 --bases X Z --decoders bm --values 0.045 0.0475 0.05 0.0525 0.055 0.0575 0.06
# XYZ^2, paper conventions (code-state start, perfect final round), paper's sequential decoder and bm
$R --code xyz2 --variant paper --noise phen --bases X Y --decoders seq_hard --values 0.0275 0.03 0.0325 0.035 0.0375
$R --code xyz2 --variant paper --noise phen --bases X Y --decoders bm --values 0.04 0.0425 0.045 0.0475 0.05
$R --code xyz2 --variant paper --noise phen_b10 --bases X Y --decoders seq_hard --values 0.0275 0.03 0.0325 0.035 0.0375
$R --code xyz2 --variant paper --noise phen_b10 --bases X Y --decoders bm --values 0.045 0.0475 0.05 0.0525 0.055 0.0575
