#!/bin/bash
# usage: q.sh noise_regex decoder_regex rounds
f=/pscratch/sd/s/sohan100/XYZ-2-MDR-Logical-State-Preparation/docs/data/campaign/thresholds.csv
awk -F, -v n="$1" -v d="$2" -v r="$3" 'NR>1 && $1 ~ ("^"n"$") && $2 ~ ("^"d"$") && $3==r {printf "%s %s r=%s %s pth=%.4f err=%.4f stat=%.4f syst=%.4f nu=%.3f chi2=%.2f d=%s-%s drift=%s flag=%s\n",$1,$2,$3,$4,$5*100,$6*100,$7*100,$8*100,$9,$10,$12,$13,$17,$18}' $f
