#!/bin/sh
# HCM runs of the Quantinuum models used in the paper: rounds at the device point, scans of p/p0, H2, CB gates.
# The value passed to --noise helios, h2 and helios_cb is p/p0, the two-qubit error rate in units of the
# device point p0 (1e-3 for Helios, 1.875e-3 for H2). helios_plots.py converts it to p.
cd "$(dirname "$0")/../.."
R="python3 scripts/run_decoder_threshold_sweep.py --decoders corr_gauge --max-shots 2000000 --max-errors 400 --time-limit 90"
for r in 1 2 3 4 5 6 7; do
  $R --rounds $r --distances 3 5 7 --noise helios --values 1.0 --out docs/data/helios/helios_hcm_rounds.csv
done
$R --noise helios --distances 3 5 7 9 --values 0.5 0.75 1.0 1.25 1.5 --out docs/data/helios/helios_hcm.csv
$R --noise h2 --distances 3 5 7 --values 1.0 --out docs/data/helios/helios_hcm.csv
$R --noise h2 --rounds 1 --distances 3 5 7 --values 1.0 --out docs/data/helios/helios_hcm.csv
$R --noise helios_cb --distances 3 5 7 --values 1.0 --out docs/data/helios/helios_hcm.csv
$R --noise helios_cb --rounds 1 --distances 3 5 7 --values 1.0 --out docs/data/helios/helios_hcm.csv
echo ALLDONE
