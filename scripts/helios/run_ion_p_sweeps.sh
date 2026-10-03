#!/bin/sh
# Threshold sweeps of the one-parameter Helios and H2 models (CircuitNoise.trapped_ion),
# r = d rounds, MWPM and HCM, dense around the crossing points; also without crosstalk.
cd "$(dirname "$0")/../.."
OUT=docs/data/helios/ion_p_sweeps.csv
M="python3 scripts/run_decoder_threshold_sweep.py --decoders mwpm --distances 3 5 7 9 --max-shots 4000000 --max-errors 600 --time-limit 60 --out $OUT"
H="python3 scripts/run_decoder_threshold_sweep.py --decoders corr_gauge --distances 3 5 7 9 --max-shots 2000000 --max-errors 400 --time-limit 60 --out $OUT"
$M --noise helios_p --values 6e-4 7e-4 8e-4 8.5e-4 9e-4 9.5e-4 1.05e-3 1.1e-3 1.15e-3 1.2e-3 1.3e-3 1.4e-3 1.6e-3
$M --noise h2_p --values 1.2e-3 1.4e-3 1.6e-3 1.8e-3 2.0e-3 2.1e-3 2.2e-3 2.3e-3 2.4e-3 2.5e-3 2.6e-3 2.8e-3 3.0e-3 3.2e-3
$H --noise helios_p --values 7e-4 8e-4 9e-4 1.1e-3 1.2e-3 1.3e-3 1.4e-3
$H --noise h2_p --values 1.4e-3 1.6e-3 1.8e-3 2.0e-3 2.2e-3 2.4e-3 2.6e-3 2.8e-3 3.0e-3
$M --noise helios_p_noxt --values 2.5e-3 3e-3 3.5e-3 4e-3 4.5e-3 5e-3 5.5e-3 6e-3
$M --noise h2_p_noxt --values 2.5e-3 3e-3 3.5e-3 4e-3 4.5e-3 5e-3 5.5e-3 6e-3
echo ALLDONE
