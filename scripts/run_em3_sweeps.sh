#!/bin/sh
# Threshold sweeps under the EM3 model (pair-measurement extraction), r = d rounds.
# The published EM3 points are in docs/data/thresholds/sweeps_pooled.csv. To pool new
# runs with them, set XYZ2_EXTRA_SWEEPS=data/ft_mdr/em3_sweeps.csv before running the
# analysis scripts in paper/analysis.
cd "$(dirname "$0")/.."
OUT=${1:-data/ft_mdr/em3_sweeps.csv}
python3 scripts/run_decoder_threshold_sweep.py --noise em3 --decoders mwpm --distances 3 5 7 9 \
  --values 1.4e-3 1.6e-3 1.8e-3 1.9e-3 2.0e-3 2.1e-3 2.2e-3 2.3e-3 2.4e-3 2.6e-3 2.8e-3 3.0e-3 \
  --max-shots 2000000 --max-errors 600 --time-limit 90 --out $OUT
python3 scripts/run_decoder_threshold_sweep.py --noise em3 --decoders corr_gauge --distances 3 5 7 9 \
  --values 1.6e-3 1.8e-3 2.0e-3 2.2e-3 2.3e-3 2.4e-3 2.5e-3 2.6e-3 2.7e-3 2.8e-3 3.0e-3 3.4e-3 \
  --max-shots 2000000 --max-errors 400 --time-limit 90 --out $OUT
echo ALLDONE
