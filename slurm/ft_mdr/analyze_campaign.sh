#!/bin/bash
#SBATCH --job-name=ftmdr_analysis
#SBATCH --output=logs/analysis_%j.out
#SBATCH -C cpu
#SBATCH -q preempt
#SBATCH --requeue
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256

# Merge the campaign CSVs, write the progress report, fit every threshold and make every figure.
# Runs after the campaign array (submit_campaign.sh adds the dependency); can also be run by hand
# on a login node: bash slurm/ft_mdr/analyze_campaign.sh

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source slurm/ft_mdr/env.sh
mkdir -p docs/data/campaign
python scripts/campaign.py merge "data/campaign/points_*.csv" --out docs/data/campaign/points.csv
: > docs/data/campaign/status.txt
for T in data/campaign/tasks*.jsonl; do
    echo "== ${T}" | tee -a docs/data/campaign/status.txt
    python scripts/campaign.py status "${T}" "data/campaign/points_*.csv" | tee -a docs/data/campaign/status.txt
done
python paper/analysis/campaign_report.py --workers "${REPORT_WORKERS:-64}"
