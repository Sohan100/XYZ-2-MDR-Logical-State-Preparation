#!/bin/bash
#SBATCH --job-name=ftmdr_campaign
#SBATCH --output=logs/campaign_%A_%a.out
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --exclusive

# One element of the threshold campaign (scripts/campaign.py): runs chunk SLURM_ARRAY_TASK_ID of
# NCHUNKS of data/campaign/tasks.jsonl on one CPU node, one task per physical core, and appends
# the counts to data/campaign/points_<chunk>.csv every few minutes. A chunk that hits the time
# limit continues where it stopped when the same array is submitted again (same NCHUNKS).
# Submit with slurm/ft_mdr/submit_campaign.sh, which sets NCHUNKS.

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source slurm/ft_mdr/env.sh
: "${NCHUNKS:?NCHUNKS is not set; submit with slurm/ft_mdr/submit_campaign.sh}"
TASKS="${TASKS:-data/campaign/tasks.jsonl}"
python scripts/campaign.py run "${TASKS}" "data/campaign/${PREFIX:-points}_${SLURM_ARRAY_TASK_ID}.csv" \
    --chunk "${SLURM_ARRAY_TASK_ID}" --nchunks "${NCHUNKS}" --workers "${WORKERS:-128}"
