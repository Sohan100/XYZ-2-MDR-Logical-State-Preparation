#!/bin/bash
# Submit the whole threshold campaign on Perlmutter: NODES array elements of run_campaign.sh and,
# after them, analyze_campaign.sh. Run from the repository root on a login node:
#
#   bash slurm/ft_mdr/submit_campaign.sh -A <project> [-n NODES] [-t HH:MM:SS] [-q regular]
#
# The task list data/campaign/tasks.jsonl is written on the first call (scripts/campaign.py tasks
# prints the cost estimate) and reused afterwards, so calling the script again with the same -n
# resumes an interrupted campaign.
set -euo pipefail
ACCOUNT=""
NODES=24
TIME="12:00:00"
QOS="regular"
while getopts "A:n:t:q:" opt; do
    case "${opt}" in
        A) ACCOUNT="${OPTARG}" ;;
        n) NODES="${OPTARG}" ;;
        t) TIME="${OPTARG}" ;;
        q) QOS="${OPTARG}" ;;
        *) echo "usage: $0 -A <project> [-n NODES] [-t HH:MM:SS] [-q QOS]" >&2; exit 2 ;;
    esac
done
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source slurm/ft_mdr/env.sh
mkdir -p logs data/campaign docs/data/campaign
if [ ! -f data/campaign/tasks.jsonl ]; then
    python scripts/campaign.py tasks --out data/campaign/tasks.jsonl
fi
if [ -f data/campaign/nchunks ] && [ "$(cat data/campaign/nchunks)" != "${NODES}" ]; then
    echo "warning: the campaign was started with $(cat data/campaign/nchunks) chunks; using that number" >&2
    NODES="$(cat data/campaign/nchunks)"
fi
echo "${NODES}" > data/campaign/nchunks
ACC=()
if [ -n "${ACCOUNT}" ]; then
    ACC=(-A "${ACCOUNT}")
fi
JID=$(sbatch --parsable "${ACC[@]}" -q "${QOS}" -t "${TIME}" --array="0-$((NODES - 1))" \
      --export=ALL,NCHUNKS="${NODES}" slurm/ft_mdr/run_campaign.sh)
AID=$(sbatch --parsable "${ACC[@]}" --dependency="afterany:${JID}" slurm/ft_mdr/analyze_campaign.sh)
echo "campaign array ${JID} (${NODES} nodes), analysis job ${AID} after it"
echo "progress: python scripts/campaign.py status data/campaign/tasks.jsonl \"data/campaign/points_*.csv\""
