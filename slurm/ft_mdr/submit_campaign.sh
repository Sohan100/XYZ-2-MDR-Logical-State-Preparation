#!/bin/bash
# Submit the whole threshold campaign on Perlmutter: NODES array elements of run_campaign.sh and,
# after them, analyze_campaign.sh. Run from the repository root on a login node:
#
#   bash slurm/ft_mdr/submit_campaign.sh -A <project> [-n NODES] [-t HH:MM:SS] [-q preempt]
#                                        [-T data/campaign/tasks.jsonl] [-P points]
#
# The task list (-T, default data/campaign/tasks.jsonl) is written on the first call if it does not
# exist (scripts/campaign.py tasks prints the cost estimate) and reused afterwards, so calling the
# script again with the same -n resumes an interrupted campaign. A second stage uses its own task
# file and prefix, e.g. -T data/campaign/tasks2.jsonl -P points_s2 (see docs/nersc_campaign.md).
set -euo pipefail
ACCOUNT=""
NODES=24
TIME="12:00:00"
QOS="preempt"
TASKS="data/campaign/tasks.jsonl"
PREFIX="points"
while getopts "A:n:t:q:T:P:" opt; do
    case "${opt}" in
        A) ACCOUNT="${OPTARG}" ;;
        n) NODES="${OPTARG}" ;;
        t) TIME="${OPTARG}" ;;
        q) QOS="${OPTARG}" ;;
        T) TASKS="${OPTARG}" ;;
        P) PREFIX="${OPTARG}" ;;
        *) echo "usage: $0 -A <project> [-n NODES] [-t HH:MM:SS] [-q QOS]" >&2; exit 2 ;;
    esac
done
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source slurm/ft_mdr/env.sh
mkdir -p logs data/campaign docs/data/campaign
if [ ! -f "${TASKS}" ]; then
    python scripts/campaign.py tasks --out "${TASKS}"
fi
NFILE="${TASKS%.jsonl}.nchunks"
if [ -f "${NFILE}" ] && [ "$(cat "${NFILE}")" != "${NODES}" ]; then
    echo "warning: ${TASKS} was started with $(cat "${NFILE}") chunks; using that number" >&2
    NODES="$(cat "${NFILE}")"
fi
echo "${NODES}" > "${NFILE}"
ACC=()
if [ -n "${ACCOUNT}" ]; then
    ACC=(-A "${ACCOUNT}")
fi
# --requeue: a job of the preempt QOS that is preempted (after its first two hours) goes back to the
# queue instead of being cancelled; the runner then continues every task from its last flushed counts.
JID=$(sbatch --parsable "${ACC[@]}" -q "${QOS}" --requeue -t "${TIME}" --array="0-$((NODES - 1))" \
      --export=ALL,NCHUNKS="${NODES}",TASKS="${TASKS}",PREFIX="${PREFIX}" slurm/ft_mdr/run_campaign.sh)
AID=$(sbatch --parsable "${ACC[@]}" -q "${QOS}" --requeue --dependency="afterany:${JID}" \
      slurm/ft_mdr/analyze_campaign.sh)
echo "campaign array ${JID} (${NODES} nodes), analysis job ${AID} after it"
echo "progress: python scripts/campaign.py status ${TASKS} \"data/campaign/points_*.csv\""
