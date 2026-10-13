#!/bin/bash
#SBATCH --job-name=ftmdr_pool_gpu
# GPU-node variant of run_pool.sh: the pool is CPU-only, but GPU nodes (64 cores, 256 GB) on the m4980_g
# allocation are another queue. Submit with -A m4980_g.
#SBATCH --output=logs/pool_%j.out
#SBATCH -C gpu
#SBATCH -q preempt
#SBATCH --requeue
#SBATCH -t 48:00:00
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --no-kill

# A pool job of the threshold campaign: every node runs `scripts/campaign.py pool-run`, which takes
# units of the shared work pool (data/campaign/pool<k>.jsonl, written by `campaign.py pool`) until
# none is left, one task per physical core within the memory cap. Units are claimed through files
# next to the pool, so any number of pool jobs can run at once, and a unit whose node stops is taken
# over by another node a quarter of an hour later. Counts go to data/campaign/points_<pool>_u<unit>.csv.
#
# Perlmutter starts only two pending jobs per user towards a node reservation (MaxJobsAccruePU = 2,
# about a day of age), so two big jobs of the largest preempt size (128 nodes) get far more node-hours
# than an array of one-node jobs, which run only in backfill holes. Submit with
#   sbatch -A m4980 --export=ALL,POOL=data/campaign/pool1.jsonl slurm/ft_mdr/run_pool.sh

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_ROOT}"
# shellcheck disable=SC1091
source slurm/ft_mdr/env.sh
POOL="${POOL:-data/campaign/pool1.jsonl}"
srun -N "${SLURM_NNODES}" --ntasks-per-node=1 -c 128 --cpu-bind=none --kill-on-bad-exit=0 --no-kill \
    --output="logs/pool_%j_%t.out" \
    python scripts/campaign.py pool-run "${POOL}" --workers "${WORKERS:-64}" --mem-gb "${MEM_GB:-0}"
