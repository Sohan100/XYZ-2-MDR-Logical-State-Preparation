#!/bin/bash
#SBATCH --job-name=xyz2_no_spam
#SBATCH --output=xyz2_no_spam_%A_%a.out
#SBATCH --error=xyz2_no_spam_%A_%a.err
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 47:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --array=0-434

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# This is a Slurm array over:
#   3 noise models x 5 distances x 29 probabilities = 435 tasks.
# Update the --array range if the grid below changes.

# --- BEGIN USER CONFIGURABLE SECTION ---
DISTANCES=(3 5 7 9 11)
NOISE_MODELS=(z_type pure_z unbiased)
NUM_SHOTS=3000
NUM_REPLICATES=30
ROUNDS=(1 5 10)
P_SPAM=0.0
PREP_MODE="${PREP_MODE:-full_mdr}"
RECOVERY_MODE="${RECOVERY_MODE:-each_round}"
CORRECTION_MODE="${CORRECTION_MODE:-physical}"
RUN_SUFFIX="no-spam"
PYTHON_BIN="python3.11"
ROOT_DIR="${REPO_ROOT}/XYZ2-experiment-data-slurm"
SCRIPTS_DIR="${REPO_ROOT}/scripts"
RESULTS_COPY_DIR="${REPO_ROOT}/data/simulation_results"
TABLES_COPY_DIR="${REPO_ROOT}/data/tables"
PLOTS_OUTPUT_DIR="${REPO_ROOT}/data/plots"
# --- END USER CONFIGURABLE SECTION ---

CODE_FAMILY="xyz2"
EXPECTED_CODE_NAME="xyz2"

PROBABILITIES=(
1e-05 1.44543977075e-05 2.08929613085e-05 3.0199517204e-05 4.3651583224e-05 \
6.3095734448e-05 9.12010839356e-05 0.000131825673856 0.000190546071796 \
0.000275422870334 0.000398107170553 0.000575439937337 0.000831763771103 \
0.00120226443462 0.00173780082875 0.00251188643151 0.0036307805477 \
0.0052480746025 0.00758577575029 0.0109647819614 0.0158489319246 \
0.0229086765277 0.0331131121483 0.0478630092323 0.0691830970919 0.1 0.2 0.5 1
)

module load python/3.11

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-128}"
export OMP_PLACES="${OMP_PLACES:-threads}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"

RUN_TAG="${RUN_TAG:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}}"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:?This script must be submitted as a Slurm array.}"

"${PYTHON_BIN}" "${SCRIPTS_DIR}/run_slurm_array_task.py" \
    --task-index "${TASK_INDEX}" \
    --run-tag "${RUN_TAG}" \
    --code-family "${CODE_FAMILY}" \
    --code-name "${EXPECTED_CODE_NAME}" \
    --run-suffix "${RUN_SUFFIX}" \
    --root-dir "${ROOT_DIR}" \
    --scripts-dir "${SCRIPTS_DIR}" \
    --results-copy-dir "${RESULTS_COPY_DIR}" \
    --tables-copy-dir "${TABLES_COPY_DIR}" \
    --plots-output-dir "${PLOTS_OUTPUT_DIR}" \
    --distances "${DISTANCES[@]}" \
    --noise-models "${NOISE_MODELS[@]}" \
    --probabilities "${PROBABILITIES[@]}" \
    --rounds "${ROUNDS[@]}" \
    --shots "${NUM_SHOTS}" \
    --num-replicates "${NUM_REPLICATES}" \
    --p-spam "${P_SPAM}" \
    --prep-mode "${PREP_MODE}" \
    --recovery-mode "${RECOVERY_MODE}" \
    --correction-mode "${CORRECTION_MODE}"
