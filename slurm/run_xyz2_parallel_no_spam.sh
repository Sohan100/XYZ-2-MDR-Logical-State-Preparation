#!/bin/bash
#SBATCH --job-name=xyz2_mdr_no_spam
#SBATCH --output=xyz2_mdr_no_spam_%j.out
#SBATCH --error=xyz2_mdr_no_spam_%j.err
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 47:30:00
#SBATCH --nodes=18
#SBATCH --ntasks=18
#SBATCH --cpus-per-task=256

set -euo pipefail

# --- BEGIN USER CONFIGURABLE SECTION ---
DISTANCES=(3 5 7 9 11)
NOISE_MODELS=(z_type pure_z unbiased)
NUM_SHOTS=3000
NUM_REPLICATES=30
RECOVERY_MODE="each_round"
ROOT_DIR="${PWD}/XYZ2-experiment-data-slurm"
SCRIPTS_DIR="scripts"
RESULTS_COPY_DIR="${PWD}/data/simulation_results"
TABLES_COPY_DIR="${PWD}/data/tables"
# --- END USER CONFIGURABLE SECTION ---

EXPECTED_CODE_NAME="xyz2_mdr"
P_SPAM=0.0

PROBABILITIES=(
1e-05 1.44543977075e-05 2.08929613085e-05 3.0199517204e-05 4.3651583224e-05 \
6.3095734448e-05 9.12010839356e-05 0.000131825673856 0.000190546071796 \
0.000275422870334 0.000398107170553 0.000575439937337 0.000831763771103 \
0.00120226443462 0.00173780082875 0.00251188643151 0.0036307805477 \
0.0052480746025 0.00758577575029 0.0109647819614 0.0158489319246 \
0.0229086765277 0.0331131121483 0.0478630092323 0.0691830970919 0.1 0.2 0.5 1
)
NUM_PROBS=${#PROBABILITIES[@]}

module load python/3.11

export OMP_NUM_THREADS=128
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

TIMESTAMP="$(date -u +%Y%m%d-%H%M%SZ)"

for NOISE_MODEL in "${NOISE_MODELS[@]}"; do
    for DISTANCE in "${DISTANCES[@]}"; do
        RUN_NAME="Run-${TIMESTAMP}-d${DISTANCE}-${NOISE_MODEL}-no-spam"

        python3.11 "${SCRIPTS_DIR}/setup_slurm_run.py" \
            --distance "${DISTANCE}" \
            --noise-model "${NOISE_MODEL}" \
            --run-name "${RUN_NAME}" \
            --root-dir "${ROOT_DIR}" \
            --shots "${NUM_SHOTS}" \
            --num-replicates "${NUM_REPLICATES}" \
            --p-spam "${P_SPAM}" \
            --recovery-mode "${RECOVERY_MODE}" \
            --probabilities "${PROBABILITIES[@]}"

        WORKDIR="${ROOT_DIR}/${RUN_NAME}"
        LOG_DIR="${WORKDIR}/logs_job_${SLURM_JOB_ID}"
        CODE_NAME_FILE="${WORKDIR}/code_name.txt"
        SHOTS_FILE="${WORKDIR}/shots.txt"
        CONFIG_FILE="${WORKDIR}/run_config.json"

        mkdir -p "${LOG_DIR}"

        if [[ ! -f "${CODE_NAME_FILE}" ]]; then
            echo "Missing ${CODE_NAME_FILE}"
            exit 1
        fi

        if [[ ! -f "${CONFIG_FILE}" ]]; then
            echo "Missing ${CONFIG_FILE}"
            exit 1
        fi

        ACTUAL_CODE_NAME="$(tr -d '[:space:]' < "${CODE_NAME_FILE}")"
        if [[ "${ACTUAL_CODE_NAME}" != "${EXPECTED_CODE_NAME}" ]]; then
            echo "Expected code '${EXPECTED_CODE_NAME}' but found"
            echo "'${ACTUAL_CODE_NAME}' in ${CODE_NAME_FILE}"
            exit 1
        fi

        echo "${NUM_SHOTS}" > "${SHOTS_FILE}"

        echo "Run Name: ${RUN_NAME}"
        echo "Code Name: ${ACTUAL_CODE_NAME}"
        echo "Noise Model: ${NOISE_MODEL}"
        echo "Distance: ${DISTANCE}"
        echo "p_spam: ${P_SPAM}"
        echo "Shots: ${NUM_SHOTS}"
        echo "Replicates: ${NUM_REPLICATES}"
        echo "Recovery mode: ${RECOVERY_MODE}"
        echo "Number of probabilities: ${NUM_PROBS}"
        echo "Logs: ${LOG_DIR}"

        for idx in $(seq 0 $((${NUM_PROBS} - 1))); do
            prob_val_for_log="${PROBABILITIES[idx]}"
            echo "Launching idx ${idx} (p ~ ${prob_val_for_log})"
            srun --exclusive --nodes=1 --ntasks=1 \
                --cpus-per-task=${SLURM_CPUS_PER_TASK} \
                python3.11 "${SCRIPTS_DIR}/run_slurm_experiment.py" \
                "${RUN_NAME}" "${idx}" \
                --root-dir "${ROOT_DIR}" \
                > "${LOG_DIR}/run_p_idx${idx}_val${prob_val_for_log}.log" 2>&1 &
        done

        wait
        python3.11 "${SCRIPTS_DIR}/merge_slurm_results.py" \
            "${RUN_NAME}" \
            --root-dir "${ROOT_DIR}" \
            --copy-to "${RESULTS_COPY_DIR}" \
            --tables-copy-to "${TABLES_COPY_DIR}" \
            > "${LOG_DIR}/merge_results.log" 2>&1
    done
done

echo "Job ${SLURM_JOB_ID} completed successfully."
