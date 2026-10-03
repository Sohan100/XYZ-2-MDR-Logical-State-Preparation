# Environment of the campaign jobs on Perlmutter (sourced by the scripts in this folder).
# The virtual environment is made once by slurm/ft_mdr/setup_env.sh.
module load python >/dev/null 2>&1 || true
VENV="${XYZ2_VENV:-${REPO_ROOT:-$(pwd)}/.venv}"
if [ -f "${VENV}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
fi
# one thread per process: the parallelism is over tasks
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
