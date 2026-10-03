#!/bin/bash
# One-time setup on Perlmutter: a virtual environment in .venv with the package, the decoders
# (ldpc, tesseract-decoder) and pytest. Run from the repository root on a login node:
#   bash slurm/ft_mdr/setup_env.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
module load python >/dev/null 2>&1 || true
VENV="${XYZ2_VENV:-${REPO_ROOT}/.venv}"
python3 -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --upgrade pip wheel
python -m pip install -e ".[decoders,dev]"
python -c "import stim, pymatching, ldpc, tesseract_decoder, scipy, pandas, matplotlib; print('stim', stim.__version__, 'pymatching', pymatching.__version__, 'ldpc', ldpc.__version__)"
