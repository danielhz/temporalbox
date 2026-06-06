#!/usr/bin/env bash
# Set up the Python virtual environment on aisa.
# Run once from the aisa login node after the first sync-push:
#   bash ~/nesy-ltel-code/scripts/setup_env.sh
set -euo pipefail

CODE_DIR="${HOME}/nesy-ltel-code"
VENV_DIR="${CODE_DIR}/.venv"

module load Python/3.11.3-GCCcore-12.3.0

if [ ! -d "${VENV_DIR}" ]; then
    echo "[setup] Creating virtualenv at ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
pip install --upgrade pip --quiet

echo "[setup] Installing PyTorch (CUDA 12.1)"
pip install torch --index-url https://download.pytorch.org/whl/cu121 --quiet

echo "[setup] Installing ltel-box package"
pip install -e "${CODE_DIR}" --quiet

echo "[setup] Done. Verify with: python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
