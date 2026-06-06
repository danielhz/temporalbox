#!/usr/bin/env bash
# SLURM training job for ltel-box experiments.
#
# Typical submission:
#   sbatch scripts/slurm/train.sh --config configs/synthetic_small.json
#
# To request a specific GPU type:
#   sbatch --constraint=A100 scripts/slurm/train.sh ...
#
#SBATCH --job-name=ltel-box
#SBATCH --partition=slowlane
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

CODE_DIR="${HOME}/nesy-ltel-code"
VENV_DIR="${CODE_DIR}/.venv"

echo "[job] Node: $(hostname)"
echo "[job] SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "[job] GPUs: ${SLURM_JOB_GPUS:-unset}"

module load Python/3.11.3-GCCcore-12.3.0
source "${VENV_DIR}/bin/activate"

cd "${CODE_DIR}"

python train.py "$@"
