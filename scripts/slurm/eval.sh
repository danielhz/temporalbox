#!/usr/bin/env bash
# SLURM evaluation job — runs eval.py on a trained checkpoint.
#
# Usage:
#   sbatch scripts/slurm/eval.sh --checkpoint results/<run_id>/best.pt
#
#SBATCH --job-name=ltel-eval
#SBATCH --partition=slowlane
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

CODE_DIR="${HOME}/nesy-ltel-code"
VENV_DIR="${CODE_DIR}/.venv"

module load Python/3.11.3-GCCcore-12.3.0
source "${VENV_DIR}/bin/activate"

cd "${CODE_DIR}"

python eval.py "$@"
