#!/usr/bin/env bash
# SLURM GPU job for the offices feasibility slice (Option A, event ticks).
#
#   sbatch scripts/slurm/slice.sh data-slice results/slice
#
#SBATCH --job-name=ltel-slice
#SBATCH --partition=fastlane
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
CODE_DIR="${HOME}/nesy-ltel-code"
DATASET="${1:-$CODE_DIR/data-slice}"
OUT="${2:-$CODE_DIR/results/slice}"

echo "[slice] node $(hostname)  job ${SLURM_JOB_ID}  gpu ${SLURM_JOB_GPUS:-?}"
module load Python/3.11.3-GCCcore-12.3.0
source "$CODE_DIR/.venv/bin/activate"
cd "$CODE_DIR"

python scripts/slice_offices.py \
    --dataset-dir "$DATASET" \
    --dim 64 --epochs 4000 --lr 2e-2 --offset-reg 1e-3 \
    --eps 1e-3 --device cuda --out "$OUT"
