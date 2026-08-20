#!/bin/bash

# Submit with --array=0-13 after every selected object rank completed.

#SBATCH --job-name=tr-data-mh2
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -euo pipefail
: "${SOURCE_PLAN:?}"
: "${CALIBRATION:?}"
: "${DATA_RUN_DIR:?}"

CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
POLICY="${POLICY:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_v1.json}"
SAMPLE="${SAMPLE:-0}"
cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
args=(
    scripts/turkish_data_backend.py bucket
    --policy "$POLICY"
    --source-plan "$SOURCE_PLAN"
    --calibration "$CALIBRATION"
    --run-dir "$DATA_RUN_DIR"
    --rank "${SLURM_ARRAY_TASK_ID:?submit this file with --array=0-13}"
)
if [ "$SAMPLE" = 1 ]; then
    args+=(--sample)
else
    : "${RESOURCE_APPROVAL:?}"
    args+=(--resource-approval "$RESOURCE_APPROVAL")
fi
uv run --project environments/turkish-data --locked python "${args[@]}"
