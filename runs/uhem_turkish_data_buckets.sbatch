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

set -eo pipefail
source /etc/profile.d/modules.sh
module use /ari/progs/modulefiles
module purge
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0
set -u
: "${SOURCE_PLAN:?}"
: "${CALIBRATION:?}"
: "${DATA_RUN_DIR:?}"

CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
POLICY="${POLICY:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_v2.json}"
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
"$UV_BIN" run --project environments/turkish-data --locked python "${args[@]}"
