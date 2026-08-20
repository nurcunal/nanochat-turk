#!/bin/bash

# Submit only after resolve+calibration. For the bounded sample, pass the exact
# comma-separated ranks printed by `turkish_data_backend.py sample-ranks`.
# Full mode additionally requires RESOURCE_APPROVAL and should use 0-(N-1).

#SBATCH --job-name=tr-data-objects
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -eo pipefail

source /etc/profile.d/modules.sh
module use /ari/progs/modulefiles
module purge
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0
set -u

: "${SOURCE_PLAN:?Set SOURCE_PLAN to the sealed object plan}"
: "${CALIBRATION:?Set CALIBRATION to the passed backend calibration}"
: "${GLOTLID_MODEL:?Set GLOTLID_MODEL to the verified model_v3.bin}"
: "${DATA_RUN_DIR:?Set DATA_RUN_DIR to a dedicated sample or production run directory}"

CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
POLICY="${POLICY:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_v1.json}"
SAMPLE="${SAMPLE:-0}"
cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"

args=(
    scripts/turkish_data_backend.py process-object
    --policy "$POLICY"
    --source-plan "$SOURCE_PLAN"
    --calibration "$CALIBRATION"
    --model "$GLOTLID_MODEL"
    --run-dir "$DATA_RUN_DIR"
    --rank "${SLURM_ARRAY_TASK_ID:?submit this file with --array}"
    --scratch-dir "${SLURM_TMPDIR:-$DATA_RUN_DIR/.scratch}"
)
if [ "$SAMPLE" = 1 ]; then
    args+=(--sample)
else
    : "${RESOURCE_APPROVAL:?Full processing requires an accepted RESOURCE_APPROVAL}"
    args+=(--resource-approval "$RESOURCE_APPROVAL")
fi

"$UV_BIN" run --project environments/turkish-data --locked python "${args[@]}"
