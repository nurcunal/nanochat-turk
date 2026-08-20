#!/bin/bash

# Single rank priority-cluster/quality/PII stage. The bounded sample must finish
# this stage before its resource report can be accepted.

#SBATCH --job-name=tr-data-cluster
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

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
POLICY="${POLICY:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_v1.json}"
SAMPLE="${SAMPLE:-0}"
cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
args=(
    scripts/turkish_data_backend.py cluster
    --policy "$POLICY"
    --source-plan "$SOURCE_PLAN"
    --calibration "$CALIBRATION"
    --run-dir "$DATA_RUN_DIR"
)
if [ "$SAMPLE" = 1 ]; then
    args+=(--sample)
else
    : "${RESOURCE_APPROVAL:?}"
    args+=(--resource-approval "$RESOURCE_APPROVAL")
fi
"$UV_BIN" run --project environments/turkish-data --locked python "${args[@]}"
