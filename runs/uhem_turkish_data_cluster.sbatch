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
POLICY="${POLICY:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_v3.json}"
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
    : "${RESOURCE_APPROVAL:?Set accepted backend resource approval}"
    : "${MIXTURE_QUALITY_APPROVAL:?Set accepted manual quality approval}"
    : "${DATA_PREP_STORAGE_GATE:?Set passed post-safety storage gate}"
    : "${PACK_PLAN:?Set sealed production pack plan}"
    : "${OBJECT_NODE_RECEIPT_DIR:?Set packed production object node receipts}"
    : "${BUCKET_LAUNCH_RECEIPT:?Set packed production bucket launch receipt}"
    RECIPE="${RECIPE:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_wsd_v3.json}"
    CLUSTER_LAUNCH_ROOT="${CLUSTER_LAUNCH_ROOT:-$DATA_RUN_DIR/packed_production_cluster/job${SLURM_JOB_ID:?}}"
    CLUSTER_INPUT_RECEIPT="$CLUSTER_LAUNCH_ROOT/input_receipt.json"
    CLUSTER_LAUNCH_RECEIPT="$CLUSTER_LAUNCH_ROOT/launch_receipt.json"
    test ! -e "$CLUSTER_LAUNCH_ROOT"
    mkdir -p "$CLUSTER_LAUNCH_ROOT"
    "$UV_BIN" run --project environments/turkish-data --locked python \
        -m scripts.turkish_packed_production validate-gate \
        --recipe="$RECIPE" --policy="$POLICY" --source-plan="$SOURCE_PLAN" \
        --calibration="$CALIBRATION" --pack-plan="$PACK_PLAN" \
        --resource-approval="$RESOURCE_APPROVAL" \
        --mixture-quality-approval="$MIXTURE_QUALITY_APPROVAL" \
        --storage-gate="$DATA_PREP_STORAGE_GATE" \
        --write-dir="$DATA_RUN_DIR"
    "$UV_BIN" run --project environments/turkish-data --locked python \
        -m scripts.turkish_packed_production validate-cluster-inputs \
        --recipe="$RECIPE" --policy="$POLICY" --source-plan="$SOURCE_PLAN" \
        --calibration="$CALIBRATION" --pack-plan="$PACK_PLAN" \
        --resource-approval="$RESOURCE_APPROVAL" \
        --mixture-quality-approval="$MIXTURE_QUALITY_APPROVAL" \
        --storage-gate="$DATA_PREP_STORAGE_GATE" --run-dir="$DATA_RUN_DIR" \
        --object-node-receipt-dir="$OBJECT_NODE_RECEIPT_DIR" \
        --bucket-launch-receipt="$BUCKET_LAUNCH_RECEIPT" \
        --output="$CLUSTER_INPUT_RECEIPT"
    args+=(--resource-approval "$RESOURCE_APPROVAL")
fi
"$UV_BIN" run --project environments/turkish-data --locked python "${args[@]}"
if [ "$SAMPLE" != 1 ]; then
    "$UV_BIN" run --project environments/turkish-data --locked python \
        -m scripts.turkish_packed_production seal-cluster-launch \
        --recipe="$RECIPE" --policy="$POLICY" --source-plan="$SOURCE_PLAN" \
        --calibration="$CALIBRATION" --pack-plan="$PACK_PLAN" \
        --resource-approval="$RESOURCE_APPROVAL" \
        --mixture-quality-approval="$MIXTURE_QUALITY_APPROVAL" \
        --storage-gate="$DATA_PREP_STORAGE_GATE" --run-dir="$DATA_RUN_DIR" \
        --cluster-input-receipt="$CLUSTER_INPUT_RECEIPT" \
        --output="$CLUSTER_LAUNCH_RECEIPT"
    test -f "$CLUSTER_LAUNCH_RECEIPT"
fi
