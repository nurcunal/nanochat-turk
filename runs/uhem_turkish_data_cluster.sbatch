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

CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$CODE_DIR/runs/uhem_d32_require_clean_code.sh"
d32_require_clean_committed_code "$CODE_DIR"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/ari/users/nunal/nanochat-turk-d32-general-v3}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
POLICY="${POLICY:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_v4.json}"
SAMPLE="${SAMPLE:-0}"
D32_PATH_CONTRACT="${D32_PATH_CONTRACT:-v4}"
case "$D32_PATH_CONTRACT" in
    v3|v4) source "$CODE_DIR/runs/uhem_d32_${D32_PATH_CONTRACT}_paths.sh" ;;
    *) echo "D32_PATH_CONTRACT must equal v3 or v4" >&2; exit 2 ;;
esac
CONTROL_DIR="${CONTROL_DIR:-$D32_CONTROL_DIR}"
d32_bind_canonical_path CONTROL_DIR "$D32_CONTROL_DIR"
SOURCE_PLAN="${SOURCE_PLAN:-$CONTROL_DIR/source_plan.json}"
CALIBRATION="${CALIBRATION:-$CONTROL_DIR/backend_calibration.json}"
d32_bind_canonical_path SOURCE_PLAN "$D32_SOURCE_PLAN"
d32_bind_canonical_path CALIBRATION "$D32_CALIBRATION"
if [ "$SAMPLE" = 1 ]; then
    DATA_RUN_DIR="${DATA_RUN_DIR:-$D32_SAMPLE_RUN_DIR}"
    d32_bind_canonical_path DATA_RUN_DIR "$D32_SAMPLE_RUN_DIR"
else
    DATA_RUN_DIR="${DATA_RUN_DIR:-$D32_DATA_RUN_DIR}"
    d32_bind_canonical_path DATA_RUN_DIR "$D32_DATA_RUN_DIR"
fi
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
    d32_bind_canonical_path RESOURCE_APPROVAL "$D32_RESOURCE_APPROVAL"
    d32_bind_canonical_path MIXTURE_QUALITY_APPROVAL "$D32_MIXTURE_QUALITY_APPROVAL"
    d32_bind_canonical_path DATA_PREP_STORAGE_GATE "$D32_DATA_PREP_STORAGE_GATE"
    d32_bind_canonical_path PACK_PLAN "$D32_PACK_PLAN"
    RECIPE="${RECIPE:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_wsd_v4.json}"
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
