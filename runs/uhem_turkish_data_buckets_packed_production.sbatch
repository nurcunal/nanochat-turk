#!/bin/bash

# Submit after the complete packed object-node array. All fourteen fixed
# MinHash bands share one exclusive cpu2dq node and are billed once.

#SBATCH --job-name=tr-data-prod-mh14
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=14
#SBATCH --ntasks-per-node=14
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -eo pipefail
source /etc/profile.d/modules.sh
module use /ari/progs/modulefiles
module purge
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0
set -u

: "${RESOURCE_APPROVAL:?Set the accepted resource approval}"
: "${MIXTURE_QUALITY_APPROVAL:?Set the accepted manual quality approval}"
: "${DATA_PREP_STORAGE_GATE:?Set the passed post-safety storage gate}"
CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/ari/users/nunal/nanochat-turk-d32-general-v3}"
RECIPE="${RECIPE:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_wsd_v3.json}"
POLICY="${POLICY:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_v3.json}"
CONTROL_DIR="${CONTROL_DIR:-$NANOCHAT_BASE_DIR/control/data_v3}"
SOURCE_PLAN="${SOURCE_PLAN:-$CONTROL_DIR/source_plan.json}"
CALIBRATION="${CALIBRATION:-$CONTROL_DIR/backend_calibration.json}"
PACK_PLAN="${PACK_PLAN:-$CONTROL_DIR/production_source_pack_plan.json}"
DATA_RUN_DIR="${DATA_RUN_DIR:-$NANOCHAT_BASE_DIR/data_backend/production_v3}"
PACK_ROOT="${PACK_ROOT:-$DATA_RUN_DIR/packed_production_objects}"
OBJECT_NODE_RECEIPT_DIR="${OBJECT_NODE_RECEIPT_DIR:-$PACK_ROOT/node_receipts}"
LAUNCH_ROOT="$DATA_RUN_DIR/packed_production_buckets/job${SLURM_JOB_ID:?}"
TASK_RECEIPT_DIR="$LAUNCH_ROOT/tasks"
LAUNCH_RECEIPT="$LAUNCH_ROOT/launch_receipt.json"
PYTHON="${PYTHON:-$CODE_DIR/environments/turkish-data/.venv/bin/python}"

cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
for input in "$RECIPE" "$POLICY" "$SOURCE_PLAN" "$CALIBRATION" "$PACK_PLAN" \
    "$RESOURCE_APPROVAL" "$MIXTURE_QUALITY_APPROVAL" "$DATA_PREP_STORAGE_GATE"; do
    test -f "$input"
done
test -x "$PYTHON"
test -d "$OBJECT_NODE_RECEIPT_DIR"
test ! -e "$LAUNCH_ROOT"
mkdir -p "$DATA_RUN_DIR" "$TASK_RECEIPT_DIR"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
common=(--recipe="$RECIPE" --policy="$POLICY" --source-plan="$SOURCE_PLAN"
    --calibration="$CALIBRATION" --pack-plan="$PACK_PLAN"
    --resource-approval="$RESOURCE_APPROVAL"
    --mixture-quality-approval="$MIXTURE_QUALITY_APPROVAL"
    --storage-gate="$DATA_PREP_STORAGE_GATE")

"$PYTHON" -m scripts.turkish_packed_production validate-gate \
    "${common[@]}" --write-dir="$DATA_RUN_DIR"

set +e
srun --nodes=1 --ntasks=14 --ntasks-per-node=14 --cpus-per-task=8 \
    --kill-on-bad-exit=1 --output="$LAUNCH_ROOT/bucket-%t.out" \
    --error="$LAUNCH_ROOT/bucket-%t.err" \
    "$PYTHON" -m scripts.turkish_packed_production run-bucket \
    "${common[@]}" --run-dir="$DATA_RUN_DIR" \
    --object-node-receipt-dir="$OBJECT_NODE_RECEIPT_DIR" \
    --receipt-dir="$TASK_RECEIPT_DIR"
srun_rc=$?
set -e
if [ "$srun_rc" -ne 0 ]; then exit "$srun_rc"; fi

"$PYTHON" -m scripts.turkish_packed_production finalize-buckets \
    "${common[@]}" --run-dir="$DATA_RUN_DIR" \
    --object-node-receipt-dir="$OBJECT_NODE_RECEIPT_DIR" \
    --receipt-dir="$TASK_RECEIPT_DIR" --output="$LAUNCH_RECEIPT"
test -f "$LAUNCH_RECEIPT"
echo "Verified packed production MinHash launch: $LAUNCH_RECEIPT"
