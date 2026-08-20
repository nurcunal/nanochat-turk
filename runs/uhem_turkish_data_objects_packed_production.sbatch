#!/bin/bash

# Submit as --array=0-(PRODUCTION_DATA_NODES-1). Each array allocation owns one
# cpu2dq node and executes exactly 32 serial object lanes from the sealed plan.

#SBATCH --job-name=tr-data-prod32
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

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
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/ari/users/nunal/nanochat-turk-d32-general-v2}"
RECIPE="${RECIPE:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_wsd_v2.json}"
POLICY="${POLICY:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_v2.json}"
CONTROL_DIR="${CONTROL_DIR:-$NANOCHAT_BASE_DIR/control/data_v2}"
SOURCE_PLAN="${SOURCE_PLAN:-$CONTROL_DIR/source_plan.json}"
CALIBRATION="${CALIBRATION:-$CONTROL_DIR/backend_calibration.json}"
PACK_PLAN="${PACK_PLAN:-$CONTROL_DIR/production_source_pack_plan.json}"
GLOTLID_MODEL="${GLOTLID_MODEL:-$NANOCHAT_BASE_DIR/source_models/glotlid-v3/model_v3.bin}"
DATA_RUN_DIR="${DATA_RUN_DIR:-$NANOCHAT_BASE_DIR/data_backend/production_v2}"
PACK_ROOT="${PACK_ROOT:-$DATA_RUN_DIR/packed_production_objects}"
NODE_INDEX="${SLURM_ARRAY_TASK_ID:?submit with the exact zero-based node array}"
ALLOCATION_ID="${SLURM_ARRAY_JOB_ID:?}_${NODE_INDEX}"
LAUNCH_ROOT="$PACK_ROOT/launches/job$ALLOCATION_ID"
LANE_RECEIPT_DIR="$LAUNCH_ROOT/lanes"
NODE_RECEIPT_DIR="$PACK_ROOT/node_receipts"
NODE_RECEIPT="$NODE_RECEIPT_DIR/node_$(printf '%05d' "$NODE_INDEX").json"
# UHeM node-local TmpDisk is 2GB and SLURM_TMPDIR is not provided. Production
# staging must stay on an explicitly provisioned BeeGFS path.
BEEGFS_SCRATCH="${BEEGFS_SCRATCH:-$DATA_RUN_DIR/.scratch/job$ALLOCATION_ID}"
PYTHON="${PYTHON:-$CODE_DIR/environments/turkish-data/.venv/bin/python}"

cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
for input in "$RECIPE" "$POLICY" "$SOURCE_PLAN" "$CALIBRATION" "$PACK_PLAN" \
    "$GLOTLID_MODEL" "$RESOURCE_APPROVAL" "$MIXTURE_QUALITY_APPROVAL" \
    "$DATA_PREP_STORAGE_GATE"; do test -f "$input"; done
test -x "$PYTHON"
test ! -e "$LAUNCH_ROOT"
test ! -e "$NODE_RECEIPT"
mkdir -p "$DATA_RUN_DIR" "$LANE_RECEIPT_DIR" "$NODE_RECEIPT_DIR" "$BEEGFS_SCRATCH"

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
srun --nodes=1 --ntasks=32 --ntasks-per-node=32 --cpus-per-task=4 \
    --kill-on-bad-exit=1 --output="$LAUNCH_ROOT/lane-%t.out" \
    --error="$LAUNCH_ROOT/lane-%t.err" \
    "$PYTHON" -m scripts.turkish_packed_production run-object-lane \
    "${common[@]}" --model="$GLOTLID_MODEL" --run-dir="$DATA_RUN_DIR" \
    --receipt-dir="$LANE_RECEIPT_DIR" --scratch-dir="$BEEGFS_SCRATCH"
srun_rc=$?
set -e
if [ "$srun_rc" -ne 0 ]; then exit "$srun_rc"; fi

"$PYTHON" -m scripts.turkish_packed_production finalize-object-node \
    "${common[@]}" --run-dir="$DATA_RUN_DIR" --receipt-dir="$LANE_RECEIPT_DIR" \
    --node-index="$NODE_INDEX" --output="$NODE_RECEIPT"
test -f "$NODE_RECEIPT"
echo "Verified packed production object node: $NODE_RECEIPT"
