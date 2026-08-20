#!/bin/bash

# Seal the exact production backend lineage, then materialize the filtered
# pool on the same BeeGFS tree covered by the post-safety storage gate.

#SBATCH --job-name=tr-pool-prod
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
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

: "${RESOURCE_APPROVAL:?Set the accepted resource approval}"
: "${MIXTURE_QUALITY_APPROVAL:?Set the accepted manual mixture-quality approval}"
: "${DATA_PREP_STORAGE_GATE:?Set the passed post-safety storage gate}"
: "${CLUSTER_LAUNCH_RECEIPT:?Set the completed production cluster launch receipt}"

CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/ari/users/nunal/nanochat-turk-d32-general-v2}"
RECIPE="${RECIPE:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_wsd_v2.json}"
POLICY="${POLICY:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_v2.json}"
CONTROL_DIR="${CONTROL_DIR:-$NANOCHAT_BASE_DIR/control/data_v2}"
SOURCE_PLAN="${SOURCE_PLAN:-$CONTROL_DIR/source_plan.json}"
CALIBRATION="${CALIBRATION:-$CONTROL_DIR/backend_calibration.json}"
PACK_PLAN="${PACK_PLAN:-$CONTROL_DIR/production_source_pack_plan.json}"
DATA_RUN_DIR="${DATA_RUN_DIR:-$NANOCHAT_BASE_DIR/data_backend/production_v2}"
SOURCE_RECEIPT="${SOURCE_RECEIPT:-$CONTROL_DIR/source_receipt.json}"
BACKEND_RECEIPT="${BACKEND_RECEIPT:-$CONTROL_DIR/backend_receipt.json}"
POOL_DIR="${POOL_DIR:-$NANOCHAT_BASE_DIR/data_v2/filtered_pool}"
PYTHON="${PYTHON:-$CODE_DIR/environments/turkish-data/.venv/bin/python}"

cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
for input in "$RECIPE" "$POLICY" "$SOURCE_PLAN" "$CALIBRATION" "$PACK_PLAN" \
    "$RESOURCE_APPROVAL" "$MIXTURE_QUALITY_APPROVAL" \
    "$DATA_PREP_STORAGE_GATE" "$CLUSTER_LAUNCH_RECEIPT"; do test -f "$input"; done
test -x "$PYTHON"
test ! -e "$SOURCE_RECEIPT"
test ! -e "$BACKEND_RECEIPT"
test ! -e "$POOL_DIR"
mkdir -p "$POOL_DIR"

common=(--recipe="$RECIPE" --policy="$POLICY" --source-plan="$SOURCE_PLAN"
    --calibration="$CALIBRATION" --pack-plan="$PACK_PLAN"
    --resource-approval="$RESOURCE_APPROVAL"
    --mixture-quality-approval="$MIXTURE_QUALITY_APPROVAL"
    --storage-gate="$DATA_PREP_STORAGE_GATE")

"$PYTHON" -m scripts.turkish_packed_production validate-gate \
    "${common[@]}" --write-dir="$POOL_DIR" \
    --cluster-launch-receipt="$CLUSTER_LAUNCH_RECEIPT"

"$PYTHON" scripts/turkish_data_backend.py seal-source \
    --policy "$POLICY" --source-plan "$SOURCE_PLAN" \
    --calibration "$CALIBRATION" --run-dir "$DATA_RUN_DIR" \
    --cluster-launch-receipt "$CLUSTER_LAUNCH_RECEIPT" \
    --output "$SOURCE_RECEIPT"

"$PYTHON" scripts/turkish_data_backend.py seal-backend \
    --policy "$POLICY" --source-plan "$SOURCE_PLAN" \
    --calibration "$CALIBRATION" --run-dir "$DATA_RUN_DIR" \
    --cluster-launch-receipt "$CLUSTER_LAUNCH_RECEIPT" \
    --source-receipt "$SOURCE_RECEIPT" --output "$BACKEND_RECEIPT"

"$PYTHON" scripts/build_turkish_pretrain_corpus.py \
    --policy "$POLICY" production-pool \
    --source-receipt "$SOURCE_RECEIPT" \
    --backend-receipt "$BACKEND_RECEIPT" \
    --cluster-launch-receipt "$CLUSTER_LAUNCH_RECEIPT" \
    --output-dir "$POOL_DIR"

test -f "$POOL_DIR/corpus_manifest.json"
echo "Verified production pool: $POOL_DIR"
