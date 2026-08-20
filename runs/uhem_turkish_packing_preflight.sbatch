#!/bin/bash

# Bounded, tokenizer-aware crop-retention calibration. This never materializes
# the full training corpus and never auto-approves its recommendation.

#SBATCH --job-name=tr-pack-plan
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -eo pipefail
source /etc/profile.d/modules.sh
module use /ari/progs/modulefiles
module purge
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0
set -u
: "${POOL_DIR:?Set POOL_DIR to the manually QA-approved production pool}"
: "${TOKENIZER_DIR:?Set TOKENIZER_DIR to the verified raw-BPE package}"
: "${PACKING_CONTROL_DIR:?Set PACKING_CONTROL_DIR to a new control directory}"
: "${RESOURCE_APPROVAL:?}"
: "${MIXTURE_QUALITY_APPROVAL:?}"
: "${DATA_PREP_STORAGE_GATE:?}"
: "${CLUSTER_LAUNCH_RECEIPT:?}"

CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/ari/users/nunal/nanochat-turk-d32-general-v3}"
POLICY="${POLICY:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_v3.json}"
RECIPE="${RECIPE:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_wsd_v3.json}"
CONTROL_DIR="${CONTROL_DIR:-$NANOCHAT_BASE_DIR/control/data_v3}"
SOURCE_PLAN="${SOURCE_PLAN:-$CONTROL_DIR/source_plan.json}"
CALIBRATION="${CALIBRATION:-$CONTROL_DIR/backend_calibration.json}"
PACK_PLAN="${PACK_PLAN:-$CONTROL_DIR/production_source_pack_plan.json}"
PYTHON_BIN="${PYTHON_BIN:-$CODE_DIR/.venv/bin/python}"
test ! -e "$PACKING_CONTROL_DIR"
mkdir -p "$PACKING_CONTROL_DIR"
cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m scripts.turkish_packed_production validate-gate \
    --recipe="$RECIPE" --policy="$POLICY" --source-plan="$SOURCE_PLAN" \
    --calibration="$CALIBRATION" --pack-plan="$PACK_PLAN" \
    --resource-approval="$RESOURCE_APPROVAL" \
    --mixture-quality-approval="$MIXTURE_QUALITY_APPROVAL" \
    --storage-gate="$DATA_PREP_STORAGE_GATE" \
    --write-dir="$PACKING_CONTROL_DIR" \
    --cluster-launch-receipt="$CLUSTER_LAUNCH_RECEIPT" \
    --pool-dir="$POOL_DIR" --tokenizer-dir="$TOKENIZER_DIR"

"$PYTHON_BIN" scripts/build_turkish_pretrain_corpus.py \
    --policy "$POLICY" packing-preflight \
    --pool-dir "$POOL_DIR" \
    --tokenizer-dir "$TOKENIZER_DIR" \
    --max-documents-per-rank-and-mixture "${PACKING_SAMPLE_DOCUMENTS:-8192}" \
    --projection-safety-factor "${PACKING_SAFETY_FACTOR:-1.08}" \
    --output "$PACKING_CONTROL_DIR/packing_preflight_report.json"

echo "Review the sealed report, then create packing_preflight_approval.json with the approve-packing-preflight command."
