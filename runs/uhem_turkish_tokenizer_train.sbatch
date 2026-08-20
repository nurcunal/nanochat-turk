#!/bin/bash

# Raw Turkish 32k BPE training and objective quality report, bounded by the
# sealed 24-hour one-node ceiling.

#SBATCH --job-name=tr-tok-train
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -eo pipefail
source /etc/profile.d/modules.sh
module use /ari/progs/modulefiles
module purge
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0
set -u

: "${TOKENIZER_SAMPLE_DIR:?}"
: "${POOL_DIR:?}"
: "${TOKENIZER_DIR:?Set a new tokenizer output directory}"
: "${TOKENIZER_QUALITY_DIR:?Set a new tokenizer quality directory}"
: "${RESOURCE_APPROVAL:?}"
: "${MIXTURE_QUALITY_APPROVAL:?}"
: "${DATA_PREP_STORAGE_GATE:?}"
: "${CLUSTER_LAUNCH_RECEIPT:?}"

CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/ari/users/nunal/nanochat-turk-d32-general-v2}"
RECIPE="${RECIPE:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_wsd_v2.json}"
POLICY="${POLICY:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_v2.json}"
CONTROL_DIR="${CONTROL_DIR:-$NANOCHAT_BASE_DIR/control/data_v2}"
SOURCE_PLAN="${SOURCE_PLAN:-$CONTROL_DIR/source_plan.json}"
CALIBRATION="${CALIBRATION:-$CONTROL_DIR/backend_calibration.json}"
PACK_PLAN="${PACK_PLAN:-$CONTROL_DIR/production_source_pack_plan.json}"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"

cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
test ! -e "$TOKENIZER_DIR"
test ! -e "$TOKENIZER_QUALITY_DIR"
mkdir -p "$TOKENIZER_DIR" "$TOKENIZER_QUALITY_DIR"
common=(--recipe="$RECIPE" --policy="$POLICY" --source-plan="$SOURCE_PLAN"
    --calibration="$CALIBRATION" --pack-plan="$PACK_PLAN"
    --resource-approval="$RESOURCE_APPROVAL"
    --mixture-quality-approval="$MIXTURE_QUALITY_APPROVAL"
    --storage-gate="$DATA_PREP_STORAGE_GATE")
"$PYTHON" -m scripts.turkish_packed_production validate-gate \
    "${common[@]}" --write-dir="$TOKENIZER_DIR" \
    --write-dir="$TOKENIZER_QUALITY_DIR" \
    --cluster-launch-receipt="$CLUSTER_LAUNCH_RECEIPT" \
    --pool-dir="$POOL_DIR" \
    --tokenizer-sample-dir="$TOKENIZER_SAMPLE_DIR"
train_args=(--policy "$POLICY" --sample-dir "$TOKENIZER_SAMPLE_DIR"
    --output-dir "$TOKENIZER_DIR" --quality-output-dir "$TOKENIZER_QUALITY_DIR")
if [ -n "${BASELINE_TOKENIZER_DIR:-}" ]; then
    train_args+=(--baseline-tokenizer-dir "$BASELINE_TOKENIZER_DIR")
fi
"$PYTHON" scripts/train_turkish_raw_bpe.py "${train_args[@]}"
test -f "$TOKENIZER_DIR/package_manifest.json"
test -f "$TOKENIZER_QUALITY_DIR/quality_report.json"
