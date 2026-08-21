#!/bin/bash

# Seal the explicit human tokenizer-quality decision. The 12-hour request is
# the fixed fail-closed ceiling; normal execution should finish immediately.

#SBATCH --job-name=tr-tok-quality
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -eo pipefail
source /etc/profile.d/modules.sh
module use /ari/progs/modulefiles
module purge
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0
set -u

: "${TOKENIZER_REVIEWER:?Set the human reviewer identity}"
: "${TOKENIZER_REVIEWED_AT_UTC:?Set an RFC3339 UTC review timestamp}"
: "${TOKENIZER_DECISION:?Set accepted or rejected after manual review}"
: "${RESOURCE_APPROVAL:?}"
: "${MIXTURE_QUALITY_APPROVAL:?}"
: "${DATA_PREP_STORAGE_GATE:?}"
: "${CLUSTER_LAUNCH_RECEIPT:?}"

CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$CODE_DIR/runs/uhem_d32_require_clean_code.sh"
d32_require_clean_committed_code "$CODE_DIR"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/ari/users/nunal/nanochat-turk-d32-general-v3}"
RECIPE="${RECIPE:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_wsd_v4.json}"
POLICY="${POLICY:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_v4.json}"
CONTROL_DIR="${CONTROL_DIR:-$NANOCHAT_BASE_DIR/control/data_v4}"
SOURCE_PLAN="${SOURCE_PLAN:-$CONTROL_DIR/source_plan.json}"
CALIBRATION="${CALIBRATION:-$CONTROL_DIR/backend_calibration.json}"
PACK_PLAN="${PACK_PLAN:-$CONTROL_DIR/production_source_pack_plan.json}"
PYTHON="${PYTHON:-$CODE_DIR/.venv/bin/python}"

D32_PATH_CONTRACT="${D32_PATH_CONTRACT:-v4}"
case "$D32_PATH_CONTRACT" in
    v3|v4) source "$CODE_DIR/runs/uhem_d32_${D32_PATH_CONTRACT}_paths.sh" ;;
    *) echo "D32_PATH_CONTRACT must equal v3 or v4" >&2; exit 2 ;;
esac
d32_bind_canonical_path POOL_DIR "$D32_POOL_DIR"
d32_bind_canonical_path TOKENIZER_DIR "$D32_TOKENIZER_DIR"
d32_bind_canonical_path TOKENIZER_QUALITY_DIR "$D32_TOKENIZER_QUALITY_DIR"

cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
common=(--recipe="$RECIPE" --policy="$POLICY" --source-plan="$SOURCE_PLAN"
    --calibration="$CALIBRATION" --pack-plan="$PACK_PLAN"
    --resource-approval="$RESOURCE_APPROVAL"
    --mixture-quality-approval="$MIXTURE_QUALITY_APPROVAL"
    --storage-gate="$DATA_PREP_STORAGE_GATE")
"$PYTHON" -m scripts.turkish_packed_production validate-gate \
    "${common[@]}" --write-dir="$TOKENIZER_QUALITY_DIR" \
    --cluster-launch-receipt="$CLUSTER_LAUNCH_RECEIPT" \
    --pool-dir="$POOL_DIR" \
    --tokenizer-dir="$TOKENIZER_DIR" \
    --tokenizer-quality-dir="$TOKENIZER_QUALITY_DIR"
"$PYTHON" scripts/review_turkish_tokenizer_quality.py \
    --quality-dir "$TOKENIZER_QUALITY_DIR" \
    --tokenizer-dir "$TOKENIZER_DIR" \
    --reviewer "$TOKENIZER_REVIEWER" \
    --reviewed-at-utc "$TOKENIZER_REVIEWED_AT_UTC" \
    --decision "$TOKENIZER_DECISION" \
    --notes "${TOKENIZER_REVIEW_NOTES:-}"
test -f "$TOKENIZER_QUALITY_DIR/quality_approval.json"
