#!/bin/bash

# Full promotion plus exact ws8/ws16 finite capacity simulation. The source
# pool remains intact. Cleanup is a separate explicit command after this passes.

#SBATCH --job-name=tr-corpus-final
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -eo pipefail
source /etc/profile.d/modules.sh
module use /ari/progs/modulefiles
module purge
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0
set -u
: "${APPROVED_SOURCE_TOKENS:?Set this to approved_source_token_target exactly}"
: "${RESOURCE_APPROVAL:?}"
: "${MIXTURE_QUALITY_APPROVAL:?}"
: "${DATA_PREP_STORAGE_GATE:?}"
: "${CLUSTER_LAUNCH_RECEIPT:?}"

CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$CODE_DIR/runs/uhem_d32_require_clean_code.sh"
d32_require_clean_committed_code "$CODE_DIR"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/ari/users/nunal/nanochat-turk-d32-general-v3}"
POLICY="${POLICY:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_v4.json}"
FAMILY_RECIPE="${FAMILY_RECIPE:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_wsd_v4.json}"
PYTHON_BIN="${PYTHON_BIN:-$CODE_DIR/.venv/bin/python}"
DATA_PYTHON="${DATA_PYTHON:-$CODE_DIR/environments/turkish-data/.venv/bin/python}"
CONTROL_DIR="${CONTROL_DIR:-$NANOCHAT_BASE_DIR/control/data_v4}"
SOURCE_PLAN="${SOURCE_PLAN:-$CONTROL_DIR/source_plan.json}"
CALIBRATION="${CALIBRATION:-$CONTROL_DIR/backend_calibration.json}"
PACK_PLAN="${PACK_PLAN:-$CONTROL_DIR/production_source_pack_plan.json}"

D32_PATH_CONTRACT="${D32_PATH_CONTRACT:-v4}"
case "$D32_PATH_CONTRACT" in
    v3|v4) source "$CODE_DIR/runs/uhem_d32_${D32_PATH_CONTRACT}_paths.sh" ;;
    *) echo "D32_PATH_CONTRACT must equal v3 or v4" >&2; exit 2 ;;
esac
d32_bind_canonical_path POOL_DIR "$D32_POOL_DIR"
d32_bind_canonical_path TOKENIZER_DIR "$D32_TOKENIZER_DIR"
d32_bind_canonical_path TOKENIZER_QUALITY_DIR "$D32_TOKENIZER_QUALITY_DIR"
d32_bind_canonical_path PACKING_CONTROL_DIR "$D32_PACKING_CONTROL_DIR"
d32_bind_canonical_path FINAL_CORPUS_DIR "$D32_FINAL_CORPUS_DIR"
test ! -e "$FINAL_CORPUS_DIR"
mkdir -p "$FINAL_CORPUS_DIR"
cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
test -x "$DATA_PYTHON"

"$PYTHON_BIN" -m scripts.turkish_packed_production validate-gate \
    --recipe="$FAMILY_RECIPE" --policy="$POLICY" \
    --source-plan="$SOURCE_PLAN" --calibration="$CALIBRATION" \
    --pack-plan="$PACK_PLAN" --resource-approval="$RESOURCE_APPROVAL" \
    --mixture-quality-approval="$MIXTURE_QUALITY_APPROVAL" \
    --storage-gate="$DATA_PREP_STORAGE_GATE" --write-dir="$FINAL_CORPUS_DIR" \
    --cluster-launch-receipt="$CLUSTER_LAUNCH_RECEIPT" --pool-dir="$POOL_DIR" \
    --tokenizer-dir="$TOKENIZER_DIR" \
    --tokenizer-quality-dir="$TOKENIZER_QUALITY_DIR" \
    --packing-preflight-dir="$PACKING_CONTROL_DIR"

"$PYTHON_BIN" scripts/d32_family_workflow.py validate-recipe \
    --recipe "$FAMILY_RECIPE"

# Do not trust a value captured before queueing. Query effective BeeGFS
# headroom inside this allocation immediately before materialization.
QUOTA_HEADROOM_BYTES="$("$DATA_PYTHON" scripts/d32_data_prep_operator.py \
    live-beegfs-headroom --recipe "$FAMILY_RECIPE" \
    --work-dir "$NANOCHAT_BASE_DIR")"
case "$QUOTA_HEADROOM_BYTES" in
    ''|*[!0-9]*) echo "live BeeGFS headroom is not a positive integer" >&2; exit 2 ;;
esac
if [ "$QUOTA_HEADROOM_BYTES" -le 0 ]; then
    echo "live BeeGFS headroom must be positive" >&2
    exit 2
fi

"$PYTHON_BIN" scripts/build_turkish_pretrain_corpus.py \
    --policy "$POLICY" finalize \
    --pool-dir "$POOL_DIR" \
    --tokenizer-dir "$TOKENIZER_DIR" \
    --tokenizer-quality-dir "$TOKENIZER_QUALITY_DIR" \
    --packing-preflight-dir "$PACKING_CONTROL_DIR" \
    --output-dir "$FINAL_CORPUS_DIR" \
    --target-tokens "$APPROVED_SOURCE_TOKENS" \
    --shard-target-tokens "${SHARD_TARGET_TOKENS:-250000000}" \
    --quota-headroom-bytes "$QUOTA_HEADROOM_BYTES"

# Resolve all exposure-index identities from the sealed artifacts just written;
# never let a v4 corpus silently inherit an older family default.
FAMILY_ID="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["family_id"])' "$FAMILY_RECIPE")"
FAMILY_RECIPE_SHA256="$("$PYTHON_BIN" -c 'import sys; from nanochat.experiment_manifest import load_json_strict, verify_manifest_hash; print(verify_manifest_hash(load_json_strict(sys.argv[1])))' "$FAMILY_RECIPE")"
TOKENIZER_ARTIFACT_SHA256="$("$PYTHON_BIN" -c 'import sys; from nanochat.experiment_manifest import load_json_strict, verify_manifest_hash; print(verify_manifest_hash(load_json_strict(sys.argv[1])))' "$TOKENIZER_DIR/package_manifest.json")"

"$PYTHON_BIN" scripts/build_turkish_pretrain_corpus.py \
    --policy "$POLICY" exposure-index \
    --final-corpus-dir "$FINAL_CORPUS_DIR" \
    --family-id "$FAMILY_ID" \
    --study-manifest-sha256 "$FAMILY_RECIPE_SHA256" \
    --tokenizer-artifact-sha256 "$TOKENIZER_ARTIFACT_SHA256"

echo "Capacity passed. Pool data is still retained; run cleanup-pool explicitly only after verifying downstream preflight."
