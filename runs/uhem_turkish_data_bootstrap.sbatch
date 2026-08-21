#!/bin/bash

# Resolve immutable source objects, fetch the pinned GlotLID model, and seal
# the mandatory backend calibration before any corpus object is downloaded.

#SBATCH --job-name=tr-data-bootstrap
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
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
MACOCU_MANIFEST="${MACOCU_MANIFEST:-$NANOCHAT_BASE_DIR/source_data/macocu_genre_tr_v1/manifest.json}"
# Empty explicit values disable these prepared inputs for a historical policy
# override. The active v4 defaults reuse both shared manifests in place.
MOT_MANIFEST="${MOT_MANIFEST-$NANOCHAT_BASE_DIR/source_data/mot_tr_v1_11/manifest.json}"
PARLAMINT_MANIFEST="${PARLAMINT_MANIFEST-$NANOCHAT_BASE_DIR/source_data/parlamint_tr_v5_0/manifest.json}"
D32_PATH_CONTRACT="${D32_PATH_CONTRACT:-v4}"
case "$D32_PATH_CONTRACT" in
    v3|v4) source "$CODE_DIR/runs/uhem_d32_${D32_PATH_CONTRACT}_paths.sh" ;;
    *) echo "D32_PATH_CONTRACT must equal v3 or v4" >&2; exit 2 ;;
esac
CONTROL_DIR="${CONTROL_DIR:-$D32_CONTROL_DIR}"
d32_bind_canonical_path CONTROL_DIR "$D32_CONTROL_DIR"
MODEL_DIR="${MODEL_DIR:-$NANOCHAT_BASE_DIR/source_models/glotlid-v3}"
SOURCE_PLAN="${SOURCE_PLAN:-$CONTROL_DIR/source_plan.json}"
CALIBRATION="${CALIBRATION:-$CONTROL_DIR/backend_calibration.json}"
SAMPLE_RANKS="${SAMPLE_RANKS:-$CONTROL_DIR/resource_sample_ranks.json}"
d32_bind_canonical_path SOURCE_PLAN "$D32_SOURCE_PLAN"
d32_bind_canonical_path CALIBRATION "$D32_CALIBRATION"
d32_bind_canonical_path SAMPLE_RANKS "$D32_SAMPLE_RANKS"

cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
test -x "$UV_BIN"
test -x environments/turkish-data/.venv/bin/python
mkdir -p "$CONTROL_DIR" "$MODEL_DIR"
test -f "$MACOCU_MANIFEST"
PREPARED_SOURCE_ARGS=()
if [ -n "$MOT_MANIFEST" ]; then
    test -f "$MOT_MANIFEST"
    PREPARED_SOURCE_ARGS+=(--prepared-source-manifest "mot_tr_v1_11=$MOT_MANIFEST")
fi
if [ -n "$PARLAMINT_MANIFEST" ]; then
    test -f "$PARLAMINT_MANIFEST"
    PREPARED_SOURCE_ARGS+=(--prepared-source-manifest "parlamint_tr_v5_0=$PARLAMINT_MANIFEST")
fi

if [ ! -f "$SOURCE_PLAN" ]; then
    "$UV_BIN" run --project environments/turkish-data --locked python \
        scripts/turkish_data_backend.py resolve \
        --policy "$POLICY" --macocu-manifest "$MACOCU_MANIFEST" \
        "${PREPARED_SOURCE_ARGS[@]}" \
        --output "$SOURCE_PLAN"
fi

"$UV_BIN" run --project environments/turkish-data --locked python \
    scripts/turkish_data_backend.py fetch-glotlid \
    --policy "$POLICY" --output-dir "$MODEL_DIR"

if [ ! -f "$CALIBRATION" ]; then
    "$UV_BIN" run --project environments/turkish-data --locked python \
        scripts/turkish_data_backend.py calibrate \
        --policy "$POLICY" \
        --model "$MODEL_DIR/model_v3.bin" \
        --fixture configs/pretrain/glotlid_calibration_tr_v1.jsonl \
        --output "$CALIBRATION"
fi

sample_tmp="$SAMPLE_RANKS.job${SLURM_JOB_ID:?}.tmp"
"$UV_BIN" run --project environments/turkish-data --locked python \
    scripts/turkish_data_backend.py sample-ranks \
    --policy "$POLICY" --source-plan "$SOURCE_PLAN" > "$sample_tmp"
mv "$sample_tmp" "$SAMPLE_RANKS"

echo "Source plan: $SOURCE_PLAN"
echo "MaCoCu preparation: $MACOCU_MANIFEST"
[ -z "$MOT_MANIFEST" ] || echo "MOT preparation: $MOT_MANIFEST"
[ -z "$PARLAMINT_MANIFEST" ] || echo "ParlaMint preparation: $PARLAMINT_MANIFEST"
echo "GlotLID model: $MODEL_DIR/model_v3.bin"
echo "Backend calibration: $CALIBRATION"
echo "Resource-sample ranks: $SAMPLE_RANKS"
