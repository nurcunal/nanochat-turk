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
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/ari/users/nunal/nanochat-turk-d32-general}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
POLICY="${POLICY:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_v1.json}"
CONTROL_DIR="$NANOCHAT_BASE_DIR/control/data"
MODEL_DIR="$NANOCHAT_BASE_DIR/source_models/glotlid-v3"
SOURCE_PLAN="$CONTROL_DIR/source_plan.json"
CALIBRATION="$CONTROL_DIR/backend_calibration.json"
SAMPLE_RANKS="$CONTROL_DIR/resource_sample_ranks.json"

cd "$CODE_DIR"
test -x "$UV_BIN"
test -x environments/turkish-data/.venv/bin/python
mkdir -p "$CONTROL_DIR" "$MODEL_DIR"

if [ ! -f "$SOURCE_PLAN" ]; then
    "$UV_BIN" run --project environments/turkish-data --locked python \
        scripts/turkish_data_backend.py resolve \
        --policy "$POLICY" --output "$SOURCE_PLAN"
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
echo "GlotLID model: $MODEL_DIR/model_v3.bin"
echo "Backend calibration: $CALIBRATION"
echo "Resource-sample ranks: $SAMPLE_RANKS"
