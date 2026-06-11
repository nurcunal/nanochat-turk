#!/bin/bash -l

# TurkishDelight segmentation smoke benchmark for MorphBPE 32k.
#
# This intentionally writes to a separate smoke directory and caps documents so
# we can compare worker/persistence settings before committing full shards.

#SBATCH --job-name=nanochat-smoke-tdlt32k
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --array=0-5%6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=04:00:00
#SBATCH --mem=240G
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -euo pipefail

if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
elif [ -f /usr/share/lmod/lmod/init/bash ]; then
    source /usr/share/lmod/lmod/init/bash
fi

module purge
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0

echo "HOSTNAME=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "PYTHON=$(command -v python3)"
python3 --version

export PATH="$HOME/.local/bin:$PATH"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/nanochat-turk-morphbpe-tdelight-32768}"
export RAW_DATA_DIR="${RAW_DATA_DIR:-$HOME/nanochat-turk-d20-bpe32k/base_data_fineweb2_tur_latn}"
export HF_HOME="${HF_HOME:-$NANOCHAT_BASE_DIR/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$NANOCHAT_BASE_DIR/hf-datasets-cache}"

export TDELIGHT_REPO_DIR="${TDELIGHT_REPO_DIR:-$PWD/dev-ignore/vendor/turkish-delight-nlp-api}"
export TDELIGHT_VENV="${TDELIGHT_VENV:-$PWD/dev-ignore/venvs/tdelight-uhem-py312}"
export TDELIGHT_DYNET_SRC="${TDELIGHT_DYNET_SRC:-$PWD/dev-ignore/vendor/dynet-sdist/dyNET-2.1.2}"
export TDELIGHT_DYNET_LIB_DIR="${TDELIGHT_DYNET_LIB_DIR:-$TDELIGHT_DYNET_SRC/build/py3.12-64bit/dynet}"
export TDELIGHT_MODEL_NAME="${TDELIGHT_MODEL_NAME:-Turkish-jointAll-MTAG_COMP=w_sum-MORPH_COMP=w_sum-POS_COMP=w_sum-COMP_ALPHA=0.1-trialmodel}"
export TDELIGHT_SEGMENT_CMD="${TDELIGHT_SEGMENT_CMD:-$TDELIGHT_VENV/bin/python scripts/tdelight_segment_cmd.py --repo-dir $TDELIGHT_REPO_DIR --model-name $TDELIGHT_MODEL_NAME}"
export LD_LIBRARY_PATH="$TDELIGHT_DYNET_LIB_DIR:${LD_LIBRARY_PATH:-}"

export SMOKE_SHARD_INDEX="${SMOKE_SHARD_INDEX:-3}"
export SMOKE_MAX_DOCS="${SMOKE_MAX_DOCS:-1024}"
export WORD_CACHE_SIZE="${WORD_CACHE_SIZE:-500000}"
export SEGMENT_WORKER_MODE="${SEGMENT_WORKER_MODE:-process}"
export SEGMENT_TIMEOUT="${SEGMENT_TIMEOUT:-900}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYARROW_NUM_THREADS="${PYARROW_NUM_THREADS:-1}"

case "${SLURM_ARRAY_TASK_ID:-0}" in
    0)
        LABEL="legacy_w16_b512_rg32"
        export TDELIGHT_PERSISTENT_COMMAND=0
        export SEGMENT_WORKERS=16
        export SEGMENT_BATCH_SIZE=512
        export ROW_GROUP_BATCH_SIZE=32
        ;;
    1)
        LABEL="persistent_w16_b512_rg32"
        export TDELIGHT_PERSISTENT_COMMAND=1
        export SEGMENT_WORKERS=16
        export SEGMENT_BATCH_SIZE=512
        export ROW_GROUP_BATCH_SIZE=32
        ;;
    2)
        LABEL="persistent_w32_b512_rg32"
        export TDELIGHT_PERSISTENT_COMMAND=1
        export SEGMENT_WORKERS=32
        export SEGMENT_BATCH_SIZE=512
        export ROW_GROUP_BATCH_SIZE=32
        ;;
    3)
        LABEL="persistent_w64_b512_rg32"
        export TDELIGHT_PERSISTENT_COMMAND=1
        export SEGMENT_WORKERS=64
        export SEGMENT_BATCH_SIZE=512
        export ROW_GROUP_BATCH_SIZE=32
        ;;
    4)
        LABEL="persistent_w32_b2048_rg32"
        export TDELIGHT_PERSISTENT_COMMAND=1
        export SEGMENT_WORKERS=32
        export SEGMENT_BATCH_SIZE=2048
        export ROW_GROUP_BATCH_SIZE=32
        ;;
    5)
        LABEL="persistent_w64_b2048_rg32"
        export TDELIGHT_PERSISTENT_COMMAND=1
        export SEGMENT_WORKERS=64
        export SEGMENT_BATCH_SIZE=2048
        export ROW_GROUP_BATCH_SIZE=32
        ;;
    6)
        LABEL="persistent_w4_b512_rg32"
        export TDELIGHT_PERSISTENT_COMMAND=1
        export SEGMENT_WORKERS=4
        export SEGMENT_BATCH_SIZE=512
        export ROW_GROUP_BATCH_SIZE=32
        ;;
    7)
        LABEL="persistent_w8_b512_rg32"
        export TDELIGHT_PERSISTENT_COMMAND=1
        export SEGMENT_WORKERS=8
        export SEGMENT_BATCH_SIZE=512
        export ROW_GROUP_BATCH_SIZE=32
        ;;
    8)
        LABEL="persistent_w12_b512_rg32"
        export TDELIGHT_PERSISTENT_COMMAND=1
        export SEGMENT_WORKERS=12
        export SEGMENT_BATCH_SIZE=512
        export ROW_GROUP_BATCH_SIZE=32
        ;;
    9)
        LABEL="persistent_w16_b256_rg32"
        export TDELIGHT_PERSISTENT_COMMAND=1
        export SEGMENT_WORKERS=16
        export SEGMENT_BATCH_SIZE=256
        export ROW_GROUP_BATCH_SIZE=32
        ;;
    10)
        LABEL="persistent_w16_b2048_rg32"
        export TDELIGHT_PERSISTENT_COMMAND=1
        export SEGMENT_WORKERS=16
        export SEGMENT_BATCH_SIZE=2048
        export ROW_GROUP_BATCH_SIZE=32
        ;;
    11)
        LABEL="persistent_w24_b512_rg32"
        export TDELIGHT_PERSISTENT_COMMAND=1
        export SEGMENT_WORKERS=24
        export SEGMENT_BATCH_SIZE=512
        export ROW_GROUP_BATCH_SIZE=32
        ;;
    *)
        echo "Unknown smoke config ${SLURM_ARRAY_TASK_ID:-}" >&2
        exit 2
        ;;
esac

export LABEL
export SEGMENT_MAX_IN_FLIGHT="${SEGMENT_MAX_IN_FLIGHT:-$SEGMENT_WORKERS}"
export SMOKE_ROOT="${SMOKE_ROOT:-$NANOCHAT_BASE_DIR/smoke/tdelight-segmentation}"
export SEGMENTED_DATA_DIR="$SMOKE_ROOT/job-${SLURM_ARRAY_JOB_ID:-manual}/${SLURM_ARRAY_TASK_ID}_${LABEL}"

mkdir -p "$SEGMENTED_DATA_DIR"

if [ ! -f "$RAW_DATA_DIR/fineweb2_manifest.json" ]; then
    echo "Missing raw FineWeb-2 Turkish manifest at $RAW_DATA_DIR/fineweb2_manifest.json" >&2
    exit 2
fi
if [ ! -x "$TDELIGHT_VENV/bin/python" ]; then
    echo "Missing TurkishDelight venv at $TDELIGHT_VENV; run preflight first." >&2
    exit 2
fi
if [ ! -d "$TDELIGHT_REPO_DIR" ]; then
    echo "Missing TurkishDelight source at $TDELIGHT_REPO_DIR; run preflight first." >&2
    exit 2
fi
if [ ! -d ".venv" ]; then
    echo "Missing repo .venv in $PWD. Run the main setup/prep job first." >&2
    exit 2
fi

"$TDELIGHT_VENV/bin/python" - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("dynet") else "Missing dynet package; run preflight first")
PY

source .venv/bin/activate

echo "Smoke config: $LABEL"
echo "Output dir: $SEGMENTED_DATA_DIR"
echo "shard_index=$SMOKE_SHARD_INDEX max_docs=$SMOKE_MAX_DOCS"
echo "persistent=$TDELIGHT_PERSISTENT_COMMAND workers=$SEGMENT_WORKERS batch=$SEGMENT_BATCH_SIZE row_group=$ROW_GROUP_BATCH_SIZE max_in_flight=$SEGMENT_MAX_IN_FLIGHT timeout=$SEGMENT_TIMEOUT"

python3 -m scripts.morph_segment_corpus \
    --backend tdelight \
    --command "$TDELIGHT_SEGMENT_CMD" \
    --data-dir "$RAW_DATA_DIR" \
    --output-dir "$SEGMENTED_DATA_DIR" \
    --compact \
    --overwrite \
    --shard-index "$SMOKE_SHARD_INDEX" \
    --max-docs "$SMOKE_MAX_DOCS" \
    --row-group-batch-size "$ROW_GROUP_BATCH_SIZE" \
    --segment-batch-size "$SEGMENT_BATCH_SIZE" \
    --word-cache-size "$WORD_CACHE_SIZE" \
    --num-workers "$SEGMENT_WORKERS" \
    --worker-mode "$SEGMENT_WORKER_MODE" \
    --max-in-flight "$SEGMENT_MAX_IN_FLIGHT" \
    --timeout "$SEGMENT_TIMEOUT"

python3 - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["SEGMENTED_DATA_DIR"])
done_files = sorted(out.glob("*.done.json"))
if not done_files:
    raise SystemExit("No smoke .done.json was written")
payload = json.loads(done_files[0].read_text())
summary = {
    "label": os.environ["LABEL"],
    "persistent": os.environ["TDELIGHT_PERSISTENT_COMMAND"],
    "workers": int(os.environ["SEGMENT_WORKERS"]),
    "segment_batch_size": int(os.environ["SEGMENT_BATCH_SIZE"]),
    "row_group_batch_size": int(os.environ["ROW_GROUP_BATCH_SIZE"]),
    "max_docs": int(os.environ["SMOKE_MAX_DOCS"]),
    "docs": payload.get("docs"),
    "word_count": payload.get("word_count"),
    "split_words": payload.get("split_words"),
    "fallback_words": payload.get("fallback_words"),
    "elapsed_seconds": payload.get("elapsed_seconds"),
    "docs_per_sec": payload.get("docs_per_sec"),
    "words_per_sec": payload.get("words_per_sec"),
    "output_path": payload.get("output_path"),
}
print("SMOKE_RESULT " + json.dumps(summary, ensure_ascii=False, sort_keys=True))
PY
