#!/bin/bash -l

# Parallel TurkishDelightNLP segmentation for the MorphBPE 32k tokenizer.
#
# Submit after TurkishDelight preflight succeeds:
#
#   sbatch --dependency=afterok:<preflight_jobid> \
#     runs/uhem_nakane_segment_morphbpe_tdelight_32k_array.sbatch

#SBATCH --job-name=nanochat-seg-tdlt32k
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --array=0-29%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=2-00:00:00
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
export SEGMENTED_DATA_DIR="${SEGMENTED_DATA_DIR:-$NANOCHAT_BASE_DIR/segmented/tdelight-segmented}"
export HF_HOME="${HF_HOME:-$NANOCHAT_BASE_DIR/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$NANOCHAT_BASE_DIR/hf-datasets-cache}"

export TDELIGHT_REPO_DIR="${TDELIGHT_REPO_DIR:-$PWD/dev-ignore/vendor/turkish-delight-nlp-api}"
export TDELIGHT_VENV="${TDELIGHT_VENV:-$PWD/dev-ignore/venvs/tdelight-uhem-py312}"
export TDELIGHT_MODEL_NAME="${TDELIGHT_MODEL_NAME:-Turkish-jointAll-MTAG_COMP=w_sum-MORPH_COMP=w_sum-POS_COMP=w_sum-COMP_ALPHA=0.1-trialmodel}"
export TDELIGHT_SEGMENT_CMD="${TDELIGHT_SEGMENT_CMD:-$TDELIGHT_VENV/bin/python scripts/tdelight_segment_cmd.py --repo-dir $TDELIGHT_REPO_DIR --model-name $TDELIGHT_MODEL_NAME}"

export SEGMENT_BATCH_SIZE="${SEGMENT_BATCH_SIZE:-4096}"
export WORD_CACHE_SIZE="${WORD_CACHE_SIZE:-500000}"
export SEGMENT_WORKERS="${SEGMENT_WORKERS:-${SLURM_CPUS_PER_TASK:-128}}"
export SEGMENT_WORKER_MODE="${SEGMENT_WORKER_MODE:-process}"
export SEGMENT_MAX_IN_FLIGHT="${SEGMENT_MAX_IN_FLIGHT:-$SEGMENT_WORKERS}"
export SEGMENT_TIMEOUT="${SEGMENT_TIMEOUT:-300}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYARROW_NUM_THREADS="${PYARROW_NUM_THREADS:-1}"

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

python3 -m scripts.morph_segment_corpus \
    --backend tdelight \
    --command "$TDELIGHT_SEGMENT_CMD" \
    --data-dir "$RAW_DATA_DIR" \
    --output-dir "$SEGMENTED_DATA_DIR" \
    --compact \
    --resume \
    --shard-index "$SLURM_ARRAY_TASK_ID" \
    --segment-batch-size "$SEGMENT_BATCH_SIZE" \
    --word-cache-size "$WORD_CACHE_SIZE" \
    --num-workers "$SEGMENT_WORKERS" \
    --worker-mode "$SEGMENT_WORKER_MODE" \
    --max-in-flight "$SEGMENT_MAX_IN_FLIGHT" \
    --timeout "$SEGMENT_TIMEOUT"

echo "Completed TurkishDelight segmented shard index $SLURM_ARRAY_TASK_ID"
