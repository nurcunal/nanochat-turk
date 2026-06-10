#!/bin/bash -l

# Parallel Zemberek segmentation for the MorphBPE 32k tokenizer.
#
# Submit after Zemberek preflight succeeds:
#
#   sbatch --dependency=afterok:<preflight_jobid> \
#     runs/uhem_nakane_segment_morphbpe_zemberek_32k_array.sbatch

#SBATCH --job-name=nanochat-seg-zemberek32k
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --array=0-29%6
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
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/nanochat-turk-morphbpe-zemberek-32768}"
export RAW_DATA_DIR="${RAW_DATA_DIR:-$HOME/nanochat-turk-d20-bpe32k/base_data_fineweb2_tur_latn}"
export SEGMENTED_DATA_DIR="${SEGMENTED_DATA_DIR:-$NANOCHAT_BASE_DIR/segmented/zemberek-segmented}"
export HF_HOME="${HF_HOME:-$NANOCHAT_BASE_DIR/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$NANOCHAT_BASE_DIR/hf-datasets-cache}"

export SEGMENT_BATCH_SIZE="${SEGMENT_BATCH_SIZE:-2048}"
export WORD_CACHE_SIZE="${WORD_CACHE_SIZE:-500000}"
# Zemberek keeps an analyzer in each process. 64 workers uses the node
# substantially while avoiding 128 simultaneous analyzer loads by default.
export SEGMENT_WORKERS="${SEGMENT_WORKERS:-64}"
export SEGMENT_WORKER_MODE="${SEGMENT_WORKER_MODE:-process}"
export SEGMENT_MAX_IN_FLIGHT="${SEGMENT_MAX_IN_FLIGHT:-$((SEGMENT_WORKERS * 2))}"
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
if [ ! -d ".venv" ]; then
    echo "Missing .venv in $PWD. Run Zemberek preflight first." >&2
    exit 2
fi
source .venv/bin/activate

python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("zemberek") else "Missing zemberek package; run preflight first")
PY

python -m scripts.morph_segment_corpus \
    --backend zemberek \
    --data-dir "$RAW_DATA_DIR" \
    --output-dir "$SEGMENTED_DATA_DIR" \
    --compact \
    --resume \
    --shard-index "$SLURM_ARRAY_TASK_ID" \
    --segment-batch-size "$SEGMENT_BATCH_SIZE" \
    --word-cache-size "$WORD_CACHE_SIZE" \
    --num-workers "$SEGMENT_WORKERS" \
    --worker-mode "$SEGMENT_WORKER_MODE" \
    --max-in-flight "$SEGMENT_MAX_IN_FLIGHT"

echo "Completed Zemberek segmented shard index $SLURM_ARRAY_TASK_ID"
