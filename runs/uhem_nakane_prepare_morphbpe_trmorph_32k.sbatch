#!/bin/bash -l

# CPU-side preparation for the TRmorph MorphBPE 32k tokenizer run.
#
# This job does not use A100 time. It verifies/syncs the Python environment,
# materializes the TRmorph-segmented corpus, trains the MorphBPE tokenizer, and
# runs tokenizer-only metrics on the raw FineWeb-2 Turkish text.
#
# Submit from the repo root on Altay only after TRmorph segment.fst and flookup
# are available:
#   sbatch runs/uhem_nakane_prepare_morphbpe_trmorph_32k.sbatch

#SBATCH --job-name=nanochat-prep-morphbpe-trmorph32k
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=2-00:00:00
#SBATCH --mem=240G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

if [ -f /etc/profile.d/modules.sh ]; then
    # Batch shells on Altay do not always initialize Environment Modules.
    source /etc/profile.d/modules.sh
elif [ -f /usr/share/lmod/lmod/init/bash ]; then
    source /usr/share/lmod/lmod/init/bash
fi

module purge
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0

echo "HOSTNAME=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "PYTHON=$(command -v python3)"
python3 --version
echo "UV=$(command -v uv || true)"

export PATH="$HOME/.local/bin:$PATH"
export UV_PYTHON="${UV_PYTHON:-$(command -v python3)}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/nanochat-turk-morphbpe-trmorph-32768}"
export RAW_DATA_DIR="${RAW_DATA_DIR:-$HOME/nanochat-turk-d20-bpe32k/base_data_fineweb2_tur_latn}"
export SEGMENTED_DATA_DIR="${SEGMENTED_DATA_DIR:-$NANOCHAT_BASE_DIR/segmented/trmorph-segmented}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$NANOCHAT_BASE_DIR/uv-cache}"
export HF_HOME="${HF_HOME:-$NANOCHAT_BASE_DIR/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$NANOCHAT_BASE_DIR/hf-datasets-cache}"

export VOCAB_SIZE="${VOCAB_SIZE:-32768}"
export NANOCHAT_TOKENIZER_NAME="${NANOCHAT_TOKENIZER_NAME:-morphbpe_trmorph_32768}"
export TOKENIZER_CHARS="${TOKENIZER_CHARS:-2000000000}"
export TOKENIZER_METRIC_DOCS="${TOKENIZER_METRIC_DOCS:-10000}"

export TRMORPH_SEGMENT_FST="${TRMORPH_SEGMENT_FST:-$HOME/TRmorph/segment.fst}"
export TRMORPH_FLOOKUP_CMD="${TRMORPH_FLOOKUP_CMD:-flookup}"
export TRMORPH_FLOOKUP_FLAGS="${TRMORPH_FLOOKUP_FLAGS:--x}"
export TRMORPH_PICK="${TRMORPH_PICK:-fewest_segments}"
export TRMORPH_MAX_OUTPUT_LINES_PER_WORD="${TRMORPH_MAX_OUTPUT_LINES_PER_WORD:-512}"
export TRMORPH_MAX_ANALYSES_PER_WORD="${TRMORPH_MAX_ANALYSES_PER_WORD:-64}"
export SEGMENT_BATCH_SIZE="${SEGMENT_BATCH_SIZE:-2048}"
export WORD_CACHE_SIZE="${WORD_CACHE_SIZE:-500000}"
export SEGMENT_WORKERS="${SEGMENT_WORKERS:-${SLURM_CPUS_PER_TASK:-128}}"
export SEGMENT_WORKER_MODE="${SEGMENT_WORKER_MODE:-process}"
export SEGMENT_MAX_IN_FLIGHT="${SEGMENT_MAX_IN_FLIGHT:-$((SEGMENT_WORKERS * 2))}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYARROW_NUM_THREADS="${PYARROW_NUM_THREADS:-1}"

mkdir -p "$NANOCHAT_BASE_DIR" "$SEGMENTED_DATA_DIR"

if [ ! -f "$RAW_DATA_DIR/fineweb2_manifest.json" ]; then
    echo "Missing raw FineWeb-2 Turkish manifest at $RAW_DATA_DIR/fineweb2_manifest.json" >&2
    echo "Prepare or point RAW_DATA_DIR to the baseline raw dataset cache before this job." >&2
    exit 2
fi
if [ ! -f "$TRMORPH_SEGMENT_FST" ]; then
    echo "Missing TRmorph FST at $TRMORPH_SEGMENT_FST" >&2
    echo "Copy/build TRmorph segment.fst before this job." >&2
    exit 2
fi
if ! command -v "$TRMORPH_FLOOKUP_CMD" >/dev/null 2>&1; then
    echo "Missing flookup command: $TRMORPH_FLOOKUP_CMD" >&2
    echo "Install foma/flookup or set TRMORPH_FLOOKUP_CMD to its absolute path." >&2
    exit 2
fi

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv ${UV_PYTHON:+--python "$UV_PYTHON"}
uv sync --extra gpu
source .venv/bin/activate

if [ ! -f "$SEGMENTED_DATA_DIR/fineweb2_manifest.json" ]; then
    python -m scripts.morph_segment_corpus \
        --backend trmorph \
        --data-dir "$RAW_DATA_DIR" \
        --output-dir "$SEGMENTED_DATA_DIR" \
        --compact \
        --resume \
        --segment-batch-size "$SEGMENT_BATCH_SIZE" \
        --word-cache-size "$WORD_CACHE_SIZE" \
        --num-workers "$SEGMENT_WORKERS" \
        --worker-mode "$SEGMENT_WORKER_MODE" \
        --max-in-flight "$SEGMENT_MAX_IN_FLIGHT"
else
    echo "Using existing segmented manifest: $SEGMENTED_DATA_DIR/fineweb2_manifest.json"
fi

python -m scripts.tok_train \
    --implementation morphbpe \
    --data-dir "$SEGMENTED_DATA_DIR" \
    --text-column segmented_text \
    --vocab-size "$VOCAB_SIZE" \
    --max-chars "$TOKENIZER_CHARS" \
    --tokenizer-name "$NANOCHAT_TOKENIZER_NAME"

python -m scripts.tokenizer_metrics \
    --data-dir "$RAW_DATA_DIR" \
    --text-column text \
    --max-docs "$TOKENIZER_METRIC_DOCS" \
    --output "$NANOCHAT_BASE_DIR/${NANOCHAT_TOKENIZER_NAME}_raw_metrics.json"

echo "Prepared MorphBPE tokenizer under:"
echo "  $NANOCHAT_BASE_DIR/tokenizers/$NANOCHAT_TOKENIZER_NAME"
echo "Segmented corpus:"
echo "  $SEGMENTED_DATA_DIR"
du -sh "$SEGMENTED_DATA_DIR" "$NANOCHAT_BASE_DIR/tokenizers/$NANOCHAT_TOKENIZER_NAME" || true
