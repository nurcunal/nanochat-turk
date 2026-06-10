#!/bin/bash -l

# Finalize TurkishDelight segmented shards, train the MorphBPE 32k tokenizer,
# and run tokenizer-only metrics. Intended to run after the TurkishDelight
# segmentation array succeeds.

#SBATCH --job-name=nanochat-finalize-mbpe-tdlt32k
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --mem=240G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

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
echo "PYTHON=$(command -v python3)"
python3 --version

export PATH="$HOME/.local/bin:$PATH"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/nanochat-turk-morphbpe-tdelight-32768}"
export RAW_DATA_DIR="${RAW_DATA_DIR:-$HOME/nanochat-turk-d20-bpe32k/base_data_fineweb2_tur_latn}"
export SEGMENTED_DATA_DIR="${SEGMENTED_DATA_DIR:-$NANOCHAT_BASE_DIR/segmented/tdelight-segmented}"
export HF_HOME="${HF_HOME:-$NANOCHAT_BASE_DIR/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$NANOCHAT_BASE_DIR/hf-datasets-cache}"

export VOCAB_SIZE="${VOCAB_SIZE:-32768}"
export NANOCHAT_TOKENIZER_NAME="${NANOCHAT_TOKENIZER_NAME:-morphbpe_tdelight_32768}"
export TOKENIZER_CHARS="${TOKENIZER_CHARS:-2000000000}"
export TOKENIZER_METRIC_DOCS="${TOKENIZER_METRIC_DOCS:-10000}"

if [ ! -d ".venv" ]; then
    echo "Missing repo .venv in $PWD. Run the main setup/prep job first." >&2
    exit 2
fi
source .venv/bin/activate

python -m scripts.morph_segment_corpus \
    --backend tdelight \
    --data-dir "$RAW_DATA_DIR" \
    --output-dir "$SEGMENTED_DATA_DIR" \
    --compact \
    --resume \
    --finalize-only

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
    --no-report \
    --output "$NANOCHAT_BASE_DIR/${NANOCHAT_TOKENIZER_NAME}_raw_metrics.json"

echo "Prepared TurkishDelight MorphBPE tokenizer under:"
echo "  $NANOCHAT_BASE_DIR/tokenizers/$NANOCHAT_TOKENIZER_NAME"
echo "Segmented corpus:"
echo "  $SEGMENTED_DATA_DIR"
du -sh "$SEGMENTED_DATA_DIR" "$NANOCHAT_BASE_DIR/tokenizers/$NANOCHAT_TOKENIZER_NAME" || true
