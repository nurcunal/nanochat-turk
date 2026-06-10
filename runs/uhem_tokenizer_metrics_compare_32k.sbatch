#!/bin/bash -l

# Compare tokenizer-only metrics for the 32k raw and TRmorph MorphBPE tokenizers.
#
# This runs on the TRmorph-segmented corpus as a boundary reference, strips the
# boundary marker before tokenizer encoding, and reports both standard tokenizer
# compression/fertility metrics and token-crosses-morpheme-boundary metrics.

#SBATCH --job-name=nanochat-tokenizer-metrics32k
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --mem=64G
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

DEFAULT_REPO_DIR="$HOME/nanochat-turk"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [ ! -d "$REPO_DIR/.venv" ] && [ -d "$DEFAULT_REPO_DIR/.venv" ]; then
    REPO_DIR="$DEFAULT_REPO_DIR"
fi
cd "$REPO_DIR"

export PATH="$HOME/.local/bin:$PATH"
export RAW_BASE_DIR="${RAW_BASE_DIR:-$HOME/nanochat-turk-d20-bpe32k}"
export MORPH_BASE_DIR="${MORPH_BASE_DIR:-$HOME/nanochat-turk-morphbpe-trmorph-32768}"
export SEGMENTED_DATA_DIR="${SEGMENTED_DATA_DIR:-$MORPH_BASE_DIR/datasets/trmorph-segmented}"
export METRICS_DIR="${METRICS_DIR:-$MORPH_BASE_DIR/report/tokenizer_metrics_32k}"
export MAX_DOCS="${MAX_DOCS:-50000}"
export MAX_WORD_METRICS="${MAX_WORD_METRICS:-500000}"
export NUM_THREADS="${NUM_THREADS:-${SLURM_CPUS_PER_TASK:-32}}"

mkdir -p "$METRICS_DIR"

if [ ! -d ".venv" ]; then
    echo "Missing .venv in $REPO_DIR. Run tokenizer prep first." >&2
    exit 2
fi
source .venv/bin/activate

if [ ! -f "$SEGMENTED_DATA_DIR/fineweb2_manifest.json" ]; then
    echo "Missing segmented dataset manifest at $SEGMENTED_DATA_DIR/fineweb2_manifest.json" >&2
    exit 2
fi

python -m scripts.tokenizer_metrics \
    --tokenizer-dir "$RAW_BASE_DIR/tokenizers/bpe_32768" \
    --data-dir "$SEGMENTED_DATA_DIR" \
    --text-column segmented_text \
    --input-has-morph-boundaries \
    --max-docs "$MAX_DOCS" \
    --max-word-metrics "$MAX_WORD_METRICS" \
    --num-threads "$NUM_THREADS" \
    --no-report \
    --output "$METRICS_DIR/bpe_32768_metrics.json"

python -m scripts.tokenizer_metrics \
    --tokenizer-dir "$MORPH_BASE_DIR/tokenizers/morphbpe_trmorph_32768" \
    --data-dir "$SEGMENTED_DATA_DIR" \
    --text-column segmented_text \
    --input-has-morph-boundaries \
    --max-docs "$MAX_DOCS" \
    --max-word-metrics "$MAX_WORD_METRICS" \
    --num-threads "$NUM_THREADS" \
    --no-report \
    --output "$METRICS_DIR/morphbpe_trmorph_32768_metrics.json"

python -m scripts.compare_tokenizer_metrics \
    --metric "$METRICS_DIR/bpe_32768_metrics.json" \
    --metric "$METRICS_DIR/morphbpe_trmorph_32768_metrics.json" \
    --output-md "$METRICS_DIR/tokenizer_metrics_comparison.md" \
    --output-json "$METRICS_DIR/tokenizer_metrics_comparison.json"

echo "Tokenizer metrics written under $METRICS_DIR"
find "$METRICS_DIR" -maxdepth 1 -type f -print
