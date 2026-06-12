#!/bin/bash -l

# Compute full TRmorph-reference tokenizer metrics for completed large-vocab
# raw BPE tokenizers. This complements the raw 10k metrics produced by the
# tokenizer finalizer with the MorphBPE-paper metrics used in the comparison
# table.

#SBATCH --job-name=tok-metrics-bpe-large
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --mem=128G
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

echo "HOSTNAME=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "REPO_DIR=$REPO_DIR"
echo "PYTHON=$(command -v python3)"
python3 --version

if [ ! -d ".venv" ]; then
    echo "Missing .venv in $REPO_DIR" >&2
    exit 2
fi
source .venv/bin/activate

SEGMENTED_DATA_DIR="${SEGMENTED_DATA_DIR:-$HOME/nanochat-turk-morphbpe-trmorph-32768/segmented/trmorph}"
METRICS_DOCS="${METRICS_DOCS:-50000}"
WORD_METRICS="${WORD_METRICS:-500000}"
MORPH_METRICS="${MORPH_METRICS:-500000}"
NUM_THREADS="${NUM_THREADS:-32}"

if [ ! -f "$SEGMENTED_DATA_DIR/fineweb2_manifest.json" ]; then
    echo "Missing segmented dataset manifest at $SEGMENTED_DATA_DIR/fineweb2_manifest.json" >&2
    exit 2
fi

COMMON_METRIC_ARGS=(
    --data-dir "$SEGMENTED_DATA_DIR"
    --text-column segmented_text
    --input-has-morph-boundaries
    --max-docs "$METRICS_DOCS"
    --max-word-metrics "$WORD_METRICS"
    --max-morph-metrics "$MORPH_METRICS"
    --morph-consistency-max-words 50000
    --morph-consistency-clusters 100
    --morph-consistency-pairs-per-cluster 50
    --morph-consistency-resamples 10
    --num-threads "$NUM_THREADS"
    --no-report
)

run_metrics() {
    local tokenizer_name="$1"
    local tokenizer_dir="$2"
    local output="$3"

    if [ ! -f "$tokenizer_dir/tokenizer.pkl" ]; then
        echo "Missing tokenizer at $tokenizer_dir" >&2
        exit 2
    fi
    mkdir -p "$(dirname "$output")"
    python -m scripts.tokenizer_metrics \
        --tokenizer-dir "$tokenizer_dir" \
        "${COMMON_METRIC_ARGS[@]}" \
        --output "$output"
    cp "$output" "docs/tokenizer_tests/tokenizer_metrics/${tokenizer_name}_metrics.json"
}

mkdir -p docs/tokenizer_tests/tokenizer_metrics

run_metrics \
    bpe_65536 \
    "$HOME/nanochat-turk-bpe-65536/tokenizers/bpe_65536" \
    "$REPO_DIR/artifacts/tokenizers/bpe_65536/metrics/raw_vs_trmorph_reference_metrics.json"

run_metrics \
    bpe_131072 \
    "$HOME/nanochat-turk-bpe-131072/tokenizers/bpe_131072" \
    "$REPO_DIR/artifacts/tokenizers/bpe_131072/metrics/raw_vs_trmorph_reference_metrics.json"

python -m scripts.compare_tokenizer_metrics \
    --metric docs/tokenizer_tests/tokenizer_metrics/morphbpe_trmorph_32768_metrics.json \
    --metric docs/tokenizer_tests/tokenizer_metrics/morphbpe_zemberek_32768_metrics.json \
    --metric docs/tokenizer_tests/tokenizer_metrics/bpe_32768_metrics.json \
    --metric docs/tokenizer_tests/tokenizer_metrics/bpe_65536_metrics.json \
    --metric docs/tokenizer_tests/tokenizer_metrics/bpe_131072_metrics.json \
    --output-md docs/tokenizer_tests/tokenizer_metrics/tokenizer_metrics_comparison.md \
    --output-json docs/tokenizer_tests/tokenizer_metrics/tokenizer_metrics_comparison.json

echo "Wrote large BPE tokenizer metrics:"
echo "  $REPO_DIR/artifacts/tokenizers/bpe_65536/metrics/raw_vs_trmorph_reference_metrics.json"
echo "  $REPO_DIR/artifacts/tokenizers/bpe_131072/metrics/raw_vs_trmorph_reference_metrics.json"
echo "  $REPO_DIR/docs/tokenizer_tests/tokenizer_metrics/tokenizer_metrics_comparison.md"
