#!/bin/bash -l

# Compute full-corpus TRmorph-reference tokenizer metrics for all completed
# local tokenizer artifacts, using process workers over parquet row groups.
#
# This is the paper-facing companion to the 50k-document sample comparison. It
# uses the entire segmented corpus for corpus-level docs/bytes/tokens/fertility
# and boundary-crossing metrics. Morphology alignment/consistency samples remain
# capped by MAX_WORD_METRICS/MAX_MORPH_METRICS for reproducibility and runtime.

#SBATCH --job-name=tok-metrics-full
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=2-00:00:00
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

SEGMENTED_DATA_DIR="${SEGMENTED_DATA_DIR:-$HOME/nanochat-turk-morphbpe-trmorph-32768/datasets/trmorph-segmented}"
METRICS_DIR="${METRICS_DIR:-$REPO_DIR/docs/tokenizer_tests/tokenizer_metrics_full}"
MAX_DOCS="${MAX_DOCS:-0}"
MAX_WORD_METRICS="${MAX_WORD_METRICS:-500000}"
MAX_MORPH_METRICS="${MAX_MORPH_METRICS:-500000}"
MORPH_CONSISTENCY_MAX_WORDS="${MORPH_CONSISTENCY_MAX_WORDS:-50000}"
MORPH_CONSISTENCY_CLUSTERS="${MORPH_CONSISTENCY_CLUSTERS:-100}"
MORPH_CONSISTENCY_PAIRS_PER_CLUSTER="${MORPH_CONSISTENCY_PAIRS_PER_CLUSTER:-50}"
MORPH_CONSISTENCY_RESAMPLES="${MORPH_CONSISTENCY_RESAMPLES:-10}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-32}}"
TOKENIZER_THREADS_PER_WORKER="${TOKENIZER_THREADS_PER_WORKER:-1}"
PROGRESS_EVERY="${PROGRESS_EVERY:-5}"
MAX_IN_FLIGHT="${MAX_IN_FLIGHT:-64}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

if [ ! -f "$SEGMENTED_DATA_DIR/fineweb2_manifest.json" ]; then
    echo "Missing segmented dataset manifest at $SEGMENTED_DATA_DIR/fineweb2_manifest.json" >&2
    exit 2
fi

mkdir -p "$METRICS_DIR"

echo "segmented_data_dir=$SEGMENTED_DATA_DIR"
echo "metrics_dir=$METRICS_DIR"
echo "max_docs=$MAX_DOCS max_word_metrics=$MAX_WORD_METRICS max_morph_metrics=$MAX_MORPH_METRICS"
echo "num_workers=$NUM_WORKERS tokenizer_threads_per_worker=$TOKENIZER_THREADS_PER_WORKER max_in_flight=$MAX_IN_FLIGHT"

COMMON_METRIC_ARGS=(
    --data-dir "$SEGMENTED_DATA_DIR"
    --text-column segmented_text
    --input-has-morph-boundaries
    --max-docs "$MAX_DOCS"
    --max-word-metrics "$MAX_WORD_METRICS"
    --max-morph-metrics "$MAX_MORPH_METRICS"
    --morph-consistency-max-words "$MORPH_CONSISTENCY_MAX_WORDS"
    --morph-consistency-clusters "$MORPH_CONSISTENCY_CLUSTERS"
    --morph-consistency-pairs-per-cluster "$MORPH_CONSISTENCY_PAIRS_PER_CLUSTER"
    --morph-consistency-resamples "$MORPH_CONSISTENCY_RESAMPLES"
    --workers "$NUM_WORKERS"
    --num-threads "$TOKENIZER_THREADS_PER_WORKER"
    --progress-every "$PROGRESS_EVERY"
    --max-in-flight "$MAX_IN_FLIGHT"
    --no-report
)

METRIC_ARGS=()

run_metrics_if_present() {
    local tokenizer_name="$1"
    local tokenizer_dir="$2"
    local artifact_metrics_dir="$3"
    local output="$METRICS_DIR/${tokenizer_name}_metrics.json"

    if [ ! -f "$tokenizer_dir/tokenizer.pkl" ]; then
        echo "Skipping $tokenizer_name; missing tokenizer at $tokenizer_dir"
        return 0
    fi

    echo "Running full metrics for $tokenizer_name"
    python -m scripts.tokenizer_metrics \
        --tokenizer-dir "$tokenizer_dir" \
        "${COMMON_METRIC_ARGS[@]}" \
        --output "$output"

    mkdir -p "$artifact_metrics_dir"
    cp "$output" "$artifact_metrics_dir/full_trmorph_reference_metrics.json"
    METRIC_ARGS+=(--metric "$output")
}

run_metrics_if_present \
    bpe_32768 \
    "$HOME/nanochat-turk-d20-bpe32k/tokenizers/bpe_32768" \
    "$REPO_DIR/artifacts/tokenizers/bpe_32768/metrics"

run_metrics_if_present \
    morphbpe_trmorph_32768 \
    "$HOME/nanochat-turk-morphbpe-trmorph-32768/tokenizers/morphbpe_trmorph_32768" \
    "$REPO_DIR/artifacts/tokenizers/morphbpe_trmorph_32768/metrics"

run_metrics_if_present \
    morphbpe_zemberek_32768 \
    "$HOME/nanochat-turk-morphbpe-zemberek-32768/tokenizers/morphbpe_zemberek_32768" \
    "$REPO_DIR/artifacts/tokenizers/morphbpe_zemberek_32768/metrics"

run_metrics_if_present \
    morphbpe_tdelight_32768 \
    "$HOME/nanochat-turk-morphbpe-tdelight-32768/tokenizers/morphbpe_tdelight_32768" \
    "$REPO_DIR/artifacts/tokenizers/morphbpe_tdelight_32768/metrics"

run_metrics_if_present \
    bpe_65536 \
    "$HOME/nanochat-turk-bpe-65536/tokenizers/bpe_65536" \
    "$REPO_DIR/artifacts/tokenizers/bpe_65536/metrics"

run_metrics_if_present \
    morphbpe_trmorph_65536 \
    "$HOME/nanochat-turk-morphbpe-trmorph-65536/tokenizers/morphbpe_trmorph_65536" \
    "$REPO_DIR/artifacts/tokenizers/morphbpe_trmorph_65536/metrics"

run_metrics_if_present \
    morphbpe_zemberek_65536 \
    "$HOME/nanochat-turk-morphbpe-zemberek-65536/tokenizers/morphbpe_zemberek_65536" \
    "$REPO_DIR/artifacts/tokenizers/morphbpe_zemberek_65536/metrics"

run_metrics_if_present \
    morphbpe_tdelight_65536 \
    "$HOME/nanochat-turk-morphbpe-tdelight-65536/tokenizers/morphbpe_tdelight_65536" \
    "$REPO_DIR/artifacts/tokenizers/morphbpe_tdelight_65536/metrics"

run_metrics_if_present \
    bpe_131072 \
    "$HOME/nanochat-turk-bpe-131072/tokenizers/bpe_131072" \
    "$REPO_DIR/artifacts/tokenizers/bpe_131072/metrics"

run_metrics_if_present \
    morphbpe_trmorph_131072 \
    "$HOME/nanochat-turk-morphbpe-trmorph-131072/tokenizers/morphbpe_trmorph_131072" \
    "$REPO_DIR/artifacts/tokenizers/morphbpe_trmorph_131072/metrics"

run_metrics_if_present \
    morphbpe_zemberek_131072 \
    "$HOME/nanochat-turk-morphbpe-zemberek-131072/tokenizers/morphbpe_zemberek_131072" \
    "$REPO_DIR/artifacts/tokenizers/morphbpe_zemberek_131072/metrics"

run_metrics_if_present \
    morphbpe_tdelight_131072 \
    "$HOME/nanochat-turk-morphbpe-tdelight-131072/tokenizers/morphbpe_tdelight_131072" \
    "$REPO_DIR/artifacts/tokenizers/morphbpe_tdelight_131072/metrics"

if [ "${#METRIC_ARGS[@]}" -eq 0 ]; then
    echo "No tokenizer metrics were produced." >&2
    exit 2
fi

python -m scripts.compare_tokenizer_metrics \
    "${METRIC_ARGS[@]}" \
    --output-md "$METRICS_DIR/tokenizer_metrics_comparison.md" \
    --output-json "$METRICS_DIR/tokenizer_metrics_comparison.json"

echo "Wrote full-corpus tokenizer metrics:"
find "$METRICS_DIR" -maxdepth 1 -type f -print
