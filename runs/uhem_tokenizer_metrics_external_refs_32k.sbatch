#!/bin/bash -l

# Recompute MorphBPE-paper metrics for public external tokenizer references.
#
# This job intentionally does not rerun the local 32k tokenizers. It measures
# the configured Hugging Face tokenizer-file references on the same
# TRmorph-segmented corpus sample used by the checked-in local tokenizer table,
# then writes a combined comparison table with local + external rows.

#SBATCH --job-name=nanochat-tokenizer-extrefs32k
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
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

export PATH="$HOME/.local/bin:$PATH"
export MORPH_BASE_DIR="${MORPH_BASE_DIR:-$HOME/nanochat-turk-morphbpe-trmorph-32768}"
export SEGMENTED_DATA_DIR="${SEGMENTED_DATA_DIR:-$MORPH_BASE_DIR/datasets/trmorph-segmented}"
export METRICS_DIR="${METRICS_DIR:-$MORPH_BASE_DIR/report/tokenizer_metrics_32k_external_refs_50k}"
export LOCAL_METRIC_DIR="${LOCAL_METRIC_DIR:-$REPO_DIR/docs/tokenizer_tests/tokenizer_metrics}"
export INCLUDE_LOCAL_COMPARISON="${INCLUDE_LOCAL_COMPARISON:-1}"
export HF_HOME="${HF_HOME:-$MORPH_BASE_DIR/huggingface}"
export MAX_DOCS="${MAX_DOCS:-50000}"
export MAX_WORD_METRICS="${MAX_WORD_METRICS:-200000}"
export MAX_MORPH_METRICS="${MAX_MORPH_METRICS:-200000}"
export MORPH_CONSISTENCY_MAX_WORDS="${MORPH_CONSISTENCY_MAX_WORDS:-50000}"
export MORPH_CONSISTENCY_CLUSTERS="${MORPH_CONSISTENCY_CLUSTERS:-100}"
export MORPH_CONSISTENCY_PAIRS_PER_CLUSTER="${MORPH_CONSISTENCY_PAIRS_PER_CLUSTER:-50}"
export MORPH_CONSISTENCY_RESAMPLES="${MORPH_CONSISTENCY_RESAMPLES:-10}"
export NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-32}}"
export TOKENIZER_THREADS_PER_WORKER="${TOKENIZER_THREADS_PER_WORKER:-1}"
export PROGRESS_EVERY="${PROGRESS_EVERY:-5}"
export HF_BATCH_SIZE="${HF_BATCH_SIZE:-1000}"
# Semicolon-separated external tokenizer specs:
#   repo_id|output_name|implementation
export HF_TOKENIZER_SPECS="${HF_TOKENIZER_SPECS:-vngrs-ai/Kumru-2B|kumru_2b|bpe;ytu-ce-cosmos/turkish-gpt2|cosmos_turkish_gpt2|bpe}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

mkdir -p "$METRICS_DIR" "$HF_HOME"

if [ ! -d ".venv" ]; then
    echo "Missing .venv in $REPO_DIR. Run tokenizer prep first." >&2
    exit 2
fi
source .venv/bin/activate

if [ ! -f "$SEGMENTED_DATA_DIR/fineweb2_manifest.json" ]; then
    echo "Missing segmented dataset manifest at $SEGMENTED_DATA_DIR/fineweb2_manifest.json" >&2
    exit 2
fi

echo "repo_dir=$REPO_DIR"
echo "metrics_dir=$METRICS_DIR"
echo "segmented_data_dir=$SEGMENTED_DATA_DIR"
echo "local_metric_dir=$LOCAL_METRIC_DIR include_local_comparison=$INCLUDE_LOCAL_COMPARISON"
echo "max_docs=$MAX_DOCS max_word_metrics=$MAX_WORD_METRICS max_morph_metrics=$MAX_MORPH_METRICS"
echo "num_workers=$NUM_WORKERS tokenizer_threads_per_worker=$TOKENIZER_THREADS_PER_WORKER"
echo "hf_tokenizer_specs=$HF_TOKENIZER_SPECS"

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
    --hf-batch-size "$HF_BATCH_SIZE"
    --no-report
)

run_hf_tokenizer_metrics() {
    local model_id="$1"
    local output_name="$2"
    local implementation="${3:-}"
    local impl_args=()

    if [ -n "$implementation" ]; then
        impl_args=(--tokenizer-implementation "$implementation")
    fi

    echo "Pre-caching tokenizer files for $model_id"
    python - "$model_id" "$output_name" "${implementation:-}" "$HF_BATCH_SIZE" <<'PY'
import sys

from scripts.tokenizer_metrics import HuggingFaceTokenizerAdapter

model_id, name, implementation, batch_size = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
HuggingFaceTokenizerAdapter.from_identifier(
    model_id,
    name=name,
    implementation=implementation,
    batch_size=batch_size,
)
print(f"cached {model_id}")
PY

    python -m scripts.tokenizer_metrics \
        --hf-tokenizer "$model_id" \
        --tokenizer-name "$output_name" \
        "${impl_args[@]}" \
        "${COMMON_METRIC_ARGS[@]}" \
        --output "$METRICS_DIR/${output_name}_metrics.json"
}

METRIC_ARGS=()

if [ "$INCLUDE_LOCAL_COMPARISON" = "1" ]; then
    for local_name in \
        morphbpe_trmorph_32768 \
        morphbpe_zemberek_32768 \
        bpe_32768
    do
        local_metric="$LOCAL_METRIC_DIR/${local_name}_metrics.json"
        if [ -f "$local_metric" ]; then
            METRIC_ARGS+=(--metric "$local_metric")
        else
            echo "Skipping missing local comparison metric: $local_metric"
        fi
    done
fi

IFS=';' read -ra HF_SPECS <<< "$HF_TOKENIZER_SPECS"
for spec in "${HF_SPECS[@]}"; do
    [ -n "$spec" ] || continue
    IFS='|' read -r model_id output_name implementation <<< "$spec"
    if [ -z "${model_id:-}" ] || [ -z "${output_name:-}" ]; then
        echo "Bad HF tokenizer spec: $spec" >&2
        exit 2
    fi
    run_hf_tokenizer_metrics "$model_id" "$output_name" "${implementation:-}"
    METRIC_ARGS+=(--metric "$METRICS_DIR/${output_name}_metrics.json")
done

python -m scripts.compare_tokenizer_metrics \
    "${METRIC_ARGS[@]}" \
    --output-md "$METRICS_DIR/tokenizer_metrics_comparison_with_external.md" \
    --output-json "$METRICS_DIR/tokenizer_metrics_comparison_with_external.json"

echo "External tokenizer metrics written under $METRICS_DIR"
find "$METRICS_DIR" -maxdepth 1 -type f -print
