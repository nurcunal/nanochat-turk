#!/bin/bash -l

# Compare tokenizer-only metrics for local 32k tokenizers and public Turkish
# Hugging Face tokenizers.
#
# This runs on the TRmorph-segmented corpus as a boundary reference, strips the
# boundary marker before tokenizer encoding, and reports both standard tokenizer
# compression/fertility metrics and token-crosses-morpheme-boundary metrics.

#SBATCH --job-name=nanochat-tokenizer-metrics32k-full
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=2-00:00:00
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
export RAW_BASE_DIR="${RAW_BASE_DIR:-$HOME/nanochat-turk-d20-bpe32k}"
export MORPH_BASE_DIR="${MORPH_BASE_DIR:-$HOME/nanochat-turk-morphbpe-trmorph-32768}"
export SEGMENTED_DATA_DIR="${SEGMENTED_DATA_DIR:-$MORPH_BASE_DIR/datasets/trmorph-segmented}"
export METRICS_DIR="${METRICS_DIR:-$MORPH_BASE_DIR/report/tokenizer_metrics_32k_full}"
export HF_HOME="${HF_HOME:-$MORPH_BASE_DIR/huggingface}"
# 0 means full corpus. The old 50k-doc run was only a preliminary sample.
export MAX_DOCS="${MAX_DOCS:-0}"
export MAX_WORD_METRICS="${MAX_WORD_METRICS:-500000}"
export NUM_THREADS="${NUM_THREADS:-${SLURM_CPUS_PER_TASK:-32}}"
export INCLUDE_HF="${INCLUDE_HF:-0}"

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

COMMON_METRIC_ARGS=(
    --data-dir "$SEGMENTED_DATA_DIR"
    --text-column segmented_text
    --input-has-morph-boundaries
    --max-docs "$MAX_DOCS"
    --max-word-metrics "$MAX_WORD_METRICS"
    --num-threads "$NUM_THREADS"
    --no-report
)

run_local_tokenizer_metrics() {
    local tokenizer_dir="$1"
    local output_name="$2"

    python -m scripts.tokenizer_metrics \
        --tokenizer-dir "$tokenizer_dir" \
        "${COMMON_METRIC_ARGS[@]}" \
        --output "$METRICS_DIR/${output_name}_metrics.json"
}

run_hf_tokenizer_metrics() {
    local model_id="$1"
    local output_name="$2"
    local implementation="${3:-}"
    local impl_args=()

    if [ -n "$implementation" ]; then
        impl_args=(--tokenizer-implementation "$implementation")
    fi

    python -m scripts.tokenizer_metrics \
        --hf-tokenizer "$model_id" \
        --tokenizer-name "$output_name" \
        "${impl_args[@]}" \
        "${COMMON_METRIC_ARGS[@]}" \
        --output "$METRICS_DIR/${output_name}_metrics.json"
}

run_local_tokenizer_metrics "$RAW_BASE_DIR/tokenizers/bpe_32768" bpe_32768
run_local_tokenizer_metrics "$MORPH_BASE_DIR/tokenizers/morphbpe_trmorph_32768" morphbpe_trmorph_32768

METRIC_ARGS=(
    --metric "$METRICS_DIR/bpe_32768_metrics.json"
    --metric "$METRICS_DIR/morphbpe_trmorph_32768_metrics.json"
)

if [ "$INCLUDE_HF" = "1" ]; then
    run_hf_tokenizer_metrics vngrs-ai/Kumru-2B kumru_2b
    run_hf_tokenizer_metrics dbmdz/bert-base-turkish-cased berturk_cased wordpiece
    run_hf_tokenizer_metrics ytu-ce-cosmos/turkish-gpt2 cosmos_turkish_gpt2
    run_hf_tokenizer_metrics boun-tabi-LMG/TURNA turna
    run_hf_tokenizer_metrics vngrs-ai/VBART-Large-Base vbart_large_base
    METRIC_ARGS+=(
        --metric "$METRICS_DIR/kumru_2b_metrics.json"
        --metric "$METRICS_DIR/berturk_cased_metrics.json"
        --metric "$METRICS_DIR/cosmos_turkish_gpt2_metrics.json"
        --metric "$METRICS_DIR/turna_metrics.json"
        --metric "$METRICS_DIR/vbart_large_base_metrics.json"
    )
fi

python -m scripts.compare_tokenizer_metrics \
    "${METRIC_ARGS[@]}" \
    --output-md "$METRICS_DIR/tokenizer_metrics_comparison.md" \
    --output-json "$METRICS_DIR/tokenizer_metrics_comparison.json"

echo "Tokenizer metrics written under $METRICS_DIR"
find "$METRICS_DIR" -maxdepth 1 -type f -print
