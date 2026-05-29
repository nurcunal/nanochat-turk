#!/bin/bash

# Tokenizer ablation scaffold. By default this sweeps raw BPE vocab sizes while
# keeping the dataset source, architecture, and training horizon policy fixed.

set -euo pipefail

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat-turk-ablation}"
mkdir -p "$NANOCHAT_BASE_DIR"

if [ -z "${SKIP_SETUP:-}" ]; then
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -d ".venv" ] || uv venv
    uv sync --extra gpu
fi
source .venv/bin/activate

if [ -z "${WANDB_RUN:-}" ]; then
    WANDB_RUN=dummy
fi

if [ -z "${NPROC_PER_NODE:-}" ]; then
    NPROC_PER_NODE="$(python -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 1)')"
fi
HARDWARE_PROFILE="${HARDWARE_PROFILE:-a100}" # a100|h100|generic
DEPTH="${DEPTH:-12}"
TRAIN_SHARDS="${TRAIN_SHARDS:-64}"
TOKENIZER_CHARS="${TOKENIZER_CHARS:-2000000000}"
VOCAB_SIZES=(${VOCAB_SIZES:-32768 65536 131072})

case "$HARDWARE_PROFILE" in
    a100)
        DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
        WINDOW_PATTERN="${WINDOW_PATTERN:-L}"
        ;;
    h100)
        DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-32}"
        WINDOW_PATTERN="${WINDOW_PATTERN:-SSSL}"
        ;;
    *)
        DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
        WINDOW_PATTERN="${WINDOW_PATTERN:-L}"
        ;;
esac

python -m nanochat.dataset -n "$TRAIN_SHARDS" -w "${DATASET_WORKERS:-8}"

for vocab in "${VOCAB_SIZES[@]}"; do
    export NANOCHAT_TOKENIZER_NAME="bpe_${vocab}"
    MODEL_TAG="tr_ablation_d${DEPTH}_${NANOCHAT_TOKENIZER_NAME}"

    python -m scripts.tok_train \
        --vocab-size="$vocab" \
        --max-chars="$TOKENIZER_CHARS" \
        --tokenizer-name="$NANOCHAT_TOKENIZER_NAME"
    python -m scripts.tokenizer_metrics \
        --max-docs="${TOKENIZER_METRIC_DOCS:-10000}" \
        --output="$NANOCHAT_BASE_DIR/${NANOCHAT_TOKENIZER_NAME}_metrics.json"

    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
        --depth="$DEPTH" \
        --target-param-data-ratio="${TARGET_PARAM_DATA_RATIO:-20}" \
        --target-param-count="${TARGET_PARAM_COUNT:-total}" \
        --device-batch-size="$DEVICE_BATCH_SIZE" \
        --window-pattern="$WINDOW_PATTERN" \
        --model-tag="$MODEL_TAG" \
        --core-metric-every=-1 \
        --sample-every=-1 \
        --save-every=-1 \
        --run="${WANDB_RUN}_${NANOCHAT_TOKENIZER_NAME}"

    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval -- \
        --eval=bpb \
        --device-batch-size="$DEVICE_BATCH_SIZE" \
        --model-tag="$MODEL_TAG"

    if [ "${RUN_CETVEL:-0}" = "1" ]; then
        python -m scripts.cetvel_eval --suite="${CETVEL_SUITE:-fast}" --model-tag="$MODEL_TAG" --batch-size="${CETVEL_BATCH_SIZE:-1}"
    fi
done
