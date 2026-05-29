#!/bin/bash

# Turkish foundation base-model run. This intentionally stops at the base model;
# SFT will be added after the foundation pipeline is validated.

set -euo pipefail

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat-turk}"
VOCAB_SIZE="${VOCAB_SIZE:-32768}"
export NANOCHAT_TOKENIZER_NAME="${NANOCHAT_TOKENIZER_NAME:-bpe_${VOCAB_SIZE}}"
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
DEPTH="${DEPTH:-24}"
TRAIN_SHARDS="${TRAIN_SHARDS:--1}" # -1 downloads all FineWeb-2 Turkish shards plus final val shard.
MODEL_TAG="${MODEL_TAG:-tr_d${DEPTH}_${NANOCHAT_TOKENIZER_NAME}_chinchilla20}"

case "$HARDWARE_PROFILE" in
    a100)
        DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-4}"
        WINDOW_PATTERN="${WINDOW_PATTERN:-L}"
        USE_FP8="${USE_FP8:-0}"
        ;;
    h100)
        DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
        WINDOW_PATTERN="${WINDOW_PATTERN:-SSSL}"
        USE_FP8="${USE_FP8:-1}"
        ;;
    *)
        DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-4}"
        WINDOW_PATTERN="${WINDOW_PATTERN:-L}"
        USE_FP8="${USE_FP8:-0}"
        ;;
esac

python -m nanochat.report reset
python -m nanochat.dataset -n "$TRAIN_SHARDS" -w "${DATASET_WORKERS:-8}"
python -m scripts.tok_train \
    --vocab-size="$VOCAB_SIZE" \
    --max-chars="${TOKENIZER_CHARS:-2000000000}" \
    --tokenizer-name="$NANOCHAT_TOKENIZER_NAME"
python -m scripts.tok_eval
python -m scripts.tokenizer_metrics --max-docs="${TOKENIZER_METRIC_DOCS:-10000}"

FP8_ARGS=()
if [ "${USE_FP8:-1}" = "1" ]; then
    FP8_ARGS=(--fp8)
fi

TRAIN_ARGS=(
    --depth="$DEPTH" \
    --target-param-data-ratio=20 \
    --target-param-count=total \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --window-pattern="$WINDOW_PATTERN" \
    --model-tag="$MODEL_TAG" \
    --run="$WANDB_RUN"
)
[ -z "${HEAD_DIM:-}" ] || TRAIN_ARGS+=(--head-dim="$HEAD_DIM")
[ -z "${MAX_SEQ_LEN:-}" ] || TRAIN_ARGS+=(--max-seq-len="$MAX_SEQ_LEN")
[ -z "${TOTAL_BATCH_SIZE:-}" ] || TRAIN_ARGS+=(--total-batch-size="$TOTAL_BATCH_SIZE")
[ -z "${NUM_ITERATIONS:-}" ] || TRAIN_ARGS+=(--num-iterations="$NUM_ITERATIONS")
[ -z "${EVAL_EVERY:-}" ] || TRAIN_ARGS+=(--eval-every="$EVAL_EVERY")
[ -z "${EVAL_TOKENS:-}" ] || TRAIN_ARGS+=(--eval-tokens="$EVAL_TOKENS")
[ -z "${CORE_METRIC_EVERY:-}" ] || TRAIN_ARGS+=(--core-metric-every="$CORE_METRIC_EVERY")
[ -z "${SAMPLE_EVERY:-}" ] || TRAIN_ARGS+=(--sample-every="$SAMPLE_EVERY")
[ -z "${SAVE_EVERY:-}" ] || TRAIN_ARGS+=(--save-every="$SAVE_EVERY")

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
    "${TRAIN_ARGS[@]}" \
    "${FP8_ARGS[@]}"

EVAL_ARGS=(
    --eval=bpb,sample \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --model-tag="$MODEL_TAG"
)
[ -z "${BASE_EVALS:-}" ] || EVAL_ARGS[0]="--eval=$BASE_EVALS"
[ -z "${SPLIT_TOKENS:-}" ] || EVAL_ARGS+=(--split-tokens="$SPLIT_TOKENS")

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval -- \
    "${EVAL_ARGS[@]}"

if [ "${RUN_CETVEL:-0}" = "1" ]; then
    python -m scripts.cetvel_eval --suite="${CETVEL_SUITE:-core}" --model-tag="$MODEL_TAG" --batch-size="${CETVEL_BATCH_SIZE:-1}"
fi

python -m nanochat.report generate
