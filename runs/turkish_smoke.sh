#!/bin/bash

# Tiny end-to-end Turkish base-model smoke run. This is for code-path checks,
# not model quality.

set -euo pipefail

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat-turk}"
export NANOCHAT_TOKENIZER_NAME="${NANOCHAT_TOKENIZER_NAME:-bpe_32k_smoke}"
mkdir -p "$NANOCHAT_BASE_DIR"

UV_EXTRA="${UV_EXTRA:-cpu}"
if [ -z "${SKIP_SETUP:-}" ]; then
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -d ".venv" ] || uv venv
    uv sync --extra "$UV_EXTRA"
fi
source .venv/bin/activate

if [ -z "${WANDB_RUN:-}" ]; then
    WANDB_RUN=dummy
fi

python -m nanochat.report reset
python -m nanochat.dataset -n "${TRAIN_SHARDS:-2}" -w "${DATASET_WORKERS:-2}"
python -m scripts.tok_train --vocab-size=32768 --max-chars="${TOKENIZER_CHARS:-50000000}" --tokenizer-name="$NANOCHAT_TOKENIZER_NAME"
python -m scripts.tokenizer_metrics --max-docs=1000

python -m scripts.base_train \
    --depth="${DEPTH:-4}" \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=256 \
    --device-batch-size=2 \
    --total-batch-size=1024 \
    --num-iterations="${NUM_ITERATIONS:-20}" \
    --eval-every=10 \
    --eval-tokens=4096 \
    --core-metric-every=-1 \
    --sample-every=10 \
    --save-every=-1 \
    --target-param-data-ratio=20 \
    --target-param-count=total \
    --model-tag="${MODEL_TAG:-tr_smoke_${NANOCHAT_TOKENIZER_NAME}}" \
    --run="$WANDB_RUN"

python -m scripts.base_eval --eval=bpb,sample --device-batch-size=1 --split-tokens=4096 --model-tag="${MODEL_TAG:-tr_smoke_${NANOCHAT_TOKENIZER_NAME}}"
python -m nanochat.report generate
