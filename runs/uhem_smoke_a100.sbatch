#!/bin/bash

# Tiny Chinchilla-ratio A100 Slurm smoke test for UHeM-style clusters. This
# keeps the standard 32k BPE path but shrinks the model to about 6.3M params.
#
# Submit with:
#   sbatch runs/uhem_smoke_a100.sbatch

#SBATCH --job-name=nanochat-tr-smoke-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=0
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# Uncomment/adapt these for the UHeM software stack if needed.
# module purge
# module load cuda/12.8

export HARDWARE_PROFILE="${HARDWARE_PROFILE:-a100}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-${SCRATCH:-$HOME/.cache}/nanochat-turk-smoke}"
export TRAIN_SHARDS="${TRAIN_SHARDS:-2}"
export VOCAB_SIZE="${VOCAB_SIZE:-32768}"
export NANOCHAT_TOKENIZER_NAME="${NANOCHAT_TOKENIZER_NAME:-bpe_${VOCAB_SIZE}_uhem_smoke}"
export TOKENIZER_CHARS="${TOKENIZER_CHARS:-50000000}"
export TOKENIZER_METRIC_DOCS="${TOKENIZER_METRIC_DOCS:-1000}"
export DEPTH="${DEPTH:-1}"
export HEAD_DIM="${HEAD_DIM:-64}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-256}"
export DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-64}"
export TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-131072}"
export EVAL_EVERY="${EVAL_EVERY:-100}"
export EVAL_TOKENS="${EVAL_TOKENS:-65536}"
export CORE_METRIC_EVERY="${CORE_METRIC_EVERY:--1}"
export SAMPLE_EVERY="${SAMPLE_EVERY:-250}"
export SAVE_EVERY="${SAVE_EVERY:--1}"
export SPLIT_TOKENS="${SPLIT_TOKENS:-65536}"
export RUN_CETVEL="${RUN_CETVEL:-0}"

bash runs/turkish_foundation.sh
