#!/bin/bash

# Altay/UHeM one-A100 smoke test using the nakane project.
# UHeM asks <=48 hour A100 jobs to use gpu2dq; use a100q for longer runs.
#
# Submit from the repo root on altay with:
#   sbatch runs/uhem_smoke_nakane_a100.sbatch

#SBATCH --job-name=nanochat-tr-smoke-a100
#SBATCH --account=nakane
#SBATCH --partition=gpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

module purge
module load cuda/cuda-12.5-a100q
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0

echo "HOSTNAME=$(hostname)"
echo "PYTHON=$(command -v python3)"
python3 --version
echo "UV=$(command -v uv || true)"
nvidia-smi

export UV_PYTHON="${UV_PYTHON:-$(command -v python3)}"
export HARDWARE_PROFILE="${HARDWARE_PROFILE:-a100}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/nanochat-turk-smoke-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$NANOCHAT_BASE_DIR/uv-cache}"
export HF_HOME="${HF_HOME:-$NANOCHAT_BASE_DIR/huggingface}"
export TRAIN_SHARDS="${TRAIN_SHARDS:-2}"
export VOCAB_SIZE="${VOCAB_SIZE:-32768}"
export NANOCHAT_TOKENIZER_NAME="${NANOCHAT_TOKENIZER_NAME:-bpe_${VOCAB_SIZE}_uhem_smoke}"
export TOKENIZER_CHARS="${TOKENIZER_CHARS:-50000000}"
export TOKENIZER_METRIC_DOCS="${TOKENIZER_METRIC_DOCS:-1000}"
export DEPTH="${DEPTH:-2}"
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
