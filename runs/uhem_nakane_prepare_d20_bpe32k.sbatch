#!/bin/bash

# CPU-only preparation for the Altay/UHeM d20/32k production run.
# This installs/syncs the GPU Python environment and downloads all FineWeb-2
# Turkish shards before the scarce 4xA100 allocation is used.
#
# Submit from the repo root on Altay with:
#   sbatch runs/uhem_nakane_prepare_d20_bpe32k.sbatch

#SBATCH --job-name=nanochat-prep-d20-bpe32k
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

module purge
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0

echo "HOSTNAME=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "PYTHON=$(command -v python3)"
python3 --version
echo "UV=$(command -v uv || true)"

export PATH="$HOME/.local/bin:$PATH"
export UV_PYTHON="${UV_PYTHON:-$(command -v python3)}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/nanochat-turk-d20-bpe32k}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$NANOCHAT_BASE_DIR/uv-cache}"
export HF_HOME="${HF_HOME:-$NANOCHAT_BASE_DIR/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$NANOCHAT_BASE_DIR/hf-datasets-cache}"

export TRAIN_SHARDS="${TRAIN_SHARDS:--1}"
export DATASET_WORKERS="${DATASET_WORKERS:-8}"
export VOCAB_SIZE="${VOCAB_SIZE:-32768}"
export NANOCHAT_TOKENIZER_NAME="${NANOCHAT_TOKENIZER_NAME:-bpe_32768}"
export TOKENIZER_CHARS="${TOKENIZER_CHARS:-2000000000}"
export TOKENIZER_METRIC_DOCS="${TOKENIZER_METRIC_DOCS:-10000}"

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv ${UV_PYTHON:+--python "$UV_PYTHON"}
uv sync --extra gpu
source .venv/bin/activate

python -m nanochat.dataset -n "$TRAIN_SHARDS" -w "$DATASET_WORKERS"
python -m nanochat.dataset --list-local | tee "$NANOCHAT_BASE_DIR/local_parquets.txt"
python -m scripts.tok_train \
    --vocab-size="$VOCAB_SIZE" \
    --max-chars="$TOKENIZER_CHARS" \
    --tokenizer-name="$NANOCHAT_TOKENIZER_NAME"
python -m scripts.tok_eval
python -m scripts.tokenizer_metrics --max-docs="$TOKENIZER_METRIC_DOCS"

echo "Prepared data under: $NANOCHAT_BASE_DIR"
du -sh "$NANOCHAT_BASE_DIR" "$NANOCHAT_BASE_DIR/base_data_fineweb2_tur_latn" || true
du -sh "$NANOCHAT_BASE_DIR/tokenizers/$NANOCHAT_TOKENIZER_NAME" || true
df -h "$HOME" /ari 2>/dev/null || true
