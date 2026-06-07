#!/bin/bash

# Altay/UHeM production base-model run for the nakane project.
# 32k BPE, depth 20, Chinchilla-style 20 tokens per total parameter.
#
# Submit from the repo root on Altay with:
#   sbatch runs/uhem_nakane_a100x4_d20_bpe32k.sbatch

#SBATCH --job-name=nanochat-tr-d20-bpe32k
#SBATCH --account=nakane
#SBATCH --partition=a100x4q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a100:4
#SBATCH --time=10-00:00:00
#SBATCH --mem=512G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

module purge
module load cuda/cuda-12.5-a100q
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0

echo "HOSTNAME=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-}"
echo "PYTHON=$(command -v python3)"
python3 --version
echo "UV=$(command -v uv || true)"
nvidia-smi

export PATH="$HOME/.local/bin:$PATH"
export UV_PYTHON="${UV_PYTHON:-$(command -v python3)}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

export HARDWARE_PROFILE="${HARDWARE_PROFILE:-a100}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

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

export DEPTH="${DEPTH:-20}"
export MODEL_TAG="${MODEL_TAG:-tr_d20_bpe_32768_chinchilla20}"
export DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-4}"

# Keep repeated validation useful but not dominant. Final eval still runs.
export EVAL_EVERY="${EVAL_EVERY:-1000}"
export EVAL_TOKENS="${EVAL_TOKENS:-4194304}"
export CORE_METRIC_EVERY="${CORE_METRIC_EVERY:--1}"
export SAMPLE_EVERY="${SAMPLE_EVERY:-1000}"
export SAVE_EVERY="${SAVE_EVERY:-2500}"
export SPLIT_TOKENS="${SPLIT_TOKENS:-4194304}"
export BASE_EVALS="${BASE_EVALS:-bpb,sample}"

# Run CETVEL separately after the checkpoint exists; the smoke test did not
# validate CETVEL/lm-eval setup, and we should not let it jeopardize training.
export RUN_CETVEL="${RUN_CETVEL:-0}"
export WANDB_RUN="${WANDB_RUN:-dummy}"
export SKIP_SETUP="${SKIP_SETUP:-1}"
export SKIP_DATASET="${SKIP_DATASET:-1}"
export SKIP_TOKENIZER="${SKIP_TOKENIZER:-1}"

DATA_DIR="$NANOCHAT_BASE_DIR/base_data_fineweb2_tur_latn"
TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizers/$NANOCHAT_TOKENIZER_NAME"
if [ ! -d ".venv" ]; then
    echo "Missing .venv. Run runs/uhem_nakane_prepare_d20_bpe32k.sbatch before this GPU job." >&2
    exit 2
fi
if [ ! -f "$DATA_DIR/fineweb2_manifest.json" ]; then
    echo "Missing prepared dataset manifest at $DATA_DIR/fineweb2_manifest.json" >&2
    echo "Run runs/uhem_nakane_prepare_d20_bpe32k.sbatch before this GPU job." >&2
    exit 2
fi
if [ ! -f "$TOKENIZER_DIR/tokenizer.pkl" ] || [ ! -f "$TOKENIZER_DIR/token_bytes.pt" ]; then
    echo "Missing prepared tokenizer under $TOKENIZER_DIR" >&2
    echo "Run the updated runs/uhem_nakane_prepare_d20_bpe32k.sbatch before this GPU job." >&2
    exit 2
fi

bash runs/turkish_foundation.sh
