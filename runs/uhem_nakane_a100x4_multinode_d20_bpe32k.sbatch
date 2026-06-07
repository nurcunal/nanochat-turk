#!/bin/bash

# Altay/UHeM multi-node base-model run for the nakane project.
# Default: 2 nodes x 4 A100 = 8 GPUs. Use 4 or 8 nodes for larger tests
# only after a short multi-node smoke run succeeds. Avoid 3, 5, 6, 7, 9, or
# 10 nodes for this 32k-vocab run because the distributed AdamW optimizer
# requires large first dimensions to divide evenly by world size.
#
# Submit from the repo root on Altay with:
#   sbatch runs/uhem_nakane_a100x4_multinode_d20_bpe32k.sbatch

#SBATCH --job-name=nanochat-tr-d20-bpe32k-mn
#SBATCH --account=nakane
#SBATCH --partition=a100x4q
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
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

mapfile -t SLURM_HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR="${MASTER_ADDR:-${SLURM_HOSTS[0]}}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 40000))}"
export RDZV_ID="${RDZV_ID:-$SLURM_JOB_ID}"

echo "HOSTNAME=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
printf 'HOSTS=%s\n' "${SLURM_HOSTS[*]}"
echo "PYTHON=$(command -v python3)"
python3 --version
echo "UV=$(command -v uv || true)"

export PATH="$HOME/.local/bin:$PATH"
export UV_PYTHON="${UV_PYTHON:-$(command -v python3)}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

export HARDWARE_PROFILE="${HARDWARE_PROFILE:-a100}"
export NNODES="${NNODES:-${SLURM_JOB_NUM_NODES:-2}}"
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

export EVAL_EVERY="${EVAL_EVERY:-1000}"
export EVAL_TOKENS="${EVAL_TOKENS:-4194304}"
export CORE_METRIC_EVERY="${CORE_METRIC_EVERY:--1}"
export SAMPLE_EVERY="${SAMPLE_EVERY:-1000}"
export SAVE_EVERY="${SAVE_EVERY:-2500}"
export SPLIT_TOKENS="${SPLIT_TOKENS:-4194304}"
export BASE_EVALS="${BASE_EVALS:-bpb,sample}"

export RUN_CETVEL="${RUN_CETVEL:-0}"
export WANDB_RUN="${WANDB_RUN:-tr-d20-bpe32k-multinode}"
export WANDB_PROJECT="${WANDB_PROJECT:-nanochat-turk}"
export WANDB_DIR="${WANDB_DIR:-$NANOCHAT_BASE_DIR/wandb}"
export WANDB_LOG_EVERY="${WANDB_LOG_EVERY:-100}"
export WANDB_CONSOLE="${WANDB_CONSOLE:-redirect}"
export WANDB_CONSOLE_MULTIPART="${WANDB_CONSOLE_MULTIPART:-true}"
export WANDB_CONSOLE_CHUNK_MAX_SECONDS="${WANDB_CONSOLE_CHUNK_MAX_SECONDS:-60}"
export SKIP_SETUP="${SKIP_SETUP:-1}"
export SKIP_DATASET="${SKIP_DATASET:-1}"
export SKIP_TOKENIZER="${SKIP_TOKENIZER:-1}"
mkdir -p "$WANDB_DIR"

DATA_DIR="$NANOCHAT_BASE_DIR/base_data_fineweb2_tur_latn"
TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizers/$NANOCHAT_TOKENIZER_NAME"
CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
if (( VOCAB_SIZE % WORLD_SIZE != 0 )); then
    echo "Unsupported world size $WORLD_SIZE for VOCAB_SIZE=$VOCAB_SIZE." >&2
    echo "Use a node count whose total GPU count divides the vocab size; for a100x4q, 2 or 4 nodes are the practical next tests." >&2
    exit 2
fi

if [ ! -d ".venv" ]; then
    echo "Missing .venv. Run runs/uhem_nakane_prepare_d20_bpe32k.sbatch before this GPU job." >&2
    exit 2
fi
if [ ! -f "$DATA_DIR/fineweb2_manifest.json" ]; then
    echo "Missing prepared dataset manifest at $DATA_DIR/fineweb2_manifest.json" >&2
    exit 2
fi
if [ ! -f "$TOKENIZER_DIR/tokenizer.pkl" ] || [ ! -f "$TOKENIZER_DIR/token_bytes.pt" ]; then
    echo "Missing prepared tokenizer under $TOKENIZER_DIR" >&2
    exit 2
fi

if [ -n "${RESUME_FROM_STEP:-}" ]; then
    step_tag=$(printf "%06d" "$RESUME_FROM_STEP")
    if [ ! -f "$CHECKPOINT_DIR/model_${step_tag}.pt" ] || [ ! -f "$CHECKPOINT_DIR/meta_${step_tag}.json" ]; then
        echo "Missing model/meta checkpoint for resume step $RESUME_FROM_STEP in $CHECKPOINT_DIR" >&2
        exit 2
    fi
    for ((rank = 0; rank < WORLD_SIZE; rank++)); do
        if [ ! -f "$CHECKPOINT_DIR/optim_${step_tag}_rank${rank}.pt" ]; then
            echo "Missing optimizer shard for rank $rank at resume step $RESUME_FROM_STEP." >&2
            echo "Do not resume a checkpoint created with a different world size without conversion." >&2
            exit 2
        fi
    done
fi

srun \
    --kill-on-bad-exit=1 \
    --nodes="$NNODES" \
    --ntasks="$NNODES" \
    --ntasks-per-node=1 \
    --cpus-per-task="$SLURM_CPUS_PER_TASK" \
    bash runs/turkish_foundation.sh
