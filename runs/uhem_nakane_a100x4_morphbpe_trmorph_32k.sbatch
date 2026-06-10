#!/bin/bash -l

# Altay/UHeM TRmorph MorphBPE 32k base-model run.
#
# Default: 2 nodes x 4 A100 = 8 GPUs. If a100x4q is tight, submit with
# --nodes=1 or export NNODES=1 and edit/override the SBATCH node count. Use 4
# nodes only after the 2-node path has run cleanly.
#
# Submit from the repo root on Altay after CPU prep completes:
#   sbatch runs/uhem_nakane_a100x4_morphbpe_trmorph_32k.sbatch

#SBATCH --job-name=nanochat-tr-mbpe-trm32k
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

if [ -f /etc/profile.d/modules.sh ]; then
    # Batch shells on Altay do not always initialize Environment Modules.
    source /etc/profile.d/modules.sh
elif [ -f /usr/share/lmod/lmod/init/bash ]; then
    source /usr/share/lmod/lmod/init/bash
fi

module purge
module load cuda/cuda-12.5-a100q
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0

DEFAULT_REPO_DIR="$HOME/nanochat-turk"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [ ! -d "$REPO_DIR/.venv" ] && [ -d "$DEFAULT_REPO_DIR/.venv" ]; then
    REPO_DIR="$DEFAULT_REPO_DIR"
fi
cd "$REPO_DIR"

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

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/nanochat-turk-morphbpe-trmorph-32768}"
export NANOCHAT_DATA_DIR="${NANOCHAT_DATA_DIR:-$HOME/nanochat-turk-d20-bpe32k/base_data_fineweb2_tur_latn}"
export NANOCHAT_TEXT_COLUMN="${NANOCHAT_TEXT_COLUMN:-text}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$NANOCHAT_BASE_DIR/uv-cache}"
export HF_HOME="${HF_HOME:-$NANOCHAT_BASE_DIR/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$NANOCHAT_BASE_DIR/hf-datasets-cache}"

export VOCAB_SIZE="${VOCAB_SIZE:-32768}"
export NANOCHAT_TOKENIZER_NAME="${NANOCHAT_TOKENIZER_NAME:-morphbpe_trmorph_32768}"

export DEPTH="${DEPTH:-20}"
export MODEL_TAG="${MODEL_TAG:-tr_d20_morphbpe_trmorph_32768_chinchilla20}"
export DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-4}"
export TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-1048576}"
export NUM_ITERATIONS="${NUM_ITERATIONS:-17100}"

export EVAL_EVERY="${EVAL_EVERY:-1000}"
export EVAL_TOKENS="${EVAL_TOKENS:-4194304}"
export CORE_METRIC_EVERY="${CORE_METRIC_EVERY:--1}"
export SAMPLE_EVERY="${SAMPLE_EVERY:-1000}"
export SAVE_EVERY="${SAVE_EVERY:-2500}"
export SPLIT_TOKENS="${SPLIT_TOKENS:-4194304}"
export BASE_EVALS="${BASE_EVALS:-bpb,sample}"

export RUN_CETVEL="${RUN_CETVEL:-0}"
export WANDB_RUN="${WANDB_RUN:-tr-d20-morphbpe-trmorph-32k}"
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

export TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizers/$NANOCHAT_TOKENIZER_NAME"
CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
if (( VOCAB_SIZE % WORLD_SIZE != 0 )); then
    echo "Unsupported world size $WORLD_SIZE for VOCAB_SIZE=$VOCAB_SIZE." >&2
    echo "Use 1, 2, or 4 a100x4q nodes for 32k vocab runs." >&2
    exit 2
fi
if [ "$WANDB_RUN" = "dummy" ]; then
    echo "WANDB_RUN resolved to dummy; refusing production run." >&2
    exit 2
fi
if [ ! -d ".venv" ]; then
    echo "Missing .venv. Run the MorphBPE CPU prep job before this GPU job." >&2
    exit 2
fi
source .venv/bin/activate
if [ ! -f "$NANOCHAT_DATA_DIR/fineweb2_manifest.json" ]; then
    echo "Missing raw dataset manifest at $NANOCHAT_DATA_DIR/fineweb2_manifest.json" >&2
    exit 2
fi
if [ ! -f "$TOKENIZER_DIR/tokenizer.pkl" ] || [ ! -f "$TOKENIZER_DIR/token_bytes.pt" ] || [ ! -f "$TOKENIZER_DIR/tokenizer_config.json" ]; then
    echo "Missing prepared MorphBPE tokenizer under $TOKENIZER_DIR" >&2
    echo "Run runs/uhem_nakane_prepare_morphbpe_trmorph_32k.sbatch before this GPU job." >&2
    exit 2
fi
python - <<'PY'
import json
import os
from pathlib import Path

cfg = json.loads((Path(os.environ["TOKENIZER_DIR"]) / "tokenizer_config.json").read_text())
assert cfg["implementation"] == "morphbpe", cfg
assert cfg["training_boundary_semantics"] == "merge_constraint_only", cfg
assert cfg["requires_runtime_segmentation"] is False, cfg
assert cfg["decode_strip"] == "", cfg
assert cfg["vocab_size"] == int(os.environ["VOCAB_SIZE"]), cfg
print("MorphBPE tokenizer config preflight OK")
PY

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
