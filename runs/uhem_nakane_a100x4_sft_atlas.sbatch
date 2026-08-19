#!/bin/bash -l

# Guarded Türkçe Atlas SFT for tr_d20_bpe_32768_chinchilla20.
#
# Smoke (two optimizer updates, no checkpoint):
#   sbatch --export=ALL,MODE=smoke runs/uhem_nakane_a100x4_sft_atlas.sbatch
# Faster one-GPU smoke when a100x4q is congested:
#   sbatch --partition=a100q --gres=gpu:a100:1 \
#     --export=ALL,MODE=smoke,WORLD_SIZE=1 runs/uhem_nakane_a100x4_sft_atlas.sbatch
# Full (one pass through the training split, final checkpoint):
#   sbatch --export=ALL,MODE=full runs/uhem_nakane_a100x4_sft_atlas.sbatch

#SBATCH --job-name=nanochat-atlas-sft
#SBATCH --account=nakane
#SBATCH --partition=a100x4q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:a100:4
#SBATCH --time=04:00:00
#SBATCH --mem=512G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_DIR"

if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
elif [ -f /usr/share/lmod/lmod/init/bash ]; then
    source /usr/share/lmod/lmod/init/bash
elif [ -f /usr/share/lmod/8.2.7/init/bash ]; then
    source /usr/share/lmod/8.2.7/init/bash
fi
if ! command -v module >/dev/null 2>&1; then
    echo "Could not initialize environment modules on $(hostname)" >&2
    exit 127
fi

module purge
module load cuda/cuda-12.5-a100q
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0

export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_DTYPE="${NANOCHAT_DTYPE:-bfloat16}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/nanochat-turk-d20-bpe32k}"
export NANOCHAT_TOKENIZER_NAME="${NANOCHAT_TOKENIZER_NAME:-bpe_32768}"
export WANDB_DIR="${WANDB_DIR:-$NANOCHAT_BASE_DIR/wandb}"
export TOKENIZERS_PARALLELISM=false

MODE="${MODE:-smoke}"
MODEL_TAG="${MODEL_TAG:-tr_d20_bpe_32768_chinchilla20}"
MODEL_STEP="${MODEL_STEP:-17100}"
WORLD_SIZE="${WORLD_SIZE:-4}"
SFT_DATA_DIR="${SFT_DATA_DIR:-$NANOCHAT_BASE_DIR/sft_data/turkce_atlas_sft}"
TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizers/$NANOCHAT_TOKENIZER_NAME"
BASE_CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"

case "$MODE" in
    smoke)
        NUM_ITERATIONS=2
        SAVE_CHECKPOINT=0
        EVAL_TOKENS=32768
        RUN_NAME=dummy
        ;;
    full)
        NUM_ITERATIONS=-1
        SAVE_CHECKPOINT=1
        EVAL_TOKENS=851968
        RUN_NAME="${SFT_WANDB_RUN:-tr-d20-bpe32k-atlas-sft-v1}"
        ;;
    *)
        echo "MODE must be smoke or full, got: $MODE" >&2
        exit 2
        ;;
esac

if [ ! -x .venv/bin/python ]; then
    echo "Missing repository .venv at $REPO_DIR/.venv" >&2
    exit 2
fi
if [ "$MODE" = full ] && [ "$WORLD_SIZE" -ne 4 ]; then
    echo "Full SFT is locked to WORLD_SIZE=4" >&2
    exit 2
fi
if [ "$MODE" = smoke ] && [ "$WORLD_SIZE" -ne 1 ] && [ "$WORLD_SIZE" -ne 4 ]; then
    echo "Smoke SFT supports WORLD_SIZE=1 or WORLD_SIZE=4" >&2
    exit 2
fi
if [ "$MODE" = full ] && [ "${ALLOW_SFT_OVERWRITE:-0}" != 1 ]; then
    shopt -s nullglob
    existing_sft_models=("$NANOCHAT_BASE_DIR/chatsft_checkpoints/$MODEL_TAG"/model_*.pt)
    shopt -u nullglob
    if [ "${#existing_sft_models[@]}" -gt 0 ]; then
        echo "Refusing to overwrite an existing SFT model: ${existing_sft_models[*]}" >&2
        echo "Set ALLOW_SFT_OVERWRITE=1 only after reviewing the existing checkpoint." >&2
        exit 2
    fi
fi

mkdir -p "$WANDB_DIR"

echo "HOSTNAME=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "MODE=$MODE"
echo "REPO_DIR=$REPO_DIR"
echo "GIT_REVISION=$(git rev-parse HEAD)"
echo "NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"
echo "SFT_DATA_DIR=$SFT_DATA_DIR"
echo "MODEL_TAG=$MODEL_TAG"
echo "MODEL_STEP=$MODEL_STEP"
echo "NUM_ITERATIONS=$NUM_ITERATIONS"
echo "RUN_NAME=$RUN_NAME"
.venv/bin/python --version
nvidia-smi

.venv/bin/python -m scripts.preflight_turkce_atlas_sft \
    --data-dir "$SFT_DATA_DIR" \
    --tokenizer-dir "$TOKENIZER_DIR" \
    --base-checkpoint-dir "$BASE_CHECKPOINT_DIR" \
    --model-step "$MODEL_STEP" \
    --world-size 4 \
    --expected-model-tag "$MODEL_TAG" \
    --expected-total-batch-size 1048576 \
    --disk-path "$NANOCHAT_BASE_DIR" \
    --min-free-gb 10

SFT_ARGS=(
    --model-tag="$MODEL_TAG"
    --model-step="$MODEL_STEP"
    --load-optimizer=1
    --save-checkpoint="$SAVE_CHECKPOINT"
    --num-iterations="$NUM_ITERATIONS"
    --train-jsonl="$SFT_DATA_DIR/train.jsonl"
    --val-jsonl="$SFT_DATA_DIR/validation.jsonl"
    --data-manifest="$SFT_DATA_DIR/manifest.json"
    --custom-json-lazy=1
    --max-seq-len=2048
    --device-batch-size=4
    --total-batch-size=1048576
    --embedding-lr=0.3
    --unembedding-lr=0.008
    --matrix-lr=0.02
    --eval-every=-1
    --eval-tokens="$EVAL_TOKENS"
    --chatcore-every=-1
    --run="$RUN_NAME"
)

echo "SFT args: ${SFT_ARGS[*]}"
torchrun --standalone --nproc_per_node="$WORLD_SIZE" -m scripts.chat_sft -- "${SFT_ARGS[@]}"

echo "SFT $MODE completed successfully at $(date --iso-8601=seconds)"
