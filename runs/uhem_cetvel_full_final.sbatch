#!/bin/bash -l

# Run the full CETVEL benchmark on the final d20/32k base checkpoint.
#
# Submit after the training job has been launched:
#   TRAIN_JOBID=492421 \
#   sbatch --dependency=afterok:492421 runs/uhem_cetvel_full_final.sbatch

#SBATCH --job-name=nanochat-cetvel-full
#SBATCH --account=nakane
#SBATCH --partition=a100q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:1
#SBATCH --time=3-00:00:00
#SBATCH --mem=128G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_DIR"

if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
elif [ -f /usr/share/lmod/lmod/init/bash ]; then
    source /usr/share/lmod/lmod/init/bash
fi

module purge
module load cuda/cuda-12.5-a100q
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0

echo "HOSTNAME=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "TRAIN_JOBID=${TRAIN_JOBID:-}"
echo "PYTHON=$(command -v python3)"
python3 --version
echo "UV=$(command -v uv || true)"
nvidia-smi

export PATH="$HOME/.local/bin:$PATH"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/nanochat-turk-d20-bpe32k}"
export MODEL_TAG="${MODEL_TAG:-tr_d20_bpe_32768_chinchilla20}"
export NANOCHAT_TOKENIZER_NAME="${NANOCHAT_TOKENIZER_NAME:-bpe_32768}"
export HF_HOME="${HF_HOME:-$NANOCHAT_BASE_DIR/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$NANOCHAT_BASE_DIR/cetvel_hf_datasets_cache_datasets219}"
export CETVEL_DIR="${CETVEL_DIR:-$NANOCHAT_BASE_DIR/cetvel}"
export PYTHONPATH="$REPO_DIR:$CETVEL_DIR/lm-evaluation-harness:${PYTHONPATH:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-nanochat-turk}"
export WANDB_RUN="${WANDB_RUN:-cetvel-full-${MODEL_TAG}}"
export WANDB_DIR="${WANDB_DIR:-$NANOCHAT_BASE_DIR/wandb}"
export CETVEL_GENERATION_PROGRESS="${CETVEL_GENERATION_PROGRESS:-1}"
export CETVEL_GENERATION_PROGRESS_EVERY="${CETVEL_GENERATION_PROGRESS_EVERY:-1}"
mkdir -p "$WANDB_DIR" "$HF_DATASETS_CACHE"

source .venv/bin/activate
python -m pip install -q "datasets==2.19.2"

CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"
CETVEL_MODEL_STEP="${CETVEL_MODEL_STEP:-17100}"
step_tag=$(printf "%06d" "$CETVEL_MODEL_STEP")
if [ ! -f "$CHECKPOINT_DIR/model_${step_tag}.pt" ] || [ ! -f "$CHECKPOINT_DIR/meta_${step_tag}.json" ]; then
    echo "Missing final checkpoint for CETVEL step $CETVEL_MODEL_STEP in $CHECKPOINT_DIR" >&2
    exit 2
fi

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

CETVEL_ARGS=(
    --suite="${CETVEL_SUITE:-full}"
    --model-tag="$MODEL_TAG"
    --model-step="$CETVEL_MODEL_STEP"
    --batch-size="${CETVEL_BATCH_SIZE:-1}"
    --device="${CETVEL_DEVICE:-cuda:0}"
    --max-gen-tokens="${CETVEL_MAX_GEN_TOKENS:-128}"
    --output-path="${CETVEL_OUTPUT_PATH:-$NANOCHAT_BASE_DIR/cetvel_out/${CETVEL_SUITE:-full}}"
)

if [ "${CETVEL_AUTO_SETUP:-1}" = "1" ]; then
    CETVEL_ARGS+=(--auto-setup)
fi
if [ "${CETVEL_WANDB:-1}" = "1" ]; then
    CETVEL_ARGS+=(--wandb)
fi
if [ -n "${CETVEL_LIMIT:-}" ]; then
    CETVEL_ARGS+=(--limit="$CETVEL_LIMIT")
fi
if [ -n "${CETVEL_TASKS:-}" ]; then
    CETVEL_ARGS+=(--tasks="$CETVEL_TASKS")
fi
if [ "${CETVEL_TASK_PROGRESS:-1}" = "1" ]; then
    CETVEL_ARGS+=(--task-progress)
fi

echo "CETVEL args: ${CETVEL_ARGS[*]}"
python scripts/cetvel_eval.py "${CETVEL_ARGS[@]}"
python -m nanochat.report generate
