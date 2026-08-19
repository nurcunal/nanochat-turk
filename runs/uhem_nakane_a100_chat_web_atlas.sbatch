#!/bin/bash -l

# Serve the Türkçe Atlas SFT checkpoint with NanoChat's web interface.
# Access it through an SSH tunnel; do not expose the compute-node port publicly.

#SBATCH --job-name=nanochat-atlas-web
#SBATCH --account=nakane
#SBATCH --partition=a100q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
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
export NANOCHAT_DTYPE="${NANOCHAT_DTYPE:-bfloat16}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/nanochat-turk-d20-bpe32k}"
export NANOCHAT_TOKENIZER_NAME="${NANOCHAT_TOKENIZER_NAME:-bpe_32768}"
export TOKENIZERS_PARALLELISM=false

MODEL_TAG="${MODEL_TAG:-tr_d20_bpe_32768_chinchilla20}"
MODEL_STEP="${MODEL_STEP:-49}"
PORT="${PORT:-8000}"

if [ ! -x .venv/bin/python ]; then
    echo "Missing repository .venv at $REPO_DIR/.venv" >&2
    exit 2
fi

echo "HOSTNAME=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "GIT_REVISION=$(git rev-parse HEAD)"
echo "MODEL_TAG=$MODEL_TAG"
echo "MODEL_STEP=$MODEL_STEP"
echo "PORT=$PORT"

exec .venv/bin/python -m scripts.chat_web \
    --num-gpus 1 \
    --source sft \
    --model-tag "$MODEL_TAG" \
    --step "$MODEL_STEP" \
    --host 0.0.0.0 \
    --port "$PORT"
