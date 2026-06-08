#!/bin/bash

# Upload the final raw nanochat checkpoint bundle to Hugging Face Hub.
#
# Submit after the training job has been launched:
#   HF_REPO_ID=nurcunal/nanochat-turk-d20-bpe32k \
#   TRAIN_JOBID=492421 \
#   sbatch --dependency=afterok:492421 runs/uhem_upload_final_to_hf.sbatch
#
# Authenticate first on Altay with either:
#   hf auth login
# or:
#   huggingface-cli login

#SBATCH --job-name=nanochat-hf-upload
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_DIR"

if [ -z "${HF_REPO_ID:-}" ]; then
    echo "HF_REPO_ID is required, e.g. HF_REPO_ID=nurcunal/nanochat-turk-d20-bpe32k" >&2
    exit 2
fi

module purge
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0

echo "HOSTNAME=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "HF_REPO_ID=$HF_REPO_ID"
echo "TRAIN_JOBID=${TRAIN_JOBID:-}"
echo "CETVEL_JOBID=${CETVEL_JOBID:-}"
echo "PYTHON=$(command -v python3)"
python3 --version

export PATH="$HOME/.local/bin:$PATH"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/nanochat-turk-d20-bpe32k}"
export MODEL_TAG="${MODEL_TAG:-tr_d20_bpe_32768_chinchilla20}"
export NANOCHAT_TOKENIZER_NAME="${NANOCHAT_TOKENIZER_NAME:-bpe_32768}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_XET_CACHE="${HF_XET_CACHE:-$NANOCHAT_BASE_DIR/hf-xet-cache}"
mkdir -p "$HF_XET_CACHE"

source .venv/bin/activate

python - <<'PY'
import huggingface_hub
print("huggingface_hub", huggingface_hub.__version__)
PY

PRIVATE_ARGS=()
if [ "${HF_PRIVATE:-1}" = "1" ]; then
    PRIVATE_ARGS=(--private)
fi

OPTIMIZER_ARGS=()
if [ "${HF_INCLUDE_OPTIMIZER:-1}" != "1" ]; then
    OPTIMIZER_ARGS=(--no-optimizer)
fi

python scripts/upload_base_checkpoint_to_hf.py \
    --repo-id="$HF_REPO_ID" \
    --base-dir="$NANOCHAT_BASE_DIR" \
    --model-tag="$MODEL_TAG" \
    --tokenizer-name="$NANOCHAT_TOKENIZER_NAME" \
    --step="${HF_STEP:-latest}" \
    --job-id="${TRAIN_JOBID:-}" \
    --cetvel-job-id="${CETVEL_JOBID:-}" \
    "${PRIVATE_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}"
