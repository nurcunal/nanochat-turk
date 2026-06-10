#!/bin/bash -l

# Publish the trained TRmorph MorphBPE 32k tokenizer bundle.
#
# Submit after the CPU tokenizer prep job:
#   HF_TOKENIZER_REPO_ID=nurcunal/nanochat-turk-tokenizers \
#   sbatch --dependency=afterok:<prep_job_id> \
#     runs/uhem_publish_morphbpe_trmorph_32k_tokenizer.sbatch
#
# The job uploads to Hugging Face and writes a local bundle under
# $NANOCHAT_BASE_DIR/published_tokenizers/$NANOCHAT_TOKENIZER_NAME. Pull that
# local bundle back into artifacts/tokenizers/$NANOCHAT_TOKENIZER_NAME for the
# GitHub copy once prep has completed.

#SBATCH --job-name=nanochat-tokenizer-hf
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem=32G
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
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0

DEFAULT_REPO_DIR="$HOME/nanochat-turk"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [ ! -d "$REPO_DIR/.venv" ] && [ -d "$DEFAULT_REPO_DIR/.venv" ]; then
    REPO_DIR="$DEFAULT_REPO_DIR"
fi
cd "$REPO_DIR"

export PATH="$HOME/.local/bin:$PATH"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/nanochat-turk-morphbpe-trmorph-32768}"
export NANOCHAT_TOKENIZER_NAME="${NANOCHAT_TOKENIZER_NAME:-morphbpe_trmorph_32768}"
export SEGMENTED_DATA_DIR="${SEGMENTED_DATA_DIR:-$NANOCHAT_BASE_DIR/segmented/trmorph-segmented}"
export TOKENIZER_METRICS_PATH="${TOKENIZER_METRICS_PATH:-$NANOCHAT_BASE_DIR/${NANOCHAT_TOKENIZER_NAME}_raw_metrics.json}"
export TOKENIZER_PUBLISH_DIR="${TOKENIZER_PUBLISH_DIR:-$NANOCHAT_BASE_DIR/published_tokenizers/$NANOCHAT_TOKENIZER_NAME}"
export HF_TOKENIZER_REPO_ID="${HF_TOKENIZER_REPO_ID:-nurcunal/nanochat-turk-tokenizers}"
export HF_TOKENIZER_REPO_PREFIX="${HF_TOKENIZER_REPO_PREFIX:-$NANOCHAT_TOKENIZER_NAME}"
export HF_HOME="${HF_HOME:-$NANOCHAT_BASE_DIR/huggingface}"
export HF_XET_CACHE="${HF_XET_CACHE:-$NANOCHAT_BASE_DIR/hf-xet-cache}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
mkdir -p "$HF_HOME" "$HF_XET_CACHE" "$TOKENIZER_PUBLISH_DIR"

echo "HOSTNAME=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "PYTHON=$(command -v python3)"
python3 --version
echo "HF_TOKENIZER_REPO_ID=$HF_TOKENIZER_REPO_ID"
echo "NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"
echo "NANOCHAT_TOKENIZER_NAME=$NANOCHAT_TOKENIZER_NAME"
echo "TOKENIZER_PUBLISH_DIR=$TOKENIZER_PUBLISH_DIR"

if [ ! -d ".venv" ]; then
    echo "Missing .venv in $REPO_DIR. Run tokenizer prep first." >&2
    exit 2
fi
source .venv/bin/activate

python - <<'PY'
from huggingface_hub import HfApi, whoami

print("HF user:", whoami().get("name", "unknown"))
print("huggingface_hub ok")
PY

PRIVATE_ARGS=()
if [ "${HF_PRIVATE:-1}" = "1" ]; then
    PRIVATE_ARGS=(--private)
fi

python scripts/publish_tokenizer_artifact.py \
    --repo-id "$HF_TOKENIZER_REPO_ID" \
    --base-dir "$NANOCHAT_BASE_DIR" \
    --tokenizer-name "$NANOCHAT_TOKENIZER_NAME" \
    --repo-prefix "$HF_TOKENIZER_REPO_PREFIX" \
    --local-output-dir "$TOKENIZER_PUBLISH_DIR" \
    --metrics-path "$TOKENIZER_METRICS_PATH" \
    --segmentation-manifest "$SEGMENTED_DATA_DIR/manifest.json" \
    --segmented-dataset-manifest "$SEGMENTED_DATA_DIR/fineweb2_manifest.json" \
    --require-metrics \
    --require-segmentation-manifest \
    --require-segmented-dataset-manifest \
    "${PRIVATE_ARGS[@]}"

echo "Published tokenizer bundle:"
find "$TOKENIZER_PUBLISH_DIR" -maxdepth 3 -type f -print
