#!/bin/bash

# One-A100, read-only backend probe. This never launches model training.

#SBATCH --job-name=tr-d32-attn-probe
#SBATCH --account=nakane
#SBATCH --partition=a100q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -eo pipefail

source /etc/profile.d/modules.sh
module use /ari/progs/modulefiles
module purge
module load cuda/cuda-12.5-a100q
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0
set -u

CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/ari/users/nunal/nanochat-turk-d32-general-v3}"
RECIPE="${RECIPE:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_wsd_v4.json}"
PREFLIGHT_RECEIPT="${PREFLIGHT_RECEIPT:-$NANOCHAT_BASE_DIR/control/d32_v4/preflight.json}"
ATTENTION_PROBE_RECEIPT="${ATTENTION_PROBE_RECEIPT:-$NANOCHAT_BASE_DIR/control/d32_v4/attention_probe.json}"

cd "$CODE_DIR"
test -x .venv/bin/python
test -f "$PREFLIGHT_RECEIPT"
mkdir -p "$(dirname "$ATTENTION_PROBE_RECEIPT")"

export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$NANOCHAT_BASE_DIR/huggingface}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export CUDA_VISIBLE_DEVICES=0

.venv/bin/python scripts/d32_attention_probe.py \
    --recipe="$RECIPE" \
    --preflight-receipt="$PREFLIGHT_RECEIPT" \
    --output="$ATTENTION_PROBE_RECEIPT" \
    --repeats=5
