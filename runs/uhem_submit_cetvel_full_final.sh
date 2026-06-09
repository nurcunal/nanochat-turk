#!/bin/bash

# Known-good UHeM login-node launcher for the final d20/BPE-32k CETVEL full
# benchmark and its dependent Hugging Face upload.
#
# Usage on Altay:
#   cd /ari/users/nunal/nanochat-turk
#   HF_REPO_ID=nurcunal/nanochat-turk-d20-bpe32k \
#   TRAIN_JOBID=492421 \
#   bash runs/uhem_submit_cetvel_full_final.sh
#
# Useful overrides:
#   CETVEL_MODEL_STEP=17100
#   HF_INCLUDE_OPTIMIZER=1
#   CETVEL_BATCH_SIZE=1
#   CETVEL_MAX_GEN_TOKENS=128
#   CETVEL_WANDB=1
#   CETVEL_TASK_PROGRESS=1
#   CETVEL_GENERATION_PROGRESS=1
#   CETVEL_GENERATION_PROGRESS_EVERY=1
#   CETVEL_OUTPUT_PATH=$HOME/nanochat-turk-d20-bpe32k/cetvel_out/full_kvcache
#   WANDB_RUN=cetvel-full-kvcache

set -euo pipefail

REPO_DIR="${REPO_DIR:-$PWD}"
cd "$REPO_DIR"

HF_REPO_ID="${HF_REPO_ID:-nurcunal/nanochat-turk-d20-bpe32k}"
TRAIN_JOBID="${TRAIN_JOBID:-492421}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/nanochat-turk-d20-bpe32k}"
MODEL_TAG="${MODEL_TAG:-tr_d20_bpe_32768_chinchilla20}"
CETVEL_MODEL_STEP="${CETVEL_MODEL_STEP:-17100}"
CETVEL_DIR="${CETVEL_DIR:-$NANOCHAT_BASE_DIR/cetvel}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
HF_STEP="${HF_STEP:-$CETVEL_MODEL_STEP}"

step_tag=$(printf "%06d" "$CETVEL_MODEL_STEP")
checkpoint_dir="$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"

if [ ! -f "$checkpoint_dir/model_${step_tag}.pt" ] || [ ! -f "$checkpoint_dir/meta_${step_tag}.json" ]; then
    echo "Missing checkpoint files for step $CETVEL_MODEL_STEP in $checkpoint_dir" >&2
    exit 2
fi

if [ ! -d "$CETVEL_DIR/lm-evaluation-harness" ]; then
    echo "Missing CETVEL lm-evaluation-harness at $CETVEL_DIR/lm-evaluation-harness" >&2
    echo "Run once with CETVEL_AUTO_SETUP=1 in runs/uhem_cetvel_full_final.sbatch, or clone CETVEL before using this launcher." >&2
    exit 2
fi

module purge
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0
source .venv/bin/activate

export PATH="$HOME/.local/bin:$PATH"
export NANOCHAT_BASE_DIR MODEL_TAG CETVEL_DIR HF_HOME
export PYTHONPATH="$PWD:$CETVEL_DIR/lm-evaluation-harness:${PYTHONPATH:-}"

python -m py_compile scripts/cetvel_eval.py
python -m pip install -q "datasets==2.19.2" toml
python -m pip install -q -e "$CETVEL_DIR/lm-evaluation-harness"
if [ -f "$CETVEL_DIR/requirements.txt" ]; then
    python -m pip install -q -r "$CETVEL_DIR/requirements.txt"
fi

PYTHONPATH="$PYTHONPATH" python scripts/cetvel_eval.py --patch-configs-only

python - <<PY
from huggingface_hub import HfApi, whoami

repo_id = "$HF_REPO_ID"
print("HF user:", whoami()["name"])
HfApi().create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
print("HF repo ok:", f"https://huggingface.co/{repo_id}")
PY

stamp=$(date +%Y%m%d_%H%M%S)

cetvel_submit=$(
    TRAIN_JOBID="$TRAIN_JOBID" \
    NANOCHAT_BASE_DIR="$NANOCHAT_BASE_DIR" \
    MODEL_TAG="$MODEL_TAG" \
    CETVEL_MODEL_STEP="$CETVEL_MODEL_STEP" \
    CETVEL_SUITE="${CETVEL_SUITE:-full}" \
    CETVEL_BATCH_SIZE="${CETVEL_BATCH_SIZE:-1}" \
    CETVEL_MAX_GEN_TOKENS="${CETVEL_MAX_GEN_TOKENS:-128}" \
    CETVEL_OUTPUT_PATH="${CETVEL_OUTPUT_PATH:-}" \
    CETVEL_WANDB="${CETVEL_WANDB:-1}" \
    CETVEL_TASK_PROGRESS="${CETVEL_TASK_PROGRESS:-1}" \
    CETVEL_GENERATION_PROGRESS="${CETVEL_GENERATION_PROGRESS:-1}" \
    CETVEL_GENERATION_PROGRESS_EVERY="${CETVEL_GENERATION_PROGRESS_EVERY:-1}" \
    CETVEL_AUTO_SETUP="${CETVEL_AUTO_SETUP:-0}" \
    WANDB_RUN="${WANDB_RUN:-cetvel-full-${MODEL_TAG}}" \
    HF_HOME="$HF_HOME" \
    sbatch --parsable runs/uhem_cetvel_full_final.sbatch \
    2>"cetvel-submit-${stamp}.err"
)
printf "%s\n" "$cetvel_submit" | tee "cetvel-submit-${stamp}.out"
cetvel_jobid=$(printf "%s\n" "$cetvel_submit" | awk '/^[0-9]+/ {split($1,a,";"); print a[1]; exit}')

if [ -z "$cetvel_jobid" ]; then
    echo "Could not parse CETVEL job id from sbatch output:" >&2
    printf "%s\n" "$cetvel_submit" >&2
    exit 2
fi
echo "CETVEL_JOBID=$cetvel_jobid"

hf_cetvel_submit=$(
    HF_HOME="$HF_HOME" \
    HF_REPO_ID="$HF_REPO_ID" \
    TRAIN_JOBID="$TRAIN_JOBID" \
    CETVEL_JOBID="$cetvel_jobid" \
    NANOCHAT_BASE_DIR="$NANOCHAT_BASE_DIR" \
    MODEL_TAG="$MODEL_TAG" \
    HF_STEP="$HF_STEP" \
    HF_PRIVATE="${HF_PRIVATE:-1}" \
    HF_INCLUDE_OPTIMIZER="${HF_INCLUDE_OPTIMIZER:-1}" \
    sbatch --dependency=afterok:${cetvel_jobid} --parsable runs/uhem_upload_final_to_hf.sbatch \
    2>"hf-cetvel-submit-${stamp}.err"
)
printf "%s\n" "$hf_cetvel_submit" | tee "hf-cetvel-submit-${stamp}.out"
hf_cetvel_jobid=$(printf "%s\n" "$hf_cetvel_submit" | awk '/^[0-9]+/ {split($1,a,";"); print a[1]; exit}')

if [ -z "$hf_cetvel_jobid" ]; then
    echo "Could not parse HF upload job id from sbatch output:" >&2
    printf "%s\n" "$hf_cetvel_submit" >&2
    exit 2
fi
echo "HF_CETVEL_UPLOAD_JOBID=$hf_cetvel_jobid"

squeue -j "$cetvel_jobid,$hf_cetvel_jobid"

cat <<EOF

Monitor:
  squeue -j $cetvel_jobid,$hf_cetvel_jobid
  tail -f nanochat-cetvel-full-${cetvel_jobid}.err
  node=\$(squeue -h -j $cetvel_jobid -o "%N"); ssh "\$node" nvidia-smi

Cancel dependent upload if CETVEL fails:
  scancel $hf_cetvel_jobid
EOF
