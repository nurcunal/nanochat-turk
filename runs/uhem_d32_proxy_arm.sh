#!/bin/bash

# Run and seal one single-GPU proxy arm inside a packed 4xA100 allocation.

set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "usage: $0 MODEL_DEPTH CANDIDATE_ID SEED GPU_SLOT" >&2
    exit 2
fi
model_depth="$1"
candidate_id="$2"
seed="$3"
gpu_slot="$4"

: "${CODE_DIR:?}"
: "${NANOCHAT_BASE_DIR:?}"
: "${RECIPE:?}"
: "${PREFLIGHT_RECEIPT:?}"
: "${ATTENTION_PROBE_RECEIPT:?}"

cd "$CODE_DIR"
eval "$(.venv/bin/python scripts/d32_family_workflow.py proxy-env \
    --recipe="$RECIPE" \
    --model-depth="$model_depth" \
    --candidate-id="$candidate_id" \
    --seed="$seed")"
eval "$(.venv/bin/python scripts/d32_family_workflow.py attention-env \
    --recipe="$RECIPE" \
    --preflight-receipt="$PREFLIGHT_RECEIPT" \
    --attention-probe="$ATTENTION_PROBE_RECEIPT")"

export RUN_KIND=proxy
export FAMILY_ID=tr_d32_general_bpe32k_v2
export NNODES=1
export NPROC_PER_NODE=1
export CUDA_VISIBLE_DEVICES="$gpu_slot"
export STUDY_MANIFEST="$RECIPE"
export DATA_DIR="$NANOCHAT_BASE_DIR/pretrain_data/tr_general_clean_v2"
export TOKENIZER_MANIFEST="$NANOCHAT_BASE_DIR/tokenizers/tr_general_raw_bpe_32k_v2/package_manifest.json"
export VALIDATION_MANIFEST="$DATA_DIR/validation_exposure_manifest.json"
export EXPOSURE_PLAN="$DATA_DIR/training_exposure_${EXPOSURE_PLAN_KEY}.json"
export METRICS_DIR="$NANOCHAT_BASE_DIR/metrics/d32_proxy/$RUN_ID"
export CODE_REVISION ATTENTION_BACKEND WINDOW_PATTERN ATTENTION_PROBE_RECEIPT
export LAUNCH_PHASE=proxy_train
export LAUNCHER_ID=slurm_batch_direct_python_env_v1
export LAUNCH_RECEIPT_DIR="$NANOCHAT_BASE_DIR/control/d32/proxy_rank_exits/$RUN_ID"

mkdir -p "$METRICS_DIR" "$NANOCHAT_BASE_DIR/control/d32/proxy_runs"

bash runs/uhem_d32_train_node.sh

checkpoint_root="$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"
receipt="$NANOCHAT_BASE_DIR/control/d32/proxy_runs/${RUN_ID}.json"
.venv/bin/python scripts/d32_family_workflow.py seal-proxy-run \
    --recipe="$RECIPE" \
    --preflight-receipt="$PREFLIGHT_RECEIPT" \
    --attention-probe="$ATTENTION_PROBE_RECEIPT" \
    --model-depth="$model_depth" \
    --candidate-id="$candidate_id" \
    --seed="$seed" \
    --curve-log="$METRICS_DIR/training_curve.jsonl" \
    --checkpoint-root="$checkpoint_root" \
    --rank-exit-receipt="$LAUNCH_RECEIPT_DIR/rank_00000.json" \
    --slurm-job-id="${SLURM_JOB_ID:-}" \
    --output="$receipt"
