#!/bin/bash

# Production-identical fixed-global-batch smoke. Override --nodes=4 for ws16.
# This job creates raw checkpoints and launcher receipts only. A dependent CPU
# finalizer checks Slurm COMPLETED before it is allowed to seal the smoke.

#SBATCH --job-name=tr-d32-smoke
#SBATCH --account=nakane
#SBATCH --partition=a100x4q
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:4
#SBATCH --time=03:00:00
#SBATCH --mem=900G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -eo pipefail

source /etc/profile.d/modules.sh
module use /ari/progs/modulefiles
module purge
module load cuda/cuda-12.5-a100q
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0
set -u

export CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/ari/users/nunal/nanochat-turk-d32-general-v3}"
export RECIPE="${RECIPE:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_wsd_v3.json}"
export PREFLIGHT_RECEIPT="${PREFLIGHT_RECEIPT:-$NANOCHAT_BASE_DIR/control/d32/preflight.json}"
export ATTENTION_PROBE_RECEIPT="${ATTENTION_PROBE_RECEIPT:-$NANOCHAT_BASE_DIR/control/d32/attention_probe.json}"
export WSD_PROXY_APPROVAL="${WSD_PROXY_APPROVAL:-$NANOCHAT_BASE_DIR/control/d32/wd_proxy_approval.json}"
export STATIC_LAUNCHER_GATE="${STATIC_LAUNCHER_GATE:-$NANOCHAT_BASE_DIR/control/d32/static_launcher_gate_ws4.json}"
export SIGNAL_RESUME_GATE="${SIGNAL_RESUME_GATE:-$NANOCHAT_BASE_DIR/control/d32/signal_resume_gate_ws4.json}"

cd "$CODE_DIR"
NNODES="${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-0}}"
if [ "$NNODES" -ne 2 ] && [ "$NNODES" -ne 4 ]; then
    echo "Smoke allocation must contain exactly two or four 4xA100 nodes" >&2
    exit 2
fi
if [ "${SLURM_JOB_PARTITION:-}" != a100x4q ]; then
    echo "Smoke allocation must use a100x4q" >&2
    exit 2
fi
.venv/bin/python scripts/d32_family_workflow.py verify-static-launcher-gate \
    --recipe="$RECIPE" --preflight-receipt="$PREFLIGHT_RECEIPT" \
    --gate="$STATIC_LAUNCHER_GATE"
.venv/bin/python scripts/d32_family_workflow.py verify-signal-resume-gate \
    --recipe="$RECIPE" --preflight-receipt="$PREFLIGHT_RECEIPT" \
    --gate="$SIGNAL_RESUME_GATE"
export NNODES NPROC_PER_NODE=4
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
if [ "${SLURM_NTASKS:-0}" -ne "$WORLD_SIZE" ]; then
    echo "Smoke allocation must start four static tasks per node" >&2
    exit 2
fi
RECIPE_FAMILY_ID="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["family_id"])' "$RECIPE")"
if [ -n "${FAMILY_ID:-}" ] && [ "$FAMILY_ID" != "$RECIPE_FAMILY_ID" ]; then
    echo "FAMILY_ID differs from the validated recipe family_id: $RECIPE_FAMILY_ID" >&2
    exit 2
fi
export FAMILY_ID="$RECIPE_FAMILY_ID"
export RUN_KIND=smoke
export RUN_ID="${FAMILY_ID}_smoke_ws${WORLD_SIZE}"
export MODEL_TAG="$RUN_ID"
export DEPTH=32
export DEVICE_BATCH_SIZE=4
export TOTAL_BATCH_SIZE=2097152
export NUM_ITERATIONS=100
export STOP_AT_STEP=50
export SEED=42
export EVAL_EVERY=250
export LR_SCHEDULE=wsd
export WSD_COOLDOWN_START_STEP=-1
export STUDY_MANIFEST="$RECIPE"
export DATA_DIR="${DATA_DIR:-$NANOCHAT_BASE_DIR/pretrain_data/tr_general_clean_v3}"
export PACKING_CAPACITY_RECEIPT="$DATA_DIR/packing_capacity_receipt.json"
export TOKENIZER_MANIFEST="${TOKENIZER_MANIFEST:-$NANOCHAT_BASE_DIR/tokenizers/tr_general_raw_bpe_32k_v3/package_manifest.json}"
export VALIDATION_MANIFEST="$DATA_DIR/validation_exposure_manifest.json"
export EXPOSURE_PLAN="$DATA_DIR/training_exposure_smoke_ws${WORLD_SIZE}.json"
export METRICS_DIR="$NANOCHAT_BASE_DIR/metrics/d32_smoke/ws${WORLD_SIZE}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 40000))}"
mapfile -t SLURM_HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR="${MASTER_ADDR:-${SLURM_HOSTS[0]}}"
export RESUME_FROM_STEP=""
eval "$(.venv/bin/python scripts/d32_family_workflow.py optimizer-env \
    --recipe="$RECIPE" \
    --preflight-receipt="$PREFLIGHT_RECEIPT" \
    --attention-probe="$ATTENTION_PROBE_RECEIPT" \
    --wd-proxy-approval="$WSD_PROXY_APPROVAL")"
export EFFECTIVE_BASE_WEIGHT_DECAY WSD_WEIGHT_DECAY_COOLDOWN CODE_REVISION
export ATTENTION_BACKEND WINDOW_PATTERN ATTENTION_PROBE_RECEIPT
CONTROL_ROOT="$NANOCHAT_BASE_DIR/control/d32/launcher/job${SLURM_JOB_ID}"
if [ -e "$CONTROL_ROOT" ]; then
    echo "Refusing to reuse launcher receipt directory: $CONTROL_ROOT" >&2
    exit 2
fi
mkdir -p "$METRICS_DIR" "$CONTROL_ROOT"

STATIC_PROBE_DIR="$CONTROL_ROOT/static_nccl_probe"
STATIC_PROBE_RECEIPT="$CONTROL_ROOT/static_nccl_probe.json"
mkdir -p "$STATIC_PROBE_DIR"
srun --nodes="$NNODES" --ntasks="$WORLD_SIZE" --ntasks-per-node=4 \
    --cpus-per-task=16 --gpus-per-task=1 --gpu-bind=single:1 --kill-on-bad-exit=1 \
    bash runs/uhem_d32_srun_env.sh .venv/bin/python -m scripts.d32_static_launch_probe \
    --recipe="$RECIPE" --receipt-dir="$STATIC_PROBE_DIR" \
    --run-id="$RUN_ID" --phase=static_nccl_probe
.venv/bin/python scripts/d32_family_workflow.py seal-static-probe \
    --recipe="$RECIPE" --receipt-dir="$STATIC_PROBE_DIR" \
    --run-id="$RUN_ID" --phase=static_nccl_probe \
    --slurm-job-id="$SLURM_JOB_ID" --world-size="$WORLD_SIZE" --nodes="$NNODES" \
    --srun-exit-code=0 --output="$STATIC_PROBE_RECEIPT"

run_training_phase() {
    local phase="$1"
    export LAUNCH_PHASE="$phase"
    export LAUNCHER_ID=slurm_srun_direct_python_env_v1
    export LAUNCH_RECEIPT_DIR="$CONTROL_ROOT/$phase/ranks"
    local launch_receipt="$CONTROL_ROOT/$phase/launch.json"
    mkdir -p "$LAUNCH_RECEIPT_DIR"
    srun --nodes="$NNODES" --ntasks="$WORLD_SIZE" --ntasks-per-node=4 \
        --cpus-per-task=16 --gpus-per-task=1 --gpu-bind=single:1 --kill-on-bad-exit=1 \
        bash runs/uhem_d32_train_node.sh
    .venv/bin/python scripts/d32_family_workflow.py seal-static-launch \
        --recipe="$RECIPE" --receipt-dir="$LAUNCH_RECEIPT_DIR" \
        --run-id="$RUN_ID" --phase="$phase" --slurm-job-id="$SLURM_JOB_ID" \
        --world-size="$WORLD_SIZE" --nodes="$NNODES" --srun-exit-code=0 \
        --expected-exit-code=0 --output="$launch_receipt"
}

run_training_phase smoke_initial_50
export RESUME_FROM_STEP=50
export STOP_AT_STEP=100
run_training_phase smoke_resume_100

echo "Smoke payload completed. A dependent finalizer must verify Slurm state and seal it."
