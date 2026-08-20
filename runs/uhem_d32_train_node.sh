#!/bin/bash

# One direct Python rank for an srun-native launch. Despite the historical file
# name, this script is invoked once per GPU and never starts torchrun/mpirun.

set -euo pipefail

source /etc/profile.d/modules.sh
module use /ari/progs/modulefiles
module purge
module load cuda/cuda-12.5-a100q
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0

required_variables=(
    CODE_DIR NANOCHAT_BASE_DIR RUN_KIND FAMILY_ID RUN_ID MODEL_TAG DEPTH
    DEVICE_BATCH_SIZE TOTAL_BATCH_SIZE NUM_ITERATIONS STOP_AT_STEP SEED
    EVAL_EVERY EXPOSURE_PLAN METRICS_DIR STUDY_MANIFEST TOKENIZER_MANIFEST
    VALIDATION_MANIFEST DATA_DIR PREFLIGHT_RECEIPT CODE_REVISION LR_SCHEDULE ATTENTION_BACKEND
    WINDOW_PATTERN ATTENTION_PROBE_RECEIPT LAUNCH_RECEIPT_DIR LAUNCH_PHASE
    LAUNCHER_ID
)
for variable_name in "${required_variables[@]}"; do
    if [ -z "${!variable_name:-}" ]; then
        echo "Missing required environment variable: $variable_name" >&2
        exit 2
    fi
done

cd "$CODE_DIR"
if [ ! -x .venv/bin/python ]; then
    echo "The frozen .venv Python is missing" >&2
    exit 2
fi
if [ ! -f scripts/d32_wsd_train.py ]; then
    echo "Dedicated reviewed trainer scripts/d32_wsd_train.py is missing" >&2
    exit 2
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$NANOCHAT_BASE_DIR/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$NANOCHAT_BASE_DIR/hf-datasets-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$NANOCHAT_BASE_DIR/uv-cache}"

export RANK="${SLURM_PROCID:-${RANK:-0}}"
NODE_LOCAL_RANK="${SLURM_LOCALID:-0}"
# --gpus-per-task=1 remaps each task's assigned GPU to CUDA device zero.
export LOCAL_RANK=0
export WORLD_SIZE="${SLURM_NTASKS:-${WORLD_SIZE:-1}}"
export LOCAL_WORLD_SIZE="${LOCAL_WORLD_SIZE:-${NPROC_PER_NODE:-1}}"
if ! [[ "$RANK" =~ ^[0-9]+$ && "$LOCAL_RANK" =~ ^[0-9]+$ && "$WORLD_SIZE" =~ ^[0-9]+$ ]]; then
    echo "Invalid Slurm-derived rank environment" >&2
    exit 2
fi
if [ "$WORLD_SIZE" -gt 1 ]; then
    : "${MASTER_ADDR:?MASTER_ADDR is required for multi-rank training}"
    : "${MASTER_PORT:?MASTER_PORT is required for multi-rank training}"
fi

train_args=(
    --strict-mode
    "--strict-run-kind=$RUN_KIND"
    --no-fp8
    "--attention-backend=$ATTENTION_BACKEND"
    "--attention-probe=$ATTENTION_PROBE_RECEIPT"
    "--depth=$DEPTH"
    --aspect-ratio=64
    --head-dim=128
    --max-seq-len=2048
    "--window-pattern=$WINDOW_PATTERN"
    "--num-iterations=$NUM_ITERATIONS"
    --target-flops=-1
    --target-param-data-ratio=-1
    --target-param-count=scaling
    --horizon-unit=token_positions
    "--horizon-value=$((NUM_ITERATIONS * TOTAL_BATCH_SIZE))"
    "--device-batch-size=$DEVICE_BATCH_SIZE"
    "--total-batch-size=$TOTAL_BATCH_SIZE"
    --embedding-lr=0.3
    --unembedding-lr=0.008
    --matrix-lr=0.02
    --scalar-lr=0.5
    --warmup-steps=40
    --optimizer=muon_adamw
    "--lr-schedule=$LR_SCHEDULE"
    --grad-clip=0
    "--eval-every=$EVAL_EVERY"
    --eval-tokens=-1
    --core-metric-every=-1
    --core-metric-max-per-task=500
    --sample-every=-1
    --save-every=-1
    "--stop-at-step=$STOP_AT_STEP"
    "--model-tag=$MODEL_TAG"
    --run=dummy
    "--study-id=$FAMILY_ID"
    "--run-id=$RUN_ID"
    "--study-manifest=$STUDY_MANIFEST"
    "--preflight-receipt=$PREFLIGHT_RECEIPT"
    "--metrics-dir=$METRICS_DIR"
    "--tokenizer-manifest=$TOKENIZER_MANIFEST"
    "--exposure-plan=$EXPOSURE_PLAN"
    "--validation-manifest=$VALIDATION_MANIFEST"
    "--data-dir=$DATA_DIR"
    "--code-revision=$CODE_REVISION"
    "--seed=$SEED"
    --data-order=bestfit
)

if [ "$RUN_KIND" = production ] || [ "$RUN_KIND" = smoke ]; then
    : "${PACKING_CAPACITY_RECEIPT:?Production and distributed smoke require PACKING_CAPACITY_RECEIPT}"
    expected_capacity_receipt="$DATA_DIR/packing_capacity_receipt.json"
    if [ "$PACKING_CAPACITY_RECEIPT" != "$expected_capacity_receipt" ]; then
        echo "PACKING_CAPACITY_RECEIPT must be exactly $expected_capacity_receipt" >&2
        exit 2
    fi
    train_args+=("--packing-capacity-receipt=$PACKING_CAPACITY_RECEIPT")
elif [ -n "${PACKING_CAPACITY_RECEIPT:-}" ]; then
    echo "PACKING_CAPACITY_RECEIPT is forbidden for $RUN_KIND" >&2
    exit 2
fi

if [ "$LR_SCHEDULE" = wsd ]; then
    : "${EFFECTIVE_BASE_WEIGHT_DECAY:?Missing EFFECTIVE_BASE_WEIGHT_DECAY}"
    : "${WSD_WEIGHT_DECAY_COOLDOWN:?Missing WSD_WEIGHT_DECAY_COOLDOWN}"
    : "${WSD_COOLDOWN_START_STEP:?Missing WSD_COOLDOWN_START_STEP}"
    train_args+=(
        --wsd-recipe-version=tr_d32_wsd_wd_proxy_v1
        "--wsd-base-weight-decay=$EFFECTIVE_BASE_WEIGHT_DECAY"
        "--wsd-weight-decay-cooldown=$WSD_WEIGHT_DECAY_COOLDOWN"
        "--wsd-cooldown-start-step=$WSD_COOLDOWN_START_STEP"
        --wsd-cooldown-fraction=0.10
    )
    if [ -n "${WSD_PROXY_APPROVAL:-}" ]; then
        train_args+=("--wsd-proxy-approval=$WSD_PROXY_APPROVAL")
    fi
else
    : "${INPUT_WEIGHT_DECAY:?Missing INPUT_WEIGHT_DECAY for upstream control}"
    train_args+=("--weight-decay=$INPUT_WEIGHT_DECAY")
fi

if [ "$RUN_KIND" = production ]; then
    : "${PRODUCTION_GATE:?Missing PRODUCTION_GATE for production topology binding}"
    train_args+=("--production-gate=$PRODUCTION_GATE")
fi

if [ -n "${RESUME_FROM_STEP:-}" ]; then
    train_args+=("--resume-from-step=$RESUME_FROM_STEP")
fi
if [ -n "${PARENT_CHECKPOINT_DIR:-}" ]; then
    : "${SOURCE_STEP:?Missing SOURCE_STEP for parent lineage}"
    : "${PARENT_CHECKPOINT_SHA256:?Missing PARENT_CHECKPOINT_SHA256}"
    train_args+=(
        "--parent-checkpoint-dir=$PARENT_CHECKPOINT_DIR"
        "--parent-checkpoint-step=$SOURCE_STEP"
        "--parent-checkpoint-sha256=$PARENT_CHECKPOINT_SHA256"
    )
fi

child_pid=""
forward_signal() {
    local signal_name="$1"
    if [ -n "$child_pid" ]; then
        kill -s "$signal_name" "$child_pid" 2>/dev/null || true
    fi
}
trap 'forward_signal USR1' USR1
trap 'forward_signal TERM' TERM

set +e
.venv/bin/python -m scripts.d32_wsd_train "${train_args[@]}" &
child_pid=$!
while true; do
    wait "$child_pid"
    child_rc=$?
    if kill -0 "$child_pid" 2>/dev/null; then
        continue
    fi
    break
done
set -e

if [ "$child_rc" -eq 0 ] || [ "$child_rc" -eq 75 ]; then
    mkdir -p "$LAUNCH_RECEIPT_DIR"
    .venv/bin/python scripts/d32_family_workflow.py record-rank-exit \
        --recipe="$STUDY_MANIFEST" \
        --run-id="$RUN_ID" \
        --phase="$LAUNCH_PHASE" \
        --slurm-job-id="${SLURM_JOB_ID:-}" \
        --slurm-step-id="${SLURM_STEP_ID:-batch}" \
        --node="$(hostname)" \
        --rank="$RANK" \
        --local-rank="$NODE_LOCAL_RANK" \
        --world-size="$WORLD_SIZE" \
        --exit-code="$child_rc" \
        --launcher="$LAUNCHER_ID" \
        --output="$LAUNCH_RECEIPT_DIR/rank_$(printf '%05d' "$RANK").json"
fi
if [ "$child_rc" -eq 75 ]; then
    # Intentional collective preemption is evidence, not an srun task failure.
    # Returning zero prevents --kill-on-bad-exit=1 from racing peer receipts.
    exit 0
fi
exit "$child_rc"
