#!/bin/bash

# One sequential d32 family stage on the topology selected by the sealed gate.
# The default allocation is four nodes; a ws8 fallback submission overrides it
# with `sbatch --nodes=2`. This job never seals lineage itself.

#SBATCH --job-name=tr-d32-production
#SBATCH --account=nakane
#SBATCH --partition=a100x4q
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:4
#SBATCH --time=7-00:00:00
#SBATCH --mem=900G
#SBATCH --signal=USR1@900
#SBATCH --requeue
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -eo pipefail

source /etc/profile.d/modules.sh
module use /ari/progs/modulefiles
module purge
module load cuda/cuda-12.5-a100q
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0
set -u

: "${STAGE:?Set STAGE to one frozen family stage ID}"
export CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/ari/users/nunal/nanochat-turk-d32-general-v3}"
export RECIPE="${RECIPE:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_wsd_v3.json}"
export PREFLIGHT_RECEIPT="${PREFLIGHT_RECEIPT:-$NANOCHAT_BASE_DIR/control/d32/preflight.json}"
export ATTENTION_PROBE_RECEIPT="${ATTENTION_PROBE_RECEIPT:-$NANOCHAT_BASE_DIR/control/d32/attention_probe.json}"
export WSD_PROXY_APPROVAL="${WSD_PROXY_APPROVAL:-$NANOCHAT_BASE_DIR/control/d32/wd_proxy_approval.json}"
export STATIC_LAUNCHER_GATE="${STATIC_LAUNCHER_GATE:-$NANOCHAT_BASE_DIR/control/d32/static_launcher_gate_ws4.json}"
export SIGNAL_RESUME_GATE="${SIGNAL_RESUME_GATE:-$NANOCHAT_BASE_DIR/control/d32/signal_resume_gate_ws4.json}"
export PRODUCTION_GATE="${PRODUCTION_GATE:-$NANOCHAT_BASE_DIR/control/d32/production_topology_gate.json}"
export LINEAGE_DIR="${LINEAGE_DIR:-$NANOCHAT_BASE_DIR/control/d32/lineage}"

cd "$CODE_DIR"
eval "$(.venv/bin/python scripts/d32_family_workflow.py stage-env \
    --recipe="$RECIPE" --base-dir="$NANOCHAT_BASE_DIR" --stage="$STAGE" \
    --gate="$PRODUCTION_GATE")"
export FAMILY_ID STAGE_ID STAGE_KIND MODEL_TAG SOURCE_MODEL_TAG SOURCE_STEP
export TARGET_STEP NUM_ITERATIONS WSD_COOLDOWN_START_STEP EXPOSURE_PLAN_KEY
export RESUME_FROM_STEP PARENT_CHECKPOINT_DIR PARENT_CHECKPOINT_SHA256 RUN_ID
export PRODUCTION_WORLD_SIZE PRODUCTION_NODES

if [ "${SLURM_JOB_PARTITION:-}" != a100x4q ]; then
    echo "Production requires the a100x4q partition" >&2
    exit 2
fi
if [ "${SLURM_NNODES:-0}" -ne "$PRODUCTION_NODES" ] || \
   [ "${SLURM_NTASKS:-0}" -ne "$PRODUCTION_WORLD_SIZE" ]; then
    echo "Allocation does not match the sealed topology: nodes=$PRODUCTION_NODES world=$PRODUCTION_WORLD_SIZE" >&2
    exit 2
fi
.venv/bin/python scripts/d32_family_workflow.py verify-static-launcher-gate \
    --recipe="$RECIPE" --preflight-receipt="$PREFLIGHT_RECEIPT" \
    --gate="$STATIC_LAUNCHER_GATE"
.venv/bin/python scripts/d32_family_workflow.py verify-signal-resume-gate \
    --recipe="$RECIPE" --preflight-receipt="$PREFLIGHT_RECEIPT" \
    --gate="$SIGNAL_RESUME_GATE"
.venv/bin/python scripts/d32_family_workflow.py preflight-stage \
    --recipe="$RECIPE" --preflight-receipt="$PREFLIGHT_RECEIPT" \
    --attention-probe="$ATTENTION_PROBE_RECEIPT" \
    --signal-resume-gate="$SIGNAL_RESUME_GATE" \
    --wd-proxy-approval="$WSD_PROXY_APPROVAL" --gate="$PRODUCTION_GATE" \
    --base-dir="$NANOCHAT_BASE_DIR" --lineage-dir="$LINEAGE_DIR" \
    --stage="$STAGE" --world-size="$PRODUCTION_WORLD_SIZE"
eval "$(.venv/bin/python scripts/d32_family_workflow.py optimizer-env \
    --recipe="$RECIPE" --preflight-receipt="$PREFLIGHT_RECEIPT" \
    --attention-probe="$ATTENTION_PROBE_RECEIPT" \
    --wd-proxy-approval="$WSD_PROXY_APPROVAL")"
export EFFECTIVE_BASE_WEIGHT_DECAY WSD_WEIGHT_DECAY_COOLDOWN CODE_REVISION
export ATTENTION_BACKEND WINDOW_PATTERN ATTENTION_PROBE_RECEIPT

export RUN_KIND=production
export DEPTH=32
export DEVICE_BATCH_SIZE=4
export TOTAL_BATCH_SIZE=2097152
export STOP_AT_STEP="$TARGET_STEP"
export SEED=42
export EVAL_EVERY=250
export LR_SCHEDULE=wsd
export NNODES="$PRODUCTION_NODES"
export NPROC_PER_NODE=4
export STUDY_MANIFEST="$RECIPE"
export DATA_DIR="${DATA_DIR:-$NANOCHAT_BASE_DIR/pretrain_data/tr_general_clean_v3}"
export PACKING_CAPACITY_RECEIPT="$DATA_DIR/packing_capacity_receipt.json"
export TOKENIZER_MANIFEST="${TOKENIZER_MANIFEST:-$NANOCHAT_BASE_DIR/tokenizers/tr_general_raw_bpe_32k_v3/package_manifest.json}"
export VALIDATION_MANIFEST="$DATA_DIR/validation_exposure_manifest.json"
export EXPOSURE_PLAN="$DATA_DIR/training_exposure_${EXPOSURE_PLAN_KEY}.json"
export METRICS_DIR="$NANOCHAT_BASE_DIR/metrics/d32_family/$RUN_ID"
export MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 40000))}"
mapfile -t SLURM_HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR="${MASTER_ADDR:-${SLURM_HOSTS[0]}}"

RESTART_COUNT="${SLURM_RESTART_COUNT:-0}"
CONTROL_ROOT="$NANOCHAT_BASE_DIR/control/d32/launcher/job${SLURM_JOB_ID}"
ATTEMPT_ROOT="$CONTROL_ROOT/attempt${RESTART_COUNT}/${STAGE}"
export LAUNCH_PHASE="production_${STAGE}"
export LAUNCHER_ID=slurm_srun_direct_python_env_v1
export LAUNCH_RECEIPT_DIR="$ATTEMPT_ROOT/ranks"
if [ -e "$ATTEMPT_ROOT" ]; then
    echo "Refusing to reuse a production launch attempt directory: $ATTEMPT_ROOT" >&2
    exit 2
fi
mkdir -p "$LAUNCH_RECEIPT_DIR" "$METRICS_DIR" "$LINEAGE_DIR"

set +e
srun --nodes="$PRODUCTION_NODES" --ntasks="$PRODUCTION_WORLD_SIZE" \
    --ntasks-per-node=4 --cpus-per-task=16 --gpus-per-task=1 \
    --gpu-bind=single:1 --kill-on-bad-exit=1 \
    bash runs/uhem_d32_train_node.sh
srun_rc=$?
set -e
if [ "$srun_rc" -ne 0 ]; then
    echo "Static srun launch failed before a unanimous rank result (rc=$srun_rc)" >&2
    exit "$srun_rc"
fi

first_rank_receipt="$LAUNCH_RECEIPT_DIR/rank_00000.json"
if [ ! -f "$first_rank_receipt" ]; then
    echo "Static launch returned zero without rank 0 exit evidence" >&2
    exit 2
fi
child_exit="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["child_exit_code"])' "$first_rank_receipt")"
if [ "$child_exit" -ne 0 ] && [ "$child_exit" -ne 75 ]; then
    echo "Unexpected direct trainer exit classification: $child_exit" >&2
    exit 2
fi

if [ "$child_exit" -eq 75 ]; then
    launch_receipt="$ATTEMPT_ROOT/collective_preemption_launch.json"
else
    launch_receipt="$CONTROL_ROOT/final_${STAGE}_launch.json"
    if [ -e "$launch_receipt" ]; then
        echo "Refusing to overwrite a prior final launch receipt: $launch_receipt" >&2
        exit 2
    fi
fi
.venv/bin/python scripts/d32_family_workflow.py seal-static-launch \
    --recipe="$RECIPE" --receipt-dir="$LAUNCH_RECEIPT_DIR" \
    --run-id="$RUN_ID" --phase="$LAUNCH_PHASE" --slurm-job-id="$SLURM_JOB_ID" \
    --world-size="$PRODUCTION_WORLD_SIZE" --nodes="$PRODUCTION_NODES" \
    --srun-exit-code=0 --expected-exit-code="$child_exit" --output="$launch_receipt"

if [ "$child_exit" -eq 75 ]; then
    preemption_receipt="$ATTEMPT_ROOT/collective_preemption.json"
    .venv/bin/python scripts/d32_family_workflow.py seal-preemption \
        --recipe="$RECIPE" --preflight-receipt="$PREFLIGHT_RECEIPT" \
        --attention-probe="$ATTENTION_PROBE_RECEIPT" \
        --signal-resume-gate="$SIGNAL_RESUME_GATE" \
        --wd-proxy-approval="$WSD_PROXY_APPROVAL" --gate="$PRODUCTION_GATE" \
        --base-dir="$NANOCHAT_BASE_DIR" --stage="$STAGE" \
        --launch-receipt="$launch_receipt" --slurm-job-id="$SLURM_JOB_ID" \
        --slurm-restart-count="$RESTART_COUNT" --output="$preemption_receipt"
    echo "Verified collective preemption and complete resume checkpoint; requeueing $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi

echo "Stage payload completed. A dependent CPU finalizer must verify Slurm COMPLETED."
