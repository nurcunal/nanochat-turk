#!/bin/bash

# One-node bounded SIGUSR1 -> checkpoint -> rank75/wrapper0 -> requeue -> resume gate.

#SBATCH --job-name=tr-d32-signal-smoke
#SBATCH --account=nakane
#SBATCH --partition=a100x4q
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:4
#SBATCH --time=01:00:00
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

export CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/ari/users/nunal/nanochat-turk-d32-general-v2}"
export RECIPE="${RECIPE:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_wsd_v2.json}"
export PREFLIGHT_RECEIPT="${PREFLIGHT_RECEIPT:-$NANOCHAT_BASE_DIR/control/d32/preflight.json}"
export ATTENTION_PROBE_RECEIPT="${ATTENTION_PROBE_RECEIPT:-$NANOCHAT_BASE_DIR/control/d32/attention_probe.json}"
export WSD_PROXY_APPROVAL="${WSD_PROXY_APPROVAL:-$NANOCHAT_BASE_DIR/control/d32/wd_proxy_approval.json}"
export STATIC_LAUNCHER_GATE="${STATIC_LAUNCHER_GATE:-$NANOCHAT_BASE_DIR/control/d32/static_launcher_gate_ws4.json}"
cd "$CODE_DIR"

if [ "${SLURM_NNODES:-0}" -ne 1 ] || [ "${SLURM_NTASKS:-0}" -ne 4 ]; then
    echo "Signal/resume smoke requires exactly one 4xA100 node and four tasks" >&2
    exit 2
fi
.venv/bin/python scripts/d32_family_workflow.py verify-static-launcher-gate \
    --recipe="$RECIPE" --preflight-receipt="$PREFLIGHT_RECEIPT" \
    --gate="$STATIC_LAUNCHER_GATE"
eval "$(.venv/bin/python scripts/d32_family_workflow.py optimizer-env \
    --recipe="$RECIPE" --preflight-receipt="$PREFLIGHT_RECEIPT" \
    --attention-probe="$ATTENTION_PROBE_RECEIPT" \
    --wd-proxy-approval="$WSD_PROXY_APPROVAL")"
eval "$(.venv/bin/python scripts/d32_family_workflow.py signal-smoke-env \
    --recipe="$RECIPE" --base-dir="$NANOCHAT_BASE_DIR")"

export FAMILY_ID=tr_d32_general_bpe32k_v2
export RUN_KIND=signal_smoke
export RUN_ID MODEL_TAG RESUME_FROM_STEP
export DEPTH=32 DEVICE_BATCH_SIZE=4 TOTAL_BATCH_SIZE=2097152
export NUM_ITERATIONS=6 STOP_AT_STEP=6 SEED=42 EVAL_EVERY=250
export LR_SCHEDULE=wsd WSD_COOLDOWN_START_STEP=-1
export STUDY_MANIFEST="$RECIPE"
export DATA_DIR="$NANOCHAT_BASE_DIR/pretrain_data/tr_general_clean_v2"
export TOKENIZER_MANIFEST="$NANOCHAT_BASE_DIR/tokenizers/tr_general_raw_bpe_32k_v2/package_manifest.json"
export VALIDATION_MANIFEST="$DATA_DIR/validation_exposure_manifest.json"
export EXPOSURE_PLAN="$DATA_DIR/training_exposure_signal_smoke_ws4_seed42.json"
export METRICS_DIR="$NANOCHAT_BASE_DIR/metrics/d32_signal_smoke/ws4"
export CODE_REVISION EFFECTIVE_BASE_WEIGHT_DECAY WSD_WEIGHT_DECAY_COOLDOWN
export ATTENTION_BACKEND WINDOW_PATTERN ATTENTION_PROBE_RECEIPT
export NNODES=1 NPROC_PER_NODE=4
mapfile -t SLURM_HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR="${SLURM_HOSTS[0]}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 40000))}"

RESTART_COUNT="${SLURM_RESTART_COUNT:-0}"
CONTROL_ROOT="$NANOCHAT_BASE_DIR/control/d32/launcher/job${SLURM_JOB_ID}/signal_smoke"
ATTEMPT_ROOT="$CONTROL_ROOT/attempt${RESTART_COUNT}"
if [ -e "$ATTEMPT_ROOT" ]; then
    echo "Refusing to reuse signal-smoke attempt: $ATTEMPT_ROOT" >&2
    exit 2
fi
mkdir -p "$ATTEMPT_ROOT/ranks" "$METRICS_DIR"
export LAUNCH_RECEIPT_DIR="$ATTEMPT_ROOT/ranks"
export LAUNCHER_ID=slurm_srun_direct_python_env_v1
if [ "$RESTART_COUNT" -eq 0 ]; then
    if [ -n "$RESUME_FROM_STEP" ]; then
        echo "Initial signal-smoke attempt unexpectedly found a resume checkpoint" >&2
        exit 2
    fi
    export LAUNCH_PHASE=signal_smoke_preemption
else
    if [ -z "$RESUME_FROM_STEP" ]; then
        echo "Requeued signal-smoke attempt has no verified resume checkpoint" >&2
        exit 2
    fi
    export LAUNCH_PHASE=signal_smoke_resume
fi

srun --nodes=1 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=16 \
    --gpus-per-task=1 --gpu-bind=single:1 --kill-on-bad-exit=1 \
    bash runs/uhem_d32_train_node.sh &
srun_pid=$!

controller_pid=""
if [ "$RESTART_COUNT" -eq 0 ]; then
    signal_controller() {
        while kill -0 "$srun_pid" 2>/dev/null; do
            if .venv/bin/python -c 'import json,sys; p=sys.argv[1]; n=-1
try:
 f=open(p, encoding="utf-8")
 for line in f:
  x=json.loads(line)
  if x.get("event_type")=="train_update": n=max(n,int(x["updates_completed"]))
except FileNotFoundError: pass
raise SystemExit(0 if n >= 1 else 1)' "$METRICS_DIR/training_curve.jsonl"; then
                .venv/bin/python scripts/d32_family_workflow.py record-signal-request \
                    --recipe="$RECIPE" --curve-log="$METRICS_DIR/training_curve.jsonl" \
                    --slurm-job-id="$SLURM_JOB_ID" --slurm-step-id="${SLURM_JOB_ID}.0" \
                    --slurm-restart-count=0 --output="$CONTROL_ROOT/signal_request.json"
                # Slurm 20.11.8 exposes targeted signalling through scancel;
                # the .0 suffix addresses this allocation's first srun step,
                # not the batch shell. This is deliberately not `--batch` or
                # `--full`, both of which would change the signal target.
                scancel --signal=USR1 "${SLURM_JOB_ID}.0"
                return
            fi
            sleep 5
        done
        return 1
    }
    signal_controller &
    controller_pid=$!
fi

set +e
wait "$srun_pid"
srun_rc=$?
controller_rc=0
if [ -n "$controller_pid" ]; then
    wait "$controller_pid"
    controller_rc=$?
fi
set -e
if [ "$srun_rc" -ne 0 ] || [ "$controller_rc" -ne 0 ]; then
    echo "Bounded signal-smoke launch/controller failed: srun=$srun_rc controller=$controller_rc" >&2
    exit 1
fi

first_rank="$LAUNCH_RECEIPT_DIR/rank_00000.json"
child_exit="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["child_exit_code"])' "$first_rank")"
if [ "$RESTART_COUNT" -eq 0 ]; then
    if [ "$child_exit" -ne 75 ]; then
        echo "Initial signal-smoke attempt must produce trainer exit 75" >&2
        exit 2
    fi
    launch_receipt="$ATTEMPT_ROOT/preemption_launch.json"
    .venv/bin/python scripts/d32_family_workflow.py seal-static-launch \
        --recipe="$RECIPE" --receipt-dir="$LAUNCH_RECEIPT_DIR" --run-id="$RUN_ID" \
        --phase="$LAUNCH_PHASE" --slurm-job-id="$SLURM_JOB_ID" --world-size=4 \
        --nodes=1 --srun-exit-code=0 --expected-exit-code=75 --output="$launch_receipt"
    .venv/bin/python scripts/d32_family_workflow.py seal-signal-preemption \
        --recipe="$RECIPE" --base-dir="$NANOCHAT_BASE_DIR" \
        --signal-request="$CONTROL_ROOT/signal_request.json" \
        --launch-receipt="$launch_receipt" --slurm-job-id="$SLURM_JOB_ID" \
        --output="$CONTROL_ROOT/signal_preemption.json"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi

if [ "$child_exit" -ne 0 ]; then
    echo "Resumed signal-smoke attempt must terminate cleanly" >&2
    exit 2
fi
.venv/bin/python scripts/d32_family_workflow.py seal-static-launch \
    --recipe="$RECIPE" --receipt-dir="$LAUNCH_RECEIPT_DIR" --run-id="$RUN_ID" \
    --phase="$LAUNCH_PHASE" --slurm-job-id="$SLURM_JOB_ID" --world-size=4 \
    --nodes=1 --srun-exit-code=0 --expected-exit-code=0 \
    --output="$CONTROL_ROOT/final_resume_launch.json"
echo "Signal/resume payload complete; dependent CPU finalizer must verify Slurm COMPLETED."
