#!/bin/bash

# CPU-only finalizer. Must run afterok on the completed GPU smoke job.

#SBATCH --job-name=tr-d32-smoke-finalize
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -eo pipefail

source /etc/profile.d/modules.sh
module use /ari/progs/modulefiles
module purge
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0
set -u

: "${SMOKE_JOB_ID:?Set the completed parent GPU smoke job ID}"
: "${SMOKE_NODES:?Set SMOKE_NODES to 2 or 4}"
CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/ari/users/nunal/nanochat-turk-d32-general-v3}"
RECIPE="${RECIPE:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_wsd_v4.json}"
PREFLIGHT_RECEIPT="${PREFLIGHT_RECEIPT:-$NANOCHAT_BASE_DIR/control/d32_v4/preflight.json}"
ATTENTION_PROBE_RECEIPT="${ATTENTION_PROBE_RECEIPT:-$NANOCHAT_BASE_DIR/control/d32_v4/attention_probe.json}"
WSD_PROXY_APPROVAL="${WSD_PROXY_APPROVAL:-$NANOCHAT_BASE_DIR/control/d32_v4/wd_proxy_approval.json}"
STATIC_LAUNCHER_GATE="${STATIC_LAUNCHER_GATE:-$NANOCHAT_BASE_DIR/control/d32_v4/static_launcher_gate_ws4.json}"
SIGNAL_RESUME_GATE="${SIGNAL_RESUME_GATE:-$NANOCHAT_BASE_DIR/control/d32_v4/signal_resume_gate_ws4.json}"
cd "$CODE_DIR"

if [ "$SMOKE_NODES" -ne 2 ] && [ "$SMOKE_NODES" -ne 4 ]; then
    echo "SMOKE_NODES must equal 2 or 4" >&2
    exit 2
fi
WORLD_SIZE=$((SMOKE_NODES * 4))
RECIPE_FAMILY_ID="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["family_id"])' "$RECIPE")"
if [ -n "${FAMILY_ID:-}" ] && [ "$FAMILY_ID" != "$RECIPE_FAMILY_ID" ]; then
    echo "FAMILY_ID differs from the validated recipe family_id: $RECIPE_FAMILY_ID" >&2
    exit 2
fi
FAMILY_ID="$RECIPE_FAMILY_ID"
RUN_ID="${FAMILY_ID}_smoke_ws${WORLD_SIZE}"
CONTROL_ROOT="$NANOCHAT_BASE_DIR/control/d32_v4/launcher/job${SMOKE_JOB_ID}"

.venv/bin/python scripts/d32_family_workflow.py seal-smoke \
    --recipe="$RECIPE" \
    --preflight-receipt="$PREFLIGHT_RECEIPT" \
    --attention-probe="$ATTENTION_PROBE_RECEIPT" \
    --wd-proxy-approval="$WSD_PROXY_APPROVAL" \
    --nodes="$SMOKE_NODES" \
    --curve-log="$NANOCHAT_BASE_DIR/metrics/d32_v4/smoke/ws${WORLD_SIZE}/training_curve.jsonl" \
    --checkpoint-root="$NANOCHAT_BASE_DIR/base_checkpoints/$RUN_ID" \
    --static-launcher-gate="$STATIC_LAUNCHER_GATE" \
    --signal-resume-gate="$SIGNAL_RESUME_GATE" \
    --static-probe-receipt="$CONTROL_ROOT/static_nccl_probe.json" \
    --launch-receipt="$CONTROL_ROOT/smoke_initial_50/launch.json" \
    --launch-receipt="$CONTROL_ROOT/smoke_resume_100/launch.json" \
    --slurm-job-id="$SMOKE_JOB_ID" \
    --output="$NANOCHAT_BASE_DIR/control/d32_v4/smoke_ws${WORLD_SIZE}.json"
