#!/bin/bash

# CPU finalizer for the bounded signal/requeue/resume smoke.

#SBATCH --job-name=tr-d32-signal-final
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

source /etc/profile.d/modules.sh
module use /ari/progs/modulefiles
module purge
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0

: "${SIGNAL_JOB_ID:?Set the completed signal-smoke parent job ID}"
CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/ari/users/nunal/nanochat-turk-d32-general}"
RECIPE="${RECIPE:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_wsd_v1.json}"
PREFLIGHT_RECEIPT="${PREFLIGHT_RECEIPT:-$NANOCHAT_BASE_DIR/control/d32/preflight.json}"
ATTENTION_PROBE_RECEIPT="${ATTENTION_PROBE_RECEIPT:-$NANOCHAT_BASE_DIR/control/d32/attention_probe.json}"
WSD_PROXY_APPROVAL="${WSD_PROXY_APPROVAL:-$NANOCHAT_BASE_DIR/control/d32/wd_proxy_approval.json}"
STATIC_LAUNCHER_GATE="${STATIC_LAUNCHER_GATE:-$NANOCHAT_BASE_DIR/control/d32/static_launcher_gate_ws4.json}"
cd "$CODE_DIR"

CONTROL_ROOT="$NANOCHAT_BASE_DIR/control/d32/launcher/job${SIGNAL_JOB_ID}/signal_smoke"
.venv/bin/python scripts/d32_family_workflow.py finalize-signal-smoke \
    --recipe="$RECIPE" --preflight-receipt="$PREFLIGHT_RECEIPT" \
    --static-launcher-gate="$STATIC_LAUNCHER_GATE" \
    --attention-probe="$ATTENTION_PROBE_RECEIPT" \
    --wd-proxy-approval="$WSD_PROXY_APPROVAL" --base-dir="$NANOCHAT_BASE_DIR" \
    --preemption-receipt="$CONTROL_ROOT/signal_preemption.json" \
    --final-launch-receipt="$CONTROL_ROOT/final_resume_launch.json" \
    --slurm-job-id="$SIGNAL_JOB_ID" \
    --output="$NANOCHAT_BASE_DIR/control/d32/signal_resume_gate_ws4.json"
