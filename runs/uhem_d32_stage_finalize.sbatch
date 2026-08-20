#!/bin/bash

# CPU-only finalizer for one completed production stage.

#SBATCH --job-name=tr-d32-stage-final
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -eo pipefail

source /etc/profile.d/modules.sh
module use /ari/progs/modulefiles
module purge
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0
set -u

: "${STAGE:?Set the family stage ID}"
: "${STAGE_JOB_ID:?Set the completed parent GPU job ID}"
CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/ari/users/nunal/nanochat-turk-d32-general-v3}"
RECIPE="${RECIPE:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_wsd_v3.json}"
PREFLIGHT_RECEIPT="${PREFLIGHT_RECEIPT:-$NANOCHAT_BASE_DIR/control/d32/preflight.json}"
ATTENTION_PROBE_RECEIPT="${ATTENTION_PROBE_RECEIPT:-$NANOCHAT_BASE_DIR/control/d32/attention_probe.json}"
WSD_PROXY_APPROVAL="${WSD_PROXY_APPROVAL:-$NANOCHAT_BASE_DIR/control/d32/wd_proxy_approval.json}"
SIGNAL_RESUME_GATE="${SIGNAL_RESUME_GATE:-$NANOCHAT_BASE_DIR/control/d32/signal_resume_gate_ws4.json}"
PRODUCTION_GATE="${PRODUCTION_GATE:-$NANOCHAT_BASE_DIR/control/d32/production_topology_gate.json}"
LINEAGE_DIR="${LINEAGE_DIR:-$NANOCHAT_BASE_DIR/control/d32/lineage}"
cd "$CODE_DIR"

.venv/bin/python scripts/d32_family_workflow.py seal-stage \
    --recipe="$RECIPE" --preflight-receipt="$PREFLIGHT_RECEIPT" \
    --attention-probe="$ATTENTION_PROBE_RECEIPT" \
    --signal-resume-gate="$SIGNAL_RESUME_GATE" \
    --wd-proxy-approval="$WSD_PROXY_APPROVAL" --gate="$PRODUCTION_GATE" \
    --base-dir="$NANOCHAT_BASE_DIR" --stage="$STAGE" \
    --launch-receipt="$NANOCHAT_BASE_DIR/control/d32/launcher/job${STAGE_JOB_ID}/final_${STAGE}_launch.json" \
    --slurm-job-id="$STAGE_JOB_ID" --output="$LINEAGE_DIR/${STAGE}.json"
