#!/bin/bash

# Verify and optionally upload the complete d32 family. Defaults are private
# and dry-run; changing remote state requires DRY_RUN=0 explicitly.

#SBATCH --job-name=tr-d32-hf-export
#SBATCH --account=nakane
#SBATCH --partition=cpu2dq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -eo pipefail

source /etc/profile.d/modules.sh
module use /ari/progs/modulefiles
module purge
module load Python/Python-3.12.4-openmpi-5.0.3-gcc-11.4.0
set -u

: "${HF_REPO_ID:?Set the destination Hugging Face model repository}"
: "${FINAL_OPTIMIZER_POLICY:?Set FINAL_OPTIMIZER_POLICY to include or omit}"
: "${CLUSTER_LAUNCH_RECEIPT:?Set the exact production cluster launch receipt}"
if [ "$FINAL_OPTIMIZER_POLICY" != include ] && [ "$FINAL_OPTIMIZER_POLICY" != omit ]; then
    echo "FINAL_OPTIMIZER_POLICY must equal include or omit" >&2
    exit 2
fi

CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/ari/users/nunal/nanochat-turk-d32-general-v3}"
RECIPE="${RECIPE:-$CODE_DIR/configs/pretrain/tr_d32_turkish_general_wsd_v3.json}"
CONTROL_DIR="${CONTROL_DIR:-$NANOCHAT_BASE_DIR/control/d32}"
DRY_RUN="${DRY_RUN:-1}"
if [ "$DRY_RUN" != 0 ] && [ "$DRY_RUN" != 1 ]; then
    echo "DRY_RUN must equal 0 or 1" >&2
    exit 2
fi

cd "$CODE_DIR"
args=(
    --repo-id="$HF_REPO_ID"
    --private
    --base-dir="$NANOCHAT_BASE_DIR"
    --family-recipe="$RECIPE"
    --preflight-receipt="$CONTROL_DIR/preflight.json"
    --attention-probe="$CONTROL_DIR/attention_probe.json"
    --wd-proxy-approval="$CONTROL_DIR/wd_proxy_approval.json"
    --static-launcher-gate="$CONTROL_DIR/static_launcher_gate_ws4.json"
    --signal-resume-gate="$CONTROL_DIR/signal_resume_gate_ws4.json"
    --production-gate="$CONTROL_DIR/production_topology_gate.json"
    --cluster-launch-receipt="$CLUSTER_LAUNCH_RECEIPT"
    --lineage-dir="$CONTROL_DIR/lineage"
    --family-final-optimizer-policy="$FINAL_OPTIMIZER_POLICY"
)
if [ "$DRY_RUN" -eq 1 ]; then
    args+=(--dry-run)
fi
.venv/bin/python -m scripts.upload_base_checkpoint_to_hf "${args[@]}"
